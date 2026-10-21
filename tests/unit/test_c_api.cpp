// mossinfer.h: CApi.* run without a GPU (versions, struct sizes, defaults, argument checks, error reporting, the
// probe); GpuCApi.* generate through the C API with the base bundle (skipped when the weights or a device are absent).
#include "mossinfer/mossinfer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <string>
#include <thread>
#include <vector>

#include "api/pipeline_adapter.h"
#include "cuda_context_check.h"
#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "test_util.h"
#include "util/json.h"

namespace {

namespace fs = std::filesystem;

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}

std::string last_error() { return mi_last_error_message(); }

mi_gen_params gen_defaults() {
  mi_gen_params p;
  EXPECT_EQ(mi_gen_params_init(&p, sizeof p), MI_OK);
  return p;
}

mi_engine_params engine_defaults() {
  mi_engine_params p;
  EXPECT_EQ(mi_engine_params_init(&p, sizeof p), MI_OK);
  return p;
}

// A directory with the four component folders of a bundle (no weights): passes the path checks done before CUDA.
void make_fake_bundle(const fs::path& dir) {
  for (const char* c : {"text_encoder", "tokenizer", "transformer", "vae"}) fs::create_directories(dir / c);
}

// ---------------------------------------------------------------- CPU

TEST(CApi, VersionAndAbi) {
  const std::string expected = std::to_string(MI_VERSION_MAJOR) + "." + std::to_string(MI_VERSION_MINOR) + "." +
                               std::to_string(MI_VERSION_PATCH);
  EXPECT_EQ(std::string(mi_version()), expected);
  EXPECT_EQ(mi_abi_version(), MI_ABI_VERSION);
  EXPECT_EQ(mi_capabilities() & MI_CAP_PREVIEW, 0u);
  EXPECT_EQ(mi_capabilities() & MI_CAP_COMFYUI_MODELS, 0u);
  EXPECT_STREQ(mi_status_string(MI_OK), "MI_OK");
  EXPECT_STREQ(mi_status_string(MI_CANCELLED), "MI_CANCELLED");
  EXPECT_STREQ(mi_status_string(MI_ERR_BUSY), "MI_ERR_BUSY");
  EXPECT_STREQ(mi_status_string(MI_ERR_INTERNAL), "MI_ERR_INTERNAL");
  EXPECT_STREQ(mi_status_string(static_cast<mi_status>(1234)), "MI_UNKNOWN_STATUS");
}

// The C defaults mirror pipeline::Request / PipelineOptions (themselves the app's parameters.json defaults).
TEST(CApi, DefaultsMirrorThePipeline) {
  const mi_gen_params g = gen_defaults();
  const mi::pipeline::Request q;
  EXPECT_EQ(g.struct_size, sizeof(mi_gen_params));
  EXPECT_EQ(g.prompt, nullptr);
  EXPECT_EQ(g.negative_prompt, nullptr);
  EXPECT_EQ(g.width, q.width);
  EXPECT_EQ(g.height, q.height);
  EXPECT_EQ(g.steps, q.steps);
  EXPECT_EQ(g.cfg, q.cfg);
  EXPECT_EQ(g.shift, q.shift);
  EXPECT_EQ(g.seed, q.seed);
  ASSERT_EQ(g.uncond_skip_layers_count, q.uncond_skip_layers.size());
  for (std::size_t i = 0; i < q.uncond_skip_layers.size(); ++i) EXPECT_EQ(g.uncond_skip_layers[i], q.uncond_skip_layers[i]);
  EXPECT_EQ(g.slg_layers_count, q.slg_layers.size());
  EXPECT_EQ(g.slg_scale, q.slg_scale);
  EXPECT_EQ(g.slg_start, q.slg_start);
  EXPECT_EQ(g.slg_stop, q.slg_stop);
  EXPECT_EQ(g.max_sequence_length, q.max_sequence_length);
  EXPECT_EQ(g.sampler, MI_SAMPLER_EULER);
  EXPECT_EQ(g.progress_callback, nullptr);
  EXPECT_EQ(g.plan_callback, nullptr);
  EXPECT_EQ(g.preview_callback, nullptr);
  EXPECT_EQ(g.preview_interval, 1);
  EXPECT_EQ(g.user_data, nullptr);

  const mi_engine_params e = engine_defaults();
  const mi::pipeline::PipelineOptions o;
  EXPECT_EQ(e.struct_size, sizeof(mi_engine_params));
  EXPECT_EQ(e.model_path, nullptr);
  EXPECT_EQ(e.model_format, MI_MODEL_FORMAT_AUTO);
  EXPECT_EQ(e.mode, MI_MODE_BALANCED);
  EXPECT_EQ(e.device_index, 0);
  EXPECT_EQ(e.vram_budget_bytes, o.vram_budget);
  EXPECT_EQ(e.vram_reserve_bytes, o.vram_reserve);
  EXPECT_EQ(e.host_ram_budget_bytes, o.host_ram_budget);
  EXPECT_EQ(e.cache_dir, nullptr);
  EXPECT_EQ(e.embedding_cache, o.embedding_cache ? 1 : 0);
  EXPECT_EQ(e.direct_io, o.direct_io ? 1 : 0);
  EXPECT_EQ(e.block_path, MI_BLOCK_PATH_DEFAULT);  // PipelineOptions' default, fused
  EXPECT_EQ(o.block_path, mi::moss_dit::BlockPath::kFused);
  EXPECT_EQ(e._reserved1, 0);

  mi_gen_params valid = g;
  valid.prompt = "a cat";
  EXPECT_EQ(mi_gen_params_validate(&valid), MI_OK) << last_error();
}

TEST(CApi, StructSizeChecks) {
  for (const std::size_t bad : {std::size_t{0}, sizeof(mi_gen_params) - 1, sizeof(mi_gen_params) + 8}) {
    mi_gen_params g{};
    EXPECT_EQ(mi_gen_params_init(&g, bad), MI_ERR_ABI_MISMATCH) << bad;
    EXPECT_NE(last_error().find("mi_gen_params.struct_size"), std::string::npos) << last_error();
  }
  for (const std::size_t bad : {std::size_t{0}, sizeof(mi_engine_params) - 1, sizeof(mi_engine_params) + 8}) {
    mi_engine_params e{};
    EXPECT_EQ(mi_engine_params_init(&e, bad), MI_ERR_ABI_MISMATCH) << bad;
  }
  // A struct filled for another size is rejected by every function that reads it.
  mi_gen_params g = gen_defaults();
  g.prompt = "a cat";
  g.struct_size = sizeof g - 8;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_ABI_MISMATCH);
  mi_engine_params e = engine_defaults();
  e.model_path = "/nonexistent";
  e.struct_size = sizeof e + 16;
  mi_engine* engine = reinterpret_cast<mi_engine*>(&e);  // must be overwritten with NULL
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_ABI_MISMATCH);
  EXPECT_EQ(engine, nullptr);
  mi_probe_info probe{};
  probe.struct_size = 4;
  EXPECT_EQ(mi_probe(&probe), MI_ERR_ABI_MISMATCH);
  mi_device_info dev{};
  EXPECT_EQ(mi_probe_device(0, &dev), MI_ERR_ABI_MISMATCH);
}

TEST(CApi, NullHandlesAndPointers) {
  EXPECT_EQ(mi_gen_params_init(nullptr, sizeof(mi_gen_params)), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("params is NULL"), std::string::npos) << last_error();
  EXPECT_EQ(mi_engine_params_init(nullptr, sizeof(mi_engine_params)), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_gen_params_validate(nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_probe(nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_probe_device(0, nullptr), MI_ERR_INVALID_ARGUMENT);

  mi_engine* engine = nullptr;
  EXPECT_EQ(mi_engine_create(nullptr, &engine), MI_ERR_INVALID_ARGUMENT);
  const mi_engine_params e = engine_defaults();
  EXPECT_EQ(mi_engine_create(&e, nullptr), MI_ERR_INVALID_ARGUMENT);
  mi_engine_free(nullptr);
  EXPECT_EQ(mi_cancel(nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("engine is NULL"), std::string::npos) << last_error();

  mi_gen_params g = gen_defaults();
  g.prompt = "a cat";
  mi_result* result = reinterpret_cast<mi_result*>(&g);  // must be overwritten with NULL
  EXPECT_EQ(mi_generate(nullptr, &g, &result), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(result, nullptr);

  mi_result_free(nullptr);
  EXPECT_EQ(mi_result_width(nullptr), 0);
  EXPECT_EQ(mi_result_height(nullptr), 0);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  std::size_t bytes = 123;
  EXPECT_EQ(mi_result_rgb(nullptr, &bytes), nullptr);
  EXPECT_EQ(bytes, 0u);
  EXPECT_EQ(mi_result_cancelled(nullptr), 0);
  EXPECT_EQ(mi_result_seed(nullptr), 0u);
  EXPECT_EQ(mi_result_steps_run(nullptr), 0);
  EXPECT_EQ(mi_result_steps_scheduled(nullptr), 0);
  EXPECT_EQ(mi_result_plan_explanation(nullptr), nullptr);
  EXPECT_EQ(mi_result_report_json(nullptr), nullptr);
  mi_timings t{};
  t.struct_size = sizeof t;
  EXPECT_EQ(mi_result_timings(nullptr, &t), MI_ERR_INVALID_ARGUMENT);
  mi_memory_stats m{};
  m.struct_size = sizeof m;
  EXPECT_EQ(mi_result_memory(nullptr, &m), MI_ERR_INVALID_ARGUMENT);
  mi_dit_stream_stats ds{};
  ds.struct_size = sizeof ds;
  EXPECT_EQ(mi_result_dit_stream(nullptr, &ds), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  EXPECT_EQ(mi_result_block_path(nullptr), MI_BLOCK_PATH_DEFAULT);
  EXPECT_EQ(mi_result_block_path_note(nullptr), nullptr);
  EXPECT_EQ(mi_save_png(nullptr, "x.png"), MI_ERR_INVALID_ARGUMENT);
}

// Invalid requests fail with the pipeline's (the official app's) wording, without an engine.
TEST(CApi, InvalidGenParams) {
  struct Case {
    const char* what;
    void (*edit)(mi_gen_params&);
    const char* message;
  };
  static const std::int32_t kOutOfRange[] = {42};
  static const std::int32_t kDuplicate[] = {3, 3};
  const Case cases[] = {
      {"no prompt", [](mi_gen_params& p) { p.prompt = nullptr; }, "Enter a prompt."},
      {"blank prompt", [](mi_gen_params& p) { p.prompt = " \t\n"; }, "Enter a prompt."},
      {"width not /16", [](mi_gen_params& p) { p.width = 1000; }, "multiples of 16"},
      {"negative height", [](mi_gen_params& p) { p.height = -16; }, "multiples of 16"},
      {"too many pixels", [](mi_gen_params& p) { p.width = p.height = 4096; }, "megapixels"},
      {"zero steps", [](mi_gen_params& p) { p.steps = 0; }, "positive"},
      {"negative cfg", [](mi_gen_params& p) { p.cfg = -1; }, "CFG"},
      {"zero shift", [](mi_gen_params& p) { p.shift = 0; }, "shift"},
      {"seed -2", [](mi_gen_params& p) { p.seed = -2; }, "Seed"},
      {"seed 2^53", [](mi_gen_params& p) { p.seed = std::int64_t{1} << 53; }, "Seed"},
      {"max_sequence_length 0", [](mi_gen_params& p) { p.max_sequence_length = 0; }, "text length"},
      {"slg window", [](mi_gen_params& p) { p.slg_start = 0.8, p.slg_stop = 0.2; }, "SLG start"},
      {"NULL layer array", [](mi_gen_params& p) { p.slg_layers = nullptr, p.slg_layers_count = 2; }, "NULL array"},
      {"layer out of range", [](mi_gen_params& p) { p.uncond_skip_layers = kOutOfRange, p.uncond_skip_layers_count = 1; },
       "block indices"},
      {"duplicate layers", [](mi_gen_params& p) { p.slg_layers = kDuplicate, p.slg_layers_count = 2; }, "unique"},
      {"unknown sampler", [](mi_gen_params& p) { p.sampler = static_cast<mi_sampler>(7); }, "sampler"},
      {"preview interval", [](mi_gen_params& p) { p.preview_interval = 0; }, "preview_interval"},
  };
  for (const Case& c : cases) {
    mi_gen_params p = gen_defaults();
    p.prompt = "a cat";
    c.edit(p);
    EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
  }
  // Optional fields: NULL negative prompt, no skip layers at all, AB2.
  mi_gen_params p = gen_defaults();
  p.prompt = "a cat";
  p.uncond_skip_layers = nullptr;
  p.uncond_skip_layers_count = 0;
  p.sampler = MI_SAMPLER_AB2;
  p.seed = -1;
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
}

// Engine parameters are checked before CUDA is touched, so these run on any machine.
TEST(CApi, InvalidEngineParams) {
  mi::testing::TempDir dir("c_api_engine");
  const fs::path bundle = dir.path() / "bundle";
  make_fake_bundle(bundle);
  const std::string bundle_s = mi::util::path_utf8(bundle);
  const fs::path partial = dir.path() / "partial";
  fs::create_directories(partial / "transformer");
  const std::string partial_s = mi::util::path_utf8(partial);
  const fs::path file = dir.path() / "model.safetensors";
  std::ofstream(file) << "x";
  const std::string file_s = mi::util::path_utf8(file);
  const std::string missing_s = mi::util::path_utf8(dir.path() / "missing");

  struct Case {
    const char* what;
    mi_engine_params params;
    mi_status status;
    const char* message;
  };
  const auto with = [&](auto edit) {
    mi_engine_params p = engine_defaults();
    p.model_path = bundle_s.c_str();
    edit(p);
    return p;
  };
  const Case cases[] = {
      {"no model", with([](mi_engine_params& p) { p.model_path = nullptr; }), MI_ERR_INVALID_ARGUMENT, "model_path"},
      {"empty model", with([](mi_engine_params& p) { p.model_path = ""; }), MI_ERR_INVALID_ARGUMENT, "model_path"},
      {"missing dir", with([&](mi_engine_params& p) { p.model_path = missing_s.c_str(); }), MI_ERR_INVALID_ARGUMENT,
       "not found"},
      {"incomplete bundle", with([&](mi_engine_params& p) { p.model_path = partial_s.c_str(); }), MI_ERR_MODEL,
       "text_encoder"},
      {"single file", with([&](mi_engine_params& p) { p.model_path = file_s.c_str(); }), MI_ERR_NOT_SUPPORTED, "ComfyUI"},
      {"comfyui format", with([](mi_engine_params& p) { p.model_format = MI_MODEL_FORMAT_COMFYUI; }), MI_ERR_NOT_SUPPORTED,
       "ComfyUI"},
      {"unknown format", with([](mi_engine_params& p) { p.model_format = static_cast<mi_model_format>(9); }),
       MI_ERR_INVALID_ARGUMENT, "model_format"},
      {"unknown mode", with([](mi_engine_params& p) { p.mode = static_cast<mi_mode>(5); }), MI_ERR_INVALID_ARGUMENT, "mode"},
      {"unknown block path", with([](mi_engine_params& p) { p.block_path = static_cast<mi_block_path>(4); }),
       MI_ERR_INVALID_ARGUMENT, "block_path: unknown value 4"},
      {"block path max enum", with([](mi_engine_params& p) { p.block_path = MI_BLOCK_PATH_MAX_ENUM; }),
       MI_ERR_INVALID_ARGUMENT, "block_path"},
      {"tiny vram budget", with([](mi_engine_params& p) { p.vram_budget_bytes = 1000; }), MI_ERR_INVALID_ARGUMENT,
       "vram_budget_bytes"},
      {"huge host budget", with([](mi_engine_params& p) { p.host_ram_budget_bytes = std::uint64_t{1} << 50; }),
       MI_ERR_INVALID_ARGUMENT, "host_ram_budget_bytes"},
  };
  for (const Case& c : cases) {
    mi_engine* engine = nullptr;
    EXPECT_EQ(mi_engine_create(&c.params, &engine), c.status) << c.what << ": " << last_error();
    EXPECT_EQ(engine, nullptr) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
  }
}

// The explicit padding (ABI 1 layout) is zeroed by the init functions and must stay zero in input structs.
TEST(CApi, ReservedFieldsMustBeZero) {
  mi_gen_params g = gen_defaults();
  EXPECT_EQ(g._reserved0, 0);
  EXPECT_EQ(g._reserved1, 0);
  g.prompt = "a cat";
  g._reserved1 = 7;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("mi_gen_params._reserved1"), std::string::npos) << last_error();
  g._reserved1 = 0;
  g._reserved0 = -1;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("mi_gen_params._reserved0"), std::string::npos) << last_error();

  mi::testing::TempDir dir("c_api_reserved");
  make_fake_bundle(dir.path() / "bundle");
  const std::string bundle = mi::util::path_utf8(dir.path() / "bundle");
  mi_engine_params e = engine_defaults();
  EXPECT_EQ(e._reserved0, 0);
  e.model_path = bundle.c_str();
  e._reserved0 = 1;
  mi_engine* engine = nullptr;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);  // before CUDA is touched
  EXPECT_EQ(engine, nullptr);
  EXPECT_NE(last_error().find("mi_engine_params._reserved0"), std::string::npos) << last_error();
  e._reserved0 = 0;
  e._reserved1 = 2;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(engine, nullptr);
  EXPECT_NE(last_error().find("mi_engine_params._reserved1"), std::string::npos) << last_error();

  // Output structs come back with their padding zeroed, whatever the caller left in it.
  mi_probe_info info;
  std::memset(&info, 0xab, sizeof info);
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_probe(&info), MI_OK) << last_error();
  EXPECT_EQ(info._reserved0, 0);
}

// mi_engine_free never fails: it leaves the thread's last error alone.
TEST(CApi, EngineFreeLeavesLastErrorAlone) {
  EXPECT_EQ(mi_gen_params_validate(nullptr), MI_ERR_INVALID_ARGUMENT);
  const std::string before = last_error();
  mi_engine_free(nullptr);
  EXPECT_EQ(last_error(), before);
}

// The report's numbers never follow the global locale: a host application may call setlocale() or
// std::locale::global(). A synthetic pipeline result goes through the adapter (the shared renderer,
// pipeline/report_json.h, in braces) under a comma-decimal C++ locale with digit grouping (always available), and
// under a comma-decimal C locale where the system has one.
TEST(CApiReport, NumbersAreLocaleIndependent) {
  struct CommaNumpunct : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
  };
  mi::pipeline::Request q;
  q.model = "/models/MOSS-Image/base";
  q.prompt = "a cat, 1.5 m tall";
  q.width = 1024;
  q.cfg = 3.5;
  q.slg_stop = 0.75;
  mi::pipeline::Result r;
  r.seed = 1234567;
  r.width = r.height = 1024;
  r.steps_scheduled = r.steps_run = 2;
  r.timings.total_s = 12.5;
  r.timings.denoise_s = std::numeric_limits<double>::quiet_NaN();
  r.timings.decode_s = std::numeric_limits<double>::infinity();
  r.timings.step_s = {0.25, 1e-7};
  r.memory.arena_capacity = std::size_t{3} << 29;  // 1.5 GiB
  r.block_path = mi::moss_dit::BlockPath::kFused;
  r.block_path_note = "CUDA graphs need every DiT block resident: blocks ran fused, eagerly";
  r.dit_stream.streamed = true;
  r.dit_stream.resident_blocks = 20;
  r.dit_stream.bytes = 12345678901ull;
  r.dit_stream.copy_s = 0.5;
  r.dit_stream.steps = {{24.5, 0.75, 0.0625, 2}};

  const std::locale old_cpp = std::locale::global(std::locale(std::locale::classic(), new CommaNumpunct));
  const std::string old_c = std::setlocale(LC_NUMERIC, nullptr);
  std::string c_locale = "C";
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "ru_RU.UTF-8", "German_Germany.1252"}) {
    if (std::setlocale(LC_NUMERIC, name) != nullptr) {
      c_locale = name;
      break;
    }
  }
  mi::api::Generation g;
  mi::api::render_report(q, r, g);
  const std::string json = mi::api::report_json(g, "out.png");
  const std::string one_and_a_half = mi::pipeline::json_number(1.5);
  std::setlocale(LC_NUMERIC, old_c.c_str());
  std::locale::global(old_cpp);
  std::printf("[c_api] report rendered under C locale %s and a comma-decimal C++ locale\n", c_locale.c_str());

  EXPECT_EQ(one_and_a_half, "1.5");
  mi::json::Value doc;
  ASSERT_NO_THROW(doc = mi::json::parse(json)) << json;
  EXPECT_EQ(doc.at("request").at("width").as_int(), 1024);
  EXPECT_EQ(doc.at("request").at("cfg").as_double(), 3.5);
  EXPECT_EQ(doc.at("request").at("slg_stop").as_double(), 0.75);
  EXPECT_EQ(doc.at("request").at("prompt").as_string(), q.prompt);
  EXPECT_EQ(doc.at("seed_used").as_int(), 1234567);
  EXPECT_EQ(doc.at("timings_s").at("total").as_double(), 12.5);
  EXPECT_TRUE(doc.at("timings_s").at("denoise").is_null());
  EXPECT_TRUE(doc.at("timings_s").at("decode").is_null());
  EXPECT_EQ(doc.at("timings_s").at("steps").as_array().at(0).as_double(), 0.25);
  EXPECT_EQ(doc.at("timings_s").at("steps").as_array().at(1).as_double(), 1e-7);
  EXPECT_EQ(doc.at("memory_gib").at("arena_capacity").as_double(), 1.5);
  EXPECT_EQ(doc.at("image").as_string(), "out.png");
  EXPECT_EQ(doc.at("latents").as_string(), "");
  EXPECT_EQ(doc.at("dit_blocks").at("path").as_string(), "fused");
  EXPECT_EQ(doc.at("dit_blocks").at("note").as_string(), r.block_path_note);
  EXPECT_TRUE(doc.at("dit_stream").at("streamed").as_bool());
  EXPECT_EQ(doc.at("dit_stream").at("bytes").as_int(), 12345678901);
  EXPECT_EQ(doc.at("dit_stream").at("h2d_gbps").as_double(), 24.6914);  // 12345678901 / 0.5 / 1e9, %.6g
  EXPECT_EQ(doc.at("dit_stream").at("steps").as_array().at(0).at("h2d_gbps").as_double(), 24.5);
  // One document: the CLI's report fields in braces.
  EXPECT_EQ(json, "{\n" + mi::pipeline::report_fields(q, r, "out.png", "") + "\n}\n");
}

TEST(CApi, LastErrorIsThreadLocal) {
  EXPECT_EQ(mi_gen_params_validate(nullptr), MI_ERR_INVALID_ARGUMENT);
  const std::string here = last_error();
  ASSERT_FALSE(here.empty());
  std::string there = "unset";
  std::thread([&] {
    there = mi_last_error_message();
    mi_gen_params p{};
    (void)mi_gen_params_init(&p, 1);  // fails on this thread only
  }).join();
  EXPECT_EQ(there, "");
  EXPECT_EQ(last_error(), here);
  // Successful calls leave the message alone.
  mi_gen_params p = gen_defaults();
  EXPECT_EQ(last_error(), here);
  (void)p;
}

// The probe answers on machines without a driver or device (then with device_count == 0).
TEST(CApi, ProbeWithoutRequiringAGpu) {
  mi_probe_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_probe(&info), MI_OK) << last_error();
  EXPECT_EQ(info.struct_size, sizeof info);
  EXPECT_EQ(info.runtime_cuda_version, CUDART_VERSION);
  EXPECT_GE(info.device_count, 0);
  EXPECT_LE(info.supported_device_count, info.device_count);
  EXPECT_GT(std::strlen(info.message), 0u);
  const std::string flavour = info.recommended_flavour;
  EXPECT_TRUE(flavour == "cu13" || flavour == "cu12" || flavour.empty()) << flavour;
  if (!flavour.empty()) {
    EXPECT_GE(info.driver_cuda_version, 12000);
  }
  if (info.driver_cuda_version < 12000) {
    EXPECT_EQ(flavour, "");
  }
  EXPECT_EQ(std::string(info.build_flavour), CUDART_VERSION >= 13000 ? "cu13" : "cu12");
  std::printf("[c_api] probe: %s (flavour '%s')\n", info.message, info.recommended_flavour);

  mi_device_info dev{};
  dev.struct_size = sizeof dev;
  EXPECT_EQ(mi_probe_device(-1, &dev), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_probe_device(info.device_count, &dev), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("out of range"), std::string::npos) << last_error();
}

// ---------------------------------------------------------------- GPU

// mi_probe_device measures free memory without leaving a CUDA context behind (no engine in this suite).
TEST(GpuCApiProbe, ProbeDeviceLeavesNoCudaContext) {
  if (!mi::testing::cuda_context_report().empty()) GTEST_SKIP() << "the process already holds a CUDA context";
  mi_probe_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_probe(&info), MI_OK) << last_error();
  if (info.supported_device_count == 0) GTEST_SKIP() << "no supported CUDA device";
  for (int32_t i = 0; i < info.device_count; ++i) {
    mi_device_info dev{};
    dev.struct_size = sizeof dev;
    ASSERT_EQ(mi_probe_device(i, &dev), MI_OK) << last_error();
    if (dev.supported) {
      EXPECT_GT(dev.free_memory_bytes, 0u) << i;
    }
  }
  EXPECT_EQ(mi::testing::cuda_context_report(), "");
}

bool have_gpu_and_weights(std::string& why) {
  mi_probe_info info{};
  info.struct_size = sizeof info;
  if (mi_probe(&info) != MI_OK || info.supported_device_count == 0) {
    why = "no supported CUDA device";
    return false;
  }
  if (!fs::is_directory(weights_root() / "base" / "transformer")) {
    why = "weights not present (" + mi::util::path_utf8(weights_root()) + ")";
    return false;
  }
  return true;
}

// One engine for the suite: the DiT stays resident between tests.
class GpuCApi : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!have_gpu_and_weights(skip_reason_)) return;
    model_ = mi::util::path_utf8(weights_root() / "base");
    mi_engine_params p = engine_defaults();
    p.model_path = model_.c_str();
    ASSERT_EQ(mi_engine_create(&p, &engine_), MI_OK) << last_error();
  }
  static void TearDownTestSuite() {
    mi_engine_free(engine_);
    engine_ = nullptr;
  }
  void SetUp() override {
    if (engine_ == nullptr) GTEST_SKIP() << skip_reason_;
  }

  static mi_gen_params small_request(int steps) {
    mi_gen_params p = gen_defaults();
    p.prompt = "a red apple on a wooden table";
    p.negative_prompt = "blurry";
    p.width = p.height = 256;
    p.steps = steps;
    return p;
  }

  static inline mi_engine* engine_ = nullptr;
  static inline std::string model_;
  static inline std::string skip_reason_;
};

struct Events {
  std::vector<mi_progress> progress;
  int plans = 0;
  std::string plan_json;
  std::atomic<int> denoise_steps{0};
  mi_engine* engine = nullptr;
  mi_status concurrent = MI_OK;  // a second mi_generate() from another thread while this one runs
};

void on_progress(const mi_progress* p, void* user) {
  auto& ev = *static_cast<Events*>(user);
  ev.progress.push_back(*p);
  if (p->phase == MI_PHASE_DENOISE && p->step > 0) ev.denoise_steps.store(p->step);
  if (ev.engine != nullptr && p->phase == MI_PHASE_DENOISE && p->step == 1) {
    std::thread([&ev] {
      mi_gen_params q;
      (void)mi_gen_params_init(&q, sizeof q);
      q.prompt = "x";
      mi_result* r = nullptr;
      ev.concurrent = mi_generate(ev.engine, &q, &r);
      mi_result_free(r);
    }).join();
  }
}

void on_plan(const char* explanation, const char* plan_json, void* user) {
  auto& ev = *static_cast<Events*>(user);
  ++ev.plans;
  ev.plan_json = plan_json;
  EXPECT_GT(std::strlen(explanation), 0u);
}

TEST_F(GpuCApi, ProbeDevice) {
  mi_device_info dev{};
  dev.struct_size = sizeof dev;
  ASSERT_EQ(mi_probe_device(0, &dev), MI_OK) << last_error();
  EXPECT_EQ(dev.index, 0);
  EXPECT_GT(std::strlen(dev.name), 0u);
  EXPECT_GE(dev.compute_major, 8);
  EXPECT_EQ(dev.supported, 1) << dev.unsupported_reason;
  EXPECT_STREQ(dev.unsupported_reason, "");
  EXPECT_GT(dev.total_memory_bytes, 0u);
  EXPECT_GT(dev.free_memory_bytes, 0u);
  EXPECT_LE(dev.free_memory_bytes, dev.total_memory_bytes);
  std::printf("[c_api] device 0: %s (%s), cc %d.%d, %.2f / %.2f GiB free\n", dev.name, dev.pci_bus_id, dev.compute_major,
              dev.compute_minor, static_cast<double>(dev.free_memory_bytes) / (1 << 30),
              static_cast<double>(dev.total_memory_bytes) / (1 << 30));
}

TEST_F(GpuCApi, Generate256TwoSteps) {
  Events ev;
  ev.engine = engine_;
  mi_gen_params p = small_request(2);
  p.progress_callback = on_progress;
  p.plan_callback = on_plan;
  p.user_data = &ev;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(ev.concurrent, MI_ERR_BUSY);
  EXPECT_EQ(ev.plans, 1);
  EXPECT_NO_THROW(mi::json::parse(ev.plan_json));

  // Phase starts in order, with the denoise steps in between.
  std::vector<std::pair<int, int>> seen;  // (phase, step)
  for (const mi_progress& e : ev.progress) {
    EXPECT_EQ(e.struct_size, sizeof(mi_progress));
    EXPECT_EQ(e.total_steps, 2);
    seen.emplace_back(e.phase, e.step);
  }
  const std::vector<std::pair<int, int>> expected = {{MI_PHASE_TEXT_ENCODER, 0}, {MI_PHASE_DIT_LOAD, 0}, {MI_PHASE_DENOISE, 0},
                                                     {MI_PHASE_DENOISE, 1},      {MI_PHASE_DENOISE, 2},  {MI_PHASE_DECODE, 0}};
  EXPECT_EQ(seen, expected);
  for (std::size_t i = 1; i < ev.progress.size(); ++i) {
    EXPECT_GE(ev.progress[i].elapsed_seconds, ev.progress[i - 1].elapsed_seconds);
  }

  EXPECT_EQ(mi_result_cancelled(r), 0);
  EXPECT_EQ(mi_result_width(r), 256);
  EXPECT_EQ(mi_result_height(r), 256);
  EXPECT_EQ(mi_result_seed(r), 42u);
  EXPECT_EQ(mi_result_steps_run(r), 2);
  EXPECT_EQ(mi_result_steps_scheduled(r), 2);
  std::size_t bytes = 0;
  const std::uint8_t* rgb = mi_result_rgb(r, &bytes);
  ASSERT_NE(rgb, nullptr);
  ASSERT_EQ(bytes, std::size_t{256 * 256 * 3});
  double mean = 0;
  for (std::size_t i = 0; i < bytes; ++i) mean += rgb[i];
  mean /= static_cast<double>(bytes);
  EXPECT_GT(mean, 5.0);
  EXPECT_LT(mean, 250.0);
  mi_timings t{};
  t.struct_size = sizeof t;
  ASSERT_EQ(mi_result_timings(r, &t), MI_OK);
  EXPECT_GT(t.total_s, 0.0);
  EXPECT_GT(t.denoise_s, 0.0);
  EXPECT_NEAR(t.load_s, t.index_load_s + t.dit_load_s + t.vae_load_s, 1e-12);
  mi_memory_stats m{};
  m.struct_size = sizeof m;
  ASSERT_EQ(mi_result_memory(r, &m), MI_OK);
  EXPECT_GT(m.arena_high_water_bytes, 0u);
  EXPECT_LE(m.arena_high_water_bytes, m.arena_capacity_bytes);
  EXPECT_GT(m.host_private_peak_bytes, 0u);
  EXPECT_GT(std::strlen(mi_result_plan_explanation(r)), 0u);

  // The default block path is fused; op by op only when the startup self-check failed, and then the note says why.
  const mi_block_path path = mi_result_block_path(r);
  const char* note = mi_result_block_path_note(r);
  ASSERT_NE(note, nullptr);
  EXPECT_TRUE(path == MI_BLOCK_PATH_FUSED || (path == MI_BLOCK_PATH_OP_BY_OP && *note != '\0')) << path << ": " << note;
  if (path == MI_BLOCK_PATH_FUSED) {
    EXPECT_STREQ(note, "");
  }
  // 256^2 in balanced mode on the 16 GiB dev box: every DiT block resident, nothing streamed.
  mi_dit_stream_stats ds;
  std::memset(&ds, 0xab, sizeof ds);
  ds.struct_size = sizeof ds;
  ASSERT_EQ(mi_result_dit_stream(r, &ds), MI_OK) << last_error();
  EXPECT_EQ(ds.struct_size, sizeof ds);
  EXPECT_EQ(ds.streamed, 0);
  EXPECT_EQ(ds.bytes, 0u);
  EXPECT_EQ(ds.stalls, 0);
  EXPECT_EQ(ds.copy_seconds, 0.0);
  EXPECT_STREQ(ds.source, "");
  ds.struct_size = sizeof ds - 8;
  EXPECT_EQ(mi_result_dit_stream(r, &ds), MI_ERR_ABI_MISMATCH);
  mi_dit_stream_stats* no_out = nullptr;
  EXPECT_EQ(mi_result_dit_stream(r, no_out), MI_ERR_INVALID_ARGUMENT);
  std::printf("[c_api] DiT blocks %s%s%s\n", path == MI_BLOCK_PATH_FUSED ? "fused" : "op_by_op", *note ? ": " : "", note);

  mi::testing::TempDir dir("c_api_gen");
  const std::string png = mi::util::path_utf8(dir.path() / "out.png");
  ASSERT_EQ(mi_save_png(r, png.c_str()), MI_OK) << last_error();
  std::ifstream in(png, std::ios::binary);
  char magic[8] = {};
  in.read(magic, 8);
  EXPECT_EQ(std::memcmp(magic, "\x89PNG\r\n\x1a\n", 8), 0);
  EXPECT_EQ(mi_save_png(r, mi::util::path_utf8(dir.path() / "no_such_dir" / "x.png").c_str()), MI_ERR_IO);

  // The report: the CLI's --json-report shape, with the saved image.
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  EXPECT_EQ(doc.at("image").as_string(), png);
  EXPECT_EQ(doc.at("request").at("model").as_string(), model_);
  EXPECT_EQ(doc.at("request").at("width").as_int(), 256);
  EXPECT_EQ(doc.at("steps_run").as_int(), 2);
  EXPECT_FALSE(doc.at("cancelled").as_bool());
  EXPECT_TRUE(doc.at("decode").at("ran").as_bool());
  EXPECT_EQ(doc.at("mode").as_string(), "balanced");
  for (const char* key : {"plan", "embedding_cache", "timings_s", "memory_gib", "host_memory_gib", "latents", "seed_used",
                          "hardware"}) {
    EXPECT_NE(doc.find(key), nullptr) << key;
  }
  EXPECT_EQ(doc.at("dit_blocks").at("path").as_string(), path == MI_BLOCK_PATH_FUSED ? "fused" : "op_by_op");
  EXPECT_EQ(doc.at("dit_blocks").at("note").as_string(), note);
  EXPECT_FALSE(doc.at("dit_stream").at("streamed").as_bool());
  mi_result_free(r);
}

// mi_engine_create / mi_generate / mi_engine_free leave the calling thread's current CUDA context as they found it:
// none on a fresh thread, the caller's own otherwise.
TEST_F(GpuCApi, RestoresTheCallersCurrentContext) {
  std::optional<void*> before, after_create, after_generate, after_free;
  mi_status create = MI_OK, generate = MI_OK;
  std::thread([&] {
    before = mi::testing::current_context();
    mi_engine_params p = engine_defaults();
    p.model_path = model_.c_str();
    mi_engine* second = nullptr;
    create = mi_engine_create(&p, &second);
    after_create = mi::testing::current_context();
    mi_engine_free(second);
    after_free = mi::testing::current_context();
    mi_gen_params g = small_request(2);
    mi_result* r = nullptr;
    generate = mi_generate(engine_, &g, &r);
    after_generate = mi::testing::current_context();
    mi_result_free(r);
  }).join();
  ASSERT_EQ(create, MI_OK);
  ASSERT_EQ(generate, MI_OK);
  ASSERT_TRUE(before.has_value());
  EXPECT_EQ(*before, nullptr);
  EXPECT_EQ(after_create, before);
  EXPECT_EQ(after_free, before);
  EXPECT_EQ(after_generate, before);

  // A thread with its own current context keeps it.
  std::optional<void*> mine, after;
  std::thread([&] {
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) return;
    mine = mi::testing::current_context();
    mi_gen_params g = small_request(2);
    mi_result* r = nullptr;
    generate = mi_generate(engine_, &g, &r);
    mi_result_free(r);
    after = mi::testing::current_context();
  }).join();
  ASSERT_TRUE(mine.has_value());
  EXPECT_NE(*mine, nullptr);
  EXPECT_EQ(generate, MI_OK);
  EXPECT_EQ(after, mine);
}

// mi_cancel from another thread stops a long run mid-sampling; the cancel does not carry over to the next generation.
TEST_F(GpuCApi, CancelFromAnotherThread) {
  ASSERT_EQ(mi_cancel(engine_), MI_OK);  // idle: no effect
  Events ev;
  mi_gen_params p = small_request(50);
  p.progress_callback = on_progress;
  p.user_data = &ev;
  mi_result* r = nullptr;
  mi_status status = MI_OK;
  std::thread worker([&] { status = mi_generate(engine_, &p, &r); });
  while (ev.denoise_steps.load() < 2) std::this_thread::yield();
  ASSERT_EQ(mi_cancel(engine_), MI_OK);
  worker.join();
  ASSERT_EQ(status, MI_CANCELLED) << last_error();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(mi_result_cancelled(r), 1);
  EXPECT_GE(mi_result_steps_run(r), 2);
  EXPECT_LT(mi_result_steps_run(r), mi_result_steps_scheduled(r));
  EXPECT_EQ(mi_result_rgb(r, nullptr), nullptr);
  EXPECT_EQ(mi_save_png(r, "never.png"), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("cancelled"), std::string::npos) << last_error();
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  EXPECT_TRUE(doc.at("cancelled").as_bool());
  mi_result_free(r);

  // Cancels sent while idle, or after the cancelled run, do not affect the next generation.
  ASSERT_EQ(mi_cancel(engine_), MI_OK);
  mi_gen_params q = small_request(2);
  r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &q, &r), MI_OK) << last_error();
  EXPECT_EQ(mi_result_steps_run(r), 2);
  EXPECT_NE(mi_result_rgb(r, nullptr), nullptr);
  mi_result_free(r);
}

}  // namespace
