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
#include <iterator>
#include <limits>
#include <locale>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "api/pipeline_adapter.h"
#include "cuda_context_check.h"
#include "models/lora/lora.h"
#include "pipeline/pipeline.h"
#include "pipeline/png_writer.h"
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
  // P6: preview, component files, LoRA, tiers and the extended samplers; a ComfyUI file as model_path is not.
  EXPECT_EQ(mi_capabilities(), MI_CAP_PREVIEW | MI_CAP_MODEL_FILES | MI_CAP_LORA | MI_CAP_TIERS | MI_CAP_SAMPLERS);
  EXPECT_EQ(mi_capabilities() & MI_CAP_COMFYUI_MODELS, 0u);
  EXPECT_EQ(MI_CAP_PREVIEW | MI_CAP_COMFYUI_MODELS | MI_CAP_MODEL_FILES | MI_CAP_LORA | MI_CAP_TIERS | MI_CAP_SAMPLERS, 0x3fu);
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
  // P6 fields.
  EXPECT_EQ(g.scheduler, MI_SCHEDULER_MOSS);
  EXPECT_EQ(q.scheduler, mi::pipeline::SchedulerKind::kMoss);
  EXPECT_EQ(g.noise_layout, MI_NOISE_LAYOUT_OFFICIAL);
  EXPECT_EQ(q.noise_layout, mi::pipeline::NoiseLayout::kOfficial);
  EXPECT_EQ(g.noise_source, MI_NOISE_SOURCE_PHILOX);
  EXPECT_EQ(q.noise_source, mi::pipeline::NoiseSource::kPhilox);
  EXPECT_EQ(g.save_metadata, q.save_metadata ? 1 : 0);
  EXPECT_EQ(g.flow_multiplier, q.flow_multiplier);
  EXPECT_EQ(g.eta, q.eta);
  EXPECT_EQ(g.s_noise, q.s_noise);
  EXPECT_EQ(g.dataspace_path, nullptr);
  EXPECT_TRUE(q.dataspace_path.empty());
  EXPECT_EQ(g.loras, nullptr);
  EXPECT_EQ(g.loras_count, q.loras.size());
  EXPECT_EQ(g.include_prompts, mi::pipeline::GenerationMetadata{}.include_prompts ? 1 : 0);
  EXPECT_EQ(g._reserved2, 0);

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
  // P6 fields.
  EXPECT_EQ(e.dit_file, nullptr);
  EXPECT_EQ(e.text_encoder_file, nullptr);
  EXPECT_EQ(e.vae_file, nullptr);
  EXPECT_TRUE(o.dit_file.empty() && o.text_encoder_file.empty() && o.vae_file.empty());
  EXPECT_EQ(e.tier, MI_TIER_AUTO);
  EXPECT_EQ(o.tier, mi::planner::Tier::kAuto);
  EXPECT_EQ(e.attention, MI_ATTENTION_DEFAULT);
  EXPECT_FALSE(o.attention.has_value());
  EXPECT_EQ(e.lora_strict, o.lora_strict ? 1 : 0);
  EXPECT_EQ(e._reserved2, 0);

  // The adapter turns the defaults into the pipeline's defaults.
  mi_gen_params valid = g;
  valid.prompt = "a cat";
  EXPECT_EQ(mi_gen_params_validate(&valid), MI_OK) << last_error();
  const mi::pipeline::Request r = mi::api::to_request(valid);
  EXPECT_EQ(r.sampler, q.sampler);
  EXPECT_EQ(r.scheduler, q.scheduler);
  EXPECT_EQ(r.noise_layout, q.noise_layout);
  EXPECT_EQ(r.noise_source, q.noise_source);
  EXPECT_EQ(r.save_metadata, q.save_metadata);
  EXPECT_EQ(r.loras.size(), 0u);
  const mi::pipeline::PipelineOptions ro = mi::api::to_options(e);
  EXPECT_EQ(ro.tier, o.tier);
  EXPECT_EQ(ro.attention, o.attention);
  EXPECT_EQ(ro.lora_strict, o.lora_strict);
  EXPECT_EQ(ro.block_path, o.block_path);
}

// Every C enum value maps to the pipeline value of the same name; anything else is rejected.
TEST(CApi, EnumsMapToThePipeline) {
  mi::testing::TempDir dir("c_api_enums");
  const fs::path ds = dir.path() / "dataspace.safetensors";
  std::ofstream(ds) << "x";
  const std::string ds_s = mi::util::path_utf8(ds);
  const auto request = [&](auto edit) {
    mi_gen_params p = gen_defaults();
    p.prompt = "a cat";
    p.dataspace_path = ds_s.c_str();  // MI_SAMPLER_DATASPACE needs it; the others ignore it
    edit(p);
    return p;
  };
  const char* samplers[] = {"euler", "ab2", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde",
                            "res_multistep", "dataspace"};
  for (int i = 0; i < 9; ++i) {
    const mi_gen_params p = request([&](mi_gen_params& g) { g.sampler = static_cast<mi_sampler>(i); });
    EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << samplers[i] << ": " << last_error();
    EXPECT_STREQ(mi::pipeline::sampler_name(mi::api::to_request(p).sampler), samplers[i]);
  }
  EXPECT_EQ(MI_SAMPLER_DATASPACE, 8);
  const char* schedulers[] = {"moss", "simple", "normal", "karras", "exponential", "beta", "sgm_uniform"};
  for (int i = 0; i < 7; ++i) {
    const mi_gen_params p = request([&](mi_gen_params& g) { g.scheduler = static_cast<mi_scheduler>(i); });
    EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << schedulers[i] << ": " << last_error();
    EXPECT_STREQ(mi::pipeline::scheduler_name(mi::api::to_request(p).scheduler), schedulers[i]);
  }
  const char* layouts[] = {"official", "batch_stream"};
  for (int i = 0; i < 2; ++i) {
    const mi_gen_params p = request([&](mi_gen_params& g) { g.noise_layout = static_cast<mi_noise_layout>(i); });
    EXPECT_STREQ(mi::pipeline::noise_layout_name(mi::api::to_request(p).noise_layout), layouts[i]);
  }
  const char* sources[] = {"philox", "philox_device", "brownian_tree"};
  for (int i = 0; i < 3; ++i) {
    const mi_gen_params p = request([&](mi_gen_params& g) { g.noise_source = static_cast<mi_noise_source>(i); });
    EXPECT_STREQ(mi::pipeline::noise_source_name(mi::api::to_request(p).noise_source), sources[i]);
  }
  struct Bad {
    const char* field;
    void (*edit)(mi_gen_params&);
  };
  const Bad bad[] = {
      {"sampler", [](mi_gen_params& p) { p.sampler = static_cast<mi_sampler>(9); }},
      {"sampler", [](mi_gen_params& p) { p.sampler = MI_SAMPLER_MAX_ENUM; }},
      {"scheduler", [](mi_gen_params& p) { p.scheduler = static_cast<mi_scheduler>(7); }},
      {"scheduler", [](mi_gen_params& p) { p.scheduler = MI_SCHEDULER_MAX_ENUM; }},
      {"noise_layout", [](mi_gen_params& p) { p.noise_layout = static_cast<mi_noise_layout>(2); }},
      {"noise_source", [](mi_gen_params& p) { p.noise_source = static_cast<mi_noise_source>(3); }},
  };
  for (const Bad& b : bad) {
    const mi_gen_params p = request(b.edit);
    EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT) << b.field;
    EXPECT_NE(last_error().find(std::string(b.field) + ": unknown value"), std::string::npos) << last_error();
  }

  // Engine enums: tier (C order AUTO first) and attention (DEFAULT = unset).
  const mi::planner::Tier tiers[] = {mi::planner::Tier::kAuto, mi::planner::Tier::kReference, mi::planner::Tier::kHigh,
                                     mi::planner::Tier::kTurbo};
  for (int i = 0; i < 4; ++i) {
    mi_engine_params e = engine_defaults();
    e.tier = static_cast<mi_tier>(i);
    EXPECT_EQ(mi::api::to_options(e).tier, tiers[i]) << i;
  }
  EXPECT_FALSE(mi::api::to_options(engine_defaults()).attention.has_value());
  const char* attentions[] = {"exact", "q8_fp16", "q8_fp16_f32acc", "q8_fp8", "q8_fp8_f32acc"};
  for (int i = 1; i < 6; ++i) {
    mi_engine_params e = engine_defaults();
    e.attention = static_cast<mi_attention>(i);
    const std::optional<mi::planner::AttentionMode> a = mi::api::to_options(e).attention;
    ASSERT_TRUE(a.has_value()) << i;
    EXPECT_STREQ(mi::planner::attention_name(*a), attentions[i - 1]);
  }
  mi_engine_params e = engine_defaults();
  e.tier = static_cast<mi_tier>(4);
  EXPECT_THROW(mi::api::to_options(e), std::invalid_argument);
  e = engine_defaults();
  e.attention = static_cast<mi_attention>(6);
  EXPECT_THROW(mi::api::to_options(e), std::invalid_argument);
  e = engine_defaults();
  e.lora_strict = 1;
  EXPECT_TRUE(mi::api::to_options(e).lora_strict);
}

// The 0.1 sizes (mi_engine_params 80, mi_gen_params 168) are accepted: init fills only that prefix, and the library
// reads only that prefix (the later fields keep their defaults).
TEST(CApi, OldStructSizes) {
  constexpr std::size_t kGen01 = 168, kEngine01 = 80;
  mi_gen_params g;
  std::memset(&g, 0x5a, sizeof g);
  ASSERT_EQ(mi_gen_params_init(&g, kGen01), MI_OK) << last_error();
  EXPECT_EQ(g.struct_size, kGen01);
  EXPECT_EQ(g.width, 1024);
  EXPECT_EQ(g.user_data, nullptr);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&g);
  for (std::size_t i = kGen01; i < sizeof g; ++i) ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " written";
  // The tail holds garbage (unknown enum values, nonzero _reserved2, a NULL LoRA array with a count): never read.
  g.prompt = "a cat";
  g.loras_count = 3;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_OK) << last_error();
  g.width = 1000;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);  // the prefix is still checked
  EXPECT_NE(last_error().find("multiples of 16"), std::string::npos) << last_error();
  g.width = 1024;
  g._reserved1 = 1;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("_reserved1"), std::string::npos) << last_error();
  // The current size reads the tail: the same garbage is rejected.
  g._reserved1 = 0;
  g.struct_size = sizeof g;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);
  for (const std::size_t bad : {kGen01 - 8, kGen01 + 8, sizeof g - 8}) {
    g.struct_size = bad;
    EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_ABI_MISMATCH) << bad;
    EXPECT_NE(last_error().find("168 or 240"), std::string::npos) << last_error();
  }

  mi_engine_params e;
  std::memset(&e, 0x5a, sizeof e);
  ASSERT_EQ(mi_engine_params_init(&e, kEngine01), MI_OK) << last_error();
  EXPECT_EQ(e.struct_size, kEngine01);
  EXPECT_EQ(e.mode, MI_MODE_BALANCED);
  EXPECT_EQ(e._reserved1, 0);
  bytes = reinterpret_cast<const unsigned char*>(&e);
  for (std::size_t i = kEngine01; i < sizeof e; ++i) ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " written";
  // Create reads the prefix only: the tail's nonzero _reserved2 is not seen, the model path check is reached.
  mi::testing::TempDir dir("c_api_old_sizes");
  const std::string missing = mi::util::path_utf8(dir.path() / "missing");
  e.model_path = missing.c_str();
  mi_engine* engine = nullptr;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("not found"), std::string::npos) << last_error();
  EXPECT_EQ(engine, nullptr);
  e.struct_size = sizeof e;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("_reserved2"), std::string::npos) << last_error();
  for (const std::size_t bad : {kEngine01 - 8, kEngine01 + 8, sizeof e + 8}) {
    EXPECT_EQ(mi_engine_params_init(&e, bad), MI_ERR_ABI_MISMATCH) << bad;
    EXPECT_NE(last_error().find("80 or 120"), std::string::npos) << last_error();
  }
}

// Runtime LoRAs and the dataspace file are checked without a GPU, their files included.
TEST(CApi, LoraAndDataspaceValidation) {
  mi::testing::TempDir dir("c_api_lora");
  const fs::path file = dir.path() / "style.safetensors";
  std::ofstream(file) << "x";
  const std::string file_s = mi::util::path_utf8(file);
  const std::string missing_s = mi::util::path_utf8(dir.path() / "missing.safetensors");
  const std::string dir_s = mi::util::path_utf8(dir.path());
  struct Case {
    const char* what;
    std::vector<mi_lora> list;
    std::size_t count;  // loras_count; the array is NULL when `list` is empty
    const char* message;
  };
  const double inf = std::numeric_limits<double>::infinity();
  const Case cases[] = {
      {"NULL array", {}, 2, "loras: NULL array"},
      {"too many", std::vector<mi_lora>(65, mi_lora{file_s.c_str(), 1.0}), 65, "at most 64"},
      {"NULL path", {{nullptr, 1.0}}, 1, "loras[0].path is required"},
      {"empty path", {{file_s.c_str(), 1.0}, {"", 1.0}}, 2, "loras[1].path is required"},
      {"infinite strength", {{file_s.c_str(), inf}}, 1, "loras[0].strength"},
      {"NaN strength", {{file_s.c_str(), std::nan("")}}, 1, "loras[0].strength"},
      {"beyond float", {{file_s.c_str(), 1e300}}, 1, "loras[0].strength"},
      {"missing file", {{missing_s.c_str(), 1.0}}, 1, "not found"},
      {"directory", {{dir_s.c_str(), 1.0}}, 1, "not a regular file"},
  };
  for (const Case& c : cases) {
    mi_gen_params p = gen_defaults();
    p.prompt = "a cat";
    p.loras = c.list.empty() ? nullptr : c.list.data();
    p.loras_count = c.count;
    EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
  }
  // Valid: in order, with float strengths (negative allowed); 64 is the cap.
  const mi_lora ok[] = {{file_s.c_str(), 0.75}, {file_s.c_str(), -0.5}};
  mi_gen_params p = gen_defaults();
  p.prompt = "a cat";
  p.loras = ok;
  p.loras_count = 2;
  ASSERT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  const mi::pipeline::Request q = mi::api::to_request(p);
  ASSERT_EQ(q.loras.size(), 2u);
  EXPECT_EQ(q.loras[0].path, file);
  EXPECT_EQ(q.loras[0].strength, 0.75f);
  EXPECT_EQ(q.loras[1].strength, -0.5f);
  const std::vector<mi_lora> max(64, mi_lora{file_s.c_str(), 1.0});
  p.loras = max.data();
  p.loras_count = max.size();
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  p.loras = nullptr;
  p.loras_count = 0;

  // The dataspace sampler needs its file, and it must exist; the other samplers ignore dataspace_path.
  p.sampler = MI_SAMPLER_DATASPACE;
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("dataspace"), std::string::npos) << last_error();
  p.dataspace_path = "";
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  p.dataspace_path = missing_s.c_str();
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("dataspace_path: file not found"), std::string::npos) << last_error();
  p.dataspace_path = file_s.c_str();
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  EXPECT_EQ(mi::api::to_request(p).dataspace_path, file);
  p.sampler = MI_SAMPLER_EULER;
  p.dataspace_path = missing_s.c_str();
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();

  // The other new numbers go through the pipeline's checks.
  p.dataspace_path = nullptr;
  p.flow_multiplier = 0;
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("Flow multiplier"), std::string::npos) << last_error();
  p.flow_multiplier = 1;
  p.eta = -1;
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("eta"), std::string::npos) << last_error();
  p.eta = 1;
  p.s_noise = std::nan("");
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("s_noise"), std::string::npos) << last_error();
}

// Component files: each given one must exist and be a file (checked before CUDA); "" means the bundle's.
TEST(CApi, ComponentFiles) {
  mi::testing::TempDir dir("c_api_files");
  make_fake_bundle(dir.path() / "bundle");
  const std::string bundle = mi::util::path_utf8(dir.path() / "bundle");
  const fs::path dit = dir.path() / "moss_int8.safetensors";
  std::ofstream(dit) << "x";
  const std::string dit_s = mi::util::path_utf8(dit);
  const std::string missing = mi::util::path_utf8(dir.path() / "missing.safetensors");
  const std::string folder = mi::util::path_utf8(dir.path());
  for (const char* field : {"dit_file", "text_encoder_file", "vae_file"}) {
    for (const std::string* path : {&missing, &folder}) {
      mi_engine_params e = engine_defaults();
      e.model_path = bundle.c_str();
      const char*& slot = std::string(field) == "dit_file"            ? e.dit_file
                          : std::string(field) == "text_encoder_file" ? e.text_encoder_file
                                                                      : e.vae_file;
      slot = path->c_str();
      mi_engine* engine = nullptr;
      EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT) << field;
      EXPECT_EQ(engine, nullptr);
      EXPECT_NE(last_error().find(std::string(field) + ": "), std::string::npos) << last_error();
    }
  }
  mi_engine_params e = engine_defaults();
  e.dit_file = dit_s.c_str();
  e.text_encoder_file = "";
  e.vae_file = nullptr;
  const mi::pipeline::PipelineOptions o = mi::api::to_options(e);
  EXPECT_EQ(o.dit_file, dit);
  EXPECT_TRUE(o.text_encoder_file.empty());
  EXPECT_TRUE(o.vae_file.empty());
  // Unknown tier / attention values fail mi_engine_create before CUDA.
  e = engine_defaults();
  e.model_path = bundle.c_str();
  e.tier = MI_TIER_MAX_ENUM;
  mi_engine* engine = nullptr;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("tier: unknown value"), std::string::npos) << last_error();
  e.tier = MI_TIER_AUTO;
  e.attention = static_cast<mi_attention>(6);
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("attention: unknown value 6"), std::string::npos) << last_error();
  EXPECT_EQ(engine, nullptr);
}

// A synthetic pipeline result through the adapter: every P6 accessor's data and the PNG metadata.
TEST(CApi, GenerationFromAPipelineResult) {
  mi::pipeline::Request q;
  q.model = "/models/MOSS-Image/base";
  q.prompt = "a secret prompt";
  q.negative_prompt = "blurry";
  q.sampler = mi::pipeline::SamplerKind::kDpmpp2m;
  q.scheduler = mi::pipeline::SchedulerKind::kKarras;
  q.loras = {{"/loras/style.safetensors", 0.5f}};
  mi::pipeline::Result r;
  r.width = r.height = 16;
  r.rgb.assign(16 * 16 * 3, 7);
  r.seed = 42;
  r.tier.requested = mi::planner::Tier::kAuto;
  r.tier.effective = mi::planner::Tier::kHigh;
  r.tier.precision = mi::planner::Precision::kInt8;
  r.tier.attention = mi::planner::AttentionMode::kQ8Fp8;
  r.tier.certified = true;
  r.tier.certificate = "/models/moss_int8.mossinfer-cert.json";
  r.tier.reason = "auto: certified q8_fp8";
  r.gpu_arch = "sm_89";
  r.dit_source = "/models/moss_int8.safetensors";
  r.te_source = "/models/te_w4a8.safetensors";
  r.te_precision = "w4a8";
  r.vae_source = "/models/MOSS-Image/base/vae";
  r.cond_token_report.total_tokens = 40;
  r.cond_token_report.kept_tokens = 16;
  r.cond_token_report.template_tokens = 12;
  r.cond_token_report.max_length = 16;
  r.cond_token_report.truncated = true;
  r.uncond_token_report.total_tokens = 14;
  r.uncond_token_report.kept_tokens = 14;
  r.uncond_token_report.template_tokens = 12;
  r.uncond_token_report.max_length = 16;
  mi::lora::FileReport f;
  f.file = "/loras/style.safetensors";
  f.adapters = 6;
  f.side_branch = 4;
  f.merged = 2;
  f.unknown = {"a: x", "b: y"};
  f.warnings = {"blocks.0.attn.qkv: requantized"};
  f.notes = {"alpha 8 from the metadata"};
  r.loras = {f};
  r.lora_warnings = f.warnings;
  r.lora_notes = f.notes;

  for (const bool include_prompts : {true, false}) {
    mi::pipeline::Result copy = r;
    const mi::api::Generation g = mi::api::to_generation(q, copy, include_prompts);
    EXPECT_EQ(g.rgb.size(), 16u * 16 * 3);
    EXPECT_EQ(g.tier.struct_size, sizeof(mi_tier_info));
    EXPECT_EQ(g.tier.requested, MI_TIER_AUTO);
    EXPECT_EQ(g.tier.effective, MI_TIER_HIGH);
    EXPECT_EQ(g.tier.attention, MI_ATTENTION_Q8_FP8);
    EXPECT_EQ(g.tier.certified, 1);
    EXPECT_EQ(g.tier.dit_precision, MI_PRECISION_INT8);
    EXPECT_EQ(g.tier.text_encoder_precision, MI_PRECISION_W4A8);
    EXPECT_EQ(g.tier_reason, r.tier.reason);
    EXPECT_EQ(g.certificate, r.tier.certificate);
    EXPECT_EQ(g.gpu_arch, "sm_89");
    EXPECT_EQ(g.dit_source, r.dit_source);
    EXPECT_EQ(g.te_source, r.te_source);
    EXPECT_EQ(g.vae_source, r.vae_source);
    EXPECT_NO_THROW(mi::json::parse(g.quality_json)) << g.quality_json;
    EXPECT_EQ(g.tokens[0].total_tokens, 40);
    EXPECT_EQ(g.tokens[0].kept_tokens, 16);
    EXPECT_EQ(g.tokens[0].prompt_tokens, 28);
    EXPECT_EQ(g.tokens[0].dropped_tokens, 24);
    EXPECT_EQ(g.tokens[0].max_length, 16);
    EXPECT_EQ(g.tokens[0].truncated, 1);
    EXPECT_EQ(g.tokens[1].prompt_tokens, 2);
    EXPECT_EQ(g.tokens[1].truncated, 0);
    ASSERT_EQ(g.loras.size(), 1u);
    EXPECT_EQ(g.loras[0].struct_size, sizeof(mi_lora_report));
    EXPECT_EQ(g.loras[0].adapters, 6);
    EXPECT_EQ(g.loras[0].side_branch, 4);
    EXPECT_EQ(g.loras[0].merged, 2);
    EXPECT_EQ(g.loras[0].unknown_keys, 2);
    EXPECT_EQ(g.loras[0].warnings, 1);
    EXPECT_EQ(g.loras[0].notes, 1);
    EXPECT_EQ(g.lora_files, std::vector<std::string>{f.file});
    EXPECT_EQ(g.lora_warnings, f.warnings);
    EXPECT_EQ(g.lora_notes, f.notes);
    // The CLI's metadata chunks; the prompts only when asked.
    ASSERT_EQ(g.png_text.size(), 2u);
    EXPECT_EQ(g.png_text[0].first, "generation");
    EXPECT_EQ(g.png_text[1].first, "parameters");
    const mi::json::Value meta = mi::json::parse(g.png_text[0].second);
    EXPECT_EQ(meta.at("schema").as_string(), "mossinfer.generation/1");
    EXPECT_EQ(meta.at("sampler").as_string(), "dpmpp_2m");
    EXPECT_EQ(meta.at("scheduler").as_string(), "karras");
    EXPECT_EQ(meta.at("engine").at("tier").as_string(), "high");
    EXPECT_EQ(meta.at("loras").as_array().at(0).at("name").as_string(), "style.safetensors");
    EXPECT_EQ(meta.find("prompt") != nullptr, include_prompts);
    EXPECT_EQ(g.png_text[1].second.find("a secret prompt") != std::string::npos, include_prompts);
  }
  // save_metadata off, or cancelled: no chunks.
  mi::pipeline::Request no_meta = q;
  no_meta.save_metadata = false;
  mi::pipeline::Result copy = r;
  EXPECT_TRUE(mi::api::to_generation(no_meta, copy, true).png_text.empty());
  copy = r;
  copy.cancelled = true;
  EXPECT_TRUE(mi::api::to_generation(q, copy, true).png_text.empty());
  // An empty te_precision is the bundle's bf16 text encoder; a bf16 DiT is BF16.
  copy = r;
  copy.te_precision.clear();
  copy.tier.precision = mi::planner::Precision::kBf16;
  const mi::api::Generation g = mi::api::to_generation(q, copy, true);
  EXPECT_EQ(g.tier.text_encoder_precision, MI_PRECISION_BF16);
  EXPECT_EQ(g.tier.dit_precision, MI_PRECISION_BF16);
  // Reference needs no certificate: certified, also in the default decision of a generation cancelled before its tier
  // was resolved.
  mi::pipeline::Result early;
  early.cancelled = true;
  ASSERT_FALSE(early.tier.certified);
  const mi::api::Generation e = mi::api::to_generation(q, early, true);
  EXPECT_EQ(e.tier.requested, MI_TIER_AUTO);
  EXPECT_EQ(e.tier.effective, MI_TIER_REFERENCE);
  EXPECT_EQ(e.tier.attention, MI_ATTENTION_EXACT);
  EXPECT_EQ(e.tier.certified, 1);
  EXPECT_TRUE(e.png_text.empty());
  // An uncertified lossy run stays uncertified.
  copy = r;
  copy.tier.certified = false;
  EXPECT_EQ(mi::api::to_generation(q, copy, true).tier.certified, 0);
}

// mi_write_png: RGB8 to an atomic PNG without metadata.
TEST(CApi, WritePng) {
  mi::testing::TempDir dir("c_api_write_png");
  const std::string path = mi::util::path_utf8(dir.path() / "preview.png");
  std::vector<std::uint8_t> rgb(3 * 2 * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i * 13);
  ASSERT_EQ(mi_write_png(path.c_str(), rgb.data(), 3, 2), MI_OK) << last_error();
  std::ifstream in(path, std::ios::binary);
  const std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  ASSERT_GE(png.size(), 33u);
  EXPECT_EQ(std::memcmp(png.data(), "\x89PNG\r\n\x1a\n", 8), 0);
  const auto be32 = [&](std::size_t at) {
    return (std::uint32_t{png[at]} << 24) | (std::uint32_t{png[at + 1]} << 16) | (std::uint32_t{png[at + 2]} << 8) | png[at + 3];
  };
  EXPECT_EQ(std::memcmp(png.data() + 12, "IHDR", 4), 0);
  EXPECT_EQ(be32(16), 3u);  // width
  EXPECT_EQ(be32(20), 2u);  // height
  EXPECT_TRUE(mi::pipeline::read_png_text(png).empty());
  // The same bytes as the pipeline's encoder.
  EXPECT_EQ(png, mi::pipeline::encode_png_rgb8(rgb, 3, 2));

  EXPECT_EQ(mi_write_png(nullptr, rgb.data(), 3, 2), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_write_png("", rgb.data(), 3, 2), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_write_png(path.c_str(), nullptr, 3, 2), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_write_png(path.c_str(), rgb.data(), 0, 2), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_write_png(path.c_str(), rgb.data(), 3, -2), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("[1, 16384]"), std::string::npos) << last_error();
  EXPECT_EQ(mi_write_png(path.c_str(), rgb.data(), 16385, 1), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_write_png(path.c_str(), rgb.data(), 1, 16385), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_write_png(path.c_str(), rgb.data(), INT32_MAX, INT32_MAX), MI_ERR_INVALID_ARGUMENT);
  const std::vector<std::uint8_t> row(16384 * 3, 9);  // the largest side is accepted
  EXPECT_EQ(mi_write_png(path.c_str(), row.data(), 16384, 1), MI_OK) << last_error();
  EXPECT_EQ(mi_write_png(mi::util::path_utf8(dir.path() / "no_such_dir" / "x.png").c_str(), rgb.data(), 3, 2), MI_ERR_IO);
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
  // P6 accessors.
  mi_tier_info tier{};
  tier.struct_size = sizeof tier;
  EXPECT_EQ(mi_result_tier(nullptr, &tier), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  EXPECT_EQ(mi_result_text(nullptr, MI_TEXT_TIER_REASON), nullptr);
  EXPECT_EQ(mi_result_text(nullptr, MI_TEXT_QUALITY_JSON), nullptr);
  mi_prompt_tokens tokens{};
  tokens.struct_size = sizeof tokens;
  EXPECT_EQ(mi_result_prompt_tokens(nullptr, 0, &tokens), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_result_lora_count(nullptr), 0);
  mi_lora_report lr{};
  lr.struct_size = sizeof lr;
  EXPECT_EQ(mi_result_lora(nullptr, 0, &lr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_result_lora_file(nullptr, 0), nullptr);
  EXPECT_EQ(mi_result_message_count(nullptr, MI_MESSAGE_LORA_WARNING), 0);
  EXPECT_EQ(mi_result_message(nullptr, MI_MESSAGE_LORA_NOTE, 0), nullptr);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  // The output struct's size is checked before the handle: a wrong one is an ABI mismatch.
  tier.struct_size = sizeof tier - 4;
  EXPECT_EQ(mi_result_tier(nullptr, &tier), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("mi_tier_info.struct_size"), std::string::npos) << last_error();
  tokens.struct_size = sizeof tokens + 8;
  EXPECT_EQ(mi_result_prompt_tokens(nullptr, 0, &tokens), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("mi_prompt_tokens.struct_size"), std::string::npos) << last_error();
  lr.struct_size = 0;
  EXPECT_EQ(mi_result_lora(nullptr, 0, &lr), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("mi_lora_report.struct_size"), std::string::npos) << last_error();
  t.struct_size = 8;
  EXPECT_EQ(mi_result_timings(nullptr, &t), MI_ERR_ABI_MISMATCH);
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
      {"unknown sampler", [](mi_gen_params& p) { p.sampler = static_cast<mi_sampler>(9); }, "sampler"},
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
      // The three sizes share one rule: 0 or [1 MiB, 1 TiB].
      {"huge vram budget", with([](mi_engine_params& p) { p.vram_budget_bytes = (std::uint64_t{1} << 40) + 1; }),
       MI_ERR_INVALID_ARGUMENT, "vram_budget_bytes must be at most 1024 GiB"},
      {"tiny vram reserve", with([](mi_engine_params& p) { p.vram_reserve_bytes = (std::uint64_t{1} << 20) - 1; }),
       MI_ERR_INVALID_ARGUMENT, "vram_reserve_bytes must be at least 1 MiB"},
      {"huge vram reserve", with([](mi_engine_params& p) { p.vram_reserve_bytes = (std::uint64_t{1} << 40) + 1; }),
       MI_ERR_INVALID_ARGUMENT, "vram_reserve_bytes must be at most 1024 GiB"},
      {"tiny host budget", with([](mi_engine_params& p) { p.host_ram_budget_bytes = (std::uint64_t{1} << 20) - 1; }),
       MI_ERR_INVALID_ARGUMENT, "host_ram_budget_bytes must be at least 1 MiB"},
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
  g._reserved0 = 0;
  EXPECT_EQ(g._reserved2, 0);
  g._reserved2 = 3;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("mi_gen_params._reserved2"), std::string::npos) << last_error();

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
  e._reserved1 = 0;
  EXPECT_EQ(e._reserved2, 0);
  e._reserved2 = -4;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(engine, nullptr);
  EXPECT_NE(last_error().find("mi_engine_params._reserved2"), std::string::npos) << last_error();

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
  // Only meaningful in a context-free process. ctest runs each test in its own process, so this skips only when the
  // whole binary runs in one process after a test that created a context (GpuCApi.* create engines).
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
  // An index past the last device is rejected before any device is probed.
  mi_device_info dev{};
  dev.struct_size = sizeof dev;
  EXPECT_EQ(mi_probe_device(info.device_count, &dev), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("(" + std::to_string(info.device_count) + " CUDA devices)"), std::string::npos)
      << last_error();
  EXPECT_EQ(dev.name[0], '\0');
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

struct Previews {
  std::vector<int> steps;
  int width = 0, height = 0;
  double mean = 0;
};

void on_preview(const uint8_t* rgb, int32_t width, int32_t height, int32_t step, void* user) {
  auto& pv = *static_cast<Previews*>(user);
  pv.steps.push_back(step);
  pv.width = width;
  pv.height = height;
  double sum = 0;
  const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
  for (std::size_t i = 0; i < n; ++i) sum += rgb[i];
  pv.mean = sum / static_cast<double>(n);
}

// P6 options end to end: DPM++ 2M on the Karras schedule, previews, the PNG metadata, the tier and token accessors.
TEST_F(GpuCApi, SamplerPreviewMetadataAndTier) {
  Previews pv;
  mi_gen_params p = small_request(3);
  p.sampler = MI_SAMPLER_DPMPP_2M;
  p.scheduler = MI_SCHEDULER_KARRAS;
  p.preview_callback = on_preview;
  p.preview_interval = 2;  // previews after step 2 and after the last step
  p.save_metadata = 1;
  p.user_data = &pv;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  EXPECT_EQ(pv.steps, (std::vector<int>{2, 3}));
  EXPECT_EQ(pv.width, 256);
  EXPECT_EQ(pv.height, 256);
  EXPECT_GT(pv.mean, 1.0);
  EXPECT_EQ(mi_result_steps_run(r), 3);

  mi::testing::TempDir dir("c_api_p6");
  const std::string png = mi::util::path_utf8(dir.path() / "meta.png");
  ASSERT_EQ(mi_save_png(r, png.c_str()), MI_OK) << last_error();
  std::ifstream in(png, std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::vector<mi::pipeline::PngText> text = mi::pipeline::read_png_text(bytes);
  ASSERT_EQ(text.size(), 2u);
  EXPECT_EQ(text[0].keyword, "generation");
  const mi::json::Value meta = mi::json::parse(text[0].text);
  EXPECT_EQ(meta.at("sampler").as_string(), "dpmpp_2m");
  EXPECT_EQ(meta.at("scheduler").as_string(), "karras");
  EXPECT_EQ(meta.at("prompt").as_string(), p.prompt);
  EXPECT_EQ(meta.at("steps").as_int(), 3);

  mi_tier_info tier;
  std::memset(&tier, 0xab, sizeof tier);
  tier.struct_size = sizeof tier;
  ASSERT_EQ(mi_result_tier(r, &tier), MI_OK) << last_error();
  EXPECT_EQ(tier.struct_size, sizeof tier);
  EXPECT_EQ(tier.requested, MI_TIER_AUTO);
  EXPECT_NE(tier.effective, MI_TIER_AUTO);
  EXPECT_NE(tier.attention, MI_ATTENTION_DEFAULT);
  EXPECT_EQ(tier.dit_precision, MI_PRECISION_BF16);  // the bundle's DiT
  EXPECT_EQ(tier.text_encoder_precision, MI_PRECISION_BF16);
  if (tier.effective == MI_TIER_REFERENCE) {
    EXPECT_EQ(tier.attention, MI_ATTENTION_EXACT);
    EXPECT_EQ(tier.certified, 1);
  }
  const char* reason = mi_result_text(r, MI_TEXT_TIER_REASON);
  ASSERT_NE(reason, nullptr);
  std::printf("[c_api] tier auto -> effective %d, attention %d, certified %d: %s\n", tier.effective, tier.attention,
              tier.certified, reason);
  EXPECT_STRNE(mi_result_text(r, MI_TEXT_DIT_SOURCE), "");
  EXPECT_STRNE(mi_result_text(r, MI_TEXT_GPU_ARCH), "");
  EXPECT_NO_THROW(mi::json::parse(mi_result_text(r, MI_TEXT_QUALITY_JSON)));
  EXPECT_EQ(mi_result_text(r, MI_TEXT_MAX_ENUM), nullptr);

  mi_prompt_tokens tokens{};
  tokens.struct_size = sizeof tokens;
  ASSERT_EQ(mi_result_prompt_tokens(r, 0, &tokens), MI_OK) << last_error();
  EXPECT_GT(tokens.prompt_tokens, 0);
  EXPECT_GT(tokens.total_tokens, tokens.prompt_tokens);
  EXPECT_EQ(tokens.kept_tokens, tokens.total_tokens);
  EXPECT_EQ(tokens.truncated, 0);
  EXPECT_EQ(tokens.max_length, 4096);
  ASSERT_EQ(mi_result_prompt_tokens(r, 1, &tokens), MI_OK) << last_error();
  EXPECT_GT(tokens.prompt_tokens, 0);  // "blurry" under CFG 3
  EXPECT_EQ(mi_result_prompt_tokens(r, 2, &tokens), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_result_lora_count(r), 0);
  mi_lora_report lr{};
  lr.struct_size = sizeof lr;
  EXPECT_EQ(mi_result_lora(r, 0, &lr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_result_lora_file(r, 0), nullptr);
  EXPECT_EQ(mi_result_message_count(r, MI_MESSAGE_LORA_WARNING), 0);
  EXPECT_EQ(mi_result_message(r, MI_MESSAGE_LORA_WARNING, 0), nullptr);
  EXPECT_EQ(mi_result_message_count(r, static_cast<mi_message_kind>(5)), 0);
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  EXPECT_EQ(doc.at("request").at("sampler").as_string(), "dpmpp_2m");
  mi_result_free(r);
}

// A program built against the 0.1.0 header (168-byte mi_gen_params) keeps the 0.1.0 behaviour: PNGs without metadata.
TEST_F(GpuCApi, OldSizeWritesNoMetadata) {
  mi_gen_params p;
  std::memset(&p, 0x5a, sizeof p);  // the tail is not the caller's
  ASSERT_EQ(mi_gen_params_init(&p, 168), MI_OK) << last_error();
  p.prompt = "a red apple on a wooden table";
  p.width = p.height = 256;
  p.steps = 2;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  mi::testing::TempDir dir("c_api_old_size");
  const std::string png = mi::util::path_utf8(dir.path() / "old.png");
  ASSERT_EQ(mi_save_png(r, png.c_str()), MI_OK) << last_error();
  std::ifstream in(png, std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_TRUE(mi::pipeline::read_png_text(bytes).empty());
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  EXPECT_EQ(doc.at("request").at("scheduler").as_string(), "moss");  // the later fields took their defaults
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
