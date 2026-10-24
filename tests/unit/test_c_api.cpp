// mossinfer.h: CApi.* run without a GPU (versions, struct sizes, defaults, argument checks, error reporting, the
// probe); GpuCApi.* generate through the C API with the base bundle (skipped when the weights or a device are absent).
#include "mossinfer/mossinfer.h"

#include <gtest/gtest.h>

#include <cublasLt.h>  // never cublas_api.h directly: CUDA 12 requires CUBLASAPI, which cublasLt.h defines

#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <locale>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "api/pipeline_adapter.h"
#include "backend/cuda_libs.h"
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

// A dataspace.safetensors with an identity 2x2 mix and `codes` step codes.
void write_dataspace(const fs::path& file, const std::vector<float>& codes) {
  const std::size_t q_bytes = 4 * sizeof(float), c_bytes = codes.size() * sizeof(float);
  const std::string header = "{\"latent_mix.Q\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0," +
                             std::to_string(q_bytes) + "]},\"step_codes\":{\"dtype\":\"F32\",\"shape\":[" +
                             std::to_string(codes.size()) + "],\"data_offsets\":[" + std::to_string(q_bytes) + "," +
                             std::to_string(q_bytes + c_bytes) + "]}}";
  std::ofstream out(file, std::ios::binary);
  const std::uint64_t n = header.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xff));
  out << header;
  const float q[4] = {1, 0, 0, 1};
  out.write(reinterpret_cast<const char*>(q), sizeof q);
  out.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(c_bytes));
}

// ---------------------------------------------------------------- CPU

TEST(CApi, VersionAndAbi) {
  const std::string expected = std::to_string(MI_VERSION_MAJOR) + "." + std::to_string(MI_VERSION_MINOR) + "." +
                               std::to_string(MI_VERSION_PATCH);
  EXPECT_EQ(std::string(mi_version()), expected);
  EXPECT_EQ(mi_abi_version(), MI_ABI_VERSION);
  // P6: preview, component files, LoRA, tiers and the extended samplers; a ComfyUI file as model_path is not.
  EXPECT_EQ(mi_capabilities(), MI_CAP_PREVIEW | MI_CAP_MODEL_FILES | MI_CAP_LORA | MI_CAP_TIERS | MI_CAP_SAMPLERS |
                                  MI_CAP_IMAGE_INPUT | MI_CAP_HIRES | MI_CAP_DISTILL | MI_CAP_SYSTEM_CUDA | MI_CAP_CONTROL |
                                  MI_CAP_ACCEL | MI_CAP_PRELOAD | MI_CAP_RELEASE | MI_CAP_CERTIFICATE_DIRS |
                                  MI_CAP_TE_HOST_PIN | MI_CAP_PROGRESS_FRACTION);
  EXPECT_EQ(mi_capabilities() & MI_CAP_COMFYUI_MODELS, 0u);
  EXPECT_EQ(MI_CAP_PREVIEW | MI_CAP_COMFYUI_MODELS | MI_CAP_MODEL_FILES | MI_CAP_LORA | MI_CAP_TIERS | MI_CAP_SAMPLERS, 0x3fu);
  EXPECT_EQ(MI_CAP_IMAGE_INPUT, 0x40u);
  EXPECT_EQ(MI_CAP_HIRES, 0x80u);
  EXPECT_EQ(MI_CAP_DISTILL, 0x100u);
  EXPECT_EQ(MI_CAP_SYSTEM_CUDA, 0x200u);
  EXPECT_EQ(MI_CAP_CONTROL, 0x400u);
  EXPECT_EQ(MI_CAP_ACCEL, 0x800u);
  EXPECT_EQ(MI_CAP_PRELOAD, 0x1000u);
  EXPECT_EQ(MI_CAP_RELEASE, 0x2000u);
  EXPECT_EQ(MI_CAP_CERTIFICATE_DIRS, 0x4000u);
  EXPECT_EQ(MI_CAP_TE_HOST_PIN, 0x8000u);
  EXPECT_EQ(MI_CAP_PROGRESS_FRACTION, 0x10000u);
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
  EXPECT_EQ(g.steps, 0);   // unset: MI_DEFAULT_STEPS = Request::steps
  EXPECT_EQ(g.cfg, -1.0);  // unset: MI_DEFAULT_CFG = Request::cfg
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
  // Image input, hires fix, distill.
  EXPECT_EQ(g.init_image, nullptr);
  EXPECT_EQ(g.init_image_path, nullptr);
  EXPECT_EQ(g.mask, nullptr);
  EXPECT_EQ(g.mask_path, nullptr);
  EXPECT_EQ(g.mask_encoding, MI_MASK_U8);
  EXPECT_EQ(g.resize, MI_RESIZE_CROP);
  EXPECT_EQ(q.resize, mi::pipeline::ResizeMode::kCrop);
  EXPECT_EQ(g.denoise_strength, -1.0);  // unset: 1.0 = Request::denoise_strength without an init image
  EXPECT_EQ(g.hires_width, q.hires.width);
  EXPECT_EQ(g.hires_height, q.hires.height);
  EXPECT_EQ(g.hires_strength, q.hires.strength);
  EXPECT_EQ(g.hires_steps, q.hires.steps);
  EXPECT_EQ(g._reserved3, 0);
  EXPECT_EQ(g.distill_dir, nullptr);
  // Control adapter and accelerators: none, off.
  EXPECT_EQ(g.control_adapter, nullptr);
  EXPECT_FALSE(q.control.enabled());
  EXPECT_EQ(g.control_lora_strength, q.control.lora_strength);
  EXPECT_EQ(g.control_images, nullptr);
  EXPECT_EQ(g.control_images_count, q.control.images.size());
  EXPECT_EQ(g.cfg_truncation, q.accel.cfg_truncation);
  EXPECT_EQ(g.step_cache, MI_STEP_CACHE_OFF);
  EXPECT_EQ(q.accel.step_cache, mi::pipeline::accel::StepCacheKind::kOff);
  EXPECT_EQ(g._reserved4, 0);
  EXPECT_EQ(g.step_cache_threshold, 0.0);  // unset: Options::step_cache_threshold
  EXPECT_FALSE(q.accel.step_cache_threshold.has_value());

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
  EXPECT_EQ(e.time_embedding_file, nullptr);
  EXPECT_TRUE(o.time_embedding_file.empty());
  EXPECT_EQ(e.distill_dir, nullptr);
  // 1.1 fields: no extra certificate directories, the adaptive DiT load, the automatic pin policy.
  EXPECT_EQ(e.certificate_dirs, nullptr);
  EXPECT_EQ(e.certificate_dirs_count, o.certificate_dirs.size());
  EXPECT_EQ(e.dit_direct_io, MI_DIRECT_IO_AUTO);
  EXPECT_FALSE(o.dit_load_direct_io.has_value());
  EXPECT_EQ(e.te_host_pin, MI_TE_HOST_PIN_AUTO);
  EXPECT_EQ(o.te_host_pin, mi::planner::TeHostPin::kAuto);

  // The adapter turns the defaults into the pipeline's defaults.
  mi_gen_params valid = g;
  valid.prompt = "a cat";
  EXPECT_EQ(mi_gen_params_validate(&valid), MI_OK) << last_error();
  const mi::pipeline::Request r = mi::api::to_request(valid);
  EXPECT_EQ(r.sampler, q.sampler);
  EXPECT_EQ(r.steps, q.steps);  // the markers resolve to the pipeline's defaults
  EXPECT_EQ(r.cfg, q.cfg);
  EXPECT_EQ(r.denoise_strength, q.denoise_strength);
  EXPECT_EQ(r.scheduler, q.scheduler);
  EXPECT_EQ(r.noise_layout, q.noise_layout);
  EXPECT_EQ(r.noise_source, q.noise_source);
  EXPECT_EQ(r.save_metadata, q.save_metadata);
  EXPECT_EQ(r.loras.size(), 0u);
  EXPECT_EQ(r.init_image, nullptr);
  EXPECT_EQ(r.mask, nullptr);
  EXPECT_FALSE(r.hires.enabled());
  EXPECT_FALSE(r.control.enabled());
  EXPECT_EQ(r.control.lora_strength, q.control.lora_strength);
  EXPECT_TRUE(r.control.images.empty());
  EXPECT_EQ(r.accel.cfg_truncation, q.accel.cfg_truncation);
  EXPECT_EQ(r.accel.step_cache, q.accel.step_cache);
  EXPECT_EQ(r.accel.step_cache_threshold, q.accel.step_cache_threshold);  // unset <-> unset (the model's default)
  EXPECT_FALSE(r.accel.step_cache_threshold.has_value());
  const mi::pipeline::PipelineOptions ro = mi::api::to_options(e);
  EXPECT_EQ(ro.tier, o.tier);
  EXPECT_EQ(ro.attention, o.attention);
  EXPECT_EQ(ro.lora_strict, o.lora_strict);
  EXPECT_EQ(ro.block_path, o.block_path);
  EXPECT_EQ(ro.certificate_dirs, o.certificate_dirs);
  EXPECT_EQ(ro.dit_load_direct_io, o.dit_load_direct_io);
  EXPECT_EQ(ro.te_host_pin, o.te_host_pin);
  EXPECT_EQ(ro.direct_io, o.direct_io);
}

// Every C enum value maps to the pipeline value of the same name; anything else is rejected.
TEST(CApi, EnumsMapToThePipeline) {
  mi::testing::TempDir dir("c_api_enums");
  const fs::path ds = dir.path() / "dataspace.safetensors";
  write_dataspace(ds, std::vector<float>(50, 500));  // 50 step codes: the default steps fit
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
  EXPECT_EQ(g.steps, 50);  // explicit, not the unset markers of the current size
  EXPECT_EQ(g.cfg, 3.0);
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
    EXPECT_NE(last_error().find("168, 240, 320 or 376"), std::string::npos) << last_error();
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
    EXPECT_NE(last_error().find("80, 120, 136 or 160"), std::string::npos) << last_error();
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
  p.dataspace_path = file_s.c_str();  // not a dataspace file: read whenever the sampler runs
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_MODEL);
  EXPECT_NE(last_error().find("dataspace_path: "), std::string::npos) << last_error();
  const fs::path ds = dir.path() / "dataspace.safetensors";
  write_dataspace(ds, {1000, 500});
  const std::string ds_s = mi::util::path_utf8(ds);
  p.dataspace_path = ds_s.c_str();
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  EXPECT_EQ(mi::api::to_request(p).dataspace_path, ds);
  EXPECT_EQ(mi::api::to_request(p).steps, 2);  // unset steps: the file's step codes
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
  in.close();  // Windows: the rewrites below cannot replace a file that is still open
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
  // The 0.1.0 mi_timings (104 bytes) is accepted: a NULL result is then the error.
  t.struct_size = 104;
  EXPECT_EQ(mi_result_timings(nullptr, &t), MI_ERR_INVALID_ARGUMENT);
  t.struct_size = 112;
  EXPECT_EQ(mi_result_timings(nullptr, &t), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("104, 128 or 144"), std::string::npos) << last_error();
  mi_image_info image{};
  image.struct_size = sizeof image;
  EXPECT_EQ(mi_result_image(nullptr, &image), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  image.struct_size = sizeof image - 8;
  EXPECT_EQ(mi_result_image(nullptr, &image), MI_ERR_ABI_MISMATCH);
  EXPECT_EQ(mi_result_image(nullptr, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_result_text(nullptr, MI_TEXT_IMAGE_NOTE), nullptr);
  EXPECT_EQ(mi_result_text(nullptr, MI_TEXT_TIME_EMBEDDING), nullptr);
  // Control adapter and accelerators.
  mi_control_info control{};
  control.struct_size = sizeof control;
  EXPECT_EQ(mi_result_control(nullptr, &control), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  EXPECT_EQ(mi_result_control(nullptr, nullptr), MI_ERR_INVALID_ARGUMENT);
  control.struct_size = sizeof control - 8;
  EXPECT_EQ(mi_result_control(nullptr, &control), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("mi_control_info.struct_size"), std::string::npos) << last_error();
  mi_accel_info accel{};
  accel.struct_size = sizeof accel;
  EXPECT_EQ(mi_result_accel(nullptr, &accel), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_result_accel(nullptr, nullptr), MI_ERR_INVALID_ARGUMENT);
  accel.struct_size = sizeof accel + 8;
  EXPECT_EQ(mi_result_accel(nullptr, &accel), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("mi_accel_info.struct_size"), std::string::npos) << last_error();
  std::size_t count = 5;
  EXPECT_EQ(mi_result_accel_cached_steps(nullptr, &count), nullptr);
  EXPECT_EQ(count, 0u);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  EXPECT_EQ(mi_result_accel_cached_steps(nullptr, nullptr), nullptr);
  EXPECT_NE(last_error().find("count is NULL"), std::string::npos) << last_error();
  for (const mi_result_text_kind kind : {MI_TEXT_CONTROL_ADAPTER, MI_TEXT_CONTROL_KINDS, MI_TEXT_CONTROL_BASE_MODEL,
                                         MI_TEXT_CONTROL_ENCODE_SCHEDULE, MI_TEXT_ACCELERATORS}) {
    EXPECT_EQ(mi_result_text(nullptr, kind), nullptr) << kind;
  }
  EXPECT_EQ(mi_result_message_count(nullptr, MI_MESSAGE_CONTROL_WARNING), 0);
  EXPECT_EQ(mi_result_message(nullptr, MI_MESSAGE_CONTROL_NOTE, 0), nullptr);
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
      {"negative steps", [](mi_gen_params& p) { p.steps = -1; }, "positive"},  // 0 is the unset marker
      {"NaN cfg", [](mi_gen_params& p) { p.cfg = std::nan(""); }, "CFG"},       // negative is the unset marker
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
  // The quality object records the cuBLASLt copy in use (a null object when none is usable).
  const mi::backend::CublasLtStatus lt = mi::backend::cublaslt_status();
  const mi::json::Value& quality = doc.at("quality");
  EXPECT_EQ(quality.at("cublaslt").is_null(), !lt.ok);
  if (lt.ok) {
    EXPECT_EQ(quality.at("cublaslt").at("path").as_string(), lt.info.path);
  }
  EXPECT_EQ(static_cast<std::uint64_t>(quality.at("numerics").at("cublaslt").as_int()), lt.ok ? lt.info.version : 0u);
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
  // The cuBLASLt copy: the process's search result (loading it creates no CUDA context: the cpu guard checks).
  EXPECT_EQ(info._reserved1, 0);
  const mi::backend::CublasLtStatus lt = mi::backend::cublaslt_status();
  if (lt.ok) {
    EXPECT_EQ(std::string(info.cublaslt_source), lt.info.source);
    EXPECT_TRUE(lt.info.source == "system" || lt.info.source == "bundled" || lt.info.source == "env") << lt.info.source;
    EXPECT_EQ(static_cast<std::size_t>(info.cublaslt_version), lt.info.version);
    EXPECT_EQ(info.cublaslt_version / 10000, CUBLAS_VER_MAJOR);
    EXPECT_EQ(std::string(info.cublaslt_path), lt.info.path);
    EXPECT_TRUE(std::filesystem::exists(mi::util::path_from_utf8(info.cublaslt_path))) << info.cublaslt_path;
    EXPECT_STREQ(info.cublaslt_error, "");
  } else {
    EXPECT_STREQ(info.cublaslt_source, "");
    EXPECT_EQ(info.cublaslt_version, 0);
    EXPECT_STREQ(info.cublaslt_path, "");
    // The search's message names every candidate: rejected ones with the reason, missing ones by path.
    const std::string err = info.cublaslt_error;
    EXPECT_EQ(err.rfind("no usable cuBLASLt", 0), 0u) << err;
    EXPECT_TRUE(err.find("rejected:") != std::string::npos || err.find("missing (") != std::string::npos ||
                err.find("no candidates") != std::string::npos)
        << err;
  }
  std::printf("[c_api] cuBLASLt: '%s' %d %s%s\n", info.cublaslt_source, info.cublaslt_version, info.cublaslt_path,
              info.cublaslt_error);

  // The 0.2.0 size (576 bytes, before the cuBLASLt fields) is still accepted, and nothing past it is written.
  mi_probe_info prev;
  std::memset(&prev, 0x5a, sizeof prev);
  prev.struct_size = offsetof(mi_probe_info, cublaslt_source);
  ASSERT_EQ(prev.struct_size, 576u);
  ASSERT_EQ(mi_probe(&prev), MI_OK) << last_error();
  EXPECT_EQ(prev.struct_size, 576u);
  EXPECT_EQ(prev.device_count, info.device_count);
  EXPECT_EQ(prev._reserved0, 0);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&prev);
  for (std::size_t i = 576; i < sizeof prev; ++i) ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " written";
  for (const std::size_t bad : {std::size_t{568}, std::size_t{584}, sizeof prev - 8, sizeof prev + 8}) {
    mi_probe_info p{};
    p.struct_size = bad;
    EXPECT_EQ(mi_probe(&p), MI_ERR_ABI_MISMATCH) << bad;
    EXPECT_NE(last_error().find("576 or 1624"), std::string::npos) << last_error();
  }

  mi_device_info dev{};
  dev.struct_size = sizeof dev;
  EXPECT_EQ(mi_probe_device(-1, &dev), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_probe_device(info.device_count, &dev), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("out of range"), std::string::npos) << last_error();
}


// The first 0.2.0 sizes (mi_gen_params 240, mi_engine_params 120) are accepted like the 0.1.0 ones.
TEST(CApi, Earlier020StructSizes) {
  mi_gen_params g;
  std::memset(&g, 0x5a, sizeof g);
  ASSERT_EQ(mi_gen_params_init(&g, 240), MI_OK) << last_error();
  EXPECT_EQ(g.struct_size, 240u);
  EXPECT_EQ(g.save_metadata, 1);  // a 240-byte struct has the metadata on
  EXPECT_EQ(g.steps, 50);         // and explicit steps and cfg
  EXPECT_EQ(g.cfg, 3.0);
  EXPECT_EQ(g.include_prompts, 1);
  EXPECT_EQ(g._reserved2, 0);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&g);
  for (std::size_t i = 240; i < sizeof g; ++i) ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " written";
  // The tail (garbage image pointers, a nonzero _reserved3) is not read.
  g.prompt = "a cat";
  EXPECT_EQ(mi_gen_params_validate(&g), MI_OK) << last_error();
  g.struct_size = sizeof g;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);

  mi_engine_params e;
  std::memset(&e, 0x5a, sizeof e);
  ASSERT_EQ(mi_engine_params_init(&e, 120), MI_OK) << last_error();
  EXPECT_EQ(e.struct_size, 120u);
  bytes = reinterpret_cast<const unsigned char*>(&e);
  for (std::size_t i = 120; i < sizeof e; ++i) ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " written";
  mi::testing::TempDir dir("c_api_020_sizes");
  const std::string missing = mi::util::path_utf8(dir.path() / "missing");
  e.model_path = missing.c_str();
  mi_engine* engine = nullptr;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);  // the garbage file pointers are not read
  EXPECT_NE(last_error().find("not found"), std::string::npos) << last_error();
}

// The unset markers (mossinfer.h): the current size's init leaves steps, cfg and denoise_strength unset; they resolve
// to MI_DEFAULT_* without a distill bundle or an init image. Negative steps and non-finite numbers are invalid.
TEST(CApi, UnsetMarkers) {
  mi_gen_params p = gen_defaults();
  p.prompt = "a cat";
  EXPECT_EQ(p.steps, 0);
  EXPECT_EQ(p.cfg, -1.0);
  EXPECT_EQ(p.denoise_strength, -1.0);
  ASSERT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  mi::pipeline::Request q = mi::api::to_request(p);
  EXPECT_EQ(q.steps, MI_DEFAULT_STEPS);
  EXPECT_EQ(q.cfg, MI_DEFAULT_CFG);
  EXPECT_EQ(q.denoise_strength, 1.0);  // no init image
  EXPECT_EQ(MI_DEFAULT_STEPS, mi::pipeline::Request{}.steps);
  EXPECT_EQ(MI_DEFAULT_CFG, mi::pipeline::Request{}.cfg);
  p.steps = 7;
  p.cfg = 0;  // 0 is a valid scale, not the marker
  q = mi::api::to_request(p);
  EXPECT_EQ(q.steps, 7);
  EXPECT_EQ(q.cfg, 0.0);
  // The older sizes keep explicit values.
  for (const std::size_t size : {std::size_t{168}, std::size_t{240}}) {
    mi_gen_params old;
    ASSERT_EQ(mi_gen_params_init(&old, size), MI_OK);
    EXPECT_EQ(old.steps, 50) << size;
    EXPECT_EQ(old.cfg, 3.0) << size;
  }
  struct Case {
    const char* what;
    void (*edit)(mi_gen_params&);
    const char* message;
  };
  const Case cases[] = {
      {"steps -1", [](mi_gen_params& g) { g.steps = -1; }, "Steps must be positive"},
      {"cfg NaN", [](mi_gen_params& g) { g.cfg = std::nan(""); }, "CFG must be a finite number"},
      {"cfg -inf", [](mi_gen_params& g) { g.cfg = -std::numeric_limits<double>::infinity(); }, "CFG must be a finite number"},
      {"cfg +inf", [](mi_gen_params& g) { g.cfg = std::numeric_limits<double>::infinity(); }, "CFG must be a finite number"},
      {"strength NaN", [](mi_gen_params& g) { g.denoise_strength = std::nan(""); }, "Denoise strength must be a finite"},
      {"strength -inf", [](mi_gen_params& g) { g.denoise_strength = -std::numeric_limits<double>::infinity(); },
       "Denoise strength must be a finite"},
  };
  for (const Case& c : cases) {
    mi_gen_params g = gen_defaults();
    g.prompt = "a cat";
    c.edit(g);
    EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
  }
}

mi_image rgb_image(const std::vector<std::uint8_t>& px, int w, int h, int channels = 3) {
  mi_image im{};
  im.pixels = px.data();
  im.width = w;
  im.height = h;
  im.channels = channels;
  return im;
}

// Image input: pixels are copied and files decoded at generate time; validation copies and decodes nothing. At most
// one of pixels and a path per image; sizes, channels and masks are checked.
TEST(CApi, ImageInput) {
  mi::testing::TempDir dir("c_api_image_input");
  std::vector<std::uint8_t> rgb(32 * 16 * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i * 7);
  std::vector<std::uint8_t> m01(32 * 16, 0), m255(32 * 16, 0);
  for (std::size_t i = 0; i < m01.size(); i += 3) m01[i] = 1, m255[i] = 200;
  const mi_image init = rgb_image(rgb, 32, 16);
  const mi_image mask01 = rgb_image(m01, 32, 16, 1);
  const mi_image mask_u8 = rgb_image(m255, 32, 16, 1);
  const std::string png = mi::util::path_utf8(dir.path() / "init.png");
  ASSERT_EQ(mi_write_png(png.c_str(), rgb.data(), 32, 16), MI_OK) << last_error();

  // img2img from pixels: copied into the request; the default (unset) strength is 0.75 with an init image.
  mi_gen_params p = gen_defaults();
  p.prompt = "a cat";
  p.init_image = &init;
  ASSERT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  mi::pipeline::Request q = mi::api::to_request(p);
  ASSERT_NE(q.init_image, nullptr);
  EXPECT_EQ(q.init_image->width, 32);
  EXPECT_EQ(q.init_image->height, 16);
  EXPECT_EQ(q.init_image->channels, 3);
  EXPECT_EQ(q.init_image->pixels, rgb);
  EXPECT_NE(q.init_image->pixels.data(), rgb.data());
  EXPECT_EQ(q.denoise_strength, MI_DEFAULT_DENOISE_STRENGTH);
  p.denoise_strength = 1.0;  // explicit: kept (the pipeline then runs txt2img and says so)
  EXPECT_EQ(mi::api::to_request(p).denoise_strength, 1.0);
  p.denoise_strength = 0.5;
  EXPECT_EQ(mi::api::to_request(p).denoise_strength, 0.5);
  // Validation does not load the images: 1x1 stand-ins.
  q = mi::api::to_request(p, nullptr, false);
  ASSERT_NE(q.init_image, nullptr);
  EXPECT_EQ(q.init_image->width, 1);
  // The same image from a file.
  p.init_image = nullptr;
  p.init_image_path = png.c_str();
  q = mi::api::to_request(p);
  ASSERT_NE(q.init_image, nullptr);
  EXPECT_EQ(q.init_image->pixels, rgb);
  // Masks: BINARY01 scaled to 0 / 255, U8 passed as is.
  p.mask = &mask01;
  p.mask_encoding = MI_MASK_BINARY01;
  q = mi::api::to_request(p);
  ASSERT_NE(q.mask, nullptr);
  EXPECT_EQ(q.mask->channels, 1);
  for (std::size_t i = 0; i < m01.size(); ++i) ASSERT_EQ(q.mask->pixels[i], m01[i] ? 255 : 0) << i;
  p.mask = &mask_u8;
  p.mask_encoding = MI_MASK_U8;
  EXPECT_EQ(mi::api::to_request(p).mask->pixels, m255);
  // Resize modes.
  const mi::pipeline::ResizeMode modes[] = {mi::pipeline::ResizeMode::kCrop, mi::pipeline::ResizeMode::kFit,
                                            mi::pipeline::ResizeMode::kStretch};
  for (int i = 0; i < 3; ++i) {
    p.resize = static_cast<mi_resize_mode>(i);
    EXPECT_EQ(mi::api::to_request(p).resize, modes[i]) << i;
  }
  p.resize = MI_RESIZE_CROP;
  ASSERT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();

  const std::vector<std::uint8_t> m2(32 * 16, 2);
  const mi_image mask2 = rgb_image(m2, 32, 16, 1);
  const mi_image small_mask = rgb_image(m01, 16, 16, 1);
  std::vector<std::uint8_t> rgba(32 * 16 * 4);
  const mi_image four = rgb_image(rgba, 32, 16, 4);
  const std::string missing = mi::util::path_utf8(dir.path() / "missing.png");
  const std::string not_png = mi::util::path_utf8(dir.path() / "not.png");
  std::ofstream(not_png) << "not an image";
  const std::string huge = mi::util::path_utf8(dir.path() / "huge.png");
  std::ofstream(huge).close();
  fs::resize_file(huge, mi::pipeline::kMaxImageFileBytes + 1);  // sparse
  const std::string white = mi::util::path_utf8(dir.path() / "white.png");
  const std::vector<std::uint8_t> white_px(32 * 16 * 3, 255);
  ASSERT_EQ(mi_write_png(white.c_str(), white_px.data(), 32, 16), MI_OK);
  struct Case {
    const char* what;
    std::function<void(mi_gen_params&)> edit;
    const char* message;
    bool at_generate;  // found when the images load (mi_generate), not by mi_gen_params_validate
  };
  const Case cases[] = {
      {"both pixels and a path", [&](mi_gen_params& g) { g.init_image = &init; }, "both set", false},
      {"NULL pixels", [&](mi_gen_params& g) {
         static mi_image none{};
         none = init;
         none.pixels = nullptr;
         g.init_image_path = nullptr, g.init_image = &none;
       }, "init_image.pixels is NULL", false},
      {"4 channels", [&](mi_gen_params& g) { g.init_image_path = nullptr, g.init_image = &four; }, "init_image.channels must be 3", false},
      {"zero width", [&](mi_gen_params& g) {
         static mi_image zero{};
         zero = init;
         zero.width = 0;
         g.init_image_path = nullptr, g.init_image = &zero;
       }, "each side must be 1 to 16384", false},
      {"side over 16384", [&](mi_gen_params& g) {
         static mi_image wide{};
         wide = init;
         wide.width = 16385;
         g.init_image_path = nullptr, g.init_image = &wide;
       }, "16384", false},
      {"over 64 MP", [&](mi_gen_params& g) {
         static mi_image big{};
         big = init;
         big.width = 16384, big.height = 4000;  // checked before any pixel is read
         g.init_image_path = nullptr, g.init_image = &big;
       }, "64000000 pixels", false},
      {"reserved", [&](mi_gen_params& g) {
         static mi_image res{};
         res = init;
         res._reserved0 = 1;
         g.init_image_path = nullptr, g.init_image = &res;
       }, "init_image._reserved0", false},
      {"missing file", [&](mi_gen_params& g) { g.init_image_path = missing.c_str(); }, "init_image_path: file not found", false},
      {"file over 64 MiB", [&](mi_gen_params& g) { g.init_image_path = huge.c_str(); }, "bytes; at most", false},
      {"undecodable file", [&](mi_gen_params& g) { g.init_image_path = not_png.c_str(); }, "init_image_path: ", true},
      {"mask value 2", [&](mi_gen_params& g) { g.mask = &mask2, g.mask_encoding = MI_MASK_BINARY01; }, "must be 0 or 1", false},
      {"mask file value 255", [&](mi_gen_params& g) {
         g.mask = nullptr, g.mask_path = white.c_str(), g.mask_encoding = MI_MASK_BINARY01;
       }, "must be 0 or 1", true},
      {"mask encoding", [&](mi_gen_params& g) { g.mask_encoding = static_cast<mi_mask_encoding>(2); }, "mask_encoding: unknown", false},
      {"mask channels", [&](mi_gen_params& g) { g.mask = &init; }, "mask.channels must be 1", false},
      {"mask size (pixels)", [&](mi_gen_params& g) {
         g.init_image_path = nullptr, g.init_image = &init, g.mask = &small_mask;
       }, "the init image's size", false},
      {"mask size (file)", [&](mi_gen_params& g) { g.mask = &small_mask; }, "the init image's size", true},
      {"mask both", [&](mi_gen_params& g) { g.mask_path = png.c_str(); }, "mask and mask_path are both set", false},
      {"mask without image", [&](mi_gen_params& g) { g.init_image_path = nullptr, g.denoise_strength = 1; }, "needs an init image", false},
      {"strength without image", [&](mi_gen_params& g) { g.init_image_path = nullptr, g.mask = nullptr; }, "needs an init image", false},
      {"strength > 1", [&](mi_gen_params& g) { g.denoise_strength = 1.5; }, "Denoise strength", false},
      {"resize", [&](mi_gen_params& g) { g.resize = static_cast<mi_resize_mode>(3); }, "resize: unknown value 3", false},
      {"dataspace + image", [&](mi_gen_params& g) {
         static const fs::path ds = dir.path() / "ds.safetensors";
         write_dataspace(ds, {1000, 500});
         static const std::string ds_s = mi::util::path_utf8(ds);
         g.sampler = MI_SAMPLER_DATASPACE, g.dataspace_path = ds_s.c_str(), g.steps = 2;
       }, "dataspace sampler cannot start from an image", false},
      {"reserved3", [&](mi_gen_params& g) { g._reserved3 = 1; }, "mi_gen_params._reserved3", false},
  };
  for (const Case& c : cases) {
    mi_gen_params g = p;
    c.edit(g);
    if (c.at_generate) {
      EXPECT_EQ(mi_gen_params_validate(&g), MI_OK) << c.what << ": " << last_error();
      try {
        mi::api::to_request(g);
        ADD_FAILURE() << c.what << ": accepted when the images load";
      } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(c.message), std::string::npos) << c.what << ": " << e.what();
      }
      continue;
    }
    EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
  }
}

TEST(CApi, HiresFix) {
  mi_gen_params p = gen_defaults();
  p.prompt = "a cat";
  p.width = p.height = 256;
  p.hires_width = 384;
  p.hires_height = 384;
  p.hires_steps = 2;
  ASSERT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  const mi::pipeline::Request q = mi::api::to_request(p);
  EXPECT_TRUE(q.hires.enabled());
  EXPECT_EQ(q.hires.width, 384);
  EXPECT_EQ(q.hires.height, 384);
  EXPECT_EQ(q.hires.strength, 0.45);
  EXPECT_EQ(q.hires.steps, 2);
  struct Case {
    const char* what;
    void (*edit)(mi_gen_params&);
    const char* message;
  };
  const Case cases[] = {
      {"not /16", [](mi_gen_params& g) { g.hires_width = 380; }, "multiples of 16"},
      {"height 0", [](mi_gen_params& g) { g.hires_height = 0; }, "multiples of 16"},
      {"negative width", [](mi_gen_params& g) { g.hires_width = -16, g.hires_height = 0; }, "0 (off) or positive"},
      {"too large", [](mi_gen_params& g) { g.hires_width = g.hires_height = 4096; }, "4.2 megapixels"},
      {"strength", [](mi_gen_params& g) { g.hires_strength = -0.1; }, "denoise strength"},
      {"steps", [](mi_gen_params& g) { g.hires_steps = -1; }, "steps"},
  };
  for (const Case& c : cases) {
    mi_gen_params g = p;
    c.edit(g);
    EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
  }
  // Off: its strength and steps are not checked.
  mi_gen_params off = gen_defaults();
  off.prompt = "a cat";
  off.hires_strength = 5;
  off.hires_steps = -3;
  EXPECT_EQ(mi_gen_params_validate(&off), MI_OK) << last_error();
}

// Distill bundles: the unset markers take the bundle's choices; explicit values stay; malformed bundles are model errors.
TEST(CApi, DistillBundle) {
  mi::testing::TempDir dir("c_api_distill");
  const fs::path bundle = dir.path() / "dmd";
  fs::create_directories(bundle);
  write_dataspace(bundle / "dataspace.safetensors", {1000, 750, 500, 250});
  std::ofstream(bundle / "lora.safetensors") << "x";  // read at generate time only
  std::ofstream(bundle / "time_embedding.safetensors") << "x";
  const std::string bundle_s = mi::util::path_utf8(bundle);
  const fs::path user_lora = dir.path() / "style.safetensors";
  std::ofstream(user_lora) << "x";
  const std::string user_lora_s = mi::util::path_utf8(user_lora);
  const mi_lora loras[] = {{user_lora_s.c_str(), 0.5}};

  // The init defaults resolve to the bundle's choices.
  mi_gen_params p = gen_defaults();
  p.prompt = "a cat";
  p.distill_dir = bundle_s.c_str();
  p.loras = loras;
  p.loras_count = 1;
  ASSERT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  fs::path te;
  mi::pipeline::Request q = mi::api::to_request(p, &te);
  EXPECT_EQ(q.steps, 4);  // the bundle's step codes
  EXPECT_EQ(q.cfg, 1.0);
  EXPECT_EQ(q.sampler, mi::pipeline::SamplerKind::kDataSpace);
  EXPECT_EQ(q.dataspace_path, bundle / "dataspace.safetensors");
  ASSERT_EQ(q.loras.size(), 2u);
  EXPECT_EQ(q.loras[0].path, user_lora);  // the caller's first, then the bundle's at strength 1
  EXPECT_EQ(q.loras[1].path, bundle / "lora.safetensors");
  EXPECT_EQ(q.loras[1].strength, 1.0f);
  EXPECT_EQ(te, bundle / "time_embedding.safetensors");
  // Explicit steps within the bundle's codes stay; beyond them they are rejected with the count.
  p.steps = 3;
  EXPECT_EQ(mi::api::to_request(p).steps, 3);
  p.steps = 10;
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("runs 1 to 4 steps"), std::string::npos) << last_error();
  // An explicit DATASPACE without a file takes the bundle's (and EULER cannot be forced: it is the unset marker).
  p.steps = 0;
  p.sampler = MI_SAMPLER_DATASPACE;
  EXPECT_EQ(mi::api::to_request(p).dataspace_path, bundle / "dataspace.safetensors");
  // Another sampler stays, with explicit steps and cfg; the LoRA is still appended.
  p.steps = 10;
  p.cfg = 2.5;
  p.sampler = MI_SAMPLER_HEUN;
  q = mi::api::to_request(p);
  EXPECT_EQ(q.steps, 10);
  EXPECT_EQ(q.cfg, 2.5);
  EXPECT_EQ(q.sampler, mi::pipeline::SamplerKind::kHeun);
  EXPECT_TRUE(q.dataspace_path.empty());
  EXPECT_EQ(q.loras.size(), 2u);
  // The older sizes' explicit 50 steps exceed the bundle's codes.
  mi_gen_params old = gen_defaults();
  old.prompt = "a cat";
  old.distill_dir = bundle_s.c_str();
  old.steps = 50;
  EXPECT_EQ(mi_gen_params_validate(&old), MI_ERR_INVALID_ARGUMENT);
  // The bundle's dataspace sampler cannot start from an image; another sampler can.
  mi_gen_params d = gen_defaults();
  d.prompt = "a cat";
  d.distill_dir = bundle_s.c_str();
  std::vector<std::uint8_t> rgb(16 * 16 * 3, 100);
  const mi_image init = rgb_image(rgb, 16, 16);
  d.init_image = &init;
  EXPECT_EQ(mi_gen_params_validate(&d), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("distill_dir selected the bundle's dataspace sampler"), std::string::npos) << last_error();
  d.sampler = MI_SAMPLER_EULER_ANCESTRAL;
  EXPECT_EQ(mi_gen_params_validate(&d), MI_OK) << last_error();
  q = mi::api::to_request(d);
  EXPECT_EQ(q.steps, 50);  // no dataspace sampler: the default
  EXPECT_EQ(q.cfg, 1.0);
  EXPECT_EQ(q.denoise_strength, MI_DEFAULT_DENOISE_STRENGTH);
  // The LoRA cap counts the bundle's LoRA.
  const std::vector<mi_lora> many(64, mi_lora{user_lora_s.c_str(), 1.0});
  d.loras = many.data();
  d.loras_count = many.size();
  EXPECT_EQ(mi_gen_params_validate(&d), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("the distill bundle's included"), std::string::npos) << last_error();

  // The dataspace sampler without a bundle: its file's codes bound the steps too, and unset steps take their count.
  mi_gen_params ds = gen_defaults();
  ds.prompt = "a cat";
  const std::string ds_file = mi::util::path_utf8(bundle / "dataspace.safetensors");
  ds.sampler = MI_SAMPLER_DATASPACE;
  ds.dataspace_path = ds_file.c_str();
  EXPECT_EQ(mi::api::to_request(ds).steps, 4);
  ds.steps = 5;
  EXPECT_EQ(mi_gen_params_validate(&ds), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("4 step codes"), std::string::npos) << last_error();

  // Rejections: not a directory (argument), no bundle file or a malformed dataspace file (model).
  const fs::path empty = dir.path() / "empty";
  fs::create_directories(empty);
  const fs::path broken = dir.path() / "broken";
  fs::create_directories(broken);
  std::ofstream(broken / "dataspace.safetensors") << "not safetensors";
  const std::string missing_s = mi::util::path_utf8(dir.path() / "missing");
  const std::string empty_s = mi::util::path_utf8(empty);
  const std::string broken_s = mi::util::path_utf8(broken);
  mi_gen_params b = gen_defaults();
  b.prompt = "a cat";
  b.distill_dir = missing_s.c_str();
  EXPECT_EQ(mi_gen_params_validate(&b), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("distill_dir: not a directory"), std::string::npos) << last_error();
  b.distill_dir = empty_s.c_str();
  EXPECT_EQ(mi_gen_params_validate(&b), MI_ERR_MODEL);  // DistillError
  EXPECT_NE(last_error().find("holds none of"), std::string::npos) << last_error();
  b.distill_dir = broken_s.c_str();
  EXPECT_EQ(mi_gen_params_validate(&b), MI_ERR_MODEL);  // DataspaceError
  EXPECT_NE(last_error().find("distill_dir: "), std::string::npos) << last_error();

  // Engine side: the bundle's time embedding becomes the engine's; a different explicit file conflicts.
  mi_engine_params e = engine_defaults();
  e.distill_dir = bundle_s.c_str();
  EXPECT_EQ(mi::api::to_options(e).time_embedding_file, bundle / "time_embedding.safetensors");
  const std::string te_s = mi::util::path_utf8(bundle / "time_embedding.safetensors");
  e.time_embedding_file = te_s.c_str();
  EXPECT_NO_THROW(mi::api::to_options(e));
  const std::string other_s = user_lora_s;
  e.time_embedding_file = other_s.c_str();
  EXPECT_THROW(mi::api::to_options(e), std::invalid_argument);
  e = engine_defaults();
  e.time_embedding_file = other_s.c_str();
  EXPECT_EQ(mi::api::to_options(e).time_embedding_file, user_lora);
  // Through mi_engine_create (before CUDA): a missing time embedding file, a bundle without files.
  make_fake_bundle(dir.path() / "model");
  const std::string model = mi::util::path_utf8(dir.path() / "model");
  e = engine_defaults();
  e.model_path = model.c_str();
  e.time_embedding_file = missing_s.c_str();
  mi_engine* engine = nullptr;
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("time_embedding_file: file not found"), std::string::npos) << last_error();
  e.time_embedding_file = nullptr;
  e.distill_dir = empty_s.c_str();
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_MODEL);
  e.distill_dir = missing_s.c_str();
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  e.distill_dir = bundle_s.c_str();
  e.time_embedding_file = other_s.c_str();
  EXPECT_EQ(mi_engine_create(&e, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("different time embeddings"), std::string::npos) << last_error();
  EXPECT_EQ(engine, nullptr);
}

// The image fields of a synthetic pipeline result, and an INT4 DiT reported as W4A4.
TEST(CApi, ImageResultFromAPipelineResult) {
  mi::pipeline::Request q;
  q.model = "/models/MOSS-Image/base";
  q.prompt = "a cat";
  q.width = q.height = 256;
  q.hires.width = q.hires.height = 384;
  mi::pipeline::Result r;
  r.width = r.height = 384;
  r.image_mode = "inpaint";
  r.image_note = "a note";
  r.time_embedding = "/dmd/time_embedding.safetensors";
  r.first_pass = {true, 256, 256, 10, 5, 5, 0.5, true, "untiled", "untiled"};
  r.hires = {true, 384, 384, 8, 4, 4, 0.45, true, "untiled", ""};
  r.timings.vae_encoder_load_s = 0.25;
  r.timings.encode_s = 0.5;
  r.timings.image_ops_s = 0.125;
  r.tier.precision = mi::planner::Precision::kInt4;
  r.tier.effective = mi::planner::Tier::kTurbo;
  const mi::api::Generation g = mi::api::to_generation(q, r, true);
  EXPECT_EQ(g.image.struct_size, sizeof(mi_image_info));
  EXPECT_EQ(g.image.mode, MI_IMAGE_MODE_INPAINT);
  EXPECT_EQ(g.image.hires_ran, 1);
  EXPECT_EQ(g.image.first_width, 256);
  EXPECT_EQ(g.image.first_steps_scheduled, 5);
  EXPECT_EQ(g.image.first_start_step, 5);
  EXPECT_EQ(g.image.first_steps_run, 5);
  EXPECT_EQ(g.image.first_encoded, 1);
  EXPECT_EQ(g.image.first_strength, 0.5);
  EXPECT_EQ(g.image.hires_width, 384);
  EXPECT_EQ(g.image.hires_steps_scheduled, 4);
  EXPECT_EQ(g.image.hires_start_step, 4);
  EXPECT_EQ(g.image.hires_strength, 0.45);
  EXPECT_EQ(g.image_note, "a note");
  EXPECT_EQ(g.time_embedding, r.time_embedding);
  EXPECT_EQ(g.timings.vae_encoder_load_s, 0.25);
  EXPECT_EQ(g.timings.image_encode_s, 0.5);
  EXPECT_EQ(g.timings.image_ops_s, 0.125);
  EXPECT_EQ(g.tier.dit_precision, MI_PRECISION_W4A4);
  EXPECT_EQ(g.tier.effective, MI_TIER_TURBO);
  EXPECT_EQ(g.tier.certified, 0);
  // txt2img: the default pass report.
  mi::pipeline::Request t;
  t.prompt = "a cat";
  mi::pipeline::Result tr;
  tr.image_mode = "img2img";
  const mi::api::Generation tg = mi::api::to_generation(t, tr, true);
  EXPECT_EQ(tg.image.mode, MI_IMAGE_MODE_IMG2IMG);
  EXPECT_EQ(tg.image.hires_ran, 0);
  EXPECT_EQ(tg.image.hires_width, 0);
  EXPECT_EQ(tg.image.first_strength, 1.0);
}

// Control adapters and the accelerators (MI_CAP_CONTROL, MI_CAP_ACCEL): mapped into pipeline::Request, every argument
// checked without a GPU; validation neither copies nor decodes the control images. The adapter's format and kinds are
// mi_generate()'s (the pipeline's), so a stand-in file passes here.
TEST(CApi, ControlAndAccelerators) {
  mi::testing::TempDir dir("c_api_control");
  const std::string adapter = mi::util::path_utf8(dir.path() / "control.safetensors");
  std::ofstream(dir.path() / "control.safetensors") << "not read before mi_generate()";
  std::vector<std::uint8_t> rgb(32 * 16 * 3), depth(32 * 16);
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>(i * 5);
  for (std::size_t i = 0; i < depth.size(); ++i) depth[i] = static_cast<std::uint8_t>(i);
  const std::string png = mi::util::path_utf8(dir.path() / "color.png");
  ASSERT_EQ(mi_write_png(png.c_str(), rgb.data(), 32, 16), MI_OK) << last_error();
  const std::string garbage = mi::util::path_utf8(dir.path() / "garbage.png");
  std::ofstream(dir.path() / "garbage.png") << "not an image";
  const mi_image depth_px = rgb_image(depth, 32, 16, 1);

  mi_control_image images[2] = {};
  images[0].kind = "depth";
  images[0].image = &depth_px;
  images[0].strength = 0.75;
  images[0].start = 0.0;
  images[0].end = 0.5;
  images[1].kind = "color";
  images[1].image_path = png.c_str();
  images[1].strength = 2.0;
  images[1].start = 0.25;
  images[1].end = 1.0;
  mi_gen_params p = gen_defaults();
  p.prompt = "a cat";
  p.control_adapter = adapter.c_str();
  p.control_lora_strength = 0.5;
  p.control_images = images;
  p.control_images_count = 2;
  p.cfg_truncation = 0.6;
  p.step_cache = MI_STEP_CACHE_FBCACHE;
  ASSERT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  mi::pipeline::Request q = mi::api::to_request(p);
  EXPECT_TRUE(q.control.enabled());
  EXPECT_EQ(mi::util::path_utf8(q.control.adapter), adapter);
  EXPECT_EQ(q.control.lora_strength, 0.5f);
  ASSERT_EQ(q.control.images.size(), 2u);
  EXPECT_EQ(q.control.images[0].kind, "depth");
  ASSERT_NE(q.control.images[0].image, nullptr);
  EXPECT_EQ(q.control.images[0].image->channels, 1);  // grayscale pixels as given (the pipeline replicates them)
  EXPECT_EQ(q.control.images[0].image->pixels, depth);
  EXPECT_NE(q.control.images[0].image->pixels.data(), depth.data());
  EXPECT_EQ(q.control.images[0].strength, 0.75);
  EXPECT_EQ(q.control.images[0].start, 0.0);
  EXPECT_EQ(q.control.images[0].end, 0.5);
  EXPECT_EQ(q.control.images[1].kind, "color");
  ASSERT_NE(q.control.images[1].image, nullptr);
  EXPECT_EQ(q.control.images[1].image->channels, 3);  // a file is read as RGB
  EXPECT_EQ(q.control.images[1].image->width, 32);
  EXPECT_EQ(q.control.images[1].image->pixels, rgb);
  EXPECT_EQ(q.control.images[1].strength, 2.0);
  EXPECT_EQ(q.control.images[1].start, 0.25);
  EXPECT_EQ(q.accel.cfg_truncation, 0.6);
  EXPECT_EQ(q.accel.step_cache, mi::pipeline::accel::StepCacheKind::kFbCache);
  EXPECT_FALSE(q.accel.step_cache_threshold.has_value());  // 0: unset, the model's default (the pipeline's to pick)
  p.step_cache_threshold = 0.2;
  EXPECT_EQ(mi::api::to_request(p).accel.step_cache_threshold, 0.2);
  // Validation loads nothing: 1x1 stand-ins, and an undecodable file passes until mi_generate() decodes it.
  q = mi::api::to_request(p, nullptr, false);
  ASSERT_EQ(q.control.images.size(), 2u);
  EXPECT_EQ(q.control.images[0].image->width, 1);
  EXPECT_EQ(q.control.images[0].image->channels, 1);
  EXPECT_EQ(q.control.images[1].image->width, 1);
  images[1].image_path = garbage.c_str();
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  EXPECT_THROW(mi::api::to_request(p), std::invalid_argument);
  images[1].image_path = png.c_str();
  // An adapter without images is valid (every kind gets no control); a threshold without the step cache is ignored.
  p.control_images_count = 0;
  p.step_cache = MI_STEP_CACHE_OFF;
  p.step_cache_threshold = 0.5;
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  EXPECT_FALSE(mi::api::to_request(p).accel.step_cache_on());

  // Every rejection, with the C fields' names.
  struct Case {
    const char* what;
    std::function<void(mi_gen_params&, mi_control_image*)> edit;
    const char* message;
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  static const std::string missing = "/nonexistent/control.safetensors";
  const std::vector<Case> cases = {
      {"images without an adapter", [](mi_gen_params& g, mi_control_image*) { g.control_adapter = nullptr; },
       "control_images need a control_adapter"},
      {"empty adapter path", [](mi_gen_params& g, mi_control_image*) { g.control_adapter = ""; },
       "control_images need a control_adapter"},
      {"missing adapter", [](mi_gen_params& g, mi_control_image*) { g.control_adapter = missing.c_str(); },
       "control_adapter: file not found"},
      {"NULL array", [](mi_gen_params& g, mi_control_image*) { g.control_images = nullptr; }, "control_images: NULL array"},
      {"too many", [](mi_gen_params& g, mi_control_image*) { g.control_images_count = 17; }, "at most 16 entries"},
      {"no kind", [](mi_gen_params&, mi_control_image* c) { c[1].kind = nullptr; }, "control_images[1].kind is required"},
      {"empty kind", [](mi_gen_params&, mi_control_image* c) { c[0].kind = ""; }, "control_images[0].kind is required"},
      {"duplicate kind", [](mi_gen_params&, mi_control_image* c) { c[1].kind = "depth"; }, "Two control images of kind"},
      {"both image and path", [&](mi_gen_params&, mi_control_image* c) { c[1].image = &depth_px; }, "are both set"},
      {"neither image nor path", [](mi_gen_params&, mi_control_image* c) { c[1].image_path = ""; },
       "control_images[1] needs an image or an image_path"},
      {"missing image file", [](mi_gen_params&, mi_control_image* c) { c[1].image_path = "/nonexistent/x.png"; },
       "control_images[1].image_path: file not found"},
      {"4 channels", [](mi_gen_params&, mi_control_image* c) { const_cast<mi_image*>(c[0].image)->channels = 4; },
       "3 (RGB) or 1 (grayscale), not 4"},
      {"bad image size", [](mi_gen_params&, mi_control_image* c) { const_cast<mi_image*>(c[0].image)->width = 0; },
       "control_images[0].image is 0x16"},
      {"negative strength", [](mi_gen_params&, mi_control_image* c) { c[0].strength = -0.1; }, "strength must be between 0 and 10"},
      {"strength above 10", [](mi_gen_params&, mi_control_image* c) { c[0].strength = 10.5; }, "control_images[0].strength"},
      {"NaN strength", [&](mi_gen_params&, mi_control_image* c) { c[1].strength = nan; }, "control_images[1].strength"},
      {"start > end", [](mi_gen_params&, mi_control_image* c) { c[0].start = 0.6; }, "require 0 <= start <= end <= 1"},
      {"end > 1", [](mi_gen_params&, mi_control_image* c) { c[1].end = 1.5; }, "require 0 <= start <= end <= 1"},
      {"start < 0", [](mi_gen_params&, mi_control_image* c) { c[0].start = -0.5; }, "require 0 <= start <= end <= 1"},
      {"NaN end", [&](mi_gen_params&, mi_control_image* c) { c[0].end = nan; }, "require 0 <= start <= end <= 1"},
      {"infinite LoRA strength", [&](mi_gen_params& g, mi_control_image*) { g.control_lora_strength = inf; },
       "control_lora_strength must be a finite number"},
      {"LoRA strength beyond a float", [](mi_gen_params& g, mi_control_image*) { g.control_lora_strength = 1e300; },
       "control_lora_strength must be a finite number"},
      {"cfg_truncation below 0", [](mi_gen_params& g, mi_control_image*) { g.cfg_truncation = -0.1; },
       "cfg_truncation must be between 0 and 1"},
      {"cfg_truncation above 1", [](mi_gen_params& g, mi_control_image*) { g.cfg_truncation = 1.5; },
       "cfg_truncation must be between 0 and 1"},
      {"NaN cfg_truncation", [&](mi_gen_params& g, mi_control_image*) { g.cfg_truncation = nan; },
       "cfg_truncation must be between 0 and 1"},
      {"unknown step_cache", [](mi_gen_params& g, mi_control_image*) { g.step_cache = static_cast<mi_step_cache>(2); },
       "step_cache: unknown value 2"},
      {"MAX_ENUM step_cache", [](mi_gen_params& g, mi_control_image*) { g.step_cache = MI_STEP_CACHE_MAX_ENUM; },
       "step_cache: unknown value 2147483647"},
      {"negative threshold", [](mi_gen_params& g, mi_control_image*) { g.step_cache_threshold = -0.01; },
       "step_cache_threshold must be 0 (the default) or greater than 0 and at most 1"},
      {"threshold above 1", [](mi_gen_params& g, mi_control_image*) { g.step_cache_threshold = 1.01; }, "step_cache_threshold"},
      {"NaN threshold", [&](mi_gen_params& g, mi_control_image*) { g.step_cache_threshold = nan; }, "step_cache_threshold"},
      {"nonzero _reserved4", [](mi_gen_params& g, mi_control_image*) { g._reserved4 = 1; }, "mi_gen_params._reserved4"},
  };
  for (const Case& c : cases) {
    mi_image px = depth_px;
    mi_control_image ims[2] = {images[0], images[1]};
    ims[0].image = &px;
    mi_gen_params g = p;
    g.control_images = ims;
    g.control_images_count = 2;
    g.step_cache = MI_STEP_CACHE_FBCACHE;
    g.step_cache_threshold = 0;
    c.edit(g, ims);
    EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
  }
}

// The 1.0.0 mi_gen_params (320 bytes) is accepted: init fills only that prefix, with the unset markers of the current
// size; the library reads only the prefix (no control adapter, the accelerators off).
TEST(CApi, Gen100StructSize) {
  constexpr std::size_t kGen100 = 320;
  mi_gen_params g;
  std::memset(&g, 0x5a, sizeof g);
  ASSERT_EQ(mi_gen_params_init(&g, kGen100), MI_OK) << last_error();
  EXPECT_EQ(g.struct_size, kGen100);
  EXPECT_EQ(g.steps, 0);     // unset, as at the current size
  EXPECT_EQ(g.cfg, -1.0);
  EXPECT_EQ(g.denoise_strength, -1.0);
  EXPECT_EQ(g.save_metadata, 1);
  EXPECT_EQ(g.distill_dir, nullptr);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&g);
  for (std::size_t i = kGen100; i < sizeof g; ++i) ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " written";
  // The tail (garbage control pointers and enum, a nonzero _reserved4) is never read.
  g.prompt = "a cat";
  EXPECT_EQ(mi_gen_params_validate(&g), MI_OK) << last_error();
  g.width = 1000;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);  // the prefix is still checked
  g.width = 1024;
  g.struct_size = sizeof g;
  EXPECT_EQ(mi_gen_params_validate(&g), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("_reserved4"), std::string::npos) << last_error();
  // The current size's init: the control and accelerator defaults.
  std::memset(&g, 0x5a, sizeof g);
  ASSERT_EQ(mi_gen_params_init(&g, sizeof g), MI_OK);
  EXPECT_EQ(g.control_adapter, nullptr);
  EXPECT_EQ(g.control_lora_strength, 1.0);
  EXPECT_EQ(g.control_images_count, 0u);
  EXPECT_EQ(g.cfg_truncation, 1.0);
  EXPECT_EQ(g.step_cache, MI_STEP_CACHE_OFF);
  EXPECT_EQ(g._reserved4, 0);
  EXPECT_EQ(g.step_cache_threshold, 0.0);
}

// mi_result_control / mi_result_accel / mi_result_accel_cached_steps and the control texts and messages, from a
// synthetic pipeline result.
TEST(CApi, ControlAndAcceleratorResults) {
  mi::pipeline::Request q;
  q.model = "/models/MOSS-Image/base";
  q.prompt = "a cat";
  q.cfg = 3.0;
  q.control.adapter = "/ctl/moss_image_color_depth_control.safetensors";
  q.control.lora_strength = 0.8f;
  mi::pipeline::Image depth;
  depth.width = depth.height = 4;
  depth.channels = 1;
  depth.pixels.assign(16, 9);
  mi::pipeline::Request::ControlImage ci;
  ci.kind = "depth";
  ci.image = std::make_shared<const mi::pipeline::Image>(depth);
  ci.end = 0.5;
  q.control.images = {ci};
  q.accel.cfg_truncation = 0.5;
  q.accel.step_cache = mi::pipeline::accel::StepCacheKind::kFbCache;
  q.accel.step_cache_threshold = 0.05;
  mi::pipeline::Result r;
  r.width = r.height = 16;
  r.rgb.assign(16 * 16 * 3, 1);
  r.control.adapter = mi::util::path_utf8(q.control.adapter);
  r.control.kinds = {"color", "depth"};
  r.control.base_model = "moss-image-sft";
  r.control.lora.file = r.control.adapter;
  r.control.lora.adapters = 120;
  r.control.lora.side_branch = 100;
  r.control.lora.merged = 20;
  r.control.lora.unknown = {"x: y"};
  r.control.lora.warnings = {"blocks.3.attn.qkv: requantized"};
  r.control.lora.notes = {"rank 64", "alpha from the metadata"};
  r.control.encodes = 2;
  r.control.encode_schedule = "untiled";
  r.accel.cfg_truncated_steps = 3;
  r.accel.dit_calls = 6;
  r.accel.cached_calls = 2;
  r.accel.cached_steps = {4, 6};
  r.lora_warnings = {"video_embedder: LoRA delta replaced by the control adapter's embedder"};

  mi_result* res = mi::api::new_result(mi::api::to_generation(q, r, true));
  mi_control_info c;
  std::memset(&c, 0xff, sizeof c);
  c.struct_size = sizeof c;
  ASSERT_EQ(mi_result_control(res, &c), MI_OK) << last_error();
  EXPECT_EQ(c.struct_size, sizeof c);
  EXPECT_EQ(c.active, 1);
  EXPECT_EQ(c.kinds_count, 2);
  EXPECT_EQ(c.encodes, 2);
  EXPECT_EQ(c.lora_adapters, 120);
  EXPECT_EQ(c.lora_side_branch, 100);
  EXPECT_EQ(c.lora_merged, 20);
  EXPECT_EQ(c.lora_unknown_keys, 1);
  EXPECT_EQ(c.lora_warnings, 1);
  EXPECT_EQ(c.lora_notes, 2);
  EXPECT_EQ(c._reserved0, 0);  // output padding is zeroed
  c.struct_size = sizeof c - 4;
  EXPECT_EQ(mi_result_control(res, &c), MI_ERR_ABI_MISMATCH);
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_CONTROL_ADAPTER), "/ctl/moss_image_color_depth_control.safetensors");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_CONTROL_KINDS), "color,depth");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_CONTROL_BASE_MODEL), "moss-image-sft");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_CONTROL_ENCODE_SCHEDULE), "untiled");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_ACCELERATORS), "cfg_truncation 0.5, step_cache fbcache 0.05");
  ASSERT_EQ(mi_result_message_count(res, MI_MESSAGE_CONTROL_WARNING), 1);
  EXPECT_STREQ(mi_result_message(res, MI_MESSAGE_CONTROL_WARNING, 0), "blocks.3.attn.qkv: requantized");
  ASSERT_EQ(mi_result_message_count(res, MI_MESSAGE_CONTROL_NOTE), 2);
  EXPECT_STREQ(mi_result_message(res, MI_MESSAGE_CONTROL_NOTE, 1), "alpha from the metadata");
  EXPECT_EQ(mi_result_message(res, MI_MESSAGE_CONTROL_NOTE, 2), nullptr);
  EXPECT_EQ(mi_result_message_count(res, MI_MESSAGE_LORA_WARNING), 1);  // the pipeline's own, unchanged
  EXPECT_EQ(mi_result_lora_count(res), 0);  // the adapter's LoRA part is not one of the runtime LoRAs
  mi_accel_info a;
  std::memset(&a, 0xff, sizeof a);
  a.struct_size = sizeof a;
  ASSERT_EQ(mi_result_accel(res, &a), MI_OK) << last_error();
  EXPECT_EQ(a.struct_size, sizeof a);
  EXPECT_EQ(a.cfg_truncated_steps, 3);
  EXPECT_EQ(a.dit_calls, 6);
  EXPECT_EQ(a.cached_calls, 2);
  EXPECT_EQ(a.lossy, 1);
  a.struct_size = 16;
  EXPECT_EQ(mi_result_accel(res, &a), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("accepts 24 or 32"), std::string::npos) << last_error();
  std::size_t count = 0;
  const int32_t* steps = mi_result_accel_cached_steps(res, &count);
  ASSERT_EQ(count, 2u);
  ASSERT_NE(steps, nullptr);
  EXPECT_EQ(steps[0], 4);
  EXPECT_EQ(steps[1], 6);
  EXPECT_EQ(mi_result_accel_cached_steps(res, nullptr), nullptr);
  // The report carries both, as the CLI's.
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(res));
  EXPECT_EQ(doc.at("control").at("kinds").as_array().size(), 2u);
  EXPECT_EQ(doc.at("accelerators").at("lossy").as_bool(), true);
  mi_result_free(res);

  // Neither: all zero, "" and "off"; no cached steps.
  mi::pipeline::Request plain;
  plain.prompt = "a cat";
  mi::pipeline::Result pr;
  res = mi::api::new_result(mi::api::to_generation(plain, pr, true));
  std::memset(&c, 0xff, sizeof c);
  c.struct_size = sizeof c;
  ASSERT_EQ(mi_result_control(res, &c), MI_OK);
  EXPECT_EQ(c.active, 0);
  EXPECT_EQ(c.kinds_count, 0);
  EXPECT_EQ(c.encodes, 0);
  EXPECT_EQ(c.lora_adapters, 0);
  EXPECT_EQ(c._reserved0, 0);
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_CONTROL_ADAPTER), "");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_CONTROL_KINDS), "");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_ACCELERATORS), "off");
  EXPECT_EQ(mi_result_message_count(res, MI_MESSAGE_CONTROL_WARNING), 0);
  std::memset(&a, 0xff, sizeof a);
  a.struct_size = sizeof a;
  ASSERT_EQ(mi_result_accel(res, &a), MI_OK);
  EXPECT_EQ(a.cfg_truncated_steps, 0);
  EXPECT_EQ(a.dit_calls, 0);
  EXPECT_EQ(a.cached_calls, 0);
  EXPECT_EQ(a.lossy, 0);
  count = 9;
  EXPECT_EQ(mi_result_accel_cached_steps(res, &count), nullptr);
  EXPECT_EQ(count, 0u);
  // CFG truncation without CFG changes nothing: not lossy.
  plain.cfg = 1.0;
  plain.accel.cfg_truncation = 0.5;
  mi_result_free(res);
  res = mi::api::new_result(mi::api::to_generation(plain, pr, true));
  ASSERT_EQ(mi_result_accel(res, &a), MI_OK);
  EXPECT_EQ(a.lossy, 0);
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_ACCELERATORS), "off");
  mi_result_free(res);
}

// mi_tier_info.certified is about the tier (1 on REFERENCE, whatever else ran); run_certified is the run's, the
// report's models.dit.certified. The 1.0.0 size (32 bytes) gets its 32 bytes.
TEST(CApi, TierRunCertified) {
  const auto tier_of = [](const mi::pipeline::Request& q, mi::pipeline::Result r) {
    mi_result* res = mi::api::new_result(mi::api::to_generation(q, r, true));
    mi_tier_info t;
    std::memset(&t, 0xff, sizeof t);
    t.struct_size = sizeof t;
    EXPECT_EQ(mi_result_tier(res, &t), MI_OK) << last_error();
    const mi::json::Value doc = mi::json::parse(mi_result_report_json(res));
    EXPECT_EQ(doc.at("models").at("dit").at("certified").as_bool(), t.run_certified == 1);
    EXPECT_EQ(t._reserved0, 0);
    mi_result_free(res);
    return t;
  };
  mi::pipeline::Request q;
  q.prompt = "a cat";
  // REFERENCE with a LoRA: the pipeline marks the run uncertified.
  q.loras = {{"/loras/style.safetensors", 1.0f}};
  mi::pipeline::Result r;
  r.tier.requested = mi::planner::Tier::kAuto;
  r.tier.effective = mi::planner::Tier::kReference;
  r.tier.attention = mi::planner::AttentionMode::kExact;
  r.tier.certified = false;
  r.tier.reason = "auto: Reference; runtime LoRAs active: uncertified (certificates cover the base model only)";
  mi::lora::FileReport f;
  f.file = "/loras/style.safetensors";
  r.loras = {f};
  mi_tier_info t = tier_of(q, r);
  EXPECT_EQ(t.effective, MI_TIER_REFERENCE);
  EXPECT_EQ(t.certified, 1);
  EXPECT_EQ(t.run_certified, 0);
  // HIGH, certified, nothing else: both 1.
  mi::pipeline::Request plain;
  plain.prompt = "a cat";
  mi::pipeline::Result h;
  h.tier.effective = mi::planner::Tier::kHigh;
  h.tier.precision = mi::planner::Precision::kInt8;
  h.tier.attention = mi::planner::AttentionMode::kQ8Fp8;
  h.tier.certified = true;
  h.tier.certificate = "/models/moss_int8.mossinfer-cert.json";
  t = tier_of(plain, h);
  EXPECT_EQ(t.effective, MI_TIER_HIGH);
  EXPECT_EQ(t.certified, 1);
  EXPECT_EQ(t.run_certified, 1);
  // REFERENCE, nothing else: both 1. HIGH uncertified (e.g. accelerators): both 0.
  mi::pipeline::Result ref;
  ref.tier.effective = mi::planner::Tier::kReference;
  ref.tier.certified = true;
  t = tier_of(plain, ref);
  EXPECT_EQ(t.certified, 1);
  EXPECT_EQ(t.run_certified, 1);
  h.tier.certified = false;
  t = tier_of(plain, h);
  EXPECT_EQ(t.certified, 0);
  EXPECT_EQ(t.run_certified, 0);
  // The 1.0.0 mi_tier_info: 32 bytes written, run_certified untouched; any other size is a mismatch.
  mi_result* res = mi::api::new_result(mi::api::to_generation(plain, ref, true));
  alignas(mi_tier_info) unsigned char buf[sizeof(mi_tier_info)];
  std::memset(buf, 0x5a, sizeof buf);
  const std::size_t old = 32;
  std::memcpy(buf, &old, sizeof old);
  ASSERT_EQ(mi_result_tier(res, reinterpret_cast<mi_tier_info*>(buf)), MI_OK) << last_error();
  for (std::size_t i = old; i < sizeof buf; ++i) ASSERT_EQ(buf[i], 0x5a) << "byte " << i << " written";
  mi_tier_info bad{};
  bad.struct_size = 36;
  EXPECT_EQ(mi_result_tier(res, &bad), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("32 or 40"), std::string::npos) << last_error();
  mi_result_free(res);
}

// The 1.1 engine fields: certificate_dirs (copied, in order), dit_direct_io and te_host_pin map to PipelineOptions;
// bad values fail mi_engine_create() before CUDA.
TEST(CApi, EngineParams110Fields) {
  mi_engine_params e = engine_defaults();
  const std::string umlaut = "/certs/\xc3\xa4rger";  // UTF-8 "/certs/ärger"
  const char* dirs[] = {"/certs/first", umlaut.c_str(), "relative/dir"};
  e.certificate_dirs = dirs;
  e.certificate_dirs_count = 3;
  mi::pipeline::PipelineOptions o = mi::api::to_options(e);
  ASSERT_EQ(o.certificate_dirs.size(), 3u);
  EXPECT_EQ(mi::util::path_utf8(o.certificate_dirs[0]), "/certs/first");
  EXPECT_EQ(mi::util::path_utf8(o.certificate_dirs[1]), umlaut);
  EXPECT_EQ(mi::util::path_utf8(o.certificate_dirs[2]), "relative/dir");
  e.certificate_dirs_count = 0;  // a count of 0 reads nothing
  e.certificate_dirs = nullptr;
  EXPECT_TRUE(mi::api::to_options(e).certificate_dirs.empty());

  const std::pair<mi_direct_io, std::optional<bool>> io[] = {
      {MI_DIRECT_IO_AUTO, std::nullopt}, {MI_DIRECT_IO_BUFFERED, false}, {MI_DIRECT_IO_DIRECT, true}};
  for (const auto& [c, want] : io) {
    e.dit_direct_io = c;
    EXPECT_EQ(mi::api::to_options(e).dit_load_direct_io, want) << c;
    // direct_io stays its own field (it forces direct reads everywhere; the engine applies the precedence).
    e.direct_io = 1;
    EXPECT_TRUE(mi::api::to_options(e).direct_io);
    EXPECT_EQ(mi::api::to_options(e).dit_load_direct_io, want) << c;
    e.direct_io = 0;
  }
  e.dit_direct_io = MI_DIRECT_IO_AUTO;
  const std::pair<mi_te_host_pin, mi::planner::TeHostPin> pins[] = {{MI_TE_HOST_PIN_AUTO, mi::planner::TeHostPin::kAuto},
                                                                    {MI_TE_HOST_PIN_ON, mi::planner::TeHostPin::kOn},
                                                                    {MI_TE_HOST_PIN_OFF, mi::planner::TeHostPin::kOff}};
  for (const auto& [c, want] : pins) {
    e.te_host_pin = c;
    EXPECT_EQ(mi::api::to_options(e).te_host_pin, want) << c;
  }

  // Rejections, through mi_engine_create() on a bundle without weights (checked before any CUDA work).
  mi::testing::TempDir dir("c_api_engine_110");
  make_fake_bundle(dir.path() / "bundle");
  const std::string bundle = mi::util::path_utf8(dir.path() / "bundle");
  const char* with_empty[] = {"/certs/a", ""};
  const char* with_null[] = {nullptr};
  std::vector<std::string> many(17, "/certs/x");
  std::vector<const char*> many_ptrs;
  for (const std::string& s : many) many_ptrs.push_back(s.c_str());
  struct Case {
    const char* what;
    std::function<void(mi_engine_params&)> edit;
    const char* message;
  };
  const std::vector<Case> cases = {
      {"NULL array", [](mi_engine_params& p) { p.certificate_dirs_count = 1; }, "certificate_dirs: NULL array"},
      {"empty dir", [&](mi_engine_params& p) { p.certificate_dirs = with_empty; p.certificate_dirs_count = 2; },
       "certificate_dirs[1] is empty"},
      {"NULL dir", [&](mi_engine_params& p) { p.certificate_dirs = with_null; p.certificate_dirs_count = 1; },
       "certificate_dirs[0] is empty"},
      {"17 dirs", [&](mi_engine_params& p) { p.certificate_dirs = many_ptrs.data(); p.certificate_dirs_count = 17; },
       "certificate_dirs: at most 16 entries (got 17)"},
      {"dit_direct_io 3", [](mi_engine_params& p) { p.dit_direct_io = static_cast<mi_direct_io>(3); },
       "dit_direct_io: unknown value 3"},
      {"dit_direct_io MAX_ENUM", [](mi_engine_params& p) { p.dit_direct_io = MI_DIRECT_IO_MAX_ENUM; },
       "dit_direct_io: unknown value 2147483647"},
      {"te_host_pin 3", [](mi_engine_params& p) { p.te_host_pin = static_cast<mi_te_host_pin>(3); },
       "te_host_pin: unknown value 3"},
      {"te_host_pin MAX_ENUM", [](mi_engine_params& p) { p.te_host_pin = MI_TE_HOST_PIN_MAX_ENUM; },
       "te_host_pin: unknown value 2147483647"},
  };
  for (const Case& c : cases) {
    mi_engine_params p = engine_defaults();
    p.model_path = bundle.c_str();
    c.edit(p);
    mi_engine* engine = nullptr;
    EXPECT_EQ(mi_engine_create(&p, &engine), MI_ERR_INVALID_ARGUMENT) << c.what;
    EXPECT_NE(last_error().find(c.message), std::string::npos) << c.what << ": " << last_error();
    EXPECT_EQ(engine, nullptr);
  }
  // 16 directories are accepted.
  e.certificate_dirs = many_ptrs.data();
  e.certificate_dirs_count = 16;
  EXPECT_EQ(mi::api::to_options(e).certificate_dirs.size(), 16u);

  // The 1.0.0 mi_engine_params (136 bytes): init fills only that prefix; create reads only the prefix, so the garbage
  // tail (an unknown dit_direct_io, a count without an array) is never seen and the next check is reached.
  constexpr std::size_t kEngine100 = 136;
  mi_engine_params old;
  std::memset(&old, 0x5a, sizeof old);
  ASSERT_EQ(mi_engine_params_init(&old, kEngine100), MI_OK) << last_error();
  EXPECT_EQ(old.struct_size, kEngine100);
  EXPECT_EQ(old.distill_dir, nullptr);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&old);
  for (std::size_t i = kEngine100; i < sizeof old; ++i) ASSERT_EQ(bytes[i], 0x5a) << "byte " << i << " written";
  const std::string missing = mi::util::path_utf8(dir.path() / "missing");
  old.model_path = missing.c_str();
  mi_engine* engine = nullptr;
  EXPECT_EQ(mi_engine_create(&old, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("not found"), std::string::npos) << last_error();
  old.model_path = bundle.c_str();
  old.struct_size = sizeof old;  // the current size reads the tail: its garbage is rejected
  EXPECT_EQ(mi_engine_create(&old, &engine), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(engine, nullptr);
  old.struct_size = kEngine100 + 8;
  EXPECT_EQ(mi_engine_create(&old, &engine), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("80, 120, 136 or 160"), std::string::npos) << last_error();
}

// mi_timings, mi_memory_stats and mi_accel_info grew in 1.1: the current sizes get every field, the earlier sizes (128,
// 56 bytes of 1.0.0; 24 of the first 1.1.0 header) get exactly their prefix, any other size is a mismatch.
TEST(CApi, GrownOutputStructs110) {
  mi::pipeline::Request q;
  q.prompt = "a cat";
  q.accel.step_cache = mi::pipeline::accel::StepCacheKind::kFbCache;  // threshold unset: the model's default
  mi::pipeline::Result r;
  r.timings.lora_load_s = 0.25;
  r.timings.lora_merge_s = 0.125;
  r.timings.image_ops_s = 0.5;
  r.memory.host_private_peak = 7;
  r.memory.host_copy_bytes = std::uint64_t{20} << 30;
  r.memory.merge_memo_bytes = std::uint64_t{1} << 29;
  r.memory.dit_loads = 1;
  r.accel.dit_calls = 4;
  r.accel.cached_calls = 1;
  r.accel.step_cache_threshold = 0.02;
  r.accel.step_cache_threshold_source = "default for the base model";
  mi_result* res = mi::api::new_result(mi::api::to_generation(q, r, true));

  mi_timings t;
  std::memset(&t, 0xff, sizeof t);
  t.struct_size = sizeof t;
  ASSERT_EQ(mi_result_timings(res, &t), MI_OK) << last_error();
  EXPECT_EQ(t.lora_load_s, 0.25);
  EXPECT_EQ(t.lora_merge_s, 0.125);
  EXPECT_EQ(t.image_ops_s, 0.5);
  mi_memory_stats m;
  std::memset(&m, 0xff, sizeof m);
  m.struct_size = sizeof m;
  ASSERT_EQ(mi_result_memory(res, &m), MI_OK) << last_error();
  EXPECT_EQ(m.host_private_peak_bytes, 7u);
  EXPECT_EQ(m.host_copy_bytes, std::uint64_t{20} << 30);
  EXPECT_EQ(m.merge_memo_bytes, std::uint64_t{1} << 29);
  EXPECT_EQ(m.dit_loads, 1);
  EXPECT_EQ(m._reserved0, 0);  // output padding is zeroed
  mi_accel_info a;
  std::memset(&a, 0xff, sizeof a);
  a.struct_size = sizeof a;
  ASSERT_EQ(mi_result_accel(res, &a), MI_OK) << last_error();
  EXPECT_EQ(a.dit_calls, 4);
  EXPECT_EQ(a.cached_calls, 1);
  EXPECT_EQ(a.lossy, 1);
  EXPECT_EQ(a.step_cache_threshold, 0.02);  // what ran: the model's default the pipeline filled in
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE), "default for the base model");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_ACCELERATORS), "step_cache fbcache 0.02");

  // The earlier sizes: their bytes only.
  const auto old_size = [&](auto& out, std::size_t size, auto accessor, const char* what) {
    using T = std::remove_reference_t<decltype(out)>;
    alignas(T) unsigned char buf[sizeof(T)];
    std::memset(buf, 0x5a, sizeof buf);
    std::memcpy(buf, &size, sizeof size);
    ASSERT_EQ(accessor(res, reinterpret_cast<T*>(buf)), MI_OK) << what << ": " << last_error();
    std::size_t got = 0;
    std::memcpy(&got, buf, sizeof got);
    EXPECT_EQ(got, size) << what;
    for (std::size_t i = size; i < sizeof buf; ++i) ASSERT_EQ(buf[i], 0x5a) << what << ": byte " << i << " written";
    std::memcpy(&out, buf, sizeof buf);
  };
  old_size(t, 128, mi_result_timings, "mi_timings");
  EXPECT_EQ(t.image_ops_s, 0.5);
  old_size(m, 56, mi_result_memory, "mi_memory_stats");
  EXPECT_EQ(m.host_private_peak_bytes, 7u);
  old_size(a, 24, mi_result_accel, "mi_accel_info");
  EXPECT_EQ(a.lossy, 1);
  m.struct_size = 64;
  EXPECT_EQ(mi_result_memory(res, &m), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("56 or 80"), std::string::npos) << last_error();
  a.struct_size = 28;
  EXPECT_EQ(mi_result_accel(res, &a), MI_ERR_ABI_MISMATCH);
  t.struct_size = 136;
  EXPECT_EQ(mi_result_timings(res, &t), MI_ERR_ABI_MISMATCH);
  mi_result_free(res);

  // Step cache off: threshold 0, no source; a threshold the request set is the request's.
  mi::pipeline::Request plain;
  plain.prompt = "a cat";
  mi::pipeline::Result pr;
  res = mi::api::new_result(mi::api::to_generation(plain, pr, true));
  a.struct_size = sizeof a;
  ASSERT_EQ(mi_result_accel(res, &a), MI_OK);
  EXPECT_EQ(a.step_cache_threshold, 0.0);
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE), "");
  mi_result_free(res);
  // Cancelled before the pipeline resolved the unset threshold: described as "default".
  mi::pipeline::Result early;
  early.cancelled = true;
  res = mi::api::new_result(mi::api::to_generation(q, early, true));
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_ACCELERATORS), "step_cache fbcache default");
  mi_result_free(res);
}

// mi_result_dit_load, the notes, the accelerator warnings and the step cache's diffs, from synthetic pipeline results.
TEST(CApi, DitLoadNotesAndDiffs) {
  mi::pipeline::Request q;
  q.prompt = "a cat";
  q.accel.step_cache = mi::pipeline::accel::StepCacheKind::kFbCache;
  q.accel.step_cache_threshold = 0.05;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  mi::pipeline::Result r;
  r.dit_load.source = "files";
  r.dit_load.io = "buffered";
  r.dit_load.note = "buffered: 97 % of the files is in the page cache";
  r.dit_load.bytes = 12345678901ull;
  r.dit_load.seconds = 2.5;
  r.te_note = "pinned host copy not allocated: streamed from the files";
  r.preload_note = "the DiT was not loaded: the plan parks it for the text encoder";
  r.accel_warnings = {"step cache: fbcache 0.05 moves the base model's composition", "step cache never engaged"};
  r.accel.step_cache_diffs = {nan, 0.1, 0.04};
  r.accel.step_cache_threshold = 0.05;
  r.accel.step_cache_threshold_source = "request";
  mi_result* res = mi::api::new_result(mi::api::to_generation(q, r, true));
  mi_dit_load_info d;
  std::memset(&d, 0xff, sizeof d);
  d.struct_size = sizeof d;
  ASSERT_EQ(mi_result_dit_load(res, &d), MI_OK) << last_error();
  EXPECT_EQ(d.struct_size, sizeof d);
  EXPECT_EQ(d.source, MI_DIT_LOAD_SOURCE_FILES);
  EXPECT_EQ(d.io, MI_LOAD_IO_BUFFERED);
  EXPECT_EQ(d.bytes, 12345678901ull);
  EXPECT_EQ(d.seconds, 2.5);
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_DIT_LOAD_NOTE), "buffered: 97 % of the files is in the page cache");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_TE_NOTE), "pinned host copy not allocated: streamed from the files");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_PRELOAD_NOTE), "the DiT was not loaded: the plan parks it for the text encoder");
  EXPECT_STREQ(mi_result_text(res, MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE), "request");
  EXPECT_EQ(mi_result_text(res, MI_TEXT_MAX_ENUM), nullptr);
  EXPECT_EQ(mi_result_text(res, static_cast<mi_result_text_kind>(18)), nullptr);
  ASSERT_EQ(mi_result_message_count(res, MI_MESSAGE_ACCEL_WARNING), 2);
  EXPECT_STREQ(mi_result_message(res, MI_MESSAGE_ACCEL_WARNING, 1), "step cache never engaged");
  EXPECT_EQ(mi_result_message(res, MI_MESSAGE_ACCEL_WARNING, 2), nullptr);
  EXPECT_EQ(mi_result_message_count(res, static_cast<mi_message_kind>(5)), 0);
  std::size_t count = 0;
  const double* diffs = mi_result_accel_step_cache_diffs(res, &count);
  ASSERT_EQ(count, 3u);
  ASSERT_NE(diffs, nullptr);
  EXPECT_TRUE(std::isnan(diffs[0]));
  EXPECT_EQ(diffs[1], 0.1);
  EXPECT_EQ(diffs[2], 0.04);
  EXPECT_EQ(mi_result_accel_step_cache_diffs(res, nullptr), nullptr);
  EXPECT_NE(last_error().find("count is NULL"), std::string::npos) << last_error();
  count = 9;
  EXPECT_EQ(mi_result_accel_step_cache_diffs(nullptr, &count), nullptr);
  EXPECT_EQ(count, 0u);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  d.struct_size = sizeof d - 8;
  EXPECT_EQ(mi_result_dit_load(res, &d), MI_ERR_ABI_MISMATCH);
  EXPECT_NE(last_error().find("mi_dit_load_info.struct_size"), std::string::npos) << last_error();
  EXPECT_EQ(mi_result_dit_load(res, nullptr), MI_ERR_INVALID_ARGUMENT);
  d.struct_size = sizeof d;
  EXPECT_EQ(mi_result_dit_load(nullptr, &d), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("result is NULL"), std::string::npos) << last_error();
  mi_result_free(res);

  // From the host copy (no I/O mode), direct reads, and no load at all.
  const std::tuple<const char*, const char*, mi_dit_load_source, mi_load_io> loads[] = {
      {"host_copy", "", MI_DIT_LOAD_SOURCE_HOST_COPY, MI_LOAD_IO_NONE},
      {"files", "direct", MI_DIT_LOAD_SOURCE_FILES, MI_LOAD_IO_DIRECT},
      {"", "", MI_DIT_LOAD_SOURCE_NONE, MI_LOAD_IO_NONE}};
  for (const auto& [source, io, want_source, want_io] : loads) {
    mi::pipeline::Result l;
    l.dit_load.source = source;
    l.dit_load.io = io;
    l.dit_load.bytes = *source != '\0' ? 100 : 0;
    res = mi::api::new_result(mi::api::to_generation(q, l, true));
    std::memset(&d, 0xff, sizeof d);
    d.struct_size = sizeof d;
    ASSERT_EQ(mi_result_dit_load(res, &d), MI_OK);
    EXPECT_EQ(d.source, want_source) << source;
    EXPECT_EQ(d.io, want_io) << source;
    EXPECT_EQ(d.bytes, *source != '\0' ? 100u : 0u);
    EXPECT_STREQ(mi_result_text(res, MI_TEXT_DIT_LOAD_NOTE), "");
    EXPECT_STREQ(mi_result_text(res, MI_TEXT_PRELOAD_NOTE), "");
    count = 9;
    EXPECT_EQ(mi_result_accel_step_cache_diffs(res, &count), nullptr);
    EXPECT_EQ(count, 0u);
    mi_result_free(res);
  }
}

// mi_engine_preload's request: the prompt may be NULL or blank (kept blank for Pipeline::preload), everything else is
// checked as for mi_generate(); mi_gen_params_validate() still requires a prompt. Preload and release on a NULL engine.
TEST(CApi, PreloadAndReleaseWithoutAnEngine) {
  mi_gen_params p = gen_defaults();
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);  // NULL prompt
  EXPECT_NE(last_error().find("Enter a prompt."), std::string::npos) << last_error();
  EXPECT_THROW(mi::api::to_request(p), std::invalid_argument);
  mi::pipeline::Request q = mi::api::to_request(p, nullptr, true, true);
  EXPECT_EQ(q.prompt, "");
  EXPECT_EQ(q.steps, MI_DEFAULT_STEPS);
  EXPECT_EQ(q.width, 1024);
  p.prompt = "  \t ";
  EXPECT_EQ(mi_gen_params_validate(&p), MI_ERR_INVALID_ARGUMENT);
  q = mi::api::to_request(p, nullptr, true, true);
  EXPECT_EQ(q.prompt, "  \t ");  // Pipeline::preload plans a blank prompt as a short one
  p.prompt = "a long prompt that sizes the workspaces";
  EXPECT_EQ(mi::api::to_request(p, nullptr, true, true).prompt, p.prompt);
  // The rest of the request is checked: size, steps, LoRAs, the accelerators.
  p.prompt = nullptr;
  p.width = 1000;
  try {
    mi::api::to_request(p, nullptr, true, true);
    ADD_FAILURE() << "a bad size was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("multiples of 16"), std::string::npos) << e.what();
  }
  p.width = 512;
  p.step_cache_threshold = 2.0;
  EXPECT_THROW(mi::api::to_request(p, nullptr, true, true), std::invalid_argument);
  p.step_cache_threshold = 0;
  const mi_lora missing_lora{"/nonexistent/lora.safetensors", 1.0};
  p.loras = &missing_lora;
  p.loras_count = 1;
  EXPECT_THROW(mi::api::to_request(p, nullptr, true, true), std::invalid_argument);

  mi_result* r = reinterpret_cast<mi_result*>(&p);  // must be reset to NULL
  EXPECT_EQ(mi_engine_preload(nullptr, nullptr, &r), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(r, nullptr);
  EXPECT_NE(last_error().find("engine is NULL"), std::string::npos) << last_error();
  EXPECT_EQ(mi_engine_preload(nullptr, &p, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_engine_release(nullptr, 1, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("engine is NULL"), std::string::npos) << last_error();
  mi_release_info info{};
  info.struct_size = sizeof info;
  EXPECT_EQ(mi_engine_release(nullptr, 0, &info), MI_ERR_INVALID_ARGUMENT);
}

// The release report in C terms: every field, and the note cut at 511 bytes without splitting a UTF-8 sequence.
TEST(CApi, ReleaseInfo) {
  mi::pipeline::ReleaseReport rr;
  rr.seconds = 3.5;
  rr.vram_released = std::size_t{18} << 30;
  rr.dit_host_copy = true;
  rr.te_host_copy = false;
  rr.host_copy_bytes = std::uint64_t{19} << 30;
  rr.host_copy_s = 3.25;
  rr.note = "text encoder: no host copy (streamed from its files)";
  mi_release_info info = mi::api::to_release_info(rr);
  EXPECT_EQ(info.struct_size, sizeof info);
  EXPECT_EQ(info.seconds, 3.5);
  EXPECT_EQ(info.vram_released_bytes, std::uint64_t{18} << 30);
  EXPECT_EQ(info.host_copy_bytes, std::uint64_t{19} << 30);
  EXPECT_EQ(info.host_copy_seconds, 3.25);
  EXPECT_EQ(info.dit_host_copy, 1);
  EXPECT_EQ(info.te_host_copy, 0);
  EXPECT_STREQ(info.note, "text encoder: no host copy (streamed from its files)");
  // 510 ASCII bytes, then a 2-byte sequence across the 511-byte limit: cut before it.
  rr.note = std::string(510, 'a') + "\xc3\xa4" + "tail";
  info = mi::api::to_release_info(rr);
  EXPECT_EQ(std::strlen(info.note), 510u);
  // 509 bytes, then a 2-byte sequence that fits exactly.
  rr.note = std::string(509, 'a') + "\xc3\xa4" + "tail";
  info = mi::api::to_release_info(rr);
  EXPECT_EQ(std::strlen(info.note), 511u);
  EXPECT_EQ(std::string(info.note).substr(509), "\xc3\xa4");
  rr.note.clear();
  EXPECT_STREQ(mi::api::to_release_info(rr).note, "");
  // Cancelled before it started (ReleaseReport::cancelled): mi_engine_release() maps it to MI_CANCELLED. A cancelled
  // host copy (a note, no flag) is not that case.
  bool cancelled = true;
  (void)mi::api::to_release_info(rr, &cancelled);
  EXPECT_FALSE(cancelled);
  mi::pipeline::ReleaseReport early;
  early.cancelled = true;
  early.note = "cancelled before it started";
  early.dit_host_copy = true;
  early.host_copy_bytes = 123;
  info = mi::api::to_release_info(early, &cancelled);
  EXPECT_TRUE(cancelled);
  EXPECT_EQ(info.vram_released_bytes, 0u);
  EXPECT_EQ(info.dit_host_copy, 1);  // the copies already held, as they are
  EXPECT_EQ(info.host_copy_bytes, 123u);
  EXPECT_STREQ(info.note, "cancelled before it started");
  mi::pipeline::ReleaseReport copy_cancelled;
  copy_cancelled.vram_released = 1;
  copy_cancelled.note = "the DiT's host copy was cancelled";
  (void)mi::api::to_release_info(copy_cancelled, &cancelled);
  EXPECT_FALSE(cancelled);
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

// MI_CAP_PROGRESS_FRACTION: every callback has the 56-byte struct, the fraction in [0, 1) and never decreasing, and a
// known eta (the plan comes before the first phase), >= 0.
void expect_whole_job_progress(const std::vector<mi_progress>& events) {
  ASSERT_FALSE(events.empty());
  std::string trace;
  for (std::size_t i = 0; i < events.size(); ++i) {
    const mi_progress& e = events[i];
    EXPECT_EQ(e.struct_size, std::size_t{56});
    EXPECT_GE(e.fraction, 0.0) << i;
    EXPECT_LT(e.fraction, 1.0) << i;
    if (i > 0) {
      EXPECT_GE(e.fraction, events[i - 1].fraction) << i;
    }
    EXPECT_GE(e.eta_seconds, 0.0) << i;
    char buf[64];
    std::snprintf(buf, sizeof buf, " %.3f/%.2fs", e.fraction, e.eta_seconds);
    trace += buf;
  }
  std::printf("[c_api] progress fraction/eta:%s\n", trace.c_str());
}

// A generation's bar moves: from the first event (TEXT_ENCODER) to the last (the final DECODE) the fraction grows past
// one half and the eta shrinks.
void expect_bar_moves(const std::vector<mi_progress>& events) {
  ASSERT_GE(events.size(), 2u);
  const mi_progress& first = events.front();
  const mi_progress& last = events.back();
  EXPECT_EQ(first.phase, MI_PHASE_TEXT_ENCODER);
  EXPECT_EQ(last.phase, MI_PHASE_DECODE);
  EXPECT_GT(last.fraction, first.fraction);
  EXPECT_GT(last.fraction, 0.5);
  EXPECT_LT(last.eta_seconds, first.eta_seconds);
}

// The DIT_LOAD phase's estimate in a generation's events: the eta at its start less the eta at the denoise's start.
double dit_load_estimate(const std::vector<mi_progress>& events) {
  double at_load = -1, at_denoise = -1;
  for (const mi_progress& e : events) {
    if (e.phase == MI_PHASE_DIT_LOAD) at_load = e.eta_seconds;
    if (e.phase == MI_PHASE_DENOISE && e.step == 0 && at_denoise < 0) at_denoise = e.eta_seconds;
  }
  EXPECT_GE(at_load, 0.0);
  EXPECT_GE(at_denoise, 0.0);
  return at_load - at_denoise;
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
  expect_whole_job_progress(ev.progress);
  expect_bar_moves(ev.progress);

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
    EXPECT_EQ(tier.run_certified, 1);  // no LoRA, control adapter or accelerator
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

// Progress events of a generation: the last total_steps seen.
void on_total(const mi_progress* p, void* user) { *static_cast<int*>(user) = p->total_steps; }

std::vector<std::uint8_t> gradient(int w, int h) {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h * 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      std::uint8_t* q = &px[(static_cast<std::size_t>(y) * w + x) * 3];
      q[0] = static_cast<std::uint8_t>(x);
      q[1] = static_cast<std::uint8_t>(y);
      q[2] = static_cast<std::uint8_t>(255 - x);
    }
  }
  return px;
}

// img2img from pixels: half of a 4-step schedule from a generated image.
TEST_F(GpuCApi, ImageToImageFromPixels) {
  const std::vector<std::uint8_t> px = gradient(256, 256);
  mi_image init{};
  init.pixels = px.data();
  init.width = init.height = 256;
  init.channels = 3;
  int total = 0;
  mi_gen_params p = small_request(4);
  p.init_image = &init;
  p.denoise_strength = 0.5;
  p.progress_callback = on_total;
  p.user_data = &total;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  mi_image_info info;
  std::memset(&info, 0xab, sizeof info);
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_result_image(r, &info), MI_OK) << last_error();
  EXPECT_EQ(info.mode, MI_IMAGE_MODE_IMG2IMG);
  EXPECT_EQ(info.first_width, 256);
  EXPECT_EQ(info.first_encoded, 1);
  EXPECT_EQ(info.first_start_step, 2);
  EXPECT_EQ(info.first_steps_scheduled, 2);
  EXPECT_EQ(info.first_steps_run, 2);
  EXPECT_EQ(info.first_strength, 0.5);
  EXPECT_EQ(info.hires_ran, 0);
  EXPECT_EQ(info._reserved0, 0);
  EXPECT_EQ(mi_result_steps_scheduled(r), 2);
  EXPECT_EQ(total, 2);  // pipeline::schedule_steps: the strength's skipped steps are not counted
  EXPECT_NE(mi_result_rgb(r, nullptr), nullptr);
  EXPECT_STREQ(mi_result_text(r, MI_TEXT_IMAGE_NOTE), "");
  mi_timings t{};
  t.struct_size = sizeof t;
  ASSERT_EQ(mi_result_timings(r, &t), MI_OK);
  EXPECT_GT(t.image_encode_s, 0.0);
  // The 0.1.0 mi_timings gets its 104 bytes only.
  struct {
    mi_timings t;
    unsigned char guard[8];
  } old;
  std::memset(&old, 0x5a, sizeof old);
  old.t.struct_size = 104;
  ASSERT_EQ(mi_result_timings(r, &old.t), MI_OK) << last_error();
  EXPECT_EQ(old.t.struct_size, 104u);
  EXPECT_GT(old.t.total_s, 0.0);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&old.t);
  for (std::size_t i = 104; i < sizeof old.t; ++i) ASSERT_EQ(bytes[i], 0x5a) << i;
  std::printf("[c_api] img2img: %d of %d steps from step %d, encode %.3f s\n", info.first_steps_run,
              info.first_steps_scheduled + info.first_start_step, info.first_start_step, t.image_encode_s);
  mi_result_free(r);

  // The default (unset) strength with an init image is 0.75: img2img.
  p.denoise_strength = -1;
  r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  ASSERT_EQ(mi_result_image(r, &info), MI_OK);
  EXPECT_EQ(info.mode, MI_IMAGE_MODE_IMG2IMG);
  EXPECT_EQ(info.first_strength, MI_DEFAULT_DENOISE_STRENGTH);
  EXPECT_EQ(info.first_start_step, 1);  // int(4 - 4 * 0.75)
  mi_result_free(r);
  // An explicit 1.0 without a mask is txt2img, and the note says why.
  p.denoise_strength = 1.0;
  r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  ASSERT_EQ(mi_result_image(r, &info), MI_OK);
  EXPECT_EQ(info.mode, MI_IMAGE_MODE_TXT2IMG);
  EXPECT_STRNE(mi_result_text(r, MI_TEXT_IMAGE_NOTE), "");
  std::printf("[c_api] strength 1 note: %s\n", mi_result_text(r, MI_TEXT_IMAGE_NOTE));
  mi_result_free(r);
}

// Inpainting with a BINARY01 mask (regenerate the left half).
TEST_F(GpuCApi, InpaintBinary01Mask) {
  const std::vector<std::uint8_t> px = gradient(256, 256);
  std::vector<std::uint8_t> m(256 * 256);
  for (std::size_t i = 0; i < m.size(); ++i) m[i] = (i % 256) < 128 ? 1 : 0;
  mi_image init{}, mask{};
  init.pixels = px.data();
  init.width = init.height = 256;
  init.channels = 3;
  mask.pixels = m.data();
  mask.width = mask.height = 256;
  mask.channels = 1;
  mi_gen_params p = small_request(4);
  p.init_image = &init;
  p.mask = &mask;
  p.mask_encoding = MI_MASK_BINARY01;
  p.denoise_strength = 0.75;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  mi_image_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_result_image(r, &info), MI_OK);
  EXPECT_EQ(info.mode, MI_IMAGE_MODE_INPAINT);
  EXPECT_EQ(info.first_steps_run, info.first_steps_scheduled);
  // The kept right half stays near the init image (its VAE round trip), the left half is regenerated.
  std::size_t bytes = 0;
  const std::uint8_t* rgb = mi_result_rgb(r, &bytes);
  ASSERT_EQ(bytes, px.size());
  double kept = 0, regen = 0;
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      for (int c = 0; c < 3; ++c) {
        const std::size_t i = (static_cast<std::size_t>(y) * 256 + x) * 3 + c;
        (x < 128 ? regen : kept) += std::abs(static_cast<int>(rgb[i]) - static_cast<int>(px[i]));
      }
    }
  }
  kept /= 128.0 * 256 * 3;
  regen /= 128.0 * 256 * 3;
  std::printf("[c_api] inpaint: mean |diff| kept half %.1f, regenerated half %.1f\n", kept, regen);
  EXPECT_LT(kept, regen);
  mi_result_free(r);
}

// The hires fix through the C API: 256 -> 384, 2 + 2 steps.
TEST_F(GpuCApi, HiresFix) {
  int total = 0;
  mi_gen_params p = small_request(2);
  p.hires_width = p.hires_height = 384;
  p.hires_steps = 2;
  p.progress_callback = on_total;
  p.user_data = &total;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  EXPECT_EQ(mi_result_width(r), 384);
  EXPECT_EQ(mi_result_height(r), 384);
  mi_image_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_result_image(r, &info), MI_OK);
  EXPECT_EQ(info.mode, MI_IMAGE_MODE_TXT2IMG);
  EXPECT_EQ(info.hires_ran, 1);
  EXPECT_EQ(info.first_width, 256);
  EXPECT_EQ(info.first_steps_run, 2);
  EXPECT_EQ(info.hires_width, 384);
  EXPECT_EQ(info.hires_height, 384);
  EXPECT_EQ(info.hires_strength, 0.45);
  EXPECT_EQ(info.hires_steps_run, info.hires_steps_scheduled);
  EXPECT_EQ(mi_result_steps_scheduled(r), info.first_steps_scheduled + info.hires_steps_scheduled);
  EXPECT_EQ(total, mi_result_steps_scheduled(r));
  std::size_t bytes = 0;
  EXPECT_NE(mi_result_rgb(r, &bytes), nullptr);
  EXPECT_EQ(bytes, std::size_t{384 * 384 * 3});
  std::printf("[c_api] hires: first pass %d steps, hires pass %d of %d steps (from step %d)\n", info.first_steps_run,
              info.hires_steps_run, info.hires_steps_scheduled + info.hires_start_step, info.hires_start_step);
  mi_result_free(r);
}

// Community model files for the GPU tests: $MOSS_COMMUNITY_MODELS, default /workspace/Models/community.
fs::path community_models() {
  const char* env = std::getenv("MOSS_COMMUNITY_MODELS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/community");
}
fs::path distill_bundle_dir() { return community_models() / "lodestones" / "MOSS-Image-SFT-8-steps-dmd"; }
fs::path int4_dit_file() { return community_models() / "toxicdog" / "moss_image_sft_int4_convrot.safetensors"; }

// A request's distill bundle with a time embedding needs it on the engine: refused before any work otherwise.
TEST_F(GpuCApi, DistillTimeEmbeddingNeedsTheEngine) {
  if (!fs::is_regular_file(distill_bundle_dir() / "time_embedding.safetensors")) GTEST_SKIP() << "no distill bundle";
  const std::string dir = mi::util::path_utf8(distill_bundle_dir());
  mi_gen_params p = small_request(0);
  p.distill_dir = dir.c_str();
  mi_result* r = nullptr;
  EXPECT_EQ(mi_generate(engine_, &p, &r), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("mi_engine_params.distill_dir"), std::string::npos) << last_error();
  EXPECT_EQ(r, nullptr);
}

// The MOSS-Image color/depth control adapter (ComfyUI's repackaging of darask0/MOSS-Image-color-depth-control).
fs::path control_adapter_file() { return community_models() / "comfy-org" / "moss_image_color_depth_control.safetensors"; }

// A control adapter through the C API: a depth-like grayscale image from pixels, its report and texts.
TEST_F(GpuCApi, ControlAdapterDepth) {
  if (!fs::is_regular_file(control_adapter_file())) {
    GTEST_SKIP() << "no control adapter (" << mi::util::path_utf8(control_adapter_file()) << "; set MOSS_COMMUNITY_MODELS)";
  }
  const std::string adapter = mi::util::path_utf8(control_adapter_file());
  // Near = white: a bright disc in front of a darker ramp.
  std::vector<std::uint8_t> depth(256 * 256);
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      const int dx = x - 128, dy = y - 150;
      depth[static_cast<std::size_t>(y) * 256 + x] =
          static_cast<std::uint8_t>(dx * dx + dy * dy < 70 * 70 ? 230 : 40 + y / 4);
    }
  }
  const mi_image depth_px = rgb_image(depth, 256, 256, 1);
  mi_control_image image{};
  image.kind = "depth";
  image.image = &depth_px;
  image.strength = 1.0;
  image.end = 1.0;
  mi_gen_params p = small_request(3);
  p.control_adapter = adapter.c_str();
  p.control_images = &image;
  p.control_images_count = 1;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  mi_control_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_result_control(r, &info), MI_OK) << last_error();
  EXPECT_EQ(info.active, 1);
  EXPECT_GE(info.kinds_count, 1);
  EXPECT_GT(info.encodes, 0);
  EXPECT_GT(info.lora_adapters, 0);
  EXPECT_NE(std::string(mi_result_text(r, MI_TEXT_CONTROL_KINDS)).find("depth"), std::string::npos);
  EXPECT_EQ(std::string(mi_result_text(r, MI_TEXT_CONTROL_ADAPTER)), adapter);
  EXPECT_EQ(mi_result_lora_count(r), 0);  // the adapter's LoRA part is reported with the control
  // A control adapter is a LoRA: the run is uncertified (mi_tier_info.certified describes the tier: 1 on Reference).
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  EXPECT_FALSE(doc.at("models").at("dit").at("certified").as_bool());
  mi_tier_info tier{};
  tier.struct_size = sizeof tier;
  ASSERT_EQ(mi_result_tier(r, &tier), MI_OK);
  EXPECT_EQ(tier.run_certified, 0);
  EXPECT_NE(std::string(mi_result_text(r, MI_TEXT_TIER_REASON)).find("uncertified"), std::string::npos);
  std::size_t bytes = 0;
  EXPECT_NE(mi_result_rgb(r, &bytes), nullptr);
  EXPECT_EQ(bytes, std::size_t{256 * 256 * 3});
  std::printf("[c_api] control: kinds %s, %d encodes (%s), LoRA %d adapters (%d side branch, %d merged), base_model %s\n",
              mi_result_text(r, MI_TEXT_CONTROL_KINDS), info.encodes, mi_result_text(r, MI_TEXT_CONTROL_ENCODE_SCHEDULE),
              info.lora_adapters, info.lora_side_branch, info.lora_merged, mi_result_text(r, MI_TEXT_CONTROL_BASE_MODEL));
  mi_result_free(r);
  // A kind the adapter does not have is refused by mi_generate() (validation cannot know the adapter's kinds).
  image.kind = "canny";
  EXPECT_EQ(mi_gen_params_validate(&p), MI_OK) << last_error();
  EXPECT_EQ(mi_generate(engine_, &p, &r), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(last_error().find("has no \"canny\" input"), std::string::npos) << last_error();
  EXPECT_EQ(r, nullptr);
}

// CFG truncation and the FBCache step cache with a permissive threshold: what they skipped, and uncertified.
TEST_F(GpuCApi, Accelerators) {
  mi_gen_params p = small_request(6);
  p.cfg = 3.0;
  p.cfg_truncation = 0.5;
  p.step_cache = MI_STEP_CACHE_FBCACHE;
  p.step_cache_threshold = 0.5;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  mi_accel_info a{};
  a.struct_size = sizeof a;
  ASSERT_EQ(mi_result_accel(r, &a), MI_OK) << last_error();
  EXPECT_GT(a.cfg_truncated_steps, 0);
  EXPECT_GT(a.dit_calls, 0);
  EXPECT_LE(a.cached_calls, a.dit_calls);
  EXPECT_EQ(a.lossy, 1);
  std::size_t count = 0;
  const int32_t* steps = mi_result_accel_cached_steps(r, &count);
  EXPECT_EQ(count == 0, steps == nullptr);
  EXPECT_LE(count, static_cast<std::size_t>(a.cached_calls));
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  EXPECT_FALSE(doc.at("models").at("dit").at("certified").as_bool());
  EXPECT_TRUE(doc.at("accelerators").at("lossy").as_bool());
  mi_tier_info tier{};
  tier.struct_size = sizeof tier;
  ASSERT_EQ(mi_result_tier(r, &tier), MI_OK);
  EXPECT_EQ(tier.run_certified, 0);
  EXPECT_NE(std::string(mi_result_text(r, MI_TEXT_TIER_REASON)).find("lossy accelerators"), std::string::npos);
  std::string list;
  for (std::size_t i = 0; i < count; ++i) list += (i ? "," : "") + std::to_string(steps[i]);
  std::printf("[c_api] accel (%s): CFG truncated on %d steps, step cache reused %d of %d DiT calls (steps %s)\n",
              mi_result_text(r, MI_TEXT_ACCELERATORS), a.cfg_truncated_steps, a.cached_calls, a.dit_calls, list.c_str());
  // The threshold that ran and where it came from; one measured change per DiT call.
  EXPECT_EQ(a.step_cache_threshold, 0.5);
  EXPECT_STREQ(mi_result_text(r, MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE), "request");
  std::size_t diffs = 0;
  EXPECT_NE(mi_result_accel_step_cache_diffs(r, &diffs), nullptr);
  EXPECT_EQ(diffs, static_cast<std::size_t>(a.dit_calls));
  mi_result_free(r);

  // 0 = unset: the engine's default for the base model.
  p.step_cache_threshold = 0;
  r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  a.struct_size = sizeof a;
  ASSERT_EQ(mi_result_accel(r, &a), MI_OK);
  EXPECT_EQ(a.step_cache_threshold, mi::pipeline::accel::kBaseModelFbCacheThreshold);
  EXPECT_STREQ(mi_result_text(r, MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE), "default for the base model");
  EXPECT_NE(std::string(mi_result_text(r, MI_TEXT_ACCELERATORS)).find("step_cache fbcache 0.02"), std::string::npos)
      << mi_result_text(r, MI_TEXT_ACCELERATORS);
  std::printf("[c_api] accel, threshold unset: %s (%s)\n", mi_result_text(r, MI_TEXT_ACCELERATORS),
              mi_result_text(r, MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE));
  mi_result_free(r);
}

// One engine of its own per test (other models): created, used once and freed.
class GpuCApiModels : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string why;
    if (!have_gpu_and_weights(why)) GTEST_SKIP() << why;
    if (!fs::is_directory(weights_root() / "sft" / "transformer")) GTEST_SKIP() << "no sft bundle";
    model_ = mi::util::path_utf8(weights_root() / "sft");
  }
  void TearDown() override { mi_engine_free(engine_); }
  std::string model_;
  mi_engine* engine_ = nullptr;
};

// The SFT distill bundle on the engine and the request: 8 data-space steps at CFG 1, its LoRA and time embedding.
TEST_F(GpuCApiModels, DistillBundle) {
  if (!fs::is_regular_file(distill_bundle_dir() / "time_embedding.safetensors")) GTEST_SKIP() << "no distill bundle";
  const std::string dir = mi::util::path_utf8(distill_bundle_dir());
  mi_engine_params e = engine_defaults();
  e.model_path = model_.c_str();
  e.distill_dir = dir.c_str();
  ASSERT_EQ(mi_engine_create(&e, &engine_), MI_OK) << last_error();
  mi_gen_params p;
  ASSERT_EQ(mi_gen_params_init(&p, sizeof p), MI_OK);
  p.prompt = "a red apple on a wooden table";
  p.width = p.height = 256;
  p.steps = 0;  // the bundle's
  p.cfg = -1;   // the bundle's (1)
  p.distill_dir = dir.c_str();
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  EXPECT_EQ(doc.at("request").at("sampler").as_string(), "dataspace");
  EXPECT_EQ(doc.at("request").at("cfg").as_double(), 1.0);
  EXPECT_EQ(mi_result_lora_count(r), 1);
  EXPECT_STRNE(mi_result_text(r, MI_TEXT_TIME_EMBEDDING), "");
  std::printf("[c_api] distill: %d steps, time embedding %s, LoRA %s\n", mi_result_steps_run(r),
              mi_result_text(r, MI_TEXT_TIME_EMBEDDING), mi_result_lora_file(r, 0));
  EXPECT_GT(mi_result_steps_run(r), 0);
  mi_result_free(r);
}

// An INT4 DiT: tier AUTO refuses it without a certificate; tier TURBO runs it, reported as W4A4.
TEST_F(GpuCApiModels, TurboInt4) {
  if (!fs::is_regular_file(int4_dit_file())) GTEST_SKIP() << "no INT4 DiT file";
  const std::string dit = mi::util::path_utf8(int4_dit_file());
  mi_engine_params e = engine_defaults();
  e.model_path = model_.c_str();
  e.dit_file = dit.c_str();
  ASSERT_EQ(mi_engine_create(&e, &engine_), MI_OK) << last_error();
  mi_gen_params p;
  ASSERT_EQ(mi_gen_params_init(&p, sizeof p), MI_OK);
  p.prompt = "a red apple on a wooden table";
  p.width = p.height = 256;
  p.steps = 2;
  mi_result* r = nullptr;
  const mi_status auto_status = mi_generate(engine_, &p, &r);
  std::printf("[c_api] INT4 + tier auto: %s: %s\n", mi_status_string(auto_status),
              auto_status == MI_OK ? "certified" : mi_last_error_message());
  // Refused unless a certificate covers the INT4 path (next to the DiT; the test sets no cache directory).
  fs::path cert = int4_dit_file();
  cert += ".mossinfer-cert.json";
  if (!fs::exists(cert)) {
    EXPECT_EQ(auto_status, MI_ERR_INVALID_ARGUMENT);
    EXPECT_NE(last_error().find("choose tier turbo"), std::string::npos) << last_error();
    EXPECT_EQ(r, nullptr);
  }
  mi_result_free(r);
  r = nullptr;
  mi_engine_free(engine_);
  engine_ = nullptr;
  e.tier = MI_TIER_TURBO;
  ASSERT_EQ(mi_engine_create(&e, &engine_), MI_OK) << last_error();
  ASSERT_EQ(mi_generate(engine_, &p, &r), MI_OK) << last_error();
  mi_tier_info tier{};
  tier.struct_size = sizeof tier;
  ASSERT_EQ(mi_result_tier(r, &tier), MI_OK);
  EXPECT_EQ(tier.requested, MI_TIER_TURBO);
  EXPECT_EQ(tier.effective, MI_TIER_TURBO);
  EXPECT_EQ(tier.dit_precision, MI_PRECISION_W4A4);
  std::printf("[c_api] turbo: attention %d, certified %d: %s\n", tier.attention, tier.certified,
              mi_result_text(r, MI_TEXT_TIER_REASON));
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

// mi_engine_preload / mi_engine_release, each test with engines of its own (created one at a time, so that a fresh
// engine is the reference): 256^2, 2 steps.
class GpuCApiPreload : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string why;
    if (!have_gpu_and_weights(why)) GTEST_SKIP() << why;
    model_ = mi::util::path_utf8(weights_root() / "base");
  }
  mi_engine* create() {
    mi_engine_params p = engine_defaults();
    p.model_path = model_.c_str();
    mi_engine* e = nullptr;
    EXPECT_EQ(mi_engine_create(&p, &e), MI_OK) << last_error();
    return e;
  }
  static mi_gen_params request() {
    mi_gen_params p = gen_defaults();
    p.prompt = "a red apple on a wooden table";
    p.width = p.height = 256;
    p.steps = 2;
    p.seed = 11;
    return p;
  }
  struct Generated {
    std::vector<std::uint8_t> rgb;
    mi_memory_stats memory{};
    mi_dit_load_info dit_load{};
    std::vector<mi_progress> progress;
  };
  static Generated generate(mi_engine* e) {
    mi_gen_params p = request();
    mi_result* r = nullptr;
    Generated g;
    p.progress_callback = [](const mi_progress* pr, void* user) {
      static_cast<std::vector<mi_progress>*>(user)->push_back(*pr);
    };
    p.user_data = &g.progress;
    EXPECT_EQ(mi_generate(e, &p, &r), MI_OK) << last_error();
    if (r == nullptr) return g;
    std::size_t n = 0;
    const std::uint8_t* rgb = mi_result_rgb(r, &n);
    if (rgb != nullptr) g.rgb.assign(rgb, rgb + n);
    g.memory.struct_size = sizeof g.memory;
    EXPECT_EQ(mi_result_memory(r, &g.memory), MI_OK);
    g.dit_load.struct_size = sizeof g.dit_load;
    EXPECT_EQ(mi_result_dit_load(r, &g.dit_load), MI_OK);
    mi_result_free(r);
    return g;
  }
  std::string model_;
};

struct PreloadEvents {
  int progress = 0, plans = 0;
  std::vector<mi_progress> events;
  std::vector<int> order;  // -1 for the plan, else the progress phase, in arrival order
  mi_engine* cancel_engine = nullptr;  // cancelled from the progress callback when set
};

void on_preload_progress(const mi_progress* p, void* user) {
  auto& ev = *static_cast<PreloadEvents*>(user);
  ++ev.progress;
  ev.events.push_back(*p);
  ev.order.push_back(p->phase);
  EXPECT_EQ(p->step, 0);
  EXPECT_EQ(p->total_steps, 0);
  EXPECT_GE(p->elapsed_seconds, 0.0);
  if (ev.cancel_engine != nullptr) {
    EXPECT_EQ(mi_cancel(ev.cancel_engine), MI_OK);
  }
}

void on_preload_plan(const char* explanation, const char* plan_json, void* user) {
  auto& ev = *static_cast<PreloadEvents*>(user);
  ++ev.plans;
  ev.order.push_back(-1);
  EXPECT_GT(std::strlen(explanation), 0u);
  EXPECT_NO_THROW(mi::json::parse(plan_json)) << plan_json;
}

// The preload's callbacks of a txt2img request: the plan first, then the load steps in the pipeline's order, a
// host-resident text encoder's copy (TEXT_ENCODER) before the DiT (DIT_LOAD, unless the plan leaves it to the
// generation) and the VAE decoder (DECODE) after it; never DENOISE.
void expect_preload_order(const PreloadEvents& ev, bool dit_loaded) {
  ASSERT_FALSE(ev.order.empty());
  EXPECT_EQ(ev.order.front(), -1) << "the plan comes before the first load step";
  std::string seq;
  int rank = -1;
  bool dit = false;
  for (std::size_t i = 1; i < ev.order.size(); ++i) {
    const int ph = ev.order[i];
    seq += std::to_string(ph) + " ";
    EXPECT_NE(ph, -1) << "one plan";
    EXPECT_NE(ph, MI_PHASE_DENOISE);
    const int r = ph == MI_PHASE_TEXT_ENCODER ? 0 : ph == MI_PHASE_DIT_LOAD ? 1 : 2;  // no init image: DECODE is the decoder
    EXPECT_GT(r, rank) << "phases out of order: " << seq;
    rank = r;
    dit = dit || ph == MI_PHASE_DIT_LOAD;
  }
  EXPECT_EQ(dit, dit_loaded) << seq;
  std::printf("[c_api] preload callbacks: plan, then phases %s\n", seq.c_str());
}

// A preload loads what the generation needs: the generate that follows loads no DiT and gives the bits of a fresh
// engine without the preload.
TEST_F(GpuCApiPreload, PreloadThenGenerate) {
  mi_engine* e = create();
  ASSERT_NE(e, nullptr);
  PreloadEvents ev;
  mi_gen_params p = request();
  p.progress_callback = on_preload_progress;
  p.plan_callback = on_preload_plan;
  p.user_data = &ev;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_engine_preload(e, &p, &r), MI_OK) << last_error();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(ev.plans, 1);
  EXPECT_GE(ev.progress, 1);
  expect_preload_order(ev, std::strlen(mi_result_text(r, MI_TEXT_PRELOAD_NOTE)) == 0);
  expect_whole_job_progress(ev.events);  // over the load steps
  EXPECT_EQ(mi_result_cancelled(r), 0);
  EXPECT_EQ(mi_result_steps_run(r), 0);
  EXPECT_EQ(mi_result_rgb(r, nullptr), nullptr);
  EXPECT_GT(std::strlen(mi_result_plan_explanation(r)), 0u);
  mi_dit_load_info d{};
  d.struct_size = sizeof d;
  ASSERT_EQ(mi_result_dit_load(r, &d), MI_OK);
  mi_memory_stats m{};
  m.struct_size = sizeof m;
  ASSERT_EQ(mi_result_memory(r, &m), MI_OK);
  const std::string note = mi_result_text(r, MI_TEXT_PRELOAD_NOTE);
  std::printf("[c_api] preload: dit_loads %d, source %d io %d, %.2f GB in %.2f s; note \"%s\"; dit_load note \"%s\"\n",
              m.dit_loads, d.source, d.io, static_cast<double>(d.bytes) / 1e9, d.seconds, note.c_str(),
              mi_result_text(r, MI_TEXT_DIT_LOAD_NOTE));
  if (note.empty()) {
    EXPECT_EQ(m.dit_loads, 1);
    EXPECT_EQ(d.source, MI_DIT_LOAD_SOURCE_FILES);
    EXPECT_NE(d.io, MI_LOAD_IO_NONE);
    EXPECT_GT(d.bytes, 0u);
  } else {
    std::printf("[c_api] SKIPPED checks: the preload left the DiT to the generation (\"%s\"): its dit_loads 1 / source "
                "FILES, and the generation's dit_loads 0 / source NONE, are not checked\n",
                note.c_str());
  }
  mi_result_free(r);
  // out_result NULL is allowed: a second preload of the same request finds everything loaded.
  ASSERT_EQ(mi_engine_preload(e, &p, nullptr), MI_OK) << last_error();
  const Generated after = generate(e);
  expect_whole_job_progress(after.progress);
  expect_bar_moves(after.progress);
  const double preloaded_dit_s = dit_load_estimate(after.progress);
  if (note.empty()) {
    EXPECT_EQ(after.memory.dit_loads, 0);
    EXPECT_EQ(after.dit_load.source, MI_DIT_LOAD_SOURCE_NONE);
    // The preloaded DiT is known to be loaded: the eta does not jump at DIT_LOAD.
    EXPECT_LT(preloaded_dit_s, 0.5);
  }
  mi_engine_free(e);

  e = create();
  ASSERT_NE(e, nullptr);
  const Generated fresh = generate(e);
  expect_bar_moves(fresh.progress);
  // A fresh engine prices its DiT's cold load; the preloaded one did not.
  const double fresh_dit_s = dit_load_estimate(fresh.progress);
  std::printf("[c_api] DiT load estimate: %.3f s preloaded, %.3f s on a fresh engine\n", preloaded_dit_s, fresh_dit_s);
  if (note.empty()) {
    EXPECT_GT(fresh_dit_s, preloaded_dit_s);
  }
  EXPECT_EQ(fresh.memory.dit_loads, 1);
  EXPECT_EQ(fresh.dit_load.source, MI_DIT_LOAD_SOURCE_FILES);
  ASSERT_FALSE(fresh.rgb.empty());
  EXPECT_TRUE(after.rgb == fresh.rgb) << "the preloaded engine's image differs from a fresh engine's";
  mi_engine_free(e);
}

// release(keep_host_copy = 1): the next load reads the host copy (bit-identical); release(0): the files again. A blank
// prompt preloads. A release while a generation runs is MI_ERR_BUSY.
TEST_F(GpuCApiPreload, ReleaseThenGenerate) {
  mi_engine* e = create();
  ASSERT_NE(e, nullptr);
  mi_gen_params blank = request();
  blank.prompt = "";
  ASSERT_EQ(mi_engine_preload(e, &blank, nullptr), MI_OK) << last_error();
  const Generated first = generate(e);
  ASSERT_FALSE(first.rgb.empty());

  mi_release_info info;
  std::memset(&info, 0xff, sizeof info);
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_engine_release(e, 1, &info), MI_OK) << last_error();
  std::printf("[c_api] release(keep): %.2f s, %.2f GB VRAM released, host copy %.2f GB in %.2f s (dit %d te %d) \"%s\"\n",
              info.seconds, static_cast<double>(info.vram_released_bytes) / 1e9,
              static_cast<double>(info.host_copy_bytes) / 1e9, info.host_copy_seconds, info.dit_host_copy,
              info.te_host_copy, info.note);
  EXPECT_EQ(info.struct_size, sizeof info);
  EXPECT_GT(info.vram_released_bytes, 0u);
  EXPECT_GE(info.seconds, info.host_copy_seconds);
  EXPECT_LT(std::strlen(info.note), sizeof info.note);
  const Generated from_copy = generate(e);
  EXPECT_TRUE(from_copy.rgb == first.rgb) << "the image after a release differs";
  EXPECT_EQ(from_copy.memory.dit_loads, 1);
  if (info.dit_host_copy == 1) {
    EXPECT_GT(info.host_copy_bytes, 0u);
    EXPECT_EQ(from_copy.dit_load.source, MI_DIT_LOAD_SOURCE_HOST_COPY);
    EXPECT_EQ(from_copy.dit_load.io, MI_LOAD_IO_NONE);
  } else {
    // The host's free memory decides; the release says why it kept nothing.
    EXPECT_NE(std::string(info.note).find("not kept in RAM"), std::string::npos) << info.note;
    EXPECT_EQ(from_copy.dit_load.source, MI_DIT_LOAD_SOURCE_FILES);
    std::printf("[c_api] SKIPPED checks: no DiT host copy on this machine now (\"%s\"): the host-copy load is not "
                "checked\n",
                info.note);
  }

  info.struct_size = sizeof info;
  ASSERT_EQ(mi_engine_release(e, 0, &info), MI_OK) << last_error();
  EXPECT_EQ(info.dit_host_copy, 0);
  EXPECT_EQ(info.te_host_copy, 0);
  EXPECT_EQ(info.host_copy_bytes, 0u);
  ASSERT_EQ(mi_engine_release(e, 0, nullptr), MI_OK) << last_error();  // nothing left: still fine
  const Generated from_files = generate(e);
  EXPECT_EQ(from_files.dit_load.source, MI_DIT_LOAD_SOURCE_FILES);
  EXPECT_NE(from_files.dit_load.io, MI_LOAD_IO_NONE);
  EXPECT_TRUE(from_files.rgb == first.rgb);
  info.struct_size = sizeof info - 8;
  EXPECT_EQ(mi_engine_release(e, 0, &info), MI_ERR_ABI_MISMATCH);

  // Busy: release and preload from another thread while a generation runs.
  struct Busy {
    mi_engine* engine;
    mi_status release = MI_OK, preload = MI_OK;
    bool checked = false;
  } busy{e};
  mi_gen_params p = request();
  p.steps = 4;
  p.user_data = &busy;
  p.progress_callback = [](const mi_progress* pr, void* user) {
    auto& b = *static_cast<Busy*>(user);
    if (b.checked || pr->phase != MI_PHASE_DENOISE) return;
    b.checked = true;
    std::thread([&b] {
      b.release = mi_engine_release(b.engine, 1, nullptr);
      mi_gen_params q = request();
      b.preload = mi_engine_preload(b.engine, &q, nullptr);
    }).join();
  };
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(e, &p, &r), MI_OK) << last_error();
  mi_result_free(r);
  EXPECT_TRUE(busy.checked);
  EXPECT_EQ(busy.release, MI_ERR_BUSY);
  EXPECT_EQ(busy.preload, MI_ERR_BUSY);
  mi_engine_free(e);
}

// A cancelled preload returns MI_CANCELLED with a result, and the engine stays usable.
TEST_F(GpuCApiPreload, CancelDuringPreload) {
  mi_engine* e = create();
  ASSERT_NE(e, nullptr);
  PreloadEvents ev;
  ev.cancel_engine = e;  // cancelled from the first load step's callback: before that step loads anything
  mi_gen_params p = request();
  p.progress_callback = on_preload_progress;
  p.plan_callback = on_preload_plan;
  p.user_data = &ev;
  mi_result* r = nullptr;
  ASSERT_EQ(mi_engine_preload(e, &p, &r), MI_CANCELLED) << last_error();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(mi_result_cancelled(r), 1);
  EXPECT_EQ(ev.plans, 1);
  EXPECT_EQ(ev.progress, 1) << "the cancel in the first step's callback stops the preload there";
  mi_memory_stats m{};
  m.struct_size = sizeof m;
  ASSERT_EQ(mi_result_memory(r, &m), MI_OK);
  EXPECT_EQ(m.dit_loads, 0);
  mi_result_free(r);
  // From another thread while it loads (it may also finish first: both leave the engine usable).
  mi_status status = MI_OK;
  mi_gen_params q = request();
  std::thread worker([&] { status = mi_engine_preload(e, &q, nullptr); });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  ASSERT_EQ(mi_cancel(e), MI_OK);
  worker.join();
  EXPECT_TRUE(status == MI_CANCELLED || status == MI_OK) << mi_status_string(status);
  std::printf("[c_api] preload cancelled after 300 ms: %s\n", mi_status_string(status));
  const Generated g = generate(e);
  EXPECT_FALSE(g.rgb.empty());
  mi_engine_free(e);
}

// A release whose cancel is set before it starts releases nothing: the next generation needs no DiT load and gives the
// same bits. Through the adapter (mi_engine_release() clears the flag when it marks the engine busy, so only a cancel
// landing between that and the pipeline's check reaches this case; the second part races one from another thread).
TEST_F(GpuCApiPreload, ReleaseCancelledBeforeItStarts) {
  mi_engine_params ep = engine_defaults();
  ep.model_path = model_.c_str();
  std::atomic<bool> go{false}, stop{true};
  {
    mi::api::PipelineAdapter a(ep);
    const mi_gen_params g = request();
    const mi::api::Generation first = a.generate(g, go);
    ASSERT_FALSE(first.rgb.empty());
    bool cancelled = false;
    mi_release_info info = a.release(true, stop, &cancelled);
    EXPECT_TRUE(cancelled);
    EXPECT_EQ(info.vram_released_bytes, 0u);
    EXPECT_EQ(info.dit_host_copy, 0);
    EXPECT_STREQ(info.note, "cancelled before it started");
    const mi::api::Generation again = a.generate(g, go);
    EXPECT_EQ(again.memory.dit_loads, 0) << "the DiT stayed resident";
    EXPECT_EQ(again.dit_load.source, MI_DIT_LOAD_SOURCE_NONE);
    EXPECT_TRUE(again.rgb == first.rgb);
    info = a.release(false, go, &cancelled);
    EXPECT_FALSE(cancelled);
    EXPECT_GT(info.vram_released_bytes, 0u);
  }

  // mi_engine_release() with mi_cancel() spinning on another thread: MI_CANCELLED when the cancel lands before the
  // pipeline starts releasing (VRAM kept: the next generate loads no DiT), MI_OK otherwise. A few tries; a miss is no
  // failure, the mapping is covered above and by CApi.ReleaseInfo.
  mi_engine* e = create();
  ASSERT_NE(e, nullptr);
  bool seen = false;
  for (int attempt = 0; attempt < 4 && !seen; ++attempt) {
    ASSERT_FALSE(generate(e).rgb.empty());  // the DiT resident
    std::atomic<bool> done{false};
    std::thread canceller([&] {
      while (!done.load()) (void)mi_cancel(e);
    });
    mi_release_info info{};
    info.struct_size = sizeof info;
    const mi_status s = mi_engine_release(e, 0, &info);
    done.store(true);
    canceller.join();
    ASSERT_TRUE(s == MI_OK || s == MI_CANCELLED) << mi_status_string(s) << ": " << last_error();
    if (s == MI_CANCELLED) {
      seen = true;
      EXPECT_EQ(info.vram_released_bytes, 0u);
      EXPECT_STREQ(info.note, "cancelled before it started");
      const Generated g = generate(e);
      EXPECT_EQ(g.memory.dit_loads, 0);
    }
  }
  std::printf("[c_api] mi_engine_release with a racing cancel: %s\n", seen ? "MI_CANCELLED seen" : "never cancelled early");
  mi_engine_free(e);
}

}  // namespace
