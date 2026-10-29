// mi_check (MI_CAP_CHECK): a dry run of mi_engine_create() + mi_generate(). CApiCheck.* run without a GPU (arguments,
// sizes, the engine's refusals before CUDA); GpuCheckAgreement.* compare the check's verdict with the real calls'
// over a matrix of configurations (skipped without a device or the weights).
#include "mossinfer/mossinfer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_util.h"
#include "util/json.h"
#include "util/path.h"
#include "util/regular_file.h"

namespace {

namespace fs = std::filesystem;

std::string last_error() { return mi_last_error_message(); }

mi_engine_params engine_defaults() {
  mi_engine_params p;
  EXPECT_EQ(mi_engine_params_init(&p, sizeof p), MI_OK);
  p.tier = MI_TIER_REFERENCE;
  return p;
}

mi_gen_params gen_defaults() {
  mi_gen_params p;
  EXPECT_EQ(mi_gen_params_init(&p, sizeof p), MI_OK);
  p.prompt = "a red apple on a wooden table";
  return p;
}

// The check's verdict, the plan JSON (empty without a plan) and the report's info.
struct Checked {
  mi_check_info info{};
  std::string message, plan_json, warnings, notes;
};
Checked run_check(const mi_engine_params& e, const mi_gen_params& g) {
  mi_check_report* report = nullptr;
  Checked c;
  EXPECT_EQ(mi_check(&e, &g, &report), MI_OK) << last_error();
  if (report == nullptr) return c;
  c.info.struct_size = sizeof c.info;
  EXPECT_EQ(mi_check_report_info(report, &c.info), MI_OK);
  c.message = mi_check_report_text(report, MI_CHECK_TEXT_MESSAGE);
  c.plan_json = mi_check_report_text(report, MI_CHECK_TEXT_PLAN_JSON);
  c.warnings = mi_check_report_text(report, MI_CHECK_TEXT_WARNINGS);
  c.notes = mi_check_report_text(report, MI_CHECK_TEXT_NOTES);
  mi_check_report_free(report);
  return c;
}

// ---------------------------------------------------------------- CPU

TEST(CApiCheck, ArgumentsSizesAndAccessors) {
  EXPECT_NE(mi_capabilities() & MI_CAP_CHECK, 0u);
  const mi_engine_params e = engine_defaults();
  const mi_gen_params g = gen_defaults();
  mi_check_report* report = reinterpret_cast<mi_check_report*>(1);
  EXPECT_EQ(mi_check(nullptr, &g, &report), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(report, nullptr);
  EXPECT_EQ(mi_check(&e, nullptr, &report), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_check(&e, &g, nullptr), MI_ERR_INVALID_ARGUMENT);
  mi_engine_params bad = e;
  bad.struct_size = 7;
  EXPECT_EQ(mi_check(&bad, &g, &report), MI_ERR_ABI_MISMATCH);
  mi_gen_params bad_g = g;
  bad_g.struct_size = 9;
  EXPECT_EQ(mi_check(&e, &bad_g, &report), MI_ERR_ABI_MISMATCH);
  mi_check_info info{};
  info.struct_size = sizeof info;
  EXPECT_EQ(mi_check_report_info(nullptr, &info), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_check_report_text(nullptr, MI_CHECK_TEXT_MESSAGE), nullptr);
  mi_check_report_free(nullptr);
}

// Every engine refusal mi_engine_create() gives before it touches CUDA: the check gives it too (stage, status and
// message), from the same parameter set, and leaves the caller's last error alone.
TEST(CApiCheck, EngineRefusalsAgreeWithEngineCreate) {
  mi::testing::TempDir dir("c_api_check_engine");
  const fs::path bundle = dir.path() / "bundle";
  for (const char* c : {"text_encoder", "tokenizer", "transformer", "vae"}) fs::create_directories(bundle / c);
  const std::string bundle_s = mi::util::path_utf8(bundle);
  const std::string missing_s = mi::util::path_utf8(dir.path() / "missing");
  const auto with = [&](auto edit) {
    mi_engine_params p = engine_defaults();
    p.model_path = bundle_s.c_str();
    edit(p);
    return p;
  };
  const mi_engine_params cases[] = {
      with([](mi_engine_params& p) { p.model_path = nullptr; }),
      with([&](mi_engine_params& p) { p.model_path = missing_s.c_str(); }),
      with([](mi_engine_params& p) { p.tier = MI_TIER_AUTO; }),
      with([](mi_engine_params& p) { p.mode = static_cast<mi_mode>(5); }),
      with([](mi_engine_params& p) { p.vram_budget_bytes = 1000; }),
      with([](mi_engine_params& p) { p._reserved1 = 1; }),
      with([&](mi_engine_params& p) { p.dit_file = missing_s.c_str(); }),
  };
  const mi_gen_params g = gen_defaults();
  for (const mi_engine_params& e : cases) {
    mi_engine* engine = nullptr;
    const mi_status real = mi_engine_create(&e, &engine);
    ASSERT_NE(real, MI_OK);
    const std::string real_message = last_error();
    const Checked c = run_check(e, g);
    EXPECT_EQ(c.info.status, real) << real_message;
    EXPECT_EQ(c.info.stage, MI_CHECK_STAGE_ENGINE_CREATE) << real_message;
    EXPECT_EQ(c.message, real_message);
    EXPECT_EQ(c.info.planned, 0);
    EXPECT_EQ(last_error(), real_message) << "mi_check changed the last error";
  }
}

// ---------------------------------------------------------------- GPU: the check against the real calls

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
fs::path community() {
  const char* env = std::getenv("MOSS_COMMUNITY_MODELS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/community");
}

struct Real {
  mi_status status = MI_OK;
  bool engine_refused = false;
  std::string message, plan_json;
};
// mi_engine_create() then mi_generate(): the first refusal, else the plan the generation was made with.
Real run_real(const mi_engine_params& e, const mi_gen_params& g) {
  Real r;
  mi_engine* engine = nullptr;
  r.status = mi_engine_create(&e, &engine);
  if (r.status != MI_OK) {
    r.engine_refused = true;
    r.message = last_error();
    return r;
  }
  mi_gen_params p = g;
  p.user_data = &r.plan_json;
  p.plan_callback = [](const char*, const char* json, void* user) { *static_cast<std::string*>(user) = json; };
  mi_result* result = nullptr;
  r.status = mi_generate(engine, &p, &result);
  if (r.status != MI_OK) r.message = last_error();
  mi_result_free(result);
  mi_engine_free(engine);
  return r;
}

const mi::json::Value& member(const mi::json::Value& v, const char* path) {
  const mi::json::Value* at = &v;
  std::string p = path;
  for (std::size_t i = 0; i <= p.size();) {
    const std::size_t dot = std::min(p.find('.', i), p.size());
    at = at->find(p.substr(i, dot - i));
    if (at == nullptr) throw std::runtime_error(std::string("no ") + path);
    i = dot + 1;
  }
  return *at;
}

// A JSON scalar as text, for comparing plan fields.
std::string scalar(const mi::json::Value& v) {
  if (v.is_string()) return v.as_string();
  if (v.is_bool()) return v.as_bool() ? "true" : "false";
  if (v.is_number()) return std::to_string(v.as_double());
  return "?";
}

class GpuCheckAgreement : public ::testing::Test {
 protected:
  void SetUp() override {
    mi_probe_info info{};
    info.struct_size = sizeof info;
    if (mi_probe(&info) != MI_OK || info.supported_device_count == 0) GTEST_SKIP() << "no supported CUDA device";
    if (!fs::is_directory(weights_root() / "sft" / "transformer")) GTEST_SKIP() << "no sft bundle";
    if (!fs::is_regular_file(int8_)) GTEST_SKIP() << "no INT8 DiT file";
  }
  const std::string sft_ = mi::util::path_utf8(weights_root() / "sft");
  const std::string int8_ = mi::util::path_utf8(community() / "comfy-org" / "moss_image_sft_int8_convrot.safetensors");
  const std::string int4_ = mi::util::path_utf8(community() / "toxicdog" / "moss_image_sft_int4_convrot.safetensors");
  const std::string nvfp4_ = mi::util::path_utf8(community() / "bennydaball" / "moss_image_sft_nvfp4.safetensors");
  const std::string te8_ = mi::util::path_utf8(community() / "comfy-org" / "qwen3.5_9b_int8_convrot.safetensors");
  const std::string vae_ = mi::util::path_utf8(community() / "comfy-org" / "flux2-vae.safetensors");
  const std::string lora_ = mi::util::path_utf8(community() / "axiomofmind" / "Moss-image-beauty-lora" / "pytorch_lora_weights.safetensors");
  const std::string edit_ = mi::util::path_utf8(community() / "chinoll" / "Moss-Image-Edit" / "adapter_model.safetensors");
  const std::string control_ = mi::util::path_utf8(community() / "comfy-org" / "moss_image_color_depth_control.safetensors");
  const std::string distill_ = mi::util::path_utf8(community() / "lodestones" / "MOSS-Image-SFT-8-steps-dmd");
  const std::string gguf_ =
      mi::util::path_utf8(community() / "gguf" / "leejet" / "MOSS-Image-SFT-GGUF" / "MOSS-Image-SFT-Q4_K.gguf");
};

// The matrix: each configuration is checked, then run for real (mi_engine_create, mi_generate at 256x256, 2 steps);
// both must accept, or refuse in the same call with the same status and message. Accepted ones also plan the same
// residencies and arena (the real run first: its fused self-check and stored hardware profile are what the check then
// reads, as a later engine of this process would).
TEST_F(GpuCheckAgreement, CheckAgreesWithTheRealCalls) {
  mi::testing::TempDir dir("check_agreement");
  const std::string cache = mi::util::path_utf8(dir.path() / "cache");
  const std::string png = mi::util::path_utf8(dir.path() / "ref.png");
  {
    std::vector<std::uint8_t> rgb(64 * 64 * 3, 128);
    ASSERT_EQ(mi_write_png(png.c_str(), rgb.data(), 64, 64), MI_OK);
  }
  const mi_lora beauty{lora_.c_str(), 0.8};
  const mi_lora edit{edit_.c_str(), 1.0};
  mi_reference_image ref{};
  ref.image_path = png.c_str();
  mi_control_image ctl{};
  ctl.kind = "nonexistent";
  ctl.image_path = png.c_str();
  ctl.strength = 1;
  ctl.end = 1;

  struct Case {
    const char* name;
    std::vector<std::string> files;  // skipped when one is missing
    std::function<void(mi_engine_params&, mi_gen_params&)> edit;
    bool accepted;  // expected verdict (both sides)
  };
  const std::vector<Case> cases = {
      {"int8 high", {int8_}, [&](mi_engine_params& e, mi_gen_params&) { e.dit_file = int8_.c_str(); e.tier = MI_TIER_HIGH; }, true},
      {"int8 high + edit adapter + reference", {int8_, edit_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         e.dit_file = int8_.c_str();
         e.tier = MI_TIER_HIGH;
         g.loras = &edit;
         g.loras_count = 1;
         g.reference_images = &ref;
         g.reference_images_count = 1;
       },
       true},
      {"int4 turbo + LoRA side branch", {int4_, lora_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         e.dit_file = int4_.c_str();
         e.tier = MI_TIER_TURBO;
         g.loras = &beauty;
         g.loras_count = 1;
       },
       true},
      {"tier auto", {}, [&](mi_engine_params& e, mi_gen_params&) { e.tier = MI_TIER_AUTO; }, false},
      {"bf16 DiT high exact", {},
       [&](mi_engine_params& e, mi_gen_params&) { e.tier = MI_TIER_HIGH; e.attention = MI_ATTENTION_EXACT; }, false},
      {"int8 DiT reference", {int8_}, [&](mi_engine_params& e, mi_gen_params&) { e.dit_file = int8_.c_str(); }, false},
      {"int8 text encoder reference", {te8_},
       [&](mi_engine_params& e, mi_gen_params&) { e.text_encoder_file = te8_.c_str(); }, false},
      {"int4 merge placement", {int4_, lora_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         e.dit_file = int4_.c_str();
         e.tier = MI_TIER_TURBO;
         g.loras = &beauty;
         g.loras_count = 1;
         g.lora_placement = MI_LORA_PLACEMENT_MERGE;
       },
       false},
      {"int8 strict merge", {int8_, lora_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         e.dit_file = int8_.c_str();
         e.tier = MI_TIER_HIGH;
         e.lora_strict = 1;
         g.loras = &beauty;
         g.loras_count = 1;
         g.lora_placement = MI_LORA_PLACEMENT_MERGE;
       },
       false},
      {"VAE file as a LoRA", {int8_, vae_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         static mi_lora not_a_lora;
         not_a_lora = {vae_.c_str(), 1.0};
         e.dit_file = int8_.c_str();
         e.tier = MI_TIER_HIGH;
         g.loras = &not_a_lora;
         g.loras_count = 1;
       },
       false},
      {"nvfp4 off an RTX 50", {nvfp4_}, [&](mi_engine_params& e, mi_gen_params&) { e.dit_file = nvfp4_.c_str(); e.tier = MI_TIER_TURBO; },
       false},
      {"gguf turbo", {gguf_},
       [&](mi_engine_params& e, mi_gen_params&) { e.dit_file = gguf_.c_str(); e.tier = MI_TIER_TURBO; },
       false},
      {"gguf without cache_dir", {gguf_},
       [&](mi_engine_params& e, mi_gen_params&) {
         e.dit_file = gguf_.c_str();
         e.tier = MI_TIER_HIGH;
         e.cache_dir = nullptr;
       },
       false},
      {"references with SLG", {int8_, edit_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         e.dit_file = int8_.c_str();
         e.tier = MI_TIER_HIGH;
         g.loras = &edit;
         g.loras_count = 1;
         g.reference_images = &ref;
         g.reference_images_count = 1;
         g.slg_scale = 2;
       },
       false},
      {"control kind", {int8_, control_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         e.dit_file = int8_.c_str();
         e.tier = MI_TIER_HIGH;
         g.control_adapter = control_.c_str();
         g.control_images = &ctl;
         g.control_images_count = 1;
       },
       false},
      {"distill time embedding", {distill_},
       [&](mi_engine_params& e, mi_gen_params& g) { e.tier = MI_TIER_REFERENCE; g.distill_dir = distill_.c_str(); }, false},
      {"budget", {int8_},
       [&](mi_engine_params& e, mi_gen_params& g) {
         e.dit_file = int8_.c_str();
         e.tier = MI_TIER_HIGH;
         e.vram_budget_bytes = std::uint64_t{1} << 30;
         g.width = g.height = 1024;
       },
       false},
  };
  int ran = 0;
  for (const Case& c : cases) {
    bool have = true;
    for (const std::string& f : c.files) have = have && fs::exists(mi::util::path_from_utf8(f));
    if (!have) continue;
    SCOPED_TRACE(c.name);
    mi_engine_params e = engine_defaults();
    e.model_path = sft_.c_str();
    // The hardware profile the first real run stores is what every later run and check plans with (a check reads it,
    // never probes); the embedding cache is off, as the check plans a text encoder that a disk-cache hit would skip.
    e.cache_dir = cache.c_str();
    e.embedding_cache = 0;
    mi_gen_params g = gen_defaults();
    g.width = g.height = 256;
    g.steps = 2;
    c.edit(e, g);
    const Real real = run_real(e, g);
    const Checked check = run_check(e, g);
    std::printf("[ matrix   ] %-40s real %-24s check %-24s stage %d\n", c.name, mi_status_string(real.status),
                mi_status_string(check.info.status), static_cast<int>(check.info.stage));
    EXPECT_EQ(check.info.status, real.status) << "check: " << check.message << "\nreal: " << real.message;
    EXPECT_EQ(check.message, real.message);
    EXPECT_EQ(real.status == MI_OK, c.accepted) << real.message;
    EXPECT_EQ(check.info.stage, real.status == MI_OK ? MI_CHECK_STAGE_NONE
                                : real.engine_refused ? MI_CHECK_STAGE_ENGINE_CREATE
                                                      : MI_CHECK_STAGE_GENERATE);
    if (real.status == MI_OK && check.info.status == MI_OK) {
      ASSERT_EQ(check.info.planned, 1);
      const mi::json::Value a = mi::json::parse(real.plan_json), b = mi::json::parse(check.plan_json);
      for (const char* k : {"dit.residency", "dit.precision", "dit.attention", "dit.resident_blocks", "dit.device_bytes",
                            "text_encoder.residency", "vae.tiled", "peaks.text_encoder", "peaks.denoise", "peaks.decode",
                            "arena_bytes"}) {
        EXPECT_EQ(scalar(member(a, k)), scalar(member(b, k))) << k;
      }
      EXPECT_GT(member(b, "estimate.total_s").as_double(), 0) << check.plan_json;
      EXPECT_EQ(member(b, "estimate.steps").as_double(), 2);
      EXPECT_GT(check.info.eta_seconds, 0);
    }
    ++ran;
  }
  EXPECT_GE(ran, 10);
}

// A GGUF DiT the cache has not converted: the check reports the conversion and writes nothing (no folder, no lock).
TEST_F(GpuCheckAgreement, GgufNotConvertedWritesNothing) {
  if (!fs::is_regular_file(mi::util::path_from_utf8(gguf_))) GTEST_SKIP() << "no GGUF DiT";
  mi::testing::TempDir dir("check_gguf");
  const std::string cache = mi::util::path_utf8(dir.path() / "cache");
  mi_engine_params e = engine_defaults();
  e.model_path = sft_.c_str();
  e.dit_file = gguf_.c_str();
  e.tier = MI_TIER_HIGH;
  e.cache_dir = cache.c_str();
  const Checked c = run_check(e, gen_defaults());
  EXPECT_EQ(c.info.status, MI_OK) << c.message;
  EXPECT_EQ(c.info.planned, 0);
  EXPECT_GT(c.info.gguf_convert_seconds, 1);
  EXPECT_NE(c.notes.find("the first load converts"), std::string::npos) << c.notes;
  EXPECT_TRUE(c.plan_json.empty());
  EXPECT_FALSE(fs::exists(dir.path() / "cache")) << "the check created the cache folder";
}

}  // namespace
