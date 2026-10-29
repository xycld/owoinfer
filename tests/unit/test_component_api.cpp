// mossinfer.h's component API (design §7.1, §7.2 in part).
//   ComponentApi.*     CPU: the init defaults and struct sizes, calls without an engine, tensor descriptors, foreign
//                      conditioning, the run and step conversions with their messages (component_adapter.h), the pure
//                      helpers against the pipeline's own functions, the run result's mapping.
//   GpuComponentApi.*  on real weights (GTEST_SKIP when absent): an L = 1 MOSS run through the C API with the recipe's
//                      euler-on-moss host loop in an adopted staging block gives mi_generate's plan and RGB bytes; an
//                      interrupt (mi_run_end from another thread mid-step) and mi_engine_free during a step.
#include "mossinfer/mossinfer.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <regex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "api/component_adapter.h"
#include "api/pipeline_adapter.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/run_schedule.h"
#include "pipeline/samplers/initial_noise.h"
#include "test_util.h"
#include "util/json.h"
#include "util/path.h"

namespace {

namespace fs = std::filesystem;

std::string last_error() { return mi_last_error_message(); }

bool has(const std::string& s, const std::string& fragment) { return s.find(fragment) != std::string::npos; }

// A foreign conditioning of `tokens` zero rows (host BF16).
struct Cond {
  std::vector<std::uint16_t> data;
  mi_conditioning* c = nullptr;
  explicit Cond(std::int64_t tokens) : data(static_cast<std::size_t>(tokens) * MI_TEXT_DIM, 0) {
    mi_tensor t{};
    t.data = data.data();
    t.memory = MI_MEMORY_HOST;
    t.dtype = MI_DTYPE_BF16;
    t.rank = 2;
    t.shape[0] = tokens;
    t.shape[1] = MI_TEXT_DIM;
    EXPECT_EQ(mi_conditioning_create(&t, &c), MI_OK) << last_error();
  }
  ~Cond() { mi_conditioning_free(c); }
  Cond(const Cond&) = delete;
  Cond& operator=(const Cond&) = delete;
};

mi_run_params run_defaults() {
  mi_run_params p;
  EXPECT_EQ(mi_run_params_init(&p, sizeof p), MI_OK);
  return p;
}

mi::pipeline::SessionOptions session_options(int max_rows = 3, int max_latents = 1) {
  mi::pipeline::SessionOptions o;
  o.max_rows = max_rows;
  o.max_latents = max_latents;
  return o;
}

void refused(const mi_run_params& p, const mi::pipeline::SessionOptions& o, const std::string& fragment) {
  try {
    mi::api::to_run_spec(p, o);
    ADD_FAILURE() << "accepted; expected a refusal containing \"" << fragment << "\"";
  } catch (const std::invalid_argument& e) {
    EXPECT_TRUE(has(e.what(), fragment)) << e.what();
  }
}

// ---------------------------------------------------------------- CPU

TEST(ComponentApi, InitDefaultsAndSizes) {
  mi_session_params s;
  ASSERT_EQ(mi_session_params_init(&s, sizeof s), MI_OK);
  EXPECT_EQ(s.struct_size, sizeof s);
  EXPECT_EQ(s.plan_request, nullptr);
  EXPECT_EQ(s.max_rows, 3);
  EXPECT_EQ(s.max_latents, 1);
  EXPECT_EQ(s.control_adapter, nullptr);
  mi_text_params t;
  ASSERT_EQ(mi_text_params_init(&t, sizeof t), MI_OK);
  EXPECT_EQ(t.max_sequence_length, 4096);
  EXPECT_EQ(t._reserved0, 0);
  const mi_run_params r = run_defaults();
  EXPECT_EQ(r.width, 1024);
  EXPECT_EQ(r.height, 1024);
  EXPECT_EQ(r.rows, nullptr);
  EXPECT_EQ(r.sigmas, nullptr);
  EXPECT_EQ(r.scheduler, MI_SCHEDULER_MOSS);
  EXPECT_EQ(r.steps, MI_DEFAULT_STEPS);
  EXPECT_EQ(r.shift, 5.0);
  EXPECT_EQ(r.flow_multiplier, 1.0);
  EXPECT_EQ(r.denoise_strength, 1.0);
  EXPECT_EQ(r.guidance, MI_GUIDANCE_HOST);
  EXPECT_EQ(r.output, MI_STEP_OUTPUT_ROWS);
  EXPECT_EQ(r.flags, 0u);
  EXPECT_EQ(r.cfg, 3.0);
  EXPECT_EQ(r.slg_scale, 0.0);
  EXPECT_EQ(r.slg_stop, 1.0);
  EXPECT_EQ(r.control_lora_strength, 1.0);
  EXPECT_EQ(r.cfg_truncation, 1.0);
  EXPECT_EQ(r.step_cache, MI_STEP_CACHE_OFF);
  EXPECT_EQ(r.control_resize, MI_RESIZE_CROP);
  EXPECT_EQ(r.step_cache_threshold, 0.0);
  mi_step_params st;
  ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
  EXPECT_EQ(st.step, -1);
  EXPECT_EQ(st.rows, 0u);
  EXPECT_EQ(st.x.data, nullptr);
  EXPECT_EQ(st.cuda_stream, nullptr);

  // One size each: anything else is an ABI mismatch, with the accepted size in the message.
  EXPECT_EQ(mi_session_params_init(&s, sizeof s - 8), MI_ERR_ABI_MISMATCH);
  EXPECT_TRUE(has(last_error(), "48")) << last_error();
  EXPECT_EQ(mi_text_params_init(&t, 24), MI_ERR_ABI_MISMATCH);
  mi_run_params r2;
  EXPECT_EQ(mi_run_params_init(&r2, 200), MI_ERR_ABI_MISMATCH);
  EXPECT_EQ(mi_step_params_init(&st, 600), MI_ERR_ABI_MISMATCH);
  EXPECT_EQ(mi_run_params_init(nullptr, sizeof r2), MI_ERR_INVALID_ARGUMENT);
}

TEST(ComponentApi, CallsWithoutAnEngine) {
  mi_session_params s;
  ASSERT_EQ(mi_session_params_init(&s, sizeof s), MI_OK);
  EXPECT_EQ(mi_session_open(nullptr, &s), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "engine is NULL")) << last_error();
  EXPECT_EQ(mi_session_close(nullptr), MI_ERR_INVALID_ARGUMENT);
  mi_text_params t;
  ASSERT_EQ(mi_text_params_init(&t, sizeof t), MI_OK);
  mi_conditioning* out[1] = {nullptr};
  EXPECT_EQ(mi_encode_text(nullptr, &t, out), MI_ERR_INVALID_ARGUMENT);
  const mi_run_params r = run_defaults();
  EXPECT_EQ(mi_run_begin(nullptr, &r), MI_ERR_INVALID_ARGUMENT);
  mi_run_info info{};
  info.struct_size = sizeof info;
  EXPECT_EQ(mi_run_query(nullptr, &info), MI_ERR_INVALID_ARGUMENT);
  std::size_t count = 7;
  EXPECT_EQ(mi_run_sigmas(nullptr, &count), nullptr);
  EXPECT_EQ(count, 0u);
  EXPECT_EQ(mi_run_sigmas(nullptr, nullptr), nullptr);
  EXPECT_TRUE(has(last_error(), "count is NULL")) << last_error();
  int block = 0;
  EXPECT_EQ(mi_run_adopt_staging(nullptr, &block, sizeof block), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_run_staging(nullptr, nullptr, nullptr, 0, nullptr), MI_ERR_INVALID_ARGUMENT);
  mi_step_params st;
  ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
  EXPECT_EQ(mi_denoise_step(nullptr, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
  mi_result* res = reinterpret_cast<mi_result*>(&block);
  EXPECT_EQ(mi_run_end(nullptr, &res), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(res, nullptr);
  mi_image im{};
  mi_tensor tensor{};
  EXPECT_EQ(mi_vae_encode(nullptr, &im, &tensor), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_vae_decode(nullptr, &tensor, &tensor), MI_ERR_INVALID_ARGUMENT);
}

TEST(ComponentApi, TensorDescriptors) {
  float buf[2 * 3 * 4] = {};
  mi_tensor t{};
  EXPECT_EQ(mi::api::to_tensor(t, "x").data, nullptr);  // absent
  t.data = buf;
  t.memory = MI_MEMORY_HOST;
  t.dtype = MI_DTYPE_F32;
  t.rank = 3;
  t.shape[0] = 2;
  t.shape[1] = 3;
  t.shape[2] = 4;
  const mi::Tensor v = mi::api::to_tensor(t, "x");
  EXPECT_EQ(v.rank, 3);
  EXPECT_EQ(v.dim(2), 4);
  EXPECT_EQ(v.device, mi::Device::kCPU);
  EXPECT_EQ(v.nbytes(), sizeof buf);
  const mi_tensor back = mi::api::from_tensor(v);
  EXPECT_EQ(std::memcmp(&back, &t, sizeof t), 0);
  const auto refused_tensor = [&](mi_tensor bad, const std::string& fragment) {
    try {
      mi::api::to_tensor(bad, "out[1]");
      ADD_FAILURE() << "accepted; expected " << fragment;
    } catch (const std::invalid_argument& e) {
      EXPECT_TRUE(has(e.what(), fragment)) << e.what();
    }
  };
  mi_tensor bad = t;
  bad.rank = 0;
  refused_tensor(bad, "out[1].rank must be 1 to 4");
  bad = t;
  bad.rank = 5;
  refused_tensor(bad, "out[1].rank");
  bad = t;
  bad.shape[1] = 0;
  refused_tensor(bad, "out[1].shape[1] must be positive");
  bad = t;
  bad.shape[3] = 1;
  refused_tensor(bad, "out[1].shape[3] must be 0 past the rank");
  bad = t;
  bad.dtype = static_cast<mi_dtype>(9);
  refused_tensor(bad, "out[1].dtype: unknown value 9");
  bad = t;
  bad.memory = static_cast<mi_memory>(2);
  refused_tensor(bad, "out[1].memory: unknown value 2");
  bad = t;
  bad._reserved0 = 1;
  refused_tensor(bad, "out[1]._reserved0 must be 0");
  bad = t;
  bad.rank = 2;
  bad.shape[0] = bad.shape[1] = std::int64_t{1} << 32;  // 2^64 elements: refused before the product overflows
  bad.shape[2] = 0;
  refused_tensor(bad, "out[1]: too many elements");
  bad = t;
  bad.data = reinterpret_cast<char*>(buf) + 2;  // not F32-aligned
  refused_tensor(bad, "out[1].data must be aligned to its element size (4 bytes)");
  bad.dtype = MI_DTYPE_U8;  // a byte tensor has no alignment rule
  bad.shape[0] = 1;
  EXPECT_EQ(mi::api::to_tensor(bad, "rgb").dtype, mi::DType::kU8);
}

TEST(ComponentApi, ForeignConditioning) {
  // F32 rounds to bf16, nearest even: 1 + 2^-8 is a tie (to 1.0), 1 + 3 * 2^-8 a tie (to 1 + 2^-6), 1 + 2^-7 exact.
  std::vector<float> f(2 * MI_TEXT_DIM, 0.0f);
  f[0] = 1.0f + std::ldexp(1.0f, -8);
  f[1] = 1.0f + 3 * std::ldexp(1.0f, -8);
  f[2] = 1.0f + std::ldexp(1.0f, -7);
  f[3] = -2.5f;
  mi_tensor t{};
  t.data = f.data();
  t.memory = MI_MEMORY_HOST;
  t.dtype = MI_DTYPE_F32;
  t.rank = 2;
  t.shape[0] = 2;
  t.shape[1] = MI_TEXT_DIM;
  mi_conditioning* c = nullptr;
  ASSERT_EQ(mi_conditioning_create(&t, &c), MI_OK) << last_error();
  mi_conditioning_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_conditioning_query(c, &info), MI_OK);
  EXPECT_EQ(info.tokens, 2);
  EXPECT_EQ(info.text_dim, MI_TEXT_DIM);
  EXPECT_EQ(info.dtype, MI_DTYPE_BF16);
  EXPECT_EQ(info.engine_encoded, 0);
  mi_tensor view{};
  ASSERT_EQ(mi_conditioning_tensor(c, &view), MI_OK);
  EXPECT_EQ(view.memory, MI_MEMORY_HOST);
  EXPECT_EQ(view.dtype, MI_DTYPE_BF16);
  EXPECT_EQ(view.rank, 2);
  EXPECT_EQ(view.shape[0], 2);
  EXPECT_EQ(view.shape[1], MI_TEXT_DIM);
  const auto* b = static_cast<const std::uint16_t*>(view.data);
  EXPECT_EQ(b[0], 0x3f80);  // 1.0
  EXPECT_EQ(b[1], 0x3f82);  // 1 + 2^-6
  EXPECT_EQ(b[2], 0x3f81);  // 1 + 2^-7
  EXPECT_EQ(b[3], 0xc020);  // -2.5
  mi_prompt_tokens tok{};
  tok.struct_size = sizeof tok;
  ASSERT_EQ(mi_conditioning_prompt_tokens(c, &tok), MI_OK);
  EXPECT_EQ(tok.total_tokens, 0);  // foreign: zeros
  EXPECT_EQ(tok.struct_size, sizeof tok);
  // A BF16 tensor is copied as is.
  mi_tensor tb = view;
  mi_conditioning* c2 = nullptr;
  ASSERT_EQ(mi_conditioning_create(&tb, &c2), MI_OK);
  mi_tensor view2{};
  ASSERT_EQ(mi_conditioning_tensor(c2, &view2), MI_OK);
  EXPECT_NE(view2.data, view.data);
  EXPECT_EQ(std::memcmp(view2.data, view.data, 2 * MI_TEXT_DIM * 2), 0);
  mi_conditioning_free(c2);
  mi_conditioning_free(c);
  mi_conditioning_free(nullptr);  // no-op

  // Bounds: S in [1, 4096], the text dim, the dtype.
  mi_conditioning* none = reinterpret_cast<mi_conditioning*>(&info);
  t.shape[1] = 4095;
  EXPECT_EQ(mi_conditioning_create(&t, &none), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(none, nullptr);
  EXPECT_TRUE(has(last_error(), "expected f32 or bf16 [1..4096, 4096]")) << last_error();
  t.shape[1] = MI_TEXT_DIM;
  t.shape[0] = 4097;
  EXPECT_EQ(mi_conditioning_create(&t, &none), MI_ERR_INVALID_ARGUMENT);
  t.shape[0] = 2;
  t.dtype = MI_DTYPE_U8;
  EXPECT_EQ(mi_conditioning_create(&t, &none), MI_ERR_INVALID_ARGUMENT);
  t.dtype = MI_DTYPE_F32;
  t.data = nullptr;
  EXPECT_EQ(mi_conditioning_create(&t, &none), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_conditioning_create(nullptr, &none), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_conditioning_query(nullptr, &info), MI_ERR_INVALID_ARGUMENT);
  info.struct_size = 16;
  EXPECT_EQ(mi_conditioning_query(none, &info), MI_ERR_ABI_MISMATCH);
}

TEST(ComponentApi, RunSpecValidation) {
  Cond a(20), b(30);
  const mi_conditioning* ca = a.c;
  const mi_conditioning* cb = b.c;
  const std::int32_t skip22[] = {22};
  mi_run_row rows[4] = {{ca, nullptr, 0, 0, 0}, {cb, skip22, 1, 0, 0}, {ca, nullptr, 0, 0, 0}, {cb, nullptr, 0, 0, 0}};
  mi_run_params p = run_defaults();
  p.width = p.height = 512;
  p.rows = rows;
  p.rows_count = 2;
  const mi::pipeline::SessionOptions o = session_options();

  // A HOST run and a MOSS run are accepted; the spec carries what was given.
  mi::pipeline::RunSpec s = mi::api::to_run_spec(p, o);
  EXPECT_EQ(s.rows.size(), 2u);
  EXPECT_EQ(s.rows[1].skip_layers, std::vector<int>{22});
  EXPECT_EQ(s.guidance, mi::pipeline::GuidanceMode::kHost);
  EXPECT_EQ(s.steps, 50);
  EXPECT_FALSE(s.exact_sigma);
  mi_run_params m = p;
  m.guidance = MI_GUIDANCE_MOSS;
  m.output = MI_STEP_OUTPUT_GUIDED_VELOCITY;
  m.flags = MI_RUN_EXACT_SIGMA | MI_RUN_CROPS;
  m.steps = 6;
  s = mi::api::to_run_spec(m, o);
  EXPECT_EQ(s.output, mi::pipeline::StepOutput::kGuidedVelocity);
  EXPECT_TRUE(s.exact_sigma);
  EXPECT_TRUE(s.crops);
  EXPECT_EQ(s.cfg, 3.0);

  // The C fields.
  mi_run_params x = p;
  x._reserved0 = 1;
  refused(x, o, "mi_run_params._reserved0 must be 0");
  x = p;
  x.guidance = static_cast<mi_guidance>(7);
  refused(x, o, "guidance: unknown value 7");
  x = p;
  x.output = static_cast<mi_step_output>(9);
  refused(x, o, "output: unknown value 9");
  x = p;
  x.flags = 0x4u;
  refused(x, o, "flags: unknown bits 4");
  x = p;
  x.scheduler = static_cast<mi_scheduler>(99);
  refused(x, o, "scheduler: unknown value 99");
  x = p;
  x.control_resize = static_cast<mi_resize_mode>(9);
  refused(x, o, "resize: unknown value 9");
  x = p;
  x.step_cache = static_cast<mi_step_cache>(5);
  refused(x, o, "step_cache: unknown value 5");

  // Rows.
  x = p;
  x.rows_count = 0;
  refused(x, o, "A run takes 1 to max_rows (3) rows, not 0");
  x.rows_count = 4;
  refused(x, o, "A run takes 1 to max_rows (3) rows, not 4");
  x = p;
  x.rows = nullptr;
  refused(x, o, "rows: NULL array with a nonzero count");
  mi_run_row bad[2] = {rows[0], rows[1]};
  x = p;
  x.rows = bad;
  bad[1].conditioning = nullptr;
  refused(x, o, "rows[1].conditioning is NULL");
  bad[1] = rows[1];
  bad[1]._reserved0 = 3;
  refused(x, o, "rows[1]._reserved0 must be 0");
  bad[1] = rows[1];
  const std::int32_t skip42[] = {42};
  bad[1].skip_layers = skip42;
  refused(x, o, "42");
  bad[1].skip_layers = nullptr;
  refused(x, o, "rows[1].skip_layers: NULL array with a nonzero count");
  bad[1] = rows[1];
  bad[1].latent = 1;
  refused(x, o, "latent 1 is outside the session's 0..0");
  EXPECT_EQ(mi::api::to_run_spec(x, session_options(3, 3)).rows[1].latent, 1);  // L = 2 (HOST): latents 0 and 1
  bad[0].latent = 2;
  refused(x, session_options(3, 3), "no row reads latent 0");
  bad[0] = rows[0];
  bad[1] = rows[1];

  // Guidance modes.
  x = m;
  x.rows_count = 3;
  rows[2] = rows[0];
  refused(x, o, "MOSS guidance takes 1 or 2 rows (cond, uncond) on latent 0");
  x = m;
  x.rows = bad;
  bad[0].skip_layers = skip22;
  bad[0].skip_layers_count = 1;
  refused(x, o, "rows[0] is the cond row, which skips no blocks");
  bad[0] = rows[0];
  x = m;
  x.cfg = 1.0;
  refused(x, o, "Guided outputs with an uncond row need cfg != 1");
  x = p;
  x.output = MI_STEP_OUTPUT_GUIDED_DENOISED;
  refused(x, o, "Guided outputs need MOSS guidance");
  x = p;
  x.cfg = 2.0;
  refused(x, o, "HOST guidance keeps cfg, slg_* and cfg_truncation at their defaults");
  x = p;
  x.cfg_truncation = 0.5;
  refused(x, o, "HOST guidance keeps");
  x = p;
  x.cfg = -1.0;
  refused(x, o, "CFG must be a finite number >= 0");

  // The schedule.
  float sig[3] = {1.0f, 0.5f, 0.0f};
  x = p;
  x.sigmas_count = 3;
  refused(x, o, "sigmas: NULL array with a nonzero count");
  x.sigmas = sig;
  x.sigmas_count = 0;
  refused(x, o, "sigmas: an array with a count of 0");
  x.sigmas_count = 1;
  refused(x, o, "");
  x.sigmas_count = 10002;
  refused(x, o, "sigmas: at most 10001 values, not 10002");
  x.sigmas_count = 3;
  EXPECT_EQ(mi::api::to_run_spec(x, o).sigmas.size(), 3u);
  sig[1] = std::numeric_limits<float>::quiet_NaN();
  refused(x, o, "");
  sig[1] = 1.5f;  // increasing
  refused(x, o, "");
  sig[1] = 0.5f;
  x = p;
  x.denoise_strength = 0.0;
  refused(x, o, "The schedule has no step left (denoise strength 0)");
  x.denoise_strength = 1.5;
  refused(x, o, "Denoise strength must be between 0 and 1");
  x = p;
  x.steps = 0;
  refused(x, o, "Steps must be between 1 and 10000");

  // The step cache, control images, the shared mi_gen_params rules and their messages.
  x = p;
  x.step_cache = MI_STEP_CACHE_FBCACHE;
  refused(x, o, "The step cache needs an explicit threshold in (0, 1]");
  x.step_cache_threshold = 0.2;
  EXPECT_EQ(mi::api::to_run_spec(x, o).step_cache_threshold, 0.2);
  x.step_cache = MI_STEP_CACHE_OFF;
  x.step_cache_threshold = 2.0;  // ignored with the step cache off
  EXPECT_EQ(mi::api::to_run_spec(x, o).step_cache_threshold, 0.0);
  const std::uint8_t px[16 * 16 * 3] = {};
  const mi_image img{px, 16, 16, 3, 0};
  const mi_control_image ci{"depth", &img, nullptr, 1.0, 0.0, 1.0};
  x = p;
  x.control_images = &ci;
  x.control_images_count = 1;
  refused(x, o, "control_images need a control_adapter");
  x = p;
  x.width = 1000;
  refused(x, o, "multiples of 16");
  x = p;
  const mi_lora lora{"/nonexistent/mossinfer-test.safetensors", 1.0};
  x.loras = &lora;
  x.loras_count = 1;
  refused(x, o, "loras[0].path: file not found");
  x = m;
  x.slg_layers_count = 1;
  refused(x, o, "slg_layers: NULL array with a nonzero count");
}

// to_session_options: the control adapter must name an existing regular file (a Windows device path is refused by the
// boundary check before any file-system call).
TEST(ComponentApi, SessionOptions) {
  mi::testing::TempDir dir("component_session_options");
  mi_session_params sp;
  ASSERT_EQ(mi_session_params_init(&sp, sizeof sp), MI_OK);
  sp.max_rows = 4;
  sp.max_latents = 2;
  mi::pipeline::SessionOptions o = mi::api::to_session_options(sp, mi::pipeline::Request{}, 1);
  EXPECT_EQ(o.max_rows, 4);
  EXPECT_EQ(o.max_latents, 2);
  EXPECT_EQ(o.device, 1);
  EXPECT_TRUE(o.control_adapter.empty());
  const auto refused_adapter = [&](const std::string& path, const std::string& fragment) {
    sp.control_adapter = path.c_str();
    try {
      mi::api::to_session_options(sp, mi::pipeline::Request{}, 0);
      ADD_FAILURE() << "accepted " << path;
    } catch (const std::invalid_argument& e) {
      EXPECT_TRUE(has(e.what(), fragment)) << e.what();
    }
  };
  refused_adapter(mi::util::path_utf8(dir.path() / "missing.safetensors"), "control_adapter: file not found");
  refused_adapter(mi::util::path_utf8(dir.path()), "control_adapter: not a regular file");
#ifdef _WIN32
  refused_adapter(R"(\\.\pipe\mossinfer_test)", "control_adapter: not a regular file");
#endif
  const fs::path file = dir.path() / "control.safetensors";
  std::ofstream(file) << "x";
  const std::string file_s = mi::util::path_utf8(file);
  sp.control_adapter = file_s.c_str();
  EXPECT_EQ(mi::api::to_session_options(sp, mi::pipeline::Request{}, 0).control_adapter, file);
  sp.control_adapter = "";
  EXPECT_TRUE(mi::api::to_session_options(sp, mi::pipeline::Request{}, 0).control_adapter.empty());
}

TEST(ComponentApi, TokenCap) {
  // 2048^2 with L = 2 and CFG: 4 rows of 16384 + text tokens each exceed MI_MAX_CALL_TOKENS.
  Cond a(50), b(10);
  const std::int32_t skip22[] = {22};
  const mi_run_row rows[4] = {
      {a.c, nullptr, 0, 0, 0}, {b.c, skip22, 1, 0, 0}, {a.c, nullptr, 0, 1, 0}, {b.c, skip22, 1, 1, 0}};
  mi_run_params p = run_defaults();
  p.width = p.height = 2048;
  p.rows = rows;
  p.rows_count = 4;
  refused(p, session_options(4, 2), "MI_MAX_CALL_TOKENS");
  p.width = p.height = 1024;
  EXPECT_NO_THROW(mi::api::to_run_spec(p, session_options(4, 2)));
}

TEST(ComponentApi, StepArgs) {
  float x[128 * 4 * 4] = {}, o0[128 * 4 * 4] = {};
  mi_step_params p;
  ASSERT_EQ(mi_step_params_init(&p, sizeof p), MI_OK);
  p.sigma = 0.25f;
  p.rows = 2;
  p.flags = MI_STEP_CONTROL_CHANGED;
  const auto f32 = [](float* d) {
    mi_tensor t{};
    t.data = d;
    t.memory = MI_MEMORY_HOST;
    t.dtype = MI_DTYPE_F32;
    t.rank = 3;
    t.shape[0] = 128;
    t.shape[1] = 4;
    t.shape[2] = 4;
    return t;
  };
  p.x = f32(x);
  p.out[1] = f32(o0);
  const mi::pipeline::StepArgs a = mi::api::to_step_args(p);
  EXPECT_EQ(a.sigma, 0.25f);
  EXPECT_EQ(a.step, -1);
  EXPECT_EQ(a.rows, 2u);
  EXPECT_TRUE(a.control_changed);
  EXPECT_EQ(a.x.data, x);
  EXPECT_EQ(a.out[0].data, nullptr);
  EXPECT_EQ(a.out[1].data, o0);
  EXPECT_EQ(a.control.data, nullptr);
  p.flags = 0x2u;
  EXPECT_THROW(mi::api::to_step_args(p), std::invalid_argument);
  p.flags = 0;
  p.out[3] = f32(o0);
  p.out[3].rank = 7;
  try {
    mi::api::to_step_args(p);
    ADD_FAILURE() << "accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_TRUE(has(e.what(), "out[3].rank")) << e.what();
  }
}

TEST(ComponentApi, ScheduleEqualsThePipelines) {
  constexpr mi_scheduler kSchedulers[] = {MI_SCHEDULER_MOSS,       MI_SCHEDULER_SIMPLE,      MI_SCHEDULER_NORMAL,
                                          MI_SCHEDULER_KARRAS,     MI_SCHEDULER_EXPONENTIAL, MI_SCHEDULER_BETA,
                                          MI_SCHEDULER_SGM_UNIFORM};
  std::vector<float> buf(10001);
  for (const mi_scheduler k : kSchedulers) {
    for (const int steps : {1, 2, 20, 50, 1000, 1001}) {
      for (const double shift : {1.0, 3.0, 5.0}) {
        for (const double mult : {1.0, 1000.0}) {
          for (const double strength : {1.0, 0.75, 0.5, 0.1}) {
            const std::vector<float> want = mi::pipeline::engine_sigmas(
                mi::api::to_scheduler(k), steps, mi::pipeline::FlowSampling{shift, mult}, strength);
            std::size_t n = 0;
            ASSERT_EQ(mi_schedule(k, steps, shift, mult, strength, nullptr, 0, &n), MI_OK) << last_error();
            ASSERT_EQ(n, want.size());
            std::size_t got_n = 0;
            ASSERT_EQ(mi_schedule(k, steps, shift, mult, strength, buf.data(), buf.size(), &got_n), MI_OK);
            ASSERT_EQ(got_n, n);
            EXPECT_EQ(mi::pipeline::first_schedule_difference(std::span<const float>(buf.data(), n), want),
                      mi::pipeline::kSchedulesEqual)
                << k << " " << steps << " " << shift << " " << mult << " " << strength;
          }
        }
      }
    }
  }
  std::size_t n = 9;
  ASSERT_EQ(mi_schedule(MI_SCHEDULER_MOSS, 20, 5.0, 1.0, 0.0, nullptr, 0, &n), MI_OK);
  EXPECT_EQ(n, 0u);  // strength 0: no step left
  ASSERT_EQ(mi_schedule(MI_SCHEDULER_KARRAS, 20, 5.0, 1.0, 1.0, nullptr, 0, &n), MI_OK);
  EXPECT_EQ(mi_schedule(MI_SCHEDULER_KARRAS, 20, 5.0, 1.0, 1.0, buf.data(), n - 1, &n), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "capacity")) << last_error();
  EXPECT_EQ(mi_schedule(MI_SCHEDULER_MOSS, 0, 5.0, 1.0, 1.0, nullptr, 0, &n), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_schedule(static_cast<mi_scheduler>(42), 20, 5.0, 1.0, 1.0, nullptr, 0, &n), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_schedule(MI_SCHEDULER_MOSS, 20, 5.0, 1.0, 1.5, nullptr, 0, &n), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_schedule(MI_SCHEDULER_MOSS, 20, 5.0, 1.0, 1.0, nullptr, 0, nullptr), MI_ERR_INVALID_ARGUMENT);
}

TEST(ComponentApi, SamplerPredictionNoiseAndResize) {
  // SamplerDriver::prediction(): velocity for ab2 on any scheduler and for euler on moss; denoised otherwise.
  constexpr mi_sampler kSamplers[] = {MI_SAMPLER_EULER,  MI_SAMPLER_AB2,          MI_SAMPLER_EULER_ANCESTRAL,
                                      MI_SAMPLER_HEUN,   MI_SAMPLER_DPMPP_2M,     MI_SAMPLER_DPMPP_2M_SDE,
                                      MI_SAMPLER_ER_SDE, MI_SAMPLER_RES_MULTISTEP};
  for (const mi_sampler s : kSamplers) {
    for (int k = MI_SCHEDULER_MOSS; k <= MI_SCHEDULER_SGM_UNIFORM; ++k) {
      const bool velocity = s == MI_SAMPLER_AB2 || (s == MI_SAMPLER_EULER && k == MI_SCHEDULER_MOSS);
      EXPECT_EQ(mi_sampler_prediction(s, static_cast<mi_scheduler>(k)),
                velocity ? MI_STEP_OUTPUT_GUIDED_VELOCITY : MI_STEP_OUTPUT_GUIDED_DENOISED);
    }
  }
  EXPECT_EQ(mi_sampler_prediction(MI_SAMPLER_DATASPACE, MI_SCHEDULER_MOSS), MI_STEP_OUTPUT_MAX_ENUM);
  EXPECT_EQ(mi_sampler_prediction(static_cast<mi_sampler>(99), MI_SCHEDULER_MOSS), MI_STEP_OUTPUT_MAX_ENUM);
  EXPECT_EQ(mi_sampler_prediction(MI_SAMPLER_EULER, static_cast<mi_scheduler>(99)), MI_STEP_OUTPUT_MAX_ENUM);

  // mi_initial_noise = initial_noise.
  for (const mi_noise_layout layout : {MI_NOISE_LAYOUT_OFFICIAL, MI_NOISE_LAYOUT_BATCH_STREAM}) {
    std::vector<float> got(128 * 24 * 32);
    ASSERT_EQ(mi_initial_noise(1234, layout, 512, 384, got.data(), got.size()), MI_OK) << last_error();
    const std::vector<float> want =
        mi::pipeline::initial_noise(layout == MI_NOISE_LAYOUT_OFFICIAL ? mi::pipeline::NoiseLayout::kOfficial
                                                                       : mi::pipeline::NoiseLayout::kBatchStream,
                                    1234, 128, 24, 32);
    EXPECT_TRUE(got == want);
  }
  std::vector<float> small(10);
  EXPECT_EQ(mi_initial_noise(1, MI_NOISE_LAYOUT_OFFICIAL, 512, 512, small.data(), small.size()),
            MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_initial_noise(-1, MI_NOISE_LAYOUT_OFFICIAL, 16, 16, small.data(), 2048), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_initial_noise(std::int64_t{1} << 53, MI_NOISE_LAYOUT_OFFICIAL, 16, 16, small.data(), 2048),
            MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_initial_noise(1, MI_NOISE_LAYOUT_OFFICIAL, 500, 512, small.data(), 1 << 20), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_initial_noise(1, static_cast<mi_noise_layout>(5), 16, 16, small.data(), 2048), MI_ERR_INVALID_ARGUMENT);

  // mi_image_resize = resize_to, RGB and grayscale, every mode.
  for (const int ch : {1, 3}) {
    mi::pipeline::Image src;
    src.width = 37;
    src.height = 23;
    src.channels = ch;
    for (int i = 0; i < 37 * 23 * ch; ++i) src.pixels.push_back(static_cast<std::uint8_t>((i * 7919) % 251));
    const mi_image im{src.pixels.data(), 37, 23, ch, 0};
    for (const mi_resize_mode mode : {MI_RESIZE_CROP, MI_RESIZE_FIT, MI_RESIZE_STRETCH}) {
      std::vector<std::uint8_t> out(64 * 48 * ch);
      ASSERT_EQ(mi_image_resize(&im, 64, 48, mode, out.data(), out.size()), MI_OK) << last_error();
      const mi::pipeline::Image want = mi::pipeline::resize_to(src, 64, 48, mi::api::to_resize(mode));
      EXPECT_TRUE(out == want.pixels) << ch << " " << mode;
    }
    std::vector<std::uint8_t> out(10);
    EXPECT_EQ(mi_image_resize(&im, 64, 48, MI_RESIZE_CROP, out.data(), out.size()), MI_ERR_INVALID_ARGUMENT);
  }
  const std::uint8_t px[4] = {};
  std::vector<std::uint8_t> big(1);
  const mi_image huge{big.data(), 16000, 16000, 1, 0};  // 256 MP: refused before any pixel is read
  EXPECT_EQ(mi_image_resize(&huge, 64, 64, MI_RESIZE_CROP, big.data(), big.size()), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "64,000,000 pixels")) << last_error();
  const mi_image small_im{big.data(), 1, 1, 1, 0};
  EXPECT_EQ(mi_image_resize(&small_im, 16000, 16000, MI_RESIZE_CROP, big.data(), big.size()), MI_ERR_INVALID_ARGUMENT);
  const mi_image two{px, 2, 2, 2, 0};
  std::uint8_t out[64];
  EXPECT_EQ(mi_image_resize(&two, 4, 4, MI_RESIZE_CROP, out, sizeof out), MI_ERR_INVALID_ARGUMENT);
}

TEST(ComponentApi, RunResultMapping) {
  Cond a(20);
  mi::pipeline::RunResult run;
  run.spec.width = 512;
  run.spec.height = 384;
  run.spec.steps = 6;
  run.spec.guidance = mi::pipeline::GuidanceMode::kMoss;
  run.spec.rows = {{a.c->c, {}, 0}};
  run.guidance = mi::pipeline::GuidanceMode::kMoss;
  run.schedule_matches = false;
  run.off_schedule_calls = 3;
  run.calls = 9;
  run.foreign_rows = {0};
  run.note = "sigma[2] is 0.5 here.";
  run.result.width = 512;
  run.result.height = 384;
  run.result.steps_scheduled = 6;
  run.result.steps_run = 5;
  run.result.tier.tier = mi::planner::Tier::kHigh;
  mi_result* r = mi::api::new_result(mi::api::to_generation(run, {}));
  EXPECT_STREQ(mi_result_text(r, MI_TEXT_COMPONENT_NOTE), "sigma[2] is 0.5 here.");
  EXPECT_EQ(mi_result_rgb(r, nullptr), nullptr);
  EXPECT_EQ(mi_result_steps_run(r), 5);
  EXPECT_EQ(mi_result_width(r), 512);
  const mi::json::Value doc = mi::json::parse(mi_result_report_json(r));
  const mi::json::Value& c = doc.at("component");
  EXPECT_EQ(c.at("guidance").as_string(), "moss");
  EXPECT_FALSE(c.at("schedule_matches").as_bool());
  EXPECT_EQ(c.at("off_schedule_calls").as_int(), 3);
  EXPECT_EQ(c.at("calls").as_int(), 9);
  ASSERT_EQ(c.at("foreign_rows").as_array().size(), 1u);
  EXPECT_EQ(c.at("note").as_string(), "sigma[2] is 0.5 here.");
  mi_tier_info t{};
  t.struct_size = sizeof t;
  ASSERT_EQ(mi_result_tier(r, &t), MI_OK);
  EXPECT_EQ(t.certified, 0);
  EXPECT_EQ(t.run_certified, 0);
  mi_result_free(r);

  // The run's facts and a step's report.
  mi::pipeline::RunInfo i;
  i.latent_height = 24;
  i.latent_width = 32;
  i.steps = 6;
  i.outputs = 3;
  i.rows = 5;
  i.schedule_matches = true;
  i.off_schedule_calls = std::int64_t{1} << 40;
  i.calls = 7;
  i.control_channels = 256;
  i.latents = 2;
  i.staging_bytes = 123456789012ull;
  const mi_run_info o = mi::api::to_run_info(i);
  EXPECT_EQ(o.latent_channels, 128);
  EXPECT_EQ(o.latent_height, 24);
  EXPECT_EQ(o.outputs, 3);
  EXPECT_EQ(o.rows, 5u);
  EXPECT_EQ(o.schedule_matches, 1);
  EXPECT_EQ(o.off_schedule_calls, std::numeric_limits<std::int32_t>::max());  // saturated
  EXPECT_EQ(o.calls, 7);
  EXPECT_EQ(o.control_channels, 256);
  EXPECT_EQ(o.latents, 2);
  EXPECT_EQ(o.staging_bytes, 123456789012ull);
  mi::pipeline::StepReport sr;
  sr.step = 4;
  sr.rows = 3;
  sr.cached = true;
  sr.on_schedule = true;
  sr.seconds = 0.5;
  const mi_step_info si = mi::api::to_step_info(sr);
  EXPECT_EQ(si.step, 4);
  EXPECT_EQ(si.rows, 3u);
  EXPECT_EQ(si.cached, 1);
  EXPECT_EQ(si.on_schedule, 1);
  EXPECT_EQ(si.seconds, 0.5);
}

// ---------------------------------------------------------------- GPU, real weights

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
const fs::path kInt8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";

// The plan JSON without live machine state, which moves between any two calls (as GpuSession's plan_facts).
std::string plan_facts(const std::string& report) {
  const std::string key = "\n \"plan\": ";
  const std::size_t at = report.find(key);
  if (at == std::string::npos) return "no plan";
  const std::size_t from = at + key.size();
  std::string j = report.substr(from, report.find('\n', from) - from);
  j = std::regex_replace(j, std::regex("(free|host private|Arena cap) [0-9.]+ GiB"), "$1 X GiB");
  return std::regex_replace(j, std::regex("\"(host_private_bytes|arena_cap_bytes|host_limit_bytes)\": ?[0-9]+"),
                            "\"$1\": X");
}

// One engine per test (the SFT bundle, the INT8 DiT, high tier, q8_fp8 attention): 512^2, 6 steps.
class GpuComponentApi : public ::testing::Test {
 protected:
  void SetUp() override {
    mi_probe_info info{};
    info.struct_size = sizeof info;
    if (mi_probe(&info) != MI_OK || info.supported_device_count == 0) GTEST_SKIP() << "no supported CUDA device";
    if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(kInt8)) {
      GTEST_SKIP() << "SFT bundle or INT8 DiT not present";
    }
    model_ = mi::util::path_utf8(weights_root() / "sft");
    dit_ = mi::util::path_utf8(kInt8);
    engine_ = create();
    ASSERT_NE(engine_, nullptr);
  }
  void TearDown() override { mi_engine_free(engine_); }

  mi_engine* create() {
    mi_engine_params p;
    EXPECT_EQ(mi_engine_params_init(&p, sizeof p), MI_OK);
    p.model_path = model_.c_str();
    p.dit_file = dit_.c_str();
    p.tier = MI_TIER_HIGH;
    p.attention = MI_ATTENTION_Q8_FP8;
    mi_engine* e = nullptr;
    EXPECT_EQ(mi_engine_create(&p, &e), MI_OK) << last_error();
    return e;
  }

  static mi_gen_params request(int size = 512, int steps = 6) {
    mi_gen_params p;
    EXPECT_EQ(mi_gen_params_init(&p, sizeof p), MI_OK);
    p.prompt = "a red apple on a wooden table, studio light";
    p.negative_prompt = "blurry";
    p.width = p.height = size;
    p.steps = steps;
    p.seed = 7;
    return p;
  }

  static mi_tensor host_f32(float* data, std::int64_t c, std::int64_t h, std::int64_t w) {
    mi_tensor t{};
    t.data = data;
    t.memory = MI_MEMORY_HOST;
    t.dtype = MI_DTYPE_F32;
    t.rank = 3;
    t.shape[0] = c;
    t.shape[1] = h;
    t.shape[2] = w;
    return t;
  }

  // An open session with the request's two conditionings (prompt, negative prompt).
  void open_and_encode(const mi_gen_params& q, mi_conditioning* conds[2], mi_progress_fn progress = nullptr,
                       void* user = nullptr) {
    mi_session_params sp;
    ASSERT_EQ(mi_session_params_init(&sp, sizeof sp), MI_OK);
    sp.plan_request = &q;
    sp.progress_callback = progress;
    sp.user_data = user;
    ASSERT_EQ(mi_session_open(engine_, &sp), MI_OK) << last_error();
    const char* prompts[2] = {q.prompt, q.negative_prompt};
    mi_text_params tp;
    ASSERT_EQ(mi_text_params_init(&tp, sizeof tp), MI_OK);
    tp.prompts = prompts;
    tp.prompts_count = 2;
    ASSERT_EQ(mi_encode_text(engine_, &tp, conds), MI_OK) << last_error();
  }

  // mi_generate's MOSS layout at the request's size: rows {cond}, {uncond, {22}}, guided velocity, exact sigmas.
  static mi_run_params moss_run(const mi_gen_params& q, const mi_run_row rows[2]) {
    mi_run_params rp;
    EXPECT_EQ(mi_run_params_init(&rp, sizeof rp), MI_OK);
    rp.width = q.width;
    rp.height = q.height;
    rp.steps = q.steps;
    rp.rows = rows;
    rp.rows_count = 2;
    rp.guidance = MI_GUIDANCE_MOSS;
    rp.output = mi_sampler_prediction(MI_SAMPLER_EULER, MI_SCHEDULER_MOSS);
    rp.flags = MI_RUN_EXACT_SIGMA;
    return rp;
  }

  mi_engine* engine_ = nullptr;
  std::string model_, dit_;
};

struct Callbacks {
  mi_engine* engine = nullptr;
  bool checked = false;
  mi_status close = MI_OK, end = MI_OK;
};

void end_from_callback(const mi_progress* p, void* user) {
  auto* cb = static_cast<Callbacks*>(user);
  if (p->phase != MI_PHASE_DIT_LOAD || cb->checked) return;
  cb->checked = true;
  cb->close = mi_session_close(cb->engine);
  cb->end = mi_run_end(cb->engine, nullptr);
}

// §7.2 through the C API: an L = 1 MOSS run (distinct conditionings) plans exactly as mi_generate's request, and the
// recipe's euler-on-moss host loop through an adopted staging block, then mi_vae_decode, gives mi_generate's RGB bytes.
// The state errors on the way: a second session or run, mi_generate and mi_encode_text during a run, the ending calls
// from a callback of the running call.
TEST_F(GpuComponentApi, MossRunPlansAndSamplesAsGenerate) {
  const mi_gen_params q = request();
  mi_result* warm = nullptr;
  ASSERT_EQ(mi_generate(engine_, &q, &warm), MI_OK) << last_error();  // loads the models, fills the embedding cache
  mi_result_free(warm);
  mi_result* ref = nullptr;
  ASSERT_EQ(mi_generate(engine_, &q, &ref), MI_OK) << last_error();
  std::size_t ref_bytes = 0;
  const std::uint8_t* ref_rgb = mi_result_rgb(ref, &ref_bytes);
  ASSERT_NE(ref_rgb, nullptr);
  const std::string ref_plan = plan_facts(mi_result_report_json(ref));

  Callbacks cb;
  cb.engine = engine_;
  mi_conditioning* conds[2] = {nullptr, nullptr};
  open_and_encode(q, conds, end_from_callback, &cb);
  mi_session_params sp;
  ASSERT_EQ(mi_session_params_init(&sp, sizeof sp), MI_OK);
  EXPECT_EQ(mi_session_open(engine_, &sp), MI_ERR_BUSY);
  EXPECT_TRUE(has(last_error(), "a session is open")) << last_error();
  mi_conditioning_info ci{};
  ci.struct_size = sizeof ci;
  ASSERT_EQ(mi_conditioning_query(conds[0], &ci), MI_OK);
  EXPECT_EQ(ci.engine_encoded, 1);
  mi_prompt_tokens pt{};
  pt.struct_size = sizeof pt;
  mi_prompt_tokens ref_pt{};
  ref_pt.struct_size = sizeof ref_pt;
  ASSERT_EQ(mi_conditioning_prompt_tokens(conds[0], &pt), MI_OK);
  ASSERT_EQ(mi_result_prompt_tokens(ref, 0, &ref_pt), MI_OK);
  EXPECT_EQ(pt.kept_tokens, ref_pt.kept_tokens);
  EXPECT_EQ(ci.tokens, ref_pt.kept_tokens);

  const std::int32_t skip22[] = {22};
  const mi_run_row rows[2] = {{conds[0], nullptr, 0, 0, 0}, {conds[1], skip22, 1, 0, 0}};
  const mi_run_params rp = moss_run(q, rows);
  mi_step_params none;
  ASSERT_EQ(mi_step_params_init(&none, sizeof none), MI_OK);
  EXPECT_EQ(mi_denoise_step(engine_, &none, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "no run is open")) << last_error();
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  EXPECT_TRUE(cb.checked);
  EXPECT_EQ(cb.close, MI_ERR_BUSY);
  EXPECT_EQ(cb.end, MI_ERR_BUSY);
  EXPECT_EQ(mi_run_begin(engine_, &rp), MI_ERR_BUSY);
  // The conditionings may go once the run has begun.
  mi_conditioning_free(conds[0]);
  mi_conditioning_free(conds[1]);
  mi_result* during = nullptr;
  EXPECT_EQ(mi_generate(engine_, &q, &during), MI_ERR_BUSY);
  EXPECT_TRUE(has(last_error(), "mi_run_end")) << last_error();
  const char* one[1] = {"x"};
  mi_text_params tp;
  ASSERT_EQ(mi_text_params_init(&tp, sizeof tp), MI_OK);
  tp.prompts = one;
  tp.prompts_count = 1;
  mi_conditioning* c1[1] = {nullptr};
  EXPECT_EQ(mi_encode_text(engine_, &tp, c1), MI_ERR_BUSY);

  mi_run_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_run_query(engine_, &info), MI_OK) << last_error();
  EXPECT_EQ(info.outputs, 1);
  EXPECT_EQ(info.rows, 3u);
  EXPECT_EQ(info.schedule_matches, 1);
  EXPECT_EQ(info.steps, q.steps);
  EXPECT_EQ(info.latents, 1);
  EXPECT_EQ(info.control_channels, 0);
  std::size_t n_sigmas = 0;
  const float* s = mi_run_sigmas(engine_, &n_sigmas);
  ASSERT_NE(s, nullptr);
  ASSERT_EQ(n_sigmas, static_cast<std::size_t>(q.steps) + 1);

  // An adopted staging block: page-aligned caller memory, registered as pinned.
  const std::size_t bytes = (info.staging_bytes + 4095) / 4096 * 4096;
  void* block = ::operator new(bytes, std::align_val_t{4096});  // page-aligned (std::aligned_alloc is not in MSVC)
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(mi_run_adopt_staging(engine_, static_cast<char*>(block) + 64, bytes - 64), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_run_adopt_staging(engine_, block, info.staging_bytes - 1), MI_ERR_INVALID_ARGUMENT);
  const mi_status adopted = mi_run_adopt_staging(engine_, block, bytes);
#ifdef _WIN32
  ASSERT_TRUE(adopted == MI_OK || adopted == MI_ERR_NOT_SUPPORTED) << last_error();
#else
  ASSERT_EQ(adopted, MI_OK) << last_error();  // ordinary pages register on Linux
#endif
  if (adopted == MI_OK) {
    EXPECT_EQ(mi_run_adopt_staging(engine_, block, bytes), MI_ERR_INVALID_ARGUMENT);  // once per run
  }
  mi_tensor xv{}, ov[1]{};
  ASSERT_EQ(mi_run_staging(engine_, &xv, ov, 1, nullptr), MI_OK) << last_error();
  if (adopted == MI_OK) {
    EXPECT_GE(static_cast<char*>(xv.data), static_cast<char*>(block));
    EXPECT_LT(static_cast<char*>(ov[0].data), static_cast<char*>(block) + bytes);
  }
  EXPECT_EQ(xv.rank, 3);
  EXPECT_EQ(xv.shape[0], 128);
  EXPECT_EQ(xv.shape[1], info.latent_height);

  // The recipe: x = noise (sigma_0 = 1); per step x = x + h * v, h = fp32(double(s[i+1]) - double(s[i])).
  const std::size_t n = static_cast<std::size_t>(128) * info.latent_height * info.latent_width;
  std::vector<float> x(n);
  ASSERT_EQ(mi_initial_noise(q.seed, MI_NOISE_LAYOUT_OFFICIAL, q.width, q.height, x.data(), n), MI_OK);
  auto* xs = static_cast<float*>(xv.data);
  const auto* vel = static_cast<const float*>(ov[0].data);
  for (int i = 0; i < info.steps; ++i) {
    std::copy(x.begin(), x.end(), xs);
    mi_step_params st;
    ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
    st.sigma = s[i];
    st.x = xv;
    st.out[0] = ov[0];
    mi_step_info si{};
    si.struct_size = sizeof si;
    ASSERT_EQ(mi_denoise_step(engine_, &st, &si), MI_OK) << last_error();
    EXPECT_EQ(si.step, i);
    EXPECT_EQ(si.rows, i == 0 ? 3u : si.rows);
    EXPECT_EQ(si.on_schedule, 1);
    const float h = static_cast<float>(static_cast<double>(s[i + 1]) - static_cast<double>(s[i]));
    for (std::size_t k = 0; k < n; ++k) {
      const float t = h * vel[k];
      x[k] = x[k] + t;
    }
  }
  if (adopted == MI_OK) {
    EXPECT_EQ(mi_run_adopt_staging(engine_, block, bytes), MI_ERR_INVALID_ARGUMENT);  // after a step
  }
  mi_step_params bad;
  ASSERT_EQ(mi_step_params_init(&bad, sizeof bad), MI_OK);
  bad.sigma = 0.37f;  // not a schedule entry: refused under MI_RUN_EXACT_SIGMA, the run still usable
  bad.x = xv;
  bad.out[0] = ov[0];
  EXPECT_EQ(mi_denoise_step(engine_, &bad, nullptr), MI_ERR_INVALID_ARGUMENT);
  mi_tensor stray = host_f32(x.data(), 128, info.latent_height, info.latent_width);
  bad.sigma = s[0];
  bad.x = stray;  // host memory that is not the staging
  EXPECT_EQ(mi_denoise_step(engine_, &bad, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "staging")) << last_error();

  mi_result* done = nullptr;
  ASSERT_EQ(mi_run_end(engine_, &done), MI_OK) << last_error();
  ::operator delete(block, std::align_val_t{4096});  // mi_run_end unregistered it
  ASSERT_NE(done, nullptr);
  EXPECT_STREQ(mi_result_text(done, MI_TEXT_COMPONENT_NOTE), "");
  EXPECT_EQ(mi_result_rgb(done, nullptr), nullptr);
  EXPECT_EQ(mi_result_steps_run(done), q.steps);
  mi_memory_stats mem{};
  mem.struct_size = sizeof mem;
  ASSERT_EQ(mi_result_memory(done, &mem), MI_OK);
  EXPECT_EQ(mem.dit_loads, 0);
  mi_tier_info tier{};
  tier.struct_size = sizeof tier;
  ASSERT_EQ(mi_result_tier(done, &tier), MI_OK);
  EXPECT_EQ(tier.effective, MI_TIER_HIGH);
  EXPECT_EQ(tier.attention, MI_ATTENTION_Q8_FP8);
  const std::string report = mi_result_report_json(done);
  const mi::json::Value doc = mi::json::parse(report);
  EXPECT_EQ(doc.at("component").at("calls").as_int(), q.steps);
  EXPECT_EQ(doc.at("component").at("off_schedule_calls").as_int(), 0);
  EXPECT_EQ(plan_facts(report), ref_plan);
  mi_result_free(done);
  mi_result* again = reinterpret_cast<mi_result*>(&info);
  EXPECT_EQ(mi_run_end(engine_, &again), MI_OK);
  EXPECT_EQ(again, nullptr);
  EXPECT_EQ(mi_run_query(engine_, &info), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "no run is open")) << last_error();

  // mi_generate's final decode bit for bit, from host latents.
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(q.width) * q.height * 3);
  mi_tensor lat = host_f32(x.data(), 128, info.latent_height, info.latent_width);
  mi_tensor out{};
  out.data = rgb.data();
  out.memory = MI_MEMORY_HOST;
  out.dtype = MI_DTYPE_U8;
  out.rank = 3;
  out.shape[0] = q.height;
  out.shape[1] = q.width;
  out.shape[2] = 3;
  ASSERT_EQ(mi_vae_decode(engine_, &lat, &out), MI_OK) << last_error();
  ASSERT_EQ(rgb.size(), ref_bytes);
  std::size_t diff = 0;
  for (std::size_t k = 0; k < rgb.size(); ++k) diff += rgb[k] != ref_rgb[k];
  EXPECT_EQ(diff, 0u) << "of " << rgb.size() << " RGB bytes";
  mi_result_free(ref);

  // mi_generate between runs, then close twice.
  mi_result* between = nullptr;
  EXPECT_EQ(mi_generate(engine_, &q, &between), MI_OK) << last_error();
  mi_result_free(between);
  EXPECT_EQ(mi_session_close(engine_), MI_OK);
  EXPECT_EQ(mi_session_close(engine_), MI_OK);
  EXPECT_EQ(mi_vae_decode(engine_, &lat, &out), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "no session is open")) << last_error();
  // A new session after a closed one.
  ASSERT_EQ(mi_session_open(engine_, nullptr), MI_OK) << last_error();
  EXPECT_EQ(mi_session_close(engine_), MI_OK);
}

// A step at sigmas[0] on the run's staging views (host mode).
mi_step_params staging_step(mi_engine* e, float sigma) {
  mi_step_params st;
  mi_step_params_init(&st, sizeof st);
  mi_tensor xv{}, ov[1]{};
  mi_run_staging(e, &xv, ov, 1, nullptr);
  st.sigma = sigma;
  st.x = xv;
  st.out[0] = ov[0];
  return st;
}

// Returns once a step call of another thread is in flight: a probe step on this thread is MI_ERR_BUSY (a probe that
// gets the gate first runs a whole step, which the other thread's retry loop absorbs).
void until_a_step_runs(mi_engine* e, float sigma) {
  const mi_step_params st = staging_step(e, sigma);
  while (mi_denoise_step(e, &st, nullptr) != MI_ERR_BUSY) {
  }
}

// mi_generate's MOSS run at q's size, its conditionings freed once it has begun.
void begin_moss_run(mi_engine* e, const mi_gen_params& q, mi_conditioning* conds[2], std::uint32_t flags = 0) {
  const std::int32_t skip22[] = {22};
  const mi_run_row rows[2] = {{conds[0], nullptr, 0, 0, 0}, {conds[1], skip22, 1, 0, 0}};
  mi_run_params rp;
  ASSERT_EQ(mi_run_params_init(&rp, sizeof rp), MI_OK);
  rp.width = q.width;
  rp.height = q.height;
  rp.steps = q.steps;
  rp.rows = rows;
  rp.rows_count = 2;
  rp.guidance = MI_GUIDANCE_MOSS;
  rp.output = MI_STEP_OUTPUT_GUIDED_VELOCITY;
  rp.flags = flags;
  ASSERT_EQ(mi_run_begin(e, &rp), MI_OK) << last_error();
}

// An interrupt: mi_run_end from another thread while a step runs cancels it (MI_CANCELLED) and ends the run; the next
// call is "no run is open", and a new run of the same request loads nothing.
TEST_F(GpuComponentApi, RunEndFromAnotherThreadMidStep) {
  const mi_gen_params q = request(1024, 50);
  mi_conditioning* conds[2] = {nullptr, nullptr};
  open_and_encode(q, conds);
  begin_moss_run(engine_, q, conds);
  std::size_t n_sigmas = 0;
  const float* s = mi_run_sigmas(engine_, &n_sigmas);
  ASSERT_NE(s, nullptr);
  mi_engine* const e = engine_;
  mi_status last = MI_OK;
  std::string message;
  int completed = 0;
  std::thread worker([&] {
    for (int i = 0; i + 1 < static_cast<int>(n_sigmas); ++i) {
      const mi_step_params st = staging_step(e, s[i]);
      do {
        last = mi_denoise_step(e, &st, nullptr);
      } while (last == MI_ERR_BUSY);  // a probe of the main thread had the gate
      if (last != MI_OK) {
        message = last_error();
        return;
      }
      ++completed;
    }
  });
  until_a_step_runs(e, s[0]);
  mi_result* done = nullptr;
  EXPECT_EQ(mi_run_end(e, &done), MI_OK) << last_error();
  worker.join();
  EXPECT_TRUE(last == MI_CANCELLED || (last == MI_ERR_INVALID_ARGUMENT && has(message, "no run is open"))) << message;
  ASSERT_NE(done, nullptr);
  EXPECT_GE(mi_result_steps_run(done), completed > 0 ? 1 : 0);
  mi_result_free(done);
  const mi_step_params st = staging_step(e, s[0]);
  EXPECT_EQ(mi_denoise_step(e, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "no run is open")) << last_error();

  // The session is reusable without reloading weights.
  begin_moss_run(e, q, conds);
  const mi_step_params again = staging_step(e, s[0]);
  EXPECT_EQ(mi_denoise_step(e, &again, nullptr), MI_OK) << last_error();
  ASSERT_EQ(mi_run_end(e, &done), MI_OK);
  mi_memory_stats mem{};
  mem.struct_size = sizeof mem;
  ASSERT_EQ(mi_result_memory(done, &mem), MI_OK);
  EXPECT_EQ(mem.dit_loads, 0);
  mi_result_free(done);
  mi_conditioning_free(conds[0]);
  mi_conditioning_free(conds[1]);
}

// mi_cancel on a step: MI_CANCELLED and the run still usable; a cancel between steps does not carry over.
TEST_F(GpuComponentApi, CancelAStep) {
  const mi_gen_params q = request(1024, 50);
  mi_conditioning* conds[2] = {nullptr, nullptr};
  open_and_encode(q, conds);
  begin_moss_run(engine_, q, conds);
  std::size_t n_sigmas = 0;
  const float* s = mi_run_sigmas(engine_, &n_sigmas);
  ASSERT_NE(s, nullptr);
  mi_engine* const e = engine_;
  mi_status status = MI_OK;
  std::atomic<bool> done{false};
  std::thread worker([&] {
    const mi_step_params st = staging_step(e, s[1]);
    do {
      status = mi_denoise_step(e, &st, nullptr);
    } while (status == MI_ERR_BUSY || status == MI_OK);  // until a cancel lands on one of its steps
    done.store(true);
  });
  const mi_step_params probe = staging_step(e, s[0]);
  while (!done.load()) {
    // A probe that finds the worker's step in flight cancels it (one that finds the gate free runs a step itself).
    if (mi_denoise_step(e, &probe, nullptr) == MI_ERR_BUSY) {
      EXPECT_EQ(mi_cancel(e), MI_OK);
    }
  }
  worker.join();
  EXPECT_EQ(status, MI_CANCELLED) << last_error();
  EXPECT_EQ(mi_cancel(e), MI_OK);  // between steps: a no-op
  const mi_step_params st = staging_step(e, s[0]);
  mi_step_info info{};
  info.struct_size = sizeof info;
  EXPECT_EQ(mi_denoise_step(e, &st, &info), MI_OK) << last_error();
  EXPECT_EQ(info.step, 0);
  EXPECT_EQ(mi_run_end(e, nullptr), MI_OK);
  mi_conditioning_free(conds[0]);
  mi_conditioning_free(conds[1]);
}

// mi_engine_free with an open session and a step running on another thread: the step is cancelled and waited for, then
// the engine is freed (the worker's step returns MI_CANCELLED or completes; it makes no later call).
TEST_F(GpuComponentApi, EngineFreeDuringAStep) {
  const mi_gen_params q = request(1024, 50);
  mi_conditioning* conds[2] = {nullptr, nullptr};
  open_and_encode(q, conds);
  begin_moss_run(engine_, q, conds);
  mi_conditioning_free(conds[0]);
  mi_conditioning_free(conds[1]);
  std::size_t n_sigmas = 0;
  const float* s = mi_run_sigmas(engine_, &n_sigmas);
  ASSERT_NE(s, nullptr);
  const float sigma0 = s[0];
  mi_engine* const e = engine_;
  engine_ = nullptr;  // freed below
  mi_status status = MI_OK;
  const mi_step_params st = staging_step(e, sigma0);
  std::thread worker([&] {
    do {
      status = mi_denoise_step(e, &st, nullptr);
    } while (status == MI_ERR_BUSY);
  });
  until_a_step_runs(e, sigma0);
  mi_engine_free(e);
  worker.join();
  EXPECT_TRUE(status == MI_CANCELLED || status == MI_OK) << mi_status_string(status);
}

struct GenerateEvents {
  std::atomic<int> denoise_steps{0};
  mi_engine* engine = nullptr;
  bool free_from_callback = false;
  std::string message_before, message_after;
};

void on_generate_progress(const mi_progress* p, void* user) {
  auto* ev = static_cast<GenerateEvents*>(user);
  if (p->phase == MI_PHASE_DENOISE && p->step > 0) ++ev->denoise_steps;
  if (ev->free_from_callback && p->phase == MI_PHASE_DENOISE && p->step == 1) {
    ev->free_from_callback = false;
    ev->message_before = mi_last_error_message();
    mi_engine_free(ev->engine);  // forbidden here: a silent no-op
    ev->message_after = mi_last_error_message();
  }
}

// mi_engine_free from a callback of the running job does nothing (the last error unchanged, the engine usable,
// sessions still open); with an open session, a free during a generation on another thread cancels it promptly.
TEST_F(GpuComponentApi, EngineFreeFromACallbackAndDuringAGenerate) {
  mi_gen_params q = request(512, 4);
  GenerateEvents ev;
  ev.engine = engine_;
  ev.free_from_callback = true;
  q.progress_callback = on_generate_progress;
  q.user_data = &ev;
  EXPECT_EQ(mi_session_close(nullptr), MI_ERR_INVALID_ARGUMENT);  // the error the callback must leave alone
  mi_result* r = nullptr;
  ASSERT_EQ(mi_generate(engine_, &q, &r), MI_OK) << last_error();
  EXPECT_EQ(ev.message_before, "engine is NULL.");
  EXPECT_EQ(ev.message_after, ev.message_before);
  EXPECT_NE(mi_result_rgb(r, nullptr), nullptr);
  mi_result_free(r);
  ASSERT_EQ(mi_session_open(engine_, nullptr), MI_OK) << last_error();  // the free left no state behind

  // A free with the session open while mi_generate runs on another thread: the generation is cancelled.
  mi_gen_params long_q = request(1024, 50);
  GenerateEvents run;
  long_q.progress_callback = on_generate_progress;
  long_q.user_data = &run;
  mi_engine* const e = engine_;
  engine_ = nullptr;
  mi_status status = MI_OK;
  mi_result* cancelled = nullptr;
  std::thread generate([&] { status = mi_generate(e, &long_q, &cancelled); });
  while (run.denoise_steps.load() < 2) std::this_thread::yield();
  const auto t0 = std::chrono::steady_clock::now();
  mi_engine_free(e);
  const double free_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  generate.join();
  EXPECT_EQ(status, MI_CANCELLED);
  ASSERT_NE(cancelled, nullptr);
  EXPECT_EQ(mi_result_cancelled(cancelled), 1);
  EXPECT_LT(mi_result_steps_run(cancelled), 50);
  EXPECT_LT(free_s, 10.0) << "the free waited out the generation";
  mi_result_free(cancelled);
}

bool registered(const void* p) {
  cudaPointerAttributes a{};
  const bool known = cudaPointerGetAttributes(&a, p) == cudaSuccess;
  cudaGetLastError();
  return known && a.type == cudaMemoryTypeHost;
}

// The adopted block's lifecycle: registered while a run uses it, unregistered by mi_run_end and by a session close
// without mi_run_end; the same block adopted again by a later run. And the §7.1 step refusals on device tensors.
TEST_F(GpuComponentApi, AdoptedStagingLifecycleAndStepRefusals) {
  const mi_gen_params q = request(512, 6);
  mi_conditioning* conds[2] = {nullptr, nullptr};
  open_and_encode(q, conds);
  begin_moss_run(engine_, q, conds);
  mi_run_info info{};
  info.struct_size = sizeof info;
  ASSERT_EQ(mi_run_query(engine_, &info), MI_OK);
  const std::size_t bytes = (info.staging_bytes + 4095) / 4096 * 4096 + 4096;
  void* block = ::operator new(bytes, std::align_val_t{4096});
  ASSERT_EQ(mi_run_adopt_staging(engine_, block, bytes), MI_OK) << last_error();
  EXPECT_TRUE(registered(block));
  EXPECT_EQ(mi_run_end(engine_, nullptr), MI_OK);
  EXPECT_FALSE(registered(block));
  begin_moss_run(engine_, q, conds);
  EXPECT_EQ(mi_run_adopt_staging(engine_, block, bytes), MI_OK) << last_error();  // the same block again
  EXPECT_TRUE(registered(block));

  // Step refusals (the run stays usable): mixed memory kinds, an output overlapping x, an output the run lacks, control
  // without an adapter.
  const std::size_t n = static_cast<std::size_t>(128) * info.latent_height * info.latent_width;
  float* dev = nullptr;
  ASSERT_EQ(cudaMalloc(&dev, 3 * n * sizeof(float)), cudaSuccess);
  const auto device_f32 = [&](float* p) {
    mi_tensor t{};
    t.data = p;
    t.memory = MI_MEMORY_DEVICE;
    t.dtype = MI_DTYPE_F32;
    t.rank = 3;
    t.shape[0] = 128;
    t.shape[1] = info.latent_height;
    t.shape[2] = info.latent_width;
    return t;
  };
  std::size_t n_sigmas = 0;
  const float* s = mi_run_sigmas(engine_, &n_sigmas);
  mi_step_params st = staging_step(engine_, s[0]);
  st.out[0] = device_f32(dev);
  EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "all be device or all host")) << last_error();
  st.x = device_f32(dev);
  EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "out[0] overlaps x")) << last_error();
  st.out[0] = device_f32(dev + n);
  st.out[1] = device_f32(dev + 2 * n);
  EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "out[1]: the run has no such output")) << last_error();
  st.out[1] = mi_tensor{};
  st.control = device_f32(dev + 2 * n);
  EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "control: the session has no control adapter")) << last_error();
  st.control = mi_tensor{};
  EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_OK) << last_error();  // device tensors, the run still usable
  cudaFree(dev);

  // A session close without mi_run_end unregisters the block too.
  EXPECT_EQ(mi_session_close(engine_), MI_OK);
  EXPECT_FALSE(registered(block));

  // Memory the caller registered itself is used as it is: the run's end leaves it registered.
  ASSERT_EQ(cudaHostRegister(block, bytes, cudaHostRegisterPortable), cudaSuccess);
  ASSERT_EQ(mi_session_open(engine_, nullptr), MI_OK) << last_error();
  begin_moss_run(engine_, q, conds);
  EXPECT_EQ(mi_run_adopt_staging(engine_, block, bytes), MI_OK) << last_error();
  EXPECT_EQ(mi_run_end(engine_, nullptr), MI_OK);
  EXPECT_TRUE(registered(block));
  EXPECT_EQ(cudaHostUnregister(block), cudaSuccess);
  EXPECT_EQ(mi_session_close(engine_), MI_OK);
  ::operator delete(block, std::align_val_t{4096});
  mi_conditioning_free(conds[0]);
  mi_conditioning_free(conds[1]);
}

}  // namespace
