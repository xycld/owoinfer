// mossinfer.h's component API on the GPU (design §7.2), through the C API only: every comparison is against
// mi_generate on the same engine in the same process (its RGB bytes; latents of two component runs compared directly).
// Real weights (GTEST_SKIP when absent): the SFT bundle with the INT8 DiT, tier high, q8_fp8 attention, 512^2.
//   GpuComponentRuns.*  table A (euler on moss in every exchange mode and output mode, euler on simple, ab2 and heun on
//                       karras with explicit corrector steps, repeated sigmas through the lookup), off-schedule calls
//                       and a host dpmpp_2m (table B), MOSS windows (SLG, skip windows, CFG truncation) through guided
//                       outputs and the host's per-op combine, HOST rows (1, 2, 4, 8 rows, masks), latent batches of 2
//                       and 4 against single images (and the other numerics), crop calls, control per run and per call,
//                       a runtime LoRA, img2img through mi_vae_encode, foreign conditioning, ComfyUI's FLOW
//                       convention, cancels, the ending race, device buffers after a failed step, the batch and crop
//                       residency refusals, the VAE oracle against ComfyUI's VAE; GUIDED_DENOISED on moss against the
//                       host's denoised combine, DPM2 and UniPC-style host loops, the step cache at L = 2, abandon /
//                       cancel / failure then a new run, and planning and memory (the plan JSON in balanced and in
//                       min_memory, arena creations and DiT loads over 512^2 / 768^2 / 512^2, the device peak, the
//                       session's max_rows and max_latents not changing a run's plan).
// Runs under the sanitizers: MOSS_RACE_ITERATIONS=1000 with ASan / TSan builds (EndingRaceIterations, the cancels),
// MOSS_FREE_AFTER_FAIL=1 under compute-sanitizer memcheck (DeviceBuffersIdleAfterAFailedStep).
// This file's host arithmetic must round every fp32 operation separately (CMakeLists.txt: strict FP sources).
#include "mossinfer/mossinfer.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "util/json.h"
#include "util/npy.h"

namespace {

namespace fs = std::filesystem;

std::string last_error() { return mi_last_error_message(); }
bool has(const std::string& s, const std::string& fragment) { return s.find(fragment) != std::string::npos; }

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
const fs::path kCommunity = "/workspace/Models/community/comfy-org";
const fs::path kInt8 = kCommunity / "moss_image_sft_int8_convrot.safetensors";
const fs::path kBf16 = kCommunity / "moss_image_sft_bf16.safetensors";
const fs::path kControl = kCommunity / "moss_image_color_depth_control.safetensors";

constexpr int kSize = 512;
constexpr std::int64_t kLat = kSize / 16;
constexpr std::size_t kN = static_cast<std::size_t>(MI_LATENT_CHANNELS) * kLat * kLat;
constexpr std::int32_t kSkip22[] = {22};

// ---------------------------------------------------------------- fp32 arithmetic, one rounding per op

using Vec = std::vector<float>;

// x = s0 * noise + (1 - s0) * z (z empty: zeros): flow_noise_scaling, mi_generate's start.
Vec start_latents(const Vec& noise, float s0, const Vec& z = {}) {
  Vec x(noise.size());
  const float keep = 1.0f - s0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float a = s0 * noise[i];
    const float b = keep * (z.empty() ? 0.0f : z[i]);
    x[i] = a + b;
  }
  return x;
}

// CFG in velocity space: v = u + g * (c - u).
void cfg_velocity(const float* c, const float* u, float g, float* v, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const float d = c[i] - u[i];
    const float t = g * d;
    v[i] = u[i] + t;
  }
}

// guide_prediction() from MOSS rows on the host: velocity (cfg: u + g (c - u); SLG: p + s (c - k) with p the CFG
// result, or the SLG row without CFG) or denoised (D = x - v sigma per row, Du + (Dc - Du) g, D + (Dc - Dk) s).
void guide_on_host(bool velocity, const Vec& x, float sigma, const Vec& c, const Vec* u, const Vec* k, float g, float s,
                   Vec& out) {
  const std::size_t n = x.size();
  out.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    float o;
    if (velocity) {
      if (u != nullptr) {
        const float d = c[i] - (*u)[i];
        const float t = g * d;
        o = (*u)[i] + t;
      } else {
        o = k != nullptr ? (*k)[i] : c[i];
      }
      if (k != nullptr) {
        const float d = c[i] - (*k)[i];
        const float t = s * d;
        o = o + t;
      }
    } else {
      const float vc = c[i] * sigma;
      const float dc = x[i] - vc;
      if (u != nullptr) {
        const float vu = (*u)[i] * sigma;
        const float du = x[i] - vu;
        const float d = dc - du;
        const float t = d * g;
        o = du + t;
      } else {
        o = dc;
      }
      if (k != nullptr) {
        const float vk = (*k)[i] * sigma;
        const float dk = x[i] - vk;
        const float d = dc - dk;
        const float t = d * s;
        o = o + t;
      }
    }
    out[i] = o;
  }
}

// A sampler's model: the guided prediction (velocity or denoised, as the sampler needs) at (x, sigma); `step` is the
// call's explicit position, or -1 for the run's lookup.
using Model = std::function<void(const Vec& x, float sigma, int step, Vec& pred)>;

// Table A (design §5.3) on the host, transcribed from sampler_kernels.cu. `velocity_euler`: euler on moss.
Vec sample(mi_sampler sampler, bool velocity_euler, const float* s, std::size_t count, Vec x, const Model& model) {
  const int steps = static_cast<int>(count) - 1;
  Vec pred, prev, d, x2, pred2;
  double h_prev = 0;
  for (int i = 0; i < steps; ++i) {
    const float si = s[i], sn = s[i + 1];
    const double hd = static_cast<double>(sn) - static_cast<double>(si);
    const float h = static_cast<float>(hd);
    const float dt = sn - si;
    model(x, si, -1, pred);
    if (sampler == MI_SAMPLER_EULER && velocity_euler) {
      for (std::size_t k = 0; k < x.size(); ++k) {
        const float t = h * pred[k];
        x[k] = x[k] + t;
      }
    } else if (sampler == MI_SAMPLER_AB2) {
      if (i == 0) {
        for (std::size_t k = 0; k < x.size(); ++k) {
          const float t = h * pred[k];
          x[k] = x[k] + t;
        }
      } else {
        const double r = hd / (2.0 * h_prev);
        const float a = static_cast<float>(1.0 + r), b = static_cast<float>(r);
        for (std::size_t k = 0; k < x.size(); ++k) {
          const float av = a * pred[k];
          const float bv = b * prev[k];
          const float dd = av - bv;
          const float t = h * dd;
          x[k] = x[k] + t;
        }
      }
      h_prev = hd;
      prev = pred;
    } else if (sampler == MI_SAMPLER_EULER || (sampler == MI_SAMPLER_HEUN && sn == 0.0f)) {
      for (std::size_t k = 0; k < x.size(); ++k) {
        const float e = x[k] - pred[k];
        const float dk = e / si;
        const float t = dk * dt;
        x[k] = x[k] + t;
      }
    } else if (sampler == MI_SAMPLER_HEUN) {
      d.resize(x.size());
      x2.resize(x.size());
      for (std::size_t k = 0; k < x.size(); ++k) {
        const float e = x[k] - pred[k];
        d[k] = e / si;
        const float t = d[k] * dt;
        x2[k] = x[k] + t;
      }
      model(x2, sn, i, pred2);  // the corrector at s[i + 1] gives its step
      for (std::size_t k = 0; k < x.size(); ++k) {
        const float e = x2[k] - pred2[k];
        const float d2 = e / sn;
        const float sum = d[k] + d2;
        const float dp = sum * 0.5f;
        const float t = dp * dt;
        x[k] = x[k] + t;
      }
    } else {
      ADD_FAILURE() << "sampler " << sampler << " is not in table A";
      return x;
    }
  }
  return x;
}

// ---------------------------------------------------------------- engine-side helpers

Vec noise(std::int64_t seed, int width = kSize, int height = kSize) {
  Vec v(static_cast<std::size_t>(MI_LATENT_CHANNELS) * (height / 16) * (width / 16));
  EXPECT_EQ(mi_initial_noise(seed, MI_NOISE_LAYOUT_OFFICIAL, width, height, v.data(), v.size()), MI_OK) << last_error();
  return v;
}

mi_tensor tensor(void* data, mi_memory memory, mi_dtype dtype, std::initializer_list<std::int64_t> shape) {
  mi_tensor t{};
  t.data = data;
  t.memory = memory;
  t.dtype = dtype;
  t.rank = static_cast<std::int32_t>(shape.size());
  int i = 0;
  for (const std::int64_t d : shape) t.shape[i++] = d;
  return t;
}

std::vector<std::uint8_t> decode(mi_engine* e, const Vec& latents, int width = kSize, int height = kSize) {
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
  Vec copy = latents;
  const mi_tensor lat = tensor(copy.data(), MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, height / 16, width / 16});
  const mi_tensor out = tensor(rgb.data(), MI_MEMORY_HOST, MI_DTYPE_U8, {height, width, 3});
  EXPECT_EQ(mi_vae_decode(e, &lat, &out), MI_OK) << last_error();
  return rgb;
}

std::size_t differing(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
  return d;
}

std::size_t differing(const Vec& a, const Vec& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return d;
}

Vec slice(const Vec& v, std::size_t index, std::size_t n = kN) {
  const auto at = [&](std::size_t i) { return v.begin() + static_cast<std::ptrdiff_t>(i * n); };
  return Vec(at(index), at(index + 1));
}

constexpr const char* kPrompt = "a red apple on a wooden table, studio light";

mi_gen_params request(int steps = 6, std::int64_t seed = 7, const char* prompt = kPrompt,
                      const char* negative = "blurry") {
  mi_gen_params p;
  EXPECT_EQ(mi_gen_params_init(&p, sizeof p), MI_OK);
  p.prompt = prompt;
  p.negative_prompt = negative;
  p.width = p.height = kSize;
  p.steps = steps;
  p.seed = seed;
  return p;
}

// mi_generate's RGB bytes; `out` (optional) keeps the result.
std::vector<std::uint8_t> generate(mi_engine* e, const mi_gen_params& q, mi_result** out = nullptr) {
  mi_result* r = nullptr;
  EXPECT_EQ(mi_generate(e, &q, &r), MI_OK) << last_error();
  std::size_t bytes = 0;
  const std::uint8_t* rgb = r != nullptr ? mi_result_rgb(r, &bytes) : nullptr;
  std::vector<std::uint8_t> v;
  if (rgb != nullptr) v.assign(rgb, rgb + bytes);
  if (out != nullptr) {
    *out = r;
  } else {
    mi_result_free(r);
  }
  return v;
}

std::vector<int> cached_steps(const mi_result* r) {
  std::size_t n = 0;
  const std::int32_t* s = mi_result_accel_cached_steps(r, &n);
  return s != nullptr ? std::vector<int>(s, s + n) : std::vector<int>{};
}

// Conditionings owned by the test.
struct Conds {
  std::vector<mi_conditioning*> c;
  Conds() = default;
  Conds(const Conds&) = delete;
  Conds& operator=(const Conds&) = delete;
  Conds(Conds&& o) noexcept : c(std::move(o.c)) { o.c.clear(); }
  ~Conds() {
    for (mi_conditioning* x : c) mi_conditioning_free(x);
  }
  mi_conditioning* operator[](std::size_t i) const { return c.at(i); }
};

Conds encode(mi_engine* e, const std::vector<const char*>& prompts) {
  mi_text_params tp;
  EXPECT_EQ(mi_text_params_init(&tp, sizeof tp), MI_OK);
  tp.prompts = prompts.data();
  tp.prompts_count = prompts.size();
  Conds out;
  out.c.assign(prompts.size(), nullptr);
  EXPECT_EQ(mi_encode_text(e, &tp, out.c.data()), MI_OK) << last_error();
  return out;
}

mi_status open_session(mi_engine* e, int max_rows = 3, int max_latents = 1, const char* control_adapter = nullptr,
                       mi_progress_fn progress = nullptr, void* user = nullptr) {
  mi_session_params sp;
  EXPECT_EQ(mi_session_params_init(&sp, sizeof sp), MI_OK);
  sp.max_rows = max_rows;
  sp.max_latents = max_latents;
  sp.control_adapter = control_adapter;
  sp.progress_callback = progress;
  sp.user_data = user;
  return mi_session_open(e, &sp);
}

// A run's parameters at 512^2 with `rows` (the caller keeps them alive).
mi_run_params run_params(const std::vector<mi_run_row>& rows, mi_guidance guidance, mi_step_output output,
                         int steps = 6) {
  mi_run_params rp;
  EXPECT_EQ(mi_run_params_init(&rp, sizeof rp), MI_OK);
  rp.width = rp.height = kSize;
  rp.steps = steps;
  rp.rows = rows.data();
  rp.rows_count = rows.size();
  rp.guidance = guidance;
  rp.output = output;
  rp.flags = MI_RUN_EXACT_SIGMA;
  return rp;
}

mi_run_row row(const mi_conditioning* c, const std::int32_t* skip = nullptr, std::size_t skip_count = 0, int latent = 0) {
  return mi_run_row{c, skip, skip_count, latent, 0};
}

Vec run_sigmas(mi_engine* e) {
  std::size_t n = 0;
  const float* s = mi_run_sigmas(e, &n);
  EXPECT_NE(s, nullptr) << last_error();
  return s != nullptr ? Vec(s, s + n) : Vec{};
}

mi_run_info run_info(mi_engine* e) {
  mi_run_info info{};
  info.struct_size = sizeof info;
  EXPECT_EQ(mi_run_query(e, &info), MI_OK) << last_error();
  return info;
}

// Ends the run; returns the result's report JSON ("" without a result) and frees the result unless `out` takes it.
std::string end_run(mi_engine* e, mi_result** out = nullptr) {
  mi_result* r = nullptr;
  EXPECT_EQ(mi_run_end(e, &r), MI_OK) << last_error();
  std::string report = r != nullptr ? mi_result_report_json(r) : "";
  if (out != nullptr) {
    *out = r;
  } else {
    mi_result_free(r);
  }
  return report;
}

// ---------------------------------------------------------------- the exchange modes

enum class Mode {
  kHost,       // the run's pinned staging (mi_run_staging)
  kLegacy,     // device tensors, cuda_stream NULL (the legacy default stream)
  kStream,     // device tensors on the host's own non-blocking stream
  kPerThread,  // device tensors on cudaStreamPerThread
};
const char* mode_name(Mode m) {
  switch (m) {
    case Mode::kHost: return "host staging";
    case Mode::kLegacy: return "device, legacy stream";
    case Mode::kStream: return "device, foreign stream";
    case Mode::kPerThread: return "device, per-thread stream";
  }
  return "?";
}

// One model evaluation's inputs: sigma, an explicit step or -1, the HOST row mask, MI_STEP_* flags, a crop's latent
// size (0: the run's), the per-call control latent (null: the run's).
struct Call {
  float sigma = 0;
  int step = -1;
  std::uint32_t rows = 0;
  std::uint32_t flags = 0;
  std::int64_t h = 0, w = 0;
  const Vec* control = nullptr;
};

// The open run's tensors in one exchange mode. The host side always holds plain vectors: x in, the outputs out. In
// device mode the host's copies run on its stream before and after the call, so the engine's stream handoff orders
// them (a missing wait would read a stale x or hand back stale outputs).
class Exchange {
 public:
  Exchange(mi_engine* e, Mode mode) : e_(e), mode_(mode) {
    info_ = run_info(e);
    mi_tensor outs[MI_MAX_ROWS]{};
    EXPECT_EQ(mi_run_staging(e, &xv_, outs, MI_MAX_ROWS, &cv_), MI_OK) << last_error();
    for (int k = 0; k < info_.outputs; ++k) ov_[k] = outs[k];
    n_ = static_cast<std::size_t>(MI_LATENT_CHANNELS) * info_.latent_height * info_.latent_width;
    out_.assign(static_cast<std::size_t>(info_.outputs), Vec{});
    if (mode == Mode::kHost) return;
    if (mode == Mode::kStream) {
      EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
    }
    if (mode == Mode::kPerThread) stream_ = cudaStreamPerThread;
    EXPECT_EQ(cudaMalloc(&dx_, info_.latents * n_ * sizeof(float)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&dout_, info_.outputs * n_ * sizeof(float)), cudaSuccess);
    if (info_.control_channels > 0) {
      const std::size_t c = static_cast<std::size_t>(info_.latents) * info_.control_channels * info_.latent_height *
                            info_.latent_width;
      EXPECT_EQ(cudaMalloc(&dctl_, c * sizeof(float)), cudaSuccess);
    }
  }
  ~Exchange() {
    cudaFree(dx_);
    cudaFree(dout_);
    cudaFree(dctl_);
    if (mode_ == Mode::kStream) cudaStreamDestroy(stream_);
  }
  Exchange(const Exchange&) = delete;
  Exchange& operator=(const Exchange&) = delete;

  const mi_run_info& info() const { return info_; }
  const Vec& out(int k) const { return out_.at(static_cast<std::size_t>(k)); }
  const mi_step_info& last() const { return last_; }

  // One mi_denoise_step: x [L, 128, h, w] (h x w the call's), the outputs it evaluated copied into out(k).
  mi_status call(const Call& c, const Vec& x) {
    const std::int64_t h = c.h > 0 ? c.h : info_.latent_height, w = c.w > 0 ? c.w : info_.latent_width;
    const std::int64_t L = info_.latents, C = info_.control_channels;
    const std::size_t n = static_cast<std::size_t>(MI_LATENT_CHANNELS) * h * w;
    EXPECT_EQ(x.size(), static_cast<std::size_t>(L) * n);
    mi_step_params st;
    EXPECT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
    st.sigma = c.sigma;
    st.step = c.step;
    st.rows = c.rows;
    st.flags = c.flags;
    const bool device = mode_ != Mode::kHost;
    const mi_memory mem = device ? MI_MEMORY_DEVICE : MI_MEMORY_HOST;
    const auto latent_shape = [&](void* p, std::int64_t channels) {
      return L == 1 ? tensor(p, mem, MI_DTYPE_F32, {channels, h, w}) : tensor(p, mem, MI_DTYPE_F32, {L, channels, h, w});
    };
    void* xp = device ? dx_ : xv_.data;
    void* cp = device ? dctl_ : cv_.data;
    const std::size_t cbytes = c.control != nullptr ? c.control->size() * sizeof(float) : 0;
    if (device) {
      st.cuda_stream = stream_;
      EXPECT_EQ(cudaMemcpyAsync(xp, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), cudaSuccess);
      if (c.control != nullptr) {
        EXPECT_EQ(cudaMemcpyAsync(cp, c.control->data(), cbytes, cudaMemcpyHostToDevice, stream_), cudaSuccess);
      }
    } else {
      std::memcpy(xp, x.data(), x.size() * sizeof(float));
      if (c.control != nullptr) std::memcpy(cp, c.control->data(), cbytes);
    }
    st.x = latent_shape(xp, MI_LATENT_CHANNELS);
    if (c.control != nullptr) st.control = latent_shape(cp, C);
    // Every output slot poisoned (NaN) before the call, so a slot the call should write cannot pass on a value an
    // earlier call left there (a masked call writing row k into another slot).
    if (device) {
      EXPECT_EQ(cudaMemsetAsync(dout_, 0xFF, static_cast<std::size_t>(info_.outputs) * n_ * sizeof(float), stream_),
                cudaSuccess);
    } else {
      for (int k = 0; k < info_.outputs; ++k) {
        if (ov_[k].data != nullptr) std::memset(ov_[k].data, 0xFF, n_ * sizeof(float));
      }
    }
    for (int k = 0; k < info_.outputs; ++k) {
      if (ov_[k].data == nullptr) continue;  // a slot the run lacks (MOSS out[1] without an uncond row): zeroed
      void* p = device ? static_cast<void*>(static_cast<float*>(dout_) + static_cast<std::size_t>(k) * n_) : ov_[k].data;
      st.out[k] = tensor(p, mem, MI_DTYPE_F32, {MI_LATENT_CHANNELS, h, w});
    }
    last_ = mi_step_info{};
    last_.struct_size = sizeof last_;
    const mi_status status = mi_denoise_step(e_, &st, &last_);
    if (status != MI_OK) return status;
    for (int k = 0; k < info_.outputs; ++k) {
      // One output: a guided one, or the single row's (always evaluated); else each evaluated row's.
      const bool written = info_.outputs == 1 ? k == 0 : (last_.rows & (1u << k)) != 0;
      if (!written || st.out[k].data == nullptr) continue;
      Vec& o = out_[static_cast<std::size_t>(k)];
      o.resize(n);
      if (device) {
        EXPECT_EQ(cudaMemcpyAsync(o.data(), st.out[k].data, n * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                  cudaSuccess);
      } else {
        std::memcpy(o.data(), st.out[k].data, n * sizeof(float));
      }
    }
    if (device) {
      EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    }
    return status;
  }

 private:
  mi_engine* e_;
  Mode mode_;
  mi_run_info info_{};
  mi_tensor xv_{}, cv_{};
  mi_tensor ov_[MI_MAX_ROWS]{};
  std::size_t n_ = 0;
  cudaStream_t stream_ = nullptr;
  void* dx_ = nullptr;
  void* dout_ = nullptr;
  void* dctl_ = nullptr;
  std::vector<Vec> out_;
  mi_step_info last_{};
};

// The model of a guided run: out[0].
Model guided(Exchange& ex) {
  return [&ex](const Vec& x, float sigma, int step, Vec& pred) {
    Call c;
    c.sigma = sigma;
    c.step = step;
    ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
    pred = ex.out(0);
  };
}

// The model of a MOSS ROWS run: the host's per-op combine of the rows the engine evaluated.
Model moss_rows(Exchange& ex, bool velocity, double cfg, double slg_scale) {
  return [&ex, velocity, cfg, slg_scale](const Vec& x, float sigma, int step, Vec& pred) {
    Call c;
    c.sigma = sigma;
    c.step = step;
    ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
    const std::uint32_t r = ex.last().rows;
    guide_on_host(velocity, x, sigma, ex.out(0), (r & 2u) ? &ex.out(1) : nullptr, (r & 4u) ? &ex.out(2) : nullptr,
                  static_cast<float>(cfg), static_cast<float>(slg_scale), pred);
  };
}

// The model of a HOST run whose rows are (cond, uncond) pairs, latent l reading rows 2l and 2l + 1: the host's CFG per
// latent (velocity space), into pred [L, 128, h, w].
Model host_cfg(Exchange& ex, double cfg) {
  return [&ex, cfg](const Vec& x, float sigma, int step, Vec& pred) {
    Call c;
    c.sigma = sigma;
    c.step = step;
    ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
    const std::size_t L = static_cast<std::size_t>(ex.info().latents), n = x.size() / L;
    pred.resize(x.size());
    for (std::size_t l = 0; l < L; ++l) {
      cfg_velocity(ex.out(static_cast<int>(2 * l)).data(), ex.out(static_cast<int>(2 * l + 1)).data(),
                   static_cast<float>(cfg), pred.data() + l * n, n);
    }
  };
}

// ---------------------------------------------------------------- the fixture

// One engine for the suite (the SFT bundle, the INT8 DiT, tier high, q8_fp8 attention). Each test closes its session.
class GpuComponentRuns : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    mi_probe_info info{};
    info.struct_size = sizeof info;
    if (mi_probe(&info) != MI_OK || info.supported_device_count == 0) {
      skip_ = "no supported CUDA device";
      return;
    }
    if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(kInt8)) {
      skip_ = "SFT bundle or INT8 DiT not present";
      return;
    }
    engine_ = create(kInt8, MI_TIER_HIGH, MI_ATTENTION_Q8_FP8);
  }
  static void TearDownTestSuite() {
    mi_engine_free(engine_);
    engine_ = nullptr;
  }
  void SetUp() override {
    if (engine_ == nullptr) GTEST_SKIP() << skip_;
  }
  void TearDown() override {
    if (engine_ != nullptr) {
      EXPECT_EQ(mi_session_close(engine_), MI_OK) << last_error();
    }
  }

  static mi_engine* create(const fs::path& dit, mi_tier tier, mi_attention attention, std::uint64_t vram_budget = 0,
                           mi_block_path path = MI_BLOCK_PATH_DEFAULT, mi_mode mode = MI_MODE_BALANCED,
                           const fs::path& vae = fs::path()) {
    const std::string model = (weights_root() / "sft").string(), file = dit.string(), vae_file = vae.string();
    mi_engine_params p;
    EXPECT_EQ(mi_engine_params_init(&p, sizeof p), MI_OK);
    p.model_path = model.c_str();
    p.dit_file = dit.empty() ? nullptr : file.c_str();  // empty: the bundle's own bf16 DiT
    p.vae_file = vae.empty() ? nullptr : vae_file.c_str();
    p.tier = tier;
    p.attention = attention;
    p.vram_budget_bytes = vram_budget;
    p.block_path = path;
    p.mode = mode;
    mi_engine* e = nullptr;
    EXPECT_EQ(mi_engine_create(&p, &e), MI_OK) << last_error();
    return e;
  }

  // Frees the suite engine's device memory for a test that loads its own engine (the next call reloads).
  static void release_shared() { EXPECT_EQ(mi_engine_release(engine_, 0, nullptr), MI_OK) << last_error(); }

  static mi_engine* engine_;
  static std::string skip_;
};
mi_engine* GpuComponentRuns::engine_ = nullptr;
std::string GpuComponentRuns::skip_;

mi::json::Value component_of(const std::string& report) { return mi::json::parse(report).at("component"); }

// ---------------------------------------------------------------- table A

// Table A, euler on moss (design §5.3): mi_generate's RGB bytes from a host loop in every exchange mode (host staging,
// device tensors on the legacy stream, on a foreign stream and on the per-thread stream), for MOSS guidance with the
// guided velocity and with the host's per-op combine of the ROWS outputs, and for HOST rows {cond, uncond + {22}} with
// the host's CFG. The run reports the engine's schedule, every call on it, no foreign row.
TEST_F(GpuComponentRuns, TableAEulerOnMossInEveryExchange) {
  const mi_gen_params q = request();
  const std::vector<std::uint8_t> ref = generate(engine_, q);
  ASSERT_FALSE(ref.empty());
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  const Vec x0 = noise(q.seed);
  struct Case {
    mi_guidance guidance;
    mi_step_output output;
    const char* name;
  };
  const Case cases[] = {{MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, "MOSS guided velocity"},
                        {MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_ROWS, "MOSS rows, host combine"},
                        {MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS, "HOST rows, host CFG"}};
  for (const Mode mode : {Mode::kHost, Mode::kLegacy, Mode::kStream, Mode::kPerThread}) {
    for (const Case& c : cases) {
      SCOPED_TRACE(std::string(mode_name(mode)) + ", " + c.name);
      const mi_run_params rp = run_params(rows, c.guidance, c.output, q.steps);
      ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
      const Vec s = run_sigmas(engine_);
      Vec x;
      {
        Exchange ex(engine_, mode);
        const Model m = c.output == MI_STEP_OUTPUT_GUIDED_VELOCITY ? guided(ex)
                        : c.guidance == MI_GUIDANCE_MOSS           ? moss_rows(ex, true, 3.0, 0.0)
                                                                   : host_cfg(ex, 3.0);
        x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(x0, s[0]), m);
      }
      const mi::json::Value comp = component_of(end_run(engine_));
      EXPECT_EQ(comp.at("calls").as_int(), q.steps);
      EXPECT_EQ(comp.at("off_schedule_calls").as_int(), 0);
      EXPECT_TRUE(comp.at("schedule_matches").as_bool());
      EXPECT_TRUE(comp.at("foreign_rows").as_array().empty());
      EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's";
    }
  }
}

// Table A beyond euler on moss: euler on simple (GUIDED_DENOISED), ab2 on karras (GUIDED_VELOCITY) and heun on karras
// (GUIDED_DENOISED, the corrector at s[i + 1] with its explicit step i), each with an SLG window whose boundary falls
// between a step and the next (a corrector resolved by the lookup would take the next step's rows): mi_generate's RGB
// bytes. The prediction space is mi_sampler_prediction's.
TEST_F(GpuComponentRuns, TableADenoisedAb2AndHeun) {
  const std::int32_t slg[] = {7, 8, 9};
  struct Case {
    mi_sampler sampler;
    mi_scheduler scheduler;
    Mode mode;
  };
  const Case cases[] = {{MI_SAMPLER_EULER, MI_SCHEDULER_SIMPLE, Mode::kHost},
                        {MI_SAMPLER_AB2, MI_SCHEDULER_KARRAS, Mode::kLegacy},
                        {MI_SAMPLER_HEUN, MI_SCHEDULER_KARRAS, Mode::kHost},
                        {MI_SAMPLER_HEUN, MI_SCHEDULER_KARRAS, Mode::kStream}};
  bool opened = false;
  for (const Case& c : cases) {
    SCOPED_TRACE("sampler " + std::to_string(c.sampler) + ", scheduler " + std::to_string(c.scheduler) + ", " +
                 mode_name(c.mode));
    mi_gen_params q = request(6);
    q.sampler = c.sampler;
    q.scheduler = c.scheduler;
    q.slg_layers = slg;
    q.slg_layers_count = 3;
    q.slg_scale = 2.0;
    q.slg_start = 0.25;  // steps 2..4 of 6: the window opens between step 1 and step 2
    q.slg_stop = 0.75;
    const std::vector<std::uint8_t> ref = generate(engine_, q);
    if (!opened) {
      ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
      opened = true;
    }
    const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
    const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
    const mi_step_output output = mi_sampler_prediction(c.sampler, c.scheduler);
    ASSERT_NE(output, MI_STEP_OUTPUT_MAX_ENUM);
    mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, output, q.steps);
    rp.scheduler = c.scheduler;
    rp.slg_layers = slg;
    rp.slg_layers_count = 3;
    rp.slg_scale = q.slg_scale;
    rp.slg_start = q.slg_start;
    rp.slg_stop = q.slg_stop;
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, c.mode);
      const bool velocity_euler = c.scheduler == MI_SCHEDULER_MOSS;
      x = sample(c.sampler, velocity_euler, s.data(), s.size(), start_latents(noise(q.seed), s[0]), guided(ex));
    }
    const mi::json::Value comp = component_of(end_run(engine_));
    int calls = static_cast<int>(s.size()) - 1;
    if (c.sampler == MI_SAMPLER_HEUN) {
      calls += static_cast<int>(std::count_if(s.begin() + 1, s.end(), [](float v) { return v != 0.0f; }));
    }
    EXPECT_EQ(comp.at("calls").as_int(), calls);
    EXPECT_EQ(comp.at("off_schedule_calls").as_int(), 0);
    EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's";
  }
}

// Positions through the lookup (design §5.2): host sigmas with repeated values resolve to consecutive steps by the
// cursor, a restart at s[0] goes back to step 0, and each call's rows follow the step's SLG and CFG-truncation windows
// (MOSS selects them by position).
TEST_F(GpuComponentRuns, RepeatedSigmasResolveToConsecutiveSteps) {
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const mi_gen_params q = request();
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  const float sigmas[] = {1.0f, 0.8f, 0.8f, 0.8f, 0.5f, 0.5f, 0.25f, 0.0f};  // 7 steps
  const std::int32_t slg[] = {7};
  mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_ROWS, 7);
  rp.sigmas = sigmas;
  rp.sigmas_count = 8;
  rp.slg_layers = slg;
  rp.slg_layers_count = 1;
  rp.slg_scale = 1.5;
  rp.slg_start = 2.0 / 7;
  rp.slg_stop = 5.0 / 7;
  rp.cfg_truncation = 4.5 / 7;  // CFG on steps 0..4
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  EXPECT_EQ(run_info(engine_).schedule_matches, 0);
  const auto expected_rows = [&](int i) {
    const double f = static_cast<double>(i) / 7.0;
    const bool cfg = f < rp.cfg_truncation;
    std::uint32_t r = 1u | (cfg ? 2u : 0u);
    if (cfg && rp.slg_start <= f && f < rp.slg_stop) r |= 4u;
    return r;
  };
  {
    Exchange ex(engine_, Mode::kHost);
    const Vec x = noise(3);
    for (int pass = 0; pass < 2; ++pass) {  // the second pass restarts at s[0]
      for (int i = 0; i < 7; ++i) {
        Call c;
        c.sigma = sigmas[i];
        ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
        EXPECT_EQ(ex.last().step, i) << "pass " << pass;
        EXPECT_EQ(ex.last().on_schedule, 1);
        EXPECT_EQ(ex.last().rows, expected_rows(i)) << "step " << i;
      }
    }
  }
  const mi::json::Value comp = component_of(end_run(engine_));
  EXPECT_EQ(comp.at("calls").as_int(), 14);
  EXPECT_EQ(comp.at("off_schedule_calls").as_int(), 0);
  EXPECT_FALSE(comp.at("schedule_matches").as_bool());
}

// Arbitrary sigma (design §5.2) and table B (§5.3): a dpmpp_2s-style host loop (each step a call at s[i] and one at
// the off-schedule midpoint sqrt(s[i] s[i+1])) runs with no flag and no error; every midpoint counts as off schedule
// and takes its interval's position (step i: its rows follow step i's SLG window); with the step cache on, two runs
// give the same bits and reuse decisions. Then a host dpmpp_2m with std::exp and std::expm1 (table B: no bit claim)
// gives a valid sample: its RGB within a loose PSNR of mi_generate's dpmpp_2m.
TEST_F(GpuComponentRuns, OffScheduleCallsAndTableB) {
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  mi_gen_params q = request(8);
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  const std::int32_t slg[] = {7};
  const auto midpoint_run = [&](bool cache, std::vector<int>& cached) {
    mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_DENOISED, q.steps);
    rp.flags = 0;
    rp.scheduler = MI_SCHEDULER_KARRAS;
    rp.slg_layers = slg;
    rp.slg_layers_count = 1;
    rp.slg_scale = 1.5;
    rp.slg_start = 0.25;
    rp.slg_stop = 0.75;
    rp.step_cache = cache ? MI_STEP_CACHE_FBCACHE : MI_STEP_CACHE_OFF;
    rp.step_cache_threshold = cache ? 0.3 : 0.0;
    EXPECT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x = start_latents(noise(q.seed), s[0]);
    {
      Exchange ex(engine_, Mode::kHost);
      for (int i = 0; i + 1 < static_cast<int>(s.size()); ++i) {
        const float si = s[static_cast<std::size_t>(i)], sn = s[static_cast<std::size_t>(i) + 1];
        const float sm = sn > 0.0f ? std::sqrt(si * sn) : 0.0f;
        for (const float sigma : {si, sm}) {
          if (sigma == 0.0f) continue;
          Call c;
          c.sigma = sigma;
          EXPECT_EQ(ex.call(c, x), MI_OK) << last_error();
          EXPECT_EQ(ex.last().step, i) << "sigma " << sigma;
          EXPECT_EQ(ex.last().on_schedule, sigma == si ? 1 : 0);
          const bool window = i >= 2 && i < 6;  // 0.25 <= i / 8 < 0.75
          EXPECT_EQ((ex.last().rows & 4u) != 0, window) << "step " << i;
          if (ex.last().cached) cached.push_back(2 * i + (sigma == si ? 0 : 1));
          // A plain euler update from the call's denoised output, towards the next point.
          const float to = sigma == si ? (sm > 0.0f ? sm : sn) : sn;
          for (std::size_t k = 0; k < x.size(); ++k) {
            const float d = (x[k] - ex.out(0)[k]) / sigma;
            const float t = d * (to - sigma);
            x[k] = x[k] + t;
          }
        }
      }
    }
    const mi::json::Value comp = component_of(end_run(engine_));
    EXPECT_EQ(comp.at("off_schedule_calls").as_int(), q.steps - 1);  // the last step has no midpoint (s[n] = 0)
    return x;
  };
  std::vector<int> unused, cached_a, cached_b;
  midpoint_run(false, unused);
  const Vec a = midpoint_run(true, cached_a);
  const Vec b = midpoint_run(true, cached_b);
  EXPECT_FALSE(cached_a.empty()) << "the threshold reused nothing: the test shows nothing";
  EXPECT_EQ(cached_a, cached_b);
  EXPECT_EQ(differing(a, b), 0u) << "two step-cache runs differ";

  // Table B: dpmpp_2m (k-diffusion's update, t = -log sigma) on karras, the host's own libm.
  q.sampler = MI_SAMPLER_DPMPP_2M;
  q.scheduler = MI_SCHEDULER_KARRAS;
  const std::vector<std::uint8_t> ref = generate(engine_, q);
  const mi_step_output prediction = mi_sampler_prediction(MI_SAMPLER_DPMPP_2M, MI_SCHEDULER_KARRAS);
  mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, prediction, q.steps);
  rp.scheduler = MI_SCHEDULER_KARRAS;
  ASSERT_EQ(rp.output, MI_STEP_OUTPUT_GUIDED_DENOISED);
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  const Vec s = run_sigmas(engine_);
  Vec x = start_latents(noise(q.seed), s[0]);
  {
    Exchange ex(engine_, Mode::kHost);
    Vec old;
    for (int i = 0; i + 1 < static_cast<int>(s.size()); ++i) {
      const double si = s[static_cast<std::size_t>(i)], sn = s[static_cast<std::size_t>(i) + 1];
      Call c;
      c.sigma = s[static_cast<std::size_t>(i)];
      ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
      const Vec& den = ex.out(0);
      if (sn == 0.0) {
        x = den;
      } else {
        const double t = -std::log(si), tn = -std::log(sn), h = tn - t;
        const double ratio = sn / si, e = -std::expm1(-h);
        double w0 = 1.0, w1 = 0.0;
        if (!old.empty()) {
          const double r = (t + std::log(static_cast<double>(s[static_cast<std::size_t>(i) - 1]))) / h;
          w0 = 1.0 + 1.0 / (2.0 * r);
          w1 = 1.0 / (2.0 * r);
        }
        for (std::size_t k = 0; k < x.size(); ++k) {
          const double dd = w0 * den[k] - (old.empty() ? 0.0 : w1 * old[k]);
          x[k] = static_cast<float>(ratio * x[k] + e * dd);
        }
      }
      old = den;
    }
  }
  end_run(engine_);
  const std::vector<std::uint8_t> rgb = decode(engine_, x);
  ASSERT_EQ(rgb.size(), ref.size());
  double se = 0;
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    const double d = static_cast<double>(rgb[i]) - ref[i];
    se += d * d;
  }
  const double psnr = 10.0 * std::log10(255.0 * 255.0 / std::max(se / static_cast<double>(rgb.size()), 1e-12));
  std::printf("[table B] host dpmpp_2m: PSNR %.1f dB against mi_generate's (%zu of %zu RGB bytes differ)\n", psnr,
              differing(rgb, ref), rgb.size());
  EXPECT_GT(psnr, 25.0);
}

// ---------------------------------------------------------------- MOSS windows

// MOSS guidance with SLG (its window also the uncond skip window) and CFG truncation, 10 steps: the guided velocity
// and the host's per-op combine of the ROWS outputs give mi_generate's bits; each call evaluates the rows of
// denoise()'s rule for its step; the truncated steps are reported as mi_generate reports them. Then cfg 1 with SLG
// (rows cond and SLG only: ROWS keeps the SLG row at out[2], out[1] has no slot).
TEST_F(GpuComponentRuns, MossWindowsAndCfgTruncationEqualGenerate) {
  const std::int32_t slg[] = {7, 8, 9};
  mi_gen_params q = request(10);
  q.slg_layers = slg;
  q.slg_layers_count = 3;
  q.slg_scale = 2.5;
  q.slg_start = 0.2;
  q.slg_stop = 0.7;
  q.cfg_truncation = 0.6;
  mi_result* gr = nullptr;
  const std::vector<std::uint8_t> ref = generate(engine_, q, &gr);
  mi_accel_info ga{};
  ga.struct_size = sizeof ga;
  ASSERT_EQ(mi_result_accel(gr, &ga), MI_OK);
  mi_result_free(gr);
  EXPECT_EQ(ga.cfg_truncated_steps, 4);

  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  const auto moss = [&](const std::vector<mi_run_row>& r, mi_step_output output) {
    mi_run_params rp = run_params(r, MI_GUIDANCE_MOSS, output, q.steps);
    rp.slg_layers = slg;
    rp.slg_layers_count = 3;
    rp.slg_scale = q.slg_scale;
    rp.slg_start = q.slg_start;
    rp.slg_stop = q.slg_stop;
    rp.cfg_truncation = r.size() == 2 ? q.cfg_truncation : 1.0;
    return rp;
  };
  for (const mi_step_output output : {MI_STEP_OUTPUT_GUIDED_VELOCITY, MI_STEP_OUTPUT_ROWS}) {
    SCOPED_TRACE(output == MI_STEP_OUTPUT_ROWS ? "rows, host combine" : "guided velocity");
    const mi_run_params rp = moss(rows, output);
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    std::vector<std::uint32_t> seen;
    Vec x;
    {
      Exchange ex(engine_, output == MI_STEP_OUTPUT_ROWS ? Mode::kLegacy : Mode::kHost);
      const Model inner = output == MI_STEP_OUTPUT_ROWS ? moss_rows(ex, true, 3.0, q.slg_scale) : guided(ex);
      const Model m = [&](const Vec& xx, float sigma, int step, Vec& pred) {
        inner(xx, sigma, step, pred);
        seen.push_back(ex.last().rows);
      };
      x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0]), m);
    }
    std::vector<std::uint32_t> want;
    for (int i = 0; i < q.steps; ++i) {
      const double f = static_cast<double>(i) / 10.0;
      const bool cfg = f < 0.6, window = f >= 0.2 && f < 0.7;
      want.push_back(1u | (cfg ? 2u : 0u) | (cfg && window ? 4u : 0u));
    }
    EXPECT_EQ(seen, want);
    mi_result* r = nullptr;
    end_run(engine_, &r);
    mi_accel_info a{};
    a.struct_size = sizeof a;
    ASSERT_EQ(mi_result_accel(r, &a), MI_OK);
    EXPECT_EQ(a.cfg_truncated_steps, ga.cfg_truncated_steps);
    mi_result_free(r);
    EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's";
  }

  // cfg 1 + SLG: guide_prediction's velocity rule takes the SLG row as its base without CFG.
  q.cfg = 1.0;
  q.cfg_truncation = 1.0;
  const std::vector<std::uint8_t> ref1 = generate(engine_, q);
  const std::vector<mi_run_row> cond_only = {row(conds[0])};
  for (const mi_step_output output : {MI_STEP_OUTPUT_GUIDED_VELOCITY, MI_STEP_OUTPUT_ROWS}) {
    SCOPED_TRACE(output == MI_STEP_OUTPUT_ROWS ? "cfg 1, rows" : "cfg 1, guided velocity");
    const mi_run_params rp = moss(cond_only, output);
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const mi_run_info info = run_info(engine_);
    EXPECT_EQ(info.rows, 5u);
    EXPECT_EQ(info.outputs, output == MI_STEP_OUTPUT_ROWS ? 3 : 1);
    if (output == MI_STEP_OUTPUT_ROWS) {
      mi_tensor xv{}, ov[3]{};
      ASSERT_EQ(mi_run_staging(engine_, &xv, ov, 3, nullptr), MI_OK);
      EXPECT_EQ(ov[1].data, nullptr) << "out[1] has no slot without an uncond row";
      EXPECT_NE(ov[2].data, nullptr);
    }
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, Mode::kHost);
      const Model m = output == MI_STEP_OUTPUT_ROWS ? moss_rows(ex, true, 1.0, q.slg_scale) : guided(ex);
      x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0]), m);
    }
    end_run(engine_);
    EXPECT_EQ(differing(decode(engine_, x), ref1), 0u) << "RGB bytes differ from mi_generate's";
  }
}

// ---------------------------------------------------------------- HOST rows

// HOST rows (design §5.2): runs of 1, 2, 4 and 8 rows over four prompts of different lengths and different skip sets.
// Every row's output is the same in a call of all rows, in a call of that row alone and in a call of a subset, and
// across the runs (the same row in a run of 1, 2, 4 or 8 rows). Then a run of the cond row alone, sampled by the host,
// gives mi_generate's bits at cfg 1.
TEST_F(GpuComponentRuns, HostRowsEqualSeparateCalls) {
  ASSERT_EQ(open_session(engine_, 8), MI_OK) << last_error();
  const Conds conds =
      encode(engine_, {"a cat", "a red apple on a wooden table, studio light", "blurry",
                       "an old lighthouse on a cliff at dusk, waves crashing, dramatic clouds, oil painting"});
  const std::vector<std::vector<std::int32_t>> skips = {{}, {22}, {5, 6}, {}, {30}, {22}, {}, {1, 2, 3}};
  std::vector<mi_run_row> all;
  for (std::size_t k = 0; k < 8; ++k) all.push_back(row(conds[k % 4], skips[k].data(), skips[k].size()));
  const Vec x = noise(5);
  std::vector<Vec> first(8);  // each row's output in the first run that had it
  for (const std::size_t count : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
    SCOPED_TRACE(std::to_string(count) + " rows");
    const std::vector<mi_run_row> rows(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(count));
    const mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    {
      Exchange ex(engine_, count == 4 ? Mode::kLegacy : Mode::kHost);
      EXPECT_EQ(ex.info().outputs, static_cast<int>(count));
      Call c;
      c.sigma = s[1];
      ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
      EXPECT_EQ(ex.last().rows, (1u << count) - 1);
      std::vector<Vec> full;
      for (std::size_t k = 0; k < count; ++k) {
        full.push_back(ex.out(static_cast<int>(k)));
        if (first[k].empty()) {
          first[k] = full[k];
        } else {
          EXPECT_EQ(differing(full[k], first[k]), 0u) << "row " << k << " differs from a run of fewer rows";
        }
      }
      for (std::size_t k = 0; k < count && count > 1; ++k) {
        c.rows = 1u << k;
        ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
        EXPECT_EQ(ex.last().rows, c.rows);
        EXPECT_EQ(differing(ex.out(static_cast<int>(k)), full[k]), 0u) << "row " << k << " alone";
      }
      if (count >= 4) {
        c.rows = count == 8 ? 0b10110010u : 0b1010u;
        ASSERT_EQ(ex.call(c, x), MI_OK) << last_error();
        for (std::size_t k = 0; k < count; ++k) {
          if (c.rows & (1u << k)) {
            EXPECT_EQ(differing(ex.out(static_cast<int>(k)), full[k]), 0u) << "row " << k << " in a subset";
          }
        }
      }
      c.rows = 1u << count;  // a row the run does not have
      EXPECT_EQ(ex.call(c, x), MI_ERR_INVALID_ARGUMENT);
      EXPECT_TRUE(has(last_error(), "row mask")) << last_error();
    }
    end_run(engine_);
  }

  // The cond row alone, sampled by the host: mi_generate at cfg 1.
  mi_gen_params q = request();
  q.cfg = 1.0;
  const std::vector<std::uint8_t> ref = generate(engine_, q);
  const Conds cond = encode(engine_, {q.prompt});
  const std::vector<mi_run_row> one = {row(cond[0])};
  const mi_run_params rp = run_params(one, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS, q.steps);
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  const Vec s = run_sigmas(engine_);
  Vec xs;
  {
    Exchange ex(engine_, Mode::kHost);
    xs = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0]), guided(ex));
  }
  end_run(engine_);
  EXPECT_EQ(differing(decode(engine_, xs), ref), 0u) << "RGB bytes differ from mi_generate's at cfg 1";
}

// ---------------------------------------------------------------- latent batches

// Each latent's (cond, uncond + {22}) rows with the shared negative conditioning: rows 2l and 2l + 1 read latent l (the
// same negative handle on different latents: distinct row identities).
std::vector<mi_run_row> cfg_rows(const Conds& conds, std::size_t latents, std::size_t negative) {
  std::vector<mi_run_row> rows;
  for (std::size_t l = 0; l < latents; ++l) {
    rows.push_back(row(conds[l], nullptr, 0, static_cast<int>(l)));
    rows.push_back(row(conds[negative], kSkip22, 1, static_cast<int>(l)));
  }
  return rows;
}

Vec concat(std::initializer_list<Vec> parts) {
  Vec v;
  for (const Vec& p : parts) v.insert(v.end(), p.begin(), p.end());
  return v;
}

// Latent batches (design §5.2): L = 2 (4 rows) and L = 4 (8 rows), the host's CFG per latent, in host staging and in
// device mode: each image's final latents equal its own single-image run (L = 1) bit for bit, and the first image is
// mi_generate's RGB bytes. A batch over MI_MAX_CALL_TOKENS (L = 2 with CFG at 2048^2) is refused.
TEST_F(GpuComponentRuns, LatentBatchesEqualSingleImages) {
  ASSERT_EQ(open_session(engine_, 8, 4), MI_OK) << last_error();
  const char* prompts[] = {"a red apple on a wooden table, studio light", "a cat",
                           "an old lighthouse on a cliff at dusk, oil painting", "a bowl of ramen, top view"};
  const std::int64_t seeds[] = {7, 8, 9, 10};
  const Conds conds = encode(engine_, {prompts[0], prompts[1], prompts[2], prompts[3], "blurry"});
  const auto run = [&](const std::vector<mi_run_row>& rows, const Vec& x0, Mode mode) {
    const mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
    EXPECT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, mode);
      x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(x0, s[0]), host_cfg(ex, 3.0));
    }
    end_run(engine_);
    return x;
  };
  std::vector<Vec> single(4);
  for (std::size_t l = 0; l < 4; ++l) {
    const std::vector<mi_run_row> rows = {row(conds[l]), row(conds[4], kSkip22, 1)};
    single[l] = run(rows, noise(seeds[l]), Mode::kHost);
  }
  const std::vector<std::uint8_t> ref = generate(engine_, request(6, seeds[0], prompts[0], "blurry"));
  EXPECT_EQ(differing(decode(engine_, single[0]), ref), 0u) << "the single run differs from mi_generate";
  struct Case {
    std::size_t latents;
    Mode mode;
  };
  for (const Case c : {Case{2, Mode::kHost}, Case{2, Mode::kStream}, Case{4, Mode::kHost}, Case{4, Mode::kLegacy}}) {
    SCOPED_TRACE("L = " + std::to_string(c.latents) + ", " + mode_name(c.mode));
    Vec x0;
    for (std::size_t l = 0; l < c.latents; ++l) x0 = concat({x0, noise(seeds[l])});
    const Vec x = run(cfg_rows(conds, c.latents, 4), x0, c.mode);
    for (std::size_t l = 0; l < c.latents; ++l) {
      EXPECT_EQ(differing(slice(x, l), single[l]), 0u) << "latent " << l << " differs from its single-image run";
    }
  }

  const std::vector<mi_run_row> rows = cfg_rows(conds, 2, 4);
  mi_run_params big = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
  big.width = big.height = 2048;
  EXPECT_EQ(mi_run_begin(engine_, &big), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "MI_MAX_CALL_TOKENS")) << last_error();
}

// The batch's per-image bit identity on the other numerics of sm_89: the INT8 DiT with exact attention and the bf16 DiT
// (tier reference): one L = 2 call (4 rows) at two sigmas equals the two images' L = 1 calls, output for output.
TEST_F(GpuComponentRuns, LatentBatchOtherNumerics) {
  release_shared();
  struct Config {
    fs::path dit;
    mi_tier tier;
    mi_attention attention;
    const char* name;
  };
  for (const Config& cfg : {Config{kInt8, MI_TIER_HIGH, MI_ATTENTION_EXACT, "INT8, exact attention"},
                            Config{fs::path(), MI_TIER_REFERENCE, MI_ATTENTION_EXACT, "bf16, tier reference"}}) {
    SCOPED_TRACE(cfg.name);
    mi_engine* e = create(cfg.dit, cfg.tier, cfg.attention);
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(open_session(e, 4, 2), MI_OK) << last_error();
    {
      const Conds conds = encode(e, {"a red apple on a wooden table, studio light", "a cat", "blurry"});
      const Vec xa = noise(7), xb = noise(8);
      const auto calls = [&](const std::vector<mi_run_row>& rows, const Vec& x) {
        const mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
        std::vector<Vec> got;
        if (mi_run_begin(e, &rp) != MI_OK) {
          ADD_FAILURE() << "mi_run_begin: " << last_error();
          return got;
        }
        const Vec s = run_sigmas(e);
        {
          Exchange ex(e, Mode::kLegacy);
          for (const float sigma : {s[0], s[3]}) {
            Call c;
            c.sigma = sigma;
            EXPECT_EQ(ex.call(c, start_latents(x, sigma)), MI_OK) << last_error();
            for (int k = 0; k < ex.info().outputs; ++k) got.push_back(ex.out(k));
          }
        }
        end_run(e);
        return got;
      };
      const std::vector<Vec> batch = calls(cfg_rows(conds, 2, 2), concat({xa, xb}));
      const std::vector<Vec> a = calls({row(conds[0]), row(conds[2], kSkip22, 1)}, xa);
      const std::vector<Vec> b = calls({row(conds[1]), row(conds[2], kSkip22, 1)}, xb);
      ASSERT_EQ(batch.size(), 8u);
      ASSERT_EQ(a.size(), 4u);
      ASSERT_EQ(b.size(), 4u);
      for (std::size_t i = 0; i < 2; ++i) {  // the sigma
        for (std::size_t k = 0; k < 2; ++k) {
          EXPECT_EQ(differing(batch[4 * i + k], a[2 * i + k]), 0u) << "image 0, call " << i << ", row " << k;
          EXPECT_EQ(differing(batch[4 * i + 2 + k], b[2 * i + k]), 0u) << "image 1, call " << i << ", row " << k;
        }
      }
    }
    EXPECT_EQ(mi_session_close(e), MI_OK);
    mi_engine_free(e);
  }
}

// Latent batches with a side-branch LoRA: a call runs one forward per latent, so a row's side-branch GEMM sees its
// single-image call's tokens even when the latents' prompts differ in length. An L = 2 run (4 rows, the host's CFG)
// equals each image's own run bit for bit, in host staging and device mode.
TEST_F(GpuComponentRuns, LatentBatchWithASideBranchLoraEqualsSingleImages) {
  const fs::path file = "/workspace/Models/community/axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors";
  if (!fs::is_regular_file(file)) GTEST_SKIP() << "LoRA file not present";
  const std::string path = file.string();
  const mi_lora lora{path.c_str(), 0.8};
  ASSERT_EQ(open_session(engine_, 4, 2), MI_OK) << last_error();
  const char* prompts[] = {kPrompt, "an old lighthouse on a cliff at dusk, oil painting, heavy brush strokes"};
  const std::int64_t seeds[] = {7, 8};
  const Conds conds = encode(engine_, {prompts[0], prompts[1], "blurry"});
  const auto run = [&](const std::vector<mi_run_row>& rows, const Vec& x0, Mode mode) {
    mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
    rp.loras = &lora;
    rp.loras_count = 1;
    EXPECT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, mode);
      x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(x0, s[0]), host_cfg(ex, 3.0));
    }
    end_run(engine_);
    return x;
  };
  std::vector<Vec> single(2);
  for (std::size_t l = 0; l < 2; ++l) single[l] = run({row(conds[l]), row(conds[2], kSkip22, 1)}, noise(seeds[l]), Mode::kHost);
  for (const Mode mode : {Mode::kHost, Mode::kStream}) {
    SCOPED_TRACE(mode_name(mode));
    const Vec x = run(cfg_rows(conds, 2, 2), concat({noise(seeds[0]), noise(seeds[1])}), mode);
    for (std::size_t l = 0; l < 2; ++l) {
      EXPECT_EQ(differing(slice(x, l), single[l]), 0u) << "latent " << l << " differs from its single-image run";
    }
  }
}

// The residency rules at mi_run_begin (design §5.4), on engines whose budget makes the DiT stream at 1024^2:
// - a latent batch never streams more DiT bytes per step than its latent-0 rows alone: L = 4 with two rows each is
//   refused at 4.5 GiB (INT8), naming both byte counts, while latent 0's rows alone begin;
// - MI_RUN_CROPS never keeps fewer DiT blocks resident: refused at 7 GiB (bf16, four rows; its crop bound is wider
//   than the fused route's), naming both counts, while the same run without the flag begins.
// Both refusals leave the session usable.
TEST_F(GpuComponentRuns, BatchAndCropResidencyRefusals) {
  release_shared();
  const auto gib = [](double g) { return static_cast<std::uint64_t>(g * static_cast<double>(1ull << 30)); };
  {
    mi_engine* e = create(kInt8, MI_TIER_HIGH, MI_ATTENTION_Q8_FP8, gib(4.5));
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(open_session(e, 8, 4), MI_OK) << last_error();
    {
      const Conds conds =
          encode(e, {"a red apple on a wooden table, studio light", "a cat", "a dog", "a bird", "blurry"});
      const std::vector<mi_run_row> rows = cfg_rows(conds, 4, 4);
      mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
      rp.width = rp.height = 1024;
      EXPECT_EQ(mi_run_begin(e, &rp), MI_ERR_INVALID_ARGUMENT);
      const std::string msg = last_error();
      EXPECT_TRUE(has(msg, "A latent batch of 4 would stream") && has(msg, "of its latent-0 rows alone")) << msg;
      const std::vector<mi_run_row> latent0(rows.begin(), rows.begin() + 2);  // the reference: latent 0's rows alone
      rp = run_params(latent0, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
      rp.width = rp.height = 1024;
      EXPECT_EQ(mi_run_begin(e, &rp), MI_OK) << last_error();
      EXPECT_EQ(mi_run_end(e, nullptr), MI_OK);
    }
    EXPECT_EQ(mi_session_close(e), MI_OK);
    mi_engine_free(e);
  }
  {
    mi_engine* e = create(fs::path(), MI_TIER_HIGH, MI_ATTENTION_Q8_FP8, gib(7.0));
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(open_session(e, 4), MI_OK) << last_error();
    {
      const Conds conds = encode(e, {"a red apple on a wooden table, studio light", "blurry", "a cat"});
      std::vector<mi_run_row> rows;
      for (std::size_t k = 0; k < 4; ++k) rows.push_back(row(conds[k % 3]));
      mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
      rp.width = rp.height = 1024;
      rp.flags = MI_RUN_CROPS;
      EXPECT_EQ(mi_run_begin(e, &rp), MI_ERR_INVALID_ARGUMENT);
      const std::string msg = last_error();
      EXPECT_TRUE(has(msg, "MI_RUN_CROPS's crop workspace would keep") && has(msg, "without it")) << msg;
      rp.flags = 0;
      EXPECT_EQ(mi_run_begin(e, &rp), MI_OK) << last_error();
      EXPECT_EQ(mi_run_end(e, nullptr), MI_OK);
    }
    EXPECT_EQ(mi_session_close(e), MI_OK);
    mi_engine_free(e);
  }
}

// ---------------------------------------------------------------- crop calls

// Crop calls (area conds, design §5.2): in a HOST run with MI_RUN_CROPS at 512^2, a 320x384 crop (x and the outputs
// packed from the start of their slots) evaluates exactly as a run at 320x384, in host staging and device mode. Every
// crop size fits the run's planned bound. The refusals keep the run usable: a crop in MOSS guidance, a crop larger
// than the run.
TEST_F(GpuComponentRuns, CropCallsEqualRunsAtTheCropSize) {
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {"a red apple on a wooden table, studio light", "blurry"});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  constexpr std::int64_t kH = 24, kW = 20;  // 384 high, 320 wide
  const Vec xc = noise(11, static_cast<int>(kW * 16), static_cast<int>(kH * 16));
  const float sigma = 0.6f;
  // The reference: a run at the crop's size, the same sigma (off its schedule: accepted without MI_RUN_EXACT_SIGMA).
  mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
  rp.width = static_cast<int>(kW * 16);
  rp.height = static_cast<int>(kH * 16);
  rp.flags = 0;
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  Vec want0, want1;
  {
    Exchange ex(engine_, Mode::kHost);
    Call c;
    c.sigma = sigma;
    ASSERT_EQ(ex.call(c, xc), MI_OK) << last_error();
    EXPECT_EQ(ex.last().on_schedule, 0);
    want0 = ex.out(0);
    want1 = ex.out(1);
  }
  end_run(engine_);

  rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
  rp.flags = MI_RUN_CROPS;
  for (const Mode mode : {Mode::kHost, Mode::kLegacy}) {
    SCOPED_TRACE(mode_name(mode));
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    {
      Exchange ex(engine_, mode);
      Call c;
      c.sigma = sigma;
      c.h = kH;
      c.w = kW;
      ASSERT_EQ(ex.call(c, xc), MI_OK) << last_error();
      EXPECT_EQ(ex.last().cached, 0);
      EXPECT_EQ(differing(ex.out(0), want0), 0u) << "row 0 of the crop differs from the run at its size";
      EXPECT_EQ(differing(ex.out(1), want1), 0u) << "row 1 of the crop differs from the run at its size";
      // Every crop size fits the bound MI_RUN_CROPS planned (a sample of the grid, each row alone and both).
      for (std::int64_t h = 1; h <= kLat; h += h < 4 ? 1 : 9) {
        for (std::int64_t w = 1; w <= kLat; w += w < 4 ? 1 : 9) {
          for (const std::uint32_t mask : {1u, 3u}) {
            Call k;
            k.sigma = sigma;
            k.h = h;
            k.w = w;
            k.rows = mask;
            EXPECT_EQ(ex.call(k, noise(12, static_cast<int>(w * 16), static_cast<int>(h * 16))), MI_OK)
                << h << "x" << w << ": " << last_error();
          }
        }
      }
      if (mode == Mode::kHost) {
        // Larger than the run: refused, the run still usable.
        mi_tensor xv{}, ov[2]{};
        ASSERT_EQ(mi_run_staging(engine_, &xv, ov, 2, nullptr), MI_OK);
        mi_step_params st;
        ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
        st.sigma = sigma;
        st.x = tensor(xv.data, MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat + 1, kLat});
        st.out[0] = tensor(ov[0].data, MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat + 1, kLat});
        st.out[1] = tensor(ov[1].data, MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat + 1, kLat});
        EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
        EXPECT_TRUE(has(last_error(), "x: expected")) << last_error();
        ASSERT_EQ(ex.call(c, xc), MI_OK) << last_error();
        EXPECT_EQ(differing(ex.out(0), want0), 0u);
      }
    }
    end_run(engine_);
  }

  // MOSS guidance takes no crop.
  const mi_run_params moss = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY);
  ASSERT_EQ(mi_run_begin(engine_, &moss), MI_OK) << last_error();
  {
    Exchange ex(engine_, Mode::kHost);
    const Vec s = run_sigmas(engine_);
    Call c;
    c.sigma = s[0];
    c.h = kH;
    c.w = kW;
    EXPECT_EQ(ex.call(c, xc), MI_ERR_INVALID_ARGUMENT);
    EXPECT_TRUE(has(last_error(), "need HOST guidance")) << last_error();
    c.h = c.w = 0;
    EXPECT_EQ(ex.call(c, noise(1)), MI_OK) << last_error();
  }
  end_run(engine_);
}

// Crop calls leave the step cache untouched (design §5.2): a host loop with the step cache on and a crop call before
// every step's call makes the same reuse decisions and gives the same bits as the loop without crop calls; the crop
// calls never report a reuse. On the default (fused) path, and on CUDA graphs (MI_BLOCK_PATH_FUSED_GRAPHS), where crop
// calls run eagerly with the run's graphs kept: the same bits again.
TEST_F(GpuComponentRuns, CropCallsLeaveTheStepCacheAndGraphsAlone) {
  const auto loop = [](mi_engine* e, bool crops, std::vector<int>& cached) {
    const Conds conds = encode(e, {"a red apple on a wooden table, studio light", "blurry"});
    const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
    mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS, 10);
    rp.flags = MI_RUN_CROPS | MI_RUN_EXACT_SIGMA;
    rp.step_cache = MI_STEP_CACHE_FBCACHE;
    rp.step_cache_threshold = 0.3;
    EXPECT_EQ(mi_run_begin(e, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(e);
    Vec x = start_latents(noise(7), s[0]);
    const Vec xc = noise(9, 320, 384);
    {
      Exchange ex(e, Mode::kHost);
      for (int i = 0; i + 1 < static_cast<int>(s.size()); ++i) {
        if (crops) {
          Call c;
          c.sigma = s[static_cast<std::size_t>(i)];
          c.step = i;
          c.h = 24;
          c.w = 20;
          EXPECT_EQ(ex.call(c, xc), MI_OK) << last_error();
          EXPECT_EQ(ex.last().cached, 0) << "a crop call reported a reuse";
        }
        Call c;
        c.sigma = s[static_cast<std::size_t>(i)];
        c.step = i;
        EXPECT_EQ(ex.call(c, x), MI_OK) << last_error();
        if (ex.last().cached) cached.push_back(i);
        Vec v(x.size());
        cfg_velocity(ex.out(0).data(), ex.out(1).data(), 3.0f, v.data(), v.size());
        const float h = static_cast<float>(static_cast<double>(s[static_cast<std::size_t>(i) + 1]) -
                                           static_cast<double>(s[static_cast<std::size_t>(i)]));
        for (std::size_t k = 0; k < x.size(); ++k) {
          const float t = h * v[k];
          x[k] = x[k] + t;
        }
      }
    }
    end_run(e);
    return x;
  };
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  std::vector<int> plain_cached, crop_cached;
  const Vec plain = loop(engine_, false, plain_cached);
  const Vec cropped = loop(engine_, true, crop_cached);
  EXPECT_FALSE(plain_cached.empty()) << "the threshold reused nothing: the test shows nothing";
  EXPECT_EQ(crop_cached, plain_cached);
  EXPECT_EQ(differing(cropped, plain), 0u);
  EXPECT_EQ(mi_session_close(engine_), MI_OK);

  release_shared();
  mi_engine* g = create(kInt8, MI_TIER_HIGH, MI_ATTENTION_Q8_FP8, 0, MI_BLOCK_PATH_FUSED_GRAPHS);
  ASSERT_NE(g, nullptr);
  ASSERT_EQ(open_session(g), MI_OK) << last_error();
  std::vector<int> graph_plain_cached, graph_crop_cached;
  const Vec graph_plain = loop(g, false, graph_plain_cached);
  const Vec graph_cropped = loop(g, true, graph_crop_cached);
  EXPECT_EQ(graph_plain_cached, plain_cached);
  EXPECT_EQ(graph_crop_cached, plain_cached);
  EXPECT_EQ(differing(graph_plain, plain), 0u) << "graphs";
  EXPECT_EQ(differing(graph_cropped, plain), 0u) << "graphs with crop calls";
  EXPECT_EQ(mi_session_close(g), MI_OK);
  mi_engine_free(g);
}

// ---------------------------------------------------------------- control

// Synthetic control images of other sizes than the run's (the resize runs): a depth ramp (1 channel, near = white) and
// an RGB color pattern.
struct ControlInputs {
  std::vector<std::uint8_t> depth_px, color_px;
  mi_image depth{}, color{};
  ControlInputs() {
    constexpr int dw = 600, dh = 480, cw = 500, ch = 640;
    depth_px.resize(static_cast<std::size_t>(dw) * dh);
    for (int y = 0; y < dh; ++y) {
      for (int x = 0; x < dw; ++x) {
        depth_px[static_cast<std::size_t>(y) * dw + x] =
            static_cast<std::uint8_t>((x * 255 / (dw - 1) + y * 255 / (dh - 1)) / 2);
      }
    }
    color_px.resize(static_cast<std::size_t>(cw) * ch * 3);
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        std::uint8_t* p = &color_px[(static_cast<std::size_t>(y) * cw + x) * 3];
        p[0] = static_cast<std::uint8_t>(x * 255 / (cw - 1));
        p[1] = static_cast<std::uint8_t>(y * 255 / (ch - 1));
        p[2] = static_cast<std::uint8_t>(((x / 32 + y / 32) % 2) * 200);
      }
    }
    depth = mi_image{depth_px.data(), dw, dh, 1, 0};
    color = mi_image{color_px.data(), cw, ch, 3, 0};
  }
};

// The adapter's kinds in group order ("color,depth": group 0 color, group 1 depth), with windows that flip inside an
// 8-step pass: color on steps 0..3, depth on steps 3..7.
constexpr double kColorStrength = 0.8, kColorStart = 0.0, kColorEnd = 0.5;
constexpr double kDepthStrength = 1.0, kDepthStart = 0.3, kDepthEnd = 1.0;
constexpr std::int64_t kGroup = MI_LATENT_CHANNELS * kLat * kLat;  // one group's [128, h, w]

bool window_active(double start, double end, int step, int steps) {
  const double f = static_cast<double>(step) / static_cast<double>(steps);
  return start <= f && f < end;
}

// The host's control latent of one image per §5.3's recipe: the run's resize (CROP), grayscale replicated to RGB after
// the resize, mi_vae_encode, times fp32(strength).
Vec control_latent(mi_engine* e, const mi_image& im, double strength) {
  std::vector<std::uint8_t> px(static_cast<std::size_t>(kSize) * kSize * static_cast<std::size_t>(im.channels));
  EXPECT_EQ(mi_image_resize(&im, kSize, kSize, MI_RESIZE_CROP, px.data(), px.size()), MI_OK) << last_error();
  std::vector<std::uint8_t> rgb = px;
  if (im.channels == 1) {
    rgb.resize(px.size() * 3);
    for (std::size_t i = 0; i < px.size(); ++i) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = px[i];
  }
  const mi_image sized{rgb.data(), kSize, kSize, 3, 0};
  Vec z(kN);
  const mi_tensor t = tensor(z.data(), MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat, kLat});
  EXPECT_EQ(mi_vae_encode(e, &sized, &t), MI_OK) << last_error();
  if (strength != 1.0) {
    const float s = static_cast<float>(strength);
    for (float& v : z) v = v * s;
  }
  return z;
}

// The control input [256, h, w] at `step` of `steps`: each group's latent inside its window, zeros elsewhere.
Vec control_at(const Vec& color, const Vec& depth, int step, int steps) {
  Vec c(static_cast<std::size_t>(2 * kGroup), 0.0f);
  if (window_active(kColorStart, kColorEnd, step, steps)) std::copy(color.begin(), color.end(), c.begin());
  if (window_active(kDepthStart, kDepthEnd, step, steps)) std::copy(depth.begin(), depth.end(), c.begin() + kGroup);
  return c;
}

// mi_generate's control request: the adapter and both images with their windows.
struct ControlRequest {
  std::string adapter = kControl.string();
  mi_control_image images[2];
  explicit ControlRequest(const ControlInputs& in) {
    images[0] = mi_control_image{"color", &in.color, nullptr, kColorStrength, kColorStart, kColorEnd};
    images[1] = mi_control_image{"depth", &in.depth, nullptr, kDepthStrength, kDepthStart, kDepthEnd};
  }
  void apply(mi_gen_params& q, bool with_images) const {
    q.control_adapter = adapter.c_str();
    q.control_images = with_images ? images : nullptr;
    q.control_images_count = with_images ? 2 : 0;
  }
  void apply(mi_run_params& rp) const {
    rp.control_images = images;
    rp.control_images_count = 2;
  }
};

// Control per run (design §5.3): a session with the control adapter and a MOSS run with control images and windows
// (encoded at mi_run_begin) gives mi_generate's bits, with the step cache off and on (the same cached steps: a window
// flip invalidates the cache on both sides); without control images anywhere the adapter session equals mi_generate
// with the adapter and no images (zero control latents).
TEST_F(GpuComponentRuns, ControlRunImagesEqualGenerate) {
  if (!fs::is_regular_file(kControl)) GTEST_SKIP() << "control adapter not present";
  const ControlInputs in;
  const ControlRequest ctl(in);
  for (int c = 0; c < 3; ++c) {
    const bool images = c < 2, cache = c == 1;
    SCOPED_TRACE(std::string(images ? "control images" : "no control images") + (cache ? ", step cache" : ""));
    mi_gen_params q = request(8);
    ctl.apply(q, images);
    q.step_cache = cache ? MI_STEP_CACHE_FBCACHE : MI_STEP_CACHE_OFF;
    q.step_cache_threshold = cache ? 0.3 : 0.0;
    mi_result* gr = nullptr;
    const std::vector<std::uint8_t> ref = generate(engine_, q, &gr);
    const std::vector<int> ref_cached = cached_steps(gr);
    mi_result_free(gr);
    if (cache) {
      EXPECT_FALSE(ref_cached.empty()) << "the threshold reused nothing: the test shows nothing";
    }
    ASSERT_EQ(open_session(engine_, 3, 1, ctl.adapter.c_str()), MI_OK) << last_error();
    {
      const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
      const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
      mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
      if (images) ctl.apply(rp);
      rp.step_cache = q.step_cache;
      rp.step_cache_threshold = q.step_cache_threshold;
      ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
      EXPECT_EQ(run_info(engine_).control_channels, 256);
      const Vec s = run_sigmas(engine_);
      Vec x;
      {
        Exchange ex(engine_, c == 2 ? Mode::kLegacy : Mode::kHost);
        x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0]), guided(ex));
      }
      mi_result* r = nullptr;
      end_run(engine_, &r);
      EXPECT_EQ(cached_steps(r), ref_cached);
      mi_result_free(r);
      EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's";
    }
    EXPECT_EQ(mi_session_close(engine_), MI_OK);
  }
}

// Control per call (design §5.3's recipe): the host resizes, replicates grayscale to RGB, encodes with mi_vae_encode,
// multiplies by fp32(strength), passes zeros outside each window and sets MI_STEP_CONTROL_CHANGED on the calls where a
// window opens or closes: mi_generate's control run bit for bit, in host staging and device mode, with the step cache
// off and on (the same cached steps). Without the flag at the flips, the step cache reuses across a changed input:
// other cached steps than mi_generate's, which shows the flag is load-bearing.
TEST_F(GpuComponentRuns, ControlPerCallRecipeEqualsGenerate) {
  if (!fs::is_regular_file(kControl)) GTEST_SKIP() << "control adapter not present";
  const ControlInputs in;
  const ControlRequest ctl(in);
  for (const bool cache : {false, true}) {
    SCOPED_TRACE(cache ? "step cache" : "no step cache");
    mi_gen_params q = request(8);
    ctl.apply(q, true);
    q.step_cache = cache ? MI_STEP_CACHE_FBCACHE : MI_STEP_CACHE_OFF;
    q.step_cache_threshold = cache ? 0.9 : 0.0;  // reuses whatever it may: only the flips force full calls
    mi_result* gr = nullptr;
    const std::vector<std::uint8_t> ref = generate(engine_, q, &gr);
    const std::vector<int> ref_cached = cached_steps(gr);
    mi_result_free(gr);
    ASSERT_EQ(open_session(engine_, 3, 1, ctl.adapter.c_str()), MI_OK) << last_error();
    {
      const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
      const Vec color = control_latent(engine_, in.color, kColorStrength);
      const Vec depth = control_latent(engine_, in.depth, kDepthStrength);
      const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
      for (int v = 0; v < (cache ? 3 : 2); ++v) {
        const Mode mode = v == 1 ? Mode::kLegacy : Mode::kHost;
        const bool flag = v < 2;
        SCOPED_TRACE(std::string(mode_name(mode)) + (flag ? "" : ", no MI_STEP_CONTROL_CHANGED"));
        mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
        rp.step_cache = q.step_cache;
        rp.step_cache_threshold = q.step_cache_threshold;
        ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
        const Vec s = run_sigmas(engine_);
        std::vector<int> got_cached;
        Vec x;
        {
          Exchange ex(engine_, mode);
          int step = 0;
          Vec previous(static_cast<std::size_t>(2 * kGroup), 0.0f);
          const Model m = [&](const Vec& xx, float sigma, int, Vec& pred) {
            const Vec control = control_at(color, depth, step, q.steps);
            Call c;
            c.sigma = sigma;
            c.control = &control;
            c.flags = flag && control != previous ? MI_STEP_CONTROL_CHANGED : 0u;
            ASSERT_EQ(ex.call(c, xx), MI_OK) << last_error();
            if (ex.last().cached) got_cached.push_back(step + 1);
            previous = control;
            pred = ex.out(0);
            ++step;
          };
          x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0]), m);
        }
        end_run(engine_);
        if (flag) {
          EXPECT_EQ(got_cached, ref_cached);
          EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's";
        } else {
          EXPECT_NE(got_cached, ref_cached) << "without the flag the cache should reuse across a window flip";
        }
      }
    }
    EXPECT_EQ(mi_session_close(engine_), MI_OK);
  }
}

// Control sources within one run (design §5.2): run-level control images, a crop call (control NULL evaluates zeros:
// the same as explicit zeros; a nonzero control differs), the run-level call again (the run's control rewritten after
// the crop overwrote it), a per-call control equal to the run's state (the same output) and the run's again; a control
// tensor of the wrong channel count is refused. Then per-latent control at L = 2: each image's output with its own
// control latent equals its single-image call, with the adapter's LoRA part at strength 0 and at 1: each latent runs
// its own forward, so its side branch sees its single-image call's tokens (LoraRunEqualsGenerate covers LoRA parity).
TEST_F(GpuComponentRuns, ControlSourcesCropsAndLatentBatches) {
  if (!fs::is_regular_file(kControl)) GTEST_SKIP() << "control adapter not present";
  const ControlInputs in;
  const ControlRequest ctl(in);
  ASSERT_EQ(open_session(engine_, 3, 2, ctl.adapter.c_str()), MI_OK) << last_error();
  const Conds conds = encode(engine_, {"a red apple on a wooden table, studio light", "blurry", "a cat"});
  const Vec color = control_latent(engine_, in.color, kColorStrength);
  const Vec depth = control_latent(engine_, in.depth, kDepthStrength);
  const Vec x = noise(4);
  {
    const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
    mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS, 8);
    ctl.apply(rp);
    rp.flags = MI_RUN_CROPS | MI_RUN_EXACT_SIGMA;
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Exchange ex(engine_, Mode::kHost);
    Call run_level;
    run_level.sigma = s[1];
    run_level.step = 1;  // color on, depth off
    ASSERT_EQ(ex.call(run_level, x), MI_OK) << last_error();
    const Vec want0 = ex.out(0), want1 = ex.out(1);

    constexpr std::int64_t kH = 24, kW = 20;
    const Vec xc = noise(5, static_cast<int>(kW * 16), static_cast<int>(kH * 16));
    Call crop = run_level;
    crop.h = kH;
    crop.w = kW;
    ASSERT_EQ(ex.call(crop, xc), MI_OK) << last_error();
    const Vec crop_null = ex.out(0);
    const Vec zeros(static_cast<std::size_t>(256 * kH * kW), 0.0f);
    crop.control = &zeros;
    ASSERT_EQ(ex.call(crop, xc), MI_OK) << last_error();
    EXPECT_EQ(differing(ex.out(0), crop_null), 0u) << "control NULL on a crop is not zeros";
    const Vec halves(static_cast<std::size_t>(256 * kH * kW), 0.5f);
    crop.control = &halves;
    ASSERT_EQ(ex.call(crop, xc), MI_OK) << last_error();
    EXPECT_GT(differing(ex.out(0), crop_null), 0u) << "a nonzero control changed nothing";

    ASSERT_EQ(ex.call(run_level, x), MI_OK) << last_error();
    EXPECT_EQ(differing(ex.out(0), want0), 0u) << "the run's control after a crop";
    const Vec state = control_at(color, depth, 1, 8);
    Call per_call = run_level;
    per_call.control = &state;
    per_call.flags = MI_STEP_CONTROL_CHANGED;
    ASSERT_EQ(ex.call(per_call, x), MI_OK) << last_error();
    EXPECT_EQ(differing(ex.out(0), want0), 0u) << "a per-call control equal to the run's";
    EXPECT_EQ(differing(ex.out(1), want1), 0u);
    ASSERT_EQ(ex.call(run_level, x), MI_OK) << last_error();
    EXPECT_EQ(differing(ex.out(0), want0), 0u) << "the run's control after a per-call one";

    // 128 channels where the adapter takes 256.
    mi_tensor xv{}, ov[2]{}, cv{};
    ASSERT_EQ(mi_run_staging(engine_, &xv, ov, 2, &cv), MI_OK);
    EXPECT_EQ(cv.shape[0], 256);
    mi_step_params st;
    ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
    st.sigma = s[1];
    st.x = xv;
    st.out[0] = ov[0];
    st.out[1] = ov[1];
    st.control = tensor(cv.data, MI_MEMORY_HOST, MI_DTYPE_F32, {128, kLat, kLat});
    EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
    EXPECT_TRUE(has(last_error(), "control: expected f32[256")) << last_error();
  }
  end_run(engine_);

  // Per-latent control at L = 2: latent 0 with both groups, latent 1 with zeros.
  const Vec both = [&] {
    Vec c(static_cast<std::size_t>(2 * kGroup));
    std::copy(color.begin(), color.end(), c.begin());
    std::copy(depth.begin(), depth.end(), c.begin() + kGroup);
    return c;
  }();
  const Vec none(static_cast<std::size_t>(2 * kGroup), 0.0f);
  const Vec xb = noise(6);
  double lora_strength = 0.0;
  const auto one_call = [&](const std::vector<mi_run_row>& rows, const Vec& xx, const Vec* control) {
    mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS);
    rp.control_lora_strength = lora_strength;
    EXPECT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    std::vector<Vec> outs;
    {
      Exchange ex(engine_, Mode::kLegacy);
      Call c;
      c.sigma = s[2];
      c.control = control;
      EXPECT_EQ(ex.call(c, xx), MI_OK) << last_error();
      for (int k = 0; k < ex.info().outputs; ++k) outs.push_back(ex.out(k));
    }
    end_run(engine_);
    return outs;
  };
  const Vec batch_control = concat({both, none});
  for (const double strength : {0.0, 1.0}) {
    SCOPED_TRACE("control LoRA strength " + std::to_string(strength));
    lora_strength = strength;
    const std::vector<Vec> batch = one_call({row(conds[0], nullptr, 0, 0), row(conds[2], nullptr, 0, 1)},
                                            concat({x, xb}), &batch_control);
    const std::vector<Vec> a = one_call({row(conds[0])}, x, &both);
    const std::vector<Vec> b = one_call({row(conds[2])}, xb, nullptr);
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(differing(batch[0], a[0]), 0u) << "latent 0 with its control";
    EXPECT_EQ(differing(batch[1], b[0]), 0u) << "latent 1 with zeros";
  }
}

// A runtime LoRA (design §7.2, "windows and adapters"): a MOSS run with a LoRA, sampled by the host, gives mi_generate's
// bits with the same LoRA, and so does the next run after the LoRA's strength changed (kept across runs while
// unchanged, switched in place otherwise).
TEST_F(GpuComponentRuns, LoraRunEqualsGenerate) {
  const fs::path file = "/workspace/Models/community/axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors";
  if (!fs::is_regular_file(file)) GTEST_SKIP() << "LoRA file not present";
  const std::string path = file.string();
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  for (const double strength : {0.8, 0.5}) {
    SCOPED_TRACE("strength " + std::to_string(strength));
    const mi_lora lora{path.c_str(), strength};
    mi_gen_params q = request();
    q.loras = &lora;
    q.loras_count = 1;
    const std::vector<std::uint8_t> ref = generate(engine_, q);
    const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
    const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
    mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
    rp.loras = &lora;
    rp.loras_count = 1;
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, Mode::kHost);
      x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0]), guided(ex));
    }
    end_run(engine_);
    EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's";
  }
}

// ---------------------------------------------------------------- image input, conditioning

// img2img (design §5.3's recipe): mi_image_resize, mi_vae_encode and the start formula x = s0 noise + (1 - s0) z0 on
// the run's schedule at strength 0.5 (mi_schedule's), then euler on moss: mi_generate's img2img bit for bit.
TEST_F(GpuComponentRuns, Img2ImgThroughVaeEncodeEqualsGenerate) {
  constexpr int iw = 640, ih = 560;
  std::vector<std::uint8_t> px(static_cast<std::size_t>(iw) * ih * 3);
  for (int y = 0; y < ih; ++y) {
    for (int x = 0; x < iw; ++x) {
      std::uint8_t* p = &px[(static_cast<std::size_t>(y) * iw + x) * 3];
      p[0] = static_cast<std::uint8_t>((x * 7 + y * 3) & 255);
      p[1] = static_cast<std::uint8_t>(y * 255 / (ih - 1));
      p[2] = static_cast<std::uint8_t>(255 - x * 255 / (iw - 1));
    }
  }
  const mi_image init{px.data(), iw, ih, 3, 0};
  mi_gen_params q = request(8);
  q.init_image = &init;
  q.denoise_strength = 0.5;
  const std::vector<std::uint8_t> ref = generate(engine_, q);
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  std::vector<std::uint8_t> sized(static_cast<std::size_t>(kSize) * kSize * 3);
  ASSERT_EQ(mi_image_resize(&init, kSize, kSize, MI_RESIZE_CROP, sized.data(), sized.size()), MI_OK) << last_error();
  const mi_image im{sized.data(), kSize, kSize, 3, 0};
  Vec z0(kN);
  const mi_tensor zt = tensor(z0.data(), MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat, kLat});
  ASSERT_EQ(mi_vae_encode(engine_, &im, &zt), MI_OK) << last_error();
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
  rp.denoise_strength = 0.5;
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  const Vec s = run_sigmas(engine_);
  std::size_t count = 0;
  ASSERT_EQ(mi_schedule(MI_SCHEDULER_MOSS, q.steps, 5.0, 1.0, 0.5, nullptr, 0, &count), MI_OK);
  Vec helper(count);
  ASSERT_EQ(mi_schedule(MI_SCHEDULER_MOSS, q.steps, 5.0, 1.0, 0.5, helper.data(), count, &count), MI_OK);
  EXPECT_EQ(differing(helper, s), 0u) << "mi_schedule differs from the run's schedule";
  EXPECT_LT(s.size(), static_cast<std::size_t>(q.steps) + 1);
  Vec x;
  {
    Exchange ex(engine_, Mode::kHost);
    x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0], z0), guided(ex));
  }
  end_run(engine_);
  EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's img2img";
}

// Foreign conditioning (design §7.2): engine conditionings exported with mi_conditioning_tensor and imported again with
// mi_conditioning_create (BF16 host, and F32 device memory) give the same bits; the report names both rows as foreign
// (the component object and MI_TEXT_COMPONENT_NOTE).
TEST_F(GpuComponentRuns, ForeignConditioningGivesTheSameBits) {
  const mi_gen_params q = request();
  const std::vector<std::uint8_t> ref = generate(engine_, q);
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  Conds foreign;
  for (std::size_t k = 0; k < 2; ++k) {
    mi_tensor t{};
    ASSERT_EQ(mi_conditioning_tensor(conds[k], &t), MI_OK) << last_error();
    ASSERT_EQ(t.dtype, MI_DTYPE_BF16);
    ASSERT_EQ(t.rank, 2);
    const std::size_t n = static_cast<std::size_t>(t.shape[0] * t.shape[1]);
    mi_conditioning* c = nullptr;
    if (k == 0) {
      const auto* bf16 = static_cast<const std::uint16_t*>(t.data);
      std::vector<std::uint16_t> copy(bf16, bf16 + n);
      const mi_tensor in = tensor(copy.data(), MI_MEMORY_HOST, MI_DTYPE_BF16, {t.shape[0], t.shape[1]});
      ASSERT_EQ(mi_conditioning_create(&in, &c), MI_OK) << last_error();
    } else {
      // bf16 -> f32 is exact; the import rounds it back to the same bf16.
      std::vector<float> f(n);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(t.data)[i]) << 16;
        std::memcpy(&f[i], &bits, sizeof bits);
      }
      void* dev = nullptr;
      ASSERT_EQ(cudaMalloc(&dev, n * sizeof(float)), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(dev, f.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
      const mi_tensor in = tensor(dev, MI_MEMORY_DEVICE, MI_DTYPE_F32, {t.shape[0], t.shape[1]});
      EXPECT_EQ(mi_conditioning_create(&in, &c), MI_OK) << last_error();
      cudaFree(dev);
    }
    foreign.c.push_back(c);
    mi_conditioning_info info{};
    info.struct_size = sizeof info;
    ASSERT_EQ(mi_conditioning_query(c, &info), MI_OK);
    EXPECT_EQ(info.engine_encoded, 0);
    EXPECT_EQ(info.tokens, t.shape[0]);
  }
  const std::vector<mi_run_row> rows = {row(foreign[0]), row(foreign[1], kSkip22, 1)};
  const mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  const Vec s = run_sigmas(engine_);
  Vec x;
  {
    Exchange ex(engine_, Mode::kHost);
    x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(noise(q.seed), s[0]), guided(ex));
  }
  mi_result* r = nullptr;
  const mi::json::Value comp = component_of(end_run(engine_, &r));
  ASSERT_EQ(comp.at("foreign_rows").as_array().size(), 2u);
  EXPECT_EQ(comp.at("foreign_rows").as_array()[0].as_int(), 0);
  EXPECT_EQ(comp.at("foreign_rows").as_array()[1].as_int(), 1);
  const std::string note = mi_result_text(r, MI_TEXT_COMPONENT_NOTE);
  EXPECT_TRUE(has(note, "Foreign conditioning") && has(note, "rows 0, 1")) << note;
  mi_result_free(r);
  EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's";
}

// ---------------------------------------------------------------- cancels, endings, races

struct BeginEvents {
  mi_engine* engine = nullptr;
  bool cancel_at_load = false;
  std::atomic<bool> loading{false};
};

void on_begin_progress(const mi_progress* p, void* user) {
  auto* ev = static_cast<BeginEvents*>(user);
  if (p->phase != MI_PHASE_DIT_LOAD) return;
  ev->loading.store(true);
  if (ev->cancel_at_load) mi_cancel(ev->engine);
}

// Cancels (design §7.2): mi_cancel during mi_run_begin, from its DiT-load progress callback and from another thread
// while the begin loads a released DiT, gives MI_CANCELLED and no run, and the next begin works (and reloads). A step
// cancelled mid-forward leaves the run usable: with the step cache off its repeat gives the uncancelled bits; with it
// on, the cancelled call invalidated the cache, so the repeat runs in full where it would have reused.
TEST_F(GpuComponentRuns, CancelDuringBeginAndMidStep) {
  BeginEvents ev;
  ev.engine = engine_;
  ASSERT_EQ(open_session(engine_, 3, 1, nullptr, on_begin_progress, &ev), MI_OK) << last_error();
  const mi_gen_params q = request(8);
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
  ev.cancel_at_load = true;
  EXPECT_EQ(mi_run_begin(engine_, &rp), MI_CANCELLED) << last_error();
  mi_run_info info{};
  info.struct_size = sizeof info;
  EXPECT_EQ(mi_run_query(engine_, &info), MI_ERR_INVALID_ARGUMENT);
  EXPECT_TRUE(has(last_error(), "no run is open")) << last_error();
  ev.cancel_at_load = false;
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  EXPECT_EQ(mi_cancel(engine_), MI_OK);  // between calls: a no-op
  EXPECT_EQ(mi_run_end(engine_, nullptr), MI_OK);

  // From another thread while the begin loads a released DiT: from its DiT-load phase on, the canceller repeats
  // mi_cancel every millisecond until the begin returns (a cancel before the engine marks the job running is a no-op,
  // so one fixed-time cancel could miss a fast reload), and gives up after 60 s or when the begin returns without
  // reaching the phase (a failure the status check reports, not a hang).
  ASSERT_EQ(mi_engine_release(engine_, 0, nullptr), MI_OK) << last_error();
  ev.loading.store(false);
  mi_engine* const e = engine_;
  std::atomic<bool> begin_returned{false};
  std::thread canceller([&] {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!ev.loading.load() && !begin_returned.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    while (ev.loading.load() && !begin_returned.load() && std::chrono::steady_clock::now() < deadline) {
      mi_cancel(e);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  const mi_status cancelled = mi_run_begin(engine_, &rp);
  begin_returned.store(true);
  canceller.join();
  EXPECT_TRUE(ev.loading.load()) << "the begin never reached its DiT-load phase";
  EXPECT_EQ(cancelled, MI_CANCELLED) << last_error();
  EXPECT_EQ(mi_run_query(engine_, &info), MI_ERR_INVALID_ARGUMENT);
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  mi_result* r = nullptr;
  end_run(engine_, &r);
  mi_memory_stats mem{};
  mem.struct_size = sizeof mem;
  ASSERT_EQ(mi_result_memory(r, &mem), MI_OK);
  EXPECT_EQ(mem.dit_loads, 1);
  mi_result_free(r);

  // Mid-step cancels: a worker repeats a step until a cancel lands in its forward.
  const auto cancel_one = [&](const Vec& x, const Call& c) {
    mi_tensor xv{}, ov[MI_MAX_ROWS]{};
    EXPECT_EQ(mi_run_staging(e, &xv, ov, MI_MAX_ROWS, nullptr), MI_OK);
    std::memcpy(xv.data, x.data(), x.size() * sizeof(float));
    mi_step_params st;
    EXPECT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
    st.sigma = c.sigma;
    st.step = c.step;
    st.flags = c.flags;
    st.x = xv;
    const mi_run_info ri = run_info(e);
    for (int k = 0; k < ri.outputs; ++k) st.out[k] = ov[k];
    // The probe never runs a forward (a NaN sigma is refused once the probe has the gate), so the worker's cancelled
    // call is the run's last.
    mi_step_params probe = st;
    probe.sigma = std::numeric_limits<float>::quiet_NaN();
    std::atomic<bool> done{false};
    mi_status status = MI_OK;
    std::thread worker([&] {
      do {
        status = mi_denoise_step(e, &st, nullptr);
      } while (status == MI_ERR_BUSY || status == MI_OK);
      done.store(true);
    });
    while (!done.load()) {
      if (mi_denoise_step(e, &probe, nullptr) == MI_ERR_BUSY) mi_cancel(e);  // the worker's step is in flight
    }
    worker.join();
    EXPECT_EQ(status, MI_CANCELLED) << last_error();
  };
  const Vec x = noise(q.seed);
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  {
    const Vec s = run_sigmas(engine_);
    Call c;
    c.sigma = s[1];
    c.step = 1;
    const Vec xs = start_latents(x, s[1]);
    Vec want;
    {
      Exchange ex(engine_, Mode::kHost);
      ASSERT_EQ(ex.call(c, xs), MI_OK) << last_error();
      want = ex.out(0);
    }
    cancel_one(xs, c);
    Exchange ex(engine_, Mode::kHost);
    ASSERT_EQ(ex.call(c, xs), MI_OK) << last_error();
    EXPECT_EQ(differing(ex.out(0), want), 0u) << "the repeat after a cancel differs";
  }
  end_run(engine_);

  rp.step_cache = MI_STEP_CACHE_FBCACHE;
  rp.step_cache_threshold = 0.9;
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  {
    const Vec s = run_sigmas(engine_);
    const Vec xs = start_latents(x, s[1]);
    Call full;
    full.sigma = s[1];
    full.step = 1;
    full.flags = MI_STEP_CONTROL_CHANGED;  // invalidates the cache: a full call (no adapter needed)
    Call plain = full;
    plain.flags = 0;
    {
      Exchange ex(engine_, Mode::kHost);
      ASSERT_EQ(ex.call(full, xs), MI_OK) << last_error();
      EXPECT_EQ(ex.last().cached, 0);
      ASSERT_EQ(ex.call(plain, xs), MI_OK) << last_error();
      EXPECT_EQ(ex.last().cached, 1) << "the same input after a full call should reuse";
    }
    cancel_one(xs, full);
    Exchange ex(engine_, Mode::kHost);
    ASSERT_EQ(ex.call(plain, xs), MI_OK) << last_error();
    EXPECT_EQ(ex.last().cached, 0) << "the cancelled call left the cache valid";
  }
  end_run(engine_);
}

int race_iterations() {
  const char* env = std::getenv("MOSS_RACE_ITERATIONS");
  return env != nullptr && *env != '\0' ? std::max(1, std::atoi(env)) : 100;
}

// The ending sequence under contention (design §4): while thread A repeats steps, B ends the run (or, every fourth
// iteration, closes the session). Each step of A either completes or fails with MI_CANCELLED (the ending cancelled it
// in flight) or at once with "the run is ending" / "the session is closing" / "no run is open" / "no session is
// open": never anything else, and never after the free (run under ASan and TSan with MOSS_RACE_ITERATIONS=1000).
TEST_F(GpuComponentRuns, EndingRaceIterations) {
  ASSERT_EQ(open_session(engine_, 1), MI_OK) << last_error();
  const Conds conds = encode(engine_, {"a red apple on a wooden table, studio light"});
  const std::vector<mi_run_row> rows = {row(conds[0])};
  const mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS, 4);
  mi_engine* const e = engine_;
  const int n = race_iterations();
  int cancelled = 0, refused = 0;
  for (int it = 0; it < n; ++it) {
    ASSERT_EQ(mi_run_begin(e, &rp), MI_OK) << last_error();
    mi_tensor xv{}, ov[1]{};
    ASSERT_EQ(mi_run_staging(e, &xv, ov, 1, nullptr), MI_OK);
    mi_step_params st;
    ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
    st.sigma = run_sigmas(e)[1];
    st.x = xv;
    st.out[0] = ov[0];
    std::atomic<int> completed{0};
    std::atomic<bool> stopped{false};
    mi_status last = MI_OK;
    std::string message;
    std::thread a([&] {
      for (;;) {
        last = mi_denoise_step(e, &st, nullptr);
        if (last == MI_OK) {
          ++completed;
          continue;
        }
        if (last == MI_ERR_BUSY) continue;  // never expected: no other caller steps
        message = last_error();
        stopped.store(true);
        return;
      }
    });
    while (completed.load() < it % 3 && !stopped.load()) std::this_thread::yield();
    const bool close = it % 4 == 3;
    EXPECT_EQ(close ? mi_session_close(e) : mi_run_end(e, nullptr), MI_OK) << last_error();
    a.join();
    if (last == MI_CANCELLED) {
      ++cancelled;
    } else {
      ++refused;
      EXPECT_EQ(last, MI_ERR_INVALID_ARGUMENT) << message;
      EXPECT_TRUE(has(message, "the run is ending") || has(message, "the session is closing") ||
                  has(message, "no run is open") || has(message, "no session is open"))
          << message;
    }
    if (close) {
      ASSERT_EQ(open_session(e, 1), MI_OK) << last_error();
    }
  }
  std::printf("[race] %d iterations: %d steps cancelled in flight, %d refused at the gate\n", n, cancelled, refused);
  // Enough iterations land an ending mid-forward at least once: otherwise the in-flight path went unexercised.
  if (n >= 100) {
    EXPECT_GT(cancelled, 0) << "no ending cancelled a step in flight";
  }
}

// After a step fails in device mode the engine no longer touches x, out[] or control (design §4): right after a
// cancelled step (and a refused one) the host overwrites the outputs on its stream; once the device is idle they hold
// what the host wrote. With MOSS_FREE_AFTER_FAIL=1 the buffers (stream-ordered allocations) are freed at once instead,
// for a compute-sanitizer memcheck run.
TEST_F(GpuComponentRuns, DeviceBuffersIdleAfterAFailedStep) {
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  // Foreign conditionings (zeros): no text encoder, which keeps a compute-sanitizer run short.
  std::vector<std::uint16_t> zeros(static_cast<std::size_t>(16) * MI_TEXT_DIM, 0);
  const mi_tensor text = tensor(zeros.data(), MI_MEMORY_HOST, MI_DTYPE_BF16, {16, MI_TEXT_DIM});
  Conds conds;
  conds.c.assign(2, nullptr);
  ASSERT_EQ(mi_conditioning_create(&text, &conds.c[0]), MI_OK) << last_error();
  ASSERT_EQ(mi_conditioning_create(&text, &conds.c[1]), MI_OK) << last_error();
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  const mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS, 8);
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  const bool free_after = std::getenv("MOSS_FREE_AFTER_FAIL") != nullptr;
  mi_engine* const e = engine_;
  cudaStream_t hs = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&hs, cudaStreamNonBlocking), cudaSuccess);
  const std::size_t bytes = kN * sizeof(float);
  const Vec s = run_sigmas(e);
  for (int attempt = 0; attempt < 2; ++attempt) {  // 0: a cancelled step; 1: a refused one (out[1] the wrong shape)
    void* dx = nullptr;
    void* d0 = nullptr;
    void* d1 = nullptr;
    ASSERT_EQ(cudaMallocAsync(&dx, bytes, hs), cudaSuccess);
    ASSERT_EQ(cudaMallocAsync(&d0, bytes, hs), cudaSuccess);
    ASSERT_EQ(cudaMallocAsync(&d1, bytes, hs), cudaSuccess);
    const Vec x = start_latents(noise(3), s[1]);
    ASSERT_EQ(cudaMemcpyAsync(dx, x.data(), bytes, cudaMemcpyHostToDevice, hs), cudaSuccess);
    mi_step_params st;
    ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
    st.sigma = s[1];
    st.cuda_stream = hs;
    st.x = tensor(dx, MI_MEMORY_DEVICE, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat, kLat});
    st.out[0] = tensor(d0, MI_MEMORY_DEVICE, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat, kLat});
    st.out[1] = tensor(d1, MI_MEMORY_DEVICE, MI_DTYPE_F32, {MI_LATENT_CHANNELS, kLat, kLat - (attempt == 1 ? 1 : 0)});
    std::atomic<bool> done{false};
    mi_status status = MI_OK;
    std::thread worker([&] {
      do {
        status = mi_denoise_step(e, &st, nullptr);
      } while (status == MI_OK || status == MI_ERR_BUSY);
      // At once on the failing return: the host reuses or frees its buffers.
      if (free_after) {
        cudaFreeAsync(dx, hs);
        cudaFreeAsync(d0, hs);
        cudaFreeAsync(d1, hs);
      } else {
        cudaMemsetAsync(d0, 0x5A, bytes, hs);
        cudaMemsetAsync(d1, 0x5A, bytes, hs);
      }
      done.store(true);
    });
    mi_tensor xv{}, ov[2]{};
    ASSERT_EQ(mi_run_staging(e, &xv, ov, 2, nullptr), MI_OK);
    mi_step_params probe;
    ASSERT_EQ(mi_step_params_init(&probe, sizeof probe), MI_OK);
    probe.sigma = s[1];
    probe.x = xv;
    probe.out[0] = ov[0];
    probe.out[1] = ov[1];
    while (!done.load()) {
      if (attempt == 0 && mi_denoise_step(e, &probe, nullptr) == MI_ERR_BUSY) mi_cancel(e);
    }
    worker.join();
    EXPECT_EQ(status, attempt == 0 ? MI_CANCELLED : MI_ERR_INVALID_ARGUMENT) << last_error();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    if (!free_after) {
      std::vector<std::uint8_t> h(bytes);
      for (void* d : {d0, d1}) {
        ASSERT_EQ(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(std::count(h.begin(), h.end(), std::uint8_t{0x5A}), static_cast<std::ptrdiff_t>(bytes))
            << "the engine wrote an output after the step returned";
      }
      cudaFreeAsync(dx, hs);
      cudaFreeAsync(d0, hs);
      cudaFreeAsync(d1, hs);
    }
  }
  ASSERT_EQ(cudaStreamSynchronize(hs), cudaSuccess);
  cudaStreamDestroy(hs);
  // The run is still usable.
  Exchange ex(e, Mode::kLegacy);
  Call c;
  c.sigma = s[1];
  EXPECT_EQ(ex.call(c, start_latents(noise(3), s[1])), MI_OK) << last_error();
}

// ---------------------------------------------------------------- conventions

// The FLOW convention against ComfyUI's (design §5.2): the guided denoised output of a MOSS cfg-1 call is ComfyUI's
// calculate_denoised of the call's raw output, model_input - model_output * sigma (one rounding per op), bit for bit.
TEST_F(GpuComponentRuns, DenoisedIsComfyUisFlowConvention) {
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {"a red apple on a wooden table, studio light"});
  const std::vector<mi_run_row> rows = {row(conds[0])};
  const Vec x0 = noise(21);
  Vec outs[2];  // the raw output (ROWS), the guided denoised output
  float sigma = 0;
  for (int k = 0; k < 2; ++k) {
    const mi_step_output output = k == 0 ? MI_STEP_OUTPUT_ROWS : MI_STEP_OUTPUT_GUIDED_DENOISED;
    const mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, output);
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    sigma = run_sigmas(engine_)[2];
    {
      Exchange ex(engine_, Mode::kHost);
      Call c;
      c.sigma = sigma;
      ASSERT_EQ(ex.call(c, start_latents(x0, sigma)), MI_OK) << last_error();
      outs[k] = ex.out(0);
    }
    end_run(engine_);
  }
  const Vec x = start_latents(x0, sigma);
  Vec denoised(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float t = outs[0][i] * sigma;  // model_input - model_output * sigma
    denoised[i] = x[i] - t;
  }
  EXPECT_EQ(differing(outs[1], denoised), 0u) << "GUIDED_DENOISED is not x - out * sigma";
}

// The VAE oracle (design §7.2), skipped without the reference files (tests/data/component_vae, comfy.json): mi_vae_encode
// and mi_vae_decode against ComfyUI's actual VAE.encode / VAE.decode, on an engine loading the same VAE file. Model
// space is ComfyUI's LATENT as is (the same channel order, patchify and batch-norm statistics): the engine's encode of
// the image is close to ComfyUI's latent, and far closer than the same latent with the 2 x 2 patch order transposed;
// its decode of ComfyUI's latent is close to ComfyUI's decode. The numerics differ by design (the engine's exact
// GroupNorm, fp32 against ComfyUI's dtype), so the test bounds and prints the measured difference.
TEST_F(GpuComponentRuns, VaeOracleAgainstComfyUi) {
  const fs::path dir = fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "component_vae";
  const fs::path vae = kCommunity / "flux2-vae.safetensors";
  if (!fs::is_regular_file(dir / "comfy_encode.latent.npy") || !fs::is_regular_file(vae)) {
    GTEST_SKIP() << "ComfyUI reference files or flux2-vae.safetensors not present";
  }
  const mi::npy::Array lat_file = mi::npy::load(dir / "comfy_encode.latent.npy");
  const mi::npy::Array rgb_file = mi::npy::load(dir / "comfy_decode.rgb.npy");
  constexpr int W = 384, H = 256;
  ASSERT_EQ(lat_file.shape, (std::vector<std::int64_t>{MI_LATENT_CHANNELS, H / 16, W / 16}));
  ASSERT_EQ(rgb_file.shape, (std::vector<std::int64_t>{H, W, 3}));
  const Vec comfy_latent = lat_file.to_vector<float>();
  const std::vector<std::uint8_t> comfy_rgb = rgb_file.to_vector<std::uint8_t>();
  std::vector<std::uint8_t> px(static_cast<std::size_t>(W) * H * 3);  // the generator's procedural image
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      std::uint8_t* p = &px[(static_cast<std::size_t>(y) * W + x) * 3];
      p[0] = static_cast<std::uint8_t>(x * 255 / (W - 1));
      p[1] = static_cast<std::uint8_t>(y * 255 / (H - 1));
      p[2] = static_cast<std::uint8_t>(((x / 16 + y / 16) % 2) * 180 + 40);
    }
  }
  release_shared();
  mi_engine* e = create(kInt8, MI_TIER_HIGH, MI_ATTENTION_Q8_FP8, 0, MI_BLOCK_PATH_DEFAULT, MI_MODE_BALANCED, vae);
  ASSERT_NE(e, nullptr);
  ASSERT_EQ(open_session(e), MI_OK) << last_error();
  const mi_image im{px.data(), W, H, 3, 0};
  Vec z(comfy_latent.size());
  const mi_tensor zt = tensor(z.data(), MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, H / 16, W / 16});
  ASSERT_EQ(mi_vae_encode(e, &im, &zt), MI_OK) << last_error();
  std::vector<std::uint8_t> rgb(px.size());
  Vec lat = comfy_latent;
  const mi_tensor lt = tensor(lat.data(), MI_MEMORY_HOST, MI_DTYPE_F32, {MI_LATENT_CHANNELS, H / 16, W / 16});
  const mi_tensor rt = tensor(rgb.data(), MI_MEMORY_HOST, MI_DTYPE_U8, {H, W, 3});
  ASSERT_EQ(mi_vae_decode(e, &lt, &rt), MI_OK) << last_error();
  EXPECT_EQ(mi_session_close(e), MI_OK);
  mi_engine_free(e);

  // Relative L2 error of `a` against ComfyUI's latent, optionally with each 2 x 2 patch's dy and dx swapped.
  const std::size_t hw = static_cast<std::size_t>(H / 16) * (W / 16);
  const auto rel = [&](const Vec& a, bool transposed) {
    double num = 0, den = 0;
    for (std::size_t c = 0; c < static_cast<std::size_t>(MI_LATENT_CHANNELS); ++c) {
      const std::size_t base = c / 4, dy = (c % 4) / 2, dx = c % 2;
      const std::size_t src = transposed ? base * 4 + 2 * dx + dy : c;
      for (std::size_t i = 0; i < hw; ++i) {
        const double d = static_cast<double>(a[src * hw + i]) - comfy_latent[c * hw + i];
        num += d * d;
        den += static_cast<double>(comfy_latent[c * hw + i]) * comfy_latent[c * hw + i];
      }
    }
    return std::sqrt(num / den);
  };
  const double encode_err = rel(z, false), transposed_err = rel(z, true);
  double abs_sum = 0;
  int max_abs = 0;
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    const int d = std::abs(static_cast<int>(rgb[i]) - static_cast<int>(comfy_rgb[i]));
    abs_sum += d;
    max_abs = std::max(max_abs, d);
  }
  const double mean_abs = abs_sum / static_cast<double>(rgb.size());
  std::printf("[vae oracle] encode: relative L2 error %.3g against ComfyUI's latent (%.3g with the patch order "
              "transposed); decode of ComfyUI's latent: mean |RGB diff| %.3f, max %d\n",
              encode_err, transposed_err, mean_abs, max_abs);
  EXPECT_LT(encode_err, 0.02);
  EXPECT_GT(transposed_err, 10 * encode_err);
  EXPECT_LT(mean_abs, 1.0);
  EXPECT_LE(max_abs, 16);
}


// ---------------------------------------------------------------- §7.2 additions

// GUIDED_DENOISED on the moss scheduler: the engine's guidance in denoised space, sampled with euler's denoised form,
// equals the host's per-op denoised combine (D = x - v sigma per row, Du + (Dc - Du) g) of the ROWS outputs, bit for
// bit; and it is a sample of the same image as mi_generate's euler on moss (velocity form): within rounding (measured
// on the dev box: a mean of 1.9 RGB levels).
TEST_F(GpuComponentRuns, GuidedDenoisedOnMossEqualsTheHostsCombine) {
  const mi_gen_params q = request();
  const std::vector<std::uint8_t> ref = generate(engine_, q);
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  const Vec x0 = noise(q.seed);
  const auto run = [&](mi_step_output output) {
    const mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, output, q.steps);
    EXPECT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, Mode::kHost);
      const Model m = output == MI_STEP_OUTPUT_ROWS ? moss_rows(ex, false, 3.0, 0.0) : guided(ex);
      x = sample(MI_SAMPLER_EULER, false, s.data(), s.size(), start_latents(x0, s[0]), m);
    }
    end_run(engine_);
    return x;
  };
  const Vec engine_combine = run(MI_STEP_OUTPUT_GUIDED_DENOISED);
  const Vec host_combine = run(MI_STEP_OUTPUT_ROWS);
  EXPECT_EQ(differing(engine_combine, host_combine), 0u) << "GUIDED_DENOISED differs from the host's denoised combine";
  const std::vector<std::uint8_t> rgb = decode(engine_, engine_combine);
  ASSERT_EQ(rgb.size(), ref.size());
  double sum = 0;
  for (std::size_t i = 0; i < rgb.size(); ++i) sum += std::abs(static_cast<int>(rgb[i]) - static_cast<int>(ref[i]));
  // The two forms differ by rounding only (the denoised form divides by sigma): a few levels on average.
  EXPECT_LT(sum / static_cast<double>(rgb.size()), 4.0) << "the denoised-form sample is not mi_generate's image";
}

// Arbitrary sigma (design §7.2): a DPM2 host loop (a midpoint call per step, at exp of the log-sigmas' mean) and a
// UniPC-style predictor-corrector of order 2 (a multistep predictor and a corrector that reuses the next call's
// evaluation: one call per step, on the schedule) run with no error and no flag. The off-schedule calls are counted;
// the samples are finite.
TEST_F(GpuComponentRuns, Dpm2AndUniPcHostLoops) {
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const mi_gen_params q = request(8);
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_DENOISED, q.steps);
  rp.flags = 0;  // off-schedule calls allowed
  const auto finite = [](const Vec& v) { return std::all_of(v.begin(), v.end(), [](float f) { return std::isfinite(f); }); };

  // DPM2 (k-diffusion's sample_dpm_2 on a denoised prediction).
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  Vec s = run_sigmas(engine_);
  Vec x;
  {
    Exchange ex(engine_, Mode::kHost);
    const Model m = guided(ex);
    x = start_latents(noise(q.seed), s[0]);
    Vec d1, d2, x2(x.size());
    for (std::size_t i = 0; i + 1 < s.size(); ++i) {
      const float si = s[i], sn = s[i + 1];
      m(x, si, -1, d1);
      if (sn == 0.0f) {
        for (std::size_t k = 0; k < x.size(); ++k) x[k] = x[k] + ((x[k] - d1[k]) / si) * (sn - si);
        continue;
      }
      const float mid = static_cast<float>(std::exp(0.5 * (std::log(static_cast<double>(si)) + std::log(static_cast<double>(sn)))));
      for (std::size_t k = 0; k < x.size(); ++k) x2[k] = x[k] + ((x[k] - d1[k]) / si) * (mid - si);
      m(x2, mid, -1, d2);
      for (std::size_t k = 0; k < x.size(); ++k) x[k] = x[k] + ((x2[k] - d2[k]) / mid) * (sn - si);
    }
  }
  mi::json::Value comp = component_of(end_run(engine_));
  EXPECT_EQ(comp.at("calls").as_int(), 2 * q.steps - 1);
  EXPECT_EQ(comp.at("off_schedule_calls").as_int(), q.steps - 1);  // the midpoints; the last step has none
  EXPECT_TRUE(finite(x));

  // UniPC-style, in the flow parametrization: alpha = 1 - s, lambda = log(alpha / s). Predictor: DPM-Solver++(2M)
  // from the last two predictions; corrector: the trapezoid of this step's prediction and the next point's (the next
  // call's evaluation, at the predicted x), applied to x before the next step. The first step (s = 1) and the last (to
  // 0) are first order.
  ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
  s = run_sigmas(engine_);
  {
    Exchange ex(engine_, Mode::kHost);
    const Model m = guided(ex);
    x = start_latents(noise(q.seed), s[0]);
    const auto lambda = [](double sig) { return std::log((1.0 - sig) / sig); };
    Vec d, d_prev, x_prev;
    double h_prev = 0;
    m(x, s[0], -1, d);
    for (std::size_t i = 0; i + 1 < s.size(); ++i) {
      const double si = s[i], sn = s[i + 1];
      Vec xn(x.size());
      if (sn == 0.0) {
        xn = d;
      } else if (si >= 1.0) {
        for (std::size_t k = 0; k < x.size(); ++k) xn[k] = x[k] + static_cast<float>((x[k] - d[k]) / si * (sn - si));
      } else {
        const double h = lambda(sn) - lambda(si);
        const double a = sn / si, b = (1.0 - sn) * std::expm1(-h);
        const bool second = !d_prev.empty() && h_prev > 0;  // a previous step in lambda (not the first-order s = 1 one)
        const double r = second ? h_prev / h : 1.0;
        for (std::size_t k = 0; k < x.size(); ++k) {
          const double dd = second ? (1.0 + 1.0 / (2.0 * r)) * d[k] - (1.0 / (2.0 * r)) * d_prev[k] : d[k];
          xn[k] = static_cast<float>(a * x[k] - b * dd);
        }
        h_prev = h;
      }
      d_prev = d;
      x_prev = x;
      if (sn == 0.0) {
        x = xn;
        break;
      }
      m(xn, s[i + 1], static_cast<int>(i + 1), d);  // the next point's evaluation, used by the corrector and the next step
      if (si < 1.0) {
        const double h = lambda(sn) - lambda(si);
        const double a = sn / si, b = (1.0 - sn) * std::expm1(-h);
        for (std::size_t k = 0; k < x.size(); ++k) {
          xn[k] = static_cast<float>(a * x_prev[k] - b * 0.5 * (static_cast<double>(d_prev[k]) + d[k]));
        }
      }
      x = xn;
    }
  }
  comp = component_of(end_run(engine_));
  EXPECT_EQ(comp.at("calls").as_int(), q.steps);
  EXPECT_EQ(comp.at("off_schedule_calls").as_int(), 0);
  EXPECT_TRUE(finite(x));
}

// The step cache at L = 2 (design §7.2): residuals per latent, and two rows with the same text on different latents in
// separate slots. Two identical images in one run reuse exactly as the image alone (the same cached calls, each image
// its single run's bits); two different images swapped between the latents swap their outputs bit for bit with the
// same cached calls (shared slots would hand one image's residual to the other).
TEST_F(GpuComponentRuns, StepCacheAtTwoLatents) {
  ASSERT_EQ(open_session(engine_, 2, 2), MI_OK) << last_error();
  const Conds conds = encode(engine_, {"a red apple on a wooden table, studio light"});
  const Vec a = noise(7), b = noise(8);
  const auto run = [&](const std::vector<mi_run_row>& rows, const Vec& x0, std::vector<int>& cached) {
    mi_run_params rp = run_params(rows, MI_GUIDANCE_HOST, MI_STEP_OUTPUT_ROWS, 8);
    rp.step_cache = MI_STEP_CACHE_FBCACHE;
    rp.step_cache_threshold = 0.9;  // reuses whatever it may
    EXPECT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, Mode::kHost);
      int step = 0;
      const Model m = [&](const Vec& xx, float sigma, int, Vec& pred) {
        Call c;
        c.sigma = sigma;
        ASSERT_EQ(ex.call(c, xx), MI_OK) << last_error();
        if (ex.last().cached) cached.push_back(step);
        ++step;
        pred.clear();
        for (int k = 0; k < ex.info().outputs; ++k) pred.insert(pred.end(), ex.out(k).begin(), ex.out(k).end());
      };
      x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(x0, s[0]), m);
    }
    end_run(engine_);
    return x;
  };
  std::vector<int> cached_single, cached_same, cached_ab, cached_ba;
  const Vec single = run({row(conds[0])}, a, cached_single);
  EXPECT_FALSE(cached_single.empty()) << "the threshold reused nothing: the test shows nothing";
  const std::vector<mi_run_row> two = {row(conds[0]), row(conds[0], nullptr, 0, 1)};
  const Vec same = run(two, concat({a, a}), cached_same);
  EXPECT_EQ(cached_same, cached_single);
  EXPECT_EQ(differing(slice(same, 0), single), 0u) << "latent 0 differs from the image alone";
  EXPECT_EQ(differing(slice(same, 1), single), 0u) << "latent 1 differs from the image alone";
  const Vec ab = run(two, concat({a, b}), cached_ab);
  const Vec ba = run(two, concat({b, a}), cached_ba);
  EXPECT_FALSE(cached_ab.empty());
  EXPECT_EQ(cached_ab, cached_ba);
  EXPECT_GT(differing(slice(ab, 0), slice(ab, 1)), 0u);
  EXPECT_EQ(differing(slice(ab, 0), slice(ba, 1)), 0u) << "image a depends on its latent slot";
  EXPECT_EQ(differing(slice(ab, 1), slice(ba, 0)), 0u) << "image b depends on its latent slot";
}

// Interrupts (design §7.2): a run abandoned between calls, a run whose step was cancelled mid-forward, and a run whose
// call failed (a bad shape), each ended with mi_run_end; then a new mi_run_begin of the same request loads nothing
// (dit_loads 0) and gives mi_generate's bits.
TEST_F(GpuComponentRuns, AbandonCancelOrFailThenRebeginEqualsGenerate) {
  const mi_gen_params q = request();
  const std::vector<std::uint8_t> ref = generate(engine_, q);
  ASSERT_EQ(open_session(engine_), MI_OK) << last_error();
  const Conds conds = encode(engine_, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  const mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
  const Vec x0 = noise(q.seed);
  for (const char* how : {"abandoned", "cancelled", "failed"}) {
    SCOPED_TRACE(how);
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    {
      const Vec s = run_sigmas(engine_);
      Exchange ex(engine_, Mode::kHost);
      Vec x = start_latents(x0, s[0]), pred;
      const Model m = guided(ex);
      for (int i = 0; i < 2; ++i) {  // two steps, then the interruption
        m(x, s[static_cast<std::size_t>(i)], -1, pred);
        const float h = static_cast<float>(static_cast<double>(s[static_cast<std::size_t>(i) + 1]) -
                                           static_cast<double>(s[static_cast<std::size_t>(i)]));
        for (std::size_t k = 0; k < x.size(); ++k) x[k] = x[k] + h * pred[k];
      }
      Call c;
      c.sigma = s[2];
      if (std::string(how) == "cancelled") {
        // A canceller repeats mi_cancel until the step it races returns cancelled (a cancel between calls is a no-op).
        std::atomic<bool> stop{false};
        std::thread canceller([&] {
          while (!stop.load()) {
            mi_cancel(engine_);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
          }
        });
        mi_status st = MI_OK;
        for (int attempt = 0; attempt < 100 && st != MI_CANCELLED; ++attempt) st = ex.call(c, x);
        stop.store(true);
        canceller.join();
        EXPECT_EQ(st, MI_CANCELLED);
      } else if (std::string(how) == "failed") {
        mi_tensor xv{}, ov[MI_MAX_ROWS]{};
        ASSERT_EQ(mi_run_staging(engine_, &xv, ov, MI_MAX_ROWS, nullptr), MI_OK);
        mi_step_params st;
        ASSERT_EQ(mi_step_params_init(&st, sizeof st), MI_OK);
        st.sigma = s[2];
        st.x = xv;
        st.x.shape[2] = kLat + 1;  // wider than the run
        st.out[0] = ov[0];
        EXPECT_EQ(mi_denoise_step(engine_, &st, nullptr), MI_ERR_INVALID_ARGUMENT);
      }
    }
    end_run(engine_);
    ASSERT_EQ(mi_run_begin(engine_, &rp), MI_OK) << last_error();
    const Vec s = run_sigmas(engine_);
    Vec x;
    {
      Exchange ex(engine_, Mode::kHost);
      x = sample(MI_SAMPLER_EULER, true, s.data(), s.size(), start_latents(x0, s[0]), guided(ex));
    }
    mi_result* r = nullptr;
    end_run(engine_, &r);
    mi_memory_stats mem{};
    mem.struct_size = sizeof mem;
    ASSERT_EQ(mi_result_memory(r, &mem), MI_OK);
    mi_result_free(r);
    EXPECT_EQ(mem.dit_loads, 0);
    EXPECT_EQ(differing(decode(engine_, x), ref), 0u) << "RGB bytes differ from mi_generate's after the interruption";
  }
}

// ---------------------------------------------------------------- planning and memory

// The plan JSON of a report without live machine state, which moves between any two calls.
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

// The plan's decisions: its JSON without the time estimates and the explanation, which follow the transfer rates the
// engine learns from every job (a streamed plan's copy time moves between two calls).
void dump_decisions(const mi::json::Value& v, std::string& out) {
  if (v.is_object()) {
    out += "{";
    for (const auto& [k, m] : v.as_object()) {
      if (k == "time_s" || k == "explanation" || k == "transfer_s_per_step" || k == "compute_s_per_step" ||
          k == "expected_hidden" || k == "arena_cap_bytes" || k == "host_private_bytes" || k == "host_limit_bytes") {
        continue;
      }
      out += k + ":";
      dump_decisions(m, out);
      out += ",";
    }
    out += "}";
  } else if (v.is_array()) {
    out += "[";
    for (const mi::json::Value& e : v.as_array()) {
      dump_decisions(e, out);
      out += ",";
    }
    out += "]";
  } else if (v.is_string()) {
    out += "\"" + v.as_string() + "\"";
  } else if (v.is_bool()) {
    out += v.as_bool() ? "true" : "false";
  } else if (v.is_null()) {
    out += "null";
  } else {
    out += std::to_string(v.as_double());
  }
}

std::string plan_decisions(const std::string& report) {
  std::string out;
  dump_decisions(mi::json::parse(report).at("plan"), out);
  return out;
}

struct Facts {
  std::string plan, decisions;
  std::int64_t arena_creations = -1, dit_loads = -1;
  std::uint64_t arena_bytes = 0, device_peak = 0;
};

Facts facts_of(mi_result* r) {
  Facts f;
  const std::string report = r != nullptr ? mi_result_report_json(r) : "";
  f.plan = plan_facts(report);
  if (r == nullptr) return f;
  f.decisions = plan_decisions(report);
  const mi::json::Value mem = mi::json::parse(report).at("memory_gib");
  f.arena_creations = mem.at("arena_creations").as_int();
  mi_memory_stats m{};
  m.struct_size = sizeof m;
  EXPECT_EQ(mi_result_memory(r, &m), MI_OK);
  f.dit_loads = m.dit_loads;
  f.arena_bytes = m.arena_capacity_bytes;
  f.device_peak = m.device_peak_bytes;
  mi_result_free(r);
  return f;
}

Facts generate_facts(mi_engine* e, const mi_gen_params& q) {
  mi_result* r = nullptr;
  EXPECT_EQ(mi_generate(e, &q, &r), MI_OK) << last_error();
  return facts_of(r);
}

// The MOSS run equivalent to `q` (the request's size and steps), sampled in host staging; its result's facts.
Facts run_facts(mi_engine* e, const mi_gen_params& q) {
  const Conds conds = encode(e, {q.prompt, q.negative_prompt});
  const std::vector<mi_run_row> rows = {row(conds[0]), row(conds[1], kSkip22, 1)};
  mi_run_params rp = run_params(rows, MI_GUIDANCE_MOSS, MI_STEP_OUTPUT_GUIDED_VELOCITY, q.steps);
  rp.width = q.width;
  rp.height = q.height;
  EXPECT_EQ(mi_run_begin(e, &rp), MI_OK) << last_error();
  const Vec s = run_sigmas(e);
  {
    Exchange ex(e, Mode::kHost);
    sample(MI_SAMPLER_EULER, true, s.data(), s.size(),
           start_latents(noise(q.seed, q.width, q.height), s[0]), guided(ex));
  }
  mi_result* r = nullptr;
  end_run(e, &r);
  return facts_of(r);
}

mi_status open_planned(mi_engine* e, const mi_gen_params* plan, int max_rows = 3, int max_latents = 1) {
  mi_session_params sp;
  EXPECT_EQ(mi_session_params_init(&sp, sizeof sp), MI_OK);
  sp.plan_request = plan;
  sp.max_rows = max_rows;
  sp.max_latents = max_latents;
  return mi_session_open(e, &sp);
}

// With max_rows = 8 and max_latents = 4 at the session, a 2-row run plans exactly as with max_rows = 2: the same plan
// JSON and arena (the session's bounds size only the pinned staging, which follows each run's own rows).
TEST_F(GpuComponentRuns, SessionBoundsDoNotChangeTheRunsPlan) {
  const mi_gen_params q = request();
  generate_facts(engine_, q);  // the held state both sessions plan against
  ASSERT_EQ(open_planned(engine_, &q, 8, 4), MI_OK) << last_error();
  const Facts wide = run_facts(engine_, q);
  ASSERT_EQ(mi_session_close(engine_), MI_OK);
  ASSERT_EQ(open_planned(engine_, &q, 2, 1), MI_OK) << last_error();
  const Facts narrow = run_facts(engine_, q);
  EXPECT_EQ(wide.plan, narrow.plan);
  EXPECT_EQ(wide.arena_bytes, narrow.arena_bytes);
  EXPECT_NE(wide.plan, "no plan");
}

// Planning and memory against mi_generate on fresh engines (design §7.2): over three images at 512^2, 768^2, 512^2,
// each run's arena creations and DiT loads equal mi_generate's for the same image, its plan JSON too, and its device
// peak is at most mi_generate's (host staging: no device buffers of the host's; 64 MiB of slack for the free-memory
// measurement), with plan_request = the first image's request; with plan_request at the default 1024^2, at most one
// extra arena creation before the first run (the arena, and so the peak, then follows the 1024^2 plan). The first image is compared for its DiT loads and peak only: its arena was created by
// mi_encode_text, and mi_generate's first plan runs the text encoder where the run plans an embedding-cache hit. Then
// min_memory with a small budget (the DiT streamed): the run's plan decisions equal mi_generate's (the plan JSON
// without the time estimates, which follow the transfer rates learned from every job).
TEST_F(GpuComponentRuns, PlanningAndMemoryEqualGenerate) {
  release_shared();
  const int sizes[] = {512, 768, 512};
  const auto request_at = [](int size) {
    mi_gen_params q = request(4);
    q.width = q.height = size;
    return q;
  };
  std::vector<Facts> gen;
  {
    mi_engine* e = create(kInt8, MI_TIER_HIGH, MI_ATTENTION_Q8_FP8);
    ASSERT_NE(e, nullptr);
    for (const int size : sizes) gen.push_back(generate_facts(e, request_at(size)));
    mi_engine_free(e);
  }
  const mi_gen_params first = request_at(sizes[0]);
  for (const bool matched : {true, false}) {
    SCOPED_TRACE(matched ? "plan_request = the first image" : "plan_request = the defaults (1024^2)");
    mi_engine* e = create(kInt8, MI_TIER_HIGH, MI_ATTENTION_Q8_FP8);
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(open_planned(e, matched ? &first : nullptr), MI_OK) << last_error();
    std::int64_t gen_creations = 0, run_creations = 0;
    for (std::size_t i = 0; i < 3; ++i) {
      SCOPED_TRACE("image " + std::to_string(i) + " at " + std::to_string(sizes[i]));
      const Facts run = run_facts(e, request_at(sizes[i]));
      gen_creations += gen[i].arena_creations;
      run_creations += run.arena_creations;
      std::printf("[plan] %s, image %zu at %d: arena creations %lld / %lld, DiT loads %lld / %lld, device peak %.3f / "
                  "%.3f GiB (run / generate)\n", matched ? "matched" : "default plan_request", i, sizes[i],
                  static_cast<long long>(run.arena_creations), static_cast<long long>(gen[i].arena_creations),
                  static_cast<long long>(run.dit_loads), static_cast<long long>(gen[i].dit_loads),
                  static_cast<double>(run.device_peak) / (1 << 30), static_cast<double>(gen[i].device_peak) / (1 << 30));
      EXPECT_EQ(run.dit_loads, gen[i].dit_loads);
      if (matched && i > 0) {
        // The first image's arena was created by mi_encode_text (outside the run's result), and mi_generate's first
        // plan runs the text encoder where the run's plans an embedding-cache hit: from the second image on, both
        // plan against the same held state.
        EXPECT_EQ(run.arena_creations, gen[i].arena_creations);
        EXPECT_EQ(run.plan, gen[i].plan);
      }
      if (matched) {
        // The session's arena follows plan_request until a run reshapes it: with the 1024^2 default it is larger than
        // a 512^2 generate's.
        EXPECT_LE(run.device_peak, gen[i].device_peak + (std::uint64_t{64} << 20));
      }
    }
    EXPECT_LE(run_creations, gen_creations + (matched ? 0 : 1));
    EXPECT_EQ(mi_session_close(e), MI_OK);
    mi_engine_free(e);
  }

  mi_engine* e = create(kInt8, MI_TIER_HIGH, MI_ATTENTION_Q8_FP8, std::uint64_t{6} << 30, MI_BLOCK_PATH_DEFAULT,
                        MI_MODE_MIN_MEMORY);
  ASSERT_NE(e, nullptr);
  const mi_gen_params q = request(4);
  generate_facts(e, q);  // loads, fills the embedding cache: the held state both plan against
  const Facts g = generate_facts(e, q);
  EXPECT_NE(g.plan.find("\"streamed\""), std::string::npos) << g.plan;
  ASSERT_EQ(open_planned(e, &q), MI_OK) << last_error();
  const Facts r = run_facts(e, q);
  EXPECT_EQ(r.decisions, g.decisions);
  EXPECT_EQ(mi_session_close(e), MI_OK);
  mi_engine_free(e);
}

}  // namespace
