// Sampler driver: runs a sampling algorithm over a sigma schedule with a model callback, on the GPU, exactly and
// deterministically. It is model independent: the callback receives the current latents x and the step's noise level
// and returns the model's prediction, in the space the algorithm asks for (prediction()):
//   kDenoised  x0 estimate D(x, sigma). For a flow model with velocity output v: D = x - v * sigma, in fp32.
//   kVelocity  v itself (flow-matching velocity, x_sigma = (1 - sigma) x0 + sigma noise, v = noise - x0).
//
// Algorithms (x: latents, s = sigma_i, n = sigma_{i+1}, D: denoised, all arithmetic fp32 with the reference's
// rounding order; scalar coefficients are evaluated on the GPU with the CUDA math library, as torch does for its
// 0-dim tensors, except expm1, which is correctly rounded where torch's is 1 ulp off on some inputs, so dpmpp_2m,
// dpmpp_2m_sde and res_multistep match the reference within 4.5e-7 of the latents' largest magnitude instead of
// bit for bit):
//   euler            d = (x - D) / s,  x += d * (n - s)                              (Karras et al. 2022, Alg. 2)
//   euler (velocity) x += (n - s) * v      the official MOSS-Image pipeline's step (constructor's official_euler)
//   ab2              2nd-order Adams-Bashforth on v (Euler first): x += h ((1 + r) v - r v_prev), r = h / (2 h_prev)
//   euler_ancestral  rectified-flow ancestral step: sigma_down = n (1 + (n / s - 1) eta),
//                    x = (sd / s) x + (1 - sd / s) D, then x = (a_n / a_d) x + noise * sqrt(n^2 - sd^2 a_n^2 / a_d^2)
//                    with a_n = 1 - n, a_d = 1 - sd;  x = D at n == 0
//   heun             Euler predictor to n, corrector with the average slope (2 model calls; Euler when n == 0)
//   dpmpp_2m         DPM-Solver++(2M) in t = -log sigma (Lu et al. 2022)
//   dpmpp_2m_sde     DPM-Solver++(2M) SDE, midpoint, in lambda = log((1 - sigma) / sigma) (flow half-log-SNR)
//   er_sde           ER-SDE-Solver-3 (Cui et al. 2023), noise scaler x (e^{x^0.3} + 10), 200-point quadrature
//   res_multistep    2nd-order multistep exponential integrator (Zhang et al. 2023, RES), deterministic (eta 0)
//   dataspace        data-space few-step sampler (velocity): x0 = 0, then per step x = z + Q x0 (Q a channel mix,
//                    z the initial noise), v = model(x, step code), x0 = z - v; returns x0
// The SDE samplers (dpmpp_2m_sde, er_sde) move a first sigma >= 1 to flow_percent_to_sigma(1e-4) before sampling,
// as the reference does to keep the log-SNR finite; the initial latents are still the pure noise.
//
// Repeated positive sigmas (zero-length steps, e.g. a table-based schedule with more than 1000 steps) are rejected for
// ab2, dpmpp_2m, dpmpp_2m_sde, er_sde and res_multistep, whose step-size ratios would divide by zero; the euler-type
// samplers (euler, euler_ancestral, heun) take them as no-op steps, as the reference does.
//
// Noise (euler_ancestral, dpmpp_2m_sde, er_sde) is drawn once per step that adds it (n > 0), from
//   kPhilox        (default) torch's CUDA generator algorithm seeded with `seed` on the GPU-independent grid
//                  (samplers/torch_cuda_randn.h): the same noise on every GPU
//   kPhiloxDevice  the same algorithm on this GPU's grid: equals torch.randn(..., device="cuda") here, i.e. the
//                  k-diffusion reference run on this GPU (and on GPUs with the same SM count and threads per SM)
//   kBrownianTree  the k-diffusion BrownianTreeNoiseSampler on the CPU (samplers/brownian_tree.h), which the
//                  reference's dpmpp_2m_sde uses; it holds up to 45 latent-sized fp32 increments in host memory
//                  (~94 MB at 1024x1024, ~380 MB at 2048x2048) and runs on the CPU.
// Exact parity with the reference run on a GPU therefore needs kPhiloxDevice for euler_ancestral and er_sde and
// kBrownianTree for dpmpp_2m_sde.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/tensor.h"
#include "pipeline/samplers/schedules.h"
#include "runtime/device_arena.h"

namespace mi::pipeline {

enum class SamplerKind {
  kEuler,
  kAb2,
  kEulerAncestral,
  kHeun,
  kDpmpp2m,
  kDpmpp2mSde,
  kErSde,
  kResMultistep,
  kDataSpace,
};
const char* sampler_name(SamplerKind kind);
// euler, ab2, euler_ancestral, heun, dpmpp_2m, dpmpp_2m_sde, er_sde, res_multistep, dataspace;
// throws std::invalid_argument otherwise.
SamplerKind parse_sampler(std::string_view name);
// Draws noise during sampling (needs a seed).
bool sampler_is_stochastic(SamplerKind kind);
// Whether the sampler can run without SamplerOptions beyond the defaults: false for the stochastic samplers (they
// need the generation's seed) and dataspace (it needs DataSpaceParams). Front ends validate a request with
// sampler_supported_by(kind, have_options) before any heavy work.
bool sampler_needs_options(SamplerKind kind);
inline bool sampler_supported_by(SamplerKind kind, bool has_sampler_options) {
  return has_sampler_options || !sampler_needs_options(kind);
}
// Upper bound of model evaluations per step (heun: 2, all others 1), for planning and time estimates.
int model_calls_per_step(SamplerKind kind);

enum class Prediction { kVelocity, kDenoised };

enum class NoiseSource { kPhilox, kPhiloxDevice, kBrownianTree };
const char* noise_source_name(NoiseSource source);
// philox, philox_device, brownian_tree; throws std::invalid_argument otherwise.
NoiseSource parse_noise_source(std::string_view name);

// Data-space sampler parameters: a [channels, channels] channel mix and one model timestep (step code) per step.
struct DataSpaceParams {
  int channels = 0;
  std::vector<float> q;           // row-major [out, in]
  std::vector<float> step_codes;  // timesteps, e.g. 1000, 968, ...
};
// A malformed or invalid dataspace file (load_dataspace).
class DataspaceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
// Reads `latent_mix.Q` (F32 [C, C]) and `step_codes` (F32/F64/I64 [N]) from a dataspace.safetensors file.
// Throws DataspaceError for a malformed or invalid file (std::runtime_error: unreadable).
DataSpaceParams load_dataspace(const std::string& path);
// Sigmas for the first `steps` codes: code / 1000, then 0. Throws std::invalid_argument if steps is out of range.
std::vector<float> dataspace_sigmas(const DataSpaceParams& params, int steps);

struct SamplerOptions {
  std::uint64_t seed = 0;  // noise seed (stochastic samplers)
  NoiseSource noise = NoiseSource::kPhilox;
  double eta = 1.0;      // euler_ancestral, dpmpp_2m_sde
  double s_noise = 1.0;  // noise multiplier (stochastic samplers)
  FlowSampling flow;                      // for the SDE samplers' first-sigma offset
  const DataSpaceParams* dataspace = nullptr;  // required for kDataSpace
};

struct ModelCall {
  int step = 0;          // 0-based sampling step
  int call = 0;          // 0-based model call (heun makes 2 per step)
  float sigma = 0;       // noise level of x
  float timestep = 0;    // model timestep: sigma * 1000 in fp32, or the data-space step code
};

// Writes the prediction for (x, call) into `out` (both contiguous CUDA fp32 of the latent's shape) on `stream`.
// x must not be modified.
using ModelFn = std::function<void(const Tensor& x, const ModelCall& call, const Tensor& out, cudaStream_t stream)>;
// After each finished step (1-based count); return false to stop before the next one.
using StepFn = std::function<bool(int finished_steps)>;

class SamplerDriver {
 public:
  // sigmas: non-increasing schedule with its terminal value (>= 2 values). latent_shape: the latents' shape (for the
  // data-space sampler [channels, h, w]). official_euler: euler runs as the official MOSS-Image pipeline's velocity
  // step (prediction() == kVelocity) instead of the k-diffusion step on denoised predictions; denoise() derives it
  // from the schedule (the "moss" scheduler). No GPU work. Throws std::invalid_argument.
  SamplerDriver(SamplerKind kind, std::vector<float> sigmas, std::vector<std::int64_t> latent_shape,
                const SamplerOptions& options, bool official_euler = false);
  ~SamplerDriver();
  SamplerDriver(SamplerDriver&&) noexcept;
  SamplerDriver& operator=(SamplerDriver&&) noexcept;

  SamplerKind kind() const;
  Prediction prediction() const;
  int num_steps() const;
  // Model calls of a full run (heun: 2 per step except into sigma 0).
  int num_model_calls() const;
  // The sigmas the algorithm runs on (the SDE samplers' first-sigma offset applied).
  const std::vector<float>& sigmas() const;

  // Device bytes run() takes from the arena stack for `steps` steps (channels: the data-space Q size).
  static std::size_t workspace_bytes(SamplerKind kind, std::int64_t numel, int steps, int channels = 0);

  // Runs the sampling in place on x (CUDA fp32, contiguous, the latent shape; the initial noise scaled to sigmas()[0]
  // on entry, the final latents on return), once per driver. Evaluates the per-step coefficients on the GPU first
  // (one synchronization). Scratch comes from the arena stack (released on return). Returns the number of steps
  // finished (fewer iff after_step stopped the run).
  int run(const Tensor& x, const ModelFn& model, runtime::DeviceArena& arena, cudaStream_t stream,
          const StepFn& after_step = {});

  // Per-step scalar coefficients as evaluated on the GPU by run() (layout per sampler; for tests).
  const std::vector<float>& coefficients() const;
  static int coefficients_per_step(SamplerKind kind);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Flow-matching noise scaling of the initial latents: x = sigma * noise + (1 - sigma) * latent (latent absent: 0),
// in place on `noise`, fp32. Identity for sigma == 1 and an empty latent.
void flow_noise_scaling(const Tensor& noise, const Tensor& latent, float sigma, cudaStream_t stream);

// Elementwise prediction kernels for flow models (contiguous CUDA fp32 of equal numel), the reference's rounding:
//   velocity_to_denoised  out = x - v * sigma
//   cfg_denoised          out = (x - vu s) + ((x - vc s) - (x - vu s)) * g
//   slg_denoised          out = out + ((x - vc s) - (x - vs s)) * scale
void velocity_to_denoised(const Tensor& x, const Tensor& v, float sigma, const Tensor& out, cudaStream_t stream);
void cfg_denoised(const Tensor& x, const Tensor& v_cond, const Tensor& v_uncond, float sigma, float g, const Tensor& out,
                  cudaStream_t stream);
void slg_denoised(const Tensor& x, const Tensor& v_cond, const Tensor& v_skip, float sigma, float scale,
                  const Tensor& out, cudaStream_t stream);
// Guidance of the official pipeline on velocities (P:416-424):
//   cfg_combine  out = uncond + g * (cond - uncond)
//   slg_correct  out = pred + s * (cond - skipped)
void cfg_combine(const Tensor& cond, const Tensor& uncond, float g, const Tensor& out, cudaStream_t stream);
void slg_correct(const Tensor& pred, const Tensor& cond, const Tensor& skipped, float scale, const Tensor& out,
                 cudaStream_t stream);

// The guided prediction of one model call from the DiT rows' velocities, in the sampler's space (what denoise()
// hands the driver). v_uncond is used iff cfg, v_skip iff slg.
//   kVelocity  CFG and SLG on velocities as the official pipeline (with cfg off, SLG starts from v_skip);
//              neither: out = v_cond (a copy)
//   kDenoised  each row as D = x - v * sigma, CFG then SLG on denoised predictions; neither: out = x - v_cond sigma
void guide_prediction(Prediction space, const Tensor& x, float sigma, const Tensor& v_cond, const Tensor& v_uncond,
                      const Tensor& v_skip, bool cfg, float g, bool slg, float slg_scale, const Tensor& out,
                      cudaStream_t stream);

// Velocity steps of the official pipeline: x += h * v, and Adams-Bashforth x += h * (a * v - b * v_prev).
void euler_update(const Tensor& x, const Tensor& v, float h, cudaStream_t stream);
void ab2_update(const Tensor& x, const Tensor& v, const Tensor& v_prev, float h, float a, float b, cudaStream_t stream);

}  // namespace mi::pipeline
