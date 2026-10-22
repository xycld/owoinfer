// MOSS-Image sampling: classifier-free guidance with block skipping (P:405-425) around the sampler driver
// (pipeline/samplers/sampler_driver.h), on the schedules of pipeline/samplers/schedules.h.
//
// The "moss" schedule with the euler (or ab2) sampler is the official pipeline, rounded as the reference rounds each
// torch op (velocities, fp32):
//   CFG     pred = uncond + g * (cond - uncond)                     (P:418)
//   SLG     pred = pred + s * (cond - skipped), pred = skipped if g == 1  (P:419-424)
//   Euler   x = x + h * v,  h = sigma[i+1] - sigma[i] in fp32         (scheduler.step)
//   AB2     x = x + h * ((1 + r) * v - r * v_prev),  r = h / (2 h_prev) in double, scalars cast to fp32
// Every other (sampler, scheduler) pair runs the k-diffusion formulation on denoised predictions, each row's
// velocity turned into D = x - v * sigma and combined as
//   CFG     D = D_uncond + (D_cond - D_uncond) * g
//   SLG     D = D + (D_cond - D_skipped) * s
// Latents and predictions stay fp32 throughout, as in the reference.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "core/tensor.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/samplers/sampler_driver.h"
#include "pipeline/samplers/schedules.h"
#include "runtime/device_arena.h"

namespace mi::pipeline {

// Upper bound on schedule length that denoise_workspace_bytes() budgets the driver's coefficients for.
inline constexpr int kMaxDenoiseSteps = 10000;

struct Guidance {
  double cfg = 3.0;                // guidance_scale; == 1 disables the negative branch
  moss_dit::SkipSet uncond_skip;   // uncond_skip_layers (pipeline default {22})
  moss_dit::SkipSet slg_layers;    // SLG third row (cond prompt, these blocks skipped)
  double slg_scale = 0.0;
  double slg_start = 0.0;          // both skips active for slg_start <= i / num_steps < slg_stop
  double slg_stop = 1.0;

  bool do_cfg() const { return cfg != 1.0; }
  bool skip_active(int step, int num_steps) const;
  bool slg_active(int step, int num_steps) const;
};

// Progress of the sampling loop, after each finished step.
struct StepInfo {
  int step = 0;  // 1-based number of steps finished
  int num_steps = 0;
  double step_seconds = 0;  // wall time of this step (synchronized)
};
// Called after every step; return false to stop (cancel) before the next one.
using StepCallback = std::function<bool(const StepInfo&)>;

// The DiT rows of the loop's largest step, in row order: cond, uncond under CFG, the SLG row when SLG is enabled.
std::vector<std::int64_t> denoise_row_lengths(std::int64_t cond_tokens, std::int64_t uncond_tokens, const Guidance& guidance);
// Stack bytes denoise() allocates itself, without the DiT forward: velocities, guidance buffers and the sampler
// driver's state for up to kMaxDenoiseSteps steps. A caller adds MossDit::workspace_bytes for denoise_row_lengths().
std::size_t denoise_loop_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                               std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler);
// Stack bytes denoise() allocates itself (velocities, guidance buffers, the sampler driver's state for up to
// kMaxDenoiseSteps steps) plus the DiT forward workspace for the largest row set of the run, for a latent grid of
// height x width and the DiT's block path.
std::size_t denoise_workspace_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                                    std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler,
                                    moss_dit::BlockPath path = moss_dit::BlockPath::kFused);

// Runs the sampling loop in place on `latents` (CUDA fp32 [128, h, w], the initial noise scaled to the first sigma
// on entry (flow_noise_scaling; the pure noise when it is 1), the final latents on return; for the dataspace sampler
// the final x0 estimate). cond_text / uncond_text: refined text tokens (MossDit::encode_text output, CUDA bf16
// [S, 3072]); uncond_text is ignored when guidance.cfg == 1. options: noise seed and source for the stochastic
// samplers, eta / s_noise and the data-space parameters; options.flow is replaced by schedule.flow (the SDE samplers'
// first-sigma offset uses the schedule's shift), and euler takes the official velocity step iff schedule.scheduler is
// kMoss. Guidance windows use the step index (i / num_steps), heun's second call included in its step. Each model
// call's prediction is guide_prediction() of the DiT rows. All work is on `stream`, scratch on the arena stack
// (released on return). Returns the number of steps run (< schedule.num_steps() iff the callback cancelled).
// Throws std::invalid_argument.
int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const SamplerOptions& options,
            const Tensor& latents, cudaStream_t stream, const StepCallback& on_step = {});

// The deterministic samplers with default options; throws std::invalid_argument for a stochastic or the dataspace
// sampler (they need a seed or parameters).
int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const Tensor& latents,
            cudaStream_t stream, const StepCallback& on_step = {});

}  // namespace mi::pipeline
