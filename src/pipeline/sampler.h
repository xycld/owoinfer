// MOSS-Image sampling: the sigma schedule (MOSSImageScheduler.set_timesteps, P:269-287 + diffusers
// FlowMatchEulerDiscreteScheduler.set_timesteps), classifier-free guidance with block skipping (P:405-425) and
// the flow-matching integrators (scheduler.step's Euler, plus the 2nd-order Adams-Bashforth option of
// tools/reference/sampler_study.py).
//
// Every fp32 operation follows the reference's rounding order, one rounding per torch op:
//   CFG     pred = uncond + g * (cond - uncond)                     (P:418)
//   SLG     pred = pred + s * (cond - skipped), pred = skipped if g == 1  (P:419-424)
//   Euler   x = x + h * v,  h = sigma[i+1] - sigma[i] in fp32         (scheduler.step)
//   AB2     x = x + h * ((1 + r) * v - r * v_prev),  r = h / (2 h_prev) in double, scalars cast to fp32
// Latents and predictions stay fp32 throughout (docs/model-notes.md §4, §6).
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
#include "runtime/device_arena.h"

namespace mi::pipeline {

inline constexpr int kNumTrainTimesteps = 1000;

// torch.linspace(start, end, steps) on the CPU in float32, bit-exact (torch 2.14 linspace_kernel: an fp32 step
// (end - start) / (steps - 1), the first half as start + step * i and the second half as end - step * (steps - 1 - i),
// each with a single rounding). steps >= 1.
std::vector<float> torch_linspace_f32(float start, float end, std::int64_t steps);

// grid(count, shift) of MOSSImageScheduler.set_timesteps: shift * v / (1 + (shift - 1) * v) over
// v = linspace(1, shift / (1000 + shift - 1), count), each torch op rounded to fp32.
std::vector<float> moss_sigma_grid(std::int64_t count, double shift);

struct Schedule {
  std::vector<float> sigmas;     // num_steps() + 1 values, descending, terminal 0 appended
  std::vector<float> timesteps;  // sigmas * 1000 in fp32 (what the DiT receives)
  int num_steps() const { return static_cast<int>(timesteps.size()); }
};

// The pipeline's schedule for `steps` requested steps (>= 1) and `shift` (> 0, finite): the shifted grid snapped to
// the nearest training sigma (ties to the lower one), consecutive duplicates removed, so num_steps() can be lower
// than `steps`. Throws std::invalid_argument on invalid arguments.
Schedule moss_schedule(int steps, double shift);

enum class SamplerKind { kEuler, kAb2 };
const char* sampler_name(SamplerKind kind);
// "euler" or "ab2"; throws std::invalid_argument otherwise.
SamplerKind parse_sampler(std::string_view name);

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

// Elementwise fp32 kernels of the loop (exposed for tests). All tensors CUDA fp32 with equal numel, contiguous.
void cfg_combine(const Tensor& cond, const Tensor& uncond, float g, const Tensor& out, cudaStream_t stream);
void slg_correct(const Tensor& pred, const Tensor& cond, const Tensor& skipped, float scale, const Tensor& out,
                 cudaStream_t stream);
void euler_update(const Tensor& x, const Tensor& v, float h, cudaStream_t stream);
void ab2_update(const Tensor& x, const Tensor& v, const Tensor& v_prev, float h, float a, float b, cudaStream_t stream);

// Stack bytes denoise() allocates itself (velocities, guidance buffers) plus the DiT forward workspace for the
// largest row set of the run, for a latent grid of height x width.
std::size_t denoise_workspace_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                                    std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler);

// Runs the sampling loop in place on `latents` (CUDA fp32 [128, h, w], the initial noise on entry).
// cond_text / uncond_text: refined text tokens (MossDit::encode_text output, CUDA bf16 [S, 3072]); uncond_text is
// ignored when guidance.cfg == 1. All work is on `stream`, scratch on the arena stack (released on return).
// Returns the number of steps run (< schedule.num_steps() iff the callback cancelled).
int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const Tensor& latents,
            cudaStream_t stream, const StepCallback& on_step = {});

}  // namespace mi::pipeline
