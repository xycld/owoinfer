#include "pipeline/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace mi::pipeline {
namespace {

void check_latent(const Tensor& t, const char* what) {
  if (t.device != Device::kCUDA || t.dtype != DType::kF32 || !t.is_contiguous() || t.rank != 3 ||
      t.dim(0) != moss_dit::kLatentChannels) {
    throw std::invalid_argument(std::string("pipeline::denoise: ") + what + " must be contiguous CUDA fp32 [128, h, w], got " +
                                t.describe());
  }
}

}  // namespace

std::vector<float> torch_linspace_f32(float start, float end, std::int64_t steps) {
  if (steps < 1) throw std::invalid_argument("torch_linspace_f32: steps must be >= 1");
  std::vector<float> out(static_cast<std::size_t>(steps));
  if (steps == 1) {
    out[0] = start;
    return out;
  }
  // RangeFactoriesKernel.cpp linspace_kernel; GCC contracts `start + step * idx` into an FMA in both its scalar
  // and vectorized paths, so every element is rounded once.
  const float step = (end - start) / static_cast<float>(steps - 1);
  const std::int64_t halfway = steps / 2;
  for (std::int64_t i = 0; i < steps; ++i) {
    out[static_cast<std::size_t>(i)] = i < halfway ? std::fma(step, static_cast<float>(i), start)
                                                   : std::fma(-step, static_cast<float>(steps - 1 - i), end);
  }
  return out;
}

std::vector<float> moss_sigma_grid(std::int64_t count, double shift) {
  // P:276-279. Python scalars enter torch's fp32 ops cast to fp32; each op rounds once.
  const double lower = shift / (kNumTrainTimesteps + shift - 1);
  std::vector<float> v = torch_linspace_f32(1.0f, static_cast<float>(lower), count);
  const float s = static_cast<float>(shift);
  const float s1 = static_cast<float>(shift - 1);
  for (float& x : v) {
    const float num = s * x;
    const float den = 1.0f + s1 * x;
    x = num / den;
  }
  return v;
}

Schedule moss_schedule(int steps, double shift) {
  if (steps < 1 || !std::isfinite(shift) || shift <= 0) {
    throw std::invalid_argument("moss_schedule: steps must be >= 1 and shift finite and positive");
  }
  const std::vector<float> sigmas = moss_sigma_grid(steps, shift);
  std::vector<float> training = moss_sigma_grid(kNumTrainTimesteps, 1.0);  // training_shift = 1
  std::reverse(training.begin(), training.end());                          // ascending
  Schedule out;
  for (const float sigma : sigmas) {
    // torch.searchsorted (left) clamped to the last index, then nearest with ties to the lower index (P:281-283).
    const auto it = std::lower_bound(training.begin(), training.end(), sigma);
    const std::size_t upper = std::min<std::size_t>(static_cast<std::size_t>(it - training.begin()), training.size() - 1);
    const std::size_t lower = upper > 0 ? upper - 1 : 0;
    const float d_lower = std::abs(sigma - training[lower]);
    const float d_upper = std::abs(training[upper] - sigma);
    const float snapped = training[d_lower <= d_upper ? lower : upper];
    if (out.sigmas.empty() || out.sigmas.back() != snapped) out.sigmas.push_back(snapped);  // unique_consecutive
  }
  // FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...) with shift 1 (identity): timesteps = sigmas * 1000
  // in fp32, then the terminal sigma 0.
  for (const float sigma : out.sigmas) out.timesteps.push_back(sigma * static_cast<float>(kNumTrainTimesteps));
  out.sigmas.push_back(0.0f);
  return out;
}

const char* sampler_name(SamplerKind kind) { return kind == SamplerKind::kAb2 ? "ab2" : "euler"; }

SamplerKind parse_sampler(std::string_view name) {
  if (name == "euler") return SamplerKind::kEuler;
  if (name == "ab2") return SamplerKind::kAb2;
  throw std::invalid_argument("unknown sampler '" + std::string(name) + "' (expected euler or ab2)");
}

bool Guidance::skip_active(int step, int num_steps) const {
  // P:409: Python float division, compared as doubles.
  const double pos = static_cast<double>(step) / static_cast<double>(num_steps);
  return slg_start <= pos && pos < slg_stop;
}

bool Guidance::slg_active(int step, int num_steps) const {
  return slg_layers.any() && slg_scale > 0 && skip_active(step, num_steps);
}

std::size_t denoise_workspace_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                                    std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler,
                                    moss_dit::BlockPath path) {
  std::vector<std::int64_t> lengths = {cond_tokens};
  if (guidance.do_cfg()) lengths.push_back(uncond_tokens);
  if (guidance.slg_layers.any() && guidance.slg_scale > 0) lengths.push_back(cond_tokens);
  const std::size_t latent = runtime::align_up(static_cast<std::size_t>(moss_dit::kLatentChannels * height * width) * sizeof(float));
  // One velocity per row, the guided prediction, and AB2's previous prediction.
  const std::size_t buffers = lengths.size() + 1 + (sampler == SamplerKind::kAb2 ? 1 : 0);
  return buffers * latent + moss_dit::MossDit::workspace_bytes(height, width, lengths, path);
}

int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const Tensor& latents,
            cudaStream_t stream, const StepCallback& on_step) {
  using Clock = std::chrono::steady_clock;
  check_latent(latents, "latents");
  const std::int64_t h = latents.dim(1), w = latents.dim(2);
  const int num_steps = schedule.num_steps();
  if (num_steps < 1 || schedule.sigmas.size() != static_cast<std::size_t>(num_steps) + 1) {
    throw std::invalid_argument("pipeline::denoise: malformed schedule");
  }
  const bool do_cfg = guidance.do_cfg();
  const bool slg_enabled = guidance.slg_layers.any() && guidance.slg_scale > 0;

  runtime::StackScope scope(arena);
  const auto latent_tensor = [&] { return arena.alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w}); };
  const Tensor v_cond = latent_tensor();
  const Tensor v_uncond = do_cfg ? latent_tensor() : Tensor{};
  const Tensor v_slg = slg_enabled ? latent_tensor() : Tensor{};
  const Tensor pred = latent_tensor();
  const Tensor v_prev = sampler == SamplerKind::kAb2 ? latent_tensor() : Tensor{};

  double h_prev = 0;
  for (int i = 0; i < num_steps; ++i) {
    const auto t0 = Clock::now();
    const bool skip = guidance.skip_active(i, num_steps);
    moss_dit::Row rows[3];
    Tensor outs[3];
    int n = 0;
    rows[n] = {cond_text, {}};
    outs[n++] = v_cond;
    if (do_cfg) {
      rows[n] = {uncond_text, skip ? guidance.uncond_skip : moss_dit::SkipSet{}};
      outs[n++] = v_uncond;
    }
    const bool slg = slg_enabled && skip;
    if (slg) {
      rows[n] = {cond_text, guidance.slg_layers};
      outs[n++] = v_slg;
    }
    dit.forward(latents, schedule.timesteps[static_cast<std::size_t>(i)], std::span<const moss_dit::Row>(rows, n),
                std::span<const Tensor>(outs, n), stream);

    // P:416-424.
    Tensor v = v_cond;
    if (do_cfg) {
      cfg_combine(v_cond, v_uncond, static_cast<float>(guidance.cfg), pred, stream);
      v = pred;
    }
    if (slg) {
      slg_correct(do_cfg ? pred : v_slg, v_cond, v_slg, static_cast<float>(guidance.slg_scale), pred, stream);
      v = pred;
    }

    // h in double is exact (difference of two fp32 values in [0, 1]); torch rounds it to fp32 as a scalar operand.
    const float s = schedule.sigmas[static_cast<std::size_t>(i)];
    const float s_next = schedule.sigmas[static_cast<std::size_t>(i) + 1];
    const double h_step = static_cast<double>(s_next) - static_cast<double>(s);
    if (sampler == SamplerKind::kEuler || i == 0) {
      euler_update(latents, v, static_cast<float>(h_step), stream);
    } else {
      const double r = h_step / (2.0 * h_prev);
      ab2_update(latents, v, v_prev, static_cast<float>(h_step), static_cast<float>(1.0 + r), static_cast<float>(r), stream);
    }
    if (sampler == SamplerKind::kAb2) {
      MI_CUDA_CHECK(cudaMemcpyAsync(v_prev.data, v.data, v.nbytes(), cudaMemcpyDeviceToDevice, stream));
    }
    h_prev = h_step;

    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (on_step) {
      const StepInfo info{i + 1, num_steps, std::chrono::duration<double>(Clock::now() - t0).count()};
      if (!on_step(info)) return i + 1;
    }
  }
  return num_steps;
}

}  // namespace mi::pipeline
