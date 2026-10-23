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

bool Guidance::skip_active(int step, int num_steps) const {
  // P:409: Python float division, compared as doubles.
  const double pos = static_cast<double>(step) / static_cast<double>(num_steps);
  return slg_start <= pos && pos < slg_stop;
}

bool Guidance::slg_active(int step, int num_steps) const {
  return slg_layers.any() && slg_scale > 0 && skip_active(step, num_steps);
}

bool Guidance::cfg_active(int step, int num_steps) const {
  return do_cfg() && (cfg_truncation >= 1.0 || static_cast<double>(step) / static_cast<double>(num_steps) < cfg_truncation);
}

std::vector<std::int64_t> denoise_row_lengths(std::int64_t cond_tokens, std::int64_t uncond_tokens, const Guidance& guidance) {
  std::vector<std::int64_t> lengths = {cond_tokens};
  if (guidance.do_cfg()) lengths.push_back(uncond_tokens);
  if (guidance.slg_layers.any() && guidance.slg_scale > 0) lengths.push_back(cond_tokens);
  return lengths;
}

std::size_t denoise_loop_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                               std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler) {
  const std::size_t rows = denoise_row_lengths(cond_tokens, uncond_tokens, guidance).size();
  const std::int64_t numel = moss_dit::kLatentChannels * height * width;
  const std::size_t latent = runtime::align_up(static_cast<std::size_t>(numel) * sizeof(float));
  // One velocity per row, plus the driver's prediction and state.
  return rows * latent +
         SamplerDriver::workspace_bytes(sampler, numel, kMaxDenoiseSteps, static_cast<int>(moss_dit::kLatentChannels));
}

std::size_t denoise_workspace_bytes(std::int64_t height, std::int64_t width, std::int64_t cond_tokens,
                                    std::int64_t uncond_tokens, const Guidance& guidance, SamplerKind sampler,
                                    moss_dit::BlockPath path) {
  return denoise_loop_bytes(height, width, cond_tokens, uncond_tokens, guidance, sampler) +
         moss_dit::MossDit::workspace_bytes(height, width, denoise_row_lengths(cond_tokens, uncond_tokens, guidance), path);
}

int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const SamplerOptions& options,
            const Tensor& latents, cudaStream_t stream, const StepCallback& on_step) {
  using Clock = std::chrono::steady_clock;
  check_latent(latents, "latents");
  const std::int64_t h = latents.dim(1), w = latents.dim(2);
  const int num_steps = schedule.num_steps();
  if (num_steps < 1 || schedule.sigmas.size() != static_cast<std::size_t>(num_steps) + 1) {
    throw std::invalid_argument("pipeline::denoise: malformed schedule");
  }
  if (num_steps > kMaxDenoiseSteps) throw std::invalid_argument("pipeline::denoise: too many steps");
  const bool do_cfg = guidance.do_cfg();
  const bool slg_enabled = guidance.slg_layers.any() && guidance.slg_scale > 0;

  SamplerOptions opts = options;
  opts.flow = schedule.flow;  // the schedule's shift drives the SDE samplers' first-sigma offset
  const bool official_euler = schedule.scheduler == SchedulerKind::kMoss;
  SamplerDriver driver(sampler, schedule.sigmas, {moss_dit::kLatentChannels, h, w}, opts, official_euler);

  runtime::StackScope scope(arena);
  const auto latent_tensor = [&] { return arena.alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w}); };
  const Tensor v_cond = latent_tensor();
  const Tensor v_uncond = do_cfg ? latent_tensor() : Tensor{};
  const Tensor v_slg = slg_enabled ? latent_tensor() : Tensor{};

  const ModelFn model = [&](const Tensor& x, const ModelCall& call, const Tensor& out, cudaStream_t s) {
    const bool skip = guidance.skip_active(call.step, num_steps);
    // CFG truncation: past it the cond row runs alone (cfg_active is do_cfg when it is off).
    const bool cfg = guidance.cfg_active(call.step, num_steps);
    moss_dit::Row rows[3];
    Tensor outs[3];
    int n = 0;
    rows[n] = {cond_text, {}};
    outs[n++] = v_cond;
    if (cfg) {
      rows[n] = {uncond_text, skip ? guidance.uncond_skip : moss_dit::SkipSet{}};
      outs[n++] = v_uncond;
    }
    const bool slg = slg_enabled && skip && cfg == do_cfg;
    if (slg) {
      rows[n] = {cond_text, guidance.slg_layers};
      outs[n++] = v_slg;
    }
    dit.forward(x, call.timestep, std::span<const moss_dit::Row>(rows, n), std::span<const Tensor>(outs, n), s);

    guide_prediction(driver.prediction(), x, call.sigma, v_cond, v_uncond, v_slg, cfg, static_cast<float>(guidance.cfg),
                     slg, static_cast<float>(guidance.slg_scale), out, s);
  };

  auto t0 = Clock::now();
  const StepFn after_step = [&](int finished) {
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto now = Clock::now();
    const StepInfo info{finished, num_steps, std::chrono::duration<double>(now - t0).count()};
    const bool go_on = !on_step || on_step(info);
    t0 = Clock::now();
    return go_on;
  };
  return driver.run(latents, model, arena, stream, after_step);
}

int denoise(moss_dit::MossDit& dit, runtime::DeviceArena& arena, const Tensor& cond_text, const Tensor& uncond_text,
            const Schedule& schedule, const Guidance& guidance, SamplerKind sampler, const Tensor& latents,
            cudaStream_t stream, const StepCallback& on_step) {
  if (sampler_needs_options(sampler)) {
    throw std::invalid_argument(std::string("pipeline::denoise: sampler ") + sampler_name(sampler) +
                                " needs SamplerOptions (seed or data-space parameters)");
  }
  return denoise(dit, arena, cond_text, uncond_text, schedule, guidance, sampler, SamplerOptions{}, latents, stream, on_step);
}

}  // namespace mi::pipeline
