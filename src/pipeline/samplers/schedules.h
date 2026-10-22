// Sigma schedules for flow-matching models: the official MOSS-Image schedule ("moss") and the standard schedulers of
// the k-diffusion family (simple, normal, karras, exponential, beta, sgm_uniform), on a discrete flow model
// sampling with a time shift.
//
// Flow model sampling (rectified flow, x_sigma = (1 - sigma) * x0 + sigma * noise):
//   sigma(t)     = shift * u / (1 + (shift - 1) * u),  u = t / multiplier       (u itself when shift == 1)
//   timestep(s)  = s * multiplier
//   table[k]     = sigma((k + 1) / 1000 * multiplier),  k = 0..999 (ascending; table[999] == 1)
// `multiplier` only scales the timestep axis. MOSS-Image's DiT always receives sigma * 1000 (models that embed
// timestep * 1000 internally use multiplier 1, models that take the 0..1000 timestep use 1000), but the two
// multipliers round the table and the normal/sgm_uniform timestep grid differently in the last bit, so both are
// reproducible.
//
// Every value is computed as the reference does it, op by op in fp32 (torch CPU tensor ops, one rounding each),
// with Python doubles where the reference uses Python floats. Exceptions (≤ 1 ulp): karras' pow and exponential's
// exp are correctly rounded here, while torch uses SLEEF's vectorized (1.0-ulp) versions.
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace mi::pipeline {

inline constexpr int kNumTrainTimesteps = 1000;

// torch.linspace(start, end, steps) on the CPU in float32, bit-exact (torch 2.14 linspace_kernel: an fp32 step
// (end - start) / (steps - 1), the first half as start + step * i and the second half as end - step * (steps - 1 - i),
// each with a single rounding). steps >= 1.
std::vector<float> torch_linspace_f32(float start, float end, std::int64_t steps);

// grid(count, shift) of MOSSImageScheduler.set_timesteps: shift * v / (1 + (shift - 1) * v) over
// v = linspace(1, shift / (1000 + shift - 1), count), each torch op rounded to fp32.
std::vector<float> moss_sigma_grid(std::int64_t count, double shift);

enum class SchedulerKind { kMoss, kSimple, kNormal, kKarras, kExponential, kBeta, kSgmUniform };

struct FlowSampling {
  double shift = 5.0;
  double multiplier = 1.0;
};

struct Schedule {
  std::vector<float> sigmas;     // num_steps() + 1 values, descending, terminal value appended
  // sigmas * 1000 in fp32, one per step: informational (reports, the step count). The sampler computes the model
  // timestep from the sigma it actually runs at (SamplerDriver; the SDE samplers move a first sigma of 1).
  std::vector<float> timesteps;
  // Where the sigmas come from. kMoss also selects the official pipeline's arithmetic for the euler sampler
  // (velocity step, CFG on velocities); every other scheduler runs the samplers on denoised predictions.
  SchedulerKind scheduler = SchedulerKind::kMoss;
  // The flow sampling the schedule was made with; denoise() hands it to the sampler (the SDE samplers' first-sigma
  // offset uses its shift), so the shift is set in one place.
  FlowSampling flow;
  int num_steps() const { return static_cast<int>(timesteps.size()); }
};

// The official pipeline's schedule for `steps` requested steps (>= 1) and `shift` (> 0, finite): the shifted grid
// snapped to the nearest training sigma (ties to the lower one), consecutive duplicates removed, so num_steps() can
// be lower than `steps`. Throws std::invalid_argument on invalid arguments.
Schedule moss_schedule(int steps, double shift);

// table[0..999] as above. Throws std::invalid_argument unless shift > 0 and multiplier > 0, both finite.
std::vector<float> flow_sigma_table(const FlowSampling& fs);
// sigma(t) for an fp32 timestep, one fp32 rounding per op.
float flow_sigma(const FlowSampling& fs, float timestep);
// timestep(sigma) in fp32.
float flow_timestep(const FlowSampling& fs, float sigma);
// sigma at a fraction of the sampling (0 = start, 1 = end), in double: 1 at <= 0, 0 at >= 1.
double flow_percent_to_sigma(const FlowSampling& fs, double percent);

const char* scheduler_name(SchedulerKind kind);
// moss, simple, normal, karras, exponential, beta, sgm_uniform; throws std::invalid_argument otherwise.
SchedulerKind parse_scheduler(std::string_view name);

// The schedule for `steps` steps: descending sigmas ending with the terminal 0 (steps + 1 values for most
// schedulers; beta drops repeated indices and moss snaps and dedups, so they can be shorter; normal returns steps + 1
// values without an appended 0 when its last sigma is already ~0).
//   moss         official MOSS-Image grid snapped to the training sigmas (moss_schedule; ignores multiplier)
//   simple       table[999 - floor(i * 1000 / steps)]
//   sgm_uniform  sigma(linspace(timestep(sigma_max), timestep(sigma_min), steps + 1)[:-1]), then 0
//   normal       sigma(linspace(timestep(sigma_max), timestep(sigma_min), steps)), then 0
//   karras       (max^(1/7) + ramp * (min^(1/7) - max^(1/7)))^7, ramp = linspace(0, 1, steps), then 0
//   exponential  exp(linspace(log sigma_max, log sigma_min, steps)), then 0
//   beta         table[rint(BetaInv(1 - i / steps; 0.6, 0.6) * 999)] for i < steps, repeated indices dropped, then 0
// with sigma_min = table[0], sigma_max = table[999].
// `denoise` in [0, 1]: below 0.9999 (partial denoising) the schedule for floor(steps / denoise) steps is computed and
// its last steps + 1 values kept; 0 gives an empty schedule.
// The table-based schedulers (simple, beta) repeat sigmas when steps exceeds the table's 1000 entries; the multistep
// and SDE samplers reject such schedules (SamplerDriver), euler-type samplers take them as zero-length steps, as the
// reference does.
// Throws std::invalid_argument for steps < 1, denoise outside [0, 1] or not finite, more than 10^6 total steps, or
// bad FlowSampling.
std::vector<float> compute_sigmas(SchedulerKind kind, int steps, const FlowSampling& fs, double denoise = 1.0);

// compute_sigmas() as a Schedule, with timesteps = fp32(sigma * 1000) for every step (moss: moss_schedule()).
Schedule make_schedule(SchedulerKind kind, int steps, const FlowSampling& fs, double denoise = 1.0);

// Inverse of the regularized incomplete beta function I_x(a, b) (the beta distribution's quantile), for
// p in [0, 1], a, b > 0; accurate to a few ulp in double. Exposed for tests.
double beta_quantile(double p, double a, double b);

}  // namespace mi::pipeline
