#include "pipeline/samplers/schedules.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// Bit-exactness needs one rounding per reference op: no contraction of mul+add into FMAs (FMAs only where written
// as std::fma), no fast-math. cmake/modules/samplers.cmake builds this file with -ffp-contract=off -fno-fast-math
// (/fp:strict on MSVC).
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "schedules.cpp must not be built with fast-math"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mi::pipeline {
namespace {

void check(const FlowSampling& fs) {
  if (!std::isfinite(fs.shift) || fs.shift <= 0 || !std::isfinite(fs.multiplier) || fs.multiplier <= 0) {
    throw std::invalid_argument("flow sampling: shift and multiplier must be finite and positive");
  }
}

// Python float scalars enter torch's fp32 tensor ops cast to fp32.
float f32(double d) { return static_cast<float>(d); }

// Regularized incomplete beta function I_x(a, b), written from the continued fraction of DLMF 8.17.22
// (https://dlmf.nist.gov/8.17.E22):
//   I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * 1 / (1 + d_1 / (1 + d_2 / (1 + ...))),
//   d_2m = m (b - m) x / ((a + 2m - 1)(a + 2m)),  d_2m+1 = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1)),
// which converges fast for x < (a + 1) / (a + b + 2); above that the symmetry I_x(a, b) = 1 - I_(1-x)(b, a)
// (DLMF 8.17.4) is used. The fraction is evaluated with the modified Lentz algorithm (W. J. Lentz, Appl. Opt. 15
// (1976) 668; I. J. Thompson and A. R. Barnett, J. Comput. Phys. 64 (1986) 490). long double keeps the bisection in
// beta_quantile() well inside double resolution.

// d_j of DLMF 8.17.22, j >= 1.
long double dlmf_beta_numerator(int j, long double x, long double a, long double b) {
  const long double m = j / 2;
  if (j % 2 == 0) return m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
  return -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
}

// 1 + d_1 / (1 + d_2 / (1 + ...)) by modified Lentz: f_j = f_(j-1) C_j D_j with C_j = 1 + d_j / C_(j-1) and
// D_j = 1 / (1 + d_j D_(j-1)), from f_0 = C_0 = 1 and D_0 = 0; a vanishing C_j or 1 / D_j is replaced by a tiny value.
long double beta_fraction_lentz(long double x, long double a, long double b) {
  const long double tiny = std::numeric_limits<long double>::min();
  const long double tol = 4 * std::numeric_limits<long double>::epsilon();
  long double f = 1, c = 1, d = 0;
  for (int j = 1; j <= 200000; ++j) {
    const long double dj = dlmf_beta_numerator(j, x, a, b);
    long double den = 1 + dj * d;
    if (std::fabs(den) < tiny) den = tiny;
    d = 1 / den;
    c = 1 + dj / c;
    if (std::fabs(c) < tiny) c = tiny;
    const long double ratio = c * d;
    f *= ratio;
    if (std::fabs(ratio - 1) < tol) break;
  }
  return f;
}

// I_x(a, b) by the fraction, for x in (0, 1) below the switch point.
long double incbeta_below_switch(long double x, long double a, long double b) {
  const long double log_beta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  const long double prefactor = std::exp(a * std::log(x) + b * std::log1p(-x) - log_beta);  // x^a (1-x)^b / B(a, b)
  return prefactor / (a * beta_fraction_lentz(x, a, b));
}

// I_x(a, b) for x in [0, 1].
long double regularized_incomplete_beta(long double x, long double a, long double b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (x < (a + 1) / (a + b + 2)) return incbeta_below_switch(x, a, b);
  return 1 - incbeta_below_switch(1 - x, b, a);
}

std::vector<float> with_denoise(SchedulerKind kind, int steps, const FlowSampling& fs, double denoise) {
  // A partial denoise runs the tail of a longer schedule (k-diffusion front ends' convention).
  if (!std::isfinite(denoise) || denoise < 0.0 || denoise > 1.0) {
    throw std::invalid_argument("compute_sigmas: denoise must be in [0, 1]");
  }
  if (denoise > 0.9999) return compute_sigmas(kind, steps, fs, 1.0);
  if (denoise <= 0.0) return {};
  const double total_d = static_cast<double>(steps) / denoise;
  if (total_d > 1e6) throw std::invalid_argument("compute_sigmas: steps / denoise exceeds 10^6 steps");
  const int total = static_cast<int>(total_d);
  std::vector<float> s = compute_sigmas(kind, total, fs, 1.0);
  const std::size_t keep = std::min(s.size(), static_cast<std::size_t>(steps) + 1);
  return {s.end() - static_cast<std::ptrdiff_t>(keep), s.end()};
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
  // Python scalars enter torch's fp32 ops cast to fp32; each op rounds once.
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
    // torch.searchsorted (left) clamped to the last index, then nearest with ties to the lower index.
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
  out.flow = FlowSampling{shift, 1.0};
  return out;
}

float flow_sigma(const FlowSampling& fs, float timestep) {
  const float u = timestep / f32(fs.multiplier);
  if (fs.shift == 1.0) return u;
  const float num = f32(fs.shift) * u;
  const float den = 1.0f + f32(fs.shift - 1.0) * u;
  return num / den;
}

float flow_timestep(const FlowSampling& fs, float sigma) { return sigma * f32(fs.multiplier); }

double flow_percent_to_sigma(const FlowSampling& fs, double percent) {
  if (percent <= 0.0) return 1.0;
  if (percent >= 1.0) return 0.0;
  const double t = 1.0 - percent;
  if (fs.shift == 1.0) return t;
  return fs.shift * t / (1 + (fs.shift - 1) * t);
}

std::vector<float> flow_sigma_table(const FlowSampling& fs) {
  check(fs);
  // sigma((arange(1, 1001) / 1000) * multiplier): an int64 tensor divided in fp32.
  std::vector<float> table(kNumTrainTimesteps);
  for (int k = 0; k < kNumTrainTimesteps; ++k) {
    const float t = static_cast<float>(k + 1) / static_cast<float>(kNumTrainTimesteps);
    table[static_cast<std::size_t>(k)] = flow_sigma(fs, t * f32(fs.multiplier));
  }
  return table;
}

const char* scheduler_name(SchedulerKind kind) {
  switch (kind) {
    case SchedulerKind::kMoss: return "moss";
    case SchedulerKind::kSimple: return "simple";
    case SchedulerKind::kNormal: return "normal";
    case SchedulerKind::kKarras: return "karras";
    case SchedulerKind::kExponential: return "exponential";
    case SchedulerKind::kBeta: return "beta";
    case SchedulerKind::kSgmUniform: return "sgm_uniform";
  }
  return "?";
}

SchedulerKind parse_scheduler(std::string_view name) {
  for (const SchedulerKind k : {SchedulerKind::kMoss, SchedulerKind::kSimple, SchedulerKind::kNormal, SchedulerKind::kKarras,
                                SchedulerKind::kExponential, SchedulerKind::kBeta, SchedulerKind::kSgmUniform}) {
    if (name == scheduler_name(k)) return k;
  }
  throw std::invalid_argument("unknown scheduler '" + std::string(name) +
                              "' (expected moss, simple, normal, karras, exponential, beta or sgm_uniform)");
}

double beta_quantile(double p, double a, double b) {
  if (!(p >= 0.0 && p <= 1.0) || !(a > 0) || !(b > 0)) throw std::invalid_argument("beta_quantile: bad arguments");
  if (p == 0.0) return 0.0;
  if (p == 1.0) return 1.0;
  // I_x(a, b) increases with x: bisect [0, 1] until the bracket no longer splits in long double.
  const long double target = p;
  long double lo = 0, hi = 1;
  for (int iter = 0; iter < 200; ++iter) {
    const long double mid = (lo + hi) / 2;
    if (!(lo < mid && mid < hi)) break;
    (regularized_incomplete_beta(mid, a, b) < target ? lo : hi) = mid;
  }
  return static_cast<double>((lo + hi) / 2);
}

std::vector<float> compute_sigmas(SchedulerKind kind, int steps, const FlowSampling& fs, double denoise) {
  if (steps < 1) throw std::invalid_argument("compute_sigmas: steps must be >= 1");
  check(fs);
  if (denoise != 1.0) return with_denoise(kind, steps, fs, denoise);
  if (kind == SchedulerKind::kMoss) return moss_schedule(steps, fs.shift).sigmas;

  const std::vector<float> table = flow_sigma_table(fs);
  const float sigma_min = table.front(), sigma_max = table.back();
  std::vector<float> s;
  switch (kind) {
    case SchedulerKind::kSimple: {
      const double ss = static_cast<double>(table.size()) / steps;
      for (int i = 0; i < steps; ++i) {
        const auto idx = static_cast<std::size_t>(static_cast<double>(i) * ss);
        s.push_back(table[table.size() - 1 - idx]);
      }
      s.push_back(0.0f);
      break;
    }
    case SchedulerKind::kSgmUniform:
    case SchedulerKind::kNormal: {
      const float start = flow_timestep(fs, sigma_max), end = flow_timestep(fs, sigma_min);
      bool append_zero = true;
      std::vector<float> ts;
      if (kind == SchedulerKind::kSgmUniform) {
        ts = torch_linspace_f32(start, end, steps + 1);
        ts.pop_back();
      } else {
        int n = steps;
        if (std::fabs(static_cast<double>(flow_sigma(fs, end))) <= 1e-5) {  // math.isclose(.., 0, abs_tol=1e-5)
          ++n;
          append_zero = false;
        }
        ts = torch_linspace_f32(start, end, n);
      }
      for (const float t : ts) s.push_back(flow_sigma(fs, t));
      if (append_zero) s.push_back(0.0f);
      break;
    }
    case SchedulerKind::kKarras: {
      constexpr double rho = 7.0;
      const double min_inv_rho = std::pow(static_cast<double>(sigma_min), 1.0 / rho);
      const double max_inv_rho = std::pow(static_cast<double>(sigma_max), 1.0 / rho);
      const float span = f32(min_inv_rho - max_inv_rho), base = f32(max_inv_rho);
      for (const float r : torch_linspace_f32(0.0f, 1.0f, steps)) {
        const float v = base + r * span;
        s.push_back(static_cast<float>(std::pow(static_cast<double>(v), rho)));
      }
      s.push_back(0.0f);
      break;
    }
    case SchedulerKind::kExponential: {
      const float a = f32(std::log(static_cast<double>(sigma_max))), b = f32(std::log(static_cast<double>(sigma_min)));
      for (const float v : torch_linspace_f32(a, b, steps)) s.push_back(static_cast<float>(std::exp(static_cast<double>(v))));
      s.push_back(0.0f);
      break;
    }
    case SchedulerKind::kBeta: {
      // numpy: ts = 1 - linspace(0, 1, steps, endpoint=False) = 1 - i * (1 / steps); rint is half-to-even.
      constexpr double alpha = 0.6, beta = 0.6;
      const double step = 1.0 / steps;
      const auto total = static_cast<double>(table.size() - 1);
      double last = -1;
      for (int i = 0; i < steps; ++i) {
        const double t = std::nearbyint(beta_quantile(1.0 - static_cast<double>(i) * step, alpha, beta) * total);
        if (t != last) s.push_back(table[static_cast<std::size_t>(t)]);
        last = t;
      }
      s.push_back(0.0f);
      break;
    }
    case SchedulerKind::kMoss: break;
  }
  return s;
}

Schedule make_schedule(SchedulerKind kind, int steps, const FlowSampling& fs, double denoise) {
  if (kind == SchedulerKind::kMoss && denoise == 1.0) {
    Schedule m = moss_schedule(steps, fs.shift);
    m.flow = fs;
    return m;
  }
  Schedule out;
  out.scheduler = kind;
  out.flow = fs;
  out.sigmas = compute_sigmas(kind, steps, fs, denoise);
  for (std::size_t i = 0; i + 1 < out.sigmas.size(); ++i) {
    out.timesteps.push_back(out.sigmas[i] * static_cast<float>(kNumTrainTimesteps));
  }
  return out;
}

}  // namespace mi::pipeline
