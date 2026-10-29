#include "pipeline/run_schedule.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <system_error>

#include "pipeline/pipeline_image.h"

namespace mi::pipeline {
namespace {

bool same_bits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

// Shortest round-trip text, independent of the locale.
template <typename T>
std::string num(T v) {
  char buf[32];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);
  return r.ec == std::errc{} ? std::string(buf, r.ptr) : std::string("?");
}

// printf's %.9g, independent of the locale (the note's number format).
std::string g9(double v) {
  char buf[48];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general, 9);
  return r.ec == std::errc{} ? std::string(buf, r.ptr) : std::string("?");
}

std::string describe_sigma(float v) { return "sigma " + num(v); }

}  // namespace

std::vector<float> engine_sigmas(SchedulerKind kind, int steps, const FlowSampling& fs, double strength) {
  if (!std::isfinite(strength) || strength < 0.0 || strength > 1.0) {
    throw std::invalid_argument("engine_sigmas: denoise strength must be in [0, 1], got " + num(strength));
  }
  Schedule pass = strength_pass(make_schedule(kind, steps, fs), strength);
  if (pass.num_steps() == 0) return {};  // no step left (mi_generate runs no DiT)
  return std::move(pass.sigmas);
}

Schedule strength_pass(Schedule full, double strength, int* start) {
  const int skip = strength_start_step(full.num_steps(), strength);
  full.sigmas.erase(full.sigmas.begin(), full.sigmas.begin() + skip);
  full.timesteps.erase(full.timesteps.begin(), full.timesteps.begin() + skip);
  if (start != nullptr) *start = skip;
  return full;
}

std::size_t first_schedule_difference(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (!same_bits(a[i], b[i])) return i;
  }
  return a.size() == b.size() ? kSchedulesEqual : n;
}

std::string schedule_note(std::span<const float> host, std::span<const float> engine, SchedulerKind kind, int steps,
                          const FlowSampling& fs, double strength) {
  const std::size_t k = first_schedule_difference(host, engine);
  if (k == kSchedulesEqual) return {};
  std::string s = std::string("sigmas differ from the engine's ") + scheduler_name(kind) + " schedule (" +
                  std::to_string(steps) + " steps, shift " + g9(fs.shift) + ", multiplier " + g9(fs.multiplier) +
                  ", strength " + g9(strength) + ")";
  if (host.size() != engine.size()) {
    s += ": " + std::to_string(host.size()) + " host values vs " + std::to_string(engine.size()) +
         " engine values; first difference";
  }
  const auto value = [&](std::span<const float> v) { return k < v.size() ? g9(v[k]) : std::string("none"); };
  s += " at index " + std::to_string(k) + ": " + value(host) + " (host) vs " + value(engine) + " (engine)";
  return s;
}

RunSchedule::RunSchedule(std::vector<float> sigmas, bool exact_sigma) : sigmas_(std::move(sigmas)), exact_sigma_(exact_sigma) {
  if (sigmas_.size() < 2 || sigmas_.size() > kMaxRunSigmas) {
    throw std::invalid_argument("run schedule: " + std::to_string(sigmas_.size()) + " sigmas, expected 2.." +
                                std::to_string(kMaxRunSigmas) + " (steps + 1)");
  }
  for (std::size_t i = 0; i < sigmas_.size(); ++i) {
    if (!std::isfinite(sigmas_[i])) {
      throw std::invalid_argument("run schedule: sigmas[" + std::to_string(i) + "] is not finite");
    }
    if (i > 0 && sigmas_[i] > sigmas_[i - 1]) {
      throw std::invalid_argument("run schedule: sigmas must be non-increasing, but sigmas[" + std::to_string(i) +
                                  "] = " + num(sigmas_[i]) + " > sigmas[" + std::to_string(i - 1) +
                                  "] = " + num(sigmas_[i - 1]));
    }
  }
}

RunSchedule::Position RunSchedule::locate(float sigma, int step) const {
  if (!std::isfinite(sigma)) throw std::invalid_argument("denoise step: sigma is not finite");
  const int n = steps();
  const auto at = [&](int i) { return sigmas_[static_cast<std::size_t>(i)]; };
  Position p;
  if (step >= 0) {
    if (step >= n) {
      throw std::invalid_argument("denoise step: step " + std::to_string(step) + " is outside the run's " +
                                  std::to_string(n) + " steps");
    }
    p.step = step;
    p.on_schedule = same_bits(sigma, at(step)) || same_bits(sigma, at(step + 1));
    if (!p.on_schedule && exact_sigma_) {
      throw std::invalid_argument("denoise step: " + describe_sigma(sigma) + " at step " + std::to_string(step) +
                                  " is neither sigmas[" + std::to_string(step) + "] = " + num(at(step)) + " nor sigmas[" +
                                  std::to_string(step + 1) + "] = " + num(at(step + 1)) +
                                  " (the run takes exact sigmas only)");
    }
    return p;
  }
  if (step != -1) throw std::invalid_argument("denoise step: step must be -1 (from sigma) or >= 0, got " + std::to_string(step));
  for (int i = cursor_; i < n; ++i) {
    if (same_bits(at(i), sigma)) return {i, true};
  }
  for (int i = 0; i < std::min(cursor_, n); ++i) {
    if (same_bits(at(i), sigma)) return {i, true};
  }
  if (exact_sigma_) {
    throw std::invalid_argument("denoise step: " + describe_sigma(sigma) +
                                " is not an entry of the run's schedule (the run takes exact sigmas only)");
  }
  if (sigma >= at(0)) return {0, false};
  if (sigma <= at(n)) return {n - 1, false};
  // The first index j with s[j] < sigma (j in 1..n); sigma lies in (s[j], s[j - 1]), both strict: no exact hit.
  const auto it = std::partition_point(sigmas_.begin(), sigmas_.end(), [&](float v) { return v >= sigma; });
  return {static_cast<int>(it - sigmas_.begin()) - 1, false};
}

void RunSchedule::commit(const Position& p) {
  cursor_ = p.step + 1;
  ++calls_;
  if (!p.on_schedule) ++off_schedule_calls_;
}

}  // namespace mi::pipeline
