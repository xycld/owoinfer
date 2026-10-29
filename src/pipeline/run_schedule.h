// A component run's schedule: where a model call sits in it (its step position), and how a host's sigmas compare with
// the engine's own schedule. Pure host code, no GPU.
//
// The position of a call drives the step-indexed features (the guidance windows, CFG truncation, the control windows
// and the step cache's invalidation when one flips), exactly as denoise() indexes them by ModelCall.step. A call's
// sigma is the flow sigma of its latents (x = (1 - sigma) x0 + sigma noise); the schedule is a position hint, never a
// limit on the calls:
//   explicit   step >= 0 is used as given (a corrector inside step i, heun's call at s[i+1], passes i)
//   exact      step = -1: the first i in [cursor, n) with s[i] == sigma (fp32 bit equality), else the first in
//              [0, cursor); the cursor is the last step used + 1, so repeated sigmas resolve to consecutive steps
//   interval   no exact hit: the i with s[i] > sigma > s[i+1]; sigma >= s[0] gives 0, sigma <= s[n] gives n - 1.
//              The call is off schedule; refused when the run asked for exact sigmas only.
// with n = steps and s[0..n] the run's sigmas (the terminal value last).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pipeline/samplers/schedules.h"

namespace mi::pipeline {

// Longest schedule a run takes: kMaxDenoiseSteps (pipeline/sampler.h) steps plus the terminal value.
inline constexpr std::size_t kMaxRunSigmas = 10001;

// mi_generate's sigmas for one pass: make_schedule(kind, steps, fs) cropped to the tail a denoise strength leaves
// (strength_start_step, the img2img rule of Pipeline::run); strength 1 is the whole schedule, 0 leaves only the
// terminal value. Throws std::invalid_argument as make_schedule, or for a strength outside [0, 1].
std::vector<float> engine_sigmas(SchedulerKind kind, int steps, const FlowSampling& fs, double strength);

// Index of the first position where a and b differ bit for bit (a shorter array differs at its end), or npos when they
// are identical.
inline constexpr std::size_t kSchedulesEqual = static_cast<std::size_t>(-1);
std::size_t first_schedule_difference(std::span<const float> a, std::span<const float> b);

// The component note's schedule sentence: how a host's sigmas differ from the engine's schedule for the named
// scheduler fields (the first differing index with both values, %.9g; with the two lengths when they differ, a missing
// value printed as "none"); "" when they are identical.
std::string schedule_note(std::span<const float> host, std::span<const float> engine, SchedulerKind kind, int steps,
                          const FlowSampling& fs, double strength);

class RunSchedule {
 public:
  struct Position {
    int step = 0;              // the step position used
    bool on_schedule = false;  // sigma is s[step], or s[step + 1] for an explicit step (a corrector)
  };

  // sigmas: 2..kMaxRunSigmas values, finite, non-increasing (copied). exact_sigma: refuse off-schedule calls.
  // Throws std::invalid_argument naming the first offending value.
  RunSchedule(std::vector<float> sigmas, bool exact_sigma);

  // The position of a call at `sigma` with `step` (-1: from sigma). Changes nothing: a call that fails or is
  // cancelled leaves the cursor where it was. Throws std::invalid_argument for a non-finite sigma, a step outside
  // [-1, steps()), or an off-schedule sigma when the run takes exact sigmas only.
  Position locate(float sigma, int step) const;
  // Records a completed call at `p` (from locate): moves the cursor past it and counts it.
  void commit(const Position& p);

  int steps() const { return static_cast<int>(sigmas_.size()) - 1; }
  const std::vector<float>& sigmas() const { return sigmas_; }
  bool exact_sigma() const { return exact_sigma_; }
  int cursor() const { return cursor_; }
  std::int64_t calls() const { return calls_; }
  std::int64_t off_schedule_calls() const { return off_schedule_calls_; }

 private:
  std::vector<float> sigmas_;
  bool exact_sigma_ = false;
  int cursor_ = 0;
  std::int64_t calls_ = 0;
  std::int64_t off_schedule_calls_ = 0;
};

}  // namespace mi::pipeline
