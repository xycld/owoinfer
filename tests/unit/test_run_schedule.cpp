// pipeline/run_schedule.h: a component run's step positions (explicit steps, the exact lookup with its cursor, off-
// schedule intervals and the exact-sigma opt-in), mi_generate's pass schedule and the schedule comparison note.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pipeline/pipeline_image.h"
#include "pipeline/run_schedule.h"
#include "pipeline/samplers/schedules.h"

namespace {

using mi::pipeline::FlowSampling;
using mi::pipeline::RunSchedule;
using mi::pipeline::SchedulerKind;

const std::vector<float> kSix = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f, 0.0f};  // 5 steps

float up(float v) { return std::nextafter(v, 2.0f); }

std::string g9(double v) {
  char b[48];
  std::snprintf(b, sizeof(b), "%.9g", v);
  return b;
}

// Locates and commits one call, as a completed mi_denoise_step does.
RunSchedule::Position call(RunSchedule& s, float sigma, int step = -1) {
  const RunSchedule::Position p = s.locate(sigma, step);
  s.commit(p);
  return p;
}

// ---- mi_generate's pass schedule ----------------------------------------------------------------------------------

TEST(RunSchedule, EngineSigmasAtStrengthOneAreMakeSchedule) {
  const FlowSampling fs{5.0, 1.0};
  for (const SchedulerKind k : {SchedulerKind::kMoss, SchedulerKind::kSimple, SchedulerKind::kNormal,
                                SchedulerKind::kKarras, SchedulerKind::kExponential, SchedulerKind::kBeta,
                                SchedulerKind::kSgmUniform}) {
    for (const int steps : {1, 2, 20, 50}) {
      const std::vector<float> want = mi::pipeline::make_schedule(k, steps, fs).sigmas;
      const std::vector<float> got = mi::pipeline::engine_sigmas(k, steps, fs, 1.0);
      EXPECT_EQ(mi::pipeline::first_schedule_difference(got, want), mi::pipeline::kSchedulesEqual)
          << mi::pipeline::scheduler_name(k) << " " << steps;
    }
  }
}

TEST(RunSchedule, EngineSigmasCropTheHeadAsImg2Img) {
  // Pipeline::run's img2img: the strength-1 schedule without its first strength_start_step steps (not the
  // k-diffusion tail of a longer schedule, which make_schedule's own `denoise` argument computes).
  const FlowSampling fs{3.0, 1.0};
  for (const SchedulerKind k : {SchedulerKind::kMoss, SchedulerKind::kKarras, SchedulerKind::kSimple}) {
    const std::vector<float> full = mi::pipeline::make_schedule(k, 20, fs).sigmas;
    const int n = static_cast<int>(full.size()) - 1;
    for (const double strength : {0.5, 0.33, 0.01}) {
      const int start = mi::pipeline::strength_start_step(n, strength);
      const std::vector<float> want(full.begin() + start, full.end());
      EXPECT_EQ(mi::pipeline::first_schedule_difference(mi::pipeline::engine_sigmas(k, 20, fs, strength), want),
                mi::pipeline::kSchedulesEqual)
          << mi::pipeline::scheduler_name(k) << " strength " << strength;
    }
    EXPECT_EQ(mi::pipeline::engine_sigmas(k, 20, fs, 0.5).size(), static_cast<std::size_t>(n - n / 2 + 1));
    EXPECT_EQ(mi::pipeline::engine_sigmas(k, 20, fs, 0.0), std::vector<float>{full.back()});  // no step left
  }
  EXPECT_NE(mi::pipeline::first_schedule_difference(
                mi::pipeline::engine_sigmas(SchedulerKind::kKarras, 20, fs, 0.5),
                mi::pipeline::make_schedule(SchedulerKind::kKarras, 20, fs, 0.5).sigmas),
            mi::pipeline::kSchedulesEqual);
  for (const double bad : {-0.1, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(mi::pipeline::engine_sigmas(SchedulerKind::kMoss, 20, fs, bad), std::invalid_argument);
  }
  EXPECT_THROW(mi::pipeline::engine_sigmas(SchedulerKind::kMoss, 0, fs, 1.0), std::invalid_argument);
}

// ---- the schedule comparison ------------------------------------------------------------------------------------

TEST(RunSchedule, FirstDifferenceIsBitForBit) {
  std::vector<float> a = kSix;
  EXPECT_EQ(mi::pipeline::first_schedule_difference(a, kSix), mi::pipeline::kSchedulesEqual);
  a[3] = up(a[3]);
  EXPECT_EQ(mi::pipeline::first_schedule_difference(a, kSix), 3u);
  a = kSix;
  a.back() = -0.0f;  // equal as floats, not as bits
  EXPECT_EQ(mi::pipeline::first_schedule_difference(a, kSix), 5u);
  a = kSix;
  a.pop_back();
  EXPECT_EQ(mi::pipeline::first_schedule_difference(a, kSix), 5u);
  EXPECT_EQ(mi::pipeline::first_schedule_difference(kSix, a), 5u);
}

TEST(RunSchedule, ScheduleNoteNamesTheFirstDifference) {
  const FlowSampling fs{5.0, 1.0};
  EXPECT_EQ(mi::pipeline::schedule_note(kSix, kSix, SchedulerKind::kSimple, 5, fs, 1.0), "");
  std::vector<float> host = kSix;
  host[3] = up(host[3]);
  const std::string head = "sigmas differ from the engine's simple schedule (5 steps, shift 5, multiplier 1, strength 1)";
  EXPECT_EQ(mi::pipeline::schedule_note(host, kSix, SchedulerKind::kSimple, 5, fs, 1.0),
            head + " at index 3: " + g9(host[3]) + " (host) vs " + g9(kSix[3]) + " (engine)");
  EXPECT_EQ(g9(host[3]), "0.400000036");  // the 1-ulp difference is visible at %.9g

  host = {1.0f, 0.8f, 0.6f};
  EXPECT_EQ(mi::pipeline::schedule_note(host, kSix, SchedulerKind::kSimple, 5, fs, 1.0),
            head + ": 3 host values vs 6 engine values; first difference at index 3: none (host) vs 0.400000006 (engine)");
  host = {1.0f, 0.7f};
  EXPECT_EQ(mi::pipeline::schedule_note(host, kSix, SchedulerKind::kKarras, 5, FlowSampling{3.0, 1000.0}, 0.5),
            "sigmas differ from the engine's karras schedule (5 steps, shift 3, multiplier 1000, strength 0.5): 2 host "
            "values vs 6 engine values; first difference at index 1: 0.699999988 (host) vs 0.800000012 (engine)");
}

// ---- validation -------------------------------------------------------------------------------------------------

TEST(RunSchedule, RejectsMalformedSchedules) {
  EXPECT_THROW(RunSchedule({1.0f}, false), std::invalid_argument);
  EXPECT_THROW(RunSchedule(std::vector<float>(mi::pipeline::kMaxRunSigmas + 1, 0.5f), false), std::invalid_argument);
  EXPECT_THROW(RunSchedule({1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f}, false), std::invalid_argument);
  EXPECT_THROW(RunSchedule({std::numeric_limits<float>::infinity(), 0.0f}, false), std::invalid_argument);
  EXPECT_THROW(RunSchedule({0.5f, 0.6f, 0.0f}, false), std::invalid_argument);
  EXPECT_NO_THROW(RunSchedule(std::vector<float>(mi::pipeline::kMaxRunSigmas, 0.5f), false));
  EXPECT_NO_THROW(RunSchedule({0.5f, 0.5f}, false));  // repeated values: zero-length steps
  EXPECT_NO_THROW(RunSchedule({1.0f, 0.3f}, true));   // the terminal value need not be 0
  try {
    RunSchedule({1.0f, 0.5f, 0.7f}, false);
    FAIL();
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("sigmas[2] = 0.7 > sigmas[1] = 0.5"), std::string::npos) << e.what();
  }
}

TEST(RunSchedule, RejectsNonFiniteSigmasAndBadSteps) {
  const RunSchedule s(kSix, false);
  EXPECT_THROW(s.locate(std::numeric_limits<float>::quiet_NaN(), -1), std::invalid_argument);
  EXPECT_THROW(s.locate(std::numeric_limits<float>::infinity(), -1), std::invalid_argument);
  EXPECT_THROW(s.locate(-std::numeric_limits<float>::infinity(), 2), std::invalid_argument);
  EXPECT_THROW(s.locate(0.2f, 5), std::invalid_argument);  // steps 0..4
  EXPECT_THROW(s.locate(0.2f, -2), std::invalid_argument);
}

// ---- the exact lookup -------------------------------------------------------------------------------------------

TEST(RunSchedule, ExactLookupFollowsTheCursor) {
  RunSchedule s(kSix, true);
  for (int i = 0; i < 5; ++i) {
    const auto p = call(s, kSix[static_cast<std::size_t>(i)]);
    EXPECT_EQ(p.step, i);
    EXPECT_TRUE(p.on_schedule);
    EXPECT_EQ(s.cursor(), i + 1);
  }
  EXPECT_EQ(s.calls(), 5);
  EXPECT_EQ(s.off_schedule_calls(), 0);
  // A restart (a second pass over the same schedule): the cursor is past the end, the lookup wraps to the front.
  EXPECT_EQ(call(s, 1.0f).step, 0);
  EXPECT_EQ(call(s, 0.8f).step, 1);
  // Backwards (an earlier sigma again): found before the cursor.
  EXPECT_EQ(call(s, 0.8f).step, 1);
}

TEST(RunSchedule, RepeatedSigmasResolveToConsecutiveSteps) {
  RunSchedule s({1.0f, 0.5f, 0.5f, 0.5f, 0.0f}, true);
  EXPECT_EQ(call(s, 1.0f).step, 0);
  EXPECT_EQ(call(s, 0.5f).step, 1);
  EXPECT_EQ(call(s, 0.5f).step, 2);
  EXPECT_EQ(call(s, 0.5f).step, 3);
  EXPECT_EQ(call(s, 0.5f).step, 1);  // the cursor is at 4: wraps to the first repeat
}

TEST(RunSchedule, LocateWithoutCommitLeavesTheCursor) {
  // A failed or cancelled call is retried at the same position, even on a zero-length step.
  RunSchedule s({1.0f, 0.5f, 0.5f, 0.0f}, true);
  call(s, 1.0f);
  EXPECT_EQ(s.locate(0.5f, -1).step, 1);
  EXPECT_EQ(s.locate(0.5f, -1).step, 1);
  EXPECT_EQ(s.cursor(), 1);
  EXPECT_EQ(s.calls(), 1);
}

TEST(RunSchedule, ExactLookupNeedsBitEquality) {
  RunSchedule exact(kSix, true);
  EXPECT_THROW(exact.locate(up(0.6f), -1), std::invalid_argument);
  EXPECT_THROW(exact.locate(0.0f, -1), std::invalid_argument);  // the terminal value is not a step
  RunSchedule zero({1.0f, 0.5f, 0.0f}, true);
  EXPECT_THROW(zero.locate(-0.0f, -1), std::invalid_argument);
}

// ---- explicit steps ---------------------------------------------------------------------------------------------

TEST(RunSchedule, ExplicitStepsAndTheCorrector) {
  RunSchedule s(kSix, true);
  // heun: the predictor at s[i] by lookup, the corrector at s[i + 1] with step i, then the next predictor.
  for (int i = 0; i < 4; ++i) {
    const auto pred = call(s, kSix[static_cast<std::size_t>(i)]);
    EXPECT_EQ(pred.step, i);
    const auto corr = call(s, kSix[static_cast<std::size_t>(i) + 1], i);
    EXPECT_EQ(corr.step, i);
    EXPECT_TRUE(corr.on_schedule);
    EXPECT_EQ(s.cursor(), i + 1);
  }
  // Without the explicit step, the corrector's sigma resolves to the next step (whose windows it would take).
  RunSchedule t(kSix, true);
  call(t, 1.0f);
  EXPECT_EQ(t.locate(0.8f, -1).step, 1);
  // An explicit step at its own sigma is on schedule too.
  EXPECT_TRUE(t.locate(0.4f, 3).on_schedule);
  EXPECT_EQ(t.off_schedule_calls(), 0);
}

TEST(RunSchedule, ExplicitStepOffItsSigmasIsOffSchedule) {
  RunSchedule s(kSix, false);
  const auto p = call(s, 0.5f, 3);  // neither s[3] = 0.4 nor s[4] = 0.2
  EXPECT_EQ(p.step, 3);
  EXPECT_FALSE(p.on_schedule);
  EXPECT_EQ(s.off_schedule_calls(), 1);
  EXPECT_EQ(s.cursor(), 4);
  EXPECT_FALSE(call(s, 0.8f, 3).on_schedule);  // another schedule entry, not this step's
  RunSchedule exact(kSix, true);
  EXPECT_THROW(exact.locate(0.5f, 3), std::invalid_argument);
  EXPECT_THROW(exact.locate(0.8f, 3), std::invalid_argument);
}

// ---- off-schedule sigmas (the default) --------------------------------------------------------------------------

TEST(RunSchedule, OffScheduleSigmasTakeTheirInterval) {
  RunSchedule s(kSix, false);
  EXPECT_EQ(s.locate(0.7f, -1).step, 1);   // 0.8 > 0.7 > 0.6
  EXPECT_EQ(s.locate(up(0.6f), -1).step, 1);
  EXPECT_EQ(s.locate(0.19f, -1).step, 4);
  EXPECT_EQ(s.locate(1.5f, -1).step, 0);   // above s[0]
  EXPECT_EQ(s.locate(0.0f, -1).step, 4);   // the terminal value: the last step
  EXPECT_EQ(s.locate(-0.5f, -1).step, 4);  // below s[n]
  EXPECT_EQ(s.locate(-0.0f, -1).step, 4);
  EXPECT_FALSE(s.locate(0.7f, -1).on_schedule);
  EXPECT_TRUE(s.locate(0.6f, -1).on_schedule);
  // A schedule whose terminal value is not 0, and one with a zero-length step.
  RunSchedule t({1.0f, 0.5f, 0.5f, 0.25f}, false);
  EXPECT_EQ(t.locate(0.3f, -1).step, 2);
  EXPECT_EQ(t.locate(0.75f, -1).step, 0);
  EXPECT_EQ(t.locate(0.1f, -1).step, 2);
}

TEST(RunSchedule, UnboundedCallsInAnyOrder) {
  // An ancestral / restart-style host: 10x more calls than steps, midpoints and jumps backwards. Every interval call
  // counts as off schedule; the exact hits do not.
  const std::vector<float> sig = mi::pipeline::make_schedule(SchedulerKind::kMoss, 20, FlowSampling{5.0, 1.0}).sigmas;
  RunSchedule s(sig, false);
  const int n = s.steps();
  std::uint32_t lcg = 12345;
  std::int64_t off = 0, total = 0;
  for (int k = 0; k < 10 * n; ++k) {
    lcg = lcg * 1664525u + 1013904223u;
    float sigma;
    if (k % 3 == 0) {
      sigma = sig[lcg % static_cast<std::uint32_t>(n)];
    } else {
      sigma = static_cast<float>(lcg >> 8) / static_cast<float>(1u << 24) * 1.2f - 0.1f;  // [-0.1, 1.1)
    }
    const auto p = call(s, sigma);
    ++total;
    ASSERT_GE(p.step, 0);
    ASSERT_LT(p.step, n);
    if (!p.on_schedule) {
      ++off;
      const float hi = sig[static_cast<std::size_t>(p.step)], lo = sig[static_cast<std::size_t>(p.step) + 1];
      EXPECT_TRUE((sigma < hi && sigma > lo) || (p.step == 0 && sigma >= hi) || (p.step == n - 1 && sigma <= lo))
          << "sigma " << sigma << " step " << p.step;
    } else {
      EXPECT_EQ(sig[static_cast<std::size_t>(p.step)], sigma);
    }
  }
  EXPECT_EQ(s.calls(), total);
  EXPECT_EQ(s.off_schedule_calls(), off);
  EXPECT_GT(off, 0);
  EXPECT_LT(off, total);
}

TEST(RunSchedule, ExactSigmaRefusesWhatTheDefaultAccepts) {
  RunSchedule exact(kSix, true);
  for (const float sigma : {0.7f, 1.5f, -0.5f, up(0.2f)}) {
    EXPECT_THROW(exact.locate(sigma, -1), std::invalid_argument) << sigma;
  }
  EXPECT_EQ(exact.calls(), 0);
}

}  // namespace
