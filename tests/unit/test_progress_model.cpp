// src/api/progress_model.h: the whole-job progress of mi_progress.fraction / eta_seconds. Pure host logic: the
// segments a plan gives, the load estimates of models not held, and the model's corrections from measured phases and
// steps.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "api/progress_model.h"
#include "planner/planner.h"

namespace {

using mi::api::HeldModels;
using mi::api::JobShape;
using mi::api::kFractionBeforeEnd;
using mi::api::ModelBytes;
using mi::api::ProgressModel;
using mi::api::ProgressSegment;

// Text encoder 2 s, DiT load 4 s, 10 steps of 1 s, decode 1 s.
std::vector<ProgressSegment> txt2img() {
  return {{MI_PHASE_TEXT_ENCODER, 2}, {MI_PHASE_DIT_LOAD, 4}, {MI_PHASE_DENOISE, 0, 10, 1.0}, {MI_PHASE_DECODE, 1}};
}

// Every value in [0, kFractionBeforeEnd] and none below the one before.
void expect_monotonic(const std::vector<double>& fractions) {
  for (std::size_t i = 0; i < fractions.size(); ++i) {
    EXPECT_GE(fractions[i], 0.0) << i;
    EXPECT_LE(fractions[i], kFractionBeforeEnd) << i;
    if (i > 0) {
      EXPECT_GE(fractions[i], fractions[i - 1]) << i;
    }
  }
}

mi::planner::ExecutionPlan resident_plan() {
  mi::planner::ExecutionPlan plan;
  plan.arena_bytes = 12'000'000'000;
  plan.dit.residency = mi::planner::DitResidency::kResident;
  plan.dit.device_bytes = 10'000'000'000;
  plan.te.run = true;
  plan.te.residency = mi::planner::TeResidency::kStreamed;
  plan.vae.run = true;
  plan.time.text_encoder_s = 1.5;
  plan.time.denoise_s = 20;
  plan.time.decode_s = 0.5;
  return plan;
}

mi::planner::HardwareProfile slow_disk() {
  mi::planner::HardwareProfile hw;
  hw.disk_gbps = 2;
  hw.host_gbps = 8;
  hw.h2d_gbps = 12;
  return hw;
}

TEST(ProgressModel, FractionInRangeAndNeverDecreasing) {
  ProgressModel m;
  std::vector<double> f;
  m.expect(txt2img());
  m.phase(MI_PHASE_TEXT_ENCODER, 0.5);
  f.push_back(m.fraction());
  m.phase(MI_PHASE_DIT_LOAD, 6.5);  // the text encoder took 6 s, not 2
  f.push_back(m.fraction());
  m.phase(MI_PHASE_DENOISE, 7.0);
  f.push_back(m.fraction());
  // A slow first step (3 s) prices the other 9 at 3 s: the raw fraction 10 / 38 is below the last one, which stays.
  m.step(1, 3.0, 10.0);
  f.push_back(m.fraction());
  EXPECT_DOUBLE_EQ(f[3], f[2]);
  EXPECT_NEAR(m.eta_seconds(), 9 * 3.0 + 1, 1e-9);
  for (int s = 2; s <= 10; ++s) {
    m.step(s, 1.0, 10.0 + (s - 1));
    f.push_back(m.fraction());
  }
  m.phase(MI_PHASE_DECODE, 19.0);
  f.push_back(m.fraction());
  expect_monotonic(f);
  EXPECT_GT(f.back(), 0.9);
  EXPECT_LT(f.back(), 1.0);  // 1 only when the job ends
}

TEST(ProgressModel, EtaShrinksAsStepsRun) {
  ProgressModel m;
  m.expect(txt2img());
  m.phase(MI_PHASE_TEXT_ENCODER, 0.1);
  m.phase(MI_PHASE_DIT_LOAD, 2.1);
  m.phase(MI_PHASE_DENOISE, 6.1);
  EXPECT_NEAR(m.eta_seconds(), 10 * 1.0 + 1, 1e-9);
  double last = m.eta_seconds();
  for (int s = 1; s <= 10; ++s) {
    m.step(s, 0.5, 6.1 + 0.5 * s);  // measured at half the plan's step time
    EXPECT_LT(m.eta_seconds(), last) << s;
    last = m.eta_seconds();
  }
  EXPECT_NEAR(m.eta_seconds(), 1.0, 1e-9);  // only the decode is left
  m.phase(MI_PHASE_DECODE, 11.1);
  EXPECT_NEAR(m.eta_seconds(), 1.0, 1e-9);
  EXPECT_NEAR(m.fraction(), 11.1 / 12.1, 1e-9);
}

TEST(ProgressModel, MeasuredPhaseReplacesItsEstimate) {
  ProgressModel m;
  m.expect({{MI_PHASE_TEXT_ENCODER, 10}, {MI_PHASE_DIT_LOAD, 4}, {MI_PHASE_DENOISE, 0, 10, 1.0}, {MI_PHASE_DECODE, 1}});
  m.phase(MI_PHASE_TEXT_ENCODER, 0.0);
  EXPECT_NEAR(m.eta_seconds(), 10 + 4 + 10 + 1, 1e-9);
  EXPECT_EQ(m.fraction(), 0.0);
  // It took 2 s: the 2 s count as done, its 10 s estimate is gone.
  m.phase(MI_PHASE_DIT_LOAD, 2.0);
  EXPECT_NEAR(m.eta_seconds(), 4 + 10 + 1, 1e-9);
  EXPECT_NEAR(m.fraction(), 2.0 / 17.0, 1e-9);
}

TEST(ProgressModel, StepsAfterTheFirstSetThePrice) {
  ProgressModel m;
  m.expect(txt2img());
  m.phase(MI_PHASE_DENOISE, 10.0);
  // The first event 3 s after the phase start: refiner and setup, then a 1.5 s step (its own time).
  m.step(1, 1.5, 13.0);
  EXPECT_NEAR(m.eta_seconds(), 9 * 1.5 + 1, 1e-9);
  m.step(2, 0.8, 13.8);
  m.step(3, 0.8, 14.6);
  EXPECT_NEAR(m.eta_seconds(), 7 * 0.8 + 1, 1e-9);
  // Without a step time (0), the first step's price is its wall time since the phase started.
  ProgressModel n;
  n.expect(txt2img());
  n.phase(MI_PHASE_DENOISE, 10.0);
  n.step(1, 0.0, 12.0);
  EXPECT_NEAR(n.eta_seconds(), 9 * 2.0 + 1, 1e-9);
}

TEST(ProgressModel, UnknownEta) {
  ProgressModel m;
  m.phase(MI_PHASE_TEXT_ENCODER, 0.2);  // no plan yet
  EXPECT_LT(m.eta_seconds(), 0.0);
  EXPECT_EQ(m.fraction(), 0.0);
  m.expect(txt2img());
  m.phase(MI_PHASE_DIT_LOAD, 1.0);
  EXPECT_NEAR(m.eta_seconds(), 4 + 10 + 1, 1e-9);
  // A non-finite estimate: unknown again, and the fraction keeps its value.
  ProgressModel n;
  n.expect({{MI_PHASE_TEXT_ENCODER, 1}, {MI_PHASE_DIT_LOAD, std::numeric_limits<double>::infinity()}});
  n.phase(MI_PHASE_TEXT_ENCODER, 0.5);
  EXPECT_LT(n.eta_seconds(), 0.0);
  EXPECT_EQ(n.fraction(), 0.0);
}

TEST(ProgressModel, SkippedAndUnexpectedPhases) {
  ProgressModel m;
  m.expect(txt2img());
  // The text-encoder segment never starts: the DiT load's start drops it.
  m.phase(MI_PHASE_DIT_LOAD, 0.5);
  EXPECT_NEAR(m.eta_seconds(), 4 + 10 + 1, 1e-9);
  m.phase(MI_PHASE_DENOISE, 4.5);
  for (int s = 1; s <= 10; ++s) m.step(s, 1.0, 4.5 + s);
  m.phase(MI_PHASE_DECODE, 14.5);
  const double before = m.fraction();
  // A phase nothing expected counts with no estimate.
  m.phase(MI_PHASE_DECODE, 15.5);
  EXPECT_NEAR(m.eta_seconds(), 0.0, 1e-12);
  EXPECT_GE(m.fraction(), before);
  EXPECT_LE(m.fraction(), kFractionBeforeEnd);
}

TEST(ProgressModel, ColdLoadEstimate) {
  mi::planner::ExecutionPlan plan = resident_plan();
  mi::planner::HardwareProfile hw = slow_disk();
  const ModelBytes bytes{14'000'000'000, 200'000'000};
  JobShape shape;
  shape.total_steps = shape.first_steps = 20;
  HeldModels cold;
  // Nothing loaded: the DiT's plan bytes at the cold-disk rate (below the H2D rate), the VAE decoder likewise.
  mi::api::LoadEstimates e = mi::api::load_estimates(plan, hw, cold, bytes, shape);
  EXPECT_NEAR(e.dit, 10e9 / 2e9, 1e-9);
  EXPECT_NEAR(e.vae_decoder, 0.2e9 / 2e9, 1e-9);
  EXPECT_EQ(e.text_encoder, 0.0);  // a streamed encoder: the plan's own estimate
  EXPECT_EQ(e.vae_encoder, 0.0);   // no image to encode
  std::vector<ProgressSegment> s = mi::api::generate_segments(plan, hw, cold, bytes, shape);
  ASSERT_EQ(s.size(), 4u);
  EXPECT_EQ(s[1].phase, MI_PHASE_DIT_LOAD);
  EXPECT_NEAR(s[1].seconds, 5.0, 1e-9);
  EXPECT_NEAR(s[0].seconds, 1.5, 1e-9);
  EXPECT_EQ(s[2].steps, 20);
  EXPECT_NEAR(s[2].step_seconds, 1.0, 1e-9);
  EXPECT_NEAR(s[3].seconds, 0.5 + 0.1, 1e-9);
  // The measured direct-I/O rate prices a cold load, at most at the H2D rate.
  hw.direct_gbps = 20;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, cold, bytes, shape).dit, 10e9 / 12e9, 1e-9);
  hw.direct_gbps = 0;
  // Files read before: the page-cache rate, or the learned warm buffered rate.
  HeldModels warm;
  warm.dit_files_read = true;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, warm, bytes, shape).dit, 10e9 / 8e9, 1e-9);
  hw.buffered_warm_gbps = 4;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, warm, bytes, shape).dit, 10e9 / 4e9, 1e-9);
  // A soft unload's host copy: the H2D rate.
  HeldModels copy;
  copy.dit_host_copy = true;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, copy, bytes, shape).dit, 10e9 / 12e9, 1e-9);
  // Loaded with this residency in a large enough arena: nothing to load; a larger arena drops the models.
  HeldModels held;
  held.loaded = true;
  held.dit = plan.dit;
  held.vae = true;
  held.arena_bytes = plan.arena_bytes;
  e = mi::api::load_estimates(plan, hw, held, bytes, shape);
  EXPECT_EQ(e.dit, 0.0);
  EXPECT_EQ(e.vae_decoder, 0.0);
  held.arena_bytes = plan.arena_bytes - 1;
  EXPECT_GT(mi::api::load_estimates(plan, hw, held, bytes, shape).dit, 0.0);
  held.arena_bytes = plan.arena_bytes;
  // Another residency reloads.
  plan.dit.residency = mi::planner::DitResidency::kStreamed;
  EXPECT_GT(mi::api::load_estimates(plan, hw, held, bytes, shape).dit, 0.0);
  plan.dit.residency = mi::planner::DitResidency::kResident;
  // A parked DiT reloads every generation at the plan's dit_load_s.
  plan.time.dit_load_s = 3;
  EXPECT_NEAR(mi::api::generate_segments(plan, hw, held, bytes, shape)[1].seconds, 3.0, 1e-9);
  // A host-resident text encoder without its copy reads it from the files; a VRAM-resident one is capped by the H2D rate.
  plan.te.residency = mi::planner::TeResidency::kHostResident;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, held, bytes, shape).text_encoder, 14e9 / 2e9, 1e-9);
  held.te_host_copy = true;
  EXPECT_EQ(mi::api::load_estimates(plan, hw, held, bytes, shape).text_encoder, 0.0);
  plan.te.run = false;  // an embedding-cache hit
  held.te_host_copy = false;
  EXPECT_EQ(mi::api::load_estimates(plan, hw, held, bytes, shape).text_encoder, 0.0);
  // An image request loads the VAE encoder before the DiT load.
  plan.te.run = true;
  plan.te.residency = mi::planner::TeResidency::kStreamed;
  shape.encodes_image = true;
  s = mi::api::generate_segments(plan, hw, cold, bytes, shape);
  EXPECT_NEAR(s[1].seconds, 5.0 + 0.1, 1e-9);
  // No sampling step (denoise strength 0): no DiT load.
  shape.total_steps = shape.first_steps = 0;
  EXPECT_EQ(mi::api::load_estimates(plan, hw, cold, bytes, shape).dit, 0.0);
}

TEST(ProgressModel, HiresPasses) {
  const mi::planner::ExecutionPlan plan = resident_plan();
  JobShape shape;
  shape.total_steps = 20;
  shape.first_steps = 12;
  shape.hires = true;
  shape.first_pixel_share = 0.25;
  HeldModels held;
  held.loaded = true;
  held.dit = plan.dit;
  held.vae = true;
  held.encoder = true;
  held.arena_bytes = plan.arena_bytes;
  const std::vector<ProgressSegment> s = mi::api::generate_segments(plan, slow_disk(), held, ModelBytes{}, shape);
  ASSERT_EQ(s.size(), 6u);
  const std::vector<mi_phase> phases = {MI_PHASE_TEXT_ENCODER, MI_PHASE_DIT_LOAD, MI_PHASE_DENOISE,
                                        MI_PHASE_DECODE,       MI_PHASE_DENOISE,  MI_PHASE_DECODE};
  for (std::size_t i = 0; i < s.size(); ++i) EXPECT_EQ(s[i].phase, phases[i]) << i;
  EXPECT_EQ(s[2].steps, 12);
  EXPECT_EQ(s[4].steps, 8);
  EXPECT_NEAR(s[3].seconds, 0.5 * 0.25, 1e-9);
  // The first pass's measured price does not carry over: the second pass is priced at the plan until it is measured.
  ProgressModel m;
  m.expect(s);
  m.phase(MI_PHASE_TEXT_ENCODER, 0);
  m.phase(MI_PHASE_DIT_LOAD, 1.5);
  m.phase(MI_PHASE_DENOISE, 1.5);
  for (int k = 1; k <= 12; ++k) m.step(k, 0.25, 1.5 + 0.25 * k);
  EXPECT_NEAR(m.eta_seconds(), 0.125 + 8 * 1.0 + 0.5, 1e-9);
  m.phase(MI_PHASE_DECODE, 4.5);
  m.phase(MI_PHASE_DENOISE, 4.7);
  EXPECT_NEAR(m.eta_seconds(), 8 * 1.0 + 0.5, 1e-9);
  m.step(13, 2.0, 6.7);
  m.step(14, 2.0, 8.7);
  EXPECT_NEAR(m.eta_seconds(), 6 * 2.0 + 0.5, 1e-9);
}

TEST(ProgressModel, PreloadLoadSteps) {
  mi::planner::ExecutionPlan plan = resident_plan();
  plan.te.residency = mi::planner::TeResidency::kHostResident;
  const ModelBytes bytes{14'000'000'000, 200'000'000};
  JobShape shape;
  shape.total_steps = shape.first_steps = 20;
  const std::vector<ProgressSegment> s = mi::api::preload_segments(plan, slow_disk(), HeldModels{}, bytes, shape);
  // The loads only, in the pipeline's order: the encoder's host copy, the DiT, the VAE decoder. No denoise.
  ASSERT_EQ(s.size(), 3u);
  EXPECT_EQ(s[0].phase, MI_PHASE_TEXT_ENCODER);
  EXPECT_NEAR(s[0].seconds, 7.0, 1e-9);
  EXPECT_EQ(s[1].phase, MI_PHASE_DIT_LOAD);
  EXPECT_NEAR(s[1].seconds, 5.0, 1e-9);
  EXPECT_EQ(s[2].phase, MI_PHASE_DECODE);
  EXPECT_NEAR(s[2].seconds, 0.1, 1e-9);
  for (const ProgressSegment& g : s) EXPECT_EQ(g.steps, 0);
  ProgressModel m;
  m.expect(s);
  std::vector<double> f;
  m.phase(MI_PHASE_TEXT_ENCODER, 0.1);
  f.push_back(m.fraction());
  EXPECT_NEAR(m.eta_seconds(), 12.1, 1e-9);
  m.phase(MI_PHASE_DIT_LOAD, 4.1);
  f.push_back(m.fraction());
  EXPECT_NEAR(m.eta_seconds(), 5.1, 1e-9);
  m.phase(MI_PHASE_DECODE, 6.1);
  f.push_back(m.fraction());
  EXPECT_NEAR(m.eta_seconds(), 0.1, 1e-9);
  expect_monotonic(f);
  // A DiT the plan parks for the text encoder is not preloaded, nor a VAE decoder it parks for; an image request's
  // VAE encoder comes before the DiT.
  plan.dit.park_for_text_encoder = true;
  plan.dit.park_for_decode = true;
  shape.encodes_image = true;
  const std::vector<ProgressSegment> parked = mi::api::preload_segments(plan, slow_disk(), HeldModels{}, bytes, shape);
  ASSERT_EQ(parked.size(), 2u);
  EXPECT_EQ(parked[0].phase, MI_PHASE_TEXT_ENCODER);
  EXPECT_EQ(parked[1].phase, MI_PHASE_DECODE);
}

}  // namespace
