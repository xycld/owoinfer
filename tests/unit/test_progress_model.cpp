// src/api/progress_model.h: the whole-job progress of mi_progress.fraction / eta_seconds. Pure host logic: the
// segments a plan gives, the load estimates of models not held, and the model's corrections from measured phases and
// steps.
#include <gtest/gtest.h>

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
  m.phase(MI_PHASE_TEXT_ENCODER, 1.0);
  m.phase(MI_PHASE_DIT_LOAD, 4.0);
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
  n.phase(MI_PHASE_TEXT_ENCODER, 1.0);
  n.phase(MI_PHASE_DIT_LOAD, 4.0);
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
  m.phase(MI_PHASE_TEXT_ENCODER, 0.5);
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
  std::vector<ProgressSegment> segments = txt2img();
  segments[0].optional = true;
  m.expect(segments);
  // The optional text-encoder segment never starts: the DiT load's start passes it.
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
  // A phase that is not expected next does not pass a pending one that is not optional: it counts with no estimate,
  // and the pending ones keep theirs.
  ProgressModel n;
  n.expect(txt2img());
  n.phase(MI_PHASE_DENOISE, 0.5);
  EXPECT_NEAR(n.eta_seconds(), 2 + 4 + 10 + 1, 1e-9);
  n.phase(MI_PHASE_TEXT_ENCODER, 1.0);
  EXPECT_NEAR(n.eta_seconds(), 2 + 4 + 10 + 1, 1e-9);
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
  EXPECT_FALSE(e.dit_held);
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
  for (const ProgressSegment& g : s) EXPECT_FALSE(g.optional);  // a generation enters every phase
  // The measured direct-I/O rate prices a cold load, at most at the H2D rate.
  hw.direct_gbps = 20;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, cold, bytes, shape).dit, 10e9 / 12e9, 1e-9);
  hw.direct_gbps = 0;
  // Files read before: the page-cache rate, or the learned warm buffered rate.
  HeldModels read_before;
  read_before.dit_files_read = true;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, read_before, bytes, shape).dit, 10e9 / 8e9, 1e-9);
  hw.buffered_warm_gbps = 4;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, read_before, bytes, shape).dit, 10e9 / 4e9, 1e-9);
  // A soft unload's host copy: the H2D rate.
  HeldModels copy;
  copy.dit_host_copy = true;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, copy, bytes, shape).dit, 10e9 / 12e9, 1e-9);
  // Warm with this residency and no LoRA change: nothing to load.
  HeldModels warm;
  warm.warm = true;
  warm.dit = true;
  warm.last_dit = plan.dit;
  e = mi::api::load_estimates(plan, hw, warm, bytes, shape);
  EXPECT_TRUE(e.dit_held);
  EXPECT_EQ(e.dit, 0.0);
  EXPECT_EQ(e.vae_decoder, 0.0);
  EXPECT_EQ(mi::api::generate_segments(plan, hw, warm, bytes, shape)[1].seconds, 0.0);
  // Another residency loads.
  plan.dit.residency = mi::planner::DitResidency::kStreamed;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, warm, bytes, shape).dit, 10e9 / 2e9, 1e-9);
  // A parked DiT reloads at the plan's dit_load_s, unless the last job left it loaded (no decode parking) and this one
  // runs no text encoder.
  plan.dit.residency = mi::planner::DitResidency::kParked;
  plan.time.dit_load_s = 3;
  warm.last_dit = plan.dit;
  EXPECT_NEAR(mi::api::generate_segments(plan, hw, warm, bytes, shape)[1].seconds, 3.0, 1e-9);
  plan.te.run = false;
  EXPECT_EQ(mi::api::generate_segments(plan, hw, warm, bytes, shape)[1].seconds, 0.0);
  plan.dit.park_for_decode = true;  // the last job parked it for its decode: evicted, reloaded at dit_load_s
  warm.last_dit = plan.dit;
  EXPECT_NEAR(mi::api::generate_segments(plan, hw, warm, bytes, shape)[1].seconds, 3.0, 1e-9);
  // Other parking than the held DiT's: not the same residency, a load.
  warm.last_dit.park_for_decode = false;
  EXPECT_NEAR(mi::api::generate_segments(plan, hw, warm, bytes, shape)[1].seconds, 5.0, 1e-9);
  // Cold, a parked DiT costs the larger of its load and the plan's reload.
  EXPECT_NEAR(mi::api::generate_segments(plan, hw, cold, bytes, shape)[1].seconds, 5.0, 1e-9);
  plan.te.run = true;
  plan.dit = resident_plan().dit;
  plan.time.dit_load_s = 0;
  // A host-resident text encoder without its copy reads it from the files; a VRAM-resident one loads unless the record
  // holds it (te_vram), capped by the H2D rate.
  plan.te.residency = mi::planner::TeResidency::kHostResident;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, warm, bytes, shape).text_encoder, 14e9 / 2e9, 1e-9);
  warm.te_host_copy = true;
  EXPECT_EQ(mi::api::load_estimates(plan, hw, warm, bytes, shape).text_encoder, 0.0);
  plan.te.run = false;  // an embedding-cache hit
  EXPECT_EQ(mi::api::load_estimates(plan, hw, cold, bytes, shape).text_encoder, 0.0);
  plan.te.run = true;
  plan.te.residency = mi::planner::TeResidency::kVramResident;
  EXPECT_NEAR(mi::api::load_estimates(plan, hw, cold, bytes, shape).text_encoder, 14e9 / 2e9, 1e-9);
  warm.te_vram = true;
  EXPECT_EQ(mi::api::load_estimates(plan, hw, warm, bytes, shape).text_encoder, 0.0);
  plan.te.residency = mi::planner::TeResidency::kStreamed;
  // An image request loads the VAE encoder before the DiT load when cold.
  shape.encodes_image = true;
  s = mi::api::generate_segments(plan, hw, cold, bytes, shape);
  EXPECT_NEAR(s[1].seconds, 5.0 + 0.1, 1e-9);
  EXPECT_EQ(mi::api::load_estimates(plan, hw, warm, bytes, shape).vae_encoder, 0.0);
  // No sampling step (denoise strength 0): no DiT load.
  shape.total_steps = shape.first_steps = 0;
  EXPECT_EQ(mi::api::load_estimates(plan, hw, cold, bytes, shape).dit, 0.0);
  EXPECT_NEAR(mi::api::generate_segments(plan, hw, cold, bytes, shape)[1].seconds, 0.1, 1e-12);
}

// The warm / cold record across jobs, as the adapter feeds it.
TEST(ProgressModel, HeldModelsRecord) {
  const mi::planner::ExecutionPlan plan = resident_plan();
  const mi::planner::HardwareProfile hw = slow_disk();
  const ModelBytes bytes{14'000'000'000, 200'000'000};
  JobShape shape;
  shape.total_steps = shape.first_steps = 20;
  const auto dit = [&](const HeldModels& h) { return mi::api::load_estimates(plan, hw, h, bytes, shape).dit; };
  const auto vae = [&](const HeldModels& h) { return mi::api::load_estimates(plan, hw, h, bytes, shape).vae_decoder; };

  HeldModels h;  // a new engine: cold
  EXPECT_FALSE(h.warm);
  EXPECT_FALSE(h.dit);
  EXPECT_GT(dit(h), 0.0);
  // A generation that loaded from the files: warm, the DiT held, and the files were read.
  h.job_done(plan, shape, false, false, 1, "files");
  EXPECT_TRUE(h.warm);
  EXPECT_TRUE(h.dit);
  EXPECT_TRUE(h.dit_files_read);
  EXPECT_EQ(dit(h), 0.0);
  EXPECT_EQ(vae(h), 0.0);
  // Cancelled without a new arena, with the same DiT: still warm. Cancelled after creating one: cold.
  h.job_done(plan, shape, false, true, 0, "");
  EXPECT_TRUE(h.warm);
  EXPECT_TRUE(h.dit);
  h.job_done(plan, shape, false, true, 1, "");
  EXPECT_FALSE(h.warm);
  EXPECT_FALSE(h.dit);
  h.job_done(plan, shape, false, false, 0, "");
  EXPECT_TRUE(h.warm);
  // A cancelled job without a new arena that used another residency or LoRA stack: the held DiT may be gone.
  mi::planner::ExecutionPlan streamed = plan;
  streamed.dit.residency = mi::planner::DitResidency::kStreamed;
  h.job_done(streamed, shape, false, true, 0, "");
  EXPECT_TRUE(h.warm);
  EXPECT_FALSE(h.dit);
  EXPECT_GT(dit(h), 0.0);
  h.job_done(plan, shape, false, false, 0, "");
  JobShape other_loras = shape;
  other_loras.merge_key = "/loras/b.safetensors\n1.000000\n";
  h.job_done(plan, other_loras, false, true, 0, "");
  EXPECT_FALSE(h.dit);
  h.job_done(plan, shape, false, false, 0, "");
  // An exception from the pipeline: cold.
  h.job_failed();
  EXPECT_FALSE(h.warm);
  EXPECT_FALSE(h.dit);
  EXPECT_NEAR(dit(h), 10e9 / 8e9, 1e-9);  // the files read before: the page-cache rate
  EXPECT_GT(vae(h), 0.0);
  h.job_done(plan, shape, false, false, 0, "");
  // A LoRA change prices a DiT load (the VAE stays loaded); the same stack again does not.
  shape.merge_key = "/loras/a.safetensors\n0.800000\n";
  EXPECT_NEAR(dit(h), 10e9 / 8e9, 1e-9);
  EXPECT_EQ(vae(h), 0.0);
  h.job_done(plan, shape, false, false, 0, "files");
  EXPECT_EQ(dit(h), 0.0);
  // A release with the DiT's host copy: cold, and the next load reads the copy at the H2D rate; it is gone after it.
  h.released(true, false);
  EXPECT_FALSE(h.warm);
  EXPECT_FALSE(h.dit);
  EXPECT_NEAR(dit(h), 10e9 / 12e9, 1e-9);
  EXPECT_GT(vae(h), 0.0);
  h.job_done(plan, shape, false, false, 1, "host_copy");
  EXPECT_FALSE(h.dit_host_copy);
  EXPECT_TRUE(h.warm);
  // A host-resident text encoder's copy is kept by a job that ran it, and by a release that keeps it.
  mi::planner::ExecutionPlan host_te = plan;
  host_te.te.residency = mi::planner::TeResidency::kHostResident;
  h.job_done(host_te, shape, false, false, 0, "");
  EXPECT_TRUE(h.te_host_copy);
  h.released(false, true);
  EXPECT_TRUE(h.te_host_copy);
  EXPECT_EQ(mi::api::load_estimates(host_te, hw, h, bytes, shape).text_encoder, 0.0);
  h.released(false, false);
  EXPECT_NEAR(mi::api::load_estimates(host_te, hw, h, bytes, shape).text_encoder, 14e9 / 2e9, 1e-9);

  // A job that did not use the DiT does not mark it held: no sampling step (img2img at strength 0), or a preload that
  // left a DiT parked for the text encoder to the generation.
  HeldModels no_dit;
  JobShape no_steps = shape;
  no_steps.total_steps = no_steps.first_steps = 0;
  no_dit.job_done(plan, no_steps, false, false, 1, "");
  EXPECT_TRUE(no_dit.warm);
  EXPECT_FALSE(no_dit.dit);
  EXPECT_GT(dit(no_dit), 0.0);
  mi::planner::ExecutionPlan parked = plan;
  parked.dit.residency = mi::planner::DitResidency::kParked;
  parked.dit.park_for_text_encoder = true;
  HeldModels preloaded_parked;
  preloaded_parked.job_done(parked, shape, true, false, 1, "");
  EXPECT_FALSE(preloaded_parked.dit);
  // A new arena in a job without the DiT drops a held one.
  no_dit.job_done(plan, shape, false, false, 0, "files");
  EXPECT_TRUE(no_dit.dit);
  no_dit.job_done(plan, no_steps, false, false, 1, "");
  EXPECT_FALSE(no_dit.dit);
  // Parking differs: another residency (the planner keeps a DiT only with the same parking).
  HeldModels parked_held;
  parked_held.job_done(parked, shape, false, false, 1, "files");
  mi::planner::ExecutionPlan parked_decode = parked;
  parked_decode.dit.park_for_decode = true;
  EXPECT_TRUE(mi::api::load_estimates(parked, hw, parked_held, bytes, shape).dit_held);
  EXPECT_FALSE(mi::api::load_estimates(parked_decode, hw, parked_held, bytes, shape).dit_held);

  // A preload, then a generation of a like request: the generation's DIT_LOAD and DECODE price no load.
  HeldModels fresh;
  const std::vector<ProgressSegment> pre = mi::api::preload_segments(plan, hw, fresh, bytes, shape);
  ASSERT_EQ(pre.size(), 2u);
  EXPECT_NEAR(pre[0].seconds, 10e9 / 2e9, 1e-9);
  fresh.job_done(plan, shape, true, false, 1, "files");
  const std::vector<ProgressSegment> gen = mi::api::generate_segments(plan, hw, fresh, bytes, shape);
  EXPECT_EQ(gen[1].seconds, 0.0);
  EXPECT_NEAR(gen[3].seconds, plan.time.decode_s, 1e-12);

  // A VRAM-resident text encoder: a preload does not load it, so the generation after it prices its load; a generation
  // that ran it holds it; an embedding-cache hit neither loads nor drops it; a new arena drops it.
  mi::planner::ExecutionPlan vram_te = plan;
  vram_te.te.residency = mi::planner::TeResidency::kVramResident;
  const auto te = [&](const HeldModels& m) { return mi::api::load_estimates(vram_te, hw, m, bytes, shape).text_encoder; };
  HeldModels v;
  v.job_done(vram_te, shape, true, false, 1, "files");
  EXPECT_TRUE(v.warm);
  EXPECT_FALSE(v.te_vram);
  EXPECT_NEAR(te(v), 14e9 / 2e9, 1e-9);
  EXPECT_NEAR(mi::api::generate_segments(vram_te, hw, v, bytes, shape)[0].seconds, vram_te.time.text_encoder_s + 7.0, 1e-9);
  v.job_done(vram_te, shape, false, false, 0, "");
  EXPECT_TRUE(v.te_vram);
  EXPECT_EQ(te(v), 0.0);
  mi::planner::ExecutionPlan cache_hit = vram_te;
  cache_hit.te.run = false;
  v.job_done(cache_hit, shape, false, false, 0, "");
  EXPECT_TRUE(v.te_vram);
  v.job_done(cache_hit, shape, false, false, 1, "");
  EXPECT_FALSE(v.te_vram);
  EXPECT_NEAR(te(v), 14e9 / 2e9, 1e-9);
}

TEST(ProgressModel, HiresPasses) {
  const mi::planner::ExecutionPlan plan = resident_plan();
  JobShape shape;
  shape.total_steps = 20;
  shape.first_steps = 12;
  shape.hires = true;
  shape.first_pixel_share = 0.25;
  HeldModels held;
  held.warm = true;
  held.dit = true;
  held.last_dit = plan.dit;
  const std::vector<ProgressSegment> s = mi::api::generate_segments(plan, slow_disk(), held, ModelBytes{}, shape);
  ASSERT_EQ(s.size(), 6u);
  const std::vector<mi_phase> phases = {MI_PHASE_TEXT_ENCODER, MI_PHASE_DIT_LOAD, MI_PHASE_DENOISE,
                                        MI_PHASE_DECODE,       MI_PHASE_DENOISE,  MI_PHASE_DECODE};
  for (std::size_t i = 0; i < s.size(); ++i) EXPECT_EQ(s[i].phase, phases[i]) << i;
  EXPECT_EQ(s[2].steps, 12);
  EXPECT_EQ(s[4].steps, 8);
  // As the planner counts them: the 12 first-pass steps at a quarter of a target step, 8 + 12 / 4 = 11 target steps.
  EXPECT_NEAR(s[4].step_seconds, 20.0 / 11, 1e-9);
  EXPECT_NEAR(s[2].step_seconds, 20.0 / 11 * 0.25, 1e-9);
  EXPECT_NEAR(s[2].steps * s[2].step_seconds + s[4].steps * s[4].step_seconds, plan.time.denoise_s, 1e-9);
  EXPECT_NEAR(s[3].seconds, 0.5 * 0.25, 1e-9);
  ProgressModel m;
  m.expect(s);
  m.phase(MI_PHASE_TEXT_ENCODER, 0);
  m.phase(MI_PHASE_DIT_LOAD, 1.5);
  m.phase(MI_PHASE_DENOISE, 1.5);
  EXPECT_NEAR(m.eta_seconds(), 20 + 0.125 + 0.5, 1e-9);
  // Pass 1 measured at 0.25 s a step: pass 2's steps are priced at 0.25 / 0.25 = 1 s until they are measured.
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
  EXPECT_TRUE(s[0].optional);
  EXPECT_EQ(s[1].phase, MI_PHASE_DIT_LOAD);
  EXPECT_NEAR(s[1].seconds, 5.0, 1e-9);
  EXPECT_FALSE(s[1].optional);
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
  // The optional text-encoder step may not run: the DiT load's start passes it.
  ProgressModel skip;
  skip.expect(s);
  skip.phase(MI_PHASE_DIT_LOAD, 0.1);
  EXPECT_NEAR(skip.eta_seconds(), 5.1, 1e-9);
  // A DiT the plan parks for the text encoder is not preloaded, nor a VAE decoder it parks for; the VAE encoder of a
  // request that encodes an image comes before the DiT.
  plan.dit.park_for_text_encoder = true;
  plan.dit.park_for_decode = true;
  shape.encodes_image = true;
  const std::vector<ProgressSegment> parked = mi::api::preload_segments(plan, slow_disk(), HeldModels{}, bytes, shape);
  ASSERT_EQ(parked.size(), 2u);
  EXPECT_EQ(parked[0].phase, MI_PHASE_TEXT_ENCODER);
  EXPECT_EQ(parked[1].phase, MI_PHASE_DECODE);
}

// A preload's DECODE that nothing expected (a VAE encoder the shape did not predict) before the DiT: it does not take
// the decoder's segment past the pending DiT load, whose estimate stays.
TEST(ProgressModel, UnexpectedDecodeInPreload) {
  const ModelBytes bytes{14'000'000'000, 200'000'000};
  JobShape shape;
  shape.total_steps = shape.first_steps = 20;
  const std::vector<ProgressSegment> s =
      mi::api::preload_segments(resident_plan(), slow_disk(), HeldModels{}, bytes, shape);
  ASSERT_EQ(s.size(), 2u);  // DIT_LOAD 5 s, DECODE (the decoder) 0.1 s
  ProgressModel m;
  m.expect(s);
  std::vector<double> f;
  m.phase(MI_PHASE_DECODE, 0.2);  // the encoder: unexpected, no estimate
  f.push_back(m.fraction());
  EXPECT_NEAR(m.eta_seconds(), 5.0 + 0.1, 1e-9);
  m.phase(MI_PHASE_DIT_LOAD, 0.4);
  f.push_back(m.fraction());
  EXPECT_NEAR(m.eta_seconds(), 5.0 + 0.1, 1e-9);
  m.phase(MI_PHASE_DECODE, 5.4);
  f.push_back(m.fraction());
  EXPECT_NEAR(m.eta_seconds(), 0.1, 1e-9);
  expect_monotonic(f);
  EXPECT_GT(f.back(), 0.9);
}

}  // namespace

// Reference images: each pass's prefill (the plan's reference_prefill_s, inside denoise_s) is that denoise segment's
// work before its first step, and the steps are priced without it.
TEST(ProgressModel, ReferencePrefillIsTheFirstUnitOfEachPass) {
  mi::planner::ExecutionPlan plan = resident_plan();
  const double denoise = plan.time.denoise_s;
  plan.time.reference_prefill_s = 2.0;
  plan.time.denoise_s = denoise + 2.0;
  JobShape shape;
  shape.total_steps = shape.first_steps = 20;
  HeldModels warm;
  warm.warm = true;
  std::vector<ProgressSegment> s = mi::api::generate_segments(plan, slow_disk(), warm, ModelBytes{}, shape);
  ASSERT_EQ(s[2].phase, MI_PHASE_DENOISE);
  EXPECT_NEAR(s[2].seconds, 2.0, 1e-9);
  EXPECT_NEAR(s[2].step_seconds, denoise / 20, 1e-9);
  // A hires fix: one prefill per pass.
  shape.hires = true;
  shape.first_steps = 10;
  shape.first_pixel_share = 0.5;
  shape.encodes_image = true;  // the reference encode: no VAE encoder load in the second pass
  s = mi::api::generate_segments(plan, slow_disk(), warm, ModelBytes{}, shape);
  ASSERT_EQ(s[4].phase, MI_PHASE_DENOISE);
  EXPECT_NEAR(s[2].seconds, 1.0, 1e-9);
  EXPECT_NEAR(s[4].seconds, 1.0, 1e-9);
  EXPECT_NEAR(s[2].step_seconds * 10 + s[4].step_seconds * 10, denoise, 1e-9);
}
