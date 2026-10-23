// Pipeline residency across calls (1.1): preload before the first generate(), the soft unload
// (release_device_memory) and its host copy, the DiT load's I/O mode, the text encoder kept host-resident, and the cost
// of switching a control adapter on, on real weights (GTEST_SKIP when absent). Small images and few steps: these check
// reuse, timings and bit identity, not speed.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "planner/host_memory.h"
#include "planner/planner.h"
#include "test_util.h"
#include "util/json.h"
#include "weights/test_seams.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"

namespace {

namespace fs = std::filesystem;
using mi::pipeline::Pipeline;
using mi::pipeline::PipelineOptions;
using mi::pipeline::Request;
using mi::pipeline::Result;

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
const fs::path kInt4 = "/workspace/Models/community/toxicdog/moss_image_sft_int4_convrot.safetensors";
const fs::path kW4a8Te = "/workspace/Models/community/comfy-org/qwen3.5_9b_w4a8_gptq.safetensors";
const fs::path kControl = "/workspace/Models/community/comfy-org/moss_image_color_depth_control.safetensors";

std::size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return n;
}

std::size_t free_device_bytes() {
  std::size_t free = 0, total = 0;
  MI_CUDA_CHECK(cudaMemGetInfo(&free, &total));
  return free;
}

// The INT4 Turbo DiT (2.8 GiB: quick to load on the dev box) with the SFT bundle, 256^2, 4 steps.
bool have_int4() { return fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") && fs::is_regular_file(kInt4); }
PipelineOptions int4_options() {
  PipelineOptions o;
  o.dit_file = kInt4;
  o.tier = mi::planner::Tier::kTurbo;
  return o;
}
Request small_request() {
  Request r;
  r.model = weights_root() / "sft";
  r.prompt = "a lighthouse on a cliff at dusk, photo";
  r.width = r.height = 256;
  r.steps = 4;
  r.seed = 11;
  return r;
}

void expect_same_image(const Result& a, const Result& b, const char* what) {
  EXPECT_EQ(mismatches(a.latents, b.latents), 0u) << what;
  EXPECT_EQ(a.rgb, b.rgb) << what;
}

// The base / SFT lineage of a DiT, from its input embedder: the official bundles and the quantized files derived from
// the SFT model (which keep the embedder in bf16). CPU only: reads 786 KB of each file.
TEST(DitLineage, OfficialModelsAndDerivedFiles) {
  const fs::path int8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
  int checked = 0;
  for (const char* model : {"base", "sft"}) {
    const fs::path dir = weights_root() / model;
    if (!fs::is_regular_file(dir / "moss.mossidx")) continue;
    EXPECT_EQ(mi::pipeline::dit_lineage(mi::weights::load_index(dir / "moss.mossidx", dir)), model);
    ++checked;
  }
  for (const fs::path& file : {kInt4, int8}) {
    if (!fs::is_regular_file(file)) continue;
    const mi::quant::QuantIndex q =
        mi::quant::to_model_index(mi::quant::open_quant_model(file, mi::quant::ModelKind::kDiT), {.native_int4 = true});
    EXPECT_EQ(mi::pipeline::dit_lineage(q.index), "sft") << file;
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no MOSS-Image DiT present";
  EXPECT_EQ(mi::pipeline::dit_lineage(mi::weights::ModelIndex{}), "");
}

// The DiT load's I/O choice (choose_dit_load_io), every branch: forced either way (and forced direct on files that do
// not gain from it), unsupported files, a parking reload, the page cache holding >= 90 % or less, and, with residency
// unknown, the learner's measured rates or the cold default.
TEST(DitLoadIo, EveryBranch) {
  using mi::pipeline::choose_dit_load_io;
  using mi::pipeline::DitIoInputs;
  DitIoInputs in;
  in.cached_fraction = 0.2;
  EXPECT_TRUE(choose_dit_load_io(in).direct);
  EXPECT_TRUE(choose_dit_load_io(in).note.empty());
  DitIoInputs f = in;
  f.forced_buffered = true;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "buffered reads requested");
  f = in;
  f.forced_direct = true;
  f.support = {false, "x.safetensors is on a RAM file system"};
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "direct I/O requested; x.safetensors is on a RAM file system");
  f.forced_direct = false;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "x.safetensors is on a RAM file system");
  f = in;
  f.parked = true;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_NE(choose_dit_load_io(f).note.find("parking reload"), std::string::npos);
  f = in;
  f.cached_fraction = 0.95;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_NE(choose_dit_load_io(f).note.find("95% of the DiT's files are in the page cache"), std::string::npos);
  f.cached_fraction = 0.89;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  f.buffered_gbps = 9;  // rates count only where residency is unknown
  f.direct_gbps = 3;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  f.cached_fraction = -1;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_NE(choose_dit_load_io(f).note.find("buffered loads measured faster"), std::string::npos);
  f.buffered_gbps = 2;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  f.buffered_gbps = 0;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "page-cache residency unknown: direct, the cold-cache choice");
}

// With the files in the page cache the learned rates decide: both warm rates known (either way, the measured rates in
// the note), one known (the other is tried, unless the load is too small to measure or direct was refused on the
// files), the hysteresis around a tie, a parking reload (direct only where the warm choice is direct), and what stays
// as it was: cold files, unknown residency, the forced modes, unsupported files.
TEST(DitLoadIo, WarmChoiceIsMeasured) {
  using mi::pipeline::choose_dit_load_io;
  using mi::pipeline::DitIoInputs;
  DitIoInputs warm;
  warm.cached_fraction = 0.97;
  warm.bytes = 10'160'000'000;  // the SFT bf16 DiT
  // Both known, RTX 5090: direct beats even a fully cached buffered read.
  DitIoInputs f = warm;
  f.direct_gbps = 9.2;
  f.buffered_warm_gbps = 4.3;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_TRUE(choose_dit_load_io(f).warm);
  EXPECT_EQ(choose_dit_load_io(f).note, "97% of the DiT's files are in the page cache; warm: direct measured 9.2 GB/s vs buffered 4.3 GB/s");
  // Both known, the dev laptop: buffered from RAM wins.
  f.direct_gbps = 6.2;
  f.buffered_warm_gbps = 11.3;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "97% of the DiT's files are in the page cache; warm: buffered measured 11.3 GB/s vs direct 6.2 GB/s");
  f.warm_direct_before = true;  // direct in use, buffered far faster: switched
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  // Only one known: the other mode, once.
  f = warm;
  f.direct_gbps = 9.2;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note,
            "97% of the DiT's files are in the page cache; warm: direct measured 9.2 GB/s, buffered not yet: buffered this once");
  f.buffered_gbps = 4.0;  // the mixed buffered rate (cold loads) is not a warm one
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  f = warm;
  f.buffered_warm_gbps = 4.3;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note,
            "97% of the DiT's files are in the page cache; warm: buffered measured 4.3 GB/s, direct not yet: direct this once");
  // A refused O_DIRECT open on these files earlier: direct is not explored again (said so); a known comparison stands.
  f.direct_refused = true;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "97% of the DiT's files are in the page cache; warm: buffered measured 4.3 GB/s, "
                                        "direct not tried: an O_DIRECT open of these files was refused");
  f.direct_gbps = 9.2;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  // A load too small to measure explores nothing: buffered from RAM, either rate known.
  f = warm;
  f.bytes = mi::pipeline::kDitLoadMeasureBytes - 1;
  f.buffered_warm_gbps = 4.3;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "97% of the DiT's files are in the page cache: buffered reads serve them from RAM");
  f.buffered_warm_gbps = 0;
  f.direct_gbps = 9.2;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "97% of the DiT's files are in the page cache: buffered reads serve them from RAM");
  f.bytes = mi::pipeline::kDitLoadMeasureBytes;
  EXPECT_NE(choose_dit_load_io(f).note.find("buffered this once"), std::string::npos);
  f.buffered_warm_gbps = 4.3;  // both known: the comparison holds for small loads too
  f.bytes = 0;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  // Neither known: buffered, as before the rates (its load measures the warm buffered rate).
  EXPECT_FALSE(choose_dit_load_io(warm).direct);
  // Hysteresis: the mode in use stays until the other is more than kWarmLoadHysteresis faster.
  f = warm;
  f.buffered_warm_gbps = 10.0;
  f.direct_gbps = 11.0;  // 10 % faster
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "97% of the DiT's files are in the page cache; warm: buffered measured 10.0 GB/s vs direct 11.0 GB/s (kept: within 15% of it)");
  f.warm_direct_before = true;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  f.direct_gbps = 9.0;  // direct in use, buffered 11 % faster: kept
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_NE(choose_dit_load_io(f).note.find("warm: direct measured 9.0 GB/s vs buffered 10.0 GB/s (kept"), std::string::npos);
  f.direct_gbps = 8.6;  // buffered 16 % faster: switched
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  f.warm_direct_before = false;
  f.direct_gbps = 11.6;  // direct 16 % faster: switched
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_TRUE(mi::planner::warm_dit_load_direct(11.6, 10.0, false));
  EXPECT_FALSE(mi::planner::warm_dit_load_direct(11.4, 10.0, false));
  EXPECT_TRUE(mi::planner::warm_dit_load_direct(8.8, 10.0, true));
  EXPECT_FALSE(mi::planner::warm_dit_load_direct(8.6, 10.0, true));
  // Parking reloads: buffered unless both rates are known and direct is the warm choice, warm or cold files alike.
  f = warm;
  f.parked = true;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_NE(choose_dit_load_io(f).note.find("parking reload: buffered"), std::string::npos);
  f.direct_gbps = 9.2;  // one known: no exploration on a parking reload
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  f.buffered_warm_gbps = 4.3;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "a parking reload; warm: direct measured 9.2 GB/s vs buffered 4.3 GB/s");
  f.cached_fraction = 0.1;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_FALSE(choose_dit_load_io(f).warm);
  f.buffered_warm_gbps = 11.3;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_NE(choose_dit_load_io(f).note.find("parking reload: buffered"), std::string::npos);
  // Unchanged by the warm rates: cold files read direct, unknown residency compares the mixed rates, the forced
  // modes and unsupported files win.
  f = warm;
  f.direct_gbps = 6.2;
  f.buffered_warm_gbps = 11.3;
  f.cached_fraction = 0.5;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_TRUE(choose_dit_load_io(f).note.empty());
  f.cached_fraction = -1;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "page-cache residency unknown: direct, the cold-cache choice");
  f.cached_fraction = 0.97;
  f.forced_direct = true;
  EXPECT_TRUE(choose_dit_load_io(f).direct);
  EXPECT_TRUE(choose_dit_load_io(f).warm);
  f.forced_direct = false;
  f.buffered_warm_gbps = 4.3;
  f.direct_gbps = 9.2;
  f.forced_buffered = true;
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  f.forced_buffered = false;
  f.support = {false, "x.safetensors is on a RAM file system"};
  EXPECT_FALSE(choose_dit_load_io(f).direct);
  EXPECT_EQ(choose_dit_load_io(f).note, "x.safetensors is on a RAM file system");
}

// The merge memo goes when the plan is over the target its configuration is held to, or over an explicit budget.
TEST(DitLoadIo, MergeMemoDropRule) {
  mi::planner::ExecutionPlan p;
  p.host_target_bytes = std::size_t{8} << 30;
  p.host_held_target_bytes = std::size_t{20} << 30;  // a pinned encoder kept above the target
  p.host_private_bytes = std::size_t{12} << 30;
  EXPECT_FALSE(mi::pipeline::merge_memo_unaffordable(p, 0)) << "under the held target";
  EXPECT_TRUE(mi::pipeline::merge_memo_unaffordable(p, std::size_t{10} << 30)) << "over the explicit budget";
  p.host_private_bytes = std::size_t{21} << 30;
  EXPECT_TRUE(mi::pipeline::merge_memo_unaffordable(p, 0));
}

// A release whose cancel is already set does nothing: no device memory freed, no copy made, the note says why. Nothing
// is loaded (the constructor still creates the CUDA streams, so it is a Gpu suite), so it needs no weights.
TEST(GpuPipelineRelease, CancelledBeforeItStartedReleasesNothing) {
  Pipeline pipe;
  std::atomic<bool> cancel{true};
  for (const bool keep : {false, true}) {
    const mi::pipeline::ReleaseReport rep = pipe.release_device_memory(keep, &cancel);
    EXPECT_EQ(rep.note, "cancelled before it started");
    EXPECT_EQ(rep.vram_released, std::size_t{0});
    EXPECT_FALSE(rep.dit_host_copy);
    EXPECT_FALSE(rep.te_host_copy);
    EXPECT_EQ(rep.host_copy_bytes, std::uint64_t{0});
  }
}

// preload() loads what the plan keeps (the DiT and the VAE decoder here); the next generate() of that request loads
// nothing and gives the bits of a pipeline without the preload. A cancelled preload loads nothing and leaves the
// pipeline usable; one that cannot fit the budgets throws before loading.
TEST(GpuPipelinePreload, FollowingGenerateReusesItBitIdentically) {
  if (!have_int4()) GTEST_SKIP() << "SFT bundle or INT4 DiT not present";
  const Request req = small_request();
  Result plain;
  {
    Pipeline pipe(int4_options());
    plain = pipe.generate(req);
  }
  mi::pipeline::PreloadOptions po;
  po.request = req;
  using mi::pipeline::Phase;
  std::vector<std::string> events;  // "plan" and the phases, in order
  const auto name = [](Phase p) { return "phase" + std::to_string(static_cast<int>(p)); };
  mi::planner::ExecutionPlan seen;
  po.on_plan = [&](const mi::planner::ExecutionPlan& plan) {
    seen = plan;
    events.push_back("plan");
  };
  po.on_phase = [&](Phase p) {
    events.push_back(name(p));
    return true;
  };
  {
    Pipeline pipe(int4_options());
    const Result pre = pipe.preload(po);
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.front(), "plan") << "the plan callback fires before the first load phase";
    // The order of run()'s preload block, from the plan: the text encoder's host copy, the DiT (unless parked for the
    // text encoder), the VAE decoder (unless the DiT is parked for the decode). A text-to-image request has no VAE encoder.
    std::vector<std::string> expected{"plan"};
    if (seen.te.run && seen.te.residency == mi::planner::TeResidency::kHostResident) expected.push_back(name(Phase::kTextEncoder));
    if (!seen.dit.park_for_text_encoder) expected.push_back(name(Phase::kDitLoad));
    if (!seen.dit.park_for_decode) expected.push_back(name(Phase::kDecode));
    EXPECT_EQ(events, expected) << "plan first, then one phase per model load, in load order";
    EXPECT_FALSE(pre.cancelled);
    EXPECT_EQ(pre.memory.dit_loads, 1);
    EXPECT_EQ(pre.dit_load.source, "files");
    EXPECT_GT(pre.timings.dit_load_s, 0.0);
    EXPECT_GT(pre.timings.vae_load_s, 0.0);
    EXPECT_EQ(pre.steps_run, 0);
    EXPECT_TRUE(pre.rgb.empty());
    EXPECT_FALSE(pre.plan.explanation.empty());
    const Result r = pipe.generate(req);
    EXPECT_EQ(r.memory.dit_loads, 0);
    EXPECT_FALSE(r.memory.dit_reloaded);
    EXPECT_EQ(r.memory.arena_creations, 0);
    EXPECT_EQ(r.timings.dit_load_s, 0.0);
    EXPECT_EQ(r.timings.vae_load_s, 0.0);
    EXPECT_TRUE(r.dit_load.source.empty());
    expect_same_image(r, plain, "generate after preload vs without");
    std::printf("[preload] DiT %.2f s + VAE %.2f s (%s, %s); following generate %.2f s, without preload %.2f s (dev)\n",
                pre.timings.dit_load_s, pre.timings.vae_load_s, pre.dit_load.io.c_str(), pre.dit_load.note.c_str(),
                r.timings.total_s, plain.timings.total_s);
  }

  // on_phase returning false cancels the preload at that phase, before its load.
  {
    Pipeline pipe(int4_options());
    mi::pipeline::PreloadOptions stop = po;
    stop.on_plan = nullptr;
    stop.on_phase = [](Phase p) { return p != Phase::kDitLoad; };
    const Result r = pipe.preload(stop);
    EXPECT_TRUE(r.cancelled);
    EXPECT_EQ(r.memory.dit_loads, 0);
  }
  // An earlier or later phase: the text encoder's host copy if the plan has one (nothing after it loads), else the VAE
  // (the DiT before it stays loaded, the VAE is not).
  const bool te_first = std::find(events.begin(), events.end(), name(Phase::kTextEncoder)) != events.end();
  if (te_first || std::find(events.begin(), events.end(), name(Phase::kDecode)) != events.end()) {
    const Phase at = te_first ? Phase::kTextEncoder : Phase::kDecode;
    Pipeline pipe(int4_options());
    mi::pipeline::PreloadOptions stop = po;
    std::vector<std::string> got;
    stop.on_plan = nullptr;
    stop.on_phase = [&](Phase p) {
      got.push_back(name(p));
      return p != at;
    };
    const Result r = pipe.preload(stop);
    EXPECT_TRUE(r.cancelled);
    ASSERT_FALSE(got.empty());
    EXPECT_EQ(got.back(), name(at)) << "nothing fires after the cancelling phase";
    if (te_first) {
      EXPECT_EQ(r.memory.dit_loads, 0);
      EXPECT_EQ(r.timings.dit_load_s, 0.0);
    }
    EXPECT_EQ(r.timings.vae_load_s, 0.0);
  }
  // A preload of a blank prompt plans for a short one; a cancelled preload loads nothing.
  Pipeline fresh(int4_options());
  std::atomic<bool> cancel{true};
  po.request.prompt.clear();
  po.cancel = &cancel;
  const Result cancelled = fresh.preload(po);
  EXPECT_TRUE(cancelled.cancelled);
  EXPECT_EQ(cancelled.memory.dit_loads, 0);
  expect_same_image(fresh.generate(req), plain, "generate after a cancelled preload");

  PipelineOptions tight = int4_options();
  tight.vram_budget = std::size_t{600} << 20;
  Pipeline small(tight);
  po.cancel = nullptr;
  EXPECT_THROW(small.preload(po), mi::planner::BudgetError);
}

// The soft unload frees the device memory; with keep_host_copy the DiT's files stay in pinned RAM and the next load
// re-uploads them from there (reported as "host_copy", faster than reading the files, bit-identical), then frees them.
// Without it nothing stays and the next load reads the files.
TEST(GpuPipelineRelease, SoftUnloadReuploadsFromRamBitIdentically) {
  if (!have_int4()) GTEST_SKIP() << "SFT bundle or INT4 DiT not present";
  Pipeline pipe(int4_options());
  const Request req = small_request();
  const Result first = pipe.generate(req);
  ASSERT_EQ(first.memory.dit_loads, 1);
  const std::size_t free_before = free_device_bytes();
  const mi::pipeline::ReleaseReport kept = pipe.release_device_memory(true);
  const std::size_t free_after = free_device_bytes();
  EXPECT_GT(kept.vram_released, std::size_t{2} << 30);
  EXPECT_GT(free_after, free_before + (std::size_t{2} << 30)) << "the arena went back to the driver";
  if (!kept.dit_host_copy) GTEST_SKIP() << "the DiT copy did not fit this machine's RAM: " << kept.note;
  EXPECT_GE(kept.host_copy_bytes, fs::file_size(kInt4));
  const Result again = pipe.generate(req);
  EXPECT_EQ(again.memory.dit_loads, 1);
  EXPECT_FALSE(again.memory.dit_reloaded) << "an explicit unload is not an unplanned reload";
  EXPECT_EQ(again.dit_load.source, "host_copy");
  EXPECT_EQ(again.memory.host_copy_bytes, 0u) << "the copy is freed once loaded";
  expect_same_image(again, first, "after a soft unload with the host copy");
  const mi::json::Value rep = mi::json::parse("{" + mi::pipeline::report_fields(req, again, "", "") + "}");
  EXPECT_EQ(rep.at("dit_load").at("source").as_string(), "host_copy");

  const mi::pipeline::ReleaseReport dropped = pipe.release_device_memory(false);
  EXPECT_FALSE(dropped.dit_host_copy);
  EXPECT_EQ(dropped.host_copy_bytes, 0u);
  const Result cold = pipe.generate(req);
  EXPECT_EQ(cold.dit_load.source, "files");
  expect_same_image(cold, first, "after a soft unload without the host copy");
  EXPECT_LT(again.dit_load.seconds, cold.dit_load.seconds) << "RAM re-upload vs reading the files";
  std::printf("[release] copy kept in %.2f s (%.2f GB); reload from RAM %.2f s (%.1f GB/s) vs from the files %.2f s (%.1f GB/s, %s) (dev)\n",
              kept.host_copy_s, static_cast<double>(kept.host_copy_bytes) / 1e9, again.dit_load.seconds, again.dit_load.gbps(),
              cold.dit_load.seconds, cold.dit_load.gbps(), cold.dit_load.io.c_str());
}

// The soft unload's copy is optional: a failed pinned allocation (the test seam), a cancel or a host budget that cannot
// hold it leaves no copy, says why, and the release itself completes; the next load reads the files, bit-identically.
TEST(GpuPipelineRelease, CopyFailureCancelAndRefusalStillRelease) {
  if (!have_int4()) GTEST_SKIP() << "SFT bundle or INT4 DiT not present";
  struct Clear {
    ~Clear() { mi::weights::testing::clear_fail_pinned_alloc(); }
  } clear;
  const Request req = small_request();
  Result first;
  {
    Pipeline pipe(int4_options());
    first = pipe.generate(req);
    mi::weights::testing::fail_pinned_alloc_after(0);
    const mi::pipeline::ReleaseReport failed = pipe.release_device_memory(true);
    mi::weights::testing::clear_fail_pinned_alloc();
    EXPECT_FALSE(failed.dit_host_copy);
    EXPECT_GT(failed.vram_released, std::size_t{2} << 30) << "the device memory went first";
    EXPECT_NE(failed.note.find("not kept in RAM"), std::string::npos) << failed.note;
    const Result again = pipe.generate(req);
    EXPECT_EQ(again.dit_load.source, "files");
    expect_same_image(again, first, "after a failed host copy");
    std::atomic<bool> cancel{true};
    const mi::pipeline::ReleaseReport cancelled = pipe.release_device_memory(true, &cancel);
    EXPECT_FALSE(cancelled.dit_host_copy);
    EXPECT_EQ(cancelled.note, "cancelled before it started");
    EXPECT_EQ(cancelled.vram_released, std::size_t{0}) << "a cancel set on entry releases nothing";
    EXPECT_EQ(pipe.generate(req).memory.dit_loads, 0) << "the DiT is still resident";
  }
  // A host budget that holds the generation but not a 2.8 GiB copy.
  PipelineOptions tight = int4_options();
  tight.host_ram_budget = mi::planner::process_host_memory().private_bytes + (std::size_t{2} << 30) + (std::size_t{512} << 20);
  Pipeline pipe(tight);
  try {
    expect_same_image(pipe.generate(req), first, "within a host budget");
  } catch (const mi::planner::BudgetError& e) {
    GTEST_SKIP() << "the generation itself does not fit the test's host budget here: " << e.what();
  }
  const mi::pipeline::ReleaseReport refused = pipe.release_device_memory(true);
  EXPECT_FALSE(refused.dit_host_copy);
  EXPECT_NE(refused.note.find("host budget"), std::string::npos) << refused.note;
}

// A cancel that arrives while the DiT's files are being copied to RAM: the release still completes (the device memory
// is freed), the partial copy is dropped with a note, and the next load reads the files, bit-identical.
TEST(GpuPipelineRelease, CancelMidHostCopyKeepsNoCopyAndStillReleases) {
  if (!have_int4()) GTEST_SKIP() << "SFT bundle or INT4 DiT not present";
  struct Clear {
    ~Clear() { mi::weights::testing::cancel_after_first_host_chunk(nullptr); }
  } clear;
  const Request req = small_request();
  Pipeline pipe(int4_options());
  const Result first = pipe.generate(req);
  std::atomic<bool> cancel{false};
  mi::weights::testing::cancel_after_first_host_chunk(&cancel);
  const mi::pipeline::ReleaseReport rep = pipe.release_device_memory(true, &cancel);
  mi::weights::testing::cancel_after_first_host_chunk(nullptr);
  if (!cancel.load()) GTEST_SKIP() << "the DiT copy did not start (no room for it here): " << rep.note;
  EXPECT_GT(rep.vram_released, std::size_t{2} << 30);
  EXPECT_FALSE(rep.dit_host_copy);
  EXPECT_EQ(rep.host_copy_bytes, std::uint64_t{0});
  EXPECT_NE(rep.note.find("cancelled"), std::string::npos) << rep.note;
  const Result again = pipe.generate(req);
  EXPECT_EQ(again.dit_load.source, "files");
  expect_same_image(again, first, "after a release cancelled mid host copy");
}

// A pinned text-encoder copy that cannot be allocated (the test seam): the encoder streams from its file instead, the
// bits are the streamed encoder's, the plan and the report say so, and pinning stays off for that encoder until a full
// soft unload.
TEST(GpuPipelinePinFallback, PinnedTextEncoderFallsBackToStreaming) {
  if (!have_int4() || !fs::is_regular_file(kW4a8Te)) GTEST_SKIP() << "SFT bundle, INT4 DiT or W4A8 text encoder not present";
  struct Clear {
    ~Clear() { mi::weights::testing::clear_fail_pinned_alloc(); }
  } clear;
  Request req = small_request();
  req.decode = false;
  PipelineOptions o = int4_options();
  o.text_encoder_file = kW4a8Te;
  o.embedding_cache = false;
  Result streamed;
  {
    PipelineOptions so = o;
    so.te_host_pin = mi::planner::TeHostPin::kOff;
    Pipeline pipe(so);
    streamed = pipe.generate(req);
  }
  o.te_host_pin = mi::planner::TeHostPin::kOn;
  Pipeline pipe(o);
  mi::weights::testing::fail_pinned_alloc_after(0);
  const Result fell = pipe.generate(req);
  mi::weights::testing::clear_fail_pinned_alloc();
  EXPECT_NE(fell.te_note.find("could not be allocated"), std::string::npos) << fell.te_note;
  EXPECT_EQ(fell.plan.te.residency, mi::planner::TeResidency::kStreamed);
  EXPECT_FALSE(fell.plan.te.pinned);
  EXPECT_NE(fell.plan.explanation.find("could not be allocated"), std::string::npos);
  EXPECT_EQ(mismatches(fell.latents, streamed.latents), 0u);
  const Result sticky = pipe.generate(req);
  EXPECT_FALSE(sticky.plan.te.pinned) << "no pinning after a failed allocation";
  EXPECT_NE(sticky.te_note.find("pinning disabled"), std::string::npos) << sticky.te_note;
  pipe.release_device_memory(false);
  const Result again = pipe.generate(req);
  EXPECT_TRUE(again.plan.te.pinned) << "a full soft unload lets pinning try again";
  EXPECT_EQ(mismatches(again.latents, streamed.latents), 0u);
}

// The DiT load's I/O mode is reported: direct where the file system gains from it, else buffered with the reason;
// buffered on request. The bits never depend on it.
TEST(GpuPipelineDirectIo, LoadModeIsReportedAndChangesNoBit) {
  if (!have_int4()) GTEST_SKIP() << "SFT bundle or INT4 DiT not present";
  const Request req = small_request();
  Result dflt, buffered;
  {
    Pipeline pipe(int4_options());
    dflt = pipe.generate(req);
  }
  {
    PipelineOptions o = int4_options();
    o.dit_load_direct_io = false;
    Pipeline pipe(o);
    buffered = pipe.generate(req);
  }
  EXPECT_EQ(dflt.dit_load.source, "files");
  EXPECT_TRUE(dflt.dit_load.io == "direct" || dflt.dit_load.io == "buffered") << dflt.dit_load.io;
  if (dflt.dit_load.io == "buffered") {
    EXPECT_FALSE(dflt.dit_load.note.empty());
  }
  EXPECT_EQ(buffered.dit_load.io, "buffered");
  EXPECT_EQ(buffered.dit_load.note, "buffered reads requested");
  EXPECT_EQ(dflt.dit_load.bytes, buffered.dit_load.bytes);
  expect_same_image(dflt, buffered, "direct-I/O default vs buffered");
  std::printf("[direct io] default: %s (%s) %.2f s; buffered %.2f s (dev)\n", dflt.dit_load.io.c_str(), dflt.dit_load.note.c_str(),
              dflt.dit_load.seconds, buffered.dit_load.seconds);
}

// A host-resident text encoder (here the W4A8 file with te_host_pin on, so the plan pins it) is loaded by preload()
// and kept across requests: the next cache miss neither reads nor allocates the copy again, and the bits are those of
// an encoder streamed from its file.
TEST(GpuPipelinePreload, HostResidentTextEncoderIsKept) {
  if (!have_int4() || !fs::is_regular_file(kW4a8Te)) GTEST_SKIP() << "SFT bundle, INT4 DiT or W4A8 text encoder not present";
  const std::size_t te_bytes = fs::file_size(kW4a8Te);
  if (mi::planner::host_available_bytes() < te_bytes + (std::size_t{4} << 30)) GTEST_SKIP() << "not enough RAM for the copy";
  Request req = small_request();
  req.decode = false;
  Result streamed;
  {
    PipelineOptions o = int4_options();
    o.text_encoder_file = kW4a8Te;
    o.embedding_cache = false;
    o.mode = mi::planner::Mode::kMinMemory;  // never host-resident
    Pipeline pipe(o);
    streamed = pipe.generate(req);
    ASSERT_EQ(streamed.plan.te.residency, mi::planner::TeResidency::kStreamed);
  }
  PipelineOptions o = int4_options();
  o.text_encoder_file = kW4a8Te;
  o.embedding_cache = false;  // every request is a cache miss
  o.te_host_pin = mi::planner::TeHostPin::kOn;
  Pipeline pipe(o);
  mi::pipeline::PreloadOptions po;
  po.request = req;
  const Result pre = pipe.preload(po);
  ASSERT_EQ(pre.plan.te.residency, mi::planner::TeResidency::kHostResident) << pre.plan.explanation;
  EXPECT_TRUE(pre.plan.te.pinned);
  EXPECT_GE(pre.memory.host_copy_bytes, te_bytes / 2);
  const Result r1 = pipe.generate(req);
  EXPECT_EQ(r1.memory.dit_loads, 0);
  EXPECT_EQ(r1.memory.host_copy_bytes, pre.memory.host_copy_bytes);
  EXPECT_EQ(mismatches(r1.latents, streamed.latents), 0u) << "host-resident vs streamed encoder";
  // A soft unload keeps the copy; the next encoder adopts it.
  const mi::pipeline::ReleaseReport rel = pipe.release_device_memory(true);
  EXPECT_TRUE(rel.te_host_copy);
  const Result r2 = pipe.generate(req);
  EXPECT_EQ(mismatches(r2.latents, streamed.latents), 0u);
  EXPECT_GE(r2.memory.host_copy_bytes, pre.memory.host_copy_bytes) << "the encoder's copy was adopted, not read again";
  std::printf("[te host] preload %.2f s; text encode %.2f s (host-resident) vs %.2f s (streamed) (dev)\n",
              pre.timings.te_weight_load_s, r1.timings.text_encode_s, streamed.timings.text_encode_s);
}

// Switching a control adapter on in a running pipeline: its LoRA part (reading the file, the side branches) and the
// merges into the loaded DiT are timed (lora_load_s, lora_merge_s), and almost nothing of the call is left outside a
// timing field; switching back off times the un-merge, and switching on again stores the merged bits the DiT kept
// (MossDit::rebase_merges' memo) instead of merging again. The bits match a fresh pipeline's.
TEST(GpuPipelineControlToggle, SwitchOnCostIsTimed) {
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(kControl)) {
    GTEST_SKIP() << "SFT bundle or control adapter not present";
  }
  auto depth = std::make_shared<mi::pipeline::Image>();
  depth->width = depth->height = 256;
  depth->channels = 1;
  depth->pixels.assign(256 * 256, 0);
  for (int y = 96; y < 160; ++y) {
    for (int x = 96; x < 160; ++x) depth->pixels[static_cast<std::size_t>(y) * 256 + static_cast<std::size_t>(x)] = 220;
  }
  Request plain = small_request();
  plain.decode = false;
  Request ctl = plain;
  ctl.control.adapter = kControl;
  ctl.control.images.push_back({"depth", depth, 1.0, 0.0, 1.0});
  const auto unaccounted = [](const Result& r) {
    const mi::pipeline::Timings& t = r.timings;
    return t.total_s - (t.index_load_s + t.tokenize_s + t.plan_s + t.text_encode_s + t.dit_load_s + t.lora_load_s + t.lora_merge_s +
                        t.refine_s + t.denoise_s + t.vae_load_s + t.decode_s + t.vae_encoder_load_s + t.encode_s + t.image_ops_s);
  };
  Pipeline pipe;
  const Result p0 = pipe.generate(plain);
  const Result c1 = pipe.generate(ctl);
  const Result c2 = pipe.generate(ctl);
  const Result p1 = pipe.generate(plain);
  const Result c3 = pipe.generate(ctl);
  EXPECT_EQ(c1.memory.dit_loads, 0);
  EXPECT_EQ(c3.memory.dit_loads, 0);
  EXPECT_LT(c3.timings.lora_merge_s, 0.5 * c1.timings.lora_merge_s) << "the kept merged bits, not a new merge";
  EXPECT_EQ(mismatches(c3.latents, c1.latents), 0u) << "switched on again from the kept bits";
  EXPECT_GT(c1.timings.lora_load_s, 0.0);
  EXPECT_GT(c1.timings.lora_merge_s, 0.0) << "the adapter's text-refiner LoRA is merged into the loaded DiT";
  EXPECT_EQ(c2.timings.lora_load_s, 0.0) << "unchanged stack: reused";
  EXPECT_EQ(c2.timings.lora_merge_s, 0.0);
  EXPECT_GT(p1.timings.lora_merge_s, 0.0) << "switching off restores the pristine weights";
  EXPECT_GT(p1.memory.merge_memo_bytes, 0u) << "the merged bits are kept for switching back on";
  EXPECT_LE(p1.memory.merge_memo_bytes, mi::moss_dit::MossDit::kMaxMergeMemoBytes);
  for (const Result* r : {&p0, &c1, &c2, &p1, &c3}) EXPECT_LT(unaccounted(*r), 0.15) << "time outside every timing field";
  EXPECT_EQ(mismatches(p1.latents, p0.latents), 0u);
  EXPECT_EQ(mismatches(c2.latents, c1.latents), 0u);
  Pipeline fresh;
  EXPECT_EQ(mismatches(fresh.generate(ctl).latents, c1.latents), 0u) << "switched on in place vs a fresh pipeline";
  // Over the memo's cap (forced down to 1 MiB) nothing is kept: switching on again merges again, the same bits.
  mi::moss_dit::MossDit::set_merge_memo_cap_for_tests(std::size_t{1} << 20);
  struct Restore {
    ~Restore() { mi::moss_dit::MossDit::set_merge_memo_cap_for_tests(mi::moss_dit::MossDit::kMaxMergeMemoBytes); }
  } restore;
  Pipeline capped;
  const Result k0 = capped.generate(plain);
  const Result k1 = capped.generate(ctl);
  const Result k2 = capped.generate(plain);
  EXPECT_EQ(k2.memory.merge_memo_bytes, 0u) << "over the cap: not stored";
  const Result k3 = capped.generate(ctl);
  EXPECT_GT(k3.timings.lora_merge_s, 0.5 * k1.timings.lora_merge_s) << "merged again";
  EXPECT_EQ(mismatches(k3.latents, c1.latents), 0u);
  (void)k0;
  std::printf("[control toggle] on: LoRA load %.3f s, merge %.3f s, unaccounted %.3f s; off: merge %.3f s, unaccounted %.3f s; "
              "on again: LoRA load %.3f s, merge %.3f s, unaccounted %.3f s (dev)\n",
              c1.timings.lora_load_s, c1.timings.lora_merge_s, unaccounted(c1), p1.timings.lora_merge_s, unaccounted(p1),
              c3.timings.lora_load_s, c3.timings.lora_merge_s, unaccounted(c3));
}

}  // namespace
