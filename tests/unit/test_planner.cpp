// Planner (src/planner/) on synthetic budgets: pure logic, no GPU. Model shapes are the real MOSS-Image group sizes
// (moss-infer inspect); the workspace queries are smooth stand-ins for the models' exact *_workspace_bytes() with the
// measured magnitudes (DiT activations 1.19 GiB at 1456^2 + 4096 tokens, fp32 decode ~1.5 KB per pixel).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "planner/host_memory.h"
#include "planner/planner.h"
#include "util/json.h"

namespace {

using namespace mi::planner;

constexpr double kGiBd = 1024.0 * 1024.0 * 1024.0;
std::size_t gib(double g) { return static_cast<std::size_t>(g * kGiBd); }

ModelShapes moss_shapes() {
  ModelShapes s;
  s.te_layers = 32;
  s.te_layer_bytes = 436'814'208;
  s.te_stream_bytes = 15'873'369'088 - 2'034'237'440;
  s.te_embed_bytes = 2'034'237'440;
  s.dit_blocks = 42;
  s.dit_block_bytes = 226'505'216;
  s.dit_bytes = 10'164'617'280 + (std::size_t{64} << 20);  // + GEMM workspace
  s.dit_dim = 3072;
  s.vae_bytes = 111'789'972;
  return s;
}

// Untiled fp32 decode: two 192-channel ping-pong buffers at full resolution + latents + RGB, ~1.5 KB per pixel.
std::size_t decode_bytes(int h, int w) { return static_cast<std::size_t>(1540.0 * h * w); }

// Stand-in for Flux2VaeDecoder::plan_tiled: full-width row bands with a 228-row halo; recomputation grows as the
// bands shrink. Budget 0: the smallest schedule.
std::optional<TiledDecode> tiled_decode(int h, int w, std::size_t budget) {
  for (const int band : {1024, 512, 256, 128, 64, 32, 16, 8}) {
    TiledDecode t;
    t.tile = band;
    t.bytes = static_cast<std::size_t>(1540.0 * w * (std::min(h, band) + 228)) + (std::size_t{16} << 20);
    t.cost = 1.3 + 228.0 / band;
    if (budget == 0 ? band == 8 : t.bytes <= budget) return t;
  }
  return std::nullopt;
}

struct Scenario {
  int size = 1456;
  std::vector<std::int64_t> lengths = {4096, 12};
  double vram_free_gib = 12.0;
  double host_available_gib = 20.0;
  double host_now_gib = 1.0;
  Mode mode = Mode::kBalanced;
  bool tiles = true;
  bool streaming = false;
  bool parking = true;
  bool te_cached = false;
  bool host_pinned = false;     // the DiT streamer can hold a pinned host copy
  bool dit_weights_cached = true;  // the page cache holds the DiT's weight files
  HardwareProfile hardware;

  PlanInputs inputs() const {
    PlanInputs in;
    in.shapes = moss_shapes();
    in.request.height = in.request.width = size;
    in.request.text_lengths = lengths;
    in.request.steps = 50;
    in.request.rows_per_step = 2;
    in.request.te_cached = te_cached;
    in.request.dit_weights_cached = dit_weights_cached;
    in.hardware = hardware;
    const ModelShapes s = in.shapes;
    std::int64_t tokens = 0;
    for (const auto l : lengths) tokens += l;
    const std::size_t te_act = static_cast<std::size_t>(tokens) * 80 * 1024;
    in.workspaces.text_encoder = [te_act, s](TeResidency r, int buffers) {
      return te_act + (r == TeResidency::kVramResident ? 0 : static_cast<std::size_t>(buffers) * s.te_layer_bytes);
    };
    const std::int64_t hw = static_cast<std::int64_t>(size / 16) * (size / 16);
    in.workspaces.denoise = static_cast<std::size_t>(hw + tokens) * 98 * 1024;
    const int px = size;
    in.workspaces.decode = decode_bytes(px, px);
    if (tiles) in.workspaces.decode_tiled = [px](std::size_t budget) { return tiled_decode(px, px, budget); };
    in.workspaces.host_buffers = std::size_t{32} << 20;
    in.caps.te_max_buffers = 2;
    in.caps.te_ring_bytes = std::size_t{256} << 20;
    in.caps.dit_parking = parking;
    in.caps.dit_streaming = streaming;
    in.caps.dit_ring_bytes = std::size_t{256} << 20;
    in.caps.dit_host_pinned = host_pinned;
    in.budgets.vram_free = gib(vram_free_gib);
    in.budgets.host_available = gib(host_available_gib);
    in.budgets.host_private_now = gib(host_now_gib);
    in.mode = mode;
    return in;
  }
};

void expect_consistent(const ExecutionPlan& p) {
  EXPECT_LE(p.arena_bytes, p.arena_cap_bytes);
  EXPECT_GE(p.arena_bytes, std::max({p.peaks.text_encoder, p.peaks.denoise, p.peaks.decode}) + kArenaSlack);
  EXPECT_LE(p.host_private_bytes, p.host_limit_bytes);
  EXPECT_FALSE(p.explanation.empty());
  if (p.dit.residency == DitResidency::kResident) {
    EXPECT_FALSE(p.dit.park_for_text_encoder);
    EXPECT_FALSE(p.dit.park_for_decode);
  }
}

TEST(Planner, ModeNamesRoundTrip) {
  for (const Mode m : {Mode::kMinMemory, Mode::kBalanced, Mode::kMaxSpeed}) EXPECT_EQ(parse_mode(mode_name(m)), m);
  EXPECT_THROW(parse_mode("fast"), std::invalid_argument);
  EXPECT_THROW(parse_mode(""), std::invalid_argument);
  EXPECT_STREQ(residency_name(DitResidency::kParked), "parked");
  EXPECT_STREQ(residency_name(TeResidency::kVramResident), "vram_resident");
}

TEST(Planner, ArenaCapFromBudgets) {
  Budgets b;
  b.vram_free = gib(12);
  EXPECT_EQ(arena_cap(b), gib(12) - b.vram_reserve);
  b.vram_budget = gib(7.5);  // NVML-visible: context and reserve come off
  EXPECT_EQ(arena_cap(b), gib(7.5) - b.context_bytes - b.vram_reserve);
  b.vram_budget = gib(40);  // never above what is free
  EXPECT_EQ(arena_cap(b), gib(12) - b.vram_reserve);
  b.vram_free = 0;
  EXPECT_EQ(arena_cap(b), 0u);
}

// The dev box (about 12 GiB free) on the AC7 scenario: the DiT stays resident and the decode is tiled instead of
// parking it; the TE streams with prefetch overlap.
TEST(Planner, DevBoxAc7KeepsDitResidentWithTiledDecode) {
  const ExecutionPlan p = plan(Scenario{}.inputs());
  expect_consistent(p);
  EXPECT_EQ(p.mode, Mode::kBalanced);
  EXPECT_EQ(p.dit.residency, DitResidency::kResident);
  EXPECT_EQ(p.vae.kind, VaeDecodeKind::kTiled);
  EXPECT_EQ(p.vae.tile, 512);  // the largest band that fits next to the resident DiT
  EXPECT_FALSE(p.vae.smallest);
  EXPECT_TRUE(p.te.run);
  EXPECT_EQ(p.te.residency, TeResidency::kStreamed);
  EXPECT_EQ(p.te.buffers, 2);
  EXPECT_LE(p.device_peak_bytes, gib(15.0));  // AC7 bound
  EXPECT_LE(p.host_private_bytes, gib(8.0));  // AC7 host bound
  EXPECT_NE(p.explanation.find("Untiled decode would need"), std::string::npos) << p.explanation;
  EXPECT_NE(p.explanation.find("never parked"), std::string::npos) << p.explanation;
}

// Prefetch overlap pays off for short prompts too (the copies dominate; only one layer's compute is exposed).
TEST(Planner, ShortPromptsStreamWithPrefetch) {
  Scenario s;
  s.size = 512;
  s.lengths = {16, 10};
  const ExecutionPlan p = plan(s.inputs());
  EXPECT_EQ(p.te.buffers, 2);
  EXPECT_EQ(p.vae.kind, VaeDecodeKind::kUntiled);
}

TEST(Planner, UntiledWhenItFits) {
  Scenario s;
  s.vram_free_gib = 15.6;  // an idle 16 GiB card
  const ExecutionPlan p = plan(s.inputs());
  expect_consistent(p);
  EXPECT_EQ(p.dit.residency, DitResidency::kResident);
  EXPECT_EQ(p.vae.kind, VaeDecodeKind::kUntiled);
  s.vram_free_gib = 23.5;
  s.size = 2048;  // AC16 on a 24 GiB card
  const ExecutionPlan q = plan(s.inputs());
  EXPECT_EQ(q.vae.kind, VaeDecodeKind::kUntiled);
}

TEST(Planner, Ac16At2048TilesOnSixteenGib) {
  Scenario s;
  s.size = 2048;
  s.vram_free_gib = 15.6;
  const ExecutionPlan p = plan(s.inputs());
  expect_consistent(p);
  EXPECT_EQ(p.dit.residency, DitResidency::kResident);
  EXPECT_EQ(p.vae.kind, VaeDecodeKind::kTiled);
  EXPECT_LE(p.device_peak_bytes, gib(16));
}

TEST(Planner, ParkingIsTheLastResort) {
  Scenario s;
  s.tiles = false;  // no exact tiled decode in this build
  const ExecutionPlan p = plan(s.inputs());
  expect_consistent(p);
  EXPECT_EQ(p.dit.residency, DitResidency::kParked);
  EXPECT_TRUE(p.dit.park_for_decode);
  EXPECT_FALSE(p.dit.park_for_text_encoder);  // the TE fits next to the DiT
  EXPECT_GT(p.time.dit_load_s, 0);
  EXPECT_NE(p.explanation.find("Parking is the fallback"), std::string::npos) << p.explanation;

  s.parking = false;
  EXPECT_THROW(plan(s.inputs()), BudgetError);
}

TEST(Planner, EightGibFailsCleanlyWithoutStreaming) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 7.6;
  try {
    plan(s.inputs());
    FAIL() << "expected BudgetError";
  } catch (const BudgetError& e) {
    EXPECT_GT(e.required_arena, e.arena_cap);
    EXPECT_EQ(e.arena_cap, gib(7.6) - (std::size_t{384} << 20));
    // The leanest plan parks the DiT for both other phases, so it needs the DiT plus the denoise activations.
    EXPECT_GE(e.required_arena, moss_shapes().dit_bytes);
    const std::string msg = e.what();
    EXPECT_NE(msg.find("in any mode"), std::string::npos) << msg;
    EXPECT_NE(msg.find("GiB"), std::string::npos) << msg;
  }
}

TEST(Planner, StreamedDitInEightGib) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 7.6;
  s.streaming = true;
  const ExecutionPlan bal = plan(s.inputs());
  expect_consistent(bal);
  EXPECT_EQ(bal.dit.residency, DitResidency::kStreamed);
  EXPECT_GT(bal.dit.resident_blocks, 0);  // balanced keeps as many blocks as fit
  EXPECT_LT(bal.dit.resident_blocks, 42);
  s.mode = Mode::kMinMemory;
  const ExecutionPlan min = plan(s.inputs());
  expect_consistent(min);
  EXPECT_EQ(min.dit.residency, DitResidency::kStreamed);
  EXPECT_EQ(min.dit.resident_blocks, 0);
  EXPECT_LT(min.arena_bytes, bal.arena_bytes);
  EXPECT_GE(min.time.total_s(), bal.time.total_s());
}

TEST(Planner, ExplicitVramBudgetIsAProcessBudget) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 15.6;
  s.streaming = true;
  PlanInputs in = s.inputs();
  in.budgets.vram_budget = gib(7.5);  // AC8: --vram-budget 7.5 on the 16 GiB card
  const ExecutionPlan p = plan(in);
  expect_consistent(p);
  EXPECT_LE(p.device_peak_bytes + in.budgets.vram_reserve, gib(7.5));
  EXPECT_EQ(p.dit.residency, DitResidency::kStreamed);
}

TEST(Planner, MinMemoryNeverExceedsBalanced) {
  for (const double free : {12.0, 15.6, 23.5}) {
    for (const int size : {1024, 1456, 2048}) {
      Scenario s;
      s.size = size;
      s.vram_free_gib = free;
      s.streaming = true;
      const ExecutionPlan bal = plan(s.inputs());
      s.mode = Mode::kMinMemory;
      const ExecutionPlan min = plan(s.inputs());
      s.mode = Mode::kMaxSpeed;
      const ExecutionPlan max = plan(s.inputs());
      expect_consistent(min);
      expect_consistent(max);
      EXPECT_LE(min.arena_bytes, bal.arena_bytes) << free << " GiB, " << size;
      EXPECT_LE(max.time.total_s(), bal.time.total_s() + 1e-9) << free << " GiB, " << size;
      EXPECT_LE(min.host_private_bytes, gib(4.0));
    }
  }
}

TEST(Planner, EmbeddingCacheHitSkipsTextEncoder) {
  Scenario s;
  s.te_cached = true;
  const ExecutionPlan p = plan(s.inputs());
  expect_consistent(p);
  EXPECT_FALSE(p.te.run);
  EXPECT_EQ(p.peaks.text_encoder, 0u);
  EXPECT_EQ(p.time.text_encoder_s, 0.0);
  EXPECT_NE(p.explanation.find("embedding cache hit"), std::string::npos) << p.explanation;
  EXPECT_TRUE(p.cache.keep_negative);
}

TEST(Planner, TextEncoderResidencyFollowsTheMode) {
  // A 48 GiB card with plenty of RAM: max_speed keeps the TE in VRAM when the build supports it.
  Scenario s;
  s.vram_free_gib = 47.0;
  s.host_available_gib = 60.0;
  s.mode = Mode::kMaxSpeed;
  PlanInputs in = s.inputs();
  in.caps.te_vram_resident = true;
  in.caps.te_host_resident = true;
  ExecutionPlan p = plan(in);
  expect_consistent(p);
  EXPECT_EQ(p.te.residency, TeResidency::kVramResident);
  EXPECT_GE(p.peaks.denoise, moss_shapes().te_stream_bytes + moss_shapes().dit_bytes);

  // 24 GiB card, cold page cache, 64 GiB RAM: TE weights host-resident in max_speed only; balanced stays within
  // its 8 GiB host target and streams from the files.
  s.vram_free_gib = 23.0;
  in = s.inputs();
  in.caps.te_vram_resident = true;
  in.caps.te_host_resident = true;
  in.request.weights_cached = false;
  p = plan(in);
  expect_consistent(p);
  EXPECT_EQ(p.te.residency, TeResidency::kHostResident);
  EXPECT_GT(p.host_private_bytes, moss_shapes().te_stream_bytes);
  in.mode = Mode::kBalanced;
  p = plan(in);
  EXPECT_EQ(p.te.residency, TeResidency::kStreamed);
  EXPECT_LE(p.host_private_bytes, gib(8.0));
}

// The pipeline learns weights_cached from its last file-streamed text-encoder pass; a cold measurement makes
// max_speed hold the TE weights in host RAM (24 GiB card, 64 GiB RAM), a warm one keeps streaming from the files.
TEST(Planner, LearnsWeightsCachedFromTheTextEncoderStream) {
  const HardwareProfile hw;  // host 8 GB/s, h2d 12 GB/s: cached rate 8 GB/s
  const std::uint64_t bytes = moss_shapes().te_stream_bytes;
  EXPECT_TRUE(weights_cached_from_stream(bytes, 1.8, 0.05, hw));   // 7.7 GB/s, loads hidden
  EXPECT_FALSE(weights_cached_from_stream(bytes, 7.0, 5.0, hw));   // 2 GB/s and the compute stream waited
  EXPECT_TRUE(weights_cached_from_stream(bytes, 7.0, 0.3, hw));    // slow but hidden: compute-bound pass
  EXPECT_TRUE(weights_cached_from_stream(bytes, 1.0, 0.9, hw));    // stalled at the page-cache rate
  EXPECT_TRUE(weights_cached_from_stream(0, 0.0, 0.0, hw));        // nothing measured

  Scenario s;
  s.vram_free_gib = 23.0;
  s.host_available_gib = 60.0;
  s.mode = Mode::kMaxSpeed;
  PlanInputs in = s.inputs();
  in.caps.te_host_resident = true;
  in.request.weights_cached = weights_cached_from_stream(bytes, 7.0, 5.0, hw);
  ExecutionPlan p = plan(in);
  expect_consistent(p);
  EXPECT_EQ(p.te.residency, TeResidency::kHostResident);
  in.request.weights_cached = weights_cached_from_stream(bytes, 1.8, 0.05, hw);
  p = plan(in);
  expect_consistent(p);
  EXPECT_EQ(p.te.residency, TeResidency::kStreamed);
  // Not within balanced's 8 GiB host target even when cold.
  in.mode = Mode::kBalanced;
  in.request.weights_cached = false;
  EXPECT_EQ(plan(in).te.residency, TeResidency::kStreamed);
}

// The text-encoder estimate covers only the sequences the encoder runs on (embedding-cache misses).
TEST(Planner, TextEncoderTimeCountsCacheMissesOnly) {
  Scenario s;
  PlanInputs in = s.inputs();  // 4096 + 12 tokens
  const ExecutionPlan both = plan(in);
  in.request.te_lengths = {12};  // the long prompt hit the cache, only the negative is encoded
  const ExecutionPlan one = plan(in);
  EXPECT_GT(both.time.text_encoder_s, 0.0);
  EXPECT_LT(one.time.text_encoder_s, both.time.text_encoder_s);
  EXPECT_EQ(one.time.denoise_s, both.time.denoise_s);  // the denoise still runs every row
}

// A VAE decoder resident from an earlier request counts in every phase of a latents-only request: the arena the plan
// asks for is what the pipeline actually uses with the decoder still in it.
TEST(Planner, ResidentVaeCountsWithoutDecode) {
  Scenario s;
  PlanInputs in = s.inputs();
  in.request.decode = false;
  const ExecutionPlan bare = plan(in);
  EXPECT_FALSE(bare.vae.run);
  in.request.vae_resident = true;
  const ExecutionPlan held = plan(in);
  expect_consistent(held);
  EXPECT_FALSE(held.vae.run);
  EXPECT_EQ(held.peaks.denoise, bare.peaks.denoise + moss_shapes().vae_bytes);
  EXPECT_EQ(held.peaks.text_encoder, bare.peaks.text_encoder + moss_shapes().vae_bytes);
  EXPECT_EQ(held.arena_bytes, bare.arena_bytes + moss_shapes().vae_bytes);
  // A decoding request counts it either way.
  in.request.decode = true;
  const ExecutionPlan dec = plan(in);
  in.request.vae_resident = false;
  EXPECT_EQ(plan(in).arena_bytes, dec.arena_bytes);
}

// A process that already holds an arena and host memory: the held arena counts as free device memory (the pipeline
// recreates it), and the host baseline shrinks what the embedding cache may take within the mode's target.
TEST(Planner, ExistingArenaAndHostMemory) {
  Scenario s;
  s.vram_free_gib = 1.5;  // live free memory alone: not even the DiT fits
  EXPECT_THROW(plan(s.inputs()), BudgetError);
  PlanInputs in = s.inputs();
  in.budgets.vram_free += gib(10.5);  // + the 10.5 GiB arena this process holds
  ExecutionPlan p = plan(in);
  expect_consistent(p);
  EXPECT_EQ(p.dit.residency, DitResidency::kResident);
  EXPECT_LE(p.arena_bytes, gib(12.0) - in.budgets.vram_reserve);

  // Balanced (8 GiB target) with 7.5 GiB already private: the cache gets what is left, the floor at least.
  in.budgets.host_private_now = gib(7.5);
  p = plan(in);
  expect_consistent(p);
  EXPECT_GE(p.host_private_bytes, gib(7.5));
  EXPECT_LT(p.cache.host_bytes, std::size_t{1} << 30);
  EXPECT_GE(p.cache.host_bytes, std::size_t{64} << 20);
  in.budgets.host_private_now = gib(1.0);
  EXPECT_EQ(plan(in).cache.host_bytes, std::size_t{1} << 30);
}

TEST(Planner, HostBudgets) {
  Scenario s;
  s.mode = Mode::kMinMemory;
  s.host_available_gib = 14.0;  // a 16 GiB PC
  ExecutionPlan p = plan(s.inputs());
  EXPECT_EQ(p.host_target_bytes, gib(4.0));
  EXPECT_LE(p.cache.host_bytes, std::size_t{256} << 20);
  EXPECT_LE(p.host_private_bytes, gib(4.0));
  s.mode = Mode::kMaxSpeed;
  s.host_available_gib = 60.0;
  p = plan(s.inputs());
  EXPECT_EQ(p.cache.host_bytes, std::size_t{4} << 30);
  EXPECT_EQ(p.host_limit_bytes, gib(60.0) + gib(1.0) - gib(1.0));  // available + current - headroom

  // An explicit host budget below what the process already uses cannot be met in any mode.
  PlanInputs in = s.inputs();
  in.budgets.host_budget = std::size_t{512} << 20;
  try {
    plan(in);
    FAIL() << "expected BudgetError";
  } catch (const BudgetError& e) {
    EXPECT_GT(e.required_host, e.host_limit);
    EXPECT_NE(std::string(e.what()).find("host memory"), std::string::npos);
  }
}

// Sweeping the device budget: every feasible plan is consistent, once the DiT stays resident it stays resident,
// and among non-parking plans more memory never makes the estimate slower.
TEST(Planner, BudgetSweepIsMonotone) {
  for (const Mode mode : {Mode::kBalanced, Mode::kMaxSpeed, Mode::kMinMemory}) {
    double last_time = 1e30;
    bool seen_unparked = false;
    int feasible = 0;
    for (double free = 6.0; free <= 26.0; free += 0.25) {
      Scenario s;
      s.mode = mode;
      s.vram_free_gib = free;
      s.streaming = true;
      ExecutionPlan p;
      try {
        p = plan(s.inputs());
      } catch (const BudgetError& e) {
        EXPECT_EQ(feasible, 0) << "a larger budget became infeasible at " << free << " GiB: " << e.what();
        continue;
      }
      ++feasible;
      expect_consistent(p);
      const bool parked = p.dit.residency == DitResidency::kParked;
      if (mode != Mode::kMinMemory) {
        if (seen_unparked) {
          EXPECT_FALSE(parked) << free;
        }
        if (!parked) {
          EXPECT_LE(p.time.total_s(), last_time + 1e-9) << mode_name(mode) << " at " << free << " GiB";
          last_time = p.time.total_s();
          seen_unparked = true;
        }
      }
    }
    EXPECT_GT(feasible, 60) << mode_name(mode);
  }
}

TEST(Planner, JsonReportParses) {
  const ExecutionPlan p = plan(Scenario{}.inputs());
  const mi::json::Value v = mi::json::parse(p.to_json());
  EXPECT_EQ(v.at("mode").as_string(), "balanced");
  EXPECT_EQ(v.at("dit").at("residency").as_string(), "resident");
  EXPECT_TRUE(v.at("vae").at("tiled").as_bool());
  EXPECT_EQ(v.at("vae").at("tile").as_int(), 512);
  EXPECT_EQ(v.at("text_encoder").at("buffers").as_int(), 2);
  EXPECT_EQ(static_cast<std::size_t>(v.at("arena_bytes").as_int()), p.arena_bytes);
  EXPECT_EQ(v.at("explanation").as_string(), p.explanation);
  EXPECT_NEAR(v.at("time_s").at("total").as_double(), p.time.total_s(), 1e-3 * p.time.total_s());
}

TEST(Planner, RejectsIncompleteInputs) {
  PlanInputs in = Scenario{}.inputs();
  in.workspaces.decode = 0;
  EXPECT_THROW(plan(in), std::invalid_argument);
  in.request.decode = false;  // latents only: no decode workspace needed
  EXPECT_NO_THROW(plan(in));
  in = Scenario{}.inputs();
  in.workspaces.text_encoder = nullptr;
  EXPECT_THROW(plan(in), std::invalid_argument);
  in.request.te_cached = true;  // not needed on a cache hit
  EXPECT_NO_THROW(plan(in));
  in = Scenario{}.inputs();
  in.shapes.dit_bytes = 0;
  EXPECT_THROW(plan(in), std::invalid_argument);
}

// The dev box's measured transfer rates (E9, P4): pinned 24 GB/s, page cache -> device 19.5 GB/s.
HardwareProfile dev_box() {
  HardwareProfile hw;
  hw.h2d_gbps = 24.0;
  hw.host_gbps = 19.5;
  hw.disk_gbps = 2.3;
  hw.bf16_tflops = 60.0;
  hw.measured = true;
  return hw;
}

// P4 on 6, 8, 12 and 16 GiB cards (free = card - desktop/context share): min_memory streams every block and stays
// under AC8's 7.5 GiB device peak with <= 4 GiB host private; balanced keeps the DiT whole where it fits and
// otherwise streams with as many resident blocks as fit; nothing is refused that streaming can serve.
TEST(Planner, StreamingAcrossCardSizes) {
  struct Card {
    double gib, free;
  };
  for (const Card card : {Card{6, 5.4}, Card{8, 7.3}, Card{12, 11.2}, Card{16, 15.2}}) {
    for (const int size : {1024, 1456, 2048}) {
      Scenario s;
      s.size = size;
      s.lengths = {282, 15};
      s.vram_free_gib = card.free;
      s.streaming = true;
      s.hardware = dev_box();
      s.mode = Mode::kMinMemory;
      const ExecutionPlan min = plan(s.inputs());
      expect_consistent(min);
      const std::string where = std::to_string(static_cast<int>(card.gib)) + " GiB card, " + std::to_string(size) + "^2";
      EXPECT_EQ(min.dit.residency, DitResidency::kStreamed) << where;
      EXPECT_EQ(min.dit.resident_blocks, 0) << where;
      EXPECT_GE(min.dit.buffers, 2) << where;
      EXPECT_EQ(min.dit.prefetch_blocks, min.dit.buffers - 1) << where;
      EXPECT_LE(min.host_private_bytes, gib(4.0)) << where;
      if (size == 1024) {
        EXPECT_LE(min.device_peak_bytes, gib(7.5)) << where << "\n" << min.explanation;  // AC8
      }
      EXPECT_GE(min.dit.hidden, 0.8) << where << "\n" << min.explanation;  // AC10 at the dev box's rates

      s.mode = Mode::kBalanced;
      const ExecutionPlan bal = plan(s.inputs());
      expect_consistent(bal);
      EXPECT_LE(min.arena_bytes, bal.arena_bytes) << where;
      const bool whole_fits = card.gib >= 16 || (card.gib >= 12 && size <= 1456);
      if (whole_fits) {
        EXPECT_EQ(bal.dit.residency, DitResidency::kResident) << where << "\n" << bal.explanation;
      } else {
        EXPECT_EQ(bal.dit.residency, DitResidency::kStreamed) << where << "\n" << bal.explanation;
      }
      if (bal.dit.residency == DitResidency::kStreamed) {
        EXPECT_EQ(bal.dit.source, DitSource::kFile) << where;
        if (card.gib >= 12 || (card.gib >= 8 && size <= 1456)) {
          EXPECT_GT(bal.dit.resident_blocks, 0) << where;
        }
        // As many resident blocks as fit: one more would exceed the cap.
        EXPECT_GT(bal.arena_bytes + moss_shapes().dit_block_bytes, bal.arena_cap_bytes) << where << "\n" << bal.explanation;
        EXPECT_NE(bal.explanation.find("does not fit"), std::string::npos) << bal.explanation;
      }
    }
  }
}

// AC16 with other applications holding ~3 GiB on the dev box: 2048^2 streams instead of being refused (E11).
TEST(Planner, TightDevBoxStreams2048) {
  Scenario s;
  s.size = 2048;
  s.lengths = {282, 15};
  s.vram_free_gib = 10.9;
  s.hardware = dev_box();
  s.streaming = false;
  EXPECT_THROW(plan(s.inputs()), BudgetError);  // P3: refused
  s.streaming = true;
  const ExecutionPlan p = plan(s.inputs());
  expect_consistent(p);
  EXPECT_EQ(p.dit.residency, DitResidency::kStreamed);
  EXPECT_GT(p.dit.resident_blocks, 10);
  EXPECT_GE(p.dit.hidden, 0.9 - 1e-9);  // ~8 s of compute per step hides the copy (the model's 90% cap)
  EXPECT_FALSE(p.dit.park_for_decode);
}

// The explanation states the residency, the per-step copy against compute, the expected overlap and the rates.
TEST(Planner, StreamingExplanationAndJson) {
  Scenario s;
  s.size = 1456;
  s.lengths = {4096, 15};
  s.vram_free_gib = 7.3;
  s.streaming = true;
  s.hardware = dev_box();
  s.mode = Mode::kMinMemory;
  const ExecutionPlan p = plan(s.inputs());
  for (const char* part : {"0 of 42 blocks resident", "copied every step from the weight files (page cache)",
                           "block lookahead", "of the copy hidden", "Transfer rates (measured): pinned H2D 24.0 GB/s",
                           "min_memory streams"}) {
    EXPECT_NE(p.explanation.find(part), std::string::npos) << part << "\n" << p.explanation;
  }
  EXPECT_NEAR(p.dit.transfer_s, 42.0 * static_cast<double>(moss_shapes().dit_block_bytes) / 19.5e9, 1e-6);
  EXPECT_GT(p.dit.compute_s, 4 * p.dit.transfer_s);
  const mi::json::Value v = mi::json::parse(p.to_json());
  const mi::json::Value& d = v.at("dit");
  EXPECT_EQ(d.at("residency").as_string(), "streamed");
  EXPECT_EQ(d.at("source").as_string(), "file");
  EXPECT_EQ(d.at("resident_blocks").as_int(), 0);
  EXPECT_EQ(d.at("buffers").as_int(), p.dit.buffers);
  EXPECT_EQ(static_cast<std::size_t>(d.at("stream_bytes_per_step").as_int()), 42 * moss_shapes().dit_block_bytes);
  EXPECT_NEAR(d.at("expected_hidden").as_double(), p.dit.hidden, 1e-5);
}

// More buffers cost device memory and buy overlap: min_memory takes 2 when the denoise phase sets the peak, the
// cost model prefers 3 where memory is not the objective.
TEST(Planner, StreamBufferCount) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 7.3;
  s.streaming = true;
  s.te_cached = true;  // no text-encoder phase: the denoise is the peak
  s.tiles = true;
  s.mode = Mode::kMinMemory;
  const ExecutionPlan min = plan(s.inputs());
  EXPECT_EQ(min.dit.buffers, 2) << min.explanation;
  s.mode = Mode::kBalanced;
  const ExecutionPlan bal = plan(s.inputs());
  EXPECT_EQ(bal.dit.buffers, 3) << bal.explanation;
  EXPECT_GT(bal.dit.hidden, min.dit.hidden - 1e-12);
}

// A pinned host copy is the source only where the page cache is not expected to serve the weights (or in
// max_speed); it costs host private memory, so min_memory's 4 GiB target keeps the file source.
TEST(Planner, HostPinnedSourceForColdWeights) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 7.3;
  s.streaming = true;
  s.host_pinned = true;
  s.hardware = dev_box();
  const ExecutionPlan warm = plan(s.inputs());
  EXPECT_EQ(warm.dit.source, DitSource::kFile);
  s.dit_weights_cached = false;
  const ExecutionPlan cold = plan(s.inputs());
  expect_consistent(cold);
  EXPECT_EQ(cold.dit.source, DitSource::kHostPinned) << cold.explanation;
  EXPECT_GE(cold.host_private_bytes, cold.dit.stream_bytes);
  EXPECT_LE(cold.host_private_bytes, cold.host_target_bytes);
  EXPECT_LT(cold.time.denoise_s, warm.time.denoise_s * 2.0);
  s.mode = Mode::kMinMemory;
  const ExecutionPlan min = plan(s.inputs());
  EXPECT_EQ(min.dit.source, DitSource::kFile);
  EXPECT_LE(min.host_private_bytes, gib(4.0));
}

// M3: the pipeline's keep-or-reload decision with a fixed budget. Requests alternate between 1024^2 and 2048^2 (which
// want different numbers of resident blocks) and the free device memory swings by +-150 MiB; the simulated pipeline
// reloads the DiT whenever the chosen plan's residency differs from the loaded one or needs a larger arena. With
// prefer_held it settles after the first switch; planning each request alone reloads on every alternation.
TEST(Planner, HeldDitAvoidsReloadChurn) {
  struct Sim {
    bool use_held;
    std::optional<HeldDit> held;
    int reloads = 0;
    void request(const PlanInputs& in) {
      ExecutionPlan p = plan(in);
      bool kept = false;
      if (use_held && held) p = prefer_held(in, std::move(p), *held, &kept);
      const bool same = held && (p.dit.residency == DitResidency::kStreamed) == (held->dit.residency == DitResidency::kStreamed) &&
                        (p.dit.residency != DitResidency::kStreamed ||
                         (p.dit.resident_blocks == held->dit.resident_blocks && p.dit.buffers == held->dit.buffers &&
                          p.dit.source == held->dit.source));
      const bool grow = held && p.arena_bytes > held->arena_bytes;
      EXPECT_FALSE(kept && (!same || grow)) << p.explanation;
      if (!held || !same || grow) {
        reloads += held ? 1 : 0;
        held = HeldDit{grow || !held ? p.arena_bytes : std::max(held->arena_bytes, p.arena_bytes), p.dit};
      }
    }
  };
  const auto inputs = [](int size, double free_gib) {
    Scenario s;
    s.size = size;
    s.lengths = {282, 15};
    s.vram_free_gib = free_gib;
    s.streaming = true;
    s.te_cached = true;
    s.hardware = dev_box();
    PlanInputs in = s.inputs();
    in.request.steps = 20;
    return in;
  };
  for (const bool use_held : {false, true}) {
    Sim sim{use_held, std::nullopt, 0};
    for (int i = 0; i < 8; ++i) sim.request(inputs(i % 2 == 0 ? 1024 : 2048, 10.9));
    std::printf("[planner] alternating 1024^2 / 2048^2 at 10.9 GiB free, %s: %d DiT reloads in 7 switches\n",
                use_held ? "prefer_held" : "fresh plans", sim.reloads);
    if (use_held) {
      EXPECT_LE(sim.reloads, 1);
    } else {
      EXPECT_GE(sim.reloads, 6);  // the scenario does churn without it
    }
    Sim swing{use_held, std::nullopt, 0};
    for (int i = 0; i < 10; ++i) swing.request(inputs(1456, 10.9 + (i % 3 - 1) * 0.15));
    std::printf("[planner] 1456^2 with +-150 MiB free-memory swings, %s: %d DiT reloads\n",
                use_held ? "prefer_held" : "fresh plans", swing.reloads);
    if (use_held) {
      EXPECT_EQ(swing.reloads, 0);
    }
  }
}

// A held DiT is dropped when the request's own plan saves more than the reload costs, and the explanation says so.
TEST(Planner, HeldDitGivesWayWhenReloadPaysOff) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 15.2;
  s.streaming = true;
  s.te_cached = true;
  s.hardware = dev_box();
  PlanInputs in = s.inputs();
  in.request.steps = 50;
  const ExecutionPlan fresh = plan(in);
  ASSERT_EQ(fresh.dit.residency, DitResidency::kResident);
  HeldDit held;  // a fully streamed DiT from an earlier min_memory request, in a small arena
  held.dit.residency = DitResidency::kStreamed;
  held.dit.buffers = 2;
  held.arena_bytes = gib(3.0);
  bool kept = true;
  const ExecutionPlan p = prefer_held(in, fresh, held, &kept);
  EXPECT_FALSE(kept);
  EXPECT_EQ(p.dit.residency, DitResidency::kResident);
  EXPECT_NE(p.explanation.find("Reloads the DiT"), std::string::npos) << p.explanation;
  // Same residency and room in the held arena: nothing to decide.
  held.dit = fresh.dit;
  held.arena_bytes = fresh.arena_bytes + gib(1.0);
  const ExecutionPlan same = prefer_held(in, fresh, held, &kept);
  EXPECT_FALSE(kept);
  EXPECT_EQ(same.explanation, fresh.explanation);
}

// L1: with fewer streamed blocks than buffers, only one buffer per streamed block is counted.
TEST(Planner, StreamBuffersCappedAtStreamedBlocks) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 11.4;
  s.streaming = true;
  s.te_cached = true;
  s.parking = false;
  s.hardware = dev_box();
  const ExecutionPlan p = plan(s.inputs());
  if (p.dit.residency == DitResidency::kStreamed) {
    const int streamed = 42 - p.dit.resident_blocks;
    const std::size_t nonblock = moss_shapes().dit_bytes - 42 * moss_shapes().dit_block_bytes;
    EXPECT_EQ(p.dit.device_bytes, nonblock + static_cast<std::size_t>(42 - streamed + std::min(p.dit.buffers, streamed)) *
                                                  moss_shapes().dit_block_bytes)
        << p.explanation;
  }
  PlanInputs in = s.inputs();
  in.keep_dit = DitPlan{};
  in.keep_dit->residency = DitResidency::kStreamed;
  in.keep_dit->resident_blocks = 41;
  in.keep_dit->buffers = 3;
  in.budgets.vram_free = gib(15.2);
  const ExecutionPlan one = plan(in);
  EXPECT_EQ(one.dit.device_bytes, moss_shapes().dit_bytes) << "41 resident + 1 buffer = the whole DiT";
}

// L4: CUDA graphs of a resident DiT count against the cap and the device peak (not against streamed plans).
TEST(Planner, GraphMemoryCountsForResidentDit) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 12.0;
  s.streaming = true;
  s.hardware = dev_box();
  PlanInputs in = s.inputs();
  const ExecutionPlan without = plan(in);
  ASSERT_EQ(without.dit.residency, DitResidency::kResident);
  in.caps.dit_graph_bytes = std::size_t{42} << 20;
  const ExecutionPlan with = plan(in);
  EXPECT_EQ(with.dit.graph_bytes, std::size_t{42} << 20);
  EXPECT_EQ(with.device_peak_bytes, with.arena_bytes + in.budgets.context_bytes + with.dit.graph_bytes);
  EXPECT_LE(with.arena_bytes + with.dit.graph_bytes, with.arena_cap_bytes);
  EXPECT_EQ(mi::json::parse(with.to_json()).at("dit").at("graph_bytes").as_int(), 42 << 20);
  // A cap the chosen plan only fits without the graphs: with them it takes another configuration (a smaller decode
  // or streaming) whose arena and graphs fit together.
  in.caps.dit_graph_bytes = 0;
  in.budgets.vram_free = without.arena_bytes + in.budgets.vram_reserve + (std::size_t{8} << 20);
  const ExecutionPlan base = plan(in);
  in.caps.dit_graph_bytes = std::size_t{64} << 20;
  const ExecutionPlan tight = plan(in);
  EXPECT_LE(tight.arena_bytes + tight.dit.graph_bytes, tight.arena_cap_bytes) << tight.explanation;
  EXPECT_LT(tight.arena_bytes, base.arena_bytes) << tight.explanation;
  EXPECT_EQ(tight.dit.graph_bytes, tight.dit.residency == DitResidency::kStreamed ? 0u : in.caps.dit_graph_bytes);
}

// N4: a held resident DiT's live CUDA graphs are reused by the kept plan. vram_free (live free memory plus the arena)
// does not count them, and the kept plan's own graph reservation must not be charged to the held arena again: a
// request that fits the held arena with the held graphs keeps the loaded DiT.
TEST(Planner, HeldDitGraphsAreNotCountedTwice) {
  Scenario s;
  s.size = 1024;
  s.lengths = {282, 15};
  s.vram_free_gib = 12.0;
  s.streaming = true;
  s.te_cached = true;
  s.hardware = dev_box();
  PlanInputs in = s.inputs();
  in.caps.dit_graph_bytes = std::size_t{42} << 20;
  const ExecutionPlan first = plan(in);
  ASSERT_EQ(first.dit.residency, DitResidency::kResident);
  ASSERT_EQ(first.dit.graph_bytes, in.caps.dit_graph_bytes);
  // Loaded in exactly the arena it needs, its graphs captured; nothing else is free on the device.
  const HeldDit held{first.arena_bytes, first.dit, first.dit.graph_bytes};
  in.budgets.vram_free = first.arena_bytes + in.budgets.vram_reserve;
  // The request's own plan wants a larger arena (it would reload): the kept plan fits the held arena and graphs.
  ExecutionPlan larger = first;
  larger.arena_bytes = first.arena_bytes + gib(1.0);
  bool kept = false;
  const ExecutionPlan p = prefer_held(in, larger, held, &kept);
  EXPECT_TRUE(kept) << p.explanation;
  EXPECT_EQ(p.arena_bytes, first.arena_bytes) << p.explanation;
  EXPECT_EQ(p.dit.graph_bytes, first.dit.graph_bytes);
  EXPECT_LE(p.arena_bytes + p.dit.graph_bytes, p.arena_cap_bytes);
  // Without the held graphs counted (none captured yet), the same budget leaves no room for them: the kept plan must
  // shrink (or the request's own plan stands).
  const HeldDit no_graphs{first.arena_bytes, first.dit, 0};
  const ExecutionPlan q = prefer_held(in, larger, no_graphs, &kept);
  EXPECT_FALSE(kept && q.arena_bytes == first.arena_bytes) << q.explanation;
}

TEST(Planner, ShapesFromIndex) {
  using namespace mi::weights;
  ModelIndex idx;
  const auto group = [&](Component c, std::string name, std::uint64_t bytes, std::vector<IndexedTensor> tensors = {}) {
    WeightGroup g;
    g.component = c;
    g.name = std::move(name);
    g.total_bytes = bytes;
    g.tensors = std::move(tensors);
    idx.groups.push_back(std::move(g));
  };
  group(Component::kTextEncoder, "embed_tokens", 2000);
  group(Component::kTextEncoder, "layers.0", 300);
  group(Component::kTextEncoder, "layers.1", 400);
  group(Component::kTextEncoder, "final_norm", 8);
  group(Component::kTransformer, "globals", 50);
  IndexedTensor norm{"layers.0.norm1.weight", DType::kBF16, {3072}, 0, 0, 6144};
  IndexedTensor qnorm{"layers.0.attn.norm_q.weight", DType::kBF16, {128}, 0, 0, 256};
  group(Component::kTransformer, "block.0", 1000, {norm, qnorm});
  group(Component::kTransformer, "block.1", 1000);
  group(Component::kVae, "decoder", 77);
  const ModelShapes s = shapes_from_index(idx);
  EXPECT_EQ(s.te_layers, 2);
  EXPECT_EQ(s.te_layer_bytes, 400u);
  EXPECT_EQ(s.te_stream_bytes, 708u);
  EXPECT_EQ(s.te_embed_bytes, 2000u);
  EXPECT_EQ(s.dit_blocks, 2);
  EXPECT_EQ(s.dit_block_bytes, 1000u);
  EXPECT_EQ(s.dit_bytes, 2050u);
  EXPECT_EQ(s.dit_dim, 3072);
  EXPECT_EQ(s.vae_bytes, 77u);
  idx.groups.pop_back();
  EXPECT_THROW(shapes_from_index(idx), std::invalid_argument);
}

TEST(PlannerHostMemory, ReportsPrivateMemory) {
  const HostMemory before = process_host_memory();
#ifdef _WIN32
  ASSERT_TRUE(before.valid);
#else
  ASSERT_TRUE(before.valid);
  EXPECT_EQ(before.private_bytes, before.anon_bytes + before.shmem_bytes + before.swap_bytes);
  EXPECT_GE(before.rss_bytes, before.anon_bytes);
#endif
  constexpr std::size_t kBytes = std::size_t{96} << 20;
  char* p = static_cast<char*>(std::malloc(kBytes));
  ASSERT_NE(p, nullptr);
  // Volatile stores to every page: resident, and the allocation cannot be optimized away.
  volatile char* v = p;
  for (std::size_t i = 0; i < kBytes; i += 4096) v[i] = 1;
  const HostMemory after = process_host_memory();
  std::free(p);
  EXPECT_GE(after.private_bytes, before.private_bytes + kBytes * 9 / 10);
  EXPECT_GT(host_available_bytes(), std::size_t{64} << 20);
}

}  // namespace
