// Planner (src/planner/) on synthetic budgets: pure logic, no GPU. Model shapes are the real MOSS-Image group sizes
// (moss-infer inspect); the workspace queries are smooth stand-ins for the models' exact *_workspace_bytes() with the
// measured magnitudes (DiT activations 1.19 GiB at 1456^2 + 4096 tokens, fp32 decode ~1.5 KB per pixel).
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
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

  PlanInputs inputs() const {
    PlanInputs in;
    in.shapes = moss_shapes();
    in.request.height = in.request.width = size;
    in.request.text_lengths = lengths;
    in.request.steps = 50;
    in.request.rows_per_step = 2;
    in.request.te_cached = te_cached;
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
