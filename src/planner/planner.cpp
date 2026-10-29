#include "planner/planner.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mi::planner {
namespace {

// Host-private targets per mode (plan §3, AC7/AC8): min_memory <= 4 GiB, balanced <= 8 GiB, max_speed whatever the
// machine allows.
constexpr std::size_t kMinMemoryHostTarget = std::size_t{4} << 30;
constexpr std::size_t kBalancedHostTarget = std::size_t{8} << 30;
// Host RAM left to the OS and other applications when the limit comes from MemAvailable.
constexpr std::size_t kHostHeadroom = std::size_t{1} << 30;
// Embedding-cache caps per mode, and the floor (two 4096-token embeddings are 64 MiB).
constexpr std::size_t kCacheMinMemory = std::size_t{256} << 20;
constexpr std::size_t kCacheBalanced = std::size_t{1} << 30;
constexpr std::size_t kCacheMaxSpeed = std::size_t{4} << 30;
constexpr std::size_t kCacheFloor = std::size_t{64} << 20;
// Cost model: the reference-precision untiled decode on the dev box takes 1.02 s at 1456^2 and 2.53 s at 2048^2
// (~0.6 s/MP at 40 TFLOPS); tiled schedules report their own cost relative to it.
constexpr double kDecodeSecondsPerMegapixel = 0.6;
constexpr double kDecodeReferenceTflops = 40.0;
// Fraction of a streamed DiT step's copy time hidden behind compute when the copy is shorter (AC10 asks >= 80%;
// measured 98-100% with 3 buffers); the rest (synchronization, memory-bandwidth
// contention) adds to the step, so resident blocks always pay off. Two buffers (one block of lookahead) absorb less
// jitter between the copy and compute of neighbouring blocks.
constexpr double kStreamOverlap = 0.9;
constexpr double kStreamOverlapTwoBuffers = 0.8;
// P6 cost model: throughput of the INT8 ConvRot GEMMs (rotation and quantization included) and of the 8-bit attention
// relative to their bf16 counterparts, from the dev box's 1024^2 CFG step (282 + 10 tokens, every block resident, A/B
// in one process): bf16 fused 1.38 s, INT8 + exact 0.67 s, INT8 + q8_fp8 0.55 s, INT8 + q8_fp16 0.58 s, with the
// attention's ~19 % FLOP share of the bf16 step split out. The fp32-accumulating variants are not measured: assumed
// slower than q8_fp16.
constexpr double kInt8GemmSpeedup = 2.7;
// INT4 ConvRot GEMMs (P6c), the same measurement on the community INT4 file (every linear INT4, 2026-10-22, one
// process): bf16 1.38 s, INT8 + exact 0.71 s, INT4 + exact 0.55 s, INT4 + q8_fp8 0.40 s; with the same attention split,
// the INT4 GEMMs run 3.8x (exact) to 4.3x (q8) the bf16 ones, about 1.55x the INT8 ones.
constexpr double kInt4GemmSpeedup = 4.0;
// INT4 linears on a GPU without INT4 MMA (sm_90+) run the INT8 GEMMs on operands unpacked every forward (moss_dit):
// the INT8 speedup less the unpack traffic, ~2 % of the step (2026-10-23, 1024^2 CFG step, 282 + 10 tokens: RTX 5090
// INT8 + exact 0.224 s vs INT4 via INT8 0.229 s, INT4 on ptxas's emulated INT4 MMA 1.029 s; RTX 4090 Laptop INT8
// 0.654 s vs INT4 via INT8 0.668 s).
constexpr double kInt4ViaInt8Speedup = 2.6;
// NVFP4 GEMMs (sm_120a, quantizers included) relative to the INT8 ConvRot ones; NVFP4 runs on compute capability 12.0
// only. Measured at the block level (GpuDitNvfp4Speed.BlockRateAndStep1024, RTX 5090, 2026-10-25): a 1024^2 CFG block
// (8484 rows) of our converter's NVFP4 SFT file 3.98 ms against Comfy-Org's INT8 block 5.21 ms, each less its 2.19 ms
// GEMM-free part (QK-norm + RoPE, exact attention): 1.68. It includes the unfused SiTU-GLU (the ffn_down quantizer
// reads the [L, 2F] up_gate output).
constexpr double kNvfp4SpeedVsInt8 = 1.68;
constexpr double kQ8Fp8AttentionSpeedup = 1.9;
constexpr double kQ8Fp16AttentionSpeedup = 1.5;
constexpr double kQ8F32AccAttentionSpeedup = 1.3;

std::string gib(std::size_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f GiB", static_cast<double>(bytes) / static_cast<double>(kGiB));
  return buf;
}

std::string secs(double s) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), s < 10 ? "%.2f s" : "%.0f s", s);
  return buf;
}

std::size_t sat_sub(std::size_t a, std::size_t b) { return a > b ? a - b : 0; }

struct Candidate {
  DitPlan dit;
  TePlan te;
  VaePlan vae;
  PhasePeaks peaks;
  std::size_t arena = 0;
  std::size_t host_without_cache = 0;
  TimeEstimate time;

  bool parks() const { return dit.residency == DitResidency::kParked; }
  std::size_t device() const { return arena + dit.graph_bytes; }  // what the arena cap must hold
};

class Planner {
 public:
  explicit Planner(const PlanInputs& in) : in_(in), s_(in.shapes), r_(in.request), hw_(in.hardware) {
    if (r_.height <= 0 || r_.width <= 0 || r_.steps < 0 || r_.rows_per_step < 1) {
      throw std::invalid_argument("planner: invalid request shape");
    }
    if (s_.dit_bytes == 0 || s_.dit_blocks <= 0) throw std::invalid_argument("planner: model shapes lack the DiT");
    if (te_runs() && (!in_.workspaces.text_encoder || s_.te_stream_bytes == 0 || r_.text_lengths.empty())) {
      throw std::invalid_argument("planner: the text encoder runs but its shapes or workspace query are missing");
    }
    if (r_.decode && in_.workspaces.decode == 0) throw std::invalid_argument("planner: decode requested without its workspace");
    cap_ = arena_cap(in_.budgets);
    const Budgets& b = in_.budgets;
    system_limit_ = b.host_available == 0 ? std::numeric_limits<std::size_t>::max()
                                          : sat_sub(b.host_private_now + b.host_available, kHostHeadroom);
    if (b.host_budget != 0) {
      host_limit_ = host_target_ = b.host_budget;
    } else {
      host_limit_ = system_limit_;
      const std::size_t preset = in_.mode == Mode::kMinMemory ? kMinMemoryHostTarget
                                 : in_.mode == Mode::kBalanced ? kBalancedHostTarget
                                                               : host_limit_;
      host_target_ = std::min(preset, host_limit_);
    }
  }

  // Seconds to load the DiT of `d` from the weight files: its persistent bytes, plus a pinned host copy.
  double load_seconds(const DitPlan& d) const {
    const std::size_t bytes = dit_persistent(d) + (d.source == DitSource::kHostPinned ? dit_streamed_bytes(d) : 0);
    return static_cast<double>(bytes) / (load_rate_gbps() * 1e9);
  }
  // A DiT load's rate (the first load; prefer_held's reload), at the I/O mode the pipeline picks (choose_dit_load_io):
  // files expected in the page cache: the warm rate (warm_load_rate_gbps); cold: the measured direct-I/O load rate (the
  // pipeline reads a cold DiT with direct I/O), else the cold-disk rate.
  double load_rate_gbps() const {
    if (r_.dit_weights_cached) return warm_load_rate_gbps();
    if (hw_.direct_gbps <= 0) return weight_rate_gbps(false, false);
    return std::max(1e-3, std::min(hw_.direct_gbps, hw_.h2d_gbps));
  }
  // A warm load: direct where the learned rates chose it (warm_direct), else buffered at its measured warm rate, else
  // the page-cache rate.
  double warm_load_rate_gbps() const {
    const double g = hw_.warm_direct ? hw_.direct_gbps : hw_.buffered_warm_gbps;
    return g > 0 ? std::max(1e-3, std::min(g, hw_.h2d_gbps)) : weight_rate_gbps(false, true);
  }
  // A parking reload: buffered (the page cache keeps the files for the next one) at the warm or cold rate, or direct
  // where the learned rates chose direct for warm loads.
  double reload_rate_gbps() const {
    return hw_.warm_direct || r_.dit_weights_cached ? warm_load_rate_gbps() : weight_rate_gbps(false, false);
  }

  ExecutionPlan run() {
    std::vector<Candidate> all = enumerate();
    std::vector<const Candidate*> feasible;
    for (const Candidate& c : all) {
      if (c.device() <= cap_ && c.host_without_cache + kCacheFloor <= host_limit_ && pin_room(c)) feasible.push_back(&c);
    }
    if (feasible.empty()) throw_budget_error(all);
    const Candidate* best = *std::min_element(feasible.begin(), feasible.end(),
                                              [&](const Candidate* a, const Candidate* b) { return rank(*a) < rank(*b); });
    return make_plan(*best, all);
  }

 private:
  bool te_runs() const { return !r_.text_lengths.empty() && !r_.te_cached; }

  bool pinned_te(const Candidate& c) const {
    return c.te.run && c.te.residency == TeResidency::kHostResident && c.te.pinned;
  }
  std::size_t pin_margin() const { return host_pin_margin(in_.budgets.host_total); }
  // Why kAuto does not pin the encoder in this mode on this machine (empty: it may, given the room).
  std::string auto_pin_refusal() const {
    if (in_.mode == Mode::kMinMemory) return "min_memory holds no copy";
    if (in_.caps.te_quantized) return {};
    if (in_.mode != Mode::kMaxSpeed) return "balanced keeps the bf16 encoder streamed (te_host_pin on pins it)";
    if (in_.budgets.host_total == 0) return "machine RAM unknown";
    if (in_.budgets.host_total < kTeBf16PinMinRam) {
      return "the bf16 encoder is pinned only on machines with " + gib(kTeBf16PinMinRam) + " of RAM or more";
    }
    return {};
  }
  // Whether a pinned text-encoder copy may be planned (te_pin, the file comment): kOn within the host limit (the
  // feasibility check), kAuto where its mode and the RAM allow it and the copy with the embedding cache leaves
  // pin_margin() of the system's available memory free, even under an explicit host budget.
  bool pin_room(const Candidate& c) const {
    if (!pinned_te(c) || in_.caps.te_pin == TeHostPin::kOn) return true;
    if (in_.caps.te_pin == TeHostPin::kOff || !auto_pin_refusal().empty()) return false;
    return fits_pin_margin(c.host_without_cache + cache_bytes(c), system_limit_, in_.budgets.host_total);
  }
  // The host-private target a candidate is held to: balanced keeps a pinned copy it may plan (a quantized encoder under
  // kAuto, any under kOn) above its target; an explicit host budget is the target itself.
  std::size_t target_for(const Candidate& c) const {
    const bool kept = in_.mode == Mode::kBalanced && in_.budgets.host_budget == 0 && pinned_te(c) &&
                      (in_.caps.te_quantized || in_.caps.te_pin == TeHostPin::kOn);
    return host_target_ + (kept ? s_.te_stream_bytes : 0);
  }

  std::size_t dit_nonblock_bytes() const {
    return sat_sub(s_.dit_bytes, static_cast<std::size_t>(s_.dit_blocks) * s_.dit_block_bytes);
  }

  // DiT device bytes outside the denoise (persistent) and during it (+ streaming buffers).
  std::size_t dit_persistent(const DitPlan& d) const {
    if (d.residency != DitResidency::kStreamed) return s_.dit_bytes;
    return dit_nonblock_bytes() + static_cast<std::size_t>(d.resident_blocks) * s_.dit_block_bytes;
  }
  // Device block buffers a streamed plan allocates: no more than there are streamed blocks.
  int dit_buffers(const DitPlan& d) const { return std::min(d.buffers, s_.dit_blocks - d.resident_blocks); }
  std::size_t dit_denoise(const DitPlan& d) const {
    if (d.residency != DitResidency::kStreamed) return s_.dit_bytes;
    return dit_persistent(d) + static_cast<std::size_t>(dit_buffers(d)) * s_.dit_block_bytes;
  }
  std::size_t dit_streamed_bytes(const DitPlan& d) const {
    if (d.residency != DitResidency::kStreamed) return 0;
    return static_cast<std::size_t>(s_.dit_blocks - d.resident_blocks) * s_.dit_block_bytes;
  }

  // `cached`: the files are expected in the page cache.
  double weight_rate_gbps(bool from_host, bool cached) const {
    const double src = from_host || cached ? hw_.host_gbps : hw_.disk_gbps;
    return std::max(1e-3, std::min(src, hw_.h2d_gbps));
  }
  double tflops() const { return std::max(1e-3, hw_.bf16_tflops); }

  double te_seconds(const TePlan& te) const {
    if (!te.run) return 0;
    std::int64_t tokens = 0;
    for (const std::int64_t t : r_.te_lengths.empty() ? r_.text_lengths : r_.te_lengths) tokens += t;
    // ~2 FLOPs per bf16 weight (1 param) per token.
    const double compute = static_cast<double>(s_.te_stream_bytes) * static_cast<double>(tokens) / (tflops() * 1e12);
    if (te.residency == TeResidency::kVramResident) return compute;
    const double rate = (te.residency == TeResidency::kHostResident && te.pinned
                             ? std::max(1e-3, hw_.h2d_gbps)
                             : weight_rate_gbps(te.residency == TeResidency::kHostResident, r_.weights_cached)) *
                        1e9;
    const double load = static_cast<double>(s_.te_stream_bytes) / rate;
    // One buffer: every layer's copy and compute in series. Two or more: copies overlap compute except for one
    // layer's worth (the first copy or the last compute).
    if (te.buffers >= 2) return std::max(load, compute) + std::min(load, compute) / std::max(1, s_.te_layers);
    return load + compute;
  }

  // One step: compute, plus the part of the streamed blocks' copy not hidden behind it (fills d's per-step fields).
  double step_seconds(DitPlan& d) const {
    const double image_tokens = static_cast<double>(r_.height / 16) * static_cast<double>(r_.width / 16);
    double flops = 0;
    for (int row = 0; row < r_.rows_per_step; ++row) {
      // Row 1 is the negative prompt; rows 0 and 2 (SLG) the prompt.
      const std::size_t text_row = row == 1 && r_.text_lengths.size() > 1 ? 1 : 0;
      const double l = image_tokens + (r_.text_lengths.empty() ? 0.0 : static_cast<double>(r_.text_lengths[text_row]));
      // Reference keys (the edit layout): every query also attends to the R reference tokens.
      flops += sequence_flops(l, l + static_cast<double>(r_.reference_tokens));
    }
    const double compute = flops / (tflops() * 1e12);
    d.compute_s = compute;
    d.stream_bytes = dit_streamed_bytes(d);
    if (d.stream_bytes == 0) {
      d.transfer_s = 0;
      d.hidden = 1.0;
      return compute;
    }
    double transfer = 0;
    const double exposed = exposed_transfer(d, compute, transfer);
    d.transfer_s = transfer;
    d.hidden = 1.0 - exposed / transfer;
    return compute + exposed;
  }

  // One sequence of l tokens attending to `keys` keys through every block, in bf16-equivalent FLOPs. GEMMs: 2 FLOPs per
  // weight per token (bf16: 2 bytes per weight); attention: QK^T and PV, 2 * L * keys * dim each, per block. P6 paths:
  // divided by their measured speedup.
  double sequence_flops(double l, double keys) const {
    const double dim = s_.dit_dim > 0 ? s_.dit_dim : 3072;
    const double f4 = s_.dit_precision == Precision::kInt4 ? std::clamp(s_.dit_int4_fraction, 0.0, 1.0) : 0.0;
    const double fn = s_.dit_precision == Precision::kNvfp4 ? std::clamp(s_.dit_nvfp4_fraction, 0.0, 1.0) : 0.0;
    const double f8 = std::clamp(s_.dit_int8_fraction, 0.0, 1.0 - f4 - fn);
    const double gemm = s_.dit_precision != Precision::kBf16
                            ? 2.0 * static_cast<double>(s_.dit_params) * l *
                                  ((1.0 - f8 - f4 - fn) + f8 / kInt8GemmSpeedup +
                                   f4 / (in_.caps.int4_mma_native ? kInt4GemmSpeedup : kInt4ViaInt8Speedup) +
                                   fn / (kInt8GemmSpeedup * (in_.caps.nvfp4_native ? kNvfp4SpeedVsInt8 : 1.0)))
                            : static_cast<double>(s_.dit_bytes) * l;
    const double attention = 4.0 * l * keys * dim * s_.dit_blocks;
    const double attention_speedup = in_.attention == AttentionMode::kExact    ? 1.0
                                     : in_.attention == AttentionMode::kQ8Fp8  ? kQ8Fp8AttentionSpeedup
                                     : in_.attention == AttentionMode::kQ8Fp16 ? kQ8Fp16AttentionSpeedup
                                                                               : kQ8F32AccAttentionSpeedup;
    return gemm + attention / attention_speedup;
  }

  // The part of one pass over the streamed blocks' copies (`transfer` seconds, set) that `compute` seconds do not hide.
  double exposed_transfer(const DitPlan& d, double compute, double& transfer) const {
    const double rate = d.source == DitSource::kHostPinned ? std::max(1e-3, hw_.h2d_gbps) : weight_rate_gbps(false, r_.dit_weights_cached);
    transfer = static_cast<double>(dit_streamed_bytes(d)) / (rate * 1e9);
    const double overlap = dit_buffers(d) >= 3 ? kStreamOverlap : kStreamOverlapTwoBuffers;
    return std::max(0.0, transfer - compute) + (1.0 - overlap) * std::min(transfer, compute);
  }

  // One reference prefill (R tokens through every block, attending to each other), plus its weight pass when blocks
  // stream.
  double prefill_seconds(const DitPlan& d) const {
    const double r = static_cast<double>(r_.reference_tokens);
    const double compute = sequence_flops(r, r) / (tflops() * 1e12);
    if (dit_streamed_bytes(d) == 0) return compute;
    double transfer = 0;
    return compute + exposed_transfer(d, compute, transfer);
  }

  double decode_seconds(const VaePlan& v) const {
    if (!v.run) return 0;
    const double untiled = static_cast<double>(r_.height) * static_cast<double>(r_.width) / 1e6 * kDecodeSecondsPerMegapixel *
                           kDecodeReferenceTflops / tflops();
    return untiled * (v.kind == VaeDecodeKind::kUntiled ? 1.0 : v.cost);
  }

  void evaluate(Candidate& c) const {
    const std::size_t te_vram = c.te.run && c.te.residency == TeResidency::kVramResident ? s_.te_stream_bytes : 0;
    const std::size_t vae = r_.decode || r_.vae_resident ? s_.vae_bytes : 0;
    const std::size_t dit_keep = dit_persistent(c.dit);
    if (c.te.run) {
      const std::size_t dit_part = c.dit.park_for_text_encoder ? 0 : dit_keep;
      c.peaks.text_encoder = dit_part + vae + te_vram + in_.workspaces.text_encoder(c.te.residency, c.te.buffers);
    }
    c.peaks.denoise = dit_denoise(c.dit) + vae + te_vram + in_.workspaces.denoise;
    if (c.vae.run) {
      const std::size_t resident = (c.dit.park_for_decode ? 0 : dit_keep) + vae + te_vram;
      if (c.vae.kind == VaeDecodeKind::kUntiled) {
        c.vae.stack_bytes = in_.workspaces.decode;
      } else {
        // The fastest schedule in what the resident models leave of the cap, or (smallest) of the other phases' peak
        // so the decode never sets the arena size unless it must; if none fits, the smallest (then the candidate may
        // be infeasible, but still describes the requirement).
        std::optional<TiledDecode> t;
        const std::size_t room = c.vae.smallest ? sat_sub(std::max(c.peaks.text_encoder, c.peaks.denoise), resident)
                                                : sat_sub(cap_, resident + kArenaSlack);
        if (room > 0) t = in_.workspaces.decode_tiled(room);
        if (!t) t = in_.workspaces.decode_tiled(0);
        if (!t) throw std::invalid_argument("planner: the tiled decode query has no smallest schedule");
        c.vae.stack_bytes = t->bytes;
        c.vae.tile = t->tile;
        c.vae.cost = t->cost;
      }
      c.peaks.decode = resident + c.vae.stack_bytes;
    }
    c.dit.device_bytes = dit_denoise(c.dit);
    // Graphs capture only wholly resident blocks (a streamed block's copies are host-driven).
    c.dit.graph_bytes = c.dit.residency == DitResidency::kStreamed ? 0 : in_.caps.dit_graph_bytes;
    c.arena = std::max({c.peaks.text_encoder, c.peaks.denoise, c.peaks.decode}) + kArenaSlack;

    std::size_t host = in_.budgets.host_private_now + in_.workspaces.host_buffers;
    if (c.te.run && c.te.residency != TeResidency::kVramResident) host += in_.caps.te_ring_bytes;
    if (c.te.residency == TeResidency::kHostResident) host += s_.te_stream_bytes;
    // Streamed DiT blocks are read from the weight files (page cache: file-backed, not private) into the readers'
    // rings, or held as a pinned host copy.
    if (c.dit.residency == DitResidency::kStreamed) {
      host += c.dit.source == DitSource::kFile ? in_.caps.dit_ring_bytes : dit_streamed_bytes(c.dit);
    }
    c.host_without_cache = host;

    c.time.text_encoder_s = te_seconds(c.te);
    c.time.dit_load_s = c.parks() ? static_cast<double>(s_.dit_bytes) / (reload_rate_gbps() * 1e9) : 0;
    c.time.denoise_s = step_seconds(c.dit) * r_.steps;
    if (r_.reference_prefills > 0) {
      c.time.reference_prefill_s = prefill_seconds(c.dit) * r_.reference_prefills;
      c.time.denoise_s += c.time.reference_prefill_s;
    }
    c.time.decode_s = decode_seconds(c.vae);
  }

  std::vector<Candidate> enumerate() const {
    std::vector<TePlan> tes;
    if (!te_runs()) {
      tes.push_back({false, TeResidency::kStreamed, 0});
    } else {
      const int max_buffers = std::max(1, in_.caps.te_max_buffers);
      for (int b = 1; b <= max_buffers; ++b) tes.push_back({true, TeResidency::kStreamed, b});
      if (in_.caps.te_host_resident && in_.caps.te_host_pageable) {
        tes.push_back({true, TeResidency::kHostResident, max_buffers, false});
      }
      if (in_.caps.te_host_resident && in_.caps.te_host_pinned && in_.caps.te_pin != TeHostPin::kOff) {
        tes.push_back({true, TeResidency::kHostResident, max_buffers, true});
      }
      if (in_.caps.te_vram_resident) tes.push_back({true, TeResidency::kVramResident, 0});
    }
    std::vector<VaePlan> vaes;
    if (!r_.decode) {
      vaes.push_back({false, VaeDecodeKind::kUntiled, false, 0, 0, 1.0});
    } else {
      vaes.push_back({true, VaeDecodeKind::kUntiled, false, 0, 0, 1.0});
      if (in_.workspaces.decode_tiled) {
        vaes.push_back({true, VaeDecodeKind::kTiled, false, 0, 0, 1.0});
        vaes.push_back({true, VaeDecodeKind::kTiled, true, 0, 0, 1.0});
      }
    }
    std::vector<DitPlan> dits;
    dits.push_back({});
    if (in_.caps.dit_parking) {
      for (const bool te : {false, true}) {
        for (const bool dec : {false, true}) {
          if ((!te && !dec) || (te && !te_runs()) || (dec && !r_.decode)) continue;
          DitPlan d;
          d.residency = DitResidency::kParked;
          d.park_for_text_encoder = te;
          d.park_for_decode = dec;
          dits.push_back(d);
        }
      }
    }

    std::vector<Candidate> out;
    for (const TePlan& te : tes) {
      for (const VaePlan& vae : vaes) {
        for (const DitPlan& dit : dits) {
          Candidate c{dit, te, vae, {}, 0, 0, {}};
          evaluate(c);
          out.push_back(c);
        }
        if (in_.keep_dit && in_.keep_dit->residency == DitResidency::kStreamed) {
          Candidate c{{}, te, vae, {}, 0, 0, {}};
          c.dit.residency = DitResidency::kStreamed;
          c.dit.resident_blocks = in_.keep_dit->resident_blocks;
          c.dit.buffers = in_.keep_dit->buffers;
          c.dit.prefetch_blocks = c.dit.buffers - 1;
          c.dit.source = in_.keep_dit->source;
          evaluate(c);
          out.push_back(c);
        }
        if (!in_.caps.dit_streaming) continue;
        // P4: no resident blocks (least memory), and as many as fit the arena cap; every buffer count; the pinned
        // host source only where it can pay for its host memory (cold page cache, or max_speed's host budget).
        // Without the file source (runtime LoRA merges on the blocks): the pinned host copy only.
        std::vector<DitSource> sources;
        if (in_.caps.dit_file_source) sources.push_back(DitSource::kFile);
        if (in_.caps.dit_host_pinned &&
            (!in_.caps.dit_file_source || !r_.dit_weights_cached || in_.mode == Mode::kMaxSpeed)) {
          sources.push_back(DitSource::kHostPinned);
        }
        for (const DitSource src : sources) {
          for (int buffers = 2; buffers <= std::max(2, in_.caps.dit_max_buffers); ++buffers) {
            Candidate c{{}, te, vae, {}, 0, 0, {}};
            c.dit.residency = DitResidency::kStreamed;
            c.dit.buffers = buffers;
            c.dit.prefetch_blocks = buffers - 1;
            c.dit.source = src;
            evaluate(c);
            out.push_back(c);
            const int extra = blocks_that_fit(c);
            if (extra > 0) {
              c.dit.resident_blocks = extra;
              evaluate(c);
              out.push_back(c);
            }
          }
        }
      }
    }
    if (in_.keep_dit) {
      const DitPlan& k = *in_.keep_dit;
      std::erase_if(out, [&](const Candidate& c) {
        if (c.dit.residency != k.residency) return true;
        if (k.residency == DitResidency::kParked) {
          return c.dit.park_for_text_encoder != k.park_for_text_encoder || c.dit.park_for_decode != k.park_for_decode;
        }
        return k.residency == DitResidency::kStreamed &&
               (c.dit.resident_blocks != k.resident_blocks || c.dit.buffers != k.buffers || c.dit.source != k.source);
      });
      if (out.empty()) throw std::invalid_argument("planner: keep_dit names a residency this build cannot plan");
    }
    return out;
  }

  // Resident blocks to add to the 0-resident candidate `c` so that it still fits the cap (< every block). Each one
  // adds its bytes to every phase except where a tiled decode gives the room back, so step down from the linear
  // estimate until the candidate fits.
  int blocks_that_fit(const Candidate& c) const {
    if (c.device() >= cap_ || s_.dit_block_bytes == 0) return 0;
    int n = static_cast<int>(std::min<std::size_t>((cap_ - c.arena) / s_.dit_block_bytes, static_cast<std::size_t>(s_.dit_blocks - 1)));
    for (; n > 0; --n) {
      Candidate t = c;
      t.dit.resident_blocks = n;
      evaluate(t);
      if (t.device() <= cap_) break;
    }
    return n;
  }

  std::size_t cache_bytes(const Candidate& c) const {
    const std::size_t preset = in_.mode == Mode::kMinMemory ? kCacheMinMemory
                               : in_.mode == Mode::kBalanced ? kCacheBalanced
                                                             : kCacheMaxSpeed;
    return std::max(kCacheFloor, std::min(preset, sat_sub(target_for(c), c.host_without_cache)));
  }

  // balanced / max_speed: a DiT that fits whole stays whole (streaming only when it does not, even where streaming a
  // few blocks would buy an untiled decode for less than its copy costs: a stable residency keeps the DiT loaded
  // across requests), and streaming beats parking.
  using RankKey = std::tuple<bool, bool, bool, std::size_t, std::size_t, double, double>;
  RankKey rank(const Candidate& c) const {
    const bool over_target = c.host_without_cache + cache_bytes(c) > target_for(c);
    const std::size_t host = c.host_without_cache + cache_bytes(c);
    if (in_.mode == Mode::kMinMemory) {
      return {over_target, false, false, c.arena, host, c.time.total_s(), 0.0};
    }
    return {over_target, c.parks(), c.dit.residency == DitResidency::kStreamed, 0, 0, c.time.total_s(),
            static_cast<double>(c.arena)};
  }

  std::string precision_text() const {
    std::string t = s_.dit_precision == Precision::kInt4
                        ? (in_.caps.int4_mma_native ? "INT4 ConvRot" : "INT4 ConvRot (on the INT8 GEMMs: no INT4 MMA on this GPU)")
                    : s_.dit_precision == Precision::kNvfp4 ? "NVFP4"
                    : s_.dit_precision == Precision::kInt8  ? "INT8 ConvRot"
                                                            : "bf16";
    return is_q8(in_.attention) ? t + " + 8-bit attention (" + attention_name(in_.attention) + ")" : t;
  }

  std::string describe_dit(const DitPlan& d) const {
    switch (d.residency) {
      case DitResidency::kResident:
        return "resident " + precision_text() + " (" + gib(s_.dit_bytes) + "), never parked" +
               (d.graph_bytes > 0 ? ", CUDA graphs up to " + gib(d.graph_bytes) + " outside the arena" : std::string());
      case DitResidency::kParked: {
        std::string when = d.park_for_text_encoder && d.park_for_decode ? "the text encoder and the decode"
                           : d.park_for_text_encoder                    ? "the text encoder"
                                                                        : "the decode";
        return "parked " + precision_text() + " (" + gib(s_.dit_bytes) + "): evicted for " + when + " and reloaded for the next denoise";
      }
      case DitResidency::kStreamed: {
        char rate[160];
        std::snprintf(rate, sizeof(rate), "%.2f GB per step in ~%.2f s against ~%.2f s of compute: ~%.0f%% of the copy hidden",
                      static_cast<double>(d.stream_bytes) / 1e9, d.transfer_s, d.compute_s, 100.0 * d.hidden);
        const std::string from = d.source == DitSource::kHostPinned ? "a pinned host copy (" + gib(d.stream_bytes) + ")"
                                 : r_.dit_weights_cached            ? std::string("the weight files (page cache)")
                                                                    : std::string("the weight files (cold: disk)");
        return "streamed " + precision_text() + ": " + std::to_string(d.resident_blocks) + " of " + std::to_string(s_.dit_blocks) +
               " blocks resident, the other " + std::to_string(s_.dit_blocks - d.resident_blocks) + " copied every step from " +
               from + " through " + std::to_string(d.buffers) + " device block buffers (" +
               std::to_string(d.prefetch_blocks) + "-block lookahead), " + gib(d.device_bytes) + " on the device; " + rate;
      }
    }
    return {};
  }

  std::string describe_te(const TePlan& t) const {
    if (!t.run) return r_.text_lengths.empty() ? "not needed" : "skipped (embedding cache hit)";
    switch (t.residency) {
      case TeResidency::kStreamed:
        return "streamed from the weight files with " + std::to_string(t.buffers) + " device layer buffer" +
               (t.buffers == 1 ? " (synchronous)" : "s (layer k+1 copied while layer k runs)");
      case TeResidency::kHostResident:
        return "host-resident (" + gib(s_.te_stream_bytes) + (t.pinned ? " pinned" : "") +
               " private, kept across requests), streamed with " + std::to_string(t.buffers) + " layer buffers" +
               (t.pinned ? " at the pinned H2D rate" : "");
      case TeResidency::kVramResident:
        return "VRAM-resident (" + gib(s_.te_stream_bytes) + ")";
    }
    return {};
  }

  std::string describe_vae(const VaePlan& v) const {
    if (!v.run) return "skipped (latents only)";
    if (v.kind == VaeDecodeKind::kUntiled) return "untiled decode";
    char cost[16];
    std::snprintf(cost, sizeof(cost), "%.2f", v.cost);
    return "exact tiled decode, " + std::to_string(v.tile) + " px row bands (" + gib(v.stack_bytes) + ", ~" + cost +
           "x the untiled time)";
  }

  std::string describe_peaks(const Candidate& c) const {
    std::string s = "peaks: ";
    s += c.te.run ? "text " + gib(c.peaks.text_encoder) : std::string("text -");
    s += ", denoise " + gib(c.peaks.denoise);
    s += c.vae.run ? ", decode " + gib(c.peaks.decode) : std::string(", decode -");
    return s;
  }

  std::string budget_text() const {
    const Budgets& b = in_.budgets;
    std::string s = "free " + gib(b.vram_free) + " - reserve " + gib(b.vram_reserve);
    if (b.vram_budget != 0) s += "; budget " + gib(b.vram_budget) + " - context " + gib(b.context_bytes) + " - reserve";
    return s;
  }

  std::string request_text() const {
    std::string s = std::to_string(r_.width) + "x" + std::to_string(r_.height);
    for (std::size_t i = 0; i < r_.text_lengths.size(); ++i) s += (i ? "+" : ", ") + std::to_string(r_.text_lengths[i]);
    return r_.text_lengths.empty() ? s : s + " tokens";
  }

  // Why the chosen VAE / DiT option and not the faster one.
  std::string reasons(const Candidate& best, const std::vector<Candidate>& all) const {
    std::string s;
    if (best.vae.run && best.vae.kind == VaeDecodeKind::kTiled) {
      for (const Candidate& c : all) {
        if (c.vae.run && c.vae.kind == VaeDecodeKind::kUntiled && c.dit.residency == best.dit.residency &&
            c.dit.resident_blocks == best.dit.resident_blocks && c.dit.buffers == best.dit.buffers &&
            c.dit.source == best.dit.source && c.te.residency == best.te.residency &&
            c.te.buffers == best.te.buffers && c.parks() == best.parks()) {
          s += "\nUntiled decode would need a " + gib(c.arena) + " arena (decode peak " + gib(c.peaks.decode) + ")" +
               (c.device() > cap_ ? ", over the " + gib(cap_) + " cap; tiling keeps the DiT resident."
                               : std::string("; tiling keeps the device peak lower."));
          break;
        }
      }
    }
    if (best.parks()) {
      s += "\nParking is the fallback: no configuration keeping the DiT resident fits the " + gib(cap_) + " cap.";
    }
    if (pinned_te(best)) {
      s += "\nThe text encoder stays in RAM (pinned" + std::string(in_.caps.te_pin == TeHostPin::kOn ? ", te_host_pin on" : "") +
           ")" + (target_for(best) > host_target_ ? " above balanced's " + gib(host_target_) + " host target" : std::string()) +
           ": a prompt cache miss streams it at the pinned H2D rate; the system keeps " +
           (system_limit_ == std::numeric_limits<std::size_t>::max()
                ? std::string("an unknown amount of memory")
                : gib(sat_sub(system_limit_, best.host_without_cache + cache_bytes(best)))) +
           " available with it.";
    } else if (best.te.run && best.te.residency != TeResidency::kHostResident && in_.caps.te_host_resident &&
               in_.caps.te_host_pinned) {
      std::string why = in_.caps.te_pin == TeHostPin::kOff ? std::string("te_host_pin off") : auto_pin_refusal();
      if (why.empty()) {
        for (const Candidate& c : all) {
          if (pinned_te(c) && !pin_room(c) && c.device() <= cap_) {
            why = "its pinned copy (" + gib(s_.te_stream_bytes) + ") would leave less than " + gib(pin_margin()) +
                  " (max(8 GiB, 25 % of the RAM)) of the system's available memory";
            break;
          }
        }
      }
      if (!why.empty() && in_.mode != Mode::kMinMemory) s += "\nThe text encoder is not kept in RAM: " + why + ".";
    }
    if (best.dit.residency == DitResidency::kStreamed) {
      if (in_.mode == Mode::kMinMemory) {
        s += "\nmin_memory streams the DiT blocks to keep the device peak low.";
      } else {
        const Candidate* lean = nullptr;
        for (const Candidate& c : all) {
          if (c.dit.residency != DitResidency::kStreamed && (lean == nullptr || c.device() < lean->device())) lean = &c;
        }
        if (lean != nullptr) {
          s += "\nThe whole DiT does not fit: keeping it " + std::string(residency_name(lean->dit.residency)) +
               " would need a " + gib(lean->device()) + " arena, over the " + gib(cap_) + " cap.";
        }
      }
    }
    return s;
  }

  ExecutionPlan make_plan(const Candidate& c, const std::vector<Candidate>& all) const {
    ExecutionPlan p;
    p.mode = in_.mode;
    p.dit = c.dit;
    p.dit.precision = s_.dit_precision;
    p.dit.attention = in_.attention;
    p.te = c.te;
    p.vae = c.vae;
    p.cache.enabled = true;
    p.cache.host_bytes = cache_bytes(c);
    p.cache.keep_negative = true;
    p.cache.disk_spill = in_.cache_disk_spill;
    p.peaks = c.peaks;
    p.arena_bytes = c.arena;
    p.device_peak_bytes = c.device() + in_.budgets.context_bytes;
    p.arena_cap_bytes = cap_;
    p.host_private_bytes = c.host_without_cache + p.cache.host_bytes;
    p.host_target_bytes = host_target_;
    p.host_held_target_bytes = target_for(c);
    p.host_limit_bytes = host_limit_;
    p.time = c.time;

    const char* objective = in_.mode == Mode::kMinMemory ? "least device memory"
                            : in_.mode == Mode::kBalanced ? "least time within the budget"
                                                          : "least time, larger host caches";
    std::string e = std::string("Mode ") + mode_name(in_.mode) + " (" + objective + "), " + request_text() + ". Arena cap " +
                    gib(cap_) + " (" + budget_text() + "); host private target " + gib(host_target_) + ".";
    e += "\nDiT: " + describe_dit(c.dit) + ".";
    e += "\nText encoder: " + describe_te(c.te) + ".";
    e += "\nVAE: " + describe_vae(c.vae) + ".";
    e += reasons(c, all);
    e += "\nArena " + gib(c.arena) + " (" + describe_peaks(c) + "); est. device peak " + gib(p.device_peak_bytes) +
         ", host private " + gib(p.host_private_bytes) + " (embedding cache " + gib(p.cache.host_bytes) +
         (p.cache.disk_spill ? ", disk spill" : "") + ").";
    e += "\nEst. time " + secs(c.time.total_s()) + " (text " + secs(c.time.text_encoder_s) + ", DiT reload " +
         secs(c.time.dit_load_s) + ", denoise " + secs(c.time.denoise_s) + ", decode " + secs(c.time.decode_s) + ")" +
         (hw_.measured ? "." : ", default hardware profile.");
    char rates[200];
    std::snprintf(rates, sizeof(rates),
                  "\nTransfer rates (%s): pinned H2D %.1f GB/s, page cache %.1f GB/s, cold disk %.1f GB/s; bf16 %.0f TFLOPS.",
                  hw_.measured ? "measured" : "defaults", hw_.h2d_gbps, hw_.host_gbps, hw_.disk_gbps, hw_.bf16_tflops);
    if (c.dit.residency == DitResidency::kStreamed || c.te.run) e += rates;
    p.explanation = std::move(e);
    return p;
  }

  [[noreturn]] void throw_budget_error(const std::vector<Candidate>& all) const {
    const Candidate* lean = &all.front();
    for (const Candidate& c : all) {
      if (std::make_pair(c.device(), c.host_without_cache) < std::make_pair(lean->device(), lean->host_without_cache)) lean = &c;
    }
    const std::string config = "DiT " + describe_dit(lean->dit) + "; text encoder " + describe_te(lean->te) + "; VAE " +
                               describe_vae(lean->vae) + "; " + describe_peaks(*lean);
    if (lean->device() > cap_) {
      throw BudgetError("not enough GPU memory for " + request_text() + " in any mode: the leanest plan needs a " +
                            gib(lean->device()) + " arena (" + config + "), but only " + gib(cap_) + " is available (" +
                            budget_text() + "). Close other GPU applications, raise the budget or lower the resolution.",
                        lean->device(), cap_, lean->host_without_cache + kCacheFloor, host_limit_);
    }
    const Candidate* host_lean = nullptr;
    for (const Candidate& c : all) {
      if (c.device() <= cap_ && (host_lean == nullptr || c.host_without_cache < host_lean->host_without_cache)) host_lean = &c;
    }
    throw BudgetError("not enough host memory for " + request_text() + " in any mode: the leanest plan that fits the GPU " +
                          "needs " + gib(host_lean->host_without_cache + kCacheFloor) + " of private host memory, over the " +
                          gib(host_limit_) + " limit. Close other applications or raise the host budget.",
                      host_lean->arena, cap_, host_lean->host_without_cache + kCacheFloor, host_limit_);
  }

  const PlanInputs& in_;
  const ModelShapes& s_;
  const RequestShape& r_;
  const HardwareProfile& hw_;
  std::size_t cap_ = 0;
  std::size_t host_limit_ = 0;
  std::size_t host_target_ = 0;
  std::size_t system_limit_ = 0;  // what the system allows (MemAvailable based) even under an explicit host budget
};

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += ch;
    } else if (c == '\n') {
      out += "\\n";
    } else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += ch;
    }
  }
  return out + "\"";
}

// Locale-independent (to_chars ignores LC_NUMERIC); JSON has no inf/nan.
std::string json_num(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[32];
  const auto res = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::general, 6);
  return std::string(buf, res.ptr);
}

const char* b2s(bool b) { return b ? "true" : "false"; }

// The text-encoder fields of ModelShapes from the text-encoder groups of `index`.
void te_shapes(const weights::ModelIndex& index, ModelShapes& s) {
  for (const weights::WeightGroup* g : index.groups_of(weights::Component::kTextEncoder)) {
    const std::size_t bytes = static_cast<std::size_t>(g->total_bytes);
    if (g->name == "embed_tokens") {
      s.te_embed_bytes = bytes;
    } else {
      s.te_stream_bytes += bytes;
      if (g->name.rfind("layers.", 0) == 0) {
        ++s.te_layers;
        s.te_layer_bytes = std::max(s.te_layer_bytes, bytes);
      }
    }
  }
}

// The DiT fields of ModelShapes from the transformer groups of `index`.
void dit_shapes(const weights::ModelIndex& index, ModelShapes& s) {
  for (const weights::WeightGroup* g : index.groups_of(weights::Component::kTransformer)) {
    const std::size_t bytes = static_cast<std::size_t>(g->total_bytes);
    s.dit_bytes += bytes;
    s.dit_params += static_cast<std::size_t>(g->total_params);
    // An NVFP4 linear (U8 weight [N, K/2] and U8 block scales; no other DiT tensor is U8): N * K parameters.
    for (const weights::IndexedTensor& t : g->tensors) {
      if (t.dtype != weights::DType::kU8) continue;
      if (t.name.ends_with(".weight")) s.dit_params += static_cast<std::size_t>(t.numel());
      if (t.name.ends_with(".weight_scale")) s.dit_params -= static_cast<std::size_t>(t.numel());
    }
    if (g->name.rfind("block.", 0) == 0) {
      ++s.dit_blocks;
      s.dit_block_bytes = std::max(s.dit_block_bytes, bytes);
      // The width is the length of the block's RMSNorm weights (an INT8 block's 1-D per-row scales are wider).
      for (const weights::IndexedTensor& t : g->tensors) {
        if (t.shape.size() == 1 && t.name.ends_with("norm1.weight")) s.dit_dim = std::max(s.dit_dim, static_cast<int>(t.shape[0]));
      }
    }
  }
}

}  // namespace

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kMinMemory: return "min_memory";
    case Mode::kBalanced: return "balanced";
    case Mode::kMaxSpeed: return "max_speed";
  }
  return "?";
}

Mode parse_mode(std::string_view name) {
  if (name == "min_memory") return Mode::kMinMemory;
  if (name == "balanced") return Mode::kBalanced;
  if (name == "max_speed") return Mode::kMaxSpeed;
  throw std::invalid_argument("mode must be min_memory, balanced or max_speed, got \"" + std::string(name) + "\"");
}

const char* te_host_pin_name(TeHostPin p) {
  switch (p) {
    case TeHostPin::kAuto: return "auto";
    case TeHostPin::kOn: return "on";
    case TeHostPin::kOff: return "off";
  }
  return "?";
}

TeHostPin parse_te_host_pin(std::string_view name) {
  if (name == "auto") return TeHostPin::kAuto;
  if (name == "on") return TeHostPin::kOn;
  if (name == "off") return TeHostPin::kOff;
  throw std::invalid_argument("text-encoder pinning must be auto, on or off, got \"" + std::string(name) + "\"");
}

const char* residency_name(DitResidency r) {
  switch (r) {
    case DitResidency::kResident: return "resident";
    case DitResidency::kParked: return "parked";
    case DitResidency::kStreamed: return "streamed";
  }
  return "?";
}

const char* source_name(DitSource s) {
  switch (s) {
    case DitSource::kFile: return "file";
    case DitSource::kHostPinned: return "host_pinned";
  }
  return "?";
}

const char* precision_name(Precision p) {
  switch (p) {
    case Precision::kBf16: return "bf16";
    case Precision::kInt8: return "int8";
    case Precision::kInt4: return "int4";
    case Precision::kNvfp4: return "nvfp4";
  }
  return "?";
}

const char* attention_name(AttentionMode a) {
  switch (a) {
    case AttentionMode::kExact: return "exact";
    case AttentionMode::kQ8Fp16: return "q8_fp16";
    case AttentionMode::kQ8Fp16F32Acc: return "q8_fp16_f32acc";
    case AttentionMode::kQ8Fp8: return "q8_fp8";
    case AttentionMode::kQ8Fp8F32Acc: return "q8_fp8_f32acc";
  }
  return "?";
}

bool is_q8(AttentionMode a) { return a != AttentionMode::kExact; }

const char* tier_name(Tier t) {
  switch (t) {
    case Tier::kReference: return "reference";
    case Tier::kHigh: return "high";
    case Tier::kTurbo: return "turbo";
  }
  return "?";
}

Tier parse_tier(std::string_view name) {
  for (const Tier t : {Tier::kReference, Tier::kHigh, Tier::kTurbo}) {
    if (name == tier_name(t)) return t;
  }
  throw std::invalid_argument("tier must be reference, high or turbo, got \"" + std::string(name) + "\"");
}

AttentionMode parse_attention(std::string_view name) {
  if (name == "q8") return AttentionMode::kQ8Fp8;
  for (const AttentionMode a : {AttentionMode::kExact, AttentionMode::kQ8Fp16, AttentionMode::kQ8Fp16F32Acc,
                                AttentionMode::kQ8Fp8, AttentionMode::kQ8Fp8F32Acc}) {
    if (name == attention_name(a)) return a;
  }
  throw std::invalid_argument("attention must be exact, q8 (= q8_fp8), q8_fp16, q8_fp16_f32acc, q8_fp8 or "
                              "q8_fp8_f32acc, got \"" + std::string(name) + "\"");
}

TierDecision resolve_tier(Tier requested, std::optional<AttentionMode> attention, Precision precision,
                          const AttentionSupported& supported, bool te_quantized, bool nvfp4_native) {
  TierDecision d;
  d.tier = requested;
  d.precision = precision;
  // Lossy weights fix the configuration's precision: a quantized DiT or text encoder is never the Reference tier, and
  // an INT4 or NVFP4 DiT is always the Turbo tier.
  const bool nvfp4 = precision == Precision::kNvfp4;
  const bool int4 = precision == Precision::kInt4 || nvfp4;  // the Turbo tier's 4-bit GEMMs
  const bool int8 = precision != Precision::kBf16 || te_quantized;
  const char* const dit = nvfp4 ? "NVFP4" : "INT4 ConvRot";
  const std::string weights = std::string(nvfp4                               ? "NVFP4 GEMMs"
                                          : precision == Precision::kInt4 ? "INT4 ConvRot GEMMs"
                                          : precision == Precision::kInt8 ? "INT8 ConvRot GEMMs"
                                                                          : "bf16 GEMMs") +
                              (te_quantized ? ", quantized text encoder" : "");
  if (nvfp4 && !nvfp4_native) {
    throw std::invalid_argument("the loaded DiT is NVFP4, which needs an RTX 50 GPU (compute capability 12.0) and a build "
                                "with the sm_120a NVFP4 kernels; use the INT8 file");
  }
  if (requested == Tier::kTurbo && !int4) {
    throw std::invalid_argument(std::string("tier turbo runs an INT4 ConvRot DiT (a convrot_w4a4 file) or an NVFP4 DiT "
                                            "loaded as the DiT, but the loaded DiT is ") +
                                precision_name(precision) + "; use reference or high");
  }
  if (requested == Tier::kHigh && int4) {
    throw std::invalid_argument(std::string("the loaded DiT is ") + dit + ", which runs the Turbo tier; use tier turbo");
  }
  if (attention && is_q8(*attention) && supported && !supported(*attention)) {
    throw std::invalid_argument(std::string("attention ") + attention_name(*attention) +
                                " is not supported on this device/build");
  }
  if (requested == Tier::kReference) {
    if (int8) {
      throw std::invalid_argument(std::string("tier reference runs bf16 weights exactly, but the loaded DiT or text encoder "
                                              "file is quantized; load the bundle's bf16 models or use tier ") +
                                  (int4 ? "turbo" : "high"));
    }
    if (attention && is_q8(*attention)) {
      throw std::invalid_argument("tier reference uses exact attention; 8-bit attention belongs to tier high");
    }
    d.reason = "Reference tier: bf16 GEMMs, exact attention";
    return d;
  }
  // High and Turbo: the tier's default attention (kTierAttention) unless the caller gave one; never a silent fallback.
  const char* const tier = requested == Tier::kTurbo ? "turbo" : "high";
  if (attention) {
    d.attention = *attention;
  } else {
    const auto it = std::find_if(std::begin(kTierAttention), std::end(kTierAttention),
                                 [&](AttentionMode m) { return !supported || supported(m); });
    if (it == std::end(kTierAttention)) {
      throw std::invalid_argument(std::string("tier ") + tier + " runs 8-bit attention, which this device/build lacks; "
                                  "pass attention exact explicitly");
    }
    d.attention = *it;
  }
  if (!int8 && d.attention == AttentionMode::kExact) {
    throw std::invalid_argument("tier high with attention exact on a bf16 DiT and text encoder is the Reference "
                                "configuration; use tier reference, or 8-bit attention");
  }
  d.reason = std::string(requested == Tier::kTurbo ? "Turbo tier" : "High tier") + ": " + weights + ", " +
             attention_name(d.attention) + " attention";
  return d;
}

const char* residency_name(TeResidency r) {
  switch (r) {
    case TeResidency::kStreamed: return "streamed";
    case TeResidency::kHostResident: return "host_resident";
    case TeResidency::kVramResident: return "vram_resident";
  }
  return "?";
}

BudgetError::BudgetError(const std::string& message, std::size_t required_arena_bytes, std::size_t arena_cap_bytes,
                         std::size_t required_host_bytes, std::size_t host_limit_bytes)
    : std::runtime_error(message),
      required_arena(required_arena_bytes),
      arena_cap(arena_cap_bytes),
      required_host(required_host_bytes),
      host_limit(host_limit_bytes) {}

ModelShapes shapes_from_index(const weights::ModelIndex& index, const weights::ModelIndex* dit_index, Precision dit_precision,
                              const weights::ModelIndex* te_index, const weights::ModelIndex* vae_index) {
  // Each component from its own index: the bundle's need not hold those given separately (weights::BuildOptions::skip).
  ModelShapes s;
  te_shapes(te_index != nullptr ? *te_index : index, s);
  dit_shapes(dit_index != nullptr ? *dit_index : index, s);
  for (const weights::WeightGroup* g : (vae_index != nullptr ? *vae_index : index).groups_of(weights::Component::kVae)) {
    s.vae_bytes += static_cast<std::size_t>(g->total_bytes);
  }
  s.dit_precision = dit_precision;
  if (dit_index != nullptr && s.dit_blocks == 0) throw std::invalid_argument("DiT index lacks the DiT blocks");
  if (te_index != nullptr && s.te_layers == 0) throw std::invalid_argument("text-encoder index lacks the decoder layers");
  if (vae_index != nullptr && s.vae_bytes == 0) throw std::invalid_argument("VAE index lacks the decoder");
  if (s.te_layers == 0 || s.dit_blocks == 0 || s.vae_bytes == 0) {
    throw std::invalid_argument("model index lacks the text encoder layers, the DiT blocks or the VAE decoder");
  }
  return s;
}

ModelShapes shapes_from_index(const weights::ModelIndex& index) {
  ModelShapes s;
  te_shapes(index, s);
  dit_shapes(index, s);
  for (const weights::WeightGroup* g : index.groups_of(weights::Component::kVae)) s.vae_bytes += static_cast<std::size_t>(g->total_bytes);
  if (s.te_layers == 0 || s.dit_blocks == 0 || s.vae_bytes == 0) {
    throw std::invalid_argument("model index lacks the text encoder layers, the DiT blocks or the VAE decoder");
  }
  return s;
}

bool weights_cached_from_stream(std::uint64_t bytes, double load_s, double wait_s, const HardwareProfile& hw) {
  if (bytes == 0 || load_s <= 0) return true;
  const double cached_bps = std::min(hw.host_gbps, hw.h2d_gbps) * 1e9;
  const bool slow = static_cast<double>(bytes) / load_s < 0.5 * cached_bps;
  const bool stalled = wait_s > 0.1 * load_s;
  return !(slow && stalled);
}

bool warm_dit_load_direct(double direct_gbps, double buffered_warm_gbps, bool direct_before) {
  if (direct_before) return buffered_warm_gbps <= direct_gbps * (1.0 + kWarmLoadHysteresis);
  return direct_gbps > buffered_warm_gbps * (1.0 + kWarmLoadHysteresis);
}

std::size_t arena_cap(const Budgets& b) {
  std::size_t cap = sat_sub(b.vram_free, b.vram_reserve);
  if (b.vram_budget != 0) cap = std::min(cap, sat_sub(b.vram_budget, b.context_bytes + b.vram_reserve));
  return cap;
}

ExecutionPlan plan(const PlanInputs& inputs) { return Planner(inputs).run(); }

namespace {

// Same device residency of the DiT: every block resident (resident or parked), or streamed alike.
bool same_residency(const DitPlan& a, const DitPlan& b) {
  const bool sa = a.residency == DitResidency::kStreamed, sb = b.residency == DitResidency::kStreamed;
  if (!sa || !sb) return sa == sb;
  return a.resident_blocks == b.resident_blocks && a.buffers == b.buffers && a.source == b.source;
}

}  // namespace

ExecutionPlan prefer_held(const PlanInputs& inputs, ExecutionPlan fresh, const HeldDit& held, bool* kept) {
  if (kept != nullptr) *kept = false;
  if (same_residency(fresh.dit, held.dit) && fresh.arena_bytes <= held.arena_bytes) return fresh;  // no reload
  PlanInputs in = inputs;
  // Arena + graph cap = the held arena (vram_free already counts it as free) and the held graphs (it does not).
  in.budgets.vram_free += held.graph_bytes;
  in.budgets.vram_budget = held.arena_bytes + held.graph_bytes + in.budgets.context_bytes + in.budgets.vram_reserve;
  if (inputs.budgets.vram_budget != 0) in.budgets.vram_budget = std::min(in.budgets.vram_budget, inputs.budgets.vram_budget);
  DitPlan keep;
  if (held.dit.residency == DitResidency::kStreamed) {
    keep.residency = DitResidency::kStreamed;
    keep.resident_blocks = held.dit.resident_blocks;
    keep.buffers = held.dit.buffers;
    keep.source = held.dit.source;
  }
  in.keep_dit = keep;
  ExecutionPlan k;
  try {
    k = plan(in);
  } catch (const BudgetError&) {
    return fresh;  // the request does not fit the held arena with the loaded DiT
  }
  if (!same_residency(k.dit, held.dit) || k.arena_bytes > held.arena_bytes) return fresh;
  const double reload_s = Planner(inputs).load_seconds(fresh.dit);
  const double saved_s = k.time.total_s() - fresh.time.total_s();
  if (saved_s > reload_s) {
    char why[200];
    std::snprintf(why, sizeof(why), "\nReloads the DiT (~%.1f s) to save ~%.1f s against the loaded one in the held %.2f GiB arena.",
                  reload_s, saved_s, static_cast<double>(held.arena_bytes) / static_cast<double>(kGiB));
    fresh.explanation += why;
    return fresh;
  }
  char why[200];
  std::snprintf(why, sizeof(why),
                "\nPlanned inside the held %.2f GiB arena with the loaded DiT: the %.2f GiB plan would reload it (~%.1f s) to save ~%.1f s.",
                static_cast<double>(held.arena_bytes) / static_cast<double>(kGiB),
                static_cast<double>(fresh.arena_bytes) / static_cast<double>(kGiB), reload_s, std::max(0.0, saved_s));
  k.explanation += why;
  if (kept != nullptr) *kept = true;
  return k;
}

std::string ExecutionPlan::to_json() const {
  std::string o = "{";
  o += "\"mode\": \"" + std::string(mode_name(mode)) + "\"";
  o += ", \"dit\": {\"residency\": \"" + std::string(residency_name(dit.residency)) + "\", \"precision\": \"" + precision_name(dit.precision) +
       "\", \"attention\": \"" + attention_name(dit.attention) + "\"" +
       ", \"resident_blocks\": " + std::to_string(dit.resident_blocks) + ", \"buffers\": " + std::to_string(dit.buffers) +
       ", \"prefetch_blocks\": " + std::to_string(dit.prefetch_blocks) + ", \"source\": \"" + source_name(dit.source) +
       "\", \"stream_bytes_per_step\": " + std::to_string(dit.stream_bytes) + ", \"transfer_s_per_step\": " +
       json_num(dit.transfer_s) + ", \"compute_s_per_step\": " + json_num(dit.compute_s) + ", \"expected_hidden\": " +
       json_num(dit.hidden) + ", \"park_for_text_encoder\": " + b2s(dit.park_for_text_encoder) +
       ", \"park_for_decode\": " + b2s(dit.park_for_decode) + ", \"device_bytes\": " + std::to_string(dit.device_bytes) +
       ", \"graph_bytes\": " + std::to_string(dit.graph_bytes) + "}";
  o += ", \"text_encoder\": {\"run\": " + std::string(b2s(te.run)) + ", \"residency\": \"" + residency_name(te.residency) +
       "\", \"buffers\": " + std::to_string(te.buffers) + ", \"pinned\": " + b2s(te.pinned) + "}";
  o += ", \"vae\": {\"run\": " + std::string(b2s(vae.run)) + ", \"tiled\": " + b2s(vae.kind == VaeDecodeKind::kTiled) +
       ", \"tile\": " + std::to_string(vae.tile) + ", \"stack_bytes\": " + std::to_string(vae.stack_bytes) +
       ", \"relative_cost\": " + json_num(vae.cost) + "}";
  o += ", \"embedding_cache\": {\"enabled\": " + std::string(b2s(cache.enabled)) + ", \"host_bytes\": " +
       std::to_string(cache.host_bytes) + ", \"keep_negative\": " + b2s(cache.keep_negative) + ", \"disk_spill\": " +
       b2s(cache.disk_spill) + "}";
  o += ", \"peaks\": {\"text_encoder\": " + std::to_string(peaks.text_encoder) + ", \"denoise\": " +
       std::to_string(peaks.denoise) + ", \"decode\": " + std::to_string(peaks.decode) + "}";
  o += ", \"arena_bytes\": " + std::to_string(arena_bytes) + ", \"device_peak_bytes\": " + std::to_string(device_peak_bytes) +
       ", \"arena_cap_bytes\": " + std::to_string(arena_cap_bytes) + ", \"host_private_bytes\": " +
       std::to_string(host_private_bytes) + ", \"host_target_bytes\": " + std::to_string(host_target_bytes) +
       ", \"host_held_target_bytes\": " + std::to_string(host_held_target_bytes) +
       ", \"host_limit_bytes\": " + std::to_string(host_limit_bytes);
  o += ", \"time_s\": {\"text_encoder\": " + json_num(time.text_encoder_s) + ", \"dit_load\": " + json_num(time.dit_load_s) +
       ", \"denoise\": " + json_num(time.denoise_s) + ", \"decode\": " + json_num(time.decode_s) + ", \"total\": " +
       json_num(time.total_s()) + "}";
  o += ", \"explanation\": " + json_string(explanation) + "}";
  return o;
}

}  // namespace mi::planner
