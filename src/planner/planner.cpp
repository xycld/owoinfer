#include "planner/planner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

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
// Fraction of a streamed DiT step's copy time hidden behind compute when the copy is shorter (AC10 asks >= 80%);
// the rest (synchronization, memory-bandwidth contention) adds to the step, so resident blocks always pay off.
constexpr double kStreamOverlap = 0.9;

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
    if (b.host_budget != 0) {
      host_limit_ = host_target_ = b.host_budget;
    } else {
      host_limit_ = b.host_available == 0 ? std::numeric_limits<std::size_t>::max()
                                          : sat_sub(b.host_private_now + b.host_available, kHostHeadroom);
      const std::size_t preset = in_.mode == Mode::kMinMemory ? kMinMemoryHostTarget
                                 : in_.mode == Mode::kBalanced ? kBalancedHostTarget
                                                               : host_limit_;
      host_target_ = std::min(preset, host_limit_);
    }
  }

  ExecutionPlan run() {
    std::vector<Candidate> all = enumerate();
    std::vector<const Candidate*> feasible;
    for (const Candidate& c : all) {
      if (c.arena <= cap_ && c.host_without_cache + kCacheFloor <= host_limit_) feasible.push_back(&c);
    }
    if (feasible.empty()) throw_budget_error(all);
    const Candidate* best = *std::min_element(feasible.begin(), feasible.end(),
                                              [&](const Candidate* a, const Candidate* b) { return rank(*a) < rank(*b); });
    return make_plan(*best, all);
  }

 private:
  bool te_runs() const { return !r_.text_lengths.empty() && !r_.te_cached; }

  std::size_t dit_nonblock_bytes() const {
    return sat_sub(s_.dit_bytes, static_cast<std::size_t>(s_.dit_blocks) * s_.dit_block_bytes);
  }

  // DiT device bytes outside the denoise (persistent) and during it (+ streaming buffers).
  std::size_t dit_persistent(const DitPlan& d) const {
    if (d.residency != DitResidency::kStreamed) return s_.dit_bytes;
    return dit_nonblock_bytes() + static_cast<std::size_t>(d.resident_blocks) * s_.dit_block_bytes;
  }
  std::size_t dit_denoise(const DitPlan& d) const {
    if (d.residency != DitResidency::kStreamed) return s_.dit_bytes;
    return dit_persistent(d) + static_cast<std::size_t>(d.prefetch_blocks + 1) * s_.dit_block_bytes;
  }

  double weight_rate_gbps(bool from_host) const {
    const double src = from_host || r_.weights_cached ? hw_.host_gbps : hw_.disk_gbps;
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
    const double rate = weight_rate_gbps(te.residency == TeResidency::kHostResident) * 1e9;
    const double load = static_cast<double>(s_.te_stream_bytes) / rate;
    // One buffer: every layer's copy and compute in series. Two or more: copies overlap compute except for one
    // layer's worth (the first copy or the last compute).
    if (te.buffers >= 2) return std::max(load, compute) + std::min(load, compute) / std::max(1, s_.te_layers);
    return load + compute;
  }

  double denoise_seconds(const DitPlan& d) const {
    const double image_tokens = static_cast<double>(r_.height / 16) * static_cast<double>(r_.width / 16);
    const double dim = s_.dit_dim > 0 ? s_.dit_dim : 3072;
    double flops = 0;
    for (int row = 0; row < r_.rows_per_step; ++row) {
      // Row 1 is the negative prompt; rows 0 and 2 (SLG) the prompt.
      const std::size_t text_row = row == 1 && r_.text_lengths.size() > 1 ? 1 : 0;
      const double l = image_tokens + (r_.text_lengths.empty() ? 0.0 : static_cast<double>(r_.text_lengths[text_row]));
      // GEMMs: 2 FLOPs per weight per token; attention: QK^T and PV, 2 * L^2 * dim each, per block.
      flops += static_cast<double>(s_.dit_bytes) * l + 4.0 * l * l * dim * s_.dit_blocks;
    }
    double step = flops / (tflops() * 1e12);
    if (d.residency == DitResidency::kStreamed) {
      const double moved = static_cast<double>(s_.dit_blocks - d.resident_blocks) * static_cast<double>(s_.dit_block_bytes);
      const double transfer = moved / (weight_rate_gbps(false) * 1e9);
      step = std::max(step, transfer) + (1.0 - kStreamOverlap) * std::min(step, transfer);
    }
    return step * r_.steps;
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
        // The fastest schedule in what the resident models leave of the cap; if none fits, the smallest (then the
        // candidate is infeasible, but still describes the requirement).
        std::optional<TiledDecode> t;
        const std::size_t room = sat_sub(cap_, resident + kArenaSlack);
        if (!c.vae.smallest && room > 0) t = in_.workspaces.decode_tiled(room);
        if (!t) t = in_.workspaces.decode_tiled(0);
        if (!t) throw std::invalid_argument("planner: the tiled decode query has no smallest schedule");
        c.vae.stack_bytes = t->bytes;
        c.vae.tile = t->tile;
        c.vae.cost = t->cost;
      }
      c.peaks.decode = resident + c.vae.stack_bytes;
    }
    c.dit.device_bytes = dit_denoise(c.dit);
    c.arena = std::max({c.peaks.text_encoder, c.peaks.denoise, c.peaks.decode}) + kArenaSlack;

    std::size_t host = in_.budgets.host_private_now + in_.workspaces.host_buffers;
    if (c.te.run && c.te.residency != TeResidency::kVramResident) host += in_.caps.te_ring_bytes;
    if (c.te.residency == TeResidency::kHostResident) host += s_.te_stream_bytes;
    // Streamed DiT blocks are read from the weight files (page cache: file-backed, not private) into the ring.
    if (c.dit.residency == DitResidency::kStreamed) host += in_.caps.dit_ring_bytes;
    c.host_without_cache = host;

    c.time.text_encoder_s = te_seconds(c.te);
    c.time.dit_load_s = c.parks() ? static_cast<double>(s_.dit_bytes) / (weight_rate_gbps(false) * 1e9) : 0;
    c.time.denoise_s = denoise_seconds(c.dit);
    c.time.decode_s = decode_seconds(c.vae);
  }

  std::vector<Candidate> enumerate() const {
    std::vector<TePlan> tes;
    if (!te_runs()) {
      tes.push_back({false, TeResidency::kStreamed, 0});
    } else {
      const int max_buffers = std::max(1, in_.caps.te_max_buffers);
      for (int b = 1; b <= max_buffers; ++b) tes.push_back({true, TeResidency::kStreamed, b});
      if (in_.caps.te_host_resident) tes.push_back({true, TeResidency::kHostResident, max_buffers});
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
        if (in_.caps.dit_streaming) {
          // P4: no resident blocks (least memory), and as many as fit the arena cap (peaks grow linearly with them).
          Candidate c{{}, te, vae, {}, 0, 0, {}};
          c.dit.residency = DitResidency::kStreamed;
          c.dit.prefetch_blocks = std::max(1, in_.caps.dit_prefetch_blocks);
          evaluate(c);
          out.push_back(c);
          if (c.arena < cap_ && s_.dit_block_bytes > 0) {
            const std::size_t fit = (cap_ - c.arena) / s_.dit_block_bytes;
            const int extra = static_cast<int>(std::min<std::size_t>(fit, static_cast<std::size_t>(s_.dit_blocks - 1)));
            if (extra > 0) {
              c.dit.resident_blocks = extra;
              evaluate(c);
              out.push_back(c);
            }
          }
        }
      }
    }
    return out;
  }

  std::size_t cache_bytes(const Candidate& c) const {
    const std::size_t preset = in_.mode == Mode::kMinMemory ? kCacheMinMemory
                               : in_.mode == Mode::kBalanced ? kCacheBalanced
                                                             : kCacheMaxSpeed;
    return std::max(kCacheFloor, std::min(preset, sat_sub(host_target_, c.host_without_cache)));
  }

  using RankKey = std::tuple<bool, bool, std::size_t, std::size_t, double, double>;
  RankKey rank(const Candidate& c) const {
    const bool over_target = c.host_without_cache + cache_bytes(c) > host_target_;
    const std::size_t host = c.host_without_cache + cache_bytes(c);
    if (in_.mode == Mode::kMinMemory) {
      return {over_target, false, c.arena, host, c.time.total_s(), 0.0};
    }
    return {over_target, c.parks(), 0, 0, c.time.total_s(), static_cast<double>(c.arena)};
  }

  std::string describe_dit(const DitPlan& d) const {
    switch (d.residency) {
      case DitResidency::kResident:
        return "resident bf16 (" + gib(s_.dit_bytes) + "), never parked";
      case DitResidency::kParked: {
        std::string when = d.park_for_text_encoder && d.park_for_decode ? "the text encoder and the decode"
                           : d.park_for_text_encoder                    ? "the text encoder"
                                                                        : "the decode";
        return "parked bf16 (" + gib(s_.dit_bytes) + "): evicted for " + when + " and reloaded for the next denoise";
      }
      case DitResidency::kStreamed:
        return "streamed bf16: " + std::to_string(d.resident_blocks) + " of " + std::to_string(s_.dit_blocks) +
               " blocks resident, the rest stream with " + std::to_string(d.prefetch_blocks) + "-block lookahead (" +
               gib(d.device_bytes) + " on the device)";
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
        return "host-resident (" + gib(s_.te_stream_bytes) + " private), streamed with " + std::to_string(t.buffers) +
               " layer buffers";
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
            c.dit.resident_blocks == best.dit.resident_blocks && c.te.residency == best.te.residency &&
            c.te.buffers == best.te.buffers && c.parks() == best.parks()) {
          s += "\nUntiled decode would need a " + gib(c.arena) + " arena (decode peak " + gib(c.peaks.decode) +
               "), over the " + gib(cap_) + " cap; tiling keeps the DiT resident.";
          break;
        }
      }
    }
    if (best.parks()) {
      s += "\nParking is the fallback: no configuration keeping the DiT resident fits the " + gib(cap_) + " cap.";
    }
    return s;
  }

  ExecutionPlan make_plan(const Candidate& c, const std::vector<Candidate>& all) const {
    ExecutionPlan p;
    p.mode = in_.mode;
    p.dit = c.dit;
    p.te = c.te;
    p.vae = c.vae;
    p.cache.enabled = true;
    p.cache.host_bytes = cache_bytes(c);
    p.cache.keep_negative = true;
    p.cache.disk_spill = in_.cache_disk_spill;
    p.peaks = c.peaks;
    p.arena_bytes = c.arena;
    p.device_peak_bytes = c.arena + in_.budgets.context_bytes;
    p.arena_cap_bytes = cap_;
    p.host_private_bytes = c.host_without_cache + p.cache.host_bytes;
    p.host_target_bytes = host_target_;
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
    p.explanation = std::move(e);
    return p;
  }

  [[noreturn]] void throw_budget_error(const std::vector<Candidate>& all) const {
    const Candidate* lean = &all.front();
    for (const Candidate& c : all) {
      if (std::make_pair(c.arena, c.host_without_cache) < std::make_pair(lean->arena, lean->host_without_cache)) lean = &c;
    }
    const std::string config = "DiT " + describe_dit(lean->dit) + "; text encoder " + describe_te(lean->te) + "; VAE " +
                               describe_vae(lean->vae) + "; " + describe_peaks(*lean);
    if (lean->arena > cap_) {
      throw BudgetError("not enough GPU memory for " + request_text() + " in any mode: the leanest plan needs a " +
                            gib(lean->arena) + " arena (" + config + "), but only " + gib(cap_) + " is available (" +
                            budget_text() + "). Close other GPU applications, raise the budget or lower the resolution.",
                        lean->arena, cap_, lean->host_without_cache + kCacheFloor, host_limit_);
    }
    const Candidate* host_lean = nullptr;
    for (const Candidate& c : all) {
      if (c.arena <= cap_ && (host_lean == nullptr || c.host_without_cache < host_lean->host_without_cache)) host_lean = &c;
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

std::string json_num(double d) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", d);
  return buf;
}

const char* b2s(bool b) { return b ? "true" : "false"; }

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

const char* residency_name(DitResidency r) {
  switch (r) {
    case DitResidency::kResident: return "resident";
    case DitResidency::kParked: return "parked";
    case DitResidency::kStreamed: return "streamed";
  }
  return "?";
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

ModelShapes shapes_from_index(const weights::ModelIndex& index) {
  ModelShapes s;
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
  for (const weights::WeightGroup* g : index.groups_of(weights::Component::kTransformer)) {
    const std::size_t bytes = static_cast<std::size_t>(g->total_bytes);
    s.dit_bytes += bytes;
    if (g->name.rfind("block.", 0) == 0) {
      ++s.dit_blocks;
      s.dit_block_bytes = std::max(s.dit_block_bytes, bytes);
      // The width is the length of the block's widest 1-D tensor (its RMSNorm weights; QK-norms are per head).
      for (const weights::IndexedTensor& t : g->tensors) {
        if (t.shape.size() == 1) s.dit_dim = std::max(s.dit_dim, static_cast<int>(t.shape[0]));
      }
    }
  }
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

std::size_t arena_cap(const Budgets& b) {
  std::size_t cap = sat_sub(b.vram_free, b.vram_reserve);
  if (b.vram_budget != 0) cap = std::min(cap, sat_sub(b.vram_budget, b.context_bytes + b.vram_reserve));
  return cap;
}

ExecutionPlan plan(const PlanInputs& inputs) { return Planner(inputs).run(); }

std::string ExecutionPlan::to_json() const {
  std::string o = "{";
  o += "\"mode\": \"" + std::string(mode_name(mode)) + "\"";
  o += ", \"dit\": {\"residency\": \"" + std::string(residency_name(dit.residency)) + "\", \"precision\": \"bf16\"" +
       ", \"resident_blocks\": " + std::to_string(dit.resident_blocks) + ", \"prefetch_blocks\": " +
       std::to_string(dit.prefetch_blocks) + ", \"park_for_text_encoder\": " + b2s(dit.park_for_text_encoder) +
       ", \"park_for_decode\": " + b2s(dit.park_for_decode) + ", \"device_bytes\": " + std::to_string(dit.device_bytes) + "}";
  o += ", \"text_encoder\": {\"run\": " + std::string(b2s(te.run)) + ", \"residency\": \"" + residency_name(te.residency) +
       "\", \"buffers\": " + std::to_string(te.buffers) + "}";
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
       ", \"host_limit_bytes\": " + std::to_string(host_limit_bytes);
  o += ", \"time_s\": {\"text_encoder\": " + json_num(time.text_encoder_s) + ", \"dit_load\": " + json_num(time.dit_load_s) +
       ", \"denoise\": " + json_num(time.denoise_s) + ", \"decode\": " + json_num(time.decode_s) + ", \"total\": " +
       json_num(time.total_s()) + "}";
  o += ", \"explanation\": " + json_string(explanation) + "}";
  return o;
}

}  // namespace mi::planner
