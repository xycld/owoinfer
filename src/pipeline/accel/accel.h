// P6b algorithmic accelerators: opt-in, lossy (they change the image), off by default. With every accelerator off the
// sampling loop runs exactly as without this module (bit-identical output).
//
// CFG truncation (Options::cfg_truncation = c in [0, 1]; 1 = off): classifier-free guidance only for the steps i with
// i / num_steps < c (the guidance windows' step-index convention, pipeline/sampler.h); the later steps run the cond row
// alone (no uncond row, no SLG row) and sample with the cond prediction. c = 0.8 drops the uncond row for the last 20 %
// of the steps (~10 % fewer DiT rows over a 50-step run). No effect without CFG (cfg == 1). The fraction is of the
// steps the pass actually runs (an img2img pass or a hires fix's second pass: of its strength-cropped schedule, as the
// guidance windows). Multistep samplers (ab2, dpmpp_2m, res_multistep, er_sde, dpmpp_2m_sde) keep the last guided
// prediction in their history, so the first unguided step mixes one guided step, as ComfyUI's CFG cutoff does. Measured (1024^2, 50
// steps, CFG 3, 4 prompts, vs the same seed without it): 0.8 -> 1.06x denoise, mean LPIPS 0.004; 0.6 -> 1.22x, mean
// LPIPS 0.014 (max 0.022), DINOv2 >= 0.99: both well inside the High tier's same-seed bounds; 0.6-0.8 are safe values.
//
// Step caching (Options::step_cache): kFbCache, the First-Block Cache of ParaAttention (chengzeyi/ParaAttention,
// para_attn/first_block_cache, "First Block Cache", 2024; the same method as ComfyUI-WaveSpeed's apply_first_block_cache),
// on the MOSS DiT through moss_dit::StepCacheHook. Every DiT call runs block 0; its output residual r0 = x1 - x0 on the
// video tokens of every row is compared with r0 of the last call that ran every block:
//   diff = sum |r0 - r0_prev| / sum |r0_prev|   (summed over all rows of the call: the batched-CFG semantics)
// diff < threshold reuses the cache: blocks 1..41 do not run (nor are streamed), and each row's video tokens become
// x1 + R with R the row's residual of blocks 1..41 from that last full call (R = x_41 - x1). Otherwise every block runs
// and r0_prev and R are refreshed. Residuals are bf16, as the reference keeps them in the model dtype. The comparison
// is always against the last full call, so the drift accumulates and forces a recompute. Each row (identified by its
// text tensor and skip set: cond, uncond, SLG) has its own residuals; a call with a row the cache has not computed yet
// (the first call, the SLG row entering its window, a changed skip set) runs every block, as does the call after
// invalidate() (the pipeline's: a control input's step window opened or closed); a row absent from a call loses its
// residuals, so a row identity that comes back never gets a stale R. The decision needs one
// device-to-host synchronization per call. Works with every block path (CUDA graphs: block 0 replays, the hook runs
// between graphs), streamed blocks (a reuse ends the streaming pass after block 0), LoRA side branches and INT8/INT4
// blocks: the hook only reads and writes the residual stream.
// TeaCache (Liu et al. 2024, "Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model") is not offered:
// its skip test rescales the modulated-input distance with a polynomial fitted per model, and no fit for MOSS-Image
// exists; FBCache needs no calibration.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/tensor.h"
#include "models/moss_dit/moss_dit.h"
#include "runtime/device_arena.h"

namespace mi::pipeline::accel {

enum class StepCacheKind { kOff, kFbCache };
const char* step_cache_name(StepCacheKind kind);
// "off" or "fbcache"; std::invalid_argument otherwise ("teacache" names why it is not offered).
StepCacheKind parse_step_cache(std::string_view name);

// The FBCache threshold recommended for the SFT model and the files derived from it (and any model the engine cannot
// place), the default when Options::step_cache_threshold is unset. Measured at 1024^2, 50 steps, CFG 3, 4 prompts
// against the same seed without the cache (2026-10-22): 0.02 reuses almost nothing; 0.03 reuses 30 % of the DiT calls
// (1.25x denoise) at mean LPIPS 0.038 (max 0.113), mean DINOv2 0.975; 0.05 / 0.08 / 0.12 reach 1.84x / 2.31x / 2.91x,
// and the images stay coherent, but the composition can move away from the uncached image's (min DINOv2 0.43-0.67).
inline constexpr double kDefaultFbCacheThreshold = 0.03;
// The highest FBCache threshold that keeps the base model's composition (RTX 5090, 2026-10-23, the same 4 prompts and
// seeds against the uncached image: base min DINOv2 0.77 at 0.03 and 0.53 at 0.05; the SFT model stays at 0.93 or
// above at both, bf16 and INT8): the default on the base model (pipeline.h dit_lineage). A threshold the user set is
// never changed: one above this on the base model gets a warning (Result::accel_warnings).
inline constexpr double kBaseModelFbCacheThreshold = 0.02;
// The default threshold for a DiT of `lineage` ("base", "sft" or "": pipeline.h dit_lineage).
double default_fbcache_threshold(std::string_view lineage);

struct Options {
  double cfg_truncation = 1.0;  // [0, 1]; 1 = off
  StepCacheKind step_cache = StepCacheKind::kOff;
  // (0, 1]; used (and validated) with a step cache only. Unset: the model's default (default_fbcache_threshold), which
  // the pipeline fills in and reports (Stats::step_cache_threshold).
  std::optional<double> step_cache_threshold;

  bool step_cache_on() const { return step_cache != StepCacheKind::kOff; }
  // Whether a lossy accelerator changes a run with (do_cfg) or without classifier-free guidance.
  bool active(bool do_cfg) const { return step_cache_on() || (do_cfg && cfg_truncation < 1.0); }
  // "cfg_truncation 0.8, step_cache fbcache 0.05" (the active ones; "off" when none; an unset threshold: "default").
  std::string describe(bool do_cfg) const;
};
// Throws std::invalid_argument for a value outside its range (the threshold only when the step cache is on).
void validate(const Options& options);

// What the accelerators did in a generation (both passes of a hires fix).
struct Stats {
  int cfg_truncated_steps = 0;  // steps run without the uncond row (CFG truncation)
  int dit_calls = 0;            // DiT calls with the step cache on
  int cached_calls = 0;         // of which reused the cache (blocks 1..41 skipped)
  std::vector<int> cached_steps;  // 1-based steps (counting on across passes) with a call that reused the cache
  // The relative change FirstBlockCache measured at every call, both passes (NaN where a call had nothing to compare
  // with): what the threshold was compared against.
  std::vector<double> step_cache_diffs;
  double step_cache_threshold = 0;  // the threshold the cache ran with (the request's, or the model's default); 0: off
  std::string step_cache_threshold_source;  // "request", or "default for the base model" (sft, unrecognized)
};

// The report's words for where a run's threshold came from: "request", or "default for the <lineage> model"
// ("unrecognized" for an empty lineage).
std::string threshold_source(const Options& requested, std::string_view lineage);
// A threshold the user set above kBaseModelFbCacheThreshold on the base model: the warning (empty otherwise; the
// threshold stands).
std::string base_model_threshold_warning(const Options& requested, std::string_view lineage);
// A step cache that never reused over a whole run: the warning with what it measured (empty when it reused, was off,
// or had fewer than 4 comparable calls: finite entries of step_cache_diffs).
std::string never_engaged_warning(const Stats& stats);

// FBCache (see above) as a moss_dit::StepCacheHook for one sampling pass at a fixed latent grid; set it on the DiT
// (MossDit::set_step_cache) for the pass's forwards.
class FirstBlockCache final : public moss_dit::StepCacheHook {
 public:
  // Arena stack bytes of a cache for up to `rows` distinct rows (three [h * w, 3072] bf16 buffers each, plus the
  // reduction's partial sums).
  static std::size_t workspace_bytes(std::int64_t height, std::int64_t width, int rows);
  // Takes its buffers from the arena stack (released with the caller's scope; the forwards' stack lies above them).
  // Throws std::invalid_argument for rows outside [1, kMaxRows] or a threshold outside (0, 1].
  FirstBlockCache(runtime::DeviceArena& arena, std::int64_t height, std::int64_t width, int rows, double threshold);
  // The DiT holds a pointer to it and its buffers are arena stack memory: neither copied nor moved.
  FirstBlockCache(const FirstBlockCache&) = delete;
  FirstBlockCache& operator=(const FirstBlockCache&) = delete;
  FirstBlockCache(FirstBlockCache&&) = delete;
  FirstBlockCache& operator=(FirstBlockCache&&) = delete;

  void before_blocks(const Tensor& x, std::span<const moss_dit::Row> rows, std::span<const std::int64_t> offsets,
                     std::int64_t video, cudaStream_t stream) override;
  bool after_first_block(const Tensor& x, std::span<const moss_dit::Row> rows, std::span<const std::int64_t> offsets,
                         std::int64_t video, cudaStream_t stream) override;
  void after_blocks(const Tensor& x, std::span<const moss_dit::Row> rows, std::span<const std::int64_t> offsets,
                    std::int64_t video, bool reused, cudaStream_t stream) override;

  // Drops every row's residuals: the next call runs every block (the caller changed the DiT's inputs beyond the rows,
  // e.g. a control input switched on or off between steps, which block 0's relative change may not reveal).
  void invalidate();

  int calls() const { return calls_; }
  int reused() const { return reused_; }
  // The relative change of every call (NaN where a call had nothing to compare with).
  const std::vector<double>& diffs() const { return diffs_; }

 private:
  struct Slot {
    const void* text = nullptr;
    std::int64_t text_length = 0;
    moss_dit::SkipSet skip;
    bool used = false;   // assigned to a row identity
    bool valid = false;  // first / rest hold a full call's residuals
    void* scratch = nullptr;  // x0 before block 0, then x1 after it (a full call)
    void* first = nullptr;    // r0 of the last full call
    void* rest = nullptr;     // R of the last full call
  };
  Slot& slot_of(const moss_dit::Row& row);
  void check(const Tensor& x, std::span<const moss_dit::Row> rows, std::span<const std::int64_t> offsets,
             std::int64_t video) const;
  static void* row_video(const Tensor& x, std::span<const std::int64_t> offsets, std::size_t r);

  std::int64_t video_ = 0;  // h * w
  double threshold_ = 0;
  std::vector<Slot> slots_;
  std::vector<Slot*> current_;  // the slot of each row of the running call
  double* partials_ = nullptr;  // device, rows * kDiffBlocks * 2
  std::vector<double> host_partials_;
  int calls_ = 0, reused_ = 0;
  std::vector<double> diffs_;
};

}  // namespace mi::pipeline::accel
