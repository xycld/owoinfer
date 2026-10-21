// End-to-end MOSS-Image generation (MOSSImagePipeline.__call__, P:340-438): tokenize prompt and negative prompt ->
// embedding cache or Qwen3.5 text encoder (layer-streamed, both prompts in one pass) -> text refiner once per CFG row
// -> torch-compatible CPU noise -> sampling loop on the DiT -> FLUX.2 VAE decode (untiled, or exact tiled) -> RGB8.
//
// Memory (P3): every generate() first asks the planner (planner/planner.h) for an ExecutionPlan from the model's
// shapes, the request, the live free device memory, the host RAM, the mode and the models' exact workspace queries.
// A request that cannot fit in any mode fails right there with planner::BudgetError, before any heavy work. The plan
// sizes one DeviceArena per pipeline: persistent region = the DiT (kept across generate() calls) and the VAE decoder;
// stack region = one phase at a time (text encoder, denoise, decode). When the plan keeps the DiT resident the decode
// runs exactly tiled if the untiled one does not fit next to it; the DiT is evicted ("parked": the arena is recreated
// without it, the weights reloaded for the next denoise) only when the plan says so, which it does as a last resort.
// One text encoder (its pinned rings and streams) lives across calls while the arena does.
// P4: when the whole DiT does not fit (or in min_memory), the plan streams DiT blocks: some stay resident, the others
// are copied every step from the weight files (page cache) or a pinned host copy into device block buffers, ahead of
// the compute on copy streams; bit-identical to the resident DiT. The transfer rates the plan's cost model uses are
// measured once (a short probe, cached as <cache_dir>/hardware_profile.json when a cache directory is set) and then
// refined from every streamed denoise (pipeline/hardware_profile.h). A loaded DiT is kept (inside its held arena, with
// its residency) unless the request's own plan saves more time than reloading the DiT costs (planner::prefer_held).
// P5: the DiT blocks run fused (PipelineOptions::block_path) once a startup self-check (once per GPU, driver, cuBLASLt
// version and fused-kernels version, cached with the profile) found the fused GEMMs bit-identical to the op-by-op ones
// here, each in the cuBLASLt configurations it validated (the others run op by op); otherwise op by op, and Result
// says why.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pipeline/sampler.h"
#include "planner/planner.h"
#include "util/json.h"

namespace mi::pipeline {

inline constexpr std::int64_t kMaxSeed = std::int64_t{1} << 53;  // the official app's browser-safe bound
inline constexpr int kMaxSteps = 10000;
// Image size bounds for GUI/CLI input. USERGUIDE.md: the model extrapolates from 0.25 MP (512x512) to 4 MP ("2K");
// 4.2 MP admits 2048x2048 (4.19 MP) and other ~4 MP aspect ratios. The side bound keeps extreme aspect ratios out.
inline constexpr int kMaxSide = 4096;
inline constexpr std::int64_t kMaxPixels = 4'200'000;
// Upper bound for PipelineOptions sizes taken from user input (vram_option_bytes).
inline constexpr std::size_t kMaxVramBytes = std::size_t{1024} << 30;

// One generation, mirroring the assets' parameters.json (gradio_app.py GenerationConfig) plus engine choices.
struct Request {
  std::filesystem::path model;  // bundle directory (base or sft): {text_encoder, tokenizer, transformer, vae}
  std::string prompt;
  std::string negative_prompt;
  int height = 1024;
  int width = 1024;
  int steps = 50;
  double cfg = 3.0;
  double shift = 5.0;
  // -1: random below 2^32 (the app's behaviour). torch seeds mt19937 with the low 32 bits, so seed s and
  // s + 2^32 give the same noise.
  std::int64_t seed = 42;
  std::vector<int> uncond_skip_layers = {22};
  std::vector<int> slg_layers;
  double slg_scale = 0.0;
  double slg_start = 0.0;
  double slg_stop = 1.0;
  int max_sequence_length = 4096;
  SamplerKind sampler = SamplerKind::kEuler;
  bool decode = true;  // false: stop after sampling (final latents only)
};

// Throws std::invalid_argument with the official app's wording where it has one.
void validate(const Request& request);

// Reads a parameters.json document: either the asset file ({"schema_version", "parameters": {...}}, versions 1 and 2
// as gradio_app.parse_parameters reads them) or a bare parameters object. Skip-layer lists may be "22,24" strings
// or integer arrays; a null seed means -1. `model` is left empty. Throws std::invalid_argument.
Request request_from_parameters(const json::Value& doc);

// GenerationConfig.layers (gradio_app.py:119-129): "22, 24" -> {22, 24}, blank -> {}. Throws std::invalid_argument
// on anything but comma-separated integers (range and uniqueness are checked by validate()).
std::vector<int> parse_layer_list(std::string_view text);

// A user-supplied memory size (`amount` units of `unit_bytes`, e.g. GiB) for PipelineOptions::vram_budget or
// vram_reserve, in bytes. Throws std::invalid_argument naming `what` unless it is finite, positive, at least 1 MiB
// (smaller values would truncate to 0, which means "no cap") and at most kMaxVramBytes.
std::size_t vram_option_bytes(double amount, double unit_bytes, std::string_view what);

struct PipelineOptions {
  planner::Mode mode = planner::Mode::kBalanced;
  // Cap on this process's device memory as NVML reports it (CUDA context included); 0 = free device memory -
  // vram_reserve. The arena gets the budget minus planner::Budgets::context_bytes and vram_reserve.
  std::size_t vram_budget = 0;
  // Device memory left outside the arena for what is not allocated from it: CUDA context growth, lazily loaded
  // kernel modules, cuBLASLt handles.
  std::size_t vram_reserve = std::size_t{384} << 20;
  // Cap on the process's private host memory (planner/host_memory.h); 0 = the mode's target.
  std::size_t host_ram_budget = 0;
  bool direct_io = false;  // O_DIRECT weight reads (cold-cache behaviour)
  bool embedding_cache = true;
  std::filesystem::path cache_dir;  // embedding-cache disk spill; empty = memory only
  // A held arena is recreated at the plan's size when a request needs less than this fraction of it, so a launcher
  // does not pin the device memory of its largest request forever (resident models are reloaded then, reported as
  // MemoryStats::dit_reloaded). 0 = never shrink.
  double arena_shrink_below = 0.5;
  // Fixed machine rates for the planner's cost model (a launcher's own profile, reproducible tests): no probe, no
  // learning from measured passes. Unset: probed at the first generate() and refined from every pass.
  std::optional<planner::HardwareProfile> hardware;
  // DiT block implementation (all give the same bits). kFused (default) falls back to kOpByOp when the startup
  // self-check fails; kFusedGraphs adds CUDA graphs (opt-in: ~0 measured gain, holds graph memory).
  moss_dit::BlockPath block_path = moss_dit::BlockPath::kFused;
};

struct Timings {
  double index_load_s = 0;  // weight index + tokenizer (first use of a bundle only)
  double tokenize_s = 0;
  double plan_s = 0;         // cache lookups + planning
  double text_encode_s = 0;  // the whole text-encoder pass (0 on a full cache hit), of which:
  double te_weight_load_s = 0;  // streaming: readers' start to the last byte on the device (overlaps compute)
  double te_weight_wait_s = 0;  // compute stream waiting for weights
  double te_compute_s = 0;
  double dit_load_s = 0;  // 0 when the DiT was resident
  double refine_s = 0;    // text refiner, all rows
  double denoise_s = 0;
  double vae_load_s = 0;
  double decode_s = 0;
  double total_s = 0;
  std::vector<double> step_s;
};

struct MemoryStats {
  std::size_t arena_capacity = 0;
  std::size_t arena_high_water = 0;   // peak arena use during the call (persistent + stack)
  std::size_t device_peak = 0;        // peak of (free memory at pipeline construction - free memory now)
  std::size_t free_at_start = 0;      // free device memory when the pipeline was constructed
  int arena_creations = 0;            // arenas created during the call (first use, growth, shrink, parking)
  bool dit_parked_for_text_encoder = false;
  bool dit_parked_for_decode = false;
  // A resident DiT was dropped outside the plan's parking (arena growth or shrink, a phase that did not fit) and
  // loaded again in this call.
  bool dit_reloaded = false;
  std::size_t host_private_start = 0;  // process private host memory when the call started
  std::size_t host_private_peak = 0;   // its peak, sampled at phase boundaries
};

// DiT block streaming of one generate() (P4 step 28), from CUDA events on the GPU timeline (moss_dit::StreamStats).
struct DitStreamStats {
  bool streamed = false;   // some blocks were streamed
  int resident_blocks = 0;
  int buffers = 0;
  std::string source;      // "file" (page cache or disk) or "host_pinned"
  std::uint64_t bytes = 0;  // copied to the device over every step
  double copy_s = 0;        // copy streams busy
  double stall_s = 0;       // compute stream waiting for streamed weights
  int stalls = 0;           // streamed blocks that made the compute stream wait (> 50 us)
  double gbps() const { return copy_s > 0 ? static_cast<double>(bytes) / copy_s / 1e9 : 0.0; }
  // Fraction of the copy time hidden behind compute (AC10).
  double hidden() const { return copy_s > 0 ? std::max(0.0, 1.0 - stall_s / copy_s) : 0.0; }
  struct Step {
    double gbps = 0, hidden = 0, stall_s = 0;
    int stalls = 0;
  };
  std::vector<Step> steps;  // per finished step
};

struct Result {
  int height = 0, width = 0;
  std::vector<std::uint8_t> rgb;  // [height, width, 3]; empty if !decode or cancelled
  // Final latents [128, height/16, width/16]; the partial result if cancelled during sampling (the latents after the
  // last finished step, also when the cancel lands inside a step), empty if cancelled before it started.
  std::vector<float> latents;
  std::uint64_t seed = 0;         // the seed actually used
  int steps_requested = 0;
  int steps_scheduled = 0;  // after the schedule's snapping
  int steps_run = 0;
  // Set whenever a cancel request was seen, even after the last step (then steps_run == steps_scheduled and the
  // decode is skipped).
  bool cancelled = false;
  int cond_tokens = 0, uncond_tokens = 0;
  int te_cache_hits = 0;    // sequences served by the embedding cache
  bool te_skipped = false;  // every sequence hit: the text encoder did not run
  // The decode actually run: tiled (exact, full-width row bands) and the schedule, or untiled. It can differ from
  // plan.vae: the decode takes the fastest schedule the arena it finds leaves room for (a held arena may be larger).
  bool decode_tiled = false;
  std::string decode_schedule;
  planner::ExecutionPlan plan;
  planner::HardwareProfile hardware;  // the transfer rates the plan was made with
  // DiT block path the denoise ran (fused_graphs only when every block was resident), and why it differs from
  // PipelineOptions::block_path (empty when it does not).
  moss_dit::BlockPath block_path = moss_dit::BlockPath::kFused;
  std::string block_path_note;
  DitStreamStats dit_stream;
  Timings timings;
  MemoryStats memory;
};

// Phases of generate(), in order. kTextEncoder is entered even on a full embedding-cache hit and kDitLoad even when
// the DiT is resident; kDecode only when decoding.
enum class Phase { kTextEncoder, kDitLoad, kDenoise, kDecode };

struct GenerateControl {
  // Once per call, after planning and before the first phase (the launcher shows plan.explanation and the mode).
  std::function<void(const planner::ExecutionPlan&)> on_plan;
  StepCallback on_step;                       // after every sampling step; returning false cancels
  std::function<bool(Phase)> on_phase;        // before each phase starts; returning false cancels
  // Checked before each phase and after every step, and inside the phases: between text-encoder layers, per DiT
  // weight group while loading, between DiT blocks and per tiled-decode band (each phase then keeps at most one
  // layer / block / band queued on the GPU). The pipeline stays usable after a cancel.
  const std::atomic<bool>* cancel = nullptr;
};

class Pipeline {
 public:
  explicit Pipeline(PipelineOptions options = {});
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // Not thread-safe; one generation at a time per instance. Throws planner::BudgetError (before any phase) when
  // the request cannot fit the budgets in any mode, std::invalid_argument on an invalid request.
  Result generate(const Request& request, const GenerateControl& control = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mi::pipeline
