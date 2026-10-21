// Execution planner (plan §3 "Mode presets", P3 step 20, P4b): model shapes + one request + memory budgets + mode +
// hardware profile -> an ExecutionPlan: where every component lives (DiT, text encoder, VAE), the arena size and
// per-phase peaks, the embedding-cache policy, a cost-model time estimate and a human-readable explanation for the
// launcher.
//
// Pure host logic: no CUDA, no I/O. The device workspaces of each option come from the models' exact
// *_workspace_bytes() queries, passed in as Workspaces, so the planner never re-derives a model's allocation pattern.
//
// How it chooses: enumerate the configurations this build supports (Capabilities), compute each one's per-phase
// arena peaks, host-private estimate and estimated time, drop those that exceed the device arena cap or the hard host
// limit, then rank by the mode's objective:
//   min_memory  smallest device peak, then smallest host private, then least time
//   balanced    least time, then smallest device peak
//   max_speed   least time (bigger host target and embedding cache, CUDA graphs)
// Exact-but-wasteful fallbacks rank last: in balanced and max_speed, parking the DiT (reloading 9.5 GiB per
// generation, a disk read when the page cache cannot hold it) is chosen only when nothing else fits; min_memory parks
// only when that lowers the device peak. In every mode, candidates over the mode's host-private target are chosen
// only when nothing within it fits. When no candidate fits at all, plan() throws BudgetError
// carrying the smallest requirement over every configuration (the request cannot fit in any mode).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "weights/model_index.h"

namespace mi::planner {

inline constexpr std::size_t kMiB = std::size_t{1} << 20;
inline constexpr std::size_t kGiB = std::size_t{1} << 30;

// Arena bytes on top of the largest phase peak: alignment of the many small allocations not covered by a query.
inline constexpr std::size_t kArenaSlack = std::size_t{64} << 20;

enum class Mode { kMinMemory, kBalanced, kMaxSpeed };
const char* mode_name(Mode mode);
// "min_memory", "balanced" or "max_speed"; throws std::invalid_argument otherwise.
Mode parse_mode(std::string_view name);

enum class DitResidency {
  kResident,  // every weight in the arena's persistent region, through every phase and across generations
  kParked,    // resident for the denoise; evicted when another phase needs the room and reloaded for the next one
  kStreamed,  // P4: resident_blocks stay, the rest stream from the weight files (page cache) through the pinned ring
};
enum class Precision { kBf16 };  // P6 adds fp8 / int8 storage and compute paths
enum class TeResidency {
  kStreamed,      // decoder layers stream from the weight files through `buffers` device layer buffers
  kHostResident,  // decoder layers held in host private memory and streamed to the device from there
  kVramResident,  // decoder layers resident in the arena across generations
};
enum class VaeDecodeKind { kUntiled, kTiled };

const char* residency_name(DitResidency r);
const char* residency_name(TeResidency r);

// Machine throughput the cost model uses. Defaults are conservative for a desktop PC; `moss-infer profile` (P4b)
// fills a measured one. Dev box: 23.9 GB/s pinned H2D, 2.4-3.1 GB/s NVMe, 40-48 TFLOPS bf16 (docs/p2-report.md).
struct HardwareProfile {
  double h2d_gbps = 12.0;     // pinned host -> device
  double disk_gbps = 2.0;     // weight files -> host with a cold page cache
  double host_gbps = 8.0;     // page cache or pageable host memory -> pinned staging
  double bf16_tflops = 40.0;  // sustained dense bf16 GEMM
  bool measured = false;
};

// Sizes of the model's weight groups (ModelIndex group totals).
struct ModelShapes {
  // Text encoder: the decoder layers and the final norm stream; the embedding table stays memory-mapped on the
  // host (file-backed, so neither device memory nor host private memory).
  int te_layers = 0;
  std::size_t te_layer_bytes = 0;   // largest decoder layer: one streaming buffer
  std::size_t te_stream_bytes = 0;  // decoder layers + final norm: what one encode moves to the device
  std::size_t te_embed_bytes = 0;
  // DiT. shapes_from_index() gives the weight bytes; the pipeline replaces dit_bytes with
  // MossDit::persistent_bytes() (weights + GEMM workspace, as allocated).
  int dit_blocks = 0;
  std::size_t dit_block_bytes = 0;  // largest block: the streaming unit
  std::size_t dit_bytes = 0;        // everything resident
  int dit_dim = 0;                  // model width (cost model: attention FLOPs)
  // VAE decoder; the pipeline replaces it with Flux2VaeDecoder::persistent_bytes().
  std::size_t vae_bytes = 0;
};

// Group totals of `index` (its text_encoder, transformer and vae groups). Throws std::invalid_argument when a
// component the engine needs is missing.
ModelShapes shapes_from_index(const weights::ModelIndex& index);

// One generation as far as memory and time are concerned.
struct RequestShape {
  int height = 1024;
  int width = 1024;
  std::vector<std::int64_t> text_lengths;  // text-encoder sequences: cond, then the negative prompt under CFG
  int steps = 50;                          // scheduled steps
  int rows_per_step = 2;                   // DiT rows per step: 1 (no CFG), 2 (CFG), 3 (CFG + SLG)
  // Sequences the text encoder actually runs on (the embedding-cache misses), for its time estimate; empty: all of
  // text_lengths.
  std::vector<std::int64_t> te_lengths;
  bool decode = true;
  bool te_cached = false;      // every sequence hits the embedding cache: the text-encoder pass is skipped
  // Weight files expected in the page cache (cost model only). The pipeline learns it from the last file-streamed
  // text-encoder pass (weights_cached_from_stream); false makes a host-resident text encoder pay off.
  bool weights_cached = true;
  // The VAE decoder already sits in the arena's persistent region from an earlier request: its bytes count in every
  // phase even when this request does not decode (only recreating the arena drops it).
  bool vae_resident = false;
};

// Whether a file-streamed text-encoder pass ran from the page cache, from its measurements (EncodeStats): `bytes`
// moved in `load_s` (readers' start to the last byte on the device) while the compute stream waited `wait_s` for
// weights. Cold files show as a load that stalled the compute stream and ran below half the page-cache rate the cost
// model assumes (min(host_gbps, h2d_gbps)). A pass that hid its loads behind compute counts as cached: either the
// files were warm or their rate does not matter.
bool weights_cached_from_stream(std::uint64_t bytes, double load_s, double wait_s, const HardwareProfile& hardware);

// One exact tiled-decode schedule.
struct TiledDecode {
  std::size_t bytes = 0;  // decode-phase stack bytes
  double cost = 1.0;      // decode time relative to the untiled decode
  int tile = 0;           // band height in image pixels
};

// Exact arena stack bytes of each phase, from the models' workspace queries.
struct Workspaces {
  // Text-encoder pass (outputs included) for a residency and device layer-buffer count. For kVramResident the
  // weights are counted separately (persistent), so this is activations and outputs only.
  std::function<std::size_t(TeResidency residency, int buffers)> text_encoder;
  std::size_t denoise = 0;  // refined text + max(text refiner, latents + sampling-loop workspace)
  std::size_t decode = 0;   // untiled decode (latents and RGB staging included)
  // Exact tiled decode (Flux2VaeDecoder::plan_tiled): the fastest schedule whose stack bytes (latents and RGB
  // included) fit `budget`, or with budget == 0 the smallest schedule; nullopt when none fits. Empty function: the
  // build has no tiled decode.
  std::function<std::optional<TiledDecode>(std::size_t budget)> decode_tiled;
  std::size_t host_buffers = 0;  // host bytes the generation itself holds (embeddings, noise, latents, RGB)
};

// What the engine build implements; the planner only chooses among these.
struct Capabilities {
  int te_max_buffers = 1;          // device layer buffers the streaming text encoder supports (2+: prefetch overlap)
  std::size_t te_ring_bytes = 0;   // pinned host staging of the streaming text encoder
  bool te_host_resident = false;
  bool te_vram_resident = false;
  bool dit_parking = true;         // P2 stopgap: evict the DiT for the text encoder or the decode
  bool dit_streaming = false;      // P4
  int dit_prefetch_blocks = 2;     // P4 lookahead (device buffers = prefetch + 1)
  std::size_t dit_ring_bytes = 0;  // P4 pinned host staging
};

struct Budgets {
  std::size_t vram_free = 0;  // live free device memory, plus what this process's arena holds (it can be recreated)
  // Device memory kept free outside the arena: CUDA context growth, lazily loaded kernel modules, library handles.
  std::size_t vram_reserve = std::size_t{384} << 20;
  // Explicit cap on this process's device memory as NVML reports it (CUDA context included); 0 = auto
  // (vram_free - vram_reserve).
  std::size_t vram_budget = 0;
  std::size_t context_bytes = std::size_t{512} << 20;  // device memory held outside the arena (context, modules)
  std::size_t host_available = 0;     // host RAM the process can still get (Linux MemAvailable)
  std::size_t host_private_now = 0;   // the process's private memory before the generation (baseline)
  std::size_t host_budget = 0;        // explicit cap on host private memory; 0 = the mode's target
};

struct DitPlan {
  DitResidency residency = DitResidency::kResident;
  Precision precision = Precision::kBf16;
  int resident_blocks = 0;  // kStreamed: blocks kept resident (the others stream)
  int prefetch_blocks = 0;  // kStreamed: lookahead on the copy stream
  bool park_for_text_encoder = false;  // kParked only
  bool park_for_decode = false;        // kParked only
  std::size_t device_bytes = 0;        // persistent + streaming buffers during the denoise
};

struct TePlan {
  bool run = true;  // false: embedding cache hit, the text encoder is skipped
  TeResidency residency = TeResidency::kStreamed;
  int buffers = 1;  // device layer buffers when streamed (2+: layer k+1 is copied while layer k computes)
};

struct VaePlan {
  bool run = true;
  VaeDecodeKind kind = VaeDecodeKind::kUntiled;
  bool smallest = false;     // tiled: the smallest schedule (else the fastest that fits next to the resident models)
  int tile = 0;              // tiled: band height in image pixels (full-width row bands)
  std::size_t stack_bytes = 0;  // decode-phase stack bytes
  double cost = 1.0;         // time relative to the untiled decode
};

struct CachePolicy {
  bool enabled = true;
  std::size_t host_bytes = 0;  // LRU cap of the in-memory embedding cache
  bool keep_negative = true;   // the negative prompt's embedding is always kept
  bool disk_spill = false;     // set by the caller when a cache directory is configured
};

// Arena bytes in use at each phase's peak (persistent + stack); 0 for a phase that does not run.
struct PhasePeaks {
  std::size_t text_encoder = 0;
  std::size_t denoise = 0;
  std::size_t decode = 0;
};

// Cost-model estimate for one generation in steady state (models already loaded when the plan keeps them).
struct TimeEstimate {
  double text_encoder_s = 0;
  double dit_load_s = 0;  // per-generation reloads (parking)
  double denoise_s = 0;
  double decode_s = 0;
  double total_s() const { return text_encoder_s + dit_load_s + denoise_s + decode_s; }
};

struct ExecutionPlan {
  Mode mode = Mode::kBalanced;
  DitPlan dit;
  TePlan te;
  VaePlan vae;
  CachePolicy cache;
  PhasePeaks peaks;
  std::size_t arena_bytes = 0;         // arena capacity to reserve: the largest phase peak + kArenaSlack
  std::size_t device_peak_bytes = 0;   // estimated process device peak: arena + context
  std::size_t arena_cap_bytes = 0;     // the arena cap the plan was made against
  std::size_t host_private_bytes = 0;  // estimated process private memory at its peak
  std::size_t host_target_bytes = 0;   // the mode's host-private target (or the explicit budget)
  std::size_t host_limit_bytes = 0;    // hard host-private limit
  TimeEstimate time;
  std::string explanation;  // human-readable, several lines; for the launcher and the JSON report

  // One JSON object (no trailing newline) with every field above.
  std::string to_json() const;
};

// No configuration fits the budgets, in any mode. The numbers are those of the configuration needing the least
// device memory (for the VRAM case) or host memory (for the host case).
struct BudgetError : std::runtime_error {
  BudgetError(const std::string& message, std::size_t required_arena, std::size_t arena_cap, std::size_t required_host,
              std::size_t host_limit);
  std::size_t required_arena;
  std::size_t arena_cap;
  std::size_t required_host;
  std::size_t host_limit;
};

struct PlanInputs {
  ModelShapes shapes;
  RequestShape request;
  Workspaces workspaces;
  Capabilities caps;
  Budgets budgets;
  HardwareProfile hardware;
  Mode mode = Mode::kBalanced;
  bool cache_disk_spill = false;
};

// Arena cap the budgets allow: min(vram_free - vram_reserve, vram_budget - context_bytes - vram_reserve) when an
// explicit budget is set, else the first term (0 if negative).
std::size_t arena_cap(const Budgets& budgets);

// Throws BudgetError (no configuration fits) or std::invalid_argument (incomplete inputs).
ExecutionPlan plan(const PlanInputs& inputs);

}  // namespace mi::planner
