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
//   max_speed   least time (bigger host target and embedding cache)
// Exact-but-wasteful fallbacks rank last: in balanced and max_speed, parking the DiT (reloading 9.5 GiB per
// generation, a disk read when the page cache cannot hold it) is chosen only when nothing else fits; min_memory parks
// only when that lowers the device peak.
// DiT block streaming (P4) is exact too. Streamed candidates keep either no block resident (least memory) or as many
// as fit the cap, with 2..dit_max_buffers device block buffers, from the weight files or (when the page cache is not
// expected to hold the weights, or in max_speed) a pinned host copy. Their time is the step's compute plus the part of
// the copy not hidden behind it. balanced and max_speed stream only when no plan keeps the whole DiT resident (then
// before parking it), keeping as many blocks resident as fit; min_memory streams whenever that lowers the device peak.
// In every mode, candidates over the mode's host-private target are chosen only when nothing within it fits. When no
// candidate fits at all, plan() throws BudgetError carrying the smallest requirement over every configuration (the
// request cannot fit in any mode).
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
  kStreamed,  // P4: resident_blocks stay in the arena, the other blocks are copied in every step (DitSource)
};
// Where streamed DiT blocks come from (moss_dit::StreamSource).
enum class DitSource {
  kFile,        // the weight files through small pinned rings: page cache (file-backed, not private) or disk
  kHostPinned,  // a pinned host copy of the streamed blocks (host private memory: their full size)
};
// Precision of the DiT's block linears, from the loaded file (P6): bf16 (the bundle, or a bf16 single file), INT8
// ConvRot (a file with any INT8 linear; mixed files that keep some linears bf16 count as int8) or INT4 ConvRot (a file
// with any INT4 linear, the Turbo tier; its INT8 / bf16 linears are costed by fraction).
enum class Precision { kBf16, kInt8, kInt4 };
// Attention kernel of the DiT blocks (moss_dit::AttentionMode, one per attention_q8 variant): exact FA2, or 8-bit
// (SageAttention-style, lossy).
enum class AttentionMode { kExact, kQ8Fp16, kQ8Fp16F32Acc, kQ8Fp8, kQ8Fp8F32Acc };
// Quality tiers (plan §P6): kReference = bf16 weights, exact math; kHigh = INT8 ConvRot GEMMs and/or 8-bit attention;
// kTurbo = INT4 ConvRot GEMMs (an INT4 DiT file) and 8-bit attention; kAuto = the fastest certified configuration of
// the loaded model.
enum class Tier { kReference, kHigh, kTurbo, kAuto };
enum class TeResidency {
  kStreamed,      // decoder layers stream from the weight files through `buffers` device layer buffers
  kHostResident,  // decoder layers held in host private memory and streamed to the device from there
  kVramResident,  // decoder layers resident in the arena across generations
};
enum class VaeDecodeKind { kUntiled, kTiled };

const char* residency_name(DitResidency r);
const char* residency_name(TeResidency r);
const char* source_name(DitSource s);
const char* precision_name(Precision p);        // "bf16", "int8", "int4"
// "exact", "q8_fp16", "q8_fp16_f32acc", "q8_fp8", "q8_fp8_f32acc" (the quality certificates' tokens).
const char* attention_name(AttentionMode a);
const char* tier_name(Tier t);                  // "reference", "high", "turbo", "auto"
// "reference" / "high" / "turbo" / "auto"; throws std::invalid_argument otherwise.
Tier parse_tier(std::string_view name);
// An attention_name(), or "q8" (= q8_fp8); throws std::invalid_argument otherwise.
AttentionMode parse_attention(std::string_view name);
bool is_q8(AttentionMode a);  // an 8-bit mode

// The quality tier a request runs (P4b/P6), decided before planning from what is loaded. The weights fix the GEMM
// precision (an INT8 DiT runs INT8 GEMMs; its weights are not the reference); the tier decides the attention kernel:
//   reference  bf16 DiT and text encoder, exact attention. Refused for quantized weights and with an 8-bit attention.
//   high       the lossy configuration of the loaded bf16 or INT8 DiT: its precision, 8-bit attention (q8_fp8 unless
//              `attention` says otherwise; exact when the device lacks the 8-bit kernel and the DiT is INT8). Runs
//              certified or not (an explicit choice). Refused for a bf16 DiT with exact attention (that is the
//              Reference tier) and for an INT4 DiT (that is the Turbo tier).
//   turbo      an INT4 DiT (refused for any other): INT4 GEMMs, 8-bit attention as for high. Runs certified or not
//              (an explicit choice, reported uncertified without a certificate).
//   auto       with `attention` given: that configuration. Otherwise each lossy configuration of the loaded DiT,
//              fastest first (8-bit attention q8_fp8, q8_fp16, q8_fp8_f32acc, q8_fp16_f32acc, then exact for a
//              quantized DiT), that this device runs: the first with a certificate. None: Reference for a bf16 DiT; an
//              INT8 DiT runs INT8 GEMMs with exact attention, reported uncertified. Auto never runs an uncertified
//              Turbo configuration: an INT4 DiT without a certificate is refused (choose tier turbo explicitly).
// `certificate` is asked for lossy configurations only (Reference needs none).
struct CertificateAnswer {
  bool found = false;
  std::string file;    // where the certificate is
  std::string reason;  // why none was found
};
using CertificateLookup = std::function<CertificateAnswer(Precision dit, AttentionMode attention)>;
using AttentionSupported = std::function<bool(AttentionMode attention)>;
struct TierDecision {
  Tier requested = Tier::kAuto;
  Tier effective = Tier::kReference;
  Precision precision = Precision::kBf16;
  AttentionMode attention = AttentionMode::kExact;
  bool certified = false;   // a certificate covers what runs (Reference: true, needs none)
  std::string certificate;  // its file (empty for Reference and uncertified runs)
  std::string reason;       // what was chosen and why, for the launcher and the report
};
// Throws std::invalid_argument (naming the conflict) for a refused combination.
// te_quantized: the text encoder is an INT8 / W4A8 file, a lossy dimension as an INT8 DiT is (reference refused; the
// exact-attention configuration is a High-tier one).
TierDecision resolve_tier(Tier requested, std::optional<AttentionMode> attention, Precision dit_precision,
                          const CertificateLookup& certificate, const AttentionSupported& supported, bool te_quantized = false);

// Machine throughput the cost model uses. Defaults are conservative for a desktop PC; the pipeline replaces the
// transfer rates with measured ones (pipeline.h: a probe at first use, cached on disk with the embedding cache, then
// updated from every streamed DiT pass). Dev box: 23.9-24.0 GB/s pinned H2D, 19.5 GB/s page cache -> device with 4
// readers, 2.3-3.1 GB/s cold reads (btrfs zstd), 40-72 TFLOPS bf16.
struct HardwareProfile {
  double h2d_gbps = 12.0;     // pinned host -> device
  double disk_gbps = 2.0;     // weight files -> host with a cold page cache
  double host_gbps = 8.0;     // page cache or pageable host memory -> pinned staging -> device (the readers' rate)
  double bf16_tflops = 40.0;  // sustained dense bf16 GEMM
  bool measured = false;      // the transfer rates were measured on this machine
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
  // P6: precision of the block linears and the parameter count (cost model: an INT8 DiT's bytes are no FLOP count).
  Precision dit_precision = Precision::kBf16;
  std::size_t dit_params = 0;
  double dit_int8_fraction = 1.0;  // kInt8: the INT8 share of the block GEMM FLOPs (< 1 for a mixed file)
  // kInt4: the INT4 share (the INT8 share is then dit_int8_fraction, capped at the rest; the remainder is bf16).
  double dit_int4_fraction = 1.0;
  // VAE decoder; the pipeline replaces it with Flux2VaeDecoder::persistent_bytes().
  std::size_t vae_bytes = 0;
};

// Group totals of `index` (its text_encoder, transformer and vae groups). Throws std::invalid_argument when a
// component the engine needs is missing.
ModelShapes shapes_from_index(const weights::ModelIndex& index);
// The same with components loaded from their own files (P6/P7a): the DiT's fields from `dit_index` (with
// `dit_precision`), the text encoder's from `te_index`, the VAE's from `vae_index`; nullptr: the bundle's.
ModelShapes shapes_from_index(const weights::ModelIndex& index, const weights::ModelIndex* dit_index, Precision dit_precision,
                              const weights::ModelIndex* te_index = nullptr, const weights::ModelIndex* vae_index = nullptr);

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
  // The DiT's weight files in the page cache (streamed blocks and reloads read them): learned from the last
  // file-streamed denoise; false makes a pinned host copy of the streamed blocks pay off.
  bool dit_weights_cached = true;
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
  int dit_max_buffers = 3;         // P4 device block buffers (>= 2; buffers - 1 blocks of lookahead)
  // P5 CUDA graphs of a wholly resident (or parked) DiT: device memory they hold outside the arena (0: none).
  std::size_t dit_graph_bytes = 0;
  std::size_t dit_ring_bytes = 0;  // P4 pinned host staging of the file source (its readers' rings)
  bool dit_host_pinned = false;    // P4 a pinned host copy of the streamed blocks as the source
  // Streamed blocks may come from the weight files (false: only from a pinned host copy, e.g. blocks that carry merged
  // runtime-LoRA weights, which the files do not hold).
  bool dit_file_source = true;
  // The GPU runs INT4 tensor-core MMA natively (moss_dit::int4_mma_native: compute capability 8.x). Without it the INT4
  // linears run the INT8 GEMMs on unpacked operands, and are costed at that speed.
  bool int4_mma_native = true;
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
  Precision precision = Precision::kBf16;          // ModelShapes::dit_precision
  AttentionMode attention = AttentionMode::kExact;  // PlanInputs::attention
  int resident_blocks = 0;  // kStreamed: blocks kept resident (the others stream); otherwise every block
  int buffers = 0;          // kStreamed: device block buffers (>= 2; min(buffers, streamed blocks) are allocated)
  int prefetch_blocks = 0;  // kStreamed: lookahead on the copy streams (buffers - 1)
  DitSource source = DitSource::kFile;  // kStreamed
  // kStreamed, per step (cost model): bytes copied, copy time at the expected rate, compute time, and the fraction
  // of the copy expected to run hidden behind compute.
  std::size_t stream_bytes = 0;
  double transfer_s = 0;
  double compute_s = 0;
  double hidden = 1.0;
  bool park_for_text_encoder = false;  // kParked only
  bool park_for_decode = false;        // kParked only
  std::size_t device_bytes = 0;        // persistent + streaming buffers during the denoise
  std::size_t graph_bytes = 0;         // CUDA graphs held outside the arena (Capabilities::dit_graph_bytes)
};

struct TePlan {
  bool run = true;  // false: embedding cache hit, the text encoder is skipped
  TeResidency residency = TeResidency::kStreamed;
  int buffers = 1;  // device layer buffers when streamed (2+: layer k+1 is copied while layer k computes)
};

struct VaePlan {
  bool run = true;
  VaeDecodeKind kind = VaeDecodeKind::kUntiled;
  // tiled: the fastest schedule within the other phases' peak, else the smallest (false: the fastest that fits the
  // cap next to the resident models)
  bool smallest = false;
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
  std::size_t device_peak_bytes = 0;   // estimated process device peak: arena + context + CUDA graphs
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
  AttentionMode attention = AttentionMode::kExact;  // the DiT's attention kernel (cost model)
  bool cache_disk_spill = false;
  // Only plans with this DiT residency (residency, and when streamed: resident blocks, buffers, source): the pipeline
  // asks whether its loaded DiT can serve the request before dropping it.
  std::optional<DitPlan> keep_dit;
};

// Arena cap the budgets allow: min(vram_free - vram_reserve, vram_budget - context_bytes - vram_reserve) when an
// explicit budget is set, else the first term (0 if negative).
std::size_t arena_cap(const Budgets& budgets);

// Throws BudgetError (no configuration fits) or std::invalid_argument (incomplete inputs).
ExecutionPlan plan(const PlanInputs& inputs);

// A pipeline's loaded DiT: the arena it lives in and its residency (residency; when streamed: resident_blocks,
// buffers, source). Every non-streamed residency is "all blocks resident".
struct HeldDit {
  std::size_t arena_bytes = 0;
  DitPlan dit;
  // Device memory the loaded DiT's CUDA graphs hold now, outside the arena. Budgets::vram_free (live free memory plus
  // the arena) does not count it, yet a kept plan reuses it: prefer_held adds it to the free memory and the cap.
  std::size_t graph_bytes = 0;
};

// Chooses between `fresh` (= plan(inputs)) and the plan that keeps the loaded DiT: the request planned inside the held
// arena with the loaded residency. Whenever running `fresh` would drop the loaded DiT (another residency, or an arena
// larger than the held one), the kept plan wins unless `fresh` saves more time than reloading the DiT costs (its
// persistent bytes, plus a pinned host copy, at the weight-file rate). Alternating request sizes or small swings of
// the free device memory therefore settle on one residency instead of reloading the DiT every time. `kept`: set to
// whether the returned plan is the kept one (the caller must then not recreate the arena).
ExecutionPlan prefer_held(const PlanInputs& inputs, ExecutionPlan fresh, const HeldDit& held, bool* kept = nullptr);

}  // namespace mi::planner
