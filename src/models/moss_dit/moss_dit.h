// MOSS-Image DiT (MOSSImageTransformer, P:189-262), bf16, its blocks resident in device memory or streamed (P4).
//
// The work is split by what it depends on:
//   - encode_text(): the text refiner (P:93-123, P:251). Timestep-independent: run it once per
//     generation per CFG row and keep the result for every step.
//   - forward(): one sampling step for several rows at once (cond, uncond, SLG). Time embedding and
//     modulation (P:232-241) are computed once and shared by all rows. The rows' joint sequences
//     [video_r | text_r] are concatenated along the token dim, so every GEMM and elementwise op runs
//     once over all rows, and joint attention is one varlen call with per-row lengths and RoPE tables.
//     A block that some rows skip runs only on the other rows' (contiguous) token ranges, leaving the
//     skipping rows' tokens untouched: exactly the reference's identity (P:253-255).
//     The reference evaluates each row alone. Batching gives the same bits wherever cuBLASLt runs every linear with a
//     K-ordered algorithm without split-K at both the batched and the per-row token count (every output element is
//     then the same MMA chain whatever M, kernels/fused/fused_gemm.h): measured on sm_89 for every run of >= 384
//     tokens, so every image of 320x320 or more (tested at 32x48 and 64x64 latent grids, 2 and 3 rows). Smaller rows
//     can get split-K at one count and not the other (e.g. ffn_down at 128x128 with a 20-token prompt) and then
//     differ from per-row evaluation in the last bits, as torch's own GEMMs differ between batch sizes. The other
//     batched ops are row-local, and torch's norm reduction order only depends on the row count below 16 rows (a row
//     has >= 16 joint tokens for any image of 64x64 or more). The text refiner and the head, where a row may be
//     shorter, run per row.
// Every op follows the reference's order and bf16/fp32 rounding.
//
// Memory: weights in the arena's persistent region (load), activations on its stack region for the
// duration of one call (sized by *_workspace_bytes). Stack memory is released when a call returns,
// with its kernels possibly still queued: reuse it only on the same stream.
// P4: the blocks' weights come from a BlockWeights provider (dit_weights.h): every block resident (the default), or
// some streamed from host memory through device block buffers that forward() adds to its stack
// (stream_workspace_bytes); streamed and resident forwards are bit-identical.
// P5: forward() runs each block fused by default (dit_block_fused.h: QK-norm+RoPE in one pass, SiTU-GLU and
// gate+residual in the GEMM epilogues, op by op where cuBLASLt's algorithm for the run's size would not match),
// bit-identical to the op-by-op block at every image size (tested from 128x128 up); with kFusedGraphs it also replays
// each block from a CUDA graph once a step shape repeats with every block resident (dit_graph.h).
// P6: a DiT loaded from a single-file checkpoint (weights/quant QuantIndex: Comfy-Org / community INT8 ConvRot files,
// our converter's output, or a bf16 file) keeps each block linear as stored, INT8 ConvRot G = 256 or bf16. A model with
// any INT8 linear (or with 8-bit attention set) runs every block on the quantized block path (dit_block_quant.h); the
// globals and the text refiner stay bf16. Streaming and graphs work alike; INT8 blocks move about half the bytes.
// P6c (Turbo tier): an INT4 ConvRot file (convrot_w4a4, quant::IndexOptions::native_int4) keeps its linears packed and
// runs them on the INT4 GEMMs, any mix of INT4, INT8 and bf16 linears per block; INT4 blocks move about a quarter of
// the bf16 bytes.
// NVFP4 (Turbo tier on compute capability 12.0, NVFP4 plan §4.6): a comfy_quant nvfp4 file (our converter's or a
// community one, usually with INT8 ConvRot keep-list linears) runs its NVFP4 linears on the sm_120a block-scaled GEMMs
// (kernels/nvfp4), any mix with INT8 and bf16 linears per block; NVFP4 blocks move 0.28 of the bf16 bytes. Elsewhere
// the model refuses the file (nvfp4_support.h, user decision D1).
// The fused route of a forward is its own snapshot unless a request pinned one (set_request_route): a pipeline sizes
// its arena from one snapshot per request and every step of the request then runs it.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "core/tensor.h"
#include "kernels/basic/gemm.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/dit_block_quant.h"
#include "models/moss_dit/dit_graph.h"
#include "models/moss_dit/dit_weights.h"
#include "models/moss_dit/distill.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/upload.h"

namespace mi::moss_dit {

inline constexpr int kDim = 3072;
inline constexpr int kHeads = 24;
inline constexpr int kHeadDim = 128;
inline constexpr int kLayers = 42;
inline constexpr int kFfnDim = 8192;
inline constexpr int kTextDim = 4096;
inline constexpr int kFreqDim = 256;
inline constexpr int kLatentChannels = 128;
static_assert(kLayers == kNumBlocks);

// Rows per forward() (cond, uncond, SLG, ... ; bounds the per-call cu_seqlens upload).
inline constexpr int kMaxRows = 8;

// CUDA graphs a model caches with kFusedGraphs (one per block of a few step shapes).
inline constexpr int kGraphCapacity = 4 * 42;

// Blocks a row skips (zero-based, P:70-74, P:253-255).
using SkipSet = std::bitset<kLayers>;

struct Row {
  Tensor text;  // refined text tokens from encode_text(): CUDA bf16 [S, kDim], S >= 1
  SkipSet skip;
};

// Debug/parity hook, called after every block with the residual stream of all rows: CUDA bf16
// [total tokens, kDim], row r at token_offsets()[r] as [video (h*w) | text (S_r)]. Enqueued work on the
// stream has not necessarily finished; synchronize before reading.
using BlockHook = std::function<void(int block, const Tensor& x)>;

// Calibration seam (the NVFP4 converter's Hessian capture, pipeline/nvfp4_calibration.h; MossDit::set_linear_input_hook):
// called with the input of each block GEMM unit, for every run of rows the block executes, as CUDA bf16
// [run tokens, K] (contiguous rows) on `stream`. The input is valid until the GEMM after the call reads it: enqueue
// work on `stream` only.
enum class BlockGemm { kQkv, kO, kUpGate, kDown };
using LinearInputHook = std::function<void(int block, BlockGemm gemm, const Tensor& x, cudaStream_t stream)>;

// Step caching (P6b, e.g. the first-block cache of pipeline/accel), around the block loop of every forward() while set
// (MossDit::set_step_cache). x: every row's residual stream as BlockHook's; rows: the forward's rows; offsets:
// token_offsets() of them (row r's video tokens are x[offsets[r], offsets[r] + video)). before_blocks runs before block
// 0, after_first_block after it: returning true reuses the hook's cache, so blocks 1..kLayers-1 do not run for any row
// in this forward (nor are their streamed weights waited for: the streaming pass is ended there), and after_blocks
// (reused = true) must then write every row's final video tokens into x; otherwise all blocks run and after_blocks
// (reused = false) sees their result. Any of them may synchronize `stream`; each runs outside CUDA graphs. Without a
// hook a forward is unchanged.
class StepCacheHook {
 public:
  virtual ~StepCacheHook() = default;
  virtual void before_blocks(const Tensor& x, std::span<const Row> rows, std::span<const std::int64_t> offsets,
                             std::int64_t video, cudaStream_t stream) = 0;
  virtual bool after_first_block(const Tensor& x, std::span<const Row> rows, std::span<const std::int64_t> offsets,
                                 std::int64_t video, cudaStream_t stream) = 0;
  virtual void after_blocks(const Tensor& x, std::span<const Row> rows, std::span<const std::int64_t> offsets,
                            std::int64_t video, bool reused, cudaStream_t stream) = 0;
};

struct LoadStats {
  double seconds = 0;       // wall time, disk to resident (includes the final copy-stream sync)
  std::uint64_t bytes = 0;  // payload bytes read
  bool direct = false;      // every weight file was read with direct I/O (an open that refused it reads buffered)
};

// The cancel flag was seen while loading or between blocks (MossDit's `cancel`). A cancelled forward() leaves
// its outputs partly written and the model usable; a cancelled load leaves what it allocated in the arena's
// persistent region (recreate the arena).
struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("moss_dit: cancelled") {}
};

// Block implementation of forward(); every path gives the same bits (tests/unit/test_dit_block_fused).
enum class BlockPath {
  kOpByOp,       // P2 engine: one kernel per reference op, cuBLASLt GEMMs (A/B baseline and fallback)
  kFused,        // run_block_fused, eager (the default)
  kFusedGraphs,  // run_block_fused, each block replayed from a CUDA graph when every block is resident (opt-in:
                 // ~0 measured gain on an idle host; holds graph memory)
};
const char* block_path_name(BlockPath p);
// "op_by_op", "fused" or "fused_graphs"; throws std::invalid_argument otherwise.
BlockPath parse_block_path(std::string_view name);
// Device memory the CUDA graphs of one model hold outside the arena at most (kFusedGraphs, every block resident):
// conservative, per cached graph times the cache's capacity.
std::size_t graph_cache_bytes();

// What a forward computes with beyond the block path (P6). Default: the bf16 blocks of P5 (bit-identical to it).
struct ComputeConfig {
  bool quantized = false;     // every block on run_block_quant: some block linear is INT8, or 8-bit attention
  bool bf16_up_gate = true;   // quantized: some block keeps ffn_up_gate bf16 (its `big` layout follows the route)
  AttentionMode attention = AttentionMode::kExact;
  std::int64_t lora_rank = 0;  // runtime LoRA side branches' max_rank (0: none): the t buffers of a forward
  bool int4 = false;           // some block linear is INT4 (with LoRA side branches: the int4_unpack buffer)
  bool int4_via_int8 = false;  // int4: every INT4 linear runs the INT8 GEMMs on unpacked operands (the same bits)
  bool nvfp4 = false;          // some block linear is NVFP4 (QuantBlockBuffers::nvfp4_sf; with LoRA: nvfp4_delta)
  bool nvfp4_up_gate = false;  // some ffn_up_gate is NVFP4: `big` holds its [L, 2F] output (nvfp4_big_cols)
  bool nvfp4_ffn = false;      // ... in a block whose ffn_down is not NVFP4: the bf16 ffn after it too
  // Runtime LoRA on NVFP4 sites (nvfp4_big_cols, QuantBlockBuffers::nvfp4_delta): an NVFP4 ffn_up_gate in a block with an
  // active ffn_up_gate or ffn_down side branch (the 3F `big`), and the widest NVFP4 site with an active side branch
  // (Δ's columns per token; 0: none).
  bool nvfp4_lora_ffn = false;
  std::int64_t nvfp4_delta_cols = 0;
  bool operator==(const ComputeConfig&) const = default;
};

// How INT4 linears run (dit_block_quant.h): kAuto = natively where the GPU has INT4 MMA (int4_mma_native()), else
// through the INT8 GEMMs on unpacked operands; kNative / kViaInt8 force one (both give the same bits; kNative on a GPU
// without INT4 MMA runs ptxas's emulation, A/B measurements only).
enum class Int4Route { kAuto, kNative, kViaInt8 };

// The precision of the block linears of a DiT index: "bf16" (quant == nullptr: the bundle, or every linear bf16),
// "int8_convrot256" (all INT8), "int4_convrot256" (all INT4) or "mixed (N int4_convrot256, M bf16)".
std::string describe_linears(const quant::QuantIndex* quant);
// Some block linear of `quant` is quantized, INT8 or INT4 (false for nullptr).
bool has_quantized_linears(const quant::QuantIndex* quant);
// Some block linear of `quant` is INT4 (false for nullptr).
bool has_int4_linears(const quant::QuantIndex* quant);
// Some block linear of `quant` is NVFP4 (false for nullptr).
bool has_nvfp4_linears(const quant::QuantIndex* quant);
// The fraction of the block GEMMs' weight elements (their FLOPs) that are INT8 / INT4 / NVFP4 (0 for nullptr): a mixed
// file's share. NVFP4 weights count their unpacked K (INT4's stored [N, K/2] count as stored).
double int8_gemm_fraction(const quant::QuantIndex* quant);
double int4_gemm_fraction(const quant::QuantIndex* quant);
double nvfp4_gemm_fraction(const quant::QuantIndex* quant);
// Runtime LoRA on a quantized DiT:
//   check_lora_merges: std::invalid_argument when an adapter of `stack` that is merged at load (not a side branch:
//     LoHa / LoKr / full deltas) targets a block linear stored INT4 or NVFP4 in `quant` (merges need bf16 or INT8
//     ConvRot). For callers to refuse a stack before they load anything; the model's constructor refuses it too.
//   int4_side_branch_fraction: the share of the block GEMMs' weight elements that are INT4 and carry an active side
//     branch of `sb` (those sites run the INT8 LoRA GEMMs, dit_block_quant.h: costed at INT8 speed).
//   shadowed_by_time_embedding: one message per adapter of `stack` that set_time_embedding's replacement overrides
//     (time_embedding.* deltas, and bias deltas of shared_block_modulation / video_head_modulation).
void check_lora_merges(const lora::LoraStack& stack, const quant::QuantIndex* quant);
double int4_side_branch_fraction(const quant::QuantIndex* quant, const lora::SideBranches& sb);
std::vector<std::string> shadowed_by_time_embedding(const lora::LoraStack& stack);

// Control adapters (models/control): a replacement video embedder over the channel concatenation [latents | extra]
// (MossDit::set_input_extension).
struct InputExtension {
  Tensor weight;  // CUDA bf16 [kDim, kLatentChannels + C], C > 0
  Tensor bias;    // CUDA bf16 [kDim]
  Tensor extra;   // CUDA fp32 [C, h, w], contiguous: on the forward's latent grid, the same for every row
};

// The video embedding of a forward (P:247-248): tokens = [latents | extra] flattened to bf16 [h*w, C_in] (each fp32
// value rounded once: the reference's .to(bf16) of the concatenated latents), then out = tokens · weightᵀ + bias with
// gemm (bf16 in, fp32 accumulation, bf16 out [h*w, kDim]). extra null (and weight [kDim, kLatentChannels]): the plain
// embedder, the ops the DiT always ran. tokens: CUDA scratch of at least h*w*C_in bf16 values.
void embed_video(kernels::Gemm& gemm, void* gemm_workspace, std::size_t gemm_workspace_bytes, const Tensor& latents,
                 const Tensor* extra, void* tokens, const Tensor& weight, const Tensor& bias, const Tensor& out,
                 cudaStream_t stream);

class MossDit {
 public:
  // Persistent arena bytes the constructor allocates (weights + GEMM workspace), every block resident or with
  // `residency`.
  static std::size_t persistent_bytes(const weights::ModelIndex& index);
  static std::size_t persistent_bytes(const weights::ModelIndex& index, const Residency& residency);
  // Stack bytes forward() adds to workspace_bytes() for streamed blocks (the device block buffers; 0 if none).
  static std::size_t stream_workspace_bytes(const weights::ModelIndex& index, const Residency& residency);
  // The config a model loaded from `quant` (nullptr: the bf16 bundle) runs with attention mode `attention`, and with the
  // side branches active in `lora` (nullptr: none): what set_lora gives a forward.
  static ComputeConfig compute_config(const quant::QuantIndex* quant, AttentionMode attention,
                                      Int4Route int4_route = Int4Route::kAuto, const lora::SideBranches* lora = nullptr);
  // The INT4 route of the following forwards (default kAuto); the forward stack follows compute_config().
  void set_int4_route(Int4Route route);
  Int4Route int4_route() const { return int4_route_; }

  // Uploads the transformer groups (globals, text_embedder and the resident blocks) of `index` into the
  // arena's persistent region and waits until they are resident; streamed blocks are validated (and copied
  // to pinned host memory for StreamSource::kHostPinned). direct_io: O_DIRECT reads (bypass the page cache)
  // instead of buffered reads for the load. host_files (optional, must outlive the constructor only): weight files
  // held in host memory, read from there instead of the files (the same bytes: a soft-unloaded model's re-upload). The arena and index must outlive the model. `cancel` (optional,
  // must outlive the model) is checked before every weight group is read and between the blocks of every
  // forward(); once set, the call waits for its queued copies/kernels and throws Cancelled.
  // merges (optional, runtime LoRA, models/lora): the stack's adapters that are merged rather than side-branched
  // (LyCORIS / full deltas, non-block targets) are merged into the weights at load: bf16 in fp32 then rounded, INT8
  // block GEMMs dequantized, merged and requantized (lossy). A streamed block with merges needs
  // StreamSource::kHostPinned (its merged pinned copy; std::invalid_argument for kFile). The stack must outlive the
  // constructor only.
  MossDit(const weights::ModelIndex& index, runtime::DeviceArena& arena, runtime::StreamSet& streams,
          bool direct_io = false, const std::atomic<bool>* cancel = nullptr, const Residency& residency = {},
          const lora::LoraStack* merges = nullptr, const weights::HostFiles* host_files = nullptr);
  // P6: from a single-file DiT (quant::to_model_index): every block linear checked against the spec its GEMM runs
  // (bf16, int8_convrot256, int4_convrot256 or nvfp4; std::runtime_error naming the linear otherwise; NVFP4 linears
  // where they cannot run: std::runtime_error with nvfp4_unsupported_reason()). persistent_bytes /
  // stream_workspace_bytes take quant.index. `quant` must outlive the model.
  MossDit(const quant::QuantIndex& quant, runtime::DeviceArena& arena, runtime::StreamSet& streams,
          bool direct_io = false, const std::atomic<bool>* cancel = nullptr, const Residency& residency = {},
          const lora::LoraStack* merges = nullptr, const weights::HostFiles* host_files = nullptr);
  ~MossDit();
  // Whether `stack` has merged adapters on block `block` (they need the block resident or a pinned host copy).
  static bool block_has_merges(const lora::LoraStack& stack, int block);
  // Host bytes of the pristine copies a model loaded with `stack`'s merges keeps (bf16 n*k*2 (+ bias), INT8 n*k + 4n),
  // and what this model keeps now.
  static std::size_t pristine_bytes(const lora::LoraStack& stack, const quant::QuantIndex* quant);
  std::size_t pristine_bytes() const;
  MossDit(const MossDit&) = delete;
  MossDit& operator=(const MossDit&) = delete;

  // A model starts with kFused. The forward's stack must hold workspace_bytes(..., path) (op-by-op may need a larger
  // activation buffer).
  void set_block_path(BlockPath path);
  BlockPath block_path() const { return path_; }
  // CUDA-graph diagnostics of kFusedGraphs.
  const StepGraphStats& graph_stats() const { return graphs_.stats(); }
  // Device memory the cached CUDA graphs hold outside the arena now (graph_cache_bytes()'s per-graph bound each).
  std::size_t graph_bytes() const;

  // Attention kernel of the blocks (default kExact). An 8-bit mode needs attention_supported() (std::runtime_error otherwise)
  // and a forward stack of workspace_bytes(..., compute_config()).
  void set_attention(AttentionMode mode);
  AttentionMode attention() const { return config_.attention; }
  const ComputeConfig& compute_config() const { return config_; }
  // The loaded file's quantized index (nullptr: the bf16 bundle).
  const quant::QuantIndex* quant_index() const { return quant_; }

  // Runtime LoRA side branches (nullptr: none; must outlive their use): while any site is active every block runs on
  // the quantized block path with the fused LoRA GEMMs, and the forward stack grows by the t buffers
  // (workspace_bytes(..., compute_config())). Set between forwards. Strength changes (SideBranches::update_strengths)
  // need no call here: each forward re-derives compute_config() from the sites active then. Throws std::invalid_argument for a QKV side branch on a block whose q/k/v differ in precision.
  void set_lora(const lora::SideBranches* side_branches);
  // After a strength change of a LoRA with merged adapters (LoraStack::set_strength on the stack the model was loaded
  // with): re-merges every merged module from the pristine weights kept at load, into the device weights and the
  // pinned host copies of streamed blocks. Synchronizes the device; call between forwards.
  void remerge_lora(const lora::LoraStack& stack);
  // Switches the merged adapters to those of `stack` (nullptr: none) without reloading: every module merged now is
  // merged again from its pristine copy (its pristine bits when `stack` does not merge it, the copy then dropped), and a
  // module `stack` merges that is not merged yet is kept and merged in place, the same bits as a load with `stack`.
  // Only modules outside the blocks (text refiner, embedders, modulation, head: always resident) can be newly merged in
  // place; when `stack` merges a block module that is not merged yet, nothing changes and it returns false (reload the
  // model with `stack` then). Synchronizes the device; call between forwards. The stack must outlive the call only.
  // memo_key (optional): names `stack` with its strengths. The merged bits of the last stack given a key are kept on
  // the host with the pristine copies they derive from (together at most kMaxMergeMemoBytes), so switching back to it
  // later (a control adapter toggled off and on) stores them again instead of downloading and merging: the same bits.
  bool rebase_merges(const lora::LoraStack* stack, const std::string& memo_key = {});
  // Host bytes rebase_merges keeps beyond the merged modules' pristine copies (pristine_bytes()): the memo's merged bits
  // and the pristine copies of modules merged by it but not by the current stack.
  std::size_t memo_bytes() const;
  // Frees the memo (and the pristine copies only it needed): the next switch back merges again.
  void drop_merge_memo();
  static constexpr std::size_t kMaxMergeMemoBytes = std::size_t{1} << 30;
  // Test seam (internal): the memo's cap for every model of the process (default kMaxMergeMemoBytes).
  static void set_merge_memo_cap_for_tests(std::size_t bytes);
  const lora::SideBranches* lora() const { return lora_; }

  // Control adapters: every following forward embeds [latents | ext->extra] with ext's weight and bias instead of the
  // loaded video embedder (nullptr: the loaded one again, the forward bit-identical to never setting one). `ext` and
  // its tensors must outlive their use; set between forwards. Throws std::invalid_argument for a weight other than
  // bf16 [kDim, kLatentChannels + C] with C > 0, or a bias other than bf16 [kDim]; forward() checks extra against its
  // latent grid. The tokens of the concatenation fit the forward's existing activations (no extra workspace).
  void set_input_extension(const InputExtension* ext);
  const InputExtension* input_extension() const { return input_ext_; }

  // Pins the fused route of every following forward (a request's snapshot, which sized its workspace); nullopt: each
  // forward takes its own snapshot (the default).
  void set_request_route(std::optional<FusedRouteSnapshot> route) { request_route_ = std::move(route); }

  // Replaces the cancel flag (nullptr: none). While a flag is set, forward() keeps the host at most one block
  // ahead of the GPU, so a cancel lands within about one block.
  void set_cancel(const std::atomic<bool>* cancel) { cancel_ = cancel; }

  // Linear-input hook (LinearInputHook; empty: none, the default). While set, forward() calls it at every block GEMM
  // unit; it needs block path kOpByOp and a bf16 DiT (std::logic_error from forward() otherwise). Set between forwards.
  void set_linear_input_hook(LinearInputHook hook) { linear_input_hook_ = std::move(hook); }

  // Step caching (P6b; nullptr: none, the default). The hook must outlive its use; set between forwards.
  void set_step_cache(StepCacheHook* hook) { step_cache_ = hook; }
  StepCacheHook* step_cache() const { return step_cache_; }

  const LoadStats& load_stats() const { return load_stats_; }
  const Residency& residency() const { return blocks_->residency(); }
  // Host memory the block provider holds (pinned rings or the pinned copy of the streamed blocks).
  std::size_t host_bytes() const { return host_bytes_; }
  // Streaming diagnostics of the last forward() / summed since reset_stream_stats() (all zero when every block is
  // resident). Reading them waits for the forward's GPU work.
  const StreamStats& last_stream_stats() { return blocks_->last_stats(); }
  const StreamStats& total_stream_stats() { return blocks_->total_stats(); }
  void reset_stream_stats() { blocks_->reset_total(); }
  // Fault injection on the streaming copy path (tests).
  void set_copy_hook(CopyHook hook) { blocks_->set_copy_hook(std::move(hook)); }

  // Stack bytes encode_text() needs for a prompt of `text_length` tokens.
  static std::size_t encode_text_workspace_bytes(std::int64_t text_length);
  // Text refiner: text_embeds CUDA bf16 [S, kTextDim] (the text encoder's last hidden state of one row,
  // unpadded) -> out CUDA bf16 [S, kDim] (caller-owned, kept across steps).
  void encode_text(const Tensor& text_embeds, const Tensor& out, cudaStream_t stream);

  // Time embedding and modulation of one step (P:232-241), shared by all rows: mod CUDA bf16 [6*kDim]
  // (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f), head CUDA bf16 [2*kDim] (shift, scale).
  // forward() calls it; public for parity tests.
  void time_modulation(float timestep, const Tensor& mod, const Tensor& head, cudaStream_t stream);

  // Distilled checkpoints (P6 step 41, distill.h): replaces the step embedder and the two modulation biases by `te`
  // (nullptr: the loaded weights again, bit-identical to before). With a replacement, time_modulation computes in fp32
  // as the published files require: the embedder on the host (time_vector_fp32, cached per timestep), the two
  // modulation linears on the device (the model's bf16 weights widened, fp32 accumulation, the replacement biases),
  // and rounds mod / head to bf16 once. Holds a small device buffer (~110 KB) outside the arena. Call between
  // forwards; `te` is copied.
  void set_time_embedding(const TimeEmbeddingFp32* te);
  bool has_time_embedding() const { return te_ != nullptr; }

  // Stack bytes forward() needs for a latent grid of height x width (= image size / 16) and the rows'
  // text lengths, in row order, with block path `path`; also enough for any subset of these rows. The fused paths
  // ask cuBLASLt on the current device under the route installed now (fused_big_cols); forward() snapshots the route
  // at its start, so a poisoning in between makes it need the op-by-op size.
  static std::size_t workspace_bytes(std::int64_t height, std::int64_t width, std::span<const std::int64_t> text_lengths,
                                     BlockPath path = BlockPath::kFused);
  // The same for compute config `config`, under `route` (nullptr: a fresh snapshot) as a pinned request route.
  static std::size_t workspace_bytes(std::int64_t height, std::int64_t width, std::span<const std::int64_t> text_lengths,
                                     BlockPath path, const ComputeConfig& config, const FusedRouteSnapshot* route);
  // Token offset of each row in the joint sequence, plus the total as the last element.
  static std::vector<std::int64_t> token_offsets(std::int64_t height, std::int64_t width,
                                                 std::span<const std::int64_t> text_lengths);

  // One DiT evaluation for every row: latents CUDA fp32 [kLatentChannels, h, w] (shared by all rows),
  // timestep as the pipeline passes it (scheduler timestep, fp32). out[r]: CUDA fp32
  // [kLatentChannels, h, w] velocity of row r (= transformer(...).sample.float(), P:413).
  void forward(const Tensor& latents, float timestep, std::span<const Row> rows, std::span<const Tensor> out,
               cudaStream_t stream, const BlockHook& hook = {});

 private:
  struct Linear {
    Tensor w;     // bf16 [N, K]
    Tensor bias;  // bf16 [N] or empty
  };
  struct TextBlock {
    Tensor norm1, norm2, q_norm, k_norm;
    Linear qkv, proj;
    Linear up_gate;  // ffn_up and ffn_gate fused along N as [up | gate] (situ_glu's [hidden | gate])
    Linear down;
  };
  using Block = BlockTensors;
  // Token range [t0, t1) of a run of contiguous rows; cu_seqlens: their prefix sums from 0.
  using Run = BlockRun;
  struct Activations;

  MossDit(const weights::ModelIndex& index, const quant::QuantIndex* quant, runtime::DeviceArena& arena,
          runtime::StreamSet& streams, bool direct_io, const std::atomic<bool>* cancel, const Residency& residency,
          const lora::LoraStack* merges, const weights::HostFiles* host_files);
  void refresh_config(AttentionMode attention);  // config_ from the loaded linears, `attention` and the LoRA
  void linear(const Tensor& out, const Tensor& x, const Linear& l, cudaStream_t stream);
  void linear(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream);
  void run_block(int bi, const Block& b, const Activations& a, const Run& run, cudaStream_t stream);

  runtime::DeviceArena& arena_;
  const quant::QuantIndex* quant_ = nullptr;
  ComputeConfig config_;
  Int4Route int4_route_ = Int4Route::kAuto;
  std::optional<FusedRouteSnapshot> request_route_;
  const lora::SideBranches* lora_ = nullptr;
  const InputExtension* input_ext_ = nullptr;
  // Runtime-LoRA merges: the pristine weights of the merged modules and where each module's merged weights live.
  struct MergeTarget {
    Tensor w, bias, scale;  // bias / scale empty when absent / bf16
    int group = 0;          // INT8 ConvRot group; 0: bf16
  };
  // A host copy of a merge target's weights, owned (also for a pinned host target, so nothing changes until store()):
  // read from the target (download) or only allocated (a re-merge overwrites every byte from the pristine copy).
  class MergeStaging {
   public:
    MergeStaging(const MergeTarget& t, bool download);
    // `copy`'s bytes for target `t`; nullptr: detached (a memo entry, stored only through a copy for a live target).
    MergeStaging(const MergeTarget* t, const MergeStaging& copy);
    MergeStaging(MergeStaging&&) noexcept = default;
    std::size_t bytes() const { return hw_.size() + hb_.size() + hs_.size(); }
    MergeStaging(const MergeStaging&) = delete;
    MergeStaging& operator=(const MergeStaging&) = delete;
    template <typename T>
    T* w() const { return reinterpret_cast<T*>(const_cast<std::byte*>(hw_.data())); }
    std::uint16_t* bias() const { return hb_.empty() ? nullptr : reinterpret_cast<std::uint16_t*>(const_cast<std::byte*>(hb_.data())); }
    float* scale() const { return hs_.empty() ? nullptr : reinterpret_cast<float*>(const_cast<std::byte*>(hs_.data())); }
    // Writes the copy into the target: device targets uploaded and synchronized (visible to every stream on return),
    // host targets copied.
    void store() const;

   private:
    const MergeTarget* target;
    std::vector<std::byte> hw_, hb_, hs_;
  };
  // staging = merge(pristine copy of `module`) for `stack`'s strengths, from `pristine` (host math only).
  static void remerge_into(const lora::PristineWeights& pristine, const lora::LoraStack& stack, const std::string& module,
                           const MergeTarget& t, MergeStaging& staging);
  std::unique_ptr<lora::PristineWeights> pristine_;
  std::vector<std::pair<std::string, MergeTarget>> merge_targets_;
  // rebase_merges' memo: the merged bits of the stack named memo_key_, by module.
  std::string memo_key_;
  std::vector<std::pair<std::string, MergeStaging>> memo_;
  // `module`'s weights in `m` (w empty when absent); throws for a quantized format merges do not support.
  MergeTarget merge_target(const weights::TensorMap& m, const std::string& module) const;
  // The views of the always-resident groups (globals, text_embedder): rebase_merges' in-place merge targets.
  weights::TensorMap resident_views_;
  kernels::Gemm gemm_;
  void* gemm_workspace_ = nullptr;
  LoadStats load_stats_;
  const std::atomic<bool>* cancel_ = nullptr;
  StepCacheHook* step_cache_ = nullptr;
  LinearInputHook linear_input_hook_;
  cudaEvent_t block_done_[2] = {nullptr, nullptr};  // blocks k - 2 and k - 1 of a forward (cancel throttle)

  Linear video_embedder_, time_in_, time_out_, block_modulation_, head_modulation_, text_proj_;
  // set_time_embedding: the replacement, its device arrays [time fp32 kDim | mod bias 6 kDim | head bias 2 kDim] and the
  // host embedder's outputs per timestep (bit pattern).
  std::unique_ptr<TimeEmbeddingFp32> te_;
  DeviceBuffer te_buf_;
  std::map<std::uint32_t, std::vector<float>> te_cache_;
  Tensor final_norm_, final_linear_;
  TextBlock text_blocks_[2];
  std::unique_ptr<BlockWeights> blocks_;
  std::size_t host_bytes_ = 0;
  BlockPath path_ = BlockPath::kFused;
  StepGraphs graphs_{kGraphCapacity};  // per-block graphs of a few step shapes
  std::uint64_t last_step_hash_ = 0;  // step shape of the previous forward (graphs are captured once it repeats)
};

}  // namespace mi::moss_dit
