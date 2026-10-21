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
// Every op follows the reference's order and bf16/fp32 rounding (docs/model-notes.md §6).
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
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "core/tensor.h"
#include "kernels/basic/gemm.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/dit_graph.h"
#include "models/moss_dit/dit_weights.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "weights/model_index.h"

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

struct LoadStats {
  double seconds = 0;       // wall time, disk to resident (includes the final copy-stream sync)
  std::uint64_t bytes = 0;  // payload bytes read
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
                 // ~0 measured gain on an idle host, docs/experiment-log.md E13; holds graph memory)
};
const char* block_path_name(BlockPath p);
// "op_by_op", "fused" or "fused_graphs"; throws std::invalid_argument otherwise.
BlockPath parse_block_path(std::string_view name);
// Device memory the CUDA graphs of one model hold outside the arena at most (kFusedGraphs, every block resident):
// conservative, per cached graph times the cache's capacity.
std::size_t graph_cache_bytes();

class MossDit {
 public:
  // Persistent arena bytes the constructor allocates (weights + GEMM workspace), every block resident or with
  // `residency`.
  static std::size_t persistent_bytes(const weights::ModelIndex& index);
  static std::size_t persistent_bytes(const weights::ModelIndex& index, const Residency& residency);
  // Stack bytes forward() adds to workspace_bytes() for streamed blocks (the device block buffers; 0 if none).
  static std::size_t stream_workspace_bytes(const weights::ModelIndex& index, const Residency& residency);

  // Uploads the transformer groups (globals, text_embedder and the resident blocks) of `index` into the
  // arena's persistent region and waits until they are resident; streamed blocks are validated (and copied
  // to pinned host memory for StreamSource::kHostPinned). direct_io: O_DIRECT reads (bypass the page cache)
  // instead of buffered reads for the load. The arena and index must outlive the model. `cancel` (optional,
  // must outlive the model) is checked before every weight group is read and between the blocks of every
  // forward(); once set, the call waits for its queued copies/kernels and throws Cancelled.
  MossDit(const weights::ModelIndex& index, runtime::DeviceArena& arena, runtime::StreamSet& streams,
          bool direct_io = false, const std::atomic<bool>* cancel = nullptr, const Residency& residency = {});
  ~MossDit();
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

  // Replaces the cancel flag (nullptr: none). While a flag is set, forward() keeps the host at most one block
  // ahead of the GPU, so a cancel lands within about one block.
  void set_cancel(const std::atomic<bool>* cancel) { cancel_ = cancel; }

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

  // Stack bytes forward() needs for a latent grid of height x width (= image size / 16) and the rows'
  // text lengths, in row order, with block path `path`; also enough for any subset of these rows. The fused paths
  // ask cuBLASLt on the current device (fused_big_cols).
  static std::size_t workspace_bytes(std::int64_t height, std::int64_t width, std::span<const std::int64_t> text_lengths,
                                     BlockPath path = BlockPath::kFused);
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

  void linear(const Tensor& out, const Tensor& x, const Linear& l, cudaStream_t stream);
  void linear(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream);
  void run_block(const Block& b, const Activations& a, const Run& run, cudaStream_t stream);

  runtime::DeviceArena& arena_;
  kernels::Gemm gemm_;
  void* gemm_workspace_ = nullptr;
  LoadStats load_stats_;
  const std::atomic<bool>* cancel_ = nullptr;
  cudaEvent_t block_done_[2] = {nullptr, nullptr};  // blocks k - 2 and k - 1 of a forward (cancel throttle)

  Linear video_embedder_, time_in_, time_out_, block_modulation_, head_modulation_, text_proj_;
  Tensor final_norm_, final_linear_;
  TextBlock text_blocks_[2];
  std::unique_ptr<BlockWeights> blocks_;
  std::size_t host_bytes_ = 0;
  BlockPath path_ = BlockPath::kFused;
  StepGraphs graphs_{kGraphCapacity};  // per-block graphs of a few step shapes
  std::uint64_t last_step_hash_ = 0;  // step shape of the previous forward (graphs are captured once it repeats)
};

}  // namespace mi::moss_dit
