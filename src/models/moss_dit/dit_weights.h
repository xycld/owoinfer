// Block weight provider of the MOSS DiT (P4 step 25): where the weights of the 42 blocks live during forward().
//
// Each block is either resident (uploaded once into the arena's persistent region) or streamed. Streamed blocks are
// copied every forward() into a few device block buffers carved from the arena stack:
//   - source kFile: reader threads pread the safetensors files (buffered, so a warm page cache serves them from RAM;
//     O_DIRECT with direct_io) into their own pinned rings and copy on their own streams;
//   - source kHostPinned: a pinned host copy of the streamed blocks (made at load) is DMA'd by one copy stream.
// Streamed block j of a forward goes to buffer j % buffers, so buffers - 1 blocks are copied ahead of the one the
// compute stream runs. The compute stream waits on the copy-done events of a block before running it; a reader
// waits on the compute-done event of the block that last used a buffer before overwriting it. Every row of the step
// (cond, uncond, SLG) runs against the one copy, so each streamed block crosses PCIe once per forward().
// Resident blocks are spread evenly over the depth (resident_mask), so copies run during resident compute too.
//
// The copied bytes and their device layout are those of the resident upload (same tensor order, kArenaAlign
// offsets, 256-byte aligned buffers), so streamed and resident forwards are bit-identical (tested).
// P6: with a quantized index (weights/quant, a single-file DiT) a block's linears are INT8 ConvRot (I8 weight + F32
// per-row scale), INT4 ConvRot (packed nibbles + F32 per-row scale), NVFP4 (packed E2M1 + swizzled UE4M3 block scales;
// the F32 tensor scale is a host value of the index) or bf16, each as the file stores it; every linear is checked
// against the spec its GEMM implements (quant::view), and a block's bytes, resident or streamed, are its stored bytes
// (an INT8 block is about half a bf16 one, an INT4 block about a quarter, an NVFP4 block 0.28 of it).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/tensor.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/upload.h"

namespace mi::moss_dit {

inline constexpr int kNumBlocks = 42;  // == kLayers (moss_dit.h)

enum class StreamSource {
  kFile,        // the weight files through pinned rings (page cache, or O_DIRECT with direct_io)
  kHostPinned,  // a pinned host copy of the streamed blocks, DMA'd directly (host memory: their full size)
};
const char* stream_source_name(StreamSource s);

struct Residency {
  int resident_blocks = kNumBlocks;  // blocks kept in device memory; the other blocks stream every forward()
  int buffers = 3;                   // device block buffers when streaming (>= 2; at most one per streamed block is
                                     // allocated): buffers - 1 blocks of lookahead
  StreamSource source = StreamSource::kFile;
  bool direct_io = false;            // kFile: O_DIRECT reads
  int reader_threads = 4;            // kFile: concurrent readers, each with its own pinned ring and copy stream
  std::size_t chunk_bytes = std::size_t{4} << 20;  // read/copy granularity
  int ring_slots = 4;                // kFile: pinned slots per reader ring

  bool streamed() const { return resident_blocks < kNumBlocks; }
  bool operator==(const Residency&) const = default;
};

// Which blocks are resident for `resident_blocks` of kNumBlocks: evenly spread, block i streams iff
// floor((i + 1) * S / 42) > floor(i * S / 42) with S = 42 - resident_blocks (so with any resident block, block 0 is
// resident and copies start during its compute).
std::bitset<kNumBlocks> resident_mask(int resident_blocks);

// An INT8 ConvRot (G = 256) linear as its GEMM consumes it (kernels/quant): weight I8 [N, K] row-major, the rotated
// and quantized W·H, and its per-row scale F32 [N]. Empty (null data) when the linear is bf16.
struct Int8Linear {
  Tensor w;
  Tensor scale;
  bool empty() const { return w.data == nullptr; }
};

// An INT4 ConvRot (G = 256) linear as its GEMM consumes it (kernels/quant4): weight I8 [N, K/2], the packed signed
// nibbles of the rotated and quantized W·H (even column in the low nibble), and its per-row scale F32 [N]. Empty when
// the linear is not INT4.
struct Int4Linear {
  Tensor w;
  Tensor scale;
  bool empty() const { return w.data == nullptr; }
  std::int64_t k() const { return 2 * w.dim(1); }
};

// An NVFP4 linear as the sm_120a GEMMs consume it (kernels/nvfp4): weight U8 [N, K/2], E2M1 codes two per byte
// (hi_first, as stored), its UE4M3 block scales in the swizzled layout (U8 [round_up(N, 128), round_up(K/16, 4)]) and
// the F32 tensor scale (weight_scale_2, the GEMM's w_scale2). Empty when the linear is not NVFP4.
struct Nvfp4Linear {
  Tensor w;
  Tensor sf;
  float scale2 = 1.0f;
  bool empty() const { return w.data == nullptr; }
  std::int64_t k() const { return 2 * w.dim(1); }
};

// Weight views of one block (P:126-148): norms, fused q/k/v, o, fused up/gate and down projections. Each linear is
// exactly one of bf16 (the Tensor), INT8 ConvRot (the Int8Linear), INT4 ConvRot (the Int4Linear) or NVFP4 (the
// Nvfp4Linear). q, k and v share one GEMM: fused (qkv / qkv8 / qkv4 / qkv_nvfp4) when they share a precision (NVFP4:
// and a tensor scale, as the parts of a file's fused proj.qkv do); otherwise qkv_split holds each part.
struct BlockTensors {
  Tensor norm1, norm2, q_norm, k_norm;
  Tensor qkv;  // proj.q/k/v fused along N: [3 * dim, dim]
  Tensor o, up_gate, down;
  Int8Linear qkv8, o8, up_gate8, down8;  // qkv8: [3 * dim, dim] with a [3 * dim] scale
  Int4Linear qkv4, o4, up_gate4, down4;  // qkv4: [3 * dim, dim / 2] with a [3 * dim] scale
  // qkv_nvfp4: [3 * dim, dim / 2] with the three parts' block scales back to back (whole 128-row tiles: the fused
  // matrix's swizzled scales)
  Nvfp4Linear qkv_nvfp4, o_nvfp4, up_gate_nvfp4, down_nvfp4;
  bool qkv_split = false;
  Tensor qkv_part[3];                    // qkv_split: bf16 q, k, v (empty where quantized)
  Int8Linear qkv8_part[3];               // qkv_split: INT8 q, k, v (empty elsewhere)
  Int4Linear qkv4_part[3];               // qkv_split: INT4 q, k, v (empty elsewhere)
  Nvfp4Linear qkv_nvfp4_part[3];         // qkv_split: NVFP4 q, k, v (empty elsewhere)

  // Some linear of the block is INT4.
  bool int4() const {
    return !qkv4.empty() || !o4.empty() || !up_gate4.empty() || !down4.empty() || !qkv4_part[0].empty() ||
           !qkv4_part[1].empty() || !qkv4_part[2].empty();
  }
  // Some linear of the block is NVFP4.
  bool nvfp4() const {
    return !qkv_nvfp4.empty() || !o_nvfp4.empty() || !up_gate_nvfp4.empty() || !down_nvfp4.empty() ||
           !qkv_nvfp4_part[0].empty() || !qkv_nvfp4_part[1].empty() || !qkv_nvfp4_part[2].empty();
  }
  // Some linear of the block is INT8, INT4 or NVFP4.
  bool quantized() const {
    return int4() || nvfp4() || !qkv8.empty() || !o8.empty() || !up_gate8.empty() || !down8.empty() ||
           !qkv8_part[0].empty() || !qkv8_part[1].empty() || !qkv8_part[2].empty();
  }
};

// Fault and delay injection (tests): called on reader thread `reader` before each chunk of its share of streamed block
// `block`; an exception thrown here fails the forward() (rethrown on the compute thread), leaving the model usable.
using CopyHook = std::function<void(int reader, int block)>;

// Streaming diagnostics of one forward() (P4 step 28), from CUDA events on the GPU timeline.
struct StreamStats {
  int streamed_blocks = 0;
  std::uint64_t bytes = 0;  // copied to the device
  double copy_s = 0;        // time the copy streams were busy (union of the per-block copy intervals)
  double stall_s = 0;       // compute stream waiting for streamed weights
  int stalls = 0;           // streamed blocks the compute stream had to wait for (> 50 us)
  double forward_s = 0;     // compute stream from the pass start (before the time embedding) to the last block

  double gbps() const { return copy_s > 0 ? static_cast<double>(bytes) / copy_s / 1e9 : 0.0; }
  // Fraction of the copy time hidden behind compute.
  double hidden() const { return copy_s > 0 ? std::max(0.0, 1.0 - stall_s / copy_s) : 0.0; }
  void add(const StreamStats& o);
};

namespace detail {
// `g` with the tensors `names` moved to the front in the given order: uploads place tensors in listed order, each on
// a kArenaAlign boundary, so equally shaped neighbours end up contiguous and can be read as one fused matrix.
weights::WeightGroup with_front(const weights::WeightGroup& g, const std::vector<std::string>& names);
// The bf16 tensor `name` of `m`, checked against `shape`.
Tensor take(const weights::TensorMap& m, const std::string& name, std::initializer_list<std::int64_t> shape);
// Weights uploaded back to back (see with_front) viewed as one tensor concatenated along dim 0 (cols 0: 1-D).
Tensor fuse(const weights::TensorMap& m, const std::vector<std::string>& names, std::int64_t rows, std::int64_t cols);
// Whether q, k and v of block `block` run as one GEMM: the same spec (NVFP4: and the same tensor scale). Always for
// the bundle's bf16 layout (quant == nullptr).
bool qkv_fused(int block, const quant::QuantIndex* quant);
// The upload order of block `block`'s group: q, k, v weights (and, when q/k/v share a quantized spec, their scales;
// NVFP4: their block scales) first, so the fused QKV weight (and scale) is contiguous.
std::vector<std::string> block_front(int block, const quant::QuantIndex* quant);
// The views of block `block` out of its uploaded group `m` (any device layout of it: resident or a stream buffer).
// quant == nullptr: the bundle's bf16 layout. Otherwise every linear goes through quant::view with the spec it is
// stored as, which must be bf16, INT8 ConvRot G = 256, INT4 ConvRot G = 256 or NVFP4 (std::runtime_error naming the
// linear otherwise).
BlockTensors block_views(int block, const weights::TensorMap& m, const quant::QuantIndex* quant);
}  // namespace detail

class BlockWeights {
 public:
  // Uploads one weight group into persistent arena memory (the model's loader: cancel checks, stats).
  using Upload = std::function<weights::TensorMap(const weights::WeightGroup&)>;

  // Device bytes of one block group (all blocks have the same layout): the streaming unit.
  static std::size_t block_device_bytes(const weights::ModelIndex& index);
  // Persistent arena bytes of the resident blocks.
  static std::size_t resident_bytes(const weights::ModelIndex& index, const Residency& residency);
  // Arena stack bytes a streaming forward() adds (min(buffers, streamed blocks) device block buffers); 0 when every
  // block is resident.
  static std::size_t buffer_bytes(const weights::ModelIndex& index, const Residency& residency);
  // Host memory the provider holds while it lives: pinned rings (kFile) or the pinned copy (kHostPinned).
  static std::size_t host_bytes(const weights::ModelIndex& index, const Residency& residency);

  // Resident blocks go through `upload` in block order; `before_group` runs before every streamed block is
  // validated or copied to pinned host memory (cancel checks). Throws std::invalid_argument on bad options.
  // quant: the specs of a quantized index (`index` must be quant->index), or nullptr for the bundle's bf16 layout.
  // host_transform (optional): applied to each streamed block's pinned host copy (kHostPinned) once it is read: the
  // model's LoRA merges, so the blocks stream merged bytes.
  BlockWeights(const weights::ModelIndex& index, runtime::DeviceArena& arena, const Residency& residency,
               const Upload& upload, const std::function<void(const weights::WeightGroup&)>& before_group,
               const quant::QuantIndex* quant = nullptr,
               const std::function<void(const weights::TensorMap&)>& host_transform = {});
  ~BlockWeights();
  BlockWeights(const BlockWeights&) = delete;
  BlockWeights& operator=(const BlockWeights&) = delete;

  const Residency& residency() const { return residency_; }
  // buffer_bytes() of this provider: the stack bytes a forward's streaming pass adds.
  std::size_t buffer_bytes() const { return buffer_bytes(index_, residency_); }
  bool resident(int block) const { return mask_.test(static_cast<std::size_t>(block)); }
  int streamed_blocks() const { return static_cast<int>(streamed_.size()); }
  std::size_t host_resident_bytes() const;  // pinned host copy held (kHostPinned)

  // sm_75 (kernels::sm75::fp16_weights()): converts the resident blocks' bf16 linear weights to fp16 in place, so the
  // fp16 GEMM path can consume them (the views are marked kF16). Streamed blocks are converted by Pass::acquire(),
  // after each block's copy. No-op on sm_80+ devices and when the build has no sm_75 kernels.
  void convert_weights_to_f16(cudaStream_t stream);
  void set_copy_hook(CopyHook hook) { hook_ = std::move(hook); }

  // Stats of the last finished forward() (synchronizes on its events the first time it is read) and the sum over
  // every forward since reset_total().
  const StreamStats& last_stats();
  const StreamStats& total_stats();
  void reset_total();


 private:
  struct Chunk {  // a slice of one tensor that one reader moves with one ring fill (or one pinned DMA)
    const weights::IndexedTensor* tensor;
    std::uint64_t rel;      // offset within the tensor
    std::size_t size;
    std::byte* dst;         // device
    const std::byte* host;  // kHostPinned: source bytes
  };

 public:
  // One forward()'s block weights. Construct after the forward's own stack allocations (it carves the buffers out of
  // the arena stack and starts the readers), then acquire(b) / release(b) every block in order 0..41, then finish().
  // Destroying an unfinished pass (an exception, a cancel) stops the readers and waits for every queued copy and
  // kernel, so the caller's stack rollback is safe. `cancel` (optional, outlives the pass): the readers check it
  // between chunks and stop the pass, and acquire() then throws Cancelled, so a cancel lands within one chunk even
  // when the weights come from slow storage.
  class Pass {
   public:
    Pass(BlockWeights& w, cudaStream_t stream, const std::atomic<bool>* cancel = nullptr);
    ~Pass();
    Pass(const Pass&) = delete;
    Pass& operator=(const Pass&) = delete;

    // The compute stream waits until the block's weights are on the device. Throws Cancelled (moss_dit.h) when the
    // pass's cancel flag stopped the readers.
    const BlockTensors& acquire(int block);
    // The block's kernels are all enqueued: its buffer may be refilled once they have run.
    void release(int block);
    // Joins the readers (rethrowing a reader's error) and records the end of the block loop.
    void finish();
    // Instead of finish(), before the last block (a step-cache reuse, moss_dit.h StepCacheHook): stops the readers,
    // waits for their queued copies and kernels, and rethrows a reader's error recorded before the readers are joined.
    // The pass's transfers are not added to the streaming stats.
    void stop_early();

   private:
    void abort() noexcept;

    BlockWeights& w_;
    cudaStream_t stream_;
    std::vector<BlockTensors> views_;          // per streamed block (sequence index)
    std::vector<std::vector<Chunk>> chunks_;   // per streamed block; reader r takes chunks r, r + R, ...
    std::vector<std::thread> threads_;
    bool finished_ = false;
  };

 private:
  struct Reader;
  struct Sync;

  BlockTensors make_views(int block, const weights::TensorMap& m) const;
  void create_streaming_resources();
  void release() noexcept;
  void reader_loop(std::size_t r, const std::vector<std::vector<Chunk>>& chunks, Sync& sync);
  void collect_stats();

  const weights::ModelIndex& index_;
  const quant::QuantIndex* quant_ = nullptr;
  runtime::DeviceArena& arena_;
  Residency residency_;
  std::bitset<kNumBlocks> mask_;
  int nbuf_ = 0;  // device block buffers of a pass
  std::size_t block_bytes_ = 0;
  std::vector<weights::WeightGroup> ordered_;  // per block, fused tensors first
  std::vector<BlockTensors> resident_;         // per block (empty views when streamed)
  std::vector<int> streamed_;                  // streamed block indices, ascending (the sequence)
  std::vector<int> seq_of_;                    // block -> sequence index, -1 when resident
  std::vector<weights::HostGroup> host_;       // kHostPinned: per sequence index
  std::unique_ptr<weights::ReaderCache> readers_;
  std::vector<std::unique_ptr<Reader>> rd_;
  std::vector<cudaEvent_t> compute_done_;      // per buffer
  cudaEvent_t buffers_free_ = nullptr;         // compute stream at the start of a pass
  // Timing (per sequence index): compute stream before/after its waits; the forward's block loop start/end.
  std::vector<cudaEvent_t> wait_begin_, wait_end_;
  cudaEvent_t loop_begin_ = nullptr, loop_end_ = nullptr;
  CopyHook hook_;
  std::unique_ptr<Sync> sync_;  // the running pass's host-side handshake
  bool stats_pending_ = false;
  StreamStats last_, total_;
};

}  // namespace mi::moss_dit
