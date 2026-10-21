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

// Weight views of one block (P:126-148): norms, fused q/k/v, o, fused up/gate and down projections.
struct BlockTensors {
  Tensor norm1, norm2, q_norm, k_norm;
  Tensor qkv;  // proj.q/k/v fused along N: [3 * dim, dim]
  Tensor o, up_gate, down;
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
  BlockWeights(const weights::ModelIndex& index, runtime::DeviceArena& arena, const Residency& residency,
               const Upload& upload, const std::function<void(const weights::WeightGroup&)>& before_group);
  ~BlockWeights();
  BlockWeights(const BlockWeights&) = delete;
  BlockWeights& operator=(const BlockWeights&) = delete;

  const Residency& residency() const { return residency_; }
  bool resident(int block) const { return mask_.test(static_cast<std::size_t>(block)); }
  int streamed_blocks() const { return static_cast<int>(streamed_.size()); }
  std::size_t host_resident_bytes() const;  // pinned host copy held (kHostPinned)
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
