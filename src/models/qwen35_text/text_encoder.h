// Qwen3.5 text encoder as MOSS-Image uses it: Qwen3_5Model(input_ids).last_hidden_state for text-only
// input, no cache (docs/model-notes.md §7). 32 decoder layers (GatedDeltaNet, full gated attention every
// 4th), each RMSNorm -> token mixer -> residual -> RMSNorm -> SiLU MLP -> residual, then the final norm.
//
// Execution: the embedding table stays host-resident (memory-mapped; only the gathered rows are uploaded).
// All sequences (cond and negative prompts) go through each layer once as one unpadded varlen batch, so
// every layer group is read once per encode() however many sequences there are: elementwise ops see the
// packed tokens, attention, conv and the GDN rule are per sequence via cu_seqlens, and GEMMs and norms are
// launched per sequence because their numerics depend on the row count (cuBLASLt algorithm choice, torch's
// reduction shape). Results are bit-identical to encoding each prompt alone, unpadded, as the reference does.
//
// Weights (P3 streaming, the default): `weight_buffers` device buffers rotate over the layers. Reader threads,
// each with its own pinned ring and copy stream, fill the buffer of layer k+1 (k+2, ...) while layer k runs on
// the compute stream; the compute stream waits on the readers' copy-done events, and a reader waits on the
// compute-done event of the layer that last used a buffer before overwriting it. Off (`streaming = false`):
// the P2 path, which loads a layer synchronously into one buffer and then runs it. Both give bit-identical
// results.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/tensor.h"
#include "runtime/device_arena.h"
#include "weights/model_index.h"

namespace mi::qwen35_text {

inline constexpr int kHidden = 4096;
inline constexpr int kIntermediate = 12288;
inline constexpr int kNumLayers = 32;
inline constexpr int kNumHeads = 16;
inline constexpr int kNumKvHeads = 4;
inline constexpr int kAttnHeadDim = 256;

// Engine numerics version of the encode path: bump it whenever a kernel or the op order on that path changes any
// output bit (the embedding cache keys on it, so stale embeddings become misses).
inline constexpr std::uint32_t kNumericsVersion = 1;

// What encode()'s output bits depend on besides the weights and the tokens: kNumericsVersion, the precision (bf16
// weights and activations, the only one so far), the current device's compute capability and the cuBLASLt version
// (its GEMM algorithm choice). The streaming, weight-source and buffer options never change a bit (tested), so
// they are not part of it.
std::uint64_t numerics_id();

// Where streamed layer weights come from.
enum class WeightSource {
  kFile,          // safetensors files: pread into the pinned rings (page cache, or O_DIRECT with direct_io)
  kHostPageable,  // all layer groups copied once into pageable host RAM, staged through the pinned rings
  kHostPinned,    // all layer groups copied once into pinned (page-locked) host RAM, DMA'd to the device
};

// Members after direct_io have defaults, so `EncoderOptions{direct_io}` keeps working.
struct EncoderOptions {
  // Read layer weights with O_DIRECT (bypassing the page cache). Off: buffered reads, which a warm page
  // cache serves from RAM. On btrfs with compressed extents the kernel serves O_DIRECT reads buffered.
  bool direct_io = false;
  bool streaming = true;  // false: P2 synchronous loading (weight_buffers and the reader settings are ignored)
  int weight_buffers = 2;  // device layer buffers, >= 1 (1 = no copy/compute overlap)
  WeightSource source = WeightSource::kFile;
  int reader_threads = 4;  // concurrent readers, each with its own copy stream (and pinned ring unless kHostPinned)
  std::size_t chunk_bytes = std::size_t{4} << 20;  // read and copy granularity
  int ring_slots = 4;      // pinned slots per reader ring, chunk_bytes + kDirectAlign each
};

// Raw output of decoder layer `layer` (before the final norm), all sequences packed: bf16 [T, 4096] on
// `stream`. Valid only during the call; copy what you need (stream-ordered).
using LayerHook = std::function<void(int layer, const Tensor& hidden, cudaStream_t stream)>;

// Test hook for fault injection, called on reader thread `reader` before it moves its share of layer `layer`, and
// once more with layer == kNumLayers after its last copy; an exception thrown here fails that reader.
using ReaderHook = std::function<void(int reader, int layer)>;

struct EncodeOptions {
  LayerHook layer_hook;
  // Checked before every decoder layer is issued; once set, encode() stops the readers, waits for every queued
  // copy and kernel, releases its arena memory and throws Cancelled. While a flag is given the host issues at most
  // one layer ahead of the GPU, so a cancel lands within about one layer.
  const std::atomic<bool>* cancel = nullptr;
  ReaderHook reader_hook;  // streaming only
};

// encode() saw EncodeOptions::cancel set. The encoder stays usable.
struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("qwen35_text: encode cancelled") {}
};

// Times of the last encode(). Synchronous path: host clock around synchronized work. Streaming: the GPU
// timeline (events), since loading and compute overlap.
struct EncodeStats {
  std::int64_t tokens = 0;
  double embed_s = 0;        // host row gather + upload
  double host_load_s = 0;    // host-resident sources: first-time load of the host copy within this call
  // Synchronous: file/page cache -> device, summed over the layers. Streaming: wall time from the start of
  // the readers to the last byte on the device.
  double weight_load_s = 0;
  double weight_wait_s = 0;  // streaming: compute stream idle waiting for weights (copy time not hidden)
  double compute_s = 0;      // decoder layers + final norm on the GPU
  double total_s = 0;
  std::uint64_t weight_bytes = 0;         // layer group bytes moved to the device in this call
  std::uint64_t read_chunks = 0;          // streaming, ring-backed sources: ring slot fills
  std::vector<std::uint64_t> layer_bytes; // per layer (streaming: counted by the rings / pinned copies)

  // Fraction of the weight-loading time hidden behind compute (streaming).
  double overlap() const { return weight_load_s > 0 ? std::max(0.0, 1.0 - weight_wait_s / weight_load_s) : 0.0; }
};

class TextEncoder {
 public:
  // Arena stack bytes one encode() or encode_to_host() of sequences with these token counts needs at its peak
  // (outputs included), exactly: the plan of encode()'s allocations, with `options`' weight buffers (one when
  // not streaming) sized for the largest layer of `index`. Throws like the constructor on a malformed index.
  static std::size_t workspace_bytes(const weights::ModelIndex& index, std::span<const std::int64_t> lengths,
                                     const EncoderOptions& options);
  // Same, for the default options.
  static std::size_t workspace_bytes(const weights::ModelIndex& index, std::span<const std::int64_t> lengths);

  // Host memory an encoder with `options` holds, exactly: pinned rings (reader_threads x ring_slots x slot;
  // allocated by the constructor) plus, for host-resident sources, the copy of every layer group (allocated by
  // preload_host() or the first encode(); pinned or pageable as the source says). Excludes the mmapped
  // embedding table (file-backed) and transient per-call host buffers.
  static std::size_t host_bytes(const weights::ModelIndex& index, const EncoderOptions& options);

  // `index` must contain the text_encoder groups and outlive the encoder; `arena` serves every device
  // allocation (weight buffer, activations, outputs) from its stack region during encode(). All work is
  // issued on `stream`.
  TextEncoder(const weights::ModelIndex& index, runtime::DeviceArena& arena, cudaStream_t stream,
              EncoderOptions options = {});
  ~TextEncoder();
  TextEncoder(const TextEncoder&) = delete;
  TextEncoder& operator=(const TextEncoder&) = delete;

  // last_hidden_state of every sequence: bf16 [S_i, 4096] device views. The outputs are allocated on the
  // arena stack below encode()'s own scratch (which is released before returning), so they stay valid
  // until the caller rolls the stack back past them, e.g. with a StackScope opened before the call.
  // Sequences must be non-empty with ids below vocab_size(). Synchronizes `stream` before returning.
  std::vector<Tensor> encode(const std::vector<std::vector<std::int32_t>>& sequences, const EncodeOptions& options = {});

  // Same, copied to host as bf16 bit patterns [S_i * 4096]; the arena stack is left as it was.
  std::vector<std::vector<std::uint16_t>> encode_to_host(const std::vector<std::vector<std::int32_t>>& sequences,
                                                         const EncodeOptions& options = {});

  // Host-resident sources: loads the host copy of every layer group now (otherwise the first encode() does).
  // No-op for kFile or when already loaded.
  void preload_host();
  // Frees the host copy (the next encode() of a host-resident source loads it again).
  void release_host();
  std::size_t host_resident_bytes() const;  // currently held host copy

  const EncoderOptions& options() const;
  const EncodeStats& last_stats() const;
  std::size_t weight_buffer_bytes() const;  // one buffer
  std::int64_t vocab_size() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mi::qwen35_text
