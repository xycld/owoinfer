// Moves weight groups (see model_index.h) from the safetensors files into device memory or host
// memory, returning name -> mi::Tensor views. Tensors keep the group's listed order in memory; each
// device tensor starts on a runtime::kArenaAlign boundary.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"

namespace mi::weights {

struct UploadError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// A group's bytes on disk do not match the checksum recorded in the index.
struct ChecksumError : UploadError {
  using UploadError::UploadError;
};

using TensorMap = std::map<std::string, Tensor>;

// Maps a safetensors dtype to the engine dtype; throws UploadError for types the engine has no
// kernels for (bool, 16/32/64-bit unsigned, f64, ...).
mi::DType to_engine_dtype(weights::DType t);

// Lazily opened FileReaders for the files of one ModelIndex. Thread-safe; the index must outlive it.
class ReaderCache {
 public:
  explicit ReaderCache(const ModelIndex& index, bool direct = true) : index_(index), direct_(direct) {}
  const FileReader& get(std::uint32_t file_id);
  const ModelIndex& index() const { return index_; }

 private:
  const ModelIndex& index_;
  bool direct_;
  std::mutex mu_;
  std::map<std::uint32_t, std::unique_ptr<FileReader>> readers_;
};

// Device bytes a group occupies: sum of its tensors rounded up to runtime::kArenaAlign.
std::size_t group_device_bytes(const WeightGroup& group);

// Streams the group through the pinned ring into dst (>= group_device_bytes, kArenaAlign-aligned).
// Asynchronous on the ring's copy stream: record an event there (StreamSet::record) and make the
// compute stream wait on it, or call ring.synchronize(), before using the tensors.
TensorMap upload_group(ReaderCache& readers, const WeightGroup& group, void* dst, std::size_t dst_bytes,
                       runtime::PinnedRing& ring);
// Same, into persistent arena space.
TensorMap upload_group(ReaderCache& readers, const WeightGroup& group, runtime::DeviceArena& arena,
                       runtime::PinnedRing& ring);

// One tensor's place in a group's device layout (see plan_group_upload).
struct UploadPiece {
  const IndexedTensor* tensor;
  std::byte* dst;  // device address of the tensor's first byte
};

// The layout upload_group() uses at dst, without moving any bytes: returns the views and fills `pieces`
// with every tensor's destination in (file, file offset) order. For callers that schedule the copies
// themselves (several readers, a host-resident source). Same destination checks as upload_group().
TensorMap plan_group_upload(const WeightGroup& group, void* dst, std::size_t dst_bytes, std::vector<UploadPiece>& pieces);

// Blocking upload without pinned memory: pageable bounce buffer + cudaMemcpy. On return the bytes are
// resident and visible to work on any stream (it ends with a legacy-stream synchronize; pageable
// cudaMemcpy alone may return before the DMA completes and does not order against non-blocking streams).
TensorMap upload_group_sync(ReaderCache& readers, const WeightGroup& group, void* dst, std::size_t dst_bytes);

enum class HostMemory { kPlain, kPinned };

// A group resident in host memory. The views stay valid as long as any copy of the group lives.
struct HostGroup {
  TensorMap tensors;
  std::shared_ptr<std::byte> storage;
  std::size_t storage_bytes = 0;
};

// Host bytes load_group_to_host() allocates for the group (HostGroup::storage_bytes), exactly.
std::size_t host_group_bytes(const WeightGroup& group);

// Reads every tensor of the group straight into host storage (no intermediate copy). Plain memory is
// page-aligned heap; pinned memory is cudaHostAlloc'ed so later H2D copies from it are async-capable.
HostGroup load_group_to_host(ReaderCache& readers, const WeightGroup& group, HostMemory memory = HostMemory::kPlain);

// Re-reads the group's bytes from disk and compares XXH64 with the index. Returns false when the
// index has no checksum for the group (nothing verified), true on a match; throws ChecksumError on mismatch.
bool verify_group(const ModelIndex& index, const WeightGroup& group);

}  // namespace mi::weights
