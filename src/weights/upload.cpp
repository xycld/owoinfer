#include "weights/upload.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <new>
#include <numeric>
#include <span>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "weights/test_seams.h"

namespace mi::weights {

namespace {

using runtime::align_up;
using runtime::kArenaAlign;

struct Placement {
  const IndexedTensor* t;
  std::size_t offset;  // within the destination block
};

std::vector<Placement> plan_device_layout(const WeightGroup& group, std::size_t* total) {
  std::vector<Placement> out;
  out.reserve(group.tensors.size());
  std::size_t off = 0;
  for (const auto& t : group.tensors) {
    out.push_back({&t, off});
    off += align_up(static_cast<std::size_t>(t.nbytes));
  }
  *total = off;
  return out;
}

Tensor make_view(const IndexedTensor& t, void* data, Device device) {
  const mi::DType dt = to_engine_dtype(t.dtype);
  if (t.shape.size() > static_cast<std::size_t>(Tensor::kMaxRank)) {
    throw UploadError("tensor '" + t.name + "': rank exceeds " + std::to_string(Tensor::kMaxRank));
  }
  Tensor v = Tensor::view(data, dt, device, t.shape.data(), static_cast<int>(t.shape.size()));
  if (v.nbytes() != t.nbytes) throw UploadError("tensor '" + t.name + "': index nbytes disagrees with dtype and shape");
  return v;
}

// Contiguous (file, offset-ordered) runs of the placements, as ring pieces.
std::map<std::uint32_t, std::vector<runtime::PinnedRing::Piece>> pieces_by_file(const std::vector<Placement>& plan,
                                                                                 std::byte* dst) {
  std::map<std::uint32_t, std::vector<runtime::PinnedRing::Piece>> by_file;
  for (const auto& p : plan) {
    by_file[p.t->file].push_back({p.t->offset, static_cast<std::size_t>(p.t->nbytes), dst + p.offset});
  }
  for (auto& [file, v] : by_file) {
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.file_offset < b.file_offset; });
  }
  return by_file;
}

void check_dst(const WeightGroup& group, void* dst, std::size_t dst_bytes, std::size_t need) {
  if (dst_bytes < need) {
    throw UploadError("group '" + group.name + "' needs " + std::to_string(need) + " device bytes, destination has " +
                      std::to_string(dst_bytes));
  }
  if (reinterpret_cast<std::uintptr_t>(dst) % kArenaAlign != 0) throw UploadError("destination is not 256-byte aligned");
}

}  // namespace

mi::DType to_engine_dtype(weights::DType t) {
  switch (t) {
    case weights::DType::kF32: return mi::DType::kF32;
    case weights::DType::kF16: return mi::DType::kF16;
    case weights::DType::kBF16: return mi::DType::kBF16;
    case weights::DType::kF8E4M3: return mi::DType::kF8E4M3;
    case weights::DType::kI8: return mi::DType::kI8;
    case weights::DType::kU8: return mi::DType::kU8;
    case weights::DType::kI32: return mi::DType::kI32;
    case weights::DType::kI64: return mi::DType::kI64;
    default: break;
  }
  throw UploadError(std::string("unsupported weight dtype ") + weights::dtype_name(t));
}

const FileReader& ReaderCache::get(std::uint32_t file_id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = readers_.find(file_id);
  if (it == readers_.end()) {
    const std::filesystem::path path = index_.file_path(file_id);
    std::unique_ptr<FileReader> reader;
    if (host_ != nullptr) {
      if (const auto held = host_->files.find(path); held != host_->files.end()) {
        reader = std::make_unique<FileReader>(path, held->second.data, held->second.size, held->second.pinned);
      }
    }
    if (!reader) reader = std::make_unique<FileReader>(path, direct_);
    it = readers_.emplace(file_id, std::move(reader)).first;
  }
  return *it->second;
}

bool ReaderCache::all_direct() {
  std::lock_guard<std::mutex> lock(mu_);
  if (readers_.empty()) return false;
  return std::all_of(readers_.begin(), readers_.end(), [](const auto& r) { return r.second->direct(); });
}

std::uint64_t HostFiles::bytes() const {
  std::uint64_t total = 0;
  for (const auto& [path, f] : files) total += f.size;
  return total;
}

HostFiles read_files_to_host(const std::vector<std::filesystem::path>& paths, bool pinned, bool direct, int threads,
                             const std::atomic<bool>* cancel) {
  constexpr std::size_t kChunk = std::size_t{64} << 20;  // a multiple of kDirectAlign: every chunk starts aligned
  HostFiles out;
  for (const std::filesystem::path& path : paths) {
    if (out.files.count(path) != 0) continue;
    const FileReader reader(path, direct);
    const std::uint64_t size = reader.size();
    // Rounded up to kDirectAlign: a direct read of the last chunk may fill its whole aligned tail.
    const std::size_t alloc = static_cast<std::size_t>((size + kDirectAlign - 1) / kDirectAlign * kDirectAlign);
    std::byte* raw = nullptr;
    std::shared_ptr<std::byte> storage;
    if (pinned) {
      testing::check_pinned_alloc(alloc);
      MI_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&raw), std::max<std::size_t>(alloc, 1), cudaHostAllocDefault));
      storage.reset(raw, [](std::byte* p) { cudaFreeHost(p); });
    } else {
      raw = static_cast<std::byte*>(::operator new(std::max<std::size_t>(alloc, 1), std::align_val_t{kDirectAlign}));
      storage.reset(raw, [](std::byte* p) { ::operator delete(p, std::align_val_t{kDirectAlign}); });
    }
    const std::uint64_t chunks = (size + kChunk - 1) / kChunk;
    std::atomic<std::uint64_t> next{0};
    std::atomic<bool> stop{false};
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(std::max(1, threads)));
    {
      std::vector<std::jthread> workers;
      for (int t = 0; t < std::max(1, threads); ++t) {
        workers.emplace_back([&, t] {
          try {
            for (std::uint64_t c = next++; c < chunks && !stop.load(); c = next++) {
              if (cancel != nullptr && cancel->load()) {
                stop.store(true);
                return;
              }
              const std::uint64_t offset = c * kChunk;
              const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, size - offset));
              reader.read_aligned(offset, n, raw + offset, alloc - static_cast<std::size_t>(offset));
              testing::after_host_chunk();
            }
          } catch (...) {
            errors[static_cast<std::size_t>(t)] = std::current_exception();
            stop.store(true);
          }
        });
      }
    }
    for (const auto& e : errors) {
      if (e) std::rethrow_exception(e);
    }
    if (cancel != nullptr && cancel->load()) return {};
    out.files.emplace(path, HostFiles::File{std::move(storage), size, pinned});
  }
  return out;
}

std::size_t group_device_bytes(const WeightGroup& group) {
  std::size_t total = 0;
  plan_device_layout(group, &total);
  return total;
}

TensorMap upload_group(ReaderCache& readers, const WeightGroup& group, void* dst, std::size_t dst_bytes,
                       runtime::PinnedRing& ring) {
  std::size_t total = 0;
  const auto plan = plan_device_layout(group, &total);
  check_dst(group, dst, dst_bytes, total);
  TensorMap out;
  for (const auto& p : plan) out.emplace(p.t->name, make_view(*p.t, static_cast<std::byte*>(dst) + p.offset, Device::kCUDA));
  for (const auto& [file, pieces] : pieces_by_file(plan, static_cast<std::byte*>(dst))) {
    ring.copy_pieces(readers.get(file), pieces);
  }
  return out;
}

TensorMap upload_group(ReaderCache& readers, const WeightGroup& group, runtime::DeviceArena& arena,
                       runtime::PinnedRing& ring) {
  const std::size_t bytes = group_device_bytes(group);
  return upload_group(readers, group, arena.alloc_persistent(bytes), bytes, ring);
}

TensorMap plan_group_upload(const WeightGroup& group, void* dst, std::size_t dst_bytes, std::vector<UploadPiece>& pieces) {
  std::size_t total = 0;
  const auto plan = plan_device_layout(group, &total);
  check_dst(group, dst, dst_bytes, total);
  TensorMap out;
  pieces.clear();
  for (const auto& p : plan) {
    auto* d = static_cast<std::byte*>(dst) + p.offset;
    out.emplace(p.t->name, make_view(*p.t, d, Device::kCUDA));
    pieces.push_back({p.t, d});
  }
  std::sort(pieces.begin(), pieces.end(), [](const UploadPiece& a, const UploadPiece& b) {
    return a.tensor->file != b.tensor->file ? a.tensor->file < b.tensor->file : a.tensor->offset < b.tensor->offset;
  });
  return out;
}

TensorMap upload_group_sync(ReaderCache& readers, const WeightGroup& group, void* dst, std::size_t dst_bytes) {
  constexpr std::size_t kBounceBytes = std::size_t{32} << 20;
  std::size_t total = 0;
  const auto plan = plan_device_layout(group, &total);
  check_dst(group, dst, dst_bytes, total);
  AlignedBuffer bounce(kBounceBytes);
  TensorMap out;
  for (const auto& p : plan) {
    auto* d = static_cast<std::byte*>(dst) + p.offset;
    out.emplace(p.t->name, make_view(*p.t, d, Device::kCUDA));
    const FileReader& f = readers.get(p.t->file);
    for (std::uint64_t done = 0; done < p.t->nbytes;) {
      const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(kBounceBytes, p.t->nbytes - done));
      f.read_at(p.t->offset + done, bounce.data(), n);
      MI_CUDA_CHECK(cudaMemcpy(d + done, bounce.data(), n, cudaMemcpyHostToDevice));
      done += n;
    }
  }
  // A pageable cudaMemcpy may return once the data is staged, before the DMA finished, and it is
  // ordered only against the legacy default stream: make the bytes visible to every stream (incl.
  // the non-blocking StreamSet ones) before returning.
  MI_CUDA_CHECK(cudaStreamSynchronize(nullptr));
  return out;
}

std::size_t host_group_bytes(const WeightGroup& group) {
  std::size_t total = 0;
  for (const auto& t : group.tensors) total += align_range(t.offset, static_cast<std::size_t>(t.nbytes)).length;
  return total;
}

HostGroup load_group_to_host(ReaderCache& readers, const WeightGroup& group, HostMemory memory) {
  // Each tensor gets a kDirectAlign-aligned slab and lands at the slab's `skip` offset (its file
  // offset modulo kDirectAlign), so read_aligned can DMA the widened range directly into place.
  struct Slab {
    const IndexedTensor* t;
    std::size_t offset, skip, length;
  };
  std::vector<Slab> slabs;
  std::size_t total = 0;
  for (const auto& t : group.tensors) {
    const AlignedRange r = align_range(t.offset, static_cast<std::size_t>(t.nbytes));
    slabs.push_back({&t, total, r.skip, r.length});
    total += r.length;
  }

  std::byte* raw = nullptr;
  std::shared_ptr<std::byte> storage;
  if (memory == HostMemory::kPinned) {
    testing::check_pinned_alloc(total);
    MI_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&raw), std::max<std::size_t>(total, 1), cudaHostAllocDefault));
    storage.reset(raw, [](std::byte* p) { cudaFreeHost(p); });
  } else {
    raw = static_cast<std::byte*>(::operator new(std::max<std::size_t>(total, 1), std::align_val_t{kDirectAlign}));
    storage.reset(raw, [](std::byte* p) { ::operator delete(p, std::align_val_t{kDirectAlign}); });
  }

  HostGroup out;
  for (const auto& s : slabs) {
    std::byte* data = raw + s.offset + s.skip;
    out.tensors.emplace(s.t->name, make_view(*s.t, data, Device::kCPU));
    if (s.t->nbytes == 0) continue;
    readers.get(s.t->file).read_aligned(s.t->offset, static_cast<std::size_t>(s.t->nbytes), raw + s.offset, s.length);
  }
  out.storage = std::move(storage);
  out.storage_bytes = total;
  return out;
}

bool verify_group(const ModelIndex& index, const WeightGroup& group) {
  if (!group.checksum) return false;
  const std::uint64_t got = checksum_group(index, group);
  if (got != *group.checksum) {
    throw ChecksumError("group '" + group.name + "': XXH64 mismatch (index " + std::to_string(*group.checksum) +
                        ", disk " + std::to_string(got) + ")");
  }
  return true;
}

}  // namespace mi::weights
