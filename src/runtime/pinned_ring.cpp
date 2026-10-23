#include "runtime/pinned_ring.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "util/path.h"

namespace mi::runtime {

namespace {
// Tensors closer than this are fetched in one read (the gap is read and discarded).
constexpr std::uint64_t kMaxCoalesceGap = 64 << 10;
}  // namespace

PinnedRing::PinnedRing(cudaStream_t copy_stream, std::size_t slots, std::size_t slot_bytes)
    : copy_stream_(copy_stream),
      slot_bytes_((std::max<std::size_t>(slot_bytes, 1) + weights::kDirectAlign - 1) / weights::kDirectAlign *
                  weights::kDirectAlign) {
  if (slots == 0) throw std::invalid_argument("PinnedRing needs at least one slot");
  // One allocation: cudaHostAlloc is page-aligned and slot_bytes_ is a page multiple, so every slot
  // is a valid direct-I/O target.
  MI_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_), slots * slot_bytes_, cudaHostAllocDefault));
  slots_.resize(slots);
  try {
    for (std::size_t i = 0; i < slots; ++i) {
      slots_[i].buf = host_ + i * slot_bytes_;
      MI_CUDA_CHECK(cudaEventCreateWithFlags(&slots_[i].done, cudaEventDisableTiming));
    }
  } catch (...) {
    for (Slot& s : slots_) {
      if (s.done != nullptr) cudaEventDestroy(s.done);
    }
    cudaFreeHost(host_);
    throw;
  }
}

PinnedRing::~PinnedRing() {
  // In-flight copies read the slots; wait before unpinning. Errors are not actionable here.
  for (Slot& s : slots_) {
    if (s.pending) cudaEventSynchronize(s.done);
    cudaEventDestroy(s.done);
  }
  cudaFreeHost(host_);
}

PinnedRing::Slot& PinnedRing::next_slot() {
  Slot& s = slots_[cursor_];
  cursor_ = (cursor_ + 1) % slots_.size();
  if (s.pending) {
    cudaError_t q = cudaEventQuery(s.done);
    if (q == cudaErrorNotReady) {
      stats_.stalls.fetch_add(1, std::memory_order_relaxed);
      MI_CUDA_CHECK(cudaEventSynchronize(s.done));
    } else {
      MI_CUDA_CHECK(q);
    }
    s.pending = false;
  }
  return s;
}

void PinnedRing::mark_enqueued(Slot& s) {
  MI_CUDA_CHECK(cudaEventRecord(s.done, copy_stream_));
  s.pending = true;
  stats_.chunks.fetch_add(1, std::memory_order_relaxed);
}

void PinnedRing::copy_pinned(const weights::FileReader& file, std::span<const Piece> pieces) {
  for (const Piece& p : pieces) {
    if (p.size == 0) continue;
    if (p.file_offset > file.size() || p.size > file.size() - p.file_offset) {
      throw weights::IoError(util::path_utf8(file.path()) + ": read of " + std::to_string(p.size) + " bytes at offset " +
                             std::to_string(p.file_offset) + " is past end of file");
    }
    MI_CUDA_CHECK(cudaMemcpyAsync(p.dst, file.memory() + p.file_offset, p.size, cudaMemcpyHostToDevice, copy_stream_));
    stats_.bytes.fetch_add(p.size, std::memory_order_relaxed);
  }
  mark_enqueued(next_slot());
}

void PinnedRing::copy_range(const weights::FileReader& file, std::uint64_t file_offset, std::size_t size, void* dst) {
  if (file.pinned()) {
    const Piece piece{file_offset, size, dst};
    copy_pinned(file, std::span<const Piece>(&piece, 1));
    return;
  }
  auto* out = static_cast<std::byte*>(dst);
  while (size > 0) {
    // Chunks after the first end on a kDirectAlign boundary, so they start aligned (skip == 0).
    const std::size_t skip = static_cast<std::size_t>(file_offset % weights::kDirectAlign);
    const std::size_t n = std::min(size, slot_bytes_ - skip);
    Slot& s = next_slot();
    const weights::AlignedRange r = file.read_aligned(file_offset, n, s.buf, slot_bytes_);
    MI_CUDA_CHECK(cudaMemcpyAsync(out, s.buf + r.skip, n, cudaMemcpyHostToDevice, copy_stream_));
    mark_enqueued(s);
    stats_.bytes.fetch_add(n, std::memory_order_relaxed);
    file_offset += n;
    out += n;
    size -= n;
  }
}

void PinnedRing::copy_host(const void* src, std::size_t size, void* dst) {
  const auto* in = static_cast<const std::byte*>(src);
  auto* out = static_cast<std::byte*>(dst);
  while (size > 0) {
    const std::size_t n = std::min(size, slot_bytes_);
    Slot& s = next_slot();
    std::memcpy(s.buf, in, n);
    MI_CUDA_CHECK(cudaMemcpyAsync(out, s.buf, n, cudaMemcpyHostToDevice, copy_stream_));
    mark_enqueued(s);
    stats_.bytes.fetch_add(n, std::memory_order_relaxed);
    in += n;
    out += n;
    size -= n;
  }
}

void PinnedRing::copy_pieces(const weights::FileReader& file, std::span<const Piece> pieces) {
  if (file.pinned()) {
    copy_pinned(file, pieces);
    return;
  }
  std::size_t i = 0;
  while (i < pieces.size()) {
    const Piece& first = pieces[i];
    if (first.size == 0) {
      ++i;
      continue;
    }
    const std::uint64_t begin = first.file_offset;
    std::uint64_t end = begin + first.size;
    if (weights::align_range(begin, first.size).length > slot_bytes_) {
      copy_range(file, begin, first.size, first.dst);
      ++i;
      continue;
    }
    std::size_t j = i + 1;
    for (; j < pieces.size(); ++j) {
      const Piece& q = pieces[j];
      if (q.size == 0) continue;
      if (q.file_offset < end) throw std::invalid_argument("PinnedRing::copy_pieces: pieces overlap or are unsorted");
      if (q.file_offset - end > kMaxCoalesceGap) break;
      const std::uint64_t new_end = q.file_offset + q.size;
      if (weights::align_range(begin, new_end - begin).length > slot_bytes_) break;
      end = new_end;
    }
    Slot& s = next_slot();
    const weights::AlignedRange r = file.read_aligned(begin, end - begin, s.buf, slot_bytes_);
    try {
      for (std::size_t k = i; k < j; ++k) {
        const Piece& p = pieces[k];
        if (p.size == 0) continue;
        MI_CUDA_CHECK(cudaMemcpyAsync(p.dst, s.buf + r.skip + (p.file_offset - begin), p.size, cudaMemcpyHostToDevice,
                                      copy_stream_));
        stats_.bytes.fetch_add(p.size, std::memory_order_relaxed);
      }
    } catch (...) {
      // Earlier copies of this group may be in flight and still read the slot: guard it with its event
      // (or, if even that fails, drain the stream) before the error leaves, so it cannot be refilled early.
      if (cudaEventRecord(s.done, copy_stream_) == cudaSuccess) {
        s.pending = true;
      } else {
        cudaStreamSynchronize(copy_stream_);
      }
      throw;
    }
    mark_enqueued(s);
    i = j;
  }
}

void PinnedRing::synchronize() {
  for (Slot& s : slots_) {
    if (!s.pending) continue;
    MI_CUDA_CHECK(cudaEventSynchronize(s.done));
    s.pending = false;
  }
}

}  // namespace mi::runtime
