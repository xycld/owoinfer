// Ring of pinned host staging slots that streams file bytes to device memory on a copy stream.
//
// Producer loop per slot: wait until the slot's previous H2D copy finished (its event), fill it from
// disk with FileReader::read_aligned (direct I/O when the reader has it), enqueue cudaMemcpyAsync on
// the copy stream, record the slot's event. With N slots the disk read of slot k+1 overlaps the PCIe
// copy of slot k; the CPU only blocks when all N slots are still in flight (counted in stats().stalls).
//
// Scheduler usage: a prefetch thread calls copy_pieces() for weight group k+1, k+2, then records an
// event on the copy stream (StreamSet::record); the compute stream waits on that event before it
// touches the group. Single producer: calls must not overlap from several threads.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "weights/file_reader.h"

namespace mi::runtime {

inline constexpr std::size_t kDefaultRingSlots = 4;
inline constexpr std::size_t kDefaultRingSlotBytes = std::size_t{64} << 20;  // 256 MiB pinned in total

// Plain snapshot of the ring's counters (see PinnedRing::stats()).
struct RingStats {
  std::uint64_t chunks = 0;        // slot fills (disk reads)
  std::uint64_t stalls = 0;        // fills that had to wait for an in-flight copy of the same slot
  std::uint64_t bytes = 0;         // payload bytes enqueued to the device
};

class PinnedRing {
 public:
  struct Piece {
    std::uint64_t file_offset;
    std::size_t size;
    void* dst;  // device address
  };

  // slot_bytes is rounded up to a multiple of weights::kDirectAlign. The copy stream must outlive the ring.
  PinnedRing(cudaStream_t copy_stream, std::size_t slots = kDefaultRingSlots,
             std::size_t slot_bytes = kDefaultRingSlotBytes);
  ~PinnedRing();
  PinnedRing(const PinnedRing&) = delete;
  PinnedRing& operator=(const PinnedRing&) = delete;

  std::size_t slots() const { return slots_.size(); }
  std::size_t slot_bytes() const { return slot_bytes_; }
  // Thread-safe snapshot (the counters are atomic; the three values are not read as one transaction).
  RingStats stats() const {
    return {stats_.chunks.load(std::memory_order_relaxed), stats_.stalls.load(std::memory_order_relaxed),
            stats_.bytes.load(std::memory_order_relaxed)};
  }

  // Streams one file range to `dst`, split into slot-sized chunks. Asynchronous w.r.t. the device.
  void copy_range(const weights::FileReader& file, std::uint64_t file_offset, std::size_t size, void* dst);

  // Streams several ranges of one file. Pieces must be sorted by file_offset and not overlap; small
  // neighbours are served by one disk read and several device copies out of the same slot.
  void copy_pieces(const weights::FileReader& file, std::span<const Piece> pieces);

  // Stages `size` bytes of host memory through the slots to `dst`: for sources already in pageable RAM,
  // which the device cannot read asynchronously. Counted in stats() like a file read.
  void copy_host(const void* src, std::size_t size, void* dst);

  // Blocks until every enqueued copy finished (slot events only; other work on the stream is not awaited).
  void synchronize();

 private:
  struct Slot {
    std::byte* buf = nullptr;
    cudaEvent_t done = nullptr;
    bool pending = false;
  };
  Slot& next_slot();
  void mark_enqueued(Slot& s);

  cudaStream_t copy_stream_;
  std::size_t slot_bytes_;
  std::byte* host_ = nullptr;
  std::vector<Slot> slots_;
  std::size_t cursor_ = 0;
  struct AtomicStats {
    std::atomic<std::uint64_t> chunks{0}, stalls{0}, bytes{0};
  } stats_;
};

}  // namespace mi::runtime
