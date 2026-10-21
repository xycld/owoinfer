// Host-RAM accounting (P3 step 24): the process's private memory, i.e. what it costs the machine beyond reclaimable
// page cache. Memory-mapped weight files are file-backed and do not count; RSS would count them.
//
// Linux: /proc/self/status RssAnon + RssShmem + VmSwap. CUDA pinned allocations (cudaHostAlloc / cudaMallocHost) are
// shared mappings of /dev/nvidiactl and show up in RssShmem, not RssAnon (measured on driver 615.71: a 256 MiB
// cudaHostAlloc moves RssShmem by 256 MiB and RssAnon by 0), so the sum covers pinned memory without runtime-side
// bookkeeping; cudaHostRegister'ed malloc memory is already anonymous. RssShmem also holds the driver's own ~90 MiB of
// context mappings. VmSwap adds the private pages swapped out: still the process's, just not resident.
// Windows: PROCESS_MEMORY_COUNTERS_EX::PrivateUsage (commit charge), which includes pinned memory.
//
// The two are not the same quantity. Linux counts touched pages only (resident or swapped): an allocation never
// written costs nothing until it is (overcommit), and the figure rises as pages are first touched. Windows counts
// committed pages whether touched or not: a malloc counts in full at once, and memory committed but never touched
// (e.g. reserve-and-commit arenas of libraries) counts too. So for the same work Windows reads equal or higher, and
// planner headroom derived on one OS is not a measurement of the other.
#pragma once

#include <cstddef>

namespace mi::planner {

struct HostMemory {
  bool valid = false;            // false when the platform query failed (fields are then 0)
  std::size_t private_bytes = 0; // the number reports and AC7/AC8 checks use (Linux: anon + shmem + swap)
  std::size_t anon_bytes = 0;    // Linux RssAnon
  std::size_t shmem_bytes = 0;   // Linux RssShmem (CUDA pinned memory, driver mappings)
  std::size_t swap_bytes = 0;    // Linux VmSwap (private pages swapped out)
  std::size_t rss_bytes = 0;     // resident set incl. file-backed pages (for comparison only)
};

HostMemory process_host_memory();

// Host RAM still available to new allocations without swapping (Linux MemAvailable, Windows ullAvailPhys); 0 if
// unknown.
std::size_t host_available_bytes();

}  // namespace mi::planner
