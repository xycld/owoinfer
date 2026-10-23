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
#include <filesystem>
#include <string_view>

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
// unknown. Linux: capped by the process's memory cgroup (cgroup_memory()), so a container's limit counts rather than
// the host's free memory.
std::size_t host_available_bytes();

// The process's memory-cgroup limit and headroom (Linux; never limited elsewhere). Pure apart from reading the cgroup
// files under `root` ("/" for the machine, a fixture directory in tests): `self_cgroup` is /proc/self/cgroup's text,
// `mountinfo` /proc/self/mountinfo's. The cgroup directory is resolved through the mount that holds it (its root and
// mount point: a container, or a namespace with the host's /sys/fs/cgroup bind-mounted, sees its own cgroup at the
// mount point); a path that does not exist falls back to the mount point. cgroup v2 first: at the cgroup and every
// ancestor up to the mount point, limit = min(memory.max, memory.high) and used = memory.current - inactive_file
// (memory.stat; reclaimable page cache is not "used": kubelet's working-set rule); the tightest headroom (limit - used)
// wins. No v2 limit (a hybrid host): cgroup v1's memory controller, with memory.limit_in_bytes (2^62 and above: none),
// memory.usage_in_bytes and total_inactive_file. A limited cgroup's headroom is at least 1, so 0 never reads as
// "unknown"; missing or malformed files count as no limit.
struct CgroupMemory {
  bool limited = false;
  std::size_t limit = 0;
  std::size_t headroom = 0;
};
CgroupMemory cgroup_memory(std::string_view self_cgroup, std::string_view mountinfo, const std::filesystem::path& root);
// Host RAM of the machine (Linux MemTotal, capped by the memory cgroup's limit as above; Windows ullTotalPhys); 0 if
// unknown.
std::size_t host_total_bytes();

// Host memory a large pinned or kept copy the engine holds on its own initiative (a pinned text-encoder copy, a soft
// unload's DiT copy) must leave available: max(8 GiB, 25 % of the machine's RAM). Page-locked memory cannot be swapped
// or reclaimed, and the OS, the page cache and other applications need room around it.
inline constexpr std::size_t kHostPinMarginFloor = std::size_t{8} << 30;
inline std::size_t host_pin_margin(std::size_t host_total) {
  return host_total / 4 > kHostPinMarginFloor ? host_total / 4 : kHostPinMarginFloor;
}
// Whether holding `needed` bytes of host memory still leaves host_pin_margin(host_total) of `room` (what the system can
// still give the process; 0 = unknown: never fits).
inline bool fits_pin_margin(std::size_t needed, std::size_t room, std::size_t host_total) {
  return room != 0 && needed <= room && room - needed >= host_pin_margin(host_total);
}

}  // namespace mi::planner
