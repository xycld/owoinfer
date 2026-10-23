// Test seams of the weights module: internal (not part of any API, not exported from libmossinfer), for tests that
// must reach failure and fallback paths a healthy machine never takes. Every seam is off by default and process-wide.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mi::weights::testing {

// Pinned host allocations (load_group_to_host, read_files_to_host) succeed until `bytes` more have been allocated,
// then each one throws CudaError as cudaHostAlloc's cudaErrorMemoryAllocation would. 0 fails the next one; reset with
// clear_fail_pinned_alloc().
void fail_pinned_alloc_after(std::uint64_t bytes);
void clear_fail_pinned_alloc();
// Called before every pinned allocation of `bytes`: throws when the seam says so.
void check_pinned_alloc(std::size_t bytes);

// page_cache_fraction's query: kAuto (cachestat, else the mincore sample), kMincore (the sample alone, as on kernels
// without cachestat) or kNone (no query: unknown).
enum class ResidencyQuery { kAuto, kMincore, kNone };
void force_residency_query(ResidencyQuery query);
ResidencyQuery residency_query();

}  // namespace mi::weights::testing
