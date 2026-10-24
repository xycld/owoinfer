#include "weights/test_seams.h"

#include <cuda_runtime.h>

#include <atomic>
#include <limits>

#include "core/cuda_check.h"

namespace mi::weights::testing {
namespace {

constexpr std::uint64_t kOff = std::numeric_limits<std::uint64_t>::max();
std::atomic<std::uint64_t> g_pinned_left{kOff};  // bytes still allowed before the seam fails pinned allocations
std::atomic<std::atomic<bool>*> g_chunk_cancel{nullptr};
std::atomic<ResidencyQuery> g_query{ResidencyQuery::kAuto};
std::atomic<bool> g_any_model_shapes{false};

}  // namespace

void fail_pinned_alloc_after(std::uint64_t bytes) { g_pinned_left.store(bytes == kOff ? kOff - 1 : bytes); }

void clear_fail_pinned_alloc() { g_pinned_left.store(kOff); }

void cancel_after_first_host_chunk(std::atomic<bool>* flag) { g_chunk_cancel.store(flag); }

void after_host_chunk() {
  if (g_chunk_cancel.load(std::memory_order_relaxed) == nullptr) return;  // the production path: one relaxed load
  if (std::atomic<bool>* flag = g_chunk_cancel.exchange(nullptr)) flag->store(true);
}

void check_pinned_alloc(std::size_t bytes) {
  std::uint64_t left = g_pinned_left.load(std::memory_order_relaxed);
  if (left == kOff) return;  // the production path: one relaxed load
  while (true) {
    if (left == kOff) return;
    if (bytes > left) {
      g_pinned_left.store(0);
      throw_cuda_error(cudaErrorMemoryAllocation, "cudaHostAlloc (test seam)", __FILE__, __LINE__);
    }
    if (g_pinned_left.compare_exchange_weak(left, left - bytes)) return;
  }
}

void force_residency_query(ResidencyQuery query) { g_query.store(query); }
ResidencyQuery residency_query() { return g_query.load(); }

void allow_any_model_shapes(bool allow) { g_any_model_shapes.store(allow); }
bool any_model_shapes() { return g_any_model_shapes.load(std::memory_order_relaxed); }

}  // namespace mi::weights::testing
