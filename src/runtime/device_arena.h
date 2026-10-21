// One up-front device reservation, sub-allocated without further cudaMalloc calls.
//
// Layout: [ persistent --> ........ free ........ <-- stack ]. Resident weights go to the persistent
// region (bump from the low end, never freed individually); per-step activations go to the stack
// region (bump from the high end) and are released in bulk via mark()/rollback(). Two opposite-end
// regions share one budget, so neither needs its size known in advance, and an overflow is exactly
// "the two cursors met". All blocks are kArenaAlign-aligned.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

#include "core/tensor.h"

namespace mi::runtime {

inline constexpr std::size_t kArenaAlign = 256;

constexpr std::size_t align_up(std::size_t n, std::size_t a = kArenaAlign) { return (n + a - 1) / a * a; }

// The reservation does not fit in the currently free device memory (other processes hold some).
struct BudgetError : std::runtime_error {
  BudgetError(std::size_t requested_bytes, std::size_t free_bytes);
  std::size_t requested;
  std::size_t free;
};

// A sub-allocation does not fit in the arena's remaining space.
struct ArenaOverflow : std::runtime_error {
  ArenaOverflow(std::size_t requested_bytes, std::size_t available_bytes);
  std::size_t requested;
  std::size_t available;
};

class DeviceArena {
 public:
  // Reserves `bytes` (rounded down to kArenaAlign). Throws BudgetError if cudaMemGetInfo reports
  // fewer free bytes than that, before touching the allocator.
  explicit DeviceArena(std::size_t bytes);
  ~DeviceArena();
  DeviceArena(DeviceArena&& o) noexcept;
  DeviceArena& operator=(DeviceArena&& o) noexcept;
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  static std::size_t free_device_bytes();

  void* base() const { return base_; }
  std::size_t capacity() const { return capacity_; }
  std::size_t persistent_used() const { return persistent_; }
  std::size_t stack_used() const { return stack_; }
  std::size_t used() const { return persistent_ + stack_; }
  std::size_t available() const { return capacity_ - used(); }
  // Peak of used() since construction (or reset_high_water()).
  std::size_t high_water() const { return high_water_; }
  void reset_high_water() { high_water_ = used(); }

  // Both throw ArenaOverflow when the request does not fit. Size 0 is allowed and consumes nothing.
  void* alloc_persistent(std::size_t bytes);
  void* alloc_stack(std::size_t bytes);

  Tensor alloc_persistent_tensor(DType dtype, std::initializer_list<std::int64_t> shape);
  // Stack-region tensor (activations).
  Tensor alloc_tensor(DType dtype, std::initializer_list<std::int64_t> shape);

  // Stack mark: pass the value back to rollback() to free everything allocated since.
  // Single-stream only: rollback() makes the space reusable immediately on the host, without waiting
  // for (or recording an event after) work still queued on any stream. It is safe only when all work
  // touching the rolled-back blocks runs on one stream, in order with the later reuse; with several
  // streams, synchronize or event-wait them before rolling back.
  std::size_t mark() const { return stack_; }
  void rollback(std::size_t mark);

 private:
  void* grow(std::size_t bytes, bool from_stack);
  void release() noexcept;

  void* base_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t persistent_ = 0;
  std::size_t stack_ = 0;
  std::size_t high_water_ = 0;
};

// Rolls the stack region back on scope exit. Same single-stream restriction as DeviceArena::rollback().
class StackScope {
 public:
  explicit StackScope(DeviceArena& arena) : arena_(arena), mark_(arena.mark()) {}
  ~StackScope() { arena_.rollback(mark_); }
  StackScope(const StackScope&) = delete;
  StackScope& operator=(const StackScope&) = delete;

 private:
  DeviceArena& arena_;
  std::size_t mark_;
};

}  // namespace mi::runtime
