#include "runtime/device_arena.h"

#include <cuda_runtime.h>

#include <string>
#include <utility>

#include "core/cuda_check.h"

namespace mi::runtime {

namespace {
std::string mib(std::size_t b) { return std::to_string(b >> 20) + " MiB (" + std::to_string(b) + " B)"; }
}  // namespace

BudgetError::BudgetError(std::size_t requested_bytes, std::size_t free_bytes)
    : std::runtime_error("device arena of " + mib(requested_bytes) + " exceeds free device memory " + mib(free_bytes)),
      requested(requested_bytes),
      free(free_bytes) {}

ArenaOverflow::ArenaOverflow(std::size_t requested_bytes, std::size_t available_bytes)
    : std::runtime_error("device arena overflow: need " + mib(requested_bytes) + ", have " + mib(available_bytes)),
      requested(requested_bytes),
      available(available_bytes) {}

std::size_t DeviceArena::free_device_bytes() {
  std::size_t free_b = 0, total_b = 0;
  MI_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  return free_b;
}

DeviceArena::DeviceArena(std::size_t bytes) {
  const std::size_t cap = bytes / kArenaAlign * kArenaAlign;
  const std::size_t free_b = free_device_bytes();
  if (cap > free_b) throw BudgetError(cap, free_b);
  if (cap > 0) {
    // free_b is a snapshot; another process can win the race. Report that as a budget problem too.
    const cudaError_t err = cudaMalloc(&base_, cap);
    if (err == cudaErrorMemoryAllocation) {
      cudaGetLastError();  // clear the recorded error so later checks do not trip over it
      base_ = nullptr;
      throw BudgetError(cap, free_device_bytes());
    }
    MI_CUDA_CHECK(err);
  }
  capacity_ = cap;
}

DeviceArena::~DeviceArena() { release(); }

void DeviceArena::release() noexcept {
  if (base_ != nullptr) cudaFree(base_);  // destructor: errors are not actionable here
  base_ = nullptr;
}

DeviceArena::DeviceArena(DeviceArena&& o) noexcept
    : base_(std::exchange(o.base_, nullptr)),
      capacity_(std::exchange(o.capacity_, 0)),
      persistent_(std::exchange(o.persistent_, 0)),
      stack_(std::exchange(o.stack_, 0)),
      high_water_(std::exchange(o.high_water_, 0)) {}

DeviceArena& DeviceArena::operator=(DeviceArena&& o) noexcept {
  if (this != &o) {
    release();
    base_ = std::exchange(o.base_, nullptr);
    capacity_ = std::exchange(o.capacity_, 0);
    persistent_ = std::exchange(o.persistent_, 0);
    stack_ = std::exchange(o.stack_, 0);
    high_water_ = std::exchange(o.high_water_, 0);
  }
  return *this;
}

void* DeviceArena::grow(std::size_t bytes, bool from_stack) {
  const std::size_t need = align_up(bytes);
  if (need < bytes || need > available()) throw ArenaOverflow(bytes, available());
  auto* b = static_cast<std::byte*>(base_);
  void* p;
  if (from_stack) {
    stack_ += need;
    p = b + (capacity_ - stack_);
  } else {
    p = b + persistent_;
    persistent_ += need;
  }
  if (used() > high_water_) high_water_ = used();
  return p;
}

void* DeviceArena::alloc_persistent(std::size_t bytes) { return grow(bytes, false); }
void* DeviceArena::alloc_stack(std::size_t bytes) { return grow(bytes, true); }

Tensor DeviceArena::alloc_persistent_tensor(DType dtype, std::initializer_list<std::int64_t> shape) {
  Tensor t = Tensor::view(nullptr, dtype, Device::kCUDA, shape);
  t.data = alloc_persistent(t.nbytes());
  return t;
}

Tensor DeviceArena::alloc_tensor(DType dtype, std::initializer_list<std::int64_t> shape) {
  Tensor t = Tensor::view(nullptr, dtype, Device::kCUDA, shape);
  t.data = alloc_stack(t.nbytes());
  return t;
}

void DeviceArena::rollback(std::size_t mark) {
  if (mark > stack_) throw std::invalid_argument("DeviceArena::rollback: mark is above the current stack top");
  stack_ = mark;
}

}  // namespace mi::runtime
