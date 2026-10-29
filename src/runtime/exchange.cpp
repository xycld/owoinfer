#include "runtime/exchange.h"

#include <cstdint>
#include <optional>

#include "backend/cuda_driver.h"
#include "core/cuda_check.h"
#include "runtime/device_arena.h"

namespace mi::runtime {
namespace {

std::string mib(std::size_t bytes) {
  return std::to_string((bytes + (std::size_t{1} << 20) - 1) >> 20) + " MiB";
}

[[noreturn]] void refuse(std::string_view what, int device, const std::string& why) {
  throw std::invalid_argument(std::string(what) + " " + why + "; device tensors must be device memory of device " +
                              std::to_string(device) + " in its primary context (as torch and CuPy allocate)");
}

}  // namespace

DevicePointerCheck::DevicePointerCheck(int device) : device_(device) {
  if (const std::optional<void*> ctx = backend::primary_context(device)) primary_ = *ctx;
}

void DevicePointerCheck::check(const void* ptr, std::size_t bytes, std::string_view what) const {
  if (ptr == nullptr) refuse(what, device_, "has no data");
  const std::optional<backend::PointerInfo> info = backend::pointer_info(ptr);
  if (!info) {
    // No cuPointerGetAttributes in this driver library: the runtime's view (type and device only).
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, ptr) != cudaSuccess) {
      cudaGetLastError();
      refuse(what, device_, "is not CUDA memory");
    }
    if (a.type != cudaMemoryTypeDevice) refuse(what, device_, a.type == cudaMemoryTypeManaged ? "is managed memory" : "is not device memory");
    if (a.device != device_) refuse(what, device_, "is memory of device " + std::to_string(a.device));
    return;
  }
  switch (info->memory_type) {
    case backend::PointerInfo::kUnknown: refuse(what, device_, "is not CUDA memory (pageable host memory?)");
    case backend::PointerInfo::kHost: refuse(what, device_, "is page-locked host memory");
    case backend::PointerInfo::kOther: refuse(what, device_, "is not device memory");
    case backend::PointerInfo::kDevice: break;
  }
  if (info->managed) refuse(what, device_, "is managed (unified) memory");
  if (info->device != device_) refuse(what, device_, "is memory of device " + std::to_string(info->device));
  if (info->context != nullptr && primary_ != nullptr && info->context != primary_) {
    refuse(what, device_, "belongs to a CUDA context other than the device's primary context");
  }
  const auto p = reinterpret_cast<std::uintptr_t>(ptr);
  if (info->range_size > 0 && (p < info->range_start || bytes > info->range_size - (p - info->range_start))) {
    refuse(what, device_, "needs " + std::to_string(bytes) + " bytes, but its allocation has " +
                              std::to_string(info->range_start + info->range_size - p) + " from that address");
  }
}

StreamHandoff::StreamHandoff() {
  MI_CUDA_CHECK(cudaEventCreateWithFlags(&in_, cudaEventDisableTiming));
  const cudaError_t e = cudaEventCreateWithFlags(&out_, cudaEventDisableTiming);
  if (e != cudaSuccess) {
    cudaEventDestroy(in_);
    MI_CUDA_CHECK(e);
  }
}

StreamHandoff::~StreamHandoff() {
  cudaEventDestroy(in_);
  cudaEventDestroy(out_);
}

void StreamHandoff::acquire(cudaStream_t host, cudaStream_t engine) {
  MI_CUDA_CHECK(cudaEventRecord(in_, host));
  MI_CUDA_CHECK(cudaStreamWaitEvent(engine, in_, 0));
}

void StreamHandoff::release(cudaStream_t engine, cudaStream_t host) {
  MI_CUDA_CHECK(cudaEventRecord(out_, engine));
  MI_CUDA_CHECK(cudaStreamWaitEvent(host, out_, 0));
  MI_CUDA_CHECK(cudaEventSynchronize(out_));
}

StagingLayout StagingLayout::plan(std::size_t x_bytes, int outputs, std::size_t out_bytes, std::size_t control_bytes) {
  if (x_bytes == 0 || out_bytes == 0 || outputs < 1 || outputs > static_cast<int>(kMaxStagingOutputs)) {
    throw std::invalid_argument("staging layout: x and 1.." + std::to_string(kMaxStagingOutputs) +
                                " outputs of nonzero size, got " + std::to_string(outputs) + " outputs");
  }
  StagingLayout l;
  l.x_bytes = x_bytes;
  l.out_bytes = out_bytes;
  l.outputs = outputs;
  l.control_bytes = control_bytes;
  std::size_t at = align_up(x_bytes);
  for (int k = 0; k < outputs; ++k) {
    l.out[static_cast<std::size_t>(k)] = at;
    at += align_up(out_bytes);
  }
  l.control = control_bytes > 0 ? at : 0;
  l.bytes = at + align_up(control_bytes);
  return l;
}

PinnedStaging::~PinnedStaging() { release(); }

void PinnedStaging::ensure(std::size_t bytes, std::size_t budget) {
  if (bytes <= block_.bytes) return;
  if (bytes > budget) {
    throw StagingError("pinned staging of " + mib(bytes) + " exceeds the host memory budget left for it (" + mib(budget) +
                           ")",
                       bytes);
  }
  release();
  void* p = nullptr;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) {
    cudaGetLastError();
    throw StagingError("pinned staging of " + mib(bytes) + ": page-locked host allocation failed", bytes);
  }
  block_ = {static_cast<std::byte*>(p), bytes, true};
}

void PinnedStaging::release() {
  // Callers synchronize their copies before a run ends (every host-mode step does); errors are not actionable here.
  if (block_.owned && block_.base != nullptr) cudaFreeHost(block_.base);
  block_ = {};
}

}  // namespace mi::runtime
