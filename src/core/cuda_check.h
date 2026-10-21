// CUDA / cuBLAS error checking and a minimal owning device buffer (tests and simple tools;
// the engine proper allocates from runtime/device_arena).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace mi {

struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] void throw_cuda_error(cudaError_t err, const char* expr, const char* file, int line);

#define MI_CUDA_CHECK(expr)                                                  \
  do {                                                                       \
    cudaError_t mi_cuda_err_ = (expr);                                       \
    if (mi_cuda_err_ != cudaSuccess) {                                       \
      ::mi::throw_cuda_error(mi_cuda_err_, #expr, __FILE__, __LINE__);       \
    }                                                                        \
  } while (0)

// Checks the most recent kernel launch.
#define MI_CUDA_CHECK_LAUNCH() MI_CUDA_CHECK(cudaGetLastError())

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  std::size_t size() const { return bytes_; }
  // Synchronous: the data is visible to every stream (including non-blocking ones) on return.
  void upload(const void* host, std::size_t bytes, std::size_t offset = 0);
  void download(void* host, std::size_t bytes, std::size_t offset = 0) const;

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace mi
