#include "core/cuda_check.h"

namespace mi {

void throw_cuda_error(cudaError_t err, const char* expr, const char* file, int line) {
  throw CudaError(std::string("CUDA error ") + cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ") in " +
                  expr + " at " + file + ":" + std::to_string(line));
}

DeviceBuffer::DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes > 0) MI_CUDA_CHECK(cudaMalloc(&ptr_, bytes));
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr_ != nullptr) cudaFree(ptr_);  // destructor: errors are not actionable here
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
  if (this != &o) {
    if (ptr_ != nullptr) cudaFree(ptr_);
    ptr_ = std::exchange(o.ptr_, nullptr);
    bytes_ = std::exchange(o.bytes_, 0);
  }
  return *this;
}

void DeviceBuffer::upload(const void* host, std::size_t bytes, std::size_t offset) {
  if (offset > bytes_ || bytes > bytes_ - offset) throw std::out_of_range("DeviceBuffer::upload out of range");
  MI_CUDA_CHECK(cudaMemcpy(static_cast<char*>(ptr_) + offset, host, bytes, cudaMemcpyHostToDevice));
  // A pageable-memory cudaMemcpy may return once the data is staged, before the DMA lands; work on
  // non-blocking streams (the engine's streams) does not order after it. Make the data visible.
  MI_CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void DeviceBuffer::download(void* host, std::size_t bytes, std::size_t offset) const {
  if (offset > bytes_ || bytes > bytes_ - offset) throw std::out_of_range("DeviceBuffer::download out of range");
  MI_CUDA_CHECK(cudaMemcpy(host, static_cast<const char*>(ptr_) + offset, bytes, cudaMemcpyDeviceToHost));
}

}  // namespace mi
