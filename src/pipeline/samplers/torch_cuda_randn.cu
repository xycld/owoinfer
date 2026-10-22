#include "pipeline/samplers/torch_cuda_randn.h"

#include <curand_kernel.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace mi::pipeline {
namespace {

constexpr unsigned kBlock = 256;  // torch's block_size_bound
constexpr int kUnroll = 4;        // curand_normal4 values per draw

__global__ void randn_kernel(float* out, std::int64_t n, std::uint64_t seed, std::uint64_t offset) {
  const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, static_cast<unsigned long long>(idx), offset, &state);
  const std::int64_t stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
  const std::int64_t rounded = ((n - 1) / (stride * kUnroll) + 1) * stride * kUnroll;
  for (std::int64_t li = idx; li < rounded; li += stride * kUnroll) {
    const float4 r = curand_normal4(&state);
    const float v[kUnroll] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int ii = 0; ii < kUnroll; ++ii) {
      const std::int64_t j = li + stride * ii;
      if (j < n) out[j] = v[ii];
    }
  }
}

void check_out(const Tensor& out) {
  if (out.device != Device::kCUDA || out.dtype != DType::kF32 || !out.is_contiguous()) {
    throw std::invalid_argument("torch_cuda_randn: expected contiguous CUDA fp32, got " + out.describe());
  }
}

}  // namespace

unsigned cuda_randn_blocks(std::int64_t n, CudaRandnGrid grid) {
  if (n < 1) throw std::invalid_argument("cuda_randn_blocks: n must be >= 1");
  const std::int64_t full = (n + kBlock - 1) / kBlock;
  if (full > 0x7fffffff) throw std::invalid_argument("cuda_randn_blocks: n too large");
  std::int64_t blocks = full;
  if (grid == CudaRandnGrid::kDevice) {
    int device = 0, sms = 0, threads = 0;
    MI_CUDA_CHECK(cudaGetDevice(&device));
    MI_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    MI_CUDA_CHECK(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    blocks = std::min<std::int64_t>(blocks, static_cast<std::int64_t>(sms) * (threads / static_cast<int>(kBlock)));
  }
  return static_cast<unsigned>(blocks);
}

std::uint64_t cuda_randn_offset_increment(std::int64_t n, unsigned blocks) {
  const auto per_call = static_cast<std::uint64_t>(kBlock) * blocks * kUnroll;
  return ((static_cast<std::uint64_t>(n) - 1) / per_call + 1) * kUnroll;
}

void torch_cuda_randn(const Tensor& out, std::uint64_t seed, std::uint64_t offset, unsigned blocks, cudaStream_t stream) {
  check_out(out);
  const std::int64_t n = out.numel();
  if (n == 0) return;
  if (blocks == 0) throw std::invalid_argument("torch_cuda_randn: blocks must be >= 1");
  randn_kernel<<<blocks, kBlock, 0, stream>>>(out.ptr<float>(), n, seed, offset);
  MI_CUDA_CHECK_LAUNCH();
}

void TorchCudaGenerator::draw(const Tensor& out, cudaStream_t stream) {
  check_out(out);
  const std::int64_t n = out.numel();
  if (n == 0) return;
  const unsigned blocks = cuda_randn_blocks(n, grid_);
  torch_cuda_randn(out, seed_, offset_, blocks, stream);
  offset_ += cuda_randn_offset_increment(n, blocks);
}

}  // namespace mi::pipeline
