// Bit-exact reimplementation of torch's CUDA-generator randn (torch.randn(n, device="cuda", generator=g) with
// g = torch.Generator("cuda").manual_seed(seed)), the per-step noise source of the k-diffusion samplers when they run
// on a GPU.
//
// torch's scheme (aten DistributionTemplates.h, distribution_nullary_kernel + normal_kernel, float output):
//  - grid of 256-thread blocks: min(ceil(n / 256), SMs * (max threads per SM / 256)); stride = 256 * blocks
//  - thread `idx` runs curand_init(seed, subsequence = idx, offset, Philox4x32-10) and, for each
//    li = idx + k * 4 * stride (k = 0, 1, ...), draws curand_normal4() and writes its four values to
//    li, li + stride, li + 2 stride, li + 3 stride (those < n)
//  - the generator's offset starts at 0 after manual_seed and advances by 4 * ceil(n / (4 * stride)) per call
// Philox and curand_normal4's Box-Muller come from the CUDA toolkit's curand_kernel.h (header only), as in torch.
//
// The values depend on the grid, so on the GPU's SM count once n > 256 * that cap (n > 116736 on a 76-SM GPU with
// 1536 threads per SM). CudaRandnGrid::kUncapped uses ceil(n / 256) blocks, which is GPU independent (and what torch
// produces on any GPU large enough to hold the grid); kDevice reproduces torch on the current GPU (and on every GPU
// with the same SM count and threads per SM).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "core/tensor.h"

namespace mi::pipeline {

enum class CudaRandnGrid { kUncapped, kDevice };

// Number of 256-thread blocks torch launches for n values under `grid` on the current device. n >= 1.
unsigned cuda_randn_blocks(std::int64_t n, CudaRandnGrid grid);
// Philox offset advance of one randn call of n values with `blocks` blocks: 4 * ceil(n / (4 * 256 * blocks)).
std::uint64_t cuda_randn_offset_increment(std::int64_t n, unsigned blocks);

// A torch CUDA generator: seed plus Philox offset. Every draw() fills `out` (contiguous CUDA fp32) as
// torch.randn(out.numel(), generator=g) would, then advances the offset.
class TorchCudaGenerator {
 public:
  explicit TorchCudaGenerator(std::uint64_t seed, CudaRandnGrid grid = CudaRandnGrid::kUncapped)
      : seed_(seed), grid_(grid) {}
  void draw(const Tensor& out, cudaStream_t stream);
  std::uint64_t seed() const { return seed_; }
  std::uint64_t offset() const { return offset_; }

 private:
  std::uint64_t seed_;
  std::uint64_t offset_ = 0;
  CudaRandnGrid grid_;
};

// One randn draw with an explicit Philox offset and block count (exposed for tests).
void torch_cuda_randn(const Tensor& out, std::uint64_t seed, std::uint64_t offset, unsigned blocks, cudaStream_t stream);

}  // namespace mi::pipeline
