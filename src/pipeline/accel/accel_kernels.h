// Internal to the step cache (accel.h): elementwise bf16 kernels and a deterministic reduction. Pointers are device
// bf16 arrays of n elements; each op computes in fp32 and rounds to bf16 once (round to nearest even), as torch's bf16
// elementwise ops do.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace mi::pipeline::accel_kernels {

// Blocks of residual_diff_partials: a fixed grid, so the summation order (and the result) never changes.
inline constexpr int kDiffBlocks = 128;

// out = bf16(a - b)
void sub_bf16(const void* a, const void* b, void* out, std::int64_t n, cudaStream_t stream);
// x = bf16(x + r)
void add_bf16(void* x, const void* r, std::int64_t n, cudaStream_t stream);
// With r = bf16(x1 - x0): partials[2 k] = sum |r - ref|, partials[2 k + 1] = sum |ref| over block k's share of the
// elements (kDiffBlocks blocks, fixed order, accumulated in fp64).
void residual_diff_partials(const void* x1, const void* x0, const void* ref, std::int64_t n, double* partials,
                            cudaStream_t stream);

}  // namespace mi::pipeline::accel_kernels
