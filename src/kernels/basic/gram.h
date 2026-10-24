// H += X^T X for the NVFP4 converter's calibration Hessians (pipeline/nvfp4_calibration.h).
//
// x: CUDA bf16 [m, k] with contiguous rows (row stride k), k % 64 == 0, 16-byte aligned; h: device fp32 [k, k],
// row-major. Only the 64 x 64 tiles on or above the diagonal (tile row <= tile column) are written, the others are left
// untouched: there h[i][j] += sum_r x[r][i] * x[r][j], each element as one fp32 fmaf chain over r ascending from 0
// (from +0), then one fp32 add into h. No atomics, no split over r: the bits depend on neither the GPU nor the launch,
// only on x and the order of the calls.
#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace mi::kernels {

// Bumped whenever gram_accumulate_upper's result bits change.
inline constexpr int kGramNumericsVersion = 1;

void gram_accumulate_upper(const Tensor& x, float* h, cudaStream_t stream);

}  // namespace mi::kernels
