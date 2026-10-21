// GroupNorm bit-identical to torch 2.14's CUDA kernel (aten/src/ATen/native/cuda/group_norm_kernel.cu) on
// NCHW-contiguous fp32 input, for NHWC activations. The decoder's output bytes are sensitive to last-bit
// differences (they grow ~100x through the up blocks), so the reference mode reproduces torch exactly:
//   statistics: RowwiseMomentsCUDAKernel, one Welford chain per (n, group, thread t) over the group's NCHW
//     elements t, t + T, ... (T = 512 threads, 32 when the group has fewer than 512 elements), then torch's
//     warp/block Welford combine tree, var = m2 / count, rstd = rsqrtf(var + eps).
//   affine: ComputeFusedParamsCUDAKernel's a = rstd * gamma and b = -a * mean + beta, both rounded per op
//     (torch's code does not contract b into an FMA), then y = fma(a, x, b) and SiLU as a separate op.
// The apply step is vae::group_norm_apply fed with stats {0, rstd} and beta' = b per channel, which makes its
// fma(-a, 0, beta') return b exactly. fp32 only, batch 1 (beta' is per channel).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "core/tensor.h"

namespace mi::flux2_vae {

std::size_t torch_group_norm_workspace_bytes(const Tensor& x, int groups);
// x, y: contiguous CUDA fp32 [1, H, W, C] (may alias); gamma, beta: fp32 [C].
void torch_group_norm(const Tensor& x, int groups, float eps, const Tensor& gamma, const Tensor& beta, bool silu,
                      const Tensor& y, void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace mi::flux2_vae
