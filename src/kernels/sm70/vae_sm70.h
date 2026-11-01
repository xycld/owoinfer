// Volta VAE: fp32 SIMT convolutions and bounded fp32 attention slabs; no hidden device allocation.
#pragma once

#include <cublasLt.h>

#include "core/tensor.h"

namespace mi::kernels::sm70 {

// Arguments are validated by vae::conv2d. The split-K=1 implicit GEMM needs no workspace.
void vae_conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
                cudaStream_t stream);

std::size_t vae_attention_workspace_bytes(std::int64_t length);
void vae_attention_fp32(cublasLtHandle_t lt, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                        float scale, void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace mi::kernels::sm70
