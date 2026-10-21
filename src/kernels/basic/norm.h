// RMSNorm variants of the MOSS DiT and the Qwen3.5 text encoder, bit-exact against torch eager.
//
// x, gate and out are CUDA bf16 views [..., D] with a contiguous last dim; leading dims may have any
// strides (e.g. per-head views [L, heads, head_dim] of a fused projection), out may alias x. Weights,
// scale and shift are contiguous bf16 [D]. Supported D: multiples of 4, >= 128, and small enough
// that torch does not split a row across warps, which is not reproduced. The limit depends on the row
// count R (torch's block shape): with bw threads per row and bh rows per block (bh = min(pow2floor(R), 16),
// bw = min(pow2floor(D/4), 512 / bh)), D must satisfy ceil(D / bw) < min(16 * bh, 256); otherwise the op
// throws. Examples: R >= 16 -> D <= 8160 (8188 fails); R = 1 -> D <= 7680. D <= 4096 is always fine.
//
// Exactness: the fp32 mean of squares is summed in the order of torch's reduce kernel for the same
// row count (block shape, 4-way vector accumulators, shared-memory and warp-shuffle trees), so every
// variant matches `x.float().square().mean(-1)` and the following ops bit for bit.
#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace mi::kernels {

inline constexpr float kRmsNormEps = 1e-6f;

// MOSS _RMSNorm (P:56-63): out = bf16(bf16(x * rsqrt(mean(x^2) + 1e-6)) * w). Rounds twice.
void rms_norm_moss(const Tensor& x, const Tensor& weight, const Tensor& out, cudaStream_t stream);

// MOSS pre-attention/FFN/head modulation (P:137, P:144, P:258): out = rms_norm_moss(x) * (1 + scale) + shift
// with each op rounded to bf16 as torch does. scale and shift are one sample's [D] vectors.
void rms_norm_modulate_moss(const Tensor& x, const Tensor& weight, const Tensor& scale, const Tensor& shift,
                            const Tensor& out, cudaStream_t stream);

// Qwen3_5RMSNorm (Q:827-841): out = bf16(x * rsqrt(mean(x^2) + eps) * (1 + float(w))), all in fp32.
void rms_norm_qwen(const Tensor& x, const Tensor& weight, const Tensor& out, cudaStream_t stream,
                   float eps = kRmsNormEps);

// Qwen3_5RMSNormGated (Q:204-220), per head: out = bf16(float(bf16(w * bf16(x * rsqrt(...)))) * silu(float(gate))).
void rms_norm_gated_qwen(const Tensor& x, const Tensor& gate, const Tensor& weight, const Tensor& out,
                         cudaStream_t stream, float eps = kRmsNormEps);

}  // namespace mi::kernels
