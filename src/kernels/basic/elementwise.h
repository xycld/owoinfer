// Elementwise ops of the MOSS DiT and the Qwen3.5 text encoder, bit-exact against torch eager: every
// step is one fp32 op rounded to bf16, in the reference's order.
//
// Tensors are CUDA views [..., C] with a contiguous last dim and arbitrary leading strides; out may alias
// an input of the same layout. Broadcast vectors (gate) are contiguous bf16 [C].
#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace mi::kernels {

// MOSS _situ (P:66-67): (4*tanh(g/4)*sigmoid(g)) * (25*tanh(h/25)), op by op in bf16.
// Fused form reads the ffn_up_gate output [..., 2F] chunked as [hidden | gate] (P:147); out is [..., F].
void situ_glu(const Tensor& up_gate, const Tensor& out, cudaStream_t stream);
// Separate inputs (text refiner: hidden = ffn_up(y), gate = ffn_gate(y), P:110).
void situ(const Tensor& hidden, const Tensor& gate, const Tensor& out, cudaStream_t stream);

// out = x + y * gate (P:142, P:148), gate a per-sample [C] vector.
void gated_residual(const Tensor& x, const Tensor& y, const Tensor& gate, const Tensor& out, cudaStream_t stream);
// out = x + y.
void residual_add(const Tensor& x, const Tensor& y, const Tensor& out, cudaStream_t stream);

// out = silu(x) (time embedding MLP).
void silu(const Tensor& x, const Tensor& out, cudaStream_t stream);
// out = silu(gate) * up (Qwen3_5MLP, Q:821).
void silu_mul(const Tensor& gate, const Tensor& up, const Tensor& out, cudaStream_t stream);
// out = x * sigmoid(gate) (Qwen3_5Attention output gate, Q:803).
void sigmoid_mul(const Tensor& x, const Tensor& gate, const Tensor& out, cudaStream_t stream);

// bf16 <-> fp32 conversion (fp32 -> bf16 rounds to nearest even, as torch .to()); same shape, any of
// bf16->f32, f32->bf16, or a same-dtype copy.
void cast(const Tensor& in, const Tensor& out, cudaStream_t stream);

}  // namespace mi::kernels
