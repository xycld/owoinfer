// Turing (sm_75) kernel family: the pieces of the engine that need bf16 tensor cores, cp.async, TF32 or > 64 KiB of
// shared memory on sm_80+ have an sm_75 fallback here, and the host-side entry points of the sm_80 kernels dispatch to
// them when the current device is compute capability 7.x.
//
// This translation unit family is compiled for sm_75 only (cmake/modules/sm75.cmake, CUDA_ARCHITECTURES 75-real, in
// its own OBJECT library), so it may use half-precision MMA and must not assume any sm_80 instruction. The host code
// that dispatches into it lives in the sm_80 translation units and is arch-independent.
//
// Numerics: sm_75 has fp16 tensor cores but no bf16 ones, so every tensor-core operand is converted bf16 -> fp16 at
// the boundary (lossless for |x| in fp16's normal range). GEMM activations get a per-row power-of-two scale that keeps
// rows above 65504 in range; weights and attention operands above it would become inf, values below about 6e-5 lose
// precision (fp16 subnormals). Accumulation stays fp32 where the sm_80 kernel had fp32 accumulation, and outputs are
// rounded once, from fp32.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mi::kernels::sm75 {

// True when this build carries the sm_75 kernels.
bool available();

// True when `major` needs them (compute capability < 8).
bool needed(int major);

// True when bf16 GEMM weights must be stored fp16: this build has the sm_75 kernels and the current device is compute
// capability 7.x (the fp16 GEMM path in kernels/basic/gemm.cpp). Cached per device.
bool fp16_weights();

// True when the current device runs the sm_75 kernel family (compute capability 7.x): the host dispatchers in the sm_80
// translation units branch on this.
inline bool turing() { return fp16_weights(); }

// Converts a bf16 GEMM weight to fp16 in place and marks the view kF16. No-op when fp16_weights() is false or the
// tensor is already fp16; throws std::logic_error for any other dtype. For resident weights (converted once).
void convert_weight_inplace(Tensor& w, cudaStream_t stream);

// Same conversion, but always runs: streamed weight buffers are refilled with bf16 bytes on every use while their view
// keeps the kF16 label from the first conversion.
void convert_weight_storage_inplace(Tensor& w, cudaStream_t stream);

// Elements of one contiguous bf16 -> fp16 conversion. `dst` may equal `src` (an in-place element-wise conversion).
void convert_bf16_to_f16(void* dst, const void* src, std::int64_t n, cudaStream_t stream);
void convert_f16_to_bf16(void* dst, const void* src, std::int64_t n, cudaStream_t stream);

// GEMM input rows: bf16 [rows, cols] (src_stride elements) -> fp16 `dst` (dst_stride), each row multiplied by the
// power of two that brings its largest magnitude within fp16's range (1 for in-range rows); inv_scale[r] (fp32,
// device) receives the inverse.
void convert_rows_bf16_to_f16_scaled(void* dst, float* inv_scale, const void* src, std::int64_t rows,
                                     std::int64_t cols, std::int64_t src_stride, std::int64_t dst_stride,
                                     cudaStream_t stream);

// GEMM output pass: `acc` is fp32 [rows, cols] (acc_stride), `out` bf16 or f32 (out_f32) [rows, cols] (out_stride),
// `bias` nullptr or [cols] of the output dtype. out[r][c] = out_type(acc[r][c] * inv_scale[r] + bias[c]), the fp32 add
// before the single final rounding (as the sm_80 path's cuBLASLt BIAS epilogue).
void convert_gemm_out_f32(void* out, bool out_f32, const float* acc, const float* inv_scale, const void* bias,
                          std::int64_t rows, std::int64_t cols, std::int64_t out_stride, std::int64_t acc_stride,
                          cudaStream_t stream);

// VAE fp32 fallbacks (the reference decoder's kernels need TF32 tensor cores and cp.async).
// conv2d: y = conv(x) + bias + optional residual, x/y NHWC fp32 [N,H,W,C], w KRSC fp32 [K,R,R,C], stride 1, pad R/2,
// R in {1,3}. No workspace.
void vae_conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
                cudaStream_t stream);
// Mid-block attention: o = softmax(q k^T * scale) v, fp32, one head, [B, L, D] with D a multiple of 32 and any row
// stride (>= D) for q/k/v; o is contiguous.
void vae_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, float scale,
                        cudaStream_t stream);
// torch conv weight [K,C,R,S] fp32 -> [K,R,S,C] in dst's dtype (kF32 or kBF16).
void vae_weight_to_krsc(const Tensor& w_kcrs, const Tensor& w_krsc, cudaStream_t stream);

// Exact (non-approximate) attention, the sm_75 counterpart of attention::flash_fwd for the same arguments and
// supported configurations (head_dim 128 non-causal for the DiT, head_dim 256 causal for the text encoder, whose
// causal mask is aligned bottom-right as FlashAttention-2's). q, k, v and out are bf16 and never modified but `out`:
// the kernel converts q/k/v to fp16 as it stages them and writes `out` from fp32 (attention_sm75.cu, fp16 m16n8k8 MMA).
void flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
               const Tensor& cu_seqlens_k, int max_seqlen, bool causal, float scale, void* workspace,
               std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace mi::kernels::sm75
