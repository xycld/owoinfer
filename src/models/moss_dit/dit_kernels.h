// Small CUDA helpers of the DiT: latent <-> token layout at the model boundary (P:247-248, P:259), host-to-device
// writes of tiny per-call parameters, and the x0 extrapolation of a latent preview.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mi::moss_dit {

// hidden_states.flatten(2).transpose(1, 2).to(bf16): latents fp32 [C, h, w] (contiguous) -> tokens
// bf16 [h*w, C] (contiguous), round to nearest even.
void latent_to_tokens(const Tensor& latents, const Tensor& tokens, cudaStream_t stream);
// The same into columns [col0, col0 + C) of tokens bf16 [h*w, C_total] (contiguous): one part of a channel
// concatenation (control adapters' widened input embedder).
void latent_to_tokens(const Tensor& latents, const Tensor& tokens, std::int64_t col0, cudaStream_t stream);

// x.transpose(1, 2).reshape(C, h, w).float(): tokens bf16 [h*w, C] (contiguous) -> fp32 [C, h, w] (contiguous).
void tokens_to_latent(const Tensor& tokens, const Tensor& latents, cudaStream_t stream);

// out = x + c * (x - x_prev), fp32 elementwise (contiguous CUDA f32 tensors of one size; out may alias x): the x0
// estimate of a latent preview, x0 = x_t - sigma_t * (x_t - x_prev) / (sigma_t - sigma_prev), with
// c = sigma_t / (sigma_prev - sigma_t).
void extrapolate_latents(const Tensor& x, const Tensor& x_prev, float c, const Tensor& out, cudaStream_t stream);

// Packed signed int4 (two values per byte, even column in the low nibble: the kernels/quant4 layout) -> int8 values:
// in [rows, cols / 2] bytes (row stride ld_in bytes) -> out [rows, cols] (row stride ld_out). cols even. The INT4
// linears' path to the INT8 LoRA GEMMs: the int32 dot products, and so the GEMM results, are the INT4 GEMM's.
void unpack_int4(const void* in, std::int64_t ld_in, void* out, std::int64_t ld_out, std::int64_t rows, std::int64_t cols,
                 cudaStream_t stream);

// out[r] = bf16(sum_c w[r, c] * x[c] + bias[r]) in fp32 (w bf16, or fp16 on sm_75, [n, k] contiguous, widened; x, bias
// f32 device arrays; out bf16 [n]): an fp32 nn.Linear on one row, rounded once. k % 8 == 0, w 16-byte aligned.
void linear_fp32_row(const Tensor& w, const float* x, const float* bias, const Tensor& out, cudaStream_t stream);

// Copies `bytes` (<= kMaxParamBytes) host bytes to dst through kernel arguments: no pinned staging,
// no host synchronization (a pageable cudaMemcpyAsync would wait for the stream), graph-capturable.
// dst: device, 4-byte aligned; bytes a multiple of 4.
inline constexpr std::size_t kMaxParamBytes = 2048;
void write_params(void* dst, const void* src, std::size_t bytes, cudaStream_t stream);

}  // namespace mi::moss_dit
