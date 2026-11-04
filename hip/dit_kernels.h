// The DiT's ops outside its blocks (dit_kernels.hip): the linears with a bias (embedders, time MLP, modulations, the text
// refiner's), the refiner's un-modulated RMSNorm, SiLU, SiTU-GLU over a GEMM output, and the RoPE table. Same conventions as hip_block.h:
// bf16 bits, 16-byte aligned operands with row strides that are multiples of 8 elements, asynchronous on `stream`, no
// allocation. The GEMMs are gemm_tile.h's kernel with fp32 accumulators and K ascending.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

#include "hip_block.h"

namespace hip_block {

// out[m, n] = bf16(x[m, k] · w[n, k]ᵀ + bias[n]): nn.Linear with a bias, the bias added to the fp32 sum before its one
// rounding (cuBLASLt's bias epilogue).
void gemm_bias(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w, std::int64_t ldw,
               const bf16_bits* bias, int m, int n, int k, hipStream_t stream);

// residual[m, n] = bf16(residual + bf16(x · wᵀ + bias)): that linear, then a residual add (torch's x + y in bf16).
void gemm_bias_residual(bf16_bits* residual, std::int64_t ldr, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w,
                        std::int64_t ldw, const bf16_bits* bias, int m, int n, int k, hipStream_t stream);

// out[rows, d] = bf16(bf16(rms_norm(x)) * weight): rms_norm_moss without modulation, eps 1e-6. The fp32 mean of squares
// is summed in the order of torch's reduce kernel for this row count (kernels/basic/norm.cu): bw threads per row with 4
// running values each, a halving tree down to 32, then a shuffle-down tree. bw is 32 from 16 rows up and grows below,
// which rms_norm_modulate (hip_block.h) does not follow. d a multiple of 4, >= 128. row_stats, for checks: when not
// null, row r's fp32 sum of squares in that order (before the mean factor) goes to row_stats[2 r] and its
// rsqrt(mean + eps) to row_stats[2 r + 1].
void rms_norm(const bf16_bits* x, std::int64_t ldx, const bf16_bits* weight, bf16_bits* out, std::int64_t ldo,
              std::int64_t rows, int d, hipStream_t stream, float* row_stats = nullptr);

// out[i] = bf16(silu(x[i])) = bf16(x / (1 + exp(-x))) in fp32, n contiguous values.
void silu(const bf16_bits* x, bf16_bits* out, std::int64_t n, hipStream_t stream);

// out[r, c] = bf16(situ(up_gate[r, c], up_gate[r, f + c])) for c < f: MOSS _situ over the [hidden | gate] halves of a
// GEMM output row (elementwise.cu's situ_glu).
void situ_glu(const bf16_bits* up_gate, std::int64_t ld, bf16_bits* out, std::int64_t ldo, std::int64_t rows, int f,
              hipStream_t stream);

// The RoPE table rows of one joint sequence (P:178-185, kernels/basic/rope.cu's moss_table_kernel): out bf16
// [height * width + text_length, 128], contiguous. Image rows first (coordinates r = 0, f = text_length, then h and w
// centred by the fp32 (n - 1) / 2), then the text rows (f = 0 .. text_length - 1); each row is sin of its 64 phases, then
// their cos, the phase the fp32 product of the coordinate and its band's frequency. height = width = 0 gives the text
// rows alone, which the text refiner uses.
void rope_table(bf16_bits* out, int height, int width, int text_length, hipStream_t stream);

}  // namespace hip_block
