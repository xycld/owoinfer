// CUTLASS GEMMs for the DiT linears with the following elementwise op fused into the epilogue, bit-exact
// against the op-by-op path (kernels::Gemm::linear, then the kernels/basic op).
//
// Why bit-exact: these kernels and the cuBLASLt kernels kernels::Gemm selects at the DiT shapes all run
// bf16 mma.sync m16n8k16 with fp32 accumulators, walk K in ascending 16-wide steps from a zero accumulator,
// and neither split nor slice K. Every output element is therefore the same chain of MMAs on the same
// inputs; the tile shape only decides which CTA computes it (measured bit-identical, tests/unit/test_fused_gemm).
// The epilogue then rounds the fp32 accumulator to bf16 first, exactly as the GEMM output tensor did, and
// applies the following op in its fp32-op / bf16-rounding order.
//
// Operands: x [M, K] and out rows with one row stride each (leading dims flatten to M rows), w [N, K]
// contiguous, all bf16, 16-byte aligned with row strides that are multiples of 8 elements. sm_80 or newer.
// No workspace, no host synchronization; graph-capturable.
#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace mi::kernels::fused {

// out[M, N] = x · wᵀ (bf16 out). Same bits as Gemm::linear(out, x, w, nullptr, ...).
void gemm(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream);

// residual[M, N] = residual + bf16(bf16(x · wᵀ) * gate): the o-proj / ffn_down linear followed by
// kernels::gated_residual (P:142, P:148). gate: bf16 [N]. residual is read and written in place.
void gemm_gated_residual(const Tensor& residual, const Tensor& x, const Tensor& w, const Tensor& gate,
                         cudaStream_t stream);

// out[M, F] = situ_glu(x · w_upgateᵀ) for the [hidden | gate] ffn_up_gate weight (P:147), given that weight
// in the layout of interleave_up_gate (w_interleaved [2F, K]). Same bits as Gemm::linear into [M, 2F] then
// kernels::situ_glu. The [M, 2F] intermediate is never stored.
void gemm_situ_glu(const Tensor& out, const Tensor& x, const Tensor& w_interleaved, cudaStream_t stream);

// Row permutation of a [hidden | gate] weight [2F, K] that puts each group of kUpGateGroup hidden rows next to
// the matching gate rows: out row 2*g*G + s*G + i = w row s*F + g*G + i (G = kUpGateGroup, s = 0 hidden, 1 gate).
// One CUTLASS epilogue access (8 accumulator columns) then holds 4 (hidden, gate) pairs. F % G == 0; out must
// not alias w.
inline constexpr int kUpGateGroup = 4;
void interleave_up_gate(const Tensor& w, const Tensor& out, cudaStream_t stream);

// Threadblock tile of the GEMMs above. kAuto picks per shape; the others are for benchmarks and the
// exactness tests (every tile gives the same bits). Process-wide.
enum class GemmTile { kAuto, k128x256, k256x128, k128x128 };
void set_gemm_tile_override(GemmTile tile);

}  // namespace mi::kernels::fused
