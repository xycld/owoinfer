// CUTLASS GEMMs for the DiT linears with the following elementwise op fused into the epilogue, bit-exact
// against the op-by-op path (kernels::Gemm::linear, then the kernels/basic op) whenever cuBLASLt runs an
// algorithm matches_cublaslt() accepts for that linear.
//
// Why: these kernels and cuBLASLt's algorithms 5, 6 and 21 all run bf16 mma.sync m16n8k16 with fp32 accumulators and
// walk K in ascending 16-wide steps from a zero accumulator, without splitting K. Every output element is then the
// same chain of MMAs on the same inputs; the tile shape only decides which CTA computes it. cuBLASLt does not always
// pick such an algorithm: measured on sm_89 (cuBLASLt 13, tests/unit/test_fused_gemm RowCountSweep), every M from 1
// to 1200 and every 16th up to 8800 (20674 for o/down) of the o, ffn_down, ffn_up_gate and qkv shapes is bit-identical
// with algorithms 5/6/21 at split-K 1 (all tiles seen), and differs with split-K (ffn_down, K = 8192, at M = 17..105,
// 168..256), algorithm 31 (ffn_down, M = 321..383) and the GEMV algorithm 13 (M = 1). Callers query the algorithm
// (Gemm::algo_config) and run the other shapes op by op; models/moss_dit does this per shape, and runs fused only the
// cuBLASLt configurations its startup self-check (check_fused_gemms, once per GPU, library and kFusedKernelsVersion)
// found bit-identical.
// The epilogue then rounds the fp32 accumulator to bf16 first, exactly as the GEMM output tensor did, and
// applies the following op in its fp32-op / bf16-rounding order.
//
// Operands: x [M, K] and out rows with one row stride each (leading dims flatten to M rows), w [N, K]
// contiguous, all bf16, 16-byte aligned with row strides that are multiples of 8 elements. sm_80 or newer.
// No workspace, no host synchronization; graph-capturable.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"

namespace mi::kernels::fused {

// Engine numerics version of these kernels and of the rule that routes a linear to them (matches_cublaslt,
// models/moss_dit fused_route): bump it whenever either changes.
inline constexpr std::uint32_t kFusedGemmVersion = 1;
// What a startup self-check's verdict depends on besides the hardware key (GPU, driver, cuBLASLt): the fused kernels
// and the op-by-op epilogue kernels they must match bit for bit (kernels::kElementwiseNumericsVersion: situ_glu,
// gated_residual). A self-check cached by another build (pipeline/hardware_profile.h) runs again when either changes.
static_assert(kernels::kElementwiseNumericsVersion < 1000);
inline constexpr std::uint32_t kFusedKernelsVersion = kFusedGemmVersion * 1000 + kernels::kElementwiseNumericsVersion;

// Threadblock tile of the GEMMs below. kAuto picks per shape (set_gemm_tile_override can force one process-wide, for
// benchmarks); the others are for the exactness checks (every tile gives the same bits).
enum class GemmTile { kAuto, k128x256, k256x128, k128x128 };
void set_gemm_tile_override(GemmTile tile);

// out[M, N] = x · wᵀ (bf16 out). Same bits as Gemm::linear(out, x, w, nullptr, ...) where matches_cublaslt.
void gemm(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream);

// residual[M, N] = residual + bf16(bf16(x · wᵀ) * gate): the o-proj / ffn_down linear followed by
// kernels::gated_residual (P:142, P:148). gate: bf16 [N]. residual is read and written in place.
void gemm_gated_residual(const Tensor& residual, const Tensor& x, const Tensor& w, const Tensor& gate,
                         cudaStream_t stream, GemmTile tile = GemmTile::kAuto);

// out[M, F] = situ_glu(x · w_up_gateᵀ) for the stored [hidden | gate] ffn_up_gate weight [2F, K] (P:147): the same
// bits as Gemm::linear into [M, 2F] then kernels::situ_glu, without storing the [M, 2F] intermediate. The GEMM reads
// w's rows permuted so that each epilogue access holds 4 hidden columns and their 4 gate columns (F % 4 == 0).
void gemm_situ_glu(const Tensor& out, const Tensor& x, const Tensor& w_up_gate, cudaStream_t stream,
                   GemmTile tile = GemmTile::kAuto);

// Whether the GEMMs above give Gemm::linear's bits for a problem cuBLASLt runs with `algo` (Gemm::algo_config of the
// op-by-op linear): one of the K-ordered algorithms 5, 6, 21, no split-K, no custom option. Conservative: anything
// else is treated as a different accumulation order.
bool matches_cublaslt(const kernels::Gemm::AlgoConfig& algo);

}  // namespace mi::kernels::fused
