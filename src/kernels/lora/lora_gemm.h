// DiT block GEMMs with the LoRA side branch fused into the GEMM: after the base mainloop, a second bf16 mainloop over
// the rank dimension adds t · Bᵀ into the same fp32 accumulators, so Δ = Σ s_i B_i (A_i x) is never stored (storing it
// and adding it in the epilogue measured +18 % on an INT8 block at 1024^2 for one rank-64 LoRA, mostly Δ's traffic).
// t = bf16(x · Aᵀ * coef) comes from models/lora compute_t.
//
// Per output element (r, c), with acc_lora = Σ_j t[r, j] * b[c, j] accumulated by bf16 mma.sync (fp32) after the base:
//   bf16 base:  acc = (x · wᵀ)[r, c] (the fused_gemm.h mainloop, same order) + acc_lora
//   INT8 base:  acc = ((float)acc_i32 * x_scale[r]) * w_scale[c] (+ bias[c]) (quant.h's dequantization, done in
//               registers before the second mainloop) + acc_lora
// then the plain variants' epilogue on acc as the rounded GEMM output: out = bf16(acc); residual + bf16(bf16(acc) *
// gate); situ(bf16(acc_h), bf16(acc_g)). Δ is not rounded to bf16 before the add.
// With t = 0 every output equals the plain variant's bit for bit (acc + 0 = acc); a side branch with all strengths 0
// is skipped by the caller (models/lora SideBranches::site() returns null).
//
// Slices: the rank columns of t serve output-column slices (the fused qkv site has q, k, v at columns 0, dim, 2 dim);
// output columns [row0, row0 + rows) get [t[:, col0 : col0 + cols] | t[:, col0b : col0b + colsb]] · bᵀ with b bf16
// [rows, cols + colsb] contiguous (the second range is optional: colsb = 0), other columns get no LoRA term. Two
// ranges let fused-qkv adapters keep one shared range of t columns next to each slice's own adapters. Each slice must
// cover whole threadblock tiles: row0 and rows multiples of 256 (or the slice ends at n). For the SiTU-GLU variants
// b's rows are in the GEMM's column order (up_gate_gemm_row), as models/lora stores them.
// Operands otherwise as the plain variants (fused_gemm.h, quant.h). t: 16-byte aligned, ldt % 8 == 0; col0, cols
// multiples of 8. No workspace, no host synchronization; graph-capturable. Invalid arguments throw
// std::invalid_argument.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "core/tensor.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/quant/quant.h"

namespace mi::kernels::lora {

// Weight row of the stored [hidden | gate] ffn_up_gate matrix ([2F, K], half = F) that GEMM column c of the SiTU-GLU
// GEMMs reads (kernels::fused::gemm_situ_glu, kernels::quant::int8_gemm_situ_glu and the variants here): groups of 4
// hidden columns, then their 4 gate columns. The kernels compute the same map (kernels/fused UpGatePermuteB).
inline constexpr std::int64_t up_gate_gemm_row(std::int64_t c, std::int64_t half) {
  return ((c / 4) % 2) * half + (c / 8) * 4 + c % 4;
}

struct LoraSlices {
  struct Slice {
    std::int64_t col0 = 0, cols = 0;    // t columns, first range
    std::int64_t col0b = 0, colsb = 0;  // second range (optional)
    std::int64_t row0 = 0, rows = 0;    // output (GEMM) columns
    const __nv_bfloat16* b = nullptr;   // [rows, cols + colsb]
  };
  static constexpr int kMaxSlices = 3;
  const __nv_bfloat16* t = nullptr;  // [m, >= col0 + cols of every slice], row stride ldt
  std::int64_t ldt = 0;
  int count = 0;
  Slice slice[kMaxSlices];
};

// bf16 base (fused_gemm.h operands).
void gemm_lora(const Tensor& out, const Tensor& x, const Tensor& w, const LoraSlices& lora, cudaStream_t stream,
               fused::GemmTile tile = fused::GemmTile::kAuto);
void gemm_gated_residual_lora(const Tensor& residual, const Tensor& x, const Tensor& w, const Tensor& gate,
                              const LoraSlices& lora, cudaStream_t stream, fused::GemmTile tile = fused::GemmTile::kAuto);
void gemm_situ_glu_lora(const Tensor& out, const Tensor& x, const Tensor& w_up_gate, const LoraSlices& lora,
                        cudaStream_t stream, fused::GemmTile tile = fused::GemmTile::kAuto);

// INT8 ConvRot base (quant.h operands).
void int8_gemm_lora(const quant::Int8Operands& op, const __nv_bfloat16* bias, const LoraSlices& lora,
                    __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream,
                    quant::Int8Tile tile = quant::Int8Tile::kAuto);
void int8_gemm_gated_residual_lora(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                                   const LoraSlices& lora, __nv_bfloat16* residual, std::int64_t ldr,
                                   cudaStream_t stream, quant::Int8Tile tile = quant::Int8Tile::kAuto);
void int8_gemm_situ_glu_lora(const quant::Int8Operands& op, const LoraSlices& lora, __nv_bfloat16* out,
                             std::int64_t ldo, cudaStream_t stream, quant::Int8Tile tile = quant::Int8Tile::kAuto);

}  // namespace mi::kernels::lora
