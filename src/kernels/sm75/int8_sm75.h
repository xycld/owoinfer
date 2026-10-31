// sm_75 INT8 ConvRot GEMMs: the sm_80 kernels use m16n8k32 INT8 MMA (mma.sync.s8, sm_80 only). Turing's native INT8
// shape is m8n8k16, so the same CUTLASS 2.x EVT kernel template is instantiated with arch::Sm75 and GemmShape<8,8,16>,
// with threadblock/warp tiles that fit Turing's 64 KiB of shared memory. The epilogues (dequantize + bias, gated
// residual, SiTU-GLU) are the sm_80 ones unchanged (they are arch-neutral visitor trees).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "core/tensor.h"
#include "kernels/quant/quant.h"

namespace mi::kernels::sm75 {

// Same contracts as kernels/quant/quant.h's int8_gemm / int8_gemm_gated_residual / int8_gemm_situ_glu (operand checks,
// output layout and numerics); the host-side dispatcher in those translation units routes compute capability 7.x here.
void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream);
void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream);
void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream);

}  // namespace mi::kernels::sm75
