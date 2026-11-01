// Volta (sm_70) tensor-core kernels. The SIMT conversions and fp16 cuBLASLt path are shared with sm75; attention
// and quantized linears use Volta's fp16 MMA, with fp32 accumulation and the engine's bf16 epilogue rounding.
#pragma once

#include "core/tensor.h"
#include "kernels/quant/quant.h"

namespace mi::kernels::sm70 {

bool available();
// Cached per device; true only on compute capability 7.0/7.2 in a build carrying the Volta kernel family.
bool volta();

void flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
               const Tensor& cu_seqlens_k, int max_seqlen, bool causal, float scale, void* workspace,
               std::size_t workspace_bytes, cudaStream_t stream);

// INT8 inputs are converted exactly to fp16 in shared memory. Partial integer dot products fit fp32 exactly;
// periodically folding them into int32 preserves the native INT8 path's bits for every supported K.
void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream);
void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream);
void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream);

}  // namespace mi::kernels::sm70
