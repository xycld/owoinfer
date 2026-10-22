// INT8 QK^T + fp16 PV attention: SageAttention's qk_int_sv_f16_attn_kernel (third_party/sageattention, varlen-
// modified), instantiated as upstream's qk_int8_sv_f16_accum_f16_attn_inst_buf ("fp16+fp32": WARP_Q 16, fp16
// accumulation per 64-key tile flushed into fp32) and qk_int8_sv_f16_accum_f32_attn (fp32 accumulation, WARP_Q 32),
// with per-warp Q / per-block K scales, bf16 output, non-causal, head_dim 128.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "core/cuda_check.h"
#include "kernels/attention_q8/q8_kernels.h"

// Vendored kernel last: its headers define global macros (div_ceil, WARP_SIZE, PACK_SIZE_*).
#include "csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh"

namespace mi::attention_q8::detail {
namespace {

constexpr int kWarpK = 64;

template <int kWarpQ, bool kInstBuf>
constexpr auto kKernel = qk_int_sv_f16_attn_kernel<kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim, DataType::kInt8,
                                                   QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, float,
                                                   kInstBuf, nv_bfloat16, ComputeUnit::kTensorCore, MaskMode::kNone,
                                                   false, false>;

template <int kWarpQ, bool kInstBuf>
void launch(const AttnParams& p, cudaStream_t stream) {
  auto kernel = kKernel<kWarpQ, kInstBuf>;
  // Q, K and V tiles while computing; the output tile reuses the space at the end.
  constexpr std::size_t kSmem = std::max<std::size_t>(kCtaQ * kHeadDim + kCtaK * kHeadDim + kCtaK * kHeadDim * sizeof(half),
                                                      kCtaQ * kHeadDim * sizeof(half));
  MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
  const dim3 grid((p.max_seqlen + kCtaQ - 1) / kCtaQ, p.heads, p.batch);
  const dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));
  kernel<<<grid, block, kSmem, stream>>>(
      const_cast<std::int8_t*>(p.q), const_cast<std::int8_t*>(p.k), static_cast<half*>(const_cast<void*>(p.v)),
      static_cast<nv_bfloat16*>(p.o), nullptr, const_cast<float*>(p.q_scale), const_cast<float*>(p.k_scale), nullptr,
      p.cu_q, p.cu_k, p.k_scale_blocks, static_cast<std::uint32_t>(p.kv_groups), p.stride_seq_q, p.stride_h_q,
      p.stride_seq_k, p.stride_h_k, p.stride_seq_v, p.stride_h_v, p.stride_seq_o, p.stride_h_o, p.sm_scale);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void run_int8_fp16(const AttnParams& p, bool f32_accum, cudaStream_t stream) {
  if (f32_accum) {
    launch<32, false>(p, stream);
  } else {
    launch<16, true>(p, stream);
  }
}

int int8_fp16_kernel_arch(bool f32_accum) {
  return f32_accum ? kernel_arch(kKernel<32, false>) : kernel_arch(kKernel<16, true>);
}

}  // namespace mi::attention_q8::detail
