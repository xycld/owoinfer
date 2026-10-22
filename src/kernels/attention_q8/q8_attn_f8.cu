// INT8 QK^T + FP8 (e4m3) PV attention: SageAttention's sm89 qk_int_sv_f8_attn_kernel (third_party/sageattention,
// varlen-modified), instantiated as upstream's qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf (SageAttention2++:
// mma.sync m16n8k32 e4m3 with fp16 accumulators over the 2 MMAs of a 64-key tile, flushed into fp32) and
// qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf (fp32 accumulators), per-warp Q / per-block K scales, per-channel
// V scale applied to the output, bf16 output, non-causal, head_dim 128. The e4m3 MMAs exist on sm_89+ only (and the
// fp16-accumulating one from nvcc 12.8); otherwise the vendored wrappers compile to __brkpt, so attention_q8() checks
// int8_fp8_kernel_arch() / int8_fp8_f16_accum_compiled() before launching.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

#include "core/cuda_check.h"
#include "kernels/attention_q8/q8_kernels.h"

// Vendored kernel last: its headers define global macros (div_ceil, WARP_SIZE, PACK_SIZE_*).
#include "csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh"

namespace mi::attention_q8::detail {
namespace {

// The vendored mma.cuh's MMA_F8F8F16_M16N8K16_ENABLED condition (mma.sync m16n8k32 .f16.e4m3.e4m3.f16).
constexpr bool kF16AccumMma = __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ >= 1208;

constexpr int kWarpQ = 32, kWarpK = 64;

template <bool kFp16Accum>
constexpr auto kKernel = qk_int_sv_f8_attn_kernel<kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim, DataType::kInt8,
                                                  QuantGranularity::kPerWarp, QuantGranularity::kPerWarp, float, true,
                                                  nv_bfloat16, ComputeUnit::kCudaCore, MaskMode::kNone, false, true,
                                                  false, kFp16Accum>;

template <bool kFp16Accum>
void launch(const AttnParams& p, cudaStream_t stream) {
  auto kernel = kKernel<kFp16Accum>;
  // Q, K and V (e4m3) tiles while computing; the output tile reuses the space at the end.
  constexpr std::size_t kSmem =
      std::max<std::size_t>(kCtaQ * kHeadDim + kCtaK * kHeadDim + kCtaK * kHeadDim, kCtaQ * kHeadDim * sizeof(half));
  MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
  const dim3 grid((p.max_seqlen + kCtaQ - 1) / kCtaQ, p.heads, p.batch);
  const dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));
  kernel<<<grid, block, kSmem, stream>>>(
      const_cast<std::int8_t*>(p.q), const_cast<std::int8_t*>(p.k),
      static_cast<std::int8_t*>(const_cast<void*>(p.v)), static_cast<nv_bfloat16*>(p.o), nullptr,
      const_cast<float*>(p.q_scale), const_cast<float*>(p.k_scale), const_cast<float*>(p.v_scale), nullptr, p.cu_q,
      p.cu_k, p.k_scale_blocks, static_cast<std::uint32_t>(p.kv_groups), p.stride_seq_q, p.stride_h_q, p.stride_seq_k,
      p.stride_h_k, p.stride_h_v, p.stride_seq_v, p.stride_seq_o, p.stride_h_o, p.sm_scale, p.p_offset);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void run_int8_fp8(const AttnParams& p, bool f32_accum, cudaStream_t stream) {
  if (f32_accum) {
    launch<false>(p, stream);
  } else if constexpr (kF16AccumMma) {
    launch<true>(p, stream);
  } else {
    throw std::runtime_error("attention_q8: int8_fp8 needs nvcc >= 12.8 at build time (fp16-accumulating e4m3 MMA)");
  }
}

int int8_fp8_kernel_arch(bool f32_accum) {
  if (f32_accum) return kernel_arch(kKernel<false>);
  if constexpr (kF16AccumMma) return kernel_arch(kKernel<true>);
  return 0;
}

bool int8_fp8_f16_accum_compiled() { return kF16AccumMma; }

}  // namespace mi::attention_q8::detail
