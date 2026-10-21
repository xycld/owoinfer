// Causal depthwise conv1d + SiLU and the beta/g gates (see gdn.h).
#include <cuda_bf16.h>

#include "core/cuda_check.h"
#include "kernels/gdn/checks.h"
#include "kernels/gdn/gdn.h"

namespace mi::gdn {
namespace {

using detail::check_aligned;
using detail::check_cu_seqlens;
using detail::check_rows;
using detail::require;

// Index of the sequence containing token t (N is tiny: cond + negative prompt).
__device__ __forceinline__ int seq_start(const int* __restrict__ cu, int num_seqs, int t) {
  int start = 0;
  for (int s = 1; s < num_seqs; ++s) start = t >= cu[s] ? cu[s] : start;
  return start;
}

constexpr int kVec = 8;  // channels per thread (one 16-byte load)

__global__ void conv1d_silu_kernel(const __nv_bfloat16* __restrict__ x, std::int64_t x_stride,
                                   const __nv_bfloat16* __restrict__ w, const int* __restrict__ cu, int num_seqs,
                                   int tokens, __nv_bfloat16* __restrict__ y, std::int64_t y_stride) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr int kGroups = kConvDim / kVec;
  if (idx >= static_cast<std::int64_t>(tokens) * kGroups) return;
  const int t = static_cast<int>(idx / kGroups);
  const int c0 = static_cast<int>(idx % kGroups) * kVec;
  const int start = seq_start(cu, num_seqs, t);

  // weight [c][k]: 8 channels x 4 taps = 32 contiguous bf16.
  __nv_bfloat16 wv[kVec * kConvKernel];
  const uint4* wsrc = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(c0) * kConvKernel);
#pragma unroll
  for (int i = 0; i < 4; ++i) reinterpret_cast<uint4*>(wv)[i] = wsrc[i];

  float acc[kVec];
#pragma unroll
  for (int c = 0; c < kVec; ++c) acc[c] = 0.f;
  // out[t] = sum_k w[k] * x[t - 3 + k]: F.conv1d with padding 3, first seq_len outputs; rows before the
  // sequence start are the zero padding.
#pragma unroll
  for (int k = 0; k < kConvKernel; ++k) {
    const int src = t - (kConvKernel - 1) + k;
    if (src < start) continue;
    uint4 raw = *reinterpret_cast<const uint4*>(x + src * x_stride + c0);
    const __nv_bfloat16* xv = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int c = 0; c < kVec; ++c) {
      acc[c] = fmaf(__bfloat162float(wv[c * kConvKernel + k]), __bfloat162float(xv[c]), acc[c]);
    }
  }
  uint4 out;
  __nv_bfloat16* ov = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
  for (int c = 0; c < kVec; ++c) {
    // conv output is rounded to bf16 before SiLU, which computes in fp32 and rounds again (torch silu).
    const float v = __bfloat162float(__float2bfloat16_rn(acc[c]));
    ov[c] = __float2bfloat16_rn(v / (1.0f + expf(-v)));
  }
  *reinterpret_cast<uint4*>(y + t * y_stride + c0) = out;
}

__global__ void gates_kernel(const __nv_bfloat16* __restrict__ a, std::int64_t a_stride,
                             const __nv_bfloat16* __restrict__ b, std::int64_t b_stride,
                             const __nv_bfloat16* __restrict__ a_log, const __nv_bfloat16* __restrict__ dt_bias,
                             int tokens, float* __restrict__ g, __nv_bfloat16* __restrict__ beta) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= tokens * kNumVHeads) return;
  const int t = idx / kNumVHeads, h = idx % kNumVHeads;
  // b.sigmoid() on bf16: fp32 opmath 1 / (1 + exp(-b)), rounded to bf16.
  const float bv = __bfloat162float(b[t * b_stride + h]);
  beta[idx] = __float2bfloat16_rn(1.0f / (1.0f + expf(-bv)));
  // -A_log.float().exp() * F.softplus(a.float() + dt_bias): softplus threshold 20, beta 1.
  const float x = __bfloat162float(a[t * a_stride + h]) + __bfloat162float(dt_bias[h]);
  const float sp = x > 20.0f ? x : log1pf(expf(x));
  g[idx] = -expf(__bfloat162float(a_log[h])) * sp;
}

}  // namespace

void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& cu_seqlens, const Tensor& y,
                        cudaStream_t stream) {
  const std::int64_t tokens = x.dim(0);
  check_rows(x, DType::kBF16, tokens, kConvDim, "conv input");
  check_rows(y, DType::kBF16, tokens, kConvDim, "conv output");
  require(weight.dtype == DType::kBF16 && weight.device == Device::kCUDA && weight.numel() == kConvDim * kConvKernel &&
              weight.is_contiguous(),
          "conv weight must be contiguous cuda bf16 [8192, 4], got " + weight.describe());
  require(x.stride[0] % kVec == 0 && y.stride[0] % kVec == 0, "conv row strides must be multiples of 8");
  check_aligned(x, 16, "conv input");
  check_aligned(y, 16, "conv output");
  check_aligned(weight, 16, "conv weight");
  check_cu_seqlens(cu_seqlens);
  if (tokens == 0) return;
  // Each output row reads the 3 previous input rows, so in-place (or any overlap) would read already
  // overwritten data. Compare the byte spans the two tensors cover.
  const auto span_end = [&](const Tensor& t) {
    return reinterpret_cast<std::uintptr_t>(t.data) +
           static_cast<std::uintptr_t>((tokens - 1) * t.stride[0] + kConvDim) * sizeof(__nv_bfloat16);
  };
  require(span_end(x) <= reinterpret_cast<std::uintptr_t>(y.data) || span_end(y) <= reinterpret_cast<std::uintptr_t>(x.data),
          "conv input and output must not overlap (in-place is not supported)");
  const std::int64_t work = tokens * (kConvDim / kVec);
  constexpr int kThreads = 256;
  conv1d_silu_kernel<<<static_cast<unsigned>((work + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      x.ptr<__nv_bfloat16>(), x.stride[0], weight.ptr<__nv_bfloat16>(), cu_seqlens.ptr<int>(),
      static_cast<int>(cu_seqlens.dim(0) - 1), static_cast<int>(tokens), y.ptr<__nv_bfloat16>(), y.stride[0]);
  MI_CUDA_CHECK_LAUNCH();
}

void gates(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias, const Tensor& g,
           const Tensor& beta, cudaStream_t stream) {
  const std::int64_t tokens = a.dim(0);
  check_rows(a, DType::kBF16, tokens, kNumVHeads, "a");
  check_rows(b, DType::kBF16, tokens, kNumVHeads, "b");
  check_rows(g, DType::kF32, tokens, kNumVHeads, "g");
  check_rows(beta, DType::kBF16, tokens, kNumVHeads, "beta");
  require(g.is_contiguous() && beta.is_contiguous(), "g and beta must be contiguous");
  for (const Tensor* p : {&a_log, &dt_bias}) {
    require(p->dtype == DType::kBF16 && p->device == Device::kCUDA && p->numel() == kNumVHeads && p->is_contiguous(),
            "A_log / dt_bias must be contiguous cuda bf16 [32], got " + p->describe());
  }
  if (tokens == 0) return;
  constexpr int kThreads = 256;
  const std::int64_t work = tokens * kNumVHeads;
  gates_kernel<<<static_cast<unsigned>((work + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      a.ptr<__nv_bfloat16>(), a.stride[0], b.ptr<__nv_bfloat16>(), b.stride[0], a_log.ptr<__nv_bfloat16>(),
      dt_bias.ptr<__nv_bfloat16>(), static_cast<int>(tokens), g.ptr<float>(), beta.ptr<__nv_bfloat16>());
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::gdn
