// Internal to src/kernels/quant and src/kernels/nvfp4: the MOSS RMSNorm + modulate prologue of the fused activation
// quantizers (quant::rms_norm_modulate_rotate_quantize, nvfp4::rms_norm_modulate_quantize_nvfp4), bit for bit the
// bf16 tensor kernels::rms_norm_modulate_moss produces for the same rows.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "kernels/basic/detail.h"
#include "kernels/basic/norm.h"

namespace mi::kernels::detail {

// 1 / rms of row xr in the fp32 order of kernels/basic/norm.cu's rms_norm_kernel launched with bw = 32 * kVT threads
// per row (torch's reduce: per-thread 4-way vector accumulators, a shared-memory halving tree down to 32 values, then
// a shuffle-down tree). One warp emulates the bw threads: lane l plays threads l + 32 * t, t < kVT; the halving
// steps (offsets >= 32) pair threads within a lane.
template <int kVT>
__device__ float torch_inv_rms(const __nv_bfloat16* xr, int k, float factor, int lane) {
  constexpr int kBw = 32 * kVT;
  float acc[kVT][4];
#pragma unroll
  for (int t = 0; t < kVT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
  for (int base = 0; base * 4 < k; base += kBw) {
#pragma unroll
    for (int t = 0; t < kVT; ++t) {
      const int idx = base + lane + 32 * t;
      if (idx * 4 + 3 < k) {
        const uint2 raw = *reinterpret_cast<const uint2*>(xr + idx * 4);
        const auto* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
        const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
        acc[t][0] = add(acc[t][0], mul(a.x, a.x));
        acc[t][1] = add(acc[t][1], mul(a.y, a.y));
        acc[t][2] = add(acc[t][2], mul(b.x, b.x));
        acc[t][3] = add(acc[t][3], mul(b.y, b.y));
      }
    }
  }
  float v[kVT];
#pragma unroll
  for (int t = 0; t < kVT; ++t) v[t] = add(add(add(acc[t][0], acc[t][1]), acc[t][2]), acc[t][3]);
#pragma unroll
  for (int off = kVT / 2; off >= 1; off >>= 1) {
#pragma unroll
    for (int t = 0; t < off; ++t) v[t] = add(v[t], v[t + off]);
  }
  float s = v[0];
  for (int off = 16; off > 0; off >>= 1) s = add(s, __shfl_down_sync(0xffffffffu, s, off));
  return rsqrtf(add(mul(s, factor), kernels::kRmsNormEps));
}

// One element of kernels/basic/norm.cu kMossModulate: bf16(bf16(bf16(bf16(x * r) * w) * bf16(1 + scale)) + shift).
__device__ __forceinline__ float moss_modulate(float x, float r, float w, float scale, float shift) {
  const float y = mul(rbf(mul(x, r)), w);
  const float one_plus = rbf(add(1.0f, scale));
  return rbf(add(rbf(mul(rbf(y), one_plus)), shift));
}

// Threads per row of torch's last-dim fp32 reduce for `rows` rows of length d (setReduceConfig, vectorized input,
// vt0 = 4, 512 threads); throws where torch would split a row across warps. Source of truth: kernels/basic/norm.cu
// torch_reduce_shape, which norm.h does not expose; keep the two in sync.
inline int torch_reduce_threads(const char* op, std::int64_t rows, std::int64_t d) {
  const auto last_pow2 = [](std::int64_t n) {
    std::int64_t p = 1;
    while (p * 2 <= n) p *= 2;
    return p;
  };
  constexpr std::int64_t kMaxThreads = 512;
  const std::int64_t dim0_pow2 = d / 4 < kMaxThreads ? last_pow2(d / 4) : kMaxThreads;
  const std::int64_t dim1_pow2 = rows < kMaxThreads ? last_pow2(rows) : kMaxThreads;
  std::int64_t bw = std::min<std::int64_t>(dim0_pow2, 32);
  const std::int64_t bh = std::min(dim1_pow2, kMaxThreads / bw);
  bw = std::min(dim0_pow2, kMaxThreads / bh);
  if ((d + bw - 1) / bw >= std::min<std::int64_t>(bh * 16, 256)) {
    fail(op, "row length " + std::to_string(d) + " needs torch's cross-warp split, not supported");
  }
  return static_cast<int>(bw);
}

}  // namespace mi::kernels::detail
