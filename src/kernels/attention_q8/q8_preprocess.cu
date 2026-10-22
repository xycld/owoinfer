/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Derived from SageAttention csrc/fused/fused.cu (commit d1a57a5); MossInfer modifications (Apache-2.0 section 4(b))
// are listed below and in third_party/sageattention/VENDORED.md.
// Quantization pre-pass of attention_q8, varlen ports of SageAttention's csrc/fused/fused.cu (Apache-2.0,
// third_party/sageattention): QuantInt8Kernel (per-warp Q / per-block K INT8 with the K mean subtracted) and
// TransposePadPermuteKernel + MeanScaleKernel (per-channel e4m3 V, transposed, padded to CTA_K and permuted for the
// e4m3 MMA). Changes from upstream, all for varlen or robustness:
//   - every kernel reads its sequence's start and length from cu_seqlens; blocks are per sequence;
//   - the K mean is computed here (upstream: torch k.mean(), a bf16 tensor) from fp32 per-chunk partial sums reduced
//     in a fixed order, so the pass is deterministic, and rounded to bf16 as upstream's (smoothing is exact for any
//     constant; matching the rounding keeps the INT8 K values those of upstream);
//   - the V transpose and the e4m3 quantization are one kernel (no bf16 transposed intermediate), with the
//     per-channel amax from the same stats pass as the K sum;
//   - an all-zero V channel gets scale 1 (upstream divides by its amax 0 and writes NaN);
//   - the fp16 V for the fp16-PV kernels is converted inside the K quantization kernel;
//   - with debug checks the statistics pass also measures the fp16 variants' per-channel |V| max.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "core/cuda_check.h"
#include "kernels/attention_q8/q8_kernels.h"

// Vendored SageAttention device helpers (after the engine headers: they define global macros such as div_ceil).
#include "csrc/numeric_conversion.cuh"
#include "csrc/reduction_utils.cuh"

namespace mi::attention_q8::detail {
namespace {

constexpr int kPack = 8;                       // bf16 per 16-byte load
constexpr int kLanes = kHeadDim / kPack;       // threads per token row (16)

__device__ __forceinline__ void load_row8(const __nv_bfloat16* p, float* x) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 f = __bfloat1622float2(h[j]);
    x[2 * j] = f.x;
    x[2 * j + 1] = f.y;
  }
}

// Partial per-channel K sums and V |max| of one kStatsChunk-key chunk of one (sequence, kv head). Block: 16 token
// rows x 16 lanes of 8 channels. A chunk past the sequence end writes zeros, so the finalize pass can sum all chunks.
template <bool kAmax>
__global__ void __launch_bounds__(256) kv_stats_partial(PrepParams p) {
  const int c = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x % kLanes, row = threadIdx.x / kLanes;
  const int start = p.cu_k[b], len = p.cu_k[b + 1] - start;
  const int t_end = min(len, (c + 1) * kStatsChunk);
  float sum[kPack] = {}, amax[kPack] = {};
  const auto* k = static_cast<const __nv_bfloat16*>(p.k);
  const auto* v = static_cast<const __nv_bfloat16*>(p.v);
  for (int t = c * kStatsChunk + row; t < t_end; t += 16) {
    float x[kPack];
    load_row8(k + (start + t) * p.k_row + h * p.k_head + lane * kPack, x);
#pragma unroll
    for (int j = 0; j < kPack; ++j) sum[j] += x[j];
    if constexpr (kAmax) {
      load_row8(v + (start + t) * p.v_row + h * p.v_head + lane * kPack, x);
#pragma unroll
      for (int j = 0; j < kPack; ++j) amax[j] = fmaxf(amax[j], fabsf(x[j]));
    }
  }
  __shared__ float s_sum[16][kHeadDim];
  __shared__ float s_amax[16][kHeadDim];
#pragma unroll
  for (int j = 0; j < kPack; ++j) {
    s_sum[row][lane * kPack + j] = sum[j];
    if constexpr (kAmax) s_amax[row][lane * kPack + j] = amax[j];
  }
  __syncthreads();
  if (threadIdx.x < kHeadDim) {
    float s = 0, a = 0;
    for (int r = 0; r < 16; ++r) {
      s += s_sum[r][threadIdx.x];
      if constexpr (kAmax) a = fmaxf(a, s_amax[r][threadIdx.x]);
    }
    const std::int64_t at = ((static_cast<std::int64_t>(b) * p.kv_heads + h) * p.chunks + c) * kHeadDim + threadIdx.x;
    p.part_sum[at] = s;
    if constexpr (kAmax) p.part_amax[at] = a;
  }
}

// k_mean = bf16(sum / len), v_scale = amax / v_range and v_recp = v_range / amax (upstream's two roundings) per
// (sequence, kv head, channel), summing the chunks in order.
template <bool kAmax>
__global__ void kv_stats_finalize(PrepParams p) {
  const int h = blockIdx.x, b = blockIdx.y, ch = threadIdx.x;
  const int len = p.cu_k[b + 1] - p.cu_k[b];
  const std::int64_t base = (static_cast<std::int64_t>(b) * p.kv_heads + h) * p.chunks * kHeadDim + ch;
  float s = 0, a = 0;
  for (int c = 0; c < p.chunks; ++c) {
    s += p.part_sum[base + c * kHeadDim];
    if constexpr (kAmax) a = fmaxf(a, p.part_amax[base + c * kHeadDim]);
  }
  const std::int64_t at = (static_cast<std::int64_t>(b) * p.kv_heads + h) * kHeadDim + ch;
  p.k_mean[at] = len > 0 ? __bfloat162float(__float2bfloat16(s / static_cast<float>(len))) : 0.f;
  if constexpr (kAmax) {
    p.v_scale[at] = a > 0.f ? a / p.v_range : 1.f;
    p.v_recp[at] = a > 0.f ? p.v_range / a : 1.f;
  }
}

// QuantInt8Kernel (num_pack_per_thread = 1): kBlock token rows of one (sequence, head) -> int8 with one scale
// amax / 127 (amax floored at 1e-7). kSubMean: K smoothing (subtract the sequence's mean). kConvertV: also writes the
// block's V rows as fp16 (the fp16-PV kernels' V). A block past the sequence end still writes its scale.
template <int kBlock, bool kSubMean, bool kConvertV>
__global__ void __launch_bounds__(kBlock * kLanes) quant_int8(PrepParams p, const void* in_v, std::int64_t row_stride,
                                                              std::int64_t head_stride, const int* cu, int heads,
                                                              std::int8_t* out, float* scale, int blocks) {
  const int bx = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int lane = threadIdx.x % kLanes;
  const int start = cu[b], len = cu[b + 1] - start;
  const int token = bx * kBlock + static_cast<int>(threadIdx.x) / kLanes;
  const bool valid = token < len;
  const auto* in = static_cast<const __nv_bfloat16*>(in_v);
  float x[kPack] = {};
  if (valid) {
    load_row8(in + (start + token) * row_stride + h * head_stride + lane * kPack, x);
    if constexpr (kSubMean) {
      const float* mean = p.k_mean + (static_cast<std::int64_t>(b) * heads + h) * kHeadDim + lane * kPack;
#pragma unroll
      for (int j = 0; j < kPack; ++j) x[j] -= mean[j];
    }
  }
  float amax = 0.0000001f;
#pragma unroll
  for (int j = 0; j < kPack; ++j) amax = fmaxf(amax, fabsf(x[j]));
  __shared__ float s_amax;
  const float block_amax = vllm::blockReduceMax(amax);
  if (threadIdx.x == 0) {
    s_amax = block_amax;
    scale[(static_cast<std::int64_t>(b) * heads + h) * blocks + bx] = block_amax / 127.0f;
  }
  __syncthreads();
  if (!valid) return;
  const float q = 127.0f / s_amax;
  char4 o[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    o[j] = make_char4(float_to_int8_rn(x[4 * j] * q), float_to_int8_rn(x[4 * j + 1] * q),
                      float_to_int8_rn(x[4 * j + 2] * q), float_to_int8_rn(x[4 * j + 3] * q));
  }
  const std::int64_t row = start + token;
  *reinterpret_cast<float2*>(out + (row * heads + h) * kHeadDim + lane * kPack) = *reinterpret_cast<const float2*>(o);
  if constexpr (kConvertV) {
    float v[kPack];
    load_row8(static_cast<const __nv_bfloat16*>(p.v) + row * p.v_row + h * p.v_head + lane * kPack, v);
    __half2 hv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) hv[j] = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
    *reinterpret_cast<uint4*>(static_cast<__half*>(p.v_out) + (row * heads + h) * kHeadDim + lane * kPack) =
        *reinterpret_cast<const uint4*>(hv);
  }
}

// TransposePadPermuteKernel + the quantizing half of MeanScaleKernel: one kCtaK-key tile of one (sequence, kv head)
// -> e4m3 V^T at [h][d][v_start + tile]. Keys past the sequence end are zero; within each 16-key group the stored
// order is 0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15 (upstream's smem row permutation).
__global__ void __launch_bounds__(kCtaK * kLanes) v_fp8_transpose_quant(PrepParams p) {
  const int bx = blockIdx.x, h = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
  const int start = p.cu_k[b], len = p.cu_k[b + 1] - start;
  if (bx * kCtaK >= len) return;
  int v_start = 0;
  for (int i = 0; i < b; ++i) v_start += (p.cu_k[i + 1] - p.cu_k[i] + kCtaK - 1) / kCtaK * kCtaK;

  __shared__ __nv_bfloat16 s_load[kCtaK][kHeadDim];
  __shared__ __nv_bfloat16 s_store[kHeadDim][kCtaK];
  const int r = t / kLanes, lane = t % kLanes;
  const int mod = r % 16;
  const int srow = (r / 16) * 16 + (mod / 8) * 2 + ((mod / 2) % 4) * 4 + (mod % 2);
  const int token = bx * kCtaK + r;
  uint4 raw = make_uint4(0, 0, 0, 0);
  if (token < len) {
    raw = *reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.v) + (start + token) * p.v_row +
                                          h * p.v_head + lane * kPack);
  }
  *reinterpret_cast<uint4*>(&s_load[srow][lane * kPack]) = raw;
  __syncthreads();
  const int row_base = t % kCtaK, col_base = t / kCtaK;
#pragma unroll
  for (int i = 0; i < 8; ++i) s_store[col_base + i * (kHeadDim / 8)][row_base] = s_load[row_base][col_base + i * (kHeadDim / 8)];
  __syncthreads();

  const int d = t / (kCtaK / kPack), seg = t % (kCtaK / kPack);
  const float recp = p.v_recp[(static_cast<std::int64_t>(b) * p.kv_heads + h) * kHeadDim + d];
  float x[kPack];
  load_row8(&s_store[d][seg * kPack], x);
#pragma unroll
  for (int j = 0; j < kPack; ++j) x[j] *= recp;
  uint32_t f8[2];
  floatx4_to_e4m3x4(f8, x, x + 2);
  floatx4_to_e4m3x4(f8 + 1, x + 4, x + 6);
  auto* out = static_cast<std::int8_t*>(p.v_out) + (static_cast<std::int64_t>(h) * kHeadDim + d) * p.padded_k + v_start +
              bx * kCtaK + seg * kPack;
  *reinterpret_cast<uint2*>(out) = make_uint2(f8[0], f8[1]);
}

}  // namespace

void preprocess(const PrepParams& p, cudaStream_t stream) {
  const dim3 stats_grid(p.chunks, p.kv_heads, p.batch);
  if (p.fp8 || p.v_amax) {
    kv_stats_partial<true><<<stats_grid, 256, 0, stream>>>(p);
    MI_CUDA_CHECK_LAUNCH();
    kv_stats_finalize<true><<<dim3(p.kv_heads, p.batch), kHeadDim, 0, stream>>>(p);
  } else {
    kv_stats_partial<false><<<stats_grid, 256, 0, stream>>>(p);
    MI_CUDA_CHECK_LAUNCH();
    kv_stats_finalize<false><<<dim3(p.kv_heads, p.batch), kHeadDim, 0, stream>>>(p);
  }
  MI_CUDA_CHECK_LAUNCH();

  // Q: per warp of rows, laid out for the attention kernel's CTA_Q / WARP_Q blocks of the longest sequence.
  const int q_blocks = (p.max_seqlen + kCtaQ - 1) / kCtaQ * (kCtaQ / p.warp_q);
  const dim3 q_grid(q_blocks, p.heads, p.batch);
  if (p.warp_q == 16) {
    quant_int8<16, false, false><<<q_grid, 16 * kLanes, 0, stream>>>(p, p.q, p.q_row, p.q_head, p.cu_q, p.heads, p.q_i8,
                                                                     p.q_scale, q_blocks);
  } else {
    quant_int8<32, false, false><<<q_grid, 32 * kLanes, 0, stream>>>(p, p.q, p.q_row, p.q_head, p.cu_q, p.heads, p.q_i8,
                                                                     p.q_scale, q_blocks);
  }
  MI_CUDA_CHECK_LAUNCH();

  // K: per kCtaK-key block, smoothed; V converted to fp16 alongside for the fp16-PV kernels.
  const int k_blocks = (p.max_seqlen + kCtaK - 1) / kCtaK;
  const dim3 k_grid(k_blocks, p.kv_heads, p.batch);
  if (p.fp8) {
    quant_int8<kCtaK, true, false><<<k_grid, kCtaK * kLanes, 0, stream>>>(p, p.k, p.k_row, p.k_head, p.cu_k, p.kv_heads,
                                                                          p.k_i8, p.k_scale, k_blocks);
    MI_CUDA_CHECK_LAUNCH();
    v_fp8_transpose_quant<<<k_grid, kCtaK * kLanes, 0, stream>>>(p);
  } else {
    quant_int8<kCtaK, true, true><<<k_grid, kCtaK * kLanes, 0, stream>>>(p, p.k, p.k_row, p.k_head, p.cu_k, p.kv_heads,
                                                                         p.k_i8, p.k_scale, k_blocks);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::attention_q8::detail
