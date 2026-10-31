// Exact attention on sm_75: a tiled FlashAttention-2-style kernel written against Turing's fp16 tensor-core MMA
// (mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32). The vendored FlashAttention-2 has an sm_75 trait branch, but its
// kernel only produces correct results on the sm_80 MMA layouts (upstream FA2 targets Ampere+), so the sm_75 image is
// this kernel instead.
//
// Algorithm: one CTA per (sequence, head, 64-query-row block), 4 warps, each warp owning 16 query rows and the full head
// dim. Keys are walked in tiles (32 for head_dim 128, 16 for head_dim 256) with an online softmax: per-thread fp32
// scores in the m16n8k8 accumulator layout, row max/sum reduced across the four lanes of a row group, and the P·V
// product reusing the normalized scores directly as the A operand (the m16n8k8 accumulator layout of Q·Kᵀ is exactly
// the A-operand layout of P·V for the next MMA). GQA maps query head h to KV head h / (heads / kv_heads); sequences are
// varlen through cu_seqlens.
//
// Precision: the engine's tensors are bf16 and sm_75's MMA is fp16. bf16 -> fp16 is exact for values in fp16's normal
// range and fp16 -> bf16 recovers the original value, so q/k/v are converted in place before the launch and restored
// after it; `out` is written as fp16 by the kernel and converted to bf16 in place.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

constexpr int kThreads = 128;  // 4 warps
constexpr int kWarps = 4;
constexpr int kBM = 64;  // query rows per CTA (16 per warp)
constexpr float kLog2e = 1.4426950408889634f;

// m16n8k8 fp16 -> fp32 MMA. a = {row group, row group + 8} 2-half registers, b = 2-half register, d = 4 fp32.
__device__ __forceinline__ void mma_m16n8k8(float (&d)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(b));
}

__device__ __forceinline__ std::uint32_t pack_half2(float lo, float hi) {
  union {
    __half2 h;
    std::uint32_t u;
  } c;
  c.h = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
  return c.u;
}

__device__ __forceinline__ std::uint32_t load_u32(const void* p) {
  return *static_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ void store_u32(void* p, std::uint32_t v) {
  *static_cast<std::uint32_t*>(p) = v;
}

struct Params {
  const __half* q = nullptr;
  const __half* k = nullptr;
  const __half* v = nullptr;
  __half* o = nullptr;
  float* lse = nullptr;
  std::int64_t q_row = 0, q_head = 0;
  std::int64_t k_row = 0, k_head = 0;
  std::int64_t v_row = 0, v_head = 0;
  std::int64_t o_row = 0, o_head = 0;
  const int* cu_q = nullptr;
  const int* cu_k = nullptr;
  int heads = 0, kv_heads = 0, total_q = 0;
  float scale = 1.f;
  bool causal = false;
};

template <int D, int BN>
__global__ void attention_kernel_sm75(const Params p) {
  constexpr int kQStride = D + 8;
  constexpr int kVStride = BN + 8;
  constexpr int kSubtiles = BN / 8;
  constexpr int kDimChunks = D / 8;

  extern __shared__ __align__(16) char smem_raw[];
  __half* qs = reinterpret_cast<__half*>(smem_raw);                 // [kBM][D]
  __half* ks = qs + kBM * kQStride;                                  // [BN][D]
  __half* vt = ks + BN * kQStride;                                   // [D][BN]

  const int batch = blockIdx.y;
  const int head = blockIdx.z;
  const int q_start = p.cu_q[batch];      // token offset of this sequence in q
  const int q_end = p.cu_q[batch + 1];
  const int k_start = p.cu_k[batch];
  const int k_end = p.cu_k[batch + 1];
  const int len_q = q_end - q_start;
  const int len_k = k_end - k_start;
  const int row_base = static_cast<int>(blockIdx.x) * kBM;
  if (row_base >= len_q) return;
  const int rows = min(kBM, len_q - row_base);
  const int kv_head = head / (p.heads / p.kv_heads);
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int group = lane >> 2;  // 0..7: the row of the 8x8 fragment
  const int tig = lane & 3;     // 0..3: two consecutive columns per lane

  // Q tile into shared memory (rows past this sequence's end are zero).
  for (int idx = tid; idx < kBM * D; idx += kThreads) {
    const int r = idx / D, d = idx % D;
    qs[r * kQStride + d] = (r < rows) ? p.q[(q_start + row_base + r) * p.q_row + head * p.q_head + d]
                                      : __float2half_rn(0.f);
  }
  __syncthreads();

  // This warp's Q fragment: rows {warp*16 + group, +8}, dims {i*8 + 2*tig, +1}.
  std::uint32_t qf[kDimChunks][2];
  const int r0 = warp * 16 + group;
  const int r1 = r0 + 8;
#pragma unroll
  for (int i = 0; i < kDimChunks; ++i) {
    qf[i][0] = load_u32(&qs[r0 * kQStride + i * 8 + tig * 2]);
    qf[i][1] = load_u32(&qs[r1 * kQStride + i * 8 + tig * 2]);
  }

  float o[kDimChunks][4];  // P·V accumulators, 4 floats per 8-dim chunk
#pragma unroll
  for (int i = 0; i < kDimChunks; ++i) {
    o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  }
  float m0 = -1e30f, m1 = -1e30f;  // running row max for r0 / r1
  float l0 = 0.f, l1 = 0.f;        // running row sum

  const int key_tiles = p.causal ? min((len_k + BN - 1) / BN, (row_base + rows + BN - 1) / BN)
                                 : (len_k + BN - 1) / BN;
  for (int tile = 0; tile < key_tiles; ++tile) {
    const int j0 = tile * BN;
    // K tile and transposed V tile into shared memory.
    for (int idx = tid; idx < BN * D; idx += kThreads) {
      const int j = idx / D, d = idx % D;
      const int token = j0 + j;
      const bool valid = token < len_k;
      ks[j * kQStride + d] = valid ? p.k[(k_start + token) * p.k_row + kv_head * p.k_head + d] : __float2half_rn(0.f);
      vt[d * kVStride + j] = valid ? p.v[(k_start + token) * p.v_row + kv_head * p.v_head + d] : __float2half_rn(0.f);
    }
    __syncthreads();

    // S = Q·Kᵀ for every 8-key subtile of this tile (fp32, scaled and masked below).
    float s[kSubtiles][4];
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
#pragma unroll
      for (int i = 0; i < kDimChunks; ++i) {
        const std::uint32_t b = load_u32(&ks[(t * 8 + group) * kQStride + i * 8 + tig * 2]);
        mma_m16n8k8(s[t], qf[i][0], qf[i][1], b);
      }
    }

    // Scale, mask (keys past the sequence, causal), row max across the tile and the four lanes of the row group.
    const int row_q0 = row_base + r0;
    const int row_q1 = row_base + r1;
    float tile_max0 = -1e30f, tile_max1 = -1e30f;
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      const int key0 = j0 + t * 8 + tig * 2;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = key0 + (e & 1);
        const int row_q = (e < 2) ? row_q0 : row_q1;
        const bool valid = key < len_k && (!p.causal || key <= row_q);
        s[t][e] = valid ? s[t][e] * p.scale : -1e30f;
      }
      tile_max0 = fmaxf(tile_max0, fmaxf(s[t][0], s[t][1]));
      tile_max1 = fmaxf(tile_max1, fmaxf(s[t][2], s[t][3]));
    }
    tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, 1));
    tile_max0 = fmaxf(tile_max0, __shfl_xor_sync(0xffffffffu, tile_max0, 2));
    tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, 1));
    tile_max1 = fmaxf(tile_max1, __shfl_xor_sync(0xffffffffu, tile_max1, 2));

    const float m_new0 = fmaxf(m0, tile_max0);
    const float m_new1 = fmaxf(m1, tile_max1);
    const float alpha0 = (m0 <= -1e29f) ? 0.f : exp2f((m0 - m_new0) * kLog2e);
    const float alpha1 = (m1 <= -1e29f) ? 0.f : exp2f((m1 - m_new1) * kLog2e);

    // P = exp(S - m_new) as fp16, and the running sum.
    float tile_sum0 = 0.f, tile_sum1 = 0.f;
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      const float p0 = exp2f((s[t][0] - m_new0) * kLog2e);
      const float p1 = exp2f((s[t][1] - m_new0) * kLog2e);
      const float p2 = exp2f((s[t][2] - m_new1) * kLog2e);
      const float p3 = exp2f((s[t][3] - m_new1) * kLog2e);
      tile_sum0 += p0 + p1;
      tile_sum1 += p2 + p3;
      s[t][0] = p0;
      s[t][1] = p1;
      s[t][2] = p2;
      s[t][3] = p3;
    }
    tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 1);
    tile_sum0 += __shfl_xor_sync(0xffffffffu, tile_sum0, 2);
    tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 1);
    tile_sum1 += __shfl_xor_sync(0xffffffffu, tile_sum1, 2);
    l0 = l0 * alpha0 + tile_sum0;
    l1 = l1 * alpha1 + tile_sum1;
    m0 = m_new0;
    m1 = m_new1;

    // O = O * alpha + P·V.
#pragma unroll
    for (int i = 0; i < kDimChunks; ++i) {
      o[i][0] *= alpha0;
      o[i][1] *= alpha0;
      o[i][2] *= alpha1;
      o[i][3] *= alpha1;
    }
#pragma unroll
    for (int t = 0; t < kSubtiles; ++t) {
      const std::uint32_t pa0 = pack_half2(s[t][0], s[t][1]);
      const std::uint32_t pa1 = pack_half2(s[t][2], s[t][3]);
#pragma unroll
      for (int i = 0; i < kDimChunks; ++i) {
        const std::uint32_t b = load_u32(&vt[(i * 8 + group) * kVStride + t * 8 + tig * 2]);
        mma_m16n8k8(o[i], pa0, pa1, b);
      }
    }
    __syncthreads();
  }

  // Normalize, then out (fp16) and the log-sum-exp workspace (fp32 [heads, total_q], unpadded).
  const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f;
  const float inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
  if (r0 < rows) {
#pragma unroll
    for (int i = 0; i < kDimChunks; ++i) {
      store_u32(&p.o[(q_start + row_base + r0) * p.o_row + head * p.o_head + i * 8 + tig * 2],
                pack_half2(o[i][0] * inv0, o[i][1] * inv0));
    }
    if (tig == 0 && p.lse != nullptr) p.lse[static_cast<std::int64_t>(head) * p.total_q + q_start + row_base + r0] =
        m0 + logf(l0);
  }
  if (r1 < rows) {
#pragma unroll
    for (int i = 0; i < kDimChunks; ++i) {
      store_u32(&p.o[(q_start + row_base + r1) * p.o_row + head * p.o_head + i * 8 + tig * 2],
                pack_half2(o[i][2] * inv1, o[i][3] * inv1));
    }
    if (tig == 0 && p.lse != nullptr) p.lse[static_cast<std::int64_t>(head) * p.total_q + q_start + row_base + r1] =
        m1 + logf(l1);
  }
}

template <int D, int BN>
void launch(const Params& p, int max_seqlen, int batch, cudaStream_t stream) {
  constexpr int kQStride = D + 8;
  constexpr int kVStride = BN + 8;
  constexpr int kSmem = (kBM * kQStride + BN * kQStride + D * kVStride) * static_cast<int>(sizeof(__half));
  static_assert(kSmem <= 64 * 1024, "sm_75 allows 64 KiB of opt-in shared memory per block");
  auto kernel = &attention_kernel_sm75<D, BN>;
  if constexpr (kSmem > 48 * 1024) {
    MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  }
  const dim3 grid(static_cast<unsigned>((max_seqlen + kBM - 1) / kBM), static_cast<unsigned>(batch),
                  static_cast<unsigned>(p.heads));
  kernel<<<grid, kThreads, kSmem, stream>>>(p);
  MI_CUDA_CHECK_LAUNCH();
}

bool head_dim_supported(int d) { return d == 128 || d == 256; }

}  // namespace

void flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
               const Tensor& cu_seqlens_k, int max_seqlen, bool causal, float scale, void* workspace,
               std::size_t workspace_bytes, cudaStream_t stream) {
  const std::int64_t total_q = q.dim(0);
  if (total_q == 0) return;
  const std::int64_t heads = q.dim(1);
  const std::int64_t head_dim = q.dim(2);
  const std::int64_t kv_heads = k.dim(1);
  if (!head_dim_supported(static_cast<int>(head_dim))) {
    throw std::invalid_argument("sm75::flash_fwd: head_dim must be 128 or 256");
  }
  if ((head_dim == 128 && causal) || (head_dim == 256 && !causal)) {
    throw std::invalid_argument("sm75::flash_fwd: head_dim 128 is non-causal, 256 is causal");
  }
  if (workspace_bytes < static_cast<std::size_t>(total_q) * heads * sizeof(float)) {
    throw std::invalid_argument("sm75::flash_fwd: workspace too small for the log-sum-exp");
  }

  convert_tensor_bf16_to_f16_inplace(q, total_q, heads, head_dim, stream);
  convert_tensor_bf16_to_f16_inplace(k, k.dim(0), kv_heads, head_dim, stream);
  convert_tensor_bf16_to_f16_inplace(v, v.dim(0), kv_heads, head_dim, stream);

  Params p;
  p.q = static_cast<const __half*>(q.data);
  p.k = static_cast<const __half*>(k.data);
  p.v = static_cast<const __half*>(v.data);
  p.o = static_cast<__half*>(out.data);
  p.lse = static_cast<float*>(workspace);
  p.q_row = q.stride[0];
  p.q_head = q.stride[1];
  p.k_row = k.stride[0];
  p.k_head = k.stride[1];
  p.v_row = v.stride[0];
  p.v_head = v.stride[1];
  p.o_row = out.stride[0];
  p.o_head = out.stride[1];
  p.cu_q = cu_seqlens_q.ptr<int>();
  p.cu_k = cu_seqlens_k.ptr<int>();
  p.heads = static_cast<int>(heads);
  p.kv_heads = static_cast<int>(kv_heads);
  p.total_q = static_cast<int>(total_q);
  p.scale = scale;
  p.causal = causal;
  const int batch = static_cast<int>(cu_seqlens_q.dim(0) - 1);

  if (head_dim == 128) {
    launch<128, 32>(p, max_seqlen, batch, stream);
  } else {
    launch<256, 16>(p, max_seqlen, batch, stream);
  }

  convert_tensor_f16_to_bf16_inplace(out, out.dim(0), out.dim(1), out.dim(2), stream);
  convert_tensor_f16_to_bf16_inplace(q, total_q, heads, head_dim, stream);
  convert_tensor_f16_to_bf16_inplace(k, k.dim(0), kv_heads, head_dim, stream);
  convert_tensor_f16_to_bf16_inplace(v, v.dim(0), kv_heads, head_dim, stream);
}

}  // namespace mi::kernels::sm75
