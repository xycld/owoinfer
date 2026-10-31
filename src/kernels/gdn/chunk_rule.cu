// Chunked gated delta rule (prefill) and its token-by-token oracle (see gdn.h).
//
// Same algorithm as transformers' torch_chunk_gated_delta_rule / flash-linear-attention's chunk_gated_delta_rule,
// split in two launches:
//   prep  (one block per chunk and head, fully parallel): L2-normalise q/k, cumulative decay, solve the unit lower
//         triangular UT system by forward substitution for u = T^-1 (beta v) and w = T^-1 (beta k exp(gcum)), and the
//         decayed q/k and the masked intra-chunk q k^T, all written to the fp32 workspace.
//   state (one block per head and 32-wide value slice, sequential over chunks): v_new = u - w S,
//         o = (q exp(gcum)) S + attn v_new, S = S exp(glast) + (k exp(glast - gcum))^T v_new.
// Everything is fp32 FFMA, as in the reference (which runs this whole function in fp32 with TF32 matmuls off); at
// prompt lengths it is a small fraction of the text encoder, so tensor-core error is not worth the risk here.
#include <cuda_bf16.h>

#include <mutex>
#include <set>
#include <type_traits>
#include <utility>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "kernels/gdn/checks.h"
#include "kernels/gdn/gdn.h"
#include "kernels/sm75/sm75.h"

namespace mi::gdn {
namespace {

using detail::check_aligned;
using detail::check_cu_seqlens;
using detail::check_rows;
using detail::require;

constexpr float kQScale = 0.08838834764831845f;  // 128 ** -0.5, the chunk fallback's `query * scaling`
constexpr float kSqrtHeadDim = 11.313708498984761f;  // the recurrent fallback divides by sqrt(128) instead
constexpr float kNormEps = 1e-6f;                // l2norm eps (FLA-aligned)
constexpr int kThreads = 256;
constexpr int kLd = kHeadDim + 4;  // smem row stride: 16-byte aligned rows, conflict-free float4 column reads
constexpr int kBV = 32;            // value columns per state block
constexpr int kLdV = kBV + 4;
constexpr int kLdA = kChunk + 4;

struct Workspace {
  float* u;     // [32][T][128] T^-1 (beta v)
  float* w;     // [32][T][128] T^-1 (beta k exp(gcum))
  float* qd;    // [32][T][128] q exp(gcum)            (q normalised and scaled)
  float* kd;    // [32][T][128] k exp(glast - gcum)    (k normalised)
  float* attn;  // [32][T][64]  (q k^T) exp(gcum_i - gcum_j) for j <= i, 0 above
  float* gcum;  // [32][T]      cumulative log decay within the chunk
};

constexpr std::size_t align_up(std::size_t v) { return (v + 255) & ~std::size_t{255}; }

std::size_t workspace_layout(std::int64_t tokens, void* base, Workspace* ws) {
  const auto t = static_cast<std::size_t>(tokens);
  const std::size_t sizes[6] = {
      kNumVHeads * t * kHeadDim, kNumVHeads * t * kHeadDim, kNumVHeads * t * kHeadDim,
      kNumVHeads * t * kHeadDim, kNumVHeads * t * kChunk,   kNumVHeads * t};
  float** slots[6] = {&ws->u, &ws->w, &ws->qd, &ws->kd, &ws->attn, &ws->gcum};
  std::size_t off = 0;
  for (int i = 0; i < 6; ++i) {
    if (base != nullptr) *slots[i] = reinterpret_cast<float*>(static_cast<char*>(base) + off);
    off += align_up(sizes[i] * sizeof(float));
  }
  return off;
}

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ void load4(const __nv_bfloat16* p, float v[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = bf(h[i]);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float dot4(float4 a, float4 b, float acc) {
  acc = fmaf(a.x, b.x, acc);
  acc = fmaf(a.y, b.y, acc);
  acc = fmaf(a.z, b.z, acc);
  return fmaf(a.w, b.w, acc);
}

__device__ __forceinline__ void axpy4(float a, float4 b, float acc[4]) {
  acc[0] = fmaf(a, b.x, acc[0]);
  acc[1] = fmaf(a, b.y, acc[1]);
  acc[2] = fmaf(a, b.z, acc[2]);
  acc[3] = fmaf(a, b.w, acc[3]);
}

struct PrepSmem {
  float q[kChunk][kLd];
  float k[kChunk][kLd];
  float a[kChunk][kChunk];  // strictly lower part of the UT system (beta_i k_i.k_j exp(gcum_i - gcum_j))
  float gc[kChunk], bt[kChunk], eg[kChunk], el[kChunk];
};

__global__ void __launch_bounds__(kThreads) prep_kernel(const __nv_bfloat16* __restrict__ qkv, std::int64_t qkv_stride,
                                                        const float* __restrict__ g,
                                                        const __nv_bfloat16* __restrict__ beta,
                                                        const int* __restrict__ cu, std::int64_t tokens, Workspace ws) {
  extern __shared__ float4 smem_raw[];
  PrepSmem& sm = *reinterpret_cast<PrepSmem*>(smem_raw);
  const int h = blockIdx.y, kh = h / (kNumVHeads / kNumKHeads), seq = blockIdx.z;
  const int start = cu[seq], len = cu[seq + 1] - start;
  if (static_cast<int>(blockIdx.x) * kChunk >= len) return;
  const int t0 = start + static_cast<int>(blockIdx.x) * kChunk;
  const int n = min(kChunk, len - static_cast<int>(blockIdx.x) * kChunk);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // q and k rows in fp32: x * rsqrt(sum(x^2) + eps), then q * scaling. Padding rows are zero.
  for (int i = warp; i < kChunk; i += kThreads / 32) {
    float qv[4] = {0.f, 0.f, 0.f, 0.f}, kv[4] = {0.f, 0.f, 0.f, 0.f};
    if (i < n) {
      const __nv_bfloat16* row = qkv + (t0 + i) * qkv_stride + kh * kHeadDim + lane * 4;
      load4(row, qv);
      load4(row + kKeyDim, kv);
      float sq = 0.f, sk = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        sq = fmaf(qv[e], qv[e], sq);
        sk = fmaf(kv[e], kv[e], sk);
      }
      const float iq = rsqrtf(warp_sum(sq) + kNormEps), ik = rsqrtf(warp_sum(sk) + kNormEps);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        qv[e] = qv[e] * iq * kQScale;
        kv[e] = kv[e] * ik;
      }
    }
    *reinterpret_cast<float4*>(&sm.q[i][lane * 4]) = make_float4(qv[0], qv[1], qv[2], qv[3]);
    *reinterpret_cast<float4*>(&sm.k[i][lane * 4]) = make_float4(kv[0], kv[1], kv[2], kv[3]);
  }
  if (tid < kChunk) {
    const bool valid = tid < n;
    sm.bt[tid] = valid ? bf(beta[(t0 + tid) * kNumVHeads + h]) : 0.f;
    sm.gc[tid] = valid ? g[(t0 + tid) * kNumVHeads + h] : 0.f;  // padded decay is 0 in log space
  }
  __syncthreads();
  if (tid == 0) {
    float acc = 0.f;
    for (int i = 0; i < kChunk; ++i) sm.gc[i] = acc += sm.gc[i];
  }
  __syncthreads();
  if (tid < kChunk) {
    sm.eg[tid] = expf(sm.gc[tid]);
    sm.el[tid] = expf(sm.gc[n - 1] - sm.gc[tid]);
    if (tid < n) ws.gcum[h * tokens + t0 + tid] = sm.gc[tid];
  }

  // k k^T and q k^T over the chunk: each thread owns rows ty + 16r and columns tx + 16c.
  {
    const int ty = tid >> 4, tx = tid & 15;
    float kk[4][4] = {}, qk[4][4] = {};
#pragma unroll 4
    for (int d = 0; d < kHeadDim; d += 4) {
      float4 qi[4], ki[4], kj[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        qi[r] = *reinterpret_cast<const float4*>(&sm.q[ty + 16 * r][d]);
        ki[r] = *reinterpret_cast<const float4*>(&sm.k[ty + 16 * r][d]);
        kj[r] = *reinterpret_cast<const float4*>(&sm.k[tx + 16 * r][d]);
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          kk[r][c] = dot4(ki[r], kj[c], kk[r][c]);
          qk[r][c] = dot4(qi[r], kj[c], qk[r][c]);
        }
      }
    }
    __syncthreads();  // gc, eg, el visible
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = ty + 16 * r;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int j = tx + 16 * c;
        const float decay = j <= i ? expf(sm.gc[i] - sm.gc[j]) : 0.f;
        sm.a[i][j] = j < i ? kk[r][c] * sm.bt[i] * decay : 0.f;
        if (i < n) ws.attn[(h * tokens + t0 + i) * kChunk + j] = qk[r][c] * decay;
      }
    }
  }

  // Decayed q and k for the state pass.
  for (int idx = tid; idx < kChunk * (kHeadDim / 4); idx += kThreads) {
    const int i = idx / (kHeadDim / 4), d = (idx % (kHeadDim / 4)) * 4;
    if (i >= n) continue;
    const float4 q = *reinterpret_cast<const float4*>(&sm.q[i][d]);
    const float4 k = *reinterpret_cast<const float4*>(&sm.k[i][d]);
    const float eg = sm.eg[i], el = sm.el[i];
    const std::int64_t o = (h * tokens + t0 + i) * kHeadDim + d;
    *reinterpret_cast<float4*>(ws.qd + o) = make_float4(q.x * eg, q.y * eg, q.z * eg, q.w * eg);
    *reinterpret_cast<float4*>(ws.kd + o) = make_float4(k.x * el, k.y * el, k.z * el, k.w * el);
  }
  __syncthreads();  // a complete

  // Forward substitution, one right-hand-side column per thread (columns 0..127: beta v -> u, 128..255:
  // beta k exp(gcum) -> w): x_i = b_i - sum_{j<i} a_ij x_j. a is zero on and above the diagonal, so the float4
  // groups may run past j = i - 1.
  const int col = tid & (kHeadDim - 1);
  const bool is_u = tid < kHeadDim;
  float x[kChunk];
  if (is_u) {
    const __nv_bfloat16* v = qkv + 2 * kKeyDim + h * kHeadDim + col;
#pragma unroll
    for (int i = 0; i < kChunk; ++i) x[i] = i < n ? bf(v[(t0 + i) * qkv_stride]) * sm.bt[i] : 0.f;
  } else {
#pragma unroll
    for (int i = 0; i < kChunk; ++i) x[i] = sm.k[i][col] * sm.bt[i] * sm.eg[i];
  }
#pragma unroll
  for (int i = 1; i < kChunk; ++i) {
    float acc = x[i];
#pragma unroll
    for (int j = 0; j < i; j += 4) {
      const float4 a = *reinterpret_cast<const float4*>(&sm.a[i][j]);
      acc = fmaf(-a.x, x[j], acc);
      acc = fmaf(-a.y, x[j + 1], acc);
      acc = fmaf(-a.z, x[j + 2], acc);
      acc = fmaf(-a.w, x[j + 3], acc);
    }
    x[i] = acc;
  }
  float* dst = (is_u ? ws.u : ws.w) + (h * tokens + t0) * kHeadDim + col;
#pragma unroll
  for (int i = 0; i < kChunk; ++i) {
    if (i < n) dst[i * kHeadDim] = x[i];
  }
}

struct StateSmem {
  float buf[2 * kChunk][kLd];  // [w; qd] rows, then reused for attn [64][kLdA] followed by kd [64][kLd]
  float s[kHeadDim][kLdV];     // state slice S[:, vb*32 : vb*32+32]
  float vn[kChunk][kLdV];      // v_new slice
};

__global__ void __launch_bounds__(kThreads) state_kernel(const int* __restrict__ cu, std::int64_t tokens, Workspace ws,
                                                         __nv_bfloat16* __restrict__ out,
                                                         float* __restrict__ final_state) {
  extern __shared__ float4 smem_raw[];
  StateSmem& sm = *reinterpret_cast<StateSmem*>(smem_raw);
  float(*attn)[kLdA] = reinterpret_cast<float(*)[kLdA]>(&sm.buf[0][0]);
  float(*kd)[kLd] = reinterpret_cast<float(*)[kLd]>(&sm.buf[0][0] + kChunk * kLdA);
  const int vb = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
  const int start = cu[seq], len = cu[seq + 1] - start;
  const int tid = threadIdx.x, rg = tid >> 3, c0 = (tid & 7) * 4, vc = vb * kBV + c0;
  const std::int64_t ht = h * tokens;

  // Thread-owned state elements: rows rg + 32 i, columns c0 .. c0 + 3 (also mirrored in smem for the GEMMs).
  float st[4][4] = {};
#pragma unroll
  for (int i = 0; i < 4; ++i) *reinterpret_cast<float4*>(&sm.s[rg + 32 * i][c0]) = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int c0tok = 0; c0tok < len; c0tok += kChunk) {
    const int t0 = start + c0tok, n = min(kChunk, len - c0tok);
#pragma unroll
    for (int idx = tid; idx < 2 * kChunk * (kHeadDim / 4); idx += kThreads) {
      const int row = idx / (kHeadDim / 4), d = (idx % (kHeadDim / 4)) * 4, r = row & (kChunk - 1);
      const float* src = (row < kChunk ? ws.w : ws.qd) + (ht + t0 + r) * kHeadDim + d;
      *reinterpret_cast<float4*>(&sm.buf[row][d]) =
          r < n ? *reinterpret_cast<const float4*>(src) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    // p = [w; qd] S: rows rg, rg+32 are w (-> v_new), rows rg+64, rg+96 are qd (-> inter-chunk output).
    float p[4][4] = {};
#pragma unroll 4
    for (int d = 0; d < kHeadDim; d += 4) {
      float4 a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = *reinterpret_cast<const float4*>(&sm.buf[rg + 32 * i][d]);
        b[i] = *reinterpret_cast<const float4*>(&sm.s[d + i][c0]);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        axpy4(a[i].x, b[0], p[i]);
        axpy4(a[i].y, b[1], p[i]);
        axpy4(a[i].z, b[2], p[i]);
        axpy4(a[i].w, b[3], p[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int row = rg + 32 * i;
      const float4 u = row < n ? *reinterpret_cast<const float4*>(ws.u + (ht + t0 + row) * kHeadDim + vc)
                               : make_float4(0.f, 0.f, 0.f, 0.f);
      *reinterpret_cast<float4*>(&sm.vn[row][c0]) =
          make_float4(u.x - p[i][0], u.y - p[i][1], u.z - p[i][2], u.w - p[i][3]);
    }
    __syncthreads();  // buf free, vn complete

#pragma unroll
    for (int idx = tid; idx < kChunk * (kChunk / 4); idx += kThreads) {
      const int row = idx / (kChunk / 4), d = (idx % (kChunk / 4)) * 4;
      *reinterpret_cast<float4*>(&attn[row][d]) =
          row < n ? *reinterpret_cast<const float4*>(ws.attn + (ht + t0 + row) * kChunk + d)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int idx = tid; idx < kChunk * (kHeadDim / 4); idx += kThreads) {
      const int row = idx / (kHeadDim / 4), d = (idx % (kHeadDim / 4)) * 4;
      *reinterpret_cast<float4*>(&kd[row][d]) =
          row < n ? *reinterpret_cast<const float4*>(ws.kd + (ht + t0 + row) * kHeadDim + d)
                  : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();

    // o = inter + attn v_new for rows rg and rg + 32.
#pragma unroll 4
    for (int j = 0; j < kChunk; j += 4) {
      const float4 a0 = *reinterpret_cast<const float4*>(&attn[rg][j]);
      const float4 a1 = *reinterpret_cast<const float4*>(&attn[rg + 32][j]);
      float4 b[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) b[e] = *reinterpret_cast<const float4*>(&sm.vn[j + e][c0]);
      axpy4(a0.x, b[0], p[2]);
      axpy4(a0.y, b[1], p[2]);
      axpy4(a0.z, b[2], p[2]);
      axpy4(a0.w, b[3], p[2]);
      axpy4(a1.x, b[0], p[3]);
      axpy4(a1.y, b[1], p[3]);
      axpy4(a1.z, b[2], p[3]);
      axpy4(a1.w, b[3], p[3]);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int row = rg + 32 * i;
      if (row >= n) continue;
      __nv_bfloat16 o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) o[e] = __float2bfloat16_rn(p[2 + i][e]);
      *reinterpret_cast<uint2*>(out + (static_cast<std::int64_t>(t0 + row) * kNumVHeads + h) * kHeadDim + vc) =
          *reinterpret_cast<const uint2*>(o);
    }

    // S = S exp(glast) + kd^T v_new.
    const float cd = expf(ws.gcum[ht + t0 + n - 1]);
    float upd[4][4] = {};
#pragma unroll 4
    for (int j = 0; j < kChunk; ++j) {
      const float4 b = *reinterpret_cast<const float4*>(&sm.vn[j][c0]);
#pragma unroll
      for (int i = 0; i < 4; ++i) axpy4(kd[j][rg + 32 * i], b, upd[i]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int e = 0; e < 4; ++e) st[i][e] = st[i][e] * cd + upd[i][e];
      *reinterpret_cast<float4*>(&sm.s[rg + 32 * i][c0]) = make_float4(st[i][0], st[i][1], st[i][2], st[i][3]);
    }
    __syncthreads();  // buf and vn are rewritten by the next chunk
  }

  if (final_state != nullptr) {
    float* fs = final_state + (static_cast<std::int64_t>(seq) * kNumVHeads + h) * kHeadDim * kHeadDim;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<float4*>(fs + (rg + 32 * i) * kHeadDim + vc) =
          make_float4(st[i][0], st[i][1], st[i][2], st[i][3]);
    }
  }
}

// Oracle: one block per (head, sequence), thread c owns state column S[:, c] (smem, [k][v]).
__global__ void __launch_bounds__(kHeadDim) recurrent_kernel(const __nv_bfloat16* __restrict__ qkv,
                                                             std::int64_t qkv_stride, const float* __restrict__ g,
                                                             const __nv_bfloat16* __restrict__ beta,
                                                             const int* __restrict__ cu, float* __restrict__ out,
                                                             float* __restrict__ final_state) {
  extern __shared__ float4 smem_raw[];
  float* s = reinterpret_cast<float*>(smem_raw);
  __shared__ float sq[kHeadDim], sk[kHeadDim], red[2][kHeadDim / 32];
  const int h = blockIdx.x, kh = h / (kNumVHeads / kNumKHeads), seq = blockIdx.y, c = threadIdx.x;
  const int lane = c & 31, warp = c >> 5;
  for (int k = 0; k < kHeadDim; ++k) s[k * kHeadDim + c] = 0.f;
  for (int t = cu[seq]; t < cu[seq + 1]; ++t) {
    const __nv_bfloat16* row = qkv + t * qkv_stride;
    const float q = bf(row[kh * kHeadDim + c]), k = bf(row[kKeyDim + kh * kHeadDim + c]);
    const float v = bf(row[2 * kKeyDim + h * kHeadDim + c]);
    const float qq = warp_sum(q * q), kk = warp_sum(k * k);
    if (lane == 0) {
      red[0][warp] = qq;
      red[1][warp] = kk;
    }
    __syncthreads();
    float qs = 0.f, ks = 0.f;
    for (int w = 0; w < kHeadDim / 32; ++w) {
      qs += red[0][w];
      ks += red[1][w];
    }
    sq[c] = q * rsqrtf(qs + kNormEps) / kSqrtHeadDim;
    sk[c] = k * rsqrtf(ks + kNormEps);
    __syncthreads();
    const float decay = expf(g[t * kNumVHeads + h]), b = bf(beta[t * kNumVHeads + h]);
    float kv = 0.f;
    for (int i = 0; i < kHeadDim; ++i) {
      const float sv = s[i * kHeadDim + c] * decay;
      s[i * kHeadDim + c] = sv;
      kv = fmaf(sv, sk[i], kv);
    }
    const float delta = (v - kv) * b;
    float o = 0.f;
    for (int i = 0; i < kHeadDim; ++i) {
      const float sv = fmaf(sk[i], delta, s[i * kHeadDim + c]);
      s[i * kHeadDim + c] = sv;
      o = fmaf(sv, sq[i], o);
    }
    out[(static_cast<std::int64_t>(t) * kNumVHeads + h) * kHeadDim + c] = o;
    __syncthreads();  // sq, sk, red are rewritten by the next token
  }
  if (final_state != nullptr) {
    float* fs = final_state + (static_cast<std::int64_t>(seq) * kNumVHeads + h) * kHeadDim * kHeadDim;
    for (int k = 0; k < kHeadDim; ++k) fs[k * kHeadDim + c] = s[k * kHeadDim + c];
  }
}

void check_rule_args(const Tensor& qkv, const Tensor& g, const Tensor& beta, const Tensor& cu_seqlens,
                     const Tensor& out, DType out_dtype, const Tensor* final_state) {
  const std::int64_t tokens = qkv.dim(0);
  check_rows(qkv, DType::kBF16, tokens, kConvDim, "qkv");
  require(qkv.stride[0] % 8 == 0, "qkv row stride must be a multiple of 8");
  check_aligned(qkv, 16, "qkv");
  check_aligned(out, 16, "out");
  check_rows(g, DType::kF32, tokens, kNumVHeads, "g");
  check_rows(beta, DType::kBF16, tokens, kNumVHeads, "beta");
  require(g.is_contiguous() && beta.is_contiguous(), "g and beta must be contiguous");
  check_cu_seqlens(cu_seqlens);
  require(out.dtype == out_dtype && out.device == Device::kCUDA && out.rank == 3 && out.dim(0) == tokens &&
              out.dim(1) == kNumVHeads && out.dim(2) == kHeadDim && out.is_contiguous(),
          "out must be contiguous cuda [T, 32, 128] " + std::string(dtype_name(out_dtype)) + ", got " + out.describe());
  if (final_state != nullptr) {
    const Tensor& fs = *final_state;
    require(fs.dtype == DType::kF32 && fs.device == Device::kCUDA && fs.rank == 4 &&
                fs.dim(0) == cu_seqlens.dim(0) - 1 && fs.dim(1) == kNumVHeads && fs.dim(2) == kHeadDim &&
                fs.dim(3) == kHeadDim && fs.is_contiguous(),
            "final_state must be contiguous cuda f32 [N, 32, 128, 128], got " + fs.describe());
  }
}

// cudaFuncSetAttribute is a driver call with a lock; do it once per kernel and device, not per launch.
template <typename Kernel>
void allow_dynamic_smem(Kernel kernel, int bytes) {
  static std::mutex mu;
  static std::set<std::pair<const void*, int>> done;
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  const std::pair<const void*, int> key{reinterpret_cast<const void*>(kernel), device};
  std::lock_guard<std::mutex> lock(mu);
  if (done.count(key) != 0) return;
  MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  done.insert(key);
}

// sm_75: the chunked kernel needs ~85 KiB of shared memory (PrepSmem), over Turing's 64 KiB per block. This per-token
// recurrent form computes the same rule (the chunked algorithm is an algebraic rearrangement of it) with the 128 state
// columns split across two launches of 64: every state column is independent, so the split is exact. 33 KiB of dynamic
// shared memory, 64 threads per block. Built in an MOSSINFER_SM75 build only (the dispatch below).
#if MOSSINFER_HAVE_SM75
constexpr int kSm75Cols = 64;

template <typename OutT>
__global__ void recurrent_sm75_kernel(const __nv_bfloat16* __restrict__ qkv, std::int64_t qkv_stride,
                                      const float* __restrict__ g, const __nv_bfloat16* __restrict__ beta,
                                      const int* __restrict__ cu, OutT* __restrict__ out,
                                      float* __restrict__ final_state) {
  extern __shared__ float smem_sm75[];
  float* s = smem_sm75;                       // [kHeadDim][kSm75Cols], state column c
  float* sq = s + kHeadDim * kSm75Cols;       // [kHeadDim]
  float* sk = sq + kHeadDim;                  // [kHeadDim]
  float* redq = sk + kHeadDim;                // [2]
  float* redk = redq + 2;                     // [2]
  const int h = blockIdx.x, kh = h / (kNumVHeads / kNumKHeads), seq = blockIdx.y;
  const int c = threadIdx.x;  // 0..63
  const int col = static_cast<int>(blockIdx.z) * kSm75Cols + c;
  const int lane = c & 31, warp = c >> 5;
  for (int i = 0; i < kHeadDim; ++i) s[i * kSm75Cols + c] = 0.f;
  for (int t = cu[seq]; t < cu[seq + 1]; ++t) {
    const __nv_bfloat16* row = qkv + t * qkv_stride;
    const float q0 = bf(row[kh * kHeadDim + c]);
    const float q1 = bf(row[kh * kHeadDim + kSm75Cols + c]);
    const float k0 = bf(row[kKeyDim + kh * kHeadDim + c]);
    const float k1 = bf(row[kKeyDim + kh * kHeadDim + kSm75Cols + c]);
    float qq = q0 * q0 + q1 * q1, kk = k0 * k0 + k1 * k1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      qq += __shfl_xor_sync(0xffffffffu, qq, off);
      kk += __shfl_xor_sync(0xffffffffu, kk, off);
    }
    if (lane == 0) {
      redq[warp] = qq;
      redk[warp] = kk;
    }
    __syncthreads();
    const float qs = redq[0] + redq[1], ks = redk[0] + redk[1];
    sq[c] = q0 * rsqrtf(qs + kNormEps) / kSqrtHeadDim;
    sq[kSm75Cols + c] = q1 * rsqrtf(qs + kNormEps) / kSqrtHeadDim;
    sk[c] = k0 * rsqrtf(ks + kNormEps);
    sk[kSm75Cols + c] = k1 * rsqrtf(ks + kNormEps);
    const float v = bf(row[2 * kKeyDim + h * kHeadDim + col]);
    __syncthreads();
    const float decay = expf(g[t * kNumVHeads + h]), b = bf(beta[t * kNumVHeads + h]);
    float kv = 0.f;
    for (int i = 0; i < kHeadDim; ++i) {
      const float sv = s[i * kSm75Cols + c] * decay;
      s[i * kSm75Cols + c] = sv;
      kv = fmaf(sv, sk[i], kv);
    }
    const float delta = (v - kv) * b;
    float o = 0.f;
    for (int i = 0; i < kHeadDim; ++i) {
      const float sv = fmaf(sk[i], delta, s[i * kSm75Cols + c]);
      s[i * kSm75Cols + c] = sv;
      o = fmaf(sv, sq[i], o);
    }
    if constexpr (std::is_same_v<OutT, float>) {
      out[(static_cast<std::int64_t>(t) * kNumVHeads + h) * kHeadDim + col] = o;
    } else {
      out[(static_cast<std::int64_t>(t) * kNumVHeads + h) * kHeadDim + col] = __float2bfloat16_rn(o);
    }
    __syncthreads();  // sq, sk and the reductions are rewritten by the next token
  }
  if (final_state != nullptr) {
    float* fs = final_state + (static_cast<std::int64_t>(seq) * kNumVHeads + h) * kHeadDim * kHeadDim;
    for (int i = 0; i < kHeadDim; ++i) fs[i * kHeadDim + col] = s[i * kSm75Cols + c];
  }
}
#endif

}  // namespace

std::size_t chunk_gated_delta_rule_workspace_bytes(std::int64_t total_tokens) {
  Workspace ws{};
  return workspace_layout(total_tokens, nullptr, &ws);
}

void chunk_gated_delta_rule(const Tensor& qkv, const Tensor& g, const Tensor& beta, const Tensor& cu_seqlens,
                            int max_seqlen, const Tensor& out, const Tensor* final_state, void* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream) {
  check_rule_args(qkv, g, beta, cu_seqlens, out, DType::kBF16, final_state);
  const std::int64_t tokens = qkv.dim(0);
  const std::size_t need = chunk_gated_delta_rule_workspace_bytes(tokens);
  require(workspace != nullptr && workspace_bytes >= need, "workspace too small: need " + std::to_string(need) +
                                                               " bytes, got " + std::to_string(workspace_bytes));
  require(reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0, "workspace must be 16-byte aligned");
  require(max_seqlen >= 0 && max_seqlen <= tokens, "max_seqlen out of range");
  if (varlen_debug_checks()) validate_cu_seqlens(cu_seqlens, max_seqlen, tokens, stream, "gdn::chunk_gated_delta_rule");
  const int num_seqs = static_cast<int>(cu_seqlens.dim(0) - 1);
  if (tokens == 0 || max_seqlen == 0) {
    if (final_state != nullptr) MI_CUDA_CHECK(cudaMemsetAsync(final_state->data, 0, final_state->nbytes(), stream));
    return;
  }
  Workspace ws{};
  workspace_layout(tokens, workspace, &ws);

  // sm_75: the chunked kernels need ~85 KiB of shared memory, over Turing's 64 KiB per block. The recurrent form above
  // computes the same rule with 33 KiB; its numerics are fp32 like the chunked path's. Only an MOSSINFER_SM75 build
  // checks (once per device, kernels::sm75::turing()); without it the probe never accepts a 7.x device.
#if MOSSINFER_HAVE_SM75
  if (kernels::sm75::turing()) {
    constexpr int kSm75Smem = (kHeadDim * kSm75Cols + 2 * kHeadDim + 4) * static_cast<int>(sizeof(float));
    const dim3 grid(kNumVHeads, num_seqs, kHeadDim / kSm75Cols);
    recurrent_sm75_kernel<__nv_bfloat16><<<grid, kSm75Cols, kSm75Smem, stream>>>(
        qkv.ptr<__nv_bfloat16>(), qkv.stride[0], g.ptr<float>(), beta.ptr<__nv_bfloat16>(), cu_seqlens.ptr<int>(),
        out.ptr<__nv_bfloat16>(), final_state ? final_state->ptr<float>() : nullptr);
    MI_CUDA_CHECK_LAUNCH();
    return;
  }
#endif

  constexpr int kPrepSmem = sizeof(PrepSmem), kStateSmem = sizeof(StateSmem);
  allow_dynamic_smem(prep_kernel, kPrepSmem);
  allow_dynamic_smem(state_kernel, kStateSmem);
  const dim3 prep_grid((max_seqlen + kChunk - 1) / kChunk, kNumVHeads, num_seqs);
  prep_kernel<<<prep_grid, kThreads, kPrepSmem, stream>>>(qkv.ptr<__nv_bfloat16>(), qkv.stride[0], g.ptr<float>(),
                                                          beta.ptr<__nv_bfloat16>(), cu_seqlens.ptr<int>(), tokens, ws);
  MI_CUDA_CHECK_LAUNCH();
  // state_kernel also writes the (zero) final state of empty sequences.
  const dim3 state_grid(kHeadDim / kBV, kNumVHeads, num_seqs);
  state_kernel<<<state_grid, kThreads, kStateSmem, stream>>>(cu_seqlens.ptr<int>(), tokens, ws,
                                                             out.ptr<__nv_bfloat16>(),
                                                             final_state ? final_state->ptr<float>() : nullptr);
  MI_CUDA_CHECK_LAUNCH();
}

void recurrent_gated_delta_rule_reference(const Tensor& qkv, const Tensor& g, const Tensor& beta,
                                          const Tensor& cu_seqlens, const Tensor& out, const Tensor* final_state,
                                          cudaStream_t stream) {
  check_rule_args(qkv, g, beta, cu_seqlens, out, DType::kF32, final_state);
  const int num_seqs = static_cast<int>(cu_seqlens.dim(0) - 1);
  constexpr int kSmem = kHeadDim * kHeadDim * sizeof(float);
  allow_dynamic_smem(recurrent_kernel, kSmem);
  recurrent_kernel<<<dim3(kNumVHeads, num_seqs), kHeadDim, kSmem, stream>>>(
      qkv.ptr<__nv_bfloat16>(), qkv.stride[0], g.ptr<float>(), beta.ptr<__nv_bfloat16>(), cu_seqlens.ptr<int>(),
      out.ptr<float>(), final_state ? final_state->ptr<float>() : nullptr);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::gdn
