// GroupNorm over NHWC activations with order-independent statistics: every element's x and x^2 are rounded once
// onto fixed-point grids and summed in integers, so the statistics of a tensor are bit-identical however it is
// split into tiles and in whatever order the tiles (and the threads within them) are accumulated. An exact tiled
// decoder relies on this: the global statistics of a layer come from its tiles, and the untiled decode computes
// the very same values. Then a deterministic fp64 finalize to fp32 {mean, rstd}, and an elementwise apply.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/vae/vae_kernels.h"

namespace mi::vae {
namespace {

constexpr int kSlotElems = 1 << 16;  // target elements per accumulation block
constexpr int kMaxChannels = 1024;

// Fixed-point grids: x on 2^-64 (signed int128 sums), x^2 on 2^-96 (unsigned 192-bit sums), each element rounded
// half up in magnitude. Elements must satisfy |x| < 2^33 (biased exponent <= 159) and a group may hold up to 2^30
// elements, which keeps both sums in range: |sum| < 2^30 * 2^97 = 2^127, sum_sq < 2^30 * 2^162 = 2^192. The grids
// resolve |x| down to 2^-64 and x^2 down to 2^-96, far below what fp32 statistics can see. Zeros and subnormals add
// nothing; out-of-range values (|x| >= 2^33, inf, NaN) are counted and turn the group's statistics into NaN.
constexpr int kMaxBiasedExp = 159;
constexpr std::int64_t kMaxGroupElems = std::int64_t{1} << 30;

using u64 = unsigned long long;

// Multi-word unsigned integers as explicit little-endian u64 words with manual carry/borrow (no compiler 128-bit
// type, which MSVC hosts lack). All arithmetic is modulo 2^(64 N), so the signed sum is plain two's complement.
template <int N>
struct Words {
  u64 w[N] = {};
};
using U128 = Words<2>;
using U192 = Words<3>;

template <int N>
__device__ __forceinline__ void add_words(Words<N>& a, const Words<N>& b) {
  u64 carry = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const u64 t = a.w[i] + carry;
    const u64 c = t < carry ? 1ull : 0ull;  // only when a = 2^64 - 1 and carry = 1: t = 0, so t + b cannot carry
    a.w[i] = t + b.w[i];
    carry = c | (a.w[i] < t ? 1ull : 0ull);
  }
}

template <int N>
__device__ __forceinline__ void sub_words(Words<N>& a, const Words<N>& b) {
  u64 borrow = 0;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const u64 t = a.w[i] - b.w[i];
    const u64 bo = a.w[i] < b.w[i] ? 1ull : 0ull;  // then t != 0, so t - borrow cannot borrow again
    a.w[i] = t - borrow;
    borrow = bo | (t < borrow ? 1ull : 0ull);
  }
}

template <int N>
__device__ __forceinline__ Words<N> negate_words(const Words<N>& a) {
  Words<N> r;
  sub_words(r, a);  // 0 - a
  return r;
}

// v << sh as an N-word value; the caller guarantees the result fits (no bits shifted past word N - 1).
template <int N>
__device__ __forceinline__ Words<N> shl_words(u64 v, int sh) {
  Words<N> r;
  const int wi = sh >> 6, b = sh & 63;
  const u64 lo = v << b, hi = b != 0 ? v >> (64 - b) : 0ull;
#pragma unroll
  for (int i = 0; i < N; ++i) r.w[i] = i == wi ? lo : (i == wi + 1 ? hi : 0ull);  // selects keep r in registers
  return r;
}

// Per (n, group) accumulator in global memory: sum (int128, two's complement), sum_sq (u192), out-of-range count.
struct Accum {
  u64 s_lo, s_hi, q0, q1, q2, bad;
};

struct Lane {  // one channel's running sums in a thread
  U128 s;
  U192 q;
  unsigned bad = 0;
};

__device__ __forceinline__ void accumulate(Lane& l, float v) {
  const unsigned bits = __float_as_uint(v);
  const int e = static_cast<int>((bits >> 23) & 0xffu);
  if (e == 0) return;  // zero or subnormal (|x| < 2^-126): below both grids
  if (e > kMaxBiasedExp) {
    l.bad = 1;
    return;
  }
  const u64 mant = (bits & 0x7fffffu) | 0x800000u;  // |x| = mant * 2^(e - 150)
  // |x| * 2^64 = mant * 2^(e - 86)
  const int sh = e - 86;  // <= 73
  U128 ax;
  if (sh >= 0) {
    ax = shl_words<2>(mant, sh);  // < 2^97
  } else {
    const int k = -sh;  // mant < 2^24: k > 24 rounds to 0
    ax.w[0] = k > 24 ? 0 : (mant + (1ull << (k - 1))) >> k;
  }
  if (bits >> 31) {
    sub_words(l.s, ax);
  } else {
    add_words(l.s, ax);
  }
  // x^2 * 2^96 = mant^2 * 2^(2e - 204), mant^2 < 2^48
  const u64 m2 = mant * mant;
  const int sh2 = 2 * e - 204;  // <= 114: the value stays below 2^162
  if (sh2 >= 0) {
    add_words(l.q, shl_words<3>(m2, sh2));
  } else {
    const int k = -sh2;  // m2 < 2^48: k > 48 rounds to 0
    if (k <= 48) {
      U192 t;
      t.w[0] = (m2 + (1ull << (k - 1))) >> k;
      add_words(l.q, t);
    }
  }
}

// 128- and 192-bit atomic adds from 64-bit atomics: each word's carry-out is known from the value the atomic
// returns, and adding the carries into the next word keeps the multi-word sum exact (mod 2^128 / 2^192).
__device__ __forceinline__ void atomic_add_u128(u64* w, const U128& v) {
  const u64 old = atomicAdd(w, v.w[0]);
  atomicAdd(w + 1, v.w[1] + (old + v.w[0] < old ? 1ull : 0ull));
}
__device__ __forceinline__ void atomic_add_u192(u64* w, const U192& v) {
  const u64 old0 = atomicAdd(w, v.w[0]);
  const u64 c0 = old0 + v.w[0] < old0 ? 1ull : 0ull;
  const u64 t1 = v.w[1] + c0;
  const u64 c1a = t1 < v.w[1] ? 1ull : 0ull;  // v1 = 2^64 - 1 and c0 = 1: t1 = 0, the carry goes straight to word 2
  const u64 old1 = atomicAdd(w + 1, t1);
  atomicAdd(w + 2, v.w[2] + c1a + (old1 + t1 < old1 ? 1ull : 0ull));
}

struct Vec4 {
  float v[4];
};

__device__ __forceinline__ Vec4 load4(const float* p) {
  const float4 f = *reinterpret_cast<const float4*>(p);
  return {{f.x, f.y, f.z, f.w}};
}
__device__ __forceinline__ Vec4 load4(const __nv_bfloat16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&u.x);
  const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
  return {{__low2float(a), __high2float(a), __low2float(b), __high2float(b)}};
}
__device__ __forceinline__ void store4(float* p, const Vec4& v) {
  *reinterpret_cast<float4*>(p) = make_float4(v.v[0], v.v[1], v.v[2], v.v[3]);
}
__device__ __forceinline__ void store4(__nv_bfloat16* p, const Vec4& v) {
  uint2 u;
  *reinterpret_cast<__nv_bfloat162*>(&u.x) = __floats2bfloat162_rn(v.v[0], v.v[1]);
  *reinterpret_cast<__nv_bfloat162*>(&u.y) = __floats2bfloat162_rn(v.v[2], v.v[3]);
  *reinterpret_cast<uint2*>(p) = u;
}

int pixels_per_block(std::int64_t c) { return static_cast<int>(std::max<std::int64_t>(1, kSlotElems / c)); }

// Block (chunk, n): threads = quads * rows; thread (row, quad) owns channels 4*quad..4*quad+3 of pixels
// row, row + rows, ... within the chunk. Per-channel sums are reduced per group through shared memory (integer
// adds, any order) and added to the global accumulators with one set of atomics per (block, group).
template <typename T>
__global__ void accumulate_kernel(const T* __restrict__ x, std::int64_t hw, int c, int groups, int ppb,
                                  Accum* __restrict__ acc) {
  extern __shared__ u64 red[];  // [rows][c][5]: s_lo, s_hi, q0, q1, q2
  __shared__ unsigned bad_group[kMaxChannels];  // groups <= C <= kMaxChannels
  const int quads = c / 4, rows = blockDim.x / quads;
  const int quad = threadIdx.x % quads, row = threadIdx.x / quads;
  const int n = blockIdx.y, cpg = c / groups;
  for (int g = threadIdx.x; g < groups; g += blockDim.x) bad_group[g] = 0;
  const std::int64_t p0 = static_cast<std::int64_t>(blockIdx.x) * ppb;
  const std::int64_t p1 = std::min<std::int64_t>(p0 + ppb, hw);
  const T* xn = x + static_cast<std::int64_t>(n) * hw * c + 4 * quad;

  Lane lane[4];
  for (std::int64_t p = p0 + row; p < p1; p += rows) {
    const Vec4 v = load4(xn + p * c);
#pragma unroll
    for (int j = 0; j < 4; ++j) accumulate(lane[j], v.v[j]);
  }
  __syncthreads();  // bad_group initialized
  {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      u64* r = red + (static_cast<std::size_t>(row) * c + 4 * quad + j) * 5;
      r[0] = lane[j].s.w[0];
      r[1] = lane[j].s.w[1];
      r[2] = lane[j].q.w[0];
      r[3] = lane[j].q.w[1];
      r[4] = lane[j].q.w[2];
      if (lane[j].bad) atomicOr(&bad_group[(4 * quad + j) / cpg], 1u);
    }
  }
  __syncthreads();
  for (int g = threadIdx.x; g < groups; g += blockDim.x) {
    U128 s;
    U192 q;
    for (int r = 0; r < rows; ++r)
      for (int ch = g * cpg; ch < (g + 1) * cpg; ++ch) {
        const u64* e = red + (static_cast<std::size_t>(r) * c + ch) * 5;
        add_words(s, U128{{e[0], e[1]}});
        add_words(q, U192{{e[2], e[3], e[4]}});
      }
    Accum* a = acc + static_cast<std::size_t>(n) * groups + g;
    atomic_add_u128(&a->s_lo, s);
    atomic_add_u192(&a->q0, q);
    if (bad_group[g]) atomicAdd(&a->bad, 1ull);
  }
}

__global__ void finalize_kernel(const Accum* __restrict__ acc, int ng, double inv_count, double eps,
                                float* __restrict__ stats) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ng) return;
  const Accum a = acc[i];
  if (a.bad != 0) {
    stats[2 * i] = stats[2 * i + 1] = __int_as_float(0x7fc00000);
    return;
  }
  const U128 s{{a.s_lo, a.s_hi}};
  const bool neg = (a.s_hi >> 63) != 0;
  const U128 mag = neg ? negate_words(s) : s;
  double sum = (static_cast<double>(mag.w[1]) * 0x1p64 + static_cast<double>(mag.w[0])) * 0x1p-64;
  if (neg) sum = -sum;
  const double sum_sq = (static_cast<double>(a.q2) * 0x1p128 + static_cast<double>(a.q1) * 0x1p64 + static_cast<double>(a.q0)) * 0x1p-96;
  const double mean = sum * inv_count;
  const double var = fmax(sum_sq * inv_count - mean * mean, 0.0);
  stats[2 * i] = static_cast<float>(mean);
  stats[2 * i + 1] = static_cast<float>(1.0 / sqrt(var + eps));
}

__device__ __forceinline__ float silu(float y) { return y / (1.0f + expf(-y)); }  // torch's silu formula

template <typename T>
__device__ __forceinline__ float round_to(float v);
template <>
__device__ __forceinline__ float round_to<float>(float v) { return v; }
template <>
__device__ __forceinline__ float round_to<__nv_bfloat16>(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// y = x * a + b with a = rstd * gamma, b = -a * mean + beta (torch's fused GroupNorm params, both FMAs), then
// SiLU on the output-dtype value (torch runs SiLU as a separate op on the stored GroupNorm result).
// x and y may alias (in-place apply), so neither is __restrict__.
template <typename T>
__global__ void apply_kernel(const T* x, const float* __restrict__ stats, const float* __restrict__ gamma,
                             const float* __restrict__ beta, T* y, std::int64_t hw, int c, int groups,
                             bool use_silu, std::int64_t total_quads) {
  const int quads = c / 4, cpg = c / groups;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total_quads;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int c0 = static_cast<int>(i % quads) * 4;
    const std::int64_t n = i / (hw * quads);
    Vec4 v = load4(x + i * 4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int ch = c0 + j;
      const float* st = stats + (n * groups + ch / cpg) * 2;
      const float a = st[1] * gamma[ch];
      const float b = fmaf(-a, st[0], beta[ch]);
      float r = fmaf(a, v.v[j], b);
      if (use_silu) r = silu(round_to<T>(r));
      v.v[j] = r;
    }
    store4(y + i * 4, v);
  }
}

struct GnShape {
  std::int64_t n, hw, c;
};

GnShape check_x(const Tensor& x, int groups, const char* op) {
  if (x.rank != 4 || x.device != Device::kCUDA || !x.is_contiguous() || (x.dtype != DType::kF32 && x.dtype != DType::kBF16))
    throw std::invalid_argument(std::string(op) + ": x must be a contiguous f32/bf16 CUDA [N,H,W,C], got " + x.describe());
  const GnShape s{x.dim(0), x.dim(1) * x.dim(2), x.dim(3)};
  if (groups <= 0 || s.c % groups != 0 || s.c % 4 != 0 || s.c > kMaxChannels || s.n > 65535)
    throw std::invalid_argument(std::string(op) + ": need C % groups == 0, C % 4 == 0, C <= 1024, N <= 65535");
  return s;
}

std::size_t accum_bytes(std::int64_t batch, int groups) {
  return static_cast<std::size_t>(batch) * static_cast<std::size_t>(groups) * sizeof(Accum);
}

constexpr std::size_t kAlign = 256;
std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

}  // namespace

std::size_t group_norm_accum_bytes(int batch, int groups) {
  if (batch <= 0 || groups <= 0) throw std::invalid_argument("group_norm_accum_bytes: batch and groups must be positive");
  return accum_bytes(batch, groups);
}

void group_norm_accum_zero(void* accum, int batch, int groups, cudaStream_t stream) {
  MI_CUDA_CHECK(cudaMemsetAsync(accum, 0, group_norm_accum_bytes(batch, groups), stream));
}

void group_norm_accumulate(const Tensor& x, int groups, void* accum, cudaStream_t stream) {
  const GnShape s = check_x(x, groups, "group_norm_accumulate");
  if (reinterpret_cast<std::uintptr_t>(x.data) % 16 || reinterpret_cast<std::uintptr_t>(accum) % 16)
    throw std::invalid_argument("group_norm_accumulate: x and accum must be 16-byte aligned");
  if (s.hw == 0 || s.n == 0) return;
  const int ppb = pixels_per_block(s.c);
  const int quads = static_cast<int>(s.c / 4);
  const int rows = std::max(1, 256 / quads);
  const std::size_t smem = static_cast<std::size_t>(rows) * static_cast<std::size_t>(s.c) * 5 * sizeof(u64);
  const dim3 grid(static_cast<unsigned>((s.hw + ppb - 1) / ppb), static_cast<unsigned>(s.n));
  auto* acc = static_cast<Accum*>(accum);
  if (x.dtype == DType::kF32) {
    accumulate_kernel<float><<<grid, quads * rows, smem, stream>>>(x.ptr<float>(), s.hw, static_cast<int>(s.c), groups, ppb, acc);
  } else {
    accumulate_kernel<__nv_bfloat16><<<grid, quads * rows, smem, stream>>>(x.ptr<__nv_bfloat16>(), s.hw, static_cast<int>(s.c),
                                                                           groups, ppb, acc);
  }
  MI_CUDA_CHECK_LAUNCH();
}

void group_norm_finalize(const void* accum, int batch, int groups, std::int64_t count_per_group, float eps,
                         const Tensor& stats, cudaStream_t stream) {
  if (stats.dtype != DType::kF32 || stats.device != Device::kCUDA || !stats.is_contiguous() ||
      stats.numel() != static_cast<std::int64_t>(batch) * groups * 2)
    throw std::invalid_argument("group_norm_finalize: stats must be contiguous f32 [N, groups, 2] on CUDA");
  if (count_per_group <= 0 || count_per_group > kMaxGroupElems)
    throw std::invalid_argument("group_norm_finalize: count_per_group must be in [1, 2^30]");
  const int ng = batch * groups;
  if (ng == 0) return;
  finalize_kernel<<<(ng + 127) / 128, 128, 0, stream>>>(static_cast<const Accum*>(accum), ng,
                                                         1.0 / static_cast<double>(count_per_group),
                                                         static_cast<double>(eps), stats.ptr<float>());
  MI_CUDA_CHECK_LAUNCH();
}

std::size_t group_norm_workspace_bytes(const Tensor& x, int groups) {
  const GnShape s = check_x(x, groups, "group_norm_workspace_bytes");
  return align_up(accum_bytes(s.n, groups)) + align_up(static_cast<std::size_t>(s.n) * groups * 2 * sizeof(float));
}

void group_norm_stats(const Tensor& x, int groups, float eps, const Tensor& stats, void* workspace,
                      std::size_t workspace_bytes, cudaStream_t stream) {
  const GnShape s = check_x(x, groups, "group_norm_stats");
  if (workspace_bytes < accum_bytes(s.n, groups)) throw std::invalid_argument("group_norm_stats: workspace too small");
  const int batch = static_cast<int>(s.n);
  group_norm_accum_zero(workspace, batch, groups, stream);
  group_norm_accumulate(x, groups, workspace, stream);
  group_norm_finalize(workspace, batch, groups, s.hw * (s.c / groups), eps, stats, stream);
}

void group_norm_apply(const Tensor& x, const Tensor& stats, const Tensor& gamma, const Tensor& beta, bool silu,
                      const Tensor& y, cudaStream_t stream) {
  if (x.rank != 4 || x.dim(0) <= 0) throw std::invalid_argument("group_norm_apply: x must be a non-empty [N,H,W,C]");
  const int groups = static_cast<int>(stats.numel() / (2 * x.dim(0)));
  const GnShape s = check_x(x, groups, "group_norm_apply");
  if (stats.numel() != s.n * groups * 2 || stats.device != Device::kCUDA || !stats.is_contiguous())
    throw std::invalid_argument("group_norm_apply: stats must be contiguous f32 [N, groups, 2] on CUDA");
  if (reinterpret_cast<std::uintptr_t>(x.data) % 16 || reinterpret_cast<std::uintptr_t>(y.data) % 16)
    throw std::invalid_argument("group_norm_apply: x and y must be 16-byte aligned");
  if (y.dtype != x.dtype || y.numel() != x.numel() || !y.is_contiguous() || y.device != Device::kCUDA)
    throw std::invalid_argument("group_norm_apply: y must match x");
  if (gamma.dtype != DType::kF32 || beta.dtype != DType::kF32 || gamma.numel() != s.c || beta.numel() != s.c ||
      stats.dtype != DType::kF32)
    throw std::invalid_argument("group_norm_apply: gamma/beta must be f32 [C], stats f32 [N, groups, 2]");
  const std::int64_t total = x.numel() / 4;
  if (total == 0) return;
  const int blocks = static_cast<int>(std::min<std::int64_t>((total + 255) / 256, 65535 * 8));
  if (x.dtype == DType::kF32) {
    apply_kernel<float><<<blocks, 256, 0, stream>>>(x.ptr<float>(), stats.ptr<float>(), gamma.ptr<float>(), beta.ptr<float>(),
                                                    y.ptr<float>(), s.hw, static_cast<int>(s.c), groups, silu, total);
  } else {
    apply_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(x.ptr<__nv_bfloat16>(), stats.ptr<float>(), gamma.ptr<float>(),
                                                            beta.ptr<float>(), y.ptr<__nv_bfloat16>(), s.hw,
                                                            static_cast<int>(s.c), groups, silu, total);
  }
  MI_CUDA_CHECK_LAUNCH();
}

void group_norm(const Tensor& x, int groups, float eps, const Tensor& gamma, const Tensor& beta, bool silu,
                const Tensor& y, void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  const GnShape s = check_x(x, groups, "group_norm");
  if (workspace_bytes < group_norm_workspace_bytes(x, groups)) throw std::invalid_argument("group_norm: workspace too small");
  const std::size_t stats_off = align_up(accum_bytes(s.n, groups));
  Tensor stats = Tensor::view(static_cast<char*>(workspace) + stats_off, DType::kF32, Device::kCUDA, {s.n, groups, 2});
  group_norm_stats(x, groups, eps, stats, workspace, stats_off, stream);
  group_norm_apply(x, stats, gamma, beta, silu, y, stream);
}

}  // namespace mi::vae
