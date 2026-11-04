// The VAE decoder's kernels (vae_kernels.h). Every workgroup is one wave of 32 threads, as in block_kernels.hip; the
// matrix products run on mma_leaf.h's leaf over bf16 pieces of the fp32 operands. The op orders and rounding points are
// the engine's reference decode's (kernels/vae: conv2d.cu's TF32 operands and bias-then-residual epilogue, group_norm.cu's
// exact statistics and fused apply, attention.cu's scale) and the pipeline's pre- and postprocessing (elementwise.cu).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "mma_leaf.h"
#include "vae_kernels.h"

namespace hip_vae {
namespace {

namespace leaf = hip_block::leaf;
using namespace hip_block::detail;
using leaf::FragA;
using leaf::FragAcc;
using leaf::FragB;

constexpr int kFrag = 16;  // fragment M, N and K
constexpr int kWave = 32;  // threads per workgroup
constexpr int kMf = kTileM / kFrag;
constexpr int kNf = kTileN / kFrag;          // piece_gemm
constexpr int kConvNf = kConvTileN / kFrag;  // conv
static_assert(kTileM == kWave, "the convolution stages one row of A per thread");

__device__ __forceinline__ int thread_id() { return static_cast<int>(threadIdx.x); }

// Round half away from zero to 10 mantissa bits (cvt.rna.tf32.f32, cuDNN's operand rounding); inf and NaN pass.
__device__ __forceinline__ float tf32(float v) {
  const std::uint32_t u = float_to_bits(v);
  if ((u & 0x7f800000u) == 0x7f800000u) return v;
  return bits_to_float((u + 0x1000u) & 0xffffe000u);
}

// v as kPieces bf16 pieces, each the nearest bf16 of the remainder (exact in fp32): host_split's order. An infinity or
// NaN is piece 0 with the other pieces 0 (its remainder would be NaN).
template <int kPieces, bool kTf32>
__device__ __forceinline__ void split(float v, std::uint16_t (&piece)[kPieces]) {
  const bool finite = (float_to_bits(v) & 0x7f800000u) != 0x7f800000u;
  float r = kTf32 ? tf32(v) : v;
  HIP_BLOCK_UNROLL
  for (int t = 0; t < kPieces; ++t) {
    const std::uint16_t b = bf(r);
    piece[t] = t == 0 || finite ? b : std::uint16_t{0};
    r = add(r, -f(b));
  }
}

// ---- The tile epilogue ------------------------------------------------------------------------------------------------

// out[m, n] = (acc + bias[n]) + residual[m, n] for the tile's rows < m_end and columns < n_end, one 16 x 16 fragment at a
// time through `epi`: thread t writes 8 columns of fragment row t / 2. bias, residual: null for none.
template <int kNfT>
__device__ __forceinline__ void store_tile_out(const FragAcc (&acc)[kMf][kNfT], float* epi, std::int64_t m_base, int n_base,
                                               std::int64_t m_end, int n_end, const float* bias, const float* residual,
                                               float* out, std::int64_t ldo) {
  const int t = thread_id(), fr = t / 2, fc = (t & 1) * 8;
  HIP_BLOCK_UNROLL
  for (int mf = 0; mf < kMf; ++mf) {
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNfT; ++nf) {
      leaf::store_acc(epi, acc[mf][nf], kFrag);
      leaf::pad_barrier();
      const std::int64_t m = m_base + mf * kFrag + fr;
      const int n0 = n_base + nf * kFrag + fc;
      if (m < m_end && n0 < n_end) {
        float y[8];
        HIP_BLOCK_UNROLL
        for (int j = 0; j < 8; ++j) y[j] = epi[fr * kFrag + fc + j];
        float* o = out + m * ldo + n0;
        if (bias != nullptr) {
          HIP_BLOCK_UNROLL
          for (int j = 0; j < 8; ++j) y[j] = n0 + j < n_end ? add(y[j], bias[n0 + j]) : y[j];
        }
        if (residual != nullptr) {
          const float* rr = residual + m * ldo + n0;
          HIP_BLOCK_UNROLL
          for (int j = 0; j < 8; ++j) y[j] = n0 + j < n_end ? add(y[j], rr[j]) : y[j];
        }
        if (n0 + 8 <= n_end) {
          reinterpret_cast<float4*>(o)[0] = make_float4(y[0], y[1], y[2], y[3]);
          reinterpret_cast<float4*>(o)[1] = make_float4(y[4], y[5], y[6], y[7]);
        } else {
          for (int j = 0; j < n_end - n0; ++j) o[j] = y[j];
        }
      }
      leaf::barrier();  // the reads of epi before the next fragment's store
    }
  }
}

template <int kMfT, int kNfT>
__device__ __forceinline__ void clear_tile(FragAcc (&acc)[kMfT][kNfT]) {
  HIP_BLOCK_UNROLL
  for (int mf = 0; mf < kMfT; ++mf) {
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNfT; ++nf) leaf::clear(acc[mf][nf]);
  }
}

// ---- Convolution ------------------------------------------------------------------------------------------------------

struct ConvKernelArgs {
  const float* x;
  const std::uint16_t* w;
  const float* bias;
  const float* residual;
  float* out;
  std::int64_t ldo;
  std::int64_t m;         // output pixels
  std::int64_t w_stride;  // elements per weight piece
  int h, wd, c, n, k;
  int n_tiles;
};

// The 16 values of A at row m (output pixel oy, ox), columns tap * c + ch .. + 16 of the im2col; 0 in the padding.
template <int kTaps, bool kUpsample>
__device__ __forceinline__ void load_row(const ConvKernelArgs& a, bool row_ok, std::int64_t m, int oy, int ox, int tap,
                                         int ch, float (&v)[16]) {
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 16; ++i) v[i] = 0.0f;
  if (!row_ok) return;
  std::int64_t pix = m;
  if (kTaps == 9) {
    const int iy = oy + tap / 3 - 1, ix = ox + tap % 3 - 1;
    if (iy < 0 || ix < 0 || iy >= a.h || ix >= a.wd) return;
    pix = kUpsample ? static_cast<std::int64_t>(iy >> 1) * (a.wd >> 1) + (ix >> 1) : static_cast<std::int64_t>(iy) * a.wd + ix;
  }
  const float4* p = reinterpret_cast<const float4*>(a.x + pix * a.c + ch);
  HIP_BLOCK_UNROLL
  for (int q = 0; q < 4; ++q) {
    const float4 f4 = p[q];
    v[4 * q] = f4.x;
    v[4 * q + 1] = f4.y;
    v[4 * q + 2] = f4.z;
    v[4 * q + 3] = f4.w;
  }
}

// Row t of the staged A pieces: 16 bf16 per piece, two 16-byte stores each.
template <int kPieces, bool kTf32>
__device__ __forceinline__ void stage_row(std::uint16_t (&stage)[kPieces][kTileM * kFrag], int t, const float (&v)[16]) {
  std::uint32_t word[kPieces][8];
  HIP_BLOCK_UNROLL
  for (int c = 0; c < 8; ++c) {
    std::uint16_t lo[kPieces], hi[kPieces];
    split<kPieces, kTf32>(v[2 * c], lo);
    split<kPieces, kTf32>(v[2 * c + 1], hi);
    HIP_BLOCK_UNROLL
    for (int i = 0; i < kPieces; ++i) word[i][c] = static_cast<std::uint32_t>(lo[i]) | static_cast<std::uint32_t>(hi[i]) << 16;
  }
  HIP_BLOCK_UNROLL
  for (int i = 0; i < kPieces; ++i) {
    uint4* dst = reinterpret_cast<uint4*>(&stage[i][t * kFrag]);
    dst[0] = make_uint4(word[i][0], word[i][1], word[i][2], word[i][3]);
    dst[1] = make_uint4(word[i][4], word[i][5], word[i][6], word[i][7]);
  }
}

// kProducts: 4 (TF32, all), 3 (TF32 without lo x lo), 9 (fp32, all). A stages through __shared__ memory, split as it is
// stored; the weight's pieces are packed fragments read from global memory.
template <int kTaps, bool kUpsample, int kProducts>
__global__ __launch_bounds__(kWave) void conv_kernel(const ConvKernelArgs a) {
  constexpr int kPieces = kProducts == 9 ? 3 : 2;
  constexpr bool kTf32 = kProducts != 9;
  alignas(16) __shared__ std::uint16_t stage[kPieces][kTileM * kFrag];
  alignas(16) __shared__ float epi[kFrag * kFrag];
  const int t = thread_id();
  const std::int64_t tile = blockIdx.x;
  const std::int64_t m_base = tile / a.n_tiles * kTileM;
  const int n_tile = static_cast<int>(tile % a.n_tiles);
  const std::int64_t m = m_base + t;  // the row of A this thread stages
  const bool row_ok = m < a.m;
  int oy = 0, ox = 0;
  if (kTaps == 9 && row_ok) {
    oy = static_cast<int>(m / a.wd);
    ox = static_cast<int>(m - static_cast<std::int64_t>(oy) * a.wd);
  }
  const int steps = a.k / kFrag;
  int tap = 0, ch = 0;  // of the next step to load; c is a multiple of 16, so a step lies in one tap
  const auto advance = [&]() {
    ch += kFrag;
    if (ch == a.c) {
      ch = 0;
      ++tap;
    }
  };
  float v[16];
  load_row<kTaps, kUpsample>(a, row_ok, m, oy, ox, tap, ch, v);
  advance();
  stage_row<kPieces, kTf32>(stage, t, v);
  leaf::pad_barrier();

  FragAcc acc[kMf][kConvNf];
  clear_tile(acc);
  const std::uint16_t* wt = a.w + static_cast<std::int64_t>(n_tile) * kConvNf * steps * (kFrag * kFrag);
  for (int step = 0; step < steps; ++step) {
    const bool more = step + 1 < steps;
    if (more) {
      load_row<kTaps, kUpsample>(a, row_ok, m, oy, ox, tap, ch, v);
      advance();
    }
    FragA fa[kPieces][kMf];
    HIP_BLOCK_UNROLL
    for (int i = 0; i < kPieces; ++i) {
      HIP_BLOCK_UNROLL
      for (int mf = 0; mf < kMf; ++mf) leaf::load_a(fa[i][mf], &stage[i][mf * kFrag * kFrag], kFrag);
    }
    HIP_BLOCK_UNROLL
    for (int jj = 0; jj < kPieces; ++jj) {
      const int j = kPieces - 1 - jj;
      FragB fb[kConvNf];
      HIP_BLOCK_UNROLL
      for (int nf = 0; nf < kConvNf; ++nf) {
        leaf::load_b(fb[nf], wt + j * a.w_stride + (static_cast<std::int64_t>(nf) * steps + step) * (kFrag * kFrag), kFrag);
      }
      HIP_BLOCK_UNROLL
      for (int ii = 0; ii < kPieces; ++ii) {
        const int i = kPieces - 1 - ii;
        if (kProducts == 3 && i + j >= 2) continue;
        HIP_BLOCK_UNROLL
        for (int nf = 0; nf < kConvNf; ++nf) {
          HIP_BLOCK_UNROLL
          for (int mf = 0; mf < kMf; ++mf) leaf::mma(acc[mf][nf], fa[i][mf], fb[nf]);
        }
      }
    }
    if (more) {
      leaf::barrier();  // this step's fragment reads of `stage` before the next step's stores
      stage_row<kPieces, kTf32>(stage, t, v);
      leaf::pad_barrier();
    }
  }
  store_tile_out(acc, epi, m_base, n_tile * kConvTileN, a.m, a.n, a.bias, a.residual, a.out, a.ldo);
}

// conv_out: one output pixel per thread, n <= 4 accumulators, K ascending (tap, then channel).
struct NarrowArgs {
  const float* x;
  const float* w;
  const float* bias;
  float* out;
  std::int64_t m;
  int h, w_out, c, n;
};
__global__ __launch_bounds__(kWave) void conv_narrow_kernel(const NarrowArgs a) {
  const std::int64_t m = static_cast<std::int64_t>(blockIdx.x) * kWave + thread_id();
  if (m >= a.m) return;
  const int oy = static_cast<int>(m / a.w_out), ox = static_cast<int>(m - static_cast<std::int64_t>(oy) * a.w_out);
  const std::int64_t kk = 9ll * a.c;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int tap = 0; tap < 9; ++tap) {
    const int iy = oy + tap / 3 - 1, ix = ox + tap % 3 - 1;
    if (iy < 0 || ix < 0 || iy >= a.h || ix >= a.w_out) continue;
    const float* xp = a.x + (static_cast<std::int64_t>(iy) * a.w_out + ix) * a.c;
    const float* wp = a.w + static_cast<std::int64_t>(tap) * a.c;
    for (int ch = 0; ch < a.c; ch += 4) {
      const float4 x4 = *reinterpret_cast<const float4*>(xp + ch);
      const float xv[4] = {tf32(x4.x), tf32(x4.y), tf32(x4.z), tf32(x4.w)};
      HIP_BLOCK_UNROLL
      for (int o = 0; o < 4; ++o) {
        if (o < a.n) {
          HIP_BLOCK_UNROLL
          for (int q = 0; q < 4; ++q) acc[o] = add(acc[o], mul(xv[q], wp[o * kk + ch + q]));  // TF32 products are exact
        }
      }
    }
  }
  for (int o = 0; o < a.n; ++o) a.out[m * a.n + o] = add(acc[o], a.bias[o]);
}

// ---- Attention --------------------------------------------------------------------------------------------------------

struct SplitArgs {
  const float* src;
  std::int64_t ld;
  std::uint16_t* dst;
  std::int64_t stride;  // elements per piece
  int rows, cols, rows_pad, col_groups;
};
// One thread per (row, 16 columns) of the packed output; along rows first when transposed (coalesced reads).
template <bool kTranspose>
__global__ __launch_bounds__(kWave) void split_pack_kernel(const SplitArgs a) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * kWave + thread_id();
  if (idx >= static_cast<std::int64_t>(a.rows_pad) * a.col_groups) return;
  const int r = static_cast<int>(kTranspose ? idx % a.rows_pad : idx / a.col_groups);
  const int g = static_cast<int>(kTranspose ? idx / a.rows_pad : idx % a.col_groups);
  std::uint32_t word[3][8];
  HIP_BLOCK_UNROLL
  for (int c = 0; c < 8; ++c) {
    std::uint16_t p[2][3];
    HIP_BLOCK_UNROLL
    for (int e = 0; e < 2; ++e) {
      const int col = g * kFrag + 2 * c + e;
      float v = 0.0f;
      if (r < a.rows && col < a.cols) v = kTranspose ? a.src[static_cast<std::int64_t>(col) * a.ld + r] : a.src[r * a.ld + col];
      split<3, false>(v, p[e]);
    }
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 3; ++i) word[i][c] = static_cast<std::uint32_t>(p[0][i]) | static_cast<std::uint32_t>(p[1][i]) << 16;
  }
  const std::int64_t at = (static_cast<std::int64_t>(r / kFrag) * a.col_groups + g) * (kFrag * kFrag) + r % kFrag * kFrag;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 3; ++i) {
    uint4* d = reinterpret_cast<uint4*>(a.dst + i * a.stride + at);
    d[0] = make_uint4(word[i][0], word[i][1], word[i][2], word[i][3]);
    d[1] = make_uint4(word[i][4], word[i][5], word[i][6], word[i][7]);
  }
}

struct PieceGemmArgs {
  const std::uint16_t* a;
  std::int64_t a_stride;
  const std::uint16_t* b;
  std::int64_t b_stride;
  float* out;
  std::int64_t ldo;
  int ksteps, m, n, n_tiles;
};
// Products i + j <= 2 of 3 pieces each, all operands packed fragments in global memory.
__global__ __launch_bounds__(kWave) void piece_gemm_kernel(const PieceGemmArgs g) {
  alignas(16) __shared__ float epi[kFrag * kFrag];
  const std::int64_t tile = blockIdx.x;
  const std::int64_t m_base = tile / g.n_tiles * kTileM;
  const int n_tile = static_cast<int>(tile % g.n_tiles);
  const std::uint16_t* ap = g.a + m_base * g.ksteps * kFrag;  // row group m_base / 16, fragment (mf, step)
  const std::uint16_t* bp = g.b + static_cast<std::int64_t>(n_tile) * kNf * g.ksteps * (kFrag * kFrag);
  FragAcc acc[kMf][kNf];
  clear_tile(acc);
  for (int step = 0; step < g.ksteps; ++step) {
    FragA fa[3][kMf];
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 3; ++i) {
      HIP_BLOCK_UNROLL
      for (int mf = 0; mf < kMf; ++mf) {
        leaf::load_a(fa[i][mf], ap + i * g.a_stride + (static_cast<std::int64_t>(mf) * g.ksteps + step) * (kFrag * kFrag), kFrag);
      }
    }
    HIP_BLOCK_UNROLL
    for (int jj = 0; jj < 3; ++jj) {
      const int j = 2 - jj;
      FragB fb[kNf];
      HIP_BLOCK_UNROLL
      for (int nf = 0; nf < kNf; ++nf) {
        leaf::load_b(fb[nf], bp + j * g.b_stride + (static_cast<std::int64_t>(nf) * g.ksteps + step) * (kFrag * kFrag), kFrag);
      }
      HIP_BLOCK_UNROLL
      for (int ii = 0; ii < 3; ++ii) {
        const int i = 2 - ii;
        if (i + j > 2) continue;
        HIP_BLOCK_UNROLL
        for (int nf = 0; nf < kNf; ++nf) {
          HIP_BLOCK_UNROLL
          for (int mf = 0; mf < kMf; ++mf) leaf::mma(acc[mf][nf], fa[i][mf], fb[nf]);
        }
      }
    }
  }
  store_tile_out(acc, epi, m_base, n_tile * kTileN, g.m, g.n, nullptr, nullptr, g.out, g.ldo);
}

// The wave's 32 values reduced through red[] by a halving tree (lane t + off into lane t): max or sum.
template <bool kMax>
__device__ __forceinline__ float wave_reduce(float v, float* red) {
  const int t = thread_id();
  red[t] = v;
  leaf::barrier();
  HIP_BLOCK_UNROLL
  for (int off = kWave / 2; off > 0; off >>= 1) {
    if (t < off) red[t] = kMax ? fmaxf(red[t], red[t + off]) : add(red[t], red[t + off]);
    leaf::barrier();
  }
  const float r = red[0];
  leaf::barrier();  // every lane has read red[0] before red is written again
  return r;
}

struct SoftmaxArgs {
  const float* s;
  std::int64_t lds;
  std::uint16_t* p;
  std::int64_t p_stride;
  int l, l_pad;
  float scale;
};
// One row per wave: the row's max, the sum of exp(s * scale - max * scale), then the probabilities' pieces.
__global__ __launch_bounds__(kWave) void softmax_pack_kernel(const SoftmaxArgs a) {
  __shared__ float red[kWave];
  const int t = thread_id();
  const int r = static_cast<int>(blockIdx.x);
  const float* s = a.s + r * a.lds;
  float mx = -3.402823466e38f;
  for (int j = t; j < a.l; j += kWave) mx = fmaxf(mx, s[j]);
  const float shift = mul(wave_reduce<true>(mx, red), a.scale);
  float sum = 0.0f;
  for (int j = t; j < a.l; j += kWave) sum = add(sum, expf(add(mul(s[j], a.scale), -shift)));
  const float total = wave_reduce<false>(sum, red);
  const std::int64_t row_at = static_cast<std::int64_t>(r / kFrag) * (a.l_pad / kFrag) * (kFrag * kFrag) + r % kFrag * kFrag;
  for (int j = t; j < a.l_pad; j += kWave) {
    const float pv = j < a.l ? div(expf(add(mul(s[j], a.scale), -shift)), total) : 0.0f;
    std::uint16_t piece[3];
    split<3, false>(pv, piece);
    const std::int64_t at = row_at + static_cast<std::int64_t>(j / kFrag) * (kFrag * kFrag) + j % kFrag;
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 3; ++i) a.p[i * a.p_stride + at] = piece[i];
  }
}

// ---- GroupNorm: the engine's exact statistics (kernels/vae/group_norm.cu) ----------------------------------------------

using u64 = unsigned long long;
constexpr int kMaxBiasedExp = 159;  // |x| < 2^33
constexpr int kGnChunkElems = 1 << 16;

template <int N>
struct Words {
  u64 w[N];
};
using U128 = Words<2>;
using U192 = Words<3>;

template <int N>
__device__ __forceinline__ void add_words(Words<N>& a, const Words<N>& b) {
  u64 carry = 0;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < N; ++i) {
    const u64 t = a.w[i] + carry;
    const u64 c = t < carry ? 1ull : 0ull;
    a.w[i] = t + b.w[i];
    carry = c | (a.w[i] < t ? 1ull : 0ull);
  }
}
template <int N>
__device__ __forceinline__ void sub_words(Words<N>& a, const Words<N>& b) {
  u64 borrow = 0;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < N; ++i) {
    const u64 t = a.w[i] - b.w[i];
    const u64 bo = a.w[i] < b.w[i] ? 1ull : 0ull;
    a.w[i] = t - borrow;
    borrow = bo | (t < borrow ? 1ull : 0ull);
  }
}
// v << sh as an N-word value that fits.
template <int N>
__device__ __forceinline__ Words<N> shl_words(u64 v, int sh) {
  Words<N> r;
  const int wi = sh >> 6, b = sh & 63;
  const u64 lo = v << b, hi = b != 0 ? v >> (64 - b) : 0ull;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < N; ++i) r.w[i] = i == wi ? lo : (i == wi + 1 ? hi : 0ull);
  return r;
}

struct Lane {
  U128 s;
  U192 q;
  unsigned bad;
};

__device__ __forceinline__ void accumulate(Lane& l, float v) {
  const std::uint32_t bits = float_to_bits(v);
  const int e = static_cast<int>((bits >> 23) & 0xffu);
  if (e == 0) return;  // zero or subnormal: below both grids
  if (e > kMaxBiasedExp) {
    l.bad = 1;
    return;
  }
  const u64 mant = (bits & 0x7fffffu) | 0x800000u;  // |x| = mant * 2^(e - 150)
  const int sh = e - 86;                            // |x| * 2^64 = mant * 2^(e - 86)
  U128 ax;
  if (sh >= 0) {
    ax = shl_words<2>(mant, sh);
  } else {
    const int k = -sh;
    ax.w[0] = k > 24 ? 0ull : (mant + (1ull << (k - 1))) >> k;
    ax.w[1] = 0;
  }
  if (bits >> 31) {
    sub_words(l.s, ax);
  } else {
    add_words(l.s, ax);
  }
  const u64 m2 = mant * mant;  // x^2 * 2^96 = m2 * 2^(2e - 204)
  const int sh2 = 2 * e - 204;
  if (sh2 >= 0) {
    add_words(l.q, shl_words<3>(m2, sh2));
  } else {
    const int k = -sh2;
    if (k <= 48) {
      const U192 t{{(m2 + (1ull << (k - 1))) >> k, 0ull, 0ull}};
      add_words(l.q, t);
    }
  }
}

struct GnAccumArgs {
  const float* x;
  u64* acc;  // [32][6]: s_lo, s_hi, q0, q1, q2, bad
  std::int64_t pixels, ppc;
  int c, cpg;
};
// One chunk of ppc pixels per wave; lane g sums group g's channels (contiguous, so the wave reads whole pixels). Integer
// sums: the order does not matter.
__global__ __launch_bounds__(kWave) void gn_accumulate_kernel(const GnAccumArgs a) {
  const int g = thread_id();
  const std::int64_t p0 = static_cast<std::int64_t>(blockIdx.x) * a.ppc;
  const std::int64_t p1 = p0 + a.ppc < a.pixels ? p0 + a.ppc : a.pixels;
  Lane l{{{0ull, 0ull}}, {{0ull, 0ull, 0ull}}, 0u};
  const float* xg = a.x + g * a.cpg;
  for (std::int64_t p = p0; p < p1; ++p) {
    for (int j = 0; j < a.cpg; ++j) accumulate(l, xg[p * a.c + j]);
  }
  u64* w = a.acc + g * 6;
  // 128- and 192-bit adds from 64-bit atomics, each word's carry from the value the atomic returns.
  const u64 s0 = atomicAdd(w, l.s.w[0]);
  atomicAdd(w + 1, l.s.w[1] + (s0 + l.s.w[0] < s0 ? 1ull : 0ull));
  const u64 q0 = atomicAdd(w + 2, l.q.w[0]);
  const u64 c0 = q0 + l.q.w[0] < q0 ? 1ull : 0ull;
  const u64 t1 = l.q.w[1] + c0;
  const u64 c1a = t1 < l.q.w[1] ? 1ull : 0ull;
  const u64 q1 = atomicAdd(w + 3, t1);
  atomicAdd(w + 4, l.q.w[2] + c1a + (q1 + t1 < q1 ? 1ull : 0ull));
  if (l.bad) atomicAdd(w + 5, 1ull);
}

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ double dmul(double a, double b) { return a * b; }
__device__ __forceinline__ double dadd(double a, double b) { return a + b; }
#else
__device__ __forceinline__ double dmul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double dadd(double a, double b) { return __dadd_rn(a, b); }
#endif

struct GnFinalizeArgs {
  const u64* acc;
  float* stats;
  double inv_count, eps;
};
__global__ __launch_bounds__(kWave) void gn_finalize_kernel(const GnFinalizeArgs a) {
  const int g = thread_id();
  const u64* w = a.acc + g * 6;
  if (w[5] != 0) {
    a.stats[2 * g] = a.stats[2 * g + 1] = bits_to_float(0x7fc00000u);
    return;
  }
  U128 mag{{w[0], w[1]}};
  const bool neg = (w[1] >> 63) != 0;
  if (neg) {
    U128 zero{{0ull, 0ull}};
    sub_words(zero, mag);
    mag = zero;
  }
  double sum = dmul(dadd(dmul(static_cast<double>(mag.w[1]), 0x1p64), static_cast<double>(mag.w[0])), 0x1p-64);
  if (neg) sum = -sum;
  const double sum_sq = dmul(dadd(dadd(dmul(static_cast<double>(w[4]), 0x1p128), dmul(static_cast<double>(w[3]), 0x1p64)),
                                  static_cast<double>(w[2])),
                             0x1p-96);
  const double mean = dmul(sum, a.inv_count);
  // sum_sq * inv_count - mean * mean as the engine's nvcc build runs it: the square rounded, then one fma.
  const double var = fmax(fma(sum_sq, a.inv_count, -dmul(mean, mean)), 0.0);
  a.stats[2 * g] = static_cast<float>(mean);
  a.stats[2 * g + 1] = static_cast<float>(1.0 / sqrt(dadd(var, a.eps)));
}

struct GnApplyArgs {
  const float* x;
  float* y;
  const float* stats;
  const float* gamma;
  const float* beta;
  std::int64_t quads;
  int c, cpg, silu;
};
__global__ __launch_bounds__(kWave) void gn_apply_kernel(const GnApplyArgs a) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kWave + thread_id();
  if (i >= a.quads) return;
  const int c0 = static_cast<int>(i % (a.c / 4)) * 4;
  float4 v = reinterpret_cast<const float4*>(a.x)[i];
  float r[4] = {v.x, v.y, v.z, v.w};
  HIP_BLOCK_UNROLL
  for (int j = 0; j < 4; ++j) {
    const int ch = c0 + j;
    const float* st = a.stats + 2 * (ch / a.cpg);
    const float sa = mul(st[1], a.gamma[ch]);
    const float sb = fmaf(-sa, st[0], a.beta[ch]);
    float y = fmaf(sa, r[j], sb);
    if (a.silu) y = silu_f32(y);
    r[j] = y;
  }
  reinterpret_cast<float4*>(a.y)[i] = make_float4(r[0], r[1], r[2], r[3]);
}

// ---- Pipeline ends ----------------------------------------------------------------------------------------------------

struct UnpackArgs {
  const float* z;
  const float* scale;
  const float* mean;
  float* y;
  std::int64_t count;
  int h, w;
};
__global__ __launch_bounds__(kWave) void latent_unpack_kernel(const UnpackArgs a) {
  const std::int64_t o = static_cast<std::int64_t>(blockIdx.x) * kWave + thread_id();
  if (o >= a.count) return;
  const int c = static_cast<int>(o % 32);
  const std::int64_t px = o / 32;
  const int ox = static_cast<int>(px % (2 * a.w)), oy = static_cast<int>(px / (2 * a.w));
  const int ch = 4 * c + 2 * (oy & 1) + (ox & 1);
  const float v = a.z[(static_cast<std::int64_t>(ch) * a.h + oy / 2) * a.w + ox / 2];
  a.y[o] = add(mul(v, a.scale[ch]), a.mean[ch]);
}

struct Rgb8Args {
  const float* x;
  std::uint8_t* rgb;
  std::int64_t n;
};
__global__ __launch_bounds__(kWave) void rgb8_kernel(const Rgb8Args a) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kWave + thread_id();
  if (i >= a.n) return;
  float v = a.x[i];
  std::uint8_t out = 0;
  if (!(v != v)) {
    v = fminf(fmaxf(v, -1.0f), 1.0f);
    v = fminf(fmaxf(add(mul(v, 0.5f), 0.5f), 0.0f), 1.0f);
    out = static_cast<std::uint8_t>(rintf(mul(v, 255.0f)));
  }
  a.rgb[i] = out;
}

// ---- Launch helpers ---------------------------------------------------------------------------------------------------

void require(bool ok, const char* op, const std::string& what) {
  if (!ok) throw std::invalid_argument(std::string("hip_vae::") + op + ": " + what);
}
void check_launch(const char* op) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("hip_vae::") + op + ": " + hipGetErrorString(e));
}
bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }
unsigned grid_of(std::int64_t items, std::int64_t per, const char* op) {
  const std::int64_t g = (items + per - 1) / per;
  require(g <= 0x7fffffff, op, "too many workgroups for one launch");
  return static_cast<unsigned>(g);
}

}  // namespace

const char* products_name(Products p) {
  switch (p) {
    case Products::kTf32: return "TF32, 4 products";
    case Products::kTf32Drop: return "TF32, 3 products (no lo x lo)";
    case Products::kFp32: return "fp32, 9 products";
  }
  return "?";
}

std::size_t packed_weight_elems(int n, int k, Products products) {
  return static_cast<std::size_t>(pieces_of(products)) * static_cast<std::size_t>(round_up(n, kConvTileN)) * static_cast<std::size_t>(k);
}

void pack_weight(const float* w, int n, int k, Products products, std::uint16_t* out) {
  require(n > 0 && k > 0 && k % kFrag == 0, "pack_weight", "n must be positive and k a positive multiple of 16");
  const int pieces = pieces_of(products);
  const std::int64_t n_pad = round_up(n, kConvTileN), stride = n_pad * k;
  std::fill(out, out + pieces * stride, std::uint16_t{0});
  std::uint16_t piece[3];
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < k; ++c) {
      host_split(w[static_cast<std::int64_t>(r) * k + c], products, piece);
      for (int t = 0; t < pieces; ++t) out[t * stride + packed_index(r, c, k)] = piece[t];
    }
  }
}

void conv(const ConvArgs& a, Products products, hipStream_t stream) {
  require(a.taps == 1 || a.taps == 9, "conv", "taps must be 1 or 9");
  require(a.c > 0 && a.c % kFrag == 0, "conv", "c must be a positive multiple of 16");
  require(a.h > 0 && a.w_out > 0 && a.n > 0, "conv", "empty shape");
  require(a.ldo >= a.n && a.ldo % 4 == 0, "conv", "ldo must cover n and be a multiple of 4");
  require(!a.upsample || (a.taps == 9 && a.h % 2 == 0 && a.w_out % 2 == 0), "conv", "upsample needs a 3x3 conv and an even output");
  require(products != Products::kFp32 || a.taps == 1, "conv", "fp32 products are for 1x1 convolutions (linears)");
  require(aligned16(a.x) && aligned16(a.w) && aligned16(a.out) && aligned16(a.residual), "conv", "pointers must be 16-byte aligned");
  const int k = a.taps * a.c;
  const ConvKernelArgs ka{a.x, a.w, a.bias, a.residual, a.out, a.ldo, static_cast<std::int64_t>(a.h) * a.w_out,
                          round_up(a.n, kConvTileN) * k, a.h, a.w_out, a.c, a.n, k,
                          static_cast<int>(round_up(a.n, kConvTileN) / kConvTileN)};
  const dim3 grid(grid_of((ka.m + kTileM - 1) / kTileM * ka.n_tiles, 1, "conv")), block(kWave);
  const bool tf32_all = products == Products::kTf32;
  if (products == Products::kFp32) {
    conv_kernel<1, false, 9><<<grid, block, 0, stream>>>(ka);
  } else if (a.taps == 1) {
    if (tf32_all) conv_kernel<1, false, 4><<<grid, block, 0, stream>>>(ka);
    else conv_kernel<1, false, 3><<<grid, block, 0, stream>>>(ka);
  } else if (a.upsample) {
    if (tf32_all) conv_kernel<9, true, 4><<<grid, block, 0, stream>>>(ka);
    else conv_kernel<9, true, 3><<<grid, block, 0, stream>>>(ka);
  } else {
    if (tf32_all) conv_kernel<9, false, 4><<<grid, block, 0, stream>>>(ka);
    else conv_kernel<9, false, 3><<<grid, block, 0, stream>>>(ka);
  }
  check_launch("conv");
}

void conv_narrow(const float* x, const float* w_tf32, const float* bias, float* out, int h, int w, int c, int n,
                 hipStream_t stream) {
  require(n >= 1 && n <= 4 && c > 0 && c % 4 == 0 && h > 0 && w > 0, "conv_narrow", "need 1 <= n <= 4, c a multiple of 4");
  require(aligned16(x), "conv_narrow", "x must be 16-byte aligned");
  const NarrowArgs a{x, w_tf32, bias, out, static_cast<std::int64_t>(h) * w, h, w, c, n};
  conv_narrow_kernel<<<dim3(grid_of(a.m, kWave, "conv_narrow")), dim3(kWave), 0, stream>>>(a);
  check_launch("conv_narrow");
}

void split_pack(const float* src, std::int64_t ld, bool transpose, int rows, int cols, std::uint16_t* dst, int rows_pad,
                int cols_pad, hipStream_t stream) {
  require(rows_pad % kFrag == 0 && cols_pad % kFrag == 0 && rows <= rows_pad && cols <= cols_pad, "split_pack",
          "padded sizes must be multiples of 16 covering the operand");
  require(aligned16(dst), "split_pack", "dst must be 16-byte aligned");
  const SplitArgs a{src, ld, dst, static_cast<std::int64_t>(rows_pad) * cols_pad, rows, cols, rows_pad, cols_pad / kFrag};
  const dim3 grid(grid_of(static_cast<std::int64_t>(rows_pad) * a.col_groups, kWave, "split_pack"));
  if (transpose) {
    split_pack_kernel<true><<<grid, dim3(kWave), 0, stream>>>(a);
  } else {
    split_pack_kernel<false><<<grid, dim3(kWave), 0, stream>>>(a);
  }
  check_launch("split_pack");
}

void piece_gemm(const std::uint16_t* a, std::int64_t a_stride, const std::uint16_t* b, std::int64_t b_stride, int k_pad,
                float* out, std::int64_t ldo, int m_rows, int n_cols, hipStream_t stream) {
  require(k_pad > 0 && k_pad % kFrag == 0, "piece_gemm", "k_pad must be a positive multiple of 16");
  require(ldo >= n_cols && ldo % 4 == 0, "piece_gemm", "ldo must cover n and be a multiple of 4");
  require(aligned16(a) && aligned16(b) && aligned16(out), "piece_gemm", "pointers must be 16-byte aligned");
  if (m_rows <= 0 || n_cols <= 0) return;
  const PieceGemmArgs g{a, a_stride, b, b_stride, out, ldo, k_pad / kFrag, m_rows, n_cols,
                        static_cast<int>(round_up(n_cols, kTileN) / kTileN)};
  const dim3 grid(grid_of(static_cast<std::int64_t>(round_up(m_rows, kTileM) / kTileM) * g.n_tiles, 1, "piece_gemm"));
  piece_gemm_kernel<<<grid, dim3(kWave), 0, stream>>>(g);
  check_launch("piece_gemm");
}

void softmax_pack(const float* s, std::int64_t lds, int rows, int l, float scale, std::uint16_t* p, std::int64_t p_stride,
                  int l_pad, hipStream_t stream) {
  require(l > 0 && l_pad >= l && l_pad % kFrag == 0 && lds >= l, "softmax_pack", "l_pad must be a multiple of 16 covering l");
  if (rows <= 0) return;
  const SoftmaxArgs a{s, lds, p, p_stride, l, l_pad, scale};
  softmax_pack_kernel<<<dim3(static_cast<unsigned>(rows)), dim3(kWave), 0, stream>>>(a);
  check_launch("softmax_pack");
}

std::size_t gn_accum_bytes() { return static_cast<std::size_t>(kGroups) * 6 * sizeof(u64); }

void group_norm_stats(const float* x, std::int64_t pixels, int c, float eps, void* accum, float* stats, hipStream_t stream) {
  require(c > 0 && c % kGroups == 0 && c <= 1024 && pixels > 0, "group_norm_stats", "c must be a multiple of 32 up to 1024");
  const int cpg = c / kGroups;
  require(pixels * cpg <= (std::int64_t{1} << 30), "group_norm_stats", "a group may hold up to 2^30 elements");
  const hipError_t e = hipMemsetAsync(accum, 0, gn_accum_bytes(), stream);
  if (e != hipSuccess) throw std::runtime_error(std::string("hip_vae::group_norm_stats: hipMemsetAsync: ") + hipGetErrorString(e));
  const std::int64_t ppc = std::max<std::int64_t>(1, kGnChunkElems / c);
  const GnAccumArgs aa{x, static_cast<u64*>(accum), pixels, ppc, c, cpg};
  gn_accumulate_kernel<<<dim3(grid_of(pixels, ppc, "group_norm_stats")), dim3(kWave), 0, stream>>>(aa);
  check_launch("group_norm_stats");
  const GnFinalizeArgs fa{static_cast<const u64*>(accum), stats, 1.0 / static_cast<double>(pixels * cpg), static_cast<double>(eps)};
  gn_finalize_kernel<<<dim3(1), dim3(kWave), 0, stream>>>(fa);
  check_launch("group_norm_stats finalize");
}

void group_norm_apply(const float* x, float* y, std::int64_t pixels, int c, const float* stats, const float* gamma,
                      const float* beta, bool silu, hipStream_t stream) {
  require(c > 0 && c % kGroups == 0 && c % 4 == 0, "group_norm_apply", "c must be a multiple of 32");
  require(aligned16(x) && aligned16(y), "group_norm_apply", "x and y must be 16-byte aligned");
  const GnApplyArgs a{x, y, stats, gamma, beta, pixels * c / 4, c, c / kGroups, silu ? 1 : 0};
  if (a.quads == 0) return;
  gn_apply_kernel<<<dim3(grid_of(a.quads, kWave, "group_norm_apply")), dim3(kWave), 0, stream>>>(a);
  check_launch("group_norm_apply");
}

void latent_unpack(const float* z, const float* scale, const float* mean, float* y, int h, int w, hipStream_t stream) {
  require(h > 0 && w > 0, "latent_unpack", "empty latent grid");
  const UnpackArgs a{z, scale, mean, y, 128ll * h * w, h, w};
  latent_unpack_kernel<<<dim3(grid_of(a.count, kWave, "latent_unpack")), dim3(kWave), 0, stream>>>(a);
  check_launch("latent_unpack");
}

void rgb8(const float* x, std::uint8_t* rgb, std::int64_t n, hipStream_t stream) {
  if (n <= 0) return;
  rgb8_kernel<<<dim3(grid_of(n, kWave, "rgb8")), dim3(kWave), 0, stream>>>(Rgb8Args{x, rgb, n});
  check_launch("rgb8");
}

}  // namespace hip_vae
