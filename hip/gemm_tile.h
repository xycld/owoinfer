// The per-wave GEMM tile walk shared by the backend's kernel sources (block_kernels.hip, dit_kernels.hip): fragment loads
// with zero-padded tails, the K-ascending walk over row-major or packed operands, the 2D block order of the tiles, the
// GEMM kernel with its epilogues, and the launch helpers. Device code in an anonymous namespace, so each source that
// includes it compiles its own copy of what it uses; the kernels' names stay hip_block::(anonymous namespace)::...
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "hip_block.h"
#include "mma_leaf.h"

namespace hip_block {
namespace {

using namespace detail;
using leaf::FragA;
using leaf::FragAcc;
using leaf::FragB;

constexpr int kFrag = 16;   // fragment M, N and K
constexpr int kWave = 32;   // threads per workgroup

// Per-wave output blocking, multiples of kFrag. Each pair of fragments is one fp32 accumulator in registers.
constexpr int kGemmTileM = 32, kGemmTileN = 48;
static_assert(kGemmTileM % kFrag == 0 && kGemmTileN % kFrag == 0);
// Tiles are walked in 2D blocks of about this many bytes per operand, so a block's operands stay in cache across its tiles.
// The order of the tiles does not change any output: within a tile K still ascends.
constexpr std::int64_t kCacheBlockBytes = 4 * 1000 * 1000;

constexpr float kRmsNormEps = 1e-6f;

__device__ __forceinline__ int thread_id() { return static_cast<int>(threadIdx.x); }

__device__ __forceinline__ int clamp_frag(int len) { return len <= 0 ? 0 : (len < kFrag ? len : kFrag); }

// Fragments over [rows, cols] of p; those that end inside the fragment go through a zero-padded copy in `pad`
// (kFrag * kFrag elements of __shared__ memory). Uniform control flow: rows and cols are the same on every thread.
__device__ __forceinline__ void load_a_padded(FragA& a, const bf16_bits* p, std::int64_t ld, int rows, int cols,
                                              bf16_bits* pad) {
  if (rows == kFrag && cols == kFrag) {
    leaf::load_a(a, p, static_cast<int>(ld));
    return;
  }
  for (int i = thread_id(); i < kFrag * kFrag; i += kWave) {
    const int r = i / kFrag, c = i % kFrag;
    pad[i] = r < rows && c < cols ? p[r * ld + c] : bf16_bits{0};
  }
  leaf::pad_barrier();
  leaf::load_a(a, pad, kFrag);
}
// B(k, n) = p[n * ld + k], cols columns of `ks` K values.
__device__ __forceinline__ void load_b_padded(FragB& b, const bf16_bits* p, std::int64_t ld, int cols, int ks,
                                              bf16_bits* pad) {
  if (cols == kFrag && ks == kFrag) {
    leaf::load_b(b, p, static_cast<int>(ld));
    return;
  }
  for (int i = thread_id(); i < kFrag * kFrag; i += kWave) {
    const int n = i / kFrag, k = i % kFrag;
    pad[i] = n < cols && k < ks ? p[n * ld + k] : bf16_bits{0};
  }
  leaf::pad_barrier();
  leaf::load_b(b, pad, kFrag);
}
// B(k, n) = p[k * ld + n], ks K values of 16 columns.
__device__ __forceinline__ void load_b_rows_padded(FragB& b, const bf16_bits* p, std::int64_t ld, int ks, bf16_bits* pad) {
  if (ks == kFrag) {
    leaf::load_b_rows(b, p, static_cast<int>(ld));
    return;
  }
  for (int i = thread_id(); i < kFrag * kFrag; i += kWave) {
    const int k = i / kFrag, n = i % kFrag;
    pad[i] = k < ks ? p[k * ld + n] : bf16_bits{0};
  }
  leaf::pad_barrier();
  leaf::load_b_rows(b, pad, kFrag);
}

template <int kMf, int kNf>
__device__ __forceinline__ void clear_tile(FragAcc (&acc)[kMf][kNf]) {
  HIP_BLOCK_UNROLL
  for (int mf = 0; mf < kMf; ++mf) {
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) leaf::clear(acc[mf][nf]);
  }
}

// acc[mf][nf] += a[rows, k_len] · b[nf]ᵀ with K ascending in steps of kFrag. a's fragment mf starts at row mf * kFrag;
// b[nf] holds b_cols[nf] columns of ldb elements (load_b's layout).
template <int kMf, int kNf>
__device__ __forceinline__ void mma_walk(FragAcc (&acc)[kMf][kNf], const bf16_bits* a, std::int64_t lda, int rows,
                                         const bf16_bits* const (&b)[kNf], std::int64_t ldb, const int (&b_cols)[kNf],
                                         int k_len, bf16_bits* pad) {
  for (int k0 = 0; k0 < k_len; k0 += kFrag) {
    const int ks = clamp_frag(k_len - k0);
    FragA fa[kMf];
    FragB fb[kNf];
    // All of a step's loads, then its WMMAs: they overlap in flight, where a load before each group of WMMAs stalls them.
    HIP_BLOCK_UNROLL
    for (int mf = 0; mf < kMf; ++mf) {
      load_a_padded(fa[mf], a + mf * kFrag * lda + k0, lda, clamp_frag(rows - mf * kFrag), ks, pad);
    }
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) load_b_padded(fb[nf], b[nf] + k0, ldb, b_cols[nf], ks, pad);
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) {
      HIP_BLOCK_UNROLL
      for (int mf = 0; mf < kMf; ++mf) leaf::mma(acc[mf][nf], fa[mf], fb[nf]);
    }
  }
}

// mma_walk on packed operands, with kg column groups per row group. a's fragments are row groups group_a + mf at column
// groups ks_a + step; b[nf]'s are row group b_group[nf] at ks_b + step. Tails read the packed zeros.
template <int kMf, int kNf>
__device__ __forceinline__ void mma_walk_packed(FragAcc (&acc)[kMf][kNf], const bf16_bits* a, std::int64_t group_a, std::int64_t ks_a,
                                                const bf16_bits* b, const std::int64_t (&b_group)[kNf], std::int64_t ks_b,
                                                std::int64_t kg, std::int64_t kg_b, int steps) {
  for (int step = 0; step < steps; ++step) {
    FragA fa[kMf];
    FragB fb[kNf];
    HIP_BLOCK_UNROLL
    for (int mf = 0; mf < kMf; ++mf) leaf::load_a(fa[mf], a + ((group_a + mf) * kg + ks_a + step) * (kFrag * kFrag), kFrag);
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) leaf::load_b(fb[nf], b + (b_group[nf] * kg_b + ks_b + step) * (kFrag * kFrag), kFrag);
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) {
      HIP_BLOCK_UNROLL
      for (int mf = 0; mf < kMf; ++mf) leaf::mma(acc[mf][nf], fa[mf], fb[nf]);
    }
  }
}

// The block order: workgroup idx to tile (gi, gj). Blocks clipped at the edges leave workgroups with nothing to do.
__device__ __forceinline__ bool tile_of(int idx, int gm, int gn, int bm, int bn, int& gi, int& gj) {
  const int per = bm * bn, block = idx / per, l = idx % per, across = (gm + bm - 1) / bm;
  gi = block % across * bm + l % bm;
  gj = block / across * bn + l / bm;
  return gi < gm && gj < gn;
}

// The tile's accumulators into row-major `mem` (stride kNf * kFrag). The caller synchronizes before reading.
template <int kMf, int kNf>
__device__ __forceinline__ void store_tile(float* mem, const FragAcc (&acc)[kMf][kNf]) {
  HIP_BLOCK_UNROLL
  for (int mf = 0; mf < kMf; ++mf) {
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) {
      leaf::store_acc(mem + mf * kFrag * kNf * kFrag + nf * kFrag, acc[mf][nf], kNf * kFrag);
    }
  }
}

// mean(x^2) -> rsqrt(mean + eps) of one 128-or-longer row, with the thread's four values already squared and summed:
// the shuffle-down tree of kernels/basic/norm.cu in __shared__ memory (lane t + off into lane t, t < off).
__device__ __forceinline__ float rsqrt_of_mean(float v, float factor, float* red) {
  const int t = thread_id();
  red[t] = v;
  leaf::barrier();
  HIP_BLOCK_UNROLL
  for (int off = kWave / 2; off > 0; off >>= 1) {
    if (t < off) red[t] = add(red[t], red[t + off]);
    leaf::barrier();
  }
  return rsqrt(add(mul(red[0], factor), kRmsNormEps));
}

// The GEMM kernels: one M x N tile per wave. Epilogues, ab the tile's fp32 sum:
//   kPlain          out = bf16(ab)
//   kGatedResidual  out = bf16(out + bf16(bf16(ab) * gate[n]))    (GatedResidualOp, in place)
//   kBias           out = bf16(ab + bias[n])                      (nn.Linear with a bias, cuBLASLt's bias epilogue)
//   kBiasResidual   out = bf16(out + bf16(ab + bias[n]))          (that linear, then a residual add, in place)
enum class Epilogue { kPlain, kGatedResidual, kBias, kBiasResidual };

struct GemmArgs {
  bf16_bits* out;
  std::int64_t ldo;
  const bf16_bits* x;
  std::int64_t ldx;
  const bf16_bits* w;
  std::int64_t ldw;
  const bf16_bits* gate;  // the gated residual's gate, or the bias
  int m, n, k;
  int bm, bn;            // block size in tiles
  const bf16_bits* xp;   // packed x and w; null: x and w row-major
  const bf16_bits* wp;
};

template <Epilogue kEpilogue>
__global__ __launch_bounds__(kWave) void gemm_kernel(const GemmArgs a) {
  constexpr int kMf = kGemmTileM / kFrag, kNf = kGemmTileN / kFrag;
  alignas(16) __shared__ float mem[kGemmTileM * kGemmTileN];
  alignas(16) __shared__ bf16_bits pad[kFrag * kFrag];
  int gi, gj;
  if (!tile_of(static_cast<int>(blockIdx.x), (a.m + kGemmTileM - 1) / kGemmTileM, (a.n + kGemmTileN - 1) / kGemmTileN, a.bm, a.bn, gi, gj)) return;
  const std::int64_t m0 = static_cast<std::int64_t>(gi) * kGemmTileM;
  const int n0 = gj * kGemmTileN;
  const int rows = static_cast<int>(a.m - m0 < kGemmTileM ? a.m - m0 : kGemmTileM);
  const int cols = a.n - n0 < kGemmTileN ? a.n - n0 : kGemmTileN;

  const bf16_bits* b[kNf];
  int b_cols[kNf];
  HIP_BLOCK_UNROLL
  for (int nf = 0; nf < kNf; ++nf) {
    b[nf] = a.w + static_cast<std::int64_t>(n0 + nf * kFrag) * a.ldw;
    b_cols[nf] = clamp_frag(cols - nf * kFrag);
  }
  FragAcc acc[kMf][kNf];
  clear_tile(acc);
  if (a.xp != nullptr) {
    std::int64_t b_group[kNf];
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) b_group[nf] = n0 / kFrag + nf;
    mma_walk_packed(acc, a.xp, m0 / kFrag, 0, a.wp, b_group, 0, a.k / kFrag, a.k / kFrag, a.k / kFrag);
  } else {
    mma_walk(acc, a.x + m0 * a.ldx, a.ldx, rows, b, a.ldw, b_cols, a.k, pad);
  }
  store_tile(mem, acc);
  leaf::barrier();

  for (int i = thread_id(); i < rows * cols; i += kWave) {
    const int r = i / cols, c = i % cols;
    const float ab = mem[r * kGemmTileN + c];
    bf16_bits* z = a.out + (m0 + r) * a.ldo + n0 + c;
    if constexpr (kEpilogue == Epilogue::kPlain) {
      *z = bf(ab);
    } else if constexpr (kEpilogue == Epilogue::kGatedResidual) {
      *z = bf(add(f(*z), rbf(mul(rbf(ab), f(a.gate[n0 + c])))));  // GatedResidualOp
    } else if constexpr (kEpilogue == Epilogue::kBias) {
      *z = bf(add(ab, f(a.gate[n0 + c])));
    } else {
      *z = bf(add(f(*z), rbf(add(ab, f(a.gate[n0 + c])))));
    }
  }
}

inline void check_launch(const char* op) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("hip_block::") + op + ": " + hipGetErrorString(e));
}

inline unsigned grid_of(std::int64_t items, int per_tile) { return static_cast<unsigned>((items + per_tile - 1) / per_tile); }

// The block order's sizes and the workgroups it takes, for a GEMM of m x n tiles over k.
struct Blocking {
  int bm, bn;
  unsigned workgroups;
};
// weight_cols: the columns of the weight a tile reads, which for SiTU is twice its tile_n.
inline Blocking blocking_of(int m, int n, int k, int tile_m, int tile_n, int weight_cols) {
  const int gm = static_cast<int>(grid_of(m, tile_m)), gn = static_cast<int>(grid_of(n, tile_n));
  const auto per = [&](int cols, int g) {
    const std::int64_t bytes = static_cast<std::int64_t>(cols) * k * static_cast<std::int64_t>(sizeof(bf16_bits));
    std::int64_t tiles = bytes == 0 ? g : kCacheBlockBytes / bytes;  // k == 0: no operand to keep in cache
    if (tiles > g) tiles = g;
    return static_cast<int>(tiles < 1 ? 1 : tiles);  // also when m == 0 leaves no tile
  };
  const int bm = per(tile_m, gm), bn = per(weight_cols, gn);
  return {bm, bn, static_cast<unsigned>((gm + bm - 1) / bm * ((gn + bn - 1) / bn) * bm * bn)};
}

}  // namespace
}  // namespace hip_block
