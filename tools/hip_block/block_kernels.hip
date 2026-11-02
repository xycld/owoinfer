// The block's kernels and their launches, one source for all three builds (hip_block.h, mma_leaf.h). The op order and
// the bf16 rounding points are those of kernels/basic/norm.cu, kernels/fused/qk_norm_rope.cu, the epilogue functors of
// kernels/fused/fused_gemm_impl.cuh and FlashAttention-2's compute_attn_1rowblock with softmax.h.
//
// Every workgroup is 32 threads: one wave on RDNA3/RDNA4, one warp on NVIDIA. Fragments are loaded straight from global
// memory; __shared__ memory holds what must cross lanes: the accumulators on their way out (and through the softmax's
// per-row rescale), per-row reductions, and the zero-padded copies of tails that end inside a fragment (a full-fragment
// load there would read outside the operand).
#include <hip/hip_runtime.h>

#include <cmath>
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
constexpr int kSituTileM = 48, kSituTileF = 16;  // F hidden columns and their F gate columns
static_assert(kPackRowsMultiple % kGemmTileM == 0 && kPackRowsMultiple % kSituTileM == 0);
// Tiles are walked in 2D blocks of about this many bytes per operand, so a block's operands stay in cache across its tiles.
// The order of the tiles does not change any output: within a tile K still ascends.
constexpr std::int64_t kCacheBlockBytes = 4 * 1000 * 1000;
constexpr int kAttnTileM = 16;                   // rows per wave; <= kWave (one lane per row in the softmax)
constexpr int kAttnStepN = kAttentionStepN;
static_assert(kGemmTileM % kFrag == 0 && kGemmTileN % kFrag == 0 && kSituTileM % kFrag == 0 && kSituTileF % kFrag == 0);
static_assert(kAttnTileM % kFrag == 0 && kAttnTileM <= kWave && kAttnStepN % kFrag == 0);
static_assert(kHeadDim % kFrag == 0);

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

// pack_fragments: lane t copies row t / 2, columns (t & 1) * 8 .. + 8 of one fragment.
using u32_alias = std::uint32_t __attribute__((may_alias));
__device__ __forceinline__ void copy_lane_of_fragment(bf16_bits* to, const bf16_bits* from, bool zero) {
  const u32_alias* f = reinterpret_cast<const u32_alias*>(from);
  u32_alias* d = reinterpret_cast<u32_alias*>(to);
  std::uint32_t v[4] = {0, 0, 0, 0};
  if (!zero) {
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 4; ++i) v[i] = f[i];
  }
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 4; ++i) d[i] = v[i];
}

struct PackArgs {
  bf16_bits* to;
  const bf16_bits* from;
  std::int64_t ld, rows;
  int kg;
};
__global__ __launch_bounds__(kWave) void pack_kernel(const PackArgs a) {
  const int t = thread_id(), r = t / 2, c0 = (t & 1) * 8;
  const std::int64_t ks = blockIdx.x, group = blockIdx.y, row = group * kFrag + r;  // along the row groups: 0.9 ms at K 8192, across them 23 ms
  copy_lane_of_fragment(a.to + (group * a.kg + ks) * (kFrag * kFrag) + r * kFrag + c0, a.from + row * a.ld + ks * kFrag + c0, row >= a.rows);
}

// Attention's q, k, v, per sequence: its row groups start at its first row, and it holds whole key steps of zeros at its tail.
__device__ __forceinline__ std::int64_t attn_group_of_seq(const std::int32_t* cu, int seq) {
  std::int64_t g = 0;
  for (int i = 0; i < seq; ++i) g += (cu[i + 1] - cu[i] + kAttentionStepN - 1) / kAttentionStepN * (kAttentionStepN / kFrag);
  return g;
}
struct PackAttnArgs {
  bf16_bits* to;
  const bf16_bits* from;
  std::int64_t ld;
  const std::int32_t* cu;
  int kg;
};
__global__ __launch_bounds__(kWave) void pack_attention_kernel(const PackAttnArgs a) {
  const int t = thread_id(), r = t / 2, c0 = (t & 1) * 8;
  const std::int64_t ks = blockIdx.x;
  const int group = static_cast<int>(blockIdx.y), seq = static_cast<int>(blockIdx.z);
  const int len = a.cu[seq + 1] - a.cu[seq];
  if (group >= (len + kAttentionStepN - 1) / kAttentionStepN * (kAttentionStepN / kFrag)) return;
  const int row = group * kFrag + r;
  const std::int64_t base = attn_group_of_seq(a.cu, seq);
  copy_lane_of_fragment(a.to + ((base + group) * a.kg + ks) * (kFrag * kFrag) + r * kFrag + c0,
                        a.from + (static_cast<std::int64_t>(a.cu[seq]) + row) * a.ld + ks * kFrag + c0, row >= len);
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

// rms_norm_modulate: one row per wave.
struct NormArgs {
  const bf16_bits* x;
  std::int64_t ldx;
  const bf16_bits* weight;
  const bf16_bits* scale;
  const bf16_bits* shift;
  bf16_bits* out;
  std::int64_t ldo;
  int d;
  float factor;  // torch MeanOps factor: float(rows) / float(rows * d)
};

__global__ __launch_bounds__(kWave) void rms_norm_modulate_kernel(const NormArgs a) {
  __shared__ float red[kWave];
  const int t = thread_id();
  const std::int64_t row = blockIdx.x;
  const bf16_bits* xr = a.x + row * a.ldx;

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int idx = t; idx * 4 + 3 < a.d; idx += kWave) {
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 4; ++i) {
      const float v = f(xr[idx * 4 + i]);
      acc[i] = add(acc[i], mul(v, v));
    }
  }
  const float r = rsqrt_of_mean(add(add(add(acc[0], acc[1]), acc[2]), acc[3]), a.factor, red);

  bf16_bits* outr = a.out + row * a.ldo;
  for (int c = t; c < a.d; c += kWave) {
    const float y = mul(rbf(mul(f(xr[c]), r)), f(a.weight[c]));
    const float one_plus = rbf(add(1.0f, f(a.scale[c])));
    outr[c] = bf(add(rbf(mul(rbf(y), one_plus)), f(a.shift[c])));
  }
}

// qk_norm_rope: one (token, q-or-k head) per wave. Lane l holds dims 4l..4l+3; RoPE pairs dim d < 64 with d + 64, which
// on NVIDIA is a shuffle and here goes through __shared__ memory.
struct QkArgs {
  bf16_bits* qkv;
  std::int64_t ld;
  const bf16_bits* q_norm;
  const bf16_bits* k_norm;
  const bf16_bits* table;
  std::int64_t table_ld;
  int heads;
  float factor;  // torch MeanOps factor of the per-head norm
};

__global__ __launch_bounds__(kWave) void qk_norm_rope_kernel(const QkArgs a) {
  __shared__ float red[kWave];
  __shared__ float norm[kHeadDim];
  const int lane = thread_id();
  const std::int64_t item = blockIdx.x;
  const std::int64_t token = item / (2 * a.heads);
  const int head = static_cast<int>(item % (2 * a.heads));  // q heads first, then k heads (columns head * 128)
  bf16_bits* row = a.qkv + token * a.ld + static_cast<std::int64_t>(head) * kHeadDim + 4 * lane;
  const bf16_bits* weight = (head < a.heads ? a.q_norm : a.k_norm) + 4 * lane;

  float x[4], acc[4];
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 4; ++i) {
    x[i] = f(row[i]);
    acc[i] = mul(x[i], x[i]);
  }
  const float r = rsqrt_of_mean(add(add(add(acc[0], acc[1]), acc[2]), acc[3]), a.factor, red);

  float y[4];
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 4; ++i) {
    y[i] = rbf(mul(rbf(mul(x[i], r)), f(weight[i])));  // rms_norm_moss output (bf16)
    norm[4 * lane + i] = y[i];
  }
  leaf::barrier();

  // Table row [sin(64) | cos(64)]; pair index t = dim mod 64.
  const bf16_bits* tab = a.table + token * a.table_ld + 4 * (lane & 15);
  const bool lo_half = lane < 16;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 4; ++i) {
    const float other = norm[(4 * lane + i) ^ 64];
    const float si = f(tab[i]), co = f(tab[64 + i]);
    const float lo = lo_half ? y[i] : other, hi = lo_half ? other : y[i];
    // moss_rope_apply's rotate_pair: lo' = lo*cos + (-hi)*sin, hi' = hi*cos + lo*sin, each op rounded.
    row[i] = lo_half ? bf(add(rbf(mul(lo, co)), rbf(mul(-hi, si)))) : bf(add(rbf(mul(hi, co)), rbf(mul(lo, si))));
  }
}

// The GEMM kernels: one M x N tile per wave.
enum class Epilogue { kPlain, kGatedResidual };

struct GemmArgs {
  bf16_bits* out;
  std::int64_t ldo;
  const bf16_bits* x;
  std::int64_t ldx;
  const bf16_bits* w;
  std::int64_t ldw;
  const bf16_bits* gate;
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
    } else {
      *z = bf(add(f(*z), rbf(mul(rbf(ab), f(a.gate[n0 + c])))));  // GatedResidualOp
    }
  }
}

// MOSS _situ (P:66-67) on bf16 values: the bf16 gate factor of g times the bf16 hidden factor of h, in fp32.
__device__ __forceinline__ float situ_value(float h, float g) {
  constexpr float kInv25 = 1.0f / 25.0f;  // torch divides by a CPU scalar as a multiply by its fp32 reciprocal
  const float left = rbf(mul(rbf(mul(4.0f, rbf(tanhf(rbf(mul(g, 0.25f)))))), rbf(sigmoid_f32(g))));
  const float right = rbf(mul(25.0f, rbf(tanhf(rbf(mul(h, kInv25))))));
  return mul(left, right);
}

// One tile of F hidden columns and their F gate columns per wave; the accumulator memory holds each row as
// [hidden | gate] (SituGluOp without its 8-column groups: its grouping served the NVIDIA epilogue's accesses).
__global__ __launch_bounds__(kWave) void gemm_situ_kernel(const GemmArgs a) {
  constexpr int kMf = kSituTileM / kFrag, kFf = kSituTileF / kFrag, kNf = 2 * kFf;
  constexpr int kMemCols = kNf * kFrag;
  alignas(16) __shared__ float mem[kSituTileM * kMemCols];
  alignas(16) __shared__ bf16_bits pad[kFrag * kFrag];
  int gi, gj;
  if (!tile_of(static_cast<int>(blockIdx.x), (a.m + kSituTileM - 1) / kSituTileM, (a.n + kSituTileF - 1) / kSituTileF, a.bm, a.bn, gi, gj)) return;
  const std::int64_t m0 = static_cast<std::int64_t>(gi) * kSituTileM;
  const int f0 = gj * kSituTileF;
  const int rows = static_cast<int>(a.m - m0 < kSituTileM ? a.m - m0 : kSituTileM);
  const int cols = a.n - f0 < kSituTileF ? a.n - f0 : kSituTileF;

  const bf16_bits* b[kNf];
  int b_cols[kNf];
  HIP_BLOCK_UNROLL
  for (int nf = 0; nf < kFf; ++nf) {
    b[nf] = a.w + static_cast<std::int64_t>(f0 + nf * kFrag) * a.ldw;
    b[kFf + nf] = a.w + static_cast<std::int64_t>(a.n + f0 + nf * kFrag) * a.ldw;
    b_cols[nf] = b_cols[kFf + nf] = clamp_frag(cols - nf * kFrag);
  }
  FragAcc acc[kMf][kNf];
  clear_tile(acc);
  if (a.xp != nullptr) {
    std::int64_t b_group[kNf];
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kFf; ++nf) {
      b_group[nf] = f0 / kFrag + nf;
      b_group[kFf + nf] = (a.n + f0) / kFrag + nf;
    }
    mma_walk_packed(acc, a.xp, m0 / kFrag, 0, a.wp, b_group, 0, a.k / kFrag, a.k / kFrag, a.k / kFrag);
  } else {
    mma_walk(acc, a.x + m0 * a.ldx, a.ldx, rows, b, a.ldw, b_cols, a.k, pad);
  }
  store_tile(mem, acc);
  leaf::barrier();

  for (int i = thread_id(); i < rows * cols; i += kWave) {
    const int r = i / cols, c = i % cols;
    // The GEMM output is a bf16 tensor in the reference: round both halves before SiTU.
    const float h = rbf(mem[r * kMemCols + c]), g = rbf(mem[r * kMemCols + kSituTileF + c]);
    a.out[(m0 + r) * a.ldo + f0 + c] = bf(situ_value(h, g));
  }
}

// Attention: one tile of kAttnTileM query rows of one head of one sequence per wave. Key tokens are walked in steps of
// kAttnStepN from the last step down to step 0, as FlashAttention's reverse iteration:
//   S = Q · K_stepᵀ
//   row_max, row_sum and acc_o rescaled to the new maximum
//   P = bf16(exp2(S * log2_scale - max * log2_scale)), row_sum += P before rounding
//   acc_o += P · V_stepᵀ
// and at the end out = bf16(acc_o / row_sum). Key tokens past the sequence's end have P = 0, as the kernel's mask.
struct AttnArgs {
  const bf16_bits* q;
  const bf16_bits* k;
  const bf16_bits* v;
  bf16_bits* o;
  std::int64_t ldq, ldk, ldv, ldo;
  std::int64_t hsq, hsk, hsv, hso;
  const std::int32_t* cu_q;
  const std::int32_t* cu_k;
  float scale_log2;
  const bf16_bits* packed;
};

__global__ __launch_bounds__(kWave) void attention_kernel(const AttnArgs a) {
  constexpr int kMf = kAttnTileM / kFrag, kSf = kAttnStepN / kFrag, kDf = kHeadDim / kFrag;
  alignas(16) __shared__ float s_mem[kAttnTileM * kAttnStepN];
  alignas(16) __shared__ bf16_bits p_mem[kAttnTileM * kAttnStepN];
  alignas(16) __shared__ float o_mem[kAttnTileM * kHeadDim];
  alignas(16) __shared__ float w_mem[kAttnTileM * kAttnStepN];
  alignas(16) __shared__ bf16_bits pad[kFrag * kFrag];
  __shared__ float row_max[kAttnTileM], row_sum[kAttnTileM], row_scale[kAttnTileM], row_max_scaled[kAttnTileM];

  const int t = thread_id();
  const int m_block = static_cast<int>(blockIdx.x), seq = static_cast<int>(blockIdx.y), head = static_cast<int>(blockIdx.z);
  const std::int64_t q_row0 = static_cast<std::int64_t>(a.cu_q[seq]) + static_cast<std::int64_t>(m_block) * kAttnTileM;
  const int seqlen_q = a.cu_q[seq + 1] - a.cu_q[seq];
  const std::int64_t k_row0 = a.cu_k[seq];
  const int seqlen_k = a.cu_k[seq + 1] - a.cu_k[seq];
  if (m_block * kAttnTileM >= seqlen_q) return;
  const int rows = seqlen_q - m_block * kAttnTileM < kAttnTileM ? seqlen_q - m_block * kAttnTileM : kAttnTileM;
  const int steps = (seqlen_k + kAttnStepN - 1) / kAttnStepN;

  const bf16_bits* q = a.q + q_row0 * a.ldq + head * a.hsq;
  const bf16_bits* k_head = a.k + k_row0 * a.ldk + head * a.hsk;
  const bf16_bits* v_head = a.v + k_row0 * a.ldv + head * a.hsv;
  // Packed: row groups of this sequence, columns in fragments of q, k and v as run_block_op packs them.
  constexpr std::int64_t kKg = 3 * kDim / kFrag, kHeadFrags = kHeadDim / kFrag;
  const std::int64_t base_group = a.packed != nullptr ? attn_group_of_seq(a.cu_q, seq) : 0;
  const std::int64_t q_ks = head * kHeadFrags, k_ks = kDim / kFrag + q_ks, v_ks = 2 * kDim / kFrag + q_ks;

  FragAcc acc_o[kMf][kDf];
  clear_tile(acc_o);
  if (t < kAttnTileM) row_sum[t] = 0.0f;  // a sequence without keys: out = 0

  for (int step = steps - 1; step >= 0; --step) {
    const bool first = step == steps - 1;
    const int n0 = step * kAttnStepN;
    const int cols = seqlen_k - n0 < kAttnStepN ? seqlen_k - n0 : kAttnStepN;

    const bf16_bits* b[kSf];
    int b_cols[kSf];
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kSf; ++nf) {
      b[nf] = k_head + static_cast<std::int64_t>(n0 + nf * kFrag) * a.ldk;
      b_cols[nf] = clamp_frag(cols - nf * kFrag);
    }
    FragAcc acc_s[kMf][kSf];
    clear_tile(acc_s);
    if (a.packed != nullptr) {
      std::int64_t b_group[kSf];
      HIP_BLOCK_UNROLL
      for (int nf = 0; nf < kSf; ++nf) b_group[nf] = base_group + step * kSf + nf;
      mma_walk_packed(acc_s, a.packed, base_group + m_block, q_ks, a.packed, b_group, k_ks, kKg, kKg, kDf);
    } else {
      mma_walk(acc_s, q, a.ldq, rows, b, a.ldk, b_cols, kHeadDim, pad);
    }
    store_tile(s_mem, acc_s);
    leaf::barrier();

    // softmax_rescale_o. One lane per row: this row's new maximum and the rescale of its running sum.
    if (t < rows) {
      const float* s = s_mem + t * kAttnStepN;
      const float prev = row_max[t];
      float cur = first ? s[0] : fmaxf(prev, s[0]);
      // Four values per read: a read from __shared__ memory takes tens of clocks, and these loops are most of a step.
      int c1 = 1;
      for (; c1 < cols && c1 % 4 != 0; ++c1) cur = fmaxf(cur, s[c1]);
      for (; c1 + 4 <= cols; c1 += 4) {
        const float v0 = s[c1], v1 = s[c1 + 1], v2 = s[c1 + 2], v3 = s[c1 + 3];
        cur = fmaxf(fmaxf(fmaxf(fmaxf(cur, v0), v1), v2), v3);
      }
      for (; c1 < cols; ++c1) cur = fmaxf(cur, s[c1]);
      row_max[t] = cur;
      float sum = 0.0f;
      if (!first) {
        const float rescale = exp2f(mul(prev - cur, a.scale_log2));
        row_scale[t] = rescale;
        sum = mul(row_sum[t], rescale);
      }
      row_sum[t] = sum;
      row_max_scaled[t] = cur == -INFINITY ? 0.0f : mul(cur, a.scale_log2);
    }
    leaf::barrier();
    // The weights on every lane, then each row's sum on one lane, in the order of its columns.
    for (int i = t; i < kAttnTileM * kAttnStepN; i += kWave) {
      const int r = i / kAttnStepN, c = i % kAttnStepN;
      if (r < rows && c < cols) {
        const float w = exp2f(fmaf(s_mem[i], a.scale_log2, -row_max_scaled[r]));
        w_mem[i] = w;
        p_mem[i] = bf(w);
      } else {
        p_mem[i] = 0;
      }
    }
    leaf::barrier();
    if (t < rows) {
      float sum = row_sum[t];
      const float* w_row = w_mem + t * kAttnStepN;
      int c2 = 0;
      for (; c2 + 4 <= cols; c2 += 4) {
        const float v0 = w_row[c2], v1 = w_row[c2 + 1], v2 = w_row[c2 + 2], v3 = w_row[c2 + 3];
        sum = add(add(add(add(sum, v0), v1), v2), v3);
      }
      for (; c2 < cols; ++c2) sum = add(sum, w_row[c2]);
      row_sum[t] = sum;
    }
    leaf::barrier();

    // The rescale of acc_o needs each element's row, which a fragment does not expose: through memory and back. A row whose
    // maximum did not change has a rescale of exactly 1, and if every row's is, there is nothing to do.
    bool rescale_any = false;
    if (!first) {
      HIP_BLOCK_UNROLL
      for (int r = 0; r < kAttnTileM; ++r) {
        if (r < rows && row_scale[r] != 1.0f) rescale_any = true;
      }
    }
    if (rescale_any) {
      store_tile(o_mem, acc_o);
      leaf::barrier();
      for (int i = t; i < rows * kHeadDim; i += kWave) o_mem[i] = mul(o_mem[i], row_scale[i / kHeadDim]);
      leaf::barrier();
      HIP_BLOCK_UNROLL
      for (int mf = 0; mf < kMf; ++mf) {
        HIP_BLOCK_UNROLL
        for (int df = 0; df < kDf; ++df) {
          leaf::load_acc(acc_o[mf][df], o_mem + mf * kFrag * kHeadDim + df * kFrag, kHeadDim);
        }
      }
    }

    // acc_o += P · V_stepᵀ, K (the step's key tokens) ascending. Fragments past the sequence's end have P = 0.
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kSf; ++nf) {
      const int ks = clamp_frag(cols - nf * kFrag);
      if (ks == 0) continue;
      FragA fp[kMf];
      HIP_BLOCK_UNROLL
      for (int mf = 0; mf < kMf; ++mf) {
        leaf::load_a(fp[mf], p_mem + mf * kFrag * kAttnStepN + nf * kFrag, kAttnStepN);
      }
      const bf16_bits* v_step = v_head + static_cast<std::int64_t>(n0 + nf * kFrag) * a.ldv;
      FragB fb[kDf];
      HIP_BLOCK_UNROLL
      for (int df = 0; df < kDf; ++df) {
        if (a.packed != nullptr) {
          leaf::load_b_rows(fb[df], a.packed + ((base_group + step * kSf + nf) * kKg + v_ks + df) * (kFrag * kFrag), kFrag);
        } else {
          load_b_rows_padded(fb[df], v_step + df * kFrag, a.ldv, ks, pad);
        }
      }
      HIP_BLOCK_UNROLL
      for (int df = 0; df < kDf; ++df) {
        HIP_BLOCK_UNROLL
        for (int mf = 0; mf < kMf; ++mf) leaf::mma(acc_o[mf][df], fp[mf], fb[df]);
      }
    }
  }

  // normalize_softmax_lse, without the log-sum-exp.
  store_tile(o_mem, acc_o);
  if (t < rows) {
    const float sum = row_sum[t];
    row_scale[t] = sum == 0.0f || sum != sum ? 1.0f : div(1.0f, sum);
  }
  leaf::barrier();
  bf16_bits* o = a.o + q_row0 * a.ldo + head * a.hso;
  for (int i = t; i < rows * kHeadDim; i += kWave) {
    const int r = i / kHeadDim, c = i % kHeadDim;
    o[r * a.ldo + c] = bf(mul(o_mem[i], row_scale[r]));
  }
}

void check_launch(const char* op) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("hip_block::") + op + ": " + hipGetErrorString(e));
}

unsigned grid_of(std::int64_t items, int per_tile) { return static_cast<unsigned>((items + per_tile - 1) / per_tile); }

// The block order's sizes and the workgroups it takes, for a GEMM of m x n tiles over k.
struct Blocking {
  int bm, bn;
  unsigned workgroups;
};
// weight_cols: the columns of the weight a tile reads, which for SiTU is twice its tile_n.
Blocking blocking_of(int m, int n, int k, int tile_m, int tile_n, int weight_cols) {
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

const char* leaf_name() { return leaf::kName; }

void pack_fragments(bf16_bits* to, const bf16_bits* from, std::int64_t ld, std::int64_t rows, std::int64_t rows_padded, int cols,
                    hipStream_t stream) {
  if (rows_padded == 0 || cols == 0) return;
  if (cols % kFrag != 0) throw std::runtime_error("pack_fragments: cols must be a multiple of 16");
  const PackArgs a{to, from, ld, rows, cols / kFrag};
  pack_kernel<<<dim3(static_cast<unsigned>(cols / kFrag), grid_of(rows_padded, kFrag)), dim3(kWave), 0, stream>>>(a);
  check_launch("pack_fragments");
}

void rms_norm_modulate(const bf16_bits* x, std::int64_t ldx, const bf16_bits* weight, const bf16_bits* scale,
                       const bf16_bits* shift, bf16_bits* out, std::int64_t ldo, std::int64_t rows, int d,
                       hipStream_t stream) {
  if (rows == 0) return;
  const NormArgs a{x, ldx, weight, scale, shift, out, ldo, d,
                   static_cast<float>(rows) / static_cast<float>(rows * d)};
  rms_norm_modulate_kernel<<<dim3(static_cast<unsigned>(rows)), dim3(kWave), 0, stream>>>(a);
  check_launch("rms_norm_modulate");
}

void gemm(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w, std::int64_t ldw,
          int m, int n, int k, hipStream_t stream) {
  if (m == 0 || n == 0) return;
  const Blocking bl = blocking_of(m, n, k, kGemmTileM, kGemmTileN, kGemmTileN);
  const GemmArgs a{out, ldo, x, ldx, w, ldw, nullptr, m, n, k, bl.bm, bl.bn, nullptr, nullptr};
  gemm_kernel<Epilogue::kPlain><<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm");
}

void qk_norm_rope(bf16_bits* qkv, std::int64_t ld, std::int64_t tokens, int heads, const bf16_bits* q_norm,
                  const bf16_bits* k_norm, const bf16_bits* table, std::int64_t table_ld, hipStream_t stream) {
  if (tokens == 0) return;
  // rms_norm_moss on a [L, heads, 128] head view sees L * heads rows.
  const std::int64_t rows = tokens * heads;
  const QkArgs a{qkv, ld, q_norm, k_norm, table, table_ld, heads,
                 static_cast<float>(rows) / static_cast<float>(rows * kHeadDim)};
  qk_norm_rope_kernel<<<dim3(static_cast<unsigned>(tokens * 2 * heads)), dim3(kWave), 0, stream>>>(a);
  check_launch("qk_norm_rope");
}

void attention(const AttentionArgs& p, hipStream_t stream) {
  constexpr double kLog2e = 1.4426950408889634074;
  // In double then rounded, as upstream (float * M_LOG2E): keeps the exp2 argument bit-identical.
  const AttnArgs a{p.q,   p.k,   p.v,   p.o,   p.ldq,  p.ldk,  p.ldv,
                   p.ldo, p.hsq, p.hsk, p.hsv, p.hso,  p.cu_q, p.cu_k,
                   static_cast<float>(static_cast<double>(p.scale) * kLog2e), p.packed};
  const dim3 grid(grid_of(p.max_seqlen, kAttnTileM), static_cast<unsigned>(p.batch), static_cast<unsigned>(p.heads));
  attention_kernel<<<grid, dim3(kWave), 0, stream>>>(a);
  check_launch("attention");
}

void gemm_gated_residual(bf16_bits* residual, std::int64_t ldr, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w,
                         std::int64_t ldw, const bf16_bits* gate, int m, int n, int k, hipStream_t stream) {
  if (m == 0 || n == 0) return;
  const Blocking bl = blocking_of(m, n, k, kGemmTileM, kGemmTileN, kGemmTileN);
  const GemmArgs a{residual, ldr, x, ldx, w, ldw, gate, m, n, k, bl.bm, bl.bn, nullptr, nullptr};
  gemm_kernel<Epilogue::kGatedResidual><<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm_gated_residual");
}

void gemm_situ_glu(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w_up_gate,
                   std::int64_t ldw, int m, int f, int k, hipStream_t stream) {
  if (m == 0 || f == 0) return;
  const Blocking bl = blocking_of(m, f, k, kSituTileM, kSituTileF, 2 * kSituTileF);
  const GemmArgs a{out, ldo, x, ldx, w_up_gate, ldw, nullptr, m, f, k, bl.bm, bl.bn, nullptr, nullptr};
  gemm_situ_kernel<<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm_situ_glu");
}

const char* block_op_name(int op) {
  static constexpr const char* kNames[kBlockOps] = {"norm1+modulate",   "qkv gemm",       "qk_norm_rope",      "attention",
                                                    "o gemm+gated_res", "norm2+modulate", "up_gate gemm+situ", "down gemm+gated_res"};
  return kNames[op];
}

namespace {

// The GEMM ops on packed operands. n is a multiple of the tile's N for all of them, so no tile runs past the weight.
std::int64_t pack_rows(int tokens) { return (static_cast<std::int64_t>(tokens) + kPackRowsMultiple - 1) / kPackRowsMultiple * kPackRowsMultiple; }

void packed_gemm_plain(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* wp, const BlockBuffers& b, int n, int k, hipStream_t stream) {
  if (b.tokens == 0) return;
  pack_fragments(b.pack, x, ldx, b.tokens, pack_rows(b.tokens), k, stream);
  const Blocking bl = blocking_of(b.tokens, n, k, kGemmTileM, kGemmTileN, kGemmTileN);
  const GemmArgs a{out, ldo, nullptr, 0, nullptr, 0, nullptr, b.tokens, n, k, bl.bm, bl.bn, b.pack, wp};
  gemm_kernel<Epilogue::kPlain><<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm (packed)");
}
void packed_gemm_gated(bf16_bits* residual, std::int64_t ldr, const bf16_bits* x, std::int64_t ldx, const bf16_bits* wp, const bf16_bits* gate, const BlockBuffers& b, int n, int k, hipStream_t stream) {
  if (b.tokens == 0) return;
  pack_fragments(b.pack, x, ldx, b.tokens, pack_rows(b.tokens), k, stream);
  const Blocking bl = blocking_of(b.tokens, n, k, kGemmTileM, kGemmTileN, kGemmTileN);
  const GemmArgs a{residual, ldr, nullptr, 0, nullptr, 0, gate, b.tokens, n, k, bl.bm, bl.bn, b.pack, wp};
  gemm_kernel<Epilogue::kGatedResidual><<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm_gated_residual (packed)");
}
void packed_gemm_situ(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* wp, const BlockBuffers& b, int f, int k, hipStream_t stream) {
  if (b.tokens == 0) return;
  pack_fragments(b.pack, x, ldx, b.tokens, pack_rows(b.tokens), k, stream);
  const Blocking bl = blocking_of(b.tokens, f, k, kSituTileM, kSituTileF, 2 * kSituTileF);
  const GemmArgs a{out, ldo, nullptr, 0, nullptr, 0, nullptr, b.tokens, f, k, bl.bm, bl.bn, b.pack, wp};
  gemm_situ_kernel<<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm_situ_glu (packed)");
}
void pack_attention(const BlockBuffers& b, hipStream_t stream) {
  const PackAttnArgs a{b.pack_attn, b.qkv, 3 * kDim, b.cu_seqlens, 3 * kDim / kFrag};
  const unsigned groups = static_cast<unsigned>((b.max_seqlen + kAttentionStepN - 1) / kAttentionStepN * (kAttentionStepN / kFrag));
  pack_attention_kernel<<<dim3(3 * kDim / kFrag, groups, static_cast<unsigned>(b.batch)), dim3(kWave), 0, stream>>>(a);
  check_launch("pack_attention");
}

}  // namespace

void run_block_op(int op, const BlockWeights& w, const BlockBuffers& b, hipStream_t stream) {
  constexpr std::int64_t ldx = kDim, ldq = 3 * kDim;
  const bool packed = w.qkv_packed != nullptr && b.pack != nullptr && b.pack_attn != nullptr;
  const auto mod = [&](int i) { return b.mod + static_cast<std::int64_t>(i) * kDim; };
  const bf16_bits* k = b.qkv + kDim;
  const bf16_bits* v = b.qkv + 2 * kDim;
  switch (op) {
    case 0: rms_norm_modulate(b.x, ldx, w.norm1, mod(1), mod(0), b.y, ldx, b.tokens, kDim, stream); break;
    case 1:
      if (packed) {
        packed_gemm_plain(b.qkv, ldq, b.y, ldx, w.qkv_packed, b, 3 * kDim, kDim, stream);
      } else {
        gemm(b.qkv, ldq, b.y, ldx, w.qkv, kDim, b.tokens, 3 * kDim, kDim, stream);
      }
      break;
    case 2: qk_norm_rope(b.qkv, ldq, b.tokens, kHeads, w.q_norm, w.k_norm, b.rope, kHeadDim, stream); break;
    case 3:
      if (packed) pack_attention(b, stream);
      attention({b.qkv, k, v, b.attn, ldq, ldq, ldq, ldx, kHeadDim, kHeadDim, kHeadDim, kHeadDim, b.cu_seqlens, b.cu_seqlens,
                 b.batch, b.max_seqlen, kHeads, static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim))), packed ? b.pack_attn : nullptr},
                stream);
      break;
    case 4:
      if (packed) {
        packed_gemm_gated(b.x, ldx, b.attn, ldx, w.o_packed, mod(2), b, kDim, kDim, stream);
      } else {
        gemm_gated_residual(b.x, ldx, b.attn, ldx, w.o, kDim, mod(2), b.tokens, kDim, kDim, stream);
      }
      break;
    case 5: rms_norm_modulate(b.x, ldx, w.norm2, mod(4), mod(3), b.y, ldx, b.tokens, kDim, stream); break;
    case 6:
      if (packed) {
        packed_gemm_situ(b.ffn, kFfnDim, b.y, ldx, w.up_gate_packed, b, kFfnDim, kDim, stream);
      } else {
        gemm_situ_glu(b.ffn, kFfnDim, b.y, ldx, w.up_gate, kDim, b.tokens, kFfnDim, kDim, stream);
      }
      break;
    default:
      if (packed) {
        packed_gemm_gated(b.x, ldx, b.ffn, kFfnDim, w.down_packed, mod(5), b, kDim, kFfnDim, stream);
      } else {
        gemm_gated_residual(b.x, ldx, b.ffn, kFfnDim, w.down, kFfnDim, mod(5), b.tokens, kDim, kFfnDim, stream);
      }
      break;
  }
}

void run_block(const BlockWeights& w, const BlockBuffers& b, hipStream_t stream) {
  for (int op = 0; op < kBlockOps; ++op) run_block_op(op, w, b, stream);
}

}  // namespace hip_block
