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
constexpr int kSituTileM = 32, kSituTileF = 16;  // F hidden columns and their F gate columns
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
  __syncthreads();
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
  __syncthreads();
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
  __syncthreads();
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
    HIP_BLOCK_UNROLL
    for (int mf = 0; mf < kMf; ++mf) {
      load_a_padded(fa[mf], a + mf * kFrag * lda + k0, lda, clamp_frag(rows - mf * kFrag), ks, pad);
    }
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kNf; ++nf) {
      FragB fb;
      load_b_padded(fb, b[nf] + k0, ldb, b_cols[nf], ks, pad);
      HIP_BLOCK_UNROLL
      for (int mf = 0; mf < kMf; ++mf) leaf::mma(acc[mf][nf], fa[mf], fb);
    }
  }
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
  __syncthreads();
  HIP_BLOCK_UNROLL
  for (int off = kWave / 2; off > 0; off >>= 1) {
    if (t < off) red[t] = add(red[t], red[t + off]);
    __syncthreads();
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
  __syncthreads();

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
};

template <Epilogue kEpilogue>
__global__ __launch_bounds__(kWave) void gemm_kernel(const GemmArgs a) {
  constexpr int kMf = kGemmTileM / kFrag, kNf = kGemmTileN / kFrag;
  alignas(16) __shared__ float mem[kGemmTileM * kGemmTileN];
  alignas(16) __shared__ bf16_bits pad[kFrag * kFrag];
  const std::int64_t m0 = static_cast<std::int64_t>(blockIdx.x) * kGemmTileM;
  const int n0 = static_cast<int>(blockIdx.y) * kGemmTileN;
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
  mma_walk(acc, a.x + m0 * a.ldx, a.ldx, rows, b, a.ldw, b_cols, a.k, pad);
  store_tile(mem, acc);
  __syncthreads();

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
  const std::int64_t m0 = static_cast<std::int64_t>(blockIdx.x) * kSituTileM;
  const int f0 = static_cast<int>(blockIdx.y) * kSituTileF;
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
  mma_walk(acc, a.x + m0 * a.ldx, a.ldx, rows, b, a.ldw, b_cols, a.k, pad);
  store_tile(mem, acc);
  __syncthreads();

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
};

__global__ __launch_bounds__(kWave) void attention_kernel(const AttnArgs a) {
  constexpr int kMf = kAttnTileM / kFrag, kSf = kAttnStepN / kFrag, kDf = kHeadDim / kFrag;
  alignas(16) __shared__ float s_mem[kAttnTileM * kAttnStepN];
  alignas(16) __shared__ bf16_bits p_mem[kAttnTileM * kAttnStepN];
  alignas(16) __shared__ float o_mem[kAttnTileM * kHeadDim];
  alignas(16) __shared__ bf16_bits pad[kFrag * kFrag];
  __shared__ float row_max[kAttnTileM], row_sum[kAttnTileM], row_scale[kAttnTileM];

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
    mma_walk(acc_s, q, a.ldq, rows, b, a.ldk, b_cols, kHeadDim, pad);
    store_tile(s_mem, acc_s);
    if (!first) store_tile(o_mem, acc_o);
    __syncthreads();

    // softmax_rescale_o, one lane per row: this row's new maximum, the rescale of its running sum, its weights.
    if (t < kAttnTileM) {
      if (t < rows) {
        const float* s = s_mem + t * kAttnStepN;
        const float prev = row_max[t];
        float cur = first ? s[0] : fmaxf(prev, s[0]);
        for (int c = 1; c < cols; ++c) cur = fmaxf(cur, s[c]);
        row_max[t] = cur;
        float sum = 0.0f;
        if (!first) {
          const float rescale = exp2f(mul(prev - cur, a.scale_log2));
          row_scale[t] = rescale;
          sum = mul(row_sum[t], rescale);
        }
        const float max_scaled = cur == -INFINITY ? 0.0f : mul(cur, a.scale_log2);
        for (int c = 0; c < cols; ++c) {
          const float w = exp2f(fmaf(s[c], a.scale_log2, -max_scaled));
          sum = add(sum, w);
          p_mem[t * kAttnStepN + c] = bf(w);
        }
        for (int c = cols; c < kAttnStepN; ++c) p_mem[t * kAttnStepN + c] = 0;
        row_sum[t] = sum;
      } else {
        for (int c = 0; c < kAttnStepN; ++c) p_mem[t * kAttnStepN + c] = 0;
      }
    }
    __syncthreads();

    // The rescale of acc_o needs each element's row, which a fragment does not expose: through memory and back.
    if (!first) {
      for (int i = t; i < rows * kHeadDim; i += kWave) o_mem[i] = mul(o_mem[i], row_scale[i / kHeadDim]);
      __syncthreads();
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
      HIP_BLOCK_UNROLL
      for (int df = 0; df < kDf; ++df) {
        FragB fb;
        load_b_rows_padded(fb, v_step + df * kFrag, a.ldv, ks, pad);
        HIP_BLOCK_UNROLL
        for (int mf = 0; mf < kMf; ++mf) leaf::mma(acc_o[mf][df], fp[mf], fb);
      }
    }
  }

  // normalize_softmax_lse, without the log-sum-exp.
  store_tile(o_mem, acc_o);
  if (t < rows) {
    const float sum = row_sum[t];
    row_scale[t] = sum == 0.0f || sum != sum ? 1.0f : div(1.0f, sum);
  }
  __syncthreads();
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

}  // namespace

const char* leaf_name() { return leaf::kName; }

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
  const GemmArgs a{out, ldo, x, ldx, w, ldw, nullptr, m, n, k};
  gemm_kernel<Epilogue::kPlain><<<dim3(grid_of(m, kGemmTileM), grid_of(n, kGemmTileN)), dim3(kWave), 0, stream>>>(a);
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
                   static_cast<float>(static_cast<double>(p.scale) * kLog2e)};
  const dim3 grid(grid_of(p.max_seqlen, kAttnTileM), static_cast<unsigned>(p.batch), static_cast<unsigned>(p.heads));
  attention_kernel<<<grid, dim3(kWave), 0, stream>>>(a);
  check_launch("attention");
}

void gemm_gated_residual(bf16_bits* residual, std::int64_t ldr, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w,
                         std::int64_t ldw, const bf16_bits* gate, int m, int n, int k, hipStream_t stream) {
  if (m == 0 || n == 0) return;
  const GemmArgs a{residual, ldr, x, ldx, w, ldw, gate, m, n, k};
  gemm_kernel<Epilogue::kGatedResidual><<<dim3(grid_of(m, kGemmTileM), grid_of(n, kGemmTileN)), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm_gated_residual");
}

void gemm_situ_glu(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w_up_gate,
                   std::int64_t ldw, int m, int f, int k, hipStream_t stream) {
  if (m == 0 || f == 0) return;
  const GemmArgs a{out, ldo, x, ldx, w_up_gate, ldw, nullptr, m, f, k};
  gemm_situ_kernel<<<dim3(grid_of(m, kSituTileM), grid_of(f, kSituTileF)), dim3(kWave), 0, stream>>>(a);
  check_launch("gemm_situ_glu");
}

const char* block_op_name(int op) {
  static constexpr const char* kNames[kBlockOps] = {"norm1+modulate",   "qkv gemm",       "qk_norm_rope",      "attention",
                                                    "o gemm+gated_res", "norm2+modulate", "up_gate gemm+situ", "down gemm+gated_res"};
  return kNames[op];
}

void run_block_op(int op, const BlockWeights& w, const BlockBuffers& b, hipStream_t stream) {
  constexpr std::int64_t ldx = kDim, ldq = 3 * kDim;
  const auto mod = [&](int i) { return b.mod + static_cast<std::int64_t>(i) * kDim; };
  const bf16_bits* k = b.qkv + kDim;
  const bf16_bits* v = b.qkv + 2 * kDim;
  switch (op) {
    case 0: rms_norm_modulate(b.x, ldx, w.norm1, mod(1), mod(0), b.y, ldx, b.tokens, kDim, stream); break;
    case 1: gemm(b.qkv, ldq, b.y, ldx, w.qkv, kDim, b.tokens, 3 * kDim, kDim, stream); break;
    case 2: qk_norm_rope(b.qkv, ldq, b.tokens, kHeads, w.q_norm, w.k_norm, b.rope, kHeadDim, stream); break;
    case 3:
      attention({b.qkv, k, v, b.attn, ldq, ldq, ldq, ldx, kHeadDim, kHeadDim, kHeadDim, kHeadDim, b.cu_seqlens, b.cu_seqlens,
                 b.batch, b.max_seqlen, kHeads, static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)))},
                stream);
      break;
    case 4: gemm_gated_residual(b.x, ldx, b.attn, ldx, w.o, kDim, mod(2), b.tokens, kDim, kDim, stream); break;
    case 5: rms_norm_modulate(b.x, ldx, w.norm2, mod(4), mod(3), b.y, ldx, b.tokens, kDim, stream); break;
    case 6: gemm_situ_glu(b.ffn, kFfnDim, b.y, ldx, w.up_gate, kDim, b.tokens, kFfnDim, kDim, stream); break;
    default: gemm_gated_residual(b.x, ldx, b.ffn, kFfnDim, w.down, kFfnDim, mod(5), b.tokens, kDim, kFfnDim, stream); break;
  }
}

void run_block(const BlockWeights& w, const BlockBuffers& b, hipStream_t stream) {
  for (int op = 0; op < kBlockOps; ++op) run_block_op(op, w, b, stream);
}

}  // namespace hip_block
