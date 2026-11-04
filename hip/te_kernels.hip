// The text encoder's kernels and their launches (te_kernels.h). The op order and the rounding points are those of the
// engine's kernels/basic/norm.cu (rms_norm_qwen, rms_norm_gated_qwen), rope.cu (qwen_rope_apply), elementwise.cu
// (silu_mul, sigmoid_mul), gdn/conv_gates.cu, gdn/chunk_rule.cu (prep and state kernels) and FlashAttention-2's causal
// compute_attn_1rowblock with softmax.h. Every workgroup is one wave of 32 threads, as in block_kernels.hip: where an
// engine kernel spreads one task over a larger block, the task is split over several workgroups along an axis whose
// outputs do not interact (rows of a chunk's products, columns of the UT system's right-hand sides, columns of the
// state), and every output keeps its own operation order.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "gemm_tile.h"
#include "hip_block.h"
#include "mma_leaf.h"
#include "te_kernels.h"

namespace hip_block {
namespace {

using te::kAttnHeadDim;
using te::kAttnStepN;
using te::kGdnChunk;
using te::kGdnConvDim;
using te::kGdnHeadDim;
using te::kGdnKeyDim;
using te::kGdnKeyHeads;
using te::kGdnValueHeads;
using te::kRopeDim;

constexpr int kMaxReduceThreads = 512;
constexpr float kGdnNormEps = 1e-6f;              // the rule's l2norm eps
constexpr float kGdnQScale = 0.08838834764831845f;  // 128 ** -0.5

// ---- RMSNorms ----

struct TeNormArgs {
  const bf16_bits* x;
  std::int64_t ldx, hsx;
  bf16_bits* out;
  std::int64_t ldo, hso;
  const bf16_bits* weight;
  const bf16_bits* gate;
  std::int64_t ldg, hsg;
  const bf16_bits* rope;
  std::int64_t rope_ld;
  int d;
  int bw;        // torch's threads per row, a power of two from 32 to 512
  float factor;  // torch MeanOps factor: float(rows) / float(rows * d)
  float eps;
};

// One (token, head) row per wave: torch's bw threads per row as lanes l, l + 32, ... (dit_kernels.hip's rms_norm).
__global__ __launch_bounds__(kWave) void te_rms_norm_kernel(const TeNormArgs a) {
  __shared__ float red[kMaxReduceThreads];
  __shared__ float rot[kRopeDim];
  const int t = thread_id();
  const std::int64_t token = blockIdx.x, head = blockIdx.y;
  const bf16_bits* xr = a.x + token * a.ldx + head * a.hsx;

  // thread_reduce: torch's thread tx sums its 4-value groups tx, tx + bw, ... in 4 running values, then adds them.
  for (int tx = t; tx < a.bw; tx += kWave) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int idx = tx; idx * 4 + 3 < a.d; idx += a.bw) {
      HIP_BLOCK_UNROLL
      for (int i = 0; i < 4; ++i) {
        const float v = f(xr[idx * 4 + i]);
        acc[i] = add(acc[i], mul(v, v));
      }
    }
    red[tx] = add(add(add(acc[0], acc[1]), acc[2]), acc[3]);
  }
  leaf::barrier();
  // block_x_reduce's halving through shared memory down to one warp, then its shuffle-down tree (the same sums as a
  // halving tree over the 32).
  for (int off = a.bw / 2; off >= 1; off >>= 1) {
    for (int tx = t; tx < off; tx += kWave) red[tx] = add(red[tx], red[tx + off]);
    leaf::barrier();
  }
  const float r = rsqrt(add(mul(red[0], a.factor), a.eps));

  bf16_bits* outr = a.out + token * a.ldo + head * a.hso;
  const bf16_bits* gr = a.gate != nullptr ? a.gate + token * a.ldg + head * a.hsg : nullptr;
  const bool rope = a.rope != nullptr;
  for (int c = t; c < a.d; c += kWave) {
    const float xn = mul(f(xr[c]), r);
    float y;
    if (a.gate == nullptr) {
      y = mul(xn, add(1.0f, f(a.weight[c])));
    } else {
      y = mul(rbf(mul(f(a.weight[c]), rbf(xn))), silu_f32(f(gr[c])));
    }
    if (rope && c < kRopeDim) {
      rot[c] = rbf(y);
    } else {
      outr[c] = bf(y);
    }
  }
  if (rope) {
    leaf::barrier();
    // qwen_rope_apply's rotate_pair on the bf16 norm output: lane c rotates (c, c + 32) by the row [cos(32) | sin(32)].
    const bf16_bits* tab = a.rope + token * a.rope_ld;
    const float lo = rot[t], hi = rot[t + kRopeDim / 2], co = f(tab[t]), si = f(tab[kRopeDim / 2 + t]);
    outr[t] = bf(add(rbf(mul(lo, co)), rbf(mul(-hi, si))));
    outr[t + kRopeDim / 2] = bf(add(rbf(mul(hi, co)), rbf(mul(lo, si))));
  }
}
static_assert(kRopeDim / 2 == kWave);

// ---- the text RoPE table ----

struct RopeArgs {
  bf16_bits* out;
  float inv_freq[32];
};

// One position per wave, lane l the frequency l: cos at column l, sin at column 32 + l (qwen_table_kernel).
__global__ __launch_bounds__(kWave) void te_rope_table_kernel(const RopeArgs a) {
  const std::int64_t r = blockIdx.x;
  const int l = thread_id();
  const float phase = mul(static_cast<float>(r), a.inv_freq[l]);
  a.out[r * kRopeDim + l] = bf(cosf(phase));
  a.out[r * kRopeDim + kRopeDim / 2 + l] = bf(sinf(phase));
}

// Qwen3_5TextRotaryEmbedding inv_freq for rotary dim 64, theta 1e7, as torch's CPU computes it (kernels/basic/rope.h).
constexpr std::uint32_t kRopeInvFreqBits[32] = {
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
    0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
    0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
    0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44};

// ---- GatedDeltaNet: conv, gates ----

// The first row of the sequence holding token t (sequences are few: cond and negative prompts).
__device__ __forceinline__ int seq_start(const std::int32_t* cu, int seqs, int t) {
  int start = 0;
  for (int s = 1; s < seqs; ++s) start = t >= cu[s] ? cu[s] : start;
  return start;
}

constexpr int kConvVec = 8;  // channels per lane
struct ConvArgs {
  const bf16_bits* x;
  std::int64_t ldx;
  const bf16_bits* w;
  bf16_bits* y;
  std::int64_t ldy;
  const std::int32_t* cu;
  int seqs;
};

__global__ __launch_bounds__(kWave) void te_conv_silu_kernel(const ConvArgs a) {
  const int t = static_cast<int>(blockIdx.x);
  const int c0 = (static_cast<int>(blockIdx.y) * kWave + thread_id()) * kConvVec;
  const int start = seq_start(a.cu, a.seqs, t);
  float acc[kConvVec];
  HIP_BLOCK_UNROLL
  for (int c = 0; c < kConvVec; ++c) acc[c] = 0.0f;
  // out[t] = sum_k w[k] * x[t - 3 + k]: rows before the sequence's start are the zero padding.
  HIP_BLOCK_UNROLL
  for (int k = 0; k < te::kGdnConvKernel; ++k) {
    const int src = t - (te::kGdnConvKernel - 1) + k;
    if (src < start) continue;
    const bf16_bits* xr = a.x + static_cast<std::int64_t>(src) * a.ldx + c0;
    HIP_BLOCK_UNROLL
    for (int c = 0; c < kConvVec; ++c) acc[c] = fmaf(f(a.w[(c0 + c) * te::kGdnConvKernel + k]), f(xr[c]), acc[c]);
  }
  bf16_bits* yr = a.y + static_cast<std::int64_t>(t) * a.ldy + c0;
  HIP_BLOCK_UNROLL
  for (int c = 0; c < kConvVec; ++c) yr[c] = bf(silu_f32(rbf(acc[c])));  // conv output rounded, then SiLU rounded
}

struct GatesArgs {
  const bf16_bits* ba;
  std::int64_t ldba;
  const bf16_bits* a_log;
  const bf16_bits* dt_bias;
  float* g;
  bf16_bits* beta;
};

// One token per wave, lane h the value head h.
__global__ __launch_bounds__(kWave) void te_gates_kernel(const GatesArgs a) {
  static_assert(kGdnValueHeads == kWave);
  const std::int64_t t = blockIdx.x;
  const int h = thread_id();
  const bf16_bits* row = a.ba + t * a.ldba;
  a.beta[t * kGdnValueHeads + h] = bf(sigmoid_f32(f(row[h])));
  const float x = add(f(row[kGdnValueHeads + h]), f(a.dt_bias[h]));
  const float sp = x > 20.0f ? x : log1pf(expf(x));
  a.g[t * kGdnValueHeads + h] = mul(-expf(f(a.a_log[h])), sp);
}

// ---- GatedDeltaNet: the chunked rule ----

struct GdnArgs {
  const bf16_bits* qkv;
  std::int64_t ld;
  const float* g;
  const bf16_bits* beta;
  const std::int32_t* cu;
  std::int64_t tokens;
  bf16_bits* out;
  te::GdnWorkspace ws;
};

// q and k of one token and key head: lane l holds dims 4l .. 4l + 3. The squares are fused multiply-adds, the 32
// partial sums a halving tree (the engine's butterfly of shuffles gives every lane this sum).
__global__ __launch_bounds__(kWave) void te_gdn_qk_kernel(const GdnArgs a) {
  __shared__ float red_q[kWave], red_k[kWave];
  const int l = thread_id();
  const std::int64_t t = blockIdx.x;
  const int kh = static_cast<int>(blockIdx.y);
  const bf16_bits* row = a.qkv + t * a.ld + kh * kGdnHeadDim + 4 * l;
  float q[4], k[4], sq = 0.0f, sk = 0.0f;
  HIP_BLOCK_UNROLL
  for (int e = 0; e < 4; ++e) {
    q[e] = f(row[e]);
    k[e] = f(row[kGdnKeyDim + e]);
    sq = fmaf(q[e], q[e], sq);
    sk = fmaf(k[e], k[e], sk);
  }
  red_q[l] = sq;
  red_k[l] = sk;
  leaf::barrier();
  HIP_BLOCK_UNROLL
  for (int off = kWave / 2; off > 0; off >>= 1) {
    if (l < off) {
      red_q[l] = add(red_q[l], red_q[l + off]);
      red_k[l] = add(red_k[l], red_k[l + off]);
    }
    leaf::barrier();
  }
  const float iq = rsqrt(add(red_q[0], kGdnNormEps)), ik = rsqrt(add(red_k[0], kGdnNormEps));
  const std::int64_t at = (kh * a.tokens + t) * kGdnHeadDim + 4 * l;
  HIP_BLOCK_UNROLL
  for (int e = 0; e < 4; ++e) {
    a.ws.qn[at + e] = mul(mul(q[e], iq), kGdnQScale);
    a.ws.kn[at + e] = mul(k[e], ik);
  }
}

// The chunk's decays and products for one value head h and 16 of its 64 rows (blockIdx.y = 4 h + quarter):
//   gcum = cumsum(g), eg = exp(gcum), el = exp(gcum[n - 1] - gcum)                (quarter 0 writes them)
//   a[i][j]    = (k_i . k_j) * beta_i * exp(gcum_i - gcum_j) for j < i, else 0
//   attn[i][j] = (q_i . k_j) * exp(gcum_i - gcum_j) for j <= i, else 0
// dot products as fused multiply-adds over d ascending, rows past the sequence zero. Lane (ty, tx) = (l / 16, l % 16)
// owns rows i0 + ty + 2 r (r < 8) and columns tx + 16 c (c < 4). q and k pass through __shared__ memory in tiles of 32
// dims.
constexpr int kChunkRows = 16;
constexpr int kDimTile = 32;
__global__ __launch_bounds__(kWave) void te_gdn_chunk_kernel(const GdnArgs a) {
  constexpr int kR = kChunkRows / 2, kC = kGdnChunk / 16;
  __shared__ float gc[kGdnChunk], bt[kGdnChunk];
  __shared__ float k_tile[kGdnChunk][kDimTile + 1], q_tile[kChunkRows][kDimTile + 1];
  const int l = thread_id();
  const int h = static_cast<int>(blockIdx.y) / 4, kh = h / 2, i0 = static_cast<int>(blockIdx.y) % 4 * kChunkRows;
  const int seq = static_cast<int>(blockIdx.z);
  const int start = a.cu[seq], len = a.cu[seq + 1] - start;
  const int c0 = static_cast<int>(blockIdx.x) * kGdnChunk;
  if (c0 >= len) return;
  const int n = len - c0 < kGdnChunk ? len - c0 : kGdnChunk;
  if (i0 >= n) return;
  const std::int64_t t0 = start + c0, T = a.tokens;

  for (int i = l; i < kGdnChunk; i += kWave) {
    const bool valid = i < n;
    bt[i] = valid ? f(a.beta[(t0 + i) * kGdnValueHeads + h]) : 0.0f;
    gc[i] = valid ? a.g[(t0 + i) * kGdnValueHeads + h] : 0.0f;  // the padding's decay is 0 in log space
  }
  leaf::barrier();
  if (l == 0) {
    float acc = 0.0f;
    for (int i = 0; i < kGdnChunk; ++i) {
      acc = add(acc, gc[i]);
      gc[i] = acc;
    }
  }
  leaf::barrier();
  if (i0 == 0) {
    for (int i = l; i < n; i += kWave) {
      const std::int64_t at = h * T + t0 + i;
      a.ws.gcum[at] = gc[i];
      a.ws.eg[at] = expf(gc[i]);
      a.ws.el[at] = expf(add(gc[n - 1], -gc[i]));
    }
  }

  const int ty = l / 16, tx = l % 16;
  float kk[kR][kC], qk[kR][kC];
  HIP_BLOCK_UNROLL
  for (int r = 0; r < kR; ++r) {
    HIP_BLOCK_UNROLL
    for (int c = 0; c < kC; ++c) kk[r][c] = qk[r][c] = 0.0f;
  }
  const float* kn = a.ws.kn + (kh * T + t0) * kGdnHeadDim;
  const float* qn = a.ws.qn + (kh * T + t0) * kGdnHeadDim;
  for (int d0 = 0; d0 < kGdnHeadDim; d0 += kDimTile) {
    leaf::barrier();  // the previous tile's readers are done
    for (int j = 0; j < kGdnChunk; ++j) k_tile[j][l] = j < n ? kn[j * kGdnHeadDim + d0 + l] : 0.0f;
    for (int i = 0; i < kChunkRows; ++i) q_tile[i][l] = i0 + i < n ? qn[(i0 + i) * kGdnHeadDim + d0 + l] : 0.0f;
    leaf::barrier();
    for (int dd = 0; dd < kDimTile; ++dd) {
      float qi[kR], ki[kR], kj[kC];
      HIP_BLOCK_UNROLL
      for (int r = 0; r < kR; ++r) {
        qi[r] = q_tile[ty + 2 * r][dd];
        ki[r] = k_tile[i0 + ty + 2 * r][dd];
      }
      HIP_BLOCK_UNROLL
      for (int c = 0; c < kC; ++c) kj[c] = k_tile[tx + 16 * c][dd];
      HIP_BLOCK_UNROLL
      for (int r = 0; r < kR; ++r) {
        HIP_BLOCK_UNROLL
        for (int c = 0; c < kC; ++c) {
          kk[r][c] = fmaf(ki[r], kj[c], kk[r][c]);
          qk[r][c] = fmaf(qi[r], kj[c], qk[r][c]);
        }
      }
    }
  }

  HIP_BLOCK_UNROLL
  for (int r = 0; r < kR; ++r) {
    const int i = i0 + ty + 2 * r;
    if (i >= n) continue;
    const std::int64_t row = (h * T + t0 + i) * kGdnChunk;
    HIP_BLOCK_UNROLL
    for (int c = 0; c < kC; ++c) {
      const int j = tx + 16 * c;
      const float decay = j <= i ? expf(add(gc[i], -gc[j])) : 0.0f;
      a.ws.a[row + j] = j < i ? mul(mul(kk[r][c], bt[i]), decay) : 0.0f;
      a.ws.attn[row + j] = mul(qk[r][c], decay);
    }
  }
}

// The UT transform of one chunk of value head h for 32 of the 256 right-hand sides (blockIdx.y = 8 h + block), lane l
// column 32 block + l: columns 0..127 b = bf16 v * beta -> u, 128..255 b = (k * beta) * exp(gcum) -> w. Forward
// substitution x_i = b_i - sum_{j < i} a_ij x_j, the j in groups of 4 from 0 (a is zero on and above the diagonal, so a
// group may run past i - 1), each step a fused multiply-add: the engine's prep kernel. Lane l keeps its column of x in
// column l of xs, which no other lane touches.
constexpr int kLdA = kGdnChunk + 4;
constexpr int kSolveCols = 32;
__global__ __launch_bounds__(kWave) void te_gdn_solve_kernel(const GdnArgs a) {
  alignas(16) __shared__ float a_s[kGdnChunk * kLdA];
  __shared__ float xs[kGdnChunk * kSolveCols];
  __shared__ float bt[kGdnChunk], eg[kGdnChunk];
  const int l = thread_id();
  constexpr int kBlocks = 2 * kGdnHeadDim / kSolveCols;
  const int h = static_cast<int>(blockIdx.y) / kBlocks, kh = h / 2;
  const int col = static_cast<int>(blockIdx.y) % kBlocks * kSolveCols + l;
  const int seq = static_cast<int>(blockIdx.z);
  const int start = a.cu[seq], len = a.cu[seq + 1] - start;
  const int c0 = static_cast<int>(blockIdx.x) * kGdnChunk;
  if (c0 >= len) return;
  const int n = len - c0 < kGdnChunk ? len - c0 : kGdnChunk;
  const std::int64_t t0 = start + c0, T = a.tokens;

  for (int idx = l; idx < kGdnChunk * kGdnChunk; idx += kWave) {
    const int i = idx / kGdnChunk, j = idx % kGdnChunk;
    a_s[i * kLdA + j] = i < n ? a.ws.a[(h * T + t0 + i) * kGdnChunk + j] : 0.0f;
  }
  for (int i = l; i < kGdnChunk; i += kWave) {
    bt[i] = i < n ? f(a.beta[(t0 + i) * kGdnValueHeads + h]) : 0.0f;
    eg[i] = i < n ? a.ws.eg[h * T + t0 + i] : 0.0f;
  }
  leaf::barrier();

  const bool is_u = col < kGdnHeadDim;
  const int c = is_u ? col : col - kGdnHeadDim;
  float* x = xs + l;
  for (int i = 0; i < kGdnChunk; ++i) {
    float b = 0.0f;
    if (i < n) {
      b = is_u ? mul(f(a.qkv[(t0 + i) * a.ld + 2 * kGdnKeyDim + h * kGdnHeadDim + c]), bt[i])
               : mul(mul(a.ws.kn[(kh * T + t0 + i) * kGdnHeadDim + c], bt[i]), eg[i]);
    }
    x[i * kSolveCols] = b;
  }
  for (int i = 1; i < n; ++i) {
    float acc = x[i * kSolveCols];
    for (int j = 0; j < i; j += 4) {
      const float4 v = *reinterpret_cast<const float4*>(a_s + i * kLdA + j);
      acc = fmaf(-v.x, x[j * kSolveCols], acc);
      acc = fmaf(-v.y, x[(j + 1) * kSolveCols], acc);
      acc = fmaf(-v.z, x[(j + 2) * kSolveCols], acc);
      acc = fmaf(-v.w, x[(j + 3) * kSolveCols], acc);
    }
    x[i * kSolveCols] = acc;
  }
  float* dst = (is_u ? a.ws.u : a.ws.w) + (h * T + t0) * kGdnHeadDim + c;
  for (int i = 0; i < n; ++i) dst[i * kGdnHeadDim] = x[i * kSolveCols];
}

// The scan over one sequence's chunks for value head h and 8 value columns vc0 .. vc0 + 7 (blockIdx.x), S [128][8] the
// state slice, zero at the start:
//   v_new = u - w S
//   o     = (q exp(gcum)) S + attn v_new                            -> out, bf16
//   S     = S exp(gcum[n - 1]) + (k exp(gcum[n - 1] - gcum))^T v_new
// as the engine's state kernel: w S and q S fused multiply-adds over d ascending; o continues from q S over j ascending
// (all 64, the padding zero); the update over j ascending from zero, then fma(S, exp(gcum[n - 1]), update). Lane l owns
// rows l and l + 32 of w S and of q S and state rows l + 32 i (i < 4).
constexpr int kStateCols = 8;
__global__ __launch_bounds__(kWave) void te_gdn_state_kernel(const GdnArgs a) {
  constexpr int kTileLd = kDimTile + 1, kAttnLd = kGdnChunk + 1;
  static_assert(2 * kGdnChunk * kTileLd >= kGdnChunk * kAttnLd);
  __shared__ float s_state[kGdnHeadDim][kStateCols];
  __shared__ float vn[kGdnChunk][kStateCols];
  __shared__ float tile[2 * kGdnChunk * kTileLd];  // [w; q exp(gcum)] rows of a 32-dim tile, then attn [64][65]
  const int l = thread_id();
  const int vc0 = static_cast<int>(blockIdx.x) * kStateCols;
  const int h = static_cast<int>(blockIdx.y), kh = h / 2;
  const int seq = static_cast<int>(blockIdx.z);
  const int start = a.cu[seq], len = a.cu[seq + 1] - start;
  const std::int64_t T = a.tokens, ht = h * T, kt = kh * T;

  float st[4][kStateCols];
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 4; ++i) {
    HIP_BLOCK_UNROLL
    for (int c = 0; c < kStateCols; ++c) {
      st[i][c] = 0.0f;
      s_state[l + 32 * i][c] = 0.0f;
    }
  }

  for (int c0 = 0; c0 < len; c0 += kGdnChunk) {
    const std::int64_t t0 = start + c0;
    const int n = len - c0 < kGdnChunk ? len - c0 : kGdnChunk;

    // p = [w; qd] S: p[0], p[1] rows l, l + 32 of w; p[2], p[3] those of qd.
    float p[4][kStateCols];
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 4; ++i) {
      HIP_BLOCK_UNROLL
      for (int c = 0; c < kStateCols; ++c) p[i][c] = 0.0f;
    }
    for (int d0 = 0; d0 < kGdnHeadDim; d0 += kDimTile) {
      leaf::barrier();  // the tile's earlier readers are done, the state written
      for (int row = 0; row < kGdnChunk; ++row) {
        const bool valid = row < n;
        tile[row * kTileLd + l] = valid ? a.ws.w[(ht + t0 + row) * kGdnHeadDim + d0 + l] : 0.0f;
        tile[(kGdnChunk + row) * kTileLd + l] =
            valid ? mul(a.ws.qn[(kt + t0 + row) * kGdnHeadDim + d0 + l], a.ws.eg[ht + t0 + row]) : 0.0f;
      }
      leaf::barrier();
      for (int dd = 0; dd < kDimTile; ++dd) {
        float s[kStateCols];
        HIP_BLOCK_UNROLL
        for (int c = 0; c < kStateCols; ++c) s[c] = s_state[d0 + dd][c];
        HIP_BLOCK_UNROLL
        for (int i = 0; i < 4; ++i) {
          const float av = tile[(l + 32 * i) * kTileLd + dd];
          HIP_BLOCK_UNROLL
          for (int c = 0; c < kStateCols; ++c) p[i][c] = fmaf(av, s[c], p[i][c]);
        }
      }
    }
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 2; ++i) {
      const int row = l + 32 * i;
      HIP_BLOCK_UNROLL
      for (int c = 0; c < kStateCols; ++c) {
        const float u = row < n ? a.ws.u[(ht + t0 + row) * kGdnHeadDim + vc0 + c] : 0.0f;
        vn[row][c] = add(u, -p[i][c]);
      }
    }
    leaf::barrier();  // the tile's readers are done, v_new complete
    for (int row = 0; row < kGdnChunk; ++row) {
      for (int j = l; j < kGdnChunk; j += kWave) {
        tile[row * kAttnLd + j] = row < n ? a.ws.attn[(ht + t0 + row) * kGdnChunk + j] : 0.0f;
      }
    }
    leaf::barrier();

    // o = q S + attn v_new for rows l and l + 32.
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 2; ++i) {
      const int row = l + 32 * i;
      float o[kStateCols];
      HIP_BLOCK_UNROLL
      for (int c = 0; c < kStateCols; ++c) o[c] = p[2 + i][c];
      for (int j = 0; j < kGdnChunk; ++j) {
        const float av = tile[row * kAttnLd + j];
        HIP_BLOCK_UNROLL
        for (int c = 0; c < kStateCols; ++c) o[c] = fmaf(av, vn[j][c], o[c]);
      }
      if (row < n) {
        bf16_bits* dst = a.out + ((t0 + row) * kGdnValueHeads + h) * kGdnHeadDim + vc0;
        HIP_BLOCK_UNROLL
        for (int c = 0; c < kStateCols; ++c) dst[c] = bf(o[c]);
      }
    }

    // S = S exp(gcum[n - 1]) + kd^T v_new, kd = k exp(gcum[n - 1] - gcum).
    const float cd = expf(a.ws.gcum[ht + t0 + n - 1]);
    float upd[4][kStateCols];
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 4; ++i) {
      HIP_BLOCK_UNROLL
      for (int c = 0; c < kStateCols; ++c) upd[i][c] = 0.0f;
    }
    for (int j = 0; j < kGdnChunk; ++j) {
      float kd[4];
      if (j < n) {
        const float el = a.ws.el[ht + t0 + j];
        HIP_BLOCK_UNROLL
        for (int i = 0; i < 4; ++i) kd[i] = mul(a.ws.kn[(kt + t0 + j) * kGdnHeadDim + l + 32 * i], el);
      } else {
        HIP_BLOCK_UNROLL
        for (int i = 0; i < 4; ++i) kd[i] = 0.0f;
      }
      HIP_BLOCK_UNROLL
      for (int i = 0; i < 4; ++i) {
        HIP_BLOCK_UNROLL
        for (int c = 0; c < kStateCols; ++c) upd[i][c] = fmaf(kd[i], vn[j][c], upd[i][c]);
      }
    }
    leaf::barrier();  // every lane's reads of the state for p are long done; the state rows below are this lane's
    HIP_BLOCK_UNROLL
    for (int i = 0; i < 4; ++i) {
      HIP_BLOCK_UNROLL
      for (int c = 0; c < kStateCols; ++c) {
        st[i][c] = fmaf(st[i][c], cd, upd[i][c]);
        s_state[l + 32 * i][c] = st[i][c];
      }
    }
  }
}

// ---- causal attention, head dim 256 ----

// One tile of 16 query rows of one head of one sequence per wave, as hip_block's attention_kernel with the causal mask
// (keys past the row's own index are -inf before the maximum), key steps of kAttnStepN and the row sums in
// FlashAttention-2's order (below); the steps walked are those holding a key some row of the tile sees, which are
// FlashAttention's for its 64-row tiles (each row's first step holds its own key, so no row starts fully masked).
constexpr int kTeAttnTileM = 16;
struct TeAttnArgs {
  const bf16_bits* q;
  const bf16_bits* k;
  const bf16_bits* v;
  bf16_bits* o;
  const bf16_bits* gate;
  std::int64_t ldq, ldk, ldv, ldo, ldg;
  std::int64_t hsq, hsk, hsv, hso, hsg;
  const std::int32_t* cu;
  int kv_group;
  float scale_log2;
};

__global__ __launch_bounds__(kWave) void te_attention_kernel(const TeAttnArgs a) {
  constexpr int kMf = kTeAttnTileM / kFrag, kSf = kAttnStepN / kFrag, kDf = kAttnHeadDim / kFrag, kDg = 4;
  static_assert(kMf == 1 && kAttnStepN % kTeAttnTileM == 0);
  alignas(16) __shared__ float s_mem[kTeAttnTileM * kAttnStepN];
  alignas(16) __shared__ bf16_bits p_mem[kTeAttnTileM * kAttnStepN];
  alignas(16) __shared__ float o_mem[kTeAttnTileM * kAttnHeadDim];
  alignas(16) __shared__ bf16_bits pad[kFrag * kFrag];
  __shared__ float row_max[kTeAttnTileM], row_scale[kTeAttnTileM], row_max_scaled[kTeAttnTileM];

  const int t = thread_id();
  const int m_block = static_cast<int>(blockIdx.x), seq = static_cast<int>(blockIdx.y), head = static_cast<int>(blockIdx.z);
  const std::int64_t row0 = a.cu[seq];
  const int len = a.cu[seq + 1] - a.cu[seq];
  const int q_first = m_block * kTeAttnTileM;
  if (q_first >= len) return;
  const int rows = len - q_first < kTeAttnTileM ? len - q_first : kTeAttnTileM;
  const int key_end = q_first + rows;  // keys 0 .. key_end - 1 are seen by some row of the tile
  const int steps = (key_end + kAttnStepN - 1) / kAttnStepN;
  const int kv_head = head / a.kv_group;

  const bf16_bits* q = a.q + (row0 + q_first) * a.ldq + head * a.hsq;
  const bf16_bits* k_head = a.k + row0 * a.ldk + kv_head * a.hsk;
  const bf16_bits* v_head = a.v + row0 * a.ldv + kv_head * a.hsv;

  // acc_o lives in o_mem (fp32, exact through memory): its 16 fragments in registers would not fit beside the walk's.
  for (int i = t; i < kTeAttnTileM * kAttnHeadDim; i += kWave) o_mem[i] = 0.0f;
  // Lane t's row sum as FlashAttention-2 keeps it: the four partial sums of the quad of threads that hold the row, thread
  // k of the quad holding columns 8 j + 2 k and 8 j + 2 k + 1 of every step (j ascending, the pair inner), each partial
  // rescaled with the row; combined as (p0 + p2) + (p1 + p3) at the end (quad_allreduce_).
  float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int step = steps - 1; step >= 0; --step) {
    const bool first = step == steps - 1;
    const int n0 = step * kAttnStepN;
    const int cols = key_end - n0 < kAttnStepN ? key_end - n0 : kAttnStepN;

    const bf16_bits* b[kSf];
    int b_cols[kSf];
    HIP_BLOCK_UNROLL
    for (int nf = 0; nf < kSf; ++nf) {
      b[nf] = k_head + static_cast<std::int64_t>(n0 + nf * kFrag) * a.ldk;
      b_cols[nf] = clamp_frag(cols - nf * kFrag);
    }
    FragAcc acc_s[kMf][kSf];
    clear_tile(acc_s);
    mma_walk(acc_s, q, a.ldq, rows, b, a.ldk, b_cols, kAttnHeadDim, pad);
    store_tile(s_mem, acc_s);
    leaf::barrier();
    // The causal mask: key n0 + c is past row q_first + r.
    for (int i = t; i < kTeAttnTileM * kAttnStepN; i += kWave) {
      if (n0 + i % kAttnStepN > q_first + i / kAttnStepN) s_mem[i] = -INFINITY;
    }
    leaf::barrier();

    // softmax_rescale_o. One lane per row: this row's new maximum and the rescale of its running sum.
    if (t < rows) {
      const float* s = s_mem + t * kAttnStepN;
      const float prev = row_max[t];
      float cur = first ? s[0] : fmaxf(prev, s[0]);
      int c1 = 1;
      for (; c1 < cols && c1 % 4 != 0; ++c1) cur = fmaxf(cur, s[c1]);
      for (; c1 + 4 <= cols; c1 += 4) {
        const float v0 = s[c1], v1 = s[c1 + 1], v2 = s[c1 + 2], v3 = s[c1 + 3];
        cur = fmaxf(fmaxf(fmaxf(fmaxf(cur, v0), v1), v2), v3);
      }
      for (; c1 < cols; ++c1) cur = fmaxf(cur, s[c1]);
      row_max[t] = cur;
      if (!first) {
        const float rescale = exp2f(mul(prev - cur, a.scale_log2));
        row_scale[t] = rescale;
        HIP_BLOCK_UNROLL
        for (int k = 0; k < 4; ++k) part[k] = mul(part[k], rescale);
      }
      row_max_scaled[t] = cur == -INFINITY ? 0.0f : mul(cur, a.scale_log2);
    }
    leaf::barrier();
    // The weights on every lane (into s_mem, each in its own place), then each row's partial sums on its lane.
    for (int i = t; i < kTeAttnTileM * kAttnStepN; i += kWave) {
      const int r = i / kAttnStepN, c = i % kAttnStepN;
      if (r < rows && c < cols) {
        const float w = exp2f(fmaf(s_mem[i], a.scale_log2, -row_max_scaled[r]));
        s_mem[i] = w;
        p_mem[i] = bf(w);
      } else {
        p_mem[i] = 0;
      }
    }
    leaf::barrier();
    if (t < rows) {
      const float* w_row = s_mem + t * kAttnStepN;
      HIP_BLOCK_UNROLL
      for (int k = 0; k < 4; ++k) {
        for (int c = 2 * k; c < cols; c += 8) {
          part[k] = add(part[k], w_row[c]);
          if (c + 1 < cols) part[k] = add(part[k], w_row[c + 1]);
        }
      }
    }
    leaf::barrier();

    // acc_o rescaled where some row's maximum changed (a rescale of exactly 1 changes nothing).
    bool rescale_any = false;
    if (!first) {
      HIP_BLOCK_UNROLL
      for (int r = 0; r < kTeAttnTileM; ++r) {
        if (r < rows && row_scale[r] != 1.0f) rescale_any = true;
      }
    }
    if (rescale_any) {
      for (int i = t; i < rows * kAttnHeadDim; i += kWave) o_mem[i] = mul(o_mem[i], row_scale[i / kAttnHeadDim]);
      leaf::barrier();
    }

    // acc_o += P · V_step, the step's keys ascending, kDg of the 16 dim fragments at a time.
    for (int dg = 0; dg < kDf; dg += kDg) {
      FragAcc acc[kDg];
      HIP_BLOCK_UNROLL
      for (int j = 0; j < kDg; ++j) leaf::load_acc(acc[j], o_mem + (dg + j) * kFrag, kAttnHeadDim);
      HIP_BLOCK_UNROLL
      for (int nf = 0; nf < kSf; ++nf) {
        const int ks = clamp_frag(cols - nf * kFrag);
        if (ks == 0) continue;
        FragA fp;
        leaf::load_a(fp, p_mem + nf * kFrag, kAttnStepN);
        const bf16_bits* v_step = v_head + static_cast<std::int64_t>(n0 + nf * kFrag) * a.ldv;
        FragB fb[kDg];
        HIP_BLOCK_UNROLL
        for (int j = 0; j < kDg; ++j) load_b_rows_padded(fb[j], v_step + (dg + j) * kFrag, a.ldv, ks, pad);
        HIP_BLOCK_UNROLL
        for (int j = 0; j < kDg; ++j) leaf::mma(acc[j], fp, fb[j]);
      }
      HIP_BLOCK_UNROLL
      for (int j = 0; j < kDg; ++j) leaf::store_acc(o_mem + (dg + j) * kFrag, acc[j], kAttnHeadDim);
    }
    leaf::barrier();
  }

  // normalize_softmax_lse without the log-sum-exp; then the output gate when given.
  if (t < rows) {
    const float sum = add(add(part[0], part[2]), add(part[1], part[3]));
    row_scale[t] = sum == 0.0f || sum != sum ? 1.0f : div(1.0f, sum);
  }
  leaf::barrier();
  bf16_bits* o = a.o + (row0 + q_first) * a.ldo + head * a.hso;
  const bf16_bits* g = a.gate != nullptr ? a.gate + (row0 + q_first) * a.ldg + head * a.hsg : nullptr;
  for (int i = t; i < rows * kAttnHeadDim; i += kWave) {
    const int r = i / kAttnHeadDim, c = i % kAttnHeadDim;
    bf16_bits out = bf(mul(o_mem[i], row_scale[r]));
    if (g != nullptr) out = bf(mul(f(out), rbf(sigmoid_f32(f(g[r * a.ldg + c])))));
    o[r * a.ldo + c] = out;
  }
}

// ---- the MLP's SiLU product ----

constexpr int kMulVec = 8;
struct SiluMulArgs {
  const bf16_bits* gu;
  std::int64_t ldg;
  bf16_bits* out;
  std::int64_t ldo;
  int cols;
};

__global__ __launch_bounds__(kWave) void te_silu_mul_kernel(const SiluMulArgs a) {
  const std::int64_t r = blockIdx.x;
  const int c0 = (static_cast<int>(blockIdx.y) * kWave + thread_id()) * kMulVec;
  if (c0 >= a.cols) return;
  const bf16_bits* g = a.gu + r * a.ldg + c0;
  bf16_bits* o = a.out + r * a.ldo + c0;
  HIP_BLOCK_UNROLL
  for (int i = 0; i < kMulVec; ++i) o[i] = bf(mul(rbf(silu_f32(f(g[i]))), f(g[a.cols + i])));
}

void require(bool ok, const char* op, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("hip_block::te::") + op + ": " + what);
}

int last_pow2(std::int64_t n) {
  int p = 1;
  while (static_cast<std::int64_t>(p) * 2 <= n && p < (1 << 30)) p *= 2;
  return p;
}

}  // namespace

namespace te {

int reduce_threads(std::int64_t rows, int d) {
  require(d % 4 == 0 && d >= 128, "reduce_threads", "row length must be a multiple of 4 and at least 128");
  constexpr int kMaxThreads = 512;
  const int dim0 = d / 4;
  const int dim0_pow2 = dim0 < kMaxThreads ? last_pow2(dim0) : kMaxThreads;
  const int dim1_pow2 = rows < kMaxThreads ? last_pow2(rows < 1 ? 1 : rows) : kMaxThreads;
  int bw = dim0_pow2 < 32 ? dim0_pow2 : 32;
  const int bh = dim1_pow2 < kMaxThreads / bw ? dim1_pow2 : kMaxThreads / bw;
  bw = dim0_pow2 < kMaxThreads / bh ? dim0_pow2 : kMaxThreads / bh;
  const int per_thread = (d + bw - 1) / bw;
  require(per_thread < (bh * 16 < 256 ? bh * 16 : 256), "reduce_threads", "the row needs torch's cross-warp split");
  return bw;
}

void rms_norm(const NormArgs& p, hipStream_t stream) {
  require(p.heads >= 1 && p.heads <= 65535, "rms_norm", "1 to 65535 heads");
  require(p.rope == nullptr || (p.gate == nullptr && p.d >= kRopeDim), "rms_norm", "RoPE needs the plain norm and d >= 64");
  if (p.tokens == 0) return;
  const std::int64_t rows = p.tokens * p.heads;
  const TeNormArgs a{p.x,        p.ldx,  p.hsx,    p.out,     p.ldo, p.hso,
                     p.weight,   p.gate, p.ldg,    p.hsg,     p.rope, p.rope_ld,
                     p.d,        reduce_threads(rows, p.d), static_cast<float>(rows) / static_cast<float>(rows * p.d), p.eps};
  te_rms_norm_kernel<<<dim3(static_cast<unsigned>(p.tokens), static_cast<unsigned>(p.heads)), dim3(kWave), 0, stream>>>(a);
  check_launch("te::rms_norm");
}

void rope_table(bf16_bits* out, std::int64_t rows, hipStream_t stream) {
  if (rows == 0) return;
  RopeArgs a{};
  a.out = out;
  for (int i = 0; i < 32; ++i) std::memcpy(&a.inv_freq[i], &kRopeInvFreqBits[i], 4);
  te_rope_table_kernel<<<dim3(static_cast<unsigned>(rows)), dim3(kWave), 0, stream>>>(a);
  check_launch("te::rope_table");
}

void conv_silu(const bf16_bits* x, std::int64_t ldx, const bf16_bits* w, bf16_bits* y, std::int64_t ldy,
               const std::int32_t* cu, int seqs, std::int64_t tokens, hipStream_t stream) {
  if (tokens == 0) return;
  const ConvArgs a{x, ldx, w, y, ldy, cu, seqs};
  te_conv_silu_kernel<<<dim3(static_cast<unsigned>(tokens), kGdnConvDim / (kWave * kConvVec)), dim3(kWave), 0, stream>>>(a);
  check_launch("te::conv_silu");
}

void gates(const bf16_bits* ba, std::int64_t ldba, const bf16_bits* a_log, const bf16_bits* dt_bias, float* g,
           bf16_bits* beta, std::int64_t tokens, hipStream_t stream) {
  if (tokens == 0) return;
  const GatesArgs a{ba, ldba, a_log, dt_bias, g, beta};
  te_gates_kernel<<<dim3(static_cast<unsigned>(tokens)), dim3(kWave), 0, stream>>>(a);
  check_launch("te::gates");
}

namespace {
std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
}  // namespace

std::size_t GdnWorkspace::bytes(std::int64_t tokens) {
  const auto t = static_cast<std::size_t>(tokens);
  const std::size_t key = align256(kGdnKeyHeads * t * kGdnHeadDim * 4), per_head = align256(kGdnValueHeads * t * 4),
                    chunk = align256(kGdnValueHeads * t * kGdnChunk * 4),
                    value = align256(kGdnValueHeads * t * kGdnHeadDim * 4);
  return 2 * key + 3 * per_head + 2 * chunk + 2 * value;
}

GdnWorkspace GdnWorkspace::at(void* base, std::int64_t tokens) {
  const auto t = static_cast<std::size_t>(tokens);
  const std::size_t key = align256(kGdnKeyHeads * t * kGdnHeadDim * 4), per_head = align256(kGdnValueHeads * t * 4),
                    chunk = align256(kGdnValueHeads * t * kGdnChunk * 4),
                    value = align256(kGdnValueHeads * t * kGdnHeadDim * 4);
  auto* p = static_cast<char*>(base);
  GdnWorkspace ws{};
  const auto take = [&](std::size_t bytes) {
    auto* at = reinterpret_cast<float*>(p);
    p += bytes;
    return at;
  };
  ws.qn = take(key);
  ws.kn = take(key);
  ws.gcum = take(per_head);
  ws.eg = take(per_head);
  ws.el = take(per_head);
  ws.a = take(chunk);
  ws.attn = take(chunk);
  ws.u = take(value);
  ws.w = take(value);
  return ws;
}

void gated_delta_rule(const bf16_bits* qkv, std::int64_t ld, const float* g, const bf16_bits* beta,
                      const std::int32_t* cu, int seqs, int max_len, std::int64_t tokens, bf16_bits* out,
                      const GdnWorkspace& ws, hipStream_t stream) {
  require(seqs >= 1 && seqs <= 65535 && max_len >= 0, "gated_delta_rule", "1 to 65535 sequences");
  if (tokens == 0 || max_len == 0) return;
  const GdnArgs a{qkv, ld, g, beta, cu, tokens, out, ws};
  const unsigned chunks = static_cast<unsigned>((max_len + kGdnChunk - 1) / kGdnChunk), s = static_cast<unsigned>(seqs);
  te_gdn_qk_kernel<<<dim3(static_cast<unsigned>(tokens), kGdnKeyHeads), dim3(kWave), 0, stream>>>(a);
  check_launch("te::gated_delta_rule (q, k)");
  te_gdn_chunk_kernel<<<dim3(chunks, kGdnValueHeads * (kGdnChunk / kChunkRows), s), dim3(kWave), 0, stream>>>(a);
  check_launch("te::gated_delta_rule (chunk)");
  te_gdn_solve_kernel<<<dim3(chunks, kGdnValueHeads * (2 * kGdnHeadDim / kSolveCols), s), dim3(kWave), 0, stream>>>(a);
  check_launch("te::gated_delta_rule (solve)");
  te_gdn_state_kernel<<<dim3(kGdnHeadDim / kStateCols, kGdnValueHeads, s), dim3(kWave), 0, stream>>>(a);
  check_launch("te::gated_delta_rule (state)");
}

void causal_attention(const AttentionArgs& p, hipStream_t stream) {
  require(p.kv_group >= 1 && p.heads % p.kv_group == 0 && p.batch >= 1 && p.batch <= 65535 && p.heads <= 65535,
          "causal_attention", "bad head grouping or batch");
  if (p.max_seqlen == 0) return;
  constexpr double kLog2e = 1.4426950408889634074;
  // In double then rounded, as upstream (float * M_LOG2E).
  const TeAttnArgs a{p.q,   p.k,   p.v,   p.o,   p.gate, p.ldq, p.ldk, p.ldv,      p.ldo,
                     p.ldg, p.hsq, p.hsk, p.hsv, p.hso,  p.hsg, p.cu,  p.kv_group, static_cast<float>(static_cast<double>(p.scale) * kLog2e)};
  const dim3 grid(grid_of(p.max_seqlen, kTeAttnTileM), static_cast<unsigned>(p.batch), static_cast<unsigned>(p.heads));
  te_attention_kernel<<<grid, dim3(kWave), 0, stream>>>(a);
  check_launch("te::causal_attention");
}

void silu_mul(const bf16_bits* gu, std::int64_t ldg, bf16_bits* out, std::int64_t ldo, std::int64_t rows, int cols,
              hipStream_t stream) {
  require(cols % kMulVec == 0, "silu_mul", "cols must be a multiple of 8");
  if (rows == 0 || cols == 0) return;
  const SiluMulArgs a{gu, ldg, out, ldo, cols};
  te_silu_mul_kernel<<<dim3(static_cast<unsigned>(rows), grid_of(cols, kWave * kMulVec)), dim3(kWave), 0, stream>>>(a);
  check_launch("te::silu_mul");
}

std::int64_t pack_rows(std::int64_t rows) { return (rows + kPackRowsMultiple - 1) / kPackRowsMultiple * kPackRowsMultiple; }

void gemm_packed(bf16_bits* out, std::int64_t ldo, const bf16_bits* xp, const bf16_bits* wp, int m, int n, int k,
                 hipStream_t stream) {
  require(k % kFrag == 0, "gemm_packed", "k must be a multiple of 16");
  if (m == 0 || n == 0) return;
  const Blocking bl = blocking_of(m, n, k, kGemmTileM, kGemmTileN, kGemmTileN);
  const GemmArgs a{out, ldo, nullptr, 0, nullptr, 0, nullptr, m, n, k, bl.bm, bl.bn, xp, wp};
  gemm_kernel<Epilogue::kPlain><<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("te::gemm_packed");
}

void gemm_packed_residual(bf16_bits* residual, std::int64_t ldr, const bf16_bits* xp, const bf16_bits* wp,
                          const bf16_bits* ones, int m, int n, int k, hipStream_t stream) {
  require(k % kFrag == 0, "gemm_packed_residual", "k must be a multiple of 16");
  if (m == 0 || n == 0) return;
  const Blocking bl = blocking_of(m, n, k, kGemmTileM, kGemmTileN, kGemmTileN);
  const GemmArgs a{residual, ldr, nullptr, 0, nullptr, 0, ones, m, n, k, bl.bm, bl.bn, xp, wp};
  gemm_kernel<Epilogue::kGatedResidual><<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch("te::gemm_packed_residual");
}

}  // namespace te
}  // namespace hip_block
