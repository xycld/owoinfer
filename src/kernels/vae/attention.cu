// fp32 single-head attention for the VAE mid-block (head dim 384) and the surrounding mid_attention block.
//
// Flash-style kernel: a CTA owns 64 query rows and walks the keys in blocks of 64 with an online softmax, so
// the L x L score matrix never exists. head_dim 384 in fp32 is too wide to keep Q (or a 64 x 384 K/V tile)
// resident next to the pipeline buffers, so every operand streams through one 3-buffer cp.async ring:
// per key block, 8 "S stages" (Q and K columns [48i, 48i+48)) build S = Q K^T, then 4 "V stages" (16 keys x
// 384) accumulate O += P V. 8 warps: for S, warp w computes rows 16(w%4).. x keys 32(w/4)..; for O, warp w
// owns rows 32(w%2).. x columns 96(w/2).. (96 fp32 accumulators per thread). Row max/alpha/P meet in smem.
// Math: TF32 m16n8k8 mma.sync; kTf32x3 splits each operand into big + small TF32 parts and accumulates
// small*big + big*small + big*big (the 3xTF32 scheme behind CUTLASS OpMultiplyAddFastF32, which the
// reference SDPA uses for fp32). Tensor-core accumulation truncates, so S and O are accumulated per stage in
// fresh registers and folded into the running sums with round-to-nearest adds.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/vae/vae_kernels.h"

namespace mi::vae {
namespace {

constexpr int kD = 384;
constexpr int kBr = 64, kBc = 64, kWarps = 8, kThreads = kWarps * 32;
constexpr int kDk = 48;                        // d columns per S stage
constexpr int kSStages = kD / kDk;             // 8
constexpr int kVKeys = 16;                     // keys per V stage
constexpr int kVStages = kBc / kVKeys;         // 4
constexpr int kStagesPerBlock = kSStages + kVStages;
constexpr int kQkStride = kDk + 4;             // 52 = 20 mod 32 banks: A/B fragment loads conflict-free
constexpr int kVStride = kD + 8;               // 392 = 8 mod 32: B fragment (k = t, n = g) loads conflict-free
constexpr int kPStride = kBc + 4;              // 68 = 4 mod 32
constexpr int kBufFloats = 2 * kBr * kQkStride;  // 6656 >= kVKeys * kVStride (6272)
constexpr int kBufs = 3;
constexpr int kSmemFloats = kBufs * kBufFloats + kBr * kPStride + 2 * kBr /*redmax*/ + 2 * kBr /*m*/ + kBr /*alpha*/ +
                            2 * kBr /*lsum*/;
constexpr int kSmemBytes = kSmemFloats * 4;
static_assert(kVKeys * kVStride <= kBufFloats, "V stage must fit a ring buffer");
static_assert(kSmemBytes <= 101376, "exceeds sm_89 opt-in shared memory");

__device__ __forceinline__ std::uint32_t to_tf32(float x) {
  std::uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

template <bool kSplit>
__device__ __forceinline__ void split(float x, std::uint32_t& big, std::uint32_t& small) {
  big = to_tf32(x);
  if constexpr (kSplit) small = to_tf32(x - __uint_as_float(big));
}

__device__ __forceinline__ void mma(float (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <bool kSplit>
__device__ __forceinline__ void mma3(float (&d)[4], const std::uint32_t (&ab)[4], const std::uint32_t (&as)[4],
                                     const std::uint32_t (&bb)[2], const std::uint32_t (&bs)[2]) {
  if constexpr (kSplit) {
    mma(d, as, bb);
    mma(d, ab, bs);
  }
  mma(d, ab, bb);
}

__device__ __forceinline__ void cp_async16(float* dst, const float* src, bool valid) {
  const unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;  // 0 = zero-fill (rows past L)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

struct AttnArgs {
  const float* q;
  const float* k;
  const float* v;
  float* o;
  std::int64_t q_batch, k_batch, v_batch, o_batch;  // element strides between batches
  std::int64_t q_ld, k_ld, v_ld;                    // row strides (o is contiguous, ld 384)
  int len;
  float scale_log2;  // scale * log2(e)
};

__device__ __forceinline__ void load_stage(const AttnArgs& a, float* buf, int stage, int q0, const float* qb,
                                           const float* kb_, const float* vb) {
  const int kb = stage / kStagesPerBlock, i = stage % kStagesPerBlock;
  if (i < kSStages) {
    // Q rows q0.., K rows kb*64.., columns [48i, 48i+48): 2 x 64 rows x 12 vectors.
    for (int vidx = threadIdx.x; vidx < 2 * kBr * (kDk / 4); vidx += kThreads) {
      const int which = vidx / (kBr * (kDk / 4));
      const int r = (vidx / (kDk / 4)) % kBr, c4 = vidx % (kDk / 4);
      const int row = (which == 0 ? q0 : kb * kBc) + r;
      const bool valid = row < a.len;
      const float* base = which == 0 ? qb : kb_;
      const std::int64_t ld = which == 0 ? a.q_ld : a.k_ld;
      const float* src = base + (valid ? static_cast<std::int64_t>(row) * ld + i * kDk + c4 * 4 : 0);
      cp_async16(buf + which * kBr * kQkStride + r * kQkStride + c4 * 4, src, valid);
    }
  } else {
    const int key0 = kb * kBc + (i - kSStages) * kVKeys;
    for (int vidx = threadIdx.x; vidx < kVKeys * (kD / 4); vidx += kThreads) {
      const int r = vidx / (kD / 4), c4 = vidx % (kD / 4);
      const int row = key0 + r;
      const bool valid = row < a.len;
      const float* src = vb + (valid ? static_cast<std::int64_t>(row) * a.v_ld + c4 * 4 : 0);
      cp_async16(buf + r * kVStride + c4 * 4, src, valid);
    }
  }
}

template <bool kSplit>
__global__ void __launch_bounds__(kThreads, 1) attention_kernel(AttnArgs a) {
  extern __shared__ float smem[];
  float* bufs = smem;
  float* p_sm = bufs + kBufs * kBufFloats;
  float* redmax = p_sm + kBr * kPStride;  // [2][64]
  float* m_sm = redmax + 2 * kBr;         // [2][64], parity double buffer
  float* alpha_sm = m_sm + 2 * kBr;       // [64]
  float* lsum = alpha_sm + kBr;           // [2][64]

  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int g = lane / 4, t = lane % 4;
  const int q0 = blockIdx.x * kBr;
  const int b = blockIdx.y;
  const float* qb = a.q + b * a.q_batch;
  const float* kbp = a.k + b * a.k_batch;
  const float* vb = a.v + b * a.v_batch;

  // S roles
  const int s_row = 16 * (warp % 4), s_col = 32 * (warp / 4), s_half = warp / 4;
  // O roles
  const int o_row = 32 * (warp % 2), o_col = 96 * (warp / 2);

  for (int i = threadIdx.x; i < kBr; i += kThreads) m_sm[i] = -INFINITY;

  float o_acc[2][12][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int nj = 0; nj < 12; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) o_acc[mi][nj][e] = 0.0f;
  float s_acc[4][4];
  float l_part[2] = {0.0f, 0.0f};

  const int num_kb = (a.len + kBc - 1) / kBc;
  const int total = num_kb * kStagesPerBlock;

  load_stage(a, bufs, 0, q0, qb, kbp, vb);
  cp_async_commit();
  if (total > 1) load_stage(a, bufs + kBufFloats, 1, q0, qb, kbp, vb);
  cp_async_commit();

  for (int stage = 0; stage < total; ++stage) {
    cp_async_wait<1>();
    __syncthreads();  // stage landed for all threads; buffer (stage + 2) % 3 is no longer read
    if (stage + 2 < total) load_stage(a, bufs + ((stage + 2) % kBufs) * kBufFloats, stage + 2, q0, qb, kbp, vb);
    cp_async_commit();

    const float* buf = bufs + (stage % kBufs) * kBufFloats;
    const int kb = stage / kStagesPerBlock, i = stage % kStagesPerBlock;

    if (i < kSStages) {
      if (i == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) s_acc[j][e] = 0.0f;
      }
      const float* qs = buf;
      const float* ks = buf + kBr * kQkStride;
      float s_tmp[4][4];  // per-stage partial, folded into s_acc with rounded adds (see the V stage note)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s_tmp[j][e] = 0.0f;
#pragma unroll
      for (int kk = 0; kk < kDk / 8; ++kk) {
        std::uint32_t ab[4], as[4];
        const float* qa = qs + (s_row + g) * kQkStride + kk * 8 + t;
        split<kSplit>(qa[0], ab[0], as[0]);
        split<kSplit>(qa[8 * kQkStride], ab[1], as[1]);
        split<kSplit>(qa[4], ab[2], as[2]);
        split<kSplit>(qa[8 * kQkStride + 4], ab[3], as[3]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          std::uint32_t bb[2], bs[2];
          const float* kp = ks + (s_col + 8 * j + g) * kQkStride + kk * 8 + t;
          split<kSplit>(kp[0], bb[0], bs[0]);
          split<kSplit>(kp[4], bb[1], bs[1]);
          mma3<kSplit>(s_tmp[j], ab, as, bb, bs);
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) s_acc[j][e] += s_tmp[j][e];
      if (i == kSStages - 1) {
        // Online softmax, part 1: mask keys past L, row max over this warp's 32 keys.
        float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int key = kb * kBc + s_col + 8 * j + 2 * t + (e & 1);
            if (key >= a.len) s_acc[j][e] = -INFINITY;
            rmax[e >> 1] = fmaxf(rmax[e >> 1], s_acc[j][e]);
          }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 1));
          rmax[h] = fmaxf(rmax[h], __shfl_xor_sync(0xffffffffu, rmax[h], 2));
        }
        if (t == 0) {
          redmax[s_half * kBr + s_row + g] = rmax[0];
          redmax[s_half * kBr + s_row + g + 8] = rmax[1];
        }
        __syncthreads();
        // Part 2: new running max, P = exp2((s - m) * scale * log2e) into smem, rescale factors.
        const float* m_old = m_sm + (kb & 1) * kBr;
        float* m_new_out = m_sm + ((kb + 1) & 1) * kBr;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = s_row + g + 8 * h;
          const float mo = m_old[row];
          const float mn = fmaxf(mo, fmaxf(redmax[row], redmax[kBr + row]));
          const float alpha = exp2f((mo - mn) * a.scale_log2);
          const float neg = -mn * a.scale_log2;
          float sum = 0.0f;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float p0 = exp2f(fmaf(s_acc[j][2 * h], a.scale_log2, neg));
            const float p1 = exp2f(fmaf(s_acc[j][2 * h + 1], a.scale_log2, neg));
            sum += p0 + p1;
            *reinterpret_cast<float2*>(p_sm + row * kPStride + s_col + 8 * j + 2 * t) = make_float2(p0, p1);
          }
          l_part[h] = l_part[h] * alpha + sum;
          if (s_half == 0 && t == 0) {
            alpha_sm[row] = alpha;
            m_new_out[row] = mn;
          }
        }
        // The next loop iteration's __syncthreads publishes P, alpha and m.
      }
    } else {
      const int vi = i - kSStages;
      if (vi == 0) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float alpha = alpha_sm[o_row + 16 * mi + g + 8 * h];
#pragma unroll
            for (int nj = 0; nj < 12; ++nj) {
              o_acc[mi][nj][2 * h] *= alpha;
              o_acc[mi][nj][2 * h + 1] *= alpha;
            }
          }
      }
      // Tensor-core accumulation truncates (it does not round to nearest), so a long chain of mma into one
      // accumulator drifts by ~2^-23 per step: over L/8 key steps that reached 1e-4 at L = 16384. Each V
      // stage therefore accumulates into a fresh register tile and is added to O with a rounded FADD.
      const float* vs = buf;
      std::uint32_t ab[kVKeys / 8][2][4], as[kVKeys / 8][2][4];
#pragma unroll
      for (int kk = 0; kk < kVKeys / 8; ++kk)
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const float* pa = p_sm + (o_row + 16 * mi + g) * kPStride + vi * kVKeys + kk * 8 + t;
          split<kSplit>(pa[0], ab[kk][mi][0], as[kk][mi][0]);
          split<kSplit>(pa[8 * kPStride], ab[kk][mi][1], as[kk][mi][1]);
          split<kSplit>(pa[4], ab[kk][mi][2], as[kk][mi][2]);
          split<kSplit>(pa[8 * kPStride + 4], ab[kk][mi][3], as[kk][mi][3]);
        }
#pragma unroll
      for (int nj = 0; nj < 12; ++nj) {
        std::uint32_t bb[kVKeys / 8][2], bs[kVKeys / 8][2];
#pragma unroll
        for (int kk = 0; kk < kVKeys / 8; ++kk) {
          const float* vp = vs + (kk * 8 + t) * kVStride + o_col + 8 * nj + g;
          split<kSplit>(vp[0], bb[kk][0], bs[kk][0]);
          split<kSplit>(vp[4 * kVStride], bb[kk][1], bs[kk][1]);
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
          for (int kk = 0; kk < kVKeys / 8; ++kk) mma3<kSplit>(tmp, ab[kk][mi], as[kk][mi], bb[kk], bs[kk]);
#pragma unroll
          for (int e = 0; e < 4; ++e) o_acc[mi][nj][e] += tmp[e];
        }
      }
    }
  }
  cp_async_wait<0>();

  // Row sums: reduce over the 4 lanes of a row and the two key halves.
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l_part[h] += __shfl_xor_sync(0xffffffffu, l_part[h], 1);
    l_part[h] += __shfl_xor_sync(0xffffffffu, l_part[h], 2);
  }
  if (t == 0) {
    lsum[s_half * kBr + s_row + g] = l_part[0];
    lsum[s_half * kBr + s_row + g + 8] = l_part[1];
  }
  __syncthreads();

  float* ob = a.o + b * a.o_batch;
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = o_row + 16 * mi + g + 8 * h;
      const int q = q0 + row;
      if (q >= a.len) continue;
      const float inv = 1.0f / (lsum[row] + lsum[kBr + row]);
      float* orow = ob + static_cast<std::int64_t>(q) * kD + o_col + 2 * t;
#pragma unroll
      for (int nj = 0; nj < 12; ++nj)
        *reinterpret_cast<float2*>(orow + 8 * nj) = make_float2(o_acc[mi][nj][2 * h] * inv, o_acc[mi][nj][2 * h + 1] * inv);
    }
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument(msg);
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

// ---- cuBLASLt fp32 linear: Y[M, N] (row stride ldy) = X[M, K] (row stride ldx) W[N, K]^T + bias (+ R) ----

void lt_check(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("mid_attention: cuBLASLt ") + what + " failed (status " +
                             std::to_string(static_cast<int>(st)) + ")");
}

struct LtLinear {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  ~LtLinear() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (ld) cublasLtMatrixLayoutDestroy(ld);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (desc) cublasLtMatmulDescDestroy(desc);
  }
};

void linear(cublasLtHandle_t lt, const float* x, std::int64_t ldx, const float* w, const float* bias, const float* r,
            std::int64_t ldr, float* y, std::int64_t ldy, std::int64_t m, std::int64_t n, std::int64_t k, bool tf32,
            void* ws, std::size_t ws_bytes, cudaStream_t stream) {
  // Column-major view: D[n x m] = W^T(op T on W viewed as [k x n]) * X(viewed as [k x m]).
  LtLinear d;
  lt_check(cublasLtMatmulDescCreate(&d.desc, tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUDA_R_32F),
           "MatmulDescCreate");
  const cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  lt_check(cublasLtMatmulDescSetAttribute(d.desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)), "TRANSA");
  lt_check(cublasLtMatmulDescSetAttribute(d.desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)), "TRANSB");
  const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
  lt_check(cublasLtMatmulDescSetAttribute(d.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)), "EPILOGUE");
  lt_check(cublasLtMatmulDescSetAttribute(d.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)), "BIAS");
  const auto u = [](std::int64_t v) { return static_cast<std::uint64_t>(v); };
  lt_check(cublasLtMatrixLayoutCreate(&d.la, CUDA_R_32F, u(k), u(n), k), "layout A");
  lt_check(cublasLtMatrixLayoutCreate(&d.lb, CUDA_R_32F, u(k), u(m), ldx), "layout B");
  lt_check(cublasLtMatrixLayoutCreate(&d.lc, CUDA_R_32F, u(n), u(m), r ? ldr : ldy), "layout C");
  lt_check(cublasLtMatrixLayoutCreate(&d.ld, CUDA_R_32F, u(n), u(m), ldy), "layout D");
  lt_check(cublasLtMatmulPreferenceCreate(&d.pref), "PreferenceCreate");
  lt_check(cublasLtMatmulPreferenceSetAttribute(d.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                                sizeof(ws_bytes)),
           "pref workspace");
  cublasLtMatmulHeuristicResult_t heur{};
  int found = 0;
  lt_check(cublasLtMatmulAlgoGetHeuristic(lt, d.desc, d.la, d.lb, d.lc, d.ld, d.pref, 1, &heur, &found), "heuristic");
  if (found == 0) throw std::runtime_error("mid_attention: no cuBLASLt algorithm for the fp32 linear");
  const float alpha = 1.0f, beta = r ? 1.0f : 0.0f;
  lt_check(cublasLtMatmul(lt, d.desc, &alpha, w, d.la, x, d.lb, &beta, r ? r : y, d.lc, y, d.ld, &heur.algo, ws, ws_bytes,
                          stream),
           "Matmul");
}

constexpr std::size_t kLtWorkspace = std::size_t{8} << 20;
constexpr std::size_t kAlign = 256;
std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

}  // namespace

void attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, float scale,
                    AttnPrecision precision, cudaStream_t stream) {
  for (const Tensor* t : {&q, &k, &v, &o}) {
    require(t->rank == 3 && t->dtype == DType::kF32 && t->device == Device::kCUDA && t->dim(2) == kD && t->stride[2] == 1,
            "attention_fp32: q/k/v/o must be f32 CUDA [B, L, 384] with unit inner stride, got " + t->describe());
    require(t->dim(0) == q.dim(0) && t->dim(1) == q.dim(1), "attention_fp32: q/k/v/o shapes differ");
    require(aligned16(t->data) && t->stride[1] % 4 == 0 && (t->dim(0) == 1 || t->stride[0] % 4 == 0),
            "attention_fp32: base pointers and strides must be 16-byte aligned");
  }
  require(o.stride[1] == kD, "attention_fp32: o rows must be contiguous");
  for (const Tensor* t : {&q, &k, &v}) {
    const auto* lo = static_cast<const char*>(t->data);
    const auto* hi = lo + (t->dim(0) - 1) * t->stride[0] * 4 + (t->dim(1) - 1) * t->stride[1] * 4 + kD * 4;
    const auto* olo = static_cast<const char*>(o.data);
    require(hi <= olo || olo + o.nbytes() <= lo, "attention_fp32: o must not overlap q/k/v");
  }
  require(q.dim(1) < (1ll << 31) && q.dim(0) <= 65535, "attention_fp32: L or B too large");
  const int len = static_cast<int>(q.dim(1));
  if (len == 0 || q.dim(0) == 0) return;
  AttnArgs a{q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), o.ptr<float>(), q.stride[0], k.stride[0], v.stride[0],
             o.stride[0], q.stride[1], k.stride[1], v.stride[1], len, scale * 1.4426950408889634f};
  const dim3 grid(static_cast<unsigned>((len + kBr - 1) / kBr), static_cast<unsigned>(q.dim(0)));
  if (precision == AttnPrecision::kTf32x3) {
    MI_CUDA_CHECK(cudaFuncSetAttribute(attention_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    attention_kernel<true><<<grid, kThreads, kSmemBytes, stream>>>(a);
  } else {
    MI_CUDA_CHECK(cudaFuncSetAttribute(attention_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    attention_kernel<false><<<grid, kThreads, kSmemBytes, stream>>>(a);
  }
  MI_CUDA_CHECK_LAUNCH();
}

namespace {

struct MidLayout {
  std::size_t xn, qkv, gn, lt, total;
};

MidLayout mid_layout(const Tensor& x) {
  const std::size_t tokens = static_cast<std::size_t>(x.dim(0) * x.dim(1) * x.dim(2));
  MidLayout l{};
  l.xn = 0;
  l.qkv = align_up(tokens * kD * sizeof(float));
  l.gn = l.qkv + align_up(tokens * 3 * kD * sizeof(float));
  l.lt = l.gn + align_up(group_norm_workspace_bytes(x, kGroupNormGroups));
  l.total = l.lt + kLtWorkspace;
  return l;
}

}  // namespace

std::size_t mid_attention_workspace_bytes(const Tensor& x) {
  require(x.rank == 4 && x.dim(3) == kD, "mid_attention: x must be [B,H,W,384]");
  return mid_layout(x).total;
}

void mid_attention(cublasLtHandle_t lt, const Tensor& x, const MidAttentionWeights& w, const Tensor* residual,
                   const Tensor& y, void* workspace, std::size_t workspace_bytes, AttnPrecision precision,
                   bool fast_linears, cudaStream_t stream) {
  require(x.rank == 4 && x.dim(3) == kD && x.dtype == DType::kF32 && x.device == Device::kCUDA && x.is_contiguous(),
          "mid_attention: x must be contiguous f32 CUDA [B,H,W,384], got " + x.describe());
  require(y.dtype == DType::kF32 && y.numel() == x.numel() && y.is_contiguous() && y.device == Device::kCUDA,
          "mid_attention: y must match x");
  if (residual)
    require(residual->dtype == DType::kF32 && residual->numel() == x.numel() && residual->is_contiguous(),
            "mid_attention: residual must match x");
  const Tensor* mats[] = {&w.wq, &w.wk, &w.wv, &w.wo};
  for (const Tensor* m : mats)
    require(m->dtype == DType::kF32 && m->numel() == kD * kD && m->is_contiguous(), "mid_attention: weights must be f32 [384,384]");
  const Tensor* vecs[] = {&w.bq, &w.bk, &w.bv, &w.bo, &w.gn_gamma, &w.gn_beta};
  for (const Tensor* m : vecs) require(m->dtype == DType::kF32 && m->numel() == kD, "mid_attention: biases/affine must be f32 [384]");
  const MidLayout l = mid_layout(x);
  require(workspace_bytes >= l.total && aligned16(workspace), "mid_attention: workspace too small or not 16-byte aligned");

  const std::int64_t batch = x.dim(0), len = x.dim(1) * x.dim(2), tokens = batch * len;
  char* ws = static_cast<char*>(workspace);
  auto* xn = reinterpret_cast<float*>(ws + l.xn);
  auto* qkv = reinterpret_cast<float*>(ws + l.qkv);
  Tensor xn_t = Tensor::view(xn, DType::kF32, Device::kCUDA, {x.dim(0), x.dim(1), x.dim(2), kD});

  group_norm(x, kGroupNormGroups, kResnetEps, w.gn_gamma, w.gn_beta, /*silu=*/false, xn_t, ws + l.gn, l.lt - l.gn, stream);
  const Tensor* proj_w[] = {&w.wq, &w.wk, &w.wv};
  const Tensor* proj_b[] = {&w.bq, &w.bk, &w.bv};
  for (int p = 0; p < 3; ++p)
    linear(lt, xn, kD, proj_w[p]->ptr<float>(), proj_b[p]->ptr<float>(), nullptr, 0, qkv + p * kD, 3 * kD, tokens, kD, kD,
           fast_linears, ws + l.lt, kLtWorkspace, stream);

  auto qkv_view = [&](int p) {
    Tensor t = Tensor::view(qkv + p * kD, DType::kF32, Device::kCUDA, {batch, len, kD});
    t.stride[1] = 3 * kD;
    t.stride[0] = len * 3 * kD;
    return t;
  };
  Tensor o = Tensor::view(xn, DType::kF32, Device::kCUDA, {batch, len, kD});  // GN output is dead after the projections
  // Known theoretical 1-ulp risk: SDPA's default scale is 1/sqrt(E) formed in double and rounded to fp32; this
  // fp32 expression is not guaranteed to round to the same float for every E (for E = 512 both give 0x3d3504f3).
  attention_fp32(qkv_view(0), qkv_view(1), qkv_view(2), o, 1.0f / std::sqrt(static_cast<float>(kD)), precision, stream);
  if (!residual) {
    linear(lt, xn, kD, w.wo.ptr<float>(), w.bo.ptr<float>(), nullptr, 0, y.ptr<float>(), kD, tokens, kD, kD, fast_linears,
           ws + l.lt, kLtWorkspace, stream);
    return;
  }
  // Reference order: round(to_out(o) + bias), then + residual. cuBLASLt's epilogue would add C before the bias,
  // so the GEMM writes into the (now free) QKV scratch and residual_add rounds the sum; y may alias the residual.
  linear(lt, xn, kD, w.wo.ptr<float>(), w.bo.ptr<float>(), nullptr, 0, qkv, kD, tokens, kD, kD, fast_linears, ws + l.lt,
         kLtWorkspace, stream);
  residual_add(Tensor::view(qkv, DType::kF32, Device::kCUDA, {x.dim(0), x.dim(1), x.dim(2), kD}), *residual, y, stream);
}

}  // namespace mi::vae
