// sm_75 VAE kernels: the reference decoder's convolutions and mid-block attention are TF32 tensor-core kernels (CUTLASS
// implicit GEMM with cp.async, and a TF32 m16n8k8 FlashAttention), neither of which exists on Turing. These fp32 SIMT
// replacements keep the reference precision (fp32 accumulate, fp32 softmax) at Turing speeds: the VAE is a small part
// of a generation, so plain fp32 is the right trade for a first working path.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "core/cuda_check.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

// ---------------------------------------------------------------- conv2d (NHWC, KRSC filters, stride 1, pad R/2)
// One block: an 8x8 output tile of 32 output channels. Input channels are walked in groups of 8 through shared memory
// (the input patch and the weights they multiply), each thread accumulating 2 pixels x 16 channels.
constexpr int kTW = 8, kTH = 8, kTK = 32, kTC = 8;
constexpr int kPX = kTW + 2, kPY = kTH + 2;
constexpr int kConvThreads = 128;

__global__ void conv2d_f32_sm75(const float* __restrict__ x, const float* __restrict__ w,
                                const float* __restrict__ bias, const float* __restrict__ residual,
                                float* __restrict__ y, int H, int W, int C, int K, int R, int kb_count) {
  __shared__ float xs[kPY * kPX * kTC];
  __shared__ float ws[kTK * 9 * kTC];
  const int n = blockIdx.z / kb_count;
  const int kb = blockIdx.z % kb_count;
  const int ox0 = blockIdx.x * kTW, oy0 = blockIdx.y * kTH;
  const int pad = (R - 1) / 2;
  const int tid = threadIdx.x;
  const int pixel = tid >> 1;      // 0..63
  const int cg = (tid & 1) * 16;   // first / second 16 output channels
  const int py = pixel / kTW, px = pixel % kTW;
  float acc[16] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int c0 = 0; c0 < C; c0 += kTC) {
    for (int i = tid; i < kPY * kPX * kTC; i += kConvThreads) {
      const int cc = i % kTC, p = i / kTC;
      const int yy = p / kPX, xx = p % kPX;
      const int gy = oy0 - pad + yy, gx = ox0 - pad + xx, gc = c0 + cc;
      xs[i] = (gy >= 0 && gy < H && gx >= 0 && gx < W && gc < C)
                  ? x[((static_cast<std::int64_t>(n) * H + gy) * W + gx) * C + gc]
                  : 0.f;
    }
    for (int i = tid; i < kTK * 9 * kTC; i += kConvThreads) {
      const int cc = i % kTC, t = i / kTC, rs = t % 9, kk = t / 9;
      const int kch = kb * kTK + kk, gc = c0 + cc;
      ws[i] = (kch < K && gc < C)
                  ? w[(static_cast<std::int64_t>(kch) * R + rs / R) * R * C + (rs % R) * C + gc]
                  : 0.f;
    }
    __syncthreads();
    for (int cc = 0; cc < kTC; ++cc) {
      if (c0 + cc >= C) break;
      for (int r = 0; r < R; ++r) {
        for (int s = 0; s < R; ++s) {
          const float xv = xs[((py + r) * kPX + (px + s)) * kTC + cc];
#pragma unroll
          for (int ch = 0; ch < 16; ++ch) acc[ch] += xv * ws[((cg + ch) * 9 + r * R + s) * kTC + cc];
        }
      }
    }
    __syncthreads();
  }

  const int ox = ox0 + px, oy = oy0 + py;
  if (ox >= W || oy >= H) return;
#pragma unroll
  for (int ch = 0; ch < 16; ++ch) {
    const int kch = kb * kTK + cg + ch;
    if (kch >= K) break;
    const std::int64_t idx = ((static_cast<std::int64_t>(n) * H + oy) * W + ox) * K + kch;
    float v = acc[ch] + bias[kch];
    if (residual != nullptr) v += residual[idx];
    y[idx] = v;
  }
}

// ---------------------------------------------------------------- mid-block attention (single head, fp32)
// One warp per query row: the lane holds D/32 elements of q, v and the output; the score is a warp reduction and the
// softmax runs online, so the L x L matrix never exists. D is a multiple of 32 (384 or 512).
constexpr int kAttnWarps = 4;

__global__ void attention_f32_sm75(const float* __restrict__ q, const float* __restrict__ k,
                                   const float* __restrict__ v, float* __restrict__ o, int L, int D, int q_rows,
                                   int k_rows, int v_rows, std::int64_t q_batch, std::int64_t k_batch,
                                   std::int64_t v_batch, std::int64_t o_batch, float scale) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kAttnWarps + warp;
  if (row >= L) return;
  const int nlanes = D / 32;
  const int b = blockIdx.y;
  const float* qrow = q + static_cast<std::int64_t>(b) * q_batch + static_cast<std::int64_t>(row) * q_rows;
  float qreg[16];
  float oacc[16];
  for (int i = 0; i < nlanes; ++i) {
    qreg[i] = qrow[lane + 32 * i];
    oacc[i] = 0.f;
  }
  float m = -1e30f, s = 0.f;
  for (int j = 0; j < L; ++j) {
    const float* krow = k + static_cast<std::int64_t>(b) * k_batch + static_cast<std::int64_t>(j) * k_rows;
    float dot = 0.f;
    for (int i = 0; i < nlanes; ++i) dot += qreg[i] * krow[lane + 32 * i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    dot *= scale;
    const float mn = fmaxf(m, dot);
    const float alpha = (m > -1e29f) ? expf(m - mn) : 0.f;
    const float p = expf(dot - mn);
    s = s * alpha + p;
    const float* vrow = v + static_cast<std::int64_t>(b) * v_batch + static_cast<std::int64_t>(j) * v_rows;
    for (int i = 0; i < nlanes; ++i) oacc[i] = oacc[i] * alpha + p * vrow[lane + 32 * i];
    m = mn;
  }
  const float inv = 1.f / s;
  float* orow = o + static_cast<std::int64_t>(b) * o_batch + static_cast<std::int64_t>(row) * D;
  for (int i = 0; i < nlanes; ++i) orow[lane + 32 * i] = oacc[i] * inv;
}

}  // namespace

void vae_conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
                cudaStream_t stream) {
  const int H = static_cast<int>(x.dim(1)), W = static_cast<int>(x.dim(2)), C = static_cast<int>(x.dim(3));
  const int K = static_cast<int>(w.dim(0)), R = static_cast<int>(w.dim(1));
  const int kb_count = (K + kTK - 1) / kTK;
  const dim3 grid(static_cast<unsigned>((W + kTW - 1) / kTW), static_cast<unsigned>((H + kTH - 1) / kTH),
                  static_cast<unsigned>(x.dim(0) * kb_count));
  conv2d_f32_sm75<<<grid, kConvThreads, 0, stream>>>(
      x.ptr<float>(), w.ptr<float>(), bias.ptr<float>(), residual != nullptr ? residual->ptr<float>() : nullptr,
      y.ptr<float>(), H, W, C, K, R, kb_count);
  MI_CUDA_CHECK_LAUNCH();
}

void vae_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, float scale,
                        cudaStream_t stream) {
  const int L = static_cast<int>(q.dim(1)), D = static_cast<int>(q.dim(2));
  const int batch = static_cast<int>(q.dim(0));
  const dim3 grid(static_cast<unsigned>((L + kAttnWarps - 1) / kAttnWarps), static_cast<unsigned>(batch));
  attention_f32_sm75<<<grid, kAttnWarps * 32, 0, stream>>>(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(),
                                                           o.ptr<float>(), L, D, static_cast<int>(q.stride[1]),
                                                           static_cast<int>(k.stride[1]),
                                                           static_cast<int>(v.stride[1]), q.stride[0], k.stride[0],
                                                           v.stride[0], o.stride[0], scale);
  MI_CUDA_CHECK_LAUNCH();
}

template <typename Out>
__global__ void kcrs_to_krsc_sm75(const float* __restrict__ src, Out* __restrict__ dst, int k, int c, int rs) {
  const std::int64_t total = static_cast<std::int64_t>(k) * c * rs;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int ci = static_cast<int>(i % c);
    const std::int64_t t = i / c;
    const int p = static_cast<int>(t % rs);
    const std::int64_t ki = t / rs;
    dst[i] = static_cast<Out>(src[(ki * c + ci) * rs + p]);
  }
}

void vae_weight_to_krsc(const Tensor& w_kcrs, const Tensor& w_krsc, cudaStream_t stream) {
  const int k = static_cast<int>(w_kcrs.dim(0)), c = static_cast<int>(w_kcrs.dim(1));
  const int rs = static_cast<int>(w_kcrs.dim(2) * w_kcrs.dim(3));
  const std::int64_t total = w_kcrs.numel();
  if (total == 0) return;
  const int blocks = static_cast<int>(std::min<std::int64_t>((total + 255) / 256, 4096));
  if (w_krsc.dtype == DType::kF32) {
    kcrs_to_krsc_sm75<float><<<blocks, 256, 0, stream>>>(w_kcrs.ptr<float>(), w_krsc.ptr<float>(), k, c, rs);
  } else {
    kcrs_to_krsc_sm75<__nv_bfloat16><<<blocks, 256, 0, stream>>>(w_kcrs.ptr<float>(), w_krsc.ptr<__nv_bfloat16>(), k,
                                                                 c, rs);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::sm75
