// Memory-bound VAE encoder ops: image preprocessing, the latent distribution + patchify + BN normalisation, and
// the padded conv_in weight layout. Exact ops use explicit __f*_rn intrinsics so nvcc cannot contract them into
// FMAs and change the reference rounding.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/vae/encoder_kernels.h"

namespace mi::vae {
namespace {

constexpr int kThreads = 256;

int grid_for(std::int64_t work) {
  return static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>((work + kThreads - 1) / kThreads, 65535 * 8)));
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument(msg);
}

void require_cuda_contig(const Tensor& t, const char* op, const char* name) {
  require(t.device == Device::kCUDA && t.is_contiguous() && t.data != nullptr,
          std::string(op) + ": " + name + " must be a contiguous CUDA tensor, got " + t.describe());
}

// numpy: np.float32(u8) / 255.0 (float32 division, round to nearest), then torch: 2.0 * x - 1.0 (2x is exact).
__device__ __forceinline__ float from_u8(std::uint8_t v) { return __fdiv_rn(static_cast<float>(v), 255.0f); }
__device__ __forceinline__ float from_f32(float v) { return v; }

template <typename T>
__global__ void preprocess_kernel(const T* __restrict__ rgb, float4* __restrict__ y, std::int64_t pixels) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < pixels;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    float v[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      float f;
      if constexpr (sizeof(T) == 1) f = from_u8(rgb[3 * i + c]); else f = from_f32(rgb[3 * i + c]);
      v[c] = __fsub_rn(__fmul_rn(2.0f, f), 1.0f);
    }
    y[i] = make_float4(v[0], v[1], v[2], 0.0f);
  }
}

// One thread per output element y[k, i, j] (NCHW, k = 4c + 2p + q): z = moments[2i+p, 2j+q, c] (mean) or
// mean + std * noise[c, 2i+p, 2j+q]; y = (z - bn_mean[k]) / sqrt(bn_var[k] + eps).
__global__ void latent_out_kernel(const float* __restrict__ moments, const float* __restrict__ bn_mean,
                                  const float* __restrict__ bn_var, float eps, const float* __restrict__ noise,
                                  float* __restrict__ y, int h2, int w2, std::int64_t total) {
  const int h = 2 * h2, w = 2 * w2;
  for (std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int j = static_cast<int>(idx % w2);
    const int i = static_cast<int>((idx / w2) % h2);
    const int k = static_cast<int>(idx / (static_cast<std::int64_t>(w2) * h2));
    const int c = k / 4, p = (k / 2) % 2, q = k % 2;
    const int row = 2 * i + p, col = 2 * j + q;
    const float* m = moments + (static_cast<std::int64_t>(row) * w + col) * 64;
    float z = m[c];
    if (noise != nullptr) {
      // DiagonalGaussianDistribution: logvar.clamp(-30, 20); std = exp(0.5 * logvar); mean + std * sample.
      const float logvar = fminf(fmaxf(m[32 + c], -30.0f), 20.0f);
      const float stdv = expf(__fmul_rn(0.5f, logvar));
      const float n = noise[(static_cast<std::int64_t>(c) * h + row) * w + col];
      z = __fadd_rn(z, __fmul_rn(stdv, n));
    }
    const float sd = __fsqrt_rn(__fadd_rn(bn_var[k], eps));
    y[idx] = __fdiv_rn(__fsub_rn(z, bn_mean[k]), sd);
  }
}

__global__ void krsc_pad_kernel(const float* __restrict__ src, float* __restrict__ dst, int k, int c, int cp, int rs) {
  const std::int64_t total = static_cast<std::int64_t>(k) * rs * cp;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int ci = static_cast<int>(i % cp);
    const std::int64_t t = i / cp;
    const int p = static_cast<int>(t % rs);
    const std::int64_t ki = t / rs;
    dst[i] = ci < c ? src[(ki * c + ci) * rs + p] : 0.0f;
  }
}

}  // namespace

void encoder_preprocess(const Tensor& rgb, const Tensor& y, cudaStream_t stream) {
  require_cuda_contig(rgb, "encoder_preprocess", "rgb");
  require_cuda_contig(y, "encoder_preprocess", "y");
  require((rgb.rank == 3 || (rgb.rank == 4 && rgb.dim(0) == 1)) && rgb.dim(-1) == 3 &&
              (rgb.dtype == DType::kU8 || rgb.dtype == DType::kF32),
          "encoder_preprocess: rgb must be u8 or f32 [H,W,3], got " + rgb.describe());
  require(y.rank == 4 && y.dim(0) == 1 && y.dim(1) == rgb.dim(-3) && y.dim(2) == rgb.dim(-2) && y.dim(3) == 4 &&
              y.dtype == DType::kF32 && reinterpret_cast<std::uintptr_t>(y.data) % 16 == 0,
          "encoder_preprocess: y must be 16-byte aligned f32 [1,H,W,4], got " + y.describe());
  const std::int64_t pixels = rgb.dim(-3) * rgb.dim(-2);
  if (pixels == 0) return;
  if (rgb.dtype == DType::kU8)
    preprocess_kernel<<<grid_for(pixels), kThreads, 0, stream>>>(rgb.ptr<std::uint8_t>(), static_cast<float4*>(y.data), pixels);
  else
    preprocess_kernel<<<grid_for(pixels), kThreads, 0, stream>>>(rgb.ptr<float>(), static_cast<float4*>(y.data), pixels);
  MI_CUDA_CHECK_LAUNCH();
}

void encoder_latent_out(const Tensor& moments, const Tensor& bn_mean, const Tensor& bn_var, float eps,
                        const Tensor* noise, const Tensor& y, cudaStream_t stream) {
  for (const Tensor* t : {&moments, &bn_mean, &bn_var, &y}) {
    require_cuda_contig(*t, "encoder_latent_out", "tensor");
    require(t->dtype == DType::kF32, "encoder_latent_out: all tensors must be f32");
  }
  require(moments.rank == 4 && moments.dim(0) == 1 && moments.dim(3) == 64 && moments.dim(1) % 2 == 0 &&
              moments.dim(2) % 2 == 0 && bn_mean.numel() == 128 && bn_var.numel() == 128,
          "encoder_latent_out: moments must be [1,h,w,64] with even h, w; bn stats [128]");
  const std::int64_t h2 = moments.dim(1) / 2, w2 = moments.dim(2) / 2;
  require(y.numel() == 128 * h2 * w2 && y.dim(-1) == w2 && y.dim(-2) == h2,
          "encoder_latent_out: y must be [1,128,h/2,w/2], got " + y.describe());
  if (noise != nullptr) {
    require_cuda_contig(*noise, "encoder_latent_out", "noise");
    require(noise->dtype == DType::kF32 && noise->numel() == 32 * moments.dim(1) * moments.dim(2) &&
                noise->dim(-1) == moments.dim(2) && noise->dim(-2) == moments.dim(1),
            "encoder_latent_out: noise must be f32 [1,32,h,w], got " + noise->describe());
  }
  const std::int64_t total = y.numel();
  if (total == 0) return;
  latent_out_kernel<<<grid_for(total), kThreads, 0, stream>>>(
      moments.ptr<float>(), bn_mean.ptr<float>(), bn_var.ptr<float>(), eps, noise ? noise->ptr<float>() : nullptr,
      y.ptr<float>(), static_cast<int>(h2), static_cast<int>(w2), total);
  MI_CUDA_CHECK_LAUNCH();
}

void conv_weight_to_krsc_padded(const Tensor& w_kcrs, const Tensor& w_krsc, cudaStream_t stream) {
  require(w_kcrs.rank == 4 && w_krsc.rank == 4 && w_kcrs.dtype == DType::kF32 && w_krsc.dtype == DType::kF32,
          "conv_weight_to_krsc_padded: expects f32 [K,C,R,S] -> [K,R,S,Cp]");
  require_cuda_contig(w_kcrs, "conv_weight_to_krsc_padded", "w_kcrs");
  require_cuda_contig(w_krsc, "conv_weight_to_krsc_padded", "w_krsc");
  const int k = static_cast<int>(w_kcrs.dim(0)), c = static_cast<int>(w_kcrs.dim(1));
  const int r = static_cast<int>(w_kcrs.dim(2)), s = static_cast<int>(w_kcrs.dim(3));
  const int cp = static_cast<int>(w_krsc.dim(3));
  require(w_krsc.dim(0) == k && w_krsc.dim(1) == r && w_krsc.dim(2) == s && cp >= c,
          "conv_weight_to_krsc_padded: shape mismatch " + w_kcrs.describe() + " -> " + w_krsc.describe());
  const std::int64_t total = w_krsc.numel();
  if (total == 0) return;
  krsc_pad_kernel<<<grid_for(total), kThreads, 0, stream>>>(w_kcrs.ptr<float>(), w_krsc.ptr<float>(), k, c, cp, r * s);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::vae
