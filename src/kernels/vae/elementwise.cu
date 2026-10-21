// Memory-bound VAE ops: nearest upsample, residual add, latent BN de-normalization + unpack, RGB8 postprocess
// and NCHW <-> NHWC conversion. The exact ops use explicit __f*_rn intrinsics so nvcc cannot contract them
// into FMAs and change the reference rounding.
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/vae/vae_kernels.h"

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
  require(t.device == Device::kCUDA && t.is_contiguous(),
          std::string(op) + ": " + name + " must be a contiguous CUDA tensor, got " + t.describe());
}

// Copies 16-byte vectors: each output pixel row of C elements is the source pixel's row.
__global__ void upsample2x_kernel(const uint4* __restrict__ x, uint4* __restrict__ y, int h, int w, int vecs_per_pixel,
                                  std::int64_t total) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int v = static_cast<int>(i % vecs_per_pixel);
    std::int64_t p = i / vecs_per_pixel;  // output pixel index over [N, 2H, 2W]
    const int ox = static_cast<int>(p % (2 * w));
    p /= 2 * w;
    const int oy = static_cast<int>(p % (2 * h));
    const std::int64_t n = p / (2 * h);
    y[i] = x[((n * h + oy / 2) * w + ox / 2) * vecs_per_pixel + v];
  }
}

template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* y, std::int64_t n);  // any of them may alias

template <>
__global__ void add_kernel<float>(const float* a, const float* b, float* y, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
    y[i] = __fadd_rn(a[i], b[i]);
}

template <>
__global__ void add_kernel<__nv_bfloat16>(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* y,
                                          std::int64_t n) {
  // torch bf16 add: fp32 opmath, one rounding to bf16.
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
    y[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(a[i]), __bfloat162float(b[i])));
}

// y[b, 2i+p, 2j+q, c] = z[b, 4c+2p+q, i, j] * sqrt(var + eps) + mean   (all [*] fp32, round-to-nearest each)
__global__ void bn_unpack_kernel(const float* __restrict__ z, const float* __restrict__ mean,
                                 const float* __restrict__ var, float eps, float* __restrict__ y, int h, int w,
                                 std::int64_t total) {
  for (std::int64_t o = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; o < total;
       o += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int c = static_cast<int>(o % 32);
    std::int64_t t = o / 32;
    const int ox = static_cast<int>(t % (2 * w));
    t /= 2 * w;
    const int oy = static_cast<int>(t % (2 * h));
    const std::int64_t b = t / (2 * h);
    const int ch = 4 * c + 2 * (oy & 1) + (ox & 1);
    const float scale = __fsqrt_rn(__fadd_rn(var[ch], eps));
    const float v = z[((b * 128 + ch) * h + oy / 2) * w + ox / 2];
    y[o] = __fadd_rn(__fmul_rn(v, scale), mean[ch]);
  }
}

__global__ void rgb8_kernel(const float* __restrict__ x, std::uint8_t* __restrict__ out, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    float v = x[i];
    if (v != v) {  // NaN: torch clamp keeps it and numpy's uint8 cast is undefined; pin it to 0
      out[i] = 0;
      continue;
    }
    v = fminf(fmaxf(v, -1.0f), 1.0f);               // images.clamp(-1, 1)
    v = __fadd_rn(__fmul_rn(v, 0.5f), 0.5f);         // images * 0.5 + 0.5 (two torch kernels)
    v = fminf(fmaxf(v, 0.0f), 1.0f);                 // .clamp(0, 1)
    v = __fmul_rn(v, 255.0f);                        // numpy float32 * 255
    out[i] = static_cast<std::uint8_t>(rintf(v));    // np.round: half to even; astype(uint8)
  }
}

// Batched 2-D transpose of [B, R, Cc] -> [B, Cc, R] through a 32x33 shared tile. Tiles are flattened into
// blockIdx.x because either dimension can exceed gridDim.y's 65535 limit (H*W at 2048^2).
__global__ void transpose_kernel(const float* __restrict__ x, float* __restrict__ y, int rows, int cols, int tiles_c) {
  __shared__ float tile[32][33];
  const std::int64_t base = static_cast<std::int64_t>(blockIdx.z) * rows * cols;
  const int c0 = static_cast<int>(blockIdx.x % tiles_c) * 32, r0 = static_cast<int>(blockIdx.x / tiles_c) * 32;
  for (int i = threadIdx.y; i < 32; i += blockDim.y) {
    const int r = r0 + i, c = c0 + threadIdx.x;
    if (r < rows && c < cols) tile[i][threadIdx.x] = x[base + static_cast<std::int64_t>(r) * cols + c];
  }
  __syncthreads();
  for (int i = threadIdx.y; i < 32; i += blockDim.y) {
    const int c = c0 + i, r = r0 + threadIdx.x;
    if (r < rows && c < cols) y[base + static_cast<std::int64_t>(c) * rows + r] = tile[threadIdx.x][i];
  }
}

void transpose(const float* x, float* y, std::int64_t batch, std::int64_t rows, std::int64_t cols, cudaStream_t stream) {
  const std::int64_t tiles_c = (cols + 31) / 32, tiles = tiles_c * ((rows + 31) / 32);
  require(batch <= 65535 && rows * cols < (1ll << 40) && tiles < (1ll << 31) && rows < (1ll << 31) && cols < (1ll << 31),
          "layout conversion: tensor too large");
  dim3 grid(static_cast<unsigned>(tiles), 1, static_cast<unsigned>(batch));
  transpose_kernel<<<grid, dim3(32, 8), 0, stream>>>(x, y, static_cast<int>(rows), static_cast<int>(cols),
                                                     static_cast<int>(tiles_c));
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void upsample_nearest2x(const Tensor& x, const Tensor& y, cudaStream_t stream) {
  require_cuda_contig(x, "upsample_nearest2x", "x");
  require_cuda_contig(y, "upsample_nearest2x", "y");
  require(x.rank == 4 && y.rank == 4 && x.dtype == y.dtype && (x.dtype == DType::kF32 || x.dtype == DType::kBF16) &&
              y.dim(0) == x.dim(0) && y.dim(1) == 2 * x.dim(1) && y.dim(2) == 2 * x.dim(2) && y.dim(3) == x.dim(3),
          "upsample_nearest2x: expects x [N,H,W,C] and y [N,2H,2W,C] of one dtype, got " + x.describe() + " -> " +
              y.describe());
  const std::int64_t row_bytes = x.dim(3) * static_cast<std::int64_t>(dtype_size(x.dtype));
  require(row_bytes % 16 == 0 && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 &&
              reinterpret_cast<std::uintptr_t>(y.data) % 16 == 0,
          "upsample_nearest2x: C * sizeof(dtype) must be a multiple of 16 bytes and x, y 16-byte aligned");
  const int vecs = static_cast<int>(row_bytes / 16);
  const std::int64_t total = y.numel() * static_cast<std::int64_t>(dtype_size(y.dtype)) / 16;
  if (total == 0) return;
  upsample2x_kernel<<<grid_for(total), kThreads, 0, stream>>>(static_cast<const uint4*>(x.data), static_cast<uint4*>(y.data),
                                                              static_cast<int>(x.dim(1)), static_cast<int>(x.dim(2)), vecs,
                                                              total);
  MI_CUDA_CHECK_LAUNCH();
}

void residual_add(const Tensor& a, const Tensor& b, const Tensor& y, cudaStream_t stream) {
  require_cuda_contig(a, "residual_add", "a");
  require_cuda_contig(b, "residual_add", "b");
  require_cuda_contig(y, "residual_add", "y");
  require(a.dtype == b.dtype && a.dtype == y.dtype && a.numel() == b.numel() && a.numel() == y.numel(),
          "residual_add: a, b, y must match in dtype and size");
  const std::int64_t n = a.numel();
  if (n == 0) return;
  if (a.dtype == DType::kF32) {
    add_kernel<float><<<grid_for(n), kThreads, 0, stream>>>(a.ptr<float>(), b.ptr<float>(), y.ptr<float>(), n);
  } else if (a.dtype == DType::kBF16) {
    add_kernel<__nv_bfloat16><<<grid_for(n), kThreads, 0, stream>>>(a.ptr<__nv_bfloat16>(), b.ptr<__nv_bfloat16>(),
                                                                    y.ptr<__nv_bfloat16>(), n);
  } else {
    throw std::invalid_argument("residual_add: dtype must be f32 or bf16");
  }
  MI_CUDA_CHECK_LAUNCH();
}

void bn_denorm_unpack(const Tensor& z, const Tensor& bn_mean, const Tensor& bn_var, float eps, const Tensor& y,
                      cudaStream_t stream) {
  for (const Tensor* t : {&z, &bn_mean, &bn_var, &y}) {
    require_cuda_contig(*t, "bn_denorm_unpack", "tensor");
    require(t->dtype == DType::kF32, "bn_denorm_unpack: all tensors must be f32");
  }
  require(z.rank == 4 && z.dim(1) == 128 && bn_mean.numel() == 128 && bn_var.numel() == 128 && y.rank == 4 &&
              y.dim(0) == z.dim(0) && y.dim(1) == 2 * z.dim(2) && y.dim(2) == 2 * z.dim(3) && y.dim(3) == 32,
          "bn_denorm_unpack: expects z [B,128,h,w] -> y [B,2h,2w,32], got " + z.describe() + " -> " + y.describe());
  const std::int64_t total = y.numel();
  if (total == 0) return;
  bn_unpack_kernel<<<grid_for(total), kThreads, 0, stream>>>(z.ptr<float>(), bn_mean.ptr<float>(), bn_var.ptr<float>(),
                                                             eps, y.ptr<float>(), static_cast<int>(z.dim(2)),
                                                             static_cast<int>(z.dim(3)), total);
  MI_CUDA_CHECK_LAUNCH();
}

void postprocess_to_rgb8(const Tensor& x, const Tensor& rgb, cudaStream_t stream) {
  require_cuda_contig(x, "postprocess_to_rgb8", "x");
  require_cuda_contig(rgb, "postprocess_to_rgb8", "rgb");
  require(x.dtype == DType::kF32 && rgb.dtype == DType::kU8 && x.rank == 4 && x.dim(3) == 3 && rgb.numel() == x.numel(),
          "postprocess_to_rgb8: expects f32 [B,H,W,3] -> u8 [B,H,W,3]");
  const std::int64_t n = x.numel();
  if (n == 0) return;
  rgb8_kernel<<<grid_for(n), kThreads, 0, stream>>>(x.ptr<float>(), rgb.ptr<std::uint8_t>(), n);
  MI_CUDA_CHECK_LAUNCH();
}

void nchw_to_nhwc(const Tensor& x, const Tensor& y, cudaStream_t stream) {
  require_cuda_contig(x, "nchw_to_nhwc", "x");
  require_cuda_contig(y, "nchw_to_nhwc", "y");
  require(x.rank == 4 && y.rank == 4 && x.dtype == DType::kF32 && y.dtype == DType::kF32 && y.dim(0) == x.dim(0) &&
              y.dim(1) == x.dim(2) && y.dim(2) == x.dim(3) && y.dim(3) == x.dim(1),
          "nchw_to_nhwc: expects f32 [N,C,H,W] -> [N,H,W,C]");
  if (x.numel() == 0) return;
  transpose(x.ptr<float>(), y.ptr<float>(), x.dim(0), x.dim(1), x.dim(2) * x.dim(3), stream);
}

void nhwc_to_nchw(const Tensor& x, const Tensor& y, cudaStream_t stream) {
  require_cuda_contig(x, "nhwc_to_nchw", "x");
  require_cuda_contig(y, "nhwc_to_nchw", "y");
  require(x.rank == 4 && y.rank == 4 && x.dtype == DType::kF32 && y.dtype == DType::kF32 && y.dim(0) == x.dim(0) &&
              y.dim(1) == x.dim(3) && y.dim(2) == x.dim(1) && y.dim(3) == x.dim(2),
          "nhwc_to_nchw: expects f32 [N,H,W,C] -> [N,C,H,W]");
  if (x.numel() == 0) return;
  transpose(x.ptr<float>(), y.ptr<float>(), x.dim(0), x.dim(1) * x.dim(2), x.dim(3), stream);
}

}  // namespace mi::vae
