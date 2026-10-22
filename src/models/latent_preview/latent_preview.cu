#include "models/latent_preview/latent_preview.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "models/latent_preview/preview_coefficients.h"

namespace mi::latent_preview {
namespace {

using detail::kPreviewCoef;
using detail::kPreviewFeatures;
using detail::kPreviewTaps;

constexpr int kLatentChannels = 128;
constexpr int kVaeChannels = 32;

// The table travels as a __grid_constant__ kernel parameter (3.4 KB, within the 4 KB parameter limit): it stays in
// the constant bank (no per-thread copy), needs no upload or synchronization, and all threads of a warp reading the same
// coefficient is a broadcast.
struct Coefficients {
  float v[3][kPreviewFeatures];
};
// All of project_kernel's parameters, in order and with their alignment padding.
struct ProjectParams {
  const float* z;
  int h, w;
  float* out;
  Coefficients coef;
};
static_assert(sizeof(ProjectParams) <= 4096, "kernel parameter limit");

// Grid limits: blocks of 32 x 8 threads, gridDim.y <= 65535.
constexpr int kBlockX = 32, kBlockY = 8;
constexpr std::int64_t kMaxRows = std::int64_t{65535} * kBlockY;
constexpr std::int64_t kMaxCols = std::int64_t{1} << 24;

// One thread per H/8 pixel (y, x): out[y][x][:] = coef . [3x3 neighbourhood of the 32 unpacked channels, 1].
__global__ void project_kernel(const float* __restrict__ z, int h, int w, float* __restrict__ out,
                               const __grid_constant__ Coefficients coef) {
  const auto& c_coef = coef.v;
  const int h8 = 2 * h, w8 = 2 * w;
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w8 || y >= h8) return;
  const std::int64_t plane = static_cast<std::int64_t>(h) * w;
  float acc[3] = {c_coef[0][kPreviewFeatures - 1], c_coef[1][kPreviewFeatures - 1], c_coef[2][kPreviewFeatures - 1]};
  for (int dy = 0; dy < kPreviewTaps; ++dy) {
    const int yy = y + dy - 1;
    if (yy < 0 || yy >= h8) continue;
    for (int dx = 0; dx < kPreviewTaps; ++dx) {
      const int xx = x + dx - 1;
      if (xx < 0 || xx >= w8) continue;
      // VAE channel c at (yy, xx) is DiT channel 4c + 2 (yy & 1) + (xx & 1) at (yy / 2, xx / 2).
      const float* src = z + static_cast<std::int64_t>(2 * (yy & 1) + (xx & 1)) * plane + static_cast<std::int64_t>(yy >> 1) * w + (xx >> 1);
      const int tap = dy * kPreviewTaps + dx;
#pragma unroll 8
      for (int c = 0; c < kVaeChannels; ++c) {
        const float v = __ldg(src + static_cast<std::int64_t>(4 * c) * plane);
        const int f = c * kPreviewTaps * kPreviewTaps + tap;
        acc[0] = fmaf(c_coef[0][f], v, acc[0]);
        acc[1] = fmaf(c_coef[1][f], v, acc[1]);
        acc[2] = fmaf(c_coef[2][f], v, acc[2]);
      }
    }
  }
  float* o = out + (static_cast<std::int64_t>(y) * w8 + x) * 3;
  o[0] = acc[0];
  o[1] = acc[1];
  o[2] = acc[2];
}

// One thread per output pixel: bilinear sample (align_corners = false, source index clamped at 0 as torch does) of the
// H/8 image, then clamp to [0, 1] and round half to even to u8.
__global__ void resample_kernel(const float* __restrict__ in, int h8, int w8, std::uint8_t* __restrict__ rgb, int out_h,
                                int out_w, float scale_y, float scale_x) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= out_w || y >= out_h) return;
  const float sy = fmaxf((static_cast<float>(y) + 0.5f) * scale_y - 0.5f, 0.0f);
  const float sx = fmaxf((static_cast<float>(x) + 0.5f) * scale_x - 0.5f, 0.0f);
  const int y0 = min(static_cast<int>(sy), h8 - 1), x0 = min(static_cast<int>(sx), w8 - 1);
  const int y1 = min(y0 + 1, h8 - 1), x1 = min(x0 + 1, w8 - 1);
  const float ly = sy - static_cast<float>(y0), lx = sx - static_cast<float>(x0);
  const float* p00 = in + (static_cast<std::int64_t>(y0) * w8 + x0) * 3;
  const float* p01 = in + (static_cast<std::int64_t>(y0) * w8 + x1) * 3;
  const float* p10 = in + (static_cast<std::int64_t>(y1) * w8 + x0) * 3;
  const float* p11 = in + (static_cast<std::int64_t>(y1) * w8 + x1) * 3;
  std::uint8_t* o = rgb + (static_cast<std::int64_t>(y) * out_w + x) * 3;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const float top = (1.0f - lx) * __ldg(p00 + k) + lx * __ldg(p01 + k);
    const float bot = (1.0f - lx) * __ldg(p10 + k) + lx * __ldg(p11 + k);
    const float v = fminf(fmaxf((1.0f - ly) * top + ly * bot, 0.0f), 1.0f);
    o[k] = static_cast<std::uint8_t>(rintf(v * 255.0f));
  }
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument("mi::latent_preview: " + msg);
}

Coefficients make_coefficients() {
  Coefficients c{};
  for (int k = 0; k < 3; ++k)
    for (int f = 0; f < kPreviewFeatures; ++f) c.v[k][f] = kPreviewCoef[k][f];
  return c;
}

}  // namespace

std::size_t workspace_bytes(std::int64_t height, std::int64_t width) {
  require(height >= 1 && width >= 1, "latent grid must be at least 1x1");
  return static_cast<std::size_t>(4 * height * width * 3) * sizeof(float);
}

void preview(const Tensor& latents, const Tensor& rgb, void* workspace, std::size_t ws_bytes, cudaStream_t stream) {
  require(latents.device == Device::kCUDA && latents.dtype == DType::kF32 && latents.is_contiguous() &&
              (latents.rank == 3 || (latents.rank == 4 && latents.dim(0) == 1)) && latents.dim(-3) == kLatentChannels &&
              latents.dim(-2) >= 1 && latents.dim(-1) >= 1,
          "latents must be contiguous CUDA fp32 [128, h, w] or [1, 128, h, w], got " + latents.describe());
  require(rgb.device == Device::kCUDA && rgb.dtype == DType::kU8 && rgb.is_contiguous() && rgb.data != nullptr &&
              (rgb.rank == 3 || (rgb.rank == 4 && rgb.dim(0) == 1)) && rgb.dim(-1) == 3 && rgb.dim(-2) >= 1 && rgb.dim(-3) >= 1,
          "rgb must be contiguous CUDA u8 [H, W, 3] or [1, H, W, 3], got " + rgb.describe());
  const std::int64_t h = latents.dim(-2), w = latents.dim(-1), out_h = rgb.dim(-3), out_w = rgb.dim(-2);
  require(2 * h <= kMaxRows && 2 * w <= kMaxCols && out_h <= kMaxRows && out_w <= kMaxCols,
          "size out of range (rows <= " + std::to_string(kMaxRows) + ", columns <= " + std::to_string(kMaxCols) +
              " at the H/8 grid and the output)");
  require(workspace != nullptr && ws_bytes >= workspace_bytes(h, w) && reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0,
          "workspace too small or not 16-byte aligned");
  static const Coefficients coef = make_coefficients();
  const int h8 = static_cast<int>(2 * h), w8 = static_cast<int>(2 * w);
  auto* lowres = static_cast<float*>(workspace);
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid1(static_cast<unsigned>((w8 + kBlockX - 1) / kBlockX), static_cast<unsigned>((h8 + kBlockY - 1) / kBlockY));
  project_kernel<<<grid1, block, 0, stream>>>(latents.ptr<float>(), static_cast<int>(h), static_cast<int>(w), lowres, coef);
  MI_CUDA_CHECK_LAUNCH();
  const dim3 grid2(static_cast<unsigned>((out_w + kBlockX - 1) / kBlockX),
                   static_cast<unsigned>((out_h + kBlockY - 1) / kBlockY));
  resample_kernel<<<grid2, block, 0, stream>>>(lowres, h8, w8, static_cast<std::uint8_t*>(rgb.data), static_cast<int>(out_h),
                                               static_cast<int>(out_w), static_cast<float>(h8) / static_cast<float>(out_h),
                                               static_cast<float>(w8) / static_cast<float>(out_w));
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::latent_preview
