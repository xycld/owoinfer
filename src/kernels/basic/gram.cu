#include "kernels/basic/gram.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <string>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"

namespace mi::kernels {
namespace {

constexpr int kTile = 64;   // output tile edge
constexpr int kRows = 32;   // rows of x staged per step
constexpr int kThreads = 256;

// One 64 x 64 tile (ti <= tj) per block; thread (ty, tx) of 16 x 16 owns rows ti*64 + ty + 16a and columns
// tj*64 + tx + 16b, a, b in [0, 4).
__global__ void __launch_bounds__(kThreads) gram_upper_kernel(const __nv_bfloat16* __restrict__ x, std::int64_t m, int k,
                                                               float* __restrict__ h) {
  // blockIdx.x -> (ti, tj), row-major over the upper triangle of the nt x nt tile grid.
  const int nt = k / kTile;
  int t = static_cast<int>(blockIdx.x), ti = 0;
  while (t >= nt - ti) {
    t -= nt - ti;
    ++ti;
  }
  const int tj = ti + t;
  __shared__ float as[kRows][kTile];
  __shared__ float bs[kRows][kTile];
  const int tx = static_cast<int>(threadIdx.x) % 16, ty = static_cast<int>(threadIdx.x) / 16;
  float acc[4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a) {
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;
  }
  // Staging: each thread loads 8 consecutive bf16 of one row for each tile (32 rows x 64 columns = 256 x 8).
  const int lr = static_cast<int>(threadIdx.x) / 8, lc = (static_cast<int>(threadIdx.x) % 8) * 8;
  for (std::int64_t r0 = 0; r0 < m; r0 += kRows) {
    const int rows = static_cast<int>(m - r0 < kRows ? m - r0 : kRows);
    if (lr < rows) {
      const __nv_bfloat16* row = x + (r0 + lr) * k;
      const uint4 va = *reinterpret_cast<const uint4*>(row + ti * kTile + lc);
      const uint4 vb = *reinterpret_cast<const uint4*>(row + tj * kTile + lc);
      const auto* pa = reinterpret_cast<const __nv_bfloat16*>(&va);
      const auto* pb = reinterpret_cast<const __nv_bfloat16*>(&vb);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        as[lr][lc + i] = __bfloat162float(pa[i]);
        bs[lr][lc + i] = __bfloat162float(pb[i]);
      }
    }
    __syncthreads();
    for (int r = 0; r < rows; ++r) {
      float av[4], bv[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) av[a] = as[r][ty + 16 * a];
#pragma unroll
      for (int b = 0; b < 4; ++b) bv[b] = bs[r][tx + 16 * b];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = __fmaf_rn(av[a], bv[b], acc[a][b]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    float* hr = h + static_cast<std::int64_t>(ti * kTile + ty + 16 * a) * k + tj * kTile;
#pragma unroll
    for (int b = 0; b < 4; ++b) hr[tx + 16 * b] = __fadd_rn(hr[tx + 16 * b], acc[a][b]);
  }
}

}  // namespace

void gram_accumulate_upper(const Tensor& x, float* h, cudaStream_t stream) {
  detail::check_rows("gram_accumulate_upper", "x", x, DType::kBF16);
  if (x.rank != 2) detail::fail("gram_accumulate_upper", "x must be [m, k], got " + x.describe());
  const std::int64_t m = x.dim(0), k = x.dim(1);
  if (k <= 0 || k % kTile != 0 || k > (1 << 20)) detail::fail("gram_accumulate_upper", "k = " + std::to_string(k) + " is not a positive multiple of 64");
  if (m > 1 && x.stride[0] != k) detail::fail("gram_accumulate_upper", "x rows must be contiguous, got " + x.describe());
  if (reinterpret_cast<std::uintptr_t>(x.data) % 16 != 0) detail::fail("gram_accumulate_upper", "x must be 16-byte aligned");
  if (h == nullptr) detail::fail("gram_accumulate_upper", "h is null");
  if (m == 0) return;
  const std::int64_t nt = k / kTile;
  gram_upper_kernel<<<static_cast<unsigned>(nt * (nt + 1) / 2), kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), m,
                                                                                         static_cast<int>(k), h);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels
