#include "kernels/lora/lora_merge.h"

#include <stdexcept>
#include <string>

#include "core/cuda_check.h"

namespace mi::kernels::lora {
namespace {

// A CTA merges a kRows x kCols tile of w; each thread kMicro rows x kMicro columns (columns kThreadsX apart, so the
// warp's shared-memory reads of a down row are consecutive). Rank columns go through shared memory kRankStep at a
// time, widened to fp64 once there. fp64 throughput bounds the kernel (each element costs R fused multiply-adds).
constexpr int kMicro = 4;
constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;
constexpr int kRows = kThreadsY * kMicro;   // 32
constexpr int kCols = kThreadsX * kMicro;   // 128
constexpr int kRankStep = 32;
constexpr int kThreads = kThreadsX * kThreadsY;

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16); }

// quant::float_to_bf16 for a finite value (round to nearest even on the bits; subnormals kept).
__device__ __forceinline__ std::uint16_t float_to_bf16_bits(float v) {
  const std::uint32_t u = __float_as_uint(v);
  if (v != v) return static_cast<std::uint16_t>((u >> 16) | 0x40);
  return static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__global__ void __launch_bounds__(kThreads) merge_low_rank_kernel(std::uint16_t* __restrict__ w, std::int64_t ldw,
                                                                  int n, int k, const float* __restrict__ up,
                                                                  std::int64_t ldu, const float* __restrict__ down,
                                                                  std::int64_t ldd, const int* __restrict__ bounds,
                                                                  const float* __restrict__ coef, int adapters,
                                                                  int* __restrict__ bad) {
  __shared__ double su[kRows][kRankStep];
  __shared__ double sd[kRankStep][kCols];
  const int tx = static_cast<int>(threadIdx.x) % kThreadsX, ty = static_cast<int>(threadIdx.x) / kThreadsX;
  const int row0 = static_cast<int>(blockIdx.y) * kRows, col0 = static_cast<int>(blockIdx.x) * kCols;

  float v[kMicro][kMicro];
#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
    const int r = row0 + ty * kMicro + i;
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int c = col0 + tx + j * kThreadsX;
      v[i][j] = r < n && c < k ? bf16_bits_to_float(w[r * ldw + c]) : 0.0f;
    }
  }

  for (int a = 0; a < adapters; ++a) {
    const int begin = bounds[a], end = bounds[a + 1];
    double acc[kMicro][kMicro];
#pragma unroll
    for (int i = 0; i < kMicro; ++i) {
#pragma unroll
      for (int j = 0; j < kMicro; ++j) acc[i][j] = 0.0;
    }
    for (int j0 = begin; j0 < end; j0 += kRankStep) {
      const int steps = min(kRankStep, end - j0);
      __syncthreads();  // the previous step's tiles are no longer read
      for (int e = static_cast<int>(threadIdx.x); e < kRows * kRankStep; e += kThreads) {
        const int rr = e / kRankStep, jj = e % kRankStep, r = row0 + rr;
        su[rr][jj] = r < n && jj < steps ? static_cast<double>(up[r * ldu + j0 + jj]) : 0.0;
      }
      for (int e = static_cast<int>(threadIdx.x); e < kRankStep * kCols; e += kThreads) {
        const int jj = e / kCols, cc = e % kCols, c = col0 + cc;
        sd[jj][cc] = c < k && jj < steps ? static_cast<double>(down[(j0 + jj) * ldd + c]) : 0.0;
      }
      __syncthreads();
      for (int jj = 0; jj < steps; ++jj) {
        double d[kMicro];
#pragma unroll
        for (int j = 0; j < kMicro; ++j) d[j] = sd[jj][tx + j * kThreadsX];
#pragma unroll
        for (int i = 0; i < kMicro; ++i) {
          const double u = su[ty * kMicro + i][jj];
#pragma unroll
          for (int j = 0; j < kMicro; ++j) acc[i][j] = __fma_rn(u, d[j], acc[i][j]);
        }
      }
    }
    const float s = coef[a];
#pragma unroll
    for (int i = 0; i < kMicro; ++i) {
#pragma unroll
      for (int j = 0; j < kMicro; ++j) v[i][j] = __fadd_rn(v[i][j], __fmul_rn(s, __double2float_rn(acc[i][j])));
    }
  }

  bool finite = true;
#pragma unroll
  for (int i = 0; i < kMicro; ++i) {
    const int r = row0 + ty * kMicro + i;
#pragma unroll
    for (int j = 0; j < kMicro; ++j) {
      const int c = col0 + tx + j * kThreadsX;
      if (r < n && c < k) {
        finite = finite && isfinite(v[i][j]);
        w[r * ldw + c] = float_to_bf16_bits(v[i][j]);
      }
    }
  }
  if (!finite) atomicExch(bad, 1);
}

}  // namespace

void merge_low_rank_bf16(__nv_bfloat16* w, std::int64_t ldw, std::int64_t n, std::int64_t k, const float* up,
                         std::int64_t ldu, const float* down, std::int64_t ldd, const int* bounds, const float* coef,
                         int adapters, int* bad, cudaStream_t stream) {
  constexpr std::int64_t kMaxIndex = 0x7FFFFFFF;
  if (w == nullptr || up == nullptr || down == nullptr || bounds == nullptr || coef == nullptr || bad == nullptr ||
      n < 0 || k < 0 || adapters < 0 || ldw < k || ldd < k || ldu < 0 || n > kMaxIndex || k > kMaxIndex) {
    throw std::invalid_argument("mi::kernels::lora::merge_low_rank_bf16: bad arguments (n " + std::to_string(n) +
                                ", k " + std::to_string(k) + ", ldw " + std::to_string(ldw) + ", ldu " +
                                std::to_string(ldu) + ", ldd " + std::to_string(ldd) + ", adapters " +
                                std::to_string(adapters) + ")");
  }
  if (n == 0 || k == 0 || adapters == 0) return;
  const dim3 grid(static_cast<unsigned>((k + kCols - 1) / kCols), static_cast<unsigned>((n + kRows - 1) / kRows));
  if (grid.y > 65535) throw std::invalid_argument("mi::kernels::lora::merge_low_rank_bf16: too many rows per call");
  std::uint16_t* const wb = reinterpret_cast<std::uint16_t*>(w);
  merge_low_rank_kernel<<<grid, kThreads, 0, stream>>>(wb, ldw, static_cast<int>(n), static_cast<int>(k), up, ldu, down,
                                                        ldd, bounds, coef, adapters, bad);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::lora
