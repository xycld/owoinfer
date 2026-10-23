#include "pipeline/accel/accel_kernels.h"

#include <cuda_bf16.h>

#include "core/cuda_check.h"

namespace mi::pipeline::accel_kernels {
namespace {

constexpr int kThreads = 256;

unsigned blocks(std::int64_t n) {
  const std::int64_t b = (n + kThreads - 1) / kThreads;
  return static_cast<unsigned>(b < 65535 * 8 ? b : 65535 * 8);
}

#define MI_GRID_LOOP(i, n)                                                                         \
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < (n); \
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)

__device__ __forceinline__ float f(__nv_bfloat16 v) { return __bfloat162float(v); }

__global__ void sub_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, std::int64_t n) {
  MI_GRID_LOOP(i, n) { out[i] = __float2bfloat16_rn(__fsub_rn(f(a[i]), f(b[i]))); }
}

__global__ void add_kernel(__nv_bfloat16* x, const __nv_bfloat16* r, std::int64_t n) {
  MI_GRID_LOOP(i, n) { x[i] = __float2bfloat16_rn(__fadd_rn(f(x[i]), f(r[i]))); }
}

__global__ void diff_kernel(const __nv_bfloat16* x1, const __nv_bfloat16* x0, const __nv_bfloat16* ref, std::int64_t n,
                            double* partials) {
  __shared__ double s_diff[kThreads], s_ref[kThreads];
  double d = 0, m = 0;
  MI_GRID_LOOP(i, n) {
    const float r = f(__float2bfloat16_rn(__fsub_rn(f(x1[i]), f(x0[i]))));
    const float p = f(ref[i]);
    d += fabs(static_cast<double>(r) - static_cast<double>(p));
    m += fabs(static_cast<double>(p));
  }
  s_diff[threadIdx.x] = d;
  s_ref[threadIdx.x] = m;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (static_cast<int>(threadIdx.x) < stride) {
      s_diff[threadIdx.x] += s_diff[threadIdx.x + stride];
      s_ref[threadIdx.x] += s_ref[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[2 * blockIdx.x] = s_diff[0];
    partials[2 * blockIdx.x + 1] = s_ref[0];
  }
}

}  // namespace

void sub_bf16(const void* a, const void* b, void* out, std::int64_t n, cudaStream_t stream) {
  if (n <= 0) return;
  sub_kernel<<<blocks(n), kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(a), static_cast<const __nv_bfloat16*>(b),
                                                 static_cast<__nv_bfloat16*>(out), n);
  MI_CUDA_CHECK(cudaGetLastError());
}

void add_bf16(void* x, const void* r, std::int64_t n, cudaStream_t stream) {
  if (n <= 0) return;
  add_kernel<<<blocks(n), kThreads, 0, stream>>>(static_cast<__nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(r), n);
  MI_CUDA_CHECK(cudaGetLastError());
}

void residual_diff_partials(const void* x1, const void* x0, const void* ref, std::int64_t n, double* partials,
                            cudaStream_t stream) {
  diff_kernel<<<kDiffBlocks, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x1),
                                                    static_cast<const __nv_bfloat16*>(x0),
                                                    static_cast<const __nv_bfloat16*>(ref), n, partials);
  MI_CUDA_CHECK(cudaGetLastError());
}

}  // namespace mi::pipeline::accel_kernels
