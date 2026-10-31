// sm_75 dtype conversion kernels: every tensor-core operand in the engine is bf16, while Turing's tensor cores take
// fp16. bf16 -> fp16 is lossless for values in fp16's normal range (fp16 has more mantissa bits than bf16), and
// fp16 -> bf16 recovers the original value. The weight conversions run in place (the same 2-byte slot); the GEMM's
// activation rows are converted into its scratch with a per-row power-of-two scale that keeps them in range.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "core/cuda_check.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;
constexpr float kF16Max = 65504.f;

// In place when dst == src (no __restrict__): each element is read and written by the same thread.
__global__ void bf16_to_f16_flat(const __nv_bfloat16* src, __half* dst, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = __float2half_rn(__bfloat162float(src[i]));
  }
}

__global__ void f16_to_bf16_flat(const __half* src, __nv_bfloat16* dst, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = __float2bfloat16_rn(__half2float(src[i]));
  }
}

int blocks_for(std::int64_t n) {
  const std::int64_t blocks = (n + kThreads - 1) / kThreads;
  return static_cast<int>(blocks < kMaxBlocks ? (blocks > 0 ? blocks : 1) : kMaxBlocks);
}

}  // namespace

bool available() { return true; }

bool needed(int major) { return major < 8; }

bool fp16_weights() {
  static std::atomic<int> cached[64];
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (device < 0 || device >= 64) return false;
  int major = cached[device].load(std::memory_order_relaxed);
  if (major == 0) {
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    cached[device].store(major, std::memory_order_relaxed);
  }
  return major < 8;
}

void convert_weight_inplace(Tensor& w, cudaStream_t stream) {
  if (w.data == nullptr) return;
  if (!fp16_weights()) return;
  if (w.dtype == DType::kF16) return;
  if (w.dtype != DType::kBF16) throw std::logic_error("sm75::convert_weight_inplace: weight is not bf16");
  convert_bf16_to_f16(w.data, w.data, w.numel(), stream);
  w.dtype = DType::kF16;
}

void convert_weight_storage_inplace(Tensor& w, cudaStream_t stream) {
  if (w.data == nullptr) return;
  if (!fp16_weights()) return;
  if (w.dtype != DType::kBF16 && w.dtype != DType::kF16) {
    throw std::logic_error("sm75::convert_weight_storage_inplace: weight is not bf16 storage");
  }
  convert_bf16_to_f16(w.data, w.data, w.numel(), stream);
  w.dtype = DType::kF16;
}

void convert_bf16_to_f16(void* dst, const void* src, std::int64_t n, cudaStream_t stream) {
  if (n <= 0) return;
  bf16_to_f16_flat<<<blocks_for(n), kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(src),
                                                           static_cast<__half*>(dst), n);
  MI_CUDA_CHECK_LAUNCH();
}

void convert_f16_to_bf16(void* dst, const void* src, std::int64_t n, cudaStream_t stream) {
  if (n <= 0) return;
  f16_to_bf16_flat<<<blocks_for(n), kThreads, 0, stream>>>(static_cast<const __half*>(src),
                                                           static_cast<__nv_bfloat16*>(dst), n);
  MI_CUDA_CHECK_LAUNCH();
}

namespace {

// One block per row: the row's largest magnitude, then the row converted with scale 2^-e, the smallest power of two
// that brings it within fp16's range (1 for every row already in range, so those convert exactly as plain bf16 ->
// fp16). Scaling by a power of two is exact, so a scaled row loses nothing but the low values it pushes below fp16's
// normal range; inv_scale[row] = 2^e undoes it in the GEMM's output pass. A row holding inf is left unscaled (a NaN
// is skipped by the max and carried through either way).
__global__ void bf16_to_f16_rows_scaled(const __nv_bfloat16* __restrict__ src, __half* __restrict__ dst,
                                        float* __restrict__ inv_scale, std::int64_t cols, std::int64_t src_stride,
                                        std::int64_t dst_stride) {
  __shared__ float warp_max[kThreads / 32];
  __shared__ float row_scale;
  const __nv_bfloat16* s = src + static_cast<std::int64_t>(blockIdx.x) * src_stride;
  __half* d = dst + static_cast<std::int64_t>(blockIdx.x) * dst_stride;
  float amax = 0.f;
  for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(s[i])));
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  if (threadIdx.x % 32 == 0) warp_max[threadIdx.x / 32] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = 0.f;
    for (int w = 0; w < kThreads / 32; ++w) m = fmaxf(m, warp_max[w]);
    int e = 0;
    if (m > kF16Max && isfinite(m)) e = static_cast<int>(ceilf(log2f(m / kF16Max)));
    while (e > 0 && ldexpf(m, -e) > kF16Max) ++e;  // log2f's rounding at an exact power of two
    row_scale = ldexpf(1.f, -e);
    inv_scale[blockIdx.x] = ldexpf(1.f, e);
  }
  __syncthreads();
  const float scale = row_scale;
  for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) d[i] = __float2half_rn(__bfloat162float(s[i]) * scale);
}

// out[r][c] = out_type(acc[r][c] * inv_scale[r] + bias[c]): the fp32 add before the single final rounding.
template <typename Out, typename Bias>
__global__ void gemm_out_f32(const float* __restrict__ acc, const float* __restrict__ inv_scale, Out* __restrict__ out,
                             const Bias* __restrict__ bias, std::int64_t cols, std::int64_t out_stride,
                             std::int64_t acc_stride) {
  const float* a = acc + static_cast<std::int64_t>(blockIdx.x) * acc_stride;
  Out* o = out + static_cast<std::int64_t>(blockIdx.x) * out_stride;
  const float sc = inv_scale[blockIdx.x];
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = a[c] * sc;
    if (bias != nullptr) v += static_cast<float>(bias[c]);
    if constexpr (std::is_same_v<Out, float>) {
      o[c] = v;
    } else {
      o[c] = __float2bfloat16_rn(v);
    }
  }
}

}  // namespace

void convert_rows_bf16_to_f16_scaled(void* dst, float* inv_scale, const void* src, std::int64_t rows,
                                     std::int64_t cols, std::int64_t src_stride, std::int64_t dst_stride,
                                     cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  bf16_to_f16_rows_scaled<<<static_cast<int>(rows), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(src), static_cast<__half*>(dst), inv_scale, cols, src_stride, dst_stride);
  MI_CUDA_CHECK_LAUNCH();
}

void convert_gemm_out_f32(void* out, bool out_f32, const float* acc, const float* inv_scale, const void* bias,
                          std::int64_t rows, std::int64_t cols, std::int64_t out_stride, std::int64_t acc_stride,
                          cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  const auto grid = static_cast<int>(rows);
  if (out_f32) {
    gemm_out_f32<float, float><<<grid, kThreads, 0, stream>>>(acc, inv_scale, static_cast<float*>(out),
                                                              static_cast<const float*>(bias), cols, out_stride,
                                                              acc_stride);
  } else {
    gemm_out_f32<__nv_bfloat16, __nv_bfloat16><<<grid, kThreads, 0, stream>>>(
        acc, inv_scale, static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(bias), cols, out_stride,
        acc_stride);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::sm75
