// sm_75 dtype conversion kernels: every tensor-core operand in the engine is bf16, while Turing's tensor cores take
// fp16. These kernels convert in place (the same 2-byte slot) or into a separate buffer; the attention wrapper relies
// on the round trip being exact for in-range values (fp16 has more mantissa bits than bf16, so bf16 -> fp16 is lossless
// and fp16 -> bf16 recovers the original value as long as it stayed in fp16's normal range).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>

#include "core/cuda_check.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::sm75 {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 65535;

__global__ void bf16_to_f16_flat(const __nv_bfloat16* __restrict__ src, __half* __restrict__ dst, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = __float2half_rn(__bfloat162float(src[i]));
  }
}

__global__ void f16_to_bf16_flat(const __half* __restrict__ src, __nv_bfloat16* __restrict__ dst, std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = __float2bfloat16_rn(__half2float(src[i]));
  }
}

// One block per token row, `dim` contiguous elements per (token, head) run.
__global__ void bf16_to_f16_row(const __nv_bfloat16* __restrict__ src, __half* __restrict__ dst, std::int64_t stride,
                                std::int64_t dim) {
  const __nv_bfloat16* s = src + static_cast<std::int64_t>(blockIdx.x) * stride;
  __half* d = dst + static_cast<std::int64_t>(blockIdx.x) * stride;
  for (std::int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    d[i] = __float2half_rn(__bfloat162float(s[i]));
  }
}

__global__ void f16_to_bf16_row(const __half* __restrict__ src, __nv_bfloat16* __restrict__ dst, std::int64_t stride,
                                std::int64_t dim) {
  const __half* s = src + static_cast<std::int64_t>(blockIdx.x) * stride;
  __nv_bfloat16* d = dst + static_cast<std::int64_t>(blockIdx.x) * stride;
  for (std::int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    d[i] = __float2bfloat16_rn(__half2float(s[i]));
  }
}

int blocks_for(std::int64_t n) {
  const std::int64_t blocks = (n + kThreads - 1) / kThreads;
  return static_cast<int>(blocks < kMaxBlocks ? (blocks > 0 ? blocks : 1) : kMaxBlocks);
}

// The attention tensors are [tokens, heads, dim] with a contiguous last dim, so every (token, head) pair is one run of
// `dim` elements at token * stride[0] + head * stride[1]. q/k/v may be views of a fused QKV output (non-contiguous
// heads), which the per-head loop handles.
void convert_3d_inplace(const Tensor& t, std::int64_t rows, std::int64_t heads, std::int64_t dim, bool to_f16,
                        cudaStream_t stream) {
  if (rows == 0 || heads == 0 || dim == 0) return;
  // One block per row on grid.x, whose limit is 2^31 - 1 (kMaxBlocks is the grid-stride cap of the flat kernels).
  if (rows > 0x7fffffff) throw std::length_error("sm75::convert_3d_inplace: more rows than one grid dimension holds");
  auto* base = static_cast<char*>(t.data);
  for (std::int64_t h = 0; h < heads; ++h) {
    char* p = base + h * static_cast<std::int64_t>(t.stride[1]) * 2;
    if (to_f16) {
      bf16_to_f16_row<<<static_cast<int>(rows), kThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(p), reinterpret_cast<__half*>(p), t.stride[0], dim);
    } else {
      f16_to_bf16_row<<<static_cast<int>(rows), kThreads, 0, stream>>>(
          reinterpret_cast<const __half*>(p), reinterpret_cast<__nv_bfloat16*>(p), t.stride[0], dim);
    }
  }
  MI_CUDA_CHECK_LAUNCH();
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

void convert_tensor_bf16_to_f16_inplace(const Tensor& t, std::int64_t rows, std::int64_t heads, std::int64_t dim,
                                        cudaStream_t stream) {
  convert_3d_inplace(t, rows, heads, dim, /*to_f16=*/true, stream);
}

void convert_tensor_f16_to_bf16_inplace(const Tensor& t, std::int64_t rows, std::int64_t heads, std::int64_t dim,
                                        cudaStream_t stream) {
  convert_3d_inplace(t, rows, heads, dim, /*to_f16=*/false, stream);
}

namespace {

__global__ void bf16_to_f16_rows(const __nv_bfloat16* __restrict__ src, __half* __restrict__ dst, std::int64_t cols,
                                 std::int64_t src_stride, std::int64_t dst_stride) {
  const __nv_bfloat16* s = src + static_cast<std::int64_t>(blockIdx.x) * src_stride;
  __half* d = dst + static_cast<std::int64_t>(blockIdx.x) * dst_stride;
  for (std::int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
    d[i] = __float2half_rn(__bfloat162float(s[i]));
  }
}

// out is bf16: `acc` may alias it (the GEMM writes fp16 into the output buffer, this pass converts in place).
__global__ void gemm_out_f16_bf16(__half* __restrict__ acc, __nv_bfloat16* __restrict__ out,
                                  const __nv_bfloat16* __restrict__ bias, std::int64_t cols, std::int64_t out_stride,
                                  std::int64_t acc_stride) {
  const __half* a = acc + static_cast<std::int64_t>(blockIdx.x) * acc_stride;
  __nv_bfloat16* o = out + static_cast<std::int64_t>(blockIdx.x) * out_stride;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = __half2float(a[c]) + (bias != nullptr ? __bfloat162float(bias[c]) : 0.f);
    o[c] = __float2bfloat16_rn(v);
  }
}

__global__ void gemm_out_f16_f32(const __half* __restrict__ acc, float* __restrict__ out, const float* __restrict__ bias,
                                 std::int64_t cols, std::int64_t out_stride, std::int64_t acc_stride) {
  const __half* a = acc + static_cast<std::int64_t>(blockIdx.x) * acc_stride;
  float* o = out + static_cast<std::int64_t>(blockIdx.x) * out_stride;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    o[c] = __half2float(a[c]) + (bias != nullptr ? bias[c] : 0.f);
  }
}

}  // namespace

void convert_rows_bf16_to_f16(void* dst, const void* src, std::int64_t rows, std::int64_t cols,
                              std::int64_t src_stride, std::int64_t dst_stride, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  bf16_to_f16_rows<<<static_cast<int>(rows), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(src), static_cast<__half*>(dst), cols, src_stride, dst_stride);
  MI_CUDA_CHECK_LAUNCH();
}

void convert_gemm_out_f16(void* out, const void* acc, const void* bias, bool bias_f32, bool out_f32,
                          std::int64_t rows, std::int64_t cols, std::int64_t out_stride, std::int64_t acc_stride,
                          cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  if (out_f32) {
    gemm_out_f16_f32<<<static_cast<int>(rows), kThreads, 0, stream>>>(
        static_cast<const __half*>(acc), static_cast<float*>(out), static_cast<const float*>(bias_f32 ? bias : nullptr),
        cols, out_stride, acc_stride);
  } else {
    gemm_out_f16_bf16<<<static_cast<int>(rows), kThreads, 0, stream>>>(
        static_cast<__half*>(const_cast<void*>(acc)), static_cast<__nv_bfloat16*>(out),
        static_cast<const __nv_bfloat16*>(bias), cols, out_stride, acc_stride);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::sm75
