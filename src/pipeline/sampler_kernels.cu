// Elementwise fp32 kernels of the sampling loop. Each torch op of the reference rounds once, so every step
// uses an explicitly rounded intrinsic: nvcc would otherwise contract a mul + add into an FMA.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "pipeline/sampler.h"

namespace mi::pipeline {
namespace {

constexpr int kThreads = 256;

void check(const Tensor& t, std::int64_t n, const char* who) {
  if (t.device != Device::kCUDA || t.dtype != DType::kF32 || !t.is_contiguous() || t.numel() != n) {
    throw std::invalid_argument(std::string("pipeline::") + who + ": expected contiguous CUDA fp32 with " +
                                std::to_string(n) + " elements, got " + t.describe());
  }
}

unsigned blocks(std::int64_t n) {
  const std::int64_t b = (n + kThreads - 1) / kThreads;
  return static_cast<unsigned>(b < 65535 * 8 ? b : 65535 * 8);
}

__global__ void cfg_kernel(const float* cond, const float* uncond, float g, float* out, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float u = uncond[i];
    out[i] = __fadd_rn(u, __fmul_rn(g, __fsub_rn(cond[i], u)));
  }
}

__global__ void slg_kernel(const float* pred, const float* cond, const float* skipped, float s, float* out,
                           std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    out[i] = __fadd_rn(pred[i], __fmul_rn(s, __fsub_rn(cond[i], skipped[i])));
  }
}

__global__ void euler_kernel(float* x, const float* v, float h, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    x[i] = __fadd_rn(x[i], __fmul_rn(h, v[i]));
  }
}

__global__ void ab2_kernel(float* x, const float* v, const float* v_prev, float h, float a, float b, std::int64_t n) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float d = __fsub_rn(__fmul_rn(a, v[i]), __fmul_rn(b, v_prev[i]));
    x[i] = __fadd_rn(x[i], __fmul_rn(h, d));
  }
}

}  // namespace

void cfg_combine(const Tensor& cond, const Tensor& uncond, float g, const Tensor& out, cudaStream_t stream) {
  const std::int64_t n = cond.numel();
  check(cond, n, "cfg_combine");
  check(uncond, n, "cfg_combine");
  check(out, n, "cfg_combine");
  if (n == 0) return;
  cfg_kernel<<<blocks(n), kThreads, 0, stream>>>(cond.ptr<float>(), uncond.ptr<float>(), g, out.ptr<float>(), n);
  MI_CUDA_CHECK_LAUNCH();
}

void slg_correct(const Tensor& pred, const Tensor& cond, const Tensor& skipped, float scale, const Tensor& out,
                 cudaStream_t stream) {
  const std::int64_t n = pred.numel();
  check(pred, n, "slg_correct");
  check(cond, n, "slg_correct");
  check(skipped, n, "slg_correct");
  check(out, n, "slg_correct");
  if (n == 0) return;
  slg_kernel<<<blocks(n), kThreads, 0, stream>>>(pred.ptr<float>(), cond.ptr<float>(), skipped.ptr<float>(), scale,
                                                 out.ptr<float>(), n);
  MI_CUDA_CHECK_LAUNCH();
}

void euler_update(const Tensor& x, const Tensor& v, float h, cudaStream_t stream) {
  const std::int64_t n = x.numel();
  check(x, n, "euler_update");
  check(v, n, "euler_update");
  if (n == 0) return;
  euler_kernel<<<blocks(n), kThreads, 0, stream>>>(x.ptr<float>(), v.ptr<float>(), h, n);
  MI_CUDA_CHECK_LAUNCH();
}

void ab2_update(const Tensor& x, const Tensor& v, const Tensor& v_prev, float h, float a, float b, cudaStream_t stream) {
  const std::int64_t n = x.numel();
  check(x, n, "ab2_update");
  check(v, n, "ab2_update");
  check(v_prev, n, "ab2_update");
  if (n == 0) return;
  ab2_kernel<<<blocks(n), kThreads, 0, stream>>>(x.ptr<float>(), v.ptr<float>(), v_prev.ptr<float>(), h, a, b, n);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::pipeline
