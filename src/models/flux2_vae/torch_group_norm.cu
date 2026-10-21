#include "models/flux2_vae/torch_group_norm.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/vae/vae_kernels.h"

namespace mi::flux2_vae {
namespace {

constexpr int kBlockReduceThreads = 512;  // cuda_utils::kCUDABlockReduceNumThreads
constexpr int kWarp = 32;
constexpr int kChainThreads = 128;
constexpr std::size_t kAlign = 256;

std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// WelfordData<float, int64_t> and WelfordOps (aten/src/ATen/native/SharedReduceOps.h) with the roundings nvcc
// gives torch's source: '/' is IEEE division, and x + y * z contracts to fma(y, z, x).
struct Welford {
  float mean, m2, nf;
  long long n;
};

__device__ __forceinline__ void welford_reduce(Welford& a, float x) {
  const long long n = a.n + 1;
  const float nf = static_cast<float>(n);
  const float delta = __fsub_rn(x, a.mean);
  const float mean = __fadd_rn(a.mean, __fdiv_rn(delta, nf));
  a.m2 = __fmaf_rn(delta, __fsub_rn(x, mean), a.m2);
  a.mean = mean;
  a.n = n;
  a.nf = nf;
}

__device__ __forceinline__ Welford welford_combine(const Welford& a, const Welford& b) {
  if (a.nf == 0.0f) return b;
  if (b.nf == 0.0f) return a;
  const float delta = __fsub_rn(b.mean, a.mean);
  const float count = __fadd_rn(a.nf, b.nf);
  const float nb_over_n = __fdiv_rn(b.nf, count);
  // a.m2 + b.m2 + delta * delta * a.nf * nb_over_n
  return {__fmaf_rn(delta, nb_over_n, a.mean),
          __fmaf_rn(__fmul_rn(__fmul_rn(delta, delta), a.nf), nb_over_n, __fadd_rn(a.m2, b.m2)), count, -1};
}

__device__ __forceinline__ Welford shfl_down(const Welford& v, int offset) {
  return {__shfl_down_sync(0xffffffffu, v.mean, offset), __shfl_down_sync(0xffffffffu, v.m2, offset),
          __shfl_down_sync(0xffffffffu, v.nf, offset), __shfl_down_sync(0xffffffffu, v.n, offset)};
}

// cuda_utils::WarpReduce.
__device__ __forceinline__ Welford warp_reduce(Welford v) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) v = welford_combine(v, shfl_down(v, offset));
  return v;
}

// One thread per Welford chain (n = 0, group g, torch thread t), g fastest so a warp reads 32 groups of one
// pixel. Chain t visits the group's NCHW elements j = t, t + T, ...: channel g*d + j / hw, pixel j % hw.
__global__ void chains_kernel(const float* __restrict__ x, long long hw, int c, int groups, int d, int threads,
                              float4* __restrict__ states) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= groups * threads) return;
  const int g = id % groups, t = id / groups;
  const long long count = static_cast<long long>(d) * hw;
  const float* xg = x + static_cast<long long>(g) * d;
  Welford w{0.0f, 0.0f, 0.0f, 0};
  long long j = t, p = t % hw;
  int ch = static_cast<int>(t / hw);
  const auto advance = [&] {
    j += threads;
    p += threads;
    while (p >= hw) p -= hw, ++ch;
  };
  // Four loads in flight ahead of the dependent Welford updates.
  while (j + 3LL * threads < count) {
    float v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      v[u] = xg[p * c + ch];
      advance();
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) welford_reduce(w, v[u]);
  }
  for (; j < count; advance()) welford_reduce(w, xg[p * c + ch]);
  states[static_cast<std::size_t>(g) * threads + t] = make_float4(w.mean, w.m2, w.nf, 0.0f);
}

// One block per group with torch's thread count: cuda_utils::WarpReduce / BlockReduce of the chains, project,
// rsqrtf, then ComputeFusedParamsCUDAKernel's b per channel (rounded per op).
__global__ void finalize_kernel(const float4* __restrict__ states, int d, float eps, const float* __restrict__ gamma,
                                const float* __restrict__ beta, float* __restrict__ stats, float* __restrict__ beta_out) {
  __shared__ Welford shared[kWarp];
  __shared__ float mean_rstd[2];
  const int g = blockIdx.x, tid = threadIdx.x, lid = tid % kWarp, wid = tid / kWarp;
  const float4 s = states[static_cast<std::size_t>(g) * blockDim.x + tid];
  Welford v{s.x, s.y, s.z, 0};
  v = warp_reduce(v);
  if (blockDim.x > kWarp) {
    __syncthreads();
    if (lid == 0) shared[wid] = v;
    __syncthreads();
    v = tid < static_cast<int>(blockDim.x) / kWarp ? shared[lid] : Welford{0.0f, 0.0f, 0.0f, 0};
    if (wid == 0) v = warp_reduce(v);
  }
  if (tid == 0) {
    const float var = __fdiv_rn(v.m2, v.nf);
    mean_rstd[0] = v.mean;
    mean_rstd[1] = rsqrtf(__fadd_rn(var, eps));  // c10::cuda::compat::rsqrt
    stats[2 * g] = 0.0f;
    stats[2 * g + 1] = mean_rstd[1];
  }
  __syncthreads();
  for (int i = tid; i < d; i += blockDim.x) {
    const int ch = g * d + i;
    const float a = __fmul_rn(mean_rstd[1], gamma[ch]);
    beta_out[ch] = __fadd_rn(__fmul_rn(-a, mean_rstd[0]), beta[ch]);
  }
}

struct Shape {
  long long hw;
  int c, d, threads;
};

Shape check(const Tensor& x, int groups) {
  if (x.rank != 4 || x.dim(0) != 1 || x.dtype != DType::kF32 || x.device != Device::kCUDA || !x.is_contiguous())
    throw std::invalid_argument("torch_group_norm: x must be contiguous CUDA fp32 [1,H,W,C], got " + x.describe());
  if (groups <= 0 || x.dim(3) % groups != 0 || x.dim(1) * x.dim(2) <= 0)
    throw std::invalid_argument("torch_group_norm: C must be a positive multiple of groups");
  Shape s{x.dim(1) * x.dim(2), static_cast<int>(x.dim(3)), static_cast<int>(x.dim(3) / groups), 0};
  s.threads = static_cast<long long>(s.d) * s.hw < kBlockReduceThreads ? kWarp : kBlockReduceThreads;
  return s;
}

struct Layout {
  std::size_t states, stats, beta, total;
};

Layout layout(const Shape& s, int groups) {
  Layout l{};
  l.states = 0;
  l.stats = align_up(static_cast<std::size_t>(groups) * s.threads * sizeof(float4));
  l.beta = l.stats + align_up(static_cast<std::size_t>(groups) * 2 * sizeof(float));
  l.total = l.beta + align_up(static_cast<std::size_t>(s.c) * sizeof(float));
  return l;
}

}  // namespace

std::size_t torch_group_norm_workspace_bytes(const Tensor& x, int groups) { return layout(check(x, groups), groups).total; }

void torch_group_norm(const Tensor& x, int groups, float eps, const Tensor& gamma, const Tensor& beta, bool silu,
                      const Tensor& y, void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  const Shape s = check(x, groups);
  const Layout l = layout(s, groups);
  if (workspace_bytes < l.total || reinterpret_cast<std::uintptr_t>(workspace) % 16 != 0)
    throw std::invalid_argument("torch_group_norm: workspace too small or misaligned");
  if (gamma.dtype != DType::kF32 || beta.dtype != DType::kF32 || gamma.numel() != s.c || beta.numel() != s.c)
    throw std::invalid_argument("torch_group_norm: gamma/beta must be fp32 [C]");
  char* ws = static_cast<char*>(workspace);
  auto* states = reinterpret_cast<float4*>(ws + l.states);
  auto* stats = reinterpret_cast<float*>(ws + l.stats);
  auto* beta_fused = reinterpret_cast<float*>(ws + l.beta);
  const int chains = groups * s.threads;
  chains_kernel<<<(chains + kChainThreads - 1) / kChainThreads, kChainThreads, 0, stream>>>(x.ptr<float>(), s.hw, s.c, groups,
                                                                                            s.d, s.threads, states);
  MI_CUDA_CHECK_LAUNCH();
  finalize_kernel<<<groups, s.threads, 0, stream>>>(states, s.d, eps, gamma.ptr<float>(), beta.ptr<float>(), stats, beta_fused);
  MI_CUDA_CHECK_LAUNCH();
  vae::group_norm_apply(x, Tensor::view(stats, DType::kF32, Device::kCUDA, {1, groups, 2}), gamma,
                        Tensor::view(beta_fused, DType::kF32, Device::kCUDA, {s.c}), silu, y, stream);
}

}  // namespace mi::flux2_vae
