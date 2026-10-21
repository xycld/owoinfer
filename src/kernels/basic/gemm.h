// cuBLASLt GEMM for nn.Linear: out = x · wᵀ (+ bias), bf16 inputs, fp32 compute and accumulation.
//
// Heuristic algorithm choice is cached per (shape, strides, dtypes, alignment, workspace) on first use;
// later calls with the same key only launch. The caller owns the workspace (no hidden allocation).
// Bias goes through cuBLASLt's BIAS epilogue: added in fp32 to the accumulator before the single final
// rounding, as torch F.linear's cublasLt path (gemm_and_bias) does.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>

#include "core/tensor.h"

namespace mi::kernels {

class Gemm {
 public:
  // Workspace size that lets cuBLASLt pick any algorithm (split-K included) on sm_80..sm_120.
  static constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20;

  Gemm();  // creates the cuBLASLt handle on the current device
  ~Gemm();
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;

  // out[..., N] = x[..., K] · w[N, K]ᵀ + bias[N].
  // x: bf16, rows of K with a contiguous last dim whose leading dims flatten to one row stride (M rows).
  // w: bf16 [N, K], contiguous rows. bias: nullptr or [N] of the output dtype (cuBLASLt has no bf16 bias
  // with f32 output; widen it with kernels::cast).
  // out: bf16 or f32 [..., N], M rows with one row stride. workspace: 256-byte aligned, or null with 0 bytes.
  // Thread-safe; launches on `stream` only.
  void linear(const Tensor& out, const Tensor& x, const Tensor& w, const Tensor* bias, void* workspace,
              std::size_t workspace_bytes, cudaStream_t stream);

  std::size_t cached_plans() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mi::kernels
