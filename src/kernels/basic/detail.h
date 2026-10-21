// Internal helpers shared by the kernels/basic translation units: argument checks, a row-offset map for
// "rows of a contiguous last dim" views with arbitrary leading strides, and exact bf16 device helpers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/tensor.h"

namespace mi::kernels::detail {

// Maps a flat row index over a tensor's leading dims to an element offset. Contiguous leading dims are
// merged on the host so the common case is a single multiply.
struct RowMap {
  int nd = 0;
  std::int64_t size[Tensor::kMaxRank]{};
  std::int64_t stride[Tensor::kMaxRank]{};

  __host__ __device__ std::int64_t offset(std::int64_t row) const {
    std::int64_t off = 0;
    for (int i = nd - 1; i >= 0; --i) {
      off += (row % size[i]) * stride[i];
      row /= size[i];
    }
    return off;
  }
};

// Row map over all dims of `t` except the innermost `inner` ones.
inline RowMap row_map(const Tensor& t, int inner = 1) {
  RowMap m;
  for (int i = 0; i < t.rank - inner; ++i) {
    const auto k = static_cast<std::size_t>(i);
    if (t.shape[k] == 1) continue;
    if (m.nd > 0 && m.stride[m.nd - 1] == t.stride[k] * t.shape[k]) {
      m.size[m.nd - 1] *= t.shape[k];
      m.stride[m.nd - 1] = t.stride[k];
    } else {
      m.size[m.nd] = t.shape[k];
      m.stride[m.nd] = t.stride[k];
      ++m.nd;
    }
  }
  return m;
}

inline std::int64_t rows_of(const Tensor& t, int inner = 1) {
  std::int64_t n = 1;
  for (int i = 0; i < t.rank - inner; ++i) n *= t.shape[static_cast<std::size_t>(i)];
  return n;
}

[[noreturn]] inline void fail(const std::string& op, const std::string& msg) {
  throw std::invalid_argument("mi::kernels::" + op + ": " + msg);
}

// A CUDA tensor of `dtype` whose last dim has stride 1 (and, if `last` >= 0, size `last`).
inline void check_rows(const char* op, const char* name, const Tensor& t, DType dtype, std::int64_t last = -1) {
  if (t.device != Device::kCUDA || t.dtype != dtype || t.rank < 1 || t.data == nullptr) {
    fail(op, std::string(name) + " must be a CUDA " + dtype_name(dtype) + " tensor, got " + t.describe());
  }
  if (t.stride[static_cast<std::size_t>(t.rank - 1)] != 1 && t.dim(-1) != 1) {
    fail(op, std::string(name) + " needs a contiguous last dim, got " + t.describe());
  }
  if (last >= 0 && t.dim(-1) != last) {
    fail(op, std::string(name) + " last dim must be " + std::to_string(last) + ", got " + t.describe());
  }
}

// A rank-1 (or [1, ..., D]) contiguous CUDA bf16 vector of length d, broadcast over rows.
inline void check_vector(const char* op, const char* name, const Tensor& t, std::int64_t d) {
  check_rows(op, name, t, DType::kBF16, d);
  if (t.numel() != d) fail(op, std::string(name) + " must hold exactly " + std::to_string(d) + " values, got " + t.describe());
}

inline void check_same_shape(const char* op, const Tensor& a, const Tensor& b) {
  bool same = a.rank == b.rank;
  for (int i = 0; same && i < a.rank; ++i) same = a.shape[static_cast<std::size_t>(i)] == b.shape[static_cast<std::size_t>(i)];
  if (!same) fail(op, "shape mismatch: " + a.describe() + " vs " + b.describe());
}

inline unsigned grid_for_rows(std::int64_t rows) {
  return static_cast<unsigned>(rows < 1 ? 1 : rows > 1048576 ? 1048576 : rows);
}

#ifdef __CUDACC__
// Every arithmetic step below is one torch eager op: an fp32 operation rounded once. The explicit _rn
// intrinsics stop nvcc from contracting a multiply and an add into an FMA, which torch never does across
// separate kernels.
__device__ __forceinline__ float f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 bf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }  // round to bf16
__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
// torch: x / (1 + exp(-x)) (silu) and 1 / (1 + exp(-x)) (sigmoid) in fp32 with IEEE division.
__device__ __forceinline__ float silu_f32(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); }
__device__ __forceinline__ float sigmoid_f32(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); }
#endif

}  // namespace mi::kernels::detail
