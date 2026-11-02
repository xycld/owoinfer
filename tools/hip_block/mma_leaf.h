// The swappable leaf of every GEMM and of attention: a 16x16x16 bf16 multiply with fp32 accumulators, per wave.
//
//   HIP_BLOCK_LEAF_ROCWMMA (AMD gfx11xx, gfx12xx)  rocWMMA fragments, v_wmma_f32_16x16x16_bf16
//   otherwise (NVIDIA verify build)                scalar: the exact bf16*bf16 product in fp32, then add, ascending K
//
// rocWMMA does not compile under nvcc, and its fragments' lane layouts are private and differ between RDNA3 and RDNA4.
// So nothing outside this file touches a fragment's elements: an accumulator's values reach the kernels only through
// store_acc, into memory.
//
// Every call is collective: all 32 threads of the wave make it, with the same (thread-independent) arguments and under
// the same control flow. p may point to global or __shared__ memory. A caller reading what store_acc wrote to
// __shared__ memory from another thread must __syncthreads() first.
// The kernels depend on lockstep: they reuse a __shared__ buffer (the zero-padded fragments, s_mem, o_mem) right after a
// collective read of it, with a barrier before the next write's readers but none between the read and that write. That
// is safe only because the 32 threads are one wave, or one warp that has not diverged at that point. A workgroup of
// several waves or warps would need a barrier after every such read.
//   load_a       A(m, k) = p[m * ld + k]
//   load_b       B(k, n) = p[n * ld + k]   (a [N, K] row-major weight, or K's tokens, one column per row of memory)
//   load_b_rows  B(k, n) = p[k * ld + n]   (V's tokens: K along the rows of memory)
//   mma          c += a * b
//   store_acc    p[m * ld + n] = c(m, n)
//   load_acc     c(m, n) = p[m * ld + n]
#pragma once

#include <cstdint>

#include "device_math.h"

#if defined(HIP_BLOCK_LEAF_ROCWMMA)

#include <rocwmma/rocwmma.hpp>

namespace hip_block::leaf {

inline constexpr const char* kName = "rocWMMA v_wmma_f32_16x16x16_bf16";

using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::row_major>;
using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, rocwmma::bfloat16_t, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void clear(FragAcc& c) { rocwmma::fill_fragment(c, 0.0f); }
__device__ __forceinline__ void load_a(FragA& a, const std::uint16_t* p, int ld) {
  rocwmma::load_matrix_sync(a, reinterpret_cast<const rocwmma::bfloat16_t*>(p), static_cast<std::uint32_t>(ld));
}
__device__ __forceinline__ void load_b(FragB& b, const std::uint16_t* p, int ld) {
  rocwmma::load_matrix_sync(b, reinterpret_cast<const rocwmma::bfloat16_t*>(p), static_cast<std::uint32_t>(ld));
}
__device__ __forceinline__ void load_b_rows(FragB& b, const std::uint16_t* p, int ld) {
  rocwmma::load_matrix_sync(b, reinterpret_cast<const rocwmma::bfloat16_t*>(p), static_cast<std::uint32_t>(ld),
                            rocwmma::mem_row_major);
}
__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) { rocwmma::mma_sync(c, a, b, c); }
__device__ __forceinline__ void store_acc(float* p, const FragAcc& c, int ld) {
  rocwmma::store_matrix_sync(p, c, static_cast<std::uint32_t>(ld), rocwmma::mem_row_major);
}
__device__ __forceinline__ void load_acc(FragAcc& c, const float* p, int ld) {
  rocwmma::load_matrix_sync(c, p, static_cast<std::uint32_t>(ld), rocwmma::mem_row_major);
}

}  // namespace hip_block::leaf

#else

namespace hip_block::leaf {

inline constexpr const char* kName = "scalar (logic check, not WMMA)";

// Thread t of the wave owns row t / 2 and columns (t & 1) * 8 .. + 8 of the accumulator: the elements of each row are
// computed from that row of A and those columns of B, copied into its registers as a fragment load would.
__device__ __forceinline__ int row_of_thread() { return static_cast<int>(threadIdx.x) / 2; }
__device__ __forceinline__ int col_of_thread() { return (static_cast<int>(threadIdx.x) & 1) * 8; }

struct FragA {
  float v[16];
};
struct FragB {
  float v[8][16];  // [owned column][k]
};
struct FragAcc {
  float v[8];
};

__device__ __forceinline__ void clear(FragAcc& c) {
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) c.v[i] = 0.0f;
}
__device__ __forceinline__ void load_a(FragA& a, const std::uint16_t* p, int ld) {
  const std::uint16_t* row = p + row_of_thread() * ld;
  HIP_BLOCK_UNROLL
  for (int k = 0; k < 16; ++k) a.v[k] = detail::f(row[k]);
}
__device__ __forceinline__ void load_b(FragB& b, const std::uint16_t* p, int ld) {
  const int c0 = col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) {
    const std::uint16_t* col = p + (c0 + i) * ld;
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 16; ++k) b.v[i][k] = detail::f(col[k]);
  }
}
__device__ __forceinline__ void load_b_rows(FragB& b, const std::uint16_t* p, int ld) {
  const int c0 = col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) {
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 16; ++k) b.v[i][k] = detail::f(p[k * ld + c0 + i]);
  }
}
__device__ __forceinline__ void mma(FragAcc& c, const FragA& a, const FragB& b) {
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) {
    float acc = c.v[i];
    HIP_BLOCK_UNROLL
    for (int k = 0; k < 16; ++k) acc = detail::add(acc, detail::mul(a.v[k], b.v[i][k]));
    c.v[i] = acc;
  }
}
__device__ __forceinline__ void store_acc(float* p, const FragAcc& c, int ld) {
  float* row = p + row_of_thread() * ld + col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) row[i] = c.v[i];
}
__device__ __forceinline__ void load_acc(FragAcc& c, const float* p, int ld) {
  const float* row = p + row_of_thread() * ld + col_of_thread();
  HIP_BLOCK_UNROLL
  for (int i = 0; i < 8; ++i) c.v[i] = row[i];
}

}  // namespace hip_block::leaf

#endif
