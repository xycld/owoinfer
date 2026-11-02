// Exact bf16 device arithmetic shared by every kernel of the prototype, on both platforms. As kernels/basic/detail.h,
// every step is one torch eager op: an fp32 operation rounded once. bf16 travels as its bits (std::uint16_t) and the
// conversions are plain integer code, so the rounding the nvcc verify build checks is the rounding the AMD builds run.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

// nvcc: the _rn intrinsics stop FMA contraction. clang has no such intrinsics: contraction is off from here on instead
// (softmax's one contracted step is written as the explicit fmaf it compiles to under nvcc).
#if defined(__HIP_PLATFORM_AMD__)
#pragma clang fp contract(off)
#endif

// Full unrolling of the fragment loops: arrays of fragments must become registers, not scratch memory.
#if defined(__HIP_PLATFORM_AMD__)
#define HIP_BLOCK_UNROLL _Pragma("clang loop unroll(full)")
#else
#define HIP_BLOCK_UNROLL _Pragma("unroll")
#endif

namespace hip_block::detail {

__device__ __forceinline__ float bits_to_float(std::uint32_t u) {
  union {
    std::uint32_t u;
    float v;
  } c{u};
  return c.v;
}
__device__ __forceinline__ std::uint32_t float_to_bits(float v) {
  union {
    float v;
    std::uint32_t u;
  } c{v};
  return c.u;
}

// bf16 bits to fp32: exact, bf16 is the top half of fp32.
__device__ __forceinline__ float f(std::uint16_t b) { return bits_to_float(static_cast<std::uint32_t>(b) << 16); }

// fp32 to bf16 bits, round to nearest even. A NaN stays a NaN (the rounding carry could otherwise wrap it to -0).
__device__ __forceinline__ std::uint16_t bf(float v) {
  const std::uint32_t u = float_to_bits(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float rbf(float v) { return f(bf(v)); }  // round to bf16

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float mul(float a, float b) { return a * b; }
__device__ __forceinline__ float add(float a, float b) { return a + b; }
__device__ __forceinline__ float div(float a, float b) { return a / b; }
__device__ __forceinline__ float rsqrt(float x) { return __ocml_rsqrt_f32(x); }
#else
__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float rsqrt(float x) { return rsqrtf(x); }
#endif

// torch: 1 / (1 + exp(-x)) in fp32 with IEEE division.
__device__ __forceinline__ float sigmoid_f32(float x) { return div(1.0f, add(1.0f, expf(-x))); }

}  // namespace hip_block::detail
