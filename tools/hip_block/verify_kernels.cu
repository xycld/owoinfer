// NVIDIA verify build only: device_math.h's bf16 conversions against the CUDA intrinsics the engine uses, on every input.
#include <cuda_bf16.h>

#include "device_math.h"
#include "verify_kernels.h"

namespace hip_block {
namespace {

// Thread t of block b covers the 256 fp32 patterns (b << 16) | (t << 8) | i.
__global__ void round_check_kernel(unsigned long long* counts) {
  for (unsigned i = 0; i < 256; ++i) {
    const std::uint32_t u = (static_cast<std::uint32_t>(blockIdx.x) << 16) | (static_cast<std::uint32_t>(threadIdx.x) << 8) | i;
    const float v = detail::bits_to_float(u);
    const std::uint16_t got = detail::bf(v), ref = __bfloat16_as_ushort(__float2bfloat16_rn(v));
    if (got == ref) continue;
    const bool got_nan = (got & 0x7fffu) > 0x7f80u, ref_nan = (ref & 0x7fffu) > 0x7f80u;
    atomicAdd(&counts[got_nan && ref_nan ? 1 : 0], 1ull);
  }
}

__global__ void widen_check_kernel(unsigned long long* counts) {
  const auto b = static_cast<std::uint16_t>((blockIdx.x << 8) | threadIdx.x);
  const std::uint32_t got = detail::float_to_bits(detail::f(b));
  const std::uint32_t ref = detail::float_to_bits(__bfloat162float(__ushort_as_bfloat16(b)));
  if (got == ref) return;
  const bool nan = ((got & 0x7fffffffu) > 0x7f800000u) && ((ref & 0x7fffffffu) > 0x7f800000u);
  atomicAdd(&counts[nan ? 1 : 0], 1ull);
}

}  // namespace

ConversionCheck check_conversions() {
  unsigned long long* counts = nullptr;
  if (cudaMalloc(&counts, 4 * sizeof(unsigned long long)) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
  cudaMemset(counts, 0, 4 * sizeof(unsigned long long));
  round_check_kernel<<<65536, 256>>>(counts);
  widen_check_kernel<<<256, 256>>>(counts + 2);
  cudaDeviceSynchronize();
  unsigned long long h[4];
  cudaMemcpy(h, counts, sizeof(h), cudaMemcpyDeviceToHost);
  cudaFree(counts);
  return {h[0], h[1], h[2], h[3]};
}

}  // namespace hip_block
