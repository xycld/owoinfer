// decode_grouped_int8 (see w4a8_decode.h). One thread per 16 consecutive columns of a row: one 8-byte load of the
// nibble plane (plus 4 bytes of the 2-bit plane for 6-bit codes), one 16-byte store. Memory-bound.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"
#include "kernels/quant/w4a8_decode.h"

namespace mi::kernels::quant {
namespace {

using kernels::detail::fail;

// OCP e4m3fn, as mi::quant::e4m3_to_float: subnormals m * 2^-9, normals (1 + m/8) * 2^(e-7), 0x7F / 0xFF NaN.
__device__ __forceinline__ float e4m3(std::uint8_t v) {
  const unsigned e = (v >> 3) & 0xFu, m = v & 7u;
  float mag;
  if (e == 0xFu && m == 7u) mag = __int_as_float(0x7FC00000);
  else if (e == 0) mag = __fmul_rn(static_cast<float>(m), 0x1p-9f);
  else mag = __int_as_float(static_cast<int>(((e + 120u) << 23) | (m << 20)));
  return (v & 0x80u) ? -mag : mag;
}

template <int kBits>
__global__ void decode_kernel(GroupedCodes w, std::int8_t* __restrict__ out, unsigned* __restrict__ invalid) {
  __shared__ float cb[16];
  const bool has_cb = w.codebook != nullptr;
  if (threadIdx.x < 16) cb[threadIdx.x] = has_cb ? w.codebook[threadIdx.x] : 0.0f;
  __syncthreads();
  const std::int64_t per_row = w.k / 16;
  const std::int64_t total = w.n * per_row;
  const std::int64_t row_bytes = w.k * kBits / 8, groups = w.k / w.group;
  const float zero = static_cast<float>(1 << (kBits - 1));
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t r = i / per_row, c0 = (i - r * per_row) * 16;
    const std::uint8_t* row = w.packed + r * row_bytes;
    const unsigned long long lo = *reinterpret_cast<const unsigned long long*>(row + c0 / 2);
    unsigned hi = 0;
    if constexpr (kBits == 6) hi = *reinterpret_cast<const unsigned*>(row + w.k / 2 + c0 / 4);
    const std::uint8_t* sr = w.s_rel + r * groups;
    alignas(16) std::int8_t q[16];
    bool bad = false;
    float s = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      // One scale load per group; group >= 16 means one per thread.
      if (j == 0 || (w.group < 16 && j % w.group == 0)) s = e4m3(sr[(c0 + j) / w.group]);
      unsigned code = static_cast<unsigned>(lo >> (4 * j)) & 0xFu;
      if constexpr (kBits == 6) code |= ((hi >> (2 * j)) & 0x3u) << 4;
      const float level = has_cb ? cb[code] : static_cast<float>(code) - zero;
      float v = rintf(__fmul_rn(level, s));
      v = fminf(fmaxf(v, -127.0f), 127.0f);
      const bool ok = !(isnan(s) || s < 0.0f);
      bad |= !ok;
      q[j] = ok ? static_cast<std::int8_t>(v) : std::int8_t{0};
    }
    *reinterpret_cast<int4*>(out + r * w.k + c0) = *reinterpret_cast<const int4*>(q);
    if (bad) atomicOr(invalid, 1u);
  }
}

bool aligned(const void* p, std::size_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; }

}  // namespace

void decode_grouped_int8(const GroupedCodes& w, std::int8_t* out, unsigned* invalid, cudaStream_t stream) {
  constexpr const char* op = "quant::decode_grouped_int8";
  if (w.bits != 4 && w.bits != 6) fail(op, "bits must be 4 or 6, got " + std::to_string(w.bits));
  if (w.n < 0 || w.k <= 0 || w.k % (w.bits == 6 ? 32 : 16) != 0) {
    fail(op, "bad shape n=" + std::to_string(w.n) + " k=" + std::to_string(w.k));
  }
  if (w.group < 4 || (16 % w.group != 0 && w.group % 16 != 0) || w.k % w.group != 0) {
    fail(op, "group " + std::to_string(w.group) + " must be >= 4, divide 16 or be a multiple of 16, and divide k");
  }
  if (w.bits == 6 && w.codebook != nullptr) fail(op, "6-bit codes have no codebook");
  if (w.packed == nullptr || w.s_rel == nullptr || out == nullptr || invalid == nullptr || !aligned(w.packed, 8) ||
      !aligned(out, 16) || !aligned(w.codebook, 4)) {
    fail(op, "packed (8-byte aligned), s_rel, out (16-byte aligned) and invalid must be device pointers");
  }
  if (w.n == 0) return;
  constexpr int kThreads = 256;
  const std::int64_t work = w.n * (w.k / 16);
  const std::int64_t blocks = std::min<std::int64_t>((work + kThreads - 1) / kThreads, std::int64_t{1} << 20);
  if (w.bits == 4) {
    decode_kernel<4><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(w, out, invalid);
  } else {
    decode_kernel<6><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(w, out, invalid);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::quant
