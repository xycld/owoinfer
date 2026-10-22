// Grouped 4-/6-bit weight storage (comfy-kitchen asym_w4a8_int8 / w6a8_int8, the community W4A8 / W6A8 GPTQ files)
// decoded on the GPU to the INT8 ConvRot grid the INT8 GEMMs consume, so a streamed linear moves its packed bytes
// (4.5 or 6.5 bits per weight with g16 scales) and is expanded on the device just before its GEMM.
//
// Bit-exact to the CPU decoder mi::quant::decode_grouped_rows (comfy-kitchen eager _dequant_int4_grouped_to_int8):
//   level = codebook[code] (16 entries, 4-bit only) or code - 2^(bits-1);
//   out   = clamp(nearbyint(level * s_rel[r, c / group]), -127, 127), the product one fp32 multiply;
// with s_rel e4m3fn. Layout per row of k * bits / 8 bytes: a k/2-byte nibble plane (even column = low nibble), then
// for 6 bits a k/4-byte plane of each code's top 2 bits (column c: byte c / 4, bits 2 * (c % 4)). The per-row scale
// (s_channel, F32 [n]) is the GEMM's w_scale as stored; it needs no decoding.
//
// Untrusted data: a NaN or negative group scale (the CPU decoder's error) decodes that group to zeros and sets
// *invalid to a non-zero value; the caller checks it after the work completes. A codebook must be finite (validated
// when the file is opened: it is 64 bytes per linear).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace mi::kernels::quant {

// Bump whenever the decoded INT8 grid could change for some input (embedding caches key on it through the encoder's
// numerics id).
inline constexpr std::uint32_t kGroupedDecodeVersion = 1;

struct GroupedCodes {
  const std::uint8_t* packed = nullptr;  // [n, k * bits / 8], rows contiguous
  const std::uint8_t* s_rel = nullptr;   // e4m3fn [n, k / group], rows contiguous
  const float* codebook = nullptr;       // [16] or nullptr (uniform levels)
  std::int64_t n = 0, k = 0;
  int bits = 4;   // 4 or 6
  int group = 16; // >= 4, dividing 16 or a multiple of 16, dividing k
};

// out: int8 [n, k] contiguous, 16-byte aligned. invalid: device word, only ever OR-ed (never cleared here).
// Needs k % 16 == 0 (6 bits: k % 32 == 0), an 8-byte aligned packed, no codebook with 6 bits. No allocation, no host
// synchronization. Invalid arguments throw std::invalid_argument.
void decode_grouped_int8(const GroupedCodes& w, std::int8_t* out, unsigned* invalid, cudaStream_t stream);

}  // namespace mi::kernels::quant
