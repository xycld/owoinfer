// Host conversions between fp32 and the engine's 16-bit types.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace mi::vk {

inline float half_to_float(std::uint16_t h) {
  const std::uint32_t exp = (h >> 10) & 0x1f;
  const std::uint32_t mant = h & 0x3ff;
  float f = 0.0f;
  if (exp == 0) {
    f = std::ldexp(static_cast<float>(mant), -24);  // zero or subnormal: mant * 2^-24, exact in fp32
  } else {
    const std::uint32_t bits = exp == 31 ? 0x7f800000u | (mant << 13) : ((exp + 127 - 15) << 23) | (mant << 13);
    std::memcpy(&f, &bits, 4);
  }
  return (h & 0x8000u) != 0 ? -f : f;
}

// Rounds to nearest, ties to even, like the hardware conversion.
inline std::uint16_t float_to_half(float f) {
  std::uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t magnitude = x & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7e00u);  // NaN
  const std::int32_t exp = static_cast<std::int32_t>(magnitude >> 23) - 127 + 15;
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);  // overflow and infinity
  if (exp < -10) return sign;                                        // below half of the smallest subnormal
  const std::uint32_t mant = magnitude & 0x7fffffu;
  const std::uint32_t significand = exp > 0 ? mant : mant | 0x800000u;
  const int shift = exp > 0 ? 13 : 14 - exp;
  std::uint32_t rounded = significand >> shift;
  const std::uint32_t rest = significand & ((1u << shift) - 1u);
  const std::uint32_t half = 1u << (shift - 1);
  if (rest > half || (rest == half && (rounded & 1u) != 0)) ++rounded;
  // An addition, not an OR: rounding can carry out of the mantissa into the exponent.
  return static_cast<std::uint16_t>(sign | ((exp > 0 ? static_cast<std::uint32_t>(exp) << 10 : 0u) + rounded));
}

inline float bf16_to_float(std::uint16_t b) {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Rounds to nearest, ties to even.
inline std::uint16_t float_to_bf16(float f) {
  std::uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>(x >> 16 | 0x7fc0u);  // NaN
  return static_cast<std::uint16_t>((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

}  // namespace mi::vk
