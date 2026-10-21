// Overflow-checked size arithmetic for sizes read from untrusted files (safetensors headers,
// .mossidx, .npy).
#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace mi::util {

// a * b, or nullopt if the product does not fit in uint64.
inline std::optional<std::uint64_t> checked_mul(std::uint64_t a, std::uint64_t b) {
  if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) return std::nullopt;
  return a * b;
}

// Element count of `shape`, or nullopt if a dimension is negative or a running product exceeds
// INT64_MAX. Checking every step (not just the result) keeps a later plain left-to-right int64
// product of the same dims, as the numel() helpers compute, free of signed overflow.
inline std::optional<std::int64_t> checked_numel(std::span<const std::int64_t> shape) {
  std::int64_t n = 1;
  for (std::int64_t d : shape) {
    if (d < 0 || (d != 0 && n > std::numeric_limits<std::int64_t>::max() / d)) return std::nullopt;
    n *= d;
  }
  return n;
}

}  // namespace mi::util
