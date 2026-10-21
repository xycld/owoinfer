// Reader for NumPy .npy files (format versions 1-3), used by golden-data tests and tools.
// Supports little-endian, C-order arrays of fixed-size numeric dtypes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace mi::npy {

struct Array {
  std::string dtype;            // numpy descr without byte order, e.g. "f4", "i8", "u2"
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;  // raw element bytes, C order

  std::int64_t numel() const;
  std::size_t itemsize() const;

  // Copies the elements out as T (memcpy, so the byte buffer's alignment does not matter). Throws
  // std::runtime_error unless dtype is exactly T's kind and size, e.g. "f4" for float.
  template <typename T>
  std::vector<T> to_vector() const {
    static_assert(std::is_arithmetic_v<T>);
    constexpr char kind = std::is_same_v<T, bool> ? 'b' : std::is_floating_point_v<T> ? 'f' : std::is_signed_v<T> ? 'i' : 'u';
    if (dtype != kind + std::to_string(sizeof(T))) {
      throw std::runtime_error("npy: array of dtype " + dtype + " read as " + kind + std::to_string(sizeof(T)));
    }
    std::vector<T> out(data.size() / sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), data.data(), out.size() * sizeof(T));
    return out;
  }
};

// Throws std::runtime_error on unsupported or malformed files.
Array load(const std::filesystem::path& path);

}  // namespace mi::npy
