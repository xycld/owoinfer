// Argument validation shared by the GDN entry points (host only).
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/tensor.h"

namespace mi::gdn::detail {

inline void require(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("gdn: " + what);
}

inline void check_cu_seqlens(const Tensor& cu) {
  require(cu.dtype == DType::kI32 && cu.device == Device::kCUDA && cu.rank == 1 && cu.dim(0) >= 2 && cu.is_contiguous(),
          "cu_seqlens must be a contiguous cuda i32 [N+1], got " + cu.describe());
}

// 2-D cuda tensor [rows, cols] with unit column stride (row stride free).
inline void check_rows(const Tensor& t, DType dtype, std::int64_t rows, std::int64_t cols, const char* name) {
  require(t.dtype == dtype && t.device == Device::kCUDA && t.rank == 2 && t.dim(0) == rows && t.dim(1) == cols &&
              t.stride[1] == 1,
          std::string(name) + " has unexpected layout " + t.describe());
}

// Vectorised kernels load 8 or 16 bytes at a time.
inline void check_aligned(const Tensor& t, std::size_t bytes, const char* name) {
  require(reinterpret_cast<std::uintptr_t>(t.data) % bytes == 0,
          std::string(name) + " must be " + std::to_string(bytes) + "-byte aligned");
}

}  // namespace mi::gdn::detail
