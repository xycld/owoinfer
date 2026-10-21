// Non-owning tensor views shared by runtime, kernels and models.
// Memory is owned elsewhere (device arena, pinned ring, mmap, DeviceBuffer); a Tensor never frees.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace mi {

enum class DType : std::uint8_t { kF32, kF16, kBF16, kF8E4M3, kI8, kU8, kI32, kI64 };
enum class Device : std::uint8_t { kCPU, kCUDA };

constexpr std::size_t dtype_size(DType t) {
  switch (t) {
    case DType::kF32: case DType::kI32: return 4;
    case DType::kF16: case DType::kBF16: return 2;
    case DType::kF8E4M3: case DType::kI8: case DType::kU8: return 1;
    case DType::kI64: return 8;
  }
  return 0;
}

const char* dtype_name(DType t);

struct Tensor {
  static constexpr int kMaxRank = 8;

  void* data = nullptr;
  DType dtype = DType::kF32;
  Device device = Device::kCPU;
  int rank = 0;
  std::array<std::int64_t, kMaxRank> shape{};
  std::array<std::int64_t, kMaxRank> stride{};  // in elements

  std::int64_t dim(int i) const { return shape[static_cast<std::size_t>(i < 0 ? i + rank : i)]; }
  std::int64_t numel() const;
  std::size_t nbytes() const { return static_cast<std::size_t>(numel()) * dtype_size(dtype); }
  bool is_contiguous() const;

  template <typename T>
  T* ptr() const { return static_cast<T*>(data); }

  // Contiguous view; throws std::invalid_argument on rank > kMaxRank or negative dims.
  static Tensor view(void* data, DType dtype, Device device, std::initializer_list<std::int64_t> shape);
  static Tensor view(void* data, DType dtype, Device device, const std::int64_t* shape, int rank);

  // Same storage, new contiguous shape with equal numel (requires is_contiguous()).
  Tensor reshaped(std::initializer_list<std::int64_t> new_shape) const;
  // Rows [begin, end) of dimension 0, keeping strides (no copy).
  Tensor slice0(std::int64_t begin, std::int64_t end) const;

  std::string describe() const;  // e.g. "bf16[4096, 3072] cuda"
};

}  // namespace mi
