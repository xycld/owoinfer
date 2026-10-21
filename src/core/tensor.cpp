#include "core/tensor.h"

namespace mi {

const char* dtype_name(DType t) {
  switch (t) {
    case DType::kF32: return "f32";
    case DType::kF16: return "f16";
    case DType::kBF16: return "bf16";
    case DType::kF8E4M3: return "f8e4m3";
    case DType::kI8: return "i8";
    case DType::kU8: return "u8";
    case DType::kI32: return "i32";
    case DType::kI64: return "i64";
  }
  return "?";
}

std::int64_t Tensor::numel() const {
  std::int64_t n = 1;
  for (int i = 0; i < rank; ++i) n *= shape[static_cast<std::size_t>(i)];
  return n;
}

bool Tensor::is_contiguous() const {
  std::int64_t expect = 1;
  for (int i = rank - 1; i >= 0; --i) {
    const auto k = static_cast<std::size_t>(i);
    if (shape[k] != 1 && stride[k] != expect) return false;
    expect *= shape[k];
  }
  return true;
}

Tensor Tensor::view(void* data, DType dtype, Device device, const std::int64_t* shape, int rank) {
  if (rank < 0 || rank > kMaxRank) throw std::invalid_argument("tensor rank out of range");
  Tensor t;
  t.data = data;
  t.dtype = dtype;
  t.device = device;
  t.rank = rank;
  std::int64_t s = 1;
  for (int i = rank - 1; i >= 0; --i) {
    const auto k = static_cast<std::size_t>(i);
    if (shape[i] < 0) throw std::invalid_argument("negative tensor dimension");
    t.shape[k] = shape[i];
    t.stride[k] = s;
    s *= shape[i];
  }
  return t;
}

Tensor Tensor::view(void* data, DType dtype, Device device, std::initializer_list<std::int64_t> shape) {
  return view(data, dtype, device, shape.begin(), static_cast<int>(shape.size()));
}

Tensor Tensor::reshaped(std::initializer_list<std::int64_t> new_shape) const {
  if (!is_contiguous()) throw std::invalid_argument("reshape of non-contiguous tensor " + describe());
  Tensor t = view(data, dtype, device, new_shape);
  if (t.numel() != numel()) throw std::invalid_argument("reshape changes numel of " + describe());
  return t;
}

Tensor Tensor::slice0(std::int64_t begin, std::int64_t end) const {
  if (rank < 1 || begin < 0 || end < begin || end > shape[0]) {
    throw std::invalid_argument("slice0 out of range on " + describe());
  }
  Tensor t = *this;
  t.data = static_cast<char*>(data) + static_cast<std::size_t>(begin * stride[0]) * dtype_size(dtype);
  t.shape[0] = end - begin;
  return t;
}

std::string Tensor::describe() const {
  std::string s = dtype_name(dtype);
  s += '[';
  for (int i = 0; i < rank; ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[static_cast<std::size_t>(i)]);
  }
  s += device == Device::kCUDA ? "] cuda" : "] cpu";
  return s;
}

}  // namespace mi
