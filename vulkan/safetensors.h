// A minimal safetensors reader: the JSON header's tensor entries (dtype, shape, byte range) and raw reads of one
// tensor's bytes. Links nothing beyond the standard library. Failures throw std::runtime_error.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace mi::vk::st {

struct TensorInfo {
  std::string name;
  std::string dtype;  // as stored: "F32", "BF16", "F16", "I64", ...
  std::vector<std::int64_t> shape;
  std::uint64_t offset = 0;  // from the start of the file
  std::uint64_t bytes = 0;
};

class File {
 public:
  explicit File(const std::string& path);

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  // nullptr when the file has no tensor of that name.
  const TensorInfo* find(const std::string& name) const;
  // Reads the tensor's bytes into dst, which must hold info.bytes.
  void read(const TensorInfo& info, void* dst);
  // A tensor of dtype F32 or BF16 as fp32 values; throws for other dtypes.
  std::vector<float> read_f32(const std::string& name);

 private:
  std::string path_;
  std::ifstream in_;
  std::uint64_t file_bytes_ = 0;
  std::vector<TensorInfo> tensors_;
};

}  // namespace mi::vk::st
