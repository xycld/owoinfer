// Minimal safetensors reader for the Vulkan backend: a file's header, and one tensor's bytes on demand. A sharded model is
// a set of files named by its model.safetensors.index.json (or diffusion_pytorch_model.safetensors.index.json).
// Reads with std::ifstream, so it is the same on every platform. Failures throw std::runtime_error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mi::vk::st {

struct TensorInfo {
  std::string dtype;  // as the header spells it: "BF16", "F32", "F16", "I64", ...
  std::vector<std::int64_t> shape;
  std::uint64_t offset = 0;  // from the start of the file
  std::uint64_t bytes = 0;
  std::filesystem::path file;
};

// The tensors of one file, or of every shard of a model directory.
class Index {
 public:
  // A .safetensors file, or a directory holding one *.safetensors.index.json (its shards) or exactly one .safetensors.
  explicit Index(const std::filesystem::path& path);

  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  // Throws if the tensor is missing.
  const TensorInfo& at(const std::string& name) const;

  // Reads a tensor's bytes, or throws.
  std::vector<std::byte> read(const std::string& name) const;
  // Reads into caller memory of exactly info.bytes.
  void read_into(const std::string& name, void* out, std::uint64_t bytes) const;

 private:
  void add_file(const std::filesystem::path& file);
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace mi::vk::st
