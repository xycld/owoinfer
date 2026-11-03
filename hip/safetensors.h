// A minimal safetensors reader: the header of one file, tensor bytes read with pread, and the shard map of a
// model.safetensors.index.json. Failures throw std::runtime_error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mi::hip::st {

struct TensorInfo {
  std::string name;
  std::string dtype;  // as stored, e.g. "BF16", "F32"
  std::vector<std::int64_t> shape;
  std::uint64_t offset = 0;  // of the first byte, from the start of the file
  std::uint64_t bytes = 0;
};

class File {
 public:
  explicit File(const std::filesystem::path& path);  // opened by its wide name on Windows
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  const std::string& path() const { return path_; }  // UTF-8, for messages
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find(const std::string& name) const;  // null if absent
  const TensorInfo& at(const std::string& name) const;    // throws if absent

  // `bytes` bytes of tensor `t` from byte `rel` of it into `dst`.
  void read(const TensorInfo& t, void* dst, std::uint64_t rel, std::uint64_t bytes) const;
  void read(const TensorInfo& t, void* dst) const { read(t, dst, 0, t.bytes); }
  // A whole F32 or BF16 tensor as fp32 values (bf16 widened exactly), or throws.
  std::vector<float> read_f32(const std::string& name) const;

 private:
  std::string path_;
  int fd_ = -1;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t> by_name_;
};

// model.safetensors.index.json's weight_map: tensor name -> shard file name (relative to the index's directory).
std::map<std::string, std::string> read_index(const std::filesystem::path& path);

// The tensors of one file, or of every shard of a model directory, by name.
class Index {
 public:
  // A .safetensors file, or a directory holding one *.safetensors.index.json (its shards) or exactly one .safetensors.
  explicit Index(const std::filesystem::path& path);

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorInfo& at(const std::string& name) const;  // throws if absent
  // Reads a whole tensor into caller memory of exactly its size, or throws.
  void read_into(const std::string& name, void* out, std::uint64_t bytes) const;

 private:
  std::vector<std::unique_ptr<File>> files_;
  std::map<std::string, std::pair<const File*, const TensorInfo*>> tensors_;
};

}  // namespace mi::hip::st
