// A minimal safetensors reader: the header of one file, tensor bytes read with pread, and the shard map of a
// model.safetensors.index.json. Failures throw std::runtime_error.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mi::vk::st {

struct TensorInfo {
  std::string name;
  std::string dtype;  // as stored, e.g. "BF16", "F32"
  std::vector<std::int64_t> shape;
  std::uint64_t offset = 0;  // of the first byte, from the start of the file
  std::uint64_t bytes = 0;
};

class File {
 public:
  explicit File(const std::string& path);
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  const std::string& path() const { return path_; }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find(const std::string& name) const;  // null if absent
  const TensorInfo& at(const std::string& name) const;    // throws if absent

  // `bytes` bytes of tensor `t` from byte `rel` of it into `dst`.
  void read(const TensorInfo& t, void* dst, std::uint64_t rel, std::uint64_t bytes) const;
  void read(const TensorInfo& t, void* dst) const { read(t, dst, 0, t.bytes); }

 private:
  std::string path_;
  int fd_ = -1;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::size_t> by_name_;
};

// model.safetensors.index.json's weight_map: tensor name -> shard file name (relative to the index's directory).
std::map<std::string, std::string> read_index(const std::string& path);

}  // namespace mi::vk::st
