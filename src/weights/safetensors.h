// Strict reader for the safetensors container: u64 little-endian header length, a JSON header, then
// a densely packed data section. Only the header is read; tensor bytes are fetched on demand through
// weights::FileReader / MappedFile using the absolute offsets recorded here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "util/path.h"

namespace mi::weights {

// Header limits. Real MOSS-Image headers are under 40 KB with 3-5 dims per tensor and at most a few
// thousand tensors per file; the caps bound what a hostile header can make the parser allocate.
inline constexpr std::uint64_t kMaxHeaderBytes = std::uint64_t{16} << 20;
inline constexpr std::size_t kMaxRank = 8;
inline constexpr std::size_t kMaxTensors = 65536;

// Thrown for any structural violation; what() names the file and, when relevant, the tensor.
struct FormatError : std::runtime_error {
  FormatError(const std::filesystem::path& file, const std::string& tensor, const std::string& msg)
      : std::runtime_error(util::path_utf8(file) + (tensor.empty() ? "" : " [tensor '" + tensor + "']") + ": " + msg) {}
};

enum class DType {
  kBool, kU8, kI8, kF8E4M3, kF8E5M2, kF8E8M0,
  kI16, kU16, kF16, kBF16,
  kI32, kU32, kF32,
  kI64, kU64, kF64,
};

std::size_t dtype_size(DType t);
// Canonical safetensors spelling, e.g. "BF16", "F8_E4M3".
const char* dtype_name(DType t);
// Returns false for names this reader does not know.
bool parse_dtype(std::string_view name, DType& out);

struct TensorInfo {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<std::int64_t> shape;
  std::uint64_t offset = 0;  // absolute byte offset in the file
  std::uint64_t nbytes = 0;

  std::int64_t numel() const;
};

class SafetensorsFile {
 public:
  // Reads and validates the header. Throws FormatError / IoError.
  explicit SafetensorsFile(const std::filesystem::path& path);

  const std::filesystem::path& path() const { return path_; }
  std::uint64_t file_size() const { return file_size_; }
  std::uint64_t data_start() const { return data_start_; }
  // Sorted by file offset (the data section is fully covered, so this is also the layout order).
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find(std::string_view name) const;
  // `__metadata__` string map (empty if absent).
  const std::map<std::string, std::string>& metadata() const { return metadata_; }

 private:
  std::filesystem::path path_;
  std::uint64_t file_size_ = 0;
  std::uint64_t data_start_ = 0;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, std::string> metadata_;
};

}  // namespace mi::weights
