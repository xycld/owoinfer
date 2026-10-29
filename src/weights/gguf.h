// Strict reader for the GGUF container (llama.cpp / ggml; versions 2 and 3, little-endian) and the CPU dequantization
// of the ggml block types MOSS-Image GGUF DiTs are published in. Only the header is read here; tensor bytes are fetched
// on demand through weights::FileReader using the absolute offsets recorded here.
//
// Layout: "GGUF", u32 version, u64 tensor count, u64 metadata count, the metadata (string key, u32 value type, value),
// the tensor infos (string name, u32 n_dims, n_dims x u64 ne (innermost first), u32 ggml type, u64 offset relative to
// the data section), then the data section at the header's end rounded up to general.alignment (default 32). Strings
// are a u64 length and that many bytes.
//
// Everything is validated as untrusted input before any tensor byte is read: every count, length, size and offset is
// bounded and overflow-checked, the header is at most kMaxGgufHeaderBytes, each tensor's bytes follow from its shape
// and type, lie inside the file, start on the alignment and overlap no other tensor's; names are unique.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mi::weights {

inline constexpr std::uint64_t kMaxGgufHeaderBytes = std::uint64_t{64} << 20;
inline constexpr std::size_t kMaxGgufTensors = 65536;
inline constexpr std::size_t kMaxGgufMetadata = 65536;
inline constexpr std::size_t kMaxGgufDims = 4;            // GGML_MAX_DIMS
inline constexpr std::size_t kMaxGgufNameBytes = 1024;    // ggml itself allows 63
inline constexpr int kMaxGgufArrayDepth = 4;

// ggml tensor types (ggml.h enum ggml_type), as stored. The ones the engine dequantizes have names here; the others are
// known by their block size only, so a file holding them parses and can be refused by name.
namespace ggml {
inline constexpr std::uint32_t kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ8_0 = 8, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14,
                               kBF16 = 30;
}

// The type's name ("Q4_K", "BF16"), or empty for an id ggml does not define.
std::string_view ggml_type_name(std::uint32_t type);
// Elements per block and bytes per block of a type ggml defines; nullopt for an unknown id.
struct GgmlBlock {
  std::uint32_t elems = 0;
  std::uint32_t bytes = 0;
};
std::optional<GgmlBlock> ggml_block(std::uint32_t type);
// Whether dequantize_gguf_rows reads the type: F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K.
bool gguf_dequantizable(std::uint32_t type);

// Bytes of one row of k elements (k a multiple of the type's block size; throws std::invalid_argument otherwise or for
// an unknown type).
std::uint64_t gguf_row_bytes(std::uint32_t type, std::int64_t k);

// Rows of a tensor of a dequantizable type as fp32, `rows` rows of k elements (k a multiple of the block size) stored
// back to back at `src`. Bit-exact to the gguf Python package's dequantize (gguf.quants), which follows ggml's
// reference dequantize_row_*: fp16 scales widened exactly, each product and difference rounded separately (built
// without FMA contraction). Throws std::invalid_argument for another type or a k that is not a block multiple.
void dequantize_gguf_rows(std::uint32_t type, const std::byte* src, std::int64_t rows, std::int64_t k, float* out);

// IEEE binary16 -> binary32 (exact; NaN stays NaN).
float half_to_float(std::uint16_t h);

struct GgufTensorInfo {
  std::string name;
  std::vector<std::int64_t> shape;  // row-major (PyTorch) order: GGUF's ne reversed, so shape.back() is the row length
  std::uint32_t type = 0;
  std::uint64_t offset = 0;  // absolute byte offset in the file
  std::uint64_t nbytes = 0;

  std::int64_t numel() const;
};

class GgufFile {
 public:
  // Reads and validates the header. Throws FormatError (safetensors.h; the message names the file and, when relevant,
  // the tensor) or IoError (a file that is not a regular file, or unreadable).
  explicit GgufFile(const std::filesystem::path& path);

  const std::filesystem::path& path() const { return path_; }
  std::uint64_t file_size() const { return file_size_; }
  std::uint32_t version() const { return version_; }
  std::uint32_t alignment() const { return alignment_; }
  std::uint64_t data_start() const { return data_start_; }
  std::uint64_t header_bytes() const { return header_bytes_; }  // up to the end of the tensor infos
  // Sorted by file offset.
  const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
  const GgufTensorInfo* find(std::string_view name) const;
  // Scalar metadata values as text (strings as stored, numbers in decimal, bools "true"/"false"); arrays as
  // "[array of N <type>]".
  const std::map<std::string, std::string>& metadata() const { return metadata_; }

 private:
  std::filesystem::path path_;
  std::uint64_t file_size_ = 0;
  std::uint32_t version_ = 0;
  std::uint32_t alignment_ = 32;
  std::uint64_t data_start_ = 0;
  std::uint64_t header_bytes_ = 0;
  std::vector<GgufTensorInfo> tensors_;
  std::map<std::string, std::string> metadata_;
};

// Whether the file starts with the GGUF magic (false for a file shorter than 4 bytes). Throws IoError for a path that is
// not a regular file or cannot be read.
bool is_gguf_file(const std::filesystem::path& path);

}  // namespace mi::weights
