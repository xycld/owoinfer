#include "weights/safetensors.h"

#include <algorithm>
#include <optional>
#include <unordered_set>

#include "util/checked_math.h"
#include "util/json.h"
#include "weights/file_reader.h"

namespace mi::weights {

namespace {

// JSON values one tensor entry can need (entry, dtype, shape, dims, data_offsets and its two ends),
// plus room for __metadata__. Bounds the DOM at a few tens of MB however the header is written.
constexpr std::size_t kMaxHeaderValues = 1 + kMaxTensors * (6 + kMaxRank) + 65536;

struct DTypeEntry {
  const char* name;
  DType type;
  std::size_t size;
};
constexpr DTypeEntry kDTypes[] = {
    {"BOOL", DType::kBool, 1},     {"U8", DType::kU8, 1},         {"I8", DType::kI8, 1},
    {"F8_E4M3", DType::kF8E4M3, 1}, {"F8_E5M2", DType::kF8E5M2, 1}, {"F8_E8M0", DType::kF8E8M0, 1},
    {"I16", DType::kI16, 2},       {"U16", DType::kU16, 2},       {"F16", DType::kF16, 2},
    {"BF16", DType::kBF16, 2},     {"I32", DType::kI32, 4},       {"U32", DType::kU32, 4},
    {"F32", DType::kF32, 4},       {"I64", DType::kI64, 8},       {"U64", DType::kU64, 8},
    {"F64", DType::kF64, 8},
};

const DTypeEntry& entry(DType t) {
  for (const auto& e : kDTypes) {
    if (e.type == t) return e;
  }
  throw std::logic_error("unknown DType");
}

}  // namespace

std::size_t dtype_size(DType t) { return entry(t).size; }
const char* dtype_name(DType t) { return entry(t).name; }

bool parse_dtype(std::string_view name, DType& out) {
  for (const auto& e : kDTypes) {
    if (name == e.name) {
      out = e.type;
      return true;
    }
  }
  return false;
}

std::int64_t TensorInfo::numel() const {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : path_(path) {
  FileReader reader(path);
  file_size_ = reader.size();
  if (file_size_ < 8) throw FormatError(path, "", "file too small for a safetensors header length (" + std::to_string(file_size_) + " bytes)");

  unsigned char len_bytes[8];
  reader.read_at(0, len_bytes, 8);
  std::uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];
  if (header_len == 0) throw FormatError(path, "", "empty header");
  if (header_len > kMaxHeaderBytes) throw FormatError(path, "", "header length " + std::to_string(header_len) + " exceeds the 16 MiB limit");
  if (header_len > file_size_ - 8) {
    throw FormatError(path, "", "header length " + std::to_string(header_len) + " exceeds file size " + std::to_string(file_size_) + " (truncated file?)");
  }
  data_start_ = 8 + header_len;
  const std::uint64_t data_size = file_size_ - data_start_;

  std::string text(static_cast<std::size_t>(header_len), '\0');
  reader.read_at(8, text.data(), text.size());

  json::Value root;
  try {
    root = json::parse(text, {.max_values = kMaxHeaderValues});
  } catch (const json::ParseError& e) {
    throw FormatError(path, "", std::string("header is not valid JSON (") + e.what() + ")");
  }
  if (!root.is_object()) throw FormatError(path, "", "header JSON is not an object");

  if (root.as_object().size() > kMaxTensors + 1) {
    throw FormatError(path, "", "more than " + std::to_string(kMaxTensors) + " tensors");
  }
  std::unordered_set<std::string> seen;
  for (const auto& [key, val] : root.as_object()) {
    if (!seen.insert(key).second) throw FormatError(path, key, "duplicate tensor name");
    if (key == "__metadata__") {
      if (!val.is_object()) throw FormatError(path, key, "__metadata__ is not an object");
      for (const auto& [mk, mv] : val.as_object()) {
        if (!mv.is_string()) throw FormatError(path, key, "metadata value for '" + mk + "' is not a string");
        metadata_[mk] = mv.as_string();
      }
      continue;
    }
    if (!val.is_object()) throw FormatError(path, key, "tensor entry is not an object");

    TensorInfo t;
    t.name = key;
    const json::Value* dt = val.find("dtype");
    if (dt == nullptr || !dt->is_string()) throw FormatError(path, key, "missing or non-string dtype");
    if (!parse_dtype(dt->as_string(), t.dtype)) throw FormatError(path, key, "unsupported dtype '" + dt->as_string() + "'");

    const json::Value* shape = val.find("shape");
    if (shape == nullptr || !shape->is_array()) throw FormatError(path, key, "missing or non-array shape");
    if (shape->as_array().size() > kMaxRank) {
      throw FormatError(path, key, "rank " + std::to_string(shape->as_array().size()) + " exceeds " + std::to_string(kMaxRank));
    }
    for (const auto& d : shape->as_array()) {
      std::int64_t dim = 0;
      try {
        dim = d.as_int();
      } catch (const std::exception&) {
        throw FormatError(path, key, "shape contains a non-integer");
      }
      if (dim < 0) throw FormatError(path, key, "negative dimension " + std::to_string(dim));
      t.shape.push_back(dim);
    }
    const std::optional<std::int64_t> numel = util::checked_numel(t.shape);
    if (!numel) throw FormatError(path, key, "element count overflows");
    const std::optional<std::uint64_t> bytes = util::checked_mul(static_cast<std::uint64_t>(*numel), dtype_size(t.dtype));
    if (!bytes) throw FormatError(path, key, "byte size overflows");
    const std::uint64_t expect = *bytes;

    const json::Value* offs = val.find("data_offsets");
    if (offs == nullptr || !offs->is_array() || offs->as_array().size() != 2) {
      throw FormatError(path, key, "data_offsets must be an array of two integers");
    }
    std::int64_t b = 0, e = 0;
    try {
      b = offs->as_array()[0].as_int();
      e = offs->as_array()[1].as_int();
    } catch (const std::exception&) {
      throw FormatError(path, key, "data_offsets contains a non-integer");
    }
    if (b < 0 || e < b) throw FormatError(path, key, "invalid data_offsets [" + std::to_string(b) + ", " + std::to_string(e) + "]");
    if (static_cast<std::uint64_t>(e) > data_size) {
      throw FormatError(path, key, "data_offsets end " + std::to_string(e) + " is past the data section (" + std::to_string(data_size) + " bytes; truncated file?)");
    }
    t.nbytes = static_cast<std::uint64_t>(e - b);
    if (t.nbytes != expect) {
      throw FormatError(path, key, "byte size " + std::to_string(t.nbytes) + " does not match shape x dtype (" + std::to_string(expect) + ")");
    }
    t.offset = data_start_ + static_cast<std::uint64_t>(b);
    tensors_.push_back(std::move(t));
  }

  // The format requires the data section to be exactly covered by the tensors, in any order.
  std::sort(tensors_.begin(), tensors_.end(), [](const TensorInfo& a, const TensorInfo& b) {
    return a.offset != b.offset ? a.offset < b.offset : a.nbytes < b.nbytes;
  });
  std::uint64_t cursor = data_start_;
  for (const auto& t : tensors_) {
    if (t.offset < cursor) throw FormatError(path, t.name, "overlaps the preceding tensor");
    if (t.offset > cursor) throw FormatError(path, t.name, "gap of " + std::to_string(t.offset - cursor) + " bytes before tensor (data section must be densely packed)");
    cursor += t.nbytes;
  }
  if (cursor != file_size_) {
    throw FormatError(path, "", std::to_string(file_size_ - cursor) + " trailing bytes not covered by any tensor");
  }
}

const TensorInfo* SafetensorsFile::find(std::string_view name) const {
  for (const auto& t : tensors_) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

}  // namespace mi::weights
