#include "weights/gguf.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

#include "util/checked_math.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "weights/file_reader.h"
#include "weights/safetensors.h"

namespace mi::weights {

namespace {

struct TypeEntry {
  std::uint32_t id;
  const char* name;
  std::uint32_t elems, bytes;
};
// ggml.h's types with their block sizes (ggml.c type_traits; gguf-py GGML_QUANT_SIZES). Removed ids (4, 5: Q4_2/Q4_3;
// 31-33: Q4_0_4_4 and kin; 36-38: IQ4_NL_4_4 and kin) are unknown.
constexpr TypeEntry kTypes[] = {
    {0, "F32", 1, 4},          {1, "F16", 1, 2},          {2, "Q4_0", 32, 18},       {3, "Q4_1", 32, 20},
    {6, "Q5_0", 32, 22},       {7, "Q5_1", 32, 24},       {8, "Q8_0", 32, 34},       {9, "Q8_1", 32, 40},
    {10, "Q2_K", 256, 84},     {11, "Q3_K", 256, 110},    {12, "Q4_K", 256, 144},    {13, "Q5_K", 256, 176},
    {14, "Q6_K", 256, 210},    {15, "Q8_K", 256, 292},    {16, "IQ2_XXS", 256, 66},  {17, "IQ2_XS", 256, 74},
    {18, "IQ3_XXS", 256, 98},  {19, "IQ1_S", 256, 50},    {20, "IQ4_NL", 32, 18},    {21, "IQ3_S", 256, 110},
    {22, "IQ2_S", 256, 82},    {23, "IQ4_XS", 256, 136},  {24, "I8", 1, 1},          {25, "I16", 1, 2},
    {26, "I32", 1, 4},         {27, "I64", 1, 8},         {28, "F64", 1, 8},         {29, "IQ1_M", 256, 56},
    {30, "BF16", 1, 2},        {34, "TQ1_0", 256, 54},    {35, "TQ2_0", 256, 66},    {39, "MXFP4", 32, 17},
    {40, "NVFP4", 64, 36},     {41, "Q1_0", 128, 18},
};

const TypeEntry* type_entry(std::uint32_t id) {
  for (const TypeEntry& e : kTypes) {
    if (e.id == id) return &e;
  }
  return nullptr;
}

std::uint16_t u16_at(const std::byte* p) { return static_cast<std::uint16_t>(std::to_integer<unsigned>(p[0]) | (std::to_integer<unsigned>(p[1]) << 8)); }
std::uint8_t u8_at(const std::byte* p) { return std::to_integer<std::uint8_t>(*p); }

// Q4_K / Q5_K: the 6-bit scale and min of sub-block j (ggml get_scale_min_k4).
void scale_min_k4(int j, const std::byte* q, std::uint8_t& d, std::uint8_t& m) {
  if (j < 4) {
    d = u8_at(q + j) & 63;
    m = u8_at(q + j + 4) & 63;
  } else {
    d = static_cast<std::uint8_t>((u8_at(q + j + 4) & 0xF) | ((u8_at(q + j - 4) >> 6) << 4));
    m = static_cast<std::uint8_t>((u8_at(q + j + 4) >> 4) | ((u8_at(q + j) >> 6) << 4));
  }
}

void dequant_q8_0(const std::byte* b, float* y) {
  const float d = half_to_float(u16_at(b));
  for (int i = 0; i < 32; ++i) y[i] = static_cast<float>(static_cast<std::int8_t>(u8_at(b + 2 + i))) * d;
}

void dequant_q4_0(const std::byte* b, float* y) {
  const float d = half_to_float(u16_at(b));
  for (int i = 0; i < 16; ++i) {
    const std::uint8_t q = u8_at(b + 2 + i);
    y[i] = d * static_cast<float>((q & 0xF) - 8);
    y[i + 16] = d * static_cast<float>((q >> 4) - 8);
  }
}

void dequant_q4_k(const std::byte* b, float* y) {
  const float d = half_to_float(u16_at(b));
  const float dmin = half_to_float(u16_at(b + 2));
  const std::byte* scales = b + 4;
  const std::byte* q = b + 16;
  for (int j = 0; j < 4; ++j, q += 32, y += 64) {
    std::uint8_t sc = 0, m = 0;
    scale_min_k4(2 * j, scales, sc, m);
    const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(m);
    scale_min_k4(2 * j + 1, scales, sc, m);
    const float d2 = d * static_cast<float>(sc), m2 = dmin * static_cast<float>(m);
    for (int l = 0; l < 32; ++l) y[l] = d1 * static_cast<float>(u8_at(q + l) & 0xF) - m1;
    for (int l = 0; l < 32; ++l) y[32 + l] = d2 * static_cast<float>(u8_at(q + l) >> 4) - m2;
  }
}

void dequant_q5_k(const std::byte* b, float* y) {
  const float d = half_to_float(u16_at(b));
  const float dmin = half_to_float(u16_at(b + 2));
  const std::byte* scales = b + 4;
  const std::byte* qh = b + 16;
  const std::byte* ql = b + 48;
  for (int j = 0; j < 4; ++j, ql += 32, y += 64) {
    std::uint8_t sc = 0, m = 0;
    scale_min_k4(2 * j, scales, sc, m);
    const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(m);
    scale_min_k4(2 * j + 1, scales, sc, m);
    const float d2 = d * static_cast<float>(sc), m2 = dmin * static_cast<float>(m);
    const unsigned u1 = 1u << (2 * j), u2 = 2u << (2 * j);
    for (int l = 0; l < 32; ++l) {
      y[l] = d1 * static_cast<float>((u8_at(ql + l) & 0xF) + ((u8_at(qh + l) & u1) ? 16 : 0)) - m1;
    }
    for (int l = 0; l < 32; ++l) {
      y[32 + l] = d2 * static_cast<float>((u8_at(ql + l) >> 4) + ((u8_at(qh + l) & u2) ? 16 : 0)) - m2;
    }
  }
}

void dequant_q6_k(const std::byte* b, float* y) {
  const std::byte* ql = b;
  const std::byte* qh = b + 128;
  const std::byte* sc = b + 192;
  const float d = half_to_float(u16_at(b + 208));
  for (int n = 0; n < 2; ++n, ql += 64, qh += 32, sc += 8, y += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const unsigned h = u8_at(qh + l);
      const int q1 = static_cast<int>((u8_at(ql + l) & 0xF) | (((h >> 0) & 3) << 4)) - 32;
      const int q2 = static_cast<int>((u8_at(ql + l + 32) & 0xF) | (((h >> 2) & 3) << 4)) - 32;
      const int q3 = static_cast<int>((u8_at(ql + l) >> 4) | (((h >> 4) & 3) << 4)) - 32;
      const int q4 = static_cast<int>((u8_at(ql + l + 32) >> 4) | (((h >> 6) & 3) << 4)) - 32;
      const auto s = [&](int i) { return d * static_cast<float>(static_cast<std::int8_t>(u8_at(sc + i))); };
      y[l] = s(is + 0) * static_cast<float>(q1);
      y[l + 32] = s(is + 2) * static_cast<float>(q2);
      y[l + 64] = s(is + 4) * static_cast<float>(q3);
      y[l + 96] = s(is + 6) * static_cast<float>(q4);
    }
  }
}

// The header bytes, read from the file on demand (at most kMaxGgufHeaderBytes), and a cursor over them.
class Cursor {
 public:
  Cursor(const FileReader& in, const std::filesystem::path& path) : in_(in), path_(path) {}

  std::uint64_t pos() const { return pos_; }
  const std::byte* take(std::uint64_t n, const char* what) {
    ensure(n, what);
    const std::byte* p = buf_.data() + pos_;
    pos_ += n;
    return p;
  }
  std::uint32_t u32(const char* what) {
    const std::byte* p = take(4, what);
    std::uint32_t v = 0;
    for (int i = 3; i >= 0; --i) v = (v << 8) | std::to_integer<std::uint32_t>(p[i]);
    return v;
  }
  std::uint64_t u64(const char* what) {
    const std::byte* p = take(8, what);
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | std::to_integer<std::uint64_t>(p[i]);
    return v;
  }
  std::string str(std::uint64_t max, const char* what) {
    const std::uint64_t n = u64(what);
    if (n > max) fail(std::string(what) + ": string length " + std::to_string(n) + " exceeds " + std::to_string(max));
    const std::byte* p = take(n, what);
    return std::string(reinterpret_cast<const char*>(p), static_cast<std::size_t>(n));
  }
  // Bytes still allowed in the header from here (the cap or the file's end, whichever is first).
  std::uint64_t remaining() const { return limit() - pos_; }
  [[noreturn]] void fail(const std::string& msg, const std::string& tensor = {}) const { throw FormatError(path_, tensor, msg); }

 private:
  std::uint64_t limit() const { return std::min(in_.size(), kMaxGgufHeaderBytes); }
  void ensure(std::uint64_t n, const char* what) {
    if (n > remaining()) {
      if (n > in_.size() - pos_) fail(std::string(what) + ": past the end of the file (truncated file?)");
      fail(std::string(what) + ": the header exceeds the " + std::to_string(kMaxGgufHeaderBytes >> 20) + " MiB limit");
    }
    const std::uint64_t need = pos_ + n;
    if (need <= buf_.size()) return;
    // Grow geometrically from 64 KiB, so a header is read in a few calls.
    std::uint64_t size = std::max<std::uint64_t>(buf_.size() * 2, std::uint64_t{64} << 10);
    size = std::min(std::max(size, need), limit());
    const std::size_t old = buf_.size();
    buf_.resize(static_cast<std::size_t>(size));
    in_.read_at(old, buf_.data() + old, buf_.size() - old);
  }

  const FileReader& in_;
  const std::filesystem::path& path_;
  std::vector<std::byte> buf_;
  std::uint64_t pos_ = 0;
};

constexpr std::uint32_t kTypeString = 8, kTypeArray = 9;
// Fixed sizes of the scalar value types (gguf.h gguf_type): uint8, int8, uint16, int16, uint32, int32, float32, bool,
// (string), (array), uint64, int64, float64.
int scalar_size(std::uint32_t t) {
  switch (t) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
  }
}
const char* value_type_name(std::uint32_t t) {
  constexpr const char* kNames[] = {"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32",
                                    "bool", "string", "array", "uint64", "int64", "float64"};
  return t < std::size(kNames) ? kNames[t] : "unknown";
}

std::string scalar_text(std::uint32_t t, const std::byte* p) {
  std::uint64_t u = 0;
  for (int i = scalar_size(t) - 1; i >= 0; --i) u = (u << 8) | std::to_integer<std::uint64_t>(p[i]);
  switch (t) {
    case 0: case 2: case 4: case 10: return std::to_string(u);
    case 1: return std::to_string(static_cast<std::int8_t>(u));
    case 3: return std::to_string(static_cast<std::int16_t>(u));
    case 5: return std::to_string(static_cast<std::int32_t>(u));
    case 11: return std::to_string(static_cast<std::int64_t>(u));
    case 6: return std::to_string(std::bit_cast<float>(static_cast<std::uint32_t>(u)));
    case 12: return std::to_string(std::bit_cast<double>(u));
    case 7: return u ? "true" : "false";
    default: return {};
  }
}

// Reads one value of type t; returns its text (scalars and strings) or an array summary.
std::string read_value(Cursor& c, std::uint32_t t, const std::string& key, int depth) {
  if (t == kTypeString) return c.str(c.remaining(), "metadata string");
  if (t == kTypeArray) {
    if (depth >= kMaxGgufArrayDepth) c.fail("metadata '" + key + "': arrays nested deeper than " + std::to_string(kMaxGgufArrayDepth));
    const std::uint32_t et = c.u32("metadata array type");
    const std::uint64_t n = c.u64("metadata array length");
    if (et == kTypeString || et == kTypeArray) {
      // Each element takes at least its 8-byte length (strings) or 12-byte type and length (arrays).
      if (n > c.remaining() / (et == kTypeString ? 8 : 12)) c.fail("metadata '" + key + "': array length " + std::to_string(n) + " exceeds the header");
      for (std::uint64_t i = 0; i < n; ++i) read_value(c, et, key, depth + 1);
    } else {
      const int es = scalar_size(et);
      if (es == 0) c.fail("metadata '" + key + "': unknown array element type " + std::to_string(et));
      if (n > c.remaining() / static_cast<std::uint64_t>(es)) c.fail("metadata '" + key + "': array length " + std::to_string(n) + " exceeds the header");
      c.take(n * static_cast<std::uint64_t>(es), "metadata array");
    }
    return "[array of " + std::to_string(n) + " " + value_type_name(et) + "]";
  }
  const int es = scalar_size(t);
  if (es == 0) c.fail("metadata '" + key + "': unknown value type " + std::to_string(t));
  return scalar_text(t, c.take(static_cast<std::uint64_t>(es), "metadata value"));
}

}  // namespace

std::string_view ggml_type_name(std::uint32_t type) {
  const TypeEntry* e = type_entry(type);
  return e != nullptr ? e->name : std::string_view();
}

std::optional<GgmlBlock> ggml_block(std::uint32_t type) {
  const TypeEntry* e = type_entry(type);
  if (e == nullptr) return std::nullopt;
  return GgmlBlock{e->elems, e->bytes};
}

bool gguf_dequantizable(std::uint32_t type) {
  using namespace ggml;
  return type == kF32 || type == kF16 || type == kBF16 || type == kQ8_0 || type == kQ4_0 || type == kQ4_K || type == kQ5_K ||
         type == kQ6_K;
}

std::uint64_t gguf_row_bytes(std::uint32_t type, std::int64_t k) {
  const TypeEntry* e = type_entry(type);
  if (e == nullptr || k < 0 || k % e->elems != 0) {
    throw std::invalid_argument("gguf_row_bytes: k=" + std::to_string(k) + " is not a multiple of the block of type " + std::to_string(type));
  }
  return static_cast<std::uint64_t>(k) / e->elems * e->bytes;
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
  if (exp == 0) {
    const float mag = static_cast<float>(mant) * 0x1p-24f;  // zero or subnormal: exact
    return std::bit_cast<float>(std::bit_cast<std::uint32_t>(mag) | sign);
  }
  if (exp == 31) return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13));
  return std::bit_cast<float>(sign | ((exp + 112) << 23) | (mant << 13));
}

void dequantize_gguf_rows(std::uint32_t type, const std::byte* src, std::int64_t rows, std::int64_t k, float* out) {
  using namespace ggml;
  const TypeEntry* e = type_entry(type);
  if (e == nullptr || !gguf_dequantizable(type)) {
    throw std::invalid_argument("dequantize_gguf_rows: type " + std::to_string(type) + " is not dequantizable");
  }
  if (rows < 0 || k < 0 || k % e->elems != 0) throw std::invalid_argument("dequantize_gguf_rows: bad shape");
  const std::int64_t n = rows * k;
  switch (type) {
    case kF32: std::memcpy(out, src, static_cast<std::size_t>(n) * 4); return;
    case kF16:
      for (std::int64_t i = 0; i < n; ++i) out[i] = half_to_float(u16_at(src + 2 * i));
      return;
    case kBF16:
      for (std::int64_t i = 0; i < n; ++i) out[i] = std::bit_cast<float>(static_cast<std::uint32_t>(u16_at(src + 2 * i)) << 16);
      return;
    default: break;
  }
  void (*block)(const std::byte*, float*) = type == kQ8_0   ? dequant_q8_0
                                            : type == kQ4_0 ? dequant_q4_0
                                            : type == kQ4_K ? dequant_q4_k
                                            : type == kQ5_K ? dequant_q5_k
                                                            : dequant_q6_k;
  const std::int64_t blocks = n / e->elems;
  for (std::int64_t i = 0; i < blocks; ++i) block(src + i * e->bytes, out + i * e->elems);
}

std::int64_t GgufTensorInfo::numel() const {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

GgufFile::GgufFile(const std::filesystem::path& path) : path_(path) {
  util::check_user_path(path, "GGUF file");
  const FileReader in(path);  // IoError for anything but a regular file, before any read
  file_size_ = in.size();
  Cursor c(in, path_);
  const std::byte* magic = c.take(4, "GGUF magic");
  if (std::memcmp(magic, "GGUF", 4) != 0) c.fail("not a GGUF file (no GGUF magic)");
  version_ = c.u32("GGUF version");
  // A big-endian file (GGUF v3 allows them) stores its version byte-swapped.
  if (version_ == 0x02000000u || version_ == 0x03000000u) c.fail("a big-endian GGUF file; only little-endian files are read");
  if (version_ != 2 && version_ != 3) c.fail("GGUF version " + std::to_string(version_) + " (versions 2 and 3 are read)");
  const std::uint64_t n_tensors = c.u64("tensor count");
  const std::uint64_t n_kv = c.u64("metadata count");
  if (n_tensors > kMaxGgufTensors) c.fail("tensor count " + std::to_string(n_tensors) + " exceeds " + std::to_string(kMaxGgufTensors));
  if (n_kv > kMaxGgufMetadata) c.fail("metadata count " + std::to_string(n_kv) + " exceeds " + std::to_string(kMaxGgufMetadata));

  for (std::uint64_t i = 0; i < n_kv; ++i) {
    std::string key = c.str(65535, "metadata key");
    const std::uint32_t t = c.u32("metadata value type");
    std::string value = read_value(c, t, key, 0);
    if (key == "general.alignment") {
      if (t != 4) c.fail("general.alignment is not a uint32");
      const std::uint64_t a = std::stoull(value);
      if (a == 0 || a > 65536 || (a & (a - 1)) != 0) c.fail("general.alignment " + value + " is not a power of two up to 65536");
      alignment_ = static_cast<std::uint32_t>(a);
    }
    if (!metadata_.emplace(std::move(key), std::move(value)).second) c.fail("duplicate metadata key");
  }

  std::unordered_set<std::string> names;
  tensors_.reserve(static_cast<std::size_t>(n_tensors));
  std::vector<std::uint64_t> rel;
  rel.reserve(static_cast<std::size_t>(n_tensors));
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    GgufTensorInfo t;
    t.name = c.str(kMaxGgufNameBytes, "tensor name");
    if (t.name.empty()) c.fail("empty tensor name");
    if (!names.insert(t.name).second) c.fail("duplicate tensor name", t.name);
    const std::uint32_t n_dims = c.u32("tensor rank");
    if (n_dims == 0 || n_dims > kMaxGgufDims) c.fail("rank " + std::to_string(n_dims) + " outside [1, " + std::to_string(kMaxGgufDims) + "]", t.name);
    std::vector<std::int64_t> ne;
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      const std::uint64_t v = c.u64("tensor dimension");
      if (v == 0 || v > static_cast<std::uint64_t>(INT64_MAX)) c.fail("dimension " + std::to_string(v) + " out of range", t.name);
      ne.push_back(static_cast<std::int64_t>(v));
    }
    t.shape.assign(ne.rbegin(), ne.rend());
    t.type = c.u32("tensor type");
    const TypeEntry* e = type_entry(t.type);
    if (e == nullptr) c.fail("unknown ggml type " + std::to_string(t.type), t.name);
    const std::optional<std::int64_t> numel = util::checked_numel(t.shape);
    if (!numel) c.fail("element count overflows", t.name);
    if (ne[0] % e->elems != 0) {
      c.fail("row length " + std::to_string(ne[0]) + " is not a multiple of the " + e->name + " block (" + std::to_string(e->elems) + ")", t.name);
    }
    const std::optional<std::uint64_t> bytes = util::checked_mul(static_cast<std::uint64_t>(*numel) / e->elems, e->bytes);
    if (!bytes) c.fail("byte size overflows", t.name);
    t.nbytes = *bytes;
    rel.push_back(c.u64("tensor offset"));
    tensors_.push_back(std::move(t));
  }
  header_bytes_ = c.pos();
  data_start_ = (header_bytes_ + alignment_ - 1) / alignment_ * alignment_;
  if (data_start_ > file_size_) c.fail("data section starts past the end of the file (truncated file?)");
  const std::uint64_t data_size = file_size_ - data_start_;
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    GgufTensorInfo& t = tensors_[i];
    if (rel[i] % alignment_ != 0) c.fail("offset " + std::to_string(rel[i]) + " is not a multiple of the alignment " + std::to_string(alignment_), t.name);
    if (rel[i] > data_size || t.nbytes > data_size - rel[i]) {
      c.fail("bytes [" + std::to_string(rel[i]) + ", +" + std::to_string(t.nbytes) + ") lie past the data section (" +
                 std::to_string(data_size) + " bytes; truncated file?)",
             t.name);
    }
    t.offset = data_start_ + rel[i];
  }
  std::sort(tensors_.begin(), tensors_.end(), [](const GgufTensorInfo& a, const GgufTensorInfo& b) {
    return a.offset != b.offset ? a.offset < b.offset : a.nbytes < b.nbytes;
  });
  for (std::size_t i = 1; i < tensors_.size(); ++i) {
    const GgufTensorInfo& prev = tensors_[i - 1];
    if (tensors_[i].offset < prev.offset + prev.nbytes) c.fail("overlaps tensor '" + prev.name + "'", tensors_[i].name);
  }
}

const GgufTensorInfo* GgufFile::find(std::string_view name) const {
  for (const GgufTensorInfo& t : tensors_) {
    if (t.name == name) return &t;
  }
  return nullptr;
}

bool is_gguf_file(const std::filesystem::path& path) {
  const FileReader in(path);
  if (in.size() < 4) return false;
  char magic[4];
  in.read_at(0, magic, 4);
  return std::memcmp(magic, "GGUF", 4) == 0;
}

}  // namespace mi::weights
