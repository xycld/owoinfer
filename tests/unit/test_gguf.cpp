// GGUF DiT import: the container reader (weights/gguf.h) on valid, malformed and hostile files, the dequantization
// against gguf-py (tests/data/gguf, tools/reference/gen_gguf_vectors.py), the DiT reader and converter
// (quant::open_gguf_dit, quant::convert), the supported-model check, inspection and the converted-file cache
// (pipeline/gguf_dit.h). CPU only.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "api/file_inspect.h"
#include "pipeline/gguf_dit.h"
#include "test_util.h"
#include "util/npy.h"
#include "weights/file_reader.h"
#include "weights/gguf.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_convert.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"

// The synthetic tensors and layouts below leave optional fields to their defaults.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

namespace {

namespace fs = std::filesystem;
namespace ggml = mi::weights::ggml;
using mi::weights::DType;
using mi::weights::FormatError;
using mi::weights::GgufFile;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/gguf";

fs::path community() {
  const char* env = std::getenv("MOSS_COMMUNITY");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/community");
}

// ---- A GGUF writer ------------------------------------------------------------------------------------------

std::string u32(std::uint32_t v) {
  std::string s(4, '\0');
  for (int i = 0; i < 4; ++i) s[static_cast<std::size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
  return s;
}
std::string u64(std::uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[static_cast<std::size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
  return s;
}
std::string str(const std::string& s) { return u64(s.size()) + s; }

struct KV {
  std::string key;
  std::uint32_t type;
  std::string value;  // encoded
};
KV kv_string(const std::string& k, const std::string& v) { return {k, 8, str(v)}; }
KV kv_u32(const std::string& k, std::uint32_t v) { return {k, 4, u32(v)}; }

struct GT {
  std::string name;
  std::vector<std::int64_t> shape;  // row-major (PyTorch) order
  std::uint32_t type = ggml::kF32;
  std::vector<std::byte> data;
  std::optional<std::uint64_t> offset;  // relative offset; default: packed at the alignment
};

struct Layout {
  std::uint32_t version = 3;
  std::uint32_t alignment = 32;
  std::optional<std::uint64_t> tensor_count, kv_count;
};

std::string gguf_bytes(const std::vector<GT>& ts, const std::vector<KV>& kvs = {}, const Layout& lay = {}) {
  std::string h = "GGUF" + u32(lay.version) + u64(lay.tensor_count.value_or(ts.size())) + u64(lay.kv_count.value_or(kvs.size()));
  for (const KV& kv : kvs) h += str(kv.key) + u32(kv.type) + kv.value;
  std::vector<std::uint64_t> offsets;
  std::uint64_t cursor = 0;
  for (const GT& t : ts) {
    h += str(t.name) + u32(static_cast<std::uint32_t>(t.shape.size()));
    for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it) h += u64(static_cast<std::uint64_t>(*it));
    const std::uint64_t off = t.offset.value_or(cursor);
    h += u32(t.type) + u64(off);
    offsets.push_back(off);
    cursor = (std::max(cursor, off + t.data.size()) + lay.alignment - 1) / lay.alignment * lay.alignment;
  }
  h.append((lay.alignment - h.size() % lay.alignment) % lay.alignment, '\0');
  std::string data;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    if (ts[i].data.empty()) continue;  // header-only tensors (a sparse data section written separately)
    if (data.size() < offsets[i] + ts[i].data.size()) data.resize(offsets[i] + ts[i].data.size(), '\0');
    std::memcpy(data.data() + offsets[i], ts[i].data.data(), ts[i].data.size());
  }
  return h + data;
}

void write_file(const fs::path& p, const std::string& bytes) {
  fs::create_directories(p.parent_path());
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

template <typename V>
std::vector<std::byte> bytes_of(const std::vector<V>& v) {
  std::vector<std::byte> b(v.size() * sizeof(V));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

// IEEE binary16 bits of x, rounded to nearest even (finite x).
std::uint16_t to_half(float x) {
  const std::uint32_t sign = std::signbit(x) ? 0x8000u : 0u;
  const float a = std::fabs(x);
  if (a < 0x1p-14f) return static_cast<std::uint16_t>(sign | static_cast<std::uint32_t>(std::nearbyint(a * 0x1p24f)));
  int e = 0;
  const float fr = std::frexp(a, &e);  // a = fr * 2^e, fr in [0.5, 1)
  auto m = static_cast<std::uint32_t>(std::nearbyint((fr * 2.0f - 1.0f) * 1024.0f));
  auto exp = static_cast<std::uint32_t>(e - 1 + 15);
  if (m == 1024) {
    m = 0;
    ++exp;
  }
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
  return static_cast<std::uint16_t>(sign | (exp << 10) | m);
}

// Rows of random stored blocks of `type` with finite fp16 scales at their ggml offsets (as gen_gguf_vectors.py).
std::vector<std::byte> random_rows(std::uint32_t type, std::int64_t rows, std::int64_t k, std::uint32_t seed) {
  std::mt19937 g(seed);
  if (type == ggml::kF32 || type == ggml::kBF16 || type == ggml::kF16) {
    std::normal_distribution<float> d(0.0f, 0.05f);
    std::vector<std::byte> out;
    for (std::int64_t i = 0; i < rows * k; ++i) {
      const float x = d(g);
      if (type == ggml::kF32) {
        const auto b = std::bit_cast<std::array<std::byte, 4>>(x);
        out.insert(out.end(), b.begin(), b.end());
      } else {
        const std::uint16_t h = type == ggml::kBF16 ? mi::quant::float_to_bf16(x) : to_half(x);
        out.push_back(static_cast<std::byte>(h & 0xFF));
        out.push_back(static_cast<std::byte>(h >> 8));
      }
    }
    return out;
  }
  const auto blk = *mi::weights::ggml_block(type);
  const std::int64_t blocks = rows * k / blk.elems;
  std::vector<std::byte> out(static_cast<std::size_t>(blocks) * blk.bytes);
  std::uniform_int_distribution<int> byte(0, 255);
  for (auto& b : out) b = static_cast<std::byte>(byte(g));
  std::normal_distribution<float> scale(0.0f, type == ggml::kQ8_0 || type == ggml::kQ4_0 ? 0.002f : 0.0002f);
  const auto put_half = [&](std::size_t at) {
    const std::uint16_t h = to_half(scale(g));
    out[at] = static_cast<std::byte>(h & 0xFF);
    out[at + 1] = static_cast<std::byte>(h >> 8);
  };
  for (std::int64_t i = 0; i < blocks; ++i) {
    const std::size_t b0 = static_cast<std::size_t>(i) * blk.bytes;
    if (type == ggml::kQ6_K) {
      put_half(b0 + 208);
    } else {
      put_half(b0);
      if (type == ggml::kQ4_K || type == ggml::kQ5_K) put_half(b0 + 2);
    }
  }
  return out;
}

std::vector<float> dequant(std::uint32_t type, const std::vector<std::byte>& raw, std::int64_t rows, std::int64_t k) {
  std::vector<float> f(static_cast<std::size_t>(rows * k));
  mi::weights::dequantize_gguf_rows(type, raw.data(), rows, k, f.data());
  return f;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// ---- Dequantization -----------------------------------------------------------------------------------------

TEST(GgufDequant, BitExactToGgufPy) {
  const std::pair<const char*, std::uint32_t> types[] = {{"F32", ggml::kF32},   {"F16", ggml::kF16},   {"BF16", ggml::kBF16},
                                                         {"Q8_0", ggml::kQ8_0}, {"Q4_0", ggml::kQ4_0}, {"Q4_K", ggml::kQ4_K},
                                                         {"Q5_K", ggml::kQ5_K}, {"Q6_K", ggml::kQ6_K}};
  for (const auto& [name, type] : types) {
    SCOPED_TRACE(name);
    const auto raw = mi::npy::load(kData / (std::string(name) + "_raw.npy"));
    const auto ref = mi::npy::load(kData / (std::string(name) + "_f32.npy"));
    ASSERT_EQ(ref.shape.size(), 2u);
    const std::int64_t rows = ref.shape[0], k = ref.shape[1];
    EXPECT_EQ(static_cast<std::uint64_t>(raw.shape[1]), mi::weights::gguf_row_bytes(type, k));
    const std::vector<float> want = ref.to_vector<float>();
    EXPECT_TRUE(same_bits(dequant(type, raw.data, rows, k), want));
    // Row by row gives the same values.
    std::vector<float> one(static_cast<std::size_t>(k));
    for (std::int64_t r = 0; r < rows; ++r) {
      mi::weights::dequantize_gguf_rows(type, raw.data.data() + r * raw.shape[1], 1, k, one.data());
      EXPECT_EQ(std::memcmp(one.data(), want.data() + r * k, static_cast<std::size_t>(k) * 4), 0) << "row " << r;
    }
  }
}

TEST(GgufDequant, HalfToFloat) {
  for (std::uint32_t h = 0; h < 0x10000; ++h) {
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const float mag = e == 0    ? std::ldexp(static_cast<float>(m), -24)
                      : e == 31 ? (m ? NAN : INFINITY)
                                : std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, e - 15);
    const float want = (h & 0x8000) ? -mag : mag;
    const float got = mi::weights::half_to_float(static_cast<std::uint16_t>(h));
    if (std::isnan(want)) {
      EXPECT_TRUE(std::isnan(got)) << h;
    } else {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(got), std::bit_cast<std::uint32_t>(want)) << h;
    }
  }
}

TEST(GgufDequant, RefusesOtherTypesAndShapes) {
  std::vector<std::byte> raw(1024);
  std::vector<float> out(1024);
  EXPECT_THROW(mi::weights::dequantize_gguf_rows(11 /* Q3_K */, raw.data(), 1, 256, out.data()), std::invalid_argument);
  EXPECT_THROW(mi::weights::dequantize_gguf_rows(99, raw.data(), 1, 256, out.data()), std::invalid_argument);
  EXPECT_THROW(mi::weights::dequantize_gguf_rows(ggml::kQ4_K, raw.data(), 1, 100, out.data()), std::invalid_argument);
  EXPECT_THROW(mi::weights::gguf_row_bytes(ggml::kQ8_0, 33), std::invalid_argument);
  EXPECT_EQ(mi::weights::gguf_row_bytes(ggml::kQ4_K, 3072), 12u * 144u);
  EXPECT_FALSE(mi::weights::gguf_dequantizable(10));  // Q2_K
  EXPECT_EQ(mi::weights::ggml_type_name(11), "Q3_K");
  EXPECT_EQ(mi::weights::ggml_type_name(99), "");
}

// The first rows of a few block GEMMs of the published files (gen_gguf_vectors.py --real), read from the files.
TEST(GgufDequant, PublishedFilesRows) {
  const std::pair<const char*, fs::path> files[] = {
      {"MOSS_Image_SFT_Q4_K", community() / "gguf/leejet/MOSS-Image-SFT-GGUF/MOSS-Image-SFT-Q4_K.gguf"},
      {"MOSS_Image_SFT_Q8_0", community() / "gguf/leejet/MOSS-Image-SFT-GGUF/MOSS-Image-SFT-Q8_0.gguf"},
      {"MOSS_Image_SFT_Q4_K_M", community() / "gguf/unsloth/MOSS-Image-SFT-GGUF/MOSS-Image-SFT-Q4_K_M.gguf"}};
  int checked = 0;
  for (const auto& [stem, path] : files) {
    if (!fs::exists(path)) continue;
    const GgufFile f(path);
    for (const auto& t : f.tensors()) {
      std::string tag = "real_" + std::string(stem) + "_" + t.name;
      std::replace(tag.begin(), tag.end(), '.', '_');
      if (!fs::exists(kData / (tag + "_raw.npy"))) continue;
      SCOPED_TRACE(tag);
      const auto raw = mi::npy::load(kData / (tag + "_raw.npy"));
      const auto ref = mi::npy::load(kData / (tag + "_f32.npy"));
      EXPECT_EQ(static_cast<std::uint32_t>(mi::npy::load(kData / (tag + "_type.npy")).to_vector<std::int32_t>()[0]), t.type);
      const std::int64_t k = t.shape.back();
      std::vector<std::byte> stored(raw.data.size());
      mi::weights::FileReader(path).read_at(t.offset, stored.data(), stored.size());
      EXPECT_EQ(stored, raw.data);
      EXPECT_TRUE(same_bits(dequant(t.type, stored, 2, k), ref.to_vector<float>()));
      ++checked;
    }
  }
  if (checked == 0) GTEST_SKIP() << "no published GGUF file (or its vectors) under " << community();
}

// ---- Container ----------------------------------------------------------------------------------------------

TEST(GgufFile, ParsesHeaderMetadataAndTensors) {
  mi::testing::TempDir dir("gguf_parse");
  const auto q8 = random_rows(ggml::kQ8_0, 2, 64, 1);
  const auto bf = random_rows(ggml::kBF16, 1, 3, 2);
  const auto f32 = random_rows(ggml::kF32, 1, 5, 3);
  // An array of strings, a nested array and a u8 array: skipped, summarized.
  const std::string strings = u32(8) + u64(2) + str("a") + str("bc");
  const std::string nested = u32(9) + u64(1) + u32(4) + u64(2) + u32(7) + u32(8);
  const std::string bytes = u32(0) + u64(3) + "xyz";
  const std::vector<KV> kvs = {kv_string("general.architecture", "moss_image"), kv_u32("general.alignment", 64),
                               {"tok", 9, strings}, {"nested", 9, nested}, {"u8s", 9, bytes}, {"f", 6, u32(0x3FC00000)},
                               {"b", 7, std::string(1, '\1')}, {"i64", 11, u64(static_cast<std::uint64_t>(-5))}};
  const fs::path p = dir.path() / "x.gguf";
  write_file(p, gguf_bytes({{"w", {2, 64}, ggml::kQ8_0, q8}, {"b", {3}, ggml::kBF16, bf}, {"f", {5}, ggml::kF32, f32}}, kvs,
                           {.alignment = 64}));
  EXPECT_TRUE(mi::weights::is_gguf_file(p));
  const GgufFile f(p);
  EXPECT_EQ(f.version(), 3u);
  EXPECT_EQ(f.alignment(), 64u);
  EXPECT_EQ(f.data_start() % 64, 0u);
  EXPECT_EQ(f.metadata().at("general.architecture"), "moss_image");
  EXPECT_EQ(f.metadata().at("tok"), "[array of 2 string]");
  EXPECT_EQ(f.metadata().at("nested"), "[array of 1 array]");
  EXPECT_EQ(f.metadata().at("u8s"), "[array of 3 uint8]");
  EXPECT_EQ(f.metadata().at("b"), "true");
  EXPECT_EQ(f.metadata().at("i64"), "-5");
  EXPECT_EQ(f.metadata().at("f").substr(0, 3), "1.5");
  ASSERT_EQ(f.tensors().size(), 3u);
  const auto* w = f.find("w");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->shape, (std::vector<std::int64_t>{2, 64}));
  EXPECT_EQ(w->nbytes, 2u * 2u * 34u);
  EXPECT_EQ(w->offset, f.data_start());
  for (const auto& t : f.tensors()) EXPECT_EQ((t.offset - f.data_start()) % 64, 0u) << t.name;
  std::vector<std::byte> back(w->nbytes);
  mi::weights::FileReader(p).read_at(w->offset, back.data(), back.size());
  EXPECT_EQ(back, q8);
  // Version 2 has the same layout.
  write_file(p, gguf_bytes({{"w", {2, 64}, ggml::kQ8_0, q8}}, {}, {.version = 2}));
  EXPECT_EQ(GgufFile(p).version(), 2u);
}

// What a malformed or hostile file is refused with: a FormatError naming the problem, before any tensor byte is read
// and without allocating what the header claims.
void expect_refused(const fs::path& p, const std::string& bytes, const std::string& what) {
  SCOPED_TRACE(what);
  write_file(p, bytes);
  try {
    GgufFile f(p);
    ADD_FAILURE() << "accepted";
  } catch (const FormatError& e) {
    EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
  }
}

TEST(GgufFile, RefusesMalformedAndHostileHeaders) {
  mi::testing::TempDir dir("gguf_hostile");
  const fs::path p = dir.path() / "h.gguf";
  const auto q8 = random_rows(ggml::kQ8_0, 1, 32, 1);
  const GT one{"w", {1, 32}, ggml::kQ8_0, q8};
  const std::string good = gguf_bytes({one}, {kv_string("k", "v")});
  write_file(p, good);
  ASSERT_NO_THROW(GgufFile f(p));

  expect_refused(p, "GGUX" + good.substr(4), "no GGUF magic");
  expect_refused(p, "GGUF" + u32(1) + good.substr(8), "version 1");
  expect_refused(p, "GGUF" + u32(4) + good.substr(8), "version 4");
  expect_refused(p, "GGUF" + u32(0x03000000) + good.substr(8), "big-endian");
  // Every truncation of the header and of the data.
  for (std::size_t n = 0; n < good.size(); ++n) {
    write_file(p, good.substr(0, n));
    EXPECT_THROW(GgufFile f(p), FormatError) << n;
  }
  // Counts and lengths far beyond the file.
  expect_refused(p, gguf_bytes({one}, {}, {.tensor_count = std::uint64_t{1} << 62}), "tensor count");
  expect_refused(p, gguf_bytes({one}, {}, {.kv_count = ~std::uint64_t{0}}), "metadata count");
  expect_refused(p, gguf_bytes({one}, {{"s", 8, u64(std::uint64_t{1} << 61) + "ab"}}), "metadata");
  expect_refused(p, gguf_bytes({one}, {{"key", 9, u32(0) + u64(std::uint64_t{1} << 62)}}), "array length");
  expect_refused(p, gguf_bytes({one}, {{"key", 9, u32(8) + u64(std::uint64_t{1} << 60)}}), "array length");
  expect_refused(p, gguf_bytes({one}, {{"key", 9, u32(9) + u64(std::uint64_t{1} << 60)}}), "array length");
  expect_refused(p, gguf_bytes({one}, {{"key", 9, u32(77) + u64(1)}}), "unknown array element type");
  expect_refused(p, gguf_bytes({one}, {{"key", 13, ""}}), "unknown value type");
  std::string deep;
  for (int i = 0; i < 6; ++i) deep += u32(9) + u64(1);
  expect_refused(p, gguf_bytes({one}, {{"key", 9, deep + u32(4) + u64(0)}}), "nested deeper");
  expect_refused(p, gguf_bytes({one}, {kv_string("k", "a"), kv_string("k", "b")}), "duplicate metadata key");
  expect_refused(p, gguf_bytes({one}, {kv_u32("general.alignment", 48)}), "power of two");
  expect_refused(p, gguf_bytes({one}, {kv_u32("general.alignment", 0)}), "power of two");
  expect_refused(p, gguf_bytes({one}, {kv_string("general.alignment", "32")}), "not a uint32");
  // Tensor infos.
  expect_refused(p, gguf_bytes({{std::string(2000, 'n'), {1, 32}, ggml::kQ8_0, q8}}), "string length");
  expect_refused(p, gguf_bytes({{"", {1, 32}, ggml::kQ8_0, q8}}), "empty tensor name");
  expect_refused(p, gguf_bytes({one, {"w", {1, 32}, ggml::kQ8_0, q8}}), "duplicate tensor name");
  expect_refused(p, gguf_bytes({{"w", {}, ggml::kQ8_0, q8}}), "rank 0");
  expect_refused(p, gguf_bytes({{"w", {1, 1, 1, 1, 32}, ggml::kQ8_0, q8}}), "rank 5");
  expect_refused(p, gguf_bytes({{"w", {0, 32}, ggml::kQ8_0, q8}}), "dimension 0");
  expect_refused(p, gguf_bytes({{"w", {-1, 32}, ggml::kQ8_0, q8}}), "out of range");
  expect_refused(p, gguf_bytes({{"w", {std::int64_t{1} << 40, std::int64_t{1} << 40}, ggml::kF32, q8}}), "element count overflows");
  expect_refused(p, gguf_bytes({{"w", {std::int64_t{1} << 62, 1}, ggml::kF32, q8}}), "byte size overflows");
  expect_refused(p, gguf_bytes({{"w", {1, 33}, ggml::kQ8_0, q8}}), "not a multiple of the Q8_0 block");
  expect_refused(p, gguf_bytes({{"w", {1, 32}, 99, q8}}), "unknown ggml type 99");
  expect_refused(p, gguf_bytes({{"w", {1, 32}, ggml::kQ8_0, q8, 16}}), "alignment");
  expect_refused(p, gguf_bytes({{"w", {1024, 32}, ggml::kQ8_0, q8}}), "past the data section");
  expect_refused(p, gguf_bytes({{"w", {1, 32}, ggml::kQ8_0, {}, std::uint64_t{1} << 62}}), "past the data section");
  expect_refused(p, gguf_bytes({{"w", {1, 32}, ggml::kQ8_0, q8}, {"v", {1, 32}, ggml::kQ8_0, q8, 0}}), "overlaps");
  expect_refused(p, gguf_bytes({{"w", {2, 32}, ggml::kQ8_0, random_rows(ggml::kQ8_0, 2, 32, 3)},
                                {"v", {1, 32}, ggml::kQ8_0, q8, 32}}),
                 "overlaps");
  // A header longer than the limit, in a sparse file: refused at the limit, never read in full.
  {
    std::string big = "GGUF" + u32(3) + u64(0) + u64(1) + str("k") + u32(8) + u64(mi::weights::kMaxGgufHeaderBytes);
    write_file(p, big);
    fs::resize_file(p, mi::weights::kMaxGgufHeaderBytes + 4096);
    EXPECT_THROW(GgufFile f(p), FormatError);
  }
}

TEST(GgufFile, RefusesWhatIsNotARegularFile) {
  mi::testing::TempDir dir("gguf_dir");
  EXPECT_THROW(GgufFile f(dir.path()), mi::weights::IoError);
  EXPECT_THROW(GgufFile f(dir.path() / "missing.gguf"), mi::weights::IoError);
}

// ---- The DiT reader and the converter on a tiny DiT -----------------------------------------------------------

constexpr std::int64_t kD = 256;

struct TinyGguf {
  std::vector<GT> ts;
};

// Two blocks: block 0 fused qkv Q8_0, o Q4_K, ffn_up_gate Q5_K, ffn_down Q6_K; block 1 separate q/k/v Q4_0, the rest
// BF16 / F16; norms F32, an F16 bias, a Q8_0 text_embedder.proj (a kept linear stored quantized), rope.bands F32.
TinyGguf tiny_gguf(const std::string& prefix = "") {
  TinyGguf d;
  std::uint32_t seed = 10;
  const auto add = [&](const std::string& name, std::vector<std::int64_t> shape, std::uint32_t type) {
    const std::int64_t k = shape.back(), rows = shape.size() == 2 ? shape[0] : 1;
    d.ts.push_back({prefix + name, std::move(shape), type, random_rows(type, rows, k, seed++)});
  };
  add("layers.0.proj.qkv.weight", {3 * kD, kD}, ggml::kQ8_0);
  add("layers.0.proj.o.weight", {kD, kD}, ggml::kQ4_K);
  add("layers.0.proj.ffn_up_gate.weight", {2 * kD, kD}, ggml::kQ5_K);
  add("layers.0.proj.ffn_down.weight", {kD, 2 * kD}, ggml::kQ6_K);
  for (const char* m : {"q", "k", "v"}) add(std::string("layers.1.proj.") + m + ".weight", {kD, kD}, ggml::kQ4_0);
  add("layers.1.proj.o.weight", {kD, kD}, ggml::kBF16);
  add("layers.1.proj.ffn_up_gate.weight", {2 * kD, kD}, ggml::kF16);
  add("layers.1.proj.ffn_down.weight", {kD, 2 * kD}, ggml::kF32);
  add("layers.0.norm1.weight", {kD}, ggml::kF32);
  add("layers.1.norm1.weight", {kD}, ggml::kF32);
  add("text_embedder.proj.weight", {kD, kD}, ggml::kQ8_0);
  add("text_embedder.proj.bias", {kD}, ggml::kF16);
  add("rope.bands", {16}, ggml::kF32);
  return d;
}

const GT& find(const TinyGguf& d, const char* name) {
  for (const GT& t : d.ts) {
    if (t.name == name) return t;
  }
  throw std::runtime_error(std::string("no ") + name);
}

// A tensor of a converted file.
std::vector<std::byte> tensor_bytes(const fs::path& file, const std::string& name) {
  const mi::weights::SafetensorsFile f(file);
  const auto* t = f.find(name);
  if (t == nullptr) throw std::runtime_error("no " + name + " in " + file.string());
  std::vector<std::byte> b(t->nbytes);
  mi::weights::FileReader(file).read_at(t->offset, b.data(), b.size());
  return b;
}

TEST(GgufDit, ReaderMapsNamesTypesAndFusedSlices) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("gguf_reader");
  const TinyGguf d = tiny_gguf("model.diffusion_model.");
  write_file(dir.path() / "t.gguf", gguf_bytes(d.ts));
  const auto m = mi::quant::open_quant_model(dir.path() / "t.gguf");
  EXPECT_EQ(m.kind, mi::quant::ModelKind::kDiT);
  ASSERT_EQ(m.linears.size(), 12u);
  for (const auto& l : m.linears) {
    EXPECT_EQ(l.encoding, mi::quant::Encoding::kGguf) << l.name;
    EXPECT_FALSE(l.spec.quantized()) << l.name;
  }
  const auto* q = m.find_linear("layers.0.proj.q");
  const auto* v = m.find_linear("layers.0.proj.v");
  ASSERT_NE(q, nullptr);
  ASSERT_NE(v, nullptr);
  EXPECT_TRUE(v->fused_slice);
  EXPECT_EQ(v->weight.nbytes, static_cast<std::uint64_t>(kD) * mi::weights::gguf_row_bytes(ggml::kQ8_0, kD));
  EXPECT_EQ(v->weight.offset, q->weight.offset + 2 * v->weight.nbytes);
  EXPECT_EQ(m.gguf_types.at(q->weight.name), ggml::kQ8_0);
  // dequantize_linear_rows reads a slice's rows.
  std::vector<float> row(static_cast<std::size_t>(kD));
  mi::quant::dequantize_linear_rows(m, *v, 3, 1, row.data());
  const GT& qkv = find(d, "model.diffusion_model.layers.0.proj.qkv.weight");
  const auto all = dequant(ggml::kQ8_0, qkv.data, 3 * kD, kD);
  EXPECT_EQ(std::memcmp(row.data(), all.data() + (2 * kD + 3) * kD, static_cast<std::size_t>(kD) * 4), 0);
  // Kept tensors under their canonical names, typed as they are written.
  const auto* bias = m.find_tensor("text_embedder.proj.bias");
  ASSERT_NE(bias, nullptr);
  EXPECT_EQ(bias->dtype, DType::kBF16);
  EXPECT_EQ(m.find_tensor("rope.bands")->dtype, DType::kF32);
  // A GGUF model is for the converter only.
  EXPECT_THROW(mi::quant::to_model_index(m), mi::quant::QuantError);
}

TEST(GgufDit, ConvertsToInt8ConvRotAndBf16) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("gguf_convert");
  const TinyGguf d = tiny_gguf();
  write_file(dir.path() / "t.gguf", gguf_bytes(d.ts));
  mi::quant::ConvertOptions o;
  o.threads = 3;
  o.metadata = {{"mossinfer_gguf_key", "abc"}};
  const fs::path out = dir.path() / "int8.safetensors";
  const auto st = mi::quant::convert(dir.path() / "t.gguf", out, o);
  EXPECT_EQ(st.quantized, 12);
  const mi::weights::SafetensorsFile f(out);
  EXPECT_EQ(f.metadata().at("mossinfer_gguf_key"), "abc");
  EXPECT_NE(f.metadata().at("mossinfer_converter").find("GGUF"), std::string::npos);
  // Each GEMM: the reference dequantization, quantized by the engine's quantizer.
  const auto check = [&](const std::string& module, const char* source, std::int64_t row0, std::int64_t n, std::int64_t k) {
    SCOPED_TRACE(module);
    const GT& t = find(d, source);
    const std::int64_t rows = t.shape[0];
    const auto all = dequant(t.type, t.data, rows, k);
    std::vector<float> w(all.begin() + row0 * k, all.begin() + (row0 + n) * k);
    std::vector<std::int8_t> q(static_cast<std::size_t>(n * k));
    std::vector<float> s(static_cast<std::size_t>(n));
    ASSERT_TRUE(mi::quant::quantize_convrot_rows(w.data(), n, k, 256, q.data(), s.data()));
    EXPECT_EQ(tensor_bytes(out, module + ".weight"), bytes_of(q));
    EXPECT_EQ(tensor_bytes(out, module + ".weight_scale"), bytes_of(s));
  };
  check("layers.0.proj.q", "layers.0.proj.qkv.weight", 0, kD, kD);
  check("layers.0.proj.v", "layers.0.proj.qkv.weight", 2 * kD, kD, kD);
  check("layers.0.proj.o", "layers.0.proj.o.weight", 0, kD, kD);
  check("layers.0.proj.ffn_up_gate", "layers.0.proj.ffn_up_gate.weight", 0, 2 * kD, kD);
  check("layers.0.proj.ffn_down", "layers.0.proj.ffn_down.weight", 0, kD, 2 * kD);
  check("layers.1.proj.k", "layers.1.proj.k.weight", 0, kD, kD);
  check("layers.1.proj.o", "layers.1.proj.o.weight", 0, kD, kD);
  check("layers.1.proj.ffn_up_gate", "layers.1.proj.ffn_up_gate.weight", 0, 2 * kD, kD);
  check("layers.1.proj.ffn_down", "layers.1.proj.ffn_down.weight", 0, kD, 2 * kD);
  // Kept tensors: BF16 rounded to nearest even (a stored-quantized one dequantized first); rope.bands F32.
  const auto bf16_of = [](const std::vector<float>& v) {
    std::vector<std::uint16_t> b(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) b[i] = mi::quant::float_to_bf16(v[i]);
    return bytes_of(b);
  };
  const GT& tp = find(d, "text_embedder.proj.weight");
  EXPECT_EQ(tensor_bytes(out, "text_embedder.proj.weight"), bf16_of(dequant(tp.type, tp.data, kD, kD)));
  const GT& tb = find(d, "text_embedder.proj.bias");
  EXPECT_EQ(tensor_bytes(out, "text_embedder.proj.bias"), bf16_of(dequant(tb.type, tb.data, 1, kD)));
  EXPECT_EQ(tensor_bytes(out, "rope.bands"), find(d, "rope.bands").data);
  EXPECT_EQ(f.find("rope.bands")->dtype, DType::kF32);

  // The bf16 target (tier reference): every GEMM rounded to bf16.
  o.keep_bf16 = {"*"};
  const fs::path out16 = dir.path() / "bf16.safetensors";
  const auto st16 = mi::quant::convert(dir.path() / "t.gguf", out16, o);
  EXPECT_EQ(st16.kept_bf16, 12);
  const GT& o0 = find(d, "layers.0.proj.o.weight");
  EXPECT_EQ(tensor_bytes(out16, "layers.0.proj.o.weight"), bf16_of(dequant(o0.type, o0.data, kD, kD)));
  // The converted file reads as an INT8 / bf16 DiT.
  EXPECT_NO_THROW(mi::quant::to_model_index(mi::quant::open_quant_model(out)));
  EXPECT_NO_THROW(mi::quant::to_model_index(mi::quant::open_quant_model(out16)));
  // No NVFP4 from GGUF.
  mi::quant::ConvertOptions fp4;
  fp4.nvfp4 = mi::quant::Nvfp4Options{};
  fp4.nvfp4->recipe = mi::quant::Nvfp4Recipe::kRtn;
  fp4.nvfp4->keep_name = "none";
  EXPECT_THROW(mi::quant::convert(dir.path() / "t.gguf", dir.path() / "fp4.safetensors", fp4), mi::quant::QuantError);
}

// A BF16 GGUF converts to exactly the bytes the safetensors bf16 file of the same weights gives.
TEST(GgufDit, Bf16GgufMatchesSafetensorsConversion) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("gguf_bf16");
  std::vector<GT> g;
  std::string st_header = "{";
  std::string st_data;
  std::uint32_t seed = 50;
  const auto add = [&](const std::string& name, std::vector<std::int64_t> shape) {
    const std::int64_t k = shape.back(), rows = shape.size() == 2 ? shape[0] : 1;
    const auto data = random_rows(ggml::kBF16, rows, k, seed++);
    g.push_back({name, shape, ggml::kBF16, data});
    st_header += std::string(st_header.size() > 1 ? "," : "") + "\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (std::size_t i = 0; i < shape.size(); ++i) st_header += (i ? "," : "") + std::to_string(shape[i]);
    st_header += "],\"data_offsets\":[" + std::to_string(st_data.size()) + "," + std::to_string(st_data.size() + data.size()) + "]}";
    st_data.append(reinterpret_cast<const char*>(data.data()), data.size());
  };
  for (const char* m : {"q", "k", "v", "o", "ffn_down"}) add(std::string("layers.0.proj.") + m + ".weight", {kD, kD});
  add("layers.0.proj.ffn_up_gate.weight", {2 * kD, kD});
  add("layers.0.norm1.weight", {kD});
  st_header += "}";
  st_header.append((8 - st_header.size() % 8) % 8, ' ');
  write_file(dir.path() / "t.gguf", gguf_bytes(g));
  write_file(dir.path() / "t.safetensors", u64(st_header.size()) + st_header + st_data);
  mi::quant::ConvertOptions o;
  mi::quant::convert(dir.path() / "t.gguf", dir.path() / "a.safetensors", o);
  mi::quant::convert(dir.path() / "t.safetensors", dir.path() / "b.safetensors", o);
  const mi::weights::SafetensorsFile a(dir.path() / "a.safetensors"), b(dir.path() / "b.safetensors");
  ASSERT_EQ(a.tensors().size(), b.tensors().size());
  for (const auto& t : a.tensors()) {
    SCOPED_TRACE(t.name);
    EXPECT_EQ(tensor_bytes(dir.path() / "a.safetensors", t.name), tensor_bytes(dir.path() / "b.safetensors", t.name));
  }
}

TEST(GgufDit, RefusesUnknownNamesShapesAndTypes) {
  mi::testing::TempDir dir("gguf_refuse");
  const fs::path p = dir.path() / "r.gguf";
  const auto expect = [&](const std::vector<GT>& ts, const std::string& what) {
    SCOPED_TRACE(what);
    write_file(p, gguf_bytes(ts));
    try {
      mi::quant::open_quant_model(p);
      ADD_FAILURE() << "accepted";
    } catch (const mi::quant::QuantError& e) {
      EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
    }
  };
  // llama.cpp-style names, a wrong shape, a type below 4 bits and a missing tensor (MOSS-Image shapes: headers only,
  // the data is a sparse tail the reader never touches).
  expect({{"blk.0.attn_q.weight", {32, 32}, ggml::kQ8_0, random_rows(ggml::kQ8_0, 32, 32, 1)}}, "not MOSS-Image DiT tensors");
  expect({{"layers.0.proj.o.weight", {64, 64}, ggml::kQ8_0, random_rows(ggml::kQ8_0, 64, 64, 1)}}, "of another shape");
  {
    const std::uint64_t q3 = mi::weights::gguf_row_bytes(11, 3072) * 3072;
    write_file(p, gguf_bytes({{"layers.0.proj.o.weight", {3072, 3072}, 11, std::vector<std::byte>(16)}}));
    fs::resize_file(p, fs::file_size(p) + q3);
    try {
      mi::quant::open_quant_model(p);
      ADD_FAILURE() << "accepted";
    } catch (const mi::quant::QuantError& e) {
      EXPECT_NE(std::string(e.what()).find("Q3_K"), std::string::npos) << e.what();
      EXPECT_NE(std::string(e.what()).find("types the engine does not read"), std::string::npos) << e.what();
    }
  }
  expect({{"layers.0.norm1.weight", {3072}, ggml::kF32, std::vector<std::byte>(3072 * 4)}}, "missing MOSS-Image DiT tensors");
  // A GGUF text encoder is refused as such.
  write_file(p, gguf_bytes({{"layers.0.norm1.weight", {3072}, ggml::kF32, std::vector<std::byte>(3072 * 4)}}));
  EXPECT_THROW(mi::quant::open_quant_model(p, mi::quant::ModelKind::kTextEncoder), mi::quant::QuantError);
  EXPECT_THROW(mi::weights::check_supported_text_encoder_file(p), mi::weights::UnsupportedModelError);
}

// ---- A full-shape MOSS-Image GGUF (sparse: headers are read, data never) --------------------------------------

// Every tensor of the MOSS-Image DiT under the official names, fused or separate q/k/v, block GEMMs in `gemm_type`,
// the rest F32 (norms, rope.bands) or BF16; the data section is a sparse tail of zeros.
void write_moss_gguf(const fs::path& p, std::uint32_t gemm_type, bool fused, const std::vector<KV>& kvs = {},
                     const std::string& rename_from = "", const std::string& rename_to = "") {
  std::vector<GT> ts;
  const auto add = [&](std::string name, std::vector<std::int64_t> shape, std::uint32_t type) {
    if (!rename_from.empty() && name == rename_from) name = rename_to;
    ts.push_back({std::move(name), std::move(shape), type, {}, std::nullopt});
  };
  for (const auto& l : mi::weights::moss_dit_linears()) {
    if (l.block >= 0) {
      if (fused && (l.gemm == 1 || l.gemm == 2)) continue;
      const bool qkv = fused && l.gemm == 0;
      add("layers." + std::to_string(l.block) + ".proj." + (qkv ? "qkv" : l.name.substr(l.name.rfind('.') + 1)) + ".weight",
          {qkv ? 3 * l.n : l.n, l.k}, gemm_type);
    } else {
      add(l.name + ".weight", {l.n, l.k}, ggml::kBF16);
      if (l.has_bias) add(l.name + ".bias", {l.n}, ggml::kF32);
    }
  }
  const auto norms = [&](const std::string& pre) {
    for (const char* n : {"norm1", "norm2"}) add(pre + n + ".weight", {3072}, ggml::kF32);
    for (const char* n : {"q_norm", "k_norm"}) add(pre + n + ".weight", {128}, ggml::kF32);
  };
  for (int b = 0; b < 42; ++b) norms("layers." + std::to_string(b) + ".");
  for (int b = 0; b < 2; ++b) norms("text_embedder.blocks." + std::to_string(b) + ".");
  add("final_norm_video.weight", {3072}, ggml::kF32);
  add("rope.bands", {16}, ggml::kF32);
  // Offsets laid out as the writer would, without materializing the data.
  std::uint64_t cursor = 0;
  for (GT& t : ts) {
    t.offset = cursor;
    const std::int64_t k = t.shape.back();
    std::int64_t rows = 1;
    for (std::size_t i = 0; i + 1 < t.shape.size(); ++i) rows *= t.shape[i];
    cursor = (cursor + static_cast<std::uint64_t>(rows) * mi::weights::gguf_row_bytes(t.type, k) + 31) / 32 * 32;
  }
  const std::string header = gguf_bytes(ts, kvs);  // empty data: the header, padded
  write_file(p, header);
  fs::resize_file(p, header.size() + cursor);
}

TEST(GgufDit, FullShapeFilesCheckedFromTheHeader) {
  mi::testing::TempDir dir("gguf_moss");
  const fs::path q4 = dir.path() / "q4k.gguf", q8 = dir.path() / "q8.gguf";
  write_moss_gguf(q4, ggml::kQ4_K, false, {kv_string("general.architecture", "moss_image")});
  write_moss_gguf(q8, ggml::kQ8_0, true);
  EXPECT_NO_THROW(mi::weights::check_supported_dit_file(q4));
  const auto m4 = mi::quant::open_quant_model(q4);
  EXPECT_EQ(m4.linears.size(), 252u);
  EXPECT_EQ(m4.tensors.size(), 379u - 168u);
  const auto i4 = mi::pipeline::inspect_gguf_dit(q4);
  EXPECT_EQ(i4.types, "Q4_K");
  EXPECT_FALSE(i4.reference);
  const auto i8 = mi::pipeline::inspect_gguf_dit(q8);
  EXPECT_EQ(i8.types, "Q8_0");
  EXPECT_TRUE(i8.reference);
  EXPECT_EQ(i8.tensors, 379);

  // Targets per tier, and what the file runs as.
  using mi::planner::Tier;
  EXPECT_EQ(mi::pipeline::gguf_target(i4, Tier::kHigh), mi::quant::QuantSpec::int8_convrot(256));
  EXPECT_THROW(mi::pipeline::gguf_target(i4, Tier::kReference), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::gguf_target(i4, Tier::kTurbo), std::invalid_argument);
  EXPECT_EQ(mi::pipeline::gguf_target(i8, Tier::kReference), mi::quant::QuantSpec::bf16());
  EXPECT_EQ(mi::pipeline::gguf_runs_as(i4, mi::quant::QuantSpec::int8_convrot(256)), "GGUF Q4_K, runs as INT8 ConvRot");
  EXPECT_EQ(mi::pipeline::gguf_runs_as(i8, mi::quant::QuantSpec::bf16()), "GGUF Q8_0, runs as bf16");
  EXPECT_THROW(mi::pipeline::gguf_converted_file(q4, i4, mi::quant::QuantSpec::int8_convrot(256), {}), std::invalid_argument);

  // Inspection: a DiT, loadable, its tiers and the format it runs as.
  {
    const auto r = mi::api::inspect_file(q4);
    EXPECT_EQ(r.info.kind, MI_FILE_KIND_DIT);
    EXPECT_EQ(r.info.loadable, 1);
    EXPECT_EQ(r.info.precision, MI_PRECISION_INT8);
    EXPECT_EQ(r.info.tensor_count, 463);
    EXPECT_EQ(r.info.tiers, 1u << MI_TIER_HIGH);
    EXPECT_EQ(r.quantization, "GGUF Q4_K, runs as INT8 ConvRot");
    EXPECT_NE(r.tier_reasons[MI_TIER_HIGH].find("(GGUF Q4_K, runs as INT8 ConvRot)"), std::string::npos) << r.tier_reasons[MI_TIER_HIGH];
    EXPECT_NE(r.tier_reasons[MI_TIER_REFERENCE].find("Q8_0, BF16 or F16"), std::string::npos) << r.tier_reasons[MI_TIER_REFERENCE];
    EXPECT_NE(r.tier_reasons[MI_TIER_TURBO].find("use tier high"), std::string::npos) << r.tier_reasons[MI_TIER_TURBO];
  }
  {
    const auto r = mi::api::inspect_file(q8);
    EXPECT_EQ(r.info.tiers, (1u << MI_TIER_HIGH) | (1u << MI_TIER_REFERENCE));
    EXPECT_EQ(r.quantization, "GGUF Q8_0, runs as INT8 ConvRot (tier high) or bf16 (tier reference)");
    EXPECT_NE(r.tier_reasons[MI_TIER_REFERENCE].find("(GGUF Q8_0, runs as bf16)"), std::string::npos) << r.tier_reasons[MI_TIER_REFERENCE];
  }

  // Refused: a type below 4 bits (a DiT, not loadable), another architecture and other names (no role).
  const fs::path q3 = dir.path() / "q3k.gguf";
  write_moss_gguf(q3, 11, true);
  {
    const auto r = mi::api::inspect_file(q3);
    EXPECT_EQ(r.info.kind, MI_FILE_KIND_DIT);
    EXPECT_EQ(r.info.loadable, 0);
    EXPECT_NE(r.note.find("Q3_K (168 tensors"), std::string::npos) << r.note;
    EXPECT_EQ(r.info.tiers, 0u);
  }
  const fs::path flux = dir.path() / "flux.gguf";
  write_moss_gguf(flux, ggml::kQ8_0, true, {kv_string("general.architecture", "flux")});
  EXPECT_THROW(mi::weights::check_supported_dit_file(flux), mi::weights::UnsupportedModelError);
  EXPECT_EQ(mi::api::inspect_file(flux).info.kind, MI_FILE_KIND_UNKNOWN);
  const fs::path renamed = dir.path() / "renamed.gguf";
  write_moss_gguf(renamed, ggml::kQ8_0, true, {}, "layers.7.proj.o.weight", "layers.7.attn_output.weight");
  EXPECT_NO_THROW(mi::weights::check_supported_dit_file(renamed));  // identified as MOSS-Image ...
  {
    const auto r = mi::api::inspect_file(renamed);  // ... whose reader names the tensor it does not know
    EXPECT_EQ(r.info.loadable, 0);
    EXPECT_NE(r.note.find("'layers.7.attn_output.weight'"), std::string::npos) << r.note;
  }
}

// ---- The converted-file cache -------------------------------------------------------------------------------------

TEST(GgufDit, ConvertedFileCache) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("gguf_cache");
  const fs::path src = dir.path() / "m.gguf", cache = dir.path() / "cache";
  write_file(src, gguf_bytes(tiny_gguf().ts));
  mi::pipeline::GgufDitInfo info;
  info.types = "Q8_0 + Q4_0";
  const auto int8 = mi::quant::QuantSpec::int8_convrot(256);
  bool converted = false;
  const fs::path a = mi::pipeline::gguf_converted_file(src, info, int8, cache, &converted);
  EXPECT_TRUE(converted);
  EXPECT_EQ(a.parent_path(), cache / "gguf");
  EXPECT_NE(a.filename().string().find("m.int8_convrot256."), std::string::npos) << a;
  const mi::weights::SafetensorsFile f(a);
  EXPECT_EQ(f.metadata().at("mossinfer_gguf_types"), "Q8_0 + Q4_0");
  // Found again without converting; a bf16 target is another file.
  EXPECT_EQ(mi::pipeline::gguf_converted_file(src, info, int8, cache, &converted), a);
  EXPECT_FALSE(converted);
  const fs::path b = mi::pipeline::gguf_converted_file(src, info, mi::quant::QuantSpec::bf16(), cache, &converted);
  EXPECT_TRUE(converted);
  EXPECT_NE(a, b);
  // A damaged cache file is converted again.
  fs::resize_file(a, 100);
  EXPECT_EQ(mi::pipeline::gguf_converted_file(src, info, int8, cache, &converted), a);
  EXPECT_TRUE(converted);
  // A new version of the GGUF (another modification time) gets a new file; the old one for that target goes, the
  // other target's stays.
  fs::last_write_time(src, fs::last_write_time(src) + std::chrono::seconds(5));
  const fs::path c = mi::pipeline::gguf_converted_file(src, info, int8, cache, &converted);
  EXPECT_TRUE(converted);
  EXPECT_NE(c, a);
  EXPECT_FALSE(fs::exists(a));
  EXPECT_TRUE(fs::exists(b));
  // A leftover temporary file of a conversion that died long ago is removed; nothing else is left behind.
  const fs::path stale = cache / "gguf" / "x.safetensors.tmp-1-0.partial";
  write_file(stale, "x");
  fs::last_write_time(stale, fs::file_time_type::clock::now() - std::chrono::hours(1));
  fs::last_write_time(src, fs::last_write_time(src) + std::chrono::seconds(5));
  mi::pipeline::gguf_converted_file(src, info, int8, cache, &converted);
  EXPECT_FALSE(fs::exists(stale));
  int files = 0;
  for (const auto& e : fs::directory_iterator(cache / "gguf")) {
    EXPECT_EQ(e.path().extension(), ".safetensors") << e.path();
    ++files;
  }
  EXPECT_EQ(files, 2);
}

}  // namespace
