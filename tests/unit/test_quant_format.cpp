// src/weights/quant: INT8 ConvRot codec, community-format decoders, the quantized-weight reader and the converter.
// Vectors in tests/data/quant come from comfy-kitchen @ dba9fac (the eager code Comfy-Org's files were made with).
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "test_util.h"
#include "util/checked_math.h"
#include "util/npy.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_convert.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
using mi::quant::Encoding;
using mi::quant::QuantFormat;
using mi::quant::QuantSpec;
using mi::weights::DType;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/quant";

fs::path community() {
  const char* env = std::getenv("MOSS_COMMUNITY");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/community");
}
fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
fs::path comfy_dit() { return community() / "comfy-org/moss_image_sft_int8_convrot.safetensors"; }
fs::path comfy_te() { return community() / "comfy-org/qwen3.5_9b_int8_convrot.safetensors"; }
fs::path tsolful() { return community() / "tsolful/moss_image_sft_W4A8_GPTQ.safetensors"; }
fs::path toxic() { return community() / "toxicdog/moss_image_sft_turbo_int8_convrot.safetensors"; }

// A community file is usable once its download finished: no aria2 control file next to it and a readable header.
bool usable(const fs::path& p) {
  if (!fs::is_regular_file(p) || fs::exists(fs::path(p) += ".aria2")) return false;
  try {
    mi::weights::SafetensorsFile f(p);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

template <typename T>
std::vector<T> load(const std::string& name) {
  return mi::npy::load(kData / (name + ".npy")).to_vector<T>();
}
std::vector<std::int64_t> shape_of(const std::string& name) { return mi::npy::load(kData / (name + ".npy")).shape; }

std::uint32_t bits(float f) { return std::bit_cast<std::uint32_t>(f); }

// ---- Synthetic safetensors -----------------------------------------------------------------------------------

struct T {
  std::string name;
  DType dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;
};

template <typename V>
T make(std::string name, DType d, std::vector<std::int64_t> shape, const std::vector<V>& v) {
  T t{std::move(name), d, std::move(shape), {}};
  t.data.resize(v.size() * sizeof(V));
  std::memcpy(t.data.data(), v.data(), t.data.size());
  return t;
}
T make_json(std::string name, const std::string& s) {
  std::vector<std::uint8_t> v(s.begin(), s.end());
  return make(std::move(name), DType::kU8, {static_cast<std::int64_t>(v.size())}, v);
}

// Laid out like the safetensors library writes files (larger dtypes first, header padded to 8 bytes), so every
// tensor is aligned to its element size.
void write_st(const fs::path& path, std::vector<T> ts, const std::string& metadata_json = "") {
  std::stable_sort(ts.begin(), ts.end(), [](const T& a, const T& b) { return mi::weights::dtype_size(a.dtype) > mi::weights::dtype_size(b.dtype); });
  std::string h = "{";
  if (!metadata_json.empty()) h += "\"__metadata__\":" + metadata_json;
  std::uint64_t off = 0;
  for (const T& t : ts) {
    if (h.size() > 1) h += ",";
    h += "\"" + t.name + "\":{\"dtype\":\"" + mi::weights::dtype_name(t.dtype) + "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
    h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.data.size()) + "]}";
    off += t.data.size();
  }
  h += "}";
  h.append((8 - h.size() % 8) % 8, ' ');
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const std::uint64_t len = h.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
  out.write(h.data(), static_cast<std::streamsize>(h.size()));
  for (const T& t : ts) out.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size()));
}

std::vector<std::uint16_t> rand_bf16(std::size_t n, float sigma, std::uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, sigma);
  std::vector<std::uint16_t> v(n);
  for (auto& x : v) x = mi::quant::float_to_bf16(d(g));
  return v;
}
std::vector<float> rand_f32(std::size_t n, float sigma, std::uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, sigma);
  std::vector<float> v(n);
  for (auto& x : v) x = d(g);
  return v;
}

std::vector<std::int8_t> bytes_i8(const std::vector<std::byte>& b) {
  std::vector<std::int8_t> v(b.size());
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}
std::vector<float> bytes_f32(const std::vector<std::byte>& b) {
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

// Largest |a - b| relative to max |b|; *exact counts bit-equal elements.
double max_rel_diff(const std::vector<float>& a, const std::vector<float>& b, std::size_t* exact) {
  double m = 0, scale = 0;
  for (float v : b) scale = std::max(scale, static_cast<double>(std::fabs(v)));
  *exact = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]) / scale);
    *exact += a[i] == b[i];
  }
  return m;
}

// ---- Codec ---------------------------------------------------------------------------------------------------

TEST(QuantCodec, RegularHadamardMatchesReference) {
  for (int g : {16, 64}) {
    const auto ref = load<float>("h" + std::to_string(g));
    for (int i = 0; i < g; ++i) {
      for (int j = 0; j < g; ++j) {
        ASSERT_EQ(static_cast<float>(mi::quant::hadamard_sign(g, i, j)) / std::sqrt(static_cast<float>(g)), ref[static_cast<std::size_t>(i * g + j)]) << g << " " << i << " " << j;
      }
    }
  }
  // H256 is symmetric and its own inverse: rotate then un-rotate a vector of powers of two (exact in fp32).
  std::vector<float> x(256), y(256);
  for (int i = 0; i < 256; ++i) x[static_cast<std::size_t>(i)] = static_cast<float>(i % 7 - 3) * 0.5f;
  mi::quant::convrot_rows(x.data(), 1, 256, 256, y.data());
  for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < i; ++j) ASSERT_EQ(mi::quant::hadamard_sign(256, i, j), mi::quant::hadamard_sign(256, j, i));
  }
  // H . H = I (in double).
  for (int j = 0; j < 256; ++j) {
    double acc = 0;
    for (int i = 0; i < 256; ++i) acc += static_cast<double>(y[static_cast<std::size_t>(i)]) * mi::quant::hadamard_sign(256, i, j) / 16.0;
    ASSERT_NEAR(acc, x[static_cast<std::size_t>(j)], 1e-6);
  }
  EXPECT_FALSE(mi::quant::valid_convrot_group(128));
  EXPECT_FALSE(mi::quant::valid_convrot_group(0));
  EXPECT_FALSE(mi::quant::valid_convrot_group(4096));
  EXPECT_TRUE(mi::quant::valid_convrot_group(4));
  EXPECT_TRUE(mi::quant::valid_convrot_group(1024));
}

TEST(QuantCodec, ScalarConversions) {
  EXPECT_EQ(mi::quant::e4m3_to_float(0x38), 1.0f);
  EXPECT_EQ(mi::quant::e4m3_to_float(0x7E), 448.0f);
  EXPECT_EQ(mi::quant::e4m3_to_float(0x01), std::ldexp(1.0f, -9));
  EXPECT_EQ(mi::quant::e4m3_to_float(0x80 | 0x40), -2.0f);
  EXPECT_TRUE(std::isnan(mi::quant::e4m3_to_float(0x7F)));
  EXPECT_TRUE(std::isnan(mi::quant::e4m3_to_float(0xFF)));
  EXPECT_EQ(mi::quant::float_to_bf16(1.0f), 0x3F80);
  EXPECT_EQ(mi::quant::float_to_bf16(std::bit_cast<float>(0x3F808000u)), 0x3F80);  // tie -> even
  EXPECT_EQ(mi::quant::float_to_bf16(std::bit_cast<float>(0x3F818000u)), 0x3F82);  // tie -> even (up)
  EXPECT_EQ(mi::quant::float_to_bf16(std::bit_cast<float>(0x3F808001u)), 0x3F81);
  EXPECT_EQ(mi::quant::bf16_to_float(0xC000), -2.0f);
}

// Rows of a weight quantized by comfy-kitchen (real SFT rows and synthetic edge rows): bit-exact codes and scales.
TEST(QuantCodec, QuantizeConvRotBitExact) {
  for (const std::string tag : {"syn_bf16", "syn_f32", "syn_g64", "syn_g16", "sft_l0_o", "sft_l7_down", "sft_l41_upgate"}) {
    const auto shape = shape_of("qz_" + tag + "_w");
    const std::int64_t rows = shape[0], k = shape[1];
    const int g = load<std::int32_t>("qz_" + tag + "_g")[0];
    const auto q_ref = load<std::int8_t>("qz_" + tag + "_q");
    const auto s_ref = load<float>("qz_" + tag + "_s");
    std::vector<std::int8_t> q(static_cast<std::size_t>(rows * k));
    std::vector<float> s(static_cast<std::size_t>(rows));
    const bool f32 = mi::npy::load(kData / ("qz_" + tag + "_w.npy")).dtype == "f4";
    const bool ok = f32 ? mi::quant::quantize_convrot_rows(load<float>("qz_" + tag + "_w").data(), rows, k, g, q.data(), s.data())
                        : mi::quant::quantize_convrot_rows_bf16(load<std::uint16_t>("qz_" + tag + "_w").data(), rows, k, g, q.data(), s.data());
    ASSERT_TRUE(ok) << tag;
    std::size_t dq = 0, ds = 0;
    for (std::size_t i = 0; i < q.size(); ++i) dq += q[i] != q_ref[i];
    for (std::size_t i = 0; i < s.size(); ++i) ds += bits(s[i]) != bits(s_ref[i]);
    EXPECT_EQ(dq, 0u) << tag;
    EXPECT_EQ(ds, 0u) << tag;
  }
  // The zero row's scale is the clamp floor.
  const auto s0 = load<float>("qz_syn_bf16_s");
  EXPECT_EQ(s0[0], 1e-30f);
}

TEST(QuantCodec, QuantizeRejectsNonFinite) {
  std::vector<float> w(256, 0.5f);
  w[17] = std::numeric_limits<float>::infinity();
  std::vector<std::int8_t> q(256);
  float s;
  EXPECT_FALSE(mi::quant::quantize_convrot_rows(w.data(), 1, 256, 256, q.data(), &s));
  EXPECT_THROW(mi::quant::quantize_convrot_rows(w.data(), 1, 256, 128, q.data(), &s), std::invalid_argument);
}

TEST(QuantCodec, GroupedW4A8W6A8DecodeBitExact) {
  for (const std::string tag : {"w4_g16", "w4_uniform", "w6_g16", "w6_g32"}) {
    const std::string p = "w4a8_" + tag + "_";
    const auto meta = load<std::int32_t>(p + "meta");
    const auto packed = load<std::int8_t>(p + "packed");
    const auto srel = load<std::uint8_t>(p + "srel");
    const auto ref = load<std::int8_t>(p + "int8");
    const auto ps = shape_of(p + "packed");
    const std::int64_t rows = ps[0], k = ps[1] * 8 / meta[0];
    std::vector<float> cb;
    if (fs::exists(kData / (p + "cb.npy"))) cb = load<float>(p + "cb");
    std::vector<std::int8_t> out(static_cast<std::size_t>(rows * k));
    ASSERT_TRUE(mi::quant::decode_grouped_rows(reinterpret_cast<const std::uint8_t*>(packed.data()), srel.data(), cb.empty() ? nullptr : cb.data(),
                                               rows, k, meta[0], meta[1], out.data()));
    EXPECT_EQ(out, ref) << tag;
  }
}

TEST(QuantCodec, Int4DecodeBitExact) {
  for (const std::string tag : {"syn", "all"}) {
    const auto packed = load<std::int8_t>("w4a4_" + tag + "_packed");
    const auto ref = load<std::int8_t>("w4a4_" + tag + "_int8");
    const auto ps = shape_of("w4a4_" + tag + "_packed");
    std::vector<std::int8_t> out(ref.size());
    mi::quant::decode_int4_rows(reinterpret_cast<const std::uint8_t*>(packed.data()), ps[0], ps[1] * 2, out.data());
    EXPECT_EQ(out, ref) << tag;
  }
}

TEST(QuantCodec, LoraMergeMatchesTorch) {
  const auto w = load<std::uint16_t>("lora_w");
  const auto down = load<float>("lora_down");
  const auto up = load<float>("lora_up");
  const auto params = load<float>("lora_params");
  const auto ref = load<float>("lora_merged");
  const auto ws = shape_of("lora_w");
  const auto ds = shape_of("lora_down");
  const std::int64_t n = ws[0], k = ws[1], rank = ds[0];
  const float coef = mi::quant::lora_coef(params[1], params[0], rank);
  std::vector<float> got;
  std::vector<double> acc(static_cast<std::size_t>(k));
  for (std::int64_t r = 0; r < n; ++r) {
    std::vector<float> row(static_cast<std::size_t>(k));
    for (std::int64_t c = 0; c < k; ++c) row[static_cast<std::size_t>(c)] = mi::quant::bf16_to_float(w[static_cast<std::size_t>(r * k + c)]);
    mi::quant::lora_merge_row(row.data(), k, up.data() + r * rank, down.data(), rank, coef, acc.data());
    got.insert(got.end(), row.begin(), row.end());
  }
  std::size_t exact = 0;
  const double rel = max_rel_diff(got, ref, &exact);
  RecordProperty("lora_exact", static_cast<int>(exact));
  RecordProperty("lora_elements", static_cast<int>(n * k));
  // The delta is summed in double and rounded once; torch's fp32 GEMM rounds every partial sum, so single-ulp
  // differences are common (about a quarter of the elements here) but bounded.
  EXPECT_LT(rel, 1e-6);
  EXPECT_EQ(mi::quant::lora_coef(1.0f, std::nullopt, 16), 1.0f);
}

TEST(QuantCodec, GlobMatch) {
  using mi::quant::glob_match;
  EXPECT_TRUE(glob_match("layers.*.proj.ffn_down", "layers.12.proj.ffn_down"));
  EXPECT_FALSE(glob_match("layers.*.proj.ffn_down", "layers.12.proj.o"));
  EXPECT_TRUE(glob_match("layers.1?.proj.*", "layers.13.proj.q"));
  EXPECT_FALSE(glob_match("layers.1?.proj.*", "layers.1.proj.q"));
  EXPECT_TRUE(glob_match("*", ""));
  EXPECT_TRUE(glob_match("**o", "proj.o"));
}

// ---- Reader on synthetic files -------------------------------------------------------------------------------

constexpr std::int64_t kD = 256;  // synthetic DiT dim (one ConvRot group)

// One quantized module (int8 convrot256) with codes/scales derived from bf16 weights.
void add_int8_module(std::vector<T>& ts, const std::string& m, std::int64_t n, std::int64_t k, std::uint32_t seed, bool comfy_quant = true) {
  const auto w = rand_bf16(static_cast<std::size_t>(n * k), 0.02f, seed);
  std::vector<std::int8_t> q(static_cast<std::size_t>(n * k));
  std::vector<float> s(static_cast<std::size_t>(n));
  mi::quant::quantize_convrot_rows_bf16(w.data(), n, k, 256, q.data(), s.data());
  ts.push_back(make(m + ".weight", DType::kI8, {n, k}, q));
  ts.push_back(make(m + ".weight_scale", DType::kF32, {n, 1}, s));
  if (comfy_quant) ts.push_back(make_json(m + ".comfy_quant", R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})"));
}

TEST(QuantModel, ComfyQuantFusedQkvIsUnfusedByByteRanges) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_fused");
  std::vector<T> ts;
  add_int8_module(ts, "layers.0.proj.qkv", 3 * kD, kD, 1);
  add_int8_module(ts, "layers.0.proj.o", kD, kD, 2);
  ts.push_back(make("layers.0.norm1.weight", DType::kBF16, {kD}, rand_bf16(kD, 1, 3)));
  ts.push_back(make_json("layers.0.comfy_attention.config", R"({"attention": "comfy_kitchen_int8"})"));
  ts.push_back(make("final_norm_video.weight", DType::kBF16, {kD}, rand_bf16(kD, 1, 4)));
  write_st(dir.path() / "dit.safetensors", ts);

  const auto m = mi::quant::open_quant_model(dir.path() / "dit.safetensors");
  EXPECT_EQ(m.kind, mi::quant::ModelKind::kDiT);
  ASSERT_EQ(m.linears.size(), 4u);
  const auto* q = m.find_linear("layers.0.proj.q");
  const auto* k = m.find_linear("layers.0.proj.k");
  const auto* v = m.find_linear("layers.0.proj.v");
  ASSERT_TRUE(q && k && v);
  EXPECT_EQ(q->spec, QuantSpec::int8_convrot(256));
  EXPECT_TRUE(q->fused_slice);
  EXPECT_EQ(q->n, kD);
  EXPECT_EQ(k->weight.offset, q->weight.offset + static_cast<std::uint64_t>(kD * kD));
  EXPECT_EQ(v->scale->offset, q->scale->offset + static_cast<std::uint64_t>(2 * kD * 4));
  EXPECT_EQ(m.excluded.size(), 1u);
  EXPECT_EQ(m.tensors.size(), 2u);

  // Row-sliced decode equals the fused tensor's rows.
  std::vector<std::int8_t> qv(static_cast<std::size_t>(kD * kD));
  std::vector<float> sv(static_cast<std::size_t>(kD));
  mi::quant::decode_rows(m, *v, 0, kD, qv.data(), sv.data());
  const auto fused_w = bytes_i8(ts[0].data);
  const auto fused_s = bytes_f32(ts[1].data);
  EXPECT_TRUE(std::equal(qv.begin(), qv.end(), fused_w.begin() + 2 * kD * kD));
  EXPECT_TRUE(std::equal(sv.begin(), sv.end(), fused_s.begin() + 2 * kD));

  // Engine view through the regular host upload of to_model_index's groups.
  const auto qi = mi::quant::to_model_index(m);
  const auto* block = qi.index.find_group(mi::weights::Component::kTransformer, "block.0");
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(block->tensors.size(), 9u);  // 4 x (weight, scale) + norm1
  ASSERT_EQ(qi.specs.size(), 4u);
  EXPECT_EQ(*qi.find_spec("layers.0.proj.v"), QuantSpec::int8_convrot(256));
  mi::weights::ReaderCache readers(qi.index, false);
  const auto host = mi::weights::load_group_to_host(readers, *block);
  const auto view = mi::quant::view(qi, host.tensors, "layers.0.proj.k", QuantSpec::int8_convrot(256));
  EXPECT_EQ(view.n, kD);
  EXPECT_EQ(view.k, kD);
  EXPECT_EQ(std::memcmp(view.weight, fused_w.data() + kD * kD, static_cast<std::size_t>(kD * kD)), 0);
  EXPECT_EQ(view.scale[3], fused_s[static_cast<std::size_t>(kD + 3)]);
  // M1: the caller's spec must be the file's.
  EXPECT_THROW(mi::quant::view(qi, host.tensors, "layers.0.proj.k", QuantSpec::bf16()), mi::quant::QuantError);
  EXPECT_THROW(mi::quant::view(qi, host.tensors, "layers.0.proj.k", QuantSpec::int8_convrot(64)), mi::quant::QuantError);
  EXPECT_THROW(mi::quant::view(qi, host.tensors, "layers.0.proj.k", QuantSpec::int8_plain()), mi::quant::QuantError);
  EXPECT_THROW(mi::quant::view(qi, host.tensors, "layers.0.norm1", QuantSpec::bf16()), mi::quant::QuantError);  // not a linear
}

TEST(QuantModel, LegacyMetadataAndGroupedFormats) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_legacy");
  std::vector<T> ts;
  // W4A8 module from the vector set (64 x 512, g16, codebook).
  const std::string p = "w4a8_w4_g16_";
  ts.push_back(make("layers.0.proj.o.weight", DType::kI8, shape_of(p + "packed"), load<std::int8_t>(p + "packed")));
  ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, shape_of(p + "srel"), load<std::uint8_t>(p + "srel")));
  ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, shape_of(p + "schan"), load<float>(p + "schan")));
  ts.push_back(make("layers.0.proj.o.weight_codebook", DType::kF32, {16}, load<float>(p + "cb")));
  // W6A8 g32 module.
  const std::string p6 = "w4a8_w6_g32_";
  ts.push_back(make("layers.0.proj.ffn_down.weight", DType::kI8, shape_of(p6 + "packed"), load<std::int8_t>(p6 + "packed")));
  ts.push_back(make("layers.0.proj.ffn_down.weight_s_rel", DType::kF8E4M3, shape_of(p6 + "srel"), load<std::uint8_t>(p6 + "srel")));
  ts.push_back(make("layers.0.proj.ffn_down.weight_s_channel", DType::kF32, shape_of(p6 + "schan"), load<float>(p6 + "schan")));
  // W4A4 module.
  ts.push_back(make("layers.0.proj.q.weight", DType::kI8, shape_of("w4a4_syn_packed"), load<std::int8_t>("w4a4_syn_packed")));
  ts.push_back(make("layers.0.proj.q.weight_scale", DType::kF32, shape_of("w4a4_syn_scale"), load<float>("w4a4_syn_scale")));
  // Plain int8 with one tensorwise scale, and an unquantized bf16 GEMM.
  ts.push_back(make("layers.0.proj.k.weight", DType::kI8, {4, 256}, std::vector<std::int8_t>(1024, 3)));
  ts.push_back(make("layers.0.proj.k.weight_scale", DType::kF32, {}, std::vector<float>{0.5f}));
  ts.push_back(make("layers.0.proj.v.weight", DType::kBF16, {4, 256}, rand_bf16(1024, 0.02f, 9)));
  const std::string md = R"({"_quantization_metadata":"{\"format_version\": \"1.0\", \"layers\": {)"
                         R"(\"layers.0.proj.o\": {\"format\": \"asym_w4a8_int8\", \"group_size\": 16, \"convrot\": true, \"convrot_groupsize\": 256}, )"
                         R"(\"layers.0.proj.ffn_down\": {\"format\": \"w6a8_int8\", \"group_size\": 32, \"convrot\": true, \"convrot_groupsize\": 256}, )"
                         R"(\"layers.0.proj.q\": {\"format\": \"convrot_w4a4\", \"convrot_groupsize\": 256, \"quant_group_size\": 64}, )"
                         R"(\"layers.0.proj.k\": {\"format\": \"int8_tensorwise\"}}}"})";
  write_st(dir.path() / "legacy.safetensors", ts, md);

  const auto m = mi::quant::open_quant_model(dir.path() / "legacy.safetensors");
  ASSERT_EQ(m.linears.size(), 5u);
  const auto* o = m.find_linear("layers.0.proj.o");
  const auto* down = m.find_linear("layers.0.proj.ffn_down");
  const auto* q = m.find_linear("layers.0.proj.q");
  const auto* k = m.find_linear("layers.0.proj.k");
  const auto* v = m.find_linear("layers.0.proj.v");
  EXPECT_EQ(o->encoding, Encoding::kW4A8);
  EXPECT_EQ(o->k, 512);
  EXPECT_EQ(down->encoding, Encoding::kW6A8);
  EXPECT_EQ(down->quant_group, 32);
  EXPECT_EQ(q->encoding, Encoding::kW4A4);
  EXPECT_EQ(k->encoding, Encoding::kScalarScale);
  EXPECT_EQ(k->spec, QuantSpec::int8_plain());
  EXPECT_EQ(v->spec, QuantSpec::bf16());
  EXPECT_THROW(mi::quant::to_model_index(m), mi::quant::QuantError);  // encoded linears need decoding first

  for (const auto& [lin, ref, schan] : {std::tuple{o, "w4a8_w4_g16_int8", "w4a8_w4_g16_schan"}, std::tuple{down, "w4a8_w6_g32_int8", "w4a8_w6_g32_schan"},
                                         std::tuple{q, "w4a4_syn_int8", "w4a4_syn_scale"}}) {
    std::vector<std::int8_t> qq(static_cast<std::size_t>(lin->n * lin->k));
    std::vector<float> ss(static_cast<std::size_t>(lin->n));
    mi::quant::decode_linear(m, *lin, qq.data(), ss.data(), 3);
    EXPECT_EQ(qq, load<std::int8_t>(ref)) << lin->name;
    EXPECT_EQ(ss, load<float>(schan)) << lin->name;
  }
  std::vector<std::int8_t> kq(2 * 256);
  std::vector<float> ks(2);
  mi::quant::decode_rows(m, *k, 1, 2, kq.data(), ks.data());
  EXPECT_EQ(ks, (std::vector<float>{0.5f, 0.5f}));
  EXPECT_EQ(kq[300], 3);
}

TEST(QuantModel, TextEncoderNamesMapAndExclusions) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_te");
  std::vector<T> ts;
  add_int8_module(ts, "model.language_model.layers.0.mlp.up_proj", 256, 256, 1);
  ts.push_back(make("model.language_model.layers.0.linear_attn.in_proj_a.weight", DType::kBF16, {32, 256}, rand_bf16(32 * 256, 0.02f, 2)));
  ts.push_back(make("model.language_model.layers.0.input_layernorm.weight", DType::kBF16, {256}, rand_bf16(256, 1, 3)));
  ts.push_back(make("model.language_model.layers.0.linear_attn.A_log", DType::kF32, {32}, rand_f32(32, 1, 4)));
  add_int8_module(ts, "model.language_model.embed_tokens", 64, 256, 5);
  ts.push_back(make("model.language_model.norm.weight", DType::kBF16, {256}, rand_bf16(256, 1, 6)));
  add_int8_module(ts, "model.visual.blocks.0.attn.qkv", 256, 256, 7);
  add_int8_module(ts, "lm_head", 64, 256, 8);
  ts.push_back(make("mtp.fc.weight", DType::kBF16, {4, 4}, rand_bf16(16, 1, 9)));
  write_st(dir.path() / "te.safetensors", ts);

  const auto m = mi::quant::open_quant_model(dir.path() / "te.safetensors");
  EXPECT_EQ(m.kind, mi::quant::ModelKind::kTextEncoder);
  ASSERT_NE(m.find_linear("language_model.layers.0.mlp.up_proj"), nullptr);
  ASSERT_NE(m.find_linear("language_model.embed_tokens"), nullptr);
  EXPECT_EQ(m.find_linear("language_model.layers.0.linear_attn.in_proj_a")->spec, QuantSpec::bf16());
  ASSERT_NE(m.find_tensor("language_model.layers.0.linear_attn.A_log"), nullptr);
  std::map<std::string, std::uint64_t> ex;
  for (const auto& e : m.excluded) ex[e.prefix] = e.tensors;
  EXPECT_EQ(ex["visual."], 3u);
  EXPECT_EQ(ex["lm_head."], 3u);
  EXPECT_EQ(ex["mtp."], 1u);
  const auto index = mi::quant::to_model_index(m).index;
  const auto* layer = index.find_group(mi::weights::Component::kTextEncoder, "layers.0");
  ASSERT_NE(layer, nullptr);
  EXPECT_EQ(layer->layer_type, "linear_attention");
  EXPECT_NE(index.find_group(mi::weights::Component::kTextEncoder, "embed_tokens"), nullptr);
  EXPECT_NE(index.find_group(mi::weights::Component::kTextEncoder, "final_norm"), nullptr);
}

// Hostile or inconsistent headers are rejected, never misread.
TEST(QuantModel, RejectsMalformedQuantization) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_bad");
  int n = 0;
  auto expect_reject = [&](std::vector<T> ts, const std::string& md, const char* what) {
    const fs::path p = dir.path() / ("bad" + std::to_string(n++) + ".safetensors");
    write_st(p, ts, md);
    EXPECT_ANY_THROW(mi::quant::open_quant_model(p)) << what;
  };
  auto base = [] {
    std::vector<T> ts;
    add_int8_module(ts, "layers.0.proj.o", 8, 256, 1, false);
    return ts;
  };
  auto with = [&](const std::string& json) {
    auto ts = base();
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", json));
    return ts;
  };
  expect_reject(with(R"({"format": "float8_e4m3fn"})"), "", "unsupported format");
  expect_reject(with(R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 128})"), "", "group not a power of 4");
  expect_reject(with(R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 1024})"), "", "group does not divide K");
  expect_reject(with(R"({"format": "int8_tensorwise", "convrot": "yes"})"), "", "non-boolean convrot");
  expect_reject(with(R"({"convrot": true})"), "", "no format");
  expect_reject(with(R"({"format": "int8_tensorwise")"), "", "truncated JSON");
  expect_reject(with(R"([1, 2])"), "", "config not an object");
  {
    auto ts = with(R"({"format": "int8_tensorwise"})");
    ts.push_back(make("layers.0.proj.o.input_scale", DType::kF32, {}, std::vector<float>{1}));
    expect_reject(ts, "", "leftover tensor in a quantized module");
  }
  {
    auto ts = base();
    ts[1] = make("layers.0.proj.o.weight_scale", DType::kF32, {7, 1}, std::vector<float>(7, 1));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "int8_tensorwise"})"));
    expect_reject(ts, "", "scale rows != weight rows");
  }
  {
    auto ts = base();
    ts[0] = make("layers.0.proj.o.weight", DType::kBF16, {8, 128}, rand_bf16(8 * 128, 1, 1));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "int8_tensorwise"})"));
    expect_reject(ts, "", "quantized weight not I8");
  }
  {
    auto ts = with(R"({"format": "int8_tensorwise"})");
    expect_reject(ts, R"({"_quantization_metadata":"{\"layers\": {\"layers.0.proj.o\": {\"format\": \"int8_tensorwise\", \"convrot\": true}}}"})",
                  "conflicting comfy_quant and legacy configs");
  }
  expect_reject(base(), R"({"_quantization_metadata":"{\"layers\": {\"layers.0.proj.x\": {\"format\": \"int8_tensorwise\"}}}"})", "config without weight");
  expect_reject(base(), R"({"_quantization_metadata":"not json"})", "legacy metadata not JSON");
  expect_reject(base(), R"({"_quantization_metadata":"{\"layers\": 3}"})", "legacy layers not an object");
  for (const std::int64_t cols : {128, 96}) {  // 128: not a whole number of 6-bit codes; 96 (K = 128): codebook with 6 bits
    std::vector<T> ts;
    ts.push_back(make("layers.0.proj.o.weight", DType::kI8, {8, cols}, std::vector<std::int8_t>(static_cast<std::size_t>(8 * cols))));
    ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, {8, 8}, std::vector<std::uint8_t>(64, 0x38)));
    ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, {8}, std::vector<float>(8, 1)));
    ts.push_back(make("layers.0.proj.o.weight_codebook", DType::kF32, {16}, std::vector<float>(16, 0)));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "w6a8_int8", "group_size": 16, "convrot_groupsize": 64})"));
    expect_reject(ts, "", cols == 128 ? "packed width vs 6-bit codes" : "codebook with 6-bit codes");
  }
  {
    std::vector<T> ts;  // group scales that do not tile K
    ts.push_back(make("layers.0.proj.o.weight", DType::kI8, {8, 64}, std::vector<std::int8_t>(512)));
    ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, {8, 4}, std::vector<std::uint8_t>(32, 0x38)));
    ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, {8}, std::vector<float>(8, 1)));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 64})"));
    expect_reject(ts, "", "s_rel shape");
  }
  {
    std::vector<T> ts;  // fused qkv whose rows are not a multiple of 3
    add_int8_module(ts, "layers.0.proj.qkv", 8, 256, 1);
    expect_reject(ts, "", "fused qkv rows % 3");
  }
  {
    std::vector<T> ts;  // fused and separate definitions of the same linear
    add_int8_module(ts, "layers.0.proj.qkv", 6, 256, 1);
    add_int8_module(ts, "layers.0.proj.q", 2, 256, 2);
    expect_reject(ts, "", "q defined twice");
  }
  {
    auto ts = base();
    std::vector<std::uint8_t> big(5000, ' ');
    ts.push_back(make("layers.0.proj.o.comfy_quant", DType::kU8, {5000}, big));
    expect_reject(ts, "", "oversized comfy_quant");
  }
  {
    auto ts = base();
    ts.push_back(make("layers.0.proj.o.comfy_quant", DType::kI8, {3}, std::vector<std::int8_t>{'{', '}', ' '}));
    expect_reject(ts, "", "comfy_quant not U8");
  }
  // Wrong shapes of the W4A8 side tensors and of a bias.
  auto w4a8 = [](std::vector<std::int64_t> schan_shape, std::vector<std::int64_t> cb_shape) {
    std::vector<T> ts;
    ts.push_back(make("layers.0.proj.o.weight", DType::kI8, {8, 128}, std::vector<std::int8_t>(8 * 128)));
    ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, {8, 16}, std::vector<std::uint8_t>(128, 0x38)));
    ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, schan_shape,
                      std::vector<float>(static_cast<std::size_t>(mi::util::checked_numel(schan_shape).value()), 1)));
    ts.push_back(make("layers.0.proj.o.weight_codebook", DType::kF32, cb_shape,
                      std::vector<float>(static_cast<std::size_t>(mi::util::checked_numel(cb_shape).value()), 0)));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256})"));
    return ts;
  };
  {
    const fs::path p = dir.path() / "w4a8_ok.safetensors";
    write_st(p, w4a8({8}, {16}));
    EXPECT_NO_THROW(mi::quant::open_quant_model(p));  // the base case is valid
  }
  expect_reject(w4a8({7}, {16}), "", "s_channel rows != N");
  expect_reject(w4a8({8, 2}, {16}), "", "s_channel [N, 2]");
  expect_reject(w4a8({8}, {8}), "", "codebook [8]");
  expect_reject(w4a8({8}, {4, 4}), "", "codebook [4, 4]");
  {
    auto ts = with(R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})");
    ts.push_back(make("layers.0.proj.o.bias", DType::kBF16, {7}, rand_bf16(7, 1, 1)));
    expect_reject(ts, "", "bias rows != N");
  }
  {
    auto ts = with(R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})");
    ts.push_back(make("layers.0.proj.o.bias", DType::kI8, {8}, std::vector<std::int8_t>(8)));
    expect_reject(ts, "", "bias not a float");
  }
  // Malformed W4A4 configs (K = 256 from 128 packed bytes).
  auto w4a4 = [](const std::string& json, std::vector<std::int64_t> scale_shape) {
    std::vector<T> ts;
    ts.push_back(make("layers.0.proj.o.weight", DType::kI8, {8, 128}, std::vector<std::int8_t>(8 * 128)));
    ts.push_back(make("layers.0.proj.o.weight_scale", DType::kF32, scale_shape,
                      std::vector<float>(static_cast<std::size_t>(mi::util::checked_numel(scale_shape).value()), 1)));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", json));
    return ts;
  };
  expect_reject(w4a4(R"({"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 96})", {8}), "", "w4a4 quant_group_size does not divide K");
  expect_reject(w4a4(R"({"format": "convrot_w4a4", "convrot_groupsize": 128})", {8}), "", "w4a4 group not a power of 4");
  expect_reject(w4a4(R"({"format": "convrot_w4a4", "convrot_groupsize": 256, "linear_dtype": "int2"})", {8}), "", "w4a4 unknown linear_dtype");
  expect_reject(w4a4(R"({"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": "64"})", {8}), "", "w4a4 non-integer group");
  expect_reject(w4a4(R"({"format": "convrot_w4a4", "convrot_groupsize": 256})", {}), "", "w4a4 scalar scale");
  // group_size outside [1, 4096], or not fitting the W4A8 layout.
  for (const char* gs : {"0", "-16", "8192", "24", "16.5"}) {
    std::vector<T> ts;
    ts.push_back(make("layers.0.proj.o.weight", DType::kI8, {8, 128}, std::vector<std::int8_t>(8 * 128)));
    ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, {8, 16}, std::vector<std::uint8_t>(128, 0x38)));
    ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, {8}, std::vector<float>(8, 1)));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", std::string(R"({"format": "asym_w4a8_int8", "convrot_groupsize": 256, "group_size": )") + gs + "}"));
    expect_reject(ts, "", (std::string("group_size ") + gs).c_str());
  }
  {
    std::vector<T> ts;  // two prefixes mapping to one canonical name
    add_int8_module(ts, "layers.0.proj.o", 8, 256, 1);
    ts.push_back(make("layers.0.norm1.weight", DType::kBF16, {256}, rand_bf16(256, 1, 2)));
    ts.push_back(make("diffusion_model.layers.0.norm1.weight", DType::kBF16, {256}, rand_bf16(256, 1, 3)));
    expect_reject(ts, "", "prefixed and unprefixed names of one tensor");
  }
  {
    std::vector<T> ts;
    add_int8_module(ts, "model.diffusion_model.layers.0.proj.o", 8, 256, 1);
    add_int8_module(ts, "diffusion_model.layers.0.proj.o", 8, 256, 2);
    expect_reject(ts, "", "two prefixed spellings of one module");
  }
  // NaN group scale: the header is fine, the decode refuses.
  std::vector<T> ts;
  ts.push_back(make("layers.0.proj.o.weight", DType::kI8, {2, 96}, std::vector<std::int8_t>(192)));
  std::vector<std::uint8_t> srel(16, 0x38);
  srel[3] = 0x7F;
  ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, {2, 8}, srel));
  ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, {2}, std::vector<float>(2, 1)));
  ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "w6a8_int8", "group_size": 16, "convrot_groupsize": 64})"));
  write_st(dir.path() / "nan.safetensors", ts);
  const auto m = mi::quant::open_quant_model(dir.path() / "nan.safetensors");
  std::vector<std::int8_t> q(256);
  std::vector<float> s(2);
  EXPECT_THROW(mi::quant::decode_rows(m, m.linears[0], 0, 2, q.data(), s.data()), mi::quant::QuantError);
}

// Stored scales are validated when read: per-row scales finite and > 0, group scales not NaN or negative (0 is valid).
TEST(QuantModel, RejectsInvalidStoredScales) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_scales");
  auto with_scale = [&](const char* tag, float bad) {
    std::vector<T> ts;
    add_int8_module(ts, "layers.0.proj.o", 8, 256, 1);
    std::vector<float> s = bytes_f32(ts[1].data);
    s[5] = bad;
    ts[1] = make("layers.0.proj.o.weight_scale", DType::kF32, {8, 1}, s);
    const fs::path p = dir.path() / (std::string(tag) + ".safetensors");
    write_st(p, ts);
    return p;
  };
  for (const auto& [tag, bad] : {std::pair{"zero", 0.0f}, std::pair{"neg", -1.0f}, std::pair{"nan", std::numeric_limits<float>::quiet_NaN()},
                                 std::pair{"inf", std::numeric_limits<float>::infinity()}}) {
    const fs::path p = with_scale(tag, bad);
    const auto m = mi::quant::open_quant_model(p);  // headers are fine
    std::vector<std::int8_t> q(8 * 256);
    std::vector<float> s(8);
    EXPECT_THROW(mi::quant::decode_rows(m, m.linears[0], 0, 8, q.data(), s.data()), mi::quant::QuantError) << tag;
    EXPECT_NO_THROW(mi::quant::decode_rows(m, m.linears[0], 0, 5, q.data(), s.data())) << tag;  // rows without the bad scale
    EXPECT_THROW(mi::quant::to_model_index(m), mi::quant::QuantError) << tag;
    EXPECT_THROW(mi::quant::convert(p, dir.path() / "out.safetensors", {}), mi::quant::QuantError) << tag;
    EXPECT_FALSE(fs::exists(dir.path() / "out.safetensors"));
    EXPECT_FALSE(fs::exists(dir.path() / "out.safetensors.partial"));
  }
  // W6A8 group scales: 0 decodes to zeros, a negative one (0xB8 = -1.0) is refused.
  for (const std::uint8_t srel3 : {std::uint8_t{0x00}, std::uint8_t{0xB8}}) {
    std::vector<T> ts;
    ts.push_back(make("layers.0.proj.o.weight", DType::kI8, {2, 96}, std::vector<std::int8_t>(192, 0x55)));
    std::vector<std::uint8_t> srel(16, 0x38);
    srel[3] = srel3;
    ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, {2, 8}, srel));
    ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, {2}, std::vector<float>(2, 1)));
    ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "w6a8_int8", "group_size": 16, "convrot_groupsize": 64})"));
    const fs::path p = dir.path() / ("srel" + std::to_string(srel3) + ".safetensors");
    write_st(p, ts);
    const auto m = mi::quant::open_quant_model(p);
    std::vector<std::int8_t> q(256);
    std::vector<float> s(2);
    if (srel3 == 0) {
      ASSERT_NO_THROW(mi::quant::decode_rows(m, m.linears[0], 0, 2, q.data(), s.data()));
      EXPECT_TRUE(std::all_of(q.begin() + 48, q.begin() + 64, [](std::int8_t v) { return v == 0; }));
      EXPECT_NE(q[47], 0);
    } else {
      EXPECT_THROW(mi::quant::decode_rows(m, m.linears[0], 0, 2, q.data(), s.data()), mi::quant::QuantError);
    }
  }
}

// A fused qkv with one tensorwise scale: every slice keeps the whole scalar.
TEST(QuantModel, ScalarScaleFusedQkv) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_scalar_qkv");
  std::vector<std::int8_t> w(6 * 256);
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::int8_t>(static_cast<int>(i % 255) - 127);
  std::vector<T> ts;
  ts.push_back(make("layers.0.proj.qkv.weight", DType::kI8, {6, 256}, w));
  ts.push_back(make("layers.0.proj.qkv.weight_scale", DType::kF32, {}, std::vector<float>{0.25f}));
  ts.push_back(make_json("layers.0.proj.qkv.comfy_quant", R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})"));
  write_st(dir.path() / "dit.safetensors", ts);
  const auto m = mi::quant::open_quant_model(dir.path() / "dit.safetensors");
  ASSERT_EQ(m.linears.size(), 3u);
  for (const char* part : {"q", "k", "v"}) {
    const auto* l = m.find_linear(std::string("layers.0.proj.") + part);
    ASSERT_NE(l, nullptr);
    EXPECT_EQ(l->encoding, Encoding::kScalarScale);
    EXPECT_EQ(l->n, 2);
    EXPECT_EQ(l->scale->numel(), 1);
  }
  const auto* v = m.find_linear("layers.0.proj.v");
  std::vector<std::int8_t> q(2 * 256);
  std::vector<float> s(2);
  mi::quant::decode_rows(m, *v, 0, 2, q.data(), s.data());
  EXPECT_EQ(s, (std::vector<float>{0.25f, 0.25f}));
  EXPECT_TRUE(std::equal(q.begin(), q.end(), w.begin() + 4 * 256));
  EXPECT_THROW(mi::quant::to_model_index(m), mi::quant::QuantError);  // needs decoding first
  // Converted: per-row scales, identical codes.
  const auto st = mi::quant::convert(dir.path() / "dit.safetensors", dir.path() / "out.safetensors", {});
  EXPECT_EQ(st.decoded, 3);
  const auto mo = mi::quant::open_quant_model(dir.path() / "out.safetensors");
  const auto* vo = mo.find_linear("layers.0.proj.v");
  ASSERT_NE(vo, nullptr);
  EXPECT_EQ(vo->encoding, Encoding::kRaw);
  EXPECT_EQ(bytes_i8(mi::quant::read_tensor(mo, vo->weight)), q);
  EXPECT_EQ(bytes_f32(mi::quant::read_tensor(mo, *vo->scale)), s);
}

TEST(QuantModel, TextEncoderEdgeCases) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_te_edge");
  // A malformed config on an excluded module is ignored (its tensors are excluded anyway).
  {
    std::vector<T> ts;
    add_int8_module(ts, "model.language_model.layers.0.mlp.up_proj", 256, 256, 1);
    ts.push_back(make("model.language_model.layers.0.linear_attn.A_log", DType::kF32, {4}, rand_f32(4, 1, 2)));
    ts.push_back(make("model.visual.blocks.0.attn.qkv.weight", DType::kBF16, {4, 4}, rand_bf16(16, 1, 3)));
    ts.push_back(make_json("model.visual.blocks.0.attn.qkv.comfy_quant", "not json {"));
    ts.push_back(make_json("lm_head.comfy_quant", R"({"format": "nonsense"})"));
    const std::string md = R"({"_quantization_metadata":"{\"layers\": {\"model.mtp.fc\": {\"format\": 7}}}"})";
    write_st(dir.path() / "te.safetensors", ts, md);
    const auto m = mi::quant::open_quant_model(dir.path() / "te.safetensors");
    EXPECT_EQ(m.kind, mi::quant::ModelKind::kTextEncoder);
    EXPECT_EQ(m.linears.size(), 1u);
    std::map<std::string, std::uint64_t> ex;
    for (const auto& e : m.excluded) ex[e.prefix] = e.tensors;
    EXPECT_EQ(ex["visual."], 2u);
    EXPECT_EQ(ex["lm_head."], 1u);
    // F32 -> bf16 rounds to nearest even.
    const auto a = mi::quant::read_tensor_bf16(m, *m.find_tensor("language_model.layers.0.linear_attn.A_log"));
    const auto raw = bytes_f32(mi::quant::read_tensor(m, *m.find_tensor("language_model.layers.0.linear_attn.A_log")));
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i], mi::quant::float_to_bf16(raw[i]));
    EXPECT_THROW(mi::quant::read_tensor_bf16(m, m.linears[0].weight), mi::quant::QuantError);  // I8
  }
  // An F32 value that rounds to infinity in bf16 is refused.
  {
    std::vector<T> ts;
    ts.push_back(make("model.language_model.norm.weight", DType::kF32, {2}, std::vector<float>{1.0f, std::numeric_limits<float>::max()}));
    write_st(dir.path() / "big.safetensors", ts);
    const auto m = mi::quant::open_quant_model(dir.path() / "big.safetensors");
    EXPECT_THROW(mi::quant::read_tensor_bf16(m, m.tensors[0]), mi::quant::QuantError);
  }
  // A text-only Qwen3.5 checkpoint gets a clear refusal (it would otherwise be misread as a DiT).
  {
    std::vector<T> ts;
    ts.push_back(make("model.layers.0.mlp.up_proj.weight", DType::kBF16, {4, 4}, rand_bf16(16, 1, 1)));
    ts.push_back(make("model.embed_tokens.weight", DType::kBF16, {4, 4}, rand_bf16(16, 1, 2)));
    write_st(dir.path() / "textonly.safetensors", ts);
    try {
      mi::quant::open_quant_model(dir.path() / "textonly.safetensors");
      ADD_FAILURE() << "text-only TE accepted";
    } catch (const mi::quant::QuantError& e) {
      EXPECT_NE(std::string(e.what()).find("text-only Qwen3.5"), std::string::npos) << e.what();
    }
  }
  // An unroutable text-encoder tensor is a QuantError at open.
  {
    std::vector<T> ts;
    ts.push_back(make("model.language_model.layers.0.input_layernorm.weight", DType::kBF16, {4}, rand_bf16(4, 1, 1)));
    ts.push_back(make("model.something_else.weight", DType::kBF16, {4}, rand_bf16(4, 1, 2)));
    write_st(dir.path() / "stray.safetensors", ts);
    EXPECT_THROW(mi::quant::open_quant_model(dir.path() / "stray.safetensors"), mi::quant::QuantError);
  }
  // A bundle root without the requested component: a domain error, not a filesystem_error.
  {
    std::vector<T> ts;
    add_int8_module(ts, "layers.0.proj.o", 8, 256, 1);
    write_st(dir.path() / "bundle" / "transformer" / "dit.safetensors", ts);
    EXPECT_NO_THROW(mi::quant::open_quant_model(dir.path() / "bundle"));
    EXPECT_THROW(mi::quant::open_quant_model(dir.path() / "bundle", mi::quant::ModelKind::kTextEncoder), mi::quant::QuantError);
  }
}

// view() checks the uploaded tensors' layout, not only the spec.
TEST(QuantModel, ViewRejectsBadLayout) {
  constexpr std::int64_t n = 4, k = 256;
  alignas(16) std::int8_t w[n * k] = {};
  alignas(16) float scale_buf[n + 1] = {1, 1, 1, 1, 1};
  alignas(16) std::uint16_t bias_buf[n + 1] = {};
  mi::quant::QuantIndex qi;
  qi.specs.emplace("m", QuantSpec::int8_convrot(256));
  qi.specs.emplace("b", QuantSpec::bf16());
  auto tensors = [&](void* scale, void* bias, mi::DType bias_dtype, mi::Device scale_dev) {
    mi::weights::TensorMap m;
    m["m.weight"] = mi::Tensor::view(w, mi::DType::kI8, mi::Device::kCPU, {n, k});
    m["m.weight_scale"] = mi::Tensor::view(scale, mi::DType::kF32, scale_dev, {n});
    if (bias != nullptr) m["m.bias"] = mi::Tensor::view(bias, bias_dtype, mi::Device::kCPU, {n});
    return m;
  };
  const auto spec = QuantSpec::int8_convrot(256);
  EXPECT_NO_THROW(mi::quant::view(qi, tensors(scale_buf, bias_buf, mi::DType::kBF16, mi::Device::kCPU), "m", spec));
  EXPECT_THROW(mi::quant::view(qi, tensors(reinterpret_cast<std::byte*>(scale_buf) + 2, nullptr, mi::DType::kBF16, mi::Device::kCPU), "m", spec),
               mi::quant::QuantError);  // misaligned scale
  EXPECT_THROW(mi::quant::view(qi, tensors(scale_buf, reinterpret_cast<std::byte*>(bias_buf) + 1, mi::DType::kBF16, mi::Device::kCPU), "m", spec),
               mi::quant::QuantError);  // misaligned bias
  EXPECT_THROW(mi::quant::view(qi, tensors(scale_buf, bias_buf, mi::DType::kI8, mi::Device::kCPU), "m", spec), mi::quant::QuantError);  // int bias
  EXPECT_THROW(mi::quant::view(qi, tensors(scale_buf, nullptr, mi::DType::kBF16, mi::Device::kCUDA), "m", spec), mi::quant::QuantError);  // devices
  EXPECT_THROW(mi::quant::view(qi, tensors(scale_buf, nullptr, mi::DType::kBF16, mi::Device::kCPU), "m", QuantSpec::int8_convrot(64)),
               mi::quant::QuantError);  // M1 spec mismatch
  // A bf16 weight must be 2-byte aligned.
  alignas(16) std::uint16_t wb[n * k + 1] = {};
  mi::weights::TensorMap mb;
  mb["b.weight"] = mi::Tensor::view(reinterpret_cast<std::byte*>(wb) + 1, mi::DType::kBF16, mi::Device::kCPU, {n, k});
  EXPECT_THROW(mi::quant::view(qi, mb, "b", QuantSpec::bf16()), mi::quant::QuantError);
  mb["b.weight"] = mi::Tensor::view(wb, mi::DType::kBF16, mi::Device::kCPU, {n, k});
  EXPECT_NO_THROW(mi::quant::view(qi, mb, "b", QuantSpec::bf16()));
}

// ---- Converter on a synthetic DiT ----------------------------------------------------------------------------

struct TinyDit {
  std::vector<T> ts;
  std::map<std::string, std::vector<std::uint16_t>> w;  // GEMM weights by module
};

TinyDit tiny_dit() {
  TinyDit d;
  std::uint32_t seed = 100;
  for (int b = 0; b < 2; ++b) {
    const std::string p = "layers." + std::to_string(b) + ".";
    for (const auto& [name, n, k] : {std::tuple{"proj.q", kD, kD}, std::tuple{"proj.k", kD, kD}, std::tuple{"proj.v", kD, kD},
                                     std::tuple{"proj.o", kD, kD}, std::tuple{"proj.ffn_up_gate", 2 * kD, kD}, std::tuple{"proj.ffn_down", kD, 2 * kD}}) {
      auto v = rand_bf16(static_cast<std::size_t>(n * k), 0.02f, seed++);
      d.ts.push_back(make(p + name + ".weight", DType::kBF16, {n, k}, v));
      d.w[p + name] = v;
    }
    d.ts.push_back(make(p + "norm1.weight", DType::kBF16, {kD}, rand_bf16(kD, 1, seed++)));
  }
  d.ts.push_back(make("text_embedder.proj.weight", DType::kBF16, {kD, kD}, rand_bf16(kD * kD, 0.02f, seed++)));
  d.ts.push_back(make("text_embedder.proj.bias", DType::kBF16, {kD}, rand_bf16(kD, 0.02f, seed++)));
  d.ts.push_back(make("rope.bands", DType::kF32, {16}, rand_f32(16, 1, seed++)));
  return d;
}

std::vector<std::int8_t> ref_quant(const std::vector<float>& w, std::int64_t n, std::int64_t k, std::vector<float>& s) {
  std::vector<std::int8_t> q(static_cast<std::size_t>(n * k));
  s.resize(static_cast<std::size_t>(n));
  EXPECT_TRUE(mi::quant::quantize_convrot_rows(w.data(), n, k, 256, q.data(), s.data()));
  return q;
}
std::vector<float> widen(const std::vector<std::uint16_t>& v) {
  std::vector<float> f(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) f[i] = mi::quant::bf16_to_float(v[i]);
  return f;
}

TEST(QuantConvert, SyntheticDitWithLoraKeepAndFusion) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_convert");
  TinyDit d = tiny_dit();
  write_st(dir.path() / "transformer" / "diffusion_pytorch_model.safetensors", d.ts);
  // kohya LoRA on layers.0.proj.q (alpha 4, rank 8), on the fused layers.1.proj.qkv (no alpha), and on a non-GEMM
  // bf16 weight (text_embedder.proj).
  const auto down0 = rand_f32(8 * kD, 0.1f, 1), up0 = rand_f32(kD * 8, 0.1f, 2);
  const auto down1 = rand_f32(4 * kD, 0.1f, 3), up1 = rand_f32(3 * kD * 4, 0.1f, 4);
  const auto down2 = rand_f32(2 * kD, 0.1f, 5), up2 = rand_f32(kD * 2, 0.1f, 6);
  write_st(dir.path() / "lora.safetensors",
           {make("diffusion_model.layers.0.proj.q.lora_down.weight", DType::kF32, {8, kD}, down0),
            make("diffusion_model.layers.0.proj.q.lora_up.weight", DType::kF32, {kD, 8}, up0),
            make("diffusion_model.layers.0.proj.q.alpha", DType::kF32, {}, std::vector<float>{4.0f}),
            make("diffusion_model.layers.1.proj.qkv.lora_A.weight", DType::kF32, {4, kD}, down1),
            make("diffusion_model.layers.1.proj.qkv.lora_B.weight", DType::kF32, {3 * kD, 4}, up1),
            make("diffusion_model.text_embedder.proj.lora_down.weight", DType::kF32, {2, kD}, down2),
            make("diffusion_model.text_embedder.proj.lora_up.weight", DType::kF32, {kD, 2}, up2)});

  mi::quant::ConvertOptions opt;
  opt.attention_config = true;  // byte-compatible with Comfy-Org output, incl. its attention configs
  opt.keep_bf16 = {"layers.1.proj.ffn_*"};
  opt.loras = {{dir.path() / "lora.safetensors", 0.5f}};
  opt.threads = 3;
  const fs::path out = dir.path() / "out.safetensors";
  const auto st = mi::quant::convert(dir.path(), out, opt);
  EXPECT_EQ(st.quantized, 10);
  EXPECT_EQ(st.kept_bf16, 2);
  EXPECT_EQ(st.lora_modules, 5);
  EXPECT_FALSE(fs::exists(fs::path(out) += ".partial"));

  const mi::weights::SafetensorsFile f(out);
  EXPECT_EQ(f.metadata().at("mossinfer_kept_bf16"), R"(["layers.1.proj.ffn_down", "layers.1.proj.ffn_up_gate"])");
  EXPECT_EQ(f.metadata().at("mossinfer_source_xxh64").size(), 16u);
  EXPECT_NE(f.metadata().at("mossinfer_lora").find("\"strength\": 0.5"), std::string::npos);
  // Tensor order: dtype descending (F32, BF16, I8, U8), then name; dense layout checked by SafetensorsFile.
  const auto order_rank = [](DType t) { return t == DType::kF32 ? 3 : t == DType::kBF16 ? 2 : t == DType::kI8 ? 1 : 0; };
  for (std::size_t i = 1; i < f.tensors().size(); ++i) {
    const auto &a = f.tensors()[i - 1], &b = f.tensors()[i];
    EXPECT_TRUE(order_rank(a.dtype) > order_rank(b.dtype) || (order_rank(a.dtype) == order_rank(b.dtype) && a.name < b.name)) << a.name << " / " << b.name;
  }

  const auto m = mi::quant::open_quant_model(out);
  ASSERT_EQ(m.linears.size(), 12u);
  EXPECT_EQ(m.find_linear("layers.1.proj.ffn_down")->spec, QuantSpec::bf16());
  ASSERT_EQ(m.excluded.size(), 1u);
  EXPECT_EQ(m.excluded[0].tensors, 2u);  // layers.N.comfy_attention.config
  // comfy_quant / attention JSON bytes exactly as Comfy-Org writes them.
  for (const auto& t : f.tensors()) {
    if (t.name == "layers.0.proj.o.comfy_quant" || t.name == "layers.1.comfy_attention.config") {
      std::string s(static_cast<std::size_t>(t.nbytes), '\0');
      mi::weights::FileReader(out).read_at(t.offset, s.data(), s.size());
      EXPECT_EQ(s, t.name.ends_with("comfy_quant") ? R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})"
                                                    : R"({"attention": "comfy_kitchen_int8"})");
    }
  }

  auto check_quant = [&](const std::string& mod, const std::vector<float>& w_ref) {
    const auto* l = m.find_linear(mod);
    ASSERT_NE(l, nullptr) << mod;
    std::vector<float> s_ref;
    const auto q_ref = ref_quant(w_ref, l->n, l->k, s_ref);
    std::vector<std::int8_t> q(q_ref.size());
    std::vector<float> s(s_ref.size());
    mi::quant::decode_rows(m, *l, 0, l->n, q.data(), s.data());
    EXPECT_EQ(q, q_ref) << mod;
    EXPECT_EQ(s, s_ref) << mod;
  };
  check_quant("layers.0.proj.o", widen(d.w["layers.0.proj.o"]));
  auto merged = [&](const std::string& mod, const std::vector<float>& up, const std::vector<float>& down, std::int64_t rank, float coef,
                    std::int64_t row_off) {
    std::vector<float> w = widen(d.w[mod]);
    const std::int64_t k = static_cast<std::int64_t>(down.size()) / rank, n = static_cast<std::int64_t>(w.size()) / k;
    std::vector<double> acc(static_cast<std::size_t>(k));
    for (std::int64_t r = 0; r < n; ++r) mi::quant::lora_merge_row(w.data() + r * k, k, up.data() + (row_off + r) * rank, down.data(), rank, coef, acc.data());
    return w;
  };
  check_quant("layers.0.proj.q", merged("layers.0.proj.q", up0, down0, 8, mi::quant::lora_coef(0.5f, 4.0f, 8), 0));
  check_quant("layers.1.proj.k", merged("layers.1.proj.k", up1, down1, 4, 0.5f, kD));
  check_quant("layers.1.proj.v", merged("layers.1.proj.v", up1, down1, 4, 0.5f, 2 * kD));
  // Kept GEMMs are copied untouched; a LoRA on a non-GEMM bf16 weight is merged and rounded to bf16.
  const auto kept = mi::quant::read_tensor(m, m.find_linear("layers.1.proj.ffn_down")->weight);
  EXPECT_EQ(std::memcmp(kept.data(), d.w["layers.1.proj.ffn_down"].data(), kept.size()), 0);
  {
    std::vector<std::uint16_t> te(static_cast<std::size_t>(kD * kD));
    std::memcpy(te.data(), d.ts[d.ts.size() - 3].data.data(), te.size() * 2);
    std::vector<float> w = widen(te);
    std::vector<double> acc(static_cast<std::size_t>(kD));
    for (std::int64_t r = 0; r < kD; ++r) mi::quant::lora_merge_row(w.data() + r * kD, kD, up2.data() + r * 2, down2.data(), 2, 0.5f, acc.data());
    const auto got = mi::quant::read_tensor(m, *m.find_tensor("text_embedder.proj.weight"));
    std::vector<std::uint16_t> g(te.size());
    std::memcpy(g.data(), got.data(), got.size());
    for (std::size_t i = 0; i < g.size(); ++i) ASSERT_EQ(g[i], mi::quant::float_to_bf16(w[i])) << i;
  }

  // Fused output: same codes, qkv rows concatenated; converting the INT8 output again copies it bit for bit.
  opt.fuse_qkv = true;
  const fs::path fused = dir.path() / "fused.safetensors";
  mi::quant::convert(dir.path(), fused, opt);
  const auto mf = mi::quant::open_quant_model(fused);
  for (const char* mod : {"layers.0.proj.q", "layers.0.proj.k", "layers.0.proj.v", "layers.1.proj.o"}) {
    EXPECT_TRUE(mf.find_linear(mod)->fused_slice || std::string(mod).ends_with(".o"));
    EXPECT_EQ(mi::quant::read_tensor(mf, mf.find_linear(mod)->weight), mi::quant::read_tensor(m, m.find_linear(mod)->weight)) << mod;
  }
  mi::quant::ConvertOptions copy;
  copy.attention_config = true;  // byte-compatible with Comfy-Org output, incl. its attention configs
  copy.keep_bf16 = {"layers.1.proj.ffn_*"};  // already bf16 in the input: allowed
  const fs::path again = dir.path() / "again.safetensors";
  const auto st2 = mi::quant::convert(out, again, copy);
  EXPECT_EQ(st2.copied, 10);
  EXPECT_EQ(st2.quantized, 0);
  const mi::weights::SafetensorsFile fa(again);
  for (const auto& t : f.tensors()) {
    const auto* u = fa.find(t.name);
    ASSERT_NE(u, nullptr) << t.name;
    std::vector<std::byte> x(static_cast<std::size_t>(t.nbytes)), y(static_cast<std::size_t>(u->nbytes));
    mi::weights::FileReader(out).read_at(t.offset, x.data(), x.size());
    mi::weights::FileReader(again).read_at(u->offset, y.data(), y.size());
    EXPECT_EQ(x, y) << t.name;
  }

  // Misuse is rejected before anything is written.
  mi::quant::ConvertOptions bad = copy;
  bad.keep_bf16 = {"layers.0.proj.o"};
  EXPECT_THROW(mi::quant::convert(out, dir.path() / "x.safetensors", bad), mi::quant::QuantError);  // keep a quantized GEMM
  bad = copy;
  bad.loras = {{dir.path() / "lora.safetensors", 1.0f}};
  EXPECT_THROW(mi::quant::convert(out, dir.path() / "x.safetensors", bad), mi::quant::QuantError);  // LoRA on quantized input
  write_st(dir.path() / "lora_bad.safetensors", {make("diffusion_model.layers.0.proj.zz.lora_down.weight", DType::kF32, {1, kD}, rand_f32(kD, 1, 1)),
                                                 make("diffusion_model.layers.0.proj.zz.lora_up.weight", DType::kF32, {kD, 1}, rand_f32(kD, 1, 2))});
  bad = mi::quant::ConvertOptions{};
  bad.loras = {{dir.path() / "lora_bad.safetensors", 1.0f}};
  EXPECT_THROW(mi::quant::convert(dir.path(), dir.path() / "x.safetensors", bad), mi::quant::QuantError);  // unmatched module
  EXPECT_FALSE(fs::exists(dir.path() / "x.safetensors"));
}

TEST(QuantConvert, GroupedImportIsLossless) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_import");
  std::vector<T> ts;
  const std::string p = "w4a8_w4_g16_";
  ts.push_back(make("layers.0.proj.o.weight", DType::kI8, shape_of(p + "packed"), load<std::int8_t>(p + "packed")));
  ts.push_back(make("layers.0.proj.o.weight_s_rel", DType::kF8E4M3, shape_of(p + "srel"), load<std::uint8_t>(p + "srel")));
  ts.push_back(make("layers.0.proj.o.weight_s_channel", DType::kF32, shape_of(p + "schan"), load<float>(p + "schan")));
  ts.push_back(make("layers.0.proj.o.weight_codebook", DType::kF32, {16}, load<float>(p + "cb")));
  ts.push_back(make_json("layers.0.proj.o.comfy_quant", R"({"format": "asym_w4a8_int8", "group_size": 16, "convrot": true, "convrot_groupsize": 256})"));
  write_st(dir.path() / "w4a8.safetensors", ts);
  const fs::path out = dir.path() / "int8.safetensors";
  const auto st = mi::quant::convert(dir.path() / "w4a8.safetensors", out, {});
  EXPECT_EQ(st.decoded, 1);
  const auto m = mi::quant::open_quant_model(out);
  const auto* l = m.find_linear("layers.0.proj.o");
  ASSERT_NE(l, nullptr);
  EXPECT_EQ(l->encoding, Encoding::kRaw);
  EXPECT_EQ(bytes_i8(mi::quant::read_tensor(m, l->weight)), load<std::int8_t>(p + "int8"));
  EXPECT_EQ(bytes_f32(mi::quant::read_tensor(m, *l->scale)), load<float>(p + "schan"));
}

// LoRA edge cases: PEFT naming, non-finite coefficients, merges that overflow; a failed conversion leaves nothing.
TEST(QuantConvert, LoraEdgeCasesAndCleanup) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_lora_edge");
  TinyDit d = tiny_dit();
  write_st(dir.path() / "transformer" / "diffusion_pytorch_model.safetensors", d.ts);
  const fs::path out = dir.path() / "out.safetensors";
  auto expect_no_output = [&](const char* what) {
    EXPECT_FALSE(fs::exists(out)) << what;
    EXPECT_FALSE(fs::exists(fs::path(out) += ".partial")) << what;
  };

  // PEFT `base_model.model.` keys merge like kohya ones.
  const auto down = rand_f32(4 * kD, 0.1f, 1), up = rand_f32(kD * 4, 0.1f, 2);
  write_st(dir.path() / "peft.safetensors", {make("base_model.model.layers.0.proj.o.lora_A.weight", DType::kF32, {4, kD}, down),
                                             make("base_model.model.layers.0.proj.o.lora_B.weight", DType::kF32, {kD, 4}, up)});
  mi::quant::ConvertOptions opt;
  opt.loras = {{dir.path() / "peft.safetensors", 1.0f}};
  opt.attention_config = false;
  const auto st = mi::quant::convert(dir.path(), out, opt);
  EXPECT_EQ(st.lora_modules, 1);
  {
    const auto m = mi::quant::open_quant_model(out);
    EXPECT_TRUE(m.excluded.empty());  // attention_config = false: no comfy_attention configs
    const mi::weights::SafetensorsFile f(out);
    for (const auto& t : f.tensors()) EXPECT_EQ(t.name.find("comfy_attention"), std::string::npos) << t.name;
    std::vector<float> w = widen(d.w["layers.0.proj.o"]);
    std::vector<double> acc(static_cast<std::size_t>(kD));
    for (std::int64_t r = 0; r < kD; ++r) mi::quant::lora_merge_row(w.data() + r * kD, kD, up.data() + r * 4, down.data(), 4, 1.0f, acc.data());
    std::vector<float> s_ref;
    const auto q_ref = ref_quant(w, kD, kD, s_ref);
    const auto* l = m.find_linear("layers.0.proj.o");
    EXPECT_EQ(bytes_i8(mi::quant::read_tensor(m, l->weight)), q_ref);
    EXPECT_EQ(bytes_f32(mi::quant::read_tensor(m, *l->scale)), s_ref);
  }
  fs::remove(out);

  // moss-edit-lora (Moss-Image-Edit): bare `<m>.lora_A / .lora_B` keys and the module's alpha from adapter_config
  // (8 at rank 4: coefficient 2), the same bits as the runtime stack's rule.
  write_st(dir.path() / "edit.safetensors", {make("layers.0.proj.o.lora_A", DType::kF32, {4, kD}, down),
                                             make("layers.0.proj.o.lora_B", DType::kF32, {kD, 4}, up)},
           R"({"adapter_config":"{\"format\": \"moss-edit-lora\", \"modules\": {\"layers.0.proj.o\": {\"alpha\": 8}}}"})");
  opt.loras = {{dir.path() / "edit.safetensors", 1.0f}};
  EXPECT_EQ(mi::quant::convert(dir.path(), out, opt).lora_modules, 1);
  {
    const auto m = mi::quant::open_quant_model(out);
    std::vector<float> w = widen(d.w["layers.0.proj.o"]);
    std::vector<double> acc(static_cast<std::size_t>(kD));
    for (std::int64_t r = 0; r < kD; ++r) mi::quant::lora_merge_row(w.data() + r * kD, kD, up.data() + r * 4, down.data(), 4, 2.0f, acc.data());
    std::vector<float> s_ref;
    const auto q_ref = ref_quant(w, kD, kD, s_ref);
    EXPECT_EQ(bytes_i8(mi::quant::read_tensor(m, m.find_linear("layers.0.proj.o")->weight)), q_ref);
  }
  fs::remove(out);

  // Non-finite alpha, a coefficient that overflows, a non-finite strength.
  for (const auto& [tag, alpha, strength] : {std::tuple{"nan_alpha", std::numeric_limits<float>::quiet_NaN(), 1.0f},
                                             std::tuple{"inf_alpha", std::numeric_limits<float>::infinity(), 1.0f},
                                             std::tuple{"huge_coef", 3e38f, 1e3f},
                                             std::tuple{"nan_strength", 4.0f, std::numeric_limits<float>::quiet_NaN()}}) {
    write_st(dir.path() / "alpha.safetensors", {make("diffusion_model.layers.0.proj.o.lora_down.weight", DType::kF32, {4, kD}, down),
                                                make("diffusion_model.layers.0.proj.o.lora_up.weight", DType::kF32, {kD, 4}, up),
                                                make("diffusion_model.layers.0.proj.o.alpha", DType::kF32, {}, std::vector<float>{alpha})});
    mi::quant::ConvertOptions bad;
    bad.loras = {{dir.path() / "alpha.safetensors", strength}};
    EXPECT_ANY_THROW(mi::quant::convert(dir.path(), out, bad)) << tag;
    expect_no_output(tag);
  }
  // Finite factors whose merge overflows, on each path: a quantized GEMM, a kept bf16 GEMM, a non-GEMM bf16 weight.
  // The failure comes mid-write, after the partial file exists: it is removed.
  for (const auto& [tag, module, keep] : {std::tuple{"quantized", "layers.0.proj.o", ""}, std::tuple{"kept", "layers.0.proj.o", "layers.0.proj.o"},
                                          std::tuple{"non_gemm", "text_embedder.proj", ""}}) {
    // The delta (3e38) fits fp32; strength 2 makes the merged weight overflow.
    std::vector<float> big_up(static_cast<std::size_t>(kD), 3e35f);
    write_st(dir.path() / "big.safetensors", {make(std::string("diffusion_model.") + module + ".lora_down.weight", DType::kF32, {1, kD}, std::vector<float>(kD, 1e3f)),
                                              make(std::string("diffusion_model.") + module + ".lora_up.weight", DType::kF32, {kD, 1}, big_up)});
    mi::quant::ConvertOptions bad;
    bad.loras = {{dir.path() / "big.safetensors", 2.0f}};
    if (*keep != '\0') bad.keep_bf16 = {keep};
    EXPECT_THROW(mi::quant::convert(dir.path(), out, bad), mi::quant::QuantError) << tag;
    expect_no_output(tag);
  }
  // The output may not be an input, nor be written into the component directory read.
  EXPECT_THROW(mi::quant::convert(dir.path(), dir.path() / "transformer" / "x.safetensors", {}), mi::quant::QuantError);
  EXPECT_THROW(mi::quant::convert(dir.path(), dir.path() / "transformer" / "diffusion_pytorch_model.safetensors", {}), mi::quant::QuantError);
  EXPECT_THROW(mi::quant::convert(dir.path() / "transformer" / "diffusion_pytorch_model.safetensors",
                                  dir.path() / "transformer" / "." / "diffusion_pytorch_model.safetensors", {}),
               mi::quant::QuantError);
  {
    mi::quant::ConvertOptions bad;
    bad.loras = {{dir.path() / "peft.safetensors", 1.0f}};
    EXPECT_THROW(mi::quant::convert(dir.path(), dir.path() / "peft.safetensors", bad), mi::quant::QuantError);
  }
  EXPECT_FALSE(fs::exists(dir.path() / "transformer" / "x.safetensors"));
  EXPECT_TRUE(fs::exists(dir.path() / "peft.safetensors"));
}

// Output metadata: the source's keys minus the legacy header and stale mossinfer_* keys; the converter line counts
// the written specs. A quantized linear outside `layers.N.` gets no attention config.
TEST(QuantConvert, MetadataAndNonBlockLinears) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("quant_meta");
  std::vector<T> ts;
  add_int8_module(ts, "layers.0.proj.o", kD, kD, 1, false);
  add_int8_module(ts, "text_embedder.proj", kD, kD, 2, false);
  ts.push_back(make("layers.0.proj.q.weight", DType::kBF16, {kD, kD}, rand_bf16(kD * kD, 0.02f, 3)));
  const std::string md = R"({"format":"pt","mossinfer_lora":"stale","modelspec.title":"tiny",)"
                         R"("_quantization_metadata":"{\"layers\": {\"layers.0.proj.o\": {\"format\": \"int8_tensorwise\", \"convrot\": true}, )"
                         R"(\"text_embedder.proj\": {\"format\": \"int8_tensorwise\", \"convrot\": true}}}"})";
  write_st(dir.path() / "in.safetensors", ts, md);
  mi::quant::ConvertOptions opt;
  opt.attention_config = true;  // byte-compatible with Comfy-Org output, incl. its attention configs
  opt.keep_bf16 = {"layers.0.proj.q"};
  mi::quant::convert(dir.path() / "in.safetensors", dir.path() / "out.safetensors", opt);
  const mi::weights::SafetensorsFile f(dir.path() / "out.safetensors");
  const auto& meta = f.metadata();
  EXPECT_EQ(meta.at("format"), "pt");
  EXPECT_EQ(meta.at("modelspec.title"), "tiny");
  EXPECT_EQ(meta.count("_quantization_metadata"), 0u);
  EXPECT_EQ(meta.count("mossinfer_lora"), 0u);
  EXPECT_NE(meta.at("mossinfer_converter").find("bf16 x1, int8_convrot256 x2"), std::string::npos) << meta.at("mossinfer_converter");
  EXPECT_EQ(meta.at("mossinfer_kept_bf16"), R"(["layers.0.proj.q"])");
  std::set<std::string> attn;
  for (const auto& t : f.tensors()) {
    if (t.name.find("comfy_attention") != std::string::npos) attn.insert(t.name);
  }
  EXPECT_EQ(attn, (std::set<std::string>{"layers.0.comfy_attention.config"}));
  const auto m = mi::quant::open_quant_model(dir.path() / "out.safetensors");
  EXPECT_EQ(m.find_linear("text_embedder.proj")->spec, QuantSpec::int8_convrot(256));
}

// ---- Real community files (skipped when absent) --------------------------------------------------------------

const std::vector<std::pair<std::string, std::vector<std::int64_t>>> kDitGemms = {
    {"proj.q", {3072, 3072}}, {"proj.k", {3072, 3072}}, {"proj.v", {3072, 3072}},
    {"proj.o", {3072, 3072}}, {"proj.ffn_up_gate", {16384, 3072}}, {"proj.ffn_down", {3072, 8192}}};

// Same GEMMs (with shapes) and the same non-GEMM tensors (names, dtypes, shapes) as the official bf16 SFT.
void expect_official_dit_layout(const mi::quant::QuantModel& m) {
  ASSERT_EQ(m.linears.size(), 252u);
  for (int b = 0; b < 42; ++b) {
    for (const auto& [name, shape] : kDitGemms) {
      const auto* l = m.find_linear("layers." + std::to_string(b) + "." + name);
      ASSERT_NE(l, nullptr) << b << " " << name;
      EXPECT_EQ(l->n, shape[0]);
      EXPECT_EQ(l->k, shape[1]);
      for (const auto* t : {l->scale ? &*l->scale : nullptr, l->s_channel ? &*l->s_channel : nullptr}) {
        if (t != nullptr) {
          EXPECT_EQ(t->offset % 4, 0u) << t->name;  // host groups keep file alignment
        }
      }
    }
  }
  const fs::path sft = weights_root() / "sft" / "transformer";
  if (!fs::is_directory(sft)) return;
  const auto ref = mi::quant::open_quant_model(sft);
  ASSERT_EQ(m.tensors.size(), ref.tensors.size());
  for (std::size_t i = 0; i < ref.tensors.size(); ++i) {
    EXPECT_EQ(m.tensors[i].name, ref.tensors[i].name);
    EXPECT_EQ(m.tensors[i].shape, ref.tensors[i].shape) << ref.tensors[i].name;
    EXPECT_EQ(m.tensors[i].dtype, ref.tensors[i].dtype) << ref.tensors[i].name;
  }
}

TEST(QuantRealFile, ComfyOrgInt8Dit) {
  if (!usable(comfy_dit())) GTEST_SKIP() << "community file absent or incomplete: " << comfy_dit();
  const auto m = mi::quant::open_quant_model(comfy_dit());
  expect_official_dit_layout(m);
  for (const auto& l : m.linears) {
    EXPECT_EQ(l.spec, QuantSpec::int8_convrot(256)) << l.name;
    EXPECT_EQ(l.encoding, Encoding::kRaw) << l.name;
    const bool qkv = l.name.ends_with(".q") || l.name.ends_with(".k") || l.name.ends_with(".v");
    EXPECT_EQ(l.fused_slice, qkv) << l.name;
  }
  for (int b = 0; b < 42; ++b) {
    const std::string p = "layers." + std::to_string(b) + ".proj.";
    const auto *q = m.find_linear(p + "q"), *k = m.find_linear(p + "k"), *v = m.find_linear(p + "v");
    EXPECT_EQ(k->weight.offset - q->weight.offset, 3072u * 3072u);
    EXPECT_EQ(v->weight.offset - k->weight.offset, 3072u * 3072u);
    EXPECT_EQ(v->scale->offset - q->scale->offset, 2u * 3072u * 4u);
  }
  // Rows of the fused layers.0.proj.qkv, dequantized by comfy-kitchen (CUDA, whole matrix), against our q/k/v slices:
  // bit-identical, since dequantize_rows sums in the reference GEMM's order.
  const auto rows = load<std::int64_t>("real_comfy_l0_qkv_rows");
  const auto ref = load<float>("real_comfy_l0_qkv_deq");
  std::vector<float> got;
  for (std::int64_t r : rows) {
    const char* part = r < 3072 ? "q" : r < 6144 ? "k" : "v";
    std::vector<float> row(3072);
    mi::quant::dequantize_linear_rows(m, *m.find_linear(std::string("layers.0.proj.") + part), r % 3072, 1, row.data());
    got.insert(got.end(), row.begin(), row.end());
  }
  std::size_t exact = 0;
  const double rel = max_rel_diff(got, ref, &exact);
  RecordProperty("dequant_exact", static_cast<int>(exact));
  RecordProperty("dequant_elements", static_cast<int>(got.size()));
  EXPECT_EQ(exact, got.size()) << "max rel diff " << rel;
  // The engine groups: 44 like the bf16 bundle, block 0 = 5 norms + 6 x (weight, scale).
  const auto index = mi::quant::to_model_index(m).index;
  EXPECT_EQ(index.groups.size(), 44u);
  EXPECT_EQ(index.find_group(mi::weights::Component::kTransformer, "block.0")->tensors.size(), 16u);
}

TEST(QuantRealFile, ToxicdogTurboInt8Dit) {
  if (!usable(toxic())) GTEST_SKIP() << "community file absent or incomplete: " << toxic();
  const auto m = mi::quant::open_quant_model(toxic());
  expect_official_dit_layout(m);
  for (const auto& l : m.linears) {
    EXPECT_EQ(l.spec, QuantSpec::int8_convrot(256)) << l.name;
    EXPECT_FALSE(l.fused_slice) << l.name;
  }
  const auto rows = load<std::int64_t>("real_toxic_l5_v_rows");
  const auto ref = load<float>("real_toxic_l5_v_deq");
  std::vector<float> got;
  for (std::int64_t r : rows) {
    std::vector<float> row(3072);
    mi::quant::dequantize_linear_rows(m, *m.find_linear("layers.5.proj.v"), r, 1, row.data());
    got.insert(got.end(), row.begin(), row.end());
  }
  std::size_t exact = 0;
  const double rel = max_rel_diff(got, ref, &exact);
  EXPECT_EQ(exact, got.size()) << "max rel diff " << rel;
  EXPECT_EQ(mi::quant::to_model_index(m).index.groups.size(), 44u);
}

TEST(QuantRealFile, TsolfulW4A8GptqDecodesExactly) {
  if (!usable(tsolful())) GTEST_SKIP() << "community file absent or incomplete: " << tsolful();
  const auto m = mi::quant::open_quant_model(tsolful());
  expect_official_dit_layout(m);
  int w4 = 0, w6 = 0;
  for (const auto& l : m.linears) {
    EXPECT_EQ(l.spec, QuantSpec::int8_convrot(256));
    const int block = std::stoi(l.name.substr(7));
    EXPECT_EQ(l.encoding, block < 24 ? Encoding::kW4A8 : Encoding::kW6A8) << l.name;
    EXPECT_EQ(l.quant_group, 16);
    (l.encoding == Encoding::kW4A8 ? w4 : w6)++;
  }
  EXPECT_EQ(w4, 144);
  EXPECT_EQ(w6, 108);
  // Reference rows (comfy-kitchen _dequant_int4_grouped_to_int8) from a fused W4 qkv and a W6 ffn_down.
  for (const auto& [tag, prefix, fused] : {std::tuple{"tsolful_l0_qkv", "layers.0.proj.", true}, std::tuple{"tsolful_l30_down", "layers.30.proj.ffn_down", false}}) {
    const auto rows = load<std::int64_t>(std::string("real_") + tag + "_rows");
    const auto ref = load<std::int8_t>(std::string("real_") + tag + "_int8");
    const auto ref_s = load<float>(std::string("real_") + tag + "_schan");
    std::vector<std::int8_t> got;
    std::vector<float> got_s;
    for (std::int64_t r : rows) {
      const auto* l = fused ? m.find_linear(std::string(prefix) + (r < 3072 ? "q" : r < 6144 ? "k" : "v")) : m.find_linear(prefix);
      std::vector<std::int8_t> q(static_cast<std::size_t>(l->k));
      float s;
      mi::quant::decode_rows(m, *l, fused ? r % 3072 : r, 1, q.data(), &s);
      got.insert(got.end(), q.begin(), q.end());
      got_s.push_back(s);
    }
    EXPECT_EQ(got, ref) << tag;
    EXPECT_EQ(got_s, ref_s) << tag;
  }
  // Every GEMM of the file decodes (no NaN scales, every code in range).
  std::vector<std::int8_t> q;
  std::vector<float> s;
  for (const auto& l : m.linears) {
    q.resize(static_cast<std::size_t>(l.n * l.k));
    s.resize(static_cast<std::size_t>(l.n));
    ASSERT_NO_THROW(mi::quant::decode_linear(m, l, q.data(), s.data())) << l.name;
    ASSERT_EQ(std::count(q.begin(), q.end(), std::int8_t{-128}), 0) << l.name;
  }
}

TEST(QuantRealFile, ComfyOrgInt8TextEncoderCoversOurEncoder) {
  if (!usable(comfy_te())) GTEST_SKIP() << "community file absent or incomplete: " << comfy_te();
  const auto m = mi::quant::open_quant_model(comfy_te());
  EXPECT_EQ(m.kind, mi::quant::ModelKind::kTextEncoder);
  // What qwen35_text::TextEncoder reads (text_encoder.cpp weight specs), Qwen3.5-9B shapes.
  constexpr std::int64_t kH = 4096, kI = 12288, kConv = 8192, kV = 4096, kVH = 32, kQg = 8192, kKv = 1024;
  const std::vector<std::pair<std::string, std::vector<std::int64_t>>> common = {
      {"input_layernorm.weight", {kH}}, {"post_attention_layernorm.weight", {kH}}};
  const std::vector<std::pair<std::string, std::vector<std::int64_t>>> common_gemm = {
      {"mlp.gate_proj", {kI, kH}}, {"mlp.up_proj", {kI, kH}}, {"mlp.down_proj", {kH, kI}}};
  const std::vector<std::pair<std::string, std::vector<std::int64_t>>> lin = {
      {"linear_attn.A_log", {kVH}}, {"linear_attn.dt_bias", {kVH}}, {"linear_attn.conv1d.weight", {kConv, 1, 4}}, {"linear_attn.norm.weight", {128}}};
  const std::vector<std::pair<std::string, std::vector<std::int64_t>>> lin_gemm = {
      {"linear_attn.in_proj_qkv", {kConv, kH}}, {"linear_attn.in_proj_z", {kV, kH}}, {"linear_attn.in_proj_b", {kVH, kH}},
      {"linear_attn.in_proj_a", {kVH, kH}},     {"linear_attn.out_proj", {kH, kV}}};
  const std::vector<std::pair<std::string, std::vector<std::int64_t>>> full = {{"self_attn.q_norm.weight", {256}}, {"self_attn.k_norm.weight", {256}}};
  const std::vector<std::pair<std::string, std::vector<std::int64_t>>> full_gemm = {
      {"self_attn.q_proj", {kQg, kH}}, {"self_attn.k_proj", {kKv, kH}}, {"self_attn.v_proj", {kKv, kH}}, {"self_attn.o_proj", {kH, 4096}}};
  std::set<std::string> needed;
  std::map<std::string, int> dtype_notes;
  for (int i = 0; i < 32; ++i) {
    const std::string p = "language_model.layers." + std::to_string(i) + ".";
    const bool fa = i % 4 == 3;
    for (const auto* specs : {&common, fa ? &full : &lin}) {
      for (const auto& [name, shape] : *specs) {
        const auto* t = m.find_tensor(p + name);
        ASSERT_NE(t, nullptr) << p + name;
        EXPECT_EQ(t->shape, shape) << p + name;
        if (t->dtype != DType::kBF16) dtype_notes[name + ":" + mi::weights::dtype_name(t->dtype)]++;
        needed.insert(t->name);
      }
    }
    for (const auto* specs : {&common_gemm, fa ? &full_gemm : &lin_gemm}) {
      for (const auto& [name, shape] : *specs) {
        const auto* l = m.find_linear(p + name);
        ASSERT_NE(l, nullptr) << p + name;
        EXPECT_EQ(l->n, shape[0]);
        EXPECT_EQ(l->k, shape[1]);
        const bool kept = name == "linear_attn.in_proj_a" || name == "linear_attn.in_proj_b";
        EXPECT_EQ(l->spec, kept ? QuantSpec::bf16() : QuantSpec::int8_convrot(256)) << p + name;
        needed.insert(l->name);
      }
    }
  }
  ASSERT_NE(m.find_tensor("language_model.norm.weight"), nullptr);
  needed.insert("language_model.norm.weight");
  const auto* embed = m.find_linear("language_model.embed_tokens");
  ASSERT_NE(embed, nullptr);
  EXPECT_EQ(embed->spec, QuantSpec::int8_convrot(256));
  EXPECT_EQ(embed->n, 248320);
  needed.insert(embed->name);
  // Nothing outside the encoder survives the mapping (visual / mtp / lm_head are excluded).
  for (const auto& t : m.tensors) EXPECT_TRUE(needed.count(t.name)) << "unexpected " << t.name;
  for (const auto& l : m.linears) EXPECT_TRUE(needed.count(l.name)) << "unexpected " << l.name;
  // The unquantized GDN parameters the file stores in fp32 (diffusers stores them in bf16).
  EXPECT_EQ(dtype_notes["linear_attn.A_log:F32"], 24);
  EXPECT_EQ(dtype_notes["linear_attn.norm.weight:F32"], 24);
  EXPECT_EQ(dtype_notes.size(), 2u);
  // Embedding rows, decoded by comfy-kitchen's dequantize_int8_embedding.
  const auto ids = load<std::int64_t>("real_te_embed_ids");
  const auto ref = load<float>("real_te_embed_deq");
  std::vector<float> got;
  for (std::int64_t id : ids) {
    std::vector<float> row(4096);
    mi::quant::dequantize_linear_rows(m, *embed, id, 1, row.data());
    got.insert(got.end(), row.begin(), row.end());
  }
  // The reference gathers the 4 rows before un-rotating them, a 16-row GEMM whose cuBLAS kernel does not accumulate
  // in k order, so single-ulp differences remain (whole-matrix dequantization above is bit-exact).
  std::size_t exact = 0;
  const double rel = max_rel_diff(got, ref, &exact);
  RecordProperty("embed_exact", static_cast<int>(exact));
  EXPECT_LT(rel, 1e-6);
  // The whole-table dequantization (comfy-kitchen's k-order un-rotation, a fixed function of each row): bit-exact in
  // fp32, and the gather helper's bf16 rows equal it rounded to the bf16 compute dtype. ComfyUI's lookup above
  // differs from it only where the fp32 sum cancels to near zero (26 of these 16384 bf16 values), because its order
  // depends on the number of tokens looked up.
  {
    EXPECT_EQ(got, load<float>("real_te_embed_full"));
    const auto ref16 = load<std::uint16_t>("real_te_embed_bf16");
    std::vector<std::uint16_t> got16(ids.size() * 4096);
    mi::quant::gather_embedding_rows_bf16(m, *embed, ids, got16.data());
    EXPECT_EQ(got16, ref16);
    const std::int64_t bad_id[] = {248320};
    EXPECT_THROW(mi::quant::gather_embedding_rows_bf16(m, *embed, bad_id, got16.data()), mi::quant::QuantError);
  }
  // The F32 GDN parameters round to exactly the diffusers bf16 tensors.
  if (fs::is_directory(weights_root() / "sft" / "text_encoder")) {
    const auto diff = mi::quant::open_quant_model(weights_root() / "sft" / "text_encoder", mi::quant::ModelKind::kTextEncoder);
    int checked = 0;
    for (const auto& t : m.tensors) {
      if (t.dtype != DType::kF32) continue;
      const auto* d = diff.find_tensor(t.name);
      ASSERT_NE(d, nullptr) << t.name;
      ASSERT_EQ(d->dtype, DType::kBF16) << t.name;
      EXPECT_EQ(mi::quant::read_tensor_bf16(m, t), mi::quant::read_tensor_bf16(diff, *d)) << t.name;
      ++checked;
    }
    EXPECT_EQ(checked, 48);
  }
  const auto index = mi::quant::to_model_index(m).index;
  EXPECT_EQ(index.groups.size(), 34u);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(index.find_group(mi::weights::Component::kTextEncoder, "layers." + std::to_string(i))->layer_type,
              i % 4 == 3 ? "full_attention" : "linear_attention");
  }
}

// The reader's groups for the official bf16 bundle are build_index's, tensor for tensor (names, dtypes, shapes, byte
// ranges and files), for the DiT and the text encoder.
TEST(QuantRealFile, OfficialBf16GroupsMatchBuildIndex) {
  const fs::path bundle = weights_root() / "sft";
  if (!fs::is_directory(bundle / "transformer") || !fs::is_directory(bundle / "text_encoder") || !fs::is_directory(bundle / "vae")) {
    GTEST_SKIP() << "official weights absent: " << bundle;
  }
  const auto ref = mi::weights::build_index(bundle);
  for (const auto& [kind, comp, dir] : {std::tuple{mi::quant::ModelKind::kDiT, mi::weights::Component::kTransformer, "transformer"},
                                        std::tuple{mi::quant::ModelKind::kTextEncoder, mi::weights::Component::kTextEncoder, "text_encoder"}}) {
    const auto qi = mi::quant::to_model_index(mi::quant::open_quant_model(bundle / dir, kind));
    const auto want = ref.groups_of(comp);
    ASSERT_FALSE(want.empty()) << dir;
    ASSERT_EQ(qi.index.groups.size(), want.size()) << dir;
    for (std::size_t g = 0; g < want.size(); ++g) {
      const auto& a = qi.index.groups[g];
      const auto& b = *want[g];
      ASSERT_EQ(a.name, b.name) << dir;
      EXPECT_EQ(a.layer_type, b.layer_type) << b.name;
      EXPECT_EQ(a.total_bytes, b.total_bytes) << b.name;
      EXPECT_EQ(a.total_params, b.total_params) << b.name;
      ASSERT_EQ(a.tensors.size(), b.tensors.size()) << b.name;
      for (std::size_t t = 0; t < b.tensors.size(); ++t) {
        const auto &x = a.tensors[t], &y = b.tensors[t];
        EXPECT_EQ(x.name, y.name) << b.name;
        EXPECT_EQ(x.dtype, y.dtype) << y.name;
        EXPECT_EQ(x.shape, y.shape) << y.name;
        EXPECT_EQ(x.offset, y.offset) << y.name;
        EXPECT_EQ(x.nbytes, y.nbytes) << y.name;
        EXPECT_EQ(std::string(dir) + "/" + qi.index.files.at(x.file).path, ref.files.at(y.file).path) << y.name;
      }
    }
    std::size_t excluded = 0;
    for (const auto& e : ref.excluded) excluded += e.component == comp;
    EXPECT_EQ(qi.index.excluded.size(), excluded) << dir;
    for (const auto& [name, spec] : qi.specs) EXPECT_EQ(spec, QuantSpec::bf16()) << name;
  }
}

// Manual round-trip check (P6 AC18): MOSS_QUANT_RT_FILE = a file written by `moss-infer quantize`, MOSS_QUANT_RT_REF =
// a directory with <module>.rows.npy (i8 row ids) and <module>.deq.npy (f4 rows) produced by loading that file through
// ComfyUI's comfy_quant loader logic and comfy-kitchen's eager dequantization.
TEST(QuantRealFile, ComfyKitchenLoaderReadsOurFile) {
  const char* file = std::getenv("MOSS_QUANT_RT_FILE");
  const char* refdir = std::getenv("MOSS_QUANT_RT_REF");
  if (file == nullptr || refdir == nullptr) GTEST_SKIP() << "set MOSS_QUANT_RT_FILE and MOSS_QUANT_RT_REF to run";
  const auto m = mi::quant::open_quant_model(file);
  std::size_t modules = 0, elements = 0, exact = 0;
  double worst = 0;
  for (const auto& l : m.linears) {
    const fs::path rp = fs::path(refdir) / (l.name + ".rows.npy");
    if (!fs::exists(rp)) continue;
    const auto rows = mi::npy::load(rp).to_vector<std::int64_t>();
    const auto ref = mi::npy::load(fs::path(refdir) / (l.name + ".deq.npy")).to_vector<float>();
    std::vector<float> got;
    for (std::int64_t r : rows) {
      std::vector<float> row(static_cast<std::size_t>(l.k));
      mi::quant::dequantize_linear_rows(m, l, r, 1, row.data());
      got.insert(got.end(), row.begin(), row.end());
    }
    std::size_t ex = 0;
    worst = std::max(worst, max_rel_diff(got, ref, &ex));
    exact += ex;
    elements += got.size();
    ++modules;
  }
  RecordProperty("modules", static_cast<int>(modules));
  RecordProperty("elements", static_cast<int>(elements));
  RecordProperty("exact", static_cast<int>(exact));
  std::printf("round trip: %zu modules, %zu elements, %zu bit-identical, max rel diff %.3g\n", modules, elements, exact, worst);
  EXPECT_EQ(modules, m.linears.size());
  EXPECT_EQ(exact, elements) << "max rel diff " << worst;
}

}  // namespace
