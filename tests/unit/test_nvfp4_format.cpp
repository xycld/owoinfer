// NVFP4 N1 (design .omc/plans/nvfp4-design.md §4.4, §4.8): the CPU codec (T1, T2), the comfy_quant "nvfp4" importer
// (T4, import side), the BennyDaBall community file and the D1 refusal message. CPU only, except GpuNvfp4Support.
// Vectors in tests/data/nvfp4 come from tools/reference/gen_nvfp4_vectors.py: comfy-kitchen @dba9fac eager (oracle)
// for the weight codec, tools/reference/nvfp4_ref.py for the engine's per-token activation recipe.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "models/moss_dit/nvfp4_support.h"
#include "pipeline/pipeline.h"
#include "test_util.h"
#include "util/json.h"
#include "util/npy.h"
#include "util/xxh64.h"
#include "weights/model_index.h"
#include "weights/quant/nvfp4_codec.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_convert.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
namespace q = mi::quant;
using mi::weights::DType;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/nvfp4";

mi::npy::Array arr(const std::string& name) { return mi::npy::load(kData / (name + ".npy")); }
template <typename T>
std::vector<T> load(const std::string& name) {
  return arr(name).to_vector<T>();
}

// ---- Codec (T1: comfy-kitchen eager, bit-exact) --------------------------------------------------------------

TEST(Nvfp4Codec, ScalarRoundingMatchesOracle) {
  const auto in = load<float>("e2m1_in");
  const auto codes = load<std::uint8_t>("e2m1_codes");
  ASSERT_EQ(in.size(), codes.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    ASSERT_EQ(q::e2m1_rne(std::clamp(in[i], -6.0f, 6.0f)), codes[i]) << "x = " << in[i];
  }
  for (int c = 0; c < 16; ++c) EXPECT_EQ(q::e2m1_rne(q::e2m1_to_float(static_cast<std::uint8_t>(c))), c == 8 ? 8 : c);
  EXPECT_EQ(q::e2m1_rne(std::nanf("")), 0);

  const auto ein = load<float>("e4m3_in");
  const auto ecodes = load<std::uint8_t>("e4m3_codes");
  ASSERT_EQ(ein.size(), ecodes.size());
  for (std::size_t i = 0; i < ein.size(); ++i) ASSERT_EQ(q::e4m3_rne(ein[i]), ecodes[i]) << "x = " << ein[i];
  for (int c = 0; c < 0x7F; ++c) EXPECT_EQ(q::e4m3_rne(q::e4m3_to_float(static_cast<std::uint8_t>(c))), c);
  EXPECT_EQ(q::e4m3_rne(std::nanf("")), 0x7F);
  // Saturating: above 448 (where an unclamped RNE would reach the NaN code 0x7F) and +inf give 448.
  EXPECT_EQ(q::e4m3_rne(464.0f), 0x7E);
  EXPECT_EQ(q::e4m3_rne(1e30f), 0x7E);
  EXPECT_EQ(q::e4m3_rne(std::numeric_limits<float>::infinity()), 0x7E);
  EXPECT_EQ(q::e4m3_rne(-1.0f), 0);
}

TEST(Nvfp4Codec, SwizzleLayout) {
  // The CUTLASS / cuBLAS formula at a few hand-checked points (design §2.1).
  EXPECT_EQ(q::nvfp4_scale_offset(0, 0, 192), 0);
  EXPECT_EQ(q::nvfp4_scale_offset(1, 0, 192), 16);
  EXPECT_EQ(q::nvfp4_scale_offset(32, 0, 192), 4);
  EXPECT_EQ(q::nvfp4_scale_offset(0, 3, 192), 3);
  EXPECT_EQ(q::nvfp4_scale_offset(0, 4, 192), 512);
  EXPECT_EQ(q::nvfp4_scale_offset(128, 0, 192), 48 * 512);
  EXPECT_EQ(q::nvfp4_scale_rows(300), 384);
  EXPECT_EQ(q::nvfp4_scale_cols(3), 4);
  // A bijection onto the padded buffer; unswizzle inverts swizzle; padding is zero.
  for (auto [rows, sfk] : {std::pair<std::int64_t, std::int64_t>{300, 3}, {128, 192}, {1, 1}, {257, 9}}) {
    const std::int64_t size = q::nvfp4_scale_rows(rows) * q::nvfp4_scale_cols(sfk);
    std::vector<int> hits(static_cast<std::size_t>(size), 0);
    for (std::int64_t m = 0; m < q::nvfp4_scale_rows(rows); ++m) {
      for (std::int64_t kb = 0; kb < q::nvfp4_scale_cols(sfk); ++kb) ++hits[static_cast<std::size_t>(q::nvfp4_scale_offset(m, kb, sfk))];
    }
    EXPECT_TRUE(std::all_of(hits.begin(), hits.end(), [](int h) { return h == 1; })) << rows << "x" << sfk;
    std::vector<std::uint8_t> rm(static_cast<std::size_t>(rows * sfk)), back(rm.size()), swz(static_cast<std::size_t>(size), 0xAA);
    for (std::size_t i = 0; i < rm.size(); ++i) rm[i] = static_cast<std::uint8_t>(1 + i % 0x7E);
    q::swizzle_scales(rm.data(), rows, sfk, swz.data());
    q::unswizzle_scales(swz.data(), rows, sfk, back.data());
    EXPECT_EQ(rm, back);
    EXPECT_TRUE(q::valid_nvfp4_scales(swz.data(), rows, sfk));
    EXPECT_EQ(std::count(swz.begin(), swz.end(), 0), size - rows * sfk);
  }
}

// comfy-kitchen quantize_nvfp4 / dequantize_nvfp4 on bf16 weights (and on their fp32 widening with an fp32 tensor
// scale).
TEST(Nvfp4Codec, WeightQuantizeDequantizeBitExact) {
  for (const std::string w : {"w1", "w2"}) {
    const auto shape = arr(w + "_x").shape;
    const std::int64_t n = shape[0], k = shape[1], sfk = k / 16;
    const auto xb = load<std::uint16_t>(w + "_x");
    std::vector<float> x(xb.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = q::bf16_to_float(xb[i]);
    for (const std::string variant : {"", "_f32"}) {
      const float s2 = load<float>(w + "_s2" + variant)[0];
      std::vector<std::uint8_t> packed(static_cast<std::size_t>(n * k / 2)), sf(static_cast<std::size_t>(q::nvfp4_scale_rows(n) * q::nvfp4_scale_cols(sfk)));
      q::quantize_nvfp4_weight(x.data(), n, k, s2, packed.data(), sf.data());
      EXPECT_EQ(packed, load<std::uint8_t>(w + "_packed" + variant)) << w << variant;
      EXPECT_EQ(sf, load<std::uint8_t>(w + "_sf" + variant)) << w << variant;
      EXPECT_TRUE(q::valid_nvfp4_scales(sf.data(), n, sfk));
    }
    // Dequantization: fp32 and ComfyUI's bf16 order, also starting mid-matrix (row0 within a scale tile).
    const float s2 = load<float>(w + "_s2")[0];
    const auto packed = load<std::uint8_t>(w + "_packed");
    const auto sf = load<std::uint8_t>(w + "_sf");
    std::vector<float> f32(static_cast<std::size_t>(n * k));
    std::vector<std::uint16_t> b16(static_cast<std::size_t>(n * k));
    q::dequantize_nvfp4_rows(packed.data(), sf.data(), 0, n, k, s2, f32.data());
    q::dequantize_nvfp4_rows_bf16(packed.data(), sf.data(), 0, n, k, s2, b16.data());
    const auto ref32 = load<float>(w + "_deq_f32");
    ASSERT_EQ(ref32.size(), f32.size());
    EXPECT_EQ(std::memcmp(f32.data(), ref32.data(), f32.size() * 4), 0) << w;
    EXPECT_EQ(b16, load<std::uint16_t>(w + "_deq_bf16")) << w;
    const std::int64_t r0 = 37, rows = n - r0;
    std::vector<float> part(static_cast<std::size_t>(rows * k));
    q::dequantize_nvfp4_rows(packed.data() + r0 * k / 2, sf.data(), r0, rows, k, s2, part.data());
    EXPECT_EQ(std::memcmp(part.data(), ref32.data() + r0 * k, part.size() * 4), 0) << w;
  }
}

// ---- Activation recipe (T2: nvfp4_ref.py, pinned to comfy-kitchen at equal row scales) ----------------------

TEST(Nvfp4Codec, ActivationRecipeMatchesReference) {
  const auto shape = arr("act_x").shape;
  const std::int64_t m = shape[0], k = shape[1];
  const auto x = load<std::uint16_t>("act_x");
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(m * k / 2));
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(q::nvfp4_scale_rows(m) * q::nvfp4_scale_cols(k / 16)), 0xAA);
  std::vector<float> g(static_cast<std::size_t>(m));
  q::quantize_nvfp4_act_bf16(x.data(), m, k, packed.data(), sf.data(), g.data());
  EXPECT_EQ(packed, load<std::uint8_t>("act_packed"));
  EXPECT_EQ(sf, load<std::uint8_t>("act_sf"));  // includes the zeroed padding rows and columns
  const auto gref = load<float>("act_g");
  ASSERT_EQ(g.size(), gref.size());
  EXPECT_EQ(std::memcmp(g.data(), gref.data(), g.size() * 4), 0);
  // Row 0 is all zero: g = 1, every code +0. Row 2 holds +-inf: saturated to +-6 (codes 0x7 / 0xF).
  EXPECT_EQ(g[0], 1.0f);
  EXPECT_TRUE(std::all_of(packed.begin(), packed.begin() + k / 2, [](std::uint8_t b) { return b == 0; }));
  EXPECT_EQ(packed[static_cast<std::size_t>(2 * k / 2 + 3 / 2)] & 0xF, 0x7);
  EXPECT_EQ(packed[static_cast<std::size_t>(2 * k / 2 + 40 / 2)] >> 4, 0xF);
}

// ---- Importer (T4, import side) on synthetic files ------------------------------------------------------------

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
  if (!v.empty()) std::memcpy(t.data.data(), v.data(), t.data.size());
  return t;
}
T make_json(std::string name, const std::string& s) { return make(std::move(name), DType::kU8, {static_cast<std::int64_t>(s.size())}, std::vector<char>(s.begin(), s.end())); }

// safetensors-library layout: larger dtypes first, header padded to 8 bytes.
void write_st(const fs::path& path, std::vector<T> ts) {
  std::stable_sort(ts.begin(), ts.end(), [](const T& a, const T& b) { return mi::weights::dtype_size(a.dtype) > mi::weights::dtype_size(b.dtype); });
  std::string h = "{";
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

std::vector<float> rand_w(std::size_t n, float sigma, std::uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, sigma);
  std::vector<float> v(n);
  for (auto& x : v) x = q::bf16_to_float(q::float_to_bf16(d(g)));
  return v;
}

struct Fp4Module {
  std::vector<float> w;
  std::vector<std::uint8_t> packed, sf;
  float s2 = 0;
};

// An nvfp4 module (ComfyUI layout) quantized from random weights with the fp32 tensor-scale recipe.
Fp4Module add_nvfp4(std::vector<T>& ts, const std::string& m, std::int64_t n, std::int64_t k, std::uint32_t seed,
                    const std::string& config = R"({"format": "nvfp4"})") {
  Fp4Module f;
  f.w = rand_w(static_cast<std::size_t>(n * k), 0.02f, seed);
  float amax = 0;
  for (float v : f.w) amax = std::max(amax, std::fabs(v));
  f.s2 = q::nvfp4_tensor_scale(amax);
  f.packed.resize(static_cast<std::size_t>(n * k / 2));
  f.sf.resize(static_cast<std::size_t>(q::nvfp4_scale_rows(n) * q::nvfp4_scale_cols(k / 16)));
  q::quantize_nvfp4_weight(f.w.data(), n, k, f.s2, f.packed.data(), f.sf.data());
  ts.push_back(make(m + ".weight", DType::kU8, {n, k / 2}, f.packed));
  ts.push_back(make(m + ".weight_scale", DType::kF8E4M3, {q::nvfp4_scale_rows(n), q::nvfp4_scale_cols(k / 16)}, f.sf));
  ts.push_back(make(m + ".weight_scale_2", DType::kF32, {}, std::vector<float>{f.s2}));
  if (!config.empty()) ts.push_back(make_json(m + ".comfy_quant", config));
  return f;
}

void add_int8(std::vector<T>& ts, const std::string& m, std::int64_t n, std::int64_t k, std::uint32_t seed) {
  const auto w = rand_w(static_cast<std::size_t>(n * k), 0.02f, seed);
  std::vector<std::int8_t> qv(static_cast<std::size_t>(n * k));
  std::vector<float> s(static_cast<std::size_t>(n));
  ASSERT_TRUE(q::quantize_convrot_rows(w.data(), n, k, 256, qv.data(), s.data()));
  ts.push_back(make(m + ".weight", DType::kI8, {n, k}, qv));
  ts.push_back(make(m + ".weight_scale", DType::kF32, {n, 1}, s));
  ts.push_back(make_json(m + ".comfy_quant", R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})"));
}

constexpr std::int64_t kD = 256;

// 64-byte aligned copies of a host group's tensors, as a device upload places them (host groups keep each tensor's
// file offset modulo 4096, and the NVFP4 view requires 16-byte aligned weights and scales).
struct AlignedMap {
  std::vector<std::unique_ptr<std::byte[]>> storage;
  mi::weights::TensorMap tensors;
};
AlignedMap aligned_copy(const mi::weights::TensorMap& m) {
  AlignedMap a;
  for (const auto& [name, t] : m) {
    const std::size_t n = t.nbytes();
    a.storage.push_back(std::make_unique<std::byte[]>(n + 64));
    auto* p = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(a.storage.back().get()) + 63) / 64 * 64);
    std::memcpy(p, t.data, n);
    a.tensors.emplace(name, mi::Tensor::view(p, t.dtype, t.device, t.shape.data(), t.rank));
  }
  return a;
}

std::string error_of(const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

TEST(Nvfp4Import, MixedFileFusedQkvAndEngineView) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("nvfp4_mixed");
  std::vector<T> ts;
  const Fp4Module qkv = add_nvfp4(ts, "layers.0.proj.qkv", 3 * kD, kD, 1);
  const Fp4Module o = add_nvfp4(ts, "layers.0.proj.o", kD, kD, 2);
  ts.push_back(make("layers.0.proj.o.input_scale", DType::kF32, {}, std::vector<float>{0.5f}));
  add_int8(ts, "layers.0.proj.ffn_up_gate", 2 * kD, kD, 3);
  ts.push_back(make("layers.0.proj.ffn_down.weight", DType::kBF16, {kD, 2 * kD}, std::vector<std::uint16_t>(2 * kD * kD, 0x3C00)));
  ts.push_back(make("layers.0.norm1.weight", DType::kBF16, {kD}, std::vector<std::uint16_t>(kD, 0x3F80)));
  ts.push_back(make_json("layers.0.comfy_attention.config", R"({"attention": "comfy_kitchen_int8"})"));
  write_st(dir.path() / "dit.safetensors", ts);

  const auto m = q::open_quant_model(dir.path() / "dit.safetensors");
  ASSERT_EQ(m.linears.size(), 6u);
  const auto* lq = m.find_linear("layers.0.proj.q");
  const auto* lk = m.find_linear("layers.0.proj.k");
  const auto* lo = m.find_linear("layers.0.proj.o");
  ASSERT_TRUE(lq && lk && lo);
  EXPECT_EQ(lq->spec, q::QuantSpec::nvfp4());
  EXPECT_EQ(q::describe(lq->spec), "nvfp4");
  EXPECT_EQ(lq->encoding, q::Encoding::kNvfp4Raw);
  EXPECT_FALSE(lq->needs_decode());
  EXPECT_EQ(lq->n, kD);
  EXPECT_EQ(lq->k, kD);
  EXPECT_TRUE(lq->fused_slice);
  // Row slices: weight rows and whole 128-row scale tiles; the tensor scale is shared.
  EXPECT_EQ(lk->weight.offset, lq->weight.offset + static_cast<std::uint64_t>(kD * kD / 2));
  EXPECT_EQ(lk->scale->offset, lq->scale->offset + static_cast<std::uint64_t>(kD * (kD / 16)));
  EXPECT_EQ(lk->scale_2->offset, lq->scale_2->offset);
  EXPECT_FALSE(lq->input_scale.has_value());
  EXPECT_TRUE(lo->input_scale.has_value());
  EXPECT_EQ(m.find_linear("layers.0.proj.ffn_up_gate")->spec, q::QuantSpec::int8_convrot(256));
  EXPECT_EQ(m.find_linear("layers.0.proj.ffn_down")->spec, q::QuantSpec::bf16());

  // CPU dequantization of the k slice equals the codec on the fused weight's rows kD..2kD.
  std::vector<float> got(static_cast<std::size_t>(kD * kD)), want(got.size());
  q::dequantize_linear_rows(m, *lk, 0, kD, got.data());
  q::dequantize_nvfp4_rows(qkv.packed.data() + kD * kD / 2, qkv.sf.data(), kD, kD, kD, qkv.s2, want.data());
  EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size() * 4), 0);
  std::vector<std::uint16_t> gotb(got.size()), wantb(got.size());
  q::dequantize_nvfp4_linear_rows(m, *lo, 100, 50, nullptr, gotb.data());
  q::dequantize_nvfp4_rows_bf16(o.packed.data() + 100 * kD / 2, o.sf.data(), 100, 50, kD, o.s2, wantb.data());
  EXPECT_TRUE(std::equal(gotb.begin(), gotb.begin() + 50 * kD, wantb.begin()));
  // Close to the bf16 weights (NVFP4 is ~6-10 % rel. error per element at most; just a sanity bound on the whole).
  double num = 0, den = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    num += std::pow(got[i] - qkv.w[static_cast<std::size_t>(kD * kD) + i], 2);
    den += std::pow(qkv.w[static_cast<std::size_t>(kD * kD) + i], 2);
  }
  EXPECT_LT(std::sqrt(num / den), 0.15);
  std::vector<std::int8_t> i8(static_cast<std::size_t>(kD * kD));
  std::vector<float> s(kD);
  EXPECT_THROW(q::decode_rows(m, *lk, 0, 1, i8.data(), s.data()), q::QuantError);

  // Engine view through the regular host upload of to_model_index's groups.
  const auto qi = q::to_model_index(m, {.native_int4 = true});
  EXPECT_EQ(*qi.find_spec("layers.0.proj.v"), q::QuantSpec::nvfp4());
  ASSERT_EQ(qi.tensor_scales.size(), 4u);
  EXPECT_EQ(qi.tensor_scales.at("layers.0.proj.v"), qkv.s2);
  EXPECT_EQ(qi.tensor_scales.at("layers.0.proj.o"), o.s2);
  const auto* block = qi.index.find_group(mi::weights::Component::kTransformer, "block.0");
  ASSERT_NE(block, nullptr);
  // 4 nvfp4 x (weight, scales), int8 (weight, scale), the bf16 ffn_down and norm1.
  EXPECT_EQ(block->tensors.size(), 4u * 2 + 2 + 1 + 1);
  mi::weights::ReaderCache readers(qi.index, false);
  const auto loaded = mi::weights::load_group_to_host(readers, *block);
  const AlignedMap host = aligned_copy(loaded.tensors);
  const auto v = q::view(qi, host.tensors, "layers.0.proj.v", q::QuantSpec::nvfp4());
  EXPECT_EQ(v.n, kD);
  EXPECT_EQ(v.k, kD);
  EXPECT_EQ(v.scale, nullptr);
  EXPECT_EQ(v.tensor_scale, qkv.s2);
  EXPECT_EQ(std::memcmp(v.weight, qkv.packed.data() + 2 * kD * kD / 2, static_cast<std::size_t>(kD * kD / 2)), 0);
  EXPECT_EQ(std::memcmp(v.block_scale, qkv.sf.data() + 2 * kD * (kD / 16), static_cast<std::size_t>(kD * (kD / 16))), 0);
  EXPECT_THROW(q::view(qi, host.tensors, "layers.0.proj.v", q::QuantSpec::int8_convrot(256)), q::QuantError);
  EXPECT_THROW(q::view(qi, host.tensors, "layers.0.proj.ffn_up_gate", q::QuantSpec::nvfp4()), q::QuantError);
  // A weight that is not 16-byte aligned (the GEMMs' TMA loads need it).
  auto shifted = host.tensors;
  mi::Tensor& w = shifted.at("layers.0.proj.v.weight");
  std::vector<std::byte> buf(w.nbytes() + 64);
  auto* odd = reinterpret_cast<std::byte*>((reinterpret_cast<std::uintptr_t>(buf.data()) + 63) / 64 * 64 + 8);
  w = mi::Tensor::view(odd, w.dtype, w.device, w.shape.data(), w.rank);
  const std::string align = error_of([&] { q::view(qi, shifted, "layers.0.proj.v", q::QuantSpec::nvfp4()); });
  EXPECT_NE(align.find("16-byte aligned"), std::string::npos) << align;

  // The INT8 converter refuses an NVFP4 input instead of compounding two roundings.
  const std::string err = error_of([&] { q::convert(dir.path() / "dit.safetensors", dir.path() / "out.safetensors", {}); });
  EXPECT_NE(err.find("nvfp4"), std::string::npos) << err;
  EXPECT_FALSE(fs::exists(dir.path() / "out.safetensors"));
}

TEST(Nvfp4Import, RefusesModelOptAndMalformedFiles) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("nvfp4_bad");
  int case_no = 0;
  // Builds a one-module file with `edit` applied to its tensors and returns the error of opening + indexing it.
  auto err = [&](const std::function<void(std::vector<T>&)>& edit, std::int64_t n = kD, std::int64_t k = kD,
                 const std::string& module = "layers.0.proj.o", const std::string& config = R"({"format": "nvfp4"})") {
    std::vector<T> ts;
    add_nvfp4(ts, module, n, k, 7, config);
    edit(ts);
    const fs::path p = dir.path() / ("f" + std::to_string(case_no++) + ".safetensors");
    write_st(p, ts);
    return error_of([&] { q::to_model_index(q::open_quant_model(p, q::ModelKind::kDiT)); });
  };
  auto find = [](std::vector<T>& ts, const std::string& suffix) -> T& {
    for (T& t : ts) {
      if (t.name.ends_with(suffix)) return t;
    }
    throw std::runtime_error("no " + suffix);
  };
  auto has = [](const std::string& e, const std::string& s) { return e.find(s) != std::string::npos; };
  EXPECT_EQ(err([](auto&) {}), "");

  // ModelOpt convention: the NVFP4 tensors without a comfy_quant config.
  std::string e = err([](auto&) {}, kD, kD, "layers.0.proj.o", "");
  EXPECT_TRUE(has(e, "ModelOpt")) << e;
  // ... or an exported directory with hf_quant_config.json.
  {
    std::vector<T> ts;
    add_nvfp4(ts, "layers.0.proj.o", kD, kD, 7);
    write_st(dir.path() / "modelopt/transformer/model.safetensors", ts);
    std::ofstream(dir.path() / "modelopt/transformer/hf_quant_config.json") << R"({"quantization": {"quant_algo": "NVFP4"}})";
    e = error_of([&] { q::open_quant_model(dir.path() / "modelopt/transformer"); });
    EXPECT_TRUE(has(e, "ModelOpt")) << e;
  }
  e = err([&](auto& ts) { ts.push_back(make("layers.0.proj.o.pre_quant_scale", DType::kBF16, {kD}, std::vector<std::uint16_t>(kD, 0x3F80))); });
  EXPECT_TRUE(has(e, "pre_quant_scale")) << e;
  e = err([&](auto& ts) { find(ts, ".weight").dtype = DType::kI8; });
  EXPECT_TRUE(has(e, "U8 [N, K/2]")) << e;
  // A linear [N, K/16] scale (ModelOpt's default) where the swizzled shape differs: N = 200 -> [256, 16].
  e = err([&](auto& ts) {
    T& s = find(ts, ".weight_scale");
    s.shape = {200, 16};
    s.data.resize(200 * 16);
  }, 200);
  EXPECT_TRUE(has(e, "swizzled")) << e;
  e = err([&](auto& ts) { find(ts, ".weight_scale_2").dtype = DType::kBF16; });
  EXPECT_TRUE(has(e, "weight_scale_2")) << e;
  e = err([&](auto& ts) { std::memcpy(find(ts, ".weight_scale_2").data.data(), std::vector<float>{-1.0f}.data(), 4); });
  EXPECT_TRUE(has(e, "weight_scale_2 is not finite")) << e;
  e = err([&](auto& ts) { find(ts, ".weight_scale").data[5] = std::byte{0x7F}; });
  EXPECT_TRUE(has(e, "holds a NaN block scale")) << e;
  e = err([&](auto& ts) { find(ts, ".weight_scale").data[5] = std::byte{0xFF}; });  // NaN with the sign bit
  EXPECT_TRUE(has(e, "holds a NaN block scale")) << e;
  e = err([&](auto& ts) { find(ts, ".weight_scale").data[5] = std::byte{0x81}; });
  EXPECT_TRUE(has(e, "holds a negative block scale")) << e;
  // Nonzero padding in the M direction: N = 200 leaves rows 200..255 of the last tile unused ...
  e = err([&](auto& ts) {
    find(ts, ".weight_scale").data[static_cast<std::size_t>(q::nvfp4_scale_offset(250, 0, kD / 16))] = std::byte{0x38};
  }, 200);
  EXPECT_TRUE(has(e, "holds a nonzero padding byte")) << e;
  // ... and in the K direction: K = 48 has 3 scales per row in a 4-column tile.
  EXPECT_EQ(err([](auto&) {}, kD, 48), "");
  e = err([&](auto& ts) { find(ts, ".weight_scale").data[static_cast<std::size_t>(q::nvfp4_scale_offset(7, 3, 3))] = std::byte{0x38}; }, kD, 48);
  EXPECT_TRUE(has(e, "holds a nonzero padding byte")) << e;
  e = err([](auto&) {}, 3 * 64, kD, "layers.0.proj.qkv");  // thirds of 64 rows: not whole scale tiles
  EXPECT_TRUE(has(e, "128-row")) << e;
  e = err([](auto&) {}, kD, kD, "layers.0.proj.o", R"({"format": "nvfp4", "group_size": 32})");
  EXPECT_TRUE(has(e, "group_size")) << e;
  e = err([](auto&) {}, kD, kD, "layers.0.proj.o", R"({"format": "nvfp4", "convrot": true})");
  EXPECT_TRUE(has(e, "rotation")) << e;
  for (const std::string field : {R"("convrot_groupsize": 256)", R"("quant_group_size": 64)", R"("linear_dtype": "int4")"}) {
    e = err([](auto&) {}, kD, kD, "layers.0.proj.o", R"({"format": "nvfp4", )" + field + "}");
    EXPECT_TRUE(has(e, "does not belong to the nvfp4 format")) << e;
  }
  e = err([](auto&) {}, kD, kD, "layers.0.proj.o", R"({"format": "nvfp4", "full_precision_matrix_mult": true})");
  EXPECT_TRUE(has(e, "full_precision_matrix_mult")) << e;
  EXPECT_EQ(err([](auto&) {}, kD, kD, "layers.0.proj.o", R"({"format": "nvfp4", "group_size": 16, "convrot": false})"), "");
  e = err([&](auto& ts) {  // K = 24
    T& w = find(ts, ".weight");
    w.shape = {kD, 12};
    w.data.resize(kD * 12);
  });
  EXPECT_TRUE(has(e, "multiple of 16")) << e;
}

// A valid file whose K the NVFP4 GEMMs cannot run (K % 32 != 0): it imports and dequantizes, the engine view refuses.
TEST(Nvfp4Import, ViewNeedsKMultipleOf32) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  mi::testing::TempDir dir("nvfp4_k48");
  std::vector<T> ts;
  const Fp4Module o = add_nvfp4(ts, "layers.0.proj.o", kD, 48, 3);
  write_st(dir.path() / "dit.safetensors", ts);
  const auto m = q::open_quant_model(dir.path() / "dit.safetensors", q::ModelKind::kDiT);
  std::vector<float> got(static_cast<std::size_t>(kD * 48)), want(got.size());
  q::dequantize_linear_rows(m, *m.find_linear("layers.0.proj.o"), 0, kD, got.data());
  q::dequantize_nvfp4_rows(o.packed.data(), o.sf.data(), 0, kD, 48, o.s2, want.data());
  EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size() * 4), 0);
  const auto qi = q::to_model_index(m);
  mi::weights::ReaderCache readers(qi.index, false);
  const auto loaded = mi::weights::load_group_to_host(readers, *qi.index.find_group(mi::weights::Component::kTransformer, "block.0"));
  const AlignedMap host = aligned_copy(loaded.tensors);
  const std::string e = error_of([&] { q::view(qi, host.tensors, "layers.0.proj.o", q::QuantSpec::nvfp4()); });
  EXPECT_NE(e.find("multiple of 32"), std::string::npos) << e;
}

// ---- The community file (BennyDaBall/MOSS-Image-SFT-NVFP4) ----------------------------------------------------

fs::path community() {
  const char* env = std::getenv("MOSS_COMMUNITY");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/community");
}

std::string hex64(std::uint64_t v) {
  std::ostringstream s;
  s << std::hex;
  s.width(16);
  s.fill('0');
  s << v;
  return s.str();
}

// Imports the mixed nvfp4 + INT8 ConvRot file and checks every nvfp4 module's CPU dequantization (bf16 and fp32)
// against comfy-kitchen's (xxh64 digests from gen_nvfp4_vectors.py --community; fused qkv = the q, k, v rows in order).
TEST(Nvfp4RealFile, BennyDaBallImportsAndMatchesComfyKitchen) {
  const fs::path file = community() / "bennydaball/moss_image_sft_nvfp4.safetensors";
  if (!fs::is_regular_file(file) || fs::exists(fs::path(file) += ".aria2")) GTEST_SKIP() << file << " not downloaded";
  std::ifstream js(kData / "bennydaball_sft_nvfp4.json");
  std::stringstream ss;
  ss << js.rdbuf();
  const mi::json::Value doc = mi::json::parse(ss.str());
  ASSERT_EQ(static_cast<std::uint64_t>(doc.find("file_size")->as_int()), fs::file_size(file));

  const auto m = q::open_quant_model(file);
  std::map<std::string, int> counts;
  for (const auto& l : m.linears) ++counts[q::describe(l.spec)];
  EXPECT_EQ(m.linears.size(), 42u * 6);
  EXPECT_EQ(counts["nvfp4"], 24 * 5);                // q, k, v, o, ffn_up_gate of blocks 0-23
  EXPECT_EQ(counts["int8_convrot256"], 18 * 5 + 42);  // the same in blocks 24-41, ffn_down everywhere
  for (const auto& l : m.linears) {
    if (l.spec.format == q::QuantFormat::kNvfp4) {
      EXPECT_FALSE(l.input_scale.has_value()) << l.name;  // dynamic activations
    }
  }
  const auto qi = q::to_model_index(m, {.native_int4 = true});  // validates every block and tensor scale
  EXPECT_EQ(qi.tensor_scales.size(), 120u);
  // globals, text_embedder and the 42 blocks
  EXPECT_EQ(qi.index.groups_of(mi::weights::Component::kTransformer).size(), 2u + 42);

  const auto& mods = doc.find("modules")->as_object();
  ASSERT_EQ(mods.size(), 72u);
  std::vector<std::pair<std::string, const mi::json::Value*>> list;
  for (const auto& [name, v] : mods) list.emplace_back(name, &v);
  std::vector<std::string> failures(list.size());
  q::parallel_for(static_cast<std::int64_t>(list.size()), 0, [&](std::int64_t b, std::int64_t e) {
    for (std::int64_t i = b; i < e; ++i) {
      const std::string& name = list[static_cast<std::size_t>(i)].first;
      std::vector<const q::QuantLinear*> parts;
      if (name.ends_with(".qkv")) {
        const std::string base = name.substr(0, name.size() - 3);
        for (const char* p : {"q", "k", "v"}) parts.push_back(m.find_linear(base + p));
      } else {
        parts.push_back(m.find_linear(name));
      }
      mi::Xxh64 h16, h32;
      for (const q::QuantLinear* l : parts) {
        if (l == nullptr || l->spec.format != q::QuantFormat::kNvfp4) {
          failures[static_cast<std::size_t>(i)] = name + ": not an nvfp4 linear";
          break;
        }
        constexpr std::int64_t kRows = 256;
        std::vector<float> f32(static_cast<std::size_t>(kRows * l->k));
        std::vector<std::uint16_t> b16(f32.size());
        for (std::int64_t r = 0; r < l->n; r += kRows) {
          const std::int64_t rows = std::min(kRows, l->n - r);
          q::dequantize_nvfp4_linear_rows(m, *l, r, rows, f32.data(), b16.data());
          h32.update(f32.data(), static_cast<std::size_t>(rows * l->k) * 4);
          h16.update(b16.data(), static_cast<std::size_t>(rows * l->k) * 2);
        }
      }
      const mi::json::Value& want = *list[static_cast<std::size_t>(i)].second;
      if (failures[static_cast<std::size_t>(i)].empty() &&
          (hex64(h16.digest()) != want.find("bf16")->as_string() || hex64(h32.digest()) != want.find("f32")->as_string())) {
        failures[static_cast<std::size_t>(i)] = name + ": bf16 " + hex64(h16.digest()) + " / f32 " + hex64(h32.digest()) + " differ from comfy-kitchen";
      }
    }
  });
  int bad = 0;
  for (const auto& f : failures) {
    if (!f.empty()) {
      ADD_FAILURE() << f;
      ++bad;
    }
  }
  RecordProperty("nvfp4_modules_bit_exact", static_cast<int>(list.size()) - bad);
}

// ---- D1: NVFP4 files are refused without FP4 tensor cores -----------------------------------------------------

TEST(Nvfp4Support, RefusalMessages) {
  const std::string need = "NVFP4 needs an RTX 50 GPU (compute capability 12.0); use the INT8 file";
  EXPECT_EQ(mi::moss_dit::nvfp4_unsupported_reason(8, 9, false), need + " (this GPU is compute capability 8.9)");
  EXPECT_EQ(mi::moss_dit::nvfp4_unsupported_reason(12, 1, true).rfind(need, 0), 0u);  // DGX Spark: no sm_120a image
  EXPECT_EQ(mi::moss_dit::nvfp4_unsupported_reason(-1, 0, false), need + " (no CUDA device)");
  EXPECT_NE(mi::moss_dit::nvfp4_unsupported_reason(12, 0, false).find("use the INT8 file"), std::string::npos);
  // An RTX 50 with the kernels runs them.
  EXPECT_EQ(mi::moss_dit::nvfp4_unsupported_reason(12, 0, true), "");
}

// A MOSS-Image DiT file with NVFP4 block linears, written sparse: the header, the comfy_quant configs and holes (the
// load path refuses the file before reading any weight). It passes weights::check_supported_dit_file.
void write_sparse_nvfp4_dit(const fs::path& path) {
  struct S {
    std::string name, dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t bytes;
    std::string content;  // comfy_quant JSON; empty: a hole
  };
  std::vector<S> ts;
  auto bf16 = [&](const std::string& n, std::vector<std::int64_t> shape) {
    std::uint64_t e = 1;
    for (auto d : shape) e *= static_cast<std::uint64_t>(d);
    ts.push_back({n, "BF16", shape, 2 * e, ""});
  };
  bf16("video_embedder.weight", {3072, 128});
  bf16("text_embedder.proj.weight", {3072, 4096});
  bf16("time_embedding.0.weight", {3072, 256});
  bf16("final_linear_video.weight", {128, 3072});
  for (int b = 0; b < 42; ++b) {
    const std::string p = "layers." + std::to_string(b) + ".";
    bf16(p + "norm1.weight", {3072});
    bf16(p + "q_norm.weight", {128});
    for (const auto& [m, n, k] : {std::tuple<const char*, std::int64_t, std::int64_t>{"proj.qkv", 9216, 3072}, {"proj.ffn_down", 3072, 8192}}) {
      const std::string mod = p + m;
      const std::string cfg = R"({"format": "nvfp4"})";
      ts.push_back({mod + ".weight_scale_2", "F32", {}, 4, ""});
      ts.push_back({mod + ".weight", "U8", {n, k / 2}, static_cast<std::uint64_t>(n * k / 2), ""});
      ts.push_back({mod + ".weight_scale", "F8_E4M3", {n, k / 16}, static_cast<std::uint64_t>(n * k / 16), ""});
      ts.push_back({mod + ".comfy_quant", "U8", {static_cast<std::int64_t>(cfg.size())}, cfg.size(), cfg});
    }
  }
  std::stable_sort(ts.begin(), ts.end(), [](const S& a, const S& b) { return (a.dtype == "F32") > (b.dtype == "F32"); });
  std::string h = "{";
  std::uint64_t off = 0;
  std::vector<std::uint64_t> offsets;
  for (const S& t : ts) {
    if (h.size() > 1) h += ",";
    h += "\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
    h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + t.bytes) + "]}";
    offsets.push_back(off);
    off += t.bytes;
  }
  h += "}";
  h.append((8 - h.size() % 8) % 8, ' ');
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t len = h.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (std::size_t i = 0; i < ts.size(); ++i) {
      if (ts[i].content.empty()) continue;
      out.seekp(static_cast<std::streamoff>(8 + h.size() + offsets[i]));
      out.write(ts[i].content.data(), static_cast<std::streamsize>(ts[i].content.size()));
    }
  }
  fs::resize_file(path, 8 + h.size() + off);
}

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}

// The device-free inspection the load path runs (pipeline::inspect_dit_file): with a refusal reason an NVFP4 file is
// answered by it before any scale is read; without one its scales are read and validated like any other layout's (the
// holes hold a zero weight_scale_2, which is refused).
TEST(Nvfp4Support, InspectDitFile) {
  mi::testing::TempDir dir("nvfp4_inspect");
  const fs::path file = dir.path() / "moss_nvfp4.safetensors";
  write_sparse_nvfp4_dit(file);
  int asked = 0;  // asked once, as the file has NVFP4 linears
  const mi::pipeline::DitFileInfo info = mi::pipeline::inspect_dit_file(file, [&] {
    ++asked;
    return std::string("no FP4 tensor cores here");
  });
  EXPECT_EQ(asked, 1);
  EXPECT_TRUE(info.nvfp4);
  EXPECT_EQ(info.quant, nullptr);
  EXPECT_EQ(info.refusal, mi::util::path_utf8(file) + ": no FP4 tensor cores here");
  try {
    mi::pipeline::inspect_dit_file(file, "");
    ADD_FAILURE() << "zero tensor scales were accepted";
  } catch (const mi::weights::IndexError& e) {
    const std::string msg = e.what();
    EXPECT_EQ(msg.rfind(mi::util::path_utf8(file), 0), 0u) << msg;
    EXPECT_NE(msg.find("weight_scale_2 is not finite and > 0"), std::string::npos) << msg;
  }
}

// The real load path (Pipeline -> open_dit_file) refuses an NVFP4 DiT as an unsupported model (MI_ERR_MODEL in the C
// API) with the D1 message where NVFP4 cannot run, before the planner, the arena or any weight upload. Where it can,
// the file's scales are validated there instead (the holes' zero tensor scales: an IndexError, MI_ERR_MODEL too).
TEST(GpuNvfp4Support, PipelineRefusesNvfp4Dit) {
  const fs::path bundle = weights_root() / "sft";
  if (!fs::is_regular_file(bundle / "transformer" / "config.json")) GTEST_SKIP() << "no MOSS-Image bundle at " << bundle;
  mi::testing::TempDir dir("nvfp4_load");
  const fs::path file = dir.path() / "moss_nvfp4.safetensors";
  write_sparse_nvfp4_dit(file);
  ASSERT_NO_THROW(mi::weights::check_supported_dit_file(file));
  mi::pipeline::PipelineOptions options;
  options.dit_file = file;
  mi::pipeline::Pipeline pipe(options);
  mi::pipeline::Request req;
  req.model = bundle;
  req.prompt = "a red apple";
  req.width = req.height = 256;
  req.steps = 1;
  req.decode = false;
  const std::string why = mi::moss_dit::nvfp4_unsupported_reason();
  try {
    pipe.generate(req);
    FAIL() << "an NVFP4 DiT with zero tensor scales was accepted";
  } catch (const mi::weights::UnsupportedModelError& e) {
    const std::string msg = e.what();
    EXPECT_FALSE(why.empty()) << msg;
    EXPECT_EQ(msg.rfind(mi::util::path_utf8(file) + ": ", 0), 0u) << msg;
    EXPECT_NE(msg.find(why), std::string::npos) << msg;
    RecordProperty("message", msg);
  } catch (const mi::weights::IndexError& e) {
    const std::string msg = e.what();
    EXPECT_TRUE(why.empty()) << msg;
    EXPECT_EQ(msg.rfind(mi::util::path_utf8(file), 0), 0u) << msg;
    EXPECT_NE(msg.find("weight_scale_2 is not finite and > 0"), std::string::npos) << msg;
    RecordProperty("message", msg);
  }
}

TEST(GpuNvfp4Support, CurrentDevice) {
  int major = 0, minor = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
  const std::string why = mi::moss_dit::nvfp4_unsupported_reason();
  if (major != 12 || minor != 0) {
    EXPECT_EQ(why, mi::moss_dit::nvfp4_unsupported_reason(major, minor, false));
  }
  RecordProperty("reason", why);
}

}  // namespace
