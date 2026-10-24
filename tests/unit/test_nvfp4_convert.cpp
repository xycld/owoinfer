// NVFP4 N3 (design .omc/plans/nvfp4-design.md §4.4, "N3 converter: reproducible output"): the converter's weight
// recipes (nvfp4_recipes.h) and `moss-infer quantize --format nvfp4` (quant_convert.h) on synthetic DiTs. CPU only,
// except GpuNvfp4Gram (the calibration Hessians' kernel).
// Vectors in tests/data/nvfp4/convert_* come from tools/reference/gen_nvfp4_convert_vectors.py: comfy-kitchen @dba9fac
// eager quantize_nvfp4 (rtn), the converter's mse rule in torch (mse), N2's nvfp4_study.gptq_quant (gptq, statistical).
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "kernels/basic/gram.h"
#include "test_util.h"
#include "util/npy.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/quant/nvfp4_codec.h"
#include "weights/quant/nvfp4_recipes.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_convert.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"

namespace {

namespace fs = std::filesystem;
namespace q = mi::quant;
using mi::weights::DType;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/nvfp4";

template <typename T>
std::vector<T> load(const std::string& name) {
  return mi::npy::load(kData / (name + ".npy")).to_vector<T>();
}
std::vector<std::int64_t> shape_of(const std::string& name) { return mi::npy::load(kData / (name + ".npy")).shape; }

std::vector<float> widen(const std::vector<std::uint16_t>& v) {
  std::vector<float> f(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) f[i] = q::bf16_to_float(v[i]);
  return f;
}

// A vector weight: bf16 bits, fp32 widening, shape.
struct Weight {
  std::vector<std::uint16_t> bf16;
  std::vector<float> f32;
  std::int64_t n = 0, k = 0;
};
Weight vector_weight(const std::string& name) {
  Weight w;
  w.bf16 = load<std::uint16_t>("convert_" + name + "_w");
  w.f32 = widen(w.bf16);
  const auto s = shape_of("convert_" + name + "_w");
  w.n = s[0];
  w.k = s[1];
  return w;
}

// A recipe's file tensors: packed bytes and swizzled scales.
struct Fp4 {
  std::vector<std::uint8_t> packed, sf;
};
Fp4 run_recipe(const Weight& w, float s2, bool mse, int threads = 1) {
  Fp4 o;
  const std::int64_t sfk = w.k / 16;
  o.packed.resize(static_cast<std::size_t>(w.n * w.k / 2));
  std::vector<std::uint8_t> rm(static_cast<std::size_t>(w.n * sfk));
  q::parallel_for(w.n, threads, [&](std::int64_t b, std::int64_t e) {
    q::quantize_nvfp4_rows_recipe(w.f32.data() + b * w.k, e - b, w.k, s2, mse, o.packed.data() + b * (w.k / 2), rm.data() + b * sfk);
  });
  o.sf.resize(static_cast<std::size_t>(q::nvfp4_scale_rows(w.n) * q::nvfp4_scale_cols(sfk)));
  q::swizzle_scales(rm.data(), w.n, sfk, o.sf.data());
  return o;
}

// Dequantized fp32 weights of packed codes and row-major scales under s2.
std::vector<double> dequant(const std::vector<std::uint8_t>& packed, const std::vector<std::uint8_t>& sf_rm, std::int64_t n, std::int64_t k, float s2) {
  std::vector<double> out(static_cast<std::size_t>(n * k));
  for (std::int64_t r = 0; r < n; ++r) {
    for (std::int64_t c = 0; c < k; ++c) {
      const std::uint8_t b = packed[static_cast<std::size_t>(r * k / 2 + c / 2)];
      const std::uint8_t code = c % 2 == 0 ? static_cast<std::uint8_t>(b >> 4) : static_cast<std::uint8_t>(b & 0xF);
      out[static_cast<std::size_t>(r * k + c)] =
          static_cast<double>(q::e2m1_to_float(code)) * static_cast<double>(s2) * q::e4m3_to_float(sf_rm[static_cast<std::size_t>(r * (k / 16) + c / 16)]);
    }
  }
  return out;
}

// GPTQ's proxy loss sum_r (w_r - q_r) H (w_r - q_r)^T, dead columns' weights counted as zero (as GPTQ sets them).
double proxy_loss(const Weight& w, const std::vector<double>& deq, const std::vector<float>& h) {
  const std::int64_t k = w.k;
  double loss = 0;
  std::vector<double> d(static_cast<std::size_t>(k));
  for (std::int64_t r = 0; r < w.n; ++r) {
    for (std::int64_t c = 0; c < k; ++c) {
      const bool dead = h[static_cast<std::size_t>(c * k + c)] == 0.0f;
      d[static_cast<std::size_t>(c)] = (dead ? 0.0 : w.f32[static_cast<std::size_t>(r * k + c)]) - deq[static_cast<std::size_t>(r * k + c)];
    }
    for (std::int64_t i = 0; i < k; ++i) {
      double s = 0;
      for (std::int64_t j = 0; j < k; ++j) s += static_cast<double>(h[static_cast<std::size_t>(std::min(i, j) * k + std::max(i, j))]) * d[static_cast<std::size_t>(j)];
      loss += d[static_cast<std::size_t>(i)] * s;
    }
  }
  return loss;
}

struct GptqOut {
  std::vector<std::uint8_t> packed, sf;  // row-major scales
};
GptqOut run_gptq(const Weight& w, float s2, const std::vector<float>& h, bool mse, int threads) {
  GptqOut o;
  o.packed.resize(static_cast<std::size_t>(w.n * w.k / 2));
  o.sf.resize(static_cast<std::size_t>(w.n * w.k / 16));
  const q::GptqFactor f = q::gptq_factor(h.data(), w.k, threads);
  q::gptq_quantize_rows(w.f32.data(), w.n, w.k, s2, f, mse, threads, o.packed.data(), o.sf.data());
  return o;
}

// ---- Recipes ---------------------------------------------------------------------------------------------------

TEST(Nvfp4Recipes, RtnIsComfyKitchenBitExact) {
  for (const std::string name : {"qkv", "down"}) {
    const Weight w = vector_weight(name);
    const float s2 = load<float>("convert_" + name + "_s2")[0];
    EXPECT_EQ(q::nvfp4_weight_tensor_scale(w.f32.data(), w.n * w.k), s2) << name;
    EXPECT_EQ(q::nvfp4_weight_tensor_scale(w.f32.data(), w.n * w.k, 5), s2) << name;
    const Fp4 o = run_recipe(w, s2, false);
    EXPECT_EQ(o.packed, load<std::uint8_t>("convert_" + name + "_rtn_packed")) << name;
    EXPECT_EQ(o.sf, load<std::uint8_t>("convert_" + name + "_rtn_sf")) << name;
  }
  const float zeros[32] = {};
  EXPECT_EQ(q::nvfp4_weight_tensor_scale(zeros, 32), 1.0f);
}

TEST(Nvfp4Recipes, MseMatchesReferenceAndBeatsRtn) {
  for (const std::string name : {"qkv", "down"}) {
    const Weight w = vector_weight(name);
    const float s2 = load<float>("convert_" + name + "_s2")[0];
    const Fp4 o = run_recipe(w, s2, true);
    EXPECT_EQ(o.packed, load<std::uint8_t>("convert_" + name + "_mse_packed")) << name;
    EXPECT_EQ(o.sf, load<std::uint8_t>("convert_" + name + "_mse_sf")) << name;
    // Never a worse block than RTN (measured here in fp64, where the rule's fp32 differences round), better overall.
    const Fp4 rtn = run_recipe(w, s2, false);
    std::vector<std::uint8_t> sm(static_cast<std::size_t>(w.n * w.k / 16)), sr(sm.size());
    q::unswizzle_scales(o.sf.data(), w.n, w.k / 16, sm.data());
    q::unswizzle_scales(rtn.sf.data(), w.n, w.k / 16, sr.data());
    const auto dm = dequant(o.packed, sm, w.n, w.k, s2), dr = dequant(rtn.packed, sr, w.n, w.k, s2);
    double em = 0, er = 0;
    for (std::int64_t b = 0; b < w.n * w.k / 16; ++b) {
      double bm = 0, br = 0;
      for (std::int64_t i = b * 16; i < b * 16 + 16; ++i) {
        const double x = w.f32[static_cast<std::size_t>(i)];
        bm += (x - dm[static_cast<std::size_t>(i)]) * (x - dm[static_cast<std::size_t>(i)]);
        br += (x - dr[static_cast<std::size_t>(i)]) * (x - dr[static_cast<std::size_t>(i)]);
      }
      ASSERT_LE(bm, br * (1 + 1e-6) + 1e-300) << name << " block " << b;
      em += bm;
      er += br;
    }
    EXPECT_LT(em, er) << name;
    RecordProperty(name + "_mse_over_rtn", std::to_string(em / er));
  }
}

TEST(Nvfp4Recipes, ThreadCountDoesNotChangeBits) {
  const Weight w = vector_weight("qkv");
  const float s2 = load<float>("convert_qkv_s2")[0];
  const auto h = load<float>("convert_qkv_h");
  for (bool mse : {false, true}) {
    const Fp4 a = run_recipe(w, s2, mse, 1), b = run_recipe(w, s2, mse, 5);
    EXPECT_EQ(a.packed, b.packed);
    EXPECT_EQ(a.sf, b.sf);
    const GptqOut g1 = run_gptq(w, s2, h, mse, 1), g7 = run_gptq(w, s2, h, mse, 7);
    EXPECT_EQ(g1.packed, g7.packed) << mse;
    EXPECT_EQ(g1.sf, g7.sf) << mse;
  }
  const q::GptqFactor f1 = q::gptq_factor(h.data(), 256, 1), f4 = q::gptq_factor(h.data(), 256, 4);
  EXPECT_EQ(std::memcmp(f1.u.data(), f4.u.data(), f1.u.size() * sizeof(double)), 0);
}

// U^T U == (damped H)^-1; dead columns flagged.
TEST(Nvfp4Recipes, GptqFactorIsCholeskyOfInverse) {
  const auto h = load<float>("convert_qkv_h");
  const std::int64_t k = 256;
  const q::GptqFactor f = q::gptq_factor(h.data(), k, 3);
  std::vector<double> hd(static_cast<std::size_t>(k * k));
  double diag = 0;
  for (std::int64_t i = 0; i < k; ++i) {
    for (std::int64_t j = 0; j < k; ++j) hd[static_cast<std::size_t>(i * k + j)] = h[static_cast<std::size_t>(std::min(i, j) * k + std::max(i, j))];
    if (hd[static_cast<std::size_t>(i * k + i)] == 0) hd[static_cast<std::size_t>(i * k + i)] = 1;
    diag += hd[static_cast<std::size_t>(i * k + i)];
  }
  for (std::int64_t i = 0; i < k; ++i) hd[static_cast<std::size_t>(i * k + i)] += 0.01 * diag / k;
  for (std::int64_t i = 0; i < k; ++i) EXPECT_EQ(f.dead[static_cast<std::size_t>(i)], i == 5 ? 1 : 0) << i;
  // (U^T U) Hd = I.
  double worst = 0;
  std::vector<double> m(static_cast<std::size_t>(k * k), 0.0);
  for (std::int64_t i = 0; i < k; ++i) {
    for (std::int64_t j = 0; j < k; ++j) {
      double s = 0;
      for (std::int64_t t = 0; t <= std::min(i, j); ++t) s += f.u[static_cast<std::size_t>(t * k + i)] * f.u[static_cast<std::size_t>(t * k + j)];
      m[static_cast<std::size_t>(i * k + j)] = s;
    }
  }
  for (std::int64_t i = 0; i < k; ++i) {
    for (std::int64_t j = 0; j < k; ++j) {
      double s = 0;
      for (std::int64_t t = 0; t < k; ++t) s += m[static_cast<std::size_t>(i * k + t)] * hd[static_cast<std::size_t>(t * k + j)];
      worst = std::max(worst, std::fabs(s - (i == j ? 1.0 : 0.0)));
    }
  }
  EXPECT_LT(worst, 1e-9);
  // Not positive definite: refused.
  std::vector<float> bad(static_cast<std::size_t>(k * k), 0.0f);
  for (std::int64_t i = 0; i < k; ++i) bad[static_cast<std::size_t>(i * k + i)] = 1.0f;
  bad[1] = 50.0f;  // |H01| > sqrt(H00 H11)
  EXPECT_THROW(q::gptq_factor(bad.data(), k, 1), q::QuantError);
  bad[1] = std::nanf("");
  EXPECT_THROW(q::gptq_factor(bad.data(), k, 1), q::QuantError);
}

// The blocked, threaded factor is the header's unblocked algorithm bit for bit. K = 1040: the panel steps take three
// column chunks of 512, the last panel and the trailing tiles are partial.
TEST(Nvfp4Recipes, GptqFactorEqualsUnblockedAlgorithm) {
  const std::int64_t k = 1040, rows = 1100;
  std::vector<float> h(static_cast<std::size_t>(k * k), 0.0f);
  std::uint64_t s = 7;
  std::vector<float> x(static_cast<std::size_t>(rows * k));
  for (auto& v : x) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    v = static_cast<float>(static_cast<std::int64_t>(s >> 40) - (std::int64_t{1} << 23)) / static_cast<float>(1 << 23);
  }
  for (std::int64_t r = 0; r < rows; ++r) x[static_cast<std::size_t>(r * k + 9)] = 0.0f;  // a dead column
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t i = 0; i < k; ++i) {
      for (std::int64_t j = i; j < k; ++j) h[static_cast<std::size_t>(i * k + j)] += x[static_cast<std::size_t>(r * k + i)] * x[static_cast<std::size_t>(r * k + j)];
    }
  }
  // Unblocked: Hd, reverse, right-looking Cholesky, right-looking forward substitution of R^T M = I, reverse.
  std::vector<double> a(static_cast<std::size_t>(k * k));
  auto at = [&](std::int64_t i, std::int64_t j) -> double& { return a[static_cast<std::size_t>(i * k + j)]; };
  double diag = 0;
  for (std::int64_t i = 0; i < k; ++i) {
    for (std::int64_t j = i; j < k; ++j) at(i, j) = at(j, i) = h[static_cast<std::size_t>(i * k + j)];
  }
  for (std::int64_t j = 0; j < k; ++j) {
    if (at(j, j) == 0.0) at(j, j) = 1.0;
    diag = diag + at(j, j);
  }
  const double damp = 0.01 * (diag / static_cast<double>(k));
  for (std::int64_t j = 0; j < k; ++j) at(j, j) = at(j, j) + damp;
  std::reverse(a.begin(), a.end());
  for (std::int64_t p = 0; p < k; ++p) {
    at(p, p) = std::sqrt(at(p, p));
    for (std::int64_t j = p + 1; j < k; ++j) at(p, j) = at(p, j) / at(p, p);
    for (std::int64_t i = p + 1; i < k; ++i) {
      for (std::int64_t j = i; j < k; ++j) at(i, j) = at(i, j) - at(p, i) * at(p, j);
    }
  }
  std::vector<double> rd(static_cast<std::size_t>(k));
  for (std::int64_t i = 0; i < k; ++i) {
    rd[static_cast<std::size_t>(i)] = at(i, i);
    for (std::int64_t c = 0; c < i; ++c) at(i, c) = 0.0;
    at(i, i) = 1.0;
  }
  for (std::int64_t p = 0; p < k; ++p) {
    for (std::int64_t c = 0; c <= p; ++c) at(p, c) = at(p, c) / rd[static_cast<std::size_t>(p)];
    for (std::int64_t i = p + 1; i < k; ++i) {
      for (std::int64_t c = 0; c <= p; ++c) at(i, c) = at(i, c) - at(p, i) * at(p, c);
    }
  }
  std::reverse(a.begin(), a.end());
  for (int threads : {1, 6}) {
    const q::GptqFactor f = q::gptq_factor(h.data(), k, threads);
    for (std::int64_t i = 0; i < k; ++i) {
      for (std::int64_t j = i; j < k; ++j) ASSERT_EQ(f.u[static_cast<std::size_t>(i * k + j)], at(i, j)) << i << "," << j << " threads " << threads;
    }
    EXPECT_EQ(f.dead[9], 1);
  }
}

// With a diagonal Hessian no error propagates: GPTQ is RTN, GPTQ-MSE is MSE, bit for bit.
TEST(Nvfp4Recipes, GptqWithDiagonalHessianIsRtnAndMse) {
  for (const std::string name : {"qkv", "down"}) {
    const Weight w = vector_weight(name);
    const float s2 = load<float>("convert_" + name + "_s2")[0];
    std::vector<float> h(static_cast<std::size_t>(w.k * w.k), 0.0f);
    for (std::int64_t i = 0; i < w.k; ++i) h[static_cast<std::size_t>(i * w.k + i)] = 1.0f + static_cast<float>(i % 7);
    for (bool mse : {false, true}) {
      const GptqOut g = run_gptq(w, s2, h, mse, 2);
      const Fp4 r = run_recipe(w, s2, mse);
      std::vector<std::uint8_t> sf(r.sf.size());
      q::swizzle_scales(g.sf.data(), w.n, w.k / 16, sf.data());
      EXPECT_EQ(g.packed, r.packed) << name << " mse " << mse;
      EXPECT_EQ(sf, r.sf) << name << " mse " << mse;
    }
  }
}

// N2's GPTQ (fp32, lazy batches) and ours (fp64, exact sequential updates) differ only by rounding: almost every code
// agrees, and the proxy loss is the same to well under 1 % (and far below RTN's).
TEST(Nvfp4Recipes, GptqAgreesWithStudyReference) {
  const Weight w = vector_weight("qkv");
  const float s2 = load<float>("convert_qkv_s2")[0];
  const auto h = load<float>("convert_qkv_h");
  const Fp4 rtn = run_recipe(w, s2, false);
  std::vector<std::uint8_t> rtn_sf(static_cast<std::size_t>(w.n * w.k / 16));
  q::unswizzle_scales(rtn.sf.data(), w.n, w.k / 16, rtn_sf.data());
  const double loss_rtn = proxy_loss(w, dequant(rtn.packed, rtn_sf, w.n, w.k, s2), h);
  for (const auto& [mse, tag] : {std::pair{false, "gptq"}, std::pair{true, "gptqm"}}) {
    const GptqOut g = run_gptq(w, s2, h, mse, 4);
    const auto ref_packed = load<std::uint8_t>(std::string("convert_qkv_") + tag + "_packed");
    const auto ref_sf = load<std::uint8_t>(std::string("convert_qkv_") + tag + "_sf");
    std::size_t same = 0, same_sf = 0;
    for (std::size_t i = 0; i < g.packed.size(); ++i) same += (g.packed[i] >> 4) == (ref_packed[i] >> 4) && (g.packed[i] & 0xF) == (ref_packed[i] & 0xF);
    for (std::size_t i = 0; i < g.sf.size(); ++i) same_sf += g.sf[i] == ref_sf[i];
    const double loss = proxy_loss(w, dequant(g.packed, g.sf, w.n, w.k, s2), h);
    const double loss_ref = proxy_loss(w, dequant(ref_packed, ref_sf, w.n, w.k, s2), h);
    const double agree = static_cast<double>(same) / static_cast<double>(g.packed.size());
    RecordProperty(std::string(tag) + "_byte_agreement", std::to_string(agree));
    RecordProperty(std::string(tag) + "_loss_over_reference", std::to_string(loss / loss_ref));
    RecordProperty(std::string(tag) + "_loss_over_rtn", std::to_string(loss / loss_rtn));
    EXPECT_GT(agree, 0.97) << tag;
    EXPECT_GT(static_cast<double>(same_sf) / static_cast<double>(g.sf.size()), 0.97) << tag;
    EXPECT_LT(std::fabs(loss / loss_ref - 1.0), 0.01) << tag << " loss " << loss << " vs " << loss_ref;
    EXPECT_LT(loss, 0.9 * loss_rtn) << tag;
  }
}

TEST(Nvfp4Recipes, NamesAndKeepLists) {
  for (const char* r : {"rtn", "mse", "gptq", "gptq-mse"}) EXPECT_STREQ(q::recipe_name(*q::parse_nvfp4_recipe(r)), r);
  EXPECT_FALSE(q::parse_nvfp4_recipe("gptq_mse"));
  std::string name;
  const auto g09 = q::parse_nvfp4_keep("g0.9", &name);
  EXPECT_EQ(name, "g0.9");
  EXPECT_EQ(g09.size(), 42u);
  EXPECT_TRUE(std::binary_search(g09.begin(), g09.end(), "layers.37.proj.ffn_down"));
  EXPECT_TRUE(std::binary_search(g09.begin(), g09.end(), "layers.0.proj.o"));
  EXPECT_EQ(q::parse_nvfp4_keep("g0.95", &name).size(), 21u);
  EXPECT_TRUE(q::parse_nvfp4_keep("none", &name).empty());
  EXPECT_EQ(name, "none");
  EXPECT_EQ(q::parse_nvfp4_keep("b37.down,layers.2.proj.qkv,b02.qkv,b5.up", &name),
            (std::vector<std::string>{"layers.2.proj.qkv", "layers.37.proj.ffn_down", "layers.5.proj.ffn_up_gate"}));
  EXPECT_EQ(name, "custom");
  for (const char* bad : {"g0.8", "b1.mlp", "layers.1.proj.q", "x", ",", "layers.01.proj.o"}) EXPECT_THROW(q::parse_nvfp4_keep(bad), q::QuantError) << bad;
  EXPECT_EQ(q::gemm_unit_of("layers.3.proj.k")->module, "layers.3.proj.qkv");
  EXPECT_EQ(q::gemm_unit_of("layers.3.proj.ffn_down")->unit, q::GemmUnit::kDown);
  EXPECT_FALSE(q::gemm_unit_of("text_embedder.proj"));
}

// ---- Converter (synthetic DiT, dim 256: q/k/v [256, 256], ffn_up_gate [512, 256], ffn_down [256, 512]) ----------

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

// Deterministic bf16 weights on every platform (no <random> distributions).
std::vector<std::uint16_t> lcg_bf16(std::size_t n, std::uint32_t seed, float scale = 0.02f) {
  std::vector<std::uint16_t> v(n);
  std::uint64_t s = seed * 6364136223846793005ull + 1442695040888963407ull;
  for (auto& x : v) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    const float u = static_cast<float>(static_cast<std::int64_t>(s >> 40) - (std::int64_t{1} << 23)) / static_cast<float>(1 << 23);
    x = q::float_to_bf16(u * scale);
  }
  return v;
}

constexpr std::int64_t kD = 256;

// Layer 0's q/k/v are the rows of convert_qkv_w and its ffn_down convert_down_w, so their nvfp4 bytes can be compared
// with the vectors; everything else is LCG data.
void write_tiny_dit(const fs::path& path, std::uint32_t first_seed = 1) {
  const Weight qkv = vector_weight("qkv"), down = vector_weight("down");
  std::vector<T> ts;
  std::uint32_t seed = first_seed;
  for (int b = 0; b < 2; ++b) {
    const std::string p = "layers." + std::to_string(b) + ".proj.";
    for (int part = 0; part < 3; ++part) {
      std::vector<std::uint16_t> w = b == 0 ? std::vector<std::uint16_t>(qkv.bf16.begin() + part * kD * kD, qkv.bf16.begin() + (part + 1) * kD * kD)
                                            : lcg_bf16(static_cast<std::size_t>(kD * kD), seed++);
      ts.push_back(make(p + "qkv"[part], DType::kBF16, {kD, kD}, w));
      ts.back().name += ".weight";
    }
    ts.push_back(make(p + "o.weight", DType::kBF16, {kD, kD}, lcg_bf16(static_cast<std::size_t>(kD * kD), seed++)));
    ts.push_back(make(p + "ffn_up_gate.weight", DType::kBF16, {2 * kD, kD}, lcg_bf16(static_cast<std::size_t>(2 * kD * kD), seed++)));
    ts.push_back(make(p + "ffn_down.weight", DType::kBF16, {kD, 2 * kD}, b == 0 ? down.bf16 : lcg_bf16(static_cast<std::size_t>(2 * kD * kD), seed++)));
    ts.push_back(make("layers." + std::to_string(b) + ".norm1.weight", DType::kBF16, {kD}, lcg_bf16(kD, seed++, 1.0f)));
  }
  ts.push_back(make("text_embedder.proj.weight", DType::kBF16, {kD, kD}, lcg_bf16(static_cast<std::size_t>(kD * kD), seed++)));
  write_st(path, ts);
}

std::vector<std::byte> tensor_bytes(const fs::path& file, const std::string& name) {
  const mi::weights::SafetensorsFile f(file);
  const auto* t = f.find(name);
  if (t == nullptr) return {};
  std::vector<std::byte> b(static_cast<std::size_t>(t->nbytes));
  mi::weights::FileReader(file).read_at(t->offset, b.data(), b.size());
  return b;
}
template <typename V>
std::vector<V> as(const std::vector<std::byte>& b) {
  std::vector<V> v(b.size() / sizeof(V));
  std::memcpy(v.data(), b.data(), v.size() * sizeof(V));
  return v;
}

std::string hex64(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

constexpr const char* kInt8GoldenXxh64 = "37a2aebebf895dfc";
constexpr const char* kInt8FusedGoldenXxh64 = "b629567f58fe7c2a";

std::uint64_t file_xxh64(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::vector<char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return mi::xxh64(b.data(), b.size());
}

// A deterministic Hessian source: each unit gets X^T X of LCG rows seeded by its name (layer 0 qkv: convert_qkv_h),
// delivered in reverse order.
q::HessianCapture fake_capture(const std::vector<q::HessianUnit>& units, const q::HessianSink& sink, std::vector<std::string>* seen) {
  const auto hq = load<float>("convert_qkv_h");
  for (std::size_t i = units.size(); i-- > 0;) {
    const q::HessianUnit& u = units[i];
    if (seen != nullptr) seen->push_back(u.module);
    if (u.module == "layers.0.proj.qkv") {
      sink(i, hq.data());
      continue;
    }
    const std::int64_t k = u.k, rows = 2 * k;
    const auto x = widen(lcg_bf16(static_cast<std::size_t>(rows * k), static_cast<std::uint32_t>(mi::xxh64(u.module.data(), u.module.size())), 1.0f));
    std::vector<float> h(static_cast<std::size_t>(k * k), 0.0f);
    for (std::int64_t r = 0; r < rows; ++r) {
      for (std::int64_t a = 0; a < k; ++a) {
        for (std::int64_t b = a; b < k; ++b) h[static_cast<std::size_t>(a * k + b)] += x[static_cast<std::size_t>(r * k + a)] * x[static_cast<std::size_t>(r * k + b)];
      }
    }
    sink(i, h.data());
  }
  return {R"({"set": "test"})", R"({"gpu_arch": "none"})"};
}

TEST(Nvfp4Convert, SyntheticDitLayoutRoundTripAndBits) {
  const mi::testing::AnyModelShapes any_shapes;
  mi::testing::TempDir dir("nvfp4_convert");
  write_tiny_dit(dir.path() / "transformer" / "diffusion_pytorch_model.safetensors");
  q::ConvertOptions opt;
  opt.threads = 3;
  opt.nvfp4 = q::Nvfp4Options{};
  opt.nvfp4->recipe = q::Nvfp4Recipe::kRtn;
  opt.nvfp4->keep = q::parse_nvfp4_keep("b01.o,b1.down", &opt.nvfp4->keep_name);
  const fs::path out = dir.path() / "rtn.safetensors";
  const auto st = q::convert(dir.path(), out, opt);
  EXPECT_EQ(st.quantized, 12);
  EXPECT_EQ(st.nvfp4, 10);  // q/k/v count as three

  // Layout as BennyDaBall's MOSS-Image-SFT-NVFP4: fused qkv, U8 / F8_E4M3 / F32 [] / comfy_quant JSON.
  const mi::weights::SafetensorsFile f(out);
  const auto* w = f.find("layers.0.proj.qkv.weight");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->dtype, DType::kU8);
  EXPECT_EQ(w->shape, (std::vector<std::int64_t>{3 * kD, kD / 2}));
  EXPECT_EQ(f.find("layers.0.proj.qkv.weight_scale")->dtype, DType::kF8E4M3);
  EXPECT_EQ(f.find("layers.0.proj.qkv.weight_scale")->shape, (std::vector<std::int64_t>{3 * kD, kD / 16}));
  EXPECT_EQ(f.find("layers.0.proj.qkv.weight_scale_2")->dtype, DType::kF32);
  EXPECT_TRUE(f.find("layers.0.proj.qkv.weight_scale_2")->shape.empty());
  EXPECT_EQ(f.find("layers.0.proj.q.weight"), nullptr);
  const auto cfg = tensor_bytes(out, "layers.0.proj.ffn_down.comfy_quant");
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(cfg.data()), cfg.size()), R"({"format": "nvfp4"})");
  const auto kept_cfg = tensor_bytes(out, "layers.1.proj.o.comfy_quant");
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(kept_cfg.data()), kept_cfg.size()),
            R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})");
  EXPECT_EQ(f.find("layers.1.proj.qkv.weight")->dtype, DType::kU8);
  EXPECT_EQ(f.find("layers.1.proj.ffn_down.weight")->dtype, DType::kI8);
  // Tensor order: dtype descending (F32, BF16, F8_E4M3, I8, U8), then name.
  const auto rank = [](DType t) { return t == DType::kF32 ? 4 : t == DType::kBF16 ? 3 : t == DType::kF8E4M3 ? 2 : t == DType::kI8 ? 1 : 0; };
  for (std::size_t i = 1; i < f.tensors().size(); ++i) {
    const auto &a = f.tensors()[i - 1], &b = f.tensors()[i];
    EXPECT_TRUE(rank(a.dtype) > rank(b.dtype) || (rank(a.dtype) == rank(b.dtype) && a.name < b.name)) << a.name << " / " << b.name;
  }
  EXPECT_EQ(f.metadata().at("mossinfer_nvfp4"),
            R"j({"converter_version": 1, "recipe": "rtn", "tensor_scale": "amax / 2688 (fp32)", "keep": "custom", "kept_int8": ["layers.1.proj.ffn_down", "layers.1.proj.o"]})j");
  EXPECT_EQ(f.metadata().count("mossinfer_nvfp4_calibration"), 0u);

  // rtn bytes are comfy-kitchen's (the vectors), the fused qkv under one tensor scale.
  EXPECT_EQ(as<std::uint8_t>(tensor_bytes(out, "layers.0.proj.qkv.weight")), load<std::uint8_t>("convert_qkv_rtn_packed"));
  EXPECT_EQ(as<std::uint8_t>(tensor_bytes(out, "layers.0.proj.qkv.weight_scale")), load<std::uint8_t>("convert_qkv_rtn_sf"));
  EXPECT_EQ(as<float>(tensor_bytes(out, "layers.0.proj.qkv.weight_scale_2")), load<float>("convert_qkv_s2"));
  EXPECT_EQ(as<std::uint8_t>(tensor_bytes(out, "layers.0.proj.ffn_down.weight")), load<std::uint8_t>("convert_down_rtn_packed"));
  EXPECT_EQ(as<std::uint8_t>(tensor_bytes(out, "layers.0.proj.ffn_down.weight_scale")), load<std::uint8_t>("convert_down_rtn_sf"));

  // Round trip through our importer: specs, the engine view's validation, and the dequantized weights.
  const auto m = q::open_quant_model(out);
  EXPECT_EQ(m.find_linear("layers.0.proj.k")->spec, q::QuantSpec::nvfp4());
  EXPECT_TRUE(m.find_linear("layers.0.proj.k")->fused_slice);
  EXPECT_EQ(m.find_linear("layers.1.proj.o")->spec, q::QuantSpec::int8_convrot(256));
  EXPECT_NO_THROW(q::to_model_index(m));
  const Weight qkv = vector_weight("qkv");
  std::vector<float> rows(static_cast<std::size_t>(kD * kD)), ref(rows.size());
  q::dequantize_nvfp4_linear_rows(m, *m.find_linear("layers.0.proj.v"), 0, kD, rows.data(), nullptr);
  const auto pk = load<std::uint8_t>("convert_qkv_rtn_packed"), sf = load<std::uint8_t>("convert_qkv_rtn_sf");
  q::dequantize_nvfp4_rows(pk.data() + 2 * kD * kD / 2, sf.data(), 2 * kD, kD, kD, load<float>("convert_qkv_s2")[0], ref.data());
  EXPECT_EQ(rows, ref);

  // mse: the reference bytes; the thread count changes nothing.
  opt.nvfp4->recipe = q::Nvfp4Recipe::kMse;
  const fs::path m1 = dir.path() / "mse1.safetensors", m7 = dir.path() / "mse7.safetensors";
  opt.threads = 1;
  q::convert(dir.path(), m1, opt);
  opt.threads = 7;
  q::convert(dir.path(), m7, opt);
  EXPECT_EQ(file_xxh64(m1), file_xxh64(m7));
  EXPECT_EQ(as<std::uint8_t>(tensor_bytes(m1, "layers.0.proj.qkv.weight")), load<std::uint8_t>("convert_qkv_mse_packed"));
  EXPECT_EQ(as<std::uint8_t>(tensor_bytes(m1, "layers.0.proj.qkv.weight_scale")), load<std::uint8_t>("convert_qkv_mse_sf"));
  EXPECT_EQ(as<std::uint8_t>(tensor_bytes(m1, "layers.0.proj.ffn_down.weight")), load<std::uint8_t>("convert_down_mse_packed"));
}

TEST(Nvfp4Convert, GptqThroughHessianSource) {
  const mi::testing::AnyModelShapes any_shapes;
  mi::testing::TempDir dir("nvfp4_convert_gptq");
  write_tiny_dit(dir.path() / "transformer" / "diffusion_pytorch_model.safetensors");
  q::ConvertOptions opt;
  opt.nvfp4 = q::Nvfp4Options{};
  opt.nvfp4->recipe = q::Nvfp4Recipe::kGptqMse;
  opt.nvfp4->keep = q::parse_nvfp4_keep("b1.o", &opt.nvfp4->keep_name);
  std::vector<std::string> asked;
  opt.nvfp4->hessians = [&](const std::vector<q::HessianUnit>& units, const q::HessianSink& sink) {
    for (const auto& u : units) EXPECT_EQ(u.k, u.unit == q::GemmUnit::kDown ? 2 * kD : kD) << u.module;
    return fake_capture(units, sink, &asked);
  };
  std::map<int, std::uint64_t> hashes;
  for (int threads : {1, 4}) {
    opt.threads = threads;
    asked.clear();
    const fs::path out = dir.path() / ("gptq" + std::to_string(threads) + ".safetensors");
    q::convert(dir.path(), out, opt);
    hashes[threads] = file_xxh64(out);
    // Every nvfp4 unit asked for once (not the kept layers.1.proj.o).
    EXPECT_EQ(std::set<std::string>(asked.begin(), asked.end()),
              (std::set<std::string>{"layers.0.proj.qkv", "layers.0.proj.o", "layers.0.proj.ffn_up_gate", "layers.0.proj.ffn_down", "layers.1.proj.qkv",
                                     "layers.1.proj.ffn_up_gate", "layers.1.proj.ffn_down"}));
    EXPECT_EQ(asked.size(), 7u);
    // Layer 0's qkv: the recipe on the vector weights and Hessian.
    const Weight qkv = vector_weight("qkv");
    const GptqOut g = run_gptq(qkv, load<float>("convert_qkv_s2")[0], load<float>("convert_qkv_h"), true, 2);
    std::vector<std::uint8_t> sf(static_cast<std::size_t>(3 * kD * kD / 16));
    q::swizzle_scales(g.sf.data(), 3 * kD, kD / 16, sf.data());
    EXPECT_EQ(as<std::uint8_t>(tensor_bytes(out, "layers.0.proj.qkv.weight")), g.packed);
    EXPECT_EQ(as<std::uint8_t>(tensor_bytes(out, "layers.0.proj.qkv.weight_scale")), sf);
    const mi::weights::SafetensorsFile f(out);
    EXPECT_EQ(f.metadata().at("mossinfer_nvfp4_calibration"), R"({"set": "test"})");
    EXPECT_EQ(f.metadata().at("mossinfer_nvfp4_capture_class"), R"({"gpu_arch": "none"})");
    EXPECT_NE(f.metadata().at("mossinfer_nvfp4").find(R"("recipe": "gptq-mse")"), std::string::npos);
    EXPECT_NO_THROW(q::to_model_index(q::open_quant_model(out)));
  }
  EXPECT_EQ(hashes[1], hashes[4]);

  // The Hessian digest: of exactly the units used, in (block, unit) order.
  const std::string hdigest = mi::weights::SafetensorsFile(dir.path() / "gptq1.safetensors").metadata().at("mossinfer_nvfp4_hessian_xxh64");
  EXPECT_EQ(hdigest.size(), 16u);
  EXPECT_NE(hdigest, std::string(16, '0'));

  // Capture to a Hessian file, solve from it: the units convert() asks for, and the same bytes as the direct run.
  const auto units = q::nvfp4_hessian_units(dir.path(), opt);
  std::vector<std::string> names;
  for (const auto& u : units) names.push_back(u.module);
  EXPECT_EQ(names, (std::vector<std::string>{"layers.0.proj.qkv", "layers.0.proj.o", "layers.0.proj.ffn_up_gate", "layers.0.proj.ffn_down",
                                             "layers.1.proj.qkv", "layers.1.proj.ffn_up_gate", "layers.1.proj.ffn_down"}));
  const fs::path hfile = dir.path() / "hessians.safetensors";
  const q::HessianFileInfo info = q::write_hessian_file(dir.path(), hfile, opt);
  EXPECT_EQ(info.units, 7u);
  EXPECT_EQ(info.capture.calibration, R"({"set": "test"})");
  EXPECT_FALSE(fs::exists(fs::path(hfile) += ".partial"));
  q::ConvertOptions from_file = opt;
  from_file.threads = 2;
  from_file.nvfp4->hessians = nullptr;
  from_file.nvfp4->hessian_file = hfile;
  const fs::path out = dir.path() / "from_file.safetensors";
  const auto st = q::convert(dir.path(), out, from_file);
  EXPECT_EQ(file_xxh64(out), hashes[1]);
  EXPECT_EQ(hex64(st.source_xxh64), hex64(info.dit_xxh64));
  // A file captured over more units (keep none) serves a smaller set: the same bytes as capturing that set directly.
  q::ConvertOptions all = opt;
  all.nvfp4->keep.clear();
  const fs::path hall = dir.path() / "hessians_all.safetensors";
  EXPECT_EQ(q::write_hessian_file(dir.path(), hall, all).units, 8u);
  from_file.nvfp4->hessian_file = hall;
  q::convert(dir.path(), out, from_file);
  EXPECT_EQ(file_xxh64(out), hashes[1]);
  // A unit the file does not hold is refused; so is -o naming the Hessian file, and a file of another DiT.
  from_file.nvfp4->hessian_file = hfile;
  from_file.nvfp4->keep.clear();
  EXPECT_THROW(q::convert(dir.path(), dir.path() / "x.safetensors", from_file), q::QuantError);
  from_file.nvfp4->keep = opt.nvfp4->keep;
  EXPECT_THROW(q::convert(dir.path(), hfile, from_file), q::QuantError);
  mi::testing::TempDir other("nvfp4_convert_gptq_other");
  write_tiny_dit(other.path() / "transformer" / "diffusion_pytorch_model.safetensors", 77);
  try {
    q::convert(other.path(), other.path() / "x.safetensors", from_file);
    ADD_FAILURE() << "a Hessian file of another DiT was accepted";
  } catch (const q::QuantError& e) {
    EXPECT_NE(std::string(e.what()).find("captured from another DiT"), std::string::npos) << e.what();
  }
  // The Hessian file's path is checked like an output: not into the input directory, not over an input file.
  EXPECT_THROW(q::write_hessian_file(dir.path(), dir.path() / "transformer" / "h.safetensors", opt), q::QuantError);
  EXPECT_THROW(q::write_hessian_file(dir.path(), dir.path() / "transformer" / "diffusion_pytorch_model.safetensors", opt), q::QuantError);
}

TEST(Nvfp4Convert, RefusesMisuse) {
  const mi::testing::AnyModelShapes any_shapes;
  mi::testing::TempDir dir("nvfp4_convert_bad");
  write_tiny_dit(dir.path() / "transformer" / "diffusion_pytorch_model.safetensors");
  const fs::path out = dir.path() / "x.safetensors";
  q::ConvertOptions base;
  base.nvfp4 = q::Nvfp4Options{};
  base.nvfp4->keep.clear();
  auto expect_fail = [&](q::ConvertOptions o, const char* what, const fs::path& in) {
    try {
      q::convert(in, out, o);
      ADD_FAILURE() << "accepted: " << what;
    } catch (const q::QuantError& e) {
      EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
    }
    EXPECT_FALSE(fs::exists(out));
  };
  expect_fail(base, "needs calibration Hessians", dir.path());  // gptq-mse (the default) without a capture
  q::ConvertOptions o = base;
  o.nvfp4->recipe = q::Nvfp4Recipe::kGptq;
  o.nvfp4->hessians = [](const std::vector<q::HessianUnit>& units, const q::HessianSink& sink) { return fake_capture(units, sink, nullptr); };
  o.loras = {{dir.path() / "lora.safetensors", 1.0f}};
  expect_fail(o, "LoRAs cannot be merged", dir.path());
  o = base;
  o.nvfp4->recipe = q::Nvfp4Recipe::kGptq;
  o.nvfp4->hessians = [](const std::vector<q::HessianUnit>& units, const q::HessianSink& sink) {
    for (std::size_t i = 1; i < units.size(); ++i) {
      std::vector<float> h(static_cast<std::size_t>(units[i].k * units[i].k), 0.0f);
      for (std::int64_t j = 0; j < units[i].k; ++j) h[static_cast<std::size_t>(j * units[i].k + j)] = 1.0f;
      sink(i, h.data());
    }
    return q::HessianCapture{};
  };
  expect_fail(o, "delivered no Hessian", dir.path());
  // An output that cannot be created fails before the capture runs.
  bool captured = false;
  o.nvfp4->hessians = [&](const std::vector<q::HessianUnit>& units, const q::HessianSink& sink) {
    captured = true;
    return fake_capture(units, sink, nullptr);
  };
  EXPECT_ANY_THROW(q::convert(dir.path(), dir.path() / "missing_dir" / "x.safetensors", o));
  EXPECT_FALSE(captured);
  o = base;
  o.nvfp4->recipe = q::Nvfp4Recipe::kRtn;
  o.nvfp4->hessian_file = dir.path() / "h.safetensors";
  expect_fail(o, "takes no Hessian file", dir.path());
  o.nvfp4->hessian_file.clear();
  o.nvfp4->keep = {"layers.7.proj.o"};
  expect_fail(o, "is not in the model", dir.path());
  EXPECT_THROW(q::nvfp4_hessian_units(dir.path(), o), q::QuantError);  // the same planning as convert()
  o.nvfp4->keep = {"layers.0.proj.o"};
  o.group = 1024;  // K = 256 is no multiple of it
  expect_fail(o, "not a multiple of the ConvRot group", dir.path());
  o.group = 256;
  o.nvfp4->keep.clear();
  o.keep_bf16 = {"layers.0.proj.q"};  // splits the fused q/k/v
  expect_fail(o, "differ in spec", dir.path());
  EXPECT_THROW(q::nvfp4_hessian_units(dir.path(), o), q::QuantError);
  o.keep_bf16.clear();
  o.policy = [](const q::QuantLinear&) { return q::QuantSpec::bf16(); };
  expect_fail(o, "cannot be combined", dir.path());
  // A quantized input: NVFP4 needs the bf16 weights.
  q::ConvertOptions int8;
  const fs::path int8_file = dir.path() / "int8.safetensors";
  q::convert(dir.path(), int8_file, int8);
  o = base;
  o.nvfp4->recipe = q::Nvfp4Recipe::kRtn;
  expect_fail(o, "NVFP4 output is converted from the bf16 weights", int8_file);
}

// The INT8 converter's bytes are pinned (the NVFP4 work must leave them as they were: this hash is the base commit's,
// 920bcf5, for this portable synthetic DiT).
TEST(Nvfp4Convert, Int8OutputUnchanged) {
  const mi::testing::AnyModelShapes any_shapes;
  mi::testing::TempDir dir("nvfp4_convert_int8");
  write_tiny_dit(dir.path() / "transformer" / "diffusion_pytorch_model.safetensors");
  q::ConvertOptions opt;
  opt.threads = 3;
  const fs::path out = dir.path() / "int8.safetensors";
  q::convert(dir.path(), out, opt);
  EXPECT_EQ(hex64(file_xxh64(out)), kInt8GoldenXxh64);
  opt.fuse_qkv = true;
  opt.keep_bf16 = {"layers.1.proj.ffn_*"};
  q::convert(dir.path(), out, opt);
  EXPECT_EQ(hex64(file_xxh64(out)), kInt8FusedGoldenXxh64);
}

// ---- Calibration Hessian kernel ----------------------------------------------------------------------------------

// gram_accumulate_upper against the defined arithmetic: per element one fmaf chain over the rows from +0, then one
// add into h; two calls accumulate; tiles below the diagonal untouched.
TEST(GpuNvfp4Gram, MatchesSequentialFmaBitExact) {
  const std::int64_t k = 192;
  const std::int64_t ms[2] = {77, 300};
  std::vector<float> ref(static_cast<std::size_t>(k * k), 0.0f);
  for (std::int64_t i = 0; i < k; ++i) {
    for (std::int64_t j = 0; j < k; ++j) ref[static_cast<std::size_t>(i * k + j)] = (i / 64 > j / 64) ? 7.0f : 0.0f;  // sentinel below
  }
  void* dh = nullptr;
  ASSERT_EQ(cudaMalloc(&dh, ref.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dh, ref.data(), ref.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  for (int call = 0; call < 2; ++call) {
    const std::int64_t m = ms[call];
    const auto x = lcg_bf16(static_cast<std::size_t>(m * k), static_cast<std::uint32_t>(9 + call), 3.0f);
    void* dx = nullptr;
    ASSERT_EQ(cudaMalloc(&dx, x.size() * 2), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    mi::kernels::gram_accumulate_upper(mi::Tensor::view(dx, mi::DType::kBF16, mi::Device::kCUDA, {m, k}), static_cast<float*>(dh), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaFree(dx), cudaSuccess);
    for (std::int64_t i = 0; i < k; ++i) {
      for (std::int64_t j = 0; j < k; ++j) {
        if (i / 64 > j / 64) continue;
        float acc = 0.0f;
        for (std::int64_t r = 0; r < m; ++r) acc = std::fma(q::bf16_to_float(x[static_cast<std::size_t>(r * k + i)]), q::bf16_to_float(x[static_cast<std::size_t>(r * k + j)]), acc);
        ref[static_cast<std::size_t>(i * k + j)] = ref[static_cast<std::size_t>(i * k + j)] + acc;
      }
    }
  }
  std::vector<float> got(ref.size());
  ASSERT_EQ(cudaMemcpy(got.data(), dh, got.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(dh), cudaSuccess);
  EXPECT_EQ(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)), 0);
  for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], ref[i]) << i / k << "," << i % k;
}

}  // namespace
