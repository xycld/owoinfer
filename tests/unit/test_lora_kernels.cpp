// LoRA side branch kernels: the LoRA-fused GEMMs (bit-identical to their plain counterparts with t = 0; bit-exact
// math on exactly summable operands), the coefficient scaling, compute_t and the side-branch layout (shared fused-qkv
// ranges, strengths, generations), and the INT8 path's unfused norm + rotate.
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/lora/lora_gemm.h"
#include "kernels/lora/lora_kernels.h"
#include "kernels/quant/quant.h"
#include "models/lora/lora.h"
#include "models/lora/side_branch.h"
#include "test_lora_util.h"
#include "test_util.h"
#include "weights/quant/quant_codec.h"

namespace mi::lora {
namespace {

using mi::testing::lora::randn;
using mi::testing::lora::StTensor;
using mi::testing::lora::write_st;
using quant::bf16_to_float;
using quant::float_to_bf16;
using weights::DType;
using U16 = std::vector<std::uint16_t>;

bool has_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

#define REQUIRE_GPU() \
  if (!has_gpu()) GTEST_SKIP() << "no CUDA device"

// A device copy of a host vector with a Tensor view.
struct Dev {
  DeviceBuffer buf;
  Tensor t;
  template <class T>
  Dev(const std::vector<T>& h, mi::DType dt, std::initializer_list<std::int64_t> shape) {
    t = Tensor::view(nullptr, dt, Device::kCUDA, shape);
    buf = DeviceBuffer(std::max<std::size_t>(t.nbytes(), 16));
    if (!h.empty()) buf.upload(h.data(), h.size() * sizeof(T));
    t.data = buf.get();
  }
  template <class T>
  std::vector<T> get() const {
    std::vector<T> h(t.nbytes() / sizeof(T));
    buf.download(h.data(), h.size() * sizeof(T));
    return h;
  }
  __nv_bfloat16* bf() const { return static_cast<__nv_bfloat16*>(t.data); }
};

U16 download_bf16(const Tensor& t) {
  U16 h(static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, h.size() * 2, cudaMemcpyDeviceToHost));
  return h;
}

U16 rand_bf16(std::size_t n, std::uint32_t seed, float scale = 1.0f) { return mi::testing::lora::to_bf16(randn(n, seed, scale)); }

// Small multiples of 2^-shift: GEMM sums of their products are exact in fp32 in any order.
U16 grid_bf16(std::size_t n, std::uint32_t seed, int lo, int hi, int shift) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(lo, hi);
  U16 out(n);
  for (auto& v : out) v = float_to_bf16(std::ldexp(static_cast<float>(d(rng)), -shift));
  return out;
}

float rbf(float v) { return bf16_to_float(float_to_bf16(v)); }


using kernels::fused::GemmTile;
constexpr GemmTile kBf16Tiles[] = {GemmTile::k128x128, GemmTile::k128x256, GemmTile::k256x128};

// acc[m, n] = Σ x·w exactly (grid inputs) as double.
std::vector<double> exact_acc(const U16& x, const U16& w, std::int64_t m, std::int64_t n, std::int64_t k) {
  std::vector<double> acc(static_cast<std::size_t>(m * n));
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      double s = 0;
      for (std::int64_t q = 0; q < k; ++q) s += static_cast<double>(bf16_to_float(x[static_cast<std::size_t>(i * k + q)])) * bf16_to_float(w[static_cast<std::size_t>(j * k + q)]);
      acc[static_cast<std::size_t>(i * n + j)] = s;
    }
  }
  return acc;
}

// The SiTU-GLU reference: v = bf16 [m, 2F] in the natural [hidden | gate] layout, through kernels::situ_glu.
U16 situ_reference(const U16& v, std::int64_t m, std::int64_t f) {
  Dev dv(v, mi::DType::kBF16, {m, 2 * f}), out(U16{}, mi::DType::kBF16, {m, f});
  kernels::situ_glu(dv.t, out.t, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return out.get<std::uint16_t>();
}

using kernels::quant::Int8Tile;
constexpr Int8Tile kInt8Tiles[] = {Int8Tile::k128x128x64, Int8Tile::k128x256x64, Int8Tile::k256x128x64,
                                   Int8Tile::k64x128x64, Int8Tile::k128x128x128};

// ---- LoRA-fused GEMMs (kernels/lora/lora_gemm.h) ----------------------------------------------------------------

namespace fused_test {

constexpr std::int64_t kM2 = 333, kK2 = 512, kN2 = 768, kF2 = 256, kR2 = 40;

// The slices of the plain / gated-residual tests: output columns [0, 256) from t columns [0, 16), none for
// [256, 512), [512, 768) from t columns [0, 8) then [24, 40) (two ranges, the shared fused-qkv layout). SiTU-GLU:
// one slice over all 2F columns from t columns [8, 32).
struct SliceDef {
  std::int64_t col0, cols, col0b, colsb, row0, rows;
};
const SliceDef kSlices[] = {{0, 16, 0, 0, 0, 256}, {0, 8, 24, 16, 512, 256}};
const SliceDef kSituSlice = {8, 24, 0, 0, 0, 2 * kF2};

struct LoraData {
  U16 t;
  std::vector<U16> b;  // per slice [rows, cols]
  std::vector<Dev> db;
  Dev dt;
  kernels::lora::LoraSlices slices;
  LoraData(const std::vector<SliceDef>& defs, bool zero_t, std::uint32_t seed)
      : t(zero_t ? U16(static_cast<std::size_t>(kM2 * kR2), 0) : grid_bf16(kM2 * kR2, seed, -8, 8, 3)),
        dt(t, mi::DType::kBF16, {kM2, kR2}) {
    for (std::size_t i = 0; i < defs.size(); ++i) {
      b.push_back(grid_bf16(static_cast<std::size_t>(defs[i].rows * (defs[i].cols + defs[i].colsb)), seed + 1 + static_cast<std::uint32_t>(i), -8, 8, 4));
    }
    for (std::size_t i = 0; i < defs.size(); ++i) db.push_back(Dev(b[i], mi::DType::kBF16, {defs[i].rows, defs[i].cols + defs[i].colsb}));
    slices.t = dt.bf();
    slices.ldt = kR2;
    slices.count = static_cast<int>(defs.size());
    for (std::size_t i = 0; i < defs.size(); ++i) {
      slices.slice[i] = {defs[i].col0, defs[i].cols, defs[i].col0b, defs[i].colsb, defs[i].row0, defs[i].rows, db[i].bf()};
    }
  }
  // Σ_j t[r, col0 + j] * b[c - row0, j] for GEMM column c (0 outside the slices), exact for grid values.
  double term(const std::vector<SliceDef>& defs, std::int64_t r, std::int64_t c) const {
    for (std::size_t i = 0; i < defs.size(); ++i) {
      const SliceDef& d = defs[i];
      if (c < d.row0 || c >= d.row0 + d.rows) continue;
      double s = 0;
      const std::int64_t kb = d.cols + d.colsb;
      for (std::int64_t j = 0; j < kb; ++j) {
        const std::int64_t col = j < d.cols ? d.col0 + j : d.col0b + (j - d.cols);
        s += static_cast<double>(bf16_to_float(t[static_cast<std::size_t>(r * kR2 + col)])) * bf16_to_float(b[i][static_cast<std::size_t>((c - d.row0) * kb + j)]);
      }
      return s;
    }
    return 0;
  }
};

}  // namespace fused_test

TEST(LoraFusedGemm, Bf16ZeroTIsBitIdenticalAndMathExact) {
  REQUIRE_GPU();
  using namespace fused_test;
  const std::vector<SliceDef> defs(std::begin(kSlices), std::end(kSlices)), situ_defs{kSituSlice};
  // Zero t: the plain variants' bits.
  {
    const U16 x = rand_bf16(kM2 * kK2, 1), w = rand_bf16(kN2 * kK2, 2, 0.05f), wug = rand_bf16(2 * kF2 * kK2, 3, 0.05f),
              gate = rand_bf16(kN2, 4), res = rand_bf16(kM2 * kN2, 5);
    Dev dx(x, mi::DType::kBF16, {kM2, kK2}), dw(w, mi::DType::kBF16, {kN2, kK2}), dug(wug, mi::DType::kBF16, {2 * kF2, kK2}),
        dg(gate, mi::DType::kBF16, {kN2});
    LoraData z(defs, true, 7), zs(situ_defs, true, 8);
    for (GemmTile tile : kBf16Tiles) {
      Dev a(U16{}, mi::DType::kBF16, {kM2, kN2}), b(U16{}, mi::DType::kBF16, {kM2, kN2});
      kernels::fused::set_gemm_tile_override(tile);
      kernels::fused::gemm(a.t, dx.t, dw.t, nullptr);
      kernels::fused::set_gemm_tile_override(GemmTile::kAuto);
      kernels::lora::gemm_lora(b.t, dx.t, dw.t, z.slices, nullptr, tile);
      Dev r1(res, mi::DType::kBF16, {kM2, kN2}), r2(res, mi::DType::kBF16, {kM2, kN2});
      kernels::fused::gemm_gated_residual(r1.t, dx.t, dw.t, dg.t, nullptr, tile);
      kernels::lora::gemm_gated_residual_lora(r2.t, dx.t, dw.t, dg.t, z.slices, nullptr, tile);
      Dev s1(U16{}, mi::DType::kBF16, {kM2, kF2}), s2(U16{}, mi::DType::kBF16, {kM2, kF2});
      kernels::fused::gemm_situ_glu(s1.t, dx.t, dug.t, nullptr, tile);
      kernels::lora::gemm_situ_glu_lora(s2.t, dx.t, dug.t, zs.slices, nullptr, tile);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      EXPECT_EQ(a.get<std::uint16_t>(), b.get<std::uint16_t>()) << "gemm, tile " << static_cast<int>(tile);
      EXPECT_EQ(r1.get<std::uint16_t>(), r2.get<std::uint16_t>()) << "gated residual, tile " << static_cast<int>(tile);
      EXPECT_EQ(s1.get<std::uint16_t>(), s2.get<std::uint16_t>()) << "situ glu, tile " << static_cast<int>(tile);
    }
  }
  // Grid operands: every sum is exact in fp32 in any order, so acc = x·wᵀ + t·bᵀ exactly and the epilogue is checked
  // bit for bit.
  const U16 x = grid_bf16(kM2 * kK2, 11, -8, 8, 3), w = grid_bf16(kN2 * kK2, 12, -8, 8, 4), wug = grid_bf16(2 * kF2 * kK2, 13, -8, 8, 4);
  const U16 gate = rand_bf16(kN2, 16), res = rand_bf16(kM2 * kN2, 17, 4.0f);
  LoraData l(defs, false, 21), ls(situ_defs, false, 31);
  const std::vector<double> acc = exact_acc(x, w, kM2, kN2, kK2), acc_ug = exact_acc(x, wug, kM2, 2 * kF2, kK2);
  U16 want_plain(acc.size()), want_res(acc.size()), v(acc_ug.size());
  for (std::int64_t r = 0; r < kM2; ++r) {
    for (std::int64_t c = 0; c < kN2; ++c) {
      const std::size_t e = static_cast<std::size_t>(r * kN2 + c);
      const float a = static_cast<float>(acc[e] + l.term(defs, r, c));
      want_plain[e] = float_to_bf16(a);
      want_res[e] = float_to_bf16(bf16_to_float(res[e]) + rbf(rbf(a) * bf16_to_float(gate[static_cast<std::size_t>(c)])));
    }
    for (std::int64_t c = 0; c < 2 * kF2; ++c) {  // GEMM column c reads weight row up_gate_gemm_row(c); b is in GEMM order
      const std::int64_t row = kernels::lora::up_gate_gemm_row(c, kF2);
      v[static_cast<std::size_t>(r * 2 * kF2 + row)] = float_to_bf16(static_cast<float>(acc_ug[static_cast<std::size_t>(r * 2 * kF2 + row)] + ls.term(situ_defs, r, c)));
    }
  }
  const U16 want_situ = situ_reference(v, kM2, kF2);
  Dev dx(x, mi::DType::kBF16, {kM2, kK2}), dw(w, mi::DType::kBF16, {kN2, kK2}), dwug(wug, mi::DType::kBF16, {2 * kF2, kK2}),
      dg(gate, mi::DType::kBF16, {kN2});
  for (GemmTile tile : kBf16Tiles) {
    Dev out(U16{}, mi::DType::kBF16, {kM2, kN2}), r(res, mi::DType::kBF16, {kM2, kN2}), s(U16{}, mi::DType::kBF16, {kM2, kF2});
    kernels::lora::gemm_lora(out.t, dx.t, dw.t, l.slices, nullptr, tile);
    kernels::lora::gemm_gated_residual_lora(r.t, dx.t, dw.t, dg.t, l.slices, nullptr, tile);
    kernels::lora::gemm_situ_glu_lora(s.t, dx.t, dwug.t, ls.slices, nullptr, tile);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(out.get<std::uint16_t>(), want_plain) << static_cast<int>(tile);
    EXPECT_EQ(r.get<std::uint16_t>(), want_res) << static_cast<int>(tile);
    EXPECT_EQ(s.get<std::uint16_t>(), want_situ) << static_cast<int>(tile);
  }
  // Slices must cover whole 256-column tiles, ascending, within t.
  kernels::lora::LoraSlices bad = l.slices;
  bad.slice[0].rows = 200;
  Dev out(U16{}, mi::DType::kBF16, {kM2, kN2});
  EXPECT_THROW(kernels::lora::gemm_lora(out.t, dx.t, dw.t, bad, nullptr), std::invalid_argument);
  bad = l.slices;
  bad.slice[1].colsb = 24;  // past ldt
  EXPECT_THROW(kernels::lora::gemm_lora(out.t, dx.t, dw.t, bad, nullptr), std::invalid_argument);
  bad = l.slices;
  bad.slice[0].cols = 0;  // the first range is required
  EXPECT_THROW(kernels::lora::gemm_lora(out.t, dx.t, dw.t, bad, nullptr), std::invalid_argument);
}

TEST(LoraFusedGemm, Int8ZeroTIsBitIdenticalAndMathExact) {
  REQUIRE_GPU();
  using namespace fused_test;
  const std::vector<SliceDef> defs(std::begin(kSlices), std::end(kSlices)), situ_defs{kSituSlice};
  std::mt19937 rng(5);
  const auto ints = [&](std::size_t n, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    std::vector<std::int8_t> v(n);
    for (auto& e : v) e = static_cast<std::int8_t>(d(rng));
    return v;
  };
  // Power-of-two scales and small codes: y = acc * 2^-a * 2^-b + bias and y + t·bᵀ are exact in fp32.
  const auto pow2 = [&](std::size_t n) {
    std::uniform_int_distribution<int> d(6, 8);
    std::vector<float> v(n);
    for (auto& e : v) e = std::ldexp(1.0f, -d(rng));
    return v;
  };
  for (const bool exact : {false, true}) {
    const auto xq = exact ? ints(kM2 * kK2, -8, 8) : ints(kM2 * kK2, -127, 127);
    const auto wq = exact ? ints(kN2 * kK2, -8, 8) : ints(kN2 * kK2, -127, 127);
    const auto wug = exact ? ints(2 * kF2 * kK2, -8, 8) : ints(2 * kF2 * kK2, -127, 127);
    const std::vector<float> xs = exact ? pow2(kM2) : randn(kM2, 41, 0.01f), ws = exact ? pow2(kN2) : randn(kN2, 42, 0.01f),
                             wsug = exact ? pow2(2 * kF2) : randn(2 * kF2, 43, 0.01f);
    std::vector<float> xs_abs = xs, ws_abs = ws, wsug_abs = wsug;
    for (auto* v : {&xs_abs, &ws_abs, &wsug_abs}) {
      for (float& e : *v) e = std::fabs(e) + (exact ? 0.0f : 1e-4f);
    }
    const U16 bias = exact ? grid_bf16(kN2, 44, -8, 8, 6) : rand_bf16(kN2, 44);
    const U16 gate = rand_bf16(kN2, 45), res = rand_bf16(kM2 * kN2, 46, 4.0f);
    LoraData l(defs, !exact, 51), ls(situ_defs, !exact, 61);
    Dev dxq(xq, mi::DType::kI8, {kM2, kK2}), dwq(wq, mi::DType::kI8, {kN2, kK2}), dwug(wug, mi::DType::kI8, {2 * kF2, kK2}),
        dxs(xs_abs, mi::DType::kF32, {kM2}), dws(ws_abs, mi::DType::kF32, {kN2}), dwsug(wsug_abs, mi::DType::kF32, {2 * kF2}),
        db(bias, mi::DType::kBF16, {kN2}), dg(gate, mi::DType::kBF16, {kN2});
    const kernels::quant::Int8Operands op{static_cast<const std::int8_t*>(dxq.t.data), kK2, static_cast<const float*>(dxs.t.data),
                                          static_cast<const std::int8_t*>(dwq.t.data), static_cast<const float*>(dws.t.data), kM2, kN2, kK2};
    const kernels::quant::Int8Operands opug{static_cast<const std::int8_t*>(dxq.t.data), kK2, static_cast<const float*>(dxs.t.data),
                                            static_cast<const std::int8_t*>(dwug.t.data), static_cast<const float*>(dwsug.t.data), kM2, 2 * kF2, kK2};
    U16 want_plain(static_cast<std::size_t>(kM2 * kN2)), want_res(want_plain.size()), v(static_cast<std::size_t>(kM2 * 2 * kF2));
    if (exact) {
      const auto acc_of = [&](const std::vector<std::int8_t>& wt, std::int64_t r, std::int64_t row) {
        std::int64_t s = 0;
        for (std::int64_t q = 0; q < kK2; ++q) s += static_cast<std::int64_t>(xq[static_cast<std::size_t>(r * kK2 + q)]) * wt[static_cast<std::size_t>(row * kK2 + q)];
        return static_cast<double>(s);
      };
      for (std::int64_t r = 0; r < kM2; ++r) {
        for (std::int64_t c = 0; c < kN2; ++c) {
          const std::size_t e = static_cast<std::size_t>(r * kN2 + c);
          const double y = acc_of(wq, r, c) * xs_abs[static_cast<std::size_t>(r)] * ws_abs[static_cast<std::size_t>(c)] + bf16_to_float(bias[static_cast<std::size_t>(c)]);
          const float a = static_cast<float>(y + l.term(defs, r, c));
          want_plain[e] = float_to_bf16(a);
          want_res[e] = float_to_bf16(bf16_to_float(res[e]) + rbf(rbf(a) * bf16_to_float(gate[static_cast<std::size_t>(c)])));
        }
        for (std::int64_t c = 0; c < 2 * kF2; ++c) {
          const std::int64_t row = kernels::lora::up_gate_gemm_row(c, kF2);
          const double y = acc_of(wug, r, row) * xs_abs[static_cast<std::size_t>(r)] * wsug_abs[static_cast<std::size_t>(row)];
          v[static_cast<std::size_t>(r * 2 * kF2 + row)] = float_to_bf16(static_cast<float>(y + ls.term(situ_defs, r, c)));
        }
      }
    }
    const U16 want_situ = exact ? situ_reference(v, kM2, kF2) : U16{};
    for (Int8Tile tile : kInt8Tiles) {
      const int ti = static_cast<int>(tile);
      Dev a(U16{}, mi::DType::kBF16, {kM2, kN2}), r(res, mi::DType::kBF16, {kM2, kN2}), s(U16{}, mi::DType::kBF16, {kM2, kF2});
      kernels::lora::int8_gemm_lora(op, db.bf(), l.slices, a.bf(), kN2, nullptr, tile);
      kernels::lora::int8_gemm_gated_residual_lora(op, db.bf(), dg.bf(), l.slices, r.bf(), kN2, nullptr, tile);
      kernels::lora::int8_gemm_situ_glu_lora(opug, ls.slices, s.bf(), kF2, nullptr, tile);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      if (exact) {
        EXPECT_EQ(a.get<std::uint16_t>(), want_plain) << "int8_gemm_lora, tile " << ti;
        EXPECT_EQ(r.get<std::uint16_t>(), want_res) << "gated residual, tile " << ti;
        EXPECT_EQ(s.get<std::uint16_t>(), want_situ) << "situ glu, tile " << ti;
      } else {  // t = 0: the plain INT8 variants' bits
        Dev a0(U16{}, mi::DType::kBF16, {kM2, kN2}), r0(res, mi::DType::kBF16, {kM2, kN2}), s0(U16{}, mi::DType::kBF16, {kM2, kF2});
        kernels::quant::int8_gemm(op, db.bf(), a0.bf(), kN2, nullptr, tile);
        kernels::quant::int8_gemm_gated_residual(op, db.bf(), dg.bf(), r0.bf(), kN2, nullptr, tile);
        kernels::quant::int8_gemm_situ_glu(opug, s0.bf(), kF2, nullptr, tile);
        MI_CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_EQ(a.get<std::uint16_t>(), a0.get<std::uint16_t>()) << "int8_gemm, tile " << ti;
        EXPECT_EQ(r.get<std::uint16_t>(), r0.get<std::uint16_t>()) << "gated residual, tile " << ti;
        EXPECT_EQ(s.get<std::uint16_t>(), s0.get<std::uint16_t>()) << "situ glu, tile " << ti;
      }
    }
  }
}

TEST(LoraKernels, ScaleColumnsExact) {
  REQUIRE_GPU();
  const std::int64_t m = 77, r = 24, ld = 40;
  const std::vector<float> t32 = randn(static_cast<std::size_t>(m * ld), 31, 3.0f), coef = randn(r, 32, 0.5f);
  Dev a(t32, mi::DType::kF32, {m, ld}), c(coef, mi::DType::kF32, {r}), out(U16(static_cast<std::size_t>(m * ld), 0xFFFF), mi::DType::kBF16, {m, ld});
  kernels::lora::scale_columns_bf16(static_cast<const float*>(a.t.data), ld, static_cast<const float*>(c.t.data), out.bf(), ld, m, r, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const U16 got = out.get<std::uint16_t>();
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < ld; ++j) {
      const std::size_t e = static_cast<std::size_t>(i * ld + j);
      EXPECT_EQ(got[e], j < r ? float_to_bf16(t32[e] * coef[static_cast<std::size_t>(j)]) : std::uint16_t{0xFFFF});
    }
  }
}

// ---- Side branch (models/lora/side_branch.h) ---------------------------------------------------------------------

namespace side_test {

constexpr std::int64_t kD = 3072, kFf = 8192;

std::vector<StTensor> lr(const std::string& p, const char* dn, const char* up, std::int64_t n, std::int64_t k,
                         std::int64_t r, std::uint32_t seed) {
  return {{p + dn, DType::kBF16, {r, k}, randn(static_cast<std::size_t>(r * k), seed, 0.05f)},
          {p + up, DType::kBF16, {n, r}, randn(static_cast<std::size_t>(n * r), seed + 1, 0.05f)}};
}

// ULP of the largest magnitude of a bf16 tensor (the "tensor scale" bound).
float scale_ulp(const std::vector<float>& v) {
  float mx = 0;
  for (float x : v) mx = std::max(mx, std::fabs(x));
  return std::ldexp(1.0f, std::ilogb(mx > 0 ? mx : 1e-30f) - 7);
}

}  // namespace side_test

// compute_t and the LoRA term of every site shape against host math, through gemm_lora on a zero weight (its output is
// then bf16(t · Bᵀ) exactly in slice columns and 0 elsewhere): fused-qkv and separate q adapters stacked (shared +
// own ranges), a k-only qkv site, the permuted ffn_up_gate, ffn_down; then strengths and generations.
TEST(LoraSideBranch, LayoutTermsStrengthsGenerations) {
  REQUIRE_GPU();
  using namespace side_test;
  mi::testing::TempDir dir("lora_side");
  std::vector<StTensor> f1 = lr("diffusion_model.layers.0.proj.qkv", ".lora_down.weight", ".lora_up.weight", 3 * kD, kD, 4, 1);
  for (auto& t : lr("diffusion_model.layers.0.proj.ffn_up_gate", ".lora_down.weight", ".lora_up.weight", 2 * kFf, kD, 8, 3)) f1.push_back(t);
  f1.push_back({"diffusion_model.layers.0.proj.qkv.alpha", DType::kF32, {}, {2.0f}});
  std::vector<StTensor> f2 = lr("transformer.layers.0.proj.q", ".lora_A.weight", ".lora_B.weight", kD, kD, 6, 5);
  for (auto& t : lr("transformer.layers.0.proj.ffn_down", ".lora_A.weight", ".lora_B.weight", kD, kFf, 5, 7)) f2.push_back(t);
  for (auto& t : lr("transformer.layers.1.proj.o", ".lora_A.weight", ".lora_B.weight", kD, kD, 3, 9)) f2.push_back(t);
  for (auto& t : lr("transformer.layers.2.proj.k", ".lora_A.weight", ".lora_B.weight", kD, kD, 2, 11)) f2.push_back(t);
  write_st(dir.path() / "f1.safetensors", f1);
  write_st(dir.path() / "f2.safetensors", f2);
  const LoraSpec specs[] = {{dir.path() / "f1.safetensors", 0.35f}, {dir.path() / "f2.safetensors", 0.6f}};
  LoraStack stack(specs, moss_dit_targets());
  SideBranches sb(stack);
  EXPECT_EQ(sb.device_bytes(), SideBranches::device_bytes(stack));
  const SiteBranch* qkv0 = sb.site(0, Site::kQkv);
  ASSERT_NE(qkv0, nullptr);
  EXPECT_EQ(qkv0->rank, 8 + 8);  // shared fused-qkv range: 4 -> 8; q's own: 6 -> 8; k, v: none
  ASSERT_EQ(qkv0->slices.size(), 3u);
  EXPECT_EQ(qkv0->slices[0].cols, 8);
  EXPECT_EQ(qkv0->slices[0].col0b, 8);
  EXPECT_EQ(qkv0->slices[0].colsb, 8);
  EXPECT_EQ(qkv0->slices[1].colsb, 0);
  ASSERT_NE(sb.site(0, Site::kUpGate), nullptr);
  ASSERT_NE(sb.site(0, Site::kDown), nullptr);
  EXPECT_EQ(sb.site(0, Site::kO), nullptr);
  ASSERT_NE(sb.site(1, Site::kO), nullptr);
  EXPECT_EQ(sb.site(1, Site::kQkv), nullptr);
  ASSERT_NE(sb.site(2, Site::kQkv), nullptr);
  EXPECT_EQ(sb.site(3, Site::kQkv), nullptr);
  EXPECT_EQ(sb.max_rank(), 16);

  kernels::Gemm gemm;
  DeviceBuffer gws(kernels::Gemm::kWorkspaceBytes);
  const std::int64_t m = 200;
  struct Case {
    int block;
    Site site;
  };
  for (const Case c : {Case{0, Site::kQkv}, Case{0, Site::kUpGate}, Case{0, Site::kDown}, Case{1, Site::kO}, Case{2, Site::kQkv}}) {
    const SiteBranch& s = *sb.site(c.block, c.site);
    const char* what = site_name(c.site);
    const U16 x = rand_bf16(static_cast<std::size_t>(m * s.k), 100 + c.block, 1.0f);
    Dev dx(x, mi::DType::kBF16, {m, s.k}), dt(U16{}, mi::DType::kBF16, {m, s.rank});
    DeviceBuffer ws(t_workspace_bytes(m, s.rank));
    compute_t(s, dx.t, dt.t, ws.get(), ws.size(), gemm, gws.get(), gws.size(), nullptr);
    // gemm_lora on a zero weight: out = bf16(0 + t · bᵀ).
    Dev zw(U16(static_cast<std::size_t>(s.n * s.k), 0), mi::DType::kBF16, {s.n, s.k}), out(U16{}, mi::DType::kBF16, {m, s.n});
    kernels::lora::gemm_lora(out.t, dx.t, zw.t, lora_slices(s, dt.t), nullptr);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    // Host math: t = bf16(f32(x · aᵀ) * coef) (sums in double), term = Σ t · b over the slice's ranges.
    const U16 a = download_bf16(s.a), tg = dt.get<std::uint16_t>(), got = out.get<std::uint16_t>();
    std::vector<float> coef(static_cast<std::size_t>(s.rank));
    MI_CUDA_CHECK(cudaMemcpy(coef.data(), s.coef.data, coef.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<float> t_want(tg.size()), t_got(tg.size());
    for (std::int64_t i = 0; i < m; ++i) {
      for (std::int64_t j = 0; j < s.rank; ++j) {
        double acc = 0;
        for (std::int64_t q = 0; q < s.k; ++q) acc += static_cast<double>(bf16_to_float(x[static_cast<std::size_t>(i * s.k + q)])) * bf16_to_float(a[static_cast<std::size_t>(j * s.k + q)]);
        t_want[static_cast<std::size_t>(i * s.rank + j)] = rbf(static_cast<float>(acc) * coef[static_cast<std::size_t>(j)]);
        t_got[static_cast<std::size_t>(i * s.rank + j)] = bf16_to_float(tg[static_cast<std::size_t>(i * s.rank + j)]);
      }
    }
    float t_worst = 0;
    for (std::size_t e = 0; e < t_got.size(); ++e) t_worst = std::max(t_worst, std::fabs(t_got[e] - t_want[e]));
    EXPECT_LE(t_worst, scale_ulp(t_want)) << what << " block " << c.block;  // cuBLASLt's fp32 sums vs double
    std::vector<float> want(static_cast<std::size_t>(m * s.n), 0.0f), gotf(want.size());
    std::vector<bool> covered(static_cast<std::size_t>(s.n), false);
    for (const SiteBranch::Slice& sl : s.slices) {
      const U16 b = download_bf16(sl.b);
      const std::int64_t kb = sl.cols + sl.colsb;
      for (std::int64_t col = 0; col < sl.rows; ++col) covered[static_cast<std::size_t>(sl.row0 + col)] = true;
      for (std::int64_t i = 0; i < m; ++i) {
        for (std::int64_t col = 0; col < sl.rows; ++col) {
          double acc = 0;
          for (std::int64_t j = 0; j < kb; ++j) {
            const std::int64_t tc = j < sl.cols ? sl.col0 + j : sl.col0b + (j - sl.cols);
            acc += static_cast<double>(t_got[static_cast<std::size_t>(i * s.rank + tc)]) * bf16_to_float(b[static_cast<std::size_t>(col * kb + j)]);
          }
          want[static_cast<std::size_t>(i * s.n + sl.row0 + col)] = rbf(static_cast<float>(acc));
        }
      }
    }
    float worst = 0;
    std::size_t nonzero_outside = 0;
    for (std::size_t e = 0; e < got.size(); ++e) {
      gotf[e] = bf16_to_float(got[e]);
      worst = std::max(worst, std::fabs(gotf[e] - want[e]));
      if (!covered[e % static_cast<std::size_t>(s.n)]) nonzero_outside += got[e] != 0;
    }
    EXPECT_LE(worst, scale_ulp(want)) << what << " block " << c.block;  // MMA fp32 sums vs double
    EXPECT_EQ(nonzero_outside, 0u) << what << " block " << c.block;
    if (c.block == 2) {
      EXPECT_EQ(s.slices.size(), 1u);  // k only: q and v columns get no term
    }
  }

  // ffn_up_gate's B rows are stored permuted: row c holds target row up_gate_gemm_row(c, F).
  {
    const SiteBranch& s = *sb.site(0, Site::kUpGate);
    const U16 b = download_bf16(s.slices[0].b);
    const Adapter& ad = *stack.adapters_of("layers.0.proj.ffn_up_gate")[0];
    const std::vector<float> up = stack.read(ad.parts[1]);
    for (std::int64_t c : {0LL, 3LL, 4LL, 7LL, 8LL, 16383LL}) {
      const std::int64_t row = kernels::lora::up_gate_gemm_row(c, kFf);
      EXPECT_EQ(b[static_cast<std::size_t>(c * (s.slices[0].cols + s.slices[0].colsb))], float_to_bf16(up[static_cast<std::size_t>(row * 8)])) << c;
    }
  }

  // Strengths: file 1 off leaves only file 2's sites; both off -> nothing active (plain kernels, bit-identical).
  SideBranches other(stack);
  const std::uint64_t g0 = sb.generation();
  EXPECT_NE(other.generation(), g0);  // process-wide unique, not per instance
  stack.set_strength(0, 0.0f);
  sb.update_strengths(stack);
  EXPECT_NE(sb.generation(), g0);
  EXPECT_NE(sb.generation(), other.generation());
  EXPECT_EQ(sb.site(0, Site::kUpGate), nullptr);
  ASSERT_NE(sb.site(0, Site::kQkv), nullptr);
  {
    std::vector<float> coef(static_cast<std::size_t>(sb.site(0, Site::kQkv)->rank));
    MI_CUDA_CHECK(cudaMemcpy(coef.data(), sb.site(0, Site::kQkv)->coef.data, coef.size() * 4, cudaMemcpyDeviceToHost));
    EXPECT_EQ(coef[0], 0.0f);   // the shared fused-qkv adapter of file 1
    EXPECT_EQ(coef[5], 0.0f);   // padding of the shared range
    EXPECT_EQ(coef[8], 0.6f);   // file 2's q adapter
    EXPECT_EQ(coef[14], 0.0f);  // padding
  }
  const std::uint64_t g1 = sb.generation();
  sb.update_strengths(stack);  // nothing changed: same generation
  EXPECT_EQ(sb.generation(), g1);
  stack.set_strength(1, 0.0f);
  sb.update_strengths(stack);
  EXPECT_FALSE(sb.any_active());
  stack.set_strength(0, 1.0f);
  sb.update_strengths(stack);
  ASSERT_NE(sb.site(0, Site::kUpGate), nullptr);
  EXPECT_EQ(sb.site(1, Site::kO), nullptr);
  // Another stack (even one with the same files) is refused, and nothing is uploaded.
  LoraStack copy(specs, moss_dit_targets());
  EXPECT_THROW(sb.update_strengths(copy), std::invalid_argument);
  ASSERT_NE(sb.site(0, Site::kUpGate), nullptr);
}

// A fused-qkv adapter (one A, B of 3 dim rows) shares one range of t columns across the q, k, v slices; the same
// adapter written as three separate q / k / v adapters with that A triples the rank. Both give the same output.
TEST(LoraSideBranch, SharedFusedQkvEqualsSeparateAdapters) {
  REQUIRE_GPU();
  using namespace side_test;
  mi::testing::TempDir dir("lora_shared");
  const std::int64_t r = 16;
  const std::vector<float> a = randn(static_cast<std::size_t>(r * kD), 21, 0.05f), b = randn(static_cast<std::size_t>(3 * kD * r), 22, 0.05f);
  write_st(dir.path() / "fused.safetensors", {{"diffusion_model.layers.3.proj.qkv.lora_down.weight", DType::kBF16, {r, kD}, a},
                                              {"diffusion_model.layers.3.proj.qkv.lora_up.weight", DType::kBF16, {3 * kD, r}, b}});
  std::vector<StTensor> sep;
  const char* parts[] = {"q", "k", "v"};
  for (int i = 0; i < 3; ++i) {
    const std::string p = std::string("diffusion_model.layers.3.proj.") + parts[i];
    sep.push_back({p + ".lora_down.weight", DType::kBF16, {r, kD}, a});
    sep.push_back({p + ".lora_up.weight", DType::kBF16, {kD, r}, std::vector<float>(b.begin() + i * kD * r, b.begin() + (i + 1) * kD * r)});
  }
  write_st(dir.path() / "sep.safetensors", sep);
  const LoraSpec fs{dir.path() / "fused.safetensors", 0.6f}, ss{dir.path() / "sep.safetensors", 0.6f};
  LoraStack sf(std::span<const LoraSpec>(&fs, 1), moss_dit_targets()), sp(std::span<const LoraSpec>(&ss, 1), moss_dit_targets());
  SideBranches bf(sf), bs(sp);
  const SiteBranch &qf = *bf.site(3, Site::kQkv), &qs = *bs.site(3, Site::kQkv);
  EXPECT_EQ(qf.rank, r);
  EXPECT_EQ(qs.rank, 3 * r);
  std::printf("[lora] fused-qkv adapter, rank %lld: shared layout R %lld (%zu device bytes) vs separate R %lld (%zu bytes)\n",
              static_cast<long long>(r), static_cast<long long>(qf.rank), bf.device_bytes(), static_cast<long long>(qs.rank),
              bs.device_bytes());
  EXPECT_LT(bf.device_bytes(), bs.device_bytes());
  kernels::Gemm gemm;
  DeviceBuffer gws(kernels::Gemm::kWorkspaceBytes);
  const std::int64_t m = 300;
  const U16 x = rand_bf16(static_cast<std::size_t>(m * kD), 23), w = rand_bf16(static_cast<std::size_t>(3 * kD * kD), 24, 0.02f);
  Dev dx(x, mi::DType::kBF16, {m, kD}), dw(w, mi::DType::kBF16, {3 * kD, kD});
  const auto run = [&](const SiteBranch& s) {
    Dev dt(U16{}, mi::DType::kBF16, {m, s.rank}), out(U16{}, mi::DType::kBF16, {m, 3 * kD});
    DeviceBuffer ws(t_workspace_bytes(m, s.rank));
    compute_t(s, dx.t, dt.t, ws.get(), ws.size(), gemm, gws.get(), gws.size(), nullptr);
    kernels::lora::gemm_lora(out.t, dx.t, dw.t, lora_slices(s, dt.t), nullptr);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    return out.get<std::uint16_t>();
  };
  const U16 of = run(qf), os = run(qs);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < of.size(); ++i) diff += of[i] != os[i];
  std::printf("[lora]   outputs: %zu of %zu differ\n", diff, of.size());
  EXPECT_EQ(of, os);
}

// The INT8 path materializes the norm output for a site with a side branch: rms_norm_modulate_moss then
// rotate_quantize must give the fused norm + rotate kernel's bits (same row count per call, as quant.h requires).
TEST(LoraKernels, SplitNormRotateEqualsFused) {
  REQUIRE_GPU();
  constexpr std::int64_t k = 3072;
  const U16 nw = rand_bf16(k, 61, 0.3f), sc = rand_bf16(k, 62, 0.2f), sh = rand_bf16(k, 63, 0.2f);
  Dev dnw(nw, mi::DType::kBF16, {k}), dsc(sc, mi::DType::kBF16, {k}), dsh(sh, mi::DType::kBF16, {k});
  for (const std::int64_t m : {1LL, 37LL, 300LL, 4242LL}) {
    const U16 x = rand_bf16(static_cast<std::size_t>(m * k), 64 + static_cast<std::uint32_t>(m), 6.0f);
    Dev dx(x, mi::DType::kBF16, {m, k}), y(U16{}, mi::DType::kBF16, {m, k});
    Dev q1(std::vector<std::int8_t>{}, mi::DType::kI8, {m, k}), q2(std::vector<std::int8_t>{}, mi::DType::kI8, {m, k});
    Dev s1(std::vector<float>{}, mi::DType::kF32, {m}), s2(std::vector<float>{}, mi::DType::kF32, {m});
    kernels::quant::rms_norm_modulate_rotate_quantize(dx.bf(), k, dnw.bf(), dsc.bf(), dsh.bf(), static_cast<std::int8_t*>(q1.t.data),
                                                      k, static_cast<float*>(s1.t.data), m, static_cast<int>(k), nullptr);
    kernels::rms_norm_modulate_moss(dx.t, dnw.t, dsc.t, dsh.t, y.t, nullptr);
    kernels::quant::rotate_quantize(y.bf(), k, static_cast<std::int8_t*>(q2.t.data), k, static_cast<float*>(s2.t.data), m,
                                    static_cast<int>(k), nullptr);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(q1.get<std::int8_t>(), q2.get<std::int8_t>()) << m << " rows";
    EXPECT_EQ(s1.get<float>(), s2.get<float>()) << m << " rows";
  }
}

}  // namespace
}  // namespace mi::lora
