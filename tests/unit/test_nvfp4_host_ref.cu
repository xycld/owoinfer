// NVFP4 N1, test T5 (design .omc/plans/nvfp4-design.md §4.8): CUTLASS's own definitions against ours, host only.
//   - The scale-factor layout CUTLASS's SM120 block-scaled GEMMs read (Sm1xxBlockScaledConfig<16>, the SFA/SFB layout)
//     is nvfp4_scale_offset, byte for byte, including the padded buffer size.
//   - CUTLASS's host reference GEMM (reference::host::gett_mainloop with GettBlockScalingMainloopParams:
//     sequential fp32 FMA of (a * sfa) * (b * sfb)) on our operands agrees with nvfp4_ref.py's fp64 accumulator
//     (tests/data/nvfp4) to within the fp32 summation bound, both with the operands as CUTLASS orders nibbles (low
//     first) and as stored (hi_first): the in-byte swap permutes K identically for both operands inside each
//     16-element block.
// No device code runs (cpu label).
#include <gtest/gtest.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/util/reference/host/gett.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "util/npy.h"
#include "weights/quant/nvfp4_codec.h"

namespace {

namespace fs = std::filesystem;
namespace q = mi::quant;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/nvfp4";

template <typename T>
std::vector<T> load(const std::string& name) {
  return mi::npy::load(kData / (name + ".npy")).to_vector<T>();
}

using Config = cutlass::detail::Sm1xxBlockScaledConfig<16>;

TEST(Nvfp4CutlassRef, ScaleLayoutIsCutlassSfLayout) {
  for (auto [rows, k] : {std::pair<int, int>{200, 96}, {128, 3072}, {9216, 3072}, {3072, 8192}, {1, 32}, {257, 160}}) {
    const auto layout = Config::tile_atom_to_shape_SFA(cute::make_shape(rows, 64, k, 1));
    const std::int64_t sfk = k / 16;
    EXPECT_EQ(static_cast<std::int64_t>(cute::cosize(layout)), q::nvfp4_scale_rows(rows) * q::nvfp4_scale_cols(sfk)) << rows << "x" << k;
    for (int m = 0; m < rows; ++m) {
      for (int kk = 0; kk < k; ++kk) {
        ASSERT_EQ(static_cast<std::int64_t>(layout(m, kk, 0)), q::nvfp4_scale_offset(m, kk / 16, sfk)) << "m " << m << " k " << kk;
      }
    }
    // SFB uses the same layout over (N, K).
    const auto lb = Config::tile_atom_to_shape_SFB(cute::make_shape(64, rows, k, 1));
    EXPECT_EQ(static_cast<std::int64_t>(lb(rows - 1, k - 1, 0)), q::nvfp4_scale_offset(rows - 1, sfk - 1, sfk));
  }
}

std::vector<std::uint8_t> swap_nibbles(std::vector<std::uint8_t> v) {
  for (auto& b : v) b = static_cast<std::uint8_t>(b << 4 | b >> 4);
  return v;
}

// acc[m][n] by CUTLASS's host reference mainloop; a/b packed [rows, k/2], low nibble = element 0 (CUTLASS order).
std::vector<float> cutlass_acc(std::vector<std::uint8_t>& a, std::vector<std::uint8_t>& sfa, std::vector<std::uint8_t>& b,
                               std::vector<std::uint8_t>& sfb, int m, int n, int k) {
  using namespace cute;
  auto tA = make_tensor(recast_ptr<cutlass::float_e2m1_t>(a.data()), make_layout(make_shape(m, k, 1), make_stride(k, _1{}, m * k)));
  auto tB = make_tensor(recast_ptr<cutlass::float_e2m1_t>(b.data()), make_layout(make_shape(n, k, 1), make_stride(k, _1{}, n * k)));
  auto tSFA = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfa.data()), Config::tile_atom_to_shape_SFA(make_shape(m, n, k, 1)));
  auto tSFB = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfb.data()), Config::tile_atom_to_shape_SFB(make_shape(m, n, k, 1)));
  cutlass::reference::host::GettBlockScalingMainloopParams<float, decltype(tA), decltype(tSFA), decltype(tB), decltype(tSFB)> params{
      tA, tSFA, tB, tSFB};
  constexpr int kBlock = 64;
  std::vector<float> out(static_cast<std::size_t>(m) * n);
  float acc[kBlock][kBlock];
  for (int m0 = 0; m0 < m; m0 += kBlock) {
    for (int n0 = 0; n0 < n; n0 += kBlock) {
      cutlass::reference::host::gett_mainloop(params, m0, n0, 0, acc);
      for (int i = 0; i < kBlock && m0 + i < m; ++i) {
        for (int j = 0; j < kBlock && n0 + j < n; ++j) out[static_cast<std::size_t>(m0 + i) * n + n0 + j] = acc[i][j];
      }
    }
  }
  return out;
}

TEST(Nvfp4CutlassRef, HostGemmMatchesFp64Emulation) {
  const auto ash = mi::npy::load(kData / "act_packed.npy").shape;
  const int m = static_cast<int>(ash[0]), k = static_cast<int>(ash[1] * 2);
  const int n = static_cast<int>(mi::npy::load(kData / "gemm_w_packed.npy").shape[0]);
  auto a = load<std::uint8_t>("act_packed");
  auto sfa = load<std::uint8_t>("act_sf");
  auto b = load<std::uint8_t>("gemm_w_packed");
  auto sfb = load<std::uint8_t>("gemm_w_sf");
  const auto ref = load<double>("gemm_acc");
  ASSERT_EQ(ref.size(), static_cast<std::size_t>(m) * n);

  // sum_k |a_k b_k| per output, from our codec (exact in fp64): the fp32 sequential-sum error bound is k * 2^-24 of it.
  std::vector<float> da(static_cast<std::size_t>(m) * k), db(static_cast<std::size_t>(n) * k);
  q::dequantize_nvfp4_rows(a.data(), sfa.data(), 0, m, k, 1.0f, da.data());
  q::dequantize_nvfp4_rows(b.data(), sfb.data(), 0, n, k, 1.0f, db.data());

  auto lo_a = swap_nibbles(a), lo_b = swap_nibbles(b);
  const std::vector<float> lo = cutlass_acc(lo_a, sfa, lo_b, sfb, m, n, k);
  const std::vector<float> stored = cutlass_acc(a, sfa, b, sfb, m, n, k);
  int nonzero = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double mag = 0;
      for (int kk = 0; kk < k; ++kk) mag += std::fabs(static_cast<double>(da[static_cast<std::size_t>(i) * k + kk]) * db[static_cast<std::size_t>(j) * k + kk]);
      const double tol = k * std::ldexp(1.0, -24) * mag * 1.0001;
      const std::size_t idx = static_cast<std::size_t>(i) * n + j;
      ASSERT_LE(std::fabs(lo[idx] - ref[idx]), tol) << "lo-first (" << i << ", " << j << ")";
      ASSERT_LE(std::fabs(stored[idx] - ref[idx]), tol) << "hi_first as stored (" << i << ", " << j << ")";
      nonzero += ref[idx] != 0;
    }
  }
  EXPECT_GT(nonzero, m * n / 2);
}

}  // namespace
