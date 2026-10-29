// kernels/nvfp4 (NVFP4 N4; design .omc/plans/nvfp4-design.md §4.1, §4.2, §4.8):
//   T3   the activation quantizers (all four DiT sites) bit for bit against the CPU codec (quantize_nvfp4_act_bf16,
//        itself pinned to tests/data/nvfp4, the T2 vectors) and against the engine's op-by-op chains
//        (rms_norm_modulate_moss / situ_glu, then the codec): non-finite rows, M tails, strided rows, the zero scale
//        padding (every byte of the scale buffer is written). Any GPU.
//   T6   the dispatch guard: where available() is false the GEMMs throw std::logic_error (after argument checks).
//   G1a  GEMM on small-integer codes with unit scales, bit for bit against CUTLASS's host reference (79a's method).
//   G1b  GEMM at the DiT shapes against an fp64 oracle: >= 99.9 % of the bf16 outputs equal the fp32 epilogue of the
//        exact sum and none is more than 1 ULP away; the pre-round value, read back with a second E1L pass
//        (delta = -y), has rel-L2 vs fp64 within (2e-6)^2 of the same two-pass readout of the exact sum.
//   G1c  every tile / schedule gives the same bits; E1L / E2L with delta = 0 equal E1 / E2; E1L / E2L with a
//        per-column constant delta (the same for every row) equal E1 / E2 with that bias; E2 equals residual +
//        bf16(E1 * gate) computed from E1's output; padded residual and delta row strides.
//   G1d  hi_first and lo_first packing (nibbles swapped in both operands) give the same bits.
//   G1e  the quantizers' fp32-compare E2M1 rounding vs cvt.rn.satfinite.e2m1x2.f32: identical codes.
//   G2   accumulator characterization against fp64: cancellation within and across MMA K steps, scale extremes,
//        subnormal block scales.
// G1/G2 skip where available() is false. Plus per-site rates vs the INT8 path (quantizer included) and the tile
// sweep behind nvfp4_tile_for, skipped unless MI_RUN_PERF=1.
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/util/reference/host/gett.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/norm.h"
#include "kernels/nvfp4/nvfp4.h"
#include "kernels/quant/quant.h"
#include "util/npy.h"
#include "weights/quant/nvfp4_codec.h"
#include "weights/quant/quant_codec.h"

namespace {

namespace nv = mi::kernels::nvfp4;
namespace q = mi::quant;
using mi::DType;
using mi::Tensor;
using nv::Nvfp4Tile;
using bf16_t = __nv_bfloat16;

const std::filesystem::path kData = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/data/nvfp4";

constexpr std::int64_t kM1024 = 2 * 4096 + 282 + 10;  // DiT rows at 1024^2 with CFG (two images + text + pad)
constexpr Nvfp4Tile kTiles[] = {Nvfp4Tile::k128x128x128, Nvfp4Tile::k128x128x128Pingpong, Nvfp4Tile::k256x128x128};

template <class T>
struct Buf {
  mi::DeviceBuffer b;
  std::size_t n = 0;
  explicit Buf(std::size_t count) : b(std::max<std::size_t>(count, 1) * sizeof(T)), n(count) {}
  explicit Buf(const std::vector<T>& h) : Buf(h.size()) { upload(h); }
  T* get() const { return static_cast<T*>(b.get()); }
  void upload(const std::vector<T>& h) { b.upload(h.data(), h.size() * sizeof(T)); }
  std::vector<T> download() const {
    std::vector<T> h(n);
    b.download(h.data(), n * sizeof(T));
    return h;
  }
  void fill_bytes(int v) { MI_CUDA_CHECK(cudaMemset(b.get(), v, std::max<std::size_t>(n, 1) * sizeof(T))); }
};

bf16_t* as_bf16(const Buf<std::uint16_t>& b) { return reinterpret_cast<bf16_t*>(b.get()); }

Tensor view_bf16(void* p, std::int64_t rows, std::int64_t cols) {
  return Tensor::view(p, DType::kBF16, mi::Device::kCUDA, {rows, cols});
}
Tensor vec_bf16(void* p, std::int64_t n) { return Tensor::view(p, DType::kBF16, mi::Device::kCUDA, {n}); }
// [rows, cols] with row stride ld.
Tensor strided_bf16(void* p, std::int64_t rows, std::int64_t cols, std::int64_t ld) {
  Tensor t = view_bf16(p, rows, cols);
  t.stride[0] = ld;
  return t;
}

std::uint64_t mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::uint16_t bf16_bits(float v) { return q::float_to_bf16(v); }
float bf16_float(std::uint16_t b) { return q::bf16_to_float(b); }

// DiT-like bf16 activations [rows, cols]: hashed +-m * 2^(c_exp - 7 - e) with m in [128, 255], e in 0..7 and a
// per-channel exponent c_exp in -3..2, plus outlier channels (every 397th) 64x larger (as test_quant_kernels.cu).
std::vector<std::uint16_t> synth_act(std::uint64_t seed, std::int64_t rows, std::int64_t cols) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows * cols));
  for (std::int64_t c = 0; c < cols; ++c) {
    const int c_exp = static_cast<int>(mix(static_cast<std::uint64_t>(c) * 7919 + seed) % 6) - 3 + (c % 397 == 5 ? 6 : 0);
    for (std::int64_t r = 0; r < rows; ++r) {
      const std::uint64_t z = mix(static_cast<std::uint64_t>(r * cols + c) + seed * 0x9E3779B97F4A7C15ull);
      const float m = static_cast<float>(128 + (z & 0x7F));
      const int e = static_cast<int>((z >> 7) & 7);
      out[static_cast<std::size_t>(r * cols + c)] = bf16_bits(std::ldexp((z >> 10) & 1 ? -m : m, c_exp - 7 - e));
    }
  }
  return out;
}

std::vector<std::uint16_t> synth_vec(std::uint64_t seed, std::size_t n, float center, float spread) {
  std::vector<std::uint16_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double u = static_cast<double>(mix(i + seed * 0x9E3779B97F4A7C15ull) >> 11) / 9007199254740992.0;
    out[i] = bf16_bits(static_cast<float>(center + spread * (2.0 * u - 1.0)));
  }
  return out;
}

// Device fills for the large operands: kind 2 bf16 uniform in [a - b, a + b]; kind 3 activations as synth_act (same
// hash family, not the same values).
__device__ std::uint64_t mix_d(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__global__ void fill_kernel(bf16_t* p, std::size_t n, std::uint64_t seed, int kind, float a, float b, int cols) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint64_t z = mix_d(i + seed * 0x9E3779B97F4A7C15ull);
    const float u = static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
    float v = a + b * (2.0f * u - 1.0f);
    if (kind == 3) {
      const int c = static_cast<int>(i % static_cast<std::size_t>(cols));
      const int c_exp = static_cast<int>(mix_d(static_cast<std::uint64_t>(c) * 7919 + seed) % 6) - 3 + (c % 397 == 5 ? 6 : 0);
      const float m = static_cast<float>(128 + (z & 0x7F));
      v = ldexpf((z >> 10) & 1 ? -m : m, c_exp - 7 - static_cast<int>((z >> 7) & 7));
    }
    p[i] = __float2bfloat16_rn(v);
  }
}

Buf<std::uint16_t> dev_act(std::uint64_t seed, std::int64_t rows, std::int64_t cols) {
  Buf<std::uint16_t> b(static_cast<std::size_t>(rows * cols));
  fill_kernel<<<1024, 256>>>(as_bf16(b), b.n, seed, 3, 0, 0, static_cast<int>(cols));
  MI_CUDA_CHECK_LAUNCH();
  return b;
}
Buf<std::uint16_t> dev_uniform(std::uint64_t seed, std::size_t n, float center, float spread) {
  Buf<std::uint16_t> b(n);
  fill_kernel<<<1024, 256>>>(as_bf16(b), n, seed, 2, center, spread, 1);
  MI_CUDA_CHECK_LAUNCH();
  return b;
}

template <class T>
std::size_t count_diff(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += !(a[i] == b[i]);
  return n;
}

std::size_t count_diff_bits(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
  return n;
}

int ulp_dist(std::uint16_t a, std::uint16_t b) {
  const auto ord = [](std::uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7FFF) : static_cast<int>(v); };
  return std::abs(ord(a) - ord(b));
}

// Rows at and around every tile boundary near both ends and the middle.
std::vector<int> sample_rows(std::int64_t m) {
  std::vector<int> r;
  for (std::int64_t v : {std::int64_t{0}, std::int64_t{1}, std::int64_t{63}, std::int64_t{64}, std::int64_t{127},
                         std::int64_t{128}, std::int64_t{129}, std::int64_t{255}, std::int64_t{256}, m / 3, m / 2,
                         m - 257, m - 129, m - 128, m - 2, m - 1}) {
    if (v >= 0 && v < m && std::find(r.begin(), r.end(), static_cast<int>(v)) == r.end()) r.push_back(static_cast<int>(v));
  }
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
// Activation quantizers (T3).

// A quantized activation on the device: q [m, k / 2], the full swizzled scale buffer, g [m].
struct DevAct {
  std::int64_t m, k;
  Buf<std::uint8_t> q, sf;
  Buf<float> g;
  DevAct(std::int64_t m_, std::int64_t k_)
      : m(m_), k(k_), q(static_cast<std::size_t>(m_ * k_ / 2)), sf(static_cast<std::size_t>(nv::sf_bytes(m_, k_))),
        g(static_cast<std::size_t>(m_)) {
    // Garbage everywhere: the quantizer must write every byte (scale padding included).
    q.fill_bytes(0xA5);
    sf.fill_bytes(0x7F);
    g.fill_bytes(0xFF);
  }
  nv::Nvfp4Act act() const { return {q.get(), sf.get(), g.get()}; }
};

struct HostAct {
  std::vector<std::uint8_t> q, sf;
  std::vector<float> g;
};

HostAct download(const DevAct& d) {
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return {d.q.download(), d.sf.download(), d.g.download()};
}

// The CPU codec (the reference) on bf16 rows [m, k].
HostAct codec(const std::vector<std::uint16_t>& x, std::int64_t m, std::int64_t k) {
  HostAct h;
  h.q.resize(static_cast<std::size_t>(m * k / 2));
  h.sf.resize(static_cast<std::size_t>(nv::sf_bytes(m, k)));
  h.g.resize(static_cast<std::size_t>(m));
  q::quantize_nvfp4_act_bf16(x.data(), m, k, h.q.data(), h.sf.data(), h.g.data());
  return h;
}

void expect_same(const HostAct& got, const HostAct& ref, const std::string& label) {
  EXPECT_EQ(count_diff(got.q, ref.q), 0u) << label << ": codes";
  EXPECT_EQ(count_diff(got.sf, ref.sf), 0u) << label << ": scales (padding included)";
  EXPECT_EQ(count_diff_bits(got.g, ref.g), 0u) << label << ": row scales";
}

HostAct run_quantize(const std::vector<std::uint16_t>& x, std::int64_t m, std::int64_t k, std::int64_t ldx) {
  std::vector<std::uint16_t> xp(static_cast<std::size_t>(std::max<std::int64_t>(m * ldx, 1)), 0x7FC0);  // NaN padding
  for (std::int64_t r = 0; r < m; ++r) std::copy_n(x.begin() + r * k, k, xp.begin() + r * ldx);
  Buf<std::uint16_t> xd(xp);
  DevAct out(m, k);
  nv::quantize_nvfp4(as_bf16(xd), ldx, out.act(), m, static_cast<int>(k), nullptr);
  return download(out);
}

// Rows with every special case of the non-finite rule and the scale recipe, then DiT-like rows.
std::vector<std::uint16_t> special_rows(std::uint64_t seed, std::int64_t m, std::int64_t k) {
  std::vector<std::uint16_t> x = synth_act(seed, m, k);
  const auto row = [&](std::int64_t r) { return x.begin() + r * k; };
  const auto set = [&](std::int64_t r, std::int64_t c, std::uint16_t v) {
    if (r < m && c < k) x[static_cast<std::size_t>(r * k + c)] = v;
  };
  for (std::int64_t r = 0; r < std::min<std::int64_t>(m, 12); ++r) {
    switch (r) {
      case 0: std::fill_n(row(r), k, std::uint16_t{0}); break;          // all zero: g = 1, codes +0
      case 1: std::fill_n(row(r), k, std::uint16_t{0x7FC0}); break;     // all NaN: as all zero
      case 2: set(r, 5, 0x7F80); break;                                  // +inf: clamped to bf16 max, saturates
      case 3: set(r, 0, 0xFF80); set(r, 17, 0x7FC0); set(r, 18, 0xFFC1); break;  // -inf and NaNs
      case 4:                                                            // bf16 subnormals only: subnormal g
        for (std::int64_t c = 0; c < k; ++c) x[static_cast<std::size_t>(r * k + c)] = static_cast<std::uint16_t>((c % 127 + 1) | (c % 3 == 0 ? 0x8000 : 0));
        break;
      case 5:                                                            // near the bf16 maximum
        for (std::int64_t c = 0; c < k; ++c) x[static_cast<std::size_t>(r * k + c)] = static_cast<std::uint16_t>(0x7F7F - (c % 9) | (c % 2 ? 0x8000 : 0));
        break;
      case 6: std::fill_n(row(r), k, std::uint16_t{0}); set(r, k - 1, 0x0001); break;  // one subnormal element
      case 7: std::fill_n(row(r), k, std::uint16_t{0x8000}); break;     // -0.0 everywhere
      case 8:                                                            // whole zero blocks among large values
        for (std::int64_t c = 0; c < k; ++c) {
          if ((c / 16) % 3 == 1) x[static_cast<std::size_t>(r * k + c)] = 0;
        }
        break;
      case 9:                                                            // one huge outlier: most blocks get sf 0
        set(r, k / 2, bf16_bits(3.0e30f));
        break;
      case 10:                                                           // exact E2M1 midpoints relative to the scale
        for (std::int64_t c = 0; c < k; ++c) {
          static const float kMid[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 6.0f};
          x[static_cast<std::size_t>(r * k + c)] = bf16_bits((c % 2 ? -1.0f : 1.0f) * kMid[c % 8]);
        }
        break;
      default: break;
    }
  }
  return x;
}

TEST(GpuNvfp4Quant, MatchesT2Vectors) {
  const auto xa = mi::npy::load(kData / "act_x.npy");
  const std::int64_t m = xa.shape[0], k = xa.shape[1];
  const HostAct got = run_quantize(xa.to_vector<std::uint16_t>(), m, k, k);
  EXPECT_EQ(count_diff(got.q, mi::npy::load(kData / "act_packed.npy").to_vector<std::uint8_t>()), 0u);
  EXPECT_EQ(count_diff(got.sf, mi::npy::load(kData / "act_sf.npy").to_vector<std::uint8_t>()), 0u);
  EXPECT_EQ(count_diff_bits(got.g, mi::npy::load(kData / "act_g.npy").to_vector<float>()), 0u);
}

TEST(GpuNvfp4Quant, QuantizeMatchesCodec) {
  struct Case {
    std::int64_t m, k, ldx;
  };
  const Case cases[] = {{1, 32, 32},     {12, 32, 40},     {31, 96, 96},     {127, 160, 168}, {128, 3072, 3072},
                        {129, 3072, 3136}, {200, 8192, 8192}, {257, 16384, 16384}, {3, 32768, 32768}, {kM1024, 3072, 3072}};
  for (const Case& c : cases) {
    const auto x = special_rows(static_cast<std::uint64_t>(c.m * 131 + c.k), c.m, c.k);
    expect_same(run_quantize(x, c.m, c.k, c.ldx), codec(x, c.m, c.k),
                "m " + std::to_string(c.m) + " k " + std::to_string(c.k) + " ldx " + std::to_string(c.ldx));
  }
}

struct NormVecs {
  Buf<std::uint16_t> weight, scale, shift;
  explicit NormVecs(std::int64_t k, std::uint64_t seed)
      : weight(synth_vec(seed, static_cast<std::size_t>(k), 1.0f, 0.5f)),
        scale(synth_vec(seed + 1, static_cast<std::size_t>(k), 0.0f, 1.0f)),
        shift(synth_vec(seed + 2, static_cast<std::size_t>(k), 0.0f, 1.0f)) {}
};

// A quantizer's optional bf16 copy (row stride ld) against the op-by-op tensor (contiguous [m, k]): equal bits, the
// padding columns still 0x7FC1.
void expect_copy(const std::vector<std::uint16_t>& want, const std::vector<std::uint16_t>& got, std::int64_t m,
                 std::int64_t k, std::int64_t ld, const std::string& what) {
  std::size_t bad = 0;
  for (std::int64_t r = 0; r < m; ++r) {
    for (std::int64_t c = 0; c < ld; ++c) {
      const std::uint16_t g = got[static_cast<std::size_t>(r * ld + c)];
      bad += c < k ? g != want[static_cast<std::size_t>(r * k + c)] : g != 0x7FC1;
    }
  }
  EXPECT_EQ(bad, 0u) << what;
}

// qkv and ffn_up_gate sites: the fused norm + modulate + quantize against kernels::rms_norm_modulate_moss into bf16,
// then the codec, at every row count class (the norm's reduce order depends on m: m = 1, 2, 7, 12 and >= 16 give 512,
// 256, 128, 64 and 32 threads per row at k = 3072).
TEST(GpuNvfp4Quant, FusedNormMatchesOpByOp) {
  const std::int64_t k = 3072;
  const NormVecs v(k, 77);
  for (std::int64_t m : {std::int64_t{1}, std::int64_t{2}, std::int64_t{7}, std::int64_t{12}, std::int64_t{33}, std::int64_t{64},
                         std::int64_t{255}, std::int64_t{256}, std::int64_t{1000}, kM1024}) {
    auto x = special_rows(static_cast<std::uint64_t>(m), m, k);
    if (m > 1) std::fill_n(x.begin(), k, std::uint16_t{0});  // a zero row: the norm gives shift, not NaN
    const std::int64_t ldx = m % 2 ? k + 64 : k;
    std::vector<std::uint16_t> xp(static_cast<std::size_t>(m * ldx), 0x7FC0);
    for (std::int64_t r = 0; r < m; ++r) std::copy_n(x.begin() + r * k, k, xp.begin() + r * ldx);
    Buf<std::uint16_t> xd(xp), y(static_cast<std::size_t>(m * k));
    DevAct out(m, k);
    nv::rms_norm_modulate_quantize_nvfp4(as_bf16(xd), ldx, as_bf16(v.weight), as_bf16(v.scale), as_bf16(v.shift),
                                         out.act(), m, static_cast<int>(k), nullptr);
    const Tensor xt = strided_bf16(xd.get(), m, k, ldx);
    mi::kernels::rms_norm_modulate_moss(xt, vec_bf16(v.weight.get(), k), vec_bf16(v.scale.get(), k),
                                        vec_bf16(v.shift.get(), k), view_bf16(y.get(), m, k), nullptr);
    // The same call storing its bf16 norm output (row stride k + 8): y bit for bit, the padding untouched, the same codes.
    const std::int64_t ldy = k + 8;
    Buf<std::uint16_t> y2(std::vector<std::uint16_t>(static_cast<std::size_t>(m * ldy), 0x7FC1));
    DevAct out2(m, k);
    nv::rms_norm_modulate_quantize_nvfp4(as_bf16(xd), ldx, as_bf16(v.weight), as_bf16(v.scale), as_bf16(v.shift),
                                         out2.act(), m, static_cast<int>(k), nullptr, as_bf16(y2), ldy);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    expect_same(download(out), codec(y.download(), m, k), "m " + std::to_string(m));
    expect_same(download(out2), download(out), "with y, m " + std::to_string(m));
    expect_copy(y.download(), y2.download(), m, k, ldy, "m " + std::to_string(m));
  }
}

// ffn_down site: situ_glu + quantize against kernels::situ_glu into bf16, then the codec.
TEST(GpuNvfp4Quant, SituGluMatchesOpByOp) {
  struct Case {
    std::int64_t m, f, ld;
  };
  for (const Case& c : {Case{1, 32, 64}, Case{37, 96, 200}, Case{129, 8192, 16384}, Case{kM1024, 8192, 16384}}) {
    // DiT-like magnitudes plus every special row in both halves.
    const auto x = special_rows(static_cast<std::uint64_t>(c.m * 7 + c.f), c.m, 2 * c.f);
    std::vector<std::uint16_t> xp(static_cast<std::size_t>(c.m * c.ld), 0x7FC0);
    for (std::int64_t r = 0; r < c.m; ++r) std::copy_n(x.begin() + r * 2 * c.f, 2 * c.f, xp.begin() + r * c.ld);
    Buf<std::uint16_t> xd(xp), y(static_cast<std::size_t>(c.m * c.f));
    DevAct out(c.m, c.f);
    nv::situ_glu_quantize_nvfp4(as_bf16(xd), c.ld, out.act(), c.m, static_cast<int>(c.f), nullptr);
    const Tensor xt = strided_bf16(xd.get(), c.m, 2 * c.f, c.ld);
    mi::kernels::situ_glu(xt, view_bf16(y.get(), c.m, c.f), nullptr);
    // The same call storing its bf16 SiTU-GLU output (row stride F + 8): ffn bit for bit, the same codes.
    const std::int64_t ldf = c.f + 8;
    Buf<std::uint16_t> y2(std::vector<std::uint16_t>(static_cast<std::size_t>(c.m * ldf), 0x7FC1));
    DevAct out2(c.m, c.f);
    nv::situ_glu_quantize_nvfp4(as_bf16(xd), c.ld, out2.act(), c.m, static_cast<int>(c.f), nullptr, as_bf16(y2), ldf);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    const std::string what = "m " + std::to_string(c.m) + " F " + std::to_string(c.f);
    expect_same(download(out), codec(y.download(), c.m, c.f), what);
    expect_same(download(out2), download(out), "with ffn, " + what);
    expect_copy(y.download(), y2.download(), c.m, c.f, ldf, what);
  }
}

TEST(GpuNvfp4Quant, RejectsBadArguments) {
  Buf<std::uint16_t> x(4096);
  DevAct out(8, 256);
  const auto bad = [&](auto&& fn) { EXPECT_THROW(fn(), std::invalid_argument); };
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 48, out.act(), 8, 48, nullptr); });   // k % 32
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 64, out.act(), 8, 0, nullptr); });    // k = 0
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 60, out.act(), 8, 32, nullptr); });   // ldx % 8
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 16, out.act(), 8, 32, nullptr); });   // ldx < k
  bad([&] { nv::quantize_nvfp4(as_bf16(x) + 1, 32, out.act(), 8, 32, nullptr); });  // misaligned x
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 32, out.act(), -1, 32, nullptr); });
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 65536, out.act(), 1, 65536, nullptr); });  // k > kMaxQuantizeK
  nv::Nvfp4Act a = out.act();
  a.sf += 4;
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 32, a, 8, 32, nullptr); });
  a = out.act();
  a.scale = nullptr;
  bad([&] { nv::quantize_nvfp4(as_bf16(x), 32, a, 8, 32, nullptr); });
  bad([&] { nv::situ_glu_quantize_nvfp4(as_bf16(x), 32, out.act(), 8, 32, nullptr); });  // ld < 2F
  bad([&] { nv::situ_glu_quantize_nvfp4(as_bf16(x), 64, out.act(), 8, 32, nullptr, as_bf16(x), 24); });  // ldf < F
  bad([&] { nv::situ_glu_quantize_nvfp4(as_bf16(x), 64, out.act(), 8, 32, nullptr, as_bf16(x) + 1, 32); });  // misaligned
  bad([&] {  // the norm output's row stride not a multiple of 8
    nv::rms_norm_modulate_quantize_nvfp4(as_bf16(x), 256, as_bf16(x), as_bf16(x), as_bf16(x), out.act(), 8, 256, nullptr,
                                         as_bf16(x), 260);
  });
  bad([&] {
    nv::rms_norm_modulate_quantize_nvfp4(as_bf16(x), 256, nullptr, as_bf16(x), as_bf16(x), out.act(), 8, 256, nullptr);
  });
  bad([&] {  // below kernels::rms_norm_moss's 128-element row minimum
    nv::rms_norm_modulate_quantize_nvfp4(as_bf16(x), 64, as_bf16(x), as_bf16(x), as_bf16(x), out.act(), 8, 64, nullptr);
  });
  // m = 0 is a no-op, even with null pointers.
  EXPECT_NO_THROW(nv::quantize_nvfp4(nullptr, 32, nv::Nvfp4Act{}, 0, 32, nullptr));
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM operands and the fp64 oracle.

// One GEMM problem on the device: activations quantized by quantize_nvfp4 (T3-verified), a weight made of the same
// quantizer's codes and scales of a uniform bf16 matrix (its per-row scales unused) with a per-tensor s2.
struct GemmData {
  std::int64_t m, n, k;
  DevAct x, w;
  float s2;
  Buf<std::uint16_t> bias, gate;
  GemmData(std::int64_t m_, std::int64_t n_, std::int64_t k_, std::uint64_t seed, float s2_ = 3.0e-4f)
      : m(m_), n(n_), k(k_), x(m_, k_), w(n_, k_), s2(s2_),
        bias(dev_uniform(seed + 4, static_cast<std::size_t>(n_), 0.0f, 2.0f)),
        gate(dev_uniform(seed + 5, static_cast<std::size_t>(n_), 0.0f, 1.5f)) {
    {
      Buf<std::uint16_t> xb = dev_act(seed, m, k);
      nv::quantize_nvfp4(as_bf16(xb), k, x.act(), m, static_cast<int>(k), nullptr);
      Buf<std::uint16_t> wb = dev_uniform(seed + 1, static_cast<std::size_t>(n * k), 0.0f, 0.05f);
      nv::quantize_nvfp4(as_bf16(wb), k, w.act(), n, static_cast<int>(k), nullptr);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
  nv::Nvfp4Operands ops() const {
    nv::Nvfp4Operands o;
    o.x = x.q.get();
    o.x_sf = x.sf.get();
    o.x_scale = x.g.get();
    o.w = w.q.get();
    o.w_sf = w.sf.get();
    o.w_scale2 = s2;
    o.m = m;
    o.n = n;
    o.k = k;
    return o;
  }
};

__device__ double e2m1_d(unsigned c) {
  const double v[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  return (c & 8) ? -v[c & 7] : v[c & 7];
}
__device__ double e4m3_d(unsigned c) {
  const unsigned e = c >> 3, m = c & 7;
  return e == 0 ? m * 0.001953125 : ldexp(1.0 + m / 8.0, static_cast<int>(e) - 7);
}
__device__ std::int64_t sf_off_d(std::int64_t row, std::int64_t kb, std::int64_t col_tiles) {
  return ((row / 128) * col_tiles + kb / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + kb % 4;
}

// acc[i, c] = sum_kb x_sf * w_sf * sum_j qx * qw for row rows[i], in fp64 (every product is exact in fp64).
__global__ void oracle_acc_kernel(const std::uint8_t* x, const std::uint8_t* xsf, const std::uint8_t* w,
                                  const std::uint8_t* wsf, int k, const int* rows, int nrows, int n, double* acc) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(nrows) * n) return;
  const int i = static_cast<int>(idx / n), c = static_cast<int>(idx % n);
  const std::int64_t r = rows[i];
  const std::int64_t col_tiles = (k / 16 + 3) / 4;
  double sum = 0;
  for (int kb = 0; kb < k / 16; ++kb) {
    const std::uint8_t* xb = x + r * (k / 2) + kb * 8;
    const std::uint8_t* wb = w + static_cast<std::int64_t>(c) * (k / 2) + kb * 8;
    double dot = 0;
    for (int j = 0; j < 8; ++j) dot += e2m1_d(xb[j] >> 4) * e2m1_d(wb[j] >> 4) + e2m1_d(xb[j] & 15) * e2m1_d(wb[j] & 15);
    sum += e4m3_d(xsf[sf_off_d(r, kb, col_tiles)]) * e4m3_d(wsf[sf_off_d(c, kb, col_tiles)]) * dot;
  }
  acc[idx] = sum;
}

std::vector<double> oracle_acc(const GemmData& d, const std::vector<int>& rows) {
  Buf<int> r(rows);
  Buf<double> acc(rows.size() * static_cast<std::size_t>(d.n));
  oracle_acc_kernel<<<static_cast<unsigned>((acc.n + 127) / 128), 128>>>(d.x.q.get(), d.x.sf.get(), d.w.q.get(),
                                                                          d.w.sf.get(), static_cast<int>(d.k), r.get(),
                                                                          static_cast<int>(rows.size()),
                                                                          static_cast<int>(d.n), acc.get());
  MI_CUDA_CHECK_LAUNCH();
  return acc.download();
}

// Rows `rows` of a device bf16 matrix with row stride ld.
std::vector<std::uint16_t> download_rows(const std::uint16_t* p, std::int64_t ld, std::int64_t cols,
                                         const std::vector<int>& rows) {
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint16_t> h(rows.size() * static_cast<std::size_t>(cols));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    MI_CUDA_CHECK(cudaMemcpy(h.data() + i * static_cast<std::size_t>(cols), p + rows[i] * ld,
                             static_cast<std::size_t>(cols) * 2, cudaMemcpyDeviceToHost));
  }
  return h;
}

// The documented fp32 epilogue on an fp32 accumulator, one rounding per op (host code; CMakeLists.txt compiles this
// file with -ffp-contract=off / /fp:precise, so no FMA contraction).
float epi_value(float acc, float xs, float s2, float delta, float bias, bool with_delta) {
  float v = acc * xs;
  v = v * s2;
  if (with_delta) v = v + delta;
  return v + bias;
}

__global__ void negate_kernel(const std::uint16_t* in, std::uint16_t* out, std::size_t n) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    out[i] = in[i] ^ 0x8000u;
  }
}

__global__ void swap_nibbles_kernel(std::uint8_t* p, std::size_t n) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    p[i] = static_cast<std::uint8_t>(p[i] << 4 | p[i] >> 4);
  }
}

#define MI_REQUIRE_NVFP4()                                                                                   \
  do {                                                                                                       \
    if (!nv::available()) GTEST_SKIP() << "nvfp4::available() is false (needs cc 12.0 and the sm_120a build)"; \
  } while (0)

// ---------------------------------------------------------------------------------------------------------------
// T6: dispatch guard.

TEST(GpuNvfp4Dispatch, GuardMatchesDevice) {
  int dev = 0, major = 0, minor = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
  const bool avail = nv::available();
  EXPECT_EQ(avail, nv::available());  // cached, stable
  if (major != 12 || minor != 0) EXPECT_FALSE(avail) << "cc " << major << "." << minor;
  RecordProperty("available", avail ? "yes" : "no");
  EXPECT_NE(nv::nvfp4_tile_for(kM1024, 3072, 3072), Nvfp4Tile::kAuto);

  GemmData d(64, 64, 64, 9);
  Buf<std::uint16_t> out(64 * 64), res(64 * 64);
  Buf<float> f(16);
  Buf<std::uint8_t> codes(16);
  // Invalid arguments are refused first (std::invalid_argument), whatever the device.
  nv::Nvfp4Operands bad = d.ops();
  bad.k = 48;
  EXPECT_THROW(nv::nvfp4_gemm(bad, nullptr, as_bf16(out), 64, nullptr), std::invalid_argument);
  EXPECT_THROW(nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(out), 60, nullptr), std::invalid_argument);
  EXPECT_THROW(nv::nvfp4_gemm_gated_residual(d.ops(), nullptr, nullptr, as_bf16(res), 64, nullptr),
               std::invalid_argument);
  EXPECT_THROW(nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(out), 64, nullptr, Nvfp4Tile::kAuto,
                              nv::Nvfp4Delta{as_bf16(res), 4}),
               std::invalid_argument);
  if (!avail) {
    // No sm_120a kernel is launched: every entry point throws std::logic_error.
    EXPECT_THROW(nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(out), 64, nullptr), std::logic_error);
    EXPECT_THROW(nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(out), 64, nullptr, Nvfp4Tile::k256x128x128,
                                nv::Nvfp4Delta{as_bf16(res), 64}),
                 std::logic_error);
    EXPECT_THROW(nv::nvfp4_gemm_gated_residual(d.ops(), nullptr, as_bf16(d.gate), as_bf16(res), 64, nullptr),
                 std::logic_error);
    EXPECT_THROW(nv::e2m1_cvt_codes(f.get(), codes.get(), 16, nullptr), std::logic_error);
  } else {
    EXPECT_NO_THROW(nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(out), 64, nullptr));
  }
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------------------------------------------
// G1a: small-integer codes, unit scales, against CUTLASS's host reference GEMM.

using SfConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// acc[m][n] by CUTLASS's host reference mainloop on hi_first operands, converted to CUTLASS's nibble order.
std::vector<float> cutlass_host_acc(std::vector<std::uint8_t> a, std::vector<std::uint8_t> sfa,
                                    std::vector<std::uint8_t> b, std::vector<std::uint8_t> sfb, int m, int n, int k) {
  using namespace cute;
  for (auto& v : a) v = static_cast<std::uint8_t>(v << 4 | v >> 4);
  for (auto& v : b) v = static_cast<std::uint8_t>(v << 4 | v >> 4);
  auto tA = make_tensor(recast_ptr<cutlass::float_e2m1_t>(a.data()), make_layout(make_shape(m, k, 1), make_stride(k, _1{}, m * k)));
  auto tB = make_tensor(recast_ptr<cutlass::float_e2m1_t>(b.data()), make_layout(make_shape(n, k, 1), make_stride(k, _1{}, n * k)));
  auto tSFA = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfa.data()), SfConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1)));
  auto tSFB = make_tensor(recast_ptr<cutlass::float_ue4m3_t>(sfb.data()), SfConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1)));
  cutlass::reference::host::GettBlockScalingMainloopParams<float, decltype(tA), decltype(tSFA), decltype(tB), decltype(tSFB)> params{
      tA, tSFA, tB, tSFB};
  constexpr int kBlk = 64;
  std::vector<float> out(static_cast<std::size_t>(m) * n);
  float acc[kBlk][kBlk];
  for (int m0 = 0; m0 < m; m0 += kBlk) {
    for (int n0 = 0; n0 < n; n0 += kBlk) {
      cutlass::reference::host::gett_mainloop(params, m0, n0, 0, acc);
      for (int i = 0; i < kBlk && m0 + i < m; ++i) {
        for (int j = 0; j < kBlk && n0 + j < n; ++j) out[static_cast<std::size_t>(m0 + i) * n + n0 + j] = acc[i][j];
      }
    }
  }
  return out;
}

// Host operand: random codes [rows, k / 2] and a swizzled scale buffer with the given used-scale generator.
struct HostOperand {
  std::vector<std::uint8_t> q, sf;
};
template <class SfGen>
HostOperand host_operand(std::uint64_t seed, std::int64_t rows, std::int64_t k, SfGen&& sf_of) {
  HostOperand h;
  h.q.resize(static_cast<std::size_t>(rows * k / 2));
  for (std::size_t i = 0; i < h.q.size(); ++i) h.q[i] = static_cast<std::uint8_t>(mix(i + seed * 0x9E3779B97F4A7C15ull) & 0xFF);
  h.sf.assign(static_cast<std::size_t>(nv::sf_bytes(rows, k)), 0);
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t kb = 0; kb < k / 16; ++kb) h.sf[static_cast<std::size_t>(nv::sf_offset(r, kb, k))] = sf_of(r, kb);
  }
  return h;
}

TEST(GpuNvfp4Gemm, G1aSmallIntegersMatchCutlassHostReference) {
  MI_REQUIRE_NVFP4();
  // A named array, not a braced list (nvcc 12.8, see test_nvfp4_host_ref.cu).
  const std::tuple<int, int, int> cases[] = {{200, 264, 512}, {129, 128, 96}, {8, 8, 32}};
  for (const auto& [m, n, k] : cases) {
    const auto one = [](std::int64_t, std::int64_t) { return std::uint8_t{0x38}; };  // e4m3 1.0
    const HostOperand a = host_operand(static_cast<std::uint64_t>(m * 3 + k), m, k, one);
    const HostOperand b = host_operand(static_cast<std::uint64_t>(n * 5 + k), n, k, one);
    const std::vector<float> ref = cutlass_host_acc(a.q, a.sf, b.q, b.sf, m, n, k);
    Buf<std::uint8_t> xq(a.q), xsf(a.sf), wq(b.q), wsf(b.sf);
    Buf<float> xs(std::vector<float>(static_cast<std::size_t>(m), 1.0f));
    nv::Nvfp4Operands o;
    o.x = xq.get();
    o.x_sf = xsf.get();
    o.x_scale = xs.get();
    o.w = wq.get();
    o.w_sf = wsf.get();
    o.w_scale2 = 1.0f;
    o.m = m;
    o.n = n;
    o.k = k;
    std::vector<std::uint16_t> want(ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) want[i] = bf16_bits(epi_value(ref[i], 1.0f, 1.0f, 0, 0.0f, false));
    Buf<std::uint16_t> out(ref.size());
    for (Nvfp4Tile tile : kTiles) {
      out.fill_bytes(0xFF);
      nv::nvfp4_gemm(o, nullptr, as_bf16(out), n, nullptr, tile);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      EXPECT_EQ(count_diff(out.download(), want), 0u) << m << "x" << n << "x" << k << " " << nv::nvfp4_tile_name(tile);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// G1b: DiT shapes against the fp64 oracle.

struct Shape {
  const char* name;
  std::int64_t m, n, k;
};
const Shape kDitShapes[] = {{"qkv", kM1024, 9216, 3072},
                            {"o", kM1024, 3072, 3072},
                            {"up_gate", kM1024, 16384, 3072},
                            {"down", kM1024, 3072, 8192},
                            {"edge", 333, 1032, 96}};

TEST(GpuNvfp4Gemm, G1bDitShapesMatchFp64Oracle) {
  MI_REQUIRE_NVFP4();
  for (const Shape& s : kDitShapes) {
    GemmData d(s.m, s.n, s.k, static_cast<std::uint64_t>(s.n + s.k));
    const auto rows = sample_rows(s.m);
    const std::vector<double> acc = oracle_acc(d, rows);
    const std::vector<float> xs = d.x.g.download();
    const std::vector<std::uint16_t> bias = d.bias.download();
    const std::size_t mn = static_cast<std::size_t>(s.m * s.n);
    Buf<std::uint16_t> y1(mn), y2(mn), neg(mn), yd(mn);
    Buf<std::uint16_t> delta = dev_uniform(91, mn, 0.0f, 0.5f);
    const std::vector<std::uint16_t> delta_h = download_rows(delta.get(), s.n, s.n, rows);

    nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(y1), s.n, nullptr);
    nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(yd), s.n, nullptr, Nvfp4Tile::kAuto, {as_bf16(delta), s.n});
    const auto got = download_rows(y1.get(), s.n, s.n, rows);
    const auto got_d = download_rows(yd.get(), s.n, s.n, rows);
    // Two-pass readout of the pre-round value (no bias): y1 = bf16(v), y2 = bf16(v - y1) (the subtraction is exact).
    nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(y1), s.n, nullptr);
    negate_kernel<<<1024, 256>>>(y1.get(), neg.get(), mn);
    MI_CUDA_CHECK_LAUNCH();
    nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(y2), s.n, nullptr, Nvfp4Tile::kAuto, {as_bf16(neg), s.n});
    const auto hi = download_rows(y1.get(), s.n, s.n, rows), lo = download_rows(y2.get(), s.n, s.n, rows);

    std::size_t same = 0, same_d = 0;
    int max_ulp = 0, max_ulp_d = 0;
    double err_gpu = 0, err_floor = 0, norm = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const float xr = xs[static_cast<std::size_t>(rows[i])];
      for (std::int64_t c = 0; c < s.n; ++c) {
        const std::size_t j = i * static_cast<std::size_t>(s.n) + static_cast<std::size_t>(c);
        const float a32 = static_cast<float>(acc[j]);
        const float b = bf16_float(bias[static_cast<std::size_t>(c)]);
        const std::uint16_t want = bf16_bits(epi_value(a32, xr, d.s2, 0, b, false));
        const std::uint16_t want_d = bf16_bits(epi_value(a32, xr, d.s2, bf16_float(delta_h[j]), b, true));
        same += got[j] == want;
        same_d += got_d[j] == want_d;
        max_ulp = std::max(max_ulp, ulp_dist(got[j], want));
        max_ulp_d = std::max(max_ulp_d, ulp_dist(got_d[j], want_d));
        const double v64 = acc[j] * static_cast<double>(xr) * static_cast<double>(d.s2);
        const float v32 = epi_value(a32, xr, d.s2, 0, 0.0f, false);
        const float r_hi = bf16_float(bf16_bits(v32));
        const double ref_read = static_cast<double>(r_hi) + bf16_float(bf16_bits(v32 - r_hi));
        const double gpu_read = static_cast<double>(bf16_float(hi[j])) + bf16_float(lo[j]);
        err_gpu += (gpu_read - v64) * (gpu_read - v64);
        err_floor += (ref_read - v64) * (ref_read - v64);
        norm += v64 * v64;
      }
    }
    const double total = static_cast<double>(got.size());
    const double rel_gpu = std::sqrt(err_gpu / norm), rel_floor = std::sqrt(err_floor / norm);
    std::printf("[nvfp4 G1b] %-7s %lld x %lld x %lld: E1 %.5f %% identical (max %d ULP), E1L %.5f %% (max %d), "
                "two-pass rel-L2 %.3g vs floor %.3g\n",
                s.name, static_cast<long long>(s.m), static_cast<long long>(s.n), static_cast<long long>(s.k),
                100.0 * static_cast<double>(same) / total, max_ulp, 100.0 * static_cast<double>(same_d) / total,
                max_ulp_d, rel_gpu, rel_floor);
    EXPECT_GE(static_cast<double>(same) / total, 0.999) << s.name;
    EXPECT_GE(static_cast<double>(same_d) / total, 0.999) << s.name << " E1L";
    EXPECT_LE(max_ulp, 1) << s.name;
    EXPECT_LE(max_ulp_d, 1) << s.name << " E1L";
    EXPECT_LE(rel_gpu * rel_gpu, rel_floor * rel_floor + 2e-6 * 2e-6) << s.name;
    RecordProperty(std::string("rel_l2_") + s.name, std::to_string(rel_gpu));
  }
}

// ---------------------------------------------------------------------------------------------------------------
// G1c: tiles, schedules and epilogue identities.

TEST(GpuNvfp4Gemm, G1cTilesAndEpiloguesAgree) {
  MI_REQUIRE_NVFP4();
  for (const Shape& s : {Shape{"o", kM1024, 3072, 3072}, Shape{"edge", 200, 264, 96}, Shape{"tiny", 1, 8, 32},
                         Shape{"tail", 333, 1032, 8192}}) {
    GemmData d(s.m, s.n, s.k, static_cast<std::uint64_t>(s.m + s.n));
    const std::size_t mn = static_cast<std::size_t>(s.m * s.n);
    // A per-column constant delta (the same for every row), as a full [m, n] matrix (no -0.0: E1L adds the null bias
    // as +0.0).
    std::vector<std::uint16_t> col = synth_vec(55, static_cast<std::size_t>(s.n), 0.0f, 3.0f);
    for (auto& v : col) v = v == 0x8000 ? 0 : v;
    std::vector<std::uint16_t> dmat(mn);
    for (std::size_t i = 0; i < mn; ++i) dmat[i] = col[i % static_cast<std::size_t>(s.n)];
    // The same with row stride n + 16, NaN in the padding (never read).
    const std::int64_t ldd = s.n + 16, ldr = s.n + 24;
    std::vector<std::uint16_t> dpad(static_cast<std::size_t>(s.m * ldd), 0x7FC0);
    for (std::int64_t r = 0; r < s.m; ++r) std::copy_n(dmat.begin() + r * s.n, s.n, dpad.begin() + r * ldd);
    Buf<std::uint16_t> colb(col), dconst(dmat), dstrided(dpad), zero(mn), out(mn), res(mn);
    Buf<std::uint16_t> res_wide(static_cast<std::size_t>(s.m * ldr));
    zero.fill_bytes(0);
    Buf<std::uint16_t> res0 = dev_act(static_cast<std::uint64_t>(s.n), s.m, s.n);
    const auto res_h = res0.download();
    const auto gate_h = d.gate.download();
    // res_wide = res0 with row stride ldr and 0x5A5A padding.
    std::vector<std::uint16_t> res_wide_h(static_cast<std::size_t>(s.m * ldr), 0x5A5A);
    for (std::int64_t r = 0; r < s.m; ++r) std::copy_n(res_h.begin() + r * s.n, s.n, res_wide_h.begin() + r * ldr);
    // Interior [m, n] of a row-stride-ldr matrix, and how many of its padding elements are not 0x5A5A.
    const auto interior = [&](const std::vector<std::uint16_t>& w, std::size_t* pad_bad) {
      std::vector<std::uint16_t> v(mn);
      *pad_bad = 0;
      for (std::int64_t r = 0; r < s.m; ++r) {
        for (std::int64_t c = 0; c < ldr; ++c) {
          const std::uint16_t e = w[static_cast<std::size_t>(r * ldr + c)];
          if (c < s.n) {
            v[static_cast<std::size_t>(r * s.n + c)] = e;
          } else {
            *pad_bad += e != 0x5A5A;
          }
        }
      }
      return v;
    };
    std::vector<std::uint16_t> ref_e1, ref_e2;
    int fill = 0;
    // Fresh garbage in the output before every call: a kernel that writes nothing cannot pass.
    const auto garbage = [&](Buf<std::uint16_t>& b) { b.fill_bytes(0x11 + 0x22 * (fill++ % 7)); };
    for (Nvfp4Tile tile : kTiles) {
      const std::string tag = std::string(s.name) + " " + nv::nvfp4_tile_name(tile);
      garbage(out);
      nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(out), s.n, nullptr, tile);
      const auto e1 = out.download();
      garbage(out);
      nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(out), s.n, nullptr, tile, {as_bf16(zero), s.n});
      EXPECT_EQ(count_diff(out.download(), e1), 0u) << tag << ": E1L with delta 0 vs E1";
      garbage(out);
      nv::nvfp4_gemm(d.ops(), as_bf16(colb), as_bf16(out), s.n, nullptr, tile);
      const auto e1_colbias = out.download();
      garbage(out);
      nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(out), s.n, nullptr, tile, {as_bf16(dconst), s.n});
      EXPECT_EQ(count_diff(out.download(), e1_colbias), 0u)
          << tag << ": E1L with a per-column constant delta (the same for every row) vs E1 with that bias";
      garbage(out);
      nv::nvfp4_gemm(d.ops(), nullptr, as_bf16(out), s.n, nullptr, tile, {as_bf16(dstrided), ldd});
      EXPECT_EQ(count_diff(out.download(), e1_colbias), 0u) << tag << ": E1L with delta row stride n + 16";

      MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), mn * 2, cudaMemcpyDeviceToDevice));
      nv::nvfp4_gemm_gated_residual(d.ops(), as_bf16(d.bias), as_bf16(d.gate), as_bf16(res), s.n, nullptr, tile);
      const auto e2 = res.download();
      std::vector<std::uint16_t> want(mn);
      for (std::size_t i = 0; i < mn; ++i) {
        const float y = bf16_float(e1[i]), g = bf16_float(gate_h[i % static_cast<std::size_t>(s.n)]);
        want[i] = bf16_bits(bf16_float(res_h[i]) + bf16_float(bf16_bits(y * g)));
      }
      EXPECT_EQ(count_diff(e2, want), 0u) << tag << ": E2 vs residual + bf16(E1 * gate)";
      MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), mn * 2, cudaMemcpyDeviceToDevice));
      nv::nvfp4_gemm_gated_residual(d.ops(), as_bf16(d.bias), as_bf16(d.gate), as_bf16(res), s.n, nullptr, tile,
                                    {as_bf16(zero), s.n});
      EXPECT_EQ(count_diff(res.download(), e2), 0u) << tag << ": E2L with delta 0 vs E2";
      MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), mn * 2, cudaMemcpyDeviceToDevice));
      nv::nvfp4_gemm_gated_residual(d.ops(), nullptr, as_bf16(d.gate), as_bf16(res), s.n, nullptr, tile,
                                    {as_bf16(dconst), s.n});
      const auto e2l = res.download();
      MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), mn * 2, cudaMemcpyDeviceToDevice));
      nv::nvfp4_gemm_gated_residual(d.ops(), as_bf16(colb), as_bf16(d.gate), as_bf16(res), s.n, nullptr, tile);
      EXPECT_EQ(count_diff(res.download(), e2l), 0u)
          << tag << ": E2L with a per-column constant delta (the same for every row) vs E2 with that bias";

      // Residual row stride n + 24 (padding untouched), and E2L with delta row stride n + 16 as well.
      std::size_t pad_bad = 0;
      res_wide.upload(res_wide_h);
      nv::nvfp4_gemm_gated_residual(d.ops(), as_bf16(d.bias), as_bf16(d.gate), as_bf16(res_wide), ldr, nullptr, tile);
      EXPECT_EQ(count_diff(interior(res_wide.download(), &pad_bad), e2), 0u) << tag << ": E2 with ldr = n + 24";
      EXPECT_EQ(pad_bad, 0u) << tag << ": E2 wrote the residual padding";
      res_wide.upload(res_wide_h);
      nv::nvfp4_gemm_gated_residual(d.ops(), nullptr, as_bf16(d.gate), as_bf16(res_wide), ldr, nullptr, tile,
                                    {as_bf16(dstrided), ldd});
      EXPECT_EQ(count_diff(interior(res_wide.download(), &pad_bad), e2l), 0u)
          << tag << ": E2L with ldr = n + 24 and delta row stride n + 16";
      EXPECT_EQ(pad_bad, 0u) << tag << ": E2L wrote the residual padding";
      if (ref_e1.empty()) {
        ref_e1 = e1;
        ref_e2 = e2;
      } else {
        EXPECT_EQ(count_diff(e1, ref_e1), 0u) << tag << ": E1 vs the first tile";
        EXPECT_EQ(count_diff(e2, ref_e2), 0u) << tag << ": E2 vs the first tile";
      }
    }
    // A padded output row stride leaves the padding untouched.
    const std::int64_t ldo = s.n + 24;
    Buf<std::uint16_t> wide(static_cast<std::size_t>(s.m * ldo));
    wide.fill_bytes(0x5A);
    nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(wide), ldo, nullptr);
    const auto wh = wide.download();
    std::size_t bad = 0, pad_bad = 0;
    for (std::int64_t r = 0; r < s.m; ++r) {
      for (std::int64_t c = 0; c < ldo; ++c) {
        const std::uint16_t v = wh[static_cast<std::size_t>(r * ldo + c)];
        if (c < s.n) {
          bad += v != ref_e1[static_cast<std::size_t>(r * s.n + c)];
        } else {
          pad_bad += v != 0x5A5A;
        }
      }
    }
    EXPECT_EQ(bad, 0u) << s.name << ": strided output";
    EXPECT_EQ(pad_bad, 0u) << s.name << ": output padding written";
  }
}

// G1d: the nibble order is irrelevant as long as both operands share it.
TEST(GpuNvfp4Gemm, G1dHiFirstEqualsLoFirst) {
  MI_REQUIRE_NVFP4();
  for (const Shape& s : {Shape{"qkv", kM1024, 9216, 3072}, Shape{"edge", 200, 264, 96}}) {
    GemmData d(s.m, s.n, s.k, 4321);
    const std::size_t mn = static_cast<std::size_t>(s.m * s.n);
    Buf<std::uint16_t> a(mn), b(mn);
    a.fill_bytes(0x33);  // different garbage in each output: neither can match by being left unwritten
    b.fill_bytes(0xCC);
    nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(a), s.n, nullptr);
    swap_nibbles_kernel<<<1024, 256>>>(d.x.q.get(), d.x.q.n);
    swap_nibbles_kernel<<<1024, 256>>>(d.w.q.get(), d.w.q.n);
    MI_CUDA_CHECK_LAUNCH();
    nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(b), s.n, nullptr);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(count_diff(a.download(), b.download()), 0u) << s.name;
  }
}

// G1e: software E2M1 rounding (the CPU codec's, which the quantizers match bit for bit: T3) vs the hardware cvt.
TEST(GpuNvfp4Gemm, G1eSoftwareE2m1MatchesCvt) {
  MI_REQUIRE_NVFP4();
  std::vector<float> in;
  for (std::uint32_t b = 0; b < 0x10000; ++b) {  // every finite bf16 value
    const float v = bf16_float(static_cast<std::uint16_t>(b));
    if (std::isfinite(v)) in.push_back(v);
  }
  for (float mid : {0.0f, 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 6.0f}) {  // around every rounding boundary
    for (float sign : {1.0f, -1.0f}) {
      float v = sign * mid;
      for (int i = 0; i < 64; ++i) v = std::nextafter(v, -std::numeric_limits<float>::infinity());
      for (int i = 0; i < 129; ++i, v = std::nextafter(v, std::numeric_limits<float>::infinity())) in.push_back(v);
    }
  }
  for (std::uint64_t i = 0; i < (1u << 20); ++i) {  // random fp32 in [-8, 8], subnormals included at the low end
    const std::uint64_t z = mix(i + 12345);
    in.push_back(std::ldexp(static_cast<float>(z & 0xFFFFFF) / 16777216.0f, -static_cast<int>((z >> 24) % 160) + 3) *
                 ((z >> 40) & 1 ? -1.0f : 1.0f));
  }
  Buf<float> x(in);
  Buf<std::uint8_t> codes(in.size());
  nv::e2m1_cvt_codes(x.get(), codes.get(), static_cast<std::int64_t>(in.size()), nullptr);
  const auto hw = codes.download();
  std::size_t diff = 0, diff_clamped = 0;
  for (std::size_t i = 0; i < in.size(); ++i) {
    const std::uint8_t sw = q::e2m1_rne(in[i]);
    if (sw != hw[i]) {
      if (diff < 8) std::printf("[nvfp4 G1e] x %.9g (0x%08x): software 0x%x, cvt 0x%x\n", in[i], std::bit_cast<std::uint32_t>(in[i]), sw, hw[i]);
      ++diff;
      diff_clamped += std::fabs(in[i]) <= 6.0f;
    }
  }
  std::printf("[nvfp4 G1e] %zu values, %zu differ (%zu within [-6, 6])\n", in.size(), diff, diff_clamped);
  RecordProperty("e2m1_mismatches", static_cast<int>(diff));
  EXPECT_EQ(diff, 0u);
}

// ---------------------------------------------------------------------------------------------------------------
// G2: accumulator characterization. Hand-built operands: x row r and w column c carry one nonzero element per K
// block, so each block partial p_b = x_sf * w_sf * qx * qw is chosen exactly; unit second-level scales, no bias.

struct BlockPlan {
  std::uint8_t sfa, sfb, qa, qb;  // e4m3 scales, E2M1 codes of the block's element 0 (others zero)
};

// One output (row 0, column 0) per plan of k / 16 blocks; rows / columns beyond 0 get zero codes.
std::uint16_t run_plan(const std::vector<BlockPlan>& plan, Nvfp4Tile tile) {
  const std::int64_t k = static_cast<std::int64_t>(plan.size()) * 16, m = 8, n = 8;
  std::vector<std::uint8_t> xq(static_cast<std::size_t>(m * k / 2), 0), wq(static_cast<std::size_t>(n * k / 2), 0);
  std::vector<std::uint8_t> xsf(static_cast<std::size_t>(nv::sf_bytes(m, k)), 0), wsf(static_cast<std::size_t>(nv::sf_bytes(n, k)), 0);
  for (std::size_t b = 0; b < plan.size(); ++b) {
    xq[b * 8] = static_cast<std::uint8_t>(plan[b].qa << 4);  // hi_first: element 0 in the high nibble
    wq[b * 8] = static_cast<std::uint8_t>(plan[b].qb << 4);
    xsf[static_cast<std::size_t>(nv::sf_offset(0, static_cast<std::int64_t>(b), k))] = plan[b].sfa;
    wsf[static_cast<std::size_t>(nv::sf_offset(0, static_cast<std::int64_t>(b), k))] = plan[b].sfb;
  }
  Buf<std::uint8_t> dxq(xq), dwq(wq), dxsf(xsf), dwsf(wsf);
  Buf<float> xs(std::vector<float>(static_cast<std::size_t>(m), 1.0f));
  Buf<std::uint16_t> out(static_cast<std::size_t>(m * n));
  nv::Nvfp4Operands o;
  o.x = dxq.get();
  o.x_sf = dxsf.get();
  o.x_scale = xs.get();
  o.w = dwq.get();
  o.w_sf = dwsf.get();
  o.m = m;
  o.n = n;
  o.k = k;
  nv::nvfp4_gemm(o, nullptr, as_bf16(out), n, nullptr, tile);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return out.download()[0];
}

double plan_exact(const std::vector<BlockPlan>& plan) {
  double s = 0;
  for (const BlockPlan& p : plan) {
    s += static_cast<double>(q::e4m3_to_float(p.sfa)) * q::e4m3_to_float(p.sfb) * q::e2m1_to_float(p.qa) *
         q::e2m1_to_float(p.qb);
  }
  return s;
}

// e4m3 code of 2^e (e in -9 .. 8; -9 .. -7 subnormal).
std::uint8_t e4m3_pow2(int e) { return e >= -6 ? static_cast<std::uint8_t>((e + 7) << 3) : static_cast<std::uint8_t>(1 << (e + 9)); }

TEST(GpuNvfp4Gemm, G2AccumulatorCharacterization) {
  MI_REQUIRE_NVFP4();
  constexpr std::uint8_t kOne = 0x2, kFour = 0x6, kNegFour = 0xE, kSix = 0x7;
  // (a) Cancellation: +big, +small, -big with big / small = 2^d. Exact result: small. A plain fp32 accumulator keeps
  // the small term up to d = 23 (d = 24 is a tie to even: lost). Terms in one MMA K step (blocks 0..3, k64) or in
  // three different steps (blocks 0, 4, 8).
  std::string survive_in_step, survive_across;
  for (int d = 18; d <= 30; ++d) {
    // big = 4 * 2^8 * 2^8 = 2^18; small = 2^(18 - d) = 1 * 2^ea * 2^eb.
    const int e_small = 18 - d, ea = std::max(-9, e_small / 2), eb = e_small - ea;
    if (eb < -9 || eb > 8) continue;
    const BlockPlan big{e4m3_pow2(8), e4m3_pow2(8), kFour, kOne}, nbig{e4m3_pow2(8), e4m3_pow2(8), kNegFour, kOne};
    const BlockPlan small{e4m3_pow2(ea), e4m3_pow2(eb), kOne, kOne}, zero{0, 0, 0, 0};
    for (int across = 0; across < 2; ++across) {
      std::vector<BlockPlan> plan;
      plan.reserve(12);
      for (int b = 0; b < (across ? 12 : 4); ++b) plan.push_back(zero);
      plan[0] = big;
      plan[across ? 4 : 1] = small;
      plan[across ? 8 : 2] = nbig;
      const double exact = plan_exact(plan);
      for (Nvfp4Tile tile : kTiles) {
        const float got = bf16_float(run_plan(plan, tile));
        // Either the exact small term or zero (lost), never anything else; tiles agree.
        EXPECT_TRUE(got == static_cast<float>(exact) || got == 0.0f) << "d " << d << " across " << across << ": " << got;
        if (tile == kTiles[0]) (across ? survive_across : survive_in_step) += got != 0.0f ? "1" : "0";
      }
    }
  }
  std::printf("[nvfp4 G2] small term survives (d = 18..30): within one k64 step %s, across steps %s\n",
              survive_in_step.c_str(), survive_across.c_str());
  RecordProperty("survive_within_step_d18_30", survive_in_step);
  RecordProperty("survive_across_steps_d18_30", survive_across);
  // fp32 accumulation across steps at least: the term survives for d <= 23.
  EXPECT_EQ(survive_across.substr(0, 6), "111111");

  // (b) Random cancellation: 512 blocks of random partials summing near zero; error within the fp32 sequential bound
  // (blocks * 2^-24 * sum |p|) plus the bf16 output rounding.
  for (int trial = 0; trial < 8; ++trial) {
    std::vector<BlockPlan> plan(512);
    double mag = 0;
    for (std::size_t b = 0; b < plan.size(); ++b) {
      const std::uint64_t z = mix(b + 1000 * static_cast<std::uint64_t>(trial));
      plan[b] = {static_cast<std::uint8_t>(0x08 + z % 0x70), static_cast<std::uint8_t>(0x08 + (z >> 8) % 0x70),
                 static_cast<std::uint8_t>(1 + (z >> 16) % 7 | ((z >> 20) & 1) << 3), kOne};
    }
    // Force near-cancellation: the second half mirrors the first with the sign flipped, plus one small residue.
    for (std::size_t b = 256; b < 512; ++b) plan[b] = {plan[b - 256].sfa, plan[b - 256].sfb, static_cast<std::uint8_t>(plan[b - 256].qa ^ 8), kOne};
    plan[511] = {0x38, 0x38, kSix, kOne};
    for (const BlockPlan& p : plan) mag += std::fabs(plan_exact({p}));
    const double exact = plan_exact(plan);
    for (Nvfp4Tile tile : kTiles) {
      const double got = bf16_float(run_plan(plan, tile));
      const double tol = 512 * std::ldexp(mag, -24) + std::fabs(exact) * std::ldexp(1.0, -8);
      EXPECT_LE(std::fabs(got - exact), tol) << "trial " << trial << " " << nv::nvfp4_tile_name(tile);
    }
  }

  // (c) Scale extremes and subnormal block scales: exact sums (each partial and every partial sum is exact in fp32),
  // so the output is the bf16 rounding of the exact value.
  const auto check_exact = [&](const std::vector<BlockPlan>& plan, const char* what) {
    const double exact = plan_exact(plan);
    for (Nvfp4Tile tile : kTiles) {
      EXPECT_EQ(run_plan(plan, tile), bf16_bits(static_cast<float>(exact))) << what << " " << nv::nvfp4_tile_name(tile);
    }
  };
  check_exact(std::vector<BlockPlan>(64, BlockPlan{0x7E, 0x7E, kSix, kSix}), "448 * 448 * 36, 64 blocks");
  {
    std::vector<BlockPlan> plan;
    for (int c = 1; c <= 7; ++c) plan.push_back({static_cast<std::uint8_t>(c), static_cast<std::uint8_t>(8 - c), kSix, kFour});
    plan.push_back({0x01, 0x01, kOne, kOne});  // 2^-18
    check_exact(plan, "subnormal block scales");
  }
  {
    std::vector<BlockPlan> plan(8, BlockPlan{0x00, 0x7E, kSix, kSix});  // sf 0 with nonzero codes contributes 0
    plan[7] = {0x38, 0x38, kOne, kOne};
    check_exact(plan, "zero block scales");
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The quantizers and GEMMs captured into a CUDA graph on a non-blocking stream (the engine's thread-local capture
// mode), replayed, against eager runs.

TEST(GpuNvfp4Quant, GraphCaptureOnStream) {
  const std::int64_t m = 300, k = 3072, f = 512;
  const NormVecs v(k, 5);
  Buf<std::uint16_t> x = dev_act(3, m, k), ug = dev_act(4, m, 2 * f);
  DevAct a1(m, k), a2(m, k), a3(m, f), e1(m, k), e2(m, k), e3(m, f);
  const bool gemm = nv::available();
  GemmData d(m, 264, k, 8);
  Buf<std::uint16_t> out(static_cast<std::size_t>(m * 264)), out_e(static_cast<std::size_t>(m * 264));
  Buf<std::uint16_t> res = dev_act(6, m, 264), res_e(static_cast<std::size_t>(m * 264));
  Buf<std::uint16_t> delta = dev_uniform(7, static_cast<std::size_t>(m * 264), 0.0f, 0.5f);
  MI_CUDA_CHECK(cudaMemcpy(res_e.get(), res.get(), res.n * 2, cudaMemcpyDeviceToDevice));
  MI_CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t stream = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  nv::quantize_nvfp4(as_bf16(x), k, a1.act(), m, static_cast<int>(k), stream);
  nv::rms_norm_modulate_quantize_nvfp4(as_bf16(x), k, as_bf16(v.weight), as_bf16(v.scale), as_bf16(v.shift), a2.act(),
                                       m, static_cast<int>(k), stream);
  nv::situ_glu_quantize_nvfp4(as_bf16(ug), 2 * f, a3.act(), m, static_cast<int>(f), stream);
  if (gemm) nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(out), 264, stream, Nvfp4Tile::k128x128x128Pingpong);
  MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  MI_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  // The gated residual updates in place: capture it (with a LoRA delta) alone and replay once.
  cudaGraph_t graph_res = nullptr;
  cudaGraphExec_t exec_res = nullptr;
  if (gemm) {
    MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    nv::nvfp4_gemm_gated_residual(d.ops(), as_bf16(d.bias), as_bf16(d.gate), as_bf16(res), 264, stream,
                                  Nvfp4Tile::k256x128x128, {as_bf16(delta), 264});
    MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_res));
    MI_CUDA_CHECK(cudaGraphInstantiate(&exec_res, graph_res, 0));
  }
  MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  if (gemm) MI_CUDA_CHECK(cudaGraphLaunch(exec_res, stream));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));

  nv::quantize_nvfp4(as_bf16(x), k, e1.act(), m, static_cast<int>(k), nullptr);
  nv::rms_norm_modulate_quantize_nvfp4(as_bf16(x), k, as_bf16(v.weight), as_bf16(v.scale), as_bf16(v.shift), e2.act(),
                                       m, static_cast<int>(k), nullptr);
  nv::situ_glu_quantize_nvfp4(as_bf16(ug), 2 * f, e3.act(), m, static_cast<int>(f), nullptr);
  if (gemm) {
    nv::nvfp4_gemm(d.ops(), as_bf16(d.bias), as_bf16(out_e), 264, nullptr, Nvfp4Tile::k128x128x128Pingpong);
    nv::nvfp4_gemm_gated_residual(d.ops(), as_bf16(d.bias), as_bf16(d.gate), as_bf16(res_e), 264, nullptr,
                                  Nvfp4Tile::k256x128x128, {as_bf16(delta), 264});
  }
  expect_same(download(a1), download(e1), "quantize_nvfp4");
  expect_same(download(a2), download(e2), "rms_norm_modulate_quantize_nvfp4");
  expect_same(download(a3), download(e3), "situ_glu_quantize_nvfp4");
  if (gemm) {
    EXPECT_EQ(count_diff(out.download(), out_e.download()), 0u) << "nvfp4_gemm";
    EXPECT_EQ(count_diff(res.download(), res_e.download()), 0u) << "nvfp4_gemm_gated_residual with a delta";
    MI_CUDA_CHECK(cudaGraphExecDestroy(exec_res));
    MI_CUDA_CHECK(cudaGraphDestroy(graph_res));
  }
  MI_CUDA_CHECK(cudaGraphExecDestroy(exec));
  MI_CUDA_CHECK(cudaGraphDestroy(graph));
  MI_CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------------------------------------------
// Performance (prints; no timing assertions). Opt-in: MI_RUN_PERF=1.

#define MI_REQUIRE_PERF()                                                                      \
  do {                                                                                         \
    const char* run_perf = std::getenv("MI_RUN_PERF");                                         \
    if (run_perf == nullptr || std::string(run_perf) != "1") GTEST_SKIP() << "set MI_RUN_PERF=1"; \
  } while (0)

class Timer {
 public:
  Timer() {
    MI_CUDA_CHECK(cudaEventCreate(&e0_));
    MI_CUDA_CHECK(cudaEventCreate(&e1_));
  }
  ~Timer() {
    cudaEventDestroy(e0_);
    cudaEventDestroy(e1_);
  }
  double ms(const std::function<void()>& fn, int iters) {
    MI_CUDA_CHECK(cudaEventRecord(e0_, nullptr));
    for (int i = 0; i < iters; ++i) fn();
    MI_CUDA_CHECK(cudaEventRecord(e1_, nullptr));
    MI_CUDA_CHECK(cudaEventSynchronize(e1_));
    float t = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&t, e0_, e1_));
    return static_cast<double>(t) / iters;
  }

 private:
  cudaEvent_t e0_ = nullptr, e1_ = nullptr;
};

double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Steady-state ms per call of each op: per round and op, a warm-up of window / 2 then a timed window (default 1000 ms,
// MI_NVFP4_PERF_WINDOW_MS overrides), ops round-robin over `rounds`; the mean over rounds. The power controller
// settles over ~1 s, so windows of that order rank ops at their sustained clocks.
std::vector<double> steady(Timer& t, const std::vector<std::function<void()>>& fns, int rounds = 2) {
  const char* env = std::getenv("MI_NVFP4_PERF_WINDOW_MS");
  const double window = env != nullptr ? std::atof(env) : 1000.0;
  std::vector<double> r(fns.size(), 0.0);
  for (int round = 0; round < rounds; ++round) {
    for (std::size_t i = 0; i < fns.size(); ++i) {
      const double one = std::max(t.ms(fns[i], 1), 1e-3);
      t.ms(fns[i], std::max(1, static_cast<int>(window / 2 / one)));
      r[i] += t.ms(fns[i], std::max(3, static_cast<int>(window / one)));
    }
  }
  for (double& v : r) v /= rounds;
  return r;
}

// Weight copies rotated through on every call (> 2x the L2: the DiT streams a fresh block's weights each time).
template <class T>
struct Rotating {
  std::vector<Buf<T>> copies;
  std::size_t next = 0;
  Rotating(std::size_t elems, std::size_t min_bytes, const std::function<void(Buf<T>&, int)>& init) {
    const std::size_t count = std::max<std::size_t>(2, (min_bytes + elems * sizeof(T) - 1) / (elems * sizeof(T)));
    for (std::size_t i = 0; i < count; ++i) {
      copies.emplace_back(elems);
      init(copies.back(), static_cast<int>(i));
    }
  }
  const T* get() {
    const T* p = copies[next].get();
    next = (next + 1) % copies.size();
    return p;
  }
};

struct Site {
  const char* name;
  std::int64_t n, k;
  int kind;  // 0 qkv (norm, plain), 1 o (quantize, gated residual), 2 up_gate (norm, plain / INT8 SiTU), 3 down (situ / rotate, gated residual)
};
constexpr Site kSites[] = {{"qkv", 9216, 3072, 0}, {"o", 3072, 3072, 1}, {"up_gate", 16384, 3072, 2}, {"down", 3072, 8192, 3}};

// MI_NVFP4_PERF_SITES (comma-separated site names, default all) selects sites, so a sweep can be split into short runs.
bool site_selected(const Site& s) {
  const char* env = std::getenv("MI_NVFP4_PERF_SITES");
  if (env == nullptr || *env == '\0') return true;
  const std::string list = std::string(",") + env + ",";
  return list.find(std::string(",") + s.name + ",") != std::string::npos;
}

// Per DiT site at M rows: the INT8 path (rotate quantizer + INT8 GEMM with its fused epilogue, kAuto tile) vs the
// NVFP4 path (its quantizer + E1 / E2 GEMM) for every tile and kAuto; plus the quantizers and GEMMs alone, the block
// total, and E1L / E2L (LoRA delta) overhead.
TEST(GpuNvfp4Kernels, Throughput) {
  MI_REQUIRE_PERF();
  if (!nv::available()) GTEST_SKIP() << "nvfp4::available() is false";
  Timer timer;
  int dev = 0, l2 = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  const std::size_t rot_bytes = 2 * static_cast<std::size_t>(l2) + (16u << 20);
  const char* m_env = std::getenv("MI_NVFP4_PERF_M");
  const std::int64_t m = m_env != nullptr ? std::atoll(m_env) : kM1024;
  const std::int64_t kmax = 8192, nmax = 16384;

  // Inputs: x bf16 [m, 8192] (norm / quantize / rotate sites read its first K columns), up_gate output [m, 16384].
  Buf<std::uint16_t> xb = dev_act(1, m, kmax), ug = dev_act(2, m, nmax), res = dev_act(3, m, 3072);
  Buf<std::uint16_t> gate = dev_uniform(4, 3072, 0.0f, 1.0f), out(static_cast<std::size_t>(m * nmax));
  Buf<std::uint16_t> nw = dev_uniform(5, 3072, 1.0f, 0.5f), msc = dev_uniform(6, 3072, 0.0f, 1.0f),
                     msh = dev_uniform(7, 3072, 0.0f, 1.0f), delta = dev_uniform(8, static_cast<std::size_t>(m * nmax), 0.0f, 0.1f);
  DevAct xa(m, kmax);
  Buf<std::int8_t> x8(static_cast<std::size_t>(m * kmax));
  Buf<float> s8(static_cast<std::size_t>(m));

  double block_int8 = 0, block_fp4 = 0;
  std::printf("[nvfp4 perf] M %lld, L2 %d MB, weights rotate through >= %zu MB\n", static_cast<long long>(m), l2 >> 20, rot_bytes >> 20);
  for (const Site& s : kSites) {
    if (!site_selected(s)) continue;
    const std::size_t wn = static_cast<std::size_t>(s.n * s.k);
    Rotating<std::int8_t> w8(wn, rot_bytes, [&](Buf<std::int8_t>& b, int i) {
      Buf<std::uint16_t> t = dev_uniform(100 + static_cast<std::uint64_t>(i), wn, 0.0f, 0.05f);
      // Any int8 bytes do: reinterpret bf16 halves.
      MI_CUDA_CHECK(cudaMemcpy(b.get(), t.get(), wn, cudaMemcpyDeviceToDevice));
    });
    Buf<float> ws8 = [&] {
      Buf<float> b(static_cast<std::size_t>(s.n));
      b.upload(std::vector<float>(static_cast<std::size_t>(s.n), 1e-4f));
      return b;
    }();
    // NVFP4 weights: q and sf of one quantized matrix per copy (one buffer each, sf right after q).
    const std::size_t qbytes = wn / 2, sfbytes = static_cast<std::size_t>(nv::sf_bytes(s.n, s.k));
    Rotating<std::uint8_t> w4(qbytes + sfbytes, rot_bytes, [&](Buf<std::uint8_t>& b, int i) {
      Buf<std::uint16_t> t = dev_uniform(200 + static_cast<std::uint64_t>(i), wn, 0.0f, 0.05f);
      Buf<float> g(static_cast<std::size_t>(s.n));
      nv::quantize_nvfp4(as_bf16(t), s.k, {b.get(), b.get() + qbytes, g.get()}, s.n, static_cast<int>(s.k), nullptr);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
    });

    const auto int8_ops = [&](const std::int8_t* w) {
      mi::kernels::quant::Int8Operands o;
      o.x_q = x8.get();
      o.ldx = s.k;
      o.x_scale = s8.get();
      o.w = w;
      o.w_scale = ws8.get();
      o.m = m;
      o.n = s.n;
      o.k = s.k;
      return o;
    };
    const auto fp4_ops = [&](const std::uint8_t* w) {
      nv::Nvfp4Operands o;
      o.x = xa.q.get();
      o.x_sf = xa.sf.get();
      o.x_scale = xa.g.get();
      o.w = w;
      o.w_sf = w + qbytes;
      o.w_scale2 = 1e-4f;
      o.m = m;
      o.n = s.n;
      o.k = s.k;
      return o;
    };
    namespace quant = mi::kernels::quant;
    const int k = static_cast<int>(s.k);
    const std::function<void()> q8 = [&] {
      if (s.kind == 0 || s.kind == 2) {
        quant::rms_norm_modulate_rotate_quantize(as_bf16(xb), kmax, as_bf16(nw), as_bf16(msc), as_bf16(msh), x8.get(), s.k,
                                                 s8.get(), m, k, nullptr);
      } else {
        quant::rotate_quantize(s.kind == 3 ? as_bf16(ug) : as_bf16(xb), s.kind == 3 ? nmax : kmax, x8.get(), s.k, s8.get(), m, k,
                               nullptr);
      }
    };
    const std::function<void()> g8 = [&] {
      const auto o = int8_ops(w8.get());
      if (s.kind == 0) quant::int8_gemm(o, nullptr, as_bf16(out), s.n, nullptr);
      if (s.kind == 1 || s.kind == 3) quant::int8_gemm_gated_residual(o, nullptr, as_bf16(gate), as_bf16(res), 3072, nullptr);
      if (s.kind == 2) quant::int8_gemm_situ_glu(o, as_bf16(out), s.n / 2, nullptr);
    };
    const std::function<void()> q4 = [&] {
      if (s.kind == 0 || s.kind == 2) {
        nv::rms_norm_modulate_quantize_nvfp4(as_bf16(xb), kmax, as_bf16(nw), as_bf16(msc), as_bf16(msh), xa.act(), m, k, nullptr);
      } else if (s.kind == 1) {
        nv::quantize_nvfp4(as_bf16(xb), kmax, xa.act(), m, k, nullptr);
      } else {
        nv::situ_glu_quantize_nvfp4(as_bf16(ug), nmax, xa.act(), m, k, nullptr);
      }
    };
    const auto g4 = [&](Nvfp4Tile tile, bool with_delta) {
      return std::function<void()>([&, tile, with_delta] {
        const nv::Nvfp4Delta dl = with_delta ? nv::Nvfp4Delta{as_bf16(delta), s.n} : nv::Nvfp4Delta{};
        if (s.kind == 1 || s.kind == 3) {
          nv::nvfp4_gemm_gated_residual(fp4_ops(w4.get()), nullptr, as_bf16(gate), as_bf16(res), 3072, nullptr, tile, dl);
        } else {
          nv::nvfp4_gemm(fp4_ops(w4.get()), nullptr, as_bf16(out), s.n, nullptr, tile, dl);
        }
      });
    };
    const auto both = [](std::function<void()> a, std::function<void()> b) {
      return std::function<void()>([a, b] {
        a();
        b();
      });
    };
    // Warm up into the power-capped steady state.
    for (const double t0 = wall_seconds(); wall_seconds() - t0 < 2.0;) timer.ms(both(q4, g4(Nvfp4Tile::kAuto, false)), 10);
    const Nvfp4Tile tiles[] = {Nvfp4Tile::kAuto, Nvfp4Tile::k128x128x128, Nvfp4Tile::k128x128x128Pingpong, Nvfp4Tile::k256x128x128};
    std::vector<std::function<void()>> fns = {both(q8, g8), q8, g8, q4, g4(Nvfp4Tile::kAuto, true)};
    for (Nvfp4Tile t : tiles) fns.push_back(both(q4, g4(t, false)));
    for (Nvfp4Tile t : tiles) fns.push_back(g4(t, false));
    const std::vector<double> ms = steady(timer, fns);
    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(s.n) * static_cast<double>(s.k);
    const double int8_site = ms[0];
    std::printf("[nvfp4 perf] %s M %lld N %lld K %lld\n", s.name, static_cast<long long>(m), static_cast<long long>(s.n),
                static_cast<long long>(s.k));
    std::printf("[nvfp4 perf]   int8 site (quantizer + GEMM)        %7.3f ms  (quantizer %.3f, GEMM %.3f = %6.1f TOPS)\n",
                int8_site, ms[1], ms[2], ops / ms[2] / 1e9);
    std::printf("[nvfp4 perf]   nvfp4 quantizer                     %7.3f ms\n", ms[3]);
    for (std::size_t i = 0; i < 4; ++i) {
      const double site = ms[5 + i], gemm_ms = ms[9 + i];
      std::printf("[nvfp4 perf]   nvfp4 site %-22s %7.3f ms  %5.2fx int8 | GEMM %.3f ms = %6.1f TOPS (%5.2fx int8 GEMM)\n",
                  nv::nvfp4_tile_name(i == 0 ? nv::nvfp4_tile_for(m, s.n, s.k) : tiles[i]), site, int8_site / site, gemm_ms,
                  ops / gemm_ms / 1e9, ms[2] / gemm_ms);
    }
    std::printf("[nvfp4 perf]   auto: %s; E%sL GEMM (LoRA delta) %.3f ms = +%.1f %% vs E%s\n",
                nv::nvfp4_tile_name(nv::nvfp4_tile_for(m, s.n, s.k)), s.kind == 1 || s.kind == 3 ? "2" : "1", ms[4],
                100.0 * (ms[4] / ms[9] - 1.0), s.kind == 1 || s.kind == 3 ? "2" : "1");
    RecordProperty(std::string("site_speedup_") + s.name, std::to_string(int8_site / ms[5]));
    block_int8 += int8_site;
    block_fp4 += ms[5];
  }
  std::printf("[nvfp4 perf] block (4 sites, quantizers included): int8 %.3f ms, nvfp4 %.3f ms: %.2fx\n", block_int8,
              block_fp4, block_int8 / block_fp4);
  RecordProperty("block_speedup", std::to_string(block_int8 / block_fp4));
}

// The nvfp4_tile_for sweep: every tile with the E1 / E2 epilogue at the DiT shapes over several row counts.
TEST(GpuNvfp4Kernels, TileSweep) {
  MI_REQUIRE_PERF();
  if (!nv::available()) GTEST_SKIP() << "nvfp4::available() is false";
  Timer timer;
  int dev = 0, l2 = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
  const std::size_t rot_bytes = 2 * static_cast<std::size_t>(l2) + (16u << 20);
  const std::int64_t mmax = 16384;
  Buf<std::uint16_t> out(static_cast<std::size_t>(mmax * 16384)), gate = dev_uniform(4, 3072, 0.0f, 1.0f);
  const Nvfp4Tile tiles[] = {Nvfp4Tile::k128x128x128, Nvfp4Tile::k128x128x128Pingpong, Nvfp4Tile::k256x128x128};
  for (const Site& s : kSites) {
    if (!site_selected(s)) continue;
    const std::size_t wn = static_cast<std::size_t>(s.n * s.k), qbytes = wn / 2;
    Rotating<std::uint8_t> w4(qbytes + static_cast<std::size_t>(nv::sf_bytes(s.n, s.k)), rot_bytes, [&](Buf<std::uint8_t>& b, int i) {
      Buf<std::uint16_t> t = dev_uniform(200 + static_cast<std::uint64_t>(i), wn, 0.0f, 0.05f);
      Buf<float> g(static_cast<std::size_t>(s.n));
      nv::quantize_nvfp4(as_bf16(t), s.k, {b.get(), b.get() + qbytes, g.get()}, s.n, static_cast<int>(s.k), nullptr);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
    });
    for (std::int64_t m : {std::int64_t{1172}, std::int64_t{2340}, std::int64_t{4378}, kM1024, std::int64_t{12377}, mmax}) {
      DevAct xs(m, s.k);
      {
        Buf<std::uint16_t> xb = dev_act(9, m, s.k);
        nv::quantize_nvfp4(as_bf16(xb), s.k, xs.act(), m, static_cast<int>(s.k), nullptr);
      }
      std::vector<std::function<void()>> fns;
      for (Nvfp4Tile t : tiles) {
        fns.push_back([&, t, m] {
          nv::Nvfp4Operands o;
          o.x = xs.q.get();
          o.x_sf = xs.sf.get();
          o.x_scale = xs.g.get();
          o.w = w4.get();
          o.w_sf = o.w + qbytes;
          o.w_scale2 = 1e-4f;
          o.m = m;
          o.n = s.n;
          o.k = s.k;
          if (s.kind == 1 || s.kind == 3) {
            nv::nvfp4_gemm_gated_residual(o, nullptr, as_bf16(gate), as_bf16(out), 3072, nullptr, t);
          } else {
            nv::nvfp4_gemm(o, nullptr, as_bf16(out), s.n, nullptr, t);
          }
        });
      }
      for (const double t0 = wall_seconds(); wall_seconds() - t0 < 1.0;) timer.ms(fns[2], 5);
      const std::vector<double> ms = steady(timer, fns);
      const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(s.n) * static_cast<double>(s.k);
      std::printf("[nvfp4 tiles] %-7s M %5lld N %5lld K %4lld |", s.name, static_cast<long long>(m), static_cast<long long>(s.n),
                  static_cast<long long>(s.k));
      std::size_t best = 0;
      for (std::size_t i = 0; i < ms.size(); ++i) {
        std::printf(" %s %.3f ms %6.1f TOPS |", nv::nvfp4_tile_name(tiles[i]), ms[i], ops / ms[i] / 1e9);
        if (ms[i] < ms[best]) best = i;
      }
      std::printf(" best %s, auto %s\n", nv::nvfp4_tile_name(tiles[best]), nv::nvfp4_tile_name(nv::nvfp4_tile_for(m, s.n, s.k)));
    }
  }
}

// The activation quantizers alone at M rows (read + write bandwidth), next to the INT8 rotate quantizers and the bf16
// kernels they replace (rms_norm_modulate_moss, situ_glu).
TEST(GpuNvfp4Kernels, QuantizerRates) {
  MI_REQUIRE_PERF();
  Timer timer;
  const char* m_env = std::getenv("MI_NVFP4_PERF_M");
  const std::int64_t m = m_env != nullptr ? std::atoll(m_env) : kM1024;
  Buf<std::uint16_t> xb = dev_act(1, m, 16384), out(static_cast<std::size_t>(m * 16384));
  Buf<std::uint16_t> nw = dev_uniform(5, 3072, 1.0f, 0.5f), msc = dev_uniform(6, 3072, 0.0f, 1.0f),
                     msh = dev_uniform(7, 3072, 0.0f, 1.0f);
  DevAct xa(m, 16384);
  Buf<std::int8_t> x8(static_cast<std::size_t>(m * 16384));
  Buf<float> s8(static_cast<std::size_t>(m));
  namespace quant = mi::kernels::quant;
  struct Op {
    const char* name;
    double bytes;  // read + written
    std::function<void()> fn;
  };
  const double md = static_cast<double>(m);
  const std::vector<Op> ops = {
      {"nvfp4 rms_norm_modulate_quantize K 3072", md * 3072 * (2 + 0.5625), [&] {
         nv::rms_norm_modulate_quantize_nvfp4(as_bf16(xb), 3072, as_bf16(nw), as_bf16(msc), as_bf16(msh), xa.act(), m, 3072, nullptr);
       }},
      {"nvfp4 quantize K 3072", md * 3072 * (2 + 0.5625), [&] { nv::quantize_nvfp4(as_bf16(xb), 3072, xa.act(), m, 3072, nullptr); }},
      {"nvfp4 quantize K 16384", md * 16384 * (2 + 0.5625), [&] { nv::quantize_nvfp4(as_bf16(xb), 16384, xa.act(), m, 16384, nullptr); }},
      {"nvfp4 situ_glu_quantize F 8192", md * 8192 * (4 + 0.5625), [&] { nv::situ_glu_quantize_nvfp4(as_bf16(xb), 16384, xa.act(), m, 8192, nullptr); }},
      {"int8 rms_norm_modulate_rotate_quantize K 3072", md * 3072 * 3, [&] {
         quant::rms_norm_modulate_rotate_quantize(as_bf16(xb), 3072, as_bf16(nw), as_bf16(msc), as_bf16(msh), x8.get(), 3072, s8.get(), m, 3072, nullptr);
       }},
      {"int8 rotate_quantize K 3072", md * 3072 * 3, [&] { quant::rotate_quantize(as_bf16(xb), 3072, x8.get(), 3072, s8.get(), m, 3072, nullptr); }},
      {"int8 rotate_quantize K 8192", md * 8192 * 3, [&] { quant::rotate_quantize(as_bf16(xb), 8192, x8.get(), 8192, s8.get(), m, 8192, nullptr); }},
      {"bf16 rms_norm_modulate_moss K 3072", md * 3072 * 4, [&] {
         mi::kernels::rms_norm_modulate_moss(view_bf16(xb.get(), m, 3072), vec_bf16(nw.get(), 3072), vec_bf16(msc.get(), 3072),
                                             vec_bf16(msh.get(), 3072), view_bf16(out.get(), m, 3072), nullptr);
       }},
      {"bf16 situ_glu F 8192", md * 8192 * 6, [&] { mi::kernels::situ_glu(view_bf16(xb.get(), m, 16384), view_bf16(out.get(), m, 8192), nullptr); }},
  };
  std::vector<std::function<void()>> fns;
  for (const Op& o : ops) fns.push_back(o.fn);
  const std::vector<double> ms = steady(timer, fns);
  for (std::size_t i = 0; i < ops.size(); ++i) {
    std::printf("[nvfp4 quant] %-46s %7.4f ms %6.0f GB/s\n", ops[i].name, ms[i], ops[i].bytes / ms[i] / 1e6);
  }
}

}  // namespace
