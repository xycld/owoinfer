// kernels/quant4 (ConvRot W4A4, the Turbo tier) and its offline quantizer weights/quant/convrot_w4a4.
// Rotate + INT4 quantize bit for bit against comfy-kitchen @dba9fac's eager math (tests/data/quant4: torch on CUDA,
// eager quantize_signed_int4_rowwise after the INT8 tier's fp32 FHT), against a host scalar reference (zero / tiny /
// NaN / inf rows, every block shape) and the engine's op-by-op norm chain; the INT4 GEMM epilogues bit for bit against
// torch._int_mm + the documented fp32 epilogue and, at the DiT shapes, against a slow scalar oracle kernel (exact int32
// dot product); end to end against comfy-kitchen's eager convrot_w4a4_linear. The inputs are splitmix64-hash tensors
// (synth_act / synth_vec), regenerated bit for bit by tools/reference/gen_quant4_vectors.py, which stores only the
// reference outputs. Real weights (skipped when absent): the quantization error at the DiT shapes, RTN vs GPTQ; and
// the throughput / sustained-clock runs (MI_RUN_PERF=1).
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/quant/quant.h"
#include "kernels/quant4/quant4.h"
#include "util/npy.h"
#include "weights/quant/convrot_w4a4.h"
#include "weights/quant/quant_codec.h"
#include "weights/safetensors.h"

namespace {

namespace q4 = mi::kernels::quant4;
using mi::DType;
using mi::Tensor;
using q4::Int4Tile;
using bf16_t = __nv_bfloat16;

const std::filesystem::path kData = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/data/quant4";
mi::npy::Array load(const std::string& name) { return mi::npy::load(kData / (name + ".npy")); }

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
};

bf16_t* as_bf16(const Buf<std::uint16_t>& b) { return reinterpret_cast<bf16_t*>(b.get()); }

Tensor view_bf16(void* p, std::int64_t rows, std::int64_t cols) {
  return Tensor::view(p, DType::kBF16, mi::Device::kCUDA, {rows, cols});
}

std::uint64_t mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::uint16_t bf16_bits(float v) { return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16); }
float bf16_float(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

float rbf_host(float v) {
  if (std::isnan(v)) return v;
  std::uint32_t b = std::bit_cast<std::uint32_t>(v);
  b += 0x7FFFu + ((b >> 16) & 1u);
  return std::bit_cast<float>(b & 0xFFFF0000u);
}

// DiT-like bf16 activations (as the INT8 tests): per-channel exponents, every 397th channel 64x larger.
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

// The special rows of the rot3072 vectors (as gen_quant4_vectors.py rot3072_edits): zero; quiet with one outlier;
// tiny; a constant group; amax below the 1e-10 clamp. Power-of-two scalings keep the values exact.
std::vector<std::uint16_t> rot3072_input() {
  auto x = synth_act(101, 40, 3072);
  const auto at = [&](int r, int c) -> std::uint16_t& { return x[static_cast<std::size_t>(r * 3072 + c)]; };
  for (int c = 0; c < 3072; ++c) {
    at(3, c) = 0;
    at(4, c) = bf16_bits(std::ldexp(bf16_float(at(4, c)), -10));
    at(5, c) = bf16_bits(std::ldexp(bf16_float(at(5, c)), -60));
    at(7, c) = bf16_bits(std::ldexp(bf16_float(at(7, c)), -45));
  }
  at(4, 1000) = bf16_bits(300.0f);
  for (int c = 0; c < 256; ++c) at(6, c) = bf16_bits(1.0f);
  return x;
}

template <class T>
std::size_t count_diff(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += !(a[i] == b[i]);
  return n;
}

std::size_t count_diff_bits(const std::vector<float>& a, const std::vector<float>& b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
  return n;
}

int ulp_dist(std::uint16_t a, std::uint16_t b) {
  const auto ord = [](std::uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7FFF) : static_cast<int>(v); };
  return std::abs(ord(a) - ord(b));
}

struct Agreement {
  double cosine = 0, rel_l2 = 0, identical = 0, within1 = 0;
  int max_ulp = 0;
};

// got vs ref (bf16 bit patterns): cosine, relative L2 ||got - ref|| / ||ref||, ULP histogram.
Agreement agreement(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& ref) {
  Agreement a;
  double dot = 0, ng = 0, nr = 0, nd = 0;
  std::size_t same = 0, w1 = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double g = bf16_float(got[i]), r = bf16_float(ref[i]);
    dot += g * r;
    ng += g * g;
    nr += r * r;
    nd += (g - r) * (g - r);
    const int d = ulp_dist(got[i], ref[i]);
    same += d == 0;
    w1 += d <= 1;
    a.max_ulp = std::max(a.max_ulp, d);
  }
  a.cosine = dot / std::sqrt(ng * nr);
  a.rel_l2 = std::sqrt(nd / nr);
  a.identical = static_cast<double>(same) / static_cast<double>(got.size());
  a.within1 = static_cast<double>(w1) / static_cast<double>(got.size());
  return a;
}

// Device-side fills of large operands.
__device__ std::uint64_t mix_d(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// kind 0: random bytes (every nibble -8..7); 1: f32 log-uniform in [a, a * 2^b]; 2: bf16 uniform in [a - b, a + b];
// 3: bf16 activations as synth_act; 4: bytes of nibbles in -7..7.
__global__ void fill_kernel(void* p, std::size_t n, std::uint64_t seed, int kind, float a, float b, int cols) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint64_t z = mix_d(i + seed * 0x9E3779B97F4A7C15ull);
    const float u = static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
    if (kind == 0) {
      static_cast<std::uint8_t*>(p)[i] = static_cast<std::uint8_t>(z & 0xFF);
    } else if (kind == 4) {
      const int lo = static_cast<int>(z % 15) - 7, hi = static_cast<int>((z >> 8) % 15) - 7;
      static_cast<std::uint8_t*>(p)[i] = static_cast<std::uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
    } else if (kind == 1) {
      static_cast<float*>(p)[i] = a * exp2f(u * b);

    } else {
      float v = a + b * (2.0f * u - 1.0f);
      if (kind == 3) {
        const int c = static_cast<int>(i % static_cast<std::size_t>(cols));
        const int c_exp = static_cast<int>(mix_d(static_cast<std::uint64_t>(c) * 7919 + seed) % 6) - 3 + (c % 397 == 5 ? 6 : 0);
        const float m = static_cast<float>(128 + (z & 0x7F));
        v = ldexpf((z >> 10) & 1 ? -m : m, c_exp - 7 - static_cast<int>((z >> 7) & 7));
      }
      static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
    }
  }
}

void fill(void* p, std::size_t n, std::uint64_t seed, int kind, float a = 0, float b = 0, int cols = 1) {
  fill_kernel<<<1024, 256>>>(p, n, seed, kind, a, b, cols);
  MI_CUDA_CHECK_LAUNCH();
}

Buf<std::uint8_t> dev_nibbles(std::uint64_t seed, std::size_t n, bool full_range) {
  Buf<std::uint8_t> b(n);
  fill(b.get(), n, seed, full_range ? 0 : 4);
  return b;
}
Buf<float> dev_scale(std::uint64_t seed, std::size_t n, float lo, float hi) {
  Buf<float> b(n);
  fill(b.get(), n, seed, 1, lo, std::log2(hi / lo));
  return b;
}
Buf<std::uint16_t> dev_vec(std::uint64_t seed, std::size_t n, float center, float spread) {
  Buf<std::uint16_t> b(n);
  fill(b.get(), n, seed, 2, center, spread);
  return b;
}
Buf<std::uint16_t> dev_act(std::uint64_t seed, std::int64_t rows, std::int64_t cols) {
  Buf<std::uint16_t> b(static_cast<std::size_t>(rows * cols));
  fill(b.get(), b.n, seed, 3, 0, 0, static_cast<int>(cols));
  return b;
}

__device__ __forceinline__ int nib(const std::uint8_t* row, int c) {
  const int byte = row[c >> 1];
  const int v = (c & 1) ? (byte >> 4) : (byte & 0xF);
  return v >= 8 ? v - 16 : v;
}

std::vector<int> all_rows(std::int64_t m) {
  std::vector<int> r(static_cast<std::size_t>(m));
  for (std::int64_t i = 0; i < m; ++i) r[static_cast<std::size_t>(i)] = static_cast<int>(i);
  return r;
}

std::vector<int> sample_rows(std::int64_t m) {
  std::vector<int> r;
  for (std::int64_t v : {std::int64_t{0}, std::int64_t{1}, std::int64_t{63}, std::int64_t{64}, std::int64_t{127},
                         std::int64_t{128}, std::int64_t{129}, std::int64_t{255}, std::int64_t{256}, m / 2, m - 257,
                         m - 129, m - 128, m - 2, m - 1}) {
    if (v >= 0 && v < m && std::find(r.begin(), r.end(), static_cast<int>(v)) == r.end()) r.push_back(static_cast<int>(v));
  }
  return r;
}

std::vector<std::uint16_t> download_rows(const std::uint16_t* p, std::int64_t ld, std::int64_t cols,
                                         const std::vector<int>& rows) {
  std::vector<std::uint16_t> h(rows.size() * static_cast<std::size_t>(cols));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    MI_CUDA_CHECK(cudaMemcpy(h.data() + i * static_cast<std::size_t>(cols), p + rows[i] * ld,
                             static_cast<std::size_t>(cols) * 2, cudaMemcpyDeviceToHost));
  }
  return h;
}

// =================================================================================================================
// Scheme 1: rotate + INT4 quantize.

struct RotOut {
  std::vector<std::uint8_t> q;  // packed [m, k/2]
  std::vector<float> s;
};

RotOut run_rotate(const std::vector<std::uint16_t>& x, std::int64_t m, int k, std::int64_t ldx = 0,
                  std::int64_t ldq = 0) {
  if (ldx == 0) ldx = k;
  if (ldq == 0) ldq = k / 2;
  std::vector<std::uint16_t> xp(static_cast<std::size_t>(m * ldx), 0x7FC0);  // NaN padding must not be read
  for (std::int64_t r = 0; r < m; ++r) std::copy_n(x.begin() + r * k, k, xp.begin() + r * ldx);
  Buf<std::uint16_t> xd(xp);
  Buf<std::uint8_t> q(std::vector<std::uint8_t>(static_cast<std::size_t>(m * ldq), std::uint8_t{0x55}));
  Buf<float> s(static_cast<std::size_t>(m));
  q4::convrot_quantize_int4(as_bf16(xd), ldx, q.get(), ldq, s.get(), m, k, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const auto qh = q.download();
  RotOut out{std::vector<std::uint8_t>(static_cast<std::size_t>(m * k / 2)), s.download()};
  for (std::int64_t r = 0; r < m; ++r) {
    std::copy_n(qh.begin() + r * ldq, k / 2, out.q.begin() + r * k / 2);
    for (std::int64_t c = k / 2; c < ldq; ++c) {
      EXPECT_EQ(qh[static_cast<std::size_t>(r * ldq + c)], 0x55) << "q padding written";
    }
  }
  return out;
}

// Host scalar reference of quant4.h's scheme-1 activation path: the INT8 tier's fp32 FHT (one op per step), bf16,
// then the eager INT4 row rule with the documented non-finite handling.
RotOut rotate_quantize_host(const std::vector<std::uint16_t>& x, std::int64_t m, int k) {
  constexpr float kBf16Max = 3.38953139e38f;
  RotOut out{std::vector<std::uint8_t>(static_cast<std::size_t>(m * k / 2)), std::vector<float>(static_cast<std::size_t>(m))};
  std::vector<float> row(static_cast<std::size_t>(k));
  for (std::int64_t r = 0; r < m; ++r) {
    for (int g = 0; g < k / 256; ++g) {
      float v[256], y[256];
      for (int e = 0; e < 256; ++e) v[e] = bf16_float(x[static_cast<std::size_t>(r * k + g * 256 + e)]);
      for (int st : {1, 4, 16, 64}) {
        for (int e = 0; e < 256; ++e) {
          const int d = (e / st) & 3, base = e - d * st;
          const float x0 = v[base], x1 = v[base + st], x2 = v[base + 2 * st], x3 = v[base + 3 * st];
          const float sum = d == 0   ? ((x0 + x1) + x2) - x3
                            : d == 1 ? ((x0 + x1) - x2) + x3
                            : d == 2 ? ((x0 - x1) + x2) + x3
                                     : ((-x0 + x1) + x2) + x3;
          y[e] = sum * 0.5f;
        }
        std::copy_n(y, 256, v);
      }
      for (int e = 0; e < 256; ++e) row[static_cast<std::size_t>(g * 256 + e)] = rbf_host(v[e]);
    }
    float amax = 0.0f;
    for (float v : row) amax = std::fmax(amax, std::fabs(v));
    const float amax_bf = rbf_host(std::fmax(std::fmin(amax, kBf16Max), 1e-10f));
    const float s = rbf_host(amax_bf * (1.0f / 7.0f));
    out.s[static_cast<std::size_t>(r)] = s;
    for (int c = 0; c < k; c += 2) {
      int qq[2];
      for (int j = 0; j < 2; ++j) {
        const float qv = std::nearbyint(rbf_host(row[static_cast<std::size_t>(c + j)] / s));
        qq[j] = static_cast<int>(std::fmin(7.0f, std::fmax(-7.0f, qv)));
      }
      out.q[static_cast<std::size_t>((r * k + c) / 2)] = static_cast<std::uint8_t>((qq[0] & 0xF) | ((qq[1] & 0xF) << 4));
    }
  }
  return out;
}

TEST(GpuQuant4Kernels, ConvRotQuantizeMatchesReference) {
  for (const char* name : {"rot3072", "rot8192"}) {
    const bool k3072 = std::string(name) == "rot3072";
    const std::int64_t m = k3072 ? 40 : 9;
    const int k = k3072 ? 3072 : 8192;
    const auto x = k3072 ? rot3072_input() : synth_act(102, m, k);
    const auto q_ref = load(std::string(name) + "_q").to_vector<std::uint8_t>();
    const auto s_ref = load(std::string(name) + "_s").to_vector<float>();
    const RotOut got = run_rotate(x, m, k);
    EXPECT_EQ(count_diff(got.q, q_ref), 0u) << name << " int4 bytes";
    EXPECT_EQ(count_diff_bits(got.s, s_ref), 0u) << name << " scales";
    const RotOut strided = run_rotate(x, m, k, k + 64, k / 2 + 32);
    EXPECT_EQ(count_diff(strided.q, q_ref), 0u) << name << " strided";
    const RotOut host = rotate_quantize_host(x, m, k);
    EXPECT_EQ(count_diff(host.q, q_ref), 0u) << name << " host reference";
    EXPECT_EQ(count_diff_bits(host.s, s_ref), 0u) << name << " host reference scales";
    std::printf("[quant4] %s: %lld x %d bit-exact vs eager INT4 row quantization\n", name, static_cast<long long>(m), k);
  }
}

// Every block shape (k 256, 768: odd group count, 12288, 32768) with zero, tiny, NaN, +-inf and overflowing rows.
TEST(GpuQuant4Kernels, ConvRotQuantizeMatchesHostReference) {
  const std::uint16_t kNaN = 0x7FC0, kInf = 0x7F80, kNegInf = 0xFF80, kBig = 0x7F70;
  for (int k : {256, 768, 12288, 32768}) {
    constexpr std::int64_t m = 11;
    auto x = synth_act(static_cast<std::uint64_t>(k) * 5 + 3, m, k);
    const auto at = [&](std::int64_t r, std::int64_t c) -> std::uint16_t& { return x[static_cast<std::size_t>(r * k + c)]; };
    for (int c = 0; c < k; ++c) {
      at(1, c) = 0;
      at(8, c) = kNaN;
      at(10, c) = bf16_bits(std::ldexp(c % 2 ? 1.0f : -1.0f, -40));  // amax below the 1e-10 clamp
    }
    at(9, 0) = kBig;
    at(9, 1) = kBig;
    at(2, k / 3) = kNaN;
    at(3, k - 1) = kInf;
    at(4, 5) = kNegInf;
    at(5, 0) = kNaN;
    at(5, k / 2 + 7) = kInf;
    at(5, k - 3) = kNegInf;
    at(6, 17) = kInf;
    at(6, 18) = kNegInf;
    const RotOut ref = rotate_quantize_host(x, m, k);
    const RotOut got = run_rotate(x, m, k);
    EXPECT_EQ(count_diff(got.q, ref.q), 0u) << "k " << k;
    EXPECT_EQ(count_diff_bits(got.s, ref.s), 0u) << "k " << k;
    EXPECT_EQ(ref.s[1], rbf_host(rbf_host(1e-10f) * (1.0f / 7.0f)));
    EXPECT_EQ(ref.s[10], rbf_host(rbf_host(1e-10f) * (1.0f / 7.0f)));
    EXPECT_EQ(std::count(ref.q.begin() + k / 2, ref.q.begin() + k, std::uint8_t{0}), k / 2);
    EXPECT_EQ(std::count(ref.q.begin() + 8 * k / 2, ref.q.begin() + 9 * k / 2, std::uint8_t{0x99}), k / 2);  // -7, -7
    EXPECT_TRUE(std::isfinite(ref.s[3]) && std::isfinite(ref.s[2]) && ref.s[2] > 0);
  }
}

struct NormInputs {
  Buf<std::uint16_t> weight, scale, shift;
};

// Fused norm + rotate + quantize against kernels::rms_norm_modulate_moss then convrot_quantize_int4.
TEST(GpuQuant4Kernels, FusedNormMatchesOpByOpAllRowCounts) {
  struct Case {
    int k;
    std::vector<std::int64_t> rows;
  };
  const Case cases[] = {
      {3072, {1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 31, 33, 64, 400, 4158, 8484}},
      {1024, {1, 7, 16, 100}},
      {7680, {1, 3}},
  };
  for (const Case& c : cases) {
    NormInputs v{Buf<std::uint16_t>(synth_vec(c.k + 1, static_cast<std::size_t>(c.k), 1.0f, 0.6f)),
                 Buf<std::uint16_t>(synth_vec(c.k + 2, static_cast<std::size_t>(c.k), 0.0f, 1.0f)),
                 Buf<std::uint16_t>(synth_vec(c.k + 3, static_cast<std::size_t>(c.k), 0.0f, 1.0f))};
    const auto vec = [&](const Buf<std::uint16_t>& b) {
      return Tensor::view(b.get(), DType::kBF16, mi::Device::kCUDA, {std::int64_t{c.k}});
    };
    for (std::int64_t m : c.rows) {
      const auto x = synth_act(static_cast<std::uint64_t>(m * 131 + c.k), m, c.k);
      Buf<std::uint16_t> xd(x), y(x.size());
      Buf<std::uint8_t> q(x.size() / 2);
      Buf<float> s(static_cast<std::size_t>(m));
      q4::rms_norm_modulate_convrot_quantize_int4(as_bf16(xd), c.k, as_bf16(v.weight), as_bf16(v.scale),
                                                  as_bf16(v.shift), q.get(), c.k / 2, s.get(), m, c.k, nullptr);
      mi::kernels::rms_norm_modulate_moss(view_bf16(xd.get(), m, c.k), vec(v.weight), vec(v.scale), vec(v.shift),
                                          view_bf16(y.get(), m, c.k), nullptr);
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      const RotOut chain = run_rotate(y.download(), m, c.k);
      EXPECT_EQ(count_diff(q.download(), chain.q), 0u) << "k " << c.k << " rows " << m;
      EXPECT_EQ(count_diff_bits(s.download(), chain.s), 0u) << "k " << c.k << " rows " << m;
    }
  }
}

TEST(GpuQuant4Kernels, ConvRotQuantizeRejectsBadArguments) {
  Buf<std::uint16_t> x(16 * 8192);
  Buf<std::uint8_t> q(16 * 8192);
  Buf<float> s(16);
  Buf<std::uint16_t> v(8192);
  bf16_t* xp = as_bf16(x);
  const auto rot = [&](const bf16_t* xx, std::int64_t ldx, std::uint8_t* qq, std::int64_t ldq, int k) {
    q4::convrot_quantize_int4(xx, ldx, qq, ldq, s.get(), 4, k, nullptr);
  };
  EXPECT_THROW(rot(xp, 3000, q.get(), 1500, 3000), std::invalid_argument);
  EXPECT_THROW(rot(xp, 0, q.get(), 0, 0), std::invalid_argument);
  EXPECT_THROW(rot(xp + 1, 3072, q.get(), 1536, 3072), std::invalid_argument);
  EXPECT_THROW(rot(xp, 3072, q.get() + 8, 1536, 3072), std::invalid_argument);
  EXPECT_THROW(rot(xp, 3072, q.get(), 1544, 3072), std::invalid_argument);  // ldq % 16
  EXPECT_THROW(rot(xp, 3072, q.get(), 1520, 3072), std::invalid_argument);  // ldq < k / 2
  EXPECT_NO_THROW(q4::convrot_quantize_int4(xp, 3072, q.get(), 1536, s.get(), 0, 3072, nullptr));
  EXPECT_THROW(q4::rms_norm_modulate_convrot_quantize_int4(xp, 8192, as_bf16(v), as_bf16(v), as_bf16(v), q.get(), 4096,
                                                          s.get(), 1, 8192, nullptr),
               std::invalid_argument);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// =================================================================================================================
// Scheme 1: INT4 GEMM.

// Scalar oracle: exact int32 dot product, then y = bf16(((float)acc * xs) * ws + bias); mode 1: residual +
// bf16(y * gate); mode 2: no bias (SiTU input).
__global__ void int4_oracle_kernel(const std::uint8_t* x, std::int64_t ldx, const std::uint8_t* w, const float* xs,
                                   const float* ws, const bf16_t* bias, const bf16_t* gate, const bf16_t* residual,
                                   std::int64_t ldr, const int* rows, int nrows, int n, int k, int mode, bf16_t* out,
                                   std::int32_t* acc_out) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(nrows) * n) return;
  const int i = static_cast<int>(idx / n), c = static_cast<int>(idx % n);
  const int r = rows[i];
  const std::uint8_t* xr = x + r * ldx;
  const std::uint8_t* wr = w + static_cast<std::int64_t>(c) * (k / 2);
  std::int32_t acc = 0;
  for (int j = 0; j < k; ++j) acc += nib(xr, j) * nib(wr, j);
  if (acc_out != nullptr) acc_out[idx] = acc;
  const float b = (bias != nullptr && mode != 2) ? __bfloat162float(bias[c]) : 0.0f;
  const float y = __bfloat162float(
      __float2bfloat16_rn(__fadd_rn(__fmul_rn(__fmul_rn(__int2float_rn(acc), xs[r]), ws[c]), b)));
  float v = y;
  if (mode == 1) {
    const float prod = __bfloat162float(__float2bfloat16_rn(__fmul_rn(y, __bfloat162float(gate[c]))));
    v = __fadd_rn(__bfloat162float(residual[r * ldr + c]), prod);
  }
  out[idx] = __float2bfloat16_rn(v);
}

struct Int4Data {
  std::int64_t m, n, k, ldx;  // ldx in bytes
  Buf<std::uint8_t> x, w;
  Buf<float> xs, ws;
  Buf<std::uint16_t> bias, gate;
  Int4Data(std::int64_t m_, std::int64_t n_, std::int64_t k_, std::uint64_t seed, std::int64_t ldx_ = 0)
      : m(m_), n(n_), k(k_), ldx(ldx_ > 0 ? ldx_ : k_ / 2),
        x(dev_nibbles(seed, static_cast<std::size_t>(m_ * ldx), true)),
        w(dev_nibbles(seed + 1, static_cast<std::size_t>(n_ * k_ / 2), true)),
        xs(dev_scale(seed + 2, static_cast<std::size_t>(m_), 0.002f, 0.2f)),
        ws(dev_scale(seed + 3, static_cast<std::size_t>(n_), 1e-5f, 1e-3f)),
        bias(dev_vec(seed + 4, static_cast<std::size_t>(n_), 0.0f, 2.0f)),
        gate(dev_vec(seed + 5, static_cast<std::size_t>(n_), 0.0f, 1.5f)) {}
  q4::Int4Operands ops() const {
    q4::Int4Operands o;
    o.x_q = x.get();
    o.ldx = ldx;
    o.x_scale = xs.get();
    o.w = w.get();
    o.w_scale = ws.get();
    o.m = m;
    o.n = n;
    o.k = k;
    return o;
  }
};

std::vector<std::uint16_t> int4_oracle(const Int4Data& d, const std::vector<int>& rows, int mode, const bf16_t* bias,
                                       const bf16_t* residual, std::vector<std::int32_t>* acc = nullptr) {
  Buf<int> r(rows);
  Buf<std::uint16_t> out(rows.size() * static_cast<std::size_t>(d.n));
  Buf<std::int32_t> a(acc != nullptr ? out.n : 1);
  const std::int64_t total = static_cast<std::int64_t>(out.n);
  int4_oracle_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
      d.x.get(), d.ldx, d.w.get(), d.xs.get(), d.ws.get(), bias, as_bf16(d.gate), residual, d.n, r.get(),
      static_cast<int>(rows.size()), static_cast<int>(d.n), static_cast<int>(d.k), mode, as_bf16(out),
      acc != nullptr ? a.get() : nullptr);
  MI_CUDA_CHECK_LAUNCH();
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  if (acc != nullptr) *acc = a.download();
  return out.download();
}

std::vector<std::uint16_t> situ_of(const std::vector<std::uint16_t>& y, std::int64_t rows, std::int64_t n) {
  Buf<std::uint16_t> yd(y), out(static_cast<std::size_t>(rows * n / 2));
  mi::kernels::situ_glu(view_bf16(yd.get(), rows, n), view_bf16(out.get(), rows, n / 2), nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return out.download();
}

constexpr std::initializer_list<Int4Tile> kTiles4 = {Int4Tile::kAuto, Int4Tile::k128x256x128, Int4Tile::k256x128x128,
                                                     Int4Tile::k64x128x128};

void check_int4_against_oracle(const Int4Data& d, const std::vector<int>& rows, std::initializer_list<Int4Tile> tiles,
                               bool with_bias, const std::string& label) {
  const std::size_t mn = static_cast<std::size_t>(d.m * d.n);
  Buf<std::uint16_t> out(mn), res(mn), res0 = dev_act(static_cast<std::uint64_t>(d.m * 31 + d.n), d.m, d.n);
  const bf16_t* bias = with_bias ? as_bf16(d.bias) : nullptr;
  const auto ref_plain = int4_oracle(d, rows, 0, bias, nullptr);
  const auto ref_res = int4_oracle(d, rows, 1, bias, as_bf16(res0));
  const auto ref_situ = situ_of(int4_oracle(d, rows, 2, nullptr, nullptr), static_cast<std::int64_t>(rows.size()), d.n);
  for (Int4Tile tile : tiles) {
    const std::string tag = label + " tile " + q4::int4_tile_name(tile);
    q4::int4_gemm(d.ops(), bias, as_bf16(out), d.n, nullptr, tile);
    EXPECT_EQ(count_diff(download_rows(out.get(), d.n, d.n, rows), ref_plain), 0u) << tag << " int4_gemm";
    MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), mn * 2, cudaMemcpyDeviceToDevice));
    q4::int4_gemm_gated_residual(d.ops(), bias, as_bf16(d.gate), as_bf16(res), d.n, nullptr, tile);
    EXPECT_EQ(count_diff(download_rows(res.get(), d.n, d.n, rows), ref_res), 0u) << tag << " gated_residual";
    q4::int4_gemm_situ_glu(d.ops(), as_bf16(out), d.n / 2, nullptr, tile);
    EXPECT_EQ(count_diff(download_rows(out.get(), d.n / 2, d.n / 2, rows), ref_situ), 0u) << tag << " situ_glu";
  }
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// The torch vectors: oracle == torch._int_mm + the CUDA-order dequant; every tile bit-exact; the agreement with
// comfy-kitchen's eager int4_linear (bf16 integer matmul) reported and bounded.
TEST(GpuQuant4Kernels, Int4GemmMatchesReference) {
  const auto xa = load("gemm_xq"), wa = load("gemm_wq");
  const std::int64_t m = xa.shape[0], kb = xa.shape[1], n = wa.shape[0], k = 2 * kb;
  Buf<std::uint8_t> x(xa.to_vector<std::uint8_t>()), w(wa.to_vector<std::uint8_t>());
  Buf<float> xs(load("gemm_xs").to_vector<float>()), ws(load("gemm_ws").to_vector<float>());
  Buf<std::uint16_t> bias(synth_vec(105, static_cast<std::size_t>(n), 0.0f, 0.2f)),
      gate(synth_vec(106, static_cast<std::size_t>(n), 0.0f, 0.8f));
  const auto residual = synth_act(107, m, n);
  q4::Int4Operands op;
  op.x_q = x.get();
  op.ldx = kb;
  op.x_scale = xs.get();
  op.w = w.get();
  op.w_scale = ws.get();
  op.m = m;
  op.n = n;
  op.k = k;
  Buf<std::uint16_t> out(static_cast<std::size_t>(m * n)), res(residual), situ(static_cast<std::size_t>(m * n / 2));
  const auto y_ref = load("gemm_y").to_vector<std::uint16_t>(), yb_ref = load("gemm_y_bias").to_vector<std::uint16_t>();
  for (Int4Tile tile : kTiles4) {
    const std::string tag = q4::int4_tile_name(tile);
    q4::int4_gemm(op, nullptr, as_bf16(out), n, nullptr, tile);
    EXPECT_EQ(count_diff(out.download(), y_ref), 0u) << tag << " no bias";
    q4::int4_gemm(op, as_bf16(bias), as_bf16(out), n, nullptr, tile);
    EXPECT_EQ(count_diff(out.download(), yb_ref), 0u) << tag << " bias";
    res.upload(residual);
    q4::int4_gemm_gated_residual(op, nullptr, as_bf16(gate), as_bf16(res), n, nullptr, tile);
    EXPECT_EQ(count_diff(res.download(), load("gemm_res").to_vector<std::uint16_t>()), 0u) << tag << " residual";
    q4::int4_gemm_situ_glu(op, as_bf16(situ), n / 2, nullptr, tile);
    EXPECT_EQ(count_diff(situ.download(), load("gemm_situ").to_vector<std::uint16_t>()), 0u) << tag << " situ";
  }
  q4::int4_gemm(op, as_bf16(bias), as_bf16(out), n, nullptr);
  const Agreement e = agreement(out.download(), load("gemm_y_eager_bias").to_vector<std::uint16_t>());
  std::printf("[quant4] int4_gemm vs eager int4_linear (bf16 integer matmul): cos %.8f, %.2f %% identical, "
              "%.2f %% within 1 ULP, max %d ULP\n", e.cosine, 100 * e.identical, 100 * e.within1, e.max_ulp);
  EXPECT_GT(e.cosine, 0.9999);
  RecordProperty("eager_int4_linear_cos", std::to_string(e.cosine));
}

// DiT shapes (M = 8484 = 1024^2 CFG batch + text) and the text-encoder-like small M, every tile, sampled rows.
TEST(GpuQuant4Kernels, Int4GemmMatchesOracleAtModelShapes) {
  struct Shape {
    std::int64_t n, k;
  };
  for (const Shape s : {Shape{9216, 3072}, Shape{3072, 3072}, Shape{16384, 3072}, Shape{3072, 8192}}) {
    const Int4Data d(8484, s.n, s.k, static_cast<std::uint64_t>(s.n + s.k));
    check_int4_against_oracle(d, sample_rows(d.m), kTiles4, true,
                              "8484x" + std::to_string(s.n) + "x" + std::to_string(s.k));
  }
  const Int4Data small(282, 4096, 4096, 99);
  check_int4_against_oracle(small, all_rows(small.m), {Int4Tile::kAuto, Int4Tile::k64x128x128}, false, "282x4096");
}

// Edge shapes: m = 1, 7, 129; K residues (k % 128 != 0); a padded x row stride; the nibble -8 present (full range).
TEST(GpuQuant4Kernels, Int4GemmEdgeShapes) {
  struct Case {
    std::int64_t m, n, k, ldx;
  };
  for (const Case c : {Case{1, 264, 512, 0}, Case{7, 128, 96, 0}, Case{129, 520, 1056, 0}, Case{300, 256, 3072, 1600},
                       Case{65, 8, 32, 16}}) {
    const Int4Data d(c.m, c.n, c.k, static_cast<std::uint64_t>(c.m * 7 + c.k), c.ldx);
    check_int4_against_oracle(d, all_rows(d.m), kTiles4, true,
                              std::to_string(c.m) + "x" + std::to_string(c.n) + "x" + std::to_string(c.k));
  }
}

TEST(GpuQuant4Kernels, Int4GemmRejectsBadOperands) {
  Int4Data d(64, 256, 512, 5);
  Buf<std::uint16_t> out(64 * 256);
  const auto bad = [&](auto mutate) {
    q4::Int4Operands o = d.ops();
    mutate(o);
    EXPECT_THROW(q4::int4_gemm(o, nullptr, as_bf16(out), 256, nullptr), std::invalid_argument);
  };
  bad([](q4::Int4Operands& o) { o.k = 48; });
  bad([](q4::Int4Operands& o) { o.ldx = 200; });
  bad([](q4::Int4Operands& o) { o.ldx = 8; });
  bad([](q4::Int4Operands& o) { o.n = 260; });
  bad([](q4::Int4Operands& o) { o.w = nullptr; });
  bad([](q4::Int4Operands& o) { o.x_q += 8; });
  bad([](q4::Int4Operands& o) { o.m = -1; });
  EXPECT_THROW(q4::int4_gemm(d.ops(), nullptr, as_bf16(out) + 4, 256, nullptr), std::invalid_argument);
  EXPECT_THROW(q4::int4_gemm_situ_glu(d.ops(), as_bf16(out), 100, nullptr), std::invalid_argument);
  EXPECT_THROW(q4::int4_gemm(d.ops(), nullptr, as_bf16(out), 256, nullptr, static_cast<Int4Tile>(17)),
               std::invalid_argument);
  q4::Int4Operands empty = d.ops();
  empty.m = 0;
  EXPECT_NO_THROW(q4::int4_gemm(empty, nullptr, as_bf16(out), 256, nullptr));
}

// End to end against comfy-kitchen's eager convrot_w4a4_linear: nearly the same quantized activations (eager's bf16
// matmul rotation agrees with the FHT on all but ~1e-6 of values and ~2e-7 of int4 values), output differing mainly by
// eager's bf16 integer matmul.
TEST(GpuQuant4Kernels, ConvRotLinearMatchesEager) {
  const auto wa = load("e2e_wq");
  const std::int64_t m = 53, k = 3072, n = wa.shape[0];
  Buf<std::uint16_t> x(synth_act(108, m, k)), out(static_cast<std::size_t>(m * n));
  Buf<std::uint8_t> w(wa.to_vector<std::uint8_t>()), xq(static_cast<std::size_t>(m * k / 2));
  Buf<float> ws(load("e2e_ws").to_vector<float>()), xs(static_cast<std::size_t>(m));
  q4::convrot_quantize_int4(as_bf16(x), k, xq.get(), k / 2, xs.get(), m, static_cast<int>(k), nullptr);
  q4::Int4Operands op;
  op.x_q = xq.get();
  op.ldx = k / 2;
  op.x_scale = xs.get();
  op.w = w.get();
  op.w_scale = ws.get();
  op.m = m;
  op.n = n;
  op.k = k;
  q4::int4_gemm(op, nullptr, as_bf16(out), n, nullptr);
  const Agreement e = agreement(out.download(), load("e2e_y_eager").to_vector<std::uint16_t>());
  std::printf("[quant4] rotate + int4_gemm vs eager convrot_w4a4_linear: cos %.8f, rel L2 %.2e, %.2f %% identical, "
              "%.2f %% within 1 ULP, max %d ULP\n", e.cosine, e.rel_l2, 100 * e.identical, 100 * e.within1, e.max_ulp);
  EXPECT_GT(e.cosine, 0.9999);
  EXPECT_GT(e.within1, 0.95);
}

// The activation and GEMM kernels captured into a CUDA graph on a non-blocking stream (the engine's thread-local
// capture mode), replayed twice, against the oracles. Under ctest each test is its own process, so the first launch of
// each kernel (module load, CUTLASS's shared-memory attribute) happens inside the capture.
TEST(GpuQuant4Kernels, GraphCaptureOnStream) {
  const Int4Data d4(200, 264, 512, 4243);
  Buf<std::uint16_t> o4(static_cast<std::size_t>(d4.m * d4.n)), res0 = dev_act(17, d4.m, d4.n),
      res(static_cast<std::size_t>(d4.m * d4.n)), situ(static_cast<std::size_t>(d4.m * d4.n / 2));
  MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), res.n * 2, cudaMemcpyDeviceToDevice));
  const auto xr = synth_act(29, 64, 768);
  Buf<std::uint16_t> xd(xr);
  Buf<std::uint8_t> q1(xr.size() / 2);
  Buf<float> s1(64);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  cudaStream_t stream = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph = nullptr, graph_res = nullptr;
  cudaGraphExec_t exec = nullptr, exec_res = nullptr;
  MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  q4::convrot_quantize_int4(as_bf16(xd), 768, q1.get(), 384, s1.get(), 64, 768, stream);
  q4::int4_gemm(d4.ops(), as_bf16(d4.bias), as_bf16(o4), d4.n, stream, Int4Tile::k64x128x128);
  q4::int4_gemm_situ_glu(d4.ops(), as_bf16(situ), d4.n / 2, stream, Int4Tile::k128x256x128);
  MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  // The gated residual updates in place: captured alone and replayed once.
  MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  q4::int4_gemm_gated_residual(d4.ops(), nullptr, as_bf16(d4.gate), as_bf16(res), d4.n, stream, Int4Tile::k256x128x128);
  MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_res));
  MI_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  MI_CUDA_CHECK(cudaGraphInstantiate(&exec_res, graph_res, 0));
  MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  MI_CUDA_CHECK(cudaGraphLaunch(exec_res, stream));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  const RotOut ref = rotate_quantize_host(xr, 64, 768);
  EXPECT_EQ(count_diff(q1.download(), ref.q), 0u) << "convrot_quantize_int4";
  EXPECT_EQ(count_diff_bits(s1.download(), ref.s), 0u);
  const auto rows = all_rows(d4.m);
  EXPECT_EQ(count_diff(o4.download(), int4_oracle(d4, rows, 0, as_bf16(d4.bias), nullptr)), 0u) << "int4_gemm";
  EXPECT_EQ(count_diff(situ.download(), situ_of(int4_oracle(d4, rows, 2, nullptr, nullptr), d4.m, d4.n)), 0u) << "situ";
  EXPECT_EQ(count_diff(res.download(), int4_oracle(d4, rows, 1, nullptr, as_bf16(res0))), 0u) << "gated_residual";
  MI_CUDA_CHECK(cudaGraphExecDestroy(exec_res));
  MI_CUDA_CHECK(cudaGraphDestroy(graph_res));
  MI_CUDA_CHECK(cudaGraphExecDestroy(exec));
  MI_CUDA_CHECK(cudaGraphDestroy(graph));
  MI_CUDA_CHECK(cudaStreamDestroy(stream));
}

// =================================================================================================================
// Offline quantizer (CPU): weights/quant/convrot_w4a4.

namespace mq = mi::quant;

// The weight of the e2e vectors (synth_vec 109), [128, 3072].
std::vector<std::uint16_t> e2e_weight() { return synth_vec(109, 128 * 3072, 0.0f, 0.035f); }

// Rotated bf16 rows (W_rot of convrot_w4a4.h) and the squared error of one row's quantization.
std::vector<float> rotated_rows(const std::vector<std::uint16_t>& w, std::int64_t n, std::int64_t k) {
  std::vector<float> f(w.size()), rot(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) f[i] = bf16_float(w[i]);
  mq::convrot_rows(f.data(), n, k, 256, rot.data());
  for (float& v : rot) v = rbf_host(v);
  return rot;
}

double row_sq_error(const float* rot, const std::uint8_t* q, float s, std::int64_t k) {
  std::vector<std::int8_t> qi(static_cast<std::size_t>(k));
  mq::decode_int4_rows(q, 1, k, qi.data());
  double e = 0;
  for (std::int64_t c = 0; c < k; ++c) {
    const double d = static_cast<double>(rot[c]) - static_cast<double>(s) * qi[static_cast<std::size_t>(c)];
    e += d * d;
  }
  return e;
}

// Against comfy-kitchen's eager quantize_convrot_w4a4_weight on the same bf16 weight: identical except where eager's
// bf16 rotation GEMM rounds a rotated value differently from the fp32 FHT (rare). The MSE scale search never has a
// larger squared row error than the absmax scale (the absmax scale is one of its candidates), row by row.
TEST(Quant4Prepare, ConvRotMatchesEagerAndMseNeverWorse) {
  const std::int64_t n = 128, k = 3072;
  const auto w = e2e_weight();
  const mq::ConvRotW4A4Weight got = mq::quantize_convrot_w4a4(w.data(), n, k);
  const auto q_ref = load("e2e_wq").to_vector<std::uint8_t>();
  const auto s_ref = load("e2e_ws").to_vector<float>();
  const std::size_t dq = count_diff(got.q, q_ref), ds = count_diff_bits(got.scale, s_ref);
  std::printf("[quant4] quantize_convrot_w4a4 vs eager: %zu / %zu bytes, %zu / %zu scales differ\n", dq, q_ref.size(), ds,
              s_ref.size());
  EXPECT_LT(static_cast<double>(dq), 1e-3 * static_cast<double>(q_ref.size()));
  EXPECT_LE(ds, 2u);
  mq::ConvRotW4A4Options mse_opt;
  mse_opt.search = mq::W4ScaleSearch::kMse;
  const mq::ConvRotW4A4Weight mse = mq::quantize_convrot_w4a4(w.data(), n, k, mse_opt);
  const auto rot = rotated_rows(w, n, k);
  int better = 0;
  for (std::int64_t r = 0; r < n; ++r) {
    const double e_abs = row_sq_error(rot.data() + r * k, got.q.data() + r * k / 2, got.scale[static_cast<std::size_t>(r)], k);
    const double e_mse = row_sq_error(rot.data() + r * k, mse.q.data() + r * k / 2, mse.scale[static_cast<std::size_t>(r)], k);
    EXPECT_LE(e_mse, e_abs) << "row " << r;
    better += e_mse < e_abs;
  }
  std::printf("[quant4] MSE scale search: %d / %lld rows strictly better than absmax\n", better, static_cast<long long>(n));
  EXPECT_THROW(mq::quantize_convrot_w4a4(w.data(), n, 3000), std::invalid_argument);
  auto bad = w;
  bad[77] = 0x7FC0;
  EXPECT_THROW(mq::quantize_convrot_w4a4(bad.data(), n, k), std::invalid_argument);
}

// Standard normal samples from raw mt19937_64 output (Box-Muller; std::normal_distribution is implementation-defined).
class Normal {
 public:
  explicit Normal(std::uint64_t seed) : rng_(seed) {}
  double operator()() {
    const double u1 = (static_cast<double>(rng_() >> 11) + 0.5) * 0x1.0p-53;
    const double u2 = static_cast<double>(rng_() >> 11) * 0x1.0p-53;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }

 private:
  std::mt19937_64 rng_;
};

// Correlated calibration inputs X [rows, k] (16 latent factors + noise) and a weight [n, k].
struct GptqCase {
  std::int64_t n = 64, k = 512, rows = 2048;
  std::vector<std::uint16_t> w, x;
  std::vector<double> hess;
  GptqCase() {
    Normal nd(7);
    std::vector<double> mixing(static_cast<std::size_t>(16 * k));
    for (double& v : mixing) v = nd();
    x.resize(static_cast<std::size_t>(rows * k));
    for (std::int64_t r = 0; r < rows; ++r) {
      double z[16];
      for (double& v : z) v = nd();
      for (std::int64_t c = 0; c < k; ++c) {
        double v = 0.1 * nd();
        for (int j = 0; j < 16; ++j) v += z[j] * mixing[static_cast<std::size_t>(j * k + c)];
        x[static_cast<std::size_t>(r * k + c)] = bf16_bits(static_cast<float>(v));
      }
    }
    w.resize(static_cast<std::size_t>(n * k));
    for (auto& v : w) v = bf16_bits(static_cast<float>(0.02 * nd()));
    hess = mq::gram_bf16(x.data(), rows, k, k);
  }
  // ||X (W_eff - W)^T||_F / ||X W^T||_F, W_eff the dequantized weight in the original basis.
  double output_error(const mq::ConvRotW4A4Weight& qw) const {
    std::vector<std::int8_t> qi(static_cast<std::size_t>(n * k));
    mq::decode_int4_rows(qw.q.data(), n, k, qi.data());
    std::vector<float> eff(static_cast<std::size_t>(n * k));
    mq::dequantize_rows(qi.data(), qw.scale.data(), n, k, 256, eff.data());
    double num = 0, den = 0;
    for (std::int64_t r = 0; r < rows; ++r) {
      for (std::int64_t o = 0; o < n; ++o) {
        double y = 0, yq = 0;
        for (std::int64_t c = 0; c < k; ++c) {
          const double xv = bf16_float(x[static_cast<std::size_t>(r * k + c)]);
          y += xv * bf16_float(w[static_cast<std::size_t>(o * k + c)]);
          yq += xv * eff[static_cast<std::size_t>(o * k + c)];
        }
        num += (yq - y) * (yq - y);
        den += y * y;
      }
    }
    return std::sqrt(num / den);
  }
};

// GPTQ lowers the output error on the calibration inputs it was given, for both scale searches; and every mode is
// independent of the thread count, byte for byte.
TEST(Quant4Prepare, GptqAndThreadIndependence) {
  const GptqCase c;
  for (mq::W4ScaleSearch search : {mq::W4ScaleSearch::kAbsmax, mq::W4ScaleSearch::kMse}) {
    mq::ConvRotW4A4Options o;
    o.search = search;
    const double rtn = c.output_error(mq::quantize_convrot_w4a4(c.w.data(), c.n, c.k, o));
    o.hessian = c.hess.data();
    const double gptq = c.output_error(mq::quantize_convrot_w4a4(c.w.data(), c.n, c.k, o));
    std::printf("[quant4] %s scales: output rel err RTN %.4f, GPTQ %.4f\n",
                search == mq::W4ScaleSearch::kMse ? "mse" : "absmax", rtn, gptq);
    EXPECT_LT(gptq, 0.8 * rtn);
  }
  for (bool gptq : {false, true}) {
    mq::ConvRotW4A4Options o;
    o.search = mq::W4ScaleSearch::kMse;
    o.hessian = gptq ? c.hess.data() : nullptr;
    o.threads = 1;
    const mq::ConvRotW4A4Weight a = mq::quantize_convrot_w4a4(c.w.data(), c.n, c.k, o);
    o.threads = 8;
    const mq::ConvRotW4A4Weight b = mq::quantize_convrot_w4a4(c.w.data(), c.n, c.k, o);
    EXPECT_EQ(count_diff(a.q, b.q), 0u) << (gptq ? "gptq" : "rtn");
    EXPECT_EQ(count_diff_bits(a.scale, b.scale), 0u) << (gptq ? "gptq" : "rtn");
  }
  const auto h1 = mq::gram_bf16(c.x.data(), c.rows, c.k, c.k, 1), h8 = mq::gram_bf16(c.x.data(), c.rows, c.k, c.k, 8);
  EXPECT_TRUE(h1 == h8) << "gram_bf16 depends on the thread count";
}

// =================================================================================================================
// Performance (prints; no timing assertions). Opt-in: MI_RUN_PERF=1.

#define MI_REQUIRE_PERF()                                                                         \
  do {                                                                                            \
    const char* run_perf = std::getenv("MI_RUN_PERF");                                            \
    if (run_perf == nullptr || std::string(run_perf) != "1") GTEST_SKIP() << "set MI_RUN_PERF=1"; \
  } while (0)

int calls_for(double window_ms, double one_ms, int min_calls) {
  return std::max(min_calls, static_cast<int>(std::min(1e6, window_ms / std::max(one_ms, 1e-3))));
}

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

__global__ void clock_probe_kernel(long long* out) {
  unsigned long long t0 = 0, t1 = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  const long long c0 = clock64();
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  } while (t1 - t0 < 200000);
  out[0] = clock64() - c0;
  out[1] = static_cast<long long>(t1 - t0);
}

double sm_mhz() {
  Buf<long long> b(2);
  clock_probe_kernel<<<1, 1>>>(b.get());
  MI_CUDA_CHECK_LAUNCH();
  const auto h = b.download();
  return 1e3 * static_cast<double>(h[0]) / static_cast<double>(h[1]);
}

struct Rate {
  double ms = 0, mhz = 0;
};

// Steady-state rate per op (window MI_QUANT_PERF_WINDOW_MS, default 300 ms, after half a window of warm-up), ops
// round-robin over `rounds`.
std::vector<Rate> steady(Timer& t, const std::vector<std::function<void()>>& fns, int rounds = 2) {
  const char* env = std::getenv("MI_QUANT_PERF_WINDOW_MS");
  const double window = env != nullptr ? std::atof(env) : 300.0;
  std::vector<Rate> r(fns.size());
  for (int round = 0; round < rounds; ++round) {
    for (std::size_t i = 0; i < fns.size(); ++i) {
      const double one = t.ms(fns[i], 1);
      t.ms(fns[i], calls_for(window / 2, one, 1));
      r[i].ms += t.ms(fns[i], calls_for(window, one, 3));
      r[i].mhz += sm_mhz();
    }
  }
  for (Rate& v : r) {
    v.ms /= rounds;
    v.mhz /= rounds;
  }
  return r;
}

double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

constexpr std::int64_t kM1024 = 8484;  // 1024^2 CFG batch: 2 x 4096 image tokens + 282 + 10 text tokens

// Per DiT site at M = 8484: bf16 (cuBLASLt, and the engine's fused CUTLASS path), INT8 ConvRot (kernels/quant,
// kAuto), INT4 ConvRot (every tile), back to back in one GPU hold; the activation
// quantization sites; the projected 1024^2 step for each tier.
TEST(GpuQuant4Kernels, Throughput) {
  MI_REQUIRE_PERF();
  Timer timer;
  mi::kernels::Gemm gemm;
  mi::DeviceBuffer gws(mi::kernels::Gemm::kWorkspaceBytes);
  const std::int64_t m = kM1024, kmax = 8192, nmax = 16384;
  Buf<std::int8_t> x8(static_cast<std::size_t>(m * kmax)), w8(static_cast<std::size_t>(nmax * kmax));
  fill(x8.get(), x8.n, 1, 0);
  fill(w8.get(), w8.n, 2, 0);
  Buf<std::uint8_t> x4 = dev_nibbles(3, static_cast<std::size_t>(m * kmax / 2), false),
                    w4 = dev_nibbles(4, static_cast<std::size_t>(nmax * kmax / 2), false);
  Buf<float> xs = dev_scale(5, static_cast<std::size_t>(m), 0.002f, 0.2f), wsc = dev_scale(6, static_cast<std::size_t>(nmax), 1e-5f, 1e-3f);
  Buf<std::uint16_t> xb = dev_act(11, m, kmax), wb = dev_act(12, nmax, kmax), res = dev_act(13, m, 3072),
                     gate = dev_vec(14, 3072, 0.0f, 1.0f), out(static_cast<std::size_t>(m * nmax));
  struct Site {
    const char* name;
    std::int64_t n, k;
    int kind;  // 0 plain (qkv), 1 gated residual (o, down), 2 SiTU-GLU (up_gate)
  };
  const Site sites[] = {{"qkv", 9216, 3072, 0}, {"o", 3072, 3072, 1}, {"up_gate", 16384, 3072, 2}, {"down", 3072, 8192, 1}};
  double sum_bf16 = 0, sum_int8 = 0, sum_int4 = 0;
  for (const double t0 = wall_seconds(); wall_seconds() - t0 < 2.0;) {  // into the power-capped steady state
    timer.ms([&] { gemm.linear(view_bf16(out.get(), m, 3072), view_bf16(xb.get(), m, 8192), view_bf16(wb.get(), 3072, 8192), nullptr, gws.get(), gws.size(), nullptr); }, 5);
  }
  for (const Site& s : sites) {
    mi::kernels::quant::Int8Operands o8;
    o8.x_q = x8.get();
    o8.ldx = s.k;
    o8.x_scale = xs.get();
    o8.w = w8.get();
    o8.w_scale = wsc.get();
    o8.m = m;
    o8.n = s.n;
    o8.k = s.k;
    q4::Int4Operands o4;
    o4.x_q = x4.get();
    o4.ldx = s.k / 2;
    o4.x_scale = xs.get();
    o4.w = w4.get();
    o4.w_scale = wsc.get();
    o4.m = m;
    o4.n = s.n;
    o4.k = s.k;
    const Tensor x_t = view_bf16(xb.get(), m, s.k), w_t = view_bf16(wb.get(), s.n, s.k);
    const Tensor out_t = view_bf16(out.get(), m, s.kind == 2 ? s.n / 2 : s.n), res_t = view_bf16(res.get(), m, 3072);
    const Tensor big_t = view_bf16(out.get(), m, s.n), gate_t = Tensor::view(gate.get(), DType::kBF16, mi::Device::kCUDA, {std::int64_t{3072}});
    const std::function<void()> bf16_cublas = [&] { gemm.linear(big_t, x_t, w_t, nullptr, gws.get(), gws.size(), nullptr); };
    const std::function<void()> bf16_engine = [&] {
      if (s.kind == 0) gemm.linear(big_t, x_t, w_t, nullptr, gws.get(), gws.size(), nullptr);
      if (s.kind == 1) mi::kernels::fused::gemm_gated_residual(res_t, x_t, w_t, gate_t, nullptr);
      if (s.kind == 2) mi::kernels::fused::gemm_situ_glu(out_t, x_t, w_t, nullptr);
    };
    const std::function<void()> int8_run = [&] {
      namespace q8 = mi::kernels::quant;
      if (s.kind == 0) q8::int8_gemm(o8, nullptr, as_bf16(out), s.n, nullptr);
      if (s.kind == 1) q8::int8_gemm_gated_residual(o8, nullptr, as_bf16(gate), as_bf16(res), 3072, nullptr);
      if (s.kind == 2) q8::int8_gemm_situ_glu(o8, as_bf16(out), s.n / 2, nullptr);
    };
    const auto int4_run = [&](Int4Tile tile) {
      return std::function<void()>([&, tile] {
        if (s.kind == 0) q4::int4_gemm(o4, nullptr, as_bf16(out), s.n, nullptr, tile);
        if (s.kind == 1) q4::int4_gemm_gated_residual(o4, nullptr, as_bf16(gate), as_bf16(res), 3072, nullptr, tile);
        if (s.kind == 2) q4::int4_gemm_situ_glu(o4, as_bf16(out), s.n / 2, nullptr, tile);
      });
    };
    std::vector<std::function<void()>> fns = {bf16_cublas, bf16_engine, int8_run};
    for (Int4Tile tile : kTiles4) fns.push_back(int4_run(tile));
    const std::vector<Rate> r = steady(timer, fns);
    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(s.n) * static_cast<double>(s.k);
    const auto line = [&](const std::string& what, const Rate& v) {
      std::printf("[quant4 perf]   %-22s %6.3f ms %6.1f T(FL)OPS @ %4.0f MHz  %5.2fx vs bf16 engine\n", what.c_str(),
                  v.ms, ops / v.ms / 1e9, v.mhz, r[1].ms / v.ms);
    };
    std::printf("[quant4 perf] %s M %lld N %lld K %lld\n", s.name, static_cast<long long>(m), static_cast<long long>(s.n),
                static_cast<long long>(s.k));
    line("bf16 cuBLASLt", r[0]);
    line("bf16 engine", r[1]);
    line("int8 auto", r[2]);
    std::size_t i = 3;
    for (Int4Tile tile : kTiles4) line(std::string("int4 ") + q4::int4_tile_name(tile), r[i++]);
    sum_bf16 += r[1].ms;
    sum_int8 += r[2].ms;
    sum_int4 += r[3].ms;
  }
  // Activation sites per block. bf16: 2 rms_norm_modulate_moss. INT8 / INT4 ConvRot: 2 fused norm+rotate+quantize, 2
  // standalone rotate+quantize (K 3072 and 8192).
  Buf<std::int8_t> q8o(static_cast<std::size_t>(m * kmax));
  Buf<std::uint8_t> q4o(static_cast<std::size_t>(m * kmax / 2));
  Buf<float> so(static_cast<std::size_t>(m));
  Buf<std::uint16_t> nw = dev_vec(17, 3072, 1.0f, 0.5f), msc = dev_vec(18, 3072, 0.0f, 1.0f), msh = dev_vec(19, 3072, 0.0f, 1.0f);
  const auto v3072 = [](const Buf<std::uint16_t>& b) {
    return Tensor::view(b.get(), DType::kBF16, mi::Device::kCUDA, {std::int64_t{3072}});
  };
  namespace q8 = mi::kernels::quant;
  const std::vector<std::function<void()>> act_fns = {
      [&] { mi::kernels::rms_norm_modulate_moss(view_bf16(xb.get(), m, 3072), v3072(nw), v3072(msc), v3072(msh), view_bf16(out.get(), m, 3072), nullptr); },
      [&] { q8::rms_norm_modulate_rotate_quantize(as_bf16(xb), 3072, as_bf16(nw), as_bf16(msc), as_bf16(msh), q8o.get(), 3072, so.get(), m, 3072, nullptr); },
      [&] { q8::rotate_quantize(as_bf16(xb), 3072, q8o.get(), 3072, so.get(), m, 3072, nullptr); },
      [&] { q8::rotate_quantize(as_bf16(xb), 8192, q8o.get(), 8192, so.get(), m, 8192, nullptr); },
      [&] { q4::rms_norm_modulate_convrot_quantize_int4(as_bf16(xb), 3072, as_bf16(nw), as_bf16(msc), as_bf16(msh), q4o.get(), 1536, so.get(), m, 3072, nullptr); },
      [&] { q4::convrot_quantize_int4(as_bf16(xb), 3072, q4o.get(), 1536, so.get(), m, 3072, nullptr); },
      [&] { q4::convrot_quantize_int4(as_bf16(xb), 8192, q4o.get(), 4096, so.get(), m, 8192, nullptr); },
  };
  std::vector<double> a;
  for (const Rate& v : steady(timer, act_fns)) a.push_back(v.ms);
  const char* names[] = {"rms_norm_modulate_moss", "int8 norm+rotate+quant", "int8 rotate K3072", "int8 rotate K8192",
                         "int4 norm+rotate+quant", "int4 rotate K3072", "int4 rotate K8192"};
  for (std::size_t i = 0; i < a.size(); ++i) std::printf("[quant4 perf] %-24s %6.3f ms\n", names[i], a[i]);
  const double act_bf16 = 2 * a[0], act_int8 = 2 * a[1] + a[2] + a[3], act_int4 = 2 * a[4] + a[5] + a[6];
  const char* step_env = std::getenv("MI_QUANT_BF16_STEP_MS");
  const double bf16_step = step_env != nullptr ? std::atof(step_env) : 1380.0;
  const double blk_bf16 = sum_bf16 + act_bf16;
  const auto project = [&](const char* tier, double gemms, double acts) {
    const double blk = gemms + acts, step = bf16_step - 42 * (blk_bf16 - blk);
    std::printf("[quant4 perf] %-14s block GEMMs %.3f + activation sites %.3f = %.3f ms (bf16 %.3f); projected 1024^2 "
                "step %.0f ms vs bf16 %.0f ms (%.2fx)\n", tier, gemms, acts, blk, blk_bf16, step, bf16_step,
                bf16_step / step);
    RecordProperty(std::string("projected_step_ms ") + tier, std::to_string(step));
  };
  project("int8 convrot", sum_int8, act_int8);
  project("int4 convrot", sum_int4, act_int4);
}

// Sustained rates in one GPU hold: a long loop of each precision at the ffn_down shape (8484 x 3072 x 8192), SM clock
// probed after each ~0.25 s window. Power comes from an external sampler (nvidia-smi); the printed wall-clock stamps
// delimit the phases.
TEST(GpuQuant4Kernels, SustainedClock) {
  MI_REQUIRE_PERF();
  const char* env = std::getenv("MI_QUANT_SUSTAINED_SECONDS");
  const double seconds = env != nullptr ? std::atof(env) : 4.0;
  const std::int64_t m = kM1024, n = 3072, k = 8192;
  Int4Data d4(m, n, k, 7);
  Buf<std::int8_t> x8(static_cast<std::size_t>(m * k)), w8(static_cast<std::size_t>(n * k));
  fill(x8.get(), x8.n, 1, 0);
  fill(w8.get(), w8.n, 2, 0);
  Buf<std::uint16_t> xb = dev_act(8, m, k), wb = dev_act(9, n, k), out(static_cast<std::size_t>(m * n));
  mi::kernels::Gemm gemm;
  mi::DeviceBuffer gws(mi::kernels::Gemm::kWorkspaceBytes);
  const Tensor x_t = view_bf16(xb.get(), m, k), w_t = view_bf16(wb.get(), n, k), o_t = view_bf16(out.get(), m, n);
  mi::kernels::quant::Int8Operands o8;
  o8.x_q = x8.get();
  o8.ldx = k;
  o8.x_scale = d4.xs.get();
  o8.w = w8.get();
  o8.w_scale = d4.ws.get();
  o8.m = m;
  o8.n = n;
  o8.k = k;
  const double ops = 2.0 * static_cast<double>(m * n) * static_cast<double>(k);
  Timer timer;
  const auto loop = [&](const char* name, const std::function<void()>& fn) {
    const int iters = calls_for(250.0, timer.ms(fn, 1), 1);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    const double t0 = wall_seconds();
    std::printf("[quant4 sustained] %s start %.3f\n", name, t0);
    double total_ops = 0, total_ms = 0, mhz_sum = 0, last_tops = 0, last_mhz = 0;
    int windows = 0;
    while (wall_seconds() - t0 < seconds) {
      const double ms = timer.ms(fn, iters);
      last_mhz = sm_mhz();
      last_tops = ops / ms / 1e9;
      mhz_sum += last_mhz;
      total_ops += iters * ops;
      total_ms += iters * ms;
      ++windows;
    }
    const double mean = total_ops / total_ms / 1e9;
    std::printf("[quant4 sustained] %s end %.3f: mean %.1f TOPS @ %.0f MHz over %.1f s (last window %.1f @ %.0f MHz)\n",
                name, wall_seconds(), mean, mhz_sum / windows, total_ms / 1e3, last_tops, last_mhz);
    return mean;
  };
  const double bf16 = loop("bf16", [&] { gemm.linear(o_t, x_t, w_t, nullptr, gws.get(), gws.size(), nullptr); });
  const double int8 = loop("int8", [&] { mi::kernels::quant::int8_gemm(o8, nullptr, as_bf16(out), n, nullptr); });
  const double int4 = loop("int4", [&] { q4::int4_gemm(d4.ops(), nullptr, as_bf16(out), n, nullptr); });
  const double bf16b = loop("bf16 again", [&] { gemm.linear(o_t, x_t, w_t, nullptr, gws.get(), gws.size(), nullptr); });
  std::printf("[quant4 sustained] bf16 %.1f / %.1f TFLOPS, int8 %.1f TOPS (%.2fx), int4 %.1f TOPS (%.2fx)\n", bf16,
              bf16b, int8, int8 / bf16, int4, int4 / bf16);
  RecordProperty("sustained_int4_vs_bf16", std::to_string(int4 / bf16));
}

// =================================================================================================================
// Real weights: the quantization error at the DiT shapes, on real activations.
//
// Activations: the inputs of the q/k/v, o, ffn_up_gate and ffn_down linears of DiT blocks 1, 21 and 41 at step 0 of
// the 1024^2 g_fridge golden case (4096 image + 62 text tokens), captured from the reference transformer (official
// SFT weights) into tests/golden/quant4_acts/ (git-ignored). Even rows calibrate (the GPTQ Hessian), odd rows are
// held out and evaluated. Weights: the official bf16 SFT transformer (MOSS_WEIGHTS, default /workspace/Models/
// MOSS-Image) quantized by weights/quant/convrot_w4a4, and the community convrot_w4a4 file (MOSS_COMMUNITY, default
// /workspace/Models/community). Error = cos and relative L2 of the W4A4 output against the bf16 GEMM output
// (cuBLASLt, same activations). MI_QUANT4_BLOCKS picks blocks (default "1,21,41"), MI_QUANT4_SITES sites (e.g.
// "qkv,down"), MI_QUANT4_GPTQ adds the GPTQ variant.

const std::filesystem::path kActDir = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/golden/quant4_acts";

std::filesystem::path env_path(const char* name, const char* fallback) {
  const char* env = std::getenv(name);
  return env != nullptr && *env != '\0' ? std::filesystem::path(env) : std::filesystem::path(fallback);
}

class TensorSource {
 public:
  explicit TensorSource(const std::vector<std::filesystem::path>& paths) {
    for (const auto& p : paths) files_.push_back(std::make_unique<mi::weights::SafetensorsFile>(p));
  }
  template <class T>
  std::vector<T> read(const std::string& name) const {
    for (const auto& f : files_) {
      const mi::weights::TensorInfo* t = f->find(name);
      if (t == nullptr) continue;
      std::vector<T> out(t->nbytes / sizeof(T));
      std::ifstream in(f->path(), std::ios::binary);
      in.seekg(static_cast<std::streamoff>(t->offset));
      in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(t->nbytes));
      if (!in) throw std::runtime_error("short read of " + name);
      return out;
    }
    throw std::runtime_error("tensor not found: " + name);
  }

 private:
  std::vector<std::unique_ptr<mi::weights::SafetensorsFile>> files_;
};

template <class T>
void append(std::vector<T>& a, const std::vector<T>& b) {
  a.insert(a.end(), b.begin(), b.end());
}

struct ErrStat {
  double cos = 0, rel_l2 = 0;
};

// got vs ref over the odd rows of [m, n] bf16 matrices on the device.
ErrStat odd_row_error(const Buf<std::uint16_t>& got, const Buf<std::uint16_t>& ref, std::int64_t m, std::int64_t n) {
  const auto g = got.download(), r = ref.download();
  double dot = 0, ng = 0, nr = 0, nd = 0;
  for (std::int64_t row = 1; row < m; row += 2) {
    for (std::int64_t c = 0; c < n; ++c) {
      const double a = bf16_float(g[static_cast<std::size_t>(row * n + c)]), b = bf16_float(r[static_cast<std::size_t>(row * n + c)]);
      dot += a * b;
      ng += a * a;
      nr += b * b;
      nd += (a - b) * (a - b);
    }
  }
  return {dot / std::sqrt(ng * nr), std::sqrt(nd / nr)};
}

TEST(GpuQuant4Kernels, RealWeightQuantError) {
  const std::filesystem::path kSftDir = env_path("MOSS_WEIGHTS", "/workspace/Models/MOSS-Image") / "sft/transformer";
  const std::filesystem::path kInt4File =
      env_path("MOSS_COMMUNITY", "/workspace/Models/community") / "toxicdog/moss_image_sft_int4_convrot.safetensors";
  if (!std::filesystem::exists(kActDir / "block01_qkv.npy") || !std::filesystem::exists(kSftDir)) {
    GTEST_SKIP() << "captured activations or official weights absent";
  }
  const bool have_int4 = std::filesystem::exists(kInt4File);
  // MI_QUANT4_GPTQ=1: GPTQ at K <= 4096 (minutes of CPU); =all: also the K = 8192 ffn_down (a K^3 factorization).
  const std::string gptq_env = std::getenv("MI_QUANT4_GPTQ") != nullptr ? std::getenv("MI_QUANT4_GPTQ") : "";
  std::vector<int> blocks = {1, 21, 41};
  if (const char* env = std::getenv("MI_QUANT4_BLOCKS")) {
    blocks.clear();
    std::stringstream ss(env);
    for (std::string tok; std::getline(ss, tok, ',');) blocks.push_back(std::stoi(tok));
  }
  std::vector<std::filesystem::path> shards;
  for (const auto& e : std::filesystem::directory_iterator(kSftDir)) {
    if (e.path().extension() == ".safetensors") shards.push_back(e.path());
  }
  std::sort(shards.begin(), shards.end());
  const TensorSource sft(shards);
  std::unique_ptr<TensorSource> community = have_int4 ? std::make_unique<TensorSource>(std::vector<std::filesystem::path>{kInt4File}) : nullptr;
  mi::kernels::Gemm gemm;
  mi::DeviceBuffer gws(mi::kernels::Gemm::kWorkspaceBytes);
  struct Site {
    const char* tag;
    std::vector<const char*> parts;
  };
  const Site sites[] = {{"qkv", {"q", "k", "v"}}, {"o", {"o"}}, {"up", {"ffn_up_gate"}}, {"down", {"ffn_down"}}};
  std::printf("[quant4 real] %-8s %-5s %-34s %12s %10s\n", "block", "site", "variant", "cos", "rel L2");
  for (int b : blocks) {
    char bname[16];
    std::snprintf(bname, sizeof bname, "block%02d", b);
    for (const Site& site : sites) {
      if (const char* only = std::getenv("MI_QUANT4_SITES"); only != nullptr) {
        std::stringstream ss(only);
        bool selected = false;
        for (std::string tok; std::getline(ss, tok, ',');) selected = selected || tok == site.tag;
        if (!selected) continue;
      }
      const auto xa = mi::npy::load(kActDir / (std::string(bname) + "_" + site.tag + ".npy"));
      const std::int64_t m = xa.shape[0], k = xa.shape[1];
      const auto x = xa.to_vector<std::uint16_t>();
      std::vector<std::uint16_t> w;
      for (const char* part : site.parts) {
        append(w, sft.read<std::uint16_t>("layers." + std::to_string(b) + ".proj." + part + ".weight"));
      }
      const std::int64_t n = static_cast<std::int64_t>(w.size()) / k;
      Buf<std::uint16_t> xd(x), wd(w), yref(static_cast<std::size_t>(m * n)), y(static_cast<std::size_t>(m * n));
      gemm.linear(view_bf16(yref.get(), m, n), view_bf16(xd.get(), m, k), view_bf16(wd.get(), n, k), nullptr, gws.get(),
                  gws.size(), nullptr);
      const auto report = [&](const std::string& variant) {
        MI_CUDA_CHECK(cudaDeviceSynchronize());
        const ErrStat e = odd_row_error(y, yref, m, n);
        std::printf("[quant4 real] %-8s %-5s %-34s %12.8f %10.5f\n", bname, site.tag, variant.c_str(), e.cos, e.rel_l2);
        RecordProperty(std::string(bname) + "_" + site.tag + "_" + variant + "_cos", std::to_string(e.cos));
        return e;
      };

      // High tier reference: INT8 ConvRot (kernels/quant) on the same weights.
      {
        std::vector<std::int8_t> q8(static_cast<std::size_t>(n * k));
        std::vector<float> s8(static_cast<std::size_t>(n));
        ASSERT_TRUE(mi::quant::quantize_convrot_rows_bf16(w.data(), n, k, 256, q8.data(), s8.data()));
        Buf<std::int8_t> wq(q8), xq(static_cast<std::size_t>(m * k));
        Buf<float> ws(s8), xs(static_cast<std::size_t>(m));
        mi::kernels::quant::rotate_quantize(as_bf16(xd), k, xq.get(), k, xs.get(), m, static_cast<int>(k), nullptr);
        mi::kernels::quant::Int8Operands o;
        o.x_q = xq.get();
        o.ldx = k;
        o.x_scale = xs.get();
        o.w = wq.get();
        o.w_scale = ws.get();
        o.m = m;
        o.n = n;
        o.k = k;
        mi::kernels::quant::int8_gemm(o, nullptr, as_bf16(y), n, nullptr);
        const ErrStat e = report("int8 convrot (High tier)");
        EXPECT_GT(e.cos, 0.999);
      }
      // INT4 ConvRot.
      Buf<std::uint8_t> xq4(static_cast<std::size_t>(m * k / 2));
      Buf<float> xs4(static_cast<std::size_t>(m));
      q4::convrot_quantize_int4(as_bf16(xd), k, xq4.get(), k / 2, xs4.get(), m, static_cast<int>(k), nullptr);
      const auto run_int4 = [&](const std::vector<std::uint8_t>& q, const std::vector<float>& sc, const std::string& v) {
        Buf<std::uint8_t> wq(q);
        Buf<float> ws(sc);
        q4::Int4Operands o;
        o.x_q = xq4.get();
        o.ldx = k / 2;
        o.x_scale = xs4.get();
        o.w = wq.get();
        o.w_scale = ws.get();
        o.m = m;
        o.n = n;
        o.k = k;
        q4::int4_gemm(o, nullptr, as_bf16(y), n, nullptr);
        const ErrStat e = report(v);
        EXPECT_GT(e.cos, 0.95) << v;
      };
      mi::quant::ConvRotW4A4Options opt;
      const mi::quant::ConvRotW4A4Weight rtn = mi::quant::quantize_convrot_w4a4(w.data(), n, k, opt);
      run_int4(rtn.q, rtn.scale, "w4a4, ours (absmax)");
      opt.search = mi::quant::W4ScaleSearch::kMse;
      const mi::quant::ConvRotW4A4Weight mse = mi::quant::quantize_convrot_w4a4(w.data(), n, k, opt);
      run_int4(mse.q, mse.scale, "w4a4, ours (mse scale)");
      if (community) {
        std::vector<std::uint8_t> q;
        std::vector<float> sc;
        for (const char* part : site.parts) {
          const std::string base = "layers." + std::to_string(b) + ".proj." + part;
          append(q, community->read<std::uint8_t>(base + ".weight"));
          append(sc, community->read<float>(base + ".weight_scale"));
        }
        run_int4(q, sc, "w4a4, community file");
      }
      if (gptq_env == "all" || (gptq_env == "1" && k <= 4096)) {
        std::vector<std::uint16_t> even;
        for (std::int64_t r = 0; r < m; r += 2) even.insert(even.end(), x.begin() + r * k, x.begin() + (r + 1) * k);
        const std::vector<double> hess = mi::quant::gram_bf16(even.data(), static_cast<std::int64_t>(even.size()) / k, k, k);
        opt.hessian = hess.data();
        const mi::quant::ConvRotW4A4Weight g = mi::quant::quantize_convrot_w4a4(w.data(), n, k, opt);
        run_int4(g.q, g.scale, "w4a4, ours (mse scale + gptq)");
      }
    }
  }
}

}  // namespace
