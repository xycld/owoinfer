// kernels/quant (INT8 ConvRot, AC19): rotate + quantize bit for bit against comfy-kitchen @dba9fac's math (tests/data/
// quant_kernels: torch on CUDA, its eager quantize_int8_rowwise after an fp32 emulation of its CUDA rotation kernel,
// _int_mm accumulators, the reference module's _RMSNorm / _situ) and against the engine's op-by-op chain
// (rms_norm_modulate_moss, then rotate_quantize); the INT8 GEMM epilogues against torch's int8_linear vectors and,
// at the DiT / text-encoder shapes and edge row counts, bit for bit against a slow scalar oracle kernel (exact int32
// dot product + the documented fp32 epilogue order; SiTU through kernels::situ_glu); rotate + quantize also against a
// host scalar reference (zero / NaN / inf rows, every block shape). Plus throughput and a sustained-clock run
// (MI_QUANT_SUSTAINED_SECONDS, default 2 s per precision), skipped unless MI_RUN_PERF=1.
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
#include <functional>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/quant/quant.h"
#include "util/npy.h"

namespace {

namespace quant = mi::kernels::quant;
using mi::DType;
using mi::Tensor;
using quant::Int8Tile;
using bf16_t = __nv_bfloat16;

const std::filesystem::path kData = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/data/quant_kernels";
mi::npy::Array load(const std::string& name) { return mi::npy::load(kData / (name + ".npy")); }

// Typed device buffer.
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

bf16_t* as_bf16(Buf<std::uint16_t>& b) { return reinterpret_cast<bf16_t*>(b.get()); }
const bf16_t* as_bf16(const Buf<std::uint16_t>& b) { return reinterpret_cast<const bf16_t*>(b.get()); }

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

// DiT-like bf16 activations [rows, cols]: hashed +-m * 2^(c_exp - 7 - e) with m in [128, 255], e in 0..7 and a
// per-channel exponent c_exp in -3..2, plus outlier channels (every 397th) 64x larger.
std::vector<std::uint16_t> synth_act(std::uint64_t seed, std::int64_t rows, std::int64_t cols) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows * cols));
  for (std::int64_t c = 0; c < cols; ++c) {
    const int c_exp = static_cast<int>(mix(static_cast<std::uint64_t>(c) * 7919 + seed) % 6) - 3 + (c % 397 == 5 ? 6 : 0);
    for (std::int64_t r = 0; r < rows; ++r) {
      const std::uint64_t z = mix(static_cast<std::uint64_t>(r * cols + c) + seed * 0x9E3779B97F4A7C15ull);
      const float m = static_cast<float>(128 + (z & 0x7F));
      const int e = static_cast<int>((z >> 7) & 7);
      const float v = std::ldexp((z >> 10) & 1 ? -m : m, c_exp - 7 - e);
      out[static_cast<std::size_t>(r * cols + c)] = bf16_bits(v);
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

template <class T>
std::size_t count_diff(const std::vector<T>& a, const std::vector<T>& b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += !(a[i] == b[i]);
  return n;
}

std::size_t count_diff_bits(const std::vector<float>& a, const std::vector<float>& b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
  return n;
}

// bf16 distance in units in the last place (ordered integer distance).
int ulp_dist(std::uint16_t a, std::uint16_t b) {
  const auto ord = [](std::uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7FFF) : static_cast<int>(v); };
  return std::abs(ord(a) - ord(b));
}

struct Agreement {
  double cosine = 0, identical = 0, within1 = 0;
  int max_ulp = 0;
};

Agreement agreement(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& ref) {
  Agreement a;
  double dot = 0, ng = 0, nr = 0;
  std::size_t same = 0, w1 = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double g = bf16_float(got[i]), r = bf16_float(ref[i]);
    dot += g * r;
    ng += g * g;
    nr += r * r;
    const int d = ulp_dist(got[i], ref[i]);
    same += d == 0;
    w1 += d <= 1;
    a.max_ulp = std::max(a.max_ulp, d);
  }
  a.cosine = dot / std::sqrt(ng * nr);
  a.identical = static_cast<double>(same) / static_cast<double>(got.size());
  a.within1 = static_cast<double>(w1) / static_cast<double>(got.size());
  return a;
}

// Device-side fills for the large operands (same hash family as the host helpers, not the same values).
__device__ std::uint64_t mix_d(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// kind 0: int8 bytes; 1: f32 log-uniform in [a, a * 2^b]; 2: bf16 uniform in [a - b, a + b]; 3: bf16 activations as
// synth_act ([., cols] with per-channel exponents).
__global__ void fill_kernel(void* p, std::size_t n, std::uint64_t seed, int kind, float a, float b, int cols) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::uint64_t z = mix_d(i + seed * 0x9E3779B97F4A7C15ull);
    const float u = static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
    if (kind == 0) {
      static_cast<std::int8_t*>(p)[i] = static_cast<std::int8_t>(z & 0xFF);
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

Buf<std::int8_t> dev_i8(std::uint64_t seed, std::size_t n) {
  Buf<std::int8_t> b(n);
  fill(b.get(), n, seed, 0);
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

// ---------------------------------------------------------------------------------------------------------------
// Scalar oracle: one thread per (sampled row, column), exact int32 dot product, then the documented epilogue:
// y = bf16(((float)acc * xs) * ws + bias); mode 1: residual + bf16(y * gate); mode 0/2: y (2 = no bias, for SiTU).
__global__ void oracle_kernel(const std::int8_t* x, std::int64_t ldx, const std::int8_t* w, const float* xs,
                              const float* ws, const bf16_t* bias, const bf16_t* gate, const bf16_t* residual,
                              std::int64_t ldr, const int* rows, int nrows, int n, int k, int mode, bf16_t* out,
                              std::int32_t* acc_out) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(nrows) * n) return;
  const int i = static_cast<int>(idx / n), c = static_cast<int>(idx % n);
  const int r = rows[i];
  const std::int8_t* xr = x + r * ldx;
  const std::int8_t* wr = w + static_cast<std::int64_t>(c) * k;
  std::int32_t acc = 0;
  for (int j = 0; j < k; ++j) acc += static_cast<std::int32_t>(xr[j]) * static_cast<std::int32_t>(wr[j]);
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

// Operands of one GEMM problem on the device (hashed int8, realistic scales).
struct GemmData {
  std::int64_t m, n, k, ldx;  // x: [m, k] with row stride ldx (the padding holds random bytes too)
  Buf<std::int8_t> x, w;
  Buf<float> xs, ws;
  Buf<std::uint16_t> bias, gate;
  GemmData(std::int64_t m_, std::int64_t n_, std::int64_t k_, std::uint64_t seed, std::int64_t ldx_ = 0)
      : m(m_), n(n_), k(k_), ldx(ldx_ > 0 ? ldx_ : k_),
        x(dev_i8(seed, static_cast<std::size_t>(m_ * ldx))),
        w(dev_i8(seed + 1, static_cast<std::size_t>(n_ * k_))),
        xs(dev_scale(seed + 2, static_cast<std::size_t>(m_), 0.002f, 0.2f)),
        ws(dev_scale(seed + 3, static_cast<std::size_t>(n_), 1e-5f, 1e-3f)),
        bias(dev_vec(seed + 4, static_cast<std::size_t>(n_), 0.0f, 2.0f)),
        gate(dev_vec(seed + 5, static_cast<std::size_t>(n_), 0.0f, 1.5f)) {}
  quant::Int8Operands ops() const {
    quant::Int8Operands o;
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

// Oracle outputs [rows.size(), n] (mode 0/1/2 as oracle_kernel); residual row stride ldr (0: n).
std::vector<std::uint16_t> oracle(const GemmData& d, const std::vector<int>& rows, int mode, const bf16_t* bias,
                                  const bf16_t* residual, std::vector<std::int32_t>* acc = nullptr,
                                  std::int64_t ldr = 0) {
  Buf<int> r(rows);
  Buf<std::uint16_t> out(rows.size() * static_cast<std::size_t>(d.n));
  Buf<std::int32_t> a(acc != nullptr ? out.n : 1);
  const std::int64_t total = static_cast<std::int64_t>(out.n);
  oracle_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(
      d.x.get(), d.ldx, d.w.get(), d.xs.get(), d.ws.get(), bias, as_bf16(d.gate), residual, ldr > 0 ? ldr : d.n, r.get(),
      static_cast<int>(rows.size()), static_cast<int>(d.n), static_cast<int>(d.k), mode, as_bf16(out),
      acc != nullptr ? a.get() : nullptr);
  MI_CUDA_CHECK_LAUNCH();
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  if (acc != nullptr) *acc = a.download();
  return out.download();
}

// The SiTU-GLU of oracle y rows [rows, 2F] through the engine's exact kernels::situ_glu.
std::vector<std::uint16_t> oracle_situ(const GemmData& d, const std::vector<int>& rows) {
  Buf<std::uint16_t> y(oracle(d, rows, 2, nullptr, nullptr));
  const auto nr = static_cast<std::int64_t>(rows.size());
  Buf<std::uint16_t> out(static_cast<std::size_t>(nr * d.n / 2));
  mi::kernels::situ_glu(view_bf16(y.get(), nr, d.n), view_bf16(out.get(), nr, d.n / 2), nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return out.download();
}

// Rows `rows` of a device bf16 matrix with row stride ld, cols columns.
std::vector<std::uint16_t> download_rows(const std::uint16_t* p, std::int64_t ld, std::int64_t cols,
                                         const std::vector<int>& rows) {
  std::vector<std::uint16_t> h(rows.size() * static_cast<std::size_t>(cols));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    MI_CUDA_CHECK(cudaMemcpy(h.data() + i * static_cast<std::size_t>(cols), p + rows[i] * ld,
                             static_cast<std::size_t>(cols) * 2, cudaMemcpyDeviceToHost));
  }
  return h;
}

std::vector<int> all_rows(std::int64_t m) {
  std::vector<int> r(static_cast<std::size_t>(m));
  for (std::int64_t i = 0; i < m; ++i) r[static_cast<std::size_t>(i)] = static_cast<int>(i);
  return r;
}

// Rows at and around every tile boundary near both ends and the middle.
std::vector<int> sample_rows(std::int64_t m) {
  std::vector<int> r;
  for (std::int64_t v : {std::int64_t{0}, std::int64_t{1}, std::int64_t{63}, std::int64_t{64}, std::int64_t{127},
                         std::int64_t{128}, std::int64_t{129}, std::int64_t{255}, std::int64_t{256}, m / 2, m - 257,
                         m - 129, m - 128, m - 2, m - 1}) {
    if (v >= 0 && v < m && std::find(r.begin(), r.end(), static_cast<int>(v)) == r.end()) r.push_back(static_cast<int>(v));
  }
  return r;
}

constexpr std::initializer_list<Int8Tile> kTiles = {Int8Tile::kAuto,      Int8Tile::k128x128x64, Int8Tile::k128x256x64,
                           Int8Tile::k256x128x64, Int8Tile::k64x128x64,  Int8Tile::k128x128x128};

// Runs the three epilogues on `d` with each of `tiles` and checks rows `rows` against the oracle, bit for bit.
void check_gemm_against_oracle(const GemmData& d, const std::vector<int>& rows, std::initializer_list<Int8Tile> tiles,
                               bool with_bias, const std::string& label) {
  const std::size_t mn = static_cast<std::size_t>(d.m * d.n);
  Buf<std::uint16_t> out(mn), res(mn), res0 = dev_act(static_cast<std::uint64_t>(d.m * 31 + d.n), d.m, d.n);
  const bf16_t* bias = with_bias ? as_bf16(d.bias) : nullptr;
  const auto ref_plain = oracle(d, rows, 0, bias, nullptr);
  const auto ref_res = oracle(d, rows, 1, bias, as_bf16(res0));
  const auto ref_situ = oracle_situ(d, rows);
  for (Int8Tile tile : tiles) {
    const std::string tag = label + " tile " + quant::int8_tile_name(tile);
    quant::int8_gemm(d.ops(), bias, as_bf16(out), d.n, nullptr, tile);
    EXPECT_EQ(count_diff(download_rows(out.get(), d.n, d.n, rows), ref_plain), 0u) << tag << " int8_gemm";
    MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), mn * 2, cudaMemcpyDeviceToDevice));
    quant::int8_gemm_gated_residual(d.ops(), bias, as_bf16(d.gate), as_bf16(res), d.n, nullptr, tile);
    EXPECT_EQ(count_diff(download_rows(res.get(), d.n, d.n, rows), ref_res), 0u) << tag << " gated_residual";
    quant::int8_gemm_situ_glu(d.ops(), as_bf16(out), d.n / 2, nullptr, tile);
    EXPECT_EQ(count_diff(download_rows(out.get(), d.n / 2, d.n / 2, rows), ref_situ), 0u) << tag << " situ_glu";
  }
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------------------------------------------
// Rotate + quantize.

struct RotOut {
  std::vector<std::int8_t> q;
  std::vector<float> s;
};

RotOut run_rotate(const std::vector<std::uint16_t>& x, std::int64_t m, int k) {
  Buf<std::uint16_t> xd(x);
  Buf<std::int8_t> q(static_cast<std::size_t>(m * k));
  Buf<float> s(static_cast<std::size_t>(m));
  quant::rotate_quantize(as_bf16(xd), k, q.get(), k, s.get(), m, k, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return {q.download(), s.download()};
}

struct NormInputs {
  Buf<std::uint16_t> weight, scale, shift;
};

RotOut run_fused_norm(const std::vector<std::uint16_t>& x, std::int64_t m, int k, const NormInputs& v) {
  Buf<std::uint16_t> xd(x);
  Buf<std::int8_t> q(static_cast<std::size_t>(m * k));
  Buf<float> s(static_cast<std::size_t>(m));
  quant::rms_norm_modulate_rotate_quantize(as_bf16(xd), k, as_bf16(v.weight), as_bf16(v.scale), as_bf16(v.shift),
                                           q.get(), k, s.get(), m, k, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return {q.download(), s.download()};
}

// The engine's op-by-op chain: kernels::rms_norm_modulate_moss into bf16, then rotate_quantize.
RotOut run_norm_op_by_op(const std::vector<std::uint16_t>& x, std::int64_t m, int k, const NormInputs& v) {
  Buf<std::uint16_t> xd(x), y(x.size());
  const auto vec = [&](const Buf<std::uint16_t>& b) {
    return Tensor::view(b.get(), DType::kBF16, mi::Device::kCUDA, {std::int64_t{k}});
  };
  mi::kernels::rms_norm_modulate_moss(view_bf16(xd.get(), m, k), vec(v.weight), vec(v.scale), vec(v.shift),
                                      view_bf16(y.get(), m, k), nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return run_rotate(y.download(), m, k);
}

TEST(GpuQuantKernels, RotateQuantizeMatchesReference) {
  for (const char* name : {"rot3072", "rot8192"}) {
    const auto xa = load(std::string(name) + "_x");
    const std::int64_t m = xa.shape[0];
    const int k = static_cast<int>(xa.shape[1]);
    const RotOut got = run_rotate(xa.to_vector<std::uint16_t>(), m, k);
    const auto q_ref = load(std::string(name) + "_q").to_vector<std::int8_t>();
    const auto s_ref = load(std::string(name) + "_s").to_vector<float>();
    EXPECT_EQ(count_diff(got.q, q_ref), 0u) << name << " int8 values";
    EXPECT_EQ(count_diff_bits(got.s, s_ref), 0u) << name << " scales";
    const auto minus128 = std::count(got.q.begin(), got.q.end(), std::int8_t{-128});
    std::printf("[quant] %s: %lld x %d bit-exact check, %lld values at -128\n", name, static_cast<long long>(m), k,
                static_cast<long long>(minus128));
  }
}

TEST(GpuQuantKernels, RotateQuantizeStridedRows) {
  const auto xa = load("rot3072_x");
  const std::int64_t m = xa.shape[0], k = xa.shape[1], ldx = k + 64, ldq = k + 32;
  const auto x = xa.to_vector<std::uint16_t>();
  std::vector<std::uint16_t> xp(static_cast<std::size_t>(m * ldx), 0x7FC0);  // NaN padding must not be read
  for (std::int64_t r = 0; r < m; ++r) std::copy_n(x.begin() + r * k, k, xp.begin() + r * ldx);
  Buf<std::uint16_t> xd(xp);
  Buf<std::int8_t> q(static_cast<std::size_t>(m * ldq));
  Buf<float> s(static_cast<std::size_t>(m));
  quant::rotate_quantize(as_bf16(xd), ldx, q.get(), ldq, s.get(), m, static_cast<int>(k), nullptr);
  const auto qh = q.download();
  const auto q_ref = load("rot3072_q").to_vector<std::int8_t>();
  std::vector<std::int8_t> qc(static_cast<std::size_t>(m * k));
  for (std::int64_t r = 0; r < m; ++r) std::copy_n(qh.begin() + r * ldq, k, qc.begin() + r * k);
  EXPECT_EQ(count_diff(qc, q_ref), 0u);
  EXPECT_EQ(count_diff_bits(s.download(), load("rot3072_s").to_vector<float>()), 0u);
}

// Host scalar reference of the activation path in quant.h, one fp32 op per step as the Python emulation of
// comfy-kitchen's CUDA rotation (radix-4 stages, strides 1, 4, 16, 64), then bf16, the NaN-ignoring / bf16-max-
// clamped row absmax and the quantization. (Host code: x86-64 baseline target, no FMA contraction possible.)
float rbf_host(float v) {
  if (std::isnan(v)) return v;
  std::uint32_t b = std::bit_cast<std::uint32_t>(v);
  b += 0x7FFFu + ((b >> 16) & 1u);  // round to nearest even (carries into inf past the bf16 max)
  return std::bit_cast<float>(b & 0xFFFF0000u);
}

RotOut rotate_quantize_host(const std::vector<std::uint16_t>& x, std::int64_t m, int k, std::int64_t ldx) {
  constexpr float kBf16Max = 3.38953139e38f;
  RotOut out{std::vector<std::int8_t>(static_cast<std::size_t>(m * k)), std::vector<float>(static_cast<std::size_t>(m))};
  std::vector<float> row(static_cast<std::size_t>(k));
  for (std::int64_t r = 0; r < m; ++r) {
    for (int g = 0; g < k / quant::kConvRotGroup; ++g) {
      float v[256], y[256];
      for (int e = 0; e < 256; ++e) v[e] = bf16_float(x[static_cast<std::size_t>(r * ldx + g * 256 + e)]);
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
    for (float v : row) amax = std::fmax(amax, std::fabs(v));  // std::fmax ignores NaN
    const float s = std::fmax(std::fmin(amax, kBf16Max) * (1.0f / 127.0f), 1e-30f);
    const float s_bf = rbf_host(s);
    out.s[static_cast<std::size_t>(r)] = s;
    for (int c = 0; c < k; ++c) {
      const float q = std::nearbyint(rbf_host(row[static_cast<std::size_t>(c)] / s_bf));
      out.q[static_cast<std::size_t>(r * k + c)] = static_cast<std::int8_t>(std::fmin(127.0f, std::fmax(-128.0f, q)));
    }
  }
  return out;
}

// rotate_quantize with row strides ldx (input padded with NaN, which must not be read) and ldq; returns q compacted.
RotOut run_rotate_strided(const std::vector<std::uint16_t>& x, std::int64_t m, int k, std::int64_t ldx,
                          std::int64_t ldq) {
  std::vector<std::uint16_t> xp(static_cast<std::size_t>(m * ldx), 0x7FC0);
  for (std::int64_t r = 0; r < m; ++r) std::copy_n(x.begin() + r * k, k, xp.begin() + r * ldx);
  Buf<std::uint16_t> xd(xp);
  Buf<std::int8_t> q(std::vector<std::int8_t>(static_cast<std::size_t>(m * ldq), std::int8_t{0x55}));
  Buf<float> s(static_cast<std::size_t>(m));
  quant::rotate_quantize(as_bf16(xd), ldx, q.get(), ldq, s.get(), m, k, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const auto qh = q.download();
  RotOut out{std::vector<std::int8_t>(static_cast<std::size_t>(m * k)), s.download()};
  for (std::int64_t r = 0; r < m; ++r) {
    std::copy_n(qh.begin() + r * ldq, k, out.q.begin() + r * k);
    for (std::int64_t c = k; c < ldq; ++c) EXPECT_EQ(qh[static_cast<std::size_t>(r * ldq + c)], 0x55) << "q padding written";
  }
  return out;
}

// The host reference reproduces the torch vectors (so it is the Python emulation), then the kernels match it at every
// block shape: k 256 (one group), 768 (odd group count: half a warp idle), 12288 (the text encoder's down-projection
// input, 4 pairs per warp), 32768 (kMaxRotateK, 16 warps), with zero, NaN, +-inf, overflowing and all-NaN rows, packed
// and strided.
TEST(GpuQuantKernels, RotateQuantizeMatchesHostReference) {
  for (const char* name : {"rot3072", "rot8192"}) {
    const auto xa = load(std::string(name) + "_x");
    const RotOut ref = rotate_quantize_host(xa.to_vector<std::uint16_t>(), xa.shape[0], static_cast<int>(xa.shape[1]),
                                            xa.shape[1]);
    EXPECT_EQ(count_diff(ref.q, load(std::string(name) + "_q").to_vector<std::int8_t>()), 0u) << name << " host ref q";
    EXPECT_EQ(count_diff_bits(ref.s, load(std::string(name) + "_s").to_vector<float>()), 0u) << name << " host ref s";
  }
  const std::uint16_t kNaN = 0x7FC0, kInf = 0x7F80, kNegInf = 0xFF80, kBig = 0x7F70;  // kBig = 3.19e38
  for (int k : {256, 768, 12288, 32768}) {
    constexpr std::int64_t m = 10;
    auto x = synth_act(static_cast<std::uint64_t>(k) * 3 + 1, m, k);
    const auto at = [&](std::int64_t r, std::int64_t c) -> std::uint16_t& { return x[static_cast<std::size_t>(r * k + c)]; };
    for (int c = 0; c < k; ++c) {
      at(1, c) = 0;     // zero row: s = 1e-30, q = 0
      at(8, c) = kNaN;  // all NaN: amax 0, every q -128
    }
    at(9, 0) = kBig;                            // x0 + x1 overflows fp32: half the group +-inf, no NaN
    at(9, 1) = kBig;
    at(2, k / 3) = kNaN;                        // one NaN: its group turns NaN, ignored by the absmax
    at(3, k - 1) = kInf;                        // +inf: absmax clamped to the bf16 max
    at(4, 5) = kNegInf;
    at(5, 0) = kNaN;                            // NaN, +inf and -inf in different groups (one group when k = 256)
    at(5, k / 2 + 7) = kInf;
    at(5, k - 3) = kNegInf;
    at(6, 17) = kInf;                           // +inf and -inf in one group: inf - inf = NaN in the butterflies
    at(6, 18) = kNegInf;
    const RotOut ref = rotate_quantize_host(x, m, k, k);
    const RotOut got = run_rotate(x, m, k);
    EXPECT_EQ(count_diff(got.q, ref.q), 0u) << "k " << k;
    EXPECT_EQ(count_diff_bits(got.s, ref.s), 0u) << "k " << k;
    const RotOut strided = run_rotate_strided(x, m, k, k + 64, k + 32);
    EXPECT_EQ(count_diff(strided.q, ref.q), 0u) << "k " << k << " strided";
    EXPECT_EQ(count_diff_bits(strided.s, ref.s), 0u) << "k " << k << " strided";
    // The documented special cases.
    EXPECT_EQ(ref.s[1], 1e-30f);
    EXPECT_EQ(std::count(ref.q.begin() + k, ref.q.begin() + 2 * k, std::int8_t{0}), k);
    EXPECT_EQ(ref.s[8], 1e-30f);
    EXPECT_EQ(std::count(ref.q.begin() + 8 * k, ref.q.begin() + 9 * k, std::int8_t{-128}), k);
    EXPECT_EQ(ref.s[3], 3.38953139e38f * (1.0f / 127.0f));
    EXPECT_EQ(ref.s[9], 3.38953139e38f * (1.0f / 127.0f));
    EXPECT_TRUE(std::isfinite(ref.s[2]) && ref.s[2] > 0) << "a NaN element must not reach the scale";
  }
}

TEST(GpuQuantKernels, FusedNormRotateMatchesReference) {
  NormInputs v{Buf<std::uint16_t>(load("norm_weight").to_vector<std::uint16_t>()),
               Buf<std::uint16_t>(load("norm_mod_scale").to_vector<std::uint16_t>()),
               Buf<std::uint16_t>(load("norm_mod_shift").to_vector<std::uint16_t>())};
  for (int m : {1, 5, 17, 40}) {
    const std::string p = "norm" + std::to_string(m);
    const auto x = load(p + "_x").to_vector<std::uint16_t>();
    const RotOut fused = run_fused_norm(x, m, 3072, v);
    EXPECT_EQ(count_diff(fused.q, load(p + "_q").to_vector<std::int8_t>()), 0u) << p << " vs torch reference";
    EXPECT_EQ(count_diff_bits(fused.s, load(p + "_s").to_vector<float>()), 0u) << p << " scales vs torch reference";
    const RotOut chain = run_norm_op_by_op(x, m, 3072, v);
    EXPECT_EQ(count_diff(fused.q, chain.q), 0u) << p << " vs op-by-op";
    EXPECT_EQ(count_diff_bits(fused.s, chain.s), 0u) << p << " scales vs op-by-op";
  }
}

// Every reduce shape torch uses (rows 1..16 and beyond), DiT run lengths, other row lengths, against the op-by-op
// chain; the K = 7680 / 1 row case runs the 512-thread reduce over 30 groups.
TEST(GpuQuantKernels, FusedNormRotateMatchesOpByOpAllRowCounts) {
  struct Case {
    int k;
    std::vector<std::int64_t> rows;
  };
  const Case cases[] = {
      {3072, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 31, 32, 33, 64, 400, 4106, 8484}},
      {1024, {1, 7, 16, 100}},
      {7680, {1, 3}},
  };
  for (const Case& c : cases) {
    NormInputs v{Buf<std::uint16_t>(synth_vec(c.k + 1, static_cast<std::size_t>(c.k), 1.0f, 0.6f)),
                 Buf<std::uint16_t>(synth_vec(c.k + 2, static_cast<std::size_t>(c.k), 0.0f, 1.0f)),
                 Buf<std::uint16_t>(synth_vec(c.k + 3, static_cast<std::size_t>(c.k), 0.0f, 1.0f))};
    for (std::int64_t m : c.rows) {
      const auto x = synth_act(static_cast<std::uint64_t>(m * 131 + c.k), m, c.k);
      const RotOut fused = run_fused_norm(x, m, c.k, v);
      const RotOut chain = run_norm_op_by_op(x, m, c.k, v);
      EXPECT_EQ(count_diff(fused.q, chain.q), 0u) << "k " << c.k << " rows " << m;
      EXPECT_EQ(count_diff_bits(fused.s, chain.s), 0u) << "k " << c.k << " rows " << m;
    }
  }
}

TEST(GpuQuantKernels, RotateRejectsBadArguments) {
  Buf<std::uint16_t> x(16 * 8192);
  Buf<std::int8_t> q(16 * 8192);
  Buf<float> s(16);
  Buf<std::uint16_t> v(8192);
  bf16_t* xp = as_bf16(x);
  const auto rot = [&](const bf16_t* xx, std::int64_t ldx, std::int8_t* qq, std::int64_t ldq, int k) {
    quant::rotate_quantize(xx, ldx, qq, ldq, s.get(), 4, k, nullptr);
  };
  EXPECT_THROW(rot(xp, 3000, q.get(), 3000, 3000), std::invalid_argument);  // K % 256 != 0
  EXPECT_THROW(rot(xp, 128, q.get(), 128, 128), std::invalid_argument);
  EXPECT_THROW(rot(xp, 0, q.get(), 0, 0), std::invalid_argument);
  EXPECT_THROW(rot(xp, 32768 + 256, q.get(), 32768 + 256, 32768 + 256), std::invalid_argument);
  EXPECT_THROW(rot(xp + 1, 3072, q.get(), 3072, 3072), std::invalid_argument);  // misaligned
  EXPECT_THROW(rot(xp, 3076, q.get(), 3072, 3072), std::invalid_argument);      // ldx % 8
  EXPECT_THROW(rot(xp, 3072, q.get(), 3080, 3072), std::invalid_argument);      // ldq % 16
  EXPECT_THROW(rot(xp, 2048, q.get(), 3072, 3072), std::invalid_argument);      // ldx < k
  EXPECT_NO_THROW(quant::rotate_quantize(xp, 3072, q.get(), 3072, s.get(), 0, 3072, nullptr));  // no rows: no-op
  // torch splits rows this long across warps at one row: not reproduced, rejected.
  EXPECT_THROW(quant::rms_norm_modulate_rotate_quantize(xp, 8192, as_bf16(v), as_bf16(v), as_bf16(v), q.get(), 8192,
                                                        s.get(), 1, 8192, nullptr),
               std::invalid_argument);
  EXPECT_THROW(quant::rms_norm_modulate_rotate_quantize(xp, 3072, as_bf16(v) + 1, as_bf16(v), as_bf16(v), q.get(),
                                                        3072, s.get(), 4, 3072, nullptr),
               std::invalid_argument);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------------------------------------------
// INT8 GEMM.

// The four kernels captured into a CUDA graph on a non-blocking stream (the engine's thread-local capture mode),
// replayed twice, against the oracle / host reference. Defined before the other GEMM tests, and each (epilogue, tile)
// pair here is one no earlier test runs, so the first launch of each kernel in the process (module load, CUTLASS's
// shared-memory attribute) happens inside the capture; under ctest every test is its own process anyway.
TEST(GpuQuantKernels, GraphCaptureOnStream) {
  GemmData d(200, 264, 512, 4242);
  const std::int64_t mn = d.m * d.n;
  Buf<std::uint16_t> out(static_cast<std::size_t>(mn)), situ(static_cast<std::size_t>(mn / 2));
  Buf<std::uint16_t> res0 = dev_act(17, d.m, d.n), res(static_cast<std::size_t>(mn));
  const auto xr = synth_act(23, 64, 768);
  Buf<std::uint16_t> xd(xr);
  Buf<std::int8_t> q(xr.size());
  Buf<float> s(64);
  MI_CUDA_CHECK(cudaMemcpy(res.get(), res0.get(), static_cast<std::size_t>(mn) * 2, cudaMemcpyDeviceToDevice));
  MI_CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t stream = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  quant::rotate_quantize(as_bf16(xd), 768, q.get(), 768, s.get(), 64, 768, stream);
  quant::int8_gemm(d.ops(), as_bf16(d.bias), as_bf16(out), d.n, stream, Int8Tile::k64x128x64);
  quant::int8_gemm_situ_glu(d.ops(), as_bf16(situ), d.n / 2, stream, Int8Tile::k128x128x128);
  MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  // The gated residual updates in place: capture it alone and replay once (two replays would add twice).
  cudaGraph_t graph_res = nullptr;
  cudaGraphExec_t exec_res = nullptr;
  MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  quant::int8_gemm_gated_residual(d.ops(), nullptr, as_bf16(d.gate), as_bf16(res), d.n, stream, Int8Tile::k256x128x64);
  MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_res));
  MI_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  MI_CUDA_CHECK(cudaGraphInstantiate(&exec_res, graph_res, 0));
  MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  MI_CUDA_CHECK(cudaGraphLaunch(exec_res, stream));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));

  const auto rows = all_rows(d.m);
  EXPECT_EQ(count_diff(out.download(), oracle(d, rows, 0, as_bf16(d.bias), nullptr)), 0u) << "int8_gemm";
  EXPECT_EQ(count_diff(situ.download(), oracle_situ(d, rows)), 0u) << "situ_glu";
  EXPECT_EQ(count_diff(res.download(), oracle(d, rows, 1, nullptr, as_bf16(res0))), 0u) << "gated_residual";
  const RotOut ref = rotate_quantize_host(xr, 64, 768, 768);
  EXPECT_EQ(count_diff(q.download(), ref.q), 0u) << "rotate_quantize";
  EXPECT_EQ(count_diff_bits(s.download(), ref.s), 0u) << "rotate_quantize scales";
  MI_CUDA_CHECK(cudaGraphExecDestroy(exec_res));
  MI_CUDA_CHECK(cudaGraphDestroy(graph_res));
  MI_CUDA_CHECK(cudaGraphExecDestroy(exec));
  MI_CUDA_CHECK(cudaGraphDestroy(graph));
  MI_CUDA_CHECK(cudaStreamDestroy(stream));
}

// The torch vectors: oracle == torch._int_mm and the CUDA-order dequant; every tile bit-exact against them; the
// AC19 numbers against comfy-kitchen's eager int8_linear (acc * (s_x * s_w), bias added in bf16).
TEST(GpuQuantKernels, Int8GemmMatchesReference) {
  const auto xa = load("gemm_xq"), wa = load("gemm_wq");
  const std::int64_t m = xa.shape[0], k = xa.shape[1], n = wa.shape[0];
  GemmData d(m, n, k, 1);
  d.x.upload(xa.to_vector<std::int8_t>());
  d.w.upload(wa.to_vector<std::int8_t>());
  d.xs.upload(load("gemm_xs").to_vector<float>());
  d.ws.upload(load("gemm_ws").to_vector<float>());
  d.bias.upload(load("gemm_bias").to_vector<std::uint16_t>());
  d.gate.upload(load("gemm_gate").to_vector<std::uint16_t>());
  const auto y_ref = load("gemm_y").to_vector<std::uint16_t>();
  const auto y_bias_ref = load("gemm_y_bias").to_vector<std::uint16_t>();
  const auto residual = load("gemm_residual").to_vector<std::uint16_t>();
  const auto res_ref = load("gemm_res").to_vector<std::uint16_t>();
  const auto situ_ref = load("gemm_situ").to_vector<std::uint16_t>();
  const auto rows = all_rows(m);

  std::vector<std::int32_t> acc;
  EXPECT_EQ(count_diff(oracle(d, rows, 0, nullptr, nullptr, &acc), y_ref), 0u) << "oracle vs torch";
  EXPECT_EQ(count_diff(acc, load("gemm_acc").to_vector<std::int32_t>()), 0u) << "oracle int32 vs torch._int_mm";

  Buf<std::uint16_t> out(static_cast<std::size_t>(m * n)), up(static_cast<std::size_t>(m * n));
  for (Int8Tile tile : kTiles) {
    const std::string tag = quant::int8_tile_name(tile);
    quant::int8_gemm(d.ops(), nullptr, as_bf16(out), n, nullptr, tile);
    EXPECT_EQ(count_diff(out.download(), y_ref), 0u) << tag << " int8_gemm";
    quant::int8_gemm(d.ops(), as_bf16(d.bias), as_bf16(out), n, nullptr, tile);
    EXPECT_EQ(count_diff(out.download(), y_bias_ref), 0u) << tag << " int8_gemm + bias";
    Buf<std::uint16_t> res(residual);
    quant::int8_gemm_gated_residual(d.ops(), nullptr, as_bf16(d.gate), as_bf16(res), n, nullptr, tile);
    EXPECT_EQ(count_diff(res.download(), res_ref), 0u) << tag << " gated residual";
    quant::int8_gemm_situ_glu(d.ops(), as_bf16(out), n / 2, nullptr, tile);
    const auto situ = out.download();
    // Against the engine's exact situ_glu on the bf16 GEMM output: bit for bit.
    up.upload(y_ref);
    Buf<std::uint16_t> ffn(static_cast<std::size_t>(m * n / 2));
    mi::kernels::situ_glu(view_bf16(up.get(), m, n), view_bf16(ffn.get(), m, n / 2), nullptr);
    std::vector<std::uint16_t> situ_got(situ.begin(), situ.begin() + m * n / 2);
    EXPECT_EQ(count_diff(situ_got, ffn.download()), 0u) << tag << " situ_glu vs kernels::situ_glu";
    const Agreement a = agreement(situ_got, situ_ref);
    EXPECT_LE(a.max_ulp, 1) << tag << " situ_glu vs torch _situ";
    if (tile == Int8Tile::kAuto) {
      std::printf("[quant] situ_glu vs torch _situ: %.4f %% bit-identical, max %d ULP\n", 100 * a.identical, a.max_ulp);
    }
  }

  // AC19: against comfy-kitchen's eager int8_linear.
  for (bool bias : {false, true}) {
    quant::int8_gemm(d.ops(), bias ? as_bf16(d.bias) : nullptr, as_bf16(out), n, nullptr);
    const Agreement a = agreement(out.download(), load(bias ? "gemm_y_eager_bias" : "gemm_y_eager").to_vector<std::uint16_t>());
    std::printf("[quant] int8_gemm%s vs eager int8_linear: cos %.9f, %.4f %% bit-identical, %.4f %% within 1 ULP, max "
                "%d ULP\n",
                bias ? " + bias" : "", a.cosine, 100 * a.identical, 100 * a.within1, a.max_ulp);
    EXPECT_GE(a.cosine, 0.99999);
    // The bias is added before the single bf16 rounding, as comfy-kitchen's CUDA epilogue (and a bf16 F.linear's
    // bias epilogue) does; eager rounds first and adds in bf16, which differs by many ULP where y + bias cancels.
    // Bit-exactness against the CUDA order is checked above; no quantized MOSS / Qwen3.5 linear has a bias.
    if (!bias) EXPECT_GE(a.within1, 0.999);
    RecordProperty(bias ? "eager_identical_bias" : "eager_identical", std::to_string(a.identical));
  }
}

struct ModelShape {
  const char* name;
  std::int64_t m, n, k;
};

// The DiT block linears over the CFG batch of a 1024^2 step (2 x 4096 image tokens + 282 + 10 text tokens), its
// cond-only run, the 1456^2 + 4096-token down projection, and the Qwen3.5 text encoder's linears at a 282-token prompt.
constexpr std::int64_t kM1024 = 2 * 4096 + 282 + 10;
const ModelShape kDitShapes[] = {
    {"dit qkv", kM1024, 9216, 3072},   {"dit o", kM1024, 3072, 3072},          {"dit up_gate", kM1024, 16384, 3072},
    {"dit down", kM1024, 3072, 8192},  {"dit o cond-only", 4378, 3072, 3072},  {"dit down 1456", 8281 + 4096, 3072, 8192},
};
const ModelShape kTeShapes[] = {
    {"te q+gate", 282, 8192, 4096}, {"te k/v", 282, 1024, 4096},     {"te o/z/out", 282, 4096, 4096},
    {"te gate/up", 282, 12288, 4096}, {"te down", 282, 4096, 12288}, {"te gdn qkv", 282, 8192, 4096},
};

TEST(GpuQuantKernels, Int8GemmMatchesOracleAtModelShapes) {
  for (const ModelShape& s : kDitShapes) {
    GemmData d(s.m, s.n, s.k, static_cast<std::uint64_t>(s.n + s.k));
    check_gemm_against_oracle(d, sample_rows(s.m), kTiles, false, s.name);
  }
  for (const ModelShape& s : kTeShapes) {
    GemmData d(s.m, s.n, s.k, static_cast<std::uint64_t>(s.n + 3 * s.k));
    check_gemm_against_oracle(d, all_rows(s.m), {Int8Tile::kAuto}, true, s.name);
  }
}

// The banded tile order with bands wider than 8 and a short last band: a 75 MiB activation (over 3/4 of any L2 up to
// 96 MiB) bands at 128x256 tiles in L2 / 2 / (256 * 8192) N tiles, 16 on 64 MiB (40 = 16 + 16 + 8), 24 on 96 MiB
// (24 + 16). Every epilogue against the oracle on rows spread over every M tile.
TEST(GpuQuantKernels, Int8GemmWideBandsMatchOracle) {
  GemmData d(9600, 10240, 8192, 77);
  std::vector<int> rows;
  for (int r = 0; r < 9600; r += 37) rows.push_back(r);
  rows.push_back(9599);
  check_gemm_against_oracle(d, rows, {Int8Tile::kAuto, Int8Tile::k128x256x64, Int8Tile::k256x128x64}, true, "wide bands");
}

// Row counts that are not tile multiples: 1, every M in 17..400 (small N/K, all rows), and tile-boundary row counts
// at the DiT N/K with every tile.
TEST(GpuQuantKernels, Int8GemmEdgeShapes) {
  for (std::int64_t m = 17; m <= 400; ++m) {
    GemmData d(m, 264, 512, static_cast<std::uint64_t>(m));
    check_gemm_against_oracle(d, all_rows(m), {Int8Tile::kAuto}, m % 2 == 0, "small m " + std::to_string(m));
  }
  for (std::int64_t m : {1, 2, 3, 15, 16, 17, 100, 127, 128, 129, 255, 257, 400}) {
    for (const ModelShape& s : {ModelShape{"o", m, 3072, 3072}, ModelShape{"down", m, 3072, 8192},
                                ModelShape{"up_gate", m, 16384, 3072}}) {
      GemmData d(s.m, s.n, s.k, static_cast<std::uint64_t>(m * 7 + s.n));
      check_gemm_against_oracle(d, all_rows(m), kTiles, true, std::string(s.name) + " m " + std::to_string(m));
    }
  }
  GemmData one_group(37, 8, 256, 99);  // K = one rotation group, N = one epilogue access
  check_gemm_against_oracle(one_group, all_rows(37), kTiles, true, "k256 n8");
  Buf<std::uint16_t> out(8);
  quant::Int8Operands empty = one_group.ops();
  empty.m = 0;
  EXPECT_NO_THROW(quant::int8_gemm(empty, nullptr, as_bf16(out), 8, nullptr));
}

// The SiTU-GLU epilogue reads SiTU's two factors from tables over every bf16 input: every value goes through it.
// acc = 1 in every column (x = e_0, every weight row e_0) and x_scale = 1, so the dequantized up_gate value of column c
// is w_scale[c] exactly and w_scale places any bf16 value, NaN and inf included, in the hidden or the gate half.
// Passes: every hidden value against gate 1, every gate value against hidden 1, every hidden value against a
// permutation of every gate value. Against kernels::situ_glu on the same bf16 [1, 2F] values, bit for bit.
TEST(GpuQuantKernels, Int8GemmSituGluEveryBf16Value) {
  constexpr std::int64_t F = 65536, n = 2 * F, k = 16;
  std::vector<std::int8_t> xh(k, 0), wh(static_cast<std::size_t>(n * k), 0);
  xh[0] = 1;
  for (std::int64_t i = 0; i < n; ++i) wh[static_cast<std::size_t>(i * k)] = 1;
  Buf<std::int8_t> x(xh), w(wh);
  Buf<float> xs(std::vector<float>{1.0f});
  const std::uint16_t one = bf16_bits(1.0f);
  for (int pass = 0; pass < 3; ++pass) {
    std::vector<float> ws(static_cast<std::size_t>(n));
    std::vector<std::uint16_t> y(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < F; ++i) {
      const auto all = static_cast<std::uint16_t>(i), perm = static_cast<std::uint16_t>((i * 40503 + 12345) & 0xFFFF);
      const std::uint16_t h = pass == 1 ? one : all, g = pass == 0 ? one : pass == 1 ? all : perm;
      const auto at = static_cast<std::size_t>(i);
      ws[at] = bf16_float(h);
      ws[at + F] = bf16_float(g);
      // The dequantization adds the zero bias: -0 arrives as +0.
      y[at] = bf16_bits(ws[at] + 0.0f);
      y[at + F] = bf16_bits(ws[at + F] + 0.0f);
    }
    Buf<float> wsb(ws);
    Buf<std::uint16_t> yb(y), got(static_cast<std::size_t>(F)), ref(static_cast<std::size_t>(F));
    quant::Int8Operands o;
    o.x_q = x.get();
    o.ldx = k;
    o.x_scale = xs.get();
    o.w = w.get();
    o.w_scale = wsb.get();
    o.m = 1;
    o.n = n;
    o.k = k;
    quant::int8_gemm_situ_glu(o, as_bf16(got), F, nullptr);
    mi::kernels::situ_glu(view_bf16(yb.get(), 1, n), view_bf16(ref.get(), 1, F), nullptr);
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    EXPECT_EQ(count_diff(got.download(), ref.download()), 0u) << "pass " << pass;
  }
}

// Padded row strides everywhere (ldx > k, ldo > n, ldr > n, SiTU ldo > F) and K = 272: 4 full 64-deep k-tiles (2 of
// the 128-deep tile) plus a 16-deep residue; N = 264 is not a tile multiple either. Every tile, every row, against the
// oracle; the padding of the outputs must stay untouched.
TEST(GpuQuantKernels, Int8GemmPaddedStridesAndKResidue) {
  constexpr std::int64_t m = 300, n = 264, k = 272, ldx = k + 48, ldo = n + 24, ldr = n + 40, ldf = n / 2 + 12;
  constexpr std::uint16_t kSentinel = 0x7FC1;
  GemmData d(m, n, k, 2718, ldx);
  const auto rows = all_rows(m);
  Buf<std::uint16_t> res0 = dev_act(31, m, ldr);
  const auto ref_plain = oracle(d, rows, 0, as_bf16(d.bias), nullptr);
  const auto ref_res = oracle(d, rows, 1, as_bf16(d.bias), as_bf16(res0), nullptr, ldr);
  const auto ref_situ = oracle_situ(d, rows);
  const auto res0_h = res0.download();
  // Rows of `cols` columns out of a [m, ld] host copy, and a check that columns [cols, ld) still hold `pad(r, c)`.
  const auto compact = [&](const std::vector<std::uint16_t>& h, std::int64_t ld, std::int64_t cols, auto pad,
                           const std::string& tag) {
    std::vector<std::uint16_t> c(static_cast<std::size_t>(m * cols));
    std::size_t touched = 0;
    for (std::int64_t r = 0; r < m; ++r) {
      std::copy_n(h.begin() + r * ld, cols, c.begin() + r * cols);
      for (std::int64_t j = cols; j < ld; ++j) touched += h[static_cast<std::size_t>(r * ld + j)] != pad(r, j);
    }
    EXPECT_EQ(touched, 0u) << tag << " wrote into the row padding";
    return c;
  };
  const auto sentinel = [&](std::int64_t, std::int64_t) { return kSentinel; };
  const auto residual_pad = [&](std::int64_t r, std::int64_t j) { return res0_h[static_cast<std::size_t>(r * ldr + j)]; };
  for (Int8Tile tile : kTiles) {
    const std::string tag = std::string("tile ") + quant::int8_tile_name(tile);
    Buf<std::uint16_t> out(std::vector<std::uint16_t>(static_cast<std::size_t>(m * ldo), kSentinel));
    quant::int8_gemm(d.ops(), as_bf16(d.bias), as_bf16(out), ldo, nullptr, tile);
    EXPECT_EQ(count_diff(compact(out.download(), ldo, n, sentinel, tag + " int8_gemm"), ref_plain), 0u)
        << tag << " int8_gemm";
    Buf<std::uint16_t> res(static_cast<std::size_t>(m * ldr));
    res.upload(res0_h);
    quant::int8_gemm_gated_residual(d.ops(), as_bf16(d.bias), as_bf16(d.gate), as_bf16(res), ldr, nullptr, tile);
    EXPECT_EQ(count_diff(compact(res.download(), ldr, n, residual_pad, tag + " gated_residual"), ref_res), 0u)
        << tag << " gated_residual";
    Buf<std::uint16_t> f(std::vector<std::uint16_t>(static_cast<std::size_t>(m * ldf), kSentinel));
    quant::int8_gemm_situ_glu(d.ops(), as_bf16(f), ldf, nullptr, tile);
    EXPECT_EQ(count_diff(compact(f.download(), ldf, n / 2, sentinel, tag + " situ_glu"), ref_situ), 0u)
        << tag << " situ_glu";
  }
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

TEST(GpuQuantKernels, Int8GemmRejectsBadOperands) {
  GemmData d(32, 272, 512, 5);
  Buf<std::uint16_t> out(32 * 272);
  const auto bad = [&](auto mutate) {
    quant::Int8Operands o = d.ops();
    mutate(o);
    return o;
  };
  const auto plain = [&](const quant::Int8Operands& o) { quant::int8_gemm(o, nullptr, as_bf16(out), 272, nullptr); };
  EXPECT_THROW(plain(bad([](auto& o) { o.k = 500; o.ldx = 512; })), std::invalid_argument);  // k % 16
  EXPECT_THROW(plain(bad([](auto& o) { o.n = 260; })), std::invalid_argument);                // n % 8
  EXPECT_THROW(plain(bad([](auto& o) { o.ldx = 500; })), std::invalid_argument);              // ldx < k
  EXPECT_THROW(plain(bad([](auto& o) { o.x_q += 1; })), std::invalid_argument);               // misaligned
  EXPECT_THROW(plain(bad([](auto& o) { o.w_scale += 1; })), std::invalid_argument);
  EXPECT_THROW(plain(bad([](auto& o) { o.x_scale = nullptr; })), std::invalid_argument);
  EXPECT_THROW(plain(bad([](auto& o) { o.m = -1; })), std::invalid_argument);
  EXPECT_THROW(plain(bad([](auto& o) { o.k = quant::kMaxInt8GemmK + 1; o.ldx = o.k; })),
               std::invalid_argument);  // int32 accumulator bound (checked before any access)
  EXPECT_THROW(quant::int8_gemm(d.ops(), nullptr, as_bf16(out) + 1, 272, nullptr), std::invalid_argument);
  EXPECT_THROW(quant::int8_gemm(d.ops(), nullptr, as_bf16(out), 268, nullptr), std::invalid_argument);  // ldo
  EXPECT_THROW(quant::int8_gemm(d.ops(), nullptr, as_bf16(out), 272, nullptr, Int8Tile::kCount), std::invalid_argument);
  EXPECT_THROW(quant::int8_gemm_gated_residual(d.ops(), nullptr, nullptr, as_bf16(out), 272, nullptr),
               std::invalid_argument);  // gate required
  // SiTU-GLU: n = 2F with F % 4 != 0.
  GemmData odd(32, 264 + 4, 512, 6);
  EXPECT_THROW(quant::int8_gemm_situ_glu(odd.ops(), as_bf16(out), 136, nullptr), std::invalid_argument);
  EXPECT_THROW(quant::int8_gemm_situ_glu(d.ops(), as_bf16(out), 134, nullptr), std::invalid_argument);      // ldo % 4
  EXPECT_THROW(quant::int8_gemm_situ_glu(d.ops(), as_bf16(out) + 1, 136, nullptr), std::invalid_argument);  // 8-byte
  EXPECT_THROW(quant::int8_gemm_situ_glu(d.ops(), as_bf16(out), 132, nullptr), std::invalid_argument);      // ldo < F
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------------------------------------------
// Performance (prints; no timing assertions). Opt-in: MI_RUN_PERF=1.

#define MI_REQUIRE_PERF()                                                                      \
  do {                                                                                         \
    const char* run_perf = std::getenv("MI_RUN_PERF");                                         \
    if (run_perf == nullptr || std::string(run_perf) != "1") GTEST_SKIP() << "set MI_RUN_PERF=1"; \
  } while (0)

// Calls of an op of `one_ms` per call that fill `window_ms` (at least min_calls). A 0 ms single-call timing (below the
// event resolution) counts as 1 us.
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

// SM clock right now: clock64 ticks over a 200 us %globaltimer window, one thread.
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
  double ms = 0;   // per call
  double mhz = 0;  // SM clock at the end of the window
};

// Steady-state rate of each op: per round and op, window / 2 of warm-up calls then a timed window (default 160 ms,
// MI_QUANT_PERF_WINDOW_MS overrides; the power controller settles over ~1 s, so short windows still favour an op that
// follows a lighter one, and back-to-back millisecond windows mostly measure that), ops round-robin over `rounds`;
// returns the means.
std::vector<Rate> steady(Timer& t, const std::vector<std::function<void()>>& fns, int rounds = 2) {
  const char* env = std::getenv("MI_QUANT_PERF_WINDOW_MS");
  const double window = env != nullptr ? std::atof(env) : 160.0;
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

// Dense tensor-core peak at `mhz` on the current device (76 SMs on the RTX 4090 Laptop): sm_89 GeForce runs bf16 with
// fp32 accumulation at 512 FLOP / clock / SM and int8 at 2048 OP / clock / SM.
double peak_tops(double mhz, bool int8) {
  int dev = 0, sms = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return sms * (int8 ? 2048.0 : 512.0) * mhz * 1e6 / 1e12;
}

double wall_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

quant::Int8Operands Int8Operands_for(const Buf<std::int8_t>& x, const Buf<float>& xs, const Buf<std::int8_t>& w,
                                     const Buf<float>& ws, std::int64_t m, std::int64_t n, std::int64_t k) {
  quant::Int8Operands o;
  o.x_q = x.get();
  o.ldx = k;
  o.x_scale = xs.get();
  o.w = w.get();
  o.w_scale = ws.get();
  o.m = m;
  o.n = n;
  o.k = k;
  return o;
}

// Per DiT shape: every INT8 tile (and kAuto) vs the bf16 cuBLASLt and CUTLASS-fused GEMMs on the same shape, back to
// back; the rotate kernels per site; and the projected 1024^2 CFG step.
TEST(GpuQuantKernels, Throughput) {
  MI_REQUIRE_PERF();
  Timer timer;
  mi::kernels::Gemm gemm;
  mi::DeviceBuffer ws(mi::kernels::Gemm::kWorkspaceBytes);
  const std::int64_t m = kM1024;
  const std::int64_t kmax = 8192, nmax = 16384;
  Buf<std::int8_t> xq = dev_i8(1, static_cast<std::size_t>(m * kmax)), wq = dev_i8(2, static_cast<std::size_t>(nmax * kmax));
  Buf<float> xs = dev_scale(3, static_cast<std::size_t>(m), 0.002f, 0.2f), wsc = dev_scale(4, static_cast<std::size_t>(nmax), 1e-5f, 1e-3f);
  Buf<std::uint16_t> xb = dev_act(5, m, kmax), wb = dev_act(6, nmax, kmax), res = dev_act(7, m, 3072),
                     gate = dev_vec(8, 3072, 0.0f, 1.0f), out(static_cast<std::size_t>(m * nmax));
  struct Site {
    const char* name;
    std::int64_t n, k;
    int kind;  // 0 plain (qkv), 1 gated residual (o, down), 2 SiTU-GLU (up_gate)
  };
  const Site sites[] = {{"qkv", 9216, 3072, 0}, {"o", 3072, 3072, 1}, {"up_gate", 16384, 3072, 2}, {"down", 3072, 8192, 1}};
  double int8_sum = 0, bf16_sum = 0;
  // Warm up into the power-capped steady state before ranking tiles.
  for (const double t0 = wall_seconds(); wall_seconds() - t0 < 2.0;) {
    timer.ms([&] { quant::int8_gemm(Int8Operands_for(xq, xs, wq, wsc, m, 3072, 8192), nullptr, as_bf16(out), 3072, nullptr); }, 10);
  }
  for (const Site& s : sites) {
    const quant::Int8Operands o = Int8Operands_for(xq, xs, wq, wsc, m, s.n, s.k);
    const Tensor x_t = view_bf16(xb.get(), m, s.k), w_t = view_bf16(wb.get(), s.n, s.k);
    const Tensor out_t = view_bf16(out.get(), m, s.kind == 2 ? s.n / 2 : s.n), res_t = view_bf16(res.get(), m, 3072);
    const Tensor big_t = view_bf16(out.get(), m, s.n), gate_t = Tensor::view(gate.get(), DType::kBF16, mi::Device::kCUDA, {std::int64_t{3072}});
    const auto int8_run = [&](Int8Tile tile) {
      return [&, tile] {
        if (s.kind == 0) quant::int8_gemm(o, nullptr, as_bf16(out), s.n, nullptr, tile);
        if (s.kind == 1) quant::int8_gemm_gated_residual(o, nullptr, as_bf16(gate), as_bf16(res), 3072, nullptr, tile);
        if (s.kind == 2) quant::int8_gemm_situ_glu(o, as_bf16(out), s.n / 2, nullptr, tile);
      };
    };
    // The bf16 path the DiT block runs today: cuBLASLt for qkv, the CUTLASS fused epilogues for the others.
    const std::function<void()> bf16_engine = [&] {
      if (s.kind == 0) gemm.linear(big_t, x_t, w_t, nullptr, ws.get(), ws.size(), nullptr);
      if (s.kind == 1) mi::kernels::fused::gemm_gated_residual(res_t, x_t, w_t, gate_t, nullptr);
      if (s.kind == 2) mi::kernels::fused::gemm_situ_glu(out_t, x_t, w_t, nullptr);
    };
    const std::function<void()> bf16_cublas = [&] { gemm.linear(big_t, x_t, w_t, nullptr, ws.get(), ws.size(), nullptr); };
    const double ops = 2.0 * static_cast<double>(m) * static_cast<double>(s.n) * static_cast<double>(s.k);
    std::vector<std::function<void()>> fns = {bf16_cublas, bf16_engine};
    for (Int8Tile tile : kTiles) fns.push_back(int8_run(tile));
    const std::vector<Rate> ms = steady(timer, fns);
    const auto line = [&](const char* what, const Rate& r, bool int8, double ref_ms) {
      const double tops = ops / r.ms / 1e9;
      std::printf("[quant perf]   %-20s %6.3f ms %6.1f T(FL)OPS @ %4.0f MHz (%3.0f %% of peak) %5.2fx\n", what, r.ms, tops,
                  r.mhz, 100 * tops / peak_tops(r.mhz, int8), ref_ms / r.ms);
    };
    std::printf("[quant perf] %s M %lld N %lld K %lld (speedup vs bf16 engine)\n", s.name, static_cast<long long>(m),
                static_cast<long long>(s.n), static_cast<long long>(s.k));
    line("bf16 cuBLASLt", ms[0], false, ms[1].ms);
    line("bf16 engine", ms[1], false, ms[1].ms);
    std::size_t i = 0;
    for (Int8Tile tile : kTiles) {
      line((std::string("int8 ") + quant::int8_tile_name(tile)).c_str(), ms[2 + i], true, ms[1].ms);
      ++i;
    }
    int8_sum += ms[2].ms;
    bf16_sum += ms[1].ms;
    RecordProperty(std::string("int8 speedup ") + s.name, std::to_string(ms[1].ms / ms[2].ms));
  }

  // Rotate sites at the same rows: 2 fused norm sites (replacing rms_norm_modulate_moss) and 2 standalone ones.
  Buf<std::int8_t> qo(static_cast<std::size_t>(m * kmax));
  Buf<float> so(static_cast<std::size_t>(m));
  Buf<std::uint16_t> nw = dev_vec(9, 3072, 1.0f, 0.5f), msc = dev_vec(10, 3072, 0.0f, 1.0f), msh = dev_vec(11, 3072, 0.0f, 1.0f);
  const auto v3072 = [](const Buf<std::uint16_t>& b) {
    return Tensor::view(b.get(), DType::kBF16, mi::Device::kCUDA, {std::int64_t{3072}});
  };
  const std::vector<std::function<void()>> rot_fns = {
      [&] {
        mi::kernels::rms_norm_modulate_moss(view_bf16(xb.get(), m, 3072), v3072(nw), v3072(msc), v3072(msh),
                                            view_bf16(out.get(), m, 3072), nullptr);
      },
      [&] {
        quant::rms_norm_modulate_rotate_quantize(as_bf16(xb), 3072, as_bf16(nw), as_bf16(msc), as_bf16(msh), qo.get(),
                                                 3072, so.get(), m, 3072, nullptr);
      },
      [&] { quant::rotate_quantize(as_bf16(xb), 3072, qo.get(), 3072, so.get(), m, 3072, nullptr); },
      [&] { quant::rotate_quantize(as_bf16(xb), 8192, qo.get(), 8192, so.get(), m, 8192, nullptr); },
  };
  std::vector<double> rot;
  for (const Rate& r : steady(timer, rot_fns)) rot.push_back(r.ms);
  const double rd_bytes3072 = static_cast<double>(m) * 3072 * 3, rd_bytes8192 = static_cast<double>(m) * 8192 * 3;
  std::printf("[quant perf] rms_norm_modulate_moss (bf16 path)   %6.3f ms\n", rot[0]);
  std::printf("[quant perf] norm+modulate+rotate+quant K 3072   %6.3f ms (%.0f GB/s)\n", rot[1], rd_bytes3072 / rot[1] / 1e6);
  std::printf("[quant perf] rotate+quant K 3072 (o input)       %6.3f ms (%.0f GB/s)\n", rot[2], rd_bytes3072 / rot[2] / 1e6);
  std::printf("[quant perf] rotate+quant K 8192 (down input)    %6.3f ms (%.0f GB/s)\n", rot[3], rd_bytes8192 / rot[3] / 1e6);
  const double rot_sum = 2 * rot[1] + rot[2] + rot[3];
  std::printf("[quant perf] rotate sites / int8 GEMMs per block: %.3f / %.3f ms = %.1f %%\n", rot_sum, int8_sum,
              100 * rot_sum / int8_sum);
  // Item 1c: SiTU-GLU in the up_gate epilogue + standalone rotate of the [M, F] output, vs a plain up_gate GEMM to
  // [M, 2F] bf16 + a SiTU + rotate + quantize kernel reading it. The latter is proxied by rotate_quantize at K = 2F:
  // the same bytes read, twice the int8 bytes written (slightly pessimistic for it).
  {
    const quant::Int8Operands up = Int8Operands_for(xq, xs, wq, wsc, m, 16384, 3072);
    Buf<std::int8_t> q2(static_cast<std::size_t>(m * 16384));
    const std::vector<Rate> r = steady(timer, {
        [&] { quant::int8_gemm_situ_glu(up, as_bf16(out), 8192, nullptr); },
        [&] { quant::rotate_quantize(as_bf16(out), 8192, qo.get(), 8192, so.get(), m, 8192, nullptr); },
        [&] { quant::int8_gemm(up, nullptr, as_bf16(out), 16384, nullptr); },
        [&] { quant::rotate_quantize(as_bf16(out), 16384, q2.get(), 16384, so.get(), m, 16384, nullptr); },
    });
    std::printf("[quant perf] ffn: situ epilogue %.3f + rotate %.3f = %.3f ms | plain up_gate %.3f + situ-rotate (proxy) "
                "%.3f = %.3f ms\n",
                r[0].ms, r[1].ms, r[0].ms + r[1].ms, r[2].ms, r[3].ms, r[2].ms + r[3].ms);
  }
  // Projected 1024^2 CFG step: a measured bf16 fused step (MI_QUANT_BF16_STEP_MS; default 1380 ms, the step measured
  // on the development GPU when this test was written) minus 42 blocks of the bf16 GEMM sites and their two
  // rms_norm_modulate_moss, plus 42 blocks of the INT8 sites and the 4 rotate sites.
  const char* step_env = std::getenv("MI_QUANT_BF16_STEP_MS");
  const double bf16_step_ms = step_env != nullptr ? std::atof(step_env) : 1380.0;
  const double block_bf16 = bf16_sum + 2 * rot[0], block_int8 = int8_sum + rot_sum;
  const double projected = bf16_step_ms - 42 * (block_bf16 - block_int8);
  std::printf("[quant perf] per block: bf16 %.3f ms -> int8 %.3f ms; projected step %.0f ms vs %.0f ms (%.2fx)\n",
              block_bf16, block_int8, projected, bf16_step_ms, bf16_step_ms / projected);
  RecordProperty("projected_step_ms", std::to_string(projected));

  // Text-encoder shapes, kAuto INT8 vs cuBLASLt bf16.
  for (const ModelShape& s : kTeShapes) {
    const quant::Int8Operands o = Int8Operands_for(xq, xs, wq, wsc, s.m, s.n, s.k);
    const Tensor x_t = view_bf16(xb.get(), s.m, s.k), w_t = view_bf16(wb.get(), s.n, s.k), o_t = view_bf16(out.get(), s.m, s.n);
    const double ops = 2.0 * static_cast<double>(s.m * s.n) * static_cast<double>(s.k);
    std::vector<std::function<void()>> fns = {[&] { gemm.linear(o_t, x_t, w_t, nullptr, ws.get(), ws.size(), nullptr); }};
    for (Int8Tile tile : kTiles) fns.push_back([&, tile] { quant::int8_gemm(o, nullptr, as_bf16(out), s.n, nullptr, tile); });
    const std::vector<Rate> ms = steady(timer, fns);
    std::printf("[quant perf] %-11s M %lld N %5lld K %5lld | bf16 cuBLASLt %6.3f ms %5.1f TFLOPS @ %4.0f MHz | int8 auto "
                "(%s) %6.3f ms %5.1f TOPS @ %4.0f MHz | %.2fx | tiles:",
                s.name, static_cast<long long>(s.m), static_cast<long long>(s.n), static_cast<long long>(s.k), ms[0].ms,
                ops / ms[0].ms / 1e9, ms[0].mhz, quant::int8_tile_name(quant::int8_tile_for(s.m, s.n, s.k)), ms[1].ms,
                ops / ms[1].ms / 1e9, ms[1].mhz, ms[0].ms / ms[1].ms);
    for (std::size_t i = 2; i < ms.size(); ++i) std::printf(" %.3f", ms[i].ms);
    std::printf(" ms\n");
  }
}

// The sustained rate under a long INT8 GEMM loop vs the same for bf16 (ffn_down shape, 1024^2 CFG batch). Clocks and
// power come from an external sampler (nvidia-smi); the printed wall-clock stamps delimit the phases.
TEST(GpuQuantKernels, SustainedClock) {
  MI_REQUIRE_PERF();
  const char* env = std::getenv("MI_QUANT_SUSTAINED_SECONDS");
  const double seconds = env != nullptr ? std::atof(env) : 2.0;
  const std::int64_t m = kM1024, n = 3072, k = 8192;
  GemmData d(m, n, k, 77);
  Buf<std::uint16_t> xb = dev_act(8, m, k), wb = dev_act(9, n, k), out(static_cast<std::size_t>(m * n));
  mi::kernels::Gemm gemm;
  mi::DeviceBuffer ws(mi::kernels::Gemm::kWorkspaceBytes);
  const Tensor x_t = view_bf16(xb.get(), m, k), w_t = view_bf16(wb.get(), n, k), o_t = view_bf16(out.get(), m, n);
  const double ops = 2.0 * static_cast<double>(m * n) * static_cast<double>(k);
  Timer timer;
  const auto loop = [&](const char* name, const std::function<void()>& fn) {
    const int iters = calls_for(250.0, timer.ms(fn, 1), 1);  // ~0.25 s windows
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    const double t0 = wall_seconds();
    std::printf("[quant sustained] %s start %.3f\n", name, t0);
    double first = 0, last = 0, total_ops = 0, total_ms = 0, mhz_sum = 0, mhz_last = 0;
    int windows = 0;
    while (wall_seconds() - t0 < seconds) {
      const double ms = timer.ms(fn, iters);
      const double mhz = sm_mhz();
      const double tops = ops / ms / 1e9;
      if (windows == 0) first = tops;
      last = tops;
      mhz_last = mhz;
      mhz_sum += mhz;
      total_ops += iters * ops;
      total_ms += iters * ms;
      ++windows;
      std::printf("[quant sustained] %s t=%6.2f s %6.1f TOPS @ %4.0f MHz\n", name, wall_seconds() - t0, tops, mhz);
    }
    std::printf("[quant sustained] %s end %.3f: first window %.1f, last %.1f (%.0f MHz), mean %.1f TOPS @ %.0f MHz over "
                "%.1f s\n",
                name, wall_seconds(), first, last, mhz_last, total_ops / total_ms / 1e9, mhz_sum / windows,
                total_ms / 1e3);
    return total_ops / total_ms / 1e9;
  };
  const double int8 = loop("int8", [&] { quant::int8_gemm(d.ops(), nullptr, as_bf16(out), n, nullptr); });
  const double bf16 = loop("bf16", [&] { gemm.linear(o_t, x_t, w_t, nullptr, ws.get(), ws.size(), nullptr); });
  std::printf("[quant sustained] mean int8 %.1f TOPS vs bf16 %.1f TFLOPS: %.2fx\n", int8, bf16, int8 / bf16);
  RecordProperty("sustained_speedup", std::to_string(int8 / bf16));
}

}  // namespace
