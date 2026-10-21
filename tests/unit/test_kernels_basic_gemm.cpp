// kernels/basic GEMM (cuBLASLt linear) against torch F.linear at the real DiT shapes, plus throughput.
// Inputs are hashed full-mantissa bf16 (synth_bf16 mirrors gen_basic_vectors.py), so only sampled torch
// output rows are stored.
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/gemm.h"
#include "util/json.h"
#include "util/npy.h"

namespace {

using mi::DType;
using mi::Tensor;

const std::filesystem::path kData = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/data/basic";

mi::npy::Array load(const std::string& name) { return mi::npy::load(kData / (name + ".npy")); }

// splitmix64-hashed bf16 values +-m * 2^(scale_exp - 7 - e), m in [128, 255], e in 0..7 (exact in bf16).
std::vector<std::uint16_t> synth_bf16(std::uint64_t seed, std::size_t n, int scale_exp) {
  std::vector<std::uint16_t> out(n);
  const std::uint64_t offset = seed * 0x9E3779B97F4A7C15ull;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = i + offset;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float m = static_cast<float>(128 + (z & 0x7F));
    const int e = static_cast<int>((z >> 7) & 7);
    const float v = std::ldexp((z >> 10) & 1 ? -m : m, scale_exp - 7 - e);
    out[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16);
  }
  return out;
}

float bf16_to_float(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

// One bf16 ULP at |v|'s binade (bf16 keeps 8 significant bits).
double bf16_ulp(double v) {
  const double a = std::fabs(v);
  if (a < 1e-30) return std::ldexp(1.0, -133);
  return std::ldexp(1.0, std::ilogb(a) - 7);
}

struct Case {
  const char* name;
  std::int64_t m, k, n;
  bool bias;
  DType out;
  std::uint64_t seed_base;  // 3 * case index in the generator
};

const Case kCases[] = {
    {"q", 4400, 3072, 3072, false, DType::kBF16, 0},
    {"up_gate", 4400, 3072, 16384, false, DType::kBF16, 3},
    {"down", 4400, 8192, 3072, false, DType::kBF16, 6},
    {"refiner_qkv", 300, 3072, 9216, true, DType::kBF16, 9},
    {"f32_bias", 300, 3072, 3072, true, DType::kF32, 12},
};

struct Problem {
  mi::DeviceBuffer x, w, b, out, ws;
  Tensor xt, wt, bt, ot;
};

Problem make_problem(const Case& c) {
  Problem p;
  const auto upload = [](mi::DeviceBuffer& buf, const std::vector<std::uint16_t>& v) {
    buf = mi::DeviceBuffer(v.size() * 2);
    buf.upload(v.data(), v.size() * 2);
  };
  upload(p.x, synth_bf16(c.seed_base + 1, static_cast<std::size_t>(c.m * c.k), 1));
  upload(p.w, synth_bf16(c.seed_base + 2, static_cast<std::size_t>(c.n * c.k), -5));
  p.xt = Tensor::view(p.x.get(), DType::kBF16, mi::Device::kCUDA, {c.m, c.k});
  p.wt = Tensor::view(p.w.get(), DType::kBF16, mi::Device::kCUDA, {c.n, c.k});
  if (c.bias) {
    const std::vector<std::uint16_t> b = synth_bf16(c.seed_base + 3, static_cast<std::size_t>(c.n), -1);
    if (c.out == DType::kBF16) {
      upload(p.b, b);
    } else {  // f32 output takes an f32 bias: the same values, widened exactly
      std::vector<float> wide(b.size());
      for (std::size_t i = 0; i < b.size(); ++i) wide[i] = bf16_to_float(b[i]);
      p.b = mi::DeviceBuffer(wide.size() * 4);
      p.b.upload(wide.data(), wide.size() * 4);
    }
    p.bt = Tensor::view(p.b.get(), c.out, mi::Device::kCUDA, {c.n});
  }
  p.out = mi::DeviceBuffer(static_cast<std::size_t>(c.m * c.n) * mi::dtype_size(c.out));
  p.ot = Tensor::view(p.out.get(), c.out, mi::Device::kCUDA, {c.m, c.n});
  p.ws = mi::DeviceBuffer(mi::kernels::Gemm::kWorkspaceBytes);
  return p;
}

}  // namespace

TEST(GpuKernelsBasicGemm, MatchesTorchLinearAtDitShapes) {
  mi::kernels::Gemm gemm;
  for (const Case& c : kCases) {
    SCOPED_TRACE(c.name);
    Problem p = make_problem(c);
    gemm.linear(p.ot, p.xt, p.wt, c.bias ? &p.bt : nullptr, p.ws.get(), p.ws.size(), nullptr);
    MI_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<std::int64_t> rows = load(std::string("gemm_") + c.name + "_rows").to_vector<std::int64_t>();
    const mi::npy::Array ref = load(std::string("gemm_") + c.name + "_out");
    const std::size_t es = mi::dtype_size(c.out);
    std::vector<char> got_row(static_cast<std::size_t>(c.n) * es);
    double dot = 0, gg = 0, rr = 0, max_abs = 0, max_ulp = 0;
    std::int64_t within = 0, total = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      p.out.download(got_row.data(), got_row.size(), static_cast<std::size_t>(rows[i] * c.n) * es);
      for (std::int64_t j = 0; j < c.n; ++j) {
        double g, r;
        const std::size_t at = i * static_cast<std::size_t>(c.n) + static_cast<std::size_t>(j);
        if (c.out == DType::kBF16) {
          g = bf16_to_float(reinterpret_cast<const std::uint16_t*>(got_row.data())[j]);
          r = bf16_to_float(reinterpret_cast<const std::uint16_t*>(ref.data.data())[at]);
        } else {
          g = reinterpret_cast<const float*>(got_row.data())[j];
          r = reinterpret_cast<const float*>(ref.data.data())[at];
        }
        dot += g * r;
        gg += g * g;
        rr += r * r;
        const double diff = std::fabs(g - r);
        max_abs = std::max(max_abs, diff);
        max_ulp = std::max(max_ulp, diff / bf16_ulp(r));
        within += diff <= bf16_ulp(r);
        ++total;
      }
    }
    const double cosine = dot / std::sqrt(gg * rr);
    const double frac = static_cast<double>(within) / static_cast<double>(total);
    const double scale_ulp = bf16_ulp(std::sqrt(rr / static_cast<double>(total)));  // ULP at the output RMS
    std::printf("[gemm %-11s] M=%lld K=%lld N=%lld%s: cosine %.9f, max|diff| %.3g (%.3f ULP at output RMS), "
                "max %.2f ULP per element, %.4f%% within 1 ULP (%lld sampled)\n",
                c.name, static_cast<long long>(c.m), static_cast<long long>(c.k), static_cast<long long>(c.n),
                c.bias ? " +bias" : "", cosine, max_abs, max_abs / scale_ulp, max_ulp, 100.0 * frac,
                static_cast<long long>(total));
    EXPECT_GE(cosine, 0.999999);
    EXPECT_GE(frac, 0.999);
  }
  EXPECT_EQ(gemm.cached_plans(), std::size(kCases));
}

TEST(GpuKernelsBasicGemm, PlanCacheAndArgumentChecks) {
  mi::kernels::Gemm gemm;
  Problem p = make_problem(kCases[3]);
  for (int i = 0; i < 3; ++i) gemm.linear(p.ot, p.xt, p.wt, &p.bt, p.ws.get(), p.ws.size(), nullptr);
  EXPECT_EQ(gemm.cached_plans(), 1u);
  gemm.linear(p.ot, p.xt, p.wt, nullptr, nullptr, 0, nullptr);  // no bias, no workspace: a second plan
  EXPECT_EQ(gemm.cached_plans(), 2u);
  // Row-strided output: x rows written into the first half of 2N-wide rows match the dense result.
  mi::DeviceBuffer wide(static_cast<std::size_t>(p.ot.dim(0) * p.ot.dim(1) * 2) * 2);
  Tensor ow = Tensor::view(wide.get(), DType::kBF16, mi::Device::kCUDA, {p.ot.dim(0), p.ot.dim(1)});
  ow.stride[0] = 2 * p.ot.dim(1);
  gemm.linear(ow, p.xt, p.wt, &p.bt, p.ws.get(), p.ws.size(), nullptr);
  gemm.linear(p.ot, p.xt, p.wt, &p.bt, p.ws.get(), p.ws.size(), nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint16_t> dense(static_cast<std::size_t>(p.ot.numel())), strided(dense.size() * 2);
  p.out.download(dense.data(), dense.size() * 2);
  wide.download(strided.data(), strided.size() * 2);
  for (std::int64_t r = 0; r < p.ot.dim(0); ++r) {
    for (std::int64_t c = 0; c < p.ot.dim(1); ++c) {
      ASSERT_EQ(strided[static_cast<std::size_t>(r * 2 * p.ot.dim(1) + c)], dense[static_cast<std::size_t>(r * p.ot.dim(1) + c)]);
    }
  }
  Tensor bad_w = p.wt;
  bad_w.shape[1] = 1024;
  EXPECT_THROW(gemm.linear(p.ot, p.xt, bad_w, nullptr, nullptr, 0, nullptr), std::invalid_argument);
  EXPECT_THROW(gemm.linear(p.ot, p.xt, p.wt, nullptr, static_cast<char*>(p.ws.get()) + 16, 1024, nullptr),
               std::invalid_argument);
}

TEST(GpuKernelsBasicGemm, ThroughputVsTorch) {
  std::ifstream f(kData / "manifest.json");
  std::stringstream ss;
  ss << f.rdbuf();
  const mi::json::Value manifest = mi::json::parse(ss.str());
  const mi::json::Value& bench = manifest.at("torch_f_linear_bench");
  mi::kernels::Gemm gemm;
  cudaEvent_t start, stop;
  MI_CUDA_CHECK(cudaEventCreate(&start));
  MI_CUDA_CHECK(cudaEventCreate(&stop));
  for (const Case& c : kCases) {
    if (c.out != DType::kBF16) continue;
    Problem p = make_problem(c);
    const Tensor* bias = c.bias ? &p.bt : nullptr;
    for (int i = 0; i < 3; ++i) gemm.linear(p.ot, p.xt, p.wt, bias, p.ws.get(), p.ws.size(), nullptr);
    constexpr int kIters = 20;
    MI_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kIters; ++i) gemm.linear(p.ot, p.xt, p.wt, bias, p.ws.get(), p.ws.size(), nullptr);
    MI_CUDA_CHECK(cudaEventRecord(stop));
    MI_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= kIters;
    const double tflops = 2.0 * static_cast<double>(c.m * c.n * c.k) / ms / 1e9;
    const double torch_tflops = bench.at(c.name).at("tflops").as_double();
    std::printf("[gemm %-11s] %.3f ms, %.1f TFLOPS (torch F.linear %.1f TFLOPS, ratio %.2f)\n", c.name, ms, tflops,
                torch_tflops, tflops / torch_tflops);
    ::testing::Test::RecordProperty(std::string("tflops_") + c.name, std::to_string(tflops));
  }
  MI_CUDA_CHECK(cudaEventDestroy(start));
  MI_CUDA_CHECK(cudaEventDestroy(stop));
}
