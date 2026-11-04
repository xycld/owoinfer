// dit_kernels.h's ops against host replays of their op order, with tails on every axis:
//   - gemm_bias and gemm_bias_residual on small integers, whose sums are exact in fp32 in any order: bit for bit on any
//     leaf, at the DiT's shapes (M = 1 for the time MLP and modulations) and at shapes with tails.
//   - rms_norm, through its row_stats output: each row's fp32 sum of squares bit for bit on any device against a replay in
//     torch's reduce order for the row count (512 threads per row at 1 row down to 32 from 16 rows up; every row count
//     from 1 to 16 and two above), and its rsqrt within 2 fp32 ULPs of the correctly rounded one, (float)(1 / sqrt(double))
//     of the same fp32 mean + eps (CUDA's rsqrtf is documented within 2). At every row count with more than 32 threads per
//     row, at least one row must have another sum in the 32-lane order, or the inputs could not tell the orders apart:
//     where the random rows give none, row 0 is redrawn from a wider range of magnitudes until it does. Then every output
//     with the bits of a replay from the device's own rsqrt, on any device.
//   - silu's and situ_glu's outputs: on the NVIDIA build (whose exp and tanh are CUDA's) every output with the replay's
//     bits; on AMD at least 99 % of them and none further than 3 ULPs of max(|replay|, the output's rms), as AMD's math
//     library may differ from the host's in a low bit that a rounding point carries into an output (3 ULPs is
//     tools/hip_block's limit for the same effect in its norms).
//   - rope_table against a replay of rope.cu's expressions with the host's sinf and cosf: at least 99.9 % of the values with
//     its bits and none further than 1 ULP, on every device (the replay's sinf is not the device's, so not exact). That
//     checks the coordinates and the layout; on the NVIDIA build the kernel's expressions are the engine's.
// Usage: hip_dit_kernel_check [--device N]. Exit status: 0 passed, 1 failed, 2 no usable device.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "dit_kernels.h"
#include "half.h"
#include "runtime.h"

namespace {

using namespace mi::hip;
using bf16 = std::uint16_t;

float rbf(float v) { return bf16_to_float(float_to_bf16(v)); }

struct Rng {
  std::uint64_t s;
  std::uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  int integer(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo + 1)); }
  // A bf16 value of magnitude about 2^e for e in [lo, hi], either sign.
  float spread(int lo, int hi) {
    const std::uint64_t z = next();
    const float m = 1.0f + static_cast<float>(z & 0x7F) / 128.0f;
    return rbf(std::ldexp((z >> 7) & 1 ? -m : m, lo + static_cast<int>((z >> 8) % static_cast<std::uint64_t>(hi - lo + 1))));
  }
};

std::vector<bf16> bits(const std::vector<float>& v) {
  std::vector<bf16> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = float_to_bf16(v[i]);
  return out;
}

DeviceBuffer upload(const std::vector<bf16>& v) {
  DeviceBuffer b(v.size() * 2);
  b.upload(v.data(), v.size() * 2);
  return b;
}

std::vector<bf16> download(const DeviceBuffer& b, std::size_t n) {
  std::vector<bf16> out(n);
  b.download(out.data(), n * 2);
  return out;
}

int failures = 0;
bool g_exact = false;  // the NVIDIA build: silu's and situ_glu's outputs must equal the replays

// Exact: every output's bits.
void report_exact(const std::string& name, const std::vector<bf16>& got, const std::vector<bf16>& want) {
  std::size_t differ = 0;
  for (std::size_t i = 0; i < got.size(); ++i) differ += got[i] != want[i];
  const bool ok = differ == 0;
  std::printf("%-44s %10zu values, %zu differ (exact) %s\n", name.c_str(), got.size(), differ, ok ? "ok" : "FAIL");
  failures += ok ? 0 : 1;
}

// Within limits: the share with the replay's bits and the largest distance in bf16 ULPs of max(|replay|, rms).
void report_close(const std::string& name, const std::vector<bf16>& got, const std::vector<bf16>& want, bool exact) {
  double sq = 0;
  for (const bf16 w : want) sq += static_cast<double>(bf16_to_float(w)) * bf16_to_float(w);
  const double rms = std::sqrt(sq / static_cast<double>(std::max<std::size_t>(want.size(), 1)));
  std::size_t same = 0, nan = 0;
  double worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float g = bf16_to_float(got[i]), w = bf16_to_float(want[i]);
    if (std::isnan(g) || std::isnan(w)) {
      nan += std::isnan(g) != std::isnan(w);
      same += std::isnan(g) && std::isnan(w);
      continue;
    }
    if (got[i] == want[i]) {
      ++same;
      continue;
    }
    const double m = std::max(static_cast<double>(std::abs(w)), rms);
    const double ulp = std::ldexp(1.0, std::ilogb(m) - 7);
    worst = std::max(worst, std::abs(static_cast<double>(g) - w) / ulp);
  }
  const double same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(got.size());
  const bool ok = nan == 0 && (exact ? same == got.size() : same_pct >= 99.0 && worst <= 3.0);
  std::printf("%-44s %10zu values, %8.4f %% same, max %.2f ULPs, %zu NaN mismatches %s\n", name.c_str(), got.size(), same_pct,
              worst, nan, ok ? "ok" : "FAIL");
  failures += ok ? 0 : 1;
}

void check_gemm(Rng& rng, int m, int n, int k, hipStream_t stream) {
  std::vector<float> x(static_cast<std::size_t>(m) * k), w(static_cast<std::size_t>(n) * k), bias(static_cast<std::size_t>(n)),
      res(static_cast<std::size_t>(m) * n);
  for (float& v : x) v = static_cast<float>(rng.integer(-3, 3));
  for (float& v : w) v = static_cast<float>(rng.integer(-3, 3));
  for (float& v : bias) v = static_cast<float>(rng.integer(-64, 64)) * 0.25f;
  for (float& v : res) v = static_cast<float>(rng.integer(-512, 512)) * 0.5f;
  const std::vector<bf16> xb = bits(x), wb = bits(w), bb = bits(bias), rb = bits(res);
  std::vector<bf16> want_bias(res.size()), want_res(res.size());
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      float s = 0.0f;
      for (int i = 0; i < k; ++i) s += x[static_cast<std::size_t>(r) * k + i] * w[static_cast<std::size_t>(c) * k + i];
      const std::size_t o = static_cast<std::size_t>(r) * n + c;
      const float biased = s + bias[static_cast<std::size_t>(c)];
      want_bias[o] = float_to_bf16(biased);
      want_res[o] = float_to_bf16(bf16_to_float(rb[o]) + rbf(biased));
    }
  }
  const DeviceBuffer dx = upload(xb), dw = upload(wb), db = upload(bb);
  DeviceBuffer out = upload(std::vector<bf16>(res.size(), 0x7fc0)), residual = upload(rb);
  hip_block::gemm_bias(out.as<bf16>(), n, dx.as<bf16>(), k, dw.as<bf16>(), k, db.as<bf16>(), m, n, k, stream);
  hip_block::gemm_bias_residual(residual.as<bf16>(), n, dx.as<bf16>(), k, dw.as<bf16>(), k, db.as<bf16>(), m, n, k, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const std::string shape = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
  report_exact("gemm_bias " + shape, download(out, res.size()), want_bias);
  report_exact("gemm_bias_residual " + shape, download(residual, res.size()), want_res);
}

// torch's threads per row for a last-dim reduction (kernels/basic/norm.cu), written out again for the replay.
int threads_per_row(int rows, int d) {
  const auto pow2 = [](int v) {
    int p = 1;
    while (p * 2 <= v) p *= 2;
    return p;
  };
  const int dim0 = d / 4;
  const int dim0_pow2 = dim0 < 512 ? pow2(dim0) : 512;
  const int dim1_pow2 = rows < 512 ? pow2(rows) : 512;
  int bw = std::min(dim0_pow2, 32);
  const int bh = std::min(dim1_pow2, 512 / bw);
  bw = std::min(dim0_pow2, 512 / bh);
  return bw;
}

// A row's fp32 sum of squares with bw threads per row, as norm.cu sums it: 4 running values per thread over its 4-value
// groups, added, then a halving tree down to one.
float row_sum(const float* xr, int d, int bw) {
  std::vector<float> v(static_cast<std::size_t>(bw));
  for (int tx = 0; tx < bw; ++tx) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int idx = tx; idx * 4 + 3 < d; idx += bw) {
      for (int i = 0; i < 4; ++i) acc[i] += xr[idx * 4 + i] * xr[idx * 4 + i];
    }
    v[static_cast<std::size_t>(tx)] = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  }
  for (int off = bw / 2; off >= 1; off /= 2) {
    for (int tx = 0; tx < off; ++tx) v[static_cast<std::size_t>(tx)] += v[static_cast<std::size_t>(tx + off)];
  }
  return v[0];
}

void check_norm(Rng& rng, int rows, int d, int spread, hipStream_t stream) {
  std::vector<float> x(static_cast<std::size_t>(rows) * d), w(static_cast<std::size_t>(d));
  for (float& v : x) v = rng.spread(-spread, spread);
  for (float& v : w) v = rng.spread(-2, 1);
  const int bw = threads_per_row(rows, d);
  // A row whose sum the two orders round differently, if no row already is: row 0 redrawn over 2^-12 .. 2^12.
  const auto order_sensitive = [&](int r) {
    const float* xr = x.data() + static_cast<std::size_t>(r) * d;
    return std::bit_cast<std::uint32_t>(row_sum(xr, d, 32)) != std::bit_cast<std::uint32_t>(row_sum(xr, d, bw));
  };
  if (bw != 32) {
    bool any = false;
    for (int r = 0; r < rows && !any; ++r) any = order_sensitive(r);
    for (int attempt = 0; !any && attempt < 1000; ++attempt) {
      for (int c = 0; c < d; ++c) x[static_cast<std::size_t>(c)] = rng.spread(-12, 12);
      any = order_sensitive(0);
    }
  }
  const std::vector<bf16> xb = bits(x), wb = bits(w);
  const float factor = static_cast<float>(rows) / static_cast<float>(static_cast<std::int64_t>(rows) * d);
  std::vector<float> sums(static_cast<std::size_t>(rows));
  std::size_t other_order = 0;
  for (int r = 0; r < rows; ++r) {
    const float* xr = x.data() + static_cast<std::size_t>(r) * d;
    sums[static_cast<std::size_t>(r)] = row_sum(xr, d, bw);
    if (bw != 32 && order_sensitive(r)) ++other_order;
  }
  const DeviceBuffer dx = upload(xb), dw = upload(wb);
  DeviceBuffer out = upload(std::vector<bf16>(x.size(), 0x7fc0));
  DeviceBuffer dstats(2 * sums.size() * sizeof(float));
  dstats.upload(std::vector<float>(2 * sums.size(), std::nanf("")).data(), 2 * sums.size() * sizeof(float));
  hip_block::rms_norm(dx.as<bf16>(), d, dw.as<bf16>(), out.as<bf16>(), d, rows, d, stream, dstats.as<float>());
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  std::vector<float> stats(2 * sums.size());
  dstats.download(stats.data(), stats.size() * sizeof(float));
  std::size_t sums_differ = 0;
  long worst_rsqrt = 0;
  std::vector<bf16> want(x.size());
  for (std::size_t r = 0; r < sums.size(); ++r) {
    const float got_sum = stats[2 * r], got_r = stats[2 * r + 1];
    sums_differ += std::bit_cast<std::uint32_t>(got_sum) != std::bit_cast<std::uint32_t>(sums[r]);
    const float mean_eps = sums[r] * factor + 1e-6f;  // the kernel's fp32 operations
    const auto host_r = static_cast<float>(1.0 / std::sqrt(static_cast<double>(mean_eps)));  // correctly rounded
    const long ulps = std::labs(static_cast<long>(std::bit_cast<std::uint32_t>(got_r)) - static_cast<long>(std::bit_cast<std::uint32_t>(host_r)));
    worst_rsqrt = std::max(worst_rsqrt, std::isfinite(got_r) ? ulps : 1L << 30);
    // The outputs from the device's own rsqrt: what is left is the kernel's rounding points.
    const float* xr = x.data() + r * static_cast<std::size_t>(d);
    for (int c = 0; c < d; ++c) want[r * static_cast<std::size_t>(d) + c] = float_to_bf16(rbf(xr[c] * got_r) * w[static_cast<std::size_t>(c)]);
  }
  const std::string name = "rms_norm " + std::to_string(rows) + "x" + std::to_string(d);
  const bool stats_ok = sums_differ == 0 && worst_rsqrt <= 2 && (bw == 32 || other_order > 0);
  std::printf("%-44s %10d rows, %zu sums of squares differ (exact; the 32-lane order differs in %zu), rsqrt %ld ULPs from the rounded %s\n",
              (name + " sums, " + std::to_string(bw) + " per row").c_str(), rows, sums_differ, other_order, worst_rsqrt,
              stats_ok ? "ok" : "FAIL");
  failures += stats_ok ? 0 : 1;
  report_close(name + " (" + std::to_string(bw) + " per row), 2^+-" + std::to_string(spread), download(out, x.size()), want, true);
}

void check_rope(int height, int width, int text, hipStream_t stream) {
  constexpr std::uint32_t kBandBits[16] = {0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2,
                                           0x3d0186e3, 0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528,
                                           0x3a83126f, 0x3a136a16, 0x39a5cb60, 0x393a7753};
  const std::int64_t hw = static_cast<std::int64_t>(height) * width, n = hw + text;
  const float h_mid = static_cast<float>((height - 1) / 2.0), w_mid = static_cast<float>((width - 1) / 2.0);
  std::vector<bf16> want(static_cast<std::size_t>(n) * 128);
  for (std::int64_t r = 0; r < n; ++r) {
    for (int c = 0; c < 128; ++c) {
      const int p = c & 63, axis = p >> 4, band = p & 15;
      float coord = 0.0f;
      if (r < hw) {
        if (axis == 1) coord = static_cast<float>(text);
        if (axis == 2) coord = static_cast<float>(r / width) - h_mid;
        if (axis == 3) coord = static_cast<float>(r % width) - w_mid;
      } else if (axis == 1) {
        coord = static_cast<float>(r - hw);
      }
      const float phase = coord * std::bit_cast<float>(kBandBits[band]);
      want[static_cast<std::size_t>(r) * 128 + static_cast<std::size_t>(c)] = float_to_bf16(c < 64 ? std::sin(phase) : std::cos(phase));
    }
  }
  DeviceBuffer out = upload(std::vector<bf16>(want.size(), 0x7fc0));
  hip_block::rope_table(out.as<bf16>(), height, width, text, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const std::vector<bf16> got = download(out, want.size());
  std::size_t same = 0, far = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    same += got[i] == want[i];
    // bf16 ULPs apart: the bit patterns on one ordered line, -0 and +0 at the same point.
    const auto ordered = [](bf16 b) { return (b & 0x8000u) != 0 ? -static_cast<int>(b & 0x7fffu) : static_cast<int>(b); };
    far += std::abs(ordered(got[i]) - ordered(want[i])) > 1;
  }
  const double pct = 100.0 * static_cast<double>(same) / static_cast<double>(got.size());
  const bool ok = pct >= 99.9 && far == 0;
  std::printf("%-44s %10zu values, %8.4f %% same, %zu further than 1 ULP %s\n",
              ("rope_table " + std::to_string(height) + "x" + std::to_string(width) + " + " + std::to_string(text)).c_str(),
              got.size(), pct, far, ok ? "ok" : "FAIL");
  failures += ok ? 0 : 1;
}

void check_silu(Rng& rng, int n, hipStream_t stream) {
  std::vector<float> x(static_cast<std::size_t>(n));
  for (float& v : x) v = rng.spread(-6, 4);
  std::vector<bf16> want(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) want[i] = float_to_bf16(x[i] / (1.0f + std::exp(-x[i])));
  const DeviceBuffer dx = upload(bits(x));
  DeviceBuffer out = upload(std::vector<bf16>(x.size(), 0x7fc0));
  hip_block::silu(dx.as<bf16>(), out.as<bf16>(), n, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  report_close("silu " + std::to_string(n), download(out, x.size()), want, g_exact);
}

void check_situ_glu(Rng& rng, int rows, int f, hipStream_t stream) {
  const std::size_t ld = 2 * static_cast<std::size_t>(f);
  std::vector<float> in(static_cast<std::size_t>(rows) * ld);
  for (float& v : in) v = rng.spread(-6, 6);
  std::vector<bf16> want(static_cast<std::size_t>(rows) * f);
  constexpr float kInv25 = 1.0f / 25.0f;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < f; ++c) {
      const float h = in[static_cast<std::size_t>(r) * ld + c], g = in[static_cast<std::size_t>(r) * ld + f + c];
      const float sig = 1.0f / (1.0f + std::exp(-g));
      const float left = rbf(rbf(4.0f * rbf(std::tanh(rbf(g * 0.25f)))) * rbf(sig));
      const float right = rbf(25.0f * rbf(std::tanh(rbf(h * kInv25))));
      want[static_cast<std::size_t>(r) * f + c] = float_to_bf16(left * right);
    }
  }
  const DeviceBuffer din = upload(bits(in));
  DeviceBuffer out = upload(std::vector<bf16>(want.size(), 0x7fc0));
  hip_block::situ_glu(din.as<bf16>(), static_cast<std::int64_t>(ld), out.as<bf16>(), f, rows, f, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  report_close("situ_glu " + std::to_string(rows) + "x" + std::to_string(f), download(out, want.size()), want, g_exact);
}

}  // namespace

int main(int argc, char** argv) {
  int index = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) index = std::atoi(argv[++i]);
    else {
      std::printf("usage: %s [--device N]\n", argv[0]);
      return 1;
    }
  }
  try {
    bool usable = false;
    for (const DeviceInfo& d : list_devices()) usable = usable || d.unusable_reason.empty();
    if (!usable && index < 0) {
      std::printf("no device runs this build (%s)\n", build_target());
      return 2;
    }
    Device device(index);
    std::printf("device %d: %s (%s), build %s, leaf %s\n", device.info().index, device.info().name.c_str(),
                device.info().arch.c_str(), build_target(), hip_block::leaf_name());
    g_exact = std::string(build_target()) == "nvidia";
    std::printf("rms_norm outputs exact from the device rsqrt; silu and situ_glu outputs %s\n", g_exact ? "exact (NVIDIA build)" : "within the AMD limits");
    Stream stream;
    Rng rng{1};
    // The DiT's: the time MLP and modulations at M = 1, the video embedder's K = 128, the refiner's; then tails.
    for (const auto& [m, n, k] : std::vector<std::array<int, 3>>{{1, 3072, 256}, {1, 18432, 3072}, {1, 6144, 3072},
                                                                 {64, 3072, 128}, {10, 3072, 4096}, {62, 1024, 3072},
                                                                 {37, 53, 72}, {33, 49, 24}, {1, 7, 16}, {95, 130, 40},
                                                                 // more than one cache block of tiles along M (20 tiles of
                                                                 // 32 rows at K 3072) and along N (13 of 48), with tails
                                                                 {705, 650, 3072}}) {
      check_gemm(rng, m, n, k, stream.get());
    }
    for (int rows = 1; rows <= 16; ++rows) check_norm(rng, rows, 3072, 4, stream.get());
    for (const int rows : {62, 100}) check_norm(rng, rows, 3072, 4, stream.get());
    for (const int rows : {1, 10, 62}) check_norm(rng, rows, 3072, 20, stream.get());
    check_norm(rng, 5, 128, 4, stream.get());
    check_norm(rng, 3, 1000, 4, stream.get());
    for (const auto& [h, w, t] : std::vector<std::array<int, 3>>{{64, 64, 62}, {64, 64, 10}, {91, 91, 29}, {0, 0, 4096}, {3, 5, 1}}) {
      check_rope(h, w, t, stream.get());
    }
    for (const int n : {3072, 3077, 1}) check_silu(rng, n, stream.get());
    for (const auto& [rows, f] : std::vector<std::array<int, 2>>{{10, 8192}, {3, 300}, {1, 17}}) check_situ_glu(rng, rows, f, stream.get());
    std::printf("%s (%d failed checks)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
