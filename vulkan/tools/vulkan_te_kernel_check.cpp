// Checks the text encoder's kernels (te_kernels.h) against host replays of their op order, on small shapes with tails and
// several sequences, and the gated delta rule also against an fp64 token-by-token recurrence. `same` is the share of
// outputs with the replay's bits; ULPs are bf16 ULPs of max(|replay|, floor), fp32 ULPs for fp32 outputs. The replays use
// the host's exp, sqrt and division, which differ from the device's in the last bits of fp32. Exit status: 0 passed, 1
// failed. Usage: vulkan_te_kernel_check [--device N]
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "half.h"
#include "runtime.h"
#include "te_kernels.h"

namespace {

using namespace mi::vk;
using bits = std::uint16_t;

float from_bf16(bits b) { return bf16_to_float(b); }
bits to_bf16(float v) { return float_to_bf16(v); }
float rbf(float v) { return from_bf16(to_bf16(v)); }

std::uint64_t mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// bf16 values +-m * 2^(scale_exp - 7 - e), m in [128, 255], e in 0..3.
std::vector<bits> synth_bf16(std::uint64_t seed, std::size_t n, int scale_exp) {
  std::vector<bits> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t z = mix(i + seed * 0x9E3779B97F4A7C15ull);
    const float m = static_cast<float>(128 + (z & 0x7F));
    const auto e = static_cast<int>((z >> 7) & 3u);
    out[i] = to_bf16(std::ldexp((z >> 10) & 1 ? -m : m, scale_exp - 7 - e));
  }
  return out;
}

template <typename T>
std::unique_ptr<Buffer> device_buffer(Device& device, const std::vector<T>& values) {
  auto b = std::make_unique<Buffer>(device, std::max<std::size_t>(values.size(), 1) * sizeof(T));
  if (!values.empty()) b->upload(values.data(), values.size() * sizeof(T));
  return b;
}

template <typename T>
std::vector<T> read(Buffer& b, std::size_t count) {
  std::vector<T> out(count);
  b.download(out.data(), count * sizeof(T));
  return out;
}

double ulps_off(double got, double ref, double floor_mag, int mantissa_bits) {
  int e = 0;
  (void)std::frexp(std::max(std::abs(ref), floor_mag), &e);
  return std::abs(got - ref) / std::ldexp(1.0, e - mantissa_bits);
}

double rms_of(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s / static_cast<double>(std::max<std::size_t>(v.size(), 1)));
}

struct Limit {
  double ulps;
  double floor_of_rms;
  double min_same_percent = 99.0;
};

// bf16 outputs against replayed values (rounded to bf16 for `same`).
bool report(const char* name, const std::vector<bits>& got, const std::vector<double>& ref, Limit limit) {
  const double floor_mag = rms_of(ref) * limit.floor_of_rms;
  double worst = 0.0;
  std::size_t nans = 0, same = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float g = from_bf16(got[i]);
    if (std::isnan(g)) ++nans;
    same += got[i] == to_bf16(static_cast<float>(ref[i]));
    worst = std::max(worst, ulps_off(g, ref[i], floor_mag, 8));
  }
  const double same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(got.size());
  const bool ok = nans == 0 && worst <= limit.ulps && same_pct >= limit.min_same_percent;
  std::printf("  %-26s %s (%zu values, %.3f %% same, limit %.1f %%; worst %.2f ULPs, limit %.1f; %zu NaN)\n", name,
              ok ? "ok" : "FAILED", got.size(), same_pct, limit.min_same_percent, worst, limit.ulps, nans);
  return ok;
}

// fp32 outputs.
bool report_f32(const char* name, const std::vector<float>& got, const std::vector<float>& ref, double max_ulps) {
  double worst = 0.0;
  std::size_t nans = 0, same = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(got[i])) ++nans;
    same += std::bit_cast<std::uint32_t>(got[i]) == std::bit_cast<std::uint32_t>(ref[i]);
    worst = std::max(worst, ulps_off(got[i], ref[i], 0.0, 24));
  }
  const double same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(got.size());
  const bool ok = nans == 0 && worst <= max_ulps;
  std::printf("  %-26s %s (%zu values, %.3f %% same; worst %.2f fp32 ULPs, limit %.1f; %zu NaN)\n", name, ok ? "ok" : "FAILED",
              got.size(), same_pct, worst, max_ulps, nans);
  return ok;
}

std::vector<double> widen(const std::vector<bits>& v) {
  std::vector<double> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = from_bf16(v[i]);
  return out;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---- norms ----

float host_inv_rms(const bits* row, int d, int bw, float factor, float eps) {
  std::vector<float> v(static_cast<std::size_t>(bw));
  for (int t = 0; t < bw; ++t) {
    std::array<float, 4> acc{};
    for (int idx = t; idx * 4 + 3 < d; idx += bw) {
      for (int i = 0; i < 4; ++i) {
        const float e = from_bf16(row[idx * 4 + i]);
        const float sq = e * e;
        acc[static_cast<std::size_t>(i)] += sq;
      }
    }
    v[static_cast<std::size_t>(t)] = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  }
  for (int off = bw / 2; off >= 32; off >>= 1) {
    for (int t = 0; t < off; ++t) v[static_cast<std::size_t>(t)] += v[static_cast<std::size_t>(t + off)];
  }
  for (int off = 16; off > 0; off >>= 1) {
    for (int t = 0; t < off; ++t) v[static_cast<std::size_t>(t)] += v[static_cast<std::size_t>(t + off)];
  }
  const float mean = v[0] * factor;
  return 1.0f / std::sqrt(mean + eps);
}

// x and out [tokens][heads] rows at (t * ld + h * hs); the replay of te_rms_norm.
std::vector<bits> replay_norm(const std::vector<bits>& x, int ldx, int hsx, const std::vector<bits>& w, const std::vector<bits>* gate,
                              int ldg, int hsg, const std::vector<bits>* table, std::vector<bits> out, int ldo, int hso,
                              int tokens, int heads, int d) {
  const std::uint64_t rows = static_cast<std::uint64_t>(tokens) * heads;
  const int bw = static_cast<int>(torch_reduce_threads(rows, static_cast<std::uint32_t>(d)));
  const float factor = static_cast<float>(rows) / static_cast<float>(rows * d);
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      const bits* xr = x.data() + static_cast<std::ptrdiff_t>(t) * ldx + static_cast<std::ptrdiff_t>(h) * hsx;
      const float r = host_inv_rms(xr, d, bw, factor, 1e-6f);
      std::vector<float> y(static_cast<std::size_t>(d));
      for (int c = 0; c < d; ++c) {
        const float xn = from_bf16(xr[c]) * r;
        if (gate == nullptr) {
          const float one_plus = 1.0f + from_bf16(w[static_cast<std::size_t>(c)]);
          y[static_cast<std::size_t>(c)] = xn * one_plus;
        } else {
          const float g = from_bf16((*gate)[static_cast<std::size_t>(t) * ldg + static_cast<std::size_t>(h) * hsg + c]);
          const float weighted = rbf(from_bf16(w[static_cast<std::size_t>(c)]) * rbf(xn));
          y[static_cast<std::size_t>(c)] = weighted * silu(g);
        }
      }
      bits* orow = out.data() + static_cast<std::ptrdiff_t>(t) * ldo + static_cast<std::ptrdiff_t>(h) * hso;
      for (int c = 0; c < d; ++c) orow[c] = to_bf16(y[static_cast<std::size_t>(c)]);
      if (table != nullptr) {
        const bits* tab = table->data() + static_cast<std::ptrdiff_t>(t) * 64;
        for (int c = 0; c < 32; ++c) {
          const float lo = rbf(y[static_cast<std::size_t>(c)]), hi = rbf(y[static_cast<std::size_t>(c + 32)]);
          const float co = from_bf16(tab[c]), si = from_bf16(tab[32 + c]);
          orow[c] = to_bf16(rbf(lo * co) + rbf(-hi * si));
          orow[c + 32] = to_bf16(rbf(hi * co) + rbf(lo * si));
        }
      }
    }
  }
  return out;
}

bool check_norms(Device& device, const TeKernels& k) {
  bool ok = true;
  // The hidden-state norm at row counts that give torch 512, 128, 64 and 32 threads per row.
  for (const int tokens : {1, 5, 9, 37}) {
    constexpr int d = 4096, ld = 4100;
    const auto x_h = synth_bf16(100 + tokens, static_cast<std::size_t>(tokens) * ld, 2), w_h = synth_bf16(7, d, -3);
    auto x = device_buffer(device, x_h), w = device_buffer(device, w_h),
         out = device_buffer(device, std::vector<bits>(static_cast<std::size_t>(tokens) * ld, 0x7fc0));
    RmsNormArgs a;
    a.x = x->address();
    a.ldx = ld;
    a.weight = w->address();
    a.out = out->address();
    a.ldo = ld;
    a.tokens = static_cast<std::uint32_t>(tokens);
    a.d = d;
    device.run([&](VkCommandBuffer cb) { k.record_rms_norm(cb, a); });
    std::vector<bits> want = replay_norm(x_h, ld, 0, w_h, nullptr, 0, 0, nullptr,
                                         std::vector<bits>(static_cast<std::size_t>(tokens) * ld, 0x7fc0), ld, 0, tokens, 1, d);
    std::vector<bits> got = read<bits>(*out, want.size());
    // Compare the rows only, not the stride's padding (which must stay untouched: NaN both sides).
    std::vector<bits> g2, w2;
    for (int t = 0; t < tokens; ++t) {
      for (int c = 0; c < d; ++c) {
        g2.push_back(got[static_cast<std::size_t>(t) * ld + c]);
        w2.push_back(want[static_cast<std::size_t>(t) * ld + c]);
      }
      for (int c = d; c < ld; ++c) ok = ok && got[static_cast<std::size_t>(t) * ld + c] == 0x7fc0;
    }
    const std::string name = "rms_norm rows " + std::to_string(tokens) + " (bw " +
                             std::to_string(torch_reduce_threads(static_cast<std::uint64_t>(tokens), d)) + ")";
    ok = report(name.c_str(), g2, widen(w2), {1.0, 0.0}) && ok;
  }

  // q norm + RoPE out of place from the [q | gate] per-head layout, and k norm + RoPE in place, as the attention layer.
  for (const int tokens : {2, 19}) {
    constexpr int heads = 3, kv_heads = 2, d = 256, ld = 2 * heads * d + kv_heads * d + 8;
    const auto qkv_h = synth_bf16(200 + tokens, static_cast<std::size_t>(tokens) * ld, 1), qw = synth_bf16(8, d, -3),
               kw = synth_bf16(9, d, -3);
    std::vector<bits> table_h(static_cast<std::size_t>(tokens) * 64);
    for (int t = 0; t < tokens; ++t) {
      for (int i = 0; i < 32; ++i) {
        const float phase = static_cast<float>(t) * std::pow(1e7f, -static_cast<float>(i) / 32.0f);
        table_h[static_cast<std::size_t>(t) * 64 + i] = to_bf16(std::cos(phase));
        table_h[static_cast<std::size_t>(t) * 64 + 32 + i] = to_bf16(std::sin(phase));
      }
    }
    auto qkv = device_buffer(device, qkv_h), qwb = device_buffer(device, qw), kwb = device_buffer(device, kw),
         table = device_buffer(device, table_h), q = device_buffer(device, std::vector<bits>(static_cast<std::size_t>(tokens) * heads * d));
    RmsNormArgs qa;
    qa.x = qkv->address();
    qa.ldx = ld;
    qa.hsx = 2 * d;
    qa.out = q->address();
    qa.ldo = heads * d;
    qa.hso = d;
    qa.weight = qwb->address();
    qa.tokens = static_cast<std::uint32_t>(tokens);
    qa.heads = heads;
    qa.d = d;
    qa.rope = table->address();
    RmsNormArgs ka = qa;
    ka.x = ka.out = qkv->address() + 2ull * heads * d * 2;
    ka.ldx = ka.ldo = ld;
    ka.hsx = ka.hso = d;
    ka.weight = kwb->address();
    ka.heads = kv_heads;
    device.run([&](VkCommandBuffer cb) {
      k.record_rms_norm(cb, qa);
      k.record_rms_norm(cb, ka);
    });
    const auto q_want = replay_norm(qkv_h, ld, 2 * d, qw, nullptr, 0, 0, &table_h,
                                    std::vector<bits>(static_cast<std::size_t>(tokens) * heads * d), heads * d, d, tokens, heads, d);
    std::vector<bits> k_in(qkv_h.begin() + 2 * heads * d, qkv_h.end());
    k_in.resize(qkv_h.size());
    const auto k_want = replay_norm(k_in, ld, d, kw, nullptr, 0, 0, &table_h, k_in, ld, d, tokens, kv_heads, d);
    const auto qkv_got = read<bits>(*qkv, qkv_h.size());
    std::vector<bits> k_got, k_ref;
    for (int t = 0; t < tokens; ++t) {
      for (int c = 0; c < kv_heads * d; ++c) {
        k_got.push_back(qkv_got[static_cast<std::size_t>(t) * ld + 2 * heads * d + c]);
        k_ref.push_back(k_want[static_cast<std::size_t>(t) * ld + c]);
      }
      for (int c = 0; c < 2 * heads * d; ++c) ok = ok && qkv_got[static_cast<std::size_t>(t) * ld + c] == qkv_h[static_cast<std::size_t>(t) * ld + c];
    }
    const std::string qn = "q norm+rope, " + std::to_string(tokens) + " tokens", kn = "k norm+rope in place, " + std::to_string(tokens);
    ok = report(qn.c_str(), read<bits>(*q, q_want.size()), widen(q_want), {1.0, 0.0}) && ok;
    ok = report(kn.c_str(), k_got, widen(k_ref), {1.0, 0.0}) && ok;
  }

  // The gated norm in place over 128-wide heads, the gate in another buffer's rows.
  {
    constexpr int tokens = 7, heads = 5, d = 128, ld = heads * d, ldg = 3 * ld;
    const auto x_h = synth_bf16(300, static_cast<std::size_t>(tokens) * ld, 0), g_h = synth_bf16(301, static_cast<std::size_t>(tokens) * ldg, 1),
               w_h = synth_bf16(302, d, 0);
    auto x = device_buffer(device, x_h), g = device_buffer(device, g_h), w = device_buffer(device, w_h);
    RmsNormArgs a;
    a.x = a.out = x->address();
    a.ldx = a.ldo = ld;
    a.hsx = a.hso = d;
    a.gate = g->address() + 2ull * ld * 2;
    a.ldg = ldg;
    a.hsg = d;
    a.weight = w->address();
    a.tokens = tokens;
    a.heads = heads;
    a.d = d;
    a.gated = true;
    device.run([&](VkCommandBuffer cb) { k.record_rms_norm(cb, a); });
    std::vector<bits> g_shift(g_h.begin() + 2 * ld, g_h.end());
    g_shift.resize(g_h.size());
    const auto want = replay_norm(x_h, ld, d, w_h, &g_shift, ldg, d, nullptr, x_h, ld, d, tokens, heads, d);
    ok = report("gated rms_norm in place", read<bits>(*x, want.size()), widen(want), {1.0, 0.0}) && ok;
  }
  return ok;
}

// ---- conv, gates, gate_mul ----

bool check_conv_gates(Device& device, const TeKernels& k) {
  bool ok = true;
  const std::vector<int> lengths = {5, 70, 1};
  constexpr int ch = kGdnConvDim, ldx = ch + 4096, ldy = ch;
  int total = 0;
  for (const int l : lengths) total += l;
  const auto x_h = synth_bf16(400, static_cast<std::size_t>(total) * ldx, 1), w_h = synth_bf16(401, static_cast<std::size_t>(ch) * 4, -1);
  auto x = device_buffer(device, x_h), w = device_buffer(device, w_h),
       y = device_buffer(device, std::vector<bits>(static_cast<std::size_t>(total) * ldy));
  std::vector<double> want(static_cast<std::size_t>(total) * ldy);
  int s0 = 0;
  for (const int len : lengths) {
    const int start = s0;
    device.run([&](VkCommandBuffer cb) {
      k.record_conv_silu(cb, x->address() + 2ull * start * ldx, ldx, w->address(), y->address() + 2ull * start * ldy, ldy,
                         static_cast<std::uint32_t>(len));
    });
    for (int t = 0; t < len; ++t) {
      for (int c = 0; c < ch; ++c) {
        float acc = 0.0f;
        for (int kk = 0; kk < 4; ++kk) {
          const int src = t - 3 + kk;
          if (src < 0) continue;
          acc = std::fma(from_bf16(w_h[static_cast<std::size_t>(c) * 4 + kk]), from_bf16(x_h[static_cast<std::size_t>(start + src) * ldx + c]), acc);
        }
        want[static_cast<std::size_t>(start + t) * ldy + c] = from_bf16(to_bf16(silu(rbf(acc))));
      }
    }
    s0 += len;
  }
  ok = report("conv1d+silu, 3 sequences", read<bits>(*y, want.size()), want, {1.0, 0.0}) && ok;

  // Gates over [b | a] rows of stride 72.
  {
    constexpr int tokens = 45, ld = 72;
    const auto ba_h = synth_bf16(410, static_cast<std::size_t>(tokens) * ld, 3), alog_h = synth_bf16(411, 32, 1),
               dt_h = synth_bf16(412, 32, 2);
    auto ba = device_buffer(device, ba_h), alog = device_buffer(device, alog_h), dt = device_buffer(device, dt_h);
    Buffer g(device, tokens * 32 * 4), beta(device, tokens * 32 * 2);
    device.run([&](VkCommandBuffer cb) { k.record_gates(cb, ba->address(), ld, alog->address(), dt->address(), g.address(), beta.address(), tokens); });
    std::vector<float> g_want(tokens * 32);
    std::vector<double> beta_want(tokens * 32);
    for (int t = 0; t < tokens; ++t) {
      for (int h = 0; h < 32; ++h) {
        const float b = from_bf16(ba_h[static_cast<std::size_t>(t) * ld + h]), a = from_bf16(ba_h[static_cast<std::size_t>(t) * ld + 32 + h]);
        beta_want[static_cast<std::size_t>(t) * 32 + h] = rbf(sigmoid(b));
        const float xx = a + from_bf16(dt_h[static_cast<std::size_t>(h)]);
        const float sp = xx > 20.0f ? xx : std::log1p(std::exp(xx));
        g_want[static_cast<std::size_t>(t) * 32 + h] = -std::exp(from_bf16(alog_h[static_cast<std::size_t>(h)])) * sp;
      }
    }
    ok = report("gates beta", read<bits>(beta, beta_want.size()), beta_want, {1.0, 0.0}) && ok;
    ok = report_f32("gates g", read<float>(g, g_want.size()), g_want, 8.0) && ok;
  }

  // silu(gate) * up from one [gate | up] row, and the attention gate over 256-wide heads of a [q | gate] layout.
  {
    constexpr int rows = 9, cols = 520, ld = 2 * cols + 8;
    const auto gu_h = synth_bf16(420, static_cast<std::size_t>(rows) * ld, 1);
    auto gu = device_buffer(device, gu_h), out = device_buffer(device, std::vector<bits>(static_cast<std::size_t>(rows) * cols));
    device.run([&](VkCommandBuffer cb) {
      k.record_gate_mul(cb, TeKernels::GateMul::kSilu, gu->address(), ld, gu->address() + 2ull * cols, ld, out->address(), cols, rows,
                        cols, cols, 0);
    });
    std::vector<double> want(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const float g = from_bf16(gu_h[static_cast<std::size_t>(r) * ld + c]), u = from_bf16(gu_h[static_cast<std::size_t>(r) * ld + cols + c]);
        want[static_cast<std::size_t>(r) * cols + c] = rbf(rbf(silu(g)) * u);
      }
    }
    ok = report("silu_mul", read<bits>(*out, want.size()), want, {1.0, 0.0}) && ok;

    constexpr int heads = 3, hd = 256, ldq = 2 * heads * hd + 16, lda = heads * hd;
    const auto qg_h = synth_bf16(421, static_cast<std::size_t>(rows) * ldq, 2), a_h = synth_bf16(422, static_cast<std::size_t>(rows) * lda, 0);
    auto qg = device_buffer(device, qg_h), at = device_buffer(device, a_h);
    device.run([&](VkCommandBuffer cb) {
      k.record_gate_mul(cb, TeKernels::GateMul::kSigmoid, at->address(), lda, qg->address() + 2ull * hd, ldq, at->address(), lda, rows,
                        lda, hd, 2 * hd);
    });
    std::vector<double> want2(static_cast<std::size_t>(rows) * lda);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < lda; ++c) {
        const float g = from_bf16(qg_h[static_cast<std::size_t>(r) * ldq + (c / hd) * 2 * hd + hd + c % hd]);
        want2[static_cast<std::size_t>(r) * lda + c] = rbf(from_bf16(a_h[static_cast<std::size_t>(r) * lda + c]) * rbf(sigmoid(g)));
      }
    }
    ok = report("sigmoid_mul in place", read<bits>(*at, want2.size()), want2, {1.0, 0.0}) && ok;
  }
  return ok;
}

// ---- the gated delta rule ----

constexpr int kKeyDim = kGdnKeyHeads * kGdnHeadDim;

// The engine's chunked rule op by op (chunk_rule.cu): bf16 [T, 32, 128] out.
std::vector<double> replay_gdn(const std::vector<bits>& qkv, int ld, const std::vector<float>& g, const std::vector<bits>& beta,
                               const std::vector<int>& cu) {
  const int T = cu.back();
  constexpr int D = kGdnHeadDim, C = kGdnChunk;
  std::vector<float> qn(static_cast<std::size_t>(kGdnKeyHeads) * T * D), kn(qn.size());
  for (int t = 0; t < T; ++t) {
    for (int kh = 0; kh < kGdnKeyHeads; ++kh) {
      for (int which = 0; which < 2; ++which) {
        const bits* x = qkv.data() + static_cast<std::ptrdiff_t>(t) * ld + which * kKeyDim + kh * D;
        std::array<float, 32> lanes{};
        for (int l = 0; l < 32; ++l) {
          float s = 0.0f;
          for (int e = 0; e < 4; ++e) s = std::fma(from_bf16(x[4 * l + e]), from_bf16(x[4 * l + e]), s);
          lanes[static_cast<std::size_t>(l)] = s;
        }
        for (int off = 16; off > 0; off >>= 1) {
          for (int l = 0; l < off; ++l) lanes[static_cast<std::size_t>(l)] += lanes[static_cast<std::size_t>(l + off)];
        }
        const float inv = 1.0f / std::sqrt(lanes[0] + 1e-6f);
        float* dst = (which == 0 ? qn : kn).data() + (static_cast<std::size_t>(kh) * T + t) * D;
        for (int d = 0; d < D; ++d) dst[d] = which == 0 ? (from_bf16(x[d]) * inv) * 0.08838834764831845f : from_bf16(x[d]) * inv;
      }
    }
  }
  std::vector<double> out(static_cast<std::size_t>(T) * 32 * D);
  for (std::size_t s = 0; s + 1 < cu.size(); ++s) {
    const int start = cu[s], len = cu[s + 1] - cu[s];
    for (int h = 0; h < kGdnValueHeads; ++h) {
      const int kh = h / 2;
      const auto Q = [&](int t, int d) { return qn[(static_cast<std::size_t>(kh) * T + t) * D + d]; };
      const auto K = [&](int t, int d) { return kn[(static_cast<std::size_t>(kh) * T + t) * D + d]; };
      std::vector<float> S(static_cast<std::size_t>(D) * D, 0.0f);  // [k][v]
      for (int c0 = 0; c0 < len; c0 += C) {
        const int t0 = start + c0, n = std::min(C, len - c0);
        std::array<float, C> gc{}, bt{}, eg{}, el{};
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) {
          acc += i < n ? g[static_cast<std::size_t>(t0 + i) * 32 + h] : 0.0f;
          gc[static_cast<std::size_t>(i)] = acc;
          bt[static_cast<std::size_t>(i)] = i < n ? from_bf16(beta[static_cast<std::size_t>(t0 + i) * 32 + h]) : 0.0f;
        }
        for (int i = 0; i < C; ++i) {
          eg[static_cast<std::size_t>(i)] = std::exp(gc[static_cast<std::size_t>(i)]);
          el[static_cast<std::size_t>(i)] = std::exp(gc[static_cast<std::size_t>(n - 1)] - gc[static_cast<std::size_t>(i)]);
        }
        std::vector<float> a(C * C, 0.0f), attn(C * C, 0.0f);
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j <= i; ++j) {
            float kk = 0.0f, qk = 0.0f;
            for (int d = 0; d < D; ++d) {
              kk = std::fma(K(t0 + i, d), K(t0 + j, d), kk);
              qk = std::fma(Q(t0 + i, d), K(t0 + j, d), qk);
            }
            const float decay = std::exp(gc[static_cast<std::size_t>(i)] - gc[static_cast<std::size_t>(j)]);
            if (j < i) a[static_cast<std::size_t>(i) * C + j] = (kk * bt[static_cast<std::size_t>(i)]) * decay;
            attn[static_cast<std::size_t>(i) * C + j] = qk * decay;
          }
        }
        std::vector<float> u(static_cast<std::size_t>(C) * D, 0.0f), w(u.size(), 0.0f);
        for (int col = 0; col < D; ++col) {
          for (int i = 0; i < n; ++i) {
            const float b = bt[static_cast<std::size_t>(i)];
            float xu = from_bf16(qkv[static_cast<std::size_t>(t0 + i) * ld + 2 * kKeyDim + h * D + col]) * b;
            float xw = (K(t0 + i, col) * b) * eg[static_cast<std::size_t>(i)];
            for (int j = 0; j < i; ++j) {
              xu = std::fma(-a[static_cast<std::size_t>(i) * C + j], u[static_cast<std::size_t>(j) * D + col], xu);
              xw = std::fma(-a[static_cast<std::size_t>(i) * C + j], w[static_cast<std::size_t>(j) * D + col], xw);
            }
            u[static_cast<std::size_t>(i) * D + col] = xu;
            w[static_cast<std::size_t>(i) * D + col] = xw;
          }
        }
        std::vector<float> vn(static_cast<std::size_t>(C) * D, 0.0f);
        for (int i = 0; i < n; ++i) {
          for (int c = 0; c < D; ++c) {
            float pw = 0.0f, pq = 0.0f;
            for (int d = 0; d < D; ++d) {
              const float sv = S[static_cast<std::size_t>(d) * D + c];
              pw = std::fma(w[static_cast<std::size_t>(i) * D + d], sv, pw);
              pq = std::fma(Q(t0 + i, d) * eg[static_cast<std::size_t>(i)], sv, pq);
            }
            vn[static_cast<std::size_t>(i) * D + c] = u[static_cast<std::size_t>(i) * D + c] - pw;
            u[static_cast<std::size_t>(i) * D + c] = pq;  // reused: q S
          }
        }
        for (int i = 0; i < n; ++i) {
          for (int c = 0; c < D; ++c) {
            float o = u[static_cast<std::size_t>(i) * D + c];
            for (int j = 0; j <= i; ++j) o = std::fma(attn[static_cast<std::size_t>(i) * C + j], vn[static_cast<std::size_t>(j) * D + c], o);
            out[(static_cast<std::size_t>(t0 + i) * 32 + h) * D + c] = o;
          }
        }
        const float cd = std::exp(gc[static_cast<std::size_t>(n - 1)]);
        for (int d = 0; d < D; ++d) {
          for (int c = 0; c < D; ++c) {
            float upd = 0.0f;
            for (int j = 0; j < n; ++j) upd = std::fma(K(t0 + j, d) * el[static_cast<std::size_t>(j)], vn[static_cast<std::size_t>(j) * D + c], upd);
            S[static_cast<std::size_t>(d) * D + c] = std::fma(S[static_cast<std::size_t>(d) * D + c], cd, upd);
          }
        }
      }
    }
  }
  return out;
}

// torch_recurrent_gated_delta_rule in fp64: the algorithm the chunked form rearranges.
std::vector<double> recurrent_gdn(const std::vector<bits>& qkv, int ld, const std::vector<float>& g, const std::vector<bits>& beta,
                                  const std::vector<int>& cu) {
  constexpr int D = kGdnHeadDim;
  std::vector<double> out(static_cast<std::size_t>(cu.back()) * 32 * D);
  for (std::size_t s = 0; s + 1 < cu.size(); ++s) {
    for (int h = 0; h < kGdnValueHeads; ++h) {
      const int kh = h / 2;
      std::vector<double> S(static_cast<std::size_t>(D) * D, 0.0);
      for (int t = cu[s]; t < cu[s + 1]; ++t) {
        const bits* row = qkv.data() + static_cast<std::ptrdiff_t>(t) * ld;
        std::array<double, D> q{}, k{}, v{};
        double sq = 0, sk = 0;
        for (int d = 0; d < D; ++d) {
          q[static_cast<std::size_t>(d)] = from_bf16(row[kh * D + d]);
          k[static_cast<std::size_t>(d)] = from_bf16(row[kKeyDim + kh * D + d]);
          v[static_cast<std::size_t>(d)] = from_bf16(row[2 * kKeyDim + h * D + d]);
          sq += q[static_cast<std::size_t>(d)] * q[static_cast<std::size_t>(d)];
          sk += k[static_cast<std::size_t>(d)] * k[static_cast<std::size_t>(d)];
        }
        for (int d = 0; d < D; ++d) {
          q[static_cast<std::size_t>(d)] /= std::sqrt(sq + 1e-6) * std::sqrt(static_cast<double>(D));
          k[static_cast<std::size_t>(d)] /= std::sqrt(sk + 1e-6);
        }
        const double decay = std::exp(static_cast<double>(g[static_cast<std::size_t>(t) * 32 + h]));
        const double b = from_bf16(beta[static_cast<std::size_t>(t) * 32 + h]);
        for (double& e : S) e *= decay;
        for (int c = 0; c < D; ++c) {
          double kv = 0;
          for (int d = 0; d < D; ++d) kv += S[static_cast<std::size_t>(d) * D + c] * k[static_cast<std::size_t>(d)];
          const double delta = (v[static_cast<std::size_t>(c)] - kv) * b;
          double o = 0;
          for (int d = 0; d < D; ++d) {
            S[static_cast<std::size_t>(d) * D + c] += k[static_cast<std::size_t>(d)] * delta;
            o += S[static_cast<std::size_t>(d) * D + c] * q[static_cast<std::size_t>(d)];
          }
          out[(static_cast<std::size_t>(t) * 32 + h) * D + c] = o;
        }
      }
    }
  }
  return out;
}

bool check_gdn(Device& device, const TeKernels& k) {
  const std::vector<int> lengths = {1, 5, 64, 130};
  std::vector<int> cu = {0};
  for (const int l : lengths) cu.push_back(cu.back() + l);
  const int T = cu.back();
  constexpr int ld = kGdnConvDim + 8;
  // Conv outputs are SiLU outputs: mostly small positive values with some negative ones.
  std::vector<bits> qkv_h = synth_bf16(500, static_cast<std::size_t>(T) * ld, 0);
  for (bits& b : qkv_h) b = to_bf16(silu(2.0f * from_bf16(b)));
  std::vector<float> g_h(static_cast<std::size_t>(T) * 32);
  std::vector<bits> beta_h(g_h.size());
  const auto gs = synth_bf16(501, g_h.size(), 1), bs = synth_bf16(502, g_h.size(), 1);
  for (std::size_t i = 0; i < g_h.size(); ++i) {
    g_h[i] = -0.6f * std::abs(from_bf16(gs[i]));  // decays exp(g) in (0.3, 1]
    beta_h[i] = to_bf16(sigmoid(from_bf16(bs[i])));
  }
  auto qkv = device_buffer(device, qkv_h), g = device_buffer(device, g_h), beta = device_buffer(device, beta_h);
  Buffer out(device, static_cast<std::size_t>(T) * 32 * kGdnHeadDim * 2), ws(device, GdnWorkspace::bytes(static_cast<std::uint32_t>(T)));
  GdnArgs a;
  a.qkv = qkv->address();
  a.ldqkv = ld;
  a.g = g->address();
  a.beta = beta->address();
  a.out = out.address();
  a.ws = GdnWorkspace::at(ws.address(), static_cast<std::uint32_t>(T));
  a.cu.assign(cu.begin(), cu.end());
  device.run([&](VkCommandBuffer cb) { k.record_gated_delta_rule(cb, a); });
  const auto got = read<bits>(out, static_cast<std::size_t>(T) * 32 * kGdnHeadDim);
  const auto want = replay_gdn(qkv_h, ld, g_h, beta_h, cu);
  bool ok = report("gated delta rule, 4 seqs", got, want, {2.0, 1.0 / 64.0, 98.0});

  // The algorithm: per (token, head) row, relative L2 against the fp64 recurrence.
  const auto exact = recurrent_gdn(qkv_h, ld, g_h, beta_h, cu);
  double worst = 0, worst_replay = 0;
  for (std::size_t r = 0; r < exact.size() / kGdnHeadDim; ++r) {
    double diff = 0, diff_replay = 0, ref = 0;
    for (int c = 0; c < kGdnHeadDim; ++c) {
      const std::size_t i = r * kGdnHeadDim + c;
      diff += std::pow(from_bf16(got[i]) - exact[i], 2);
      diff_replay += std::pow(want[i] - exact[i], 2);
      ref += exact[i] * exact[i];
    }
    worst = std::max(worst, std::sqrt(diff / ref));
    worst_replay = std::max(worst_replay, std::sqrt(diff_replay / ref));
  }
  const bool alg_ok = worst < 1e-2;
  std::printf("  %-26s %s (worst row rel-L2 %.3e vs fp64 recurrence, limit 1e-2; fp32 replay before bf16 rounding %.3e)\n",
              "gated delta rule algorithm", alg_ok ? "ok" : "FAILED", worst, worst_replay);
  return ok && alg_ok;
}

// ---- attention ----

std::vector<float> replay_attention_row(const bits* q, const bits* k, int ldk, const bits* v, int ldv, int seqlen_k, float scale) {
  constexpr int D = 256, kStep = 32;
  const float log2_scale = static_cast<float>(static_cast<double>(scale) * 1.4426950408889634074);
  std::vector<float> acc(D, 0.0f);
  std::array<float, kStep> s{}, p{};
  float row_max = 0, row_sum = 0;
  const int steps = (seqlen_k + kStep - 1) / kStep;
  for (int step = steps - 1; step >= 0; --step) {
    const bool first = step == steps - 1;
    const int n0 = step * kStep, cols = std::min(kStep, seqlen_k - n0);
    for (int c = 0; c < cols; ++c) {
      const bits* key = k + static_cast<std::ptrdiff_t>(n0 + c) * ldk;
      float sum = 0;
      for (int d = 0; d < D; ++d) {
        const float product = from_bf16(q[d]) * from_bf16(key[d]);
        sum += product;
      }
      s[static_cast<std::size_t>(c)] = sum;
    }
    const float prev = row_max;
    float cur = first ? s[0] : std::fmax(prev, s[0]);
    for (int c = 1; c < cols; ++c) cur = std::fmax(cur, s[static_cast<std::size_t>(c)]);
    row_max = cur;
    float sum = 0;
    if (!first) {
      const float rescale = std::exp2((prev - cur) * log2_scale);
      sum = row_sum * rescale;
      for (float& e : acc) e *= rescale;
    }
    const float max_scaled = cur * log2_scale;
    for (int c = 0; c < cols; ++c) {
      const float w = std::exp2(std::fma(s[static_cast<std::size_t>(c)], log2_scale, -max_scaled));
      sum += w;
      p[static_cast<std::size_t>(c)] = rbf(w);
    }
    row_sum = sum;
    for (int c = 0; c < cols; ++c) {
      const bits* val = v + static_cast<std::ptrdiff_t>(n0 + c) * ldv;
      for (int d = 0; d < D; ++d) {
        const float product = p[static_cast<std::size_t>(c)] * from_bf16(val[d]);
        acc[static_cast<std::size_t>(d)] += product;
      }
    }
  }
  const float inv_sum = row_sum == 0.0f || row_sum != row_sum ? 1.0f : 1.0f / row_sum;
  for (float& e : acc) e *= inv_sum;
  return acc;
}

bool check_attention(Device& device, const TeKernels& k) {
  constexpr int heads = 4, kv_heads = 2, D = 256, ldq = heads * D + 8, ldkv = 2 * kv_heads * D + 16, ldo = heads * D;
  const std::vector<int> lengths = {37, 70, 1};
  std::vector<int> cu = {0};
  for (const int l : lengths) cu.push_back(cu.back() + l);
  const auto T = static_cast<std::size_t>(cu.back());
  const auto q_h = synth_bf16(600, T * ldq, 1), kv_h = synth_bf16(601, T * ldkv, 1);
  auto q = device_buffer(device, q_h), kv = device_buffer(device, kv_h), o = device_buffer(device, std::vector<bits>(T * ldo));
  std::vector<double> want(T * ldo);
  for (std::size_t s = 0; s < lengths.size(); ++s) {
    const int r0 = cu[s], len = lengths[s];
    TeAttentionArgs a;
    a.q = q->address() + 2ull * r0 * ldq;
    a.k = kv->address() + 2ull * r0 * ldkv;
    a.v = a.k + 2ull * kv_heads * D;
    a.o = o->address() + 2ull * r0 * ldo;
    a.ldq = ldq;
    a.ldk = a.ldv = ldkv;
    a.ldo = ldo;
    a.hsq = a.hsk = a.hsv = a.hso = D;
    a.heads = heads;
    a.group = heads / kv_heads;
    // Two pieces of rows, as the encoder splits long sequences.
    const std::uint32_t split = static_cast<std::uint32_t>(len) / 2;
    device.run([&](VkCommandBuffer cb) {
      if (split > 0) {
        a.q_first = 0;
        a.rows = split;
        k.record_attention(cb, a);
      }
      a.q_first = split;
      a.rows = static_cast<std::uint32_t>(len) - split;
      k.record_attention(cb, a);
    });
    for (int h = 0; h < heads; ++h) {
      const int kvh = h / (heads / kv_heads);
      for (int r = 0; r < len; ++r) {
        const auto row = replay_attention_row(q_h.data() + static_cast<std::size_t>(r0 + r) * ldq + h * D,
                                              kv_h.data() + static_cast<std::size_t>(r0) * ldkv + kvh * D, ldkv,
                                              kv_h.data() + static_cast<std::size_t>(r0) * ldkv + (kv_heads + kvh) * D, ldkv, r + 1, 0.0625f);
        std::copy(row.begin(), row.end(), want.begin() + static_cast<std::ptrdiff_t>((r0 + r) * ldo + h * D));
      }
    }
  }
  return report("causal GQA attention", read<bits>(*o, want.size()), want, {1.0, 1.0});
}

}  // namespace

int main(int argc, char** argv) {
  int device_index = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) {
      device_index = std::atoi(argv[++i]);
    } else {
      std::printf("usage: %s [--device N]\n", argv[0]);
      return 2;
    }
  }
  try {
    Device device(device_index);
    std::printf("device %u: %s\ntext encoder kernels against host replays:\n", device.info().index, device.info().name.c_str());
    const TeKernels kernels(device);
    bool ok = check_norms(device, kernels);
    ok = check_conv_gates(device, kernels) && ok;
    ok = check_attention(device, kernels) && ok;
    ok = check_gdn(device, kernels) && ok;
    std::printf("%s\n", ok ? "all passed" : "FAILED");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
