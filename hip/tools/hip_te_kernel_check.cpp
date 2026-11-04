// te_kernels.h's ops against host replays of their op order, on small shapes with tails, padded strides and several
// sequences, edge inputs included:
//   - the RoPE table at every position the encoder allows, against the host's cos and sin (>= 99.99 % of the bits, within
//     1 ULP).
//   - rms_norm (plain at 1 to 17 rows, where torch's threads per row change; with the RoPE on q and k head views; gated
//     in place), conv1d+SiLU over sequences of 1 to 70 tokens, the gates (softplus past its threshold, exp underflow),
//     silu_mul, causal GQA attention with and without the output gate: bf16 outputs at >= 99 % the replay's bits and
//     within 1 ULP of max(|replay|, floor). The replays use the host's exp, rsqrt and division, which may differ from the
//     device's in the last bit of fp32, which a rounding point can carry into a bf16 output.
//   - the chunked gated delta rule over sequences of 1, 5, 64, 65, 130 and 300 tokens (chunk tails and several chunks),
//     and over 70 and 300 tokens with decays down to e^-20 per token (a chunk's cumulative decay underflows exp): the
//     replay of its fp32 algorithm (>= 98 % same bits, 2 ULPs of max(|replay|, rms / 64)), and the fp64 token-by-token
//     recurrence it rearranges (worst row relative L2 at most 4e-3 for the bf16 output, 1e-4 for the fp32 replay).
//   - the packed GEMMs on small integers, whose sums are exact in any order: bit for bit against the host and against
//     hip_block's row-major GEMMs; and on order-sensitive values at the encoder's shapes, bit for bit against the
//     row-major GEMMs (the same leaf, the same K order).
// Usage: hip_te_kernel_check [--device N]. Exit status: 0 passed, 1 failed, 2 no usable device.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "half.h"
#include "hip_block.h"
#include "runtime.h"
#include "te_kernels.h"

namespace {

using namespace mi::hip;
namespace te = hip_block::te;
using bits = std::uint16_t;
using std::size_t;

float fb(bits b) { return bf16_to_float(b); }
bits tb(float v) { return float_to_bf16(v); }
float rbf(float v) { return fb(tb(v)); }
float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
size_t z(std::int64_t v) { return static_cast<size_t>(v); }

struct Rng {
  std::uint64_t s;
  std::uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    std::uint64_t r = s;
    r = (r ^ (r >> 30)) * 0xBF58476D1CE4E5B9ull;
    r = (r ^ (r >> 27)) * 0x94D049BB133111EBull;
    return r ^ (r >> 31);
  }
  int integer(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo + 1)); }
  // A bf16 value of magnitude about 2^e for e in [lo, hi], either sign.
  float spread(int lo, int hi) {
    const std::uint64_t r = next();
    const float m = 1.0f + static_cast<float>(r & 0x7F) / 128.0f;
    return rbf(std::ldexp((r >> 7) & 1 ? -m : m, lo + static_cast<int>((r >> 8) % static_cast<std::uint64_t>(hi - lo + 1))));
  }
  float uniform() { return static_cast<float>(next() >> 40) / static_cast<float>(1u << 24); }
};

std::vector<bits> spread_bits(Rng& rng, size_t n, int lo, int hi) {
  std::vector<bits> v(n);
  for (bits& b : v) b = tb(rng.spread(lo, hi));
  return v;
}

// A pageable hipMemcpy to the device may return before its DMA lands, and the kernels run on a non-blocking stream:
// every upload waits for the device.
template <typename T>
DeviceBuffer upload(const std::vector<T>& v) {
  DeviceBuffer b(std::max<size_t>(v.size(), 1) * sizeof(T));
  if (!v.empty()) b.upload(v.data(), v.size() * sizeof(T));
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");
  return b;
}

template <typename T>
std::vector<T> download(const DeviceBuffer& b, size_t n) {
  std::vector<T> out(n);
  b.download(out.data(), n * sizeof(T));
  return out;
}

int failures = 0;

struct Limit {
  double ulps = 1.0;
  double floor_of_rms = 0.0;
  double min_same_percent = 99.0;
};

// bf16 outputs against replayed fp32 values: the share with the replay's bits (rounded) and the largest distance in bf16
// ULPs of max(|replay|, floor). Positions where `mask` is 0 (a stride's padding) must keep their bits.
void report(const std::string& name, const std::vector<bits>& got, const std::vector<float>& want, Limit limit,
            const std::vector<std::uint8_t>& mask = {}) {
  double sq = 0;
  size_t counted = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (!mask.empty() && !mask[i]) continue;
    sq += static_cast<double>(want[i]) * want[i];
    ++counted;
  }
  const double floor_mag = std::sqrt(sq / static_cast<double>(std::max<size_t>(counted, 1))) * limit.floor_of_rms;
  size_t same = 0, nans = 0, padding = 0;
  double worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!mask.empty() && !mask[i]) {
      padding += got[i] != tb(want[i]);
      continue;
    }
    const double g = fb(got[i]);
    if (std::isnan(g) != std::isnan(want[i])) ++nans;
    if (got[i] == tb(want[i])) {
      ++same;
      continue;
    }
    int e = 0;
    (void)std::frexp(std::max(std::abs(static_cast<double>(want[i])), floor_mag), &e);
    worst = std::max(worst, std::abs(g - want[i]) / std::ldexp(1.0, e - 8));
  }
  const double same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(std::max<size_t>(counted, 1));
  const bool ok = nans == 0 && padding == 0 && worst <= limit.ulps && same_pct >= limit.min_same_percent;
  std::printf("  %-44s %s (%zu values, %.3f %% same, limit %.2f %%; worst %.2f ULPs, limit %.1f; %zu NaN mismatches%s)\n",
              name.c_str(), ok ? "ok" : "FAIL", counted, same_pct, limit.min_same_percent, worst, limit.ulps, nans,
              padding != 0 ? ", padding overwritten" : "");
  failures += ok ? 0 : 1;
}

void report_f32(const std::string& name, const std::vector<float>& got, const std::vector<float>& want, double max_ulps) {
  double worst = 0;
  size_t same = 0, nans = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(got[i]) != std::isnan(want[i])) ++nans;
    same += std::bit_cast<std::uint32_t>(got[i]) == std::bit_cast<std::uint32_t>(want[i]);
    if (got[i] == want[i]) continue;
    int e = 0;
    (void)std::frexp(std::abs(static_cast<double>(want[i])), &e);
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - want[i]) / std::ldexp(1.0, e - 24));
  }
  const bool ok = nans == 0 && worst <= max_ulps;
  std::printf("  %-44s %s (%zu values, %.3f %% same; worst %.2f fp32 ULPs, limit %.1f; %zu NaN mismatches)\n", name.c_str(),
              ok ? "ok" : "FAIL", got.size(), 100.0 * static_cast<double>(same) / static_cast<double>(got.size()), worst,
              max_ulps, nans);
  failures += ok ? 0 : 1;
}

void report_exact(const std::string& name, const std::vector<bits>& got, const std::vector<bits>& want) {
  size_t differ = 0;
  for (size_t i = 0; i < got.size(); ++i) differ += got[i] != want[i];
  std::printf("  %-44s %s (%zu values, %zu differ, exact)\n", name.c_str(), differ == 0 ? "ok" : "FAIL", got.size(), differ);
  failures += differ == 0 ? 0 : 1;
}

std::vector<float> widen(const std::vector<bits>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = fb(v[i]);
  return out;
}

// ---- norms ----

// torch's reduce order for bw threads per row (te_rms_norm_kernel), then rsqrt(mean + eps) on the host.
float host_inv_rms(const bits* row, int d, int bw, float factor) {
  std::vector<float> red(static_cast<size_t>(bw));
  for (int tx = 0; tx < bw; ++tx) {
    std::array<float, 4> acc{};
    for (int idx = tx; idx * 4 + 3 < d; idx += bw) {
      for (int i = 0; i < 4; ++i) {
        const float v = fb(row[idx * 4 + i]);
        acc[z(i)] = acc[z(i)] + v * v;
      }
    }
    red[z(tx)] = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  }
  for (int off = bw / 2; off >= 1; off >>= 1) {
    for (int tx = 0; tx < off; ++tx) red[z(tx)] = red[z(tx)] + red[z(tx + off)];
  }
  return 1.0f / std::sqrt(red[0] * factor + 1e-6f);
}

// Replays te::rms_norm with its operands at x (and gate, rope when given) into want at out, marking those outputs.
void replay_norm(const te::NormArgs& a, const bits* x, const bits* weight, const bits* gate, const bits* rope, float* want,
                 std::uint8_t* mask) {
  const std::int64_t rows = a.tokens * a.heads;
  const int bw = te::reduce_threads(rows, a.d);
  const float factor = static_cast<float>(rows) / static_cast<float>(rows * a.d);
  for (std::int64_t t = 0; t < a.tokens; ++t) {
    for (int hd = 0; hd < a.heads; ++hd) {
      const bits* xr = x + t * a.ldx + hd * a.hsx;
      const float r = host_inv_rms(xr, a.d, bw, factor);
      std::vector<float> y(z(a.d));
      for (int c = 0; c < a.d; ++c) {
        const float xn = fb(xr[c]) * r;
        if (gate == nullptr) {
          y[z(c)] = xn * (1.0f + fb(weight[c]));
        } else {
          y[z(c)] = rbf(fb(weight[c]) * rbf(xn)) * silu(fb(gate[t * a.ldg + hd * a.hsg + c]));
        }
      }
      if (rope != nullptr) {
        const bits* tab = rope + t * a.rope_ld;
        for (int c = 0; c < 32; ++c) {
          const float lo = rbf(y[z(c)]), hi = rbf(y[z(c + 32)]), co = fb(tab[c]), si = fb(tab[32 + c]);
          y[z(c)] = rbf(lo * co) + rbf(-hi * si);
          y[z(c + 32)] = rbf(hi * co) + rbf(lo * si);
        }
      }
      for (int c = 0; c < a.d; ++c) {
        const std::int64_t o = t * a.ldo + hd * a.hso + c;
        want[o] = y[z(c)];
        mask[o] = 1;
      }
    }
  }
}

void check_norms(Rng& rng, hipStream_t stream) {
  // The hidden norm at row counts where torch's threads per row are 512, 256, 128, 64 and 32; a zero row and a row of
  // large values among them.
  for (const int tokens : {1, 2, 5, 10, 17}) {
    constexpr int d = 4096, ld = d + 8;
    std::vector<bits> x = spread_bits(rng, z(tokens) * ld, -6, 3), w = spread_bits(rng, d, -4, 0);
    if (tokens >= 5) {
      for (int c = 0; c < d; ++c) x[z(1 * ld + c)] = 0;
      for (int c = 0; c < d; ++c) x[z(3 * ld + c)] = tb(rng.spread(20, 28));
    }
    std::vector<bits> o0 = spread_bits(rng, x.size(), 0, 0);
    DeviceBuffer xd = upload(x), wd = upload(w), od = upload(o0);
    te::NormArgs a;
    a.x = xd.as<bits>();
    a.ldx = a.ldo = ld;
    a.out = od.as<bits>();
    a.weight = wd.as<bits>();
    a.tokens = tokens;
    a.d = d;
    te::rms_norm(a, stream);
    check(hipStreamSynchronize(stream), "sync");
    std::vector<float> want = widen(o0);
    std::vector<std::uint8_t> mask(want.size(), 0);
    replay_norm(a, x.data(), w.data(), nullptr, nullptr, want.data(), mask.data());
    report("rms_norm d 4096, " + std::to_string(tokens) + " rows (bw " + std::to_string(te::reduce_threads(tokens, d)) + ")",
           download<bits>(od, want.size()), want, {1.0, 0.0}, mask);
  }

  // q from the q|gate head view into its own rows, with the RoPE; k in place among q|gate|k|v rows, with the RoPE. 1 and 3
  // tokens make k's 4 heads 4 and 12 rows (64 threads per row).
  for (const int tokens : {1, 3, 19}) {
    constexpr int heads = 16, kv_heads = 4, d = 256;
    constexpr int ld = 2 * heads * d + 2 * kv_heads * d + 16;  // [q|gate per head | k | v | 16 padding]
    std::vector<bits> qkv = spread_bits(rng, z(tokens) * ld, -4, 4), qw = spread_bits(rng, d, -3, 0), kw = spread_bits(rng, d, -3, 0);
    std::vector<bits> rope(z(tokens) * 64);
    for (bits& b : rope) b = tb(2.0f * rng.uniform() - 1.0f);
    std::vector<bits> q0 = spread_bits(rng, z(tokens) * heads * d, 0, 0);
    DeviceBuffer qkvd = upload(qkv), qwd = upload(qw), kwd = upload(kw), roped = upload(rope), qd = upload(q0);
    te::NormArgs qn;
    qn.x = qkvd.as<bits>();
    qn.ldx = ld;
    qn.hsx = 2 * d;
    qn.out = qd.as<bits>();
    qn.ldo = heads * d;
    qn.hso = d;
    qn.weight = qwd.as<bits>();
    qn.rope = roped.as<bits>();
    qn.rope_ld = 64;
    qn.tokens = tokens;
    qn.heads = heads;
    qn.d = d;
    te::NormArgs kn = qn;
    kn.x = kn.out = qkvd.as<bits>() + 2 * heads * d;
    kn.ldx = kn.ldo = ld;
    kn.hsx = kn.hso = d;
    kn.weight = kwd.as<bits>();
    kn.heads = kv_heads;
    te::rms_norm(qn, stream);
    te::rms_norm(kn, stream);
    check(hipStreamSynchronize(stream), "sync");
    std::vector<float> q_want = widen(q0), k_want = widen(qkv);
    std::vector<std::uint8_t> q_mask(q_want.size(), 0);
    replay_norm(qn, qkv.data(), qw.data(), nullptr, rope.data(), q_want.data(), q_mask.data());
    // The k rows are checked; the q|gate and v columns around them must keep their bits.
    std::vector<std::uint8_t> k_rows(k_want.size(), 0);
    replay_norm(kn, qkv.data() + 2 * heads * d, kw.data(), nullptr, rope.data(), k_want.data() + 2 * heads * d,
                k_rows.data() + 2 * heads * d);
    report("rms_norm+rope q heads, " + std::to_string(tokens) + " tokens", download<bits>(qd, q_want.size()), q_want, {1.0, 0.0},
           q_mask);
    report("rms_norm+rope k heads in place, " + std::to_string(tokens) + " tokens", download<bits>(qkvd, k_want.size()), k_want,
           {1.0, 0.0}, k_rows);
  }

  // The gated norm in place over 32 heads of 128, the gate from another buffer's rows.
  for (const int tokens : {1, 3, 20}) {
    constexpr int heads = 32, d = 128, ld = heads * d, ldg = 12288;
    std::vector<bits> x = spread_bits(rng, z(tokens) * ld, -5, 2), w = spread_bits(rng, d, -2, 1);
    std::vector<bits> gate = spread_bits(rng, z(tokens) * ldg, -4, 3);
    DeviceBuffer xd = upload(x), wd = upload(w), gd = upload(gate);
    te::NormArgs a;
    a.x = a.out = xd.as<bits>();
    a.ldx = a.ldo = ld;
    a.hsx = a.hso = d;
    a.gate = gd.as<bits>() + 8192;
    a.ldg = ldg;
    a.hsg = d;
    a.weight = wd.as<bits>();
    a.tokens = tokens;
    a.heads = heads;
    a.d = d;
    te::rms_norm(a, stream);
    check(hipStreamSynchronize(stream), "sync");
    std::vector<float> want = widen(x);
    std::vector<std::uint8_t> mask(want.size(), 0);
    replay_norm(a, x.data(), w.data(), gate.data() + 8192, nullptr, want.data(), mask.data());
    report("gated rms_norm in place, " + std::to_string(tokens) + " tokens", download<bits>(xd, want.size()), want, {1.0, 0.0}, mask);
  }
}

// ---- the RoPE table ----

// Every position the encoder allows, against the host's cos and sin of the same fp32 phase.
void check_rope_table(hipStream_t stream) {
  static const std::uint32_t kInv[32] = {
      0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
      0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
      0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
      0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44};
  constexpr int rows = 65535;
  DeviceBuffer out(z(rows) * 64 * 2);
  te::rope_table(out.as<bits>(), rows, stream);
  check(hipStreamSynchronize(stream), "sync");
  std::vector<float> want(z(rows) * 64);
  for (int p = 0; p < rows; ++p) {
    for (int i = 0; i < 32; ++i) {
      float inv = 0;
      std::memcpy(&inv, &kInv[i], 4);
      const float phase = static_cast<float>(p) * inv;
      want[z(p) * 64 + z(i)] = std::cos(phase);
      want[z(p) * 64 + 32 + z(i)] = std::sin(phase);
    }
  }
  report("rope_table, 65535 positions", download<bits>(out, want.size()), want, {1.0, 0.0, 99.99});
}

// ---- conv, gates, silu_mul ----

void check_conv_gates(Rng& rng, hipStream_t stream) {
  {
    const std::vector<int> lengths = {5, 70, 1, 2};
    std::vector<std::int32_t> cu = {0};
    for (const int l : lengths) cu.push_back(cu.back() + l);
    const int T = cu.back();
    constexpr int C = te::kGdnConvDim, ldx = 12288, ldy = C + 16;
    std::vector<bits> x = spread_bits(rng, z(T) * ldx, -4, 2), w = spread_bits(rng, z(C) * 4, -3, 0);
    std::vector<bits> y0 = spread_bits(rng, z(T) * ldy, 0, 0);
    DeviceBuffer xd = upload(x), wd = upload(w), yd = upload(y0), cud = upload(cu);
    te::conv_silu(xd.as<bits>(), ldx, wd.as<bits>(), yd.as<bits>(), ldy, cud.as<std::int32_t>(), static_cast<int>(lengths.size()),
                  T, stream);
    check(hipStreamSynchronize(stream), "sync");
    std::vector<float> want = widen(y0);
    std::vector<std::uint8_t> mask(want.size(), 0);
    for (size_t s = 0; s < lengths.size(); ++s) {
      for (int t = cu[s]; t < cu[s + 1]; ++t) {
        for (int c = 0; c < C; ++c) {
          float acc = 0.0f;
          for (int k = 0; k < 4; ++k) {
            const int src = t - 3 + k;
            if (src < cu[s]) continue;
            acc = std::fma(fb(w[z(c * 4 + k)]), fb(x[z(src * ldx + c)]), acc);
          }
          want[z(t * ldy + c)] = silu(rbf(acc));
          mask[z(t * ldy + c)] = 1;
        }
      }
    }
    report("conv1d+silu, sequences of 5, 70, 1, 2", download<bits>(yd, want.size()), want, {1.0, 0.0}, mask);
  }
  {
    constexpr int T = 40, ldba = 72;
    std::vector<bits> ba = spread_bits(rng, z(T) * ldba, -3, 4), a_log = spread_bits(rng, 32, -3, 1), dt = spread_bits(rng, 32, -2, 2);
    // Edges: a + dt_bias past softplus's threshold 20, far below zero (exp underflows), large |b|.
    for (int t = 0; t < 4; ++t) {
      ba[z(t * ldba + 32 + t)] = tb(30.0f);
      ba[z(t * ldba + 33 + t)] = tb(-120.0f);
      ba[z(t * ldba + t)] = tb(t % 2 ? 90.0f : -90.0f);
    }
    ba[z(5 * ldba + 32)] = tb(20.0f - fb(dt[0]));  // at the threshold
    DeviceBuffer bad = upload(ba), ald = upload(a_log), dtd = upload(dt), gd(z(T) * 32 * 4), betad(z(T) * 32 * 2);
    te::gates(bad.as<bits>(), ldba, ald.as<bits>(), dtd.as<bits>(), gd.as<float>(), betad.as<bits>(), T, stream);
    check(hipStreamSynchronize(stream), "sync");
    std::vector<float> g_want(z(T) * 32), beta_want(z(T) * 32);
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < 32; ++h) {
        beta_want[z(t * 32 + h)] = sigmoid(fb(ba[z(t * ldba + h)]));
        const float xv = fb(ba[z(t * ldba + 32 + h)]) + fb(dt[z(h)]);
        const float sp = xv > 20.0f ? xv : std::log1p(std::exp(xv));
        g_want[z(t * 32 + h)] = -std::exp(fb(a_log[z(h)])) * sp;
      }
    }
    report("gates beta", download<bits>(betad, beta_want.size()), beta_want, {1.0, 0.0});
    report_f32("gates g", download<float>(gd, g_want.size()), g_want, 32.0);
  }
  {
    constexpr int rows = 7, cols = 520, ldg = 2 * cols + 8, ldo = cols + 24;
    std::vector<bits> gu = spread_bits(rng, z(rows) * ldg, -4, 4);
    gu[0] = tb(-100.0f);
    gu[1] = tb(100.0f);
    std::vector<bits> o0 = spread_bits(rng, z(rows) * ldo, 0, 0);
    DeviceBuffer gd = upload(gu), od = upload(o0);
    te::silu_mul(gd.as<bits>(), ldg, od.as<bits>(), ldo, rows, cols, stream);
    check(hipStreamSynchronize(stream), "sync");
    std::vector<float> want = widen(o0);
    std::vector<std::uint8_t> mask(want.size(), 0);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        want[z(r * ldo + c)] = rbf(silu(fb(gu[z(r * ldg + c)]))) * fb(gu[z(r * ldg + cols + c)]);
        mask[z(r * ldo + c)] = 1;
      }
    }
    report("silu_mul, 520 columns", download<bits>(od, want.size()), want, {1.0, 0.0}, mask);
  }
}

// ---- the gated delta rule ----

constexpr int D = te::kGdnHeadDim, C = te::kGdnChunk, kKeyDim = te::kGdnKeyDim, NV = te::kGdnValueHeads;

// The device's algorithm (te_kernels.hip) in host fp32: the same products and their order, the host's exp and sqrt.
std::vector<float> replay_gdn(const std::vector<bits>& qkv, int ld, const std::vector<float>& g, const std::vector<bits>& beta,
                              const std::vector<std::int32_t>& cu) {
  const int T = cu.back();
  std::vector<float> out(z(T) * NV * D, 0.0f), qn(z(T) * 16 * D), kn(z(T) * 16 * D);
  for (int t = 0; t < T; ++t) {
    for (int kh = 0; kh < 16; ++kh) {
      const bits* row = qkv.data() + z(t) * z(ld) + z(kh * D);
      std::array<float, 32> rq{}, rk{};
      for (int l = 0; l < 32; ++l) {
        for (int e = 0; e < 4; ++e) {
          const float q = fb(row[4 * l + e]), k = fb(row[kKeyDim + 4 * l + e]);
          rq[z(l)] = std::fma(q, q, rq[z(l)]);
          rk[z(l)] = std::fma(k, k, rk[z(l)]);
        }
      }
      for (int off = 16; off > 0; off >>= 1) {
        for (int l = 0; l < off; ++l) {
          rq[z(l)] = rq[z(l)] + rq[z(l + off)];
          rk[z(l)] = rk[z(l)] + rk[z(l + off)];
        }
      }
      const float iq = 1.0f / std::sqrt(rq[0] + 1e-6f), ik = 1.0f / std::sqrt(rk[0] + 1e-6f);
      for (int d = 0; d < D; ++d) {
        qn[(z(kh) * z(T) + z(t)) * D + z(d)] = (fb(row[d]) * iq) * 0.08838834764831845f;
        kn[(z(kh) * z(T) + z(t)) * D + z(d)] = fb(row[kKeyDim + d]) * ik;
      }
    }
  }
  const auto QN = [&](int kh, int t, int d) { return qn[(z(kh) * z(T) + z(t)) * D + z(d)]; };
  const auto KN = [&](int kh, int t, int d) { return kn[(z(kh) * z(T) + z(t)) * D + z(d)]; };
  for (size_t s = 0; s + 1 < cu.size(); ++s) {
    const int start = cu[s], len = cu[s + 1] - cu[s];
    for (int h = 0; h < NV; ++h) {
      const int kh = h / 2;
      std::vector<float> S(z(D) * D, 0.0f);
      for (int c0 = 0; c0 < len; c0 += C) {
        const int t0 = start + c0, n = std::min(C, len - c0);
        std::array<float, C> gc{}, bt{}, eg{}, el{};
        for (int i = 0; i < n; ++i) {
          gc[z(i)] = g[z(t0 + i) * NV + z(h)];
          bt[z(i)] = fb(beta[z(t0 + i) * NV + z(h)]);
        }
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) gc[z(i)] = acc = acc + gc[z(i)];
        for (int i = 0; i < n; ++i) {
          eg[z(i)] = std::exp(gc[z(i)]);
          el[z(i)] = std::exp(gc[z(n - 1)] - gc[z(i)]);
        }
        std::vector<float> A(z(C) * C, 0.0f), attn(z(C) * C, 0.0f);
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < C; ++j) {
            float kk = 0.0f, qk = 0.0f;
            for (int d = 0; d < D; ++d) {
              const float kj = j < n ? KN(kh, t0 + j, d) : 0.0f;
              kk = std::fma(KN(kh, t0 + i, d), kj, kk);
              qk = std::fma(QN(kh, t0 + i, d), kj, qk);
            }
            const float decay = j <= i ? std::exp(gc[z(i)] - gc[z(j)]) : 0.0f;
            A[z(i * C + j)] = j < i ? (kk * bt[z(i)]) * decay : 0.0f;
            attn[z(i * C + j)] = qk * decay;
          }
        }
        // u (columns 0..127) and w (128..255) by forward substitution.
        std::vector<float> U(z(C) * D, 0.0f), Wm(z(C) * D, 0.0f);
        for (int col = 0; col < 2 * D; ++col) {
          const bool is_u = col < D;
          const int c = is_u ? col : col - D;
          std::array<float, C> x{};
          for (int i = 0; i < n; ++i) {
            x[z(i)] = is_u ? fb(qkv[z(t0 + i) * z(ld) + z(2 * kKeyDim + h * D + c)]) * bt[z(i)]
                           : (KN(kh, t0 + i, c) * bt[z(i)]) * eg[z(i)];
          }
          for (int i = 1; i < n; ++i) {
            float a2 = x[z(i)];
            for (int j = 0; j < i; j += 4) {
              for (int e = 0; e < 4; ++e) a2 = std::fma(-A[z(i * C + j + e)], x[z(j + e)], a2);
            }
            x[z(i)] = a2;
          }
          for (int i = 0; i < n; ++i) (is_u ? U : Wm)[z(i) * D + z(c)] = x[z(i)];
        }
        // The scan.
        std::vector<float> vn(z(C) * D, 0.0f);
        std::vector<float> pq(z(C) * D, 0.0f);
        for (int row = 0; row < C; ++row) {
          for (int c = 0; c < D; ++c) {
            float pw = 0.0f, p2 = 0.0f;
            for (int d = 0; d < D; ++d) {
              const float wv = row < n ? Wm[z(row) * D + z(d)] : 0.0f;
              const float qv = row < n ? QN(kh, t0 + row, d) * eg[z(row)] : 0.0f;
              pw = std::fma(wv, S[z(d) * D + z(c)], pw);
              p2 = std::fma(qv, S[z(d) * D + z(c)], p2);
            }
            vn[z(row) * D + z(c)] = (row < n ? U[z(row) * D + z(c)] : 0.0f) - pw;
            pq[z(row) * D + z(c)] = p2;
          }
        }
        for (int row = 0; row < n; ++row) {
          for (int c = 0; c < D; ++c) {
            float o = pq[z(row) * D + z(c)];
            for (int j = 0; j < C; ++j) o = std::fma(attn[z(row * C + j)], vn[z(j) * D + z(c)], o);
            out[(z(t0 + row) * NV + z(h)) * D + z(c)] = o;
          }
        }
        const float cd = std::exp(gc[z(n - 1)]);
        for (int d = 0; d < D; ++d) {
          for (int c = 0; c < D; ++c) {
            float upd = 0.0f;
            for (int j = 0; j < C; ++j) {
              const float kd = j < n ? KN(kh, t0 + j, d) * el[z(j)] : 0.0f;
              upd = std::fma(kd, vn[z(j) * D + z(c)], upd);
            }
            S[z(d) * D + z(c)] = std::fma(S[z(d) * D + z(c)], cd, upd);
          }
        }
      }
    }
  }
  return out;
}

// torch_recurrent_gated_delta_rule in fp64: the algorithm the chunked form rearranges.
std::vector<double> recurrent_gdn(const std::vector<bits>& qkv, int ld, const std::vector<float>& g, const std::vector<bits>& beta,
                                  const std::vector<std::int32_t>& cu) {
  std::vector<double> out(z(cu.back()) * NV * D);
  for (size_t s = 0; s + 1 < cu.size(); ++s) {
    for (int h = 0; h < NV; ++h) {
      const int kh = h / 2;
      std::vector<double> S(z(D) * D, 0.0);
      for (int t = cu[s]; t < cu[s + 1]; ++t) {
        const bits* row = qkv.data() + z(t) * z(ld);
        std::array<double, D> q{}, k{}, v{};
        double sq = 0, sk = 0;
        for (int d = 0; d < D; ++d) {
          q[z(d)] = fb(row[kh * D + d]);
          k[z(d)] = fb(row[kKeyDim + kh * D + d]);
          v[z(d)] = fb(row[2 * kKeyDim + h * D + d]);
          sq += q[z(d)] * q[z(d)];
          sk += k[z(d)] * k[z(d)];
        }
        for (int d = 0; d < D; ++d) {
          q[z(d)] /= std::sqrt(sq + 1e-6) * std::sqrt(static_cast<double>(D));
          k[z(d)] /= std::sqrt(sk + 1e-6);
        }
        const double decay = std::exp(static_cast<double>(g[z(t) * NV + z(h)]));
        const double b = fb(beta[z(t) * NV + z(h)]);
        for (double& e : S) e *= decay;
        for (int c = 0; c < D; ++c) {
          double kv = 0;
          for (int d = 0; d < D; ++d) kv += S[z(d) * D + z(c)] * k[z(d)];
          const double delta = (v[z(c)] - kv) * b;
          double o = 0;
          for (int d = 0; d < D; ++d) {
            S[z(d) * D + z(c)] += k[z(d)] * delta;
            o += S[z(d) * D + z(c)] * q[z(d)];
          }
          out[(z(t) * NV + z(h)) * D + z(c)] = o;
        }
      }
    }
  }
  return out;
}

// strong: decays down to exp(-20) per token, so a chunk's cumulative decay underflows exp to zero.
void check_gdn(Rng& rng, hipStream_t stream, const std::vector<int>& lengths, bool strong) {
  std::vector<std::int32_t> cu = {0};
  int max_len = 0;
  std::string name = "gated delta rule, sequences of";
  for (const int l : lengths) {
    cu.push_back(cu.back() + l);
    max_len = std::max(max_len, l);
    name += " " + std::to_string(l);
  }
  if (strong) name += ", decays to e^-20";
  const int T = cu.back();
  constexpr int ld = te::kGdnConvDim + 8;
  // Conv outputs are SiLU outputs: mostly small positive values with some negative ones.
  std::vector<bits> qkv(z(T) * ld);
  for (bits& b : qkv) b = tb(silu(2.0f * rng.spread(-3, 1)));
  std::vector<float> g(z(T) * NV);
  std::vector<bits> beta(g.size());
  for (size_t i = 0; i < g.size(); ++i) {
    // Decays exp(g) in (0.3, 1]; strong: a quarter of them from e^-20 up.
    g[i] = strong && rng.integer(0, 3) == 0 ? -20.0f * rng.uniform() : -0.6f * std::abs(rng.spread(-4, 0));
    beta[i] = tb(sigmoid(rng.spread(-3, 1)));
  }
  if (strong) g[0] = -20.0f;
  DeviceBuffer qkvd = upload(qkv), gd = upload(g), betad = upload(beta), cud = upload(cu);
  DeviceBuffer out(z(T) * NV * D * 2), ws(te::GdnWorkspace::bytes(T));
  te::gated_delta_rule(qkvd.as<bits>(), ld, gd.as<float>(), betad.as<bits>(), cud.as<std::int32_t>(),
                       static_cast<int>(lengths.size()), max_len, T, out.as<bits>(), te::GdnWorkspace::at(ws.data(), T), stream);
  check(hipStreamSynchronize(stream), "sync");
  const auto got = download<bits>(out, z(T) * NV * D);
  const auto want = replay_gdn(qkv, ld, g, beta, cu);
  report(name, got, want, {2.0, 1.0 / 64.0, 98.0});

  // The algorithm: per (token, head) row against the fp64 recurrence, the device's bf16 output (whose rounding alone is
  // up to 2^-9 relative) and the fp32 replay before rounding (the chunked form's own error).
  constexpr double kMaxRowRel = 4e-3, kMaxReplayRowRel = 1e-4;
  const auto exact = recurrent_gdn(qkv, ld, g, beta, cu);
  double worst = 0, worst_replay = 0;
  for (size_t r = 0; r < exact.size() / D; ++r) {
    double diff = 0, diff_replay = 0, ref = 0;
    for (int c = 0; c < D; ++c) {
      const size_t i = r * D + z(c);
      diff += std::pow(fb(got[i]) - exact[i], 2);
      diff_replay += std::pow(want[i] - exact[i], 2);
      ref += exact[i] * exact[i];
    }
    if (ref == 0) continue;
    worst = std::max(worst, std::sqrt(diff / ref));
    worst_replay = std::max(worst_replay, std::sqrt(diff_replay / ref));
  }
  const bool ok = worst <= kMaxRowRel && worst_replay <= kMaxReplayRowRel;
  std::printf("  %-44s %s (worst row rel-L2 vs the fp64 recurrence: %.3e, limit %.0e; fp32 replay %.3e, limit %.0e)\n",
              (name + ", fp64").c_str(), ok ? "ok" : "FAIL", worst, kMaxRowRel, worst_replay, kMaxReplayRowRel);
  failures += ok ? 0 : 1;
}

// ---- causal attention ----

// One query row as te_attention_kernel computes it on the scalar leaf: S over dims ascending, key steps of 64 from the
// last down, each row's softmax state as FlashAttention-2's (its row sum as four per-thread partials), P·V over the keys
// ascending.
std::vector<float> replay_attention_row(const bits* q, const bits* k, std::int64_t ldk, const bits* v, std::int64_t ldv, int key_end,
                                        int row, float scale) {
  constexpr int HD = te::kAttnHeadDim, kStep = te::kAttnStepN;
  const float log2_scale = static_cast<float>(static_cast<double>(scale) * 1.4426950408889634074);
  std::vector<float> acc(HD, 0.0f);
  std::array<float, kStep> s{}, w{};
  float row_max = 0;
  std::array<float, 4> part{};  // FlashAttention-2's per-thread partial row sums
  const int steps = (key_end + kStep - 1) / kStep;
  for (int step = steps - 1; step >= 0; --step) {
    const bool first = step == steps - 1;
    const int n0 = step * kStep, cols = std::min(kStep, key_end - n0);
    for (int c = 0; c < cols; ++c) {
      float sum = 0.0f;
      for (int d = 0; d < HD; ++d) sum = sum + fb(q[d]) * fb(k[(n0 + c) * ldk + d]);
      s[z(c)] = n0 + c > row ? -INFINITY : sum;
    }
    float cur = first ? s[0] : std::fmax(row_max, s[0]);
    for (int c = 1; c < cols; ++c) cur = std::fmax(cur, s[z(c)]);
    float rescale = 1.0f;
    if (!first) {
      rescale = std::exp2((row_max - cur) * log2_scale);
      for (float& p : part) p = p * rescale;
    }
    row_max = cur;
    const float max_scaled = cur == -INFINITY ? 0.0f : cur * log2_scale;
    for (int c = 0; c < cols; ++c) w[z(c)] = std::exp2(std::fma(s[z(c)], log2_scale, -max_scaled));
    for (int q4 = 0; q4 < 4; ++q4) {
      for (int c = 2 * q4; c < cols; ++c) {
        if (c % 8 == 2 * q4 || c % 8 == 2 * q4 + 1) part[z(q4)] = part[z(q4)] + w[z(c)];
      }
    }
    if (!first) {
      for (float& a : acc) a = a * rescale;
    }
    for (int c = 0; c < cols; ++c) {
      const float p = rbf(w[z(c)]);
      for (int d = 0; d < HD; ++d) acc[z(d)] = acc[z(d)] + p * fb(v[(n0 + c) * ldv + d]);
    }
  }
  const float row_sum = (part[0] + part[2]) + (part[1] + part[3]);
  const float inv = row_sum == 0.0f || row_sum != row_sum ? 1.0f : 1.0f / row_sum;
  for (float& a : acc) a = a * inv;
  return acc;
}

void check_attention(Rng& rng, hipStream_t stream) {
  constexpr int heads = 8, kv_heads = 2, HD = te::kAttnHeadDim;
  constexpr int ldq = heads * HD + 8, ldkv = 2 * kv_heads * HD + 16, ldo = heads * HD + 24, ldg = 2 * heads * HD;
  const std::vector<int> lengths = {37, 70, 1, 130};
  std::vector<std::int32_t> cu = {0};
  int max_len = 0;
  for (const int l : lengths) {
    cu.push_back(cu.back() + l);
    max_len = std::max(max_len, l);
  }
  const int T = cu.back();
  const auto q = spread_bits(rng, z(T) * ldq, -3, 1), kv = spread_bits(rng, z(T) * ldkv, -3, 1), gate = spread_bits(rng, z(T) * ldg, -3, 3);
  const auto o0 = spread_bits(rng, z(T) * ldo, 0, 0);
  DeviceBuffer qd = upload(q), kvd = upload(kv), gd = upload(gate), cud = upload(cu);
  for (const bool gated : {false, true}) {
    DeviceBuffer od = upload(o0);
    te::AttentionArgs a;
    a.q = qd.as<bits>();
    a.k = kvd.as<bits>();
    a.v = kvd.as<bits>() + kv_heads * HD;
    a.o = od.as<bits>();
    a.gate = gated ? gd.as<bits>() + HD : nullptr;
    a.ldq = ldq;
    a.ldk = a.ldv = ldkv;
    a.ldo = ldo;
    a.ldg = ldg;
    a.hsq = a.hsk = a.hsv = a.hso = HD;
    a.hsg = 2 * HD;
    a.cu = cud.as<std::int32_t>();
    a.batch = static_cast<int>(lengths.size());
    a.max_seqlen = max_len;
    a.heads = heads;
    a.kv_group = heads / kv_heads;
    te::causal_attention(a, stream);
    check(hipStreamSynchronize(stream), "sync");
    std::vector<float> want = widen(o0);
    std::vector<std::uint8_t> mask(want.size(), 0);
    for (size_t s = 0; s < lengths.size(); ++s) {
      const int r0 = cu[s], len = lengths[s];
      for (int h = 0; h < heads; ++h) {
        const int kvh = h / (heads / kv_heads);
        for (int r = 0; r < len; ++r) {
          // The tile's keys: those of its 16 rows.
          const int key_end = std::min(len, (r / 16 + 1) * 16);
          auto row = replay_attention_row(q.data() + z(r0 + r) * ldq + z(h * HD), kv.data() + z(r0) * ldkv + z(kvh * HD), ldkv,
                                          kv.data() + z(r0) * ldkv + z((kv_heads + kvh) * HD), ldkv, key_end, r, 0.0625f);
          for (int c = 0; c < HD; ++c) {
            float o = rbf(row[z(c)]);
            if (gated) o = o * rbf(sigmoid(fb(gate[z(r0 + r) * ldg + z(h * 2 * HD + HD + c)])));
            want[z(r0 + r) * ldo + z(h * HD + c)] = o;
            mask[z(r0 + r) * ldo + z(h * HD + c)] = 1;
          }
        }
      }
    }
    report(std::string("causal GQA attention, 4 sequences") + (gated ? ", gated" : ""), download<bits>(od, want.size()), want,
           {1.0, 1.0}, mask);
  }
}

// ---- packed GEMMs ----

void check_gemms(Rng& rng, hipStream_t stream) {
  for (const auto& [m, n, k] : std::vector<std::array<int, 3>>{{1, 64, 256}, {37, 200, 272}, {100, 96, 4096}}) {
    std::vector<bits> x(z(m) * k), w(z(n) * k), res(z(m) * n), ones(z(n), tb(1.0f));
    for (bits& b : x) b = tb(static_cast<float>(rng.integer(-3, 3)));
    for (bits& b : w) b = tb(static_cast<float>(rng.integer(-3, 3)));
    for (bits& b : res) b = tb(static_cast<float>(rng.integer(-512, 512)) * 0.5f);
    std::vector<bits> want(z(m) * n), want_res(z(m) * n);
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        float s = 0.0f;
        for (int i = 0; i < k; ++i) s += fb(x[z(r) * z(k) + z(i)]) * fb(w[z(c) * z(k) + z(i)]);
        want[z(r) * z(n) + z(c)] = tb(s);
        want_res[z(r) * z(n) + z(c)] = tb(fb(res[z(r) * z(n) + z(c)]) + rbf(s));
      }
    }
    DeviceBuffer xd = upload(x), wd = upload(w), od(z(m) * n * 2), rd = upload(res), rd2 = upload(res), ond = upload(ones),
                 orow(z(m) * n * 2);
    DeviceBuffer xp(z(te::pack_rows(m)) * z(k) * 2), wp(z(te::pack_rows(n)) * z(k) * 2);
    hip_block::pack_fragments(xp.as<bits>(), xd.as<bits>(), k, m, te::pack_rows(m), k, stream);
    hip_block::pack_fragments(wp.as<bits>(), wd.as<bits>(), k, n, te::pack_rows(n), k, stream);
    te::gemm_packed(od.as<bits>(), n, xp.as<bits>(), wp.as<bits>(), m, n, k, stream);
    te::gemm_packed_residual(rd.as<bits>(), n, xp.as<bits>(), wp.as<bits>(), ond.as<bits>(), m, n, k, stream);
    hip_block::gemm(orow.as<bits>(), n, xd.as<bits>(), k, wd.as<bits>(), k, m, n, k, stream);
    hip_block::gemm_gated_residual(rd2.as<bits>(), n, xd.as<bits>(), k, wd.as<bits>(), k, ond.as<bits>(), m, n, k, stream);
    check(hipStreamSynchronize(stream), "sync");
    const std::string shape = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    const auto got = download<bits>(od, want.size()), got_res = download<bits>(rd, want.size());
    report_exact("gemm_packed " + shape, got, want);
    report_exact("gemm_packed_residual " + shape, got_res, want_res);
    report_exact("gemm_packed = row-major gemm " + shape, got, download<bits>(orow, want.size()));
    report_exact("gemm_packed_residual = row-major " + shape, got_res, download<bits>(rd2, want.size()));
  }
  // Order-sensitive data (bf16 values of magnitudes 2^-6 to 2^3, either sign): fp32 sums that round differently under
  // any other summation order, so the packed walk must take the row-major walk's order exactly.
  for (const auto& [m, n, k] : std::vector<std::array<int, 3>>{{37, 200, 272}, {100, 12352, 4096}, {65, 4096, 12288}}) {
    const std::vector<bits> x = spread_bits(rng, z(m) * z(k), -6, 3), w = spread_bits(rng, z(n) * z(k), -6, 3),
                            res = spread_bits(rng, z(m) * z(n), -2, 4), ones(z(n), tb(1.0f));
    DeviceBuffer xd = upload(x), wd = upload(w), rd = upload(res), rd2 = upload(res), ond = upload(ones), op(z(m) * z(n) * 2),
                 orow(z(m) * z(n) * 2), xp(z(te::pack_rows(m)) * z(k) * 2), wp(z(te::pack_rows(n)) * z(k) * 2);
    hip_block::pack_fragments(xp.as<bits>(), xd.as<bits>(), k, m, te::pack_rows(m), k, stream);
    hip_block::pack_fragments(wp.as<bits>(), wd.as<bits>(), k, n, te::pack_rows(n), k, stream);
    te::gemm_packed(op.as<bits>(), n, xp.as<bits>(), wp.as<bits>(), m, n, k, stream);
    hip_block::gemm(orow.as<bits>(), n, xd.as<bits>(), k, wd.as<bits>(), k, m, n, k, stream);
    te::gemm_packed_residual(rd.as<bits>(), n, xp.as<bits>(), wp.as<bits>(), ond.as<bits>(), m, n, k, stream);
    hip_block::gemm_gated_residual(rd2.as<bits>(), n, xd.as<bits>(), k, wd.as<bits>(), k, ond.as<bits>(), m, n, k, stream);
    check(hipStreamSynchronize(stream), "sync");
    const std::string shape = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    const size_t count = z(m) * z(n);
    report_exact("gemm_packed = row-major, spread " + shape, download<bits>(op, count), download<bits>(orow, count));
    report_exact("residual packed = row-major, spread " + shape, download<bits>(rd, count), download<bits>(rd2, count));
  }
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
    bool usable = false;
    for (const DeviceInfo& d : list_devices()) usable = usable || d.unusable_reason.empty();
    if (!usable) {
      std::printf("no device runs this build (%s)\n", build_target());
      return 2;
    }
    Device device(device_index);
    Stream stream;
    std::printf("device %d: %s (%s), build %s, leaf %s\ntext encoder kernels against host replays:\n", device.info().index,
                device.info().name.c_str(), device.info().arch.c_str(), build_target(), hip_block::leaf_name());
    Rng rng{20261103};
    check_norms(rng, stream.get());
    check_rope_table(stream.get());
    check_conv_gates(rng, stream.get());
    check_gemms(rng, stream.get());
    check_attention(rng, stream.get());
    check_gdn(rng, stream.get(), {1, 5, 64, 65, 130, 300}, false);
    check_gdn(rng, stream.get(), {70, 300}, true);
    std::printf("%s\n", failures == 0 ? "all passed" : "FAILED");
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
