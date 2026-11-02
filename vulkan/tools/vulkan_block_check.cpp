// Checks the Vulkan DiT block's kernels against host replays of their op order, with tails on every axis, then runs and
// times one block on synthetic activations. The replays and limits come from the HIP prototype's known-answer check: `same`
// is the share of outputs with the replay's bits, and ULPs are bf16 ULPs of max(|replay|, floor). Exit status: 0 ran,
// 1 failed. Usage: vulkan_block_check [--device N] [--rows L1,L2,...] [--iters N] [--no-block]
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "block.h"
#include "half.h"
#include "runtime.h"

namespace {

using namespace mi::vk;
using bits = std::uint16_t;

float from_bf16(bits b) { return bf16_to_float(b); }
bits to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<bits>((u >> 16) | 0x0040u);
  return static_cast<bits>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float host_rbf(float v) { return from_bf16(to_bf16(v)); }

// splitmix64-hashed bf16 values +-m * 2^(scale_exp - 7 - e), m in [128, 255], e in 0..3.
std::vector<bits> synth_bf16(std::uint64_t seed, std::size_t n, int scale_exp) {
  std::vector<bits> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = i + seed * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float m = static_cast<float>(128 + (z & 0x7F));
    const auto e = static_cast<int>((z >> 7) & 3u);
    out[i] = to_bf16(std::ldexp((z >> 10) & 1 ? -m : m, scale_exp - 7 - e));
  }
  return out;
}

std::vector<bits> synth_rope(std::uint64_t seed, std::size_t tokens) {
  const std::vector<bits> angle = synth_bf16(seed, tokens * 64, 1);
  std::vector<bits> out(tokens * 128);
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t i = 0; i < 64; ++i) {
      const float a = from_bf16(angle[t * 64 + i]);
      out[t * 128 + i] = to_bf16(std::sin(a));
      out[t * 128 + 64 + i] = to_bf16(std::cos(a));
    }
  }
  return out;
}

float host_rsqrt_of_mean(const bits* row, int d, float factor) {
  std::array<float, 32> v{};
  for (int t = 0; t < 32; ++t) {
    std::array<float, 4> acc{};
    for (int idx = t; idx * 4 + 3 < d; idx += 32) {
      for (int i = 0; i < 4; ++i) {
        const float e = from_bf16(row[idx * 4 + i]);
        const float sq = e * e;
        acc[static_cast<std::size_t>(i)] += sq;
      }
    }
    v[static_cast<std::size_t>(t)] = acc[0] + acc[1] + acc[2] + acc[3];
  }
  for (int off = 16; off > 0; off >>= 1) {
    for (int t = 0; t < off; ++t) v[static_cast<std::size_t>(t)] += v[static_cast<std::size_t>(t + off)];
  }
  const float mean = v[0] * factor;
  return 1.0f / std::sqrt(mean + 1e-6f);
}

bits host_norm_modulate(bits x, float r, bits weight, bits scale, bits shift) {
  const float xn = host_rbf(from_bf16(x) * r);
  const float y = host_rbf(xn * from_bf16(weight));
  const float one_plus = host_rbf(1.0f + from_bf16(scale));
  const float scaled = host_rbf(y * one_plus);
  return to_bf16(scaled + from_bf16(shift));
}

void host_qk_norm_rope_head(bits* row, const bits* weight, const bits* table, float factor) {
  const float r = host_rsqrt_of_mean(row, kDiTHeadDim, factor);
  std::array<float, kDiTHeadDim> y{};
  for (int d = 0; d < kDiTHeadDim; ++d) {
    y[static_cast<std::size_t>(d)] = host_rbf(host_rbf(from_bf16(row[d]) * r) * from_bf16(weight[d]));
  }
  for (int d = 0; d < kDiTHeadDim / 2; ++d) {
    const float lo = y[static_cast<std::size_t>(d)], hi = y[static_cast<std::size_t>(d + 64)];
    const float si = from_bf16(table[d]), co = from_bf16(table[64 + d]);
    row[d] = to_bf16(host_rbf(lo * co) + host_rbf(-hi * si));
    row[d + 64] = to_bf16(host_rbf(hi * co) + host_rbf(lo * si));
  }
}

float host_situ(float h, float g) {
  constexpr float kInv25 = 1.0f / 25.0f;
  const float g4 = host_rbf(std::tanh(host_rbf(g * 0.25f)));
  const float sigmoid = 1.0f / (1.0f + std::exp(-g));
  const float left = host_rbf(host_rbf(4.0f * g4) * host_rbf(sigmoid));
  const float right = host_rbf(25.0f * host_rbf(std::tanh(host_rbf(h * kInv25))));
  return left * right;
}

constexpr int kStep = 32;

std::array<float, kDiTHeadDim> replay_attention_row(const bits* q, const bits* k, std::int64_t ldk, const bits* v,
                                                    std::int64_t ldv, int seqlen_k, float scale) {
  const float log2_scale = static_cast<float>(static_cast<double>(scale) * 1.4426950408889634074);
  std::array<float, kDiTHeadDim> acc{};
  std::array<float, kStep> s{}, p{};
  float row_max = 0, row_sum = 0;
  const int steps = (seqlen_k + kStep - 1) / kStep;
  for (int step = steps - 1; step >= 0; --step) {
    const bool first = step == steps - 1;
    const int n0 = step * kStep;
    const int cols = std::min(kStep, seqlen_k - n0);
    for (int c = 0; c < cols; ++c) {
      const bits* key = k + static_cast<std::int64_t>(n0 + c) * ldk;
      float sum = 0;
      for (int d = 0; d < kDiTHeadDim; ++d) {
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
      const float diff = prev - cur;
      const float rescale = std::exp2(diff * log2_scale);
      sum = row_sum * rescale;
      for (float& e : acc) e *= rescale;
    }
    const float max_scaled = std::isinf(cur) && cur < 0 ? 0.0f : cur * log2_scale;
    for (int c = 0; c < cols; ++c) {
      const float w = std::exp2(std::fma(s[static_cast<std::size_t>(c)], log2_scale, -max_scaled));
      sum += w;
      p[static_cast<std::size_t>(c)] = host_rbf(w);
    }
    row_sum = sum;
    for (int c = 0; c < cols; ++c) {
      const bits* val = v + static_cast<std::int64_t>(n0 + c) * ldv;
      for (int d = 0; d < kDiTHeadDim; ++d) {
        const float product = p[static_cast<std::size_t>(c)] * from_bf16(val[d]);
        acc[static_cast<std::size_t>(d)] += product;
      }
    }
  }
  const float inv_sum = row_sum == 0.0f || row_sum != row_sum ? 1.0f : 1.0f / row_sum;
  for (float& e : acc) e *= inv_sum;
  return acc;
}

std::unique_ptr<Buffer> device_buffer(Device& device, const std::vector<bits>& values) {
  auto b = std::make_unique<Buffer>(device, std::max<std::size_t>(values.size(), 1) * 2);
  b->upload(values.data(), values.size() * 2);
  return b;
}

std::vector<bits> read(Buffer& b, std::size_t count) {
  std::vector<bits> out(count);
  b.download(out.data(), count * 2);
  return out;
}

double ulps_off(float got, double ref, double floor_mag) {
  int e = 0;
  (void)std::frexp(std::max(std::abs(ref), floor_mag), &e);
  return std::abs(static_cast<double>(got) - ref) / std::ldexp(1.0, e - 8);
}

double rms_of(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s / static_cast<double>(std::max<std::size_t>(v.size(), 1)));
}

constexpr double kMinSamePercent = 99.0;

struct Limit {
  double ulps;
  double floor_of_rms;
};

bool report(const char* name, const std::vector<bits>& got, const std::vector<double>& ref, Limit limit) {
  const double floor_mag = rms_of(ref) * limit.floor_of_rms;
  double worst = 0.0;
  std::size_t nans = 0, same = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float g = from_bf16(got[i]);
    if (std::isnan(g)) ++nans;
    same += got[i] == to_bf16(static_cast<float>(ref[i]));
    worst = std::max(worst, ulps_off(g, ref[i], floor_mag));
  }
  const double same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(got.size());
  const bool ok = nans == 0 && worst <= limit.ulps && same_pct >= kMinSamePercent;
  std::printf("  %-22s %s (%zu values, %.3f %% same, limit %.0f %%; worst %.2f ULPs, limit %.1f; %zu NaN)\n", name,
              ok ? "ok" : "FAILED", got.size(), same_pct, kMinSamePercent, worst, limit.ulps, nans);
  return ok;
}

double host_dot(const std::vector<bits>& x, int ldx, const std::vector<bits>& w, int ldw, int r, int n, int k) {
  double s = 0.0;
  for (int i = 0; i < k; ++i) {
    s += static_cast<double>(from_bf16(x[static_cast<std::size_t>(r) * ldx + i])) * from_bf16(w[static_cast<std::size_t>(n) * ldw + i]);
  }
  return s;
}

bool check_norms(Device& device, const BlockKernels& kernels) {
  constexpr int rows = 37, d = kDiTDim;
  const auto x_h = synth_bf16(11, static_cast<std::size_t>(rows) * d, 0), w_h = synth_bf16(12, d, 0),
             scale_h = synth_bf16(13, d, -2), shift_h = synth_bf16(14, d, -2);
  auto x = device_buffer(device, x_h), w = device_buffer(device, w_h), scale = device_buffer(device, scale_h),
       shift = device_buffer(device, shift_h), out = device_buffer(device, std::vector<bits>(x_h.size()));
  const float factor = static_cast<float>(rows) / static_cast<float>(rows * d);
  std::vector<double> ref(x_h.size());
  for (int r = 0; r < rows; ++r) {
    const bits* row = x_h.data() + static_cast<std::ptrdiff_t>(r) * d;
    const float rs = host_rsqrt_of_mean(row, d, factor);
    for (int c = 0; c < d; ++c) {
      ref[static_cast<std::size_t>(r) * d + c] = from_bf16(host_norm_modulate(row[c], rs, w_h[static_cast<std::size_t>(c)],
                                                                              scale_h[static_cast<std::size_t>(c)], shift_h[static_cast<std::size_t>(c)]));
    }
  }
  device.run([&](VkCommandBuffer cb) {
    kernels.record_norm_modulate(cb, x->address(), d, w->address(), scale->address(), shift->address(), out->address(), d, rows, d);
  });
  bool ok = report("rms_norm_modulate", read(*out, x_h.size()), ref, {3.0, 1.0});

  constexpr int tokens = 37, ld = 3 * kDiTDim;
  const auto qkv_h = synth_bf16(15, static_cast<std::size_t>(tokens) * ld, 0), q_h = synth_bf16(16, kDiTHeadDim, 0),
             k_h = synth_bf16(17, kDiTHeadDim, 0), table_h = synth_rope(18, tokens);
  auto qkv = device_buffer(device, qkv_h), qw = device_buffer(device, q_h), kw = device_buffer(device, k_h), table = device_buffer(device, table_h);
  std::vector<bits> want = qkv_h;
  const std::int64_t head_rows = std::int64_t{tokens} * kDiTHeads;
  const float qk_factor = static_cast<float>(head_rows) / static_cast<float>(head_rows * kDiTHeadDim);
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < 2 * kDiTHeads; ++h) {
      host_qk_norm_rope_head(want.data() + static_cast<std::ptrdiff_t>(t) * ld + static_cast<std::ptrdiff_t>(h) * kDiTHeadDim,
                             (h < kDiTHeads ? q_h : k_h).data(), table_h.data() + static_cast<std::ptrdiff_t>(t) * kDiTHeadDim, qk_factor);
    }
  }
  device.run([&](VkCommandBuffer cb) {
    kernels.record_qk_norm_rope(cb, qkv->address(), ld, tokens, kDiTHeads, qw->address(), kw->address(), table->address(), kDiTHeadDim);
  });
  std::vector<double> rope_ref(want.size());
  for (std::size_t i = 0; i < want.size(); ++i) rope_ref[i] = from_bf16(want[i]);
  return report("qk_norm_rope", read(*qkv, want.size()), rope_ref, {3.0, 1.0}) && ok;
}

// m, n and k: tails on every axis of both paths' tiles; the larger shape also crosses tiles and bands of tiles.
bool check_gemms(Device& device, const BlockKernels& kernels, int m, int n, int k, int f) {
  const int ld = 3 * k;
  const auto x_h = synth_bf16(21, static_cast<std::size_t>(m) * ld, 0), w_h = synth_bf16(22, static_cast<std::size_t>(n) * k, -2),
             gate_h = synth_bf16(23, n, -2), res_h = synth_bf16(24, static_cast<std::size_t>(m) * n, 0);
  auto x = device_buffer(device, x_h), w = device_buffer(device, w_h), gate = device_buffer(device, gate_h),
       out = device_buffer(device, std::vector<bits>(static_cast<std::size_t>(m) * n));
  std::vector<double> plain(static_cast<std::size_t>(m) * n), gated(plain.size()), situ(static_cast<std::size_t>(m) * f);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      const double ab = host_dot(x_h, ld, w_h, k, r, c, k);
      plain[static_cast<std::size_t>(r) * n + c] = ab;
      const float delta = host_rbf(host_rbf(static_cast<float>(ab)) * from_bf16(gate_h[static_cast<std::size_t>(c)]));
      gated[static_cast<std::size_t>(r) * n + c] = from_bf16(res_h[static_cast<std::size_t>(r) * n + c]) + delta;
    }
    for (int c = 0; c < f; ++c) {
      const float h = host_rbf(static_cast<float>(host_dot(x_h, ld, w_h, k, r, c, k)));
      const float g = host_rbf(static_cast<float>(host_dot(x_h, ld, w_h, k, r, f + c, k)));
      situ[static_cast<std::size_t>(r) * f + c] = host_situ(h, g);
    }
  }
  std::printf("  GEMMs at M %d, N %d, K %d, SiTU F %d\n", m, n, k, f);
  device.run([&](VkCommandBuffer cb) { kernels.record_gemm(cb, out->address(), n, x->address(), ld, w->address(), k, m, n, k); });
  bool ok = report("gemm", read(*out, plain.size()), plain, {1.0, 1.0 / 64.0});
  out->upload(res_h.data(), res_h.size() * 2);
  device.run([&](VkCommandBuffer cb) {
    kernels.record_gemm_gated_residual(cb, out->address(), n, x->address(), ld, w->address(), k, gate->address(), m, n, k);
  });
  ok = report("gemm_gated_residual", read(*out, plain.size()), gated, {1.0, 1.0}) && ok;
  auto ffn = device_buffer(device, std::vector<bits>(static_cast<std::size_t>(m) * f));
  device.run([&](VkCommandBuffer cb) { kernels.record_gemm_situ_glu(cb, ffn->address(), f, x->address(), ld, w->address(), k, m, f, k); });
  return report("gemm_situ_glu", read(*ffn, situ.size()), situ, {1.5, 1.0}) && ok;
}

bool check_attention(Device& device, const BlockKernels& kernels) {
  constexpr int heads = 2, ld = heads * kDiTHeadDim;
  const std::vector<int> lengths = {37, 64, 5, 33, 200};
  std::vector<std::int32_t> cu = {0};
  for (const int l : lengths) cu.push_back(cu.back() + l);
  const auto total = static_cast<std::size_t>(cu.back());
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kDiTHeadDim)));
  const auto q_h = synth_bf16(31, total * ld, 0), k_h = synth_bf16(32, total * ld, 0), v_h = synth_bf16(33, total * ld, 0);
  auto q = device_buffer(device, q_h), k = device_buffer(device, k_h), v = device_buffer(device, v_h),
       o = device_buffer(device, std::vector<bits>(total * ld));
  std::vector<double> ref(total * ld);
  for (std::size_t b = 0; b < lengths.size(); ++b) {
    const std::int64_t r0 = cu[b];
    for (int h = 0; h < heads; ++h) {
      for (std::int64_t r = r0; r < cu[b + 1]; ++r) {
        const auto row = replay_attention_row(q_h.data() + r * ld + h * kDiTHeadDim, k_h.data() + r0 * ld + h * kDiTHeadDim, ld,
                                              v_h.data() + r0 * ld + h * kDiTHeadDim, ld, lengths[b], scale);
        std::copy(row.begin(), row.end(), ref.begin() + r * ld + h * kDiTHeadDim);
      }
    }
  }
  const AttentionArgs args{q->address(), k->address(), v->address(), o->address(), ld, ld, ld, ld, kDiTHeadDim, kDiTHeadDim,
                           kDiTHeadDim, kDiTHeadDim, cu, cu, heads, scale};
  device.run([&](VkCommandBuffer cb) { kernels.record_attention(cb, args); });
  return report("attention", read(*o, total * ld), ref, {1.0, 1.0});
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool run_block(Device& device, const BlockKernels& kernels, const std::vector<int>& rows, int iters) {
  std::vector<std::int32_t> cu = {0};
  for (const int l : rows) cu.push_back(cu.back() + l);
  const auto n = static_cast<std::size_t>(cu.back());
  const std::size_t dim = kDiTDim, ffn_dim = kDiTFfnDim;
  auto norm1 = device_buffer(device, synth_bf16(1, dim, 0)), norm2 = device_buffer(device, synth_bf16(2, dim, 0)),
       q_norm = device_buffer(device, synth_bf16(3, kDiTHeadDim, 0)), k_norm = device_buffer(device, synth_bf16(4, kDiTHeadDim, 0)),
       qkv_w = device_buffer(device, synth_bf16(5, 3 * dim * dim, -5)), o_w = device_buffer(device, synth_bf16(6, dim * dim, -5)),
       up = device_buffer(device, synth_bf16(7, 2 * ffn_dim * dim, -5)), down = device_buffer(device, synth_bf16(8, dim * ffn_dim, -6));
  const auto x_init = synth_bf16(9, n * dim, 0);
  auto x = device_buffer(device, x_init), rope = device_buffer(device, synth_rope(10, n)), mod = device_buffer(device, synth_bf16(11, 6 * dim, -2));
  Buffer y(device, n * dim * 2), qkv(device, n * 3 * dim * 2), attn(device, n * dim * 2), ffn(device, n * ffn_dim * 2);
  const BlockWeights w{norm1->address(), norm2->address(), q_norm->address(), k_norm->address(), qkv_w->address(), o_w->address(), up->address(), down->address()};
  const BlockBuffers b{x->address(), y.address(), qkv.address(), attn.address(), ffn.address(), rope->address(), mod->address(), cu, static_cast<std::uint32_t>(n)};

  // One submit per op, so a slow device's watchdog sees short work and the times are per op.
  std::vector<std::vector<double>> op_ms(kBlockOps);
  std::vector<double> block_ms;
  for (int it = 0; it < iters; ++it) {
    x->upload(x_init.data(), x_init.size() * 2);
    double total = 0.0;
    for (int op = 0; op < kBlockOps; ++op) {
      const double t0 = now_ms();
      device.run([&](VkCommandBuffer cb) { record_block_op(kernels, cb, op, w, b); });
      const double ms = now_ms() - t0;
      op_ms[static_cast<std::size_t>(op)].push_back(ms);
      total += ms;
    }
    block_ms.push_back(total);
  }
  const auto out = read(*x, x_init.size());
  const auto nans = static_cast<std::size_t>(std::count_if(out.begin(), out.end(), [](bits v) { return (v & 0x7fffu) > 0x7f80u; }));
  std::printf("  %-22s %s (%zu of %zu output values are NaN)\n", "whole block", nans == 0 ? "ok" : "FAILED", nans, out.size());
  std::printf("\n%zu tokens in %zu rows, %d iterations\n", n, rows.size(), iters);
  const auto row = [](const char* name, std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    std::printf("| %-20s | %10.2f | %10.2f |\n", name, ms.front(), ms[ms.size() / 2]);
  };
  std::printf("| %-20s | %10s | %10s |\n", "op", "min ms", "median ms");
  for (int op = 0; op < kBlockOps; ++op) row(block_op_name(op), op_ms[static_cast<std::size_t>(op)]);
  row("whole block", block_ms);
  return nans == 0;
}

}  // namespace

int main(int argc, char** argv) {
  int device_index = -1, iters = 3;
  bool block = true;
  std::string path_name;
  std::vector<int> rows = {4378, 4106};
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) {
      device_index = std::atoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "--path" && i + 1 < argc) {
      path_name = argv[++i];
    } else if (arg == "--no-block") {
      block = false;
    } else if (arg == "--rows" && i + 1 < argc) {
      rows.clear();
      std::stringstream ss(argv[++i]);
      std::string item;
      while (std::getline(ss, item, ',')) rows.push_back(std::atoi(item.c_str()));
    } else {
      std::printf("usage: %s [--device N] [--path coop|plain] [--rows L1,L2,...] [--iters N] [--no-block]\n", argv[0]);
      return 2;
    }
  }
  try {
    Device device(device_index);
    std::printf("device %u: %s\nknown-answer check against host replays:\n", device.info().index, device.info().name.c_str());
    if (!path_name.empty() && path_name != "coop" && path_name != "plain") {
      std::printf("--path takes coop or plain\n");
      return 2;
    }
    const BlockKernels kernels = path_name.empty() ? BlockKernels(device)
                                                   : BlockKernels(device, path_name == "coop" ? BlockPath::kCoop : BlockPath::kPlain);
    std::printf("path: %s%s%s\n", block_path_name(kernels.path()), device.info().caps.coop_bf16 ? "" : " (no coop: ",
                device.info().caps.coop_bf16 ? "" : (device.info().caps.coop_bf16_reason + ")").c_str());
    bool ok = check_norms(device, kernels);
    ok = check_gemms(device, kernels, 37, 104, 40, 24) && ok;
    ok = check_gemms(device, kernels, 300, 1048, 264, 520) && ok;
    ok = check_attention(device, kernels) && ok;
    if (!ok) {
      std::printf("known-answer check FAILED: not timing.\n");
      return 1;
    }
    if (block) ok = run_block(device, kernels, rows, iters);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
