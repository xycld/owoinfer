// Host side shared by hip_block_bench and hip_block_verify: device buffers, bf16 bits, and the synthetic block inputs.
#pragma once

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hip_block.h"

namespace hip_block {

inline void check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

// fp32 to bf16 bits, round to nearest even (device_math.h's bf on the host).
inline bf16_bits to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<bf16_bits>((u >> 16) | 0x0040u);
  return static_cast<bf16_bits>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float from_bf16(bf16_bits b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

// Device memory holding `count` elements of T.
template <class T>
class DevBuffer {
 public:
  DevBuffer() = default;
  explicit DevBuffer(std::size_t count) : count_(count) { check(hipMalloc(&p_, std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc"); }
  DevBuffer(const DevBuffer&) = delete;
  DevBuffer& operator=(const DevBuffer&) = delete;
  DevBuffer(DevBuffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), count_(o.count_) {}
  DevBuffer& operator=(DevBuffer&& o) noexcept {
    std::swap(p_, o.p_);
    std::swap(count_, o.count_);
    return *this;
  }
  ~DevBuffer() {
    if (p_ != nullptr) (void)hipFree(p_);
  }
  T* get() const { return static_cast<T*>(p_); }
  std::size_t size() const { return count_; }
  void upload(const std::vector<T>& h) {
    check(hipMemcpy(p_, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "hipMemcpy");
  }
  std::vector<T> download() const {
    std::vector<T> h(count_);
    check(hipMemcpy(h.data(), p_, h.size() * sizeof(T), hipMemcpyDeviceToHost), "hipMemcpy");
    return h;
  }

 private:
  void* p_ = nullptr;
  std::size_t count_ = 0;
};

// splitmix64-hashed bf16 values +-m * 2^(scale_exp - 7 - e), m in [128, 255], e in 0..exp_spread - 1, a power of two. The
// default, as the engine's block tests, gives squares that sum exactly over a thread's values, so a sum's order hardly
// matters; a wider spread makes it matter.
inline std::vector<bf16_bits> synth_bf16(std::uint64_t seed, std::size_t n, int scale_exp, unsigned exp_spread = 4) {
  std::vector<bf16_bits> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = i + seed * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float m = static_cast<float>(128 + (z & 0x7F));
    const auto e = static_cast<int>((z >> (exp_spread > 4 ? 12 : 7)) & (exp_spread - 1));
    const float v = std::ldexp((z >> 10) & 1 ? -m : m, scale_exp - 7 - e);
    out[i] = to_bf16(v);
  }
  return out;
}

// RoPE rows [tokens, 128] = [sin(64) | cos(64)] of hashed angles (the block only reads sin and cos pairs).
inline std::vector<bf16_bits> synth_rope(std::uint64_t seed, std::size_t tokens) {
  const std::vector<bf16_bits> angle = synth_bf16(seed, tokens * 64, 1);
  std::vector<bf16_bits> out(tokens * 128);
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t i = 0; i < 64; ++i) {
      const float a = from_bf16(angle[t * 64 + i]);
      out[t * 128 + i] = to_bf16(std::sin(a));
      out[t * 128 + 64 + i] = to_bf16(std::cos(a));
    }
  }
  return out;
}

// Host replays of the kernels' op orders, in fp32 unless noted. Build with FP contraction off.
inline float host_rbf(float v) { return from_bf16(to_bf16(v)); }

// rsqrt(mean(row^2) + eps) in the kernels' order: 32 threads x 4 values, then the halving tree.
inline float host_rsqrt_of_mean(const bf16_bits* row, int d, float factor) {
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

inline bf16_bits host_norm_modulate(bf16_bits x, float r, bf16_bits weight, bf16_bits scale, bf16_bits shift) {
  const float xn = host_rbf(from_bf16(x) * r);
  const float y = host_rbf(xn * from_bf16(weight));
  const float one_plus = host_rbf(1.0f + from_bf16(scale));
  const float scaled = host_rbf(y * one_plus);
  return to_bf16(scaled + from_bf16(shift));
}

// One head row of qk_norm_rope, in place. table: this token's [sin(64) | cos(64)].
inline void host_qk_norm_rope_head(bf16_bits* row, const bf16_bits* weight, const bf16_bits* table, float factor) {
  const float r = host_rsqrt_of_mean(row, kHeadDim, factor);
  std::array<float, kHeadDim> y{};
  for (int d = 0; d < kHeadDim; ++d) {
    y[static_cast<std::size_t>(d)] = host_rbf(host_rbf(from_bf16(row[d]) * r) * from_bf16(weight[d]));
  }
  for (int d = 0; d < kHeadDim / 2; ++d) {
    const float lo = y[static_cast<std::size_t>(d)], hi = y[static_cast<std::size_t>(d + 64)];
    const float si = from_bf16(table[d]), co = from_bf16(table[64 + d]);
    row[d] = to_bf16(host_rbf(lo * co) + host_rbf(-hi * si));
    row[d + 64] = to_bf16(host_rbf(hi * co) + host_rbf(lo * si));
  }
}

// MOSS _situ on bf16 values, before the final rounding.
inline float host_situ(float h, float g) {
  constexpr float kInv25 = 1.0f / 25.0f;
  const float g4 = host_rbf(std::tanh(host_rbf(g * 0.25f)));
  const float sigmoid = 1.0f / (1.0f + std::exp(-g));
  const float left = host_rbf(host_rbf(4.0f * g4) * host_rbf(sigmoid));
  const float right = host_rbf(25.0f * host_rbf(std::tanh(host_rbf(h * kInv25))));
  return left * right;
}

// One output row (one sequence, one head) of attention: the kernel's reverse walk in steps of kAttentionStepN, with every
// weight rounded to bf16. T = float replays the fp32 arithmetic with K ascending. T = double has the same rounding points
// and nothing else rounded: it shows which side of a difference is further from the intended value. q: [128];
// k, v: the sequence's first key token at this head, row strides ldk, ldv.
template <class T>
std::array<T, kHeadDim> replay_attention_row(const bf16_bits* q, const bf16_bits* k, std::int64_t ldk, const bf16_bits* v,
                                             std::int64_t ldv, int seqlen_k, float scale) {
  const T log2_scale = static_cast<T>(static_cast<float>(static_cast<double>(scale) * 1.4426950408889634074));
  std::array<T, kHeadDim> acc{};
  std::array<T, kAttentionStepN> s{}, p{};
  T row_max = 0, row_sum = 0;
  const int steps = (seqlen_k + kAttentionStepN - 1) / kAttentionStepN;
  for (int step = steps - 1; step >= 0; --step) {
    const bool first = step == steps - 1;
    const int n0 = step * kAttentionStepN;
    const int cols = std::min(kAttentionStepN, seqlen_k - n0);
    for (int c = 0; c < cols; ++c) {
      const bf16_bits* key = k + static_cast<std::int64_t>(n0 + c) * ldk;
      T sum = 0;
      for (int d = 0; d < kHeadDim; ++d) {
        const T product = static_cast<T>(from_bf16(q[d])) * static_cast<T>(from_bf16(key[d]));
        sum += product;
      }
      s[static_cast<std::size_t>(c)] = sum;
    }
    const T prev = row_max;
    T cur = first ? s[0] : std::fmax(prev, s[0]);
    for (int c = 1; c < cols; ++c) cur = std::fmax(cur, s[static_cast<std::size_t>(c)]);
    row_max = cur;
    T sum = 0;
    if (!first) {
      const T diff = prev - cur;
      const T rescale = std::exp2(diff * log2_scale);
      sum = row_sum * rescale;
      for (T& e : acc) e *= rescale;
    }
    const T max_scaled = std::isinf(cur) && cur < 0 ? T(0) : cur * log2_scale;
    for (int c = 0; c < cols; ++c) {
      const T w = std::exp2(std::fma(s[static_cast<std::size_t>(c)], log2_scale, -max_scaled));
      sum += w;
      p[static_cast<std::size_t>(c)] = static_cast<T>(host_rbf(static_cast<float>(w)));
    }
    row_sum = sum;
    for (int c = 0; c < cols; ++c) {
      const bf16_bits* val = v + static_cast<std::int64_t>(n0 + c) * ldv;
      for (int d = 0; d < kHeadDim; ++d) {
        const T product = p[static_cast<std::size_t>(c)] * static_cast<T>(from_bf16(val[d]));
        acc[static_cast<std::size_t>(d)] += product;
      }
    }
  }
  const T inv_sum = row_sum == T(0) || row_sum != row_sum ? T(1) : T(1) / row_sum;
  for (T& e : acc) e *= inv_sum;
  return acc;
}

// One block's synthetic weights and activations, device side, in the layouts of hip_block.h. Magnitudes keep
// activations near 1 through the block, which makes softmax weights spread (S * scale has a standard deviation near 1).
struct BlockInputs {
  DevBuffer<bf16_bits> norm1, norm2, q_norm, k_norm, qkv, o, up_gate, down, x_init, rope, mod;

  explicit BlockInputs(std::int64_t tokens) {
    const auto n = static_cast<std::size_t>(tokens);
    const auto make = [](std::uint64_t seed, std::size_t count, int exp) {
      DevBuffer<bf16_bits> b(count);
      b.upload(synth_bf16(seed, count, exp));
      return b;
    };
    norm1 = make(1, kDim, 0);
    norm2 = make(2, kDim, 0);
    q_norm = make(3, kHeadDim, 0);
    k_norm = make(4, kHeadDim, 0);
    qkv = make(5, std::size_t{3} * kDim * kDim, -5);
    o = make(6, std::size_t{kDim} * kDim, -5);
    up_gate = make(7, std::size_t{2} * kFfnDim * kDim, -5);
    down = make(8, std::size_t{kDim} * kFfnDim, -6);
    x_init = make(9, n * kDim, 0);
    rope = DevBuffer<bf16_bits>(n * kHeadDim);
    rope.upload(synth_rope(10, n));
    mod = make(11, std::size_t{6} * kDim, -2);
  }
  BlockWeights weights() const {
    return {norm1.get(), norm2.get(), q_norm.get(), k_norm.get(), qkv.get(), o.get(), up_gate.get(), down.get()};
  }
};

// One run's work buffers.
struct BlockActivations {
  DevBuffer<bf16_bits> x, y, qkv, attn, ffn;
  explicit BlockActivations(std::int64_t tokens) {
    const auto n = static_cast<std::size_t>(tokens);
    x = DevBuffer<bf16_bits>(n * kDim);
    y = DevBuffer<bf16_bits>(n * kDim);
    qkv = DevBuffer<bf16_bits>(n * 3 * kDim);
    attn = DevBuffer<bf16_bits>(n * kDim);
    ffn = DevBuffer<bf16_bits>(n * kFfnDim);
  }
  BlockBuffers buffers(const bf16_bits* rope, const bf16_bits* mod, const std::int32_t* cu_seqlens, int batch, int max_seqlen,
                       int tokens) const {
    return {x.get(), y.get(), qkv.get(), attn.get(), ffn.get(), rope, mod, cu_seqlens, batch, max_seqlen, tokens};
  }
};

}  // namespace hip_block
