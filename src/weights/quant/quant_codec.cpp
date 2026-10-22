#include "weights/quant/quant_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mi::quant {
namespace {

constexpr int kMaxGroup = 1024;  // largest table built (4 MiB of fp32); real files use 16, 64 and 256
constexpr float kInv127 = 1.0f / 127.0f;
constexpr float kMinScale = 1e-30f;
constexpr int kJBlock = 16;  // output columns kept in registers by convrot_rows

int log4(int g) { return std::countr_zero(static_cast<unsigned>(g)) / 2; }

// H_G / sqrt(G) as fp32, row-major [G, G]; built once per size.
const float* hadamard_table(int g) {
  static std::array<std::unique_ptr<float[]>, 5> tables;  // g = 4^1 .. 4^5
  static std::array<std::once_flag, 5> once;
  const int slot = log4(g) - 1;
  std::call_once(once[static_cast<std::size_t>(slot)], [g, slot] {
    auto t = std::make_unique<float[]>(static_cast<std::size_t>(g) * g);
    const float inv = 1.0f / static_cast<float>(1 << log4(g));  // 1 / sqrt(4^n) = 2^-n, exact
    for (int i = 0; i < g; ++i) {
      for (int j = 0; j < g; ++j) t[static_cast<std::size_t>(i) * g + j] = static_cast<float>(hadamard_sign(g, i, j)) * inv;
    }
    tables[static_cast<std::size_t>(slot)] = std::move(t);
  });
  return tables[static_cast<std::size_t>(slot)].get();
}

void check_group(std::int64_t k, int g) {
  if (!valid_convrot_group(g) || k % g != 0) {
    throw std::invalid_argument("mi::quant: ConvRot group " + std::to_string(g) + " is not a power of 4 in [4, 1024] dividing K=" + std::to_string(k));
  }
}

// One group: out[j] = sum_i w[i] * H[i][j], accumulated in i order from 0 (bit-identical to the cuBLAS reference;
// each product is exact, so a fused multiply-add rounds the same as the separate add).
void rotate_group(const float* w, const float* h, int g, float* out) {
  if (g < kJBlock) {
    for (int j = 0; j < g; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < g; ++i) acc += w[i] * h[static_cast<std::size_t>(i) * g + j];
      out[j] = acc;
    }
    return;
  }
  for (int jb = 0; jb < g; jb += kJBlock) {
    float acc[kJBlock] = {};
    for (int i = 0; i < g; ++i) {
      const float wi = w[i];
      const float* hr = h + static_cast<std::size_t>(i) * g + jb;
      for (int t = 0; t < kJBlock; ++t) acc[t] += wi * hr[t];
    }
    for (int t = 0; t < kJBlock; ++t) out[jb + t] = acc[t];
  }
}

void quantize_row(const float* x, std::int64_t k, std::int8_t* q, float* scale) {
  float amax = 0.0f;
  for (std::int64_t c = 0; c < k; ++c) amax = std::max(amax, std::fabs(x[c]));
  const float s = std::max(amax * kInv127, kMinScale);
  for (std::int64_t c = 0; c < k; ++c) {
    const float v = std::nearbyint(x[c] / s);
    q[c] = static_cast<std::int8_t>(std::clamp(v, -128.0f, 127.0f));
  }
  *scale = s;
}

bool finite_row(const float* x, std::int64_t k) {
  for (std::int64_t c = 0; c < k; ++c) {
    if (!std::isfinite(x[c])) return false;
  }
  return true;
}

// One finite-checked row: rotate into `rot` (k floats of scratch), then quantize.
bool quantize_convrot_row(const float* w, std::int64_t k, int g, const float* h, float* rot, std::int8_t* q, float* scale) {
  if (!finite_row(w, k)) return false;
  for (std::int64_t c = 0; c < k; c += g) rotate_group(w + c, h, g, rot + c);
  quantize_row(rot, k, q, scale);
  return true;
}

}  // namespace

bool valid_convrot_group(std::int64_t g) {
  if (g < 4 || g > kMaxGroup || (g & (g - 1)) != 0) return false;
  return std::countr_zero(static_cast<std::uint64_t>(g)) % 2 == 0;
}

int hadamard_sign(int g, int i, int j) {
  // kron^n(H4)[i][j] is the product over base-4 digits of H4[i_d][j_d]; H4[a][b] = -1 exactly when a + b == 3.
  int neg = 0;
  for (int n = log4(g); n > 0; --n, i >>= 2, j >>= 2) neg ^= ((i & 3) + (j & 3) == 3) ? 1 : 0;
  return neg ? -1 : 1;
}

float bf16_to_float(std::uint16_t v) { return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16); }

std::uint16_t float_to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if (std::isnan(v)) return static_cast<std::uint16_t>((u >> 16) | 0x40);
  return static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

float e4m3_to_float(std::uint8_t v) {
  const int e = (v >> 3) & 0xF, m = v & 7;
  float mag;
  if (e == 0xF && m == 7) mag = std::numeric_limits<float>::quiet_NaN();
  else if (e == 0) mag = std::ldexp(static_cast<float>(m), -9);
  else mag = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
  return (v & 0x80) ? -mag : mag;
}

void convrot_rows(const float* w, std::int64_t rows, std::int64_t k, int g, float* out) {
  check_group(k, g);
  const float* h = hadamard_table(g);
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < k; c += g) rotate_group(w + r * k + c, h, g, out + r * k + c);
  }
}

void quantize_rows_int8(const float* w_rot, std::int64_t rows, std::int64_t k, std::int8_t* q, float* scale) {
  for (std::int64_t r = 0; r < rows; ++r) quantize_row(w_rot + r * k, k, q + r * k, scale + r);
}

bool quantize_convrot_rows(const float* w, std::int64_t rows, std::int64_t k, int g, std::int8_t* q, float* scale) {
  check_group(k, g);
  const float* h = hadamard_table(g);
  std::vector<float> rot(static_cast<std::size_t>(k));
  bool ok = true;
  for (std::int64_t r = 0; r < rows; ++r) ok = quantize_convrot_row(w + r * k, k, g, h, rot.data(), q + r * k, scale + r) && ok;
  return ok;
}

bool quantize_convrot_rows_bf16(const std::uint16_t* w, std::int64_t rows, std::int64_t k, int g, std::int8_t* q,
                                float* scale) {
  check_group(k, g);
  const float* h = hadamard_table(g);
  std::vector<float> row(static_cast<std::size_t>(k)), rot(static_cast<std::size_t>(k));
  bool ok = true;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < k; ++c) row[static_cast<std::size_t>(c)] = bf16_to_float(w[r * k + c]);
    ok = quantize_convrot_row(row.data(), k, g, h, rot.data(), q + r * k, scale + r) && ok;
  }
  return ok;
}

void dequantize_rows(const std::int8_t* q, const float* scale, std::int64_t rows, std::int64_t k, int g, float* out) {
  if (g == 0) {
    for (std::int64_t i = 0; i < rows * k; ++i) out[i] = static_cast<float>(q[i]) * scale[i / k];
    return;
  }
  check_group(k, g);
  const float* h = hadamard_table(g);
  float v[kMaxGroup];  // one group of q * scale
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c0 = 0; c0 < k; c0 += g) {
      for (int i = 0; i < g; ++i) v[i] = static_cast<float>(q[r * k + c0 + i]) * scale[r];
      rotate_group(v, h, g, out + r * k + c0);
    }
  }
}

bool valid_scales(const float* s, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    if (!(std::isfinite(s[i]) && s[i] > 0.0f)) return false;
  }
  return true;
}

bool decode_grouped_rows(const std::uint8_t* packed, const std::uint8_t* s_rel_e4m3, const float* codebook,
                         std::int64_t rows, std::int64_t k, int bits, int group, std::int8_t* out) {
  if ((bits != 4 && bits != 6) || group <= 0 || k % group != 0 || k % 4 != 0 || (bits == 6 && codebook != nullptr)) {
    throw std::invalid_argument("mi::quant: invalid grouped decode geometry");
  }
  const std::int64_t row_bytes = k * bits / 8, groups = k / group;
  const float zero = static_cast<float>(1 << (bits - 1));
  for (std::int64_t r = 0; r < rows; ++r) {
    const std::uint8_t* p = packed + r * row_bytes;
    const std::uint8_t* hi = p + k / 2;
    const std::uint8_t* sr = s_rel_e4m3 + r * groups;
    for (std::int64_t gi = 0; gi < groups; ++gi) {
      const float s = e4m3_to_float(sr[gi]);
      if (std::isnan(s) || s < 0.0f) return false;
      for (std::int64_t c = gi * group; c < (gi + 1) * group; ++c) {
        unsigned code = (p[c / 2] >> ((c & 1) * 4)) & 0xFu;
        if (bits == 6) code |= ((hi[c / 4] >> (2 * (c % 4))) & 0x3u) << 4;
        const float level = codebook != nullptr ? codebook[code] : static_cast<float>(code) - zero;
        const float v = std::nearbyint(level * s);
        out[r * k + c] = static_cast<std::int8_t>(std::clamp(v, -127.0f, 127.0f));
      }
    }
  }
  return true;
}

void decode_int4_rows(const std::uint8_t* packed, std::int64_t rows, std::int64_t k, std::int8_t* out) {
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < k; ++c) {
      const int nib = (packed[r * (k / 2) + c / 2] >> ((c & 1) * 4)) & 0xF;
      out[r * k + c] = static_cast<std::int8_t>(nib >= 8 ? nib - 16 : nib);
    }
  }
}

void parallel_rows(std::int64_t n, int threads, const void* ctx, void (*fn)(const void*, std::int64_t, std::int64_t)) {
  if (n <= 0) return;
  if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const std::int64_t t = std::min<std::int64_t>(threads, n);
  if (t == 1) {
    fn(ctx, 0, n);
    return;
  }
  const std::int64_t chunk = (n + t - 1) / t;
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(t));
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(t));
  try {
    for (std::int64_t i = 0; i < t; ++i) {
      pool.emplace_back([&, i] {
        try {
          const std::int64_t b = i * chunk, e = std::min(n, b + chunk);
          if (b < e) fn(ctx, b, e);
        } catch (...) {
          errors[static_cast<std::size_t>(i)] = std::current_exception();
        }
      });
    }
  } catch (...) {
    // A thread failed to start: the running ones reference this frame, so they must finish first.
    for (auto& th : pool) th.join();
    throw;
  }
  for (auto& th : pool) th.join();
  for (auto& e : errors) {
    if (e) std::rethrow_exception(e);
  }
}

}  // namespace mi::quant
