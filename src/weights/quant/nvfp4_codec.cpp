#include "weights/quant/nvfp4_codec.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "weights/quant/quant_codec.h"

namespace mi::quant {
namespace {

constexpr float kE2M1Values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr float kBf16Max = 3.38953139e38f;  // largest finite bf16

void check_k(std::int64_t k) {
  if (k <= 0 || k % kNvfp4Block != 0) throw std::invalid_argument("mi::quant: NVFP4 K=" + std::to_string(k) + " is not a positive multiple of 16");
}

// One row: amax per block, scale, codes. g > 0 finite.
void quantize_row(const float* x, std::int64_t k, float g, std::uint8_t* packed, std::uint8_t* scales) {
  for (std::int64_t b = 0; b < k / kNvfp4Block; ++b) {
    const float* xb = x + b * kNvfp4Block;
    float amax = 0.0f;
    for (int i = 0; i < kNvfp4Block; ++i) amax = std::max(amax, std::fabs(xb[i]));
    const std::uint8_t sf = e4m3_rne(std::min((amax / kNvfp4E2M1Max) / g, kNvfp4E4M3Max));
    scales[b] = sf;
    const float total = g * e4m3_to_float(sf);
    std::uint8_t* pb = packed + b * (kNvfp4Block / 2);
    for (int i = 0; i < kNvfp4Block; i += 2) {
      std::uint8_t c[2] = {0, 0};
      if (total != 0.0f) {
        for (int j = 0; j < 2; ++j) c[j] = e2m1_rne(std::clamp(xb[i + j] / total, -kNvfp4E2M1Max, kNvfp4E2M1Max));
      }
      pb[i / 2] = static_cast<std::uint8_t>(c[0] << 4 | c[1]);
    }
  }
}

}  // namespace

std::uint8_t e2m1_rne(float x) {
  const float a = std::fabs(x);
  // Midpoints .25 .75 1.25 1.75 2.5 3.5 5; ties go to the even mantissa: .25->0, .75->1, 1.25->1, 1.75->2, 2.5->2,
  // 3.5->4, 5->4. NaN fails every compare: magnitude 0.
  const int mag = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
  return static_cast<std::uint8_t>(mag | (std::signbit(x) ? 0x8 : 0));
}

float e2m1_to_float(std::uint8_t code) {
  const float v = kE2M1Values[code & 0x7];
  return (code & 0x8) != 0 ? -v : v;
}

std::uint8_t e4m3_rne(float x) {
  if (std::isnan(x)) return 0x7F;
  if (x <= 0.0f) return 0;
  x = std::min(x, kNvfp4E4M3Max);
  if (x < 0.015625f) {  // below 2^-6: subnormal, steps of 2^-9 (x * 512 is exact); 8 * 2^-9 rounds up to 0x08
    return static_cast<std::uint8_t>(std::nearbyint(x * 512.0f));
  }
  int e2 = 0;
  const float f = std::frexp(x, &e2);  // x = f * 2^e2, f in [0.5, 1)
  int e = e2 - 1;                      // x = m * 2^e, m in [1, 2)
  auto q = static_cast<int>(std::nearbyint((f * 2.0f - 1.0f) * 8.0f));  // exact scaling; RNE to 3 mantissa bits
  if (q == 8) {
    q = 0;
    ++e;
  }
  return static_cast<std::uint8_t>((e + 7) << 3 | q);
}

std::int64_t nvfp4_scale_rows(std::int64_t rows) { return (rows + 127) / 128 * 128; }
std::int64_t nvfp4_scale_cols(std::int64_t sfk) { return (sfk + 3) / 4 * 4; }

std::int64_t nvfp4_scale_offset(std::int64_t m, std::int64_t kb, std::int64_t sfk) {
  return ((m / 128) * ((sfk + 3) / 4) + kb / 4) * 512 + (m % 32) * 16 + ((m % 128) / 32) * 4 + kb % 4;
}

void swizzle_scales(const std::uint8_t* row_major, std::int64_t rows, std::int64_t sfk, std::uint8_t* swizzled) {
  std::memset(swizzled, 0, static_cast<std::size_t>(nvfp4_scale_rows(rows) * nvfp4_scale_cols(sfk)));
  for (std::int64_t m = 0; m < rows; ++m) {
    for (std::int64_t kb = 0; kb < sfk; ++kb) swizzled[nvfp4_scale_offset(m, kb, sfk)] = row_major[m * sfk + kb];
  }
}

void unswizzle_scales(const std::uint8_t* swizzled, std::int64_t rows, std::int64_t sfk, std::uint8_t* row_major) {
  for (std::int64_t m = 0; m < rows; ++m) {
    for (std::int64_t kb = 0; kb < sfk; ++kb) row_major[m * sfk + kb] = swizzled[nvfp4_scale_offset(m, kb, sfk)];
  }
}

void quantize_nvfp4_rows(const float* x, std::int64_t rows, std::int64_t k, const float* g, std::int64_t g_stride,
                         std::uint8_t* packed, std::uint8_t* scales) {
  check_k(k);
  for (std::int64_t r = 0; r < rows; ++r) {
    quantize_row(x + r * k, k, g[r * g_stride], packed + r * (k / 2), scales + r * (k / kNvfp4Block));
  }
}

float nvfp4_tensor_scale(float amax) { return amax / kNvfp4TensorScaleDiv; }

void quantize_nvfp4_weight(const float* w, std::int64_t n, std::int64_t k, float s2, std::uint8_t* packed,
                           std::uint8_t* swizzled) {
  check_k(k);
  const std::int64_t sfk = k / kNvfp4Block;
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(n * sfk));
  quantize_nvfp4_rows(w, n, k, &s2, 0, packed, sf.data());
  swizzle_scales(sf.data(), n, sfk, swizzled);
}

void quantize_nvfp4_act_bf16(const std::uint16_t* x, std::int64_t rows, std::int64_t k, std::uint8_t* packed,
                             std::uint8_t* scales_swizzled, float* g) {
  check_k(k);
  const std::int64_t sfk = k / kNvfp4Block;
  std::vector<float> row(static_cast<std::size_t>(k));
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(rows * sfk));
  for (std::int64_t r = 0; r < rows; ++r) {
    float amax = 0.0f;
    for (std::int64_t c = 0; c < k; ++c) {
      float v = bf16_to_float(x[r * k + c]);
      v = std::isnan(v) ? 0.0f : std::clamp(v, -kBf16Max, kBf16Max);
      row[static_cast<std::size_t>(c)] = v;
      amax = std::max(amax, std::fabs(v));
    }
    g[r] = amax == 0.0f ? 1.0f : amax / kNvfp4TensorScaleDiv;
    quantize_row(row.data(), k, g[r], packed + r * (k / 2), sf.data() + r * sfk);
  }
  swizzle_scales(sf.data(), rows, sfk, scales_swizzled);
}

void dequantize_nvfp4_rows(const std::uint8_t* packed, const std::uint8_t* swizzled, std::int64_t row0, std::int64_t rows,
                           std::int64_t k, float s2, float* out) {
  check_k(k);
  const std::int64_t sfk = k / kNvfp4Block;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t b = 0; b < sfk; ++b) {
      const float total = s2 * e4m3_to_float(swizzled[nvfp4_scale_offset(row0 + r, b, sfk)]);
      const std::uint8_t* pb = packed + r * (k / 2) + b * (kNvfp4Block / 2);
      float* ob = out + r * k + b * kNvfp4Block;
      for (int i = 0; i < kNvfp4Block / 2; ++i) {
        ob[2 * i] = e2m1_to_float(static_cast<std::uint8_t>(pb[i] >> 4)) * total;
        ob[2 * i + 1] = e2m1_to_float(pb[i] & 0xF) * total;
      }
    }
  }
}

void dequantize_nvfp4_rows_bf16(const std::uint8_t* packed, const std::uint8_t* swizzled, std::int64_t row0, std::int64_t rows,
                                std::int64_t k, float s2, std::uint16_t* out) {
  check_k(k);
  const std::int64_t sfk = k / kNvfp4Block;
  const float s2_bf16 = bf16_to_float(float_to_bf16(s2));
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t b = 0; b < sfk; ++b) {
      // Both products are exact in fp32 (8 x 4 and 2 x 8 significant bits) outside the subnormal range, where fp32
      // rounds first exactly as PyTorch's bf16 arithmetic (computed in fp32) does.
      const float total = bf16_to_float(float_to_bf16(s2_bf16 * e4m3_to_float(swizzled[nvfp4_scale_offset(row0 + r, b, sfk)])));
      const std::uint8_t* pb = packed + r * (k / 2) + b * (kNvfp4Block / 2);
      std::uint16_t* ob = out + r * k + b * kNvfp4Block;
      for (int i = 0; i < kNvfp4Block / 2; ++i) {
        ob[2 * i] = float_to_bf16(e2m1_to_float(static_cast<std::uint8_t>(pb[i] >> 4)) * total);
        ob[2 * i + 1] = float_to_bf16(e2m1_to_float(pb[i] & 0xF) * total);
      }
    }
  }
}

const char* nvfp4_scales_problem(const std::uint8_t* swizzled, std::int64_t n, std::int64_t sfk) {
  const std::int64_t cols = nvfp4_scale_cols(sfk);
  for (std::int64_t m = 0; m < nvfp4_scale_rows(n); ++m) {
    for (std::int64_t kb = 0; kb < cols; ++kb) {
      const std::uint8_t v = swizzled[nvfp4_scale_offset(m, kb, sfk)];
      if (m >= n || kb >= sfk) {
        if (v != 0) return "a nonzero padding byte";
      } else if ((v & 0x7F) == 0x7F) {
        return "a NaN block scale";
      } else if ((v & 0x80) != 0) {
        return "a negative block scale";
      }
    }
  }
  return nullptr;
}

}  // namespace mi::quant
