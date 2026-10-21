#include "pipeline/rng_torch_compat.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

// Bit-exactness needs every rounding to happen where torch's kernel rounds: FMAs only where written as
// std::fma, never contracted by the compiler (GCC contracts by default in gnu++ mode once FMA
// instructions are enabled, e.g. with -march=native), and no value-changing math optimizations.
// CMakeLists.txt compiles this file with -ffp-contract=off -fno-fast-math (/fp:precise on MSVC)
// after any global flags; the pragmas and checks below keep that true if the file is built otherwise.
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "rng_torch_compat.cpp must not be built with fast-math: torch_cpu_randn would lose bit-exactness"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

namespace mi::rng {
namespace {

constexpr float f32(std::uint32_t bits) { return std::bit_cast<float>(bits); }
constexpr std::uint32_t u32(float f) { return std::bit_cast<std::uint32_t>(f); }

// Constants of torch's avx_mathfun.h (zlib license, Julien Pommier / Giovanni Garberoglio, from
// Cephes), as the float bit patterns stored in libtorch_cpu.so, so no decimal rounding is involved.
constexpr float kMinNormPos = f32(0x00800000);
constexpr std::uint32_t kInvMantMask = 0x807fffff;
constexpr float kSqrtHf = f32(0x3f3504f3);
constexpr float kLogP[9] = {f32(0x3d9021bb), f32(0xbdebd1b8), f32(0x3def251a),
                            f32(0xbdfe5d4f), f32(0x3e11e9bf), f32(0xbe2aae50),
                            f32(0x3e4cceac), f32(0xbe7ffffc), f32(0x3eaaaaaa)};
constexpr float kLogQ1 = f32(0xb95e8083);
constexpr float kLogQ2 = f32(0x3f318000);
constexpr float kFourOverPi = f32(0x3fa2f983);
constexpr float kMinusDp1 = f32(0xbf490000);
constexpr float kMinusDp2 = f32(0xb97da000);
constexpr float kMinusDp3 = f32(0xb3222169);
constexpr float kCosP0 = f32(0x37ccf5ce);
constexpr float kCosP1 = f32(0xbab6061a);
constexpr float kCosP2 = f32(0x3d2aaaa5);
constexpr float kSinP0 = f32(0xb94ca1f9);
constexpr float kSinP1 = f32(0x3c08839e);
constexpr float kSinP2 = f32(0xbe2aaaa3);
constexpr float kTwoPi = f32(0x40c90fdb);  // float(2 * pi)
constexpr std::uint32_t kSignMask = 0x80000000;

// One lane of NormalFill16<float>::operator() for CPU_CAPABILITY_AVX2: u1 -> *lo, u2 -> *hi in,
// r*cos -> *lo, r*sin -> *hi out. Each statement is one instruction of the compiled kernel
// (libtorch_cpu.so, NormalFill16<float, true>::operator() .constprop.0); masks are bit ops so signed
// zeros come out the same.
inline void box_muller_lane(float* lo, float* hi) {
  // log256_ps(1 - u1)
  const float u1 = 1.0f - *lo;
  const bool invalid = u1 <= 0.0f;
  float x = u1 > kMinNormPos ? u1 : kMinNormPos;
  const auto exponent = static_cast<std::int32_t>(u32(x) >> 23) - 0x7f;
  x = f32((u32(x) & kInvMantMask) | u32(0.5f));
  float e = static_cast<float>(exponent) + 1.0f;
  const bool below_sqrthf = x < kSqrtHf;
  const float tmp = below_sqrthf ? x : 0.0f;
  e = e - (below_sqrthf ? 1.0f : 0.0f);
  x = x + -1.0f;
  x = x + tmp;
  const float z = x * x;
  float y = std::fma(kLogP[0], x, kLogP[1]);
  for (int k = 2; k < 9; ++k) y = std::fma(y, x, kLogP[k]);
  y = x * y;
  y = std::fma(y, z, e * kLogQ1);
  y = std::fma(-z, 0.5f, y);
  x = x + y;
  float log_u1 = std::fma(e, kLogQ2, x);
  if (invalid) log_u1 = f32(0xffffffff);  // unreachable: u1 is in (0, 1]
  const float radius = std::sqrt(log_u1 * -2.0f);

  // sincos256_ps(2 * pi * u2)
  const float theta = kTwoPi * *hi;
  float ax = f32(u32(theta) & ~kSignMask);
  std::uint32_t sign_sin = u32(theta) & kSignMask;
  std::int32_t j = static_cast<std::int32_t>(ax * kFourOverPi);  // < 8: no cvttps2dq overflow
  j = (j + 1) & ~1;
  const auto jf = static_cast<float>(j);
  sign_sin ^= static_cast<std::uint32_t>(j & 4) << 29;
  const bool sin_poly = (j & 2) == 0;
  const std::uint32_t sign_cos = static_cast<std::uint32_t>(~(j - 2) & 4) << 29;
  ax = std::fma(jf, kMinusDp1, ax);
  ax = std::fma(kMinusDp2, jf, ax);
  ax = std::fma(jf, kMinusDp3, ax);
  const float z2 = ax * ax;
  const float half_z2 = z2 * 0.5f;
  float yc = std::fma(kCosP0, z2, kCosP1);
  yc = std::fma(yc, z2, kCosP2);
  yc = z2 * yc;
  yc = std::fma(z2, yc, -half_z2);
  yc = yc + 1.0f;
  float ys = std::fma(kSinP0, z2, kSinP1);
  ys = std::fma(ys, z2, kSinP2);
  ys = std::fma(ax, z2 * ys, ax);
  const float ysin2 = sin_poly ? ys : 0.0f;
  const float ysin1 = sin_poly ? 0.0f : yc;
  yc = yc - ysin1;
  ys = ys - ysin2;
  const float sin_theta = f32(u32(ysin2 + ysin1) ^ sign_sin);
  const float cos_theta = f32(u32(ys + yc) ^ sign_cos);

  // fma(n, std=1, mean=0) is exact except that it turns -0 into +0; n + 0 does the same.
  *lo = radius * cos_theta + 0.0f;
  *hi = radius * sin_theta + 0.0f;
}

void normal_fill_16(float* data) {
  for (int j = 0; j < 8; ++j) box_muller_lane(data + j, data + j + 8);
}

// at::uniform_real_distribution<float>(0, 1) on CPUGeneratorImpl::random(): exact.
float uniform_float(std::mt19937& engine) {
  return static_cast<float>(engine() & 0xffffffu) * 0x1p-24f;
}

// at::uniform_real_distribution<double>(0, 1) on random64() = (first << 32) | second.
double uniform_double(std::mt19937& engine) {
  const std::uint64_t first = engine();
  const std::uint64_t second = engine();
  return static_cast<double>(((first << 32) | second) & ((std::uint64_t{1} << 53) - 1)) * 0x1p-53;
}

// at::normal_distribution<double>(0, 1) as CPUGeneratorImpl samples it, with its cached second value.
void normal_scalar(std::mt19937& engine, float* data, std::int64_t numel) {
  constexpr double kTwoPiD = 6.283185307179586476925;
  double cached = 0.0;
  bool has_cached = false;
  for (std::int64_t i = 0; i < numel; ++i) {
    double value;
    if (has_cached) {
      value = cached;
      has_cached = false;
    } else {
      const double u1 = uniform_double(engine);
      const double u2 = uniform_double(engine);
      const double r = std::sqrt(-2.0 * std::log1p(-u2));
      const double theta = kTwoPiD * u1;
      double sin_theta;
      double cos_theta;
#if defined(__GLIBC__)
      // torch's GCC build merges sin and cos into one glibc sincos call.
      ::sincos(theta, &sin_theta, &cos_theta);
#else
      sin_theta = std::sin(theta);
      cos_theta = std::cos(theta);
#endif
      cached = r * sin_theta;
      has_cached = true;
      value = r * cos_theta;
    }
    data[i] = static_cast<float>(value + 0.0);  // fma(value, std=1, mean=0)
  }
}

}  // namespace

std::vector<float> torch_cpu_randn(std::uint64_t seed, std::int64_t numel) {
  if (numel < 0) throw std::invalid_argument("torch_cpu_randn: negative numel");
  std::vector<float> out(static_cast<std::size_t>(numel));
  // torch's mt19937 keeps only the low 32 bits of the seed; std::mt19937 is the same generator.
  std::mt19937 engine(static_cast<std::uint32_t>(seed));
  float* data = out.data();
  if (numel < 16) {
    normal_scalar(engine, data, numel);
    return out;
  }
  for (std::int64_t i = 0; i < numel; ++i) data[i] = uniform_float(engine);
  for (std::int64_t i = 0; i + 16 <= numel; i += 16) normal_fill_16(data + i);
  if (numel % 16 != 0) {
    // torch redraws and re-transforms the last 16 elements, overlapping the final full chunk.
    float* tail = data + numel - 16;
    for (int i = 0; i < 16; ++i) tail[i] = uniform_float(engine);
    normal_fill_16(tail);
  }
  return out;
}

}  // namespace mi::rng

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#endif
