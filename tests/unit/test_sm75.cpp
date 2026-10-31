// src/kernels/sm75: the Turing kernel family's build presence (any device, a smoke test that the sm_75 object library
// is linked in) and its dtype conversion (built 75-real only, so it runs on a Turing device alone).
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/sm75/sm75.h"

namespace {

using mi::DType;
using mi::Device;
using mi::Tensor;

[[maybe_unused]] float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float f = 0;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

std::uint16_t float_to_bf16_bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof u);
  // Round-to-nearest-even, as __float2bfloat16_rn: the test only uses values that are exact in bf16.
  const std::uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(rounded >> 16);
}

// A bf16 device buffer of `n` elements filled from host floats, with the values read back as bf16 bits.
struct DeviceBf16 {
  void* data = nullptr;
  std::int64_t n = 0;
  explicit DeviceBf16(const std::vector<float>& v) : n(static_cast<std::int64_t>(v.size())) {
    std::vector<std::uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) bits[i] = float_to_bf16_bits(v[i]);
    MI_CUDA_CHECK(cudaMalloc(&data, v.size() * sizeof(std::uint16_t)));
    MI_CUDA_CHECK(cudaMemcpy(data, bits.data(), v.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
  }
  ~DeviceBf16() {
    if (data != nullptr) cudaFree(data);
  }
  std::vector<std::uint16_t> host() const {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n));
    MI_CUDA_CHECK(cudaMemcpy(bits.data(), data, bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    return bits;
  }
  Tensor tensor() const {
    return Tensor::view(data, DType::kBF16, Device::kCUDA, {n});
  }
};

[[maybe_unused]] std::uint16_t float_to_f16_bits(float f) {
  const __half h = __float2half_rn(f);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof bits);
  return bits;
}

}  // namespace

TEST(Sm75Build, KernelFamilyPresence) {
#ifndef MOSSINFER_HAVE_SM75
  GTEST_SKIP() << "built without MOSSINFER_SM75";
#else
  EXPECT_TRUE(mi::kernels::sm75::available());
  EXPECT_TRUE(mi::kernels::sm75::needed(7));
  EXPECT_FALSE(mi::kernels::sm75::needed(8));
#endif
}

// bf16 -> fp16 is lossless for in-range values and fp16 -> bf16 recovers the exact bf16 bits, which is what the sm_75
// attention relies on when it converts q/k/v in place and restores them.
TEST(GpuSm75Convert, RoundTripIsExactInFp16Range) {
#ifndef MOSSINFER_HAVE_SM75
  GTEST_SKIP() << "built without MOSSINFER_SM75";
#else
  if (!mi::kernels::sm75::turing()) GTEST_SKIP() << "the sm_75 kernels are built 75-real: needs a Turing device";
  // 65504 is the largest fp16 value but not representable in bf16 (it rounds to 65536, which overflows fp16 to inf):
  // values above fp16's range are the documented sm_75 numerics difference, not tested here.
  const std::vector<float> values = {0.f, 1.f, -1.f, 2.5f, -3.14159f, 123.456f, -1024.f, 32768.f, 1e-3f, -7e-2f};
  DeviceBf16 buf(values);
  const Tensor t = buf.tensor();
  mi::kernels::sm75::convert_bf16_to_f16(t.data, t.data, t.numel(), nullptr);
  mi::kernels::sm75::convert_f16_to_bf16(t.data, t.data, t.numel(), nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<std::uint16_t> got = buf.host();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(got[i], float_to_bf16_bits(values[i])) << "value " << values[i];
    EXPECT_EQ(bf16_to_float(got[i]), bf16_to_float(float_to_bf16_bits(values[i]))) << "value " << values[i];
  }
#endif
}

// GEMM input rows: an in-range row converts exactly with scale 1; a row above fp16's range is scaled by the power of
// two that brings it in range, and inv_scale undoes it exactly (bf16 values times a power of two stay representable).
TEST(GpuSm75Convert, GemmRowsScaleOnlyOutOfRangeRows) {
#ifndef MOSSINFER_HAVE_SM75
  GTEST_SKIP() << "built without MOSSINFER_SM75";
#else
  if (!mi::kernels::sm75::turing()) GTEST_SKIP() << "the sm_75 kernels are built 75-real: needs a Turing device";
  constexpr int kRows = 3, kCols = 8, kSrcStride = 12;
  const std::vector<float> rows[kRows] = {{1.f, -2.5f, 3.f, 0.f, 1e-3f, -7e-2f, 123.f, 32768.f},
                                          {1e5f, -2.5f, 3.f, 0.f, 1e-2f, -7e-2f, 123.f, 65536.f},
                                          {3e38f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};
  std::vector<float> src(static_cast<std::size_t>(kRows) * kSrcStride, 0.f);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      src[static_cast<std::size_t>(r) * kSrcStride + c] = rows[r][static_cast<std::size_t>(c)];
    }
  }
  DeviceBf16 in(src);
  void* dst = nullptr;
  float* inv = nullptr;
  MI_CUDA_CHECK(cudaMalloc(&dst, kRows * kCols * 2));
  MI_CUDA_CHECK(cudaMalloc(&inv, kRows * sizeof(float)));
  mi::kernels::sm75::convert_rows_bf16_to_f16_scaled(dst, inv, in.data, kRows, kCols, kSrcStride, kCols, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint16_t> got(kRows * kCols);
  std::vector<float> inv_host(kRows);
  MI_CUDA_CHECK(cudaMemcpy(got.data(), dst, got.size() * 2, cudaMemcpyDeviceToHost));
  MI_CUDA_CHECK(cudaMemcpy(inv_host.data(), inv, inv_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dst);
  cudaFree(inv);
  EXPECT_EQ(inv_host[0], 1.f);
  EXPECT_EQ(inv_host[1], 2.f);  // max 1e5 (bf16 99840): 99840 / 2 fits, 99840 does not
  EXPECT_GT(inv_host[2], 1e33f);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const float bf = bf16_to_float(float_to_bf16_bits(rows[r][static_cast<std::size_t>(c)]));
      __half_raw raw;
      raw.x = got[static_cast<std::size_t>(r) * kCols + c];
      const float back = __half2float(__half(raw)) * inv_host[r];
      // Row 2's 1.0 times 2^-112 is below fp16's range: the documented loss of a scaled row's low values.
      EXPECT_EQ(back, r == 2 && c == 1 ? 0.f : bf) << "row " << r << " col " << c;
    }
  }
#endif
}
