// src/kernels/sm75: the Turing kernel family's build presence and its dtype conversion (the only piece that runs on
// any device, so this file also works as a smoke test that the sm_75 object library is linked in).
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

float bf16_to_float(std::uint16_t bits) {
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

// The attention tensors are [tokens, heads, dim] views with free row/head strides: the in-place conversion must touch
// exactly the view's elements and no neighbour.
TEST(GpuSm75Convert, StridedInPlaceConversionTouchesOnlyTheView) {
#ifndef MOSSINFER_HAVE_SM75
  GTEST_SKIP() << "built without MOSSINFER_SM75";
#else
  constexpr int kRows = 3, kHeads = 2, kDim = 8, kHeadStride = 16;  // heads interleaved: [token][head][16]
  std::vector<float> storage(static_cast<std::size_t>(kRows) * kHeads * kHeadStride, -12345.f);
  DeviceBf16 buf(storage);
  Tensor view = Tensor::view(buf.data, DType::kBF16, Device::kCUDA, {kRows, kHeads, kDim});
  view.stride[0] = kHeads * kHeadStride;
  view.stride[1] = kHeadStride;
  view.stride[2] = 1;
  const std::vector<std::uint16_t> before = buf.host();
  mi::kernels::sm75::convert_tensor_bf16_to_f16_inplace(view, kRows, kHeads, kDim, nullptr);
  mi::kernels::sm75::convert_tensor_f16_to_bf16_inplace(view, kRows, kHeads, kDim, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<std::uint16_t> after = buf.host();
  for (int r = 0; r < kRows; ++r) {
    for (int h = 0; h < kHeads; ++h) {
      for (int d = 0; d < kDim; ++d) {
        const std::size_t i = static_cast<std::size_t>(r) * kHeads * kHeadStride + h * kHeadStride + d;
        EXPECT_EQ(after[i], before[i]) << "r " << r << " h " << h << " d " << d;
      }
      for (int d = kDim; d < kHeadStride; ++d) {
        const std::size_t i = static_cast<std::size_t>(r) * kHeads * kHeadStride + h * kHeadStride + d;
        EXPECT_EQ(after[i], before[i]) << "padding r " << r << " h " << h << " d " << d;
      }
    }
  }
#endif
}
