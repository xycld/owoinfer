#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "core/varlen.h"

TEST(Tensor, ViewsAndSlices) {
  std::vector<float> host(24);
  auto t = mi::Tensor::view(host.data(), mi::DType::kF32, mi::Device::kCPU, {2, 3, 4});
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.nbytes(), 96u);
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_EQ(t.stride[0], 12);
  EXPECT_EQ(t.dim(-1), 4);
  auto s = t.slice0(1, 2);
  EXPECT_EQ(s.ptr<float>(), host.data() + 12);
  EXPECT_EQ(s.dim(0), 1);
  EXPECT_EQ(t.reshaped({6, 4}).dim(0), 6);
  EXPECT_THROW(t.reshaped({5, 4}), std::invalid_argument);
  EXPECT_THROW(t.slice0(1, 3), std::invalid_argument);
  EXPECT_EQ(t.describe(), "f32[2, 3, 4] cpu");
}

TEST(GpuDeviceBuffer, RoundTrip) {
  std::vector<int> in{1, 2, 3, 4}, out(4);
  mi::DeviceBuffer buf(sizeof(int) * 4);
  buf.upload(in.data(), sizeof(int) * 4);
  buf.download(out.data(), sizeof(int) * 4);
  EXPECT_EQ(in, out);
  EXPECT_THROW(buf.upload(in.data(), 17), std::out_of_range);
}

TEST(Varlen, FromLengthsBuildsConsistentValues) {
  const std::vector<int> lens = {5, 0, 17, 1};
  const mi::Varlen v = mi::Varlen::from_lengths(lens);
  EXPECT_EQ(v.cu_host, (std::vector<std::int32_t>{0, 5, 5, 22, 23}));
  EXPECT_EQ(v.max_seqlen, 17);
  EXPECT_EQ(v.total, 23);
  EXPECT_EQ(v.batch(), 4);
  EXPECT_EQ(v.cu_bytes(), 5 * sizeof(std::int32_t));
  const mi::Varlen empty = mi::Varlen::from_lengths({});
  EXPECT_EQ(empty.batch(), 0);
  EXPECT_EQ(empty.total, 0);
  const std::vector<int> bad = {3, -1};
  EXPECT_THROW(mi::Varlen::from_lengths(bad), std::invalid_argument);
  const std::vector<int> huge = {2'000'000'000, 2'000'000'000};
  EXPECT_THROW(mi::Varlen::from_lengths(huge), std::invalid_argument);
}

TEST(Varlen, UploadAndValidateOnDevice) {
  const std::vector<int> lens = {4, 0, 9};
  const mi::Varlen v = mi::Varlen::from_lengths(lens);
  mi::DeviceBuffer dev(v.cu_bytes());
  v.upload_cu(dev.get(), nullptr);
  const mi::Tensor cu = mi::Tensor::view(dev.get(), mi::DType::kI32, mi::Device::kCUDA, {v.batch() + 1});
  EXPECT_NO_THROW(mi::validate_cu_seqlens(cu, v.max_seqlen, v.total, nullptr, "test"));
  EXPECT_NO_THROW(mi::validate_cu_seqlens(cu, v.max_seqlen + 5, v.total, nullptr, "test"));
  EXPECT_THROW(mi::validate_cu_seqlens(cu, v.max_seqlen - 1, v.total, nullptr, "test"), std::invalid_argument);
  EXPECT_THROW(mi::validate_cu_seqlens(cu, v.max_seqlen, v.total + 1, nullptr, "test"), std::invalid_argument);
  const std::vector<std::int32_t> bad = {0, 6, 3, 9};  // not monotone
  dev.upload(bad.data(), bad.size() * sizeof(std::int32_t));
  EXPECT_THROW(mi::validate_cu_seqlens(cu, 9, 9, nullptr, "test"), std::invalid_argument);
  const std::vector<std::int32_t> nonzero = {1, 4, 4, 9};
  dev.upload(nonzero.data(), nonzero.size() * sizeof(std::int32_t));
  EXPECT_THROW(mi::validate_cu_seqlens(cu, 9, 9, nullptr, "test"), std::invalid_argument);
}
