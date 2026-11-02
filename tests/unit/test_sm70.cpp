// Host references for Volta's integer Tensor Core path, including sums above fp32's exact-integer range.
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include "backend/probe.h"
#include "core/cuda_check.h"
#include "kernels/basic/elementwise.h"
#include "kernels/sm70/sm70.h"
#include "kernels/sm70/vae_sm70.h"
#include "kernels/sm75/sm75.h"
#include "kernels/vae/vae_kernels.h"

namespace {

using mi::DType;
using mi::Device;
using mi::Tensor;

std::uint16_t bf16(float f) {
  const auto u = std::bit_cast<std::uint32_t>(f);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float f32(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

template <class T>
class Buffer {
 public:
  explicit Buffer(const std::vector<T>& values) : n_(values.size()) {
    MI_CUDA_CHECK(cudaMalloc(&p_, n_ * sizeof(T)));
    MI_CUDA_CHECK(cudaMemcpy(p_, values.data(), n_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Buffer() { cudaFree(p_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T* get() const { return p_; }
  std::vector<T> host() const {
    std::vector<T> values(n_);
    MI_CUDA_CHECK(cudaMemcpy(values.data(), p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
    return values;
  }
 private:
  T* p_ = nullptr;
  std::size_t n_;
};

void check_gemm(int m, int n, int k, bool adversarial) {
  const int ldx = k + 16, ldo = n + 8;
  std::vector<std::int8_t> x(static_cast<std::size_t>(m) * ldx, 0), w(static_cast<std::size_t>(n) * k);
  std::uint32_t rng = 42;
  const auto next = [&] {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return static_cast<std::int8_t>(static_cast<int>(rng % 256) - 128);
  };
  for (int r = 0; r < m; ++r) {
    for (int d = 0; d < k; ++d) x[static_cast<std::size_t>(r) * ldx + d] = adversarial ? -128 : next();
  }
  for (int c = 0; c < n; ++c) {
    for (int d = 0; d < k; ++d) {
      // Near-cancelling large positive/negative partial sums expose an unbounded fp32 accumulator's lost units.
      const int v = d < k / 2 ? 127 : -127;
      w[static_cast<std::size_t>(c) * k + d] = adversarial ? static_cast<std::int8_t>(d == k - 1 ? c - 8 : v) : next();
    }
  }
  std::vector<float> xs(static_cast<std::size_t>(m)), ws(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> bias(static_cast<std::size_t>(n)), gate(static_cast<std::size_t>(n));
  for (int r = 0; r < m; ++r) xs[static_cast<std::size_t>(r)] = 0.001f * static_cast<float>(r % 7 + 1);
  for (int c = 0; c < n; ++c) {
    ws[static_cast<std::size_t>(c)] = 0.002f * static_cast<float>(c % 5 + 1);
    bias[static_cast<std::size_t>(c)] = bf16(static_cast<float>(c % 13 - 6) * 0.125f);
    gate[static_cast<std::size_t>(c)] = bf16(static_cast<float>(c % 9 - 4) * 0.25f);
  }
  const std::uint16_t sentinel = bf16(-123.f);
  std::vector<std::uint16_t> expected(static_cast<std::size_t>(m) * ldo, sentinel);
  std::vector<std::uint16_t> unbiased(static_cast<std::size_t>(m) * n);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      std::int64_t acc = 0;
      for (int d = 0; d < k; ++d) acc += static_cast<int>(x[static_cast<std::size_t>(r) * ldx + d]) *
                                         static_cast<int>(w[static_cast<std::size_t>(c) * k + d]);
      const float value = (static_cast<float>(acc) * xs[static_cast<std::size_t>(r)]) * ws[static_cast<std::size_t>(c)];
      expected[static_cast<std::size_t>(r) * ldo + c] = bf16(value + f32(bias[static_cast<std::size_t>(c)]));
      unbiased[static_cast<std::size_t>(r) * n + c] = bf16(value);
    }
  }
  Buffer<std::int8_t> dx(x), dw(w);
  Buffer<float> dxs(xs), dws(ws);
  Buffer<std::uint16_t> db(bias), dg(gate), out(std::vector<std::uint16_t>(expected.size(), sentinel));
  const mi::kernels::quant::Int8Operands p{dx.get(), ldx, dxs.get(), dw.get(), dws.get(), m, n, k};
  const auto ptr = [](std::uint16_t* b) { return reinterpret_cast<__nv_bfloat16*>(b); };
  mi::kernels::quant::int8_gemm(p, ptr(db.get()), ptr(out.get()), ldo, nullptr);
  ASSERT_EQ(out.host(), expected);
  mi::kernels::quant::int8_gemm_gated_residual(p, ptr(db.get()), ptr(dg.get()), ptr(out.get()), ldo, nullptr);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      auto& e = expected[static_cast<std::size_t>(r) * ldo + c];
      e = bf16(f32(e) + f32(bf16(f32(e) * f32(gate[static_cast<std::size_t>(c)]))));
    }
  }
  ASSERT_EQ(out.host(), expected);

  const int f = n / 2, lds = f + 4;
  Buffer<std::uint16_t> linear(unbiased), situ(std::vector<std::uint16_t>(static_cast<std::size_t>(m) * lds, sentinel));
  Buffer<std::uint16_t> ref(std::vector<std::uint16_t>(static_cast<std::size_t>(m) * lds, sentinel));
  Tensor st = Tensor::view(ref.get(), DType::kBF16, Device::kCUDA, {m, f});
  st.stride[0] = lds;
  mi::kernels::situ_glu(Tensor::view(linear.get(), DType::kBF16, Device::kCUDA, {m, n}), st, nullptr);
  mi::kernels::quant::int8_gemm_situ_glu(p, ptr(situ.get()), lds, nullptr);
  ASSERT_EQ(situ.host(), ref.host());

  // No allocation or synchronization in the GEMM: graph replay must obey its stream and preserve stride padding.
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  MI_CUDA_CHECK(cudaStreamCreate(&stream));
  MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  mi::kernels::quant::int8_gemm_situ_glu(p, ptr(situ.get()), lds, stream);
  MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  MI_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 3; ++i) MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  EXPECT_EQ(situ.host(), ref.host());
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

}  // namespace

TEST(Sm70Build, KernelFamilyPresence) {
  EXPECT_EQ(mi::kernels::sm70::available(), MOSSINFER_HAVE_SM70 != 0);
}

TEST(Sm70Build, ProbeAndCuda12Recommendation) {
  const auto archs = mi::backend::parse_cuda_architectures("70-real,80-real");
  mi::backend::DeviceInfo d;
  d.cc_major = 7;
  d.cc_minor = 0;
  mi::backend::evaluate_device(d, 13000, 12090, archs);
  EXPECT_EQ(d.supported, MOSSINFER_HAVE_SM70 != 0);
  mi::backend::evaluate_device(d, 13000, 13000, archs);
  EXPECT_FALSE(d.supported);
  mi::backend::ProbeResult r;
  r.driver_version = 13000;
  r.devices = {d};
  mi::backend::recommend(r, archs);
  EXPECT_EQ(r.recommended, MOSSINFER_HAVE_SM70 ? mi::backend::Flavour::kCuda12 : mi::backend::Flavour::kNone);
  d.cc_major = 8;
  r.devices.push_back(d);
  mi::backend::recommend(r, archs);
  EXPECT_EQ(r.recommended, mi::backend::Flavour::kCuda13);
}

TEST(GpuSm70, Int8EpiloguesAndStrides) {
  if (!mi::kernels::sm70::volta()) GTEST_SKIP() << "needs MOSSINFER_SM70 and a Volta device";
  check_gemm(3, 8, 16, false);
  check_gemm(67, 136, 1040, false);
  check_gemm(129, 256, 4096, false);
  check_gemm(1025, 136, 1040, false);
}

TEST(GpuSm70, Int8ExactLargeCancellation) {
  if (!mi::kernels::sm70::volta()) GTEST_SKIP() << "needs MOSSINFER_SM70 and a Volta device";
  check_gemm(3, 16, 131056, true);
}

#if MOSSINFER_HAVE_SM70
TEST(GpuSm70, VaeConvolutionFp32AndResidualAlias) {
  if (!mi::kernels::sm70::volta()) GTEST_SKIP() << "needs MOSSINFER_SM70 and a Volta device";
  // Spatial/channel tails, both decoder widths, 1x1/3x3, padding, batch and in-place residual epilogues.
  for (const auto& shape : {std::array{2, 7, 9, 4, 3, 3}, std::array{1, 13, 11, 12, 65, 1},
                            std::array{1, 9, 17, 128, 128, 3}, std::array{1, 5, 7, 384, 96, 3},
                            std::array{1, 5, 7, 512, 256, 3}}) {
    const auto [n, h, width, c, k, r] = shape;
    SCOPED_TRACE(::testing::Message() << "C=" << c << " K=" << k << " R=" << r);
    std::uint32_t rng = 42;
    const auto values = [&](std::size_t count) {
      std::vector<float> v(count);
      for (float& f : v) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        f = static_cast<float>(static_cast<int>(rng % 2001) - 1000) / 4096.f;
      }
      return v;
    };
    const auto hx = values(static_cast<std::size_t>(n) * h * width * c), hw = values(static_cast<std::size_t>(k) * r * r * c);
    const auto hb = values(static_cast<std::size_t>(k)), hr = values(static_cast<std::size_t>(n) * h * width * k);
    Buffer<float> dx(hx), dw(hw), db(hb), out(hr);
    const Tensor x = Tensor::view(dx.get(), DType::kF32, Device::kCUDA, {n, h, width, c});
    const Tensor w = Tensor::view(dw.get(), DType::kF32, Device::kCUDA, {k, r, r, c});
    const Tensor bias = Tensor::view(db.get(), DType::kF32, Device::kCUDA, {k});
    const Tensor y = Tensor::view(out.get(), DType::kF32, Device::kCUDA, {n, h, width, k});
    EXPECT_EQ(mi::vae::conv2d_workspace_bytes(x, w), 0u);
    for (bool residual : {false, true}) {
      MI_CUDA_CHECK(cudaMemcpy(out.get(), hr.data(), hr.size() * sizeof(float), cudaMemcpyHostToDevice));
      cudaStream_t stream = nullptr;
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      MI_CUDA_CHECK(cudaStreamCreate(&stream));
      MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      mi::vae::conv2d(x, w, bias, residual ? &y : nullptr, y, nullptr, 0, stream);
      MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      MI_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      cudaGraphExecDestroy(exec);
      cudaGraphDestroy(graph);
      cudaStreamDestroy(stream);
      const auto actual = out.host();
      double max_error = 0.;
      for (int batch = 0; batch < n; ++batch) for (int oh = 0; oh < h; ++oh) for (int ow = 0; ow < width; ++ow) {
        for (int oc = 0; oc < k; ++oc) {
          double sum = 0.;
          for (int rh = 0; rh < r; ++rh) for (int rw = 0; rw < r; ++rw) {
            const int ih = oh + rh - r / 2, iw = ow + rw - r / 2;
            if (ih < 0 || ih >= h || iw < 0 || iw >= width) continue;
            for (int ic = 0; ic < c; ++ic)
              sum += static_cast<double>(hx[static_cast<std::size_t>(((batch * h + ih) * width + iw) * c + ic)]) *
                     hw[static_cast<std::size_t>(((oc * r + rh) * r + rw) * c + ic)];
          }
          const auto i = static_cast<std::size_t>(((batch * h + oh) * width + ow) * k + oc);
          const float expected = (static_cast<float>(sum) + hb[static_cast<std::size_t>(oc)]) + (residual ? hr[i] : 0.f);
          ASSERT_TRUE(std::isfinite(actual[i]));
          max_error = std::max(max_error, std::abs(static_cast<double>(actual[i]) - expected));
        }
      }
      EXPECT_LE(max_error, 3e-5);
    }
    EXPECT_EQ(dx.host(), hx);
    EXPECT_EQ(dw.host(), hw);
  }
}

TEST(GpuSm70, VaeAttentionFp32SlabsAndStrides) {
  if (!mi::kernels::sm70::volta()) GTEST_SKIP() << "needs MOSSINFER_SM70 and a Volta device";
  struct Handle {
    cublasLtHandle_t lt = nullptr;
    ~Handle() { if (lt) cublasLtDestroy(lt); }
  } handle;
  ASSERT_EQ(cublasLtCreate(&handle.lt), CUBLAS_STATUS_SUCCESS);
  for (const auto& [length, dim] : {std::pair{1, 384}, std::pair{67, 512}, std::pair{2053, 384}}) {
    constexpr int batch = 2;
    const int stride = 3 * dim + 4;
    std::vector<float> values(static_cast<std::size_t>(batch) * length * stride);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = std::sin(static_cast<float>(i % 997) * 0.137f) * 0.7f;
    }
    Buffer<float> qkv(values), out(std::vector<float>(static_cast<std::size_t>(batch) * length * dim, -123.f));
    Buffer<float> reference(std::vector<float>(static_cast<std::size_t>(batch) * length * dim));
    const auto view = [&](int offset) {
      Tensor t = Tensor::view(qkv.get() + offset, DType::kF32, Device::kCUDA, {batch, length, dim});
      t.stride[1] = stride;
      t.stride[0] = static_cast<std::int64_t>(length) * stride;
      return t;
    };
    const Tensor q = view(0), k = view(dim), v = view(2 * dim);
    const Tensor o = Tensor::view(out.get(), DType::kF32, Device::kCUDA, {batch, length, dim});
    const Tensor ref = Tensor::view(reference.get(), DType::kF32, Device::kCUDA, {batch, length, dim});
    const std::size_t bytes = mi::kernels::sm70::vae_attention_workspace_bytes(length);
    Buffer<float> allocation(std::vector<float>(bytes / sizeof(float) + 4));
    float* scratch = allocation.get() + 4;  // preserve mid_attention's 16-byte alignment contract
    const float scale = 1.f / std::sqrt(static_cast<float>(dim));
    mi::kernels::sm75::vae_attention_fp32(q, k, v, ref, scale, nullptr);
    mi::kernels::sm70::vae_attention_fp32(handle.lt, q, k, v, o, scale, scratch, bytes, nullptr);
    // The warmed cuBLASLt plans and caller-owned slab also work on a captured, non-default stream.
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    MI_CUDA_CHECK(cudaStreamCreate(&stream));
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    MI_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    mi::kernels::sm70::vae_attention_fp32(handle.lt, q, k, v, o, scale, scratch, bytes, stream);
    MI_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    MI_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 2; ++i) MI_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    const auto actual = out.host(), expected = reference.host();
    float max_error = 0.f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i]));
      max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    EXPECT_LE(max_error, 2e-5f) << "L=" << length << " D=" << dim;
    EXPECT_EQ(qkv.host(), values);
    EXPECT_THROW(mi::kernels::sm70::vae_attention_fp32(handle.lt, q, k, v, o, scale, scratch, bytes - 1, nullptr),
                 std::invalid_argument);
  }
}
#endif
