// V100 rate probe, not installed. Run sm70_check first; timings are CUDA events after warm-up, not host launch time.
// Usage: sm70_bench [GEMM rows (default 8222)] [attention sequence length (default 4111)], or sm70_bench vae.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <stdexcept>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/sm70/sm70.h"
#include "kernels/sm70/vae_sm70.h"
#include "kernels/vae/vae_kernels.h"

namespace {

class Buffer {
 public:
  explicit Buffer(std::size_t bytes) {
    MI_CUDA_CHECK(cudaMalloc(&p_, bytes));
    MI_CUDA_CHECK(cudaMemset(p_, 0, bytes));
  }
  ~Buffer() { cudaFree(p_); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  template <class T> T* get() const { return static_cast<T*>(p_); }
 private:
  void* p_ = nullptr;
};

template <class F>
float time_ms(F&& f) {
  for (int i = 0; i < 5; ++i) f();
  cudaEvent_t start = nullptr, end = nullptr;
  MI_CUDA_CHECK(cudaEventCreate(&start));
  MI_CUDA_CHECK(cudaEventCreate(&end));
  MI_CUDA_CHECK(cudaEventRecord(start));
  constexpr int kIters = 20;
  for (int i = 0; i < kIters; ++i) f();
  MI_CUDA_CHECK(cudaEventRecord(end));
  MI_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0.f;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / kIters;
}

void gemm(int m, int n, int k, int epilogue) {
  Buffer x(static_cast<std::size_t>(m) * k), w(static_cast<std::size_t>(n) * k);
  Buffer xs(static_cast<std::size_t>(m) * 4), ws(static_cast<std::size_t>(n) * 4);
  Buffer bias(static_cast<std::size_t>(n) * 2), gate(static_cast<std::size_t>(n) * 2);
  Buffer out(static_cast<std::size_t>(m) * n * 2);
  mi::kernels::quant::Int8Operands p{x.get<std::int8_t>(), k, xs.get<float>(), w.get<std::int8_t>(), ws.get<float>(), m, n, k};
  const float ms = time_ms([&] {
    if (epilogue == 2) mi::kernels::quant::int8_gemm_situ_glu(p, out.get<__nv_bfloat16>(), n / 2, nullptr);
    else if (epilogue == 1) mi::kernels::quant::int8_gemm_gated_residual(p, bias.get<__nv_bfloat16>(), gate.get<__nv_bfloat16>(),
                                                                     out.get<__nv_bfloat16>(), n, nullptr);
    else mi::kernels::quant::int8_gemm(p, bias.get<__nv_bfloat16>(), out.get<__nv_bfloat16>(), n, nullptr);
  });
  std::printf("gemm M=%d N=%d K=%d epilogue=%d: %.3f ms, %.2f TOP/s\n", m, n, k, epilogue, ms,
              2.0 * m * n * k / (ms * 1e9));
}

void attention(int length, int heads, int kv_heads, int dim, bool causal) {
  constexpr int batch = 2;
  const int tokens = batch * length;
  Buffer q(static_cast<std::size_t>(tokens) * heads * dim * 2), out(static_cast<std::size_t>(tokens) * heads * dim * 2);
  Buffer k(static_cast<std::size_t>(tokens) * kv_heads * dim * 2), v(static_cast<std::size_t>(tokens) * kv_heads * dim * 2);
  Buffer cu((batch + 1) * sizeof(int)), lse(static_cast<std::size_t>(tokens) * heads * sizeof(float));
  const int bounds[] = {0, length, tokens};
  MI_CUDA_CHECK(cudaMemcpy(cu.get<int>(), bounds, sizeof(bounds), cudaMemcpyHostToDevice));
  const auto tensor = [](const Buffer& b, int t, int h, int d) {
    return mi::Tensor::view(b.get<void>(), mi::DType::kBF16, mi::Device::kCUDA, {t, h, d});
  };
  const auto tq = tensor(q, tokens, heads, dim), tk = tensor(k, tokens, kv_heads, dim), tv = tensor(v, tokens, kv_heads, dim);
  const auto to = tensor(out, tokens, heads, dim);
  const auto tc = mi::Tensor::view(cu.get<int>(), mi::DType::kI32, mi::Device::kCUDA, {batch + 1});
  const float ms = time_ms([&] {
    mi::attention::flash_fwd(tq, tk, tv, to, tc, tc, length, causal, 1.f / std::sqrt(static_cast<float>(dim)),
                              lse.get<void>(), static_cast<std::size_t>(tokens) * heads * sizeof(float), nullptr);
  });
  std::printf("attention L=%d H=%d KV=%d D=%d causal=%d: %.3f ms\n", length, heads, kv_heads, dim, causal, ms);
}

void vae() {
  using mi::DType;
  using mi::Device;
  using mi::Tensor;
  for (int h : {128, 256, 512}) {
    constexpr int channels = 384;
    Buffer x(static_cast<std::size_t>(h) * h * channels * sizeof(float));
    Buffer w(static_cast<std::size_t>(channels) * 3 * 3 * channels * sizeof(float)), bias(channels * sizeof(float));
    Buffer out(static_cast<std::size_t>(h) * h * channels * sizeof(float));
    const auto tx = Tensor::view(x.get<void>(), DType::kF32, Device::kCUDA, {1, h, h, channels});
    const auto tw = Tensor::view(w.get<void>(), DType::kF32, Device::kCUDA, {channels, 3, 3, channels});
    const auto tb = Tensor::view(bias.get<void>(), DType::kF32, Device::kCUDA, {channels});
    const auto ty = Tensor::view(out.get<void>(), DType::kF32, Device::kCUDA, {1, h, h, channels});
    const float ms = time_ms([&] { mi::vae::conv2d(tx, tw, tb, nullptr, ty, nullptr, 0, nullptr); });
    std::printf("vae fp32 conv H=W=%d C=K=%d R=3: %.3f ms\n", h, channels, ms);
  }
  constexpr int length = 16384, dim = 384;
  Buffer qkv(static_cast<std::size_t>(length) * dim * 3 * sizeof(float));
  Buffer out(static_cast<std::size_t>(length) * dim * sizeof(float));
  const auto q = Tensor::view(qkv.get<float>(), DType::kF32, Device::kCUDA, {1, length, dim});
  const auto k = Tensor::view(qkv.get<float>() + length * dim, DType::kF32, Device::kCUDA, {1, length, dim});
  const auto v = Tensor::view(qkv.get<float>() + 2 * length * dim, DType::kF32, Device::kCUDA, {1, length, dim});
  const auto o = Tensor::view(out.get<float>(), DType::kF32, Device::kCUDA, {1, length, dim});
  const auto bytes = mi::kernels::sm70::vae_attention_workspace_bytes(length);
  Buffer scratch(bytes);
  cublasLtHandle_t lt = nullptr;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLASLt handle creation failed");
  const float ms = time_ms([&] {
    mi::kernels::sm70::vae_attention_fp32(lt, q, k, v, o, 1.f / std::sqrt(static_cast<float>(dim)), scratch.get<void>(),
                                         bytes, nullptr);
  });
  cublasLtDestroy(lt);
  std::printf("vae fp32 attention L=%d D=%d: %.3f ms, %.1f MiB scratch\n", length, dim, ms,
              static_cast<double>(bytes) / (1024 * 1024));
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (!mi::kernels::sm70::volta()) return 1;
  if (argc > 1 && std::string_view(argv[1]) == "vae") { vae(); return 0; }
  const int m = argc > 1 ? std::atoi(argv[1]) : 8222;
  const int length = argc > 2 ? std::atoi(argv[2]) : 4111;
  if (m <= 0 || length <= 0) return 2;
  gemm(m, 9216, 3072, 0);
  gemm(m, 3072, 3072, 1);
  gemm(m, 16384, 3072, 2);
  gemm(m, 3072, 8192, 1);
  attention(length, 24, 24, 128, false);
  attention(282, 16, 4, 256, true);
}
