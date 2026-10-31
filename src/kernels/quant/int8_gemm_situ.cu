// int8_gemm_situ_glu (see quant.h).
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

#include "kernels/quant/int8_gemm_impl.cuh"
#include "kernels/sm75/int8_sm75.h"
#include "kernels/sm75/sm75.h"

namespace mi::kernels::quant {
namespace detail {
namespace {

// SituFactorTables' storage (256 KiB per device, inside the planner's context_bytes), filled once per device by
// situ_tables_kernel (see situ_factor_tables).
constexpr int kBf16Values = 1 << 16;
__device__ std::uint16_t g_gate_factor[kBf16Values];
__device__ std::uint16_t g_hidden_factor[kBf16Values];

__global__ void situ_tables_kernel() {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const float x = __uint_as_float(static_cast<std::uint32_t>(i) << 16);
  g_gate_factor[i] = __bfloat16_as_ushort(__float2bfloat16_rn(situ_gate_factor(x)));
  g_hidden_factor[i] = __bfloat16_as_ushort(__float2bfloat16_rn(situ_hidden_factor(x)));
}

}  // namespace

// Fill once per device: the first call outside a stream capture fills on its stream and synchronizes it, then marks
// the device (under a lock, so one fill and every later reader sees complete tables). A call inside a capture before
// that enqueues a fill into the graph (it runs before the GEMM in every replay) and leaves the device unmarked. The
// symbol addresses are cached per device.
SituFactorTables situ_factor_tables(cudaStream_t stream) {
  constexpr int kMaxDevices = 64;
  struct Device {
    std::atomic<bool> filled{false};
    std::atomic<const std::uint16_t*> gate{nullptr}, hidden{nullptr};
  };
  static Device devices[kMaxDevices];
  static std::mutex fill_mutex;
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  if (dev < 0 || dev >= kMaxDevices) throw std::runtime_error("mi::kernels::quant: device ordinal out of range");
  Device& d = devices[dev];
  const auto fill = [&] {
    constexpr int kThreads = 256;
    situ_tables_kernel<<<kBf16Values / kThreads, kThreads, 0, stream>>>();
    MI_CUDA_CHECK_LAUNCH();
  };
  if (!d.filled.load(std::memory_order_acquire)) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    MI_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
      fill();
    } else {
      const std::lock_guard<std::mutex> lock(fill_mutex);
      if (!d.filled.load(std::memory_order_relaxed)) {
        fill();
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        d.filled.store(true, std::memory_order_release);
      }
    }
  }
  if (d.gate.load(std::memory_order_acquire) == nullptr) {
    void *gate = nullptr, *hidden = nullptr;
    MI_CUDA_CHECK(cudaGetSymbolAddress(&gate, g_gate_factor));
    MI_CUDA_CHECK(cudaGetSymbolAddress(&hidden, g_hidden_factor));
    d.hidden.store(static_cast<const std::uint16_t*>(hidden), std::memory_order_release);
    d.gate.store(static_cast<const std::uint16_t*>(gate), std::memory_order_release);
  }
  return {d.gate.load(std::memory_order_acquire), d.hidden.load(std::memory_order_acquire)};
}

namespace {

template <class Tile>
void situ_glu(const Problem& p, const Int8Operands& a, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  using N = Nodes<Tile>;
  using Gemm = typename Int8Kernel<Tile, typename N::SituGlu, UpGatePermuteB>::Gemm;
  using namespace cute;
  // D as bytes: [M, 2F] bytes with a row stride of 2 * ldo bytes is exactly the bf16 [M, F] output.
  auto* d = reinterpret_cast<std::uint8_t*>(out);
  const std::int64_t ldd = 2 * ldo;
  const typename N::SituGlu::Arguments cb{
      {{}, {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}}, {a.w_scale}, situ_factor_tables(stream)},
      {d, {ldd, _1{}, ldd * p.m}}};
  launch<Gemm>("quant::int8_gemm_situ_glu", p, a, cb, stream);
}

}  // namespace
}  // namespace detail

void int8_gemm_situ_glu(const Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream,
                        Int8Tile tile) {
  constexpr const char* name = "quant::int8_gemm_situ_glu";
  if (op.n % (2 * detail::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const detail::Problem p = detail::check_operands(name, op);
  // Each epilogue access stores 4 bf16 (8 bytes) at column 4j.
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " +
                                    std::to_string(ldo));
  }
  if (p.m == 0) return;
  if (kernels::sm75::turing()) {
    detail::check_sm75_tile(name, tile);
    kernels::sm75::int8_gemm_situ_glu(op, out, ldo, stream);
    return;
  }
  detail::with_tile(detail::resolve_tile(tile, p),
                    [&](auto cfg) { detail::situ_glu<decltype(cfg)>(p, op, out, ldo, stream); });
}

}  // namespace mi::kernels::quant
