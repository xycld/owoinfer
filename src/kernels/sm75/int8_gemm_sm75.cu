// sm_75 INT8 ConvRot GEMMs (see kernels/sm75/int8_sm75.h): the sm_80 CUTLASS EVT kernel with Turing's m8n8k16 INT8
// MMA and tiles that fit 64 KiB of shared memory.
#include "kernels/sm75/int8_sm75.h"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

#include "kernels/quant/int8_gemm_impl.cuh"

namespace mi::kernels::sm75 {
namespace qd = quant::detail;

namespace {

// The SiTU factor tables of the Turing epilogue: the sm_80 tables (int8_gemm_situ.cu) are filled by a kernel built for
// sm_80+ only, so the sm_75 path keeps its own 256 KiB copy, filled once per device by a kernel built for 75. The
// factors are the same functions, so a lookup is bit-identical to the sm_80 table's value.
constexpr int kBf16Values = 1 << 16;
__device__ std::uint16_t g_situ_gate[kBf16Values];
__device__ std::uint16_t g_situ_hidden[kBf16Values];

__global__ void situ_tables_sm75_kernel() {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const float x = __uint_as_float(static_cast<std::uint32_t>(i) << 16);
  g_situ_gate[i] = __bfloat16_as_ushort(__float2bfloat16_rn(qd::situ_gate_factor(x)));
  g_situ_hidden[i] = __bfloat16_as_ushort(__float2bfloat16_rn(qd::situ_hidden_factor(x)));
}

// Fill once per device (see quant::detail::situ_factor_tables; the same lock and capture handling).
qd::SituFactorTables situ_tables(cudaStream_t stream) {
  constexpr int kMaxDevices = 64;
  struct Device {
    std::atomic<bool> filled{false};
    std::atomic<const std::uint16_t*> gate{nullptr}, hidden{nullptr};
  };
  static Device devices[kMaxDevices];
  static std::mutex fill_mutex;
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  if (dev < 0 || dev >= kMaxDevices) throw std::runtime_error("mi::kernels::sm75: device ordinal out of range");
  Device& d = devices[dev];
  const auto fill = [&] {
    constexpr int kThreads = 256;
    situ_tables_sm75_kernel<<<kBf16Values / kThreads, kThreads, 0, stream>>>();
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
    MI_CUDA_CHECK(cudaGetSymbolAddress(&gate, g_situ_gate));
    MI_CUDA_CHECK(cudaGetSymbolAddress(&hidden, g_situ_hidden));
    d.hidden.store(static_cast<const std::uint16_t*>(hidden), std::memory_order_release);
    d.gate.store(static_cast<const std::uint16_t*>(gate), std::memory_order_release);
  }
  return {d.gate.load(std::memory_order_acquire), d.hidden.load(std::memory_order_acquire)};
}

using Sm75Tile = qd::TileCfg<64, 128, 64, 32, 64, 2>;  // threadblock 64x128, warp 32x64, K 64, 2 stages: 24 KiB
using Sm75Instr = cutlass::gemm::GemmShape<8, 8, 16>;

template <class Callbacks, class PermuteB = cutlass::layout::NoPermute>
using Sm75Kernel = typename qd::Int8Kernel<Sm75Tile, Callbacks, PermuteB, cutlass::arch::Sm75, Sm75Instr>::Gemm;

void plain(const qd::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out,
           std::int64_t ldo, cudaStream_t stream) {
  using N = qd::Nodes<Sm75Tile>;
  using Gemm = Sm75Kernel<typename N::Plain>;
  using namespace cute;
  const typename N::Plain::Arguments cb{
      {{},
       {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}},
       {a.w_scale, 0.f, {_0{}, _1{}, p.n}},
       {reinterpret_cast<const qd::bf16*>(bias), qd::bf16(0), {_0{}, _1{}, p.n}},
       {}},
      {reinterpret_cast<qd::bf16*>(out), {ldo, _1{}, ldo * p.m}}};
  qd::launch<Gemm>("sm75::int8_gemm", p, a, cb, stream);
}

void gated_residual(const qd::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias,
                    const __nv_bfloat16* gate, __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  using N = qd::Nodes<Sm75Tile>;
  using Gemm = Sm75Kernel<typename N::GatedResidual>;
  using namespace cute;
  auto* r = reinterpret_cast<qd::bf16*>(residual);
  const typename N::GatedResidual::Arguments cb{
      {{},
       {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}},
       {a.w_scale, 0.f, {_0{}, _1{}, p.n}},
       {reinterpret_cast<const qd::bf16*>(bias), qd::bf16(0), {_0{}, _1{}, p.n}},
       {reinterpret_cast<const qd::bf16*>(gate), qd::bf16(0), {_0{}, _1{}, p.n}},
       {r, qd::bf16(0), {ldr, _1{}, ldr * p.m}},
       {}},
      {r, {ldr, _1{}, ldr * p.m}}};
  qd::launch<Gemm>("sm75::int8_gemm_gated_residual", p, a, cb, stream);
}

void situ_glu(const qd::Problem& p, const quant::Int8Operands& a, __nv_bfloat16* out, std::int64_t ldo,
              cudaStream_t stream) {
  using N = qd::Nodes<Sm75Tile>;
  using Gemm = Sm75Kernel<typename N::SituGlu, qd::UpGatePermuteB>;
  using namespace cute;
  auto* d = reinterpret_cast<std::uint8_t*>(out);
  const std::int64_t ldd = 2 * ldo;
  const typename N::SituGlu::Arguments cb{
      {{}, {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}}, {a.w_scale}, situ_tables(stream)},
      {d, {ldd, _1{}, ldd * p.m}}};
  qd::launch<Gemm>("sm75::int8_gemm_situ_glu", p, a, cb, stream);
}

}  // namespace

void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "out", out, ldo, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  plain(p, op, bias, out, ldo, stream);
}

void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm_gated_residual";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "residual", residual, ldr, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  qd::check_vector(name, "gate", gate, /*optional=*/false);
  if (p.m == 0) return;
  gated_residual(p, op, bias, gate, residual, ldr, stream);
}

void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm_situ_glu";
  if (op.n % (2 * qd::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const qd::Problem p = qd::check_operands(name, op);
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " +
                                    std::to_string(ldo));
  }
  if (p.m == 0) return;
  situ_glu(p, op, out, ldo, stream);
}

}  // namespace mi::kernels::sm75
