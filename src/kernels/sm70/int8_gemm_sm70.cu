// Volta INT8 ConvRot GEMMs: fp16 tensor-core partials, exact int32 sums and the existing quant epilogues.
#include "kernels/sm70/sm70.h"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

#include "kernels/quant/int8_gemm_impl.cuh"

namespace mi::kernels::sm70 {
namespace qd = quant::detail;

namespace {

// The SiTU factor tables of the quantized epilogue: the sm_80 tables (int8_gemm_situ.cu) are filled by a kernel built for
// sm_80+ only, so the sm_70 path keeps its own 256 KiB copy, filled once per device by a kernel built for 70. The
// factors are the same functions, so a lookup is bit-identical to the sm_80 table's value.
constexpr int kBf16Values = 1 << 16;
__device__ std::uint16_t g_situ_gate[kBf16Values];
__device__ std::uint16_t g_situ_hidden[kBf16Values];

__global__ void situ_tables_sm70_kernel() {
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
  if (dev < 0 || dev >= kMaxDevices) throw std::runtime_error("mi::kernels::sm70: device ordinal out of range");
  Device& d = devices[dev];
  const auto fill = [&] {
    constexpr int kThreads = 256;
    situ_tables_sm70_kernel<<<kBf16Values / kThreads, kThreads, 0, stream>>>();
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

// The main loop uses CUTLASS's Volta shared-memory swizzle and pipelined register loads. Global iterators still
// read INT8; MmaPipelined's NumericArrayConverter stages exact fp16 integers. Fold every 1024 products into int32:
// each fp32 partial is bounded by 2^24, preserving integer sums even with cancellation at the maximum supported K.
using Sm70SmallTile = qd::TileCfg<64, 128, 32, 32, 64, 2>;
using Sm70LargeTile = qd::TileCfg<128, 128, 32, 32, 64, 2>;
using Sm70Instr = cutlass::gemm::GemmShape<8, 8, 4>;

template <class Base>
struct ExactMma : Base {
  using typename Base::FragmentC;
  using typename Base::IteratorA;
  using typename Base::IteratorB;
  using typename Base::SharedStorage;
  SharedStorage& storage;
  int thread, warp, lane;
  CUTLASS_DEVICE ExactMma(SharedStorage& s, int t, int w, int l)
      : Base(s, t, w, l), storage(s), thread(t), warp(w), lane(l) {}

  CUTLASS_DEVICE void operator()(int iterations, FragmentC& accum, IteratorA a, IteratorB b, const FragmentC& src) {
    cutlass::Array<int, FragmentC::kElements> sums;
    sums.clear();
    constexpr int kIterations = 1024 / Base::Shape::kK;
    for (int start = 0; start < iterations; start += kIterations) {
      FragmentC partial, zero;
      zero.clear();
      Base mma(storage, thread, warp, lane);
      mma(min(kIterations, iterations - start), partial, a, b, zero);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentC::kElements; ++i) sums[i] += __float2int_rn(partial[i]);
      __syncthreads();
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kIterations; ++j) {
        ++a;
        ++b;
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentC::kElements; ++i) accum[i] = __int2float_rn(sums[i]) + src[i];
  }
};

enum class Epilogue { kPlain, kResidual, kSitu };
using qd::add;
using qd::mul;
using qd::rbf;
using qd::dequant;

template <Epilogue E, typename Tile>
struct Kernel {
  using Core = cutlass::gemm::threadblock::DefaultMmaCore<
      typename Tile::Threadblock, typename Tile::Warp, Sm70Instr, cutlass::half_t, qd::RowMajor,
      cutlass::half_t, qd::ColumnMajor, float, qd::RowMajor, cutlass::arch::OpClassTensorOp, 2>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<Core::Shape::kM, Core::Shape::kK>, std::int8_t, qd::RowMajor, 1,
      typename Core::IteratorThreadMapA, 8>;
  using PermuteB = std::conditional_t<E == Epilogue::kSitu, qd::UpGatePermuteB, cutlass::layout::NoPermute>;
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<Core::Shape::kK, Core::Shape::kN>, std::int8_t, qd::ColumnMajor, 0,
      typename Core::IteratorThreadMapB, 8, false, PermuteB>;
  using Mma = ExactMma<cutlass::gemm::threadblock::MmaPipelined<
      typename Core::Shape, IteratorA, typename Core::SmemIteratorA, IteratorB, typename Core::SmemIteratorB,
      float, qd::RowMajor, typename Core::MmaPolicy>>;
  union SharedStorage {
    typename Mma::SharedStorage mainloop;
    float result[Tile::Threadblock::kM * Tile::Threadblock::kN];
  };
};

template <Epilogue E, typename Tile>
__global__ void int8_kernel(quant::Int8Operands p, typename Kernel<E, Tile>::IteratorA::Params pa,
                             typename Kernel<E, Tile>::IteratorB::Params pb, const __nv_bfloat16* bias,
                             const __nv_bfloat16* gate, __nv_bfloat16* out, std::int64_t ldo,
                             qd::SituFactorTables tables) {
  using K = Kernel<E, Tile>;
  constexpr int BM = Tile::Threadblock::kM, BN = Tile::Threadblock::kN;
  constexpr int kThreads = (BM / Tile::Warp::kM) * (BN / Tile::Warp::kN) * 32;
  extern __shared__ __align__(16) char buffer[];
  auto& storage = *reinterpret_cast<typename K::SharedStorage*>(buffer);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  typename K::IteratorA a(pa, const_cast<std::int8_t*>(p.x_q), {int(p.m), int(p.k)}, tid, {m0, 0});
  typename K::IteratorB b(pb, const_cast<std::int8_t*>(p.w), {int(p.k), int(p.n)}, tid, {0, n0});
  typename K::Mma mma(storage.mainloop, tid, warp, lane);
  typename K::Mma::FragmentC accum, zero;
  zero.clear();
  mma((p.k + Tile::Threadblock::kK - 1) / Tile::Threadblock::kK, accum, a, b, zero);
  __syncthreads();
  using IteratorC = typename K::Core::MmaTensorOp::IteratorC;
  IteratorC c({storage.result, qd::RowMajor(BN)}, lane);
  c.add_tile_offset({warp % (BM / Tile::Warp::kM), warp / (BM / Tile::Warp::kM)});
  c.store(accum);
  __syncthreads();
  const float* result = storage.result;
  constexpr int kOutN = E == Epilogue::kSitu ? BN / 2 : BN;
  for (int i = tid; i < BM * kOutN; i += kThreads) {
    const int row = i / kOutN, c = i % kOutN;
    const int r = m0 + row, col = (E == Epilogue::kSitu ? n0 / 2 : n0) + c;
    if (r >= p.m || col >= (E == Epilogue::kSitu ? p.n / 2 : p.n)) continue;
    const float xs = p.x_scale[r];
    float value;
    if constexpr (E == Epilogue::kSitu) {
      constexpr int g = fused::detail::kUpGateGroup;
      const int hc = (c / g) * (2 * g) + c % g;
      const float h = rbf(dequant(result[row * BN + hc], xs, p.w_scale[col], 0.f));
      const float gt = rbf(dequant(result[row * BN + hc + g], xs, p.w_scale[col + p.n / 2], 0.f));
      value = qd::mul(qd::bf16_entry(tables.hidden, qd::bf16_index(h)),
                      qd::bf16_entry(tables.gate, qd::bf16_index(gt)));
    } else {
      value = dequant(result[row * BN + c], xs, p.w_scale[col], bias ? __bfloat162float(bias[col]) : 0.f);
      if constexpr (E == Epilogue::kResidual) {
        value = add(__bfloat162float(out[r * ldo + col]), rbf(mul(rbf(value), __bfloat162float(gate[col]))));
      }
    }
    out[r * ldo + col] = __float2bfloat16_rn(value);
  }
}

template <Epilogue E, typename Tile>
void launch_tile(const quant::Int8Operands& p, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
             __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr int kBM = Tile::Threadblock::kM, kBN = Tile::Threadblock::kN;
  constexpr int kThreads = (kBM / Tile::Warp::kM) * (kBN / Tile::Warp::kN) * 32;
  const dim3 grid(static_cast<unsigned>((p.m + kBM - 1) / kBM), static_cast<unsigned>((p.n + kBN - 1) / kBN));
  const typename Kernel<E, Tile>::IteratorA::Params pa{qd::RowMajor(p.ldx)};
  const typename Kernel<E, Tile>::IteratorB::Params pb{qd::ColumnMajor(p.k)};
  qd::SituFactorTables tables{};
  if constexpr (E == Epilogue::kSitu) tables = situ_tables(stream);
  constexpr int bytes = sizeof(typename Kernel<E, Tile>::SharedStorage);
  auto kernel = &int8_kernel<E, Tile>;
  if constexpr (bytes > 48 * 1024) MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  kernel<<<grid, kThreads, bytes, stream>>>(p, pa, pb, bias, gate, out, ldo, tables);
  MI_CUDA_CHECK_LAUNCH();
}

template <Epilogue E>
void launch(const quant::Int8Operands& p, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
             __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  // More rows reuse each weight tile during DiT prefill; short text/boundary shapes retain the smaller tile.
  if (p.m >= 1024) launch_tile<E, Sm70LargeTile>(p, bias, gate, out, ldo, stream);
  else launch_tile<E, Sm70SmallTile>(p, bias, gate, out, ldo, stream);
}

}  // namespace

void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream) {
  constexpr const char* name = "sm70::int8_gemm";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "out", out, ldo, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  launch<Epilogue::kPlain>(op, bias, nullptr, out, ldo, stream);
}

void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  constexpr const char* name = "sm70::int8_gemm_gated_residual";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "residual", residual, ldr, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  qd::check_vector(name, "gate", gate, /*optional=*/false);
  if (p.m == 0) return;
  launch<Epilogue::kResidual>(op, bias, gate, residual, ldr, stream);
}

void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr const char* name = "sm70::int8_gemm_situ_glu";
  if (op.n % (2 * qd::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const qd::Problem p = qd::check_operands(name, op);
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " +
                                    std::to_string(ldo));
  }
  if (p.m == 0) return;
  launch<Epilogue::kSitu>(op, nullptr, nullptr, out, ldo, stream);
}

}  // namespace mi::kernels::sm70
