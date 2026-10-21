// Internal to src/kernels/fused: CUTLASS 2.x kernel types, the fused epilogue functors and the launch helper
// shared by the fused_gemm_*.cu translation units (one per epilogue, so they compile in parallel).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "kernels/basic/detail.h"
#include "kernels/fused/fused_gemm.h"

namespace mi::kernels::fused::detail {

using bf16 = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;  // w [N, K] row-major is B [K, N] column-major
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // bf16 mma.sync, the instruction cuBLASLt uses
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
constexpr int kAlign = 8;  // elements per 16-byte access (A, B, C/D of the bf16 GEMMs)

// Tile configurations (threadblock, warp, stages); all fit the 99 KiB opt-in shared memory of sm_86/89/120.
template <int kM, int kN, int kK, int kWM, int kWN, int kStages>
struct TileCfg {
  using Threadblock = cutlass::gemm::GemmShape<kM, kN, kK>;
  using Warp = cutlass::gemm::GemmShape<kWM, kWN, kK>;
  static constexpr int stages = kStages;
};
using Tile128x256 = TileCfg<128, 256, 32, 64, 64, 3>;
using Tile256x128 = TileCfg<256, 128, 32, 64, 64, 3>;
using Tile128x128 = TileCfg<128, 128, 32, 64, 64, 4>;

// One torch-eager op per step, as kernels/basic (detail.h); _rn intrinsics keep nvcc from contracting into FMAs.
using kernels::detail::mul;
using kernels::detail::rbf;
using kernels::detail::sigmoid_f32;

// MOSS _situ (P:66-67) on bf16 values, the op order of kernels/basic/elementwise.cu situ_value.
__device__ __forceinline__ float situ_value(float h, float g) {
  constexpr float kInv25 = 1.0f / 25.0f;  // torch divides by a CPU scalar as a multiply by its fp32 reciprocal
  const float left = rbf(mul(rbf(mul(4.0f, rbf(tanhf(rbf(mul(g, 0.25f)))))), rbf(sigmoid_f32(g))));
  const float right = rbf(mul(25.0f, rbf(tanhf(rbf(mul(h, kInv25))))));
  return mul(left, right);
}

__device__ __forceinline__ std::uint16_t bf16_bits_rn(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}

// Standard-epilogue functor of gemm_situ_glu. The D operand is declared as bytes: an access of 8 accumulator
// columns [8j, 8j + 8) = hidden 4j..4j+3 then gate 4j..4j+3 (interleave_up_gate, G = 4) produces 4 bf16 results
// = 8 bytes at byte column 8j, which is bf16 column 4j of the [M, F] output.
struct SituGluOp {
  static_assert(kUpGateGroup == 4, "SituGluOp assumes groups of 4 hidden + 4 gate columns per access");
  using ElementOutput = std::uint8_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static int const kCount = 8;
  static cutlass::FloatRoundStyle const kRound = cutlass::FloatRoundStyle::round_to_nearest;
  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentSource = FragmentOutput;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;
  struct Params {};

  CUTLASS_HOST_DEVICE explicit SituGluOp(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc) const {
    FragmentOutput out;
    auto* words = reinterpret_cast<std::uint16_t*>(&out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kUpGateGroup; ++i) {
      // The GEMM output is a bf16 tensor in the reference: round both halves before SiTU.
      words[i] = bf16_bits_rn(situ_value(rbf(acc[i]), rbf(acc[kUpGateGroup + i])));
    }
    return out;
  }
  CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc, FragmentSource const&) const {
    return (*this)(acc);
  }
};

// EpilogueWithBroadcast functor of gemm_gated_residual: Z = C + bf16(bf16(AB) * V), C the residual (source,
// read in place), V the per-column gate. kernels::gated_residual's op order on the rounded GEMM output.
struct GatedResidualOp {
  using ElementOutput = bf16;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementZ = bf16;
  using ElementT = bf16;
  using ElementVector = bf16;
  static int const kElementsPerAccess = 8;
  static int const kCount = kElementsPerAccess;
  static bool const kIsSingleSource = true;
  static bool const kStoreZ = true;
  static bool const kStoreT = false;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;
  using FragmentC = cutlass::Array<ElementOutput, kCount>;
  using FragmentZ = cutlass::Array<ElementZ, kCount>;
  using FragmentT = cutlass::Array<ElementT, kCount>;
  using FragmentOutput = FragmentZ;
  struct Params {};

  CUTLASS_HOST_DEVICE explicit GatedResidualOp(Params const&) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}

  CUTLASS_DEVICE void operator()(FragmentZ& z, FragmentT&, FragmentAccumulator const& ab, FragmentC const& c,
                                 FragmentCompute const& v) const {
    using kernels::detail::add;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      const float residual = static_cast<float>(c[i]);
      z[i] = bf16(add(residual, rbf(mul(rbf(ab[i]), v[i]))));
    }
  }
  // Never called (is_source_needed() is always true); required by the epilogue interface.
  CUTLASS_DEVICE void operator()(FragmentZ& z, FragmentT&, FragmentAccumulator const& ab, FragmentCompute const& v) const {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) z[i] = bf16(rbf(mul(rbf(ab[i]), v[i])));
  }
};

using PlainOp = cutlass::epilogue::thread::LinearCombination<bf16, kAlign, float, float,
                                                             cutlass::epilogue::thread::ScaleType::Nothing>;

template <class Tile, class Op, class ElementD>
using DeviceGemm = cutlass::gemm::device::Gemm<bf16, RowMajor, bf16, ColumnMajor, ElementD, RowMajor, float,
                                               cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                                               typename Tile::Threadblock, typename Tile::Warp, InstructionShape, Op,
                                               Swizzle, Tile::stages, kAlign, kAlign>;

template <class Tile>
using DeviceGemmGatedResidual = cutlass::gemm::device::GemmUniversalWithBroadcast<
    bf16, RowMajor, bf16, ColumnMajor, bf16, RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    typename Tile::Threadblock, typename Tile::Warp, InstructionShape, GatedResidualOp, Swizzle, Tile::stages, kAlign,
    kAlign>;

// Validated problem of one call: M x N x K with row strides in elements.
struct Problem {
  int m = 0, n = 0, k = 0;
  std::int64_t ldx = 0, ldw = 0, ldo = 0;
};

// Checks x [.., K], w [N, K], out [.., n_out] (bf16, one row stride each, 16-byte aligned) and returns the
// problem with N = w.dim(0).
Problem check_problem(const char* op, const Tensor& out, const Tensor& x, const Tensor& w, std::int64_t n_out);

GemmTile tile_for(const Problem& p);

void check_status(cutlass::Status status, const char* op, const char* stage);

template <class Gemm>
void launch(const char* op, const typename Gemm::Arguments& args, cudaStream_t stream) {
  Gemm gemm;
  check_status(Gemm::can_implement(args), op, "can_implement");
  check_status(gemm.initialize(args, nullptr, stream), op, "initialize");
  check_status(gemm.run(stream), op, "run");
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::fused::detail
