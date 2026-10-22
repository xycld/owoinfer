// SPDX-License-Identifier: Apache-2.0
// Internal to src/kernels/quant4: the CUTLASS 2.x INT4 GEMM kernel types (ConvRot W4A4) and the launch
// helper shared by the int4_gemm*.cu translation units (one per epilogue, so they compile in parallel).
//
// The kernel is comfy-kitchen @dba9fac's FusedInt4Gemm (comfy_kitchen/backends/cuda/ops/convrot_w4a4.cu, Copyright
// (c) 2025 Comfy Org, Apache-2.0): Sm80 DefaultGemm with cutlass::int4b_t operands, the 16x8x64 instruction,
// OpMultiplyAddSaturate and the acc * x_scale[row] * w_scale[col] (+ bias) visitor tree. The epilogue visitor trees
// themselves (one fused compute node per epilogue, the gated-residual and SiTU-GLU nodes, the permuted ffn_up_gate
// operand) are the INT8 tier's, reused from kernels/quant/int8_gemm_impl.cuh: the accumulator type and the output
// layout are the same, only the operand element type differs. This file is a modified version (Apache-2.0 §4(b)).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/quant/int8_gemm_impl.cuh"  // Nodes, TileCfg, UpGatePermuteB, kUpGateGroup
#include "kernels/quant4/quant4.h"

namespace mi::kernels::quant4::detail {

using quant::detail::bf16;
using quant::detail::ColumnMajor;
using quant::detail::Nodes;
using quant::detail::RowMajor;
using quant::detail::TileCfg;
using quant::detail::UpGatePermuteB;
using quant::detail::kUpGateGroup;

using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;  // int4 mma.sync m16n8k64
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
constexpr int kAlignAB = 32;  // int4 elements per 16-byte access

// Tile configurations (threadblock, warp, stages); all fit the 99 KiB opt-in shared memory of sm_86/89/120.
using Tile128x256x128 = TileCfg<128, 256, 128, 64, 64, 3>;
using Tile256x128x128 = TileCfg<256, 128, 128, 64, 64, 3>;
using Tile64x128x128 = TileCfg<64, 128, 128, 32, 64, 4>;

template <class F>
void with_tile(Int4Tile tile, F&& f) {
  switch (tile) {
    case Int4Tile::k256x128x128: f(Tile256x128x128{}); break;
    case Int4Tile::k64x128x128: f(Tile64x128x128{}); break;
    default: f(Tile128x256x128{}); break;
  }
}

template <class Tile, class Callbacks, class PermuteB = cutlass::layout::NoPermute>
struct Int4Kernel {
  using GemmBase = typename cutlass::gemm::kernel::DefaultGemmUniversal<
      cutlass::int4b_t, RowMajor, cutlass::ComplexTransform::kNone, kAlignAB, cutlass::int4b_t, ColumnMajor,
      cutlass::ComplexTransform::kNone, kAlignAB, bf16, RowMajor, std::int32_t, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, typename Tile::Threadblock, typename Tile::Warp, InstructionShape,
      cutlass::epilogue::thread::LinearCombination<bf16, quant::detail::kAlignC, std::int32_t, float>, Swizzle,
      Tile::stages, cutlass::arch::OpMultiplyAddSaturate, cutlass::gemm::SharedMemoryClearOption::kNone, false, false,
      false, cutlass::layout::NoPermute, cutlass::layout::NoPermute, PermuteB>::GemmKernel;
  using Epilogue =
      cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<typename GemmBase::Epilogue, Callbacks, 1>;
  using Kernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmBase::Mma, Epilogue, Swizzle>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

struct Problem {
  int m = 0, n = 0, k = 0;
  std::int64_t ldx = 0;  // in int4 elements (CUTLASS's leading dimension)
};

Problem check_operands(const char* op, const Int4Operands& a);
void check_output(const char* op, const char* name, const void* p, std::int64_t ld, std::int64_t n_out);
void check_vector(const char* op, const char* name, const void* p, bool optional);
Int4Tile resolve_tile(Int4Tile tile, const Problem& p);
void check_status(cutlass::Status status, const char* op, const char* stage);

template <class Gemm>
void launch(const char* op, const Problem& p, const Int4Operands& a,
            const typename Gemm::GemmKernel::FusionCallbacks::Arguments& cb, cudaStream_t stream) {
  const auto* x = reinterpret_cast<const cutlass::int4b_t*>(a.x_q);
  const auto* w = reinterpret_cast<const cutlass::int4b_t*>(a.w);
  const typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {p.m, p.n, p.k}, /*batch_count=*/1, cb,
                                      x, w, /*ptr_C=*/nullptr, /*ptr_D=*/nullptr, 0, 0, 0, 0, p.ldx, p.k, 0, 0);
  Gemm gemm;
  check_status(Gemm::can_implement(args), op, "can_implement");
  if (Gemm::get_workspace_size(args) != 0) check_status(cutlass::Status::kErrorWorkspaceNull, op, "workspace");
  check_status(gemm.initialize(args, nullptr, stream), op, "initialize");
  check_status(gemm.run(stream), op, "run");
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::quant4::detail
