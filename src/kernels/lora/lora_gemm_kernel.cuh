// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
// Internal to src/kernels/lora: the LoRA-fused GEMM kernel of lora_gemm.h. LoraGemmKernel::run and VisitorKernel are
// adapted from CUTLASS's GemmWithEpilogueVisitor and DefaultGemmWithVisitor (cutlass/gemm/kernel/
// gemm_universal_with_visitor.h, default_gemm_universal_with_visitor.h, Copyright (c) 2023 - 2026 NVIDIA CORPORATION &
// AFFILIATES, BSD-3-Clause; see THIRD_PARTY_NOTICES.md), with a second mainloop: base mainloop (bf16 or INT8
// MmaMultistage) -> accumulators converted to fp32 (INT8: dequantized in registers) -> bf16 MmaMultistage over the
// slice's rank columns (up to two column ranges) into the same accumulators -> Sm80 visitor epilogue.
//
// The two mainloops share the accumulator fragment: both are MmaTensorOp warp tiles of the same M x N with m16n8
// instructions, whose fragment holds instruction tile (m, n) at [(m + n * kRowIterations) * 4, + 4) with the PTX m16n8
// C layout (c0, c1 at row lane / 4, columns 2 (lane % 4) + {0, 1}; c2, c3 eight rows below) for s32 and f32 alike
// (cutlass/gemm/warp/mma_tensor_op.h). The INT8 dequantization uses that map to find each element's row and column.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <type_traits>

#include "kernels/lora/lora_gemm.h"
// INT8 tiles, Int8Kernel, dequant, check_operands; through it the visitor GEMM headers and kernels/fused's internal
// header (bf16 tiles, situ_value, check_problem, tile_for).
#include "kernels/quant/int8_gemm_impl.cuh"

namespace mi::kernels::lora::detail {

using bf16 = cutlass::bfloat16_t;
using kernels::detail::add;
using kernels::detail::mul;
using kernels::detail::rbf;

// DefaultGemmUniversal's bf16 mainloop (the MmaMultistage of the kernels/fused GEMMs, same template arguments) with an
// epilogue visitor tree, without the stream-K selection.
template <class Tile, class Callbacks, class PermuteB = cutlass::layout::NoPermute>
struct VisitorKernel {
  using GemmBase = typename cutlass::gemm::kernel::DefaultGemmUniversal<
      bf16, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8, bf16, cutlass::layout::ColumnMajor,
      cutlass::ComplexTransform::kNone, 8, bf16, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, typename Tile::Threadblock, typename Tile::Warp, fused::detail::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>, fused::detail::Swizzle, Tile::stages,
      cutlass::arch::OpMultiplyAdd, cutlass::gemm::SharedMemoryClearOption::kNone, false, false, false,
      cutlass::layout::NoPermute, cutlass::layout::NoPermute, PermuteB>::GemmKernel;
  using Epilogue =
      cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<typename GemmBase::Epilogue, Callbacks, 1>;
};

// ---- Epilogue visitor trees on the fp32 accumulator (already the GEMM output y, plus the LoRA term) ----

template <class T>
struct IdentityFn {  // out = bf16(acc): the node's output conversion rounds
  CUTLASS_DEVICE T operator()(T const& acc) const { return acc; }
};

template <class T>
struct GatedResidualFn {  // residual + bf16(bf16(acc) * gate), kernels::gated_residual's order
  CUTLASS_DEVICE T operator()(T const& acc, T const& gate, T const& residual) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) out[i] = add(residual[i], rbf(mul(rbf(acc[i]), gate[i])));
    return out;
  }
};

// SiTU-GLU on 8 accumulator columns (4 hidden + their 4 gates, the permuted B) -> 4 bf16 as 8 bytes.
struct SituGluNode : cutlass::epilogue::threadblock::VisitorImpl2x<> {
  using VisitorImpl2x<>::VisitorImpl2x;
  struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
    template <class ElementAccumulator, int kFrag>
    CUTLASS_DEVICE cutlass::Array<std::uint8_t, kFrag> visit(int, int, int, int,
                                                            cutlass::Array<ElementAccumulator, kFrag> const&,
                                                            cutlass::Array<ElementAccumulator, kFrag> const& acc) {
      constexpr int kG = fused::detail::kUpGateGroup;
      static_assert(kFrag == 2 * kG, "SituGluNode assumes accesses of 4 hidden + 4 gate columns");
      cutlass::Array<std::uint8_t, kFrag> out;
      auto* words = reinterpret_cast<std::uint16_t*>(&out);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kG; ++i) {
        words[i] = fused::detail::bf16_bits_rn(fused::detail::situ_value(rbf(static_cast<float>(acc[i])),
                                                                         rbf(static_cast<float>(acc[kG + i]))));
      }
      return out;
    }
  };
  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord, int, ProblemShape) {
    return Callbacks();
  }
};

template <class Threadblock, class Warp>
struct Trees {
  template <class... Ts>
  using EVT = cutlass::epilogue::threadblock::Sm80EVT<Ts...>;
  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<Threadblock, Warp, bf16, 8, 1>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using StrideD = cute::Stride<std::int64_t, cute::_1, std::int64_t>;
  using Aux = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, bf16, StrideD>;
  using Gate = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, bf16,
                                                                   cute::Stride<cute::_0, cute::_1, std::int32_t>>;
  template <class Element>
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, Element,
                                                                cutlass::FloatRoundStyle::round_to_nearest, StrideD>;
  template <template <class> class Fn>
  using Compute = cutlass::epilogue::threadblock::VisitorCompute<Fn, bf16, float,
                                                                 cutlass::FloatRoundStyle::round_to_nearest>;
  using Plain = EVT<Store<bf16>, EVT<Compute<IdentityFn>, Accum>>;
  using GatedResidual = EVT<Store<bf16>, EVT<Compute<GatedResidualFn>, Accum, Gate, Aux>>;
  using SituGlu = EVT<Store<std::uint8_t>, EVT<SituGluNode, Accum>>;
};

// ---- Accumulator conversion between the mainloops ----

struct NoDequant {};  // bf16 base: the fp32 accumulators carry over

struct Dequant {  // INT8 base: ((float)acc * x_scale[r]) * w_scale[c'] (+ bias[c']), c' the weight row of column c
  const float* x_scale = nullptr;
  const float* w_scale = nullptr;
  const bf16* bias = nullptr;
  int up_gate_half = 0;  // > 0: columns read weight rows up_gate_row(c, half) (SiTU-GLU)
};

// Mainloop operands of the LoRA term for one CTA: t columns [col0, col0 + cols) then [col0b, col0b + colsb) against
// b's columns [0, cols) then [cols, cols + colsb).
struct SliceParams {
  int col0 = 0, cols = 0, col0b = 0, colsb = 0, row0 = 0, rows = 0;
  const bf16* b = nullptr;
};

template <class MmaMain, class MmaLora, class Epilogue, class Swizzle, class Convert>
struct LoraGemmKernel {
  using FusionCallbacks = typename Epilogue::FusionCallbacks;
  static constexpr int kThreadCount = MmaMain::WarpCount::kCount * 32;
  static constexpr int kTileM = MmaMain::Shape::kM, kTileN = MmaMain::Shape::kN;
  static_assert(MmaLora::WarpCount::kCount * 32 == kThreadCount, "mainloops must use the same threads");
  static_assert(MmaMain::Shape::kM == MmaLora::Shape::kM && MmaMain::Shape::kN == MmaLora::Shape::kN, "tile mismatch");
  using WarpShape = typename MmaLora::Operator::Shape;
  static_assert(MmaMain::Operator::Shape::kM == WarpShape::kM && MmaMain::Operator::Shape::kN == WarpShape::kN,
                "warp tile mismatch");
  static_assert(int(MmaMain::FragmentC::kElements) == int(MmaLora::FragmentC::kElements), "accumulator mismatch");
  static_assert(MmaLora::WarpCount::kK == 1 && MmaMain::WarpCount::kK == 1, "no intra-CTA split-K");

  union SharedStorage {
    typename MmaMain::SharedStorage main;
    typename MmaLora::SharedStorage lora;
    typename Epilogue::SharedStorage epilogue;
  };

  struct Params {
    cutlass::gemm::GemmCoord problem;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile = 0;
    typename MmaMain::IteratorA::Params pa;
    typename MmaMain::IteratorB::Params pb;
    const void* a = nullptr;
    const void* b = nullptr;
    typename MmaLora::IteratorA::Params pt;
    typename MmaLora::IteratorB::Params pl[LoraSlices::kMaxSlices];
    const bf16* t = nullptr;
    int slices = 0;
    SliceParams slice[LoraSlices::kMaxSlices];
    Convert convert;
    typename FusionCallbacks::Params output_op;
    cute::Shape<int32_t, int32_t, int32_t> problem_shape;
  };

  CUTLASS_DEVICE static void convert(const Params& p, typename MmaMain::FragmentC const& in,
                                     typename MmaLora::FragmentC& out, cutlass::gemm::GemmCoord tile, int warp_idx,
                                     int lane) {
    if constexpr (std::is_same_v<Convert, NoDequant>) {
      out = in;
    } else {
      constexpr int kRowIt = WarpShape::kM / 16, kColIt = WarpShape::kN / 8;
      const int warp_m = warp_idx % MmaLora::WarpCount::kM, warp_n = warp_idx / MmaLora::WarpCount::kM;
      const int row_base = tile.m() * MmaLora::Shape::kM + warp_m * WarpShape::kM + lane / 4;
      const int col_base = tile.n() * MmaLora::Shape::kN + warp_n * WarpShape::kN + 2 * (lane % 4);
      const Dequant& d = p.convert;
      float xs[kRowIt][2], ws[kColIt][2], bs[kColIt][2];
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < kRowIt; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          const int r = row_base + m * 16 + h * 8;
          xs[m][h] = r < p.problem.m() ? d.x_scale[r] : 0.0f;
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kColIt; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          const int c = col_base + n * 8 + h;
          const int w = c < p.problem.n() ? (d.up_gate_half > 0 ? quant::detail::up_gate_row(c, d.up_gate_half) : c) : -1;
          ws[n][h] = w >= 0 ? d.w_scale[w] : 0.0f;
          bs[n][h] = w >= 0 && d.bias != nullptr ? static_cast<float>(d.bias[w]) : 0.0f;
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < kColIt; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < kRowIt; ++m) {
          const int base = (m + n * kRowIt) * 4;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < 4; ++i) {
            out[base + i] = quant::detail::dequant(__int2float_rn(in[base + i]), xs[m][i / 2], ws[n][i % 2], bs[n][i % 2]);
          }
        }
      }
    }
  }

  CUTLASS_DEVICE static void run(const Params& p, SharedStorage& smem) {
    Swizzle swizzle;
    const cutlass::gemm::GemmCoord tile = swizzle.get_tile_offset(p.swizzle_log_tile);
    if (p.grid_tiled_shape.m() <= tile.m() || p.grid_tiled_shape.n() <= tile.n()) return;
    const int thread_idx = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane = threadIdx.x % 32;
    const int m0 = tile.m() * MmaMain::Shape::kM, n0 = tile.n() * MmaMain::Shape::kN;

    typename MmaLora::FragmentC acc;
    {
      using ElementA = typename MmaMain::IteratorA::Element;
      using ElementB = typename MmaMain::IteratorB::Element;
      typename MmaMain::IteratorA it_a(p.pa, const_cast<ElementA*>(static_cast<const ElementA*>(p.a)),
                                       {p.problem.m(), p.problem.k()}, thread_idx, {m0, 0});
      typename MmaMain::IteratorB it_b(p.pb, const_cast<ElementB*>(static_cast<const ElementB*>(p.b)),
                                       {p.problem.k(), p.problem.n()}, thread_idx, {0, n0});
      MmaMain mma(smem.main, thread_idx, warp_idx, lane);
      typename MmaMain::FragmentC acc_main;
      acc_main.clear();
      const int iters = (p.problem.k() + MmaMain::Shape::kK - 1) / MmaMain::Shape::kK;
      mma(iters, acc_main, it_a, it_b, acc_main);
      convert(p, acc_main, acc, tile, warp_idx, lane);
    }
    int s = -1;
    for (int i = 0; i < p.slices; ++i) {
      if (n0 >= p.slice[i].row0 && n0 < p.slice[i].row0 + p.slice[i].rows) s = i;
    }
    if (s >= 0) {
      const SliceParams& sl = p.slice[s];
      for (int range = 0; range < 2; ++range) {
        const int t0 = range == 0 ? sl.col0 : sl.col0b, cols = range == 0 ? sl.cols : sl.colsb, b0 = range == 0 ? 0 : sl.cols;
        if (cols == 0) continue;
        __syncthreads();  // the shared memory of the previous mainloop is reused
        typename MmaLora::IteratorA it_t(p.pt, const_cast<bf16*>(p.t), {p.problem.m(), t0 + cols}, thread_idx, {m0, t0});
        typename MmaLora::IteratorB it_l(p.pl[s], const_cast<bf16*>(sl.b), {b0 + cols, sl.rows}, thread_idx,
                                         {b0, n0 - sl.row0});
        MmaLora mma(smem.lora, thread_idx, warp_idx, lane);
        const int iters = (cols + MmaLora::Shape::kK - 1) / MmaLora::Shape::kK;
        mma(iters, acc, it_t, it_l, acc);
      }
    }
    __syncthreads();
    Epilogue epilogue(p.output_op, smem.epilogue, thread_idx, warp_idx, lane);
    epilogue(acc, tile, p.problem_shape, thread_idx);
  }
};

template <class K>
__global__ void __launch_bounds__(K::kThreadCount) lora_gemm_kernel(const typename K::Params p) {
  extern __shared__ int smem_raw[];
  K::run(p, *reinterpret_cast<typename K::SharedStorage*>(smem_raw));
}

// Mainloop and epilogue types: the bf16 LoRA mainloop for a tile shape (M, N) and warp (WM, WN). K = 64 in 2 stages:
// a rank-64 slice is one pipeline step (the mainloop runs once per CTA after the base, so its fill latency is exposed).
template <int kM, int kN, int kWM, int kWN>
using LoraTile = fused::detail::TileCfg<kM, kN, 64, kWM, kWN, 2>;

template <class Tile, class Tree>
struct LoraParts {  // the LoRA (bf16, no permutation) mainloop and the epilogue with tree `Tree`
  using Base = VisitorKernel<Tile, Tree>;
  using Mma = typename Base::GemmBase::Mma;
  using Epilogue = typename Base::Epilogue;
};

// The LoRA mainloop tile for each base tile: same M x N and warp tile.
template <class T>
struct LoraTileFor;
template <>
struct LoraTileFor<quant::detail::Tile128x128x64> { using type = LoraTile<128, 128, 64, 64>; };
template <>
struct LoraTileFor<quant::detail::Tile128x256x64> { using type = LoraTile<128, 256, 64, 64>; };
template <>
struct LoraTileFor<quant::detail::Tile256x128x64> { using type = LoraTile<256, 128, 64, 64>; };
template <>
struct LoraTileFor<quant::detail::Tile64x128x64> { using type = LoraTile<64, 128, 32, 64>; };
template <>
struct LoraTileFor<quant::detail::Tile128x128x128> { using type = LoraTile<128, 128, 64, 64>; };
template <>
struct LoraTileFor<fused::detail::Tile128x256> { using type = LoraTile<128, 256, 64, 64>; };
template <>
struct LoraTileFor<fused::detail::Tile256x128> { using type = LoraTile<256, 128, 64, 64>; };
template <>
struct LoraTileFor<fused::detail::Tile128x128> { using type = LoraTile<128, 128, 64, 64>; };

// Kernels per base. bf16: the fused_gemm.h tile for both mainloops. INT8: Int8Kernel's mainloop + LoraTileFor's.
template <class Tile, template <class, class> class TreeOf, class PermuteB = cutlass::layout::NoPermute>
struct Bf16LoraKernel {
  using LTile = typename LoraTileFor<Tile>::type;
  using Tree = TreeOf<typename LTile::Threadblock, typename LTile::Warp>;
  using MmaMain = typename VisitorKernel<Tile, Tree, PermuteB>::GemmBase::Mma;
  using Parts = LoraParts<LTile, Tree>;
  using Kernel = LoraGemmKernel<MmaMain, typename Parts::Mma, typename Parts::Epilogue, fused::detail::Swizzle, NoDequant>;
};
template <class Tile, template <class, class> class TreeOf, class PermuteB = cutlass::layout::NoPermute>
struct Int8LoraKernel {
  using LTile = typename LoraTileFor<Tile>::type;
  using Tree = TreeOf<typename LTile::Threadblock, typename LTile::Warp>;
  using MmaMain = typename quant::detail::Int8Kernel<Tile, typename quant::detail::Nodes<Tile>::Plain, PermuteB>::GemmBase::Mma;
  using Parts = LoraParts<LTile, Tree>;
  using Kernel = LoraGemmKernel<MmaMain, typename Parts::Mma, typename Parts::Epilogue, fused::detail::Swizzle, Dequant>;
};

template <class TB, class W>
using PlainTree = typename Trees<TB, W>::Plain;
template <class TB, class W>
using GatedResidualTree = typename Trees<TB, W>::GatedResidual;
template <class TB, class W>
using SituGluTree = typename Trees<TB, W>::SituGlu;

// Validates the slices against an n-column GEMM and m rows; fills params.
template <class K>
void set_slices(const char* op, const LoraSlices& l, std::int64_t m, std::int64_t n, typename K::Params& p) {
  const auto bad = [&](const std::string& msg) { kernels::detail::fail(std::string("lora::") + op, msg); };
  if (l.t == nullptr || reinterpret_cast<std::uintptr_t>(l.t) % 16 != 0 || l.ldt <= 0 || l.ldt % 8 != 0) {
    bad("t must be 16-byte aligned with a row stride that is a positive multiple of 8");
  }
  if (l.count < 1 || l.count > LoraSlices::kMaxSlices) bad("1 to 3 slices");
  if (m > INT32_MAX || l.ldt > INT32_MAX) bad("dimensions exceed int32");
  std::int64_t prev_end = 0;
  const auto range_ok = [&](std::int64_t c0, std::int64_t c, bool optional) {
    if (optional && c == 0) return true;
    return c > 0 && c % 8 == 0 && c0 >= 0 && c0 % 8 == 0 && c0 + c <= l.ldt;
  };
  for (int i = 0; i < l.count; ++i) {
    const LoraSlices::Slice& s = l.slice[i];
    const bool cols_ok = range_ok(s.col0, s.cols, false) && range_ok(s.col0b, s.colsb, true) &&
                         s.cols + s.colsb <= INT32_MAX;
    const bool rows_ok = s.row0 >= prev_end && s.rows > 0 && s.row0 % 256 == 0 &&
                         (s.rows % 256 == 0 || s.row0 + s.rows == n) && s.row0 + s.rows <= n;
    if (!cols_ok || !rows_ok || s.b == nullptr || reinterpret_cast<std::uintptr_t>(s.b) % 16 != 0) {
      bad("slice " + std::to_string(i) + " invalid (t columns [" + std::to_string(s.col0) + ", +" + std::to_string(s.cols) +
          ") and [" + std::to_string(s.col0b) + ", +" + std::to_string(s.colsb) + "), output columns [" + std::to_string(s.row0) +
          ", +" + std::to_string(s.rows) + ") of " + std::to_string(n) +
          "): columns multiples of 8 within ldt, output ranges ascending, 256-aligned, 16-byte aligned b");
    }
    prev_end = s.row0 + s.rows;
    p.slice[i] = {static_cast<int>(s.col0), static_cast<int>(s.cols), static_cast<int>(s.col0b), static_cast<int>(s.colsb),
                  static_cast<int>(s.row0), static_cast<int>(s.rows), reinterpret_cast<const bf16*>(s.b)};
    using PL = std::remove_reference_t<decltype(p.pl[0])>;
    p.pl[i] = PL(cutlass::layout::ColumnMajor(static_cast<int>(s.cols + s.colsb)));
  }
  p.slices = l.count;
  p.t = reinterpret_cast<const bf16*>(l.t);
  using PT = std::remove_reference_t<decltype(p.pt)>;
  p.pt = PT(cutlass::layout::RowMajor(static_cast<int>(l.ldt)));
}

template <class K>
void launch(const char* op, typename K::Params& p, cudaStream_t stream) {
  using Swizzle = typename fused::detail::Swizzle;
  Swizzle swizzle;
  p.grid_tiled_shape = swizzle.get_tiled_shape(p.problem, {K::kTileM, K::kTileN, 1}, 1);
  p.swizzle_log_tile = swizzle.get_log_tile(p.grid_tiled_shape);
  p.problem_shape = {p.problem.m(), p.problem.n(), 1};
  const dim3 grid = swizzle.get_grid_shape(p.grid_tiled_shape);
  constexpr int kSmem = static_cast<int>(sizeof(typename K::SharedStorage));
  static_assert(kSmem <= 99 * 1024, "shared memory exceeds the sm_86/89/120 opt-in limit");
  if constexpr (kSmem >= 48 * 1024) {
    MI_CUDA_CHECK(cudaFuncSetAttribute(lora_gemm_kernel<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  }
  lora_gemm_kernel<K><<<grid, K::kThreadCount, kSmem, stream>>>(p);
  static_cast<void>(op);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::lora::detail
