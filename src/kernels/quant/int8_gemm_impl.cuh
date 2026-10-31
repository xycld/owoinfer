// SPDX-License-Identifier: Apache-2.0
// Internal to src/kernels/quant: the CUTLASS 2.x INT8 GEMM kernel types, their epilogue visitor trees (EVT) and the
// launch helper shared by the int8_gemm*.cu translation units (one per epilogue, so they compile in parallel).
//
// Adapted from comfy-kitchen @dba9fac comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu (Copyright (c) 2025 Comfy
// Org, Apache-2.0): the Sm80 EVT DefaultGemmWithVisitor kernel with OpMultiplyAddSaturate, the 16x8x32 instruction,
// the acc * x_scale[row] * w_scale[col] (+ bias) visitor order and the tile list. Changes: a single fused compute node
// per epilogue with explicit _rn arithmetic (no FMA contraction), the gated-residual and SiTU-GLU epilogues, the
// permuted ffn_up_gate operand, and no stream-K (no workspace). This file is a modified version (Apache-2.0 §4(b)).
// UpGateScaleBroadcast below derives from CUTLASS's VisitorRowBroadcast (cutlass/epilogue/threadblock/fusion/
// visitor_load.hpp, Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause). See
// THIRD_PARTY_NOTICES.md.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/layout/permute.h"
#include "kernels/fused/fused_gemm_impl.cuh"  // situ_gate_factor, situ_hidden_factor, kUpGateGroup, UpGatePermuteB
// The visitor headers rely on the GEMM and epilogue headers above being included first.
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "kernels/basic/detail.h"
#include "kernels/quant/quant.h"

namespace mi::kernels::quant::detail {

using bf16 = cutlass::bfloat16_t;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;  // w [N, K] row-major is B [K, N] column-major
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;  // int8 mma.sync m16n8k32
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
constexpr int kAlignAB = 16;  // int8 elements per 16-byte access
constexpr int kAlignC = 8;    // bf16 elements per 16-byte epilogue access

// Tile configurations (threadblock, warp, stages); all fit the 99 KiB opt-in shared memory of sm_86/89/120.
template <int kM, int kN, int kK, int kWM, int kWN, int kStages>
struct TileCfg {
  using Threadblock = cutlass::gemm::GemmShape<kM, kN, kK>;
  using Warp = cutlass::gemm::GemmShape<kWM, kWN, kK>;
  static constexpr int stages = kStages;
};
using Tile128x128x64 = TileCfg<128, 128, 64, 64, 64, 4>;
using Tile128x256x64 = TileCfg<128, 256, 64, 64, 64, 3>;
using Tile256x128x64 = TileCfg<256, 128, 64, 64, 64, 3>;
using Tile64x128x64 = TileCfg<64, 128, 64, 32, 64, 4>;
using Tile128x128x128 = TileCfg<128, 128, 128, 64, 64, 3>;

// Calls f(Tile{}) with the configuration of `tile` (kAuto must be resolved first).
template <class F>
void with_tile(Int8Tile tile, F&& f) {
  switch (tile) {
    case Int8Tile::k128x256x64: f(Tile128x256x64{}); break;
    case Int8Tile::k256x128x64: f(Tile256x128x64{}); break;
    case Int8Tile::k64x128x64: f(Tile64x128x64{}); break;
    case Int8Tile::k128x128x128: f(Tile128x128x128{}); break;
    default: f(Tile128x128x64{}); break;
  }
}

using kernels::detail::add;
using kernels::detail::mul;
using kernels::detail::rbf;

// comfy-kitchen's dequantization: ((float)acc * x_scale) * w_scale (+ bias), one fp32 rounding per op.
__device__ __forceinline__ float dequant(float acc, float xs, float ws, float bias) {
  return add(mul(mul(acc, xs), ws), bias);
}

// MOSS _situ's factors on bf16 values and the ffn_up_gate column permutation of the bf16 fused GEMMs, reused as they
// are.
using fused::detail::situ_gate_factor;
using fused::detail::situ_hidden_factor;
using fused::detail::kUpGateGroup;
using fused::detail::UpGatePermuteB;

// EVT compute functors (VisitorCompute converts every input to float first: int32 acc rounds to nearest).
template <class T>
struct DequantFn {  // int8_gemm: y = dequant(acc) + bias, rounded to bf16 by the node's output conversion
  CUTLASS_DEVICE T operator()(T const& acc, T const& xs, T const& ws, T const& bias) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) out[i] = dequant(acc[i], xs[i], ws[i], bias[i]);
    return out;
  }
};

template <class T>
struct GatedResidualFn {  // residual + bf16(bf16(y) * gate), kernels::gated_residual's order on the bf16 linear
  CUTLASS_DEVICE T operator()(T const& acc, T const& xs, T const& ws, T const& bias, T const& gate,
                              T const& residual) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) {
      out[i] = add(residual[i], rbf(mul(rbf(dequant(acc[i], xs[i], ws[i], bias[i])), gate[i])));
    }
    return out;
  }
};

// Weight row read by GEMM column n of int8_gemm_situ_glu (half = F): ((n / G) % 2) * F + (n / 2G) * G + n % G.
// Source of truth: kernels/fused UpGatePermuteB::operator(), which computes it inline (not exposed); keep in sync.
CUTLASS_HOST_DEVICE int up_gate_row(int n, int half) {
  return ((n / kUpGateGroup) % 2) * half + (n / (2 * kUpGateGroup)) * kUpGateGroup + n % kUpGateGroup;
}

// SiTU's two bf16 factors (fused::detail::situ_gate_factor / situ_hidden_factor) tabulated for every bf16 input,
// indexed by the input's bits: computed on the device by the same code (int8_gemm_situ.cu), so a lookup is the
// computation bit for bit, NaN and inf inputs included. The epilogue's transcendentals (two tanhf, an expf and an
// IEEE division per output) otherwise serialize after the main loop: ~12 % of the GEMM's time on sm_89.
struct SituFactorTables {
  // No defaults; SituGluNode::to_underlying_arguments throws on a null table (an argument list that leaves them out).
  const std::uint16_t* gate;    // [65536] bf16 bits of situ_gate_factor(g)
  const std::uint16_t* hidden;  // [65536] bf16 bits of situ_hidden_factor(h)
};

// The tables (static device memory of int8_gemm_situ.cu), filled on `stream` the first time per device (int8_gemm_situ.cu
// says how): before every launch of a SituGlu epilogue (int8_gemm_situ_glu, quant4::int4_gemm_situ_glu).
SituFactorTables situ_factor_tables(cudaStream_t stream);

// A bf16-representable float's bits (exact: its low 16 bits are zero) and back.
__device__ __forceinline__ std::uint32_t bf16_index(float v) { return __float_as_uint(v) >> 16; }
__device__ __forceinline__ float bf16_entry(const std::uint16_t* table, std::uint32_t i) {
  return __uint_as_float(static_cast<std::uint32_t>(__ldg(table + i)) << 16);
}

// The int8_gemm_situ_glu epilogue node: an access of 8 accumulator columns [8j, 8j + 8) = hidden 4j..4j+3 then gate
// 4j..4j+3 gives 4 bf16 results, returned as their 8 bytes (the store writes them at byte column 8j of the output
// declared as bytes, i.e. bf16 column 4j of [M, F]). situ_value(h, g) = gate factor * hidden factor, from the tables.
struct SituGluNode {
  using Arguments = SituFactorTables;
  using Params = Arguments;
  struct SharedStorage {};

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    if (args.gate == nullptr || args.hidden == nullptr) {
      throw std::invalid_argument("mi::kernels::quant: the SiTU-GLU epilogue needs situ_factor_tables()");
    }
    return args;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  CUTLASS_HOST_DEVICE SituGluNode() {}
  CUTLASS_HOST_DEVICE SituGluNode(Params const& params, SharedStorage const&) : params_ptr(&params) {}
  Params const* params_ptr = nullptr;

  struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
    SituFactorTables t;
    template <class ElementAccumulator, class ElementXs, class ElementWs, int kFrag>
    CUTLASS_DEVICE cutlass::Array<std::uint8_t, kFrag> visit(int, int, int, int,
                                                            cutlass::Array<ElementAccumulator, kFrag> const&,
                                                            cutlass::Array<ElementAccumulator, kFrag> const& acc,
                                                            cutlass::Array<ElementXs, kFrag> const& xs,
                                                            cutlass::Array<ElementWs, kFrag> const& ws) {
      static_assert(kFrag == 2 * kUpGateGroup, "SituGluNode assumes accesses of 4 hidden + 4 gate columns");
      cutlass::Array<std::uint8_t, kFrag> out;
      auto* words = reinterpret_cast<std::uint16_t*>(&out);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kUpGateGroup; ++i) {
        const int g = kUpGateGroup + i;
        // The GEMM output is a bf16 tensor in the reference: round both halves before SiTU.
        const float h = rbf(dequant(__int2float_rn(acc[i]), xs[i], ws[i], 0.0f));
        const float gt = rbf(dequant(__int2float_rn(acc[g]), xs[g], ws[g], 0.0f));
        const float v = mul(bf16_entry(t.gate, bf16_index(gt)), bf16_entry(t.hidden, bf16_index(h)));
        words[i] = __bfloat16_as_ushort(__float2bfloat16_rn(v));
      }
      return out;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord, int, ProblemShape) {
    Callbacks c;
    c.t = *params_ptr;
    return c;
  }
};

// VisitorRowBroadcast of w_scale [2F] read through int8_gemm_situ_glu's column permutation: each 4-float vector at
// logical column 4i holds rows up_gate_row(4i) .. + 3, which are contiguous.
template <class ThreadMap>
struct UpGateScaleBroadcast {
  struct Arguments {
    float const* ptr_row = nullptr;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }
  struct SharedStorage {};

  static constexpr int vec_bits = ThreadMap::kElementsPerAccess * 32;
  using VecType = cutlass::uint_bit_t<cute::min(128, vec_bits)>;
  static constexpr int VecLength = sizeof(VecType) / sizeof(float);
  static_assert(VecLength == kUpGateGroup, "a scale vector must cover exactly one hidden or gate group");

  CUTLASS_HOST_DEVICE UpGateScaleBroadcast() {}
  CUTLASS_HOST_DEVICE UpGateScaleBroadcast(Params const& params, SharedStorage const&) : params_ptr(&params) {}
  Params const* params_ptr = nullptr;

  template <class RTensor, class CTensor>
  struct Callbacks : cutlass::epilogue::threadblock::EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(RTensor&& r, CTensor&& c, int n_, Params const* p)
        : tC_rRow(cute::forward<RTensor>(r)), tC_cRow(cute::forward<CTensor>(c)), n(n_), params_ptr(p) {}
    RTensor tC_rRow;
    CTensor tC_cRow;
    int n;
    Params const* params_ptr;

    CUTLASS_DEVICE void begin_epilogue() {
      cute::clear(tC_rRow);
      auto coord_v = cute::filter(tC_cRow);
      auto dst_v = cute::filter(tC_rRow);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(dst_v); ++i) {
        const int col = cute::get<1>(coord_v(i));
        const bool guard = col < n;
        const float* src = params_ptr->ptr_row + (guard ? up_gate_row(col, n / 2) : 0);
        cutlass::arch::global_load<VecType, sizeof(VecType)>(dst_v(i), static_cast<void const*>(src), guard);
      }
    }

    template <class ElementAccumulator, int kFrag>
    CUTLASS_DEVICE auto visit(int, int, int column_idx, int, cutlass::Array<ElementAccumulator, kFrag> const&) {
      auto frg = cute::recast<cutlass::Array<float, kFrag>>(cute::coalesce(tC_rRow));
      return frg(column_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord tile_offset, int thread_idx, ProblemShape problem_shape) {
    using namespace cute;  // (mi::Tensor would shadow cute::Tensor: auto below)
    auto mRow = make_tensor(make_gmem_ptr(params_ptr->ptr_row), problem_shape, Stride<_0, _1, int32_t>{});
    auto tC_gRow = recast<VecType>(ThreadMap::partition(mRow, thread_idx, tile_offset))(_, _, _0{}, _0{}, _0{}, _0{});
    auto tC_rRow = make_tensor_like(tC_gRow);
    auto cRow = make_identity_tensor(mRow.shape());
    auto tC_cRow = outer_partition(ThreadMap::partition(cRow, thread_idx, tile_offset)(_, _, _0{}, _0{}, _0{}, _0{}),
                                     Shape<Int<VecLength>>{}, (_0{}));
    return Callbacks<decltype(tC_rRow), decltype(tC_cRow)>(cute::move(tC_rRow), cute::move(tC_cRow),
                                                           get<1>(problem_shape), params_ptr);
  }
};

// Order of the output tiles over the CTAs (the kernel's threadblock swizzle): bands of `band` adjacent N tiles; within
// a band the band's N tiles of one M tile are consecutive CTAs, then the next M tile's. The weight tiles of a band
// stay in L2 while the activation streams through once per band. band = 8 is GemmIdentityThreadblockSwizzle<8>'s
// order. The grid is (M tiles, N tiles), linearized x first (the order CTAs are launched in); the band travels in
// the params' swizzle_log_tile (Int8Gemm::set_band). The tile order never changes a result: every output tile is
// still one CTA's exact int32 sum and the same epilogue.
struct BandedSwizzle {
  using GemmCoord = cutlass::gemm::GemmCoord;
  CUTLASS_HOST_DEVICE BandedSwizzle() {}
  CUTLASS_HOST_DEVICE static GemmCoord get_tiled_shape(GemmCoord problem, GemmCoord tile, int split_k_slices) {
    return {(problem.m() + tile.m() - 1) / tile.m(), (problem.n() + tile.n() - 1) / tile.n(), split_k_slices};
  }
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(GemmCoord tiled) { return dim3(tiled.m(), tiled.n(), tiled.k()); }
  CUTLASS_HOST_DEVICE static int get_log_tile(GemmCoord) { return 8; }
  CUTLASS_DEVICE static GemmCoord get_tile_offset(int band) {
    const long long m_tiles = gridDim.x, n_tiles = gridDim.y;
    const long long id = blockIdx.y * m_tiles + blockIdx.x, per_band = m_tiles * band;
    const long long b = id / per_band, r = id - b * per_band;
    const long long width = min(static_cast<long long>(band), n_tiles - b * band);
    return {static_cast<int>(r / width), static_cast<int>(b * band + r % width), static_cast<int>(blockIdx.z)};
  }
};

// GemmUniversalAdapter without the operand transposition (never needed here: row-major D) plus the band setter.
template <class Kernel>
struct Int8Gemm : cutlass::gemm::device::GemmUniversalBase<Kernel> {
  void set_band(int band) { this->params_.swizzle_log_tile = band; }
};

// The kernel: DefaultGemmWithVisitor (default_gemm_universal_with_visitor.h) with a B-operand permutation, without
// the stream-K selection, in BandedSwizzle's tile order (GemmBase keeps CUTLASS's swizzle: only its Mma is used).
// ArchTag/InstrShape default to the sm_80 16x8x32 INT8 MMA; the sm_75 kernel family (src/kernels/sm75/int8_gemm_sm75.cu)
// instantiates the same template with arch::Sm75 and the 8x8x16 MMA.
template <class Tile, class Callbacks, class PermuteB = cutlass::layout::NoPermute,
          class ArchTag = cutlass::arch::Sm80, class InstrShape = cutlass::gemm::GemmShape<16, 8, 32>>
struct Int8Kernel {
  using GemmBase = typename cutlass::gemm::kernel::DefaultGemmUniversal<
      std::int8_t, RowMajor, cutlass::ComplexTransform::kNone, kAlignAB, std::int8_t, ColumnMajor,
      cutlass::ComplexTransform::kNone, kAlignAB, bf16, RowMajor, std::int32_t, cutlass::arch::OpClassTensorOp,
      ArchTag, typename Tile::Threadblock, typename Tile::Warp, InstrShape,
      cutlass::epilogue::thread::LinearCombination<bf16, kAlignC, std::int32_t, float>, Swizzle, Tile::stages,
      cutlass::arch::OpMultiplyAddSaturate, cutlass::gemm::SharedMemoryClearOption::kNone, false, false, false,
      cutlass::layout::NoPermute, cutlass::layout::NoPermute, PermuteB>::GemmKernel;
  using Epilogue =
      cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<typename GemmBase::Epilogue, Callbacks, 1>;
  using Kernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmBase::Mma, Epilogue, BandedSwizzle>;
  using Gemm = Int8Gemm<Kernel>;
};

// Visitor nodes of one tile configuration.
template <class Tile>
struct Nodes {
  template <class Op, class... Ts>  // Op named: MSVC cannot expand a pack into Sm80EVT's non-pack NodeOp
  using EVT = cutlass::epilogue::threadblock::Sm80EVT<Op, Ts...>;
  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<typename Tile::Threadblock,
                                                                           typename Tile::Warp, bf16, kAlignC, 1>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, float, cute::Stride<cute::_1, cute::_0, std::int32_t>>;
  template <class Element>
  using RowVec = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, Element, cute::Stride<cute::_0, cute::_1, std::int32_t>>;
  using StrideD = cute::Stride<std::int64_t, cute::_1, std::int64_t>;
  template <class Element>
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<ThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest, StrideD>;
  template <template <class> class Fn>
  using Compute = cutlass::epilogue::threadblock::VisitorCompute<Fn, bf16, float, cutlass::FloatRoundStyle::round_to_nearest>;

  using Plain = EVT<Store<bf16>, EVT<Compute<DequantFn>, Accum, XScale, RowVec<float>, RowVec<bf16>>>;
  using Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<ThreadMap, bf16, StrideD>;
  using GatedResidual = EVT<Store<bf16>, EVT<Compute<GatedResidualFn>, Accum, XScale, RowVec<float>, RowVec<bf16>,
                                             RowVec<bf16>, Residual>>;
  using SituGlu = EVT<Store<std::uint8_t>, EVT<SituGluNode, Accum, XScale, UpGateScaleBroadcast<ThreadMap>>>;
};

// Validated problem of one call (all dimensions fit int32).
struct Problem {
  int m = 0, n = 0, k = 0;
  std::int64_t ldx = 0;
};

// Checks the operands (see quant.h) and returns the problem.
Problem check_operands(const char* op, const Int8Operands& a);
// Checks an output / residual pointer with its row stride against n_out columns.
void check_output(const char* op, const char* name, const void* p, std::int64_t ld, std::int64_t n_out);
void check_vector(const char* op, const char* name, const void* p, bool optional);

Int8Tile resolve_tile(Int8Tile tile, const Problem& p);
// sm_75 runs one tile (src/kernels/sm75/int8_gemm_sm75.cu, 64x128x64): a forced tile names an sm_80 instantiation and
// is refused rather than ignored.
void check_sm75_tile(const char* op, Int8Tile tile);

void check_status(cutlass::Status status, const char* op, const char* stage);

// BandedSwizzle's band for an m x n x k problem in tiles of tile_n columns: 8 (the identity order) while the
// activation fits in three quarters of L2 (or 48 MiB on smaller L2s: comfy-kitchen's measured crossovers for its banded
// order), else as many N tiles as half of L2 holds, at least 8.
int int8_band(const Problem& p, int tile_n);

template <class Gemm>
void launch(const char* op, const Problem& p, const Int8Operands& a,
            const typename Gemm::GemmKernel::FusionCallbacks::Arguments& cb, cudaStream_t stream) {
  const typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {p.m, p.n, p.k}, /*batch_count=*/1, cb,
                                      a.x_q, a.w, /*ptr_C=*/nullptr, /*ptr_D=*/nullptr, 0, 0, 0, 0, p.ldx, p.k, 0, 0);
  Gemm gemm;
  check_status(Gemm::can_implement(args), op, "can_implement");
  // initialize gets no workspace (no allocation: graph-capturable). A real check, not assert(): the project's builds
  // define NDEBUG.
  if (Gemm::get_workspace_size(args) != 0) check_status(cutlass::Status::kErrorWorkspaceNull, op, "workspace");
  check_status(gemm.initialize(args, nullptr, stream), op, "initialize");
  gemm.set_band(int8_band(p, Gemm::GemmKernel::Mma::Shape::kN));
  check_status(gemm.run(stream), op, "run");
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::quant::detail
