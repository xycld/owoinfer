// Internal to src/kernels/nvfp4: the CUTLASS 3.x SM120 block-scaled NVFP4 GEMM types, their epilogue visitor trees
// and the launch, shared by the gemm_sm120a_<epilogue>_<tile>.cu translation units (one instantiation each, so they
// compile in parallel; each needs ~5.5 GB of RAM). Compiled for sm_120a only (cmake/modules/nvfp4.cmake): plain
// sm_120 cannot assemble the block-scaled MMA.
//
// The kernel composition (collective builders, GemmUniversal, GemmUniversalAdapter, the scale-factor layouts) follows
// CUTLASS v4.7.1 examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu (Copyright (c) 2025 -
// 2026 NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause; see THIRD_PARTY_NOTICES.md). Changes: our Sm90 EVT epilogues
// (nvfp4.h: per-row x scale, per-tensor w scale, optional bias, LoRA delta, gated residual in place) in place of the
// linear combination, the Pingpong schedule as a third configuration, no workspace (graph-capturable), and the kernel
// is launched through gemm_kernel below instead of GemmUniversalAdapter's cutlass::device_kernel.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "core/cuda_check.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "kernels/basic/detail.h"
#include "kernels/nvfp4/gemm_sm120a.h"

#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#error "src/kernels/nvfp4/gemm_sm120a_*.cu need CUDA >= 12.8 (CUTLASS_ARCH_MMA_SM120_SUPPORTED)"
#endif

namespace mi::kernels::nvfp4::detail {

namespace fusion = cutlass::epilogue::fusion;
using bf16 = cutlass::bfloat16_t;
using ElementAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int kAlignAB = 32;  // e2m1 elements per 16-byte access
constexpr int kAlignD = 8;    // bf16 elements per 16-byte access
constexpr auto kRn = cutlass::FloatRoundStyle::round_to_nearest;

// Threadblock tile and mainloop schedule of each Nvfp4Tile. KernelScheduleAuto is the Cooperative schedule (two
// consumer warp groups on one tile); Pingpong alternates two tiles between the consumer warp groups.
template <Nvfp4Tile kTile>
struct TileConfig;
template <>
struct TileConfig<Nvfp4Tile::k128x128x128> {
  using Shape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using Schedule = cutlass::gemm::collective::KernelScheduleAuto;
};
template <>
struct TileConfig<Nvfp4Tile::k128x128x128Pingpong> {
  using Shape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using Schedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
};
template <>
struct TileConfig<Nvfp4Tile::k256x128x128> {
  using Shape = cute::Shape<cute::_256, cute::_128, cute::_128>;
  using Schedule = cutlass::gemm::collective::KernelScheduleAuto;
};

// ---- Epilogue compute nodes: one fused node per epilogue, one fp32 rounding per operation (_rn: no FMA contraction).
// Sm90Compute converts every input to float (bf16 -> float is exact) and rounds the result to bf16 (round to nearest).

using kernels::detail::add;
using kernels::detail::mul;
using kernels::detail::rbf;

// The INT8 epilogue's dequantization (kernels/quant int8_gemm_impl.cuh dequant) with g for s_x and s2 for s_w.
__device__ __forceinline__ float dequant(float acc, float xs, float ws) { return mul(mul(acc, xs), ws); }

template <class T>
struct PlainFn {  // E1: dequant + bias
  CUTLASS_DEVICE T operator()(T const& acc, T const& xs, T const& ws, T const& bias) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) out[i] = add(dequant(acc[i], xs[i], ws[i]), bias[i]);
    return out;
  }
};

template <class T>
struct PlainDeltaFn {  // E1L: (dequant + delta) + bias
  CUTLASS_DEVICE T operator()(T const& acc, T const& xs, T const& ws, T const& delta, T const& bias) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) out[i] = add(add(dequant(acc[i], xs[i], ws[i]), delta[i]), bias[i]);
    return out;
  }
};

template <class T>
struct GatedResidualFn {  // E2: residual + bf16(bf16(dequant + bias) * gate), kernels::gated_residual's order
  CUTLASS_DEVICE T operator()(T const& acc, T const& xs, T const& ws, T const& bias, T const& gate,
                              T const& residual) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) {
      out[i] = add(residual[i], rbf(mul(rbf(add(dequant(acc[i], xs[i], ws[i]), bias[i])), gate[i])));
    }
    return out;
  }
};

template <class T>
struct GatedResidualDeltaFn {  // E2L: E2 on the E1L value
  CUTLASS_DEVICE T operator()(T const& acc, T const& xs, T const& ws, T const& delta, T const& bias, T const& gate,
                              T const& residual) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; ++i) {
      const float y = rbf(add(add(dequant(acc[i], xs[i], ws[i]), delta[i]), bias[i]));
      out[i] = add(residual[i], rbf(mul(y, gate[i])));
    }
    return out;
  }
};

// Visitor nodes and trees of one tile shape. Children in the compute functor's argument order.
template <class Tile, Epi kEpi>
struct EpilogueTree {
  using XScale = fusion::Sm90ColBroadcast<0, Tile, float, float, cute::Stride<cute::_1, cute::_0, std::int64_t>>;
  using WScale = fusion::Sm90ScalarBroadcast<float>;
  using RowVec = fusion::Sm90RowBroadcast<0, Tile, bf16, float, cute::Stride<cute::_0, cute::_1, std::int64_t>>;
  // The delta, read straight from global memory into the accumulator layout's registers (no shared-memory stage).
  using Delta = fusion::Sm90AuxLoad<0, Tile, bf16, cute::Stride<std::int64_t, cute::_1, std::int64_t>, void, void,
                                    kAlignD, false>;
  using Residual = fusion::Sm90SrcFetch<bf16>;  // C = the residual, D = the same buffer
  template <template <class> class Fn>
  using Compute = fusion::Sm90Compute<Fn, bf16, float, kRn>;

  static constexpr bool kResidual = kEpi == Epi::kGatedResidual || kEpi == Epi::kGatedResidualDelta;
  using ElementC = cute::conditional_t<kResidual, bf16, void>;
  using Tree = cute::conditional_t<
      kEpi == Epi::kPlain, fusion::Sm90EVT<Compute<PlainFn>, fusion::Sm90AccFetch, XScale, WScale, RowVec>,
      cute::conditional_t<
          kEpi == Epi::kPlainDelta,
          fusion::Sm90EVT<Compute<PlainDeltaFn>, fusion::Sm90AccFetch, XScale, WScale, Delta, RowVec>,
          cute::conditional_t<
              kEpi == Epi::kGatedResidual,
              fusion::Sm90EVT<Compute<GatedResidualFn>, fusion::Sm90AccFetch, XScale, WScale, RowVec, RowVec, Residual>,
              fusion::Sm90EVT<Compute<GatedResidualDeltaFn>, fusion::Sm90AccFetch, XScale, WScale, Delta, RowVec,
                              RowVec, Residual>>>>;
};

template <Nvfp4Tile kTile, Epi kEpi>
struct GemmType {
  using Tile = typename TileConfig<kTile>::Shape;
  using Tree = EpilogueTree<Tile, kEpi>;
  using Arch = cutlass::arch::Sm120;
  using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;  // GeForce Blackwell: no multicast, cluster 1x1x1
  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;  // w [N, K] row-major is B [K, N] column-major (TN only)
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      typename Tree::ElementC, RowMajor, kAlignD, bf16, RowMajor, kAlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, typename Tree::Tree>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass, ElementAB, RowMajor, kAlignAB, ElementAB, ColumnMajor, kAlignAB, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      typename TileConfig<kTile>::Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// The kernel's parameter, Kernel::Params, as bytes. Params holds TMA descriptors (CUtensorMap), which CUDA 13 declares
// alignas(128), and MSVC x64 rejects a by-value parameter aligned that much (C2719) in the host launch stub nvcc
// generates for cutlass::device_kernel<Kernel>(Params). Only the device pass gives this type Params' alignment; the
// host pass (the stub) sees a plain byte array of the same size, which a launch copies byte for byte either way. One
// parameter at offset 0 with Params' size and alignment: the device code is cutlass::device_kernel's.
template <class Kernel>
struct KernelParams {
#if defined(__CUDA_ARCH__)
  alignas(typename Kernel::Params)
#endif
      unsigned char bytes[sizeof(typename Kernel::Params)];
};

template <class Kernel>
__global__ void __launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
    gemm_kernel(CUTLASS_GRID_CONSTANT const KernelParams<Kernel> params) {
  extern __shared__ char smem[];
  Kernel op;
  op(*reinterpret_cast<const typename Kernel::Params*>(params.bytes), smem);
  cutlass::arch::synclog_print();
}

inline void check_status(cutlass::Status status, const char* op, const char* stage) {
  if (status != cutlass::Status::kSuccess) {
    kernels::detail::fail(op, std::string("CUTLASS ") + stage + " failed: " + cutlassGetStatusString(status));
  }
}

// gemm_sm120a.h: each gemm_sm120a_<epilogue>_<tile>.cu instantiates one (tile, epilogue) pair explicitly.
template <Nvfp4Tile kTile, Epi kEpi>
void gemm(const GemmArgs& g, int sm_count, cudaStream_t stream) {
  constexpr const char* op = "nvfp4_gemm";
  using T = GemmType<kTile, kEpi>;
  using Gemm = typename T::Gemm;
  using Kernel = typename T::Kernel;
  using Mainloop = typename T::Mainloop;
  using Sf = typename Mainloop::Sm1xxBlkScaledConfig;
  using Data = typename ElementAB::DataType;
  using Scale = typename ElementAB::ScaleFactorType;
  const Nvfp4Operands& a = g.op;
  const int m = static_cast<int>(a.m), n = static_cast<int>(a.n), k = static_cast<int>(a.k);
  const auto shape = cute::make_shape(m, n, k, 1);

  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {m, n, k, 1};
  args.mainloop.ptr_A = reinterpret_cast<const Data*>(a.x);
  args.mainloop.dA = cutlass::make_cute_packed_stride(typename Mainloop::StrideA{}, {m, k, 1});
  args.mainloop.ptr_B = reinterpret_cast<const Data*>(a.w);
  args.mainloop.dB = cutlass::make_cute_packed_stride(typename Mainloop::StrideB{}, {n, k, 1});
  args.mainloop.ptr_SFA = reinterpret_cast<const Scale*>(a.x_sf);
  args.mainloop.layout_SFA = Sf::tile_atom_to_shape_SFA(shape);
  args.mainloop.ptr_SFB = reinterpret_cast<const Scale*>(a.w_sf);
  args.mainloop.layout_SFB = Sf::tile_atom_to_shape_SFB(shape);
  // EVT arguments: {children..., node}, children in the compute functor's order.
  const typename T::Tree::XScale::Arguments xs{a.x_scale, 0.0f, {}};
  const typename T::Tree::WScale::Arguments ws{{a.w_scale2}};
  const typename T::Tree::RowVec::Arguments bias{reinterpret_cast<const bf16*>(g.bias), bf16(0.0f), {}};
  const typename T::Tree::Delta::Arguments delta{reinterpret_cast<const bf16*>(g.delta.p), bf16(0.0f),
                                                 {g.delta.ld, cute::Int<1>{}, std::int64_t{0}}};
  auto& t = args.epilogue.thread;
  if constexpr (kEpi == Epi::kPlain) {
    t = {{}, xs, ws, bias, {}};
  } else if constexpr (kEpi == Epi::kPlainDelta) {
    t = {{}, xs, ws, delta, bias, {}};
  } else {
    const typename T::Tree::RowVec::Arguments gate{reinterpret_cast<const bf16*>(g.gate), bf16(0.0f), {}};
    if constexpr (kEpi == Epi::kGatedResidual) {
      t = {{}, xs, ws, bias, gate, {}, {}};
    } else {
      t = {{}, xs, ws, delta, bias, gate, {}, {}};
    }
  }
  const auto d_stride = cute::make_stride(g.ldo, cute::Int<1>{}, std::int64_t{0});
  if constexpr (T::Tree::kResidual) {
    args.epilogue.ptr_C = reinterpret_cast<const bf16*>(g.out);  // in place: each tile reads C before writing D
    args.epilogue.dC = d_stride;
  } else {
    args.epilogue.ptr_C = nullptr;
  }
  args.epilogue.ptr_D = reinterpret_cast<bf16*>(g.out);
  args.epilogue.dD = d_stride;
  MI_CUDA_CHECK(cudaGetDevice(&args.hw_info.device_id));
  args.hw_info.sm_count = sm_count;

  check_status(Gemm::can_implement(args), op, "can_implement");
  if (Gemm::get_workspace_size(args) != 0) check_status(cutlass::Status::kErrorWorkspaceNull, op, "workspace");
  // GemmUniversalAdapter::initialize() and run() for this kernel (static 1x1x1 cluster, no PDL: a plain launch).
  check_status(Kernel::initialize_workspace(args, nullptr, stream), op, "initialize");
  const typename Kernel::Params params = Kernel::to_underlying_arguments(args, nullptr);
  static_assert(sizeof(KernelParams<Kernel>) == sizeof(params));
  KernelParams<Kernel> kernel_params;
  std::memcpy(kernel_params.bytes, &params, sizeof(params));
  constexpr int smem = Kernel::SharedStorageSize;
  if constexpr (smem >= (48 << 10)) {
    MI_CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  }
  gemm_kernel<Kernel><<<Kernel::get_grid_shape(params), Kernel::get_block_shape(), smem, stream>>>(kernel_params);
  MI_CUDA_CHECK_LAUNCH();
}

template <Nvfp4Tile kTile, Epi kEpi>
cudaError_t probe() {
  cudaFuncAttributes attr{};
  const cudaError_t err = cudaFuncGetAttributes(&attr, gemm_kernel<typename GemmType<kTile, kEpi>::Kernel>);
  if (err != cudaSuccess) (void)cudaGetLastError();  // not sticky; leave no error behind for the next check
  return err;
}

}  // namespace mi::kernels::nvfp4::detail
