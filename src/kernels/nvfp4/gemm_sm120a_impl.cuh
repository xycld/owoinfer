// Internal to src/kernels/nvfp4: the CUTLASS 3.x SM120 block-scaled NVFP4 GEMM type and its launch, shared by the
// gemm_sm120a_*.cu translation units (one tile each, so they compile in parallel; each needs ~5.5 GB of RAM).
// Compiled for sm_120a only (cmake/modules/nvfp4.cmake): plain sm_120 cannot assemble the block-scaled MMA.
//
// The kernel composition (collective builders, GemmUniversal, GemmUniversalAdapter, the scale-factor layouts) follows
// CUTLASS v4.7.1 examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu (Copyright (c) 2025 -
// 2026 NVIDIA CORPORATION & AFFILIATES, BSD-3-Clause; see THIRD_PARTY_NOTICES.md). Changes: our Sm90 EVT epilogue
// (per-row x scale, per-tensor w scale, optional bias) in place of the linear combination, no source C, and no
// workspace (graph-capturable).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
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

// E1: D = bf16((acc * x_scale[m]) * w_scale2 + bias[n]). The fp32 rounding of each step is not pinned yet (the
// compiler may contract the multiply and the add); N4 fixes it with the numerics version.
template <class Tile>
struct PlainEpilogue {
  using XScale = fusion::Sm90ColBroadcast<0, Tile, float, float, cute::Stride<cute::_1, cute::_0, std::int64_t>>;
  using WScale = fusion::Sm90ScalarBroadcast<float>;
  using Bias = fusion::Sm90RowBroadcast<0, Tile, bf16, float, cute::Stride<cute::_0, cute::_1, std::int64_t>>;
  using Mul = fusion::Sm90Compute<cutlass::multiplies, float, float, kRn>;
  using AccX = fusion::Sm90EVT<Mul, fusion::Sm90AccFetch, XScale>;
  using AccXW = fusion::Sm90EVT<Mul, AccX, WScale>;
  using Tree = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::plus, bf16, float, kRn>, AccXW, Bias>;
};

template <class Tile>
struct GemmType {
  using Arch = cutlass::arch::Sm120;
  using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;  // GeForce Blackwell: no multicast, cluster 1x1x1
  using RowMajor = cutlass::layout::RowMajor;
  using ColumnMajor = cutlass::layout::ColumnMajor;  // w [N, K] row-major is B [K, N] column-major (TN only)
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, RowMajor,
      kAlignD, bf16, RowMajor, kAlignD, cutlass::epilogue::collective::EpilogueScheduleAuto,
      typename PlainEpilogue<Tile>::Tree>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass, ElementAB, RowMajor, kAlignAB, ElementAB, ColumnMajor, kAlignAB, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

inline void check_status(cutlass::Status status, const char* op, const char* stage) {
  if (status != cutlass::Status::kSuccess) {
    kernels::detail::fail(op, std::string("CUTLASS ") + stage + " failed: " + cutlassGetStatusString(status));
  }
}

template <class Tile>
void launch(const char* op, const Nvfp4Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
            int sm_count, cudaStream_t stream) {
  using T = GemmType<Tile>;
  using Gemm = typename T::Gemm;
  using Mainloop = typename T::Mainloop;
  using Sf = typename Mainloop::Sm1xxBlkScaledConfig;
  using Data = typename ElementAB::DataType;
  using Scale = typename ElementAB::ScaleFactorType;
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
  // EVT arguments nest as {children..., node}: ((acc * x_scale) * w_scale2) + bias.
  auto& t = args.epilogue.thread;
  t = {{{{}, {a.x_scale, 0.0f, {}}, {}}, {{a.w_scale2}}, {}},
       {reinterpret_cast<const bf16*>(bias), bf16(0.0f), {}},
       {}};
  args.epilogue.ptr_C = nullptr;
  args.epilogue.ptr_D = reinterpret_cast<bf16*>(out);
  args.epilogue.dD = cute::make_stride(ldo, cute::Int<1>{}, std::int64_t{0});
  MI_CUDA_CHECK(cudaGetDevice(&args.hw_info.device_id));
  args.hw_info.sm_count = sm_count;

  Gemm gemm;
  check_status(Gemm::can_implement(args), op, "can_implement");
  if (Gemm::get_workspace_size(args) != 0) check_status(cutlass::Status::kErrorWorkspaceNull, op, "workspace");
  check_status(gemm.initialize(args, nullptr, stream), op, "initialize");
  check_status(gemm.run(stream), op, "run");
  MI_CUDA_CHECK_LAUNCH();
}

template <class Tile>
cudaError_t probe() {
  cudaFuncAttributes attr{};
  const cudaError_t err = cudaFuncGetAttributes(&attr, cutlass::device_kernel<typename GemmType<Tile>::Kernel>);
  if (err != cudaSuccess) (void)cudaGetLastError();  // not sticky; leave no error behind for the next check
  return err;
}

}  // namespace mi::kernels::nvfp4::detail
