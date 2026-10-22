// gemm_gated_residual (see fused_gemm.h).
#include "kernels/fused/fused_gemm_impl.cuh"

namespace mi::kernels::fused {
namespace detail {
namespace {

template <class Tile>
void gated_residual(const Problem& p, const Tensor& residual, const Tensor& x, const Tensor& w, const Tensor& gate,
                    cudaStream_t stream) {
  using Gemm = DeviceGemmGatedResidual<Tile>;
  const typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {p.m, p.n, p.k}, /*batch_count=*/1,
                                      typename GatedResidualOp::Params(), x.data, w.data, /*ptr_C=*/residual.data,
                                      /*ptr_D=*/residual.data, /*ptr_Vector=*/gate.data, /*ptr_Tensor=*/nullptr,
                                      0, 0, 0, 0, 0, 0, p.ldx, p.ldw, p.ldo, p.ldo, /*ldr=*/0, /*ldt=*/0);
  launch<Gemm>("gemm_gated_residual", args, stream);
}

}  // namespace
}  // namespace detail

void gemm_gated_residual(const Tensor& residual, const Tensor& x, const Tensor& w, const Tensor& gate,
                         cudaStream_t stream, GemmTile tile) {
  constexpr const char* op = "gemm_gated_residual";
  const detail::Problem p = detail::check_problem(op, residual, x, w, w.dim(0));
  kernels::detail::check_vector(op, "gate", gate, p.n);
  if (reinterpret_cast<std::uintptr_t>(gate.data) % 16 != 0) {
    throw std::invalid_argument(std::string("mi::kernels::fused::") + op + ": gate must be 16-byte aligned");
  }
  if (p.m == 0) return;
  detail::with_tile(tile == GemmTile::kAuto ? detail::tile_for(p) : tile, [&](auto cfg) {
    detail::gated_residual<decltype(cfg)>(p, residual, x, w, gate, stream);
  });
}

}  // namespace mi::kernels::fused
