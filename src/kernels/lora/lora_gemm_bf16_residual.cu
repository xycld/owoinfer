// gemm_gated_residual_lora (see lora_gemm.h).
#include <stdexcept>

#include "kernels/lora/lora_gemm_kernel.cuh"

namespace mi::kernels::lora {
namespace detail {
namespace {

template <class Tile>
void gated_residual(const fused::detail::Problem& p, const Tensor& residual, const Tensor& x, const Tensor& w,
                    const Tensor& gate, const LoraSlices& l, cudaStream_t stream) {
  using K = typename Bf16LoraKernel<Tile, GatedResidualTree>::Kernel;
  using namespace cute;
  typename K::Params prm;
  prm.problem = {p.m, p.n, p.k};
  prm.pa = decltype(prm.pa)(cutlass::layout::RowMajor(static_cast<int>(p.ldx)));
  prm.pb = decltype(prm.pb)(cutlass::layout::ColumnMajor(static_cast<int>(p.ldw)));
  prm.a = x.data;
  prm.b = w.data;
  set_slices<K>("gemm_gated_residual_lora", l, p.m, p.n, prm);
  auto* r = residual.ptr<bf16>();
  // The residual is loaded and stored by the same thread in the same epilogue step: in place is safe.
  const typename K::FusionCallbacks::Arguments cb{
      {{}, {gate.ptr<const bf16>(), bf16(0), {_0{}, _1{}, p.n}}, {r, bf16(0), {p.ldo, _1{}, p.ldo * p.m}}, {}},
      {r, {p.ldo, _1{}, p.ldo * p.m}}};
  prm.output_op = K::FusionCallbacks::to_underlying_arguments(prm.problem, cb, nullptr);
  launch<K>("gemm_gated_residual_lora", prm, stream);
}

}  // namespace
}  // namespace detail

void gemm_gated_residual_lora(const Tensor& residual, const Tensor& x, const Tensor& w, const Tensor& gate,
                              const LoraSlices& lora, cudaStream_t stream, fused::GemmTile tile) {
  constexpr const char* op = "lora::gemm_gated_residual_lora";
  const fused::detail::Problem p = fused::detail::check_problem(op, residual, x, w, w.dim(0));
  kernels::detail::check_vector(op, "gate", gate, p.n);
  if (reinterpret_cast<std::uintptr_t>(gate.data) % 16 != 0) {
    throw std::invalid_argument(std::string("mi::kernels::") + op + ": gate must be 16-byte aligned");
  }
  if (p.m == 0) return;
  fused::detail::with_tile(tile == fused::GemmTile::kAuto ? fused::detail::tile_for(p) : tile, [&](auto cfg) {
    detail::gated_residual<decltype(cfg)>(p, residual, x, w, gate, lora, stream);
  });
}

}  // namespace mi::kernels::lora
