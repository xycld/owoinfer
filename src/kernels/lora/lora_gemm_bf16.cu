// gemm_lora (see lora_gemm.h).
#include "kernels/lora/lora_gemm_kernel.cuh"

namespace mi::kernels::lora {
namespace detail {
namespace {

template <class Tile>
void plain(const fused::detail::Problem& p, const Tensor& out, const Tensor& x, const Tensor& w, const LoraSlices& l,
           cudaStream_t stream) {
  using K = typename Bf16LoraKernel<Tile, PlainTree>::Kernel;
  using namespace cute;
  typename K::Params prm;
  prm.problem = {p.m, p.n, p.k};
  prm.pa = decltype(prm.pa)(cutlass::layout::RowMajor(static_cast<int>(p.ldx)));
  prm.pb = decltype(prm.pb)(cutlass::layout::ColumnMajor(static_cast<int>(p.ldw)));
  prm.a = x.data;
  prm.b = w.data;
  set_slices<K>("gemm_lora", l, p.m, p.n, prm);
  const typename K::FusionCallbacks::Arguments cb{{{}, {}}, {out.ptr<bf16>(), {p.ldo, _1{}, p.ldo * p.m}}};
  prm.output_op = K::FusionCallbacks::to_underlying_arguments(prm.problem, cb, nullptr);
  launch<K>("gemm_lora", prm, stream);
}

}  // namespace
}  // namespace detail

void gemm_lora(const Tensor& out, const Tensor& x, const Tensor& w, const LoraSlices& lora, cudaStream_t stream,
               fused::GemmTile tile) {
  const fused::detail::Problem p = fused::detail::check_problem("lora::gemm_lora", out, x, w, w.dim(0));
  if (p.m == 0) return;
  fused::detail::with_tile(tile == fused::GemmTile::kAuto ? fused::detail::tile_for(p) : tile,
                           [&](auto cfg) { detail::plain<decltype(cfg)>(p, out, x, w, lora, stream); });
}

}  // namespace mi::kernels::lora
