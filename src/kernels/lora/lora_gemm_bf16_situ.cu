// gemm_situ_glu_lora (see lora_gemm.h).
#include <stdexcept>

#include "kernels/lora/lora_gemm_kernel.cuh"

namespace mi::kernels::lora {
namespace detail {
namespace {

template <class Tile>
void situ_glu(const fused::detail::Problem& p, const Tensor& out, const Tensor& x, const Tensor& w,
              const LoraSlices& l, cudaStream_t stream) {
  using K = typename Bf16LoraKernel<Tile, SituGluTree, fused::detail::UpGatePermuteB>::Kernel;
  using namespace cute;
  typename K::Params prm;
  prm.problem = {p.m, p.n, p.k};
  prm.pa = decltype(prm.pa)(cutlass::layout::RowMajor(static_cast<int>(p.ldx)));
  prm.pb = decltype(prm.pb)(cutlass::layout::ColumnMajor(static_cast<int>(p.ldw)));
  prm.a = x.data;
  prm.b = w.data;
  set_slices<K>("gemm_situ_glu_lora", l, p.m, p.n, prm);
  // D as bytes: [M, 2F] bytes with a row stride of 2 * ldo bytes is exactly the bf16 [M, F] output.
  auto* d = static_cast<std::uint8_t*>(out.data);
  const std::int64_t ldd = 2 * p.ldo;
  const typename K::FusionCallbacks::Arguments cb{{{}, quant::detail::situ_factor_tables(stream)}, {d, {ldd, _1{}, ldd * p.m}}};
  prm.output_op = K::FusionCallbacks::to_underlying_arguments(prm.problem, cb, nullptr);
  launch<K>("gemm_situ_glu_lora", prm, stream);
}

}  // namespace
}  // namespace detail

void gemm_situ_glu_lora(const Tensor& out, const Tensor& x, const Tensor& w_up_gate, const LoraSlices& lora,
                        cudaStream_t stream, fused::GemmTile tile) {
  constexpr const char* op = "lora::gemm_situ_glu_lora";
  if (w_up_gate.rank != 2 || w_up_gate.dim(0) % (2 * fused::detail::kUpGateGroup) != 0) {
    throw std::invalid_argument(std::string("mi::kernels::") + op + ": w must be [2F, K] with F % 4 == 0, got " + w_up_gate.describe());
  }
  const fused::detail::Problem p = fused::detail::check_problem(op, out, x, w_up_gate, w_up_gate.dim(0) / 2);
  if (p.m == 0) return;
  fused::detail::with_tile(tile == fused::GemmTile::kAuto ? fused::detail::tile_for(p) : tile,
                           [&](auto cfg) { detail::situ_glu<decltype(cfg)>(p, out, x, w_up_gate, lora, stream); });
}

}  // namespace mi::kernels::lora
