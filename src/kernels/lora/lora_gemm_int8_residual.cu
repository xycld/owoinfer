// int8_gemm_gated_residual_lora (see lora_gemm.h).
#include "kernels/lora/lora_gemm_kernel.cuh"

namespace mi::kernels::lora {
namespace detail {
namespace {

template <class Tile>
void gated_residual(const quant::detail::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias,
                    const __nv_bfloat16* gate, const LoraSlices& l, __nv_bfloat16* residual, std::int64_t ldr,
                    cudaStream_t stream) {
  using K = typename Int8LoraKernel<Tile, GatedResidualTree>::Kernel;
  using namespace cute;
  typename K::Params prm;
  prm.problem = {p.m, p.n, p.k};
  prm.pa = decltype(prm.pa)(cutlass::layout::RowMajor(static_cast<int>(p.ldx)));
  prm.pb = decltype(prm.pb)(cutlass::layout::ColumnMajor(p.k));
  prm.a = a.x_q;
  prm.b = a.w;
  prm.convert = {a.x_scale, a.w_scale, reinterpret_cast<const bf16*>(bias), 0};
  set_slices<K>("int8_gemm_gated_residual_lora", l, p.m, p.n, prm);
  auto* r = reinterpret_cast<bf16*>(residual);
  // The residual is loaded and stored by the same thread in the same epilogue step: in place is safe.
  const typename K::FusionCallbacks::Arguments cb{
      {{}, {reinterpret_cast<const bf16*>(gate), bf16(0), {_0{}, _1{}, p.n}}, {r, bf16(0), {ldr, _1{}, ldr * p.m}}, {}},
      {r, {ldr, _1{}, ldr * p.m}}};
  prm.output_op = K::FusionCallbacks::to_underlying_arguments(prm.problem, cb, nullptr);
  launch<K>("int8_gemm_gated_residual_lora", prm, stream);
}

}  // namespace
}  // namespace detail

void int8_gemm_gated_residual_lora(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                                   const LoraSlices& lora, __nv_bfloat16* residual, std::int64_t ldr,
                                   cudaStream_t stream, quant::Int8Tile tile) {
  constexpr const char* name = "lora::int8_gemm_gated_residual_lora";
  const quant::detail::Problem p = quant::detail::check_operands(name, op);
  quant::detail::check_output(name, "residual", residual, ldr, p.n);
  quant::detail::check_vector(name, "bias", bias, /*optional=*/true);
  quant::detail::check_vector(name, "gate", gate, /*optional=*/false);
  if (p.m == 0) return;
  quant::detail::with_tile(quant::detail::resolve_tile(tile, p), [&](auto cfg) {
    detail::gated_residual<decltype(cfg)>(p, op, bias, gate, lora, residual, ldr, stream);
  });
}

}  // namespace mi::kernels::lora
