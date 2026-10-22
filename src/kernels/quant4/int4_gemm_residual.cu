// int4_gemm_gated_residual (see quant4.h).
#include "kernels/quant4/int4_gemm_impl.cuh"

namespace mi::kernels::quant4 {
namespace detail {
namespace {

template <class Tile>
void gated_residual(const Problem& p, const Int4Operands& a, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                    __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  using N = Nodes<Tile>;
  using Gemm = typename Int4Kernel<Tile, typename N::GatedResidual>::Gemm;
  using namespace cute;
  auto* r = reinterpret_cast<bf16*>(residual);
  // The residual is loaded and stored by the same thread in the same epilogue step: in place is safe.
  const typename N::GatedResidual::Arguments cb{
      {{},
       {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}},
       {a.w_scale, 0.f, {_0{}, _1{}, p.n}},
       {reinterpret_cast<const bf16*>(bias), bf16(0), {_0{}, _1{}, p.n}},
       {reinterpret_cast<const bf16*>(gate), bf16(0), {_0{}, _1{}, p.n}},
       {r, bf16(0), {ldr, _1{}, ldr * p.m}},
       {}},
      {r, {ldr, _1{}, ldr * p.m}}};
  launch<Gemm>("quant4::int4_gemm_gated_residual", p, a, cb, stream);
}

}  // namespace
}  // namespace detail

void int4_gemm_gated_residual(const Int4Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream, Int4Tile tile) {
  constexpr const char* name = "quant4::int4_gemm_gated_residual";
  const detail::Problem p = detail::check_operands(name, op);
  detail::check_output(name, "residual", residual, ldr, p.n);
  detail::check_vector(name, "bias", bias, /*optional=*/true);
  detail::check_vector(name, "gate", gate, /*optional=*/false);
  if (p.m == 0) return;
  detail::with_tile(detail::resolve_tile(tile, p), [&](auto cfg) {
    detail::gated_residual<decltype(cfg)>(p, op, bias, gate, residual, ldr, stream);
  });
}

}  // namespace mi::kernels::quant4
