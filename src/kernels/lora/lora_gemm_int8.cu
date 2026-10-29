// int8_gemm_lora (see lora_gemm.h).
#include "kernels/lora/lora_gemm_kernel.cuh"

namespace mi::kernels::lora {
namespace detail {
namespace {

template <class Tile>
void plain(const quant::detail::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias, const LoraSlices& l,
           __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  using K = typename Int8LoraKernel<Tile, PlainTree>::Kernel;
  using namespace cute;
  typename K::Params prm;
  prm.problem = {p.m, p.n, p.k};
  prm.pa = decltype(prm.pa)(cutlass::layout::RowMajor(static_cast<int>(p.ldx)));
  prm.pb = decltype(prm.pb)(cutlass::layout::ColumnMajor(p.k));
  prm.a = a.x_q;
  prm.b = a.w;
  prm.convert = {a.x_scale, a.w_scale, reinterpret_cast<const bf16*>(bias), 0};
  set_slices<K>("int8_gemm_lora", l, p.m, p.n, prm);
  const typename K::FusionCallbacks::Arguments cb{{{}, {}}, {reinterpret_cast<bf16*>(out), {ldo, _1{}, ldo * p.m}}};
  prm.output_op = K::FusionCallbacks::to_underlying_arguments(prm.problem, cb, nullptr);
  launch<K>("int8_gemm_lora", prm, stream, quant::detail::int8_band(p, K::kTileN));
}

}  // namespace
}  // namespace detail

void int8_gemm_lora(const quant::Int8Operands& op, const __nv_bfloat16* bias, const LoraSlices& lora,
                    __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream, quant::Int8Tile tile) {
  constexpr const char* name = "lora::int8_gemm_lora";
  const quant::detail::Problem p = quant::detail::check_operands(name, op);
  quant::detail::check_output(name, "out", out, ldo, p.n);
  quant::detail::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  quant::detail::with_tile(quant::detail::resolve_tile(tile, p),
                           [&](auto cfg) { detail::plain<decltype(cfg)>(p, op, bias, lora, out, ldo, stream); });
}

}  // namespace mi::kernels::lora
