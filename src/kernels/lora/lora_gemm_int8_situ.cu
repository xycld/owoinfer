// int8_gemm_situ_glu_lora (see lora_gemm.h).
#include <string>

#include "kernels/lora/lora_gemm_kernel.cuh"

namespace mi::kernels::lora {
namespace detail {
namespace {

template <class Tile>
void situ_glu(const quant::detail::Problem& p, const quant::Int8Operands& a, const LoraSlices& l, __nv_bfloat16* out,
              std::int64_t ldo, cudaStream_t stream) {
  using K = typename Int8LoraKernel<Tile, SituGluTree, quant::detail::UpGatePermuteB>::Kernel;
  using namespace cute;
  typename K::Params prm;
  prm.problem = {p.m, p.n, p.k};
  prm.pa = decltype(prm.pa)(cutlass::layout::RowMajor(static_cast<int>(p.ldx)));
  prm.pb = decltype(prm.pb)(cutlass::layout::ColumnMajor(p.k));
  prm.a = a.x_q;
  prm.b = a.w;
  prm.convert = {a.x_scale, a.w_scale, nullptr, p.n / 2};
  set_slices<K>("int8_gemm_situ_glu_lora", l, p.m, p.n, prm);
  // D as bytes: [M, 2F] bytes with a row stride of 2 * ldo bytes is exactly the bf16 [M, F] output.
  auto* d = reinterpret_cast<std::uint8_t*>(out);
  const std::int64_t ldd = 2 * ldo;
  const typename K::FusionCallbacks::Arguments cb{{{}, {}}, {d, {ldd, _1{}, ldd * p.m}}};
  prm.output_op = K::FusionCallbacks::to_underlying_arguments(prm.problem, cb, nullptr);
  launch<K>("int8_gemm_situ_glu_lora", prm, stream);
}

}  // namespace
}  // namespace detail

void int8_gemm_situ_glu_lora(const quant::Int8Operands& op, const LoraSlices& lora, __nv_bfloat16* out,
                             std::int64_t ldo, cudaStream_t stream, quant::Int8Tile tile) {
  constexpr const char* name = "lora::int8_gemm_situ_glu_lora";
  if (op.n % (2 * quant::detail::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const quant::detail::Problem p = quant::detail::check_operands(name, op);
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " + std::to_string(ldo));
  }
  if (p.m == 0) return;
  quant::detail::with_tile(quant::detail::resolve_tile(tile, p),
                           [&](auto cfg) { detail::situ_glu<decltype(cfg)>(p, op, lora, out, ldo, stream); });
}

}  // namespace mi::kernels::lora
