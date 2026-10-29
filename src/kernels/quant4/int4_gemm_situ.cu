// int4_gemm_situ_glu (see quant4.h).
#include <string>

#include "kernels/quant4/int4_gemm_impl.cuh"

namespace mi::kernels::quant4 {
namespace detail {
namespace {

template <class Tile>
void situ_glu(const Problem& p, const Int4Operands& a, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  using N = Nodes<Tile>;
  using Gemm = typename Int4Kernel<Tile, typename N::SituGlu, UpGatePermuteB>::Gemm;
  using namespace cute;
  // D as bytes: [M, 2F] bytes with a row stride of 2 * ldo bytes is exactly the bf16 [M, F] output.
  auto* d = reinterpret_cast<std::uint8_t*>(out);
  const std::int64_t ldd = 2 * ldo;
  const typename N::SituGlu::Arguments cb{
      {{}, {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}}, {a.w_scale}, quant::detail::situ_factor_tables(stream)},
      {d, {ldd, _1{}, ldd * p.m}}};
  launch<Gemm>("quant4::int4_gemm_situ_glu", p, a, cb, stream);
}

}  // namespace
}  // namespace detail

void int4_gemm_situ_glu(const Int4Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream,
                        Int4Tile tile) {
  constexpr const char* name = "quant4::int4_gemm_situ_glu";
  if (op.n % (2 * detail::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const detail::Problem p = detail::check_operands(name, op);
  // Each epilogue access stores 4 bf16 (8 bytes) at column 4j.
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " +
                                    std::to_string(ldo));
  }
  if (p.m == 0) return;
  detail::with_tile(detail::resolve_tile(tile, p),
                    [&](auto cfg) { detail::situ_glu<decltype(cfg)>(p, op, out, ldo, stream); });
}

}  // namespace mi::kernels::quant4
