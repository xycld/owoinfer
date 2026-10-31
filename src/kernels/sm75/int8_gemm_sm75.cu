// sm_75 INT8 ConvRot GEMMs (see kernels/sm75/int8_sm75.h): the sm_80 CUTLASS EVT kernel with Turing's m8n8k16 INT8
// MMA and tiles that fit 64 KiB of shared memory.
#include "kernels/sm75/int8_sm75.h"

#include <string>

#include "kernels/quant/int8_gemm_impl.cuh"

namespace mi::kernels::sm75 {
namespace qd = quant::detail;

namespace {

using Sm75Tile = qd::TileCfg<64, 128, 64, 32, 64, 2>;  // threadblock 64x128, warp 32x64, K 64, 2 stages: 24 KiB
using Sm75Instr = cutlass::gemm::GemmShape<8, 8, 16>;

template <class Callbacks, class PermuteB = cutlass::layout::NoPermute>
using Sm75Kernel = typename qd::Int8Kernel<Sm75Tile, Callbacks, PermuteB, cutlass::arch::Sm75, Sm75Instr>::Gemm;

void plain(const qd::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out,
           std::int64_t ldo, cudaStream_t stream) {
  using N = qd::Nodes<Sm75Tile>;
  using Gemm = Sm75Kernel<typename N::Plain>;
  using namespace cute;
  const typename N::Plain::Arguments cb{
      {{},
       {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}},
       {a.w_scale, 0.f, {_0{}, _1{}, p.n}},
       {reinterpret_cast<const qd::bf16*>(bias), qd::bf16(0), {_0{}, _1{}, p.n}},
       {}},
      {reinterpret_cast<qd::bf16*>(out), {ldo, _1{}, ldo * p.m}}};
  qd::launch<Gemm>("sm75::int8_gemm", p, a, cb, stream);
}

void gated_residual(const qd::Problem& p, const quant::Int8Operands& a, const __nv_bfloat16* bias,
                    const __nv_bfloat16* gate, __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  using N = qd::Nodes<Sm75Tile>;
  using Gemm = Sm75Kernel<typename N::GatedResidual>;
  using namespace cute;
  auto* r = reinterpret_cast<qd::bf16*>(residual);
  const typename N::GatedResidual::Arguments cb{
      {{},
       {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}},
       {a.w_scale, 0.f, {_0{}, _1{}, p.n}},
       {reinterpret_cast<const qd::bf16*>(bias), qd::bf16(0), {_0{}, _1{}, p.n}},
       {reinterpret_cast<const qd::bf16*>(gate), qd::bf16(0), {_0{}, _1{}, p.n}},
       {r, qd::bf16(0), {ldr, _1{}, ldr * p.m}},
       {}},
      {r, {ldr, _1{}, ldr * p.m}}};
  qd::launch<Gemm>("sm75::int8_gemm_gated_residual", p, a, cb, stream);
}

void situ_glu(const qd::Problem& p, const quant::Int8Operands& a, __nv_bfloat16* out, std::int64_t ldo,
              cudaStream_t stream) {
  using N = qd::Nodes<Sm75Tile>;
  using Gemm = Sm75Kernel<typename N::SituGlu, qd::UpGatePermuteB>;
  using namespace cute;
  auto* d = reinterpret_cast<std::uint8_t*>(out);
  const std::int64_t ldd = 2 * ldo;
  const typename N::SituGlu::Arguments cb{
      {{}, {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}}, {a.w_scale}, {}},
      {d, {ldd, _1{}, ldd * p.m}}};
  qd::launch<Gemm>("sm75::int8_gemm_situ_glu", p, a, cb, stream);
}

}  // namespace

void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "out", out, ldo, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  plain(p, op, bias, out, ldo, stream);
}

void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm_gated_residual";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "residual", residual, ldr, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  qd::check_vector(name, "gate", gate, /*optional=*/false);
  if (p.m == 0) return;
  gated_residual(p, op, bias, gate, residual, ldr, stream);
}

void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr const char* name = "sm75::int8_gemm_situ_glu";
  if (op.n % (2 * qd::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const qd::Problem p = qd::check_operands(name, op);
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " +
                                    std::to_string(ldo));
  }
  if (p.m == 0) return;
  situ_glu(p, op, out, ldo, stream);
}

}  // namespace mi::kernels::sm75
