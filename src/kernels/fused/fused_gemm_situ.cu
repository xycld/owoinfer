// gemm_situ_glu (see fused_gemm.h).
#include "kernels/fused/fused_gemm_impl.cuh"

namespace mi::kernels::fused {
namespace detail {
namespace {

template <class Tile>
void situ_glu(const Problem& p, const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream) {
  using Gemm = DeviceGemmSituGlu<Tile>;
  // D as bytes: [M, 2F] bytes with a row stride of 2 * ldo bytes is exactly the bf16 [M, F] output.
  auto* d = static_cast<std::uint8_t*>(out.data);
  const typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {p.m, p.n, p.k}, /*batch_count=*/1,
                                      typename SituGluOp::Params(), x.data, w.data, d, d, 0, 0, 0, 0, p.ldx, p.ldw,
                                      2 * p.ldo, 2 * p.ldo);
  launch<Gemm>("gemm_situ_glu", args, stream);
}

}  // namespace
}  // namespace detail

void gemm_situ_glu(const Tensor& out, const Tensor& x, const Tensor& w_up_gate, cudaStream_t stream) {
  constexpr const char* op = "gemm_situ_glu";
  if (w_up_gate.rank != 2 || w_up_gate.dim(0) % (2 * detail::kUpGateGroup) != 0) {
    throw std::invalid_argument(std::string("mi::kernels::fused::") + op + ": w must be [2F, K] with F % 4 == 0, got " +
                                w_up_gate.describe());
  }
  const detail::Problem p = detail::check_problem(op, out, x, w_up_gate, w_up_gate.dim(0) / 2);
  if (p.m == 0) return;
  detail::with_tile(detail::tile_for(p),
                    [&](auto tile) { detail::situ_glu<decltype(tile)>(p, out, x, w_up_gate, stream); });
}

}  // namespace mi::kernels::fused
