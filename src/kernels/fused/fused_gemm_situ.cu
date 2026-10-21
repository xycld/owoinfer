// gemm_situ_glu (see fused_gemm.h).
#include "kernels/fused/fused_gemm_impl.cuh"

namespace mi::kernels::fused {
namespace detail {
namespace {

template <class Tile>
void situ_glu(const Problem& p, const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream) {
  using Gemm = DeviceGemm<Tile, SituGluOp, std::uint8_t>;
  // D as bytes: [M, 2F] bytes with a row stride of 2 * ldo bytes is exactly the bf16 [M, F] output.
  auto* d = static_cast<std::uint8_t*>(out.data);
  const typename Gemm::Arguments args({p.m, p.n, p.k}, {x.ptr<const bf16>(), p.ldx}, {w.ptr<const bf16>(), p.ldw},
                                      {d, 2 * p.ldo}, {d, 2 * p.ldo}, typename SituGluOp::Params());
  launch<Gemm>("gemm_situ_glu", args, stream);
}

}  // namespace
}  // namespace detail

void gemm_situ_glu(const Tensor& out, const Tensor& x, const Tensor& w_interleaved, cudaStream_t stream) {
  constexpr const char* op = "gemm_situ_glu";
  if (w_interleaved.rank != 2 || w_interleaved.dim(0) % (2 * kUpGateGroup) != 0) {
    throw std::invalid_argument(std::string("mi::kernels::fused::") + op + ": w must be [2F, K] with F % 4 == 0, got " +
                                w_interleaved.describe());
  }
  const detail::Problem p = detail::check_problem(op, out, x, w_interleaved, w_interleaved.dim(0) / 2);
  if (p.m == 0) return;
  switch (detail::tile_for(p)) {
    case GemmTile::k128x256: detail::situ_glu<detail::Tile128x256>(p, out, x, w_interleaved, stream); break;
    case GemmTile::k128x128: detail::situ_glu<detail::Tile128x128>(p, out, x, w_interleaved, stream); break;
    default: detail::situ_glu<detail::Tile256x128>(p, out, x, w_interleaved, stream); break;
  }
}

}  // namespace mi::kernels::fused
