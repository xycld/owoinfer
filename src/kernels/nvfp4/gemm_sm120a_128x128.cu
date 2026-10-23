// NVFP4 GEMM, 128x128x128 Cooperative tile (sm_120a only, see gemm_sm120a_impl.cuh), plus the image probe.
#include "kernels/nvfp4/gemm_sm120a_impl.cuh"

namespace mi::kernels::nvfp4::detail {

using Tile128x128 = cute::Shape<cute::_128, cute::_128, cute::_128>;

void gemm_128x128(const Nvfp4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                  int sm_count, cudaStream_t stream) {
  launch<Tile128x128>("nvfp4_gemm", op, bias, out, ldo, sm_count, stream);
}

cudaError_t probe_image() { return probe<Tile128x128>(); }

}  // namespace mi::kernels::nvfp4::detail
