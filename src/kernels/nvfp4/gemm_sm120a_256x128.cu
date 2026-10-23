// NVFP4 GEMM, 256x128x128 Cooperative tile (sm_120a only, see gemm_sm120a_impl.cuh).
#include "kernels/nvfp4/gemm_sm120a_impl.cuh"

namespace mi::kernels::nvfp4::detail {

void gemm_256x128(const Nvfp4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                  int sm_count, cudaStream_t stream) {
  launch<cute::Shape<cute::_256, cute::_128, cute::_128>>("nvfp4_gemm", op, bias, out, ldo, sm_count, stream);
}

}  // namespace mi::kernels::nvfp4::detail
