// NVFP4 GEMM, epilogue E2 (gated residual), tile 128x128x128 Pingpong (sm_120a only, see gemm_sm120a_impl.cuh).
#include "kernels/nvfp4/gemm_sm120a_impl.cuh"

namespace mi::kernels::nvfp4::detail {

template void gemm<Nvfp4Tile::k128x128x128Pingpong, Epi::kGatedResidual>(const GemmArgs&, int, cudaStream_t);

}  // namespace mi::kernels::nvfp4::detail
