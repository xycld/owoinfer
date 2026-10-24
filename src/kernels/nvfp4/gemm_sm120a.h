// Internal to src/kernels/nvfp4: the entry points of the sm_120a-only translation units (gemm_sm120a_*.cu), which
// cmake/modules/nvfp4.cmake builds for 120a-real alone and only when NVFP4 is enabled. nvfp4.cpp calls them only
// when MOSSINFER_HAVE_NVFP4_SM120A is defined and available() holds. Arguments are already checked.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/nvfp4/nvfp4.h"

namespace mi::kernels::nvfp4::detail {

// The GEMM epilogues of nvfp4.h: E1, E1L (with the LoRA delta), E2, E2L.
enum class Epi { kPlain, kPlainDelta, kGatedResidual, kGatedResidualDelta };

struct GemmArgs {
  Nvfp4Operands op;
  const __nv_bfloat16* bias = nullptr;  // or nullptr
  const __nv_bfloat16* gate = nullptr;  // E2 / E2L
  Nvfp4Delta delta;                     // E1L / E2L
  __nv_bfloat16* out = nullptr;         // E2 / E2L: the residual, read and written in place
  std::int64_t ldo = 0;
};

// One instantiation per translation unit (gemm_sm120a_<epilogue>_<tile>.cu); tile is never kAuto.
template <Nvfp4Tile kTile, Epi kEpi>
void gemm(const GemmArgs& args, int sm_count, cudaStream_t stream);

// cudaFuncGetAttributes on the 128x128 E1 kernel: cudaSuccess when the current device can run the sm_120a image.
cudaError_t probe_image();

// nvfp4.h e2m1_cvt_codes (test G1e).
void e2m1_cvt(const float* x, std::uint8_t* codes, std::int64_t n, cudaStream_t stream);

}  // namespace mi::kernels::nvfp4::detail
