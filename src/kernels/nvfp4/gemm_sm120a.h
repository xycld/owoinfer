// Internal to src/kernels/nvfp4: the entry points of the sm_120a-only translation units (gemm_sm120a_*.cu), which
// cmake/modules/nvfp4.cmake builds for 120a-real alone and only when NVFP4 is enabled. nvfp4.cpp calls them only
// when MOSSINFER_HAVE_NVFP4_SM120A is defined and available() holds. Arguments are already checked.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/nvfp4/nvfp4.h"

namespace mi::kernels::nvfp4::detail {

void gemm_128x128(const Nvfp4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                  int sm_count, cudaStream_t stream);
void gemm_256x128(const Nvfp4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                  int sm_count, cudaStream_t stream);

// cudaFuncGetAttributes on the 128x128 kernel: cudaSuccess when the current device can run the sm_120a image.
cudaError_t probe_image();

}  // namespace mi::kernels::nvfp4::detail
