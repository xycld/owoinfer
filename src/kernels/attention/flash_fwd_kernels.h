// Entry points of the FlashAttention-2 forward instantiations, one per .cu file so they compile in
// parallel. Declared apart from flash_fwd_launch.cuh so the dispatcher does not parse the kernel headers.
#pragma once

#include <cuda_runtime.h>

#include "flash.h"

namespace mi::attention::detail {

// head_dim 128, non-causal; tile kBlockM x kBlockN, 4 warps.
void run_fwd_hdim128_128x64(const flash::Flash_fwd_params& params, cudaStream_t stream);
void run_fwd_hdim128_128x32(const flash::Flash_fwd_params& params, cudaStream_t stream);
// head_dim 256, causal; 128x64 with 8 warps, 64x64 with 4 warps.
void run_fwd_hdim256_causal_128x64(const flash::Flash_fwd_params& params, cudaStream_t stream);
void run_fwd_hdim256_causal_64x64(const flash::Flash_fwd_params& params, cudaStream_t stream);

}  // namespace mi::attention::detail
