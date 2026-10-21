// FlashAttention-2 forward instantiation (see flash_fwd_kernels.h).
#include "kernels/attention/flash_fwd_launch.cuh"

namespace mi::attention::detail {

void run_fwd_hdim128_128x32(const flash::Flash_fwd_params& params, cudaStream_t stream) {
  run_flash_fwd<Hdim128Tile128x32, false>(params, stream);
}

}  // namespace mi::attention::detail
