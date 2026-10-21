// Torch-free replacement for FlashAttention-2's flash_fwd_launch_template.h: one kernel per tile
// configuration with every optional feature compiled out (no dropout, local window, ALiBi, softcap or
// returned softmax). Varlen always (Is_even_MN = false) and head_dim equal to the tile's (Is_even_K = true).
// Internal to src/kernels/attention: include only from its .cu files.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>

#include "core/cuda_check.h"
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "kernels/attention/flash_fwd_kernels.h"
#include "kernel_traits.h"

namespace mi::attention::detail {

// As upstream DEFINE_FLASH_FORWARD_KERNEL: __grid_constant__ params, no __launch_bounds__ (keeps the
// register allocation of the upstream/PyTorch builds of the same kernel).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define MI_FLASH_PARAM_MODIFIER __grid_constant__
#else
#define MI_FLASH_PARAM_MODIFIER
#endif

template <typename Traits, bool kCausal>
__global__ void flash_fwd_kernel(MI_FLASH_PARAM_MODIFIER const flash::Flash_fwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  flash::compute_attn<Traits, /*Is_dropout=*/false, kCausal, /*Is_local=*/false, /*Has_alibi=*/false,
                      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false, /*Return_softmax=*/false>(params);
#endif
}

template <typename Traits, bool kCausal>
void run_flash_fwd(const flash::Flash_fwd_params& params, cudaStream_t stream) {
  constexpr int kSmem = Traits::kSmemSize;
  auto kernel = &flash_fwd_kernel<Traits, kCausal>;
  if constexpr (kSmem >= 48 * 1024) {
    // Once per device and instantiation (a driver call with a lock, not for every launch).
    static std::array<std::atomic<bool>, 64> configured{};
    int device = 0;
    MI_CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= static_cast<int>(configured.size()) || !configured[device].load()) {
      MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
      if (device >= 0 && device < static_cast<int>(configured.size())) configured[device].store(true);
    }
  }
  const dim3 grid((params.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM, params.b, params.h);
  kernel<<<grid, Traits::kNThreads, kSmem, stream>>>(params);
  MI_CUDA_CHECK_LAUNCH();
}

// Tile configurations, chosen as upstream run_mha_fwd_hdim128/256 do (see flash_attention.cu).
using Hdim128Tile128x64 = ::Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using Hdim128Tile128x32 = ::Flash_fwd_kernel_traits<128, 128, 32, 4, false, false, cutlass::bfloat16_t>;
using Hdim256Tile128x64 = ::Flash_fwd_kernel_traits<256, 128, 64, 8, false, false, cutlass::bfloat16_t>;
using Hdim256Tile64x64 = ::Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t>;

}  // namespace mi::attention::detail
