# Vendored FlashAttention-2 (forward kernel headers only)

- Upstream: https://github.com/Dao-AILab/flash-attention, tag v2.8.3, commit 060c9188beec3a8b62b33a3bfa6d5d2d44975fab
- License: BSD-3-Clause (`LICENSE`, contributors in `AUTHORS`).
- CUTLASS: upstream pins its `csrc/cutlass` submodule at dc4817921edda44a549197ff3a9dcf5df0636e7b. That copy is **not**
  vendored: these headers build and pass the tests unchanged against the repo's CUTLASS 4.7.1 (`third_party/cutlass`)
  with CUDA 13.4 and GCC 16.

## Kept

From `csrc/flash_attn/src/`, the headers the forward kernel needs:
`flash.h`, `flash_fwd_kernel.h`, `kernel_traits.h`, `block_info.h`, `utils.h`, `softmax.h`, `mask.h`, `dropout.h`,
`rotary.h`, `alibi.h`, `philox.cuh`, `philox_unpack.cuh`, `namespace_config.h`.

Removed: everything else. That covers the backward kernels, the split-KV launch path, the per-head-dim `.cu`
instantiations, `flash_fwd_launch_template.h`, `flash_api.cpp` (the PyTorch extension), the Hopper (FA3) and ROCm
trees, and the Python package.

## Local changes

1. Added `csrc/flash_attn/src/mi_torch_shim.h`. It defines an inert `at::PhiloxCudaState` and
   `at::cuda::philox::unpack`, the only PyTorch symbols the forward kernel touches. They are used only for dropout,
   which MossInfer never enables.
2. `flash.h`: `#include <ATen/cuda/CUDAGeneratorImpl.h>` became `#include "mi_torch_shim.h"`.
3. `philox_unpack.cuh`: `#include <ATen/cuda/detail/UnpackRaw.cuh>` became `#include "mi_torch_shim.h"`.

Nothing else changed: no kernel code was touched.

## Where the replacements live

The launcher and instantiations that replace `flash_api.cpp` and `flash_fwd_launch_template.h` live in
`src/kernels/attention/`:

- `flash_attention.cu` builds `Flash_fwd_params` from `mi::Tensor` views, mirroring `mha_varlen_fwd`.
- `flash_fwd_launch.cuh` holds one kernel per tile, with dropout, local, ALiBi, softcap and return-softmax compiled out.
- `flash_fwd_*.cu` hold the four instantiations:
  - head dim 128, non-causal: tiles 128x64 and 128x32
  - head dim 256, causal: tiles 128x64 (8 warps) and 64x64

Their include path is set per source in `cmake/modules/attention.cmake`.
