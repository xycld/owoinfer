# HIP backend

MOSS-Image's native kernels for AMD GPUs through ROCm, and the same sources through HIP's NVIDIA headers, which run them on a CUDA device with a scalar leaf for checking. It is its own CMake project; the engine's build does not include this directory and nothing here needs CUDA on AMD.

| Target | Compiler | Leaf (`mma_leaf.h`) |
|---|---|---|
| `gfx1100` (RDNA3) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `gfx1200`, `gfx1201` (RDNA4) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `nvidia` | nvcc with HIP's NVIDIA headers | scalar: the exact bf16*bf16 product in fp32, then add, K ascending |

An AMD build holds a code object for each architecture in `HIP_TARGET` (a CMake list) and runs on those only; the release package's `moss-infer-hip` holds all three. It needs ROCm 7 at run time (`libamdhip64.so.7`). Every kernel launches workgroups of one wave (32 threads) and depends on that (`mma_leaf.h`).

## Files

| File | |
|---|---|
| `hip_block.h`, `block_kernels.hip` | One DiT block's kernels and their launches: RMSNorm with modulation, the GEMMs (plain, gated residual, SiTU-GLU), QK norm with RoPE, varlen attention, operand packing. Checked against the engine by `tools/hip_block` (its README has the rounding points and the results on an RX 9060 XT). |
| `gemm_tile.h` | The per-wave GEMM walk the kernel sources share: fragment loads with zero-padded tails, the K-ascending walk over row-major or packed operands, the 2D block order of the tiles, the GEMM kernel and its epilogues, launch helpers. |
| `mma_leaf.h`, `device_math.h` | The 16x16x16 leaf, and the bf16 conversions and rounded fp32 operations every kernel uses. |
| `runtime.h` | `mi::hip`: device listing and choice, device and pinned host buffers, streams, events, error checks. |
| `safetensors.h`, `npy.h`, `half.h` | Host-only: the safetensors reader (one file or a sharded directory), the `.npy` reader and writer of the checks, fp16/bf16 conversions. |
| `vae.h`, `vae.cpp` | The FLUX.2 VAE decoder at the engine's reference precision: the final latents [128, h, w] to RGB8 [16h, 16w, 3]. `workspace_bytes(h, w)` is its device memory (1.50 GiB at 1024x1024, 3.04 GiB at 1456x1456, 6.02 GiB at 2048x2048, plus 113 MB of weights); `decode()` refuses a size the device cannot hold, or past the kernels' limits, before it runs. |
| `vae_kernels.h`, `vae_kernels.hip` | The decoder's kernels. The products run on the bf16 leaf over bf16 pieces of the fp32 operands, whose products are exact: the convolutions multiply the TF32-rounded operands exactly (2 pieces each), the attention's linears the fp32 operands (3 each), the attention's scores and P V are fp32-accurate; GroupNorm uses the engine's exact statistics. |
| `check_device.sh` | Static check of an AMD device build from its assembly and code object (below). |
| `tools/hip_check.cpp` | The runtime's smoke test. |
| `tools/hip_vae_check.cpp` | The VAE: each kernel against host replays, golden latents against the reference pipeline's PNGs (PSNR, identical bytes), non-square decodes against a host replay of the whole decoder, the size limits. Needs zlib. |

## Building

`ROCM_PATH` is the ROCm 7 root holding `include/hip` and `lib/llvm`, as a CMake cache variable or in the environment. `tools/hip_vae_check` also needs zlib (`zlib1g-dev` or `zlib-devel`).

```sh
cmake -S hip -B build-hip-amd -G Ninja "-DHIP_TARGET=gfx1100;gfx1200;gfx1201"   # or one of them
cmake --build build-hip-amd -j4
cpack --config build-hip-amd/CPackConfig.cmake  # mossinfer-<version>-linux-hip-x86_64.tar.gz: moss-infer-hip, licences
cmake -S hip -B build-hip-nvidia -G Ninja -DHIP_TARGET=nvidia     # CMAKE_CUDA_ARCHITECTURES, default 89
cmake --build build-hip-nvidia -j4
build-hip-amd/hip_check --list    # the devices HIP sees and whether this build runs on each
build-hip-amd/hip_check           # copies and a known-answer GEMM on the first usable device
```

Warnings are errors (`MI_HIP_WERROR`). An AMD build compiles each kernel source a second time on its own for each architecture, device only and with the library's flags, and runs `check_device.sh` on each assembly and code object: each kernel must hold the number of `v_wmma_f32_16x16x16_bf16` the script lists for that architecture, use wave size and workgroup size 32, no scratch, no spills and no dynamic stack, be one of the kernels the script knows, and the code object must have no undefined symbols. The counts are ROCm 7.2.4's; `tools/hip_block/README.md` says how to record another toolchain's. The executables find `libamdhip64` in `$ROCM_PATH/lib` in the build tree; the installed `moss-infer-hip` looks in `/opt/rocm/lib`, then on the loader path (`LD_LIBRARY_PATH` for a ROCm installed elsewhere).

`hip_check` exits 0 when every check passes, 1 on a failure and 2 when no device runs this build (an AMD build on a machine without that GPU, for one). A device whose architecture is not the build's target is listed with the reason and never chosen.

## Runtime

`mi::hip::Device(index)` makes a device current; `index < 0` takes the first device this build runs on. `DeviceBuffer` and `HostBuffer` (page-locked, the source of copies that overlap kernels) free their memory when destroyed; `Stream` is non-blocking; `Event` records, waits across streams and times. Every HIP call that fails throws `std::runtime_error` naming the call and HIP's error string; kernel launches check the launch the same way (`hip_block::` entry points).

The kernel entry points (`hip_block.h`) take device pointers to bf16 bits (`std::uint16_t`) and a stream, launch asynchronously and allocate nothing. Operands are 16-byte aligned with row strides that are multiples of 8 elements.
