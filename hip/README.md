# HIP backend

MOSS-Image's native kernels for AMD GPUs through ROCm, and the same sources through HIP's NVIDIA headers, which run them on a CUDA device with a scalar leaf for checking. It is its own CMake project; the engine's build does not include this directory and nothing here needs CUDA on AMD.

| Target | Compiler | Leaf (`mma_leaf.h`) |
|---|---|---|
| `gfx1100` (RDNA3) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `gfx1200`, `gfx1201` (RDNA4) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `nvidia` | nvcc with HIP's NVIDIA headers | scalar: the exact bf16*bf16 product in fp32, then add, K ascending |

A binary runs only on its target's ISA. Every kernel launches workgroups of one wave (32 threads) and depends on that (`mma_leaf.h`).

## Files

| File | |
|---|---|
| `hip_block.h`, `block_kernels.hip` | One DiT block's kernels and their launches: RMSNorm with modulation, the GEMMs (plain, gated residual, SiTU-GLU), QK norm with RoPE, varlen attention, operand packing. Checked against the engine by `tools/hip_block` (its README has the rounding points and the results on an RX 9060 XT). |
| `gemm_tile.h` | The per-wave GEMM walk the kernel sources share: fragment loads with zero-padded tails, the K-ascending walk over row-major or packed operands, the 2D block order of the tiles, the GEMM kernel and its epilogues (plain, gated residual, bias, bias then residual add), launch helpers. |
| `dit_kernels.h`, `dit_kernels.hip` | The DiT's ops outside its blocks: the linears with a bias, the text refiner's un-modulated RMSNorm, SiLU, SiTU-GLU over a `[hidden \| gate]` GEMM output. |
| `dit.h`, `dit.cpp` | The whole MOSS-Image DiT (below). |
| `mma_leaf.h`, `device_math.h` | The 16x16x16 leaf, and the bf16 conversions and rounded fp32 operations every kernel uses. |
| `runtime.h` | `mi::hip`: device listing and choice, device and pinned host buffers, streams, events, error checks. |
| `safetensors.h`, `npy.h`, `half.h` | Host-only: the safetensors reader (one file or a sharded directory), the `.npy` reader and writer of the checks, fp16/bf16 conversions. |
| `check_device.sh` | Static check of an AMD device build from its assembly and code object (below). |
| `tools/hip_check.cpp` | The runtime's smoke test. |
| `tools/hip_dit_kernel_check.cpp` | `dit_kernels.h`'s ops against host replays. |
| `tools/hip_dit_run.cpp` | The DiT's sampling loop on a golden case, against the official pipeline's outputs. It takes the case's cfg from `cases.json` and the uncond row's skipped blocks from the case's `dit_step0_uncond.skip.json`, and reads them with the engine's JSON reader (`src/util/json.cpp`, compiled in). |

## Building

`ROCM_PATH` is the ROCm root holding `include/hip` and `lib/llvm`, as a CMake cache variable or in the environment.

```sh
cmake -S hip -B build-hip-gfx1200 -G Ninja -DHIP_TARGET=gfx1200   # gfx1100, gfx1201 likewise
cmake --build build-hip-gfx1200 -j4
cmake -S hip -B build-hip-nvidia -G Ninja -DHIP_TARGET=nvidia     # CMAKE_CUDA_ARCHITECTURES, default 89
cmake --build build-hip-nvidia -j4
build-hip-gfx1200/hip_check --list    # the devices HIP sees and whether this build runs on each
build-hip-gfx1200/hip_check           # copies and a known-answer GEMM on the first usable device
build-hip-gfx1200/hip_dit_kernel_check
build-hip-gfx1200/hip_dit_run MODEL/transformer tests/golden g_fridge_1024 [--steps N] [--out FILE.npy] [--dit-budget GiB] [--time-ops] [--expect FILE.npy]
```

`hip_dit_run` fails when step 0's cond or uncond prediction is under cosine 0.99988 against the golden one (the CUDA engine's own test's bar) or its rms difference over 0.0155 of the golden rms (what that cosine allows between vectors of equal norm), on a NaN in any step, and, with every step run, when the final latent is under cosine 0.95 or its rms difference over 0.316 of the golden rms. That bound is a gate for gross errors only: 50 steps of sampling amplify small differences (two Vulkan runs differing only in their text encoder, cosine 0.9998 between those, ended 0.9706 apart), and on these golden text-encoder outputs the Vulkan DiT reaches 0.957 (coop) and 0.971 (plain), the CUDA engine 0.992. It was set from those figures before any HIP run reached the final latent. `--expect` adds an exact comparison of the latent after the steps run with a file `--out` wrote, so a streamed run can be held to a resident run's bits.

Warnings are errors (`MI_HIP_WERROR`). An AMD build compiles each kernel source a second time on its own, device only and with the library's flags, and runs `check_device.sh` on the assembly and code object: each kernel must hold the number of `v_wmma_f32_16x16x16_bf16` the script lists for that target, use wave size and workgroup size 32, no scratch, no spills and no dynamic stack, be one of the kernels the script knows, and the code object must have no undefined symbols. The counts are ROCm 7.2.4's; `tools/hip_block/README.md` says how to record another toolchain's. At run time ROCm's libraries must be on the loader path if they are not installed system-wide: `LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib`.

The checks exit 0 when everything passes, 1 on a failure and 2 when no device runs this build (an AMD build on a machine without that GPU, for one). A device whose architecture is not the build's target is listed with the reason and never chosen.

## Runtime

`mi::hip::Device(index)` makes a device current; `index < 0` takes the first device this build runs on. `DeviceBuffer` and `HostBuffer` (page-locked, the source of copies that overlap kernels) free their memory when destroyed; `Stream` is non-blocking; `Event` records, waits across streams and times. Every HIP call that fails throws `std::runtime_error` naming the call and HIP's error string; kernel launches check the launch the same way (`hip_block::` entry points).

The kernel entry points (`hip_block.h`) take device pointers to bf16 bits (`std::uint16_t`) and a stream, launch asynchronously and allocate nothing. Operands are 16-byte aligned with row strides that are multiples of 8 elements.

## DiT

`dit.h` runs MOSSImageTransformer with the op order and rounding of the engine's CUDA forward (`src/models/moss_dit`): the time embedding (a float64 sinusoid on the host, then the bf16 MLP and the shared block and head modulations as linears with M = 1), the video embedder, the text refiner once per prompt (`encode_text`), the 42 blocks over the rows' joint sequences `[video_r | text_r]` concatenated, a block that some rows skip running as one varlen batch per run of the rows that execute it (the uncond row's skip set, {22} or {24} in the golden cases), and the head. The RoPE table is a kernel with the expressions of the engine's (`rope.cu`, device `sinf` and `cosf`). The video tokens and the transpose of the output are host code; CFG and the Euler step belong to the caller (`hip_dit_run` does them in fp32 on the host, in the engine's order). A forward of more joint tokens than the engine's `kMaxRunRows` (61506) is refused, as the engine refuses it.

The blocks run `hip_block`'s packed path. Their GEMM weights are packed into its fragment layout on the host while loading (9.5 GiB of bf16 weights in all, about 216 MiB per block). `DitOptions::device_budget` caps the device memory of the weights: the blocks past it stay in pinned host memory and are copied into one of two device slots on a second stream, each copy waiting for the block that last used its slot and each block waiting for its copy, so the next block's copy runs while the current one computes. Without a budget every block is resident. A budget below the resident globals and refiner plus the two slots (about 1.03 GiB) is refused. Streaming does not change a bit of the result. The activations at 1024x1024 with a 62-token and a 10-token prompt take about 0.7 GiB (`Dit::activation_bytes`).

The refiner's norm follows torch's reduce order for its row count: a prompt shorter than 16 tokens (the golden uncond prompt has 10) gives more threads per row there, and `hip_block::rms_norm` sums in that order where `rms_norm_modulate` keeps the 32-lane one. The GEMMs accumulate in fp32 with K ascending, as every GEMM here; the engine's cuBLASLt picks its own order, so their sums can differ in the last bit before rounding.

## Verified

On an RTX 4090 Laptop (sm_89) with the NVIDIA build, CUDA 13, ROCm 7.2.4 headers; the AMD builds with ROCm 7.2.4's clang.

- `hip_dit_run`, step 0 against the official pipeline's predictions (cosine; rms of the difference over the golden rms):

| Case | Uncond skips | cond | uncond |
|---|---|---|---|
| g_fridge_1024 (prompts of 62 and 10 tokens) | {22} | 0.999976078; 0.00695 | 0.999984162; 0.00563 |
| g_short_1024 (29 and 10 tokens) | {24} | 0.999972706; 0.00739 | 0.999983890; 0.00571 |

- Two steps of g_fridge_1024 with `--dit-budget 4` (14 blocks resident, 28 streamed through the two slots from 5.91 GiB of pinned memory) give the all-resident run's latent bit for bit (`--expect`), and no step has a NaN.
- A budget of 0.5 GiB is refused before any work, naming the 1.03 GiB the DiT needs at least.
- `hip_dit_kernel_check` passes, three runs of three:
  - the bias GEMMs bit for bit, at the DiT's shapes, with tails on M, N and K, and at 705 x 650 x 3072, where both M and N span more than one cache block of tiles;
  - `rms_norm`'s sums of squares bit for bit at every row count from 1 to 16 and at 62 and 100 (512 down to 32 threads per row), with 1 to 5 rows at every count below 16 whose sum the 32-lane order rounds differently; its rsqrt within 1 ULP of the correctly rounded one (limit 2); every output equal to the replay from the device's rsqrt;
  - `silu` and `situ_glu` with every output equal to the host replay;
  - `rope_table` within 1 ULP of a host replay everywhere and equal in 99.9998 % or more of the values.
- `hip_check` passes.
- gfx1100, gfx1200 and gfx1201 compile with warnings as errors. `check_device.sh` on `dit_kernels.hip`, 6 kernels: the two bias GEMM kernels hold 12 WMMA each, 92 VGPRs on gfx12 and 110 on gfx1100, 6656 bytes of LDS; `rms_norm_kernel` 15 VGPRs and 2048 bytes of LDS, `rope_table_kernel` 21 VGPRs (22 on gfx1100), `silu_kernel` 9, `situ_glu_kernel` 13; no scratch, no spills, wave and workgroup size 32, no undefined symbols. `block_kernels.hip`'s device assembly is the same as before it moved here, apart from the source-path hash in `__hip_cuid_`.

On the scalar leaf a step takes 590 to 720 s on that GPU, over 90 % of it in attention: the scalar leaf checks logic, and its times say nothing about an AMD card. A 50-step run was not made there.

## Not verified

- Anything on an AMD card: the DiT, its new kernels and the streaming have only been compiled for gfx1100, gfx1200 and gfx1201. `rms_norm` below 16 rows (its halving through `__shared__` memory) has only run on NVIDIA's scalar build.
- The final latent of a full sampling run, on any device; `hip_dit_run`'s final-latent bound has not been met or missed by any HIP run.
