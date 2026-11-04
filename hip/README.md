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
| `gemm_tile.h` | The per-wave GEMM walk the kernel sources share: fragment loads with zero-padded tails, the K-ascending walk over row-major or packed operands, the 2D block order of the tiles, the GEMM kernel and its epilogues (plain, gated residual, bias, bias then residual add), launch helpers. |
| `dit_kernels.h`, `dit_kernels.hip` | The DiT's ops outside its blocks: the linears with a bias, the text refiner's un-modulated RMSNorm, SiLU, SiTU-GLU over a `[hidden \| gate]` GEMM output. |
| `dit.h`, `dit.cpp` | The whole MOSS-Image DiT (below). |
| `mma_leaf.h`, `device_math.h` | The 16x16x16 leaf, and the bf16 conversions and rounded fp32 operations every kernel uses. |
| `runtime.h` | `mi::hip`: device listing and choice, device and pinned host buffers, streams, events, error checks. |
| `safetensors.h`, `npy.h`, `half.h` | Host-only: the safetensors reader (one file or a sharded directory), the `.npy` reader and writer of the checks, fp16/bf16 conversions. |
| `vae.h`, `vae.cpp` | The FLUX.2 VAE decoder at the engine's reference precision: the final latents [128, h, w] to RGB8 [16h, 16w, 3]. `workspace_bytes(h, w)` is its device memory (1.50 GiB at 1024x1024, 3.04 GiB at 1456x1456, 6.02 GiB at 2048x2048, plus 113 MB of weights); `decode()` refuses a size the device cannot hold, or past the kernels' limits, before it runs. |
| `vae_kernels.h`, `vae_kernels.hip` | The decoder's kernels. The products run on the bf16 leaf over bf16 pieces of the fp32 operands, whose products are exact: the convolutions multiply the TF32-rounded operands exactly (2 pieces each), the attention's linears the fp32 operands (3 each), the attention's scores and P V are fp32-accurate; GroupNorm uses the engine's exact statistics. |
| `check_device.sh` | Static check of an AMD device build from its assembly and code object (below). |
| `tools/hip_check.cpp` | The runtime's smoke test. |
| `tools/hip_vae_check.cpp` | The VAE: each kernel against host replays, golden latents against the reference pipeline's PNGs (PSNR, identical bytes), non-square decodes against a host replay of the whole decoder, the size limits. Needs zlib. |
| `tools/hip_dit_kernel_check.cpp` | `dit_kernels.h`'s ops against host replays. |
| `tools/hip_dit_run.cpp` | The DiT's sampling loop on a golden case, against the official pipeline's outputs. It takes the case's cfg from `cases.json` and the uncond row's skipped blocks from the case's `dit_step0_uncond.skip.json`, and reads them with the engine's JSON reader (`src/util/json.cpp`, compiled in). |
| `text_encoder.h`, `text_encoder.cpp` | The Qwen3.5 text encoder (`last_hidden_state` of the MOSS prompts), in the engine's op order and rounding, streaming its weights from the files one layer at a time. |
| `te_kernels.h`, `te_kernels.hip` | The text encoder's kernels: the Qwen RMSNorms with the text RoPE, the GatedDeltaNet's conv1d + SiLU, gates and chunked gated delta rule, causal GQA attention at head dim 256 with the output gate, the MLP's SiLU product; its GEMMs are `hip_block`'s. |
| `tools/hip_te_kernel_check.cpp` | `te_kernels.h`'s ops against host replays of their op order, and the gated delta rule against an fp64 recurrence. |
| `tools/hip_te_check.cpp` | The whole text encoder against the official pipeline's golden hidden states, gated as the engine's test (`tests/unit/test_qwen35_text.cpp`). |
| `generate.h`, `generate.cpp` | Text to image: the VAE's weights first, the text encoder, the DiT's sampling loop with CFG and the official Euler step on the host in the engine's order, the VAE decode. The caller supplies the token ids, the initial noise and the schedule; a VRAM budget the run cannot keep to is refused before the first step, with the sizes. |
| `tools/moss_infer_hip.cpp` | `moss-infer-hip`, the standalone program: `moss-infer generate`'s flags that the backend runs (reference tier), with the engine's tokenizer, initial noise, schedule and PNG writer compiled in (`mossinfer_hip_host`), so a seed gives the same inputs as on CUDA. |

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
build-hip-amd/hip_dit_kernel_check
build-hip-amd/hip_dit_run MODEL/transformer tests/golden g_fridge_1024 [--steps N] [--out FILE.npy] [--dit-budget GiB] [--time-ops] [--expect FILE.npy]
build-hip-amd/moss-infer-hip --model MODEL --prompt "..." -o out.png [--size WxH] [--steps N] [--vram-budget GiB]
```

`hip_dit_run` fails when step 0's cond or uncond prediction is under cosine 0.99988 against the golden one (the CUDA engine's own test's bar) or its rms difference over 0.0155 of the golden rms (what that cosine allows between vectors of equal norm), on a NaN in any step, and, with every step run, when the final latent is under cosine 0.95 or its rms difference over 0.316 of the golden rms. That bound is a gate for gross errors only: 50 steps of sampling amplify small differences (two Vulkan runs differing only in their text encoder, cosine 0.9998 between those, ended 0.9706 apart), and on these golden text-encoder outputs the Vulkan DiT reaches 0.957 (coop) and 0.971 (plain), the CUDA engine 0.992. It was set from those figures before any HIP run reached the final latent. `--expect` adds an exact comparison of the latent after the steps run with a file `--out` wrote, so a streamed run can be held to a resident run's bits.

Warnings are errors (`MI_HIP_WERROR`). An AMD build compiles each kernel source a second time on its own for each architecture, device only and with the library's flags, and runs `check_device.sh` on each assembly and code object: each kernel must hold the number of `v_wmma_f32_16x16x16_bf16` the script lists for that architecture, use wave size and workgroup size 32, no scratch, no spills and no dynamic stack, be one of the kernels the script knows, and the code object must have no undefined symbols. The counts are ROCm 7.2.4's; `tools/hip_block/README.md` says how to record another toolchain's. The executables find `libamdhip64` in `$ROCM_PATH/lib` in the build tree; the installed `moss-infer-hip` on `LD_LIBRARY_PATH`, then in `/opt/rocm/lib` (its RUNPATH), then in the system's library paths.

The checks exit 0 when everything passes, 1 on a failure and 2 when no device runs this build (an AMD build on a machine without that GPU, for one). A device whose architecture is not one of the build's targets is listed with the reason and never chosen.

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

On a Radeon RX 9060 XT (gfx1200), built on the card with its ROCm (HIP 7.16, AMD clang 24), 2026-11-04:

- `check_device.sh` on the four kernel sources with the expected WMMA counts; `hip_check`, `hip_dit_kernel_check` (`rms_norm` at every row count from 1 to 16 included) and `hip_te_kernel_check` pass.
- `hip_dit_run` step 0: g_fridge_1024 0.999975845 / 0.999983823 (rms 0.00726 / 0.00581), g_fridge_1456 0.999978707 / 0.999985122. 50 steps of g_fridge_1024 on the golden text-encoder outputs: final latent cosine 0.98658, rms diff 0.143 of the golden rms. `hip_dit_run` takes 4.25 s per step at 1024x1024, 14.3 s at 1456x1456.
- Per-op device time of a 1024x1024 step (ms): attention 2331, up_gate GEMM+SiTU 754, down GEMM 441, qkv GEMM 403, o GEMM 167, the rest 82.
- `hip_te_check` on every golden sequence: within 2.5x of torch's floor; the 4096-token prompt at 2.18x (layer 31) and 2.07x (last hidden).
- `moss-infer-hip` g_fridge_1024 end to end, 50 steps: final latent cosine 0.957 to the golden one, PNG 26.9 dB, the same image with small layout shifts. Streamed runs (5 blocks resident at 512x512, 2 and 17 at 1024x1024) give the resident latent byte for byte. The CI's release package (ROCm 7.2.4) gives the card build's latent byte for byte at 512x512.

## Not verified

- RDNA3 (gfx1100) and gfx1201 on a card: built and checked by `check_device.sh` only.
