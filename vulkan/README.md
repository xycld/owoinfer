# Vulkan backend

The fallback for GPUs the CUDA engine does not support: `moss-infer-vulkan`, a standalone program built from this
directory without CUDA (user guide: [docs/vulkan.md](../docs/vulkan.md)). CUDA stays the path on every GPU it supports.
The program compiles in the engine's host sources it shares with `moss-infer` (tokenizer, initial noise, schedule, PNG
writer); the engine's own build does not include this directory.

Every model runs on one of three kernel paths, chosen per device (the first the device has) and reported:

| Path | Needs | |
|---|---|---|
| `coop` | Vulkan 1.3, `VK_KHR_cooperative_matrix` with `VK_KHR_shader_bfloat16` (16x16x16 bf16 x bf16 -> f32 in subgroup scope), compute subgroup size 32 with vote and shuffle, 40 KiB of shared memory | `Capabilities::coop_bf16` |
| `coop-f16` | the same with 16x16x16 f16 x f16 -> f32 instead of bf16 fragments; the VAE runs `plain` | `Capabilities::coop_f16` |
| `plain` | Vulkan 1.2, buffer device addresses, 64-bit integers, 32 KiB of shared memory | every other device; no cooperative matrices, 16-bit arithmetic or particular subgroup size |

The Vulkan loader is opened at run time (`vk_api.h`), so nothing links against it.

## Build and check

```sh
cmake -S vulkan -B build-vulkan -G Ninja
cmake --build build-vulkan
build-vulkan/vulkan_check --list                 # each device, and why a device gets the plain path
build-vulkan/vulkan_block_check [--path P]       # the DiT block's kernels against host replays, then one block's timing
build-vulkan/vulkan_block_real DIR               # real blocks against the engine's output (files captured from its forward)
build-vulkan/vulkan_dit_run MODEL/transformer tests/golden g_fridge_1024 [--steps N] [--dit-budget GiB]
build-vulkan/vulkan_te_kernel_check              # the text encoder's kernels against host replays
build-vulkan/vulkan_te_check --golden tests/golden
build-vulkan/vulkan_vae_check                    # the VAE's kernels, then the golden latent against the golden PNG
```

It needs the Vulkan headers with `VK_KHR_shader_bfloat16`, a `glslc` that compiles `GL_EXT_bfloat16`, and zlib for the
VAE check. Shaders compile into the binary. Every check exits non-zero on a failure, and passes under the Khronos
validation layer and on llvmpipe (Mesa's CPU driver: 8-wide subgroups, 32 KiB of shared memory, the plain path).
`MI_VK_PIPELINE_STATS=1` prints each pipeline's registers and spills where the driver reports them.

## DiT

`block.h` holds one DiT block's kernels on every path, with the engine's rounding points: RMSNorm with modulation, the
GEMMs with bias, gated-residual and SiTU-GLU epilogues, QK norm with RoPE, and flash attention (an online softmax over
32-key steps; the `coop` one rescales its output only when a row's maximum grew). `dit.h` runs the whole
MOSSImageTransformer on them: the time embedding and shared modulation, the text refiner, the joint sequences of the CFG
rows with per-row skipped blocks, and the head.

The blocks that do not fit next to the activations (from `VK_EXT_memory_budget`) stay in host memory. Two device slots
alternate: each streamed block is copied on the transfer queue while the block before it runs. Every op over many
tokens runs in token ranges of at most 200 GFLOP (`coop`) or 20 GFLOP (`plain`) per submit, attention in query ranges,
so a slow device does not trip a display driver's watchdog. Neither changes a bit of the result.

Against the engine's own output on real blocks of the 1024x1024 golden case (captured from its forward):

| Block | Path | Same bits | Max scaled ULPs | Cosine |
|---:|---|---:|---:|---:|
| 1 | coop | 98.14 % | 1.00 | 0.999999999 |
| 22 | coop | 96.16 % | 2.00 | 0.999999989 |
| 41 | coop | 98.56 % | 32.00 | 0.999999919 |

The `coop` GEMMs round like the tensor cores, hence so many identical bits. Step 0 of g_fridge_1024 against the official
pipeline: cosine 0.999976 (cond) and 0.999983 (uncond, block 22 skipped); the CUDA engine's test asks for 0.99988.

`coop-f16` runs the `coop` kernels built with `F16`. The operands stay bf16 in memory and become f16 as they are
staged: `row_scale.comp` finds, per row of each GEMM's A and B, the power of two that puts the row's largest value in
[2^15, 2^16), where every bf16 value is an f16 value; the GEMM scales the rows by it on load and the accumulators back
before the epilogue, all exactly, so every product is the bf16 one while the values are above 2^-29 of their row's
largest. Attention converts q, k and v unscaled (bits lost below 2^-14, infinite above 65504) and feeds P as
bf16(p) * 2^15. On the 4090 it gives the table above to
the digit (98.14 %, 96.16 %, 98.56 %), step 0 at 0.999976 and 0.999985, and the text encoder's figures of `coop`;
`vulkan_block_check` adds a GEMM case whose rows span 2^-90 to 2^100. Its staging conversions and row scans cost
about 7 % of a 512x512 step there, nothing measurable at 1024x1024.

## VAE

`vae.h` is the FLUX.2 decoder from the final latents to RGB8. Its convolutions round both operands to TF32 as cuDNN does
(on `coop` each TF32 operand is two exact bf16 pieces), so every product is torch's and only the fp32 accumulation order
differs; GroupNorm, the mid attention and the upsampling follow the engine. It decodes in one piece: about 1.6 GiB at
1024x1024. The golden latent against the official pipeline's PNG: 69.39 dB (`coop`), 69.35 dB (`plain`), 99.25 % of the
bytes identical and none off by more than 1, as the CUDA engine's reference decode.

## Text encoder

`text_encoder.h` runs the Qwen3.5 text encoder (`Qwen3_5Model(input_ids).last_hidden_state`, bf16 [S, 4096] per prompt)
for several prompts at once. Its GEMMs are `BlockKernels`' on either path. Its own kernels (`te_kernels.h`) need no
cooperative matrices, 16-bit arithmetic or particular subgroup size, so every path shares them: the Qwen RMSNorms (with
torch's reduction order for the row count, and RoPE fused into the q/k norm), conv1d with SiLU, the GatedDeltaNet gates,
the engine's chunked gated delta rule in fp32 (four kernels, each within 32 KiB of shared memory), causal grouped-query
attention at head_dim 256, and the gated products. Weights stream from the safetensors files: four threads read layer
k + 2 into host staging while layer k runs and layer k + 1 is copied to the device on the transfer queue
(`Device::copy_async`). The embedding table stays on the host. Device memory is two layer buffers (833 MiB) plus about
130 KiB per token.

```sh
build-vulkan/vulkan_te_kernel_check                       # each kernel against a host replay of its op order
build-vulkan/vulkan_te_check --golden tests/golden        # the encoder against the official pipeline's hidden states
build-vulkan/vulkan_te_check --golden tests/golden --case g_long4096_1024 --profile
```

Against the goldens of the official PyTorch pipeline on an RTX 4090 Laptop, last_hidden_state, with torch's own
deviation from the same goldens when one of its kernels changes (the worse of SDPA math and cuBLASLt,
`tests/data/qwen35_text/noise_floor.json`):

| Prompt | Tokens | Cosine, coop | Cosine, plain | Cosine, torch | Mean row rel. L2, coop | plain | torch |
|---|---:|---:|---:|---:|---:|---:|---:|
| g_fridge cond | 62 | 0.9998786 | 0.9999012 | 0.9998949 | 1.32e-2 | 1.27e-2 | 1.29e-2 |
| g_fridge uncond | 10 | 0.9998012 | 0.9997889 | 0.9998333 | 1.84e-2 | 1.90e-2 | 1.60e-2 |
| g_short cond | 29 | 0.9998910 | 0.9997967 | 0.9999026 | 1.40e-2 | 1.65e-2 | 1.33e-2 |
| g_layout cond | 172 | 0.9998827 | 0.9998885 | 0.9998857 | 1.43e-2 | 1.40e-2 | 1.44e-2 |
| g_layout uncond | 16 | 0.9998486 | 0.9998133 | 0.9998634 | 1.64e-2 | 1.81e-2 | 1.52e-2 |
| g_long4096 cond | 4096 | 0.9985149 | 0.9984682 | 0.9992133 | 2.29e-2 | 2.32e-2 | 1.89e-2 |

Differences grow over the 32 layers until they reach torch's own sensitivity to a kernel change; after the first layer
they are at the level of one bf16 rounding (layer 0, row relative L2 at most 7.1e-3 on both paths).

Times on that GPU with nothing else running, the weights read from a drive that held little of them in the page cache
(about 2.5 GB/s with four readers), which bounds the short prompts:

| Prompts | Tokens | Coop: encode, of it compute | Plain: encode, of it compute |
|---|---:|---:|---:|
| g_fridge cond + uncond | 72 | 6.7 s, 0.6 s | 6.8 s, 2.8 s |
| g_long4096 cond + uncond | 4112 | 8.4 s, 7.2 s | 186 s, 185 s |

At 4112 tokens on the cooperative-matrix path the GEMMs take 5.6 s of the 7.2 s, attention 0.7 s and the gated delta rule
0.8 s (`--profile`).

## Files

| File | Purpose |
|---|---|
| `vk_api.h`, `vk_api.cpp` | the Vulkan functions, resolved from the loader at run time |
| `runtime.h`, `runtime.cpp` | device selection, capabilities, buffers, pipelines, barriers, transfer-queue copies |
| `block.h`, `block.cpp` | the DiT block's kernels on every path |
| `dit.h`, `dit.cpp` | the DiT: text refiner, forward, block streaming |
| `text_encoder.h`, `te_kernels.h` and their `.cpp` | the Qwen3.5 text encoder |
| `vae.h`, `vae.cpp` | the FLUX.2 VAE decoder |
| `generate.h`, `generate.cpp` | text to image: the three models one after another, CFG and the official Euler step |
| `gemm.h`, `gemm.cpp` | the standalone f16 GEMM of the first bring-up (`vulkan_check`) |
| `safetensors.h`, `npy.h`, `half.h` | readers and conversions |
| `shaders/` | compute shaders |
| `tools/moss_infer_vulkan.cpp` | the program |
| `tools/` | checks and benchmarks |
