# Vulkan backend

The fallback for hardware the engine does not otherwise support. CUDA stays the path on every GPU it supports. This is a
standalone project for now: it links no engine code and nothing here affects the engine's build. Design and research:
[design](../docs/vulkan-backend-design.md), [research](../docs/vulkan-backend-research.md).

What exists is the device runtime, the GEMM and one DiT block's kernels. It cannot generate images yet: the text encoder, the VAE, the DiT's outer layers and the pipeline are still to come.

## Build and check

```sh
cmake -S vulkan -B build-vulkan
cmake --build build-vulkan
build-vulkan/vulkan_check --list          # each device, and which GEMM it gets
build-vulkan/vulkan_check                 # checks against an fp64 CPU reference, then the engine's shapes
build-vulkan/vulkan_check --path plain --no-bench
```

It needs the Vulkan headers and loader, and `glslc`. Shaders compile into the binary. The exit code is non-zero when a check
fails. A check poisons C and uses strides larger than the rows, so a write past a row or past M fails it.

## GEMM

`C[M,N] = A[M,K] * B[N,K]^T`, with B in the engine's weight layout, f32 accumulation and f32 output. M and N are free.

| Path | Used when | Inputs | K |
|---|---|---|---|
| Cooperative matrix (`gemm_coop.comp`) | the device has 16x16x16 f16 x f16 -> f32 in subgroup scope, a compute subgroup size fixable at 32, and 24 KiB of shared memory | f16 | multiple of 16 |
| Plain compute (`gemm_plain.comp`) | every other device with Vulkan 1.2, buffer device addresses and 64-bit integers | f16 or bf16 | multiple of 4 |

`--list` says why a device gets the plain path. The plain shader declares only `Shader`, `Int64` and
`PhysicalStorageBufferAddresses`: it reads 32-bit words, so it needs no 16-bit support and does not depend on the subgroup
size.

## Results

TFLOPS on an RTX 4090 Laptop, driver 615.78, with cuBLASLt (f16 inputs, f32 compute) on the same GPU. M = 8222 is the DiT at
1024x1024 with CFG.

| Shape (M x N x K) | Cooperative matrix | Plain | cuBLASLt |
|---|---:|---:|---:|
| 8222 x 9216 x 3072 (QKV) | 27.0 | 1.39 | 52.6 |
| 8222 x 3072 x 3072 (projection) | 33.1 | 1.62 | 57.4 |
| 8222 x 16384 x 3072 (up/gate) | 23.5 | 1.14 | 45.5 |
| 8222 x 3072 x 8192 (down) | 24.3 | 1.52 | 70.1 |
| 282 x 12288 x 4096 (text encoder) | 27.3 | 1.02 | 59.5 |
| 282 x 4096 x 12288 (text encoder) | 18.1 | | 52.2 |

Checks pass on that GPU on both paths, and through the plain path on llvmpipe (Mesa's CPU driver, 8-wide subgroups, 32 KiB of
shared memory), with core validation clean.

## DiT block

`block.h` is the plain compute path of one bf16 DiT block: RMSNorm with modulation, the GEMMs with their gated residual and SiTU-GLU epilogues, QK norm with RoPE, and varlen attention with an online softmax. It has the engine's rounding points and needs no cooperative matrices, 16-bit support or particular subgroup size.

```sh
build-vulkan/vulkan_block_check            # each kernel against a host replay of its op order, then one block
build-vulkan/vulkan_block_real DIR         # real blocks against the engine's output, from files captured from its forward
```

On real weights and activations at the 1024x1024 golden case, against the engine:

| Block | Same bits | p99 scaled ULPs | Max scaled ULPs | Cosine |
|---:|---:|---:|---:|---:|
| 1 | 83.98 % | 0.06 | 2.00 | 0.999999977 |
| 22 | 72.36 % | 0.25 | 4.00 | 0.999999867 |
| 41 | 69.23 % | 1.00 | 96.00 | 0.999997331 |

Bits differ because fp32 scalar and tensor-core accumulation are not bit-compatible. Known-answer checks pass on an RTX 4090 Laptop and on llvmpipe. One block of 8484 tokens takes about 2 s on that GPU through this path, against 31 ms for the engine, so about 86 s per CFG step at 1024x1024.

## Numerics

The cooperative-matrix path rounds like the tensor cores: its error matches cuBLASLt on the same data up to K = 1040 and is
biased toward zero. At K = 4096 it reaches 8.1e-4 against cuBLASLt's 1.3e-4, on values near 80. The plain path has no such
bias.

## Files

| File | Purpose |
|---|---|
| `runtime.h`, `runtime.cpp` | device selection, capabilities, buffers, pipelines, barriers |
| `gemm.h`, `gemm.cpp` | path selection, argument checks, dispatch |
| `block.h`, `block.cpp` | one DiT block's plain compute kernels |
| `half.h` | host fp16 and bf16 conversions |
| `shaders/` | compute shaders |
| `tools/vulkan_check.cpp` | device list, GEMM checks, timing |
| `tools/vulkan_block_check.cpp` | the block's kernels against host replays, one block's timing |
| `tools/vulkan_block_real.cpp` | real blocks against the engine's captured output |
| `tools/probe.py` | a device's cooperative-matrix shapes and heaps, using only the Vulkan loader (no headers) |

## Text encoder

`text_encoder.h` runs the Qwen3.5 text encoder (`Qwen3_5Model(input_ids).last_hidden_state`, bf16 [S, 4096] per prompt)
for several prompts at once. Its GEMMs are `BlockKernels`' on either path. Its own kernels (`te_kernels.h`) need no
cooperative matrices, 16-bit arithmetic or particular subgroup size, so both paths share them: the Qwen RMSNorms (with
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
