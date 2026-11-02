# Vulkan backend

The fallback for hardware the engine does not otherwise support. CUDA stays the path on every GPU it supports. This is a
standalone project for now: it links no engine code and nothing here affects the engine's build. Design and research:
[design](../docs/vulkan-backend-design.md), [research](../docs/vulkan-backend-research.md).

What exists is the device runtime and the GEMM. It cannot generate images yet.

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

## Numerics

The cooperative-matrix path rounds like the tensor cores: its error matches cuBLASLt on the same data up to K = 1040 and is
biased toward zero. At K = 4096 it reaches 8.1e-4 against cuBLASLt's 1.3e-4, on values near 80. The plain path has no such
bias.

## Files

| File | Purpose |
|---|---|
| `runtime.h`, `runtime.cpp` | device selection, capabilities, buffers, pipelines, barriers |
| `gemm.h`, `gemm.cpp` | path selection, argument checks, dispatch |
| `half.h` | host fp16 and bf16 conversions |
| `shaders/` | compute shaders |
| `tools/vulkan_check.cpp` | device list, checks, timing |
| `tools/probe.py` | a device's cooperative-matrix shapes and heaps, using only the Vulkan loader (no headers) |
