# Vulkan GEMM bring-up

The first artefact of the Vulkan backend work ([research](../../docs/vulkan-backend-research.md),
[design](../../docs/vulkan-backend-design.md)): a standalone tool that runs one f16 GEMM on a Vulkan device with
cooperative matrices, checks it against a CPU reference and times it at the engine's matrix shapes.

It links no engine code and is not part of the engine build. The point is to answer the two questions the design
document leaves open before any integration work starts:

1. Does `VK_KHR_cooperative_matrix` give a correct f16 GEMM with f32 accumulation on this device?
2. How fast is it, compared with cuBLASLt on the same GPU?

## What the kernel does

`gemm_f16.comp` computes

```
C[M,N] = A[M,K] * B[K,N]
```

with `A` row-major (stride `K`) and `B` column-major (stride `K`), i.e. `B[k,n]` lives at `n*K + k`. That is exactly the
engine's weight layout (`[N,K]` row-major), so the GEMM consumes weights as stored, with no transpose.

- One workgroup per 64x64 output tile, 128 threads: four full subgroups of 32, which the pipeline requires.
- Each subgroup owns a 32x32 quarter, computed as four `16x16x16` cooperative-matrix operations (`f16` inputs, `f32`
  accumulate) per K step, 16 elements at a time.
- Output tiles are walked in bands of 8 N-tiles (the same idea as the engine's `BandedSwizzle` for INT8 GEMMs): a band
  of weight tiles stays in L2 while every M tile streams past it.
- Arguments (buffer device addresses, sizes, strides) go through push constants; the buffers are addressed with
  `GL_EXT_buffer_reference`, so there are no descriptor sets.

## Build

With the Vulkan SDK installed:

```sh
cmake -S tools/vulkan_gemm -B build-vulkan-gemm -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan-gemm
```

The SDK's `glslc` compiles the shader; the build fails with a clear message when either the SDK or `glslc` is missing.

Without the SDK (for example a CI machine or a user who only has the driver), pass the two paths explicitly:

```sh
cmake -S tools/vulkan_gemm -B build-vulkan-gemm \
  -DVULKAN_INCLUDE_DIR=<a Vulkan-Headers checkout>/include \
  -DVULKAN_LIBRARY=<vulkan-1.lib>
```

Notes from a machine without administrator rights (where the SDK installer cannot run):

- Headers: the [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) repository's `include/vulkan` is enough.
  Its `vulkan_core.h` includes `vk_video/*`, which is only in the SDK; a local copy either adds those headers or
  defines the handful of `StdVideo*` types the core header names (the tool never touches video).
- Import library: `dumpbin /exports C:\Windows\System32\vulkan-1.dll`, keep the `vk*` names in a `.def` file, and
  `lib /def:vulkan-1.def /machine:x64 /out:vulkan-1.lib`.
- `glslc` can be extracted from the SDK installer with 7-Zip (`7z x VulkanSDK.exe -o<sdk>`); it lands in `Bin/`.

## Run

```sh
build-vulkan-gemm/vulkan_gemm                      # the checks, then the engine's shapes
build-vulkan-gemm/vulkan_gemm --device 1           # another device; without it, the first one that can run the kernel
build-vulkan-gemm/vulkan_gemm --bench 8222 9216 3072 5 --no-check
```

`N` must be a multiple of 64 and `K` of 16. `M` is free: the kernel runs whole 64-row tiles, so the tool allocates A and C
with `M` rounded up to 64 rows. The exit code is non-zero when a check fails. The CPU reference is `O(M*N*K)`, so it runs
for shapes up to 2^31 multiply-adds; the defaults check one tile, a partial last band, several bands, and row tails with
NaN in the padding rows.

The tool refuses a device it cannot run on and says why: no Vulkan 1.3, no `VK_KHR_cooperative_matrix`, no
16x16x16 f16 x f16 -> f32 subgroup shape, or no way to fix the compute subgroup size at 32.

## Results

RTX 4090 Laptop, NVIDIA driver 615.78, Vulkan 1.4.351, against cuBLASLt f16 inputs with f32 compute on the same GPU in the
same session. The shapes are the DiT's at 1024x1024 with CFG (M = 8222) and the text encoder's:

| Shape (M x N x K) | Vulkan | cuBLASLt | Vulkan / cuBLASLt |
|---|---:|---:|---:|
| 8222 x 9216 x 3072 (QKV) | 20.1 TFLOPS | 52.6 | 38 % |
| 8222 x 3072 x 3072 (projection) | 31.1 | 57.4 | 54 % |
| 8222 x 16384 x 3072 (up/gate) | 19.7 | 45.5 | 43 % |
| 8222 x 3072 x 8192 (down) | 13.5 | 70.1 | 19 % |
| 282 x 12288 x 4096 | 25.0 | 59.5 | 42 % |
| 282 x 4096 x 12288 | 12.7 | 52.2 | 24 % |

RTX 2080 8 GiB, driver 621.01, at the earlier 3072-row shapes: 8-9 TFLOPS. The checks match a double-precision CPU
reference to within fp16 input and f32 accumulator rounding (1e-5 at 64 x 64 x 64, 5e-5 at 512^3).

## Status and limitations

- The fragments load straight from global memory, so large K suffers most. Staging A and B tiles in shared memory is the
  next step for speed.
- No bias or epilogue fusion, no INT8/INT4 path, no path for devices without cooperative matrices, one queue only.
- N must be a multiple of 64. Every N in the engine's DiT and text encoder GEMMs is.
- Vulkan's synchronization validation does not report on this kernel's accesses: it stays quiet with or without barriers,
  because they go through buffer device addresses. Core validation runs clean.

## Files

| File | Purpose |
|---|---|
| `gemm_f16.comp` | the cooperative-matrix GEMM (GLSL 460, compiled by `glslc`) |
| `main.cpp` | device selection and setup, buffers, pipeline, CPU check, timing |
| `probe.py` | optional: lists the device's cooperative-matrix shapes and memory heaps using only the Vulkan loader (no SDK) |
| `CMakeLists.txt` | standalone build |
