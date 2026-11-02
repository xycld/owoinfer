# Vulkan GEMM bring-up

The first artefact of the Vulkan backend work ([research](../docs/vulkan-backend-research.md),
[design](../docs/vulkan-backend-design.md)): a standalone tool that runs one f16 GEMM on a Vulkan device with
cooperative matrices, checks it against a CPU reference and times it at the engine's matrix shapes.

It links no engine code and is not part of the engine build. The point is to answer the two questions the design
document leaves open before any integration work starts:

1. Does `VK_KHR_cooperative_matrix` give a correct f16 GEMM with f32 accumulation on this device?
2. How fast is it, compared with the cuBLASLt path the engine uses?

## What the kernel does

`gemm_f16.comp` computes

```
C[M,N] = A[M,K] * B[K,N]
```

with `A` row-major (stride `K`) and `B` column-major (stride `K`), i.e. `B[k,n]` lives at `n*K + k`. That is exactly the
engine's weight layout (`[N,K]` row-major), so the GEMM consumes weights as stored, with no transpose.

- One workgroup per 64x64 output tile, 128 threads (four 32-wide subgroups).
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

Two notes from a machine without administrator rights (where the SDK installer cannot run):

- Headers: the [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) repository's `include/vulkan` is enough.
  Its `vulkan_core.h` includes `vk_video/*`, which is only in the SDK; a local copy either adds those headers or
  defines the handful of `StdVideo*` types the core header names (the tool never touches video).
- Import library: `dumpbin /exports C:\Windows\System32\vulkan-1.dll`, keep the `vk*` names in a `.def` file, and
  `lib /def:vulkan-1.def /machine:x64 /out:vulkan-1.lib`.
- `glslc` can be extracted from the SDK installer with 7-Zip (`7z x VulkanSDK.exe -o<sdk>`); it lands in `Bin/`.

## Run

```sh
build-vulkan-gemm/vulkan_gemm                      # check 64^3, then the engine shapes
build-vulkan-gemm/vulkan_gemm --bench 3072 3072 3072 10
build-vulkan-gemm/vulkan_gemm --bench 3072 12288 3072 5 --no-check
```

`M` and `N` must be multiples of 64 and `K` a multiple of 16 (the kernel has no bounds checks yet). `--spv` selects a
different SPIR-V file. The CPU reference is `O(M*N*K)`, so it only runs for shapes up to 2^26 multiply-adds (64³, not
512³ or the engine shapes); `--no-check` disables it.

Output on an RTX 2080 8 GiB, NVIDIA driver 621.01, Vulkan 1.4.357:

```
device: NVIDIA GeForce RTX 2080 (Vulkan 1.4.357)
features: shaderFloat16=1 storage16=1 storage8=1 bda=1 coopmat=1
cooperative matrix shapes (17):
  16x16x16 A=f16 B=f16 C=f16 R=f16 sat=0
  16x8x16 A=f16 B=f16 C=f16 R=f16 sat=0
  ...
  16x16x16 A=f16 B=f16 C=f32 R=f32 sat=0      <- the shape the kernel uses
  ...
  16x16x32 A=s8 B=s8 C=s32 R=s32 sat=0        <- the INT8 path, later
  16x16x16 A=bf16 B=bf16 C=f32 R=f32 sat=0    <- driver-emulated on Turing; not used
check 64x64x64: max|C - CPU| = 1.01342e-05
gemm 64x64x64:    0.102 ms      0.0 TFLOPS
gemm 512x512x512:    0.038 ms      7.0 TFLOPS
gemm 3072x3072x3072:    6.349 ms      9.1 TFLOPS
gemm 3072x12288x3072:   24.712 ms      9.4 TFLOPS
gemm 3072x3072x12288:   49.082 ms      4.7 TFLOPS
```

The small shapes mostly measure dispatch overhead (one dispatch is ~0.1 ms here); the large ones are what matters.

## Status and limitations

- **Correctness**: the 64³ case matches a double-precision CPU reference to `1e-5` (fp16 rounding of the inputs and the
  f32 accumulator), which is the expected error for this data. The same kernel runs the engine's shapes.
- **Performance**: ~8-9 TFLOPS at 3072³, roughly 10-15 % of the card's fp16 tensor peak. The kernel is memory-bound:
  the cooperative-matrix fragments load straight from global memory, so every subgroup issues strided per-lane loads and
  each A/B element is fetched by several workgroups. Making the tiles larger (64x128) did not help, which confirmed the
  diagnosis.
- **Next step**: stage A and B tiles in shared memory with vectorised coalesced copies and double buffering, then load
  the fragments from shared memory. The 48 KiB shared-memory limit and the 64x64x32 tile staging are worked out in the
  design document; the expected result is 20-30 TFLOPS, which would put a Vulkan DiT step within 2x of the CUDA path.
- No INT8/INT4 path yet, no bias/epilogue fusion, no multi-queue overlap, and one queue only.

## Files

| File | Purpose |
|---|---|
| `gemm_f16.comp` | the cooperative-matrix GEMM (GLSL 460, compiled by `glslc`) |
| `main.cpp` | device setup, buffers, pipeline, CPU check, timing |
| `probe.py` | optional: lists the device's cooperative-matrix shapes and memory heaps using only `vulkan-1.dll` (no SDK) |
| `CMakeLists.txt` | standalone build |
