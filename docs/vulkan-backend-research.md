# Vulkan backend: research notes

Status: research only; no Vulkan code is in the engine yet. This document and
[`vulkan-backend-design.md`](vulkan-backend-design.md) are the basis for a decision on whether, and how, MossInfer gets a
Vulkan compute backend.

## Why a Vulkan backend

MossInfer is CUDA-only: the pipeline calls cuBLASLt, CUTLASS kernels and CUDA kernels directly, and the probe refuses any
GPU whose compute capability is below 8.0 (7.5 with `MOSSINFER_SM75`). That confines the engine to NVIDIA on Windows and
Linux, with the CUDA toolkit's driver and memory-management model.

A Vulkan compute backend would give:

- **Other GPUs.** AMD (RDNA2/3/4), Intel (Arc/Xe), and NVIDIA on systems where the CUDA toolkit is not desirable. The
  engine's model code is portable C++/CUDA; only the kernels and the device runtime are CUDA-specific.
- **One packaging story** for Windows and Linux without the CUDA runtime dependency, and a path to Steam Deck (RADV)
  and other low-power devices.
- **An escape hatch from CUDA's memory model.** Vulkan exposes `VK_EXT_memory_budget`, explicit heaps and suballocation,
  which fits the engine's planner/streaming design; CUDA's `cudaMemGetInfo` is coarser.

The cost is a second implementation of every tensor-core kernel, plus a device/memory/command abstraction the engine
does not have today (see the design document).

## What this machine exposes (primary evidence)

Probed on the target machine: Windows 11, RTX 2080 8 GiB, NVIDIA driver 621.01 (Vulkan driver 621.1.0.0). The Vulkan
loader in `C:\Windows\System32\vulkan-1.dll` reports **instance 1.4.357**; the NVIDIA ICD reports **device API 1.4.357**
and one physical device.

### Limits that shape kernel design

| Limit | Value | Consequence |
|---|---|---|
| `maxComputeSharedMemorySize` | 48 KiB | 64 KiB is a CUDA opt-in only; Vulkan tiles must fit 48 KiB (Turing's hardware limit is 64 KiB but Vulkan caps at 48 KiB here). |
| `maxComputeWorkGroupInvocations` | 1024 | Up to 32 warps per workgroup. |
| `subgroupSize` | 32 | Cooperative-matrix `gl_ScopeSubgroup` = one warp. |
| `maxStorageBufferRange` | 2^32-1 bytes | 4 GiB per buffer; weights are 75 MB at most per tensor, so a small number of big buffers is fine. |
| Device-local heap | 7989 MiB | The engine's planner sees the same ~8 GiB. A second device-local heap of 214 MiB exists; host-visible heap is 16 GiB. |

### Extensions a compute backend needs (all present)

`VK_KHR_cooperative_matrix` (revision 2), `VK_KHR_16bit_storage`, `VK_KHR_8bit_storage`, `VK_KHR_shader_float16_int8`,
`VK_KHR_shader_integer_dot_product`, `VK_KHR_buffer_device_address`, `VK_EXT_memory_budget`, `VK_KHR_timeline_semaphore`,
`VK_KHR_synchronization2`, `VK_EXT_descriptor_indexing`, `VK_KHR_maintenance4`, `VK_KHR_maintenance5`,
`VK_EXT_subgroup_size_control`, `VK_KHR_pipeline_library`, `VK_EXT_shader_module_identifier`,
`VK_KHR_shader_float_controls` (and `2`), `VK_KHR_vulkan_memory_model`, `VK_KHR_shader_clock`,
`VK_KHR_performance_query`, `VK_NV_cooperative_matrix`, `VK_NV_cooperative_matrix2`, `VK_NV_cooperative_vector`.

Shader features: `storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess`, `storageBuffer8BitAccess`,
`uniformAndStorageBuffer8BitAccess`, `shaderFloat16`, `shaderInt8`, `shaderBFloat16Type`,
`shaderBFloat16CooperativeMatrix`, `shaderSubgroupClock`, `shaderDeviceClock`, `shaderFmaFloat16`.

### Cooperative-matrix shapes (queried with `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`)

17 combinations, all `gl_ScopeSubgroup`, non-saturating:

| Shape (M×N×K) | A | B | C | Result |
|---|---|---|---|---|
| 16×16×16, 16×8×16, 16×8×8 | f16 | f16 | f32 | f32 |
| 16×16×16, 16×8×16, 16×8×8 | f16 | f16 | f16 | f16 |
| 16×16×32, 16×8×32 | s8 | s8 | s32 | s32 |
| 16×16×32, 16×8×32 | s8 | u8 / u8×s8 / u8×u8 | s32 / u32 | s32 / u32 |
| 16×16×16 | bf16 | bf16 | f32 | f32 |

Notes:

- **f16 with f32 accumulate is available at 16×16×16 and 16×8×16**, which is the same HMMA shape the CUDA sm_75 port
  uses (`m16n8k16`; the 16×8×16 cooperative matrix maps to it). A Vulkan GEMM can therefore reach the same tensor-core
  instruction on the same hardware.
- **INT8 at 16×16×32 / 16×8×32 with s32 accumulate** covers the engine's INT8 ConvRot GEMMs (CUTLASS `m8n8k16` on
  sm_75 is a different instruction, but the arithmetic is an exact int32 dot product either way).
- **bf16 cooperative matrix is reported by this Turing driver**, but Turing has no bf16 tensor core: the driver must be
  emulating it (likely f16 down-conversion or split-f16 math). The engine already converts bf16 weights to fp16 for
  sm_75, so the Vulkan backend should keep fp16 as its compute type and treat bf16 as a storage conversion, not rely on
  the bf16 cooperative-matrix path.
- `VK_NV_cooperative_matrix` is still exposed (the pre-1.4 NV API); `vkGetPhysicalDeviceCooperativeMatrixPropertiesNV`
  returned a table that our probe could not decode reliably (the struct's enums are narrower than the KHR ones). The
  KHR query is the authoritative one on this driver; the backend should prefer it and fall back to the NV query only on
  older drivers.

### Memory heaps

| Heap | Size | Flags | Use |
|---|---|---|---|
| 0 | 7989 MiB | device-local | Model weights, activations |
| 1 | 16347 MiB | host-visible (0x6 = visible + coherent; 0xe adds cached) | Staging ring, streamed blocks, host-cached weights |
| 2 | 214 MiB | device-local | Small warm allocations |

That matches the CUDA picture the planner already models (8 GiB device + ~16 GiB host), so the existing planner cost
model (bytes per block, PCIe rate, hidden-copy fraction) transfers directly; the Vulkan backend only needs to report the
free device budget from `VK_EXT_memory_budget` instead of `cudaMemGetInfo`.

## Prior art and what to take from it

### ggml-vulkan (llama.cpp, whisper.cpp, stable-diffusion.cpp)

The most complete Vulkan inference backend in the open-source ecosystem. Design points worth copying:

- **One backend file plus generated shaders.** `ggml-vulkan.cpp` (~20k lines) houses device setup, memory arenas,
  pipelines and per-op dispatch; the compute shaders are GLSL templates compiled offline to SPIR-V and embedded (the
  build generates a C++ header with the SPIR-V arrays).
- **Specialization constants and push constants.** Workgroup sizes, tile sizes and strides are specialization
  constants, so one shader covers many shapes; per-dispatch arguments (pointers via BDA, offsets, strides) go in push
  constants, avoiding descriptor-set churn.
- **Cooperative matrix paths.** `coopmat` (KHR 16×16×16) and `coopmat2` (VK_NV_cooperative_matrix2) variants for matrix
  multiply and flash attention; a scalar fallback for devices without cooperative matrix.
- **Pipeline cache.** The driver's pipeline cache is serialized to disk and reused; shader compilation is a large part
  of a cold start otherwise.
- **Memory arenas.** Buffers are suballocated from a few large `VkDeviceMemory` blocks with size classes and a
  buffer-device-address allocator; the engine's streaming design can use the same idea for its block buffers.
- **Flash attention shaders.** Tiled FA with online softmax, GQA support (`coopmat2` shader in llama.cpp #12559) and
  separate paths for quantized KV; attention is the piece most likely to be slower than CUDA.

### stable-diffusion.cpp

Runs SD1.5/SDXL/Flux families through ggml, with the same Vulkan backend. It demonstrates diffusion inference on Vulkan
end-to-end, including VAE decode and img2img, on consumer GPUs. Its known weaknesses map to ours: performance depends
strongly on the GPU vendor's Vulkan driver, AMD RDNA3/4 is often close to CUDA, small Intel Arc parts are slower, and
attention remains the bottleneck. It is the closest prior art to what MossInfer needs.

### Others

- **ncnn Vulkan**: mobile/desktop CNN inference; mature Vulkan kernels (convolution, pooling), good evidence that
  Vulkan convolution is practical (relevant for the VAE decoder).
- **Kompute / vuh**: C++ Vulkan compute wrappers; convenient for prototyping but no tensor-core help.
- **IREE HAL / ONNX Runtime**: IREE has a Vulkan HAL; ONNX Runtime's GPU EP is DirectML on Windows, not Vulkan. Neither
  offers a drop-in path for a custom engine, but their Vulkan memory/scheduling designs are useful references.
- **wgpu / WebGPU**: a higher-level portability layer with a compute language (WGSL) and no cooperative matrix in the
  standard; shader-f16 and subgroup support are optional. It would be easier to port but cannot reach tensor-core
  throughput on Turing; not a candidate for the "fast" backend, only for a "works everywhere, slowly" fallback.

## Performance model

Measured CUDA numbers on this machine (RTX 2080 8 GiB, SFT bundle), from `docs/sm75.md`:

| Tier | 256×256 step | Compute per step (approx.) |
|---|---|---|
| reference (bf16→fp16 GEMMs) | 0.80 s | 42 blocks × ~0.015 s + text encoder ~3 s per prompt |
| high (INT8 DiT) | 0.19 s | 42 blocks × ~0.004 s |
| turbo (INT4 via INT8) | 0.22 s | unpack overhead + INT8 |

Turing's fp16 tensor-core peak on the 2080 is roughly 50–60 TFLOPS (NVIDIA's figures for the card are in that range,
depending on fp16 vs fp32 accumulate); cuBLASLt reaches a large fraction of it on the GEMM shapes of this model
(3072×12288×3072 etc.). A Vulkan cooperative-matrix GEMM with 16×16×16 f16×f16→f32 tiles can use
the same instruction; the realistic outcome is **50–80 % of cuBLAS** for large GEMMs, dropping for small ones (the DiT's
per-block GEMMs are 3072-wide, large enough; the modulation/norm kernels are bandwidth-bound and vendor-independent).

What that means per step at 256² (INT8 high, 0.19 s): a Vulkan backend at 60 % of the CUDA GEMM throughput would be
~0.30 s/step; the planner's cost model would then choose the Vulkan tier/streaming plan accordingly. The VAE decode is
fp32/conv-heavy and would be the second-largest cost; Vulkan convolution on Turing has no tensor-core help for fp32, so
it stays bandwidth-bound (same as the CUDA fp32 VAE path).

The critical unknowns to measure in a prototype (M0):

1. Cooperative-matrix GEMM throughput at the engine's shapes (f16, f32 accumulate, 48 KiB tiles).
2. Dispatch overhead per kernel (engine runs ~20–40 dispatches per block; command-buffer batching must keep the CPU
   cost per step below the GPU cost).
3. Attention throughput (fp16 QK^T + softmax + PV, 4096-token joint attention at 1024²).
4. PCIe upload rate for streamed DiT blocks (the CUDA path measures 2.9–4.6 GB/s from file/page cache).
5. fp16 vs bf16 numerics: the engine's reference tier is bf16 storage with fp32 accumulate; converting to fp16 for
   Vulkan changes rounding in the same way the sm_75 CUDA path does (documented in `docs/sm75.md`).

## Numerics and formats

- **Storage:** the engine's official bundle is bf16; the sm_75 port converts it to fp16 in place, and the Vulkan backend
  would do the same (bf16→fp16 is exact inside fp16's range, and the reverse conversion recovers the bf16 bits).
  Community INT8/INT4 ConvRot files are already quantized; no conversion is needed for weights.
- **Compute:** f16 inputs with f32 accumulate via cooperative matrix (matching cuBLASLt's `CUBLAS_COMPUTE_32F` use in the
  engine's fp16 path); INT8 weights use s32 accumulate (exact, same result as CUTLASS on CUDA); the VAE decoder stays
  fp32 (SIMT), since Vulkan exposes no fp32 tensor core.
- **No FP8 on Turing** (`VK_KHR_shader_fp8` is not exposed), so the engine's `q8_fp8` attention path cannot be ported
  to this GPU; `q8_fp16` would be the 8-bit attention variant on Vulkan, and `exact` remains the default on sm_75-class
  hardware.
- **Denormals:** f16 cooperative matrix on NVIDIA flushes subnormal inputs/outputs? The CUDA sm_75 port relies on fp16
  exactly matching bf16 inside the normal range; the prototype must compare bit-for-bit or cosine against the CUDA path
  and document the deviation.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Cooperative matrix on AMD/Intel differs (16×16×16 only, no int8) | Kernels must specialize per vendor; Vulkan backend becomes NVIDIA-first | Query properties at startup and select kernels; scalar/DP4A fallbacks |
| Shader compilation time and driver bugs | Cold start and hangs | Precompile SPIR-V at build time, ship pipeline cache, validation layers in CI |
| 48 KiB shared memory | Smaller tiles than CUDA, lower GEMM efficiency | 3-stage 64×64×32 f16 tiles; split-K for skinny shapes |
| Dispatch overhead (CPU-bound at small sizes) | Step time worse than CUDA | Batch dispatches into one command buffer; reuse command buffers; push constants + BDA |
| `VK_EXT_memory_budget` imprecision | Planner over/under-estimates free VRAM | Use budget + own allocation accounting; keep the CUDA planner model |
| bf16 coopmat is emulated on Turing | Wrong performance/numerics assumptions | Compute in f16, store bf16 as f16, never use bf16 coopmat |
| Windows WDDM TDR (2 s) on long kernels | Device reset on big attention/GEMM | Keep dispatches short; split work (the engine already tiles) |
| Two backends to test | Maintenance cost | Shared test vectors and CI matrix; backend-agnostic golden images |

## Open questions for the design phase

1. Is the target "NVIDIA parity plus AMD/Intel support", or "works on anything"? That decides how many fallbacks the
   kernel layer needs.
2. Should the Vulkan path reuse the engine's existing planner/streaming code through a device abstraction, or start as a
   separate `moss-vulkan` runner and merge later? (The design document recommends the latter first.)
3. Which attention variant is acceptable on Vulkan: `exact` only (like sm_75), or a `q8_fp16` port?
4. Is a WebGPU/wgpu fallback worth a third backend, or is Vulkan enough for the non-NVIDIA target?
