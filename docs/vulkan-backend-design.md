# Vulkan backend: design proposal

Companion to [`vulkan-backend-research.md`](vulkan-backend-research.md) (device audit, prior art, performance model).
Status: proposal; nothing in this document is implemented yet.

## Scope and goal

Add a **Vulkan compute backend** as the fallback for hardware the engine does not otherwise support, able to run the
MossInfer pipeline there without regressing the CUDA path. CUDA stays the path on every GPU it supports.

Requirements:

- Same model files (official bf16 bundle, ComfyUI INT8/INT4 ConvRot files) and the same public API/CLI/server surface.
- CPU side identical: tokenizer, samplers, PNG, planner, reports.
- The backend for a device follows one fixed rule, never a measurement: CUDA where the engine supports the device, Vulkan
  otherwise. A caller can name a backend explicitly, and the probe and the report always say which one ran.
- Correctness first: every kernel validated against the CUDA implementation on the same weights; end-to-end images
  compared statistically (Vulkan numerics differ in accumulation order, like sm_75 does today).

Non-goals for the first iteration: macOS/MoltenVK, mobile, multi-GPU, Vulkan ray tracing/video, FP8 (not on Turing),
NVFP4 (Blackwell only), and training.

## Where the CUDA coupling is

The engine is not CUDA-abstracted today:

- `Tensor` carries `Device::kCUDA` and a raw `void*`; every kernel takes `cudaStream_t` and uses `cudaMalloc`-style
  arenas (`runtime/DeviceArena`).
- `Planner`/`pipeline` call `cudaMemGetInfo`-based free-memory helpers, `cudaEvent`s and cuBLASLt handles, and the
  model code calls kernel functions directly (`kernels::Gemm::linear`, `moss_dit::MossDit`, ...).
- About 70 CUDA translation units implement the kernels (GEMM, attention, norm, RoPE, GDN, VAE, quantized formats).

A Vulkan backend therefore needs two layers: a **device runtime** (buffers, queues, commands, events, memory budget) and
a **kernel set** behind the same call shapes the model code uses.

## Options

| Option | What it means | Cost | Risk |
|---|---|---|---|
| **A. Backend abstraction inside the engine** | Introduce `mi::runtime::Backend` (device/buffer/stream/event) and `mi::kernels::Backend` (GEMM, attention, norm, ...) interfaces; CUDA and Vulkan implement them; model/pipeline code talks to the interfaces. | High upfront refactor (about 70 kernel files and their callers), but one pipeline for both. | Refactor stalls the CUDA path; hard to land incrementally. |
| **B. Separate Vulkan runner first** | A standalone `moss-vulkan` executable (or tool) that loads the same bundle and runs the pipeline with Vulkan kernels; the engine stays untouched. Merge later through A. | Medium; no risk to CUDA; proves kernels and performance with real weights. | Duplicated pipeline logic until merged; second code path to maintain. |
| **C. CUDA shim over Vulkan** | Implement the CUDA runtime API subset (`cudaMalloc`, `cudaStream_t`, `cuBLASLt` calls) on top of Vulkan so the engine compiles unchanged. | Tempting but the semantics (streams/events, cuBLASLt descriptors, error model) do not map cleanly; CUTLASS kernels are SASS, not SPIR-V, so "compile the same kernels" is impossible. | Dead end for the kernel layer. |

**Recommendation: B first, then A.** Build a standalone Vulkan runner that exercises the real weights and produces
images; once the kernels and the memory model are proven, introduce the runtime/kernel interfaces (A) and move the
pipeline behind them, with CUDA as the default and the Vulkan runner's kernels reused as the backend implementation.
This keeps `main` shippable at every step and gives measurable checkpoints before the refactor.

## Architecture (target state)

```
              pipeline / planner / models (unchanged, backend-agnostic)
                                   |
                    mi::runtime (Device, Buffer, Stream, Event, Budget)
                     /                                   \
          runtime/cuda                          runtime/vulkan
     (DeviceArena, StreamSet, ...)        (VkDevice, queues, allocator, ...)
                     |                                   |
              kernels (CUDA)                      kernels (Vulkan)
     gemm / attention / norm / rope / gdn / vae   same interfaces, SPIR-V shaders
```

- `mi::runtime::Device`: allocation (`Buffer` with size/usage/alignment), upload/download, `Stream` (queue + command
  buffer), `Event` (timeline semaphore value), `free_budget_bytes()`, `properties()` (name, vendor, subgroup, coopmat
  shapes).
- `mi::kernels::*`: free functions in the existing headers, with the CUDA implementation in `src/kernels/cuda/...` and the
  Vulkan one in `src/kernels/vulkan/...`; both built into their own targets and linked according to the selected backend.
- The planner's memory model is backend-neutral: it consumes `free_budget_bytes()` and the transfer-rate profile it
  already measures.

## Module layout and build

```
cmake/modules/vulkan.cmake        option MOSSINFER_VULKAN, find Vulkan (loader + headers), shader compilation
src/runtime/vulkan/               device, queues, allocator, staging ring, pipeline cache, descriptor/BDA helpers
src/kernels/vulkan/               GEMM, attention, norm, rope, gdn, vae, quantized formats (C++ dispatch)
shaders/vulkan/                   *.comp GLSL sources; compiled to SPIR-V at build time by glslc
tests/vulkan/                     kernel parity tests, end-to-end image tests
tools/vulkan_check.cpp            bring-up tool (like tools/sm75_check.cu): device report, GEMM/attention/VAE checks
```

- **Shaders**: GLSL 4.60 + `GL_KHR_cooperative_matrix` + `GL_EXT_shader_16bit_storage`/`8bit_storage` +
  `GL_EXT_shader_explicit_arithmetic_types_*` + `GL_KHR_shader_subgroup_*`, compiled by `glslc` to SPIR-V at build time
  and embedded as `const uint32_t[]` (a generated header), so the binary has no external files. A pipeline cache file is
  written next to the model cache and reused.
- **Build**: `-DMOSSINFER_VULKAN=ON` adds the runtime and kernels, and the CLI/server gain `--backend vulkan` (or
  `MOSSINFER_BACKEND=vulkan`). Without the option nothing changes, and the package stays as today. A `vulkan` CPack suffix
  like the sm75 one can follow once the backend ships.
- **Dependencies**: Vulkan loader and headers only (`vulkan-1.dll`/`libvulkan.so.1` at run time); no Vulkan SDK at run
  time. Shader compilation needs `glslc` at build time (vendored or found).

## Runtime design

### Device and queues

- One `VkDevice`, one compute queue (family 0 with `COMPUTE`), one transfer queue if the family differs.
- Single command buffer stream for compute; a second for uploads, synchronized with timeline semaphores. The engine's
  CUDA path already separates compute and copy streams (`runtime::StreamSet`), so the abstraction matches.
- `Stream` = a command buffer in recording; `submit(stream)` batches recorded commands; `Event` = timeline semaphore
  value; `wait(event)` maps to `vkWaitSemaphores`.

### Memory

- A size-class suballocator over a few large `VkDeviceMemory` blocks (256 B alignment, dedicated allocations above
  64 MiB), with `BDA` (buffer device address) so kernels take plain 64-bit device pointers in push constants.
- Heaps: device-local (weights, activations), host-visible coherent (staging ring, 8–16 MiB, mapped), and a
  host-cached path for streamed blocks (readback-free upload through a mapped ring).
- Budget: `VK_EXT_memory_budget` (`heap_budget`, `heap_usage`) plus the allocator's own accounting; report
  `free_budget_bytes = heap_budget - heap_usage - reserve` to the planner, the same contract `cudaMemGetInfo` provides.
- Weight streaming: the engine's DiT block streaming (5.16 GB INT8 at 256², 9.6 GB bf16) becomes: host-side ring buffer →
  transfer queue → device-local block buffers, with timeline semaphores per block and a per-step upload budget. The
  planner's existing estimate of copies hidden behind compute applies unchanged.

### Dispatch and command recording

- Each kernel is a `VkPipeline` (specialization constants for tile sizes/strides, push constants for pointers/offsets).
- The Vulkan runner records the whole block's kernel sequence into one command buffer (norm → modulation → QKV GEMM →
  attention → out GEMM → FFN up/gate GEMM → SiTU → down GEMM → residual), so the CPU cost is one submit per block, not
  20–40. `vkCmdPipelineBarrier2` with compute-to-compute barriers is inserted by the runtime (or explicit per kernel).
- Secondary command buffers (or one pre-recorded buffer per shape) make the per-step recording cost negligible; a
  block resident in device memory can replay a pre-recorded buffer, comparable to the CUDA `fused_graphs` path.

## Kernel plan

Legend: CM = cooperative matrix, SIMT = plain compute shader.

| Stage | Operation | Shapes (DiT 3072, its FFN 8192; text encoder 4096, its FFN 12288; VAE 96–384) | Vulkan kernel |
|---|---|---|---|
| DiT block | modulation/norm | [tokens, 3072] | SIMT, vectorized f16 |
| DiT block | QKV / out / FFN up-gate / FFN down / final (N×K, M = tokens) | 9216×3072, 3072×3072, 16384×3072, 3072×8192 | CM f16 16×16×16, f32 acc; epilogues in shader (bias, gated residual) |
| DiT block | joint attention | tokens 4096 (1024²), heads 24, head_dim 128, non-causal varlen | tiled FA: CM QK^T, f32 softmax, SIMT PV or CM PV |
| DiT block | RoPE, SiTU-GLU, residual | elementwise | SIMT |
| Text encoder | RMSNorm, linear, RoPE, causal attention, GDN recurrence | hidden 4096, 32 layers | GEMM CM; attention CM/SIMT; GDN SIMT scan |
| VAE decode | conv2d (NHWC/KRSC), groupnorm, mid attention, upsample | 96/192/384, 64×64→1024² | SIMT f32 conv (like the sm_75 kernels), attention SIMT |
| INT8 path | ConvRot GEMM + dequant | same GEMM shapes | CM s8 16×16×32, s32 acc, dequant epilogue |
| INT4 path | unpack to INT8 then GEMM | same | SIMT unpack + INT8 CM |

### Devices

Tile sizes come from the device's limits, not from this document: Vulkan only guarantees 16 KiB of shared memory, and
subgroup sizes differ (the tiles below need 32). A device without a usable cooperative-matrix shape runs a plain compute
GEMM, slower but correct; that path is what makes the backend a fallback rather than a second fast path.

### GEMM tiling (48 KiB shared memory, f16, f32 accumulate)

M is free: whole tiles always run, A and C are allocated with M rounded up to 64 rows, and row m of C depends only on row m
of A. N is a multiple of 64 in every engine GEMM.

- Workgroup 64×64 output tile, 4 warps (2×2), each warp computes 32×32 via four 16×16×16 cooperative-matrix ops.
- K step 32: A tile 64×32 f16 = 4 KiB, B tile 32×64 f16 = 4 KiB; 3-stage pipeline = 24 KiB, leaves room for epilogue
  staging and the bias/gate vectors.
- Registers: 32×32 f32 accumulator per warp = 1024 floats = 32 per thread, plus A/B fragments; well within a 255-register
  budget at 128 threads/workgroup.
- Split-K for skinny GEMMs (e.g., final layer, modulation) or when M is small (prompt tokens); the engine's CUDA GEMM
  already chunks rows, so the Vulkan dispatcher can reuse that policy.
- Epilogues are fused in the shader: bias add, `ffn_down` residual add, gated residual (multiply by gate), SiTU-GLU
  (tanh/exp on f32), and the INT8 dequant (`acc * x_scale[row] * w_scale[col] + bias`).

### Attention kernel

- One workgroup per (block of queries, head, sequence chunk). Staged f16 tiles must fit 48 KiB, Q + K + V + P:
  head_dim 128 (DiT) with 64 queries and 32 keys is 16 + 8 + 8 + 4 = 36 KiB; head_dim 256 (Qwen) with 32 queries and
  16 keys is 16 + 8 + 8 + 1 = 33 KiB.
- QK^T on cooperative matrix (f16 inputs, f32 accumulate), softmax in SIMT f32 with the same online algorithm as the
  CUDA sm_75 kernel (`src/kernels/sm75/attention_sm75.cu`), PV on cooperative matrix or SIMT.
- Varlen/GQA: per-sequence offsets and head mapping in push constants; the DiT joint attention is non-causal over
  [text + image] tokens, Qwen is causal.
- 4096-token attention at 1024² walks those tiles over 64 query blocks and 24 heads.

### Quantization formats

- INT8 ConvRot weights are stored as int8 with per-row (activation) and per-column (weight) f32 scales; the engine's
  `Int8Operands` contract carries them. The Vulkan INT8 GEMM uses CM s8 16×16×32 (exact int32 accumulation) and the same
  dequant epilogue, so results should be bit-identical to CUTLASS.
- INT4 ConvRot: reuse the CUDA engine's INT4→INT8 unpack (a SIMT kernel) and the INT8 GEMM, as on Hopper/Blackwell and
  in the sm_75 port.
- ConvRot (Hadamard rotation) is already folded into the stored weights, so the GEMM kernel is rotation-agnostic.

## Integration with the engine

- **Probe**: `src/backend/probe.*` gains a Vulkan path when built with `MOSSINFER_VULKAN`: enumerate physical devices,
  report name/vendor/heap/coopmat shapes and, per device, the backend that the fixed rule above gives it.
- **Planner**: the cost model's rates (`bf16_tflops`, PCIe rate) are learned per backend and per device; the tier rules
  are backend-neutral (reference/high/turbo), except that on Vulkan the 8-bit attention modes may not exist.
- **Tier/attention**: Vulkan on Turing-class hardware resolves an unset attention to `exact` (same rule as sm_75); on
  devices with f16 cooperative matrix a `q8_fp16` port can come later.
- **CLI/server**: `--backend cuda|vulkan` overrides the rule; the report gains `quality.backend = "vulkan"` and the device string;
  `moss-server` passes it through `load`'s new optional `backend` parameter (protocol addition).
- **Packaging**: a `-vulkan` package flavour (loader + shaders embedded); the CUDA package stays the default.

## Testing and validation

1. **Kernel parity** (`tests/vulkan/`): for each kernel, run the same inputs through CUDA and Vulkan, compare exact
   (INT8), cosine ≥ 0.99999 / max|diff| (f16), and report. Reuses the vectors `tests/data` already stores for the CUDA
   kernels.
2. **Block parity**: one DiT block (same weights, same input) CUDA vs Vulkan, cosine on the block output.
3. **End-to-end**: generate the same seed/prompt/size on CUDA and Vulkan; compare PSNR/SSIM and the report; expected
   differences are accumulation-order only.
4. **Performance gates**: step time at 256²/512²/1024², per-tier, on the 2080 (CUDA baseline in `docs/sm75.md`) and on
   at least one AMD device (RX 6000/7000) and one Intel Arc if available.
5. **Stability**: validation layers (`VK_LAYER_KHRONOS_validation`) in debug builds, pipeline cache round-trip, memory
   budget accounting, 100 consecutive generations without leaks.

## Milestones

| Milestone | Deliverable | Acceptance | Estimate |
|---|---|---|---|
| **V0** | Probe tool + GEMM microbenchmark (`tools/vulkan_gemm`); done for the GEMM | GEMM correct and measured at the engine's shapes against cuBLASLt | done |
| **V1** | Vulkan runtime (device, allocator, commands, pipeline cache) + f16 GEMM, shared-memory tiles and a plain compute path | GEMM cosine ≥ 0.9999 at engine shapes, on a device without cooperative matrices too | 1–2 weeks |
| **V2** | Attention + norm/RoPE/SiTU kernels; DiT block runner on real weights | Block output matches CUDA block (cosine ≥ 0.999) | 1–2 weeks |
| **V3** | Text encoder (GEMM, RMSNorm, RoPE, causal attention, GDN) and VAE decoder | Encode/decode parity; first full Vulkan image | 2–3 weeks |
| **V4** | Pipeline integration (runtime abstraction, planner, streaming, CLI/server) | `moss-infer --backend vulkan` and `moss-server` produce images on the 2080 | 2–4 weeks |
| **V5** | Quantized paths (INT8/INT4), packaging, CI, docs | `high`/`turbo` tiers on Vulkan; `-vulkan` package; tests in CI | 1–2 weeks |

The estimates assume one experienced engineer and the existing CUDA code as the specification; the GDN recurrence and
the VAE convolutions are the least predictable parts.

## Risks and mitigations

- **Scope explosion.** B-first keeps the engine shippable; V4 (refactor) is gated on V2/V3 parity.
- **Vendor differences.** Cooperative matrix shapes and shared-memory limits vary; the kernel set is selected from a
  capability table (NVIDIA first, AMD/Intel fallbacks second).
- **Performance.** As a fallback it never replaces CUDA, so its speed gate is whether a generation is usable on the
  fallback device. That can only be measured on one.
- **Maintenance.** Two kernel sets are expensive; the shared parity tests and the runtime abstraction are the
  mitigation. If the backend cannot be maintained, V1–V3 remain a standalone tool.

## Decisions requested

1. Decided: Vulkan is the fallback for hardware the engine does not otherwise support.
2. Choose the first fallback devices for validation.
3. Decide whether the Vulkan path ships as a separate package flavour or inside the default package (loader dependency).
