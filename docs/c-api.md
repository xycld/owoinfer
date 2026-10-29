# MossInfer C API

`include/mossinfer/mossinfer.h` is the stable interface to the engine. It is plain C99, in the style of `llama.h`. The library is `libmossinfer.so` or `mossinfer.dll`, and CMake consumers use `find_package(MossInfer)` → `MossInfer::mossinfer` (see `cmake/modules/packaging.cmake`). `moss-server` (`docs/server-protocol.md`) and `examples/c/minimal.c` use only this header.

## Quick start

```c
#include "mossinfer/mossinfer.h"

mi_engine_params ep;
mi_engine_params_init(&ep, sizeof ep);          /* defaults; struct_size is set for you */
ep.model_path = "/models/MOSS-Image/base";      /* Hugging Face bundle directory, UTF-8 */
ep.dit_file = "/models/moss_image_base_int8_convrot.safetensors";  /* optional: the everyday INT8 DiT */
ep.tier = MI_TIER_HIGH;                         /* required: REFERENCE, HIGH or TURBO */
mi_engine* engine = NULL;
if (mi_engine_create(&ep, &engine) != MI_OK) { fprintf(stderr, "%s\n", mi_last_error_message()); return 1; }

mi_gen_params gp;
mi_gen_params_init(&gp, sizeof gp);             /* the app's parameters.json defaults */
gp.prompt = "a red apple on a wooden table";
gp.width = gp.height = 1024;
mi_result* result = NULL;
mi_status s = mi_generate(engine, &gp, &result);   /* MI_OK, or MI_CANCELLED after mi_cancel() */
if (s == MI_OK) mi_save_png(result, "apple.png");
mi_result_free(result);                          /* NULL-safe; set on MI_OK and MI_CANCELLED */
mi_engine_free(engine);
```

Build and run the example:

```
cmake --build <build> --target mossinfer_example_c
<build>/mossinfer_example_c /models/MOSS-Image/base apple.png "a red apple on a wooden table" 512 20
```

## Conventions

- **Strings** are NUL-terminated UTF-8, paths included, on Windows too. The library copies every string it keeps. A string ends at its first NUL, so a prompt cannot contain one. `moss-server` rejects a JSON string with an escaped `\u0000` as `invalid_params` instead of cutting it short.
- **Versioned structs** start with `size_t struct_size`. Always fill them with `*_init(&s, sizeof s)`, then change the fields you need.
  - Structs only grow at the end. The library accepts every size a struct has had (see [Struct sizes](#struct-sizes)), and rejects any other `struct_size` with `MI_ERR_ABI_MISMATCH`.
  - Output structs (`mi_timings`, `mi_memory_stats`, `mi_dit_stream_stats`, `mi_probe_info`, `mi_device_info`, `mi_tier_info`, `mi_prompt_tokens`, `mi_lora_report`, `mi_image_info`, `mi_control_info`, `mi_accel_info`) need `struct_size` set by the caller before the call.
  - Padding is explicit: the gaps are `int32_t _reserved*` fields (see [Frozen layout](#frozen-layout-abi-1)). `*_init()` zeroes them. `mi_engine_create()`, `mi_generate()` and `mi_gen_params_validate()` reject a nonzero one with `MI_ERR_INVALID_ARGUMENT`. The library zeroes them in output structs.
- **Enums** are 32-bit: each one has a `*_MAX_ENUM = 0x7fffffff` member. Unknown values are rejected with `MI_ERR_INVALID_ARGUMENT`.
- **Errors.** Every fallible call returns `mi_status`. `mi_last_error_message()` returns the reason for the last failure on the calling thread. It is thread-local, never NULL, not cleared by successful calls, and valid until the next failing call on that thread.
  - Recording a message never throws. If copying it fails for lack of memory, the message is a fixed "out of memory while recording the error message".
  - `mi_engine_free()` and `mi_result_free()` never fail and never throw, and they leave the message alone. A CUDA error while releasing device memory is ignored.
  - Validation messages use the official app's wording, e.g. "Height and width must be positive multiples of 16."
  - Accessors without a status (`mi_result_width()` and the like) return 0 / NULL for a NULL handle, and set the message.
  - Accessors that fill an output struct check its `struct_size` before the handle: a wrong size is `MI_ERR_ABI_MISMATCH` even with a NULL result.
- **No C++ crosses the boundary.** Every entry point catches every exception and returns a status.
- **Versions.** `MI_VERSION_*` and `MI_ABI_VERSION` in the header are the only source of the version (`cmake/version.cmake` reads them). At runtime, `mi_version()` and `mi_abi_version()` report the library that was actually loaded. A consumer must refuse a library whose ABI differs from the header's.

| Status | Meaning |
|---|---|
| `MI_OK` (0) | success |
| `MI_CANCELLED` (1) | `mi_generate()` or `mi_engine_preload()` stopped by `mi_cancel()`: a result is still returned (no image, but timings and the report). Also `mi_engine_release()` stopped before it freed anything. |
| `MI_ERR_INVALID_ARGUMENT` | NULL handle or pointer, value out of range, invalid request |
| `MI_ERR_ABI_MISMATCH` | unknown `struct_size` |
| `MI_ERR_NOT_SUPPORTED` | not in this build, e.g. a ComfyUI file as `model_path` (`MI_MODEL_FORMAT_COMFYUI`) |
| `MI_ERR_BUSY` | the engine is already running a job: a generation, a preload or a release |
| `MI_ERR_OUT_OF_MEMORY` | the request fits the device or host budgets in no mode (planner), or a host allocation failed |
| `MI_ERR_MODEL` | model files missing, incomplete or malformed, or not a supported MOSS-Image model ([integration.md](integration.md#supported-models)); a LoRA file that cannot be applied (with `lora_strict`, also unknown keys and lossy merges) |
| `MI_ERR_IO` | file-system error: reading weights, writing the PNG |
| `MI_ERR_CUDA` | CUDA error or no usable device. Free the engine afterwards. Also `mi_engine_create()` when no usable cuBLASLt is found: the message lists every candidate tried and why it was rejected (see [cuBLASLt](#cublaslt)). |
| `MI_ERR_INTERNAL` | anything else |

## Threading model

- **One job at a time per engine.** A job is `mi_generate()`, `mi_engine_preload()`, `mi_engine_release()` or a [component call](#component-api). A concurrent one on the same engine returns `MI_ERR_BUSY`. Separate engines may run concurrently, but they share the GPU and each one plans against its own budgets.
- **`mi_cancel(engine)` is thread-safe** and may be called from any thread, including from a callback.
  - A generation stops within one text-encoder layer, DiT block or decode band, and `mi_generate()` returns `MI_CANCELLED`. A preload stops between models and DiT weight groups (`MI_CANCELLED`). A release stops before it frees anything (`MI_CANCELLED`, nothing released), or, once it is freeing, stops the DiT's host copy between 64 MiB reads, keeps a text encoder's copy and still returns `MI_OK`.
  - A cancel only affects the job running at that moment. It is a no-op when the engine is idle, and it never carries over to the next job.
- **Callbacks** (`progress_callback`, `plan_callback`, `preview_callback`) run synchronously on the thread that called `mi_generate()` (or `mi_engine_preload()`), so keep them short.
  - They must not throw or longjmp.
  - They must not call `mi_generate()`, `mi_engine_preload()`, `mi_engine_release()` or `mi_engine_free()` on the same engine.
- **`mi_engine_free()` may race a running job** on another thread (2.1): it cancels the job, waits for its call to return, then frees. From a callback of the engine's running job it stays forbidden, and does nothing. A call that has not started when `mi_engine_free()` begins (`mi_cancel()` included, and a call still reading its arguments before it reaches the engine) must not overlap it: the caller orders them. Using the engine pointer after `mi_engine_free()` returns is the caller's bug.
- **`mi_result` is independent of its engine.** It may outlive the engine, and it may be used from any thread, one at a time.
- **The CUDA device is selected per call.** `mi_generate()` selects `device_index` on the calling thread, so engines may be driven from any thread.
  - `mi_engine_create()`, `mi_generate()` and `mi_engine_free()` restore the calling thread's current CUDA context before they return: none if the thread had none, else the caller's own.
  - They use the driver API (`cuCtxGetCurrent` / `cuCtxSetCurrent`, loaded at run time). Without it, they fall back to restoring the runtime's current device.

## Engine

`mi_engine_create()` does three things:
1. Checks the parameters, the model path (the four bundle folders `text_encoder`, `tokenizer`, `transformer` and `vae`), and that each given component file exists and is a regular file. This happens before CUDA is touched. The model folder's configs, the architecture of the DiT and text-encoder files and the component files' format are checked when they load, at the first `mi_generate()`, which then fails with `MI_ERR_MODEL`. A folder or file of another model fails the same way, with a message naming what was found ([supported models](integration.md#supported-models)).
2. Selects the device.
3. Creates the pipeline.

The weights load on the first `mi_generate()`, or earlier with `mi_engine_preload()` ([Preload and soft unload](#preload-and-soft-unload)). The DiT stays resident across generations when the plan allows it, so the first call pays the load (several seconds, depending on storage and the page cache) and later ones do not.

| `mi_engine_params` field | Default | Meaning |
|---|---|---|
| `model_path` | — (required) | bundle directory. The tokenizer and configs always come from it. A component that `dit_file`, `text_encoder_file` or `vae_file` replaces needs only its config in its folder: the engine identifies each file as its component by its shapes, and the folder's weights for that component are then not read, whether there or not (a `moss.mossidx` still serves). A component no file replaces needs its weights: without them `mi_generate()` fails with `MI_ERR_MODEL`, naming the component ("missing weights for …: add them or pass its component file" when a `moss.mossidx` lists them). |
| `model_format` | `AUTO` | `AUTO`: a directory is an HF bundle, a file is a ComfyUI checkpoint (not supported: `MI_ERR_NOT_SUPPORTED`). `COMFYUI` is reserved and not supported either. |
| `mode` | `BALANCED` | `MIN_MEMORY` / `BALANCED` / `MAX_SPEED` (planner objective) |
| `device_index` | 0 | CUDA ordinal |
| `vram_budget_bytes` | 0 = free memory − reserve | cap on the process's device memory, CUDA context included |
| `vram_reserve_bytes` | 384 MiB | device memory left outside the arena; 0 = none |
| `host_ram_budget_bytes` | 0 = the mode's target | cap on private host memory |
| `cache_dir` | NULL = memory only | embedding-cache disk spill |
| `embedding_cache` | 1 | reuse the text embeddings of repeated prompts |
| `direct_io` | 0 | 1: O_DIRECT for every weight read (cold-cache behaviour), whatever `dit_direct_io` says. 0: the DiT load follows `dit_direct_io` |
| `block_path` | `DEFAULT` (fused) | DiT block implementation: `OP_BY_OP`, `FUSED` or `FUSED_GRAPHS`; all give the same image bits. `FUSED` runs op by op when the startup self-check finds the fused GEMMs not bit-identical on this GPU, and `FUSED_GRAPHS` replays CUDA graphs only when every DiT block is resident. `mi_result_block_path()` reports what ran. |
| `dit_file` | NULL = the bundle's `transformer/` | single-file DiT: INT8 ConvRot (G = 256), INT4 ConvRot (the `TURBO` tier), NVFP4 (the `TURBO` tier on an RTX 50: compute capability 12.0; refused elsewhere with `MI_ERR_MODEL`), mixed INT8/bf16 or bf16 safetensors in the ComfyUI layout (Comfy-Org and community files, `moss-infer quantize` output) |
| `text_encoder_file` | NULL = the bundle's `text_encoder/` | INT8 ConvRot or W4A8 Qwen3.5 text encoder. Its embeddings are cached under its own identity. |
| `vae_file` | NULL = the bundle's `vae/` | single FLUX.2 VAE file, decoded (and, when it holds an encoder, encoded) with its own latent statistics rounded to bf16 at load, as the bundle stores them; the bundle's `vae/` weights are not read. For the Comfy-Org files these are the bundle's values, so the image is the bundle's bit for bit. img2img with a decoder-only file still needs the bundle's `vae/diffusion_pytorch_model.safetensors` for the encoder. |
| `tier` | none: **required** | quality tier, see below. The zero default `MI_TIER_AUTO` is refused by `mi_engine_create()` with `MI_ERR_INVALID_ARGUMENT` |
| `attention` | `DEFAULT` (the tier's choice) | DiT attention kernel: `EXACT` (FlashAttention-2) or an 8-bit mode |
| `lora_strict` | 0 | 1: unknown LoRA keys and lossy merges fail the generation with `MI_ERR_MODEL` instead of being reported |
| `time_embedding_file` | NULL = none | a step-embedder replacement (`time_embedding.safetensors`: the step embedder and modulation biases, computed in fp32), applied to the loaded DiT |
| `distill_dir` | NULL = none | a distill bundle directory. The engine takes its time embedding (as `time_embedding_file`); `mi_gen_params.distill_dir` takes the rest per generation (see [Distill bundles](#distill-bundles)). |
| `certificate_dirs`, `certificate_dirs_count` | NULL, 0 | reserved since 2.0: ignored, neither read nor checked |
| `dit_direct_io` | `MI_DIRECT_IO_AUTO` | how the DiT's load reads its files when `direct_io` is 0 (below) |
| `te_host_pin` | `MI_TE_HOST_PIN_AUTO` | whether a host-resident text encoder may hold a pinned copy (`MI_CAP_TE_HOST_PIN`, below) |

The three sizes (`vram_budget_bytes`, `vram_reserve_bytes`, `host_ram_budget_bytes`) share one range rule: 0, or a value in [1 MiB, 1 TiB]. Anything else fails `mi_engine_create()` with `MI_ERR_INVALID_ARGUMENT`, naming the field.

The time embedding is per engine because it replaces part of the loaded DiT. Setting both `time_embedding_file` and a `distill_dir` whose bundle has a different time embedding is `MI_ERR_INVALID_ARGUMENT`. A `distill_dir` that is not a directory is `MI_ERR_INVALID_ARGUMENT`; a directory with none of the bundle files is `MI_ERR_MODEL`. A malformed time embedding file fails `mi_engine_create()` with `MI_ERR_MODEL`: it is read when the pipeline is constructed.

**DiT reads.** `direct_io` 1 forces direct reads everywhere. `dit_direct_io` only refines the DiT's load when `direct_io` is 0:

| `mi_direct_io` | The DiT load |
|---|---|
| `MI_DIRECT_IO_AUTO` (0) | adaptive: direct I/O (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`), unless it gains nothing on the files (tmpfs, network file systems, btrfs compressed extents, a refused open); buffered for a parking reload. With at least 90 % of the files already in the page cache (Linux: probed before every load; Windows: once buffered loads measure faster), whichever mode this machine measured faster from the cache: each mode is tried once while its rate is unknown, and the rates are kept in the hardware profile under `cache_dir`, so a new process tries a mode again only when its rate is missing there (a parking reload then follows that choice too) |
| `MI_DIRECT_IO_BUFFERED` (1) | buffered reads |
| `MI_DIRECT_IO_DIRECT` (2) | direct I/O; a refused open still reads buffered |

Streamed DiT blocks (a DiT that does not fit the device) read buffered every step, served by the page cache, unless `direct_io` is 1. `mi_result_dit_load()` and `MI_TEXT_DIT_LOAD_NOTE` say what a load used and why.

**Text-encoder pinning.** A host-resident text encoder (the plan keeps it in RAM between prompts) can hold its copy in pinned memory, which uploads faster and costs page-locked RAM:

| `mi_te_host_pin` | Pinned copy |
|---|---|
| `MI_TE_HOST_PIN_AUTO` (0) | where the machine clearly has the room |
| `MI_TE_HOST_PIN_ON` (1) | whenever the copy fits the host limit |
| `MI_TE_HOST_PIN_OFF` (2) | never |

A pinned copy whose allocation fails streams the encoder from its files instead; `MI_TEXT_TE_NOTE` says so.

The three file fields are UTF-8 paths; NULL or `""` means the bundle's component. A path that does not exist, or names a directory, fails `mi_engine_create()` with `MI_ERR_INVALID_ARGUMENT`, naming the field. ComfyUI and quantized files are supported as component files only: `model_path` stays a bundle directory (`MI_CAP_MODEL_FILES`, not `MI_CAP_COMFYUI_MODELS`).

### Tier and attention

The caller chooses the tier; the engine runs it or refuses it. The loaded weights fix the GEMM precision (an INT8 DiT runs INT8 GEMMs) and the tier decides the attention kernel. A tier the loaded models cannot run fails `mi_generate()` with `MI_ERR_INVALID_ARGUMENT`, and the message names the reason. The decision depends only on the tier, `attention`, the loaded files and the GPU, never on the request's size or settings.

| `mi_tier` | Runs |
|---|---|
| `AUTO` (0) | reserved since 2.0: refused by `mi_engine_create()` (`MI_ERR_INVALID_ARGUMENT`, the message names the choices) |
| `REFERENCE` (1) | the DiT's bf16 GEMMs with exact attention. Refused for a quantized DiT or text encoder file and for an 8-bit `attention`. |
| `HIGH` (2) | the loaded DiT's GEMMs (INT8 or bf16) with 8-bit attention. Refused for a 4-bit DiT, and for a bf16 DiT with exact attention (that is `REFERENCE`). |
| `TURBO` (3) | a 4-bit DiT file (`dit_file`: INT4 ConvRot, or NVFP4 on an RTX 50) with 8-bit attention. Refused for any other DiT. |

`HIGH` and `TURBO` default to `Q8_FP8` where the GPU has FP8 tensor cores (compute capability 8.9 and later), else `Q8_FP16`; `attention` overrides it. Capability refusals stay: an NVFP4 DiT needs FP4 tensor cores (`MI_ERR_MODEL` on other GPUs), an 8-bit `attention` the GPU or build cannot run is refused, and so are `HIGH` and `TURBO` on a GPU or build without any 8-bit kernel unless `attention` is `EXACT` (never a silent fallback). `mi_result_tier()` and the report say what ran.

| `mi_attention` | Pipeline name |
|---|---|
| `DEFAULT` (0) | unset: the tier's default |
| `EXACT` (1) | `exact` |
| `Q8_FP16` (2) | `q8_fp16` |
| `Q8_FP16_F32ACC` (3) | `q8_fp16_f32acc` |
| `Q8_FP8` (4) | `q8_fp8` |
| `Q8_FP8_F32ACC` (5) | `q8_fp8_f32acc` |

What each tier, mode and budget costs and gives: [integration.md, Performance](integration.md#performance).

### Preload and soft unload

A launcher can load the models before the first generation, and free the GPU without losing the engine.

**`mi_engine_preload(engine, params, &result)`** (`MI_CAP_PRELOAD`) loads what the engine keeps resident for a request like `params`, without generating:
- the arena at the plan's size;
- the DiT, as the plan keeps it. When the plan parks the DiT for the text encoder, it is not loaded: a generation that encodes a new prompt would evict it first. `MI_TEXT_PRELOAD_NOTE` says so;
- the VAE decoder, and the VAE encoder of a request that encodes an image (an init image below strength 1 or with a mask, control images, or a hires fix);
- the request's LoRAs and control adapter;
- a host-resident text encoder's copy (an encoder streamed from its files has nothing to keep).

The request is planned as `mi_generate()` plans it: size, steps, CFG, LoRAs, control adapter, and the prompt's length, which sizes the workspaces. `params` NULL means the defaults. A NULL or blank prompt is allowed here only, and plans for a short prompt. `mi_gen_params_validate()` checks a generation's request and so requires a prompt: validate a preload request as a copy with a stand-in prompt. Everything else is checked as for `mi_generate()`, with the same failures. The embedding cache is not consulted: the text encoder is planned as for a new prompt. A following `mi_generate()` of a request like it loads nothing (`mi_memory_stats.dit_loads` 0) and gives the same image bits as without the preload.
- It occupies the engine like a generation: `MI_ERR_BUSY` while a job runs, and `mi_cancel()` stops it between models and DiT weight groups. It then returns `MI_CANCELLED` with a result. The engine stays usable and keeps what was loaded before the cancel; a DiT whose load was cancelled is dropped.
- `params`' callbacks run on the calling thread. The plan callback runs once, after planning and before any load. Then the progress callback runs before each load step, also for a model that is already resident, with `step` 0 and `total_steps` 0, in this order: `MI_PHASE_TEXT_ENCODER` before a host-resident text encoder's copy, `MI_PHASE_DECODE` before the VAE encoder of a request that encodes an image, `MI_PHASE_DIT_LOAD` before the DiT, `MI_PHASE_DECODE` before the VAE decoder. A step the plan skips reports nothing; `MI_PHASE_DENOISE` never comes, and there is no preview. `fraction` and `eta_seconds` cover these load steps. A `mi_cancel()` from a callback stops the preload before that step (`MI_CANCELLED`).
- `result` may be NULL. Otherwise it receives the preload's result on `MI_OK` and `MI_CANCELLED`: the plan, the tier, the timings and memory, `mi_result_dit_load()` and `MI_TEXT_PRELOAD_NOTE`. It has no image and `steps_run` 0. Release it with `mi_result_free()`.

**`mi_engine_release(engine, keep_host_copy, &info)`** (`MI_CAP_RELEASE`) is a soft unload. It frees the engine's device memory (the arena, every resident model, LoRA side branches, a control adapter's embedder) so that another application can use the GPU, and keeps the engine. The next `mi_generate()` plans and loads again.
- With `keep_host_copy` nonzero, the weights stay in host RAM where the host has the room: a margin of the available memory stays free, and the copy fits `host_ram_budget_bytes` when that is set. The DiT's weight files are read into pinned memory, which takes seconds; its next load uploads them from there over PCIe, bit-identical, and frees them (`MI_DIT_LOAD_SOURCE_HOST_COPY`). Streamed blocks, if any, still read the files. A host-resident text encoder keeps its copy.
- With `keep_host_copy` 0, every host copy is freed too.
- The embedding cache stays either way.
- `mi_cancel()` from another thread stops the DiT's host copy between 64 MiB reads. The release itself always completes: it frees the device memory, keeps a host-resident text encoder's copy and frees nothing else. A copy that did not fit, failed or was cancelled is not kept, and `info.note` says why.
- A `mi_cancel()` from another thread that lands during the call, after it started and before it frees anything, stops it whole (a cancel sent before the call has no effect: like every job, the release clears the cancel flag when it marks the engine busy, so a cancel never carries over): nothing is released, it returns `MI_CANCELLED`, and `info` has `vram_released_bytes` 0, the host copies already held, and the note `cancelled before it started`. The engine keeps its device memory, so the next generation loads nothing.
- It returns `MI_OK`, or `MI_CANCELLED` as above, unless: `MI_ERR_INVALID_ARGUMENT` for a NULL engine, `MI_ERR_ABI_MISMATCH` for an unknown `info->struct_size`, `MI_ERR_BUSY` while a generation, a preload or another release runs (these three before anything is freed), and `MI_ERR_CUDA` when the device fails (free the engine then).
- `info` may be NULL. Otherwise it is an output struct (`struct_size` set by the caller):

| `mi_release_info` field | Meaning |
|---|---|
| `seconds` | the whole release, `host_copy_seconds` included |
| `vram_released_bytes` | the arena plus the device memory held outside it |
| `host_copy_bytes` | host RAM the kept copies hold |
| `host_copy_seconds` | reading the DiT's files into RAM |
| `dit_host_copy` | 1: the DiT's weights are kept in RAM |
| `te_host_copy` | 1: a host-resident text encoder's copy is kept |
| `note[512]` | why a copy was not kept; `""` otherwise. Cut at 511 bytes on a UTF-8 character boundary. |

`mi_engine_free()` remains the hard unload: it frees everything, host copies and the embedding cache included.

## Generation

`mi_gen_params` mirrors the asset `parameters.json`:
- `prompt`, `negative_prompt`
- `width`, `height`, `steps`, `cfg`, `shift`, `seed` (-1 = random). `steps` and `cfg` default to "unset" (see [Unset markers](#unset-markers)).
- `uncond_skip_layers` (default `{22}`)
- `slg_layers`, `slg_scale`, `slg_start`, `slg_stop`
- `max_sequence_length`
- `sampler` (default `EULER`)

The fields after the 0.1.0 layout (new in 0.2.0):

| `mi_gen_params` field | Default | Meaning |
|---|---|---|
| `scheduler` | `MOSS` | sigma schedule. `MOSS` with `EULER` is the official pipeline, bit for bit. |
| `noise_layout` | `OFFICIAL` | layout of the initial noise |
| `noise_source` | `PHILOX` | noise of the stochastic samplers |
| `save_metadata` | 1 (0 for a 168-byte struct) | `mi_save_png()` embeds the generation settings (see [PNG metadata](#png-metadata)) |
| `flow_multiplier` | 1.0 | flow sigma multiplier; finite, > 0. `MOSS` ignores it. |
| `eta` | 1.0 | `EULER_ANCESTRAL`, `DPMPP_2M_SDE`; finite, ≥ 0 |
| `s_noise` | 1.0 | noise multiplier of the stochastic samplers; finite, ≥ 0 |
| `dataspace_path` | NULL | the `DATASPACE` sampler's parameters file (`latent_mix.Q` and `step_codes`). Required by it, and then it must exist. The other samplers ignore it. |
| `loras`, `loras_count` | NULL, 0 | runtime LoRAs, see below |
| `include_prompts` | 1 (0 for a 168-byte struct) | the PNG metadata includes the prompt and negative prompt |
| `_reserved2` | 0 | must be 0 |

The fields after the 240-byte layout:

| `mi_gen_params` field | Default | Meaning |
|---|---|---|
| `init_image`, `init_image_path` | NULL | the init image of img2img: RGB pixels (`mi_image`, 3 channels) or a PNG / JPEG file. At most one of the two. |
| `mask`, `mask_path` | NULL | the inpainting mask: 1-channel pixels or a file (read as grayscale), the init image's size. At most one of the two; needs an init image. |
| `mask_encoding` | `MI_MASK_U8` | how mask values are read (below) |
| `resize` | `MI_RESIZE_CROP` | how the init image and mask are brought to `width` × `height` (below) |
| `denoise_strength` | -1 (unset: 0.75 with an init image, else 1.0) | [0, 1], the share of the schedule run from the init image. Below 1 needs an init image. |
| `hires_width`, `hires_height` | 0, 0 (off) | the hires fix's target size: multiples of 16, at most 4096 per side and 4.2 MP. Negative is invalid. `hires_strength` and `hires_steps` are checked only when the hires fix is on. |
| `hires_strength` | 0.45 | [0, 1], the hires pass's denoise strength |
| `hires_steps` | 0 (= `steps`) | the hires pass's schedule, 0 to 10000 |
| `_reserved3` | 0 | must be 0 |
| `distill_dir` | NULL | a distill bundle directory, expanded per generation |

The fields after the 320-byte (1.0.0) layout:

| `mi_gen_params` field | Default | Meaning |
|---|---|---|
| `control_adapter` | NULL (none) | a control adapter file (`MI_CAP_CONTROL`), see [Control adapters](#control-adapters). It must exist. |
| `control_lora_strength` | 1.0 | the strength of the adapter's LoRA part; finite and within the range of a float |
| `control_images`, `control_images_count` | NULL, 0 | its conditioning images (`mi_control_image`), at most 16; they need `control_adapter` |
| `cfg_truncation` | 1.0 (off) | [0, 1]: CFG truncation (`MI_CAP_ACCEL`), see [Accelerators](#accelerators) |
| `step_cache` | `MI_STEP_CACHE_OFF` | the step cache |
| `_reserved4` | 0 | must be 0 |
| `step_cache_threshold` | 0 (unset: the model's default) | 0, or (0, 1]. Used with the step cache only, but checked always. |

`mi_gen_params_validate()` performs the same checks as `mi_generate()` without an engine or GPU: the LoRA and control adapter files' existence, the distill bundle (opened) and the dataspace file (read: its step codes bound `steps`), and the image inputs without copying or decoding them. Pixel images get their sizes and channels checked, and a `MI_MASK_BINARY01` mask its values. Image files get checked for existence, for being regular files and for size (at most 64 MiB); they are decoded by `mi_generate()`, which reports a file it cannot decode, or a binary01 file mask with other values, as `MI_ERR_INVALID_ARGUMENT`. Also left to `mi_generate()`: the match of a bundle's time embedding with the engine's, and the control adapter's format and kinds. A launcher can call it as the user types.

### Unset markers

`mi_gen_params_init()` at the current size (376 bytes) and at the 1.0.0 size (320 bytes) leaves the fields whose default depends on the rest of the request unset. Each resolves when the request is checked:

| Field | Unset | Resolves to | Invalid |
|---|---|---|---|
| `steps` | 0 | the dataspace file's step count with the `DATASPACE` sampler (e.g. a distill bundle's), else `MI_DEFAULT_STEPS` (50) | negative |
| `cfg` | negative (finite) | 1.0 with `distill_dir`, else `MI_DEFAULT_CFG` (3.0) | NaN, ±infinity |
| `denoise_strength` | negative (finite) | `MI_DEFAULT_DENOISE_STRENGTH` (0.75) with an init image, else 1.0 | NaN, ±infinity; above 1 |
| `sampler` | `EULER` with no `dataspace_path` | a distill bundle's `DATASPACE` sampler, when the bundle has a dataspace file | |

- The macros `MI_DEFAULT_STEPS`, `MI_DEFAULT_CFG` and `MI_DEFAULT_DENOISE_STRENGTH` let a UI pre-fill its fields with what an unset field resolves to.
- An explicit value stays as given. An explicit `denoise_strength` of 1.0 with an init image and no mask runs txt2img, and `MI_TEXT_IMAGE_NOTE` says why.
- With the `DATASPACE` sampler, `steps` must lie in [1, the file's step-code count]; otherwise `MI_ERR_INVALID_ARGUMENT` names the count.
- With a distill bundle that has a dataspace file, `EULER` is the unset marker and cannot be forced: choose any other sampler to keep the bundle's from being selected.
- The older sizes keep their init values: steps 50 and cfg 3.0 for 168- and 240-byte structs. Those cannot carry an init image, so their strength resolves to 1.0.

### Image input

`mi_image {const uint8_t* pixels; int32_t width, height, channels; int32_t _reserved0;}` is a fixed 24-byte element without `struct_size`. The pixels are row-major, tightly packed and 8-bit: `[height][width][channels]`. An init image has 3 channels (RGB), a mask 1. The library copies the pixels during `mi_gen_params_validate()` / `mi_generate()`; the caller may free them as soon as the call returns.

Limits, checked before any pixel is read: each side 1 to 16384, at most 64,000,000 pixels; files at most 64 MiB. `pixels` NULL, another channel count, a nonzero `_reserved0`, both pixels and a path for one image, or a file that cannot be read or decoded are `MI_ERR_INVALID_ARGUMENT`. A file that cannot be read is `MI_ERR_INVALID_ARGUMENT` too, not `MI_ERR_IO`: the decoder does not tell read errors from format errors.

**The output size is `width` × `height`.** The init image (and its mask) is resized to it with `resize`; the library does not take the size from the image. Set `width` and `height` from the init image (rounded to multiples of 16 within the size bounds) to keep its aspect ratio. `moss-infer generate --init-image` does this itself when no size is given; the C API does not.

| Mode | Set | Runs |
|---|---|---|
| img2img | an init image, `denoise_strength` < 1 | the init image is resized to the request size, VAE-encoded, and the last `denoise_strength` share of the schedule runs from it |
| inpainting | an init image and a mask | the same, blending the kept region back after every step. White (≥ 128) regenerates, black keeps. Works at any strength. |
| hires fix | `hires_width` × `hires_height` | the first pass (txt2img, img2img or inpainting) is decoded, Lanczos-upscaled, encoded and refined at the target size |

- `denoise_strength` unset with an init image is 0.75. An explicit 1 with an init image and no mask is txt2img: the image is not used, and `MI_TEXT_IMAGE_NOTE` says so. 0 runs no step: the result is the image's VAE round trip.
- The steps that run: of a pass's N scheduled steps (after snapping), the first `int(N - N * strength)` are skipped. `mi_result_steps_scheduled()` and the progress callback's `total_steps` count the steps that run, both passes of a hires fix included.
- The `DATASPACE` sampler cannot start from an image (img2img, inpainting, hires fix).

| `mi_mask_encoding` | Values |
|---|---|
| `MI_MASK_U8` (0) | 0..255 as given: ≥ 128 regenerates, < 128 keeps |
| `MI_MASK_BINARY01` (1) | every value 0 (keep) or 1 (regenerate), scaled to 0 / 255. Any other value is `MI_ERR_INVALID_ARGUMENT`. Applies to `mask` and `mask_path` alike. |

| `mi_resize_mode` | Pipeline name | Meaning |
|---|---|---|
| `MI_RESIZE_CROP` (0) | `crop` | cover the target keeping the aspect ratio, then center-crop |
| `MI_RESIZE_FIT` (1) | `fit` | fit inside keeping the aspect ratio, fill the two bands from the edge rows / columns |
| `MI_RESIZE_STRETCH` (2) | `stretch` | scale each axis independently |

**Untrusted images.** The image file decoder (`init_image_path`, `mask_path`, and `mi_control_image.image_path`) is the vendored stb_image. It is hardened (the file-size cap, a header check before any allocation, fuzzing) but it is not memory-safe. A program that takes images from untrusted parties (a network, web uploads, other users' files) must decode them itself with a memory-safe decoder and pass pixels in `init_image` / `mask` / `mi_control_image.image`, or decode in a sandboxed process.

### Control adapters

A control adapter (`MI_CAP_CONTROL`; e.g. the MOSS-Image color/depth control, `darask0/MOSS-Image-color-depth-control`) is one safetensors file that pairs a LoRA with a widened DiT input embedder. With `control_adapter` set, the embedder replaces the DiT's (never scaled), and the file's LoRA part joins the runtime LoRAs after `loras`, at `control_lora_strength`. It is the same engine path as `moss-infer generate --control`.

`mi_control_image {const char* kind; const mi_image* image; const char* image_path; double strength, start, end;}` is a fixed 48-byte array element without `struct_size`. There is no init function: set every field. A zeroed entry has strength 0 and an empty window, which means no control; the trained behaviour is strength 1, start 0, end 1.

| Field | Meaning |
|---|---|
| `kind` | required: one of the adapter's kinds (`MI_TEXT_CONTROL_KINDS`, e.g. `color`, `depth`), at most one image per kind. A kind without an image gets no control. |
| `image`, `image_path` | exactly one of the two. Pixels: 3 channels (RGB) or 1 (grayscale, replicated to RGB: a depth map, near = white), with the limits of [Image input](#image-input). A file is decoded as RGB, so a grayscale file is replicated. |
| `strength` | [0, 10]: multiplies the control latent; 0 = no control |
| `start`, `end` | 0 ≤ `start` ≤ `end` ≤ 1: the control is active on the steps i of a pass with `start` ≤ i / steps < `end` (the guidance windows' convention) |

Each image is brought to the pass's size with `resize` (as an init image), VAE-encoded once per pass size (a hires fix encodes it again at the target size), multiplied by its strength, and fed clean to every CFG row on the steps of its window. A control adapter works with every tier, streaming, sampler, img2img / inpainting and the hires fix.

Validation: `mi_gen_params_validate()` checks the adapter file's existence, the array (NULL with a count, more than 16 entries), each entry's kind, image (as an init image, without copying or decoding) and ranges, and a repeated kind. `mi_generate()` reads the adapter: a file that is not a control adapter is `MI_ERR_MODEL`, a kind it does not have is `MI_ERR_INVALID_ARGUMENT` (the message lists its kinds). `mi_result_control()` reports it.

### Reference images

Edit adapters (`MI_CAP_REFERENCE_IMAGES`; moss-edit-lora files such as `chinoll/Moss-Image-Edit`) are LoRAs trained to edit reference images. Load the adapter in `loras` and give the images in `reference_images` (at most 4, in order, the source image first); `reference_max_edge` (0: 512; 16 to 2048, floored to a multiple of 16; the adapter was trained at 256, its authors' workflow uses 512) sets their size. It is the same engine path as `moss-infer generate --reference`.

`mi_reference_image {const mi_image* image; const char* image_path; int32_t _reserved0, _reserved1;}` is a fixed 24-byte array element without `struct_size`; exactly one of `image` and `image_path`, the reserved fields 0. Pixels may have 1, 3 or 4 channels (converted like PIL `convert("RGB")`: gray replicated, alpha dropped); a file is decoded as RGB with its EXIF orientation.

- Each image is fit within the max edge (each side rounded to 16, at least 16, never upscaled beyond the rounding: the training rule), resized with PIL's Lanczos, and VAE-encoded once per request.
- Each sampling pass computes every block's keys and values of the reference tokens once (at t = 0, the references attending only to each other), and every row (cond and uncond) attends to its own tokens plus all reference tokens. A hires fix computes them again for its second pass.
- Skipped layers (`uncond_skip_layers`, the edit workflow's 22) skip only the rows: the reference keys are computed without skips (ComfyUI's cached path; the authors' uncached node would also skip layer 22 for the references in the uncond pass).
- Limits: at most 16384 reference tokens ((w/16)(h/16) summed over the sized images). `slg_scale > 0` (a third row) is refused (`MI_ERR_INVALID_ARGUMENT`), and so is a request that runs no sampling step (an init image at `denoise_strength` 0 without a hires pass that samples).
- Cost: a key / value cache of 12 KiB per reference token per block (504 KiB per token; 378 MiB for a 384x512 reference) on the denoise stack, attention over the extra keys every step, and one prefill (an R-token forward) per pass; the plan prices all three. Without reference images nothing of this runs.

`MI_TEXT_REFERENCES` gives the report's `references` object as JSON ("" when none ran); `MI_MESSAGE_REFERENCE_WARNING` warns about references without a moss-edit-lora adapter and an edit adapter without references. Reference images are `mi_generate()`'s only: a session's `plan_request` with them is `MI_ERR_INVALID_ARGUMENT`.

### Accelerators

The accelerators (`MI_CAP_ACCEL`) are lossy and opt-in, and off by default. They change the image; the tier is resolved as if they were off. They apply to each sampling pass (a hires fix's too). It is the same engine path as `moss-infer generate --cfg-truncation` / `--step-cache`.

- **CFG truncation**, `cfg_truncation` = c in [0, 1] (1 = off): classifier-free guidance only on the steps i of a pass with i / steps < c. The later steps run the conditional row alone. It has no effect without CFG (`cfg` 1).
- **Step cache**, `step_cache` = `MI_STEP_CACHE_FBCACHE` (First-Block Cache): every DiT call runs block 0. While block 0's output changed by less than `step_cache_threshold` (relative L1) since the last call that ran every block, the other blocks are skipped and their residual from that call is reused. `step_cache_threshold` 0 leaves it unset: the engine uses its default for the loaded model, 0.02 on the base model (higher values move its composition) and 0.03 on the SFT model, the files derived from it, and a model it cannot place. Larger values are faster and move the image further from the uncached one. A threshold you set is never changed, but one above 0.02 on the base model gets an `MI_MESSAGE_ACCEL_WARNING`.

| `mi_step_cache` | Pipeline name |
|---|---|
| `MI_STEP_CACHE_OFF` (0) | `off` |
| `MI_STEP_CACHE_FBCACHE` (1) | `fbcache` |

`mi_result_accel()` reports what they skipped and the threshold that ran (`MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE` says whether it was the request's or the model's default); `mi_result_accel_step_cache_diffs()` gives the change of block 0 at every DiT call, which is what the threshold was compared against; `MI_TEXT_ACCELERATORS` names the active ones, and `MI_MESSAGE_ACCEL_WARNING` what to tell the user about them.

### Distill bundles

A distill bundle (the lodestones data-space DMD repositories and their derivatives) is a directory with any of `time_embedding.safetensors`, `lora.safetensors` and `dataspace.safetensors`. `distill_dir` expands it per generation, as `moss-infer generate --distill` does:

| Bundle file | Effect |
|---|---|
| `lora.safetensors` | appended to `loras` at strength 1, after the caller's |
| `dataspace.safetensors` | sampler `DATASPACE` when the sampler is unset; `dataspace_path` whenever the sampler is `DATASPACE` without one (an explicit `DATASPACE` included); `steps` = its step-code count when `steps` is unset |
| (any bundle) | `cfg` 1, when `cfg` is unset |
| `time_embedding.safetensors` | must be the engine's: `mi_engine_params.distill_dir` (the same bundle) or `time_embedding_file` (the same file). Otherwise `mi_generate()` fails with `MI_ERR_INVALID_ARGUMENT`, saying to set it on the engine. |

The markers are those of [Unset markers](#unset-markers): with `mi_gen_params_init()`'s defaults, a bundle's choices apply. With the bundle's dataspace sampler, an init image or a hires fix is `MI_ERR_INVALID_ARGUMENT`: set another sampler. A directory that is not one is `MI_ERR_INVALID_ARGUMENT`; one without bundle files, or a malformed dataspace file, is `MI_ERR_MODEL`.

### Samplers and schedulers

| `mi_sampler` | Pipeline name | Notes |
|---|---|---|
| `EULER` (0) | `euler` | the official flow-matching Euler step |
| `AB2` (1) | `ab2` | 2nd-order Adams-Bashforth |
| `EULER_ANCESTRAL` (2) | `euler_ancestral` | stochastic; `eta`, `s_noise` |
| `HEUN` (3) | `heun` | 2 model calls per step |
| `DPMPP_2M` (4) | `dpmpp_2m` | |
| `DPMPP_2M_SDE` (5) | `dpmpp_2m_sde` | stochastic; `eta`, `s_noise` |
| `ER_SDE` (6) | `er_sde` | stochastic; `s_noise` |
| `RES_MULTISTEP` (7) | `res_multistep` | |
| `DATASPACE` (8) | `dataspace` | few-step data-space sampler; needs `dataspace_path`. Its schedule comes from the file, so `scheduler` does not apply. |

The stochastic samplers draw their noise from `seed`.

| `mi_scheduler` | Pipeline name |
|---|---|
| `MOSS` (0) | `moss` |
| `SIMPLE` (1) | `simple` |
| `NORMAL` (2) | `normal` |
| `KARRAS` (3) | `karras` |
| `EXPONENTIAL` (4) | `exponential` |
| `BETA` (5) | `beta` |
| `SGM_UNIFORM` (6) | `sgm_uniform` |

| `mi_noise_layout` | Pipeline name | Meaning |
|---|---|---|
| `OFFICIAL` (0) | `official` | one draw per image, seeded per image |
| `BATCH_STREAM` (1) | `batch_stream` | image 0 of one batch-shaped draw from the seed |

| `mi_noise_source` | Pipeline name | Meaning |
|---|---|---|
| `PHILOX` (0) | `philox` | torch's CUDA generator on a GPU-independent grid: the same noise on every GPU |
| `PHILOX_DEVICE` (1) | `philox_device` | the same algorithm on this GPU's grid (`torch.randn(device="cuda")` here) |
| `BROWNIAN_TREE` (2) | `brownian_tree` | k-diffusion's Brownian tree, on the CPU. It holds up to 45 latent-sized buffers in host memory. |

### LoRAs

`mi_lora {const char* path; double strength;}` is a fixed 16-byte array element without `struct_size`. `loras` points to `loras_count` of them, applied in order. The array and its strings are read during `mi_generate()` only.

Validation, in `mi_gen_params_validate()` and `mi_generate()`:
- at most 64 entries; a NULL `loras` with a nonzero count is rejected;
- every `path` is non-empty and names an existing regular file;
- every `strength` is finite and within the range of a float.

The files are safetensors in the training tools' formats: kohya and PEFT / diffusers low-rank adapters, LyCORIS LoHa and LoKr, and full deltas (`.diff`, `.diff_b`). Low-rank adapters on the block GEMMs run as side branches, so a strength change between generations is instant. The others are merged into the DiT when it loads. A file with no applicable key fails the generation with `MI_ERR_MODEL`. Unknown keys and lossy merges (adapters requantized on INT8 GEMMs; one warning per file) are reported as warnings, or fail with `MI_ERR_MODEL` when `lora_strict` is set. `mi_result_lora()` and `mi_result_message()` report them. Edit adapters (moss-edit-lora files, bare `.lora_A` / `.lora_B` keys, alphas in the `adapter_config` header) are read too: see [Reference images](#reference-images).

`lora_placement` (`mi_lora_placement`) fixes where the low-rank block adapters go, per module from its stored precision; nothing switches on its own:

| Placement | Ordinary LoRA | Edit adapter (moss-edit-lora) |
|---|---|---|
| `MI_LORA_PLACEMENT_DEFAULT` | side branch | side branch |
| `MI_LORA_PLACEMENT_MERGE` | merged (INT8 requantized); a module that cannot take a merge (INT4, NVFP4, INT8 without ConvRot) is `MI_ERR_MODEL`, naming its format | the same |
| `MI_LORA_PLACEMENT_SIDE_BRANCH` | side branch | side branch |

The default is the side branch on every precision: loads are fast and strength changes, or adding and removing an adapter, cost no reload. For the edit adapter (rank 64) it costs about 6.5 % per step on bf16 and about 15 % on INT8 (dev box). `MI_LORA_PLACEMENT_MERGE` costs nothing per step but merges on the host at load (dev box, edit adapter: about 19 s more per bf16 load, about 50 s on INT8), and changing what is merged reloads the DiT: on bf16 it suits long editing sessions with a fixed adapter. On INT8, merge + requantize also moves the edit further from the reference implementation than the side branch or a baked file does. **For frequent editing on INT8, use a DiT file with the adapter baked in** (`moss-infer quantize <bundle> -o edit_int8.safetensors --lora <adapter>`, about 2 minutes once): no per-step cost, no merge at load, no LoRA to pass, and it streams from its file. An edit adapter's merged block modules keep no pristine copy (it would be the whole DiT's size in RAM), so its strength or placement changes reload the DiT; other merged modules re-merge in place as before. With `lora_strict`, an edit adapter merged into INT8 (`MI_LORA_PLACEMENT_MERGE`) fails, naming `MI_LORA_PLACEMENT_SIDE_BRANCH` and a baked file. Merged blocks that stream need a pinned host copy of them: when the host budget cannot hold it, the plan fails with `MI_ERR_OUT_OF_MEMORY`, naming the side branch, a larger `host_ram_budget` and a baked file. A DiT file with the adapter baked in (`moss-infer quantize --lora`, which reads moss-edit-lora files) needs no merge and streams from its file.

### Callbacks

- **progress** — `mi_progress {phase, step, total_steps, step_seconds, elapsed_seconds, fraction, eta_seconds}`. Called at each phase start (`TEXT_ENCODER`, `DIT_LOAD`, `DENOISE`, `DECODE`, in that order; the first two are entered even on a cache hit or a resident DiT) and after every sampling step. `total_steps` is the schedule's step count, which can be below `steps` (the `MOSS` and `BETA` schedules drop repeated sigmas).
  - `fraction` and `eta_seconds` (`MI_CAP_PROGRESS_FRACTION`) cover the whole job: every phase, both passes of a hires fix, a preload's load steps. `fraction` is the share done, 0 to 1, never decreasing within a job and below 1 in every callback (the call's return is the end). `eta_seconds` is the estimated time left, negative when unknown. Drive a progress bar and its ETA from them: `step / total_steps` is 0 through the text encoder and the loads, and covers only the sampling steps.
  - The estimate starts from the plan's time estimate, plus a load for each model not loaded yet: its size at the hardware profile's transfer rates (probed or learned; the built-in defaults before the first probe). The rates lag by one job: a job is priced with the previous job's rates, an engine's first job with the profile cached under `cache_dir` (or the defaults). The compute of an image encode is not priced, only the VAE encoder's load. It is corrected as the job runs: a phase that ended counts at its measured time, and the remaining steps of a pass at its measured step time (the first step, which carries the pass's setup, sets the price only until a later step is measured). The first job of an engine has the roughest estimate.
  - A DiT load is a single step: no callback comes during it.
  - The library allocates the struct and sets `struct_size`. Read `fraction` and `eta_seconds` only when `struct_size >= 56`; a library without them passes 40.
- **plan** — called once per generation, after planning and before the first phase. It receives the plan as human-readable lines and as JSON.
- **preview** — called after every `preview_interval`-th sampling step and after the last one, before that step's progress callback, with `(rgb, width, height, step, user_data)`.
  - `rgb` is RGB8 `[height][width][3]` at the requested image size (`width` × `height` of the request), valid during the call only.
  - It is a linear latent → RGB map of the x0 estimate after that step, without the VAE: a coarse preview, not the final image.
  - Setting the callback adds the preview's buffers to the denoise phase's plan. Leave it NULL when no preview is shown.
  - Each preview is a full-size device-to-host copy, synchronous on the generating thread, plus whatever the callback does (e.g. a PNG write). For large images use a `preview_interval` of about 5.
  - `mi_write_png()` writes a preview to a file.

### PNG metadata

With `save_metadata` set (the default of 240-, 320- and 376-byte structs), `mi_save_png()` writes two text chunks after `IHDR`, the same ones as `moss-infer generate`:

| Keyword | Chunk | Content |
|---|---|---|
| `generation` | iTXt | one JSON object, schema `mossinfer.generation/1`: engine version and tier, models by file name, LoRAs and strengths, prompt and negative prompt, sampler, scheduler, steps, CFG, shift, seed, size, and extra settings (attention, DiT linears, noise layout, noise source, the image input and hires fix, the control adapter and its images, the accelerators that changed the image) |
| `parameters` | tEXt (iTXt when not ASCII) | the plain-text settings block that image tools and galleries read |

With `include_prompts` 0, the prompt and negative prompt are left out of both chunks. Model and LoRA names are reduced to their base names, so no directory reaches the file. `save_metadata` 0 writes a PNG without text chunks.

## Results

| Accessor | Returns |
|---|---|
| `mi_result_rgb(r, &bytes)` | row-major RGB8 `[height][width][3]`, or NULL when cancelled |
| `mi_result_width/height/seed/steps_run/steps_scheduled/cancelled` | scalars |
| `mi_result_timings(r, &t)` | `total, load, encode, denoise, decode` and their parts, in seconds; then `vae_encoder_load_s`, `image_encode_s` (every VAE encode) and `image_ops_s` (resizes, the hires upscale, mask preparation, inpainting blends), which are not part of `load_s`; then `lora_load_s` (reading LoRA and control adapter files, uploading the side branches) and `lora_merge_s` (merging adapters into a loaded DiT in place), 0 when the stack was reused. A 104-byte (0.1.0) or 128-byte (1.0.0) `mi_timings` is accepted and gets its bytes. |
| `mi_result_image(r, &i)` | `mi_image_info`: the mode (`TXT2IMG`, `IMG2IMG`, `INPAINT`), and per pass (the first, the hires fix's) the size, steps scheduled, the start step (skipped by the strength), steps run and the strength; `first_encoded`; `hires_ran`. The hires fields are 0 without the hires fix. |
| `mi_result_memory(r, &m)` | arena capacity and high-water, device peak, host private start and peak; then, at the end of the call, `host_copy_bytes` (host copies kept across calls: a release's, a host-resident text encoder's), `merge_memo_bytes` (host RAM the DiT keeps to switch a merged adapter set back on, at most 1 GiB) and `dit_loads` (DiT loads in the call: 0 when it was resident). A 56-byte (1.0.0) `mi_memory_stats` is accepted and gets its 56 bytes. |
| `mi_result_dit_stream(r, &s)` | DiT block streaming over the denoise: streamed, resident blocks, buffers, stalls, bytes copied, copy and stall seconds, GB/s, hidden fraction, source (`"file"` / `"host_pinned"`). All zero when every block was resident. |
| `mi_result_dit_load(r, &d)` | `mi_dit_load_info` of the call's last DiT load (`MI_CAP_RELEASE`): `source` (`MI_DIT_LOAD_SOURCE_NONE`: no load, `FILES`, `HOST_COPY`: a release's copy), `io` (`MI_LOAD_IO_DIRECT` or `BUFFERED` for `FILES`, else `NONE`), `bytes`, `seconds`; `MI_TEXT_DIT_LOAD_NOTE` says why it read buffered. All zero without a load. |
| `mi_result_block_path(r)` / `mi_result_block_path_note(r)` | the DiT block path that ran (`OP_BY_OP`, `FUSED` or `FUSED_GRAPHS`), and why it differs from the requested one (`""` when it does not). A cancel that landed before the denoise (text encoder or DiT load) ran no block; the path is then `FUSED`, the engine default, whatever was requested. |
| `mi_result_plan_explanation(r)` | the execution plan as text |
| `mi_result_report_json(r)` | the `moss-infer generate --json-report` document; `"image"` is the last path given to `mi_save_png()`. NULL, with the message set, for a NULL result or when rendering fails (out of memory). |
| `mi_save_png(r, path)` | atomic PNG write (temporary file, then rename), with the [metadata](#png-metadata) when `save_metadata` was set |
| `mi_result_tier(r, &t)` | `mi_tier_info`: requested and effective tier (the same since 2.0), the attention that ran, DiT and text-encoder precision; `certified` and `run_certified` are reserved since 2.0 (0). A 32-byte (1.0.0) `mi_tier_info` is accepted and gets its 32 bytes. |
| `mi_result_text(r, kind)` | a string of the result (below); `""` when empty, NULL for a NULL result or an unknown kind |
| `mi_result_prompt_tokens(r, negative, &t)` | `mi_prompt_tokens` of the prompt (`negative` 0) or the negative prompt (1; all zero when CFG is off): `total_tokens` (chat template included), `kept_tokens`, `prompt_tokens` (the text's own), `dropped_tokens`, `max_length`, `truncated`. Any other `negative` is `MI_ERR_INVALID_ARGUMENT`. |
| `mi_result_lora_count(r)`, `mi_result_lora(r, i, &l)`, `mi_result_lora_file(r, i)` | per LoRA, in request order: `mi_lora_report {adapters, side_branch, merged, unknown_keys, warnings, notes}` and the file. An index out of range is `MI_ERR_INVALID_ARGUMENT` / NULL. |
| `mi_result_message_count(r, kind)`, `mi_result_message(r, kind, i)` | `MI_MESSAGE_LORA_WARNING`: losses to show the user (`"<module>: <reason>"`). `MI_MESSAGE_LORA_NOTE`: how the files were read. `MI_MESSAGE_CONTROL_WARNING` (2) and `MI_MESSAGE_CONTROL_NOTE` (3): the same for the control adapter's LoRA part. `MI_MESSAGE_ACCEL_WARNING` (4): what to tell the user about the accelerators for this model (a step-cache threshold that moves the base model's composition, a cache that never engaged). 0 / NULL for an unknown kind or index. |
| `mi_result_control(r, &c)` | `mi_control_info`: `active` (a control adapter was applied), `kinds_count` (the adapter's kinds), `encodes` (control image VAE encodes, one per image and pass size), and its LoRA part as in `mi_lora_report`: `lora_adapters`, `lora_side_branch`, `lora_merged`, `lora_unknown_keys`, `lora_warnings`, `lora_notes`. All zero without a control adapter. The LoRA part is not one of `mi_result_lora()`'s files. |
| `mi_result_accel(r, &a)` | `mi_accel_info`, over both passes of a hires fix: `cfg_truncated_steps` (steps run without the negative branch), `dit_calls` (DiT calls with the step cache on), `cached_calls` (of them, the calls that reused the cache), `lossy` (the request's settings activate an accelerator), `step_cache_threshold` (the threshold the cache ran with, the request's or the model's default; 0 with the cache off). All zero with the accelerators off. `lossy` and `MI_TEXT_ACCELERATORS` describe the requested, active settings, not what ran: a run cancelled before sampling still reports them. A 24-byte `mi_accel_info` (a pre-release header) is accepted and gets its 24 bytes. |
| `mi_result_accel_cached_steps(r, &n)` | the 1-based steps (counting on across the passes of a hires fix) with a call that reused the step cache, in order, and their count in `n`. NULL with `n` = 0 when there are none, and for a NULL result or a NULL `n`. Owned by the result. |
| `mi_result_accel_step_cache_diffs(r, &n)` | the relative change of block 0's output the step cache measured at every DiT call with the cache on (both passes), in order, and their count in `n`; NaN where a call had nothing to compare with. NULL with `n` = 0 when there are none, and for a NULL result or a NULL `n`. Owned by the result. |

| `mi_result_text_kind` | Text |
|---|---|
| `MI_TEXT_TIER_REASON` (0) | what the tier runs, e.g. `High tier: INT8 ConvRot GEMMs, q8_fp8 attention`, with any runtime LoRAs or lossy accelerators named |
| `MI_TEXT_CERTIFICATE` (1) | reserved since 2.0: `""` |
| `MI_TEXT_GPU_ARCH` (2) | the GPU class, e.g. `sm_89` |
| `MI_TEXT_DIT_SOURCE` (3) | the bundle's `transformer` folder, or `dit_file` |
| `MI_TEXT_TEXT_ENCODER_SOURCE` (4) | the bundle's `text_encoder` folder, or `text_encoder_file` |
| `MI_TEXT_VAE_SOURCE` (5) | the bundle's `vae` folder, or `vae_file` |
| `MI_TEXT_QUALITY_JSON` (6) | the report's `quality` object, as JSON |
| `MI_TEXT_IMAGE_NOTE` (7) | a remark on the image input, e.g. why an init image was not used; `""` for none |
| `MI_TEXT_TIME_EMBEDDING` (8) | the replacement step embedder that ran; `""` for none |
| `MI_TEXT_CONTROL_ADAPTER` (9) | the control adapter file; `""` without one |
| `MI_TEXT_CONTROL_KINDS` (10) | its kinds in channel order, comma-separated, e.g. `color,depth` |
| `MI_TEXT_CONTROL_BASE_MODEL` (11) | the file's `base_model` metadata (informational); `""` when absent |
| `MI_TEXT_CONTROL_ENCODE_SCHEDULE` (12) | how the control images were VAE-encoded: `untiled` or the tile plans |
| `MI_TEXT_ACCELERATORS` (13) | the accelerators that changed the run, e.g. `cfg_truncation 0.8, step_cache fbcache 0.03`; `off` when none (CFG truncation without CFG counts as off). An unset threshold shows the model's default that ran, or `default` when the run stopped before it was resolved. |
| `MI_TEXT_DIT_LOAD_NOTE` (14) | why the DiT's load read buffered, or what decided; `""` for a plain direct load or no load |
| `MI_TEXT_TE_NOTE` (15) | a remark on the text encoder, e.g. a pinned copy that could not be allocated; `""` for none |
| `MI_TEXT_PRELOAD_NOTE` (16) | `mi_engine_preload()`: what it did not load and why; `""` otherwise |
| `MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE` (17) | `request`, or `default for the base model` (`sft`, `unrecognized`); `""` with the step cache off or before the threshold was resolved |

`mi_tier_info.requested` and `effective` are the engine's tier, and `attention` is never `DEFAULT`. A generation cancelled before its tier was resolved reports the engine's tier with attention `EXACT` and DiT precision `BF16`.

The precisions map the pipeline's report as follows:

| Pipeline | `mi_precision` |
|---|---|
| DiT `bf16` (the bundle, or a bf16 file) | `BF16` (0) |
| DiT `int8` (any INT8 linear; mixed files count as INT8) | `INT8` (1) |
| text encoder `bf16` (the bundle's) | `BF16` (0) |
| text encoder `int8` | `INT8` (1) |
| text encoder `w4a8` | `W4A8` (2) |
| DiT `int4` (an INT4 ConvRot file, the Turbo tier) | `W4A4` (3) |
| DiT `nvfp4` (any NVFP4 linear, keep-list INT8 linears included; the Turbo tier on an RTX 50) | `NVFP4` (4) |

Strings returned by accessors are owned by the result. They stay valid until `mi_result_free()`; the report string stays valid until the next `mi_result_report_json()` or `mi_save_png()` call on the same result.

`mi_write_png(path, rgb, width, height)` writes RGB8 `[height][width][3]` as a PNG without metadata, atomically. It needs no result and is meant for previews. A NULL or empty path, NULL `rgb`, or a width or height outside [1, 16384] is `MI_ERR_INVALID_ARGUMENT`; a write failure is `MI_ERR_IO`.

The JSON report carries all of this as well: `quality`, `models` (sources, linears, tiers, text-encoder precision, time embedding), `loras`, `lora_warnings`, `lora_notes`, `tokens`, `img2img` (mode, note, VAE encoder, both passes), `control` (`null` without an adapter: the file, kinds, inputs, its LoRA part, encodes) and `accelerators` (`lossy`, the active ones, what they skipped, the cached steps).

## File inspection

`mi_inspect_file(path, &report)` (`MI_CAP_INSPECT`) says what a file is to the engine, without an engine, a GPU or any CUDA call, so a launcher can sort a models folder and offer each file in its role before it loads anything. The engine's own loaders decide, run up to the point where they would upload: `pipeline::inspect_dit_file()` for a DiT (the supported-model check and the quantization importer, as the pipeline loads a DiT file), the text-encoder and VAE file readers, `open_control_file()` and the LoRA stack. They read the safetensors header and the small tensors the loaders validate (the quantization configs, up to 4 KiB each, a DiT's scales, a text encoder's scales and F32 tensors, a LoRA's alpha tensors, a control adapter's embedder, checked for non-finite values as `mi_generate()` does), never the GEMM weights. It is thread-safe and holds no state, so it may run while an engine generates. The answer depends only on the file's content and the library's version and build, not on the GPU: a launcher may cache it per file (path, size, modification time), `mi_version()` and `mi_probe_info.build_flavour` (the NVFP4 kernels are a build option).

The kinds, in the order they are tried: `MI_FILE_KIND_CONTROL_ADAPTER` (a widened `video_embedder`), `DIT`, `TEXT_ENCODER`, `VAE`, `LORA` (at least one tensor applies to the MOSS-Image DiT; or tensor names in the LoRA conventions, LoRA, LoHa or LoKr factor names such as `.lora_A.`, `.lora_down.`, `.lora.up.`, `.hada_w`, `.lokr_w` (a bare `.alpha` does not count) (`weights::looks_like_lora()`: what the model-file checks refuse as a LoRA), that the LoRA reader refuses: a LoRA for another model or a malformed one, `loadable` 0 with the reader's message), else `UNKNOWN`. A file the engine cannot use is an answer, not an error: `UNKNOWN`, or its kind with `loadable` 0, and the reason in `MI_FILE_TEXT_NOTE` (the loader's message, e.g. a GGUF file, a distill bundle file, a header over 16 MiB, a W4A8 DiT that needs `moss-infer quantize`, an NVFP4 DiT). Only the call fails: `MI_ERR_INVALID_ARGUMENT` for a NULL or empty path or one that is not a regular file (a directory, a device, a pipe; on POSIX its type is read without opening it, then confirmed on a handle opened without blocking; on Windows, device and pipe namespace paths, DOS device names and UNC pipe, mailslot and IPC$ shares are refused by name before any open; any other path is opened for its attributes only (identification-level SQOS) and refused by its type), `MI_ERR_IO` for a missing or unreadable file (the system's reason, e.g. a folder without search permission).

**Untrusted files.** The paths may come from anyone. The header reader bounds what a hostile header costs: at most 16 MiB of JSON with at most 65,536 tensors, rank 8 and nesting depth 128, and every tensor inside the file and the data section exactly covered. Before any tensor is read, the header is checked against what the readers would read: the quantization importer (`quant::open_quant_model()`) requires every DiT linear to have MOSS-Image's shape (`weights::moss_dit_linears()`) and every text-encoder linear and tensor the MOSS-Image text encoder's (`weights::qwen35_text_tensors()`), and matches each scale to its linear. That bounds a DiT's scales (about 290 MiB in all for a fully NVFP4 file, read and held one linear at a time) and a text encoder's scales, codebooks, F32 tensors and their copies (a few MB). A control adapter's embedder and its copy may take at most 256 MiB (its header may name any number of control kinds); a file over that is not loadable (the note says so), while the published adapter's takes 2.4 MB. The quantization configs are read too, at most 4 KiB each. So inspection reads at most the 16 MiB header (parsed into a DOM of a few tens of MB at most), the configs and those tensor bounds, and holds at most about 256 MiB of tensors, whatever the header claims. `CApiInspect.MalformedFiles`, `HostileDitShapes`, `Nvfp4Dit` and `TextEncoderTensorShapes` cover truncated, oversized, deeply nested and lying headers.

| `mi_file_info` field | Meaning |
|---|---|
| `kind`, `loadable` | the role, and whether its loader accepts the file. What depends on the request or the GPU comes later: a control image's kind, a LoRA merge on INT4 weights, NVFP4's compute capability 12.0. |
| `file_bytes`, `tensor_count` | the file's size and its header's tensors (0 when the header cannot be read) |
| `precision` | a loadable DiT: `BF16`, `INT8`, `W4A4` or `NVFP4` (as `mi_tier_info.dit_precision` reports it after a run); a text encoder: `BF16`, `INT8` or `W4A8` |
| `tiers` | a loadable DiT: bit `1u << tier` for each `mi_tier` that `mi_generate()` accepts with it: `planner::resolve_tier()`'s verdict, the same call generate makes, with the bundle's bf16 text encoder and no explicit `attention`. Every GPU the engine supports runs an 8-bit attention kernel, so the bits hold on each of them; an NVFP4 DiT is judged for an RTX 50 with this build's kernels (`TURBO` only; a build without them refuses it). Bit 0 (`AUTO`) is never set. `mi_file_report_tier_reason(report, tier)` gives what the tier runs (e.g. `High tier: INT8 ConvRot GEMMs, 8-bit attention (q8_fp8; q8_fp16 on GPUs without FP8)`, with `(RTX 50 only)` for an NVFP4 file) or why it is refused. |
| `lora_rank`, `lora_adapters`, `lora_unknown_keys` | a LoRA or a control adapter's LoRA part, as the LoRA stack reads it: the largest low-rank rank (0 for LoHa / LoKr / full deltas only), the adapters on the DiT and the tensors not applied (as in `mi_lora_report`) |
| `control_kind_count` | a control adapter's kinds; `mi_file_report_control_kind(report, i)` gives each in channel order: the strings `mi_control_image.kind` takes |

`mi_file_report_text()` gives `MI_FILE_TEXT_NOTE`, `MI_FILE_TEXT_QUANTIZATION` (a loadable DiT's block linears as the engine runs them: `bf16`, `int8_convrot256`, `int4_convrot256` or `mixed (...)`; a text encoder's `bf16`, `int8_convrot` or `w4a8`), `MI_FILE_TEXT_BASE_MODEL` (the `base_model` metadata of a LoRA or control adapter) and `MI_FILE_TEXT_ADAPTER_TYPES` (`lora`, `loha`, `lokr`, `full`, comma-separated). The strings are owned by the report; release it with `mi_file_report_free()`.

## Component API

The component API (`MI_CAP_COMPONENTS`, 2.1) is `mi_generate()` split into its parts, so the host owns the sampling loop: its own sampler, guidance, schedule and control scheduling, with the engine's text encoder, DiT and VAE. Each `mi_denoise_step()` is one model evaluation of every row the host names, in one batched forward. A host running the engine's sampler formulas gets `mi_generate()`'s bits.

A session and its run are state of the engine: at most one session per engine and one run per session, so every call takes the `mi_engine*` and nothing can dangle. Conditionings are the only new handles.

```c
mi_session_params sp;  mi_session_params_init(&sp, sizeof sp);
mi_session_open(engine, &sp);

const char* prompts[2] = {"a red apple on a wooden table", ""};
mi_text_params tp;  mi_text_params_init(&tp, sizeof tp);
tp.prompts = prompts;  tp.prompts_count = 2;
mi_conditioning* cond[2];
mi_encode_text(engine, &tp, cond);

const int32_t skip22[] = {22};
const mi_run_row rows[2] = {{cond[0], NULL, 0, 0, 0}, {cond[1], skip22, 1, 0, 0}};
mi_run_params rp;  mi_run_params_init(&rp, sizeof rp);
rp.width = rp.height = 1024;  rp.steps = 50;
rp.rows = rows;  rp.rows_count = 2;
rp.guidance = MI_GUIDANCE_MOSS;                                   /* mi_generate's rows and windows */
rp.output = mi_sampler_prediction(MI_SAMPLER_EULER, MI_SCHEDULER_MOSS);  /* guided velocity */
rp.flags = MI_RUN_EXACT_SIGMA;
mi_run_begin(engine, &rp);

size_t n;  const float* s = mi_run_sigmas(engine, &n);
mi_tensor x, v;  mi_run_staging(engine, &x, &v, 1, NULL);        /* pinned host views */
mi_initial_noise(seed, MI_NOISE_LAYOUT_OFFICIAL, 1024, 1024, x.data, 128 * 64 * 64);
for (size_t i = 0; i + 1 < n; ++i) {
  mi_step_params st;  mi_step_params_init(&st, sizeof st);
  st.sigma = s[i];  st.x = x;  st.out[0] = v;
  mi_denoise_step(engine, &st, NULL);
  /* x = x + h * v, h = fp32(s[i+1] - s[i]), one rounding per op */
}
memcpy(latents, x.data, 128 * 64 * 64 * sizeof(float));         /* the staging is the run's: copy x out first */
mi_result* r;  mi_run_end(engine, &r);                            /* plan, tier, timings: no image */
mi_tensor lat = x;  lat.data = latents;                          /* your own host buffer */
mi_vae_decode(engine, &lat, &rgb);                              /* rgb: U8 [1024, 1024, 3] */
mi_session_close(engine);
```

**Conventions.**
- **Latents** are model space, F32 `[128, H/16, W/16]` per image (`[L, 128, h, w]` for a batch): the FLUX.2 VAE's 32 channels patchified 2 × 2 and batch-normed. That is ComfyUI's Flux2 `LATENT` as is: exchange it without any conversion, in either direction.
- **FLOW.** x = (1 − σ)·x0 + σ·noise. Each output is the DiT's raw output (the velocity), so denoised = x − σ·out. The DiT's timestep is fp32(σ × 1000).
- **No shift inside a step.** The engine applies no shift or multiplier to a call's σ: the host's model sampling has already applied them. The shift enters only the engine's own schedule (`mi_schedule()`, or `mi_run_params.sigmas` NULL).
- **The schedule is a hint.** A call's σ may be anything: an exact entry gives its step, any other σ the interval it falls in (`off_schedule_calls` counts them). `MI_RUN_EXACT_SIGMA` refuses off-schedule sigmas, a reproduction's guard against a one-ulp slip. Calls are unbounded: correctors, midpoints and restarts all work.
- **Rows.** `MI_GUIDANCE_HOST` (default): the host owns guidance and each call names its rows (`mi_step_params.rows`); every output is a row's raw velocity. `MI_GUIDANCE_MOSS`: `mi_generate()`'s layout, rows and windows, required for the guided outputs. A run has up to the session's `max_rows` rows over a latent batch of up to `max_latents`.
- **Tier.** A run uses the engine's explicit tier and attention, as `mi_generate()` does. Every image-affecting parameter is explicit; defaults are fixed constants (the step cache's threshold is required when it is on).

**Exchange.** Tensors are all device memory of the engine's device (`MI_MEMORY_DEVICE`, handed over by events on the host's `cuda_stream`, no copy) or all the run's pinned staging (`mi_run_staging()`; another host pointer is refused). `mi_run_adopt_staging()` makes caller-owned memory, e.g. a shared-memory region, the staging block. On return from `mi_denoise_step()`, whatever its status, the engine no longer touches x, control or the outputs.

**Ownership and threading.**
- Every array and string is read during the call only; a run copies what it keeps. A conditioning may be freed once `mi_run_begin()` has returned. Staging views are the engine's until `mi_run_end()`.
- Every component call is a job of the engine, as `mi_generate()` is: one at a time (`MI_ERR_BUSY`), and `mi_cancel()` stops it. `mi_generate()`, preload and release work between runs; while a run is open they return `MI_ERR_BUSY`.
- **Interrupts.** `mi_run_end()`, `mi_session_close()` and `mi_engine_free()` work at any moment from any thread: a step in flight is cancelled (`MI_CANCELLED`) and waited for, then the run ends; models stay loaded, so the next run loads nothing. From a callback of the running call, end and close return `MI_ERR_BUSY` and do nothing.
- After a run or session ended, calls on it return `MI_ERR_INVALID_ARGUMENT` ("no run is open").

**Performance.** One evaluation is one call: put every row of an evaluation (cond, uncond, the SLG row, the batch's latents) in the same `mi_denoise_step()`. A component run is planned as the equivalent `mi_generate()` request, holds the same memory and runs the same kernels, so a step costs what `mi_generate()`'s step costs. Splitting an evaluation over several calls costs speed and graph replay; it is never needed.

## Capabilities

`mi_capabilities()` returns the features this build implements. A launcher checks the bits at startup and shows the matching UI only when a bit is set, so the same launcher works with older and newer builds.

| Bit | Value | This build | Covers |
|---|---|---|---|
| `MI_CAP_PREVIEW` | 0x1 | set | `preview_callback` is called |
| `MI_CAP_COMFYUI_MODELS` | 0x2 | unset | a ComfyUI file or folder as `model_path` (`MI_MODEL_FORMAT_COMFYUI`) |
| `MI_CAP_MODEL_FILES` | 0x4 | set | `dit_file`, `text_encoder_file`, `vae_file`, including ComfyUI / community INT8 ConvRot and W4A8 files |
| `MI_CAP_LORA` | 0x8 | set | `loras`, `lora_strict`, the LoRA accessors |
| `MI_CAP_TIERS` | 0x10 | set | `tier`, `attention`, `mi_result_tier()` |
| `MI_CAP_SAMPLERS` | 0x20 | set | the samplers after `AB2`, `scheduler`, `noise_layout`, `noise_source`, `flow_multiplier`, `eta`, `s_noise`, `dataspace_path` |
| `MI_CAP_IMAGE_INPUT` | 0x40 | set | img2img and inpainting: `init_image`, `mask` and their paths, `mask_encoding`, `resize`, `denoise_strength`, `mi_result_image()` |
| `MI_CAP_HIRES` | 0x80 | set | `hires_width`, `hires_height`, `hires_strength`, `hires_steps` |
| `MI_CAP_DISTILL` | 0x100 | set | `distill_dir` (both structs), `time_embedding_file` |
| `MI_CAP_SYSTEM_CUDA` | 0x200 | set | cuBLASLt loaded at run time: the bundled copy, else (or with `MOSSINFER_CUDA_LIBS=system`) a compatible system copy; `mi_probe_info.cublaslt_*` |
| `MI_CAP_CONTROL` | 0x400 | set | control adapters: `control_adapter`, `control_lora_strength`, `control_images` (`mi_control_image`), `mi_result_control()`, `MI_TEXT_CONTROL_*`, `MI_MESSAGE_CONTROL_*` |
| `MI_CAP_ACCEL` | 0x800 | set | the lossy accelerators: `cfg_truncation`, `step_cache`, `step_cache_threshold`, `mi_result_accel()`, `mi_result_accel_cached_steps()`, `MI_TEXT_ACCELERATORS`; with the 2.0 fields `mi_result_accel_step_cache_diffs()`, `MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE` and `MI_MESSAGE_ACCEL_WARNING` |
| `MI_CAP_PRELOAD` | 0x1000 | set | `mi_engine_preload()`, `MI_TEXT_PRELOAD_NOTE` |
| `MI_CAP_RELEASE` | 0x2000 | set | `mi_engine_release()` with `keep_host_copy`, `mi_release_info`, `mi_result_dit_load()`, `MI_TEXT_DIT_LOAD_NOTE` |
| `MI_CAP_CERTIFICATE_DIRS` | 0x4000 | not set | reserved since 2.0 (`certificate_dirs` is ignored) |
| `MI_CAP_TE_HOST_PIN` | 0x8000 | set | `mi_engine_params.te_host_pin`, `MI_TEXT_TE_NOTE` |
| `MI_CAP_PROGRESS_FRACTION` | 0x10000 | set | `mi_progress.fraction` and `eta_seconds` |
| `MI_CAP_INSPECT` | 0x20000 | set | `mi_inspect_file()` and the `mi_file_report_*` accessors |
| `MI_CAP_COMPONENTS` | 0x40000 | set | the [component API](#component-api): `mi_session_*`, `mi_encode_text()`, `mi_conditioning_*`, `mi_run_*`, `mi_denoise_step()`, `mi_vae_*`, `mi_schedule()`, `mi_sampler_prediction()`, `mi_initial_noise()`, `mi_image_resize()`, `MI_TEXT_COMPONENT_NOTE` |
| `MI_CAP_REFERENCE_IMAGES` | 0x80000 | set | [reference images](#reference-images) of edit adapters: `mi_gen_params.reference_images` / `reference_max_edge`, `MI_TEXT_REFERENCES`, `MI_MESSAGE_REFERENCE_WARNING` |

`MI_CAP_TIERS` covers `TURBO` as well. `mi_engine_params.dit_direct_io` has no bit of its own: it comes with the 160-byte `mi_engine_params`, which every library with `MI_CAP_PRELOAD` accepts.

## Device probe

`mi_probe()` reports:
- the driver's CUDA version and this build's version, and whether they are compatible
- the device counts
- this library's flavour and the flavour the machine should install (`"cu13"`, `"cu12"` or `""`)
- a summary message
- the cuBLASLt copy the engine uses (`MI_CAP_SYSTEM_CUDA`, the fields after the 576-byte layout, see [cuBLASLt](#cublaslt))

It works on machines without a driver or GPU and creates no CUDA context. When the driver is older than this build's CUDA version (a `cu13` build on a CUDA 12 driver), the CUDA runtime cannot list devices; the probe then lists them through the driver API. Each device the build could otherwise run is marked unsupported with the driver as the reason, and `"cu12"` is recommended when such a device exists.

`mi_probe_device(i)` adds the name, PCI bus id, compute capability, total and free memory, and whether this build supports the device (and why not). It is implemented by `src/backend/probe.h`. An `i` outside [0, device count) fails with `MI_ERR_INVALID_ARGUMENT` before any free-memory query, and only device `i` has its free memory measured (`ProbeOptions::free_memory_device`). It also leaves no CUDA context behind:
- **Free memory comes from NVML** (`libnvidia-ml.so.1` / `nvml.dll`, loaded at run time; `nvmlDeviceGetMemoryInfo` by PCI bus id). NVML needs no context.
- **Fallback without NVML:** the driver API (`libcuda.so.1` / `nvcuda.dll`). It retains the device's primary context, calls `cuMemGetInfo` and releases the context again. The last release destroys it. If an engine already holds that context, the probe shares it and creates nothing.
- **Never** through the CUDA runtime (whose context would stay for the life of the process), and never `cudaDeviceReset`.
- `mi::backend::DeviceInfo::free_memory_source` records which method answered (`"nvml"` or `"driver"`). With a standard NVIDIA driver install it is NVML.

### cuBLASLt

The engine does not link cuBLASLt. It loads it at run time, once per process, at the first `mi_probe()` (with the current `struct_size`) or `mi_engine_create()`: by default (`auto`) the copy bundled next to the library, then a compatible system copy (the only choice of a `-system-cuda` package, which has no bundled copy). `MOSSINFER_CUDA_LIBS` (`auto`, `system`, `bundled`) and `MOSSINFER_CUBLASLT` (one file) steer the search; [integration.md](integration.md#cuda-libraries) lists the locations and the compatibility rule. Loading cuBLASLt creates no CUDA context.

| Field | Value |
|---|---|
| `cublaslt_source` | `"system"`, `"bundled"` or `"env"` (`MOSSINFER_CUBLASLT`); `""` when no copy is usable |
| `cublaslt_version` | `cublasLtGetVersion()` of that copy (`major * 10000 + minor * 100 + patch`, e.g. 130800); 0 when none |
| `cublaslt_path` | the file loaded (symbolic links resolved), UTF-8; `""` when none. A path of 512 bytes or more is never cut: the field stays `""` and `cublaslt_error` says why, while `cublaslt_source` and `cublaslt_version` are still set. |
| `cublaslt_error` | when no copy is usable: every candidate tried and why it was rejected (missing, not loadable, wrong major version, older than the build's), cut at 511 bytes (`mi_engine_create()`'s message has all of it); `""` otherwise |

Without a usable copy, `mi_probe()` still returns `MI_OK` with the reason in `cublaslt_error`: the rejected candidates first (wrong version, missing entry point, the bundled file reached by a system search), then one line with the files that do not exist. Strings in `mi_probe_info` and `mi_device_info` are cut at a UTF-8 character boundary. `mi_engine_create()` then fails with `MI_ERR_CUDA` and the same message, after its argument checks and before any CUDA call. A 576-byte `mi_probe_info` (the 0.2.0 size) gets no cuBLASLt fields and starts no search.

`src/backend/cuda_libs.{h,cpp}` implements the search, and `src/backend/cublaslt_shim.cpp` defines every `cublasLt*` function the engine calls, forwarding to the loaded copy. A new `cublasLt*` call needs an entry there, or the link fails with an undefined reference. The shared library keeps these definitions hidden, so it has no load-time dependency on cuBLASLt and exports no `cublasLt*` symbol (`check_package.py` checks both).

## Struct sizes

A versioned struct only grows at the end, so a program built against an older header passes a smaller `struct_size`. The library accepts every size the struct has had:

| Struct | Accepted `struct_size` |
|---|---|
| `mi_engine_params` | 80 (the 0.1.0 size, up to `_reserved1`), 120 (an earlier 0.2.0 header, up to `_reserved2`), 136 (0.2.0 and 1.0.0, up to `distill_dir`), 160 (2.0.0) |
| `mi_gen_params` | 168 (the 0.1.0 size, up to `user_data`), 240 (an earlier 0.2.0 header, up to `_reserved2`), 320 (0.2.0 and 1.0.0, up to `distill_dir`), 376 (2.0.0 and 2.1.0, up to `step_cache_threshold`), 408 (2.x, with the reference images and `lora_placement`) |
| `mi_timings` (output) | 104 (the 0.1.0 size, up to `vae_load_s`), 128 (0.2.0 and 1.0.0, up to `image_ops_s`), 144 (2.0.0) |
| `mi_memory_stats` (output) | 56 (1.0.0, up to `host_private_peak_bytes`), 80 (2.0.0) |
| `mi_probe_info` (output) | 576 (the first 0.2.0 size, up to `_reserved0`), 1624 (0.2.0 with the cuBLASLt fields) |
| `mi_tier_info` (output) | 32 (0.2.0 and 1.0.0, up to `text_encoder_precision`), 40 (2.0.0, with `run_certified`) |
| `mi_accel_info` (output) | 24 (a pre-release header, up to `lossy`), 32 (2.0.0, with `step_cache_threshold`) |
| every other struct | its one size so far (table below), the component structs (2.1) included |

Any other size is `MI_ERR_ABI_MISMATCH`, and the message lists the accepted sizes. `mi_progress` is the other way round: the library fills it for the progress callback with its own size in `struct_size` (40 up to 1.0.0, 56 from 2.0.0), and the client reads the fields that size covers.
- **`*_init(p, size)`** writes only the first `size` bytes: the defaults of the fields that layout has. The bytes after them are left alone.
- **Input structs** (`mi_engine_create()`, `mi_gen_params_validate()`, `mi_generate()`): the library reads only `struct_size` bytes. It copies them over a full struct filled with the defaults for that size, so every later field keeps the behaviour of the older release:
  - an 80-byte `mi_engine_params` has no `tier` field, so since 2.0 it creates no engine (`MI_TIER_AUTO` is refused); from 120 bytes on the caller sets the tier;
  - a 168-byte `mi_gen_params` runs with the `MOSS` scheduler, the default noise and no LoRAs, and its `save_metadata` and `include_prompts` are 0: `mi_save_png()` writes a PNG without text chunks, as 0.1.0 did. A 240-, 320- or 376-byte struct defaults `save_metadata` and `include_prompts` to 1;
  - a 120-byte `mi_engine_params` or a 240-byte `mi_gen_params` runs without image input, hires fix or distill bundle;
  - a 320-byte `mi_gen_params` runs without a control adapter and with the accelerators off;
  - a 136-byte `mi_engine_params` runs with the adaptive DiT load and the automatic pin policy.
- **Output structs**: the library writes only `struct_size` bytes.

`CApi.OldStructSizes`, `CApi.Earlier020StructSizes` and `CApi.Gen100StructSize` (`tests/unit/test_c_api.cpp`) and the C test `c_abi_compat` (`tests/c/test_abi_compat.c`, a hand-written 0.1.0-layout struct, label `cpu`; `c_abi_compat_gpu`, label `gpu`, also generates from it) check this.

## Frozen layout (ABI 1)

These are the layouts on 64-bit targets (LP64 and LLP64; 32-bit is not supported). Every field sits at its natural alignment, and every gap is a named `_reserved*` field, so GCC, Clang and MSVC produce the same layout from C and from C++.
- `src/api/c_api.cpp` `static_assert`s each size and offset.
- The C test `c_abi_layout` (`tests/c/test_abi_layout.c`, label `cpu`) checks the same table from a C11 compiler.

The enums are 4 bytes. ABI 1 is defined by this table: a change to an existing entry needs `MI_ABI_VERSION` 2. New trailing fields, structs and enum values do not.

| Struct | Size | Fields: offset |
|---|---|---|
| `mi_device_info` | 592 | `struct_size` 0, `index` 8, `name[256]` 12, `pci_bus_id[32]` 268, `compute_major` 300, `compute_minor` 304, `_reserved0` 308, `total_memory_bytes` 312, `free_memory_bytes` 320, `supported` 328, `unsupported_reason[256]` 332, `_reserved1` 588 |
| `mi_probe_info` | 1624 | `struct_size` 0, `driver_cuda_version` 8, `runtime_cuda_version` 12, `driver_compatible` 16, `device_count` 20, `supported_device_count` 24, `build_flavour[16]` 28, `recommended_flavour[16]` 44, `message[512]` 60, `_reserved0` 572; then `cublaslt_source[16]` 576, `cublaslt_version` 592, `_reserved1` 596, `cublaslt_path[512]` 600, `cublaslt_error[512]` 1112 |
| `mi_engine_params` | 160 | `struct_size` 0, `model_path` 8, `model_format` 16, `mode` 20, `device_index` 24, `_reserved0` 28, `vram_budget_bytes` 32, `vram_reserve_bytes` 40, `host_ram_budget_bytes` 48, `cache_dir` 56, `embedding_cache` 64, `direct_io` 68, `block_path` 72, `_reserved1` 76; then `dit_file` 80, `text_encoder_file` 88, `vae_file` 96, `tier` 104, `attention` 108, `lora_strict` 112, `_reserved2` 116; then `time_embedding_file` 120, `distill_dir` 128; then `certificate_dirs` 136, `certificate_dirs_count` 144, `dit_direct_io` 152, `te_host_pin` 156 (no padding needed) |
| `mi_progress` | 56 | `struct_size` 0, `phase` 8, `step` 12, `total_steps` 16, `_reserved0` 20, `step_seconds` 24, `elapsed_seconds` 32; then `fraction` 40, `eta_seconds` 48 |
| `mi_gen_params` | 320 | `struct_size` 0, `prompt` 8, `negative_prompt` 16, `width` 24, `height` 28, `steps` 32, `_reserved0` 36, `cfg` 40, `shift` 48, `seed` 56, `uncond_skip_layers` 64, `uncond_skip_layers_count` 72, `slg_layers` 80, `slg_layers_count` 88, `slg_scale` 96, `slg_start` 104, `slg_stop` 112, `max_sequence_length` 120, `sampler` 124, `progress_callback` 128, `plan_callback` 136, `preview_callback` 144, `preview_interval` 152, `_reserved1` 156, `user_data` 160; then `scheduler` 168, `noise_layout` 172, `noise_source` 176, `save_metadata` 180, `flow_multiplier` 184, `eta` 192, `s_noise` 200, `dataspace_path` 208, `loras` 216, `loras_count` 224, `include_prompts` 232, `_reserved2` 236; then `init_image` 240, `init_image_path` 248, `mask` 256, `mask_path` 264, `mask_encoding` 272, `resize` 276, `denoise_strength` 280, `hires_width` 288, `hires_height` 292, `hires_strength` 296, `hires_steps` 304, `_reserved3` 308, `distill_dir` 312; then `control_adapter` 320, `control_lora_strength` 328, `control_images` 336, `control_images_count` 344, `cfg_truncation` 352, `step_cache` 360, `_reserved4` 364, `step_cache_threshold` 368; then `reference_images` 376, `reference_images_count` 384, `reference_max_edge` 392, `_reserved5` 396, `lora_placement` 400, `_reserved6` 404 (408 bytes) |
| `mi_lora` | 16 | `path` 0, `strength` 8 (array element, no `struct_size`: frozen as it is) |
| `mi_image` | 24 | `pixels` 0, `width` 8, `height` 12, `channels` 16, `_reserved0` 20 (no `struct_size`: frozen as it is) |
| `mi_control_image` | 48 | `kind` 0, `image` 8, `image_path` 16, `strength` 24, `start` 32, `end` 40 (array element, no `struct_size`: frozen as it is) |
| `mi_reference_image` | 24 | `image` 0, `image_path` 8, `_reserved0` 16, `_reserved1` 20 (array element, no `struct_size`: frozen as it is) |
| `mi_timings` | 144 | `struct_size` 0, then 12 doubles from 8 in header order (`total_s` … `vae_load_s` at 96); then `vae_encoder_load_s` 104, `image_encode_s` 112, `image_ops_s` 120; then `lora_load_s` 128, `lora_merge_s` 136 |
| `mi_memory_stats` | 80 | `struct_size` 0, then 6 `uint64_t` from 8 in header order (`arena_capacity_bytes` … `host_private_peak_bytes` at 48); then `host_copy_bytes` 56, `merge_memo_bytes` 64, `dit_loads` 72, `_reserved0` 76 |
| `mi_dit_load_info` | 32 | `struct_size` 0, `source` 8, `io` 12, `bytes` 16, `seconds` 24 |
| `mi_release_info` | 560 | `struct_size` 0, `seconds` 8, `vram_released_bytes` 16, `host_copy_bytes` 24, `host_copy_seconds` 32, `dit_host_copy` 40, `te_host_copy` 44, `note[512]` 48 |
| `mi_dit_stream_stats` | 80 | `struct_size` 0, `streamed` 8, `resident_blocks` 12, `buffers` 16, `stalls` 20, `bytes` 24, `copy_seconds` 32, `stall_seconds` 40, `h2d_gbps` 48, `hidden_fraction` 56, `source[16]` 64 (no padding needed) |
| `mi_tier_info` | 40 | `struct_size` 0, `requested` 8, `effective` 12, `attention` 16, `certified` 20, `dit_precision` 24, `text_encoder_precision` 28; then `run_certified` 32, `_reserved0` 36 |
| `mi_prompt_tokens` | 32 | `struct_size` 0, `total_tokens` 8, `kept_tokens` 12, `prompt_tokens` 16, `dropped_tokens` 20, `max_length` 24, `truncated` 28 |
| `mi_lora_report` | 32 | `struct_size` 0, `adapters` 8, `side_branch` 12, `merged` 16, `unknown_keys` 20, `warnings` 24, `notes` 28 |
| `mi_image_info` | 80 | `struct_size` 0, `mode` 8, `hires_ran` 12, `first_width` 16, `first_height` 20, `first_steps_scheduled` 24, `first_start_step` 28, `first_steps_run` 32, `first_encoded` 36, `hires_width` 40, `hires_height` 44, `hires_steps_scheduled` 48, `hires_start_step` 52, `hires_steps_run` 56, `_reserved0` 60, `first_strength` 64, `hires_strength` 72 |
| `mi_control_info` | 48 | `struct_size` 0, `active` 8, `kinds_count` 12, `encodes` 16, `lora_adapters` 20, `lora_side_branch` 24, `lora_merged` 28, `lora_unknown_keys` 32, `lora_warnings` 36, `lora_notes` 40, `_reserved0` 44 |
| `mi_accel_info` | 32 | `struct_size` 0, `cfg_truncated_steps` 8, `dit_calls` 12, `cached_calls` 16, `lossy` 20; then `step_cache_threshold` 24 |
| `mi_file_info` | 56 | `struct_size` 0, `kind` 8, `loadable` 12, `file_bytes` 16, `tensor_count` 24, `precision` 28, `tiers` 32, `lora_rank` 36, `lora_adapters` 40, `lora_unknown_keys` 44, `control_kind_count` 48, `_reserved0` 52 |
| `mi_tensor` | 56 | `data` 0, `memory` 8, `dtype` 12, `rank` 16, `_reserved0` 20, `shape[4]` 24 (no `struct_size`: frozen as it is) |
| `mi_session_params` | 48 | `struct_size` 0, `plan_request` 8, `progress_callback` 16, `user_data` 24, `max_rows` 32, `max_latents` 36, `control_adapter` 40 |
| `mi_text_params` | 32 | `struct_size` 0, `prompts` 8, `prompts_count` 16, `max_sequence_length` 24, `_reserved0` 28 |
| `mi_conditioning_info` | 24 | `struct_size` 0, `tokens` 8, `text_dim` 12, `dtype` 16, `engine_encoded` 20 |
| `mi_run_row` | 32 | `conditioning` 0, `skip_layers` 8, `skip_layers_count` 16, `latent` 24, `_reserved0` 28 (array element, no `struct_size`: frozen as it is) |
| `mi_run_params` | 208 | `struct_size` 0, `width` 8, `height` 12, `rows` 16, `rows_count` 24, `sigmas` 32, `sigmas_count` 40, `scheduler` 48, `steps` 52, `shift` 56, `flow_multiplier` 64, `denoise_strength` 72, `guidance` 80, `output` 84, `flags` 88, `_reserved0` 92, `cfg` 96, `slg_layers` 104, `slg_layers_count` 112, `slg_scale` 120, `slg_start` 128, `slg_stop` 136, `loras` 144, `loras_count` 152, `control_lora_strength` 160, `control_images` 168, `control_images_count` 176, `cfg_truncation` 184, `step_cache` 192, `control_resize` 196, `step_cache_threshold` 200 |
| `mi_run_info` | 64 | `struct_size` 0, `latent_channels` 8, `latent_height` 12, `latent_width` 16, `steps` 20, `outputs` 24, `rows` 28, `schedule_matches` 32, `off_schedule_calls` 36, `calls` 40, `control_channels` 44, `latents` 48, `_reserved0` 52, `staging_bytes` 56 |
| `mi_step_params` | 592 | `struct_size` 0, `sigma` 8, `step` 12, `rows` 16, `flags` 20, `cuda_stream` 24, `control` 32, `x` 88, `out[8]` 144 |
| `mi_step_info` | 32 | `struct_size` 0, `step` 8, `rows` 12, `cached` 16, `on_schedule` 20, `seconds` 24 |

The 32-bit enums: `mi_status`, `mi_mode`, `mi_model_format`, `mi_block_path`, `mi_tier`, `mi_attention`, `mi_precision`, `mi_sampler`, `mi_scheduler`, `mi_noise_layout`, `mi_noise_source`, `mi_phase`, `mi_result_text_kind`, `mi_message_kind`, `mi_mask_encoding`, `mi_resize_mode`, `mi_image_mode`, `mi_step_cache`, `mi_direct_io`, `mi_te_host_pin`, `mi_dit_load_source`, `mi_load_io`, `mi_file_kind`, `mi_file_text_kind`, and since 2.1 `mi_memory`, `mi_dtype`, `mi_guidance`, `mi_step_output`.

History, all with `MI_ABI_VERSION` 1:
- 2026-10-21, during 0.1.0: the padding was made explicit; `mi_engine_params` grew `block_path` and `_reserved1` (72 → 80 bytes); `mi_dit_stream_stats` was added. Binaries built against an earlier header must be rebuilt.
- 0.2.0: `mi_engine_params` grew to 120 bytes and `mi_gen_params` to 240, as trailing fields. The 0.1.0 sizes, 80 and 168 bytes, stay accepted ([Struct sizes](#struct-sizes)). `mi_lora`, `mi_tier_info`, `mi_prompt_tokens` and `mi_lora_report` were added, and `mi_sampler` gained values 2–8.
- 0.2.0, image input: `steps` 0, a negative `cfg` and a negative `denoise_strength` became the unset markers, and `mi_gen_params_init()` at 320 bytes sets them. `mi_engine_params` grew to 136 bytes, `mi_gen_params` to 320 and `mi_timings` to 128, as trailing fields; the 120-, 240- and 104-byte layouts stay accepted. `mi_image` and `mi_image_info` were added.
- 0.2.0, system CUDA: `mi_probe_info` grew to 1624 bytes (the cuBLASLt fields), as trailing fields; the 576-byte layout stays accepted. `MI_CAP_SYSTEM_CUDA` was added.
- 1.0.0: first stable release. No layout or behaviour change from 0.2.0; the CMake package now requires the same MAJOR version (`SameMajorVersion`).
- 2.0.0, control adapters and accelerators: `mi_gen_params` grew to 376 bytes, as trailing fields; the 320-byte layout stays accepted (with its unset markers). `mi_control_image`, `mi_step_cache`, `mi_control_info` and `mi_accel_info` were added, `mi_result_text_kind` gained values 9–13, `mi_message_kind` values 2–3, and `MI_CAP_CONTROL` / `MI_CAP_ACCEL` were added. `mi_tier_info` grew to 40 bytes (`run_certified`, reserved: see the explicit-tiers entry), as trailing fields; the 32-byte layout stays accepted.
- 2.0.0, preload and soft unload: `mi_engine_params` grew to 160 bytes (`certificate_dirs` and its count, reserved; `dit_direct_io`, `te_host_pin`), `mi_timings` to 144 (`lora_load_s`, `lora_merge_s`), `mi_memory_stats` to 80 (`host_copy_bytes`, `merge_memo_bytes`, `dit_loads`) and `mi_accel_info` to 32 (`step_cache_threshold`), as trailing fields; the 136-, 128-, 56- and 24-byte layouts stay accepted. `mi_engine_preload()`, `mi_engine_release()`, `mi_release_info`, `mi_result_dit_load()`, `mi_dit_load_info`, `mi_result_accel_step_cache_diffs()` and the enums `mi_direct_io`, `mi_te_host_pin`, `mi_dit_load_source`, `mi_load_io` were added, `mi_result_text_kind` gained values 14–17, `mi_message_kind` value 4, and `MI_CAP_PRELOAD`, `MI_CAP_RELEASE`, `MI_CAP_TE_HOST_PIN` and the reserved `MI_CAP_CERTIFICATE_DIRS` were added. `step_cache_threshold` 0 now means the model's default (0.02 on the base model) instead of a fixed 0.03.
- 2.0.0, whole-job progress: `mi_progress` grew to 56 bytes (`fraction`, `eta_seconds`), as trailing fields; the library sets `struct_size`, so a client built against the 40-byte layout reads what it knows. `MI_CAP_PROGRESS_FRACTION` was added.
- 2.0.0, file inspection: `mi_inspect_file()`, the opaque `mi_file_report` with `mi_file_report_info()`, `_text()`, `_control_kind()`, `_tier_reason()` and `_free()`, `mi_file_info`, the enums `mi_file_kind` and `mi_file_text_kind`, and `MI_CAP_INSPECT` were added.
- 2.0.0, explicit tiers (a major version: callers must act, the layout is unchanged): `MI_TIER_AUTO`, the zero default of `mi_engine_params.tier`, is refused by `mi_engine_create()`. Quality certificates are gone from the engine; their fields stay in place, reserved: `certificate_dirs` / `certificate_dirs_count` are ignored, `MI_CAP_CERTIFICATE_DIRS` is never set, `mi_tier_info.certified` and `run_certified` are 0, `MI_TEXT_CERTIFICATE` is `""`. The report's `models.dit` has `tier` instead of `tier_requested`, `tier_effective` and `certified`, and its `quality` object no longer has `reason`, `certificate` or the models' content hashes. The SONAME stays `libmossinfer.so.1` (`MI_ABI_VERSION` 1): no symbol, field, offset or enum value moved, so a 1.x binary loads and fails only where it relied on `AUTO`, with a clear `MI_ERR_INVALID_ARGUMENT`. Migration: [integration.md](integration.md#migrating-from-1x).
- 2.1.0, the component API (additive): the structs `mi_tensor`, `mi_session_params`, `mi_text_params`, `mi_conditioning_info`, `mi_run_row`, `mi_run_params`, `mi_run_info`, `mi_step_params` and `mi_step_info`, the enums `mi_memory`, `mi_dtype`, `mi_guidance` and `mi_step_output`, the opaque `mi_conditioning`, 25 functions, `MI_CAP_COMPONENTS` (0x40000) and `MI_TEXT_COMPONENT_NOTE` (18) were added. `mi_engine_free()` now cancels and waits for a job running on another thread instead of leaving that race undefined.
- 2.x (the release after 2.1.0), reference images and LoRA placement (additive): `mi_gen_params` grew to 408 bytes (`reference_images`, `reference_images_count`, `reference_max_edge`, `_reserved5`, `lora_placement`, `_reserved6`), as trailing fields; the 376-byte layout stays accepted (no reference images, the default placement). `mi_reference_image` (24 bytes), `mi_lora_placement`, `MI_CAP_REFERENCE_IMAGES` (0x80000), `MI_TEXT_REFERENCES` (19) and `MI_MESSAGE_REFERENCE_WARNING` (5) were added.

## Symbol visibility

| Defines | Effect |
|---|---|
| `MOSSINFER_SHARED` | consuming the shared library: `dllimport` on Windows. The CMake package adds it. |
| `MOSSINFER_SHARED` + `MOSSINFER_BUILD` | building the shared library: `dllexport` |
| neither | static linking |

On ELF platforms `MOSSINFER_API` is always `visibility("default")`. The shared library compiles the core with hidden visibility and a version script (`mi_*` only). Besides the C API it exports one internal symbol, `mi_internal_cli_main`: release packages compile the `moss-infer` commands into the library, and this is the entry point of `bin/moss-infer`. It is internal to `moss-infer`, not part of the C API, and may change without notice. `check_package.py` checks that the export list is exactly these names.

## Implementation

| File | Role |
|---|---|
| `src/api/c_api.cpp` | the entry points, exception → status mapping, the engine's job gate (`runtime::JobGate`: one job at a time, the cancel flag, the component run's ending sequence) |
| `src/api/component_adapter.{h,cpp}` | the component API over `pipeline::Session`: the run, step and tensor conversions, the session calls on the engine's device, the run result as an `mi_result` |
| `src/api/pipeline_adapter.{h,cpp}` | the only code that touches `pipeline::Pipeline`: converts the C structs into `Request` / `PipelineOptions`, and flattens `pipeline::Result` into C-API types and the JSON report |
| `src/api/progress_model.{h,cpp}` | the whole-job progress (`mi_progress.fraction`, `eta_seconds`): the plan's phases with their time and load estimates, corrected from the measured phases and steps; pure host code |
| `src/api/file_inspect.{h,cpp}` | `mi_inspect_file()`: the weights, quant and models readers and `pipeline::inspect_dit_file()` in the pipeline's order, without the device; the tiers from `planner::resolve_tier()` |
| `src/pipeline/report_json.{h,cpp}` | the report renderer shared with `moss-infer generate --json-report`, so both documents are the same field for field |

When the `Pipeline` fields change, the adapter is the only code to update.

## Tests

Label `cpu` (`tests/unit/test_c_api.cpp`):
- **`CApi.*`** covers versions, ABI and capability bits, defaults against `pipeline::Request` / `PipelineOptions`, `struct_size` checks, the 0.1 sizes (`OldStructSizes`), reserved fields, NULL handles (every result accessor included), invalid generation and engine parameters (with messages, unknown `block_path` values included), the thread-local error, `mi_engine_free` leaving the error alone, and the probe. P6:
  - `EnumsMapToThePipeline`: every value of `mi_sampler`, `mi_scheduler`, `mi_noise_layout`, `mi_noise_source`, `mi_tier` and `mi_attention` maps to the pipeline value of the same name; unknown values are rejected.
  - `LoraAndDataspaceValidation`: a NULL array with a count, more than 64 entries, NULL or empty paths, non-finite or out-of-float-range strengths, missing files and directories; the dataspace sampler without its file or with a missing one; the new numbers.
  - `ComponentFiles`: missing or directory `dit_file`, `text_encoder_file` and `vae_file`, and unknown tier and attention values, before CUDA.
  - `GenerationFromAPipelineResult`: a synthetic pipeline result through the adapter: tier info, texts, prompt tokens, LoRA reports and messages, and the PNG metadata with and without the prompts, off, and after a cancel.
  - `WritePng`: the `mi_write_png()` round trip (size, no text chunks, the pipeline encoder's bytes) and its errors.
  - `Earlier020StructSizes`, `UnsetMarkers`: the 240- and 120-byte layouts; steps 0 and a negative cfg.
  - `ImageInput`: pixels copied and files decoded, both forms of one image, NULL pixels, channels, sizes over the limits, a nonzero `_reserved0`, missing and undecodable files, `MI_MASK_BINARY01` scaling and a value of 2, mask channels and size, strength and mask without an init image, the resize modes, the dataspace sampler with an image, `_reserved3`.
  - `HiresFix`: the mapping, multiples of 16, the size bound, strength and steps.
  - `DistillBundle`: a synthetic bundle (dataspace, LoRA and time-embedding files): the unset markers take its choices, explicit values stay, its LoRA is appended, its dataspace sampler refuses an image; not a directory (`INVALID_ARGUMENT`), no bundle files and a malformed dataspace file (`MODEL`); the engine side (`to_options`, `mi_engine_create`) with a missing, a matching and a conflicting `time_embedding_file`.
  - `ImageResultFromAPipelineResult`: `mi_image_info`, the image note, the time embedding, the new timings and an INT4 DiT as `W4A4` from a synthetic result.
  - `ControlAndAccelerators`: the mapping of the control adapter, its images (pixels copied, a file decoded as RGB, 1x1 stand-ins and no decode in validation) and the accelerators into `pipeline::Request`; every rejection (images without an adapter, a missing adapter or image file, a NULL array, more than 16, no kind, a repeated kind, both or neither of image and path, channels, sizes, strength and window ranges, the LoRA strength, `cfg_truncation`, unknown `step_cache` values, the threshold rules, `_reserved4`).
  - `Gen100StructSize`: the 320-byte layout (its unset markers, the tail neither written nor read); the new fields' defaults.
  - `TierInfoReservedFields`: the tier info from synthetic results (requested = effective; the reserved `certified` and `run_certified` written as 0; the report's `models.dit.tier`, no `certified`; `MI_TEXT_CERTIFICATE` `""`), the 32-byte `mi_tier_info` and a wrong size.
  - `ControlAndAcceleratorResults`: `mi_result_control()`, `mi_result_accel()`, `mi_result_accel_cached_steps()`, the control texts and messages from a synthetic result (and without either), the zeroed padding and wrong sizes.
  - `EngineParams110Fields`: `certificate_dirs` ignored (a count without an array is not read), `dit_direct_io` (`AUTO` → unset, `BUFFERED` → false, `DIRECT` → true, next to `direct_io`) and `te_host_pin` into `PipelineOptions`; unknown enum values rejected by `mi_engine_create()` before CUDA; the 136-byte layout (the tail neither written nor read).
  - `GrownOutputStructs110`: the new `mi_timings`, `mi_memory_stats` and `mi_accel_info` fields from a synthetic result (an unset threshold reported as the model's default that ran, and as `default` when cancelled before it), the 128-, 56- and 24-byte sizes getting exactly their bytes, wrong sizes.
  - `DitLoadNotesAndDiffs`: `mi_result_dit_load()` for every source and I/O mode, the four new texts, `MI_MESSAGE_ACCEL_WARNING`, `mi_result_accel_step_cache_diffs()` (NaN kept), NULL and wrong-size cases.
  - `PreloadAndReleaseWithoutAnEngine`: preload's request with a NULL or blank prompt (validation still requires one), the other checks kept, and both functions on a NULL engine. `ReleaseInfo`: `mi_release_info` from a release report, the note cut on a UTF-8 boundary.
  - `ControlAndAccelerators` and `DefaultsMirrorThePipeline` check that `step_cache_threshold` 0 maps to an unset threshold.
- **`ProgressModel.*`** (`tests/unit/test_progress_model.cpp`) covers the whole-job progress (`src/api/progress_model.h`) without a GPU: the fraction in range and never decreasing (also when a slow first step would lower it), the eta shrinking as steps run, a measured phase replacing its estimate, the steps after the first setting the price, an unknown eta (before the plan, a non-finite estimate), skipped and unexpected phases (a preload's unexpected `DECODE` does not pass the pending DiT load), the cold-load estimates (cold disk, measured direct I/O, a warm page cache, a soft unload's host copy, a warm engine, another residency, parking, the text encoder's copies, the VAE encoder), the warm / cold record (jobs, cancels with and without a new arena or with another DiT, jobs that do not use the DiT, other parking, an exception, a LoRA change, releases, a preload then a generation, a VRAM-resident text encoder after a preload and a cache hit), the hires fix's passes priced by pixel share, and a preload's load-only sequence.
- **`CApiInspect.*`** (`tests/unit/test_c_api_inspect.cpp`) covers `mi_inspect_file()` on files written by the test: bf16, INT8 ConvRot and INT4 ConvRot DiTs (precision, quantization, the tier bits and reasons) and one that needs decoding; DiT headers with a GEMM or a scale of another shape (`HostileDitShapes`: refused before any scale is read); a fully NVFP4 DiT (`Nvfp4Dit`: its NVFP4 verdict or, with the NVFP4 kernels in the build, its scale validation, not a size refusal); an NVFP4 linear with a static input_scale (`DitLinearsText`: the quantization text and the note's remark, as indexed with no NVFP4 refusal); a LoRA with metadata and an unknown key, one for another model and a malformed one (both LoRAs, not loadable); control adapters with and without `control_channels`, without a LoRA part, with kinds that do not match and with a NaN in the embedder; a text encoder, and one with a 180 MiB F32 tensor (`TextEncoderTensorShapes`: refused by the importer's shape check before any read; skipped where the test folder has no sparse files); a narrowed VAE decoder; garbage, empty, truncated, oversized, deeply nested and lying headers, GGUF and pickle files, a file with `.alpha` parameters but no LoRA factors (a `LORA` the reader refuses); a FIFO (POSIX); NULL and empty paths, a missing file, a folder without search permission, a directory, `/dev/zero`, and every accessor's NULL, range and size checks. `CApi.GenerationFromAPipelineResult` checks the text encoder precision of every `qwen35_text::precision_name()`. `CommunityFiles` inspects the community files under `$MOSS_COMMUNITY_MODELS` (default `/workspace/Models/community`) when present, and skips otherwise.
- **`c_abi_compat`** (`tests/c/test_abi_compat.c`, C11): a hand-written 0.1.0-layout `mi_engine_params` and `mi_gen_params` through `*_init`, `mi_gen_params_validate()` and `mi_engine_create()` (as far as the model path check). With weights, `c_abi_compat_gpu` (label `gpu`) also generates from them and checks that the PNG has no text chunks.
- **`ComponentApi.*`** (`tests/unit/test_component_api.cpp`) covers the component API without a GPU: init defaults and sizes, calls without an engine, tensor descriptors, foreign conditioning (bf16 rounding to nearest even), the run spec's validation with its messages (rows, latents, guidance modes, the schedule, the step cache, control images, the shared `mi_gen_params` rules, the token cap), step arguments, `mi_schedule()` against the pipeline's schedule over every scheduler, steps, shift, multiplier and strength, `mi_sampler_prediction()`, `mi_initial_noise()`, `mi_image_resize()`, and the run result's mapping (the note, the report's `component` object). **`c_component_api`** (`tests/c/test_component_api.c`, C99) links every component symbol and checks the init sizes, the helpers and calls without an engine; `c_abi_layout` checks the component layouts too, and `c_abi_compat` that every 2.0 capability bit stays set.
- **`CApiReport.*`** renders the report (through the adapter and the shared renderer) from a synthetic result under a comma-decimal C++ locale with digit grouping, and under a comma-decimal C locale where the system has one. It checks for valid JSON, exact numbers, `null` for NaN and infinity, the `dit_blocks` and `dit_stream` keys, `quality.cublaslt`, and that the document is the CLI's fields in braces.
- **`PipelineReport.*`** (`tests/unit/test_pipeline_report.cpp`) covers the shared renderer itself: every key, the head/tail split around the image path, U+FFFD for ill-formed UTF-8, `null` for non-finite numbers, and the classic locale.
- **`CudaLibs.*`** (`tests/unit/test_backend_cuda_libs.cpp`) covers the cuBLASLt search: `MOSSINFER_CUDA_LIBS` parsing, the compatibility rule, the candidate lists of each policy on Linux and Windows (injected environment and module directory), `MOSSINFER_CUBLASLT` as the only candidate, the error listing for a missing file and for a library without cuBLASLt's symbols, the system search on this machine, and the shim forwarding to the process's copy. `CApi.ProbeWithoutRequiringAGpu` checks the probe's cuBLASLt fields and the 576-byte size.

Label `gpu`:
- **`GpuCApiProbe.*`**: after `mi_probe_device()`, the process holds no CUDA context. This is checked through the driver's primary-context state and NVML's process list.
- **`GpuCApi.*`** covers the device probe, and a 256², 2-step generation with callbacks (the 56-byte `mi_progress`: `fraction` in [0, 1) and never decreasing, `eta_seconds` known and not negative; from `TEXT_ENCODER` to the final `DECODE` the fraction grows past one half and the eta shrinks), a concurrent call rejected as busy, the block path accessors and `mi_result_dit_stream()` (nothing streamed in balanced mode on a 16 GiB card), and PNG and report checks. It also covers cancel from another thread with no carry-over, and the caller's current context restored by create, generate and free. `SamplerPreviewMetadataAndTier` runs a 256², 3-step `DPMPP_2M` + `KARRAS` generation with `preview_interval` 2 (previews after steps 2 and 3, at the image size), checks the `generation` chunk of the saved PNG, `mi_result_tier()` (tier `REFERENCE` on the bf16 bundle runs exact attention; the reserved fields are 0), the texts, the prompt tokens, and the LoRA and message accessors' bounds. `OldSizeWritesNoMetadata` generates from a 168-byte `mi_gen_params` and checks that the PNG has no text chunks. `ImageToImageFromPixels` (256², 4 steps at strength 0.5: 2 run from step 2, `total_steps` 2, a 104-byte `mi_timings` gets 104 bytes), `InpaintBinary01Mask` (the kept half stays closer to the init image than the regenerated half), `HiresFix` (256 → 384, 2 + 2 steps) and `DistillTimeEmbeddingNeedsTheEngine` cover image input and the engine check. `ControlAdapterDepth` runs the color/depth control adapter (`$MOSS_COMMUNITY_MODELS/comfy-org/moss_image_color_depth_control.safetensors`, skipped when absent) with a generated grayscale depth image from pixels, 256², 3 steps, and checks `mi_result_control()` and the texts, then a kind the adapter lacks (`MI_ERR_INVALID_ARGUMENT` from `mi_generate()`, not from validation). `Accelerators` runs CFG truncation 0.5 with the step cache at threshold 0.5, 256², 6 steps, and checks `mi_result_accel()`, the cached steps and the diffs; then with the threshold unset, which runs the base model's default (0.02, source `default for the base model`). **`GpuCApiPreload.*`** create engines of their own, one at a time, at 256², 2 steps: `PreloadThenGenerate` (the plan before the first load step and the phases in order, with `fraction` and `eta_seconds` as for a generation; the generation after the preload prices no DiT load, a fresh engine's does; the generation after a preload loads no DiT and matches a fresh engine's image bit for bit), `ReleaseThenGenerate` (a blank-prompt preload; `release(1)`: the next load reads the host copy and gives the same bits; `release(0)`: the files again; release and preload are `MI_ERR_BUSY` during a generation) `CancelDuringPreload` (`MI_CANCELLED` from the first load step's callback, which stops it there, and from another thread; the engine stays usable) and `ReleaseCancelledBeforeItStarts` (through the adapter, a release with its cancel set on entry releases nothing and the next generation loads no DiT, with the same bits; then `mi_engine_release()` against a racing `mi_cancel()`: `MI_CANCELLED` with nothing released when the cancel lands first). `CApi.ReleaseInfo` covers the mapping of the early cancel on the CPU. **`GpuCApiModels.*`** create their own engine on the SFT bundle when the files exist under `/workspace/Models/community`: `DistillBundle` (the lodestones SFT 8-step bundle on the engine and the request: 8 data-space steps at CFG 1, its LoRA and time embedding) and `TurboInt4` (an INT4 DiT: tier `HIGH` refused with the reason, `TURBO` runs it as `W4A4` with 8-bit attention). Runtime LoRA files of one's own on the GPU are covered only by the distill bundle's LoRA; the repository has no small LoRA file.
- **`GpuComponentApi.*`** (SFT bundle, INT8 DiT, `HIGH`, 512² and 1024²): an L = 1 MOSS run through the C API, sampled by the euler-on-moss host loop in an adopted staging block and decoded with `mi_vae_decode()`, has `mi_generate()`'s plan and RGB bytes; the state errors on the way (a second session or run, `mi_generate()` and `mi_encode_text()` during a run, end and close from a callback); `mi_run_end()` from another thread mid-step (`MI_CANCELLED`, then a new run that loads nothing); `mi_cancel()` on a step (the run stays usable, no carry-over); `mi_engine_free()` during a step, from a callback (a no-op) and during a generation with a session open (cancelled at once); the adopted block's lifecycle (unregistered by `mi_run_end` and by a close without it, adopted again by a later run, a caller-registered block left registered); the step refusals on device tensors. **`SessionSlot.*`** (CPU, also under TSan and ASan) covers the session object's refs, replacement and teardown.

Run the GPU tests on a machine with a supported GPU:

```
GTEST_FILTER='GpuCApi.*' <build>/mossinfer_tests
```
