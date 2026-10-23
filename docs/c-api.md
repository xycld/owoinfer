# MossInfer C API

`include/mossinfer/mossinfer.h` is the stable interface to the engine. It is plain C99, in the style of `llama.h`. The library is `libmossinfer.so` or `mossinfer.dll`, and CMake consumers use `find_package(MossInfer)` → `MossInfer::mossinfer` (see `cmake/modules/packaging.cmake`). `moss-server` (`docs/server-protocol.md`) and `examples/c/minimal.c` use only this header.

## Quick start

```c
#include "mossinfer/mossinfer.h"

mi_engine_params ep;
mi_engine_params_init(&ep, sizeof ep);          /* defaults; struct_size is set for you */
ep.model_path = "/models/MOSS-Image/base";      /* Hugging Face bundle directory, UTF-8 */
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
  - Output structs (`mi_timings`, `mi_memory_stats`, `mi_dit_stream_stats`, `mi_probe_info`, `mi_device_info`, `mi_tier_info`, `mi_prompt_tokens`, `mi_lora_report`) need `struct_size` set by the caller before the call.
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
| `MI_CANCELLED` (1) | `mi_generate()` stopped by `mi_cancel()`. A result is still returned: no image, but timings and the report. |
| `MI_ERR_INVALID_ARGUMENT` | NULL handle or pointer, value out of range, invalid request |
| `MI_ERR_ABI_MISMATCH` | unknown `struct_size` |
| `MI_ERR_NOT_SUPPORTED` | not in this build, e.g. a ComfyUI file as `model_path` (`MI_MODEL_FORMAT_COMFYUI`) |
| `MI_ERR_BUSY` | the engine is already generating |
| `MI_ERR_OUT_OF_MEMORY` | the request fits the device or host budgets in no mode (planner), or a host allocation failed |
| `MI_ERR_MODEL` | model files missing, incomplete or malformed, or not a supported MOSS-Image model ([integration.md](integration.md#supported-models)); a LoRA file that cannot be applied (with `lora_strict`, also unknown keys and lossy merges) |
| `MI_ERR_IO` | file-system error: reading weights, writing the PNG |
| `MI_ERR_CUDA` | CUDA error or no usable device. Free the engine afterwards. Also `mi_engine_create()` when no usable cuBLASLt is found: the message lists every candidate tried and why it was rejected (see [cuBLASLt](#cublaslt)). |
| `MI_ERR_INTERNAL` | anything else |

## Threading model

- **One generation at a time per engine.** A concurrent `mi_generate()` on the same engine returns `MI_ERR_BUSY`. Separate engines may run concurrently, but they share the GPU and each one plans against its own budgets.
- **`mi_cancel(engine)` is thread-safe** and may be called from any thread, including from a callback.
  - The generation stops within one text-encoder layer, DiT block or decode band, and `mi_generate()` returns `MI_CANCELLED`.
  - A cancel only affects the generation running at that moment. It is a no-op when the engine is idle, and it never carries over to the next generation.
- **Callbacks** (`progress_callback`, `plan_callback`, `preview_callback`) run synchronously on the thread that called `mi_generate()`, so keep them short.
  - They must not throw or longjmp.
  - They must not call `mi_generate()` or `mi_engine_free()` on the same engine.
- **`mi_engine_free()` must not race** with a running `mi_generate()` on that engine. Cancel first, then wait for the call to return.
- **`mi_result` is independent of its engine.** It may outlive the engine, and it may be used from any thread, one at a time.
- **The CUDA device is selected per call.** `mi_generate()` selects `device_index` on the calling thread, so engines may be driven from any thread.
  - `mi_engine_create()`, `mi_generate()` and `mi_engine_free()` restore the calling thread's current CUDA context before they return: none if the thread had none, else the caller's own.
  - They use the driver API (`cuCtxGetCurrent` / `cuCtxSetCurrent`, loaded at run time). Without it, they fall back to restoring the runtime's current device.

## Engine

`mi_engine_create()` does three things:
1. Checks the parameters, the model path (the four bundle folders `text_encoder`, `tokenizer`, `transformer` and `vae`), and that each given component file exists and is a regular file. This happens before CUDA is touched. The model folder's configs, the architecture of the DiT and text-encoder files and the component files' format are checked when they load, at the first `mi_generate()`, which then fails with `MI_ERR_MODEL`. A folder or file of another model fails the same way, with a message naming what was found ([supported models](integration.md#supported-models)).
2. Selects the device.
3. Creates the pipeline.

The weights load on the first `mi_generate()`. The DiT stays resident across generations when the plan allows it, so the first call pays the load (several seconds, depending on storage and the page cache) and later ones do not.

| `mi_engine_params` field | Default | Meaning |
|---|---|---|
| `model_path` | — (required) | bundle directory. The tokenizer and configs always come from it. |
| `model_format` | `AUTO` | `AUTO`: a directory is an HF bundle, a file is a ComfyUI checkpoint (not supported: `MI_ERR_NOT_SUPPORTED`). `COMFYUI` is reserved and not supported either. |
| `mode` | `BALANCED` | `MIN_MEMORY` / `BALANCED` / `MAX_SPEED` (planner objective) |
| `device_index` | 0 | CUDA ordinal |
| `vram_budget_bytes` | 0 = free memory − reserve | cap on the process's device memory, CUDA context included |
| `vram_reserve_bytes` | 384 MiB | device memory left outside the arena; 0 = none |
| `host_ram_budget_bytes` | 0 = the mode's target | cap on private host memory |
| `cache_dir` | NULL = memory only | embedding-cache disk spill |
| `embedding_cache` | 1 | reuse the text embeddings of repeated prompts |
| `direct_io` | 0 | 1: O_DIRECT for every weight read (cold-cache behaviour). 0: the DiT load picks direct or buffered reads itself (direct where the file system supports it and the files are not already in the page cache; the report's `dit_load` says which and why) |
| `block_path` | `DEFAULT` (fused) | DiT block implementation: `OP_BY_OP`, `FUSED` or `FUSED_GRAPHS`; all give the same image bits. `FUSED` runs op by op when the startup self-check finds the fused GEMMs not bit-identical on this GPU, and `FUSED_GRAPHS` replays CUDA graphs only when every DiT block is resident. `mi_result_block_path()` reports what ran. |
| `dit_file` | NULL = the bundle's `transformer/` | single-file DiT: INT8 ConvRot (G = 256), INT4 ConvRot (the `TURBO` tier), mixed INT8/bf16 or bf16 safetensors in the ComfyUI layout (Comfy-Org and community files, `moss-infer quantize` output) |
| `text_encoder_file` | NULL = the bundle's `text_encoder/` | INT8 ConvRot or W4A8 Qwen3.5 text encoder. Its embeddings are cached under its own identity. |
| `vae_file` | NULL = the bundle's `vae/` | single FLUX.2 VAE file, decoded with the bundle's latent statistics |
| `tier` | `AUTO` | quality tier, see below |
| `attention` | `DEFAULT` (the tier's choice) | DiT attention kernel: `EXACT` (FlashAttention-2) or an 8-bit mode |
| `lora_strict` | 0 | 1: unknown LoRA keys and lossy merges fail the generation with `MI_ERR_MODEL` instead of being reported |
| `time_embedding_file` | NULL = none | a step-embedder replacement (`time_embedding.safetensors`: the step embedder and modulation biases, computed in fp32), applied to the loaded DiT |
| `distill_dir` | NULL = none | a distill bundle directory. The engine takes its time embedding (as `time_embedding_file`); `mi_gen_params.distill_dir` takes the rest per generation (see [Distill bundles](#distill-bundles)). |

The three sizes (`vram_budget_bytes`, `vram_reserve_bytes`, `host_ram_budget_bytes`) share one range rule: 0, or a value in [1 MiB, 1 TiB]. Anything else fails `mi_engine_create()` with `MI_ERR_INVALID_ARGUMENT`, naming the field.

The time embedding is per engine because it replaces part of the loaded DiT. Setting both `time_embedding_file` and a `distill_dir` whose bundle has a different time embedding is `MI_ERR_INVALID_ARGUMENT`. A `distill_dir` that is not a directory is `MI_ERR_INVALID_ARGUMENT`; a directory with none of the bundle files is `MI_ERR_MODEL`. A malformed time embedding file fails `mi_engine_create()` with `MI_ERR_MODEL`: it is read when the pipeline is constructed.

The three file fields are UTF-8 paths; NULL or `""` means the bundle's component. A path that does not exist, or names a directory, fails `mi_engine_create()` with `MI_ERR_INVALID_ARGUMENT`, naming the field. ComfyUI and quantized files are supported as component files only: `model_path` stays a bundle directory (`MI_CAP_MODEL_FILES`, not `MI_CAP_COMFYUI_MODELS`).

### Tier and attention

The loaded weights fix the GEMM precision: an INT8 DiT runs INT8 GEMMs. The tier decides the attention kernel and refuses conflicting combinations. A refused combination fails `mi_generate()` with `MI_ERR_INVALID_ARGUMENT`, and the message names the conflict.

| `mi_tier` | Runs |
|---|---|
| `AUTO` (0) | with `attention` set: that configuration. Otherwise the fastest lossy configuration of the loaded DiT that has a quality certificate. Without one: Reference for a bf16 DiT and text encoder; the loaded GEMMs with exact attention, uncertified, for an INT8 DiT or a quantized text encoder. An INT4 DiT without a certificate is refused (`MI_ERR_INVALID_ARGUMENT`, with the pipeline's message): choose `TURBO` explicitly. |
| `REFERENCE` (1) | bf16 DiT and text encoder, exact attention. Refused for quantized files and for an 8-bit `attention`. |
| `HIGH` (2) | the loaded DiT's precision plus 8-bit attention (`Q8_FP8` unless `attention` says otherwise), certified or not. Refused for a bf16 DiT with exact attention. |
| `TURBO` (3) | an INT4 DiT file (`dit_file`): INT4 GEMMs plus 8-bit attention, as for `HIGH`. Runs certified or not (reported uncertified without a certificate). Refused for any other DiT. |

1.0 ships no quality certificates: without one of your own, `AUTO` runs Reference for the bf16 bundle and refuses an INT4 file. Offer `TURBO` for INT4 DiT files only, as an explicit choice.

| `mi_attention` | Pipeline name |
|---|---|
| `DEFAULT` (0) | unset: the tier's choice |
| `EXACT` (1) | `exact` |
| `Q8_FP16` (2) | `q8_fp16` |
| `Q8_FP16_F32ACC` (3) | `q8_fp16_f32acc` |
| `Q8_FP8` (4) | `q8_fp8` |
| `Q8_FP8_F32ACC` (5) | `q8_fp8_f32acc` |

Certificates are searched in this order, and the first usable one wins: `<cache_dir>/quality_certs/`, the DiT file's sidecar (`<dit>.mossinfer-cert.json`), then the install tree's `<prefix>/share/mossinfer/quality_certs/` (found from the library's own location when it sits in the prefix's `lib/` or `bin/`). The engine also takes extra directories, searched first; they are not yet exposed in this API. A generation with LoRAs is uncertified. `mi_result_tier()` reports what ran.

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

`mi_gen_params_validate()` performs the same checks as `mi_generate()` without an engine or GPU: the LoRA files' existence, the distill bundle (opened) and the dataspace file (read: its step codes bound `steps`), and the image inputs without copying or decoding them. Pixel images get their sizes and channels checked, and a `MI_MASK_BINARY01` mask its values. Image files get checked for existence, for being regular files and for size (at most 64 MiB); they are decoded by `mi_generate()`, which reports a file it cannot decode, or a binary01 file mask with other values, as `MI_ERR_INVALID_ARGUMENT`. The match of a bundle's time embedding with the engine's is also left to `mi_generate()`. A launcher can call it as the user types.

### Unset markers

`mi_gen_params_init()` at the current size (320 bytes) leaves the fields whose default depends on the rest of the request unset. Each resolves when the request is checked:

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

**Untrusted images.** The image file decoder (`init_image_path`, `mask_path`) is the vendored stb_image. It is hardened (the file-size cap, a header check before any allocation, fuzzing) but it is not memory-safe. A program that takes images from untrusted parties (a network, web uploads, other users' files) must decode them itself with a memory-safe decoder and pass pixels in `init_image` / `mask`, or decode in a sandboxed process.

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

The files are safetensors in the training tools' formats: kohya and PEFT / diffusers low-rank adapters, LyCORIS LoHa and LoKr, and full deltas (`.diff`, `.diff_b`). Low-rank adapters on the block GEMMs run as side branches, so a strength change between generations is instant. The others are merged into the DiT when it loads. A file with no applicable key fails the generation with `MI_ERR_MODEL`. Unknown keys and lossy merges (adapters requantized on INT8 GEMMs) are reported as warnings, or fail with `MI_ERR_MODEL` when `lora_strict` is set. `mi_result_lora()` and `mi_result_message()` report them.

### Callbacks

- **progress** — `mi_progress {phase, step, total_steps, step_seconds, elapsed_seconds}`. Called at each phase start (`TEXT_ENCODER`, `DIT_LOAD`, `DENOISE`, `DECODE`, in that order; the first two are entered even on a cache hit or a resident DiT) and after every sampling step. `total_steps` is the schedule's step count, which can be below `steps` (the `MOSS` and `BETA` schedules drop repeated sigmas).
- **plan** — called once per generation, after planning and before the first phase. It receives the plan as human-readable lines and as JSON.
- **preview** — called after every `preview_interval`-th sampling step and after the last one, before that step's progress callback, with `(rgb, width, height, step, user_data)`.
  - `rgb` is RGB8 `[height][width][3]` at the requested image size (`width` × `height` of the request), valid during the call only.
  - It is a linear latent → RGB map of the x0 estimate after that step, without the VAE: a coarse preview, not the final image.
  - Setting the callback adds the preview's buffers to the denoise phase's plan. Leave it NULL when no preview is shown.
  - Each preview is a full-size device-to-host copy, synchronous on the generating thread, plus whatever the callback does (e.g. a PNG write). For large images use a `preview_interval` of about 5.
  - `mi_write_png()` writes a preview to a file.

### PNG metadata

With `save_metadata` set (the default of 240- and 320-byte structs), `mi_save_png()` writes two text chunks after `IHDR`, the same ones as `moss-infer generate`:

| Keyword | Chunk | Content |
|---|---|---|
| `generation` | iTXt | one JSON object, schema `mossinfer.generation/1`: engine version and tier, models by file name, LoRAs and strengths, prompt and negative prompt, sampler, scheduler, steps, CFG, shift, seed, size, and extra settings (attention, DiT linears, noise layout, noise source) |
| `parameters` | tEXt (iTXt when not ASCII) | the plain-text settings block that image tools and galleries read |

With `include_prompts` 0, the prompt and negative prompt are left out of both chunks. Model and LoRA names are reduced to their base names, so no directory reaches the file. `save_metadata` 0 writes a PNG without text chunks.

## Results

| Accessor | Returns |
|---|---|
| `mi_result_rgb(r, &bytes)` | row-major RGB8 `[height][width][3]`, or NULL when cancelled |
| `mi_result_width/height/seed/steps_run/steps_scheduled/cancelled` | scalars |
| `mi_result_timings(r, &t)` | `total, load, encode, denoise, decode` and their parts, in seconds; then `vae_encoder_load_s`, `image_encode_s` (every VAE encode) and `image_ops_s` (resizes, the hires upscale, mask preparation, inpainting blends), which are not part of `load_s`. A 104-byte (0.1.0) `mi_timings` is accepted and gets its 104 bytes. |
| `mi_result_image(r, &i)` | `mi_image_info`: the mode (`TXT2IMG`, `IMG2IMG`, `INPAINT`), and per pass (the first, the hires fix's) the size, steps scheduled, the start step (skipped by the strength), steps run and the strength; `first_encoded`; `hires_ran`. The hires fields are 0 without the hires fix. |
| `mi_result_memory(r, &m)` | arena capacity and high-water, device peak, host private start and peak |
| `mi_result_dit_stream(r, &s)` | DiT block streaming over the denoise: streamed, resident blocks, buffers, stalls, bytes copied, copy and stall seconds, GB/s, hidden fraction, source (`"file"` / `"host_pinned"`). All zero when every block was resident. |
| `mi_result_block_path(r)` / `mi_result_block_path_note(r)` | the DiT block path that ran (`OP_BY_OP`, `FUSED` or `FUSED_GRAPHS`), and why it differs from the requested one (`""` when it does not). A cancel that landed before the denoise (text encoder or DiT load) ran no block; the path is then `FUSED`, the engine default, whatever was requested. |
| `mi_result_plan_explanation(r)` | the execution plan as text |
| `mi_result_report_json(r)` | the `moss-infer generate --json-report` document; `"image"` is the last path given to `mi_save_png()`. NULL, with the message set, for a NULL result or when rendering fails (out of memory). |
| `mi_save_png(r, path)` | atomic PNG write (temporary file, then rename), with the [metadata](#png-metadata) when `save_metadata` was set |
| `mi_result_tier(r, &t)` | `mi_tier_info`: requested and effective tier, the attention that ran, certified, DiT and text-encoder precision (below) |
| `mi_result_text(r, kind)` | a string of the result (below); `""` when empty, NULL for a NULL result or an unknown kind |
| `mi_result_prompt_tokens(r, negative, &t)` | `mi_prompt_tokens` of the prompt (`negative` 0) or the negative prompt (1; all zero when CFG is off): `total_tokens` (chat template included), `kept_tokens`, `prompt_tokens` (the text's own), `dropped_tokens`, `max_length`, `truncated`. Any other `negative` is `MI_ERR_INVALID_ARGUMENT`. |
| `mi_result_lora_count(r)`, `mi_result_lora(r, i, &l)`, `mi_result_lora_file(r, i)` | per LoRA, in request order: `mi_lora_report {adapters, side_branch, merged, unknown_keys, warnings, notes}` and the file. An index out of range is `MI_ERR_INVALID_ARGUMENT` / NULL. |
| `mi_result_message_count(r, kind)`, `mi_result_message(r, kind, i)` | `MI_MESSAGE_LORA_WARNING`: losses to show the user (`"<module>: <reason>"`). `MI_MESSAGE_LORA_NOTE`: how the files were read. 0 / NULL for an unknown kind or index. |

| `mi_result_text_kind` | Text |
|---|---|
| `MI_TEXT_TIER_REASON` (0) | what the tier chose and why |
| `MI_TEXT_CERTIFICATE` (1) | the certificate file used; `""` for none |
| `MI_TEXT_GPU_ARCH` (2) | the GPU class certificates are matched on, e.g. `sm_89` |
| `MI_TEXT_DIT_SOURCE` (3) | the bundle's `transformer` folder, or `dit_file` |
| `MI_TEXT_TEXT_ENCODER_SOURCE` (4) | the bundle's `text_encoder` folder, or `text_encoder_file` |
| `MI_TEXT_VAE_SOURCE` (5) | the bundle's `vae` folder, or `vae_file` |
| `MI_TEXT_QUALITY_JSON` (6) | the report's `quality` object, as JSON |
| `MI_TEXT_IMAGE_NOTE` (7) | a remark on the image input, e.g. why an init image was not used; `""` for none |
| `MI_TEXT_TIME_EMBEDDING` (8) | the replacement step embedder that ran; `""` for none |

`mi_tier_info.effective` is never `AUTO`, and `attention` is never `DEFAULT`. `certified` is 1 whenever `effective` is `REFERENCE`, which needs no certificate. A generation cancelled before its tier was resolved reports requested `AUTO`, effective `REFERENCE`, attention `EXACT`, certified 1. The precisions map the pipeline's report as follows:

| Pipeline | `mi_precision` |
|---|---|
| DiT `bf16` (the bundle, or a bf16 file) | `BF16` (0) |
| DiT `int8` (any INT8 linear; mixed files count as INT8) | `INT8` (1) |
| text encoder `bf16` (the bundle's) | `BF16` (0) |
| text encoder `int8` | `INT8` (1) |
| text encoder `w4a8` | `W4A8` (2) |
| DiT `int4` (an INT4 ConvRot file, the Turbo tier) | `W4A4` (3) |

Strings returned by accessors are owned by the result. They stay valid until `mi_result_free()`; the report string stays valid until the next `mi_result_report_json()` or `mi_save_png()` call on the same result.

`mi_write_png(path, rgb, width, height)` writes RGB8 `[height][width][3]` as a PNG without metadata, atomically. It needs no result and is meant for previews. A NULL or empty path, NULL `rgb`, or a width or height outside [1, 16384] is `MI_ERR_INVALID_ARGUMENT`; a write failure is `MI_ERR_IO`.

The JSON report carries all of this as well: `quality`, `models` (sources, linears, tiers, text-encoder precision, time embedding), `loras`, `lora_warnings`, `lora_notes`, `tokens` and `img2img` (mode, note, VAE encoder, both passes). Its `control` (`null`) and `accelerators` (inactive) objects are set only by `moss-infer` in 1.0: the C API gets control LoRAs and the accelerators in 1.1.

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

`MI_CAP_TIERS` covers `TURBO` as well.

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
| `mi_engine_params` | 80 (the 0.1.0 size, up to `_reserved1`), 120 (an earlier 0.2.0 header, up to `_reserved2`), 136 (0.2.0) |
| `mi_gen_params` | 168 (the 0.1.0 size, up to `user_data`), 240 (an earlier 0.2.0 header, up to `_reserved2`), 320 (0.2.0) |
| `mi_timings` (output) | 104 (the 0.1.0 size, up to `vae_load_s`), 128 (0.2.0) |
| `mi_probe_info` (output) | 576 (the first 0.2.0 size, up to `_reserved0`), 1624 (0.2.0 with the cuBLASLt fields) |
| every other output struct | its one size so far (table below) |

Any other size is `MI_ERR_ABI_MISMATCH`, and the message lists the accepted sizes.
- **`*_init(p, size)`** writes only the first `size` bytes: the defaults of the fields that layout has. The bytes after them are left alone.
- **Input structs** (`mi_engine_create()`, `mi_gen_params_validate()`, `mi_generate()`): the library reads only `struct_size` bytes. It copies them over a full struct filled with the defaults for that size, so every later field keeps the behaviour of the older release:
  - an 80-byte `mi_engine_params` runs with the bundle's components, tier `AUTO` and default attention;
  - a 168-byte `mi_gen_params` runs with the `MOSS` scheduler, the default noise and no LoRAs, and its `save_metadata` and `include_prompts` are 0: `mi_save_png()` writes a PNG without text chunks, as 0.1.0 did. A 240- or 320-byte struct defaults `save_metadata` and `include_prompts` to 1;
  - a 120-byte `mi_engine_params` or a 240-byte `mi_gen_params` runs without image input, hires fix or distill bundle.
- **Output structs**: the library writes only `struct_size` bytes.

`CApi.OldStructSizes` and `CApi.Earlier020StructSizes` (`tests/unit/test_c_api.cpp`) and the C test `c_abi_compat` (`tests/c/test_abi_compat.c`, a hand-written 0.1.0-layout struct, label `cpu`; `c_abi_compat_gpu`, label `gpu`, also generates from it) check this.

## Frozen layout (ABI 1)

These are the layouts on 64-bit targets (LP64 and LLP64; 32-bit is not supported). Every field sits at its natural alignment, and every gap is a named `_reserved*` field, so GCC, Clang and MSVC produce the same layout from C and from C++.
- `src/api/c_api.cpp` `static_assert`s each size and offset.
- The C test `c_abi_layout` (`tests/c/test_abi_layout.c`, label `cpu`) checks the same table from a C11 compiler.

The enums are 4 bytes. ABI 1 is defined by this table: a change to an existing entry needs `MI_ABI_VERSION` 2. New trailing fields, structs and enum values do not.

| Struct | Size | Fields: offset |
|---|---|---|
| `mi_device_info` | 592 | `struct_size` 0, `index` 8, `name[256]` 12, `pci_bus_id[32]` 268, `compute_major` 300, `compute_minor` 304, `_reserved0` 308, `total_memory_bytes` 312, `free_memory_bytes` 320, `supported` 328, `unsupported_reason[256]` 332, `_reserved1` 588 |
| `mi_probe_info` | 1624 | `struct_size` 0, `driver_cuda_version` 8, `runtime_cuda_version` 12, `driver_compatible` 16, `device_count` 20, `supported_device_count` 24, `build_flavour[16]` 28, `recommended_flavour[16]` 44, `message[512]` 60, `_reserved0` 572; then `cublaslt_source[16]` 576, `cublaslt_version` 592, `_reserved1` 596, `cublaslt_path[512]` 600, `cublaslt_error[512]` 1112 |
| `mi_engine_params` | 136 | `struct_size` 0, `model_path` 8, `model_format` 16, `mode` 20, `device_index` 24, `_reserved0` 28, `vram_budget_bytes` 32, `vram_reserve_bytes` 40, `host_ram_budget_bytes` 48, `cache_dir` 56, `embedding_cache` 64, `direct_io` 68, `block_path` 72, `_reserved1` 76; then `dit_file` 80, `text_encoder_file` 88, `vae_file` 96, `tier` 104, `attention` 108, `lora_strict` 112, `_reserved2` 116; then `time_embedding_file` 120, `distill_dir` 128 |
| `mi_progress` | 40 | `struct_size` 0, `phase` 8, `step` 12, `total_steps` 16, `_reserved0` 20, `step_seconds` 24, `elapsed_seconds` 32 |
| `mi_gen_params` | 320 | `struct_size` 0, `prompt` 8, `negative_prompt` 16, `width` 24, `height` 28, `steps` 32, `_reserved0` 36, `cfg` 40, `shift` 48, `seed` 56, `uncond_skip_layers` 64, `uncond_skip_layers_count` 72, `slg_layers` 80, `slg_layers_count` 88, `slg_scale` 96, `slg_start` 104, `slg_stop` 112, `max_sequence_length` 120, `sampler` 124, `progress_callback` 128, `plan_callback` 136, `preview_callback` 144, `preview_interval` 152, `_reserved1` 156, `user_data` 160; then `scheduler` 168, `noise_layout` 172, `noise_source` 176, `save_metadata` 180, `flow_multiplier` 184, `eta` 192, `s_noise` 200, `dataspace_path` 208, `loras` 216, `loras_count` 224, `include_prompts` 232, `_reserved2` 236; then `init_image` 240, `init_image_path` 248, `mask` 256, `mask_path` 264, `mask_encoding` 272, `resize` 276, `denoise_strength` 280, `hires_width` 288, `hires_height` 292, `hires_strength` 296, `hires_steps` 304, `_reserved3` 308, `distill_dir` 312 |
| `mi_lora` | 16 | `path` 0, `strength` 8 (array element, no `struct_size`: frozen as it is) |
| `mi_image` | 24 | `pixels` 0, `width` 8, `height` 12, `channels` 16, `_reserved0` 20 (no `struct_size`: frozen as it is) |
| `mi_timings` | 128 | `struct_size` 0, then 12 doubles from 8 in header order (`total_s` … `vae_load_s` at 96); then `vae_encoder_load_s` 104, `image_encode_s` 112, `image_ops_s` 120 |
| `mi_memory_stats` | 56 | `struct_size` 0, then 6 `uint64_t` from 8 in header order (`arena_capacity_bytes` … `host_private_peak_bytes` at 48) |
| `mi_dit_stream_stats` | 80 | `struct_size` 0, `streamed` 8, `resident_blocks` 12, `buffers` 16, `stalls` 20, `bytes` 24, `copy_seconds` 32, `stall_seconds` 40, `h2d_gbps` 48, `hidden_fraction` 56, `source[16]` 64 (no padding needed) |
| `mi_tier_info` | 32 | `struct_size` 0, `requested` 8, `effective` 12, `attention` 16, `certified` 20, `dit_precision` 24, `text_encoder_precision` 28 |
| `mi_prompt_tokens` | 32 | `struct_size` 0, `total_tokens` 8, `kept_tokens` 12, `prompt_tokens` 16, `dropped_tokens` 20, `max_length` 24, `truncated` 28 |
| `mi_lora_report` | 32 | `struct_size` 0, `adapters` 8, `side_branch` 12, `merged` 16, `unknown_keys` 20, `warnings` 24, `notes` 28 |
| `mi_image_info` | 80 | `struct_size` 0, `mode` 8, `hires_ran` 12, `first_width` 16, `first_height` 20, `first_steps_scheduled` 24, `first_start_step` 28, `first_steps_run` 32, `first_encoded` 36, `hires_width` 40, `hires_height` 44, `hires_steps_scheduled` 48, `hires_start_step` 52, `hires_steps_run` 56, `_reserved0` 60, `first_strength` 64, `hires_strength` 72 |

The 32-bit enums: `mi_status`, `mi_mode`, `mi_model_format`, `mi_block_path`, `mi_tier`, `mi_attention`, `mi_precision`, `mi_sampler`, `mi_scheduler`, `mi_noise_layout`, `mi_noise_source`, `mi_phase`, `mi_result_text_kind`, `mi_message_kind`, `mi_mask_encoding`, `mi_resize_mode`, `mi_image_mode`.

History, all with `MI_ABI_VERSION` 1:
- 2026-10-21, during 0.1.0: the padding was made explicit; `mi_engine_params` grew `block_path` and `_reserved1` (72 → 80 bytes); `mi_dit_stream_stats` was added. Binaries built against an earlier header must be rebuilt.
- 0.2.0: `mi_engine_params` grew to 120 bytes and `mi_gen_params` to 240, as trailing fields. The 0.1.0 sizes, 80 and 168 bytes, stay accepted ([Struct sizes](#struct-sizes)). `mi_lora`, `mi_tier_info`, `mi_prompt_tokens` and `mi_lora_report` were added, and `mi_sampler` gained values 2–8.
- 0.2.0, image input: `steps` 0, a negative `cfg` and a negative `denoise_strength` became the unset markers, and `mi_gen_params_init()` at 320 bytes sets them. `mi_engine_params` grew to 136 bytes, `mi_gen_params` to 320 and `mi_timings` to 128, as trailing fields; the 120-, 240- and 104-byte layouts stay accepted. `mi_image` and `mi_image_info` were added.
- 0.2.0, system CUDA: `mi_probe_info` grew to 1624 bytes (the cuBLASLt fields), as trailing fields; the 576-byte layout stays accepted. `MI_CAP_SYSTEM_CUDA` was added.
- 1.0.0: first stable release. No layout or behaviour change from 0.2.0; the CMake package now requires the same MAJOR version (`SameMajorVersion`).

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
| `src/api/c_api.cpp` | the entry points, exception → status mapping, the busy/cancel state |
| `src/api/pipeline_adapter.{h,cpp}` | the only code that touches `pipeline::Pipeline`: converts the C structs into `Request` / `PipelineOptions`, and flattens `pipeline::Result` into C-API types and the JSON report |
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
- **`c_abi_compat`** (`tests/c/test_abi_compat.c`, C11): a hand-written 0.1.0-layout `mi_engine_params` and `mi_gen_params` through `*_init`, `mi_gen_params_validate()` and `mi_engine_create()` (as far as the model path check). With weights, `c_abi_compat_gpu` (label `gpu`) also generates from them and checks that the PNG has no text chunks.
- **`CApiReport.*`** renders the report (through the adapter and the shared renderer) from a synthetic result under a comma-decimal C++ locale with digit grouping, and under a comma-decimal C locale where the system has one. It checks for valid JSON, exact numbers, `null` for NaN and infinity, the `dit_blocks` and `dit_stream` keys, `quality.cublaslt`, and that the document is the CLI's fields in braces.
- **`PipelineReport.*`** (`tests/unit/test_pipeline_report.cpp`) covers the shared renderer itself: every key, the head/tail split around the image path, U+FFFD for ill-formed UTF-8, `null` for non-finite numbers, and the classic locale.
- **`CudaLibs.*`** (`tests/unit/test_backend_cuda_libs.cpp`) covers the cuBLASLt search: `MOSSINFER_CUDA_LIBS` parsing, the compatibility rule, the candidate lists of each policy on Linux and Windows (injected environment and module directory), `MOSSINFER_CUBLASLT` as the only candidate, the error listing for a missing file and for a library without cuBLASLt's symbols, the system search on this machine, and the shim forwarding to the process's copy. `CApi.ProbeWithoutRequiringAGpu` checks the probe's cuBLASLt fields and the 576-byte size.

Label `gpu`:
- **`GpuCApiProbe.*`**: after `mi_probe_device()`, the process holds no CUDA context. This is checked through the driver's primary-context state and NVML's process list.
- **`GpuCApi.*`** covers the device probe, and a 256², 2-step generation with callbacks, a concurrent call rejected as busy, the block path accessors and `mi_result_dit_stream()` (nothing streamed in balanced mode on a 16 GiB card), and PNG and report checks. It also covers cancel from another thread with no carry-over, and the caller's current context restored by create, generate and free. `SamplerPreviewMetadataAndTier` runs a 256², 3-step `DPMPP_2M` + `KARRAS` generation with `preview_interval` 2 (previews after steps 2 and 3, at the image size), checks the `generation` chunk of the saved PNG, `mi_result_tier()` (the bf16 bundle with tier `AUTO` and no certificate runs `REFERENCE` with exact attention), the texts, the prompt tokens, and the LoRA and message accessors' bounds. `OldSizeWritesNoMetadata` generates from a 168-byte `mi_gen_params` and checks that the PNG has no text chunks. `ImageToImageFromPixels` (256², 4 steps at strength 0.5: 2 run from step 2, `total_steps` 2, a 104-byte `mi_timings` gets 104 bytes), `InpaintBinary01Mask` (the kept half stays closer to the init image than the regenerated half), `HiresFix` (256 → 384, 2 + 2 steps) and `DistillTimeEmbeddingNeedsTheEngine` cover image input and the engine check. **`GpuCApiModels.*`** create their own engine on the SFT bundle when the files exist under `/workspace/Models/community`: `DistillBundle` (the lodestones SFT 8-step bundle on the engine and the request: 8 data-space steps at CFG 1, its LoRA and time embedding) and `TurboInt4` (an INT4 DiT: tier `AUTO` refused, `TURBO` runs it as `W4A4`). Runtime LoRA files of one's own on the GPU are covered only by the distill bundle's LoRA; the repository has no small LoRA file.

Run the GPU tests on a machine with a supported GPU:

```
GTEST_FILTER='GpuCApi.*' <build>/mossinfer_tests
```
