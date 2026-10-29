# Integrating MossInfer into a launcher

MossInfer is a native C++/CUDA engine for MOSS-Image text-to-image generation, with no PyTorch or Python at run time. This guide is for teams that build it into an application, such as the MOSS-Image launcher. It covers what is supported, what to ship, how to start and drive the engine, and how to prepare for the features that are coming.

The reference documents are:
- [c-api.md](c-api.md): the C library, `mossinfer.h`.
- [server-protocol.md](server-protocol.md): the `moss-server` sidecar.

This page does not repeat them. It explains how the pieces fit together, and which choices the launcher has to make.

Version 2.0.0 (ABI 1, server protocol 2). See [Versioning](#versioning) for what stays compatible, and [Migrating from 1.x](#migrating-from-1x) for what changed.

MossInfer is a low-level engine: it runs what the caller asks for, refuses what the hardware or the files cannot run with the reason, and reports what ran. It does not choose for the caller. The launcher picks the tier, the memory mode and the budgets; [Performance](#performance) says what each costs and gives.

## Support status

### Supported now

| Area | What works |
|---|---|
| Platform | Linux x86-64 and Windows x64 (MSVC), each in a `cu12` and a `cu13` flavour, as a full package and a `-system-cuda` package. CI builds, packages and runs the CPU tests of all four flavours; GPU runs are verified on Linux `cu13`. |
| GPUs | NVIDIA compute capability 8.x and 12.x: RTX 30xx, RTX 40xx, RTX 50xx, A100/A10/A40, L4/L40, RTX PRO Blackwell |
| Models | MOSS-Image Base and SFT only: the official Hugging Face folders (bf16), read in place, and fine-tunes with the same architecture; other models are refused when they load. See [Supported models](#supported-models). |
| Component model files | a single-file DiT, text encoder or VAE next to the bundle folder: ComfyUI and community files, INT8 ConvRot, INT4 ConvRot and NVFP4 DiT (NVFP4 on RTX 50 only), INT8 ConvRot and W4A8 text encoder (`dit_file`, `text_encoder_file`, `vae_file`) |
| Generation | text-to-image with the official app's parameters: prompt, negative prompt, size, steps, CFG, shift, seed, skip layers, skip-layer guidance |
| Quality tier | required, the caller's choice: `reference` (the DiT's bf16 GEMMs, exact attention), `high` (the loaded DiT's INT8 or bf16 GEMMs, 8-bit attention), `turbo` (a 4-bit DiT: INT4, or NVFP4 on an RTX 50, 8-bit attention); a tier the loaded files cannot run is refused with the reason, and the tier that ran is reported |
| Image input | img2img (an init image and a denoise strength) and inpainting (a mask), with crop / fit / stretch resizing; pixels (library) or PNG / JPEG files (both surfaces) |
| Hires fix | a second pass at a larger size from the first pass upscaled, with its own strength and steps |
| Distill bundles | few-step distilled checkpoints shipped as a bundle folder (time embedding, LoRA, data-space sampler parameters), applied per engine and per generation |
| Control adapters | a control adapter file (a LoRA with a widened DiT input embedder, e.g. the MOSS-Image color / depth control) with one conditioning image per kind, each with a strength and a step window; pixels (library) or PNG / JPEG files (both surfaces). Opt-in. |
| Lossy accelerators | CFG truncation and a step cache (First-Block Cache), off by default. Opt-in: they change the image. |
| Attention mode | exact (FlashAttention-2) or one of four 8-bit modes, as an override of the tier's default |
| LoRAs | a list of files with strengths per generation, applied at run time: kohya, PEFT / diffusers, LyCORIS LoHa and LoKr, and full deltas; per-file reports and warnings |
| Samplers and schedulers | 9 samplers (Euler, AB2, Euler ancestral, Heun, DPM++ 2M, DPM++ 2M SDE, ER-SDE, RES multistep, data-space), 7 schedulers, noise layout and source, eta, s_noise, flow multiplier |
| Preview | a latent preview image during sampling, every N steps (library: a callback with RGB8; server: a PNG rewritten in place and a `preview` event) |
| PNG metadata | the generation settings embedded in the PNG (`generation` record, schema `mossinfer.generation/1`, plus `parameters`), the prompts optional |
| Prompt token report | token counts per prompt and a truncation warning |
| Memory modes | `min_memory`, `balanced`, `max_speed`, plus explicit VRAM and RAM budgets and a VRAM reserve. The engine plans against the VRAM that is actually free and streams the model when it doesn't fit. |
| Integration surfaces | the C library (`libmossinfer`) and the sidecar process (`moss-server`), with the same features |
| Run-time control | progress: the phase and step, and the whole job's share done and time left over every phase, the text encoder and model loads included (`fraction`, `eta_seconds`); cancel from any thread, one job (generation, preload or release) at a time per engine, device probe and build recommendation |
| Preload | load the models for a request before the first generation (`mi_engine_preload()`, server `preload`), so the first image starts sampling at once; the next generation of a like request loads nothing and gives the same bits |
| Soft unload | free the VRAM and keep the engine (`mi_engine_release()`, server `release`), optionally with the DiT kept in RAM for a fast reload over PCIe (`keep_host_copy`); the embedding cache survives it. See [Fast switching](#fast-switching). |
| System CUDA reuse | a compatible cuBLASLt from the user's CUDA install, used by the `-system-cuda` package or on request (`MOSSINFER_CUDA_LIBS=system`); the probe reports which copy runs (verified on Linux; see [CUDA libraries](#cuda-libraries)) |
| Output | PNG file (both surfaces), or RGB8 in memory (library), plus a JSON report per image |
| File inspection | what a file is before anything loads, without an engine or a GPU (`mi_inspect_file()`, server `inspect`): DiT, text encoder, VAE, LoRA or control adapter, whether the engine accepts it and why not; a DiT's precision, quantization and the tiers it allows; a LoRA's rank, adapters and `base_model`; a control adapter's kinds |
| Accepted values | `hello`'s `values`: the names of every enum-like `load` / `generate` parameter, from the tables the server validates against |

### Control adapters and accelerators

They run on all three surfaces: `mi_gen_params` (`MI_CAP_CONTROL`, `MI_CAP_ACCEL`; [c-api.md](c-api.md#control-adapters)), `generate`'s `control`, `cfg_truncation`, `step_cache` and `step_cache_threshold` ([server-protocol.md](server-protocol.md#generate)), and `moss-infer generate`. All of them change the image and are opt-in:
- **Control adapters:** a control adapter file (`--control <file>`) and its LoRA strength (`--control-scale S`, default 1). Each conditioning image (`--control-image <kind>=<png|jpg>`) gives the input of one of the adapter's kinds, with a strength (`--control-strength X`, default 1, 0 to 10) and a step window (`--control-start X` / `--control-end X`, default 0 / 1: the share of the steps it applies to). A kind without an image gets no control.
- **Lossy accelerators**, off by default: CFG truncation (`--cfg-truncation X`, 0 to 1, default 1 = off) drops the negative branch from step X · steps on; the step cache (`--step-cache fbcache`, default `off`) reuses DiT blocks 1..41 from the last full call while block 0's output changed by less than the threshold (`--step-cache-threshold`; default 0.02 on the base model and 0.03 on the SFT model and its derived files, and the report's `accelerators.step_cache_threshold` and `step_cache_threshold_source` say which ran). Both change the image; the tier is resolved as if they were off. The step cache's gain depends on the model, precision, step count and size: on an RTX 5090 at 1024², 50 steps, threshold 0.03 reused 27–30 % of the DiT calls with bf16 and INT8, and none with INT4 Turbo (its own rounding noise keeps block 0's change above the threshold; the report warns when the cache never engaged).
- The report records them in `control` and `accelerators`, and `dit_stream.steps[i].cached` marks the steps that reused the cache. The server adds a `done.warnings` line for the accelerators.

### Planned: integrate now, enabled later

These are designed and partly built. The launcher can prepare for them now (see [Preparing for planned features](#preparing-for-planned-features)).

| Feature | State | How the launcher prepares |
|---|---|---|
| System CUDA reuse on Windows | the search paths are written (`CUDA_PATH`, `CUDA_PATH_V<major>_*`), not yet verified on Windows | the same logic as on Linux: read `probe.cublaslt` |

### Not supported

| Area | Status |
|---|---|
| GPUs with compute capability below 8.0 (GTX 16xx, RTX 20xx and older) | not supported: the engine needs bf16 tensor cores |
| GPUs with compute capability 9.x, 10.x or 11.x (H100/H200, B200/GB200 and similar) | not supported: no kernels are built for them |
| AMD, Intel and Apple GPUs; CPU-only machines; macOS | not supported |
| 32-bit systems | not supported |
| One generation split across several GPUs | not supported. One engine per GPU is the intended pattern (not yet tested on multi-GPU machines). |
| Several generations at once on one engine | not supported: queue them in the launcher |
| A ComfyUI checkpoint or models folder as the model path | not supported: the model path is a bundle folder; ComfyUI files load as component files |
| An automatic tier | not supported since 2.0: the caller chooses `reference`, `high` or `turbo` |
| NVFP4 DiT files on GPUs other than the RTX 50 series (compute capability 12.0) | not supported: refused (`MI_ERR_MODEL`) with `NVFP4 needs an RTX 50 GPU (compute capability 12.0); use the INT8 file …`. There is no bf16 decode fallback. A build without the sm_120a NVFP4 kernels (`MOSSINFER_NVFP4=OFF`, CUDA < 12.8) refuses them on an RTX 50 too. |
| Pixel data over the server protocol | not supported: the server takes image file paths |
| Server output other than PNG | not supported |
| Progress inside a DiT load | not supported: a DiT load is one step, so `fraction` holds still while it runs and moves on at the next phase |
| Inspecting a folder | not supported: `inspect` takes one file; a bundle folder is checked by `load` |
| LoRAs for other models | not supported: `inspect` reports a file in the LoRA naming conventions none of whose tensors applies to the MOSS-Image DiT as a LoRA with `loadable` false, and the LoRA reader's reason as its note |

## Supported models

The engine is built for one model architecture, MOSS-Image: a 5B DiT, the Qwen3.5-9B text encoder and the FLUX.2 small VAE decoder. Other models named MOSS, other sizes of MOSS-Image, and MOSS-Image files in formats not listed below do not run.

**Official models.** The revisions below are the ones tested. Their weight files are byte-identical in every revision published so far (up to 2026-10-21), and in the ModelScope copies (`openmoss/MOSS-Image-Base`, `openmoss/MOSS-Image-SFT`).

| Model | Hugging Face repository | Tested revision | Size |
|---|---|---|---|
| MOSS-Image Base | `OpenMOSS-Team/MOSS-Image-Base` | `32edbc8a53ff3117ed52af34ce27a031129f2d0e` | 27.2 GB |
| MOSS-Image SFT | `OpenMOSS-Team/MOSS-Image-SFT` | `2c7954437269461767f533b006823e8f85973a60` | 27.2 GB |

```sh
hf download OpenMOSS-Team/MOSS-Image-Base --revision 32edbc8a53ff3117ed52af34ce27a031129f2d0e --local-dir MOSS-Image/base --exclude 'assets/*'
hf download OpenMOSS-Team/MOSS-Image-SFT  --revision 2c7954437269461767f533b006823e8f85973a60 --local-dir MOSS-Image/sft  --exclude 'assets/*'
```

- **Fine-tunes** with the same architecture (the same config values, tensor names and shapes; class names may differ) load and run like the official folders.
- **How the engine checks.** It identifies a model by its architecture, from configs and safetensors headers only (no hashing): the folder's `transformer/config.json`, `text_encoder/config.json` and `vae/config.json`, and the tensor names and shapes of the DiT and text-encoder files. Anything else fails the first generation with `MI_ERR_MODEL` (server: `model`), and `moss-infer index` refuses it too. The message names what was found, e.g.: `MOSS-Image/x is not a supported MOSS-Image model (found transformer num_layers 40 (MOSS-Image: 42)). Supported: OpenMOSS-Team/MOSS-Image-Base and -SFT; see docs/integration.md#supported-models`. A folder without `transformer/config.json` or `text_encoder/config.json` is reported as incomplete (`… (incomplete download, or not the model folder?)`), also `MI_ERR_MODEL`. A LoRA, control adapter or distill bundle file given as the DiT or text encoder is named as such.

**Community files verified during 1.0 and 2.0 development.** Files loaded next to an official folder. The last column names the run:
- **E24, E25:** RTX 5090 runs on pre-release commits `211ef35` and `1b141ce` (ancestors of v1.0.0); images generated end to end.
- **4090L:** RTX 4090 Laptop, the 2.0 development tree (2026-10-23); images generated end to end.
- **Tests:** the GPU test suite loads the file and checks its results against reference values.

The revision is the one whose files were tested (same SHA-256); copies of the same file under another name work the same.

| Repository (revision) | File | What it is | How it loads | Verified by |
|---|---|---|---|---|
| `Comfy-Org/MOSS-Image` (`24147db`) | `diffusion_models/moss_image_sft_int8_convrot.safetensors` | SFT DiT, INT8 ConvRot | `dit_file`, tier `high` | E25, 4090L |
| `Comfy-Org/MOSS-Image` (`24147db`) | `diffusion_models/moss_image_base_int8_convrot.safetensors` | base DiT, INT8 ConvRot | `dit_file`, tier `high` | RTX 5090 (sm_120) |
| `Comfy-Org/MOSS-Image` (`24147db`) | `text_encoders/qwen3.5_9b_int8_convrot.safetensors` | text encoder, INT8 ConvRot | `text_encoder_file` | 4090L, tests |
| `Comfy-Org/MOSS-Image` (`24147db`) | `text_encoders/qwen3.5_9b_w4a8_gptq.safetensors` | text encoder, W4A8 GPTQ | `text_encoder_file` | 4090L, tests |
| `Comfy-Org/MOSS-Image` (`24147db`) | `vae/flux2-vae.safetensors` | full FLUX.2 VAE (decoder and encoder) | `vae_file` | 4090L, tests |
| `Comfy-Org/MOSS-Image` (`24147db`) | `vae/flux2-vae_small.safetensors` | FLUX.2 small VAE in fp32 (the folder's VAE is the same model rounded to bf16) | `vae_file` | tests |
| `Comfy-Org/MOSS-Image` (`24147db`) | `loras/moss_image_color_depth_control.safetensors` | colour and depth control adapter (the same file as `darask0/MOSS-Image-color-depth-control`'s `moss-control.safetensors`) | `moss-infer --control` (CLI only in 1.0) | E24 |
| `toxicdog/MOSS-Image-SFT-INT4-ConvRot-ComfyUI` (`220f724`) | `diffusion_models/moss_image_sft_int4_convrot.safetensors` | SFT DiT, INT4 ConvRot | `dit_file`, tier `turbo` | E25, 4090L |
| our converter: `moss-infer quantize <sft> --format nvfp4` | e.g. `moss_image_sft_nvfp4.safetensors` | SFT or base DiT, NVFP4 with INT8 ConvRot keep-list units (default recipe `gptq-mse`, keep list `g0.9`) | `dit_file`, tier `turbo`, RTX 50 only | E39, tests |
| `BennyDaBall/MOSS-Image-SFT-NVFP4` (SHA-256 `b478cb62…`) | `moss_image_sft_nvfp4.safetensors` | SFT DiT, NVFP4 with INT8 ConvRot linears (late blocks and every ffn_down) | `dit_file`, tier `turbo`, RTX 50 only | E39, tests |
| `toxicdog/MOSS-Image-SFT-Turbo-INT8-ConvRot` (`a04afd6`) | `diffusion_models/moss_image_sft_turbo_int8_convrot.safetensors` | 8-step SFT variant, INT8 ConvRot | `dit_file`, tier `high`, 8 steps, CFG 1 | format tests only |
| `tsolful/MOSS_Image_INT6_INT4GPTQ` (`782fa67`) | `moss_image_sft_W4A8_GPTQ.safetensors` | SFT DiT, W4A8 GPTQ | convert first: `moss-infer quantize <file> -o <out>` decodes it to INT8 ConvRot (bit-exact); then `dit_file`. As `dit_file` directly it is refused (`MI_ERR_MODEL`, "needs decoding"). | conversion: tests |
| `lodestones/MOSS-Image-SFT-8-steps-dmd` (`f95df83`) | `time_embedding.safetensors`, `lora.safetensors`, `dataspace.safetensors` | 8-step distill bundle for SFT | `distill_dir`, 8 steps, CFG 1 | E25, tests |
| `akhaliq/MOSS-Image-SFT-4-steps` (`2ded95c`) | `lora.safetensors` | 4-step distill LoRA for SFT | LoRA at strength 1, 4 steps, CFG 1 | E25, tests |
| `axiomofmind/Moss-image-beauty-lora` (`967ff12`) | `pytorch_lora_weights.safetensors` | PEFT LoRA | `loras` | tests |
| `congruency/MOSS-Image-Turbo-MFD` (`4c7e9c2`) | `MOSS-Image-SFT-Turbo-MFD-v0.1.comfy.safetensors` | kohya LoRA | `loras` | tests |
| `comfyanonymous/MOSS_image_MAX_ULTRA_hires_lora_test` (`6784131`) | `moss_hires_lora_test_v0.1.safetensors` | LyCORIS LoHa | `loras` | tests |
| `nynxz/MOSS-Image` (`138539a`) | `loras/moss_sft_destripe_top64_bf16.safetensors` | LoRA | `loras` | tests |

Same formats, not tested: Comfy-Org's Base INT8 DiT and its bf16 DiTs, tsolful's W6A8 and Base files, `unsloth/MOSS-Image-SFT-FP8`'s `MOSS-Image-SFT-text_encoder-INT8-ConvRot.safetensors` (ComfyUI INT8), `lodestones/MOSS-Image-Base-8-steps-dmd`, `darask0/MOSS-Image_4-step_adapter_lora`.

**Not supported.** Messages start with the file or folder path and end with the supported-models pointer.

| Model or file | What it is | What the engine says |
|---|---|---|
| Other OpenMOSS models: MOSS-VL, moss-video-preview, MOVA, MOSS-TTS, MOSS-Audio, moss-moon | not MOSS-Image (vision-language, video, speech and language models) | as the model folder: `… is not a supported MOSS-Image model (found <its pipeline class>)` for a diffusers folder; otherwise the incomplete-folder message |
| `nebulette/moss-image-2b-encoder` | a Qwen3.5-2B text encoder | as `text_encoder_file`: `… is not a supported MOSS-Image text encoder (found embed_tokens.weight [248320, 2048] (MOSS-Image: [248320, 4096]), no norm.weight, 24 layers (MOSS-Image: 32))` |
| `PotatoForge/moss-image-4b-encoder-adapter` | a projection from Qwen3.5-4B to 9B features, not a model | refused as a text encoder |
| `chinoll/Moss-Image-Edit`, `akhaliq/MOSS-Image-Edit-HQ` and other edit adapters | image editing (a reference-image input) | 1.0 has no editing input |
| GGUF files (`unsloth/…-GGUF`, `leejet/…-GGUF`, `realrebelai/MOSS-Image_GGUFs`, …) | llama.cpp format | `… (found a GGUF file (llama.cpp / ComfyUI-GGUF format); the engine reads safetensors files)` |
| torchao files: `unsloth/MOSS-Image-SFT-FP8`'s `MOSS-Image-SFT-FP8.safetensors`, `MOSS-Image-SFT-INT8.safetensors` and `MOSS-Image-SFT-INT8-ConvRot.safetensors` (despite the name, not ComfyUI's INT8 ConvRot) | torchao-quantized weights | `… (found block weights stored as 'layers.0.proj.ffn_down._weight_qdata', a quantization format the engine does not read)` |
| ComfyUI FP8 files (`fp8_scaled`, `float8_e4m3fn`) | 8-bit float weights | `… (found block weights stored as F8_E4M3 (FP8), a quantization format the engine does not read)` |
| NVFP4 files on GPUs other than the RTX 50 series | 4-bit float weights (ComfyUI `nvfp4`) | `… NVFP4 needs an RTX 50 GPU (compute capability 12.0); use the INT8 file …` (`MI_ERR_MODEL`) |
| NVFP4 files in NVIDIA's ModelOpt convention (no `comfy_quant` configs), or with a `pre_quant_scale` | other NVFP4 layouts | refused naming the layout; only ComfyUI `comfy_quant` `nvfp4` files load |
| Other quantization formats in a ComfyUI file | a `comfy_quant` format the engine does not read | `… quantization format '<name>' is not supported (see docs/integration.md#supported-models)` |
| MLX (`mlx-community/…`, `mflux-community/…`), SDNQ (`OzzyGT/…`), DFloat11 (`mingyi456/…`), DiffSynth EntroPack, PyTorch `.bin` / `.pt` files | other runtimes' formats | not tested and not expected to load; a `.bin` / `.pt` file: `… (found a PyTorch pickle (.bin / .pt / .ckpt); the engine reads safetensors files)` |

## Choosing the integration surface

| | `moss-server` (sidecar process) | `libmossinfer` (in-process library) |
|---|---|---|
| Launcher language | any: talk JSON over pipes | C, or anything with a C FFI (C++, Rust, C#, Python ctypes, Node N-API) |
| Crash isolation | a GPU fault or driver reset kills only the server | a fault takes down the launcher |
| Releasing VRAM | `release` (keeps the engine, optionally the weights in RAM), `unload`, or end the process (always complete) | `mi_engine_release()` (keeps the engine), `mi_engine_free()` |
| Image hand-off | a PNG written to a path you choose | the RGB8 buffer in memory, or a PNG |
| Progress and cancel | JSON events; a `cancel` request | callbacks; `mi_cancel()` from any thread |
| What you ship | `moss-server` plus `libmossinfer` and the CUDA libraries | `libmossinfer` and the CUDA libraries |

**Recommendation: use the sidecar for the launcher.** The CUDA driver keeps a lot of state per process. A failed CUDA call can leave a process unable to use the GPU again, and the only reliable recovery is a new process. With the sidecar, recovery is "restart `moss-server`". The library is the better fit for embedding in another native program, or when you need images in memory with no file in between.

Both surfaces expose the same engine with the same parameters and reports. The server is a thin layer over the public C API (`tools/server/main.cpp`).

## System requirements

**GPU:** see [Support status](#support-status). The engine targets cards with 8 GiB of VRAM or more. On smaller budgets it streams the model from RAM or disk instead of failing. Generation fails with `out_of_memory` only when no plan fits at all.

**Driver.** Each release comes in two builds, called **flavours**, which differ only in the CUDA version they were built with. The NVIDIA driver is never bundled.

| Flavour | Minimum driver | Use when |
|---|---|---|
| `cu13` | Linux ≥ 580.65.06, Windows ≥ 580.88 | the default: the driver supports CUDA 13 |
| `cu12` | Linux ≥ 525.60.13, Windows ≥ 527.41 | the driver is older and supports only CUDA 12 |

RTX 50xx (Blackwell) cards need driver series 570 or newer, whatever the flavour.

**Host RAM.** The engine's own memory aims to stay within a target per mode, and exceeds it only when no plan fits inside it. `host_ram_budget_bytes` overrides the target.

| Mode | Target |
|---|---|
| `min_memory` | 4 GiB |
| `balanced` | 8 GiB |
| `max_speed` | whatever the machine has free, minus 1 GiB |

On top of that, the weights are read through the operating system's page cache. Free RAM beyond the target keeps the model files cached, which makes streaming and reloads faster.

**Disk:**

| What | Size |
|---|---|
| MOSS-Image Base folder | 27.2 GB |
| MOSS-Image SFT folder | 27.2 GB, or about 10 GB extra when its text encoder and VAE files are hard links to Base's (they are byte-identical) |
| Engine package | several hundred MB, most of it NVIDIA cuBLASLt; the `-system-cuda` archive leaves cuBLASLt out |

## Getting the binaries

**From a release.** CI builds two archives per OS and flavour: `mossinfer-<version>-<os>-cu<12|13>-<arch>.zip` (the full package) and `mossinfer-<version>-<os>-cu<12|13>-<arch>-system-cuda.zip` (the same without cuBLASLt); Linux also gets `.tar.gz` of both. A GitHub release attaches the archives with one `SHA256SUMS` file. Release candidates (tags `vX.Y.Z-rcN`) are GitHub pre-releases; their archives and binaries carry the plain `X.Y.Z` version. See [Package contents](#package-contents).

**From source** (Linux shown):

```sh
# CUDA 13.x toolkit, CMake >= 3.28, Ninja, GCC >= 12, GoogleTest
cmake --preset linux-cuda13
cmake --build --preset linux-cuda13 -j8
CUDA_VISIBLE_DEVICES="" ctest --preset linux-cuda13 -L '^cpu$'
cmake --install build-linux-cuda13                # -> build-linux-cuda13/install
cmake --install build-linux-cuda13 --component Unspecified --prefix build-linux-cuda13/install-system-cuda
(cd build-linux-cuda13 && cpack --config CPackConfig.cmake)   # -> mossinfer-2.0.0-linux-cu13-x86_64.{tar.gz,zip}
```

- **glibc:** the archive needs the glibc of the machine that built it, or a newer one. Build on the oldest distribution you want to support. CI uses Ubuntu 22.04, which gives glibc 2.35. A build from a rolling distribution such as Arch will not start on Ubuntu LTS.
- **Other presets:** `linux-cuda12`, `windows-msvc-cuda13` and `windows-msvc-cuda12`. The Windows presets run from a Visual Studio 2022 x64 developer prompt.

### Package contents

| Path | What |
|---|---|
| `bin/moss-server` (`moss-server.exe`) | the sidecar |
| `bin/moss-infer` | command-line tool for debugging and for converting models. The launcher doesn't need it. It runs the engine in `libmossinfer` rather than carrying its own copy. |
| `lib/libmossinfer.so.1` (Windows: `bin/mossinfer.dll` + `lib/mossinfer.lib`) | the engine, the C API and the `moss-infer` commands. It exports the C API functions and `mi_internal_cli_main`, `moss-infer`'s entry point, which is not part of the C API. |
| `lib/libcublasLt.so.<major>` (Windows: `bin/cublasLt64_<major>.dll`) | NVIDIA cuBLASLt, redistributed under the CUDA EULA; full package only |
| Windows: `bin/vcruntime140*.dll`, `bin/msvcp140.dll` | the MSVC runtime, so users need no Visual C++ Redistributable |
| `include/mossinfer/mossinfer.h` | the C API header |
| `share/cmake/MossInfer/` | the CMake package: `find_package(MossInfer)` |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `share/doc/mossinfer/` | licences (see [Licensing](#licensing-and-redistribution)) |

**Two variants per flavour.**
- **Full** (`mossinfer-…-cu13-x86_64.zip`): bundles cuBLASLt and the CUDA EULA. It runs on any machine with a suitable driver, and uses its own cuBLASLt by default (`auto`, see [CUDA libraries](#cuda-libraries)): results are reproducible across machines.
- **System CUDA** (`mossinfer-…-cu13-x86_64-system-cuda.zip`): the same files without cuBLASLt and the EULA, several hundred MB smaller. It runs only where a compatible cuBLASLt is installed. It is made from the release presets, which link the CUDA runtime statically; a tree installed the same way from a build with a shared cudart (the dev preset) also needs a system cudart.
- **Users with a CUDA toolkit** can use its cuBLASLt instead in two ways: download the smaller `-system-cuda` archive, or keep the full one and set `MOSSINFER_CUDA_LIBS=system`. A system copy of another cuBLASLt version can change result bits (see Reproducibility below).
- **Which to offer:** the launcher decides from the probe. Run the probe of any installed copy with `MOSSINFER_CUDA_LIBS=system` (`hello`'s `probe.cublaslt`, or `mi_probe()`): when `source` is `"system"`, the machine has a compatible copy and the `-system-cuda` archive works there; otherwise offer the full one. A launcher that installs once for many machines ships the full archive.

**Ship the tree as it is.**
- `moss-server` and `moss-infer` find `libmossinfer` relative to themselves (RUNPATH `$ORIGIN/../lib`), so keep `lib/` with `bin/`. On Windows, all DLLs sit in `bin/`.
- `libmossinfer` has no load-time dependency on cuBLASLt: it loads cuBLASLt at run time (a system copy, or the bundled one next to it by absolute path).
- The full package needs no CUDA toolkit and no environment variables.
- The CUDA runtime is linked statically on every OS, and libstdc++ is linked statically on Linux.

### CUDA libraries

cuBLASLt is the only CUDA library the engine loads besides the driver. The engine loads it at run time, once per process: at the first `mi_probe()` or `mi_engine_create()` (server: `hello` or `load`).

**Requirements for a system copy:**
- the same CUDA major version as the flavour (`cu13`: cuBLASLt 13.x, `cu12`: 12.x);
- a cuBLASLt version not older than the one the build was compiled against;
- findable in one of the locations below.

A copy that fails a check is skipped, and the next location is tried.

**Search order.** `MOSSINFER_CUDA_LIBS` selects the candidates. The default `auto` tries the bundled copy, then the system locations: a full package uses its own copy, and a `-system-cuda` package, which has none, a compatible system copy. `system` tries only the system locations, also in a full package; `bundled` only the bundled copy.

| | Linux | Windows |
|---|---|---|
| System | `libcublasLt.so.<major>` through the dynamic loader (`LD_LIBRARY_PATH`, the `ld.so` cache), then by absolute path: `lib64/` and `targets/<arch>-linux/lib/` under `$CUDA_PATH`, `$CUDA_HOME`, `/usr/local/cuda-<major>` and `/usr/local/cuda`, then `/opt/cuda/lib64`, `/usr/lib/<arch>-linux-gnu`, `/usr/lib64` and `/usr/lib` (distribution packages). Relative `CUDA_PATH` / `CUDA_HOME` values are ignored. | `bin\x64\cublasLt64_<major>.dll` and `bin\cublasLt64_<major>.dll` under `%CUDA_PATH%`, then under every `%CUDA_PATH_V<major>_<minor>%` (newest first). Never the bare DLL name: the application directory would come first. |
| Bundled | next to `libmossinfer` (`lib/`); for an executable that links the engine statically (`moss-infer` in a development build), also `../lib` | next to `mossinfer.dll` (`bin/`) |

A system search that ends at the bundled file does not count as a system copy. The soname search uses the RUNPATH of the module that asks, so it reaches the bundled file first when that module's RUNPATH, or a directory on `LD_LIBRARY_PATH`, points at the package's `lib/`. The module that asks is `libmossinfer`, or the statically linked `moss-infer` of a development build, and only a build with a shared cudart (the dev preset) gives them such a RUNPATH. Release builds link cudart statically: `libmossinfer` has no RUNPATH, and their `moss-infer` has one only to find `libmossinfer`, which runs the search. The loader then rejects that file and goes on with the absolute paths, which is why the distribution directories are listed there too.

**Environment variables:**

| Variable | Values | Effect |
|---|---|---|
| `MOSSINFER_CUDA_LIBS` | `auto` (default), `system`, `bundled` | which candidates are tried (`auto`: bundled, then system); another value is an error |
| `MOSSINFER_CUBLASLT` | an absolute file path | only this file is tried (reported as `"env"`); it must pass the same checks. A relative path is refused. |

Both variables choose code that the process loads and runs. Treat them like `LD_LIBRARY_PATH`: set them only from trusted configuration, never from user-supplied or downloaded data.

**What the launcher sees.**
- `mi_probe_info.cublaslt_source`, `cublaslt_version`, `cublaslt_path` and `cublaslt_error`, or `probe.cublaslt` in `hello`, report the copy in use.
- Without a usable copy, the probe still answers, with the reason in `cublaslt_error`: every candidate it tried, and why each was rejected.
- `mi_engine_create()` then fails with `MI_ERR_CUDA` and the same message (server: `load` fails with error code `cuda`).

**Reproducibility.**
- A different cuBLASLt version can change the bits of a result.
- The engine reruns its fused-kernel self-check for each cuBLASLt version, so the correctness check always covers the copy in use.
- Each report records the copy (`quality.cublaslt`).
- The default (`auto` in a full package) and `MOSSINFER_CUDA_LIBS=bundled` run the release's own library, so results can be compared across machines.

## Model files

The engine reads the official Hugging Face model folders in place, with no conversion step. [Supported models](#supported-models) lists them, with the download commands.

- **The model path is the folder** that contains `text_encoder/`, `tokenizer/`, `transformer/` and `vae/`, e.g. `MOSS-Image/base`.
- **The folder can be read-only.** The engine writes nothing into it. An optional `moss.mossidx` index, made with `moss-infer index <folder>`, makes the first load slightly faster. Without one, the engine builds the index in memory.
- **To save disk with both models,** hard-link SFT's `text_encoder/*.safetensors` and `vae/*.safetensors` to Base's. They are byte-identical.
- **ComfyUI and quantized files** load as component files next to the bundle folder: `dit_file` (INT8 ConvRot, INT4 ConvRot, NVFP4 (RTX 50 only), mixed INT8/bf16 or bf16 DiT), `text_encoder_file` (INT8 ConvRot or W4A8) and `vae_file`. The tokenizer and configs still come from the folder, but the folder's weights for a component a file replaces are not read: a folder with only the configs and the tokenizer serves when all three files are given. A VAE file is decoded with its own latent statistics rounded to bf16 as the folder stores them, so the Comfy-Org VAE files give the folder's image bit for bit (see `vae_file` in [c-api.md](c-api.md#engine)). A file as the model path itself (`MI_MODEL_FORMAT_COMFYUI`, or a file with `MI_MODEL_FORMAT_AUTO`) returns `MI_ERR_NOT_SUPPORTED`, and the server answers `not_supported`, while `MI_CAP_COMFYUI_MODELS` is unset.
- **Quantized weights are lossy.** An INT8 DiT runs the `high` tier's INT8 GEMMs, and `reference` refuses it. An INT4 or NVFP4 DiT runs only under `turbo`: offer it for INT4 and NVFP4 files, and only for them. Runtime LoRAs on NVFP4 linears run as side branches: their Δ is stored in bf16 (rounded once, unlike the INT8 path, which adds it unrounded inside the GEMM) and added in the NVFP4 GEMM's epilogue; LoHa, LoKr and full-delta adapters on NVFP4 linears are refused, as on INT4 ones. The report's `quality` object says what ran.
- **Paths name regular files and folders.** On Windows the engine refuses device and pipe paths (`\\.\…`, `\\?\…` other than a drive or `UNC\` path, DOS device names such as `NUL` or `COM1`, and the `pipe`, `mailslot` and `IPC$` shares) with `MI_ERR_INVALID_ARGUMENT` (the server: `invalid_params`) before opening anything. UNC paths to file shares (`\\nas\models\…`) are allowed: opening one authenticates to the named host with the engine process's account, as any Windows program does. A `moss-server` that takes paths from less-trusted clients (another user, a network front end) must not pass them through unchecked: restrict them to the folders it serves, so a client cannot make the engine authenticate to a host of its choosing.
- **Sort files before loading them** with `mi_inspect_file()` / `inspect`: it names each file's role (DiT, text encoder, VAE, LoRA, control adapter), says whether the engine accepts it (and why not: a GGUF or FP8 file, a W4A8 DiT to convert with `moss-infer quantize`, an NVFP4 DiT in a build without the NVFP4 kernels), and gives a DiT's tiers, so the tier menu offers `turbo` for exactly the INT4 and NVFP4 files. Inspection judges an NVFP4 file for an RTX 50 (it never looks at the GPU); on other GPUs the load refuses it.

## Performance

The launcher sets three things: the tier (with the DiT file it loads), the memory mode and, when it shares the GPU or the RAM, the budgets. The engine plans everything else within them. **For everyday use, load an INT8 DiT (`dit_file`) with tier `high`:** 2.4 times the speed of the bf16 Reference in 40 % less VRAM.

| Tier | Runs | Gives |
|---|---|---|
| `reference` | the DiT's bf16 GEMMs, exact attention | the official numerics, bit-reproducible run to run |
| `high` | the loaded DiT's INT8 or bf16 GEMMs, 8-bit attention | with an INT8 DiT, the everyday configuration; with the bf16 bundle, the 8-bit attention alone |
| `turbo` | a 4-bit DiT (INT4, or NVFP4 on an RTX 50), 8-bit attention | the fastest steps and the smallest weights |

RTX 5090, SFT, 1024², 50 steps, every model resident:

| Configuration | Denoise step | Speed | VRAM peak |
|---|---|---|---|
| bf16 bundle, `reference` | 0.466 s | 1.0× | 11.7 GiB |
| INT8 DiT, `high` (q8_fp8) | 0.192 s | 2.4× | 7.2 GiB |
| NVFP4 DiT, `turbo` (q8_fp8) | 0.143 s | 3.3× | 5.5 GiB |

The gain holds at 2048²: INT8 `high` runs 1.18 s per step against 2.77 s for bf16. 8-bit attention runs at every size; `attention` picks another kernel (`exact`, or the fp32-accumulating variants) when the caller wants one.

**Memory modes** trade host RAM for speed. `balanced` (the default) keeps the process near 8 GiB of private RAM, `min_memory` near 4 GiB (more streaming and parking, the lowest peaks), and `max_speed` uses what the machine has free (the text encoder held in RAM, pinned where there is room, and a larger embedding cache, so a new prompt starts sooner).

**VRAM budgets.** A DiT that does not fit stays exact: its blocks stream to the GPU every step, from the page cache or a pinned copy in RAM, and each step pays the part of the copy not hidden behind compute. Streaming from disk instead of RAM costs several seconds per step, so leave RAM free for the page cache. An INT8 DiT moves half the bytes of the bf16 one: on 8 to 12 GiB cards it is also the fast choice. Leave `vram_budget_bytes` at 0 unless the GPU is shared; the engine plans against the memory that is actually free.

**How to go faster:**
- Keep one engine alive: the DiT stays resident between generations, and repeated prompts reuse their embeddings (a `cache_dir` keeps them, and the measured transfer rates, across runs).
- `preload` before the first generation, and `release` with `keep_host_copy` to give the VRAM back and reload over PCIe without reading the files again ([Fast switching](#fast-switching)).
- `block_path` `fused_graphs` replays CUDA graphs when every DiT block is resident; all block paths give the same image.
- The lossy accelerators (step cache, CFG truncation) are opt-in: they change the image, so offer them as settings, off by default.

## Picking the build: cu12 or cu13

Ask the engine. `mi_probe()`, or the sidecar's `hello`, works on machines without a driver or GPU, and leaves no CUDA context behind. A `cu13` build also answers correctly on a CUDA 12 driver.

```
>> {"id": 1, "method": "hello", "params": {"client": "launcher", "protocol_version": 2, "devices": true}}
<< {"id": 1, "result": {..., "probe": {"driver_cuda_version": 13040, "build_flavour": "cu13",
    "recommended_flavour": "cu13", "driver_compatible": true, "supported_device_count": 1,
    "message": "recommended flavour cu13 (the driver supports CUDA 13.4)", "devices": [...]}}}
```

Decision procedure for the launcher:
1. Install the `cu13` flavour by default. Its `moss-server` starts on any machine, with or without a GPU.
2. Run `hello` with `"devices": true`.
3. Branch on `probe.recommended_flavour`:
   - `"cu13"`: done.
   - `"cu12"`: the driver supports only CUDA 12. Install the `cu12` flavour, or ask the user to update the driver (580 or newer). The `cu13` server still lists the GPUs, each marked unsupported with the driver as the reason, and its `load` fails with `cuda`.
   - `""`: no usable GPU. Show `probe.message` (e.g. "no flavour recommended: no device has compute capability >= 8.0") and the per-device `unsupported_reason`.
4. Use `devices[i].free_memory_bytes` for sizing hints in the UI. The engine checks memory again itself at generation time.

## Driving the sidecar

The full protocol, with an example of every message, is in [server-protocol.md](server-protocol.md). A launcher needs this much of it.

**Start.**
- Spawn `moss-server --parent-pid <launcher pid>` with pipes on stdin and stdout.
- Send stderr to a log file. The engine and CUDA log there; stdout only ever carries protocol lines.
- `--parent-pid` makes the server exit on its own if the launcher dies.
- On Linux, spawn it from a long-lived thread, such as the main thread: the kernel's parent-death signal follows the thread that spawned the server.

**Messages.**
- One JSON object per line, UTF-8.
- Every request has an `id` and gets exactly one response with that `id`.
- Events (`plan`, `progress`, `preview`, `done`, `log`) carry the `id` of the request they belong to, and can arrive between responses. Read stdout on a dedicated thread, and dispatch by `id` and `event`.

**Lifecycle.**
1. `hello`: send and check `protocol_version` 2 and `abi_version`, and pick the flavour (see above). Fill the settings' menus from `values`. `inspect` `{"path": ...}` classifies the user's model files at any time, before `load` too.
2. `load` `{"model": "<folder>", "dit_file": "<INT8 DiT>", "tier": "high", "mode": "balanced", "cache_dir": "<launcher cache dir>/mossinfer"}`. `tier` is required. This returns quickly. The weights load during `generate`: the DiT in the first `dit_load` phase, and the text encoder streams in `text_encoder` whenever the prompt is not in the embedding cache.
3. `generate` `{"prompt": ..., "output": "<path>.png", ...}`. The answer is immediate: `{"accepted": true}`. Then come `plan` once, `progress` at each phase start and after each step, `preview` before a step's `progress` when `preview_output` is set, and finally `done` with `status` `ok`, `cancelled` or `error`. Show `done.warnings` (LoRA losses, a truncated prompt) next to the image.
4. `cancel` `{"id": <the generate request's id>}` from the UI's stop button. The id must match exactly, JSON type included (`"g1"` is not `1`). The generation stops within one block or layer, and `done` arrives with `"status": "cancelled"`.
5. `release` frees the VRAM and keeps the engine, so the next `generate` needs no `load` (see [Fast switching](#fast-switching)). `unload` frees the engine too. `load` with different settings replaces the engine: the old one is released first, so if the new `load` fails, no engine is loaded.
6. `shutdown` on exit. The server cancels any running generation, answers, and exits with status 0. Kill it if it hasn't exited after a few seconds.

**Progress bar.**
- Drive the bar from `fraction` and the remaining time from `eta_seconds`, both in every `progress` event (and in `status`). `fraction` is the share of the whole job done, 0 to 1: the text encoder, the model loads, every sampling step (both passes of a hires fix) and the decode. It never goes back, and `done` is the end. `eta_seconds` is `null` while unknown.
- Do not build the bar from `step / total_steps`: it is 0 through the text encoder and the loads, which can be most of a first run, and covers only the sampling steps.
- `progress` events also report the `phase` (`text_encoder`, then `dit_load`, `denoise` and `decode`) for a label; `denoise` also reports `step` and `total_steps`.
- The estimate is roughest on the first run and corrects itself as phases and steps are measured. A DiT load is one step: the bar holds still while it runs.
- `hello`'s `capabilities.progress_fraction` says whether they are there. A server older than them sends neither field; this server on a library without them sends both as `null`. Without them, fall back to `step / total_steps` during `denoise`.

**One generation at a time.** A second `generate` while one runs is answered with `busy`, so queue requests in the launcher. The same holds for `preload` and `release`: one job at a time. `done` is written only after the server is idle again, so the next `generate` can go out as soon as `done` is read.

### Fast switching

A launcher that shares the GPU with other tabs or applications keeps the engine and moves the weights between VRAM and RAM:
1. **At startup, or when the user selects a model:** `load`, then `preload` with the request the UI will send (its size, steps, CFG, LoRAs; the prompt may be left out). The first `generate` then starts sampling without the `dit_load` wait. `done.preload_note` says when the plan leaves the DiT to the first generation.
2. **When the user leaves the image tab, or starts another GPU application** (a game, a video editor, another model): `release` with `keep_host_copy` true. The VRAM is free when the response arrives. The DiT stays in RAM where the host has the room (`dit_host_copy`; `note` says why not), and the embedding cache survives, so repeated prompts skip the text encoder.
3. **When the user comes back:** `generate` as usual, or `preload` first to pay the reload before the click. The DiT loads from the host copy over PCIe (the report's `dit_load.source` `"host_copy"`) and gives the same image bits.
4. **To give the RAM back as well:** `release` with `keep_host_copy` false. `unload` frees everything, the in-memory embedding cache included (its spill in `cache_dir` stays), and needs a new `load`.

A `release` that keeps a host copy reads the DiT's files into RAM and takes seconds; it is answered when it ends, and a `cancel` meanwhile stops the copy (the VRAM is freed), or the whole release when it lands before anything was freed (the response's `cancelled`: the engine keeps its VRAM). The library has the same calls: `mi_engine_preload()` and `mi_engine_release()`, on the generation thread, with `mi_cancel()` from any thread.

**Errors.**
- **Request errors** arrive as `{"error": {"code", "message"}}`.
  - When the engine rejects a setting, the `invalid_params` message is written for end users and can be shown as it is. Where the official app has a message, the wording is the same (e.g. "Height and width must be positive multiples of 16.").
  - Type errors from the server itself (`"width" must be an integer in [...]`) indicate a launcher bug.
- **Generation errors** arrive in `done` as `"status": "error"`, with the same `{code, message}`.
- **`out_of_memory`:** suggest `min_memory` mode or a smaller image.
- **`cuda`:** restart the server process. CUDA errors can leave a process unable to use the GPU again.
- **`invalid_argument`** (from `load`): the model path does not exist, or a setting is out of range.
- **`model`:** the folder is not a MOSS-Image model folder, or its files are damaged.
- **`io`:** a file could not be read, or the PNG could not be written.
- **`not_supported`:** a feature that this build does not enable, e.g. a ComfyUI file as the model path.
- **`invalid_params` from `load` without a `tier`:** the tier is required since 2.0 (`reference`, `high` or `turbo`).
- **`invalid_argument` in `done`:** a tier the loaded files cannot run, e.g. `reference` with an INT8 `dit_file`. The message names the reason.

**Images from untrusted sources.** moss-server decodes `init_image`, `mask` and `control` image files in-process with stb_image, which is hardened (at most 64 MiB per file, 16384 pixels per side and 64,000,000 pixels, checked before any pixel allocation) but not memory-safe. When the images come from a network, web uploads or other users' files, decode them in the launcher with a memory-safe decoder and pass pixels through the C API (`mi_gen_params.init_image` / `mask`, `mi_control_image.image`), or run moss-server sandboxed. Files the user created or trusts are the case the server's file input is meant for; downloads are untrusted.

**Paths.** The server trusts its input: it reads and writes exactly the paths it is given. The launcher must choose `output` and `cache_dir` itself, or validate them, and must not forward raw paths from untrusted sources (see "Security" in [server-protocol.md](server-protocol.md)).

## Using the library directly

With CMake:

```cmake
find_package(MossInfer 2.0 REQUIRED CONFIG)   # CMAKE_PREFIX_PATH=<unpacked package>
target_link_libraries(launcher PRIVATE MossInfer::mossinfer)
```

Without CMake:
- Add `<package>/include` to the include path.
- Link `libmossinfer.so` (Windows: `mossinfer.lib`).
- Define `MOSSINFER_SHARED`. On Windows this selects `dllimport`.
- No CUDA headers or toolkit are needed.
- At run time, `libmossinfer` loads cuBLASLt from a compatible CUDA install or from its own directory (see [CUDA libraries](#cuda-libraries)), so keep the package's `lib/` (Windows: `bin/`) together.

The minimal C program is [`examples/c/minimal.c`](../examples/c/minimal.c), and [c-api.md](c-api.md) has a quick start. The rules that matter for integration:
- **Check the ABI at startup.** Refuse to run when `mi_abi_version() != MI_ABI_VERSION`. On Windows this is the only protection against a mismatched DLL.
- **Fill every struct with its `*_init(&s, sizeof s)` function** before setting fields. Never zero-initialize a struct yourself and skip `*_init`: the size field is what will let newer libraries accept older programs.
- **One generation per engine at a time.** Run `mi_generate()` on a worker thread, and call `mi_cancel()` from the UI thread.
- **Callbacks run on the generating thread, while the GPU waits.** Post to the UI and return.
- **`MI_CANCELLED` still returns a result** (timings, no image). Free it with `mi_result_free()`.
- **After `MI_ERR_CUDA`, free the engine.** If creating a new one also fails, the process needs a restart. This is the main reason to prefer the sidecar.

Bindings in other languages can follow the frozen struct layouts in [c-api.md](c-api.md#frozen-layout-abi-1), which give every field's offset on 64-bit targets.

## Settings to expose in the launcher

| Launcher setting | Engine field (C API / server `load`) | Recommendation |
|---|---|---|
| Memory mode | `mode`: `min_memory` / `balanced` / `max_speed` | Default `balanced`; it already adapts to free VRAM. Offer `min_memory` for "share the GPU with other apps / games", and `max_speed` for machines with plenty of RAM. |
| VRAM limit | `vram_budget_bytes` (0 = automatic) | Leave it automatic. Expose it as an advanced option, e.g. to keep room for another app. |
| Cache folder | `cache_dir` | **Always set it** to a folder the launcher owns. It holds the prompt-embedding cache and a small hardware profile, which spare later runs a short measurement and a self-check. Several processes may use one folder at once, engines of other versions included: every file is written under a unique temporary name and renamed into place, so none reads a partial file. An embedding entry in another format version is a miss and stays until the size cap ages it out; a hardware profile of another version or GPU is ignored and rewritten, which costs a measurement. Sharing the folder shares embedding-cache hits between engines that compute the same embeddings (the same text encoder, numerics and kind of GPU). It does not share the GPU: each engine plans against the VRAM free when it plans, so two engines on one GPU still compete for memory. |
| GPU | `device_index` / `device` | From `hello`'s device list. Supported devices only. |
| Model | `model_path` / `model` | The Base or SFT folder. |
| Model files | `dit_file`, `text_encoder_file`, `vae_file` | Optional file pickers next to the model folder, empty by default (the folder's own components). Offer them when `MI_CAP_MODEL_FILES` / `model_files` is set. |
| Quality tier | `tier`: `reference` / `high` / `turbo` (required) | The launcher chooses; the engine has no default. Recommended: an INT8 DiT with `high`. Build the menu from the DiT's `tiers` (`mi_inspect_file()` / `inspect`, the engine's own decision) and pre-select `high` for a bf16 or INT8 DiT and `turbo` for an INT4 or NVFP4 one. Show the tier that ran (`mi_result_tier()`, or the report's `quality` and `models`). |
| Distill bundle | `distill_dir` / `distill` (engine) | An optional folder picker. Set the same bundle on the engine and on each generation (the engine takes its time embedding). Leave steps, CFG and the sampler unset to get the bundle's. |
| Attention | `attention`: `default` / `exact` / `q8_fp8` / ... | An advanced setting, default `default` (the tier's default: 8-bit for `high` and `turbo`). |
| LoRA strictness | `lora_strict` | Off. Turn it on only for workflows where a partly applied LoRA must fail instead of warn. |
| Text-encoder pinning | `te_host_pin`: `auto` / `on` / `off` | An advanced setting, default `auto`. A pinned copy of a host-resident text encoder uploads faster and holds page-locked RAM: `off` on machines that need the RAM for other applications, `on` where RAM is plentiful. Offer it when `MI_CAP_TE_HOST_PIN` / `te_host_pin` is set. |
| DiT reads | `dit_direct_io`: `auto` / `buffered` / `direct` | An advanced setting, default `auto` (direct I/O unless the file system gains nothing from it or the files are already cached). `buffered` when the model sits on storage that handles direct I/O badly; `direct_io` (all reads direct) is for cold-cache measurements only. |

**Per-image settings** (`generate` / `mi_gen_params`) mirror the official app's `parameters.json`, and so do the defaults:
- 1024×1024, 50 steps, CFG 3.0, shift 5.0 (steps and CFG are left unset by `mi_gen_params_init()` and an absent server field, and resolve to these unless a distill bundle applies: see [c-api.md](c-api.md#unset-markers));
- seed 42 (`-1` = random; the seed used is in the report);
- the negative branch skips DiT block 22;
- Euler sampler, `moss` scheduler.

Width and height must be multiples of 16, at most 4096 per side, and at most 4.2 megapixels in total. `mi_gen_params_validate()` checks a request without a GPU, which suits live form validation; the server checks a `generate` before it looks at its own state, so it can validate the same way.

| Launcher setting | Engine field (`mi_gen_params` / server `generate`) | Recommendation |
|---|---|---|
| LoRA list | `loras` (`{path, strength}` each, at most 64) | A list per generation with a strength slider each. Show `done.warnings` (`mi_result_message()`) after the run. |
| Sampler and scheduler | `sampler`, `scheduler` | Default Euler + `moss` (the official result). List all names (server: `hello`'s `values`); group the stochastic samplers' `eta` and `s_noise`, and `noise_source`, as advanced options. The `dataspace` sampler needs its parameters file (`dataspace`). |
| Embed settings | `save_metadata` | On by default (`mi_gen_params_init(p, sizeof *p)`; a program built against the 0.1.0 header gets no metadata). |
| Include prompts | `include_prompts` | On by default. Offer it as a privacy option for images that will be shared. |
| Init image | `init_image` (pixels) or `init_image_path` / `init_image` (path) | An optional image picker with a denoise strength slider, pre-filled with `MI_DEFAULT_DENOISE_STRENGTH` (0.75, what an unset strength resolves to; 1 without a mask ignores the image). The output size stays `width` × `height` and the image is resized to it (crop by default): set the size from the image to keep its aspect ratio. `moss-infer` does that itself; the C API and the server do not. |
| Inpainting mask | `mask` / `mask_path` / `mask`, `mask_encoding` | A mask painter next to the init image. Pass 0/1 masks with `binary01`, 0/255 masks with `u8`. |
| Hires fix | `hires_width`, `hires_height`, `hires_strength`, `hires_steps` / `hires` | Off by default; a target size (or a scale rounded to multiples of 16) with a strength (default 0.45). |
| Control adapter and images | `control_adapter`, `control_lora_strength`, `control_images` (`{kind, image or image_path, strength, start, end}` each, at most 16) / `control` | An advanced setting, off by default: an adapter file picker, then one image picker per kind the adapter has (`mi_file_report_control_kind()` / `inspect`'s `control_kinds` before a run, `MI_TEXT_CONTROL_KINDS` / the report's `control.kinds` after one), each with a strength slider (default 1) and a step window (default 0 to 1). The images are resized to the output size like an init image. |
| CFG truncation | `cfg_truncation` | An advanced setting, default 1 (off). Lossy: it changes the image. No effect at CFG 1 (the server's `done.warnings` says so). |
| Step cache | `step_cache` (`off` / `fbcache`), `step_cache_threshold` | An advanced setting, default off; the threshold left unset (0 in the C API, absent on the server) takes the engine's per-model default (0.02 base, 0.03 SFT and derived files). Higher thresholds are faster and change the image more (`mi_accel_info.lossy` and the report's `accelerators.lossy` say an accelerator was requested). |
| Preview | `preview_callback` + `preview_interval` / `preview_output` + `preview_every` | On. Every step is fine for small images; use about 5 for large ones, because each preview is a full-size device-to-host copy plus (server) a PNG write, on the generating thread while the GPU waits. The preview is coarse (no VAE). On Windows, do not keep the server's preview file open: the server renames each new preview over it, a held file makes that fail, and the server then logs a warning and goes on. |

Leave `block_path`, `direct_io`, `noise_layout`, `flow_multiplier` and the skip-layer guidance fields (`slg_*`) at their defaults unless you have a reason to change them. `block_path` changes speed only, never the image.

**Prompt length.** The text encoder keeps at most `max_sequence_length` tokens of the formatted prompt (4096 by default). `mi_result_prompt_tokens()`, the report's `tokens` and `done.warnings` say when a prompt was cut. Show the warning next to the prompt box.

**Reproducibility.** The same model, seed, settings and engine version give the same image on the same GPU and flavour, whatever the VRAM budget and memory mode.

## Preparing for planned features

Features are announced by a capability bit. Check `mi_capabilities()`, or `capabilities` in the server's `hello`, at startup, and enable the matching UI only when the bit is set. The same launcher then works with older and newer builds.

| Capability | C API | `hello` | This build | Enables |
|---|---|---|---|---|
| Preview | `MI_CAP_PREVIEW` (0x1) | `"preview"` | set | `preview_callback` / `preview_output` |
| ComfyUI file as the model path | `MI_CAP_COMFYUI_MODELS` (0x2) | `"comfyui_models"` | unset | `MI_MODEL_FORMAT_COMFYUI` and a file as `model_path` |
| Component model files | `MI_CAP_MODEL_FILES` (0x4) | `"model_files"` | set | `dit_file`, `text_encoder_file`, `vae_file` |
| LoRAs | `MI_CAP_LORA` (0x8) | `"lora"` | set | `loras`, `lora_strict` |
| Tiers and attention | `MI_CAP_TIERS` (0x10) | `"tiers"` | set | `tier`, `attention` |
| Samplers and schedulers | `MI_CAP_SAMPLERS` (0x20) | `"samplers"` | set | the samplers after AB2, `scheduler`, the noise options |
| Image input | `MI_CAP_IMAGE_INPUT` (0x40) | `"image_input"` | set | img2img and inpainting |
| Hires fix | `MI_CAP_HIRES` (0x80) | `"hires"` | set | the hires fix |
| Distill bundles | `MI_CAP_DISTILL` (0x100) | `"distill"` | set | `distill_dir` / `distill`, `time_embedding_file` |
| System CUDA reuse | `MI_CAP_SYSTEM_CUDA` (0x200) | `"system_cuda"` | set | `mi_probe_info.cublaslt_*` / `probe.cublaslt`; `MOSSINFER_CUDA_LIBS`, `MOSSINFER_CUBLASLT` |
| Control adapters | `MI_CAP_CONTROL` (0x400) | `"control"` | set | `control_adapter`, `control_lora_strength`, `control_images` / `control` |
| Lossy accelerators | `MI_CAP_ACCEL` (0x800) | `"accel"` | set | `cfg_truncation`, `step_cache`, `step_cache_threshold` |
| Preload | `MI_CAP_PRELOAD` (0x1000) | `"preload"` | set | `mi_engine_preload()` / `preload` |
| Soft unload | `MI_CAP_RELEASE` (0x2000) | `"release"` | set | `mi_engine_release()` / `release`, the report's `dit_load` |
| Text-encoder pinning | `MI_CAP_TE_HOST_PIN` (0x8000) | `"te_host_pin"` | set | `te_host_pin` |
| File inspection | `MI_CAP_INSPECT` (0x20000) | `"inspect"` | set | `mi_inspect_file()` / `inspect` |

Future features will arrive the same way: **new trailing struct fields, new enum values and new capability bits** in the C API, and **new optional parameters** for the server. Leaving them unset keeps today's behaviour. A program built against an older header keeps working with a newer library, because the library accepts every older struct size (see [Versioning](#versioning)).

## Reports and diagnostics

Every result carries a JSON report: `mi_result_report_json()`, or `done.report` from the server. It holds:
- the request and the seed used;
- the tier that ran, the quality object (configuration, engine numerics, the cuBLASLt copy in use, GPU class) and the model sources;
- the prompt token counts, and the LoRA reports, warnings and notes;
- the execution plan: what was resident, and the memory per phase;
- timings per phase, and VRAM and RAM peaks;
- model streaming statistics and the hardware profile;
- the control adapter and the accelerators (`control`, `accelerators`);
- how the DiT's weights were read (`dit_load`: from the files, direct or buffered, or from a soft unload's host copy), and a preload's `preload_note`.

The library also has accessors for the parts a launcher shows: `mi_result_tier()`, `mi_result_text()`, `mi_result_prompt_tokens()`, `mi_result_lora()`, `mi_result_message()`, `mi_result_control()` and `mi_result_accel()` (see [c-api.md](c-api.md#results)).

Attach it to bug reports, together with the server's stderr log and `hello`'s `probe`.

`moss-infer generate --json-report` writes the same report from the command line. It is the quickest way to tell whether a problem is in the launcher or in the engine.

## Versioning

- **Version numbers.**
  - `mi_version()` returns the release version (`2.0.0`).
  - `mi_abi_version()` and `MI_ABI_VERSION` give the binary interface version (1).
  - The server's `hello` reports its `protocol_version` (2).
- **Additive changes keep ABI 1:** new functions, new fields at the end of structs, new enum values, new optional server parameters and events.
  - A newer library accepts structs from programs built against an older header: their `struct_size` is smaller, the library reads only those bytes, and the missing fields take their defaults. It writes only `struct_size` bytes of output structs.
  - This is implemented: every earlier size of `mi_engine_params` and `mi_gen_params` is accepted ([c-api.md](c-api.md#struct-sizes)); any other size returns `MI_ERR_ABI_MISMATCH`. A 0.1.0 `mi_gen_params` keeps the 0.1.0 behaviour (its PNGs carry no metadata). The 0.1.0 `mi_engine_params` (80 bytes) has no `tier` field, so since 2.0 it creates no engine.
- **Incompatible binary changes raise `MI_ABI_VERSION`.** On Linux the shared library's SONAME follows the ABI (`libmossinfer.so.1`). 2.0 keeps ABI 1: no symbol, field, offset or enum value moved; retired fields stay in place as reserved.
- **Releases with the same MAJOR version are source and binary compatible.** A launcher built against a 2.x header runs with any later 2.y library: additions arrive as new functions, enum values and trailing struct fields (versioned by `struct_size`). A behaviour change that callers must act on raises the MAJOR version, as 2.0 did.
  - **Build time:** the CMake package (`SameMajorVersion`) satisfies a request from any release with the same MAJOR and at least the requested version: `find_package(MossInfer 2.0)` finds every 2.x, but not 1.x or 3.0.
- **The server protocol version** rises when a request that was valid fails, or a documented field goes away: protocol 2 (moss-server 2.0) requires `load`'s `tier` and drops `done.certified`. A `hello` with another `protocol_version` is `unsupported_protocol`, so a 1.x launcher stops at the handshake rather than at its first `load`.
- **Server clients:** unknown parameters are rejected with `invalid_params`, so a typo never goes unnoticed. Send only parameters that the server's version documents. Ignore unknown fields in responses and events, because new ones will appear.

### Migrating from 1.x

2.0 makes the caller choose. The binary layout is unchanged, so a 1.x program loads 2.0 and needs two edits:
- **Pass a tier.** `mi_engine_params.tier` (server: `load`'s `tier`, CLI: `--tier`) is required: `MI_TIER_REFERENCE`, `MI_TIER_HIGH` or `MI_TIER_TURBO`. `MI_TIER_AUTO`, the zero default, now fails `mi_engine_create()` with `MI_ERR_INVALID_ARGUMENT`; a `load` without `tier` is `invalid_params`. What 1.x `auto` ran without a certificate: `REFERENCE` for the bf16 bundle, `HIGH` with `attention` `EXACT` for an INT8 DiT. For speed, use `HIGH` with its default 8-bit attention.
- **Treat the certificate fields as inert.** The engine reads no quality certificates and packages ship none. `certificate_dirs` is ignored, `MI_CAP_CERTIFICATE_DIRS` is never set, `mi_tier_info.certified` and `run_certified` are 0, `MI_TEXT_CERTIFICATE` is `""`, and the server's `done` has no `certified`. The report's `models.dit` has `tier` instead of `tier_requested`, `tier_effective` and `certified`; its `quality` object keeps what ran (tier, configuration, numerics, cuBLASLt, GPU class, model paths) without `reason` or `certificate`.

`hello`'s `values.tier` lists exactly the accepted names, and `inspect`'s `tiers` exactly the tiers `generate` accepts for a DiT. The server protocol is version 2: send `protocol_version` 2 in `hello`.

## Licensing and redistribution

- **MossInfer** is Apache-2.0. Ship `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` with the binaries. The package has them at its root and in `share/doc/mossinfer/`. A launcher's "About / licences" page should show or link them.
- **Third-party code** that is bundled or derived (CUTLASS, FlashAttention-2, SageAttention, comfy-kitchen and others) is listed in `THIRD_PARTY_NOTICES.md`. That file also carries the licence texts that have no file of their own. The vendored licence files are in `share/doc/mossinfer/third_party/<component>/`.
- **cuBLASLt** is redistributed under the NVIDIA CUDA EULA: `share/doc/mossinfer/third_party/cuda/EULA.txt`. The `-system-cuda` archives redistribute no CUDA library and carry no EULA.
- **The NVIDIA driver** is not redistributed.
- **The MOSS-Image weights** are published by OpenMOSS under Apache-2.0. They are downloaded separately, never bundled.
