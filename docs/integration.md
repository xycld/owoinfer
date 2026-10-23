# Integrating MossInfer into a launcher

MossInfer is a native C++/CUDA engine for MOSS-Image text-to-image generation, with no PyTorch or Python at run time. This guide is for teams that build it into an application, such as the MOSS-Image launcher. It covers what is supported, what to ship, how to start and drive the engine, and how to prepare for the features that are coming.

The reference documents are:
- [c-api.md](c-api.md): the C library, `mossinfer.h`.
- [server-protocol.md](server-protocol.md): the `moss-server` sidecar.

This page does not repeat them. It explains how the pieces fit together, and which choices the launcher has to make.

Version 1.0.0 (ABI 1, server protocol 1). See [Versioning](#versioning) for what stays compatible.

## Support status

### Supported now

| Area | What works |
|---|---|
| Platform | Linux x86-64 and Windows x64 (MSVC), each in a `cu12` and a `cu13` flavour, as a full package and a `-system-cuda` package. CI builds, packages and runs the CPU tests of all four flavours; GPU runs are verified on Linux `cu13`. |
| GPUs | NVIDIA compute capability 8.x and 12.x: RTX 30xx, RTX 40xx, RTX 50xx, A100/A10/A40, L4/L40, RTX PRO Blackwell |
| Models | the official MOSS-Image Base and SFT Hugging Face folders (bf16), read in place |
| Component model files | a single-file DiT, text encoder or VAE next to the bundle folder: ComfyUI and community files, INT8 ConvRot and INT4 ConvRot DiT, INT8 ConvRot and W4A8 text encoder (`dit_file`, `text_encoder_file`, `vae_file`) |
| Generation | text-to-image with the official app's parameters: prompt, negative prompt, size, steps, CFG, shift, seed, skip layers, skip-layer guidance |
| Quality tier | `auto` (default: the fastest configuration with a quality certificate; 1.0 ships no certificates, so without one of your own the bf16 bundle runs Reference, and quantized files run their own GEMMs with exact attention, uncertified), `reference` (bf16, exact), `high` (the loaded weights plus 8-bit attention), `turbo` (an INT4 DiT file, chosen explicitly); the tier that ran is reported |
| Image input | img2img (an init image and a denoise strength) and inpainting (a mask), with crop / fit / stretch resizing; pixels (library) or PNG / JPEG files (both surfaces) |
| Hires fix | a second pass at a larger size from the first pass upscaled, with its own strength and steps |
| Distill bundles | few-step distilled checkpoints shipped as a bundle folder (time embedding, LoRA, data-space sampler parameters), applied per engine and per generation |
| Attention mode | exact (FlashAttention-2) or one of four 8-bit modes, as an override of the tier's choice |
| LoRAs | a list of files with strengths per generation, applied at run time: kohya, PEFT / diffusers, LyCORIS LoHa and LoKr, and full deltas; per-file reports and warnings |
| Samplers and schedulers | 9 samplers (Euler, AB2, Euler ancestral, Heun, DPM++ 2M, DPM++ 2M SDE, ER-SDE, RES multistep, data-space), 7 schedulers, noise layout and source, eta, s_noise, flow multiplier |
| Preview | a latent preview image during sampling, every N steps (library: a callback with RGB8; server: a PNG rewritten in place and a `preview` event) |
| PNG metadata | the generation settings embedded in the PNG (`generation` record, schema `mossinfer.generation/1`, plus `parameters`), the prompts optional |
| Prompt token report | token counts per prompt and a truncation warning |
| Memory modes | `min_memory`, `balanced`, `max_speed`, plus explicit VRAM and RAM budgets and a VRAM reserve. The engine plans against the VRAM that is actually free and streams the model when it doesn't fit. |
| Integration surfaces | the C library (`libmossinfer`) and the sidecar process (`moss-server`), with the same features |
| Run-time control | progress, cancel from any thread, one generation at a time per engine, device probe and build recommendation |
| System CUDA reuse | a compatible cuBLASLt from the user's CUDA install, used by the `-system-cuda` package or on request (`MOSSINFER_CUDA_LIBS=system`); the probe reports which copy runs (verified on Linux; see [CUDA libraries](#cuda-libraries)) |
| Output | PNG file (both surfaces), or RGB8 in memory (library), plus a JSON report per image |

### Planned: integrate now, enabled later

These are designed and partly built. The launcher can prepare for them now (see [Preparing for planned features](#preparing-for-planned-features)).

| Feature | State | How the launcher prepares |
|---|---|---|
| System CUDA reuse on Windows | the search paths are written (`CUDA_PATH`, `CUDA_PATH_V<major>_*`), not yet verified on Windows | the same logic as on Linux: read `probe.cublaslt` |
| Control LoRAs and the lossy accelerators (CFG truncation, step cache) | `moss-infer` only in 1.0 (below); the C API and `moss-server` get them in 1.1 | nothing yet; enable them when `mi_capabilities()` or `hello`'s `capabilities` announce them |

**`moss-infer` only in 1.0.** These run from `moss-infer generate`, not through the C API or `moss-server`:
- **Control LoRAs:** `--control <file>` loads a control adapter (a LoRA with a widened DiT input embedder, e.g. the MOSS-Image color / depth control) and `--control-scale S` sets its LoRA strength (default 1). Each `--control-image <kind>=<png|jpg>` gives the input of one of the adapter's kinds, and may be followed by `--control-strength X` (default 1, 0 to 10) and `--control-start X` / `--control-end X` (default 0 / 1: the share of the steps it applies to).
- **Lossy accelerators**, off by default: `--cfg-truncation X` (0 to 1, default 1 = off) drops the negative branch from step X · steps on; `--step-cache fbcache` (default `off`) reuses DiT blocks 1..41 from the last full call while block 0's output changed by less than `--step-cache-threshold` (default 0.03). Both change the image and make the run uncertified. The tier is resolved as if they were off, so `auto` picks the same tier. The step cache's gain depends on the step count and size: at the default threshold it reused about 30 % of the DiT calls at 1024², 50 steps, and none at 768², 30 steps.
- The report records them in `control` and `accelerators`, and `dit_stream.steps[i].cached` marks the steps that reused the cache.

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
| An uncertified Turbo run under tier `auto` | not supported: `auto` refuses an INT4 DiT without a quality certificate (none exists yet). Choose `turbo` explicitly. |
| Pixel data over the server protocol | not supported: the server takes image file paths |
| Server output other than PNG | not supported |

## Choosing the integration surface

| | `moss-server` (sidecar process) | `libmossinfer` (in-process library) |
|---|---|---|
| Launcher language | any: talk JSON over pipes | C, or anything with a C FFI (C++, Rust, C#, Python ctypes, Node N-API) |
| Crash isolation | a GPU fault or driver reset kills only the server | a fault takes down the launcher |
| Releasing VRAM | `unload`, or end the process (always complete) | `mi_engine_free()` |
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
| MOSS-Image Base folder | 26 GB |
| MOSS-Image SFT folder | 26 GB, or about 10 GB extra when its text encoder and VAE files are hard links to Base's (they are byte-identical) |
| Engine package | several hundred MB, most of it NVIDIA cuBLASLt; the `-system-cuda` archive leaves cuBLASLt out |

## Getting the binaries

**From a release.** CI builds two archives per OS and flavour: `mossinfer-<version>-<os>-cu<12|13>-<arch>.zip` (the full package) and `mossinfer-<version>-<os>-cu<12|13>-<arch>-system-cuda.zip` (the same without cuBLASLt); Linux also gets `.tar.gz` of both. A GitHub release attaches the archives with one `SHA256SUMS` file. See [Package contents](#package-contents).

**From source** (Linux shown):

```sh
# CUDA 13.x toolkit, CMake >= 3.28, Ninja, GCC >= 12, GoogleTest
cmake --preset linux-cuda13
cmake --build --preset linux-cuda13 -j8
CUDA_VISIBLE_DEVICES="" ctest --preset linux-cuda13 -L '^cpu$'
cmake --install build-linux-cuda13                # -> build-linux-cuda13/install
cmake --install build-linux-cuda13 --component Unspecified --prefix build-linux-cuda13/install-system-cuda
(cd build-linux-cuda13 && cpack --config CPackConfig.cmake)   # -> mossinfer-1.0.0-linux-cu13-x86_64.{tar.gz,zip}
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
- **Full** (`mossinfer-…-cu13-x86_64.zip`): bundles cuBLASLt and the CUDA EULA. It runs on any machine with a suitable driver, and uses its own cuBLASLt by default (`auto`, see [CUDA libraries](#cuda-libraries)): results are reproducible across machines, and quality certificates measured with that cuBLASLt apply.
- **System CUDA** (`mossinfer-…-cu13-x86_64-system-cuda.zip`): the same files without cuBLASLt and the EULA, several hundred MB smaller. It runs only where a compatible cuBLASLt is installed. It is made from the release presets, which link the CUDA runtime statically; a tree installed the same way from a build with a shared cudart (the dev preset) also needs a system cudart.
- **Users with a CUDA toolkit** can use its cuBLASLt instead in two ways: download the smaller `-system-cuda` archive, or keep the full one and set `MOSSINFER_CUDA_LIBS=system`. A system copy of another cuBLASLt version can change result bits, and certificates measured with the bundled copy do not apply to it (see Reproducibility below).
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
- Quality certificates are bound to the cuBLASLt version they were measured with. A run with another version (a system copy) finds no usable certificate, so tier `auto` runs Reference, and the report's reason names the certificate, the cuBLASLt version it was measured with and the one loaded.
- Each report records the copy (`quality.cublaslt`).
- The default (`auto` in a full package) and `MOSSINFER_CUDA_LIBS=bundled` run the release's own library, so results can be compared across machines.

## Model files

The engine reads the official Hugging Face model folders in place, with no conversion step:

```sh
hf download OpenMOSS-Team/MOSS-Image-Base --local-dir MOSS-Image/base --exclude 'assets/*'
hf download OpenMOSS-Team/MOSS-Image-SFT  --local-dir MOSS-Image/sft  --exclude 'assets/*'
```

- **The model path is the folder** that contains `text_encoder/`, `tokenizer/`, `transformer/` and `vae/`, e.g. `MOSS-Image/base`.
- **The folder can be read-only.** The engine writes nothing into it. An optional `moss.mossidx` index, made with `moss-infer index <folder>`, makes the first load slightly faster. Without one, the engine builds the index in memory.
- **To save disk with both models,** hard-link SFT's `text_encoder/*.safetensors` and `vae/*.safetensors` to Base's. They are byte-identical.
- **ComfyUI and quantized files** load as component files next to the bundle folder: `dit_file` (INT8 ConvRot, INT4 ConvRot, mixed INT8/bf16 or bf16 DiT), `text_encoder_file` (INT8 ConvRot or W4A8) and `vae_file`. The tokenizer and configs still come from the folder. A file as the model path itself (`MI_MODEL_FORMAT_COMFYUI`, or a file with `MI_MODEL_FORMAT_AUTO`) returns `MI_ERR_NOT_SUPPORTED`, and the server answers `not_supported`, while `MI_CAP_COMFYUI_MODELS` is unset.
- **Quantized weights are lossy.** An INT8 DiT runs the `high` tier's INT8 GEMMs, and `reference` refuses it. An INT4 DiT runs only under `turbo`, which must be chosen explicitly: offer it for INT4 files, and only for them. A quality certificate (`<dit>.mossinfer-cert.json`, or in `<cache_dir>/quality_certs/`; 1.0 ships none) lets `auto` pick a lossy configuration; the report's `quality` object says which one ran and whether it was certified.

## Picking the build: cu12 or cu13

Ask the engine. `mi_probe()`, or the sidecar's `hello`, works on machines without a driver or GPU, and leaves no CUDA context behind. A `cu13` build also answers correctly on a CUDA 12 driver.

```
>> {"id": 1, "method": "hello", "params": {"client": "launcher", "protocol_version": 1, "devices": true}}
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
1. `hello`: check `protocol_version == 1` and `abi_version`, and pick the flavour (see above).
2. `load` `{"model": "<folder>", "mode": "balanced", "cache_dir": "<launcher cache dir>/mossinfer"}`. This returns quickly. The weights load during `generate`: the DiT in the first `dit_load` phase, and the text encoder streams in `text_encoder` whenever the prompt is not in the embedding cache.
3. `generate` `{"prompt": ..., "output": "<path>.png", ...}`. The answer is immediate: `{"accepted": true}`. Then come `plan` once, `progress` at each phase start and after each step, `preview` before a step's `progress` when `preview_output` is set, and finally `done` with `status` `ok`, `cancelled` or `error`. Show `done.warnings` (LoRA losses, a truncated prompt) next to the image.
4. `cancel` `{"id": <the generate request's id>}` from the UI's stop button. The id must match exactly, JSON type included (`"g1"` is not `1`). The generation stops within one block or layer, and `done` arrives with `"status": "cancelled"`.
5. `unload` frees the VRAM while the launcher stays open. `load` with different settings replaces the engine: the old one is released first, so if the new `load` fails, no engine is loaded.
6. `shutdown` on exit. The server cancels any running generation, answers, and exits with status 0. Kill it if it hasn't exited after a few seconds.

**Progress bar.**
- `progress` events report the `phase`: `text_encoder`, then `dit_load`, `denoise` and `decode`. `denoise` also reports `step` and `total_steps`.
- `denoise` takes most of the time. `text_encoder` also takes a few seconds for each prompt not in the embedding cache, and so does the first `dit_load`.
- A simple bar is `step / total_steps` during `denoise`, with the other phases shown as indeterminate.

**One generation at a time.** A second `generate` while one runs is answered with `busy`, so queue requests in the launcher. `done` is written only after the server is idle again, so the next `generate` can go out as soon as `done` is read.

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
- **`invalid_argument` in `done`:** a tier combination the engine refuses, e.g. `reference` with an INT8 `dit_file`. The message names the conflict.

**Images from untrusted sources.** moss-server decodes `init_image` and `mask` files in-process with stb_image, which is hardened (at most 64 MiB per file, 16384 pixels per side and 64,000,000 pixels, checked before any pixel allocation) but not memory-safe. When the images come from a network, web uploads or other users' files, decode them in the launcher with a memory-safe decoder and pass pixels through the C API (`mi_gen_params.init_image` / `mask`), or run moss-server sandboxed. Files the user created or trusts are the case the server's file input is meant for; downloads are untrusted.

**Paths.** The server trusts its input: it reads and writes exactly the paths it is given. The launcher must choose `output` and `cache_dir` itself, or validate them, and must not forward raw paths from untrusted sources (see "Security" in [server-protocol.md](server-protocol.md)).

## Using the library directly

With CMake:

```cmake
find_package(MossInfer 1.0 REQUIRED CONFIG)   # CMAKE_PREFIX_PATH=<unpacked package>
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
| Cache folder | `cache_dir` | **Always set it** to a folder the launcher owns. It holds the prompt-embedding cache and a small hardware profile, which spare later runs a short measurement and a self-check. |
| GPU | `device_index` / `device` | From `hello`'s device list. Supported devices only. |
| Model | `model_path` / `model` | The Base or SFT folder. |
| Model files | `dit_file`, `text_encoder_file`, `vae_file` | Optional file pickers next to the model folder, empty by default (the folder's own components). Offer them when `MI_CAP_MODEL_FILES` / `model_files` is set. |
| Quality tier | `tier`: `auto` / `reference` / `high` / `turbo` | Default **Auto**. Show the tier that ran (`mi_result_tier()`, or the report's `quality` and `models`), and mark uncertified lossy runs. Offer `turbo` only when the DiT file is INT4, and select it for such a file: `auto` refuses an uncertified INT4 DiT. |
| Distill bundle | `distill_dir` / `distill` (engine) | An optional folder picker. Set the same bundle on the engine and on each generation (the engine takes its time embedding). Leave steps, CFG and the sampler unset to get the bundle's. |
| Attention | `attention`: `default` / `exact` / `q8_fp8` / ... | An advanced setting, default `default` (the tier's choice). |
| LoRA strictness | `lora_strict` | Off. Turn it on only for workflows where a partly applied LoRA must fail instead of warn. |

**Per-image settings** (`generate` / `mi_gen_params`) mirror the official app's `parameters.json`, and so do the defaults:
- 1024×1024, 50 steps, CFG 3.0, shift 5.0 (steps and CFG are left unset by `mi_gen_params_init()` and an absent server field, and resolve to these unless a distill bundle applies: see [c-api.md](c-api.md#unset-markers));
- seed 42 (`-1` = random; the seed used is in the report);
- the negative branch skips DiT block 22;
- Euler sampler, `moss` scheduler.

Width and height must be multiples of 16, at most 4096 per side, and at most 4.2 megapixels in total. `mi_gen_params_validate()` checks a request without a GPU, which suits live form validation; the server checks a `generate` before it looks at its own state, so it can validate the same way.

| Launcher setting | Engine field (`mi_gen_params` / server `generate`) | Recommendation |
|---|---|---|
| LoRA list | `loras` (`{path, strength}` each, at most 64) | A list per generation with a strength slider each. Show `done.warnings` (`mi_result_message()`) after the run. |
| Sampler and scheduler | `sampler`, `scheduler` | Default Euler + `moss` (the official result). List all names; group the stochastic samplers' `eta` and `s_noise`, and `noise_source`, as advanced options. The `dataspace` sampler needs its parameters file (`dataspace`). |
| Embed settings | `save_metadata` | On by default (`mi_gen_params_init(p, sizeof *p)`; a program built against the 0.1.0 header gets no metadata). |
| Include prompts | `include_prompts` | On by default. Offer it as a privacy option for images that will be shared. |
| Init image | `init_image` (pixels) or `init_image_path` / `init_image` (path) | An optional image picker with a denoise strength slider, pre-filled with `MI_DEFAULT_DENOISE_STRENGTH` (0.75, what an unset strength resolves to; 1 without a mask ignores the image). The output size stays `width` × `height` and the image is resized to it (crop by default): set the size from the image to keep its aspect ratio. `moss-infer` does that itself; the C API and the server do not. |
| Inpainting mask | `mask` / `mask_path` / `mask`, `mask_encoding` | A mask painter next to the init image. Pass 0/1 masks with `binary01`, 0/255 masks with `u8`. |
| Hires fix | `hires_width`, `hires_height`, `hires_strength`, `hires_steps` / `hires` | Off by default; a target size (or a scale rounded to multiples of 16) with a strength (default 0.45). |
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

Future features will arrive the same way: **new trailing struct fields, new enum values and new capability bits** in the C API, and **new optional parameters** for the server. Leaving them unset keeps today's behaviour. A program built against an older header keeps working with a newer library, because the library accepts every older struct size (see [Versioning](#versioning)).

## Reports and diagnostics

Every result carries a JSON report: `mi_result_report_json()`, or `done.report` from the server. It holds:
- the request and the seed used;
- the tier that ran, the quality object (certificate, configuration, engine numerics, the cuBLASLt copy in use) and the model sources;
- the prompt token counts, and the LoRA reports, warnings and notes;
- the execution plan: what was resident, and the memory per phase;
- timings per phase, and VRAM and RAM peaks;
- model streaming statistics and the hardware profile;
- the control adapter and the accelerators (`control`, `accelerators`), which only `moss-infer` sets in 1.0.

The library also has accessors for the parts a launcher shows: `mi_result_tier()`, `mi_result_text()`, `mi_result_prompt_tokens()`, `mi_result_lora()` and `mi_result_message()` (see [c-api.md](c-api.md#results)).

Attach it to bug reports, together with the server's stderr log and `hello`'s `probe`.

`moss-infer generate --json-report` writes the same report from the command line. It is the quickest way to tell whether a problem is in the launcher or in the engine.

## Versioning

- **Version numbers.**
  - `mi_version()` returns the release version (`1.0.0`).
  - `mi_abi_version()` and `MI_ABI_VERSION` give the binary interface version (1).
  - The server's `hello` reports its `protocol_version` (1).
- **Additive changes keep ABI 1:** new functions, new fields at the end of structs, new enum values, new optional server parameters and events.
  - A newer library accepts structs from programs built against an older header: their `struct_size` is smaller, the library reads only those bytes, and the missing fields take their defaults. It writes only `struct_size` bytes of output structs.
  - This is implemented: 1.0.0 accepts the 0.1.0 sizes of `mi_engine_params` (80 bytes) and `mi_gen_params` (168 bytes) as well as its own (136 and 320, and 120 and 240 from an earlier 0.2.0 header). Any other size returns `MI_ERR_ABI_MISMATCH`. A 0.1.0 program keeps the 0.1.0 behaviour: its PNGs carry no metadata. See [c-api.md](c-api.md#struct-sizes).
- **Incompatible changes raise `MI_ABI_VERSION`.** On Linux the shared library's SONAME follows the ABI (`libmossinfer.so.1`).
- **From 1.0, releases with the same MAJOR version are source and binary compatible.** A launcher built against a 1.x header runs with any later 1.y library: additions arrive as new functions, enum values and trailing struct fields (versioned by `struct_size`). Only a new MAJOR version may raise the ABI.
  - **Build time:** the CMake package (`SameMajorVersion`) satisfies a request from any release with the same MAJOR and at least the requested version: `find_package(MossInfer 1.0)` finds every 1.x, but not 2.0. Request the oldest 1.x whose header has what you use.
- **Server clients:** unknown parameters are rejected with `invalid_params`, so a typo never goes unnoticed. Send only parameters that the server's version documents. Ignore unknown fields in responses and events, because new ones will appear.

## Licensing and redistribution

- **MossInfer** is Apache-2.0. Ship `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` with the binaries. The package has them at its root and in `share/doc/mossinfer/`. A launcher's "About / licences" page should show or link them.
- **Third-party code** that is bundled or derived (CUTLASS, FlashAttention-2, SageAttention, comfy-kitchen and others) is listed in `THIRD_PARTY_NOTICES.md`. That file also carries the licence texts that have no file of their own. The vendored licence files are in `share/doc/mossinfer/third_party/<component>/`.
- **cuBLASLt** is redistributed under the NVIDIA CUDA EULA: `share/doc/mossinfer/third_party/cuda/EULA.txt`. The `-system-cuda` archives redistribute no CUDA library and carry no EULA.
- **The NVIDIA driver** is not redistributed.
- **The MOSS-Image weights** are published by OpenMOSS under Apache-2.0. They are downloaded separately, never bundled.
