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
  - The library rejects a `struct_size` it does not know with `MI_ERR_ABI_MISMATCH`.
  - Structs only grow at the end. A newer library will accept every older size and default the missing fields.
  - Output structs (`mi_timings`, `mi_memory_stats`, `mi_probe_info`, `mi_device_info`) need `struct_size` set by the caller before the call.
  - Padding is explicit: the gaps are `int32_t _reserved*` fields (see [Frozen layout](#frozen-layout-abi-1)). `*_init()` zeroes them. `mi_engine_create()`, `mi_generate()` and `mi_gen_params_validate()` reject a nonzero one with `MI_ERR_INVALID_ARGUMENT`. The library zeroes them in output structs.
- **Enums** are 32-bit: each one has a `*_MAX_ENUM = 0x7fffffff` member. Unknown values are rejected with `MI_ERR_INVALID_ARGUMENT`.
- **Errors.** Every fallible call returns `mi_status`. `mi_last_error_message()` returns the reason for the last failure on the calling thread. It is thread-local, never NULL, not cleared by successful calls, and valid until the next failing call on that thread.
  - Recording a message never throws. If copying it fails for lack of memory, the message is a fixed "out of memory while recording the error message".
  - `mi_engine_free()` and `mi_result_free()` never fail and never throw, and they leave the message alone. A CUDA error while releasing device memory is ignored.
  - Validation messages use the official app's wording, e.g. "Height and width must be positive multiples of 16."
  - Accessors without a status (`mi_result_width()` and the like) return 0 / NULL for a NULL handle.
- **No C++ crosses the boundary.** Every entry point catches every exception and returns a status.
- **Versions.** `MI_VERSION_*` and `MI_ABI_VERSION` in the header are the only source of the version (`cmake/version.cmake` reads them). At runtime, `mi_version()` and `mi_abi_version()` report the library that was actually loaded. A consumer must refuse a library whose ABI differs from the header's.

| Status | Meaning |
|---|---|
| `MI_OK` (0) | success |
| `MI_CANCELLED` (1) | `mi_generate()` stopped by `mi_cancel()`. A result is still returned: no image, but timings and the report. |
| `MI_ERR_INVALID_ARGUMENT` | NULL handle or pointer, value out of range, invalid request |
| `MI_ERR_ABI_MISMATCH` | unknown `struct_size` |
| `MI_ERR_NOT_SUPPORTED` | not in this build, e.g. ComfyUI checkpoints (`MI_MODEL_FORMAT_COMFYUI`) |
| `MI_ERR_BUSY` | the engine is already generating |
| `MI_ERR_OUT_OF_MEMORY` | the request fits the device or host budgets in no mode (planner), or a host allocation failed |
| `MI_ERR_MODEL` | model files missing, incomplete or malformed |
| `MI_ERR_IO` | file-system error: reading weights, writing the PNG |
| `MI_ERR_CUDA` | CUDA error or no usable device. Free the engine afterwards. |
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
1. Checks the parameters and the model path (the four bundle folders `text_encoder`, `tokenizer`, `transformer` and `vae`). This happens before CUDA is touched.
2. Selects the device.
3. Creates the pipeline.

The weights load on the first `mi_generate()`. The DiT stays resident across generations when the plan allows it, so the first call pays the load (about 7 s warm on the dev box) and later ones do not.

| `mi_engine_params` field | Default | Meaning |
|---|---|---|
| `model_path` | — (required) | bundle directory |
| `model_format` | `AUTO` | `AUTO`: a directory is an HF bundle, a file is a ComfyUI checkpoint (not supported yet) |
| `mode` | `BALANCED` | `MIN_MEMORY` / `BALANCED` / `MAX_SPEED` (planner objective) |
| `device_index` | 0 | CUDA ordinal |
| `vram_budget_bytes` | 0 = free memory − reserve | cap on the process's device memory, CUDA context included; nonzero values in [1 MiB, 1 TiB] |
| `vram_reserve_bytes` | 384 MiB | device memory left outside the arena |
| `host_ram_budget_bytes` | 0 = the mode's target | cap on private host memory |
| `cache_dir` | NULL = memory only | embedding-cache disk spill |
| `embedding_cache` | 1 | reuse the text embeddings of repeated prompts |
| `direct_io` | 0 | O_DIRECT weight reads |

## Generation

`mi_gen_params` mirrors the asset `parameters.json`:
- `prompt`, `negative_prompt`
- `width`, `height`, `steps`, `cfg`, `shift`, `seed` (-1 = random)
- `uncond_skip_layers` (default `{22}`)
- `slg_layers`, `slg_scale`, `slg_start`, `slg_stop`
- `max_sequence_length`
- `sampler` (`EULER` / `AB2`)

`mi_gen_params_validate()` performs the same checks as `mi_generate()` without an engine or GPU. A launcher can call it as the user types.

The callbacks:
- **progress** — `mi_progress {phase, step, total_steps, step_seconds, elapsed_seconds}`. Called at each phase start (`TEXT_ENCODER`, `DIT_LOAD`, `DENOISE`, `DECODE`, in that order; the first two are entered even on a cache hit or a resident DiT) and after every sampling step. `total_steps` is the schedule's step count after snapping, which can be below `steps`.
- **plan** — called once per generation, after planning and before the first phase. It receives the plan as human-readable lines and as JSON.
- **preview** — accepted, but never called while `mi_capabilities()` lacks `MI_CAP_PREVIEW`. This build has no latent→RGB preview.

## Results

| Accessor | Returns |
|---|---|
| `mi_result_rgb(r, &bytes)` | row-major RGB8 `[height][width][3]`, or NULL when cancelled |
| `mi_result_width/height/seed/steps_run/steps_scheduled/cancelled` | scalars |
| `mi_result_timings(r, &t)` | `total, load, encode, denoise, decode` and their parts, in seconds |
| `mi_result_memory(r, &m)` | arena capacity and high-water, device peak, host private start and peak |
| `mi_result_plan_explanation(r)` | the execution plan as text |
| `mi_result_report_json(r)` | the `moss-infer generate --json-report` document; `"image"` is the last path given to `mi_save_png()` |
| `mi_save_png(r, path)` | atomic PNG write (temporary file, then rename) |

Strings returned by accessors are owned by the result. They stay valid until `mi_result_free()`; the report string stays valid until the next `mi_result_report_json()` or `mi_save_png()` call on the same result.

## Device probe

`mi_probe()` reports:
- the driver's CUDA version and this build's version, and whether they are compatible
- the device counts
- this library's flavour and the flavour the machine should install (`"cu13"`, `"cu12"` or `""`)
- a summary message

It works on machines without a driver or GPU and creates no CUDA context.

`mi_probe_device(i)` adds the name, PCI bus id, compute capability, total and free memory, and whether this build supports the device (and why not). It is implemented by `src/backend/probe.h`. It also leaves no CUDA context behind:
- **Free memory comes from NVML** (`libnvidia-ml.so.1` / `nvml.dll`, loaded at run time; `nvmlDeviceGetMemoryInfo` by PCI bus id). NVML needs no context.
- **Fallback without NVML:** the driver API (`libcuda.so.1` / `nvcuda.dll`). It retains the device's primary context, calls `cuMemGetInfo` and releases the context again. The last release destroys it. If an engine already holds that context, the probe shares it and creates nothing.
- **Never** through the CUDA runtime (whose context would stay for the life of the process), and never `cudaDeviceReset`.
- `mi::backend::DeviceInfo::free_memory_source` records which method answered (`"nvml"` or `"driver"`). On the dev box it is NVML.

## Frozen layout (ABI 1)

These are the layouts on 64-bit targets (LP64 and LLP64; 32-bit is not supported). Every field sits at its natural alignment, and every gap is a named `_reserved*` field, so GCC, Clang and MSVC produce the same layout from C and from C++.
- `src/api/c_api.cpp` `static_assert`s each size and offset.
- The C test `c_abi_layout` (`tests/c/test_abi_layout.c`, label `cpu`) checks the same table from a C11 compiler.

The enums are 4 bytes. ABI 1 is defined by this table: a change to it needs `MI_ABI_VERSION` 2.

| Struct | Size | Fields: offset |
|---|---|---|
| `mi_device_info` | 592 | `struct_size` 0, `index` 8, `name[256]` 12, `pci_bus_id[32]` 268, `compute_major` 300, `compute_minor` 304, `_reserved0` 308, `total_memory_bytes` 312, `free_memory_bytes` 320, `supported` 328, `unsupported_reason[256]` 332, `_reserved1` 588 |
| `mi_probe_info` | 576 | `struct_size` 0, `driver_cuda_version` 8, `runtime_cuda_version` 12, `driver_compatible` 16, `device_count` 20, `supported_device_count` 24, `build_flavour[16]` 28, `recommended_flavour[16]` 44, `message[512]` 60, `_reserved0` 572 |
| `mi_engine_params` | 72 | `struct_size` 0, `model_path` 8, `model_format` 16, `mode` 20, `device_index` 24, `_reserved0` 28, `vram_budget_bytes` 32, `vram_reserve_bytes` 40, `host_ram_budget_bytes` 48, `cache_dir` 56, `embedding_cache` 64, `direct_io` 68 |
| `mi_progress` | 40 | `struct_size` 0, `phase` 8, `step` 12, `total_steps` 16, `_reserved0` 20, `step_seconds` 24, `elapsed_seconds` 32 |
| `mi_gen_params` | 168 | `struct_size` 0, `prompt` 8, `negative_prompt` 16, `width` 24, `height` 28, `steps` 32, `_reserved0` 36, `cfg` 40, `shift` 48, `seed` 56, `uncond_skip_layers` 64, `uncond_skip_layers_count` 72, `slg_layers` 80, `slg_layers_count` 88, `slg_scale` 96, `slg_start` 104, `slg_stop` 112, `max_sequence_length` 120, `sampler` 124, `progress_callback` 128, `plan_callback` 136, `preview_callback` 144, `preview_interval` 152, `_reserved1` 156, `user_data` 160 |
| `mi_timings` | 104 | `struct_size` 0, then 12 doubles from 8 in header order (`total_s` … `vae_load_s` at 96) |
| `mi_memory_stats` | 56 | `struct_size` 0, then 6 `uint64_t` from 8 in header order (`arena_capacity_bytes` … `host_private_peak_bytes` at 48) |

The padding was made explicit on 2026-10-21, before any release. `MI_ABI_VERSION` stays 1, and binaries built against the earlier header must be rebuilt.

## Symbol visibility

| Defines | Effect |
|---|---|
| `MOSSINFER_SHARED` | consuming the shared library: `dllimport` on Windows. The CMake package adds it. |
| `MOSSINFER_SHARED` + `MOSSINFER_BUILD` | building the shared library: `dllexport` |
| neither | static linking |

On ELF platforms `MOSSINFER_API` is always `visibility("default")`. The shared library compiles the core with hidden visibility and a version script (`mi_*` only).

## Implementation

| File | Role |
|---|---|
| `src/api/c_api.cpp` | the entry points, exception → status mapping, the busy/cancel state |
| `src/api/pipeline_adapter.{h,cpp}` | the only code that touches `pipeline::Pipeline`: converts the C structs into `Request` / `PipelineOptions`, and flattens `pipeline::Result` into C-API types and the JSON report |

When the `Pipeline` fields change (P4), the adapter is the whole rebase.

## Tests

Label `cpu` (`tests/unit/test_c_api.cpp`):
- **`CApi.*`** covers versions and ABI, defaults against `pipeline::Request` / `PipelineOptions`, `struct_size` checks, reserved fields, NULL handles, invalid generation and engine parameters (with messages), the thread-local error, `mi_engine_free` leaving the error alone, and the probe.
- **`CApiReport.*`** renders the report from a synthetic result under a comma-decimal C++ locale with digit grouping, and under a comma-decimal C locale where the system has one. It checks for valid JSON, exact numbers, and `null` for NaN and infinity.

Label `gpu`:
- **`GpuCApiProbe.*`**: after `mi_probe_device()`, the process holds no CUDA context. This is checked through the driver's primary-context state and NVML's process list.
- **`GpuCApi.*`** covers the device probe, and a 256², 2-step generation with callbacks, a concurrent call rejected as busy, and PNG and report checks. It also covers cancel from another thread with no carry-over, and the caller's current context restored by create, generate and free.

Run the GPU tests on the dev box under the GPU lock:

```
GTEST_FILTER='GpuCApi.*' flock /workspace/Projects/MossInfer/.gpu.lock <build>/mossinfer_tests
```
