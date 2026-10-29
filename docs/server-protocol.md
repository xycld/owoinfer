# moss-server protocol (version 2)

`moss-server` is the engine as a sidecar process for launchers, in the role `llama-server` plays for llama.cpp.
- The launcher spawns it and talks **JSON lines over stdin/stdout**. There are no ports, no firewall prompts and no extra dependencies.
- It is built only on the public C API (`docs/c-api.md`), linked against `libmossinfer` when the shared library is built.
- Source: `tools/server/main.cpp`. Target: `moss-server` (full builds; `cmake/modules/server.cmake`). Tests: `tests/server/test_protocol.py`.

```
moss-server                     # speak the protocol on stdin/stdout
moss-server --parent-pid <pid>  # the same, and exit when process <pid> (the launcher) exits
moss-server --version           # "moss-server 2.1.0 (protocol 2, ABI 1)"
```

### `--parent-pid`

A launcher that crashes or is killed normally closes the server's stdin, and the server exits at end of input. That fails when the pipe's write end survives elsewhere, for example inherited by another child of the launcher. `--parent-pid <pid>` covers that case: the server exits with status 3 (and a line on stderr) when process `<pid>` exits.
- **Windows:** `OpenProcess(SYNCHRONIZE)` + `WaitForSingleObject`.
- **POSIX:** a poll every 250 ms.
  - When `<pid>` is the server's own parent, it watches for re-parenting (`getppid()`), which happens even while the dead parent is an unreaped zombie.
  - For any other process, it calls `kill(pid, 0)`.
- **Linux, when `<pid>` is the parent:** `PR_SET_PDEATHSIG(SIGTERM)` as well, so the exit is immediate. The kernel sends it when the parent *thread* that spawned the server exits, not the whole process. A launcher must therefore spawn `moss-server` from a thread that lives as long as the launcher (its main thread, say), or omit `--parent-pid`.

If `<pid>` is not running at startup, the server exits with status 1. A malformed value is a usage error (status 2).

## Security

The server trusts its launcher completely: whoever writes to its stdin has the server's file-system rights. The protocol is a local pipe between a launcher and the child it spawned, not an interface for untrusted input.
- **No network surface.** The server opens no socket and listens on no port. Its only input is stdin.
- **Paths are used as given.** `generate`'s `output` and `preview_output` are written wherever they point, and an existing file there is overwritten (atomically: a temporary file renamed over it). `load`'s `model`, `dit_file`, `text_encoder_file`, `vae_file`, `time_embedding_file`, `distill` and `cache_dir`, and `generate`'s `loras` paths, `dataspace`, `init_image`, `mask`, `distill` and `control` (the adapter and image paths), are read, and the cache directory is written to. `inspect` reads its `path` (only a regular file is opened; its header and small tensors, with the bounds of [c-api.md](c-api.md#file-inspection)). The server does not confine any path to a directory, resolve symlinks, or check ownership.
- **The launcher validates paths.** A launcher that takes any of these paths from a user, a web page, a workflow file or another program must check them first. For example, it can confine outputs to its own output directory, reject `..` components and symlinks it did not create, and refuse to overwrite files it does not own.
- **Run it as the user.** Run the server with the launcher's own (unprivileged) rights, never elevated.
- **Images are decoded in-process.** `generate`'s `init_image`, `mask` and `control` image paths are PNG / JPEG files that moss-server decodes itself, with the vendored stb_image decoder. That decoder is hardened (at most 64 MiB per file, at most 16384 pixels per side and 64,000,000 pixels in all, checked from the header before any pixel allocation; fuzzed), but it is not memory-safe: a crafted file can exploit it.
  - The server takes image paths only, never pixel data over the protocol.
  - A launcher that takes images from an untrusted source (a network, web uploads, files from other users) must decode them itself with a memory-safe decoder and use the C API's pixel input (`mi_gen_params.init_image` / `mask`, `mi_control_image.image`, [c-api.md](c-api.md#image-input)), or run moss-server sandboxed (no network, no access beyond its input and output directories).

## Framing

- **One message per line.** Each message is a JSON object encoded as UTF-8 and terminated by `\n` (`\r\n` is accepted).
  - Blank lines are ignored.
  - A line longer than 1 MiB is rejected.
  - Input that is not valid UTF-8 is rejected as `parse_error`.
- **stdout carries protocol lines only.** At startup the server keeps a private copy of fd 1 for the protocol and points fd 1 at stderr. Anything else that prints to stdout therefore lands on stderr: the engine, CUDA, a library. In detail:
  - **fds 0–2:** any that the launcher left closed are first opened on the null device (`/dev/null`, `NUL`), so no later file can take their place.
  - **The protocol copy** is fd 3 or higher, and is not inherited by child processes: `F_DUPFD_CLOEXEC`, or on Windows `_dup` + `SetHandleInformation(HANDLE_FLAG_INHERIT, 0)`.
  - **Unusable stderr:** when stderr is the protocol stream itself (`2>&1`; not a terminal), fds 1 and 2 both go to the null device. On Windows, the same applies when the stderr handle is invalid.
- **Logs** go to stderr, or arrive as `log` events.
- **Exit.** The server exits with status 0 after `shutdown` or at end of input (stdin closed). In both cases a running generation is cancelled first.

## Messages

| Direction | Shape |
|---|---|
| request | `{"id": <string, integer or null>, "method": "<name>", "params": {...}}`. `params` is optional; no other members are allowed. |
| response | `{"id": <same id>, "result": {...}}` or `{"id": <same id>, "error": {"code": "<code>", "message": "<text>"}}` |
| event | `{"event": "progress" \| "plan" \| "preview" \| "done" \| "log", "id": <id of the generate or preload request>, ...}` |

Every request gets exactly one response. Responses to different requests can interleave with events, so a client should match responses by `id`. A request whose `id` could not be read (malformed JSON, for example) is answered with `"id": null`. Most responses come in request order; a `release` and an `inspect` are answered when they end, so requests sent meanwhile (`status`, `cancel`, another `inspect`) may be answered first.

Unknown members in `params` are rejected (`invalid_params`), so a typo cannot be silently ignored. So is a member given twice, in `params` and in its objects (`loras` entries, `hires`, `control` and its images), because one of its values would be lost.

### Error codes

| Code | When |
|---|---|
| `parse_error` | malformed JSON, invalid UTF-8, nesting deeper than 128 |
| `invalid_request` | not an object; missing or non-string `method`; bad `id` type; unknown top-level member; line over 1 MiB |
| `method_not_found` | unknown `method` |
| `invalid_params` | wrong parameter type or value, unknown parameter, a string containing NUL (`\u0000`: the C API's strings are NUL-terminated), or a generation request the engine rejects (the message uses the official app's wording) |
| `unsupported_protocol` | `hello` asked for another protocol version |
| `no_engine` | a valid `generate`, `preload` or `release` before `load` |
| `busy` | a valid `generate`, `preload` or `release`, or `load` or `unload`, while a job (a generation, preload or release) runs; an `inspect` while two run |
| `invalid_argument`, `abi_mismatch`, `not_supported`, `out_of_memory`, `model`, `io`, `cuda`, `internal` | the engine's `mi_status` (`MI_ERR_<CODE>`), with `mi_last_error_message()` as the message |

## Methods

### `hello`

Params (all optional):
- `client` (string)
- `protocol_version` (integer; `unsupported_protocol` unless it is 2)
- `devices` (bool): adds the per-device list, with free memory from NVML or a short-lived driver context. No CUDA context remains afterwards (`mi_probe_device()`, docs/c-api.md).

```
>> {"id": 1, "method": "hello", "params": {"client": "launcher", "protocol_version": 2}}
<< {"id": 1, "result": {"protocol_version": 2, "server": "moss-server", "version": "2.0.0", "abi_version": 1,
    "capabilities": {"preview": true, "comfyui_models": false, "model_files": true, "lora": true, "tiers": true,
                     "samplers": true, "image_input": true, "hires": true, "distill": true, "system_cuda": true,
                     "control": true, "accel": true, "preload": true, "release": true, "certificate_dirs": false,
                     "te_host_pin": true, "progress_fraction": true, "inspect": true},
    "values": {"mode": ["min_memory", "balanced", "max_speed"], "block_path": ["default", "op_by_op", "fused", "fused_graphs"],
               "tier": ["reference", "high", "turbo"], "attention": ["default", "exact", ...], ...,
               "sampler": ["euler", "ab2", ...], ..., "step_cache": ["off", "fbcache"]},
    "probe": {"driver_cuda_version": 13040, "runtime_cuda_version": 13040, "driver_compatible": true,
              "device_count": 1, "supported_device_count": 1, "build_flavour": "cu13", "recommended_flavour": "cu13",
              "message": "recommended flavour cu13 (the driver supports CUDA 13.4)",
              "cublaslt": {"source": "bundled", "version": 130800,
                           "path": "/opt/mossinfer/lib/libcublasLt.so.13", "error": ""}}}}
```

With `"devices": true`, `probe.devices` is a list of objects with these fields:
- `index`, `name`, `pci_bus_id`, `compute_capability` (e.g. `"8.9"`)
- `total_memory_bytes`, `free_memory_bytes`
- `supported`, `unsupported_reason`

`hello` works without a GPU or driver; the reason is then in `message`.

`probe.cublaslt` is the cuBLASLt copy the engine uses (`mi_probe_info.cublaslt_*`, docs/c-api.md):
- `source`: `"system"`, `"bundled"` or `"env"`; `""` when no copy is usable
- `version`: `cublasLtGetVersion()` of that copy, e.g. 130800 for 13.8.0; 0 when none
- `path`: the file loaded, with symbolic links resolved; `""` when none (or too long for the C field, with a note in `error`)
- `error`: why no copy is usable (every candidate tried); `""` when one is. `load` then fails with `cuda` and the same message.

The server's environment steers the search: `MOSSINFER_CUDA_LIBS` (`auto`: the bundled copy, then a system one; `system`; `bundled`) and `MOSSINFER_CUBLASLT` (docs/integration.md, "CUDA libraries"). Set them when starting the process. To learn whether the machine has a compatible system copy, start the server with `MOSSINFER_CUDA_LIBS=system` and read `probe.cublaslt`.

`capabilities` mirrors the C API's `mi_capabilities()` bits. A launcher enables a feature's UI only when its capability is true.

| Capability | C bit | This build | Covers |
|---|---|---|---|
| `preview` | `MI_CAP_PREVIEW` | true | `generate`'s `preview_output` / `preview_every`, the `preview` event |
| `comfyui_models` | `MI_CAP_COMFYUI_MODELS` | false | a ComfyUI file or folder as `load`'s `model` |
| `model_files` | `MI_CAP_MODEL_FILES` | true | `load`'s `dit_file`, `text_encoder_file`, `vae_file` (ComfyUI / community INT8 ConvRot and W4A8 files included) |
| `lora` | `MI_CAP_LORA` | true | `generate`'s `loras`, `load`'s `lora_strict` |
| `tiers` | `MI_CAP_TIERS` | true | `load`'s `tier` and `attention` |
| `samplers` | `MI_CAP_SAMPLERS` | true | the samplers after `ab2`, `scheduler`, `noise_layout`, `noise_source`, `flow_multiplier`, `eta`, `s_noise`, `dataspace` |
| `image_input` | `MI_CAP_IMAGE_INPUT` | true | `generate`'s `init_image`, `mask`, `mask_encoding`, `resize`, `denoise_strength` |
| `hires` | `MI_CAP_HIRES` | true | `generate`'s `hires` |
| `distill` | `MI_CAP_DISTILL` | true | `load`'s `time_embedding_file` and `distill`, `generate`'s `distill` |
| `system_cuda` | `MI_CAP_SYSTEM_CUDA` | true | `probe.cublaslt`; cuBLASLt loaded at run time: the bundled copy, else (or with `MOSSINFER_CUDA_LIBS=system`) a compatible system copy |
| `control` | `MI_CAP_CONTROL` | true | `generate`'s `control` |
| `accel` | `MI_CAP_ACCEL` | true | `generate`'s `cfg_truncation`, `step_cache`, `step_cache_threshold` |
| `preload` | `MI_CAP_PRELOAD` | true | the `preload` method |
| `release` | `MI_CAP_RELEASE` | true | the `release` method; the report's `dit_load` |
| `certificate_dirs` | `MI_CAP_CERTIFICATE_DIRS` | false | reserved since 2.0: `load`'s `certificate_dirs` is ignored |
| `te_host_pin` | `MI_CAP_TE_HOST_PIN` | true | `load`'s `te_host_pin` |
| `progress_fraction` | `MI_CAP_PROGRESS_FRACTION` | true | `fraction` and `eta_seconds` in `progress` events and in `status` |
| `inspect` | `MI_CAP_INSPECT` | true | the `inspect` method |

`values` lists the accepted names of every enum-like string parameter, in the C enum's order, from the same tables the server validates against (`tools/server/names.h`), so a launcher can fill its menus from them: `load`'s `mode`, `block_path`, `tier`, `attention`, `dit_direct_io` and `te_host_pin`, and `generate`'s (and `preload`'s) `sampler`, `scheduler`, `noise_layout`, `noise_source`, `resize`, `mask_encoding` and `step_cache`. A name outside the list is `invalid_params`, and the message lists the names. Gate a feature's values on its capability as before (e.g. `turbo` needs an INT4 DiT, `inspect` says which files are).

### `load`

Creates the engine. A loaded engine is released first, because two engines would hold two arenas.

| Param | Type | Default | Notes |
|---|---|---|---|
| `model` | string | required | bundle directory (`{text_encoder, tokenizer, transformer, vae}`) |
| `tier` | string | required | `"reference"`, `"high"`, `"turbo"` ([c-api.md](c-api.md#tier-and-attention)). Absent or null: `invalid_params` (`"tier" is required: reference, high or turbo`), checked once every other parameter is valid. |
| `mode` | string | `"balanced"` | `"min_memory"`, `"balanced"`, `"max_speed"` |
| `device` | integer | 0 | CUDA ordinal |
| `vram_budget_bytes` | integer | 0 (auto) | cap on the process's device memory, CUDA context included |
| `vram_reserve_bytes` | integer | 384 MiB | |
| `host_ram_budget_bytes` | integer | 0 (mode target) | |
| `cache_dir` | string | none | embedding-cache disk spill |
| `embedding_cache` | bool | true | |
| `direct_io` | bool | false | direct I/O (`O_DIRECT`) for every weight read, whatever `dit_direct_io` says |
| `dit_direct_io` | string | `"auto"` | how the DiT's load reads its files when `direct_io` is false: `"auto"` (adaptive: direct I/O unless the file system gains nothing from it or the files are already in the page cache), `"buffered"`, `"direct"`. Streamed DiT blocks read buffered unless `direct_io`. The report's `dit_load` says what a load used and why. |
| `te_host_pin` | string | `"auto"` | whether a host-resident text encoder may hold a pinned copy: `"auto"` (where the machine clearly has the room), `"on"` (whenever it fits the host limit), `"off"`. A pinned copy that cannot be allocated streams instead, with a `done.warnings` line. |
| `certificate_dirs` | any | none | reserved since 2.0: accepted and ignored |
| `block_path` | string | `"default"` (fused) | DiT block implementation: `"default"`, `"op_by_op"`, `"fused"`, `"fused_graphs"`. All give the same image; the path that ran is the report's `dit_blocks.path`. |
| `dit_file` | string | `""` (the bundle's `transformer/`) | single-file DiT: INT8 ConvRot, INT4 ConvRot (`turbo`), mixed INT8/bf16 or bf16, ComfyUI layout |
| `text_encoder_file` | string | `""` (the bundle's `text_encoder/`) | INT8 ConvRot or W4A8 text encoder |
| `vae_file` | string | `""` (the bundle's `vae/`) | single FLUX.2 VAE file |
| `attention` | string | `"default"` (the tier's default: `high` and `turbo` run `q8_fp8`, or `q8_fp16` on a GPU without FP8) | `"default"`, `"exact"`, `"q8_fp16"`, `"q8_fp16_f32acc"`, `"q8_fp8"`, `"q8_fp8_f32acc"` |
| `lora_strict` | bool | false | unknown LoRA keys and lossy merges fail the generation (`model`) instead of being reported |
| `time_embedding_file` | string | none | a step-embedder replacement (`time_embedding.safetensors`) applied to the loaded DiT |
| `distill` | string | none | a distill bundle directory: the engine takes its time embedding. Send the same bundle as `generate`'s `distill` to use the rest. |

`tier` and `attention` take exactly the pipeline's spellings (`tier_name()`, `attention_name()`), plus `"default"` for an unset attention. The CLI's alias `q8` (= `q8_fp8`) is not accepted. The tokenizer and configs always come from `model`, which stays a bundle directory; its weights for a component a file replaces are not read (`model_path` in [c-api.md](c-api.md#engine)). A component file that does not exist fails `load` with `invalid_argument`, naming the parameter; a malformed one, or a folder or file of another model ([supported models](integration.md#supported-models)), fails the first `generate` with `model`. The tiers are described in [c-api.md](c-api.md#tier-and-attention): a combination the tier refuses (e.g. `"reference"` with an INT8 `dit_file`) is accepted by `generate` and then ends in `done` with `"status": "error"` and the code `invalid_argument`. After a run, the report's `quality` and `models` objects say what ran.

```
>> {"id": 2, "method": "load", "params": {"model": "/models/MOSS-Image/base", "tier": "reference", "mode": "balanced"}}
<< {"event": "log", "id": 2, "level": "info", "message": "engine created for /models/MOSS-Image/base (mode balanced, device 0); weights load on the first generate"}
<< {"id": 2, "result": {"model": "/models/MOSS-Image/base", "mode": "balanced", "device": 0}}
```

`load` checks the path and initializes CUDA. The weights load during the first `generate` (the `dit_load` phase), or during a `preload`, and then stay resident as the plan allows.

`te_host_pin` takes the pipeline's spellings (`te_host_pin_name()`); `dit_direct_io` has the server's own.

### `generate`

Starts one generation and **answers at once** with `{"accepted": true}`. Progress then arrives as events. The outcome, including the image path and the report, arrives in the `done` event with the same `id`. Only one generation runs at a time; another `generate` meanwhile gets `busy`.

The params are those of `parameters.json`, plus `output`:

| Param | Type | Default |
|---|---|---|
| `output` | string, required | PNG path. It is written atomically (temporary file, then rename). |
| `prompt` | string, required | |
| `negative_prompt` | string | `""` |
| `width`, `height` | integer | 1024. Multiples of 16, at most 4096 per side and 4.2 MP in total. |
| `steps` | integer ≥ 1 | absent = unset, resolved as the dataspace file's step count with the `dataspace` sampler (e.g. the `distill` bundle's), else 50. With the `dataspace` sampler it must not exceed the file's step codes. |
| `cfg` | number ≥ 0 | absent = unset, resolved as 1.0 with `distill`, else 3.0 |
| `shift` | number | 5.0 |
| `seed` | integer or null | 42. `-1` or `null` means random; the seed used is in the report. |
| `uncond_skip_layers` | integer array or `"22, 24"` string or null | `[22]`. `[]`, `""` or `null` means none. |
| `slg_layers` | same | `[]` |
| `slg_scale`, `slg_start`, `slg_stop` | number | 0, 0, 1 |
| `max_sequence_length` | integer | 4096 |
| `sampler` | string | `"euler"`. One of `"euler"`, `"ab2"`, `"euler_ancestral"`, `"heun"`, `"dpmpp_2m"`, `"dpmpp_2m_sde"`, `"er_sde"`, `"res_multistep"`, `"dataspace"`. |
| `scheduler` | string | `"moss"`. One of `"moss"`, `"simple"`, `"normal"`, `"karras"`, `"exponential"`, `"beta"`, `"sgm_uniform"`. `"moss"` with `"euler"` is the official pipeline, bit for bit. |
| `flow_multiplier` | number | 1.0; finite and > 0. `"moss"` ignores it. |
| `noise_layout` | string | `"official"`, or `"batch_stream"` |
| `noise_source` | string | `"philox"`, or `"philox_device"`, `"brownian_tree"`: the stochastic samplers' noise |
| `eta` | number | 1.0 (`euler_ancestral`, `dpmpp_2m_sde`); finite and ≥ 0 |
| `s_noise` | number | 1.0 (the stochastic samplers); finite and ≥ 0 |
| `dataspace` | string | none. The `dataspace` sampler's parameters file: required by it, and then it must exist. |
| `loras` | array | `[]`. Objects `{"path": string, "strength": number}`, applied in order, at most 64. `path` is required and must name an existing file; `strength` defaults to 1 and must be finite. Other members, and a member given twice, are rejected. |
| `save_metadata` | bool | true: the PNG carries the generation settings (`generation` and `parameters` text chunks, [c-api.md](c-api.md#png-metadata)) |
| `include_prompts` | bool | true: the metadata includes the prompt and negative prompt |
| `preview_output` | string | none. A PNG path that the server rewrites with a latent preview during sampling (see the `preview` event). |
| `preview_every` | integer ≥ 1 | 1: steps between previews (the last step always gets one). Only valid together with `preview_output`. Use about 5 for large images (see the `preview` event). |
| `init_image` | string | none. img2img: a PNG or JPEG file (decoded by the server, see [Security](#security)), brought to `width` × `height` by `resize`. |
| `mask` | string | none. Inpainting: a PNG or JPEG file, read as grayscale, the init image's size. White (≥ 128) regenerates, black keeps. Needs `init_image`. |
| `mask_encoding` | string | `"u8"` (values as given), or `"binary01"` (every value 0 or 1, scaled to 0 / 255; any other value is rejected) |
| `denoise_strength` | number in [0, 1] | absent = unset, resolved as 0.75 with `init_image`, else 1.0. The share of the schedule run from `init_image`; below 1 needs `init_image`. An explicit 1 without a mask ignores the image (`done.warnings` says so). |
| `resize` | string | `"crop"`, or `"fit"`, `"stretch"` (the pipeline's `resize_mode_name()` spellings) |
| `hires` | object | none. The hires fix: `{"width": int, "height": int, "strength": number, "steps": int}`. `width` and `height` are required and positive (multiples of 16, at most 4096 per side and 4.2 MP); `strength` defaults to 0.45, `steps` to 0 (= `steps`). Other or repeated members are rejected. |
| `distill` | string | none. A distill bundle directory: its LoRA is appended at strength 1, its dataspace sampler and step count and CFG 1 apply when `sampler`, `steps` and `cfg` are absent (`"euler"` counts as absent and cannot be forced with a dataspace bundle; any other sampler can). A bundle with a time embedding needs the same bundle in `load`'s `distill` (or its file in `time_embedding_file`); otherwise the generation ends in `done` with `invalid_argument`. |
| `control` | object | none. A control adapter ([c-api.md](c-api.md#control-adapters)): `{"adapter": string, "lora_strength": number, "images": [{"kind": string, "path": string, "strength": number, "start": number, "end": number}]}`. `adapter` (the file) is required; `lora_strength` (its LoRA part) defaults to 1. `images` defaults to `[]`, at most 16, one per kind: `kind` (one of the adapter's, e.g. `"depth"`) and `path` (a PNG or JPEG file, decoded by the server as RGB, see [Security](#security)) are required; `strength` defaults to 1 (0 to 10), `start` and `end` to 0 and 1 (0 ≤ `start` ≤ `end` ≤ 1: the share of the steps the control is active in). Other or repeated members are rejected. It changes the image. |
| `cfg_truncation` | number in [0, 1] | 1 (off). Lossy: classifier-free guidance only for the first `cfg_truncation` share of the steps; no effect without CFG (`cfg` 1). |
| `step_cache` | string | `"off"`, or `"fbcache"` (the First-Block Cache: DiT blocks after the first are skipped while block 0's output barely changes). Lossy. |
| `step_cache_threshold` | number in (0, 1] | absent = the engine's default for the loaded model: 0.02 on the base model, 0.03 on the SFT model and the files derived from it. The step cache's relative-change threshold: larger is faster and further from the uncached image. Ignored without `step_cache`. The report's `accelerators.step_cache_threshold` and `step_cache_threshold_source` say which ran. |

`save_metadata` and `include_prompts` default to true for every client. The server has not been released before 0.2.0, so no client depends on the C API's 0.1.0 behaviour (no metadata), which only a program built against the 0.1.0 header keeps ([c-api.md](c-api.md#struct-sizes)).

The names are the pipeline's own spellings (`sampler_name()`, `scheduler_name()`, `noise_layout_name()`, `noise_source_name()`, `step_cache_name()`). The server maps them to the C API's enum values through its own table (`tools/server/names.h`), which a `cpu` test checks against the pipeline.

Paths (`init_image`, `mask`, `distill`, `control`'s `adapter` and image `path`s, and `load`'s `time_embedding_file` and `distill`) must be non-empty. The request is validated before it is accepted, so an invalid request gets an `invalid_params` error and no events. Image files are checked then for existence and size only; the worker decodes them once, during the generation, so an undecodable image (or a `binary01` mask file with other values) ends in `done` with `"status": "error"` and the code `invalid_argument`. Error messages in responses and in `done` name the protocol's parameters, not the C API's fields. Only whole field names are renamed: a file path in a message comes back as given, even when it contains a field name (`/data/my_control_adapter.safetensors`). Validation comes before the server's state: an invalid request gets `invalid_params` even without an engine or while a generation runs. It includes the existence of the LoRA, dataspace and control adapter files. The control adapter itself is read during the generation: a file that is not a control adapter ends in `done` with `model`, and an image `kind` the adapter does not have with `invalid_argument` (the message lists its kinds).

```
>> {"id": "bad", "method": "generate", "params": {"prompt": "x", "width": 1000, "output": "x.png"}}
<< {"id": "bad", "error": {"code": "invalid_params", "message": "Height and width must be positive multiples of 16."}}
```

A complete run. Lines are shortened; a real `plan` and `report` are longer:

```
>> {"id": "g1", "method": "generate", "params": {"prompt": "a red apple on a wooden table", "width": 256, "height": 256, "steps": 2, "seed": 7, "output": "/tmp/apple.png"}}
<< {"id": "g1", "result": {"accepted": true}}
<< {"event": "plan", "id": "g1", "explanation": "Mode balanced (least time within the budget), 256x256, 17+10 tokens. ...", "plan": {"mode": "balanced", "dit": {...}, ...}}
<< {"event": "progress", "id": "g1", "phase": "text_encoder", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 0.17703, "fraction": 0.018, "eta_seconds": 9.5}
<< {"event": "progress", "id": "g1", "phase": "dit_load", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 7.07368, "fraction": 0.8, "eta_seconds": 1.8}
<< {"event": "progress", "id": "g1", "phase": "denoise", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 8.29395, "fraction": 0.965, "eta_seconds": 0.3}
<< {"event": "progress", "id": "g1", "phase": "denoise", "step": 1, "total_steps": 2, "step_seconds": 0.12715, "elapsed_seconds": 8.42648, "fraction": 0.98, "eta_seconds": 0.17}
<< {"event": "progress", "id": "g1", "phase": "denoise", "step": 2, "total_steps": 2, "step_seconds": 0.116925, "elapsed_seconds": 8.54342, "fraction": 0.994, "eta_seconds": 0.05}
<< {"event": "progress", "id": "g1", "phase": "decode", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 8.54358, "fraction": 0.994, "eta_seconds": 0.05}
<< {"event": "done", "id": "g1", "status": "ok", "image": "/tmp/apple.png", "report": {"request": {...}, "seed_used": 7, "steps_run": 2, "cancelled": false, "image": "/tmp/apple.png", ...}, "warnings": []}
```

With a preview (`"preview_output": "/tmp/preview.png"`), each denoise step that gets one sends a `preview` right before its `progress`:

```
<< {"event": "preview", "id": "g1", "step": 1, "total_steps": 2, "image": "/tmp/preview.png", "width": 256, "height": 256}
<< {"event": "progress", "id": "g1", "phase": "denoise", "step": 1, "total_steps": 2, "step_seconds": 0.13021, "elapsed_seconds": 8.43117, "fraction": 0.98, "eta_seconds": 0.17}
```

### Events of a generation

| Event | Fields |
|---|---|
| `plan` | `explanation` (several lines of text), `plan` (object: residencies, arena, per-phase peaks, time estimate). Sent once, before the first phase. |
| `progress` | `phase` (`text_encoder`, `dit_load`, `denoise`, `decode`, in that order), `step`, `total_steps`, `step_seconds`, `elapsed_seconds`, `fraction`, `eta_seconds` |
| `preview` | `step`, `total_steps`, `image` (the `preview_output` path), `width`, `height`. Only with `preview_output`. |
| `done` | `status`, `warnings`, plus fields that depend on `status` (below) |
| `log` | `level` (`info`, `warning`), `message`; `id` when it belongs to a request |

`progress` is sent at each phase start (`step` 0) and after every sampling step.
- `text_encoder` is entered even on a full embedding-cache hit, and `dit_load` even when the DiT is resident.
- `total_steps` counts the scheduled steps, which can be fewer than the requested `steps`.
- `fraction` (0 to 1) is the share of the whole job done: every phase, both passes of a `hires` fix, a `preload`'s loads. It never decreases within a job and stays below 1 until `done`, which is the end. `eta_seconds` is the estimated time left, `null` when unknown. Both are `null` in every event when the server runs on a library without them (`capabilities.progress_fraction` false). Drive a progress bar and its ETA from these two, not from `step / total_steps`, which is 0 through the text encoder and the loads and covers only the sampling steps.
- They come from the plan's time estimate, plus a load estimate for each model not loaded yet (its size at the hardware profile's transfer rates: probed or learned, the built-in defaults before the first probe), corrected as the job runs: a phase that ended counts at its measured time, and the remaining steps of a pass at its measured step time (the first step, which carries the pass's setup, sets the price only until a later one is measured). The first run's estimate is the roughest.
- A DiT load is one step: no `progress` comes during it, so the bar holds still while the DiT loads.

`preview` is sent after every `preview_every`-th sampling step and after the last one, before that step's `progress`.
- The server first rewrites the `preview_output` PNG through `mi_write_png()` (atomically: a reader never sees a partial file), then sends the event. Read the file when the event arrives; the next preview replaces it.
- The image has the requested size (`width` × `height`). It is a linear latent → RGB map of the current x0 estimate, without the VAE: coarse, not the final image.
- Each preview costs a full-size device-to-host copy and a PNG write (about 3 bytes per pixel, uncompressed), both on the generating thread while the GPU waits. For large images, a `preview_every` of about 5 keeps that cost small.
- A preview that cannot be written is skipped with a `log` event of level `warning`; the generation goes on. On Windows this happens when a reader holds the preview file open while the server renames the new one over it: open the file, read it and close it again, or copy it, rather than keeping it open.

The fields of `done` depend on `status`:

| `status` | Fields |
|---|---|
| `"ok"` | `image`, `report` |
| `"cancelled"` | `steps_run`, `report` (no image is written) |
| `"error"` | `error: {code, message}`, plus `report` when only the PNG write failed |

`warnings` is always present: an array of strings for the launcher to show, empty when there is nothing to say.
- One entry per LoRA warning, `"LoRA <module>: <reason>"` (e.g. an adapter requantized on INT8 GEMMs), and per warning on the control adapter's LoRA part, `"control adapter LoRA <module>: <reason>"`.
- `"lossy accelerators (<the active ones>): CFG truncated on N steps, step cache reused C of D DiT calls"` when `cfg_truncation` (with CFG) or `step_cache` is active, with only the active accelerators' parts.
- `"step_cache_threshold has no effect without step_cache \"fbcache\""` and `"cfg_truncation has no effect without CFG (cfg 1)"` for settings that change nothing, as `moss-infer generate` warns. A control adapter, like any LoRA, gets no such line: the report's `quality.config.lora` is `"active"`.
- `"prompt truncated to K of N tokens (D dropped; max_sequence_length M)"`, and the same for the `"negative prompt"`, when the text encoder cut the end of a prompt.
- One entry per accelerator warning of the engine (`mi_result_message(MI_MESSAGE_ACCEL_WARNING)`): a step-cache threshold that moves the base model's composition, or a cache that never engaged.
- `"image input: <note>"` when the image input has a remark, e.g. an init image not used at strength 1 without a mask.
- `"text encoder: <note>"` when the text encoder has a remark, e.g. a pinned host copy that could not be allocated.

`report` is the same document as `moss-infer generate --json-report`, or `null` in the rare case that the library could not render it (`mi_result_report_json()` returned NULL, e.g. out of host memory):
- `request`, `seed_used`, `steps_scheduled`, `steps_run`, `cancelled`, `cond_tokens`, `uncond_tokens`
- `image`, `latents` (always `""` here), `mode`, `plan`, `embedding_cache`, `decode`
- `img2img`: the image mode and note, the VAE encoder, and both passes (`first_pass`, `hires`)
- `dit_blocks` (`path` that ran, `note` saying why it differs from the requested one, or `""`)
- `timings_s`, `memory_gib`, `host_memory_gib`, `hardware`
- `dit_stream`: DiT block streaming (`streamed`, `resident_blocks`, `buffers`, `source`, `bytes`, `copy_s`, `stall_s`, `stalls`, `h2d_gbps`, `hidden`, and per-step `steps`, each with `cached`)
- `tokens`: per prompt, `total`, `kept`, `template`, `prompt`, `dropped`, `max_length`, `truncated`
- `quality` (tier, configuration, engine numerics, cuBLASLt copy, GPU class, model paths) and `models` (DiT, text encoder and VAE sources, DiT linears, tier, text-encoder precision)
- `loras` (per file: adapters, side branch, merged, unknown keys, strength), `lora_warnings`, `lora_notes`
- `control` (`null` without `control`: the adapter, kinds, base model, inputs, its LoRA part, encodes, encode schedule) and `accelerators` (`lossy`, `active`, `step_cache_threshold`, `step_cache_threshold_source`, `cfg_truncated_steps`, `step_cache_calls`, `step_cache_reused`, `cached_steps`, `step_cache_diffs`, `warnings`)
- `dit_load` (how the call's last DiT load read the weights: `source` `"files"`, `"host_copy"` or `null`, `io`, `note`, `bytes`, `seconds`, `gbps`) and, after a `preload`, `preload_note`

The server is idle again by the time `done` is written, so the next `generate` can follow it immediately.

### `preload`

Loads what the engine keeps resident for a request, without generating (`mi_engine_preload()`, [c-api.md](c-api.md#preload-and-soft-unload)): the DiT (unless the plan parks it for the text encoder; `preload_note` then says so), the VAE, the request's LoRAs and control adapter, and a host-resident text encoder's copy. A `generate` of a request like it then starts sampling without a `dit_load` wait, with the same image bits.

The params are `generate`'s without `output`, `preview_output`, `preview_every`, `save_metadata` and `include_prompts` (those are rejected as unknown). `prompt` is optional: an absent or blank prompt plans for a short one, and a long prompt sizes the workspaces for its length. The other params drive the plan: size, steps, CFG, `loras`, `control`, `init_image`, `hires`, the accelerators. They are validated as for `generate`, before the server's state.

Like `generate`, it answers at once with `{"accepted": true}`, then sends events with its `id`: one `plan` after planning and before any load, then a `progress` (`step` 0, `total_steps` 0) before each load step, also for a model already resident (`text_encoder` before a host-resident text encoder's copy, `decode` before the VAE encoder of a request that encodes an image (an `init_image` below strength 1 or with a `mask`, `control` images, or a `hires` fix), `dit_load` before the DiT, `decode` before the VAE decoder; never `denoise`), and `done`. Their `fraction` and `eta_seconds` cover the load steps. `busy` and `cancel` work as for `generate`; a cancel stops it between models and DiT weight groups, and the engine keeps what was loaded before it.

```
>> {"id": "p1", "method": "preload", "params": {"width": 1024, "height": 1024, "steps": 30}}
<< {"id": "p1", "result": {"accepted": true}}
<< {"event": "plan", "id": "p1", "explanation": "Mode balanced ...", "plan": {...}}
<< {"event": "progress", "id": "p1", "phase": "dit_load", "step": 0, "total_steps": 0, "step_seconds": 0, "elapsed_seconds": 0.1, "fraction": 0.045, "eta_seconds": 2.1}
<< {"event": "progress", "id": "p1", "phase": "decode", "step": 0, "total_steps": 0, "step_seconds": 0, "elapsed_seconds": 2.1, "fraction": 0.95, "eta_seconds": 0.1}
<< {"event": "done", "id": "p1", "status": "ok", "report": {...}, "preload_note": "", "dit_load": {"source": "files", "io": "direct", "bytes": 10000000000, "seconds": 2.0, "note": ""}, "warnings": []}
```

`done` has `status` `"ok"` (with `report`), `"cancelled"` (with `steps_run` 0 and `report`) or `"error"` (with `error`). With a result it also carries, at the top level:
- `preload_note`: what the preload did not load and why; `""` otherwise;
- `dit_load`: `{"source": "files" | "host_copy" | null, "io": "direct" | "buffered" | null, "bytes", "seconds", "note"}`. `source` is `null` when the DiT was already resident or was not loaded;
- `warnings`, as for `generate`.

### `release`

A soft unload (`mi_engine_release()`): frees the engine's device memory so that another application can use the GPU, and keeps the engine. Unlike `unload`, the next `generate` needs no `load`: it plans and loads again. The embedding cache stays.

Params: `keep_host_copy` (bool, default true). With true, the DiT's weights are read into RAM where the host has the room, and its next load uploads them from there (the report's `dit_load.source` `"host_copy"`); a host-resident text encoder keeps its copy. With false, every host copy is freed too.

The response comes when the release ends. Reading the DiT into RAM takes seconds (it reads the DiT's weight files), so the release runs on the worker like a job: requests sent meanwhile are answered first, `status` reports `"releasing"`, and `cancel` (with or without the release's `id`) stops it. A cancel that lands before the release frees anything stops it whole: `cancelled` is true, nothing is released (`vram_released_bytes` 0; the engine keeps its device memory and the next `generate` loads nothing), and `note` is `"cancelled before it started"`. Later, it stops the DiT's host copy (before its first read or between its 64 MiB reads): the release completes (`cancelled` false), frees the device memory, keeps a host-resident text encoder's copy and frees nothing else; the interrupted DiT copy is not kept, and `note` says so. A copy that finished before the cancel reached it is kept. The values in the examples of this section are illustrative.

```
>> {"id": "r1", "method": "release", "params": {"keep_host_copy": true}}
<< {"id": "r1", "result": {"cancelled": false, "seconds": 5.0, "vram_released_bytes": 11000000000, "host_copy_bytes": 10000000000, "host_copy_seconds": 5.0, "dit_host_copy": true, "te_host_copy": false, "note": ""}}
```

| Result field | Meaning |
|---|---|
| `cancelled` | true: a cancel stopped the release before it freed anything (the other fields: nothing released, the copies already held) |
| `seconds` | the whole release, `host_copy_seconds` included |
| `vram_released_bytes` | device memory freed: the arena and what the engine held outside it |
| `host_copy_bytes` | host RAM the kept copies hold |
| `host_copy_seconds` | reading the DiT's files into RAM |
| `dit_host_copy` | the DiT's weights are kept in RAM |
| `te_host_copy` | a host-resident text encoder's copy is kept |
| `note` | why a copy was not kept (the host budget, a cancel); `""` otherwise |

`release` answers `busy` while a generation or preload runs, and `no_engine` before `load`. An engine error is the response's `error`.

### `cancel`

Params: `id` (optional). The id of the job to cancel (a `generate`, `preload` or `release`); it is ignored unless it matches the running one.

A generation stops within one text-encoder layer, DiT block or decode band, and then sends `done` with `"status": "cancelled"`. A preload stops between models and weight groups (`done` with `"cancelled"`). A release stops before it frees anything (`"cancelled": true`), or stops the DiT's host copy (the text encoder's copy stays), and answers as usual. A cancel never carries over to the next job.

```
>> {"id": 4, "method": "cancel", "params": {"id": "g2"}}
<< {"id": 4, "result": {"cancelling": true, "generating_id": "g2"}}
<< {"event": "done", "id": "g2", "status": "cancelled", "steps_run": 2, "report": {...}, "warnings": []}
```

When nothing runs, the answer is `{"cancelling": false}`. When another job runs, the answer also carries its id as `"generating_id"` (the name predates `preload` and `release`, and covers them).

### `status`

```
>> {"id": 3, "method": "status"}
<< {"id": 3, "result": {"state": "generating", "model": "/models/MOSS-Image/base", "mode": "balanced", "generating_id": "g2",
    "phase": "denoise", "step": 2, "total_steps": 30, "elapsed_seconds": 5.94924, "fraction": 0.3, "eta_seconds": 14,
    "cancelling": false}}
```

`state` is `"unloaded"` (no engine), `"idle"`, `"generating"`, `"preloading"` or `"releasing"`; the last two appear only to a client that sent `preload` or `release`. `model` and `mode` are present when an engine is loaded. While a job runs, `generating_id` is its id and `cancelling` says whether a cancel was requested; the progress fields (`phase`, `step`, `total_steps`, `elapsed_seconds`, `fraction`, `eta_seconds`) are present while generating or preloading. They are the last `progress` event's (before the first one: `phase` `null`, `fraction` 0 and `eta_seconds` `null`; on a library without them both stay `null`).

### `inspect`

Params: `path` (string, required): a file. Says what it is to the engine (`mi_inspect_file()`, [c-api.md](c-api.md#file-inspection)) without an engine or a GPU, so it works before `load` and in every state, also while a job runs: it touches no engine state. It runs on a thread of its own and is answered when it is done, so requests sent after it (`status`, `cancel`) may be answered first: match the response by `id`. At most two run at a time; a third is answered `busy` (send it again after one answers). `shutdown` waits for the running ones, which are answered before its own response.

```
>> {"id": 5, "method": "inspect", "params": {"path": "/models/moss_image_sft_int4_convrot.safetensors"}}
<< {"id": 5, "result": {"path": "/models/moss_image_sft_int4_convrot.safetensors", "kind": "dit", "loadable": true,
    "file_bytes": 3000000000, "tensors": 700, "note": "", "precision": "w4a4", "quantization": "int4_convrot256",
    "tiers": ["turbo"], "tier_reasons": {"reference": "tier reference runs bf16 weights exactly, ...",
    "high": "the loaded DiT is INT4 ConvRot, which runs the Turbo tier; use tier turbo",
    "turbo": "Turbo tier: INT4 ConvRot GEMMs, 8-bit attention (q8_fp8; q8_fp16 on GPUs without FP8)"}}}
>> {"id": 6, "method": "inspect", "params": {"path": "/models/moss_image_color_depth_control.safetensors"}}
<< {"id": 6, "result": {"path": "...", "kind": "control_adapter", "loadable": true, "file_bytes": 300000000, "tensors": 500,
    "note": "", "control_kinds": ["color", "depth"], "base_model": "moss-base", "adapter_types": ["lora"], "lora_rank": 64,
    "lora_adapters": 250, "lora_unknown_keys": 0}}
```

Every answer has `path`, `kind` (`"dit"`, `"text_encoder"`, `"vae"`, `"lora"`, `"control_adapter"` or `"unknown"`), `loadable` (the loader for that role accepts the file), `file_bytes`, `tensors` and `note` (why not loadable, or what an unknown file is instead, e.g. a GGUF file or a distill bundle file; `""` for none). By kind:
- `dit`: `precision` (`"bf16"`, `"int8"`, `"w4a4"` or `"nvfp4"`; `null` when not loadable), `quantization` (its block linears as the engine runs them, e.g. `"int8_convrot256"`; `""` when not loadable), `tiers` (exactly the `load` `tier` values `generate` accepts for it with the bundle's text encoder and no explicit `attention`: the engine's own decision, the same on every GPU it supports; an NVFP4 DiT is judged for an RTX 50 with this build's kernels) and `tier_reasons` (for `reference`, `high` and `turbo`: what the tier runs, e.g. `"High tier: INT8 ConvRot GEMMs, 8-bit attention (q8_fp8; q8_fp16 on GPUs without FP8)"`, with `(RTX 50 only)` for an NVFP4 file, or why it is refused).
- `text_encoder`: `precision` (`"bf16"`, `"int8"` or `"w4a8"`) and `quantization`.
- `lora`: `base_model` (the metadata, `""` when absent), `adapter_types` (`"lora"`, `"loha"`, `"lokr"`, `"full"`), `lora_rank` (the largest low-rank rank), `lora_adapters` (on the DiT) and `lora_unknown_keys`. A file in the LoRA naming conventions that the LoRA reader refuses (a LoRA for another model, a malformed one) is a `lora` with `loadable` false and the reader's message as `note`.
- `control_adapter`: `control_kinds` (in channel order: exactly the strings `generate`'s `control.images[].kind` takes) and the `lora` fields of its LoRA part.

A file the engine cannot use is a result (`unknown` or `loadable` false), not an error. Errors: `invalid_params` for a missing or empty `path`, unknown params, and a path that is not a regular file (a directory, a device, a pipe: refused from its type without opening it on POSIX; on Windows, device and pipe namespace paths, DOS device names and UNC pipe, mailslot and IPC$ shares are refused by name before any open; any other path is opened for its attributes only (identification-level SQOS) and refused by its type); `io` for a file that does not exist or cannot be read (the system's reason); `busy` beyond two running inspects.

The result depends only on the file's content and the engine's version and build, not on the GPU: a launcher may cache it per file (path, size, modification time), `hello`'s `version` and `probe.build_flavour` (the NVFP4 kernels are a build option).

### `unload`

Releases the engine and its device memory, and answers `{"unloaded": true}`, or `false` when nothing was loaded. While a job runs it answers `busy`. A launcher that wants the GPU back but keeps the model selected sends `release` instead: it keeps the engine (and optionally the weights in RAM), so the next `generate` needs no `load`.

### `shutdown`

Cancels a running job and waits for it to finish. That job's `done` event (a release's response) comes first. The server then answers `{}` and exits with status 0.

## Client sketch (Python)

```python
import json, subprocess
p = subprocess.Popen(["moss-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def send(msg): p.stdin.write((json.dumps(msg) + "\n").encode()); p.stdin.flush()
def read(): return json.loads(p.stdout.readline())
send({"id": 1, "method": "load", "params": {"model": "/models/MOSS-Image/base"}})
while "event" in (m := read()): pass            # skip the log event, keep the response
send({"id": 2, "method": "generate", "params": {"prompt": "a cat", "output": "cat.png"}})
while not ((m := read()).get("event") == "done" and m["id"] == 2):
    if m.get("event") == "progress":                # the whole job's share and time left (null: unknown)
        f = m["fraction"] or 0.0
        eta = "?" if m["eta_seconds"] is None else "%.0f s" % m["eta_seconds"]
        print("\r[%-30s] %3.0f%% %s left (%s)" % ("#" * int(30 * f), 100 * f, eta, m["phase"]), end="", flush=True)
print("\n" + m["status"], m.get("image"))
send({"id": 3, "method": "shutdown"})
```

## Tests

`tests/server/test_protocol.py` uses the standard library only. The server's JSON output helpers (`tools/server/json_out.h`; the server links only the C API, so it cannot use the pipeline's report renderer) are checked by the `cpu` gtest suite `ServerJson.*` (`tests/unit/test_server_json.cpp`): strings and numbers are written exactly as `pipeline::json_escape` / `json_number` write them, numbers ignore the C and C++ locales, and a NULL report becomes `null`. `ServerJson.EnumNamesAreThePipelines` checks the name tables of `tools/server/names.h` against the pipeline's names and the C enum values, and `ServerJson.HelloValuesAreTheTables` that hello's `values` lists every table under the parameter it validates.

| ctest name | Label | What it runs |
|---|---|---|
| `server_protocol` | `cpu` | No GPU or weights needed (details below) |
| `package_install_tree` | `cpu` | `tests/server/test_install_tree.py`: installs the build twice (full, and `--component Unspecified` without CUDA libraries), asks each tree's `hello` which cuBLASLt it uses under `MOSSINFER_CUDA_LIBS` `auto` / `system` / `bundled` and `MOSSINFER_CUBLASLT` (a bundled copy found, a system copy outside the tree, clean errors for a missing copy or file and a bad policy), and runs `check_package.py` on both trees. Neither tree has a `share/mossinfer/quality_certs/` directory. |
| `server_protocol_gpu` | `gpu` | With `MOSSINFER_GPU_TESTS=1`: a 256², 2-step generate (events, PNG, report with `dit_blocks` and `dit_stream`, `busy`, validation, NUL in the prompt; `fraction` and `eta_seconds` in every `progress`: the fraction in [0, 1) and never decreasing, the eta known and not negative, and from `text_encoder` to the final `decode` the fraction past one half and the eta smaller), cancel mid-run (a `status` mid-run with `fraction` and a known `eta_seconds`), the engine reused after a cancel, cancel on shutdown; a `high` load of the bf16 bundle (8-bit attention in the report's `quality`; with `attention` `"exact"` the generate ends in `invalid_argument` with the Reference tier's reason); and a 256², 3-step `euler_ancestral` + `karras` generate with `preview_output` (one `preview` event per step before its `progress`, the preview PNG's size and no metadata), the `generation` chunk of the image, a truncated prompt in `done.warnings`, then `include_prompts` false and `save_metadata` false; an invalid `generate` sent while one runs gets `invalid_params`; a 256² img2img from a PNG path (4 steps at strength 0.5: 2 run, `total_steps` 2) and the strength-1 note in `done.warnings`; a 256², 4-step generate with the color/depth `control` adapter (`$MOSS_COMMUNITY_MODELS`, skipped when absent) and a depth PNG plus `step_cache` `fbcache` (the report's `control` and `accelerators`, the LoRA part in `quality.config.lora`, the accelerator line in `done.warnings`, no `done.certified`), then a kind the adapter lacks (`done` with `invalid_argument`), then `cfg_truncation` at CFG 1 and a threshold without the step cache (the two no-effect warnings, nothing lossy); `preload` → `release` → `generate` at 256², 2 steps (the `preload` events, their `fraction` and `eta_seconds` as for `generate` and the following generations' too, and `done` with `dit_load` and `preload_note`, `release` answering `busy` during the preload, the release's stats, the next generate loading from the host copy, a `cancel` sent right after a second `release`: one of the documented outcomes, the text encoder's copy kept, and a 2-step generate after it running every step) |

`server_protocol` covers:
- `hello` (the capabilities, `values` and `probe.cublaslt` included), `status`, and idle `cancel` / `unload`
- every name in `hello`'s `values` passing the server's validation (`load`'s to the engine's own check, `generate`'s to `no_engine`), and a name outside it refused with a message that lists exactly those names
- `inspect` without an engine: a bf16 DiT (precision, quantization, tiers and their reasons), a LoRA with `base_model`, a control adapter whose `control_kinds` `generate` accepts, an oversized header and a GGUF file as results; a missing file (`io`), a directory, a missing or empty `path` and an unknown param (`invalid_params`); eight inspects, a `status` and a `shutdown` sent at once: every request answered exactly once, each inspect with its result or `busy`, all before `shutdown`'s answer; two inspects sent on reading two answers, twenty times: never `busy`
- malformed input of every kind, unknown methods and params (an unknown or non-string `block_path` included), and NUL in strings
- `load` errors mapped from engine statuses
- the P6 `load` params: bad `tier` / `attention` names (`"q8"` and `"auto"` included) and types, a missing or null `tier` (`invalid_params`), every valid name, missing component files
- the image, hires and distill params: bad paths, types and names, a mask without an init image, a `binary01` mask with other values, bad `hires` shapes (not an object, a missing, unknown or repeated member, a non-integer size, not multiples of 16, a bad strength), a missing `distill` directory, the bundle's dataspace sampler with an image; messages name the protocol's parameters (`init_image`, `mask`, `distill`, `hires`), never the C fields; `load`'s `time_embedding_file` and `distill` errors; every valid combination reaches `no_engine`
- the control and accelerator params: `control` shapes (not an object, no or an empty `adapter`, unknown and repeated members in `control` and its images, a missing `kind` or `path`, non-number strengths and windows), a missing adapter or image file, the ranges, a repeated kind, more than 16 images, bad `cfg_truncation`, `step_cache` names and types, the threshold rules; messages name `control.adapter`, `control.images[i].path` and so on, never the C fields; valid combinations (an unknown kind and an undecodable image included: those are the generation's to check) reach `no_engine`; file paths containing `control_adapter`, `image_path`, `mask_path`, `distill_dir` and `dataspace_path` come back unchanged in the messages while the field names are renamed
- the P6 `generate` params, checked without an engine: bad enum names and types, bad numbers, the `dataspace` sampler without or with a missing file (the message names `dataspace`), repeated members in `params` and in a `loras` entry, bad `loras` shapes (not an array, a string element, no or empty `path`, unknown members, a bad `strength`, a missing file), `preview_every` without `preview_output`, and every valid name
- the 2.0 `load` params: `certificate_dirs` of any shape accepted and ignored, bad `dit_direct_io` and `te_host_pin` names and types, every valid value reaching the engine
- `preload` and `release` params: `generate`'s output, preview and metadata params rejected as unknown in `preload`, its request validated as for `generate` (a prompt optional, `generate` still requiring one), `keep_host_copy` types, and `no_engine` for every valid request
- `shutdown`, end of input, and `--version` / usage errors
- stdout set up with stderr sent to a file, to `DEVNULL`, merged into stdout, or closed. On Linux, `/proc` shows exactly one fd on the protocol pipe, at 3 or higher, close-on-exec.
- stdin closed
- `--parent-pid` for the server's own parent and for another process, and for a process that is already gone

```
MOSS_SERVER=<build>/moss-server python3 tests/server/test_protocol.py
MOSSINFER_GPU_TESTS=1 MOSS_SERVER=<build>/moss-server MOSS_WEIGHTS=<models>/MOSS-Image python3 tests/server/test_protocol.py
```
