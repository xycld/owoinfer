# moss-server protocol (version 1)

`moss-server` is the engine as a sidecar process for launchers, in the role `llama-server` plays for llama.cpp.
- The launcher spawns it and talks **JSON lines over stdin/stdout**. There are no ports, no firewall prompts and no extra dependencies.
- It is built only on the public C API (`docs/c-api.md`), linked against `libmossinfer` when the shared library is built.
- Source: `tools/server/main.cpp`. Target: `moss-server` (full builds; `cmake/modules/server.cmake`). Tests: `tests/server/test_protocol.py`.

```
moss-server                     # speak the protocol on stdin/stdout
moss-server --parent-pid <pid>  # the same, and exit when process <pid> (the launcher) exits
moss-server --version           # "moss-server 0.1.0 (protocol 1, ABI 1)"
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
- **Paths are used as given.** `generate`'s `output` is written wherever it points, and an existing file there is overwritten (atomically: a temporary file renamed over it). `load`'s `model` and `cache_dir` are read, and the cache directory is written to. The server does not confine any path to a directory, resolve symlinks, or check ownership.
- **The launcher validates paths.** A launcher that takes `output`, `model` or `cache_dir` from a user, a web page, a workflow file or another program must check them first. For example, it can confine outputs to its own output directory, reject `..` components and symlinks it did not create, and refuse to overwrite files it does not own.
- **Run it as the user.** Run the server with the launcher's own (unprivileged) rights, never elevated.

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
| event | `{"event": "progress" \| "plan" \| "done" \| "log", "id": <id of the generate request>, ...}` |

Every request gets exactly one response. Responses to different requests can interleave with events, so a client should match responses by `id`. A request whose `id` could not be read (malformed JSON, for example) is answered with `"id": null`.

Unknown members in `params` are rejected (`invalid_params`), so a typo cannot be silently ignored.

### Error codes

| Code | When |
|---|---|
| `parse_error` | malformed JSON, invalid UTF-8, nesting deeper than 128 |
| `invalid_request` | not an object; missing or non-string `method`; bad `id` type; unknown top-level member; line over 1 MiB |
| `method_not_found` | unknown `method` |
| `invalid_params` | wrong parameter type or value, unknown parameter, a string containing NUL (`\u0000`: the C API's strings are NUL-terminated), or a generation request the engine rejects (the message uses the official app's wording) |
| `unsupported_protocol` | `hello` asked for another protocol version |
| `no_engine` | `generate` before `load` |
| `busy` | `generate`, `load` or `unload` while a generation runs |
| `invalid_argument`, `abi_mismatch`, `not_supported`, `out_of_memory`, `model`, `io`, `cuda`, `internal` | the engine's `mi_status` (`MI_ERR_<CODE>`), with `mi_last_error_message()` as the message |

## Methods

### `hello`

Params (all optional):
- `client` (string)
- `protocol_version` (integer; an error unless it is 1)
- `devices` (bool): adds the per-device list, with free memory from NVML or a short-lived driver context. No CUDA context remains afterwards (`mi_probe_device()`, docs/c-api.md).

```
>> {"id": 1, "method": "hello", "params": {"client": "launcher", "protocol_version": 1}}
<< {"id": 1, "result": {"protocol_version": 1, "server": "moss-server", "version": "0.1.0", "abi_version": 1,
    "capabilities": {"preview": false, "comfyui_models": false},
    "probe": {"driver_cuda_version": 13040, "runtime_cuda_version": 13040, "driver_compatible": true,
              "device_count": 1, "supported_device_count": 1, "build_flavour": "cu13", "recommended_flavour": "cu13",
              "message": "recommended flavour cu13 (the driver supports CUDA 13.4)"}}}
```

With `"devices": true`, `probe.devices` is a list of objects with these fields:
- `index`, `name`, `pci_bus_id`, `compute_capability` (e.g. `"8.9"`)
- `total_memory_bytes`, `free_memory_bytes`
- `supported`, `unsupported_reason`

`hello` works without a GPU or driver; the reason is then in `message`.

### `load`

Creates the engine. A loaded engine is released first, because two engines would hold two arenas.

| Param | Type | Default | Notes |
|---|---|---|---|
| `model` | string | required | bundle directory (`{text_encoder, tokenizer, transformer, vae}`) |
| `mode` | string | `"balanced"` | `"min_memory"`, `"balanced"`, `"max_speed"` |
| `device` | integer | 0 | CUDA ordinal |
| `vram_budget_bytes` | integer | 0 (auto) | cap on the process's device memory, CUDA context included |
| `vram_reserve_bytes` | integer | 384 MiB | |
| `host_ram_budget_bytes` | integer | 0 (mode target) | |
| `cache_dir` | string | none | embedding-cache disk spill |
| `embedding_cache` | bool | true | |
| `direct_io` | bool | false | |
| `block_path` | string | `"default"` (fused) | DiT block implementation: `"default"`, `"op_by_op"`, `"fused"`, `"fused_graphs"`. All give the same image; the path that ran is the report's `dit_blocks.path`. |

```
>> {"id": 2, "method": "load", "params": {"model": "/models/MOSS-Image/base", "mode": "balanced"}}
<< {"event": "log", "id": 2, "level": "info", "message": "engine created for /models/MOSS-Image/base (mode balanced, device 0); weights load on the first generate"}
<< {"id": 2, "result": {"model": "/models/MOSS-Image/base", "mode": "balanced", "device": 0}}
```

`load` checks the path and initializes CUDA. The weights load during the first `generate` (the `dit_load` phase), and then stay resident as the plan allows.

### `generate`

Starts one generation and **answers at once** with `{"accepted": true}`. Progress then arrives as events. The outcome, including the image path and the report, arrives in the `done` event with the same `id`. Only one generation runs at a time; another `generate` meanwhile gets `busy`.

The params are those of `parameters.json`, plus `output`:

| Param | Type | Default |
|---|---|---|
| `output` | string, required | PNG path. It is written atomically (temporary file, then rename). |
| `prompt` | string, required | |
| `negative_prompt` | string | `""` |
| `width`, `height` | integer | 1024. Multiples of 16, at most 4096 per side and 4.2 MP in total. |
| `steps` | integer | 50 |
| `cfg` | number | 3.0 |
| `shift` | number | 5.0 |
| `seed` | integer or null | 42. `-1` or `null` means random; the seed used is in the report. |
| `uncond_skip_layers` | integer array or `"22, 24"` string or null | `[22]`. `[]`, `""` or `null` means none. |
| `slg_layers` | same | `[]` |
| `slg_scale`, `slg_start`, `slg_stop` | number | 0, 0, 1 |
| `max_sequence_length` | integer | 4096 |
| `sampler` | `"euler"` or `"ab2"` | `"euler"` |

The request is validated before it is accepted, so an invalid request gets an `invalid_params` error and no events:

```
>> {"id": "bad", "method": "generate", "params": {"prompt": "x", "width": 1000, "output": "x.png"}}
<< {"id": "bad", "error": {"code": "invalid_params", "message": "Height and width must be positive multiples of 16."}}
```

A complete run. Lines are shortened; a real `plan` and `report` are longer:

```
>> {"id": "g1", "method": "generate", "params": {"prompt": "a red apple on a wooden table", "width": 256, "height": 256, "steps": 2, "seed": 7, "output": "/tmp/apple.png"}}
<< {"id": "g1", "result": {"accepted": true}}
<< {"event": "plan", "id": "g1", "explanation": "Mode balanced (least time within the budget), 256x256, 17+10 tokens. ...", "plan": {"mode": "balanced", "dit": {...}, ...}}
<< {"event": "progress", "id": "g1", "phase": "text_encoder", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 0.17703}
<< {"event": "progress", "id": "g1", "phase": "dit_load", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 7.07368}
<< {"event": "progress", "id": "g1", "phase": "denoise", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 8.29395}
<< {"event": "progress", "id": "g1", "phase": "denoise", "step": 1, "total_steps": 2, "step_seconds": 0.12715, "elapsed_seconds": 8.42648}
<< {"event": "progress", "id": "g1", "phase": "denoise", "step": 2, "total_steps": 2, "step_seconds": 0.116925, "elapsed_seconds": 8.54342}
<< {"event": "progress", "id": "g1", "phase": "decode", "step": 0, "total_steps": 2, "step_seconds": 0, "elapsed_seconds": 8.54358}
<< {"event": "done", "id": "g1", "status": "ok", "image": "/tmp/apple.png", "report": {"request": {...}, "seed_used": 7, "steps_run": 2, "cancelled": false, "image": "/tmp/apple.png", ...}}
```

### Events of a generation

| Event | Fields |
|---|---|
| `plan` | `explanation` (several lines of text), `plan` (object: residencies, arena, per-phase peaks, time estimate). Sent once, before the first phase. |
| `progress` | `phase` (`text_encoder`, `dit_load`, `denoise`, `decode`, in that order), `step`, `total_steps`, `step_seconds`, `elapsed_seconds` |
| `done` | `status`, plus fields that depend on it (below) |
| `log` | `level` (`info`), `message`; `id` when it belongs to a request |

`progress` is sent at each phase start (`step` 0) and after every sampling step.
- `text_encoder` is entered even on a full embedding-cache hit, and `dit_load` even when the DiT is resident.
- `total_steps` counts the scheduled steps, which can be fewer than the requested `steps`.

The fields of `done` depend on `status`:

| `status` | Fields |
|---|---|
| `"ok"` | `image`, `report` |
| `"cancelled"` | `steps_run`, `report` (no image is written) |
| `"error"` | `error: {code, message}`, plus `report` when only the PNG write failed |

`report` is the same document as `moss-infer generate --json-report`, or `null` in the rare case that the library could not render it (`mi_result_report_json()` returned NULL, e.g. out of host memory):
- `request`, `seed_used`, `steps_scheduled`, `steps_run`, `cancelled`, `cond_tokens`, `uncond_tokens`
- `image`, `latents` (always `""` here), `mode`, `plan`, `embedding_cache`, `decode`
- `dit_blocks` (`path` that ran, `note` saying why it differs from the requested one, or `""`)
- `timings_s`, `memory_gib`, `host_memory_gib`, `hardware`
- `dit_stream`: DiT block streaming (`streamed`, `resident_blocks`, `buffers`, `source`, `bytes`, `copy_s`, `stall_s`, `stalls`, `h2d_gbps`, `hidden`, and per-step `steps`)

The server is idle again by the time `done` is written, so the next `generate` can follow it immediately.

### `cancel`

Params: `id` (optional). The id of the generation to cancel; it is ignored unless it matches the running one.

The generation stops within one text-encoder layer, DiT block or decode band, and then sends `done` with `"status": "cancelled"`. A cancel never carries over to the next generation.

```
>> {"id": 4, "method": "cancel", "params": {"id": "g2"}}
<< {"id": 4, "result": {"cancelling": true, "generating_id": "g2"}}
<< {"event": "done", "id": "g2", "status": "cancelled", "steps_run": 2, "report": {...}}
```

When nothing runs, the answer is `{"cancelling": false}`. When another generation runs, the answer also carries its `"generating_id"`.

### `status`

```
>> {"id": 3, "method": "status"}
<< {"id": 3, "result": {"state": "generating", "model": "/models/MOSS-Image/base", "mode": "balanced", "generating_id": "g2",
    "phase": "denoise", "step": 2, "total_steps": 30, "elapsed_seconds": 5.94924, "cancelling": false}}
```

`state` is `"unloaded"` (no engine), `"idle"` or `"generating"`. `model` and `mode` are present when an engine is loaded. The progress fields are present while generating.

### `unload`

Releases the engine and its device memory, and answers `{"unloaded": true}`, or `false` when nothing was loaded. While a generation runs it answers `busy`.

### `shutdown`

Cancels a running generation and waits for it to finish. That generation's `done` event comes first. The server then answers `{}` and exits with status 0.

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
    if m.get("event") == "progress": print(m["phase"], m["step"], "/", m["total_steps"])
print(m["status"], m.get("image"))
send({"id": 3, "method": "shutdown"})
```

## Tests

`tests/server/test_protocol.py` uses the standard library only. The server's JSON output helpers (`tools/server/json_out.h`; the server links only the C API, so it cannot use the pipeline's report renderer) are checked by the `cpu` gtest suite `ServerJson.*` (`tests/unit/test_server_json.cpp`): strings and numbers are written exactly as `pipeline::json_escape` / `json_number` write them, numbers ignore the C and C++ locales, and a NULL report becomes `null`.

| ctest name | Label | What it runs |
|---|---|---|
| `server_protocol` | `cpu` | No GPU or weights needed (details below) |
| `server_protocol_gpu` | `gpu` | With `MOSSINFER_GPU_TESTS=1`: a 256², 2-step generate (events, PNG, report with `dit_blocks` and `dit_stream`, `busy`, validation, NUL in the prompt), cancel mid-run, the engine reused after a cancel, cancel on shutdown |

`server_protocol` covers:
- `hello`, `status`, and idle `cancel` / `unload`
- malformed input of every kind, unknown methods and params (an unknown or non-string `block_path` included), and NUL in strings
- `load` errors mapped from engine statuses
- `shutdown`, end of input, and `--version` / usage errors
- stdout set up with stderr sent to a file, to `DEVNULL`, merged into stdout, or closed. On Linux, `/proc` shows exactly one fd on the protocol pipe, at 3 or higher, close-on-exec.
- stdin closed
- `--parent-pid` for the server's own parent and for another process, and for a process that is already gone

```
MOSS_SERVER=<build>/moss-server python3 tests/server/test_protocol.py
MOSSINFER_GPU_TESTS=1 MOSS_SERVER=<build>/moss-server MOSS_WEIGHTS=<models>/MOSS-Image python3 tests/server/test_protocol.py
```
