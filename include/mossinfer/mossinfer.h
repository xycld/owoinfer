/* SPDX-License-Identifier: Apache-2.0
 *
 * MossInfer C API: text-to-image generation with the native MOSS-Image engine.
 *
 * Stable C99 interface in the style of llama.h. No C++ type or exception crosses it. See docs/c-api.md.
 *
 * Conventions
 * - Strings are UTF-8 and NUL-terminated, paths included (Windows too). A string ends at its first NUL, so it cannot
 *   carry one (moss-server rejects an escaped \u0000 instead of truncating).
 * - Versioned structs start with `size_t struct_size`. Fill them with the matching *_init() function, then change
 *   the fields you need:
 *       mi_gen_params p;
 *       mi_gen_params_init(&p, sizeof p);
 *   The library rejects a struct_size it does not know with MI_ERR_ABI_MISMATCH. Structs only grow at the end;
 *   MI_ABI_VERSION changes when an existing field or function changes meaning.
 * - Struct layouts are explicit: every field sits at its natural alignment with no compiler-inserted padding, and the
 *   gaps are named `_reserved*` int32 fields. *_init() zeroes them; input structs with a nonzero one are rejected with
 *   MI_ERR_INVALID_ARGUMENT, output structs get them zeroed. The layouts (64-bit targets) are frozen in docs/c-api.md
 *   and checked at compile time by the library and by a C test.
 * - Functions that can fail return mi_status. On failure, mi_last_error_message() returns the reason. It is
 *   thread-local and stays valid until the next failing mi_* call on the same thread.
 * - Enums are 32-bit (each has a *_MAX_ENUM = 0x7fffffff member for that); unknown values are rejected with
 *   MI_ERR_INVALID_ARGUMENT.
 * - Handles (mi_engine, mi_result) are opaque. NULL handles are rejected with MI_ERR_INVALID_ARGUMENT, or by a
 *   0 / NULL return from accessors.
 *
 * Threading
 * - Each engine runs one generation at a time. A concurrent mi_generate() on the same engine fails with MI_ERR_BUSY.
 *   Different engines may run concurrently, but they share the GPU and the memory budgets are per engine.
 * - mi_cancel() is thread-safe and may be called from any thread, including from inside a callback.
 * - Callbacks run on the thread that called mi_generate(), synchronously, while the GPU waits. Keep them short.
 *   They must not throw (C++), longjmp, or call mi_generate() / mi_engine_free() on the same engine.
 * - mi_engine_free() must not race with a running mi_generate() on that engine.
 * - mi_result objects are independent of their engine. They may outlive it and be used from any one thread at a time.
 */
#ifndef MOSSINFER_MOSSINFER_H
#define MOSSINFER_MOSSINFER_H

#include <stddef.h>
#include <stdint.h>

/* Symbol visibility. Define MOSSINFER_SHARED when using the shared library (the CMake package does this). The library
 * build itself also defines MOSSINFER_BUILD. With neither defined (static linking), MOSSINFER_API is empty on Windows. */
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(MOSSINFER_SHARED)
#if defined(MOSSINFER_BUILD)
#define MOSSINFER_API __declspec(dllexport)
#else
#define MOSSINFER_API __declspec(dllimport)
#endif
#else
#define MOSSINFER_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define MOSSINFER_API __attribute__((visibility("default")))
#else
#define MOSSINFER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------------ version */

#define MI_VERSION_MAJOR 0
#define MI_VERSION_MINOR 1
#define MI_VERSION_PATCH 0
/* Incremented on every incompatible change of an existing function, struct field or enum value. */
#define MI_ABI_VERSION 1

/* "MAJOR.MINOR.PATCH" of the library actually loaded (static storage). */
MOSSINFER_API const char* mi_version(void);
/* MI_ABI_VERSION of the library actually loaded. A consumer compiled against another value must not use it. */
MOSSINFER_API int32_t mi_abi_version(void);

/* ------------------------------------------------------------------------------------------------ status */

typedef enum mi_status {
  MI_OK = 0,
  MI_CANCELLED = 1,                /* mi_generate(): stopped by mi_cancel(); a result is still returned */
  MI_ERR_INVALID_ARGUMENT = -1,    /* NULL handle or pointer, value out of range, invalid request */
  MI_ERR_ABI_MISMATCH = -2,        /* struct_size this library does not know */
  MI_ERR_NOT_SUPPORTED = -3,       /* feature not in this build (e.g. ComfyUI single-file checkpoints) */
  MI_ERR_BUSY = -4,                /* the engine is already generating */
  MI_ERR_OUT_OF_MEMORY = -5,       /* request does not fit the device or host memory budgets in any mode */
  MI_ERR_MODEL = -6,               /* model files missing, incomplete or malformed */
  MI_ERR_IO = -7,                  /* file system error (reading weights, writing an image) */
  MI_ERR_CUDA = -8,                /* CUDA driver/runtime error, no usable device */
  MI_ERR_INTERNAL = -9,            /* anything else */
  MI_STATUS_MAX_ENUM = 0x7fffffff
} mi_status;

/* Static name of a status, e.g. "MI_ERR_BUSY"; "MI_UNKNOWN_STATUS" for values not listed above. */
MOSSINFER_API const char* mi_status_string(mi_status status);
/* Reason for the last failing call on this thread ("" if none). Never NULL. */
MOSSINFER_API const char* mi_last_error_message(void);

/* ------------------------------------------------------------------------------------------------ capabilities */

#define MI_CAP_PREVIEW 0x1u        /* mi_gen_params.preview_callback is called */
#define MI_CAP_COMFYUI_MODELS 0x2u /* MI_MODEL_FORMAT_COMFYUI is accepted */

/* Bit set of MI_CAP_* features this build implements. */
MOSSINFER_API uint32_t mi_capabilities(void);

/* ------------------------------------------------------------------------------------------------ device probe */

typedef struct mi_device_info {
  size_t struct_size;
  int32_t index;                /* CUDA device ordinal (mi_engine_params.device_index) */
  char name[256];               /* UTF-8, NUL-terminated */
  char pci_bus_id[32];          /* "0000:01:00.0"; "" if unavailable */
  int32_t compute_major;
  int32_t compute_minor;
  int32_t _reserved0;
  uint64_t total_memory_bytes;
  uint64_t free_memory_bytes;   /* at probe time; 0 for unsupported devices */
  /* This build can run on the device: compute capability >= 8.0, a driver new enough for the build's CUDA version,
   * and a kernel image for the architecture. */
  int32_t supported;
  char unsupported_reason[256]; /* why not, when supported == 0; "" otherwise */
  int32_t _reserved1;
} mi_device_info;

typedef struct mi_probe_info {
  size_t struct_size;
  int32_t driver_cuda_version;   /* newest CUDA version the installed driver supports, e.g. 13000; 0: no driver */
  int32_t runtime_cuda_version;  /* CUDA version this library was built with, e.g. 13040 */
  int32_t driver_compatible;     /* the driver can run this build */
  int32_t device_count;          /* CUDA devices visible to the process */
  int32_t supported_device_count;
  char build_flavour[16];        /* release flavour of this library: "cu13" or "cu12" */
  /* Release flavour this machine should use: "cu13", "cu12", or "" when neither can run (no or too old driver, no
   * device with compute capability >= 8.0). */
  char recommended_flavour[16];
  char message[512];             /* human-readable summary */
  int32_t _reserved0;
} mi_probe_info;

/* Fills `info` (struct_size set by the caller). Works without a GPU: a machine without driver or device returns MI_OK
 * with device_count == 0 and the reason in `message`. Creates no CUDA context. */
MOSSINFER_API mi_status mi_probe(mi_probe_info* info);
/* Fills `info` (struct_size set by the caller) for device `index` in [0, device_count). The free memory comes from NVML
 * when the driver ships it, else from a temporary CUDA context (driver API) that is released before returning: the
 * call leaves no CUDA context behind and does not change the calling thread's current context. */
MOSSINFER_API mi_status mi_probe_device(int32_t index, mi_device_info* info);

/* ------------------------------------------------------------------------------------------------ engine */

typedef struct mi_engine mi_engine;

typedef enum mi_mode {
  MI_MODE_MIN_MEMORY = 0, /* smallest device peak, then host memory, then time */
  MI_MODE_BALANCED = 1,   /* least time among plans within the budgets (default) */
  MI_MODE_MAX_SPEED = 2,  /* least time, larger host target and embedding cache */
  MI_MODE_MAX_ENUM = 0x7fffffff
} mi_mode;

typedef enum mi_model_format {
  MI_MODEL_FORMAT_AUTO = 0,      /* a directory is a Hugging Face bundle; a file is a ComfyUI checkpoint */
  MI_MODEL_FORMAT_HF_BUNDLE = 1, /* {text_encoder, tokenizer, transformer, vae} directory (base or sft) */
  MI_MODEL_FORMAT_COMFYUI = 2,   /* reserved: ComfyUI models folder / single file (MI_ERR_NOT_SUPPORTED for now) */
  MI_MODEL_FORMAT_MAX_ENUM = 0x7fffffff
} mi_model_format;

/* DiT block implementation. All give the same image bits; they differ in speed and memory. */
typedef enum mi_block_path {
  MI_BLOCK_PATH_DEFAULT = 0,      /* the engine's default: fused */
  MI_BLOCK_PATH_OP_BY_OP = 1,     /* one kernel per op */
  /* Fused GEMM epilogues and QK-norm+RoPE. Runs op by op when the engine's startup self-check finds the fused GEMMs
   * not bit-identical on this GPU (mi_result_block_path_note() then says why). */
  MI_BLOCK_PATH_FUSED = 2,
  /* Fused, replayed from CUDA graphs when every DiT block is resident (else fused, eagerly). Holds graph memory. */
  MI_BLOCK_PATH_FUSED_GRAPHS = 3,
  MI_BLOCK_PATH_MAX_ENUM = 0x7fffffff
} mi_block_path;

typedef struct mi_engine_params {
  size_t struct_size;
  const char* model_path;          /* required: bundle directory */
  mi_model_format model_format;    /* default MI_MODEL_FORMAT_AUTO */
  mi_mode mode;                    /* default MI_MODE_BALANCED */
  int32_t device_index;            /* CUDA device ordinal, default 0 */
  int32_t _reserved0;              /* must be 0 */
  /* Cap on this process's device memory, CUDA context included; 0 (default) = free memory - vram_reserve_bytes.
   * Nonzero values must be in [1 MiB, 1 TiB]. */
  uint64_t vram_budget_bytes;
  uint64_t vram_reserve_bytes;     /* device memory left outside the engine's arena; default 384 MiB */
  uint64_t host_ram_budget_bytes;  /* cap on private host memory; 0 (default) = the mode's target */
  const char* cache_dir;           /* embedding-cache disk spill directory; NULL or "" (default) = memory only */
  int32_t embedding_cache;         /* reuse text embeddings of repeated prompts; default 1 */
  int32_t direct_io;               /* O_DIRECT weight reads (cold-cache behaviour); default 0 */
  mi_block_path block_path;        /* default MI_BLOCK_PATH_DEFAULT */
  int32_t _reserved1;              /* must be 0 */
} mi_engine_params;

/* Fills `params` with the defaults; struct_size must be sizeof(mi_engine_params) of the caller's header. */
MOSSINFER_API mi_status mi_engine_params_init(mi_engine_params* params, size_t struct_size);

/* Creates an engine on `params->device_index`. Checks the parameters and the model path and initializes CUDA; the
 * weights load on the first mi_generate() (and stay resident as the plan allows). The strings are copied. The calling
 * thread's current CUDA context is restored before returning (also by mi_generate()). */
MOSSINFER_API mi_status mi_engine_create(const mi_engine_params* params, mi_engine** out_engine);
/* Releases the engine and its device memory. NULL is a no-op. Never fails and never throws: a CUDA error while
 * releasing is ignored, and mi_last_error_message() is left unchanged. */
MOSSINFER_API void mi_engine_free(mi_engine* engine);

/* Requests the generation running on `engine` to stop. It stops within one text-encoder layer, DiT block or decode
 * band, and mi_generate() returns MI_CANCELLED. No effect when nothing runs (a cancel never carries over to the next
 * generation). Thread-safe. */
MOSSINFER_API mi_status mi_cancel(mi_engine* engine);

/* ------------------------------------------------------------------------------------------------ generation */

typedef struct mi_result mi_result;

typedef enum mi_sampler {
  MI_SAMPLER_EULER = 0, /* the reference flow-matching Euler step (default) */
  MI_SAMPLER_AB2 = 1,   /* 2nd-order Adams-Bashforth */
  MI_SAMPLER_MAX_ENUM = 0x7fffffff
} mi_sampler;

typedef enum mi_phase {
  MI_PHASE_TEXT_ENCODER = 0, /* entered even on a full embedding-cache hit */
  MI_PHASE_DIT_LOAD = 1,     /* entered even when the DiT is resident */
  MI_PHASE_DENOISE = 2,
  MI_PHASE_DECODE = 3,
  MI_PHASE_MAX_ENUM = 0x7fffffff
} mi_phase;

typedef struct mi_progress {
  size_t struct_size;
  mi_phase phase;
  int32_t step;            /* MI_PHASE_DENOISE: sampling steps finished (0 when the phase starts); else 0 */
  int32_t total_steps;     /* sampling steps scheduled (may be below the requested steps after snapping) */
  int32_t _reserved0;
  double step_seconds;     /* wall time of the step just finished; 0 at a phase start */
  double elapsed_seconds;  /* since mi_generate() was called */
} mi_progress;

/* Called when a phase starts and after every sampling step. */
typedef void (*mi_progress_fn)(const mi_progress* progress, void* user_data);
/* Called once per generation, after planning and before the first phase. `explanation` is the plan in several
 * human-readable lines, `plan_json` the plan as one JSON object. Both are valid during the call only. */
typedef void (*mi_plan_fn)(const char* explanation, const char* plan_json, void* user_data);
/* Low-resolution RGB8 preview of the current latents. Not called unless mi_capabilities() has MI_CAP_PREVIEW. */
typedef void (*mi_preview_fn)(const uint8_t* rgb, int32_t width, int32_t height, int32_t step, void* user_data);

/* One generation. Defaults mirror the MOSS-Image app's parameters.json. */
typedef struct mi_gen_params {
  size_t struct_size;
  const char* prompt;              /* required, not blank */
  const char* negative_prompt;     /* NULL = "" */
  int32_t width;                   /* positive multiple of 16, <= 4096, width * height <= 4.2 MP; default 1024 */
  int32_t height;                  /* default 1024 */
  int32_t steps;                   /* 1..10000; default 50 */
  int32_t _reserved0;              /* must be 0 */
  double cfg;                      /* guidance scale >= 0; 1 disables the negative branch; default 3.0 */
  double shift;                    /* schedule shift > 0; default 5.0 */
  int64_t seed;                    /* 0..2^53-1, or -1 for random (the seed used is in the result); default 42 */
  /* DiT blocks skipped in the negative branch, unique, in [0, 42). Default {22} (count 1, library-owned storage);
   * count 0 = none. The array is read during mi_generate() only. */
  const int32_t* uncond_skip_layers;
  size_t uncond_skip_layers_count;
  /* Skip-layer guidance: an extra conditional row with these blocks skipped. Default none. */
  const int32_t* slg_layers;
  size_t slg_layers_count;
  double slg_scale;                /* >= 0; default 0 */
  double slg_start;                /* 0 <= slg_start <= slg_stop <= 1, fractions of the schedule; default 0 */
  double slg_stop;                 /* default 1 */
  int32_t max_sequence_length;     /* text tokens per prompt, 1..4096; default 4096 */
  mi_sampler sampler;              /* default MI_SAMPLER_EULER */
  mi_progress_fn progress_callback; /* optional */
  mi_plan_fn plan_callback;         /* optional */
  mi_preview_fn preview_callback;   /* optional; see MI_CAP_PREVIEW */
  int32_t preview_interval;         /* steps between previews; default 1 */
  int32_t _reserved1;               /* must be 0 */
  void* user_data;                  /* passed to every callback */
} mi_gen_params;

/* Fills `params` with the defaults; struct_size must be sizeof(mi_gen_params) of the caller's header. */
MOSSINFER_API mi_status mi_gen_params_init(mi_gen_params* params, size_t struct_size);
/* Checks `params` the way mi_generate() does, without an engine or GPU. MI_ERR_INVALID_ARGUMENT names the problem. */
MOSSINFER_API mi_status mi_gen_params_validate(const mi_gen_params* params);

/* Runs one generation to completion. Returns MI_OK with the image, or MI_CANCELLED after mi_cancel() (the result then
 * has no image but carries the timings so far); in both cases *out_result is set and must be released with
 * mi_result_free(). On any other status *out_result is NULL. The engine stays usable after every status except
 * MI_ERR_CUDA, after which it should be freed. */
MOSSINFER_API mi_status mi_generate(mi_engine* engine, const mi_gen_params* params, mi_result** out_result);

/* ------------------------------------------------------------------------------------------------ results */

typedef struct mi_timings {
  size_t struct_size;
  double total_s;
  double load_s;     /* index_load_s + dit_load_s + vae_load_s */
  double encode_s;   /* tokenize_s + text_encode_s + refine_s */
  double denoise_s;
  double decode_s;
  double index_load_s;   /* weight index and tokenizer, first use of a bundle only */
  double tokenize_s;
  double plan_s;
  double text_encode_s;  /* 0 on a full embedding-cache hit */
  double dit_load_s;     /* 0 when the DiT was resident */
  double refine_s;
  double vae_load_s;
} mi_timings;

typedef struct mi_memory_stats {
  size_t struct_size;
  uint64_t arena_capacity_bytes;
  uint64_t arena_high_water_bytes;  /* peak arena use during the generation */
  uint64_t device_peak_bytes;       /* peak device memory used by the engine (free at creation - free now) */
  uint64_t device_free_at_start_bytes;
  uint64_t host_private_start_bytes;
  uint64_t host_private_peak_bytes;
} mi_memory_stats;

/* DiT block streaming over the whole denoise (all zero when every block was resident). The DiT streams when it does
 * not fit the device budget, or in MI_MODE_MIN_MEMORY: the streamed blocks are copied to the device every step, ahead
 * of their compute. */
typedef struct mi_dit_stream_stats {
  size_t struct_size;
  int32_t streamed;         /* some blocks were streamed */
  int32_t resident_blocks;  /* of 42 */
  int32_t buffers;          /* device block buffers the streamed blocks rotate through */
  int32_t stalls;           /* streamed blocks the compute waited for (> 50 us each) */
  uint64_t bytes;           /* copied to the device, all steps */
  double copy_seconds;      /* copy streams busy */
  double stall_seconds;     /* compute waiting for streamed weights */
  double h2d_gbps;          /* bytes / copy_seconds, in GB/s (1e9) */
  double hidden_fraction;   /* share of the copy time hidden behind compute, 0..1 */
  char source[16];          /* "file" (page cache or disk), "host_pinned", or "" when nothing streamed */
} mi_dit_stream_stats;

MOSSINFER_API int32_t mi_result_width(const mi_result* result);
MOSSINFER_API int32_t mi_result_height(const mi_result* result);
/* Row-major RGB8 [height][width][3], owned by the result; NULL when cancelled. `out_bytes` may be NULL. */
MOSSINFER_API const uint8_t* mi_result_rgb(const mi_result* result, size_t* out_bytes);
MOSSINFER_API int32_t mi_result_cancelled(const mi_result* result);
MOSSINFER_API uint64_t mi_result_seed(const mi_result* result); /* the seed actually used */
MOSSINFER_API int32_t mi_result_steps_scheduled(const mi_result* result);
MOSSINFER_API int32_t mi_result_steps_run(const mi_result* result);
/* `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_result_timings(const mi_result* result, mi_timings* out);
MOSSINFER_API mi_status mi_result_memory(const mi_result* result, mi_memory_stats* out);
MOSSINFER_API mi_status mi_result_dit_stream(const mi_result* result, mi_dit_stream_stats* out);
/* The DiT block path the denoise actually ran (never MI_BLOCK_PATH_DEFAULT, except for a NULL result). It can differ
 * from mi_engine_params.block_path; mi_result_block_path_note() then says why. */
MOSSINFER_API mi_block_path mi_result_block_path(const mi_result* result);
/* Why the block path differs from the requested one; "" when it does not. Owned by the result. */
MOSSINFER_API const char* mi_result_block_path_note(const mi_result* result);
/* The execution plan in several human-readable lines; owned by the result. */
MOSSINFER_API const char* mi_result_plan_explanation(const mi_result* result);
/* The generation report as one JSON document, the same shape as `moss-infer generate --json-report` ("image" is the
 * last path given to mi_save_png(), "" before). Owned by the result, valid until the next call of this function or
 * mi_save_png() on it. */
MOSSINFER_API const char* mi_result_report_json(mi_result* result);
/* Writes the image as a PNG (atomically: a temporary file renamed over `path`). MI_ERR_INVALID_ARGUMENT when the
 * result has no image (cancelled). */
MOSSINFER_API mi_status mi_save_png(mi_result* result, const char* path);
/* NULL is a no-op. */
MOSSINFER_API void mi_result_free(mi_result* result);

#ifdef __cplusplus
}
#endif

#endif /* MOSSINFER_MOSSINFER_H */
