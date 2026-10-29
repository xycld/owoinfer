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
 *   Structs only grow at the end, and the library accepts every size a struct has had (the 0.1.0 sizes and today's):
 *   it reads only the caller's struct_size bytes of an input struct (the later fields keep their defaults) and writes
 *   only struct_size bytes of an output struct. Any other struct_size is rejected with MI_ERR_ABI_MISMATCH.
 *   MI_ABI_VERSION (the shared library's SONAME) changes when the binary interface does: an exported function is
 *   removed or changes signature, or a struct field or enum value moves or changes type. A field or value retired
 *   without such a change stays in place as "reserved"; a behaviour change that callers must act on bumps
 *   MI_VERSION_MAJOR (2.0: the tier is required, the certificate fields are reserved).
 * - Struct layouts are explicit: every field sits at its natural alignment with no compiler-inserted padding, and the
 *   gaps are named `_reserved*` int32 fields. *_init() zeroes them; input structs with a nonzero one are rejected with
 *   MI_ERR_INVALID_ARGUMENT, output structs get them zeroed. The layouts (64-bit targets) are frozen in docs/c-api.md
 *   and checked at compile time by the library and by a C test.
 * - Functions that can fail return mi_status. On failure, mi_last_error_message() returns the reason. It is
 *   thread-local and stays valid until the next failing mi_* call on the same thread.
 * - Enums are 32-bit (each has a *_MAX_ENUM = 0x7fffffff member for that); unknown values are rejected with
 *   MI_ERR_INVALID_ARGUMENT.
 * - Handles (mi_engine, mi_result, mi_file_report, mi_conditioning) are opaque. NULL handles are rejected with
 *   MI_ERR_INVALID_ARGUMENT, or by a 0 / NULL return from accessors.
 *
 * Threading
 * - Each engine runs one job at a time: mi_generate(), mi_engine_preload(), mi_engine_release() or a component call
 *   (below). A concurrent one on the same engine fails with MI_ERR_BUSY. Different engines may run concurrently, but
 *   they share the GPU and the memory budgets are per engine.
 * - mi_cancel() is thread-safe and may be called from any thread, including from inside a callback. It stops the job
 *   running on the engine, whichever it is.
 * - Callbacks run on the thread that called mi_generate() (or mi_engine_preload()), synchronously, while the GPU
 *   waits. Keep them short. They must not throw (C++), longjmp, or call mi_generate(), mi_engine_preload(),
 *   mi_engine_release() or mi_engine_free() on the same engine.
 * - mi_engine_free() may be called while a job runs on that engine on another thread: it cancels the job, waits for it
 *   to return, then frees (2.1). A call not yet started when mi_engine_free() begins (mi_cancel() included, or one still
 *   reading its arguments) must not overlap it. Using the engine after mi_engine_free() returns is the caller's bug.
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

#define MI_VERSION_MAJOR 2
#define MI_VERSION_MINOR 1
#define MI_VERSION_PATCH 0
/* Incremented on every incompatible change of the binary interface (see the conventions above). */
#define MI_ABI_VERSION 1

/* What the unset markers of mi_gen_params resolve to (see "Unset markers" below), for a UI to pre-fill its fields. */
#define MI_DEFAULT_STEPS 50
#define MI_DEFAULT_CFG 3.0
#define MI_DEFAULT_DENOISE_STRENGTH 0.75 /* with an init image (img2img, inpainting); 1.0 without one */

/* "MAJOR.MINOR.PATCH" of the library actually loaded (static storage). */
MOSSINFER_API const char* mi_version(void);
/* MI_ABI_VERSION of the library actually loaded. A consumer compiled against another value must not use it. */
MOSSINFER_API int32_t mi_abi_version(void);

/* ------------------------------------------------------------------------------------------------ status */

typedef enum mi_status {
  MI_OK = 0,
  MI_CANCELLED = 1,                /* stopped by mi_cancel(): a generation or preload (a result is returned), or a release
                                    * before it freed anything */
  MI_ERR_INVALID_ARGUMENT = -1,    /* NULL handle or pointer, value out of range, invalid request */
  MI_ERR_ABI_MISMATCH = -2,        /* struct_size this library does not know */
  MI_ERR_NOT_SUPPORTED = -3,       /* feature not in this build (e.g. ComfyUI single-file checkpoints) */
  MI_ERR_BUSY = -4,                /* the engine is already running a job (generation, preload or release) */
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
#define MI_CAP_COMFYUI_MODELS 0x2u /* MI_MODEL_FORMAT_COMFYUI (a ComfyUI file as model_path) is accepted */
/* mi_engine_params.dit_file / text_encoder_file / vae_file: single component files, ComfyUI / community INT8 ConvRot
 * and W4A8 files included. */
#define MI_CAP_MODEL_FILES 0x4u
#define MI_CAP_LORA 0x8u           /* mi_gen_params.loras, mi_engine_params.lora_strict, mi_result_lora() */
#define MI_CAP_TIERS 0x10u         /* mi_engine_params.tier / attention, mi_result_tier() */
#define MI_CAP_SAMPLERS 0x20u      /* the samplers after MI_SAMPLER_AB2, the schedulers and the noise options */
#define MI_CAP_IMAGE_INPUT 0x40u   /* img2img and inpainting: mi_gen_params.init_image / mask, mi_result_image() */
#define MI_CAP_HIRES 0x80u         /* the hires fix: mi_gen_params.hires_* */
#define MI_CAP_DISTILL 0x100u      /* distill bundles and time embeddings: distill_dir, time_embedding_file */
/* cuBLASLt is loaded at run time: the bundled copy by default, a compatible system copy (a CUDA install) in a package
 * without one or with MOSSINFER_CUDA_LIBS=system (also MOSSINFER_CUBLASLT; docs/integration.md), and
 * mi_probe_info.cublaslt_* reports the copy. */
#define MI_CAP_SYSTEM_CUDA 0x200u
/* Control adapters: mi_gen_params.control_adapter / control_lora_strength / control_images, mi_result_control(), the
 * MI_TEXT_CONTROL_* texts and the MI_MESSAGE_CONTROL_* messages. */
#define MI_CAP_CONTROL 0x400u
/* The lossy accelerators: mi_gen_params.cfg_truncation / step_cache / step_cache_threshold, mi_result_accel(),
 * mi_result_accel_cached_steps() and MI_TEXT_ACCELERATORS. */
#define MI_CAP_ACCEL 0x800u
/* mi_engine_preload(): loads the models for a request before the first mi_generate(). */
#define MI_CAP_PRELOAD 0x1000u
/* mi_engine_release(): a soft unload that frees the device memory and keeps the engine, optionally with the weights
 * kept in host RAM (keep_host_copy) for a fast reload; mi_result_dit_load() says where a DiT load read from. */
#define MI_CAP_RELEASE 0x2000u
/* Reserved since 2.0 (never set): it announced mi_engine_params.certificate_dirs, which is now ignored. */
#define MI_CAP_CERTIFICATE_DIRS 0x4000u
/* mi_engine_params.te_host_pin: whether a host-resident text encoder may hold a pinned copy. */
#define MI_CAP_TE_HOST_PIN 0x8000u
/* mi_progress.fraction and eta_seconds: the whole job's progress and remaining time. */
#define MI_CAP_PROGRESS_FRACTION 0x10000u
/* mi_inspect_file(): what a model file is to the engine, without an engine or a GPU. */
#define MI_CAP_INSPECT 0x20000u
/* The component API ("components" below): mi_session_*, mi_encode_text(), mi_conditioning_*, mi_run_*,
 * mi_denoise_step(), mi_vae_*, the pure helpers and MI_TEXT_COMPONENT_NOTE. */
#define MI_CAP_COMPONENTS 0x40000u
/* Reference images of edit adapters (moss-edit-lora files such as Moss-Image-Edit): mi_gen_params.reference_images /
 * reference_max_edge, MI_TEXT_REFERENCES and MI_MESSAGE_REFERENCE_WARNING. mi_generate() only (not the components). */
#define MI_CAP_REFERENCE_IMAGES 0x80000u

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
  /* Fields after the 576-byte layout (MI_CAP_SYSTEM_CUDA): the cuBLASLt copy the engine uses, searched for (and
   * loaded, without a CUDA context) by the first mi_probe() or mi_engine_create() of the process. */
  char cublaslt_source[16];      /* "system", "bundled" or "env" (MOSSINFER_CUBLASLT); "" when no copy is usable */
  int32_t cublaslt_version;      /* cublasLtGetVersion() of that copy, e.g. 130800 for 13.8.0; 0 when none */
  int32_t _reserved1;
  char cublaslt_path[512];       /* UTF-8; "" when none, or when the path does not fit (cublaslt_error says so) */
  char cublaslt_error[512];      /* why no copy is usable (every candidate tried, cut at 511 bytes); "" when one is */
} mi_probe_info;

/* Fills `info` (struct_size set by the caller: the current size or the 576-byte one). Works without a GPU: a machine
 * without driver or device returns MI_OK with device_count == 0 and the reason in `message`, and a machine without a
 * usable cuBLASLt returns MI_OK with cublaslt_source "" and the reason in cublaslt_error. Creates no CUDA context. */
MOSSINFER_API mi_status mi_probe(mi_probe_info* info);
/* Fills `info` (struct_size set by the caller) for device `index` in [0, device_count); any other index fails with
 * MI_ERR_INVALID_ARGUMENT before any free-memory query. Only device `index` has its free memory measured: from NVML
 * when the driver ships it, else from a temporary CUDA context (driver API) that is released before returning. The
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
  /* Reserved: a ComfyUI models folder / single file as model_path (MI_ERR_NOT_SUPPORTED for now). ComfyUI component
   * files load through dit_file / text_encoder_file / vae_file next to a bundle (MI_CAP_MODEL_FILES). */
  MI_MODEL_FORMAT_COMFYUI = 2,
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

/* Quality tier (MI_CAP_TIERS), the caller's choice. The weights fix the GEMM precision (an INT8 DiT runs INT8 GEMMs);
 * the tier decides the attention kernel. HIGH and TURBO default to Q8_FP8 where the GPU has FP8 (compute capability
 * 8.9+), else Q8_FP16; mi_engine_params.attention overrides it. A tier the loaded models cannot run fails
 * mi_generate() with MI_ERR_INVALID_ARGUMENT, naming the reason. */
typedef enum mi_tier {
  /* Reserved since 2.0: mi_engine_create() refuses it with MI_ERR_INVALID_ARGUMENT (choose one of the tiers below). */
  MI_TIER_AUTO = 0,
  MI_TIER_REFERENCE = 1, /* the DiT's bf16 GEMMs, exact attention; refused for quantized DiT or text encoder files */
  MI_TIER_HIGH = 2,      /* the loaded DiT's GEMMs (INT8 or bf16) plus 8-bit attention; refused for a 4-bit DiT */
  /* a 4-bit DiT (INT4 or NVFP4 file): its 4-bit GEMMs plus 8-bit attention; refused for any other DiT. NVFP4 runs on an
   * RTX 50 (compute capability 12.0) only; elsewhere the file is refused with MI_ERR_MODEL. */
  MI_TIER_TURBO = 3,
  MI_TIER_MAX_ENUM = 0x7fffffff
} mi_tier;

/* DiT attention kernel (MI_CAP_TIERS). The 8-bit modes are lossy (SageAttention-style). */
typedef enum mi_attention {
  MI_ATTENTION_DEFAULT = 0, /* the tier's choice */
  MI_ATTENTION_EXACT = 1,   /* FlashAttention-2 */
  MI_ATTENTION_Q8_FP16 = 2,
  MI_ATTENTION_Q8_FP16_F32ACC = 3,
  MI_ATTENTION_Q8_FP8 = 4,
  MI_ATTENTION_Q8_FP8_F32ACC = 5,
  MI_ATTENTION_MAX_ENUM = 0x7fffffff
} mi_attention;

/* Weight precision of a loaded model, for reporting (mi_tier_info). */
typedef enum mi_precision {
  MI_PRECISION_BF16 = 0, /* the bundle, or a bf16 single file */
  MI_PRECISION_INT8 = 1, /* INT8 ConvRot (a DiT with any INT8 linear counts as INT8) */
  MI_PRECISION_W4A8 = 2, /* 4-bit weights, 8-bit activations (text encoder files) */
  MI_PRECISION_W4A4 = 3, /* 4-bit weights and activations: an INT4 DiT (the Turbo tier) */
  /* 4-bit float weights and activations (E2M1, UE4M3 block scales): an NVFP4 DiT (the Turbo tier on an RTX 50); a DiT
   * with any NVFP4 linear counts as NVFP4 (its keep-list INT8 linears included) */
  MI_PRECISION_NVFP4 = 4,
  MI_PRECISION_MAX_ENUM = 0x7fffffff
} mi_precision;

/* How the DiT reads its weight files when it loads (mi_engine_params.dit_direct_io). mi_engine_params.direct_io = 1
 * forces direct reads for every weight read; dit_direct_io only refines the DiT's load when direct_io is 0. Streamed DiT
 * blocks (a DiT that does not fit the device) read buffered every step unless direct_io. */
typedef enum mi_direct_io {
  /* Adaptive (default): direct I/O, unless it gains nothing on the files (tmpfs, network file systems, compressed
   * extents, a refused open); buffered for a parking reload. With most of the files in the page cache already (Linux:
   * probed before every load; Windows: once buffered loads measure faster), the mode this machine measured faster from
   * the cache (each is tried once while its rate is unknown; the rates are kept in the hardware profile under
   * cache_dir). */
  MI_DIRECT_IO_AUTO = 0,
  MI_DIRECT_IO_BUFFERED = 1, /* buffered reads (the page cache) */
  MI_DIRECT_IO_DIRECT = 2,   /* direct I/O (O_DIRECT / FILE_FLAG_NO_BUFFERING); a refused open still reads buffered */
  MI_DIRECT_IO_MAX_ENUM = 0x7fffffff
} mi_direct_io;

/* Whether a host-resident text encoder may hold a pinned host copy (MI_CAP_TE_HOST_PIN). A pinned copy uploads faster
 * and costs page-locked RAM; one whose allocation fails streams from the files instead (MI_TEXT_TE_NOTE says so). */
typedef enum mi_te_host_pin {
  MI_TE_HOST_PIN_AUTO = 0, /* where the machine clearly has the room (default) */
  MI_TE_HOST_PIN_ON = 1,   /* whenever the copy fits the host limit */
  MI_TE_HOST_PIN_OFF = 2,  /* never */
  MI_TE_HOST_PIN_MAX_ENUM = 0x7fffffff
} mi_te_host_pin;

typedef struct mi_engine_params {
  size_t struct_size;
  const char* model_path;          /* required: bundle directory */
  mi_model_format model_format;    /* default MI_MODEL_FORMAT_AUTO */
  mi_mode mode;                    /* default MI_MODE_BALANCED */
  int32_t device_index;            /* CUDA device ordinal, default 0 */
  int32_t _reserved0;              /* must be 0 */
  /* The three sizes share one range rule: 0, or a value in [1 MiB, 1 TiB]; anything else fails mi_engine_create()
   * with MI_ERR_INVALID_ARGUMENT. vram_budget_bytes caps this process's device memory, CUDA context included;
   * 0 (default) = free memory - vram_reserve_bytes. */
  uint64_t vram_budget_bytes;
  uint64_t vram_reserve_bytes;     /* device memory left outside the engine's arena; default 384 MiB, 0 = none */
  uint64_t host_ram_budget_bytes;  /* cap on private host memory; 0 (default) = the mode's target */
  const char* cache_dir;           /* embedding-cache disk spill directory; NULL or "" (default) = memory only */
  int32_t embedding_cache;         /* reuse text embeddings of repeated prompts; default 1 */
  int32_t direct_io;               /* O_DIRECT weight reads (cold-cache behaviour); default 0 */
  mi_block_path block_path;        /* default MI_BLOCK_PATH_DEFAULT */
  int32_t _reserved1;              /* must be 0 */
  /* Fields after the 0.1.0 layout (80 bytes), since 0.2.0. */
  /* Component files used instead of the bundle's folders (MI_CAP_MODEL_FILES); the tokenizer and configs always come
   * from model_path. UTF-8; NULL or "" = the bundle's. A given file must exist. */
  const char* dit_file;            /* single-file DiT: INT8 ConvRot, mixed INT8/bf16 or bf16 (ComfyUI layout) */
  const char* text_encoder_file;   /* INT8 ConvRot or W4A8 Qwen3.5 text encoder */
  const char* vae_file;            /* single FLUX.2 VAE file */
  mi_tier tier;                    /* required: REFERENCE, HIGH or TURBO (the default MI_TIER_AUTO is refused) */
  mi_attention attention;          /* default MI_ATTENTION_DEFAULT */
  int32_t lora_strict;             /* unknown LoRA keys and lossy merges fail the generation (MI_ERR_MODEL); default 0 */
  int32_t _reserved2;              /* must be 0 */
  /* Fields after the 120-byte layout (MI_CAP_DISTILL). A step-embedder replacement is per engine: */
  /* a time_embedding.safetensors (replacement step embedder and modulation biases, computed in fp32) applied to the
   * loaded DiT. UTF-8; NULL or "" = none. The file must exist. */
  const char* time_embedding_file;
  /* A distill bundle directory (time_embedding / lora / dataspace .safetensors). The engine takes its time embedding
   * (as time_embedding_file; both set with different files is MI_ERR_INVALID_ARGUMENT); mi_gen_params.distill_dir
   * takes the rest per generation. NULL or "" = none. */
  const char* distill_dir;
  /* Fields after the 136-byte layout (1.0.0). */
  /* Reserved since 2.0: ignored (neither read nor checked). Leave them NULL and 0. */
  const char* const* certificate_dirs;
  size_t certificate_dirs_count;
  mi_direct_io dit_direct_io;      /* default MI_DIRECT_IO_AUTO; see mi_direct_io for its interplay with direct_io */
  mi_te_host_pin te_host_pin;      /* default MI_TE_HOST_PIN_AUTO */
} mi_engine_params;

/* Fills the first `struct_size` bytes of `params` with the defaults. struct_size is sizeof(mi_engine_params) of the
 * caller's header: 160, 136 (1.0.0), 120 (an earlier 0.2.0 header) or 80 (0.1.0). */
MOSSINFER_API mi_status mi_engine_params_init(mi_engine_params* params, size_t struct_size);

/* Creates an engine on `params->device_index`. Checks the parameters, the model path and that the component files exist
 * (their format is checked when they load, at the first mi_generate(): MI_ERR_MODEL), loads cuBLASLt (no usable
 * copy: MI_ERR_CUDA, the message lists every candidate tried) and initializes CUDA; the weights load on the first
 * mi_generate() (and stay resident as the plan allows). The strings are copied. The calling thread's current CUDA
 * context is restored before returning (also by mi_generate()). */
MOSSINFER_API mi_status mi_engine_create(const mi_engine_params* params, mi_engine** out_engine);
/* Releases the engine and its device memory. NULL is a no-op. Never fails and never throws: a CUDA error while
 * releasing is ignored, and mi_last_error_message() is left unchanged. A job running on another thread is cancelled
 * and waited for first (2.1); called from a callback of the engine's running job (forbidden) it does nothing. */
MOSSINFER_API void mi_engine_free(mi_engine* engine);

/* Requests the generation running on `engine` to stop. It stops within one text-encoder layer, DiT block or decode
 * band, and mi_generate() returns MI_CANCELLED. It also stops a running mi_engine_preload() (MI_CANCELLED) and a running
 * mi_engine_release(): before it frees anything (MI_CANCELLED), else its host copy. No effect when nothing runs (a cancel
 * never carries over to the next job).
 * Thread-safe. */
MOSSINFER_API mi_status mi_cancel(mi_engine* engine);

/* ------------------------------------------------------------------------------------------------ generation */

typedef struct mi_result mi_result;

typedef enum mi_sampler {
  MI_SAMPLER_EULER = 0,           /* the reference flow-matching Euler step (default) */
  MI_SAMPLER_AB2 = 1,             /* 2nd-order Adams-Bashforth */
  /* MI_CAP_SAMPLERS. The stochastic ones (EULER_ANCESTRAL, DPMPP_2M_SDE, ER_SDE) draw their noise from `seed`. */
  MI_SAMPLER_EULER_ANCESTRAL = 2, /* uses eta and s_noise */
  MI_SAMPLER_HEUN = 3,            /* 2 model calls per step */
  MI_SAMPLER_DPMPP_2M = 4,
  MI_SAMPLER_DPMPP_2M_SDE = 5,    /* uses eta and s_noise */
  MI_SAMPLER_ER_SDE = 6,          /* uses s_noise */
  MI_SAMPLER_RES_MULTISTEP = 7,
  MI_SAMPLER_DATASPACE = 8,       /* few-step data-space sampler: needs dataspace_path; its schedule is the file's */
  MI_SAMPLER_MAX_ENUM = 0x7fffffff
} mi_sampler;

/* Sigma schedule (MI_CAP_SAMPLERS). MOSS with MI_SAMPLER_EULER is the official pipeline, bit for bit. */
typedef enum mi_scheduler {
  MI_SCHEDULER_MOSS = 0, /* the official grid snapped to the training sigmas (default; ignores flow_multiplier) */
  MI_SCHEDULER_SIMPLE = 1,
  MI_SCHEDULER_NORMAL = 2,
  MI_SCHEDULER_KARRAS = 3,
  MI_SCHEDULER_EXPONENTIAL = 4,
  MI_SCHEDULER_BETA = 5,
  MI_SCHEDULER_SGM_UNIFORM = 6,
  MI_SCHEDULER_MAX_ENUM = 0x7fffffff
} mi_scheduler;

/* Layout of the initial noise (MI_CAP_SAMPLERS). */
typedef enum mi_noise_layout {
  MI_NOISE_LAYOUT_OFFICIAL = 0,     /* one draw per image, seeded per image (default) */
  MI_NOISE_LAYOUT_BATCH_STREAM = 1, /* image 0 of one batch-shaped draw from the seed */
  MI_NOISE_LAYOUT_MAX_ENUM = 0x7fffffff
} mi_noise_layout;

/* Noise of the stochastic samplers (MI_CAP_SAMPLERS). */
typedef enum mi_noise_source {
  MI_NOISE_SOURCE_PHILOX = 0,        /* torch's CUDA generator on a GPU-independent grid: the same on every GPU (default) */
  MI_NOISE_SOURCE_PHILOX_DEVICE = 1, /* the same algorithm on this GPU's grid (torch.randn on this device) */
  MI_NOISE_SOURCE_BROWNIAN_TREE = 2, /* k-diffusion's Brownian tree, on the CPU (up to ~380 MB of host memory at 4 MP) */
  MI_NOISE_SOURCE_MAX_ENUM = 0x7fffffff
} mi_noise_source;

/* One runtime LoRA (MI_CAP_LORA). A fixed 16-byte layout without struct_size. */
typedef struct mi_lora {
  const char* path;  /* UTF-8, required; the file must exist */
  double strength;   /* finite; 1 = as trained */
} mi_lora;

/* An 8-bit image in memory (MI_CAP_IMAGE_INPUT), row-major and tightly packed: [height][width][channels]. A fixed
 * 24-byte layout without struct_size. The pixels are read (copied) during mi_gen_params_validate() / mi_generate()
 * only; the caller may free them when the call returns. */
typedef struct mi_image {
  const uint8_t* pixels;
  int32_t width;     /* 1..16384 */
  int32_t height;    /* 1..16384, width * height <= 64,000,000 */
  int32_t channels;  /* 3 (RGB) for an init image, 1 for a mask */
  int32_t _reserved0; /* must be 0 */
} mi_image;

/* Where the low-rank adapters of `loras` on the DiT's block GEMMs go (MI_CAP_REFERENCE_IMAGES), a fixed rule per
 * module and base precision:
 *   DEFAULT      every low-rank block adapter on the side branch, edit adapters (moss-edit-lora) included: strength
 *                changes are instant, loads fast, the weights untouched (for INT8 with no per-step cost: a DiT file
 *                with the adapter baked in, quantize --lora)
 *   MERGE        every low-rank block adapter merged: no per-step cost (about 6.5 % faster per step than the side
 *                branch on bf16 for the edit adapter), a slower load (bf16: merged on the GPU, about a second
 *                for a rank-64 adapter on every block linear; INT8: requantized on the host, tens of seconds, one
 *                MI_MESSAGE_LORA_WARNING per file, refused with lora_strict; an INT4 / NVFP4 module is MI_ERR_MODEL)
 *   SIDE_BRANCH  every low-rank block adapter on the side branch
 * Merging costs nothing per step but a load-time merge; an edit adapter's merged blocks keep no pristine copy, so its
 * strength or placement changes reload the DiT. Merged blocks that stream need a pinned host copy: when the host
 * budget cannot hold it the plan fails (MI_ERR_OUT_OF_MEMORY) and names the alternatives. */
typedef enum mi_lora_placement {
  MI_LORA_PLACEMENT_DEFAULT = 0,
  MI_LORA_PLACEMENT_MERGE = 1,
  MI_LORA_PLACEMENT_SIDE_BRANCH = 2,
  MI_LORA_PLACEMENT_MAX_ENUM = 0x7fffffff
} mi_lora_placement;

/* One reference image of an edit adapter (MI_CAP_REFERENCE_IMAGES). A fixed 24-byte layout without struct_size. The
 * image comes as pixels (1, 3 or 4 channels: converted like PIL convert("RGB"), alpha dropped) or as a PNG / JPEG file
 * (read as RGB, EXIF orientation applied), exactly one of the two. */
typedef struct mi_reference_image {
  const mi_image* image;  /* NULL = use image_path */
  const char* image_path; /* UTF-8; NULL or "" = use image */
  int32_t _reserved0;     /* must be 0 */
  int32_t _reserved1;     /* must be 0 */
} mi_reference_image;

/* One conditioning image of a control adapter (MI_CAP_CONTROL). A fixed 48-byte layout without struct_size. The image
 * comes as pixels or as a PNG / JPEG file, exactly one of the two, and is brought to each pass's size by
 * mi_gen_params.resize, like an init image. Pixels are read (copied) during mi_gen_params_validate() / mi_generate()
 * only. A file is decoded in the library by the same decoder as init_image_path (not memory-safe: decode untrusted
 * images yourself and pass pixels), as RGB (a grayscale file is replicated to RGB). It has no init function: set every
 * field (a zeroed entry has strength 0 and an empty window, i.e. no control; as trained is strength 1, start 0, end 1). */
typedef struct mi_control_image {
  const char* kind;        /* required: one of the adapter's kinds (MI_TEXT_CONTROL_KINDS), e.g. "depth"; unique */
  const mi_image* image;   /* RGB (3 channels), or grayscale (1, replicated to RGB: a depth map, near = white) */
  const char* image_path;  /* UTF-8; NULL or "" = none */
  double strength;         /* [0, 10]: multiplies the control latent (0 = no control); 1 = as trained */
  /* 0 <= start <= end <= 1: the control is active on the steps i of a pass with start <= i / steps < end (the
   * guidance windows' convention); outside it this kind gets no control. Use 0 and 1 for every step. */
  double start;
  double end;
} mi_control_image;

/* Step caching (MI_CAP_ACCEL). */
typedef enum mi_step_cache {
  MI_STEP_CACHE_OFF = 0, /* default */
  /* First-Block Cache: while the output of DiT block 0 changed by less than step_cache_threshold (relative L1) since
   * the last call that ran every block, the other blocks are skipped and their residual from that call is reused. */
  MI_STEP_CACHE_FBCACHE = 1,
  MI_STEP_CACHE_MAX_ENUM = 0x7fffffff
} mi_step_cache;

/* How mask values are read (mi_gen_params.mask / mask_path). */
typedef enum mi_mask_encoding {
  MI_MASK_U8 = 0,       /* 0..255 as given: >= 128 regenerates, < 128 keeps (default) */
  MI_MASK_BINARY01 = 1, /* every value 0 (keep) or 1 (regenerate), scaled to 0 / 255; any other value is rejected */
  MI_MASK_ENCODING_MAX_ENUM = 0x7fffffff
} mi_mask_encoding;

/* How an init image (and its mask) is brought to the request size (and a hires fix's first pass to its target). */
typedef enum mi_resize_mode {
  MI_RESIZE_CROP = 0,    /* cover the target keeping the aspect ratio, center-crop (default) */
  MI_RESIZE_FIT = 1,     /* fit inside keeping the aspect ratio, fill the bands from the edge rows / columns */
  MI_RESIZE_STRETCH = 2, /* scale each axis independently */
  MI_RESIZE_MAX_ENUM = 0x7fffffff
} mi_resize_mode;

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
  /* Fields after the 40-byte layout (MI_CAP_PROGRESS_FRACTION). The engine allocates the struct and sets struct_size:
   * read them only when struct_size >= 56. They cover the whole job (every phase, both passes of a hires fix, a
   * preload's loads), estimated from the plan's time estimate, the measured transfer rates for models not loaded yet,
   * and the phases and steps measured so far. */
  /* Share of the job done, 0..1, never decreasing within a job. Below 1 in every callback: the call's return is 1. */
  double fraction;
  double eta_seconds;      /* estimated seconds remaining; < 0 when unknown (no estimate yet) */
} mi_progress;

/* Called when a phase starts and after every sampling step; also during a GGUF dit_file's conversion (its first load,
 * before planning and the plan callback): MI_PHASE_DIT_LOAD with step 0, fraction 0 and eta_seconds < 0. */
typedef void (*mi_progress_fn)(const mi_progress* progress, void* user_data);
/* Called once per generation, after planning and before the first phase. `explanation` is the plan in several
 * human-readable lines, `plan_json` the plan as one JSON object. Both are valid during the call only. */
typedef void (*mi_plan_fn)(const char* explanation, const char* plan_json, void* user_data);
/* Latent preview (MI_CAP_PREVIEW): RGB8 [height][width][3] at the requested image size, a linear latent -> RGB map of
 * the x0 estimate after sampling step `step` (no VAE). Called after every preview_interval-th step and after the last
 * one, before that step's progress callback. `rgb` is valid during the call only. */
typedef void (*mi_preview_fn)(const uint8_t* rgb, int32_t width, int32_t height, int32_t step, void* user_data);

/* One generation. Defaults mirror the MOSS-Image app's parameters.json. */
typedef struct mi_gen_params {
  size_t struct_size;
  const char* prompt;              /* required, not blank */
  const char* negative_prompt;     /* NULL = "" */
  int32_t width;                   /* positive multiple of 16, <= 4096, width * height <= 4.2 MP; default 1024 */
  int32_t height;                  /* default 1024 */
  int32_t steps;                   /* 1..10000, or 0 = unset (default; see "Unset markers") */
  int32_t _reserved0;              /* must be 0 */
  double cfg;                      /* guidance scale >= 0 (1 disables the negative branch), or negative = unset (default) */
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
  int32_t preview_interval;         /* steps between previews, >= 1; default 1 */
  int32_t _reserved1;               /* must be 0 */
  void* user_data;                  /* passed to every callback */
  /* Fields after the 0.1.0 layout (168 bytes), since 0.2.0. */
  mi_scheduler scheduler;           /* default MI_SCHEDULER_MOSS */
  mi_noise_layout noise_layout;     /* default MI_NOISE_LAYOUT_OFFICIAL */
  mi_noise_source noise_source;     /* default MI_NOISE_SOURCE_PHILOX */
  /* mi_save_png() embeds the generation settings ("generation" and "parameters" text chunks); default 1 (0 for a
   * 168-byte struct). */
  int32_t save_metadata;
  double flow_multiplier;           /* flow sigma multiplier, finite and > 0; default 1.0 */
  double eta;                       /* EULER_ANCESTRAL, DPMPP_2M_SDE; finite and >= 0; default 1.0 */
  double s_noise;                   /* stochastic samplers' noise multiplier; finite and >= 0; default 1.0 */
  /* MI_SAMPLER_DATASPACE's parameters file (UTF-8; required by it, and then it must exist); ignored by the other
   * samplers. Default NULL. */
  const char* dataspace_path;
  /* Runtime LoRAs (MI_CAP_LORA), applied in order; at most 64. The array is read during mi_generate() only. Default
   * none. */
  const mi_lora* loras;
  size_t loras_count;
  int32_t include_prompts;          /* the metadata includes the prompt and negative prompt; default 1 (168 bytes: 0) */
  int32_t _reserved2;               /* must be 0 */
  /* Fields after the 240-byte layout. Image input (MI_CAP_IMAGE_INPUT): img2img with an init image, inpainting with a
   * mask as well. Each image comes as pixels or as a PNG / JPEG file, at most one of the two (both: invalid). Files are
   * decoded in the library by a decoder that is not memory-safe: decode untrusted images yourself and pass pixels. */
  const mi_image* init_image;       /* RGB (3 channels); NULL = none */
  const char* init_image_path;      /* UTF-8; NULL or "" = none */
  const mi_image* mask;             /* 1 channel, the init image's size; needs an init image. NULL = none */
  const char* mask_path;            /* read as grayscale; NULL or "" = none */
  mi_mask_encoding mask_encoding;   /* default MI_MASK_U8; applies to both mask and mask_path */
  mi_resize_mode resize;            /* default MI_RESIZE_CROP */
  /* [0, 1]: the share of the schedule run from the init image (0: its VAE round trip; 1 without a mask: txt2img, the
   * image is not used, MI_TEXT_IMAGE_NOTE says so), or negative = unset (default). Below 1 needs an init image. */
  double denoise_strength;
  /* Hires fix (MI_CAP_HIRES): a second pass at hires_width x hires_height (multiples of 16, <= 4096, <= 4.2 MP) from
   * the first pass upscaled; 0 x 0 = off (default). */
  int32_t hires_width;
  int32_t hires_height;
  double hires_strength;            /* [0, 1]; default 0.45 */
  int32_t hires_steps;              /* 0..10000; 0 (default) = `steps` */
  int32_t _reserved3;               /* must be 0 */
  /* A distill bundle directory (MI_CAP_DISTILL), expanded like `moss-infer generate --distill`: its lora.safetensors
   * appended to `loras` at strength 1; its dataspace.safetensors as sampler DATASPACE + dataspace_path when the
   * sampler is unset (EULER with no dataspace_path); cfg 1 when cfg is unset (< 0); steps = its step count when steps
   * is unset (0). A bundle with a time_embedding.safetensors needs that file on the engine (mi_engine_params
   * distill_dir or time_embedding_file), else mi_generate() fails with MI_ERR_INVALID_ARGUMENT. NULL or "" = none. */
  const char* distill_dir;
  /* Fields after the 320-byte layout (1.0.0). A control adapter (MI_CAP_CONTROL): a file pairing a LoRA with a widened
   * DiT input embedder (e.g. the MOSS-Image color/depth control), and its conditioning images. Its LoRA part joins the
   * runtime LoRAs after `loras`. Default none. */
  const char* control_adapter;      /* UTF-8; NULL or "" = none. The file must exist; mi_generate() reads its format
                                     * (MI_ERR_MODEL when it is not a control adapter) */
  double control_lora_strength;     /* the adapter's LoRA part (the embedder is never scaled); finite; default 1.0 */
  /* At most 16, one per kind; needs control_adapter. mi_generate() checks each kind against the adapter's
   * (MI_ERR_INVALID_ARGUMENT); a kind without an image gets no control. The array and its images are read during
   * mi_gen_params_validate() / mi_generate() only. Default none. */
  const mi_control_image* control_images;
  size_t control_images_count;
  /* Lossy accelerators (MI_CAP_ACCEL), off by default: an explicit opt-in that changes the image (the tier is resolved
   * as without them). They apply to each sampling pass (a hires fix's too).
   * CFG truncation: classifier-free guidance only on the steps i of a pass with i / steps < cfg_truncation; the later
   * steps run the conditional row alone. [0, 1]; default 1.0 = off. No effect without CFG (cfg 1). */
  double cfg_truncation;
  mi_step_cache step_cache;         /* default MI_STEP_CACHE_OFF */
  int32_t _reserved4;               /* must be 0 */
  /* The step cache's threshold: 0 (default) = unset, the engine's default for the loaded model (0.02 on the base
   * model, 0.03 on the SFT model, the files derived from it and a model it cannot place; mi_accel_info and
   * MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE report what ran), else (0, 1]; larger is faster and further from the uncached
   * image. Ignored with the step cache off. */
  double step_cache_threshold;
  /* Fields after the 376-byte layout (2.1). Reference images (MI_CAP_REFERENCE_IMAGES) for an edit adapter loaded in
   * `loras` (a moss-edit-lora file, e.g. Moss-Image-Edit): in order, the source image first; at most 4, each fit
   * within reference_max_edge (each side rounded to 16, never upscaled beyond that rounding) and VAE-encoded once.
   * Every row (cond and uncond) then attends to the reference tokens, at most 16384 in all; skipped layers
   * (uncond_skip_layers) skip the rows only. Refused with slg_scale > 0 (MI_ERR_INVALID_ARGUMENT). The array and its
   * images are read during mi_gen_params_validate() / mi_generate() only. Default none. */
  const mi_reference_image* reference_images;
  size_t reference_images_count;
  int32_t reference_max_edge;       /* 0 (default) = 512; else 16..2048, floored to a multiple of 16 */
  int32_t _reserved5;               /* must be 0 */
  mi_lora_placement lora_placement; /* default MI_LORA_PLACEMENT_DEFAULT */
  int32_t _reserved6;               /* must be 0 */
} mi_gen_params;

/* Fills the first `struct_size` bytes of `params` with the defaults. struct_size is sizeof(mi_gen_params) of the
 * caller's header: 376, 320 (the 1.0.0 header), 240 (an earlier 0.2.0 header), or 168 for a program built against
 * the 0.1.0 header (whose generations then keep the 0.1.0 behaviour: save_metadata and include_prompts are 0). The
 * fields after an older size keep their defaults (for 320 bytes: no control adapter, accelerators off).
 *
 * Unset markers. At 376 and 320 bytes, init leaves the context-dependent fields unset, and each resolves when the
 * request is checked:
 *   steps 0                  -> the dataspace file's step count with the DATASPACE sampler (a distill bundle's), else
 *                               MI_DEFAULT_STEPS (50). Negative is invalid.
 *   cfg negative (finite)    -> 1.0 with distill_dir, else MI_DEFAULT_CFG (3.0). NaN and infinities are invalid.
 *   denoise_strength negative (finite) -> MI_DEFAULT_DENOISE_STRENGTH (0.75) with an init image, else 1.0. NaN and
 *                               infinities are invalid; a value >= 0 must be in [0, 1].
 *   sampler EULER with no dataspace_path -> a distill bundle's dataspace sampler (EULER cannot be forced with such a
 *                               bundle; any other sampler can).
 * The older sizes keep their init values: steps 50 and cfg 3.0 (168 and 240 bytes; a strength there resolves to 1.0:
 * they cannot carry an init image). */
MOSSINFER_API mi_status mi_gen_params_init(mi_gen_params* params, size_t struct_size);
/* Checks `params` the way mi_generate() does, without an engine or GPU: the existence of the LoRA, dataspace and
 * control adapter files, the image inputs (control images included) and the distill bundle (opened, its dataspace
 * file read) included. Image inputs are checked (sizes, channels, file existence and size) but neither copied nor
 * decoded. Left to mi_generate(): the match of a distill bundle's time embedding with the engine's, and the control
 * adapter's format and kinds. MI_ERR_INVALID_ARGUMENT names the problem; a malformed distill bundle or dataspace file
 * is MI_ERR_MODEL. It checks a generation's request, which needs a prompt: to check a mi_engine_preload() request
 * without one, validate a copy with a stand-in prompt (e.g. "preload"). */
MOSSINFER_API mi_status mi_gen_params_validate(const mi_gen_params* params);

/* Runs one generation to completion. Returns MI_OK with the image, or MI_CANCELLED after mi_cancel() (the result then
 * has no image but carries the timings so far); in both cases *out_result is set and must be released with
 * mi_result_free(). On any other status *out_result is NULL. The engine stays usable after every status except
 * MI_ERR_CUDA, after which it should be freed. */
MOSSINFER_API mi_status mi_generate(mi_engine* engine, const mi_gen_params* params, mi_result** out_result);

/* Loads what the engine keeps resident for a request like `params`, without generating (MI_CAP_PRELOAD), so that the
 * first mi_generate() starts at once: the arena at the plan's size, the DiT as the plan keeps it (not when the plan
 * parks it for the text encoder: MI_TEXT_PRELOAD_NOTE says so), the VAE decoder, the VAE encoder of a request that
 * encodes an image (an init image below strength 1 or with a mask, control images, or a hires fix), the LoRAs and
 * control adapter it names, and a host-resident text encoder's copy. The request is planned as
 * mi_generate() plans it (size, steps, CFG, LoRAs, control adapter, the prompt's length). `params` NULL = the defaults;
 * a NULL or blank prompt is allowed here only (it plans for a short prompt). Otherwise `params` is checked as for
 * mi_generate(), with the same failures. A following mi_generate() of a request like it reuses everything
 * (mi_memory_stats.dit_loads 0) and gives the same image bits as without the preload.
 * It occupies the engine like a generation: MI_ERR_BUSY while a job runs, and mi_cancel() stops it between models and
 * weight groups (MI_CANCELLED with a result; the engine stays usable and keeps what was loaded before the cancel).
 * params' callbacks run on this thread: the plan once, after planning and before any load; then a progress (step 0,
 * total_steps 0) before each load step, also for a model already resident, in this order: MI_PHASE_TEXT_ENCODER before
 * a host-resident text encoder's copy, MI_PHASE_DECODE before the VAE encoder of a request that encodes an image,
 * MI_PHASE_DIT_LOAD before the DiT, MI_PHASE_DECODE before the VAE decoder. A step the plan skips reports nothing;
 * MI_PHASE_DENOISE never comes, and there is no preview. The progress fraction and eta_seconds cover these loads. A
 * mi_cancel() from a callback stops the preload at once. `out_result` may be NULL; else it receives the preload's
 * result on MI_OK and MI_CANCELLED (release it with mi_result_free()): the plan, the tier, the timings, the memory,
 * mi_result_dit_load() and MI_TEXT_PRELOAD_NOTE; no image, steps_run 0. */
MOSSINFER_API mi_status mi_engine_preload(mi_engine* engine, const mi_gen_params* params, mi_result** out_result);

/* What mi_engine_release() freed and kept. An output struct: struct_size set by the caller. */
typedef struct mi_release_info {
  size_t struct_size;
  double seconds;                /* the whole release, host_copy_seconds included */
  uint64_t vram_released_bytes;  /* the arena plus the device memory held outside it (LoRA side branches, control) */
  uint64_t host_copy_bytes;      /* host RAM the kept copies hold */
  double host_copy_seconds;      /* reading the DiT's files into RAM */
  int32_t dit_host_copy;         /* the DiT's weights are kept in RAM: its next load uploads them from there */
  int32_t te_host_copy;          /* a host-resident text encoder's copy is kept */
  char note[512];                /* why a copy was not kept (the host budget, a cancel), cut at 511 bytes; "" if none */
} mi_release_info;

/* Soft unload (MI_CAP_RELEASE): frees the engine's device memory (the arena, every resident model, LoRA side branches,
 * a control adapter's embedder) so that another application can use the GPU, and keeps the engine: the next
 * mi_generate() plans and loads again. With keep_host_copy nonzero the weights stay in host RAM where the host has the
 * room (a margin of the available memory stays free; within host_ram_budget_bytes when set): the DiT's weight files are
 * read into pinned memory (seconds; its next load uploads them from there, bit-identical, then frees them) and a
 * host-resident text encoder keeps its copy. With 0 every host copy is freed too. The embedding cache stays either way.
 * mi_cancel() from another thread, during this call (one sent before the call has no effect: like every job, the
 * release clears the cancel flag when it marks the engine busy):
 * - landing after the call started and before it frees anything: nothing is released, MI_CANCELLED, `out` with
 *   vram_released_bytes 0, the host copies already held and note "cancelled before it started";
 * - during the DiT's host copy (between 64 MiB reads): the copy stops; the release still completes (MI_OK), keeps a
 *   host-resident text encoder's copy, frees nothing else, and the note says the copy was cancelled.
 * `out` (may be NULL) receives what was released and kept, and why not. Returns MI_OK or MI_CANCELLED (above), else:
 * MI_ERR_INVALID_ARGUMENT for a NULL engine, MI_ERR_ABI_MISMATCH for an unknown out->struct_size, MI_ERR_BUSY while a
 * generation, a preload or another release runs (all three before anything is freed), and MI_ERR_CUDA when the device
 * fails (free the engine then). */
MOSSINFER_API mi_status mi_engine_release(mi_engine* engine, int32_t keep_host_copy, mi_release_info* out);

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
  /* Fields after the 104-byte layout: image input and the hires fix (not part of load_s). */
  double vae_encoder_load_s;
  double image_encode_s;  /* every VAE encode (the init image, the hires fix's upscaled first pass) */
  double image_ops_s;     /* host-side image work: resizes, the hires upscale, mask preparation, inpainting blends */
  /* Fields after the 128-byte layout (1.0.0): runtime LoRAs and a control adapter's LoRA part; 0 when reused as is. */
  double lora_load_s;     /* reading the adapter files and uploading the side branches */
  double lora_merge_s;    /* merging adapters into a loaded DiT in place (a strength change, a control switched) */
} mi_timings;

typedef struct mi_memory_stats {
  size_t struct_size;
  uint64_t arena_capacity_bytes;
  uint64_t arena_high_water_bytes;  /* peak arena use during the generation */
  uint64_t device_peak_bytes;       /* peak device memory used by the engine (free at creation - free now) */
  uint64_t device_free_at_start_bytes;
  uint64_t host_private_start_bytes;
  uint64_t host_private_peak_bytes;
  /* Fields after the 56-byte layout (1.0.0), at the end of the call. */
  uint64_t host_copy_bytes;   /* host copies kept across calls (mi_engine_release(), a host-resident text encoder) */
  uint64_t merge_memo_bytes;  /* host RAM the DiT keeps to switch a merged adapter set back on (at most 1 GiB) */
  int32_t dit_loads;          /* DiT loads in the call: 0 when it was resident (or not needed) */
  int32_t _reserved0;
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

/* Where the call's last DiT load read the weights from (MI_CAP_RELEASE). */
typedef enum mi_dit_load_source {
  MI_DIT_LOAD_SOURCE_NONE = 0,      /* no DiT load in the call (it was resident, or not needed) */
  MI_DIT_LOAD_SOURCE_FILES = 1,     /* the weight files (page cache or disk) */
  MI_DIT_LOAD_SOURCE_HOST_COPY = 2, /* the host copy mi_engine_release() kept */
  MI_DIT_LOAD_SOURCE_MAX_ENUM = 0x7fffffff
} mi_dit_load_source;
typedef enum mi_load_io {
  MI_LOAD_IO_NONE = 0,     /* not read from the files */
  MI_LOAD_IO_BUFFERED = 1, /* buffered reads (MI_TEXT_DIT_LOAD_NOTE says why) */
  MI_LOAD_IO_DIRECT = 2,   /* direct I/O */
  MI_LOAD_IO_MAX_ENUM = 0x7fffffff
} mi_load_io;
typedef struct mi_dit_load_info {
  size_t struct_size;
  mi_dit_load_source source;
  mi_load_io io;             /* MI_LOAD_IO_NONE unless source is MI_DIT_LOAD_SOURCE_FILES */
  uint64_t bytes;            /* read by the load */
  double seconds;
} mi_dit_load_info;
/* `out` has struct_size set by the caller. All zero when the call loaded no DiT. */
MOSSINFER_API mi_status mi_result_dit_load(const mi_result* result, mi_dit_load_info* out);
/* The DiT block path the denoise actually ran (never MI_BLOCK_PATH_DEFAULT, except for a NULL result). It can differ
 * from mi_engine_params.block_path; mi_result_block_path_note() then says why. When a cancel landed before the DiT
 * was ready for the denoise (text encoder or DiT load), no block ran and this is MI_BLOCK_PATH_FUSED, the engine
 * default, whatever was requested. */
MOSSINFER_API mi_block_path mi_result_block_path(const mi_result* result);
/* Why the block path differs from the requested one; "" when it does not. Owned by the result. */
MOSSINFER_API const char* mi_result_block_path_note(const mi_result* result);
/* The execution plan in several human-readable lines; owned by the result. */
MOSSINFER_API const char* mi_result_plan_explanation(const mi_result* result);
/* The generation report as one JSON document, the same shape as `moss-infer generate --json-report` ("image" is the
 * last path given to mi_save_png(), "" before). Owned by the result, valid until the next call of this function or
 * mi_save_png() on it. NULL (with mi_last_error_message() set) for a NULL result or when rendering fails (out of
 * memory). */
MOSSINFER_API const char* mi_result_report_json(mi_result* result);
/* Writes the image as a PNG (atomically: a temporary file renamed over `path`). When mi_gen_params.save_metadata was
 * set, the PNG carries the generation settings as text chunks (without the prompts when include_prompts was 0).
 * MI_ERR_INVALID_ARGUMENT when the result has no image (cancelled). */
MOSSINFER_API mi_status mi_save_png(mi_result* result, const char* path);

/* The tier that ran (MI_CAP_TIERS). A generation cancelled before its tier was resolved reports the engine's tier with
 * attention EXACT and dit_precision BF16. */
typedef struct mi_tier_info {
  size_t struct_size;
  mi_tier requested;                   /* mi_engine_params.tier */
  mi_tier effective;                   /* the same: the requested tier runs, or the generation is refused */
  mi_attention attention;              /* what ran; never MI_ATTENTION_DEFAULT */
  int32_t certified;                   /* reserved since 2.0: 0 */
  mi_precision dit_precision;          /* BF16, INT8, W4A4 or NVFP4 */
  mi_precision text_encoder_precision; /* BF16, INT8 or W4A8 */
  /* Fields after the 32-byte layout (1.0.0). */
  int32_t run_certified;               /* reserved since 2.0: 0 */
  int32_t _reserved0;
} mi_tier_info;
/* `out` has struct_size set by the caller: the current size or the 32-byte one. */
MOSSINFER_API mi_status mi_result_tier(const mi_result* result, mi_tier_info* out);

typedef enum mi_result_text_kind {
  MI_TEXT_TIER_REASON = 0,         /* what the tier runs, e.g. "High tier: INT8 ConvRot GEMMs, q8_fp8 attention" */
  MI_TEXT_CERTIFICATE = 1,         /* reserved since 2.0: "" */
  MI_TEXT_GPU_ARCH = 2,            /* the GPU class, e.g. "sm_89" */
  MI_TEXT_DIT_SOURCE = 3,          /* the bundle's transformer folder, or dit_file */
  MI_TEXT_TEXT_ENCODER_SOURCE = 4, /* the bundle's text_encoder folder, or text_encoder_file */
  MI_TEXT_VAE_SOURCE = 5,          /* the bundle's vae folder, or vae_file */
  MI_TEXT_QUALITY_JSON = 6,        /* the report's "quality" object, as JSON */
  MI_TEXT_IMAGE_NOTE = 7,          /* a remark on the image input, e.g. why an init image was not used; "" for none */
  MI_TEXT_TIME_EMBEDDING = 8,      /* the replacement step embedder that ran (time_embedding_file); "" for none */
  /* MI_CAP_CONTROL; all "" without a control adapter. */
  MI_TEXT_CONTROL_ADAPTER = 9,     /* the control adapter file */
  MI_TEXT_CONTROL_KINDS = 10,      /* its kinds in channel order, comma-separated, e.g. "color,depth" */
  MI_TEXT_CONTROL_BASE_MODEL = 11, /* the file's base_model metadata (informational); "" when absent */
  MI_TEXT_CONTROL_ENCODE_SCHEDULE = 12, /* how the control images were VAE-encoded: "untiled" or the tile plans */
  /* MI_CAP_ACCEL: the requested lossy accelerators that are active for this request (not what ran: a run cancelled
   * before sampling reports them too), e.g. "cfg_truncation 0.8, step_cache fbcache 0.03" (an unset threshold as the
   * model's default that ran, or "default" when the run stopped before it was resolved); "off" when none. */
  MI_TEXT_ACCELERATORS = 13,
  MI_TEXT_DIT_LOAD_NOTE = 14,      /* why the DiT's load read buffered, or what decided; "" for a plain direct load */
  MI_TEXT_TE_NOTE = 15,            /* a remark on the text encoder, e.g. a pinned copy that could not be allocated */
  MI_TEXT_PRELOAD_NOTE = 16,       /* mi_engine_preload(): what it did not load and why; "" otherwise */
  /* MI_CAP_ACCEL: where the step cache's threshold came from: "request", or "default for the base model" (sft,
   * unrecognized); "" with the step cache off or before it was resolved (a run cancelled early). */
  MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE = 17,
  /* MI_CAP_COMPONENTS, a result of mi_run_end(): how the run's sigmas differ from the engine's schedule for its
   * scheduler fields (the first differing index, both values), the calls whose sigma was off the schedule, and which
   * rows had foreign conditioning; "" when none applies (and for every other result). Facts only. */
  MI_TEXT_COMPONENT_NOTE = 18,
  /* MI_CAP_REFERENCE_IMAGES: the report's "references" object as JSON (count, max_edge, sizes, tokens, cache_bytes,
   * prefills, encode_schedule, warnings); "" when no reference image ran. */
  MI_TEXT_REFERENCES = 19,
  MI_TEXT_MAX_ENUM = 0x7fffffff
} mi_result_text_kind;
/* Owned by the result; "" when empty (e.g. cancelled early). NULL for a NULL result or an unknown kind. */
MOSSINFER_API const char* mi_result_text(const mi_result* result, mi_result_text_kind kind);

/* The control adapter of a generation (MI_CAP_CONTROL); all zero without one. */
typedef struct mi_control_info {
  size_t struct_size;
  int32_t active;             /* a control adapter was applied (0 also when cancelled before it was read) */
  int32_t kinds_count;        /* the adapter's kinds (MI_TEXT_CONTROL_KINDS) */
  int32_t encodes;            /* control image VAE encodes (one per image and pass size) */
  /* How the adapter's LoRA part was applied, as in mi_lora_report (it is not one of mi_result_lora()'s files). */
  int32_t lora_adapters;
  int32_t lora_side_branch;
  int32_t lora_merged;
  int32_t lora_unknown_keys;
  int32_t lora_warnings;      /* MI_MESSAGE_CONTROL_WARNING */
  int32_t lora_notes;         /* MI_MESSAGE_CONTROL_NOTE */
  int32_t _reserved0;
} mi_control_info;
/* `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_result_control(const mi_result* result, mi_control_info* out);

/* What the lossy accelerators did (MI_CAP_ACCEL), over both passes of a hires fix; all zero with them off. */
typedef struct mi_accel_info {
  size_t struct_size;
  int32_t cfg_truncated_steps; /* steps run without the negative branch (CFG truncation) */
  int32_t dit_calls;           /* DiT calls with the step cache on */
  int32_t cached_calls;        /* of them, calls that reused the cache (the blocks after block 0 skipped) */
  /* The request's settings activate a lossy accelerator (MI_TEXT_ACCELERATORS). It describes the requested, active
   * settings, not what ran: a run cancelled before sampling still reports 1. */
  int32_t lossy;
  /* Fields after the 24-byte layout (a pre-release header). The threshold the step cache ran with: the request's,
   * or the model's default (MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE says which); 0 with the step cache off. */
  double step_cache_threshold;
} mi_accel_info;
/* `out` has struct_size set by the caller: the current size or the 24-byte one. */
MOSSINFER_API mi_status mi_result_accel(const mi_result* result, mi_accel_info* out);
/* The 1-based steps (counting on across the passes of a hires fix) with a DiT call that reused the step cache, in
 * order; owned by the result. `count` (required) receives their number. NULL with *count 0 when there are none, and
 * for a NULL result or count (mi_last_error_message() then set). */
MOSSINFER_API const int32_t* mi_result_accel_cached_steps(const mi_result* result, size_t* count);
/* The relative change of DiT block 0's output that the step cache measured at every DiT call with the cache on (both
 * passes of a hires fix), in order: what the threshold was compared against; NaN where a call had nothing to compare
 * with. Owned by the result; `count` (required) receives their number. NULL with *count 0 when there are none, and for a
 * NULL result or count (mi_last_error_message() then set). */
MOSSINFER_API const double* mi_result_accel_step_cache_diffs(const mi_result* result, size_t* count);

/* What the image input ran (MI_CAP_IMAGE_INPUT, MI_CAP_HIRES). */
typedef enum mi_image_mode {
  MI_IMAGE_MODE_TXT2IMG = 0, /* also an init image at denoise_strength 1 without a mask (MI_TEXT_IMAGE_NOTE says so) */
  MI_IMAGE_MODE_IMG2IMG = 1,
  MI_IMAGE_MODE_INPAINT = 2,
  MI_IMAGE_MODE_MAX_ENUM = 0x7fffffff
} mi_image_mode;

/* The sampling passes of a generation: the request's pass, and the hires fix's second pass. A pass's steps:
 * *_steps_scheduled = the pass's schedule after snapping minus *_start_step (the steps its strength skips). */
typedef struct mi_image_info {
  size_t struct_size;
  mi_image_mode mode;
  int32_t hires_ran;              /* the hires pass was reached (not cancelled before it) */
  int32_t first_width;            /* the request's size */
  int32_t first_height;
  int32_t first_steps_scheduled;
  int32_t first_start_step;       /* 0 for txt2img */
  int32_t first_steps_run;
  int32_t first_encoded;          /* an image was VAE-encoded for the first pass (img2img start or inpainting blend) */
  int32_t hires_width;            /* 0 without the hires fix */
  int32_t hires_height;
  int32_t hires_steps_scheduled;
  int32_t hires_start_step;
  int32_t hires_steps_run;
  int32_t _reserved0;
  double first_strength;          /* the denoise strength of each pass (1 for txt2img) */
  double hires_strength;
} mi_image_info;
/* `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_result_image(const mi_result* result, mi_image_info* out);

/* Token counts of one prompt as the text encoder saw it (chat template included). */
typedef struct mi_prompt_tokens {
  size_t struct_size;
  int32_t total_tokens;    /* the formatted sequence before truncation */
  int32_t kept_tokens;     /* min(total_tokens, max_length): what the text encoder received */
  int32_t prompt_tokens;   /* tokens of the prompt text itself (total_tokens minus the template's) */
  int32_t dropped_tokens;  /* total_tokens - kept_tokens */
  int32_t max_length;      /* mi_gen_params.max_sequence_length */
  int32_t truncated;       /* total_tokens > max_length: the end of the prompt was cut */
} mi_prompt_tokens;
/* negative = 0: the prompt; 1: the negative prompt (all zero when CFG is off). `out` has struct_size set by the
 * caller. */
MOSSINFER_API mi_status mi_result_prompt_tokens(const mi_result* result, int32_t negative, mi_prompt_tokens* out);

/* How one LoRA file was read and applied (MI_CAP_LORA), in mi_gen_params.loras order. */
typedef struct mi_lora_report {
  size_t struct_size;
  int32_t adapters;      /* adapters applied (a fused qkv adapter counts as three) */
  int32_t side_branch;   /* of them run as side branches (a strength change is instant) */
  int32_t merged;        /* of them merged into the DiT at load */
  int32_t unknown_keys;  /* tensors not applied */
  int32_t warnings;      /* adapters applied with a loss (e.g. requantized on INT8 GEMMs) */
  int32_t notes;         /* remarks on how the file was read (e.g. the metadata alpha applied) */
} mi_lora_report;
MOSSINFER_API int32_t mi_result_lora_count(const mi_result* result);
/* `index` in [0, mi_result_lora_count()); `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_result_lora(const mi_result* result, int32_t index, mi_lora_report* out);
/* The file of LoRA `index`; owned by the result. NULL for a NULL result or an index out of range. */
MOSSINFER_API const char* mi_result_lora_file(const mi_result* result, int32_t index);

typedef enum mi_message_kind {
  MI_MESSAGE_LORA_WARNING = 0, /* losses to show the user, "<module>: <reason>" */
  MI_MESSAGE_LORA_NOTE = 1,    /* how the files were read */
  /* MI_CAP_CONTROL: the same for the control adapter's LoRA part. */
  MI_MESSAGE_CONTROL_WARNING = 2,
  MI_MESSAGE_CONTROL_NOTE = 3,
  /* MI_CAP_ACCEL: what the user should know about the accelerators for this model, e.g. a step-cache threshold that
   * moves the base model's composition, or a cache that never engaged. */
  MI_MESSAGE_ACCEL_WARNING = 4,
  /* MI_CAP_REFERENCE_IMAGES: reference images without an edit adapter, or an edit adapter without them. */
  MI_MESSAGE_REFERENCE_WARNING = 5,
  MI_MESSAGE_MAX_ENUM = 0x7fffffff
} mi_message_kind;
/* 0 for a NULL result or an unknown kind. */
MOSSINFER_API int32_t mi_result_message_count(const mi_result* result, mi_message_kind kind);
/* Owned by the result. NULL for a NULL result, an unknown kind or an index out of range. */
MOSSINFER_API const char* mi_result_message(const mi_result* result, mi_message_kind kind, int32_t index);

/* Writes RGB8 [height][width][3] as a PNG without metadata, atomically (a temporary file renamed over `path`); meant
 * for previews. MI_ERR_INVALID_ARGUMENT for a NULL or empty path, NULL `rgb` or a side outside [1, 16384]; MI_ERR_IO when
 * the file cannot be written. */
MOSSINFER_API mi_status mi_write_png(const char* path, const uint8_t* rgb, int32_t width, int32_t height);
/* NULL is a no-op. */
MOSSINFER_API void mi_result_free(mi_result* result);

/* ------------------------------------------------------------------------------------------------ file inspection */

/* The role a file has for the engine (MI_CAP_INSPECT). */
typedef enum mi_file_kind {
  MI_FILE_KIND_UNKNOWN = 0,         /* none of the roles below; MI_FILE_TEXT_NOTE says what was found */
  MI_FILE_KIND_DIT = 1,             /* a single-file DiT (mi_engine_params.dit_file) */
  MI_FILE_KIND_TEXT_ENCODER = 2,    /* mi_engine_params.text_encoder_file */
  MI_FILE_KIND_VAE = 3,             /* mi_engine_params.vae_file */
  /* A LoRA (mi_gen_params.loras): adapters on the MOSS-Image DiT, or tensors in the LoRA naming conventions that the
   * LoRA reader refuses (a LoRA for another model, a malformed one: loadable 0). */
  MI_FILE_KIND_LORA = 4,
  MI_FILE_KIND_CONTROL_ADAPTER = 5, /* mi_gen_params.control_adapter */
  MI_FILE_KIND_MAX_ENUM = 0x7fffffff
} mi_file_kind;

typedef struct mi_file_report mi_file_report;

/* What mi_inspect_file() found. An output struct: struct_size set by the caller. */
typedef struct mi_file_info {
  size_t struct_size;
  mi_file_kind kind;
  /* The engine's loader for that role accepts the file. What depends on the request or the GPU is checked later (a
   * control image's kind, a LoRA merge on INT4 weights). 0 for MI_FILE_KIND_UNKNOWN;
   * MI_FILE_TEXT_NOTE says why not. */
  int32_t loadable;
  uint64_t file_bytes;
  int32_t tensor_count;     /* tensors in the safetensors (or GGUF) header; 0 when the header cannot be read */
  /* A loadable DiT: BF16, INT8, W4A4 or NVFP4, as mi_tier_info.dit_precision; a loadable text encoder: BF16, INT8 or W4A8;
   * else BF16. */
  mi_precision precision;
  /* A loadable DiT: bit (1u << tier) for each mi_tier mi_generate() accepts for it, with the bundle's bf16 text encoder
   * (a quantized one refuses MI_TIER_REFERENCE too) and no explicit attention: the same decision generate makes, on
   * every GPU the engine supports (an NVFP4 DiT is judged for an RTX 50, the only GPU that loads it). Bit 0
   * (MI_TIER_AUTO) is never set. 0 otherwise. */
  uint32_t tiers;
  int32_t lora_rank;            /* LoRA, control adapter: the largest rank of its low-rank adapters; 0 without */
  /* adapters on the DiT, as mi_lora_report.adapters (a fused qkv adapter counts as three) */
  int32_t lora_adapters;
  int32_t lora_unknown_keys;    /* tensors not applied, as mi_lora_report.unknown_keys */
  int32_t control_kind_count;   /* control adapter: its kinds (mi_file_report_control_kind()) */
  int32_t _reserved0;
} mi_file_info;

typedef enum mi_file_text_kind {
  MI_FILE_TEXT_NOTE = 0,          /* why the file is not loadable or what it is instead, or a remark; "" for none */
  /* A loadable DiT: its block linears as the engine runs them, e.g. "bf16", "int8_convrot256", "int4_convrot256",
   * "mixed (80 bf16, 172 int8_convrot256)"; a GGUF DiT: its types and the format it is converted to and runs as, e.g.
   * "GGUF Q4_K + Q6_K + BF16 + F32, runs as INT8 ConvRot"; a loadable text encoder: "bf16", "int8_convrot" or "w4a8"; else "". */
  MI_FILE_TEXT_QUANTIZATION = 1,
  /* LoRA, control adapter: the file's base_model metadata (informational); "" if absent */
  MI_FILE_TEXT_BASE_MODEL = 2,
  /* LoRA, control adapter: its adapter types in file order, comma-separated: "lora", "loha", "lokr", "full". */
  MI_FILE_TEXT_ADAPTER_TYPES = 3,
  MI_FILE_TEXT_MAX_ENUM = 0x7fffffff
} mi_file_text_kind;

/* Reads what the file at `path` (UTF-8) is to the engine (MI_CAP_INSPECT): no engine, no GPU, no CUDA call. It reads
 * what the engine's loaders read before they upload anything: the safetensors header (at most 16 MiB), the
 * quantization configs (at most 4 KiB each), a DiT's scales (bounded by MOSS-Image's shapes, checked first: about
 * 290 MiB at most, one linear's at a time), a text encoder's scales and small F32 tensors (bounded the same way by
 * its shapes: a few MB), a control adapter's embedder (at most 256 MiB, checked from the header first: a file that
 * would need more is not loadable), a LoRA's alpha tensors. Thread-safe. The answer depends only on the file's
 * content and the library's version and build (mi_version() and mi_probe_info.build_flavour: the NVFP4 kernels are a
 * build option), not on the GPU.
 * A file the engine cannot use (another model or container, a malformed or truncated header) is not an error: MI_OK
 * with MI_FILE_KIND_UNKNOWN or loadable 0, and the reason in MI_FILE_TEXT_NOTE. Fails with MI_ERR_INVALID_ARGUMENT for
 * a NULL or empty path or one that is not a regular file (a directory, a device, a pipe: never read), and MI_ERR_IO
 * when it does not exist or cannot be read. On MI_OK *out_report is set: release it with mi_file_report_free(). */
MOSSINFER_API mi_status mi_inspect_file(const char* path, mi_file_report** out_report);
/* `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_file_report_info(const mi_file_report* report, mi_file_info* out);
/* Owned by the report; "" when empty. NULL for a NULL report or an unknown kind. */
MOSSINFER_API const char* mi_file_report_text(const mi_file_report* report, mi_file_text_kind kind);
/* A control adapter's kind `index` in [0, control_kind_count), in channel order: the strings mi_control_image.kind
 * (and MI_TEXT_CONTROL_KINDS) use. Owned by the report. NULL for a NULL report or an index out of range. */
MOSSINFER_API const char* mi_file_report_control_kind(const mi_file_report* report, int32_t index);
/* The engine's tier decision for a loadable DiT: what `tier` would run (e.g. "High tier: INT8 ConvRot GEMMs, 8-bit
 * attention (q8_fp8; q8_fp16 on GPUs without FP8)", with "(RTX 50 only)" for an NVFP4 file) when mi_file_info.tiers has
 * its bit, else why the engine refuses it (MI_TIER_AUTO: reserved); for other files, why there is no tier. Owned by the
 * report. NULL for a NULL report or an unknown tier. */
MOSSINFER_API const char* mi_file_report_tier_reason(const mi_file_report* report, mi_tier tier);
/* NULL is a no-op. */
MOSSINFER_API void mi_file_report_free(mi_file_report* report);

/* ------------------------------------------------------------------------------------------------ components */

/* The component API (MI_CAP_COMPONENTS, 2.1): mi_generate() split into parts, so a host owns the sampling loop. Open a
 * session, encode texts, begin a run, call mi_denoise_step() once per model evaluation, end the run, decode. A session
 * and its run are state of the engine (at most one session per engine, one run per session): the calls take the
 * mi_engine*, and mi_engine_free() closes the session. Conditionings are the only new handles.
 *
 * Every session call is a job of the engine (one at a time, MI_ERR_BUSY otherwise; mi_cancel() stops it). Between
 * runs mi_generate(), mi_engine_preload() and mi_engine_release() work; while a run is open they (and mi_encode_text,
 * mi_vae_*, a second mi_run_begin) fail with MI_ERR_BUSY. mi_run_end(), mi_session_close() and mi_engine_free() never
 * fail for a call in flight on another thread: they cancel it (a step, or any session call for a close; mi_run_end and
 * mi_session_close wait for a generation without cancelling it, mi_engine_free cancels whatever runs), wait for it to
 * return, then end. From the moment an ending starts, new calls on what it ends fail with MI_ERR_INVALID_ARGUMENT ("the
 * run is ending" / "the session is closing"). From a callback of the engine's running job (any job), mi_run_end() and
 * mi_session_close() return MI_ERR_BUSY and do nothing, even with nothing open; mi_engine_free() stays forbidden
 * there.
 *
 * Runs use the engine's tier and attention (mi_engine_params; explicit since 2.0) and are planned as mi_generate()
 * plans the equivalent request, within the engine's mode and budgets. Every image-affecting parameter is explicit; a
 * default is always a fixed constant.
 *
 * Conventions (FLOW). x = (1 - sigma) x0 + sigma noise; each output is the DiT's raw output (the velocity), so
 * denoised = x - sigma * out; the DiT's timestep is fp32(sigma * 1000). No shift or multiplier is applied to a call's
 * sigma: they enter only the engine's own schedule (mi_schedule(), or mi_run_params.sigmas NULL).
 *
 * Latents: F32 [MI_LATENT_CHANNELS, height / MI_LATENT_SCALE, width / MI_LATENT_SCALE] per image, a batch of L as
 * [L, 128, h, w]. Model space (what the DiT denoises): the FLUX.2 VAE's 32 channels patchified 2 x 2 into channel
 * c * 4 + 2 * dy + dx and normalized by the VAE's batch-norm statistics. That is ComfyUI's Flux2 LATENT as is (its VAE
 * patchifies and normalizes inside encode and undoes both in decode; latent_formats.Flux2's process_in / process_out
 * are the identity): no conversion in either direction. 2 MiB per image at 1024^2. Conditioning: [tokens, MI_TEXT_DIM].
 * Images: RGB8 [height][width][3]. */
#define MI_LATENT_CHANNELS 128
#define MI_LATENT_SCALE 16
#define MI_TEXT_DIM 4096
/* The most tokens one call may carry, summed over its rows (h * w + prompt tokens each): the longest run the fused
 * block path's startup self-check validated. mi_run_begin() refuses a run whose rows exceed it. */
#define MI_MAX_CALL_TOKENS 61506
#define MI_MAX_ROWS 8 /* rows per run and per call (the DiT's varlen limit), latents included */

typedef enum mi_memory {
  MI_MEMORY_HOST = 0,   /* host memory; in mi_denoise_step(), only the run's staging (mi_run_staging()) */
  MI_MEMORY_DEVICE = 1, /* memory of the engine's device_index in its primary context (as torch and CuPy allocate) */
  MI_MEMORY_MAX_ENUM = 0x7fffffff
} mi_memory;

typedef enum mi_dtype { MI_DTYPE_F32 = 0, MI_DTYPE_BF16 = 1, MI_DTYPE_U8 = 2, MI_DTYPE_MAX_ENUM = 0x7fffffff } mi_dtype;

/* A dense row-major tensor (no strides); data aligned to its element size (4 bytes for F32: a misaligned pointer is
 * refused). A fixed 56-byte layout without struct_size. */
typedef struct mi_tensor {
  void* data;
  mi_memory memory;
  mi_dtype dtype;
  int32_t rank;       /* 1..4 */
  int32_t _reserved0; /* must be 0 */
  int64_t shape[4];   /* shape[0..rank) > 0, the rest 0 */
} mi_tensor;

/* The text encoder's last hidden state of one prompt (the tensor before the DiT's text refiner), bf16 on the host.
 * Immutable and independent of any engine: it may outlive engines, serve several runs and engines, and be read by
 * threads at once. */
typedef struct mi_conditioning mi_conditioning;

/* ---- session */

typedef struct mi_session_params {
  size_t struct_size;
  /* The request planned for before the first run fixes its shape. Copied; prompt may be NULL or blank (as
   * mi_engine_preload()). NULL (default) = the defaults. Preload with mi_engine_preload(engine, same request). */
  const mi_gen_params* plan_request;
  /* As mi_gen_params': MI_PHASE_TEXT_ENCODER (mi_encode_text), MI_PHASE_DIT_LOAD (mi_run_begin), MI_PHASE_DECODE
   * (mi_vae_*, control-image encodes). Never MI_PHASE_DENOISE. mi_generate()'s callback rules apply. */
  mi_progress_fn progress_callback;
  void* user_data;
  /* Fixed for the session. They bound the session's runs and size only the pinned host staging; each run is planned
   * for its own rows. The most rows of a run or call, 1..MI_MAX_ROWS (default 3: cond, uncond, SLG); */
  int32_t max_rows;
  /* the most latents of a run's latent batch, 1..max_rows (default 1); */
  int32_t max_latents;
  /* the control adapter (UTF-8 path; NULL or "" = none, default). With one, every call of the session runs the widened
   * input embedder, and its control latent is zeros wherever none is given (zeros = no control, as mi_generate's). */
  const char* control_adapter;
} mi_session_params;

MOSSINFER_API mi_status mi_session_params_init(mi_session_params* params, size_t struct_size);
/* Opens the engine's session. MI_ERR_BUSY when a job runs or a session is open. */
MOSSINFER_API mi_status mi_session_open(mi_engine* engine, const mi_session_params* params);
/* Ends an open run, frees the staging and events; a session call running on another thread is cancelled and waited
 * for first. MI_ERR_BUSY (nothing done) from a callback of the running call; MI_OK when no session is open. Models
 * stay loaded, as after mi_generate(). */
MOSSINFER_API mi_status mi_session_close(mi_engine* engine);

/* ---- text */

typedef struct mi_text_params {
  size_t struct_size;
  const char* const* prompts;  /* 1..8 prompts, UTF-8; "" allowed (an empty negative prompt) */
  size_t prompts_count;
  int32_t max_sequence_length; /* 1..4096; default 4096 */
  int32_t _reserved0;
} mi_text_params;

/* An output struct: struct_size set by the caller. */
typedef struct mi_conditioning_info {
  size_t struct_size;
  int32_t tokens;         /* S */
  int32_t text_dim;       /* MI_TEXT_DIM */
  mi_dtype dtype;         /* MI_DTYPE_BF16 */
  int32_t engine_encoded; /* 1: mi_encode_text(); 0: mi_conditioning_create() (foreign) */
} mi_conditioning_info;

MOSSINFER_API mi_status mi_text_params_init(mi_text_params* params, size_t struct_size);
/* Every prompt in one text-encoder pass, through the embedding cache, planned for the actual token counts.
 * out_conditionings[prompts_count] on MI_OK, none otherwise. A job: mi_cancel() stops it within one layer
 * (MI_CANCELLED). */
MOSSINFER_API mi_status mi_encode_text(mi_engine* engine, const mi_text_params* params,
                                       mi_conditioning** out_conditionings);
/* Foreign conditioning, copied: [S, MI_TEXT_DIM], 1 <= S <= 4096, F32 (rounded to bf16, nearest even) or BF16, host
 * or device (complete before the call). No engine. Used exactly like engine-encoded conditioning; the run's note says
 * it was foreign. */
MOSSINFER_API mi_status mi_conditioning_create(const mi_tensor* embeddings, mi_conditioning** out_conditioning);
MOSSINFER_API mi_status mi_conditioning_query(const mi_conditioning* c, mi_conditioning_info* out);
/* The prompt's token report (zeros for foreign conditioning). `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_conditioning_prompt_tokens(const mi_conditioning* c, mi_prompt_tokens* out);
/* Host BF16 view [S, MI_TEXT_DIM], owned by the conditioning. */
MOSSINFER_API mi_status mi_conditioning_tensor(const mi_conditioning* c, mi_tensor* out);
/* NULL is a no-op. mi_run_begin() copies what it needs: a conditioning may be freed while a run that uses it is
 * open. */
MOSSINFER_API void mi_conditioning_free(mi_conditioning* c);

/* ---- runs */

typedef enum mi_guidance {
  /* The host owns guidance (default): each call evaluates the rows it names (mi_step_params.rows), each with its own
   * fixed skip set; outputs are MI_STEP_OUTPUT_ROWS only; cfg, slg_* and cfg_truncation must keep their defaults. */
  MI_GUIDANCE_HOST = 0,
  /* mi_generate's layout: rows[0] cond, rows[1] uncond (optional), whose skip set is the uncond skip set and applies
   * inside the skip window (slg_start..slg_stop) only; the SLG row is rows[0] with slg_layers. The engine picks each
   * step's rows by mi_generate's rule; mi_step_params.rows must be 0. All outputs. */
  MI_GUIDANCE_MOSS = 1,
  MI_GUIDANCE_MAX_ENUM = 0x7fffffff
} mi_guidance;

typedef enum mi_step_output {
  /* Each evaluated row's velocity as the DiT returns it (F32), out[k] for row bit k. MOSS uses fixed role slots:
   * out[0] cond, out[1] uncond, out[2] the SLG row, so with cfg 1 + SLG out[1] is unused (pass it zeroed) and the SLG
   * row stays at out[2] on every step. A row the call does not evaluate is left untouched: read mi_step_info.rows.
   * Default. */
  MI_STEP_OUTPUT_ROWS = 0,
  MI_STEP_OUTPUT_GUIDED_VELOCITY = 1, /* MOSS only. out[0]: mi_generate's guidance (CFG + SLG), velocity space */
  MI_STEP_OUTPUT_GUIDED_DENOISED = 2, /* MOSS only. out[0]: the guidance in denoised space (D = x - v sigma) */
  MI_STEP_OUTPUT_MAX_ENUM = 0x7fffffff
} mi_step_output;

/* One row of a run. A fixed 32-byte layout without struct_size (array element). */
typedef struct mi_run_row {
  const mi_conditioning* conditioning; /* required */
  const int32_t* skip_layers;          /* DiT blocks it skips, unique, in [0, 42); the official uncond row: {22} */
  size_t skip_layers_count;
  int32_t latent;                      /* the row's latent in the call's batch, 0..latents-1 (default 0) */
  int32_t _reserved0;                  /* must be 0 */
} mi_run_row;

/* mi_run_params.flags */
#define MI_RUN_EXACT_SIGMA 0x1u /* refuse a sigma that is not an exact schedule entry (a host's check); default off */
/* This run may make crop calls (area conds: x smaller than the run's size, HOST guidance): its plan reserves the
 * forward workspace's crop bound. mi_run_begin() refuses it (MI_ERR_INVALID_ARGUMENT, naming both counts) when that
 * bound would keep fewer DiT blocks resident than the run without it. Without the flag (default) a crop call that does
 * not fit the run's planned stack is refused, naming the crop. */
#define MI_RUN_CROPS 0x2u

/* One sampling pass. Names, ranges and defaults are mi_gen_params' where the meaning is shared; every default is a
 * fixed constant. */
typedef struct mi_run_params {
  size_t struct_size;
  int32_t width;  /* default 1024 */
  int32_t height; /* default 1024 */
  /* required: 1..session max_rows rows; their latent indices cover 0..L-1 with L <= session max_latents (the run's
   * latent batch). MOSS: 1 or 2 rows on latent 0, L = 1. */
  const mi_run_row* rows;
  size_t rows_count;
  /* The schedule: a position and precompute hint, never a limit on the calls. sigmas_count = steps + 1 values,
   * finite, non-increasing, terminal last, <= 10001; copied. NULL (default) = the engine's: mi_schedule(scheduler,
   * steps, shift, flow_multiplier, denoise_strength). With host sigmas, the scheduler fields name the schedule they are
   * compared with (mi_run_info.schedule_matches, a reported fact). */
  const float* sigmas;
  size_t sigmas_count;
  mi_scheduler scheduler;   /* default MI_SCHEDULER_MOSS */
  int32_t steps;            /* 1..10000; default MI_DEFAULT_STEPS */
  double shift;             /* default 5.0 (schedule creation only; no shift is applied in a step) */
  double flow_multiplier;   /* default 1.0 */
  double denoise_strength;  /* [0, 1], the pass's share of the schedule (img2img); default 1.0 */
  mi_guidance guidance;     /* default MI_GUIDANCE_HOST */
  mi_step_output output;    /* default MI_STEP_OUTPUT_ROWS */
  uint32_t flags;           /* MI_RUN_*; default 0 */
  int32_t _reserved0;       /* must be 0 */
  double cfg;               /* MOSS guided outputs: >= 0, != 1 with an uncond row; default 3.0 */
  const int32_t* slg_layers; /* MOSS only; default none */
  size_t slg_layers_count;
  double slg_scale;         /* MOSS only; default 0 */
  double slg_start;         /* MOSS only: the skip and SLG window; default 0 */
  double slg_stop;          /* default 1 */
  const mi_lora* loras;     /* applied at mi_run_begin(); kept for later runs while unchanged */
  size_t loras_count;
  /* The session's control adapter (mi_session_params.control_adapter): its LoRA part's strength for this run, and the
   * run-level control latents, from control_images (encoded once at mi_run_begin() at width x height, with their
   * windows; latent 0 only) or per call (mi_step_params.control, which overrides them). Without either: zeros. */
  double control_lora_strength; /* default 1.0 */
  const mi_control_image* control_images;
  size_t control_images_count;
  double cfg_truncation;    /* MOSS only; lossy; default 1.0 = off */
  mi_step_cache step_cache; /* lossy; default MI_STEP_CACHE_OFF */
  /* How control_images are brought to width x height (as mi_gen_params.resize; grayscale is replicated to RGB after
   * the resize). Default MI_RESIZE_CROP. */
  mi_resize_mode control_resize;
  /* (0, 1], required with the step cache on (0 is refused then: no model-dependent default); ignored when off. */
  double step_cache_threshold;
} mi_run_params;

/* An output struct: struct_size set by the caller. */
typedef struct mi_run_info {
  size_t struct_size;
  int32_t latent_channels, latent_height, latent_width;
  int32_t steps;              /* sigmas_count - 1 */
  int32_t outputs;            /* output slots per call: 1 (guided), else the highest bit of `rows` + 1 */
  uint32_t rows;              /* bit k: row k exists (MOSS: bit 2 = the SLG row) */
  int32_t schedule_matches;   /* sigmas == the engine's schedule for the scheduler fields, bit for bit */
  int32_t off_schedule_calls; /* calls whose sigma was not an exact schedule entry */
  int32_t calls;              /* completed mi_denoise_step() calls (no upper bound) */
  int32_t control_channels;   /* the per-call control latent's channels (128 per adapter kind); 0 without adapter */
  int32_t latents;            /* the run's latent batch L */
  int32_t _reserved0;
  uint64_t staging_bytes;     /* the size of the run's staging block (mi_run_adopt_staging) */
} mi_run_info;

typedef struct mi_step_params {
  size_t struct_size;
  float sigma; /* the flow sigma of x; timestep = sigma * 1000 */
  /* 0-based step, or -1 (default) = from sigma: the first exact schedule entry from the cursor (the last step used
   * + 1), else the interval s[i] > sigma > s[i + 1]. A corrector call inside a step (heun's, at sigmas[step + 1])
   * gives its step, or the lookup takes the next step's windows. */
  int32_t step;
  uint32_t rows;  /* HOST: bit k evaluates row k; 0 (default) = every row. MOSS: must be 0 */
  uint32_t flags; /* MI_STEP_*; default 0 */
  /* Device tensors: the host's cudaStream_t, any handle (NULL = the legacy default stream, torch's default;
   * cudaStreamLegacy, cudaStreamPerThread). The engine waits for the work enqueued on it before reading x, and makes it
   * wait for the outputs. Ignored for host tensors. */
  void* cuda_stream;
  /* Optional control latent through the adapter's widened input channels (ComfyUI's concat_cond): F32
   * [L, control_channels, h, w] ([control_channels, h, w] for L = 1), the same memory kind as x (host: the staging
   * view). data NULL (default) = the run's control latents (control_images, zeros where a window is closed; zeros
   * without them; always zeros on a crop call). Given, it is used as is for this call: the host owns windows and
   * strengths; zeros = no control. When it differs from the previous call's, set MI_STEP_CONTROL_CHANGED. */
  mi_tensor control;
  /* F32 model-space latents [L, 128, h, w] ([128, h, w] for L = 1), read only. h x w is the run's latent size, or a
   * smaller crop (an area cond; HOST guidance only). */
  mi_tensor x;
  mi_tensor out[MI_MAX_ROWS]; /* F32 [128, h, w] each (x's h x w), out[k] for row k; mi_run_info.outputs of them */
} mi_step_params;

/* mi_step_params.flags */
/* The per-call control latent differs from the previous call's: the step cache is invalidated before this call (the
 * next reuse needs a full call), as mi_generate invalidates it when a control window opens or closes. */
#define MI_STEP_CONTROL_CHANGED 0x1u

/* An output struct: struct_size set by the caller. */
typedef struct mi_step_info {
  size_t struct_size;
  int32_t step;        /* the step position used */
  uint32_t rows;       /* rows evaluated */
  int32_t cached;      /* the step cache reused its residual */
  int32_t on_schedule; /* sigma is sigmas[step], or sigmas[step + 1] with an explicit step (a corrector) */
  double seconds;      /* wall time of the call */
} mi_step_info;

MOSSINFER_API mi_status mi_run_params_init(mi_run_params* params, size_t struct_size);
/* Plans the pass as mi_generate() plans the equivalent request, within the engine's mode and budgets; then, a job like
 * a preload: loads what the plan keeps, applies LoRAs and control, encodes the control images, pins a route snapshot,
 * refines each distinct conditioning once, takes the denoise stack and the step cache. mi_cancel() stops it between
 * models, weight groups and rows (MI_CANCELLED, no run). MI_ERR_OUT_OF_MEMORY when no mode fits; MI_ERR_BUSY when a run
 * is open or a job runs; MI_ERR_INVALID_ARGUMENT without a session, for more rows than the session's max_rows, a run
 * over MI_MAX_CALL_TOKENS, or a latent batch whose plan would stream more DiT bytes per step than its latent-0 rows. */
MOSSINFER_API mi_status mi_run_begin(mi_engine* engine, const mi_run_params* params);
/* The open run's facts. `out` has struct_size set by the caller. Not a job: it does not wait for a step. */
MOSSINFER_API mi_status mi_run_query(const mi_engine* engine, mi_run_info* out);
/* The run's schedule, owned by the engine until mi_run_end(). `count` required. NULL without a run. */
MOSSINFER_API const float* mi_run_sigmas(const mi_engine* engine, size_t* count);
/* Makes caller-owned host memory the run's staging block, e.g. a shared-memory region another process also maps.
 * Call after mi_run_begin() and before the first mi_denoise_step() (refused once one has been called, whatever its
 * status), once per run. `base`: page-aligned (4096); `bytes` >= mi_run_info.staging_bytes. The engine registers the
 * pages the layout uses as pinned (cudaHostRegister, portable; memory the caller registered already is used as it is
 * and stays registered) and lays x, out[] and control out at the same offsets as its own block;
 * mi_run_staging() then returns views into it (offset = view.data - base). The memory must stay mapped until
 * mi_run_end(), which unregisters it. MI_ERR_NOT_SUPPORTED when the registration is refused (e.g. some Windows
 * file-mapping views): the engine's own staging stays in use, and the caller copies into it. Not a job a cancel
 * reaches: MI_ERR_BUSY while a step runs, MI_ERR_INVALID_ARGUMENT ("the run is ending") once an ending began. */
MOSSINFER_API mi_status mi_run_adopt_staging(mi_engine* engine, void* base, size_t bytes);
/* Host views of the run's pinned staging: x, out[0..outputs) (an unused MOSS slot zeroed) and, with an adapter, the
 * control latent. The only host memory mi_denoise_step() takes. Valid until mi_run_end() (or a later
 * mi_run_adopt_staging()). `outs` (outs_count of them, at most outputs written) and `control` may be NULL. */
MOSSINFER_API mi_status mi_run_staging(mi_engine* engine, mi_tensor* x, mi_tensor* outs, size_t outs_count,
                                       mi_tensor* control);
MOSSINFER_API mi_status mi_step_params_init(mi_step_params* params, size_t struct_size);
/* One model evaluation of the call's rows at (x, sigma) in one batched varlen forward (rows of different prompt
 * lengths included); returns when the outputs are written (device mode: also ordered into cuda_stream). Tensors: all
 * device, or all host and exactly the staging views; x must not alias an output; else MI_ERR_INVALID_ARGUMENT (the run
 * usable). A job: MI_ERR_BUSY while one runs. mi_cancel() stops it within one DiT block: MI_CANCELLED, outputs
 * undefined, the run usable (step cache invalidated). On return, whatever the status, the engine no longer reads x or
 * control and no longer writes out[]. `out_info` (struct_size set by the caller) may be NULL. */
MOSSINFER_API mi_status mi_denoise_step(mi_engine* engine, const mi_step_params* params, mi_step_info* out_info);
/* Ends the run, interrupt-safe: cancels a step running on another thread and waits for it (MI_ERR_BUSY, nothing done,
 * from a callback of the running call); also after a cancelled or failed step. Releases the run's stack, step cache
 * and control latents; models, LoRAs and the staging stay. `out_result` (may be NULL): an mi_result without image
 * (plan, the tier that ran with its attention and precisions, timings, memory, dit_stream, accel, control, LoRA
 * reports, MI_TEXT_COMPONENT_NOTE, the report JSON). MI_OK when no run is open (*out_result NULL then). */
MOSSINFER_API mi_status mi_run_end(mi_engine* engine, mi_result** out_result);

/* ---- VAE: synchronous; device inputs complete before the call, device outputs complete on return */

/* RGB8 (sides multiples of 16, the request bounds) -> model-space latents [128, H/16, W/16] F32, host or device (the
 * posterior mean): exactly mi_generate's img2img start and control-image encodes. Planned for the image's size. A
 * job; mi_cancel() stops it between bands. Needs an open session and no open run. */
MOSSINFER_API mi_status mi_vae_encode(mi_engine* engine, const mi_image* image, const mi_tensor* out_latents);
/* Model-space latents [128, h, w] F32 -> RGB8 [16h, 16w, 3] U8 (host or device): mi_generate's final decode bit for
 * bit. Planned for the latents' size. A job; mi_cancel() stops it between bands. Needs an open session and no open
 * run. */
MOSSINFER_API mi_status mi_vae_decode(mi_engine* engine, const mi_tensor* latents, const mi_tensor* out_rgb);

/* ---- pure helpers: no engine, no GPU, thread-safe; each is the pipeline's own function */

/* mi_generate's schedule for one pass: the scheduler's schedule at strength 1, then, below strength 1, cropped by the
 * diffusers get_timesteps rule mi_generate's img2img uses (not k-diffusion's `denoise` tail). Strength 0 gives an empty
 * schedule (mi_run_begin() refuses it: the VAE round trip needs no run). sigmas NULL: *count = the length only;
 * otherwise MI_ERR_INVALID_ARGUMENT when capacity is too small (*count = the length needed). */
MOSSINFER_API mi_status mi_schedule(mi_scheduler scheduler, int32_t steps, double shift, double flow_multiplier,
                                    double denoise_strength, float* sigmas, size_t capacity, size_t* count);
/* The prediction space mi_generate's `sampler` runs on with `scheduler`: GUIDED_VELOCITY for ab2 (any scheduler) and
 * euler on moss; GUIDED_DENOISED for every other pair. MI_STEP_OUTPUT_MAX_ENUM for an unknown sampler or scheduler
 * (and for MI_SAMPLER_DATASPACE, which the component API does not offer). */
MOSSINFER_API mi_step_output mi_sampler_prediction(mi_sampler sampler, mi_scheduler scheduler);
/* mi_generate's initial noise for `seed` (0..2^53-1) and `layout` at width x height into out[128 * h * w]
 * (count >= that many floats). */
MOSSINFER_API mi_status mi_initial_noise(int64_t seed, mi_noise_layout layout, int32_t width, int32_t height,
                                         float* out, size_t count);
/* The pipeline's resize (PIL-exact Lanczos, `mode`) of an RGB or 1-channel image into out[width * height * channels]
 * (out_bytes >= that many bytes). */
MOSSINFER_API mi_status mi_image_resize(const mi_image* image, int32_t width, int32_t height, mi_resize_mode mode,
                                        uint8_t* out, size_t out_bytes);

#ifdef __cplusplus
}
#endif

#endif /* MOSSINFER_MOSSINFER_H */
