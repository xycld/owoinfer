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

#define MI_VERSION_MAJOR 1
#define MI_VERSION_MINOR 0
#define MI_VERSION_PATCH 0
/* Incremented on every incompatible change of an existing function, struct field or enum value. */
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

/* Quality tier (MI_CAP_TIERS). The weights fix the GEMM precision (an INT8 DiT runs INT8 GEMMs); the tier decides the
 * attention kernel. A refused combination fails mi_generate() with MI_ERR_INVALID_ARGUMENT, naming the conflict. */
typedef enum mi_tier {
  /* The fastest certified configuration of the loaded models; else Reference for a bf16 DiT. Refuses an INT4 DiT
   * without a certificate (choose MI_TIER_TURBO explicitly). */
  MI_TIER_AUTO = 0,
  MI_TIER_REFERENCE = 1, /* bf16 DiT and text encoder, exact attention; refused for quantized files */
  MI_TIER_HIGH = 2,      /* the loaded DiT's precision plus 8-bit attention (Q8_FP8 unless `attention` says otherwise) */
  MI_TIER_TURBO = 3,     /* an INT4 DiT file: INT4 GEMMs plus 8-bit attention; refused for any other DiT */
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
  MI_PRECISION_MAX_ENUM = 0x7fffffff
} mi_precision;

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
  mi_tier tier;                    /* default MI_TIER_AUTO */
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
} mi_engine_params;

/* Fills the first `struct_size` bytes of `params` with the defaults. struct_size is sizeof(mi_engine_params) of the
 * caller's header: 136, 120 (an earlier 0.2.0 header) or 80 (0.1.0). */
MOSSINFER_API mi_status mi_engine_params_init(mi_engine_params* params, size_t struct_size);

/* Creates an engine on `params->device_index`. Checks the parameters, the model path and that the component files exist
 * (their format is checked when they load, at the first mi_generate(): MI_ERR_MODEL), loads cuBLASLt (no usable
 * copy: MI_ERR_CUDA, the message lists every candidate tried) and initializes CUDA; the weights load on the first
 * mi_generate() (and stay resident as the plan allows). The strings are copied. The calling thread's current CUDA
 * context is restored before returning (also by mi_generate()). */
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
} mi_progress;

/* Called when a phase starts and after every sampling step. */
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
  /* Runtime LoRAs (MI_CAP_LORA), applied in order; at most 64. The array is read during mi_generate() only. A result
   * with LoRAs is uncertified. Default none. */
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
   * runtime LoRAs after `loras`, and a result with a control adapter is uncertified, like any LoRA. Default none. */
  const char* control_adapter;      /* UTF-8; NULL or "" = none. The file must exist; mi_generate() reads its format
                                     * (MI_ERR_MODEL when it is not a control adapter) */
  double control_lora_strength;     /* the adapter's LoRA part (the embedder is never scaled); finite; default 1.0 */
  /* At most 16, one per kind; needs control_adapter. mi_generate() checks each kind against the adapter's
   * (MI_ERR_INVALID_ARGUMENT); a kind without an image gets no control. The array and its images are read during
   * mi_gen_params_validate() / mi_generate() only. Default none. */
  const mi_control_image* control_images;
  size_t control_images_count;
  /* Lossy accelerators (MI_CAP_ACCEL), off by default. They change the image, and a result with any of them active is
   * uncertified (the tier is resolved as without them). They apply to each sampling pass (a hires fix's too).
   * CFG truncation: classifier-free guidance only on the steps i of a pass with i / steps < cfg_truncation; the later
   * steps run the conditional row alone. [0, 1]; default 1.0 = off. No effect without CFG (cfg 1). */
  double cfg_truncation;
  mi_step_cache step_cache;         /* default MI_STEP_CACHE_OFF */
  int32_t _reserved4;               /* must be 0 */
  /* The step cache's threshold: 0 (default) = the engine's recommended value (0.03), else (0, 1]; larger is faster
   * and further from the uncached image. Ignored with the step cache off. */
  double step_cache_threshold;
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
 * is MI_ERR_MODEL. */
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
  /* Fields after the 104-byte layout: image input and the hires fix (not part of load_s). */
  double vae_encoder_load_s;
  double image_encode_s;  /* every VAE encode (the init image, the hires fix's upscaled first pass) */
  double image_ops_s;     /* host-side image work: resizes, the hires upscale, mask preparation, inpainting blends */
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

/* The tier that ran (MI_CAP_TIERS). A generation cancelled before its tier was resolved reports requested AUTO and
 * effective REFERENCE. `certified` describes the tier: it stays 1 for REFERENCE even when runtime LoRAs, a control
 * adapter or lossy accelerators make the run as a whole uncertified (MI_TEXT_TIER_REASON and the report's
 * models.dit.certified say so; mi_accel_info.lossy, mi_control_info.active). */
typedef struct mi_tier_info {
  size_t struct_size;
  mi_tier requested;                   /* mi_engine_params.tier */
  mi_tier effective;                   /* never MI_TIER_AUTO */
  mi_attention attention;              /* what ran; never MI_ATTENTION_DEFAULT */
  int32_t certified;                   /* a quality certificate covers what ran (always 1 for REFERENCE) */
  mi_precision dit_precision;          /* BF16 or INT8 */
  mi_precision text_encoder_precision; /* BF16, INT8 or W4A8 */
} mi_tier_info;
/* `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_result_tier(const mi_result* result, mi_tier_info* out);

typedef enum mi_result_text_kind {
  MI_TEXT_TIER_REASON = 0,         /* what the tier chose and why */
  MI_TEXT_CERTIFICATE = 1,         /* the certificate file used; "" for none */
  MI_TEXT_GPU_ARCH = 2,            /* the GPU class certificates are matched on, e.g. "sm_89" */
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
  /* MI_CAP_ACCEL: the lossy accelerators that changed the run, e.g. "cfg_truncation 0.8, step_cache fbcache 0.03";
   * "off" when none. */
  MI_TEXT_ACCELERATORS = 13,
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
  int32_t lossy;               /* a lossy accelerator was active, so the run is uncertified (MI_TEXT_ACCELERATORS) */
} mi_accel_info;
/* `out` has struct_size set by the caller. */
MOSSINFER_API mi_status mi_result_accel(const mi_result* result, mi_accel_info* out);
/* The 1-based steps (counting on across the passes of a hires fix) with a DiT call that reused the step cache, in
 * order; owned by the result. `count` (required) receives their number. NULL with *count 0 when there are none, and
 * for a NULL result or count (mi_last_error_message() then set). */
MOSSINFER_API const int32_t* mi_result_accel_cached_steps(const mi_result* result, size_t* count);

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

#ifdef __cplusplus
}
#endif

#endif /* MOSSINFER_MOSSINFER_H */
