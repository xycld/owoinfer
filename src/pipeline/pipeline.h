// End-to-end MOSS-Image generation (MOSSImagePipeline.__call__, P:340-438): tokenize prompt and negative prompt ->
// embedding cache or Qwen3.5 text encoder (layer-streamed, both prompts in one pass) -> text refiner once per CFG row
// -> torch-compatible CPU noise -> sampling loop on the DiT -> FLUX.2 VAE decode (untiled, or exact tiled) -> RGB8.
//
// Memory (P3): every generate() first asks the planner (planner/planner.h) for an ExecutionPlan from the model's
// shapes, the request, the live free device memory, the host RAM, the mode and the models' exact workspace queries.
// A request that cannot fit in any mode fails right there with planner::BudgetError, before any heavy work. The plan
// sizes one DeviceArena per pipeline: persistent region = the DiT (kept across generate() calls) and the VAE decoder;
// stack region = one phase at a time (text encoder, denoise, decode). When the plan keeps the DiT resident the decode
// runs exactly tiled if the untiled one does not fit next to it; the DiT is evicted ("parked": the arena is recreated
// without it, the weights reloaded for the next denoise) only when the plan says so, which it does as a last resort.
// One text encoder (its pinned rings and streams) lives across calls while the arena does.
// P4: when the whole DiT does not fit (or in min_memory), the plan streams DiT blocks: some stay resident, the others
// are copied every step from the weight files (page cache) or a pinned host copy into device block buffers, ahead of
// the compute on copy streams; bit-identical to the resident DiT. The transfer rates the plan's cost model uses are
// measured once (a short probe, cached as <cache_dir>/hardware_profile.json when a cache directory is set) and then
// refined from every streamed denoise (pipeline/hardware_profile.h). A loaded DiT is kept (inside its held arena, with
// its residency) unless the request's own plan saves more time than reloading the DiT costs (planner::prefer_held).
// P5: the DiT blocks run fused (PipelineOptions::block_path) once a startup self-check (once per GPU, driver, cuBLASLt
// version and fused-kernels version, cached with the profile) found the fused GEMMs bit-identical to the op-by-op ones
// here, each in the cuBLASLt configurations it validated (the others run op by op); otherwise op by op, and Result
// says why.
// P6: the DiT can come from a single file (PipelineOptions::dit_file: an INT8 ConvRot or bf16 safetensors in the
// ComfyUI layout) while the tokenizer, configs, text encoder and VAE come from the bundle. The caller's quality tier
// (PipelineOptions::tier, planner::resolve_tier) decides the lossy toggles with what is loaded: an INT8 DiT runs the
// High tier's INT8 GEMMs, plus 8-bit attention when the tier asks for it. Streamed INT8 blocks move about half the
// bytes, and the plan counts the INT8 sizes. Each request takes one fused-route snapshot, sizes its arena from it and
// runs every step under it. With every P6/P7 option off (the bundle's models, tier reference, exact attention, no LoRA,
// the moss scheduler with euler and the official noise) the output is bit-identical to P5.
// P7b: img2img, inpainting and the hires fix (Request::init_image, mask, hires; pipeline_image.h for the host side):
// the VAE encoder (flux2_vae_encoder) sits next to the decoder in the arena's persistent region and the plan counts it
// in every phase; each encode runs on the stack with the room the resident models leave (exact tiling as needed), the
// init image's before the DiT is loaded. A hires fix plans for its target size and reuses the loaded models.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "models/lora/side_branch.h"
#include "pipeline/accel/accel.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/png_writer.h"
#include "pipeline/quality_report.h"
#include "pipeline/sampler.h"
#include "pipeline/samplers/initial_noise.h"
#include "tokenizer/prompt_tokens.h"
#include "planner/planner.h"
#include "util/json.h"
#include "weights/file_reader.h"
#include "weights/quant/quant_model.h"

namespace mi::pipeline {

inline constexpr std::int64_t kMaxSeed = std::int64_t{1} << 53;  // the official app's browser-safe bound
inline constexpr int kMaxSteps = 10000;
// Image size bounds for GUI/CLI input. USERGUIDE.md: the model extrapolates from 0.25 MP (512x512) to 4 MP ("2K");
// 4.2 MP admits 2048x2048 (4.19 MP) and other ~4 MP aspect ratios. The side bound keeps extreme aspect ratios out.
inline constexpr int kMaxSide = 4096;
inline constexpr std::int64_t kMaxPixels = 4'200'000;
// Upper bound for PipelineOptions sizes taken from user input (vram_option_bytes).
inline constexpr std::size_t kMaxVramBytes = std::size_t{1024} << 30;

// One generation, mirroring the assets' parameters.json (gradio_app.py GenerationConfig) plus engine choices.
struct Request {
  std::filesystem::path model;  // bundle directory (base or sft): {text_encoder, tokenizer, transformer, vae}
  std::string prompt;
  std::string negative_prompt;
  int height = 1024;
  int width = 1024;
  int steps = 50;
  double cfg = 3.0;
  double shift = 5.0;
  // -1: random below 2^32 (the app's behaviour). torch seeds mt19937 with the low 32 bits, so seed s and
  // s + 2^32 give the same noise.
  std::int64_t seed = 42;
  std::vector<int> uncond_skip_layers = {22};
  std::vector<int> slg_layers;
  double slg_scale = 0.0;
  double slg_start = 0.0;
  double slg_stop = 1.0;
  int max_sequence_length = 4096;
  SamplerKind sampler = SamplerKind::kEuler;
  // Standard samplers and schedulers (pipeline/samplers): kMoss with euler keeps the official arithmetic bit for bit.
  SchedulerKind scheduler = SchedulerKind::kMoss;
  double flow_multiplier = 1.0;                       // FlowSampling::multiplier
  NoiseLayout noise_layout = NoiseLayout::kOfficial;  // the initial noise's layout (initial_noise)
  NoiseSource noise_source = NoiseSource::kPhilox;    // stochastic samplers' noise
  double eta = 1.0;                                   // euler_ancestral, dpmpp_2m_sde
  double s_noise = 1.0;                               // stochastic samplers' noise multiplier
  std::filesystem::path dataspace_path;               // the dataspace sampler's parameters (load_dataspace)
  bool save_metadata = true;  // front ends: write the generation settings into the PNG (png_writer.h)
  // Runtime LoRAs (models/lora), applied in order with their strengths: low-rank adapters on the block GEMMs run as
  // side branches (a strength change between requests is instant); the others are merged into the DiT at load (a
  // strength change re-merges them from the pristine weights kept on the host).
  std::vector<lora::LoraSpec> loras;
  // img2img (roadmap P7b, diffusers' img2img / inpainting convention). init_image (null: none): RGB (3 channels), any
  // size up to kMaxInputSide per side and kMaxInputPixels, shared (a request copy does not copy the pixels); it is
  // brought to height x width by `resize` (pipeline_image.h; crop by default) and VAE-encoded (the posterior mean) to
  // z0. Of the N scheduled steps (after the scheduler's snapping), the first t = int(max(N - min(N * s, N), 0)) are
  // skipped for denoise_strength s (strength_start_step), and sampling starts from
  //   x = sigma[t] * noise + (1 - sigma[t]) * z0
  // with `noise` the request's seeded initial noise (the same as txt2img). s = 1 without a mask is txt2img, bit for
  // bit (the init image is not used: Result::image_mode says "txt2img" and image_note why); s = 0 runs no step: the
  // result is the VAE round trip decode(encode(image)), and neither the text encoder nor the DiT runs (nor is it
  // planned). Not with the dataspace sampler.
  // Guidance windows (uncond_skip_layers, slg_start / slg_stop) keep the official pipeline's step-index convention:
  // they are measured over the steps a pass actually runs (step i of the strength-cropped schedule over its length),
  // not over the full schedule.
  std::shared_ptr<const Image> init_image;
  double denoise_strength = 1.0;  // [0, 1]
  // Inpainting ("latent blend" with the base model: MOSS-Image has no inpainting checkpoint; null: none): 1 channel, the
  // init image's size, resized like it; >= 128 regenerate, < 128 keep (binarized at 0.5 as diffusers does), then
  // bilinearly interpolated to the latent grid (latent_mask). After every step, at the next step's sigma s',
  //   x = (1 - m) * (s' * noise + (1 - s') * z0) + m * x
  // so the kept region ends exactly at z0 (its VAE round trip). Needs init_image; works at any strength. The mask
  // interpolation is diffusers' FLUX.2 Klein inpainting pipeline's (F.interpolate(mode="bilinear") of the binarized
  // mask, pipeline_flux2_klein_inpaint.py prepare_mask_latents).
  std::shared_ptr<const Image> mask;
  ResizeMode resize = ResizeMode::kCrop;
  // Hires fix: a second pass at width x height (multiples of 16, within the image size bounds; 0 x 0 = off). The
  // first pass (txt2img or img2img at the request size) is decoded, Lanczos-upscaled to the target with `resize`
  // (pixel space: decode -> resize -> encode), encoded, and refined as an img2img pass with `strength` over
  // `steps` scheduled steps (0: the request's steps). Same seed, prompt, sampler and scheduler; the models stay
  // loaded: the first pass is decoded next to the resident DiT (exact tiling as needed, never parking it; the plan
  // counts that decode's smallest tiling in the denoise phase), so the DiT is loaded at most once per request. With a
  // mask the second pass blends against the init image brought to the target size.
  struct Hires {
    int width = 0, height = 0;
    double strength = 0.45;  // [0, 1]; typically 0.3-0.6
    int steps = 0;
    bool enabled() const { return width > 0 || height > 0; }
  } hires;
  bool decode = true;  // false: stop after sampling (final latents only)
  // Control adapter (roadmap P8, models/control: format and semantics): a file pairing a LoRA with a widened DiT input
  // embedder, and its conditioning images. The embedder replaces the DiT's while the adapter is set (never scaled);
  // the file's LoRA joins the runtime LoRA stack after `loras` with lora_strength. Each image is brought to the pass's
  // size by `resize` (as init_image), VAE-encoded (posterior mean) once per pass size, multiplied by its strength and
  // fed as its kind's control latent (clean, the same for every CFG row) on the steps i of a pass with
  // start <= i / steps < end; a kind without an image, or outside its window, gets the zero latent (no control).
  // Works with every tier, streaming, sampler, img2img / inpainting and the hires fix (the images are encoded again at
  // the hires size).
  struct ControlImage {
    std::string kind;                    // one of the adapter's kinds (its control_channels, e.g. "color", "depth")
    std::shared_ptr<const Image> image;  // RGB, or grayscale (replicated to RGB: a depth map, near = white)
    double strength = 1.0;               // [0, 10]: multiplies the control latent (0 = no control)
    double start = 0.0, end = 1.0;       // 0 <= start <= end <= 1
  };
  struct Control {
    std::filesystem::path adapter;  // empty: none
    float lora_strength = 1.0f;     // the file's LoRA part (finite)
    std::vector<ControlImage> images;
    bool enabled() const { return !adapter.empty(); }
  } control;
  // P6b opt-in accelerators (pipeline/accel/accel.h): CFG truncation and step caching, lossy and off by default. The
  // tier is resolved as without them. They apply to each sampling pass (a hires fix's too).
  accel::Options accel;
  // Reference images (the edit layout of Moss-Image-Edit adapters, "index_timestep_zero_text_start" as in ComfyUI PR
  // #17296; moss_dit::RefKv): each image is converted like PIL convert("RGB") (gray replicated, alpha dropped), sized
  // by the training rule (reference_size, PIL Lanczos) and VAE-encoded (posterior mean, normalized: img2img's z0) once
  // per request; each sampling pass then computes every block's reference keys and values once (the prefill, at
  // t = 0), and every row (cond and uncond) attends to its own tokens plus all reference tokens. An uncond row's
  // skipped layers skip only the row: the reference keys are computed without skips (ComfyUI's cached path). At most
  // kMaxReferenceImages images and kMaxReferenceTokens tokens (Σ (h/16)(w/16) of the sized images); not with an SLG row
  // (slg_scale > 0). Meant with a moss-edit-lora adapter in `loras` (Result warns otherwise). Empty: none, and nothing
  // of this runs.
  struct References {
    std::vector<std::shared_ptr<const Image>> images;  // in order (the source image first)
    int max_edge = 512;  // [16, 2048], floored to a multiple of 16; the training used 256, the authors' workflow 512
    bool enabled() const { return !images.empty(); }
  } references;
};

inline constexpr int kMaxReferenceImages = 4;
inline constexpr std::int64_t kMaxReferenceTokens = 16384;  // moss_dit::kMaxReferenceTokens
// Moss-Image-Edit's reference sizing (moss_edit/processing.py resize_image): limit = max_edge // 16 * 16,
// scale = min(1, limit / max(width, height)), each side min(limit, max(16, floor(side * scale / 16 + 0.5) * 16)).
struct ReferenceSize {
  int width = 0, height = 0;
};
ReferenceSize reference_size(int width, int height, int max_edge);
// A reference image as the VAE encodes it: RGB, at reference_size (PIL convert("RGB") then Image.resize(LANCZOS)).
Image reference_pixels(const Image& image, int max_edge);

// Throws std::invalid_argument with the official app's wording where it has one.
void validate(const Request& request);

// The total number of denoise steps the request runs: the first pass's schedule (the dataspace sampler's step count,
// else the scheduler's) minus the steps its denoise strength skips when it starts from an init image, plus the hires
// fix's second pass likewise. Does not validate; generate()'s steps_scheduled.
std::int64_t schedule_steps(const Request& request);

// Reads a parameters.json document: either the asset file ({"schema_version", "parameters": {...}}, versions 1 and 2
// as gradio_app.parse_parameters reads them) or a bare parameters object. Skip-layer lists may be "22,24" strings
// or integer arrays; a null seed means -1. `model` is left empty. Engine keys beyond the app's: denoise_strength,
// resize, hires_strength, hires_steps (the init image, mask and hires size are a front end's: e.g. the CLI reads
// "init_image" / "mask" paths and "hires"). Throws std::invalid_argument.
Request request_from_parameters(const json::Value& doc);

// GenerationConfig.layers (gradio_app.py:119-129): "22, 24" -> {22, 24}, blank -> {}. Throws std::invalid_argument
// on anything but comma-separated integers (range and uniqueness are checked by validate()).
std::vector<int> parse_layer_list(std::string_view text);

// A user-supplied memory size (`amount` units of `unit_bytes`, e.g. GiB) for PipelineOptions::vram_budget or
// vram_reserve, in bytes. Throws std::invalid_argument naming `what` unless it is finite, positive, at least 1 MiB
// (smaller values would truncate to 0, which means "no cap") and at most kMaxVramBytes.
std::size_t vram_option_bytes(double amount, double unit_bytes, std::string_view what);

struct PipelineOptions {
  planner::Mode mode = planner::Mode::kBalanced;
  // Cap on this process's device memory as NVML reports it (CUDA context included); 0 = free device memory -
  // vram_reserve. The arena gets the budget minus planner::Budgets::context_bytes and vram_reserve.
  std::size_t vram_budget = 0;
  // Device memory left outside the arena for what is not allocated from it: CUDA context growth, lazily loaded
  // kernel modules, cuBLASLt handles.
  std::size_t vram_reserve = std::size_t{384} << 20;
  // Cap on the process's private host memory (planner/host_memory.h); 0 = the mode's target.
  std::size_t host_ram_budget = 0;
  bool direct_io = false;  // O_DIRECT for every weight read (cold-cache behaviour)
  // The DiT's load from its weight files: unset = adaptive: direct I/O (O_DIRECT / FILE_FLAG_NO_BUFFERING; RTX 5090:
  // 10.9 GB/s against 4.3 GB/s buffered) unless direct I/O gains nothing on the files (weights::direct_io_support:
  // tmpfs, network file systems, btrfs compressed extents, a refused open), the load is a parking reload, or at least
  // 90 % of the files is in the page cache already (Linux: weights::page_cache_fraction before every load; Windows: once
  // buffered loads are measured faster than direct ones); then buffered. With the files in the page cache the learned
  // load rates decide once both modes were measured from it (each is tried once when only the other is known), and a
  // parking reload reads direct when they chose direct (choose_dit_load_io). false = buffered; true = direct I/O (a
  // refused open still reads buffered). direct_io implies true. Streamed DiT blocks keep reading buffered every step (the page
  // cache serves them) unless direct_io. Result::dit_load reports what a load used and why.
  std::optional<bool> dit_load_direct_io;
  bool embedding_cache = true;
  std::filesystem::path cache_dir;  // embedding-cache disk spill; empty = memory only
  // A held arena is recreated at the plan's size when a request needs less than this fraction of it, so a launcher
  // does not pin the device memory of its largest request forever (resident models are reloaded then, reported as
  // MemoryStats::dit_reloaded). 0 = never shrink.
  double arena_shrink_below = 0.5;
  // Fixed machine rates for the planner's cost model (a launcher's own profile, reproducible tests): no probe, no
  // learning from measured passes. Unset: probed at the first generate() and refined from every pass.
  std::optional<planner::HardwareProfile> hardware;
  // DiT block implementation (all give the same bits). kFused (default) falls back to kOpByOp when the startup
  // self-check fails; kFusedGraphs adds CUDA graphs (opt-in: ~0 measured gain, holds graph memory).
  moss_dit::BlockPath block_path = moss_dit::BlockPath::kFused;
  // P6/P7a model sources, each used instead of the bundle's component (empty: the bundle's); the tokenizer and configs
  // always come from the bundle. Each file is identified as its component by its shapes before the bundle is indexed,
  // and the bundle's index then leaves that component out (weights::BuildOptions::skip): its folder's weights are not
  // read, so the folder needs only its config.
  //   dit_file: a single-file DiT (INT8 ConvRot G = 256, mixed INT8/bf16, or bf16 safetensors: the Comfy-Org /
  //     community files and `moss-infer quantize` output).
  //   text_encoder_file: an INT8 ConvRot or W4A8 Qwen3.5 text encoder (qwen35_text::QuantTextModel); its embeddings
  //     are cached under its own identity and numerics. Held in RAM it is pinned (the plan's host-resident residency).
  //   vae_file: a single FLUX.2 VAE file (flux2_vae::index_vae_file: e.g. the full decoder), decoded (and, when it
  //     holds an encoder, encoded) with its own latent statistics rounded to bf16 as the bundle stores them: the
  //     official pipeline's values for the Comfy-Org files. img2img with a decoder-only file encodes with the bundle's
  //     vae/diffusion_pytorch_model.safetensors, read directly, which it then needs.
  std::filesystem::path dit_file;
  std::filesystem::path text_encoder_file;
  std::filesystem::path vae_file;
  // P6 step 41: a step-embedder replacement (moss_dit::load_time_embedding, e.g. a distill bundle's
  // time_embedding.safetensors) applied to whichever DiT is loaded, computed in fp32 (MossDit::set_time_embedding).
  // Read when the pipeline is constructed (moss_dit::DistillError for a bad file).
  std::filesystem::path time_embedding_file;
  bool lora_strict = false;  // lora::LoadOptions::strict: unknown LoRA keys and lossy merges are errors
  // Whether a host-resident text encoder may hold a pinned copy (planner::TeHostPin, planner.h's file comment): auto
  // (default) where the machine clearly has the room, on whenever it fits the host limit, off never. A pinned copy
  // whose allocation fails falls back to streaming (Result::te_note says so).
  planner::TeHostPin te_host_pin = planner::TeHostPin::kAuto;
  // Quality tier and attention kernel (planner::resolve_tier: what each tier runs and refuses; unset attention: the
  // tier's default). The public interfaces (C API, CLI, moss-server) make the caller choose the tier.
  planner::Tier tier = planner::Tier::kReference;
  std::optional<planner::AttentionMode> attention;
};

struct Timings {
  double index_load_s = 0;  // weight index + tokenizer (first use of a bundle only)
  double tokenize_s = 0;
  double plan_s = 0;         // cache lookups + planning
  double text_encode_s = 0;  // the whole text-encoder pass (0 on a full cache hit), of which:
  double te_weight_load_s = 0;  // streaming: readers' start to the last byte on the device (overlaps compute)
  double te_weight_wait_s = 0;  // compute stream waiting for weights
  double te_compute_s = 0;
  double dit_load_s = 0;  // 0 when the DiT was resident
  // Runtime LoRAs and a control adapter's LoRA part (Result::loras / control): reading and parsing the adapter files
  // and uploading the side branches (lora_load_s), and merging adapters into a loaded DiT in place: a strength change's
  // re-merge, or a control adapter switched on or off (lora_merge_s). 0 when the stack was reused unchanged.
  double lora_load_s = 0;
  double lora_merge_s = 0;
  double refine_s = 0;    // text refiner, all rows
  double denoise_s = 0;
  double vae_load_s = 0;
  double decode_s = 0;     // every decode (the hires fix decodes twice)
  // img2img / inpainting / hires fix: VAE encoder load, every encode, and the host-side image work (resizes,
  // the hires upscale, mask preparation, per-step inpainting blends).
  double vae_encoder_load_s = 0;
  double encode_s = 0;
  double image_ops_s = 0;
  // Reference images (Request::references; 0 without): their encodes (within the VAE encoder's phase, as img2img's)
  // and every pass's prefill (before its first step; not in denoise_s or step_s).
  double reference_encode_s = 0;
  double reference_prefill_s = 0;
  double total_s = 0;
  std::vector<double> step_s;  // every sampling step, both passes of a hires fix
};

struct MemoryStats {
  std::size_t arena_capacity = 0;
  std::size_t arena_high_water = 0;   // peak arena use during the call (persistent + stack)
  std::size_t device_peak = 0;        // peak of (free memory at pipeline construction - free memory now)
  std::size_t free_at_start = 0;      // free device memory when the pipeline was constructed
  int arena_creations = 0;            // arenas created during the call (first use, growth, shrink, parking)
  bool dit_parked_for_text_encoder = false;
  bool dit_parked_for_decode = false;
  // A resident DiT was dropped outside the plan's parking (arena growth or shrink, a phase that did not fit) and
  // loaded again in this call.
  bool dit_reloaded = false;
  int dit_loads = 0;  // DiT loads in the call (0 when it was resident or not needed; at most 1 unless the plan parks)
  // Host copies the pipeline keeps across calls (Pipeline::release_device_memory, a host-resident text encoder) at the
  // end of the call.
  std::uint64_t host_copy_bytes = 0;
  // Host memory the loaded DiT keeps for switching a merged adapter set back on (moss_dit::MossDit::memo_bytes: a
  // control adapter's merged text-refiner bits and their pristine copies, at most 1 GiB), at the end of the call.
  std::uint64_t merge_memo_bytes = 0;
  std::size_t host_private_start = 0;  // process private host memory when the call started
  std::size_t host_private_peak = 0;   // its peak, sampled at phase boundaries
};

// DiT block streaming of one generate() (P4 step 28), from CUDA events on the GPU timeline (moss_dit::StreamStats).
struct DitStreamStats {
  bool streamed = false;   // some blocks were streamed
  int resident_blocks = 0;
  int buffers = 0;
  std::string source;      // "file" (page cache or disk) or "host_pinned"
  std::uint64_t bytes = 0;  // copied to the device over every step
  double copy_s = 0;        // copy streams busy
  double stall_s = 0;       // compute stream waiting for streamed weights
  int stalls = 0;           // streamed blocks that made the compute stream wait (> 50 us)
  double gbps() const { return copy_s > 0 ? static_cast<double>(bytes) / copy_s / 1e9 : 0.0; }
  // Fraction of the copy time hidden behind compute (AC10).
  double hidden() const { return copy_s > 0 ? std::max(0.0, 1.0 - stall_s / copy_s) : 0.0; }
  struct Step {
    double gbps = 0, hidden = 0, stall_s = 0;
    int stalls = 0;
    // A call of this step reused the P6b step cache: the streaming pass ended after block 0 and its copies are not
    // counted (all zero when every call of the step reused it).
    bool cached = false;
  };
  std::vector<Step> steps;  // per finished step
};

// How the DiT's weights were read by the last load of a call (MemoryStats::dit_loads > 0): the report's "dit_load".
struct DitLoadReport {
  std::string source;   // "" (no load in the call), "files", or "host_copy" (the weights a soft unload kept in RAM)
  std::string io;       // "files": "direct" or "buffered"
  std::string note;     // why buffered (PipelineOptions::dit_load_direct_io, or what the file system allows)
  std::uint64_t bytes = 0;
  double seconds = 0;
  double gbps() const { return seconds > 0 ? static_cast<double>(bytes) / seconds / 1e9 : 0.0; }
};

// One sampling pass of an image request (the request's pass, and the hires fix's second pass).
struct PassReport {
  bool ran = false;     // the pass was reached (not cancelled before it)
  int width = 0, height = 0;
  int schedule_steps = 0;  // N: the full schedule's steps after snapping
  int start_step = 0;      // steps skipped by the denoise strength (strength_start_step); 0 for txt2img
  int steps_run = 0;
  double strength = 1.0;
  bool encoded = false;           // an image was VAE-encoded for this pass (img2img start or inpainting blend)
  std::string encode_schedule;    // the encode(s) run: "untiled" or the tile plan (flux2_vae_encoder::TilePlan)
  // The hires fix's first pass only: its decode for the upscale ("untiled" or the tile plan; next to the DiT, which is
  // never parked for it). The final decode is Result::decode_schedule.
  std::string decode_schedule;
};

struct Result {
  // The final image size: the hires fix's target when it ran, else the request's. Latents always match it.
  int height = 0, width = 0;
  std::vector<std::uint8_t> rgb;  // [height, width, 3]; empty if !decode or cancelled
  // Final latents [128, height/16, width/16]; the partial result if cancelled during sampling (the latents after the
  // last finished step, also when the cancel lands inside a step), empty if cancelled before it started.
  std::vector<float> latents;
  std::uint64_t seed = 0;         // the seed actually used
  int steps_requested = 0;
  // Steps the request samples, both passes (after the schedule's snapping and the denoise strengths), and those run.
  int steps_scheduled = 0;
  int steps_run = 0;
  // img2img / inpainting / hires fix: what ran ("txt2img", "img2img" or "inpaint"), each pass, and the VAE encoder
  // (its file; empty when nothing was encoded).
  std::string image_mode = "txt2img";
  std::string image_note;  // e.g. why an init image was not used (strength 1 without a mask)
  PassReport first_pass;
  PassReport hires;
  std::string vae_encoder_source;
  // Set whenever a cancel request was seen, even after the last step (then steps_run == steps_scheduled and the
  // decode is skipped).
  bool cancelled = false;
  int cond_tokens = 0, uncond_tokens = 0;
  // Prompt token counts and truncation (tokenizer/prompt_tokens.h); uncond only under CFG.
  tok::PromptTokenReport cond_token_report, uncond_token_report;
  int te_cache_hits = 0;    // sequences served by the embedding cache
  bool te_skipped = false;  // every sequence hit: the text encoder did not run
  // The decode actually run: tiled (exact, full-width row bands) and the schedule, or untiled. It can differ from
  // plan.vae: the decode takes the fastest schedule the arena it finds leaves room for (a held arena may be larger).
  bool decode_tiled = false;
  std::string decode_schedule;
  planner::ExecutionPlan plan;
  planner::HardwareProfile hardware;  // the transfer rates the plan was made with
  // DiT block path the denoise ran (fused_graphs only when every block was resident), and why it differs from
  // PipelineOptions::block_path (empty when it does not).
  moss_dit::BlockPath block_path = moss_dit::BlockPath::kFused;
  std::string block_path_note;
  // P6: the tier that ran (tier, DiT precision, attention mode, what runs), the DiT's source (the bundle's transformer
  // directory or PipelineOptions::dit_file) and its block linears (moss_dit::describe_linears).
  planner::TierDecision tier;
  QualityReport quality;  // the report's "quality" object (pipeline/quality_report.h)
  std::string gpu_arch;   // the GPU class of the device ("sm_89")
  std::string dit_source;
  std::string dit_linears;
  std::string time_embedding;  // PipelineOptions::time_embedding_file when a replacement step embedder ran (else empty)
  std::string te_source;     // the bundle's text_encoder directory or PipelineOptions::text_encoder_file
  std::string te_precision;  // qwen35_text::precision_name: "bf16", "int8_convrot", "w4a8"
  std::string te_note;       // e.g. a pinned host copy that could not be allocated (the encoder streamed instead)
  std::string preload_note;  // preload(): what it did not load and why (e.g. a DiT the plan parks for the encoder)
  std::string vae_source;    // the bundle's vae directory or PipelineOptions::vae_file
  // Runtime LoRAs: how each file was read and applied (lora::FileReport), and the losses to show the user (merged
  // adapters requantized on INT8 block GEMMs).
  std::vector<lora::FileReport> loras;
  std::vector<std::string> lora_warnings;
  std::vector<std::string> lora_notes;  // how files were read (e.g. the metadata alpha applied)
  // The control adapter (Request::control; adapter empty: none): its file, kinds in channel order, how its LoRA part was
  // read (not in `loras`), and the encodes of its images.
  struct ControlReport {
    std::string adapter;
    std::vector<std::string> kinds;
    std::string base_model;  // the file's metadata, informational
    lora::FileReport lora;
    int encodes = 0;
    std::string encode_schedule;  // "untiled" or the tile plans, per encode
  } control;
  // Reference images (Request::references; count 0: none, and front ends then report nothing of it): the sized images,
  // their tokens, the per-pass key / value cache and how often it was computed, the encodes' tilings, and what the
  // user should know (no edit adapter loaded, an edit adapter without references).
  struct ReferenceReport {
    int count = 0;
    std::vector<ReferenceSize> sizes;
    std::int64_t tokens = 0;
    std::size_t cache_bytes = 0;
    int prefills = 0;
    std::string encode_schedule;
  } references;
  std::vector<std::string> reference_warnings;
  DitStreamStats dit_stream;
  DitLoadReport dit_load;
  // P6b: what the accelerators did (CFG-truncated steps, step-cache calls and reuses), and what the user should know
  // about them for this model (e.g. a step-cache threshold that moves the base model's composition, or a cache that
  // could not engage).
  accel::Stats accel;
  std::vector<std::string> accel_warnings;
  Timings timings;
  MemoryStats memory;
};

// Phases of generate(), in order. kTextEncoder is entered even on a full embedding-cache hit and kDitLoad even when
// the DiT is resident; kDecode only when decoding. img2img encodes the init image in kDitLoad (before the DiT load);
// the hires fix enters kDecode (the first pass, decoded for the upscale), then kDenoise (encode and second pass) and
// kDecode again.
enum class Phase { kTextEncoder, kDitLoad, kDenoise, kDecode };

// A latent preview during sampling (latent_preview: a linear latent -> RGB map, no VAE) of the x0 estimate after a
// step: x0 = x_t - sigma_t * v, with the step's effective velocity v = (x_t - x_prev) / (sigma_t - sigma_prev).
struct PreviewImage {
  int step = 0, num_steps = 0;  // after `step` of `num_steps` finished steps
  int width = 0, height = 0;    // the image size
  const std::uint8_t* rgb = nullptr;  // [height, width, 3], valid during the callback only
};

struct GenerateControl {
  // Once per call, after planning and before the first phase (the launcher shows plan.explanation and the mode).
  std::function<void(const planner::ExecutionPlan&)> on_plan;
  // After every sampling step; returning false cancels. With the hires fix the steps count on across both passes
  // (step and num_steps cover both), as do the previews'.
  StepCallback on_step;
  // After every preview_every-th step (and the last), before on_step. Its buffers count in the denoise phase's plan.
  std::function<void(const PreviewImage&)> on_preview;
  int preview_every = 1;
  std::function<bool(Phase)> on_phase;        // before each phase starts; returning false cancels
  // Checked before each phase and after every step, and inside the phases: between text-encoder layers, per DiT
  // weight group while loading, between DiT blocks and per tiled-decode band (each phase then keeps at most one
  // layer / block / band queued on the GPU). The pipeline stays usable after a cancel.
  const std::atomic<bool>* cancel = nullptr;
  // Internal (the NVFP4 converter's calibration capture, pipeline/nvfp4_calibration.h): installed on the DiT for the
  // call (moss_dit::MossDit::set_linear_input_hook). Needs PipelineOptions::block_path kOpByOp and a bf16 DiT.
  moss_dit::LinearInputHook dit_linear_inputs;
};

// The model a DiT was trained as, from its input embedder (video_embedder.weight, which the INT8 / INT4 ConvRot
// conversions keep in bf16, so the quantized files of a model share it): "base" or "sft" for the official MOSS-Image
// weights and the files derived from them, "" for anything else (or unreadable). XXH64 of the tensor's 786 KB; `index`
// is a bundle's or a single file's (quant::to_model_index).
std::string dit_lineage(const weights::ModelIndex& index);

// What a single DiT file (PipelineOptions::dit_file) is, as loading it determines before any device work: the
// supported-model check (weights::check_supported_dit_file), the quantized importer (quant::open_quant_model: every
// linear's layout and MOSS-Image shapes) and its index (quant::to_model_index with INT4 linears kept packed, which
// reads and validates every quantized linear's scales: ~1.5 MB for an INT8 DiT, ~130 MB of block scales for a
// half-NVFP4 one). Device-free; Pipeline loads a DiT file through it.
struct DitFileInfo {
  std::unique_ptr<quant::QuantIndex> quant;  // the index the DiT loads from; null when `refusal` is set
  // Of the block linears: kInt4 when some is INT4, else kNvfp4 when some is NVFP4, else kInt8 when some is INT8, else
  // kBf16.
  planner::Precision precision = planner::Precision::kBf16;
  // moss_dit::describe_linears (e.g. "int8_convrot256", "mixed (210 int4_convrot256, 42 bf16)"): Result::dit_linears.
  std::string linears;
  // Linears with a static activation scale (input_scale), which the engine ignores (it scales activations per token),
  // and the remark that says so ("N linears carry a static input_scale; ignored (activations are scaled per token)";
  // empty when there are none). The plan explanation shows it.
  int static_input_scales = 0;
  std::string remarks;
  bool nvfp4 = false;   // some linear is NVFP4
  std::string refusal;  // "<file>: <reason>" when nvfp4 and the NVFP4 refusal reason is not empty, else empty
};
// `nvfp4_refusal` says why NVFP4 linears cannot run where the file is meant to; it is called only for a file with NVFP4
// linears (Pipeline passes moss_dit::nvfp4_unsupported_reason, which queries the device). A non-empty reason answers
// the file by `refusal` alone (no index, precision or linears), before its block scales are read; an empty one (or no
// function) indexes NVFP4 linears like the others.
// Throws std::invalid_argument when `file` exists but is not a regular file ("<file>: not a regular file"),
// weights::IoError (MI_ERR_IO) when it does not exist (any more: "<file>: no such file") or cannot be read,
// weights::UnsupportedModelError for another model, container or add-on, weights::IndexError (MI_ERR_MODEL, the message
// naming the file) for a layout or shape the importer refuses or a file mixing INT4 ConvRot and NVFP4 linears,
// weights::FormatError for a malformed header. (Pipeline
// keeps its own std::invalid_argument "DiT file not found: <file>" for a dit_file that does not exist.)
DitFileInfo inspect_dit_file(const std::filesystem::path& file, const std::function<std::string()>& nvfp4_refusal);
// The same with the reason already known (e.g. for another device, or "" to index NVFP4 linears).
DitFileInfo inspect_dit_file(const std::filesystem::path& file, std::string_view nvfp4_refusal);
// The same on a header the caller already parsed (messages and reads use header.path(); the header is not read
// again). A parsed header is a safetensors file, so the GGUF / pickle container check of the path form is skipped.
DitFileInfo inspect_dit_file(const weights::SafetensorsFile& header,
                             const std::function<std::string()>& nvfp4_refusal);

// How a DiT load reads its weight files (PipelineOptions::dit_load_direct_io; pure, for the pipeline and its tests).
// forced_*: the options; support: weights::direct_io_support of the files (the first that does not gain); parked: a
// parking reload; cached_fraction: the files' bytes in the page cache (weights::page_cache_fraction), negative when
// unknown; *_gbps: the learner's load rates (0: none measured): direct_gbps (warm or cold alike), buffered_warm_gbps
// (buffered loads from a warm cache), buffered_gbps (every buffered load, warm and cold mixed); warm_direct_before: the
// warm mode in use (HardwareProfile::warm_direct), for the hysteresis; bytes: the files' size; direct_refused: an
// O_DIRECT open of these files was refused earlier in this process.
//   - at least kDitCachedFraction resident: with both warm rates known, the faster mode (planner::warm_dit_load_direct,
//     with hysteresis); with one known, the other, so the learner measures both: only for a load it measures (bytes
//     >= kDitLoadMeasureBytes), and direct never where an open refused it; otherwise buffered. The rates are kept in
//     the hardware profile, so a new process explores again only when a rate is missing from it;
//   - a parking reload: buffered, so the page cache keeps the files for the next one, unless both warm rates are known
//     and direct is the warm choice;
//   - less resident: direct; residency unknown (Windows): buffered only once the mixed buffered rate beats direct.
inline constexpr double kDitCachedFraction = 0.9;
inline constexpr std::uint64_t kDitLoadMeasureBytes = std::uint64_t{1} << 30;  // smaller loads are not measured
struct DitIoInputs {
  bool forced_direct = false, forced_buffered = false;
  weights::DirectIoSupport support{true, {}};
  bool parked = false;
  double cached_fraction = -1.0;
  double buffered_gbps = 0, direct_gbps = 0, buffered_warm_gbps = 0;
  bool warm_direct_before = false;
  std::uint64_t bytes = 0;
  bool direct_refused = false;
};
struct DitIoDecision {
  bool direct = true;
  std::string note;   // why buffered, or what decided (empty for a plain direct load)
  bool warm = false;  // at least kDitCachedFraction resident: a buffered load's rate is a warm one
};
DitIoDecision choose_dit_load_io(const DitIoInputs& in);

// Whether a loaded DiT's merge memo (MossDit::memo_bytes, a control adapter's switch-back bits) is dropped before this
// plan runs: the plan's host-private estimate (which holds the memo) is over the target its configuration is held to
// (ExecutionPlan::host_held_target_bytes), or over an explicit host budget. Pure.
bool merge_memo_unaffordable(const planner::ExecutionPlan& plan, std::size_t host_budget);

// The PNG metadata of a generation (png_writer.h generation_text_chunks): settings, the tier that ran, and the models
// by file name (the bundle, and each separately loaded file); sha256 left empty (not computed here).
GenerationMetadata generation_metadata(const Request& request, const Result& result);

// Pipeline::preload: the request to load the models for, before the first generate().
struct PreloadOptions {
  // Planned as generate() plans it: the models it names (the bundle, PipelineOptions' files, its LoRAs and control
  // adapter), its size, steps and CFG, and its prompt's token count (it sizes the text-encoder and denoise workspaces;
  // a blank prompt plans for a short one, and the arena's slack covers a few hundred more tokens). The embedding cache
  // is not consulted: the text encoder is planned as for a new prompt.
  Request request;
  // Checked between the models, per DiT weight group and between a host-resident text encoder's layer groups. A
  // cancelled preload leaves the pipeline usable (Result::cancelled is set): the models loaded before the cancel stay;
  // a DiT whose load was cancelled is dropped with its partial weights (the arena is recreated), and a text encoder's
  // partial host copy is freed.
  const std::atomic<bool>* cancel = nullptr;
  // As GenerateControl's. on_plan: once, right after planning and before any load (a BudgetError is thrown before it).
  // on_phase: before each load step, whether or not that model is already resident; returning false cancels
  // (Result::cancelled, as `cancel`). kTextEncoder before a host-resident text encoder's host copy; kDecode before the
  // VAE steps (Phase has no VAE value: kDecode is the VAE's; an image request fires it for the encoder and again for
  // the decoder); kDitLoad before the DiT. A step the plan skips fires nothing. kDenoise is never reported.
  std::function<void(const planner::ExecutionPlan&)> on_plan;
  std::function<bool(Phase)> on_phase;
};

// Pipeline::release_device_memory: what was freed and kept.
struct ReleaseReport {
  double seconds = 0;
  std::size_t vram_released = 0;      // the arena's capacity plus the device memory held outside it (LoRA side branches,
                                      // a control adapter's embedder)
  bool dit_host_copy = false;         // the DiT's weight files are held in RAM: the next load re-uploads them from there
  bool te_host_copy = false;          // a host-resident text encoder's copy is kept
  std::uint64_t host_copy_bytes = 0;  // host memory the kept copies hold
  double host_copy_s = 0;             // reading the DiT's files into RAM (part of seconds)
  std::string note;                   // why a copy was not kept (the host budget), or empty
  bool cancelled = false;             // the cancel flag was set on entry: returned at once, nothing was released
};

class Pipeline {
 public:
  explicit Pipeline(PipelineOptions options = {});
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // Not thread-safe; one generation at a time per instance. Throws planner::BudgetError (before any phase) when
  // the request cannot fit the budgets in any mode, std::invalid_argument on an invalid request.
  Result generate(const Request& request, const GenerateControl& control = {});

  // Loads what the planner keeps resident for `options.request` without generating: the arena at the plan's size, the
  // DiT with the plan's residency (not when the plan parks it for the text encoder: the first generate() would evict it
  // before using it; Result::preload_note says so), the VAE decoder (unless the plan parks the DiT for the decode), the
  // VAE encoder when
  // the request starts from an image, its LoRA side branches and control embedder, and the text encoder's host copy
  // when the plan keeps it host-resident (a text encoder streamed from its files has nothing to keep). The same budgets
  // and errors as generate(); a following generate() of a request like it reuses everything (MemoryStats::dit_loads
  // == 0) and gives the same bits as without the preload. Returns the plan, the tier, the load timings and memory
  // (steps_run 0, no image).
  Result preload(const PreloadOptions& options);

  // Soft unload: frees the arena and the device memory held outside it (every resident model, LoRA side branches, a
  // control adapter's embedder) first, so another application can use the GPU; the next generate() plans and loads
  // again. Then, with keep_host_copy, the weights stay in host RAM where the host has the room (the copy leaves
  // planner::host_pin_margin() of the available memory free and fits PipelineOptions::host_ram_budget when one is
  // set): the DiT's weight files are read into pinned memory (the next load re-uploads them from there over PCIe,
  // bit-identical, then frees them; its streamed blocks, if any, still read the files) and a host-resident text encoder
  // keeps its copy. A copy that does not fit, fails or is cancelled (`cancel`, checked between 64 MiB reads) is not
  // kept, with the reason in ReleaseReport::note; the release itself always completes. A `cancel` already set on
  // entry means the caller no longer wants the unload: nothing is released, the report is all zero (the copies
  // already held are reported as they are) and note is "cancelled before it started". Without keep_host_copy every
  // host copy is freed too. The embedding cache stays. Not thread-safe (not during generate()).
  ReleaseReport release_device_memory(bool keep_host_copy, const std::atomic<bool>* cancel = nullptr);

 private:
  friend class Session;  // the component session (pipeline/session.h) drives the same state
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Result run(const Request& request, const GenerateControl& control, bool preload);
};


}  // namespace mi::pipeline
