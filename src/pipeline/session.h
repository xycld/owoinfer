// The component session (component API design §3-§5): mi_generate() split into parts, so a host owns the sampling loop.
// A session over a Pipeline encodes texts, runs the VAE and holds at most one run; a run is one sampling pass at one
// size over a latent batch, and each step() is one model evaluation of the rows the host (or mi_generate's guidance
// layout) picks, in one batched varlen forward.
//
// Every call is a job of the engine's one JobGate (runtime/job_gate.h): one job at a time, the gate's cancel flag
// watched by the call's models, and the ending sequence of end_run() / close(). The session has no busy state of its own.
//
// Planning: every call plans the equivalent mi_generate request with the same functions as Pipeline::run (Impl's
// setup_models, setup_dit_call, denoise_stack_bytes, plan_call, setup_arena, prepare_dit), within the engine's mode and
// budgets. The session keeps that request (model, size, LoRAs, the session's control adapter with the run's LoRA
// strength and images, the step cache with its explicit threshold, sampler euler) and the last run's shape; an
// mi_generate request a run is equivalent to plans exactly as the run (an embedding-cache hit, decode on).
//
// Positions, rows and conventions are design §5.2's: x = (1 - sigma) x0 + sigma noise, each output the DiT's velocity
// (denoised = x - sigma out), the timestep fp32(sigma * 1000), latents in model space ([L, 128, h, w] F32, ComfyUI's
// Flux2 LATENT as is), no shift applied in a step.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/embedding_cache.h"
#include "pipeline/pipeline.h"
#include "runtime/job_gate.h"

namespace mi::pipeline {

// The most tokens one call carries over its rows (h * w + text tokens each): the longest run the fused block path's
// startup self-check validated (MI_MAX_CALL_TOKENS). begin_run refuses a run whose largest call exceeds it.
inline constexpr std::int64_t kMaxCallTokens = moss_dit::kMaxRunRows;
// Rows of a run and of a call (MI_MAX_ROWS), latents included.
inline constexpr int kMaxSessionRows = moss_dit::kMaxRows;

// mi_run_params.cfg's default: a HOST run keeps it (the host guides).
inline constexpr double kDefaultRunCfg = 3.0;

// A session call stopped by the gate's cancel flag (JobGate::cancel, or an ending that cancels the call in flight):
// MI_CANCELLED. The session and an open run stay usable.
struct SessionCancelled : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// Session::adopt_staging: the CUDA runtime refused to register the caller's memory as pinned (MI_ERR_NOT_SUPPORTED).
struct StagingNotPinned : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The text encoder's last hidden state of one prompt, bf16 [tokens, moss_dit::kTextDim] on the host (the tensor before
// the DiT's text refiner). Immutable: it may serve several runs and sessions; a run copies what it needs.
struct Conditioning {
  std::shared_ptr<const Embedding> embedding;
  bool engine_encoded = false;     // encode_texts(); false: a host's own tensor (foreign)
  tok::PromptTokenReport tokens;   // engine-encoded: the prompt's token report
  std::int64_t length() const { return static_cast<std::int64_t>(embedding->data.size()) / embedding->hidden; }
};

struct SessionOptions {
  // The request planned for before the first run fixes the session request's shape: its model, size, steps, CFG,
  // LoRAs and accelerators (validated as mi_generate's; a blank prompt plans for a short one, as preload()).
  Request plan_request;
  int max_rows = 3;     // rows of a run or call, 1..kMaxSessionRows; sizes the pinned staging only
  int max_latents = 1;  // latents of a run's batch, 1..max_rows; sizes the pinned staging only
  // The control adapter (empty: none), fixed for the session: every call then runs the widened input embedder, with
  // zero control latents wherever none is given.
  std::filesystem::path control_adapter;
  // Before each phase a call runs (kTextEncoder: encode_texts; kDitLoad: begin_run; kDecode: the VAE calls and control
  // image encodes); returning false cancels the call. Never kDenoise.
  std::function<bool(Phase)> on_phase;
  // The engine's device (mi_engine_params.device_index): host device tensors must be its memory.
  int device = 0;
};

enum class GuidanceMode {
  kHost,  // the host owns guidance and picks each call's rows (StepArgs::rows), each with its own fixed skip set
  kMoss,  // mi_generate's layout: rows[0] cond, rows[1] uncond, the SLG row; the engine picks each call's rows
};
enum class StepOutput {
  kRows,             // each evaluated row's velocity, out[k] for row k (MOSS: out[2] the SLG row)
  kGuidedVelocity,   // MOSS only: out[0] = guide_prediction() in velocity space
  kGuidedDenoised,   // MOSS only: out[0] = guide_prediction() in denoised space
};

// One row of a run: its conditioning (kept alive by the run), the DiT blocks it skips and its latent in the batch.
struct RunRow {
  std::shared_ptr<const Conditioning> conditioning;
  std::vector<int> skip_layers;
  int latent = 0;
};

// One sampling pass (mi_run_params, already converted from the C structs). Defaults are fixed constants.
struct RunSpec {
  int width = 1024, height = 1024;
  std::vector<RunRow> rows;  // 1..max_rows; their latents cover 0..L-1, L <= max_latents. MOSS: 1 or 2 rows, L = 1
  // The schedule (a position and precompute hint, never a limit on the calls): 2..kMaxRunSigmas values, finite,
  // non-increasing. Empty: engine_sigmas(scheduler, steps, {shift, flow_multiplier}, denoise_strength). With host
  // sigmas the scheduler fields name the schedule they are compared with (RunInfo::schedule_matches).
  std::vector<float> sigmas;
  SchedulerKind scheduler = SchedulerKind::kMoss;
  int steps = 50;
  double shift = 5.0;
  double flow_multiplier = 1.0;
  double denoise_strength = 1.0;
  GuidanceMode guidance = GuidanceMode::kHost;
  StepOutput output = StepOutput::kRows;
  bool exact_sigma = false;  // MI_RUN_EXACT_SIGMA: refuse a sigma that is not an exact schedule entry
  bool crops = false;        // MI_RUN_CROPS: the plan reserves crop_workspace_bound (refused when it costs residency)
  // MOSS only (HOST keeps these defaults).
  double cfg = kDefaultRunCfg;
  std::vector<int> slg_layers;
  double slg_scale = 0.0;
  double slg_start = 0.0;
  double slg_stop = 1.0;
  double cfg_truncation = 1.0;
  std::vector<lora::LoraSpec> loras;
  // The session's control adapter: its LoRA part's strength, and run-level control images (encoded once at
  // width x height by control_resize, with their windows; latent 0 only).
  float control_lora_strength = 1.0f;
  std::vector<Request::ControlImage> control_images;
  ResizeMode control_resize = ResizeMode::kCrop;
  accel::StepCacheKind step_cache = accel::StepCacheKind::kOff;
  double step_cache_threshold = 0;  // (0, 1], required with the step cache on
};

struct RunInfo {
  int latent_channels = moss_dit::kLatentChannels, latent_height = 0, latent_width = 0;
  int steps = 0;               // sigmas - 1
  int outputs = 0;             // output tensors per call: 1 (guided) or the run's rows (MOSS: + the SLG row)
  std::uint32_t rows = 0;      // bit k: row k exists (MOSS: bit 2 = the SLG row)
  bool schedule_matches = false;  // the sigmas are the engine's schedule for the scheduler fields, bit for bit
  std::int64_t off_schedule_calls = 0;
  std::int64_t calls = 0;      // completed steps
  int control_channels = 0;    // the per-call control latent's channels; 0 without an adapter
  int latents = 1;             // the run's latent batch L
  std::size_t staging_bytes = 0;
};

// Host views of the run's pinned staging (Device::kCPU): x [L, 128, h, w] ([128, h, w] for L = 1), out[k] [128, h, w],
// and with an adapter the control latent in x's layout.
struct StagingViews {
  Tensor x;
  std::array<Tensor, kMaxSessionRows> out{};
  int outputs = 0;
  Tensor control;
};

// One model evaluation. Tensors are F32 views, all device (kCUDA: memory of the engine's device, checked per call) or
// all host (kCPU: exactly the staging views, a crop packed from their start).
struct StepArgs {
  float sigma = 0;
  int step = -1;                  // the step position, or -1: from sigma (RunSchedule)
  std::uint32_t rows = 0;         // HOST: bit k evaluates row k, 0 = every row; MOSS: must be 0
  bool control_changed = false;   // MI_STEP_CONTROL_CHANGED: the step cache is invalidated before this call
  cudaStream_t stream = nullptr;  // device tensors: the host's stream (any handle; nullptr = the legacy stream)
  Tensor control;                 // data null: the run's control latents (zeros on a crop call)
  Tensor x;                       // [L, 128, h, w] or [128, h, w]; h x w the run's, or a smaller crop (HOST only)
  std::array<Tensor, kMaxSessionRows> out{};  // out[k] for each evaluated row k (guided: out[0]), x's h x w
};

struct StepReport {
  int step = 0;                // the step position used
  std::uint32_t rows = 0;      // rows evaluated
  bool cached = false;         // the step cache reused its residual
  bool on_schedule = false;
  double seconds = 0;
};

// What end_run returns: the run's Result (plan, tier, timings, memory, accelerators, control and LoRA reports; no
// image, no latents) and the component facts of its report.
struct RunResult {
  Result result;
  RunSpec spec;  // the run's spec (what its report describes)
  GuidanceMode guidance = GuidanceMode::kHost;
  bool schedule_matches = false;
  std::int64_t off_schedule_calls = 0;
  std::int64_t calls = 0;
  std::vector<int> foreign_rows;  // rows whose conditioning was not engine-encoded
  std::uint64_t graph_replays = 0;  // CUDA graph replays of the run's forwards (kFusedGraphs; tests)
  // How the sigmas differ from the engine's schedule for the scheduler fields, the off-schedule calls and the foreign
  // rows (MI_TEXT_COMPONENT_NOTE; "" when none applies).
  std::string note;
};

class Session {
 public:
  // Opens the session: a job of `gate` (JobGate::Busy while another job runs or a session is open). The pipeline and
  // the gate must outlive the session. Throws std::invalid_argument for invalid options (nothing opened).
  Session(Pipeline& pipeline, runtime::JobGate& gate, SessionOptions options);
  // close(), errors swallowed (from the running call's own thread nothing is closed: call close() first).
  // P2: the C ABI must make destroying the session from the running call's own thread impossible (mi_engine_free stays
  // forbidden from a callback; mi_session_close returns MI_ERR_BUSY there and keeps the session).
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // The gate's ending sequence for the session, its open run first: a call in flight is cancelled and waited for.
  // Frees the staging and events; models stay loaded. False when the session was closed already. JobGate::Busy from
  // a callback of the running call.
  bool close();

  // Every prompt in one text-encoder pass, through the embedding cache, planned for the real token counts with the last
  // run's shape (plan_request's before the first run). One conditioning per prompt.
  std::vector<std::shared_ptr<const Conditioning>> encode_texts(std::span<const std::string> prompts,
                                                                int max_sequence_length);

  // Plans and opens a run (above): loads what the plan keeps, applies LoRAs and control, encodes the control images,
  // refines each distinct conditioning once and holds the denoise stack until end_run. Throws std::invalid_argument
  // (the spec, the token cap, a batch or crop bound that would keep fewer DiT blocks resident), planner::BudgetError,
  // SessionCancelled (no run), JobGate::Busy / Refused.
  void begin_run(const RunSpec& spec);
  // The open run's facts, schedule and staging views, read under the gate (JobGate::read_run: Refused without an open
  // run or once its ending began). Not jobs: they do not wait for a step. The views stay valid until end_run.
  RunInfo run_info() const;
  const std::vector<float>& run_sigmas() const;
  StagingViews staging() const;

  // Makes caller-owned host memory the open run's staging block (mi_run_adopt_staging; e.g. a shared-memory region
  // another process maps): registered as pinned (cudaHostRegister, portable), laid out as the run's own block, so
  // staging() then returns views into it; unregistered when the run ends. Before the run's first step call returns only,
  // once per run. `base` page-aligned, `bytes` >= RunInfo::staging_bytes. A job on the run (JobGate::Busy while a step
  // runs, Refused once the run's ending began); std::invalid_argument; StagingNotPinned when the registration is
  // refused (the run's own staging stays in use).
  void adopt_staging(void* base, std::size_t bytes);

  // One model evaluation (above). Every exit synchronizes the engine stream: on return the engine no longer reads x
  // or control and no longer writes out[]; on success (device tensors) the host stream is ordered after the outputs.
  // SessionCancelled (the run usable, the step cache invalidated), std::invalid_argument (the run usable).
  StepReport step(const StepArgs& args);

  // The gate's ending sequence for the run: a step in flight is cancelled and waited for. Releases the run's stack,
  // step cache and control latents; models, LoRAs and the staging stay. nullopt when no run was open. With no run
  // open it still waits for any session call in flight (encode_texts, vae_*: the gate cannot tell them from a
  // begin_run, whose run it would end).
  std::optional<RunResult> end_run();

  // RGB8 (3 channels, sides multiples of 16, the request bounds) -> model-space latents [128, H/16, W/16] F32 (host
  // or device): the posterior mean, exactly mi_generate's img2img and control-image encodes.
  void vae_encode(const Image& image, const Tensor& latents);
  // Model-space latents [128, h, w] F32 -> RGB8 [16h, 16w, 3] U8 (host or device): mi_generate's final decode.
  void vae_decode(const Tensor& latents, const Tensor& rgb);

  // The checks begin_run applies to a spec before any work (the token cap included), for a session of `options`.
  // Throws std::invalid_argument.
  static void validate(const RunSpec& spec, const SessionOptions& options);

 private:
  struct State;
  struct Run;
  static StagingViews staging_views(const Run& run);
  std::unique_ptr<State> s_;
};

}  // namespace mi::pipeline
