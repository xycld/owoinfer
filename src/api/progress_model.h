// Whole-job progress for mi_progress.fraction and eta_seconds (MI_CAP_PROGRESS_FRACTION). Pure host logic, no CUDA: the
// adapter turns the plan into the job's expected phases with time estimates (generate_segments, preload_segments), then
// feeds ProgressModel the phase starts and sampling steps as they happen.
//
// Estimates: the plan's TimeEstimate is the steady state (models loaded as the plan keeps them; dit_load_s covers only
// parking reloads). A model that is not loaded yet adds its load: its bytes at the hardware profile's transfer rates
// (planner::HardwareProfile). The pipeline has no live residency accessor, so the adapter keeps a warm / cold record
// (HeldModels) from the earlier jobs' results and releases; the load estimates are rough, and the measurements replace
// them as the job runs. A hires fix's first pass is priced by its share of the target's pixels, as the planner counts it.
//
// Correction: a phase that ended counts at its measured time (everything before the latest event is done), and the
// remaining sampling steps of the running pass are priced at its measured step time. The first step of a pass also
// carries the pass's setup (text refiner, CUDA graph capture, lazily loaded kernels), so once a later step is measured
// the steps after the first set the price; until then the first step's own time does. A pass not measured yet is priced
// at its plan price scaled by the last measured pass's measured / plan ratio.
//
// Known gaps: the compute of an image encode (img2img, inpainting, control images, the hires fix's upscale) is not
// priced, only the VAE encoder's load; and the rates lag by one job: a job is priced with the rates of the job before
// it (the first job of an engine with the cached hardware profile, or the planner's defaults before the first probe).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "mossinfer/mossinfer.h"
#include "planner/planner.h"

namespace mi::api {

// One expected phase of a job. A DENOISE segment is one sampling pass: `seconds` its work before the first step (an
// image encode's VAE encoder load), then `steps` steps of `step_seconds` each. `optional`: the pipeline may skip it
// (a preload's load step the progress cannot predict exactly); a later phase may start in its place.
struct ProgressSegment {
  mi_phase phase = MI_PHASE_TEXT_ENCODER;
  double seconds = 0;
  int steps = 0;
  double step_seconds = 0;
  bool optional = false;
};

// What the engine holds when a job starts, as far as the adapter can tell (the pipeline reports no residency between
// calls): warm (the last job's models are loaded) or cold, and the host-side copies.
struct HeldModels {
  // A generate or preload finished since the engine was created or last released, and nothing failed or dropped the
  // models since: its plan's models are loaded, the DiT with last_dit's residency and the runtime LoRAs of
  // last_merge_key.
  bool warm = false;
  planner::DitPlan last_dit;
  std::string last_merge_key;
  bool te_host_copy = false;    // a host-resident text encoder's copy (host memory: survives a release that keeps it)
  bool dit_host_copy = false;   // a soft unload kept the DiT's weight files in pinned RAM (the next load reads it)
  bool dit_files_read = false;  // this engine read the DiT's files before: the page cache likely holds them

  // A generate (or preload) returned. Not cancelled: warm, with this plan's DiT and `merge_key`. Cancelled after creating
  // an arena: cold (what it loaded is unknown). `dit_load_source`: Result::dit_load.source ("files", "host_copy", "").
  void job_done(const planner::ExecutionPlan& plan, const std::string& merge_key, bool cancelled, int arena_creations,
                std::string_view dit_load_source);
  // The pipeline call threw: what is loaded is unknown.
  void job_failed() { warm = false; }
  // A release that was not cancelled: nothing stays on the device; the host copies are the release's.
  void released(bool kept_dit_host_copy, bool kept_te_host_copy);
};

// Weight-file bytes of the models whose loads the plan does not size (the DiT's are the plan's).
struct ModelBytes {
  std::uint64_t text_encoder = 0;
  std::uint64_t vae = 0;  // the VAE file (encoder and decoder): each load is priced at it
};

// The request's passes and adapters as the progress sees them.
struct JobShape {
  int total_steps = 0;           // both passes (pipeline::schedule_steps)
  int first_steps = 0;           // the first pass's; total_steps without a hires fix
  bool hires = false;
  double first_pixel_share = 1;  // hires: the first pass's pixels over the target's
  // An image encode before the DiT load: an init image below strength 1 or with a mask, or control images. (A hires fix
  // encodes in its second pass.)
  bool encodes_image = false;
  std::string merge_key;         // the runtime LoRAs and the control adapter's LoRA part: paths and strengths
};

// Seconds to load each model the plan needs and `held` lacks (0 when it is loaded, or the plan does not use it).
struct LoadEstimates {
  double text_encoder = 0;  // the host or VRAM copy of a resident text encoder (a streamed one is the plan's)
  // The DiT is loaded with this plan's residency and LoRAs (held.warm and both match); else `dit` is its load.
  bool dit_held = false;
  double dit = 0;
  double vae_decoder = 0;
  double vae_encoder = 0;
};
LoadEstimates load_estimates(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw, const HeldModels& held,
                             const ModelBytes& bytes, const JobShape& shape);

// The phases of a generation in the pipeline's order: TEXT_ENCODER, DIT_LOAD (the image encode first), DENOISE, DECODE;
// with a hires fix DENOISE, DECODE (the first pass, for the upscale), DENOISE, DECODE.
std::vector<ProgressSegment> generate_segments(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw,
                                               const HeldModels& held, const ModelBytes& bytes, const JobShape& shape);
// The load steps of a preload in the pipeline's order: TEXT_ENCODER (a host-resident text encoder's copy), DECODE (the
// VAE encoder of a request that encodes an image), DIT_LOAD, DECODE (the VAE decoder); a step the plan skips is left
// out.
std::vector<ProgressSegment> preload_segments(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw,
                                              const HeldModels& held, const ModelBytes& bytes, const JobShape& shape);

// The fraction stays at or below this until the job ends (the end itself is 1).
inline constexpr double kFractionBeforeEnd = 0.999;

class ProgressModel {
 public:
  // The job's expected phases, in order: once, before the first phase starts. Until then the eta is unknown.
  void expect(std::vector<ProgressSegment> segments);
  // A phase starts `t` seconds into the job; the running one has ended. It takes the next expected segment of its kind,
  // looking past optional segments only (they did not run); otherwise it is a phase nothing expected, counted with no
  // estimate, and the expected segments stay pending.
  void phase(mi_phase phase, double t);
  // `step` sampling steps of the job are finished (they count on across passes), `t` seconds into it; `step_seconds`
  // is the last step's own time.
  void step(int step, double step_seconds, double t);
  double fraction() const { return fraction_; }
  double eta_seconds() const { return eta_; }

 private:
  void update(double t);
  double step_price() const;

  std::vector<ProgressSegment> segments_;
  bool expected_ = false;
  std::size_t next_ = 0;  // the first segment not started
  long current_ = -1;     // the running segment (-1: none yet)
  // The running DENOISE pass: job steps finished when it started and now, the last step's time, the first step's own
  // time, and the steps after it with their summed wall time.
  int pass_start_step_ = 0;
  int steps_ = 0;
  double last_step_t_ = 0;
  double first_step_s_ = -1;
  int later_steps_ = 0;
  double later_s_ = 0;
  double scale_ = 1;  // measured / plan step price of the last measured pass (prices the passes not measured yet)
  double fraction_ = 0;
  double eta_ = -1;
};

}  // namespace mi::api
