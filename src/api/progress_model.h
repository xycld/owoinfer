// Whole-job progress for mi_progress.fraction and eta_seconds (MI_CAP_PROGRESS_FRACTION). Pure host logic, no CUDA: the
// adapter turns the plan into the job's expected phases with time estimates (generate_segments, preload_segments), then
// feeds ProgressModel the phase starts and sampling steps as they happen.
//
// Estimates: the plan's TimeEstimate is the steady state (models loaded as the plan keeps them; dit_load_s covers only
// parking reloads). A model that is not loaded yet adds its load: its bytes at the machine's transfer rates
// (planner::HardwareProfile). The pipeline has no live residency accessor, so the adapter keeps HeldModels from the
// earlier jobs' results and releases; the load estimates are rough, and the measurements replace them as the job runs.
//
// Correction: a phase that ended counts at its measured time (everything before the latest event is done), and the
// remaining sampling steps of the running pass are priced at its measured step time. The first step of a pass also
// carries the pass's setup (text refiner, CUDA graph capture, lazily loaded kernels), so once a later step is measured
// the steps after the first set the price; until then the first step's own time does.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mossinfer/mossinfer.h"
#include "planner/planner.h"

namespace mi::api {

// One expected phase of a job. A DENOISE segment is one sampling pass: `seconds` its work before the first step (an
// image encode's VAE encoder load), then `steps` steps of `step_seconds` each.
struct ProgressSegment {
  mi_phase phase = MI_PHASE_TEXT_ENCODER;
  double seconds = 0;
  int steps = 0;
  double step_seconds = 0;
};

// What the engine holds when a job starts, as far as the adapter can tell from the earlier jobs' results.
struct HeldModels {
  // The device models of the last finished job are loaded: the DiT with `dit`'s residency, the VAE decoder when it ran
  // (`vae`), the VAE encoder when it encoded (`encoder`), in an arena of `arena_bytes`.
  bool loaded = false;
  planner::DitPlan dit;
  bool vae = false;
  bool encoder = false;
  std::size_t arena_bytes = 0;
  bool te_vram = false;           // a VRAM-resident text encoder (in the arena: kept with it)
  bool te_host_copy = false;      // a host-resident text encoder's copy (host memory: survives the arena)
  bool dit_host_copy = false;     // a soft unload kept the DiT's weight files in pinned RAM (the next load reads it)
  bool dit_files_read = false;    // this engine read the DiT's files before: the page cache likely holds them
};

// Weight-file bytes of the models whose loads the plan does not size (the DiT's are the plan's).
struct ModelBytes {
  std::uint64_t text_encoder = 0;
  std::uint64_t vae = 0;  // the VAE file (encoder and decoder): each load is priced at it
};

// The request's passes as the progress sees them.
struct JobShape {
  int total_steps = 0;           // both passes (pipeline::schedule_steps)
  int first_steps = 0;           // the first pass's; total_steps without a hires fix
  bool hires = false;
  double first_pixel_share = 1;  // hires: the first pass's pixels over the target's (its decode's share)
  bool encodes_image = false;    // an init image or control images: the VAE encoder runs before the DiT load
};

// Seconds to load each model the plan needs and `held` lacks (0 when it is loaded, or the plan does not use it).
struct LoadEstimates {
  double text_encoder = 0;  // the host or VRAM copy of a resident text encoder (a streamed one is the plan's)
  double dit = 0;
  double vae_decoder = 0;
  double vae_encoder = 0;
};
LoadEstimates load_estimates(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw, const HeldModels& held,
                             const ModelBytes& bytes, const JobShape& shape);

// The phases of a generation in the pipeline's order: TEXT_ENCODER, DIT_LOAD (the init image's encode first), DENOISE,
// DECODE; with a hires fix DENOISE, DECODE (the first pass, for the upscale), DENOISE, DECODE.
std::vector<ProgressSegment> generate_segments(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw,
                                               const HeldModels& held, const ModelBytes& bytes, const JobShape& shape);
// The load steps of a preload in the pipeline's order: TEXT_ENCODER (a host-resident text encoder's copy), DECODE (the
// VAE encoder of an image request), DIT_LOAD, DECODE (the VAE decoder); a step the plan skips is left out.
std::vector<ProgressSegment> preload_segments(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw,
                                              const HeldModels& held, const ModelBytes& bytes, const JobShape& shape);

// The fraction stays at or below this until the job ends (the end itself is 1).
inline constexpr double kFractionBeforeEnd = 0.999;

class ProgressModel {
 public:
  // The job's expected phases, in order: once, before the first phase starts. Until then the eta is unknown.
  void expect(std::vector<ProgressSegment> segments);
  // A phase starts `t` seconds into the job; the running one has ended. It takes the next expected segment of its kind
  // (the expected ones before it did not run); a phase not expected counts with no estimate.
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
  double fraction_ = 0;
  double eta_ = -1;
};

}  // namespace mi::api
