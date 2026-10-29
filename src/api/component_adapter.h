// The component API's contact with pipeline::Session (mossinfer.h "components", design §3-§5): converts the component
// structs into the session's types and back, and holds an engine's session. The conversions are CPU only and public
// for the CPU tests; ComponentSession runs the calls on the engine's device.
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "api/pipeline_adapter.h"
#include "core/tensor.h"
#include "mossinfer/mossinfer.h"
#include "pipeline/session.h"

// One conditioning (mossinfer.h): immutable, shared by the runs that use it.
struct mi_conditioning {
  std::shared_ptr<const mi::pipeline::Conditioning> c;
};

namespace mi::runtime {
class JobGate;
}  // namespace mi::runtime

namespace mi::api {

// A tensor descriptor as a Tensor view: rank, shape, enums and the reserved field checked, `what` naming it. data NULL
// is an absent tensor (an empty Tensor). Throws std::invalid_argument.
Tensor to_tensor(const mi_tensor& t, const char* what);
// A view's descriptor (data, memory kind, dtype, shape).
mi_tensor from_tensor(const Tensor& t);

// A foreign conditioning, copied from `embeddings` ([S, MI_TEXT_DIM], 1 <= S <= 4096, F32 rounded to bf16 to nearest
// even, or BF16; host or device memory complete before the call). Throws std::invalid_argument.
std::shared_ptr<const pipeline::Conditioning> to_conditioning(const mi_tensor& embeddings);

// The session's options from `params` (a full-size struct), with its plan request (PipelineAdapter::session_request,
// or the defaults') and the engine's device. Throws std::invalid_argument.
pipeline::SessionOptions to_session_options(const mi_session_params& params, pipeline::Request plan_request, int device);

// A run's spec from `params` (a full-size struct) for a session of `options`: the component fields checked here, the
// fields mi_gen_params shares (sizes, steps, schedule fields, LoRAs, control images, accelerators) read and checked as
// mi_generate() reads them, with its messages, then Session::validate. Throws std::invalid_argument; an unreadable
// control image file is pipeline::ImageReadError.
pipeline::RunSpec to_run_spec(const mi_run_params& params, const pipeline::SessionOptions& options);

// One call's arguments from `params` (a full-size struct). Throws std::invalid_argument.
pipeline::StepArgs to_step_args(const mi_step_params& params);

mi_run_info to_run_info(const pipeline::RunInfo& info);
mi_step_info to_step_info(const pipeline::StepReport& report);

// A finished run as mi_run_end() returns it: no image; the report renders the run as its equivalent request with a
// "component" object ({"guidance", "schedule_matches", "off_schedule_calls", "foreign_rows", "calls"}); the note is
// MI_TEXT_COMPONENT_NOTE. `control_adapter`: the session's.
Generation to_generation(pipeline::RunResult& run, const std::filesystem::path& control_adapter);

// An engine's open session (mi_session_open .. mi_session_close). Every call selects the engine's device and restores
// the caller's CUDA context; the session's own gate rules apply (pipeline/session.h).
class ComponentSession {
 public:
  // Opens the session: a job of `gate`. Throws as Session's constructor (JobGate::Busy, std::invalid_argument).
  ComponentSession(PipelineAdapter& adapter, runtime::JobGate& gate, const mi_session_params& params);
  ~ComponentSession();
  ComponentSession(const ComponentSession&) = delete;
  ComponentSession& operator=(const ComponentSession&) = delete;

  bool close();
  std::vector<std::shared_ptr<const pipeline::Conditioning>> encode_text(const mi_text_params& params);
  void begin_run(const mi_run_params& params);
  mi_run_info run_info() const;
  const std::vector<float>& run_sigmas() const;
  pipeline::StagingViews staging() const;
  void adopt_staging(void* base, std::size_t bytes);
  mi_step_info step(const mi_step_params& params);
  std::optional<Generation> end_run();
  void vae_encode(const mi_image& image, const mi_tensor& latents);
  void vae_decode(const mi_tensor& latents, const mi_tensor& rgb);

 private:
  PipelineAdapter& adapter_;
  pipeline::SessionOptions options_;  // a copy for to_run_spec (the session keeps its own)
  std::unique_ptr<pipeline::Session> session_;
};

}  // namespace mi::api
