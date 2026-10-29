#include "api/component_adapter.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "backend/cuda_driver.h"
#include "core/cuda_check.h"
#include "models/control/control.h"
#include "models/lora/lora.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/report_json.h"
#include "pipeline/run_schedule.h"
#include "pipeline/sampler.h"
#include "runtime/job_gate.h"
#include "util/path.h"
#include "util/regular_file.h"

namespace mi::api {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

[[noreturn]] void unknown(const std::string& field, int value) {
  throw std::invalid_argument(field + ": unknown value " + std::to_string(value) + ".");
}

void check_reserved(std::int32_t value, const std::string& name) {
  if (value != 0) throw std::invalid_argument(name + " must be 0.");
}

std::vector<int> layer_list(const int32_t* layers, std::size_t count, const std::string& name) {
  if (count == 0) return {};
  if (layers == nullptr) throw std::invalid_argument(name + ": NULL array with a nonzero count.");
  if (count > 1024) throw std::invalid_argument(name + ": too many entries.");
  return std::vector<int>(layers, layers + count);
}

std::int32_t count32(std::int64_t n) {
  return static_cast<std::int32_t>(std::min<std::int64_t>(n, std::numeric_limits<std::int32_t>::max()));
}

const char* guidance_name(pipeline::GuidanceMode g) { return g == pipeline::GuidanceMode::kMoss ? "moss" : "host"; }

// The start of the session call running on this thread: its progress callbacks' elapsed time.
thread_local Clock::time_point t_call_start;

// One component call that runs on the device: the engine's device selected, the caller's CUDA context restored on
// return.
struct DeviceCall {
  backend::ScopedCurrentContext restore;
  explicit DeviceCall(int device) {
    t_call_start = Clock::now();
    MI_CUDA_CHECK(cudaSetDevice(device));
  }
};

// A call that may load or drop models: the adapter's progress record goes cold (before, and after: a generate may
// start between the two); model-file errors are ModelError (MI_ERR_MODEL), as for mi_generate().
template <class F>
auto loading(PipelineAdapter& adapter, F&& call) {
  adapter.models_changed();
  struct After {
    PipelineAdapter& a;
    ~After() { a.models_changed(); }
  } after{adapter};
  try {
    return call();
  } catch (const lora::LoraError& e) {
    throw ModelError(e.what());
  } catch (const control::ControlError& e) {
    throw ModelError(e.what());
  }
}

}  // namespace

Tensor to_tensor(const mi_tensor& t, const char* what) {
  if (t.data == nullptr) return Tensor{};
  const std::string n(what);
  check_reserved(t._reserved0, n + "._reserved0");
  DType dtype = DType::kF32;
  switch (t.dtype) {
    case MI_DTYPE_F32: dtype = DType::kF32; break;
    case MI_DTYPE_BF16: dtype = DType::kBF16; break;
    case MI_DTYPE_U8: dtype = DType::kU8; break;
    default: unknown(n + ".dtype", static_cast<int>(t.dtype));
  }
  Device device = Device::kCPU;
  switch (t.memory) {
    case MI_MEMORY_HOST: device = Device::kCPU; break;
    case MI_MEMORY_DEVICE: device = Device::kCUDA; break;
    default: unknown(n + ".memory", static_cast<int>(t.memory));
  }
  if (t.rank < 1 || t.rank > 4) {
    throw std::invalid_argument(n + ".rank must be 1 to 4, not " + std::to_string(t.rank) + ".");
  }
  std::int64_t numel = 1;
  for (int i = 0; i < 4; ++i) {
    const std::int64_t d = t.shape[i];
    if (i < t.rank && (d < 1 || d > (std::int64_t{1} << 32))) {
      throw std::invalid_argument(n + ".shape[" + std::to_string(i) + "] must be positive, not " + std::to_string(d) +
                                  ".");
    }
    if (i >= t.rank && d != 0) {
      throw std::invalid_argument(n + ".shape[" + std::to_string(i) + "] must be 0 past the rank.");
    }
    if (i < t.rank) numel *= d;
    if (numel > (std::int64_t{1} << 40)) throw std::invalid_argument(n + ": too many elements.");
  }
  return Tensor::view(t.data, dtype, device, t.shape, t.rank);
}

mi_tensor from_tensor(const Tensor& t) {
  mi_tensor out{};
  if (t.data == nullptr) return out;
  out.data = t.data;
  out.memory = t.device == Device::kCUDA ? MI_MEMORY_DEVICE : MI_MEMORY_HOST;
  out.dtype = t.dtype == DType::kBF16 ? MI_DTYPE_BF16 : t.dtype == DType::kU8 ? MI_DTYPE_U8 : MI_DTYPE_F32;
  out.rank = t.rank;
  for (int i = 0; i < t.rank && i < 4; ++i) out.shape[i] = t.shape[static_cast<std::size_t>(i)];
  return out;
}

std::shared_ptr<const pipeline::Conditioning> to_conditioning(const mi_tensor& embeddings) {
  const Tensor e = to_tensor(embeddings, "embeddings");
  if (e.data == nullptr) throw std::invalid_argument("embeddings.data is NULL.");
  if (e.rank != 2 || e.dim(1) != MI_TEXT_DIM || e.dim(0) < 1 || e.dim(0) > 4096 ||
      (e.dtype != DType::kF32 && e.dtype != DType::kBF16)) {
    throw std::invalid_argument("embeddings: expected f32 or bf16 [1..4096, " + std::to_string(MI_TEXT_DIM) +
                                "], got " + e.describe() + ".");
  }
  std::vector<std::byte> host(e.nbytes());
  if (e.device == Device::kCUDA) {
    MI_CUDA_CHECK(cudaMemcpy(host.data(), e.data, host.size(), cudaMemcpyDeviceToHost));
  } else {
    std::memcpy(host.data(), e.data, host.size());
  }
  auto emb = std::make_shared<pipeline::Embedding>();
  emb->hidden = MI_TEXT_DIM;
  emb->data.resize(static_cast<std::size_t>(e.numel()));
  if (e.dtype == DType::kBF16) {
    std::memcpy(emb->data.data(), host.data(), host.size());
  } else {
    const auto* f = reinterpret_cast<const float*>(host.data());
    for (std::size_t i = 0; i < emb->data.size(); ++i) {
      const __nv_bfloat16 b = __float2bfloat16_rn(f[i]);  // round to nearest even
      std::memcpy(&emb->data[i], &b, sizeof b);
    }
  }
  auto c = std::make_shared<pipeline::Conditioning>();
  c->embedding = std::move(emb);
  c->engine_encoded = false;
  return c;
}

pipeline::SessionOptions to_session_options(const mi_session_params& p, pipeline::Request plan_request, int device) {
  pipeline::SessionOptions o;
  o.plan_request = std::move(plan_request);
  o.max_rows = p.max_rows;
  o.max_latents = p.max_latents;
  if (p.control_adapter != nullptr && *p.control_adapter != '\0') {
    o.control_adapter = util::path_from_utf8(p.control_adapter);
    util::check_user_path(o.control_adapter, "control_adapter");
    std::error_code ec;
    const fs::file_status st = fs::status(o.control_adapter, ec);
    if (!fs::exists(st)) {
      throw std::invalid_argument(std::string("control_adapter: file not found: ") + p.control_adapter);
    }
    if (!fs::is_regular_file(st)) {
      throw std::invalid_argument(std::string("control_adapter: not a regular file: ") + p.control_adapter);
    }
  }
  o.device = device;
  return o;
}

pipeline::RunSpec to_run_spec(const mi_run_params& p, const pipeline::SessionOptions& options) {
  check_reserved(p._reserved0, "mi_run_params._reserved0");
  pipeline::RunSpec s;
  switch (p.guidance) {
    case MI_GUIDANCE_HOST: s.guidance = pipeline::GuidanceMode::kHost; break;
    case MI_GUIDANCE_MOSS: s.guidance = pipeline::GuidanceMode::kMoss; break;
    default: unknown("guidance", static_cast<int>(p.guidance));
  }
  switch (p.output) {
    case MI_STEP_OUTPUT_ROWS: s.output = pipeline::StepOutput::kRows; break;
    case MI_STEP_OUTPUT_GUIDED_VELOCITY: s.output = pipeline::StepOutput::kGuidedVelocity; break;
    case MI_STEP_OUTPUT_GUIDED_DENOISED: s.output = pipeline::StepOutput::kGuidedDenoised; break;
    default: unknown("output", static_cast<int>(p.output));
  }
  if ((p.flags & ~(MI_RUN_EXACT_SIGMA | MI_RUN_CROPS)) != 0) {
    throw std::invalid_argument("flags: unknown bits " +
                                std::to_string(p.flags & ~(MI_RUN_EXACT_SIGMA | MI_RUN_CROPS)) + ".");
  }
  s.exact_sigma = (p.flags & MI_RUN_EXACT_SIGMA) != 0;
  s.crops = (p.flags & MI_RUN_CROPS) != 0;

  // Rows: their conditionings are kept alive by the run.
  if (p.rows_count > 0 && p.rows == nullptr) throw std::invalid_argument("rows: NULL array with a nonzero count.");
  if (p.rows_count < 1 || p.rows_count > static_cast<std::size_t>(options.max_rows)) {
    throw std::invalid_argument("A run takes 1 to max_rows (" + std::to_string(options.max_rows) + ") rows, not " +
                                std::to_string(p.rows_count) + ".");
  }
  for (std::size_t k = 0; k < p.rows_count; ++k) {
    const mi_run_row& r = p.rows[k];
    const std::string name = "rows[" + std::to_string(k) + "]";
    check_reserved(r._reserved0, name + "._reserved0");
    if (r.conditioning == nullptr) throw std::invalid_argument(name + ".conditioning is NULL.");
    s.rows.push_back(
        {r.conditioning->c, layer_list(r.skip_layers, r.skip_layers_count, name + ".skip_layers"), r.latent});
  }

  // The schedule.
  if (p.sigmas == nullptr && p.sigmas_count > 0) {
    throw std::invalid_argument("sigmas: NULL array with a nonzero count.");
  }
  if (p.sigmas != nullptr && p.sigmas_count == 0) throw std::invalid_argument("sigmas: an array with a count of 0.");
  if (p.sigmas_count > pipeline::kMaxRunSigmas) {
    throw std::invalid_argument("sigmas: at most " + std::to_string(pipeline::kMaxRunSigmas) + " values, not " +
                                std::to_string(p.sigmas_count) + ".");
  }
  if (p.sigmas != nullptr) s.sigmas.assign(p.sigmas, p.sigmas + p.sigmas_count);
  if (p.steps < 1) {
    throw std::invalid_argument("Steps must be between 1 and " + std::to_string(pipeline::kMaxDenoiseSteps) + ".");
  }
  if (!std::isfinite(p.denoise_strength) || p.denoise_strength < 0 || p.denoise_strength > 1) {
    throw std::invalid_argument("Denoise strength must be between 0 and 1.");
  }
  if (!std::isfinite(p.cfg) || p.cfg < 0) throw std::invalid_argument("CFG must be a finite number >= 0.");
  if (!std::isfinite(p.step_cache_threshold) || p.step_cache_threshold < 0 || p.step_cache_threshold > 1) {
    throw std::invalid_argument("step_cache_threshold must be in (0, 1] with the step cache on (0 when it is off).");
  }

  // The fields mi_gen_params shares, read and checked as mi_generate() reads them.
  mi_gen_params g;
  if (mi_gen_params_init(&g, sizeof g) != MI_OK) throw std::logic_error("mi_gen_params_init failed");
  const std::string adapter = util::path_utf8(options.control_adapter);
  g.prompt = "component";  // the rows carry the conditionings
  g.width = p.width;
  g.height = p.height;
  g.steps = p.steps;
  g.cfg = p.cfg;
  g.shift = p.shift;
  g.flow_multiplier = p.flow_multiplier;
  g.scheduler = p.scheduler;
  g.uncond_skip_layers = nullptr;
  g.uncond_skip_layers_count = 0;
  g.slg_layers = p.slg_layers;
  g.slg_layers_count = p.slg_layers_count;
  g.slg_scale = p.slg_scale;
  g.slg_start = p.slg_start;
  g.slg_stop = p.slg_stop;
  g.loras = p.loras;
  g.loras_count = p.loras_count;
  g.control_adapter = adapter.empty() ? nullptr : adapter.c_str();
  g.control_lora_strength = p.control_lora_strength;
  g.control_images = p.control_images;
  g.control_images_count = p.control_images_count;
  g.resize = p.control_resize;
  g.cfg_truncation = p.cfg_truncation;
  g.step_cache = p.step_cache;
  g.step_cache_threshold = p.step_cache_threshold;
  const pipeline::Request q = to_request(g, nullptr, true, false);

  s.width = q.width;
  s.height = q.height;
  s.scheduler = q.scheduler;
  s.steps = q.steps;
  s.shift = q.shift;
  s.flow_multiplier = q.flow_multiplier;
  s.denoise_strength = p.denoise_strength;
  s.cfg = q.cfg;
  s.slg_layers = q.slg_layers;
  s.slg_scale = q.slg_scale;
  s.slg_start = q.slg_start;
  s.slg_stop = q.slg_stop;
  s.cfg_truncation = q.accel.cfg_truncation;
  s.loras = q.loras;
  s.control_lora_strength = q.control.lora_strength;
  s.control_images = q.control.images;
  s.control_resize = q.resize;
  s.step_cache = q.accel.step_cache;
  s.step_cache_threshold = p.step_cache_threshold;
  pipeline::Session::validate(s, options);
  return s;
}

pipeline::StepArgs to_step_args(const mi_step_params& p) {
  if ((p.flags & ~MI_STEP_CONTROL_CHANGED) != 0) {
    throw std::invalid_argument("flags: unknown bits " + std::to_string(p.flags & ~MI_STEP_CONTROL_CHANGED) + ".");
  }
  pipeline::StepArgs a;
  a.sigma = p.sigma;
  a.step = p.step;
  a.rows = p.rows;
  a.control_changed = (p.flags & MI_STEP_CONTROL_CHANGED) != 0;
  a.stream = static_cast<cudaStream_t>(p.cuda_stream);
  a.control = to_tensor(p.control, "control");
  a.x = to_tensor(p.x, "x");
  for (int k = 0; k < MI_MAX_ROWS; ++k) {
    const std::string name = "out[" + std::to_string(k) + "]";
    a.out[static_cast<std::size_t>(k)] = to_tensor(p.out[k], name.c_str());
  }
  return a;
}

mi_run_info to_run_info(const pipeline::RunInfo& i) {
  mi_run_info o{};
  o.struct_size = sizeof o;
  o.latent_channels = i.latent_channels;
  o.latent_height = i.latent_height;
  o.latent_width = i.latent_width;
  o.steps = i.steps;
  o.outputs = i.outputs;
  o.rows = i.rows;
  o.schedule_matches = i.schedule_matches ? 1 : 0;
  o.off_schedule_calls = count32(i.off_schedule_calls);
  o.calls = count32(i.calls);
  o.control_channels = i.control_channels;
  o.latents = i.latents;
  o.staging_bytes = i.staging_bytes;
  return o;
}

mi_step_info to_step_info(const pipeline::StepReport& r) {
  mi_step_info o{};
  o.struct_size = sizeof o;
  o.step = r.step;
  o.rows = r.rows;
  o.cached = r.cached ? 1 : 0;
  o.on_schedule = r.on_schedule ? 1 : 0;
  o.seconds = r.seconds;
  return o;
}

Generation to_generation(pipeline::RunResult& run, const std::filesystem::path& control_adapter) {
  const pipeline::RunSpec& s = run.spec;
  const bool moss = s.guidance == pipeline::GuidanceMode::kMoss;
  // The run as its equivalent mi_generate request (sampler euler, no prompt, no image), for the report.
  pipeline::Request q;
  q.prompt.clear();
  q.width = s.width;
  q.height = s.height;
  q.steps = s.steps;
  q.cfg = moss && s.rows.size() == 2 ? s.cfg : 1.0;
  q.shift = s.shift;
  q.flow_multiplier = s.flow_multiplier;
  q.scheduler = s.scheduler;
  q.sampler = pipeline::SamplerKind::kEuler;
  q.uncond_skip_layers = moss && s.rows.size() == 2 ? s.rows[1].skip_layers : std::vector<int>{};
  q.slg_layers = s.slg_layers;
  q.slg_scale = s.slg_scale;
  q.slg_start = s.slg_start;
  q.slg_stop = s.slg_stop;
  q.loras = s.loras;
  q.control.adapter = control_adapter;
  q.control.lora_strength = s.control_lora_strength;
  q.control.images = s.control_images;
  q.resize = s.control_resize;
  q.accel.cfg_truncation = s.cfg_truncation;
  q.accel.step_cache = s.step_cache;
  if (s.step_cache != pipeline::accel::StepCacheKind::kOff) q.accel.step_cache_threshold = s.step_cache_threshold;
  q.save_metadata = false;
  Generation g = to_generation(q, run.result, false);
  g.component_note = run.note;
  std::string rows = "[";
  for (std::size_t i = 0; i < run.foreign_rows.size(); ++i) {
    rows += (i ? ", " : "") + std::to_string(run.foreign_rows[i]);
  }
  rows += "]";
  const std::string component = ",\n \"component\": {\"guidance\": \"" + std::string(guidance_name(run.guidance)) +
                                "\", \"schedule_matches\": " + (run.schedule_matches ? "true" : "false") +
                                ", \"off_schedule_calls\": " + std::to_string(run.off_schedule_calls) +
                                ", \"foreign_rows\": " + rows + ", \"calls\": " + std::to_string(run.calls) +
                                ", \"note\": " + pipeline::json_escape(run.note) + "}";
  const std::string end = "\n}\n";  // render_report's closing brace
  if (g.report_tail.size() >= end.size() &&
      g.report_tail.compare(g.report_tail.size() - end.size(), end.size(), end) == 0) {
    g.report_tail.insert(g.report_tail.size() - end.size(), component);
  }
  return g;
}

ComponentSession::ComponentSession(PipelineAdapter& adapter, runtime::JobGate& gate, const mi_session_params& params)
    : adapter_(adapter) {
  pipeline::Request plan;
  if (params.plan_request != nullptr) {
    plan = adapter.session_request(*params.plan_request);
  } else {
    mi_gen_params g;
    if (mi_gen_params_init(&g, sizeof g) != MI_OK) throw std::logic_error("mi_gen_params_init failed");
    plan = adapter.session_request(g);
  }
  options_ = to_session_options(params, std::move(plan), adapter.device());
  // The phases a call enters, as progress callbacks (step 0; the fraction is not estimated for component calls).
  if (params.progress_callback != nullptr) {
    const mi_progress_fn fn = params.progress_callback;
    void* const user = params.user_data;
    options_.on_phase = [fn, user](pipeline::Phase phase) {
      mi_progress p{};
      p.struct_size = sizeof p;
      p.phase = to_phase(phase);
      p.elapsed_seconds = std::chrono::duration<double>(Clock::now() - t_call_start).count();
      p.eta_seconds = -1;
      fn(&p, user);
      return true;
    };
  }
  const DeviceCall call(adapter.device());
  session_ = std::make_unique<pipeline::Session>(adapter.pipeline(), gate, options_);
}

ComponentSession::~ComponentSession() = default;

bool ComponentSession::close() {
  const DeviceCall call(adapter_.device());
  return session_->close();
}

std::vector<std::shared_ptr<const pipeline::Conditioning>> ComponentSession::encode_text(const mi_text_params& p) {
  check_reserved(p._reserved0, "mi_text_params._reserved0");
  if (p.prompts_count > 0 && p.prompts == nullptr) {
    throw std::invalid_argument("prompts: NULL array with a nonzero count.");
  }
  if (p.prompts_count < 1 || p.prompts_count > 8) throw std::invalid_argument("mi_encode_text takes 1 to 8 prompts.");
  std::vector<std::string> prompts;
  for (std::size_t i = 0; i < p.prompts_count; ++i) {
    if (p.prompts[i] == nullptr) throw std::invalid_argument("prompts[" + std::to_string(i) + "] is NULL.");
    prompts.emplace_back(p.prompts[i]);
  }
  const DeviceCall call(adapter_.device());
  return loading(adapter_, [&] { return session_->encode_texts(prompts, p.max_sequence_length); });
}

void ComponentSession::begin_run(const mi_run_params& p) {
  const pipeline::RunSpec spec = to_run_spec(p, options_);
  const DeviceCall call(adapter_.device());
  loading(adapter_, [&] {
    session_->begin_run(spec);
    return 0;
  });
}

mi_run_info ComponentSession::run_info() const { return to_run_info(session_->run_info()); }

const std::vector<float>& ComponentSession::run_sigmas() const { return session_->run_sigmas(); }

pipeline::StagingViews ComponentSession::staging() const { return session_->staging(); }

void ComponentSession::adopt_staging(void* base, std::size_t bytes) {
  const DeviceCall call(adapter_.device());
  session_->adopt_staging(base, bytes);
}

mi_step_info ComponentSession::step(const mi_step_params& p) {
  const pipeline::StepArgs args = to_step_args(p);
  const DeviceCall call(adapter_.device());
  return to_step_info(session_->step(args));
}

std::optional<Generation> ComponentSession::end_run() {
  const DeviceCall call(adapter_.device());
  std::optional<pipeline::RunResult> r = session_->end_run();
  if (!r) return std::nullopt;
  return to_generation(*r, options_.control_adapter);
}

void ComponentSession::vae_encode(const mi_image& image, const mi_tensor& latents) {
  check_reserved(image._reserved0, "image._reserved0");
  if (image.pixels == nullptr) throw std::invalid_argument("image.pixels is NULL.");
  if (image.channels != 3) {
    throw std::invalid_argument("image.channels must be 3 (RGB), not " + std::to_string(image.channels) + ".");
  }
  if (image.width < 1 || image.height < 1 || image.width > pipeline::kMaxSide || image.height > pipeline::kMaxSide) {
    throw std::invalid_argument("image size " + std::to_string(image.width) + "x" + std::to_string(image.height) +
                                ": each side must be 1 to " + std::to_string(pipeline::kMaxSide) + ".");
  }
  pipeline::Image im;
  im.width = image.width;
  im.height = image.height;
  im.channels = 3;
  im.pixels.assign(image.pixels, image.pixels + static_cast<std::size_t>(image.width) * image.height * 3);
  const Tensor out = to_tensor(latents, "out_latents");
  const DeviceCall call(adapter_.device());
  loading(adapter_, [&] {
    session_->vae_encode(im, out);
    return 0;
  });
}

void ComponentSession::vae_decode(const mi_tensor& latents, const mi_tensor& rgb) {
  const Tensor in = to_tensor(latents, "latents");
  const Tensor out = to_tensor(rgb, "out_rgb");
  if (in.data == nullptr) throw std::invalid_argument("latents.data is NULL.");
  const DeviceCall call(adapter_.device());
  loading(adapter_, [&] {
    session_->vae_decode(in, out);
    return 0;
  });
}

}  // namespace mi::api
