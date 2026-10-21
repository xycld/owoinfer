#include "api/pipeline_adapter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

#include "backend/cuda_driver.h"
#include "core/cuda_check.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
#include "pipeline/png_writer.h"
#include "pipeline/report_json.h"
#include "pipeline/sampler.h"
#include "planner/planner.h"
#include "util/path.h"

namespace mi::api {

// The CLI's document (pipeline/report_json.h) in braces; the C API writes no latents.
void render_report(const pipeline::Request& q, const pipeline::Result& r, Generation& g) {
  pipeline::ReportParts parts = pipeline::report_parts(q, r, "");
  g.report_head = "{\n" + std::move(parts.head);
  g.report_tail = std::move(parts.tail) + "\n}\n";
}

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------- C structs -> pipeline types

std::vector<int> layer_list(const int32_t* layers, std::size_t count, const char* name) {
  if (count == 0) return {};
  if (layers == nullptr) throw std::invalid_argument(std::string(name) + ": NULL array with a nonzero count.");
  if (count > 1024) throw std::invalid_argument(std::string(name) + ": too many entries.");
  return std::vector<int>(layers, layers + count);
}

pipeline::Request to_request(const mi_gen_params& p) {
  if (p.prompt == nullptr) throw std::invalid_argument("Enter a prompt.");
  pipeline::Request q;
  q.prompt = p.prompt;
  q.negative_prompt = p.negative_prompt != nullptr ? p.negative_prompt : "";
  q.width = p.width;
  q.height = p.height;
  q.steps = p.steps;
  q.cfg = p.cfg;
  q.shift = p.shift;
  q.seed = p.seed;
  q.uncond_skip_layers = layer_list(p.uncond_skip_layers, p.uncond_skip_layers_count, "uncond_skip_layers");
  q.slg_layers = layer_list(p.slg_layers, p.slg_layers_count, "slg_layers");
  q.slg_scale = p.slg_scale;
  q.slg_start = p.slg_start;
  q.slg_stop = p.slg_stop;
  q.max_sequence_length = p.max_sequence_length;
  switch (p.sampler) {
    case MI_SAMPLER_EULER: q.sampler = pipeline::SamplerKind::kEuler; break;
    case MI_SAMPLER_AB2: q.sampler = pipeline::SamplerKind::kAb2; break;
    default: throw std::invalid_argument("sampler: unknown value " + std::to_string(static_cast<int>(p.sampler)) + ".");
  }
  if (p.preview_interval < 1) throw std::invalid_argument("preview_interval must be at least 1.");
  pipeline::validate(q);
  return q;
}

std::size_t budget_bytes(std::uint64_t bytes, const char* name) {
  return pipeline::vram_option_bytes(static_cast<double>(bytes), 1.0, name);
}

pipeline::PipelineOptions to_options(const mi_engine_params& p) {
  pipeline::PipelineOptions o;
  switch (p.mode) {
    case MI_MODE_MIN_MEMORY: o.mode = planner::Mode::kMinMemory; break;
    case MI_MODE_BALANCED: o.mode = planner::Mode::kBalanced; break;
    case MI_MODE_MAX_SPEED: o.mode = planner::Mode::kMaxSpeed; break;
    default: throw std::invalid_argument("mode: unknown value " + std::to_string(static_cast<int>(p.mode)) + ".");
  }
  o.vram_budget = p.vram_budget_bytes != 0 ? budget_bytes(p.vram_budget_bytes, "vram_budget_bytes") : 0;
  o.vram_reserve = p.vram_reserve_bytes != 0 ? budget_bytes(p.vram_reserve_bytes, "vram_reserve_bytes") : 0;
  o.host_ram_budget = p.host_ram_budget_bytes != 0 ? budget_bytes(p.host_ram_budget_bytes, "host_ram_budget_bytes") : 0;
  o.embedding_cache = p.embedding_cache != 0;
  o.direct_io = p.direct_io != 0;
  if (p.cache_dir != nullptr && *p.cache_dir != '\0') o.cache_dir = util::path_from_utf8(p.cache_dir);
  switch (p.block_path) {
    case MI_BLOCK_PATH_DEFAULT: break;  // PipelineOptions' default
    case MI_BLOCK_PATH_OP_BY_OP: o.block_path = moss_dit::BlockPath::kOpByOp; break;
    case MI_BLOCK_PATH_FUSED: o.block_path = moss_dit::BlockPath::kFused; break;
    case MI_BLOCK_PATH_FUSED_GRAPHS: o.block_path = moss_dit::BlockPath::kFusedGraphs; break;
    default: throw std::invalid_argument("block_path: unknown value " + std::to_string(static_cast<int>(p.block_path)) + ".");
  }
  return o;
}

mi_block_path to_block_path(moss_dit::BlockPath path) {
  switch (path) {
    case moss_dit::BlockPath::kOpByOp: return MI_BLOCK_PATH_OP_BY_OP;
    case moss_dit::BlockPath::kFused: return MI_BLOCK_PATH_FUSED;
    case moss_dit::BlockPath::kFusedGraphs: return MI_BLOCK_PATH_FUSED_GRAPHS;
  }
  return MI_BLOCK_PATH_DEFAULT;
}

// The bundle directory `p` names, after the format checks.
fs::path model_dir(const mi_engine_params& p) {
  if (p.model_path == nullptr || *p.model_path == '\0') throw std::invalid_argument("model_path is required.");
  const fs::path path = util::path_from_utf8(p.model_path);
  std::error_code ec;
  const fs::file_status st = fs::status(path, ec);
  if (p.model_format == MI_MODEL_FORMAT_COMFYUI || (p.model_format == MI_MODEL_FORMAT_AUTO && fs::is_regular_file(st))) {
    throw NotSupportedError("ComfyUI model files are not supported yet; pass a Hugging Face bundle directory.");
  }
  if (p.model_format != MI_MODEL_FORMAT_AUTO && p.model_format != MI_MODEL_FORMAT_HF_BUNDLE) {
    throw std::invalid_argument("model_format: unknown value " + std::to_string(static_cast<int>(p.model_format)) + ".");
  }
  if (!fs::is_directory(st)) throw std::invalid_argument("model bundle not found: " + std::string(p.model_path));
  for (const char* component : {"text_encoder", "tokenizer", "transformer", "vae"}) {
    if (!fs::is_directory(path / component, ec)) {
      throw ModelError("not a MOSS-Image bundle (no " + std::string(component) + "/ directory): " + std::string(p.model_path));
    }
  }
  return path;
}

mi_phase to_phase(pipeline::Phase phase) {
  switch (phase) {
    case pipeline::Phase::kTextEncoder: return MI_PHASE_TEXT_ENCODER;
    case pipeline::Phase::kDitLoad: return MI_PHASE_DIT_LOAD;
    case pipeline::Phase::kDenoise: return MI_PHASE_DENOISE;
    case pipeline::Phase::kDecode: return MI_PHASE_DECODE;
  }
  return MI_PHASE_TEXT_ENCODER;
}

}  // namespace

void validate(const mi_gen_params& params) { to_request(params); }

struct PipelineAdapter::Impl {
  fs::path model;
  int device = 0;
  std::unique_ptr<pipeline::Pipeline> pipe;
};

PipelineAdapter::PipelineAdapter(const mi_engine_params& params) : impl_(std::make_unique<Impl>()) {
  impl_->model = model_dir(params);
  const pipeline::PipelineOptions options = to_options(params);
  const backend::ScopedCurrentContext restore;  // the caller's CUDA context is current again on return
  int count = 0;
  MI_CUDA_CHECK(cudaGetDeviceCount(&count));
  if (params.device_index < 0 || params.device_index >= count) {
    throw std::invalid_argument("device_index " + std::to_string(params.device_index) + " out of range (" +
                                std::to_string(count) + " CUDA devices).");
  }
  impl_->device = params.device_index;
  MI_CUDA_CHECK(cudaSetDevice(impl_->device));
  impl_->pipe = std::make_unique<pipeline::Pipeline>(options);
}

// Must not throw (mi_engine_free() has no way to report it): only CUDA C calls whose errors are ignored, and the
// pipeline's destructors release with plain CUDA calls as well.
PipelineAdapter::~PipelineAdapter() {
  // The pipeline frees its device memory on the engine's device, whichever context the caller has current.
  const backend::ScopedCurrentContext restore;
  if (cudaSetDevice(impl_->device) != cudaSuccess) cudaGetLastError();
  impl_->pipe.reset();
}

Generation PipelineAdapter::generate(const mi_gen_params& params, const std::atomic<bool>& cancel) {
  const auto t0 = Clock::now();
  pipeline::Request q = to_request(params);
  q.model = impl_->model;
  // The CUDA device is per host thread; the caller's thread may not be the one that created the engine.
  const backend::ScopedCurrentContext restore;
  MI_CUDA_CHECK(cudaSetDevice(impl_->device));
  const int total_steps = pipeline::moss_schedule(q.steps, q.shift).num_steps();
  const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  const auto report = [&](mi_phase phase, int step, int total, double step_s) {
    if (params.progress_callback == nullptr) return;
    mi_progress p{};
    p.struct_size = sizeof p;
    p.phase = phase;
    p.step = step;
    p.total_steps = total;
    p.step_seconds = step_s;
    p.elapsed_seconds = elapsed();
    params.progress_callback(&p, params.user_data);
  };
  pipeline::GenerateControl control;
  control.cancel = &cancel;
  control.on_phase = [&](pipeline::Phase phase) {
    report(to_phase(phase), 0, total_steps, 0.0);
    return true;
  };
  control.on_step = [&](const pipeline::StepInfo& s) {
    report(MI_PHASE_DENOISE, s.step, s.num_steps, s.step_seconds);
    return true;
  };
  if (params.plan_callback != nullptr) {
    control.on_plan = [&](const planner::ExecutionPlan& plan) {
      params.plan_callback(plan.explanation.c_str(), plan.to_json().c_str(), params.user_data);
    };
  }
  pipeline::Result r = impl_->pipe->generate(q, control);

  Generation g;
  render_report(q, r, g);  // before r.rgb is moved out
  g.width = r.width;
  g.height = r.height;
  g.cancelled = r.cancelled;
  if (!r.cancelled) g.rgb = std::move(r.rgb);
  g.seed = r.seed;
  g.steps_scheduled = r.steps_scheduled;
  g.steps_run = r.steps_run;
  const pipeline::Timings& t = r.timings;
  g.timings.struct_size = sizeof g.timings;
  g.timings.total_s = t.total_s;
  g.timings.load_s = t.index_load_s + t.dit_load_s + t.vae_load_s;
  g.timings.encode_s = t.tokenize_s + t.text_encode_s + t.refine_s;
  g.timings.denoise_s = t.denoise_s;
  g.timings.decode_s = t.decode_s;
  g.timings.index_load_s = t.index_load_s;
  g.timings.tokenize_s = t.tokenize_s;
  g.timings.plan_s = t.plan_s;
  g.timings.text_encode_s = t.text_encode_s;
  g.timings.dit_load_s = t.dit_load_s;
  g.timings.refine_s = t.refine_s;
  g.timings.vae_load_s = t.vae_load_s;
  const pipeline::MemoryStats& m = r.memory;
  g.memory.struct_size = sizeof g.memory;
  g.memory.arena_capacity_bytes = m.arena_capacity;
  g.memory.arena_high_water_bytes = m.arena_high_water;
  g.memory.device_peak_bytes = m.device_peak;
  g.memory.device_free_at_start_bytes = m.free_at_start;
  g.memory.host_private_start_bytes = m.host_private_start;
  g.memory.host_private_peak_bytes = m.host_private_peak;
  g.plan_explanation = r.plan.explanation;
  g.block_path = to_block_path(r.block_path);
  g.block_path_note = r.block_path_note;
  const pipeline::DitStreamStats& ds = r.dit_stream;
  g.dit_stream.struct_size = sizeof g.dit_stream;
  g.dit_stream.streamed = ds.streamed ? 1 : 0;
  g.dit_stream.resident_blocks = ds.resident_blocks;
  g.dit_stream.buffers = ds.buffers;
  g.dit_stream.stalls = ds.stalls;
  g.dit_stream.bytes = ds.bytes;
  g.dit_stream.copy_seconds = ds.copy_s;
  g.dit_stream.stall_seconds = ds.stall_s;
  g.dit_stream.h2d_gbps = ds.gbps();
  g.dit_stream.hidden_fraction = ds.hidden();
  const std::size_t n = std::min(ds.source.size(), sizeof g.dit_stream.source - 1);  // "host_pinned" fits
  std::memcpy(g.dit_stream.source, ds.source.data(), n);
  g.dit_stream.source[n] = '\0';
  return g;
}

std::string report_json(const Generation& generation, const std::string& image_path_utf8) {
  return generation.report_head + pipeline::json_escape(image_path_utf8) + generation.report_tail;
}

void write_png(const Generation& generation, const std::string& path_utf8) {
  if (generation.rgb.empty()) throw std::invalid_argument("the result has no image (the generation was cancelled).");
  pipeline::write_png_rgb8(util::path_from_utf8(path_utf8), generation.rgb, generation.width, generation.height);
}

}  // namespace mi::api
