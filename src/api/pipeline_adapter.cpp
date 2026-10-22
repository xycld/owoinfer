#include "api/pipeline_adapter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <system_error>
#include <utility>

#include "backend/cuda_driver.h"
#include "core/cuda_check.h"
#include "models/lora/lora.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
#include "pipeline/png_writer.h"
#include "pipeline/quality_cert.h"
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

[[noreturn]] void unknown(const char* field, int value) {
  throw std::invalid_argument(std::string(field) + ": unknown value " + std::to_string(value) + ".");
}

std::vector<int> layer_list(const int32_t* layers, std::size_t count, const char* name) {
  if (count == 0) return {};
  if (layers == nullptr) throw std::invalid_argument(std::string(name) + ": NULL array with a nonzero count.");
  if (count > 1024) throw std::invalid_argument(std::string(name) + ": too many entries.");
  return std::vector<int>(layers, layers + count);
}

// A UTF-8 path that must name an existing regular file.
fs::path existing_file(const char* utf8, const std::string& name) {
  const fs::path path = util::path_from_utf8(utf8);
  std::error_code ec;
  const fs::file_status st = fs::status(path, ec);
  if (!fs::exists(st)) throw std::invalid_argument(name + ": file not found: " + utf8);
  if (!fs::is_regular_file(st)) throw std::invalid_argument(name + ": not a regular file: " + utf8);
  return path;
}

pipeline::SamplerKind to_sampler(mi_sampler s) {
  using K = pipeline::SamplerKind;
  switch (s) {
    case MI_SAMPLER_EULER: return K::kEuler;
    case MI_SAMPLER_AB2: return K::kAb2;
    case MI_SAMPLER_EULER_ANCESTRAL: return K::kEulerAncestral;
    case MI_SAMPLER_HEUN: return K::kHeun;
    case MI_SAMPLER_DPMPP_2M: return K::kDpmpp2m;
    case MI_SAMPLER_DPMPP_2M_SDE: return K::kDpmpp2mSde;
    case MI_SAMPLER_ER_SDE: return K::kErSde;
    case MI_SAMPLER_RES_MULTISTEP: return K::kResMultistep;
    case MI_SAMPLER_DATASPACE: return K::kDataSpace;
    case MI_SAMPLER_MAX_ENUM: break;
  }
  unknown("sampler", static_cast<int>(s));
}

pipeline::SchedulerKind to_scheduler(mi_scheduler s) {
  using K = pipeline::SchedulerKind;
  switch (s) {
    case MI_SCHEDULER_MOSS: return K::kMoss;
    case MI_SCHEDULER_SIMPLE: return K::kSimple;
    case MI_SCHEDULER_NORMAL: return K::kNormal;
    case MI_SCHEDULER_KARRAS: return K::kKarras;
    case MI_SCHEDULER_EXPONENTIAL: return K::kExponential;
    case MI_SCHEDULER_BETA: return K::kBeta;
    case MI_SCHEDULER_SGM_UNIFORM: return K::kSgmUniform;
    case MI_SCHEDULER_MAX_ENUM: break;
  }
  unknown("scheduler", static_cast<int>(s));
}

pipeline::NoiseLayout to_noise_layout(mi_noise_layout l) {
  switch (l) {
    case MI_NOISE_LAYOUT_OFFICIAL: return pipeline::NoiseLayout::kOfficial;
    case MI_NOISE_LAYOUT_BATCH_STREAM: return pipeline::NoiseLayout::kBatchStream;
    case MI_NOISE_LAYOUT_MAX_ENUM: break;
  }
  unknown("noise_layout", static_cast<int>(l));
}

pipeline::NoiseSource to_noise_source(mi_noise_source s) {
  switch (s) {
    case MI_NOISE_SOURCE_PHILOX: return pipeline::NoiseSource::kPhilox;
    case MI_NOISE_SOURCE_PHILOX_DEVICE: return pipeline::NoiseSource::kPhiloxDevice;
    case MI_NOISE_SOURCE_BROWNIAN_TREE: return pipeline::NoiseSource::kBrownianTree;
    case MI_NOISE_SOURCE_MAX_ENUM: break;
  }
  unknown("noise_source", static_cast<int>(s));
}

planner::Tier to_tier(mi_tier t) {
  switch (t) {
    case MI_TIER_AUTO: return planner::Tier::kAuto;
    case MI_TIER_REFERENCE: return planner::Tier::kReference;
    case MI_TIER_HIGH: return planner::Tier::kHigh;
    case MI_TIER_TURBO: return planner::Tier::kTurbo;
    case MI_TIER_MAX_ENUM: break;
  }
  unknown("tier", static_cast<int>(t));
}

std::optional<planner::AttentionMode> to_attention(mi_attention a) {
  using A = planner::AttentionMode;
  switch (a) {
    case MI_ATTENTION_DEFAULT: return std::nullopt;  // the tier's choice
    case MI_ATTENTION_EXACT: return A::kExact;
    case MI_ATTENTION_Q8_FP16: return A::kQ8Fp16;
    case MI_ATTENTION_Q8_FP16_F32ACC: return A::kQ8Fp16F32Acc;
    case MI_ATTENTION_Q8_FP8: return A::kQ8Fp8;
    case MI_ATTENTION_Q8_FP8_F32ACC: return A::kQ8Fp8F32Acc;
    case MI_ATTENTION_MAX_ENUM: break;
  }
  unknown("attention", static_cast<int>(a));
}

std::size_t budget_bytes(std::uint64_t bytes, const char* name) {
  return pipeline::vram_option_bytes(static_cast<double>(bytes), 1.0, name);
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
    throw NotSupportedError("ComfyUI model files are not supported as model_path; pass a Hugging Face bundle directory "
                            "(component files go in dit_file, text_encoder_file and vae_file).");
  }
  if (p.model_format != MI_MODEL_FORMAT_AUTO && p.model_format != MI_MODEL_FORMAT_HF_BUNDLE) {
    unknown("model_format", static_cast<int>(p.model_format));
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

// ---------------------------------------------------------------- pipeline types -> C structs

mi_tier from_tier(planner::Tier t) {
  switch (t) {
    case planner::Tier::kReference: return MI_TIER_REFERENCE;
    case planner::Tier::kHigh: return MI_TIER_HIGH;
    case planner::Tier::kTurbo: return MI_TIER_TURBO;
    case planner::Tier::kAuto: return MI_TIER_AUTO;
  }
  return MI_TIER_AUTO;
}

mi_attention from_attention(planner::AttentionMode a) {
  switch (a) {
    case planner::AttentionMode::kExact: return MI_ATTENTION_EXACT;
    case planner::AttentionMode::kQ8Fp16: return MI_ATTENTION_Q8_FP16;
    case planner::AttentionMode::kQ8Fp16F32Acc: return MI_ATTENTION_Q8_FP16_F32ACC;
    case planner::AttentionMode::kQ8Fp8: return MI_ATTENTION_Q8_FP8;
    case planner::AttentionMode::kQ8Fp8F32Acc: return MI_ATTENTION_Q8_FP8_F32ACC;
  }
  return MI_ATTENTION_DEFAULT;
}

// Result::te_precision (qwen35_text::precision_name); empty means the bundle's bf16 text encoder.
mi_precision te_precision(const std::string& name) {
  if (name == "int8") return MI_PRECISION_INT8;
  if (name == "w4a8") return MI_PRECISION_W4A8;
  if (name == "w4a4") return MI_PRECISION_W4A4;
  return MI_PRECISION_BF16;
}

std::int32_t count32(std::size_t n) {
  return n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ? std::numeric_limits<std::int32_t>::max()
                                                                                : static_cast<std::int32_t>(n);
}

mi_prompt_tokens to_tokens(const tok::PromptTokenReport& t) {
  mi_prompt_tokens out{};
  out.struct_size = sizeof out;
  out.total_tokens = count32(t.total_tokens);
  out.kept_tokens = count32(t.kept_tokens);
  out.prompt_tokens = count32(t.total_tokens >= t.template_tokens ? t.prompt_tokens() : 0);
  out.dropped_tokens = count32(t.total_tokens >= t.kept_tokens ? t.dropped_tokens() : 0);
  out.max_length = count32(t.max_length);
  out.truncated = t.truncated ? 1 : 0;
  return out;
}

}  // namespace

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
  q.sampler = to_sampler(p.sampler);
  q.scheduler = to_scheduler(p.scheduler);
  q.noise_layout = to_noise_layout(p.noise_layout);
  q.noise_source = to_noise_source(p.noise_source);
  q.flow_multiplier = p.flow_multiplier;
  q.eta = p.eta;
  q.s_noise = p.s_noise;
  q.save_metadata = p.save_metadata != 0;
  if (p.dataspace_path != nullptr && *p.dataspace_path != '\0') q.dataspace_path = util::path_from_utf8(p.dataspace_path);
  if (p.loras_count > 0 && p.loras == nullptr) throw std::invalid_argument("loras: NULL array with a nonzero count.");
  if (p.loras_count > kMaxLoras) {
    throw std::invalid_argument("loras: at most " + std::to_string(kMaxLoras) + " entries (got " +
                                std::to_string(p.loras_count) + ").");
  }
  for (std::size_t i = 0; i < p.loras_count; ++i) {
    const mi_lora& l = p.loras[i];
    const std::string name = "loras[" + std::to_string(i) + "]";
    if (l.path == nullptr || *l.path == '\0') throw std::invalid_argument(name + ".path is required.");
    // LoraSpec holds a float: a finite double beyond its range would become infinite.
    if (!std::isfinite(l.strength) || std::fabs(l.strength) > FLT_MAX) {
      throw std::invalid_argument(name + ".strength must be a finite number.");
    }
    q.loras.push_back({existing_file(l.path, name + ".path"), static_cast<float>(l.strength)});
  }
  if (p.preview_interval < 1) throw std::invalid_argument("preview_interval must be at least 1.");
  pipeline::validate(q);
  if (q.sampler == pipeline::SamplerKind::kDataSpace) existing_file(p.dataspace_path, "dataspace_path");
  return q;
}

void validate(const mi_gen_params& params) { to_request(params); }

pipeline::PipelineOptions to_options(const mi_engine_params& p) {
  pipeline::PipelineOptions o;
  switch (p.mode) {
    case MI_MODE_MIN_MEMORY: o.mode = planner::Mode::kMinMemory; break;
    case MI_MODE_BALANCED: o.mode = planner::Mode::kBalanced; break;
    case MI_MODE_MAX_SPEED: o.mode = planner::Mode::kMaxSpeed; break;
    default: unknown("mode", static_cast<int>(p.mode));
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
    default: unknown("block_path", static_cast<int>(p.block_path));
  }
  const auto component = [](const char* utf8, const char* name) {
    return utf8 != nullptr && *utf8 != '\0' ? existing_file(utf8, name) : fs::path();
  };
  o.dit_file = component(p.dit_file, "dit_file");
  o.text_encoder_file = component(p.text_encoder_file, "text_encoder_file");
  o.vae_file = component(p.vae_file, "vae_file");
  o.tier = to_tier(p.tier);
  o.attention = to_attention(p.attention);
  o.lora_strict = p.lora_strict != 0;
  return o;
}

Generation to_generation(const pipeline::Request& q, pipeline::Result& r, bool include_prompts) {
  Generation g;
  render_report(q, r, g);  // before r.rgb is moved out
  if (q.save_metadata && !r.cancelled) {
    // The CLI's metadata (moss-infer generate), with the prompts left out on request.
    pipeline::GenerationMetadata meta = pipeline::generation_metadata(q, r);
    meta.include_prompts = include_prompts;
    for (pipeline::PngText& t : pipeline::generation_text_chunks(meta)) g.png_text.emplace_back(std::move(t.keyword), std::move(t.text));
  }
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

  const planner::TierDecision& td = r.tier;
  g.tier.struct_size = sizeof g.tier;
  g.tier.requested = from_tier(td.requested);
  g.tier.effective = from_tier(td.effective);
  g.tier.attention = from_attention(td.attention);
  // Reference needs no certificate: certified, also for the default decision of a generation cancelled before it.
  g.tier.certified = td.certified || td.effective == planner::Tier::kReference ? 1 : 0;
  g.tier.dit_precision = td.precision == planner::Precision::kInt8 ? MI_PRECISION_INT8 : MI_PRECISION_BF16;
  g.tier.text_encoder_precision = te_precision(r.te_precision);
  g.tier_reason = td.reason;
  g.certificate = td.certificate;
  g.gpu_arch = r.gpu_arch;
  g.dit_source = r.dit_source;
  g.te_source = r.te_source;
  g.vae_source = r.vae_source;
  g.quality_json = pipeline::quality_report_json(r.quality);
  g.tokens[0] = to_tokens(r.cond_token_report);
  g.tokens[1] = to_tokens(r.uncond_token_report);
  for (const lora::FileReport& f : r.loras) {
    mi_lora_report l{};
    l.struct_size = sizeof l;
    l.adapters = f.adapters;
    l.side_branch = f.side_branch;
    l.merged = f.merged;
    l.unknown_keys = count32(f.unknown.size());
    l.warnings = count32(f.warnings.size());
    l.notes = count32(f.notes.size());
    g.loras.push_back(l);
    g.lora_files.push_back(f.file);
  }
  g.lora_warnings = r.lora_warnings;
  g.lora_notes = r.lora_notes;
  return g;
}

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
  // The pipeline's schedule length (pipeline.cpp): the dataspace sampler runs one step per requested step.
  const int total_steps = q.sampler == pipeline::SamplerKind::kDataSpace
                              ? q.steps
                              : pipeline::make_schedule(q.scheduler, q.steps, {q.shift, q.flow_multiplier}).num_steps();
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
  // Only with a callback: the preview's device buffers count in the denoise phase's plan.
  if (params.preview_callback != nullptr) {
    control.preview_every = params.preview_interval;
    control.on_preview = [&](const pipeline::PreviewImage& im) {
      params.preview_callback(im.rgb, im.width, im.height, im.step, params.user_data);
    };
  }
  pipeline::Result r;
  try {
    r = impl_->pipe->generate(q, control);
  } catch (const lora::LoraError& e) {
    throw ModelError(e.what());  // a LoRA file that cannot be applied (or, strict, an unknown key or a lossy merge)
  }
  return to_generation(q, r, params.include_prompts != 0);
}

std::string report_json(const Generation& generation, const std::string& image_path_utf8) {
  return generation.report_head + pipeline::json_escape(image_path_utf8) + generation.report_tail;
}

void write_png(const Generation& generation, const std::string& path_utf8) {
  if (generation.rgb.empty()) throw std::invalid_argument("the result has no image (the generation was cancelled).");
  std::vector<pipeline::PngText> text;
  for (const auto& [keyword, body] : generation.png_text) text.push_back({keyword, body});
  pipeline::write_png_rgb8(util::path_from_utf8(path_utf8), generation.rgb, generation.width, generation.height, text);
}

void write_png_rgb(const std::string& path_utf8, const std::uint8_t* rgb, int width, int height) {
  const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
  pipeline::write_png_rgb8(util::path_from_utf8(path_utf8), std::span<const std::uint8_t>(rgb, bytes), width, height);
}

}  // namespace mi::api
