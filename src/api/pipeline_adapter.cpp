#include "api/pipeline_adapter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <system_error>
#include <utility>

#include "backend/cuda_driver.h"
#include "backend/cuda_libs.h"
#include "core/cuda_check.h"
#include "models/control/control.h"
#include "models/lora/lora.h"
#include "models/moss_dit/distill.h"
#include "api/progress_model.h"
#include "models/moss_dit/moss_dit.h"
#include "models/qwen35_text/quant_text_model.h"
#include "pipeline/gguf_dit.h"
#include "pipeline/hardware_profile.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/png_writer.h"
#include "pipeline/quality_report.h"
#include "pipeline/report_json.h"
#include "pipeline/sampler.h"
#include "planner/planner.h"
#include "util/path.h"
#include "util/regular_file.h"

namespace mi::api {

// The CLI's document (pipeline/report_json.h) in braces; the C API writes no latents.
void render_report(const pipeline::Request& q, const pipeline::Result& r, Generation& g) {
  pipeline::ReportParts parts = pipeline::report_parts(q, r, "");
  g.report_head = "{\n" + std::move(parts.head);
  g.report_tail = std::move(parts.tail) + "\n}\n";
}

mi_precision to_precision(planner::Precision p) {
  switch (p) {
    case planner::Precision::kBf16: return MI_PRECISION_BF16;
    case planner::Precision::kInt8: return MI_PRECISION_INT8;
    case planner::Precision::kInt4: return MI_PRECISION_W4A4;
    case planner::Precision::kNvfp4: return MI_PRECISION_NVFP4;
  }
  return MI_PRECISION_BF16;
}

mi_precision to_precision(qwen35_text::Precision p) {
  switch (p) {
    case qwen35_text::Precision::kBF16: return MI_PRECISION_BF16;
    case qwen35_text::Precision::kInt8: return MI_PRECISION_INT8;
    case qwen35_text::Precision::kW4A8: return MI_PRECISION_W4A8;
  }
  return MI_PRECISION_BF16;
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

// A UTF-8 path field, through the boundary check (util::check_user_path: a Windows device or pipe path is refused
// before any file-system call touches it). Every path field of the C API enters through here.
fs::path user_path(const char* utf8, const std::string& name) {
  fs::path path = util::path_from_utf8(utf8);
  util::check_user_path(path, name);
  return path;
}

// A UTF-8 path that must name an existing regular file.
fs::path existing_file(const char* utf8, const std::string& name) {
  const fs::path path = user_path(utf8, name);
  const util::FileInfo info = util::file_info(path);
  if (!info.exists()) throw std::invalid_argument(name + ": file not found: " + utf8);
  if (!info.regular()) throw std::invalid_argument(name + ": not a regular file: " + utf8);
  return path;
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
    case MI_TIER_AUTO:  // reserved since 2.0: the caller chooses
      throw std::invalid_argument("tier is required: choose MI_TIER_REFERENCE, MI_TIER_HIGH or MI_TIER_TURBO "
                                  "(reference | high | turbo).");
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
  const fs::path path = user_path(p.model_path, "model_path");
  const util::FileInfo info = util::file_info(path);
  if (p.model_format == MI_MODEL_FORMAT_COMFYUI || (p.model_format == MI_MODEL_FORMAT_AUTO && info.regular())) {
    throw NotSupportedError("ComfyUI model files are not supported as model_path; pass a Hugging Face bundle directory "
                            "(component files go in dit_file, text_encoder_file and vae_file).");
  }
  if (p.model_format != MI_MODEL_FORMAT_AUTO && p.model_format != MI_MODEL_FORMAT_HF_BUNDLE) {
    unknown("model_format", static_cast<int>(p.model_format));
  }
  if (!info.directory()) throw std::invalid_argument("model bundle not found: " + std::string(p.model_path));
  for (const char* component : {"text_encoder", "tokenizer", "transformer", "vae"}) {
    if (!util::file_info(path / component).directory()) {
      throw ModelError("not a MOSS-Image bundle (no " + std::string(component) + "/ directory): " + std::string(p.model_path));
    }
  }
  return path;
}

// ---------------------------------------------------------------- pipeline types -> C structs

mi_tier from_tier(planner::Tier t) {
  switch (t) {
    case planner::Tier::kReference: return MI_TIER_REFERENCE;
    case planner::Tier::kHigh: return MI_TIER_HIGH;
    case planner::Tier::kTurbo: return MI_TIER_TURBO;
  }
  return MI_TIER_REFERENCE;
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

// Result::te_precision is qwen35_text::precision_name() of the encoder that ran ("int8_convrot" for INT8); empty means
// the bundle's bf16 text encoder.
mi_precision te_precision(const std::string& name) {
  for (const qwen35_text::Precision p :
       {qwen35_text::Precision::kBF16, qwen35_text::Precision::kInt8, qwen35_text::Precision::kW4A8}) {
    if (name == qwen35_text::precision_name(p)) return to_precision(p);
  }
  return MI_PRECISION_BF16;
}

std::int32_t count32(std::size_t n) {
  return n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ? std::numeric_limits<std::int32_t>::max()
                                                                                : static_cast<std::int32_t>(n);
}

bool given(const char* utf8) { return utf8 != nullptr && *utf8 != '\0'; }

// A distill bundle directory: DistillError (a bundle without any of its files, an unreadable entry) is a model error;
// a path that is not a directory is an argument error.
moss_dit::DistillBundle distill_bundle(const char* utf8, const std::string& name) {
  const fs::path dir = user_path(utf8, name);
  if (!util::file_info(dir).directory()) throw std::invalid_argument(name + ": not a directory: " + utf8);
  try {
    return moss_dit::open_distill_bundle(dir);
  } catch (const moss_dit::DistillError& e) {
    throw ModelError(name + ": " + e.what());
  }
}

// An image input of a request, checked without copying or decoding anything: pixels (sizes, channels, a BINARY01
// mask's values) or a file (an existing regular file within kMaxImageFileBytes). At most one of the two.
struct ImageInput {
  const mi_image* pixels = nullptr;
  fs::path file;
  bool given() const { return pixels != nullptr || !file.empty(); }
};

[[noreturn]] void not_binary01(const std::string& n, std::size_t i, int v) {
  throw std::invalid_argument(n + ": with MI_MASK_BINARY01 every value must be 0 or 1 (pixel " + std::to_string(i) +
                              " is " + std::to_string(v) + ").");
}

ImageInput check_image(const mi_image* px, const char* path, int channels, bool binary01, const std::string& n) {
  if (px != nullptr && given(path)) throw std::invalid_argument(n + " and " + n + "_path are both set; pass the image one way.");
  ImageInput in;
  if (px != nullptr) {
    if (px->pixels == nullptr) throw std::invalid_argument(n + ".pixels is NULL.");
    if (px->_reserved0 != 0) throw std::invalid_argument(n + "._reserved0 must be 0.");
    if (px->channels != channels) {
      throw std::invalid_argument(n + ".channels must be " + std::to_string(channels) +
                                  (channels == 3 ? " (RGB)" : " (grayscale)") + ", not " + std::to_string(px->channels) + ".");
    }
    if (px->width < 1 || px->height < 1 || px->width > pipeline::kMaxInputSide || px->height > pipeline::kMaxInputSide ||
        std::int64_t{px->width} * px->height > pipeline::kMaxInputPixels) {
      throw std::invalid_argument(n + " is " + std::to_string(px->width) + "x" + std::to_string(px->height) +
                                  "; each side must be 1 to " + std::to_string(pipeline::kMaxInputSide) + " and the image at most " +
                                  std::to_string(pipeline::kMaxInputPixels) + " pixels.");
    }
    if (binary01) {
      const std::size_t bytes = static_cast<std::size_t>(px->width) * static_cast<std::size_t>(px->height) * channels;
      for (std::size_t i = 0; i < bytes; ++i) {
        if (px->pixels[i] > 1) not_binary01(n, i, px->pixels[i]);
      }
    }
    in.pixels = px;
  } else if (given(path)) {
    in.file = existing_file(path, n + "_path");
    const std::uint64_t size = util::file_info(in.file).size;
    if (size > pipeline::kMaxImageFileBytes) {
      throw std::invalid_argument(n + "_path: the file is " + std::to_string(size) + " bytes; at most " +
                                  std::to_string(pipeline::kMaxImageFileBytes) + " are read.");
    }
  }
  return in;
}

// The image of a checked input: the pixels copied or the file decoded (an unreadable file is an I/O error, one that cannot be
// decoded an argument error), a BINARY01 mask scaled to 0 / 255.
std::shared_ptr<const pipeline::Image> load_input(const ImageInput& in, int channels, bool binary01, const std::string& n) {
  if (!in.given()) return nullptr;
  pipeline::Image im;
  if (in.pixels != nullptr) {
    im.width = in.pixels->width;
    im.height = in.pixels->height;
    im.channels = channels;
    const std::size_t bytes = static_cast<std::size_t>(im.width) * static_cast<std::size_t>(im.height) * channels;
    im.pixels.assign(in.pixels->pixels, in.pixels->pixels + bytes);
  } else {
    try {
      im = pipeline::load_image(in.file, channels);
    } catch (const pipeline::ImageReadError& e) {
      throw pipeline::ImageReadError(n + "_path: " + e.what());  // unreadable file: MI_ERR_IO (c_api.cpp guarded)
    } catch (const pipeline::ImageFormatError& e) {
      throw std::invalid_argument(n + "_path: " + e.what());  // undecodable, too large
    }
  }
  if (binary01) {
    for (std::size_t i = 0; i < im.pixels.size(); ++i) {
      if (im.pixels[i] > 1) not_binary01(n, i, im.pixels[i]);
      im.pixels[i] = static_cast<std::uint8_t>(im.pixels[i] * 255);
    }
  }
  return std::make_shared<const pipeline::Image>(std::move(im));
}

// A 1x1 stand-in that lets pipeline::validate check a request whose images are not loaded (mi_gen_params_validate).
std::shared_ptr<const pipeline::Image> stand_in(int channels) {
  pipeline::Image im;
  im.width = im.height = 1;
  im.channels = channels;
  im.pixels.assign(static_cast<std::size_t>(channels), 0);
  return std::make_shared<const pipeline::Image>(std::move(im));
}

mi_image_mode to_image_mode(const std::string& mode) {
  if (mode == "img2img") return MI_IMAGE_MODE_IMG2IMG;
  if (mode == "inpaint") return MI_IMAGE_MODE_INPAINT;
  return MI_IMAGE_MODE_TXT2IMG;
}

}  // namespace

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

mi_phase to_phase(pipeline::Phase phase) {
  switch (phase) {
    case pipeline::Phase::kTextEncoder: return MI_PHASE_TEXT_ENCODER;
    case pipeline::Phase::kDitLoad: return MI_PHASE_DIT_LOAD;
    case pipeline::Phase::kDenoise: return MI_PHASE_DENOISE;
    case pipeline::Phase::kDecode: return MI_PHASE_DECODE;
  }
  return MI_PHASE_TEXT_ENCODER;
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

pipeline::ResizeMode to_resize(mi_resize_mode m) {
  switch (m) {
    case MI_RESIZE_CROP: return pipeline::ResizeMode::kCrop;
    case MI_RESIZE_FIT: return pipeline::ResizeMode::kFit;
    case MI_RESIZE_STRETCH: return pipeline::ResizeMode::kStretch;
    case MI_RESIZE_MAX_ENUM: break;
  }
  unknown("resize", static_cast<int>(m));
}

pipeline::Request to_request(const mi_gen_params& p, fs::path* distill_time_embedding, bool load_images, bool preload) {
  if (p.prompt == nullptr && !preload) throw std::invalid_argument("Enter a prompt.");
  // A distill bundle supplies what the caller left unset (cmd_generate's --distill).
  std::optional<moss_dit::DistillBundle> distill;
  if (given(p.distill_dir)) distill = distill_bundle(p.distill_dir, "distill_dir");
  if (distill_time_embedding != nullptr) *distill_time_embedding = distill ? distill->time_embedding : fs::path();
  pipeline::Request q;
  q.prompt = p.prompt != nullptr ? p.prompt : "";
  q.negative_prompt = p.negative_prompt != nullptr ? p.negative_prompt : "";
  q.width = p.width;
  q.height = p.height;
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
  if (given(p.dataspace_path)) q.dataspace_path = user_path(p.dataspace_path, "dataspace_path");
  if (p.loras_count > 0 && p.loras == nullptr) throw std::invalid_argument("loras: NULL array with a nonzero count.");
  const auto check_lora_count = [](std::size_t n, const char* what) {
    if (n > kMaxLoras) {
      throw std::invalid_argument(std::string("loras: at most ") + std::to_string(kMaxLoras) + " entries" + what + " (got " +
                                  std::to_string(n) + ").");
    }
  };
  check_lora_count(p.loras_count, "");
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

  // The distill bundle: its LoRA, and its dataspace file for the sampler when that is unset (EULER without a dataspace
  // file) and for an explicit DATASPACE without one.
  bool distill_dataspace = false;
  if (distill) {
    if (!distill->lora.empty()) q.loras.push_back({distill->lora, 1.0f});
    check_lora_count(q.loras.size(), ", the distill bundle's included");
    if (!distill->dataspace.empty() && p.sampler == MI_SAMPLER_EULER && q.dataspace_path.empty()) {
      q.sampler = pipeline::SamplerKind::kDataSpace;
      distill_dataspace = true;
    }
    if (!distill->dataspace.empty() && q.sampler == pipeline::SamplerKind::kDataSpace && q.dataspace_path.empty()) {
      q.dataspace_path = distill->dataspace;
    }
  }
  // The dataspace sampler's file, read whenever it runs: its step codes bound the steps.
  std::optional<std::size_t> step_codes;
  if (q.sampler == pipeline::SamplerKind::kDataSpace && !q.dataspace_path.empty()) {
    const fs::path file = existing_file(util::path_utf8(q.dataspace_path).c_str(), "dataspace_path");
    try {
      step_codes = pipeline::load_dataspace(util::path_utf8(file)).step_codes.size();
    } catch (const pipeline::DataspaceError& e) {
      throw ModelError(std::string(distill_dataspace ? "distill_dir: " : "dataspace_path: ") + e.what());
    }
  }

  // The unset markers (mossinfer.h): steps 0, a negative cfg, a negative denoise strength.
  if (p.steps < 0) throw std::invalid_argument("Steps must be positive (0 means unset).");
  q.steps = p.steps != 0 ? p.steps : step_codes ? static_cast<int>(*step_codes) : MI_DEFAULT_STEPS;
  if (!std::isfinite(p.cfg)) throw std::invalid_argument("CFG must be a finite number (a negative value means unset).");
  q.cfg = p.cfg >= 0 ? p.cfg : distill ? 1.0 : MI_DEFAULT_CFG;
  if (!std::isfinite(p.denoise_strength)) {
    throw std::invalid_argument("Denoise strength must be a finite number (a negative value means unset).");
  }

  // Image input and the hires fix.
  q.resize = to_resize(p.resize);
  if (p.mask_encoding != MI_MASK_U8 && p.mask_encoding != MI_MASK_BINARY01) {
    unknown("mask_encoding", static_cast<int>(p.mask_encoding));
  }
  const bool binary01 = p.mask_encoding == MI_MASK_BINARY01;
  const ImageInput init = check_image(p.init_image, p.init_image_path, 3, false, "init_image");
  const ImageInput mask = check_image(p.mask, p.mask_path, 1, binary01, "mask");
  q.denoise_strength = p.denoise_strength >= 0 ? p.denoise_strength : init.given() ? MI_DEFAULT_DENOISE_STRENGTH : 1.0;
  if (load_images) {
    q.init_image = load_input(init, 3, false, "init_image");
    q.mask = load_input(mask, 1, binary01, "mask");
  } else {
    // Not loaded: 1x1 stand-ins for the pipeline's checks, and its size rule for two pixel images here.
    if (init.given()) q.init_image = stand_in(3);
    if (mask.given()) q.mask = stand_in(1);
    if (init.pixels != nullptr && mask.pixels != nullptr &&
        (init.pixels->width != mask.pixels->width || init.pixels->height != mask.pixels->height)) {
      throw std::invalid_argument("The inpainting mask must have the init image's size (" + std::to_string(init.pixels->width) +
                                  "x" + std::to_string(init.pixels->height) + "), not " + std::to_string(mask.pixels->width) +
                                  "x" + std::to_string(mask.pixels->height) + ".");
    }
  }
  if (p.hires_width < 0 || p.hires_height < 0) {
    throw std::invalid_argument("Hires fix: hires_width and hires_height must be 0 (off) or positive multiples of 16.");
  }
  q.hires.width = p.hires_width;
  q.hires.height = p.hires_height;
  q.hires.strength = p.hires_strength;
  q.hires.steps = p.hires_steps;

  // Control adapter: the file must exist; its format and kinds are the pipeline's to check (generate()).
  if (given(p.control_adapter)) q.control.adapter = existing_file(p.control_adapter, "control_adapter");
  if (!std::isfinite(p.control_lora_strength) || std::fabs(p.control_lora_strength) > FLT_MAX) {
    throw std::invalid_argument("control_lora_strength must be a finite number.");  // Control::lora_strength is a float
  }
  q.control.lora_strength = static_cast<float>(p.control_lora_strength);
  if (p.control_images_count > 0 && p.control_images == nullptr) {
    throw std::invalid_argument("control_images: NULL array with a nonzero count.");
  }
  if (p.control_images_count > kMaxControlImages) {
    throw std::invalid_argument("control_images: at most " + std::to_string(kMaxControlImages) + " entries (got " +
                                std::to_string(p.control_images_count) + ").");
  }
  if (p.control_images_count > 0 && !q.control.enabled()) {
    throw std::invalid_argument("control_images need a control_adapter.");
  }
  for (std::size_t i = 0; i < p.control_images_count; ++i) {
    const mi_control_image& c = p.control_images[i];
    const std::string name = "control_images[" + std::to_string(i) + "]";
    if (!given(c.kind)) throw std::invalid_argument(name + ".kind is required (one of the adapter's, e.g. depth).");
    if (c.image == nullptr && !given(c.image_path)) throw std::invalid_argument(name + " needs an image or an image_path.");
    if (c.image != nullptr && c.image->channels != 1 && c.image->channels != 3) {
      throw std::invalid_argument(name + ".image.channels must be 3 (RGB) or 1 (grayscale), not " +
                                  std::to_string(c.image->channels) + ".");
    }
    if (!std::isfinite(c.strength) || c.strength < 0 || c.strength > 10) {
      throw std::invalid_argument(name + ".strength must be between 0 and 10.");
    }
    if (!(0 <= c.start && c.start <= c.end && c.end <= 1)) throw std::invalid_argument(name + ": require 0 <= start <= end <= 1.");
    // A file is read as RGB (a grayscale file replicated, as `moss-infer generate --control-image`); pixels as given.
    const int channels = c.image != nullptr ? c.image->channels : 3;
    const ImageInput in = check_image(c.image, c.image_path, channels, false, name + ".image");
    pipeline::Request::ControlImage ci;
    ci.kind = c.kind;
    ci.image = load_images ? load_input(in, channels, false, name + ".image") : stand_in(channels);
    ci.strength = c.strength;
    ci.start = c.start;
    ci.end = c.end;
    q.control.images.push_back(std::move(ci));
  }

  // The lossy accelerators (pipeline/accel/accel.h): a threshold of 0 is unset (the model's default, which the pipeline
  // fills in and reports).
  q.accel.cfg_truncation = p.cfg_truncation;
  switch (p.step_cache) {
    case MI_STEP_CACHE_OFF: q.accel.step_cache = pipeline::accel::StepCacheKind::kOff; break;
    case MI_STEP_CACHE_FBCACHE: q.accel.step_cache = pipeline::accel::StepCacheKind::kFbCache; break;
    default: unknown("step_cache", static_cast<int>(p.step_cache));
  }
  if (!std::isfinite(p.step_cache_threshold) || p.step_cache_threshold < 0 || p.step_cache_threshold > 1) {
    throw std::invalid_argument("step_cache_threshold must be 0 (the default) or greater than 0 and at most 1.");
  }
  q.accel.step_cache_threshold = p.step_cache_threshold > 0 ? std::optional<double>(p.step_cache_threshold) : std::nullopt;

  // Reference images (MI_CAP_REFERENCE_IMAGES): pixels (copied) or files (read as RGB), in order; their sizes and the
  // token limit are pipeline::validate's. Not loaded (validate only): a file's size is unknown here, so a 16x16
  // stand-in (the file's existence is checked).
  if (p.reference_images_count > 0 && p.reference_images == nullptr) {
    throw std::invalid_argument("reference_images: NULL array with a nonzero count.");
  }
  if (p.reference_images_count > static_cast<std::size_t>(pipeline::kMaxReferenceImages)) {
    throw std::invalid_argument("reference_images: at most " + std::to_string(pipeline::kMaxReferenceImages) +
                                " entries (got " + std::to_string(p.reference_images_count) + ").");
  }
  switch (p.lora_placement) {
    case MI_LORA_PLACEMENT_DEFAULT: q.lora_placement = lora::Placement::kDefault; break;
    case MI_LORA_PLACEMENT_MERGE: q.lora_placement = lora::Placement::kMerge; break;
    case MI_LORA_PLACEMENT_SIDE_BRANCH: q.lora_placement = lora::Placement::kSideBranch; break;
    default: unknown("lora_placement", static_cast<int>(p.lora_placement));
  }
  if (p.reference_max_edge != 0 && (p.reference_max_edge < 16 || p.reference_max_edge > 2048)) {
    throw std::invalid_argument("reference_max_edge must be 0 (the default, 512) or 16 to 2048.");
  }
  q.references.max_edge = p.reference_max_edge != 0 ? p.reference_max_edge : 512;
  for (std::size_t i = 0; i < p.reference_images_count; ++i) {
    const mi_reference_image& r = p.reference_images[i];
    const std::string name = "reference_images[" + std::to_string(i) + "]";
    if (r._reserved0 != 0 || r._reserved1 != 0) throw std::invalid_argument(name + "._reserved0 / _reserved1 must be 0.");
    if (r.image == nullptr && !given(r.image_path)) throw std::invalid_argument(name + " needs an image or an image_path.");
    if (r.image != nullptr && r.image->channels != 1 && r.image->channels != 3 && r.image->channels != 4) {
      throw std::invalid_argument(name + ".image.channels must be 1, 3 or 4, not " + std::to_string(r.image->channels) + ".");
    }
    const int channels = r.image != nullptr ? r.image->channels : 3;
    const ImageInput in = check_image(r.image, r.image_path, channels, false, name + ".image");
    if (load_images || in.pixels != nullptr) {
      q.references.images.push_back(load_input(in, channels, false, name + ".image"));
    } else {
      pipeline::Image s;
      s.width = s.height = 16;
      s.pixels.assign(16 * 16 * 3, 0);
      q.references.images.push_back(std::make_shared<const pipeline::Image>(std::move(s)));
    }
  }

  if (distill_dataspace && (init.given() || q.hires.enabled())) {
    throw std::invalid_argument("distill_dir selected the bundle's dataspace sampler, which cannot start from an image "
                                "(init_image, hires_width / hires_height); set sampler to use another sampler.");
  }
  if (preload) {
    // The prompt is the one thing a preload need not have: the rest is checked as for a generation.
    std::string prompt = std::move(q.prompt);
    q.prompt = "preload";
    pipeline::validate(q);
    q.prompt = std::move(prompt);
  } else {
    pipeline::validate(q);
  }
  if (step_codes && static_cast<std::size_t>(q.steps) > *step_codes) {
    throw std::invalid_argument("The dataspace sampler runs 1 to " + std::to_string(*step_codes) + " steps (its file has " +
                                std::to_string(*step_codes) + " step codes), not " + std::to_string(q.steps) + ".");
  }
  return q;
}

void validate(const mi_gen_params& params) { to_request(params, nullptr, false); }

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
  if (p.cache_dir != nullptr && *p.cache_dir != '\0') o.cache_dir = user_path(p.cache_dir, "cache_dir");
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
  // certificate_dirs / certificate_dirs_count: reserved since 2.0, ignored.
  switch (p.dit_direct_io) {
    case MI_DIRECT_IO_AUTO: o.dit_load_direct_io = std::nullopt; break;  // adaptive
    case MI_DIRECT_IO_BUFFERED: o.dit_load_direct_io = false; break;
    case MI_DIRECT_IO_DIRECT: o.dit_load_direct_io = true; break;
    default: unknown("dit_direct_io", static_cast<int>(p.dit_direct_io));
  }
  switch (p.te_host_pin) {
    case MI_TE_HOST_PIN_AUTO: o.te_host_pin = planner::TeHostPin::kAuto; break;
    case MI_TE_HOST_PIN_ON: o.te_host_pin = planner::TeHostPin::kOn; break;
    case MI_TE_HOST_PIN_OFF: o.te_host_pin = planner::TeHostPin::kOff; break;
    default: unknown("te_host_pin", static_cast<int>(p.te_host_pin));
  }
  // The step-embedder replacement is per engine: given directly, or a distill bundle's.
  o.time_embedding_file = component(p.time_embedding_file, "time_embedding_file");
  if (given(p.distill_dir)) {
    const moss_dit::DistillBundle b = distill_bundle(p.distill_dir, "distill_dir");
    if (!b.time_embedding.empty()) {
      if (!o.time_embedding_file.empty() &&
          !util::same_file(util::file_info(o.time_embedding_file), util::file_info(b.time_embedding))) {
        throw std::invalid_argument("time_embedding_file and distill_dir name different time embeddings (" +
                                    util::path_utf8(o.time_embedding_file) + ", " + util::path_utf8(b.time_embedding) +
                                    "); set one of them.");
      }
      o.time_embedding_file = b.time_embedding;
    }
  }
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
  g.timings.vae_encoder_load_s = t.vae_encoder_load_s;
  g.timings.image_encode_s = t.encode_s;
  g.timings.image_ops_s = t.image_ops_s;
  g.timings.lora_load_s = t.lora_load_s;
  g.timings.lora_merge_s = t.lora_merge_s;
  g.image.struct_size = sizeof g.image;
  g.image.mode = to_image_mode(r.image_mode);
  g.image.hires_ran = r.hires.ran ? 1 : 0;
  g.image.first_width = r.first_pass.width;
  g.image.first_height = r.first_pass.height;
  g.image.first_steps_scheduled = std::max(0, r.first_pass.schedule_steps - r.first_pass.start_step);
  g.image.first_start_step = r.first_pass.start_step;
  g.image.first_steps_run = r.first_pass.steps_run;
  g.image.first_encoded = r.first_pass.encoded ? 1 : 0;
  g.image.first_strength = r.first_pass.strength;
  if (q.hires.enabled()) {
    g.image.hires_width = r.hires.ran ? r.hires.width : q.hires.width;
    g.image.hires_height = r.hires.ran ? r.hires.height : q.hires.height;
    g.image.hires_steps_scheduled = std::max(0, r.hires.schedule_steps - r.hires.start_step);
    g.image.hires_start_step = r.hires.start_step;
    g.image.hires_steps_run = r.hires.steps_run;
    g.image.hires_strength = r.hires.ran ? r.hires.strength : q.hires.strength;
  }
  g.image_note = r.image_note;
  g.time_embedding = r.time_embedding;
  const pipeline::MemoryStats& m = r.memory;
  g.memory.struct_size = sizeof g.memory;
  g.memory.arena_capacity_bytes = m.arena_capacity;
  g.memory.arena_high_water_bytes = m.arena_high_water;
  g.memory.device_peak_bytes = m.device_peak;
  g.memory.device_free_at_start_bytes = m.free_at_start;
  g.memory.host_private_start_bytes = m.host_private_start;
  g.memory.host_private_peak_bytes = m.host_private_peak;
  g.memory.host_copy_bytes = m.host_copy_bytes;
  g.memory.merge_memo_bytes = m.merge_memo_bytes;
  g.memory.dit_loads = m.dit_loads;
  const pipeline::DitLoadReport& dl = r.dit_load;
  g.dit_load.struct_size = sizeof g.dit_load;
  g.dit_load.source = dl.source == "files"       ? MI_DIT_LOAD_SOURCE_FILES
                      : dl.source == "host_copy" ? MI_DIT_LOAD_SOURCE_HOST_COPY
                                                 : MI_DIT_LOAD_SOURCE_NONE;
  g.dit_load.io = g.dit_load.source != MI_DIT_LOAD_SOURCE_FILES ? MI_LOAD_IO_NONE
                  : dl.io == "direct"                           ? MI_LOAD_IO_DIRECT
                  : dl.io == "buffered"                         ? MI_LOAD_IO_BUFFERED
                                                                : MI_LOAD_IO_NONE;
  g.dit_load.bytes = dl.bytes;
  g.dit_load.seconds = dl.seconds;
  g.dit_load_note = dl.note;
  g.te_note = r.te_note;
  g.preload_note = r.preload_note;
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
  g.tier.requested = from_tier(td.tier);
  g.tier.effective = from_tier(td.tier);  // the caller's tier runs, or is refused
  g.tier.attention = from_attention(td.attention);
  // certified and run_certified: reserved since 2.0, 0 (the zeroed struct).
  g.tier.dit_precision = to_precision(td.precision);
  g.tier.text_encoder_precision = te_precision(r.te_precision);
  g.tier_reason = td.reason;
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

  const pipeline::Result::ControlReport& c = r.control;
  g.control.struct_size = sizeof g.control;
  g.control.active = c.adapter.empty() ? 0 : 1;
  g.control.kinds_count = count32(c.kinds.size());
  g.control.encodes = c.encodes;
  g.control.lora_adapters = c.lora.adapters;
  g.control.lora_side_branch = c.lora.side_branch;
  g.control.lora_merged = c.lora.merged;
  g.control.lora_unknown_keys = count32(c.lora.unknown.size());
  g.control.lora_warnings = count32(c.lora.warnings.size());
  g.control.lora_notes = count32(c.lora.notes.size());
  g.control_adapter = c.adapter;
  for (const std::string& k : c.kinds) g.control_kinds += (g.control_kinds.empty() ? "" : ",") + k;
  g.control_base_model = c.base_model;
  g.control_encode_schedule = c.encode_schedule;
  g.control_warnings = c.lora.warnings;
  g.control_notes = c.lora.notes;

  const pipeline::accel::Stats& a = r.accel;
  g.accel.struct_size = sizeof g.accel;
  g.accel.cfg_truncated_steps = a.cfg_truncated_steps;
  g.accel.dit_calls = a.dit_calls;
  g.accel.cached_calls = a.cached_calls;
  g.accel.lossy = q.accel.active(q.cfg != 1.0) ? 1 : 0;
  g.accel.step_cache_threshold = a.step_cache_threshold;
  // As the report's accelerators.active: an unset threshold as the model's default that ran.
  pipeline::accel::Options ran = q.accel;
  if (ran.step_cache_on() && a.step_cache_threshold > 0) ran.step_cache_threshold = a.step_cache_threshold;
  g.accelerators = ran.describe(q.cfg != 1.0);
  g.cached_steps.assign(a.cached_steps.begin(), a.cached_steps.end());
  g.step_cache_diffs = a.step_cache_diffs;
  g.step_cache_threshold_source = a.step_cache_threshold_source;
  g.accel_warnings = r.accel_warnings;
  g.references_json = pipeline::references_json(q, r);
  g.reference_warnings = r.reference_warnings;
  return g;
}

struct PipelineAdapter::Impl {
  fs::path model;
  fs::path time_embedding;  // PipelineOptions::time_embedding_file
  fs::path cache_dir;       // PipelineOptions::cache_dir (the hardware profile's directory)
  int device = 0;
  std::unique_ptr<pipeline::Pipeline> pipe;
  // The progress model's inputs that outlive a job: the weight files' sizes, the warm / cold record (from the jobs'
  // results, their exceptions and the releases), and the transfer rates of the last plan (none yet: the cached profile,
  // else the defaults).
  ModelBytes model_bytes;
  HeldModels held;
  std::atomic<bool> held_stale{false};  // models_changed() since the last job
  // At a job's start: after a component call, what is loaded is unknown.
  void sync_held() {
    if (held_stale.exchange(false)) held.job_failed();
  }
  std::optional<planner::HardwareProfile> hardware;
  const planner::HardwareProfile& rates();
};

// The rates the pipeline plans with: the last result's, else (the engine's first job) the profile the pipeline reads
// from (or has just probed into) its cache directory, else the planner's defaults. Called from the plan callback.
const planner::HardwareProfile& PipelineAdapter::Impl::rates() {
  if (!hardware) hardware = stored_rates(cache_dir);
  return *hardware;
}

PipelineAdapter::PipelineAdapter(const mi_engine_params& params, bool plan_only) : impl_(std::make_unique<Impl>()) {
  impl_->model = model_dir(params);
  pipeline::PipelineOptions options = to_options(params);
  options.plan_only = plan_only;
  impl_->time_embedding = options.time_embedding_file;
  impl_->cache_dir = options.cache_dir;
  impl_->model_bytes = model_bytes(impl_->model, options.text_encoder_file, options.vae_file);
  backend::load_cublaslt();  // before any CUDA work: no usable copy is CudaLibraryError (MI_ERR_CUDA), with every candidate
  const backend::ScopedCurrentContext restore;  // the caller's CUDA context is current again on return
  int count = 0;
  MI_CUDA_CHECK(cudaGetDeviceCount(&count));
  if (params.device_index < 0 || params.device_index >= count) {
    throw std::invalid_argument("device_index " + std::to_string(params.device_index) + " out of range (" +
                                std::to_string(count) + " CUDA devices).");
  }
  impl_->device = params.device_index;
  MI_CUDA_CHECK(cudaSetDevice(impl_->device));
  try {
    impl_->pipe = std::make_unique<pipeline::Pipeline>(options);
  } catch (const moss_dit::DistillError& e) {
    throw ModelError(e.what());
  }
}

// Must not throw (mi_engine_free() has no way to report it): only CUDA C calls whose errors are ignored, and the
// pipeline's destructors release with plain CUDA calls as well.
PipelineAdapter::~PipelineAdapter() {
  // The pipeline frees its device memory on the engine's device, whichever context the caller has current.
  const backend::ScopedCurrentContext restore;
  if (cudaSetDevice(impl_->device) != cudaSuccess) cudaGetLastError();
  impl_->pipe.reset();
}

namespace {

// A distill bundle's step embedder is the engine's (PipelineOptions): the request's bundle must use the same one.
void check_distill_time_embedding(const fs::path& engine_te, const fs::path& distill_te) {
  if (distill_te.empty()) return;
  if (engine_te.empty() || !util::same_file(util::file_info(engine_te), util::file_info(distill_te))) {
    throw std::invalid_argument("distill_dir has a time embedding (" + util::path_utf8(distill_te) +
                                ") that this engine does not use; set the bundle as mi_engine_params.distill_dir "
                                "(or its file as time_embedding_file) when creating the engine.");
  }
}

// A pipeline call with its model-file errors as ModelError (MI_ERR_MODEL).
template <class F>
pipeline::Result model_errors(F&& call) {
  try {
    return call();
  } catch (const lora::LoraError& e) {
    throw ModelError(e.what());  // a LoRA file that cannot be applied (or, strict, an unknown key or a lossy merge)
  } catch (const pipeline::DataspaceError& e) {
    throw ModelError(e.what());  // a dataspace file that changed since to_request read it
  } catch (const control::ControlError& e) {
    throw ModelError(e.what());  // control_adapter is not a control adapter (no widened embedder, bad kinds metadata)
  }
}

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// The progress callback of `params`, `t` seconds into the job, with the job's fraction and eta from `estimate`.
void progress(const mi_gen_params& params, double t, mi_phase phase, int step, int total, double step_s,
              const ProgressModel& estimate) {
  if (params.progress_callback == nullptr) return;
  mi_progress p{};
  p.struct_size = sizeof p;
  p.phase = phase;
  p.step = step;
  p.total_steps = total;
  p.step_seconds = step_s;
  p.elapsed_seconds = t;
  p.fraction = estimate.fraction();
  p.eta_seconds = estimate.eta_seconds();
  params.progress_callback(&p, params.user_data);
}

// A pipeline call whose exception leaves the progress record cold: what the call loaded or dropped is unknown.
template <class F>
auto cold_on_throw(HeldModels& held, F&& call) {
  try {
    return call();
  } catch (...) {
    held.job_failed();
    throw;
  }
}

}  // namespace

Generation PipelineAdapter::generate(const mi_gen_params& params, const std::atomic<bool>& cancel) {
  const auto t0 = Clock::now();
  fs::path distill_te;
  pipeline::Request q = to_request(params, &distill_te, true);
  q.model = impl_->model;
  check_distill_time_embedding(impl_->time_embedding, distill_te);
  // The CUDA device is per host thread; the caller's thread may not be the one that created the engine.
  const backend::ScopedCurrentContext restore;
  MI_CUDA_CHECK(cudaSetDevice(impl_->device));
  impl_->sync_held();
  const JobShape shape = job_shape(q);
  ProgressModel estimate;
  pipeline::GenerateControl control;
  control.cancel = &cancel;
  control.on_plan = [&](const planner::ExecutionPlan& plan) {
    std::vector<ProgressSegment> segments = generate_segments(plan, impl_->rates(), impl_->held, impl_->model_bytes, shape);
    if (params.plan_callback != nullptr) {
      const std::string json = with_estimate(plan.to_json(), estimate_of(segments, plan, impl_->rates()));
      params.plan_callback(plan.explanation.c_str(), json.c_str(), params.user_data);
    }
    estimate.expect(std::move(segments));
  };
  control.on_phase = [&](pipeline::Phase phase) {
    const double t = since(t0);
    estimate.phase(to_phase(phase), t);
    progress(params, t, to_phase(phase), 0, shape.total_steps, 0.0, estimate);
    return true;
  };
  // A GGUF DiT's conversion, before the plan: MI_PHASE_DIT_LOAD at fraction 0 with no eta (nothing is planned yet).
  control.on_convert = [&](double) {
    progress(params, since(t0), MI_PHASE_DIT_LOAD, 0, shape.total_steps, 0.0, estimate);
    return true;
  };
  control.on_step = [&](const pipeline::StepInfo& s) {
    const double t = since(t0);
    estimate.step(s.step, s.step_seconds, t);
    progress(params, t, MI_PHASE_DENOISE, s.step, s.num_steps, s.step_seconds, estimate);
    return true;
  };
  // Only with a callback: the preview's device buffers count in the denoise phase's plan.
  if (params.preview_callback != nullptr) {
    control.preview_every = params.preview_interval;
    control.on_preview = [&](const pipeline::PreviewImage& im) {
      params.preview_callback(im.rgb, im.width, im.height, im.step, params.user_data);
    };
  }
  pipeline::Result r =
      cold_on_throw(impl_->held, [&] { return model_errors([&] { return impl_->pipe->generate(q, control); }); });
  impl_->hardware = r.hardware;
  impl_->held.job_done(r.plan, shape, false, r.cancelled, r.memory.arena_creations, r.dit_load.source);
  return to_generation(q, r, params.include_prompts != 0);
}

void PipelineAdapter::check(const mi_gen_params& params, CheckFindings& out) {
  // As generate(): the request, the distill bundle's step embedder, the device.
  fs::path distill_te;
  pipeline::Request q = to_request(params, &distill_te, true);
  q.model = impl_->model;
  check_distill_time_embedding(impl_->time_embedding, distill_te);
  const backend::ScopedCurrentContext restore;
  MI_CUDA_CHECK(cudaSetDevice(impl_->device));
  const JobShape shape = job_shape(q);
  pipeline::GenerateControl control;
  if (params.preview_callback != nullptr) {  // the preview's buffers count in the denoise phase's plan
    control.preview_every = params.preview_interval;
    control.on_preview = [](const pipeline::PreviewImage&) {};
  }
  pipeline::Result r;
  try {
    r = model_errors([&] { return impl_->pipe->check(q, control); });
  } catch (const pipeline::GgufNotConverted& e) {
    out.info.gguf_convert_seconds = e.seconds;
    char size[32];
    std::snprintf(size, sizeof size, "%.1f GB", static_cast<double>(e.converted_bytes) / 1e9);
    out.notes.push_back(std::string(e.what()) + " (" + e.runs_as + ", " + size + "; about " +
                        std::to_string(static_cast<long>(std::lround(e.seconds))) +
                        " s); the text encoder, VAE, LoRA and plan checks follow the conversion");
    return;
  }
  const planner::ExecutionPlan& plan = r.plan;
  const Estimate e = estimate_of(generate_segments(plan, impl_->rates(), impl_->held, impl_->model_bytes, shape), plan,
                                 impl_->rates());
  mi_check_info& i = out.info;
  i.planned = 1;
  i.dit_streamed = plan.dit.residency == planner::DitResidency::kStreamed ? 1 : 0;
  i.dit_parked = plan.dit.residency == planner::DitResidency::kParked ? 1 : 0;
  i.dit_resident_blocks = i.dit_streamed != 0 ? plan.dit.resident_blocks : moss_dit::kLayers;
  i.steps = e.steps;
  i.rates_measured = e.rates_measured ? 1 : 0;
  i.arena_bytes = plan.arena_bytes;
  i.arena_cap_bytes = plan.arena_cap_bytes;
  i.device_peak_bytes = plan.device_peak_bytes;
  i.host_private_bytes = plan.host_private_bytes;
  i.host_limit_bytes = plan.host_limit_bytes;
  i.eta_seconds = e.total_s;
  i.text_encoder_seconds = e.text_encoder_s;
  i.dit_load_seconds = e.dit_load_s;
  i.denoise_seconds = e.denoise_s;
  i.step_seconds = e.step_s;
  i.decode_seconds = e.decode_s;
  out.explanation = plan.explanation;
  out.plan_json = with_estimate(plan.to_json(), e);
  out.tier_reason = r.tier.reason;
  for (const std::vector<std::string>* w : {&r.lora_warnings, &r.control.lora.warnings, &r.reference_warnings, &r.accel_warnings}) {
    out.warnings.insert(out.warnings.end(), w->begin(), w->end());
  }
  out.notes = r.plan_only_notes;
}

Generation PipelineAdapter::preload(const mi_gen_params& params, const std::atomic<bool>& cancel) {
  const auto t0 = Clock::now();
  fs::path distill_te;
  pipeline::PreloadOptions options;
  // The images are decoded (load_images true) although the plan needs no pixels: its residencies and workspaces follow
  // the request's size and which images it has, but the plan's explanation names the init image's own size (a 1x1
  // stand-in would show there), and a file that cannot be decoded fails the preload as it would the generation instead
  // of passing it. One decode is small next to the model loads.
  options.request = to_request(params, &distill_te, true, true);
  pipeline::Request& q = options.request;
  q.model = impl_->model;
  q.save_metadata = false;  // no image
  check_distill_time_embedding(impl_->time_embedding, distill_te);
  options.cancel = &cancel;
  const backend::ScopedCurrentContext restore;
  MI_CUDA_CHECK(cudaSetDevice(impl_->device));
  // The plan once, before any load; a progress (step 0, no sampling steps) before each load step, resident models
  // included: kTextEncoder (a host-resident text encoder's copy), kDecode (an image encode's VAE encoder), kDitLoad and
  // kDecode (the VAE decoder). Returning false cancels, as the cancel flag does (a callback's mi_cancel() lands at once).
  // The fraction and eta cover these load steps.
  impl_->sync_held();
  const JobShape shape = job_shape(q);
  ProgressModel estimate;
  options.on_plan = [&](const planner::ExecutionPlan& plan) {
    std::vector<ProgressSegment> segments = preload_segments(plan, impl_->rates(), impl_->held, impl_->model_bytes, shape);
    if (params.plan_callback != nullptr) {
      const std::string json = with_estimate(plan.to_json(), estimate_of(segments, plan, impl_->rates()));
      params.plan_callback(plan.explanation.c_str(), json.c_str(), params.user_data);
    }
    estimate.expect(std::move(segments));
  };
  options.on_phase = [&](pipeline::Phase phase) {
    const double t = since(t0);
    estimate.phase(to_phase(phase), t);
    progress(params, t, to_phase(phase), 0, 0, 0.0, estimate);
    return !cancel.load();
  };
  options.on_convert = [&](double) {
    progress(params, since(t0), MI_PHASE_DIT_LOAD, 0, 0, 0.0, estimate);
    return !cancel.load();
  };
  pipeline::Result r =
      cold_on_throw(impl_->held, [&] { return model_errors([&] { return impl_->pipe->preload(options); }); });
  impl_->hardware = r.hardware;
  impl_->held.job_done(r.plan, shape, true, r.cancelled, r.memory.arena_creations, r.dit_load.source);
  return to_generation(q, r, false);
}

mi_release_info PipelineAdapter::release(bool keep_host_copy, const std::atomic<bool>& cancel, bool* cancelled) {
  const backend::ScopedCurrentContext restore;
  MI_CUDA_CHECK(cudaSetDevice(impl_->device));
  impl_->sync_held();
  const pipeline::ReleaseReport report =
      cold_on_throw(impl_->held, [&] { return impl_->pipe->release_device_memory(keep_host_copy, &cancel); });
  if (!report.cancelled) impl_->held.released(report.dit_host_copy, report.te_host_copy);
  return to_release_info(report, cancelled);
}

pipeline::Pipeline& PipelineAdapter::pipeline() { return *impl_->pipe; }

int PipelineAdapter::device() const { return impl_->device; }

pipeline::Request PipelineAdapter::session_request(const mi_gen_params& params) {
  fs::path distill_te;
  pipeline::Request q = to_request(params, &distill_te, true, true);  // as preload(): the prompt optional
  q.model = impl_->model;
  q.save_metadata = false;
  check_distill_time_embedding(impl_->time_embedding, distill_te);
  return q;
}

void PipelineAdapter::models_changed() { impl_->held_stale.store(true); }

mi_release_info to_release_info(const pipeline::ReleaseReport& rr, bool* cancelled) {
  if (cancelled != nullptr) *cancelled = rr.cancelled;  // the cancel was set on entry: nothing was released
  mi_release_info out{};
  out.struct_size = sizeof out;
  out.seconds = rr.seconds;
  out.vram_released_bytes = rr.vram_released;
  out.host_copy_bytes = rr.host_copy_bytes;
  out.host_copy_seconds = rr.host_copy_s;
  out.dit_host_copy = rr.dit_host_copy ? 1 : 0;
  out.te_host_copy = rr.te_host_copy ? 1 : 0;
  // Cut at 511 bytes, never inside a UTF-8 sequence.
  std::size_t n = std::min(rr.note.size(), sizeof out.note - 1);
  if (n < rr.note.size()) {
    while (n > 0 && (static_cast<unsigned char>(rr.note[n]) & 0xC0) == 0x80) --n;
  }
  std::memcpy(out.note, rr.note.data(), n);
  out.note[n] = '\0';
  return out;
}

std::string report_json(const Generation& generation, const std::string& image_path_utf8) {
  return generation.report_head + pipeline::json_escape(image_path_utf8) + generation.report_tail;
}

void write_png(const Generation& generation, const std::string& path_utf8) {
  if (generation.rgb.empty()) throw std::invalid_argument("the result has no image (the generation was cancelled).");
  std::vector<pipeline::PngText> text;
  for (const auto& [keyword, body] : generation.png_text) text.push_back({keyword, body});
  const fs::path path = user_path(path_utf8.c_str(), "path");
  pipeline::write_png_rgb8(path, generation.rgb, generation.width, generation.height, text);
}

void write_png_rgb(const std::string& path_utf8, const std::uint8_t* rgb, int width, int height) {
  const std::size_t bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
  const fs::path path = user_path(path_utf8.c_str(), "path");
  pipeline::write_png_rgb8(path, std::span<const std::uint8_t>(rgb, bytes), width, height);
}

}  // namespace mi::api
