#include "api/file_inspect.h"

#include <algorithm>
#include <exception>
#include <iterator>
#include <new>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "models/control/control.h"
#include "models/flux2_vae/vae_file.h"
#include "models/lora/lora.h"
#include "models/moss_dit/moss_dit.h"
#include "models/moss_dit/nvfp4_support.h"
#include "models/qwen35_text/quant_text_model.h"
#include "planner/planner.h"
#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"

namespace mi::api {

namespace {

namespace fs = std::filesystem;

// One loader's verdict on the file: true when it accepts it, else false with its message in `why`. An I/O error or an
// allocation failure is not a verdict on the file: it propagates.
template <class F>
bool accepts(F&& load, std::string& why) {
  try {
    load();
    return true;
  } catch (const weights::IoError&) {
    throw;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    why = e.what();
    return false;
  }
}

// The LoRA stack's reading of `spec` (a LoRA, or a control adapter's LoRA part), as the pipeline builds it.
bool inspect_lora(const lora::LoraSpec& spec, FileInspection& r) {
  std::optional<lora::LoraStack> stack;
  if (!accepts([&] { stack.emplace(std::vector<lora::LoraSpec>{spec}, lora::moss_dit_targets()); }, r.note)) return false;
  const lora::FileReport& rep = stack->reports().front();
  r.info.lora_adapters = rep.adapters;
  r.info.lora_unknown_keys = static_cast<std::int32_t>(rep.unknown.size());
  if (!rep.unknown.empty()) r.note = std::to_string(rep.unknown.size()) + " tensors not applied, first: " + rep.unknown.front();
  std::vector<std::string> types;
  for (const lora::Adapter& a : stack->adapters()) {
    if (a.kind == lora::AdapterKind::kLowRank) r.info.lora_rank = std::max(r.info.lora_rank, static_cast<std::int32_t>(a.rank));
    const std::string type = lora::kind_name(a.kind);
    if (std::find(types.begin(), types.end(), type) == types.end()) types.push_back(type);
  }
  for (const std::string& t : types) r.adapter_types += (r.adapter_types.empty() ? "" : ",") + t;
  return true;
}

// open_dit_file's (pipeline.cpp) steps after check_supported_dit_file, without the device: NVFP4 linears are judged
// for the best device (an RTX 50 with the kernels); the tiers are planner::resolve_tier's verdicts.
void inspect_dit(const fs::path& path, FileInspection& r) {
  r.info.kind = MI_FILE_KIND_DIT;
  std::optional<quant::QuantModel> model;
  if (!accepts([&] { model.emplace(quant::open_quant_model(path, quant::ModelKind::kDiT)); }, r.note)) return;
  if (std::any_of(model->linears.begin(), model->linears.end(),
                  [](const quant::QuantLinear& l) { return l.spec.format == quant::QuantFormat::kNvfp4; })) {
    if (const std::string why = moss_dit::nvfp4_unsupported_reason(12, 0, true); !why.empty()) {
      r.note = util::path_utf8(path) + ": " + why;
      return;
    }
    r.note = "NVFP4 linears run on compute capability 12.0 only (an RTX 50 GPU)";
  }
  std::optional<quant::QuantIndex> index;
  if (!accepts([&] { index.emplace(quant::to_model_index(*model, {.native_int4 = true})); }, r.note)) return;
  const planner::Precision precision = moss_dit::has_int4_linears(&*index)        ? planner::Precision::kInt4
                                       : moss_dit::has_quantized_linears(&*index) ? planner::Precision::kInt8
                                                                                  : planner::Precision::kBf16;
  r.info.loadable = 1;
  r.info.precision = precision == planner::Precision::kInt4   ? MI_PRECISION_W4A4
                     : precision == planner::Precision::kInt8 ? MI_PRECISION_INT8
                                                              : MI_PRECISION_BF16;
  r.quantization = moss_dit::describe_linears(&*index);
  const planner::CertificateLookup no_certificate = [](planner::Precision, planner::AttentionMode) {
    return planner::CertificateAnswer{false, "", "no certificate looked up"};
  };
  constexpr planner::Tier kTiers[] = {planner::Tier::kAuto, planner::Tier::kReference, planner::Tier::kHigh,
                                      planner::Tier::kTurbo};  // by mi_tier
  for (std::size_t t = 0; t < std::size(kTiers); ++t) {
    try {
      r.tier_reasons[t] = planner::resolve_tier(kTiers[t], std::nullopt, precision, no_certificate, {}).reason;
      r.info.tiers |= 1u << t;
    } catch (const std::invalid_argument& e) {
      r.tier_reasons[t] = e.what();
    }
  }
}

void inspect_text_encoder(const fs::path& path, FileInspection& r) {
  r.info.kind = MI_FILE_KIND_TEXT_ENCODER;
  std::optional<qwen35_text::QuantTextModel> model;
  if (!accepts([&] { model.emplace(path); }, r.note)) return;
  const qwen35_text::Precision p = model->precision();
  r.info.loadable = 1;
  r.info.precision = p == qwen35_text::Precision::kInt8   ? MI_PRECISION_INT8
                     : p == qwen35_text::Precision::kW4A8 ? MI_PRECISION_W4A8
                                                          : MI_PRECISION_BF16;
  r.quantization = qwen35_text::precision_name(p);
}

void inspect_control(const fs::path& path, FileInspection& r) {
  r.info.kind = MI_FILE_KIND_CONTROL_ADAPTER;
  std::optional<control::ControlFile> file;
  if (!accepts([&] { file.emplace(control::open_control_file(path)); }, r.note)) return;
  r.control_kinds = file->kinds;
  r.info.control_kind_count = file->groups();
  r.base_model = file->base_model;
  r.info.loadable = inspect_lora({path, 1.0f, file->consumed()}, r) ? 1 : 0;
}

}  // namespace

FileInspection inspect_file(const fs::path& path) {
  std::error_code ec;
  const fs::file_status status = fs::status(path, ec);
  if (!fs::exists(status)) throw weights::IoError(util::path_utf8(path) + ": no such file");
  // Not a directory, device or pipe: reading one could block or never end.
  if (!fs::is_regular_file(status)) throw std::invalid_argument(util::path_utf8(path) + " is not a regular file");
  FileInspection r;
  r.info.struct_size = sizeof r.info;
  r.info.kind = MI_FILE_KIND_UNKNOWN;
  r.info.precision = MI_PRECISION_BF16;
  r.info.file_bytes = fs::file_size(path);
  for (std::string& reason : r.tier_reasons) reason = "not a loadable DiT file";

  // check_supported_dit_file names other containers (GGUF, pickles) and the add-on roles (LoRA, control adapter,
  // distill bundle file); its message describes a file no loader accepts.
  std::string dit_why;
  const bool dit = accepts([&] { weights::check_supported_dit_file(path); }, dit_why);
  std::optional<weights::SafetensorsFile> header;
  std::string header_why;
  if (!accepts([&] { header.emplace(path); }, header_why)) {
    r.note = dit_why;
    return r;
  }
  r.info.tensor_count = static_cast<std::int32_t>(header->tensors().size());

  // The roles in the pipeline's order of precedence: a control adapter is refused as a LoRA, then each model file.
  if (control::is_control_file(path)) {
    inspect_control(path, r);
    return r;
  }
  if (dit) {
    inspect_dit(path, r);
    return r;
  }
  std::string why;
  if (accepts([&] { weights::check_supported_text_encoder_file(path); }, why)) {
    inspect_text_encoder(path, r);
    return r;
  }
  if (accepts([&] { flux2_vae::index_vae_file(path); }, why)) {
    r.info.kind = MI_FILE_KIND_VAE;
    r.info.loadable = 1;
    return r;
  }
  FileInspection as_lora = r;
  if (inspect_lora({path, 1.0f, {}}, as_lora)) {
    r = std::move(as_lora);
    r.info.kind = MI_FILE_KIND_LORA;
    r.info.loadable = 1;
    if (const auto it = header->metadata().find("base_model"); it != header->metadata().end()) r.base_model = it->second;
    return r;
  }
  r.note = dit_why;
  return r;
}

}  // namespace mi::api
