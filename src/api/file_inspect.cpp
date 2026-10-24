#include "api/file_inspect.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#endif

#include "api/pipeline_adapter.h"
#include "models/control/control.h"
#include "models/flux2_vae/vae_file.h"
#include "models/lora/lora.h"
#include "models/moss_dit/moss_dit.h"
#include "models/moss_dit/nvfp4_support.h"
#include "models/qwen35_text/quant_text_model.h"
#include "planner/planner.h"
#include "util/json.h"
#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"

namespace mi::api {

namespace {

namespace fs = std::filesystem;

// The tensor bytes inspection lets the loaders read and the scale buffers it lets them allocate, beyond the header
// (at most weights::kMaxHeaderBytes): a header cannot make it read or allocate more. Far above what real files need
// (the scales of an INT8 or INT4 DiT: a few MB; an NVFP4 DiT's block scales: about 130 MB).
constexpr std::uint64_t kMaxTensorReads = std::uint64_t{256} << 20;

// Whether the sm_120a NVFP4 GEMMs are compiled in (cmake/modules/nvfp4.cmake defines this for this file as for
// kernels/nvfp4).
#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
constexpr bool kNvfp4Kernels = true;
#else
constexpr bool kNvfp4Kernels = false;
#endif

// One loader's verdict on the file: true when it accepts it, else false with its message in `why`. Only the loaders'
// verdicts on the file's content are caught; an I/O error, an allocation failure or a logic error propagates.
template <class F>
bool accepts(F&& load, std::string& why) {
  try {
    load();
    return true;
  } catch (const weights::FormatError& e) {
    why = e.what();
  } catch (const weights::IndexError& e) {  // UnsupportedModelError included
    why = e.what();
  } catch (const quant::QuantError& e) {
    why = e.what();
  } catch (const lora::LoraError& e) {
    why = e.what();
  } catch (const control::ControlError& e) {
    why = e.what();
  } catch (const json::ParseError& e) {
    why = e.what();
  } catch (const std::invalid_argument& e) {
    why = e.what();
  }
  return false;
}

// The file's size, from a handle opened without blocking (a FIFO's open would wait for a writer) and only when it is
// a regular file: a directory, device or pipe is never handed to the loaders. They reopen the file by name, so a path
// swapped for a pipe in between is the readers' to refuse.
std::uint64_t regular_file_bytes(const fs::path& path) {
  const std::string where = util::path_utf8(path);
#ifdef _WIN32
  const HANDLE h = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    const DWORD e = GetLastError();
    throw weights::IoError(where + ": " + (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND
                                               ? std::string("no such file")
                                               : std::system_category().message(static_cast<int>(e))));
  }
  BY_HANDLE_FILE_INFORMATION info{};
  const bool disk = GetFileType(h) == FILE_TYPE_DISK && GetFileInformationByHandle(h, &info) != 0 &&
                    (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
  CloseHandle(h);
  if (!disk) throw std::invalid_argument(where + " is not a regular file");
  return (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    const int e = errno;
    throw weights::IoError(where + ": " + (e == ENOENT || e == ENOTDIR ? std::string("no such file")
                                                                      : std::generic_category().message(e)));
  }
  struct stat st {};
  const int stat_error = ::fstat(fd, &st) == 0 ? 0 : errno;
  ::close(fd);
  if (stat_error != 0) throw weights::IoError(where + ": " + std::generic_category().message(stat_error));
  if (!S_ISREG(st.st_mode)) throw std::invalid_argument(where + " is not a regular file");
  return static_cast<std::uint64_t>(st.st_size);
#endif
}

// Tensor names in the conventions the LoRA stack reads (the ones check_supported_dit_file names a LoRA by).
bool has_lora_keys(const weights::SafetensorsFile& header) {
  for (const weights::TensorInfo& t : header.tensors()) {
    for (const std::string_view part : {".lora_A.", ".lora_B.", ".lora_down.", ".lora_up.", ".hada_w", ".lokr_w"}) {
      if (t.name.find(part) != std::string::npos) return true;
    }
    if (t.name.ends_with(".alpha")) return true;
  }
  return false;
}

std::string mib(std::uint64_t bytes) { return std::to_string((bytes + (std::uint64_t{1} << 20) - 1) >> 20) + " MiB"; }

// What the importer's readers (to_model_index, QuantTextModel) read before anything is uploaded, from the header
// alone: the stored scales and codebooks, a text encoder's F32 tensors (and their bf16 copies), and a float buffer of
// the linear's rows per scale. Throws std::invalid_argument when that exceeds kMaxTensorReads.
void check_reads(const quant::QuantModel& m) {
  std::uint64_t total = 0;
  const auto add = [&](std::uint64_t bytes) { total = bytes > kMaxTensorReads ? kMaxTensorReads + 1 : total + bytes; };
  for (const quant::QuantLinear& l : m.linears) {
    for (const std::optional<weights::IndexedTensor>* t : {&l.scale, &l.scale_2, &l.s_channel, &l.codebook}) {
      if (t->has_value()) add((*t)->nbytes);
    }
    if (l.scale || l.s_channel) {
      const bool rows_fit = l.n > 0 && static_cast<std::uint64_t>(l.n) <= kMaxTensorReads / 4;
      add(rows_fit ? 4 * static_cast<std::uint64_t>(l.n) : kMaxTensorReads + 1);
    }
    if (total > kMaxTensorReads) break;
  }
  for (const weights::IndexedTensor& t : m.tensors) {
    if (m.kind == quant::ModelKind::kTextEncoder && t.dtype == weights::DType::kF32) add(t.nbytes + t.nbytes / 2);
  }
  if (total > kMaxTensorReads) {
    throw std::invalid_argument("the file's scales and small tensors take more than " + mib(kMaxTensorReads) +
                                ", more than any MOSS-Image file; not read");
  }
}

std::string shape_text(std::int64_t n, std::int64_t k) { return "[" + std::to_string(n) + ", " + std::to_string(k) + "]"; }

// Every GEMM of a DiT file against MOSS-Image's (lora::moss_dit_targets), and each stored scale against the linear's
// rows, before to_model_index reads any of them. Throws std::invalid_argument naming the first difference.
void check_dit_linears(const quant::QuantModel& m) {
  const lora::TargetModel targets = lora::moss_dit_targets();
  const auto round_up = [](std::int64_t v, std::int64_t to) { return (v + to - 1) / to * to; };
  for (const quant::QuantLinear& l : m.linears) {
    const lora::TargetModule* t = targets.find(l.name);
    if (t == nullptr) throw std::invalid_argument("linear '" + l.name + "' is not a MOSS-Image DiT linear");
    if (t->n != l.n || t->k != l.k) {
      throw std::invalid_argument("linear '" + l.name + "' is " + shape_text(l.n, l.k) + "; MOSS-Image's is " +
                                  shape_text(t->n, t->k));
    }
    std::int64_t rows_scale = l.n;  // per-row scales: raw INT8, W4A4, the grouped formats' s_channel
    if (l.encoding == quant::Encoding::kScalarScale) rows_scale = 1;
    if (l.encoding == quant::Encoding::kNvfp4Raw) rows_scale = round_up(l.n, 128) * round_up(l.k / 16, 4);
    const auto check = [&](const std::optional<weights::IndexedTensor>& s, std::int64_t want, const char* what) {
      if (s && s->numel() != want) {
        throw std::invalid_argument("linear '" + l.name + "' has a " + what + " of " + std::to_string(s->numel()) +
                                    " values; its shape needs " + std::to_string(want));
      }
    };
    if (l.encoding == quant::Encoding::kW4A8 || l.encoding == quant::Encoding::kW6A8) {
      check(l.s_channel, l.n, "weight_s_channel");
      check(l.codebook, 16, "weight_codebook");
    } else {
      check(l.scale, rows_scale, "weight_scale");
    }
    check(l.scale_2, 1, "weight_scale_2");
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
// for the best device (an RTX 50) with this build's kernels; the tiers are planner::resolve_tier's verdicts, with
// every attention kernel available.
void inspect_dit(const fs::path& path, FileInspection& r) {
  r.info.kind = MI_FILE_KIND_DIT;
  std::optional<quant::QuantModel> model;
  if (!accepts([&] { model.emplace(quant::open_quant_model(path, quant::ModelKind::kDiT)); }, r.note)) return;
  if (!accepts([&] { check_dit_linears(*model); }, r.note) || !accepts([&] { check_reads(*model); }, r.note)) return;
  std::string remarks;
  if (std::any_of(model->linears.begin(), model->linears.end(),
                  [](const quant::QuantLinear& l) { return l.spec.format == quant::QuantFormat::kNvfp4; })) {
    if (const std::string why = moss_dit::nvfp4_unsupported_reason(12, 0, kNvfp4Kernels); !why.empty()) {
      r.note = util::path_utf8(path) + ": " + why;
      return;
    }
    remarks = "NVFP4 linears run on compute capability 12.0 only (an RTX 50 GPU)";
  }
  const auto static_input_scales = std::count_if(model->linears.begin(), model->linears.end(),
                                                 [](const quant::QuantLinear& l) { return l.input_scale.has_value(); });
  if (static_input_scales > 0) {
    remarks += (remarks.empty() ? "" : "; ") + std::to_string(static_input_scales) +
               " linears carry a static input_scale; ignored (activations are scaled per token)";
  }
  std::optional<quant::QuantIndex> index;
  if (!accepts([&] { index.emplace(quant::to_model_index(*model, {.native_int4 = true})); }, r.note)) return;
  const planner::Precision precision = moss_dit::has_int4_linears(&*index)        ? planner::Precision::kInt4
                                       : moss_dit::has_quantized_linears(&*index) ? planner::Precision::kInt8
                                                                                  : planner::Precision::kBf16;
  r.note = remarks;
  r.info.loadable = 1;
  r.info.precision = to_precision(precision);
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
  std::optional<quant::QuantModel> model;
  if (!accepts([&] { model.emplace(quant::open_quant_model(path, quant::ModelKind::kTextEncoder)); }, r.note) ||
      !accepts([&] { check_reads(*model); }, r.note)) {
    return;
  }
  model.reset();
  std::optional<qwen35_text::QuantTextModel> te;
  if (!accepts([&] { te.emplace(path); }, r.note)) return;
  r.info.loadable = 1;
  r.info.precision = to_precision(te->precision());
  r.quantization = qwen35_text::precision_name(te->precision());
}

void inspect_control(const fs::path& path, const weights::SafetensorsFile& header, FileInspection& r) {
  r.info.kind = MI_FILE_KIND_CONTROL_ADAPTER;
  std::optional<control::ControlFile> file;
  if (!accepts([&] { file.emplace(control::open_control_file(path)); }, r.note)) return;
  r.control_kinds = file->kinds;
  r.info.control_kind_count = file->groups();
  r.base_model = file->base_model;
  // read_embedder reads the embedder and holds its bf16 copy: within the read bound, then checked as generate does.
  const weights::TensorInfo* w = header.find(control::kEmbedderWeight);
  const weights::TensorInfo* b = header.find(control::kEmbedderBias);
  if (w == nullptr || b == nullptr || w->nbytes + b->nbytes > kMaxTensorReads / 2) {
    r.note = "the control adapter's embedder takes more than " + mib(kMaxTensorReads / 2) + "; not read";
    return;
  }
  if (!accepts([&] { control::read_embedder(*file); }, r.note)) return;
  r.info.loadable = inspect_lora({path, 1.0f, file->consumed()}, r) ? 1 : 0;
}

}  // namespace

FileInspection inspect_file(const fs::path& path) {
  FileInspection r;
  r.info.struct_size = sizeof r.info;
  r.info.kind = MI_FILE_KIND_UNKNOWN;
  r.info.precision = MI_PRECISION_BF16;
  r.info.file_bytes = regular_file_bytes(path);
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
    inspect_control(path, *header, r);
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
  const bool lora = inspect_lora({path, 1.0f, {}}, as_lora);
  if (lora || has_lora_keys(*header)) {
    // A file in the LoRA conventions is a LoRA, also one the stack refuses (for another model, or malformed): its
    // message says why.
    r = std::move(as_lora);
    r.info.kind = MI_FILE_KIND_LORA;
    r.info.loadable = lora ? 1 : 0;
    if (const auto it = header->metadata().find("base_model"); it != header->metadata().end()) r.base_model = it->second;
    return r;
  }
  r.note = dit_why;
  return r;
}

}  // namespace mi::api
