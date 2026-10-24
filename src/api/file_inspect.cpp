#include "api/file_inspect.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <stdexcept>
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
#include "pipeline/pipeline.h"
#include "planner/planner.h"
#include "util/json.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"

namespace mi::api {

namespace {

namespace fs = std::filesystem;

// The embedder bytes inspection lets a control adapter's loader read, and its bf16 copy: half this each. Far above what
// real files need (the published control adapter's embedder: 2.4 MB); open_control_file takes any number of control
// groups, so its header bounds nothing. A DiT's and a text encoder's reads need no cap: quant::open_quant_model refuses
// every linear and text-encoder tensor without MOSS-Image's shape before any is read, which bounds a DiT's scales (at
// most about 290 MiB in all, for a fully NVFP4 file; to_model_index holds one linear's at a time) and a text encoder's
// scales and F32 tensors (a few MB).
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

// The file's size, only when it is a regular file: util::open_regular_file reads the path's type before opening it
// (opening a device runs its driver) and opens it without blocking (a FIFO's open would wait for a writer), as the
// loaders' readers do when they reopen the file by name.
std::uint64_t regular_file_bytes(const fs::path& path) {
  const std::string where = util::path_utf8(path);
  const auto not_regular = [&] { return std::invalid_argument(where + ": " + util::kNotRegularFile); };
#ifdef _WIN32
  const auto io_error = [&](DWORD e) {
    return weights::IoError(where + ": " + (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND
                                                ? std::string("no such file")
                                                : std::system_category().message(static_cast<int>(e))));
  };
  // util::open_regular_file refuses the \\.\ device namespace unopened; \\?\GLOBALROOT\ paths name devices too.
  if (path.native().starts_with(L"\\\\?\\GLOBALROOT")) throw not_regular();
  void* handle = nullptr;
  const util::OpenStatus status = util::open_regular_file(path, FILE_ATTRIBUTE_NORMAL, handle);
  if (status == util::OpenStatus::kNotRegular) throw not_regular();
  if (status != util::OpenStatus::kOk) throw io_error(GetLastError());
  LARGE_INTEGER size{};
  const bool sized = GetFileSizeEx(static_cast<HANDLE>(handle), &size) != 0;
  const DWORD size_error = sized ? 0 : GetLastError();
  CloseHandle(static_cast<HANDLE>(handle));
  if (!sized) throw io_error(size_error);
  return static_cast<std::uint64_t>(size.QuadPart);
#else
  const auto io_error = [&](int e) {
    return weights::IoError(where + ": " + (e == ENOENT || e == ENOTDIR ? std::string("no such file")
                                                                       : std::generic_category().message(e)));
  };
  int fd = -1;
  const util::OpenStatus status = util::open_regular_file(path, 0, fd);
  if (status == util::OpenStatus::kNotRegular) throw not_regular();
  if (status != util::OpenStatus::kOk) throw io_error(errno);
  struct stat st {};
  const int stat_error = ::fstat(fd, &st) == 0 ? 0 : errno;
  ::close(fd);
  if (stat_error != 0) throw io_error(stat_error);
  return static_cast<std::uint64_t>(st.st_size);
#endif
}

std::string mib(std::uint64_t bytes) { return std::to_string((bytes + (std::uint64_t{1} << 20) - 1) >> 20) + " MiB"; }

// The LoRA stack's reading of `spec` (a LoRA, or a control adapter's LoRA part), as the pipeline builds it.
bool inspect_lora(const lora::LoraSpec& spec, FileInspection& r) {
  std::optional<lora::LoraStack> stack;
  if (!accepts([&] { stack.emplace(std::vector<lora::LoraSpec>{spec}, lora::moss_dit_targets()); }, r.note)) {
    return false;
  }
  const lora::FileReport& rep = stack->reports().front();
  r.info.lora_adapters = rep.adapters;
  r.info.lora_unknown_keys = static_cast<std::int32_t>(rep.unknown.size());
  if (!rep.unknown.empty()) {
    r.note = std::to_string(rep.unknown.size()) + " tensors not applied, first: " + rep.unknown.front();
  }
  std::vector<std::string> types;
  for (const lora::Adapter& a : stack->adapters()) {
    if (a.kind == lora::AdapterKind::kLowRank) {
      r.info.lora_rank = std::max(r.info.lora_rank, static_cast<std::int32_t>(a.rank));
    }
    const std::string type = lora::kind_name(a.kind);
    if (std::find(types.begin(), types.end(), type) == types.end()) types.push_back(type);
  }
  for (const std::string& t : types) r.adapter_types += (r.adapter_types.empty() ? "" : ",") + t;
  return true;
}

// pipeline::inspect_dit_file, as the pipeline loads a DiT file, without the device: NVFP4 linears are judged for the
// best device (an RTX 50) with this build's kernels; the tiers are planner::resolve_tier's verdicts, with every
// attention kernel available.
void inspect_dit(const fs::path& path, FileInspection& r) {
  r.info.kind = MI_FILE_KIND_DIT;
  std::optional<pipeline::DitFileInfo> dit;
  const auto nvfp4_refusal = [] { return moss_dit::nvfp4_unsupported_reason(12, 0, kNvfp4Kernels); };
  if (!accepts([&] { dit.emplace(pipeline::inspect_dit_file(path, nvfp4_refusal)); }, r.note)) return;
  if (!dit->refusal.empty()) {
    r.note = dit->refusal;
    return;
  }
  describe_dit_linears(*dit, r);
  r.info.loadable = 1;
  r.info.precision = to_precision(dit->precision);
  const planner::CertificateLookup no_certificate = [](planner::Precision, planner::AttentionMode) {
    return planner::CertificateAnswer{false, "", "no certificate looked up"};
  };
  constexpr planner::Tier kTiers[] = {planner::Tier::kAuto, planner::Tier::kReference, planner::Tier::kHigh,
                                      planner::Tier::kTurbo};  // by mi_tier
  for (std::size_t t = 0; t < std::size(kTiers); ++t) {
    try {
      r.tier_reasons[t] = planner::resolve_tier(kTiers[t], std::nullopt, dit->precision, no_certificate, {}).reason;
      r.info.tiers |= 1u << t;
    } catch (const std::invalid_argument& e) {
      r.tier_reasons[t] = e.what();
    }
  }
}

void inspect_text_encoder(const fs::path& path, FileInspection& r) {
  r.info.kind = MI_FILE_KIND_TEXT_ENCODER;
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

void describe_dit_linears(const pipeline::DitFileInfo& dit, FileInspection& r) {
  r.quantization = dit.linears;
  r.note = dit.nvfp4 ? "NVFP4 linears run on compute capability 12.0 only (an RTX 50 GPU)" : "";
  if (!dit.remarks.empty()) r.note += (r.note.empty() ? "" : "; ") + dit.remarks;
}

FileInspection inspect_file(const fs::path& path) {
  FileInspection r;
  r.info.struct_size = sizeof r.info;
  r.info.kind = MI_FILE_KIND_UNKNOWN;
  r.info.precision = MI_PRECISION_BF16;
  r.info.file_bytes = regular_file_bytes(path);
  for (std::string& reason : r.tier_reasons) reason = "not a loadable DiT file";

  std::optional<weights::SafetensorsFile> header;
  std::string header_why;
  if (!accepts([&] { header.emplace(path); }, header_why)) {
    // check_supported_dit_file's path form names other containers (GGUF, pickles) before it reads the header.
    std::string container_why;
    r.note = accepts([&] { weights::check_supported_dit_file(path); }, container_why) ? header_why : container_why;
    return r;
  }
  r.info.tensor_count = static_cast<std::int32_t>(header->tensors().size());
  // check_supported_dit_file names the add-on roles (LoRA, control adapter, distill bundle file); its message describes
  // a file no loader accepts.
  std::string dit_why;
  const bool dit = accepts([&] { weights::check_supported_dit_file(*header); }, dit_why);

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
  if (accepts([&] { weights::check_supported_text_encoder_file(*header); }, why)) {
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
  if (lora || weights::looks_like_lora(*header)) {
    // A file in the LoRA conventions is a LoRA, also one the stack refuses (for another model, or malformed): its
    // message says why.
    r = std::move(as_lora);
    r.info.kind = MI_FILE_KIND_LORA;
    r.info.loadable = lora ? 1 : 0;
    const auto& meta = header->metadata();
    if (const auto it = meta.find("base_model"); it != meta.end()) r.base_model = it->second;
    return r;
  }
  r.note = dit_why;
  return r;
}

}  // namespace mi::api
