#include "pipeline/pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include "core/cuda_check.h"
#include "kernels/nvfp4/nvfp4.h"
#include "mossinfer/mossinfer.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/flux2_vae/vae_file.h"
#include "models/control/control.h"
#include "models/flux2_vae_encoder/flux2_vae_encoder.h"
#include "models/latent_preview/latent_preview.h"
#include "models/moss_dit/dit_kernels.h"
#include "models/moss_dit/moss_dit.h"
#include "models/moss_dit/nvfp4_support.h"
#include "models/qwen35_text/text_encoder.h"
#include "pipeline/embedding_cache.h"
#include "pipeline/hardware_profile.h"
#include "pipeline/quality_report.h"
#include "pipeline/report_json.h"
#include "pipeline/rng_torch_compat.h"
#include "planner/host_memory.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "tokenizer/qwen_bpe.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"
#include "weights/upload.h"

namespace mi::pipeline {
namespace {

using Clock = std::chrono::steady_clock;
using runtime::align_up;

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

moss_dit::SkipSet skip_set(const std::vector<int>& layers) {
  moss_dit::SkipSet s;
  for (const int l : layers) s.set(static_cast<std::size_t>(l));
  return s;
}

void check_layers(const std::vector<int>& layers, const char* name) {
  for (std::size_t i = 0; i < layers.size(); ++i) {
    if (layers[i] < 0 || std::count(layers.begin(), layers.end(), layers[i]) != 1) {
      throw std::invalid_argument("SLG indices must be nonnegative and unique.");
    }
    if (layers[i] >= moss_dit::kLayers) {
      throw std::invalid_argument(std::string(name) + " must contain integer block indices in [0, " +
                                  std::to_string(moss_dit::kLayers) + ").");
    }
  }
}

std::string_view trim_ascii(std::string_view s) {
  const auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
  while (!s.empty() && ws(s.front())) s.remove_prefix(1);
  while (!s.empty() && ws(s.back())) s.remove_suffix(1);
  return s;
}

// A skip-layer list from parameters.json: a "22,24" string or an integer array.
std::vector<int> parse_layers(const json::Value& v, const char* name) {
  if (v.is_null()) return {};
  if (v.is_string()) return parse_layer_list(v.as_string());
  if (!v.is_array()) throw std::invalid_argument(std::string(name) + " must be a comma-separated string or an integer array.");
  std::vector<int> out;
  for (const auto& e : v.as_array()) {
    if (!e.is_int() || e.as_int() < std::numeric_limits<int>::min() || e.as_int() > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(name) + " must contain integers.");
    }
    out.push_back(static_cast<int>(e.as_int()));
  }
  return out;
}

struct Bundle {
  std::filesystem::path dir;
  weights::ModelIndex index;  // without the components left out for the pipeline's component files
  tok::Tokenizer tokenizer;
  std::uint64_t te_hash = 0;  // embedding-cache identity of the text encoder's weights (0: not indexed)
};

// The bundle's index without the components in `skip` (weights::BuildOptions::skip; a .mossidx is read the same way)
// and its tokenizer. The folder has passed check_supported_bundle.
std::unique_ptr<Bundle> open_bundle(const std::filesystem::path& dir, weights::ComponentSet skip) {
  const std::filesystem::path idx = dir / "moss.mossidx";
  weights::ModelIndex index = std::filesystem::is_regular_file(idx) ? weights::load_index(idx, dir, skip)
                                                                    : weights::build_index(dir, {.skip = skip});
  auto b = std::unique_ptr<Bundle>(new Bundle{dir, std::move(index), tok::Tokenizer::load(dir / "tokenizer" / "tokenizer.json"), 0});
  if (!skip.has(weights::Component::kTextEncoder)) b->te_hash = text_encoder_hash(b->index);
  return b;
}

// Where the DiT's weights come from (P6): the bundle's transformer, or a single file (quant::to_model_index; INT8,
// mixed or bf16). `index` is what MossDit and the planner read: the bundle's index or quant->index.
struct DitSource {
  std::filesystem::path path;  // the bundle's transformer directory or the file
  const weights::ModelIndex* index = nullptr;
  std::unique_ptr<quant::QuantIndex> quant;  // single file only
  planner::Precision precision = planner::Precision::kBf16;
  std::string linears;  // moss_dit::describe_linears
  std::string remarks;  // DitFileInfo::remarks (single file only)
  // The weight files the DiT is read from, and whether direct I/O gains on them (weights::direct_io_support of each;
  // the first that does not decides), checked once per source.
  std::vector<std::filesystem::path> files;
  mutable std::optional<weights::DirectIoSupport> direct;
  // An O_DIRECT open of the files was refused (the load read buffered): direct I/O is not explored on them again in
  // this process (choose_dit_load_io).
  mutable bool direct_refused = false;
  // The model it was trained as (dit_lineage): "base", "sft" or "" (unknown).
  std::string lineage;
};

}  // namespace

std::string dit_lineage(const weights::ModelIndex& index) {
  constexpr std::uint64_t kBase = 0x06b5ba6b83f6335cULL, kSft = 0x8b85f7f1ea6f9562ULL;
  const weights::WeightGroup* g = index.find_group(weights::Component::kTransformer, "globals");
  if (g == nullptr) return {};
  for (const weights::IndexedTensor& t : g->tensors) {
    if (t.name != "video_embedder.weight" || t.nbytes > (std::uint64_t{16} << 20)) continue;
    try {
      std::vector<std::byte> bytes(static_cast<std::size_t>(t.nbytes));
      weights::FileReader(index.file_path(t.file)).read_at(t.offset, bytes.data(), bytes.size());
      const std::uint64_t h = xxh64(bytes.data(), bytes.size());
      return h == kBase ? "base" : h == kSft ? "sft" : "";
    } catch (const std::exception&) {
      return {};  // unreadable weights fail at the load, with a better error
    }
  }
  return {};
}

bool merge_memo_unaffordable(const planner::ExecutionPlan& plan, std::size_t host_budget) {
  return plan.host_private_bytes > plan.host_held_target_bytes || (host_budget != 0 && plan.host_private_bytes > host_budget);
}

DitIoDecision choose_dit_load_io(const DitIoInputs& in) {
  const bool warm = in.cached_fraction >= kDitCachedFraction;
  if (in.forced_buffered) return {false, "buffered reads requested", warm};
  if (in.forced_direct) {
    return {true, in.support.supported ? std::string() : "direct I/O requested; " + in.support.reason, warm};
  }
  if (!in.support.supported) return {false, in.support.reason, warm};
  char text[240];
  // Both warm rates measured: the faster mode, unless the hysteresis keeps the one in use (said so).
  const bool both = in.direct_gbps > 0 && in.buffered_warm_gbps > 0;
  const bool warm_direct = both && planner::warm_dit_load_direct(in.direct_gbps, in.buffered_warm_gbps, in.warm_direct_before);
  std::string rates;
  if (both) {
    const double chosen = warm_direct ? in.direct_gbps : in.buffered_warm_gbps;
    const double other = warm_direct ? in.buffered_warm_gbps : in.direct_gbps;
    std::snprintf(text, sizeof(text), "warm: %s measured %.1f GB/s vs %s %.1f GB/s", warm_direct ? "direct" : "buffered", chosen,
                  warm_direct ? "buffered" : "direct", other);
    rates = text;
    if (chosen < other) {
      std::snprintf(text, sizeof(text), " (kept: within %.0f%% of it)", 100.0 * planner::kWarmLoadHysteresis);
      rates += text;
    }
  }
  if (in.parked) {
    if (warm_direct) return {true, "a parking reload; " + rates, warm};
    return {false, "a parking reload: buffered, so the page cache keeps the DiT's files for the next one", warm};
  }
  if (warm) {
    std::snprintf(text, sizeof(text), "%.0f%% of the DiT's files are in the page cache", 100.0 * in.cached_fraction);
    const std::string resident = text;
    if (both) return {warm_direct, resident + "; " + rates, true};
    // One mode measured: the other, once, so both are known from then on (a load too small to measure explores nothing).
    const bool measured = in.bytes >= kDitLoadMeasureBytes;
    if (in.direct_gbps > 0 && measured) {
      std::snprintf(text, sizeof(text), "; warm: direct measured %.1f GB/s, buffered not yet: buffered this once", in.direct_gbps);
      return {false, resident + text, true};
    }
    if (in.buffered_warm_gbps > 0 && measured) {
      std::snprintf(text, sizeof(text),
                    in.direct_refused ? "; warm: buffered measured %.1f GB/s, direct not tried: an O_DIRECT open of these files was refused"
                                      : "; warm: buffered measured %.1f GB/s, direct not yet: direct this once",
                    in.buffered_warm_gbps);
      return {!in.direct_refused, resident + text, true};
    }
    return {false, resident + ": buffered reads serve them from RAM", true};
  }
  if (in.cached_fraction >= 0) return {true, {}};
  // Residency unknown (Windows, or a file the process may not query): the learner's measured rates decide, else the
  // cold default.
  if (in.buffered_gbps > 0 && in.direct_gbps > 0 && in.buffered_gbps > in.direct_gbps) {
    std::snprintf(text, sizeof(text), "buffered loads measured faster here (%.1f GB/s against %.1f GB/s direct)",
                  in.buffered_gbps, in.direct_gbps);
    return {false, text};
  }
  return {true, "page-cache residency unknown: direct, the cold-cache choice"};
}

namespace {

// The DiT weight files of `index` (its transformer groups' files; a single file's index holds only the DiT).
std::vector<std::filesystem::path> dit_files(const weights::ModelIndex& index) {
  std::set<std::uint32_t> ids;
  for (const weights::WeightGroup* g : index.groups_of(weights::Component::kTransformer)) {
    for (const weights::IndexedTensor& t : g->tensors) ids.insert(t.file);
  }
  std::vector<std::filesystem::path> out;
  for (const std::uint32_t id : ids) out.push_back(index.file_path(id));
  return out;
}

// The enums share their order (static_asserts): planner / moss_dit attention modes.
static_assert(static_cast<int>(planner::AttentionMode::kQ8Fp8F32Acc) == static_cast<int>(moss_dit::AttentionMode::kQ8Fp8F32Acc) &&
              static_cast<int>(planner::AttentionMode::kQ8Fp16) == static_cast<int>(moss_dit::AttentionMode::kQ8Fp16) &&
              static_cast<int>(planner::AttentionMode::kExact) == 0 && static_cast<int>(moss_dit::AttentionMode::kExact) == 0);
moss_dit::AttentionMode dit_attention(planner::AttentionMode a) { return static_cast<moss_dit::AttentionMode>(a); }

// A file's size and modification time, for keys of state built from its contents: a file replaced under the same
// path gets another stamp (empty when it cannot be read; the load that follows reports why).
std::string file_stamp(const std::filesystem::path& p) {
  std::error_code ec;
  const std::uintmax_t size = std::filesystem::file_size(p, ec);
  if (ec) return {};
  const auto mtime = std::filesystem::last_write_time(p, ec);
  if (ec) return {};
  return std::to_string(size) + "@" + std::to_string(mtime.time_since_epoch().count());
}

// Bytes of a weight file, or of the *.safetensors files directly inside a directory.
std::uint64_t weight_bytes(const std::filesystem::path& p) {
  std::error_code ec;
  if (std::filesystem::is_regular_file(p, ec)) return std::filesystem::file_size(p, ec);
  std::uint64_t total = 0;
  for (const auto& e : std::filesystem::directory_iterator(p, ec)) {
    if (e.is_regular_file(ec) && e.path().extension() == ".safetensors") total += e.file_size(ec);
  }
  return total;
}

// The importer's QuantError for a component file (an unknown quantization format, a layout to convert first, a
// malformed file) as a model error (IndexError: MI_ERR_MODEL, server `model`) naming the file.
template <class F>
auto model_file_errors(const std::filesystem::path& file, F&& f) -> decltype(f()) {
  try {
    return f();
  } catch (const quant::QuantError& e) {
    const std::string where = util::path_utf8(file);
    const std::string what = e.what();
    throw weights::IndexError(what.find(where) == std::string::npos ? where + ": " + what : what);
  }
}

// The boundary check of the options' and the request's paths (util::check_user_path), before any std::filesystem call
// touches them (cache keys use util::path_key, which opens nothing on Windows).
void check_option_paths(const PipelineOptions& o) {
  const std::pair<const std::filesystem::path*, const char*> paths[] = {
      {&o.cache_dir, "cache_dir"},
      {&o.dit_file, "dit_file"},
      {&o.text_encoder_file, "text_encoder_file"},
      {&o.vae_file, "vae_file"},
      {&o.time_embedding_file, "time_embedding_file"}};
  for (const auto& [path, what] : paths) util::check_user_path(*path, what);
}

void check_user_paths(const PipelineOptions& o, const Request& r) {
  check_option_paths(o);
  const std::pair<const std::filesystem::path*, const char*> paths[] = {
      {&r.model, "model"}, {&r.dataspace_path, "dataspace_path"}, {&r.control.adapter, "control_adapter"}};
  for (const auto& [path, what] : paths) util::check_user_path(*path, what);
  for (const lora::LoraSpec& l : r.loras) util::check_user_path(l.path, "loras");
}

std::unique_ptr<DitSource> open_dit_file(const std::filesystem::path& file) {
  // NVFP4 linears run only on cc 12.0 with the sm_120a kernels (user decision D1); otherwise the file is refused as an
  // unsupported model (MI_ERR_MODEL).
  if (!std::filesystem::exists(file)) throw std::invalid_argument("DiT file not found: " + util::path_utf8(file));
  DitFileInfo info = inspect_dit_file(file, [] { return moss_dit::nvfp4_unsupported_reason(); });
  if (!info.refusal.empty()) throw weights::UnsupportedModelError(info.refusal);
  auto d = std::make_unique<DitSource>();
  d->path = file;
  d->quant = std::move(info.quant);
  d->index = &d->quant->index;
  d->precision = info.precision;
  d->linears = std::move(info.linears);
  d->remarks = std::move(info.remarks);
  return d;
}

// Where the text encoder comes from (P7a): the bundle's bf16 encoder, or a single INT8 / W4A8 file
// (qwen35_text::QuantTextModel). The tokenizer is always the bundle's. `identity` and `numerics` key the embedding cache:
// the weights (text_encoder_hash: names, shapes and each file's path, size, mtime and header) and the encoder's numerics
// (precision and kernel versions), so embeddings of one encoder are never served for another.
struct TeSource {
  std::filesystem::path path;  // the bundle's text_encoder directory or the file
  const weights::ModelIndex* index = nullptr;           // the bundle's (bf16)
  std::unique_ptr<qwen35_text::QuantTextModel> model;   // file only
  qwen35_text::Precision precision = qwen35_text::Precision::kBF16;
  std::uint64_t identity = 0;
  std::uint64_t numerics = 0;
  // A pinned host copy of this encoder failed to allocate: no more pinning (the plans hold no pinned copy) until
  // release_device_memory(false) or a new pipeline.
  mutable bool pin_failed = false;
};

// Where the VAE decoder comes from (P7a): the bundle's, or a single FLUX.2 VAE file (flux2_vae::index_vae_file, e.g.
// the full decoder) decoded with its own latent statistics rounded to bf16 as the bundle stores them (Flux2VaeDecoder's
// round_stats). `index` is used whole, never merged.
struct VaeSource {
  std::filesystem::path path;
  const weights::ModelIndex* index = nullptr;
  const weights::ModelIndex* file = nullptr;  // file only: Impl::vae_file_index
  flux2_vae::DecoderConfig config;
};

// Where the VAE encoder comes from (P7b img2img): PipelineOptions::vae_file when that file holds an encoder (its own
// latent statistics, rounded as for the decoder), else the bundle's VAE file, read directly (not through the index).
// `persistent` is Flux2VaeEncoder::persistent_bytes (the file validated).
struct EncoderSource {
  std::filesystem::path file;
  const weights::ModelIndex* vae_file = nullptr;  // the VAE file's index (round_stats), or nullptr: the bundle's file
  std::size_t persistent = 0;
};

// Transfer rates of this machine (P4b), measured in ~0.2 s: pinned host -> device (best of 4 copies), and the weight
// files -> pinned host memory with the DiT streamer's 4 readers over one DiT block (the better of two passes after a
// first one: page cache; the first pass counts as a cold read only when clearly slower). Throws on CUDA errors (e.g. no room for its buffers); I/O
// errors leave the defaults.
planner::HardwareProfile probe_hardware(const weights::ModelIndex& index, cudaStream_t stream) {
  planner::HardwareProfile hw;
  constexpr std::size_t kSlice = std::size_t{16} << 20;
  constexpr int kReaders = 4;
  struct Pinned {
    void* p = nullptr;
    ~Pinned() {
      if (p != nullptr) cudaFreeHost(p);
    }
  } host;
  MI_CUDA_CHECK(cudaHostAlloc(&host.p, kSlice * kReaders, cudaHostAllocDefault));
  {
    DeviceBuffer dev(kSlice * kReaders);
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    MI_CUDA_CHECK(cudaEventCreate(&e0));
    MI_CUDA_CHECK(cudaEventCreate(&e1));
    MI_CUDA_CHECK(cudaMemcpyAsync(dev.get(), host.p, kSlice * kReaders, cudaMemcpyHostToDevice, stream));  // warm-up
    // Best of a few copies: the bus is shared with whatever else runs on the machine.
    float best = 0;
    for (int i = 0; i < 4; ++i) {
      MI_CUDA_CHECK(cudaEventRecord(e0, stream));
      MI_CUDA_CHECK(cudaMemcpyAsync(dev.get(), host.p, kSlice * kReaders, cudaMemcpyHostToDevice, stream));
      MI_CUDA_CHECK(cudaEventRecord(e1, stream));
      MI_CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0;
      MI_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      if (ms > 0 && (best == 0 || ms < best)) best = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (best > 0) hw.h2d_gbps = static_cast<double>(kSlice * kReaders) / (static_cast<double>(best) * 1e6);
  }
  const weights::WeightGroup* g = index.find_group(weights::Component::kTransformer, "block.21");
  if (g == nullptr || g->tensors.empty()) return hw;
  try {
    // One file's contiguous span of the group (all of it unless the group straddles a shard boundary).
    const std::uint32_t file = g->tensors.front().file;
    std::uint64_t begin = std::numeric_limits<std::uint64_t>::max(), end = 0;
    for (const auto& t : g->tensors) {
      if (t.file != file) continue;
      begin = std::min(begin, t.offset);
      end = std::max(end, t.offset + t.nbytes);
    }
    weights::ReaderCache readers(index, false);
    const weights::FileReader& f = readers.get(file);
    const auto pass = [&] {
      const auto t0 = Clock::now();
      std::vector<std::thread> threads;
      std::exception_ptr error;
      std::mutex mu;
      const auto join = [&] {
        for (auto& t : threads) t.join();
      };
      try {
        for (int r = 0; r < kReaders; ++r) {
          threads.emplace_back([&, r] {
            try {
              std::byte* dst = static_cast<std::byte*>(host.p) + static_cast<std::size_t>(r) * kSlice;
              for (std::uint64_t o = begin + static_cast<std::uint64_t>(r) * kSlice; o < end; o += kSlice * kReaders) {
                f.read_at(o, dst, static_cast<std::size_t>(std::min<std::uint64_t>(kSlice, end - o)));
              }
            } catch (...) {
              std::lock_guard<std::mutex> lock(mu);
              error = std::current_exception();
            }
          });
        }
      } catch (...) {
        join();  // a thread that could not be created: the started ones still read into `host`
        throw;
      }
      join();
      if (error) std::rethrow_exception(error);
      return since(t0);
    };
    const double cold = pass(), warm = std::min(pass(), pass());
    const double bytes = static_cast<double>(end - begin);
    if (warm > 0) hw.host_gbps = bytes / warm / 1e9;
    if (cold > 3 * warm && cold > 0) hw.disk_gbps = bytes / cold / 1e9;
  } catch (const std::exception&) {
    return hw;  // unreadable weights fail later, with a better error
  }
  hw.measured = true;
  return hw;
}

// Seconds since the Unix epoch (the profile's age).
std::int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// The fused-block self-check result of each hardware key this process has checked (first use per process).
std::optional<moss_dit::FusedCheck> process_fused_check(const std::string& key, const moss_dit::FusedCheck* set = nullptr) {
  static std::mutex mu;
  static std::map<std::string, moss_dit::FusedCheck> checks;
  std::lock_guard<std::mutex> lock(mu);
  if (set != nullptr) checks[key] = *set;
  const auto it = checks.find(key);
  if (it == checks.end()) return std::nullopt;
  return it->second;
}

}  // namespace

DitFileInfo inspect_dit_file(const weights::SafetensorsFile& header,
                             const std::function<std::string()>& nvfp4_refusal) {
  const std::filesystem::path& file = header.path();
  weights::check_supported_dit_file(header);
  quant::QuantModel model =
      model_file_errors(file, [&] { return quant::open_quant_model(header, quant::ModelKind::kDiT); });
  DitFileInfo info;
  info.nvfp4 = std::any_of(model.linears.begin(), model.linears.end(),
                           [](const quant::QuantLinear& l) { return l.spec.format == quant::QuantFormat::kNvfp4; });
  // One 4-bit format per DiT: a file mixing INT4 ConvRot (convrot_w4a4) and NVFP4 linears is not a supported layout.
  if (info.nvfp4 && std::any_of(model.linears.begin(), model.linears.end(),
                                [](const quant::QuantLinear& l) { return l.encoding == quant::Encoding::kW4A4; })) {
    throw weights::IndexError(util::path_utf8(file) +
                              ": mixes INT4 ConvRot (convrot_w4a4) and NVFP4 linears; a DiT may use one 4-bit format "
                              "(with INT8 ConvRot and bf16 linears beside it)");
  }
  // Refused before to_model_index reads the block scales.
  if (info.nvfp4 && nvfp4_refusal) {
    if (const std::string why = nvfp4_refusal(); !why.empty()) {
      info.refusal = util::path_utf8(file) + ": " + why;
      return info;
    }
  }
  const auto scaled = [](const quant::QuantLinear& l) { return l.input_scale.has_value(); };
  info.static_input_scales = static_cast<int>(std::count_if(model.linears.begin(), model.linears.end(), scaled));
  // INT4 ConvRot (convrot_w4a4) linears stay packed for the INT4 GEMMs (the Turbo tier).
  info.quant = std::make_unique<quant::QuantIndex>(
      model_file_errors(file, [&] { return quant::to_model_index(model, {.native_int4 = true}); }));
  info.precision = moss_dit::has_int4_linears(info.quant.get())       ? planner::Precision::kInt4
                   : info.nvfp4                                         ? planner::Precision::kNvfp4
                   : moss_dit::has_quantized_linears(info.quant.get()) ? planner::Precision::kInt8
                                                                       : planner::Precision::kBf16;
  info.linears = moss_dit::describe_linears(info.quant.get());
  if (info.static_input_scales > 0) {
    info.remarks = std::to_string(info.static_input_scales) +
                   " linears carry a static input_scale; ignored (activations are scaled per token)";
  }
  return info;
}

DitFileInfo inspect_dit_file(const std::filesystem::path& file, const std::function<std::string()>& nvfp4_refusal) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(file, ec)) {
    if (std::filesystem::exists(file, ec)) throw std::invalid_argument(util::path_utf8(file) + ": not a regular file");
    throw weights::IoError(util::path_utf8(file) + ": no such file");
  }
  // GGUF / pickle containers are named before the header is parsed as safetensors.
  weights::check_supported_dit_file(file);
  return inspect_dit_file(weights::SafetensorsFile(file), nvfp4_refusal);
}

DitFileInfo inspect_dit_file(const std::filesystem::path& file, std::string_view nvfp4_refusal) {
  return inspect_dit_file(file, [why = std::string(nvfp4_refusal)] { return why; });
}

std::vector<int> parse_layer_list(std::string_view text) {
  std::vector<int> out;
  if (trim_ascii(text).empty()) return out;
  for (std::size_t pos = 0;;) {
    const std::size_t comma = text.find(',', pos);
    std::string_view item = trim_ascii(text.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
    if (!item.empty() && item.front() == '+') item.remove_prefix(1);
    int x = 0;
    const auto [end, ec] = std::from_chars(item.data(), item.data() + item.size(), x);
    if (item.empty() || ec != std::errc{} || end != item.data() + item.size()) {
      throw std::invalid_argument("SLG layers must be comma-separated integer indices, e.g. 22.");
    }
    out.push_back(x);
    if (comma == std::string_view::npos) return out;
    pos = comma + 1;
  }
}

void validate(const Request& r) {
  if (tok::python_strip(r.prompt).empty()) throw std::invalid_argument("Enter a prompt.");
  if (r.height <= 0 || r.width <= 0 || r.height % 16 != 0 || r.width % 16 != 0) {
    throw std::invalid_argument("Height and width must be positive multiples of 16.");
  }
  if (r.height > kMaxSide || r.width > kMaxSide) {
    throw std::invalid_argument("Height and width must be at most " + std::to_string(kMaxSide) + ".");
  }
  if (std::int64_t{r.height} * r.width > kMaxPixels) {
    throw std::invalid_argument("Image size must be at most 4.2 megapixels (e.g. 2048x2048); the model is meant for "
                                "0.25 to 4 megapixels.");
  }
  if (r.steps < 1) throw std::invalid_argument("Steps and image count must be positive.");
  if (r.steps > kMaxSteps) throw std::invalid_argument("Steps must be at most " + std::to_string(kMaxSteps) + ".");
  if (!std::isfinite(r.cfg) || r.cfg < 0 || !std::isfinite(r.shift) || r.shift <= 0) {
    throw std::invalid_argument("CFG must be finite and nonnegative; shift must be finite and positive.");
  }
  if (r.seed < -1 || r.seed >= kMaxSeed) {
    throw std::invalid_argument("Seed must be -1 (random) or an integer from 0 to 2^53-1 (browser-safe range).");
  }
  if (r.max_sequence_length < 1 || r.max_sequence_length > 4096) {
    throw std::invalid_argument("Maximum text length must be between 1 and 4096.");
  }
  if (!std::isfinite(r.slg_scale) || r.slg_scale < 0) throw std::invalid_argument("SLG scale must be finite and nonnegative.");
  if (!(0 <= r.slg_start && r.slg_start <= r.slg_stop && r.slg_stop <= 1)) {
    throw std::invalid_argument("Require 0 <= SLG start <= SLG stop <= 1.");
  }
  check_layers(r.uncond_skip_layers, "uncond_skip_layers");
  check_layers(r.slg_layers, "slg_layers");
  if (!std::isfinite(r.flow_multiplier) || r.flow_multiplier <= 0) throw std::invalid_argument("Flow multiplier must be finite and positive.");
  if (!std::isfinite(r.eta) || r.eta < 0 || !std::isfinite(r.s_noise) || r.s_noise < 0) {
    throw std::invalid_argument("eta and s_noise must be finite and nonnegative.");
  }
  // img2img, inpainting and the hires fix.
  const auto check_image = [](const Image& im, int channels, const char* what) {
    if (im.width <= 0 || im.height <= 0 || im.width > kMaxInputSide || im.height > kMaxInputSide ||
        std::int64_t{im.width} * im.height > kMaxInputPixels) {
      throw std::invalid_argument(std::string(what) + " must be 1 to " + std::to_string(kMaxInputSide) + " pixels on a side and at most " +
                                  std::to_string(kMaxInputPixels / 1000000) + " megapixels.");
    }
    if (im.channels != channels) {
      throw std::invalid_argument(std::string(what) + (channels == 3 ? " must be an RGB image (3 channels)." : " must be a grayscale image (1 channel)."));
    }
    if (im.pixels.size() != static_cast<std::size_t>(im.width) * static_cast<std::size_t>(im.height) * static_cast<std::size_t>(channels)) {
      throw std::invalid_argument(std::string(what) + ": the pixel buffer size does not match width x height x channels.");
    }
  };
  if (!std::isfinite(r.denoise_strength) || r.denoise_strength < 0 || r.denoise_strength > 1) {
    throw std::invalid_argument("Denoise strength must be between 0 and 1.");
  }
  if (r.init_image == nullptr) {
    if (r.denoise_strength != 1.0) throw std::invalid_argument("A denoise strength below 1 needs an init image.");
    if (r.mask != nullptr) throw std::invalid_argument("An inpainting mask needs an init image.");
  } else {
    check_image(*r.init_image, 3, "The init image");
  }
  if (r.mask != nullptr) {
    check_image(*r.mask, 1, "The inpainting mask");
    if (r.mask->width != r.init_image->width || r.mask->height != r.init_image->height) {
      throw std::invalid_argument("The inpainting mask must have the init image's size (" + std::to_string(r.init_image->width) +
                                  "x" + std::to_string(r.init_image->height) + "), not " + std::to_string(r.mask->width) +
                                  "x" + std::to_string(r.mask->height) + ".");
    }
  }
  if (r.hires.enabled()) {
    const Request::Hires& hf = r.hires;
    if (hf.height <= 0 || hf.width <= 0 || hf.height % 16 != 0 || hf.width % 16 != 0) {
      throw std::invalid_argument("Hires fix: height and width must be positive multiples of 16.");
    }
    if (hf.height > kMaxSide || hf.width > kMaxSide || std::int64_t{hf.height} * hf.width > kMaxPixels) {
      throw std::invalid_argument("Hires fix: the size must be at most " + std::to_string(kMaxSide) +
                                  " on a side and 4.2 megapixels.");
    }
    if (!std::isfinite(hf.strength) || hf.strength < 0 || hf.strength > 1) {
      throw std::invalid_argument("Hires fix: the denoise strength must be between 0 and 1.");
    }
    if (hf.steps < 0 || hf.steps > kMaxSteps) {
      throw std::invalid_argument("Hires fix: steps must be 0 (the request's) to " + std::to_string(kMaxSteps) + ".");
    }
  }
  if (r.sampler == SamplerKind::kDataSpace && (r.init_image != nullptr || r.hires.enabled())) {
    throw std::invalid_argument("The dataspace sampler cannot start from an image (img2img, inpainting, hires fix).");
  }
  // The pipeline passes SamplerOptions (seed, noise source) to every sampler; the dataspace sampler also needs its file.
  if (r.sampler == SamplerKind::kDataSpace && r.dataspace_path.empty()) {
    throw std::invalid_argument("The dataspace sampler needs its parameter file (dataspace_path).");
  }
  // Control adapter (the kinds are checked against the file in generate()).
  if (!r.control.enabled() && !r.control.images.empty()) throw std::invalid_argument("Control images need a control adapter.");
  if (r.control.enabled() && !std::isfinite(r.control.lora_strength)) throw std::invalid_argument("The control adapter's LoRA strength must be finite.");
  for (std::size_t i = 0; i < r.control.images.size(); ++i) {
    const Request::ControlImage& c = r.control.images[i];
    if (c.kind.empty()) throw std::invalid_argument("A control image needs its kind (e.g. color, depth).");
    for (std::size_t j = 0; j < i; ++j) {
      if (r.control.images[j].kind == c.kind) throw std::invalid_argument("Two control images of kind \"" + c.kind + "\".");
    }
    const std::string what = "The " + c.kind + " control image";
    if (c.image == nullptr) throw std::invalid_argument(what + " is missing.");
    check_image(*c.image, c.image->channels == 1 ? 1 : 3, what.c_str());
    if (!std::isfinite(c.strength) || c.strength < 0 || c.strength > 10) {
      throw std::invalid_argument(what + ": the strength must be between 0 and 10.");
    }
    if (!(0 <= c.start && c.start <= c.end && c.end <= 1)) throw std::invalid_argument(what + ": require 0 <= start <= end <= 1.");
  }
  accel::validate(r.accel);
}

std::size_t vram_option_bytes(double amount, double unit_bytes, std::string_view what) {
  const std::string name(what);
  if (!std::isfinite(amount) || amount <= 0) throw std::invalid_argument(name + " must be a finite positive number.");
  const double bytes = amount * unit_bytes;
  if (!(bytes <= static_cast<double>(kMaxVramBytes))) {
    throw std::invalid_argument(name + " must be at most " + std::to_string(kMaxVramBytes >> 30) + " GiB.");
  }
  if (bytes < static_cast<double>(std::size_t{1} << 20)) throw std::invalid_argument(name + " must be at least 1 MiB.");
  return static_cast<std::size_t>(bytes);
}

Request request_from_parameters(const json::Value& doc) {
  if (!doc.is_object()) throw std::invalid_argument("parameters: expected a JSON object");
  const json::Value* params = doc.find("parameters");
  std::int64_t version = 2;
  if (params != nullptr) {
    const json::Value* v = doc.find("schema_version");
    if (v != nullptr) {
      if (!v->is_int() || (v->as_int() != 1 && v->as_int() != 2)) throw std::invalid_argument("Unsupported generation metadata version.");
      version = v->as_int();
    }
    if (!params->is_object()) throw std::invalid_argument("parameters: \"parameters\" must be an object");
  } else {
    params = &doc;
  }
  const json::Value& p = *params;
  Request r;
  const auto integer = [&](const char* key, auto& dst) {
    const json::Value* v = p.find(key);
    if (v == nullptr) return;
    if (!v->is_int()) throw std::invalid_argument(std::string(key) + " must be an integer.");
    const std::int64_t x = v->as_int();
    using T = std::remove_reference_t<decltype(dst)>;
    if (x < std::numeric_limits<T>::min() || x > std::numeric_limits<T>::max()) {
      throw std::invalid_argument(std::string(key) + " is out of range.");
    }
    dst = static_cast<T>(x);
  };
  const auto number = [&](const char* key, double& dst) {
    const json::Value* v = p.find(key);
    if (v == nullptr) return;
    if (!v->is_number()) throw std::invalid_argument(std::string(key) + " must be a number.");
    dst = v->as_double();
  };
  const auto string = [&](const char* key, std::string& dst) {
    const json::Value* v = p.find(key);
    if (v == nullptr || v->is_null()) return;
    if (!v->is_string()) throw std::invalid_argument(std::string(key) + " must be a string.");
    dst = v->as_string();
  };
  string("prompt", r.prompt);
  string("negative_prompt", r.negative_prompt);
  integer("height", r.height);
  integer("width", r.width);
  integer("steps", r.steps);
  number("cfg", r.cfg);
  number("shift", r.shift);
  if (const json::Value* s = p.find("seed"); s != nullptr) {
    if (s->is_null()) r.seed = -1;  // parse_parameters: missing seed -> random
    else integer("seed", r.seed);
  }
  if (const json::Value* n = p.find("num_images"); n != nullptr) {
    if (!n->is_int() || n->as_int() < 1) throw std::invalid_argument("Steps and image count must be positive.");
    if (n->as_int() != 1) throw std::invalid_argument("num_images > 1 is not supported; run one generation per image.");
  }
  integer("max_sequence_length", r.max_sequence_length);
  if (version == 1) {
    // Version 1 called negative-branch skipping "slg_layers" (gradio_app.parse_parameters).
    const json::Value* slg = p.find("slg");
    const bool enabled = slg == nullptr || (slg->is_bool() && slg->as_bool());
    const json::Value* layers = p.find("slg_layers");
    r.uncond_skip_layers = enabled && layers != nullptr ? parse_layers(*layers, "slg_layers") : std::vector<int>{};
    r.slg_layers.clear();
  } else {
    if (const json::Value* v = p.find("uncond_skip_layers"); v != nullptr) r.uncond_skip_layers = parse_layers(*v, "uncond_skip_layers");
    if (const json::Value* v = p.find("slg_layers"); v != nullptr) r.slg_layers = parse_layers(*v, "slg_layers");
  }
  number("slg_scale", r.slg_scale);
  number("slg_start", r.slg_start);
  number("slg_stop", r.slg_stop);
  // img2img / hires fix scalars (engine keys; the images themselves are front-end inputs, e.g. the CLI's paths).
  number("denoise_strength", r.denoise_strength);
  if (const json::Value* v = p.find("resize"); v != nullptr && !v->is_null()) {
    if (!v->is_string()) throw std::invalid_argument("resize must be a string.");
    r.resize = parse_resize_mode(v->as_string());
  }
  number("hires_strength", r.hires.strength);
  integer("hires_steps", r.hires.steps);
  // P6b accelerators (engine keys).
  number("cfg_truncation", r.accel.cfg_truncation);
  if (const json::Value* v = p.find("step_cache"); v != nullptr && !v->is_null()) {
    if (!v->is_string()) throw std::invalid_argument("step_cache must be a string.");
    r.accel.step_cache = accel::parse_step_cache(v->as_string());
  }
  if (const json::Value* v = p.find("step_cache_threshold"); v != nullptr && !v->is_null()) {
    double threshold = 0;
    number("step_cache_threshold", threshold);
    r.accel.step_cache_threshold = threshold;
  }
  return r;
}

GenerationMetadata generation_metadata(const Request& q, const Result& r) {
  GenerationMetadata m;
  m.engine_version = std::to_string(MI_VERSION_MAJOR) + "." + std::to_string(MI_VERSION_MINOR) + "." + std::to_string(MI_VERSION_PATCH);
  m.tier = planner::tier_name(r.tier.tier);
  const auto name = [](const std::string& p) { return util::path_utf8(util::path_from_utf8(p).filename()); };
  m.models.push_back({"bundle", util::path_utf8(util::path_key(q.model).filename()), ""});
  if (!r.dit_source.empty()) m.models.push_back({"dit", name(r.dit_source), ""});
  if (!r.te_source.empty()) m.models.push_back({"text_encoder", name(r.te_source), ""});
  if (!r.vae_source.empty()) m.models.push_back({"vae", name(r.vae_source), ""});
  m.prompt = q.prompt;
  m.negative_prompt = q.negative_prompt;
  m.sampler = sampler_name(q.sampler);
  m.scheduler = scheduler_name(q.scheduler);
  m.steps = q.steps;
  m.cfg = q.cfg;
  m.shift = q.shift;
  m.seed = r.seed;
  m.width = r.width;
  m.height = r.height;
  for (const lora::LoraSpec& l : q.loras) m.loras.push_back({util::path_utf8(l.path.filename()), "", l.strength});
  m.extra.emplace_back("attention", planner::attention_name(r.tier.attention));
  m.extra.emplace_back("dit_linears", r.dit_linears);
  if (!r.time_embedding.empty()) m.extra.emplace_back("time_embedding", name(r.time_embedding));
  m.extra.emplace_back("noise_layout", noise_layout_name(q.noise_layout));
  if (sampler_is_stochastic(q.sampler)) m.extra.emplace_back("noise_source", noise_source_name(q.noise_source));
  // img2img / inpainting / hires fix: the settings that shaped the image (file names only, no directories).
  const auto num = [](double d) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", d);
    return std::string(buf);
  };
  const auto image_label = [](const Image& im) {
    return (im.name.empty() ? std::string() : util::path_utf8(util::path_from_utf8(im.name).filename()) + " ") +
           std::to_string(im.width) + "x" + std::to_string(im.height);
  };
  const bool image_used = q.init_image != nullptr && r.image_mode != "txt2img";
  if (image_used) {
    m.extra.emplace_back("mode", r.image_mode);
    m.extra.emplace_back("init_image", image_label(*q.init_image));
    m.extra.emplace_back("denoise_strength", num(q.denoise_strength));
    m.extra.emplace_back("resize", resize_mode_name(q.resize));
    if (q.mask != nullptr) m.extra.emplace_back("mask", image_label(*q.mask));
  }
  if (q.hires.enabled()) {
    m.extra.emplace_back("hires", std::to_string(q.hires.width) + "x" + std::to_string(q.hires.height));
    m.extra.emplace_back("hires_base", std::to_string(q.width) + "x" + std::to_string(q.height));
    m.extra.emplace_back("hires_strength", num(q.hires.strength));
    m.extra.emplace_back("hires_steps", std::to_string(q.hires.steps > 0 ? q.hires.steps : q.steps));
    m.extra.emplace_back("hires_upscaler", "lanczos");
    if (!image_used) m.extra.emplace_back("resize", resize_mode_name(q.resize));
  }
  if (q.control.enabled()) {
    m.extra.emplace_back("control", name(util::path_utf8(q.control.adapter)));
    m.extra.emplace_back("control_lora_strength", num(q.control.lora_strength));
    for (const Request::ControlImage& c : q.control.images) {
      m.extra.emplace_back("control_" + c.kind, image_label(*c.image) + " strength " + num(c.strength) + " steps " +
                                                    num(c.start) + "-" + num(c.end));
    }
    if (!q.control.images.empty() && !image_used && !q.hires.enabled()) m.extra.emplace_back("resize", resize_mode_name(q.resize));
  }
  // P6b accelerators that changed the image.
  if (q.accel.cfg_truncation < 1.0 && q.cfg != 1.0) m.extra.emplace_back("cfg_truncation", num(q.accel.cfg_truncation));
  if (q.accel.step_cache_on()) {
    m.extra.emplace_back("step_cache", accel::step_cache_name(q.accel.step_cache));
    // What ran: the request's threshold, or the model's default the pipeline filled in.
    const double threshold = r.accel.step_cache_threshold > 0
                                 ? r.accel.step_cache_threshold
                                 : q.accel.step_cache_threshold.value_or(accel::kDefaultFbCacheThreshold);
    m.extra.emplace_back("step_cache_threshold", num(threshold));
  }
  return m;
}

struct Pipeline::Impl {
  PipelineOptions options;
  runtime::StreamSet streams;
  std::size_t free_at_start = 0;
  std::map<std::string, std::unique_ptr<Bundle>> bundles;
  std::unique_ptr<EmbeddingCache> cache;

  std::unique_ptr<runtime::DeviceArena> arena;
  // Long-lived while the arena lives (it holds a reference to it): its pinned rings and streams are set up once.
  std::unique_ptr<qwen35_text::TextEncoder> te;
  const TeSource* te_src = nullptr;
  std::map<std::string, std::unique_ptr<TeSource>> te_sources;
  std::unique_ptr<moss_dit::MossDit> dit;
  const DitSource* dit_src = nullptr;
  // DiT sources by key ("bundle:" / "file:" + canonical path), kept while the pipeline lives (a model references one).
  std::map<std::string, std::unique_ptr<DitSource>> dit_sources;
  // Runtime LoRAs: the stack and its device side branches (outside the arena), the files they were read for, and the
  // merged adapters baked into the loaded DiT (empty: none).
  std::unique_ptr<lora::LoraStack> lora_stack;
  std::unique_ptr<lora::SideBranches> lora_sb;
  std::string lora_files_key;
  std::string dit_merge_key;
  // Control adapter: its file (header facts) and replacement embedder on the device (outside the arena, kept while the
  // file stays the same), and the DiT input extension a pass points at its control latents.
  std::unique_ptr<control::ControlFile> control_file;
  std::unique_ptr<control::DeviceEmbedder> control_emb;
  std::string control_key;
  moss_dit::InputExtension control_ext;
  std::unique_ptr<flux2_vae::Flux2VaeDecoder> vae;
  const VaeSource* vae_src = nullptr;
  std::map<std::string, std::unique_ptr<VaeSource>> vae_sources;
  std::unique_ptr<weights::ModelIndex> vae_file;  // options.vae_file's index (vae_file_index)
  // img2img's VAE encoder: persistent in the arena next to the decoder once loaded (dropped with the arena).
  std::unique_ptr<flux2_vae_encoder::Flux2VaeEncoder> encoder;
  const EncoderSource* encoder_src = nullptr;
  std::map<std::string, std::unique_ptr<EncoderSource>> encoder_sources;
  // DiT source whose resident DiT was last dropped outside the plan's parking (its next load is a reload).
  const DitSource* dit_lost = nullptr;
  // DiT source the plan last parked (its next load is a parking reload: buffered unless direct is the warm choice,
  // dit_direct_io).
  const DitSource* dit_parked = nullptr;
  bool dit_load_warm = false;  // the last DiT load from the files found them in the page cache (DitIoDecision::warm)
  // Transfer rates for the cost model and whether the weight files are page-cached: probed (or read from the cache
  // directory) at the first generate(), refined from every pass (pipeline/hardware_profile.h). Unset with
  // options.hardware.
  std::optional<ProfileLearner> learner;
  std::string hw_key;  // hardware_key() of the learner's profile
  // This pipeline's fused-block self-check (from the profile, the process, or run at the first fused generate()).
  std::optional<moss_dit::FusedCheck> fused_check;
  // Host copies kept across calls. A soft unload's copy of the DiT's weight files (release_device_memory): the next
  // load of that source reads it instead of the files, then frees it.
  std::optional<weights::HostFiles> dit_host;
  const DitSource* dit_host_src = nullptr;
  // A host-resident text encoder's copy, kept past the encoder (an arena recreation, a soft unload) while the plan keeps
  // that source host-resident with the same memory kind: the next encoder adopts it instead of reading the weights.
  std::vector<weights::HostGroup> te_kept;
  const TeSource* te_kept_src = nullptr;
  bool te_kept_pinned = false;

  // Per generate() call.
  MemoryStats* mem = nullptr;

  explicit Impl(PipelineOptions o) : options(std::move(o)) {
    check_option_paths(options);  // before the cache folder or a time embedding is touched
    free_at_start = runtime::DeviceArena::free_device_bytes();
    if (options.embedding_cache) {
      EmbeddingCacheOptions co;
      co.disk_dir = options.cache_dir;
      cache = std::make_unique<EmbeddingCache>(co);
    }
    if (!options.time_embedding_file.empty()) {
      time_embedding = std::make_unique<moss_dit::TimeEmbeddingFp32>(moss_dit::load_time_embedding(options.time_embedding_file));
    }
  }

  ~Impl() {
    te.reset();
    encoder.reset();
    vae.reset();
    dit.reset();
    arena.reset();
  }

  // The bundle of a call. The component files are opened first, each identified as its component by its shapes
  // (open_dit_file, check_supported_text_encoder_file, index_vae_file refuse any other file), and only then are the
  // components they replace left out of the bundle's index (weights::BuildOptions::skip): their folders' weights are not
  // read, so the folders need hold only their configs. The key carries the replaced set, so an index without a
  // component never serves a call that needs it.
  Bundle& bundle(const std::filesystem::path& dir, double& load_s) {
    weights::ComponentSet replaced;
    if (!options.text_encoder_file.empty()) replaced.add(weights::Component::kTextEncoder);
    if (!options.dit_file.empty()) replaced.add(weights::Component::kTransformer);
    if (!options.vae_file.empty()) replaced.add(weights::Component::kVae);
    const std::string key = util::path_utf8(util::path_key(dir)) + "|" + std::to_string(replaced.bits);
    auto it = bundles.find(key);
    if (it == bundles.end()) {
      const auto t0 = Clock::now();
      if (!std::filesystem::is_directory(dir)) throw std::invalid_argument("model bundle not found: " + util::path_utf8(dir));
      weights::check_supported_bundle(dir);  // plain refusal of other models, before any weight file is read
      if (replaced.has(weights::Component::kTransformer)) dit_file_source(load_s);
      if (replaced.has(weights::Component::kTextEncoder)) te_file_source(load_s);
      if (replaced.has(weights::Component::kVae)) vae_file_index();
      it = bundles.emplace(key, open_bundle(dir, replaced)).first;
      load_s = since(t0);
    }
    return *it->second;
  }

  // The DiT of this call: options.dit_file, else the bundle's transformer.
  DitSource& dit_source(const Bundle& b, double& load_s) {
    if (!options.dit_file.empty()) return dit_file_source(load_s);
    return cached_dit("bundle:" + util::path_utf8(util::path_key(b.dir)), load_s, [&] {
      auto d = std::make_unique<DitSource>();
      d->path = b.dir / "transformer";
      d->index = &b.index;
      d->linears = moss_dit::describe_linears(nullptr);
      return d;
    });
  }
  // options.dit_file's DiT (opened before the bundle is indexed: Impl::bundle).
  DitSource& dit_file_source(double& load_s) {
    return cached_dit("file:" + util::path_utf8(util::path_key(options.dit_file)), load_s,
                      [&] { return open_dit_file(options.dit_file); });
  }
  template <class Open>
  DitSource& cached_dit(const std::string& key, double& load_s, Open&& open) {
    auto it = dit_sources.find(key);
    if (it == dit_sources.end()) {
      const auto t0 = Clock::now();
      std::unique_ptr<DitSource> d = open();
      d->files = dit_files(*d->index);
      d->lineage = dit_lineage(*d->index);
      it = dit_sources.emplace(key, std::move(d)).first;
      load_s += since(t0);
    }
    return *it->second;
  }

  // The text encoder of this call: options.text_encoder_file, else the bundle's.
  TeSource& te_source(const Bundle& b, double& load_s) {
    if (!options.text_encoder_file.empty()) return te_file_source(load_s);
    return cached_te("bundle:" + util::path_utf8(util::path_key(b.dir)), load_s, [&] {
      auto t = std::make_unique<TeSource>();
      t->path = b.dir / "text_encoder";
      t->index = &b.index;
      t->identity = b.te_hash;
      return t;
    });
  }
  // options.text_encoder_file's text encoder (opened before the bundle is indexed: Impl::bundle).
  TeSource& te_file_source(double& load_s) {
    const std::filesystem::path& path = options.text_encoder_file;
    return cached_te("file:" + util::path_utf8(util::path_key(path)), load_s, [&] {
      if (!std::filesystem::is_regular_file(path)) throw std::invalid_argument("text encoder file not found: " + util::path_utf8(path));
      weights::check_supported_text_encoder_file(path);
      auto t = std::make_unique<TeSource>();
      t->path = path;
      t->model = model_file_errors(path, [&] { return std::make_unique<qwen35_text::QuantTextModel>(path); });
      t->precision = t->model->precision();
      t->identity = text_encoder_hash(t->model->index());
      return t;
    });
  }
  template <class Open>
  TeSource& cached_te(const std::string& key, double& load_s, Open&& open) {
    auto it = te_sources.find(key);
    if (it == te_sources.end()) {
      const auto t0 = Clock::now();
      std::unique_ptr<TeSource> t = open();
      t->numerics = qwen35_text::numerics_id(t->precision);
      it = te_sources.emplace(key, std::move(t)).first;
      load_s += since(t0);
    }
    return *it->second;
  }

  // options.vae_file's index, read once: flux2_vae::index_vae_file refuses a file without a complete FLUX.2 decoder,
  // decoder_config one whose shapes are no supported decoder's.
  const weights::ModelIndex& vae_file_index() {
    if (!vae_file) {
      if (!std::filesystem::is_regular_file(options.vae_file)) {
        throw std::invalid_argument("VAE file not found: " + util::path_utf8(options.vae_file));
      }
      auto index = std::make_unique<weights::ModelIndex>(flux2_vae::index_vae_file(options.vae_file));
      flux2_vae::decoder_config(*index);
      vae_file = std::move(index);
    }
    return *vae_file;
  }

  // The VAE decoder of this call: options.vae_file, else the bundle's.
  VaeSource& vae_source(const Bundle& b, double& load_s) {
    const bool file = !options.vae_file.empty();
    const std::filesystem::path path = file ? options.vae_file : b.dir;
    const std::string key = (file ? "file:" : "bundle:") + util::path_utf8(util::path_key(path));
    auto it = vae_sources.find(key);
    if (it == vae_sources.end()) {
      const auto t0 = Clock::now();
      auto v = std::make_unique<VaeSource>();
      if (file) {
        v->path = path;
        v->file = &vae_file_index();
        v->index = v->file;
      } else {
        v->path = b.dir / "vae";
        v->index = &b.index;
      }
      v->config = flux2_vae::decoder_config(*v->index);
      it = vae_sources.emplace(key, std::move(v)).first;
      load_s += since(t0);
    }
    return *it->second;
  }

  // The VAE encoder of this call (EncoderSource).
  EncoderSource& encoder_source(const Bundle& b, double& load_s) {
    const std::string key = (options.vae_file.empty() ? std::string() : "file:" + util::path_utf8(util::path_key(options.vae_file)) + "|") +
                            "bundle:" + util::path_utf8(util::path_key(b.dir));
    auto it = encoder_sources.find(key);
    if (it == encoder_sources.end()) {
      const auto t0 = Clock::now();
      auto e = std::make_unique<EncoderSource>();
      // A VAE file with an encoder (a tensor under "encoder." or "<prefix>.encoder.") is used whole, its errors
      // reported; a decoder-only file (e.g. the small decoder) encodes with the bundle's VAE.
      bool has_encoder = false;
      if (!options.vae_file.empty()) {
        const weights::SafetensorsFile header(options.vae_file);
        for (const weights::TensorInfo& t : header.tensors()) {
          if (t.name.rfind("encoder.", 0) == 0 || t.name.find(".encoder.") != std::string::npos) has_encoder = true;
        }
      }
      if (has_encoder) {
        e->vae_file = &vae_file_index();
        e->persistent = flux2_vae_encoder::Flux2VaeEncoder::persistent_bytes(options.vae_file, e->vae_file);
        e->file = options.vae_file;
      } else {
        e->file = b.dir / "vae" / "diffusion_pytorch_model.safetensors";
        if (!std::filesystem::is_regular_file(e->file)) {
          throw std::invalid_argument("img2img needs the VAE encoder, but the bundle has no " + util::path_utf8(e->file));
        }
        e->persistent = flux2_vae_encoder::Flux2VaeEncoder::persistent_bytes(e->file);
      }
      it = encoder_sources.emplace(key, std::move(e)).first;
      load_s += since(t0);
    }
    return *it->second;
  }

  // PipelineOptions::time_embedding_file, read at construction.
  std::unique_ptr<moss_dit::TimeEmbeddingFp32> time_embedding;

  // Data-space sampler parameters by file (read once per pipeline).
  std::map<std::string, DataSpaceParams> dataspaces;
  const DataSpaceParams& dataspace(const std::filesystem::path& file) {
    const std::string key = util::path_utf8(util::path_key(file));
    auto it = dataspaces.find(key);
    if (it == dataspaces.end()) it = dataspaces.emplace(key, load_dataspace(util::path_utf8(file))).first;
    return it->second;
  }

  void sample_memory() {
    const std::size_t f = runtime::DeviceArena::free_device_bytes();
    if (free_at_start > f) mem->device_peak = std::max(mem->device_peak, free_at_start - f);
    if (arena) mem->arena_high_water = std::max(mem->arena_high_water, arena->high_water());
    mem->host_private_peak = std::max(mem->host_private_peak, planner::process_host_memory().private_bytes);
  }

  // `pinned`: a host-resident copy is page-locked (planner::TePlan::pinned). A quantized encoder (3.9-6.9 GB) held in RAM
  // is always pinned (Capabilities::te_host_pageable off: its cache-miss target needs the direct DMA); the bf16 one
  // (13.8 GB) only where the plan pins it (PipelineOptions::te_host_pin, planner.h), else pageable.
  qwen35_text::EncoderOptions te_options(const TeSource& t, planner::TeResidency residency, int buffers, bool pinned) const {
    qwen35_text::EncoderOptions o;
    o.direct_io = options.direct_io;
    o.weight_buffers = std::max(1, buffers);
    const qwen35_text::WeightSource host =
        pinned || t.model ? qwen35_text::WeightSource::kHostPinned : qwen35_text::WeightSource::kHostPageable;
    o.source = residency == planner::TeResidency::kHostResident ? host : qwen35_text::WeightSource::kFile;
    return o;
  }

  // The encoder's host copy into te_kept before the encoder is dropped (it shares the memory).
  void keep_te_host() {
    if (!te || te->host_resident_bytes() == 0) return;
    te_kept = te->share_host();
    te_kept_src = te_src;
    te_kept_pinned = te->options().source == qwen35_text::WeightSource::kHostPinned;
  }
  void drop_te_host() {
    te_kept.clear();
    te_kept_src = nullptr;
  }
  // Host bytes of the text encoder's copy: the encoder's, else the kept one.
  std::size_t te_host_bytes() const {
    if (te) return te->host_resident_bytes();
    std::size_t total = 0;
    for (const weights::HostGroup& g : te_kept) total += g.storage_bytes;
    return total;
  }
  std::size_t dit_host_bytes() const { return dit_host ? static_cast<std::size_t>(dit_host->bytes()) : 0; }
  // Empty when a host copy of `bytes` the engine keeps (a soft unload's) fits: it leaves planner::host_pin_margin() of
  // the available host memory (cgroup-aware) free and, with a host budget, stays within it; else why not.
  std::string host_copy_room(std::uint64_t bytes) const {
    const auto gib = [](std::uint64_t b) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.1f GiB", static_cast<double>(b) / static_cast<double>(std::size_t{1} << 30));
      return std::string(buf);
    };
    const std::size_t available = planner::host_available_bytes();
    const std::size_t total = planner::host_total_bytes();
    const std::size_t margin = planner::host_pin_margin(total);
    if (available == 0) return "the available host memory is unknown";
    if (!planner::fits_pin_margin(static_cast<std::size_t>(bytes), available, total)) {
      return "its " + gib(bytes) + " would leave less than " + gib(margin) + " of the " + gib(available) + " available";
    }
    const std::size_t budget = options.host_ram_budget;
    if (budget != 0 && planner::process_host_memory().private_bytes + bytes > budget) {
      return "its " + gib(bytes) + " would exceed the " + gib(budget) + " host budget";
    }
    return {};
  }
  std::size_t te_workspace(const TeSource& t, std::span<const std::int64_t> lengths, const qwen35_text::EncoderOptions& o) const {
    return t.model ? qwen35_text::TextEncoder::workspace_bytes(*t.model, lengths, o)
                   : qwen35_text::TextEncoder::workspace_bytes(*t.index, lengths, o);
  }
  static std::size_t te_host(const TeSource& t, const qwen35_text::EncoderOptions& o) {
    return t.model ? qwen35_text::TextEncoder::host_bytes(*t.model, o) : qwen35_text::TextEncoder::host_bytes(*t.index, o);
  }

  // A failed in-place re-merge (remerge_lora / rebase_merges) leaves the loaded DiT's merged weights unknown: drop it
  // (reported as a reload when the next request loads it) and forget its merge key, so nothing runs on those weights.
  void drop_dit_after_failed_merge() {
    if (dit) dit->set_lora(nullptr);
    if (arena) reset_arena(arena->capacity());
    dit_merge_key.clear();
  }

  // Drops every resident model and recreates the arena with `capacity` bytes (all stack phases must have ended).
  // `park`: the plan evicts the DiT here (anything else that drops it makes its next load a reported reload).
  void reset_arena(std::size_t capacity, bool park = false) {
    MI_CUDA_CHECK(cudaStreamSynchronize(streams.compute()));
    if (dit && !park) dit_lost = dit_src;
    if (dit && park) dit_parked = dit_src;
    keep_te_host();
    te.reset();
    te_src = nullptr;
    encoder.reset();
    encoder_src = nullptr;
    vae.reset();
    vae_src = nullptr;
    dit.reset();
    dit_src = nullptr;
    if (arena) {
      sample_memory();
      arena.reset();
    }
    arena = std::make_unique<runtime::DeviceArena>(capacity);
    ++mem->arena_creations;
    mem->arena_capacity = std::max(mem->arena_capacity, arena->capacity());
  }

  // Drops every resident model and the arena itself (all stack phases must have ended): frees the device memory for an
  // allocation outside it. The next plan creates the arena again; a resident DiT dropped here is reloaded (reported).
  void release_arena() {
    MI_CUDA_CHECK(cudaStreamSynchronize(streams.compute()));
    if (dit) {
      dit->set_lora(nullptr);
      dit_lost = dit_src;
    }
    keep_te_host();
    te.reset();
    te_src = nullptr;
    encoder.reset();
    encoder_src = nullptr;
    vae.reset();
    vae_src = nullptr;
    dit.reset();
    dit_src = nullptr;
    dit_merge_key.clear();
    if (arena) {
      sample_memory();
      arena.reset();
    }
  }

  moss_dit::Residency dit_residency(const planner::DitPlan& d) const {
    moss_dit::Residency r;
    if (d.residency != planner::DitResidency::kStreamed) return r;
    r.resident_blocks = d.resident_blocks;
    r.buffers = d.buffers;
    r.source = d.source == planner::DitSource::kHostPinned ? moss_dit::StreamSource::kHostPinned : moss_dit::StreamSource::kFile;
    r.direct_io = options.direct_io;
    return r;
  }

  std::filesystem::path profile_path() const {
    return options.cache_dir.empty() ? std::filesystem::path() : options.cache_dir / "hardware_profile.json";
  }

  bool weights_cached() const { return !learner || learner->te_weights_cached(); }
  bool dit_weights_cached() const { return !learner || learner->dit_weights_cached(); }

  planner::HardwareProfile hardware(const weights::ModelIndex& dit_index) {
    if (options.hardware) return *options.hardware;
    if (learner) return learner->current();
    hw_key = hardware_key();
    const std::int64_t now = unix_now();
    std::optional<StoredProfile> stored;
    if (!profile_path().empty()) stored = read_profile(profile_path(), hw_key, now);
    if (!stored) {
      StoredProfile p;
      try {
        p.probe = probe_hardware(dit_index, streams.compute());
      } catch (const std::exception&) {
        p.probe = planner::HardwareProfile{};  // e.g. no room for the probe's 64 MiB buffers: plan with the defaults
      }
      p.learned = p.probe;
      p.probed_at = now;
      stored = p;
      if (p.probe.measured) save_profile(*stored);
    }
    learner.emplace(*stored);
    return learner->current();
  }

  void save_profile(const StoredProfile& p) const {
    if (!profile_path().empty() && p.probe.measured) write_profile(profile_path(), hw_key, p);
  }

  // The fused GEMMs' self-check: the profile's (once it ran on this hardware key with these fused kernels), else this
  // process's, else run now (~1 s, before any arena exists on the first call) and cached in both; a passed check's
  // configurations are installed for fused_route(). A check that cannot run (no device memory) is not stored, but this
  // pipeline keeps its outcome: op by op for its lifetime, never switching block path between calls.
  moss_dit::FusedCheck check_fused() {
    if (fused_check) return *fused_check;
    std::optional<moss_dit::FusedCheck> c;
    if (learner && learner->stored().fused_check) {
      c = learner->stored().fused_check;
    } else {
      const std::string key = learner ? hw_key : hardware_key();
      c = process_fused_check(key);
      if (!c) {
        try {
          c = moss_dit::check_fused_gemms(streams.compute());
        } catch (const std::exception& e) {
          fused_check = moss_dit::FusedCheck{false, std::string("self-check could not run: ") + e.what(), {}};
          return *fused_check;
        }
        process_fused_check(key, &*c);
      }
      if (learner) {
        learner->stored().fused_check = c;
        save_profile(learner->stored());
      }
    }
    moss_dit::install_fused_check(*c);
    fused_check = c;
    return *c;
  }

  // The block path for this call: options.block_path unless the fused GEMMs failed their self-check.
  moss_dit::BlockPath block_path(std::string& note) {
    if (options.block_path == moss_dit::BlockPath::kOpByOp) return options.block_path;
    const moss_dit::FusedCheck c = check_fused();
    if (c.ok) return options.block_path;
    note = std::string("fused DiT blocks disabled, running op by op: ") + c.detail;
    return moss_dit::BlockPath::kOpByOp;
  }

  // What a finished denoise measured, for the next plans (ProfileLearner decides what sticks):
  //  - compute: the median step minus its streaming stalls against the plan's compute estimate gives a TFLOPS estimate;
  //  - a file-streamed DiT's median rate to the device: page-cached or cold, and the page-cache rate.
  void learn_from_denoise(const Result& r) {
    if (!learner || options.hardware) return;
    bool changed = false;
    const std::vector<double>& steps = r.timings.step_s;
    // Compute rates are learned from the Reference numerics only: the P6 paths' speedups are cost-model constants.
    const bool reference = r.tier.tier == planner::Tier::kReference;
    if (reference && steps.size() >= 3 && r.plan.dit.compute_s > 0) {
      std::vector<double> sorted(steps.begin() + 1, steps.end());  // the first step pays warm-up (GEMM plans)
      std::sort(sorted.begin(), sorted.end());
      const double stall = r.dit_stream.streamed ? r.dit_stream.stall_s / static_cast<double>(steps.size()) : 0.0;
      const double compute = sorted[sorted.size() / 2] - stall;
      if (compute > 0) changed = learner->observe_tflops(r.hardware.bf16_tflops * r.plan.dit.compute_s / compute);
    }
    // The median step's rate: the first step may read blocks the page cache did not hold yet.
    const DitStreamStats& st = r.dit_stream;
    if (st.streamed && st.source == "file" && st.steps.size() >= 2 && !options.direct_io) {
      std::vector<double> rates;
      for (std::size_t i = 1; i < st.steps.size(); ++i) rates.push_back(st.steps[i].gbps);
      std::sort(rates.begin(), rates.end());
      changed = learner->observe_dit_stream(rates[rates.size() / 2]) || changed;
    }
    if (changed) save_profile(learner->stored());
  }

  // A new encoder with a host source of the kept copy's kind adopts it; any other drops the kept copy (its RAM is not
  // planned any more).
  void ensure_te(const TeSource& t, const qwen35_text::EncoderOptions& o) {
    if (te && te_src == &t) {
      const qwen35_text::EncoderOptions& cur = te->options();
      if (cur.weight_buffers == o.weight_buffers && cur.source == o.source && cur.direct_io == o.direct_io) return;
    }
    keep_te_host();
    te.reset();
    te = t.model ? std::make_unique<qwen35_text::TextEncoder>(*t.model, *arena, streams.compute(), o)
                 : std::make_unique<qwen35_text::TextEncoder>(*t.index, *arena, streams.compute(), o);
    te_src = &t;
    const bool pinned = o.source == qwen35_text::WeightSource::kHostPinned;
    if (o.source != qwen35_text::WeightSource::kFile && te_kept_src == &t && te_kept_pinned == pinned && !te_kept.empty()) {
      te->adopt_host(te_kept);
    } else {
      drop_te_host();
    }
  }

  // A host-resident encoder's copy, loaded now if it is not yet (cancellable). A pinned copy whose allocation fails
  // (cudaHostAlloc: the OS will not lock that much) is not an error: the encoder is recreated streaming from its files,
  // the reason goes to r.te_note, and false is returned. Throws qwen35_text::Cancelled.
  bool load_te_host(const TeSource& t, int buffers, const std::atomic<bool>* cancel, Result& r) {
    if (!te || te->options().source == qwen35_text::WeightSource::kFile || te->host_resident_bytes() > 0) return true;
    try {
      te->preload_host(cancel);
      return true;
    } catch (const CudaError& e) {
      if (te->options().source != qwen35_text::WeightSource::kHostPinned) throw;
      r.te_note = std::string("its pinned host copy could not be allocated (") + e.what() +
                  "), so it streamed from its files; no more pinning for it until release_device_memory(false) or a restart";
      t.pin_failed = true;
    } catch (const std::bad_alloc&) {
      r.te_note = "its host copy could not be allocated (out of host memory), so it streamed from its files";
    }
    drop_te_host();
    ensure_te(t, te_options(t, planner::TeResidency::kStreamed, buffers, false));
    r.plan.te.residency = planner::TeResidency::kStreamed;  // what ran
    r.plan.te.pinned = false;
    return false;
  }

  // How a load of `src` from its files reads them (choose_dit_load_io on what this machine says now: the files'
  // direct-I/O support, checked once per source; their page-cache residency, per load; the learner's load rates and its
  // warm choice).
  DitIoDecision dit_direct_io(const DitSource& src, bool parked) const {
    DitIoInputs in;
    in.forced_direct = options.direct_io || options.dit_load_direct_io == true;
    in.forced_buffered = !options.direct_io && options.dit_load_direct_io == false;
    in.parked = parked;
    in.direct_refused = src.direct_refused;
    // Forced buffered reads skip the residency query: such a load is not counted as warm, so its rate never feeds the
    // warm buffered one (DitIoDecision::warm stays false).
    if (!in.forced_buffered) {
      if (!src.direct) {
        src.direct = weights::DirectIoSupport{true, {}};
        for (const std::filesystem::path& f : src.files) {
          weights::DirectIoSupport d = weights::direct_io_support(f);
          if (!d.supported) {
            src.direct = std::move(d);
            break;
          }
        }
      }
      in.support = *src.direct;
      double cached = 0, total = 0;
      bool known = !src.files.empty();
      for (const std::filesystem::path& f : src.files) {
        std::error_code ec;
        const double bytes = static_cast<double>(std::filesystem::file_size(f, ec));
        const double fraction = weights::page_cache_fraction(f);
        if (ec || fraction < 0) known = false;
        cached += bytes * std::max(0.0, fraction);
        total += bytes;
      }
      in.cached_fraction = known && total > 0 ? cached / total : -1.0;
      in.bytes = static_cast<std::uint64_t>(total);
      if (learner) {
        in.buffered_gbps = learner->dit_load_gbps(false);
        in.direct_gbps = learner->current().direct_gbps;
        in.buffered_warm_gbps = learner->current().buffered_warm_gbps;
        in.warm_direct_before = learner->current().warm_direct;
      }
    }
    return choose_dit_load_io(in);
  }

  // The plan counts the DiT and every resident model, so a resident DiT is never expected to be dropped here; when
  // it is (the arena found does not match the plan), the reload is reported as MemoryStats::dit_reloaded.
  // `stack_need` excludes the streamed blocks' device buffers (added here from `residency`). A loaded DiT with another
  // residency is replaced: a planned change, not a reload. A soft unload's host copy of `src` is read instead of the
  // files (then freed); `load` reports how the weights were read.
  void ensure_dit(DitSource& src, const moss_dit::Residency& residency, std::size_t stack_need, const std::atomic<bool>* cancel,
                  Timings& tm, DitLoadReport& load, const lora::LoraStack* merges, const std::string& merge_key) {
    stack_need += moss_dit::MossDit::stream_workspace_bytes(*src.index, residency);
    const bool same = dit && dit_src == &src && dit->residency() == residency && dit_merge_key == merge_key;
    if (same && arena->available() >= stack_need) return;
    const std::size_t persistent = moss_dit::MossDit::persistent_bytes(*src.index, residency);
    if (dit || arena->available() < persistent + stack_need) reset_arena(arena->capacity(), dit && !same);
    if (arena->available() < persistent + stack_need) {
      throw runtime::BudgetError(persistent + stack_need, arena->available());
    }
    const weights::HostFiles* host = dit_host && dit_host_src == &src ? &*dit_host : nullptr;
    DitIoDecision io;
    if (host == nullptr) io = dit_direct_io(src, dit_parked == &src);
    const bool direct = host == nullptr && io.direct;
    std::string note = std::move(io.note);
    dit_load_warm = io.warm;
    dit_parked = nullptr;
    const auto t0 = Clock::now();
    try {
      dit = src.quant
                ? std::make_unique<moss_dit::MossDit>(*src.quant, *arena, streams, direct, cancel, residency, merges, host)
                : std::make_unique<moss_dit::MossDit>(*src.index, *arena, streams, direct, cancel, residency, merges, host);
      if (time_embedding) dit->set_time_embedding(time_embedding.get());
    } catch (...) {
      // A partial load (cancelled or failed) leaves its weights in the persistent region: start over clean.
      reset_arena(arena->capacity());
      throw;
    }
    dit_src = &src;
    dit_merge_key = merge_key;
    ++mem->dit_loads;
    mem->dit_reloaded = mem->dit_reloaded || dit_lost == &src;
    dit_lost = nullptr;
    const double seconds = since(t0);
    tm.dit_load_s += seconds;
    load.source = host != nullptr ? "host_copy" : "files";
    // What the readers did (an open that refused O_DIRECT reads buffered).
    load.io = host != nullptr ? std::string() : dit->load_stats().direct ? "direct" : "buffered";
    if (direct && !dit->load_stats().direct) {
      src.direct_refused = true;
      note += (note.empty() ? "" : "; ") + std::string("direct I/O was refused: the files were read buffered");
    }
    if (host != nullptr && residency.streamed()) {
      note = "the streamed blocks read the weight files (the host copy serves the resident part)";
    }
    load.note = std::move(note);
    load.bytes = dit->load_stats().bytes;
    load.seconds = seconds;
    if (host != nullptr) {
      dit_host.reset();  // the copy served its purpose: its RAM is not planned
      dit_host_src = nullptr;
    }
    sample_memory();
  }

  void ensure_vae(const VaeSource& v, std::size_t stack_need, bool park_dit, Timings& tm) {
    if (vae && vae_src == &v && arena->available() >= stack_need && !(park_dit && dit)) return;
    const std::size_t persistent = flux2_vae::Flux2VaeDecoder::persistent_bytes(*v.index);
    if (park_dit || vae || arena->available() < persistent + stack_need) {
      // Parking: only where the plan asks for it, or (never expected with the exact workspace queries) as the
      // exact fallback when the decode does not fit; either way it is reported.
      mem->dit_parked_for_decode = mem->dit_parked_for_decode || dit != nullptr;
      reset_arena(arena->capacity(), park_dit);
    }
    if (arena->available() < persistent + stack_need) {
      throw runtime::BudgetError(persistent + stack_need, arena->available());
    }
    const auto t0 = Clock::now();
    vae = std::make_unique<flux2_vae::Flux2VaeDecoder>(*v.index, *arena, streams, false, options.direct_io, nullptr,
                                                     v.file != nullptr);
    vae_src = &v;
    tm.vae_load_s += since(t0);
    sample_memory();
  }

  // The VAE encoder, persistent next to the other resident models. The plan counts it (with the decoder) in every
  // phase, so it is expected to fit; when it does not, the arena is recreated (a resident DiT is then reloaded).
  void ensure_encoder(const EncoderSource& e, std::size_t stack_need, Timings& tm) {
    if (encoder && encoder_src == &e && arena->available() >= stack_need) return;
    if (encoder || arena->available() < e.persistent + stack_need) reset_arena(arena->capacity());
    if (arena->available() < e.persistent + stack_need) throw runtime::BudgetError(e.persistent + stack_need, arena->available());
    const auto t0 = Clock::now();
    try {
      encoder = std::make_unique<flux2_vae_encoder::Flux2VaeEncoder>(e.file, *arena, streams, options.direct_io, e.vae_file,
                                                                     e.vae_file != nullptr);
    } catch (...) {
      reset_arena(arena->capacity());  // a partial upload stays in the persistent region
      throw;
    }
    encoder_src = &e;
    tm.vae_encoder_load_s += since(t0);
    sample_memory();
  }

  // Private host memory the pipeline holds itself across calls (cache contents, text-encoder rings and host copy),
  // which the planner adds on its own and must not count twice. The planner adds the text encoder's share only when
  // it runs: on a full embedding-cache hit the encoder's held memory stays in the baseline.
  // A soft unload's DiT copy counts as held too: the call frees it when it loads the DiT, and the plan does not count it.
  std::size_t held_host_bytes(bool te_runs) const {
    std::size_t held = cache ? cache->stats().bytes : 0;
    if (dit) held += dit->host_bytes() + dit->pristine_bytes();  // rings / pinned copy, LoRA pristine weights: counted by the plan
    if (te_runs && te && te_src != nullptr) {
      qwen35_text::EncoderOptions file = te->options();
      file.source = qwen35_text::WeightSource::kFile;
      held += te_host(*te_src, file);
    }
    if (te_runs) held += te_host_bytes();  // the encoder's copy, or the one kept past it
    return held + dit_host_bytes();
  }
};

std::int64_t schedule_steps(const Request& req) {
  const FlowSampling flow{req.shift, req.flow_multiplier};
  // The dataspace schedule has one step per requested step (generate() checks them against the file's step codes).
  const int steps1 = req.sampler == SamplerKind::kDataSpace ? req.steps : make_schedule(req.scheduler, req.steps, flow).num_steps();
  const int start1 = (req.init_image == nullptr) ? 0 : strength_start_step(steps1, req.denoise_strength);
  std::int64_t total = steps1 - start1;
  if (req.hires.enabled()) {
    const int steps2 = make_schedule(req.scheduler, req.hires.steps > 0 ? req.hires.steps : req.steps, flow).num_steps();
    total += steps2 - strength_start_step(steps2, req.hires.strength);
  }
  return total;
}

Pipeline::Pipeline(PipelineOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
Pipeline::~Pipeline() = default;

ReleaseReport Pipeline::release_device_memory(bool keep_host_copy, const std::atomic<bool>* cancel) {
  Impl& m = *impl_;
  ReleaseReport rep;
  const auto t0 = Clock::now();
  if (cancel != nullptr && cancel->load()) {
    rep.note = "cancelled before it started";
    rep.cancelled = true;
    rep.dit_host_copy = m.dit_host.has_value();
    rep.te_host_copy = !m.te_kept.empty();
    rep.host_copy_bytes = m.te_host_bytes() + m.dit_host_bytes();
    rep.seconds = since(t0);
    return rep;
  }
  MemoryStats stats;  // the arena release samples into it
  m.mem = &stats;
  struct MemScope {
    Impl& m;
    ~MemScope() { m.mem = nullptr; }
  } mem_scope{m};
  // The device first: everything below is freed whatever happens to the host copy afterwards.
  const DitSource* copy_src = keep_host_copy && m.dit ? m.dit_src : nullptr;
  rep.vram_released = (m.arena ? m.arena->capacity() : 0) + (m.lora_sb ? m.lora_sb->device_bytes() : 0) +
                      (m.control_emb ? m.control_emb->device_bytes() : 0);
  m.release_arena();  // keeps a host-resident text encoder's copy (keep_te_host)
  m.dit_lost = nullptr;  // an explicit unload: the next load is planned, not a reported reload
  m.lora_sb.reset();
  m.lora_stack.reset();
  m.lora_files_key.clear();
  m.control_emb.reset();
  m.control_file.reset();
  m.control_key.clear();
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  if (!keep_host_copy) {
    m.dit_host.reset();
    m.dit_host_src = nullptr;
    m.drop_te_host();
    for (auto& [key, t] : m.te_sources) t->pin_failed = false;  // a full unload: pinning may be tried again
  } else if (copy_src != nullptr && !(m.dit_host && m.dit_host_src == copy_src)) {
    // The DiT's weight files into pinned RAM (the next load reads the same bytes from there) when the host has the room
    // (Impl::host_copy_room). A failure or a cancel only means no copy: the release itself is done.
    m.dit_host.reset();
    m.dit_host_src = nullptr;
    std::uint64_t bytes = 0;
    for (const std::filesystem::path& f : copy_src->files) {
      std::error_code ec;
      bytes += std::filesystem::file_size(f, ec);
    }
    if (const std::string why = m.host_copy_room(bytes); !why.empty()) {
      rep.note = "the DiT was not kept in RAM: " + why;
    } else {
      const auto t1 = Clock::now();
      try {
        const bool direct = m.dit_direct_io(*copy_src, false).direct;
        weights::HostFiles copy = weights::read_files_to_host(copy_src->files, true, direct, 4, cancel);
        if (cancel != nullptr && cancel->load()) {
          rep.note = "the DiT's host copy was cancelled";
        } else {
          m.dit_host = std::move(copy);
          m.dit_host_src = copy_src;
        }
      } catch (const CudaError& e) {
        rep.note = std::string("the DiT was not kept in RAM: ") + e.what();
      } catch (const weights::IoError& e) {
        rep.note = std::string("the DiT was not kept in RAM: ") + e.what();
      } catch (const std::bad_alloc&) {
        rep.note = "the DiT was not kept in RAM: out of host memory";
      } catch (const std::exception& e) {
        rep.note = std::string("the DiT was not kept in RAM: ") + e.what();
      }
      rep.host_copy_s = since(t1);
    }
  }
  rep.dit_host_copy = m.dit_host.has_value();
  rep.te_host_copy = !m.te_kept.empty();
  rep.host_copy_bytes = m.te_host_bytes() + m.dit_host_bytes();
  rep.seconds = since(t0);
  return rep;
}

Result Pipeline::generate(const Request& req, const GenerateControl& control) { return run(req, control, false); }

Result Pipeline::preload(const PreloadOptions& options) {
  Request req = options.request;
  if (tok::python_strip(req.prompt).empty()) req.prompt = "preload";  // plans for a short prompt
  GenerateControl control;
  control.on_plan = options.on_plan;
  control.on_phase = options.on_phase;
  control.cancel = options.cancel;
  return run(req, control, true);
}

// generate(), or with `preload` the same up to the plan and the arena, then the loads of the models the plan keeps.
Result Pipeline::run(const Request& req, const GenerateControl& control, bool preload) {
  Impl& m = *impl_;
  check_user_paths(m.options, req);
  validate(req);
  Result r;
  r.tier.tier = m.options.tier;  // what a run cancelled before the tier is resolved reports
  Timings& tm = r.timings;
  m.mem = &r.memory;
  r.memory.free_at_start = m.free_at_start;
  r.memory.host_private_start = planner::process_host_memory().private_bytes;
  r.memory.host_private_peak = r.memory.host_private_start;
  const auto t_total = Clock::now();
  const cudaStream_t stream = m.streams.compute();
  // Cancellation: before each phase (and after every step, below). A seen request always ends the call.
  const auto cancel_requested = [&] { return control.cancel != nullptr && control.cancel->load(); };
  const auto keep_going = [&](Phase phase) {
    if (control.on_phase && !control.on_phase(phase)) r.cancelled = true;
    if (cancel_requested()) r.cancelled = true;
    return !r.cancelled;
  };
  const auto finish = [&] {
    tm.total_s = since(t_total);
    r.memory.host_copy_bytes = m.te_host_bytes() + m.dit_host_bytes();
    r.memory.merge_memo_bytes = m.dit ? m.dit->memo_bytes() : 0;
    m.sample_memory();
    m.mem = nullptr;
    if (m.learner) m.learner->end_generation();
    return std::move(r);
  };
  // The models keep the call's cancel flag only for the call (the flag may not outlive it).
  struct CancelScope {
    Impl& m;
    ~CancelScope() {
      if (m.dit) {
        m.dit->set_cancel(nullptr);
        m.dit->set_request_route(std::nullopt);
        m.dit->set_input_extension(nullptr);  // a pass's control latents live on its stack
        m.dit->set_step_cache(nullptr);       // a pass's step cache too (run_pass also clears it)
        m.dit->set_linear_input_hook({});     // and the call's calibration hook
      }
      if (m.vae) m.vae->set_cancel(nullptr);
    }
  } cancel_scope{m};

  Bundle& b = m.bundle(req.model, tm.index_load_s);
  DitSource& src = m.dit_source(b, tm.index_load_s);
  const TeSource& tes = m.te_source(b, tm.index_load_s);
  const VaeSource& vs = m.vae_source(b, tm.index_load_s);
  r.te_source = util::path_utf8(tes.path);
  r.te_precision = qwen35_text::precision_name(tes.precision);
  r.vae_source = util::path_utf8(vs.path);
  r.dit_source = util::path_utf8(src.path);
  r.dit_linears = src.linears;
  if (m.time_embedding) r.time_embedding = util::path_utf8(m.options.time_embedding_file);
  r.gpu_arch = current_gpu_arch();
  if (m.dit_host && m.dit_host_src != &src) {
    m.dit_host.reset();  // a soft unload's copy of another DiT than this request's
    m.dit_host_src = nullptr;
  }
  // Runtime LoRA files: the request's, then a control adapter's LoRA part (its embedder tensors are read separately).
  std::vector<lora::LoraSpec> loras = req.loras;
  const control::ControlFile* ctl = nullptr;
  if (req.control.enabled()) {
    // The request's kinds against the file's, before anything is read, uploaded or released.
    const auto check_kinds = [&](const control::ControlFile& f) {
      for (const Request::ControlImage& c : req.control.images) {
        if (f.kind_index(c.kind) < 0) {
          std::string kinds;
          for (const std::string& k : f.kinds) kinds += (kinds.empty() ? "" : ", ") + k;
          throw std::invalid_argument("The control adapter has no \"" + c.kind + "\" input (its inputs: " + kinds + ").");
        }
      }
    };
    const std::string key = util::path_utf8(util::path_key(req.control.adapter));
    if (!m.control_file || m.control_key != key) {
      if (!std::filesystem::is_regular_file(req.control.adapter)) {
        throw std::invalid_argument("control adapter not found: " + util::path_utf8(req.control.adapter));
      }
      const auto t0c = Clock::now();
      auto f = std::make_unique<control::ControlFile>(control::open_control_file(req.control.adapter));
      check_kinds(*f);
      const control::EmbedderWeights ew = control::read_embedder(*f);
      // Outside the arena, as the LoRA side branches: make room when a held arena leaves too little free memory.
      const std::size_t bytes = control::DeviceEmbedder::device_bytes(ew.out_features, ew.in_features);
      if (m.dit) m.dit->set_input_extension(nullptr);
      m.control_emb.reset();
      if (m.arena && runtime::DeviceArena::free_device_bytes() < bytes + m.options.vram_reserve) m.release_arena();
      std::unique_ptr<control::DeviceEmbedder> e;
      try {
        e = std::make_unique<control::DeviceEmbedder>(ew);
      } catch (const CudaError&) {
        if (!m.arena) throw;
        m.release_arena();
        e = std::make_unique<control::DeviceEmbedder>(ew);
      }
      m.control_file = std::move(f);
      m.control_emb = std::move(e);
      m.control_key = key;
      tm.index_load_s += since(t0c);
    }
    ctl = m.control_file.get();
    check_kinds(*ctl);
    m.control_ext.weight = m.control_emb->weight();
    m.control_ext.bias = m.control_emb->bias();
    loras.push_back({req.control.adapter, req.control.lora_strength, ctl->consumed()});
    r.control.adapter = util::path_utf8(req.control.adapter);
    r.control.kinds = ctl->kinds;
    r.control.base_model = ctl->base_model;
  } else {
    if (m.dit) m.dit->set_input_extension(nullptr);
    m.control_emb.reset();
    m.control_file.reset();
    m.control_key.clear();
  }
  const bool lora_requested = !loras.empty();
  {
    // Quality tier from what is loaded, before anything heavy (its refusals throw std::invalid_argument here, ahead of
    // the LoRA files too). The lossy accelerators are an explicit speed opt-in: the tier is resolved without them.
    const std::filesystem::path te_file = tes.model ? tes.path : std::filesystem::path();
    const auto supported = [](planner::AttentionMode mode) { return moss_dit::attention_supported(dit_attention(mode)); };
    r.tier = planner::resolve_tier(m.options.tier, m.options.attention, src.precision, supported,
                                   tes.precision != qwen35_text::Precision::kBF16,
                                   src.precision == planner::Precision::kNvfp4 && kernels::nvfp4::available());
    if (lora_requested) r.tier.reason += "; runtime LoRAs active";
    if (req.accel.step_cache_on()) {
      // An unset threshold takes the model's default (0.02 on the base model, told apart by its DiT's input embedder:
      // dit_lineage; 0.03 otherwise); one the user set stands.
      r.accel.step_cache_threshold = req.accel.step_cache_threshold.value_or(accel::default_fbcache_threshold(src.lineage));
      r.accel.step_cache_threshold_source = accel::threshold_source(req.accel, src.lineage);
    }
    if (std::string w = accel::base_model_threshold_warning(req.accel, src.lineage); !w.empty()) {
      r.accel_warnings.push_back(std::move(w));
    }
    if (req.accel.active(req.cfg != 1.0)) {
      // P6b: the tier above was chosen without them (the Reference tier's exactness does not hold with them either).
      accel::Options ran = req.accel;
      if (ran.step_cache_on()) ran.step_cache_threshold = r.accel.step_cache_threshold;
      r.tier.reason += "; lossy accelerators active (" + ran.describe(req.cfg != 1.0) + ")";
    }
    // The report's quality object (quality_report_json): what ran and the models it ran.
    QualityReport& q = r.quality;
    q.tier = r.tier.tier;
    q.config = {r.tier.precision, r.tier.attention, tes.precision, lora_requested,
                src.precision == planner::Precision::kNvfp4 && moss_dit::int8_gemm_fraction(src.quant.get()) > 0};
    q.gpu_arch = r.gpu_arch;
    q.dit = src.path;
    q.dit_bytes = weight_bytes(src.path);
    q.te = te_file;
    q.te_bytes = te_file.empty() ? 0 : weight_bytes(te_file);
  }
  // Runtime LoRAs: the stack is kept while its files (paths, sizes and modification times) and the DiT source stay the
  // same; a file replaced under the same path loads anew, and so do its merges and their memo. A strength change
  // updates the side branches in place and re-merges merged adapters from the pristine weights. The loaded DiT carries
  // merged adapters of the files in `merge_key`. A new stack is built into locals and committed only once it is complete.
  std::string merge_key;
  if (!lora_requested) {
    if (m.dit) m.dit->set_lora(nullptr);
    m.lora_sb.reset();
    m.lora_stack.reset();
    m.lora_files_key.clear();
  } else {
    std::string files = util::path_utf8(src.path) + (m.options.lora_strict ? "|strict" : "");
    // The control adapter's entry is tagged: the same file as a plain LoRA is another stack (and is refused below).
    for (std::size_t i = 0; i < loras.size(); ++i) {
      files += (ctl != nullptr && i + 1 == loras.size() ? "|control:" : "|") +
               util::path_utf8(util::path_key(loras[i].path)) + "#" + file_stamp(loras[i].path);
    }
    if (!m.lora_stack || files != m.lora_files_key) {
      for (const lora::LoraSpec& l : req.loras) {
        if (control::is_control_file(l.path)) {
          throw std::invalid_argument(util::path_utf8(l.path) + " is a control adapter (a widened DiT input embedder): pass it as "
                                      "the request's control adapter (Request::control), not as a LoRA.");
        }
      }
      const auto t0l = Clock::now();
      lora::LoadOptions lo;
      lo.strict = m.options.lora_strict;
      lo.int8_block_gemms = src.precision != planner::Precision::kBf16;
      auto stack = std::make_unique<lora::LoraStack>(loras, lora::moss_dit_targets(), lo);
      moss_dit::check_lora_merges(*stack, src.quant.get());  // before anything is released or loaded
      // The side branches live outside the arena: make room first when a held arena leaves too little free memory
      // (the arena is recreated at the plan's size below), and retry once if the allocation still fails.
      const std::size_t sb_bytes = lora::SideBranches::device_bytes(*stack);
      if (m.dit) m.dit->set_lora(nullptr);
      m.lora_sb.reset();
      std::unique_ptr<lora::SideBranches> sb;
      if (m.arena && runtime::DeviceArena::free_device_bytes() < sb_bytes + m.options.vram_reserve) m.release_arena();
      try {
        sb = std::make_unique<lora::SideBranches>(*stack);
      } catch (const CudaError&) {
        if (!m.arena) throw;
        m.release_arena();
        sb = std::make_unique<lora::SideBranches>(*stack);
      }
      m.lora_stack = std::move(stack);
      m.lora_sb = std::move(sb);
      m.lora_files_key = files;
      tm.lora_load_s += since(t0l);
    } else {
      bool changed = false;
      for (int i = 0; i < m.lora_stack->files(); ++i) {
        const float strength = loras[static_cast<std::size_t>(i)].strength;
        if (m.lora_stack->strength(i) != strength) {
          m.lora_stack->set_strength(i, strength);
          changed = true;
        }
      }
      if (changed) {
        m.lora_sb->update_strengths(*m.lora_stack);  // between forwards: nothing reads them now
        // Merged adapters: re-merged from the pristine weights kept at load, in place (no reload).
        if (m.dit && m.dit_merge_key == files && !m.lora_stack->merge_modules().empty()) {
          const auto t0m = Clock::now();
          try {
            m.dit->remerge_lora(*m.lora_stack);
          } catch (...) {
            m.drop_dit_after_failed_merge();
            throw;
          }
          tm.lora_merge_s += since(t0m);
        }
      }
    }
    if (!m.lora_stack->merge_modules().empty()) merge_key = files;
    r.loras = m.lora_stack->reports();
    if (ctl != nullptr) {
      // The control adapter's LoRA part is the stack's last file: reported with the control (its notes and warnings
      // too), not in `loras`.
      r.control.lora = r.loras.back();
      r.loras.pop_back();
    }
    for (const lora::FileReport& f : r.loras) {
      for (const std::string& w : f.warnings) r.lora_warnings.push_back(f.file + ": " + w);
      for (const std::string& n : f.notes) r.lora_notes.push_back(f.file + ": " + n);
    }
    if (ctl != nullptr) {
      for (const std::string& mod : m.lora_stack->merge_modules()) {
        if (mod == "video_embedder") r.lora_warnings.push_back("video_embedder: LoRA delta replaced by the control adapter's embedder");
      }
    }
    if (m.time_embedding) {
      for (const std::string& w : moss_dit::shadowed_by_time_embedding(*m.lora_stack)) r.lora_warnings.push_back(w);
    }
  }
  if (req.seed >= 0) {
    r.seed = static_cast<std::uint64_t>(req.seed);
  } else {
    std::random_device rd;
    r.seed = std::uniform_int_distribution<std::uint64_t>(0, (std::uint64_t{1} << 32) - 1)(rd);
  }
  Guidance g;
  g.cfg = req.cfg;
  g.uncond_skip = skip_set(req.uncond_skip_layers);
  g.slg_layers = skip_set(req.slg_layers);
  g.slg_scale = req.slg_scale;
  g.slg_start = req.slg_start;
  g.slg_stop = req.slg_stop;
  g.cfg_truncation = req.accel.cfg_truncation;
  // Schedule (pipeline/samplers): the official one for kMoss, the scheduler's otherwise; the dataspace sampler's from its
  // step codes.
  const FlowSampling flow{req.shift, req.flow_multiplier};
  const DataSpaceParams* dataspace = nullptr;
  Schedule schedule;
  if (req.sampler == SamplerKind::kDataSpace) {
    dataspace = &m.dataspace(req.dataspace_path);
    schedule.sigmas = dataspace_sigmas(*dataspace, req.steps);
    schedule.timesteps.assign(dataspace->step_codes.begin(), dataspace->step_codes.begin() + req.steps);
    schedule.scheduler = SchedulerKind::kSimple;  // not kMoss: the data-space arithmetic
    schedule.flow = flow;
  } else {
    schedule = make_schedule(req.scheduler, req.steps, flow);
  }
  SamplerOptions sampler_options;
  sampler_options.noise = req.noise_source;
  sampler_options.eta = req.eta;
  sampler_options.s_noise = req.s_noise;
  sampler_options.flow = flow;
  sampler_options.dataspace = dataspace;
  // img2img / inpainting / hires fix (P7b): the request's pass at its size, from noise or from the init image, and the
  // hires fix's second pass at its target size from the upscaled first image. Each pass runs the tail of its schedule
  // that its denoise strength leaves (diffusers get_timesteps); strength 1 without a mask needs no encode at all.
  const bool hires = req.hires.enabled();
  const bool inpaint = req.mask != nullptr;
  const bool img2img = req.init_image != nullptr && (req.denoise_strength < 1.0 || inpaint);
  const auto crop_schedule = [](Schedule s, int start) {
    s.sigmas.erase(s.sigmas.begin(), s.sigmas.begin() + start);
    s.timesteps.erase(s.timesteps.begin(), s.timesteps.begin() + start);
    return s;
  };
  const int H1 = req.height, W1 = req.width;
  const int H2 = hires ? req.hires.height : H1, W2 = hires ? req.hires.width : W1;
  r.image_mode = inpaint ? "inpaint" : img2img ? "img2img" : "txt2img";
  if (req.init_image != nullptr && !img2img) {
    r.image_note = "denoise strength 1 without a mask samples from pure noise: the init image was not used";
  }
  const int start1 = (req.init_image == nullptr) ? 0 : strength_start_step(schedule.num_steps(), req.denoise_strength);
  const Schedule sched1 = crop_schedule(schedule, start1);
  r.first_pass.width = W1;
  r.first_pass.height = H1;
  r.first_pass.schedule_steps = schedule.num_steps();
  r.first_pass.start_step = start1;
  r.first_pass.strength = (req.init_image == nullptr) ? 1.0 : req.denoise_strength;
  Schedule sched2;
  if (hires) {
    const Schedule full = make_schedule(req.scheduler, req.hires.steps > 0 ? req.hires.steps : req.steps, flow);
    const int start2 = strength_start_step(full.num_steps(), req.hires.strength);
    sched2 = crop_schedule(full, start2);
    r.hires.width = W2;
    r.hires.height = H2;
    r.hires.schedule_steps = full.num_steps();
    r.hires.start_step = start2;
    r.hires.strength = req.hires.strength;
  }
  const bool need_dit = sched1.num_steps() > 0 || (hires && sched2.num_steps() > 0);
  // A loaded DiT whose merged adapters differ from this request's (e.g. a control adapter, with its merged text-refiner
  // adapters, toggled on or off): switched in place from the pristine copies when no block module is newly merged,
  // the same bits as a reload without its cost. A failure drops the DiT (the next load starts clean).
  if (need_dit && m.dit && m.dit_src == &src && m.dit_merge_key != merge_key) {
    const auto t0m = Clock::now();
    // The stack and its strengths name the merged bits: switching back to them later stores the DiT's kept copy.
    std::string memo_key;
    if (!merge_key.empty()) {
      memo_key = merge_key;
      for (int i = 0; i < m.lora_stack->files(); ++i) {
        char strength[32];
        std::snprintf(strength, sizeof(strength), "|%a", static_cast<double>(m.lora_stack->strength(i)));
        memo_key += strength;
      }
    }
    try {
      if (m.dit->rebase_merges(merge_key.empty() ? nullptr : m.lora_stack.get(), memo_key)) m.dit_merge_key = merge_key;
    } catch (...) {
      m.drop_dit_after_failed_merge();
      throw;
    }
    tm.lora_merge_s += since(t0m);
  }
  // Control images are encoded once per pass size (not without a sampling step).
  const bool ctl_encode = ctl != nullptr && !req.control.images.empty() && need_dit;
  const bool need_encoder = img2img || hires || ctl_encode;
  const EncoderSource* enc_src = need_encoder ? &m.encoder_source(b, tm.index_load_s) : nullptr;
  if (enc_src != nullptr) r.vae_encoder_source = util::path_utf8(enc_src->file);
  r.height = H1;
  r.width = W1;
  r.steps_requested = req.steps;
  const int total_steps = static_cast<int>(schedule_steps(req));
  r.steps_scheduled = total_steps;

  // Tokenize (P:321-327). The negative prompt is encoded only with CFG (P:388-394).
  auto t0 = Clock::now();
  std::vector<std::vector<std::int32_t>> seqs;
  seqs.push_back(tok::encode_moss_prompt_report(b.tokenizer, req.prompt, static_cast<std::size_t>(req.max_sequence_length),
                                                r.cond_token_report));
  if (g.do_cfg()) {
    seqs.push_back(tok::encode_moss_prompt_report(b.tokenizer, req.negative_prompt,
                                                  static_cast<std::size_t>(req.max_sequence_length), r.uncond_token_report));
  }
  tm.tokenize_s = since(t0);
  const std::int64_t sc = static_cast<std::int64_t>(seqs[0].size());
  const std::int64_t su = g.do_cfg() ? static_cast<std::int64_t>(seqs[1].size()) : 0;
  r.cond_tokens = static_cast<int>(sc);
  r.uncond_tokens = static_cast<int>(su);

  // Embedding cache: sequences it serves skip the text encoder. Without a sampling step (denoise strength 0) nothing
  // needs the embeddings: the text encoder does not run.
  t0 = Clock::now();
  std::vector<EmbeddingKey> keys(seqs.size());
  std::vector<std::shared_ptr<const Embedding>> embeds(seqs.size());
  std::vector<std::size_t> missing;
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    keys[i] = EmbeddingKey{tes.identity, tes.numerics, req.max_sequence_length, seqs[i]};
    if (m.cache && need_dit && !preload) embeds[i] = m.cache->find(keys[i]);  // a preload plans for a new prompt
    if (embeds[i]) {
      ++r.te_cache_hits;
    } else if (need_dit) {
      missing.push_back(i);
    }
  }
  r.te_skipped = missing.empty();
  std::vector<std::int64_t> te_lengths;
  for (const std::size_t i : missing) te_lengths.push_back(static_cast<std::int64_t>(seqs[i].size()));

  // Machine profile and block path (the first call probes and self-checks; both are cached).
  r.hardware = m.hardware(*src.index);
  const moss_dit::BlockPath block_path = m.block_path(r.block_path_note);
  // One fused route for the whole request: it sizes the denoise workspace and every step runs under it, so a poisoning
  // meanwhile (another pipeline's failed self-check) applies from the next request on instead of failing a step.
  const moss_dit::FusedRouteSnapshot route =
      block_path == moss_dit::BlockPath::kOpByOp ? moss_dit::FusedRouteSnapshot{} : moss_dit::fused_route_snapshot();
  const moss_dit::AttentionMode attention = dit_attention(r.tier.attention);
  // The side branches run on the quantized block path (its fused LoRA GEMMs; on NVFP4 sites a stored Δ).
  const moss_dit::ComputeConfig dit_config =
      moss_dit::MossDit::compute_config(src.quant.get(), attention, moss_dit::Int4Route::kAuto, m.lora_sb.get());

  // Exact stack need of the phases this request runs, and the plan.
  const auto latent_bytes_of = [](int H, int W) {
    return align_up(static_cast<std::size_t>(moss_dit::kLatentChannels) * static_cast<std::size_t>(H / 16) *
                    static_cast<std::size_t>(W / 16) * sizeof(float));
  };
  const auto image_bytes_of = [](int H, int W) { return static_cast<std::size_t>(H) * static_cast<std::size_t>(W) * 3; };
  const auto bf16_rows = [](std::int64_t s, std::int64_t c) { return align_up(static_cast<std::size_t>(s * c) * 2); };
  const std::size_t refined_bytes = bf16_rows(sc, moss_dit::kDim) + (g.do_cfg() ? bf16_rows(su, moss_dit::kDim) : 0);
  const std::size_t refine_need = bf16_rows(sc, moss_dit::kTextDim) + (g.do_cfg() ? bf16_rows(su, moss_dit::kTextDim) : 0) +
                                  moss_dit::MossDit::encode_text_workspace_bytes(std::max(sc, su));
  const std::vector<std::int64_t> row_lengths = denoise_row_lengths(sc, su, g);
  // The denoise phase of one pass at H x W: the refined text, then the larger of the refiner and (latents, a latent
  // preview's buffers, the sampling loop's own buffers and the forward for the loop's largest row set under this
  // request's route and config). A pass starting from an image also uploads z0 next to the latents before sampling.
  const auto denoise_need_of = [&](int H, int W, bool from_image) {
    const std::int64_t lh = H / 16, lw = W / 16;
    const std::size_t lat = latent_bytes_of(H, W);
    // P6b step cache: its residual buffers sit below the forwards on the stack, for every distinct row of the loop.
    const std::size_t loop_buffers =
        denoise_loop_bytes(lh, lw, sc, su, g, req.sampler) +
        (req.accel.step_cache_on()
             ? accel::FirstBlockCache::workspace_bytes(lh, lw, static_cast<int>(row_lengths.size()))
             : 0);
    const std::size_t forward_bytes = moss_dit::MossDit::workspace_bytes(lh, lw, row_lengths, block_path, dit_config, &route);
    // A control adapter's input latents [G * 128, h, w] (moss_dit::InputExtension::extra), next to the latents.
    const std::size_t control_bytes = ctl != nullptr ? static_cast<std::size_t>(ctl->groups()) * lat : 0;
    // A latent preview: the previous latents, the x0 estimate, the preview's scratch and its RGB image.
    const std::size_t preview_bytes =
        control.on_preview ? 2 * lat + align_up(latent_preview::workspace_bytes(lh, lw)) + align_up(image_bytes_of(H, W)) : 0;
    return refined_bytes +
           std::max(refine_need, lat + control_bytes + preview_bytes + std::max(loop_buffers + forward_bytes, from_image ? lat : 0));
  };
  std::size_t denoise_need = 0;
  if (sched1.num_steps() > 0) denoise_need = denoise_need_of(H1, W1, img2img);
  if (hires && sched2.num_steps() > 0) denoise_need = std::max(denoise_need, denoise_need_of(H2, W2, true));
  // VAE encodes (latents + the smallest exact tiling; at run time the encode takes the fastest tiling the arena leaves):
  // the init image at the request size, the upscaled first image (and, inpainting, the init image) at the hires size.
  std::size_t encode_need = 0;
  const auto encode_need_of = [&](int H, int W) {
    return latent_bytes_of(H, W) + flux2_vae_encoder::Flux2VaeEncoder::plan_min_memory(H, W).workspace_bytes;
  };
  if (img2img || (ctl_encode && sched1.num_steps() > 0)) encode_need = encode_need_of(H1, W1);
  if (hires) encode_need = std::max(encode_need, encode_need_of(H2, W2));
  // The decodes that run: the hires fix's first pass (for the upscale) and the final one; the plan sizes the larger.
  int dec_h = 0, dec_w = 0;
  const auto consider_decode = [&](int H, int W) {
    if (std::int64_t{H} * W > std::int64_t{dec_h} * dec_w) dec_h = H, dec_w = W;
  };
  if (hires) consider_decode(H1, W1);
  if (req.decode) consider_decode(H2, W2);
  const bool need_decoder = dec_h > 0;
  // Exact GroupNorm statistics: the untiled decode and every tiled one the plan may pick from the free VRAM give the
  // same bytes, so the image does not depend on the memory budget.
  constexpr flux2_vae::DecodeMode decode_mode = flux2_vae::DecodeMode::reference();
  static_assert(!decode_mode.torch_group_norm, "torch's GroupNorm cannot be tiled");

  planner::PlanInputs in;
  in.shapes = planner::shapes_from_index(b.index, src.quant ? src.index : nullptr, src.precision,
                                         tes.model ? &tes.model->index() : nullptr, vs.file);
  in.shapes.dit_bytes = moss_dit::MossDit::persistent_bytes(*src.index);
  if (src.quant) in.shapes.dit_int8_fraction = moss_dit::int8_gemm_fraction(src.quant.get());
  if (src.quant) in.shapes.dit_int4_fraction = moss_dit::int4_gemm_fraction(src.quant.get());
  if (src.quant) in.shapes.dit_nvfp4_fraction = moss_dit::nvfp4_gemm_fraction(src.quant.get());
  if (src.quant && m.lora_sb) {
    // INT4 sites with an active LoRA side branch run the INT8 LoRA GEMMs: costed at INT8 speed.
    const double f = moss_dit::int4_side_branch_fraction(src.quant.get(), *m.lora_sb);
    in.shapes.dit_int4_fraction -= f;
    in.shapes.dit_int8_fraction = moss_dit::int8_gemm_fraction(src.quant.get()) + f;
  }
  in.shapes.dit_block_bytes = align_up(moss_dit::BlockWeights::block_device_bytes(*src.index));  // the buffers' size
  // The VAE side of the persistent region: the decoder and the encoder this request uses, plus those an earlier
  // request left resident (counted in every phase; when that alone makes the request infeasible, plan without them and
  // recreate the arena, dropping them, below).
  const std::size_t dec_bytes = flux2_vae::Flux2VaeDecoder::persistent_bytes(*vs.index);
  const std::size_t needed_vae = (need_decoder ? dec_bytes : 0) + (enc_src != nullptr ? enc_src->persistent : 0);
  const std::size_t held_vae = (!need_decoder && m.vae ? dec_bytes : 0) +
                               (enc_src == nullptr && m.encoder && m.encoder_src != nullptr ? m.encoder_src->persistent : 0);
  in.shapes.vae_bytes = needed_vae + held_vae > 0 ? needed_vae + held_vae : dec_bytes;
  in.request.height = H2;
  in.request.width = W2;
  in.request.text_lengths = {sc};
  if (g.do_cfg()) in.request.text_lengths.push_back(su);
  {
    // DiT evaluations (heun: 2 per step); the first pass of a hires fix counted at the target's cost by area.
    const double area = static_cast<double>(H1) * W1 / (static_cast<double>(H2) * W2);
    const int steps = (hires ? sched2.num_steps() : 0) + static_cast<int>(std::ceil(sched1.num_steps() * area));
    in.request.steps = steps * model_calls_per_step(req.sampler);
  }
  in.request.rows_per_step = 1 + (g.do_cfg() ? 1 : 0) + (g.slg_scale > 0 && g.slg_layers.any() ? 1 : 0);
  in.request.te_lengths = te_lengths;
  in.request.decode = need_decoder;
  in.request.te_cached = missing.empty();
  in.request.weights_cached = m.weights_cached();
  in.request.dit_weights_cached = m.dit_weights_cached();
  in.request.vae_resident = needed_vae + held_vae > 0;
  in.workspaces.text_encoder = [&](planner::TeResidency residency, int buffers) {
    return m.te_workspace(tes, te_lengths, m.te_options(tes, residency, buffers, false));
  };
  // The hires fix decodes its first pass next to the resident DiT (exact tiling, never parking it), so the DiT is loaded
  // once per request: the smallest tiled decode of the first pass counts in the denoise phase, where the DiT is resident.
  const std::size_t first_decode_need =
      hires ? latent_bytes_of(H1, W1) +
                  flux2_vae::Flux2VaeDecoder::plan_tiled_min_memory(vs.config, H1 / 16, W1 / 16, decode_mode, 8).workspace_bytes
            : 0;
  in.workspaces.denoise = std::max({denoise_need, encode_need, first_decode_need});
  if (need_decoder) {
    const std::int64_t dh = dec_h / 16, dw = dec_w / 16;
    const std::size_t dec_latent = latent_bytes_of(dec_h, dec_w);
    in.workspaces.decode = dec_latent + flux2_vae::Flux2VaeDecoder::workspace_bytes(vs.config, dh, dw, decode_mode);
    in.workspaces.decode_tiled = [&, dh, dw, dec_latent](std::size_t budget) -> std::optional<planner::TiledDecode> {
      std::optional<flux2_vae::TilePlan> tp;
      if (budget == 0) {
        tp = flux2_vae::Flux2VaeDecoder::plan_tiled_min_memory(vs.config, dh, dw, decode_mode, 8);
      } else if (budget > dec_latent) {
        tp = flux2_vae::Flux2VaeDecoder::plan_tiled(vs.config, dh, dw, decode_mode, budget - dec_latent);
      }
      if (!tp) return std::nullopt;
      return planner::TiledDecode{dec_latent + tp->workspace_bytes, tp->relative_cost, tp->tile};
    };
  }
  std::size_t embed_bytes = 0;
  for (const auto& s : seqs) embed_bytes += s.size() * qwen35_text::kHidden * 2;
  {
    // Host buffers of the generation: embeddings, noise and latents, the final and preview images, and the image
    // work of img2img / inpainting / hires (resized images, z0, the upscaled first image).
    const std::size_t lat = std::max(latent_bytes_of(H1, W1), latent_bytes_of(H2, W2));
    const std::size_t img = std::max(image_bytes_of(H1, W1), image_bytes_of(H2, W2));
    in.workspaces.host_buffers = embed_bytes + 2 * lat + (req.decode ? img : 0) + (control.on_preview ? img : 0);
    if (req.init_image != nullptr || hires) in.workspaces.host_buffers += 4 * lat + 3 * img;
  }
  // The pristine copies of LoRA-merged modules the DiT keeps on the host (held_host_bytes takes a loaded DiT's out of
  // the baseline, so they count once).
  if (!merge_key.empty()) in.workspaces.host_buffers += moss_dit::MossDit::pristine_bytes(*m.lora_stack, src.quant.get());
  in.caps.te_max_buffers = 2;
  in.caps.te_ring_bytes = Impl::te_host(tes, m.te_options(tes, planner::TeResidency::kStreamed, 2, false));
  in.caps.te_host_resident = true;
  in.caps.te_host_pinned = !tes.pin_failed;
  in.caps.te_host_pageable = !tes.model;  // a quantized encoder held in RAM is always pinned (te_options)
  in.caps.te_quantized = tes.model != nullptr;
  in.caps.te_pin = m.options.te_host_pin;
  if (tes.pin_failed) {
    r.te_note = "pinning disabled for this text encoder after a failed pinned allocation (until "
                "release_device_memory(false) or a restart)";
  }
  in.caps.dit_parking = need_dit;
  in.caps.dit_streaming = need_dit;
  in.caps.dit_max_buffers = 3;
  {
    moss_dit::Residency file;
    file.resident_blocks = 0;
    in.caps.dit_ring_bytes = moss_dit::BlockWeights::host_bytes(*src.index, file);
  }
  in.caps.dit_host_pinned = true;
  in.caps.int4_mma_native = moss_dit::int4_mma_native();
  in.caps.nvfp4_native = src.precision == planner::Precision::kNvfp4 && kernels::nvfp4::available();
  if (m.lora_stack) {
    for (int bi = 0; bi < moss_dit::kLayers; ++bi) {
      if (moss_dit::MossDit::block_has_merges(*m.lora_stack, bi)) in.caps.dit_file_source = false;
    }
  }
  in.caps.dit_graph_bytes = block_path == moss_dit::BlockPath::kFusedGraphs ? moss_dit::graph_cache_bytes() : 0;
  in.hardware = r.hardware;
  in.budgets.vram_free = runtime::DeviceArena::free_device_bytes() + (m.arena ? m.arena->capacity() : 0);
  in.budgets.vram_reserve = m.options.vram_reserve;
  in.budgets.vram_budget = m.options.vram_budget;
  // The LoRA side branches hold device memory outside the arena, as the context does: it counts against the budget.
  if (m.lora_sb) in.budgets.context_bytes += m.lora_sb->device_bytes();
  if (m.control_emb) in.budgets.context_bytes += m.control_emb->device_bytes();
  if (ctl_encode) {
    // Control images: per kind its sized RGB copy and encoded latent (at the larger pass size).
    in.workspaces.host_buffers += req.control.images.size() * (std::max(latent_bytes_of(H1, W1), latent_bytes_of(H2, W2)) +
                                                                std::max(image_bytes_of(H1, W1), image_bytes_of(H2, W2)));
  }
  const std::size_t held_host = m.held_host_bytes(!missing.empty());
  // Of it, what the plan counts again (and so would replace): not a soft unload's DiT copy, nor a held DiT's host
  // memory when this request plans no DiT.
  std::size_t counted_again = held_host - std::min(held_host, m.dit_host_bytes());
  if (!need_dit && m.dit) counted_again -= std::min(counted_again, m.dit->host_bytes() + m.dit->pristine_bytes());
  // What the pipeline holds and the plan counts again would be freed for it: available to the plan too (otherwise a
  // host-resident text encoder's own copy would count against keeping it on the next request).
  const std::size_t host_available = planner::host_available_bytes();
  in.budgets.host_available = host_available == 0 ? 0 : host_available + counted_again;  // 0: unknown
  in.budgets.host_total = planner::host_total_bytes();
  in.budgets.host_private_now = r.memory.host_private_start > held_host ? r.memory.host_private_start - held_host : 0;
  in.budgets.host_budget = m.options.host_ram_budget;
  in.mode = m.options.mode;
  in.attention = r.tier.attention;
  in.cache_disk_spill = m.cache && m.cache->disk_enabled();
  if (!need_dit) {
    // No sampling step (denoise strength 0): the DiT is neither loaded nor planned (the planner wants a DiT shape: one
    // byte, resident).
    in.shapes.dit_bytes = 1;
    in.shapes.dit_block_bytes = 0;
    in.caps.dit_graph_bytes = 0;
    in.caps.dit_ring_bytes = 0;
  }
  // Throws planner::BudgetError before any heavy work.
  bool drop_resident = false;
  try {
    r.plan = planner::plan(in);
  } catch (const planner::BudgetError&) {
    if (held_vae == 0) throw;
    in.shapes.vae_bytes = needed_vae > 0 ? needed_vae : dec_bytes;
    in.request.vae_resident = needed_vae > 0;
    r.plan = planner::plan(in);
    drop_resident = true;
  }
  // A loaded DiT stays (in its held arena, with its residency) unless the request's own plan saves more than reloading
  // it costs: no reload churn when request sizes alternate or the free device memory moves a little.
  bool kept = false;
  if (need_dit && m.arena && m.dit && m.dit_src == &src && m.dit_merge_key == merge_key && !drop_resident) {
    const moss_dit::Residency& loaded = m.dit->residency();
    planner::HeldDit held;
    held.arena_bytes = m.arena->capacity();
    held.graph_bytes = m.dit->graph_bytes();
    if (loaded.streamed()) {
      held.dit.residency = planner::DitResidency::kStreamed;
      held.dit.resident_blocks = loaded.resident_blocks;
      held.dit.buffers = loaded.buffers;
      held.dit.source = loaded.source == moss_dit::StreamSource::kHostPinned ? planner::DitSource::kHostPinned : planner::DitSource::kFile;
    }
    r.plan = planner::prefer_held(in, std::move(r.plan), held, &kept);
  }
  r.plan.explanation += "\nQuality: " + r.tier.reason + "; DiT " + r.dit_linears + " from " + r.dit_source + ".";
  if (!src.remarks.empty()) r.plan.explanation += "\nDiT: " + src.remarks + ".";
  if (m.time_embedding) {
    r.plan.explanation += "\nDiT step embedder replaced (fp32) from " + util::path_utf8(m.options.time_embedding_file) + ".";
  }
  if (!need_dit) r.plan.explanation += "\nNo sampling step (denoise strength 0): the DiT is neither planned nor loaded.";
  if (req.init_image != nullptr || hires) {
    std::string what = r.image_mode;
    if (!r.image_note.empty()) what += " (" + r.image_note + ")";
    if (img2img) {
      what += " from a " + std::to_string(req.init_image->width) + "x" + std::to_string(req.init_image->height) + " image (" +
              resize_mode_name(req.resize) + ", strength " + json_number(req.denoise_strength) + ": " +
              std::to_string(sched1.num_steps()) + " of " + std::to_string(schedule.num_steps()) + " steps)";
    }
    if (hires) {
      what += "; hires fix " + std::to_string(W1) + "x" + std::to_string(H1) + " -> " + std::to_string(W2) + "x" +
              std::to_string(H2) + " (strength " + json_number(req.hires.strength) + ": " + std::to_string(sched2.num_steps()) +
              " of " + std::to_string(r.hires.schedule_steps) + " steps), the first pass decoded next to the DiT (tiled as "
              "needed: the DiT is loaded once)";
    }
    r.plan.explanation += "\nImage: " + what + "; VAE encoder from " + r.vae_encoder_source + ".";
  }
  if (ctl != nullptr) {
    std::string inputs;
    for (const Request::ControlImage& c : req.control.images) {
      inputs += (inputs.empty() ? "" : ", ") + c.kind + " (strength " + json_number(c.strength) + ", steps " +
                json_number(c.start) + "-" + json_number(c.end) + ")";
    }
    r.plan.explanation += "\nControl adapter " + r.control.adapter + " (LoRA strength " + json_number(req.control.lora_strength) +
                          "): " + (inputs.empty() ? std::string("no images (zero control latents)") : inputs) + ".";
  }
  const planner::ExecutionPlan& plan = r.plan;
  if (m.cache) m.cache->set_max_bytes(plan.cache.host_bytes);
  // A DiT's merge memo (a control adapter's switch-back bits) goes when the host is tight (merge_memo_unaffordable). It
  // only saves a merge.
  if (m.dit && m.dit->memo_bytes() > 0 && merge_memo_unaffordable(plan, m.options.host_ram_budget)) m.dit->drop_merge_memo();
  tm.plan_s = since(t0);
  if (control.on_plan) control.on_plan(plan);

  // The plan's arena. A larger existing one is kept (its resident models with it) unless the request needs less than
  // arena_shrink_below of it and the plan does not keep the loaded DiT.
  // Without a sampling step a held DiT is kept too (the arena is not shrunk under it).
  const bool shrink = m.arena && !kept && need_dit &&
                      static_cast<double>(plan.arena_bytes) < m.options.arena_shrink_below * static_cast<double>(m.arena->capacity());
  if (!m.arena || m.arena->capacity() < plan.arena_bytes || drop_resident || shrink) m.reset_arena(plan.arena_bytes);
  m.arena->reset_high_water();
  r.memory.arena_capacity = std::max(r.memory.arena_capacity, m.arena->capacity());
  m.sample_memory();

  const moss_dit::Residency residency = m.dit_residency(plan.dit);
  // Loads the DiT when it is not resident (first use, or parked / dropped since) and sets this request's options.
  const auto prepare_dit = [&]() {
    const int loads_before = r.memory.dit_loads;
    try {
      m.ensure_dit(src, residency, denoise_need, control.cancel, tm, r.dit_load, merge_key.empty() ? nullptr : m.lora_stack.get(),
                   merge_key);
    } catch (const moss_dit::Cancelled&) {
      r.cancelled = true;
      return false;
    }
    m.dit->set_cancel(control.cancel);
    if (control.dit_linear_inputs) m.dit->set_linear_input_hook(control.dit_linear_inputs);
    // A DiT read buffered from the weight files: page-cached or cold (the loader's one reader). Direct reads and a soft
    // unload's host copy bypass the page cache, so they say nothing about it.
    // Each mode's load rate is kept, buffered ones from a warm cache apart (they price loads and decide the warm mode;
    // on Windows direct and buffered decide the mode).
    const moss_dit::LoadStats& ls = m.dit->load_stats();
    if (r.memory.dit_loads > loads_before && m.learner && r.dit_load.source == "files" && ls.seconds > 0 &&
        ls.bytes >= kDitLoadMeasureBytes) {
      const double gbps = static_cast<double>(ls.bytes) / ls.seconds / 1e9;
      if (m.learner->observe_dit_load_rate(ls.direct, gbps, m.dit_load_warm) && !m.options.hardware) {
        m.save_profile(m.learner->stored());
      }
      if (!ls.direct) m.learner->observe_dit_load(gbps);
    }
    if (m.dit->block_path() != block_path) m.dit->set_block_path(block_path);
    if (m.dit->attention() != attention) m.dit->set_attention(attention);
    m.dit->set_lora(m.lora_sb.get());
    m.dit->set_request_route(route);
    r.block_path = block_path;
    if (block_path == moss_dit::BlockPath::kFusedGraphs && residency.streamed()) {
      r.block_path = moss_dit::BlockPath::kFused;  // streamed blocks wait on host-driven copies: never captured
      if (r.block_path_note.empty()) r.block_path_note = "CUDA graphs need every DiT block resident: blocks ran fused, eagerly";
    }
    return true;
  };

  if (preload) {
    // What the plan keeps across requests, in generate()'s order: a host-resident text encoder's copy (its device side
    // is all stack), the VAE encoder of a request that starts from an image, the DiT, and the VAE decoder unless the plan
    // parks the DiT for the decode (generate() then loads it after the denoise).
    if (!missing.empty() && plan.te.run && plan.te.residency == planner::TeResidency::kHostResident) {
      if (!keep_going(Phase::kTextEncoder)) return finish();
      const auto t0t = Clock::now();
      m.ensure_te(tes, m.te_options(tes, plan.te.residency, plan.te.buffers, plan.te.pinned));
      try {
        if (!m.load_te_host(tes, plan.te.buffers, control.cancel, r)) r.plan.explanation += "\nText encoder: " + r.te_note + ".";
      } catch (const qwen35_text::Cancelled&) {
        r.cancelled = true;
        tm.te_weight_load_s = since(t0t);
        return finish();
      }
      tm.te_weight_load_s = since(t0t);
      m.sample_memory();
    }
    if (enc_src != nullptr) {
      if (!keep_going(Phase::kDecode)) return finish();
      m.ensure_encoder(*enc_src, encode_need, tm);
    }
    if (need_dit && plan.dit.park_for_text_encoder) {
      r.preload_note = "the DiT was not loaded: the plan parks it for the text encoder, so a generate() that encodes a "
                       "new prompt would evict it first; it loads after the encoder";
    } else if (need_dit) {
      if (!keep_going(Phase::kDitLoad)) return finish();
      if (!prepare_dit()) {
        r.cancelled = true;
        return finish();
      }
    }
    if (need_decoder && !plan.dit.park_for_decode) {
      if (!keep_going(Phase::kDecode)) return finish();
      m.ensure_vae(vs, plan.vae.stack_bytes, false, tm);
    }
    return finish();
  }

  // 1. Text encoder for the sequences the cache did not serve; its device memory is released when it returns.
  if (!keep_going(Phase::kTextEncoder)) return finish();
  if (!missing.empty()) {
    if (plan.dit.park_for_text_encoder && (m.dit || m.vae || m.encoder)) {
      r.memory.dit_parked_for_text_encoder = m.dit != nullptr;
      m.reset_arena(m.arena->capacity(), true);
    }
    std::vector<std::vector<std::int32_t>> todo;
    for (const std::size_t i : missing) todo.push_back(seqs[i]);
    t0 = Clock::now();
    m.ensure_te(tes, m.te_options(tes, plan.te.residency, plan.te.buffers, plan.te.pinned));
    qwen35_text::EncodeOptions eo;
    eo.cancel = control.cancel;
    std::vector<std::vector<std::uint16_t>> out;
    try {
      // A host-resident copy not loaded yet: loaded here, cancellable between layer groups (encode() would load it
      // without the flag). A pinned copy that cannot be allocated falls back to streaming from the files.
      if (!m.load_te_host(tes, plan.te.buffers, control.cancel, r)) r.plan.explanation += "\nText encoder: " + r.te_note + ".";
      out = m.te->encode_to_host(todo, eo);
    } catch (const qwen35_text::Cancelled&) {
      r.cancelled = true;
      tm.text_encode_s = since(t0);
      return finish();
    }
    tm.text_encode_s = since(t0);
    const qwen35_text::EncodeStats& st = m.te->last_stats();
    tm.te_weight_load_s = st.weight_load_s;
    tm.te_weight_wait_s = st.weight_wait_s;
    tm.te_compute_s = st.compute_s;
    if (m.te->options().source == qwen35_text::WeightSource::kFile && m.learner && !m.options.direct_io) {
      m.learner->observe_te_stream(planner::weights_cached_from_stream(st.weight_bytes, st.weight_load_s, st.weight_wait_s, in.hardware));
    }
    m.sample_memory();
    for (std::size_t k = 0; k < missing.size(); ++k) {
      auto e = std::make_shared<Embedding>();
      e->hidden = qwen35_text::kHidden;
      e->data = std::move(out[k]);
      // The negative prompt (row 1) is kept: the usual empty one then never costs a pass again.
      if (m.cache) m.cache->insert(keys[missing[k]], *e, missing[k] == 1);
      embeds[missing[k]] = std::move(e);
    }
  }

  // Host latents of an RGB image at its own size: the VAE encoder's posterior mean, on the arena stack with what the
  // resident models leave (encode_within: exact tiling when the untiled encode does not fit, same bytes).
  const auto encode_image = [&](const Image& im, PassReport& pass) {
    m.ensure_encoder(*enc_src, encode_need_of(im.height, im.width), tm);
    const auto te0 = Clock::now();
    runtime::StackScope scope(*m.arena);
    const Tensor lat = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, im.height / 16, im.width / 16});
    const Tensor rgb = Tensor::view(const_cast<std::uint8_t*>(im.pixels.data()), DType::kU8, Device::kCPU, {im.height, im.width, 3});
    const flux2_vae_encoder::TilePlan used =
        m.encoder->encode_within(rgb, lat, flux2_vae_encoder::EncodeOptions{}, m.arena->available(), stream);
    std::vector<float> z(static_cast<std::size_t>(lat.numel()));
    MI_CUDA_CHECK(cudaMemcpyAsync(z.data(), lat.data, lat.nbytes(), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    pass.encoded = true;
    pass.encode_schedule += (pass.encode_schedule.empty() ? "" : "; ") + (used.tiled() ? used.describe() : std::string("untiled"));
    tm.encode_s += since(te0);
    m.sample_memory();
    return z;
  };
  // An image brought to H x W by the request's resize mode (host, PIL-exact Lanczos).
  const auto sized = [&](const Image& im, int H, int W) {
    const auto ti = Clock::now();
    Image out = resize_to(im, W, H, req.resize);
    tm.image_ops_s += since(ti);
    return out;
  };
  const auto sized_mask = [&](int H, int W) {
    const auto ti = Clock::now();
    std::vector<float> out = latent_mask(resize_to(*req.mask, W, H, req.resize), H / 16, W / 16);
    tm.image_ops_s += since(ti);
    return out;
  };

  // 2. img2img: the init image at the request size, encoded before the DiT load (when the DiT is not resident yet, the
  // encode has the room to itself); then the DiT, resident across calls.
  if (!keep_going(Phase::kDitLoad)) return finish();
  std::vector<float> z0_1, mask_1;
  if (img2img) {
    const Image init = sized(*req.init_image, H1, W1);
    z0_1 = encode_image(init, r.first_pass);
    if (inpaint) mask_1 = sized_mask(H1, W1);
    if (cancel_requested()) {
      r.cancelled = true;
      return finish();
    }
  }
  // Control images at a pass size: per kind of the adapter its encoded latent times its strength (empty: no image).
  std::vector<std::vector<float>> ctl_lat;
  const auto encode_controls = [&](int H, int W) {
    ctl_lat.assign(static_cast<std::size_t>(ctl->groups()), {});
    for (const Request::ControlImage& c : req.control.images) {
      Image im = sized(*c.image, H, W);
      if (im.channels == 1) {
        // A grayscale map replicated to RGB (PIL convert("RGB")), after the resize: PIL's Lanczos filters each channel
        // alone, so this gives the bytes of resizing the replicated image.
        const auto ti = Clock::now();
        std::vector<std::uint8_t> rgb(im.pixels.size() * 3);
        for (std::size_t i = 0; i < im.pixels.size(); ++i) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = im.pixels[i];
        im.pixels = std::move(rgb);
        im.channels = 3;
        tm.image_ops_s += since(ti);
      }
      PassReport enc;
      std::vector<float> z = encode_image(im, enc);
      if (c.strength != 1.0) {
        const float s = static_cast<float>(c.strength);
        for (float& v : z) v *= s;
      }
      ctl_lat[static_cast<std::size_t>(ctl->kind_index(c.kind))] = std::move(z);
      ++r.control.encodes;
      r.control.encode_schedule += (r.control.encode_schedule.empty() ? "" : "; ") + enc.encode_schedule;
    }
  };
  if (ctl_encode && sched1.num_steps() > 0) {
    encode_controls(H1, W1);
    if (cancel_requested()) {
      r.cancelled = true;
      return finish();
    }
  }
  if (need_dit && !prepare_dit()) return finish();

  // 3. Sampling passes: the text refiner once per row, then the sampling loop at H x W over `sched`, from the seeded
  // noise (z0 null) or from z0 at the schedule's first sigma; with a mask, the inpainting blend against blend_z0 after
  // every step. Leaves the pass's latents in r.latents (the partial ones when cancelled); returns false when cancelled.
  int steps_done = 0;  // finished steps of the earlier passes (progress counts on across passes)
  const auto run_pass = [&](int H, int W, const Schedule& sched, const std::vector<float>* z0, const std::vector<float>* blend_z0,
                            const std::vector<float>* lmask, PassReport& pass) {
    const std::int64_t h = H / 16, w = W / 16;
    pass.ran = true;
    r.height = H;
    r.width = W;
    if (sched.num_steps() == 0) {
      // Denoise strength 0: sigma = 0 at the start, x = z0 (the VAE round trip of the image).
      r.latents = *z0;
      return true;
    }
    const std::vector<float> noise = initial_noise(req.noise_layout, r.seed, moss_dit::kLatentChannels, h, w);
    runtime::StackScope phase(*m.arena);
    const Tensor ref_c = m.arena->alloc_tensor(DType::kBF16, {sc, moss_dit::kDim});
    const Tensor ref_u = g.do_cfg() ? m.arena->alloc_tensor(DType::kBF16, {su, moss_dit::kDim}) : Tensor{};
    {
      runtime::StackScope refine(*m.arena);
      const auto tr = Clock::now();
      for (std::size_t i = 0; i < embeds.size(); ++i) {
        const std::int64_t s = static_cast<std::int64_t>(seqs[i].size());
        const Tensor te = m.arena->alloc_tensor(DType::kBF16, {s, moss_dit::kTextDim});
        MI_CUDA_CHECK(cudaMemcpyAsync(te.data, embeds[i]->data.data(), te.nbytes(), cudaMemcpyHostToDevice, stream));
        m.dit->encode_text(te, i == 0 ? ref_c : ref_u, stream);
      }
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      tm.refine_s += since(tr);
    }

    // P:400-402: fp32 CPU noise [1, 128, h, w].
    const Tensor latents = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w});
    const std::size_t image_bytes = image_bytes_of(H, W);
    Tensor prev, x0, preview_rgb;
    void* preview_ws = nullptr;
    const std::size_t preview_ws_bytes = latent_preview::workspace_bytes(h, w);
    std::vector<std::uint8_t> preview_host;
    if (control.on_preview) {
      prev = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w});
      x0 = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w});
      preview_ws = m.arena->alloc_stack(preview_ws_bytes);
      preview_rgb = m.arena->alloc_tensor(DType::kU8, {H, W, 3});
      preview_host.resize(image_bytes);
    }
    MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, noise.data(), latents.nbytes(), cudaMemcpyHostToDevice, stream));
    if (z0 == nullptr) {
      // x = sigma_0 * noise (identity for the official schedule's sigma_0 = 1).
      flow_noise_scaling(latents, Tensor{}, sched.sigmas.front(), stream);
    } else {
      // img2img (scheduler.scale_noise): x = sigma_t * noise + (1 - sigma_t) * z0.
      runtime::StackScope upload(*m.arena);
      const Tensor z = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w});
      MI_CUDA_CHECK(cudaMemcpyAsync(z.data, z0->data(), z.nbytes(), cudaMemcpyHostToDevice, stream));
      flow_noise_scaling(latents, z, sched.sigmas.front(), stream);
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (control.on_preview) MI_CUDA_CHECK(cudaMemcpyAsync(prev.data, latents.data, latents.nbytes(), cudaMemcpyDeviceToDevice, stream));
    // Control adapter: its input latents for this pass (zeros for kinds without an image or outside their window),
    // rewritten between steps when a window opens or closes. The DiT points at them until the pass ends (CancelScope
    // clears the pointer on every exit).
    Tensor ctl_extra;
    std::optional<control::PassInputs> ctl_in;
    if (ctl != nullptr) {
      ctl_extra = m.arena->alloc_tensor(DType::kF32, {ctl->groups() * control::kGroupChannels, h, w});
      ctl_in.emplace(ctl->groups(), h, w);
      for (const Request::ControlImage& c : req.control.images) {
        const int gi = ctl->kind_index(c.kind);
        ctl_in->set(gi, std::move(ctl_lat[static_cast<std::size_t>(gi)]), c.start, c.end);  // re-encoded for a later pass
      }
      ctl_in->apply(ctl_extra, 0, sched.num_steps(), stream);
      m.control_ext.extra = ctl_extra;
      m.dit->set_input_extension(&m.control_ext);
    } else {
      m.dit->set_input_extension(nullptr);
    }
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Streaming diagnostics per step: the provider's running totals (read after the step's synchronization), added to
    // the earlier passes' totals.
    DitStreamStats& ds = r.dit_stream;
    const DitStreamStats before = ds;
    ds.streamed = ds.streamed || residency.streamed();
    if (residency.streamed()) {
      ds.resident_blocks = residency.resident_blocks;
      ds.buffers = residency.buffers;
      ds.source = moss_dit::stream_source_name(residency.source);
      m.dit->reset_stream_stats();
    }
    moss_dit::StreamStats seen;
    std::vector<float> blend_host(lmask != nullptr ? noise.size() : 0);
    int finished = 0;
    // P6b step cache (FBCache) for this pass: its buffers are taken right before sampling (planned in denoise_need_of);
    // the hook is removed from the DiT however the pass ends.
    std::optional<accel::FirstBlockCache> step_cache;
    int cached_seen = 0;
    struct StepCacheReset {
      moss_dit::MossDit* dit = nullptr;
      ~StepCacheReset() {
        if (dit != nullptr) dit->set_step_cache(nullptr);
      }
    } step_cache_reset;
    const StepCallback on_step = [&](const StepInfo& info) {
      finished = info.step;
      if (ctl_in && info.step < sched.num_steps() && ctl_in->apply(ctl_extra, info.step, sched.num_steps(), stream) > 0 &&
          step_cache) {
        step_cache->invalidate();  // a control window opened or closed: the DiT's input changed, the next call runs in full
      }
      const bool step_cached = step_cache && step_cache->reused() > cached_seen;
      if (step_cached) {
        r.accel.cached_steps.push_back(steps_done + info.step);
        cached_seen = step_cache->reused();
      }
      if (lmask != nullptr) {
        // Inpainting: the kept region back on the init image's trajectory at the next step's sigma (0 after the last).
        const auto tb = Clock::now();
        MI_CUDA_CHECK(cudaMemcpyAsync(blend_host.data(), latents.data, latents.nbytes(), cudaMemcpyDeviceToHost, stream));
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        inpaint_blend(blend_host, *blend_z0, noise, *lmask, sched.sigmas[static_cast<std::size_t>(info.step)]);
        MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, blend_host.data(), latents.nbytes(), cudaMemcpyHostToDevice, stream));
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        tm.image_ops_s += since(tb);
      }
      if (residency.streamed()) {
        const moss_dit::StreamStats& total = m.dit->total_stream_stats();
        DitStreamStats::Step st;
        const double copy_s = total.copy_s - seen.copy_s;
        st.gbps = copy_s > 0 ? static_cast<double>(total.bytes - seen.bytes) / copy_s / 1e9 : 0.0;
        st.stall_s = total.stall_s - seen.stall_s;
        st.hidden = copy_s > 0 ? std::max(0.0, 1.0 - st.stall_s / copy_s) : 0.0;
        st.stalls = total.stalls - seen.stalls;
        st.cached = step_cached;
        ds.steps.push_back(st);
        seen = total;
        ds.bytes = before.bytes + total.bytes;
        ds.copy_s = before.copy_s + total.copy_s;
        ds.stall_s = before.stall_s + total.stall_s;
        ds.stalls = before.stalls + total.stalls;
      }
      tm.step_s.push_back(info.step_seconds);
      const StepInfo overall{steps_done + info.step, total_steps, info.step_seconds};
      if (control.on_preview) {
        // x0 estimate from this step's effective velocity (the last step's latents are their own x0: sigma = 0).
        const int every = std::max(1, control.preview_every);
        if (overall.step % every == 0 || overall.step == overall.num_steps) {
          const double s_now = sched.sigmas[static_cast<std::size_t>(info.step)];
          const double s_prev = sched.sigmas[static_cast<std::size_t>(info.step) - 1];
          const float c = s_prev != s_now ? static_cast<float>(s_now / (s_prev - s_now)) : 0.0f;
          moss_dit::extrapolate_latents(latents, prev, c, x0, stream);
          latent_preview::preview(x0, preview_rgb, preview_ws, preview_ws_bytes, stream);
          MI_CUDA_CHECK(cudaMemcpyAsync(preview_host.data(), preview_rgb.data, image_bytes, cudaMemcpyDeviceToHost, stream));
          MI_CUDA_CHECK(cudaStreamSynchronize(stream));
          control.on_preview(PreviewImage{overall.step, overall.num_steps, W, H, preview_host.data()});
        }
        MI_CUDA_CHECK(cudaMemcpyAsync(prev.data, latents.data, latents.nbytes(), cudaMemcpyDeviceToDevice, stream));
      }
      if (control.on_step && !control.on_step(overall)) r.cancelled = true;
      if (cancel_requested()) r.cancelled = true;
      return !r.cancelled;
    };
    if (req.accel.step_cache_on()) {
      step_cache.emplace(*m.arena, h, w, static_cast<int>(row_lengths.size()), r.accel.step_cache_threshold);
      m.dit->set_step_cache(&*step_cache);
      step_cache_reset.dit = m.dit.get();
    }
    const auto ts = Clock::now();
    int run = 0;
    try {
      sampler_options.seed = r.seed;
      run = denoise(*m.dit, *m.arena, ref_c, ref_u, sched, g, req.sampler, sampler_options, latents, stream, on_step);
    } catch (const moss_dit::Cancelled&) {
      // Inside a step's forward, before its update: the latents are those after the last finished step.
      r.cancelled = true;
      run = finished;
    }
    tm.denoise_s += since(ts);
    if (ctl_in) m.dit->set_input_extension(nullptr);
    if (step_cache) {
      r.accel.dit_calls += step_cache->calls();
      r.accel.cached_calls += step_cache->reused();
      r.accel.step_cache_diffs.insert(r.accel.step_cache_diffs.end(), step_cache->diffs().begin(), step_cache->diffs().end());
    }
    for (int i = 0; i < run; ++i) r.accel.cfg_truncated_steps += g.do_cfg() && !g.cfg_active(i, sched.num_steps());
    pass.steps_run = run;
    steps_done += run;
    r.steps_run = steps_done;
    m.sample_memory();
    r.latents.resize(static_cast<std::size_t>(latents.numel()));
    MI_CUDA_CHECK(cudaMemcpyAsync(r.latents.data(), latents.data, latents.nbytes(), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    return !r.cancelled;
  };

  // 4. VAE decode (P:428-438) of host latents to RGB8: untiled when it fits the arena next to the resident models, else
  // tiled, with the same bytes either way (decode_mode). Returns false when cancelled (`out` is then cleared).
  // `first`: the hires fix's first pass, decoded next to the DiT without parking it (the plan counted its smallest
  // tiling in the denoise phase); its schedule goes to first_pass.decode_schedule. Otherwise the final decode, as planned.
  const auto decode_latents = [&](const std::vector<float>& lat, int H, int W, std::vector<std::uint8_t>& out, bool first) {
    if (first) {
      m.ensure_vae(vs, first_decode_need, false, tm);
    } else {
      m.ensure_vae(vs, plan.vae.stack_bytes, plan.dit.park_for_decode, tm);
    }
    runtime::StackScope phase(*m.arena);
    const Tensor latents = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, H / 16, W / 16});
    MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, lat.data(), latents.nbytes(), cudaMemcpyHostToDevice, stream));
    out.resize(image_bytes_of(H, W));
    const Tensor rgb = Tensor::view(out.data(), DType::kU8, Device::kCPU, {H, W, 3});
    const auto td = Clock::now();
    m.vae->set_cancel(control.cancel);
    try {
      const std::optional<flux2_vae::TilePlan> used =
          m.vae->decode_within(latents, rgb, decode_mode, m.arena->available(), stream);
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      if (first) {
        r.first_pass.decode_schedule = used ? used->describe() : "untiled";
      } else {
        r.decode_tiled = used.has_value();
        r.decode_schedule = used ? used->describe() : "untiled";
      }
    } catch (const flux2_vae::Cancelled&) {
      r.cancelled = true;
      out.clear();
    }
    tm.decode_s += since(td);
    m.sample_memory();
    return !r.cancelled;
  };

  if (!keep_going(Phase::kDenoise)) return finish();
  if (!run_pass(H1, W1, sched1, img2img ? &z0_1 : nullptr, inpaint ? &z0_1 : nullptr, inpaint ? &mask_1 : nullptr, r.first_pass)) {
    return finish();
  }
  // Compute and streaming rates from a single pass at one size (a hires fix's passes mix sizes).
  // Not from accelerated steps (P6b: cached / truncated steps are no measure of the DiT's rate).
  if (!hires && sched1.num_steps() > 0 && !req.accel.active(g.do_cfg())) m.learn_from_denoise(r);

  // 5. Hires fix: decode the first pass, upscale it in pixel space, encode it at the target size, and refine it there.
  if (hires) {
    if (!keep_going(Phase::kDecode)) return finish();
    Image first;
    first.width = W1;
    first.height = H1;
    first.channels = 3;
    if (!decode_latents(r.latents, H1, W1, first.pixels, true)) return finish();
    if (!keep_going(Phase::kDenoise)) return finish();
    std::vector<float> z0_2, blend_2, mask_2;
    {
      const auto ti = Clock::now();
      const Image up = resize_to(first, W2, H2, req.resize);
      first.pixels = {};
      tm.image_ops_s += since(ti);
      z0_2 = encode_image(up, r.hires);
    }
    if (inpaint) {
      blend_2 = encode_image(sized(*req.init_image, H2, W2), r.hires);
      mask_2 = sized_mask(H2, W2);
    }
    if (ctl_encode && sched2.num_steps() > 0) encode_controls(H2, W2);
    if (cancel_requested()) {
      r.cancelled = true;
      return finish();
    }
    if (sched2.num_steps() > 0 && !prepare_dit()) return finish();
    if (!run_pass(H2, W2, sched2, &z0_2, inpaint ? &blend_2 : nullptr, inpaint ? &mask_2 : nullptr, r.hires)) return finish();
  }

  // A step cache that never reused over a whole run: said with what it measured, so a threshold below this DiT's floor
  // of block-0 change is not mistaken for a working cache.
  if (!r.cancelled) {
    if (std::string w = accel::never_engaged_warning(r.accel); !w.empty()) r.accel_warnings.push_back(std::move(w));
  }

  // 6. The final decode. Not after a cancel, even one seen at the last step.
  if (req.decode && !r.cancelled && keep_going(Phase::kDecode)) decode_latents(r.latents, r.height, r.width, r.rgb, false);
  return finish();
}

}  // namespace mi::pipeline
