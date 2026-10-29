// Pipeline::Impl, the state a Pipeline keeps across calls (models, arena, sources, caches), and the helpers it uses.
// Internal: included by pipeline.cpp and the component session (pipeline/session.cpp), which drive the same state.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "models/control/control.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/flux2_vae/vae_file.h"
#include "models/flux2_vae_encoder/flux2_vae_encoder.h"
#include "models/lora/lora.h"
#include "models/lora/side_branch.h"
#include "models/moss_dit/moss_dit.h"
#include "models/moss_dit/nvfp4_support.h"
#include "models/qwen35_text/text_encoder.h"
#include "pipeline/embedding_cache.h"
#include "pipeline/hardware_profile.h"
#include "pipeline/pipeline.h"
#include "pipeline/samplers/sampler_driver.h"
#include "planner/host_memory.h"
#include "planner/planner.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "tokenizer/qwen_bpe.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"
#include "weights/upload.h"

namespace mi::pipeline {
// One definition for every translation unit that includes this header; Impl and pipeline.cpp use the short names.
namespace detail {

using Clock = std::chrono::steady_clock;

inline double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// Exact GroupNorm statistics: the untiled decode and every tiled one the plan may pick from the free VRAM give the same
// bytes, so the image does not depend on the memory budget.
inline constexpr flux2_vae::DecodeMode kDecodeMode = flux2_vae::DecodeMode::reference();
static_assert(!kDecodeMode.torch_group_norm, "torch's GroupNorm cannot be tiled");

// Stack bytes of one latent [128, H / 16, W / 16] (F32), of an RGB8 image, and of bf16 rows [s, c].
inline std::size_t latent_bytes_of(int H, int W) {
  return runtime::align_up(static_cast<std::size_t>(moss_dit::kLatentChannels) * static_cast<std::size_t>(H / 16) *
                           static_cast<std::size_t>(W / 16) * sizeof(float));
}
inline std::size_t image_bytes_of(int H, int W) { return static_cast<std::size_t>(H) * static_cast<std::size_t>(W) * 3; }
inline std::size_t bf16_rows(std::int64_t s, std::int64_t c) { return runtime::align_up(static_cast<std::size_t>(s * c) * 2); }
// A VAE encode at H x W: its latents and the smallest exact tiling (at run time the encode takes the fastest tiling the
// arena leaves).
inline std::size_t encode_need_of(int H, int W) {
  return latent_bytes_of(H, W) + flux2_vae_encoder::Flux2VaeEncoder::plan_min_memory(H, W).workspace_bytes;
}

// The denoise phase's stack of one pass at height x width (denoise_stack_bytes): the refined texts, then the larger of
// the refiner (its inputs and workspace) and the pass's buffers: the latents, a control adapter's input latents, a
// latent preview's buffers, then the loop's own buffers and the forward for its largest row set. A pass starting from
// an image uploads z0 next to the latents before sampling instead.
struct DenoiseStack {
  int height = 0, width = 0;
  int latents = 1;                  // the latent batch
  std::vector<std::int64_t> texts;  // the refined texts' tokens, one buffer each
  std::vector<std::int64_t> rows;   // the forward's largest row set (each row's text tokens)
  std::size_t loop_bytes = 0;       // the loop's own buffers below the forward (velocities, sampler state, step cache)
  int control_groups = 0;           // a control adapter's input groups (0: none)
  bool preview = false;
  bool from_image = false;
};
std::size_t denoise_stack_bytes(const DenoiseStack& d, moss_dit::BlockPath path, const moss_dit::ComputeConfig& config,
                                const moss_dit::FusedRouteSnapshot& route);

struct Bundle {
  std::filesystem::path dir;
  weights::ModelIndex index;  // without the components left out for the pipeline's component files
  tok::Tokenizer tokenizer;
  std::uint64_t te_hash = 0;  // embedding-cache identity of the text encoder's weights (0: not indexed)
};

// The bundle's index without the components in `skip` (weights::BuildOptions::skip; a .mossidx is read the same way)
// and its tokenizer. The folder has passed check_supported_bundle.
inline std::unique_ptr<Bundle> open_bundle(const std::filesystem::path& dir, weights::ComponentSet skip) {
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

// The DiT weight files of `index` (its transformer groups' files; a single file's index holds only the DiT).
inline std::vector<std::filesystem::path> dit_files(const weights::ModelIndex& index) {
  std::set<std::uint32_t> ids;
  for (const weights::WeightGroup* g : index.groups_of(weights::Component::kTransformer)) {
    for (const weights::IndexedTensor& t : g->tensors) ids.insert(t.file);
  }
  std::vector<std::filesystem::path> out;
  for (const std::uint32_t id : ids) out.push_back(index.file_path(id));
  return out;
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
inline void check_option_paths(const PipelineOptions& o) {
  const std::pair<const std::filesystem::path*, const char*> paths[] = {
      {&o.cache_dir, "cache_dir"},
      {&o.dit_file, "dit_file"},
      {&o.text_encoder_file, "text_encoder_file"},
      {&o.vae_file, "vae_file"},
      {&o.time_embedding_file, "time_embedding_file"}};
  for (const auto& [path, what] : paths) util::check_user_path(*path, what);
}

inline std::unique_ptr<DitSource> open_dit_file(const std::filesystem::path& file) {
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
inline planner::HardwareProfile probe_hardware(const weights::ModelIndex& index, cudaStream_t stream) {
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
inline std::int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// The fused-block self-check result of each hardware key this process has checked (first use per process).
inline std::optional<moss_dit::FusedCheck> process_fused_check(const std::string& key, const moss_dit::FusedCheck* set = nullptr) {
  static std::mutex mu;
  static std::map<std::string, moss_dit::FusedCheck> checks;
  std::lock_guard<std::mutex> lock(mu);
  if (set != nullptr) checks[key] = *set;
  const auto it = checks.find(key);
  if (it == checks.end()) return std::nullopt;
  return it->second;
}

}  // namespace detail

using namespace detail;

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

  // The models of a call, set up for its request (Pipeline::run, the component session): the bundle and the DiT, text
  // encoder and VAE sources (reported in `r`); the control adapter's file and device embedder, its LoRA part joining
  // the runtime LoRA stack after the request's LoRAs; the quality tier and report; the runtime LoRA stack, kept,
  // updated or rebuilt. Throws before anything heavy for a refused tier, kind or LoRA file.
  struct CallModels {
    Bundle* b = nullptr;
    DitSource* src = nullptr;
    const TeSource* tes = nullptr;
    const VaeSource* vs = nullptr;
    const control::ControlFile* ctl = nullptr;  // the request's control adapter (nullptr: none)
    std::string merge_key;                      // the LoRA files merged into the DiT (empty: none)
  };
  CallModels setup_models(const Request& req, Result& r);

  // The DiT side of a call (setup_dit_call): the machine profile (into r.hardware), the block path, the call's fused
  // route (one snapshot sizes its workspace and runs every step), the attention and the compute config.
  struct CallDit {
    moss_dit::BlockPath block_path = moss_dit::BlockPath::kFused;
    moss_dit::FusedRouteSnapshot route;
    moss_dit::AttentionMode attention = moss_dit::AttentionMode::kExact;
    moss_dit::ComputeConfig config;
  };
  CallDit setup_dit_call(const DitSource& src, Result& r);

  // A call's shape as its plan sees it (plan_call); the rest comes from the models, the held state and the machine.
  struct PlanCall {
    int height = 0, width = 0;               // the largest pass
    std::vector<std::int64_t> text_lengths;  // the refined texts' tokens
    std::vector<std::int64_t> te_lengths;    // the sequences the text encoder runs (empty: none)
    int steps = 0;                           // DiT evaluations, at the plan size's cost
    int rows_per_step = 1;
    bool need_dit = true;
    const EncoderSource* encoder = nullptr;   // the VAE encoder the call uses (nullptr: none)
    int decode_height = 0, decode_width = 0;  // the largest decode (0: none)
    std::size_t denoise = 0;       // the denoise phase's stack: its passes, encodes and a hires fix's first decode
    std::size_t host_buffers = 0;  // the call's own host buffers
  };
  // The call's plan into r.plan (throws planner::BudgetError before any heavy work): planner::plan within the engine's
  // mode and budgets, without the VAE models an earlier call left resident when they alone make it infeasible
  // (drop_resident), then planner::prefer_held for a loaded DiT of the same source and merges (kept).
  struct Planned {
    bool kept = false;
    bool drop_resident = false;
  };
  Planned plan_call(const CallModels& models, const PlanCall& c, moss_dit::BlockPath block_path, Result& r);
  // The plan's arena: a larger existing one is kept unless the call needs less than arena_shrink_below of it and the
  // plan does not keep the loaded DiT.
  void setup_arena(const planner::ExecutionPlan& plan, bool kept, bool need_dit, bool drop_resident, Result& r);
  // Loads the DiT when it is not resident (first use, or parked / dropped since) and sets the call's options (cancel
  // flag, linear-input hook, block path, attention, LoRA side branches, route). False when the load was cancelled
  // (r.cancelled set).
  bool prepare_dit(DitSource& src, const moss_dit::Residency& residency, std::size_t denoise_need, const CallDit& d,
                   const std::string& merge_key, const std::atomic<bool>* cancel, const moss_dit::LinearInputHook& linear_inputs,
                   Result& r);

  // Host latents [128, H / 16, W / 16] of an RGB image at its own size (sides multiples of 16): the VAE encoder `e`'s
  // posterior mean (loaded as needed), on the arena stack with what the resident models leave (encode_within: exact
  // tiling when the untiled encode does not fit, same bytes). The tiling used is appended to pass.encode_schedule.
  std::vector<float> encode_image(const EncoderSource& e, const Image& im, PassReport& pass, Timings& tm);
  // VAE decode (P:428-438) of host latents [128, H / 16, W / 16] to RGB8 `out` [H, W, 3] by the loaded decoder
  // (ensure_vae first): untiled when it fits the arena next to the resident models, else tiled, with the same bytes
  // either way (kDecodeMode). Returns the tiling used (nullopt: untiled). Throws flux2_vae::Cancelled (`cancel`).
  std::optional<flux2_vae::TilePlan> decode_image(const std::vector<float>& lat, int H, int W, std::vector<std::uint8_t>& out,
                                                  const std::atomic<bool>* cancel, Timings& tm);
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

}  // namespace mi::pipeline
