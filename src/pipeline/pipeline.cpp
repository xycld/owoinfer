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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include "core/cuda_check.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/moss_dit/moss_dit.h"
#include "models/qwen35_text/text_encoder.h"
#include "pipeline/embedding_cache.h"
#include "pipeline/hardware_profile.h"
#include "pipeline/rng_torch_compat.h"
#include "planner/host_memory.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "tokenizer/qwen_bpe.h"
#include "util/path.h"
#include "weights/model_index.h"
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
  weights::ModelIndex index;
  tok::Tokenizer tokenizer;
  std::uint64_t te_hash = 0;  // embedding-cache identity of the text encoder's weights
};

std::unique_ptr<Bundle> open_bundle(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) throw std::invalid_argument("model bundle not found: " + util::path_utf8(dir));
  const std::filesystem::path idx = dir / "moss.mossidx";
  weights::ModelIndex index = std::filesystem::is_regular_file(idx) ? weights::load_index(idx, dir) : weights::build_index(dir);
  auto b = std::unique_ptr<Bundle>(new Bundle{dir, std::move(index), tok::Tokenizer::load(dir / "tokenizer" / "tokenizer.json"), 0});
  b->te_hash = text_encoder_hash(b->index);
  return b;
}

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
  return r;
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
  const Bundle* te_bundle = nullptr;
  std::unique_ptr<moss_dit::MossDit> dit;
  const Bundle* dit_bundle = nullptr;
  std::unique_ptr<flux2_vae::Flux2VaeDecoder> vae;
  const Bundle* vae_bundle = nullptr;
  // Bundle whose resident DiT was last dropped outside the plan's parking (its next load is a reload).
  const Bundle* dit_lost = nullptr;
  // Transfer rates for the cost model and whether the weight files are page-cached: probed (or read from the cache
  // directory) at the first generate(), refined from every pass (pipeline/hardware_profile.h). Unset with
  // options.hardware.
  std::optional<ProfileLearner> learner;
  std::string hw_key;  // hardware_key() of the learner's profile
  // This pipeline's fused-block self-check (from the profile, the process, or run at the first fused generate()).
  std::optional<moss_dit::FusedCheck> fused_check;
  const std::uint64_t te_numerics = qwen35_text::numerics_id();  // embedding-cache key part

  // Per generate() call.
  MemoryStats* mem = nullptr;

  explicit Impl(PipelineOptions o) : options(std::move(o)) {
    free_at_start = runtime::DeviceArena::free_device_bytes();
    if (options.embedding_cache) {
      EmbeddingCacheOptions co;
      co.disk_dir = options.cache_dir;
      cache = std::make_unique<EmbeddingCache>(co);
    }
  }

  ~Impl() {
    te.reset();
    vae.reset();
    dit.reset();
    arena.reset();
  }

  Bundle& bundle(const std::filesystem::path& dir, double& load_s) {
    const std::string key = util::path_utf8(std::filesystem::weakly_canonical(dir));
    auto it = bundles.find(key);
    if (it == bundles.end()) {
      const auto t0 = Clock::now();
      it = bundles.emplace(key, open_bundle(dir)).first;
      load_s = since(t0);
    }
    return *it->second;
  }

  void sample_memory() {
    const std::size_t f = runtime::DeviceArena::free_device_bytes();
    if (free_at_start > f) mem->device_peak = std::max(mem->device_peak, free_at_start - f);
    if (arena) mem->arena_high_water = std::max(mem->arena_high_water, arena->high_water());
    mem->host_private_peak = std::max(mem->host_private_peak, planner::process_host_memory().private_bytes);
  }

  qwen35_text::EncoderOptions te_options(planner::TeResidency residency, int buffers) const {
    qwen35_text::EncoderOptions o;
    o.direct_io = options.direct_io;
    o.weight_buffers = std::max(1, buffers);
    // Host-resident weights stay pageable: the whole model is never pinned (plan step 24).
    o.source = residency == planner::TeResidency::kHostResident ? qwen35_text::WeightSource::kHostPageable
                                                                 : qwen35_text::WeightSource::kFile;
    return o;
  }

  // Drops every resident model and recreates the arena with `capacity` bytes (all stack phases must have ended).
  // `park`: the plan evicts the DiT here (anything else that drops it makes its next load a reported reload).
  void reset_arena(std::size_t capacity, bool park = false) {
    MI_CUDA_CHECK(cudaStreamSynchronize(streams.compute()));
    if (dit && !park) dit_lost = dit_bundle;
    te.reset();
    te_bundle = nullptr;
    vae.reset();
    vae_bundle = nullptr;
    dit.reset();
    dit_bundle = nullptr;
    if (arena) {
      sample_memory();
      arena.reset();
    }
    arena = std::make_unique<runtime::DeviceArena>(capacity);
    ++mem->arena_creations;
    mem->arena_capacity = std::max(mem->arena_capacity, arena->capacity());
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

  planner::HardwareProfile hardware(const Bundle& b) {
    if (options.hardware) return *options.hardware;
    if (learner) return learner->current();
    hw_key = hardware_key();
    const std::int64_t now = unix_now();
    std::optional<StoredProfile> stored;
    if (!profile_path().empty()) stored = read_profile(profile_path(), hw_key, now);
    if (!stored) {
      StoredProfile p;
      try {
        p.probe = probe_hardware(b.index, streams.compute());
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
    if (steps.size() >= 3 && r.plan.dit.compute_s > 0) {
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

  void ensure_te(const Bundle& b, const qwen35_text::EncoderOptions& o) {
    if (te && te_bundle == &b) {
      const qwen35_text::EncoderOptions& cur = te->options();
      if (cur.weight_buffers == o.weight_buffers && cur.source == o.source && cur.direct_io == o.direct_io) return;
    }
    te.reset();
    te = std::make_unique<qwen35_text::TextEncoder>(b.index, *arena, streams.compute(), o);
    te_bundle = &b;
  }

  // The plan counts the DiT and every resident model, so a resident DiT is never expected to be dropped here; when
  // it is (the arena found does not match the plan), the reload is reported as MemoryStats::dit_reloaded.
  // `stack_need` excludes the streamed blocks' device buffers (added here from `residency`). A loaded DiT with another
  // residency is replaced: a planned change, not a reload.
  void ensure_dit(const Bundle& b, const moss_dit::Residency& residency, std::size_t stack_need, const std::atomic<bool>* cancel,
                  Timings& tm) {
    stack_need += moss_dit::MossDit::stream_workspace_bytes(b.index, residency);
    const bool same = dit && dit_bundle == &b && dit->residency() == residency;
    if (same && arena->available() >= stack_need) return;
    const std::size_t persistent = moss_dit::MossDit::persistent_bytes(b.index, residency);
    if (dit || arena->available() < persistent + stack_need) reset_arena(arena->capacity(), dit && !same);
    if (arena->available() < persistent + stack_need) {
      throw runtime::BudgetError(persistent + stack_need, arena->available());
    }
    const auto t0 = Clock::now();
    try {
      dit = std::make_unique<moss_dit::MossDit>(b.index, *arena, streams, options.direct_io, cancel, residency);
    } catch (...) {
      // A partial load (cancelled or failed) leaves its weights in the persistent region: start over clean.
      reset_arena(arena->capacity());
      throw;
    }
    dit_bundle = &b;
    mem->dit_reloaded = mem->dit_reloaded || dit_lost == &b;
    dit_lost = nullptr;
    tm.dit_load_s = since(t0);
    sample_memory();
  }

  void ensure_vae(const Bundle& b, std::size_t stack_need, bool park_dit, Timings& tm) {
    if (vae && vae_bundle == &b && arena->available() >= stack_need && !(park_dit && dit)) return;
    const std::size_t persistent = flux2_vae::Flux2VaeDecoder::persistent_bytes(b.index);
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
    vae = std::make_unique<flux2_vae::Flux2VaeDecoder>(b.index, *arena, streams, false, options.direct_io);
    vae_bundle = &b;
    tm.vae_load_s = since(t0);
    sample_memory();
  }

  // Private host memory the pipeline holds itself across calls (cache contents, text-encoder rings and host copy),
  // which the planner adds on its own and must not count twice. The planner adds the text encoder's share only when
  // it runs: on a full embedding-cache hit the encoder's held memory stays in the baseline.
  std::size_t held_host_bytes(bool te_runs) const {
    std::size_t held = cache ? cache->stats().bytes : 0;
    if (dit) held += dit->host_bytes();  // a streamed DiT's rings or pinned copy: the planner counts its own
    if (te_runs && te && te_bundle != nullptr) {
      qwen35_text::EncoderOptions file = te->options();
      file.source = qwen35_text::WeightSource::kFile;
      held += te->host_resident_bytes() + qwen35_text::TextEncoder::host_bytes(te_bundle->index, file);
    }
    return held;
  }
};

Pipeline::Pipeline(PipelineOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
Pipeline::~Pipeline() = default;

Result Pipeline::generate(const Request& req, const GenerateControl& control) {
  Impl& m = *impl_;
  validate(req);
  Result r;
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
    m.sample_memory();
    m.mem = nullptr;
    if (m.learner) m.learner->end_generation();
    return std::move(r);
  };
  // The models keep the call's cancel flag only for the call (the flag may not outlive it).
  struct CancelScope {
    Impl& m;
    ~CancelScope() {
      if (m.dit) m.dit->set_cancel(nullptr);
      if (m.vae) m.vae->set_cancel(nullptr);
    }
  } cancel_scope{m};

  Bundle& b = m.bundle(req.model, tm.index_load_s);
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
  const Schedule schedule = moss_schedule(req.steps, req.shift);
  r.height = req.height;
  r.width = req.width;
  r.steps_requested = req.steps;
  r.steps_scheduled = schedule.num_steps();

  // Tokenize (P:321-327). The negative prompt is encoded only with CFG (P:388-394).
  auto t0 = Clock::now();
  std::vector<std::vector<std::int32_t>> seqs;
  seqs.push_back(tok::encode_moss_prompt(b.tokenizer, req.prompt, static_cast<std::size_t>(req.max_sequence_length)));
  if (g.do_cfg()) {
    seqs.push_back(tok::encode_moss_prompt(b.tokenizer, req.negative_prompt, static_cast<std::size_t>(req.max_sequence_length)));
  }
  tm.tokenize_s = since(t0);
  const std::int64_t sc = static_cast<std::int64_t>(seqs[0].size());
  const std::int64_t su = g.do_cfg() ? static_cast<std::int64_t>(seqs[1].size()) : 0;
  r.cond_tokens = static_cast<int>(sc);
  r.uncond_tokens = static_cast<int>(su);

  // Embedding cache: sequences it serves skip the text encoder.
  t0 = Clock::now();
  std::vector<EmbeddingKey> keys(seqs.size());
  std::vector<std::shared_ptr<const Embedding>> embeds(seqs.size());
  std::vector<std::size_t> missing;
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    keys[i] = EmbeddingKey{b.te_hash, m.te_numerics, req.max_sequence_length, seqs[i]};
    if (m.cache) embeds[i] = m.cache->find(keys[i]);
    if (embeds[i]) {
      ++r.te_cache_hits;
    } else {
      missing.push_back(i);
    }
  }
  r.te_skipped = missing.empty();
  std::vector<std::int64_t> te_lengths;
  for (const std::size_t i : missing) te_lengths.push_back(static_cast<std::int64_t>(seqs[i].size()));

  // Machine profile and block path (the first call probes and self-checks; both are cached).
  r.hardware = m.hardware(b);
  const moss_dit::BlockPath block_path = m.block_path(r.block_path_note);

  // Exact stack need of the phases this request runs, and the plan.
  const std::int64_t h = req.height / 16, w = req.width / 16;
  const std::size_t latent_bytes = align_up(static_cast<std::size_t>(moss_dit::kLatentChannels * h * w) * sizeof(float));
  const auto bf16_rows = [](std::int64_t s, std::int64_t c) { return align_up(static_cast<std::size_t>(s * c) * 2); };
  const std::size_t refined_bytes = bf16_rows(sc, moss_dit::kDim) + (g.do_cfg() ? bf16_rows(su, moss_dit::kDim) : 0);
  const std::size_t refine_need = bf16_rows(sc, moss_dit::kTextDim) + (g.do_cfg() ? bf16_rows(su, moss_dit::kTextDim) : 0) +
                                  moss_dit::MossDit::encode_text_workspace_bytes(std::max(sc, su));
  const std::size_t denoise_need =
      refined_bytes + std::max(refine_need, latent_bytes + denoise_workspace_bytes(h, w, sc, su, g, req.sampler, block_path));
  // Exact GroupNorm statistics: the untiled decode and every tiled one the plan may pick from the free VRAM give the
  // same bytes, so the image does not depend on the memory budget.
  constexpr flux2_vae::DecodeMode decode_mode = flux2_vae::DecodeMode::reference();
  static_assert(!decode_mode.torch_group_norm, "torch's GroupNorm cannot be tiled");

  planner::PlanInputs in;
  in.shapes = planner::shapes_from_index(b.index);
  in.shapes.dit_bytes = moss_dit::MossDit::persistent_bytes(b.index);
  in.shapes.dit_block_bytes = align_up(moss_dit::BlockWeights::block_device_bytes(b.index));  // the buffers' size
  in.shapes.vae_bytes = flux2_vae::Flux2VaeDecoder::persistent_bytes(b.index);
  in.request.height = req.height;
  in.request.width = req.width;
  in.request.text_lengths = {sc};
  if (g.do_cfg()) in.request.text_lengths.push_back(su);
  in.request.steps = schedule.num_steps();
  in.request.rows_per_step = 1 + (g.do_cfg() ? 1 : 0) + (g.slg_scale > 0 && g.slg_layers.any() ? 1 : 0);
  in.request.te_lengths = te_lengths;
  in.request.decode = req.decode;
  in.request.te_cached = missing.empty();
  in.request.weights_cached = m.weights_cached();
  in.request.dit_weights_cached = m.dit_weights_cached();
  in.request.vae_resident = m.vae != nullptr;
  in.workspaces.text_encoder = [&](planner::TeResidency residency, int buffers) {
    return qwen35_text::TextEncoder::workspace_bytes(b.index, te_lengths, m.te_options(residency, buffers));
  };
  in.workspaces.denoise = denoise_need;
  if (req.decode) {
    in.workspaces.decode = latent_bytes + flux2_vae::Flux2VaeDecoder::workspace_bytes(h, w, decode_mode);
    in.workspaces.decode_tiled = [&](std::size_t budget) -> std::optional<planner::TiledDecode> {
      std::optional<flux2_vae::TilePlan> tp;
      if (budget == 0) {
        tp = flux2_vae::Flux2VaeDecoder::plan_tiled_min_memory(h, w, decode_mode, 8);
      } else if (budget > latent_bytes) {
        tp = flux2_vae::Flux2VaeDecoder::plan_tiled(h, w, decode_mode, budget - latent_bytes);
      }
      if (!tp) return std::nullopt;
      return planner::TiledDecode{latent_bytes + tp->workspace_bytes, tp->relative_cost, tp->tile};
    };
  }
  std::size_t embed_bytes = 0;
  for (const auto& s : seqs) embed_bytes += s.size() * qwen35_text::kHidden * 2;
  in.workspaces.host_buffers = embed_bytes + 2 * latent_bytes +
                               (req.decode ? static_cast<std::size_t>(req.height) * static_cast<std::size_t>(req.width) * 3 : 0);
  in.caps.te_max_buffers = 2;
  in.caps.te_ring_bytes = qwen35_text::TextEncoder::host_bytes(b.index, m.te_options(planner::TeResidency::kStreamed, 2));
  in.caps.te_host_resident = true;
  in.caps.dit_parking = true;
  in.caps.dit_streaming = true;
  in.caps.dit_max_buffers = 3;
  {
    moss_dit::Residency file;
    file.resident_blocks = 0;
    in.caps.dit_ring_bytes = moss_dit::BlockWeights::host_bytes(b.index, file);
  }
  in.caps.dit_host_pinned = true;
  in.caps.dit_graph_bytes = block_path == moss_dit::BlockPath::kFusedGraphs ? moss_dit::graph_cache_bytes() : 0;
  in.hardware = r.hardware;
  in.budgets.vram_free = runtime::DeviceArena::free_device_bytes() + (m.arena ? m.arena->capacity() : 0);
  in.budgets.vram_reserve = m.options.vram_reserve;
  in.budgets.vram_budget = m.options.vram_budget;
  in.budgets.host_available = planner::host_available_bytes();
  const std::size_t held_host = m.held_host_bytes(!missing.empty());
  in.budgets.host_private_now = r.memory.host_private_start > held_host ? r.memory.host_private_start - held_host : 0;
  in.budgets.host_budget = m.options.host_ram_budget;
  in.mode = m.options.mode;
  in.cache_disk_spill = m.cache && m.cache->disk_enabled();
  // Throws planner::BudgetError before any heavy work. A resident VAE decoder the request does not need is counted;
  // when that alone makes the request infeasible, plan without it and recreate the arena (dropping it) below.
  bool drop_resident = false;
  try {
    r.plan = planner::plan(in);
  } catch (const planner::BudgetError&) {
    if (!in.request.vae_resident || req.decode) throw;
    in.request.vae_resident = false;
    r.plan = planner::plan(in);
    drop_resident = true;
  }
  // A loaded DiT stays (in its held arena, with its residency) unless the request's own plan saves more than reloading
  // it costs: no reload churn when request sizes alternate or the free device memory moves a little.
  bool kept = false;
  if (m.arena && m.dit && m.dit_bundle == &b && !drop_resident) {
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
  const planner::ExecutionPlan& plan = r.plan;
  if (m.cache) m.cache->set_max_bytes(plan.cache.host_bytes);
  tm.plan_s = since(t0);
  if (control.on_plan) control.on_plan(plan);

  // The plan's arena. A larger existing one is kept (its resident models with it) unless the request needs less than
  // arena_shrink_below of it and the plan does not keep the loaded DiT.
  const bool shrink = m.arena && !kept &&
                      static_cast<double>(plan.arena_bytes) < m.options.arena_shrink_below * static_cast<double>(m.arena->capacity());
  if (!m.arena || m.arena->capacity() < plan.arena_bytes || drop_resident || shrink) m.reset_arena(plan.arena_bytes);
  m.arena->reset_high_water();
  r.memory.arena_capacity = std::max(r.memory.arena_capacity, m.arena->capacity());
  m.sample_memory();

  // 1. Text encoder for the sequences the cache did not serve; its device memory is released when it returns.
  if (!keep_going(Phase::kTextEncoder)) return finish();
  if (!missing.empty()) {
    if (plan.dit.park_for_text_encoder && (m.dit || m.vae)) {
      r.memory.dit_parked_for_text_encoder = m.dit != nullptr;
      m.reset_arena(m.arena->capacity(), true);
    }
    std::vector<std::vector<std::int32_t>> todo;
    for (const std::size_t i : missing) todo.push_back(seqs[i]);
    t0 = Clock::now();
    m.ensure_te(b, m.te_options(plan.te.residency, plan.te.buffers));
    qwen35_text::EncodeOptions eo;
    eo.cancel = control.cancel;
    std::vector<std::vector<std::uint16_t>> out;
    try {
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

  // 2. DiT, resident across calls.
  if (!keep_going(Phase::kDitLoad)) return finish();
  const moss_dit::Residency residency = m.dit_residency(plan.dit);
  try {
    m.ensure_dit(b, residency, denoise_need, control.cancel, tm);
  } catch (const moss_dit::Cancelled&) {
    r.cancelled = true;
    return finish();
  }
  m.dit->set_cancel(control.cancel);
  // A DiT read from the weight files: page-cached or cold (the loader's one reader; O_DIRECT reads say nothing).
  const moss_dit::LoadStats& ls = m.dit->load_stats();
  if (tm.dit_load_s > 0 && m.learner && !m.options.direct_io && ls.seconds > 0 && ls.bytes >= (std::uint64_t{1} << 30)) {
    m.learner->observe_dit_load(static_cast<double>(ls.bytes) / ls.seconds / 1e9);
  }
  if (m.dit->block_path() != block_path) m.dit->set_block_path(block_path);
  r.block_path = block_path;
  if (block_path == moss_dit::BlockPath::kFusedGraphs && residency.streamed()) {
    r.block_path = moss_dit::BlockPath::kFused;  // streamed blocks wait on host-driven copies: never captured
    if (r.block_path_note.empty()) r.block_path_note = "CUDA graphs need every DiT block resident: blocks ran fused, eagerly";
  }

  // 3. Text refiner once per row, then the sampling loop.
  if (!keep_going(Phase::kDenoise)) return finish();
  {
    runtime::StackScope phase(*m.arena);
    const Tensor ref_c = m.arena->alloc_tensor(DType::kBF16, {sc, moss_dit::kDim});
    const Tensor ref_u = g.do_cfg() ? m.arena->alloc_tensor(DType::kBF16, {su, moss_dit::kDim}) : Tensor{};
    {
      runtime::StackScope refine(*m.arena);
      t0 = Clock::now();
      for (std::size_t i = 0; i < embeds.size(); ++i) {
        const std::int64_t s = static_cast<std::int64_t>(seqs[i].size());
        const Tensor te = m.arena->alloc_tensor(DType::kBF16, {s, moss_dit::kTextDim});
        MI_CUDA_CHECK(cudaMemcpyAsync(te.data, embeds[i]->data.data(), te.nbytes(), cudaMemcpyHostToDevice, stream));
        m.dit->encode_text(te, i == 0 ? ref_c : ref_u, stream);
      }
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      tm.refine_s = since(t0);
    }

    // P:400-402: fp32 CPU noise [1, 128, h, w].
    const Tensor latents = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w});
    {
      const std::vector<float> noise = rng::torch_cpu_randn(r.seed, latents.numel());
      MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, noise.data(), latents.nbytes(), cudaMemcpyHostToDevice, stream));
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Streaming diagnostics per step: the provider's running totals (read after the step's synchronization).
    DitStreamStats& ds = r.dit_stream;
    ds.streamed = residency.streamed();
    if (ds.streamed) {
      ds.resident_blocks = residency.resident_blocks;
      ds.buffers = residency.buffers;
      ds.source = moss_dit::stream_source_name(residency.source);
      m.dit->reset_stream_stats();
    }
    moss_dit::StreamStats seen;
    const StepCallback on_step = [&](const StepInfo& info) {
      if (ds.streamed) {
        const moss_dit::StreamStats& total = m.dit->total_stream_stats();
        DitStreamStats::Step st;
        const double copy_s = total.copy_s - seen.copy_s;
        st.gbps = copy_s > 0 ? static_cast<double>(total.bytes - seen.bytes) / copy_s / 1e9 : 0.0;
        st.stall_s = total.stall_s - seen.stall_s;
        st.hidden = copy_s > 0 ? std::max(0.0, 1.0 - st.stall_s / copy_s) : 0.0;
        st.stalls = total.stalls - seen.stalls;
        ds.steps.push_back(st);
        seen = total;
        ds.bytes = total.bytes;
        ds.copy_s = total.copy_s;
        ds.stall_s = total.stall_s;
        ds.stalls = total.stalls;
      }
      tm.step_s.push_back(info.step_seconds);
      if (control.on_step && !control.on_step(info)) r.cancelled = true;
      if (cancel_requested()) r.cancelled = true;
      return !r.cancelled;
    };
    t0 = Clock::now();
    try {
      r.steps_run = denoise(*m.dit, *m.arena, ref_c, ref_u, schedule, g, req.sampler, latents, stream, on_step);
    } catch (const moss_dit::Cancelled&) {
      // Inside a step's forward, before its update: the latents are those after the last finished step.
      r.cancelled = true;
      r.steps_run = static_cast<int>(tm.step_s.size());
    }
    tm.denoise_s = since(t0);
    if (!r.cancelled) m.learn_from_denoise(r);
    m.sample_memory();
    r.latents.resize(static_cast<std::size_t>(latents.numel()));
    MI_CUDA_CHECK(cudaMemcpyAsync(r.latents.data(), latents.data, latents.nbytes(), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // 4. VAE decode (P:428-438) to RGB8: untiled when it fits the arena next to the resident models, else tiled, with the
  // same bytes either way (decode_mode). Not after a cancel, even one seen at the last step.
  if (req.decode && !r.cancelled && keep_going(Phase::kDecode)) {
    m.ensure_vae(b, plan.vae.stack_bytes, plan.dit.park_for_decode, tm);
    runtime::StackScope phase(*m.arena);
    const Tensor latents = m.arena->alloc_tensor(DType::kF32, {moss_dit::kLatentChannels, h, w});
    MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, r.latents.data(), latents.nbytes(), cudaMemcpyHostToDevice, stream));
    r.rgb.resize(static_cast<std::size_t>(req.height) * static_cast<std::size_t>(req.width) * 3);
    const Tensor rgb = Tensor::view(r.rgb.data(), DType::kU8, Device::kCPU, {req.height, req.width, 3});
    t0 = Clock::now();
    m.vae->set_cancel(control.cancel);
    try {
      const std::optional<flux2_vae::TilePlan> used =
          m.vae->decode_within(latents, rgb, decode_mode, m.arena->available(), stream);
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      r.decode_tiled = used.has_value();
      r.decode_schedule = used ? used->describe() : "untiled";
    } catch (const flux2_vae::Cancelled&) {
      r.cancelled = true;
      r.rgb.clear();
    }
    tm.decode_s = since(t0);
    m.sample_memory();
  }
  return finish();
}

}  // namespace mi::pipeline
