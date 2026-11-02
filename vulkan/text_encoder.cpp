#include "text_encoder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "half.h"
#include "safetensors.h"
#include "te_kernels.h"

namespace mi::vk {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("vulkan text encoder: " + what); }

constexpr std::uint32_t kQkvDim =
    2 * kTeHeads * kTeHeadDim + 2 * kTeKvHeads * kTeHeadDim;       // [q|gate per head] k v: 10240
constexpr std::uint32_t kQgDim = 2 * kTeHeads * kTeHeadDim;        // 8192
constexpr std::uint32_t kKvDim = kTeKvHeads * kTeHeadDim;          // 1024
constexpr std::uint32_t kValueDim = kGdnValueHeads * kGdnHeadDim;  // 4096
constexpr std::uint32_t kQkvzDim = kGdnConvDim + kValueDim;        // 12288
constexpr std::uint32_t kGateUpDim = 2 * kTeIntermediate;          // 24576
constexpr std::uint32_t kRopeDim = 64;
constexpr std::uint32_t kMaxTokens = 65535;

std::uint64_t align256(std::uint64_t v) { return (v + 255) & ~std::uint64_t{255}; }

// Qwen3_5TextRotaryEmbedding inv_freq for rotary dim 64, theta 1e7, as torch's CPU computes it (kernels/basic/rope.h).
constexpr std::uint32_t kRopeInvFreqBits[32] = {
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
    0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
    0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
    0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44};

// Rows [cos(32) | sin(32)] of positions 0 .. len - 1: fp32 phase = position * inv_freq, cos and sin rounded to bf16.
std::vector<std::uint16_t> rope_table(std::uint32_t len) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(len) * kRopeDim);
  for (std::uint32_t p = 0; p < len; ++p) {
    for (std::uint32_t i = 0; i < 32; ++i) {
      float f = 0;
      std::memcpy(&f, &kRopeInvFreqBits[i], 4);
      const float phase = static_cast<float>(p) * f;
      out[static_cast<std::size_t>(p) * kRopeDim + i] = float_to_bf16(std::cos(phase));
      out[static_cast<std::size_t>(p) * kRopeDim + 32 + i] = float_to_bf16(std::sin(phase));
    }
  }
  return out;
}

struct Piece {
  const st::File* file;
  const st::TensorInfo* tensor;
  std::uint64_t offset;  // in the layer buffer
};

// Where a layer's tensors sit in the device buffer. Tensors read by one GEMM are adjacent, in this order: GatedDeltaNet
// in_proj_qkv | in_proj_z and in_proj_b | in_proj_a, attention q_proj | k_proj | v_proj, MLP gate_proj | up_proj.
struct LayerPlan {
  bool full_attention = false;
  std::vector<Piece> pieces;
  std::uint64_t bytes = 0;
  std::uint64_t in_norm = 0, post_norm = 0, gate_up = 0, down = 0;
  std::uint64_t qkvz = 0, ba = 0, conv = 0, a_log = 0, dt_bias = 0, gdn_norm = 0, gdn_out = 0;
  std::uint64_t qkv = 0, o = 0, q_norm = 0, k_norm = 0;
};

// One scratch buffer, laid out per phase of a layer (offsets for `t` tokens).
struct GdnScratch {
  std::uint64_t qkvz, ba, conv, g, beta, core, ws, end;
  explicit GdnScratch(std::uint32_t t) {
    qkvz = 0;
    ba = align256(qkvz + std::uint64_t{t} * kQkvzDim * 2);
    conv = align256(ba + std::uint64_t{t} * 64 * 2);
    g = align256(conv + std::uint64_t{t} * kGdnConvDim * 2);
    beta = align256(g + std::uint64_t{t} * kGdnValueHeads * 4);
    core = align256(beta + std::uint64_t{t} * kGdnValueHeads * 2);
    ws = align256(core + std::uint64_t{t} * kValueDim * 2);
    end = ws + GdnWorkspace::bytes(t);
  }
};
struct AttnScratch {
  std::uint64_t qkv, q, attn, end;
  explicit AttnScratch(std::uint32_t t) {
    qkv = 0;
    q = align256(qkv + std::uint64_t{t} * kQkvDim * 2);
    attn = align256(q + std::uint64_t{t} * kTeHidden * 2);
    end = attn + std::uint64_t{t} * kTeHidden * 2;
  }
};
struct MlpScratch {
  std::uint64_t gu, act, end;
  explicit MlpScratch(std::uint32_t t) {
    gu = 0;
    act = align256(gu + std::uint64_t{t} * kGateUpDim * 2);
    end = act + std::uint64_t{t} * kTeIntermediate * 2;
  }
};

// Collects recorded work and submits it in pieces of at most `budget` estimated FLOP.
class Submitter {
 public:
  Submitter(Device& device, double budget, bool profile, TextEncoderStats& stats)
      : device_(device), budget_(budget), profile_(profile), stats_(stats) {}

  // `name` labels the op in TextEncoderStats::op_seconds when profiling, which submits every op on its own.
  void add(const char* name, double flop, std::function<void(VkCommandBuffer)> op) {
    if (profile_) {
      flush();
      ops_.push_back(std::move(op));
      const auto t0 = Clock::now();
      flush();
      stats_.op_seconds[name] += seconds_since(t0);
      return;
    }
    if (!ops_.empty() && work_ + flop > budget_) flush();
    ops_.push_back(std::move(op));
    work_ += flop;
  }
  void flush() {
    if (ops_.empty()) return;
    const auto t0 = Clock::now();
    device_.run([&](VkCommandBuffer cb) {
      for (const auto& op : ops_) op(cb);
    });
    stats_.compute_s += seconds_since(t0);
    ++stats_.submits;
    ops_.clear();
    work_ = 0;
  }
  double budget() const { return budget_; }

 private:
  Device& device_;
  double budget_;
  bool profile_;
  TextEncoderStats& stats_;
  std::vector<std::function<void(VkCommandBuffer)>> ops_;
  double work_ = 0;
};

// The reader thread's handoff: slot s holds layer `filled[s]` once read; the main thread frees it after its copy.
struct Handoff {
  std::mutex mu;
  std::condition_variable cv;
  int filled[2] = {-1, -1};
  bool busy[2] = {false, false};
  bool abort = false;
  std::exception_ptr error;
};

}  // namespace

struct TextEncoder::Impl {
  Device& device;
  BlockPath path;
  TextEncoderOptions options;
  BlockKernels gemm;
  TeKernels te;
  std::map<std::string, std::unique_ptr<st::File>> files;
  std::map<std::string, std::pair<const st::File*, const st::TensorInfo*>> tensors;
  const st::File* embed_file = nullptr;
  const st::TensorInfo* embed = nullptr;
  std::pair<const st::File*, const st::TensorInfo*> final_norm{};
  std::vector<LayerPlan> layers;
  std::uint64_t layer_bytes = 0;
  TextEncoderStats stats;

  Impl(Device& d, const std::filesystem::path& dir, BlockPath p, TextEncoderOptions o)
      : device(d), path(p), options(o), gemm(d, p), te(d) {
    const std::map<std::string, std::string> index = st::read_index(dir / "model.safetensors.index.json");
    for (const auto& [name, shard] : index) {
      if (name.rfind("language_model.", 0) != 0) continue;
      auto& file = files[shard];
      if (!file) file = std::make_unique<st::File>(dir / shard);
      tensors[name] = {file.get(), &file->at(name)};
    }
    const auto& e = lookup("language_model.embed_tokens.weight", {kTeVocab, kTeHidden});
    embed_file = e.first;
    embed = e.second;
    final_norm = lookup("language_model.norm.weight", {kTeHidden});
    for (int i = 0; i < kTeLayers; ++i) layers.push_back(plan_layer(i));
    for (const LayerPlan& l : layers) layer_bytes = std::max(layer_bytes, l.bytes);
  }

  const std::pair<const st::File*, const st::TensorInfo*>& lookup(const std::string& name,
                                                                  const std::vector<std::int64_t>& shape) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) fail("no tensor " + name);
    if (it->second.second->dtype != "BF16" || it->second.second->shape != shape)
      fail("tensor " + name + " has an unexpected dtype or shape");
    return it->second;
  }

  LayerPlan plan_layer(int i) const {
    LayerPlan l;
    l.full_attention = i % 4 == 3;
    const std::string p = "language_model.layers." + std::to_string(i) + ".";
    // Tensors of one group are placed back to back: their sizes are multiples of 256 bytes, so alignment adds no gap.
    const auto add = [&](const std::string& name, const std::vector<std::int64_t>& shape) {
      const auto& [file, t] = lookup(p + name, shape);
      if (t->bytes % 256 != 0 && shape.size() == 2) fail("tensor " + t->name + " is not a multiple of 256 bytes");
      l.bytes = align256(l.bytes);
      const std::uint64_t at = l.bytes;
      l.pieces.push_back({file, t, at});
      l.bytes += t->bytes;
      return at;
    };
    l.in_norm = add("input_layernorm.weight", {kTeHidden});
    l.post_norm = add("post_attention_layernorm.weight", {kTeHidden});
    l.gate_up = add("mlp.gate_proj.weight", {kTeIntermediate, kTeHidden});
    add("mlp.up_proj.weight", {kTeIntermediate, kTeHidden});
    l.down = add("mlp.down_proj.weight", {kTeHidden, kTeIntermediate});
    if (l.full_attention) {
      l.qkv = add("self_attn.q_proj.weight", {kQgDim, kTeHidden});
      add("self_attn.k_proj.weight", {kKvDim, kTeHidden});
      add("self_attn.v_proj.weight", {kKvDim, kTeHidden});
      l.o = add("self_attn.o_proj.weight", {kTeHidden, kTeHeads * kTeHeadDim});
      l.q_norm = add("self_attn.q_norm.weight", {kTeHeadDim});
      l.k_norm = add("self_attn.k_norm.weight", {kTeHeadDim});
    } else {
      l.qkvz = add("linear_attn.in_proj_qkv.weight", {kGdnConvDim, kTeHidden});
      add("linear_attn.in_proj_z.weight", {kValueDim, kTeHidden});
      l.ba = add("linear_attn.in_proj_b.weight", {kGdnValueHeads, kTeHidden});
      add("linear_attn.in_proj_a.weight", {kGdnValueHeads, kTeHidden});
      l.conv = add("linear_attn.conv1d.weight", {kGdnConvDim, 1, 4});
      l.a_log = add("linear_attn.A_log", {kGdnValueHeads});
      l.dt_bias = add("linear_attn.dt_bias", {kGdnValueHeads});
      l.gdn_norm = add("linear_attn.norm.weight", {kGdnHeadDim});
      l.gdn_out = add("linear_attn.out_proj.weight", {kTeHidden, kValueDim});
    }
    l.bytes = align256(l.bytes);
    return l;
  }

  double budget() const {
    const double gflop = options.submit_gflop > 0 ? options.submit_gflop : (path == BlockPath::kCoop ? 200.0 : 20.0);
    return gflop * 1e9;
  }

  std::vector<std::vector<std::uint16_t>> encode(const std::vector<std::vector<std::int32_t>>& seqs,
                                                 const TeLayerHook& hook);

  // A layer's tensors into `dst` at their offsets, in chunks of 16 MiB that `reader_threads` threads take in turn:
  // several reads in flight are what a cold NVMe drive needs to reach its speed.
  void read_layer(const LayerPlan& l, std::byte* dst) const {
    constexpr std::uint64_t kChunk = std::uint64_t{16} << 20;
    struct Chunk {
      const Piece* piece;
      std::uint64_t rel, bytes;
    };
    std::vector<Chunk> chunks;
    for (const Piece& p : l.pieces) {
      for (std::uint64_t rel = 0; rel < p.tensor->bytes; rel += kChunk)
        chunks.push_back({&p, rel, std::min(kChunk, p.tensor->bytes - rel)});
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(std::max(1, options.reader_threads)));
    const auto work = [&](std::size_t t) {
      try {
        for (std::size_t i = next++; i < chunks.size(); i = next++) {
          const Chunk& c = chunks[i];
          c.piece->file->read(*c.piece->tensor, dst + c.piece->offset + c.rel, c.rel, c.bytes);
        }
      } catch (...) {
        errors[t] = std::current_exception();
        next = chunks.size();
      }
    };
    {
      std::vector<std::jthread> helpers;
      for (std::size_t t = 1; t < errors.size(); ++t) helpers.emplace_back(work, t);
      work(0);
    }
    for (const auto& e : errors) {
      if (e) std::rethrow_exception(e);
    }
  }
};

TextEncoder::TextEncoder(Device& device, const std::filesystem::path& dir)
    : TextEncoder(device, dir, device.info().caps.coop_bf16 ? BlockPath::kCoop : BlockPath::kPlain) {}

TextEncoder::TextEncoder(Device& device, const std::filesystem::path& dir, BlockPath path, TextEncoderOptions options)
    : impl_(std::make_unique<Impl>(device, dir, path, options)) {}

TextEncoder::~TextEncoder() = default;

BlockPath TextEncoder::path() const { return impl_->path; }
const TextEncoderStats& TextEncoder::last_stats() const { return impl_->stats; }

std::vector<std::vector<std::uint16_t>> TextEncoder::encode(const std::vector<std::vector<std::int32_t>>& sequences,
                                                            const TeLayerHook& hook) {
  return impl_->encode(sequences, hook);
}

std::vector<std::vector<std::uint16_t>> TextEncoder::Impl::encode(const std::vector<std::vector<std::int32_t>>& seqs,
                                                                  const TeLayerHook& hook) {
  const auto t_start = Clock::now();
  if (seqs.empty()) throw std::invalid_argument("vulkan text encoder: no sequences");
  std::vector<std::int32_t> cu = {0};
  std::uint32_t max_len = 0;
  for (const auto& s : seqs) {
    if (s.empty()) throw std::invalid_argument("vulkan text encoder: empty sequence");
    for (const std::int32_t id : s) {
      if (id < 0 || id >= kTeVocab)
        throw std::invalid_argument("vulkan text encoder: token id " + std::to_string(id) + " outside the vocabulary");
    }
    if (static_cast<std::uint64_t>(cu.back()) + s.size() > kMaxTokens)
      throw std::invalid_argument("vulkan text encoder: more than 65535 tokens");
    cu.push_back(cu.back() + static_cast<std::int32_t>(s.size()));
    max_len = std::max(max_len, static_cast<std::uint32_t>(s.size()));
  }
  const auto T = static_cast<std::uint32_t>(cu.back());
  stats = TextEncoderStats{};
  stats.tokens = T;

  const GdnScratch gs(T);
  const AttnScratch as(T);
  const MlpScratch ms(T);
  const std::uint64_t scratch_bytes = std::max({gs.end, as.end, ms.end});
  const std::uint64_t act_bytes = std::uint64_t{T} * kTeHidden * 2;

  // Layer k runs from weights[k % 2] while layer k + 1 is copied into the other.
  std::unique_ptr<Buffer> weights[2] = {std::make_unique<Buffer>(device, layer_bytes, Buffer::Kind::kDevice, true),
                                        std::make_unique<Buffer>(device, layer_bytes, Buffer::Kind::kDevice, true)};
  Buffer x(device, act_bytes), h(device, act_bytes), scratch(device, scratch_bytes);
  Buffer rope(device, std::uint64_t{max_len} * kRopeDim * 2), small(device, 3 * kTeHidden * 2);
  stats.device_bytes = 2 * layer_bytes + x.bytes() + h.bytes() + scratch.bytes() + rope.bytes() + small.bytes();
  std::unique_ptr<Buffer> staging[2] = {std::make_unique<Buffer>(device, layer_bytes, Buffer::Kind::kStaging, true),
                                        std::make_unique<Buffer>(device, layer_bytes, Buffer::Kind::kStaging, true)};
  stats.staging_bytes = 2 * layer_bytes;

  // Small constants: the final norm weight and a vector of ones (the gate that makes the gated-residual GEMM a plain
  // residual add: bf16(x + bf16(ab) * 1) = bf16(x + bf16(ab))).
  const VkDeviceAddress final_norm_w = small.address(), ones = small.address() + kTeHidden * 2;
  {
    std::vector<std::uint16_t> host(2 * kTeHidden, 0x3f80);
    final_norm.first->read(*final_norm.second, host.data());
    small.upload(host.data(), host.size() * 2);
    const std::vector<std::uint16_t> table = rope_table(max_len);
    rope.upload(table.data(), table.size() * 2);
  }

  Handoff handoff;
  double read_s = 0;
  std::thread reader([&] {
    try {
      for (int k = 0; k < kTeLayers; ++k) {
        const int s = k % 2;
        {
          std::unique_lock<std::mutex> lock(handoff.mu);
          handoff.cv.wait(lock, [&] { return handoff.abort || (!handoff.busy[s] && handoff.filled[s] < 0); });
          if (handoff.abort) return;
          handoff.busy[s] = true;
        }
        const auto t0 = Clock::now();
        read_layer(layers[static_cast<std::size_t>(k)], static_cast<std::byte*>(staging[s]->mapped()));
        read_s += seconds_since(t0);
        std::lock_guard<std::mutex> lock(handoff.mu);
        handoff.busy[s] = false;
        handoff.filled[s] = k;
        handoff.cv.notify_all();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(handoff.mu);
      handoff.error = std::current_exception();
      handoff.abort = true;
      handoff.cv.notify_all();
    }
  });
  const auto stop_reader = [&] {
    {
      std::lock_guard<std::mutex> lock(handoff.mu);
      handoff.abort = true;
      handoff.cv.notify_all();
    }
    if (reader.joinable()) reader.join();
  };

  std::vector<std::vector<std::uint16_t>> result;
  std::vector<std::uint64_t> in_flight;  // copies into the buffers above, which must finish before the buffers go
  try {
    // Embedding rows of the ids, gathered on the host.
    {
      const auto t0 = Clock::now();
      std::vector<std::uint16_t> rows(static_cast<std::size_t>(T) * kTeHidden);
      std::size_t r = 0;
      for (const auto& s : seqs) {
        for (const std::int32_t id : s) {
          embed_file->read(*embed, rows.data() + r++ * kTeHidden, std::uint64_t(id) * kTeHidden * 2, kTeHidden * 2);
        }
      }
      x.upload(rows.data(), rows.size() * 2);
      stats.embed_s = seconds_since(t0);
    }

    Submitter sub(device, budget(), options.profile, stats);
    const VkDeviceAddress xa = x.address(), ha = h.address(), sa = scratch.address();
    const auto bf = [](VkDeviceAddress base, std::uint64_t elements) { return base + elements * 2; };

    // GEMMs split along M so that each piece fits the submit budget; rows of the output do not depend on the split.
    const auto rows_per_piece = [&](std::uint32_t n, std::uint32_t k) {
      const double per_row = 2.0 * n * k;
      auto rows = static_cast<std::uint32_t>(std::max(1.0, std::floor(sub.budget() / per_row)));
      if (rows >= 128) rows -= rows % 128;
      return rows;
    };
    const auto linear = [&](const char* name, VkDeviceAddress out, std::uint32_t ldo, VkDeviceAddress in,
                            std::uint32_t ldi, VkDeviceAddress w, std::uint32_t n, std::uint32_t k) {
      const std::uint32_t step = rows_per_piece(n, k);
      for (std::uint32_t m0 = 0; m0 < T; m0 += step) {
        const std::uint32_t m = std::min(step, T - m0);
        sub.add(name, 2.0 * m * n * k, [=, this](VkCommandBuffer cb) {
          gemm.record_gemm(cb, bf(out, std::uint64_t{m0} * ldo), ldo, bf(in, std::uint64_t{m0} * ldi), ldi, w, k, m, n,
                           k);
        });
      }
    };
    const auto linear_residual = [&](const char* name, VkDeviceAddress in, std::uint32_t ldi, VkDeviceAddress w,
                                     std::uint32_t k) {
      const std::uint32_t step = rows_per_piece(kTeHidden, k);
      for (std::uint32_t m0 = 0; m0 < T; m0 += step) {
        const std::uint32_t m = std::min(step, T - m0);
        sub.add(name, 2.0 * m * kTeHidden * k, [=, this](VkCommandBuffer cb) {
          gemm.record_gemm_gated_residual(cb, bf(xa, std::uint64_t{m0} * kTeHidden), kTeHidden,
                                          bf(in, std::uint64_t{m0} * ldi), ldi, w, k, ones, m, kTeHidden, k);
        });
      }
    };
    const auto for_each_sequence = [&](auto&& f) {
      for (std::size_t i = 0; i + 1 < cu.size(); ++i) {
        f(static_cast<std::uint32_t>(cu[i]), static_cast<std::uint32_t>(cu[i + 1] - cu[i]));
      }
    };
    const auto hidden_norm = [&](VkDeviceAddress weight, VkDeviceAddress out) {
      for_each_sequence([&](std::uint32_t s0, std::uint32_t len) {
        RmsNormArgs a;
        a.x = bf(xa, std::uint64_t{s0} * kTeHidden);
        a.out = bf(out, std::uint64_t{s0} * kTeHidden);
        a.weight = weight;
        a.ldx = a.ldo = kTeHidden;
        a.tokens = len;
        a.d = kTeHidden;
        sub.add("rms_norm", 8.0 * len * kTeHidden, [=, this](VkCommandBuffer cb) { te.record_rms_norm(cb, a); });
      });
    };

    const auto wait_staged = [&](int j) {
      const auto t0 = Clock::now();
      std::unique_lock<std::mutex> lock(handoff.mu);
      handoff.cv.wait(lock, [&] { return handoff.error != nullptr || handoff.filled[j % 2] == j; });
      if (handoff.error) std::rethrow_exception(handoff.error);
      stats.read_wait_s += seconds_since(t0);
    };
    const auto release_staged = [&](int j) {
      std::lock_guard<std::mutex> lock(handoff.mu);
      handoff.filled[j % 2] = -1;
      handoff.cv.notify_all();
    };
    // Layer j's staging -> device copy, on the transfer queue where there is one.
    const auto copy = [&](int j) {
      in_flight.push_back(
          device.copy_async(*staging[j % 2], 0, *weights[j % 2], 0, layers[static_cast<std::size_t>(j)].bytes));
      return in_flight.back();
    };
    // Blocks until layer j's copy is done, then hands its staging slot back to the reader.
    const auto copied = [&](int j, std::uint64_t ticket) {
      const auto t0 = Clock::now();
      device.host_wait(ticket);
      in_flight.erase(std::find(in_flight.begin(), in_flight.end(), ticket));
      stats.upload_s += seconds_since(t0);
      release_staged(j);
    };
    std::uint64_t tickets[kTeLayers] = {};
    wait_staged(0);
    tickets[0] = copy(0);
    copied(0, tickets[0]);

    for (int k = 0; k < kTeLayers; ++k) {
      const LayerPlan& l = layers[static_cast<std::size_t>(k)];
      device.gpu_wait(tickets[k]);  // the layer's first submit waits for its weights
      // The next layer's copy runs while this layer computes.
      if (k + 1 < kTeLayers) {
        wait_staged(k + 1);
        tickets[k + 1] = copy(k + 1);
      }
      const VkDeviceAddress wa = weights[k % 2]->address();
      const auto W = [&](std::uint64_t offset) { return wa + offset; };

      hidden_norm(W(l.in_norm), ha);
      if (l.full_attention) {
        const VkDeviceAddress qkv = sa + as.qkv, q = sa + as.q, attn = sa + as.attn;
        linear("gemm q|k|v", qkv, kQkvDim, ha, kTeHidden, W(l.qkv), kQkvDim, kTeHidden);
        for_each_sequence([&](std::uint32_t s0, std::uint32_t len) {
          RmsNormArgs qn;
          qn.x = bf(qkv, std::uint64_t{s0} * kQkvDim);
          qn.ldx = kQkvDim;
          qn.hsx = 2 * kTeHeadDim;
          qn.out = bf(q, std::uint64_t{s0} * kTeHidden);
          qn.ldo = kTeHidden;
          qn.hso = kTeHeadDim;
          qn.weight = W(l.q_norm);
          qn.tokens = len;
          qn.heads = kTeHeads;
          qn.d = kTeHeadDim;
          qn.rope = rope.address();
          RmsNormArgs kn = qn;
          kn.x = kn.out = bf(qkv, std::uint64_t{s0} * kQkvDim + kQgDim);
          kn.ldx = kn.ldo = kQkvDim;
          kn.hsx = kn.hso = kTeHeadDim;
          kn.weight = W(l.k_norm);
          kn.heads = kTeKvHeads;
          sub.add("q, k norm+rope", 8.0 * len * kQgDim, [=, this](VkCommandBuffer cb) {
            te.record_rms_norm(cb, qn);
            te.record_rms_norm(cb, kn);
          });
        });
        for_each_sequence([&](std::uint32_t s0, std::uint32_t len) {
          TeAttentionArgs a;
          a.q = bf(q, std::uint64_t{s0} * kTeHidden);
          a.k = bf(qkv, std::uint64_t{s0} * kQkvDim + kQgDim);
          a.v = bf(qkv, std::uint64_t{s0} * kQkvDim + kQgDim + kKvDim);
          a.o = bf(attn, std::uint64_t{s0} * kTeHidden);
          a.ldq = a.ldo = kTeHidden;
          a.ldk = a.ldv = kQkvDim;
          a.hsq = a.hsk = a.hsv = a.hso = kTeHeadDim;
          a.heads = kTeHeads;
          a.group = kTeHeads / kTeKvHeads;
          a.scale = 0.0625f;
          // Query rows in pieces: row i costs about 4 * 256 * (i + 1) FLOP per head.
          const double per_row = 4.0 * kTeHeadDim * kTeHeads * len;
          const auto step = std::max(16u, static_cast<std::uint32_t>(std::floor(sub.budget() / per_row)) / 16 * 16);
          for (std::uint32_t r0 = 0; r0 < len; r0 += step) {
            a.q_first = r0;
            a.rows = std::min(step, len - r0);
            sub.add("attention", per_row * a.rows, [=, this](VkCommandBuffer cb) { te.record_attention(cb, a); });
          }
        });
        sub.add("sigmoid gate", 4.0 * T * kTeHidden, [=, this](VkCommandBuffer cb) {
          te.record_gate_mul(cb, TeKernels::GateMul::kSigmoid, attn, kTeHidden, bf(qkv, kTeHeadDim), kQkvDim, attn,
                             kTeHidden, T, kTeHidden, kTeHeadDim, 2 * kTeHeadDim);
        });
        linear_residual("gemm o+residual", attn, kTeHidden, W(l.o), kTeHidden);
      } else {
        const VkDeviceAddress qkvz = sa + gs.qkvz, ba = sa + gs.ba, conv = sa + gs.conv, g = sa + gs.g,
                              beta = sa + gs.beta, core = sa + gs.core;
        linear("gemm qkv|z", qkvz, kQkvzDim, ha, kTeHidden, W(l.qkvz), kQkvzDim, kTeHidden);
        linear("gemm b|a", ba, 64, ha, kTeHidden, W(l.ba), 64, kTeHidden);
        for_each_sequence([&](std::uint32_t s0, std::uint32_t len) {
          sub.add("conv1d+silu", 8.0 * len * kGdnConvDim, [=, this](VkCommandBuffer cb) {
            te.record_conv_silu(cb, bf(qkvz, std::uint64_t{s0} * kQkvzDim), kQkvzDim, W(l.conv),
                                bf(conv, std::uint64_t{s0} * kGdnConvDim), kGdnConvDim, len);
          });
        });
        GdnArgs ga;
        ga.qkv = conv;
        ga.ldqkv = kGdnConvDim;
        ga.g = g;
        ga.beta = beta;
        ga.out = core;
        ga.ws = GdnWorkspace::at(sa + gs.ws, T);
        ga.cu = cu;
        // A rough count per token and value head of the chunk products, the solve and the scan; only sizes the submits.
        const double gdn_flop = 2.0 * T * kGdnValueHeads * (2.0 * 64 * 128 * 2 + 256.0 * 64 + 6.0 * 128 * 128);
        sub.add("gates+gated delta rule", gdn_flop, [=, this](VkCommandBuffer cb) {
          te.record_gates(cb, ba, 64, W(l.a_log), W(l.dt_bias), g, beta, T);
          te.record_gated_delta_rule(cb, ga);
        });
        for_each_sequence([&](std::uint32_t s0, std::uint32_t len) {
          RmsNormArgs a;
          a.x = a.out = bf(core, std::uint64_t{s0} * kValueDim);
          a.ldx = a.ldo = kValueDim;
          a.hsx = a.hso = kGdnHeadDim;
          a.gate = bf(qkvz, std::uint64_t{s0} * kQkvzDim + kGdnConvDim);
          a.ldg = kQkvzDim;
          a.hsg = kGdnHeadDim;
          a.weight = W(l.gdn_norm);
          a.tokens = len;
          a.heads = kGdnValueHeads;
          a.d = kGdnHeadDim;
          a.gated = true;
          sub.add("gated rms_norm", 12.0 * len * kValueDim,
                  [=, this](VkCommandBuffer cb) { te.record_rms_norm(cb, a); });
        });
        linear_residual("gemm out+residual", core, kValueDim, W(l.gdn_out), kValueDim);
      }

      hidden_norm(W(l.post_norm), ha);
      const VkDeviceAddress gu = sa + ms.gu, act = sa + ms.act;
      linear("gemm gate|up", gu, kGateUpDim, ha, kTeHidden, W(l.gate_up), kGateUpDim, kTeHidden);
      sub.add("silu_mul", 8.0 * T * kTeIntermediate, [=, this](VkCommandBuffer cb) {
        te.record_gate_mul(cb, TeKernels::GateMul::kSilu, gu, kGateUpDim, bf(gu, kTeIntermediate), kGateUpDim, act,
                           kTeIntermediate, T, kTeIntermediate, kTeIntermediate, 0);
      });
      linear_residual("gemm down+residual", act, kTeIntermediate, W(l.down), kTeIntermediate);
      sub.flush();
      if (k + 1 < kTeLayers) copied(k + 1, tickets[k + 1]);
      if (hook) hook(k, x, T);
    }
    stop_reader();

    hidden_norm(final_norm_w, ha);
    sub.flush();
    std::vector<std::uint16_t> out(static_cast<std::size_t>(T) * kTeHidden);
    h.download(out.data(), out.size() * 2);
    for (std::size_t i = 0; i + 1 < cu.size(); ++i) {
      result.emplace_back(out.begin() + std::ptrdiff_t{cu[i]} * kTeHidden,
                          out.begin() + std::ptrdiff_t{cu[i + 1]} * kTeHidden);
    }
  } catch (...) {
    stop_reader();
    for (const std::uint64_t ticket : in_flight) {
      try {
        device.host_wait(ticket);
      } catch (...) {  // already failing; the device is likely lost
      }
    }
    throw;
  }
  stats.read_s = read_s;
  stats.total_s = seconds_since(t_start);
  return result;
}

}  // namespace mi::vk
