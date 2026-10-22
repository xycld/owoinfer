#include "models/qwen35_text/text_encoder.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cublasLt.h>

#include "core/cuda_check.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "kernels/gdn/gdn.h"
#include "kernels/quant/quant.h"
#include "kernels/quant/w4a8_decode.h"
#include "runtime/pinned_ring.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/quant/quant_codec.h"
#include "weights/upload.h"

namespace mi::qwen35_text {
namespace {

using Clock = std::chrono::steady_clock;
using weights::Component;

constexpr std::int64_t kQgDim = 2 * kNumHeads * kAttnHeadDim;  // q_proj output: per head [q(256) | gate(256)]
constexpr std::int64_t kKvDim = kNumKvHeads * kAttnHeadDim;
constexpr float kAttnScale = 0.0625f;                           // head_dim ** -0.5

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("qwen35_text: " + msg); }

struct WeightSpec {
  const char* name;  // relative to the layer prefix
  std::vector<std::int64_t> shape;
};

const std::vector<WeightSpec> kCommonSpecs = {
    {"input_layernorm.weight", {kHidden}},
    {"post_attention_layernorm.weight", {kHidden}},
    {"mlp.gate_proj.weight", {kIntermediate, kHidden}},
    {"mlp.up_proj.weight", {kIntermediate, kHidden}},
    {"mlp.down_proj.weight", {kHidden, kIntermediate}},
};
const std::vector<WeightSpec> kLinearAttnSpecs = {
    {"linear_attn.A_log", {gdn::kNumVHeads}},
    {"linear_attn.dt_bias", {gdn::kNumVHeads}},
    {"linear_attn.conv1d.weight", {gdn::kConvDim, 1, gdn::kConvKernel}},
    {"linear_attn.in_proj_qkv.weight", {gdn::kConvDim, kHidden}},
    {"linear_attn.in_proj_z.weight", {gdn::kValueDim, kHidden}},
    {"linear_attn.in_proj_b.weight", {gdn::kNumVHeads, kHidden}},
    {"linear_attn.in_proj_a.weight", {gdn::kNumVHeads, kHidden}},
    {"linear_attn.norm.weight", {gdn::kHeadDim}},
    {"linear_attn.out_proj.weight", {kHidden, gdn::kValueDim}},
};
const std::vector<WeightSpec> kFullAttnSpecs = {
    {"self_attn.q_proj.weight", {kQgDim, kHidden}},
    {"self_attn.k_proj.weight", {kKvDim, kHidden}},
    {"self_attn.v_proj.weight", {kKvDim, kHidden}},
    {"self_attn.o_proj.weight", {kHidden, kNumHeads * kAttnHeadDim}},
    {"self_attn.q_norm.weight", {kAttnHeadDim}},
    {"self_attn.k_norm.weight", {kAttnHeadDim}},
};

const weights::IndexedTensor& find_tensor(const weights::WeightGroup& g, const std::string& name) {
  for (const auto& t : g.tensors) {
    if (t.name == name) return t;
  }
  fail("group " + g.name + " has no tensor " + name);
}

void check_tensor(const weights::IndexedTensor& t, const std::vector<std::int64_t>& shape) {
  if (t.dtype != weights::DType::kBF16 || t.shape != shape) {
    std::string want;
    for (auto d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
    fail("tensor " + t.name + " must be BF16 [" + want + "]");
  }
}

struct Layer {
  const weights::WeightGroup* group = nullptr;
  bool full_attention = false;
  std::string prefix;  // tensor name prefix, e.g. "language_model.layers.3."
};

// Tensor `name` of group g, with its dtype and shape.
void check_group_tensor(const weights::WeightGroup& g, const std::string& name, std::initializer_list<weights::DType> dtypes,
                        const std::vector<std::int64_t>& shape) {
  const weights::IndexedTensor& t = find_tensor(g, name);
  if (std::find(dtypes.begin(), dtypes.end(), t.dtype) == dtypes.end() || t.shape != shape) {
    std::string want;
    for (auto d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
    fail("tensor " + name + " must be " + weights::dtype_name(*dtypes.begin()) + " [" + want + "]");
  }
}

// Checks the tensors a quantized layer's linear `module` (of the bf16 shape [n, k]) contributes; returns their count.
std::size_t check_quant_linear(const weights::WeightGroup& g, const std::string& module, const LinearInfo& info,
                               const std::vector<std::int64_t>& shape) {
  using weights::DType;
  if (shape != std::vector<std::int64_t>{info.n, info.k}) fail("linear " + module + " has an unexpected shape");
  switch (info.kind) {
    case LinearInfo::Kind::kBF16:
      check_group_tensor(g, module + ".weight", {DType::kBF16}, shape);
      return 1;
    case LinearInfo::Kind::kInt8:
      check_group_tensor(g, module + ".weight", {DType::kI8}, shape);
      check_group_tensor(g, module + ".weight_scale", {DType::kF32}, {info.n});
      return 2;
    case LinearInfo::Kind::kGrouped:
      check_group_tensor(g, module + ".weight", {DType::kI8, DType::kU8}, {info.n, info.k * info.bits / 8});
      check_group_tensor(g, module + ".weight_s_rel", {DType::kF8E4M3, DType::kU8}, {info.n, info.k / info.group});
      check_group_tensor(g, module + ".weight_s_channel", {DType::kF32}, {info.n});
      if (info.codebook) check_group_tensor(g, module + ".weight_codebook", {DType::kF32}, {16});
      return info.codebook ? 4 : 3;
  }
  return 0;
}

Layer validate_layer(const weights::ModelIndex& index, int i, const QuantTextModel* qm = nullptr) {
  Layer l;
  l.group = index.find_group(Component::kTextEncoder, "layers." + std::to_string(i));
  if (l.group == nullptr) fail("bundle index has no text_encoder group layers." + std::to_string(i));
  l.full_attention = l.group->layer_type == "full_attention";
  if (!l.full_attention && l.group->layer_type != "linear_attention") {
    fail("layers." + std::to_string(i) + " has unknown layer type '" + l.group->layer_type + "'");
  }
  // Qwen3.5 interleaves three GatedDeltaNet layers with one full-attention layer.
  if (l.full_attention != (i % 4 == 3)) fail("unexpected layer type at layers." + std::to_string(i));
  const std::string suffix = "input_layernorm.weight";
  for (const auto& t : l.group->tensors) {
    if (t.name.size() >= suffix.size() && t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      l.prefix = t.name.substr(0, t.name.size() - suffix.size());
    }
  }
  if (l.prefix.empty()) fail("layers." + std::to_string(i) + " has no input_layernorm.weight");
  const auto& mixer = l.full_attention ? kFullAttnSpecs : kLinearAttnSpecs;
  std::size_t expected = 0, converted = 0;
  for (const auto* specs : {&kCommonSpecs, &mixer}) {
    for (const auto& s : *specs) {
      const std::string name = l.prefix + s.name;
      const std::string module = name.size() > 7 ? name.substr(0, name.size() - 7) : std::string();
      const LinearInfo* info = qm != nullptr && name.ends_with(".weight") ? qm->find_linear(module) : nullptr;
      if (info != nullptr) {
        expected += check_quant_linear(*l.group, module, *info, s.shape);
      } else if (qm != nullptr && qm->bf16_tensors().count(name) != 0) {
        if (qm->bf16_shape(name) != s.shape) fail("tensor " + name + " has an unexpected shape");
        ++converted;
      } else {
        check_tensor(find_tensor(*l.group, name), s.shape);
        ++expected;
      }
    }
  }
  if (qm != nullptr) {
    // Every F32 tensor of the layer the model converted must be one the layer reads.
    std::size_t f32 = 0;
    for (const auto& [name, v] : qm->bf16_tensors()) f32 += name.starts_with(l.prefix);
    if (f32 != converted) fail("layers." + std::to_string(i) + " has unexpected F32 tensors");
  }
  if (l.group->tensors.size() != expected) {
    fail("layers." + std::to_string(i) + " has " + std::to_string(l.group->tensors.size()) + " tensors, expected " +
         std::to_string(expected));
  }
  return l;
}

const weights::IndexedTensor& single_tensor(const weights::ModelIndex& index, const char* group) {
  const weights::WeightGroup* g = index.find_group(Component::kTextEncoder, group);
  if (g == nullptr || g->tensors.size() != 1) fail(std::string("bundle index needs a one-tensor text_encoder group ") + group);
  return g->tensors.front();
}

Tensor strided(void* data, DType dtype, std::initializer_list<std::int64_t> shape,
               std::initializer_list<std::int64_t> stride) {
  Tensor t = Tensor::view(data, dtype, Device::kCUDA, shape);
  std::size_t i = 0;
  for (auto s : stride) t.stride[i++] = s;
  return t;
}

// One encode()'s packed batch.
struct Batch {
  std::vector<std::int64_t> cu;  // host token offsets, cu[0] = 0, cu[N] = tokens
  Tensor cu_dev;                 // i32 [N + 1]
  std::int64_t tokens = 0;
  int max_len = 0;
  Tensor rope;  // bf16 [max_len, 64], positions 0..max_len-1 (each sequence restarts at 0)

  template <typename F>
  void for_each_sequence(F&& f) const {
    for (std::size_t i = 0; i + 1 < cu.size(); ++i) f(cu[i], cu[i + 1]);
  }
};

// A slot holds one chunk plus the direct-I/O widening of its unaligned start.
std::size_t ring_slot_bytes(const EncoderOptions& o) {
  return runtime::align_up(o.chunk_bytes + weights::kDirectAlign, weights::kDirectAlign);
}

bool host_resident(const EncoderOptions& o) { return o.streaming && o.source != WeightSource::kFile; }
bool uses_rings(const EncoderOptions& o) { return o.streaming && o.source != WeightSource::kHostPinned; }

void check_options(const EncoderOptions& o) {
  if (!o.streaming) return;
  if (o.weight_buffers < 1 || o.reader_threads < 1 || o.ring_slots < 1 || o.chunk_bytes == 0) {
    throw std::invalid_argument("qwen35_text: weight_buffers, reader_threads, ring_slots and chunk_bytes must be positive");
  }
}

int device_buffers(const EncoderOptions& o) { return o.streaming ? o.weight_buffers : 1; }

// A slice of one tensor that one reader moves with one ring fill (or one pinned DMA).
struct Chunk {
  const weights::IndexedTensor* tensor;
  std::uint64_t rel;           // offset within the tensor
  std::size_t size;
  std::byte* dst;              // device
  const std::byte* host;       // host-resident sources: the slice's source bytes
};

struct LayerPlan {
  weights::TensorMap w;
  std::vector<Chunk> chunks;   // file order; reader r takes chunks r, r + R, r + 2R, ...
};

// Host-side handshake between the compute (issuing) thread and the readers. CUDA waits bind to the event
// record current when cudaStreamWaitEvent is called, so each side waits here until the other has *recorded*
// the event it is about to wait on: a reader filling buffer b for layer k waits until layer k - buffers was
// issued (its compute-done event recorded); the compute thread waits until every reader recorded layer k's
// copy-done event.
struct StreamSync {
  struct Aborted {};
  std::mutex mu;
  std::condition_variable cv;
  int issued = 0;
  std::vector<int> enqueued;
  bool abort = false;
  std::exception_ptr error;

  explicit StreamSync(int readers) : enqueued(static_cast<std::size_t>(readers), 0) {}

  void fail(std::exception_ptr e) {
    std::lock_guard<std::mutex> lock(mu);
    if (!error) error = e;
    abort = true;
    cv.notify_all();
  }
  void stop() {
    std::lock_guard<std::mutex> lock(mu);
    abort = true;
    cv.notify_all();
  }
  void wait_issued(int n) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return abort || issued >= n; });
    if (abort) throw Aborted{};
  }
  void set_issued(int n) {
    std::lock_guard<std::mutex> lock(mu);
    issued = n;
    cv.notify_all();
  }
  void set_enqueued(std::size_t reader, int n) {
    std::lock_guard<std::mutex> lock(mu);
    enqueued[reader] = n;
    cv.notify_all();
  }
  void wait_enqueued(int n) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return abort || std::all_of(enqueued.begin(), enqueued.end(), [&](int e) { return e >= n; }); });
    if (error) std::rethrow_exception(error);
    if (abort) throw std::runtime_error("qwen35_text: weight streaming aborted");
  }
};

bool cancel_requested(const EncodeOptions& o) { return o.cancel != nullptr && o.cancel->load(); }

float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

// Submits the stream's enqueued work before this thread blocks on a host-side wait (WDDM batches submissions in the
// driver). cudaErrorNotReady is the expected answer; real errors surface at the next checked call.
void flush(cudaStream_t s) { static_cast<void>(cudaStreamQuery(s)); }

// Bytes of the device buffer holding a quantized model's bf16-rounded F32 parameters, each at an arena alignment.
std::size_t aux_bytes(const QuantTextModel* qm) {
  std::size_t total = 0;
  if (qm != nullptr) {
    for (const auto& [name, v] : qm->bf16_tensors()) total += runtime::align_up(v.size() * sizeof(std::uint16_t));
  }
  return total;
}

bool quantized(const QuantTextModel* qm) { return qm != nullptr && qm->precision() != Precision::kBF16; }

// encode()'s quantized-model scratch over t tokens, in its allocation order: the F32 parameters, the per-token INT8
// activations (widest input: down_proj's 12288) and their scales, the W4A8 decode buffer and its flag.
std::size_t quant_scratch_bytes(const QuantTextModel* qm, std::int64_t t) {
  using runtime::align_up;
  std::size_t total = aux_bytes(qm);
  if (quantized(qm)) total += align_up(static_cast<std::size_t>(t * kIntermediate)) + align_up(static_cast<std::size_t>(t) * sizeof(float));
  if (qm != nullptr && qm->max_grouped_elems() > 0) {
    total += align_up(static_cast<std::size_t>(qm->max_grouped_elems())) + align_up(sizeof(unsigned));
  }
  return total;
}

std::size_t workspace_bytes_impl(const weights::ModelIndex& index, const QuantTextModel* qm,
                                 std::span<const std::int64_t> lengths, const EncoderOptions& options);

}  // namespace

struct TextEncoder::Impl {
  Impl(const weights::ModelIndex& idx, runtime::DeviceArena& a, cudaStream_t s, EncoderOptions o, const QuantTextModel* q)
      : index(idx), arena(a), stream(s), opts(o), readers(idx, o.direct_io), qm(q) {}
  ~Impl();

  // Streaming: one per reader thread.
  struct Reader {
    cudaStream_t stream = nullptr;
    std::unique_ptr<runtime::PinnedRing> ring;  // null for kHostPinned
    std::vector<cudaEvent_t> copy_done;         // per device buffer
    cudaEvent_t end = nullptr;                  // timing: after the reader's last copy of a call
  };
  // Timing events on the compute stream (and reader 0 for the start), indexed below.
  static constexpr int kEvStart = 0, kEvEmbedded = 1, kEvLayerStart = 2, kEvLayerEnd = 2 + kNumLayers,
                       kEvEnd = 2 + 2 * kNumLayers, kNumTimingEvents = 3 + 2 * kNumLayers;

  const weights::ModelIndex& index;
  runtime::DeviceArena& arena;
  cudaStream_t stream;
  EncoderOptions opts;
  weights::ReaderCache readers;
  std::vector<Reader> rd;
  std::vector<cudaEvent_t> compute_done;  // per device buffer
  cudaEvent_t buffers_free = nullptr;     // compute stream at the start of a call: earlier users of the buffers done
  std::vector<cudaEvent_t> timing;
  std::vector<weights::HostGroup> host_layers;  // host-resident sources, once loaded
  kernels::Gemm gemm;
  std::vector<Layer> layers;
  const weights::WeightGroup* final_norm = nullptr;
  std::unique_ptr<weights::MappedFile> embed_map;
  const std::byte* embed_rows = nullptr;
  std::int64_t vocab = 0;
  std::size_t weight_bytes = 0;
  EncodeStats stats;
  void* gemm_ws = nullptr;  // valid during encode()
  // Quantized models (null for the bf16 bundle).
  const QuantTextModel* qm = nullptr;
  std::vector<weights::TensorMap> aux;  // per layer: the bf16-rounded F32 parameters, valid during encode()
  // Valid during encode() of a quantized model: the per-token INT8 activations of the current site (xq_src is the bf16
  // tensor they were made from, xq_k its width), the W4A8 decode buffer and the invalid-group-scale flag.
  std::int8_t* xq = nullptr;
  float* xs = nullptr;
  const void* xq_src = nullptr;
  std::int64_t xq_k = 0;
  std::int8_t* decode_buf = nullptr;
  unsigned* invalid = nullptr;
  const EncodeOptions* hooks = nullptr;  // the running encode()'s options (linear / decode hooks)

  // Row-count-dependent ops run per sequence, so every row is computed exactly as in the reference's
  // single-prompt forward (and as when the sequence is encoded alone): cuBLASLt picks its algorithm (tiles,
  // split-K) from M, and torch picks its RMSNorm reduction shape from the row count.
  void linear(const Batch& b, const Tensor& out, const Tensor& x, const Tensor& w) {
    b.for_each_sequence([&](std::int64_t s, std::int64_t e) {
      gemm.linear(out.slice0(s, e), x.slice0(s, e), w, nullptr, gemm_ws, kernels::Gemm::kWorkspaceBytes, stream);
    });
  }

  // Rotates and quantizes x (bf16 [T, K]) per token for the quantized linears that read it next (no-op for bf16).
  void quantize_input(const Batch& b, const Tensor& x) {
    if (!quantized(qm)) return;
    kernels::quant::rotate_quantize(x.ptr<__nv_bfloat16>(), x.stride[0], xq, x.dim(1), xs, b.tokens,
                                    static_cast<int>(x.dim(1)), stream);
    xq_src = x.data;
    xq_k = x.dim(1);
  }

  // out = x . W^T for the linear `module` as stored: bf16 per sequence, or INT8 over all tokens from the activations
  // quantize_input(x) made (grouped weights decoded first).
  void lin(const Batch& b, const Tensor& out, const Tensor& x, const weights::TensorMap& w, const std::string& module) {
    const LinearInfo* info = qm != nullptr ? qm->find_linear(module) : nullptr;
    if (info == nullptr || info->kind == LinearInfo::Kind::kBF16) {
      linear(b, out, x, w.at(module + ".weight"));
      if (hooks != nullptr && hooks->linear_hook) hooks->linear_hook(module, out, stream);
      return;
    }
    if (xq_src != x.data || xq_k != x.dim(1)) fail("internal: " + module + " input was not quantized");
    kernels::quant::Int8Operands op;
    op.x_q = xq;
    op.ldx = xq_k;
    op.x_scale = xs;
    op.m = b.tokens;
    op.n = info->n;
    op.k = info->k;
    if (info->kind == LinearInfo::Kind::kInt8) {
      op.w = w.at(module + ".weight").ptr<std::int8_t>();
      op.w_scale = w.at(module + ".weight_scale").ptr<float>();
    } else {
      kernels::quant::GroupedCodes c;
      c.packed = static_cast<const std::uint8_t*>(w.at(module + ".weight").data);
      auto* s_rel = static_cast<std::uint8_t*>(w.at(module + ".weight_s_rel").data);
      c.s_rel = s_rel;
      c.codebook = info->codebook ? w.at(module + ".weight_codebook").ptr<float>() : nullptr;
      c.n = info->n;
      c.k = info->k;
      c.bits = info->bits;
      c.group = info->group;
      if (hooks != nullptr && hooks->decode_hook) {
        hooks->decode_hook(module, s_rel, c.n * (c.k / c.group), stream);
      }
      kernels::quant::decode_grouped_int8(c, decode_buf, invalid, stream);
      op.w = decode_buf;
      op.w_scale = w.at(module + ".weight_s_channel").ptr<float>();
    }
    kernels::quant::int8_gemm(op, nullptr, out.ptr<__nv_bfloat16>(), out.stride[0], stream);
    if (hooks != nullptr && hooks->linear_hook) hooks->linear_hook(module, out, stream);
  }

  void norm(const Batch& b, const Tensor& x, const Tensor& w, const Tensor& out) {
    b.for_each_sequence([&](std::int64_t s, std::int64_t e) { kernels::rms_norm_qwen(x.slice0(s, e), w, out.slice0(s, e), stream); });
  }

  weights::TensorMap load(const weights::WeightGroup& g, void* dst, std::size_t bytes) {
    weights::TensorMap m = weights::upload_group_sync(readers, g, dst, bytes);
    // The copies ran as pageable cudaMemcpy on the legacy default stream, whose last DMA may still be in
    // flight on return, and `stream` may be non-blocking: wait for them before any kernel reads the group.
    MI_CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy));
    return m;
  }

  void gather_embeddings(const std::vector<std::vector<std::int32_t>>& seqs, const Tensor& x) {
    constexpr std::size_t kRowBytes = kHidden * sizeof(std::uint16_t);
    std::vector<std::byte> host(static_cast<std::size_t>(x.nbytes()));
    if (qm != nullptr) {
      // Quantized table: each distinct id decoded once (reads of a few KB each, spread over threads).
      std::vector<std::int64_t> ids;
      for (const auto& s : seqs) ids.insert(ids.end(), s.begin(), s.end());
      std::vector<std::int64_t> uniq = ids;
      std::sort(uniq.begin(), uniq.end());
      uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
      std::vector<std::uint16_t> rows(uniq.size() * kHidden);
      constexpr std::int64_t kRowsPerTask = 64;
      const auto tasks = static_cast<std::int64_t>((uniq.size() + kRowsPerTask - 1) / kRowsPerTask);
      quant::parallel_for(tasks, std::min(opts.reader_threads * 2, 16), [&](std::int64_t tb, std::int64_t te) {
        for (std::int64_t t = tb; t < te; ++t) {
          const auto r0 = static_cast<std::size_t>(t * kRowsPerTask);
          const std::size_t nr = std::min<std::size_t>(kRowsPerTask, uniq.size() - r0);
          quant::gather_embedding_rows_bf16(qm->model(), qm->embed_tokens(), std::span(uniq.data() + r0, nr),
                                            rows.data() + r0 * kHidden);
        }
      });
      for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto u = static_cast<std::size_t>(std::lower_bound(uniq.begin(), uniq.end(), ids[i]) - uniq.begin());
        std::memcpy(host.data() + i * kRowBytes, rows.data() + u * kHidden, kRowBytes);
      }
    } else {
      std::size_t row = 0;
      for (const auto& s : seqs) {
        for (std::int32_t id : s) {
          std::memcpy(host.data() + row++ * kRowBytes, embed_rows + static_cast<std::size_t>(id) * kRowBytes, kRowBytes);
        }
      }
    }
    MI_CUDA_CHECK(cudaMemcpyAsync(x.data, host.data(), host.size(), cudaMemcpyHostToDevice, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Qwen3_5Attention (Q:772-805): writes o_proj's output to y.
  void attention(const Layer& l, const weights::TensorMap& w, const Batch& b, const Tensor& h, const Tensor& y) {
    const std::string& p = l.prefix;
    const std::int64_t t = b.tokens;
    runtime::StackScope scope(arena);
    Tensor qg = arena.alloc_tensor(DType::kBF16, {t, kQgDim});
    Tensor k = arena.alloc_tensor(DType::kBF16, {t, kKvDim});
    Tensor v = arena.alloc_tensor(DType::kBF16, {t, kKvDim});
    quantize_input(b, h);
    lin(b, qg, h, w, p + "self_attn.q_proj");
    lin(b, k, h, w, p + "self_attn.k_proj");
    lin(b, v, h, w, p + "self_attn.v_proj");
    // q_proj output viewed per head as [q | gate].
    Tensor q_raw = strided(qg.data, DType::kBF16, {t, kNumHeads, kAttnHeadDim}, {kQgDim, 2 * kAttnHeadDim, 1});
    Tensor gate = strided(qg.ptr<std::uint16_t>() + kAttnHeadDim, DType::kBF16, {t, kNumHeads, kAttnHeadDim},
                          {kQgDim, 2 * kAttnHeadDim, 1});
    Tensor q = arena.alloc_tensor(DType::kBF16, {t, kNumHeads, kAttnHeadDim});
    Tensor k3 = k.reshaped({t, kNumKvHeads, kAttnHeadDim});
    Tensor v3 = v.reshaped({t, kNumKvHeads, kAttnHeadDim});
    const Tensor& q_norm = w.at(p + "self_attn.q_norm.weight");
    const Tensor& k_norm = w.at(p + "self_attn.k_norm.weight");
    b.for_each_sequence([&](std::int64_t s, std::int64_t e) {
      const Tensor table = b.rope.slice0(0, e - s);
      kernels::rms_norm_qwen(q_raw.slice0(s, e), q_norm, q.slice0(s, e), stream);
      kernels::rms_norm_qwen(k3.slice0(s, e), k_norm, k3.slice0(s, e), stream);
      kernels::qwen_rope_apply(q.slice0(s, e), table, q.slice0(s, e), stream);
      kernels::qwen_rope_apply(k3.slice0(s, e), table, k3.slice0(s, e), stream);
    });
    Tensor attn = arena.alloc_tensor(DType::kBF16, {t, kNumHeads, kAttnHeadDim});
    const std::size_t ws_bytes = attention::flash_fwd_workspace_bytes(t, kNumHeads);
    void* ws = arena.alloc_stack(ws_bytes);
    attention::flash_fwd(q, k3, v3, attn, b.cu_dev, b.cu_dev, b.max_len, /*causal=*/true, kAttnScale, ws, ws_bytes,
                         stream);
    kernels::sigmoid_mul(attn, gate, attn, stream);
    const Tensor attn2 = attn.reshaped({t, kNumHeads * kAttnHeadDim});
    quantize_input(b, attn2);
    lin(b, y, attn2, w, p + "self_attn.o_proj");
  }

  // Qwen3_5GatedDeltaNet (Q:490-650) without cache: writes out_proj's output to y.
  void gated_delta_net(const Layer& l, const weights::TensorMap& w, const Batch& b, const Tensor& h, const Tensor& y) {
    const std::string& p = l.prefix;
    const std::int64_t t = b.tokens;
    runtime::StackScope scope(arena);
    Tensor qkv = arena.alloc_tensor(DType::kBF16, {t, gdn::kConvDim});
    Tensor z = arena.alloc_tensor(DType::kBF16, {t, gdn::kValueDim});
    Tensor bb = arena.alloc_tensor(DType::kBF16, {t, gdn::kNumVHeads});
    Tensor aa = arena.alloc_tensor(DType::kBF16, {t, gdn::kNumVHeads});
    quantize_input(b, h);
    lin(b, qkv, h, w, p + "linear_attn.in_proj_qkv");
    lin(b, z, h, w, p + "linear_attn.in_proj_z");
    lin(b, bb, h, w, p + "linear_attn.in_proj_b");
    lin(b, aa, h, w, p + "linear_attn.in_proj_a");
    Tensor conv = arena.alloc_tensor(DType::kBF16, {t, gdn::kConvDim});
    gdn::causal_conv1d_silu(qkv, w.at(p + "linear_attn.conv1d.weight").reshaped({gdn::kConvDim, gdn::kConvKernel}),
                            b.cu_dev, conv, stream);
    Tensor g = arena.alloc_tensor(DType::kF32, {t, gdn::kNumVHeads});
    Tensor beta = arena.alloc_tensor(DType::kBF16, {t, gdn::kNumVHeads});
    gdn::gates(aa, bb, w.at(p + "linear_attn.A_log"), w.at(p + "linear_attn.dt_bias"), g, beta, stream);
    Tensor core = arena.alloc_tensor(DType::kBF16, {t, gdn::kNumVHeads, gdn::kHeadDim});
    const std::size_t ws_bytes = gdn::chunk_gated_delta_rule_workspace_bytes(t);
    void* ws = arena.alloc_stack(ws_bytes);
    gdn::chunk_gated_delta_rule(conv, g, beta, b.cu_dev, b.max_len, core, nullptr, ws, ws_bytes, stream);
    // RMSNormGated per 128-d head; z is the gate.
    const Tensor z3 = z.reshaped({t, gdn::kNumVHeads, gdn::kHeadDim});
    const Tensor& norm_w = w.at(p + "linear_attn.norm.weight");
    b.for_each_sequence([&](std::int64_t s, std::int64_t e) {
      kernels::rms_norm_gated_qwen(core.slice0(s, e), z3.slice0(s, e), norm_w, core.slice0(s, e), stream);
    });
    const Tensor core2 = core.reshaped({t, gdn::kValueDim});
    quantize_input(b, core2);
    lin(b, y, core2, w, p + "linear_attn.out_proj");
  }

  // Qwen3_5DecoderLayer.forward (Q:871-900) on the residual stream x in place; h and y are [T, 4096] scratch.
  void run_layer(const Layer& l, const weights::TensorMap& w, const Batch& b, const Tensor& x, const Tensor& h,
                 const Tensor& y) {
    const std::string& p = l.prefix;
    norm(b, x, w.at(p + "input_layernorm.weight"), h);
    if (l.full_attention) {
      attention(l, w, b, h, y);
    } else {
      gated_delta_net(l, w, b, h, y);
    }
    kernels::residual_add(x, y, x, stream);
    xq_src = nullptr;  // h is about to hold the next site's input
    norm(b, x, w.at(p + "post_attention_layernorm.weight"), h);
    {
      runtime::StackScope scope(arena);
      Tensor gate = arena.alloc_tensor(DType::kBF16, {b.tokens, kIntermediate});
      Tensor up = arena.alloc_tensor(DType::kBF16, {b.tokens, kIntermediate});
      quantize_input(b, h);
      lin(b, gate, h, w, p + "mlp.gate_proj");
      lin(b, up, h, w, p + "mlp.up_proj");
      kernels::silu_mul(gate, up, gate, stream);
      quantize_input(b, gate);
      lin(b, y, gate, w, p + "mlp.down_proj");
    }
    kernels::residual_add(x, y, x, stream);
    xq_src = nullptr;
  }

  void init_layers();
  void alloc_quant_scratch(std::int64_t tokens);
  void release_quant_scratch();
  void check_decode();
  void create_streaming_resources();
  void load_host();
  std::vector<LayerPlan> plan_layers(const std::vector<void*>& bufs) const;
  void reader_loop(std::size_t r, const std::vector<LayerPlan>& plans, StreamSync& sync,
                   std::vector<std::uint64_t>& layer_bytes, std::uint64_t& chunks, const ReaderHook& hook);
  void run_layers_sync(const Batch& b, void* weight_buf, const Tensor& x, const Tensor& h, const Tensor& y,
                       const EncodeOptions& options);
  void run_layers_streaming(const std::vector<std::vector<std::int32_t>>& sequences, const Batch& b,
                            const std::vector<void*>& bufs, const Tensor& x, const Tensor& h, const Tensor& y,
                            const EncodeOptions& options);
};

TextEncoder::Impl::~Impl() {
  // Destructor: errors are not actionable. Rings first (they wait for their own in-flight copies).
  for (Reader& r : rd) {
    r.ring.reset();
    if (r.stream != nullptr) cudaStreamSynchronize(r.stream);
    for (cudaEvent_t e : r.copy_done) cudaEventDestroy(e);
    if (r.end != nullptr) cudaEventDestroy(r.end);
    if (r.stream != nullptr) cudaStreamDestroy(r.stream);
  }
  for (cudaEvent_t e : compute_done) cudaEventDestroy(e);
  if (buffers_free != nullptr) cudaEventDestroy(buffers_free);
  for (cudaEvent_t e : timing) cudaEventDestroy(e);
}

void TextEncoder::Impl::create_streaming_resources() {
  rd.resize(static_cast<std::size_t>(opts.reader_threads));
  for (Reader& r : rd) {
    MI_CUDA_CHECK(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
    for (int b = 0; b < opts.weight_buffers; ++b) {
      MI_CUDA_CHECK(cudaEventCreateWithFlags(&r.copy_done.emplace_back(), cudaEventDisableTiming));
    }
    MI_CUDA_CHECK(cudaEventCreate(&r.end));
    if (uses_rings(opts)) {
      r.ring = std::make_unique<runtime::PinnedRing>(r.stream, static_cast<std::size_t>(opts.ring_slots), ring_slot_bytes(opts));
    }
  }
  for (int b = 0; b < opts.weight_buffers; ++b) {
    MI_CUDA_CHECK(cudaEventCreateWithFlags(&compute_done.emplace_back(), cudaEventDisableTiming));
  }
  MI_CUDA_CHECK(cudaEventCreateWithFlags(&buffers_free, cudaEventDisableTiming));
  for (int i = 0; i < kNumTimingEvents; ++i) MI_CUDA_CHECK(cudaEventCreate(&timing.emplace_back()));
}

// Reads every layer group into host memory, groups spread over the reader threads.
void TextEncoder::Impl::load_host() {
  if (!host_layers.empty()) return;
  const weights::HostMemory mem =
      opts.source == WeightSource::kHostPinned ? weights::HostMemory::kPinned : weights::HostMemory::kPlain;
  std::vector<weights::HostGroup> loaded(kNumLayers);
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(opts.reader_threads));
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  {
    std::vector<std::jthread> threads;
    for (int t = 0; t < opts.reader_threads; ++t) {
      threads.emplace_back([&, t] {
        try {
          MI_CUDA_CHECK(cudaSetDevice(device));
          for (int i = t; i < kNumLayers; i += opts.reader_threads) {
            loaded[static_cast<std::size_t>(i)] = weights::load_group_to_host(readers, *layers[static_cast<std::size_t>(i)].group, mem);
          }
        } catch (...) {
          errors[static_cast<std::size_t>(t)] = std::current_exception();
        }
      });
    }
  }
  for (const auto& e : errors) {
    if (e) std::rethrow_exception(e);
  }
  host_layers = std::move(loaded);
}

std::vector<LayerPlan> TextEncoder::Impl::plan_layers(const std::vector<void*>& bufs) const {
  std::vector<LayerPlan> plans(kNumLayers);
  std::vector<weights::UploadPiece> pieces;
  for (int i = 0; i < kNumLayers; ++i) {
    LayerPlan& p = plans[static_cast<std::size_t>(i)];
    const weights::WeightGroup& g = *layers[static_cast<std::size_t>(i)].group;
    p.w = weights::plan_group_upload(g, bufs[static_cast<std::size_t>(i) % bufs.size()], weight_bytes, pieces);
    if (!aux.empty()) p.w.insert(aux[static_cast<std::size_t>(i)].begin(), aux[static_cast<std::size_t>(i)].end());
    for (const weights::UploadPiece& piece : pieces) {
      const std::byte* host = nullptr;
      if (!host_layers.empty()) {
        host = static_cast<const std::byte*>(host_layers[static_cast<std::size_t>(i)].tensors.at(piece.tensor->name).data);
      }
      for (std::uint64_t rel = 0; rel < piece.tensor->nbytes; rel += opts.chunk_bytes) {
        const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(opts.chunk_bytes, piece.tensor->nbytes - rel));
        p.chunks.push_back({piece.tensor, rel, n, piece.dst + rel, host == nullptr ? nullptr : host + rel});
      }
    }
  }
  return plans;
}

void TextEncoder::Impl::reader_loop(std::size_t r, const std::vector<LayerPlan>& plans, StreamSync& sync,
                                    std::vector<std::uint64_t>& layer_bytes, std::uint64_t& chunks,
                                    const ReaderHook& hook) {
  Reader& me = rd[r];
  const int nbuf = opts.weight_buffers;
  const std::size_t stride = rd.size();
  const runtime::RingStats start = me.ring ? me.ring->stats() : runtime::RingStats{};
  // The weight buffers were just carved out of the arena stack: work queued earlier on the compute stream may still
  // use that memory, so no copy may land before it is done.
  MI_CUDA_CHECK(cudaStreamWaitEvent(me.stream, buffers_free, 0));
  for (int k = 0; k < kNumLayers; ++k) {
    const auto b = static_cast<std::size_t>(k % nbuf);
    if (hook) hook(static_cast<int>(r), k);
    if (k >= nbuf) {
      flush(me.stream);
      sync.wait_issued(k - nbuf + 1);
      MI_CUDA_CHECK(cudaStreamWaitEvent(me.stream, compute_done[b], 0));
    }
    const std::uint64_t before = me.ring ? me.ring->stats().bytes : 0;
    std::uint64_t pinned = 0;
    const std::vector<Chunk>& cs = plans[static_cast<std::size_t>(k)].chunks;
    for (std::size_t i = r; i < cs.size(); i += stride) {
      const Chunk& c = cs[i];
      switch (opts.source) {
        case WeightSource::kFile:
          me.ring->copy_range(readers.get(c.tensor->file), c.tensor->offset + c.rel, c.size, c.dst);
          break;
        case WeightSource::kHostPageable:
          me.ring->copy_host(c.host, c.size, c.dst);
          break;
        case WeightSource::kHostPinned:
          MI_CUDA_CHECK(cudaMemcpyAsync(c.dst, c.host, c.size, cudaMemcpyHostToDevice, me.stream));
          pinned += c.size;
          break;
      }
    }
    layer_bytes[static_cast<std::size_t>(k)] = me.ring ? me.ring->stats().bytes - before : pinned;
    MI_CUDA_CHECK(cudaEventRecord(me.copy_done[b], me.stream));
    flush(me.stream);
    sync.set_enqueued(r, k + 1);
  }
  if (hook) hook(static_cast<int>(r), kNumLayers);
  MI_CUDA_CHECK(cudaEventRecord(me.end, me.stream));
  chunks = me.ring ? me.ring->stats().chunks - start.chunks : 0;
}

// P2 path: load layer k, run it, synchronize, move on.
void TextEncoder::Impl::run_layers_sync(const Batch& b, void* weight_buf, const Tensor& x, const Tensor& h,
                                        const Tensor& y, const EncodeOptions& options) {
  for (int i = 0; i < kNumLayers; ++i) {
    // The previous layer ended with a synchronize: nothing is queued, so stopping here needs no cleanup.
    if (cancel_requested(options)) throw Cancelled();
    const Layer& l = layers[static_cast<std::size_t>(i)];
    auto t0 = Clock::now();
    weights::TensorMap w = load(*l.group, weight_buf, weight_bytes);
    if (!aux.empty()) w.insert(aux[static_cast<std::size_t>(i)].begin(), aux[static_cast<std::size_t>(i)].end());
    stats.weight_load_s += seconds_since(t0);
    stats.layer_bytes[static_cast<std::size_t>(i)] = l.group->total_bytes;
    t0 = Clock::now();
    run_layer(l, w, b, x, h, y);
    if (options.layer_hook) options.layer_hook(i, x, stream);
    // Also guards the weight buffer: the next layer's upload overwrites it.
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    stats.compute_s += seconds_since(t0);
  }
}

// Embeds (overlapping the first reads), then runs the layers as their weights arrive. On return every reader
// thread has finished; on success all copies are enqueued before work the compute stream waits on.
void TextEncoder::Impl::run_layers_streaming(const std::vector<std::vector<std::int32_t>>& sequences, const Batch& b,
                                             const std::vector<void*>& bufs, const Tensor& x, const Tensor& h,
                                             const Tensor& y, const EncodeOptions& options) {
  const std::vector<LayerPlan> plans = plan_layers(bufs);
  const std::size_t nr = rd.size();
  std::vector<std::vector<std::uint64_t>> layer_bytes(nr, std::vector<std::uint64_t>(kNumLayers, 0));
  std::vector<std::uint64_t> chunks(nr, 0);
  StreamSync sync(static_cast<int>(nr));
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  // Recorded before the readers start, so their waits bind to it (see reader_loop).
  MI_CUDA_CHECK(cudaEventRecord(buffers_free, stream));
  MI_CUDA_CHECK(cudaEventRecord(timing[kEvStart], rd[0].stream));
  std::vector<std::thread> threads;
  const auto join = [&] {
    for (std::thread& t : threads) t.join();
    threads.clear();
  };
  try {
    for (std::size_t r = 0; r < nr; ++r) {
      threads.emplace_back([&, r] {
        try {
          MI_CUDA_CHECK(cudaSetDevice(device));
          reader_loop(r, plans, sync, layer_bytes[r], chunks[r], options.reader_hook);
        } catch (const StreamSync::Aborted&) {
        } catch (...) {
          sync.fail(std::current_exception());
        }
      });
    }
    auto t0 = Clock::now();
    gather_embeddings(sequences, x);
    stats.embed_s = seconds_since(t0);
    MI_CUDA_CHECK(cudaEventRecord(timing[kEvEmbedded], stream));

    for (int k = 0; k < kNumLayers; ++k) {
      const auto buf = static_cast<std::size_t>(k % opts.weight_buffers);
      if (options.cancel != nullptr) {
        // Keep at most one layer queued ahead of the one the GPU runs, so a cancel takes effect within a layer.
        if (k >= 2) MI_CUDA_CHECK(cudaEventSynchronize(timing[static_cast<std::size_t>(kEvLayerEnd + k - 2)]));
        if (cancel_requested(options)) throw Cancelled();
      }
      flush(stream);
      sync.wait_enqueued(k + 1);
      for (const Reader& r : rd) MI_CUDA_CHECK(cudaStreamWaitEvent(stream, r.copy_done[buf], 0));
      MI_CUDA_CHECK(cudaEventRecord(timing[static_cast<std::size_t>(kEvLayerStart + k)], stream));
      run_layer(layers[static_cast<std::size_t>(k)], plans[static_cast<std::size_t>(k)].w, b, x, h, y);
      if (options.layer_hook) options.layer_hook(k, x, stream);
      MI_CUDA_CHECK(cudaEventRecord(timing[static_cast<std::size_t>(kEvLayerEnd + k)], stream));
      MI_CUDA_CHECK(cudaEventRecord(compute_done[buf], stream));
      sync.set_issued(k + 1);
    }
  } catch (...) {
    sync.stop();
    join();
    // Copies may still target the weight buffers the caller is about to roll back.
    for (const Reader& r : rd) cudaStreamSynchronize(r.stream);
    cudaStreamSynchronize(stream);
    throw;
  }
  join();
  // A reader can still fail after the compute thread's last wait (its final timing record): not lost.
  if (sync.error) {
    for (const Reader& r : rd) cudaStreamSynchronize(r.stream);
    cudaStreamSynchronize(stream);
    std::rethrow_exception(sync.error);
  }
  for (std::size_t r = 0; r < nr; ++r) {
    for (int k = 0; k < kNumLayers; ++k) stats.layer_bytes[static_cast<std::size_t>(k)] += layer_bytes[r][static_cast<std::size_t>(k)];
    stats.read_chunks += chunks[r];
  }
}

std::uint64_t numerics_id() { return numerics_id(Precision::kBF16); }

std::uint64_t numerics_id(Precision precision) {
  // bf16 keeps the code it had before quantized encoders existed, so cached bf16 embeddings stay valid.
  const std::uint64_t code = precision == Precision::kBF16 ? 1 : precision == Precision::kInt8 ? 2 : 3;
  int device = 0, major = 0, minor = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  // Quantized precisions also depend on the INT8 kernels (rotate+quantize, GEMM epilogue) and, for W4A8, the decoder;
  // bf16 hashes exactly the fields it always did.
  const std::uint64_t int8_version = precision == Precision::kBF16 ? 0 : kernels::quant::kInt8NumericsVersion;
  const std::uint64_t decode_version = precision == Precision::kW4A8 ? kernels::quant::kGroupedDecodeVersion : 0;
  const std::uint64_t fields[] = {kNumericsVersion, code, static_cast<std::uint64_t>(major * 10 + minor),
                                  static_cast<std::uint64_t>(cublasLtGetVersion()), int8_version, decode_version};
  return xxh64(fields, precision == Precision::kBF16 ? 4 * sizeof(std::uint64_t) : sizeof(fields));
}

TextEncoder::TextEncoder(const weights::ModelIndex& index, runtime::DeviceArena& arena, cudaStream_t stream,
                         EncoderOptions options)
    : impl_(std::make_unique<Impl>(index, arena, stream, options, nullptr)) {
  check_options(options);
  impl_->init_layers();
  Impl& m = *impl_;
  const weights::IndexedTensor& embed = single_tensor(index, "embed_tokens");
  if (embed.dtype != weights::DType::kBF16 || embed.shape.size() != 2 || embed.shape[1] != kHidden || embed.shape[0] < 1) {
    fail("tensor " + embed.name + " must be BF16 [vocab, 4096]");
  }
  m.vocab = embed.shape[0];
  m.embed_map = std::make_unique<weights::MappedFile>(index.file_path(embed.file));
  if (embed.offset > m.embed_map->size() || embed.nbytes > m.embed_map->size() - embed.offset ||
      embed.nbytes != static_cast<std::uint64_t>(m.vocab) * kHidden * sizeof(std::uint16_t)) {
    fail("tensor " + embed.name + " lies outside its file");
  }
  m.embed_rows = m.embed_map->data() + embed.offset;
  if (options.streaming) m.create_streaming_resources();
}

TextEncoder::TextEncoder(const QuantTextModel& model, runtime::DeviceArena& arena, cudaStream_t stream,
                         EncoderOptions options)
    : impl_(std::make_unique<Impl>(model.index(), arena, stream, options, &model)) {
  check_options(options);
  impl_->init_layers();
  const quant::QuantLinear& embed = model.embed_tokens();
  if (embed.k != kHidden) fail("embed_tokens must have 4096 columns, got " + std::to_string(embed.k));
  impl_->vocab = embed.n;
  if (options.streaming) impl_->create_streaming_resources();
}

// Allocation order and sizes as quant_scratch_bytes().
void TextEncoder::Impl::alloc_quant_scratch(std::int64_t tokens) {
  if (qm == nullptr) return;
  aux.assign(kNumLayers, {});
  if (!qm->bf16_tensors().empty()) {
    auto* dst = static_cast<std::byte*>(arena.alloc_stack(aux_bytes(qm)));
    const std::string prefix = "language_model.layers.";
    for (const auto& [name, v] : qm->bf16_tensors()) {
      const std::size_t bytes = v.size() * sizeof(std::uint16_t);
      MI_CUDA_CHECK(cudaMemcpyAsync(dst, v.data(), bytes, cudaMemcpyHostToDevice, stream));
      // "language_model.layers.<N>.<...>", as QuantTextModel routed it.
      int layer = -1;
      const char* digits = name.data() + prefix.size();
      const auto r = std::from_chars(digits, name.data() + name.size(), layer);
      if (!name.starts_with(prefix) || r.ec != std::errc() || r.ptr == digits || *r.ptr != '.' || layer < 0 ||
          layer >= kNumLayers) {
        fail("tensor " + name + " belongs to no layer");
      }
      const std::vector<std::int64_t>& shape = qm->bf16_shape(name);
      aux[static_cast<std::size_t>(layer)].emplace(name, Tensor::view(dst, DType::kBF16, Device::kCUDA, shape.data(),
                                                                       static_cast<int>(shape.size())));
      dst += runtime::align_up(bytes);
    }
  }
  if (quantized(qm)) {
    xq = static_cast<std::int8_t*>(arena.alloc_stack(static_cast<std::size_t>(tokens * kIntermediate)));
    xs = static_cast<float*>(arena.alloc_stack(static_cast<std::size_t>(tokens) * sizeof(float)));
  }
  if (qm->max_grouped_elems() > 0) {
    decode_buf = static_cast<std::int8_t*>(arena.alloc_stack(static_cast<std::size_t>(qm->max_grouped_elems())));
    invalid = static_cast<unsigned*>(arena.alloc_stack(sizeof(unsigned)));
    MI_CUDA_CHECK(cudaMemsetAsync(invalid, 0, sizeof(unsigned), stream));
  }
}

void TextEncoder::Impl::release_quant_scratch() {
  aux.clear();
  xq = nullptr;
  xs = nullptr;
  xq_src = nullptr;
  xq_k = 0;
  decode_buf = nullptr;
  invalid = nullptr;
}

// After the stream is synchronized: a NaN or negative W4A8 group scale decoded to zeros somewhere.
void TextEncoder::Impl::check_decode() {
  if (invalid == nullptr) return;
  unsigned flag = 0;
  MI_CUDA_CHECK(cudaMemcpy(&flag, invalid, sizeof(flag), cudaMemcpyDeviceToHost));
  if (flag != 0) throw quant::QuantError("qwen35_text: a W4A8 group scale of the file is NaN or negative");
}

void TextEncoder::Impl::init_layers() {
  for (int i = 0; i < kNumLayers; ++i) {
    layers.push_back(validate_layer(index, i, qm));
    weight_bytes = std::max(weight_bytes, weights::group_device_bytes(*layers.back().group));
  }
  if (index.find_group(Component::kTextEncoder, "layers." + std::to_string(kNumLayers)) != nullptr) {
    fail("bundle has more than " + std::to_string(kNumLayers) + " text encoder layers");
  }
  check_tensor(single_tensor(index, "final_norm"), {kHidden});
  final_norm = index.find_group(Component::kTextEncoder, "final_norm");
}

TextEncoder::~TextEncoder() = default;

// Mirrors encode(): outputs, the fixed scratch, then the largest of one layer's token-mixer and MLP scratch.
// Keep in sync with encode(), attention(), gated_delta_net() and run_layer() (the test compares it with the
// arena's high-water mark).
std::size_t TextEncoder::workspace_bytes(const weights::ModelIndex& index, std::span<const std::int64_t> lengths) {
  return workspace_bytes(index, lengths, EncoderOptions{});
}

std::size_t TextEncoder::workspace_bytes(const weights::ModelIndex& index, std::span<const std::int64_t> lengths,
                                         const EncoderOptions& options) {
  return workspace_bytes_impl(index, nullptr, lengths, options);
}

std::size_t TextEncoder::workspace_bytes(const QuantTextModel& model, std::span<const std::int64_t> lengths,
                                         const EncoderOptions& options) {
  return workspace_bytes_impl(model.index(), &model, lengths, options);
}

namespace {

std::size_t workspace_bytes_impl(const weights::ModelIndex& index, const QuantTextModel* qm,
                                 std::span<const std::int64_t> lengths, const EncoderOptions& options) {
  using runtime::align_up;
  check_options(options);
  std::int64_t t = 0, max_len = 0;
  for (const std::int64_t s : lengths) {
    t += s;
    max_len = std::max(max_len, s);
  }
  std::size_t weight_bytes = 0;
  for (int i = 0; i < kNumLayers; ++i) {
    weight_bytes = std::max(weight_bytes, weights::group_device_bytes(*validate_layer(index, i, qm).group));
  }
  single_tensor(index, "final_norm");
  const std::size_t norm_bytes = weights::group_device_bytes(*index.find_group(Component::kTextEncoder, "final_norm"));
  const auto rows = [&](std::int64_t cols, std::size_t elem) { return align_up(static_cast<std::size_t>(t * cols) * elem); };
  const std::size_t fixed = rows(kHidden, 2) +  // outputs
                            static_cast<std::size_t>(device_buffers(options)) * align_up(weight_bytes) +
                            align_up(kernels::Gemm::kWorkspaceBytes) + align_up(norm_bytes) +
                            align_up((lengths.size() + 1) * sizeof(std::int32_t)) +
                            align_up(static_cast<std::size_t>(max_len * kernels::kQwenRopeDim) * 2) +
                            3 * rows(kHidden, 2) +  // x, h, y
                            quant_scratch_bytes(qm, t);
  const std::size_t attn = rows(kQgDim, 2) + 2 * rows(kKvDim, 2) + 2 * rows(kNumHeads * kAttnHeadDim, 2) +
                           align_up(attention::flash_fwd_workspace_bytes(t, kNumHeads));
  const std::size_t delta_net = 2 * rows(gdn::kConvDim, 2) + rows(gdn::kValueDim, 2) + 3 * rows(gdn::kNumVHeads, 2) +
                          rows(gdn::kNumVHeads, 4) + rows(gdn::kNumVHeads * gdn::kHeadDim, 2) +
                          align_up(gdn::chunk_gated_delta_rule_workspace_bytes(t));
  const std::size_t mlp = 2 * rows(kIntermediate, 2);
  return fixed + std::max({attn, delta_net, mlp});
}

std::size_t host_bytes_impl(const weights::ModelIndex& index, const QuantTextModel* qm, const EncoderOptions& options) {
  check_options(options);
  std::size_t total = 0;
  if (uses_rings(options)) {
    total += static_cast<std::size_t>(options.reader_threads) * static_cast<std::size_t>(options.ring_slots) *
             ring_slot_bytes(options);
  }
  if (host_resident(options)) {
    for (int i = 0; i < kNumLayers; ++i) total += weights::host_group_bytes(*validate_layer(index, i, qm).group);
  }
  return total;
}

}  // namespace

std::size_t TextEncoder::host_bytes(const weights::ModelIndex& index, const EncoderOptions& options) {
  return host_bytes_impl(index, nullptr, options);
}

std::size_t TextEncoder::host_bytes(const QuantTextModel& model, const EncoderOptions& options) {
  return host_bytes_impl(model.index(), &model, options);
}

void TextEncoder::preload_host() {
  if (host_resident(impl_->opts)) impl_->load_host();
}

void TextEncoder::release_host() { impl_->host_layers.clear(); }

std::size_t TextEncoder::host_resident_bytes() const {
  std::size_t total = 0;
  for (const auto& g : impl_->host_layers) total += g.storage_bytes;
  return total;
}

const EncoderOptions& TextEncoder::options() const { return impl_->opts; }
const EncodeStats& TextEncoder::last_stats() const { return impl_->stats; }
std::size_t TextEncoder::weight_buffer_bytes() const { return impl_->weight_bytes; }
std::int64_t TextEncoder::vocab_size() const { return impl_->vocab; }
Precision TextEncoder::precision() const { return impl_->qm != nullptr ? impl_->qm->precision() : Precision::kBF16; }
std::uint64_t TextEncoder::numerics() const { return numerics_id(precision()); }

std::vector<Tensor> TextEncoder::encode(const std::vector<std::vector<std::int32_t>>& sequences,
                                        const EncodeOptions& options) {
  Impl& m = *impl_;
  const auto t_start = Clock::now();
  if (sequences.empty()) throw std::invalid_argument("qwen35_text::encode: no sequences");
  Batch b;
  b.cu.push_back(0);
  for (const auto& s : sequences) {
    if (s.empty()) throw std::invalid_argument("qwen35_text::encode: empty sequence");
    for (std::int32_t id : s) {
      if (id < 0 || id >= m.vocab) {
        throw std::invalid_argument("qwen35_text::encode: token id " + std::to_string(id) + " outside the vocabulary");
      }
    }
    b.cu.push_back(b.cu.back() + static_cast<std::int64_t>(s.size()));
    b.max_len = std::max(b.max_len, static_cast<int>(std::min<std::size_t>(s.size(), std::numeric_limits<int>::max())));
  }
  b.tokens = b.cu.back();
  if (b.tokens > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("qwen35_text::encode: too many tokens");
  const auto n = static_cast<std::int64_t>(sequences.size());
  m.stats = EncodeStats{};
  m.stats.tokens = b.tokens;
  m.stats.layer_bytes.assign(kNumLayers, 0);
  const bool streaming = m.opts.streaming;
  if (host_resident(m.opts) && m.host_layers.empty()) {
    const auto t0 = Clock::now();
    m.load_host();
    m.stats.host_load_s = seconds_since(t0);
  }

  // Outputs first, so they survive the scratch rollback below. On failure nothing stays allocated.
  const std::size_t mark = m.arena.mark();
  Tensor out = m.arena.alloc_tensor(DType::kBF16, {b.tokens, kHidden});
  m.hooks = &options;
  try {
    runtime::StackScope scratch(m.arena);
    std::vector<void*> weight_bufs;
    for (int i = 0; i < device_buffers(m.opts); ++i) weight_bufs.push_back(m.arena.alloc_stack(m.weight_bytes));
    m.gemm_ws = m.arena.alloc_stack(kernels::Gemm::kWorkspaceBytes);
    const std::size_t norm_bytes = weights::group_device_bytes(*m.final_norm);
    void* norm_buf = m.arena.alloc_stack(norm_bytes);

    b.cu_dev = m.arena.alloc_tensor(DType::kI32, {n + 1});
    std::vector<std::int32_t> cu32(b.cu.begin(), b.cu.end());
    MI_CUDA_CHECK(cudaMemcpyAsync(b.cu_dev.data, cu32.data(), cu32.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                  m.stream));
    b.rope = m.arena.alloc_tensor(DType::kBF16, {b.max_len, kernels::kQwenRopeDim});
    kernels::qwen_rope_table(b.rope, m.stream);

    Tensor x = m.arena.alloc_tensor(DType::kBF16, {b.tokens, kHidden});  // residual stream
    Tensor h = m.arena.alloc_tensor(DType::kBF16, {b.tokens, kHidden});
    Tensor y = m.arena.alloc_tensor(DType::kBF16, {b.tokens, kHidden});
    m.alloc_quant_scratch(b.tokens);

    // The final norm (16 KiB) is loaded up front in both modes.
    auto t0 = Clock::now();
    const weights::TensorMap fw = m.load(*m.final_norm, norm_buf, norm_bytes);
    const double norm_load_s = seconds_since(t0);

    if (streaming) {
      m.run_layers_streaming(sequences, b, weight_bufs, x, h, y, options);
    } else {
      m.stats.weight_load_s += norm_load_s;
      t0 = Clock::now();
      m.gather_embeddings(sequences, x);
      m.stats.embed_s = seconds_since(t0);
      m.run_layers_sync(b, weight_bufs.front(), x, h, y, options);
    }

    t0 = Clock::now();
    m.norm(b, x, fw.begin()->second, out);
    if (streaming) MI_CUDA_CHECK(cudaEventRecord(m.timing[Impl::kEvEnd], m.stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(m.stream));
    if (streaming) {
      // GPU timeline: compute = layer bodies + final norm; wait = gaps before each layer (weights not there yet);
      // loading = first read start to the last reader's last copy.
      const auto ev = [&](int i) { return m.timing[static_cast<std::size_t>(i)]; };
      double compute_ms = elapsed_ms(ev(Impl::kEvLayerEnd + kNumLayers - 1), ev(Impl::kEvEnd)), wait_ms = 0, load_ms = 0;
      for (int k = 0; k < kNumLayers; ++k) {
        compute_ms += elapsed_ms(ev(Impl::kEvLayerStart + k), ev(Impl::kEvLayerEnd + k));
        wait_ms += elapsed_ms(ev(k == 0 ? Impl::kEvEmbedded : Impl::kEvLayerEnd + k - 1), ev(Impl::kEvLayerStart + k));
      }
      for (const auto& r : m.rd) {
        // The compute stream only waited for the readers' copies, not for their later end records.
        MI_CUDA_CHECK(cudaEventSynchronize(r.end));
        load_ms = std::max<double>(load_ms, elapsed_ms(ev(Impl::kEvStart), r.end));
      }
      m.stats.compute_s = compute_ms / 1e3;
      m.stats.weight_wait_s = wait_ms / 1e3;
      m.stats.weight_load_s = load_ms / 1e3;
    } else {
      m.stats.compute_s += seconds_since(t0);
    }
    for (const std::uint64_t v : m.stats.layer_bytes) m.stats.weight_bytes += v;
    m.check_decode();
    m.release_quant_scratch();
    m.gemm_ws = nullptr;
    m.hooks = nullptr;
  } catch (...) {
    m.hooks = nullptr;
    m.release_quant_scratch();
    m.gemm_ws = nullptr;
    m.arena.rollback(mark);
    throw;
  }
  m.stats.total_s = seconds_since(t_start);

  std::vector<Tensor> result;
  b.for_each_sequence([&](std::int64_t s, std::int64_t e) { result.push_back(out.slice0(s, e)); });
  return result;
}

std::vector<std::vector<std::uint16_t>> TextEncoder::encode_to_host(const std::vector<std::vector<std::int32_t>>& sequences,
                                                                    const EncodeOptions& options) {
  runtime::StackScope scope(impl_->arena);
  const std::vector<Tensor> dev = encode(sequences, options);
  std::vector<std::vector<std::uint16_t>> host;
  for (const Tensor& t : dev) {
    host.emplace_back(static_cast<std::size_t>(t.numel()));
    MI_CUDA_CHECK(cudaMemcpyAsync(host.back().data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost, impl_->stream));
  }
  MI_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  return host;
}

}  // namespace mi::qwen35_text
