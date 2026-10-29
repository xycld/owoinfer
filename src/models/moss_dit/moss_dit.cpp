#include "models/moss_dit/moss_dit.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <map>
#include <utility>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "models/moss_dit/dit_kernels.h"
#include "models/moss_dit/nvfp4_support.h"
#include "runtime/pinned_ring.h"
#include "weights/upload.h"

namespace mi::moss_dit {
namespace {

using runtime::align_up;
using weights::Component;
using weights::TensorMap;
using weights::WeightGroup;

// SDPA's default scale 1/sqrt(head_dim), computed in double and rounded once as torch does.
const float kAttnScale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));
constexpr std::int64_t kQkvDim = 3 * kDim;
constexpr std::int64_t kUpGateDim = 2 * kFfnDim;
// The big activation buffer holds [qkv | attn] during attention and [up_gate | ffn] during the FFN (op-by-op and the
// text refiner); the fused block stores up_gate only for runs whose ffn_up_gate goes op by op (fused_big_cols).
constexpr std::int64_t kBigCols = std::max<std::int64_t>(kQkvDim + kDim, kUpGateDim + kFfnDim);
static_assert(kBigCols == kOpByOpBigCols);
static_assert(kFusedBigCols == kQkvDim + kDim && kFusedBigCols >= kFfnDim && kFusedBigCols >= kLatentChannels);

std::int64_t big_cols(BlockPath path, std::int64_t hw, std::span<const std::int64_t> text_lengths,
                      const FusedRouteSnapshot& route) {
  return path == BlockPath::kOpByOp ? kBigCols : fused_big_cols(hw, text_lengths, route);
}

[[noreturn]] void fail(const std::string& msg) { throw std::invalid_argument("mi::moss_dit: " + msg); }

// Never null (throws instead); a pointer so call sites with temporary names do not bind a reference.
const WeightGroup* find_group(const weights::ModelIndex& index, const std::string& name) {
  const WeightGroup* g = index.find_group(Component::kTransformer, name);
  if (g == nullptr) throw std::runtime_error("mi::moss_dit: transformer group '" + name + "' missing from the index");
  return g;
}

using detail::fuse;
using detail::take;
using detail::with_front;

// [rows, heads, head_dim] view of columns [col, col + kDim) of a row-major [rows, ld] bf16 buffer.
Tensor heads_view(const Tensor& buf, std::int64_t col) {
  Tensor t = Tensor::view(static_cast<char*>(buf.data) + static_cast<std::size_t>(col) * sizeof(std::uint16_t),
                          DType::kBF16, Device::kCUDA, {buf.dim(0), kHeads, kHeadDim});
  t.stride[0] = buf.stride[0];
  return t;
}

Tensor vec(const Tensor& flat, std::int64_t index) {
  return Tensor::view(static_cast<char*>(flat.data) + static_cast<std::size_t>(index * kDim) * sizeof(std::uint16_t),
                      DType::kBF16, Device::kCUDA, {kDim});
}

std::uint16_t bf16_bits(float v) {  // round to nearest even, as torch .to(bfloat16) (finite inputs)
  std::uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}

// Stack allocations of one call, in order; workspace queries replay the same sequence.
class StackPlan {
 public:
  explicit StackPlan(runtime::DeviceArena* arena) : arena_(arena) {}
  void* bytes(std::size_t n) {
    total_ += align_up(n);
    return arena_ != nullptr ? arena_->alloc_stack(n) : nullptr;
  }
  Tensor tensor(DType dt, std::initializer_list<std::int64_t> shape) {
    Tensor t = Tensor::view(nullptr, dt, Device::kCUDA, shape);
    t.data = bytes(t.nbytes());
    return t;
  }
  std::size_t total() const { return total_; }

 private:
  runtime::DeviceArena* arena_;
  std::size_t total_ = 0;
};

// Upper bound of the cu_seqlens entries of all runs [r0, r1) of `rows` rows: rows*(rows+1)/2 runs of
// at most rows + 1 entries.
std::int64_t cu_pool_ints(std::int64_t rows) { return rows * (rows + 1) / 2 * (rows + 1); }
static_assert(kMaxRows * (kMaxRows + 1) / 2 * (kMaxRows + 1) * 4 <= kMaxParamBytes, "cu pool exceeds write_params");

void check_device(const Tensor& t, DType dt, const char* what) {
  if (t.device != Device::kCUDA || t.dtype != dt || t.data == nullptr || !t.is_contiguous()) {
    fail(std::string(what) + " must be a contiguous CUDA " + dtype_name(dt) + " tensor, got " + t.describe());
  }
}

}  // namespace

// Activation buffers of one forward() (`step`) or encode_text() call over `tokens` joint tokens, with a big buffer of
// `big_cols` per token (kBigCols: every view; kFusedBigCols: qkv and attn, the fused block's own views of `big`).
struct MossDit::Activations {
  Tensor x, y;          // bf16 [L, kDim]: residual stream (caller-provided in encode_text), scratch
  Tensor big;           // bf16 [L, big_cols], contiguous
  Tensor qkv, attn;     // bf16 [L, 3*kDim], [L, kDim]  (big buffer, attention phase)
  Tensor up_gate, ffn;  // bf16 [L, 2*kFfnDim], [L, kFfnDim]  (big buffer, FFN phase; kBigCols only)
  Tensor rope;          // bf16 [rope_rows, 128]
  Tensor cu_seqlens;    // int32 pool: zero-based cu_seqlens of every row run (see cu_pool_ints)
  // Reference slots (slot_rows > 0, BlockRun::slot_rows): `big` holds the slotted qkv region [L + 2 slot_rows, 3*kDim]
  // during attention (qkv and attn are its row and attn views), and cu_seqlens_k the key cu_seqlens pool.
  std::int64_t slot_rows = 0;
  Tensor cu_seqlens_k;
  void* lse = nullptr;
  std::size_t lse_bytes = 0;
  // step only: bf16 [6*kDim] block modulation, [2*kDim] head modulation.
  Tensor mod, head;

  Activations(StackPlan& p, std::int64_t tokens, std::int64_t rope_rows, std::int64_t rows, bool step,
              std::int64_t big_cols = kBigCols, std::int64_t slots = 0)
      : slot_rows(slots) {
    if (step) x = p.tensor(DType::kBF16, {tokens, kDim});
    y = p.tensor(DType::kBF16, {tokens, kDim});
    // With slots, whole rows of big_cols more until the slotted attention phase fits.
    const std::int64_t extra = slots > 0 ? (std::max<std::int64_t>(0, slotted_big(tokens, slots) - tokens * big_cols) +
                                            big_cols - 1) / big_cols
                                         : 0;
    big = p.tensor(DType::kBF16, {tokens + extra, big_cols});
    auto* b = static_cast<std::uint16_t*>(big.data);
    qkv = Tensor::view(b + slots * kQkvDim, DType::kBF16, Device::kCUDA, {tokens, kQkvDim});
    attn = Tensor::view(b + (tokens + 2 * slots) * kQkvDim, DType::kBF16, Device::kCUDA, {tokens, kDim});
    if (big_cols >= kBigCols) {
      up_gate = Tensor::view(b, DType::kBF16, Device::kCUDA, {tokens, kUpGateDim});
      ffn = Tensor::view(b + tokens * kUpGateDim, DType::kBF16, Device::kCUDA, {tokens, kFfnDim});
    }
    rope = p.tensor(DType::kBF16, {rope_rows, kernels::kMossRopeDim});
    cu_seqlens = p.tensor(DType::kI32, {cu_pool_ints(rows)});
    if (slots > 0) cu_seqlens_k = p.tensor(DType::kI32, {cu_pool_ints(rows)});
    lse_bytes = attention::flash_fwd_workspace_bytes(tokens, kHeads);
    lse = p.bytes(lse_bytes);
    if (step) {
      mod = p.tensor(DType::kBF16, {6 * kDim});
      head = p.tensor(DType::kBF16, {2 * kDim});
    }
  }
};

const char* block_path_name(BlockPath p) {
  switch (p) {
    case BlockPath::kOpByOp: return "op_by_op";
    case BlockPath::kFused: return "fused";
    case BlockPath::kFusedGraphs: return "fused_graphs";
  }
  return "?";
}

BlockPath parse_block_path(std::string_view name) {
  for (const BlockPath p : {BlockPath::kOpByOp, BlockPath::kFused, BlockPath::kFusedGraphs}) {
    if (name == block_path_name(p)) return p;
  }
  throw std::invalid_argument("DiT block path must be op_by_op, fused or fused_graphs, got \"" + std::string(name) + "\"");
}

std::size_t graph_cache_bytes() {
  // Measured ~100 KiB of device memory per instantiated block graph on sm_89 (cudaMemGetInfo around 42 captures,
  // tests/unit/test_dit_block_fused); 256 KiB leaves room for other drivers and GPUs.
  constexpr std::size_t kPerGraph = std::size_t{256} << 10;
  return kPerGraph * kGraphCapacity;
}

std::string describe_linears(const quant::QuantIndex* quant) {
  if (quant == nullptr) return "bf16";
  std::map<std::string, int> counts;
  for (const auto& [module, spec] : quant->specs) {
    if (module.rfind("layers.", 0) == 0) ++counts[quant::describe(spec)];
  }
  if (counts.size() == 1) return counts.begin()->first;
  std::string out = "mixed (";
  for (const auto& [name, n] : counts) out += (out.size() > 7 ? ", " : "") + std::to_string(n) + " " + name;
  return out + ")";
}

bool has_quantized_linears(const quant::QuantIndex* quant) {
  if (quant == nullptr) return false;
  return std::any_of(quant->specs.begin(), quant->specs.end(),
                     [](const auto& kv) { return kv.first.rfind("layers.", 0) == 0 && kv.second.quantized(); });
}

bool has_int4_linears(const quant::QuantIndex* quant) {
  if (quant == nullptr) return false;
  return std::any_of(quant->specs.begin(), quant->specs.end(), [](const auto& kv) {
    return kv.first.rfind("layers.", 0) == 0 && kv.second.format == quant::QuantFormat::kInt4ConvRot;
  });
}

bool has_nvfp4_linears(const quant::QuantIndex* quant) {
  if (quant == nullptr) return false;
  return std::any_of(quant->specs.begin(), quant->specs.end(), [](const auto& kv) {
    return kv.first.rfind("layers.", 0) == 0 && kv.second.format == quant::QuantFormat::kNvfp4;
  });
}

namespace {
// The share of the block GEMMs' weight elements whose spec has format `f` (NVFP4 weights [N, K/2] count N * K).
double gemm_fraction(const quant::QuantIndex* quant, quant::QuantFormat f) {
  if (quant == nullptr) return 0.0;
  double part = 0, all = 0;
  for (const weights::WeightGroup& g : quant->index.groups) {
    for (const weights::IndexedTensor& t : g.tensors) {
      if (t.name.rfind("layers.", 0) != 0 || !t.name.ends_with(".weight") || t.shape.size() != 2) continue;
      const quant::QuantSpec* spec = quant->find_spec(std::string_view(t.name).substr(0, t.name.size() - 7));
      if (spec == nullptr) continue;
      const double elems = static_cast<double>(t.shape[0]) * static_cast<double>(t.shape[1]) *
                           (spec->format == quant::QuantFormat::kNvfp4 ? 2.0 : 1.0);
      all += elems;
      if (spec->format == f) part += elems;
    }
  }
  return all > 0 ? part / all : 0.0;
}
}  // namespace

double int8_gemm_fraction(const quant::QuantIndex* quant) { return gemm_fraction(quant, quant::QuantFormat::kInt8ConvRot); }
double int4_gemm_fraction(const quant::QuantIndex* quant) { return gemm_fraction(quant, quant::QuantFormat::kInt4ConvRot); }
double nvfp4_gemm_fraction(const quant::QuantIndex* quant) { return gemm_fraction(quant, quant::QuantFormat::kNvfp4); }

namespace {
// The merge refusal for the 4-bit (INT4 and NVFP4) linears (load time and check_lora_merges).
std::string packed4_merge_message(const std::string& module) {
  return "mi::moss_dit: cannot merge a LoRA into '" + module +
         "' (merges need a bf16 or INT8 ConvRot linear; INT4 and NVFP4 linears take side-branch adapters only)";
}
bool is_int4(const quant::QuantIndex* quant, const std::string& module) {
  const quant::QuantSpec* spec = quant != nullptr ? quant->find_spec(module) : nullptr;
  return spec != nullptr && spec->format == quant::QuantFormat::kInt4ConvRot;
}
bool is_nvfp4(const quant::QuantIndex* quant, const std::string& module) {
  const quant::QuantSpec* spec = quant != nullptr ? quant->find_spec(module) : nullptr;
  return spec != nullptr && spec->format == quant::QuantFormat::kNvfp4;
}
// A 4-bit linear (INT4 or NVFP4): side branches only, no merges.
bool is_packed4(const quant::QuantIndex* quant, const std::string& module) {
  return is_int4(quant, module) || is_nvfp4(quant, module);
}
}  // namespace

void check_lora_merges(const lora::LoraStack& stack, const quant::QuantIndex* quant) {
  for (const lora::Adapter& a : stack.adapters()) {
    if (!a.side_branch && is_packed4(quant, a.module)) {
      throw std::invalid_argument(packed4_merge_message(a.module));
    }
  }
}

double int4_side_branch_fraction(const quant::QuantIndex* quant, const lora::SideBranches& sb) {
  if (quant == nullptr) return 0.0;
  constexpr double kD = kDim, kF = kFfnDim;
  const struct {
    lora::Site site;
    const char* parts[3];
    double elems;  // per part
  } sites[] = {{lora::Site::kQkv, {"q", "k", "v"}, kD * kD},
               {lora::Site::kO, {"o", nullptr, nullptr}, kD * kD},
               {lora::Site::kUpGate, {"ffn_up_gate", nullptr, nullptr}, 2 * kF * kD},
               {lora::Site::kDown, {"ffn_down", nullptr, nullptr}, kD * kF}};
  double int4 = 0;
  for (int b = 0; b < kLayers; ++b) {
    for (const auto& s : sites) {
      if (sb.site(b, s.site) == nullptr) continue;
      for (const char* part : s.parts) {
        if (part != nullptr && is_int4(quant, "layers." + std::to_string(b) + ".proj." + part)) int4 += s.elems;
      }
    }
  }
  return int4 / (kLayers * (4 * kD * kD + 3 * kF * kD));
}

std::vector<std::string> shadowed_by_time_embedding(const lora::LoraStack& stack) {
  std::vector<std::string> out;
  for (const lora::Adapter& a : stack.adapters()) {
    if (a.module.rfind("time_embedding.", 0) == 0) {
      out.push_back(a.module + ": overridden by the replacement step embedder (time_embedding_file), not applied");
    } else if ((a.module == "shared_block_modulation" || a.module == "video_head_modulation") && a.bias_diff) {
      out.push_back(a.module + ": its bias delta is overridden by the replacement step embedder's bias, not applied");
    }
  }
  return out;
}

ComputeConfig MossDit::compute_config(const quant::QuantIndex* quant, AttentionMode attention, Int4Route int4_route,
                                      const lora::SideBranches* lora) {
  ComputeConfig c;
  c.attention = attention;
  c.quantized = has_quantized_linears(quant) || attention != AttentionMode::kExact;
  c.int4 = has_int4_linears(quant);
  c.int4_via_int8 = c.int4 && (int4_route == Int4Route::kViaInt8 || (int4_route == Int4Route::kAuto && !int4_mma_native()));
  c.bf16_up_gate = quant == nullptr;
  c.nvfp4 = has_nvfp4_linears(quant);
  if (quant != nullptr) {
    for (const auto& [module, spec] : quant->specs) {
      if (module.rfind("layers.", 0) != 0 || !module.ends_with(".proj.ffn_up_gate")) continue;
      if (!spec.quantized()) c.bf16_up_gate = true;
      if (spec.format == quant::QuantFormat::kNvfp4) {
        c.nvfp4_up_gate = true;
        const std::string down = module.substr(0, module.size() - 11) + "ffn_down";
        if (!is_nvfp4(quant, down)) c.nvfp4_ffn = true;
      }
    }
  }
  if (lora != nullptr && lora->any_active()) {
    c.quantized = true;  // the side branches run on the quantized block path
    c.lora_rank = lora->max_rank();
    for (int b = 0; quant != nullptr && b < kLayers; ++b) {
      const std::string p = "layers." + std::to_string(b) + ".proj.";
      const bool up = lora->site(b, lora::Site::kUpGate) != nullptr, down = lora->site(b, lora::Site::kDown) != nullptr;
      const auto widen = [&](lora::Site s, const char* module, std::int64_t n) {
        if (lora->site(b, s) != nullptr && is_nvfp4(quant, p + module)) c.nvfp4_delta_cols = std::max(c.nvfp4_delta_cols, n);
      };
      widen(lora::Site::kQkv, "q", 3 * kDim);  // a QKV side branch needs q, k and v fused (set_lora)
      widen(lora::Site::kO, "o", kDim);
      widen(lora::Site::kUpGate, "ffn_up_gate", 2 * kFfnDim);
      widen(lora::Site::kDown, "ffn_down", kDim);
      if ((up || down) && is_nvfp4(quant, p + "ffn_up_gate")) c.nvfp4_lora_ffn = true;
    }
  }
  return c;
}

void MossDit::set_attention(AttentionMode mode) {
  if (!attention_supported(mode)) {
    throw std::runtime_error(std::string("mi::moss_dit: attention ") + attention_mode_name(mode) +
                             " is not supported on this device/build");
  }
  refresh_config(mode);
}

std::size_t MossDit::graph_bytes() const { return graphs_.size() * (graph_cache_bytes() / kGraphCapacity); }

void MossDit::set_block_path(BlockPath path) {
  path_ = path;
  graphs_.clear();
  last_step_hash_ = 0;
}

std::size_t MossDit::persistent_bytes(const weights::ModelIndex& index) { return persistent_bytes(index, Residency{}); }

std::size_t MossDit::persistent_bytes(const weights::ModelIndex& index, const Residency& residency) {
  std::size_t total = kernels::Gemm::kWorkspaceBytes;
  total += weights::group_device_bytes(*find_group(index, "globals"));
  total += weights::group_device_bytes(*find_group(index, "text_embedder"));
  return total + BlockWeights::resident_bytes(index, residency);
}

std::size_t MossDit::stream_workspace_bytes(const weights::ModelIndex& index, const Residency& residency) {
  return BlockWeights::buffer_bytes(index, residency);
}

MossDit::MossDit(const weights::ModelIndex& index, runtime::DeviceArena& arena, runtime::StreamSet& streams,
                 bool direct_io, const std::atomic<bool>* cancel, const Residency& residency, const lora::LoraStack* merges,
                 const weights::HostFiles* host_files)
    : MossDit(index, nullptr, arena, streams, direct_io, cancel, residency, merges, host_files) {}

MossDit::MossDit(const quant::QuantIndex& quant, runtime::DeviceArena& arena, runtime::StreamSet& streams, bool direct_io,
                 const std::atomic<bool>* cancel, const Residency& residency, const lora::LoraStack* merges,
                 const weights::HostFiles* host_files)
    : MossDit(quant.index, &quant, arena, streams, direct_io, cancel, residency, merges, host_files) {}

bool MossDit::block_has_merges(const lora::LoraStack& stack, int block) {
  for (const std::string& module : stack.merge_modules()) {
    const lora::TargetModule* t = stack.model().find(module);
    if (t != nullptr && t->block == block) return true;
  }
  return false;
}

// A merged module's weights as uploaded (device) or held (pinned host copy of a streamed block): an owned host copy.
MossDit::MergeStaging::MergeStaging(const MergeTarget& t, bool download) : target(&t) {
  const auto fetch = [&](const Tensor& x, std::vector<std::byte>& host) {
    if (x.data == nullptr) return;
    host.resize(x.nbytes());
    if (!download) return;
    if (x.device == Device::kCUDA) {
      // Device-to-pageable copies return once complete; callers have synchronized the device (or the load's ring).
      MI_CUDA_CHECK(cudaMemcpy(host.data(), x.data, x.nbytes(), cudaMemcpyDeviceToHost));
    } else {
      std::memcpy(host.data(), x.data, x.nbytes());
    }
  };
  fetch(t.w, hw_);
  fetch(t.bias, hb_);
  fetch(t.scale, hs_);
}

MossDit::MergeStaging::MergeStaging(const MergeTarget* t, const MergeStaging& copy)
    : target(t), hw_(copy.hw_), hb_(copy.hb_), hs_(copy.hs_) {}

void MossDit::MergeStaging::store() const {
  if (target == nullptr) throw std::logic_error("mi::moss_dit: a detached merge staging has no target");
  bool device = false;
  const auto put = [&](const Tensor& x, const std::vector<std::byte>& host) {
    if (x.data == nullptr) return;
    if (x.device == Device::kCUDA) {
      MI_CUDA_CHECK(cudaMemcpy(x.data, host.data(), x.nbytes(), cudaMemcpyHostToDevice));
      device = true;
    } else {
      std::memcpy(x.data, host.data(), x.nbytes());
    }
  };
  put(target->w, hw_);
  put(target->bias, hb_);
  put(target->scale, hs_);
  // A pageable host-to-device cudaMemcpy may return before its DMA lands, and the engine's non-blocking streams do not
  // order after the legacy stream: wait for the copies here.
  if (device) MI_CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void MossDit::remerge_into(const lora::PristineWeights& pristine, const lora::LoraStack& stack, const std::string& module,
                           const MergeTarget& t, MergeStaging& staging) {
  if (t.group == 0) {
    pristine.remerge_bf16(stack, module, staging.w<std::uint16_t>(), staging.bias());
  } else {
    pristine.remerge_int8(stack, module, staging.w<std::int8_t>(), staging.scale());
  }
}

MossDit::MergeTarget MossDit::merge_target(const TensorMap& m, const std::string& module) const {
  MergeTarget t;
  const auto w = m.find(module + ".weight");
  if (w == m.end()) return t;
  t.w = w->second;
  if (t.w.dtype == DType::kBF16) {
    const auto b = m.find(module + ".bias");
    if (b != m.end() && b->second.dtype == DType::kBF16) t.bias = b->second;
  } else {
    t.scale = m.at(module + ".weight_scale");
    const quant::QuantSpec* spec = quant_ != nullptr ? quant_->find_spec(module) : nullptr;
    if (spec == nullptr || spec->format != quant::QuantFormat::kInt8ConvRot) throw std::runtime_error(packed4_merge_message(module));
    t.group = spec->group;
  }
  return t;
}

namespace {
std::atomic<std::size_t> g_merge_memo_cap{MossDit::kMaxMergeMemoBytes};
}  // namespace

void MossDit::set_merge_memo_cap_for_tests(std::size_t bytes) { g_merge_memo_cap.store(bytes); }

bool MossDit::rebase_merges(const lora::LoraStack* stack, const std::string& memo_key) {
  static const lora::LoraStack kNone(std::span<const lora::LoraSpec>{}, lora::moss_dit_targets());
  const lora::LoraStack& s = stack != nullptr ? *stack : kNone;
  if (no_pristine_) return false;  // merged without pristine copies: only a reload changes them
  const std::vector<std::string> want = s.merge_modules();
  const auto merged = [&](const std::string& module) {
    return std::any_of(merge_targets_.begin(), merge_targets_.end(), [&](const auto& e) { return e.first == module; });
  };
  std::vector<std::pair<std::string, MergeTarget>> added;
  for (const std::string& module : want) {
    if (merged(module)) continue;
    const MergeTarget t = merge_target(resident_views_, module);
    if (t.w.data == nullptr) return false;  // a block module: only a load can merge it
    added.emplace_back(module, t);
  }
  MI_CUDA_CHECK(cudaDeviceSynchronize());  // no forward may read the weights while they change
  // Transactional: the pristine copies of the new modules and every module's new bits are computed into owned host
  // buffers first; a failure there (a non-finite merge, a bad factor, out of memory) drops the new copies and leaves the
  // model and merge_targets_ as they were. Only then are the buffers written in (a CUDA error during those copies is
  // not recoverable here: the caller discards the model). Peak host memory: the merged modules' size once more.
  std::vector<std::pair<std::string, MergeTarget>> all = merge_targets_;
  all.insert(all.end(), added.begin(), added.end());
  const auto memo_of = [&](const std::string& module) -> const MergeStaging* {
    for (const auto& [m, st] : memo_) {
      if (m == module) return &st;
    }
    return nullptr;
  };
  // The memo serves a stack it was computed for when it holds every module that stack merges (the others go back to
  // their pristine bits, computed as usual).
  bool memo_hit = !memo_key.empty() && memo_key == memo_key_;
  for (const std::string& module : want) memo_hit = memo_hit && memo_of(module) != nullptr;
  std::vector<std::string> new_copies;
  std::vector<MergeStaging> staged;
  staged.reserve(all.size());
  try {
    for (const auto& [module, t] : added) {
      if (!pristine_) pristine_ = std::make_unique<lora::PristineWeights>();
      if (pristine_->has(module)) continue;
      const MergeStaging loaded(t, true);  // never merged: the loaded bits
      if (t.group == 0) {
        pristine_->keep_bf16(module, loaded.w<std::uint16_t>(), t.w.dim(0), t.w.dim(1), loaded.bias());
      } else {
        pristine_->keep_int8(module, loaded.w<std::int8_t>(), loaded.scale(), t.w.dim(0), t.w.dim(1), t.group);
      }
      new_copies.push_back(module);
    }
    for (const auto& [module, t] : all) {
      const bool wanted = std::find(want.begin(), want.end(), module) != want.end();
      if (memo_hit && wanted) {
        staged.emplace_back(&t, *memo_of(module));
        continue;
      }
      staged.emplace_back(t, false);
      remerge_into(*pristine_, s, module, t, staged.back());
    }
  } catch (...) {
    for (const std::string& module : new_copies) pristine_->drop(module);
    throw;
  }
  for (const MergeStaging& st : staged) st.store();
  if (!memo_key.empty() && !memo_hit) {
    // The new stack's merged bits, when they and their pristine copies fit the cap (a stack merging more is merged again
    // when it comes back).
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
      if (std::find(want.begin(), want.end(), all[i].first) != want.end()) {
        bytes += staged[i].bytes() + pristine_->host_bytes(all[i].first);
      }
    }
    memo_.clear();
    memo_key_.clear();
    if (bytes <= g_merge_memo_cap.load()) {
      for (std::size_t i = 0; i < all.size(); ++i) {
        if (std::find(want.begin(), want.end(), all[i].first) != want.end()) {
          memo_.emplace_back(all[i].first, MergeStaging(nullptr, staged[i]));  // detached: `all` is local
        }
      }
      memo_key_ = memo_key;
    }
  }
  std::vector<std::pair<std::string, MergeTarget>> kept;
  for (const auto& [module, t] : all) {
    if (std::find(want.begin(), want.end(), module) != want.end()) {
      kept.emplace_back(module, t);
    } else if (memo_of(module) == nullptr) {
      pristine_->drop(module);  // a module the memo merges keeps its pristine copy: the memo's switch back needs it
    }
  }
  merge_targets_ = std::move(kept);
  if (merge_targets_.empty() && memo_.empty()) pristine_.reset();
  return true;
}

void MossDit::drop_merge_memo() {
  if (memo_.empty()) return;
  for (const auto& [module, st] : memo_) {
    const bool merged =
        std::any_of(merge_targets_.begin(), merge_targets_.end(), [&](const auto& e) { return e.first == module; });
    if (!merged && pristine_) pristine_->drop(module);
  }
  memo_.clear();
  memo_key_.clear();
  if (merge_targets_.empty()) pristine_.reset();
}

std::size_t MossDit::memo_bytes() const {
  std::size_t total = 0;
  for (const auto& [module, st] : memo_) {
    total += st.bytes();
    const bool merged =
        std::any_of(merge_targets_.begin(), merge_targets_.end(), [&](const auto& e) { return e.first == module; });
    if (!merged && pristine_) total += pristine_->host_bytes(module);
  }
  return total;
}

void MossDit::remerge_lora(const lora::LoraStack& stack) {
  if (merge_targets_.empty()) return;
  if (no_pristine_) throw std::logic_error("moss_dit: remerge_lora on a model merged without pristine copies (reload it)");
  MI_CUDA_CHECK(cudaDeviceSynchronize());  // no forward may read the weights while they change
  // Module by module (a failure leaves earlier modules re-merged: the caller discards the model then).
  for (const auto& [module, t] : merge_targets_) {
    MergeStaging staging(t, false);
    remerge_into(*pristine_, stack, module, t, staging);
    staging.store();
  }
}

std::size_t MossDit::pristine_bytes(const lora::LoraStack& stack, const quant::QuantIndex* quant) {
  std::size_t total = 0;
  for (const std::string& module : stack.merge_modules()) {
    const lora::TargetModule* t = stack.model().find(module);
    if (t == nullptr || !stack.keeps_pristine(module)) continue;
    const auto n = static_cast<std::size_t>(t->n), k = static_cast<std::size_t>(t->k);
    const quant::QuantSpec* spec = quant != nullptr ? quant->find_spec(module) : nullptr;
    total += spec != nullptr && spec->quantized() ? n * k + 4 * n : 2 * n * k + (t->has_bias ? 2 * n : 0);
  }
  return total;
}

std::size_t MossDit::pristine_bytes() const {
  std::size_t total = 0;
  for (const auto& [module, t] : merge_targets_) total += pristine_ ? pristine_->host_bytes(module) : 0;
  return total;
}

void MossDit::set_lora(const lora::SideBranches* side_branches) {
  if (side_branches != nullptr) {
    for (int b = 0; b < kLayers; ++b) {
      if (side_branches->site(b, lora::Site::kQkv) == nullptr) continue;
      if (!detail::qkv_fused(b, quant_)) {
        throw std::invalid_argument("mi::moss_dit: a LoRA on the QKV of block " + std::to_string(b) +
                                    ", whose q, k and v differ in precision (or are NVFP4 parts with different tensor scales), is "
                                    "not supported");
      }
    }
  }
  lora_ = side_branches;
  refresh_config(config_.attention);
}

void MossDit::set_int4_route(Int4Route route) {
  int4_route_ = route;
  refresh_config(config_.attention);
}

void MossDit::refresh_config(AttentionMode attention) { config_ = compute_config(quant_, attention, int4_route_, lora_); }

MossDit::MossDit(const weights::ModelIndex& index, const quant::QuantIndex* quant, runtime::DeviceArena& arena,
                 runtime::StreamSet& streams, bool direct_io, const std::atomic<bool>* cancel, const Residency& residency,
                 const lora::LoraStack* merges, const weights::HostFiles* host_files)
    : arena_(arena), quant_(quant), config_(compute_config(quant, AttentionMode::kExact)), cancel_(cancel) {
  if (config_.nvfp4) {
    if (const std::string why = nvfp4_unsupported_reason(); !why.empty()) throw std::runtime_error("mi::moss_dit: " + why);
  }
  const auto t0 = std::chrono::steady_clock::now();
  gemm_workspace_ = arena.alloc_persistent(kernels::Gemm::kWorkspaceBytes);
  weights::ReaderCache readers(index, direct_io, host_files);
  runtime::PinnedRing ring(streams.copy());
  const auto check_cancel = [&] {
    if (cancel_ != nullptr && cancel_->load()) {
      ring.synchronize();  // in-flight copies target the arena the caller will recreate
      throw Cancelled();
    }
  };
  // LoRA merges (lossy for INT8 block GEMMs) into a group's views, on the device (staging copy, merged, uploaded) or
  // in a pinned host copy (in place). The pristine weights are kept on the host first, so a strength change re-merges
  // from them (remerge_lora) instead of reloading.
  const auto merge = [&](const TensorMap& m) {
    if (merges == nullptr) return;
    for (const std::string& module : merges->merge_modules()) {
      const MergeTarget t = merge_target(m, module);
      if (t.w.data == nullptr) continue;
      if (!merges->keeps_pristine(module)) {
        // Merged in place from the loaded bits, without a pristine copy (a merged edit adapter's block modules): a
        // strength or placement change reloads the model (can_remerge()).
        MergeStaging staging(t, true);
        if (t.group == 0) {
          lora::merge_bf16(*merges, module, staging.w<std::uint16_t>(), t.w.dim(0), t.w.dim(1), staging.bias());
        } else {
          lora::merge_requantize_int8(*merges, module, staging.w<std::int8_t>(), staging.scale(), t.w.dim(0), t.w.dim(1),
                                      t.group);
        }
        staging.store();
        merge_targets_.emplace_back(module, t);
        no_pristine_ = true;
        continue;
      }
      if (!pristine_) pristine_ = std::make_unique<lora::PristineWeights>();
      const bool keep = !pristine_->has(module);
      MergeStaging staging(t, keep);
      if (keep) {
        if (t.group == 0) {
          pristine_->keep_bf16(module, staging.w<std::uint16_t>(), t.w.dim(0), t.w.dim(1), staging.bias());
        } else {
          pristine_->keep_int8(module, staging.w<std::int8_t>(), staging.scale(), t.w.dim(0), t.w.dim(1), t.group);
        }
      }
      remerge_into(*pristine_, *merges, module, t, staging);
      staging.store();
      merge_targets_.emplace_back(module, t);
    }
  };
  const auto upload = [&](const WeightGroup& g) {
    check_cancel();
    load_stats_.bytes += g.total_bytes;
    TensorMap m = weights::upload_group(readers, g, arena, ring);
    if (merges != nullptr && !merges->merge_modules().empty()) {
      ring.synchronize();  // the bytes are on the device before they are read back
      merge(m);
    }
    return m;
  };
  if (merges != nullptr && residency.streamed() && residency.source == StreamSource::kFile) {
    const auto mask = resident_mask(residency.resident_blocks);
    for (int b = 0; b < kLayers; ++b) {
      if (!mask.test(static_cast<std::size_t>(b)) && block_has_merges(*merges, b)) {
        throw std::invalid_argument("mi::moss_dit: streamed block " + std::to_string(b) +
                                    " has merged LoRA adapters: stream from a pinned host copy (StreamSource::kHostPinned)");
      }
    }
  }

  const TensorMap glob = upload(*find_group(index, "globals"));
  resident_views_ = glob;
  video_embedder_ = {take(glob, "video_embedder.weight", {kDim, kLatentChannels}), take(glob, "video_embedder.bias", {kDim})};
  time_in_ = {take(glob, "time_embedding.0.weight", {kDim, kFreqDim}), take(glob, "time_embedding.0.bias", {kDim})};
  time_out_ = {take(glob, "time_embedding.2.weight", {kDim, kDim}), take(glob, "time_embedding.2.bias", {kDim})};
  block_modulation_ = {take(glob, "shared_block_modulation.weight", {6 * kDim, kDim}),
                       take(glob, "shared_block_modulation.bias", {6 * kDim})};
  head_modulation_ = {take(glob, "video_head_modulation.weight", {2 * kDim, kDim}),
                      take(glob, "video_head_modulation.bias", {2 * kDim})};
  final_norm_ = take(glob, "final_norm_video.weight", {kDim});
  final_linear_ = take(glob, "final_linear_video.weight", {kLatentChannels, kDim});

  {
    const WeightGroup& g = *find_group(index, "text_embedder");
    std::vector<std::string> front;
    for (int i = 0; i < 2; ++i) {
      const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
      for (const char* n : {"ffn_up.weight", "ffn_gate.weight", "ffn_up.bias", "ffn_gate.bias"}) front.push_back(p + n);
    }
    WeightGroup ordered = with_front(g, {front[0], front[1], front[2], front[3], front[4], front[5], front[6], front[7]});
    const TensorMap m = upload(ordered);
    resident_views_.insert(m.begin(), m.end());
    text_proj_ = {take(m, "text_embedder.proj.weight", {kDim, kTextDim}), take(m, "text_embedder.proj.bias", {kDim})};
    for (int i = 0; i < 2; ++i) {
      const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
      TextBlock& b = text_blocks_[i];
      b.norm1 = take(m, p + "norm1.weight", {kDim});
      b.norm2 = take(m, p + "norm2.weight", {kDim});
      b.q_norm = take(m, p + "q_norm.weight", {kHeadDim});
      b.k_norm = take(m, p + "k_norm.weight", {kHeadDim});
      b.qkv = {take(m, p + "qkv.weight", {kQkvDim, kDim}), take(m, p + "qkv.bias", {kQkvDim})};
      b.proj = {take(m, p + "proj.weight", {kDim, kDim}), take(m, p + "proj.bias", {kDim})};
      b.up_gate = {fuse(m, {p + "ffn_up.weight", p + "ffn_gate.weight"}, kFfnDim, kDim),
                   fuse(m, {p + "ffn_up.bias", p + "ffn_gate.bias"}, kFfnDim, 0)};
      b.down = {take(m, p + "ffn_down.weight", {kDim, kFfnDim}), take(m, p + "ffn_down.bias", {kDim})};
    }
  }

  // Resident blocks are uploaded like the other groups; streamed ones are validated (and, for a pinned host source,
  // read into pinned memory).
  blocks_ = std::make_unique<BlockWeights>(index, arena, residency, upload, [&](const WeightGroup& g) {
    check_cancel();
    if (residency.source == StreamSource::kHostPinned) load_stats_.bytes += g.total_bytes;
  }, quant, [&](const TensorMap& host) { merge(host); });
  host_bytes_ = BlockWeights::host_bytes(index, residency);
  ring.synchronize();
  load_stats_.direct = readers.all_direct();
  for (cudaEvent_t& e : block_done_) MI_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  load_stats_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

MossDit::~MossDit() {
  for (cudaEvent_t e : block_done_) {
    if (e != nullptr) cudaEventDestroy(e);
  }
}

void MossDit::linear(const Tensor& out, const Tensor& x, const Linear& l, cudaStream_t stream) {
  gemm_.linear(out, x, l.w, l.bias.data != nullptr ? &l.bias : nullptr, gemm_workspace_, kernels::Gemm::kWorkspaceBytes,
               stream);
}

void MossDit::linear(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream) {
  gemm_.linear(out, x, w, nullptr, gemm_workspace_, kernels::Gemm::kWorkspaceBytes, stream);
}

std::size_t MossDit::encode_text_workspace_bytes(std::int64_t text_length) {
  StackPlan plan(nullptr);
  Activations(plan, text_length, text_length + 1, 1, false);
  return plan.total();
}

void MossDit::encode_text(const Tensor& text_embeds, const Tensor& out, cudaStream_t stream) {
  check_device(text_embeds, DType::kBF16, "text_embeds");
  check_device(out, DType::kBF16, "out");
  const std::int64_t s = text_embeds.dim(0);
  if (text_embeds.rank != 2 || text_embeds.dim(1) != kTextDim || s < 1 || out.rank != 2 || out.dim(0) != s ||
      out.dim(1) != kDim) {
    fail("encode_text expects text_embeds [S, 4096] and out [S, 3072], got " + text_embeds.describe() + " and " +
         out.describe());
  }
  runtime::StackScope scope(arena_);
  StackPlan plan(&arena_);
  Activations a(plan, s, s + 1, 1, false);
  a.x = out;
  const std::int32_t cu[2] = {0, static_cast<std::int32_t>(s)};
  write_params(a.cu_seqlens.data, cu, sizeof(cu), stream);
  if (varlen_debug_checks()) validate_cu_seqlens(a.cu_seqlens, static_cast<int>(s), s, stream, "moss_dit::encode_text");
  // Text rows of the joint table (P:251) do not depend on the image grid: build a 1x1 grid and skip it.
  kernels::moss_rope_table(a.rope, 1, 1, static_cast<int>(s), stream);
  const Tensor rope = a.rope.slice0(1, s + 1);
  const Tensor q = heads_view(a.qkv, 0), k = heads_view(a.qkv, kDim), v = heads_view(a.qkv, 2 * kDim);
  const Tensor attn = heads_view(a.attn, 0);

  linear(a.x, text_embeds, text_proj_, stream);  // P:120
  for (const TextBlock& b : text_blocks_) {     // _TextBlock.forward, P:104-110
    kernels::rms_norm_moss(a.x, b.norm1, a.y, stream);
    linear(a.qkv, a.y, b.qkv, stream);
    kernels::rms_norm_moss(q, b.q_norm, q, stream);
    kernels::rms_norm_moss(k, b.k_norm, k, stream);
    kernels::moss_rope_apply(q, rope, q, stream);
    kernels::moss_rope_apply(k, rope, k, stream);
    attention::flash_fwd(q, k, v, attn, a.cu_seqlens, a.cu_seqlens, static_cast<int>(s), false, kAttnScale, a.lse,
                         a.lse_bytes, stream);
    linear(a.y, a.attn, b.proj, stream);
    kernels::residual_add(a.x, a.y, a.x, stream);
    kernels::rms_norm_moss(a.x, b.norm2, a.y, stream);
    linear(a.up_gate, a.y, b.up_gate, stream);
    kernels::situ_glu(a.up_gate, a.ffn, stream);
    linear(a.y, a.ffn, b.down, stream);
    kernels::residual_add(a.x, a.y, a.x, stream);
  }
}

std::vector<std::int64_t> MossDit::token_offsets(std::int64_t height, std::int64_t width,
                                                 std::span<const std::int64_t> text_lengths) {
  std::vector<std::int64_t> off{0};
  for (std::int64_t s : text_lengths) off.push_back(off.back() + height * width + s);
  return off;
}

namespace {
// time_modulation() scratch: bf16 [1, 256] sinusoid, [1, kDim] MLP hidden and output.
struct TimeScratch {
  Tensor t_in, hidden, time;
  explicit TimeScratch(StackPlan& p)
      : t_in(p.tensor(DType::kBF16, {1, kFreqDim})),
        hidden(p.tensor(DType::kBF16, {1, kDim})),
        time(p.tensor(DType::kBF16, {1, kDim})) {}
};
}  // namespace

void MossDit::set_time_embedding(const TimeEmbeddingFp32* te) {
  te_cache_.clear();
  if (te == nullptr) {
    te_.reset();
    te_buf_ = DeviceBuffer();
    return;
  }
  const auto sized = [](const std::vector<float>& v, std::size_t n) { return v.size() == n; };
  if (!sized(te->w0, std::size_t{kDim} * kFreqDim) || !sized(te->b0, kDim) || !sized(te->w2, std::size_t{kDim} * kDim) ||
      !sized(te->b2, kDim) || !(te->mod_bias.empty() || sized(te->mod_bias, 6 * kDim)) ||
      !(te->head_bias.empty() || sized(te->head_bias, 2 * kDim))) {
    fail("set_time_embedding: replacement tensors have unexpected sizes");
  }
  // The biases the device linears add: the replacement's, or the model's own widened.
  std::vector<float> biases(8 * kDim);
  const auto own = [&](const Tensor& b, std::size_t at) {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(b.numel()));
    MI_CUDA_CHECK(cudaMemcpy(h.data(), b.data, h.size() * 2, cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < h.size(); ++i) {
      const std::uint32_t u = static_cast<std::uint32_t>(h[i]) << 16;
      std::memcpy(&biases[at + i], &u, 4);
    }
  };
  if (te->mod_bias.empty()) own(block_modulation_.bias, 0);
  else std::copy(te->mod_bias.begin(), te->mod_bias.end(), biases.begin());
  if (te->head_bias.empty()) own(head_modulation_.bias, 6 * kDim);
  else std::copy(te->head_bias.begin(), te->head_bias.end(), biases.begin() + 6 * kDim);
  DeviceBuffer buf((9 * kDim) * sizeof(float));
  buf.upload(biases.data(), biases.size() * sizeof(float), kDim * sizeof(float));
  te_ = std::make_unique<TimeEmbeddingFp32>(*te);
  te_buf_ = std::move(buf);
}

void MossDit::time_modulation(float timestep, const Tensor& mod, const Tensor& head, cudaStream_t stream) {
  check_device(mod, DType::kBF16, "mod");
  check_device(head, DType::kBF16, "head");
  if (mod.numel() != 6 * kDim || head.numel() != 2 * kDim) fail("mod/head must hold 6*3072 / 2*3072 values");
  if (te_) {
    std::uint32_t key = 0;
    std::memcpy(&key, &timestep, 4);
    auto it = te_cache_.find(key);
    if (it == te_cache_.end()) {
      if (te_cache_.size() >= 256) te_cache_.clear();
      it = te_cache_.emplace(key, time_vector_fp32(*te_, timestep)).first;
    }
    float* const dev = static_cast<float*>(te_buf_.get());
    const std::vector<float>& time = it->second;
    constexpr std::size_t kChunk = kMaxParamBytes / sizeof(float);
    for (std::size_t i = 0; i < time.size(); i += kChunk) {
      write_params(dev + i, time.data() + i, std::min(kChunk, time.size() - i) * sizeof(float), stream);
    }
    linear_fp32_row(block_modulation_.w, dev, dev + kDim, mod, stream);
    linear_fp32_row(head_modulation_.w, dev, dev + 7 * kDim, head, stream);
    return;
  }
  runtime::StackScope scope(arena_);
  StackPlan plan(&arena_);
  TimeScratch t(plan);
  // P:232-241: float64 sinusoid -> fp32 -> bf16 MLP; the .float() and back to bf16 are exact.
  // Known theoretical risk: libm pow/cos/sin are not guaranteed correctly rounded, so a 1-ulp double difference
  // from torch's could flip a bf16 rounding of one embedding value (not observed on the tested timesteps).
  constexpr int kHalf = kFreqDim / 2;
  std::uint16_t emb[kFreqDim];
  for (int i = 0; i < kHalf; ++i) {
    const double freq = std::pow(10000.0, -static_cast<double>(i) / kHalf);
    const double phase = static_cast<double>(timestep) * freq;
    emb[i] = bf16_bits(static_cast<float>(std::cos(phase)));
    emb[kHalf + i] = bf16_bits(static_cast<float>(std::sin(phase)));
  }
  write_params(t.t_in.data, emb, sizeof(emb), stream);
  linear(t.hidden, t.t_in, time_in_, stream);
  kernels::silu(t.hidden, t.hidden, stream);
  linear(t.time, t.hidden, time_out_, stream);
  linear(Tensor::view(mod.data, DType::kBF16, Device::kCUDA, {1, 6 * kDim}), t.time, block_modulation_, stream);
  linear(Tensor::view(head.data, DType::kBF16, Device::kCUDA, {1, 2 * kDim}), t.time, head_modulation_, stream);
}

namespace {
// Big-buffer columns the NVFP4 ffn_up_gate linears of `config` need.
std::int64_t nvfp4_cols(const ComputeConfig& config) {
  return nvfp4_big_cols(config.nvfp4_up_gate, config.nvfp4_ffn || config.nvfp4_lora_ffn);
}

// The quantized block's buffers for `total` tokens, after the activations (every caller's stack order): per-token
// scales, the 8-bit attention workspace of q8_bytes, the LoRA t buffers (for `side_branches`), the INT4 unpack buffer
// and the NVFP4 scales and Δ.
QuantBlockBuffers quant_buffers(StackPlan& plan, const ComputeConfig& config, std::int64_t total, std::size_t q8_bytes,
                                const lora::SideBranches* side_branches = nullptr) {
  QuantBlockBuffers qa;
  qa.scale = plan.tensor(DType::kF32, {total});
  qa.attn_workspace_bytes = q8_bytes;
  if (q8_bytes > 0) qa.attn_workspace = plan.bytes(q8_bytes);
  if (config.lora_rank > 0) {
    qa.lora = side_branches;
    qa.lora_t = plan.tensor(DType::kBF16, {total, config.lora_rank});
    qa.lora_workspace_bytes = lora::t_workspace_bytes(total, config.lora_rank);
    qa.lora_workspace = plan.bytes(qa.lora_workspace_bytes);
  }
  qa.int4_via_int8 = config.int4_via_int8;
  if (config.int4 && (config.lora_rank > 0 || config.int4_via_int8)) {
    qa.int4_unpack_bytes = kInt4UnpackBytes;
    qa.int4_unpack = plan.bytes(kInt4UnpackBytes);
  }
  if (config.nvfp4) {
    qa.nvfp4_sf_bytes = nvfp4_sf_bytes(total);
    qa.nvfp4_sf = plan.bytes(qa.nvfp4_sf_bytes);
    if (config.nvfp4_delta_cols > 0) {
      qa.nvfp4_delta = plan.tensor(DType::kBF16, {total, config.nvfp4_delta_cols});
      qa.nvfp4_scratch_bytes = lora::delta_scratch_bytes(total, config.lora_rank, config.nvfp4_delta_cols);
      qa.nvfp4_scratch = plan.bytes(qa.nvfp4_scratch_bytes);
    }
  }
  return qa;
}
}  // namespace

std::size_t MossDit::workspace_bytes(std::int64_t height, std::int64_t width, std::span<const std::int64_t> text_lengths,
                                     BlockPath path) {
  return workspace_bytes(height, width, text_lengths, path, ComputeConfig{}, nullptr);
}

std::size_t MossDit::workspace_bytes(std::int64_t height, std::int64_t width, std::span<const std::int64_t> text_lengths,
                                     BlockPath path, const ComputeConfig& config, const FusedRouteSnapshot* pinned,
                                     std::int64_t ref_tokens) {
  const std::int64_t total = token_offsets(height, width, text_lengths).back(), hw = height * width;
  const FusedRouteSnapshot route =
      path == BlockPath::kOpByOp ? FusedRouteSnapshot{} : (pinned != nullptr ? *pinned : fused_route_snapshot());
  StackPlan plan(nullptr);
  Activations(plan, total, total, static_cast<std::int64_t>(text_lengths.size()), true,
              config.quantized ? quant_big_cols(hw, text_lengths, config.bf16_up_gate, route, nvfp4_cols(config))
                               : big_cols(path, hw, text_lengths, route),
              ref_tokens);
  if (config.quantized) {
    quant_buffers(plan, config, total, q8_attention_workspace_bytes(hw, text_lengths, config.attention, ref_tokens));
  }
  TimeScratch{plan};
  return plan.total();
}

std::size_t MossDit::reference_cache_bytes(std::int64_t ref_tokens) {
  return static_cast<std::size_t>(kLayers) * static_cast<std::size_t>(ref_tokens) * 2 * kDim * sizeof(std::uint16_t);
}

std::size_t MossDit::prefill_workspace_bytes(std::int64_t ref_tokens, BlockPath path, const ComputeConfig& config,
                                             const FusedRouteSnapshot* pinned) {
  const FusedRouteSnapshot route =
      path == BlockPath::kOpByOp ? FusedRouteSnapshot{} : (pinned != nullptr ? *pinned : fused_route_snapshot());
  // One sequence of R tokens without text (prefill_references' allocations, in order).
  const std::int64_t none[] = {0};
  StackPlan plan(nullptr);
  Activations(plan, ref_tokens, ref_tokens, 1, true,
              config.quantized ? quant_big_cols(ref_tokens, none, config.bf16_up_gate, route, nvfp4_cols(config))
                               : big_cols(path, ref_tokens, none, route));
  if (config.quantized) quant_buffers(plan, config, ref_tokens, q8_attention_workspace_bytes(ref_tokens, none, config.attention));
  TimeScratch{plan};
  return plan.total();
}

std::size_t MossDit::prefill_stack_bytes(std::int64_t ref_tokens) const {
  const ComputeConfig config = lora_ != nullptr ? compute_config(quant_, config_.attention, int4_route_, lora_) : config_;
  const FusedRouteSnapshot* route = request_route_ ? &*request_route_ : nullptr;
  return prefill_workspace_bytes(ref_tokens, path_, config, route) + blocks_->buffer_bytes();
}

void MossDit::set_references(const RefKv* refs) {
  if (refs != nullptr) {
    const Tensor& t = refs->kv;
    check_device(t, DType::kBF16, "reference cache");
    if (t.rank != 3 || t.dim(0) != kLayers || t.dim(1) < 1 || t.dim(1) > kMaxReferenceTokens || t.dim(2) != 2 * kDim) {
      fail("the reference cache must be [42, R, 6144] with 1 <= R <= " + std::to_string(kMaxReferenceTokens) + ", got " +
           t.describe());
    }
  }
  refs_ = refs;
}

std::size_t MossDit::forward_stack_bytes(std::int64_t height, std::int64_t width,
                                        std::span<const std::int64_t> text_lengths) const {
  const ComputeConfig config = lora_ != nullptr ? compute_config(quant_, config_.attention, int4_route_, lora_) : config_;
  const FusedRouteSnapshot* route = request_route_ ? &*request_route_ : nullptr;
  return workspace_bytes(height, width, text_lengths, path_, config, route, refs_ != nullptr ? refs_->tokens() : 0) +
         blocks_->buffer_bytes();
}

void embed_video(kernels::Gemm& gemm, void* gemm_workspace, std::size_t gemm_workspace_bytes, const Tensor& latents,
                 const Tensor* extra, void* tokens, const Tensor& weight, const Tensor& bias, const Tensor& out,
                 cudaStream_t stream) {
  const std::int64_t hw = latents.dim(1) * latents.dim(2);
  if (extra == nullptr) {
    const Tensor t = Tensor::view(tokens, DType::kBF16, Device::kCUDA, {hw, kLatentChannels});
    latent_to_tokens(latents, t, stream);
    gemm.linear(out, t, weight, bias.data != nullptr ? &bias : nullptr, gemm_workspace, gemm_workspace_bytes, stream);
    return;
  }
  const std::int64_t cin = kLatentChannels + extra->dim(0);
  if (weight.rank != 2 || weight.dim(1) != cin) fail("embed_video: weight must be [N, " + std::to_string(cin) + "], got " + weight.describe());
  const Tensor t = Tensor::view(tokens, DType::kBF16, Device::kCUDA, {hw, cin});
  latent_to_tokens(latents, t, 0, stream);
  latent_to_tokens(*extra, t, kLatentChannels, stream);
  gemm.linear(out, t, weight, bias.data != nullptr ? &bias : nullptr, gemm_workspace, gemm_workspace_bytes, stream);
}

void MossDit::set_input_extension(const InputExtension* ext) {
  if (ext != nullptr) {
    check_device(ext->weight, DType::kBF16, "input extension weight");
    check_device(ext->bias, DType::kBF16, "input extension bias");
    if (ext->weight.rank != 2 || ext->weight.dim(0) != kDim || ext->weight.dim(1) <= kLatentChannels || !ext->weight.is_contiguous()) {
      fail("input extension weight must be a contiguous [3072, 128 + C] with C > 0, got " + ext->weight.describe());
    }
    if (ext->bias.rank != 1 || ext->bias.dim(0) != kDim) fail("input extension bias must be [3072], got " + ext->bias.describe());
    // The widened tokens use the big activation buffer: at most kFusedBigCols columns per token.
    if (ext->weight.dim(1) > kFusedBigCols) fail("input extension: too many input channels");
  }
  input_ext_ = ext;
}

void MossDit::run_block(int bi, const Block& b, const Activations& a, const Run& run, cudaStream_t stream) {
  // _Block.forward (P:138-148) on tokens [t0, t1); mod chunks (P:139): shift_a, scale_a, gate_a, shift_f, scale_f, gate_f.
  const Tensor& mod = a.mod;
  const Tensor x = a.x.slice0(run.t0, run.t1), y = a.y.slice0(run.t0, run.t1);
  const Tensor qkv = a.qkv.slice0(run.t0, run.t1);
  const Tensor q = heads_view(qkv, 0), k = heads_view(qkv, kDim), v = heads_view(qkv, 2 * kDim);
  const Tensor rope = a.rope.slice0(run.t0, run.t1);
  // The keys and values attention reads: the run's own, or with reference slots its key rows of the slotted region.
  Tensor kv_rows = qkv;
  if (run.slot_rows > 0) {
    kv_rows = Tensor::view(a.big.data, DType::kBF16, Device::kCUDA, {a.qkv.dim(0) + 2 * run.slot_rows, kQkvDim})
                  .slice0(run.k0, run.k1);
  }

  const auto inputs = [&](BlockGemm g, const Tensor& in) {
    if (linear_input_hook_) linear_input_hook_(bi, g, in, stream);
  };
  kernels::rms_norm_modulate_moss(x, b.norm1, vec(mod, 1), vec(mod, 0), y, stream);
  inputs(BlockGemm::kQkv, y);
  linear(qkv, y, b.qkv, stream);
  kernels::rms_norm_moss(q, b.q_norm, q, stream);
  kernels::rms_norm_moss(k, b.k_norm, k, stream);
  kernels::moss_rope_apply(q, rope, q, stream);
  kernels::moss_rope_apply(k, rope, k, stream);
  if (run.kv_tap != nullptr) {
    tap_reference_kv(qkv, run.kv_tap, stream);
    if (run.tap_only) return;
  }
  const Tensor attn = a.attn.slice0(run.t0, run.t1);
  if (run.slot_rows == 0) {
    attention::flash_fwd(q, k, v, heads_view(attn, 0), run.cu_seqlens, run.cu_seqlens, run.max_seqlen, false,
                         kAttnScale, a.lse, a.lse_bytes, stream);
  } else {
    attention::flash_fwd(q, heads_view(kv_rows, kDim), heads_view(kv_rows, 2 * kDim), heads_view(attn, 0), run.cu_seqlens,
                         run.cu_seqlens_k, run.max_seqlen, false, kAttnScale, a.lse, a.lse_bytes, stream);
  }
  inputs(BlockGemm::kO, attn);
  linear(y, attn, b.o, stream);
  kernels::gated_residual(x, y, vec(mod, 2), x, stream);
  kernels::rms_norm_modulate_moss(x, b.norm2, vec(mod, 4), vec(mod, 3), y, stream);
  const Tensor up_gate = a.up_gate.slice0(run.t0, run.t1), ffn = a.ffn.slice0(run.t0, run.t1);
  inputs(BlockGemm::kUpGate, y);
  linear(up_gate, y, b.up_gate, stream);
  kernels::situ_glu(up_gate, ffn, stream);
  inputs(BlockGemm::kDown, ffn);
  linear(y, ffn, b.down, stream);
  kernels::gated_residual(x, y, vec(mod, 5), x, stream);
}

void MossDit::forward(const Tensor& latents, float timestep, std::span<const Row> rows, std::span<const Tensor> out,
                      cudaStream_t stream, const BlockHook& hook) {
  check_device(latents, DType::kF32, "latents");
  // [128, h, w], or a latent batch [L, 128, h, w].
  const bool batched = latents.rank == 4;
  if (!(latents.rank == 3 && latents.dim(0) == kLatentChannels) &&
      !(batched && latents.dim(0) >= 1 && latents.dim(1) == kLatentChannels)) {
    fail("latents must be [128, h, w] or [L, 128, h, w], got " + latents.describe());
  }
  const std::int64_t num_latents = batched ? latents.dim(0) : 1;
  const std::int64_t height = latents.dim(latents.rank - 2), width = latents.dim(latents.rank - 1), hw = height * width;
  if (rows.empty() || out.size() != rows.size()) fail("forward needs >= 1 row and one output per row");
  if (rows.size() > static_cast<std::size_t>(kMaxRows)) fail("at most " + std::to_string(kMaxRows) + " rows per forward");
  std::vector<std::int64_t> lengths;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const Tensor& t = rows[r].text;
    check_device(t, DType::kBF16, "row text");
    if (t.rank != 2 || t.dim(0) < 1 || t.dim(1) != kDim) fail("row text must be [S >= 1, 3072], got " + t.describe());
    if (rows[r].latent < 0 || rows[r].latent >= num_latents) {
      fail("row " + std::to_string(r) + " reads latent " + std::to_string(rows[r].latent) + " of " +
           std::to_string(num_latents));
    }
    check_device(out[r], DType::kF32, "out");
    if (out[r].rank != 3 || out[r].dim(0) != kLatentChannels || out[r].dim(1) != height || out[r].dim(2) != width) {
      fail("out must match latents [128, h, w], got " + out[r].describe());
    }
    lengths.push_back(t.dim(0));
  }
  // An input extension's latents: checked with the rows, before any work is enqueued.
  if (input_ext_ != nullptr) {
    const Tensor& e = input_ext_->extra;
    const std::int64_t c = input_ext_->weight.dim(1) - kLatentChannels;
    check_device(e, DType::kF32, "input extension extra");
    if (e.rank != latents.rank || (batched && e.dim(0) != num_latents) || e.dim(e.rank - 3) != c ||
        e.dim(e.rank - 2) != height || e.dim(e.rank - 1) != width || !e.is_contiguous()) {
      fail("input extension extra must be a contiguous [" + (batched ? std::to_string(num_latents) + ", " : std::string()) +
           std::to_string(c) + ", " + std::to_string(height) + ", " + std::to_string(width) + "], got " + e.describe());
    }
  }
  const std::vector<std::int64_t> off = token_offsets(height, width, lengths);
  const std::int64_t total = off.back();
  // Past the self-check's longest run the linears would go op by op unchecked: refused instead (a latent batch or many
  // long rows; mi_generate's rows stay within it).
  if (total > kMaxRunRows) {
    fail("a forward of " + std::to_string(total) + " joint tokens exceeds " + std::to_string(kMaxRunRows));
  }
  // Reference keys (set_references): each run's sequences take a slot next to them, which covers every run of one
  // latent and at most 2 rows (a run is then {row 0}, {row 1} or both, and a block has one run).
  const std::int64_t ref_tokens = refs_ != nullptr ? refs_->tokens() : 0;
  if (ref_tokens > 0 && (num_latents != 1 || rows.size() > 2)) {
    fail("a forward with reference images takes one latent and at most 2 rows, got " + std::to_string(num_latents) +
         " latents and " + std::to_string(rows.size()) + " rows");
  }

  // One route for the whole forward (its big buffer, graph keys and every block): an install_fused_check meanwhile, e.g.
  // another pipeline's poisoning, applies from the next forward on instead of failing this one's buffer check.
  // The LoRA sites active now (strengths may have changed since set_lora) decide the block path and the t buffers.
  if (lora_ != nullptr) refresh_config(config_.attention);
  // A request's pinned route (set_request_route) instead, so every step of the request matches the workspace it sized.
  const FusedRouteSnapshot route = path_ == BlockPath::kOpByOp ? FusedRouteSnapshot{}
                                   : request_route_              ? *request_route_
                                                                 : fused_route_snapshot();
  const bool quantized = config_.quantized;
  if (linear_input_hook_ && (quantized || path_ != BlockPath::kOpByOp)) {
    throw std::logic_error("moss_dit: a linear-input hook needs the bf16 op-by-op block path");
  }
  runtime::StackScope scope(arena_);
  StackPlan plan(&arena_);
  Activations a(plan, total, total, static_cast<std::int64_t>(rows.size()), true,
                quantized ? quant_big_cols(hw, lengths, config_.bf16_up_gate, route, nvfp4_cols(config_))
                          : big_cols(path_, hw, lengths, route),
                ref_tokens);
  // Quantized blocks: per-token activation scales and the 8-bit attention workspace (workspace_bytes' order).
  QuantBlockBuffers qa;
  if (quantized) {
    qa = quant_buffers(plan, config_, total, q8_attention_workspace_bytes(hw, lengths, config_.attention, ref_tokens),
                       lora_);
  }
  // Streamed blocks: device buffers after the activations; the first copies overlap the setup below.
  BlockWeights::Pass weights(*blocks_, stream, cancel_);

  // Each maximal run of contiguous rows that executes a block is one varlen sub-batch with its own
  // zero-based cu_seqlens (flash_fwd's contract); without skips that is one run over all rows.
  std::vector<std::vector<Run>> block_runs(kLayers);
  {
    std::map<std::pair<std::size_t, std::size_t>, Run> unique;
    std::vector<std::int32_t> cu, cu_k;
    for (int bi = 0; bi < kLayers; ++bi) {
      const auto skips = [&](std::size_t r) { return rows[r].skip.test(static_cast<std::size_t>(bi)); };
      for (std::size_t r0 = 0; r0 < rows.size();) {
        if (skips(r0)) {
          ++r0;
          continue;
        }
        std::size_t r1 = r0;
        while (r1 < rows.size() && !skips(r1)) ++r1;
        auto it = unique.find({r0, r1});
        if (it == unique.end()) {
          const auto at = static_cast<std::int64_t>(cu.size());
          std::int64_t max_len = 0;
          for (std::size_t r = r0; r <= r1; ++r) {
            cu.push_back(static_cast<std::int32_t>(off[r] - off[r0]));
            if (r < r1) max_len = std::max(max_len, off[r + 1] - off[r]);
          }
          Run run;
          run.t0 = off[r0];
          run.t1 = off[r1];
          run.cu_seqlens = a.cu_seqlens.slice0(at, at + static_cast<std::int64_t>(r1 - r0) + 1);
          run.max_seqlen = static_cast<int>(max_len);
          if (ref_tokens > 0) {
            // The first sequence takes slot A when the run starts at row 0, every other one slot B (after the last
            // row): its keys are [slot A | own] or [own | slot B], contiguous rows of the slotted region.
            const bool slot_a = r0 == 0, slot_b = r1 - r0 == 2 || !slot_a;
            run.slot_rows = ref_tokens;
            run.k0 = slot_a ? off[r0] : ref_tokens + off[r0];
            run.k1 = ref_tokens + off[r1] + (slot_b ? ref_tokens : 0);
            run.cu_seqlens_k = a.cu_seqlens_k.slice0(at, at + static_cast<std::int64_t>(r1 - r0) + 1);
            run.max_seqlen = static_cast<int>(max_len + ref_tokens);
            cu_k.resize(cu.size() - (r1 - r0) - 1);
            for (std::size_t r = r0; r <= r1; ++r) {
              cu_k.push_back(static_cast<std::int32_t>(off[r] - off[r0] + static_cast<std::int64_t>(r - r0) * ref_tokens));
            }
          }
          it = unique.emplace(std::make_pair(r0, r1), run).first;
        }
        block_runs[static_cast<std::size_t>(bi)].push_back(it->second);
        r0 = r1;
      }
    }
    write_params(a.cu_seqlens.data, cu.data(), cu.size() * sizeof(std::int32_t), stream);
    if (ref_tokens > 0) write_params(a.cu_seqlens_k.data, cu_k.data(), cu_k.size() * sizeof(std::int32_t), stream);
    if (varlen_debug_checks()) {
      for (const auto& [key, run] : unique) {
        validate_cu_seqlens(run.cu_seqlens, run.max_seqlen, run.t1 - run.t0, stream, "moss_dit::forward");
        if (ref_tokens > 0) validate_cu_seqlens(run.cu_seqlens_k, run.max_seqlen, run.k1 - run.k0, stream, "moss_dit::forward");
      }
    }
  }
  // Which slots each block's run reads (one run per block with references): their k and v columns are filled from the
  // cache before the block.
  std::vector<std::uint8_t> slot_use(ref_tokens > 0 ? kLayers : 0);
  for (std::size_t bi = 0; bi < slot_use.size(); ++bi) {
    for (const Run& run : block_runs[bi]) {
      slot_use[bi] |= (run.k0 == 0 ? 1 : 0) | (run.k1 > run.slot_rows + run.t1 ? 2 : 0);
    }
  }

  time_modulation(timestep, a.mod, a.head, stream);

  // Per-row RoPE tables (P:250, text length differs per row) and the joint inputs [video_r | text_r] (P:247-252).
  // The video embedding is the same for every row of a latent: computed once, into the latent's first row, and copied
  // to its other rows (one latent: row 0's embedding copied to every other row).
  // Latent l's [C, h, w] plane of a [L, C, h, w] batch (the tensor itself when not batched).
  const auto plane = [&](const Tensor& t, int l) {
    if (!batched) return t;
    const std::int64_t c = t.dim(1);
    return Tensor::view(t.ptr<float>() + static_cast<std::size_t>(l) * static_cast<std::size_t>(c * hw), DType::kF32,
                        Device::kCUDA, {c, height, width});
  };
  // src[r]: the row whose video tokens hold row r's embedding (the first row reading its latent).
  std::size_t src[kMaxRows];
  for (std::size_t r = 0; r < rows.size(); ++r) {
    src[r] = r;
    for (std::size_t q = 0; q < r; ++q) {
      if (rows[q].latent == rows[r].latent) {
        src[r] = q;
        break;
      }
    }
    if (src[r] != r) continue;
    // The tokens live in the big buffer ([total, >= kFusedBigCols] bf16, idle until the first block), also when widened.
    const Tensor extra = input_ext_ != nullptr ? plane(input_ext_->extra, rows[r].latent) : Tensor{};
    embed_video(gemm_, gemm_workspace_, kernels::Gemm::kWorkspaceBytes, plane(latents, rows[r].latent),
                input_ext_ != nullptr ? &extra : nullptr, a.big.data,
                input_ext_ != nullptr ? input_ext_->weight : video_embedder_.w,
                input_ext_ != nullptr ? input_ext_->bias : video_embedder_.bias, a.x.slice0(off[r], off[r] + hw), stream);
  }
  const std::size_t row_bytes = kDim * sizeof(std::uint16_t);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    kernels::moss_rope_table(a.rope.slice0(off[r], off[r + 1]), static_cast<int>(height), static_cast<int>(width),
                             static_cast<int>(lengths[r]), stream);
    if (src[r] != r) {
      MI_CUDA_CHECK(cudaMemcpyAsync(a.x.slice0(off[r], off[r] + hw).data, a.x.slice0(off[src[r]], off[src[r]] + hw).data,
                                    static_cast<std::size_t>(hw) * row_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    MI_CUDA_CHECK(cudaMemcpyAsync(a.x.slice0(off[r] + hw, off[r + 1]).data, rows[r].text.data,
                                  static_cast<std::size_t>(lengths[r]) * row_bytes, cudaMemcpyDeviceToDevice, stream));
  }

  // Fused blocks: kernels/fused + cuBLASLt QKV on the same activations. With kFusedGraphs and every block resident (a
  // streamed block's acquire/release are host handshakes and cross-stream waits, never captured), each block is
  // replayed from a CUDA graph keyed by the exact step shape and every address it binds; graphs are captured only once
  // a step shape repeats (a one-off forward stays eager). The cancel throttle and the hook run between blocks, outside
  // the graphs, so cancel latency is unchanged.
  const FusedBlockBuffers fused{a.x, a.y, a.big, a.rope, a.mod, a.lse, a.lse_bytes};
  qa.f = fused;
  const QkvGemm qkv_gemm{&gemm_, gemm_workspace_, kernels::Gemm::kWorkspaceBytes};
  const auto run_blocks = [&](int bi, const Block& b, const std::vector<Run>& runs, cudaStream_t s) {
    for (const Run& run : runs) {
      if (quantized) {
        run_block_quant(b, qa, run, qkv_gemm, route, config_.attention, s, bi);
      } else if (path_ == BlockPath::kOpByOp) {
        run_block(bi, b, a, run, s);
      } else {
        run_block_fused(b, fused, run, qkv_gemm, route, s);
      }
    }
  };
  StepGraphKey key;
  bool use_graphs = false;
  // attention_q8's debug checks synchronize the stream: never inside a capture.
  const bool capturable = !(config_.attention != AttentionMode::kExact && varlen_debug_checks());
  // A forward with references runs eagerly (its slot copies are outside the captured blocks).
  if (path_ == BlockPath::kFusedGraphs && !graphs_paused_ && !blocks_->residency().streamed() && stream != nullptr &&
      capturable && ref_tokens == 0) {
    key.height = height;
    key.width = width;
    key.text_lengths = lengths;
    for (const Row& r : rows) key.skips.push_back(r.skip);
    key.addresses = {a.x.data, a.y.data, a.big.data, a.rope.data, a.mod.data, a.cu_seqlens.data, a.lse, gemm_workspace_};
    if (quantized) {
      key.addresses.push_back(qa.scale.data);
      key.addresses.push_back(qa.attn_workspace);
      key.addresses.push_back(qa.lora_t.data);
      key.addresses.push_back(qa.lora_workspace);
      key.addresses.push_back(qa.int4_unpack);
      key.addresses.push_back(qa.nvfp4_sf);
      key.addresses.push_back(qa.nvfp4_delta.data);
      key.addresses.push_back(qa.nvfp4_scratch);
    }
    // The block variant, and the LoRA side branches' generation (their set of active sites changes the kernels).
    key.variant = quantized ? 1 + static_cast<std::uint64_t>(config_.attention) : 0;
    if (config_.int4_via_int8) key.variant |= std::uint64_t{1} << 7;
    if (qa.lora != nullptr) key.variant |= qa.lora->generation() << 8;
    key.route_generation = route.generation;
    const std::uint64_t h = key.hash();
    use_graphs = h == last_step_hash_;
    last_step_hash_ = h;
  }

  // Step caching (StepCacheHook): a reuse after block 0 ends the loop and the streaming pass there.
  if (step_cache_ != nullptr) step_cache_->before_blocks(a.x, rows, off, hw, stream);
  bool reused = false;
  for (int bi = 0; bi < kLayers && !reused; ++bi) {
    if (cancel_ != nullptr) {
      // Issue block bi only once block bi - 2 has finished: at most one block is queued ahead of the running one.
      if (bi >= 2) MI_CUDA_CHECK(cudaEventSynchronize(block_done_[bi % 2]));
      if (cancel_->load()) {
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        throw Cancelled();  // `weights` stops the readers and waits for their copies
      }
    }
    const Block& b = weights.acquire(bi);
    const std::vector<Run>& runs = block_runs[static_cast<std::size_t>(bi)];
    if (ref_tokens > 0) {
      // Block bi's reference keys and values into the k | v columns of the slots its run reads.
      const std::uint16_t* cached = static_cast<const std::uint16_t*>(refs_->kv.data) +
                                    static_cast<std::size_t>(bi) * static_cast<std::size_t>(ref_tokens) * 2 * kDim;
      for (const int s : {1, 2}) {
        if (!(slot_use[static_cast<std::size_t>(bi)] & s)) continue;
        const std::int64_t row0 = s == 1 ? 0 : ref_tokens + total;
        MI_CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::uint16_t*>(a.big.data) + row0 * kQkvDim + kDim,
                                        kQkvDim * sizeof(std::uint16_t), cached, 2 * row_bytes, 2 * row_bytes,
                                        static_cast<std::size_t>(ref_tokens), cudaMemcpyDeviceToDevice, stream));
      }
    }
    if (!use_graphs) {
      run_blocks(bi, b, runs, stream);
    } else {
      key.block_begin = bi;
      key.block_end = bi + 1;
      graphs_.run(key, stream, [&](cudaStream_t s) { run_blocks(bi, b, runs, s); });
    }
    weights.release(bi);
    if (hook) hook(bi, a.x);
    if (cancel_ != nullptr) MI_CUDA_CHECK(cudaEventRecord(block_done_[bi % 2], stream));
    if (bi == 0 && step_cache_ != nullptr) reused = step_cache_->after_first_block(a.x, rows, off, hw, stream);
  }
  if (reused) {
    weights.stop_early();  // stops the readers, waits for their copies (the lookahead is not used), rethrows their error
  } else {
    weights.finish();
  }
  if (step_cache_ != nullptr) step_cache_->after_blocks(a.x, rows, off, hw, reused, stream);

  // Head (P:256-259): final norm with head modulation (shift = head[0], scale = head[1]), linear, transpose.
  const Tensor proj = Tensor::view(a.big.data, DType::kBF16, Device::kCUDA, {hw, kLatentChannels});
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const Tensor xv = a.x.slice0(off[r], off[r] + hw), yv = a.y.slice0(off[r], off[r] + hw);
    kernels::rms_norm_modulate_moss(xv, final_norm_, vec(a.head, 1), vec(a.head, 0), yv, stream);
    linear(proj, yv, final_linear_, stream);
    tokens_to_latent(proj, out[r], stream);
  }
}

void MossDit::prefill_references(std::span<const Tensor> latents, const RefKv& out, cudaStream_t stream) {
  if (latents.empty()) fail("prefill_references needs at least one reference latent");
  std::int64_t total = 0;
  for (const Tensor& l : latents) {
    check_device(l, DType::kF32, "reference latent");
    if (l.rank != 3 || l.dim(0) != kLatentChannels || l.dim(1) < 1 || l.dim(2) < 1) {
      fail("a reference latent must be [128, h, w], got " + l.describe());
    }
    total += l.dim(1) * l.dim(2);
    if (total > kMaxReferenceTokens) fail("reference images exceed " + std::to_string(kMaxReferenceTokens) + " tokens");
  }
  check_device(out.kv, DType::kBF16, "reference cache");
  if (out.kv.rank != 3 || out.kv.dim(0) != kLayers || out.kv.dim(1) != total || out.kv.dim(2) != 2 * kDim) {
    fail("the reference cache must be [42, " + std::to_string(total) + ", 6144], got " + out.kv.describe());
  }
  if (linear_input_hook_) throw std::logic_error("moss_dit: prefill_references does not run with a linear-input hook");

  // forward()'s setup for one sequence of `total` tokens without text (prefill_workspace_bytes' allocations).
  if (lora_ != nullptr) refresh_config(config_.attention);
  const FusedRouteSnapshot route = path_ == BlockPath::kOpByOp ? FusedRouteSnapshot{}
                                   : request_route_              ? *request_route_
                                                                 : fused_route_snapshot();
  const bool quantized = config_.quantized;
  const std::int64_t none[] = {0};
  runtime::StackScope scope(arena_);
  StackPlan plan(&arena_);
  Activations a(plan, total, total, 1, true,
                quantized ? quant_big_cols(total, none, config_.bf16_up_gate, route, nvfp4_cols(config_))
                          : big_cols(path_, total, none, route));
  QuantBlockBuffers qa;
  if (quantized) qa = quant_buffers(plan, config_, total, q8_attention_workspace_bytes(total, none, config_.attention), lora_);
  BlockWeights::Pass weights(*blocks_, stream, cancel_);

  const std::int32_t cu[] = {0, static_cast<std::int32_t>(total)};
  write_params(a.cu_seqlens.data, cu, sizeof(cu), stream);
  Run run;
  run.t0 = 0;
  run.t1 = total;
  run.cu_seqlens = a.cu_seqlens.slice0(0, 2);
  run.max_seqlen = static_cast<int>(total);

  // The t = 0 modulation (ComfyUI preprocess_refs: timestep_embedding(zeros)), each reference's RoPE table and video
  // embedding (an input extension's channels zero: [latents | 0] through its embedder).
  time_modulation(0.0f, a.mod, a.head, stream);
  std::int64_t at = 0;
  for (std::size_t i = 0; i < latents.size(); ++i) {
    const Tensor& l = latents[i];
    const std::int64_t n = l.dim(1) * l.dim(2);
    kernels::moss_rope_table_reference(a.rope.slice0(at, at + n), static_cast<int>(l.dim(1)), static_cast<int>(l.dim(2)),
                                       static_cast<int>(i), stream);
    if (input_ext_ == nullptr) {
      embed_video(gemm_, gemm_workspace_, kernels::Gemm::kWorkspaceBytes, l, nullptr, a.big.data, video_embedder_.w,
                  video_embedder_.bias, a.x.slice0(at, at + n), stream);
    } else {
      const std::int64_t cin = input_ext_->weight.dim(1);
      const Tensor t = Tensor::view(a.big.data, DType::kBF16, Device::kCUDA, {n, cin});
      latent_to_tokens(l, t, 0, stream);
      MI_CUDA_CHECK(cudaMemset2DAsync(static_cast<std::uint16_t*>(a.big.data) + kLatentChannels,
                                      static_cast<std::size_t>(cin) * sizeof(std::uint16_t), 0,
                                      static_cast<std::size_t>(cin - kLatentChannels) * sizeof(std::uint16_t),
                                      static_cast<std::size_t>(n), stream));
      const Tensor x = a.x.slice0(at, at + n);
      gemm_.linear(x, t, input_ext_->weight, &input_ext_->bias, gemm_workspace_, kernels::Gemm::kWorkspaceBytes, stream);
    }
    at += n;
  }

  const FusedBlockBuffers fused{a.x, a.y, a.big, a.rope, a.mod, a.lse, a.lse_bytes};
  qa.f = fused;
  const QkvGemm qkv_gemm{&gemm_, gemm_workspace_, kernels::Gemm::kWorkspaceBytes};
  for (int bi = 0; bi < kLayers; ++bi) {
    if (cancel_ != nullptr) {
      if (bi >= 2) MI_CUDA_CHECK(cudaEventSynchronize(block_done_[bi % 2]));
      if (cancel_->load()) {
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        throw Cancelled();
      }
    }
    const Block& b = weights.acquire(bi);
    Run tap = run;
    tap.kv_tap = static_cast<std::uint16_t*>(out.kv.data) +
                 static_cast<std::size_t>(bi) * static_cast<std::size_t>(total) * 2 * kDim;
    tap.tap_only = bi == kLayers - 1;  // the last block's keys and values are all the cache needs of it
    if (quantized) {
      run_block_quant(b, qa, tap, qkv_gemm, route, config_.attention, stream, bi);
    } else if (path_ == BlockPath::kOpByOp) {
      run_block(bi, b, a, tap, stream);
    } else {
      run_block_fused(b, fused, tap, qkv_gemm, route, stream);
    }
    weights.release(bi);
    if (cancel_ != nullptr) MI_CUDA_CHECK(cudaEventRecord(block_done_[bi % 2], stream));
  }
  weights.finish();
}

}  // namespace mi::moss_dit
