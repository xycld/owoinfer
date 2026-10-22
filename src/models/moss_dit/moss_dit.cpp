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
  void* lse = nullptr;
  std::size_t lse_bytes = 0;
  // step only: bf16 [6*kDim] block modulation, [2*kDim] head modulation.
  Tensor mod, head;

  Activations(StackPlan& p, std::int64_t tokens, std::int64_t rope_rows, std::int64_t rows, bool step,
              std::int64_t big_cols = kBigCols) {
    if (step) x = p.tensor(DType::kBF16, {tokens, kDim});
    y = p.tensor(DType::kBF16, {tokens, kDim});
    big = p.tensor(DType::kBF16, {tokens, big_cols});
    auto* b = static_cast<std::uint16_t*>(big.data);
    qkv = Tensor::view(b, DType::kBF16, Device::kCUDA, {tokens, kQkvDim});
    attn = Tensor::view(b + tokens * kQkvDim, DType::kBF16, Device::kCUDA, {tokens, kDim});
    if (big_cols >= kBigCols) {
      up_gate = Tensor::view(b, DType::kBF16, Device::kCUDA, {tokens, kUpGateDim});
      ffn = Tensor::view(b + tokens * kUpGateDim, DType::kBF16, Device::kCUDA, {tokens, kFfnDim});
    }
    rope = p.tensor(DType::kBF16, {rope_rows, kernels::kMossRopeDim});
    cu_seqlens = p.tensor(DType::kI32, {cu_pool_ints(rows)});
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
                 bool direct_io, const std::atomic<bool>* cancel, const Residency& residency)
    : arena_(arena), cancel_(cancel) {
  const auto t0 = std::chrono::steady_clock::now();
  gemm_workspace_ = arena.alloc_persistent(kernels::Gemm::kWorkspaceBytes);
  weights::ReaderCache readers(index, direct_io);
  runtime::PinnedRing ring(streams.copy());
  const auto check_cancel = [&] {
    if (cancel_ != nullptr && cancel_->load()) {
      ring.synchronize();  // in-flight copies target the arena the caller will recreate
      throw Cancelled();
    }
  };
  const auto upload = [&](const WeightGroup& g) {
    check_cancel();
    load_stats_.bytes += g.total_bytes;
    return weights::upload_group(readers, g, arena, ring);
  };

  const TensorMap glob = upload(*find_group(index, "globals"));
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
  });
  host_bytes_ = BlockWeights::host_bytes(index, residency);
  ring.synchronize();
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

void MossDit::time_modulation(float timestep, const Tensor& mod, const Tensor& head, cudaStream_t stream) {
  check_device(mod, DType::kBF16, "mod");
  check_device(head, DType::kBF16, "head");
  if (mod.numel() != 6 * kDim || head.numel() != 2 * kDim) fail("mod/head must hold 6*3072 / 2*3072 values");
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

std::size_t MossDit::workspace_bytes(std::int64_t height, std::int64_t width, std::span<const std::int64_t> text_lengths,
                                     BlockPath path) {
  const std::int64_t total = token_offsets(height, width, text_lengths).back();
  const FusedRouteSnapshot route = path == BlockPath::kOpByOp ? FusedRouteSnapshot{} : fused_route_snapshot();
  StackPlan plan(nullptr);
  Activations(plan, total, total, static_cast<std::int64_t>(text_lengths.size()), true,
              big_cols(path, height * width, text_lengths, route));
  TimeScratch{plan};
  return plan.total();
}

void MossDit::run_block(const Block& b, const Activations& a, const Run& run, cudaStream_t stream) {
  // _Block.forward (P:138-148) on tokens [t0, t1); mod chunks (P:139): shift_a, scale_a, gate_a, shift_f, scale_f, gate_f.
  const Tensor& mod = a.mod;
  const Tensor x = a.x.slice0(run.t0, run.t1), y = a.y.slice0(run.t0, run.t1);
  const Tensor qkv = a.qkv.slice0(run.t0, run.t1);
  const Tensor q = heads_view(qkv, 0), k = heads_view(qkv, kDim), v = heads_view(qkv, 2 * kDim);
  const Tensor rope = a.rope.slice0(run.t0, run.t1);

  kernels::rms_norm_modulate_moss(x, b.norm1, vec(mod, 1), vec(mod, 0), y, stream);
  linear(qkv, y, b.qkv, stream);
  kernels::rms_norm_moss(q, b.q_norm, q, stream);
  kernels::rms_norm_moss(k, b.k_norm, k, stream);
  kernels::moss_rope_apply(q, rope, q, stream);
  kernels::moss_rope_apply(k, rope, k, stream);
  const Tensor attn = a.attn.slice0(run.t0, run.t1);
  attention::flash_fwd(q, k, v, heads_view(attn, 0), run.cu_seqlens, run.cu_seqlens, run.max_seqlen, false,
                       kAttnScale, a.lse, a.lse_bytes, stream);
  linear(y, attn, b.o, stream);
  kernels::gated_residual(x, y, vec(mod, 2), x, stream);
  kernels::rms_norm_modulate_moss(x, b.norm2, vec(mod, 4), vec(mod, 3), y, stream);
  const Tensor up_gate = a.up_gate.slice0(run.t0, run.t1), ffn = a.ffn.slice0(run.t0, run.t1);
  linear(up_gate, y, b.up_gate, stream);
  kernels::situ_glu(up_gate, ffn, stream);
  linear(y, ffn, b.down, stream);
  kernels::gated_residual(x, y, vec(mod, 5), x, stream);
}

void MossDit::forward(const Tensor& latents, float timestep, std::span<const Row> rows, std::span<const Tensor> out,
                      cudaStream_t stream, const BlockHook& hook) {
  check_device(latents, DType::kF32, "latents");
  if (latents.rank != 3 || latents.dim(0) != kLatentChannels) fail("latents must be [128, h, w], got " + latents.describe());
  const std::int64_t height = latents.dim(1), width = latents.dim(2), hw = height * width;
  if (rows.empty() || out.size() != rows.size()) fail("forward needs >= 1 row and one output per row");
  if (rows.size() > static_cast<std::size_t>(kMaxRows)) fail("at most " + std::to_string(kMaxRows) + " rows per forward");
  std::vector<std::int64_t> lengths;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const Tensor& t = rows[r].text;
    check_device(t, DType::kBF16, "row text");
    if (t.rank != 2 || t.dim(0) < 1 || t.dim(1) != kDim) fail("row text must be [S >= 1, 3072], got " + t.describe());
    check_device(out[r], DType::kF32, "out");
    if (out[r].rank != 3 || out[r].dim(0) != kLatentChannels || out[r].dim(1) != height || out[r].dim(2) != width) {
      fail("out must match latents [128, h, w], got " + out[r].describe());
    }
    lengths.push_back(t.dim(0));
  }
  const std::vector<std::int64_t> off = token_offsets(height, width, lengths);
  const std::int64_t total = off.back();
  if (total > INT32_MAX) fail("too many tokens");

  // One route for the whole forward (its big buffer, graph keys and every block): an install_fused_check meanwhile, e.g.
  // another pipeline's poisoning, applies from the next forward on instead of failing this one's buffer check.
  const FusedRouteSnapshot route = path_ == BlockPath::kOpByOp ? FusedRouteSnapshot{} : fused_route_snapshot();
  runtime::StackScope scope(arena_);
  StackPlan plan(&arena_);
  Activations a(plan, total, total, static_cast<std::int64_t>(rows.size()), true, big_cols(path_, hw, lengths, route));
  // Streamed blocks: device buffers after the activations; the first copies overlap the setup below.
  BlockWeights::Pass weights(*blocks_, stream, cancel_);

  // Each maximal run of contiguous rows that executes a block is one varlen sub-batch with its own
  // zero-based cu_seqlens (flash_fwd's contract); without skips that is one run over all rows.
  std::vector<std::vector<Run>> block_runs(kLayers);
  {
    std::map<std::pair<std::size_t, std::size_t>, Run> unique;
    std::vector<std::int32_t> cu;
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
          const Run run{off[r0], off[r1], a.cu_seqlens.slice0(at, at + static_cast<std::int64_t>(r1 - r0) + 1),
                        static_cast<int>(max_len)};
          it = unique.emplace(std::make_pair(r0, r1), run).first;
        }
        block_runs[static_cast<std::size_t>(bi)].push_back(it->second);
        r0 = r1;
      }
    }
    write_params(a.cu_seqlens.data, cu.data(), cu.size() * sizeof(std::int32_t), stream);
    if (varlen_debug_checks()) {
      for (const auto& [key, run] : unique) {
        validate_cu_seqlens(run.cu_seqlens, run.max_seqlen, run.t1 - run.t0, stream, "moss_dit::forward");
      }
    }
  }

  time_modulation(timestep, a.mod, a.head, stream);

  // Per-row RoPE tables (P:250, text length differs per row) and the joint inputs [video_r | text_r] (P:247-252).
  // The video embedding is the same for every row: computed once, copied to the other rows.
  const Tensor tokens = Tensor::view(a.big.data, DType::kBF16, Device::kCUDA, {hw, kLatentChannels});
  latent_to_tokens(latents, tokens, stream);
  linear(a.x.slice0(0, hw), tokens, video_embedder_, stream);
  const std::size_t row_bytes = kDim * sizeof(std::uint16_t);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    kernels::moss_rope_table(a.rope.slice0(off[r], off[r + 1]), static_cast<int>(height), static_cast<int>(width),
                             static_cast<int>(lengths[r]), stream);
    if (r > 0) {
      MI_CUDA_CHECK(cudaMemcpyAsync(a.x.slice0(off[r], off[r] + hw).data, a.x.data, static_cast<std::size_t>(hw) * row_bytes,
                                    cudaMemcpyDeviceToDevice, stream));
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
  const QkvGemm qkv_gemm{&gemm_, gemm_workspace_, kernels::Gemm::kWorkspaceBytes};
  StepGraphKey key;
  bool use_graphs = false;
  if (path_ == BlockPath::kFusedGraphs && !blocks_->residency().streamed() && stream != nullptr) {
    key.height = height;
    key.width = width;
    key.text_lengths = lengths;
    for (const Row& r : rows) key.skips.push_back(r.skip);
    key.addresses = {a.x.data, a.y.data, a.big.data, a.rope.data, a.mod.data, a.cu_seqlens.data, a.lse, gemm_workspace_};
    key.route_generation = route.generation;
    const std::uint64_t h = key.hash();
    use_graphs = h == last_step_hash_;
    last_step_hash_ = h;
  }

  for (int bi = 0; bi < kLayers; ++bi) {
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
    if (path_ == BlockPath::kOpByOp) {
      for (const Run& run : runs) run_block(b, a, run, stream);
    } else if (!use_graphs) {
      for (const Run& run : runs) run_block_fused(b, fused, run, qkv_gemm, route, stream);
    } else {
      key.block_begin = bi;
      key.block_end = bi + 1;
      graphs_.run(key, stream, [&](cudaStream_t s) {
        for (const Run& run : runs) run_block_fused(b, fused, run, qkv_gemm, route, s);
      });
    }
    weights.release(bi);
    if (hook) hook(bi, a.x);
    if (cancel_ != nullptr) MI_CUDA_CHECK(cudaEventRecord(block_done_[bi % 2], stream));
  }
  weights.finish();

  // Head (P:256-259): final norm with head modulation (shift = head[0], scale = head[1]), linear, transpose.
  const Tensor proj = Tensor::view(a.big.data, DType::kBF16, Device::kCUDA, {hw, kLatentChannels});
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const Tensor xv = a.x.slice0(off[r], off[r] + hw), yv = a.y.slice0(off[r], off[r] + hw);
    kernels::rms_norm_modulate_moss(xv, final_norm_, vec(a.head, 1), vec(a.head, 0), yv, stream);
    linear(proj, yv, final_linear_, stream);
    tokens_to_latent(proj, out[r], stream);
  }
}

}  // namespace mi::moss_dit
