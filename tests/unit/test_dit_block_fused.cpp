// P5: the fused DiT block (models/moss_dit/dit_block_fused) and step graphs (dit_graph) on the real weights.
//
// Exactness: the engine (MossDit::forward, BlockPath::kOpByOp) runs the golden 1024^2 CFG step (cond + uncond with
// skip {22}) and a 282-token-prompt variant, and its residual stream is captured after several blocks; its other block
// paths (fused, fused + CUDA graphs) must give the same velocities bit for bit. Then, with every block resident in a
// second arena, each captured input is pushed through (a) a test-local copy of MossDit::run_block and (b)
// run_block_fused, and both must reproduce the engine's next residual stream bit for bit, also through 41 chained blocks.
// Speed: per-op, per-block and 42-block-step timings of op-by-op vs fused vs fused + CUDA graphs, interleaved A/B in
// one process (the laptop GPU's clock drifts with CPU load, docs/experiment-log.md E10), at 1024^2 (real activations)
// and 1456^2 + 4096 tokens (synthetic activations).
#include <gtest/gtest.h>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/fused/qk_norm_rope.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/dit_graph.h"
#include "models/moss_dit/moss_dit.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
namespace md = mi::moss_dit;
using mi::DType;
using mi::Tensor;

constexpr std::int64_t kDim = md::kDim, kQkv = 3 * md::kDim, kUpGate = 2 * md::kFfnDim;
constexpr std::int64_t kOpBigCols = 3 * md::kFfnDim;  // MossDit's [up_gate | ffn] (>= [qkv | attn])
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
const float kAttnScale = static_cast<float>(1.0 / std::sqrt(128.0));
const char* const kCase = "g_fridge_1024";  // golden case (its latents and timestep also feed the 282-token bench run)
// Blocks after which the engine's residual stream is captured: inputs/outputs of blocks 1, 21, 22 (uncond skips it),
// 23 and 41, and the 41-block chain 1..41.
constexpr int kCaptured[] = {0, 1, 20, 21, 22, 23, 40, 41};

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path golden_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "golden"; }

std::uint16_t to_bf16(float v) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(v);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}
float from_bf16(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

std::vector<std::uint16_t> download(const Tensor& t) {
  std::vector<std::uint16_t> h(static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, h.size() * 2, cudaMemcpyDeviceToHost));
  return h;
}
void upload(const Tensor& t, const std::vector<std::uint16_t>& h) {
  ASSERT_EQ(h.size(), static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(t.data, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
}

struct Compare {
  std::size_t differ = 0;
  double cosine = 0, max_abs = 0;
};
Compare compare(const std::vector<std::uint16_t>& ref, const std::vector<std::uint16_t>& got) {
  Compare c;
  double dot = 0, rr = 0, gg = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = from_bf16(ref[i]), g = from_bf16(got[i]);
    c.differ += ref[i] != got[i];
    dot += r * g;
    rr += r * r;
    gg += g * g;
    c.max_abs = std::max(c.max_abs, std::abs(r - g));
  }
  c.cosine = dot / std::sqrt(rr * gg);
  return c;
}

// splitmix64-hashed bf16 values +-m * 2^(scale_exp - 7 - e), m in [128, 255], e in 0..3.
std::vector<std::uint16_t> synth_bf16(std::uint64_t seed, std::size_t n, int scale_exp) {
  std::vector<std::uint16_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = i + seed * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float m = static_cast<float>(128 + (z & 0x7F));
    const float v = std::ldexp((z >> 10) & 1 ? -m : m, scale_exp - 7 - static_cast<int>((z >> 7) & 3));
    out[i] = to_bf16(v);
  }
  return out;
}

Tensor view(void* p, std::initializer_list<std::int64_t> shape) { return Tensor::view(p, DType::kBF16, mi::Device::kCUDA, shape); }
Tensor vec(const Tensor& mod, std::int64_t i) {
  return view(static_cast<std::uint16_t*>(mod.data) + i * kDim, {kDim});
}
Tensor heads(const Tensor& buf, std::int64_t col) {
  Tensor t = Tensor::view(static_cast<std::uint16_t*>(buf.data) + col, DType::kBF16, mi::Device::kCUDA,
                          {buf.dim(0), md::kHeads, md::kHeadDim});
  t.stride[0] = buf.stride[0];
  return t;
}

// The varlen runs of every block for rows with these text lengths and skip sets (MossDit::forward's construction).
struct StepRuns {
  std::vector<std::vector<md::BlockRun>> per_block;
  std::int64_t total = 0;
  std::vector<std::int64_t> off;
};
StepRuns make_runs(std::int64_t hw, const std::vector<std::int64_t>& lengths, const std::vector<md::SkipSet>& skips,
                   const Tensor& cu_pool) {
  StepRuns s;
  s.off = {0};
  for (std::int64_t l : lengths) s.off.push_back(s.off.back() + hw + l);
  s.total = s.off.back();
  std::map<std::pair<std::size_t, std::size_t>, md::BlockRun> unique;
  std::vector<std::int32_t> cu;
  s.per_block.resize(md::kLayers);
  for (int bi = 0; bi < md::kLayers; ++bi) {
    const auto skip = [&](std::size_t r) { return skips[r].test(static_cast<std::size_t>(bi)); };
    for (std::size_t r0 = 0; r0 < lengths.size();) {
      if (skip(r0)) {
        ++r0;
        continue;
      }
      std::size_t r1 = r0;
      while (r1 < lengths.size() && !skip(r1)) ++r1;
      auto it = unique.find({r0, r1});
      if (it == unique.end()) {
        const auto at = static_cast<std::int64_t>(cu.size());
        std::int64_t max_len = 0;
        for (std::size_t r = r0; r <= r1; ++r) {
          cu.push_back(static_cast<std::int32_t>(s.off[r] - s.off[r0]));
          if (r < r1) max_len = std::max(max_len, s.off[r + 1] - s.off[r]);
        }
        md::BlockRun run{s.off[r0], s.off[r1], cu_pool.slice0(at, at + static_cast<std::int64_t>(r1 - r0) + 1),
                         static_cast<int>(max_len)};
        it = unique.emplace(std::make_pair(r0, r1), run).first;
      }
      s.per_block[static_cast<std::size_t>(bi)].push_back(it->second);
      r0 = r1;
    }
  }
  MI_CUDA_CHECK(cudaMemcpy(cu_pool.data, cu.data(), cu.size() * 4, cudaMemcpyHostToDevice));
  return s;
}

// Device activations of a step over `total` tokens, carved from the arena stack. `big` is MossDit's buffer, viewed as
// contiguous [qkv | attn] then [up_gate | ffn] (op-by-op) or [ffn] (fused; it needs only total * kFusedBigCols).
struct Acts {
  Tensor x, y, big, rope, mod, cu_pool;
  void* lse = nullptr;
  std::size_t lse_bytes = 0;
  std::int64_t total = 0;

  static std::size_t bytes(std::int64_t total) {
    using mi::runtime::align_up;
    return 2 * align_up(static_cast<std::size_t>(total * kDim * 2)) + align_up(static_cast<std::size_t>(total * kOpBigCols * 2)) +
           align_up(static_cast<std::size_t>(total * 128 * 2)) + align_up(6 * kDim * 2) + align_up(1024) +
           align_up(mi::attention::flash_fwd_workspace_bytes(total, md::kHeads));
  }
  Acts(mi::runtime::DeviceArena& arena, std::int64_t tokens) : total(tokens) {
    x = arena.alloc_tensor(DType::kBF16, {total, kDim});
    y = arena.alloc_tensor(DType::kBF16, {total, kDim});
    big = arena.alloc_tensor(DType::kBF16, {total, kOpBigCols});
    rope = arena.alloc_tensor(DType::kBF16, {total, 128});
    mod = arena.alloc_tensor(DType::kBF16, {6 * kDim});
    cu_pool = arena.alloc_tensor(DType::kI32, {256});
    lse_bytes = mi::attention::flash_fwd_workspace_bytes(total, md::kHeads);
    lse = arena.alloc_stack(lse_bytes);
  }
  md::FusedBlockBuffers fused() const {
    return {x, y, big, rope, mod, lse, lse_bytes};
  }
};

// Ops of one block, op by op exactly as MossDit::run_block (same calls, same buffer layout and strides).
struct OpByOp {
  mi::kernels::Gemm& gemm;
  void* ws;
  void linear(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t s) const {
    gemm.linear(out, x, w, nullptr, ws, mi::kernels::Gemm::kWorkspaceBytes, s);
  }
  // `mark(i)` is called before op i (and after the last op with i = kOps) for per-op timing.
  static constexpr int kOps = 14;
  void run(const md::BlockTensors& b, const Acts& a, const md::BlockRun& run, cudaStream_t s,
           const std::function<void(int)>& mark = {}) const {
    const auto m = [&](int i) {
      if (mark) mark(i);
    };
    auto* big = static_cast<std::uint16_t*>(a.big.data);
    const Tensor qkv = view(big, {a.total, kQkv}).slice0(run.t0, run.t1);
    const Tensor attn = view(big + a.total * kQkv, {a.total, kDim}).slice0(run.t0, run.t1);
    const Tensor up_gate = view(big, {a.total, kUpGate}).slice0(run.t0, run.t1);
    const Tensor ffn = view(big + a.total * kUpGate, {a.total, md::kFfnDim}).slice0(run.t0, run.t1);
    const Tensor x = a.x.slice0(run.t0, run.t1), y = a.y.slice0(run.t0, run.t1), rope = a.rope.slice0(run.t0, run.t1);
    const Tensor q = heads(qkv, 0), k = heads(qkv, kDim), v = heads(qkv, 2 * kDim);
    m(0);
    mi::kernels::rms_norm_modulate_moss(x, b.norm1, vec(a.mod, 1), vec(a.mod, 0), y, s);
    m(1);
    linear(qkv, y, b.qkv, s);
    m(2);
    mi::kernels::rms_norm_moss(q, b.q_norm, q, s);
    m(3);
    mi::kernels::rms_norm_moss(k, b.k_norm, k, s);
    m(4);
    mi::kernels::moss_rope_apply(q, rope, q, s);
    m(5);
    mi::kernels::moss_rope_apply(k, rope, k, s);
    m(6);
    mi::attention::flash_fwd(q, k, v, heads(attn, 0), run.cu_seqlens, run.cu_seqlens, run.max_seqlen, false, kAttnScale,
                             a.lse, a.lse_bytes, s);
    m(7);
    linear(y, attn, b.o, s);
    m(8);
    mi::kernels::gated_residual(x, y, vec(a.mod, 2), x, s);
    m(9);
    mi::kernels::rms_norm_modulate_moss(x, b.norm2, vec(a.mod, 4), vec(a.mod, 3), y, s);
    m(10);
    linear(up_gate, y, b.up_gate, s);
    m(11);
    mi::kernels::situ_glu(up_gate, ffn, s);
    m(12);
    linear(y, ffn, b.down, s);
    m(13);
    mi::kernels::gated_residual(x, y, vec(a.mod, 5), x, s);
    m(14);
  }
};
const char* const kOpNames[OpByOp::kOps] = {"norm1+modulate", "qkv gemm",   "q norm",          "k norm",    "q rope",
                                            "k rope",         "attention",  "o gemm",          "gated_res", "norm2+modulate",
                                            "up_gate gemm",   "situ_glu",   "down gemm",       "gated_res"};

// The fused block's ops, split as run_block_fused (for per-op timing; the test checks it equals run_block_fused).
constexpr int kFusedOps = 8;
const char* const kFusedOpNames[kFusedOps] = {"norm1+modulate",   "qkv gemm",       "qk_norm_rope",      "attention",
                                              "o gemm+gated_res", "norm2+modulate", "up_gate gemm+situ", "down gemm+gated_res"};
void fused_ops(const md::BlockTensors& b, const md::FusedBlockBuffers& a, const md::BlockRun& run, const md::QkvGemm& g,
               cudaStream_t s, const std::function<void(int)>& mark) {
  const std::int64_t l = a.x.dim(0);
  auto* big = static_cast<std::uint16_t*>(a.big.data);
  const Tensor qkv = view(big, {l, kQkv}).slice0(run.t0, run.t1);
  const Tensor attn = view(big + l * kQkv, {l, kDim}).slice0(run.t0, run.t1);
  const Tensor ffn = view(big, {l, md::kFfnDim}).slice0(run.t0, run.t1);
  const Tensor x = a.x.slice0(run.t0, run.t1), y = a.y.slice0(run.t0, run.t1);
  mark(0);
  mi::kernels::rms_norm_modulate_moss(x, b.norm1, vec(a.mod, 1), vec(a.mod, 0), y, s);
  mark(1);
  g.gemm->linear(qkv, y, b.qkv, nullptr, g.workspace, g.workspace_bytes, s);
  mark(2);
  mi::kernels::fused::qk_norm_rope_moss(qkv, md::kHeads, b.q_norm, b.k_norm, a.rope.slice0(run.t0, run.t1), s);
  mark(3);
  mi::attention::flash_fwd(heads(qkv, 0), heads(qkv, kDim), heads(qkv, 2 * kDim), heads(attn, 0), run.cu_seqlens,
                           run.cu_seqlens, run.max_seqlen, false, kAttnScale, a.lse, a.lse_bytes, s);
  mark(4);
  mi::kernels::fused::gemm_gated_residual(x, attn, b.o, vec(a.mod, 2), s);
  mark(5);
  mi::kernels::rms_norm_modulate_moss(x, b.norm2, vec(a.mod, 4), vec(a.mod, 3), y, s);
  mark(6);
  mi::kernels::fused::gemm_situ_glu(ffn, y, b.up_gate, s);
  mark(7);
  mi::kernels::fused::gemm_gated_residual(x, ffn, b.down, vec(a.mod, 5), s);
  mark(8);
}

// Event-timed GPU work on a stream.
class Timer {
 public:
  explicit Timer(cudaStream_t s) : s_(s) {
    MI_CUDA_CHECK(cudaEventCreate(&e0_));
    MI_CUDA_CHECK(cudaEventCreate(&e1_));
  }
  ~Timer() {
    cudaEventDestroy(e0_);
    cudaEventDestroy(e1_);
  }
  double ms(const std::function<void()>& fn) {
    MI_CUDA_CHECK(cudaEventRecord(e0_, s_));
    fn();
    MI_CUDA_CHECK(cudaEventRecord(e1_, s_));
    MI_CUDA_CHECK(cudaEventSynchronize(e1_));
    float v = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&v, e0_, e1_));
    return static_cast<double>(v);
  }

 private:
  cudaStream_t s_;
  cudaEvent_t e0_ = nullptr, e1_ = nullptr;
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Host load for the record (CPU load lowers the laptop GPU's clocks, E10).
std::string host_load() {
  double load = 0;
  std::ifstream in("/proc/loadavg");
  in >> load;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "host loadavg %.2f", load);
  return buf;
}

// One engine step's inputs and its residual stream after the kCaptured blocks.
struct EngineRun {
  const char* name = "";
  std::int64_t h = 0, w = 0;
  std::vector<std::int64_t> lengths;
  std::vector<md::SkipSet> skips;
  std::map<int, std::vector<std::uint16_t>> x_after;  // block -> residual stream of all rows after it
};

// The fused block runs a pair fused only in configurations a passed self-check installed (as the pipeline does).
void install_self_check(cudaStream_t stream) {
  const md::FusedCheck c = md::check_fused_gemms(stream);
  EXPECT_TRUE(c.ok) << c.detail;
  md::install_fused_check(c);
}

class DitBlockFused : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::is_regular_file(bundle_dir() / "moss.mossidx") || !fs::is_directory(golden_dir() / kCase) ||
        !fs::is_regular_file(golden_dir() / "g_long4096_1024.cond.te_last_hidden.npy")) {
      return;
    }
    index_ = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir()));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    install_self_check(stream());
    capture_engine();
    load_blocks();
  }
  static void TearDownTestSuite() {
    blocks_.reset();
    gemm_.reset();
    arena_.reset();
    streams_.reset();
    index_.reset();
  }
  void SetUp() override {
    if (!blocks_) GTEST_SKIP() << "weights or goldens not present (" << bundle_dir() << ")";
  }
  static cudaStream_t stream() { return streams_->compute(); }
  static md::QkvGemm qkv_gemm() { return {gemm_.get(), gemm_ws_, mi::kernels::Gemm::kWorkspaceBytes}; }

  // Phase 1: the engine (MossDit::forward) on two inputs sharing the golden latents and timestep:
  //   golden_: the golden case as recorded (cond S = 62, uncond S = 10, uncond skips block 22);
  //   bench_:  AC9b's prompt length, cond = the first 282 tokens of the 4096-token golden prompt's encoder output.
  static void capture_engine() {
    const fs::path g = golden_dir();
    const auto noise = mi::npy::load(g / kCase / "initial_noise.npy");
    const std::int64_t h = noise.shape[2], w = noise.shape[3];
    const float timestep = static_cast<float>(mi::npy::load(g / kCase / "timestep0.npy").to_vector<double>().at(0));
    const auto te_c = mi::npy::load(g / (std::string(kCase) + ".cond.te_last_hidden.npy"));
    const auto te_u = mi::npy::load(g / (std::string(kCase) + ".uncond.te_last_hidden.npy"));
    const auto te_long = mi::npy::load(g / "g_long4096_1024.cond.te_last_hidden.npy");
    constexpr std::int64_t kBenchCond = 282;
    md::SkipSet uncond_skip;
    {
      std::ifstream in(g / kCase / "dit_step0_uncond.skip.json");
      std::stringstream ss;
      ss << in.rdbuf();
      const mi::json::Value doc = mi::json::parse(ss.str());
      for (const auto& v : doc.as_array()) uncond_skip.set(static_cast<std::size_t>(v.as_int()));
    }
    golden_ = {kCase, h, w, {te_c.shape[0], te_u.shape[0]}, {md::SkipSet{}, uncond_skip}, {}};
    bench_ = {"1024^2 bench", h, w, {kBenchCond, te_u.shape[0]}, {md::SkipSet{}, uncond_skip}, {}};

    const std::int64_t s_max = std::max<std::int64_t>({te_c.shape[0], te_u.shape[0], kBenchCond});
    const std::size_t io = mi::runtime::align_up(static_cast<std::size_t>(128 * h * w * 4)) * 3 +
                           4 * mi::runtime::align_up(static_cast<std::size_t>(s_max * (md::kTextDim + kDim) * 2)) +
                           mi::runtime::align_up(6 * kDim * 2) + mi::runtime::align_up(2 * kDim * 2);
    const std::size_t bytes = md::MossDit::persistent_bytes(*index_) + io +
                              std::max({md::MossDit::workspace_bytes(h, w, golden_.lengths, md::BlockPath::kOpByOp),
                                        md::MossDit::workspace_bytes(h, w, bench_.lengths, md::BlockPath::kOpByOp),
                                        md::MossDit::encode_text_workspace_bytes(s_max)});
    std::printf("[dit_block_fused] phase 1 (engine) arena %.3f GiB (free %.3f GiB)\n", static_cast<double>(bytes) / kGiB,
                static_cast<double>(mi::runtime::DeviceArena::free_device_bytes()) / kGiB);
    mi::runtime::DeviceArena arena(bytes);
    md::MossDit model(*index_, arena, *streams_);
    model.set_block_path(md::BlockPath::kOpByOp);  // the reference engine path; the other paths are checked against it
    // Refined text of the first `rows` encoder-output rows.
    const auto refine = [&](const mi::npy::Array& te, std::int64_t rows) {
      const auto v = te.to_vector<float>();
      std::vector<std::uint16_t> hv(static_cast<std::size_t>(rows * md::kTextDim));
      for (std::size_t i = 0; i < hv.size(); ++i) hv[i] = to_bf16(v[i]);
      const Tensor in = arena.alloc_persistent_tensor(DType::kBF16, {rows, md::kTextDim});
      upload(in, hv);
      const Tensor out = arena.alloc_persistent_tensor(DType::kBF16, {rows, kDim});
      model.encode_text(in, out, stream());
      return out;
    };
    const Tensor text_c = refine(te_c, golden_.lengths[0]), text_u = refine(te_u, golden_.lengths[1]);
    const Tensor text_bench = refine(te_long, kBenchCond);
    const Tensor latents = arena.alloc_persistent_tensor(DType::kF32, {128, h, w});
    {
      const auto nv = noise.to_vector<float>();
      MI_CUDA_CHECK(cudaMemcpy(latents.data, nv.data(), nv.size() * 4, cudaMemcpyHostToDevice));
    }
    const Tensor outs[] = {arena.alloc_persistent_tensor(DType::kF32, {128, h, w}),
                           arena.alloc_persistent_tensor(DType::kF32, {128, h, w})};
    const md::Row golden_rows[] = {{text_c, golden_.skips[0]}, {text_u, golden_.skips[1]}};
    const md::Row bench_rows[] = {{text_bench, bench_.skips[0]}, {text_u, bench_.skips[1]}};
    for (EngineRun* run : {&golden_, &bench_}) {
      const std::int64_t total = 2 * h * w + run->lengths[0] + run->lengths[1];
      model.forward(latents, timestep, run == &golden_ ? golden_rows : bench_rows, outs, stream(),
                    [&](int block, const Tensor& x) {
                      if (std::find(std::begin(kCaptured), std::end(kCaptured), block) == std::end(kCaptured)) return;
                      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
                      ASSERT_EQ(x.dim(0), total);
                      run->x_after[block] = download(x);
                    });
    }
    const Tensor mod = arena.alloc_persistent_tensor(DType::kBF16, {6 * kDim});
    const Tensor head = arena.alloc_persistent_tensor(DType::kBF16, {2 * kDim});
    model.time_modulation(timestep, mod, head, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    mod_ = download(mod);

    // MossDit::forward with each block path: the velocities of every row bit for bit equal to op-by-op's. kFusedGraphs
    // runs three times (eager, then capture, then replay).
    for (const md::Row* rows : {golden_rows, bench_rows}) {
      const std::span<const md::Row> r(rows, 2);
      const auto velocities = [&] {
        MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
        std::vector<float> v(static_cast<std::size_t>(2 * 128 * h * w));
        for (int i = 0; i < 2; ++i) {
          MI_CUDA_CHECK(cudaMemcpy(v.data() + i * 128 * h * w, outs[i].data, outs[i].nbytes(), cudaMemcpyDeviceToHost));
        }
        return v;
      };
      model.set_block_path(md::BlockPath::kOpByOp);
      model.forward(latents, timestep, r, outs, stream());
      const auto want = velocities();
      for (const md::BlockPath path : {md::BlockPath::kFused, md::BlockPath::kFusedGraphs}) {
        model.set_block_path(path);
        for (int i = 0; i < (path == md::BlockPath::kFusedGraphs ? 3 : 1); ++i) {
          model.forward(latents, timestep, r, outs, stream());
          const auto got = velocities();
          std::size_t diff = 0;
          for (std::size_t j = 0; j < want.size(); ++j) diff += std::bit_cast<std::uint32_t>(want[j]) != std::bit_cast<std::uint32_t>(got[j]);
          char buf[160];
          std::snprintf(buf, sizeof(buf), "%s %s call %d: %zu of %zu velocities differ from op_by_op",
                        rows == golden_rows ? golden_.name : bench_.name, md::block_path_name(path), i + 1, diff, want.size());
          path_checks_.emplace_back(buf, diff);
        }
      }
    }
    graph_stats_ = model.graph_stats();

    // The engine's steady-state step at AC9b's shape, op-by-op vs fused + graphs, interleaved.
    Timer timer(stream());
    std::vector<double> t_obo, t_fused;
    for (int i = 0; i < 4; ++i) {
      for (const md::BlockPath path : {md::BlockPath::kOpByOp, md::BlockPath::kFusedGraphs}) {
        if (model.block_path() != path) {
          model.set_block_path(path);
          model.forward(latents, timestep, bench_rows, outs, stream());  // warm-up (and graph shape sighting)
        }
        const double ms = timer.ms([&] { model.forward(latents, timestep, bench_rows, outs, stream()); });
        (path == md::BlockPath::kOpByOp ? t_obo : t_fused).push_back(ms);
      }
    }
    engine_step_ms_ = median(t_obo);
    std::printf("[dit_block_fused] engine MossDit::forward 1024^2, S=282+10, uncond skip {22}: op_by_op %.1f ms/step, "
                "fused_graphs %.1f ms/step -> %.3fx (%s)\n",
                engine_step_ms_, median(t_fused), engine_step_ms_ / median(t_fused), host_load().c_str());
  }

  // Phase 2: every block resident in a new arena with activation room for the largest benchmark shape.
  static void load_blocks() {
    const std::size_t weights = md::BlockWeights::resident_bytes(*index_, md::Residency{});
    const std::size_t bytes = weights + mi::runtime::align_up(mi::kernels::Gemm::kWorkspaceBytes) + Acts::bytes(kTokens1456) +
                              mi::runtime::align_up(static_cast<std::size_t>(kTokens1024Max * kDim * 2));
    std::printf("[dit_block_fused] phase 2 (resident blocks) arena %.3f GiB (free %.3f GiB)\n",
                static_cast<double>(bytes) / kGiB, static_cast<double>(mi::runtime::DeviceArena::free_device_bytes()) / kGiB);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(bytes);
    mi::weights::ReaderCache readers(*index_, false);
    mi::runtime::PinnedRing ring(streams_->copy());
    blocks_ = std::make_unique<md::BlockWeights>(
        *index_, *arena_, md::Residency{},
        [&](const mi::weights::WeightGroup& g) { return mi::weights::upload_group(readers, g, *arena_, ring); },
        [](const mi::weights::WeightGroup&) {});
    ring.synchronize();
    md::BlockWeights::Pass pass(*blocks_, stream());
    for (int b = 0; b < md::kLayers; ++b) views_.push_back(pass.acquire(b));
    pass.finish();
    gemm_ = std::make_unique<mi::kernels::Gemm>();
    gemm_ws_ = arena_->alloc_persistent(mi::kernels::Gemm::kWorkspaceBytes);
    x0_ = arena_->alloc_persistent(static_cast<std::size_t>(kTokens1024Max * kDim * 2));
  }

  OpByOp op_by_op() const { return {*gemm_, gemm_ws_}; }

  // A 1024^2 step's activations for an engine run: per-row rope tables, the engine's modulation, the runs.
  struct Step {
    std::unique_ptr<mi::runtime::StackScope> scope;
    std::unique_ptr<Acts> acts;
    StepRuns runs;
  };
  static Step step_for(const EngineRun& e) {
    Step s;
    s.scope = std::make_unique<mi::runtime::StackScope>(*arena_);
    const std::int64_t hw = e.h * e.w;
    s.acts = std::make_unique<Acts>(*arena_, 2 * hw + e.lengths[0] + e.lengths[1]);
    s.runs = make_runs(hw, e.lengths, e.skips, s.acts->cu_pool);
    for (std::size_t r = 0; r < 2; ++r) {
      mi::kernels::moss_rope_table(s.acts->rope.slice0(s.runs.off[r], s.runs.off[r + 1]), static_cast<int>(e.h),
                                   static_cast<int>(e.w), static_cast<int>(e.lengths[r]), stream());
    }
    upload(s.acts->mod, mod_);
    return s;
  }

  static void run_blocks(const Acts& a, const StepRuns& runs, int first, int last, bool fused) {
    for (int b = first; b <= last; ++b) {
      for (const md::BlockRun& run : runs.per_block[static_cast<std::size_t>(b)]) {
        if (fused) {
          md::run_block_fused(views_[static_cast<std::size_t>(b)], a.fused(), run, qkv_gemm(), stream());
        } else {
          OpByOp{*gemm_, gemm_ws_}.run(views_[static_cast<std::size_t>(b)], a, run, stream());
        }
      }
    }
  }

  static constexpr std::int64_t kTokens1024Max = 2 * 4096 + 2 * 4096;  // room for a 1024^2 step's x (any prompt)
  static constexpr std::int64_t kTokens1456 = 2 * 91 * 91 + 4096 + 16;

  static std::unique_ptr<mi::weights::ModelIndex> index_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<md::BlockWeights> blocks_;
  static std::vector<md::BlockTensors> views_;
  static std::unique_ptr<mi::kernels::Gemm> gemm_;
  static void* gemm_ws_;
  static void* x0_;
  static EngineRun golden_, bench_;
  static std::vector<std::uint16_t> mod_;
  static double engine_step_ms_;
  static std::vector<std::pair<std::string, std::size_t>> path_checks_;
  static md::StepGraphStats graph_stats_;
};

std::unique_ptr<mi::weights::ModelIndex> DitBlockFused::index_;
std::unique_ptr<mi::runtime::StreamSet> DitBlockFused::streams_;
std::unique_ptr<mi::runtime::DeviceArena> DitBlockFused::arena_;
std::unique_ptr<md::BlockWeights> DitBlockFused::blocks_;
std::vector<md::BlockTensors> DitBlockFused::views_;
std::unique_ptr<mi::kernels::Gemm> DitBlockFused::gemm_;
void* DitBlockFused::gemm_ws_ = nullptr;
void* DitBlockFused::x0_ = nullptr;
EngineRun DitBlockFused::golden_, DitBlockFused::bench_;
std::vector<std::uint16_t> DitBlockFused::mod_;
double DitBlockFused::engine_step_ms_ = 0;
std::vector<std::pair<std::string, std::size_t>> DitBlockFused::path_checks_;
md::StepGraphStats DitBlockFused::graph_stats_;

// MossDit::forward's block paths (phase 1): fused and fused + CUDA graphs give op-by-op's velocities bit for bit, and
// the graphs were captured once and then replayed.
TEST_F(DitBlockFused, ForwardBlockPathsBitIdentical) {
  ASSERT_EQ(path_checks_.size(), 8u);
  for (const auto& [what, diff] : path_checks_) {
    std::printf("[dit_block_fused] %s\n", what.c_str());
    EXPECT_EQ(diff, 0u) << what;
  }
  std::printf("[dit_block_fused] graphs: %llu captures, %llu replays, %llu collisions\n",
              static_cast<unsigned long long>(graph_stats_.captures), static_cast<unsigned long long>(graph_stats_.replays),
              static_cast<unsigned long long>(graph_stats_.collisions));
  EXPECT_GE(graph_stats_.captures, 2u * md::kLayers);  // the third kFusedGraphs call of each input replays
  EXPECT_GE(graph_stats_.replays, 2u * md::kLayers);
  EXPECT_EQ(graph_stats_.collisions, 0u);
}

// Each captured block input of the golden case through the test-local op-by-op block and through run_block_fused:
// both reproduce the engine's output of that block bit for bit (block 22 runs on the cond row only).
TEST_F(DitBlockFused, RealActivationsBlocksBitExactVsEngine) {
  Step s = step_for(golden_);
  for (int b : {1, 21, 22, 23, 41}) {
    const auto& in = golden_.x_after.at(b - 1);
    const auto& want = golden_.x_after.at(b);
    for (const bool fused : {false, true}) {
      upload(s.acts->x, in);
      run_blocks(*s.acts, s.runs, b, b, fused);
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      const Compare c = compare(want, download(s.acts->x));
      std::printf("[dit_block_fused] %s block %02d %-9s vs engine: %zu of %zu differ, cosine %.12f, max|diff| %.3g\n",
                  golden_.name, b, fused ? "fused" : "op-by-op", c.differ, want.size(), c.cosine, c.max_abs);
      EXPECT_EQ(c.differ, 0u) << "block " << b << (fused ? " fused" : " op-by-op");
    }
  }
  // The per-op split used for timing is run_block_fused.
  upload(s.acts->x, golden_.x_after.at(0));
  for (const md::BlockRun& run : s.runs.per_block[1]) fused_ops(views_[1], s.acts->fused(), run, qkv_gemm(), stream(), [](int) {});
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(compare(golden_.x_after.at(1), download(s.acts->x)).differ, 0u);
}

// Blocks 1..41 chained, fused: the engine's residual stream after block 41, bit for bit (both inputs).
TEST_F(DitBlockFused, FusedChainOf41BlocksBitExactVsEngine) {
  for (const EngineRun* e : {&golden_, &bench_}) {
    Step s = step_for(*e);
    upload(s.acts->x, e->x_after.at(0));
    run_blocks(*s.acts, s.runs, 1, md::kLayers - 1, true);
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const auto& want = e->x_after.at(md::kLayers - 1);
    const Compare c = compare(want, download(s.acts->x));
    std::printf("[dit_block_fused] %s (S=%lld+%lld): blocks 1..41 fused vs engine: %zu of %zu differ, cosine %.12f\n",
                e->name, static_cast<long long>(e->lengths[0]), static_cast<long long>(e->lengths[1]), c.differ,
                want.size(), c.cosine);
    EXPECT_EQ(c.differ, 0u) << e->name;
  }
}

// Per-op time of one block over the CFG batch, op-by-op vs fused (medians of interleaved rounds).
void per_op_breakdown(const char* label, const md::BlockTensors& w, const Acts& a, const md::BlockRun& run,
                      const OpByOp& obo, const md::QkvGemm& g, cudaStream_t stream) {
  constexpr int kRounds = 7;
  std::vector<cudaEvent_t> ev(OpByOp::kOps + 1);
  for (cudaEvent_t& e : ev) MI_CUDA_CHECK(cudaEventCreate(&e));
  std::vector<std::vector<double>> base(OpByOp::kOps), fused(kFusedOps);
  std::vector<double> base_total, fused_total;
  const auto collect = [&](int n, std::vector<std::vector<double>>& per, std::vector<double>& total) {
    MI_CUDA_CHECK(cudaEventSynchronize(ev[static_cast<std::size_t>(n)]));
    double sum = 0;
    for (int i = 0; i < n; ++i) {
      float ms = 0;
      MI_CUDA_CHECK(cudaEventElapsedTime(&ms, ev[static_cast<std::size_t>(i)], ev[static_cast<std::size_t>(i + 1)]));
      per[static_cast<std::size_t>(i)].push_back(ms);
      sum += ms;
    }
    total.push_back(sum);
  };
  const auto mark = [&](int i) { MI_CUDA_CHECK(cudaEventRecord(ev[static_cast<std::size_t>(i)], stream)); };
  for (int r = 0; r < kRounds; ++r) {
    obo.run(w, a, run, stream, mark);
    collect(OpByOp::kOps, base, base_total);
    fused_ops(w, a.fused(), run, g, stream, mark);
    collect(kFusedOps, fused, fused_total);
  }
  std::printf("[dit_block_fused perf] %s, one block over %lld tokens:\n", label, static_cast<long long>(run.t1 - run.t0));
  for (int i = 0; i < OpByOp::kOps; ++i) {
    std::printf("    op-by-op %-20s %7.3f ms\n", kOpNames[i], median(base[static_cast<std::size_t>(i)]));
  }
  for (int i = 0; i < kFusedOps; ++i) {
    std::printf("    fused    %-20s %7.3f ms\n", kFusedOpNames[i], median(fused[static_cast<std::size_t>(i)]));
  }
  const double b = median(base_total), f = median(fused_total);
  std::printf("    block total: op-by-op %.3f ms, fused %.3f ms -> %.3fx\n", b, f, b / f);
  for (cudaEvent_t e : ev) cudaEventDestroy(e);
}

// Whole 42-block step: op-by-op (= the engine's block loop) vs fused vs fused through CUDA graphs (one graph for the
// step, and one per block), interleaved rounds; medians.
void step_benchmark(const char* label, const std::function<void(bool)>& eager, const std::function<void()>& graph_step,
                    const std::function<void()>& graph_blocks, cudaStream_t stream) {
  Timer timer(stream);
  eager(false);
  eager(true);
  graph_step();
  graph_blocks();
  std::vector<double> base, fused, g_step, g_blocks;
  for (int r = 0; r < 3; ++r) {
    base.push_back(timer.ms([&] { eager(false); }));
    fused.push_back(timer.ms([&] { eager(true); }));
    g_step.push_back(timer.ms(graph_step));
    g_blocks.push_back(timer.ms(graph_blocks));
  }
  const double b = median(base), f = median(fused), gs = median(g_step), gb = median(g_blocks);
  std::printf("[dit_block_fused perf] %s, 42-block CFG step: op-by-op %.1f ms | fused %.1f ms (%.3fx) | fused + step "
              "graph %.1f ms (%.3fx) | fused + per-block graphs %.1f ms (%.3fx)  [%s]\n",
              label, b, f, b / f, gs, b / gs, gb, b / gb, host_load().c_str());
}

TEST_F(DitBlockFused, Speed1024RealActivations) {
  Step s = step_for(bench_);
  const std::size_t x_bytes = static_cast<std::size_t>(s.acts->total * kDim * 2);
  upload(s.acts->x, bench_.x_after.at(0));
  MI_CUDA_CHECK(cudaMemcpy(x0_, s.acts->x.data, x_bytes, cudaMemcpyDeviceToDevice));
  per_op_breakdown("1024^2 S=282+10 (real activations, block 1)", views_[1], *s.acts, s.runs.per_block[1][0], op_by_op(),
                   qkv_gemm(), stream());

  // Every step starts from the same residual stream (block 0's output), so all variants do identical work.
  const auto reset = [&] { MI_CUDA_CHECK(cudaMemcpyAsync(s.acts->x.data, x0_, x_bytes, cudaMemcpyDeviceToDevice, stream())); };
  const auto eager = [&](bool fused) {
    reset();
    run_blocks(*s.acts, s.runs, 0, md::kLayers - 1, fused);
  };
  md::StepGraphs graphs(64);
  const md::StepGraphKey key{bench_.h, bench_.w, bench_.lengths, {bench_.skips[0], bench_.skips[1]}, 0, md::kLayers,
                             {s.acts->x.data, s.acts->big.data}};
  const auto graph_step = [&] {
    graphs.run(key, stream(), [&](cudaStream_t) {
      reset();
      run_blocks(*s.acts, s.runs, 0, md::kLayers - 1, true);
    });
  };
  const auto graph_blocks = [&] {
    reset();
    for (int b = 0; b < md::kLayers; ++b) {
      md::StepGraphKey k = key;
      k.block_begin = b;
      k.block_end = b + 1;
      graphs.run(k, stream(), [&](cudaStream_t) { run_blocks(*s.acts, s.runs, b, b, true); });
    }
  };
  step_benchmark("1024^2 S=282+10, uncond skip {22}", eager, graph_step, graph_blocks, stream());
  std::printf("[dit_block_fused perf] engine MossDit::forward at this shape (phase 1, incl. embed/head): %.1f ms/step\n",
              engine_step_ms_);

  // Graph replays are bit-identical to the eager fused step (the timed step runs all 42 blocks from block 0's output,
  // so it has no engine counterpart; FusedChainOf41BlocksBitExactVsEngine checks blocks 1..41 against the engine).
  eager(true);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const auto eager_out = download(s.acts->x);
  graph_step();
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(compare(eager_out, download(s.acts->x)).differ, 0u);
  graph_blocks();
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(compare(eager_out, download(s.acts->x)).differ, 0u);
  EXPECT_EQ(graphs.stats().captures, 1u + md::kLayers);
  EXPECT_EQ(graphs.stats().collisions, 0u);
}

TEST_F(DitBlockFused, Speed1456With4096Tokens) {
  mi::runtime::StackScope scope(*arena_);
  const std::int64_t grid = 91, hw = grid * grid;
  const std::vector<std::int64_t> lengths = {4096, 16};
  md::SkipSet skip22;
  skip22.set(22);
  Acts a(*arena_, kTokens1456);
  const StepRuns runs = make_runs(hw, lengths, {md::SkipSet{}, skip22}, a.cu_pool);
  ASSERT_EQ(runs.total, kTokens1456);
  for (std::size_t r = 0; r < 2; ++r) {
    mi::kernels::moss_rope_table(a.rope.slice0(runs.off[r], runs.off[r + 1]), static_cast<int>(grid), static_cast<int>(grid),
                                 static_cast<int>(lengths[r]), stream());
  }
  upload(a.mod, mod_);
  const auto x_init = synth_bf16(42, static_cast<std::size_t>(kTokens1456 * kDim), 1);
  upload(a.x, x_init);
  per_op_breakdown("1456^2 + 4096 tokens (synthetic activations, block 1)", views_[1], a, runs.per_block[1][0], op_by_op(),
                   qkv_gemm(), stream());
  const auto eager = [&](bool fused) {
    upload(a.x, x_init);
    run_blocks(a, runs, 0, md::kLayers - 1, fused);
  };
  // Graphs are measured at 1024^2 (launch overhead does not grow with the shape); here the eager A/B only.
  step_benchmark("1456^2 + 4096 tokens (synthetic)", eager, [&] { eager(true); }, [&] { eager(true); }, stream());
}

// FA2 tile choice for the DiT's joint attention on this GPU: the default (128x32 on 100 KiB-smem parts) vs 128x64, on
// the q/k/v block 1 computes from `a.x` over `run`. A different tile changes the online-softmax grouping: the
// exact-match fraction against the default is reported. Overwrites a.x and a.y.
void attention_tile_sweep(const char* label, const md::BlockTensors& w, const Acts& a, const md::BlockRun& run,
                          mi::kernels::Gemm& gemm, void* gemm_ws, cudaStream_t stream) {
  const Tensor qkv = view(a.big.data, {a.total, kQkv});
  mi::kernels::rms_norm_modulate_moss(a.x, w.norm1, vec(a.mod, 1), vec(a.mod, 0), a.y, stream);
  gemm.linear(qkv, a.y, w.qkv, nullptr, gemm_ws, mi::kernels::Gemm::kWorkspaceBytes, stream);
  mi::kernels::fused::qk_norm_rope_moss(qkv, md::kHeads, w.q_norm, w.k_norm, a.rope, stream);
  const Tensor out_a = a.x.slice0(run.t0, run.t1), out_b = a.y.slice0(run.t0, run.t1);  // scratch outputs
  const Tensor q = heads(qkv.slice0(run.t0, run.t1), 0), k = heads(qkv.slice0(run.t0, run.t1), kDim),
               v = heads(qkv.slice0(run.t0, run.t1), 2 * kDim);
  Timer timer(stream);
  using mi::attention::FlashTile;
  const auto attend = [&](FlashTile tile, const Tensor& out) {
    mi::attention::flash_fwd_set_tile_override_for_testing(tile);
    mi::attention::flash_fwd(q, k, v, heads(out, 0), run.cu_seqlens, run.cu_seqlens, run.max_seqlen, false, kAttnScale,
                             a.lse, a.lse_bytes, stream);
  };
  std::vector<double> t32, t64;
  attend(FlashTile::kHdim128_128x32, out_a);
  attend(FlashTile::kHdim128_128x64, out_b);
  for (int r = 0; r < 7; ++r) {
    t32.push_back(timer.ms([&] { attend(FlashTile::kHdim128_128x32, out_a); }));
    t64.push_back(timer.ms([&] { attend(FlashTile::kHdim128_128x64, out_b); }));
  }
  attend(FlashTile::kAuto, out_a);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto auto_out = download(out_a);
  attend(FlashTile::kHdim128_128x64, out_b);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  const Compare c = compare(auto_out, download(out_b));
  mi::attention::flash_fwd_set_tile_override_for_testing(FlashTile::kAuto);
  std::vector<std::int32_t> cu(static_cast<std::size_t>(run.cu_seqlens.numel()));
  MI_CUDA_CHECK(cudaMemcpy(cu.data(), run.cu_seqlens.data, cu.size() * 4, cudaMemcpyDeviceToHost));
  double flops = 0;
  for (std::size_t i = 1; i < cu.size(); ++i) flops += 4.0 * 128 * md::kHeads * std::pow(static_cast<double>(cu[i] - cu[i - 1]), 2);
  std::printf("[dit_block_fused attention] %s: 128x32 (default) %.3f ms (%.1f TFLOPS), 128x64 %.3f ms (%.1f TFLOPS); "
              "128x64 vs default: %zu of %zu differ, cosine %.9f\n",
              label, median(t32), flops / median(t32) / 1e9, median(t64), flops / median(t64) / 1e9, c.differ,
              auto_out.size(), c.cosine);
}

TEST_F(DitBlockFused, AttentionTileSweep) {
  {
    Step s = step_for(bench_);
    upload(s.acts->x, bench_.x_after.at(0));
    attention_tile_sweep("1024^2 S=282+10 (real, block 1)", views_[1], *s.acts, s.runs.per_block[1][0], *gemm_, gemm_ws_,
                         stream());
  }
  mi::runtime::StackScope scope(*arena_);
  const std::int64_t grid = 91, hw = grid * grid;
  const std::vector<std::int64_t> lengths = {4096, 16};
  Acts a(*arena_, kTokens1456);
  const StepRuns runs = make_runs(hw, lengths, {md::SkipSet{}, md::SkipSet{}}, a.cu_pool);
  for (std::size_t r = 0; r < 2; ++r) {
    mi::kernels::moss_rope_table(a.rope.slice0(runs.off[r], runs.off[r + 1]), static_cast<int>(grid), static_cast<int>(grid),
                                 static_cast<int>(lengths[r]), stream());
  }
  upload(a.mod, mod_);
  upload(a.x, synth_bf16(43, static_cast<std::size_t>(kTokens1456 * kDim), 1));
  attention_tile_sweep("1456^2 S=4096+16 (synthetic, block 1)", views_[1], a, runs.per_block[1][0], *gemm_, gemm_ws_,
                       stream());
}

TEST(DitGraphKey, HashAndEqualityCoverEveryField) {
  md::StepGraphKey a{64, 64, {282, 10}, {md::SkipSet{}, md::SkipSet{}}, 0, 42, {reinterpret_cast<const void*>(0x1000)}};
  md::StepGraphKey b = a;
  EXPECT_EQ(a, b);
  EXPECT_EQ(a.hash(), b.hash());
  std::vector<md::StepGraphKey> variants(8, a);
  variants[0].height = 65;
  variants[1].width = 63;
  variants[2].text_lengths[0] = 283;
  variants[3].skips[1].set(22);
  variants[4].block_begin = 1;
  variants[5].block_end = 41;
  variants[6].addresses[0] = reinterpret_cast<const void*>(0x2000);
  variants[7].route_generation = 1;
  for (const auto& v : variants) {
    EXPECT_FALSE(v == a);
    EXPECT_NE(v.hash(), a.hash());
  }
}

// Capture once, replay afterwards, LRU eviction and re-capture.
TEST(DitGraph, CapturesReplaysAndEvicts) {
  cudaStream_t stream = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  mi::DeviceBuffer buf(1024 * 4);
  const Tensor t = Tensor::view(buf.get(), DType::kF32, mi::Device::kCUDA, {1024});
  md::StepGraphs graphs(2);
  int bodies = 0;
  const auto body = [&](cudaStream_t s) {
    ++bodies;
    mi::kernels::cast(t, t, s);  // a kernel and a memset on the captured stream
    MI_CUDA_CHECK(cudaMemsetAsync(t.data, 0, 16, s));
  };
  md::StepGraphKey k1{1, 1, {1}, {md::SkipSet{}}, 0, 42, {buf.get()}};
  md::StepGraphKey k2 = k1, k3 = k1;
  k2.height = 2;
  k3.height = 3;
  EXPECT_FALSE(graphs.run(k1, stream, body));
  EXPECT_TRUE(graphs.run(k1, stream, body));
  EXPECT_FALSE(graphs.run(k2, stream, body));
  EXPECT_FALSE(graphs.run(k3, stream, body));  // evicts k1
  EXPECT_EQ(graphs.size(), 2u);
  EXPECT_FALSE(graphs.run(k1, stream, body));  // captured again
  EXPECT_EQ(bodies, 4);
  EXPECT_EQ(graphs.stats().captures, 4u);
  EXPECT_EQ(graphs.stats().replays, 1u);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  MI_CUDA_CHECK(cudaStreamDestroy(stream));
}

// The GPU and library the routing facts asserted below were measured on: sm_89 with cuBLASLt 13.
bool measured_platform() {
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  MI_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return prop.major == 8 && prop.minor == 9 && cublasLtGetVersion() / 10000 == 13;
}

// Run lengths of a CFG step whose rows are `hw` latent tokens plus each text length: every non-empty set of the rows
// (a block that a row skips runs the others), as fused_big_cols sizes for.
std::vector<std::int64_t> run_lengths(std::int64_t hw, std::span<const std::int64_t> text_lengths) {
  std::vector<std::int64_t> out;
  for (std::uint32_t mask = 1; mask < (std::uint32_t{1} << text_lengths.size()); ++mask) {
    std::int64_t m = 0;
    for (std::size_t r = 0; r < text_lengths.size(); ++r) {
      if (mask & (std::uint32_t{1} << r)) m += hw + text_lengths[r];
    }
    out.push_back(m);
  }
  return out;
}

// M1 startup self-check: finds and checks at least one fused configuration on this GPU (cheaply), under every CUTLASS
// tile, routes only the configurations it validated, covers the production run lengths, and reports a mismatch
// injected into one fused result.
TEST(FusedSelfCheck, PassesAndReportsAnInjectedMismatch) {
  cudaStream_t stream = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto t0 = std::chrono::steady_clock::now();
  const md::FusedCheck ok = md::check_fused_gemms(stream);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // Device memory the check allocates: per op, its operands at the largest row count it checked, plus the GEMM
  // workspace (ffn_up_gate: x, w, the [rows, 2F] op-by-op output, ref and got; the others: x, w, gate, the op-by-op
  // output, ref and got).
  std::size_t operand_bytes = 0;
  for (const auto& [op, k, n] : {std::tuple{"o_proj", kDim, kDim}, std::tuple{"ffn_up_gate", kDim, kUpGate},
                                  std::tuple{"ffn_down", std::int64_t{md::kFfnDim}, kDim}}) {
    std::int64_t m = 0;
    for (const md::FusedConfig& f : ok.validated) m = f.op == op ? std::max(m, f.rows) : m;
    if (m == 0) continue;
    const std::int64_t elems = std::string(op) == "ffn_up_gate" ? m * k + n * k + 2 * m * n : m * k + n * k + n + 3 * m * n;
    operand_bytes = std::max(operand_bytes, static_cast<std::size_t>(elems) * 2);
  }
  const std::size_t check_bytes = operand_bytes + mi::kernels::Gemm::kWorkspaceBytes;
  std::printf("[fused self-check] %s: %s (%.2f s, %.0f MiB device memory)\n", ok.ok ? "ok" : "FAILED", ok.detail.c_str(), s,
              static_cast<double>(check_bytes) / (1 << 20));
  for (const md::FusedConfig& f : ok.validated) {
    std::printf("[fused self-check]   %-11s @ %5lld rows: %s\n", f.op.c_str(), static_cast<long long>(f.rows),
                f.config.describe().c_str());
  }
  ASSERT_TRUE(ok.ok) << ok.detail;
  ASSERT_FALSE(ok.validated.empty());  // a pass checked something
  EXPECT_LE(s, 30.0);                  // a hang guard: ~1 s on an idle 4090 Laptop GPU, slower under load or ASan
  EXPECT_LE(check_bytes, std::size_t{1} << 28);
  // Each validated configuration routes fused at its row count with the validated set, and op by op without it (a
  // configuration the check did not validate never runs fused).
  for (std::size_t i = 0; i < ok.validated.size(); ++i) {
    const md::FusedConfig& f = ok.validated[i];
    const auto pick = [&](const md::FusedRoute& r) { return f.op == "o_proj" ? r.o : f.op == "ffn_up_gate" ? r.up_gate : r.down; };
    EXPECT_TRUE(pick(md::fused_route(f.rows, ok.validated))) << f.op << " @ " << f.rows;
    std::vector<md::FusedConfig> others = ok.validated;
    others.erase(others.begin() + static_cast<std::ptrdiff_t>(i));
    EXPECT_FALSE(pick(md::fused_route(f.rows, others))) << f.op << " @ " << f.rows;
    const md::FusedRoute none = md::fused_route(f.rows, std::span<const md::FusedConfig>());
    EXPECT_FALSE(none.o || none.up_gate || none.down);
  }
  // The configurations cuBLASLt runs at the run lengths of the measured gates (1024^2 with 282 + 10 tokens, 1456^2 with
  // 4096 + 16) are validated ones, so those steps run every pair fused.
  struct Gate {
    const char* name;
    std::int64_t grid;
    std::vector<std::int64_t> lengths;
  };
  const Gate gates[] = {{"1024^2 282+10", 64, {282, 10}}, {"1456^2 4096+16", 91, {4096, 16}}};
  mi::kernels::Gemm probe;
  const auto view = [](std::int64_t rows, std::int64_t cols) {  // never dereferenced: algo_config reads the layout
    return Tensor::view(reinterpret_cast<void*>(std::uintptr_t{1} << 20), DType::kBF16, mi::Device::kCUDA, {rows, cols});
  };
  for (const Gate& g : gates) {
    for (const std::int64_t m : run_lengths(g.grid * g.grid, g.lengths)) {
      const md::FusedRoute r = md::fused_route(m, ok.validated);
      for (const auto& [op, k, n, fused] :
           {std::tuple{"o_proj", kDim, kDim, r.o}, std::tuple{"ffn_up_gate", kDim, kUpGate, r.up_gate},
            std::tuple{"ffn_down", std::int64_t{md::kFfnDim}, kDim, r.down}}) {
        mi::kernels::Gemm::AlgoConfig c =
            probe.algo_config(view(m, n), view(m, k), view(n, k), nullptr, mi::kernels::Gemm::kWorkspaceBytes);
        c.workspace_bytes = 0;
        const bool validated = std::any_of(ok.validated.begin(), ok.validated.end(),
                                           [&](const md::FusedConfig& f) { return f.op == op && f.config == c; });
        std::printf("[fused self-check] %-14s %5lld rows %-11s %s: %s\n", g.name, static_cast<long long>(m), op,
                    c.describe().c_str(), validated ? "validated" : "NOT validated");
        EXPECT_EQ(fused, validated && mi::kernels::fused::matches_cublaslt(c)) << g.name << " " << op << " @ " << m;
        if (measured_platform()) {
          EXPECT_TRUE(validated) << g.name << " " << op << " @ " << m;
        }
      }
    }
  }
  int calls = 0;
  std::string where;
  const md::FusedCheck bad = md::check_fused_gemms(stream, [&](const char* op, std::int64_t rows, const Tensor& out) {
    if (++calls != 2) return;
    where = std::string(op) + " at " + std::to_string(rows) + " rows";
    std::uint16_t v = 0;
    MI_CUDA_CHECK(cudaMemcpy(&v, out.data, 2, cudaMemcpyDeviceToHost));
    v ^= 1;  // one ulp of one output
    MI_CUDA_CHECK(cudaMemcpy(out.data, &v, 2, cudaMemcpyHostToDevice));
  });
  std::printf("[fused self-check] injected into %s: %s\n", where.c_str(), bad.detail.c_str());
  EXPECT_FALSE(bad.ok);
  EXPECT_TRUE(bad.validated.empty());
  EXPECT_NE(bad.detail.find(where + ": 1 of"), std::string::npos) << bad.detail;
  EXPECT_NE(bad.detail.find("CUTLASS tile 128x256"), std::string::npos) << bad.detail;  // the second tile run
  MI_CUDA_CHECK(cudaStreamDestroy(stream));
}

// A failed self-check poisons the device: configurations another check installed (before or after) never run fused
// again, workspaces are sized for the op-by-op ffn_up_gate, a fused block over a buffer sized for the old route refuses
// to run, and the graph key changes. In a child process: the poisoning is process-wide and permanent.
int poisoned_route_failures() {
  int failures = 0;
  const auto expect = [&](bool cond, const std::string& what) {
    if (!cond) {
      ++failures;
      std::fprintf(stderr, "poisoned route: FAILED %s\n", what.c_str());
    }
  };
  cudaStream_t stream = nullptr;
  MI_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const md::FusedCheck ok = md::check_fused_gemms(stream);
  expect(ok.ok, "self-check: " + ok.detail);
  md::install_fused_check(ok);
  const std::int64_t lengths[] = {282, 10};
  const std::uint64_t gen_ok = md::fused_route_generation();
  expect(gen_ok > 0, "installing configurations changes the generation");
  bool any_fused = false;
  for (const md::FusedConfig& f : ok.validated) {
    const md::FusedRoute r = md::fused_route(f.rows);
    any_fused = any_fused || r.o || r.up_gate || r.down;
  }
  expect(any_fused, "the installed check runs something fused");
  if (measured_platform()) expect(md::fused_big_cols(64 * 64, lengths) == md::kFusedBigCols, "1024^2 fused big buffer");

  md::install_fused_check(md::FusedCheck{false, "injected failure", {}});
  const std::uint64_t gen_poisoned = md::fused_route_generation();
  expect(gen_poisoned != gen_ok, "poisoning changes the generation");
  md::install_fused_check(ok);  // another pipeline's passed check afterwards
  md::install_fused_check(md::FusedCheck{false, "again", {}});
  expect(md::fused_route_generation() == gen_poisoned, "nothing else changes the generation");
  for (const md::FusedConfig& f : ok.validated) {
    const md::FusedRoute r = md::fused_route(f.rows);
    expect(!(r.o || r.up_gate || r.down), f.op + " @ " + std::to_string(f.rows) + " runs op by op");
    const md::FusedRoute t = md::fused_route(f.rows, ok.validated);  // the explicit-set overload ignores poisoning
    expect(t.o || t.up_gate || t.down, "explicit set @ " + std::to_string(f.rows));
  }
  expect(md::fused_big_cols(64 * 64, lengths) == md::kOpByOpBigCols, "op-by-op big buffer after poisoning");
  expect(md::MossDit::workspace_bytes(64, 64, lengths, md::BlockPath::kFused) ==
             md::MossDit::workspace_bytes(64, 64, lengths, md::BlockPath::kOpByOp),
         "fused workspace = op-by-op workspace after poisoning");
  // A block over a big buffer sized for the fused route (kFusedBigCols per token) is refused before any launch.
  constexpr int kRows = 16;
  mi::DeviceBuffer x(kRows * kDim * 2), y(kRows * kDim * 2), big(kRows * md::kFusedBigCols * 2), mod(6 * kDim * 2);
  const md::FusedBlockBuffers a{Tensor::view(x.get(), DType::kBF16, mi::Device::kCUDA, {kRows, kDim}),
                                Tensor::view(y.get(), DType::kBF16, mi::Device::kCUDA, {kRows, kDim}),
                                Tensor::view(big.get(), DType::kBF16, mi::Device::kCUDA, {kRows, md::kFusedBigCols}),
                                Tensor{},
                                Tensor::view(mod.get(), DType::kBF16, mi::Device::kCUDA, {6 * kDim})};
  mi::kernels::Gemm gemm;
  const md::QkvGemm qkv{&gemm, nullptr, 0};
  std::string error;
  try {
    md::run_block_fused(md::BlockTensors{}, a, md::BlockRun{0, kRows, Tensor{}, kRows}, qkv, stream);
  } catch (const std::invalid_argument& e) {
    error = e.what();
  }
  expect(error.find(std::to_string(md::kOpByOpBigCols) + " per token") != std::string::npos,
         "run_block_fused refuses the small buffer: '" + error + "'");
  MI_CUDA_CHECK(cudaStreamDestroy(stream));
  if (failures == 0) std::fprintf(stderr, "poisoned route: ok\n");
  return failures;
}

TEST(FusedSelfCheck, FailedCheckPoisonsTheDevice) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");  // re-executes the binary: a fresh CUDA context and route state
  EXPECT_EXIT(std::exit(poisoned_route_failures()), ::testing::ExitedWithCode(0), "poisoned route: ok");
}

// Small images, where cuBLASLt's algorithm (and so fused_route) changes with the run length: MossDit::forward with the
// fused block paths gives op-by-op's velocities bit for bit at 128^2 and 256^2 for cond-only, CFG (uncond skipping
// block 22) and CFG + SLG steps with 20-, 62- and 282-token prompts (128^2 + 20 tokens runs ffn_down at 84 rows and
// + 282 tokens at 346 rows: op by op on sm_89). Also measures the device memory the CUDA graphs hold.
TEST(DitSmallImages, ForwardFusedMatchesOpByOpBitForBit) {
  if (!fs::is_regular_file(bundle_dir() / "moss.mossidx")) GTEST_SKIP() << "weights not present (" << bundle_dir() << ")";
  const mi::weights::ModelIndex index = mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir());
  mi::runtime::StreamSet streams;
  const cudaStream_t stream = streams.compute();
  constexpr std::int64_t kMaxText = 282, kUncond = 10;
  const std::int64_t grids[] = {8, 16};  // 128^2, 256^2
  std::size_t work = 0;
  for (const std::int64_t g : grids) {
    const std::int64_t lengths[] = {kMaxText, kUncond, kMaxText};
    work = std::max(work, md::MossDit::workspace_bytes(g, g, lengths, md::BlockPath::kOpByOp));
  }
  const std::size_t io = 3 * mi::runtime::align_up(static_cast<std::size_t>(kMaxText * kDim * 2)) +
                         4 * mi::runtime::align_up(static_cast<std::size_t>(128 * 16 * 16 * 4));
  // Every block resident when it fits (then the graph paths run too), else the blocks that fit.
  md::Residency residency;
  const std::size_t free = mi::runtime::DeviceArena::free_device_bytes() - (std::size_t{512} << 20);
  for (; residency.resident_blocks > 0; --residency.resident_blocks) {
    if (md::MossDit::persistent_bytes(index, residency) + io + work + md::MossDit::stream_workspace_bytes(index, residency) <= free) break;
  }
  mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(index, residency) + io + work +
                                 md::MossDit::stream_workspace_bytes(index, residency));
  md::MossDit model(index, arena, streams, false, nullptr, residency);
  const Tensor cond = arena.alloc_persistent_tensor(DType::kBF16, {kMaxText, kDim});
  const Tensor uncond = arena.alloc_persistent_tensor(DType::kBF16, {kUncond, kDim});
  upload(cond, synth_bf16(51, static_cast<std::size_t>(kMaxText * kDim), 0));
  upload(uncond, synth_bf16(52, static_cast<std::size_t>(kUncond * kDim), 0));
  install_self_check(stream);
  for (const std::int64_t m : {84, 158, 211, 256, 346, 1100}) {
    const md::FusedRoute r = md::fused_route(m);
    std::printf("[dit small] fused_route(%lld rows): o %s, up_gate %s, down %s\n", static_cast<long long>(m),
                r.o ? "fused" : "op by op", r.up_gate ? "fused" : "op by op", r.down ? "fused" : "op by op");
  }
  if (measured_platform()) {
    EXPECT_FALSE(md::fused_route(84).down) << "sm_89: cuBLASLt splits K for ffn_down at 84 rows";
  }
  // Whether a forward over `rows` ran some pairs fused and others op by op (in one run, or across its runs).
  const auto mixed = [](std::int64_t hw, std::span<const md::Row> rows) {
    bool any_fused = false, any_op = false;
    const auto add = [&](std::int64_t m) {
      const md::FusedRoute r = md::fused_route(m);
      for (const bool f : {r.o, r.up_gate, r.down}) (f ? any_fused : any_op) = true;
    };
    for (int block = 0; block < md::kLayers; ++block) {
      std::int64_t run = 0;  // contiguous rows that execute the block
      for (const md::Row& row : rows) {
        if (row.skip.test(static_cast<std::size_t>(block))) {
          if (run > 0) add(run);
          run = 0;
        } else {
          run += hw + row.text.dim(0);
        }
      }
      if (run > 0) add(run);
    }
    return any_fused && any_op;
  };
  std::size_t mixed_forwards = 0;
  md::SkipSet skip22, slg;
  skip22.set(22);
  slg.set(7);
  slg.set(8);
  std::size_t checks = 0, graph_captures = 0;
  double graph_bytes_per_capture = -1;
  for (const std::int64_t g : grids) {
    const Tensor latents = arena.alloc_persistent_tensor(DType::kF32, {128, g, g});
    std::vector<float> lv(static_cast<std::size_t>(latents.numel()));
    for (std::size_t i = 0; i < lv.size(); ++i) lv[i] = std::sin(0.37f * static_cast<float>(i));
    MI_CUDA_CHECK(cudaMemcpy(latents.data, lv.data(), lv.size() * 4, cudaMemcpyHostToDevice));
    const Tensor outs[] = {arena.alloc_persistent_tensor(DType::kF32, {128, g, g}),
                           arena.alloc_persistent_tensor(DType::kF32, {128, g, g}),
                           arena.alloc_persistent_tensor(DType::kF32, {128, g, g})};
    for (const std::int64_t sc : {20, 62, 282}) {
      const Tensor c = cond.slice0(0, sc);
      const md::Row sets[3][3] = {{{c, {}}}, {{c, {}}, {uncond, skip22}}, {{c, {}}, {uncond, skip22}, {c, slg}}};
      for (std::size_t n = 1; n <= 3; ++n) {
        const std::span<const md::Row> rows(sets[n - 1], n);
        const std::span<const Tensor> out(outs, n);
        const auto velocities = [&] {
          MI_CUDA_CHECK(cudaStreamSynchronize(stream));
          std::vector<float> v(static_cast<std::size_t>(n * 128 * g * g));
          for (std::size_t i = 0; i < n; ++i) {
            MI_CUDA_CHECK(cudaMemcpy(v.data() + i * 128 * g * g, outs[i].data, outs[i].nbytes(), cudaMemcpyDeviceToHost));
          }
          return v;
        };
        model.set_block_path(md::BlockPath::kOpByOp);
        model.forward(latents, 500.0f, rows, out, stream);
        const auto want = velocities();
        mixed_forwards += mixed(g * g, rows);
        for (const md::BlockPath path : {md::BlockPath::kFused, md::BlockPath::kFusedGraphs}) {
          if (path == md::BlockPath::kFusedGraphs && residency.streamed()) continue;
          model.set_block_path(path);
          for (int i = 0; i < (path == md::BlockPath::kFusedGraphs ? 3 : 1); ++i) {
            const std::size_t free_before = mi::runtime::DeviceArena::free_device_bytes();
            const std::uint64_t captures_before = model.graph_stats().captures;
            model.forward(latents, 500.0f, rows, out, stream);
            const auto got = velocities();
            const std::uint64_t captured = model.graph_stats().captures - captures_before;
            if (captured > 0 && graph_bytes_per_capture < 0) {
              const std::size_t free_after = mi::runtime::DeviceArena::free_device_bytes();
              graph_bytes_per_capture =
                  free_before > free_after ? static_cast<double>(free_before - free_after) / static_cast<double>(captured) : 0.0;
            }
            graph_captures += captured;
            std::size_t diff = 0;
            for (std::size_t j = 0; j < want.size(); ++j) diff += std::bit_cast<std::uint32_t>(want[j]) != std::bit_cast<std::uint32_t>(got[j]);
            ++checks;
            EXPECT_EQ(diff, 0u) << g * 16 << "^2, S=" << sc << ", " << n << " rows, " << md::block_path_name(path)
                                << " call " << i + 1;
          }
        }
      }
    }
  }
  std::printf("[dit small] %zu fused forwards at 128^2 and 256^2 bit-identical to op_by_op (%d of 42 blocks resident); "
              "%zu graph captures, ~%.0f KiB device memory per captured block graph\n",
              checks, residency.resident_blocks, graph_captures, graph_bytes_per_capture / 1024.0);
  std::printf("[dit small] %zu of the checked forwards mixed fused and op-by-op pairs\n", mixed_forwards);
  if (measured_platform()) {
    EXPECT_GT(mixed_forwards, 0u);  // the fallback ran inside fused forwards that matched op_by_op
  } else {
    std::printf("[dit small] not sm_89 + cuBLASLt 13: whether some forward mixes fused and op-by-op pairs is not asserted\n");
  }
  if (graph_bytes_per_capture >= 0) {
    EXPECT_LE(graph_bytes_per_capture * md::kGraphCapacity, static_cast<double>(md::graph_cache_bytes()));
  }
}

}  // namespace
