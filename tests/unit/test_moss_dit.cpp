// MOSS DiT parity (AC5) against tests/golden/<case>/ step-0 outputs, varlen batching vs per-row runs,
// activation memory at 1456^2 with a 4096-token prompt, and steady-state step timing.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "models/moss_dit/moss_dit.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;
using mi::moss_dit::MossDit;
using mi::moss_dit::Row;
using mi::moss_dit::SkipSet;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kAc5Cosine = 0.99988;

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path golden_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "golden"; }
fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "moss_dit"; }

std::uint16_t to_bf16(float v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}
float from_bf16(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float v = 0;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}

// Device tensor owning its memory (test I/O lives outside the model's arena).
struct DevTensor {
  mi::DeviceBuffer buf;
  Tensor t;
  DevTensor(mi::DType dt, std::initializer_list<std::int64_t> shape) {
    t = Tensor::view(nullptr, dt, mi::Device::kCUDA, shape);
    buf = mi::DeviceBuffer(std::max<std::size_t>(t.nbytes(), 1));
    t.data = buf.get();
  }
};

// fp32 copies of bf16 values (text-encoder output) -> device bf16 [S, C], exact.
std::unique_ptr<DevTensor> upload_bf16(const std::vector<float>& v, std::int64_t rows, std::int64_t cols) {
  auto d = std::make_unique<DevTensor>(mi::DType::kBF16, std::initializer_list<std::int64_t>{rows, cols});
  std::vector<std::uint16_t> h(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) h[i] = to_bf16(v[i]);
  d->buf.upload(h.data(), h.size() * 2);
  return d;
}

std::vector<std::uint16_t> download_u16(const Tensor& t) {
  std::vector<std::uint16_t> h(static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, h.size() * 2, cudaMemcpyDeviceToHost));
  return h;
}

std::vector<float> download_f32(const Tensor& t) {
  std::vector<float> h(static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, h.size() * 4, cudaMemcpyDeviceToHost));
  return h;
}

struct Compare {
  double cosine = 0, max_abs = 0, exact = 0;
};
template <typename Get>
Compare compare(std::size_t n, const float* ref, Get got) {
  double dot = 0, gg = 0, rr = 0, max_abs = 0;
  std::size_t same = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double g = got(i), r = ref[i];
    dot += g * r;
    gg += g * g;
    rr += r * r;
    max_abs = std::max(max_abs, std::abs(g - r));
    same += g == r;
  }
  return {dot / std::sqrt(gg * rr), max_abs, static_cast<double>(same) / static_cast<double>(n)};
}

SkipSet read_skip(const fs::path& p) {
  std::ifstream in(p);
  std::stringstream ss;
  ss << in.rdbuf();
  SkipSet s;
  const mi::json::Value doc = mi::json::parse(ss.str());
  for (const auto& v : doc.as_array()) s.set(static_cast<std::size_t>(v.as_int()));
  return s;
}

// Linear + attention FLOPs of one forward over rows with these joint lengths and active-block counts.
double forward_flops(const std::vector<std::int64_t>& joint, const std::vector<int>& active_blocks) {
  constexpr double kLinearPerToken = 2.0 * 3072.0 * (3 * 3072 + 3072 + 16384 + 8192);
  double f = 0;
  for (std::size_t r = 0; r < joint.size(); ++r) {
    const double l = static_cast<double>(joint[r]);
    f += active_blocks[r] * (kLinearPerToken * l + 4.0 * l * l * 3072.0);
  }
  return f;
}

class MossDitTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::is_regular_file(bundle_dir() / "moss.mossidx") || !fs::is_directory(golden_dir() / "g_fridge_1024")) return;
    index_ = std::make_unique<mi::weights::ModelIndex>(
        mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir()));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    // Largest workload here: 1456^2 (91x91 latent grid) with a 4096-token cond row and a 16-token uncond row.
    const std::int64_t lengths[] = {4096, 16};
    // encode_text and forward never overlap on the stack.
    const std::size_t work = std::max(MossDit::workspace_bytes(91, 91, lengths), MossDit::encode_text_workspace_bytes(4096));
    // Every block resident when it fits; otherwise (other applications hold device memory) the blocks that fit, the
    // rest streamed: bit-identical (test_dit_stream), so the parity tests hold either way.
    const std::size_t free = mi::runtime::DeviceArena::free_device_bytes() - (std::size_t{256} << 20);
    for (residency_.resident_blocks = mi::moss_dit::kNumBlocks; residency_.resident_blocks > 0; --residency_.resident_blocks) {
      if (MossDit::persistent_bytes(*index_, residency_) + work + MossDit::stream_workspace_bytes(*index_, residency_) <= free) break;
    }
    const std::size_t bytes =
        MossDit::persistent_bytes(*index_, residency_) + work + MossDit::stream_workspace_bytes(*index_, residency_);
    std::printf("[moss_dit] arena %.3f GiB (free %.3f GiB), %d of 42 blocks resident\n", static_cast<double>(bytes) / kGiB,
                static_cast<double>(mi::runtime::DeviceArena::free_device_bytes()) / kGiB, residency_.resident_blocks);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(bytes);
    // MOSS_DIT_DIRECT_IO=1 loads with O_DIRECT (cold-cache behaviour) instead of buffered reads.
    const char* direct_env = std::getenv("MOSS_DIT_DIRECT_IO");
    const bool direct = direct_env != nullptr && std::string(direct_env) == "1";
    model_ = std::make_unique<MossDit>(*index_, *arena_, *streams_, direct, nullptr, residency_);
    const auto& st = model_->load_stats();
    std::printf("[moss_dit] load: %.2f GiB in %.2f s (%.2f GB/s, %s reads), persistent %.3f GiB\n",
                static_cast<double>(st.bytes) / kGiB, st.seconds, static_cast<double>(st.bytes) / st.seconds / 1e9,
                direct ? "direct" : "buffered", static_cast<double>(arena_->persistent_used()) / kGiB);
  }
  static void TearDownTestSuite() {
    model_.reset();
    arena_.reset();
    streams_.reset();
    index_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "weights or goldens not present (" << bundle_dir() << ")";
  }
  cudaStream_t stream() const { return streams_->compute(); }

  // Refined text tokens of one row from an fp32 copy of the text encoder's bf16 last hidden state.
  std::unique_ptr<DevTensor> refine(const std::vector<float>& te, std::int64_t s) {
    auto in = upload_bf16(te, s, mi::moss_dit::kTextDim);
    auto out = std::make_unique<DevTensor>(mi::DType::kBF16, std::initializer_list<std::int64_t>{s, mi::moss_dit::kDim});
    model_->encode_text(in->t, out->t, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    return out;
  }

  struct Case {
    std::unique_ptr<DevTensor> latents, cond, uncond;
    std::int64_t h = 0, w = 0, s_cond = 0, s_uncond = 0;
    float timestep = 0;
    SkipSet uncond_skip;
  };
  Case load_case(const std::string& id) {
    Case c;
    const fs::path g = golden_dir();
    const auto noise = mi::npy::load(g / id / "initial_noise.npy");
    c.h = noise.shape[2];
    c.w = noise.shape[3];
    c.latents = std::make_unique<DevTensor>(mi::DType::kF32, std::initializer_list<std::int64_t>{128, c.h, c.w});
    const auto nv = noise.to_vector<float>();
    c.latents->buf.upload(nv.data(), nv.size() * 4);
    c.timestep = static_cast<float>(mi::npy::load(g / id / "timestep0.npy").to_vector<double>().at(0));
    const auto te_c = mi::npy::load(g / (id + ".cond.te_last_hidden.npy"));
    const auto te_u = mi::npy::load(g / (id + ".uncond.te_last_hidden.npy"));
    c.s_cond = te_c.shape[0];
    c.s_uncond = te_u.shape[0];
    c.cond = refine(te_c.to_vector<float>(), c.s_cond);
    c.uncond = refine(te_u.to_vector<float>(), c.s_uncond);
    c.uncond_skip = read_skip(g / id / "dit_step0_uncond.skip.json");
    return c;
  }

  static std::unique_ptr<mi::weights::ModelIndex> index_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<MossDit> model_;
  static mi::moss_dit::Residency residency_;
};

std::unique_ptr<mi::weights::ModelIndex> MossDitTest::index_;
std::unique_ptr<mi::runtime::StreamSet> MossDitTest::streams_;
std::unique_ptr<mi::runtime::DeviceArena> MossDitTest::arena_;
std::unique_ptr<MossDit> MossDitTest::model_;
mi::moss_dit::Residency MossDitTest::residency_;

// Sub-model parity against tools/reference/gen_moss_dit_vectors.py (tests/data/moss_dit/).
Compare compare_bf16(const std::vector<std::uint16_t>& ref, const std::vector<std::uint16_t>& got) {
  std::vector<float> r(ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) r[i] = from_bf16(ref[i]);
  return compare(r.size(), r.data(), [&](std::size_t i) { return from_bf16(got[i]); });
}

TEST_F(MossDitTest, TextRefinerMatchesReference) {
  const auto manifest = [] {
    std::ifstream in(data_dir() / "manifest.json");
    std::stringstream ss;
    ss << in.rdbuf();
    return mi::json::parse(ss.str());
  }();
  for (const auto& e : manifest.at("text").as_array()) {
    const auto& o = e;
    const auto te = mi::npy::load(golden_dir() / o.at("input").as_string());
    const std::int64_t s = te.shape[0];
    auto refined = refine(te.to_vector<float>(), s);
    const auto ref = mi::npy::load(data_dir() / o.at("file").as_string()).to_vector<std::uint16_t>();
    const auto got = download_u16(refined->t);
    ASSERT_EQ(ref.size(), got.size());
    const Compare cmp = compare_bf16(ref, got);
    std::printf("[moss_dit text refiner] %s (S=%lld): cosine %.9f, max|diff| %.4g, exact %.4f%%\n",
                o.at("input").as_string().c_str(), static_cast<long long>(s), cmp.cosine, cmp.max_abs, 100.0 * cmp.exact);
    EXPECT_GE(cmp.cosine, 0.99999);
  }
}

TEST_F(MossDitTest, TimeModulationMatchesReference) {
  const auto manifest = [] {
    std::ifstream in(data_dir() / "manifest.json");
    std::stringstream ss;
    ss << in.rdbuf();
    return mi::json::parse(ss.str());
  }();
  DevTensor mod(mi::DType::kBF16, {6 * mi::moss_dit::kDim}), head(mi::DType::kBF16, {2 * mi::moss_dit::kDim});
  for (const auto& e : manifest.at("time").as_array()) {
    const auto& o = e;
    const float t = static_cast<float>(o.at("timestep").as_double());
    model_->time_modulation(t, mod.t, head.t, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const std::string prefix = o.at("prefix").as_string();
    for (const char* part : {"mod", "head"}) {
      const auto ref = mi::npy::load(data_dir() / (prefix + "_" + part + ".npy")).to_vector<std::uint16_t>();
      const auto got = download_u16(std::string(part) == "mod" ? mod.t : head.t);
      ASSERT_EQ(ref.size(), got.size());
      const Compare cmp = compare_bf16(ref, got);
      std::printf("[moss_dit time] t=%.1f %-4s: cosine %.9f, max|diff| %.4g, exact %.4f%%\n", static_cast<double>(t), part,
                  cmp.cosine, cmp.max_abs, 100.0 * cmp.exact);
      EXPECT_GE(cmp.cosine, 0.99999);
    }
  }
}

class MossDitGolden : public MossDitTest, public ::testing::WithParamInterface<const char*> {};

TEST_P(MossDitGolden, Step0CondUncondAndBlocks) {
  const std::string id = GetParam();
  Case c = load_case(id);
  const fs::path dir = golden_dir() / id;
  const std::int64_t hw = c.h * c.w, l_cond = hw + c.s_cond;
  constexpr int kBlocks[] = {0, 21, 22, 41};
  std::map<int, std::vector<std::uint16_t>> captured;
  const auto hook = [&](int block, const Tensor& x) {
    if (std::find(std::begin(kBlocks), std::end(kBlocks), block) == std::end(kBlocks)) return;
    auto& h = captured[block];
    h.resize(static_cast<std::size_t>(l_cond * mi::moss_dit::kDim));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    MI_CUDA_CHECK(cudaMemcpy(h.data(), x.data, h.size() * 2, cudaMemcpyDeviceToHost));  // cond row = row 0
  };
  DevTensor out_c(mi::DType::kF32, {128, c.h, c.w}), out_u(mi::DType::kF32, {128, c.h, c.w});
  const Row rows[] = {{c.cond->t, SkipSet{}}, {c.uncond->t, c.uncond_skip}};
  const Tensor outs[] = {out_c.t, out_u.t};
  model_->forward(c.latents->t, c.timestep, rows, outs, stream(), hook);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));

  for (int b : kBlocks) {
    const auto ref = mi::npy::load(dir / ("dit_step0_cond_block" + std::string(b < 10 ? "0" : "") + std::to_string(b) + ".npy"))
                         .to_vector<float>();
    ASSERT_EQ(ref.size(), captured[b].size());
    const auto& got = captured[b];
    const auto bf = [&](std::size_t i) { return from_bf16(got[i]); };
    const Compare joint = compare(ref.size(), ref.data(), bf);
    const Compare video = compare(static_cast<std::size_t>(hw * mi::moss_dit::kDim), ref.data(), bf);
    std::printf("[moss_dit %s] cond block %02d: joint cosine %.9f max|diff| %.4g exact %.4f%%; video cosine %.9f\n",
                id.c_str(), b, joint.cosine, joint.max_abs, 100.0 * joint.exact, video.cosine);
    // Diagnostic only (AC5 is defined on the step output): rounding differences grow through the blocks,
    // most for long prompts (g_long4096 block 41: video 0.9992, its step output still 0.99994). The
    // joint stream also holds text tokens of norm up to ~4e4 whose bf16 ULPs dominate its cosine.
    EXPECT_GE(video.cosine, 0.999) << "block " << b;
  }
  for (const char* name : {"cond", "uncond"}) {
    const auto ref = mi::npy::load(dir / ("dit_step0_" + std::string(name) + ".npy")).to_vector<float>();
    const auto got = download_f32(std::string(name) == "cond" ? out_c.t : out_u.t);
    ASSERT_EQ(ref.size(), got.size());
    const Compare cmp = compare(ref.size(), ref.data(), [&](std::size_t i) { return got[i]; });
    std::printf("[moss_dit %s] %-6s (S=%lld, skip %s): cosine %.9f, max|diff| %.4g, exact %.4f%%\n", id.c_str(), name,
                static_cast<long long>(std::string(name) == "cond" ? c.s_cond : c.s_uncond),
                std::string(name) == "cond" ? "{}" : c.uncond_skip.to_string().c_str(), cmp.cosine, cmp.max_abs,
                100.0 * cmp.exact);
    RecordProperty(std::string(name) + "_cosine", std::to_string(cmp.cosine));
    EXPECT_GE(cmp.cosine, kAc5Cosine) << name;
  }
}

INSTANTIATE_TEST_SUITE_P(Golden, MossDitGolden,
                         ::testing::Values("g_fridge_1024", "g_layout_1024", "g_short_1024", "g_long4096_1024",
                                           "g_fridge_1456", "g_layout_1456", "g_short_1456"));

// One varlen batch == each row run alone, bit for bit (the header's claim): 2 and 3 rows, two latent grids, real
// prompt lengths, and skip patterns that make a block run on one contiguous run or on two non-contiguous runs.
TEST_F(MossDitTest, VarlenBatchBitExactPerRow) {
  Case c = load_case("g_layout_1024");  // 64x64 grid (1024^2), real cond/uncond refined text
  SkipSet s22, s24;
  s22.set(22);
  s24.set(24);
  const Row cond{c.cond->t, SkipSet{}}, uncond{c.uncond->t, SkipSet{}}, uncond22{c.uncond->t, s22};
  const Row slg22{c.cond->t, s22}, slg24{c.cond->t, s24};
  struct RowSet {
    const char* name;
    std::vector<Row> rows;
  };
  const RowSet sets[] = {
      {"{cond, uncond}", {cond, uncond}},
      {"{cond, uncond-skip22}", {cond, uncond22}},
      {"{cond, uncond-skip22, slg-skip22}", {cond, uncond22, slg22}},   // block 22: row 0 only
      {"{cond, uncond-skip22, slg-skip24}", {cond, uncond22, slg24}},   // block 22: rows 0 and 2, two runs
      {"{uncond-skip22, cond, slg-skip22}", {uncond22, cond, slg22}},   // block 22: the middle row only
  };
  struct Grid {
    std::int64_t h, w;
  };
  const Grid grids[] = {{32, 48}, {c.h, c.w}};
  const bool checks = mi::varlen_debug_checks();
  mi::set_varlen_debug_checks(true);  // also validates the cu_seqlens / max_seqlen contract on device
  for (const Grid& g : grids) {
    const bool golden_grid = g.h == c.h && g.w == c.w;
    DevTensor synth(mi::DType::kF32, {128, g.h, g.w});
    if (!golden_grid) {
      std::vector<float> lv(static_cast<std::size_t>(synth.t.numel()));
      for (std::size_t i = 0; i < lv.size(); ++i) lv[i] = std::sin(0.37f * static_cast<float>(i));
      synth.buf.upload(lv.data(), lv.size() * 4);
    }
    const Tensor& latents = golden_grid ? c.latents->t : synth.t;
    for (const RowSet& set : sets) {
      const std::size_t n = set.rows.size();
      std::vector<std::int64_t> lengths;
      for (const Row& r : set.rows) lengths.push_back(r.text.dim(0));
      ASSERT_LE(MossDit::workspace_bytes(g.h, g.w, lengths) + MossDit::stream_workspace_bytes(*index_, residency_), arena_->available());
      std::vector<std::unique_ptr<DevTensor>> batched, alone;
      std::vector<Tensor> outs;
      for (std::size_t r = 0; r < n; ++r) {
        batched.push_back(std::make_unique<DevTensor>(mi::DType::kF32, std::initializer_list<std::int64_t>{128, g.h, g.w}));
        alone.push_back(std::make_unique<DevTensor>(mi::DType::kF32, std::initializer_list<std::int64_t>{128, g.h, g.w}));
        outs.push_back(batched.back()->t);
      }
      model_->forward(latents, c.timestep, set.rows, outs, stream());
      for (std::size_t r = 0; r < n; ++r) {
        model_->forward(latents, c.timestep, std::span<const Row>(&set.rows[r], 1), std::span<const Tensor>(&alone[r]->t, 1),
                        stream());
      }
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      for (std::size_t r = 0; r < n; ++r) {
        const auto b = download_f32(batched[r]->t);
        const auto a = download_f32(alone[r]->t);
        std::size_t differ = 0;
        for (std::size_t i = 0; i < a.size(); ++i) differ += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
        std::printf("[moss_dit varlen] grid %lldx%lld %-36s row %zu (S=%lld): %zu of %zu values differ from the row alone\n",
                    static_cast<long long>(g.h), static_cast<long long>(g.w), set.name, r,
                    static_cast<long long>(lengths[r]), differ, a.size());
        EXPECT_EQ(differ, 0u) << set.name << " row " << r << " grid " << g.h << "x" << g.w;
      }
    }
  }
  mi::set_varlen_debug_checks(checks);
}

// Cancel between blocks (the flag is set by the hook of block 10, so the check before block 11 must see it): forward
// throws Cancelled without issuing block 11, the stack is released, and the next forward is bit-identical to an
// uncancelled one (also while a never-set flag throttles the host). A flag already set when loading stops the
// loader at its first weight group.
TEST_F(MossDitTest, CancelBetweenBlocksAndBeforeLoading) {
  Case c = load_case("g_layout_1024");
  const Row rows[] = {{c.cond->t, SkipSet{}}, {c.uncond->t, c.uncond_skip}};
  DevTensor ref_c(mi::DType::kF32, {128, c.h, c.w}), ref_u(mi::DType::kF32, {128, c.h, c.w});
  DevTensor got_c(mi::DType::kF32, {128, c.h, c.w}), got_u(mi::DType::kF32, {128, c.h, c.w});
  const Tensor ref_outs[] = {ref_c.t, ref_u.t}, got_outs[] = {got_c.t, got_u.t};
  model_->forward(c.latents->t, c.timestep, rows, ref_outs, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));

  std::atomic<bool> cancel{false};
  model_->set_cancel(&cancel);
  int last_block = -1;
  const auto hook = [&](int block, const Tensor&) {
    last_block = block;
    if (block == 10) cancel.store(true);
  };
  const std::size_t used = arena_->used();
  EXPECT_THROW(model_->forward(c.latents->t, c.timestep, rows, got_outs, stream(), hook), mi::moss_dit::Cancelled);
  EXPECT_EQ(last_block, 10);
  EXPECT_EQ(arena_->used(), used);
  EXPECT_EQ(cudaStreamQuery(stream()), cudaSuccess);  // nothing left queued

  cancel.store(false);
  const auto same_bits = [](const Tensor& a, const Tensor& b) {
    const auto va = download_f32(a), vb = download_f32(b);
    return va.size() == vb.size() && std::memcmp(va.data(), vb.data(), va.size() * sizeof(float)) == 0;
  };
  for (const bool throttled : {true, false}) {
    model_->set_cancel(throttled ? &cancel : nullptr);
    model_->forward(c.latents->t, c.timestep, rows, got_outs, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    EXPECT_TRUE(same_bits(got_c.t, ref_c.t)) << "throttled " << throttled;
    EXPECT_TRUE(same_bits(got_u.t, ref_u.t)) << "throttled " << throttled;
  }

  // The loader: only the GEMM workspace is allocated before the first group's check.
  mi::runtime::DeviceArena small(std::size_t{128} << 20);
  cancel.store(true);
  EXPECT_THROW(MossDit(*index_, small, *streams_, false, &cancel), mi::moss_dit::Cancelled);
  EXPECT_LE(small.persistent_used(), mi::runtime::align_up(mi::kernels::Gemm::kWorkspaceBytes));
}

struct StepShape {
  const char* name;
  std::int64_t grid, s_cond, s_uncond;
  double p0_seconds;  // docs/p0-report.md, eager GPU-resident reference
};

// Memory high-water and steady-state timing of one CFG step (cond + uncond with skip {22}).
TEST_F(MossDitTest, StepMemoryAndTiming) {
  // Real text-encoder outputs cut to the benchmark lengths (values do not affect timing or memory).
  const auto te_long = mi::npy::load(golden_dir() / "g_long4096_1024.cond.te_last_hidden.npy").to_vector<float>();
  const StepShape shapes[] = {{"1024^2, 282+10 tokens", 64, 282, 10, 3.03}, {"1456^2, 4096+16 tokens", 91, 4096, 16, 9.10}};
  for (const StepShape& sh : shapes) {
    const auto row_te = [&](std::int64_t s) {
      return std::vector<float>(te_long.begin(), te_long.begin() + s * mi::moss_dit::kTextDim);
    };
    auto cond = refine(row_te(sh.s_cond), sh.s_cond);
    auto uncond = refine(row_te(sh.s_uncond), sh.s_uncond);
    DevTensor latents(mi::DType::kF32, {128, sh.grid, sh.grid});
    std::vector<float> lv(static_cast<std::size_t>(latents.t.numel()));
    for (std::size_t i = 0; i < lv.size(); ++i) lv[i] = std::sin(0.37f * static_cast<float>(i));
    latents.buf.upload(lv.data(), lv.size() * 4);
    DevTensor oc(mi::DType::kF32, {128, sh.grid, sh.grid}), ou(mi::DType::kF32, {128, sh.grid, sh.grid});
    SkipSet skip;
    skip.set(22);
    const Row rows[] = {{cond->t, SkipSet{}}, {uncond->t, skip}};
    const Tensor outs[] = {oc.t, ou.t};
    const std::int64_t lengths[] = {sh.s_cond, sh.s_uncond};
    const std::size_t predicted =
        MossDit::workspace_bytes(sh.grid, sh.grid, lengths) + MossDit::stream_workspace_bytes(*index_, residency_);

    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    arena_->reset_high_water();
    const std::size_t base = arena_->used();
    model_->forward(latents.t, 700.0f, rows, outs, stream());  // warm-up (GEMM plans)
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const std::size_t peak = arena_->high_water() - base;

    constexpr int kIters = 3;
    cudaEvent_t e0, e1;
    MI_CUDA_CHECK(cudaEventCreate(&e0));
    MI_CUDA_CHECK(cudaEventCreate(&e1));
    MI_CUDA_CHECK(cudaEventRecord(e0, stream()));
    for (int i = 0; i < kIters; ++i) model_->forward(latents.t, 700.0f, rows, outs, stream());
    MI_CUDA_CHECK(cudaEventRecord(e1, stream()));
    MI_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    MI_CUDA_CHECK(cudaEventDestroy(e0));
    MI_CUDA_CHECK(cudaEventDestroy(e1));
    const double step_s = ms / 1e3 / kIters;
    const std::int64_t hw = sh.grid * sh.grid;
    const double flops = forward_flops({hw + sh.s_cond, hw + sh.s_uncond}, {42, 41});
    std::printf("[moss_dit step %s] activations: predicted %.3f GiB, arena peak %.3f GiB; step %.3f s "
                "(P0 eager %.2f s, %.2fx), %.1f TFLOP -> %.1f TFLOPS\n",
                sh.name, static_cast<double>(predicted) / kGiB, static_cast<double>(peak) / kGiB, step_s, sh.p0_seconds,
                sh.p0_seconds / step_s, flops / 1e12, flops / step_s / 1e12);
    RecordProperty(std::string("step_s_") + std::to_string(sh.grid), std::to_string(step_s));
    EXPECT_EQ(peak, predicted);
    EXPECT_LE(static_cast<double>(peak), 2.0 * kGiB);
    const auto v = download_f32(oc.t);
    EXPECT_TRUE(std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }));
  }
}

}  // namespace
