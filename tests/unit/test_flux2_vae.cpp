// FLUX.2 VAE decoder against the reference pipeline's images: every golden final latent
// (tests/golden/<case>/final_latent.npy -> image.png), the 2048^2 bench latent and a non-square asset.
// Also: the torch-identical GroupNorm against torch (tests/data/flux2_vae, tools/reference/gen_flux2_vae_vectors.py),
// host vs device output, the workspace query vs the arena's high-water mark, and decode timing per mode.
// Exact tiled decode: byte-identical to the untiled decode in the same mode for every latent size, band height and
// budget tried (tiny bands near the halo limit, bands not dividing the image), with its memory high-water and time.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO_WARNINGS
#include <stb_image.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/flux2_vae/torch_group_norm.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "test_util.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;
using mi::flux2_vae::DecodeMode;
using mi::flux2_vae::Flux2VaeDecoder;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
// 8-bit agreement with the reference PNGs, set from the reference's own reproducibility (measured with the dev
// venv on this GPU): swapping only torch's SDPA backend (mem-efficient -> math) in this decode leaves 99.46-99.49%
// of the bytes identical (max diff 1) on g_short/g_fridge_1024, and torch's standalone decode reproduces the
// asset/2048^2 PNGs (decoded inside the full pipeline) to only 99.54% / 99.34%. Everything up to the mid-block
// attention is bit-identical here; the attention (kernels/vae kTf32x3) is fp32-accurate but not bit-identical to
// torch's kernel, and the TF32 operand rounding of every later conv amplifies those last-bit differences
// (4e-7 relative after the attention, 2e-3 at up_blocks.3). Bounds: at most 3x torch's own backend-swap
// fraction of differing bytes, and never more than 1 step.
constexpr double kMinIdentical = 0.984;
constexpr int kMaxAbsDiff = 1;
constexpr double kMaxMeanAbsDiff = 0.015;

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path source_dir() { return fs::path(MOSSINFER_SOURCE_DIR); }
fs::path golden_dir() { return source_dir() / "tests" / "golden"; }
fs::path data_dir() { return source_dir() / "tests" / "data" / "flux2_vae"; }
fs::path bench_latent_2048() { return source_dir() / "tools/reference/work/bench/images/ac16_2048.latent.npy"; }
fs::path asset_latent() { return source_dir() / "tests/reference_images/asset001_base.latent.npy"; }  // 1776 x 1184

std::vector<std::string> golden_cases() {
  std::vector<std::string> ids;
  for (const auto& e : fs::directory_iterator(golden_dir()))
    if (fs::is_regular_file(e.path() / "final_latent.npy") && fs::is_regular_file(e.path() / "image.png"))
      ids.push_back(e.path().filename().string());
  std::sort(ids.begin(), ids.end());
  return ids;
}

struct DevTensor {
  mi::DeviceBuffer buf;
  Tensor t;
  DevTensor(mi::DType dt, std::initializer_list<std::int64_t> shape) {
    t = Tensor::view(nullptr, dt, mi::Device::kCUDA, shape);
    buf = mi::DeviceBuffer(std::max<std::size_t>(t.nbytes(), 1));
    t.data = buf.get();
  }
};

// [1, 128, h, w] fp32 .npy -> device [128, h, w].
std::unique_ptr<DevTensor> load_latent(const fs::path& p) {
  const auto a = mi::npy::load(p);
  if (a.shape.size() != 4 || a.shape[0] != 1 || a.shape[1] != 128) throw std::runtime_error("bad latent " + p.string());
  auto d = std::make_unique<DevTensor>(mi::DType::kF32, std::initializer_list<std::int64_t>{128, a.shape[2], a.shape[3]});
  const auto v = a.to_vector<float>();
  d->buf.upload(v.data(), v.size() * sizeof(float));
  return d;
}

struct Image {
  int w = 0, h = 0;
  std::vector<std::uint8_t> rgb;
};
Image read_png(const fs::path& p) {
  Image img;
  int comp = 0;
  stbi_uc* data = stbi_load(p.string().c_str(), &img.w, &img.h, &comp, 3);
  if (data == nullptr) throw std::runtime_error("cannot decode " + p.string() + ": " + stbi_failure_reason());
  img.rgb.assign(data, data + static_cast<std::size_t>(img.w) * img.h * 3);
  stbi_image_free(data);
  return img;
}

struct Diff {
  double identical = 0, mean_abs = 0;
  int max_abs = 0;
};
Diff compare(const std::vector<std::uint8_t>& ref, const std::vector<std::uint8_t>& got) {
  std::size_t same = 0;
  double sum = 0;
  int mx = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const int d = std::abs(static_cast<int>(ref[i]) - static_cast<int>(got[i]));
    same += d == 0;
    sum += d;
    mx = std::max(mx, d);
  }
  const auto n = static_cast<double>(ref.size());
  return {static_cast<double>(same) / n, sum / n, mx};
}

const char* mode_name(const DecodeMode& m) {
  if (m.bf16_convs) return "fastest (bf16 convs, tf32 attn+linears)";
  if (m.attention == mi::vae::AttnPrecision::kTf32) return "fast attention (tf32 attn+linears)";
  return "reference (tf32 convs, tf32x3 attn)";
}

// tools/reference/gen_flux2_vae_vectors.py synth(): float32 ops only, reproduced exactly.
std::vector<float> synth(std::uint64_t seed, std::size_t n, float offset) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t h = (i * 2654435761ull + seed * 97ull) & 0xffffffffull;
    h ^= h >> 15;
    h = (h * 2246822519ull) & 0xffffffffull;
    h ^= h >> 13;
    const float u = static_cast<float>(h >> 8) * 0x1p-24f;
    v[i] = (u - 0.5f) * 8.0f + offset;
  }
  return v;
}

TEST(Flux2VaeTorchGroupNorm, BitIdenticalToTorch) {
  std::ifstream in(data_dir() / "manifest.json");
  ASSERT_TRUE(in) << "missing " << data_dir() / "manifest.json";
  std::stringstream ss;
  ss << in.rdbuf();
  const mi::json::Value manifest = mi::json::parse(ss.str());
  for (const auto& e : manifest.at("gn").as_array()) {
    const std::string name = e.at("name").as_string();
    const std::int64_t c = e.at("c").as_int(), h = e.at("h").as_int(), w = e.at("w").as_int();
    const bool silu = e.at("silu").as_bool();
    const auto x = synth(static_cast<std::uint64_t>(e.at("seed").as_int()), static_cast<std::size_t>(h * w * c),
                         static_cast<float>(e.at("offset").as_double()));
    const auto gamma = mi::npy::load(data_dir() / ("gn_" + name + ".gamma.npy")).to_vector<float>();
    const auto beta = mi::npy::load(data_dir() / ("gn_" + name + ".beta.npy")).to_vector<float>();
    const auto ref = mi::npy::load(data_dir() / ("gn_" + name + ".y.npy")).to_vector<float>();
    DevTensor xd(mi::DType::kF32, {1, h, w, c}), yd(mi::DType::kF32, {1, h, w, c});
    DevTensor gd(mi::DType::kF32, {c}), bd(mi::DType::kF32, {c});
    xd.buf.upload(x.data(), x.size() * 4);
    gd.buf.upload(gamma.data(), gamma.size() * 4);
    bd.buf.upload(beta.data(), beta.size() * 4);
    const std::size_t ws_bytes = mi::flux2_vae::torch_group_norm_workspace_bytes(xd.t, 32);
    mi::DeviceBuffer ws(ws_bytes);
    std::vector<float> got(x.size());
    for (int in_place = 0; in_place < 2; ++in_place) {
      const Tensor& y = in_place ? xd.t : yd.t;
      mi::flux2_vae::torch_group_norm(xd.t, 32, 1e-6f, gd.t, bd.t, silu, y, ws.get(), ws_bytes, nullptr);
      MI_CUDA_CHECK(cudaMemcpy(got.data(), y.data, got.size() * 4, cudaMemcpyDeviceToHost));
      std::size_t same = 0;
      for (std::size_t i = 0; i < got.size(); ++i) same += got[i] == ref[i];
      std::printf("[flux2_vae] torch_group_norm %-12s [1,%lld,%lld,%lld] silu=%d %s: %zu/%zu bit-identical\n", name.c_str(),
                  static_cast<long long>(h), static_cast<long long>(w), static_cast<long long>(c), silu,
                  in_place ? "in place" : "out of place", same, got.size());
      EXPECT_EQ(same, got.size()) << name;
    }
  }
}

class Flux2VaeTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::is_regular_file(bundle_dir() / "moss.mossidx")) return;
    index_ = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir()));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    // Largest decode here: 2048^2 (128x128 latent grid) in the reference mode.
    const std::size_t bytes = Flux2VaeDecoder::persistent_bytes(*index_, true) +
                              Flux2VaeDecoder::workspace_bytes(128, 128, DecodeMode::reference());
    std::printf("[flux2_vae] arena %.3f GiB (free %.3f GiB)\n", static_cast<double>(bytes) / kGiB,
                static_cast<double>(mi::runtime::DeviceArena::free_device_bytes()) / kGiB);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(bytes);
    model_ = std::make_unique<Flux2VaeDecoder>(*index_, *arena_, *streams_, /*bf16_weights=*/true);
    const auto& st = model_->load_stats();
    std::printf("[flux2_vae] load: %.1f MB in %.3f s, persistent %.1f MB (fp32 + bf16 conv weights)\n",
                static_cast<double>(st.bytes) / 1e6, st.seconds, static_cast<double>(arena_->persistent_used()) / 1e6);
  }
  static void TearDownTestSuite() {
    model_.reset();
    arena_.reset();
    streams_.reset();
    index_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "weights not present (" << bundle_dir() << ")";
  }
  cudaStream_t stream() const { return streams_->compute(); }

  std::vector<std::uint8_t> decode_host(const DevTensor& latent, const DecodeMode& mode) {
    const std::int64_t h = latent.t.dim(1) * 16, w = latent.t.dim(2) * 16;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(h * w * 3));
    const Tensor rgb = Tensor::view(out.data(), mi::DType::kU8, mi::Device::kCPU, {h, w, 3});
    model_->decode(latent.t, rgb, mode, stream());
    return out;
  }

  // Decodes and compares with the PNG; returns the diff (already printed).
  Diff check_against_png(const std::string& label, const fs::path& latent_path, const fs::path& png_path,
                         const DecodeMode& mode = DecodeMode::reference()) {
    const auto latent = load_latent(latent_path);
    const Image ref = read_png(png_path);
    EXPECT_EQ(ref.h, latent->t.dim(1) * 16);
    EXPECT_EQ(ref.w, latent->t.dim(2) * 16);
    const auto got = decode_host(*latent, mode);
    EXPECT_EQ(got.size(), ref.rgb.size());
    if (got.size() != ref.rgb.size()) return {};
    const Diff d = compare(ref.rgb, got);
    std::printf("[flux2_vae] %-16s %4dx%-4d %-40s identical %.5f%%  max|diff| %d  mean|diff| %.6f\n", label.c_str(), ref.w,
                ref.h, mode_name(mode), 100.0 * d.identical, d.max_abs, d.mean_abs);
    RecordProperty(label + ".identical", std::to_string(d.identical));
    RecordProperty(label + ".max_abs", std::to_string(d.max_abs));
    return d;
  }

  static std::unique_ptr<mi::weights::ModelIndex> index_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<Flux2VaeDecoder> model_;
};

std::unique_ptr<mi::weights::ModelIndex> Flux2VaeTest::index_;
std::unique_ptr<mi::runtime::StreamSet> Flux2VaeTest::streams_;
std::unique_ptr<mi::runtime::DeviceArena> Flux2VaeTest::arena_;
std::unique_ptr<Flux2VaeDecoder> Flux2VaeTest::model_;

TEST_F(Flux2VaeTest, GoldenFinalLatentsMatchReferenceImages) {
  const auto ids = golden_cases();
  ASSERT_EQ(ids.size(), 7u);
  for (const auto& id : ids) {
    const Diff d = check_against_png(id, golden_dir() / id / "final_latent.npy", golden_dir() / id / "image.png");
    EXPECT_GE(d.identical, kMinIdentical) << id;
    EXPECT_LE(d.max_abs, kMaxAbsDiff) << id;
    EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff) << id;
  }
}

TEST_F(Flux2VaeTest, Bench2048MatchesReferenceImage) {
  if (!fs::is_regular_file(bench_latent_2048())) GTEST_SKIP() << "reference bench output absent: " << bench_latent_2048();
  fs::path png = bench_latent_2048();
  png.replace_filename("ac16_2048.png");
  const Diff d = check_against_png("ac16_2048", bench_latent_2048(), png);
  EXPECT_GE(d.identical, kMinIdentical);
  EXPECT_LE(d.max_abs, kMaxAbsDiff);
  EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff);
}

TEST_F(Flux2VaeTest, NonSquareAssetMatchesReferenceImage) {
  fs::path png = asset_latent();
  png.replace_filename("asset001_base.png");
  const Diff d = check_against_png("asset001_base", asset_latent(), png);
  EXPECT_GE(d.identical, kMinIdentical);
  EXPECT_LE(d.max_abs, kMaxAbsDiff);
  EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff);
}

// Faster modes trade exactness for speed: report how far they move the 8-bit image.
TEST_F(Flux2VaeTest, FastModesStayCloseToReferenceImages) {
  for (const std::string id : {"g_fridge_1024", "g_layout_1456"}) {
    for (const DecodeMode& mode : {DecodeMode::fast_attention(), DecodeMode::fastest()}) {
      const Diff d = check_against_png(id, golden_dir() / id / "final_latent.npy", golden_dir() / id / "image.png", mode);
      EXPECT_LE(d.mean_abs, mode.bf16_convs ? 0.5 : 0.05) << id;
    }
  }
}

TEST_F(Flux2VaeTest, DeviceOutputEqualsHostOutputAndIsDeterministic) {
  const auto latent = load_latent(golden_dir() / "g_short_1024" / "final_latent.npy");
  const auto host = decode_host(*latent, DecodeMode::reference());
  DevTensor rgb(mi::DType::kU8, {1024, 1024, 3});
  model_->decode(latent->t, rgb.t, DecodeMode::reference(), stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  std::vector<std::uint8_t> dev(host.size());
  rgb.buf.download(dev.data(), dev.size());
  EXPECT_EQ(host, dev);
}

TEST_F(Flux2VaeTest, WorkspaceQueryCoversPeakAndIsReported) {
  struct Size {
    const char* name;
    std::int64_t h, w;
  };
  const Size sizes[] = {{"1024x1024", 64, 64}, {"1456x1456", 91, 91}, {"2048x2048", 128, 128}, {"1776x1184", 111, 74}};
  const std::size_t persistent_fp32 = Flux2VaeDecoder::persistent_bytes(*index_, false);
  std::printf("[flux2_vae] persistent: %.1f MB fp32 weights, %.1f MB with bf16 copies\n",
              static_cast<double>(persistent_fp32) / 1e6,
              static_cast<double>(Flux2VaeDecoder::persistent_bytes(*index_, true)) / 1e6);
  for (const Size& s : sizes) {
    for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::fastest()}) {
      const std::size_t ws = Flux2VaeDecoder::workspace_bytes(s.h, s.w, mode);
      std::printf("[flux2_vae] workspace %-9s %-40s %7.3f GiB (+ weights = %.3f GiB)\n", s.name, mode_name(mode),
                  static_cast<double>(ws) / kGiB,
                  static_cast<double>(ws + (mode.bf16_convs ? Flux2VaeDecoder::persistent_bytes(*index_, true) : persistent_fp32)) / kGiB);
    }
  }
  // The stack high-water of a real decode equals the query (all allocations happen up front).
  for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::fastest()}) {
    const auto latent = load_latent(asset_latent());
    DevTensor rgb(mi::DType::kU8, {latent->t.dim(1) * 16, latent->t.dim(2) * 16, 3});
    arena_->reset_high_water();
    model_->decode(latent->t, rgb.t, mode, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const std::size_t peak = arena_->high_water() - arena_->persistent_used();
    EXPECT_EQ(peak, Flux2VaeDecoder::workspace_bytes(latent->t.dim(1), latent->t.dim(2), mode));
    EXPECT_EQ(arena_->stack_used(), 0u);
  }
}

TEST_F(Flux2VaeTest, DecodeTiming) {
  struct Case {
    const char* name;
    fs::path latent;
    double ref_min_s, ref_max_s;  // P0 torch decode (docs/p0-report.md), incl. DiT parking at 1456/2048
  };
  // Asserted bound: the slow end of torch's range (at 1024^2 the reference mode runs at torch's speed: same exact
  // TF32 conv numerics).
  const Case cases[] = {{"1024x1024", golden_dir() / "g_fridge_1024" / "final_latent.npy", 0.56, 0.89},
                        {"1456x1456", golden_dir() / "g_fridge_1456" / "final_latent.npy", 5.09, 5.09},
                        {"2048x2048", bench_latent_2048(), 12.6, 12.6}};
  cudaEvent_t a = nullptr, b = nullptr;
  MI_CUDA_CHECK(cudaEventCreate(&a));
  MI_CUDA_CHECK(cudaEventCreate(&b));
  for (const Case& c : cases) {
    if (!fs::is_regular_file(c.latent)) {
      std::printf("[flux2_vae] timing %s skipped: %s absent\n", c.name, c.latent.string().c_str());
      continue;
    }
    const auto latent = load_latent(c.latent);
    DevTensor rgb(mi::DType::kU8, {latent->t.dim(1) * 16, latent->t.dim(2) * 16, 3});
    for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::fast_attention(), DecodeMode::fastest()}) {
      model_->decode(latent->t, rgb.t, mode, stream());  // warm-up (cuBLASLt heuristics, kernel attributes)
      constexpr int kIters = 3;
      MI_CUDA_CHECK(cudaEventRecord(a, stream()));
      for (int i = 0; i < kIters; ++i) model_->decode(latent->t, rgb.t, mode, stream());
      MI_CUDA_CHECK(cudaEventRecord(b, stream()));
      MI_CUDA_CHECK(cudaEventSynchronize(b));
      float ms = 0;
      MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      const double s = ms / 1e3 / kIters;
      std::printf("[flux2_vae] decode %-9s %-40s %7.3f s  (torch %.2f-%.2f s: %.1f-%.1fx)\n", c.name, mode_name(mode), s,
                  c.ref_min_s, c.ref_max_s, c.ref_min_s / s, c.ref_max_s / s);
      RecordProperty(std::string(c.name) + "." + (mode.bf16_convs ? "fastest" : mode.attention == mi::vae::AttnPrecision::kTf32 ? "fast_attn" : "reference") + ".s",
                     std::to_string(s));
      if (mi::testing::kAssertTimings) {
        EXPECT_LT(s, c.ref_max_s) << c.name << " " << mode_name(mode);
      }
    }
  }
  MI_CUDA_CHECK(cudaEventDestroy(a));
  MI_CUDA_CHECK(cudaEventDestroy(b));
}

// ---- Exact tiled decode -------------------------------------------------------------------------------------

// The reference precision with exact (tileable) GroupNorm statistics: what tiled decodes are byte-identical to.
constexpr DecodeMode exact_reference() {
  DecodeMode m = DecodeMode::reference();
  m.torch_group_norm = false;
  return m;
}

double max_abs_diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  int mx = 0;
  for (std::size_t i = 0; i < a.size(); ++i) mx = std::max(mx, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
  return mx;
}

TEST(Flux2VaeTiledPlan, BudgetsAndQueries) {
  struct Size {
    const char* name;
    std::int64_t h, w;
  };
  const Size sizes[] = {{"1024x1024", 64, 64}, {"1456x1456", 91, 91}, {"2048x2048", 128, 128}, {"1184x1776", 74, 111}};
  for (const DecodeMode& mode : {exact_reference(), DecodeMode::fastest()}) {
    for (const Size& s : sizes) {
      std::printf("[flux2_vae] tiled %-9s %-40s untiled %.3f GiB\n", s.name, mode_name(mode),
                  static_cast<double>(Flux2VaeDecoder::workspace_bytes(s.h, s.w, mode)) / kGiB);
      for (const int tile : {8, 16, 64, 128, 256}) {
        const mi::flux2_vae::TilePlan p = Flux2VaeDecoder::plan_tiled_min_memory(s.h, s.w, mode, tile);
        EXPECT_EQ(p.workspace_bytes, Flux2VaeDecoder::tiled_workspace_bytes(s.h, s.w, tile, mode));
        std::printf("[flux2_vae]   min memory  %s\n", p.describe().c_str());
      }
      for (const double gib : {0.75, 1.0, 1.5, 2.0}) {
        const auto budget = static_cast<std::size_t>(gib * kGiB);
        const auto p = Flux2VaeDecoder::plan_tiled(s.h, s.w, mode, budget);
        if (!p) {
          std::printf("[flux2_vae]   budget %.2f GiB: no schedule\n", gib);
          continue;
        }
        EXPECT_LE(p->workspace_bytes, budget);
        std::printf("[flux2_vae]   budget %.2f GiB: %s\n", gib, p->describe().c_str());
      }
    }
  }
  // Plan step 23 targets (reference precision): 1456^2 within 1.0 GiB, 2048^2 within 1.5 GiB.
  EXPECT_TRUE(Flux2VaeDecoder::plan_tiled(91, 91, exact_reference(), std::size_t{1} << 30).has_value());
  EXPECT_TRUE(Flux2VaeDecoder::plan_tiled(128, 128, exact_reference(), std::size_t{3} << 29).has_value());
  EXPECT_THROW(Flux2VaeDecoder::plan_tiled_min_memory(64, 64, exact_reference(), 12), std::invalid_argument);
}

class Flux2VaeTiledTest : public Flux2VaeTest {
 protected:
  struct Run {
    std::vector<std::uint8_t> rgb;
    double seconds = 0;
    std::size_t peak = 0;  // stack high-water of the decode
  };

  // Decodes into device memory; timed with events after an optional warm-up.
  template <typename Fn>
  Run timed(const DevTensor& latent, bool warm_up, Fn&& fn) {
    const std::int64_t h = latent.t.dim(1) * 16, w = latent.t.dim(2) * 16;
    DevTensor rgb(mi::DType::kU8, {h, w, 3});
    MI_CUDA_CHECK(cudaMemset(rgb.t.data, 0xAB, static_cast<std::size_t>(rgb.t.nbytes())));
    if (warm_up) fn(rgb.t);
    cudaEvent_t a = nullptr, b = nullptr;
    MI_CUDA_CHECK(cudaEventCreate(&a));
    MI_CUDA_CHECK(cudaEventCreate(&b));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    arena_->reset_high_water();
    MI_CUDA_CHECK(cudaEventRecord(a, stream()));
    fn(rgb.t);
    MI_CUDA_CHECK(cudaEventRecord(b, stream()));
    MI_CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    MI_CUDA_CHECK(cudaEventDestroy(a));
    MI_CUDA_CHECK(cudaEventDestroy(b));
    Run r;
    r.seconds = ms / 1e3;
    r.peak = arena_->high_water() - arena_->persistent_used();
    r.rgb.resize(static_cast<std::size_t>(rgb.t.nbytes()));
    rgb.buf.download(r.rgb.data(), r.rgb.size());
    EXPECT_EQ(arena_->stack_used(), 0u);
    return r;
  }

  // Untiled decodes (exact and torch GroupNorm) as the baseline, then every plan: byte-identical to the exact
  // untiled decode, within 1/255 of the torch-GroupNorm one, stack high-water equal to the plan's workspace.
  void check_plans(const std::string& label, const fs::path& latent_path, const DecodeMode& mode,
                   const std::vector<mi::flux2_vae::TilePlan>& plans, const fs::path& png = {}) {
    const auto latent = load_latent(latent_path);
    const std::int64_t h = latent->t.dim(1), w = latent->t.dim(2);
    const Run untiled = timed(*latent, true, [&](const Tensor& rgb) { model_->decode(latent->t, rgb, mode, stream()); });
    DecodeMode torch_gn = mode;
    torch_gn.torch_group_norm = true;
    const Run torch_run = mode.bf16_convs ? Run{}
                                          : timed(*latent, false, [&](const Tensor& rgb) { model_->decode(latent->t, rgb, torch_gn, stream()); });
    std::printf("[flux2_vae] tiled %-14s %4lldx%-4lld %-40s untiled %.3f s, %.3f GiB\n", label.c_str(),
                static_cast<long long>(w * 16), static_cast<long long>(h * 16), mode_name(mode), untiled.seconds,
                static_cast<double>(untiled.peak) / kGiB);
    if (!torch_run.rgb.empty())
      std::printf("[flux2_vae]   untiled exact-GN vs torch-GN: identical %.5f%%, max|diff| %.0f\n",
                  100.0 * compare(torch_run.rgb, untiled.rgb).identical, max_abs_diff(torch_run.rgb, untiled.rgb));
    if (!png.empty()) {
      const Image ref = read_png(png);
      const Diff d = compare(ref.rgb, untiled.rgb);
      std::printf("[flux2_vae]   untiled exact-GN vs reference PNG: identical %.5f%%, max|diff| %d\n", 100.0 * d.identical, d.max_abs);
      EXPECT_LE(d.max_abs, kMaxAbsDiff);
      EXPECT_GE(d.identical, kMinIdentical);
    }
    for (const auto& plan : plans) {
      const Run t = timed(*latent, false, [&](const Tensor& rgb) { model_->decode_tiled(latent->t, rgb, mode, plan, stream()); });
      const bool same = t.rgb == untiled.rgb;
      std::printf("[flux2_vae]   %s  peak %.3f GiB  %.3f s (%.2fx untiled)  %s\n", plan.describe().c_str(),
                  static_cast<double>(t.peak) / kGiB, t.seconds, t.seconds / untiled.seconds,
                  same ? "byte-identical" : "DIFFERS");
      EXPECT_TRUE(same) << label << " " << plan.describe() << ": identical " << compare(untiled.rgb, t.rgb).identical
                        << " max|diff| " << max_abs_diff(untiled.rgb, t.rgb);
      EXPECT_EQ(t.peak, plan.workspace_bytes) << plan.describe();
      if (!torch_run.rgb.empty()) {
        EXPECT_LE(max_abs_diff(torch_run.rgb, t.rgb), 1.0) << plan.describe();
      }
      RecordProperty(label + ".tile" + std::to_string(plan.tile) + ".gib", std::to_string(static_cast<double>(t.peak) / kGiB));
      RecordProperty(label + ".tile" + std::to_string(plan.tile) + ".x_untiled", std::to_string(t.seconds / untiled.seconds));
    }
  }

  static std::vector<mi::flux2_vae::TilePlan> plans(std::int64_t h, std::int64_t w, const DecodeMode& mode,
                                                    std::initializer_list<int> min_memory_tiles,
                                                    std::initializer_list<double> budgets_gib) {
    std::vector<mi::flux2_vae::TilePlan> v;
    for (const int t : min_memory_tiles) v.push_back(Flux2VaeDecoder::plan_tiled_min_memory(h, w, mode, t));
    for (const double gib : budgets_gib) {
      const auto p = Flux2VaeDecoder::plan_tiled(h, w, mode, static_cast<std::size_t>(gib * kGiB));
      EXPECT_TRUE(p.has_value()) << gib << " GiB";
      if (p) v.push_back(*p);
    }
    return v;
  }
};

TEST_F(Flux2VaeTiledTest, ByteIdentical1024) {
  // Tiles: 8 px (one latent row per band at H/8, far below the halo), 40 (does not divide 1024), the whole image
  // (every segment one band: the executor without halos), and two budgets.
  check_plans("g_short_1024", golden_dir() / "g_short_1024" / "final_latent.npy", exact_reference(),
              plans(64, 64, exact_reference(), {8, 40, 1024}, {0.5, 0.75}), golden_dir() / "g_short_1024" / "image.png");
}

TEST_F(Flux2VaeTiledTest, ByteIdentical1456) {
  check_plans("g_fridge_1456", golden_dir() / "g_fridge_1456" / "final_latent.npy", exact_reference(),
              plans(91, 91, exact_reference(), {16, 72}, {1.0, 1.5}), golden_dir() / "g_fridge_1456" / "image.png");
}

TEST_F(Flux2VaeTiledTest, ByteIdenticalNonSquareAsset) {
  const auto a = mi::npy::load(asset_latent());
  fs::path png = asset_latent();
  png.replace_filename("asset001_base.png");
  check_plans("asset001_base", asset_latent(), exact_reference(), plans(a.shape[2], a.shape[3], exact_reference(), {24, 168}, {1.0}), png);
}

TEST_F(Flux2VaeTiledTest, ByteIdentical2048) {
  if (!fs::is_regular_file(bench_latent_2048())) GTEST_SKIP() << "reference bench output absent: " << bench_latent_2048();
  fs::path png = bench_latent_2048();
  png.replace_filename("ac16_2048.png");
  check_plans("ac16_2048", bench_latent_2048(), exact_reference(), plans(128, 128, exact_reference(), {136}, {1.5, 2.0, 2.5}), png);
}

TEST_F(Flux2VaeTiledTest, ByteIdenticalBf16Mode) {
  check_plans("g_layout_1024", golden_dir() / "g_layout_1024" / "final_latent.npy", DecodeMode::fastest(),
              plans(64, 64, DecodeMode::fastest(), {16, 104}, {0.4}));
}

// Cancel per band: set from the head's last stage hook (the first band's check must see it), and from another thread
// in the middle of the bands (the decode must stop within a few bands). Either way decode_tiled throws Cancelled with
// nothing queued and the stack released, and the next decode is byte-identical.
TEST_F(Flux2VaeTiledTest, CancelBetweenBands) {
  const auto latent = load_latent(golden_dir() / "g_short_1024" / "final_latent.npy");
  const std::int64_t h = latent->t.dim(1), w = latent->t.dim(2);
  const DecodeMode mode = exact_reference();
  const auto plan = Flux2VaeDecoder::plan_tiled_min_memory(h, w, mode, 8);  // the most bands
  DevTensor rgb(mi::DType::kU8, {h * 16, w * 16, 3});
  std::vector<std::uint8_t> want(static_cast<std::size_t>(rgb.t.nbytes())), got(want.size());
  const auto t0 = std::chrono::steady_clock::now();
  model_->decode_tiled(latent->t, rgb.t, mode, plan, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const double full_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  rgb.buf.download(want.data(), want.size());

  std::atomic<bool> cancel{false};
  model_->set_cancel(&cancel);
  const mi::flux2_vae::StageHook hook = [&](const char* stage, const Tensor&) {
    if (std::string(stage) == "mid.attention") cancel.store(true);
  };
  EXPECT_THROW(model_->decode_tiled(latent->t, rgb.t, mode, plan, stream(), hook), mi::flux2_vae::Cancelled);
  EXPECT_EQ(arena_->stack_used(), 0u);
  EXPECT_EQ(cudaStreamQuery(stream()), cudaSuccess);

  cancel.store(false);
  std::chrono::steady_clock::time_point set_at;
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::duration<double>(full_s / 3));
    set_at = std::chrono::steady_clock::now();
    cancel.store(true);
  });
  bool cancelled = false;
  try {
    model_->decode_tiled(latent->t, rgb.t, mode, plan, stream());
  } catch (const mi::flux2_vae::Cancelled&) {
    cancelled = true;
  }
  const auto done_at = std::chrono::steady_clock::now();
  canceller.join();
  EXPECT_TRUE(cancelled);
  const double latency_s = std::chrono::duration<double>(done_at - set_at).count();
  std::printf("[flux2_vae] cancel mid-bands: stopped %.1f ms after the flag (full tiled decode %.0f ms, %s)\n", latency_s * 1e3,
              full_s * 1e3, plan.describe().c_str());
  EXPECT_LT(latency_s, full_s / 4 + 0.05);
  EXPECT_EQ(arena_->stack_used(), 0u);

  cancel.store(false);  // a set-but-clear flag only throttles
  model_->decode_tiled(latent->t, rgb.t, mode, plan, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  rgb.buf.download(got.data(), got.size());
  EXPECT_TRUE(got == want);
  model_->set_cancel(nullptr);
}

TEST_F(Flux2VaeTiledTest, DecodeWithinBudget) {
  const auto latent = load_latent(golden_dir() / "g_fridge_1456" / "final_latent.npy");
  const std::int64_t h = latent->t.dim(1), w = latent->t.dim(2);
  DevTensor rgb(mi::DType::kU8, {h * 16, w * 16, 3});
  std::vector<std::uint8_t> got(static_cast<std::size_t>(rgb.t.nbytes())), want(got.size());
  const Tensor host_want = Tensor::view(want.data(), mi::DType::kU8, mi::Device::kCPU, {h * 16, w * 16, 3});
  // Fits: untiled, torch GroupNorm kept.
  EXPECT_FALSE(model_->decode_within(latent->t, rgb.t, DecodeMode::reference(),
                                     Flux2VaeDecoder::workspace_bytes(h, w, DecodeMode::reference()), stream()));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  rgb.buf.download(got.data(), got.size());
  model_->decode(latent->t, host_want, DecodeMode::reference(), stream());
  EXPECT_EQ(got, want);
  // 1 GiB: tiled with exact GroupNorm statistics, equal to the untiled exact decode (host output path).
  const Tensor host_got = Tensor::view(got.data(), mi::DType::kU8, mi::Device::kCPU, {h * 16, w * 16, 3});
  const auto plan = model_->decode_within(latent->t, host_got, DecodeMode::reference(), std::size_t{1} << 30, stream());
  ASSERT_TRUE(plan.has_value());
  EXPECT_LE(plan->workspace_bytes, std::size_t{1} << 30);
  model_->decode(latent->t, host_want, exact_reference(), stream());
  EXPECT_EQ(got, want);
  EXPECT_THROW(model_->decode_within(latent->t, host_got, DecodeMode::reference(), std::size_t{64} << 20, stream()), std::runtime_error);
  EXPECT_THROW(model_->decode_tiled(latent->t, host_got, DecodeMode::reference(), *plan, stream()), std::invalid_argument);
}

}  // namespace
