// FLUX.2 VAE decoder against the reference pipeline's images: every golden final latent
// (tests/golden/<case>/final_latent.npy -> image.png), the 2048^2 bench latent and a non-square asset.
// Also: the torch-identical GroupNorm against torch (tests/data/flux2_vae, from the PyTorch reference),
// host vs device output, the workspace query vs the arena's high-water mark, and decode timing per mode.
// Exact tiled decode: byte-identical to the untiled decode in the same mode for every latent size, band height and
// budget tried (tiny bands near the halo limit, bands not dividing the image), with its memory high-water and time.
// The reference-image tests run the pipeline's precision (DecodeMode::reference(), exact GroupNorm) and the
// torch-GroupNorm parity mode (torch_reference()).
// The full FLUX.2 decoder (widths 128/256/512/512, loaded from the single-file flux2-vae.safetensors) runs the same
// suites (Flux2VaeFull*) against diffusers decodes of the same latents made with the dev venv: a tracked crop
// (tests/data/flux2_vae/full_crop.*) and, when present locally, full-size decodes of every golden latent. Also: the
// single-file loader's key mapping and the mid attention at both head dims. MI_FLUX2_VAE_DUMP=<dir> writes every decode
// compared with a reference image as <dir>/<label>.<decoder>.<mode>.ppm (for LPIPS outside the tests).
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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/vae/attention.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/flux2_vae/torch_group_norm.h"
#include "models/flux2_vae/vae_file.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "test_util.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"
#include "weights/safetensors.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;
using mi::flux2_vae::DecodeMode;
using mi::flux2_vae::DecoderConfig;
using mi::flux2_vae::Flux2VaeDecoder;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
// 8-bit agreement with the reference PNGs, set from the reference's own reproducibility (measured with the dev
// venv on this GPU): swapping only torch's SDPA backend (mem-efficient -> math) in this decode leaves 99.46-99.49%
// of the bytes identical (max diff 1) on g_short/g_fridge_1024, and torch's standalone decode reproduces the
// asset/2048^2 PNGs (decoded inside the full pipeline) to only 99.54% / 99.34%. In torch_reference() everything up
// to the mid-block attention is bit-identical here; the attention (kernels/vae kTf32x3) is fp32-accurate but not
// bit-identical to torch's kernel, and the TF32 operand rounding of every later conv amplifies those last-bit
// differences (4e-7 relative after the attention, 2e-3 at up_blocks.3). reference()'s exact GroupNorm statistics
// differ from torch's in the last bits as well. Bounds (both modes): at most 3x torch's own backend-swap fraction of
// differing bytes, and never more than 1 step.
constexpr double kMinIdentical = 0.984;
constexpr int kMaxAbsDiff = 1;
constexpr double kMaxMeanAbsDiff = 0.015;
// The full decoder meets the same bounds against its diffusers references (decoded with the bundle's latent
// statistics, as the pipeline does).

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path source_dir() { return fs::path(MOSSINFER_SOURCE_DIR); }
fs::path golden_dir() { return source_dir() / "tests" / "golden"; }
fs::path data_dir() { return source_dir() / "tests" / "data" / "flux2_vae"; }
fs::path bench_latent_2048() { return source_dir() / "tools/reference/work/bench/images/ac16_2048.latent.npy"; }
fs::path asset_latent() { return source_dir() / "tests/reference_images/asset001_base.latent.npy"; }  // 1776 x 1184
fs::path community_file(const char* env_name, const char* name) {
  const char* env = std::getenv(env_name);
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/community/comfy-org") / name;
}
fs::path full_vae_file() { return community_file("MOSS_FULL_VAE", "flux2-vae.safetensors"); }
fs::path small_vae_file() { return community_file("MOSS_SMALL_VAE_FILE", "flux2-vae_small.safetensors"); }
// Diffusers decodes of the golden latents with the full decoder (local, made by the dev venv's reference script).
fs::path full_ref_dir() { return source_dir() / "tools/reference/work/flux2_vae_full"; }

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
  return m.torch_group_norm ? "torch reference (tf32x3 attn, torch GN)" : "reference (tf32 convs, tf32x3 attn)";
}

// The vector generator's synth(): float32 ops only, reproduced exactly.
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

void write_ppm(const fs::path& p, std::int64_t w, std::int64_t h, const std::vector<std::uint8_t>& rgb) {
  std::ofstream f(p, std::ios::binary);
  f << "P6\n" << w << " " << h << "\n255\n";
  f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

// kFull = false: the bundle's small decoder (moss.mossidx); true: the full decoder from its single file.
template <bool kFull>
class VaeFixture : public ::testing::Test {
 protected:
  static const char* tag() { return kFull ? "[flux2_vae full]" : "[flux2_vae]"; }
  static const char* decoder_name() { return kFull ? "full" : "small"; }

  static void SetUpTestSuite() {
    if (!fs::is_regular_file(bundle_dir() / "moss.mossidx")) return;
    // The full decoder decodes with the bundle's latent statistics (as the official pipeline does).
    if (kFull && !fs::is_regular_file(full_vae_file())) return;
    auto bundle = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir()));
    if (kFull) {
      index_ = std::make_unique<mi::weights::ModelIndex>(mi::flux2_vae::index_vae_file(full_vae_file()));
      bundle_ = std::move(bundle);
    } else {
      index_ = std::move(bundle);
    }
    const mi::weights::ModelIndex& index = *index_;
    const mi::weights::ModelIndex* stats = bundle_.get();
    const DecoderConfig cfg = mi::flux2_vae::decoder_config(index);
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    // Largest decode here: 2048^2 (128x128 latent grid) in the fp32 modes.
    const std::size_t bytes = Flux2VaeDecoder::persistent_bytes(index, true, stats) +
                              std::max(Flux2VaeDecoder::workspace_bytes(cfg, 128, 128, DecodeMode::reference()),
                                       Flux2VaeDecoder::workspace_bytes(cfg, 128, 128, DecodeMode::torch_reference()));
    std::printf("%s arena %.3f GiB (free %.3f GiB)\n", tag(), static_cast<double>(bytes) / kGiB,
                static_cast<double>(mi::runtime::DeviceArena::free_device_bytes()) / kGiB);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(bytes);
    model_ = std::make_unique<Flux2VaeDecoder>(index, *arena_, *streams_, /*bf16_weights=*/true, /*direct_io=*/false, stats);
    EXPECT_EQ(model_->config(), kFull ? DecoderConfig::full() : DecoderConfig::small());
    const auto& st = model_->load_stats();
    std::printf("%s load: %.1f MB in %.3f s, persistent %.1f MB (fp32 + bf16 conv weights)\n", tag(),
                static_cast<double>(st.bytes) / 1e6, st.seconds, static_cast<double>(arena_->persistent_used()) / 1e6);
  }
  static void TearDownTestSuite() {
    model_.reset();
    arena_.reset();
    streams_.reset();
    index_.reset();
    bundle_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "weights not present (" << (kFull ? full_vae_file() : bundle_dir()) << ")";
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
    std::printf("%s %-16s %4dx%-4d %-40s identical %.5f%%  max|diff| %d  mean|diff| %.6f\n", tag(), label.c_str(), ref.w,
                ref.h, mode_name(mode), 100.0 * d.identical, d.max_abs, d.mean_abs);
    if (const char* dump = std::getenv("MI_FLUX2_VAE_DUMP"); dump != nullptr && *dump != '\0') {
      const char* m = mode.bf16_convs ? "fastest" : mode.attention == mi::vae::AttnPrecision::kTf32 ? "fast_attn"
                                                  : mode.torch_group_norm                         ? "torch_reference"
                                                                                                  : "reference";
      write_ppm(fs::path(dump) / (label + "." + decoder_name() + "." + m + ".ppm"), ref.w, ref.h, got);
    }
    RecordProperty(label + ".identical", std::to_string(d.identical));
    RecordProperty(label + ".max_abs", std::to_string(d.max_abs));
    return d;
  }

  static std::unique_ptr<mi::weights::ModelIndex> index_;   // the decoder weights
  static std::unique_ptr<mi::weights::ModelIndex> bundle_;  // full decoder: the bundle, for the latent statistics
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<Flux2VaeDecoder> model_;
};

template <bool kFull>
std::unique_ptr<mi::weights::ModelIndex> VaeFixture<kFull>::index_;
template <bool kFull>
std::unique_ptr<mi::weights::ModelIndex> VaeFixture<kFull>::bundle_;
template <bool kFull>
std::unique_ptr<mi::runtime::StreamSet> VaeFixture<kFull>::streams_;
template <bool kFull>
std::unique_ptr<mi::runtime::DeviceArena> VaeFixture<kFull>::arena_;
template <bool kFull>
std::unique_ptr<Flux2VaeDecoder> VaeFixture<kFull>::model_;

using Flux2VaeTest = VaeFixture<false>;
using Flux2VaeFullTest = VaeFixture<true>;

TEST_F(Flux2VaeTest, GoldenFinalLatentsMatchReferenceImages) {
  const auto ids = golden_cases();
  ASSERT_EQ(ids.size(), 7u);
  for (const auto& id : ids) {
    for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference()}) {
      const Diff d = check_against_png(id, golden_dir() / id / "final_latent.npy", golden_dir() / id / "image.png", mode);
      EXPECT_GE(d.identical, kMinIdentical) << id << " " << mode_name(mode);
      EXPECT_LE(d.max_abs, kMaxAbsDiff) << id << " " << mode_name(mode);
      EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff) << id << " " << mode_name(mode);
    }
  }
}

TEST_F(Flux2VaeTest, Bench2048MatchesReferenceImage) {
  if (!fs::is_regular_file(bench_latent_2048())) GTEST_SKIP() << "reference bench output absent: " << bench_latent_2048();
  fs::path png = bench_latent_2048();
  png.replace_filename("ac16_2048.png");
  for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference()}) {
    const Diff d = check_against_png("ac16_2048", bench_latent_2048(), png, mode);
    EXPECT_GE(d.identical, kMinIdentical) << mode_name(mode);
    EXPECT_LE(d.max_abs, kMaxAbsDiff) << mode_name(mode);
    EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff) << mode_name(mode);
  }
}

TEST_F(Flux2VaeTest, NonSquareAssetMatchesReferenceImage) {
  fs::path png = asset_latent();
  png.replace_filename("asset001_base.png");
  for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference()}) {
    const Diff d = check_against_png("asset001_base", asset_latent(), png, mode);
    EXPECT_GE(d.identical, kMinIdentical) << mode_name(mode);
    EXPECT_LE(d.max_abs, kMaxAbsDiff) << mode_name(mode);
    EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff) << mode_name(mode);
  }
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

// A plan made for the full decoder is refused (the full decoder refuses the small one's, Flux2VaeFullTest).
TEST_F(Flux2VaeTest, RejectsFullDecoderPlan) {
  const auto latent = load_latent(golden_dir() / "g_short_1024" / "final_latent.npy");
  DevTensor rgb(mi::DType::kU8, {1024, 1024, 3});
  EXPECT_THROW(model_->decode_tiled(latent->t, rgb.t, DecodeMode::reference(),
                                    Flux2VaeDecoder::plan_tiled_min_memory(DecoderConfig::full(), 64, 64, DecodeMode::reference(), 64),
                                    stream()),
               std::invalid_argument);
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
      const std::size_t ws = Flux2VaeDecoder::workspace_bytes(DecoderConfig::small(), s.h, s.w, mode);
      std::printf("[flux2_vae] workspace %-9s %-40s %7.3f GiB (+ weights = %.3f GiB)\n", s.name, mode_name(mode),
                  static_cast<double>(ws) / kGiB,
                  static_cast<double>(ws + (mode.bf16_convs ? Flux2VaeDecoder::persistent_bytes(*index_, true) : persistent_fp32)) / kGiB);
    }
  }
  // The stack high-water of a real decode equals the query (all allocations happen up front).
  for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference(), DecodeMode::fastest()}) {
    const auto latent = load_latent(asset_latent());
    DevTensor rgb(mi::DType::kU8, {latent->t.dim(1) * 16, latent->t.dim(2) * 16, 3});
    arena_->reset_high_water();
    model_->decode(latent->t, rgb.t, mode, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const std::size_t peak = arena_->high_water() - arena_->persistent_used();
    EXPECT_EQ(peak, Flux2VaeDecoder::workspace_bytes(DecoderConfig::small(), latent->t.dim(1), latent->t.dim(2), mode));
    EXPECT_EQ(arena_->stack_used(), 0u);
  }
}

TEST_F(Flux2VaeTest, DecodeTiming) {
  struct Case {
    const char* name;
    fs::path latent;
    double ref_min_s, ref_max_s;  // P0 torch decode, measured on the dev box, incl. DiT parking at 1456/2048
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
    for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference(), DecodeMode::fast_attention(),
                                   DecodeMode::fastest()}) {
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
      const char* key = mode.bf16_convs                                   ? "fastest"
                        : mode.attention == mi::vae::AttnPrecision::kTf32 ? "fast_attn"
                        : mode.torch_group_norm                           ? "torch_reference"
                                                                          : "reference";
      RecordProperty(std::string(c.name) + "." + key + ".s", std::to_string(s));
      if (mi::testing::kAssertTimings) {
        EXPECT_LT(s, c.ref_max_s) << c.name << " " << mode_name(mode);
      }
    }
  }
  MI_CUDA_CHECK(cudaEventDestroy(a));
  MI_CUDA_CHECK(cudaEventDestroy(b));
}

// ---- Exact tiled decode -------------------------------------------------------------------------------------

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
  for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::fastest()}) {
    for (const Size& s : sizes) {
      std::printf("[flux2_vae] tiled %-9s %-40s untiled %.3f GiB\n", s.name, mode_name(mode),
                  static_cast<double>(Flux2VaeDecoder::workspace_bytes(DecoderConfig::small(), s.h, s.w, mode)) / kGiB);
      for (const int tile : {8, 16, 64, 128, 256}) {
        const mi::flux2_vae::TilePlan p = Flux2VaeDecoder::plan_tiled_min_memory(DecoderConfig::small(), s.h, s.w, mode, tile);
        EXPECT_EQ(p.workspace_bytes, Flux2VaeDecoder::tiled_workspace_bytes(DecoderConfig::small(), s.h, s.w, tile, mode));
        std::printf("[flux2_vae]   min memory  %s\n", p.describe().c_str());
      }
      for (const double gib : {0.75, 1.0, 1.5, 2.0}) {
        const auto budget = static_cast<std::size_t>(gib * kGiB);
        const auto p = Flux2VaeDecoder::plan_tiled(DecoderConfig::small(), s.h, s.w, mode, budget);
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
  EXPECT_TRUE(Flux2VaeDecoder::plan_tiled(DecoderConfig::small(), 91, 91, DecodeMode::reference(), std::size_t{1} << 30).has_value());
  EXPECT_TRUE(Flux2VaeDecoder::plan_tiled(DecoderConfig::small(), 128, 128, DecodeMode::reference(), std::size_t{3} << 29).has_value());
  EXPECT_THROW(Flux2VaeDecoder::plan_tiled_min_memory(DecoderConfig::small(), 64, 64, DecodeMode::reference(), 12), std::invalid_argument);
}

template <bool kFull>
class TiledFixture : public VaeFixture<kFull> {
 protected:
  using Base = VaeFixture<kFull>;
  using Base::arena_;
  using Base::model_;
  using Base::stream;
  using Base::tag;

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

  // Untiled decodes (exact and torch GroupNorm) as the baseline, then every plan: byte-identical to the untiled decode
  // in `mode` (reference(): the pipeline's decode is the same tiled and untiled), within 1/255 of the torch-GroupNorm
  // one, stack high-water equal to the plan's workspace.
  void check_plans(const std::string& label, const fs::path& latent_path, const DecodeMode& mode,
                   const std::vector<mi::flux2_vae::TilePlan>& plans, const fs::path& png = {}) {
    const auto latent = load_latent(latent_path);
    const std::int64_t h = latent->t.dim(1), w = latent->t.dim(2);
    const Run untiled = timed(*latent, true, [&](const Tensor& rgb) { model_->decode(latent->t, rgb, mode, stream()); });
    DecodeMode torch_gn = mode;
    torch_gn.torch_group_norm = true;
    const Run torch_run = mode.bf16_convs ? Run{}
                                          : timed(*latent, false, [&](const Tensor& rgb) { model_->decode(latent->t, rgb, torch_gn, stream()); });
    std::printf("%s tiled %-14s %4lldx%-4lld %-40s untiled %.3f s, %.3f GiB\n", tag(), label.c_str(),
                static_cast<long long>(w * 16), static_cast<long long>(h * 16), mode_name(mode), untiled.seconds,
                static_cast<double>(untiled.peak) / kGiB);
    if (!torch_run.rgb.empty())
      std::printf("%s   untiled exact-GN vs torch-GN: identical %.5f%%, max|diff| %.0f\n", tag(),
                  100.0 * compare(torch_run.rgb, untiled.rgb).identical, max_abs_diff(torch_run.rgb, untiled.rgb));
    if (!png.empty()) {
      const Image ref = read_png(png);
      const Diff d = compare(ref.rgb, untiled.rgb);
      std::printf("%s   untiled exact-GN vs reference PNG: identical %.5f%%, max|diff| %d\n", tag(), 100.0 * d.identical, d.max_abs);
      EXPECT_LE(d.max_abs, kMaxAbsDiff);
      EXPECT_GE(d.identical, kMinIdentical);
    }
    for (const auto& plan : plans) {
      const Run t = timed(*latent, false, [&](const Tensor& rgb) { model_->decode_tiled(latent->t, rgb, mode, plan, stream()); });
      const bool same = t.rgb == untiled.rgb;
      std::printf("%s   %s  peak %.3f GiB  %.3f s (%.2fx untiled)  %s\n", tag(), plan.describe().c_str(),
                  static_cast<double>(t.peak) / kGiB, t.seconds, t.seconds / untiled.seconds,
                  same ? "byte-identical" : "DIFFERS");
      EXPECT_TRUE(same) << label << " " << plan.describe() << ": identical " << compare(untiled.rgb, t.rgb).identical
                        << " max|diff| " << max_abs_diff(untiled.rgb, t.rgb);
      EXPECT_EQ(t.peak, plan.workspace_bytes) << plan.describe();
      if (!torch_run.rgb.empty()) {
        EXPECT_LE(max_abs_diff(torch_run.rgb, t.rgb), 1.0) << plan.describe();
      }
      this->RecordProperty(label + ".tile" + std::to_string(plan.tile) + ".gib", std::to_string(static_cast<double>(t.peak) / kGiB));
      this->RecordProperty(label + ".tile" + std::to_string(plan.tile) + ".x_untiled", std::to_string(t.seconds / untiled.seconds));
    }
  }

  static std::vector<mi::flux2_vae::TilePlan> plans(std::int64_t h, std::int64_t w, const DecodeMode& mode,
                                                    std::initializer_list<int> min_memory_tiles,
                                                    std::initializer_list<double> budgets_gib) {
    const DecoderConfig cfg = kFull ? DecoderConfig::full() : DecoderConfig::small();
    std::vector<mi::flux2_vae::TilePlan> v;
    for (const int t : min_memory_tiles) v.push_back(Flux2VaeDecoder::plan_tiled_min_memory(cfg, h, w, mode, t));
    for (const double gib : budgets_gib) {
      const auto p = Flux2VaeDecoder::plan_tiled(cfg, h, w, mode, static_cast<std::size_t>(gib * kGiB));
      EXPECT_TRUE(p.has_value()) << gib << " GiB";
      if (p) v.push_back(*p);
    }
    return v;
  }
};

using Flux2VaeTiledTest = TiledFixture<false>;
using Flux2VaeFullTiledTest = TiledFixture<true>;

TEST_F(Flux2VaeTiledTest, ByteIdentical1024) {
  // Tiles: 8 px (one latent row per band at H/8, far below the halo), 40 (does not divide 1024), the whole image
  // (every segment one band: the executor without halos), and two budgets.
  check_plans("g_short_1024", golden_dir() / "g_short_1024" / "final_latent.npy", DecodeMode::reference(),
              plans(64, 64, DecodeMode::reference(), {8, 40, 1024}, {0.5, 0.75}),
              golden_dir() / "g_short_1024" / "image.png");
}

TEST_F(Flux2VaeTiledTest, ByteIdentical1456) {
  check_plans("g_fridge_1456", golden_dir() / "g_fridge_1456" / "final_latent.npy", DecodeMode::reference(),
              plans(91, 91, DecodeMode::reference(), {16, 72}, {1.0, 1.5}), golden_dir() / "g_fridge_1456" / "image.png");
}

TEST_F(Flux2VaeTiledTest, ByteIdenticalNonSquareAsset) {
  const auto a = mi::npy::load(asset_latent());
  fs::path png = asset_latent();
  png.replace_filename("asset001_base.png");
  check_plans("asset001_base", asset_latent(), DecodeMode::reference(),
              plans(a.shape[2], a.shape[3], DecodeMode::reference(), {24, 168}, {1.0}), png);
}

TEST_F(Flux2VaeTiledTest, ByteIdentical2048) {
  if (!fs::is_regular_file(bench_latent_2048())) GTEST_SKIP() << "reference bench output absent: " << bench_latent_2048();
  fs::path png = bench_latent_2048();
  png.replace_filename("ac16_2048.png");
  check_plans("ac16_2048", bench_latent_2048(), DecodeMode::reference(),
              plans(128, 128, DecodeMode::reference(), {136}, {1.5, 2.0, 2.5}), png);
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
  const DecodeMode mode = DecodeMode::reference();
  const auto plan = Flux2VaeDecoder::plan_tiled_min_memory(DecoderConfig::small(), h, w, mode, 8);  // the most bands
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

// decode_within in the pipeline's mode gives the same bytes untiled and tiled; the torch-GroupNorm parity mode tiles
// with exact statistics (its own decode_tiled refuses it).
TEST_F(Flux2VaeTiledTest, DecodeWithinBudget) {
  const auto latent = load_latent(golden_dir() / "g_fridge_1456" / "final_latent.npy");
  const std::int64_t h = latent->t.dim(1), w = latent->t.dim(2);
  DevTensor rgb(mi::DType::kU8, {h * 16, w * 16, 3});
  std::vector<std::uint8_t> got(static_cast<std::size_t>(rgb.t.nbytes())), want(got.size());
  const Tensor host_want = Tensor::view(want.data(), mi::DType::kU8, mi::Device::kCPU, {h * 16, w * 16, 3});
  const Tensor host_got = Tensor::view(got.data(), mi::DType::kU8, mi::Device::kCPU, {h * 16, w * 16, 3});
  model_->decode(latent->t, host_want, DecodeMode::reference(), stream());
  // Fits: untiled (device output path).
  EXPECT_FALSE(model_->decode_within(latent->t, rgb.t, DecodeMode::reference(),
                                     Flux2VaeDecoder::workspace_bytes(DecoderConfig::small(), h, w, DecodeMode::reference()), stream()));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  rgb.buf.download(got.data(), got.size());
  EXPECT_EQ(got, want);
  // 1 GiB: tiled, byte-identical to the untiled decode (host output path).
  const auto plan = model_->decode_within(latent->t, host_got, DecodeMode::reference(), std::size_t{1} << 30, stream());
  ASSERT_TRUE(plan.has_value());
  EXPECT_LE(plan->workspace_bytes, std::size_t{1} << 30);
  EXPECT_EQ(got, want);
  // torch_reference(): untiled when it fits, tiled with exact statistics (= reference()) when it does not.
  std::fill(got.begin(), got.end(), std::uint8_t{0});
  EXPECT_TRUE(model_->decode_within(latent->t, host_got, DecodeMode::torch_reference(), std::size_t{1} << 30, stream()));
  EXPECT_EQ(got, want);
  EXPECT_FALSE(model_->decode_within(latent->t, host_got, DecodeMode::torch_reference(),
                                     Flux2VaeDecoder::workspace_bytes(DecoderConfig::small(), h, w, DecodeMode::torch_reference()), stream()));
  model_->decode(latent->t, host_want, DecodeMode::torch_reference(), stream());
  EXPECT_EQ(got, want);
  EXPECT_THROW(model_->decode_within(latent->t, host_got, DecodeMode::reference(), std::size_t{64} << 20, stream()),
               std::runtime_error);
  EXPECT_THROW(model_->decode_tiled(latent->t, host_got, DecodeMode::torch_reference(), *plan, stream()),
               std::invalid_argument);
}

// ---- Single-file VAE loader ---------------------------------------------------------------------------------

TEST(Flux2VaeFile, KeyMapping) {
  using mi::flux2_vae::vae_file_tensor_name;
  EXPECT_EQ(vae_file_tensor_name("decoder.mid.block_2.norm1.weight"), "decoder.mid_block.resnets.1.norm1.weight");
  EXPECT_EQ(vae_file_tensor_name("decoder.mid.attn_1.proj_out.bias"), "decoder.mid_block.attentions.0.to_out.0.bias");
  EXPECT_EQ(vae_file_tensor_name("decoder.mid.attn_1.norm.weight"), "decoder.mid_block.attentions.0.group_norm.weight");
  EXPECT_EQ(vae_file_tensor_name("decoder.up.0.block.0.nin_shortcut.weight"), "decoder.up_blocks.3.resnets.0.conv_shortcut.weight");
  EXPECT_EQ(vae_file_tensor_name("decoder.up.3.upsample.conv.bias"), "decoder.up_blocks.0.upsamplers.0.conv.bias");
  EXPECT_EQ(vae_file_tensor_name("decoder.norm_out.bias"), "decoder.conv_norm_out.bias");
  EXPECT_EQ(vae_file_tensor_name("decoder.post_quant_conv.weight"), "post_quant_conv.weight");
  EXPECT_EQ(vae_file_tensor_name("decoder.up_blocks.2.resnets.1.conv1.weight"), "decoder.up_blocks.2.resnets.1.conv1.weight");
  EXPECT_EQ(vae_file_tensor_name("bn.running_var"), "bn.running_var");
  EXPECT_EQ(vae_file_tensor_name("bn.num_batches_tracked"), "");
  EXPECT_EQ(vae_file_tensor_name("encoder.conv_in.weight"), "");
  EXPECT_EQ(vae_file_tensor_name("quant_conv.weight"), "");
}

// Both community files index to exactly the tensor names and shapes the bundle's decoder group has (the full file with
// every width x 4/3); widths come from the shapes.
TEST(Flux2VaeFile, IndexesBothLayoutsLikeTheBundle) {
  if (!fs::is_regular_file(bundle_dir() / "moss.mossidx")) GTEST_SKIP() << "weights not present";
  const auto bundle = mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir());
  const auto* bg = bundle.find_group(mi::weights::Component::kVae, "decoder");
  ASSERT_NE(bg, nullptr);
  EXPECT_EQ(mi::flux2_vae::decoder_config(bundle), DecoderConfig::small());
  const auto scale = [](std::int64_t d) { return d == 384 || d == 192 || d == 96 ? d / 3 * 4 : d; };
  for (const bool full : {false, true}) {
    const fs::path f = full ? full_vae_file() : small_vae_file();
    if (!fs::is_regular_file(f)) {
      std::printf("[flux2_vae] %s absent, skipped\n", f.string().c_str());
      continue;
    }
    const auto idx = mi::flux2_vae::index_vae_file(f);
    EXPECT_EQ(mi::flux2_vae::decoder_config(idx), full ? DecoderConfig::full() : DecoderConfig::small());
    const auto* g = idx.find_group(mi::weights::Component::kVae, "decoder");
    ASSERT_NE(g, nullptr);
    std::map<std::string, std::vector<std::int64_t>> want, got;
    for (const auto& t : bg->tensors) {
      if (t.name == "bn.num_batches_tracked") continue;  // the bundle index keeps it; the decoder never reads it
      std::vector<std::int64_t> shape = t.shape;
      if (full)
        for (auto& d : shape) d = scale(d);
      want[t.name] = shape;
    }
    for (const auto& t : g->tensors) got[t.name] = t.shape;
    EXPECT_EQ(got, want) << f;
    std::printf("[flux2_vae] %s: %zu decoder tensors, %.1f MB (+ %llu excluded tensors)\n", f.filename().string().c_str(),
                g->tensors.size(), static_cast<double>(g->total_bytes) / 1e6,
                static_cast<unsigned long long>(idx.excluded.empty() ? 0 : idx.excluded[0].tensors));
  }
}

// The official pipeline de-normalizes MOSS latents with the bundle's bn statistics, which are the FLUX.2 VAE's fp32
// originals rounded to bf16 (stored as F32); the community single files carry the unrounded originals. Checks that
// relation for both files and reports the differences.
TEST(Flux2VaeFile, LatentStatsAreTheBundlesAfterBf16Rounding) {
  const fs::path bundle_vae = bundle_dir() / "vae" / "diffusion_pytorch_model.safetensors";
  if (!fs::is_regular_file(bundle_vae)) GTEST_SKIP() << "weights not present";
  const auto read = [](const fs::path& f, const std::string& name) {
    const mi::weights::SafetensorsFile st(f);
    const auto* t = st.find(name);
    EXPECT_NE(t, nullptr) << f << " " << name;
    std::vector<float> v(t != nullptr ? static_cast<std::size_t>(t->numel()) : 0);
    if (t == nullptr || t->dtype != mi::weights::DType::kF32) return v;
    std::ifstream in(f, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(t->offset));
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    return v;
  };
  const auto bf16 = [](float x) {  // round to nearest even (finite inputs)
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
  };
  int files = 0;
  for (const fs::path& f : {full_vae_file(), small_vae_file()}) {
    if (!fs::is_regular_file(f)) continue;
    ++files;
    for (const char* name : {"bn.running_mean", "bn.running_var"}) {
      const auto file_v = read(f, name), bundle_v = read(bundle_vae, name);
      ASSERT_EQ(file_v.size(), bundle_v.size());
      std::size_t differ = 0, rounded_equal = 0;
      double max_abs = 0, max_rel = 0;
      for (std::size_t i = 0; i < file_v.size(); ++i) {
        const double d = std::abs(static_cast<double>(file_v[i]) - bundle_v[i]);
        differ += d != 0;
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(1e-30, std::abs(static_cast<double>(file_v[i]))));
        rounded_equal += bf16(file_v[i]) == bundle_v[i];
      }
      std::printf("[flux2_vae] %s %s vs bundle: %zu/%zu values differ, max|diff| %.3g (max rel %.3g); bf16(file) == bundle "
                  "for %zu/%zu\n", f.filename().string().c_str(), name, differ, file_v.size(), max_abs, max_rel, rounded_equal,
                  file_v.size());
      EXPECT_EQ(rounded_equal, file_v.size()) << f << " " << name;
    }
  }
  if (files == 0) GTEST_SKIP() << "community VAE files absent";
}

// The decoder rejects a group with a tensor it does not consume, and a single-file group spliced into another index
// (its file ids then name the bundle's files).
TEST(Flux2VaeFile, DecoderRejectsUnexpectedAndSplicedTensors) {
  if (!fs::is_regular_file(full_vae_file()) || !fs::is_regular_file(bundle_dir() / "moss.mossidx"))
    GTEST_SKIP() << "weights not present";
  const auto bundle = mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir());
  mi::runtime::StreamSet streams;
  auto extra = mi::flux2_vae::index_vae_file(full_vae_file());
  auto& g = extra.groups.at(0);
  mi::weights::IndexedTensor t = g.tensors.front();
  t.name = "decoder.extra." + t.name;
  g.tensors.push_back(t);
  {
    mi::runtime::DeviceArena arena(Flux2VaeDecoder::persistent_bytes(extra, false, &bundle) + (std::size_t{64} << 20));
    EXPECT_THROW(
        {
          try {
            Flux2VaeDecoder d(extra, arena, streams, false, false, &bundle);
          } catch (const std::runtime_error& e) {
            std::printf("[flux2_vae] extra tensor: %s\n", e.what());
            throw;
          }
        },
        std::runtime_error);
  }
  auto spliced = bundle;
  for (auto& bg : spliced.groups)
    if (bg.component == mi::weights::Component::kVae && bg.name == "decoder") bg = mi::flux2_vae::index_vae_file(full_vae_file()).groups.at(0);
  {
    mi::runtime::DeviceArena arena(Flux2VaeDecoder::persistent_bytes(spliced, false) + (std::size_t{64} << 20));
    EXPECT_THROW(
        {
          try {
            Flux2VaeDecoder d(spliced, arena, streams);
          } catch (const std::runtime_error& e) {
            std::printf("[flux2_vae] spliced group: %s\n", e.what());
            throw;
          }
        },
        std::runtime_error);
  }
}

// ---- Mid-block attention, head dims 384 and 512 ----------------------------------------------------------------

// vae::attention_fp32 against an fp64 host reference for both head dims, on q/k/v packed in one [B, L, 3D] buffer (the
// mid attention's layout): L below one 64-row tile, L a multiple of 64, L not a multiple, and B = 2. Hash-RNG data
// with a softmax that is neither flat nor one-hot. (The 384 path's bytes are pinned by the stored digests in
// test_flux2_vae_regress.cpp, from the kernel before it was templated.)
TEST(Flux2VaeAttention, BothHeadDimsMatchFp64) {
  EXPECT_TRUE(mi::vae::attention_head_dim_supported(384));
  EXPECT_TRUE(mi::vae::attention_head_dim_supported(512));
  EXPECT_FALSE(mi::vae::attention_head_dim_supported(256));
  struct Shape {
    std::int64_t b, len;
  };
  for (const int d : {384, 512}) {
    for (const Shape sh : {Shape{1, 40}, Shape{1, 1024}, Shape{1, 1000}, Shape{2, 200}}) {
      const std::int64_t rows = sh.b * sh.len;
      std::vector<float> qkv = synth(static_cast<std::uint64_t>(d + sh.len), static_cast<std::size_t>(rows * 3 * d), 0.0f);
      for (float& x : qkv) x *= 0.25f;
      DevTensor buf(mi::DType::kF32, {rows, 3 * d}), o(mi::DType::kF32, {sh.b, sh.len, d});
      buf.buf.upload(qkv.data(), qkv.size() * 4);
      const auto view = [&](int p) {
        Tensor t = Tensor::view(static_cast<float*>(buf.t.data) + p * d, mi::DType::kF32, mi::Device::kCUDA, {sh.b, sh.len, d});
        t.stride[1] = 3 * d;
        t.stride[0] = sh.len * 3 * d;
        return t;
      };
      const float scale = 1.0f / std::sqrt(static_cast<float>(d));
      const auto at = [&](std::int64_t row, int part, int c) {
        return static_cast<double>(qkv[static_cast<std::size_t>(row * 3 * d + part * d + c)]);
      };
      std::vector<double> ref(static_cast<std::size_t>(rows * d), 0.0), sc(static_cast<std::size_t>(sh.len));
      for (std::int64_t bi = 0; bi < sh.b; ++bi)
        for (std::int64_t i = 0; i < sh.len; ++i) {
          const std::int64_t qi = bi * sh.len + i;
          double mx = -1e300;
          for (std::int64_t j = 0; j < sh.len; ++j) {
            double acc = 0;
            for (int c = 0; c < d; ++c) acc += at(qi, 0, c) * at(bi * sh.len + j, 1, c);
            sc[static_cast<std::size_t>(j)] = acc * scale;
            mx = std::max(mx, sc[static_cast<std::size_t>(j)]);
          }
          double sum = 0;
          for (auto& v : sc) sum += (v = std::exp(v - mx));
          for (std::int64_t j = 0; j < sh.len; ++j) {
            const double p = sc[static_cast<std::size_t>(j)] / sum;
            for (int c = 0; c < d; ++c) ref[static_cast<std::size_t>(qi * d + c)] += p * at(bi * sh.len + j, 2, c);
          }
        }
      for (const auto prec : {mi::vae::AttnPrecision::kTf32x3, mi::vae::AttnPrecision::kTf32}) {
        mi::vae::attention_fp32(view(0), view(1), view(2), o.t, scale, prec, nullptr);
        std::vector<float> got(static_cast<std::size_t>(rows * d));
        MI_CUDA_CHECK(cudaMemcpy(got.data(), o.t.data, got.size() * 4, cudaMemcpyDeviceToHost));
        double dot = 0, n1 = 0, n2 = 0, max_err = 0, max_ref = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
          dot += got[i] * ref[i];
          n1 += static_cast<double>(got[i]) * got[i];
          n2 += ref[i] * ref[i];
          max_err = std::max(max_err, std::abs(got[i] - ref[i]));
          max_ref = std::max(max_ref, std::abs(ref[i]));
        }
        const bool x3 = prec == mi::vae::AttnPrecision::kTf32x3;
        const double cos = dot / std::sqrt(n1 * n2);
        std::printf("[flux2_vae] attention D=%d B=%lld L=%-4lld %-6s vs fp64: cosine %.9f, max|err| %.3g of max|out|\n", d,
                    static_cast<long long>(sh.b), static_cast<long long>(sh.len), x3 ? "tf32x3" : "tf32", cos, max_err / max_ref);
        EXPECT_GE(cos, 0.99999);
        EXPECT_LE(max_err / max_ref, x3 ? 1e-5 : 1e-3);
      }
    }
  }
}

// ---- Full decoder ---------------------------------------------------------------------------------------------

// Same 8-bit bounds as the small decoder against its reference PNGs (see kMinIdentical).
TEST_F(Flux2VaeFullTest, TrackedCropMatchesDiffusers) {
  for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference()}) {
    const Diff d = check_against_png("full_crop", data_dir() / "full_crop.latent.npy", data_dir() / "full_crop.png", mode);
    EXPECT_GE(d.identical, kMinIdentical) << mode_name(mode);
    EXPECT_LE(d.max_abs, kMaxAbsDiff) << mode_name(mode);
    EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff) << mode_name(mode);
  }
}

TEST_F(Flux2VaeFullTest, ReferenceImagesMatchDiffusers) {
  if (!fs::is_directory(full_ref_dir())) GTEST_SKIP() << "full-decoder reference decodes absent: " << full_ref_dir();
  std::vector<std::pair<std::string, fs::path>> cases;
  for (const auto& id : golden_cases()) cases.emplace_back(id, golden_dir() / id / "final_latent.npy");
  cases.emplace_back("asset001_base", asset_latent());
  cases.emplace_back("ac16_2048", bench_latent_2048());
  int n = 0;
  for (const auto& [id, latent] : cases) {
    const fs::path png = full_ref_dir() / (id + ".png");
    if (!fs::is_regular_file(png) || !fs::is_regular_file(latent)) continue;
    ++n;
    for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference()}) {
      const Diff d = check_against_png(id, latent, png, mode);
      EXPECT_GE(d.identical, kMinIdentical) << id << " " << mode_name(mode);
      EXPECT_LE(d.max_abs, kMaxAbsDiff) << id << " " << mode_name(mode);
      EXPECT_LE(d.mean_abs, kMaxMeanAbsDiff) << id << " " << mode_name(mode);
    }
  }
  EXPECT_GE(n, 8);
}

TEST_F(Flux2VaeFullTest, FastModesStayCloseToReferenceImages) {
  if (!fs::is_directory(full_ref_dir())) GTEST_SKIP() << "full-decoder reference decodes absent: " << full_ref_dir();
  for (const std::string id : {"g_fridge_1024", "g_layout_1456"}) {
    for (const DecodeMode& mode : {DecodeMode::fast_attention(), DecodeMode::fastest()}) {
      const Diff d = check_against_png(id, golden_dir() / id / "final_latent.npy", full_ref_dir() / (id + ".png"), mode);
      EXPECT_LE(d.mean_abs, mode.bf16_convs ? 0.5 : 0.05) << id;
    }
  }
}

// Memory of both decoders at the sizes the launcher uses (the planner's queries), and the stack high-water of real
// full-decoder decodes against the query.
TEST_F(Flux2VaeFullTest, WorkspaceQueryCoversPeakAndIsReported) {
  struct Size {
    const char* name;
    std::int64_t h, w;
  };
  const Size sizes[] = {{"1024x1024", 64, 64}, {"1248x1664", 104, 78}, {"2048x2048", 128, 128}};
  const std::size_t pers = Flux2VaeDecoder::persistent_bytes(*index_, false, bundle_.get()),
                    pers_bf16 = Flux2VaeDecoder::persistent_bytes(*index_, true, bundle_.get());
  std::printf("%s persistent: %.1f MB fp32 weights, %.1f MB with bf16 copies\n", tag(), static_cast<double>(pers) / 1e6,
              static_cast<double>(pers_bf16) / 1e6);
  for (const DecoderConfig cfg : {DecoderConfig::small(), DecoderConfig::full()}) {
    for (const Size& s : sizes) {
      for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::fastest()}) {
        std::printf("[flux2_vae %s] %-9s %-40s untiled %6.3f GiB | tiled min (tile 8/64/256) %.3f / %.3f / %.3f GiB\n", cfg.name(),
                    s.name, mode_name(mode), static_cast<double>(Flux2VaeDecoder::workspace_bytes(cfg, s.h, s.w, mode)) / kGiB,
                    static_cast<double>(Flux2VaeDecoder::tiled_workspace_bytes(cfg, s.h, s.w, 8, mode)) / kGiB,
                    static_cast<double>(Flux2VaeDecoder::tiled_workspace_bytes(cfg, s.h, s.w, 64, mode)) / kGiB,
                    static_cast<double>(Flux2VaeDecoder::tiled_workspace_bytes(cfg, s.h, s.w, 256, mode)) / kGiB);
        for (const double gib : {1.0, 1.5, 2.0, 3.0}) {
          const auto p = Flux2VaeDecoder::plan_tiled(cfg, s.h, s.w, mode, static_cast<std::size_t>(gib * kGiB));
          if (p) {
            EXPECT_LE(p->workspace_bytes, static_cast<std::size_t>(gib * kGiB));
          }
          std::printf("[flux2_vae %s]   budget %.1f GiB: %s\n", cfg.name(), gib, p ? p->describe().c_str() : "no schedule");
        }
      }
    }
  }
  for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::torch_reference(), DecodeMode::fastest()}) {
    const auto latent = load_latent(asset_latent());
    DevTensor rgb(mi::DType::kU8, {latent->t.dim(1) * 16, latent->t.dim(2) * 16, 3});
    arena_->reset_high_water();
    model_->decode(latent->t, rgb.t, mode, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const std::size_t peak = arena_->high_water() - arena_->persistent_used();
    EXPECT_EQ(peak, Flux2VaeDecoder::workspace_bytes(model_->config(), latent->t.dim(1), latent->t.dim(2), mode));
    EXPECT_EQ(arena_->stack_used(), 0u);
  }
  // A plan for the small decoder is refused (and the other way round, Flux2VaeTest.RejectsFullDecoderPlan).
  const auto latent = load_latent(golden_dir() / "g_short_1024" / "final_latent.npy");
  DevTensor rgb(mi::DType::kU8, {1024, 1024, 3});
  EXPECT_THROW(model_->decode_tiled(latent->t, rgb.t, DecodeMode::reference(),
                                    Flux2VaeDecoder::plan_tiled_min_memory(DecoderConfig::small(), 64, 64, DecodeMode::reference(), 64),
                                    stream()),
               std::invalid_argument);
}

TEST_F(Flux2VaeFullTest, DecodeTiming) {
  const std::pair<const char*, fs::path> cases[] = {{"1024x1024", golden_dir() / "g_fridge_1024" / "final_latent.npy"},
                                                   {"1456x1456", golden_dir() / "g_fridge_1456" / "final_latent.npy"},
                                                   {"2048x2048", bench_latent_2048()}};
  cudaEvent_t a = nullptr, b = nullptr;
  MI_CUDA_CHECK(cudaEventCreate(&a));
  MI_CUDA_CHECK(cudaEventCreate(&b));
  for (const auto& [name, path] : cases) {
    if (!fs::is_regular_file(path)) continue;
    const auto latent = load_latent(path);
    DevTensor rgb(mi::DType::kU8, {latent->t.dim(1) * 16, latent->t.dim(2) * 16, 3});
    for (const DecodeMode& mode : {DecodeMode::reference(), DecodeMode::fast_attention(), DecodeMode::fastest()}) {
      model_->decode(latent->t, rgb.t, mode, stream());
      constexpr int kIters = 2;
      MI_CUDA_CHECK(cudaEventRecord(a, stream()));
      for (int i = 0; i < kIters; ++i) model_->decode(latent->t, rgb.t, mode, stream());
      MI_CUDA_CHECK(cudaEventRecord(b, stream()));
      MI_CUDA_CHECK(cudaEventSynchronize(b));
      float ms = 0;
      MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      std::printf("%s decode %-9s %-40s %7.3f s (dev)\n", tag(), name, mode_name(mode), ms / 1e3 / kIters);
    }
  }
  MI_CUDA_CHECK(cudaEventDestroy(a));
  MI_CUDA_CHECK(cudaEventDestroy(b));
}

fs::path full_ref(const std::string& id) {
  const fs::path p = full_ref_dir() / (id + ".png");
  return fs::is_regular_file(p) ? p : fs::path{};
}

TEST_F(Flux2VaeFullTiledTest, ByteIdentical1024) {
  check_plans("g_short_1024", golden_dir() / "g_short_1024" / "final_latent.npy", DecodeMode::reference(),
              plans(64, 64, DecodeMode::reference(), {8, 40, 1024}, {0.75, 1.0}), full_ref("g_short_1024"));
}

TEST_F(Flux2VaeFullTiledTest, ByteIdentical1456) {
  check_plans("g_fridge_1456", golden_dir() / "g_fridge_1456" / "final_latent.npy", DecodeMode::reference(),
              plans(91, 91, DecodeMode::reference(), {16, 72}, {1.5, 2.0}), full_ref("g_fridge_1456"));
}

TEST_F(Flux2VaeFullTiledTest, ByteIdenticalNonSquareAsset) {
  const auto a = mi::npy::load(asset_latent());
  check_plans("asset001_base", asset_latent(), DecodeMode::reference(),
              plans(a.shape[2], a.shape[3], DecodeMode::reference(), {24, 168}, {1.5}), full_ref("asset001_base"));
}

TEST_F(Flux2VaeFullTiledTest, ByteIdentical2048) {
  if (!fs::is_regular_file(bench_latent_2048())) GTEST_SKIP() << "reference bench output absent: " << bench_latent_2048();
  check_plans("ac16_2048", bench_latent_2048(), DecodeMode::reference(),
              plans(128, 128, DecodeMode::reference(), {136}, {2.0, 3.0}), full_ref("ac16_2048"));
}

TEST_F(Flux2VaeFullTiledTest, ByteIdenticalBf16Mode) {
  check_plans("g_layout_1024", golden_dir() / "g_layout_1024" / "final_latent.npy", DecodeMode::fastest(),
              plans(64, 64, DecodeMode::fastest(), {16, 104}, {0.5}));
}

TEST_F(Flux2VaeFullTiledTest, DecodeWithinBudget) {
  const auto latent = load_latent(golden_dir() / "g_fridge_1456" / "final_latent.npy");
  const std::int64_t h = latent->t.dim(1), w = latent->t.dim(2);
  std::vector<std::uint8_t> got(static_cast<std::size_t>(h * w * 256 * 3)), want(got.size());
  const Tensor host_want = Tensor::view(want.data(), mi::DType::kU8, mi::Device::kCPU, {h * 16, w * 16, 3});
  const Tensor host_got = Tensor::view(got.data(), mi::DType::kU8, mi::Device::kCPU, {h * 16, w * 16, 3});
  model_->decode(latent->t, host_want, DecodeMode::reference(), stream());
  const std::size_t untiled = Flux2VaeDecoder::workspace_bytes(DecoderConfig::full(), h, w, DecodeMode::reference());
  EXPECT_FALSE(model_->decode_within(latent->t, host_got, DecodeMode::reference(), untiled, stream()));
  EXPECT_EQ(got, want);
  std::fill(got.begin(), got.end(), std::uint8_t{0});
  const auto plan = model_->decode_within(latent->t, host_got, DecodeMode::reference(), std::size_t{3} << 29, stream());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->config, DecoderConfig::full());
  EXPECT_LE(plan->workspace_bytes, std::size_t{3} << 29);
  EXPECT_EQ(got, want);
}

}  // namespace
