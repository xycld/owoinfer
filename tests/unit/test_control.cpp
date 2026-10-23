// Control adapters (models/control, roadmap P8): the file reader, the LoRA stack's consumed tensors, the step windows,
// the per-pass control latents, the widened DiT video embedding against torch (tests/data/control), the request
// validation / report / PNG metadata, and end-to-end generations with the real adapter (skipped without the weights).
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/gemm.h"
#include "models/control/control.h"
#include "models/lora/lora.h"
#include "models/moss_dit/dit_kernels.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "test_lora_util.h"
#include "test_util.h"
#include "util/npy.h"
#include "weights/quant/quant_codec.h"

namespace mi::control {
namespace {

namespace fs = std::filesystem;
namespace pl = mi::pipeline;
using mi::testing::lora::randn;
using mi::testing::lora::StTensor;
using mi::testing::lora::write_st;
using WDType = weights::DType;

const fs::path kControlFile = "/workspace/Models/community/comfy-org/moss_image_color_depth_control.safetensors";
const fs::path kPeftLora = "/workspace/Models/community/axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors";
fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/control"; }

// A synthetic control file: an embedder [3072, in] (+ bias) and one PEFT adapter on layers.0.proj.q.
void write_control(const fs::path& p, std::int64_t in, WDType dt, const std::string& meta, bool bias = true,
                   std::int64_t bias_n = kDim) {
  std::vector<StTensor> ts;
  ts.push_back({"layers.0.proj.q.lora_A.weight", WDType::kBF16, {4, kDim}, randn(4 * kDim, 1, 0.01f)});
  ts.push_back({"layers.0.proj.q.lora_B.weight", WDType::kBF16, {kDim, 4}, randn(4 * kDim, 2, 0.01f)});
  ts.push_back({kEmbedderWeight, dt, {kDim, in}, randn(static_cast<std::size_t>(kDim * in), 3, 0.05f)});
  if (bias) ts.push_back({kEmbedderBias, dt, {bias_n}, randn(static_cast<std::size_t>(bias_n), 4, 0.05f)});
  write_st(p, ts, meta);
}

TEST(ControlFile, ReadsKindsShapesAndConvertsToBf16) {
  testing::TempDir dir("control_file");
  const fs::path a = dir.path() / "a.safetensors";
  write_control(a, 384, WDType::kF32, R"({"control_channels":"color, depth","base_model":"moss-base"})");
  const ControlFile f = open_control_file(a);
  EXPECT_EQ(f.kinds, (std::vector<std::string>{"color", "depth"}));
  EXPECT_EQ(f.in_features, 384);
  EXPECT_EQ(f.groups(), 2);
  EXPECT_EQ(f.kind_index("depth"), 1);
  EXPECT_EQ(f.kind_index("canny"), -1);
  EXPECT_EQ(f.base_model, "moss-base");
  EXPECT_TRUE(is_control_file(a));
  // F32 values that are bf16-representable come back bit for bit (randn rounds to bf16).
  const EmbedderWeights e = read_embedder(f);
  const std::vector<float> ref = randn(static_cast<std::size_t>(kDim * 384), 3, 0.05f);
  ASSERT_EQ(e.w.size(), ref.size());
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) mismatches += e.w[i] != quant::float_to_bf16(ref[i]);
  EXPECT_EQ(mismatches, 0u);
  EXPECT_EQ(e.bias.size(), static_cast<std::size_t>(kDim));

  // No metadata: generic kinds; three control latents.
  const fs::path b = dir.path() / "b.safetensors";
  write_control(b, 512, WDType::kBF16, "");
  EXPECT_EQ(open_control_file(b).kinds, (std::vector<std::string>{"control0", "control1", "control2"}));
}

TEST(ControlFile, RejectsFilesWithoutAValidEmbedder) {
  testing::TempDir dir("control_bad");
  const auto path = [&](const char* n) { return dir.path() / n; };
  write_control(path("plain.safetensors"), 128, WDType::kBF16, "");  // not widened
  write_control(path("odd.safetensors"), 200, WDType::kBF16, "");    // not a multiple of 128
  write_control(path("nobias.safetensors"), 384, WDType::kBF16, "", false);
  write_control(path("badbias.safetensors"), 384, WDType::kBF16, "", true, 128);
  write_control(path("kinds.safetensors"), 384, WDType::kBF16, R"({"control_channels":"color"})");
  write_control(path("dup.safetensors"), 384, WDType::kBF16, R"({"control_channels":"color,color"})");
  write_control(path("i8.safetensors"), 384, WDType::kI8, "");
  for (const char* n : {"plain.safetensors", "odd.safetensors", "nobias.safetensors", "badbias.safetensors",
                        "kinds.safetensors", "dup.safetensors", "i8.safetensors"}) {
    EXPECT_THROW(open_control_file(path(n)), ControlError) << n;
  }
  EXPECT_FALSE(is_control_file(path("plain.safetensors")));
  EXPECT_FALSE(is_control_file(path("missing.safetensors")));
  EXPECT_THROW(open_control_file(path("missing.safetensors")), std::runtime_error);
}

TEST(ControlFile, PublishedAdapter) {
  if (!fs::is_regular_file(kControlFile)) GTEST_SKIP() << "control adapter not present";
  const ControlFile f = open_control_file(kControlFile);
  EXPECT_EQ(f.kinds, (std::vector<std::string>{"color", "depth"}));
  EXPECT_EQ(f.in_features, 384);
  EXPECT_EQ(f.dtype, WDType::kBF16);
  EXPECT_EQ(f.base_model, "moss-base");
  if (fs::is_regular_file(kPeftLora)) {
    EXPECT_FALSE(is_control_file(kPeftLora));
  }

  // The LoRA part: 42 blocks x 6 GEMMs on the side branch, the text refiner's 2 x 5 linears merged; the embedder
  // tensors consumed (noted, not unknown). Without `consumed` they would be reported unknown (strict: an error).
  const std::vector<lora::LoraSpec> with = {{kControlFile, 1.0f, f.consumed()}};
  const lora::LoraStack stack(with, lora::moss_dit_targets());
  const lora::FileReport& rep = stack.reports().at(0);
  EXPECT_EQ(rep.adapters, 262);
  EXPECT_EQ(rep.side_branch, 252);
  EXPECT_EQ(rep.merged, 10);
  EXPECT_TRUE(rep.unknown.empty());
  ASSERT_EQ(rep.notes.size(), 1u);
  EXPECT_NE(rep.notes[0].find("2 tensors read outside the LoRA stack"), std::string::npos);
  for (const lora::Adapter& a : stack.adapters()) EXPECT_DOUBLE_EQ(a.alpha_scale, 1.0) << a.module;  // alpha 64 / rank 64
  const std::vector<lora::LoraSpec> without = {{kControlFile, 1.0f}};
  EXPECT_EQ(lora::LoraStack(without, lora::moss_dit_targets()).reports().at(0).unknown.size(), 2u);
  lora::LoadOptions strict;
  strict.strict = true;
  EXPECT_THROW(lora::LoraStack(without, lora::moss_dit_targets(), strict), lora::LoraError);
  EXPECT_NO_THROW(lora::LoraStack(with, lora::moss_dit_targets(), strict));
}

TEST(ControlWindow, StepFractions) {
  EXPECT_TRUE(window_active(0, 1, 0, 8));
  EXPECT_TRUE(window_active(0, 1, 7, 8));
  EXPECT_TRUE(window_active(0, 0.5, 3, 8));
  EXPECT_FALSE(window_active(0, 0.5, 4, 8));
  EXPECT_FALSE(window_active(0.25, 0.5, 1, 8));
  EXPECT_TRUE(window_active(0.25, 0.5, 2, 8));
  EXPECT_FALSE(window_active(0.5, 0.5, 4, 8));  // empty window
  EXPECT_FALSE(window_active(0, 1, 0, 0));
}

std::vector<float> download(const Tensor& t) {
  std::vector<float> h(static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost));
  return h;
}

TEST(ControlPassInputs, WritesOnlyChangedGroups) {
  const std::int64_t h = 2, w = 3, plane = kGroupChannels * h * w;
  DeviceBuffer buf(static_cast<std::size_t>(2 * plane) * sizeof(float));
  const Tensor extra = Tensor::view(buf.get(), DType::kF32, Device::kCUDA, {2 * kGroupChannels, h, w});
  MI_CUDA_CHECK(cudaMemset(buf.get(), 0x7F, buf.size()));  // garbage: the first apply writes every group
  PassInputs in(2, h, w);
  std::vector<float> lat(static_cast<std::size_t>(plane));
  for (std::size_t i = 0; i < lat.size(); ++i) lat[i] = static_cast<float>(i) * 0.5f - 3.0f;
  in.set(0, lat, 0.0, 0.5);  // group 1: no image
  EXPECT_THROW(in.set(2, lat, 0, 1), std::invalid_argument);
  EXPECT_THROW(in.set(0, std::vector<float>(3), 0, 1), std::invalid_argument);
  EXPECT_EQ(in.apply(extra, 0, 4, nullptr), 2);
  std::vector<float> got = download(extra);
  EXPECT_EQ(std::memcmp(got.data(), lat.data(), lat.size() * sizeof(float)), 0);
  for (std::int64_t i = plane; i < 2 * plane; ++i) ASSERT_EQ(got[static_cast<std::size_t>(i)], 0.0f);
  EXPECT_EQ(in.apply(extra, 1, 4, nullptr), 0);  // nothing changed
  EXPECT_EQ(in.apply(extra, 2, 4, nullptr), 1);  // the window closed: group 0 zeroed
  got = download(extra);
  for (float v : got) ASSERT_EQ(v, 0.0f);
  EXPECT_EQ(in.apply(extra, 3, 4, nullptr), 0);

  // Channel order: the published file's kinds are [color, depth], so a depth image is group 1, channels [128, 256).
  if (fs::is_regular_file(kControlFile)) {
    const int depth = open_control_file(kControlFile).kind_index("depth");
    ASSERT_EQ(depth, 1);
    PassInputs d(2, h, w);
    d.set(depth, lat, 0, 1);
    d.apply(extra, 0, 4, nullptr);
    got = download(extra);
    for (std::int64_t i = 0; i < plane; ++i) ASSERT_EQ(got[static_cast<std::size_t>(i)], 0.0f);
    EXPECT_EQ(std::memcmp(got.data() + plane, lat.data(), lat.size() * sizeof(float)), 0);
  }
}

// The widened embedding (latent_to_tokens into the two column ranges, then the bf16 GEMM with the replacement weight)
// against torch's F.linear on the concatenation (tools/reference/gen_control_vectors.py).
TEST(ControlEmbedding, MatchesTorchReference) {
  if (!fs::is_regular_file(kControlFile) || !fs::exists(data_dir() / "out.npy")) {
    GTEST_SKIP() << "control adapter or tests/data/control vectors not present";
  }
  const EmbedderWeights ew = read_embedder(open_control_file(kControlFile));
  const DeviceEmbedder emb(ew);
  EXPECT_EQ(emb.device_bytes(), DeviceEmbedder::device_bytes(kDim, 384));
  const std::vector<float> lat = npy::load(data_dir() / "latents.npy").to_vector<float>();
  const std::vector<float> ctl = npy::load(data_dir() / "control.npy").to_vector<float>();
  const std::vector<std::uint16_t> ref_tokens = npy::load(data_dir() / "tokens.npy").to_vector<std::uint16_t>();
  const std::vector<std::uint16_t> ref = npy::load(data_dir() / "out.npy").to_vector<std::uint16_t>();
  const std::int64_t h = 8, w = 8, hw = h * w;
  DeviceBuffer dl(lat.size() * 4), dc(ctl.size() * 4), dt(static_cast<std::size_t>(hw * 384) * 2), dout(static_cast<std::size_t>(hw * kDim) * 2);
  dl.upload(lat.data(), lat.size() * 4);
  dc.upload(ctl.data(), ctl.size() * 4);
  const Tensor tl = Tensor::view(dl.get(), DType::kF32, Device::kCUDA, {128, h, w});
  const Tensor tc = Tensor::view(dc.get(), DType::kF32, Device::kCUDA, {256, h, w});
  const Tensor out = Tensor::view(dout.get(), DType::kBF16, Device::kCUDA, {hw, kDim});
  kernels::Gemm gemm;
  DeviceBuffer ws(kernels::Gemm::kWorkspaceBytes);
  moss_dit::embed_video(gemm, ws.get(), ws.size(), tl, &tc, dt.get(), emb.weight(), emb.bias(), out, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint16_t> tokens(static_cast<std::size_t>(hw * 384)), got(static_cast<std::size_t>(hw * kDim));
  dt.download(tokens.data(), tokens.size() * 2);
  dout.download(got.data(), got.size() * 2);
  EXPECT_EQ(tokens, ref_tokens);  // the concatenation, each value rounded once
  // Accuracy vs torch: bit-exact elements, max |diff| in bf16 ULPs of the reference value, cosine.
  std::size_t exact = 0;
  int max_ulp = 0;
  double dot = 0, n1 = 0, n2 = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    exact += got[i] == ref[i];
    const float a = quant::bf16_to_float(got[i]), b = quant::bf16_to_float(ref[i]);
    const auto ord = [](std::uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7FFF) : static_cast<int>(v); };
    max_ulp = std::max(max_ulp, std::abs(ord(got[i]) - ord(ref[i])));
    dot += static_cast<double>(a) * b;
    n1 += static_cast<double>(a) * a;
    n2 += static_cast<double>(b) * b;
  }
  const double cosine = dot / std::sqrt(n1 * n2);
  std::printf("[control] widened embedding vs torch: %zu / %zu bit-exact, max %d ULP, cosine %.9f\n", exact, got.size(), max_ulp,
              cosine);
  EXPECT_LE(max_ulp, 1);
  EXPECT_GE(static_cast<double>(exact), 0.999 * static_cast<double>(got.size()));

  // The plain embedder path (no extra): the original ops, and with an all-zero control and a [W0 | 0] weight the widened
  // path gives the same bits (zero products add exact zeros).
  std::vector<std::uint16_t> w0(static_cast<std::size_t>(kDim * 128)), wz(static_cast<std::size_t>(kDim * 384), 0);
  for (std::int64_t r = 0; r < kDim; ++r) {
    for (std::int64_t c = 0; c < 128; ++c) {
      w0[static_cast<std::size_t>(r * 128 + c)] = wz[static_cast<std::size_t>(r * 384 + c)] = ew.w[static_cast<std::size_t>(r * 384 + c)];
    }
  }
  DeviceBuffer dw0(w0.size() * 2), dwz(wz.size() * 2), dout2(dout.size());
  dw0.upload(w0.data(), w0.size() * 2);
  dwz.upload(wz.data(), wz.size() * 2);
  MI_CUDA_CHECK(cudaMemset(dc.get(), 0, dc.size()));
  const Tensor out2 = Tensor::view(dout2.get(), DType::kBF16, Device::kCUDA, {hw, kDim});
  moss_dit::embed_video(gemm, ws.get(), ws.size(), tl, nullptr, dt.get(),
                        Tensor::view(dw0.get(), DType::kBF16, Device::kCUDA, {kDim, 128}), emb.bias(), out, nullptr);
  moss_dit::embed_video(gemm, ws.get(), ws.size(), tl, &tc, dt.get(),
                        Tensor::view(dwz.get(), DType::kBF16, Device::kCUDA, {kDim, 384}), emb.bias(), out2, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint16_t> a(got.size()), b(got.size());
  dout.download(a.data(), a.size() * 2);
  dout2.download(b.data(), b.size() * 2);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
  std::printf("[control] zero control with [W0 | 0] vs the plain embedder: %zu / %zu elements differ\n", diff, a.size());
  EXPECT_EQ(diff, 0u);
}

// The column-offset latent_to_tokens: at column 0 of a C-wide token matrix it is the plain transpose.
TEST(ControlEmbedding, OffsetTransposeMatchesPlain) {
  const std::int64_t c = 128, h = 5, w = 7, hw = h * w;
  const std::vector<float> lat = randn(static_cast<std::size_t>(c * hw), 9, 3.0f);
  DeviceBuffer dl(lat.size() * 4), d1(static_cast<std::size_t>(hw * c) * 2), d2(static_cast<std::size_t>(hw * 3 * c) * 2);
  dl.upload(lat.data(), lat.size() * 4);
  MI_CUDA_CHECK(cudaMemset(d2.get(), 0xFF, d2.size()));
  const Tensor tl = Tensor::view(dl.get(), DType::kF32, Device::kCUDA, {c, h, w});
  moss_dit::latent_to_tokens(tl, Tensor::view(d1.get(), DType::kBF16, Device::kCUDA, {hw, c}), nullptr);
  const Tensor wide = Tensor::view(d2.get(), DType::kBF16, Device::kCUDA, {hw, 3 * c});
  moss_dit::latent_to_tokens(tl, wide, 0, nullptr);
  moss_dit::latent_to_tokens(tl, wide, 2 * c, nullptr);
  EXPECT_THROW(moss_dit::latent_to_tokens(tl, wide, 2 * c + 1, nullptr), std::invalid_argument);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint16_t> plain(static_cast<std::size_t>(hw * c)), all(static_cast<std::size_t>(hw * 3 * c));
  d1.download(plain.data(), plain.size() * 2);
  d2.download(all.data(), all.size() * 2);
  for (std::int64_t t = 0; t < hw; ++t) {
    for (std::int64_t k = 0; k < c; ++k) {
      const std::uint16_t p = plain[static_cast<std::size_t>(t * c + k)];
      ASSERT_EQ(all[static_cast<std::size_t>(t * 3 * c + k)], p);
      ASSERT_EQ(all[static_cast<std::size_t>(t * 3 * c + c + k)], 0xFFFF);  // untouched
      ASSERT_EQ(all[static_cast<std::size_t>(t * 3 * c + 2 * c + k)], p);
    }
  }
}

// ---- Request, report and metadata ----------------------------------------------------------------------------------

std::shared_ptr<const pl::Image> gray(int w, int h, std::uint8_t v) {
  auto im = std::make_shared<pl::Image>();
  im->width = w;
  im->height = h;
  im->channels = 1;
  im->pixels.assign(static_cast<std::size_t>(w) * h, v);
  im->name = "depth.png";
  return im;
}

pl::Request control_request() {
  pl::Request r;
  r.prompt = "a room";
  r.control.adapter = "ctl.safetensors";
  pl::Request::ControlImage c;
  c.kind = "depth";
  c.image = gray(64, 48, 200);
  c.end = 0.5;
  r.control.images.push_back(c);
  return r;
}

TEST(ControlRequest, Validation) {
  EXPECT_NO_THROW(pl::validate(control_request()));
  pl::Request r = control_request();
  r.control.adapter.clear();
  EXPECT_THROW(pl::validate(r), std::invalid_argument);  // images without an adapter
  r = control_request();
  r.control.images.push_back(r.control.images[0]);
  EXPECT_THROW(pl::validate(r), std::invalid_argument);  // duplicate kind
  for (double s : {-0.1, 10.5, std::nan("")}) {
    r = control_request();
    r.control.images[0].strength = s;
    EXPECT_THROW(pl::validate(r), std::invalid_argument) << s;
  }
  r = control_request();
  r.control.images[0].start = 0.6;  // > end
  EXPECT_THROW(pl::validate(r), std::invalid_argument);
  r = control_request();
  r.control.images[0].kind.clear();
  EXPECT_THROW(pl::validate(r), std::invalid_argument);
  r = control_request();
  r.control.images[0].image = nullptr;
  EXPECT_THROW(pl::validate(r), std::invalid_argument);
  r = control_request();
  r.control.lora_strength = INFINITY;
  EXPECT_THROW(pl::validate(r), std::invalid_argument);
  r = control_request();
  r.control.images.clear();  // an adapter without images: zero control latents
  EXPECT_NO_THROW(pl::validate(r));
}

TEST(ControlRequest, ReportAndMetadata) {
  pl::Request q = control_request();
  pl::Result r;
  r.control.adapter = "ctl.safetensors";
  r.control.kinds = {"color", "depth"};
  r.control.lora.file = "ctl.safetensors";
  r.control.lora.adapters = 262;
  r.control.encodes = 1;
  const std::string rep = pl::report_fields(q, r, "x.png", "");
  EXPECT_NE(rep.find("\"control\": {\"adapter\": \"ctl.safetensors\", \"lora_strength\": 1, \"kinds\": [\"color\", \"depth\"]"),
            std::string::npos)
      << rep;
  EXPECT_NE(rep.find("{\"kind\": \"depth\", \"image\": {\"name\": \"depth.png\", \"width\": 64, \"height\": 48}, \"strength\": 1, "
                     "\"start\": 0, \"end\": 0.5}"),
            std::string::npos)
      << rep;
  EXPECT_NE(rep.find("\"adapters\": 262"), std::string::npos);
  const pl::GenerationMetadata m = pl::generation_metadata(q, r);
  bool control = false, depth = false;
  for (const auto& [k, v] : m.extra) {
    control |= k == "control" && v == "ctl.safetensors";
    depth |= k == "control_depth" && v == "depth.png 64x48 strength 1 steps 0-0.5";
  }
  EXPECT_TRUE(control);
  EXPECT_TRUE(depth);
  pl::Request none;
  none.prompt = "a room";
  EXPECT_NE(pl::report_fields(none, pl::Result{}, "x.png", "").find("\"control\": null"), std::string::npos);
}

// ---- End to end (real weights) -------------------------------------------------------------------------------------

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
bool have_e2e() { return fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") && fs::is_regular_file(kControlFile); }

std::size_t bit_mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return n;
}

// A depth map: a bright disc (near) on a dark background.
std::shared_ptr<const pl::Image> disc_depth(int w, int h) {
  auto im = std::make_shared<pl::Image>();
  im->width = w;
  im->height = h;
  im->channels = 1;
  im->pixels.resize(static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double dx = (x - w * 0.5) / (w * 0.3), dy = (y - h * 0.5) / (h * 0.3);
      im->pixels[static_cast<std::size_t>(y) * w + x] = dx * dx + dy * dy < 1 ? 230 : 25;
    }
  }
  im->name = "disc.png";
  return im;
}

pl::Request plain_request() {
  pl::Request req;
  req.model = weights_root() / "sft";
  req.prompt = "a red ball on a wooden table, photo";
  req.width = 512;
  req.height = 512;
  req.steps = 4;
  req.seed = 7;
  req.decode = false;
  return req;
}

// `base` with the control adapter and one image of `kind` (the disc depth map) over the step window [0, end).
pl::Request control_request(const pl::Request& base, const char* kind = "depth", double end = 1.0) {
  pl::Request c = base;
  c.control.adapter = kControlFile;
  pl::Request::ControlImage im;
  im.kind = kind;
  im.image = disc_depth(base.width, base.height);
  im.end = end;
  c.control.images.push_back(im);
  return c;
}

// The control changes the result and is deterministic; toggling it off and on switches the merged text-refiner
// adapters in place (no DiT load: the first request sized the arena with the control) and a request without it is
// bit-identical to one from a pipeline that never saw a control; the control's LoRA strength alone re-merges in place;
// the kind decides the channels; a control file passed as a plain LoRA is refused right after a control request (the
// stack cache does not hide it).
TEST(GpuControlPipeline, ControlRunsAndLeavesNoState) {
  if (!have_e2e()) GTEST_SKIP() << "SFT bundle or control adapter not present";
  const pl::Request req = plain_request();
  const pl::Request c = control_request(req);
  std::vector<float> plain_after;
  {
    pl::Pipeline pipe;
    const pl::Result a = pipe.generate(c);
    ASSERT_EQ(a.first_pass.schedule_steps, 4);
    EXPECT_EQ(a.control.encodes, 1);
    EXPECT_EQ(a.control.lora.adapters, 262);
    EXPECT_EQ(a.control.lora.merged, 10);
    EXPECT_EQ(a.loras.size(), 0u);      // the control's LoRA part is reported with the control
    EXPECT_TRUE(a.lora_notes.empty());  // and so are its notes
    EXPECT_FALSE(a.control.lora.notes.empty());
    EXPECT_FALSE(a.tier.certified);

    // Control off: restored in place.
    const pl::Result off = pipe.generate(req);
    EXPECT_EQ(off.memory.dit_loads, 0) << "control off: restored in place";
    EXPECT_FALSE(off.memory.dit_reloaded);
    EXPECT_EQ(off.control.adapter, "");
    EXPECT_GT(bit_mismatches(a.latents, off.latents), a.latents.size() / 2);
    plain_after = off.latents;

    // Control on again: merged in place, the same bits as the first control run.
    const pl::Result b = pipe.generate(c);
    EXPECT_EQ(b.memory.dit_loads, 0) << "control on: merged in place";
    EXPECT_FALSE(b.memory.dit_reloaded);
    EXPECT_EQ(bit_mismatches(a.latents, b.latents), 0u);

    // Channel order: the same map as the colour input (channels 128..255) and as the depth input (256..383).
    const pl::Result as_color = pipe.generate(control_request(req, "color"));
    EXPECT_GT(bit_mismatches(as_color.latents, a.latents), a.latents.size() / 2);

    // The LoRA strength alone: re-merged from the pristine copies in place, and back to the same bits.
    pl::Request half_lora = c;
    half_lora.control.lora_strength = 0.5f;
    const pl::Result hl = pipe.generate(half_lora);
    EXPECT_EQ(hl.memory.dit_loads, 0);
    EXPECT_GT(bit_mismatches(hl.latents, a.latents), a.latents.size() / 2);
    const pl::Result back = pipe.generate(c);
    EXPECT_EQ(back.memory.dit_loads, 0);
    EXPECT_EQ(bit_mismatches(back.latents, a.latents), 0u);

    // M1: right after a control request, the same file as a plain LoRA is refused.
    pl::Request wrong = req;
    wrong.loras = {{kControlFile, 1.0f}};
    EXPECT_THROW(pipe.generate(wrong), std::invalid_argument);
    // An unknown kind.
    EXPECT_THROW(pipe.generate(control_request(req, "canny")), std::invalid_argument);
    // Off once more after all that: still the same plain bits.
    EXPECT_EQ(bit_mismatches(pipe.generate(req).latents, plain_after), 0u);
  }
  // A pipeline that never saw a control.
  pl::Pipeline fresh;
  EXPECT_EQ(bit_mismatches(fresh.generate(req).latents, plain_after), 0u);
}

// Step windows: [0, 0.5) of 4 steps is steps {0, 1}: the same as [0, 0.3), and different from one step fewer ([0, 0.25):
// {0}) or one more ([0, 0.51): {0, 1, 2}); a window that never opens equals strength 0 (zero latents, adapter loaded).
TEST(GpuControlPipeline, StepWindows) {
  if (!have_e2e()) GTEST_SKIP() << "SFT bundle or control adapter not present";
  pl::Pipeline pipe;
  const pl::Request req = plain_request();
  const auto run = [&](double end) { return pipe.generate(control_request(req, "depth", end)).latents; };
  const std::vector<float> full = run(1.0), half = run(0.5), half2 = run(0.3), earlier = run(0.25), later = run(0.51),
                           never = run(0.0);
  pl::Request s0 = control_request(req);
  s0.control.images[0].strength = 0;
  const std::vector<float> zero = pipe.generate(s0).latents;
  EXPECT_EQ(bit_mismatches(half, half2), 0u);
  EXPECT_GT(bit_mismatches(half, earlier), 0u);
  EXPECT_GT(bit_mismatches(half, later), 0u);
  EXPECT_GT(bit_mismatches(half, full), 0u);
  EXPECT_GT(bit_mismatches(later, full), 0u);  // [0, 0.51) misses step 3
  EXPECT_EQ(bit_mismatches(never, zero), 0u);
  EXPECT_GT(bit_mismatches(never, earlier), 0u);
}

// Hires fix: the image is encoded again at the target size, and the window is measured over each pass's own steps:
// with 4 first-pass steps and 3 hires steps, [0, 0.5) and [0, 0.3) agree on the first pass ({0, 1}) but not on the
// second ({0, 1} vs {0}).
TEST(GpuControlPipeline, HiresReencodesAndWindowsEachPass) {
  if (!have_e2e()) GTEST_SKIP() << "SFT bundle or control adapter not present";
  pl::Pipeline pipe;
  const pl::Request req = plain_request();
  EXPECT_EQ(bit_mismatches(pipe.generate(control_request(req, "depth", 0.5)).latents,
                           pipe.generate(control_request(req, "depth", 0.3)).latents),
            0u);
  const auto hires = [&](double end) {
    pl::Request h = control_request(req, "depth", end);
    h.hires.width = 640;
    h.hires.height = 640;
    h.hires.steps = 6;
    h.hires.strength = 0.5;
    return pipe.generate(h);
  };
  const pl::Result h5 = hires(0.5), h3 = hires(0.3);
  ASSERT_TRUE(h5.hires.ran);
  ASSERT_EQ(h5.hires.schedule_steps - h5.hires.start_step, 3);
  EXPECT_EQ(h5.control.encodes, 2);
  // Two encodes: at 512 x 512, then at 640 x 640.
  const std::string& es = h5.control.encode_schedule;
  EXPECT_NE(es.find("; "), std::string::npos) << es;
  EXPECT_EQ(h5.width, 640);
  EXPECT_EQ(h5.latents.size(), static_cast<std::size_t>(128 * 40 * 40));
  EXPECT_GT(bit_mismatches(h5.latents, h3.latents), 0u);
}

// A cancel inside a control pass leaves no state: a plain run afterwards is bit-identical to a fresh plain run.
TEST(GpuControlPipeline, CancelLeavesNoState) {
  if (!have_e2e()) GTEST_SKIP() << "SFT bundle or control adapter not present";
  pl::Pipeline pipe;
  const pl::Request req = plain_request();
  const pl::Result plain = pipe.generate(req);
  pl::GenerateControl stop;
  stop.on_step = [](const pl::StepInfo& info) { return info.step < 2; };
  const pl::Result cancelled = pipe.generate(control_request(req), stop);
  EXPECT_TRUE(cancelled.cancelled);
  EXPECT_EQ(cancelled.steps_run, 2);
  const pl::Result after = pipe.generate(req);
  EXPECT_EQ(bit_mismatches(after.latents, plain.latents), 0u);
}

// M1, the other order: LoRA Y + control X, then plain LoRAs [Y, X] (the same files in the same order) is refused.
TEST(GpuControlPipeline, ControlFileAsLoraAfterControlWithLora) {
  if (!have_e2e() || !fs::is_regular_file(kPeftLora)) GTEST_SKIP() << "SFT bundle, control adapter or LoRA not present";
  pl::Pipeline pipe;
  pl::Request req = plain_request();
  req.steps = 2;
  req.loras = {{kPeftLora, 0.5f}};
  const pl::Result r = pipe.generate(control_request(req));
  EXPECT_EQ(r.loras.size(), 1u);
  EXPECT_EQ(r.control.lora.adapters, 262);
  pl::Request wrong = req;
  wrong.loras.push_back({kControlFile, 1.0f});
  EXPECT_THROW(pipe.generate(wrong), std::invalid_argument);
}

// ---- In-place merge switching (MossDit::rebase_merges) against fresh pipelines --------------------------------------

const fs::path kInt8Dit = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";

// Synthetic LoRA files: S, a rank-4 PEFT adapter on text_embedder.blocks.0.qkv (a module the control also merges);
// S_huge, the same with all-one factors (B·A = 4 everywhere: a strength of 1e38 overflows the merged weight); D, a full
// delta on layers.0.proj.q (a block module: merged at load only).
struct SyntheticLoras {
  testing::TempDir dir{"control_loras"};
  fs::path s = dir.path() / "s.safetensors", s_huge = dir.path() / "s_huge.safetensors", d = dir.path() / "d.safetensors";
  SyntheticLoras() {
    const std::string m = "text_embedder.blocks.0.qkv";
    write_st(s, {{m + ".lora_A.weight", WDType::kBF16, {4, kDim}, randn(4 * kDim, 11, 0.02f)},
                 {m + ".lora_B.weight", WDType::kBF16, {3 * kDim, 4}, randn(12 * kDim, 12, 0.02f)}});
    write_st(s_huge, {{m + ".lora_A.weight", WDType::kBF16, {4, kDim}, std::vector<float>(4 * kDim, 1.0f)},
                      {m + ".lora_B.weight", WDType::kBF16, {3 * kDim, 4}, std::vector<float>(12 * kDim, 1.0f)}});
    write_st(d, {{"layers.0.proj.q.diff", WDType::kBF16, {kDim, kDim}, randn(static_cast<std::size_t>(kDim * kDim), 13, 1e-3f)}});
  }
};

std::vector<float> fresh_run(const pl::Request& q, const pl::PipelineOptions& o = {}) {
  pl::Pipeline p(o);
  return p.generate(q).latents;
}

// H1: an in-place re-merge that fails (a non-finite merged weight) leaves nothing behind: the request throws, and the
// next control request gives the bits of a fresh pipeline's (the DiT reloaded clean), and a plain request works.
TEST(GpuControlMerges, FailedRebaseLeavesNoState) {
  if (!have_e2e()) GTEST_SKIP() << "SFT bundle or control adapter not present";
  const SyntheticLoras f;
  const pl::Request req = plain_request();
  const pl::Request c = control_request(req);
  std::vector<float> first, again, plain;
  {
    pl::Pipeline pipe;
    first = pipe.generate(c).latents;
    pl::Request huge = c;
    huge.loras = {{f.s_huge, 1e38f}};
    EXPECT_THROW(pipe.generate(huge), std::runtime_error);
    again = pipe.generate(c).latents;
    const pl::Result pr = pipe.generate(req);
    EXPECT_EQ(pr.steps_run, 4);
    plain = pr.latents;
  }
  EXPECT_EQ(bit_mismatches(again, first), 0u);
  EXPECT_EQ(bit_mismatches(plain, fresh_run(req)), 0u);
}

// A LoRA on a module the control also merges, toggled with the control and a strength change: every request equals
// a fresh pipeline's, and no request after the first loads the DiT.
TEST(GpuControlMerges, SharedModuleToggleMatchesFresh) {
  if (!have_e2e()) GTEST_SKIP() << "SFT bundle or control adapter not present";
  const SyntheticLoras f;
  pl::Request s = plain_request();
  s.loras = {{f.s, 1.0f}};
  const pl::Request sc = control_request(s);
  pl::Request sc5 = sc;
  sc5.loras[0].strength = 0.5f;
  std::vector<std::vector<float>> seq;
  {
    pl::Pipeline pipe;
    int i = 0;
    for (const pl::Request* q : std::vector<const pl::Request*>{&sc, &s, &sc5, &s}) {
      const pl::Result r = pipe.generate(*q);
      if (i++ > 0) {
        EXPECT_EQ(r.memory.dit_loads, 0) << "request " << i;
        EXPECT_FALSE(r.memory.dit_reloaded) << "request " << i;
      }
      seq.push_back(r.latents);
    }
  }
  const std::vector<float> fresh_sc = fresh_run(sc), fresh_s = fresh_run(s), fresh_sc5 = fresh_run(sc5);
  EXPECT_EQ(bit_mismatches(seq[0], fresh_sc), 0u);
  EXPECT_EQ(bit_mismatches(seq[1], fresh_s), 0u);
  EXPECT_EQ(bit_mismatches(seq[2], fresh_sc5), 0u);
  EXPECT_EQ(bit_mismatches(seq[3], fresh_s), 0u);
  EXPECT_GT(bit_mismatches(fresh_s, fresh_sc), 0u);
  EXPECT_GT(bit_mismatches(fresh_sc, fresh_sc5), 0u);
}

// A regular (side-branch) LoRA stays when the control goes: Y + control, then Y, equals a fresh Y.
TEST(GpuControlMerges, RegularLoraStaysWhenControlGoes) {
  if (!have_e2e() || !fs::is_regular_file(kPeftLora)) GTEST_SKIP() << "SFT bundle, control adapter or LoRA not present";
  pl::Request y = plain_request();
  y.loras = {{kPeftLora, 0.8f}};
  std::vector<float> after;
  {
    pl::Pipeline pipe;
    pipe.generate(control_request(y));
    const pl::Result r = pipe.generate(y);
    EXPECT_EQ(r.memory.dit_loads, 0);
    after = r.latents;
  }
  EXPECT_EQ(bit_mismatches(after, fresh_run(y)), 0u);
}

// The false-return path: after a control request, a stack that newly merges a block module reloads the DiT once
// (merged at load) and equals a fresh pipeline's.
TEST(GpuControlMerges, NewBlockMergeReloads) {
  if (!have_e2e()) GTEST_SKIP() << "SFT bundle or control adapter not present";
  const SyntheticLoras f;
  pl::Request d = plain_request();
  d.loras = {{f.d, 1.0f}};
  std::vector<float> after;
  {
    pl::Pipeline pipe;
    pipe.generate(control_request(plain_request()));
    const pl::Result r = pipe.generate(d);
    EXPECT_EQ(r.memory.dit_loads, 1);
    after = r.latents;
  }
  EXPECT_EQ(bit_mismatches(after, fresh_run(d)), 0u);
  EXPECT_GT(bit_mismatches(after, fresh_run(plain_request())), 0u);
}

// The INT8 DiT (High tier): control on, off, on again, in place; the plain run equals a fresh pipeline's.
TEST(GpuControlMerges, Int8ToggleMatchesFresh) {
  if (!have_e2e() || !fs::is_regular_file(kInt8Dit)) GTEST_SKIP() << "SFT bundle, control adapter or INT8 DiT not present";
  pl::PipelineOptions o;
  o.dit_file = kInt8Dit;
  o.tier = planner::Tier::kHigh;
  const pl::Request req = plain_request();
  const pl::Request c = control_request(req);
  std::vector<float> first, plain, again;
  {
    pl::Pipeline pipe(o);
    first = pipe.generate(c).latents;
    const pl::Result off = pipe.generate(req);
    EXPECT_EQ(off.memory.dit_loads, 0);
    plain = off.latents;
    const pl::Result on = pipe.generate(c);
    EXPECT_EQ(on.memory.dit_loads, 0);
    again = on.latents;
  }
  EXPECT_EQ(bit_mismatches(again, first), 0u);
  EXPECT_EQ(bit_mismatches(plain, fresh_run(req, o)), 0u);
  EXPECT_GT(bit_mismatches(plain, first), plain.size() / 2);
}

}  // namespace
}  // namespace mi::control
