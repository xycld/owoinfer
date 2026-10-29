// Edit adapters' reference images (Moss-Image-Edit, moss_dit::RefKv):
//   - the reference RoPE table against the image table (golden-tested against the official _RoPE): every reference row
//     is the image rows' arithmetic with the axis-0 coordinate i + 1 (= the text rows' axis-1 values at position i + 1);
//   - the DiT against ComfyUI's implementation (PR #17296, tools/reference/edit_oracle.py) on the oracle's first step:
//     the reference keys / values of blocks 0, 21 and 41 and both CFG calls' velocities, gated against the no-reference
//     difference between the two implementations; again under 8-bit attention (q8_fp8), against its own floor;
//   - the reference path's own invariants, bit for bit: a row's output does not depend on the other row of its forward
//     (exact and q8_fp8), set_references(nullptr) restores the no-reference forward, and the op-by-op prefill and
//     forward equal the fused ones.
// The DiT tests need the oracle's tensors as .npy (MOSS_EDIT_ORACLE: x, ctx_{cond,uncond}, out_{cond,uncond}[_noref],
// kv_{0,21,41}, ref_latent_in), the SFT bf16 DiT file (MOSS_EDIT_DIT) and the chinoll adapter (MOSS_EDIT_LORA); they
// skip without them.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/rope.h"
#include "models/flux2_vae_encoder/flux2_vae_encoder.h"
#include "models/lora/lora.h"
#include "models/lora/side_branch.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/report_json.h"
#include "planner/host_memory.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "test_util.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/quant/quant_convert.h"
#include "weights/safetensors.h"
#include "weights/test_seams.h"

namespace {

namespace fs = std::filesystem;
namespace md = mi::moss_dit;
using mi::Tensor;

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
float from_bf16(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float v = 0;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}
std::uint16_t to_bf16(float v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
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

TEST(GpuEditReferences, RopeTableIsTheImageArithmeticOnAxisZero) {
  for (const auto& [h, w] : {std::pair{32, 24}, std::pair{7, 5}, std::pair{1, 9}}) {
    const std::int64_t n = std::int64_t{h} * w;
    for (const int index : {0, 1, 3}) {
      // Image rows of the same grid (axes 2, 3; axis 0 is 0) and text rows (axis 1 = position) long enough to hold
      // position index + 1.
      const int text = index + 2;
      DevTensor ref(mi::DType::kBF16, {n, 128}), img(mi::DType::kBF16, {n + text, 128});
      mi::kernels::moss_rope_table_reference(ref.t, h, w, index, nullptr);
      mi::kernels::moss_rope_table(img.t, h, w, text, nullptr);
      const auto r = download_u16(ref.t), m = download_u16(img.t);
      const std::size_t pos = static_cast<std::size_t>(n + index + 1) * 128;  // text row at position index + 1
      std::size_t differ = 0;
      for (std::int64_t t = 0; t < n; ++t) {
        for (int c = 0; c < 128; ++c) {
          const int axis = (c & 63) >> 4, band = c & 15, half = c & 64;
          std::uint16_t want = 0;
          if (axis == 0) want = m[pos + static_cast<std::size_t>(half + 16 + band)];  // text axis 1 at index + 1
          if (axis == 1) want = half ? 0x3f80 : 0;  // coordinate 0: sin 0 = 0, cos 0 = 1
          if (axis >= 2) want = m[static_cast<std::size_t>(t) * 128 + static_cast<std::size_t>(c)];
          differ += r[static_cast<std::size_t>(t) * 128 + static_cast<std::size_t>(c)] != want;
        }
      }
      EXPECT_EQ(differ, 0u) << h << "x" << w << " reference " << index;
    }
  }
  DevTensor bad(mi::DType::kBF16, {10, 128});
  EXPECT_THROW(mi::kernels::moss_rope_table_reference(bad.t, 3, 3, 0, nullptr), std::invalid_argument);
  EXPECT_THROW(mi::kernels::moss_rope_table_reference(bad.t, 2, 5, -1, nullptr), std::invalid_argument);
}

fs::path env_path(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? fs::path(v) : fs::path(fallback);
}

struct Stats {
  double cosine = 0, rel_l2 = 0;
};
Stats compare(const std::vector<float>& want, const std::vector<float>& got) {
  double dot = 0, ww = 0, gg = 0, dd = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    dot += double{want[i]} * got[i];
    ww += double{want[i]} * want[i];
    gg += double{got[i]} * got[i];
    dd += (double{want[i]} - got[i]) * (double{want[i]} - got[i]);
  }
  return {dot / std::sqrt(ww * gg), std::sqrt(dd / ww)};
}

// moss_edit/processing.py resize_image's sizes (training rule): limit = max_edge // 16 * 16, fit, round each side to 16,
// never above the limit nor below 16, no upscale beyond the rounding.
TEST(EditReferences, SizesFollowTheTrainingRule) {
  using mi::pipeline::reference_size;
  const auto eq = [](mi::pipeline::ReferenceSize s, int w, int h) { return s.width == w && s.height == h; };
  EXPECT_TRUE(eq(reference_size(1536, 2048, 512), 384, 512));   // the oracle's asset019
  EXPECT_TRUE(eq(reference_size(2048, 1536, 256), 256, 192));
  EXPECT_TRUE(eq(reference_size(1000, 1000, 520), 512, 512));   // max edge floored to 512
  EXPECT_TRUE(eq(reference_size(300, 200, 512), 304, 208));      // no downscale: sides rounded to 16 (half up)
  EXPECT_TRUE(eq(reference_size(296, 200, 512), 304, 208));      // 18.5 -> 19
  EXPECT_TRUE(eq(reference_size(4000, 20, 512), 512, 16));       // at least 16
  EXPECT_TRUE(eq(reference_size(8, 8, 512), 16, 16));
  EXPECT_TRUE(eq(reference_size(509, 100, 512), 512, 96));       // rounded up, capped at the limit
}

// RGB of 1 / 3 / 4 channels as PIL convert("RGB"), then the PIL-exact Lanczos; the oracle's asset resized byte for byte
// as its ref.png (MOSS_EDIT_ORACLE, with ref.png copied in).
TEST(EditReferences, PixelsAreConvertRgbThenLanczos) {
  mi::pipeline::Image rgba;
  rgba.width = 3;
  rgba.height = 1;
  rgba.channels = 4;
  rgba.pixels = {1, 2, 3, 9, 4, 5, 6, 9, 7, 8, 9, 9};
  const mi::pipeline::Image a = mi::pipeline::reference_pixels(rgba, 512);  // 3x1 -> 16x16
  mi::pipeline::Image rgb = rgba;
  rgb.channels = 3;
  rgb.pixels = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_EQ(a.pixels, mi::pipeline::reference_pixels(rgb, 512).pixels);
  mi::pipeline::Image gray = rgba;
  gray.channels = 1;
  gray.pixels = {10, 20, 30};
  const mi::pipeline::Image g = mi::pipeline::reference_pixels(gray, 512);
  ASSERT_EQ(g.channels, 3);
  for (std::size_t i = 0; i < g.pixels.size(); i += 3) {
    EXPECT_EQ(g.pixels[i], g.pixels[i + 1]);
    EXPECT_EQ(g.pixels[i], g.pixels[i + 2]);
  }
  const fs::path oracle = env_path("MOSS_EDIT_ORACLE", "");
  const fs::path asset = fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "reference_images" / "asset019_sft.png";
  if (oracle.empty() || !fs::is_regular_file(oracle / "ref.png")) GTEST_SKIP() << "MOSS_EDIT_ORACLE/ref.png not present";
  const mi::pipeline::Image got = mi::pipeline::reference_pixels(mi::pipeline::load_image(asset, 3), 512);
  const mi::pipeline::Image want = mi::pipeline::load_image(oracle / "ref.png", 3);
  ASSERT_EQ(got.width, want.width);
  ASSERT_EQ(got.height, want.height);
  EXPECT_TRUE(got.pixels == want.pixels) << "the reference resize differs from the oracle's ref.png";
}

// The engine's VAE encode (the bundle's VAE, posterior mean, normalized: img2img's z0) of the oracle's ref.png against
// ComfyUI's process_latent_in(vae.encode(...)).
TEST(GpuEditReferences, EncodeMatchesTheOracle) {
  const fs::path oracle = env_path("MOSS_EDIT_ORACLE", "");
  const fs::path vae = env_path("MOSS_EDIT_VAE", "/workspace/Models/MOSS-Image/sft/vae/diffusion_pytorch_model.safetensors");
  if (oracle.empty() || !fs::is_regular_file(oracle / "ref.png") || !fs::is_regular_file(vae)) {
    GTEST_SKIP() << "MOSS_EDIT_ORACLE/ref.png or the VAE not present";
  }
  const mi::pipeline::Image im = mi::pipeline::load_image(oracle / "ref.png", 3);
  mi::runtime::StreamSet streams;
  const std::size_t bytes = mi::flux2_vae_encoder::Flux2VaeEncoder::persistent_bytes(vae) +
                            mi::flux2_vae_encoder::Flux2VaeEncoder::workspace_bytes(im.height, im.width);
  mi::runtime::DeviceArena arena(bytes + (std::size_t{64} << 20));
  mi::flux2_vae_encoder::Flux2VaeEncoder enc(vae, arena, streams);
  DevTensor lat(mi::DType::kF32, {128, im.height / 16, im.width / 16});
  const Tensor rgb = Tensor::view(const_cast<std::uint8_t*>(im.pixels.data()), mi::DType::kU8, mi::Device::kCPU,
                                  {im.height, im.width, 3});
  enc.encode(rgb, lat.t, mi::flux2_vae_encoder::EncodeOptions{}, streams.compute());
  MI_CUDA_CHECK(cudaStreamSynchronize(streams.compute()));
  // The oracle's VAE ran in ComfyUI's default bf16 (0.99986 from its own fp32 encode): the gate is against the fp32
  // encode of the same file (ref_latent_fp32.npy) when present, the bf16 one is reported and held to that floor.
  const std::vector<float> got = download_f32(lat.t);
  const Stats s = compare(mi::npy::load(oracle / "ref_latent_in.npy").to_vector<float>(), got);
  std::printf("[edit parity] reference latent vs the oracle (bf16 VAE): cosine %.7f rel L2 %.5f\n", s.cosine, s.rel_l2);
  EXPECT_GE(s.cosine, 0.9998);
  if (fs::is_regular_file(oracle / "ref_latent_fp32.npy")) {
    const Stats f = compare(mi::npy::load(oracle / "ref_latent_fp32.npy").to_vector<float>(), got);
    std::printf("[edit parity] reference latent vs ComfyUI's fp32 encode: cosine %.7f rel L2 %.5f\n", f.cosine, f.rel_l2);
    EXPECT_GE(f.cosine, 0.9999);
  }
}

class EditDit : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    oracle_ = env_path("MOSS_EDIT_ORACLE", "");
    const fs::path dit = env_path("MOSS_EDIT_DIT", "/workspace/Models/community/comfy-org/moss_image_sft_bf16.safetensors");
    const fs::path lora = env_path("MOSS_EDIT_LORA", "/workspace/Models/community/chinoll/Moss-Image-Edit/adapter_model.safetensors");
    if (oracle_.empty() || !fs::is_regular_file(oracle_ / "kv_0.npy") || !fs::is_regular_file(dit) || !fs::is_regular_file(lora)) {
      return;
    }
    info_ = std::make_unique<mi::pipeline::DitFileInfo>(mi::pipeline::inspect_dit_file(dit, std::string_view{}));
    const mi::lora::LoraSpec spec{lora, 1.0f};
    mi::lora::LoadOptions lo;
    lo.placement = mi::lora::Placement::kSideBranch;  // the adapter on the side branch: the model's weights untouched
    stack_ = std::make_unique<mi::lora::LoraStack>(std::span<const mi::lora::LoraSpec>(&spec, 1), mi::lora::moss_dit_targets(),
                                                   lo);
    sb_ = std::make_unique<mi::lora::SideBranches>(*stack_);
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    const mi::weights::ModelIndex& index = info_->quant->index;
    // The oracle's grid (74x55) with its two rows, the op-by-op path (the largest big buffer), the cache and the prefill.
    const std::int64_t lengths[] = {32, 10};
    // Both attention modes the tests run: exact and q8_fp8 (the quantized block path).
    std::size_t work = 0;
    for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
      const md::ComputeConfig config = md::MossDit::compute_config(info_->quant.get(), mode, md::Int4Route::kAuto, sb_.get());
      for (const md::BlockPath p : {md::BlockPath::kFused, md::BlockPath::kOpByOp}) {
        work = std::max({work, md::MossDit::workspace_bytes(74, 55, lengths, p, config, nullptr, kRefTokens),
                         md::MossDit::prefill_workspace_bytes(kRefTokens, p, config, nullptr),
                         md::MossDit::encode_text_workspace_bytes(32)});
      }
    }
    work += 2 * md::MossDit::reference_cache_bytes(kRefTokens) + (std::size_t{64} << 20);
    const std::size_t free = mi::runtime::DeviceArena::free_device_bytes() - (std::size_t{512} << 20);
    for (residency_.resident_blocks = md::kNumBlocks; residency_.resident_blocks > 0; --residency_.resident_blocks) {
      if (md::MossDit::persistent_bytes(index, residency_) + work + md::MossDit::stream_workspace_bytes(index, residency_) <= free) break;
    }
    const std::size_t bytes =
        md::MossDit::persistent_bytes(index, residency_) + work + md::MossDit::stream_workspace_bytes(index, residency_);
    std::printf("[edit] arena %.2f GiB, %d of 42 blocks resident\n", static_cast<double>(bytes) / (1 << 30),
                residency_.resident_blocks);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(bytes);
    model_ = std::make_unique<md::MossDit>(*info_->quant, *arena_, *streams_, false, nullptr, residency_);
    model_->set_lora(sb_.get());
  }
  static void TearDownTestSuite() {
    model_.reset();
    arena_.reset();
    streams_.reset();
    sb_.reset();
    stack_.reset();
    info_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "MOSS_EDIT_ORACLE, the SFT bf16 DiT or the edit adapter not present";
  }
  cudaStream_t stream() const { return streams_->compute(); }
  static mi::npy::Array load(const std::string& name) { return mi::npy::load(oracle_ / (name + ".npy")); }
  static std::unique_ptr<DevTensor> upload_f32(const mi::npy::Array& a, std::initializer_list<std::int64_t> shape) {
    auto d = std::make_unique<DevTensor>(mi::DType::kF32, shape);
    const auto v = a.to_vector<float>();
    d->buf.upload(v.data(), v.size() * 4);
    return d;
  }
  // The text refiner on the oracle's TE output (bf16 values held as fp32).
  std::unique_ptr<DevTensor> refine(const std::string& name) {
    const mi::npy::Array a = load(name);
    const std::int64_t s = a.shape[0];
    DevTensor in(mi::DType::kBF16, {s, md::kTextDim});
    const auto v = a.to_vector<float>();
    std::vector<std::uint16_t> h(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) h[i] = to_bf16(v[i]);
    in.buf.upload(h.data(), h.size() * 2);
    auto out = std::make_unique<DevTensor>(mi::DType::kBF16, std::initializer_list<std::int64_t>{s, md::kDim});
    model_->encode_text(in.t, out->t, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    return out;
  }

  static constexpr std::int64_t kRefTokens = 32 * 24;  // the oracle's 384x512 reference
  static fs::path oracle_;
  static std::unique_ptr<mi::pipeline::DitFileInfo> info_;
  static std::unique_ptr<mi::lora::LoraStack> stack_;
  static std::unique_ptr<mi::lora::SideBranches> sb_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<md::MossDit> model_;
  static md::Residency residency_;
};
fs::path EditDit::oracle_;
std::unique_ptr<mi::pipeline::DitFileInfo> EditDit::info_;
std::unique_ptr<mi::lora::LoraStack> EditDit::stack_;
std::unique_ptr<mi::lora::SideBranches> EditDit::sb_;
std::unique_ptr<mi::runtime::StreamSet> EditDit::streams_;
std::unique_ptr<mi::runtime::DeviceArena> EditDit::arena_;
std::unique_ptr<md::MossDit> EditDit::model_;
md::Residency EditDit::residency_;

// The cache's block b as fp32 [2, R, kDim] (k then v), the oracle's ref_kv layout.
std::vector<float> cache_block(const DevTensor& cache, int b) {
  const auto all = download_u16(cache.t);
  const std::size_t r = static_cast<std::size_t>(cache.t.dim(1)), d = md::kDim;
  std::vector<float> out(2 * r * d);
  const std::uint16_t* blk = all.data() + static_cast<std::size_t>(b) * r * 2 * d;
  for (std::size_t i = 0; i < r; ++i) {
    for (std::size_t c = 0; c < d; ++c) {
      out[i * d + c] = from_bf16(blk[i * 2 * d + c]);
      out[r * d + i * d + c] = from_bf16(blk[i * 2 * d + d + c]);
    }
  }
  return out;
}

TEST_F(EditDit, OracleStepZero) {
  const auto ref = upload_f32(load("ref_latent_in"), {128, 32, 24});
  const auto x = upload_f32(load("x"), {128, 74, 55});
  const auto cond = refine("ctx_cond"), uncond = refine("ctx_uncond");
  DevTensor cache(mi::DType::kBF16, {md::kLayers, kRefTokens, 2 * md::kDim});
  const md::RefKv refs{cache.t};
  const Tensor refs_in[] = {ref->t};
  model_->prefill_references(refs_in, refs, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  for (const int b : {0, 21, 41}) {
    const auto want = load("kv_" + std::to_string(b)).to_vector<float>();
    const auto got = cache_block(cache, b);
    const std::size_t half = want.size() / 2;
    const Stats k = compare(std::vector<float>(want.begin(), want.begin() + half), std::vector<float>(got.begin(), got.begin() + half));
    const Stats v = compare(std::vector<float>(want.begin() + half, want.end()), std::vector<float>(got.begin() + half, got.end()));
    std::printf("[edit parity] block %2d: K cosine %.6f rel L2 %.4f, V cosine %.6f rel L2 %.4f\n", b, k.cosine, k.rel_l2,
                v.cosine, v.rel_l2);
    EXPECT_GE(k.cosine, 0.999) << "block " << b;
    EXPECT_GE(v.cosine, 0.999) << "block " << b;
  }

  // Step 0 of the oracle: ComfyUI's timestep 1.0 is the scheduler's 1000 (flux timestep_embedding's time_factor).
  md::SkipSet s22;
  s22.set(22);
  const md::Row rows[] = {{cond->t, md::SkipSet{}}, {uncond->t, s22}};
  DevTensor o0(mi::DType::kF32, {128, 74, 55}), o1(mi::DType::kF32, {128, 74, 55});
  DevTensor n0(mi::DType::kF32, {128, 74, 55}), n1(mi::DType::kF32, {128, 74, 55});
  const Tensor with[] = {o0.t, o1.t}, without[] = {n0.t, n1.t};
  model_->forward(x->t, 1000.0f, rows, without, stream());
  model_->set_references(&refs);
  model_->forward(x->t, 1000.0f, rows, with, stream());
  model_->set_references(nullptr);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const char* names[] = {"cond", "uncond"};
  for (int r = 0; r < 2; ++r) {
    const std::string n = names[r];
    const Stats floor = compare(load("out_" + n + "_noref").to_vector<float>(), download_f32(without[r]));
    const Stats got = compare(load("out_" + n).to_vector<float>(), download_f32(with[r]));
    const Stats moved = compare(load("out_" + n + "_noref").to_vector<float>(), load("out_" + n).to_vector<float>());
    std::printf("[edit parity] %-6s velocity: with references cosine %.6f rel L2 %.4f; no-reference floor cosine %.6f rel "
                "L2 %.4f; the references move the oracle by rel L2 %.4f\n",
                n.c_str(), got.cosine, got.rel_l2, floor.cosine, floor.rel_l2, moved.rel_l2);
    EXPECT_LE(got.rel_l2, 1.5 * floor.rel_l2 + 1e-3) << n;
    EXPECT_GE(got.cosine, std::min(floor.cosine - 1e-4, 0.99988)) << n;
  }
}

// 8-bit attention (q8_fp8, the quantized block path: the prefill's attention, and every forward's over the slots through
// cu_seqlens_k with the workspace's total keys) against the oracle, gated against q8's own no-reference floor; then,
// without the side branch, a row's bits with references do not depend on the other row of its forward.
TEST_F(EditDit, OracleStepZeroQ8) {
  if (!md::attention_supported(md::AttentionMode::kQ8Fp8)) GTEST_SKIP() << "q8_fp8 attention needs FP8 (sm_89+)";
  struct RestoreExact {
    ~RestoreExact() {
      model_->set_lora(sb_.get());
      model_->set_attention(md::AttentionMode::kExact);
    }
  } restore;
  model_->set_attention(md::AttentionMode::kQ8Fp8);
  const auto ref = upload_f32(load("ref_latent_in"), {128, 32, 24});
  const auto x = upload_f32(load("x"), {128, 74, 55});
  const auto cond = refine("ctx_cond"), uncond = refine("ctx_uncond");
  DevTensor cache(mi::DType::kBF16, {md::kLayers, kRefTokens, 2 * md::kDim});
  const md::RefKv refs{cache.t};
  const Tensor refs_in[] = {ref->t};
  model_->prefill_references(refs_in, refs, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  for (const int b : {0, 21, 41}) {
    const auto want = load("kv_" + std::to_string(b)).to_vector<float>();
    const Stats kv = compare(want, cache_block(cache, b));
    std::printf("[edit parity q8] block %2d: K|V cosine %.6f rel L2 %.4f\n", b, kv.cosine, kv.rel_l2);
    EXPECT_GE(kv.cosine, 0.995) << "block " << b;
  }
  md::SkipSet s22;
  s22.set(22);
  const md::Row rows[] = {{cond->t, md::SkipSet{}}, {uncond->t, s22}};
  DevTensor o0(mi::DType::kF32, {128, 74, 55}), o1(mi::DType::kF32, {128, 74, 55});
  DevTensor n0(mi::DType::kF32, {128, 74, 55}), n1(mi::DType::kF32, {128, 74, 55});
  const Tensor with[] = {o0.t, o1.t}, without[] = {n0.t, n1.t};
  model_->forward(x->t, 1000.0f, rows, without, stream());
  model_->set_references(&refs);
  model_->forward(x->t, 1000.0f, rows, with, stream());
  model_->set_references(nullptr);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const char* names[] = {"cond", "uncond"};
  for (int r = 0; r < 2; ++r) {
    const std::string n = names[r];
    const Stats floor = compare(load("out_" + n + "_noref").to_vector<float>(), download_f32(without[r]));
    const Stats got = compare(load("out_" + n).to_vector<float>(), download_f32(with[r]));
    std::printf("[edit parity q8] %-6s velocity: with references cosine %.6f rel L2 %.4f; no-reference floor cosine %.6f "
                "rel L2 %.4f\n",
                n.c_str(), got.cosine, got.rel_l2, floor.cosine, floor.rel_l2);
    EXPECT_LE(got.rel_l2, 1.5 * floor.rel_l2 + 1e-3) << n;
    EXPECT_GE(got.cosine, floor.cosine - 1e-3) << n;
  }

  model_->set_lora(nullptr);
  DevTensor b0(mi::DType::kF32, {128, 74, 55}), b1(mi::DType::kF32, {128, 74, 55}), a0(mi::DType::kF32, {128, 74, 55});
  const Tensor bo[] = {b0.t, b1.t};
  model_->set_references(&refs);
  model_->forward(x->t, 1000.0f, rows, bo, stream());
  model_->forward(x->t, 1000.0f, std::span<const md::Row>(rows, 1), std::span<const Tensor>(&a0.t, 1), stream());
  model_->set_references(nullptr);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(download_f32(b0.t), download_f32(a0.t)) << "q8: the cond row depends on the uncond row of its forward";
}

// The stack queries the planner prices (workspace_bytes with reference tokens, prefill_workspace_bytes) are what a
// forward with references and a prefill take: not over, and tight (one alignment unit per allocation).
TEST_F(EditDit, StackBytesAreWhatTheCallsTake) {
  const auto ref = upload_f32(load("ref_latent_in"), {128, 32, 24});
  const auto x = upload_f32(load("x"), {128, 74, 55});
  const auto cond = refine("ctx_cond"), uncond = refine("ctx_uncond");
  DevTensor cache(mi::DType::kBF16, {md::kLayers, kRefTokens, 2 * md::kDim});
  const md::RefKv refs{cache.t};
  const Tensor refs_in[] = {ref->t};
  const std::size_t prefill_need = model_->prefill_stack_bytes(kRefTokens);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  arena_->reset_high_water();
  std::size_t before = arena_->used();
  model_->prefill_references(refs_in, refs, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  std::size_t took = arena_->high_water() - before;
  std::printf("[edit] prefill took %zu stack bytes, prefill_stack_bytes %zu\n", took, prefill_need);
  EXPECT_LE(took, prefill_need);
  EXPECT_GE(took + 64 * mi::runtime::kArenaAlign, prefill_need);

  const std::int64_t lengths[] = {cond->t.dim(0), uncond->t.dim(0)};
  md::SkipSet s22;
  s22.set(22);
  const md::Row rows[] = {{cond->t, md::SkipSet{}}, {uncond->t, s22}};
  DevTensor o0(mi::DType::kF32, {128, 74, 55}), o1(mi::DType::kF32, {128, 74, 55});
  const Tensor outs[] = {o0.t, o1.t};
  const std::size_t plain = model_->forward_stack_bytes(74, 55, lengths);
  model_->set_references(&refs);
  const std::size_t need = model_->forward_stack_bytes(74, 55, lengths);
  EXPECT_GT(need, plain);
  arena_->reset_high_water();
  before = arena_->used();
  model_->forward(x->t, 1000.0f, rows, outs, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  model_->set_references(nullptr);
  took = arena_->high_water() - before;
  std::printf("[edit] forward with references took %zu stack bytes, forward_stack_bytes %zu (%zu without)\n", took, need,
              plain);
  EXPECT_LE(took, need);
  EXPECT_GE(took + 64 * mi::runtime::kArenaAlign, need);
  EXPECT_EQ(md::MossDit::reference_cache_bytes(kRefTokens), std::size_t{12288} * 42 * kRefTokens);
}

TEST_F(EditDit, BitExactInvariants) {
  const auto ref = upload_f32(load("ref_latent_in"), {128, 32, 24});
  const auto x = upload_f32(load("x"), {128, 74, 55});
  const auto cond = refine("ctx_cond"), uncond = refine("ctx_uncond");
  DevTensor cache(mi::DType::kBF16, {md::kLayers, kRefTokens, 2 * md::kDim});
  DevTensor op_cache(mi::DType::kBF16, {md::kLayers, kRefTokens, 2 * md::kDim});
  const md::RefKv refs{cache.t}, op_refs{op_cache.t};
  const Tensor refs_in[] = {ref->t};
  model_->prefill_references(refs_in, refs, stream());
  model_->set_block_path(md::BlockPath::kOpByOp);
  model_->prefill_references(refs_in, op_refs, stream());
  model_->set_block_path(md::BlockPath::kFused);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(download_u16(cache.t), download_u16(op_cache.t)) << "op-by-op prefill differs from the fused one";

  // The row invariants on the plain bf16 blocks (the LoRA side-branch GEMMs are not batch-invariant: their bits depend
  // on the run's token count with or without references).
  model_->set_lora(nullptr);
  struct RestoreLora {
    ~RestoreLora() { model_->set_lora(sb_.get()); }
  } restore;
  md::SkipSet s22;
  s22.set(22);
  const md::Row both[] = {{cond->t, md::SkipSet{}}, {uncond->t, s22}};
  const auto out = [&] { return std::make_unique<DevTensor>(mi::DType::kF32, std::initializer_list<std::int64_t>{128, 74, 55}); };
  auto b0 = out(), b1 = out(), a0 = out(), n0 = out(), n0b = out();
  model_->forward(x->t, 1000.0f, std::span<const md::Row>(both, 1), std::span<const Tensor>(&n0->t, 1), stream());
  model_->set_references(&refs);
  const Tensor bo[] = {b0->t, b1->t};
  model_->forward(x->t, 1000.0f, both, bo, stream());
  model_->forward(x->t, 1000.0f, std::span<const md::Row>(both, 1), std::span<const Tensor>(&a0->t, 1), stream());
  model_->set_references(nullptr);
  model_->forward(x->t, 1000.0f, std::span<const md::Row>(both, 1), std::span<const Tensor>(&n0b->t, 1), stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(download_f32(b0->t), download_f32(a0->t)) << "the cond row depends on the uncond row of its forward";
  EXPECT_EQ(download_f32(n0->t), download_f32(n0b->t)) << "set_references(nullptr) does not restore the plain forward";
  EXPECT_NE(download_f32(n0->t), download_f32(a0->t));

  // The op-by-op block path with references (its own slots and attention call): the fused path's bits, both rows.
  {
    auto p0 = out(), p1 = out();
    const Tensor po[] = {p0->t, p1->t};
    model_->set_block_path(md::BlockPath::kOpByOp);
    model_->set_references(&refs);
    model_->forward(x->t, 1000.0f, both, po, stream());
    model_->set_references(nullptr);
    model_->set_block_path(md::BlockPath::kFused);
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    EXPECT_EQ(download_f32(p0->t), download_f32(b0->t)) << "op-by-op forward with references differs from the fused one (cond)";
    EXPECT_EQ(download_f32(p1->t), download_f32(b1->t)) << "op-by-op forward with references differs from the fused one (uncond)";
  }

  // Refusals: 3 rows, a latent batch of 2, a malformed cache.
  model_->set_references(&refs);
  const md::Row three[] = {both[0], both[1], both[0]};
  const Tensor three_out[] = {b0->t, b1->t, a0->t};
  EXPECT_THROW(model_->forward(x->t, 1000.0f, three, three_out, stream()), std::invalid_argument);
  model_->set_references(nullptr);
  const md::RefKv bad{cache.t.slice0(0, 41)};
  EXPECT_THROW(model_->set_references(&bad), std::invalid_argument);
}

// One engine, requests alternating without and with references (INT8 DiT, the edit adapter on the side branch, the
// default): the no-reference ones stay bit-identical to the first, the reference ones to each other, the DiT is loaded
// once (a reference changes no load key), a cancel during a reference pass leaves nothing behind, a hires fix computes
// the references once per pass, and repeated requests take the same arena; an explicit merge reloads on every change.
TEST(GpuEditPipeline, AlternatingRequests) {
  const fs::path com = "/workspace/Models/community";
  const fs::path dit = com / "comfy-org/moss_image_sft_int8_convrot.safetensors";
  const fs::path te = com / "comfy-org/qwen3.5_9b_int8_convrot.safetensors";
  const fs::path lora = com / "chinoll/Moss-Image-Edit/adapter_model.safetensors";
  const fs::path bundle = "/workspace/Models/MOSS-Image/sft";
  if (!fs::is_regular_file(dit) || !fs::is_regular_file(te) || !fs::is_regular_file(lora) || !fs::is_directory(bundle)) {
    GTEST_SKIP() << "the INT8 SFT DiT, the INT8 text encoder, the edit adapter or the SFT bundle not present";
  }
  mi::pipeline::PipelineOptions o;
  o.hardware = mi::planner::HardwareProfile{};
  o.hardware->h2d_gbps = 24;
  o.hardware->disk_gbps = 3;
  o.hardware->host_gbps = 18;
  o.hardware->bf16_tflops = 150;
  o.hardware->measured = true;
  o.te_host_pin = mi::planner::TeHostPin::kOff;
  o.tier = mi::planner::Tier::kHigh;
  o.attention = mi::planner::AttentionMode::kQ8Fp8;
  o.vram_budget = std::size_t{9} << 30;
  o.host_ram_budget = std::size_t{8} << 30;
  o.dit_file = dit;
  o.text_encoder_file = te;
  mi::pipeline::Pipeline p(o);

  auto ref_img = std::make_shared<mi::pipeline::Image>();
  ref_img->width = 300;
  ref_img->height = 200;
  ref_img->channels = 3;
  ref_img->pixels.resize(300 * 200 * 3);
  for (std::size_t i = 0; i < ref_img->pixels.size(); ++i) {
    ref_img->pixels[i] = static_cast<std::uint8_t>(127.5 + 120 * std::sin(0.013 * static_cast<double>(i)));
  }
  const auto request = [&](bool refs, bool with_lora) {
    mi::pipeline::Request r;
    r.model = bundle;
    r.prompt = "make the sky green";
    r.negative_prompt = "";
    r.width = r.height = 512;
    r.steps = 3;
    r.cfg = 4.0;
    r.seed = 3;
    r.decode = false;
    if (with_lora) r.loras.push_back({lora, 1.0f});
    if (refs) r.references.images.push_back(ref_img);
    return r;
  };
  struct Out {
    std::vector<float> latents;
    int dit_loads = 0, prefills = 0;
    std::size_t high_water = 0;
    bool cancelled = false;
  };
  const auto run = [&](const mi::pipeline::Request& r, const mi::pipeline::GenerateControl& c = {}) {
    const mi::pipeline::Result res = p.generate(r, c);
    return Out{res.latents, res.memory.dit_loads, res.references.prefills, res.memory.arena_high_water, res.cancelled};
  };
  const Out a1 = run(request(false, true));
  const Out b1 = run(request(true, true));
  const Out a2 = run(request(false, true));
  const Out b2 = run(request(true, true));
  EXPECT_EQ(a1.dit_loads, 1);
  for (const Out* o2 : {&b1, &a2, &b2}) EXPECT_EQ(o2->dit_loads, 0);
  EXPECT_EQ(a1.prefills, 0);
  EXPECT_EQ(b1.prefills, 1);
  EXPECT_EQ(a1.latents, a2.latents) << "a request without references changed after one with them";
  EXPECT_EQ(b1.latents, b2.latents);
  EXPECT_NE(a1.latents, b1.latents);
  EXPECT_EQ(b1.high_water, b2.high_water);
  // A cancel inside a reference pass (after its first step), then a plain request: unchanged.
  mi::pipeline::GenerateControl stop;
  stop.on_step = [](const mi::pipeline::StepInfo& s) { return s.step < 1; };
  EXPECT_TRUE(run(request(true, true), stop).cancelled);
  const Out a3 = run(request(false, true));
  EXPECT_EQ(a3.latents, a1.latents);
  // The same arena as the plain request before (a1 ran the text encoder and had no VAE encoder resident yet).
  EXPECT_EQ(a3.high_water, a2.high_water);
  // Without the adapter, then with it again.
  const Out c1 = run(request(false, false));
  EXPECT_EQ(run(request(true, true)).latents, b1.latents);
  EXPECT_EQ(run(request(false, false)).latents, c1.latents);
  // A hires fix: the references computed once per pass.
  mi::pipeline::Request h = request(true, true);
  h.hires.width = h.hires.height = 640;
  h.hires.steps = 2;
  EXPECT_EQ(run(h).prefills, 2);
  {
    // A cancel inside the second pass's prefill: the result's latents are that pass's starting latents (its size), no
    // step of it ran.
    std::atomic<bool> stop_flag{false};
    mi::weights::testing::cancel_at_reference_prefill(&stop_flag, 2);
    mi::pipeline::GenerateControl c;
    c.cancel = &stop_flag;
    const mi::pipeline::Result res = p.generate(h, c);
    mi::weights::testing::cancel_at_reference_prefill(nullptr, 0);
    EXPECT_TRUE(res.cancelled);
    EXPECT_EQ(res.references.prefills, 1);
    EXPECT_EQ(res.first_pass.steps_run, 3);
    EXPECT_TRUE(res.hires.ran);
    EXPECT_EQ(res.hires.steps_run, 0);
    EXPECT_EQ(res.width, 640);
    EXPECT_EQ(res.latents.size(), std::size_t{128} * (640 / 16) * (640 / 16));
  }
  // An explicit merge on INT8 (no pristine copies): it reloads the DiT, a strength change reloads it again (no in-place
  // re-merge), and so does going back to the default placement, whose bits are the side branch's again.
  mi::pipeline::Request mq = request(true, true);
  mq.lora_placement = mi::lora::Placement::kMerge;
  const Out m1 = run(mq);
  EXPECT_EQ(m1.dit_loads, 1);
  EXPECT_NE(m1.latents, b1.latents);
  mq.loras[0].strength = 0.5f;
  const Out m2 = run(mq);
  EXPECT_EQ(m2.dit_loads, 1);
  EXPECT_NE(m2.latents, m1.latents);
  const Out back = run(request(true, true));
  EXPECT_EQ(back.dit_loads, 1);
  EXPECT_EQ(back.latents, b1.latents);
}

// Merged streamed blocks need a pinned host copy: over the host budget the plan fails (out of memory) naming the side
// branch, a larger host_ram_budget and a baked file, and the same request on the side branch plans and runs from the
// file; a GPU shortfall with the same merges is not blamed on them.
TEST(GpuEditPipeline, PlanFailureNamesThePlacement) {
  const fs::path com = "/workspace/Models/community";
  const fs::path dit = com / "comfy-org/moss_image_sft_int8_convrot.safetensors";
  const fs::path te = com / "comfy-org/qwen3.5_9b_int8_convrot.safetensors";
  const fs::path lora = com / "chinoll/Moss-Image-Edit/adapter_model.safetensors";
  const fs::path bundle = "/workspace/Models/MOSS-Image/sft";
  if (!fs::is_regular_file(dit) || !fs::is_regular_file(te) || !fs::is_regular_file(lora) || !fs::is_directory(bundle)) {
    GTEST_SKIP() << "the INT8 SFT DiT, the INT8 text encoder, the edit adapter or the SFT bundle not present";
  }
  const auto options = [&](std::size_t vram, std::size_t host) {
    mi::pipeline::PipelineOptions o;
    o.hardware = mi::planner::HardwareProfile{};
    o.hardware->h2d_gbps = 24;
    o.hardware->disk_gbps = 3;
    o.hardware->host_gbps = 18;
    o.hardware->bf16_tflops = 150;
    o.hardware->measured = true;
    o.te_host_pin = mi::planner::TeHostPin::kOff;
    o.tier = mi::planner::Tier::kHigh;
    o.attention = mi::planner::AttentionMode::kQ8Fp8;
    o.mode = mi::planner::Mode::kMinMemory;
    o.vram_budget = vram;
    o.host_ram_budget = host;
    o.dit_file = dit;
    o.text_encoder_file = te;
    return o;
  };
  const auto request = [&](mi::lora::Placement placement) {
    mi::pipeline::Request r;
    r.model = bundle;
    r.prompt = "make the sky green";
    r.negative_prompt = "";
    r.width = r.height = 512;
    r.steps = 1;
    r.cfg = 4.0;
    r.seed = 3;
    r.decode = false;
    r.loras.push_back({lora, 1.0f});
    r.lora_placement = placement;
    return r;
  };
  const auto plan_error = [&](mi::pipeline::Pipeline& p, const mi::pipeline::Request& r) {
    try {
      p.generate(r);
    } catch (const mi::planner::BudgetError& e) {
      return std::string(e.what());
    }
    return std::string("planned");
  };
  {
    // 3 GiB of VRAM streams (nearly) every block; the host holds the generation from the files but not the merged
    // streamed blocks' pinned copy (~130 MB each).
    mi::pipeline::Pipeline p(options(std::size_t{3} << 30,
                                     mi::planner::process_host_memory().private_bytes + (std::size_t{3} << 30)));
    const std::string e = plan_error(p, request(mi::lora::Placement::kMerge));
    EXPECT_NE(e.find("not enough host memory"), std::string::npos) << e;
    EXPECT_NE(e.find("use lora_placement side_branch, a larger host_ram_budget, or a DiT file with the adapter baked in"),
              std::string::npos)
        << e;
    const mi::pipeline::Result res = p.generate(request(mi::lora::Placement::kSideBranch));
    EXPECT_EQ(res.steps_run, 1);
    EXPECT_EQ(res.dit_stream.source, "file");
  }
  {
    // Too little GPU memory for any plan: the merges are not what failed.
    mi::pipeline::Pipeline p(options(std::size_t{1} << 30, 0));
    const std::string e = plan_error(p, request(mi::lora::Placement::kMerge));
    EXPECT_NE(e.find("not enough GPU memory"), std::string::npos) << e;
    EXPECT_EQ(e.find("lora_placement"), std::string::npos) << e;
  }
}

// A DiT baked with the edit adapter (quant::convert with it as a LoRA, `moss-infer quantize --lora`) is edit-capable:
// its metadata records the adapter, so reference images run on it without the no-adapter warning, the report and the
// file's remarks (the plan explanation, inspect's note) name it, and the same adapter passed again as a runtime LoRA is
// warned about (applied twice; refused with lora_strict). An ordinary INT8 file with references still warns.
TEST(GpuEditPipeline, BakedEditAdapter) {
  const fs::path com = "/workspace/Models/community";
  const fs::path bf16 = com / "comfy-org/moss_image_sft_bf16.safetensors";
  const fs::path int8 = com / "comfy-org/moss_image_sft_int8_convrot.safetensors";
  const fs::path te = com / "comfy-org/qwen3.5_9b_int8_convrot.safetensors";
  const fs::path lora = com / "chinoll/Moss-Image-Edit/adapter_model.safetensors";
  const fs::path bundle = "/workspace/Models/MOSS-Image/sft";
  if (!fs::is_regular_file(bf16) || !fs::is_regular_file(int8) || !fs::is_regular_file(te) || !fs::is_regular_file(lora) ||
      !fs::is_directory(bundle)) {
    GTEST_SKIP() << "the bf16 and INT8 SFT DiTs, the INT8 text encoder, the edit adapter or the SFT bundle not present";
  }
  mi::testing::TempDir dir("baked_edit");
  const fs::path baked = dir.path() / "sft_int8_edit.safetensors";
  mi::quant::ConvertOptions co;
  co.loras = {{lora, 1.0f}};
  mi::quant::convert(bf16, baked, co);
  {
    const mi::weights::SafetensorsFile f(baked);
    EXPECT_NE(f.metadata().at("mossinfer_lora").find("\"format\": \"moss-edit-lora\""), std::string::npos);
    const std::string named = mi::pipeline::baked_edit_adapters(f.metadata());
    EXPECT_EQ(named.rfind("adapter_model.safetensors (strength 1, xxh64 ", 0), 0u) << named;
    const mi::pipeline::DitFileInfo info = mi::pipeline::inspect_dit_file(baked, std::function<std::string()>());
    EXPECT_EQ(info.baked_edit, named);
    EXPECT_NE(info.remarks.find("edit adapter baked in (moss-edit-lora): " + named), std::string::npos) << info.remarks;
    EXPECT_EQ(mi::pipeline::inspect_dit_file(int8, std::function<std::string()>()).baked_edit, "");
  }

  auto ref_img = std::make_shared<mi::pipeline::Image>();
  ref_img->width = ref_img->height = 256;
  ref_img->channels = 3;
  ref_img->pixels.resize(256 * 256 * 3);
  for (std::size_t i = 0; i < ref_img->pixels.size(); ++i) {
    ref_img->pixels[i] = static_cast<std::uint8_t>(127.5 + 120 * std::sin(0.013 * static_cast<double>(i)));
  }
  const auto options = [&](const fs::path& dit, bool strict) {
    mi::pipeline::PipelineOptions o;
    o.te_host_pin = mi::planner::TeHostPin::kOff;
    o.tier = mi::planner::Tier::kHigh;
    o.attention = mi::planner::AttentionMode::kQ8Fp16;  // runs on every supported GPU
    o.dit_file = dit;
    o.text_encoder_file = te;
    o.embedding_cache = false;
    o.lora_strict = strict;
    return o;
  };
  const auto request = [&](bool with_lora) {
    mi::pipeline::Request r;
    r.model = bundle;
    r.prompt = "make the sky green";
    r.negative_prompt = "";
    r.width = r.height = 256;
    r.steps = 1;
    r.cfg = 4.0;
    r.seed = 3;
    r.decode = false;
    r.references.images.push_back(ref_img);
    if (with_lora) r.loras.push_back({lora, 1.0f});
    return r;
  };
  const auto twice = [](const std::vector<std::string>& warnings) {
    return std::count_if(warnings.begin(), warnings.end(),
                         [](const std::string& w) { return w.find("the edit applies twice") != std::string::npos; });
  };
  {
    mi::pipeline::Pipeline p(options(baked, false));
    const mi::pipeline::Request req = request(false);
    const mi::pipeline::Result r = p.generate(req);
    EXPECT_EQ(r.steps_run, 1);
    EXPECT_EQ(r.references.count, 1);
    EXPECT_TRUE(r.reference_warnings.empty()) << r.reference_warnings.front();
    EXPECT_EQ(twice(r.lora_warnings), 0);
    EXPECT_NE(r.dit_baked_edit.find("adapter_model.safetensors"), std::string::npos);
    EXPECT_NE(r.plan.explanation.find("edit adapter baked in (moss-edit-lora)"), std::string::npos);
    const mi::json::Value doc = mi::json::parse("{" + mi::pipeline::report_fields(req, r, "", "") + "}");
    EXPECT_EQ(doc.at("models").at("dit").at("baked_edit_adapter").as_string(), r.dit_baked_edit);

    // The same adapter again at run time: warned once, references still without the no-adapter warning.
    const mi::pipeline::Result again = p.generate(request(true));
    EXPECT_EQ(twice(again.lora_warnings), 1) << ::testing::PrintToString(again.lora_warnings);
    EXPECT_TRUE(again.reference_warnings.empty());
  }
  {
    mi::pipeline::Pipeline strict(options(baked, true));
    try {
      strict.generate(request(true));
      ADD_FAILURE() << "an edit adapter on a baked DiT ran under lora_strict";
    } catch (const mi::lora::LoraError& e) {
      EXPECT_NE(std::string(e.what()).find("the edit applies twice"), std::string::npos) << e.what();
    }
  }
  {
    // An ordinary INT8 file with references and no adapter: the warning stays; its report has no baked adapter.
    mi::pipeline::Pipeline p(options(int8, false));
    const mi::pipeline::Request req = request(false);
    const mi::pipeline::Result r = p.generate(req);
    ASSERT_EQ(r.reference_warnings.size(), 1u);
    EXPECT_NE(r.reference_warnings[0].find("without a moss-edit-lora adapter"), std::string::npos);
    EXPECT_EQ(r.dit_baked_edit, "");
    const mi::json::Value doc = mi::json::parse("{" + mi::pipeline::report_fields(req, r, "", "") + "}");
    EXPECT_EQ(doc.at("models").at("dit").find("baked_edit_adapter"), nullptr);
  }
}

}  // namespace
