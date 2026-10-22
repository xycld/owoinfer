// P6 step 41 (lane T): distill bundles (models/moss_dit/distill.h) and the DiT's fp32 replacement step embedder
// (MossDit::set_time_embedding), on lodestones' MOSS-Image-SFT-8-steps-dmd files and the official SFT bundle:
//   - the bundle's files are found; a directory without them is refused;
//   - load_time_embedding returns the file's values unrounded, and refuses files that are not a replacement;
//   - time_modulation with the replacement equals an fp32 host evaluation of the published math (each mod / head value
//     within 1 bf16 ULP; the device sums in another order), differs from the model's own, and removing the
//     replacement restores the model's bits exactly;
//   - generations complete (512^2): the lodestones bundle (step embedder + LoRA + data-space sampler, 8 steps, CFG 1),
//     akhaliq's 4-step LoRA (stock euler, 4 steps, CFG 1) and the INT4 Turbo tier.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#include "core/cuda_check.h"
#include "models/moss_dit/distill.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "test_lora_util.h"
#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/safetensors.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
namespace md = mi::moss_dit;
namespace tl = mi::testing::lora;
using mi::DType;
using mi::Tensor;

fs::path env_path(const char* var, const char* fallback) {
  const char* env = std::getenv(var);
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path(fallback);
}
fs::path bundle_dir() { return env_path("MOSS_DISTILL_BUNDLE", "/workspace/Models/community/lodestones/MOSS-Image-SFT-8-steps-dmd"); }
fs::path akhaliq_lora() { return env_path("MOSS_AKHALIQ_LORA", "/workspace/Models/community/akhaliq/MOSS-Image-SFT-4-steps/lora.safetensors"); }
fs::path int4_file() { return env_path("MOSS_INT4_DIT", "/workspace/Models/community/toxicdog/moss_image_sft_int4_convrot.safetensors"); }
fs::path sft_dir() { return env_path("MOSS_WEIGHTS", "/workspace/Models/MOSS-Image") / "sft"; }

float from_bf16(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }
int ordered(std::uint16_t b) { return (b & 0x8000) ? -static_cast<int>(b & 0x7fff) : static_cast<int>(b); }

std::vector<float> read_f32(const fs::path& file, const char* name) {
  const mi::weights::SafetensorsFile st(file);
  const mi::weights::TensorInfo* t = st.find(name);
  if (t == nullptr || t->dtype != mi::weights::DType::kF32) return {};
  std::vector<float> v(static_cast<std::size_t>(t->nbytes / 4));
  mi::weights::FileReader(file).read_at(t->offset, v.data(), t->nbytes);
  return v;
}

TEST(DitDistill, BundleFiles) {
  const fs::path tmp = fs::temp_directory_path() / "mossinfer_test_dit_distill_empty";
  fs::create_directories(tmp);
  EXPECT_THROW(md::open_distill_bundle(tmp), md::DistillError);
  EXPECT_THROW(md::open_distill_bundle(tmp / "missing"), md::DistillError);
  fs::remove_all(tmp);
  if (!fs::is_directory(bundle_dir())) GTEST_SKIP() << "absent: " << bundle_dir();
  const md::DistillBundle b = md::open_distill_bundle(bundle_dir());
  EXPECT_EQ(b.time_embedding, bundle_dir() / "time_embedding.safetensors");
  EXPECT_EQ(b.lora, bundle_dir() / "lora.safetensors");
  EXPECT_EQ(b.dataspace, bundle_dir() / "dataspace.safetensors");
}

TEST(DitDistill, LoadTimeEmbedding) {
  const fs::path file = bundle_dir() / "time_embedding.safetensors";
  if (!fs::is_regular_file(file)) GTEST_SKIP() << "absent: " << file;
  const md::TimeEmbeddingFp32 te = md::load_time_embedding(file);
  // Every value as stored (F32, unrounded).
  EXPECT_EQ(te.w0, read_f32(file, "time_embedding.0.weight"));
  EXPECT_EQ(te.b0, read_f32(file, "time_embedding.0.bias"));
  EXPECT_EQ(te.w2, read_f32(file, "time_embedding.2.weight"));
  EXPECT_EQ(te.b2, read_f32(file, "time_embedding.2.bias"));
  EXPECT_EQ(te.mod_bias, read_f32(file, "shared_block_modulation.bias"));
  EXPECT_EQ(te.head_bias, read_f32(file, "video_head_modulation.bias"));
  ASSERT_EQ(te.mod_bias.size(), 6u * 3072u);
  // Something that is not a replacement: refused, naming the tensor.
  const fs::path bad = fs::temp_directory_path() / "mossinfer_test_dit_distill_bad.safetensors";
  tl::write_st(bad, {{"diffusion_model.layers.0.proj.q.lora_down.weight", mi::weights::DType::kF32, {2, 4}, std::vector<float>(8, 0.0f)}});
  try {
    md::load_time_embedding(bad);
    ADD_FAILURE() << "a LoRA file was read as a time embedding";
  } catch (const md::DistillError& e) {
    EXPECT_NE(std::string(e.what()).find("lora_down"), std::string::npos) << e.what();
  }
  tl::write_st(bad, {{"time_embedding.0.bias", mi::weights::DType::kF32, {3072}, std::vector<float>(3072, 0.0f)}});
  EXPECT_THROW(md::load_time_embedding(bad), md::DistillError);  // the rest missing
  const auto refused = [&](const std::vector<tl::StTensor>& ts, const char* what, const char* says) {
    tl::write_st(bad, ts);
    try {
      md::load_time_embedding(bad);
      ADD_FAILURE() << what << " accepted";
    } catch (const md::DistillError& e) {
      EXPECT_NE(std::string(e.what()).find(says), std::string::npos) << what << ": " << e.what();
    }
  };
  std::vector<float> nan_bias(3072, 0.5f);
  nan_bias[17] = std::nanf("");
  refused({{"time_embedding.0.bias", mi::weights::DType::kF32, {3071}, std::vector<float>(3071, 0.0f)}}, "a wrong shape",
          "unexpected shape");
  refused({{"time_embedding.0.bias", mi::weights::DType::kI8, {3072}, std::vector<float>(3072, 0.0f)}}, "an I8 tensor",
          "must be F32, BF16 or F16");
  refused({{"time_embedding.0.bias", mi::weights::DType::kF32, {3072}, nan_bias}}, "a NaN", "non-finite");
  refused({{"time_embedding.0.bias", mi::weights::DType::kF32, {3072}, std::vector<float>(3072, 0.0f)},
           {"diffusion_model.time_embedding.0.bias", mi::weights::DType::kF32, {3072}, std::vector<float>(3072, 0.0f)}},
          "a duplicate after prefix stripping", "appears twice");
  fs::remove(bad);
}

TEST(DitDistill, ReplacementStepEmbedderIsApplied) {
  const fs::path file = bundle_dir() / "time_embedding.safetensors";
  if (!fs::is_regular_file(file) || !fs::is_regular_file(sft_dir() / "moss.mossidx")) GTEST_SKIP() << "files absent";
  const md::TimeEmbeddingFp32 te = md::load_time_embedding(file);
  const mi::weights::ModelIndex index = mi::weights::load_index(sft_dir() / "moss.mossidx", sft_dir());
  mi::runtime::StreamSet streams;
  md::Residency res;
  res.resident_blocks = 0;  // only the globals matter here
  mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(index, res) + (std::size_t{64} << 20));
  md::MossDit dit(index, arena, streams, false, nullptr, res);
  const cudaStream_t s = streams.compute();
  mi::DeviceBuffer mod_buf(6 * 3072 * 2), head_buf(2 * 3072 * 2);
  const Tensor mod = Tensor::view(mod_buf.get(), DType::kBF16, mi::Device::kCUDA, {6 * 3072});
  const Tensor head = Tensor::view(head_buf.get(), DType::kBF16, mi::Device::kCUDA, {2 * 3072});
  const auto run = [&](float t) {
    dit.time_modulation(t, mod, head, s);
    MI_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<std::uint16_t> out(8 * 3072);
    MI_CUDA_CHECK(cudaMemcpy(out.data(), mod.data, 6 * 3072 * 2, cudaMemcpyDeviceToHost));
    MI_CUDA_CHECK(cudaMemcpy(out.data() + 6 * 3072, head.data, 2 * 3072 * 2, cudaMemcpyDeviceToHost));
    return out;
  };
  // The modulation weights as the model holds them (bf16), for the host evaluation.
  mi::weights::ReaderCache readers(index, false);
  const mi::weights::HostGroup glob = mi::weights::load_group_to_host(
      readers, *index.find_group(mi::weights::Component::kTransformer, "globals"), mi::weights::HostMemory::kPlain);
  const auto* wm = static_cast<const std::uint16_t*>(glob.tensors.at("shared_block_modulation.weight").data);
  const auto* wh = static_cast<const std::uint16_t*>(glob.tensors.at("video_head_modulation.weight").data);
  for (const float t : {1000.0f, 875.25f, 31.0f}) {
    const auto own = run(t);
    dit.set_time_embedding(&te);
    EXPECT_TRUE(dit.has_time_embedding());
    const auto got = run(t);
    const auto again = run(t);  // the cached embedder output
    EXPECT_EQ(again, got);
    const std::vector<float> time = md::time_vector_fp32(te, t);
    int max_ulp = 0;
    std::size_t diff_own = 0;
    for (std::size_t r = 0; r < 8 * 3072; ++r) {
      const bool is_mod = r < 6 * 3072;
      const std::uint16_t* w = is_mod ? wm + r * 3072 : wh + (r - 6 * 3072) * 3072;
      double acc = is_mod ? te.mod_bias[r] : te.head_bias[r - 6 * 3072];
      for (std::size_t c = 0; c < 3072; ++c) acc += static_cast<double>(from_bf16(w[c])) * time[c];
      const auto want_f = static_cast<float>(acc);
      std::uint32_t u = std::bit_cast<std::uint32_t>(want_f);
      u += 0x7fffu + ((u >> 16) & 1u);
      const auto want = static_cast<std::uint16_t>(u >> 16);
      max_ulp = std::max(max_ulp, std::abs(ordered(want) - ordered(got[r])));
      diff_own += got[r] != own[r];
    }
    std::printf("[distill] t=%.2f: fp32 replacement vs host fp32 evaluation: max %d bf16 ULP; %zu of %d values differ from "
                "the model's own embedder\n", static_cast<double>(t), max_ulp, diff_own, 8 * 3072);
    EXPECT_LE(max_ulp, 1);
    EXPECT_GT(diff_own, std::size_t{8 * 3072 / 2});
    dit.set_time_embedding(nullptr);
    EXPECT_EQ(run(t), own);
  }
}

// Per-channel standard deviation of an RGB8 image, averaged: a blank or constant image is near 0.
double rgb_std(const std::vector<std::uint8_t>& rgb) {
  double total = 0;
  for (int c = 0; c < 3; ++c) {
    double sum = 0, sq = 0;
    const std::size_t n = rgb.size() / 3;
    for (std::size_t i = 0; i < n; ++i) {
      const double v = rgb[3 * i + static_cast<std::size_t>(c)];
      sum += v;
      sq += v * v;
    }
    total += std::sqrt(std::max(0.0, sq / static_cast<double>(n) - (sum / static_cast<double>(n)) * (sum / static_cast<double>(n))));
  }
  return total / 3;
}

TEST(DitDistillPipeline, GenerationsComplete) {
  if (!fs::is_directory(bundle_dir()) || !fs::is_regular_file(akhaliq_lora()) || !fs::is_regular_file(int4_file()) ||
      !fs::is_regular_file(sft_dir() / "moss.mossidx")) {
    GTEST_SKIP() << "distill bundle, akhaliq LoRA, INT4 DiT or SFT bundle absent";
  }
  mi::pipeline::Request base;
  base.model = sft_dir();
  base.prompt = "A lighthouse on a rocky coast under a stormy sky, oil painting";
  base.width = base.height = 512;
  base.seed = 3;
  const auto check = [](const mi::pipeline::Result& r, int steps, const char* what) {
    EXPECT_EQ(r.steps_run, steps) << what;
    EXPECT_EQ(r.rgb.size(), std::size_t{512} * 512 * 3) << what;
    const double sd = rgb_std(r.rgb);
    std::printf("[distill] %s: %d steps in %.2f s (denoise %.2f s), RGB std %.1f; %s\n", what, r.steps_run, r.timings.total_s,
                r.timings.denoise_s, sd, r.tier.reason.c_str());
    EXPECT_GT(sd, 10.0) << what;
  };
  {
    // lodestones: every bundle file, as --distill applies them.
    const md::DistillBundle b = md::open_distill_bundle(bundle_dir());
    mi::pipeline::PipelineOptions o;
    o.time_embedding_file = b.time_embedding;
    mi::pipeline::Pipeline p(o);
    mi::pipeline::Request q = base;
    q.cfg = 1.0;
    q.steps = 8;
    q.sampler = mi::pipeline::SamplerKind::kDataSpace;
    q.dataspace_path = b.dataspace;
    q.loras = {{b.lora, 1.0f}};
    const mi::pipeline::Result r = p.generate(q);
    check(r, 8, "lodestones 8-step DMD bundle");
    EXPECT_NE(r.plan.explanation.find("step embedder replaced"), std::string::npos) << r.plan.explanation;
    ASSERT_EQ(r.loras.size(), 1u);
    // Recorded for reproducibility: the report and the PNG metadata name the replacement.
    EXPECT_EQ(r.time_embedding, mi::util::path_utf8(b.time_embedding));
    const mi::pipeline::GenerationMetadata md = mi::pipeline::generation_metadata(q, r);
    bool in_png = false;
    for (const auto& [k, v] : md.extra) in_png = in_png || (k == "time_embedding" && v == "time_embedding.safetensors");
    EXPECT_TRUE(in_png);
    EXPECT_NE(mi::pipeline::report_fields(q, r, "x.png", "").find("\"time_embedding\": \"" + r.time_embedding + "\""),
              std::string::npos);
    EXPECT_GT(r.cond_tokens, 0);
    EXPECT_EQ(r.uncond_tokens, 0);  // CFG 1: no negative prompt
  }
  {
    // akhaliq's 4-step LoRA on the stock schedule (euler, the moss scheduler), CFG 1.
    mi::pipeline::Pipeline p;
    mi::pipeline::Request q = base;
    q.cfg = 1.0;
    q.steps = 4;
    q.loras = {{akhaliq_lora(), 1.0f}};
    check(p.generate(q), 4, "akhaliq 4-step LoRA");
  }
  {
    // The Turbo tier: the INT4 DiT, chosen explicitly (uncertified).
    mi::pipeline::PipelineOptions o;
    o.dit_file = int4_file();
    o.tier = mi::planner::Tier::kTurbo;
    mi::pipeline::Pipeline p(o);
    mi::pipeline::Request q = base;
    q.steps = 6;
    const mi::pipeline::Result r = p.generate(q);
    check(r, 6, "INT4 Turbo tier");
    EXPECT_EQ(r.tier.effective, mi::planner::Tier::kTurbo);
    EXPECT_EQ(r.tier.precision, mi::planner::Precision::kInt4);
    EXPECT_FALSE(r.tier.certified);
    EXPECT_EQ(r.dit_linears, "int4_convrot256");
    // A merged (LoHa) adapter on an INT4 linear is refused before anything loads.
    const fs::path loha = fs::temp_directory_path() / ("mossinfer_test_dit_distill_loha_" + std::to_string(std::random_device{}()) +
                                                       ".safetensors");
    const std::string m3 = "diffusion_model.layers.3.proj.o";
    tl::write_st(loha, {{m3 + ".hada_w1_a", mi::weights::DType::kBF16, {3072, 2}, tl::randn(6144, 1, 0.1f)},
                        {m3 + ".hada_w1_b", mi::weights::DType::kBF16, {2, 3072}, tl::randn(6144, 2, 0.1f)},
                        {m3 + ".hada_w2_a", mi::weights::DType::kBF16, {3072, 2}, tl::randn(6144, 3, 0.1f)},
                        {m3 + ".hada_w2_b", mi::weights::DType::kBF16, {2, 3072}, tl::randn(6144, 4, 0.1f)}});
    mi::pipeline::Request ql = q;
    ql.loras = {{loha, 1.0f}};
    EXPECT_THROW(p.generate(ql), std::invalid_argument);
    fs::remove(loha);
    EXPECT_EQ(p.generate(q).steps_run, 6);  // still usable
    // Auto does not run it.
    mi::pipeline::PipelineOptions a;
    a.dit_file = int4_file();
    mi::pipeline::Pipeline pa(a);
    EXPECT_THROW(pa.generate(q), std::invalid_argument);
  }
}

}  // namespace
