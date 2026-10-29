// LoRA stack: file formats, validation, reports and the host merges (CPU only).
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "models/lora/lora.h"
#include "test_lora_util.h"
#include "test_util.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_convert.h"

namespace mi::lora {
namespace {

using mi::testing::TempDir;
using mi::testing::lora::randn;
using mi::testing::lora::StTensor;
using mi::testing::lora::write_st;
using weights::DType;
namespace fs = std::filesystem;

constexpr std::int64_t kD = 3072, kF = 8192;

const TargetModel& model() {
  static const TargetModel m = moss_dit_targets();
  return m;
}

LoraStack open1(const fs::path& p, float strength = 1.0f, LoadOptions opt = {}) {
  const LoraSpec s{p, strength};
  return LoraStack(std::span<const LoraSpec>(&s, 1), model(), opt);
}

std::vector<StTensor> low_rank(const std::string& prefix, const std::string& down, const std::string& up,
                               std::int64_t n, std::int64_t k, std::int64_t r, std::uint32_t seed,
                               DType dt = DType::kBF16, float alpha = -1) {
  std::vector<StTensor> t{{prefix + down, dt, {r, k}, randn(static_cast<std::size_t>(r * k), seed, 0.02f)},
                          {prefix + up, dt, {n, r}, randn(static_cast<std::size_t>(n * r), seed + 1, 0.02f)}};
  if (alpha >= 0) t.push_back({prefix + ".alpha", DType::kF32, {}, {alpha}});
  return t;
}

TEST(LoraTargets, MossDitTable) {
  const TargetModel& m = model();
  EXPECT_EQ(m.blocks, 42);
  EXPECT_EQ(m.modules.size(), 42u * 6 + 1 + 2 * 5 + 2 + 2 + 2);
  const TargetModule* up = m.find("layers.41.proj.ffn_up_gate");
  ASSERT_NE(up, nullptr);
  EXPECT_EQ(up->n, 2 * kF);
  EXPECT_EQ(up->k, kD);
  EXPECT_EQ(up->block, 41);
  EXPECT_EQ(up->gemm, BlockGemm::kUpGate);
  const TargetModule* te = m.find("text_embedder.proj");
  ASSERT_NE(te, nullptr);
  EXPECT_EQ(te->k, 4096);
  EXPECT_TRUE(te->has_bias);
  EXPECT_EQ(te->block, -1);
  EXPECT_EQ(m.find("layers.42.proj.q"), nullptr);
  EXPECT_EQ(m.find("layers.0.proj.qkv"), nullptr);  // an alias, not a module
}

// Every key spelling of the formats in use, one module each, with scales and placements.
TEST(LoraFormat, KeySpellingsAndScales) {
  TempDir dir("lora_keys");
  std::vector<StTensor> t;
  for (auto& x : low_rank("lora_unet_layers_0_proj_q", ".lora_down.weight", ".lora_up.weight", kD, kD, 4, 1, DType::kF32, 2.0f)) t.push_back(x);
  for (auto& x : low_rank("diffusion_model.layers.1.proj.ffn_up_gate", ".lora_down.weight", ".lora_up.weight", 2 * kF, kD, 8, 3, DType::kBF16, 8.0f)) t.push_back(x);
  for (auto& x : low_rank("transformer.layers.2.proj.o", ".lora_A.weight", ".lora_B.weight", kD, kD, 4, 5)) t.push_back(x);
  for (auto& x : low_rank("base_model.model.layers.3.proj.ffn_down", ".lora_A.weight", ".lora_B.weight", kD, kF, 4, 7, DType::kF16)) t.push_back(x);
  for (auto& x : low_rank("layers.4.proj.v", ".lora.down.weight", ".lora.up.weight", kD, kD, 2, 9)) t.push_back(x);
  for (auto& x : low_rank("model.diffusion_model.text_embedder.blocks.1.ffn_down", ".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight", kD, kF, 2, 11)) t.push_back(x);
  for (auto& x : low_rank("lora_unet_shared_block_modulation", ".lora_down.weight", ".lora_up.weight", 6 * kD, kD, 2, 13, DType::kBF16, 1.0f)) t.push_back(x);
  write_st(dir.path() / "a.safetensors", t);
  const LoraStack s = open1(dir.path() / "a.safetensors", 0.5f);
  ASSERT_EQ(s.adapters().size(), 7u);
  const auto get = [&](const char* m) {
    const auto v = s.adapters_of(m);
    EXPECT_EQ(v.size(), 1u) << m;
    return v.empty() ? Adapter{} : *v[0];
  };
  Adapter a = get("layers.0.proj.q");
  EXPECT_EQ(a.kind, AdapterKind::kLowRank);
  EXPECT_EQ(a.rank, 4);
  EXPECT_DOUBLE_EQ(a.alpha_scale, 0.5f);
  EXPECT_TRUE(a.side_branch);
  EXPECT_EQ(a.source, "lora_unet_layers_0_proj_q");
  EXPECT_FLOAT_EQ(s.coef(a), 0.25f);
  a = get("layers.1.proj.ffn_up_gate");
  EXPECT_DOUBLE_EQ(a.alpha_scale, 1.0f);
  EXPECT_TRUE(a.side_branch);
  a = get("layers.2.proj.o");
  EXPECT_DOUBLE_EQ(a.alpha_scale, 1.0f);  // no alpha anywhere: scale 1
  EXPECT_TRUE(get("layers.3.proj.ffn_down").side_branch);
  EXPECT_TRUE(get("layers.4.proj.v").side_branch);
  a = get("text_embedder.blocks.1.ffn_down");
  EXPECT_FALSE(a.side_branch);  // outside the block GEMMs: merged
  a = get("shared_block_modulation");
  EXPECT_FALSE(a.side_branch);
  EXPECT_DOUBLE_EQ(a.alpha_scale, 0.5f);
  ASSERT_EQ(s.reports().size(), 1u);
  EXPECT_TRUE(s.reports()[0].unknown.empty());
  EXPECT_EQ(s.reports()[0].side_branch, 5);
  EXPECT_EQ(s.reports()[0].merged, 2);
  EXPECT_EQ(s.merge_modules(), (std::vector<std::string>{"shared_block_modulation", "text_embedder.blocks.1.ffn_down"}));
}

TEST(LoraFormat, PeftMetadataAlpha) {
  TempDir dir("lora_meta");
  write_st(dir.path() / "p.safetensors", low_rank("transformer.layers.0.proj.k", ".lora_A.weight", ".lora_B.weight", kD, kD, 16, 1),
           R"({"format":"pt","lora_adapter_metadata":"{\"r\": 16, \"lora_alpha\": 8, \"peft_type\": \"LORA\"}"})");
  const LoraStack s = open1(dir.path() / "p.safetensors");
  ASSERT_EQ(s.adapters().size(), 1u);
  EXPECT_DOUBLE_EQ(s.adapters()[0].alpha_scale, 0.5f);
  // A per-module alpha_pattern is refused rather than half-applied.
  write_st(dir.path() / "q.safetensors", low_rank("transformer.layers.0.proj.k", ".lora_A.weight", ".lora_B.weight", kD, kD, 16, 1),
           R"({"lora_adapter_metadata":"{\"r\": 16, \"lora_alpha\": 8, \"alpha_pattern\": {\"proj.q\": 4}}"})");
  EXPECT_THROW(open1(dir.path() / "q.safetensors"), weights::FormatError);
}

TEST(LoraFormat, FusedQkvSplitsByRows) {
  TempDir dir("lora_qkv");
  auto t = low_rank("diffusion_model.layers.5.proj.qkv", ".lora_down.weight", ".lora_up.weight", 3 * kD, kD, 4, 21);
  write_st(dir.path() / "f.safetensors", t);
  const LoraStack s = open1(dir.path() / "f.safetensors");
  ASSERT_EQ(s.adapters().size(), 3u);
  const char* names[] = {"layers.5.proj.q", "layers.5.proj.k", "layers.5.proj.v"};
  for (int i = 0; i < 3; ++i) {
    const Adapter& a = s.adapters()[static_cast<std::size_t>(i)];
    EXPECT_EQ(a.module, names[i]);
    EXPECT_EQ(a.row0, i * kD);
    EXPECT_EQ(a.n, kD);
    EXPECT_EQ(a.stored_rows, 3 * kD);
    EXPECT_TRUE(a.side_branch);
    // Its rows of the stored up matrix are the target's B.
    const std::vector<float> up = s.read(a.parts[1], a.row0, a.n);
    EXPECT_EQ(up.front(), t[1].values[static_cast<std::size_t>(i * kD * 4)]);
  }
  EXPECT_EQ(s.reports()[0].adapters, 3);
}

TEST(LoraFormat, LohaLokrFullAreMerged) {
  TempDir dir("lora_lyco");
  std::vector<StTensor> t;
  const std::string h = "diffusion_model.layers.0.proj.o";
  t.push_back({h + ".hada_w1_a", DType::kF16, {kD, 2}, randn(kD * 2, 1, 0.1f)});
  t.push_back({h + ".hada_w1_b", DType::kF16, {2, kD}, randn(kD * 2, 2, 0.1f)});
  t.push_back({h + ".hada_w2_a", DType::kF16, {kD, 3}, randn(kD * 3, 3, 0.1f)});
  t.push_back({h + ".hada_w2_b", DType::kF16, {3, kD}, randn(kD * 3, 4, 0.1f)});
  t.push_back({h + ".alpha", DType::kF16, {}, {4.0f}});
  const std::string k = "lycoris_time_embedding_0";
  t.push_back({k + ".lokr_w1", DType::kF32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}});
  t.push_back({k + ".lokr_w2_a", DType::kF32, {1536, 2}, randn(1536 * 2, 5, 0.1f)});
  t.push_back({k + ".lokr_w2_b", DType::kF32, {2, 128}, randn(2 * 128, 6, 0.1f)});
  t.push_back({k + ".alpha", DType::kF32, {}, {1.0f}});
  t.push_back({"diffusion_model.video_embedder.diff", DType::kBF16, {kD, 128}, randn(kD * 128, 7, 0.01f)});
  t.push_back({"diffusion_model.video_embedder.diff_b", DType::kBF16, {kD}, randn(kD, 8, 0.01f)});
  write_st(dir.path() / "l.safetensors", t);
  const LoraStack s = open1(dir.path() / "l.safetensors");
  ASSERT_EQ(s.adapters().size(), 3u);
  const Adapter& loha = *s.adapters_of("layers.0.proj.o").at(0);
  EXPECT_EQ(loha.kind, AdapterKind::kLoHa);
  EXPECT_FALSE(loha.side_branch);  // a block GEMM, but not low rank: merged
  EXPECT_DOUBLE_EQ(loha.alpha_scale, 2.0f);  // alpha / w1_b rows
  const Adapter& lokr = *s.adapters_of("time_embedding.0").at(0);
  EXPECT_EQ(lokr.kind, AdapterKind::kLoKr);
  EXPECT_DOUBLE_EQ(lokr.alpha_scale, 0.5f);  // alpha / rank of the factored w2
  const Adapter& full = *s.adapters_of("video_embedder").at(0);
  EXPECT_EQ(full.kind, AdapterKind::kFull);
  ASSERT_TRUE(full.bias_diff.has_value());

  // LoKr: element (i, j) = coef * w1[i / 1536, j / 128] * w2[i % 1536, j % 128], w2 = w2a . w2b.
  std::vector<float> w(static_cast<std::size_t>(kD * 256), 0.0f), bias(kD, 0.0f);
  EXPECT_EQ(add_merged_rows(s, "time_embedding.0", 0, kD, w.data(), bias.data()), 1);
  const auto& w2a = t[6].values;
  const auto& w2b = t[7].values;
  const float w1[4] = {1, 2, 3, 4};
  for (const auto [i, j] : {std::pair<int, int>{0, 0}, {1535, 127}, {1536, 0}, {3071, 255}, {700, 130}}) {
    const double w2 = static_cast<double>(w2a[static_cast<std::size_t>((i % 1536) * 2)]) * w2b[static_cast<std::size_t>(j % 128)] +
                      static_cast<double>(w2a[static_cast<std::size_t>((i % 1536) * 2 + 1)]) * w2b[static_cast<std::size_t>(128 + j % 128)];
    const float want = 0.0f + 0.5f * (w1[(i / 1536) * 2 + j / 128] * static_cast<float>(w2));
    EXPECT_EQ(w[static_cast<std::size_t>(i * 256 + j)], want) << i << "," << j;
  }
  // Full delta and bias delta: w + coef * diff exactly.
  std::vector<float> v(static_cast<std::size_t>(kD * 128), 1.0f), vb(kD, 1.0f);
  EXPECT_EQ(add_merged_rows(s, "video_embedder", 0, kD, v.data(), vb.data()), 1);
  EXPECT_EQ(v[5], 1.0f + t[9].values[5]);
  EXPECT_EQ(vb[7], 1.0f + t[10].values[7]);
  // LoHa: coef * f32(w1a . w1b) * f32(w2a . w2b), on the values as stored (F16).
  std::vector<float> o(static_cast<std::size_t>(2 * kD), 0.0f);
  EXPECT_EQ(add_merged_rows(s, "layers.0.proj.o", 100, 2, o.data()), 1);
  const std::vector<float> w1a = s.read(loha.parts[0]), w1b = s.read(loha.parts[1]), w2a_ = s.read(loha.parts[2]),
                           w2b_ = s.read(loha.parts[3]);
  for (int j : {0, 1000, 3071}) {
    const auto dot = [&](const std::vector<float>& a, const std::vector<float>& b, int r, int row) {
      double acc = 0;
      for (int q = 0; q < r; ++q) acc += static_cast<double>(a[static_cast<std::size_t>(row * r + q)]) * b[static_cast<std::size_t>(q * kD + j)];
      return static_cast<float>(acc);
    };
    const float want = 0.0f + 2.0f * (dot(w1a, w1b, 2, 101) * dot(w2a_, w2b_, 3, 101));
    EXPECT_EQ(o[static_cast<std::size_t>(kD + j)], want);
  }
}

TEST(LoraFormat, UnknownKeysAreReported) {
  TempDir dir("lora_unknown");
  auto t = low_rank("diffusion_model.layers.0.proj.q", ".lora_down.weight", ".lora_up.weight", kD, kD, 2, 1);
  t.push_back({"diffusion_model.layers.99.proj.q.lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 2)});
  t.push_back({"lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight", DType::kBF16, {2, 8}, randn(16, 3)});
  t.push_back({"lora_unet_nonexistent_block.lora_up.weight", DType::kBF16, {8, 2}, randn(16, 4)});
  t.push_back({"video_embedder.weight", DType::kBF16, {kD, 3}, randn(kD * 3, 5)});  // a replacement weight
  for (auto& x : low_rank("diffusion_model.layers.1.proj.q", ".lora_down.weight", ".lora_up.weight", kD, kD, 2, 6)) t.push_back(x);
  t.push_back({"diffusion_model.layers.1.proj.q.dora_scale", DType::kBF16, {kD, 1}, randn(kD, 7)});
  t.push_back({"diffusion_model.layers.2.proj.q.alpha", DType::kF32, {}, {1.0f}});
  write_st(dir.path() / "u.safetensors", t);
  const LoraStack s = open1(dir.path() / "u.safetensors");
  ASSERT_EQ(s.adapters().size(), 1u);
  EXPECT_EQ(s.adapters()[0].module, "layers.0.proj.q");
  const auto& unknown = s.reports()[0].unknown;
  EXPECT_EQ(unknown.size(), 8u);  // 1 + 1 + 1 + 1 + (DoRA module: 3 tensors) + orphan alpha
  const auto has = [&](const std::string& needle) {
    for (const std::string& u : unknown) {
      if (u.find(needle) != std::string::npos) return true;
    }
    return false;
  };
  EXPECT_TRUE(has("layers.99.proj.q.lora_down.weight: no module"));
  EXPECT_TRUE(has("text-encoder adapter"));
  EXPECT_TRUE(has("lora_unet_nonexistent_block.lora_up.weight: no module"));
  EXPECT_TRUE(has("video_embedder.weight: not an adapter tensor"));
  EXPECT_TRUE(has("DoRA"));
  EXPECT_TRUE(has("alpha without adapter weights"));
  // strict: every report is an error.
  EXPECT_THROW(open1(dir.path() / "u.safetensors", 1.0f, LoadOptions{true}), LoraError);
  // A file in which nothing applies is an error.
  write_st(dir.path() / "none.safetensors", {{"foo.weight", DType::kF32, {2}, {1, 2}}});
  EXPECT_THROW(open1(dir.path() / "none.safetensors"), LoraError);
}

TEST(LoraFormat, RejectsInvalidAdapters) {
  TempDir dir("lora_bad");
  int n = 0;
  const auto expect_bad = [&](std::vector<StTensor> t, const char* what) {
    const fs::path p = dir.path() / ("b" + std::to_string(n++) + ".safetensors");
    write_st(p, t);
    EXPECT_THROW(open1(p), weights::FormatError) << what;
  };
  const std::string q = "diffusion_model.layers.0.proj.q";
  expect_bad({{q + ".lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 1)}}, "missing up");
  expect_bad({{q + ".lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 1)},
              {q + ".lora_up.weight", DType::kBF16, {kD, 3}, randn(3 * kD, 2)}}, "rank mismatch");
  expect_bad({{q + ".lora_down.weight", DType::kBF16, {2, 100}, randn(200, 1)},
              {q + ".lora_up.weight", DType::kBF16, {kD, 2}, randn(2 * kD, 2)}}, "k mismatch");
  expect_bad({{q + ".lora_down.weight", DType::kI8, {2, kD}, std::vector<float>(2 * kD, 1.0f)},
              {q + ".lora_up.weight", DType::kBF16, {kD, 2}, randn(2 * kD, 2)}}, "int dtype");
  expect_bad({{q + ".lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 1)},
              {q + ".lora_up.weight", DType::kBF16, {kD, 2}, randn(2 * kD, 2)},
              {q + ".alpha", DType::kF32, {2}, {1, 1}}}, "alpha not scalar");
  expect_bad({{q + ".lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 1)},
              {q + ".lora_up.weight", DType::kBF16, {kD, 2}, randn(2 * kD, 2)},
              {q + ".alpha", DType::kF32, {}, {NAN}}}, "alpha NaN");
  expect_bad({{q + ".lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 1)},
              {q + ".lora_up.weight", DType::kBF16, {kD, 2}, randn(2 * kD, 2)},
              {"lora_unet_layers_0_proj_q.lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 1)}}, "two spellings");
  expect_bad({{q + ".lora_down.weight", DType::kBF16, {2, kD}, randn(2 * kD, 1)},
              {q + ".lora_up.weight", DType::kBF16, {kD, 2}, randn(2 * kD, 2)},
              {q + ".diff", DType::kBF16, {kD, kD}, std::vector<float>(kD * kD, 0.0f)}}, "mixed kinds");
  expect_bad({{q + ".hada_w1_a", DType::kBF16, {kD, 2}, randn(2 * kD, 1)}}, "incomplete LoHa");
  expect_bad({{"diffusion_model.time_embedding.0.lokr_w1", DType::kF32, {3, 2}, randn(6, 1)},
              {"diffusion_model.time_embedding.0.lokr_w2", DType::kF32, {1024, 100}, randn(1024 * 100, 2)}}, "kron shape");
  expect_bad({{"diffusion_model.final_linear_video.diff_b", DType::kBF16, {128}, randn(128, 1)}}, "bias delta on a bias-less module");
  expect_bad({{"diffusion_model.time_embedding.0.lora_down.weight", DType::kBF16, {300, 256}, randn(300 * 256, 1)},
              {"diffusion_model.time_embedding.0.lora_up.weight", DType::kBF16, {kD, 300}, randn(300 * kD, 2)}}, "rank above min(n, k)");

  // Non-finite factor values are caught when read (merge / upload), naming the tensor.
  auto t = low_rank("diffusion_model.text_embedder.proj", ".lora_down.weight", ".lora_up.weight", kD, 4096, 2, 3);
  t[0].values[17] = INFINITY;
  write_st(dir.path() / "inf.safetensors", t);
  const LoraStack s = open1(dir.path() / "inf.safetensors");
  std::vector<std::uint16_t> w(static_cast<std::size_t>(kD * 4096), 0);
  EXPECT_THROW(merge_bf16(s, "text_embedder.proj", w.data(), kD, 4096), weights::FormatError);

  const LoraSpec bad{dir.path() / "inf.safetensors", NAN};
  EXPECT_THROW(LoraStack(std::span<const LoraSpec>(&bad, 1), model()), LoraError);
}

// The low-rank merge is the converter's (quant::lora_merge_row), and merge_bf16 is that merge on the widened weight
// rounded once; strength 0 leaves the weight bit-identical.
TEST(LoraMerge, LowRankMatchesConverterAndBf16Rounding) {
  TempDir dir("lora_merge");
  std::vector<StTensor> t = low_rank("lora_unet_text_embedder_blocks_0_ffn_up", ".lora_down.weight", ".lora_up.weight", kF, kD, 8, 31, DType::kF32, 4.0f);
  write_st(dir.path() / "a.safetensors", t);
  write_st(dir.path() / "b.safetensors", low_rank("transformer.text_embedder.blocks.0.ffn_up", ".lora_A.weight", ".lora_B.weight", kF, kD, 4, 41));
  const LoraSpec specs[] = {{dir.path() / "a.safetensors", 0.35f}, {dir.path() / "b.safetensors", 0.6f}};
  LoraStack s(specs, model());
  const std::vector<float> base = randn(static_cast<std::size_t>(kF * kD), 99, 0.05f);
  std::vector<std::uint16_t> wb = mi::testing::lora::to_bf16(base);

  // Reference: per file in order, the converter's row merge.
  const std::int64_t rows0 = 37, rows = 5;
  std::vector<float> ref(base.begin() + rows0 * kD, base.begin() + (rows0 + rows) * kD);
  std::vector<double> acc(kD);
  for (const Adapter* a : s.adapters_of("text_embedder.blocks.0.ffn_up")) {
    const std::vector<float> down = s.read(a->parts[0]), up = s.read(a->parts[1]);
    const float coef = quant::lora_coef(s.strength(a->lora), a->lora == 0 ? std::optional<float>(4.0f) : std::nullopt, a->rank);
    EXPECT_EQ(coef, s.coef(*a));
    for (std::int64_t i = 0; i < rows; ++i) quant::lora_merge_row(ref.data() + i * kD, kD, up.data() + (rows0 + i) * a->rank, down.data(), a->rank, coef, acc.data());
  }
  std::vector<float> got(base.begin() + rows0 * kD, base.begin() + (rows0 + rows) * kD);
  EXPECT_EQ(add_merged_rows(s, "text_embedder.blocks.0.ffn_up", rows0, rows, got.data()), 2);
  EXPECT_EQ(got, ref);

  std::vector<float> widened(static_cast<std::size_t>(kF * kD));
  for (std::size_t i = 0; i < widened.size(); ++i) widened[i] = quant::bf16_to_float(wb[i]);
  add_merged_rows(s, "text_embedder.blocks.0.ffn_up", 0, kF, widened.data());
  std::vector<std::uint16_t> merged = wb;
  EXPECT_EQ(merge_bf16(s, "text_embedder.blocks.0.ffn_up", merged.data(), kF, kD, nullptr, 3), 2);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < merged.size(); ++i) diff += merged[i] != quant::float_to_bf16(widened[i]);
  EXPECT_EQ(diff, 0u);

  s.set_strength(0, 0.0f);
  s.set_strength(1, 0.0f);
  std::vector<std::uint16_t> same = wb;
  EXPECT_EQ(merge_bf16(s, "text_embedder.blocks.0.ffn_up", same.data(), kF, kD), 0);
  EXPECT_EQ(same, wb);
  EXPECT_THROW(merge_bf16(s, "text_embedder.blocks.0.ffn_up", same.data(), kF, kD + 1), LoraError);
}

// LoHa on an INT8 ConvRot base: dequantize, merge, requantize, as the reference sequence of quant_codec calls.
TEST(LoraMerge, RequantizeInt8ConvRot) {
  TempDir dir("lora_requant");
  const std::string h = "diffusion_model.layers.3.proj.ffn_down";
  std::vector<StTensor> t;
  t.push_back({h + ".hada_w1_a", DType::kF16, {kD, 4}, randn(kD * 4, 1, 0.1f)});
  t.push_back({h + ".hada_w1_b", DType::kF16, {4, kF}, randn(kF * 4, 2, 0.5f)});
  t.push_back({h + ".hada_w2_a", DType::kF16, {kD, 4}, randn(kD * 4, 3, 0.01f)});
  t.push_back({h + ".hada_w2_b", DType::kF16, {4, kF}, randn(kF * 4, 4, 0.5f)});
  t.push_back({h + ".alpha", DType::kF16, {}, {4.0f}});
  write_st(dir.path() / "h.safetensors", t);
  const LoraStack s = open1(dir.path() / "h.safetensors", 0.6f);
  const std::int64_t n = kD, k = kF;
  const std::vector<float> w = randn(static_cast<std::size_t>(n * k), 5, 0.02f);
  std::vector<std::int8_t> q(static_cast<std::size_t>(n * k));
  std::vector<float> sc(static_cast<std::size_t>(n));
  ASSERT_TRUE(quant::quantize_convrot_rows(w.data(), n, k, 256, q.data(), sc.data()));
  std::vector<float> deq(static_cast<std::size_t>(n * k));
  quant::dequantize_rows(q.data(), sc.data(), n, k, 256, deq.data());
  add_merged_rows(s, "layers.3.proj.ffn_down", 0, n, deq.data());
  std::vector<std::int8_t> want_q(q.size());
  std::vector<float> want_s(sc.size());
  ASSERT_TRUE(quant::quantize_convrot_rows(deq.data(), n, k, 256, want_q.data(), want_s.data()));
  std::vector<std::int8_t> got_q = q;
  std::vector<float> got_s = sc;
  EXPECT_EQ(merge_requantize_int8(s, "layers.3.proj.ffn_down", got_q.data(), got_s.data(), n, k, 256, 4), 1);
  EXPECT_EQ(got_q, want_q);
  EXPECT_EQ(got_s, want_s);
  // Nothing to merge: untouched.
  std::vector<std::int8_t> q2 = q;
  EXPECT_EQ(merge_requantize_int8(s, "layers.3.proj.o", q2.data(), sc.data(), kD, kD, 256), 0);
  EXPECT_EQ(q2, q);
}

// The surveyed community files (header only keys, synthetic values): every key applies or is reported.
TEST(LoraFormat, SurveyedLayouts) {
  TempDir dir("lora_survey");
  struct Case {
    const char* name;
    std::vector<StTensor> tensors;
    std::string meta;
    int adapters, side, unknown;
  };
  std::vector<Case> cases;
  {  // Comfy-Org color/depth control: unprefixed PEFT keys + a replacement video_embedder (P8, not applied here)
    Case c{"control", {}, R"({"alpha":"64.0","rank":"64"})", 0, 0, 0};
    for (auto& x : low_rank("layers.0.proj.q", ".lora_A.weight", ".lora_B.weight", kD, kD, 8, 1)) c.tensors.push_back(x);
    for (auto& x : low_rank("text_embedder.blocks.0.qkv", ".lora_A.weight", ".lora_B.weight", 3 * kD, kD, 8, 2)) c.tensors.push_back(x);
    c.tensors.push_back({"video_embedder.weight", DType::kBF16, {kD, 384}, randn(kD * 384, 3)});
    c.tensors.push_back({"video_embedder.bias", DType::kBF16, {kD}, randn(kD, 4)});
    c.adapters = 2, c.side = 1, c.unknown = 2;
    cases.push_back(std::move(c));
  }
  {  // MATLOWAI deband: full deltas on the fused qkv
    Case c{"deband", {{"diffusion_model.layers.21.proj.qkv.diff", DType::kBF16, {3 * kD, kD}, std::vector<float>(3 * kD * kD, 0.001f)}}, "", 3, 0, 0};
    cases.push_back(std::move(c));
  }
  {  // lodestones / akhaliq DMD: kohya F32 with alpha
    Case c{"dmd", low_rank("diffusion_model.layers.7.proj.ffn_down", ".lora_down.weight", ".lora_up.weight", kD, kF, 16, 5, DType::kF32, 16.0f), R"({"rank":"128","alpha":"128.0"})", 1, 1, 0};
    cases.push_back(std::move(c));
  }
  {  // comfyanonymous hires: LoHa on the fused qkv, F16
    Case c{"hires", {}, "", 3, 0, 0};
    const std::string h = "diffusion_model.layers.2.proj.qkv";
    c.tensors = {{h + ".alpha", DType::kF16, {}, {16.0f}},
                 {h + ".hada_w1_a", DType::kF16, {3 * kD, 2}, randn(3 * kD * 2, 1, 0.1f)},
                 {h + ".hada_w1_b", DType::kF16, {2, kD}, randn(2 * kD, 2, 0.1f)},
                 {h + ".hada_w2_a", DType::kF16, {3 * kD, 2}, randn(3 * kD * 2, 3, 0.1f)},
                 {h + ".hada_w2_b", DType::kF16, {2, kD}, randn(2 * kD, 4, 0.1f)}};
    cases.push_back(std::move(c));
  }
  for (Case& c : cases) {
    const fs::path p = dir.path() / (std::string(c.name) + ".safetensors");
    write_st(p, c.tensors, c.meta);
    const LoraStack s = open1(p);
    EXPECT_EQ(static_cast<int>(s.adapters().size()), c.adapters) << c.name;
    EXPECT_EQ(s.reports()[0].side_branch, c.side) << c.name;
    EXPECT_EQ(static_cast<int>(s.reports()[0].unknown.size()), c.unknown) << c.name;
  }
}

// The downloaded community LoRAs parse completely (skipped when absent).
TEST(LoraFormat, RealFiles) {
  namespace tl = mi::testing::lora;
  struct Real {
    const char* path;
    int adapters, side;
  };
  const Real files[] = {{tl::kKohyaLora, 254, 252}, {tl::kPeftLora, 252, 252}, {tl::kTinyLora, 25, 25}, {tl::kLohaLora, 252, 0}};
  int seen = 0;
  for (const Real& r : files) {
    if (!fs::exists(r.path)) continue;
    ++seen;
    const LoraStack s = open1(r.path);
    EXPECT_EQ(static_cast<int>(s.adapters().size()), r.adapters) << r.path;
    EXPECT_EQ(s.reports()[0].side_branch, r.side) << r.path;
    EXPECT_TRUE(s.reports()[0].unknown.empty()) << r.path << ": " << s.reports()[0].unknown.front();
    for (const Adapter& a : s.adapters()) EXPECT_TRUE(std::isfinite(a.alpha_scale));
  }
  if (seen == 0) GTEST_SKIP() << "no community LoRA files under /workspace/Models/community";
}

// The shared alpha rule: an alpha tensor wins, else the PEFT metadata alpha (noted in the report), else 1; the
// converter's coefficient (quant::lora_coef) uses the same rule.
TEST(LoraFormat, AlphaRuleSharedWithConverter) {
  TempDir dir("lora_alpha");
  const std::string meta = R"({"lora_adapter_metadata":"{\"r\": 16, \"lora_alpha\": 8}"})";
  auto t = low_rank("transformer.layers.0.proj.k", ".lora_A.weight", ".lora_B.weight", kD, kD, 16, 1);
  for (auto& x : low_rank("transformer.layers.0.proj.v", ".lora_A.weight", ".lora_B.weight", kD, kD, 16, 3, DType::kBF16, 4.0f)) t.push_back(x);
  write_st(dir.path() / "m.safetensors", t, meta);
  const LoraStack s = open1(dir.path() / "m.safetensors", 0.6f);
  const Adapter& k = *s.adapters_of("layers.0.proj.k").at(0);
  const Adapter& v = *s.adapters_of("layers.0.proj.v").at(0);
  EXPECT_DOUBLE_EQ(k.alpha_scale, 0.5);   // metadata lora_alpha 8 / r 16
  EXPECT_DOUBLE_EQ(v.alpha_scale, 0.25);  // the alpha tensor (4) wins
  ASSERT_EQ(s.reports()[0].notes.size(), 1u);
  EXPECT_NE(s.reports()[0].notes[0].find("lora_alpha 8 from lora_adapter_metadata applied to 1 modules"), std::string::npos);
  EXPECT_TRUE(s.reports()[0].warnings.empty());
  // The converter's rule, through the same functions.
  std::map<std::string, std::string> md{{"lora_adapter_metadata", R"({"r": 16, "lora_alpha": 8})"}};
  ASSERT_EQ(quant::lora_metadata_alpha(md), std::optional<double>(8.0));
  EXPECT_EQ(quant::lora_coef(0.6f, quant::lora_metadata_alpha(md), 16), s.coef(k));
  EXPECT_EQ(quant::lora_coef(0.6f, 4.0f, 16), s.coef(v));
  EXPECT_EQ(quant::lora_metadata_alpha({}), std::nullopt);
  md["lora_adapter_metadata"] = R"({"alpha_pattern": {"q": 2}})";
  EXPECT_THROW(quant::lora_metadata_alpha(md), quant::QuantError);
}

// moss-edit-lora (Moss-Image-Edit): bare .lora_A / .lora_B keys, each module's alpha from the adapter_config header
// (here alpha != rank), q / k / v as separate modules of the fused qkv, and the file's format in its report.
TEST(LoraFormat, MossEditLoraAdapterConfig) {
  TempDir dir("lora_edit");
  auto t = low_rank("layers.0.proj.q", ".lora_A", ".lora_B", kD, kD, 16, 1);
  for (auto& x : low_rank("layers.0.proj.k", ".lora_A", ".lora_B", kD, kD, 16, 2)) t.push_back(x);
  for (auto& x : low_rank("layers.0.proj.v", ".lora_A", ".lora_B", kD, kD, 16, 3)) t.push_back(x);
  const std::string meta =
      R"({"adapter_config":"{\"format\": \"moss-edit-lora\", \"format_version\": 1, \"modules\": {)"
      R"(\"layers.0.proj.q\": {\"alpha\": 32.0, \"rank\": 16}, \"layers.0.proj.k\": {\"alpha\": 8, \"rank\": 16}}}"})";
  write_st(dir.path() / "e.safetensors", t, meta);
  const LoraStack s = open1(dir.path() / "e.safetensors");
  EXPECT_DOUBLE_EQ(s.adapters_of("layers.0.proj.q").at(0)->alpha_scale, 2.0);
  EXPECT_DOUBLE_EQ(s.adapters_of("layers.0.proj.k").at(0)->alpha_scale, 0.5);
  EXPECT_DOUBLE_EQ(s.adapters_of("layers.0.proj.v").at(0)->alpha_scale, 1.0);  // no alpha: 1
  EXPECT_EQ(s.reports()[0].format, "moss-edit-lora");
  EXPECT_TRUE(s.reports()[0].unknown.empty());
  EXPECT_NE(s.reports()[0].notes.back().find("adapter_config (moss-edit-lora) applied to 2 modules"), std::string::npos);
  std::map<std::string, std::string> md{{"adapter_config", R"({"format": "other", "modules": {"q": {"alpha": 2}}})"}};
  EXPECT_TRUE(quant::lora_adapter_config_alphas(md).empty());  // only the moss-edit-lora format's
  EXPECT_EQ(quant::lora_adapter_config_format(md), "other");
  md["adapter_config"] = R"({"format": "moss-edit-lora", "modules": {"q": {"alpha": "x"}}})";
  EXPECT_THROW(quant::lora_adapter_config_alphas(md), quant::QuantError);
  md["adapter_config"] = "not json";
  EXPECT_THROW(quant::lora_adapter_config_format(md), quant::QuantError);
}

// On an INT8 base every merged adapter of a block GEMM is reported as a warning; strict turns warnings into errors.
TEST(LoraFormat, Int8MergeWarnings) {
  TempDir dir("lora_warn");
  std::vector<StTensor> t;
  const std::string h = "diffusion_model.layers.4.proj.o";
  t.push_back({h + ".hada_w1_a", DType::kF16, {kD, 2}, randn(kD * 2, 1, 0.1f)});
  t.push_back({h + ".hada_w1_b", DType::kF16, {2, kD}, randn(kD * 2, 2, 0.1f)});
  t.push_back({h + ".hada_w2_a", DType::kF16, {kD, 2}, randn(kD * 2, 3, 0.1f)});
  t.push_back({h + ".hada_w2_b", DType::kF16, {2, kD}, randn(kD * 2, 4, 0.1f)});
  for (auto& x : low_rank("diffusion_model.layers.4.proj.q", ".lora_down.weight", ".lora_up.weight", kD, kD, 4, 5)) t.push_back(x);
  for (auto& x : low_rank("diffusion_model.text_embedder.proj", ".lora_down.weight", ".lora_up.weight", kD, 4096, 4, 7)) t.push_back(x);
  write_st(dir.path() / "w.safetensors", t);
  const LoraStack bf = open1(dir.path() / "w.safetensors");
  EXPECT_TRUE(bf.reports()[0].warnings.empty());  // bf16 base: merging adds no requantization
  LoadOptions opt;
  opt.int8_block_gemms = true;
  const LoraStack q = open1(dir.path() / "w.safetensors", 1.0f, opt);
  // One warning per file: only the LoHa on layers.4.proj.o is merged into INT8 (side branch and text_embedder: none).
  ASSERT_EQ(q.reports()[0].warnings.size(), 1u);
  EXPECT_EQ(q.reports()[0].warnings[0], "1 adapter merged into INT8 (requantization error)");
  opt.strict = true;
  EXPECT_THROW(open1(dir.path() / "w.safetensors", 1.0f, opt), LoraError);
}

// lora_placement, per module from its base format: an ordinary LoRA stays on the side branch under default and merges
// under merge (refused on a packed 4-bit module); a moss-edit-lora file too: the side branch everywhere by default,
// merged under merge (on INT8 one warning per file, refused by strict naming side_branch); side_branch never merges.
// side_branch; side_branch never merges. Merged edit block modules keep no pristine copy.
TEST(LoraFormat, PlacementPerModule) {
  TempDir dir("lora_place");
  auto plain = low_rank("transformer.layers.0.proj.q", ".lora_A.weight", ".lora_B.weight", kD, kD, 4, 1);
  for (auto& x : low_rank("transformer.layers.1.proj.o", ".lora_A.weight", ".lora_B.weight", kD, kD, 4, 2)) plain.push_back(x);
  write_st(dir.path() / "plain.safetensors", plain);
  const std::string edit_meta = R"({"adapter_config":"{\"format\": \"moss-edit-lora\", \"modules\": {}}"})";
  auto edit = low_rank("layers.0.proj.q", ".lora_A", ".lora_B", kD, kD, 4, 3);
  for (auto& x : low_rank("layers.1.proj.o", ".lora_A", ".lora_B", kD, kD, 4, 4)) edit.push_back(x);
  for (auto& x : low_rank("layers.2.proj.k", ".lora_A", ".lora_B", kD, kD, 4, 5)) edit.push_back(x);
  write_st(dir.path() / "edit.safetensors", edit, edit_meta);
  write_st(dir.path() / "edit8.safetensors", low_rank("layers.0.proj.q", ".lora_A", ".lora_B", kD, kD, 4, 3), edit_meta);
  // Block 0 INT8, block 1 packed 4-bit, the others bf16.
  const auto base = [](std::string_view m) {
    return m.starts_with("layers.0.")   ? mi::lora::BaseFormat::kInt8
           : m.starts_with("layers.1.") ? mi::lora::BaseFormat::kPacked4
                                        : mi::lora::BaseFormat::kBf16;
  };
  const auto open = [&](const char* file, mi::lora::Placement p, bool strict = false) {
    LoadOptions o;
    o.placement = p;
    o.base_format = base;
    o.strict = strict;
    return open1(dir.path() / file, 1.0f, o);
  };
  using mi::lora::Placement;
  const auto side = [](const LoraStack& s, const char* m) { return s.adapters_of(m).at(0)->side_branch; };
  {
    const LoraStack s = open("plain.safetensors", Placement::kDefault);
    EXPECT_TRUE(side(s, "layers.0.proj.q"));
    EXPECT_TRUE(side(s, "layers.1.proj.o"));
    EXPECT_TRUE(s.reports()[0].warnings.empty());
  }
  EXPECT_THROW(open("plain.safetensors", Placement::kMerge), LoraError);  // layers.1.proj.o is packed
  {
    const LoraStack s = open("edit.safetensors", Placement::kDefault, true);  // strict: nothing lossy by default
    EXPECT_TRUE(side(s, "layers.0.proj.q"));
    EXPECT_TRUE(side(s, "layers.1.proj.o"));
    EXPECT_TRUE(side(s, "layers.2.proj.k"));
    EXPECT_TRUE(s.merge_modules().empty());
    EXPECT_TRUE(s.reports()[0].warnings.empty());
  }
  {
    // merge on a bf16 module: no warning, no pristine copy for an edit adapter's block module.
    write_st(dir.path() / "edit16.safetensors", low_rank("layers.2.proj.k", ".lora_A", ".lora_B", kD, kD, 4, 5), edit_meta);
    const LoraStack s = open("edit16.safetensors", Placement::kMerge, true);
    EXPECT_FALSE(side(s, "layers.2.proj.k"));
    EXPECT_FALSE(s.keeps_pristine("layers.2.proj.k"));
    EXPECT_TRUE(s.reports()[0].warnings.empty());
  }
  {
    const LoraStack s = open("edit8.safetensors", Placement::kMerge);
    EXPECT_FALSE(side(s, "layers.0.proj.q"));
    EXPECT_FALSE(s.keeps_pristine("layers.0.proj.q"));
    EXPECT_EQ(s.reports()[0].warnings, std::vector<std::string>{"1 adapter merged into INT8 (requantization error)"});
  }
  try {
    open("edit8.safetensors", Placement::kMerge, true);
    ADD_FAILURE() << "strict accepted an INT8 merge";
  } catch (const LoraError& e) {
    EXPECT_NE(std::string(e.what()).find("lora_placement side_branch"), std::string::npos) << e.what();
  }
  {
    const LoraStack s = open("edit.safetensors", Placement::kSideBranch, true);
    EXPECT_TRUE(side(s, "layers.0.proj.q"));
    EXPECT_TRUE(side(s, "layers.2.proj.k"));
    EXPECT_TRUE(s.merge_modules().empty());
  }
  EXPECT_EQ(mi::lora::parse_placement("side_branch"), Placement::kSideBranch);
  EXPECT_STREQ(mi::lora::placement_name(Placement::kDefault), "default");
  EXPECT_THROW(mi::lora::parse_placement("auto"), std::invalid_argument);
}

// PristineWeights: a strength change re-merges from the kept base weights; strength 0 restores them bit for bit.
TEST(LoraMerge, PristineRemerge) {
  TempDir dir("lora_pristine");
  write_st(dir.path() / "a.safetensors", low_rank("lora_unet_text_embedder_proj", ".lora_down.weight", ".lora_up.weight", kD, 4096, 8, 1, DType::kF32, 8.0f));
  const std::string hq = "diffusion_model.layers.2.proj.ffn_down";
  write_st(dir.path() / "h.safetensors", {{hq + ".hada_w1_a", DType::kF16, {kD, 2}, randn(kD * 2, 1, 0.1f)},
                                          {hq + ".hada_w1_b", DType::kF16, {2, kF}, randn(kF * 2, 2, 0.5f)},
                                          {hq + ".hada_w2_a", DType::kF16, {kD, 2}, randn(kD * 2, 3, 0.01f)},
                                          {hq + ".hada_w2_b", DType::kF16, {2, kF}, randn(kF * 2, 4, 0.5f)}});
  const LoraSpec specs[] = {{dir.path() / "a.safetensors", 0.35f}, {dir.path() / "h.safetensors", 0.6f}};
  LoraStack s(specs, model());
  const std::vector<std::uint16_t> w = mi::testing::lora::to_bf16(randn(static_cast<std::size_t>(kD * 4096), 9, 0.02f));
  const std::vector<std::uint16_t> bias = mi::testing::lora::to_bf16(randn(kD, 10, 0.02f));
  const std::vector<float> wf = randn(static_cast<std::size_t>(kD * kF), 11, 0.02f);
  std::vector<std::int8_t> q(static_cast<std::size_t>(kD * kF));
  std::vector<float> sc(static_cast<std::size_t>(kD));
  ASSERT_TRUE(quant::quantize_convrot_rows(wf.data(), kD, kF, 256, q.data(), sc.data()));
  PristineWeights pw;
  pw.keep_bf16("text_embedder.proj", w.data(), kD, 4096, bias.data());
  pw.keep_int8("layers.2.proj.ffn_down", q.data(), sc.data(), kD, kF, 256);
  EXPECT_TRUE(pw.has("text_embedder.proj"));
  EXPECT_EQ(pw.host_bytes(), w.size() * 2 + bias.size() * 2 + q.size() + sc.size() * 4);
  EXPECT_THROW(pw.keep_bf16("text_embedder.proj", w.data(), kD, 4096), LoraError);

  // At 0.35: the merge of the pristine weight (merge_bf16 on a fresh copy).
  std::vector<std::uint16_t> dst(w.size()), dst_b(bias.size()), want = w, want_b = bias;
  EXPECT_EQ(pw.remerge_bf16(s, "text_embedder.proj", dst.data(), dst_b.data()), 1);
  merge_bf16(s, "text_embedder.proj", want.data(), kD, 4096, want_b.data());
  EXPECT_EQ(dst, want);
  // Strength change: merged again from the pristine copy, not on top of the previous merge.
  s.set_strength(0, 0.7f);
  want = w;
  merge_bf16(s, "text_embedder.proj", want.data(), kD, 4096, want_b.data());
  EXPECT_EQ(pw.remerge_bf16(s, "text_embedder.proj", dst.data(), dst_b.data()), 1);
  EXPECT_EQ(dst, want);
  s.set_strength(0, 0.0f);
  EXPECT_EQ(pw.remerge_bf16(s, "text_embedder.proj", dst.data(), dst_b.data()), 0);
  EXPECT_EQ(dst, w);
  EXPECT_EQ(dst_b, bias);
  // INT8: requantized from the pristine grid; strength 0 restores it.
  std::vector<std::int8_t> dq(q.size()), wq = q;
  std::vector<float> ds(sc.size()), ws = sc;
  EXPECT_EQ(pw.remerge_int8(s, "layers.2.proj.ffn_down", dq.data(), ds.data()), 1);
  merge_requantize_int8(s, "layers.2.proj.ffn_down", wq.data(), ws.data(), kD, kF, 256);
  EXPECT_EQ(dq, wq);
  EXPECT_EQ(ds, ws);
  s.set_strength(1, 0.0f);
  EXPECT_EQ(pw.remerge_int8(s, "layers.2.proj.ffn_down", dq.data(), ds.data()), 0);
  EXPECT_EQ(dq, q);
  EXPECT_THROW(pw.remerge_bf16(s, "layers.2.proj.ffn_down", dst.data()), LoraError);  // kept as INT8
  EXPECT_THROW(pw.remerge_bf16(s, "layers.0.proj.o", dst.data()), LoraError);         // not kept
}

}  // namespace
}  // namespace mi::lora
