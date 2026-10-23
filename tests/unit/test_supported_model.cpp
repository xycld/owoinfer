// Identification of supported models (weights/supported_model.h) and of unsupported quantization formats
// (quant::open_quant_model): other architectures, a MOSS-Image folder with a changed config, single files with other
// shapes and an unknown format are refused with a plain message; MOSS-Image configs and files (official, or a
// fine-tune with the same architecture) pass. Configs and headers only: tiny synthetic files, plus the real bundles
// and community files when present.
#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "test_util.h"
#include "weights/quant/quant_model.h"
#include "weights/supported_model.h"

namespace {

using namespace mi::weights;
namespace fs = std::filesystem;
using mi::testing::TempDir;
using mi::testing::TensorSpec;
using mi::testing::write_safetensors;

constexpr const char* kHint = "Supported: OpenMOSS-Team/MOSS-Image-Base and -SFT; see docs/integration.md#supported-models";

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

// The official configs (OpenMOSS-Team/MOSS-Image-Base and -SFT ship the same ones), with `edit` applied to the text.
std::string layer_types() {
  std::string s = "[";
  for (int i = 0; i < 32; ++i) s += std::string(i ? "," : "") + (i % 4 == 3 ? "\"full_attention\"" : "\"linear_attention\"");
  return s + "]";
}
const std::string kModelIndex = R"({"_class_name": "MOSSImagePipeline", "_diffusers_version": "0.38.0",
  "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"], "text_encoder": ["transformers", "Qwen3_5Model"],
  "tokenizer": ["transformers", "Qwen2Tokenizer"], "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
  "vae": ["diffusers", "AutoencoderKLFlux2"]})";
const std::string kTransformer = R"({"_class_name": "MOSSImageTransformer", "_diffusers_version": "0.38.0", "dim": 3072,
  "ffn_dim": 8192, "freq_dim": 256, "latent_channels": 128, "num_heads": 24, "num_layers": 42, "rope_base": 10000.0, "text_dim": 4096})";
std::string text_encoder_config(const std::string& text_config_extra = "") {
  return R"({"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5", "text_config": {)" + text_config_extra +
         R"("head_dim": 256, "hidden_size": 4096, "intermediate_size": 12288, "layer_types": )" + layer_types() +
         R"(, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
         "linear_value_head_dim": 128, "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4,
         "vocab_size": 248320, "model_type": "qwen3_5_text"}})";
}
const std::string kVae = R"({"_class_name": "AutoencoderKLFlux2", "latent_channels": 32, "patch_size": [2, 2]})";

struct Folder {
  explicit Folder(const std::string& tag) : dir(tag) {}
  TempDir dir;
  fs::path root() const { return dir.path(); }
  void write(const std::string& model_index, const std::string& transformer, const std::string& text_encoder, const std::string& vae) {
    if (!model_index.empty()) write_text(root() / "model_index.json", model_index);
    if (!transformer.empty()) write_text(root() / "transformer/config.json", transformer);
    if (!text_encoder.empty()) write_text(root() / "text_encoder/config.json", text_encoder);
    if (!vae.empty()) write_text(root() / "vae/config.json", vae);
  }
};

std::string replace(std::string s, const std::string& from, const std::string& to) {
  const auto at = s.find(from);
  EXPECT_NE(at, std::string::npos) << from;
  return at == std::string::npos ? s : s.replace(at, from.size(), to);
}

// `fn` throws UnsupportedModelError whose message holds every part of `parts` and the supported-models hint.
void expect_refused(const std::function<void()>& fn, const std::vector<std::string>& parts) {
  try {
    fn();
    ADD_FAILURE() << "expected UnsupportedModelError";
  } catch (const UnsupportedModelError& e) {
    const std::string m = e.what();
    for (const std::string& p : parts) EXPECT_NE(m.find(p), std::string::npos) << "'" << p << "' not in: " << m;
    EXPECT_NE(m.find(kHint), std::string::npos) << m;
  }
}

// A single-file DiT header: MOSS-Image names with the given hidden size and block count (ComfyUI prefix optional).
fs::path write_dit(const fs::path& path, std::int64_t dim, int blocks, const std::string& prefix = "", DType gemm = DType::kI8,
                   int first = 0) {
  std::vector<TensorSpec> t;
  for (int i = first; i < first + blocks; ++i) {
    const std::string b = prefix + "layers." + std::to_string(i) + ".";
    t.push_back({b + "norm1.weight", DType::kBF16, {dim}});
    t.push_back({b + "q_norm.weight", DType::kBF16, {128}});
    t.push_back({b + "proj.q.weight", gemm, {4, 4}});
    t.push_back({b + "proj.ffn_down.weight", gemm, {4, 4}});
  }
  t.push_back({prefix + "video_embedder.weight", DType::kBF16, {dim, 128}});
  t.push_back({prefix + "text_embedder.proj.weight", DType::kBF16, {dim, 4096}});
  t.push_back({prefix + "time_embedding.0.weight", DType::kBF16, {dim, 256}});
  t.push_back({prefix + "final_linear_video.weight", DType::kBF16, {128, dim}});
  write_safetensors(path, t);
  return path;
}

// A single-file text encoder header: Qwen3_5Model names with the given hidden size and layer count.
fs::path write_te(const fs::path& path, std::int64_t hidden, int layers, std::int64_t vocab = 248320) {
  std::vector<TensorSpec> t = {{"model.language_model.embed_tokens.weight", DType::kI8, {vocab, 4}},
                               {"model.language_model.norm.weight", DType::kBF16, {hidden}},
                               {"lm_head.weight", DType::kI8, {4, 4}}};
  for (int i = 0; i < layers; ++i) {
    const std::string l = "model.language_model.layers." + std::to_string(i) + ".";
    t.push_back({l + "input_layernorm.weight", DType::kBF16, {hidden}});
    if (i % 4 == 3) t.push_back({l + "self_attn.q_norm.weight", DType::kBF16, {256}});
  }
  write_safetensors(path, t);
  return path;
}

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}

}  // namespace

TEST(SupportedModel, OfficialConfigsAndFineTunesPass) {
  Folder f("sm_ok");
  f.write(kModelIndex, kTransformer, text_encoder_config(), kVae);
  EXPECT_NO_THROW(check_supported_bundle(f.root()));
  // A fine-tune keeps the architecture: other diffusers / transformers versions, extra keys, no model_index.json or
  // VAE config, an integer rope_base.
  fs::remove(f.root() / "model_index.json");
  fs::remove(f.root() / "vae/config.json");
  write_text(f.root() / "transformer/config.json", replace(replace(kTransformer, "0.38.0", "0.40.1"), "10000.0", "10000"));
  write_text(f.root() / "text_encoder/config.json", text_encoder_config(R"("trained_on": "my data", )"));
  EXPECT_NO_THROW(check_supported_bundle(f.root()));
}

// Class names only describe what was found: the same architecture under other class names loads.
TEST(SupportedModel, ClassNamesDoNotDecide) {
  Folder f("sm_classes");
  f.write(replace(kModelIndex, "\"MOSSImagePipeline\"", "\"MyMossPipeline\""),
          replace(kTransformer, "\"MOSSImageTransformer\"", "\"MyMossTransformer\""),
          replace(text_encoder_config(), "\"model_type\": \"qwen3_5\", ", "\"model_type\": \"qwen3_5_custom\", "),
          replace(kVae, "\"AutoencoderKLFlux2\"", "\"MyVae\""));
  EXPECT_NO_THROW(check_supported_bundle(f.root()));
  // List-form or non-string class names are ignored.
  write_text(f.root() / "model_index.json", R"({"_class_name": ["MOSSImagePipeline"], "transformer": 5})");
  write_text(f.root() / "vae/config.json", R"({"_class_name": 7, "latent_channels": 32})");
  EXPECT_NO_THROW(check_supported_bundle(f.root()));
  write_text(f.root() / "transformer/config.json", replace(replace(kTransformer, "\"MOSSImageTransformer\"", "[\"x\"]"), "\"dim\": 3072", "\"dim\": 2048"));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found transformer dim 2048 (MOSS-Image: 3072))"});
  // A differing field under a pipeline of another name: the name leads, the field follows.
  write_text(f.root() / "model_index.json", replace(kModelIndex, "\"MOSSImagePipeline\"", "\"MyMossPipeline\""));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found MyMossPipeline with transformer MOSSImageTransformer, transformer dim 2048 (MOSS-Image: 3072))"});
}

// (a) Another architecture: a FLUX diffusers folder is named by its classes, a folder in the MOSS-Image layout whose
// configs describe other dims by those dims.
TEST(SupportedModel, OtherArchitectureFolders) {
  Folder flux("sm_flux");
  flux.write(R"({"_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"], "vae": ["diffusers", "AutoencoderKL"]})",
             R"({"_class_name": "FluxTransformer2DModel", "num_layers": 19, "num_single_layers": 38})",
             R"({"architectures": ["CLIPTextModel"], "model_type": "clip_text_model"})", R"({"_class_name": "AutoencoderKL"})");
  expect_refused([&] { check_supported_bundle(flux.root()); },
                 {mi::util::path_utf8(flux.root()) + " is not a supported MOSS-Image model (found FluxPipeline with transformer FluxTransformer2DModel)"});
  // Without model_index.json the component classes name it.
  fs::remove(flux.root() / "model_index.json");
  expect_refused([&] { check_supported_bundle(flux.root()); }, {"found transformer FluxTransformer2DModel, text encoder CLIPTextModel)"});
  write_text(flux.root() / "transformer/config.json", kTransformer);
  expect_refused([&] { check_supported_bundle(flux.root()); }, {"found text encoder CLIPTextModel)"});

  Folder small("sm_small");
  small.write(kModelIndex, replace(replace(kTransformer, "\"dim\": 3072", "\"dim\": 2048"), "\"num_heads\": 24", "\"num_heads\": 16"),
              replace(text_encoder_config(), "\"hidden_size\": 4096", "\"hidden_size\": 2560"), kVae);
  expect_refused([&] { check_supported_bundle(small.root()); },
                 {"transformer dim 2048 (MOSS-Image: 3072)", "transformer num_heads 16 (MOSS-Image: 24)", "text encoder hidden_size 2560 (MOSS-Image: 4096)"});

  // Another pipeline's folder without our configs is still another model.
  TempDir other("sm_other");
  write_text(other.path() / "model_index.json", R"({"_class_name": "StableDiffusion3Pipeline"})");
  expect_refused([&] { check_supported_bundle(other.path()); }, {"found StableDiffusion3Pipeline)"});
}

// Not a model folder, or an incomplete one: a plain IndexError, not "another model".
TEST(SupportedModel, MissingFolderOrConfigIsAnIndexError) {
  auto expect_index_error = [](const fs::path& dir, const std::string& part) {
    try {
      check_supported_bundle(dir);
      ADD_FAILURE() << "expected IndexError";
    } catch (const UnsupportedModelError& e) {
      ADD_FAILURE() << "not another model: " << e.what();
    } catch (const IndexError& e) {
      EXPECT_NE(std::string(e.what()).find(part), std::string::npos) << e.what();
    }
  };
  TempDir empty("sm_empty");
  expect_index_error(empty.path(), "no transformer/config.json (incomplete download, or not the model folder?)");
  expect_index_error(empty.path() / "missing", "not a directory");
  write_text(empty.path() / "transformer/config.json", kTransformer);
  expect_index_error(empty.path(), "no text_encoder/config.json (incomplete download");
  write_text(empty.path() / "model_index.json", kModelIndex);
  expect_index_error(empty.path(), "no text_encoder/config.json (incomplete download");
  // A malformed config (an interrupted download) too.
  write_text(empty.path() / "text_encoder/config.json", "{\"text_config\": ");
  expect_index_error(empty.path(), "text_encoder");
}

// (b) The MOSS-Image folder with one config value changed.
TEST(SupportedModel, ChangedMossImageConfig) {
  Folder f("sm_changed");
  auto reset = [&] { f.write(kModelIndex, kTransformer, text_encoder_config(), kVae); };
  reset();
  write_text(f.root() / "transformer/config.json", replace(kTransformer, "\"num_layers\": 42", "\"num_layers\": 40"));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found transformer num_layers 40 (MOSS-Image: 42))"});

  reset();
  write_text(f.root() / "transformer/config.json", replace(kTransformer, "\"rope_base\": 10000.0", "\"rope_base\": 1000000.0"));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found transformer rope_base 1000000 (MOSS-Image: 10000))"});

  reset();
  write_text(f.root() / "transformer/config.json", replace(kTransformer, ", \"text_dim\": 4096", ""));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found transformer without text_dim)"});

  reset();
  write_text(f.root() / "text_encoder/config.json", replace(text_encoder_config(), "\"num_hidden_layers\": 32", "\"num_hidden_layers\": 24"));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found text encoder num_hidden_layers 24 (MOSS-Image: 32))"});

  reset();
  write_text(f.root() / "text_encoder/config.json", replace(text_encoder_config(), layer_types(), replace(layer_types(), "\"full_attention\"", "\"linear_attention\"")));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found text encoder layer_types other than MOSS-Image's"});

  reset();
  write_text(f.root() / "text_encoder/config.json", R"({"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5"})");
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found a text encoder config without text_config)"});

  reset();
  write_text(f.root() / "vae/config.json", replace(kVae, "\"latent_channels\": 32", "\"latent_channels\": 16"));
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found VAE latent_channels 16 (MOSS-Image: 32))"});
  // With another VAE class, the class names it.
  write_text(f.root() / "vae/config.json", R"({"_class_name": "AutoencoderKL", "latent_channels": 16})");
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found VAE AutoencoderKL)"});

  // Many differences: the first four are named, the rest counted (7 transformer fields).
  reset();
  write_text(f.root() / "transformer/config.json", R"({"dim": 1, "ffn_dim": 1, "freq_dim": 1, "latent_channels": 1, "num_heads": 1, "num_layers": 1, "text_dim": 1})");
  expect_refused([&] { check_supported_bundle(f.root()); }, {"found transformer dim 1 (MOSS-Image: 3072), ", " and 3 more differences)"});
  write_text(f.root() / "transformer/config.json", R"({"dim": 1, "ffn_dim": 1, "freq_dim": 1, "latent_channels": 1, "num_heads": 24, "num_layers": 42, "text_dim": 1})");
  expect_refused([&] { check_supported_bundle(f.root()); }, {" and 1 more difference)"});
}

// (c) A single-file DiT with other shapes, other names, another block count, another role or format; and the
// text-encoder file check.
TEST(SupportedModel, SingleFilesWithOtherShapes) {
  TempDir dir("sm_files");
  EXPECT_NO_THROW(check_supported_dit_file(write_dit(dir.path() / "ok.safetensors", 3072, 42)));
  EXPECT_NO_THROW(check_supported_dit_file(write_dit(dir.path() / "comfy.safetensors", 3072, 42, "model.diffusion_model.")));
  EXPECT_NO_THROW(check_supported_dit_file(write_dit(dir.path() / "bf16.safetensors", 3072, 42, "", DType::kBF16)));

  expect_refused([&] { check_supported_dit_file(write_dit(dir.path() / "d2048.safetensors", 2048, 42)); },
                 {"d2048.safetensors is not a supported MOSS-Image DiT (found video_embedder.weight [2048, 128] (MOSS-Image: [3072, 128])"});
  expect_refused([&] { check_supported_dit_file(write_dit(dir.path() / "b40.safetensors", 3072, 40)); }, {"found 40 blocks (MOSS-Image: 42))"});
  expect_refused([&] { check_supported_dit_file(write_dit(dir.path() / "b1.safetensors", 3072, 42, "", DType::kI8, 1)); },
                 {"42 blocks numbered 1..42 (MOSS-Image: 42 blocks, 0..41)"});

  const fs::path flux = dir.path() / "flux.safetensors";
  write_safetensors(flux, {{"double_blocks.0.img_attn.qkv.weight", DType::kBF16, {4, 4}}, {"img_in.weight", DType::kBF16, {4, 2}}});
  expect_refused([&] { check_supported_dit_file(flux); }, {"found tensors such as 'double_blocks.0.img_attn.qkv.weight')"});
  const fs::path empty = dir.path() / "empty.safetensors";
  write_safetensors(empty, {});
  expect_refused([&] { check_supported_dit_file(empty); }, {"found no tensors)"});
  expect_refused([&] { check_supported_text_encoder_file(empty); }, {"found no tensors)"});
  // The text encoder file given as the DiT, and the reverse.
  const fs::path te = write_te(dir.path() / "te.safetensors", 4096, 32);
  expect_refused([&] { check_supported_dit_file(te); }, {"found text-encoder tensors such as"});
  expect_refused([&] { check_supported_text_encoder_file(dir.path() / "ok.safetensors"); }, {"found DiT tensors such as"});

  // Other containers and quantization layouts of MOSS-Image itself.
  const fs::path gguf = dir.path() / "moss.gguf";
  write_text(gguf, std::string("GGUF\x03\0\0\0", 8) + std::string(64, '\0'));
  expect_refused([&] { check_supported_dit_file(gguf); }, {"found a GGUF file"});
  expect_refused([&] { check_supported_text_encoder_file(gguf); }, {"found a GGUF file"});
  const fs::path zip = dir.path() / "diffusion_pytorch_model.bin";
  write_text(zip, std::string("PK\x03\x04", 4) + std::string(64, '\0'));
  expect_refused([&] { check_supported_dit_file(zip); }, {"found a PyTorch pickle (.bin / .pt / .ckpt)"});
  const fs::path legacy = dir.path() / "legacy.pt";  // the pre-zip format: a pickle PROTO opcode first
  write_text(legacy, std::string("\x80\x02\x8a\x0a", 4) + std::string(64, '\0'));
  expect_refused([&] { check_supported_dit_file(legacy); }, {"found a PyTorch pickle"});
  expect_refused([&] { check_supported_text_encoder_file(legacy); }, {"found a PyTorch pickle"});
  // A safetensors header length whose bytes look like a container's magic (0x80 0x02: a pickle PROTO 2) is not one.
  const fs::path len640 = dir.path() / "len640.safetensors";
  {
    std::string header = R"({"img_in.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})";
    header.resize(0x280, ' ');
    mi::testing::write_raw(len640, header, 4);
  }
  expect_refused([&] { check_supported_dit_file(len640); }, {"found tensors such as 'img_in.weight')"});
  const fs::path torchao = dir.path() / "torchao.safetensors";
  {
    std::vector<TensorSpec> t;
    for (int i = 0; i < 42; ++i) {
      const std::string b = "layers." + std::to_string(i) + ".";
      t.push_back({b + "norm1.weight", DType::kBF16, {3072}});
      t.push_back({b + "q_norm.weight", DType::kBF16, {128}});
      t.push_back({b + "proj.q._weight_qdata", DType::kF8E4M3, {4, 4}});
      t.push_back({b + "proj.q._weight_scale", DType::kF32, {4, 1}});
    }
    t.push_back({"video_embedder.weight", DType::kBF16, {3072, 128}});
    t.push_back({"text_embedder.proj.weight", DType::kBF16, {3072, 4096}});
    t.push_back({"time_embedding.0.weight", DType::kBF16, {3072, 256}});
    t.push_back({"final_linear_video.weight", DType::kBF16, {128, 3072}});
    write_safetensors(torchao, t);
  }
  expect_refused([&] { check_supported_dit_file(torchao); },
                 {"found block weights stored as 'layers.0.proj.q._weight_qdata', a quantization format the engine does not read)"});
  // ComfyUI fp8_scaled (F8 `.weight`, with or without a comfy_quant config).
  expect_refused([&] { check_supported_dit_file(write_dit(dir.path() / "fp8.safetensors", 3072, 42, "model.diffusion_model.", DType::kF8E4M3)); },
                 {"found block weights stored as F8_E4M3 (FP8), a quantization format the engine does not read)"});

  EXPECT_NO_THROW(check_supported_text_encoder_file(te));
  // A smaller Qwen3.5 (e.g. a 2B / 4B encoder) is another model.
  expect_refused([&] { check_supported_text_encoder_file(write_te(dir.path() / "te4b.safetensors", 2560, 32)); },
                 {"te4b.safetensors is not a supported MOSS-Image text encoder (found norm.weight [2560] (MOSS-Image: [4096])"});
  expect_refused([&] { check_supported_text_encoder_file(write_te(dir.path() / "te24.safetensors", 4096, 24)); },
                 {"found 24 layers (MOSS-Image: 32))"});
  expect_refused([&] { check_supported_text_encoder_file(write_te(dir.path() / "tevocab.safetensors", 4096, 32, 151936)); },
                 {"found embed_tokens.weight [151936, 4] (MOSS-Image: [248320, 4096]))"});
  const fs::path te_bf16 = dir.path() / "te_bf16_2048.safetensors";
  write_safetensors(te_bf16, {{"model.language_model.embed_tokens.weight", DType::kBF16, {248320, 1}}, {"model.language_model.layers.0.input_layernorm.weight", DType::kBF16, {1}}});
  expect_refused([&] { check_supported_text_encoder_file(te_bf16); }, {"found embed_tokens.weight [248320, 1] (MOSS-Image: [248320, 4096])"});
}

// A LoRA, control adapter or distill bundle file given as a model file is named as such.
TEST(SupportedModel, AddOnFilesNameTheirRole) {
  TempDir dir("sm_roles");
  const fs::path kohya = dir.path() / "kohya.safetensors";
  write_safetensors(kohya, {{"diffusion_model.layers.0.proj.q.alpha", DType::kF32, {}}, {"diffusion_model.layers.0.proj.q.lora_down.weight", DType::kBF16, {4, 3072}},
                            {"diffusion_model.layers.0.proj.q.lora_up.weight", DType::kBF16, {3072, 4}}});
  expect_refused([&] { check_supported_dit_file(kohya); }, {"found a LoRA; load it as a LoRA"});
  expect_refused([&] { check_supported_text_encoder_file(kohya); }, {"found a LoRA; load it as a LoRA"});
  const fs::path peft = dir.path() / "peft.safetensors";
  write_safetensors(peft, {{"transformer.layers.0.proj.q.lora_A.weight", DType::kBF16, {4, 3072}}, {"transformer.layers.0.proj.q.lora_B.weight", DType::kBF16, {3072, 4}}});
  expect_refused([&] { check_supported_dit_file(peft); }, {"found a LoRA"});
  const fs::path control = dir.path() / "control.safetensors";
  write_safetensors(control, {{"layers.0.proj.q.lora_A.weight", DType::kBF16, {4, 3072}}, {"layers.0.proj.q.lora_B.weight", DType::kBF16, {3072, 4}},
                              {"video_embedder.weight", DType::kBF16, {3072, 384}}, {"video_embedder.bias", DType::kBF16, {3072}}});
  expect_refused([&] { check_supported_dit_file(control); }, {"found a control adapter", "`moss-infer generate --control`"});
  const fs::path dataspace = dir.path() / "dataspace.safetensors";
  write_safetensors(dataspace, {{"latent_mix.Q", DType::kF32, {128, 128}}, {"step_codes", DType::kF32, {8}}});
  expect_refused([&] { check_supported_dit_file(dataspace); }, {"found a distill bundle file", "`distill_dir`"});
  const fs::path time = dir.path() / "time_embedding.safetensors";
  write_safetensors(time, {{"time_embedding.0.weight", DType::kF32, {3072, 256}}, {"shared_block_modulation.bias", DType::kF32, {18432}}});
  expect_refused([&] { check_supported_dit_file(time); }, {"found a distill bundle file"});
}

// (d) A quantization format the importer does not read is named as such, before any layout check.
TEST(SupportedModel, UnknownQuantizationFormat) {
  TempDir dir("sm_quant");
  for (const char* format : {"float8_e4m3fn", "mxfp8", "gguf_q4_k"}) {
    SCOPED_TRACE(format);
    const fs::path p = dir.path() / (std::string(format) + ".safetensors");
    write_safetensors(p, {{"layers.0.proj.q.weight", DType::kF8E4M3, {8, 8}}, {"layers.0.proj.q.weight_scale", DType::kF32, {}}},
                      std::string(R"({"_quantization_metadata":"{\"layers\": {\"layers.0.proj.q\": {\"format\": \")") + format + R"(\"}}}"})");
    try {
      mi::quant::open_quant_model(p);
      ADD_FAILURE() << "expected QuantError";
    } catch (const mi::quant::QuantError& e) {
      const std::string m = e.what();
      EXPECT_NE(m.find(std::string("quantization format '") + format + "' is not supported"), std::string::npos) << m;
      EXPECT_NE(m.find("docs/integration.md#supported-models"), std::string::npos) << m;
      EXPECT_EQ(m.find("must be I8"), std::string::npos) << m;
    }
  }
}

// The real bundles and the community files the engine was verified with pass, when present; add-on files are named.
TEST(SupportedModel, RealModelsPass) {
  int checked = 0;
  for (const char* name : {"base", "sft"}) {
    const fs::path b = weights_root() / name;
    if (!fs::is_directory(b / "transformer")) continue;
    EXPECT_NO_THROW(check_supported_bundle(b)) << b;
    ++checked;
  }
  const fs::path c = "/workspace/Models/community";
  // The NVFP4 file is MOSS-Image: its refusal (NVFP4 support) comes after this check.
  for (const char* f : {"comfy-org/moss_image_sft_int8_convrot.safetensors", "toxicdog/moss_image_sft_int4_convrot.safetensors",
                        "toxicdog/moss_image_sft_turbo_int8_convrot.safetensors", "tsolful/moss_image_sft_W4A8_GPTQ.safetensors",
                        "bennydaball/moss_image_sft_nvfp4.safetensors"}) {
    if (!fs::is_regular_file(c / f)) continue;
    EXPECT_NO_THROW(check_supported_dit_file(c / f)) << f;
    ++checked;
  }
  for (const char* f : {"comfy-org/qwen3.5_9b_int8_convrot.safetensors", "comfy-org/qwen3.5_9b_w4a8_gptq.safetensors"}) {
    if (!fs::is_regular_file(c / f)) continue;
    EXPECT_NO_THROW(check_supported_text_encoder_file(c / f)) << f;
    ++checked;
  }
  const std::pair<const char*, const char*> roles[] = {
      {"akhaliq/MOSS-Image-SFT-4-steps/lora.safetensors", "found a LoRA"},
      {"axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors", "found a LoRA"},
      {"comfyanonymous/MOSS_image_MAX_ULTRA_hires_lora_test/moss_hires_lora_test_v0.1.safetensors", "found a LoRA"},
      {"comfy-org/moss_image_color_depth_control.safetensors", "found a control adapter"},
      {"lodestones/MOSS-Image-SFT-8-steps-dmd/time_embedding.safetensors", "found a distill bundle file"},
      {"lodestones/MOSS-Image-SFT-8-steps-dmd/dataspace.safetensors", "found a distill bundle file"}};
  for (const auto& [f, part] : roles) {
    if (!fs::is_regular_file(c / f)) continue;
    expect_refused([&] { check_supported_dit_file(c / f); }, {part});
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no real bundle or community file present";
}
