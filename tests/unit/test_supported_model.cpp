// Identification of supported models (weights/supported_model.h) and of unsupported quantization formats
// (quant::open_quant_model): other architectures, a MOSS-Image folder with a changed config, single files with other
// shapes and an unknown format are refused with a plain message; MOSS-Image configs and files (official, or a
// fine-tune with the same architecture) pass. Configs and headers only: tiny synthetic files, plus the real bundles
// and community files when present. Also the importer's DiT linear shapes (hostile shapes refused before any byte is
// read; sparse full-size files), the pipeline's device-free DiT file inspection and FIFOs given as model files.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>

#include <cerrno>
#endif

#include "models/lora/lora.h"
#include "models/qwen35_text/quant_text_model.h"
#include "pipeline/pipeline.h"
#include "test_util.h"
#include "util/json.h"
#include "weights/quant/quant_model.h"
#include "weights/supported_model.h"

namespace {

using namespace mi::weights;
namespace fs = std::filesystem;
using mi::testing::SparseTensor;
using mi::testing::TempDir;
using mi::testing::TensorSpec;
using mi::testing::write_safetensors;
using mi::testing::write_sparse_safetensors;

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

// ---- DiT linear shapes (quant::open_quant_model): sparse files, holes for every weight ----

SparseTensor comfy_quant(const std::string& module, const std::string& json) {
  return {module + ".comfy_quant", DType::kU8, {static_cast<std::int64_t>(json.size())}, json};
}

// The full-size cases below write sparse files of up to 5 GB, which NTFS files are not by default: Linux / macOS only.
#ifndef _WIN32
constexpr const char* kInt8ConvRot = R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})";

// F32 bytes holding `n` copies of `v`: stored scales the importer accepts.
std::string f32_bytes(std::int64_t n, float v) {
  std::string out(static_cast<std::size_t>(n) * sizeof(float), '\0');
  for (std::int64_t i = 0; i < n; ++i) std::memcpy(out.data() + i * sizeof(float), &v, sizeof v);
  return out;
}

// An INT8 ConvRot linear as Comfy-Org stores it: I8 [n, k] (a hole), F32 [n, 1] scales (ones when `scales`, else a
// hole) and its comfy_quant config.
void add_int8(std::vector<SparseTensor>& ts, const std::string& m, std::int64_t n, std::int64_t k,
              bool scales = false) {
  ts.push_back({m + ".weight", DType::kI8, {n, k}, ""});
  ts.push_back({m + ".weight_scale", DType::kF32, {n, 1}, scales ? f32_bytes(n, 1.0f) : ""});
  ts.push_back(comfy_quant(m, kInt8ConvRot));
}

// A whole MOSS-Image DiT file (the bf16 tensors check_supported_dit_file reads, 42 blocks of Comfy-Org INT8 linears
// with fused qkv and stored scales of 1), with `edit` applied to the tensor list: a sparse file of 5 GB.
fs::path write_int8_dit(const fs::path& path, const std::function<void(std::vector<SparseTensor>&)>& edit = {}) {
  std::vector<SparseTensor> ts = {{"video_embedder.weight", DType::kBF16, {3072, 128}, ""},
                                  {"text_embedder.proj.weight", DType::kBF16, {3072, 4096}, ""},
                                  {"time_embedding.0.weight", DType::kBF16, {3072, 256}, ""},
                                  {"final_linear_video.weight", DType::kBF16, {128, 3072}, ""}};
  for (int b = 0; b < 42; ++b) {
    const std::string p = "layers." + std::to_string(b) + ".";
    ts.push_back({p + "norm1.weight", DType::kBF16, {3072}, ""});
    ts.push_back({p + "q_norm.weight", DType::kBF16, {128}, ""});
    add_int8(ts, p + "proj.qkv", 9216, 3072, true);
    add_int8(ts, p + "proj.o", 3072, 3072, true);
    add_int8(ts, p + "proj.ffn_up_gate", 16384, 3072, true);
    add_int8(ts, p + "proj.ffn_down", 3072, 8192, true);
  }
  if (edit) edit(ts);
  write_sparse_safetensors(path, ts);
  return path;
}

// What open_quant_model(path, kind) throws ("" when it returns).
std::string quant_error(const fs::path& path, mi::quant::ModelKind kind = mi::quant::ModelKind::kDiT) {
  try {
    mi::quant::open_quant_model(path, kind);
  } catch (const mi::quant::QuantError& e) {
    return e.what();
  }
  return {};
}
#endif

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

// The single-file checks on a header parsed once give the path forms' answers; looks_like_lora names adapter files.
TEST(SupportedModel, ParsedHeaderOverloadsAndLoraDetection) {
  TempDir dir("sm_parsed");
  const fs::path ok = write_dit(dir.path() / "ok.safetensors", 3072, 42, "model.diffusion_model.");
  EXPECT_NO_THROW(check_supported_dit_file(SafetensorsFile(ok)));
  const fs::path te = write_te(dir.path() / "te.safetensors", 4096, 32);
  EXPECT_NO_THROW(check_supported_text_encoder_file(SafetensorsFile(te)));
  const auto message = [](const std::function<void()>& fn) {
    try {
      fn();
    } catch (const UnsupportedModelError& e) {
      return std::string(e.what());
    }
    return std::string("accepted");
  };
  const fs::path d2048 = write_dit(dir.path() / "d2048.safetensors", 2048, 42);
  const fs::path b40 = write_dit(dir.path() / "b40.safetensors", 3072, 40);
  for (const fs::path& f : {d2048, b40, te}) {
    SCOPED_TRACE(mi::util::path_utf8(f.filename()));
    const std::string by_path = message([&] { check_supported_dit_file(f); });
    EXPECT_NE(by_path, "accepted");
    EXPECT_EQ(message([&] { check_supported_dit_file(SafetensorsFile(f)); }), by_path);
  }
  const fs::path te4b = write_te(dir.path() / "te4b.safetensors", 2560, 32);
  EXPECT_EQ(message([&] { check_supported_text_encoder_file(SafetensorsFile(te4b)); }),
            message([&] { check_supported_text_encoder_file(te4b); }));
  EXPECT_EQ(message([&] { check_supported_text_encoder_file(SafetensorsFile(ok)); }),
            message([&] { check_supported_text_encoder_file(ok); }));

  const fs::path kohya = dir.path() / "kohya.safetensors";
  write_safetensors(kohya, {{"lora_unet_layers_0_proj_q.alpha", DType::kF32, {}},
                            {"lora_unet_layers_0_proj_q.lora_down.weight", DType::kBF16, {4, 3072}},
                            {"lora_unet_layers_0_proj_q.lora_up.weight", DType::kBF16, {3072, 4}}});
  const fs::path peft = dir.path() / "peft.safetensors";
  write_safetensors(peft, {{"transformer.layers.0.proj.q.lora_A.weight", DType::kBF16, {4, 3072}},
                           {"transformer.layers.0.proj.q.lora_B.weight", DType::kBF16, {3072, 4}}});
  const fs::path loha = dir.path() / "loha.safetensors";
  write_safetensors(loha, {{"layers.0.proj.q.hada_w1_a", DType::kBF16, {3072, 2}},
                           {"layers.0.proj.q.hada_w1_b", DType::kBF16, {2, 3072}}});
  const fs::path control = dir.path() / "control.safetensors";
  write_safetensors(control, {{"layers.0.proj.q.lora_A.weight", DType::kBF16, {4, 3072}},
                              {"layers.0.proj.q.lora_B.weight", DType::kBF16, {3072, 4}},
                              {"video_embedder.weight", DType::kBF16, {3072, 384}}});
  for (const fs::path& f : {kohya, peft, loha, control}) EXPECT_TRUE(looks_like_lora(SafetensorsFile(f))) << f;
  const fs::path dataspace = dir.path() / "dataspace.safetensors";
  write_safetensors(dataspace, {{"latent_mix.Q", DType::kF32, {128, 128}}, {"step_codes", DType::kF32, {8}}});
  for (const fs::path& f : {ok, te, dataspace}) EXPECT_FALSE(looks_like_lora(SafetensorsFile(f))) << f;
  expect_refused([&] { check_supported_dit_file(SafetensorsFile(control)); }, {"found a control adapter"});
}

// The one table of the DiT's linear shapes is what the LoRA targets list, block GEMMs included.
TEST(SupportedModel, DitLinearTableIsTheLoraTargets) {
  const std::vector<DitLinear>& table = moss_dit_linears();
  EXPECT_EQ(table.size(), 42u * 6 + 1 + 2 * 5 + 6);
  const mi::lora::TargetModel targets = mi::lora::moss_dit_targets();
  ASSERT_EQ(targets.modules.size(), table.size());
  for (std::size_t i = 0; i < table.size(); ++i) {
    const mi::lora::TargetModule& t = targets.modules[i];
    EXPECT_EQ(t.name, table[i].name);
    EXPECT_EQ(t.n, table[i].n) << t.name;
    EXPECT_EQ(t.k, table[i].k) << t.name;
    EXPECT_EQ(t.has_bias, table[i].has_bias) << t.name;
    EXPECT_EQ(t.block, table[i].block) << t.name;
    if (t.block >= 0) {
      EXPECT_EQ(std::string(mi::lora::block_gemm_name(t.gemm)), t.name.substr(t.name.rfind('.') + 1));
    }
  }
  const DitLinear* down = find_moss_dit_linear("layers.41.proj.ffn_down");
  ASSERT_NE(down, nullptr);
  EXPECT_EQ(down->n, 3072);
  EXPECT_EQ(down->k, 8192);
  EXPECT_EQ(find_moss_dit_linear("layers.0.proj.qkv"), nullptr);
  EXPECT_EQ(find_moss_dit_linear("layers.42.proj.q"), nullptr);
  EXPECT_EQ(find_moss_dit_linear("layers.0.norm1"), nullptr);
}

// A hostile DiT file: an I8 [N, 1] weight with an F32 [N] scale (N = 2^28: a 1 GiB scale) is refused by its shape
// before the scale is read. The data are holes, so a read would have cost the 1 GiB (and failed differently: zeros are
// not valid scales).
TEST(DitLinearShapes, HostileScaleIsRefusedBeforeItsBytesAreRead) {
  TempDir dir("dls_hostile");
#ifdef _WIN32
  constexpr std::int64_t kN = std::int64_t{1} << 16;  // NTFS files are not sparse by default
#else
  constexpr std::int64_t kN = std::int64_t{1} << 28;
#endif
  const fs::path p = dir.path() / "hostile.safetensors";
  write_sparse_safetensors(p, {{"layers.0.proj.q.weight", DType::kI8, {kN, 1}, ""},
                               {"layers.0.proj.q.weight_scale", DType::kF32, {kN}, ""},
                               comfy_quant("layers.0.proj.q", R"({"format": "int8_tensorwise"})")});
  const std::string want =
      "tensor 'layers.0.proj.q.weight' I8 [" + std::to_string(kN) + ", 1] (MOSS-Image: [3072, 3072] as int8_plain)";
  try {
    mi::quant::to_model_index(mi::quant::open_quant_model(p, mi::quant::ModelKind::kDiT));
    ADD_FAILURE() << "the hostile file was indexed";
  } catch (const mi::quant::QuantError& e) {
    EXPECT_NE(std::string(e.what()).find(want), std::string::npos) << e.what();
  }
#ifndef _WIN32
  // In a whole MOSS-Image DiT (a sparse file of 5 GB), through the pipeline's file inspection: a model error naming
  // the file.
  const fs::path whole = write_int8_dit(dir.path() / "whole.safetensors", [&](std::vector<SparseTensor>& ts) {
    for (SparseTensor& t : ts) {
      if (t.name == "layers.0.proj.qkv.weight") t.shape = {kN, 1};
      if (t.name == "layers.0.proj.qkv.weight_scale") t = {t.name, DType::kF32, {kN}, ""};
      // Without ConvRot any K passes the layout checks (a ConvRot group must divide K).
      if (t.name == "layers.0.proj.qkv.comfy_quant") {
        t = comfy_quant("layers.0.proj.qkv", R"({"format": "int8_tensorwise"})");
      }
    }
  });
  ASSERT_NO_THROW(check_supported_dit_file(whole));
  try {
    mi::pipeline::inspect_dit_file(whole, "");
    ADD_FAILURE() << "the hostile file was inspected";
  } catch (const UnsupportedModelError& e) {
    ADD_FAILURE() << e.what();
  } catch (const IndexError& e) {
    const std::string m = e.what();
    EXPECT_EQ(m.rfind(mi::util::path_utf8(whole), 0), 0u) << m;
    const std::string part =
        "tensor 'layers.0.proj.qkv.weight' I8 [" + std::to_string(kN) + ", 1] (MOSS-Image: [9216, 3072] as int8_plain)";
    EXPECT_NE(m.find(part), std::string::npos) << m;
  }
#endif
}

#ifndef _WIN32
// Weights of other shapes, per layout (the expected shape as the layout stores it), linears the DiT does not have and
// scale tensors that do not fit [n, k] are refused, all from the header (full-size sparse files).
TEST(DitLinearShapes, OtherShapesAndScalesAreRefused) {
  TempDir dir("dls_refused");
  int i = 0;
  const auto check = [&](std::vector<SparseTensor> ts, const std::string& want) {
    const fs::path p = dir.path() / ("case" + std::to_string(i++) + ".safetensors");
    write_sparse_safetensors(p, std::move(ts));
    const std::string m = quant_error(p);
    EXPECT_NE(m.find(want), std::string::npos) << "'" << want << "' not in: " << m;
  };
  const std::string w4a4 = R"({"format": "convrot_w4a4", "convrot_groupsize": 256})";
  const std::string nvfp4 = R"({"format": "nvfp4"})";
  check({{"layers.0.proj.ffn_down.weight", DType::kI8, {3072, 3072}, ""},
         {"layers.0.proj.ffn_down.weight_scale", DType::kF32, {3072, 1}, ""},
         comfy_quant("layers.0.proj.ffn_down", kInt8ConvRot)},
        "tensor 'layers.0.proj.ffn_down.weight' I8 [3072, 3072] (MOSS-Image: [3072, 8192] as int8_convrot256)");
  check({{"model.diffusion_model.layers.3.proj.qkv.weight", DType::kI8, {6144, 3072}, ""},
         {"model.diffusion_model.layers.3.proj.qkv.weight_scale", DType::kF32, {6144, 1}, ""},
         comfy_quant("model.diffusion_model.layers.3.proj.qkv", kInt8ConvRot)},
        "tensor 'layers.3.proj.qkv.weight' I8 [6144, 3072] (MOSS-Image: [9216, 3072] as int8_convrot256)");
  check({{"layers.0.proj.o.weight", DType::kI8, {3072, 3072}, ""},
         {"layers.0.proj.o.weight_scale", DType::kF32, {3072}, ""},
         comfy_quant("layers.0.proj.o", w4a4)},
        "tensor 'layers.0.proj.o.weight' I8 [3072, 3072] (MOSS-Image: [3072, 1536] as w4a4)");
  check({{"layers.0.proj.q.weight", DType::kBF16, {3072, 1024}, ""}},
        "tensor 'layers.0.proj.q.weight' BF16 [3072, 1024] (MOSS-Image: [3072, 3072] as bf16)");
  check({{"text_embedder.proj.weight", DType::kI8, {3072, 3072}, ""},
         {"text_embedder.proj.weight_scale", DType::kF32, {3072}, ""},
         comfy_quant("text_embedder.proj", kInt8ConvRot)},
        "tensor 'text_embedder.proj.weight' I8 [3072, 3072] (MOSS-Image: [3072, 4096] as int8_convrot256)");
  check({{"layers.0.norm1.weight", DType::kI8, {3072, 1}, ""}, {"layers.0.norm1.weight_scale", DType::kF32, {3072}, ""},
         comfy_quant("layers.0.norm1", R"({"format": "int8_tensorwise"})")},
        "'layers.0.norm1' is not a linear of the MOSS-Image DiT (tensor 'layers.0.norm1.weight' I8 [3072, 1])");
  // Scales that do not fit the canonical [n, k].
  check({{"layers.0.proj.q.weight", DType::kI8, {3072, 3072}, ""},
         {"layers.0.proj.q.weight_scale", DType::kF32, {3071}, ""},
         comfy_quant("layers.0.proj.q", kInt8ConvRot)},
        "weight_scale must be F32 [N], [N, 1] or a scalar, got F32 [3071]");
  check({{"layers.0.proj.q.weight", DType::kI8, {3072, 1536}, ""},
         {"layers.0.proj.q.weight_scale", DType::kF32, {3072, 2}, ""},
         comfy_quant("layers.0.proj.q", w4a4)},
        "convrot_w4a4 weight_scale must be F32 [N]");
  check({{"layers.0.proj.ffn_up_gate.weight", DType::kI8, {16384, 1536}, ""},
         {"layers.0.proj.ffn_up_gate.weight_s_rel", DType::kF8E4M3, {16384, 100}, ""},
         {"layers.0.proj.ffn_up_gate.weight_s_channel", DType::kF32, {16384}, ""},
         comfy_quant("layers.0.proj.ffn_up_gate", R"({"format": "asym_w4a8_int8", "group_size": 16})")},
        "weight_s_rel must be F8_E4M3 [16384, 192], got F8_E4M3 [16384, 100]");
  check({{"layers.0.proj.o.weight", DType::kU8, {3072, 1536}, ""},
         {"layers.0.proj.o.weight_scale", DType::kF8E4M3, {3072, 96}, ""},
         {"layers.0.proj.o.weight_scale_2", DType::kF32, {}, ""},
         comfy_quant("layers.0.proj.o", nvfp4)},
        "nvfp4 weight_scale must be F8_E4M3 [3072, 192] (the swizzled block scales), got F8_E4M3 [3072, 96]");
  check({{"layers.0.proj.o.weight", DType::kU8, {3072, 3072}, ""},
         {"layers.0.proj.o.weight_scale", DType::kF8E4M3, {3072, 384}, ""},
         {"layers.0.proj.o.weight_scale_2", DType::kF32, {}, ""},
         comfy_quant("layers.0.proj.o", nvfp4)},
        "tensor 'layers.0.proj.o.weight' U8 [3072, 3072] (MOSS-Image: [3072, 1536] as nvfp4)");
}

// Every layout the engine reads, at MOSS-Image shapes (Comfy-Org's fused INT8, separate q/k/v, INT4, W4A8 / W6A8,
// NVFP4, a scalar scale, mixed bf16, a quantized non-block linear), passes; nothing but the configs is read.
TEST(DitLinearShapes, MossImageLayoutsPass) {
  TempDir dir("dls_ok");
  int i = 0;
  const auto accepted = [&](const std::vector<SparseTensor>& ts) {
    const fs::path p = dir.path() / ("case" + std::to_string(i++) + ".safetensors");
    write_sparse_safetensors(p, ts);
    EXPECT_EQ(quant_error(p), "") << p;
    return mi::quant::open_quant_model(p, mi::quant::ModelKind::kDiT);
  };
  std::vector<SparseTensor> comfy;
  using Gemm = std::tuple<const char*, std::int64_t, std::int64_t>;
  for (const auto& [m, n, k] :
       {Gemm{"proj.qkv", 9216, 3072}, Gemm{"proj.o", 3072, 3072}, Gemm{"proj.ffn_up_gate", 16384, 3072},
        Gemm{"proj.ffn_down", 3072, 8192}}) {
    add_int8(comfy, std::string("model.diffusion_model.layers.7.") + m, n, k);
  }
  const mi::quant::QuantModel fused = accepted(comfy);
  ASSERT_NE(fused.find_linear("layers.7.proj.v"), nullptr);
  EXPECT_EQ(fused.find_linear("layers.7.proj.v")->n, 3072);

  const std::string w4a4_config = R"({"format": "convrot_w4a4", "convrot_groupsize": 256})";
  std::vector<SparseTensor> separate, w4a4, w4a8, mixed;
  for (const char* m : {"q", "k", "v"}) {
    add_int8(separate, std::string("layers.0.proj.") + m, 3072, 3072);
    w4a4.push_back({std::string("layers.0.proj.") + m + ".weight", DType::kI8, {3072, 1536}, ""});
    w4a4.push_back({std::string("layers.0.proj.") + m + ".weight_scale", DType::kF32, {3072}, ""});
    w4a4.push_back(comfy_quant(std::string("layers.0.proj.") + m, w4a4_config));
    mixed.push_back({std::string("layers.0.proj.") + m + ".weight", DType::kBF16, {3072, 3072}, ""});
  }
  accepted(separate);
  EXPECT_EQ(accepted(w4a4).find_linear("layers.0.proj.k")->k, 3072);
  add_int8(mixed, "layers.0.proj.o", 3072, 3072);
  mixed.push_back({"layers.1.proj.qkv.weight", DType::kBF16, {9216, 3072}, ""});
  mixed.push_back({"layers.1.proj.ffn_down.weight", DType::kBF16, {3072, 8192}, ""});
  accepted(mixed);

  w4a8 = {{"layers.0.proj.qkv.weight", DType::kI8, {9216, 1536}, ""},
          {"layers.0.proj.qkv.weight_s_rel", DType::kF8E4M3, {9216, 192}, ""},
          {"layers.0.proj.qkv.weight_s_channel", DType::kF32, {9216}, ""},
          {"layers.0.proj.qkv.weight_codebook", DType::kF32, {16}, ""},
          comfy_quant("layers.0.proj.qkv", R"({"format": "asym_w4a8_int8", "group_size": 16})"),
          {"layers.0.proj.ffn_down.weight", DType::kI8, {3072, 6144}, ""},
          {"layers.0.proj.ffn_down.weight_s_rel", DType::kF8E4M3, {3072, 512}, ""},
          {"layers.0.proj.ffn_down.weight_s_channel", DType::kF32, {3072}, ""},
          comfy_quant("layers.0.proj.ffn_down", R"({"format": "w6a8_int8", "group_size": 16})")};
  accepted(w4a8);
  accepted({{"layers.0.proj.qkv.weight", DType::kU8, {9216, 1536}, ""},
            {"layers.0.proj.qkv.weight_scale", DType::kF8E4M3, {9216, 192}, ""},
            {"layers.0.proj.qkv.weight_scale_2", DType::kF32, {}, ""},
            comfy_quant("layers.0.proj.qkv", R"({"format": "nvfp4"})")});
  accepted({{"layers.0.proj.ffn_up_gate.weight", DType::kI8, {16384, 3072}, ""},
            {"layers.0.proj.ffn_up_gate.weight_scale", DType::kF32, {}, ""},
            comfy_quant("layers.0.proj.ffn_up_gate", R"({"format": "int8_tensorwise"})")});
  accepted({{"text_embedder.proj.weight", DType::kI8, {3072, 4096}, ""},
            {"text_embedder.proj.weight_scale", DType::kF32, {3072, 1}, ""},
            {"text_embedder.proj.bias", DType::kBF16, {3072}, ""},
            comfy_quant("text_embedder.proj", kInt8ConvRot)});
}
#endif

#ifndef _WIN32
// The pipeline's device-free view of a DiT file: an INT8 file's index, precision and linears summary, as open_dit_file
// takes them (the stored scales are read and validated; the weights are holes of a sparse file, which NTFS files are
// not by default).
TEST(DitFileInspect, Int8FileAndMixedFile) {
  TempDir dir("dfi_int8");
  const fs::path p = write_int8_dit(dir.path() / "int8.safetensors");
  int asked = 0;  // the NVFP4 reason (a device query in the pipeline) is asked for NVFP4 files only
  const mi::pipeline::DitFileInfo info = mi::pipeline::inspect_dit_file(p, [&] {
    ++asked;
    return std::string("NVFP4 is not supported here");
  });
  EXPECT_EQ(asked, 0);
  ASSERT_NE(info.quant, nullptr);
  EXPECT_TRUE(info.refusal.empty()) << info.refusal;
  EXPECT_FALSE(info.nvfp4);
  EXPECT_EQ(info.precision, mi::planner::Precision::kInt8);
  EXPECT_EQ(info.linears, "int8_convrot256");
  EXPECT_EQ(info.quant->specs.size(), 42u * 6);
  EXPECT_EQ(*info.quant->find_spec("layers.5.proj.k"), mi::quant::QuantSpec::int8_convrot(256));
  EXPECT_NE(info.quant->index.find_group(mi::weights::Component::kTransformer, "block.41"), nullptr);

  // Block 0's ffn_down kept in bf16.
  const fs::path m = write_int8_dit(dir.path() / "mixed.safetensors", [](std::vector<SparseTensor>& ts) {
    std::erase_if(ts, [](const SparseTensor& t) { return t.name.starts_with("layers.0.proj.ffn_down."); });
    ts.push_back({"layers.0.proj.ffn_down.weight", DType::kBF16, {3072, 8192}, ""});
  });
  const mi::pipeline::DitFileInfo mixed = mi::pipeline::inspect_dit_file(m, "");
  EXPECT_EQ(mixed.precision, mi::planner::Precision::kInt8);
  EXPECT_EQ(mixed.linears, "mixed (1 bf16, 251 int8_convrot256)");
  // Not a regular file, or no file: the refusals open_dit_file gives.
  const auto refusal = [](const fs::path& f) {
    try {
      mi::pipeline::inspect_dit_file(f, "");
    } catch (const std::invalid_argument& e) {
      return std::string(e.what());
    }
    return std::string("accepted");
  };
  EXPECT_EQ(refusal(dir.path()), mi::util::path_utf8(dir.path()) + ": not a regular file");
  const fs::path missing = dir.path() / "missing.safetensors";
  EXPECT_EQ(refusal(missing), "DiT file not found: " + mi::util::path_utf8(missing));
}
#endif

// The community DiT files and their load results, when present: the importer accepts every one, and the inspection
// gives the precision the pipeline loads them with (W4A8 needs converting first; NVFP4 is refused by the reason given).
TEST(DitFileInspect, RealCommunityFiles) {
  const fs::path c = "/workspace/Models/community";
  struct Case {
    const char* file;
    mi::planner::Precision precision;
    const char* linears;  // "" : refused by to_model_index (needs decoding)
  };
  const Case cases[] = {
      {"comfy-org/moss_image_sft_int8_convrot.safetensors", mi::planner::Precision::kInt8, "int8_convrot256"},
      {"comfy-org/moss_image_base_int8_convrot.safetensors", mi::planner::Precision::kInt8, "int8_convrot256"},
      {"toxicdog/moss_image_sft_turbo_int8_convrot.safetensors", mi::planner::Precision::kInt8, "int8_convrot256"},
      {"toxicdog/moss_image_sft_int4_convrot.safetensors", mi::planner::Precision::kInt4, "int4_convrot256"},
      {"tsolful/moss_image_sft_W4A8_GPTQ.safetensors", mi::planner::Precision::kBf16, ""}};
  int checked = 0;
  for (const Case& k : cases) {
    const fs::path f = c / k.file;
    if (!fs::is_regular_file(f)) continue;
    SCOPED_TRACE(k.file);
    EXPECT_NO_THROW(mi::quant::open_quant_model(f, mi::quant::ModelKind::kDiT));
    if (*k.linears == '\0') {
      try {
        mi::pipeline::inspect_dit_file(f, "");
        ADD_FAILURE() << "indexed";
      } catch (const IndexError& e) {
        EXPECT_NE(std::string(e.what()).find("needs decoding"), std::string::npos) << e.what();
      }
    } else {
      const mi::pipeline::DitFileInfo info = mi::pipeline::inspect_dit_file(f, "no NVFP4");
      EXPECT_EQ(info.precision, k.precision);
      EXPECT_EQ(info.linears, k.linears);
    }
    ++checked;
  }
  const fs::path nvfp4 = c / "bennydaball/moss_image_sft_nvfp4.safetensors";
  if (fs::is_regular_file(nvfp4)) {
    EXPECT_NO_THROW(mi::quant::open_quant_model(nvfp4, mi::quant::ModelKind::kDiT));
    const mi::pipeline::DitFileInfo info = mi::pipeline::inspect_dit_file(nvfp4, "no NVFP4");
    EXPECT_TRUE(info.nvfp4);
    EXPECT_EQ(info.quant, nullptr);
    EXPECT_EQ(info.refusal, mi::util::path_utf8(nvfp4) + ": no NVFP4");
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no community DiT file present";
}

// The text encoder's tensor table is the official bundle's text encoder: the same names and shapes, which follow from
// its config (text_config).
TEST(TextEncoderShapes, TableIsTheBundlesTextEncoder) {
  const std::vector<TeTensor>& table = qwen35_text_tensors();
  EXPECT_EQ(table.size(), 2u + 32 * 5 + 24 * 9 + 8 * 6);
  const TeTensor* q = find_qwen35_text_tensor("language_model.layers.3.self_attn.q_proj.weight");
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(q->linear);
  EXPECT_EQ(find_qwen35_text_tensor("language_model.layers.2.self_attn.q_proj.weight"), nullptr);  // a GDN layer
  EXPECT_EQ(find_qwen35_text_tensor("language_model.layers.32.input_layernorm.weight"), nullptr);
  const fs::path te = weights_root() / "sft" / "text_encoder";
  if (!fs::is_regular_file(te / "config.json")) GTEST_SKIP() << "no MOSS-Image bundle at " << te.parent_path();
  // The shapes the config gives.
  std::ifstream in(te / "config.json", std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const mi::json::Value cfg = mi::json::parse(text);
  const mi::json::Value& tc = *cfg.find("text_config");
  const auto num = [&](const char* key) { return static_cast<std::int64_t>(tc.find(key)->as_double()); };
  const std::int64_t hidden = num("hidden_size"), heads = num("num_attention_heads"), head_dim = num("head_dim");
  const std::int64_t key_dim = num("linear_num_key_heads") * num("linear_key_head_dim");
  const std::int64_t value_dim = num("linear_num_value_heads") * num("linear_value_head_dim");
  const auto shape_of = [](const char* name) { return find_qwen35_text_tensor(name)->shape; };
  EXPECT_EQ(shape_of("language_model.embed_tokens.weight"), (std::vector<std::int64_t>{num("vocab_size"), hidden}));
  using Shape = std::vector<std::int64_t>;
  EXPECT_EQ(shape_of("language_model.layers.7.self_attn.q_proj.weight"), (Shape{2 * heads * head_dim, hidden}));
  EXPECT_EQ(shape_of("language_model.layers.7.self_attn.k_proj.weight"),
            (std::vector<std::int64_t>{num("num_key_value_heads") * head_dim, hidden}));
  EXPECT_EQ(shape_of("language_model.layers.0.mlp.down_proj.weight"), (Shape{hidden, num("intermediate_size")}));
  EXPECT_EQ(shape_of("language_model.layers.0.linear_attn.conv1d.weight"),
            (std::vector<std::int64_t>{2 * key_dim + value_dim, 1, num("linear_conv_kernel_dim")}));
  EXPECT_EQ(shape_of("language_model.layers.0.linear_attn.out_proj.weight"), (Shape{hidden, value_dim}));
  // Every language_model tensor of the bundle, by name and shape (the vision tower is not the encoder's).
  std::map<std::string, std::vector<std::int64_t>> bundle;
  for (const auto& e : fs::directory_iterator(te)) {
    if (e.path().extension() != ".safetensors") continue;
    const SafetensorsFile shard(e.path());
    for (const TensorInfo& t : shard.tensors()) {
      if (t.name.starts_with("language_model.")) bundle.emplace(t.name, t.shape);
    }
  }
  ASSERT_EQ(bundle.size(), table.size());
  for (const TeTensor& t : table) {
    const auto it = bundle.find(t.name);
    ASSERT_NE(it, bundle.end()) << t.name;
    EXPECT_EQ(it->second, t.shape) << t.name;
  }
}

// A hostile text-encoder file: a quantized linear with an I8 [N, 1] weight and an F32 [N] scale, or an F32 layer tensor
// of N values, is refused by its shape before the text encoder reads it whole (holes: a read would cost N * 4 bytes).
TEST(TextEncoderShapes, HostileTensorsAreRefusedBeforeTheyAreRead) {
  TempDir dir("tes_hostile");
#ifdef _WIN32
  constexpr std::int64_t kN = std::int64_t{1} << 16;
#else
  constexpr std::int64_t kN = std::int64_t{1} << 28;
#endif
  const std::string up = "model.language_model.layers.0.mlp.up_proj";
  const fs::path linear = dir.path() / "linear.safetensors";
  write_sparse_safetensors(linear, {{up + ".weight", DType::kI8, {kN, 1}, ""},
                                    {up + ".weight_scale", DType::kF32, {kN}, ""},
                                    comfy_quant(up, R"({"format": "int8_tensorwise"})")});
  const fs::path f32 = dir.path() / "f32.safetensors";
  write_sparse_safetensors(f32, {{"model.language_model.layers.0.linear_attn.A_log", DType::kF32, {kN}, ""}});
  const std::pair<fs::path, std::string> cases[] = {
      {linear, "tensor 'language_model.layers.0.mlp.up_proj.weight' I8 [" + std::to_string(kN) +
                   ", 1] (MOSS-Image: [12288, 4096] as int8_plain)"},
      {f32, "tensor 'language_model.layers.0.linear_attn.A_log' F32 [" + std::to_string(kN) + "] (MOSS-Image: [32])"}};
  for (const auto& [file, want] : cases) {
    SCOPED_TRACE(mi::util::path_utf8(file.filename()));
    // The importer, and the text encoder model built on it (which reads the scales and the F32 tensors).
    const auto kind = mi::quant::ModelKind::kTextEncoder;
    const std::function<void()> opens[] = {[&] { mi::quant::open_quant_model(file, kind); },
                                           [&] { mi::qwen35_text::QuantTextModel m(file); }};
    for (const auto& open : opens) {
      try {
        open();
        ADD_FAILURE() << "the hostile file was read";
      } catch (const mi::quant::QuantError& e) {
        EXPECT_NE(std::string(e.what()).find(want), std::string::npos) << e.what();
      }
    }
  }
}

#ifndef _WIN32
// Text-encoder linears and tensors of other shapes, names it does not have and scales that do not fit are refused;
// Comfy-Org's layouts (INT8 ConvRot, W4A8 GPTQ with F32 GDN tensors) and the bf16 layout pass. Full-size sparse files.
TEST(TextEncoderShapes, OtherShapesAreRefusedLayoutsPass) {
  TempDir dir("tes_shapes");
  int i = 0;
  const auto write = [&](const std::vector<SparseTensor>& ts) {
    const fs::path p = dir.path() / ("case" + std::to_string(i++) + ".safetensors");
    write_sparse_safetensors(p, ts);
    return p;
  };
  const auto refused = [&](const std::vector<SparseTensor>& ts, const std::string& want) {
    const std::string m = quant_error(write(ts), mi::quant::ModelKind::kTextEncoder);
    EXPECT_NE(m.find(want), std::string::npos) << "'" << want << "' not in: " << m;
  };
  const std::string p = "model.language_model.layers.3.";
  refused({{p + "self_attn.q_proj.weight", DType::kI8, {4096, 4096}, ""},
           {p + "self_attn.q_proj.weight_scale", DType::kF32, {4096, 1}, ""},
           comfy_quant(p + "self_attn.q_proj", kInt8ConvRot)},
          "tensor 'language_model.layers.3.self_attn.q_proj.weight' I8 [4096, 4096] "
          "(MOSS-Image: [8192, 4096] as int8_convrot256)");
  refused({{p + "mlp.down_proj.weight", DType::kI8, {4096, 12288}, ""},
           {p + "mlp.down_proj.weight_scale", DType::kF32, {4095}, ""},
           comfy_quant(p + "mlp.down_proj", kInt8ConvRot)},
          "weight_scale must be F32 [N], [N, 1] or a scalar, got F32 [4095]");
  refused({{p + "mlp.up_proj.weight", DType::kBF16, {12288, 1024}, ""}},
          "tensor 'language_model.layers.3.mlp.up_proj.weight' BF16 [12288, 1024] (MOSS-Image: [12288, 4096] as bf16)");
  // Layer 3 is a full-attention layer: GatedDeltaNet tensors are not among its tensors.
  refused({{p + "linear_attn.in_proj_z.weight", DType::kBF16, {4096, 4096}, ""}},
          "'language_model.layers.3.linear_attn.in_proj_z' is not a linear of the MOSS-Image text encoder");
  refused({{"model.language_model.layers.0.linear_attn.conv1d.weight", DType::kBF16, {8192, 1, 3}, ""}},
          "tensor 'language_model.layers.0.linear_attn.conv1d.weight' BF16 [8192, 1, 3] (MOSS-Image: [8192, 1, 4])");
  refused({{"model.language_model.layers.3.self_attn.q_norm.weight", DType::kBF16, {256}, ""},
           {"model.language_model.layers.3.self_attn.rope.weight", DType::kBF16, {64}, ""}},
          "tensor 'language_model.layers.3.self_attn.rope.weight' BF16 [64] is not a tensor of the MOSS-Image text "
          "encoder");

  const std::string l0 = "model.language_model.layers.0.";
  const fs::path comfy = write({{"model.language_model.embed_tokens.weight", DType::kI8, {248320, 4096}, ""},
                                {"model.language_model.embed_tokens.weight_scale", DType::kF32, {248320, 1}, ""},
                                comfy_quant("model.language_model.embed_tokens", kInt8ConvRot),
                                {l0 + "linear_attn.in_proj_qkv.weight", DType::kI8, {8192, 4096}, ""},
                                {l0 + "linear_attn.in_proj_qkv.weight_scale", DType::kF32, {8192, 1}, ""},
                                comfy_quant(l0 + "linear_attn.in_proj_qkv", kInt8ConvRot),
                                {l0 + "linear_attn.in_proj_a.weight", DType::kBF16, {32, 4096}, ""},
                                {l0 + "linear_attn.A_log", DType::kF32, {32}, ""},
                                {l0 + "linear_attn.norm.weight", DType::kF32, {128}, ""},
                                {l0 + "linear_attn.conv1d.weight", DType::kBF16, {8192, 1, 4}, ""},
                                {"model.language_model.norm.weight", DType::kBF16, {4096}, ""},
                                {"lm_head.weight", DType::kBF16, {16, 16}, ""}});
  EXPECT_EQ(quant_error(comfy, mi::quant::ModelKind::kTextEncoder), "");
  const fs::path w4a8 = write({{p + "mlp.down_proj.weight", DType::kI8, {4096, 6144}, ""},
                               {p + "mlp.down_proj.weight_s_rel", DType::kF8E4M3, {4096, 768}, ""},
                               {p + "mlp.down_proj.weight_s_channel", DType::kF32, {4096}, ""},
                               {p + "mlp.down_proj.weight_codebook", DType::kF32, {16}, ""},
                               comfy_quant(p + "mlp.down_proj", R"({"format": "asym_w4a8_int8", "group_size": 16})"),
                               {p + "self_attn.k_proj.weight", DType::kBF16, {1024, 4096}, ""}});
  EXPECT_EQ(quant_error(w4a8, mi::quant::ModelKind::kTextEncoder), "");
}
#endif

// The real text encoders load as before: the bundle's bf16 encoder (importer) and Comfy-Org's INT8 and W4A8 files
// (importer and the text encoder model, which reads their scales and F32 tensors).
TEST(TextEncoderShapes, RealTextEncodersPass) {
  int checked = 0;
  const fs::path bundle = weights_root() / "sft" / "text_encoder";
  if (fs::is_directory(bundle)) {
    EXPECT_NO_THROW(mi::quant::open_quant_model(bundle, mi::quant::ModelKind::kTextEncoder));
    ++checked;
  }
  for (const char* f : {"qwen3.5_9b_int8_convrot.safetensors", "qwen3.5_9b_w4a8_gptq.safetensors"}) {
    const fs::path file = fs::path("/workspace/Models/community/comfy-org") / f;
    if (!fs::is_regular_file(file)) continue;
    EXPECT_NO_THROW(mi::qwen35_text::QuantTextModel m(file)) << f;
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no real text encoder present";
}

#ifndef _WIN32
// A FIFO given as a model file (or as a bundle config) is refused at once, not waited on; so is a directory.
TEST(SupportedModel, FifoAndDirectoryAreRefusedPromptly) {
  TempDir dir("sm_fifo");
  const fs::path fifo = dir.path() / "moss.safetensors";
  ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0) << std::strerror(errno);
  const std::string not_regular = mi::util::path_utf8(fifo) + ": not a regular file";
  const std::pair<const char*, std::function<void()>> calls[] = {
      {"check_supported_dit_file", [&] { check_supported_dit_file(fifo); }},
      {"check_supported_text_encoder_file", [&] { check_supported_text_encoder_file(fifo); }}};
  for (const auto& [name, call] : calls) {
    const std::optional<std::string> e = mi::testing::error_within(fifo, call);
    ASSERT_TRUE(e.has_value()) << name << " blocked on a FIFO";
    EXPECT_EQ(*e, not_regular) << name;
  }
  const std::function<void()> other_calls[] = {[&] { mi::quant::open_quant_model(fifo, mi::quant::ModelKind::kDiT); },
                                               [&] { mi::pipeline::inspect_dit_file(fifo, ""); }};
  for (const auto& call : other_calls) {
    const std::optional<std::string> e = mi::testing::error_within(fifo, call);
    ASSERT_TRUE(e.has_value()) << "blocked on a FIFO";
    EXPECT_NE(*e, "");
  }
  try {
    check_supported_dit_file(dir.path());
    ADD_FAILURE() << "a directory passed";
  } catch (const IoError& e) {
    EXPECT_EQ(std::string(e.what()), mi::util::path_utf8(dir.path()) + ": not a regular file");
  }
  // A bundle whose transformer config is a FIFO.
  Folder f("sm_fifo_bundle");
  f.write(kModelIndex, "", text_encoder_config(), kVae);
  fs::create_directories(f.root() / "transformer");
  const fs::path config = f.root() / "transformer/config.json";
  ASSERT_EQ(::mkfifo(config.c_str(), 0600), 0) << std::strerror(errno);
  const std::optional<std::string> e = mi::testing::error_within(config, [&] { check_supported_bundle(f.root()); });
  ASSERT_TRUE(e.has_value()) << "check_supported_bundle blocked on a FIFO config";
  EXPECT_EQ(*e, mi::util::path_utf8(config) + ": not a regular file");
}
#endif
