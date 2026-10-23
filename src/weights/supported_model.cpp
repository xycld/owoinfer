#include "weights/supported_model.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "util/json.h"
#include "util/path.h"
#include "weights/safetensors.h"

namespace mi::weights {

namespace fs = std::filesystem;
using util::path_utf8;

namespace {

constexpr const char* kSupported = "Supported: OpenMOSS-Team/MOSS-Image-Base and -SFT; see docs/integration.md#supported-models";
constexpr std::size_t kMaxListed = 4;  // differences named in one message

[[noreturn]] void refuse(const fs::path& path, const char* what, const std::vector<std::string>& found) {
  std::string list;
  for (std::size_t i = 0; i < found.size() && i < kMaxListed; ++i) list += (i ? ", " : "") + found[i];
  if (found.size() > kMaxListed) {
    const std::size_t more = found.size() - kMaxListed;
    list += " and " + std::to_string(more) + (more == 1 ? " more difference" : " more differences");
  }
  throw UnsupportedModelError(path_utf8(path) + " is not a supported MOSS-Image " + what + " (found " + list + "). " + kSupported);
}

// A config of the bundle, or nullopt when the file does not exist. Same bounds as build_index's configs.
std::optional<json::Value> read_config(const fs::path& path) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) return std::nullopt;
  const std::uintmax_t size = fs::file_size(path, ec);
  if (ec) throw IndexError("cannot open " + path_utf8(path));
  if (size > kMaxConfigBytes) throw IndexError(path_utf8(path) + ": " + std::to_string(size) + " bytes exceeds the limit of " + std::to_string(kMaxConfigBytes));
  std::ifstream in(path, std::ios::binary);
  if (!in) throw IndexError("cannot open " + path_utf8(path));
  std::ostringstream ss;
  ss << in.rdbuf();
  try {
    json::Value v = json::parse(ss.str(), {.max_values = kMaxConfigValues});
    if (!v.is_object()) throw IndexError(path_utf8(path) + ": not a JSON object");
    return v;
  } catch (const json::ParseError& e) {
    throw IndexError(path_utf8(path) + ": " + e.what());
  }
}

// A string field, or "" when absent or not a string (class names only describe what was found).
std::string string_field(const json::Value& cfg, std::string_view key) {
  const json::Value* v = cfg.find(key);
  return v != nullptr && v->is_string() ? v->as_string() : std::string();
}

std::string number_text(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%.10g", v);
  return b;
}

struct Expected {
  const char* key;
  double value;
};

// Each numeric field must be present and equal; a difference reads "<label> <key> <found> (MOSS-Image: <expected>)".
void compare_fields(const json::Value& cfg, const std::string& label, std::initializer_list<Expected> fields,
                    std::vector<std::string>& found) {
  for (const Expected& f : fields) {
    const json::Value* v = cfg.find(f.key);
    if (v == nullptr || !v->is_number()) {
      found.push_back(label + " without " + f.key);
    } else if (v->as_double() != f.value) {
      found.push_back(label + " " + f.key + " " + number_text(v->as_double()) + " (MOSS-Image: " + number_text(f.value) + ")");
    }
  }
}

std::string shape_text(const std::vector<std::int64_t>& s) {
  std::string out = "[";
  for (std::size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

// Tensor names of a single file with a known prefix removed, and the layer numbers seen under `layers_prefix`.
struct Names {
  std::map<std::string, const TensorInfo*> tensors;
  std::set<int> layers;
  std::string first;  // first tensor name as stored, for messages
  bool has(std::string_view part) const {
    for (const auto& [name, t] : tensors) {
      if (t->name.find(part) != std::string::npos) return true;
    }
    return false;
  }
  const TensorInfo* find(const std::string& name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
  }
};

Names read_names(const SafetensorsFile& file, std::initializer_list<std::string_view> prefixes, std::string_view layers_prefix) {
  Names n;
  for (const TensorInfo& t : file.tensors()) {
    if (n.first.empty()) n.first = t.name;
    std::string_view name = t.name;
    for (std::string_view p : prefixes) {
      if (name.starts_with(p)) name.remove_prefix(p.size());
    }
    if (name.starts_with(layers_prefix)) {
      std::string_view rest = name.substr(layers_prefix.size());
      int layer = 0;
      if (take_layer_index(rest, layer)) n.layers.insert(layer);
    }
    n.tensors.emplace(std::string(name), &t);
  }
  return n;
}

struct ExpectedTensor {
  const char* name;
  std::vector<std::int64_t> shape;
};

void compare_tensors(const Names& n, std::initializer_list<ExpectedTensor> want, std::vector<std::string>& found) {
  for (const ExpectedTensor& w : want) {
    const TensorInfo* t = n.find(w.name);
    if (t == nullptr) found.push_back(std::string("no ") + w.name);
    else if (t->shape != w.shape) found.push_back(std::string(w.name) + " " + shape_text(t->shape) + " (MOSS-Image: " + shape_text(w.shape) + ")");
  }
}

// Layers 0..expected-1, each present.
void compare_layers(const Names& n, int expected, const char* what, std::vector<std::string>& found) {
  const std::string want = std::to_string(expected) + " " + what + ", 0.." + std::to_string(expected - 1);
  if (static_cast<int>(n.layers.size()) != expected) {
    found.push_back(std::to_string(n.layers.size()) + " " + what + " (MOSS-Image: " + std::to_string(expected) + ")");
  } else if (*n.layers.begin() != 0 || *n.layers.rbegin() != expected - 1) {
    found.push_back(std::to_string(expected) + " " + what + " numbered " + std::to_string(*n.layers.begin()) + ".." +
                    std::to_string(*n.layers.rbegin()) + " (MOSS-Image: " + want + ")");
  }
}

bool is_fp8(DType d) { return d == DType::kF8E4M3 || d == DType::kF8E5M2; }

// GGUF files and PyTorch pickles (zip archives, or the pre-zip format that starts with a pickle PROTO opcode) are named
// as such (SafetensorsFile would only call them malformed). A safetensors file's first 8 bytes are its header length,
// which can take any value, so its header's opening '{' at byte 8 always wins.
void refuse_other_container(const fs::path& file, const char* what) {
  char magic[9] = {};
  std::ifstream in(file, std::ios::binary);
  if (!in.read(magic, sizeof magic)) return;  // too short: SafetensorsFile reports it
  if (magic[8] == '{') return;
  const std::string_view m(magic, 4);
  const auto proto = static_cast<unsigned char>(magic[1]);
  if (m == "GGUF") refuse(file, what, {"a GGUF file (llama.cpp / ComfyUI-GGUF format); the engine reads safetensors files"});
  if (m == std::string_view("PK\x03\x04", 4) || (static_cast<unsigned char>(magic[0]) == 0x80 && proto >= 2 && proto <= 5)) {
    refuse(file, what, {"a PyTorch pickle (.bin / .pt / .ckpt); the engine reads safetensors files"});
  }
}

// A DiT add-on given where a model file is expected: a LoRA, a control adapter or a distill bundle file.
void refuse_other_role(const fs::path& file, const char* what, const Names& n) {
  const bool lora = n.has(".lora_A.") || n.has(".lora_B.") || n.has(".lora_down.") || n.has(".lora_up.") || n.has(".hada_w") ||
                    n.has(".lokr_w") || n.has(".alpha");
  const TensorInfo* embedder = n.find("video_embedder.weight");
  if (lora && embedder != nullptr && embedder->shape.size() == 2 && embedder->shape[1] > 128) {
    refuse(file, what, {"a control adapter (a LoRA with a widened video_embedder); load it as the control adapter (`moss-infer generate --control`)"});
  }
  if (lora) refuse(file, what, {"a LoRA; load it as a LoRA (`loras`, `--lora`)"});
  const bool dataspace = n.find("step_codes") != nullptr || n.has("latent_mix.");
  const bool time_embedding = n.find("time_embedding.0.weight") != nullptr && n.layers.empty() && embedder == nullptr;
  if (dataspace || time_embedding) {
    refuse(file, what, {"a distill bundle file; pass the bundle's folder as the distill bundle (`distill_dir` / `--distill`)"});
  }
}

}  // namespace

void check_supported_bundle(const fs::path& dir) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) throw IndexError(path_utf8(dir) + ": not a directory");

  // Class names only describe what was found: a folder whose architecture fields all match is accepted whatever its
  // classes are called. model_index.json naming another pipeline does make a missing config "another model".
  std::string other_pipeline;
  if (const std::optional<json::Value> mi = read_config(dir / "model_index.json")) {
    const std::string cls = string_field(*mi, "_class_name");
    if (!cls.empty() && cls != "MOSSImagePipeline") {
      other_pipeline = cls;
      const json::Value* t = mi->find("transformer");
      if (t != nullptr && t->is_array() && t->as_array().size() == 2 && t->as_array()[1].is_string()) {
        other_pipeline += " with transformer " + t->as_array()[1].as_string();
      }
    }
  }
  auto required = [&](const char* rel) {
    std::optional<json::Value> cfg = read_config(dir / rel);
    if (cfg) return *std::move(cfg);
    if (!other_pipeline.empty()) refuse(dir, "model", {other_pipeline});
    throw IndexError(path_utf8(dir) + ": no " + rel + " (incomplete download, or not the model folder?)");
  };

  // Per component: its differing fields, described by its class name when that is another one too.
  std::vector<std::string> found;
  bool other_classes = false;
  auto add = [&](const std::string& other_class, std::vector<std::string> diffs) {
    if (diffs.empty()) return;
    if (!other_class.empty()) {
      diffs = {other_class};
      other_classes = true;
    }
    found.insert(found.end(), diffs.begin(), diffs.end());
  };

  const json::Value dit = required("transformer/config.json");
  {
    std::vector<std::string> diffs;
    compare_fields(dit, "transformer", {{"dim", 3072}, {"ffn_dim", 8192}, {"freq_dim", 256}, {"latent_channels", 128},
                                        {"num_heads", 24}, {"num_layers", 42}, {"text_dim", 4096}}, diffs);
    if (dit.find("rope_base") != nullptr) compare_fields(dit, "transformer", {{"rope_base", 10000}}, diffs);
    const std::string cls = string_field(dit, "_class_name");
    add(!cls.empty() && cls != "MOSSImageTransformer" ? "transformer " + cls : "", std::move(diffs));
  }

  const json::Value te = required("text_encoder/config.json");
  {
    std::vector<std::string> diffs;
    const json::Value* tc = te.find("text_config");
    if (tc == nullptr || !tc->is_object()) {
      diffs.push_back("a text encoder config without text_config");
    } else {
      compare_fields(*tc, "text encoder", {{"vocab_size", 248320}, {"hidden_size", 4096}, {"num_hidden_layers", 32},
                                           {"intermediate_size", 12288}, {"num_attention_heads", 16}, {"num_key_value_heads", 4},
                                           {"head_dim", 256}, {"linear_num_key_heads", 16}, {"linear_num_value_heads", 32},
                                           {"linear_key_head_dim", 128}, {"linear_value_head_dim", 128}, {"linear_conv_kernel_dim", 4}},
                     diffs);
      // Three linear-attention layers, then one full-attention layer, eight times.
      const json::Value* lt = tc->find("layer_types");
      bool pattern = lt != nullptr && lt->is_array() && lt->as_array().size() == 32;
      for (std::size_t i = 0; pattern && i < 32; ++i) {
        const json::Value& v = lt->as_array()[i];
        pattern = v.is_string() && v.as_string() == (i % 4 == 3 ? "full_attention" : "linear_attention");
      }
      if (!pattern) diffs.push_back("text encoder layer_types other than MOSS-Image's (3 linear + 1 full attention, 8 times)");
    }
    std::string cls;
    if (const std::string type = string_field(te, "model_type"); !type.empty() && type != "qwen3_5") {
      const json::Value* arch = te.find("architectures");
      const bool named = arch != nullptr && arch->is_array() && !arch->as_array().empty() && arch->as_array()[0].is_string();
      cls = "text encoder " + (named ? arch->as_array()[0].as_string() : type);
    }
    add(cls, std::move(diffs));
  }

  if (const std::optional<json::Value> vae = read_config(dir / "vae" / "config.json")) {
    std::vector<std::string> diffs;
    if (vae->find("latent_channels") != nullptr) compare_fields(*vae, "VAE", {{"latent_channels", 32}}, diffs);
    const std::string cls = string_field(*vae, "_class_name");
    add(!cls.empty() && cls != "AutoencoderKLFlux2" ? "VAE " + cls : "", std::move(diffs));
  }

  // Another pipeline's name leads; with other component classes too, it says enough on its own.
  if (found.empty()) return;
  if (!other_pipeline.empty()) {
    if (other_classes) found.clear();
    found.insert(found.begin(), other_pipeline);
  }
  refuse(dir, "model", found);
}

void check_supported_dit_file(const fs::path& file) {
  refuse_other_container(file, "DiT");
  const SafetensorsFile st(file);
  const Names n = read_names(st, {"model.diffusion_model.", "diffusion_model."}, "layers.");
  if (n.tensors.empty()) refuse(file, "DiT", {"no tensors"});
  refuse_other_role(file, "DiT", n);
  if (n.find("video_embedder.weight") == nullptr && n.find("layers.0.norm1.weight") == nullptr) {
    const bool te = n.has("language_model.");
    refuse(file, "DiT", {std::string(te ? "text-encoder tensors" : "tensors") + " such as '" + n.first + "'"});
  }
  std::vector<std::string> found;
  // Kept in bf16 by every supported format (INT8 / INT4 ConvRot, W4A8 / W6A8 GPTQ, mixed and bf16 files).
  compare_tensors(n, {{"video_embedder.weight", {3072, 128}}, {"text_embedder.proj.weight", {3072, 4096}},
                      {"time_embedding.0.weight", {3072, 256}}, {"final_linear_video.weight", {128, 3072}},
                      {"layers.0.norm1.weight", {3072}}, {"layers.0.q_norm.weight", {128}}},
                  found);
  compare_layers(n, 42, "blocks", found);
  // The GEMM weights under the names and types the importer reads (`.weight`, plus its quantization tensors): other
  // quantization tools store them differently (torchao's `._weight_qdata` / `._weight_scale`, FP8 `.weight`).
  const TensorInfo* q = n.find("layers.0.proj.q.weight");
  if (q == nullptr) q = n.find("layers.0.proj.qkv.weight");
  const TensorInfo* down = n.find("layers.0.proj.ffn_down.weight");
  if (q == nullptr || down == nullptr) {
    std::string example;
    for (const auto& [name, t] : n.tensors) {
      if (name.starts_with("layers.0.proj.") && !name.ends_with(".weight")) {
        example = name;
        break;
      }
    }
    found.push_back(example.empty() ? std::string("no layers.0.proj.q.weight / ffn_down.weight")
                                    : "block weights stored as '" + example + "', a quantization format the engine does not read");
  } else if (is_fp8(q->dtype) || is_fp8(down->dtype)) {
    found.push_back("block weights stored as " + std::string(dtype_name(is_fp8(q->dtype) ? q->dtype : down->dtype)) +
                    " (FP8), a quantization format the engine does not read");
  }
  if (!found.empty()) refuse(file, "DiT", found);
}

void check_supported_text_encoder_file(const fs::path& file) {
  refuse_other_container(file, "text encoder");
  const SafetensorsFile st(file);
  // Qwen3_5Model (`[model.]language_model.*`) and text-only (`model.*`) names alike; the importer explains the layout.
  const Names n = read_names(st, {"model.", "language_model."}, "layers.");
  if (n.tensors.empty()) refuse(file, "text encoder", {"no tensors"});
  refuse_other_role(file, "text encoder", n);
  if (n.has("video_embedder.")) refuse(file, "text encoder", {"DiT tensors such as '" + n.first + "'"});
  const TensorInfo* embed = n.find("embed_tokens.weight");
  if (embed == nullptr && n.layers.empty()) refuse(file, "text encoder", {"tensors such as '" + n.first + "'"});
  std::vector<std::string> found;
  // Rows of the embedding table are the vocabulary in every format (quantized tables keep them); a float table also
  // shows the hidden size.
  if (embed == nullptr) {
    found.push_back("no embed_tokens.weight");
  } else {
    const bool rows = !embed->shape.empty() && embed->shape[0] == 248320;
    const bool is_float = embed->dtype == DType::kBF16 || embed->dtype == DType::kF16 || embed->dtype == DType::kF32;
    if (!rows || (is_float && embed->shape != std::vector<std::int64_t>{248320, 4096})) {
      found.push_back("embed_tokens.weight " + shape_text(embed->shape) + " (MOSS-Image: [248320, 4096])");
    }
  }
  compare_tensors(n, {{"norm.weight", {4096}}, {"layers.3.self_attn.q_norm.weight", {256}}}, found);
  compare_layers(n, 32, "layers", found);
  if (const TensorInfo* down = n.find("layers.0.mlp.down_proj.weight"); down != nullptr && is_fp8(down->dtype)) {
    found.push_back("layer weights stored as " + std::string(dtype_name(down->dtype)) + " (FP8), a quantization format the engine does not read");
  }
  if (!found.empty()) refuse(file, "text encoder", found);
}

}  // namespace mi::weights
