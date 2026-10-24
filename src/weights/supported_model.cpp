#include "weights/supported_model.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "util/json.h"
#include "util/path.h"
#include "weights/file_reader.h"
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

// A config of the bundle, or nullopt when the file does not exist. Same bounds as build_index's configs. Anything that
// exists is opened through FileReader, whose open refuses what is not a regular file (a directory, a FIFO, a device:
// IndexError "<path>: not a regular file"): a type check before the open could be raced by a file replaced between
// the check and the open, and a FIFO swapped in would then block the open.
std::optional<json::Value> read_config(const fs::path& path) {
  std::error_code ec;
  if (!fs::exists(path, ec)) return std::nullopt;
  std::string text;
  try {
    const FileReader in(path);
    if (in.size() > kMaxConfigBytes) {
      throw IndexError(path_utf8(path) + ": " + std::to_string(in.size()) + " bytes exceeds the limit of " +
                       std::to_string(kMaxConfigBytes));
    }
    text.resize(static_cast<std::size_t>(in.size()));
    in.read_at(0, text.data(), text.size());
  } catch (const IoError& e) {
    throw IndexError(e.what());
  }
  try {
    json::Value v = json::parse(text, {.max_values = kMaxConfigValues});
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
  std::string name;
  std::vector<std::int64_t> shape;
};

void compare_tensors(const Names& n, std::initializer_list<ExpectedTensor> want, std::vector<std::string>& found) {
  for (const ExpectedTensor& w : want) {
    const TensorInfo* t = n.find(w.name);
    if (t == nullptr) found.push_back("no " + w.name);
    else if (t->shape != w.shape) found.push_back(w.name + " " + shape_text(t->shape) + " (MOSS-Image: " + shape_text(w.shape) + ")");
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
  const FileReader in(file);  // IoError for a FIFO or a device, before any read could block
  if (in.size() < sizeof magic) return;  // too short: SafetensorsFile reports it
  in.read_at(0, magic, sizeof magic);
  if (magic[8] == '{') return;
  const std::string_view m(magic, 4);
  const auto proto = static_cast<unsigned char>(magic[1]);
  if (m == "GGUF") refuse(file, what, {"a GGUF file (llama.cpp / ComfyUI-GGUF format); the engine reads safetensors files"});
  if (m == std::string_view("PK\x03\x04", 4) || (static_cast<unsigned char>(magic[0]) == 0x80 && proto >= 2 && proto <= 5)) {
    refuse(file, what, {"a PyTorch pickle (.bin / .pt / .ckpt); the engine reads safetensors files"});
  }
}

// A DiT add-on given where a model file is expected: a LoRA, a control adapter or a distill bundle file.
void refuse_other_role(const SafetensorsFile& st, const char* what, const Names& n) {
  const fs::path& file = st.path();
  const bool lora = looks_like_lora(st);
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

bool looks_like_lora(const SafetensorsFile& file) {
  // The factors only: a LoRA's `.alpha` always sits next to its module's factors, and an alpha of its own is another
  // model's parameter (Snake activations' `.act.alpha`, `.alphas`).
  return std::any_of(file.tensors().begin(), file.tensors().end(), [](const TensorInfo& t) {
    for (std::string_view part : {".lora_A.", ".lora_B.", ".lora_down.", ".lora_up.", ".lora.down.", ".lora.up.",
                                  ".lora_linear_layer.", ".hada_w", ".lokr_w"}) {
      if (t.name.find(part) != std::string::npos) return true;
    }
    return false;
  });
}

const std::vector<DitLinear>& moss_dit_linears() {
  static const std::vector<DitLinear> table = [] {
    constexpr std::int64_t kDim = 3072, kFfn = 8192, kTextDim = 4096, kFreqDim = 256, kLatent = 128;
    constexpr int kLayers = 42, kTextBlocks = 2;
    std::vector<DitLinear> t;
    const auto add = [&](std::string name, std::int64_t n, std::int64_t k, bool bias, int block = -1, int gemm = 0) {
      t.push_back({std::move(name), n, k, bias, block, gemm});
    };
    for (int b = 0; b < kLayers; ++b) {
      const std::string p = "layers." + std::to_string(b) + ".proj.";
      add(p + "q", kDim, kDim, false, b, 0);
      add(p + "k", kDim, kDim, false, b, 1);
      add(p + "v", kDim, kDim, false, b, 2);
      add(p + "o", kDim, kDim, false, b, 3);
      add(p + "ffn_up_gate", 2 * kFfn, kDim, false, b, 4);
      add(p + "ffn_down", kDim, kFfn, false, b, 5);
    }
    add("text_embedder.proj", kDim, kTextDim, true);
    for (int b = 0; b < kTextBlocks; ++b) {
      const std::string p = "text_embedder.blocks." + std::to_string(b) + ".";
      add(p + "qkv", 3 * kDim, kDim, true);
      add(p + "proj", kDim, kDim, true);
      add(p + "ffn_gate", kFfn, kDim, true);
      add(p + "ffn_up", kFfn, kDim, true);
      add(p + "ffn_down", kDim, kFfn, true);
    }
    add("time_embedding.0", kDim, kFreqDim, true);
    add("time_embedding.2", kDim, kDim, true);
    add("shared_block_modulation", 6 * kDim, kDim, true);
    add("video_head_modulation", 2 * kDim, kDim, true);
    add("video_embedder", kDim, kLatent, true);
    add("final_linear_video", kLatent, kDim, false);
    std::sort(t.begin(), t.end(), [](const DitLinear& a, const DitLinear& b) { return a.name < b.name; });
    return t;
  }();
  return table;
}

const DitLinear* find_moss_dit_linear(std::string_view module) {
  const std::vector<DitLinear>& t = moss_dit_linears();
  const auto it =
      std::lower_bound(t.begin(), t.end(), module, [](const DitLinear& l, std::string_view m) { return l.name < m; });
  return it != t.end() && it->name == module ? &*it : nullptr;
}

const std::vector<TeTensor>& qwen35_text_tensors() {
  static const std::vector<TeTensor> table = [] {
    constexpr std::int64_t kVocab = 248320, kHidden = 4096, kInter = 12288;
    constexpr std::int64_t kHeads = 16, kKvHeads = 4, kHeadDim = 256;                // full attention
    constexpr std::int64_t kKHeads = 16, kVHeads = 32, kLinHeadDim = 128, kConv = 4;  // GatedDeltaNet
    constexpr std::int64_t kKeyDim = kKHeads * kLinHeadDim, kValueDim = kVHeads * kLinHeadDim;
    constexpr std::int64_t kConvDim = 2 * kKeyDim + kValueDim;
    constexpr int kLayers = 32;
    std::vector<TeTensor> t;
    t.push_back({"language_model.embed_tokens.weight", {kVocab, kHidden}, true});
    t.push_back({"language_model.norm.weight", {kHidden}, false});
    for (int i = 0; i < kLayers; ++i) {
      const std::string p = "language_model.layers." + std::to_string(i) + ".";
      const auto add = [&](const char* name, std::vector<std::int64_t> shape, bool linear) {
        t.push_back({p + name, std::move(shape), linear});
      };
      add("input_layernorm.weight", {kHidden}, false);
      add("post_attention_layernorm.weight", {kHidden}, false);
      add("mlp.gate_proj.weight", {kInter, kHidden}, true);
      add("mlp.up_proj.weight", {kInter, kHidden}, true);
      add("mlp.down_proj.weight", {kHidden, kInter}, true);
      if (i % 4 == 3) {
        add("self_attn.q_proj.weight", {2 * kHeads * kHeadDim, kHidden}, true);
        add("self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden}, true);
        add("self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden}, true);
        add("self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim}, true);
        add("self_attn.q_norm.weight", {kHeadDim}, false);
        add("self_attn.k_norm.weight", {kHeadDim}, false);
      } else {
        add("linear_attn.A_log", {kVHeads}, false);
        add("linear_attn.dt_bias", {kVHeads}, false);
        add("linear_attn.conv1d.weight", {kConvDim, 1, kConv}, false);
        add("linear_attn.in_proj_qkv.weight", {kConvDim, kHidden}, true);
        add("linear_attn.in_proj_z.weight", {kValueDim, kHidden}, true);
        add("linear_attn.in_proj_b.weight", {kVHeads, kHidden}, true);
        add("linear_attn.in_proj_a.weight", {kVHeads, kHidden}, true);
        add("linear_attn.norm.weight", {kLinHeadDim}, false);
        add("linear_attn.out_proj.weight", {kHidden, kValueDim}, true);
      }
    }
    std::sort(t.begin(), t.end(), [](const TeTensor& a, const TeTensor& b) { return a.name < b.name; });
    return t;
  }();
  return table;
}

const TeTensor* find_qwen35_text_tensor(std::string_view name) {
  const std::vector<TeTensor>& t = qwen35_text_tensors();
  const auto it =
      std::lower_bound(t.begin(), t.end(), name, [](const TeTensor& e, std::string_view n) { return e.name < n; });
  return it != t.end() && it->name == name ? &*it : nullptr;
}

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
  check_supported_dit_file(SafetensorsFile(file));
}

void check_supported_dit_file(const SafetensorsFile& st) {
  const fs::path& file = st.path();
  const Names n = read_names(st, {"model.diffusion_model.", "diffusion_model."}, "layers.");
  if (n.tensors.empty()) refuse(file, "DiT", {"no tensors"});
  refuse_other_role(st, "DiT", n);
  if (n.find("video_embedder.weight") == nullptr && n.find("layers.0.norm1.weight") == nullptr) {
    const bool te = n.has("language_model.");
    refuse(file, "DiT", {std::string(te ? "text-encoder tensors" : "tensors") + " such as '" + n.first + "'"});
  }
  std::vector<std::string> found;
  // Kept in bf16 by every supported format (INT8 / INT4 ConvRot, W4A8 / W6A8 GPTQ, mixed and bf16 files).
  const auto linear = [](const char* module) {
    const DitLinear* l = find_moss_dit_linear(module);
    return ExpectedTensor{std::string(module) + ".weight", {l->n, l->k}};
  };
  compare_tensors(n,
                  {linear("video_embedder"), linear("text_embedder.proj"), linear("time_embedding.0"),
                   linear("final_linear_video"), {"layers.0.norm1.weight", {3072}}, {"layers.0.q_norm.weight", {128}}},
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
  check_supported_text_encoder_file(SafetensorsFile(file));
}

void check_supported_text_encoder_file(const SafetensorsFile& st) {
  const fs::path& file = st.path();
  // Qwen3_5Model (`[model.]language_model.*`) and text-only (`model.*`) names alike; the importer explains the layout.
  const Names n = read_names(st, {"model.", "language_model."}, "layers.");
  if (n.tensors.empty()) refuse(file, "text encoder", {"no tensors"});
  refuse_other_role(st, "text encoder", n);
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
