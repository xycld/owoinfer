// open_gguf_dit (quant_model.h): a MOSS-Image GGUF DiT as a QuantModel for the converter.
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "util/file_time.h"
#include "util/path.h"
#include "weights/gguf.h"
#include "weights/quant/quant_model.h"
#include "weights/supported_model.h"
#include "weights/test_seams.h"

namespace mi::quant {

using weights::DType;
using weights::IndexedTensor;

namespace {

[[noreturn]] void fail(const std::string& msg) { throw QuantError(msg); }

constexpr int kLayers = 42, kTextBlocks = 2;  // transformer/config.json num_layers; the text refiner's blocks
constexpr std::int64_t kDim = 3072, kHeadDim = 128, kRopeBands = 16;

struct Expected {
  std::vector<std::int64_t> shape;
  bool optional = false;
};

// Every DiT tensor that is not a block GEMM, with its shape: the other linears' weights and biases, the norms and the
// rope.bands buffer (derived; the engine rebuilds it, so a file may leave it out).
const std::map<std::string, Expected>& kept_tensors() {
  static const std::map<std::string, Expected> table = [] {
    std::map<std::string, Expected> m;
    for (const weights::DitLinear& l : weights::moss_dit_linears()) {
      if (l.block >= 0) continue;
      m[l.name + ".weight"] = {{l.n, l.k}};
      if (l.has_bias) m[l.name + ".bias"] = {{l.n}};
    }
    const auto norms = [&](const std::string& p) {
      m[p + "norm1.weight"] = {{kDim}};
      m[p + "norm2.weight"] = {{kDim}};
      m[p + "q_norm.weight"] = {{kHeadDim}};
      m[p + "k_norm.weight"] = {{kHeadDim}};
    };
    for (int b = 0; b < kLayers; ++b) norms("layers." + std::to_string(b) + ".");
    for (int b = 0; b < kTextBlocks; ++b) norms("text_embedder.blocks." + std::to_string(b) + ".");
    m["final_norm_video.weight"] = {{kDim}};
    m["rope.bands"] = {{kRopeBands}, true};
    return m;
  }();
  return table;
}

std::string shape_text(const std::vector<std::int64_t>& s) {
  std::string out = "[";
  for (std::size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

// At most a few items of a list, then "and N more".
std::string listed(const std::vector<std::string>& items) {
  constexpr std::size_t kMax = 4;
  std::string out;
  for (std::size_t i = 0; i < items.size() && i < kMax; ++i) out += (i ? ", " : "") + items[i];
  if (items.size() > kMax) out += " and " + std::to_string(items.size() - kMax) + " more";
  return out;
}

}  // namespace

QuantModel open_gguf_dit(const weights::GgufFile& file) {
  const std::string where = util::path_utf8(file.path());
  const auto refuse = [&](const std::string& msg) { fail(where + ": " + msg); };
  QuantModel model;
  model.kind = ModelKind::kDiT;
  model.base_dir = file.path().parent_path();
  model.files.push_back({util::path_utf8(file.path().filename()), file.file_size(), util::file_mtime_ns(file.path())});

  // Custom quantization stored in GGUF (ComfyUI-GGUF's comfy.gguf.quant.* configs over I8 / U8 tensors, e.g. molbal's
  // Q8_CR and Q4_CR_L): no ggml type, so named as such.
  for (const auto& [key, value] : file.metadata()) {
    if (key.starts_with("comfy.gguf.quant.")) {
      refuse("custom quantization in GGUF (" + key + " = " + value.substr(0, 120) +
             "), not a ggml type; GGUF DiTs are read in Q8_0, Q6_K, Q5_K, Q4_K, Q4_0, BF16, F16 and F32");
    }
  }
  const bool any_shapes = weights::testing::any_model_shapes();
  std::vector<std::string> unknown, bad_shape;
  std::map<std::string, std::pair<int, std::string>> unsupported;  // type -> (tensors, first name)
  std::set<std::string> seen;
  std::vector<QuantLinear> linears;
  for (const weights::GgufTensorInfo& t : file.tensors()) {
    std::string_view name = t.name;
    for (std::string_view p : {"model.diffusion_model.", "diffusion_model."}) {
      if (name.starts_with(p)) {
        name.remove_prefix(p.size());
        break;
      }
    }
    const std::string canon(name);
    if (!seen.insert(canon).second) refuse("two tensors map to the name '" + canon + "'");
    // A block GEMM (fused or separate q/k/v), or one of the kept tensors. The test seam takes any `layers.N.proj.*`
    // 2-D weight as a GEMM of its own shape, and any other tensor as kept.
    const weights::DitLinear* gemm = nullptr;
    bool fused = false, is_gemm = false;
    std::string module;
    if (canon.starts_with("layers.") && canon.ends_with(".weight")) {
      module = canon.substr(0, canon.size() - 7);
      fused = module.ends_with(".proj.qkv");
      gemm = weights::find_moss_dit_linear(fused ? module.substr(0, module.size() - 3) + "q" : module);
      if (gemm != nullptr && gemm->block < 0) gemm = nullptr;
      is_gemm = gemm != nullptr || (any_shapes && module.find(".proj.") != std::string::npos && t.shape.size() == 2 &&
                                    (!fused || t.shape[0] % 3 == 0));
    }
    std::vector<std::int64_t> want;
    const auto kept = kept_tensors().find(canon);
    if (any_shapes) {
      want = t.shape;
    } else if (gemm != nullptr) {
      want = {fused ? 3 * gemm->n : gemm->n, gemm->k};
    } else if (kept != kept_tensors().end()) {
      want = kept->second.shape;
    } else {
      unknown.push_back("'" + t.name + "'");
      continue;
    }
    if (t.shape != want) {
      bad_shape.push_back("'" + t.name + "' " + shape_text(t.shape) + " (MOSS-Image: " + shape_text(want) + ")");
      continue;
    }
    if (!weights::gguf_dequantizable(t.type)) {
      std::string type(weights::ggml_type_name(t.type));
      auto& u = unsupported[type];
      if (u.first++ == 0) u.second = t.name;
      continue;
    }
    if (!is_gemm) {
      const DType out = canon == "rope.bands" ? DType::kF32 : DType::kBF16;
      model.tensors.push_back({canon, out, t.shape, 0, t.offset, t.nbytes});
      model.gguf_types[canon] = t.type;
      continue;
    }
    model.gguf_types[t.name] = t.type;
    const std::int64_t n = fused ? t.shape[0] / 3 : t.shape[0], k = t.shape[1];
    const std::uint64_t part_bytes = t.nbytes / (fused ? 3 : 1);
    const std::string base = module.substr(0, module.size() - (fused ? 3 : 0));
    for (int p = 0; p < (fused ? 3 : 1); ++p) {
      QuantLinear l;
      l.name = fused ? base + "qkv"[p] : module;
      l.spec = QuantSpec::bf16();
      l.encoding = Encoding::kGguf;
      l.n = n;
      l.k = k;
      l.weight = {t.name, DType::kBF16, {n, k}, 0, t.offset + static_cast<std::uint64_t>(p) * part_bytes, part_bytes};
      l.fused_slice = fused;
      l.source_module = std::string(t.name.substr(0, t.name.size() - 7));
      linears.push_back(std::move(l));
    }
  }
  if (!unknown.empty()) {
    refuse("GGUF tensors " + listed(unknown) +
           " are not MOSS-Image DiT tensors (GGUF DiTs are read under the official names, optionally prefixed "
           "model.diffusion_model.)");
  }
  if (!bad_shape.empty()) refuse("GGUF tensors of another shape: " + listed(bad_shape));
  if (!unsupported.empty()) {
    std::vector<std::string> types;
    for (const auto& [type, u] : unsupported) {
      types.push_back(type + " (" + std::to_string(u.first) + " tensors, e.g. '" + u.second + "')");
    }
    refuse("GGUF tensors of types the engine does not read: " + listed(types) +
           "; GGUF DiTs are read in Q8_0, Q6_K, Q5_K, Q4_K, Q4_0, BF16, F16 and F32");
  }
  std::sort(linears.begin(), linears.end(), [](const QuantLinear& a, const QuantLinear& b) { return a.name < b.name; });
  for (std::size_t i = 1; i < linears.size(); ++i) {
    if (linears[i].name == linears[i - 1].name) refuse("linear '" + linears[i].name + "' is defined twice (fused and separate)");
  }
  std::set<std::string> linear_names;
  for (const QuantLinear& l : linears) linear_names.insert(l.name);
  std::vector<std::string> missing;
  for (const weights::DitLinear& l : weights::moss_dit_linears()) {
    if (l.block >= 0 && !linear_names.count(l.name)) missing.push_back("'" + l.name + ".weight'");
  }
  for (const auto& [name, e] : kept_tensors()) {
    if (!e.optional && !seen.count(name)) missing.push_back("'" + name + "'");
  }
  if (!missing.empty() && !any_shapes) refuse("missing MOSS-Image DiT tensors " + listed(missing));
  model.linears = std::move(linears);
  std::sort(model.tensors.begin(), model.tensors.end(), [](const IndexedTensor& a, const IndexedTensor& b) { return a.name < b.name; });
  return model;
}

}  // namespace mi::quant
