#include "models/qwen35_text/quant_text_model.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

#include "weights/quant/quant_codec.h"

namespace mi::qwen35_text {
namespace {

[[noreturn]] void fail(const std::string& msg) { throw quant::QuantError("qwen35_text: " + msg); }

constexpr int kConvRotGroup = 256;  // the activation rotate kernels' group
constexpr const char* kEmbed = "language_model.embed_tokens";

std::vector<float> read_f32(const quant::QuantModel& m, const weights::IndexedTensor& t) {
  const std::vector<std::byte> raw = quant::read_tensor(m, t);
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}

weights::IndexedTensor renamed(weights::IndexedTensor t, std::string name) {
  t.name = std::move(name);
  return t;
}

}  // namespace

const char* precision_name(Precision p) {
  switch (p) {
    case Precision::kBF16: return "bf16";
    case Precision::kInt8: return "int8_convrot";
    case Precision::kW4A8: return "w4a8";
  }
  return "?";
}

QuantTextModel::QuantTextModel(const std::filesystem::path& path)
    : model_(quant::open_quant_model(path, quant::ModelKind::kTextEncoder)) {
  if (model_.kind != quant::ModelKind::kTextEncoder) fail("not a text encoder file");
  std::map<std::string, weights::WeightGroup> groups;
  auto put = [&](weights::IndexedTensor t) {
    const weights::TensorRoute r = weights::classify_tensor(weights::Component::kTextEncoder, t.name);
    if (!r.exclude.empty()) fail("tensor '" + t.name + "' routes to exclusion " + r.exclude);
    if (r.group == "embed_tokens") fail("unexpected embedding tensor '" + t.name + "'");
    weights::WeightGroup& g = groups[r.group];
    if (!r.layer_kind.empty()) {
      if (!g.layer_type.empty() && g.layer_type != r.layer_kind) fail(r.group + " mixes linear_attn and self_attn tensors");
      g.layer_type = r.layer_kind;
    }
    g.tensors.push_back(std::move(t));
  };

  bool any_int8 = false, any_grouped = false;
  for (const quant::QuantLinear& l : model_.linears) {
    if (l.name == kEmbed) {
      if (l.k <= 0 || l.n <= 0) fail("bad embed_tokens shape");
      embed_ = &l;
      continue;
    }
    if (l.bias) fail("'" + l.name + "' has a bias; the Qwen3.5 linears have none");
    LinearInfo info;
    info.n = l.n;
    info.k = l.k;
    const bool convrot256 = l.spec == quant::QuantSpec::int8_convrot(kConvRotGroup);
    if (!l.spec.quantized()) {
      info.kind = LinearInfo::Kind::kBF16;
      put(renamed(l.weight, l.name + ".weight"));
    } else if (l.encoding == quant::Encoding::kRaw && convrot256) {
      info.kind = LinearInfo::Kind::kInt8;
      const std::vector<float> s = read_f32(model_, *l.scale);
      if (!quant::valid_scales(s.data(), l.n)) fail("'" + l.name + "': weight_scale holds a value that is not finite and > 0");
      put(renamed(l.weight, l.name + ".weight"));
      weights::IndexedTensor st = renamed(*l.scale, l.name + ".weight_scale");
      st.shape = {l.n};
      put(std::move(st));
      any_int8 = true;
    } else if ((l.encoding == quant::Encoding::kW4A8 || l.encoding == quant::Encoding::kW6A8) && convrot256) {
      info.kind = LinearInfo::Kind::kGrouped;
      info.bits = l.encoding == quant::Encoding::kW6A8 ? 6 : 4;
      info.group = l.quant_group;
      info.codebook = l.codebook.has_value();
      const std::vector<float> s = read_f32(model_, *l.s_channel);
      if (!quant::valid_scales(s.data(), l.n)) fail("'" + l.name + "': weight_s_channel holds a value that is not finite and > 0");
      put(renamed(l.weight, l.name + ".weight"));
      put(renamed(*l.s_rel, l.name + ".weight_s_rel"));
      weights::IndexedTensor sc = renamed(*l.s_channel, l.name + ".weight_s_channel");
      sc.shape = {l.n};
      put(std::move(sc));
      if (l.codebook) {
        for (float v : read_f32(model_, *l.codebook)) {
          if (!std::isfinite(v)) fail("'" + l.name + "': non-finite codebook entry");
        }
        put(renamed(*l.codebook, l.name + ".weight_codebook"));
      }
      max_grouped_elems_ = std::max(max_grouped_elems_, l.n * l.k);
      any_grouped = true;
    } else {
      fail("'" + l.name + "' is stored as " + quant::describe(l.spec) + " / " + quant::encoding_name(l.encoding) +
           "; the text encoder runs bf16, INT8 ConvRot256 per-row and grouped W4A8 / W6A8 ConvRot256 linears");
    }
    linears_.emplace(l.name, info);
  }
  if (embed_ == nullptr) fail("the file has no language_model.embed_tokens");

  for (const weights::IndexedTensor& t : model_.tensors) {
    if (t.dtype == weights::DType::kF32 && t.name.starts_with("language_model.layers.") &&
        weights::classify_tensor(weights::Component::kTextEncoder, t.name).group.starts_with("layers.")) {
      bf16_.emplace(t.name, quant::read_tensor_bf16(model_, t));
      bf16_shapes_.emplace(t.name, t.shape);
      continue;
    }
    if (t.dtype != weights::DType::kBF16) fail("tensor '" + t.name + "' must be BF16, got " + weights::dtype_name(t.dtype));
    put(t);
  }

  index_.bundle_dir = model_.base_dir;
  index_.files = model_.files;
  index_.excluded = model_.excluded;
  std::vector<std::pair<int, std::string>> order;
  for (const auto& [name, g] : groups) {
    if (name == "final_norm") continue;
    // classify_tensor builds every other group name as "layers.<canonical digits>".
    const std::string_view digits = std::string_view(name).substr(name.starts_with("layers.") ? 7 : name.size());
    int n = -1;
    const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), n);
    if (digits.empty() || r.ec != std::errc() || r.ptr != digits.data() + digits.size() || n < 0) {
      fail("unexpected text-encoder group '" + name + "'");
    }
    order.emplace_back(n, name);
  }
  std::sort(order.begin(), order.end());
  order.emplace_back(0, "final_norm");
  for (const auto& [n, name] : order) {
    auto it = groups.find(name);
    if (it == groups.end()) continue;
    weights::WeightGroup g = std::move(it->second);
    g.component = weights::Component::kTextEncoder;
    g.name = name;
    // File order: the readers then move each group in ascending offsets.
    std::sort(g.tensors.begin(), g.tensors.end(), [](const weights::IndexedTensor& a, const weights::IndexedTensor& b) {
      return a.file != b.file ? a.file < b.file : a.offset < b.offset;
    });
    for (const auto& t : g.tensors) {
      g.total_bytes += t.nbytes;
      g.total_params += t.numel();
    }
    index_.groups.push_back(std::move(g));
  }
  precision_ = any_grouped ? Precision::kW4A8 : any_int8 ? Precision::kInt8 : Precision::kBF16;
}

const LinearInfo* QuantTextModel::find_linear(std::string_view module) const {
  auto it = linears_.find(module);
  return it == linears_.end() ? nullptr : &it->second;
}

const std::vector<std::int64_t>& QuantTextModel::bf16_shape(std::string_view name) const {
  auto it = bf16_shapes_.find(name);
  if (it == bf16_shapes_.end()) fail("no bf16 tensor " + std::string(name));
  return it->second;
}

}  // namespace mi::qwen35_text
