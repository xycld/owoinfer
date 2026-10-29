// LoRA file parsing, validation and the MOSS DiT target table (see lora.h).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <utility>

#include "models/lora/lora.h"
#include "util/checked_math.h"
#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_convert.h"
#include "weights/supported_model.h"

namespace mi::lora {

namespace {

using weights::DType;
using weights::FormatError;

[[noreturn]] void fail(const std::string& msg) { throw LoraError("mi::lora: " + msg); }

// MOSS-Image DiT config (transformer/config.json).
constexpr std::int64_t kDim = 3072;
constexpr int kLayers = 42;

// The part of a module a tensor holds.
enum class Part {
  kDown, kUp, kAlpha, kHadaW1A, kHadaW1B, kHadaW2A, kHadaW2B, kLokrW1, kLokrW1A, kLokrW1B, kLokrW2, kLokrW2A, kLokrW2B,
  kDiff, kDiffB, kUnsupported, kDora
};

struct SuffixRule {
  std::string_view suffix;
  Part part;
  const char* why = nullptr;  // kUnsupported / kDora: the reason reported
};

// Longest suffixes first where one ends another (".lora_down.weight" before ".weight"-style ambiguity never arises:
// every rule ends in a distinct token).
constexpr SuffixRule kSuffixes[] = {
    {".lora_down.weight", Part::kDown},
    {".lora_up.weight", Part::kUp},
    {".lora_A.weight", Part::kDown},
    {".lora_B.weight", Part::kUp},
    {".lora_A", Part::kDown},  // moss-edit-lora (Moss-Image-Edit): bare PEFT names, alphas in adapter_config
    {".lora_B", Part::kUp},
    {".lora.down.weight", Part::kDown},
    {".lora.up.weight", Part::kUp},
    {".lora_linear_layer.down.weight", Part::kDown},
    {".lora_linear_layer.up.weight", Part::kUp},
    {".alpha", Part::kAlpha},
    {".hada_w1_a", Part::kHadaW1A},
    {".hada_w1_b", Part::kHadaW1B},
    {".hada_w2_a", Part::kHadaW2A},
    {".hada_w2_b", Part::kHadaW2B},
    {".lokr_w1", Part::kLokrW1},
    {".lokr_w1_a", Part::kLokrW1A},
    {".lokr_w1_b", Part::kLokrW1B},
    {".lokr_w2", Part::kLokrW2},
    {".lokr_w2_a", Part::kLokrW2A},
    {".lokr_w2_b", Part::kLokrW2B},
    {".diff", Part::kDiff},
    {".diff_b", Part::kDiffB},
    {".lora_mid.weight", Part::kUnsupported, "Tucker (LoCon conv) factors do not apply to linear layers"},
    {".hada_t1", Part::kUnsupported, "Tucker LoHa factors do not apply to linear layers"},
    {".hada_t2", Part::kUnsupported, "Tucker LoHa factors do not apply to linear layers"},
    {".lokr_t2", Part::kUnsupported, "Tucker LoKr factors do not apply to linear layers"},
    {".dora_scale", Part::kDora, "DoRA (weight-decomposed) adapters are not supported"},
    {".lora_magnitude_vector", Part::kDora, "DoRA (weight-decomposed) adapters are not supported"},
    {".lora_magnitude_vector.weight", Part::kDora, "DoRA (weight-decomposed) adapters are not supported"},
};

constexpr std::string_view kDotPrefixes[] = {"model.diffusion_model.", "diffusion_model.", "base_model.model.",
                                             "transformer."};
constexpr std::string_view kUnderscorePrefixes[] = {"lora_unet_", "lora_transformer_", "lycoris_"};
constexpr std::string_view kTextEncoderPrefixes[] = {"lora_te_", "lora_te1_", "lora_te2_", "text_encoder.",
                                                     "lora_te."};

bool is_float(DType d) { return d == DType::kF32 || d == DType::kF16 || d == DType::kBF16; }

std::string shape_str(const std::vector<std::int64_t>& s) {
  std::string out = "[";
  for (std::size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

float widen(const std::byte* p, DType d) {
  switch (d) {
    case DType::kF32: {
      float v;
      std::memcpy(&v, p, 4);
      return v;
    }
    case DType::kBF16: {
      std::uint16_t v;
      std::memcpy(&v, p, 2);
      return quant::bf16_to_float(v);
    }
    case DType::kF16: {
      std::uint16_t h;
      std::memcpy(&h, p, 2);
      const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
      const float mag = e == 0    ? std::ldexp(static_cast<float>(m), -24)
                        : e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                                  : std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, e - 15);
      return (h & 0x8000) ? -mag : mag;
    }
    default: return std::numeric_limits<float>::quiet_NaN();  // unreachable: dtypes are checked at parse time
  }
}

// A fused alias target: module name in a file -> (target, row offset) parts, all with the same row count.
struct Alias {
  std::vector<std::pair<std::string, std::int64_t>> parts;
  std::int64_t stored_rows = 0;
};

// Raw parts of one module of one file.
struct ModuleParts {
  std::map<Part, const weights::TensorInfo*> parts;
  std::vector<std::string> rejected;  // tensor names, when the module is rejected
  std::string reject_reason;
};

}  // namespace

const char* block_gemm_name(BlockGemm g) {
  switch (g) {
    case BlockGemm::kQ: return "q";
    case BlockGemm::kK: return "k";
    case BlockGemm::kV: return "v";
    case BlockGemm::kO: return "o";
    case BlockGemm::kUpGate: return "ffn_up_gate";
    case BlockGemm::kDown: return "ffn_down";
  }
  return "?";
}

const char* kind_name(AdapterKind k) {
  switch (k) {
    case AdapterKind::kLowRank: return "lora";
    case AdapterKind::kLoHa: return "loha";
    case AdapterKind::kLoKr: return "lokr";
    case AdapterKind::kFull: return "full";
  }
  return "?";
}

const TargetModule* TargetModel::find(std::string_view name) const {
  const auto it = std::lower_bound(modules.begin(), modules.end(), name,
                                   [](const TargetModule& m, std::string_view n) { return m.name < n; });
  return it != modules.end() && it->name == name ? &*it : nullptr;
}

TargetModel moss_dit_targets() {
  TargetModel t;
  t.blocks = kLayers;
  t.dim = kDim;
  // weights::moss_dit_linears: the one table of the DiT's linear shapes (sorted by name, as TargetModel keeps them).
  static_assert(static_cast<int>(BlockGemm::kQ) == 0 && static_cast<int>(BlockGemm::kDown) == kBlockGemms - 1);
  for (const weights::DitLinear& l : weights::moss_dit_linears()) {
    t.modules.push_back({l.name, l.n, l.k, l.has_bias, l.block, static_cast<BlockGemm>(l.gemm)});
  }
  return t;
}

float lora_coef(float strength, double alpha_scale) {
  const double v = static_cast<double>(strength) * alpha_scale;
  return std::fabs(v) <= std::numeric_limits<float>::max() ? static_cast<float>(v) : std::numeric_limits<float>::infinity();
}

struct LoraStack::File {
  std::filesystem::path path;
  weights::SafetensorsFile st;
  weights::FileReader reader;
  explicit File(const std::filesystem::path& p) : path(p), st(p), reader(p) {}
};

namespace {

// The shared alpha rule's metadata part (quant::lora_metadata_alpha), its errors as format errors of this file.
std::optional<double> metadata_alpha(const std::filesystem::path& file, const std::map<std::string, std::string>& meta) {
  try {
    return quant::lora_metadata_alpha(meta);
  } catch (const quant::QuantError& e) {
    throw FormatError(file, "", e.what());
  }
}

// The module a canonical-looking name refers to, after the dotted prefixes; "" when nothing matches.
std::string strip_dot_prefix(std::string_view m) {
  for (std::string_view p : kDotPrefixes) {
    if (m.starts_with(p)) return std::string(m.substr(p.size()));
  }
  return std::string(m);
}

}  // namespace

LoraStack::LoraStack(std::span<const LoraSpec> specs, const TargetModel& model, const LoadOptions& options)
    : model_(model), specs_(specs.begin(), specs.end()) {
  // Fused aliases and kohya's underscore spellings of every addressable module.
  std::map<std::string, Alias, std::less<>> aliases;
  for (int b = 0; b < model_.blocks; ++b) {
    const std::string p = "layers." + std::to_string(b) + ".proj.";
    if (model_.find(p + "q") && model_.find(p + "k") && model_.find(p + "v")) {
      aliases[p + "qkv"] = Alias{{{p + "q", 0}, {p + "k", model_.dim}, {p + "v", 2 * model_.dim}}, 3 * model_.dim};
    }
  }
  std::map<std::string, std::string, std::less<>> underscore;
  const auto add_underscore = [&](const std::string& name) {
    std::string u = name;
    std::replace(u.begin(), u.end(), '.', '_');
    // Two names that differ only in '.' vs '_' would make kohya keys ambiguous.
    if (!underscore.emplace(u, name).second) fail("underscore spelling '" + u + "' is ambiguous in the target model");
  };
  for (const TargetModule& m : model_.modules) add_underscore(m.name);
  for (const auto& [name, alias] : aliases) add_underscore(name);

  for (std::size_t fi = 0; fi < specs_.size(); ++fi) {
    const LoraSpec& spec = specs_[fi];
    if (!std::isfinite(spec.strength)) fail("strength of " + util::path_utf8(spec.path) + " is not finite");
    files_.push_back(std::make_unique<File>(spec.path));
    const File& f = *files_.back();
    FileReport rep;
    rep.file = util::path_utf8(spec.path.filename());
    const std::optional<double> meta_alpha = metadata_alpha(spec.path, f.st.metadata());
    int meta_alpha_used = 0;
    std::map<std::string, double> config_alphas;  // moss-edit-lora: per module, by name as written
    try {
      config_alphas = quant::lora_adapter_config_alphas(f.st.metadata());
      rep.format = quant::lora_adapter_config_format(f.st.metadata());
    } catch (const quant::QuantError& e) {
      throw FormatError(spec.path, "", e.what());
    }
    int config_alpha_used = 0;

    // Group tensors by (resolved) module name.
    std::map<std::string, ModuleParts> mods;   // canonical module or alias name -> parts
    std::map<std::string, std::string> source_of;  // canonical -> name as written
    int consumed = 0;
    for (const weights::TensorInfo& t : f.st.tensors()) {
      if (std::find(spec.consumed.begin(), spec.consumed.end(), t.name) != spec.consumed.end()) {
        ++consumed;
        continue;
      }
      const SuffixRule* rule = nullptr;
      for (const SuffixRule& r : kSuffixes) {
        if (t.name.size() > r.suffix.size() && std::string_view(t.name).ends_with(r.suffix) &&
            (rule == nullptr || r.suffix.size() > rule->suffix.size())) {
          rule = &r;
        }
      }
      if (rule == nullptr) {
        rep.unknown.push_back(t.name + ": not an adapter tensor (no known LoRA / LoHa / LoKr / diff suffix)");
        continue;
      }
      const std::string_view raw = std::string_view(t.name).substr(0, t.name.size() - rule->suffix.size());
      bool text_encoder = false;
      for (std::string_view p : kTextEncoderPrefixes) text_encoder |= raw.starts_with(p);
      if (text_encoder) {
        rep.unknown.push_back(t.name + ": text-encoder adapter (only the DiT is patched)");
        continue;
      }
      std::string module;
      for (std::string_view p : kUnderscorePrefixes) {
        if (raw.starts_with(p)) {
          const auto it = underscore.find(raw.substr(p.size()));
          module = it == underscore.end() ? std::string() : it->second;
          if (module.empty()) module = "\x01";  // an underscore name that maps to nothing
          break;
        }
      }
      if (module.empty()) module = strip_dot_prefix(raw);
      if (module == "\x01" || (model_.find(module) == nullptr && aliases.find(module) == aliases.end())) {
        rep.unknown.push_back(t.name + ": no module '" + std::string(raw) + "' in the model");
        continue;
      }
      ModuleParts& mp = mods[module];
      source_of.emplace(module, std::string(raw));
      if (rule->part == Part::kUnsupported || rule->part == Part::kDora) {
        mp.reject_reason = rule->why;
        mp.rejected.push_back(t.name);
        continue;
      }
      if (!mp.parts.emplace(rule->part, &t).second) {
        throw FormatError(spec.path, t.name, "duplicate adapter part for module '" + module + "' (two spellings of one key?)");
      }
    }

    std::vector<Adapter> file_adapters;
    for (auto& [module, mp] : mods) {
      const auto has = [&](Part p) { return mp.parts.count(p) != 0; };
      const auto all_names = [&] {
        std::vector<std::string> n = mp.rejected;
        for (const auto& [p, t] : mp.parts) n.push_back(t->name);
        return n;
      };
      if (!mp.reject_reason.empty()) {
        for (const std::string& n : all_names()) rep.unknown.push_back(n + ": " + mp.reject_reason + " (module not applied)");
        continue;
      }
      const bool low = has(Part::kDown) || has(Part::kUp);
      const bool loha = has(Part::kHadaW1A) || has(Part::kHadaW1B) || has(Part::kHadaW2A) || has(Part::kHadaW2B);
      const bool lokr = has(Part::kLokrW1) || has(Part::kLokrW1A) || has(Part::kLokrW1B) || has(Part::kLokrW2) ||
                        has(Part::kLokrW2A) || has(Part::kLokrW2B);
      const bool full = has(Part::kDiff) || has(Part::kDiffB);
      const int kinds = int(low) + int(loha) + int(lokr) + int(has(Part::kDiff));
      if (kinds > 1) throw FormatError(spec.path, source_of[module], "module mixes adapter types (lora / loha / lokr / diff)");
      if (kinds == 0 && !full) {
        for (const std::string& n : all_names()) rep.unknown.push_back(n + ": alpha without adapter weights");
        continue;
      }

      // Target rows: the module itself, or the parts of a fused alias.
      const auto alias = aliases.find(module);
      const TargetModule* direct = model_.find(module);
      const std::int64_t stored_rows = alias != aliases.end() ? alias->second.stored_rows : direct->n;
      const std::int64_t k = alias != aliases.end() ? model_.find(alias->second.parts[0].first)->k : direct->k;
      const std::string& src = source_of[module];

      const auto ref = [&](Part p) {
        const weights::TensorInfo* t = mp.parts.at(p);
        if (!is_float(t->dtype)) throw FormatError(spec.path, t->name, std::string("adapter tensor must be F32, F16 or BF16, got ") + weights::dtype_name(t->dtype));
        return TensorRef{static_cast<int>(fi), t->name, t->dtype, t->shape, t->offset, t->nbytes};
      };
      const auto matrix = [&](const TensorRef& r, std::int64_t rows, std::int64_t cols) {
        if (r.shape.size() != 2 || r.shape[0] != rows || r.shape[1] != cols) {
          throw FormatError(spec.path, r.name, "shape " + shape_str(r.shape) + " does not fit module '" + module + "' (expected [" +
                                                   std::to_string(rows) + ", " + std::to_string(cols) + "])");
        }
      };
      const auto rank_of = [&](const TensorRef& r, int dim) {
        if (r.shape.size() != 2 || r.shape[static_cast<std::size_t>(dim)] <= 0) {
          throw FormatError(spec.path, r.name, "expected a 2-D factor with a positive rank, got " + shape_str(r.shape));
        }
        return r.shape[static_cast<std::size_t>(dim)];
      };

      Adapter a;
      a.lora = static_cast<int>(fi);
      a.source = src;
      a.k = k;
      a.stored_rows = stored_rows;
      std::optional<double> alpha;
      if (has(Part::kAlpha)) {
        const TensorRef r = ref(Part::kAlpha);
        if (util::checked_numel(r.shape).value_or(0) != 1) throw FormatError(spec.path, r.name, "alpha must be a scalar");
        std::vector<std::byte> raw(static_cast<std::size_t>(r.nbytes));
        f.reader.read_at(r.offset, raw.data(), raw.size());
        alpha = widen(raw.data(), r.dtype);
        if (!std::isfinite(*alpha)) throw FormatError(spec.path, r.name, "non-finite alpha");
      }
      std::int64_t divisor = 0;
      if (low) {
        if (!has(Part::kDown) || !has(Part::kUp)) throw FormatError(spec.path, src, "LoRA module needs both its down (A) and up (B) weights");
        a.kind = AdapterKind::kLowRank;
        TensorRef down = ref(Part::kDown), up = ref(Part::kUp);
        const std::int64_t r = rank_of(down, 0);
        matrix(down, r, k);
        matrix(up, stored_rows, r);
        if (r > std::min(stored_rows, k)) throw FormatError(spec.path, down.name, "rank " + std::to_string(r) + " exceeds the module's dimensions");
        a.rank = divisor = r;
        a.parts = {down, up};
        // Without an alpha tensor, a moss-edit-lora adapter_config's module alpha, else diffusers' metadata alpha
        // (PEFT: scaling = lora_alpha / r).
        if (const auto it = config_alphas.find(src); !alpha && it != config_alphas.end()) {
          alpha = it->second;
          ++config_alpha_used;
        }
        if (!alpha && meta_alpha) {
          alpha = *meta_alpha;
          ++meta_alpha_used;
        }
      } else if (loha) {
        for (Part p : {Part::kHadaW1A, Part::kHadaW1B, Part::kHadaW2A, Part::kHadaW2B}) {
          if (!has(p)) throw FormatError(spec.path, src, "LoHa module needs hada_w1_a, hada_w1_b, hada_w2_a and hada_w2_b");
        }
        a.kind = AdapterKind::kLoHa;
        TensorRef w1a = ref(Part::kHadaW1A), w1b = ref(Part::kHadaW1B), w2a = ref(Part::kHadaW2A), w2b = ref(Part::kHadaW2B);
        const std::int64_t r1 = rank_of(w1b, 0), r2 = rank_of(w2b, 0);
        matrix(w1a, stored_rows, r1);
        matrix(w1b, r1, k);
        matrix(w2a, stored_rows, r2);
        matrix(w2b, r2, k);
        if (r1 > std::min(stored_rows, k) || r2 > std::min(stored_rows, k)) throw FormatError(spec.path, src, "LoHa rank exceeds the module's dimensions");
        a.rank = divisor = r1;
        a.parts = {w1a, w1b, w2a, w2b};
      } else if (lokr) {
        a.kind = AdapterKind::kLoKr;
        std::int64_t dims[2][2] = {};  // [w][rows, cols]
        const Part whole[2] = {Part::kLokrW1, Part::kLokrW2}, fa[2] = {Part::kLokrW1A, Part::kLokrW2A},
                   fb[2] = {Part::kLokrW1B, Part::kLokrW2B};
        for (int w = 0; w < 2; ++w) {
          if (has(whole[w]) && (has(fa[w]) || has(fb[w]))) throw FormatError(spec.path, src, "LoKr module has both a whole and a factored w" + std::to_string(w + 1));
          if (has(whole[w])) {
            TensorRef t = ref(whole[w]);
            dims[w][0] = rank_of(t, 0);
            dims[w][1] = rank_of(t, 1);
            a.parts.push_back(t);
          } else {
            if (!has(fa[w]) || !has(fb[w])) throw FormatError(spec.path, src, "LoKr module needs lokr_w" + std::to_string(w + 1) + " or both its _a and _b factors");
            TensorRef ta = ref(fa[w]), tb = ref(fb[w]);
            const std::int64_t r = rank_of(tb, 0);
            dims[w][0] = rank_of(ta, 0);
            dims[w][1] = rank_of(tb, 1);
            matrix(ta, dims[w][0], r);
            if (r > std::min(dims[w][0], dims[w][1])) throw FormatError(spec.path, tb.name, "LoKr factor rank exceeds its matrix");
            a.lokr_split[w] = true;
            divisor = r;  // ComfyUI: the last factored w's rank
            a.parts.push_back(ta);
            a.parts.push_back(tb);
          }
        }
        const auto prod = [](std::int64_t x, std::int64_t y) { return util::checked_mul(static_cast<std::uint64_t>(x), static_cast<std::uint64_t>(y)); };
        if (prod(dims[0][0], dims[1][0]) != static_cast<std::uint64_t>(stored_rows) || prod(dims[0][1], dims[1][1]) != static_cast<std::uint64_t>(k)) {
          throw FormatError(spec.path, src, "LoKr kron(w1, w2) shape does not match module '" + module + "'");
        }
        a.rank = divisor;
      } else {
        a.kind = AdapterKind::kFull;
        if (has(Part::kDiff)) {
          TensorRef d = ref(Part::kDiff);
          matrix(d, stored_rows, k);
          a.parts = {d};
        }
      }
      if (has(Part::kDiffB)) {
        TensorRef d = ref(Part::kDiffB);
        const bool bias_ok = alias == aliases.end() && direct->has_bias;
        if (!bias_ok) throw FormatError(spec.path, d.name, "module '" + module + "' has no bias");
        if (d.shape.size() != 1 || d.shape[0] != stored_rows) throw FormatError(spec.path, d.name, "bias delta shape " + shape_str(d.shape) + " does not match");
        a.bias_diff = d;
      }
      if (alpha && divisor > 0 && a.kind != AdapterKind::kFull) {
        const double s = quant::lora_alpha_scale(*alpha, divisor);
        if (!(std::fabs(s) <= std::numeric_limits<float>::max())) throw FormatError(spec.path, src, "alpha / rank is outside the fp32 range");
        a.alpha_scale = s;
      }

      // One adapter per target.
      std::vector<std::pair<std::string, std::int64_t>> targets;
      if (alias != aliases.end()) targets = alias->second.parts;
      else targets.push_back({module, 0});
      for (const auto& [name, row0] : targets) {
        const TargetModule* tm = model_.find(name);
        Adapter t = a;
        t.module = name;
        t.n = tm->n;
        t.row0 = row0;
        t.side_branch = t.kind == AdapterKind::kLowRank && tm->block >= 0 && !t.bias_diff;
        if (!t.side_branch && tm->block >= 0 && options.int8_block_gemms) {
          rep.warnings.push_back(name + ": " + kind_name(t.kind) + " adapter merged into INT8: adds requantization error");
        }
        file_adapters.push_back(std::move(t));
      }
    }
    if (file_adapters.empty()) {
      fail(util::path_utf8(spec.path) + ": no tensor applies to this model (" + std::to_string(rep.unknown.size()) +
           " keys reported" + (rep.unknown.empty() ? std::string() : ", first: " + rep.unknown.front()) + ")");
    }
    if (options.strict && !rep.unknown.empty()) {
      fail(util::path_utf8(spec.path) + ": " + std::to_string(rep.unknown.size()) + " keys not applied, first: " + rep.unknown.front());
    }
    if (options.strict && !rep.warnings.empty()) {
      fail(util::path_utf8(spec.path) + ": " + std::to_string(rep.warnings.size()) + " warnings, first: " + rep.warnings.front());
    }
    if (consumed > 0) rep.notes.push_back(std::to_string(consumed) + " tensors read outside the LoRA stack (LoraSpec::consumed)");
    if (meta_alpha_used > 0) {
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.9g", *meta_alpha);
      rep.notes.push_back(std::string("lora_alpha ") + buf + " from lora_adapter_metadata applied to " +
                          std::to_string(meta_alpha_used) + " modules without an alpha tensor");
    }
    if (config_alpha_used > 0) {
      rep.notes.push_back("alphas from adapter_config (moss-edit-lora) applied to " + std::to_string(config_alpha_used) +
                          " modules");
    }
    rep.adapters = static_cast<int>(file_adapters.size());
    for (const Adapter& a : file_adapters) {
      (a.side_branch ? rep.side_branch : rep.merged) += 1;
    }
    for (Adapter& a : file_adapters) adapters_.push_back(std::move(a));
    reports_.push_back(std::move(rep));
  }
}

LoraStack::~LoraStack() = default;
LoraStack::LoraStack(LoraStack&&) noexcept = default;
LoraStack& LoraStack::operator=(LoraStack&&) noexcept = default;

void LoraStack::set_strength(int lora, float strength) {
  if (!std::isfinite(strength)) fail("strength is not finite");
  specs_.at(static_cast<std::size_t>(lora)).strength = strength;
}

float LoraStack::coef(const Adapter& a) const { return lora_coef(strength(a.lora), a.alpha_scale); }

std::vector<std::string> LoraStack::merge_modules() const {
  std::set<std::string> s;
  for (const Adapter& a : adapters_) {
    if (!a.side_branch) s.insert(a.module);
  }
  return {s.begin(), s.end()};
}

bool LoraStack::has_merge(std::string_view module) const {
  return std::any_of(adapters_.begin(), adapters_.end(), [&](const Adapter& a) { return !a.side_branch && a.module == module; });
}

std::vector<const Adapter*> LoraStack::adapters_of(std::string_view module) const {
  std::vector<const Adapter*> out;
  for (const Adapter& a : adapters_) {
    if (a.module == module) out.push_back(&a);
  }
  return out;
}

std::vector<float> LoraStack::read(const TensorRef& t, std::int64_t row0, std::int64_t rows) const {
  const File& f = *files_.at(static_cast<std::size_t>(t.file));
  const std::int64_t lead = t.shape.empty() ? 1 : t.shape[0];
  if (rows < 0) rows = lead - row0;
  if (row0 < 0 || rows < 0 || row0 + rows > lead) fail("row range out of bounds for " + t.name);
  const std::int64_t numel = util::checked_numel(t.shape).value_or(-1);
  const std::int64_t row_elems = lead > 0 ? numel / lead : 0;
  const std::size_t es = weights::dtype_size(t.dtype);
  const std::size_t count = static_cast<std::size_t>(rows * row_elems);
  std::vector<std::byte> raw(count * es);
  f.reader.read_at(t.offset + static_cast<std::uint64_t>(row0 * row_elems) * es, raw.data(), raw.size());
  std::vector<float> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = widen(raw.data() + i * es, t.dtype);
    if (!std::isfinite(out[i])) throw FormatError(f.path, t.name, "non-finite value");
  }
  return out;
}

}  // namespace mi::lora
