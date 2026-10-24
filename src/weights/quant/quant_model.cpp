#include "weights/quant/quant_model.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "util/checked_math.h"
#include "util/file_time.h"
#include "util/json.h"
#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/quant/nvfp4_codec.h"
#include "weights/quant/quant_codec.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"
#include "weights/test_seams.h"

namespace mi::quant {

namespace fs = std::filesystem;
using weights::DType;
using weights::IndexedTensor;

namespace {

// A comfy_quant JSON is ~70 bytes; anything this large is not a config.
constexpr std::uint64_t kMaxConfigBytes = 4096;
constexpr std::size_t kMaxConfigValues = 256;
constexpr std::size_t kMaxQuantMetadataValues = 1'000'000;
constexpr std::int64_t kMaxGroupSize = 4096;
// The comfy_quant formats the reader reads; Builder::quantized has one branch per entry.
constexpr std::string_view kFormats[] = {"int8_tensorwise", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8", "nvfp4"};
// The only convrot_w4a4 quant_group_size comfy-kitchen @dba9fac accepts (_INT4_GROUP_SIZE).
constexpr int kW4A4QuantGroup = 64;

[[noreturn]] void fail(const std::string& msg) { throw QuantError("mi::quant: " + msg); }

// "<digits>." at the front of `rest`; advances past it.
bool take_index(std::string_view& rest) {
  int n = 0;
  return weights::take_layer_index(rest, n);
}

std::string shape_str(const std::vector<std::int64_t>& s) {
  std::string out = "[";
  for (std::size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

// ---- Names -------------------------------------------------------------------------------------------------

struct Canon {
  std::string name;     // canonical name (empty when excluded)
  std::string exclude;  // exclusion prefix
};

// `raw` is a tensor name, or a module name (a config key) when `module`. Text-encoder exclusions are
// weights::classify_tensor's (a module is probed as its weight); a name it cannot route throws IndexError.
Canon canonical(ModelKind kind, std::string_view raw, bool module) {
  Canon c;
  std::string_view n = raw;
  if (kind == ModelKind::kDiT) {
    for (std::string_view p : {"model.diffusion_model.", "diffusion_model."}) {
      if (n.starts_with(p)) {
        n.remove_prefix(p.size());
        break;
      }
    }
    std::string_view rest = n;
    if (rest.starts_with("layers.")) {
      rest.remove_prefix(7);
      // ComfyUI's per-block attention backend choice ({"attention": "comfy_kitchen_int8"}): not a weight.
      if (take_index(rest) && rest.starts_with("comfy_attention.")) {
        c.exclude = "comfy_attention.";
        return c;
      }
    }
  } else {
    if (n.starts_with("model.")) n.remove_prefix(6);
    c.exclude = weights::classify_tensor(weights::Component::kTextEncoder, std::string(n) + (module ? ".weight" : "")).exclude;
    if (!c.exclude.empty()) return c;
  }
  c.name = std::string(n);
  return c;
}

constexpr std::string_view kDitGemms[] = {"q", "k", "v", "o", "ffn_up_gate", "ffn_down"};
constexpr std::string_view kTeGemms[] = {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
                                         "linear_attn.in_proj_b",   "linear_attn.out_proj",  "self_attn.q_proj",
                                         "self_attn.k_proj",        "self_attn.v_proj",      "self_attn.o_proj",
                                         "mlp.gate_proj",           "mlp.up_proj",           "mlp.down_proj"};

bool is_fused_qkv(ModelKind kind, std::string_view module) {
  if (kind != ModelKind::kDiT || !module.starts_with("layers.")) return false;
  module.remove_prefix(7);
  return take_index(module) && module == "proj.qkv";
}

// ---- Configs -----------------------------------------------------------------------------------------------

struct Config {
  std::string format;
  bool convrot = false;
  std::int64_t convrot_groupsize = 256;
  std::int64_t group_size = 16;
  std::int64_t quant_group_size = 0;
  std::string linear_dtype;
  bool full_precision_mm = false;
  std::set<std::string, std::less<>> present;  // the fields above that the config sets
  std::string text;  // canonical JSON text, for conflict checks
};

// Field of the config or, as ComfyUI also accepts, of its "params" object.
const json::Value* field(const json::Value& conf, std::string_view key) {
  if (const json::Value* v = conf.find(key)) return v;
  const json::Value* p = conf.find("params");
  return p != nullptr && p->is_object() ? p->find(key) : nullptr;
}

Config parse_config(const json::Value& conf, const std::string& module) {
  if (!conf.is_object()) fail("quantization config of '" + module + "' is not a JSON object");
  Config c;
  const json::Value* f = conf.find("format");
  if (f == nullptr || !f->is_string()) fail("quantization config of '" + module + "' has no format");
  c.format = f->as_string();
  auto get_bool = [&](std::string_view key, bool& out) {
    if (const json::Value* v = field(conf, key)) {
      c.present.emplace(key);
      if (!v->is_bool()) fail("'" + module + "': config field " + std::string(key) + " is not a boolean");
      out = v->as_bool();
    }
  };
  auto get_int = [&](std::string_view key, std::int64_t& out) {
    if (const json::Value* v = field(conf, key)) {
      c.present.emplace(key);
      if (!v->is_int() || v->as_int() < 1 || v->as_int() > kMaxGroupSize) {
        fail("'" + module + "': config field " + std::string(key) + " must be an integer in [1, 4096]");
      }
      out = v->as_int();
    }
  };
  get_bool("convrot", c.convrot);
  get_bool("full_precision_matrix_mult", c.full_precision_mm);
  get_int("convrot_groupsize", c.convrot_groupsize);
  get_int("group_size", c.group_size);
  get_int("quant_group_size", c.quant_group_size);
  if (const json::Value* v = field(conf, "linear_dtype")) {
    if (!v->is_string()) fail("'" + module + "': config field linear_dtype is not a string");
    c.present.emplace("linear_dtype");
    c.linear_dtype = v->as_string();
  }
  c.text = c.format + "|" + std::to_string(c.convrot) + "|" + std::to_string(c.convrot_groupsize) + "|" + std::to_string(c.group_size) +
           "|" + std::to_string(c.quant_group_size) + "|" + c.linear_dtype + "|" + std::to_string(c.full_precision_mm);
  return c;
}

// ---- Tensors -----------------------------------------------------------------------------------------------

struct Raw {
  IndexedTensor t;  // t.name = name in the file
  bool used = false;
};

bool is_float(DType d) { return d == DType::kBF16 || d == DType::kF16 || d == DType::kF32; }

// Rows [r0, r0 + rows) of a tensor whose dim 0 is the row dimension.
IndexedTensor slice_rows(const IndexedTensor& t, std::int64_t r0, std::int64_t rows) {
  if (t.shape.empty() || t.shape[0] <= 0 || r0 < 0 || rows < 0 || r0 + rows > t.shape[0]) {
    fail("row slice [" + std::to_string(r0) + ", " + std::to_string(r0 + rows) + ") outside tensor '" + t.name + "' " + shape_str(t.shape));
  }
  const std::uint64_t row_bytes = t.nbytes / static_cast<std::uint64_t>(t.shape[0]);
  IndexedTensor s = t;
  s.shape[0] = rows;
  s.offset = t.offset + static_cast<std::uint64_t>(r0) * row_bytes;
  s.nbytes = static_cast<std::uint64_t>(rows) * row_bytes;
  return s;
}

std::int64_t numel(const IndexedTensor& t) { return t.numel(); }

// F32 holding one value per row: [n] or [n, 1].
bool per_row(const IndexedTensor& t, std::int64_t n) {
  return t.dtype == DType::kF32 && numel(t) == n && (t.shape == std::vector<std::int64_t>{n} || t.shape == std::vector<std::int64_t>{n, 1});
}

class Builder {
 public:
  Builder(std::vector<Raw>& raws, std::unordered_map<std::string, std::size_t>& by_name) : raws_(raws), by_name_(by_name) {}

  Raw* find(const std::string& name) {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &raws_[it->second];
  }
  IndexedTensor take(const std::string& name) {
    Raw* r = find(name);
    if (r == nullptr) fail("tensor '" + name + "' missing");
    r->used = true;
    return r->t;
  }
  std::optional<IndexedTensor> take_opt(const std::string& name) {
    Raw* r = find(name);
    if (r == nullptr) return std::nullopt;
    r->used = true;
    return r->t;
  }

  // A module quantized per `c`; `m` is its canonical name, the file's tensors are looked up by canonical name too.
  QuantLinear quantized(const std::string& m, const Config& c) {
    QuantLinear l;
    l.name = l.source_module = m;
    l.full_precision_mm = c.full_precision_mm;
    l.weight = take(m + ".weight");
    // Before any layout check, so a format the engine does not read is named as such (not as a wrong dtype).
    if (std::find(std::begin(kFormats), std::end(kFormats), c.format) == std::end(kFormats)) {
      fail("'" + m + "': quantization format '" + c.format + "' is not supported (see docs/integration.md#supported-models)");
    }
    if (c.format == "nvfp4") {
      nvfp4(l, c);
      attach_bias(l);
      return l;
    }
    const IndexedTensor& w = l.weight;
    if (w.dtype != DType::kI8 || w.shape.size() != 2 || w.shape[0] <= 0 || w.shape[1] <= 0) {
      fail("'" + m + "': " + c.format + " weight must be I8 [N, C], got " + weights::dtype_name(w.dtype) + " " + shape_str(w.shape));
    }
    l.n = w.shape[0];
    const std::int64_t cols = w.shape[1];
    auto convrot_spec = [&](std::int64_t k) {
      if (!valid_convrot_group(c.convrot_groupsize) || k % c.convrot_groupsize != 0) {
        fail("'" + m + "': convrot_groupsize " + std::to_string(c.convrot_groupsize) + " is not a power of 4 in [4, 1024] dividing K=" + std::to_string(k));
      }
      return QuantSpec::int8_convrot(static_cast<int>(c.convrot_groupsize));
    };
    if (c.format == "int8_tensorwise") {
      l.k = cols;
      l.spec = c.convrot ? convrot_spec(l.k) : QuantSpec::int8_plain();
      l.scale = take(m + ".weight_scale");
      if (per_row(*l.scale, l.n)) l.encoding = Encoding::kRaw;
      else if (l.scale->dtype == DType::kF32 && numel(*l.scale) == 1) l.encoding = Encoding::kScalarScale;
      else fail("'" + m + "': weight_scale must be F32 [N], [N, 1] or a scalar, got " + weights::dtype_name(l.scale->dtype) + " " + shape_str(l.scale->shape));
    } else if (c.format == "convrot_w4a4") {
      if (!util::checked_mul(static_cast<std::uint64_t>(cols), 2)) fail("'" + m + "': K overflows");
      l.k = cols * 2;
      l.spec = convrot_spec(l.k);
      l.encoding = Encoding::kW4A4;
      if (c.quant_group_size != 0 && l.k % c.quant_group_size != 0) fail("'" + m + "': quant_group_size does not divide K");
      if (!c.linear_dtype.empty() && c.linear_dtype != "int4" && c.linear_dtype != "int8") fail("'" + m + "': unknown linear_dtype '" + c.linear_dtype + "'");
      l.int8_activations = c.linear_dtype == "int8";
      l.quant_group = static_cast<int>(c.quant_group_size);
      l.scale = take(m + ".weight_scale");
      if (!per_row(*l.scale, l.n)) fail("'" + m + "': convrot_w4a4 weight_scale must be F32 [N]");
    } else {  // asym_w4a8_int8 or w6a8_int8, the rest of kFormats
      const int bits = c.format == "w6a8_int8" ? 6 : 4;
      if ((cols * 8) % bits != 0) fail("'" + m + "': packed width " + std::to_string(cols) + " is not a whole number of " + std::to_string(bits) + "-bit codes");
      l.k = cols * 8 / bits;
      l.encoding = bits == 6 ? Encoding::kW6A8 : Encoding::kW4A8;
      const std::int64_t gs = c.group_size;
      // comfy-kitchen's layout contract (validate_w4a8_operands / _check_six_bit_layout).
      if (gs < 4 || (16 % gs != 0 && gs % 16 != 0) || l.k % 16 != 0 || l.k % gs != 0 || (bits == 6 && (l.k % 32 != 0 || gs % 16 != 0))) {
        fail("'" + m + "': group_size " + std::to_string(gs) + " / K=" + std::to_string(l.k) + " violate the " + c.format + " layout");
      }
      l.quant_group = static_cast<int>(gs);
      l.spec = convrot_spec(l.k);
      l.s_rel = take(m + ".weight_s_rel");
      if ((l.s_rel->dtype != DType::kF8E4M3 && l.s_rel->dtype != DType::kU8) || l.s_rel->shape != std::vector<std::int64_t>{l.n, l.k / gs}) {
        fail("'" + m + "': weight_s_rel must be F8_E4M3 " + shape_str({l.n, l.k / gs}) + ", got " + weights::dtype_name(l.s_rel->dtype) + " " + shape_str(l.s_rel->shape));
      }
      l.s_channel = take(m + ".weight_s_channel");
      if (!per_row(*l.s_channel, l.n)) fail("'" + m + "': weight_s_channel must be F32 [N]");
      l.codebook = take_opt(m + ".weight_codebook");
      if (l.codebook && (bits == 6 || l.codebook->dtype != DType::kF32 || l.codebook->shape != std::vector<std::int64_t>{16})) {
        fail("'" + m + "': weight_codebook must be F32 [16] and only with 4-bit codes");
      }
    }
    attach_bias(l);
    return l;
  }

  // comfy_quant {"format": "nvfp4"} (ComfyUI QUANT_ALGOS["nvfp4"]): weight U8 [N, K/2], weight_scale F8_E4M3 (or U8)
  // swizzled [round_up(N,128), round_up(K/16,4)], weight_scale_2 F32 scalar, optional input_scale.
  void nvfp4(QuantLinear& l, const Config& c) {
    const std::string& m = l.name;
    const IndexedTensor& w = l.weight;
    if (w.dtype != DType::kU8 || w.shape.size() != 2 || w.shape[0] <= 0 || w.shape[1] <= 0) {
      fail("'" + m + "': nvfp4 weight must be U8 [N, K/2], got " + weights::dtype_name(w.dtype) + " " + shape_str(w.shape));
    }
    if (c.convrot) fail("'" + m + "': nvfp4 has no rotation; a convrot flag is not part of the format");
    for (const char* key : {"convrot_groupsize", "quant_group_size", "linear_dtype"}) {
      if (c.present.count(key) != 0) fail("'" + m + "': config field " + key + " does not belong to the nvfp4 format");
    }
    // ComfyUI would dequantize such a layer and run it in bf16; the engine has no such path for NVFP4.
    if (c.full_precision_mm) fail("'" + m + "': nvfp4 with full_precision_matrix_mult is not supported");
    if (c.present.count("group_size") != 0 && c.group_size != kNvfp4Block) {
      fail("'" + m + "': nvfp4 group_size must be 16, got " + std::to_string(c.group_size));
    }
    if (!util::checked_mul(static_cast<std::uint64_t>(w.shape[1]), 2)) fail("'" + m + "': K overflows");
    l.n = w.shape[0];
    l.k = w.shape[1] * 2;
    if (l.k % kNvfp4Block != 0) fail("'" + m + "': nvfp4 K=" + std::to_string(l.k) + " is not a multiple of 16");
    l.spec = QuantSpec::nvfp4();
    l.encoding = Encoding::kNvfp4Raw;
    l.quant_group = kNvfp4Block;
    l.scale = take(m + ".weight_scale");
    const std::vector<std::int64_t> sf_shape{nvfp4_scale_rows(l.n), nvfp4_scale_cols(l.k / kNvfp4Block)};
    if ((l.scale->dtype != DType::kF8E4M3 && l.scale->dtype != DType::kU8) || l.scale->shape != sf_shape) {
      fail("'" + m + "': nvfp4 weight_scale must be F8_E4M3 " + shape_str(sf_shape) + " (the swizzled block scales), got " +
           weights::dtype_name(l.scale->dtype) + " " + shape_str(l.scale->shape));
    }
    l.scale_2 = take(m + ".weight_scale_2");
    if (l.scale_2->dtype != DType::kF32 || numel(*l.scale_2) != 1 || l.scale_2->shape.size() > 1) {
      fail("'" + m + "': nvfp4 weight_scale_2 must be an F32 scalar, got " + weights::dtype_name(l.scale_2->dtype) + " " + shape_str(l.scale_2->shape));
    }
    l.input_scale = take_opt(m + ".input_scale");
    if (l.input_scale && (l.input_scale->dtype != DType::kF32 || numel(*l.input_scale) != 1)) fail("'" + m + "': input_scale must be an F32 scalar");
    if (find(m + ".pre_quant_scale") != nullptr) {
      fail("'" + m + "': nvfp4 with pre_quant_scale (AWQ-style smoothing) is not supported; use a file without it");
    }
  }

  QuantLinear unquantized(const std::string& m) {
    QuantLinear l;
    l.name = l.source_module = m;
    // NVFP4 tensors without a comfy_quant config: NVIDIA ModelOpt's export convention. Its defaults (low nibble
    // first, linear [N, K/16] scales) differ from ComfyUI's and cannot be told apart by shape, so it is refused.
    if (find(m + ".weight_scale_2") != nullptr) {
      fail("'" + m + "' holds NVFP4 tensors (weight_scale, weight_scale_2) but no comfy_quant config: an NVIDIA ModelOpt-"
           "convention file, whose default nibble order and scale layout differ from ComfyUI's and cannot be told apart by "
           "shape; only ComfyUI comfy_quant nvfp4 files are supported");
    }
    l.weight = take(m + ".weight");
    if (l.weight.dtype != DType::kBF16 || l.weight.shape.size() != 2 || l.weight.shape[0] <= 0 || l.weight.shape[1] <= 0) {
      fail("'" + m + "': unquantized GEMM weight must be BF16 [N, K], got " + weights::dtype_name(l.weight.dtype) + " " + shape_str(l.weight.shape));
    }
    l.n = l.weight.shape[0];
    l.k = l.weight.shape[1];
    attach_bias(l);
    return l;
  }

  void attach_bias(QuantLinear& l) {
    l.bias = take_opt(l.name + ".bias");
    if (l.bias && (!is_float(l.bias->dtype) || l.bias->shape != std::vector<std::int64_t>{l.n})) {
      fail("'" + l.name + "': bias must be a float [N] tensor, got " + shape_str(l.bias->shape));
    }
  }

  // proj.qkv -> proj.q / proj.k / proj.v by row thirds.
  std::vector<QuantLinear> unfuse(const QuantLinear& f) {
    if (f.n % 3 != 0) fail("'" + f.name + "': fused qkv has " + std::to_string(f.n) + " rows, not a multiple of 3");
    const std::int64_t rows = f.n / 3;
    const std::string base = f.name.substr(0, f.name.size() - 3);
    std::vector<QuantLinear> out;
    const char* parts[] = {"q", "k", "v"};
    for (int i = 0; i < 3; ++i) {
      QuantLinear l = f;
      l.name = base + parts[i];
      l.n = rows;
      l.fused_slice = true;
      const std::int64_t r0 = i * rows;
      l.weight = slice_rows(f.weight, r0, rows);
      if (f.encoding == Encoding::kNvfp4Raw) {
        // Whole 128-row tiles of the swizzled scales are contiguous [128 * cols] byte runs: rows slice as rows.
        if (rows % 128 != 0) fail("'" + f.name + "': nvfp4 fused qkv thirds of " + std::to_string(rows) + " rows are not whole 128-row scale tiles");
        l.scale = slice_rows(*f.scale, r0, rows);
      } else if (f.scale && f.encoding != Encoding::kScalarScale) {
        l.scale = slice_rows(*f.scale, r0, rows);
      }
      if (f.s_rel) l.s_rel = slice_rows(*f.s_rel, r0, rows);
      if (f.s_channel) l.s_channel = slice_rows(*f.s_channel, r0, rows);
      if (f.bias) l.bias = slice_rows(*f.bias, r0, rows);
      out.push_back(std::move(l));
    }
    return out;
  }

 private:
  std::vector<Raw>& raws_;
  std::unordered_map<std::string, std::size_t>& by_name_;
};

// A linear must be one of the model's with its shape (DiT: weights::moss_dit_linears, a fused proj.qkv being proj.q / k
// / v by rows; text encoder: the linears of weights::qwen35_text_tensors), checked before any of its bytes are read:
// to_model_index and the text encoder read every scale whole, so a hostile I8 [N, 1] weight with an F32 [N] scale
// would make them read and copy 4N bytes. Builder checked the scale tensors against [n, k] per layout, so with the
// canonical [n, k] they are bounded too.
void check_linear_shape(ModelKind kind, const QuantLinear& l) {
  if (weights::testing::any_model_shapes()) return;
  bool known = false;
  std::int64_t n = 0, k = 0;
  if (kind == ModelKind::kDiT) {
    const bool fused = is_fused_qkv(kind, l.name);
    const std::string module = fused ? l.name.substr(0, l.name.size() - 3) + "q" : l.name;  // a fused part's rows
    if (const weights::DitLinear* t = weights::find_moss_dit_linear(module)) {
      known = true;
      n = fused ? 3 * t->n : t->n;
      k = t->k;
    }
  } else if (const weights::TeTensor* t = weights::find_qwen35_text_tensor(l.name + ".weight");
             t != nullptr && t->linear) {
    known = true;
    n = t->shape[0];
    k = t->shape[1];
  }
  if (known && l.n == n && l.k == k) return;
  const std::string found =
      "tensor '" + l.weight.name + "' " + weights::dtype_name(l.weight.dtype) + " " + shape_str(l.weight.shape);
  const char* model = kind == ModelKind::kDiT ? "DiT" : "text encoder";
  if (!known) fail("'" + l.name + "' is not a linear of the MOSS-Image " + model + " (" + found + ")");
  // [n, k] as the layout stores it: packed codes are narrower.
  std::int64_t cols = k;
  if (l.encoding == Encoding::kW4A4 || l.encoding == Encoding::kW4A8 || l.encoding == Encoding::kNvfp4Raw) {
    cols = k / 2;
  } else if (l.encoding == Encoding::kW6A8) {
    cols = k * 3 / 4;
  }
  const std::string layout = !l.spec.quantized()            ? std::string("bf16")
                             : l.encoding == Encoding::kRaw ? describe(l.spec)
                                                            : std::string(encoding_name(l.encoding));
  fail(found + " (MOSS-Image: " + shape_str({n, cols}) + " as " + layout + ")");
}

// Every other kept tensor of a text encoder (norms, conv1d, A_log, dt_bias, a bf16 embedding table) must be one of
// weights::qwen35_text_tensors with its shape, before any is read (the text encoder reads the F32 ones whole).
void check_text_tensor(const IndexedTensor& t) {
  if (weights::testing::any_model_shapes()) return;
  const weights::TeTensor* e = weights::find_qwen35_text_tensor(t.name);
  if (e != nullptr && e->shape == t.shape) return;
  const std::string found = "tensor '" + t.name + "' " + weights::dtype_name(t.dtype) + " " + shape_str(t.shape);
  if (e == nullptr) fail(found + " is not a tensor of the MOSS-Image text encoder");
  fail(found + " (MOSS-Image: " + shape_str(e->shape) + ")");
}

std::vector<fs::path> component_files(const fs::path& dir) {
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (e.is_regular_file() && util::path_utf8(e.path().filename()).ends_with(".safetensors")) files.push_back(e.path());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) fail(util::path_utf8(dir) + ": no .safetensors files");
  return files;
}

std::string read_config_text(const fs::path& path, const weights::TensorInfo& t) {
  if (t.dtype != DType::kU8 || t.shape.size() != 1 || t.nbytes == 0 || t.nbytes > kMaxConfigBytes) {
    throw weights::FormatError(path, t.name, "comfy_quant must be a U8 vector of 1.." + std::to_string(kMaxConfigBytes) + " bytes");
  }
  std::string s(static_cast<std::size_t>(t.nbytes), '\0');
  weights::FileReader(path).read_at(t.offset, s.data(), s.size());
  return s;
}

}  // namespace

const char* format_name(QuantFormat f) {
  switch (f) {
    case QuantFormat::kBF16: return "bf16";
    case QuantFormat::kInt8ConvRot: return "int8_convrot";
    case QuantFormat::kInt8Plain: return "int8_plain";
    case QuantFormat::kInt4ConvRot: return "int4_convrot";
    case QuantFormat::kNvfp4: return "nvfp4";
  }
  return "?";
}

std::string describe(const QuantSpec& s) {
  if (s.format == QuantFormat::kInt8ConvRot || s.format == QuantFormat::kInt4ConvRot) return format_name(s.format) + std::to_string(s.group);
  return format_name(s.format);
}

const char* encoding_name(Encoding e) {
  switch (e) {
    case Encoding::kRaw: return "raw";
    case Encoding::kScalarScale: return "int8_scalar_scale";
    case Encoding::kW4A8: return "w4a8";
    case Encoding::kW6A8: return "w6a8";
    case Encoding::kW4A4: return "w4a4";
    case Encoding::kNvfp4Raw: return "nvfp4";
  }
  return "?";
}

const char* kind_name(ModelKind k) { return k == ModelKind::kDiT ? "dit" : "text_encoder"; }

bool is_gemm_module(ModelKind kind, std::string_view m) {
  if (kind == ModelKind::kDiT) {
    if (!m.starts_with("layers.")) return false;
    m.remove_prefix(7);
    if (!take_index(m) || !m.starts_with("proj.")) return false;
    m.remove_prefix(5);
    return std::find(std::begin(kDitGemms), std::end(kDitGemms), m) != std::end(kDitGemms);
  }
  if (!m.starts_with("language_model.layers.")) return false;
  m.remove_prefix(22);
  if (!take_index(m)) return false;
  return std::find(std::begin(kTeGemms), std::end(kTeGemms), m) != std::end(kTeGemms);
}

const QuantLinear* QuantModel::find_linear(std::string_view name) const {
  auto it = std::lower_bound(linears.begin(), linears.end(), name, [](const QuantLinear& l, std::string_view n) { return l.name < n; });
  return it != linears.end() && it->name == name ? &*it : nullptr;
}

const IndexedTensor* QuantModel::find_tensor(std::string_view name) const {
  auto it = std::lower_bound(tensors.begin(), tensors.end(), name, [](const IndexedTensor& t, std::string_view n) { return t.name < n; });
  return it != tensors.end() && it->name == name ? &*it : nullptr;
}

namespace {

// `parsed`: the header of the single file `path`, already read (open_quant_model(const SafetensorsFile&)).
QuantModel open_quant_model_impl(const fs::path& path, std::optional<ModelKind> kind,
                                 const weights::SafetensorsFile* parsed = nullptr) {
  QuantModel model;
  std::vector<fs::path> paths;
  if (parsed != nullptr) {
    paths.push_back(path);
    model.base_dir = path.parent_path();
  } else if (fs::is_regular_file(path)) {
    paths.push_back(path);
    model.base_dir = path.parent_path();
  } else if (fs::is_directory(path)) {
    fs::path dir = path;
    const bool bundle = fs::is_directory(path / "transformer") || fs::is_directory(path / "text_encoder");
    if (bundle) {
      if (!kind) kind = ModelKind::kDiT;
      dir = path / (*kind == ModelKind::kDiT ? "transformer" : "text_encoder");
      if (!fs::is_directory(dir)) fail(util::path_utf8(path) + ": bundle has no " + util::path_utf8(dir.filename()) + "/ directory");
    }
    if (fs::is_regular_file(dir / "hf_quant_config.json")) {
      fail(util::path_utf8(dir) + ": hf_quant_config.json marks an NVIDIA ModelOpt export; only ComfyUI comfy_quant files are "
           "supported");
    }
    paths = component_files(dir);
    model.base_dir = dir;
  } else {
    fail(util::path_utf8(path) + ": no such file or directory");
  }

  // Headers and raw tensors. Configs are only collected here: whether a module is excluded (and its config ignored)
  // depends on the model kind, known once every name has been seen.
  struct PendingConfig {
    std::string module, where;
    std::optional<json::Value> value;  // legacy header entry
    fs::path file;                     // comfy_quant tensor: read when needed
    weights::TensorInfo tensor;
  };
  std::vector<Raw> raws;
  std::vector<PendingConfig> pending;
  for (const fs::path& p : paths) {
    std::optional<weights::SafetensorsFile> owned;
    const weights::SafetensorsFile& file = parsed != nullptr ? *parsed : owned.emplace(p);
    const auto id = static_cast<std::uint32_t>(model.files.size());
    model.files.push_back({util::path_utf8(p.lexically_relative(model.base_dir)), file.file_size(), util::file_mtime_ns(p)});
    if (id == 0) model.metadata = file.metadata();
    auto md = file.metadata().find("_quantization_metadata");
    if (md != file.metadata().end()) {
      json::Value doc;
      try {
        doc = json::parse(md->second, {.max_values = kMaxQuantMetadataValues});
      } catch (const json::ParseError& e) {
        throw weights::FormatError(p, "", std::string("_quantization_metadata is not valid JSON (") + e.what() + ")");
      }
      const json::Value* layers = doc.find("layers");
      if (layers == nullptr || !layers->is_object()) throw weights::FormatError(p, "", "_quantization_metadata has no \"layers\" object");
      for (const auto& [module, conf] : layers->as_object()) pending.push_back({module, "_quantization_metadata", conf, {}, {}});
    }
    for (const weights::TensorInfo& t : file.tensors()) {
      if (t.name.ends_with(".comfy_quant")) {
        pending.push_back({t.name.substr(0, t.name.size() - 12), t.name, std::nullopt, p, t});
        continue;
      }
      raws.push_back({IndexedTensor{t.name, t.dtype, t.shape, id, t.offset, t.nbytes}, false});
    }
  }
  // Qwen3_5ForCausalLM checkpoints name the decoder `model.layers.*`; the engine's encoder reads the Qwen3_5Model
  // layout that diffusers and Comfy-Org ship.
  for (const Raw& r : raws) {
    if (r.t.name.starts_with("model.layers.") || r.t.name.starts_with("model.embed_tokens.")) {
      fail(util::path_utf8(path) + ": a text-only Qwen3.5 checkpoint ('" + r.t.name + "'); the text encoder must use the Qwen3_5Model "
           "layout (model.language_model.*, as in the diffusers bundle and Comfy-Org's qwen3.5 file)");
    }
  }
  if (!kind) {
    kind = std::any_of(raws.begin(), raws.end(), [](const Raw& r) { return r.t.name.find("language_model.") != std::string::npos; })
               ? ModelKind::kTextEncoder : ModelKind::kDiT;
  }
  model.kind = *kind;

  // Canonical names; exclusions.
  std::map<std::string, weights::ExcludedSummary> excluded;
  auto exclude = [&](const std::string& prefix, std::uint64_t bytes) {
    auto& e = excluded[prefix];
    e.component = model.kind == ModelKind::kDiT ? weights::Component::kTransformer : weights::Component::kTextEncoder;
    e.prefix = prefix;
    ++e.tensors;
    e.bytes += bytes;
  };
  std::vector<Raw> kept;
  std::unordered_map<std::string, std::size_t> by_name;
  for (Raw& r : raws) {
    Canon c = canonical(model.kind, r.t.name, false);
    if (!c.exclude.empty()) {
      exclude(c.exclude, r.t.nbytes);
      continue;
    }
    if (!by_name.emplace(c.name, kept.size()).second) fail("two tensors map to the canonical name '" + c.name + "'");
    r.t.name = c.name;
    kept.push_back(std::move(r));
  }

  // Configs of kept modules, keyed by canonical name.
  std::map<std::string, std::pair<Config, std::string>> configs;  // module -> (config, where it came from)
  for (const PendingConfig& pc : pending) {
    const Canon c = canonical(model.kind, pc.module, true);
    if (!c.exclude.empty()) {
      if (pc.value == std::nullopt) exclude(c.exclude, 0);  // a comfy_quant tensor of an excluded module
      continue;
    }
    json::Value v;
    if (pc.value) {
      v = *pc.value;
    } else {
      const std::string text = read_config_text(pc.file, pc.tensor);
      try {
        v = json::parse(text, {.max_values = kMaxConfigValues});
      } catch (const json::ParseError& e) {
        throw weights::FormatError(pc.file, pc.tensor.name, std::string("not valid JSON (") + e.what() + ")");
      }
    }
    Config cfg = parse_config(v, pc.module);
    auto [it, fresh] = configs.emplace(c.name, std::make_pair(cfg, pc.where));
    if (!fresh && it->second.first.text != cfg.text) {
      fail("module '" + c.name + "' has conflicting quantization configs (" + it->second.second + " vs " + pc.where + ")");
    }
  }

  Builder b(kept, by_name);
  std::vector<QuantLinear> linears;
  std::set<std::string> modules_done;
  auto add = [&](QuantLinear l) {
    check_linear_shape(model.kind, l);
    if (is_fused_qkv(model.kind, l.name)) {
      for (auto& part : b.unfuse(l)) linears.push_back(std::move(part));
    } else {
      linears.push_back(std::move(l));
    }
  };
  for (const auto& [module, cw] : configs) {
    if (b.find(module + ".weight") == nullptr) fail("quantization config for '" + module + "' but no weight tensor");
    add(b.quantized(module, cw.first));
    modules_done.insert(module);
  }
  // Unquantized GEMMs (kept bf16, or a plain bf16 checkpoint).
  for (std::size_t i = 0; i < kept.size(); ++i) {
    const std::string& n = kept[i].t.name;
    if (kept[i].used || !n.ends_with(".weight")) continue;
    const std::string m = n.substr(0, n.size() - 7);
    if (modules_done.count(m) || !(is_gemm_module(model.kind, m) || is_fused_qkv(model.kind, m))) continue;
    add(b.unquantized(m));
    modules_done.insert(m);
  }
  // A quantized module's tensors are all consumed; leftovers (a correction term, an input scale) would be dropped.
  for (const std::string& m : modules_done) {
    const std::string prefix = m + ".";
    for (const Raw& r : kept) {
      if (!r.used && r.t.name.starts_with(prefix)) fail("unexpected tensor '" + r.t.name + "' in quantized module '" + m + "'");
    }
  }
  for (Raw& r : kept) {
    if (r.used) continue;
    if (model.kind == ModelKind::kTextEncoder) check_text_tensor(r.t);
    model.tensors.push_back(std::move(r.t));
  }
  std::sort(linears.begin(), linears.end(), [](const QuantLinear& a, const QuantLinear& c) { return a.name < c.name; });
  for (std::size_t i = 1; i < linears.size(); ++i) {
    if (linears[i].name == linears[i - 1].name) fail("linear '" + linears[i].name + "' is defined twice (fused and separate?)");
  }
  model.linears = std::move(linears);
  std::sort(model.tensors.begin(), model.tensors.end(), [](const IndexedTensor& a, const IndexedTensor& c) { return a.name < c.name; });
  for (const auto& t : model.tensors) {
    if (model.find_linear(t.name.substr(0, t.name.rfind('.'))) != nullptr) fail("tensor '" + t.name + "' collides with a linear");
  }
  for (auto& [p, e] : excluded) model.excluded.push_back(e);
  return model;
}

}  // namespace

QuantModel open_quant_model(const weights::SafetensorsFile& file, std::optional<ModelKind> kind) {
  try {
    return open_quant_model_impl(file.path(), kind, &file);
  } catch (const weights::IndexError& e) {
    fail(util::path_utf8(file.path()) + ": " + e.what());
  } catch (const fs::filesystem_error& e) {
    fail(util::path_utf8(file.path()) + ": cannot read (" + e.what() + ")");
  }
}

QuantModel open_quant_model(const fs::path& path, std::optional<ModelKind> kind) {
  try {
    return open_quant_model_impl(path, kind);
  } catch (const weights::IndexError& e) {
    fail(util::path_utf8(path) + ": " + e.what());
  } catch (const fs::filesystem_error& e) {
    fail(util::path_utf8(path) + ": cannot read (" + e.what() + ")");
  }
}

// ---- Engine view -------------------------------------------------------------------------------------------

const QuantSpec* QuantIndex::find_spec(std::string_view module) const {
  auto it = specs.find(module);
  return it == specs.end() ? nullptr : &it->second;
}

QuantWeights view(const QuantIndex& index, const weights::TensorMap& m, std::string_view module, const QuantSpec& expected) {
  const std::string mod(module);
  const QuantSpec* spec = index.find_spec(module);
  if (spec == nullptr) fail("'" + mod + "' is not a linear of the model");
  if (*spec != expected) fail("'" + mod + "' is stored as " + describe(*spec) + ", but the caller expects " + describe(expected));
  auto w = m.find(mod + ".weight");
  if (w == m.end()) fail("weight '" + mod + ".weight' missing");
  const Tensor& wt = w->second;
  const bool fp4 = spec->format == QuantFormat::kNvfp4;
  const mi::DType want = fp4 ? mi::DType::kU8 : spec->quantized() ? mi::DType::kI8 : mi::DType::kBF16;
  if (wt.dtype != want || wt.rank != 2 || !wt.is_contiguous()) fail("weight '" + mod + "' must be contiguous " + dtype_name(want) + " [N, K], got " + wt.describe());
  QuantWeights v;
  v.spec = *spec;
  v.n = wt.dim(0);
  v.k = spec->format == QuantFormat::kInt4ConvRot || fp4 ? 2 * wt.dim(1) : wt.dim(1);  // packed: two values per byte
  v.weight = wt.data;
  v.device = wt.device;
  if (reinterpret_cast<std::uintptr_t>(v.weight) % mi::dtype_size(wt.dtype) != 0) fail("weight '" + mod + "' is not aligned in memory");
  if ((spec->format == QuantFormat::kInt8ConvRot || spec->format == QuantFormat::kInt4ConvRot) &&
      (!valid_convrot_group(spec->group) || v.k % spec->group != 0)) {
    fail("'" + mod + "': bad ConvRot group");
  }
  if (fp4) {
    auto s = m.find(mod + ".weight_scale");
    if (s == m.end()) fail("block scales '" + mod + ".weight_scale' missing");
    const Tensor& st = s->second;
    const std::int64_t rows = nvfp4_scale_rows(v.n), cols = nvfp4_scale_cols(v.k / kNvfp4Block);
    if (v.k % kNvfp4Block != 0 || st.dtype != mi::DType::kU8 || st.rank != 2 || st.dim(0) != rows || st.dim(1) != cols || !st.is_contiguous()) {
      fail("block scales '" + mod + "' must be contiguous U8 [" + std::to_string(rows) + ", " + std::to_string(cols) + "], got " + st.describe());
    }
    if (st.device != v.device) fail("block scales '" + mod + "' are on another device than the weight");
    auto ts = index.tensor_scales.find(mod);
    if (ts == index.tensor_scales.end()) fail("'" + mod + "': nvfp4 tensor scale missing from the index");
    v.block_scale = st.ptr<const std::uint8_t>();
    v.tensor_scale = ts->second;
    // The sm_120a GEMMs read both operands with 128-bit TMA accesses (kernels/nvfp4/nvfp4.h).
    if (v.k % 32 != 0) fail("'" + mod + "': nvfp4 K=" + std::to_string(v.k) + " is not a multiple of 32, which the NVFP4 GEMMs need");
    if (reinterpret_cast<std::uintptr_t>(v.weight) % 16 != 0 || reinterpret_cast<std::uintptr_t>(v.block_scale) % 16 != 0) {
      fail("'" + mod + "': nvfp4 weight and block scales must be 16-byte aligned in memory");
    }
  } else if (spec->quantized()) {
    auto s = m.find(mod + ".weight_scale");
    if (s == m.end()) fail("scale '" + mod + ".weight_scale' missing");
    const Tensor& st = s->second;
    if (st.dtype != mi::DType::kF32 || st.numel() != v.n || !st.is_contiguous()) fail("scale '" + mod + "' must be contiguous F32 [N], got " + st.describe());
    if (st.device != v.device) fail("scale '" + mod + "' is on another device than its weight");
    v.scale = st.ptr<const float>();
    if (reinterpret_cast<std::uintptr_t>(v.scale) % alignof(float) != 0) fail("scale '" + mod + "' is not 4-byte aligned in memory");
  }
  auto b = m.find(mod + ".bias");
  if (b != m.end()) {
    const Tensor& bt = b->second;
    if (bt.dtype != mi::DType::kBF16 && bt.dtype != mi::DType::kF16 && bt.dtype != mi::DType::kF32) fail("bias '" + mod + "' must be bf16, f16 or f32, got " + bt.describe());
    if (bt.rank != 1 || bt.dim(0) != v.n || !bt.is_contiguous()) fail("bias '" + mod + "' must be contiguous [N], got " + bt.describe());
    if (bt.device != v.device) fail("bias '" + mod + "' is on another device than its weight");
    v.bias = bt.data;
    if (reinterpret_cast<std::uintptr_t>(v.bias) % mi::dtype_size(bt.dtype) != 0) fail("bias '" + mod + "' is not aligned in memory");
    v.bias_dtype = bt.dtype;
  }
  return v;
}

QuantIndex to_model_index(const QuantModel& model, const IndexOptions& options) {
  QuantIndex qi;
  weights::ModelIndex& index = qi.index;
  index.bundle_dir = model.base_dir;
  index.files = model.files;
  index.excluded = model.excluded;
  const bool dit = model.kind == ModelKind::kDiT;
  const weights::Component comp = dit ? weights::Component::kTransformer : weights::Component::kTextEncoder;

  std::map<std::string, weights::WeightGroup> groups;
  auto put = [&](IndexedTensor t) {
    weights::TensorRoute r;
    try {
      r = weights::classify_tensor(comp, t.name);
    } catch (const weights::IndexError& e) {
      fail(e.what());
    }
    if (!r.exclude.empty()) fail("internal: kept tensor '" + t.name + "' routes to exclusion " + r.exclude);
    weights::WeightGroup& g = groups[r.group];
    if (!r.layer_kind.empty()) {
      if (!g.layer_type.empty() && g.layer_type != r.layer_kind) fail(r.group + " mixes linear_attn and self_attn tensors");
      g.layer_type = r.layer_kind;
    }
    g.tensors.push_back(std::move(t));
  };
  for (const QuantLinear& l : model.linears) {
    // Native INT4: the packed nibbles and per-row scales as stored (the same bytes decode_rows reads).
    // comfy-kitchen (every backend) runs convrot_w4a4 only with quant_group_size 64 (its default when absent), a
    // layout requirement on K that does not enter the per-row math; other values keep the decode path.
    const bool int4 = options.native_int4 && l.encoding == Encoding::kW4A4 && !l.int8_activations &&
                      (l.quant_group == 0 || l.quant_group == kW4A4QuantGroup) && l.k % kW4A4QuantGroup == 0;
    if (l.needs_decode() && !int4) fail("linear '" + l.name + "' is stored as " + encoding_name(l.encoding) + " and needs decoding; convert the file with `moss-infer quantize` first");
    const QuantSpec spec = int4 ? QuantSpec::int4_convrot(l.spec.group) : l.spec;
    qi.specs.emplace(l.name, spec);
    IndexedTensor w = l.weight;
    w.name = l.name + ".weight";
    put(std::move(w));
    if (spec.format == QuantFormat::kNvfp4) {
      const std::int64_t sfk = l.k / kNvfp4Block;
      const std::vector<std::byte> sf = read_tensor(model, *l.scale);
      if (const char* why = nvfp4_scales_problem(reinterpret_cast<const std::uint8_t*>(sf.data()), l.n, sfk)) {
        fail("'" + l.name + "': nvfp4 weight_scale holds " + why);
      }
      const std::vector<std::byte> s2 = read_tensor(model, *l.scale_2);
      float v;
      std::memcpy(&v, s2.data(), sizeof v);
      if (!valid_scales(&v, 1)) fail("'" + l.name + "': nvfp4 weight_scale_2 is not finite and > 0");
      qi.tensor_scales.emplace(l.name, v);
      IndexedTensor s = *l.scale;
      s.name = l.name + ".weight_scale";
      s.dtype = DType::kU8;  // UE4M3 bytes: the GEMMs read them as raw scale factors
      put(std::move(s));
    } else if (spec.quantized()) {
      IndexedTensor s = *l.scale;
      const std::vector<std::byte> bytes = read_tensor(model, s);
      std::vector<float> v(static_cast<std::size_t>(l.n));
      std::memcpy(v.data(), bytes.data(), v.size() * sizeof(float));
      if (!valid_scales(v.data(), l.n)) fail("'" + l.name + "': weight_scale holds a value that is not finite and > 0");
      s.name = l.name + ".weight_scale";
      s.shape = {l.n};
      put(std::move(s));
    }
    if (l.bias) {
      IndexedTensor b = *l.bias;
      b.name = l.name + ".bias";
      put(std::move(b));
    }
  }
  for (const IndexedTensor& t : model.tensors) put(t);
  // Group order as weights::build_index declares them.
  std::vector<std::string> order = dit ? std::vector<std::string>{"globals", "text_embedder"} : std::vector<std::string>{"embed_tokens"};
  std::vector<std::pair<int, std::string>> numbered;
  for (const auto& [name, g] : groups) {
    if (std::find(order.begin(), order.end(), name) != order.end() || name == "final_norm") continue;
    // "block.N" / "layers.N" as classify_tensor builds them from a parsed int.
    int n = 0;
    std::from_chars(name.data() + name.find('.') + 1, name.data() + name.size(), n);
    numbered.emplace_back(n, name);
  }
  std::sort(numbered.begin(), numbered.end());
  for (auto& [n, name] : numbered) order.push_back(name);
  if (!dit) order.push_back("final_norm");
  for (const std::string& name : order) {
    auto it = groups.find(name);
    if (it == groups.end()) continue;
    weights::WeightGroup g = std::move(it->second);
    g.component = comp;
    g.name = name;
    std::sort(g.tensors.begin(), g.tensors.end(), [](const IndexedTensor& a, const IndexedTensor& c) {
      return a.file != c.file ? a.file < c.file : a.offset < c.offset;
    });
    for (const auto& t : g.tensors) {
      g.total_bytes += t.nbytes;
      g.total_params += t.numel();
    }
    index.groups.push_back(std::move(g));
  }
  return qi;
}

// ---- CPU decode --------------------------------------------------------------------------------------------

std::vector<std::byte> read_tensor(const QuantModel& model, const IndexedTensor& t) {
  std::vector<std::byte> out(static_cast<std::size_t>(t.nbytes));
  weights::FileReader(model.file_path(t.file)).read_at(t.offset, out.data(), out.size());
  return out;
}

namespace {

// Bytes of rows [r0, r0 + rows) of a row-major tensor.
std::vector<std::byte> read_rows(const QuantModel& model, const IndexedTensor& t, std::int64_t r0, std::int64_t rows) {
  return read_tensor(model, slice_rows(t, r0, rows));
}

}  // namespace

void decode_rows(const QuantModel& model, const QuantLinear& l, std::int64_t row0, std::int64_t rows, std::int8_t* q,
                 float* scale) {
  if (!l.spec.quantized()) fail("linear '" + l.name + "' is not quantized");
  if (l.encoding == Encoding::kNvfp4Raw) fail("linear '" + l.name + "' is nvfp4, which has no INT8 grid (dequantize it instead)");
  if (row0 < 0 || rows < 0 || row0 + rows > l.n) fail("rows out of range for '" + l.name + "'");
  if (rows == 0) return;
  const auto n_elems = static_cast<std::size_t>(rows * l.k);
  const std::vector<std::byte> w = read_rows(model, l.weight, row0, rows);
  const auto* wp = reinterpret_cast<const std::uint8_t*>(w.data());
  switch (l.encoding) {
    case Encoding::kRaw:
    case Encoding::kScalarScale:
      std::copy(wp, wp + n_elems, reinterpret_cast<std::uint8_t*>(q));
      break;
    case Encoding::kW4A4:
      decode_int4_rows(wp, rows, l.k, q);
      break;
    case Encoding::kNvfp4Raw:
      break;  // refused above
    case Encoding::kW4A8:
    case Encoding::kW6A8: {
      const std::vector<std::byte> sr = read_rows(model, *l.s_rel, row0, rows);
      std::vector<float> cb;
      if (l.codebook) {
        const std::vector<std::byte> raw = read_tensor(model, *l.codebook);
        cb.resize(16);
        std::memcpy(cb.data(), raw.data(), 16 * sizeof(float));
        for (float v : cb) {
          if (!std::isfinite(v)) fail("'" + l.name + "': non-finite codebook entry");
        }
      }
      if (!decode_grouped_rows(wp, reinterpret_cast<const std::uint8_t*>(sr.data()), l.codebook ? cb.data() : nullptr, rows, l.k,
                               l.encoding == Encoding::kW6A8 ? 6 : 4, l.quant_group, q)) {
        fail("'" + l.name + "': NaN or negative group scale");
      }
      break;
    }
  }
  if (l.encoding == Encoding::kScalarScale) {
    const std::vector<std::byte> s = read_tensor(model, *l.scale);
    float v;
    std::memcpy(&v, s.data(), sizeof v);
    std::fill(scale, scale + rows, v);
  } else {
    const IndexedTensor& st = (l.encoding == Encoding::kW4A8 || l.encoding == Encoding::kW6A8) ? *l.s_channel : *l.scale;
    const std::vector<std::byte> s = read_rows(model, st, row0, rows);
    std::memcpy(scale, s.data(), static_cast<std::size_t>(rows) * sizeof(float));
  }
  if (!valid_scales(scale, rows)) fail("'" + l.name + "': a stored per-row scale is not finite and > 0");
}

void decode_linear(const QuantModel& model, const QuantLinear& l, std::int8_t* q, float* scale, int threads) {
  if (!l.spec.quantized()) fail("linear '" + l.name + "' is not quantized");
  // Fixed chunks of rows keep each read a few MB and the work balanced across threads.
  const std::int64_t chunk = std::max<std::int64_t>(1, (std::int64_t{4} << 20) / std::max<std::int64_t>(1, l.k));
  const std::int64_t chunks = (l.n + chunk - 1) / chunk;
  parallel_for(chunks, threads, [&](std::int64_t b, std::int64_t e) {
    for (std::int64_t c = b; c < e; ++c) {
      const std::int64_t r0 = c * chunk, rows = std::min(chunk, l.n - r0);
      decode_rows(model, l, r0, rows, q + r0 * l.k, scale + r0);
    }
  });
}

void dequantize_linear_rows(const QuantModel& model, const QuantLinear& l, std::int64_t row0, std::int64_t rows, float* out) {
  if (!l.spec.quantized()) {
    if (row0 < 0 || rows < 0 || row0 + rows > l.n) fail("rows out of range for '" + l.name + "'");
    const std::vector<std::byte> w = read_rows(model, l.weight, row0, rows);
    for (std::size_t i = 0; i < static_cast<std::size_t>(rows * l.k); ++i) {
      std::uint16_t v;
      std::memcpy(&v, w.data() + 2 * i, 2);
      out[i] = bf16_to_float(v);
    }
    return;
  }
  if (l.encoding == Encoding::kNvfp4Raw) {
    dequantize_nvfp4_linear_rows(model, l, row0, rows, out, nullptr);
    return;
  }
  std::vector<std::int8_t> q(static_cast<std::size_t>(rows * l.k));
  std::vector<float> s(static_cast<std::size_t>(rows));
  decode_rows(model, l, row0, rows, q.data(), s.data());
  dequantize_rows(q.data(), s.data(), rows, l.k, l.spec.format == QuantFormat::kInt8ConvRot ? l.spec.group : 0, out);
}

void dequantize_nvfp4_linear_rows(const QuantModel& model, const QuantLinear& l, std::int64_t row0, std::int64_t rows,
                                  float* out, std::uint16_t* out_bf16) {
  if (l.encoding != Encoding::kNvfp4Raw) fail("linear '" + l.name + "' is not nvfp4");
  if (row0 < 0 || rows < 0 || row0 + rows > l.n) fail("rows out of range for '" + l.name + "'");
  if (rows == 0) return;
  const std::int64_t sfk = l.k / kNvfp4Block;
  const std::vector<std::byte> w = read_rows(model, l.weight, row0, rows);
  // The 128-row scale tiles covering the rows (each tile row is one contiguous run of the swizzled tensor).
  const std::int64_t t0 = row0 / 128, t1 = (row0 + rows + 127) / 128;
  const std::vector<std::byte> sf = read_rows(model, *l.scale, t0 * 128, (t1 - t0) * 128);
  const std::vector<std::byte> s2b = read_tensor(model, *l.scale_2);
  float s2;
  std::memcpy(&s2, s2b.data(), sizeof s2);
  const auto* sfp = reinterpret_cast<const std::uint8_t*>(sf.data());
  if (!valid_scales(&s2, 1)) fail("'" + l.name + "': nvfp4 weight_scale_2 is not finite and > 0");
  if (const char* why = nvfp4_scales_problem(sfp, std::min<std::int64_t>((t1 - t0) * 128, l.n - t0 * 128), sfk)) {
    fail("'" + l.name + "': nvfp4 weight_scale holds " + why);
  }
  const auto* wp = reinterpret_cast<const std::uint8_t*>(w.data());
  if (out != nullptr) dequantize_nvfp4_rows(wp, sfp, row0 - t0 * 128, rows, l.k, s2, out);
  if (out_bf16 != nullptr) dequantize_nvfp4_rows_bf16(wp, sfp, row0 - t0 * 128, rows, l.k, s2, out_bf16);
}

void gather_embedding_rows_bf16(const QuantModel& model, const QuantLinear& table, std::span<const std::int64_t> ids,
                                std::uint16_t* out) {
  const auto k = static_cast<std::size_t>(table.k);
  std::vector<std::int8_t> q(k);
  std::vector<float> row(k);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::int64_t id = ids[i];
    if (id < 0 || id >= table.n) fail("'" + table.name + "': token id " + std::to_string(id) + " outside [0, " + std::to_string(table.n) + ")");
    std::uint16_t* dst = out + i * k;
    if (!table.spec.quantized()) {
      const std::vector<std::byte> w = read_rows(model, table.weight, id, 1);
      std::memcpy(dst, w.data(), k * sizeof(std::uint16_t));
      continue;
    }
    float s = 0;
    decode_rows(model, table, id, 1, q.data(), &s);
    dequantize_rows(q.data(), &s, 1, table.k, table.spec.format == QuantFormat::kInt8ConvRot ? table.spec.group : 0, row.data());
    for (std::size_t c = 0; c < k; ++c) dst[c] = float_to_bf16(row[c]);
  }
}

std::vector<std::uint16_t> read_tensor_bf16(const QuantModel& model, const IndexedTensor& t) {
  if (t.dtype != DType::kBF16 && t.dtype != DType::kF32) fail("'" + t.name + "': expected BF16 or F32, got " + weights::dtype_name(t.dtype));
  const std::vector<std::byte> raw = read_tensor(model, t);
  const std::size_t es = weights::dtype_size(t.dtype);
  std::vector<std::uint16_t> out(raw.size() / es);
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (t.dtype == DType::kBF16) {
      std::memcpy(&out[i], raw.data() + 2 * i, 2);
    } else {
      float v;
      std::memcpy(&v, raw.data() + 4 * i, 4);
      out[i] = float_to_bf16(v);
    }
    if (!std::isfinite(bf16_to_float(out[i]))) fail("'" + t.name + "' holds a value that is or rounds to a NaN or infinity in bf16");
  }
  return out;
}

}  // namespace mi::quant
