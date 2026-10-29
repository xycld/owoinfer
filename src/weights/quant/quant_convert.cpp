#include "weights/quant/quant_convert.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "util/checked_math.h"
#include "util/json.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/quant/nvfp4_codec.h"
#include "weights/quant/quant_codec.h"
#include "weights/safetensors.h"

namespace mi::quant {

namespace fs = std::filesystem;
using weights::DType;
using weights::IndexedTensor;

namespace {

[[noreturn]] void fail(const std::string& msg) { throw QuantError("mi::quant: " + msg); }

// The output while it is written: a new temporary file next to it (util::OutputFile::Create::kTemporary), removed
// unless commit() published it.
class PartialFile {
 public:
  explicit PartialFile(const fs::path& target) : file_(target, util::OutputFile::Create::kTemporary) {
    if (!file_.is_open()) fail("cannot create a temporary file next to " + util::path_utf8(target));
  }
  PartialFile(const PartialFile&) = delete;
  PartialFile& operator=(const PartialFile&) = delete;
  ~PartialFile() {
    if (!committed_) file_.discard();
  }
  util::OutputFile& file() { return file_; }
  const fs::path& path() const { return file_.path(); }
  void commit(const fs::path& final_path) {
    if (!file_.publish(final_path)) fail("cannot rename " + util::path_utf8(path()) + " to " + util::path_utf8(final_path));
    committed_ = true;
  }

 private:
  util::OutputFile file_;
  bool committed_ = false;
};

// Rows of a merged weight rounded to bf16 (nearest even); false if a value is, or rounds to, a NaN or infinity.
bool to_bf16_finite(const float* src, std::size_t n, std::uint16_t* dst) {
  bool ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    dst[i] = float_to_bf16(src[i]);
    ok = ok && std::isfinite(bf16_to_float(dst[i]));
  }
  return ok;
}

std::string hex64(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

std::string quote(std::string_view s) {
  std::string out = "\"";
  for (char ch : s) {
    const auto c = static_cast<unsigned char>(ch);
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      out += b;
    } else {
      out += ch;
    }
  }
  return out + "\"";
}

// Order of the safetensors library's Dtype enum; files list tensors by this rank descending, then by name.
int dtype_rank(DType d) {
  switch (d) {
    case DType::kBool: return 0;
    case DType::kU8: return 4;
    case DType::kI8: return 5;
    case DType::kF8E5M2: return 6;
    case DType::kF8E4M3: return 7;
    case DType::kF8E8M0: return 8;
    case DType::kI16: return 9;
    case DType::kU16: return 10;
    case DType::kF16: return 11;
    case DType::kBF16: return 12;
    case DType::kI32: return 13;
    case DType::kU32: return 14;
    case DType::kF32: return 15;
    case DType::kF64: return 17;
    case DType::kI64: return 18;
    case DType::kU64: return 19;
  }
  return 0;
}

float to_float(const std::byte* p, DType d) {
  switch (d) {
    case DType::kF32: {
      float v;
      std::memcpy(&v, p, 4);
      return v;
    }
    case DType::kBF16: {
      std::uint16_t v;
      std::memcpy(&v, p, 2);
      return bf16_to_float(v);
    }
    case DType::kF16: {
      std::uint16_t h;
      std::memcpy(&h, p, 2);
      const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
      float mag = e == 0 ? std::ldexp(static_cast<float>(m), -24)
                  : e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                            : std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, e - 15);
      return (h & 0x8000) ? -mag : mag;
    }
    default: fail(std::string("unsupported float dtype ") + weights::dtype_name(d));
  }
}

std::vector<float> to_floats(const std::vector<std::byte>& raw, DType d) {
  const std::size_t es = weights::dtype_size(d);
  std::vector<float> out(raw.size() / es);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = to_float(raw.data() + i * es, d);
  return out;
}

// ---- LoRA --------------------------------------------------------------------------------------------------

struct LoraFactor {
  std::vector<float> down;  // [rank, k]
  std::vector<float> up;    // [n, rank]
  std::int64_t rank = 0, n = 0, k = 0;
  float coef = 0;
};

using LoraMap = std::map<std::string, std::vector<LoraFactor>>;

struct LoraKey {
  std::string module, part;  // part: down | up | alpha
};

// kohya `diffusion_model.<m>.lora_{down,up}.weight` + `<m>.alpha` (the distill repos' style) and PEFT
// `<prefix><m>.lora_{A,B}.weight`.
std::optional<LoraKey> parse_lora_key(std::string_view name) {
  for (std::string_view p : {"model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer."}) {
    if (name.starts_with(p)) {
      name.remove_prefix(p.size());
      break;
    }
  }
  const std::pair<std::string_view, const char*> suffixes[] = {
      {".lora_down.weight", "down"}, {".lora_up.weight", "up"}, {".lora_A.weight", "down"}, {".lora_B.weight", "up"}, {".alpha", "alpha"}};
  for (const auto& [s, part] : suffixes) {
    if (name.ends_with(s) && name.size() > s.size()) return LoraKey{std::string(name.substr(0, name.size() - s.size())), part};
  }
  return std::nullopt;
}

struct LoraFileInfo {
  std::string file;
  float strength;
  std::uint64_t xxh64;
  int modules;
};

LoraFileInfo load_lora(const LoraInput& in, LoraMap& out) {
  weights::SafetensorsFile f(in.file);
  weights::FileReader reader(in.file);
  struct Parts {
    const weights::TensorInfo *down = nullptr, *up = nullptr, *alpha = nullptr;
  };
  std::map<std::string, Parts> mods;
  for (const auto& t : f.tensors()) {
    auto key = parse_lora_key(t.name);
    if (!key) throw weights::FormatError(in.file, t.name, "not a LoRA tensor (expected <module>.lora_{down,up,A,B}.weight or <module>.alpha)");
    Parts& p = mods[key->module];
    const weights::TensorInfo*& slot = key->part == "down" ? p.down : key->part == "up" ? p.up : p.alpha;
    if (slot != nullptr) throw weights::FormatError(in.file, t.name, "duplicate LoRA part");
    slot = &t;
  }
  // The shared alpha rule (lora_metadata_alpha): without an alpha tensor, the diffusers / PEFT metadata alpha.
  std::optional<double> meta_alpha;
  try {
    meta_alpha = lora_metadata_alpha(f.metadata());
  } catch (const QuantError& e) {
    throw weights::FormatError(in.file, "", e.what());
  }
  Xxh64 hash;
  auto read = [&](const weights::TensorInfo& t) {
    std::vector<std::byte> raw(static_cast<std::size_t>(t.nbytes));
    reader.read_at(t.offset, raw.data(), raw.size());
    hash.update(raw.data(), raw.size());
    return raw;
  };
  for (const auto& [m, p] : mods) {
    if (p.down == nullptr || p.up == nullptr) throw weights::FormatError(in.file, m, "LoRA module needs both down and up weights");
    if (p.down->shape.size() != 2 || p.up->shape.size() != 2 || p.down->shape[0] != p.up->shape[1] || p.down->shape[0] <= 0) {
      throw weights::FormatError(in.file, m, "LoRA down [r, K] / up [N, r] shapes do not agree");
    }
    LoraFactor lf;
    lf.rank = p.down->shape[0];
    lf.k = p.down->shape[1];
    lf.n = p.up->shape[0];
    lf.down = to_floats(read(*p.down), p.down->dtype);
    lf.up = to_floats(read(*p.up), p.up->dtype);
    std::optional<double> alpha = meta_alpha;
    if (p.alpha != nullptr) {
      if (p.alpha->numel() != 1) throw weights::FormatError(in.file, p.alpha->name, "alpha must be a scalar");
      alpha = to_floats(read(*p.alpha), p.alpha->dtype)[0];
      if (!std::isfinite(*alpha)) throw weights::FormatError(in.file, p.alpha->name, "non-finite LoRA alpha");
    }
    for (float v : lf.down) {
      if (!std::isfinite(v)) throw weights::FormatError(in.file, m, "non-finite LoRA weight");
    }
    for (float v : lf.up) {
      if (!std::isfinite(v)) throw weights::FormatError(in.file, m, "non-finite LoRA weight");
    }
    lf.coef = lora_coef(in.strength, alpha, lf.rank);
    if (!std::isfinite(lf.coef)) throw weights::FormatError(in.file, m, "LoRA coefficient strength * alpha / rank is not finite");
    out[m].push_back(std::move(lf));
  }
  return {util::path_utf8(in.file.filename()), in.strength, hash.digest(), static_cast<int>(mods.size())};
}

// A fused-qkv LoRA lands on proj.q / proj.k / proj.v by thirds of its up rows.
void split_fused_lora(LoraMap& loras) {
  std::vector<std::string> fused;
  for (const auto& [m, v] : loras) {
    if (m.ends_with(".proj.qkv") && m.starts_with("layers.")) fused.push_back(m);
  }
  for (const std::string& m : fused) {
    const std::string base = m.substr(0, m.size() - 3);
    for (LoraFactor& f : loras[m]) {
      if (f.n % 3 != 0) fail("fused qkv LoRA '" + m + "' has " + std::to_string(f.n) + " rows");
      const std::int64_t rows = f.n / 3;
      const char* parts[] = {"q", "k", "v"};
      for (int i = 0; i < 3; ++i) {
        LoraFactor p;
        p.rank = f.rank;
        p.k = f.k;
        p.n = rows;
        p.coef = f.coef;
        p.down = f.down;
        p.up.assign(f.up.begin() + i * rows * f.rank, f.up.begin() + (i + 1) * rows * f.rank);
        loras[base + parts[i]].push_back(std::move(p));
      }
    }
    loras.erase(m);
  }
}

// ---- Output plan -------------------------------------------------------------------------------------------

struct OutTensor {
  std::string name;
  DType dtype;
  std::vector<std::int64_t> shape;
  std::uint64_t nbytes = 0;
  std::uint64_t offset = 0;  // absolute, filled by the layout
};

struct OutModule {
  std::string name;                       // as written (fused: layers.N.proj.qkv)
  std::vector<const QuantLinear*> parts;  // 1, or q/k/v
  QuantSpec spec;
  std::int64_t n = 0, k = 0;
  // Indices into the tensor list (scale/scale_2/config/bias: SIZE_MAX if absent; scale_2: nvfp4 only).
  std::size_t weight = 0, scale = 0, scale_2 = 0, config = 0, bias = 0;
};

// An nvfp4 module's tensors: packed [n, k/2], swizzled block scales, tensor scale.
struct Nvfp4Out {
  std::vector<std::uint8_t> packed, scales;
  float s2 = 0;
};

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

std::string comfy_quant_json(const QuantSpec& s) {
  if (s.format == QuantFormat::kInt8ConvRot) {
    return "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": " + std::to_string(s.group) + "}";
  }
  if (s.format == QuantFormat::kNvfp4) return "{\"format\": \"nvfp4\"}";
  return "{\"format\": \"int8_tensorwise\"}";
}

// The NVFP4 study's keep lists (E30, lists of gptq_token_none_int8: greedy by teacher-forced CFG velocity error on the
// calibration prompts; the rtn lists are the same).
constexpr std::string_view kKeepG090[] = {
    "b00.o", "b01.o", "b02.o", "b03.o", "b04.o", "b05.o", "b06.o", "b07.o", "b08.o", "b09.o", "b10.o",
    "b11.o", "b12.o", "b13.o", "b14.o", "b15.o", "b16.o", "b17.o", "b18.o", "b19.o", "b20.o", "b21.o",
    "b22.o", "b23.o", "b24.o", "b25.o", "b26.o", "b31.o", "b32.o", "b33.o", "b35.o", "b36.down", "b36.o",
    "b37.down", "b37.o", "b38.down", "b38.o", "b39.down", "b39.o", "b40.down", "b40.o", "b41.o"};
constexpr std::string_view kKeepG095[] = {"b00.o", "b01.o", "b02.o", "b04.o", "b06.o", "b08.o", "b13.o",
                                          "b14.o", "b15.o", "b16.o", "b17.o", "b21.o", "b23.o", "b24.o",
                                          "b25.o", "b36.o", "b37.down", "b37.o", "b40.down", "b40.o", "b41.o"};

const char* unit_suffix(GemmUnit u) {
  switch (u) {
    case GemmUnit::kQkv: return "qkv";
    case GemmUnit::kO: return "o";
    case GemmUnit::kUpGate: return "ffn_up_gate";
    case GemmUnit::kDown: return "ffn_down";
  }
  return "?";
}

// "b<N>.<qkv|o|up|down>" -> module name, or nullopt.
std::optional<std::string> study_unit_module(std::string_view s) {
  if (s.size() < 3 || s[0] != 'b') return std::nullopt;
  const std::size_t dot = s.find('.');
  if (dot == std::string_view::npos || dot < 2) return std::nullopt;
  int b = 0;
  for (char c : s.substr(1, dot - 1)) {
    if (c < '0' || c > '9' || b > 1000) return std::nullopt;
    b = b * 10 + (c - '0');
  }
  const std::string_view t = s.substr(dot + 1);
  const char* suffix = t == "qkv" ? "qkv" : t == "o" ? "o" : t == "up" ? "ffn_up_gate" : t == "down" ? "ffn_down" : nullptr;
  if (suffix == nullptr) return std::nullopt;
  return "layers." + std::to_string(b) + ".proj." + suffix;
}

constexpr std::string_view kAttentionConfig = "{\"attention\": \"comfy_kitchen_int8\"}";

class Writer {
 public:
  Writer(util::OutputFile& out, const std::vector<OutTensor>& tensors, const std::string& header)
      : path_(out.path()), out_(out) {
    const fs::path& path = path_;
    std::uint64_t len = header.size();
    char lb[8];
    for (int i = 0; i < 8; ++i) lb[i] = static_cast<char>((len >> (8 * i)) & 0xFF);
    out_.write(lb, 8);
    if (!out_.write(header)) fail("write failed: " + util::path_utf8(path));
    for (const auto& t : tensors) {
      end_ = std::max(end_, t.offset + t.nbytes);
      planned_ += t.nbytes;
    }
  }
  void write(const OutTensor& t, const void* data, std::size_t size) {
    if (size != t.nbytes) fail("internal: tensor '" + t.name + "' produced " + std::to_string(size) + " bytes, planned " + std::to_string(t.nbytes));
    if (!out_.seek(t.offset) || !out_.write(data, size)) fail("write failed: " + util::path_utf8(path_));
    written_ += size;
  }
  // Writes the final header over the first one (same length) and flushes the file to stable storage (so the rename
  // that publishes it cannot outrun its bytes); it stays open for PartialFile::commit.
  std::uint64_t finish(const std::string& header) {
    if (written_ != planned_) fail("internal: wrote " + std::to_string(written_) + " tensor bytes, planned " + std::to_string(planned_));
    if (!out_.seek(8) || !out_.write(header)) fail("write failed: " + util::path_utf8(path_));
    if (!out_.sync()) fail("cannot flush " + util::path_utf8(path_) + " to disk");
    if (util::file_info(path_).size != end_) fail("output size mismatch (internal)");
    return end_;
  }

 private:
  fs::path path_;
  util::OutputFile& out_;
  std::uint64_t end_ = 0, written_ = 0, planned_ = 0;
};

class Readers {
 public:
  explicit Readers(const QuantModel& m) : m_(m) {}
  std::vector<std::byte> read(const IndexedTensor& t) {
    auto& r = readers_[t.file];
    if (!r) r = std::make_unique<weights::FileReader>(m_.file_path(t.file));
    std::vector<std::byte> out(static_cast<std::size_t>(t.nbytes));
    r->read_at(t.offset, out.data(), out.size());
    return out;
  }

 private:
  const QuantModel& m_;
  std::map<std::uint32_t, std::unique_ptr<weights::FileReader>> readers_;
};

}  // namespace

bool glob_match(std::string_view p, std::string_view s) {
  // Iterative wildcard match with backtracking to the last '*'.
  std::size_t pi = 0, si = 0, star = std::string_view::npos, mark = 0;
  while (si < s.size()) {
    if (pi < p.size() && (p[pi] == '?' || p[pi] == s[si])) {
      ++pi;
      ++si;
    } else if (pi < p.size() && p[pi] == '*') {
      star = pi++;
      mark = si;
    } else if (star != std::string_view::npos) {
      pi = star + 1;
      si = ++mark;
    } else {
      return false;
    }
  }
  while (pi < p.size() && p[pi] == '*') ++pi;
  return pi == p.size();
}

std::optional<HessianUnit> gemm_unit_of(std::string_view module) {
  std::string_view rest = module;
  int b = 0;
  if (!rest.starts_with("layers.")) return std::nullopt;
  rest.remove_prefix(7);
  if (!weights::take_layer_index(rest, b) || !rest.starts_with("proj.")) return std::nullopt;
  rest.remove_prefix(5);
  HessianUnit u;
  u.block = b;
  if (rest == "qkv" || rest == "q" || rest == "k" || rest == "v") u.unit = GemmUnit::kQkv;
  else if (rest == "o") u.unit = GemmUnit::kO;
  else if (rest == "ffn_up_gate") u.unit = GemmUnit::kUpGate;
  else if (rest == "ffn_down") u.unit = GemmUnit::kDown;
  else return std::nullopt;
  u.module = "layers." + std::to_string(b) + ".proj." + unit_suffix(u.unit);
  return u;
}

std::vector<std::string> parse_nvfp4_keep(std::string_view spec, std::string* name) {
  std::vector<std::string> units;
  std::string label = "custom";
  if (spec == "none") {
    label = "none";
  } else if (spec == "g0.9" || spec == "g0.95") {
    label = std::string(spec);
    for (std::string_view u : spec == "g0.9" ? std::span<const std::string_view>(kKeepG090) : std::span<const std::string_view>(kKeepG095)) {
      units.push_back(*study_unit_module(u));
    }
  } else {
    std::size_t b = 0;
    while (b <= spec.size()) {
      const std::size_t e = std::min(spec.find(',', b), spec.size());
      const std::string_view item = spec.substr(b, e - b);
      b = e + 1;
      if (item.empty()) continue;
      std::optional<std::string> m = study_unit_module(item);
      if (!m) {
        const auto u = gemm_unit_of(item);
        if (u && u->module == item) m = u->module;
      }
      if (!m) {
        fail("--keep: '" + std::string(item) + "' is not a preset (none, g0.9, g0.95) or a block GEMM unit (b<N>.<qkv|o|up|down> or "
             "layers.<N>.proj.<qkv|o|ffn_up_gate|ffn_down>)");
      }
      units.push_back(*m);
    }
    if (units.empty()) fail("--keep: empty list (use 'none' to keep no unit)");
  }
  std::sort(units.begin(), units.end());
  units.erase(std::unique(units.begin(), units.end()), units.end());
  if (name != nullptr) *name = label;
  return units;
}

std::optional<double> lora_metadata_alpha(const std::map<std::string, std::string>& metadata) {
  const auto it = metadata.find("lora_adapter_metadata");
  if (it == metadata.end()) return std::nullopt;
  json::Value v;
  try {
    v = json::parse(it->second, json::ParseLimits{1 << 16});
  } catch (const json::ParseError& e) {
    fail(std::string("malformed lora_adapter_metadata: ") + e.what());
  }
  if (!v.is_object()) fail("lora_adapter_metadata is not a JSON object");
  if (const json::Value* p = v.find("alpha_pattern"); p != nullptr && p->is_object() && !p->as_object().empty()) {
    fail("lora_adapter_metadata alpha_pattern (per-module alphas) is not supported");
  }
  const json::Value* a = v.find("lora_alpha");
  if (a == nullptr || a->is_null()) return std::nullopt;
  if (!a->is_number() || !std::isfinite(a->as_double())) fail("lora_adapter_metadata lora_alpha is not a finite number");
  return a->as_double();
}

double lora_alpha_scale(std::optional<double> alpha, std::int64_t rank) {
  return alpha ? *alpha / static_cast<double>(rank) : 1.0;
}

float lora_coef(float strength, std::optional<double> alpha, std::int64_t rank) {
  // (strength * alpha / rank) * delta with the Python double rounded to fp32 by the fp32 tensor multiply.
  const double v = static_cast<double>(strength) * lora_alpha_scale(alpha, rank);
  // Outside the fp32 range (or NaN) the coefficient is reported as infinite; callers reject it.
  return std::fabs(v) <= std::numeric_limits<float>::max() ? static_cast<float>(v) : std::numeric_limits<float>::infinity();
}

void lora_merge_row(float* w, std::int64_t k, const float* up_row, const float* down, std::int64_t rank, float coef, double* acc) {
  std::fill(acc, acc + k, 0.0);
  for (std::int64_t r = 0; r < rank; ++r) {
    const double u = up_row[r];
    const float* d = down + r * k;
    for (std::int64_t c = 0; c < k; ++c) acc[c] += u * static_cast<double>(d[c]);
  }
  for (std::int64_t c = 0; c < k; ++c) w[c] = w[c] + coef * static_cast<float>(acc[c]);
}

namespace {

// `inner` is `outer` or lies inside the directory `outer` (both util::resolved_path).
bool same_or_inside(const fs::path& inner, const fs::path& outer) {
  const fs::path rel = inner.lexically_relative(outer);
  return !rel.empty() && *rel.begin() != "..";
}

// The output must not replace, or be written into, anything the conversion reads.
void check_output_path(const fs::path& input, const fs::path& output, const QuantModel& model, const ConvertOptions& options) {
  const fs::path out = util::resolved_path(output);
  const util::FileInfo out_info = util::file_info(output);
  auto clash = [&](const fs::path& in) {
    if (util::resolved_path(in) == out || util::same_file(util::file_info(in), out_info)) {
      fail("output " + util::path_utf8(output) + " would overwrite the input " + util::path_utf8(in));
    }
  };
  for (std::uint32_t i = 0; i < model.files.size(); ++i) clash(model.file_path(i));
  for (const LoraInput& li : options.loras) clash(li.file);
  if (options.nvfp4 && !options.nvfp4->hessian_file.empty()) clash(options.nvfp4->hessian_file);
  // A file written into the component directory read would become one of its shards the next time it is read.
  if (util::file_info(input).directory() && same_or_inside(out, util::resolved_path(model.base_dir))) {
    fail("output " + util::path_utf8(output) + " lies inside the input directory " + util::path_utf8(model.base_dir));
  }
}

// The output modules and their specs: one per GEMM, or the fused proj.qkv; throws for every misuse of the options.
std::vector<OutModule> plan_modules(const QuantModel& model, const ConvertOptions& options, const LoraMap& loras) {
  const Nvfp4Options* fp4 = options.nvfp4 ? &*options.nvfp4 : nullptr;
  const bool fuse_qkv = options.fuse_qkv || fp4 != nullptr;
  auto spec_of = [&](const QuantLinear& l) -> QuantSpec {
    const bool keep = std::any_of(options.keep_bf16.begin(), options.keep_bf16.end(), [&](const std::string& p) { return glob_match(p, l.name); });
    if (l.spec.format == QuantFormat::kNvfp4) {
      fail("'" + l.name + "' is stored as nvfp4; converting NVFP4 to INT8 would compound two roundings (convert from the bf16 "
           "weights instead)");
    }
    if (l.spec.quantized()) {
      if (keep) fail("--keep-bf16 matches '" + l.name + "', which the input stores quantized (the bf16 weights are gone)");
      if (loras.count(l.name)) fail("a LoRA targets '" + l.name + "', which the input stores quantized; merge into the bf16 weights instead");
      if (fp4 != nullptr) fail("'" + l.name + "' is stored " + describe(l.spec) + "; NVFP4 output is converted from the bf16 weights");
      return l.spec;
    }
    if (fp4 != nullptr) {
      if (keep) return QuantSpec::bf16();
      const auto u = gemm_unit_of(l.name);
      if (!u) fail("'" + l.name + "' is not a block GEMM unit; NVFP4 output covers the block GEMMs only");
      if (!std::binary_search(fp4->keep.begin(), fp4->keep.end(), u->module)) return QuantSpec::nvfp4();
      if (l.k % options.group != 0) fail("--keep: '" + u->module + "' has K=" + std::to_string(l.k) + ", not a multiple of the ConvRot group " + std::to_string(options.group));
      return QuantSpec::int8_convrot(options.group);
    }
    QuantSpec s = options.policy ? options.policy(l) : (keep ? QuantSpec::bf16() : QuantSpec::int8_convrot(options.group));
    if (s.format == QuantFormat::kInt8Plain || (s.format == QuantFormat::kInt8ConvRot && (!valid_convrot_group(s.group) || l.k % s.group != 0))) {
      fail("policy returned an unsupported spec " + describe(s) + " for '" + l.name + "'");
    }
    return s;
  };

  std::vector<OutModule> mods;
  std::set<std::string> fused_done;
  for (const QuantLinear& l : model.linears) {
    OutModule om;
    const bool qkv_part = l.name.ends_with(".proj.q") || l.name.ends_with(".proj.k") || l.name.ends_with(".proj.v");
    if (fuse_qkv && qkv_part) {
      const std::string base = l.name.substr(0, l.name.size() - 1);
      if (fused_done.count(base)) continue;
      fused_done.insert(base);
      om.name = base + "qkv";
      for (const char* p : {"q", "k", "v"}) {
        const QuantLinear* part = model.find_linear(base + p);
        if (part == nullptr) fail("--fuse-qkv: '" + base + p + "' missing");
        om.parts.push_back(part);
      }
      om.spec = spec_of(*om.parts[0]);
      for (const QuantLinear* p : om.parts) {
        if (spec_of(*p) != om.spec || p->k != om.parts[0]->k || p->n != om.parts[0]->n || p->bias.has_value() != om.parts[0]->bias.has_value() ||
            (p->bias && p->bias->dtype != om.parts[0]->bias->dtype)) {
          fail("--fuse-qkv: q/k/v of '" + base + "' differ in spec, shape or bias");
        }
      }
      om.n = 3 * om.parts[0]->n;
    } else {
      om.name = l.name;
      om.parts = {&l};
      om.spec = spec_of(l);
      om.n = l.n;
    }
    om.k = om.parts[0]->k;
    mods.push_back(std::move(om));
  }
  if (fp4 != nullptr) {
    for (const std::string& u : fp4->keep) {
      if (std::none_of(mods.begin(), mods.end(), [&](const OutModule& om) { return om.name == u; })) fail("--keep: unit '" + u + "' is not in the model");
    }
  }
  return mods;
}

// The nvfp4 modules of a plan as Hessian units, in (block, unit) order, with their indices into `mods`.
std::vector<std::pair<HessianUnit, std::size_t>> hessian_units_of(const std::vector<OutModule>& mods) {
  std::vector<std::pair<HessianUnit, std::size_t>> units;
  for (std::size_t i = 0; i < mods.size(); ++i) {
    if (mods[i].spec.format != QuantFormat::kNvfp4) continue;
    HessianUnit u = *gemm_unit_of(mods[i].name);
    u.k = mods[i].k;
    units.emplace_back(std::move(u), i);
  }
  std::sort(units.begin(), units.end(), [](const auto& a, const auto& b) { return std::pair(a.first.block, a.first.unit) < std::pair(b.first.block, b.first.unit); });
  return units;
}

// convert()'s source digest of a model read without LoRAs: XXH64 of (canonical name, XXH64 of the bytes) pairs, sorted
// by name, over every stored tensor of every linear and every other tensor. Computed up front, from `threads` readers.
std::uint64_t source_digest(const QuantModel& model, int threads) {
  std::vector<std::pair<std::string, const IndexedTensor*>> all;
  for (const QuantLinear& l : model.linears) {
    all.emplace_back(l.name + ".weight", &l.weight);
    const std::pair<const std::optional<IndexedTensor>*, const char*> stored[] = {
        {&l.scale, "weight_scale"}, {&l.s_rel, "weight_s_rel"}, {&l.s_channel, "weight_s_channel"}, {&l.codebook, "weight_codebook"}, {&l.bias, "bias"}};
    for (const auto& [opt, suffix] : stored) {
      if (opt->has_value() && (l.spec.quantized() || std::string_view(suffix) == "bias")) all.emplace_back(l.name + "." + suffix, &**opt);
    }
  }
  for (const IndexedTensor& t : model.tensors) all.emplace_back(t.name, &t);
  std::vector<std::uint64_t> digests(all.size());
  std::atomic<std::size_t> next{0};
  parallel_for(threads > 0 ? std::min<std::int64_t>(threads, 16) : 16, threads, [&](std::int64_t, std::int64_t) {
    Readers readers(model);
    for (std::size_t i = next++; i < all.size(); i = next++) {
      const std::vector<std::byte> b = readers.read(*all[i].second);
      digests[i] = xxh64(b.data(), b.size());
    }
  });
  std::map<std::string, std::uint64_t> sorted;
  for (std::size_t i = 0; i < all.size(); ++i) sorted[all[i].first] = digests[i];
  Xxh64 h;
  for (const auto& [name, d] : sorted) {
    h.update(name.data(), name.size());
    h.update("\0", 1);
    h.update(&d, sizeof d);
  }
  return h.digest();
}

// One unit's Hessian digest: XXH64 of its module name and its upper triangle (row by row, fp32 bytes).
std::uint64_t hessian_unit_digest(const HessianUnit& u, const float* h) {
  Xxh64 d;
  d.update(u.module.data(), u.module.size());
  for (std::int64_t i = 0; i < u.k; ++i) d.update(h + i * u.k + i, static_cast<std::size_t>(u.k - i) * sizeof(float));
  return d.digest();
}

void check_capture(const HessianCapture& cap) {
  if (cap.calibration.size() > kMaxCaptureStringBytes || cap.capture_class.size() > kMaxCaptureStringBytes) {
    fail("internal: calibration capture strings longer than " + std::to_string(kMaxCaptureStringBytes) + " bytes");
  }
}
// Header bytes reserved for the two capture strings (each at most kMaxCaptureStringBytes, JSON-escaped: up to 2x).
constexpr std::size_t kCaptureReserve = 4 * kMaxCaptureStringBytes;

// A Hessian file as a source for exactly `units` (each must be in it, with its k).
HessianCapture read_hessian_file(const fs::path& path, const std::vector<HessianUnit>& units, const HessianSink& sink) {
  const weights::SafetensorsFile f(path);
  weights::FileReader reader(path);
  const auto cal = f.metadata().find("mossinfer_nvfp4_calibration");
  const auto cls = f.metadata().find("mossinfer_nvfp4_capture_class");
  if (cal == f.metadata().end() || cls == f.metadata().end()) fail(util::path_utf8(path) + " is not a Hessian file of moss-infer quantize");
  std::vector<float> packed, h;
  for (std::size_t u = 0; u < units.size(); ++u) {
    const std::int64_t k = units[u].k;
    const weights::TensorInfo* t = f.find(units[u].module);
    if (t == nullptr || t->dtype != DType::kF32 || t->shape != std::vector<std::int64_t>{k * (k + 1) / 2}) {
      fail(util::path_utf8(path) + " holds no Hessian of K=" + std::to_string(k) + " for '" + units[u].module + "'");
    }
    packed.resize(static_cast<std::size_t>(k * (k + 1) / 2));
    reader.read_at(t->offset, packed.data(), packed.size() * sizeof(float));
    h.assign(static_cast<std::size_t>(k * k), 0.0f);
    std::size_t p = 0;
    for (std::int64_t i = 0; i < k; ++i) {
      for (std::int64_t j = i; j < k; ++j) h[static_cast<std::size_t>(i * k + j)] = packed[p++];
    }
    sink(u, h.data());
  }
  return HessianCapture{cal->second, cls->second};
}

}  // namespace

ConvertStats convert(const fs::path& input, const fs::path& output, const ConvertOptions& options) {
  const auto t0 = std::chrono::steady_clock::now();
  auto log = [&](const std::string& s) {
    if (options.log) options.log(s);
  };
  // The boundary check of every path the conversion takes (util::check_user_path), before any file-system call.
  util::check_user_path(input, "input");
  util::check_user_path(output, "output");
  for (const LoraInput& li : options.loras) util::check_user_path(li.file, "LoRA");
  if (!valid_convrot_group(options.group)) fail("--group must be a power of 4 in [4, 1024]");
  if (options.threads < 0) fail("--threads must be >= 0");
  for (const LoraInput& li : options.loras) {
    if (!std::isfinite(li.strength)) fail("LoRA strength of " + util::path_utf8(li.file) + " is not finite");
  }
  const Nvfp4Options* fp4 = options.nvfp4 ? &*options.nvfp4 : nullptr;
  const bool gptq = fp4 != nullptr && needs_hessian(fp4->recipe);
  if (fp4 != nullptr) {
    if (options.policy) fail("a spec policy cannot be combined with nvfp4 output");
    if (!fp4->hessian_file.empty()) util::check_user_path(fp4->hessian_file, "Hessian file");
    if (gptq && fp4->hessians && !fp4->hessian_file.empty()) fail("internal: both a Hessian source and a Hessian file");
    if (gptq && !fp4->hessians && fp4->hessian_file.empty()) {
      fail(std::string("recipe ") + recipe_name(fp4->recipe) + " needs calibration Hessians, and no capture is configured");
    }
    if (!gptq && !fp4->hessian_file.empty()) fail(std::string("recipe ") + recipe_name(fp4->recipe) + " takes no Hessian file");
    if (gptq && !options.loras.empty()) {
      fail(std::string("LoRAs cannot be merged under recipe ") + recipe_name(fp4->recipe) +
           ": its calibration capture runs the model without them (use --recipe rtn or mse, or convert a merged bf16 file)");
    }
  }

  QuantModel model = open_quant_model(input, ModelKind::kDiT);
  if (model.kind != ModelKind::kDiT) fail("only the DiT can be converted");
  const bool gguf = !model.gguf_types.empty();
  if (gguf && fp4 != nullptr) fail("NVFP4 output is converted from bf16 weights, not from a GGUF file");
  for (const auto& [key, value] : options.metadata) {
    if (!key.starts_with("mossinfer_")) fail("internal: extra metadata key '" + key + "' must start with mossinfer_");
  }
  check_output_path(input, output, model, options);
  log("input: " + std::to_string(model.linears.size()) + " GEMMs, " + std::to_string(model.tensors.size()) + " other tensors");

  ConvertStats stats;
  LoraMap loras;
  std::vector<LoraFileInfo> lora_info;
  for (const LoraInput& li : options.loras) lora_info.push_back(load_lora(li, loras));
  split_fused_lora(loras);
  std::set<std::string> lora_used;

  std::vector<OutModule> mods = plan_modules(model, options, loras);
  // Tensor list.
  std::vector<OutTensor> tensors;
  auto add = [&](std::string name, DType d, std::vector<std::int64_t> shape) {
    const auto numel = util::checked_numel(shape);
    if (!numel) fail("tensor '" + name + "' is too large");
    tensors.push_back({std::move(name), d, std::move(shape), static_cast<std::uint64_t>(*numel) * weights::dtype_size(d), 0});
    return tensors.size() - 1;
  };
  std::vector<std::string> kept;
  std::map<std::string, int> spec_counts;  // describe(spec) -> GEMMs written with it
  for (OutModule& om : mods) {
    om.scale = om.scale_2 = om.config = om.bias = kNone;
    spec_counts[describe(om.spec)] += static_cast<int>(om.parts.size());
    if (om.spec.format == QuantFormat::kNvfp4) {
      if (om.k % kNvfp4Block != 0) fail("'" + om.name + "': K=" + std::to_string(om.k) + " is not a multiple of 16, which nvfp4 needs");
      om.weight = add(om.name + ".weight", DType::kU8, {om.n, om.k / 2});
      om.scale = add(om.name + ".weight_scale", DType::kF8E4M3, {nvfp4_scale_rows(om.n), nvfp4_scale_cols(om.k / kNvfp4Block)});
      om.scale_2 = add(om.name + ".weight_scale_2", DType::kF32, {});
      om.config = add(om.name + ".comfy_quant", DType::kU8, {static_cast<std::int64_t>(comfy_quant_json(om.spec).size())});
    } else if (om.spec.quantized()) {
      om.weight = add(om.name + ".weight", DType::kI8, {om.n, om.k});
      om.scale = add(om.name + ".weight_scale", DType::kF32, {om.n, 1});
      om.config = add(om.name + ".comfy_quant", DType::kU8, {static_cast<std::int64_t>(comfy_quant_json(om.spec).size())});
    } else {
      om.weight = add(om.name + ".weight", DType::kBF16, {om.n, om.k});
      for (const QuantLinear* p : om.parts) kept.push_back(p->name);
    }
    if (om.parts[0]->bias) om.bias = add(om.name + ".bias", om.parts[0]->bias->dtype, {om.n});
  }
  std::vector<std::size_t> copy_index;  // model.tensors[i] -> tensor list index
  for (const IndexedTensor& t : model.tensors) copy_index.push_back(add(t.name, t.dtype, t.shape));
  std::vector<std::size_t> attn_index;
  if (options.attention_config) {
    // One config per DiT block that has a linear; linears outside `layers.N.` (none in the official layout) get none.
    std::set<int> blocks;
    for (const QuantLinear& l : model.linears) {
      std::string_view rest = l.name;
      int b = 0;
      if (rest.starts_with("layers.")) {
        rest.remove_prefix(7);
        if (weights::take_layer_index(rest, b)) blocks.insert(b);
      }
    }
    for (int b : blocks) {
      attn_index.push_back(add("layers." + std::to_string(b) + ".comfy_attention.config", DType::kU8, {static_cast<std::int64_t>(kAttentionConfig.size())}));
    }
  }
  {
    std::set<std::string> names;
    for (const OutTensor& t : tensors) {
      if (!names.insert(t.name).second) fail("output tensor '" + t.name + "' planned twice");
    }
  }

  // Layout: safetensors order, compact JSON header padded with spaces to 8 bytes.
  std::vector<std::size_t> order(tensors.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const int ra = dtype_rank(tensors[a].dtype), rb = dtype_rank(tensors[b].dtype);
    return ra != rb ? ra > rb : tensors[a].name < tensors[b].name;
  });
  std::uint64_t cursor = 0;
  std::vector<std::uint64_t> rel(tensors.size());
  for (std::size_t i : order) {
    rel[i] = cursor;
    cursor += tensors[i].nbytes;
  }

  // Source tensors are hashed as they are read: digest of (canonical name, XXH64 of its bytes) pairs sorted by name.
  std::map<std::string, std::uint64_t> source_digests;
  auto note_source = [&](const std::string& name, const std::vector<std::byte>& bytes) { source_digests[name] = xxh64(bytes.data(), bytes.size()); };

  Readers readers(model);
  // Validate LoRA targets before writing anything.
  for (const auto& [m, fs_] : loras) {
    const QuantLinear* l = model.find_linear(m);
    const IndexedTensor* t = l == nullptr ? model.find_tensor(m + ".weight") : nullptr;
    const std::int64_t n = l ? l->n : (t && t->shape.size() == 2 ? t->shape[0] : -1);
    const std::int64_t k = l ? l->k : (t && t->shape.size() == 2 ? t->shape[1] : -1);
    if (l == nullptr && (t == nullptr || t->shape.size() != 2 || (t->dtype != DType::kBF16 && t->dtype != DType::kF32))) {
      fail("LoRA module '" + m + "' matches no 2-D bf16/f32 weight of the model");
    }
    for (const LoraFactor& f : fs_) {
      if (f.n != n || f.k != k) fail("LoRA module '" + m + "' has shape [" + std::to_string(f.n) + ", " + std::to_string(f.k) + "], the weight is [" + std::to_string(n) + ", " + std::to_string(k) + "]");
    }
  }
  const int threads = options.threads;

  // NVFP4: a module's weights as fp32 [n, k] (parts stacked, LoRAs merged), hashed as they are read; then its tensors.
  auto module_f32 = [&](const OutModule& om) {
    std::vector<float> w(static_cast<std::size_t>(om.n * om.k));
    std::int64_t row0 = 0;
    for (const QuantLinear* p : om.parts) {
      const auto lf = loras.find(p->name);
      const std::vector<LoraFactor>* lfs = lf == loras.end() ? nullptr : &lf->second;
      if (lfs) {
        lora_used.insert(p->name);
        ++stats.lora_modules;
      }
      const std::vector<std::byte> src = readers.read(p->weight);
      note_source(p->name + ".weight", src);
      const auto* w16 = reinterpret_cast<const std::uint16_t*>(src.data());
      const std::int64_t k = om.k;
      float* dst = w.data() + row0 * k;
      std::atomic<bool> bad{false};
      parallel_for(p->n, threads, [&](std::int64_t b, std::int64_t e) {
        std::vector<double> acc(lfs ? static_cast<std::size_t>(k) : 0);
        for (std::int64_t r = b; r < e; ++r) {
          float* row = dst + r * k;
          for (std::int64_t c = 0; c < k; ++c) row[c] = bf16_to_float(w16[r * k + c]);
          if (lfs) {
            for (const LoraFactor& f : *lfs) lora_merge_row(row, k, f.up.data() + r * f.rank, f.down.data(), f.rank, f.coef, acc.data());
          }
          for (std::int64_t c = 0; c < k; ++c) {
            if (!std::isfinite(row[c])) bad = true;
          }
        }
      });
      if (bad) fail("'" + p->name + "' holds non-finite weights" + (lfs ? " after the LoRA merge" : ""));
      row0 += p->n;
    }
    return w;
  };
  auto quantize_nvfp4_module = [&](const OutModule& om, const std::vector<float>& w, const GptqFactor* factor) {
    Nvfp4Out o;
    const std::int64_t k = om.k, sfk = k / kNvfp4Block;
    o.s2 = nvfp4_weight_tensor_scale(w.data(), om.n * k, threads);
    o.packed.resize(static_cast<std::size_t>(om.n * k / 2));
    std::vector<std::uint8_t> sf(static_cast<std::size_t>(om.n * sfk));
    if (factor != nullptr) {
      gptq_quantize_rows(w.data(), om.n, k, o.s2, *factor, fp4->recipe == Nvfp4Recipe::kGptqMse, threads, o.packed.data(), sf.data());
    } else {
      const bool mse = fp4->recipe == Nvfp4Recipe::kMse;
      parallel_for(om.n, threads, [&](std::int64_t b, std::int64_t e) {
        quantize_nvfp4_rows_recipe(w.data() + b * k, e - b, k, o.s2, mse, o.packed.data() + b * (k / 2), sf.data() + b * sfk);
      });
    }
    o.scales.resize(static_cast<std::size_t>(nvfp4_scale_rows(om.n) * nvfp4_scale_cols(sfk)));
    swizzle_scales(sf.data(), om.n, sfk, o.scales.data());
    return o;
  };

  // gptq recipes: the Hessian units in (block, unit) order; from a Hessian file only if it was captured from this DiT.
  const std::vector<std::pair<HessianUnit, std::size_t>> unit_plan = gptq ? hessian_units_of(mods) : std::vector<std::pair<HessianUnit, std::size_t>>{};
  std::optional<std::uint64_t> expected_source;
  if (gptq && !fp4->hessian_file.empty()) {
    const weights::SafetensorsFile hf(fp4->hessian_file);
    const auto it = hf.metadata().find("mossinfer_nvfp4_dit_xxh64");
    if (it == hf.metadata().end()) fail(util::path_utf8(fp4->hessian_file) + " is not a Hessian file of moss-infer quantize");
    log("checking the input against the Hessian file's DiT");
    expected_source = source_digest(model, threads);
    if (it->second != hex64(*expected_source)) {
      fail(util::path_utf8(fp4->hessian_file) + " was captured from another DiT (source xxh64 " + it->second + ", this input " +
           hex64(*expected_source) + ")");
    }
  }

  // Metadata: the source's keys (minus the legacy quantization header and earlier mossinfer_* keys, which would
  // misdescribe this file), then ours. It needs the source hash, known only after reading everything, and for gptq
  // recipes the capture's strings and Hessian digest, known only after the capture: the header is reserved with
  // fixed-width placeholders (for the capture strings, kCaptureReserve bytes of JSON whitespace after the metadata) and
  // rewritten at the end, so the gptq modules can be written as their Hessians arrive.
  const HessianCapture no_capture;
  auto build_header = [&](const std::string& source_hex, const HessianCapture& cap, const std::string& hessian_hex) {
    auto metadata = [&](const HessianCapture& c) {
      std::map<std::string, std::string> md;
      for (const auto& [k, v] : model.metadata) {
        if (k != "_quantization_metadata" && !k.starts_with("mossinfer_")) md[k] = v;
      }
      std::string kept_json = "[";
      for (std::size_t i = 0; i < kept.size(); ++i) kept_json += (i ? ", " : "") + quote(kept[i]);
      kept_json += "]";
      std::string specs;
      for (const auto& [name, count] : spec_counts) specs += (specs.empty() ? "" : ", ") + name + " x" + std::to_string(count);
      if (fp4 == nullptr) {
        md["mossinfer_converter"] = "moss-infer quantize " + std::to_string(kConverterVersion) + " (" + specs +
                                    (gguf ? "; GGUF input dequantized to fp32, then quantized round-to-nearest or rounded to bf16)"
                                          : "; bf16 inputs quantized round-to-nearest, quantized inputs copied or decoded exactly)");
      } else {
        md["mossinfer_converter"] = "moss-infer quantize " + std::to_string(kConverterVersion) + " (" + specs + "; nvfp4 recipe " +
                                    recipe_name(fp4->recipe) + ", int8_convrot round-to-nearest)";
        std::string keep_json = "[";
        for (std::size_t i = 0; i < fp4->keep.size(); ++i) keep_json += (i ? ", " : "") + quote(fp4->keep[i]);
        md["mossinfer_nvfp4"] = "{\"converter_version\": " + std::to_string(kNvfp4ConverterVersion) + ", \"recipe\": " + quote(recipe_name(fp4->recipe)) +
                                ", \"tensor_scale\": \"amax / 2688 (fp32)\", \"keep\": " + quote(fp4->keep_name) + ", \"kept_int8\": " + keep_json + "]}";
        if (gptq) {
          md["mossinfer_nvfp4_calibration"] = c.calibration;
          md["mossinfer_nvfp4_capture_class"] = c.capture_class;
          md["mossinfer_nvfp4_hessian_xxh64"] = hessian_hex;
        }
      }
      md["mossinfer_kept_bf16"] = kept_json;
      md["mossinfer_source_xxh64"] = source_hex;
      for (const auto& [key, value] : options.metadata) md[key] = value;
      if (!lora_info.empty()) {
        std::string j = "[";
        for (std::size_t i = 0; i < lora_info.size(); ++i) {
          char s[32];
          std::snprintf(s, sizeof s, "%.9g", static_cast<double>(lora_info[i].strength));
          j += (i ? ", " : "") + std::string("{\"file\": ") + quote(lora_info[i].file) + ", \"strength\": " + s + ", \"xxh64\": \"" + hex64(lora_info[i].xxh64) + "\"}";
        }
        md["mossinfer_lora"] = j + "]";
      }
      std::string h = "{\"__metadata__\":{";
      bool first = true;
      for (const auto& [k, v] : md) {
        h += (first ? "" : ",") + quote(k) + ":" + quote(v);
        first = false;
      }
      return h + "}";
    };
    std::string h = metadata(cap);
    if (gptq) {
      const std::size_t room = metadata(no_capture).size() + kCaptureReserve;
      if (h.size() > room) fail("internal: the capture strings exceed the header reserve");
      h.append(room - h.size(), ' ');
    }
    for (std::size_t i : order) {
      const OutTensor& t = tensors[i];
      h += "," + quote(t.name) + ":{\"dtype\":\"" + weights::dtype_name(t.dtype) + "\",\"shape\":[";
      for (std::size_t d = 0; d < t.shape.size(); ++d) h += (d ? "," : "") + std::to_string(t.shape[d]);
      h += "],\"data_offsets\":[" + std::to_string(rel[i]) + "," + std::to_string(rel[i] + t.nbytes) + "]}";
    }
    h += "}";
    h.append((8 - h.size() % 8) % 8, ' ');
    return h;
  };
  const std::string zeros(16, '0');
  const std::string header0 = build_header(zeros, no_capture, zeros);
  const std::uint64_t data_start = 8 + header0.size();
  for (std::size_t i = 0; i < tensors.size(); ++i) tensors[i].offset = data_start + rel[i];

  // The output is created before any long work, so an unwritable destination fails at once.
  PartialFile partial(output);
  Writer writer(partial.file(), tensors, header0);
  constexpr std::int64_t kBatch = 64;  // rows widened (and LoRA-merged) per quantize call

  int done = 0;
  std::vector<char> written(mods.size(), 0);
  auto write_nvfp4 = [&](std::size_t mi, const Nvfp4Out& o) {
    const OutModule& om = mods[mi];
    writer.write(tensors[om.weight], o.packed.data(), o.packed.size());
    writer.write(tensors[om.scale], o.scales.data(), o.scales.size());
    writer.write(tensors[om.scale_2], &o.s2, sizeof o.s2);
    const std::string cfg = comfy_quant_json(om.spec);
    writer.write(tensors[om.config], cfg.data(), cfg.size());
    std::vector<std::byte> bias_bytes;
    for (const QuantLinear* p : om.parts) {
      if (!p->bias) continue;
      const std::vector<std::byte> b = readers.read(*p->bias);
      note_source(p->name + ".bias", b);
      bias_bytes.insert(bias_bytes.end(), b.begin(), b.end());
    }
    if (om.bias != kNone) writer.write(tensors[om.bias], bias_bytes.data(), bias_bytes.size());
    stats.quantized += static_cast<int>(om.parts.size());
    stats.nvfp4 += static_cast<int>(om.parts.size());
    written[mi] = 1;
    if (++done % 24 == 0) log("  " + std::to_string(done) + "/" + std::to_string(mods.size()) + " GEMMs");
  };

  // gptq recipes: each nvfp4 module is quantized and written as its Hessian arrives.
  HessianCapture capture;
  std::string hessian_hex = zeros;
  if (gptq) {
    std::vector<HessianUnit> units;
    for (const auto& [u, mi] : unit_plan) units.push_back(u);
    std::vector<char> seen(units.size(), 0);
    std::vector<std::uint64_t> unit_digest(units.size(), 0);
    double factor_s = 0, solve_s = 0;
    const auto tc = std::chrono::steady_clock::now();
    log("capturing calibration Hessians for " + std::to_string(units.size()) + " nvfp4 GEMMs");
    const HessianSink sink = [&](std::size_t u, const float* h) {
      if (u >= units.size() || seen[u]) fail("internal: a Hessian delivered twice or for an unknown unit");
      seen[u] = 1;
      unit_digest[u] = hessian_unit_digest(units[u], h);
      const std::size_t mi = unit_plan[u].second;
      const OutModule& om = mods[mi];
      const std::vector<float> w = module_f32(om);
      const auto ta = std::chrono::steady_clock::now();
      const GptqFactor f = gptq_factor(h, om.k, threads);
      const auto tb = std::chrono::steady_clock::now();
      write_nvfp4(mi, quantize_nvfp4_module(om, w, &f));
      factor_s += std::chrono::duration<double>(tb - ta).count();
      solve_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
    };
    capture = fp4->hessian_file.empty() ? fp4->hessians(units, sink) : read_hessian_file(fp4->hessian_file, units, sink);
    for (std::size_t u = 0; u < units.size(); ++u) {
      if (!seen[u]) fail("the calibration capture delivered no Hessian for '" + units[u].module + "'");
    }
    check_capture(capture);
    Xxh64 hd;
    for (std::uint64_t d : unit_digest) hd.update(&d, sizeof d);
    hessian_hex = hex64(hd.digest());
    log("calibration and GPTQ: " + std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count()) +
        " s (GPTQ factors " + std::to_string(factor_s) + " s, rows " + std::to_string(solve_s) + " s)");
  }

  for (std::size_t mi = 0; mi < mods.size(); ++mi) {
    const OutModule& om = mods[mi];
    const std::int64_t k = om.k;
    if (written[mi]) continue;
    if (om.spec.format == QuantFormat::kNvfp4) {
      write_nvfp4(mi, quantize_nvfp4_module(om, module_f32(om), nullptr));
      continue;
    }
    std::vector<std::int8_t> q;
    std::vector<float> scale;
    std::vector<std::uint16_t> bf;
    if (om.spec.quantized()) {
      q.resize(static_cast<std::size_t>(om.n * k));
      scale.resize(static_cast<std::size_t>(om.n));
    } else {
      bf.resize(static_cast<std::size_t>(om.n * k));
    }
    std::vector<std::byte> bias_bytes;
    std::int64_t row0 = 0;
    for (const QuantLinear* p : om.parts) {
      const auto lf = loras.find(p->name);
      const std::vector<LoraFactor>* lfs = lf == loras.end() ? nullptr : &lf->second;
      if (lfs) {
        lora_used.insert(p->name);
        ++stats.lora_modules;
      }
      if (p->spec.quantized()) {
        // Copy or exact decode; hash every stored tensor of the module.
        const std::pair<const std::optional<IndexedTensor>*, const char*> stored[] = {
            {nullptr, "weight"}, {&p->scale, "weight_scale"}, {&p->s_rel, "weight_s_rel"}, {&p->s_channel, "weight_s_channel"}, {&p->codebook, "weight_codebook"}};
        for (const auto& [opt, suffix] : stored) {
          if (opt != nullptr && !opt->has_value()) continue;
          const IndexedTensor& t = opt == nullptr ? p->weight : **opt;
          const std::vector<std::byte> b = readers.read(t);
          note_source(p->name + "." + suffix, b);
          if (p->encoding != Encoding::kRaw) continue;
          if (opt == nullptr) std::memcpy(q.data() + row0 * k, b.data(), b.size());
          else std::memcpy(scale.data() + row0, b.data(), b.size());
        }
        if (p->encoding != Encoding::kRaw) {
          decode_linear(model, *p, q.data() + row0 * k, scale.data() + row0, threads);  // validates the scales
          ++stats.decoded;
        } else {
          if (!valid_scales(scale.data() + row0, p->n)) fail("'" + p->name + "': a stored weight_scale is not finite and > 0");
          ++stats.copied;
        }
      } else {
        const std::vector<std::byte> src = readers.read(p->weight);
        note_source(p->name + ".weight", src);
        const auto* w16 = reinterpret_cast<const std::uint16_t*>(src.data());
        // A GGUF weight: its rows dequantized to fp32 in place of the bf16 widening.
        const std::uint32_t gtype = p->encoding == Encoding::kGguf ? model.gguf_types.at(p->weight.name) : 0;
        const std::uint64_t row_bytes = p->weight.nbytes / static_cast<std::uint64_t>(p->n);
        if (om.spec.quantized() || lfs || p->encoding == Encoding::kGguf) {
          std::atomic<bool> bad{false};
          parallel_for(p->n, threads, [&](std::int64_t b, std::int64_t e) {
            // Per-thread scratch, reused for every batch of rows.
            std::vector<float> rows(static_cast<std::size_t>(kBatch * k));
            std::vector<double> acc(lfs ? static_cast<std::size_t>(k) : 0);
            for (std::int64_t r0 = b; r0 < e; r0 += kBatch) {
              const std::int64_t nb = std::min(kBatch, e - r0);
              if (p->encoding == Encoding::kGguf) {
                weights::dequantize_gguf_rows(gtype, src.data() + static_cast<std::uint64_t>(r0) * row_bytes, nb, k, rows.data());
              } else {
                for (std::int64_t i = 0; i < nb * k; ++i) rows[static_cast<std::size_t>(i)] = bf16_to_float(w16[r0 * k + i]);
              }
              if (lfs) {
                for (std::int64_t r = 0; r < nb; ++r) {
                  for (const LoraFactor& f : *lfs) lora_merge_row(rows.data() + r * k, k, f.up.data() + (r0 + r) * f.rank, f.down.data(), f.rank, f.coef, acc.data());
                }
              }
              const std::int64_t o = row0 + r0;
              const bool ok = om.spec.quantized() ? quantize_convrot_rows(rows.data(), nb, k, om.spec.group, q.data() + o * k, scale.data() + o)
                                                  : to_bf16_finite(rows.data(), static_cast<std::size_t>(nb * k), bf.data() + o * k);
              if (!ok) bad = true;
            }
          });
          if (bad) fail("'" + p->name + "' holds non-finite weights" + (lfs ? " after the LoRA merge" : ""));
          if (om.spec.quantized()) ++stats.quantized;
        } else {
          std::memcpy(bf.data() + row0 * k, src.data(), src.size());
        }
        if (!om.spec.quantized()) ++stats.kept_bf16;
      }
      if (p->bias) {
        const std::vector<std::byte> b = readers.read(*p->bias);
        note_source(p->name + ".bias", b);
        bias_bytes.insert(bias_bytes.end(), b.begin(), b.end());
      }
      row0 += p->n;
    }
    if (om.spec.quantized()) {
      writer.write(tensors[om.weight], q.data(), q.size());
      writer.write(tensors[om.scale], scale.data(), scale.size() * sizeof(float));
      const std::string cfg = comfy_quant_json(om.spec);
      writer.write(tensors[om.config], cfg.data(), cfg.size());
    } else {
      writer.write(tensors[om.weight], bf.data(), bf.size() * 2);
    }
    if (om.bias != kNone) writer.write(tensors[om.bias], bias_bytes.data(), bias_bytes.size());
    if (++done % 24 == 0) log("  " + std::to_string(done) + "/" + std::to_string(mods.size()) + " GEMMs");
  }
  for (std::size_t i = 0; i < model.tensors.size(); ++i) {
    const IndexedTensor& t = model.tensors[i];
    std::vector<std::byte> b = readers.read(t);
    note_source(t.name, b);
    if (const auto g = model.gguf_types.find(t.name); g != model.gguf_types.end()) {
      // A GGUF tensor: dequantized, then in its output dtype (BF16 rounded to nearest even, or F32), finite.
      const std::int64_t k = t.shape.back(), numel = t.numel();
      std::vector<float> f(static_cast<std::size_t>(numel));
      weights::dequantize_gguf_rows(g->second, b.data(), numel / k, k, f.data());
      bool ok = std::all_of(f.begin(), f.end(), [](float x) { return std::isfinite(x); });
      if (t.dtype == DType::kBF16) {
        std::vector<std::uint16_t> v(f.size());
        ok = ok && to_bf16_finite(f.data(), f.size(), v.data());
        b.resize(v.size() * 2);
        std::memcpy(b.data(), v.data(), b.size());
      } else {
        b.resize(f.size() * 4);
        std::memcpy(b.data(), f.data(), b.size());
      }
      if (!ok) fail("'" + t.name + "' holds non-finite values");
    }
    const std::string module = t.name.ends_with(".weight") ? t.name.substr(0, t.name.size() - 7) : std::string();
    auto lf = module.empty() ? loras.end() : loras.find(module);
    if (lf != loras.end()) {
      // A LoRA on a non-GEMM weight (kept in its own dtype): merge in fp32, round back (bf16: nearest even).
      lora_used.insert(module);
      ++stats.lora_modules;
      const std::int64_t n = t.shape[0], k = t.shape[1];
      std::vector<float> w = to_floats(b, t.dtype);
      std::vector<double> acc(static_cast<std::size_t>(k));
      for (std::int64_t r = 0; r < n; ++r) {
        for (const LoraFactor& f : lf->second) lora_merge_row(w.data() + r * k, k, f.up.data() + r * f.rank, f.down.data(), f.rank, f.coef, acc.data());
      }
      bool ok = true;
      if (t.dtype == DType::kBF16) {
        std::vector<std::uint16_t> v(w.size());
        ok = to_bf16_finite(w.data(), w.size(), v.data());
        std::memcpy(b.data(), v.data(), v.size() * 2);
      } else {
        ok = std::all_of(w.begin(), w.end(), [](float x) { return std::isfinite(x); });
        std::memcpy(b.data(), w.data(), w.size() * 4);
      }
      if (!ok) fail("'" + t.name + "' holds non-finite weights after the LoRA merge");
    }
    writer.write(tensors[copy_index[i]], b.data(), b.size());
  }
  for (std::size_t i : attn_index) writer.write(tensors[i], kAttentionConfig.data(), kAttentionConfig.size());
  for (const auto& [m, v] : loras) {
    if (!lora_used.count(m)) fail("LoRA module '" + m + "' was not applied (internal)");
  }

  // Source digest, then the final header (same length: only the 16 hex digits change).
  Xxh64 h;
  for (const auto& [name, d] : source_digests) {
    h.update(name.data(), name.size());
    h.update("\0", 1);
    h.update(&d, sizeof d);
  }
  stats.source_xxh64 = h.digest();
  if (expected_source && *expected_source != stats.source_xxh64) fail("internal: the input changed while it was converted");
  const std::string header = build_header(hex64(stats.source_xxh64), capture, hessian_hex);
  if (header.size() != header0.size()) fail("internal: header size changed");
  stats.bytes_written = writer.finish(header);
  // The result must read back before it is published.
  {
    const weights::SafetensorsFile check(partial.path());
    if (check.tensors().size() != tensors.size() || check.file_size() != stats.bytes_written) fail("internal: written file does not read back as planned");
  }
  partial.commit(output);
  stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return stats;
}

std::vector<HessianUnit> nvfp4_hessian_units(const fs::path& input, const ConvertOptions& options) {
  util::check_user_path(input, "input");
  if (!options.nvfp4) fail("Hessian units need nvfp4 output");
  const QuantModel model = open_quant_model(input, ModelKind::kDiT);
  std::vector<HessianUnit> out;
  for (auto& [u, mi] : hessian_units_of(plan_modules(model, options, LoraMap{}))) out.push_back(std::move(u));
  return out;
}

HessianFileInfo write_hessian_file(const fs::path& input, const fs::path& path, const ConvertOptions& options) {
  util::check_user_path(input, "input");
  util::check_user_path(path, "Hessian file");
  if (!options.nvfp4 || !needs_hessian(options.nvfp4->recipe) || !options.nvfp4->hessians) {
    fail("a Hessian file needs nvfp4 output with a gptq recipe and a calibration capture");
  }
  if (!options.loras.empty()) fail("LoRAs cannot be merged under the gptq recipes: their calibration capture runs the model without them");
  const QuantModel model = open_quant_model(input, ModelKind::kDiT);
  if (model.kind != ModelKind::kDiT) fail("only the DiT can be converted");
  check_output_path(input, path, model, options);
  std::vector<HessianUnit> units;
  for (auto& [u, mi] : hessian_units_of(plan_modules(model, options, LoraMap{}))) units.push_back(std::move(u));
  HessianFileInfo info;
  info.units = units.size();
  info.dit_xxh64 = source_digest(model, options.threads);

  // Tensors by name, as the safetensors library orders a single-dtype file. The capture's strings are known only after
  // the capture, so the header keeps kCaptureReserve bytes of JSON whitespace after the metadata and is rewritten at
  // the end with the same length.
  std::vector<OutTensor> tensors;
  for (const HessianUnit& u : units) {
    tensors.push_back({u.module, DType::kF32, {u.k * (u.k + 1) / 2}, static_cast<std::uint64_t>(u.k * (u.k + 1) / 2) * 4, 0});
  }
  std::vector<std::size_t> order(tensors.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return tensors[a].name < tensors[b].name; });
  std::vector<std::uint64_t> rel(tensors.size());
  std::uint64_t cursor = 0;
  for (std::size_t i : order) {
    rel[i] = cursor;
    cursor += tensors[i].nbytes;
  }
  auto metadata = [&](const HessianCapture& c) {
    return "{\"__metadata__\":{" + quote("mossinfer_nvfp4_calibration") + ":" + quote(c.calibration) + "," + quote("mossinfer_nvfp4_capture_class") +
           ":" + quote(c.capture_class) + "," + quote("mossinfer_nvfp4_dit_xxh64") + ":" + quote(hex64(info.dit_xxh64)) + "}";
  };
  auto build_header = [&](const HessianCapture& c) {
    std::string h = metadata(c);
    const std::size_t room = metadata(HessianCapture{}).size() + kCaptureReserve;
    if (h.size() > room) fail("internal: the capture strings exceed the header reserve");
    h.append(room - h.size(), ' ');
    for (std::size_t i : order) {
      const OutTensor& t = tensors[i];
      h += "," + quote(t.name) + ":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(t.shape[0]) + "],\"data_offsets\":[" + std::to_string(rel[i]) + "," +
           std::to_string(rel[i] + t.nbytes) + "]}";
    }
    h += "}";
    h.append((8 - h.size() % 8) % 8, ' ');
    return h;
  };
  const std::string header0 = build_header(HessianCapture{});
  for (std::size_t i = 0; i < tensors.size(); ++i) tensors[i].offset = 8 + header0.size() + rel[i];
  PartialFile partial(path);
  Writer writer(partial.file(), tensors, header0);
  std::vector<char> seen(units.size(), 0);
  std::vector<float> packed;
  info.capture = options.nvfp4->hessians(units, [&](std::size_t u, const float* h) {
    if (u >= units.size() || seen[u]) fail("internal: a Hessian delivered twice or for an unknown unit");
    seen[u] = 1;
    const std::int64_t k = units[u].k;
    packed.resize(static_cast<std::size_t>(k * (k + 1) / 2));
    std::size_t p = 0;
    for (std::int64_t i = 0; i < k; ++i) {
      for (std::int64_t j = i; j < k; ++j) packed[p++] = h[i * k + j];
    }
    writer.write(tensors[u], packed.data(), packed.size() * sizeof(float));
  });
  for (std::size_t u = 0; u < units.size(); ++u) {
    if (!seen[u]) fail("the calibration capture delivered no Hessian for '" + units[u].module + "'");
  }
  check_capture(info.capture);
  const std::string header = build_header(info.capture);
  if (header.size() != header0.size()) fail("internal: Hessian file header size changed");
  const std::uint64_t size = writer.finish(header);
  // The result must read back before it is published.
  {
    const weights::SafetensorsFile check(partial.path());
    if (check.tensors().size() != tensors.size() || check.file_size() != size) fail("internal: written Hessian file does not read back as planned");
  }
  partial.commit(path);
  return info;
}

}  // namespace mi::quant
