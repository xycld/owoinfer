#include "weights/quant/quant_convert.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "util/checked_math.h"
#include "util/path.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/quant/quant_codec.h"
#include "weights/safetensors.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mi::quant {

namespace fs = std::filesystem;
using weights::DType;
using weights::IndexedTensor;

namespace {

[[noreturn]] void fail(const std::string& msg) { throw QuantError("mi::quant: " + msg); }

// Flushes a closed file's data to stable storage, so the rename that publishes it cannot outrun its bytes.
void sync_file(const fs::path& path) {
#ifdef _WIN32
  HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  const bool ok = h != INVALID_HANDLE_VALUE && FlushFileBuffers(h);
  if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  const bool ok = fd >= 0 && ::fsync(fd) == 0;
  if (fd >= 0) ::close(fd);
#endif
  if (!ok) fail("cannot flush " + util::path_utf8(path) + " to disk");
}

// The output while it is written: removed unless commit() published it.
class PartialFile {
 public:
  explicit PartialFile(fs::path path) : path_(std::move(path)) {}
  PartialFile(const PartialFile&) = delete;
  PartialFile& operator=(const PartialFile&) = delete;
  ~PartialFile() {
    if (!committed_) {
      std::error_code ec;
      fs::remove(path_, ec);
    }
  }
  const fs::path& path() const { return path_; }
  void commit(const fs::path& final_path) {
    fs::rename(path_, final_path);
    committed_ = true;
  }

 private:
  fs::path path_;
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
    std::optional<float> alpha;
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
  std::size_t weight = 0, scale = 0, config = 0, bias = 0;  // indices into the tensor list (scale/config/bias: SIZE_MAX if absent)
};

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

std::string comfy_quant_json(const QuantSpec& s) {
  if (s.format == QuantFormat::kInt8ConvRot) {
    return "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": " + std::to_string(s.group) + "}";
  }
  return "{\"format\": \"int8_tensorwise\"}";
}

constexpr std::string_view kAttentionConfig = "{\"attention\": \"comfy_kitchen_int8\"}";

class Writer {
 public:
  Writer(const fs::path& path, const std::vector<OutTensor>& tensors, const std::string& header) : path_(path) {
    out_.open(path, std::ios::binary | std::ios::trunc);
    if (!out_) fail("cannot create " + util::path_utf8(path));
    std::uint64_t len = header.size();
    char lb[8];
    for (int i = 0; i < 8; ++i) lb[i] = static_cast<char>((len >> (8 * i)) & 0xFF);
    out_.write(lb, 8);
    out_.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!out_) fail("write failed: " + util::path_utf8(path));
    for (const auto& t : tensors) {
      end_ = std::max(end_, t.offset + t.nbytes);
      planned_ += t.nbytes;
    }
  }
  void write(const OutTensor& t, const void* data, std::size_t size) {
    if (size != t.nbytes) fail("internal: tensor '" + t.name + "' produced " + std::to_string(size) + " bytes, planned " + std::to_string(t.nbytes));
    out_.seekp(static_cast<std::streamoff>(t.offset));
    out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out_) fail("write failed: " + util::path_utf8(path_));
    written_ += size;
  }
  std::uint64_t finish() {
    if (written_ != planned_) fail("internal: wrote " + std::to_string(written_) + " tensor bytes, planned " + std::to_string(planned_));
    out_.flush();
    if (!out_) fail("write failed: " + util::path_utf8(path_));
    out_.close();
    if (!out_) fail("cannot close " + util::path_utf8(path_));
    if (fs::file_size(path_) != end_) fail("output size mismatch (internal)");
    return end_;
  }

 private:
  fs::path path_;
  std::ofstream out_;
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

float lora_coef(float strength, std::optional<float> alpha, std::int64_t rank) {
  // ComfyUI: alpha = alpha_tensor.item() / rank (1.0 without alpha), applied as (strength * alpha) * delta with the
  // Python double rounded to fp32 by the fp32 tensor multiply.
  const double a = alpha ? static_cast<double>(*alpha) / static_cast<double>(rank) : 1.0;
  const double v = static_cast<double>(strength) * a;
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

// `inner` is `outer` or lies inside the directory `outer` (both weakly canonical).
bool same_or_inside(const fs::path& inner, const fs::path& outer) {
  const fs::path rel = inner.lexically_relative(outer);
  return !rel.empty() && *rel.begin() != "..";
}

// The output must not replace, or be written into, anything the conversion reads.
void check_output_path(const fs::path& input, const fs::path& output, const QuantModel& model, const ConvertOptions& options) {
  const fs::path out = fs::weakly_canonical(output);
  std::error_code ec;
  auto clash = [&](const fs::path& in) {
    if (fs::weakly_canonical(in) == out || (fs::exists(output, ec) && fs::equivalent(in, output, ec))) {
      fail("output " + util::path_utf8(output) + " would overwrite the input " + util::path_utf8(in));
    }
  };
  for (std::uint32_t i = 0; i < model.files.size(); ++i) clash(model.file_path(i));
  for (const LoraInput& li : options.loras) clash(li.file);
  // A file written into the component directory read would become one of its shards the next time it is read.
  if (fs::is_directory(input) && same_or_inside(out, fs::weakly_canonical(model.base_dir))) {
    fail("output " + util::path_utf8(output) + " lies inside the input directory " + util::path_utf8(model.base_dir));
  }
}

}  // namespace

ConvertStats convert(const fs::path& input, const fs::path& output, const ConvertOptions& options) {
  const auto t0 = std::chrono::steady_clock::now();
  auto log = [&](const std::string& s) {
    if (options.log) options.log(s);
  };
  if (!valid_convrot_group(options.group)) fail("--group must be a power of 4 in [4, 1024]");
  if (options.threads < 0) fail("--threads must be >= 0");
  for (const LoraInput& li : options.loras) {
    if (!std::isfinite(li.strength)) fail("LoRA strength of " + util::path_utf8(li.file) + " is not finite");
  }

  QuantModel model = open_quant_model(input, ModelKind::kDiT);
  if (model.kind != ModelKind::kDiT) fail("only the DiT can be converted");
  check_output_path(input, output, model, options);
  log("input: " + std::to_string(model.linears.size()) + " GEMMs, " + std::to_string(model.tensors.size()) + " other tensors");

  ConvertStats stats;
  LoraMap loras;
  std::vector<LoraFileInfo> lora_info;
  for (const LoraInput& li : options.loras) lora_info.push_back(load_lora(li, loras));
  split_fused_lora(loras);
  std::set<std::string> lora_used;

  // Specs.
  auto spec_of = [&](const QuantLinear& l) -> QuantSpec {
    const bool keep = std::any_of(options.keep_bf16.begin(), options.keep_bf16.end(), [&](const std::string& p) { return glob_match(p, l.name); });
    if (l.spec.quantized()) {
      if (keep) fail("--keep-bf16 matches '" + l.name + "', which the input stores quantized (the bf16 weights are gone)");
      if (loras.count(l.name)) fail("a LoRA targets '" + l.name + "', which the input stores quantized; merge into the bf16 weights instead");
      return l.spec;
    }
    QuantSpec s = options.policy ? options.policy(l) : (keep ? QuantSpec::bf16() : QuantSpec::int8_convrot(options.group));
    if (s.format == QuantFormat::kInt8Plain || (s.format == QuantFormat::kInt8ConvRot && (!valid_convrot_group(s.group) || l.k % s.group != 0))) {
      fail("policy returned an unsupported spec " + describe(s) + " for '" + l.name + "'");
    }
    return s;
  };

  // Output modules.
  std::vector<OutModule> mods;
  std::set<std::string> fused_done;
  for (const QuantLinear& l : model.linears) {
    OutModule om;
    const bool qkv_part = l.name.ends_with(".proj.q") || l.name.ends_with(".proj.k") || l.name.ends_with(".proj.v");
    if (options.fuse_qkv && qkv_part) {
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
    om.scale = om.config = om.bias = kNone;
    spec_counts[describe(om.spec)] += static_cast<int>(om.parts.size());
    if (om.spec.quantized()) {
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
    if (l != nullptr) spec_of(*l);  // rejects LoRAs on quantized inputs
  }

  // Metadata: the source's keys (minus the legacy quantization header and earlier mossinfer_* keys, which would
  // misdescribe this file), then ours. It needs the source hash, known only after reading everything: the header is
  // reserved with a fixed-width placeholder and rewritten at the end.
  auto build_header = [&](const std::string& source_hex) {
    std::map<std::string, std::string> md;
    for (const auto& [k, v] : model.metadata) {
      if (k != "_quantization_metadata" && !k.starts_with("mossinfer_")) md[k] = v;
    }
    std::string kept_json = "[";
    for (std::size_t i = 0; i < kept.size(); ++i) kept_json += (i ? ", " : "") + quote(kept[i]);
    kept_json += "]";
    std::string specs;
    for (const auto& [name, count] : spec_counts) specs += (specs.empty() ? "" : ", ") + name + " x" + std::to_string(count);
    md["mossinfer_converter"] = "moss-infer quantize " + std::to_string(kConverterVersion) + " (" + specs +
                                "; bf16 inputs quantized round-to-nearest, quantized inputs copied or decoded exactly)";
    md["mossinfer_kept_bf16"] = kept_json;
    md["mossinfer_source_xxh64"] = source_hex;
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
    h += "}";
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
  const std::string header0 = build_header(std::string(16, '0'));
  const std::uint64_t data_start = 8 + header0.size();
  for (std::size_t i = 0; i < tensors.size(); ++i) tensors[i].offset = data_start + rel[i];

  PartialFile partial(fs::path(output) += ".partial");
  Writer writer(partial.path(), tensors, header0);
  const int threads = options.threads;
  constexpr std::int64_t kBatch = 64;  // rows widened (and LoRA-merged) per quantize call

  int done = 0;
  for (const OutModule& om : mods) {
    const std::int64_t k = om.k;
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
        if (om.spec.quantized() || lfs) {
          std::atomic<bool> bad{false};
          parallel_for(p->n, threads, [&](std::int64_t b, std::int64_t e) {
            // Per-thread scratch, reused for every batch of rows.
            std::vector<float> rows(static_cast<std::size_t>(kBatch * k));
            std::vector<double> acc(lfs ? static_cast<std::size_t>(k) : 0);
            for (std::int64_t r0 = b; r0 < e; r0 += kBatch) {
              const std::int64_t nb = std::min(kBatch, e - r0);
              for (std::int64_t i = 0; i < nb * k; ++i) rows[static_cast<std::size_t>(i)] = bf16_to_float(w16[r0 * k + i]);
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
  const std::string header = build_header(hex64(stats.source_xxh64));
  if (header.size() != header0.size()) fail("internal: header size changed");
  stats.bytes_written = writer.finish();
  {
    std::fstream f(partial.path(), std::ios::binary | std::ios::in | std::ios::out);
    if (!f) fail("cannot reopen " + util::path_utf8(partial.path()));
    f.seekp(8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.flush();
    if (!f) fail("cannot finalize " + util::path_utf8(partial.path()));
    f.close();
    if (!f) fail("cannot close " + util::path_utf8(partial.path()));
  }
  sync_file(partial.path());
  // The result must read back before it is published.
  {
    const weights::SafetensorsFile check(partial.path());
    if (check.tensors().size() != tensors.size() || check.file_size() != stats.bytes_written) fail("internal: written file does not read back as planned");
  }
  partial.commit(output);
  stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return stats;
}

}  // namespace mi::quant
