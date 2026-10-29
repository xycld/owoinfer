#include "weights/model_index.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include "util/checked_math.h"
#include "util/json.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "util/replaceable_file.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"

namespace mi::weights {

namespace fs = std::filesystem;
using util::path_from_utf8;
using util::path_utf8;

namespace {

constexpr const char* kFormatName = "mossidx";
// Sanity bound on config layer counts (the real model has 32 text-encoder layers and 42 DiT blocks);
// group names are built from these counts before any tensor is seen.
constexpr std::int64_t kMaxLayers = 1024;
constexpr std::size_t kChecksumChunk = std::size_t{8} << 20;
constexpr std::string_view kShardIndexSuffix = ".safetensors.index.json";
constexpr std::string_view kShardSuffix = ".safetensors";

bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

// Reads a whole file, refusing one larger than max_bytes before allocating anything.
std::string slurp(const fs::path& path, std::uintmax_t max_bytes) {
  const util::FileInfo info = util::file_info(path);
  if (!info.regular()) throw IndexError("cannot open " + path_utf8(path));
  const std::uintmax_t size = info.size;
  if (size > max_bytes) {
    throw IndexError(path_utf8(path) + ": " + std::to_string(size) + " bytes exceeds the limit of " + std::to_string(max_bytes));
  }
  // Read through a concurrent writer's rename (`moss-infer index` replaces moss.mossidx atomically).
  std::string text;
  const util::ReadStatus status = util::read_replaceable_file(path, static_cast<std::size_t>(max_bytes), text);
  // The file may have grown between the size check and the read.
  if (status == util::ReadStatus::kTooLarge) {
    throw IndexError(path_utf8(path) + ": file grew past the limit of " + std::to_string(max_bytes) + " bytes");
  }
  if (status != util::ReadStatus::kOk) throw IndexError("cannot open " + path_utf8(path));
  return text;
}

json::Value parse_json_file(const fs::path& path, std::uintmax_t max_bytes, std::size_t max_values) {
  try {
    return json::parse(slurp(path, max_bytes), {.max_values = max_values});
  } catch (const json::ParseError& e) {
    throw IndexError(path_utf8(path) + ": " + e.what());
  }
}

// ---- Reading a component's safetensors files --------------------------------------------------

// Opens every safetensors file of a component dir (following the shard index when present),
// registers them in index.files and returns all tensors with their file ids.
std::vector<IndexedTensor> load_component_tensors(ModelIndex& index, Component comp) {
  const fs::path dir = index.bundle_dir / component_name(comp);
  if (!util::file_info(dir).directory()) throw IndexError("bundle is missing directory " + path_utf8(dir));

  std::vector<std::string> shards, shard_indexes;
  std::vector<util::DirEntry> entries;  // names only: each shard is opened as a regular file when it is read
  if (!util::list_directory(dir, entries)) throw IndexError("cannot list " + path_utf8(dir));
  for (const util::DirEntry& e : entries) {
    std::string n = path_utf8(e.name);
    if (ends_with(n, kShardIndexSuffix)) shard_indexes.push_back(n);
    else if (ends_with(n, kShardSuffix)) shards.push_back(n);
  }
  std::sort(shards.begin(), shards.end());
  if (shard_indexes.size() > 1) throw IndexError(path_utf8(dir) + ": more than one *.safetensors.index.json");

  std::unordered_map<std::string, std::string> weight_map;  // tensor -> shard file name
  if (shard_indexes.size() == 1) {
    const fs::path index_path = dir / path_from_utf8(shard_indexes[0]);
    json::Value doc = parse_json_file(index_path, kMaxShardIndexBytes, kMaxShardIndexValues);
    const json::Value* wm = doc.find("weight_map");
    if (wm == nullptr || !wm->is_object()) throw IndexError(path_utf8(index_path) + ": missing weight_map");
    std::set<std::string> referenced;
    for (const auto& [tensor, shard] : wm->as_object()) {
      if (!shard.is_string()) throw IndexError(path_utf8(index_path) + ": weight_map entry '" + tensor + "' is not a file name");
      if (!weight_map.emplace(tensor, shard.as_string()).second) {
        throw IndexError(path_utf8(index_path) + ": duplicate weight_map entry '" + tensor + "'");
      }
      referenced.insert(shard.as_string());
    }
    // A shard on disk that the index never mentions would silently drop weights.
    if (std::set<std::string>(shards.begin(), shards.end()) != referenced) {
      throw IndexError(path_utf8(dir) + ": *.safetensors files on disk differ from those listed in " + shard_indexes[0]);
    }
  } else if (shards.size() != 1) {
    throw IndexError(path_utf8(dir) + ": expected one .safetensors file or a shard index, found " + std::to_string(shards.size()) + " files and no index");
  }

  std::vector<IndexedTensor> out;
  std::size_t mapped = 0;
  for (const std::string& shard : shards) {
    const fs::path path = dir / path_from_utf8(shard);
    SafetensorsFile file(path);
    auto id = static_cast<std::uint32_t>(index.files.size());
    index.files.push_back({std::string(component_name(comp)) + "/" + shard, file.file_size(), util::file_info(path).mtime_ns});
    for (const TensorInfo& t : file.tensors()) {
      if (!weight_map.empty()) {
        auto it = weight_map.find(t.name);
        if (it == weight_map.end()) throw FormatError(path, t.name, "tensor is not listed in the shard index");
        if (it->second != shard) throw FormatError(path, t.name, "shard index places this tensor in " + it->second);
        ++mapped;
      }
      out.push_back({t.name, t.dtype, t.shape, id, t.offset, t.nbytes});
    }
  }
  if (!weight_map.empty() && mapped != weight_map.size()) {
    throw IndexError(path_utf8(dir) + ": shard index lists " + std::to_string(weight_map.size() - mapped) + " tensors missing from the shards");
  }
  return out;
}

// ---- Classification ---------------------------------------------------------------------------

using Route = TensorRoute;

[[noreturn]] void unclassified(Component c, const std::string& name) {
  throw IndexError(std::string("unclassified ") + component_name(c) + " tensor '" + name + "'");
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

// Qwen3_5Model checkpoints name tensors `language_model.*` / `visual.*`; wrapped variants carry a
// leading `model.`.
Route classify_text_encoder(const std::string& full) {
  std::string_view name = full;
  if (starts_with(name, "model.")) name.remove_prefix(6);
  Route r;
  if (starts_with(name, "visual.")) { r.exclude = "visual."; return r; }
  if (starts_with(name, "lm_head.")) { r.exclude = "lm_head."; return r; }
  if (starts_with(name, "mtp.")) { r.exclude = "mtp."; return r; }
  if (!starts_with(name, "language_model.")) unclassified(Component::kTextEncoder, full);
  name.remove_prefix(15);
  if (starts_with(name, "embed_tokens.")) { r.group = "embed_tokens"; return r; }
  if (starts_with(name, "norm.")) { r.group = "final_norm"; return r; }
  if (starts_with(name, "layers.")) {
    name.remove_prefix(7);
    int n = 0;
    if (!take_layer_index(name, n)) unclassified(Component::kTextEncoder, full);
    r.group = "layers." + std::to_string(n);
    if (starts_with(name, "linear_attn.")) r.layer_kind = "linear_attention";
    else if (starts_with(name, "self_attn.")) r.layer_kind = "full_attention";
    return r;
  }
  unclassified(Component::kTextEncoder, full);
}

Route classify_transformer(const std::string& full) {
  std::string_view name = full;
  Route r;
  if (starts_with(name, "layers.")) {
    name.remove_prefix(7);
    int n = 0;
    if (!take_layer_index(name, n)) unclassified(Component::kTransformer, full);
    r.group = "block." + std::to_string(n);
    return r;
  }
  if (starts_with(name, "text_embedder.")) { r.group = "text_embedder"; return r; }
  // `rope.bands` is a derived buffer (the reference always rebuilds it) but is stored in the file.
  for (std::string_view p : {"video_embedder.", "time_embedding.", "shared_block_modulation.", "video_head_modulation.",
                              "final_norm_video.", "final_linear_video.", "rope."}) {
    if (starts_with(name, p)) { r.group = "globals"; return r; }
  }
  unclassified(Component::kTransformer, full);
}

// Only the decoder path is needed for generation.
Route classify_vae(const std::string& full) {
  Route r;
  for (std::string_view p : {"decoder.", "post_quant_conv.", "bn."}) {
    if (starts_with(full, p)) { r.group = "decoder"; return r; }
  }
  for (std::string_view p : {"encoder.", "quant_conv."}) {
    if (starts_with(full, p)) { r.exclude = std::string(p); return r; }
  }
  unclassified(Component::kVae, full);
}

// Builds the declared groups (in `group_names` order) from classified tensors.
void assemble(ModelIndex& index, Component comp, std::vector<IndexedTensor> tensors,
              const std::vector<std::string>& group_names, Route (*classify)(const std::string&)) {
  const std::size_t first = index.groups.size();
  std::unordered_map<std::string, std::size_t> slot;
  for (const auto& n : group_names) {
    slot[n] = index.groups.size();
    WeightGroup g;
    g.component = comp;
    g.name = n;
    index.groups.push_back(std::move(g));
  }
  std::map<std::string, ExcludedSummary> excluded;
  for (auto& t : tensors) {
    Route r = classify(t.name);
    if (!r.exclude.empty()) {
      ExcludedSummary& e = excluded[r.exclude];
      e.component = comp;
      e.prefix = r.exclude;
      ++e.tensors;
      e.bytes += t.nbytes;
      continue;
    }
    auto it = slot.find(r.group);
    if (it == slot.end()) {
      throw IndexError(std::string(component_name(comp)) + " tensor '" + t.name + "' belongs to group '" + r.group + "', which the model config does not define");
    }
    WeightGroup& g = index.groups[it->second];
    if (!r.layer_kind.empty()) {
      if (!g.layer_type.empty() && g.layer_type != r.layer_kind) {
        throw IndexError("text_encoder " + g.name + " mixes linear_attn and self_attn tensors ('" + t.name + "')");
      }
      g.layer_type = r.layer_kind;
    }
    g.tensors.push_back(std::move(t));
  }
  for (std::size_t i = first; i < index.groups.size(); ++i) {
    WeightGroup& g = index.groups[i];
    if (g.tensors.empty()) throw IndexError(std::string(component_name(comp)) + " group '" + g.name + "' has no tensors in the checkpoint");
    std::sort(g.tensors.begin(), g.tensors.end(), [](const IndexedTensor& a, const IndexedTensor& b) {
      return a.file != b.file ? a.file < b.file : a.offset < b.offset;
    });
    for (const auto& t : g.tensors) {
      g.total_bytes += t.nbytes;
      g.total_params += t.numel();
    }
  }
  for (auto& [prefix, e] : excluded) index.excluded.push_back(std::move(e));
}

std::vector<std::string> numbered(const std::string& prefix, std::int64_t n) {
  std::vector<std::string> v;
  for (std::int64_t i = 0; i < n; ++i) v.push_back(prefix + std::to_string(i));
  return v;
}

// `weights` false: the config is read and checked, no weight file is opened (BuildOptions::skip).
void index_text_encoder(ModelIndex& index, bool weights) {
  const fs::path cfg_path = index.bundle_dir / "text_encoder" / "config.json";
  json::Value cfg = parse_json_file(cfg_path, kMaxConfigBytes, kMaxConfigValues);
  const json::Value* tc = cfg.find("text_config");
  const json::Value* lt = tc ? tc->find("layer_types") : nullptr;
  if (lt == nullptr || !lt->is_array()) throw IndexError(path_utf8(cfg_path) + ": missing text_config.layer_types");
  if (static_cast<std::int64_t>(lt->as_array().size()) > kMaxLayers) {
    throw IndexError(path_utf8(cfg_path) + ": more than " + std::to_string(kMaxLayers) + " layer_types");
  }
  std::vector<std::string> types;
  for (const auto& v : lt->as_array()) {
    if (!v.is_string()) throw IndexError(path_utf8(cfg_path) + ": text_config.layer_types holds a non-string");
    types.push_back(v.as_string());
    if (types.back() != "full_attention" && types.back() != "linear_attention") {
      throw IndexError(path_utf8(cfg_path) + ": unknown layer type '" + types.back() + "'");
    }
  }
  std::vector<std::string> names{"embed_tokens"};
  for (auto& n : numbered("layers.", static_cast<std::int64_t>(types.size()))) names.push_back(std::move(n));
  names.push_back("final_norm");
  if (!weights) return;

  const std::size_t first = index.groups.size();
  assemble(index, Component::kTextEncoder, load_component_tensors(index, Component::kTextEncoder), names, classify_text_encoder);
  for (std::size_t i = 0; i < types.size(); ++i) {
    WeightGroup& g = index.groups[first + 1 + i];
    if (g.layer_type.empty()) throw IndexError("text_encoder " + g.name + " has neither linear_attn nor self_attn tensors");
    if (g.layer_type != types[i]) {
      throw IndexError("text_encoder " + g.name + ": tensors are " + g.layer_type + " but config layer_types says " + types[i]);
    }
  }
}

void index_transformer(ModelIndex& index, bool weights) {
  const fs::path cfg_path = index.bundle_dir / "transformer" / "config.json";
  json::Value cfg = parse_json_file(cfg_path, kMaxConfigBytes, kMaxConfigValues);
  const json::Value* nl = cfg.find("num_layers");
  if (nl == nullptr || !nl->is_int()) throw IndexError(path_utf8(cfg_path) + ": missing or non-integer num_layers");
  if (nl->as_int() < 0 || nl->as_int() > kMaxLayers) {
    throw IndexError(path_utf8(cfg_path) + ": num_layers " + std::to_string(nl->as_int()) + " is outside [0, " + std::to_string(kMaxLayers) + "]");
  }
  if (!weights) return;
  std::vector<std::string> names{"globals", "text_embedder"};
  for (auto& n : numbered("block.", nl->as_int())) names.push_back(std::move(n));
  assemble(index, Component::kTransformer, load_component_tensors(index, Component::kTransformer), names, classify_transformer);
}

void index_vae(ModelIndex& index) {
  assemble(index, Component::kVae, load_component_tensors(index, Component::kVae), {"decoder"}, classify_vae);
}

// ---- JSON output ------------------------------------------------------------------------------

std::string quote(std::string_view s) {
  std::string out = "\"";
  for (char ch : s) {
    auto c = static_cast<unsigned char>(ch);
    if (ch == '"' || ch == '\\') { out += '\\'; out += ch; }
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
    else out += ch;
  }
  return out + "\"";
}

std::string hex64(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

Component parse_component(const std::string& s) {
  for (Component c : {Component::kTextEncoder, Component::kTransformer, Component::kVae}) {
    if (s == component_name(c)) return c;
  }
  throw IndexError("unknown component '" + s + "' in index");
}

std::uint64_t as_u64(const json::Value& v) {
  std::int64_t i = v.as_int();
  if (i < 0) throw IndexError("negative value in index");
  return static_cast<std::uint64_t>(i);
}

// A stored file path must name something inside the bundle on every platform: '/'-separated, no
// root, no empty, "." or ".." components, and no '\\' or ':' (separator / drive on Windows).
bool is_plain_relative_path(std::string_view p) {
  if (p.empty()) return false;
  for (std::size_t start = 0;;) {
    const std::size_t end = p.find('/', start);
    const std::string_view part = p.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
    if (part.empty() || part == "." || part == "..") return false;
    for (char c : part) {
      if (c == '\\' || c == ':' || static_cast<unsigned char>(c) < 0x20) return false;
    }
    if (end == std::string_view::npos) break;
    start = end + 1;
  }
  const fs::path rel = path_from_utf8(p);
  return !rel.is_absolute() && !rel.has_root_path();
}

// First difference between a loaded index and one built fresh from the bundle, or "" if they agree
// (checksums aside, which only the loaded one has).
std::string first_difference(const ModelIndex& a, const ModelIndex& b) {
  if (a.files.size() != b.files.size()) return "number of files";
  for (std::size_t i = 0; i < a.files.size(); ++i) {
    const IndexedFile &x = a.files[i], &y = b.files[i];
    if (x.path != y.path || x.size != y.size || x.mtime_ns != y.mtime_ns) return "file " + std::to_string(i) + " ('" + x.path + "')";
  }
  if (a.excluded.size() != b.excluded.size()) return "excluded tensor summary";
  for (std::size_t i = 0; i < a.excluded.size(); ++i) {
    const ExcludedSummary &x = a.excluded[i], &y = b.excluded[i];
    if (x.component != y.component || x.prefix != y.prefix || x.tensors != y.tensors || x.bytes != y.bytes) {
      return "excluded prefix '" + x.prefix + "'";
    }
  }
  if (a.groups.size() != b.groups.size()) {
    return "group count (" + std::to_string(a.groups.size()) + " stored, " + std::to_string(b.groups.size()) + " in the bundle)";
  }
  for (std::size_t i = 0; i < a.groups.size(); ++i) {
    const WeightGroup &x = a.groups[i], &y = b.groups[i];
    if (x.component != y.component || x.name != y.name || x.layer_type != y.layer_type || x.tensors.size() != y.tensors.size()) {
      return std::string("group ") + component_name(x.component) + "/" + x.name;
    }
    for (std::size_t t = 0; t < x.tensors.size(); ++t) {
      const IndexedTensor &u = x.tensors[t], &v = y.tensors[t];
      if (u.name != v.name || u.dtype != v.dtype || u.shape != v.shape || u.file != v.file || u.offset != v.offset || u.nbytes != v.nbytes) {
        return "tensor '" + u.name + "' of group " + x.name;
      }
    }
  }
  return {};
}

}  // namespace

// ---- Public API -------------------------------------------------------------------------------

const char* component_name(Component c) {
  switch (c) {
    case Component::kTextEncoder: return "text_encoder";
    case Component::kTransformer: return "transformer";
    case Component::kVae: return "vae";
  }
  return "?";
}

bool take_layer_index(std::string_view& rest, int& n) {
  auto dot = rest.find('.');
  if (dot == std::string_view::npos || dot == 0) return false;
  if (rest[0] < '0' || rest[0] > '9' || (rest[0] == '0' && dot > 1)) return false;
  auto r = std::from_chars(rest.data(), rest.data() + dot, n);
  if (r.ec != std::errc() || r.ptr != rest.data() + dot) return false;
  rest.remove_prefix(dot + 1);
  return true;
}

TensorRoute classify_tensor(Component c, const std::string& name) {
  switch (c) {
    case Component::kTextEncoder: return classify_text_encoder(name);
    case Component::kTransformer: return classify_transformer(name);
    case Component::kVae: return classify_vae(name);
  }
  unclassified(c, name);
}

std::int64_t IndexedTensor::numel() const {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

const WeightGroup* ModelIndex::find_group(Component c, std::string_view name) const {
  for (const auto& g : groups) {
    if (g.component == c && g.name == name) return &g;
  }
  return nullptr;
}

std::vector<const WeightGroup*> ModelIndex::groups_of(Component c) const {
  std::vector<const WeightGroup*> v;
  for (const auto& g : groups) {
    if (g.component == c) v.push_back(&g);
  }
  return v;
}

std::uint64_t checksum_group(const ModelIndex& index, const WeightGroup& group) {
  Xxh64 hash;
  std::map<std::uint32_t, FileReader> readers;
  std::vector<std::byte> buf(kChecksumChunk);
  for (const auto& t : group.tensors) {
    auto it = readers.find(t.file);
    if (it == readers.end()) it = readers.emplace(t.file, FileReader(index.file_path(t.file))).first;
    for (std::uint64_t done = 0; done < t.nbytes;) {
      auto n = static_cast<std::size_t>(std::min<std::uint64_t>(buf.size(), t.nbytes - done));
      it->second.read_at(t.offset + done, buf.data(), n);
      hash.update(buf.data(), n);
      done += n;
    }
  }
  return hash.digest();
}

ModelIndex build_index(const fs::path& bundle_dir, const BuildOptions& options) {
  ModelIndex index;
  index.bundle_dir = bundle_dir;
  try {
    index_text_encoder(index, !options.skip.has(Component::kTextEncoder));
    index_transformer(index, !options.skip.has(Component::kTransformer));
    if (!options.skip.has(Component::kVae)) index_vae(index);
    if (options.checksum) {
      for (auto& g : index.groups) g.checksum = checksum_group(index, g);
    }
  } catch (const fs::filesystem_error& e) {
    throw IndexError(path_utf8(bundle_dir) + ": cannot read bundle (" + e.what() + ")");
  }
  return index;
}

void write_index(const ModelIndex& index, const fs::path& path) {
  std::ostringstream o;
  o << "{\n\"format\": " << quote(kFormatName) << ",\n\"version\": " << kIndexVersion << ",\n\"files\": [\n";
  for (std::size_t i = 0; i < index.files.size(); ++i) {
    const auto& f = index.files[i];
    o << " {\"path\": " << quote(f.path) << ", \"size\": " << f.size << ", \"mtime_ns\": " << f.mtime_ns << "}"
      << (i + 1 < index.files.size() ? ",\n" : "\n");
  }
  o << "],\n\"excluded\": [\n";
  for (std::size_t i = 0; i < index.excluded.size(); ++i) {
    const auto& e = index.excluded[i];
    o << " {\"component\": " << quote(component_name(e.component)) << ", \"prefix\": " << quote(e.prefix)
      << ", \"tensors\": " << e.tensors << ", \"bytes\": " << e.bytes << "}" << (i + 1 < index.excluded.size() ? ",\n" : "\n");
  }
  o << "],\n\"groups\": [\n";
  for (std::size_t gi = 0; gi < index.groups.size(); ++gi) {
    const auto& g = index.groups[gi];
    o << " {\"component\": " << quote(component_name(g.component)) << ", \"name\": " << quote(g.name);
    if (!g.layer_type.empty()) o << ", \"layer_type\": " << quote(g.layer_type);
    o << ", \"bytes\": " << g.total_bytes << ", \"params\": " << g.total_params;
    if (g.checksum) o << ", \"xxh64\": " << quote(hex64(*g.checksum));
    o << ",\n  \"tensors\": [\n";
    for (std::size_t ti = 0; ti < g.tensors.size(); ++ti) {
      const auto& t = g.tensors[ti];
      o << "   [" << quote(t.name) << ", " << quote(dtype_name(t.dtype)) << ", [";
      for (std::size_t d = 0; d < t.shape.size(); ++d) o << (d ? ", " : "") << t.shape[d];
      o << "], " << t.file << ", " << t.offset << ", " << t.nbytes << "]" << (ti + 1 < g.tensors.size() ? ",\n" : "\n");
    }
    o << "  ]}" << (gi + 1 < index.groups.size() ? ",\n" : "\n");
  }
  o << "]\n}\n";

  // Write-then-rename so a crash never leaves a half-written index that parses as stale.
  util::OutputFile out(path, util::OutputFile::Create::kTemporary);
  if (!out.write(o.str()) || !out.publish(path)) {
    out.discard();
    throw IndexError("cannot write " + path_utf8(path));
  }
}

ModelIndex load_index(const fs::path& path, const fs::path& bundle_dir, ComponentSet skip) {
  ModelIndex index;
  index.bundle_dir = bundle_dir.empty() ? path.parent_path() : bundle_dir;
  if (index.bundle_dir.empty()) index.bundle_dir = ".";
  const std::string where = path_utf8(path);
  json::Value doc = parse_json_file(path, kMaxMossIdxBytes, kMaxMossIdxValues);
  auto fail = [&](const std::string& what) -> void { throw IndexError(where + ": " + what); };
  try {
    const json::Value* fmt = doc.find("format");
    if (fmt == nullptr || !fmt->is_string() || fmt->as_string() != kFormatName) fail("not a .mossidx file");
    const json::Value* ver = doc.find("version");
    if (ver == nullptr || !ver->is_number()) fail("missing version");
    if (ver->as_int() != kIndexVersion) {
      fail("index version " + std::to_string(ver->as_int()) + " is not supported (expected " + std::to_string(kIndexVersion) + "); rebuild with `moss-infer index`");
    }

    // The index ships inside a downloaded bundle, so nothing in it is trusted: paths are confined
    // to the bundle before anything is opened, and every size is range-checked before use.
    // A skipped component's files are dropped unopened (build_index names them "<component>/<file>"); the kept files'
    // ids are renumbered (stored id -> kept id, -1: dropped).
    std::vector<std::int64_t> kept_id;
    const auto skipped_file = [&](std::string_view p) {
      const std::string_view folder = p.substr(0, p.find('/'));
      for (Component c : {Component::kTextEncoder, Component::kTransformer, Component::kVae}) {
        if (skip.has(c) && folder == component_name(c)) return true;
      }
      return false;
    };
    for (const auto& f : doc.at("files").as_array()) {
      IndexedFile file{f.at("path").as_string(), as_u64(f.at("size")), f.at("mtime_ns").as_int()};
      if (!is_plain_relative_path(file.path)) fail("file path '" + file.path + "' is not a plain bundle-relative path");
      if (skipped_file(file.path)) {
        kept_id.push_back(-1);
        continue;
      }
      kept_id.push_back(static_cast<std::int64_t>(index.files.size()));
      const fs::path p = index.bundle_dir / path_from_utf8(file.path);
      const util::FileInfo info = util::file_info(p);
      // Gone: its component's weights are missing (rebuilding cannot help); a skipped component's are never looked at.
      if (!info.regular()) {
        const std::string folder = file.path.substr(0, file.path.find('/'));
        fail("missing weights for " + folder + " (" + file.path + "): add them or pass its component file");
      }
      if (info.size != file.size || info.mtime_ns != file.mtime_ns) {
        throw StaleIndexError(where + ": " + file.path + " changed since the index was built; rebuild with `moss-infer index`");
      }
      index.files.push_back(std::move(file));
    }
    for (const auto& e : doc.at("excluded").as_array()) {
      const Component c = parse_component(e.at("component").as_string());
      if (skip.has(c)) continue;
      index.excluded.push_back({c, e.at("prefix").as_string(), as_u64(e.at("tensors")), as_u64(e.at("bytes"))});
    }
    for (const auto& gv : doc.at("groups").as_array()) {
      WeightGroup g;
      g.component = parse_component(gv.at("component").as_string());
      if (skip.has(g.component)) continue;
      g.name = gv.at("name").as_string();
      if (const json::Value* lt = gv.find("layer_type")) g.layer_type = lt->as_string();
      if (const json::Value* cs = gv.find("xxh64")) {
        const std::string& h = cs->as_string();
        std::uint64_t v = 0;
        auto r = std::from_chars(h.data(), h.data() + h.size(), v, 16);
        if (r.ec != std::errc() || r.ptr != h.data() + h.size()) fail("bad xxh64 in group " + g.name);
        g.checksum = v;
      }
      // Unsigned sums wrap instead of overflowing; any wrapped total fails the comparison below.
      std::uint64_t bytes = 0, params = 0;
      for (const auto& tv : gv.at("tensors").as_array()) {
        const json::Array& a = tv.as_array();
        if (a.size() != 6) fail("malformed tensor entry in group " + g.name);
        IndexedTensor t;
        t.name = a[0].as_string();
        if (!parse_dtype(a[1].as_string(), t.dtype)) fail("tensor '" + t.name + "': unknown dtype " + a[1].as_string());
        const json::Array& dims = a[2].as_array();
        if (dims.size() > kMaxRank) fail("tensor '" + t.name + "': rank " + std::to_string(dims.size()) + " exceeds " + std::to_string(kMaxRank));
        for (const auto& d : dims) t.shape.push_back(d.as_int());
        const std::optional<std::int64_t> numel = util::checked_numel(t.shape);
        if (!numel) fail("tensor '" + t.name + "': negative dimension or element count overflow");
        const std::uint64_t file_id = as_u64(a[3]);
        t.offset = as_u64(a[4]);
        t.nbytes = as_u64(a[5]);
        if (file_id >= kept_id.size()) fail("tensor '" + t.name + "': bad file id");
        if (kept_id[file_id] < 0) fail("tensor '" + t.name + "': its file belongs to a component left out");
        t.file = static_cast<std::uint32_t>(kept_id[file_id]);
        if (t.offset > index.files[t.file].size || t.nbytes > index.files[t.file].size - t.offset) fail("tensor '" + t.name + "': range outside its file");
        const std::optional<std::uint64_t> expect = util::checked_mul(static_cast<std::uint64_t>(*numel), dtype_size(t.dtype));
        if (!expect || t.nbytes != *expect) fail("tensor '" + t.name + "': byte size does not match shape x dtype");
        bytes += t.nbytes;
        params += static_cast<std::uint64_t>(*numel);
        g.tensors.push_back(std::move(t));
      }
      if (bytes != as_u64(gv.at("bytes")) || params != as_u64(gv.at("params"))) {
        fail("group " + g.name + ": stored totals do not match its tensors");
      }
      g.total_bytes = bytes;
      g.total_params = static_cast<std::int64_t>(params);
      index.groups.push_back(std::move(g));
    }

    // No two tensors share bytes (zero-length tensors may sit at the same offset).
    std::vector<const IndexedTensor*> all;
    for (const auto& g : index.groups) {
      for (const auto& t : g.tensors) all.push_back(&t);
    }
    std::sort(all.begin(), all.end(), [](const IndexedTensor* a, const IndexedTensor* b) {
      return std::tie(a->file, a->offset, a->nbytes) < std::tie(b->file, b->offset, b->nbytes);
    });
    std::uint64_t end = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
      if (i > 0 && all[i]->file != all[i - 1]->file) end = 0;
      if (all[i]->offset < end) fail("tensor '" + all[i]->name + "' overlaps another tensor in " + index.files[all[i]->file].path);
      end = std::max(end, all[i]->offset + all[i]->nbytes);
    }

    // Finally the whole index must equal what build_index derives from the bundle now: the same
    // configs (group set, layer count and types), safetensors headers (names, dtypes, shapes,
    // offsets) and classification. Headers are a few tens of KB per file, so this is cheap.
    ModelIndex fresh;
    try {
      fresh = build_index(index.bundle_dir, {.skip = skip});
    } catch (const std::exception& e) {
      fail(std::string("bundle does not validate (") + e.what() + ")");
    }
    const std::string diff = first_difference(index, fresh);
    if (!diff.empty()) throw StaleIndexError(where + ": index does not match the bundle: " + diff + " differs; rebuild with `moss-infer index`");
  } catch (const IndexError&) {
    throw;
  } catch (const std::exception& e) {
    throw IndexError(where + ": malformed index (" + e.what() + ")");
  }
  return index;
}

// ---- Summary ----------------------------------------------------------------------------------

namespace {

struct Totals {
  std::uint64_t tensors = 0, bytes = 0;
  std::int64_t params = 0;
};

std::string with_commas(std::uint64_t v) {
  std::string s = std::to_string(v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<std::size_t>(i), ",");
  return s;
}

std::string gib(std::uint64_t bytes) {
  char b[32];
  std::snprintf(b, sizeof b, "%.3f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  return b;
}

}  // namespace

void print_summary(const ModelIndex& index, std::ostream& out) {
  out << "bundle: " << path_utf8(index.bundle_dir) << "  (" << index.files.size() << " files)\n";
  for (Component c : {Component::kTextEncoder, Component::kTransformer, Component::kVae}) {
    auto groups = index.groups_of(c);
    Totals all;
    std::map<DType, Totals> by_dtype;
    for (const auto* g : groups) {
      for (const auto& t : g->tensors) {
        Totals& d = by_dtype[t.dtype];
        ++d.tensors; d.bytes += t.nbytes; d.params += t.numel();
        ++all.tensors; all.bytes += t.nbytes; all.params += t.numel();
      }
    }
    out << "\n[" << component_name(c) << "] " << groups.size() << " groups, " << all.tensors << " tensors, "
        << with_commas(static_cast<std::uint64_t>(all.params)) << " params, " << with_commas(all.bytes) << " bytes (" << gib(all.bytes) << ")\n";
    for (const auto& [dt, t] : by_dtype) {
      out << "  " << std::left << std::setw(8) << dtype_name(dt) << std::right << std::setw(6) << t.tensors << " tensors "
          << std::setw(18) << with_commas(static_cast<std::uint64_t>(t.params)) << " params " << std::setw(18) << with_commas(t.bytes) << " bytes\n";
    }
    if (c == Component::kTransformer) {
      std::set<std::uint64_t> sizes;
      std::uint64_t n = 0;
      std::int64_t params = 0;
      for (const auto* g : groups) {
        if (g->name.rfind("block.", 0) != 0) continue;
        ++n; sizes.insert(g->total_bytes); params = g->total_params;
      }
      out << "  DiT blocks: " << n;
      if (!sizes.empty()) {
        out << ", " << (sizes.size() == 1 ? "all identical" : "SIZES DIFFER") << ", " << with_commas(*sizes.begin()) << " bytes ("
            << gib(*sizes.begin()) << ") / " << with_commas(static_cast<std::uint64_t>(params)) << " params each";
      }
      out << "\n";
      for (const char* n2 : {"globals", "text_embedder"}) {
        if (const auto* g = index.find_group(c, n2)) {
          out << "  " << n2 << ": " << g->tensors.size() << " tensors, " << with_commas(static_cast<std::uint64_t>(g->total_params)) << " params, " << with_commas(g->total_bytes) << " bytes\n";
        }
      }
    }
    if (c == Component::kTextEncoder) {
      std::map<std::string, std::vector<const WeightGroup*>> by_type;
      for (const auto* g : groups) {
        if (!g->layer_type.empty()) by_type[g->layer_type].push_back(g);
      }
      for (const auto& [type, gs] : by_type) {
        std::uint64_t lo = gs[0]->total_bytes, hi = lo;
        for (const auto* g : gs) { lo = std::min(lo, g->total_bytes); hi = std::max(hi, g->total_bytes); }
        out << "  " << type << " layers: " << gs.size() << ", " << with_commas(lo) << (lo == hi ? "" : ".." + with_commas(hi)) << " bytes each, "
            << with_commas(static_cast<std::uint64_t>(gs[0]->total_params)) << " params each\n";
      }
      for (const char* n2 : {"embed_tokens", "final_norm"}) {
        if (const auto* g = index.find_group(c, n2)) {
          out << "  " << n2 << ": " << with_commas(static_cast<std::uint64_t>(g->total_params)) << " params, " << with_commas(g->total_bytes) << " bytes\n";
        }
      }
    }
    for (const auto& e : index.excluded) {
      if (e.component == c) out << "  excluded " << e.prefix << "*: " << e.tensors << " tensors, " << with_commas(e.bytes) << " bytes\n";
    }
    for (const auto* g : groups) {
      if (g->checksum) { out << "  xxh64 " << g->name << " = " << hex64(*g->checksum) << "\n"; }
    }
  }
}

}  // namespace mi::weights
