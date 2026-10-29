#include "pipeline/gguf_dit.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "pipeline/session.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/gguf.h"
#include "weights/quant/quant_convert.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"

namespace mi::pipeline {

namespace fs = std::filesystem;

namespace {

// Bumped whenever the converted file of the same GGUF bytes would change (the reader, the dequantization, the layout,
// the metadata); the converter's own version (quant::kConverterVersion) is part of the key too.
constexpr int kGgufImportVersion = 2;
// Free space kept beyond the converted file's size.
constexpr std::uint64_t kSpaceMargin = std::uint64_t{256} << 20;
// How often a lock held by another process is tried again (and the cancel flag checked meanwhile).
constexpr auto kLockPoll = std::chrono::milliseconds(50);
constexpr std::string_view kExtension = ".safetensors";

std::string hex64(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

// The GGUF file's identity: its util::FileInfo (size, modification time, volume, 128-bit file id) and the XXH64 of its
// header (metadata and tensor infos).
struct Identity {
  util::FileInfo info;
  std::uint64_t header_xxh64 = 0;
  bool operator==(const Identity&) const = default;
};
Identity identity_of(const fs::path& file) {
  const util::FileInfo info = util::file_info(file);  // before the header is read
  const weights::GgufFile g(file);
  std::vector<std::byte> header(static_cast<std::size_t>(g.header_bytes()));
  weights::FileReader(file).read_at(0, header.data(), header.size());
  return {info, xxh64(header.data(), header.size())};
}

// The file stem for a cache name: "gguf-" (never a DOS device name such as aux, con or com1 on Windows), then at most
// 48 characters of the GGUF's stem in [A-Za-z0-9._-], others replaced by '_'.
std::string safe_stem(const fs::path& file) {
  std::string s = util::path_utf8(file.stem());
  if (s.size() > 48) s.resize(48);
  for (char& c : s) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    if (!ok) c = '_';
  }
  return "gguf-" + (s.empty() ? std::string("dit") : s);
}

// The value of a cache file's metadata key, or "" when the file is not a readable safetensors file or has none.
std::string cache_metadata(const fs::path& p, const std::string& key) {
  try {
    const weights::SafetensorsFile st(p);
    const auto it = st.metadata().find(key);
    return it == st.metadata().end() ? std::string() : it->second;
  } catch (const std::exception&) {
    return {};
  }
}

bool cancelled(const GgufConvertHooks& hooks) { return hooks.cancel != nullptr && hooks.cancel->load(); }

[[noreturn]] void throw_cancelled() { throw SessionCancelled("the call was cancelled"); }

// `mode` on the lock file, waiting while another holder's conflicts: IoError when it cannot be used, SessionCancelled
// when the call is cancelled meanwhile.
util::FileLock take_lock(const fs::path& path, util::FileLock::Mode mode, const GgufConvertHooks& hooks) {
  util::FileLock lock;
  for (int polls = 0;; ++polls) {
    const util::FileLock::Status s = lock.try_lock(path, mode);
    if (s == util::FileLock::Status::kLocked) return lock;
    if (s == util::FileLock::Status::kError) {
      throw weights::IoError(util::path_utf8(path) + ": cannot create or lock the GGUF cache entry's lock file");
    }
    if (cancelled(hooks)) throw_cancelled();
    if (polls % 20 == 0 && hooks.progress && !hooks.progress(0.0)) throw_cancelled();
    std::this_thread::sleep_for(kLockPoll);
  }
}

// The cache folder's stale entries, each removed under the exclusive lock of the converted file it belongs to (a file
// in use, whose lock a load holds shared, and one being written are never touched): converted files whose recorded
// source no longer exists, or that an earlier version of `source` gave for `target_name` (another key); temporaries of
// conversions that ended without publishing; lock files of converted files that are gone. Files without the
// importer's metadata are left alone. `keep` is the call's own file. Best effort: what cannot be removed stays.
void sweep(const fs::path& dir, const fs::path& resolved_dir, const std::string& keep, const std::string& source,
           const std::string& target_name) {
  std::vector<util::DirEntry> entries;
  if (!util::list_directory(dir, entries)) return;
  std::vector<std::string> names;
  for (const util::DirEntry& e : entries) names.push_back(util::path_utf8(e.name));
  const auto listed = [&](const std::string& n) { return std::find(names.begin(), names.end(), n) != names.end(); };
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const std::string& name = names[i];
    if (!entries[i].info.regular() || entries[i].link) continue;
    std::string owner;  // the converted file this entry belongs to
    bool stale = false;
    if (const auto t = name.find(std::string(kExtension) + ".tmp-"); t != std::string::npos) {
      owner = name.substr(0, t + kExtension.size());
      stale = true;
    } else if (name.ends_with(std::string(kExtension) + ".lock")) {
      owner = name.substr(0, name.size() - 5);
      stale = !listed(owner);
    } else if (name.ends_with(kExtension)) {
      owner = name;
      const fs::path p = dir / util::path_from_utf8(name);
      const std::string src = cache_metadata(p, "mossinfer_gguf_source");
      if (src.empty()) continue;
      stale = !util::file_info(util::path_from_utf8(src)).exists() ||
              (src == source && cache_metadata(p, "mossinfer_gguf_target") == target_name);
    }
    if (!stale || owner == keep) continue;
    util::FileLock lock;
    const fs::path lock_path = dir / util::path_from_utf8(owner + ".lock");
    if (lock.try_lock(lock_path, util::FileLock::Mode::kExclusive) != util::FileLock::Status::kLocked) continue;
    if (name != owner + ".lock") util::remove_file_in(resolved_dir, dir / util::path_from_utf8(name));
    if (!util::file_info(dir / util::path_from_utf8(owner)).exists()) lock.unlock_and_remove(resolved_dir);
  }
}

}  // namespace

GgufDitInfo inspect_gguf_dit(const fs::path& file) {
  const weights::GgufFile g(file);
  weights::check_supported_gguf_dit(g);
  quant::open_gguf_dit(g);  // every tensor a MOSS-Image DiT tensor of a type the converter reads
  // The types of every stored tensor (the block GEMMs, the other linears, the norms), by their bytes.
  std::map<std::uint32_t, std::uint64_t> bytes;
  for (const weights::GgufTensorInfo& t : g.tensors()) bytes[t.type] += t.nbytes;
  std::vector<std::pair<std::uint64_t, std::uint32_t>> order;
  for (const auto& [type, n] : bytes) order.emplace_back(n, type);
  std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
  GgufDitInfo info;
  info.reference = true;
  for (const auto& [n, type] : order) {
    info.types += (info.types.empty() ? "" : " + ") + std::string(weights::ggml_type_name(type));
    using namespace weights::ggml;
    if (type != kBF16 && type != kF16 && type != kF32) info.reference = false;
  }
  info.tensors = static_cast<int>(g.tensors().size());
  return info;
}

quant::QuantSpec gguf_target(const GgufDitInfo& info, planner::Tier tier) {
  const std::string what = "the GGUF DiT is " + info.types + ", which runs as INT8 ConvRot (tier high)";
  switch (tier) {
    case planner::Tier::kHigh: return quant::QuantSpec::int8_convrot(256);
    case planner::Tier::kReference:
      if (info.reference) return quant::QuantSpec::bf16();
      throw std::invalid_argument("tier reference runs bf16 weights, but " + what +
                                  "; tier reference reads GGUF files whose every tensor is BF16, F16 or F32");
    case planner::Tier::kTurbo: break;
  }
  throw std::invalid_argument("tier turbo runs a 4-bit DiT, but " + what + (info.reference ? " or bf16 (tier reference)" : "") +
                              "; use tier high");
}

std::string gguf_runs_as(const GgufDitInfo& info, const quant::QuantSpec& target) {
  return "GGUF " + info.types + ", runs as " + (target.quantized() ? "INT8 ConvRot" : "bf16");
}

GgufCachedFile gguf_converted_file(const fs::path& file, const GgufDitInfo& info, const quant::QuantSpec& target,
                                   const fs::path& cache_dir, const GgufConvertHooks& hooks) {
  const std::string target_name = quant::describe(target);
  if (cache_dir.empty()) {
    throw std::invalid_argument("a GGUF DiT is converted once into " + std::string(target.quantized() ? "INT8 ConvRot" : "bf16") +
                                " and kept in the cache directory: set cache_dir (moss-infer --cache-dir)");
  }
  const Identity id = identity_of(file);
  const util::FileInfo& f = id.info;
  const std::string key_text = "mossinfer-gguf-dit|import " + std::to_string(kGgufImportVersion) + "|converter " +
                               std::to_string(quant::kConverterVersion) + "|" + target_name + "|" + std::to_string(f.size) +
                               "|" + std::to_string(f.mtime_ns) + "|" + hex64(f.volume) + ":" + hex64(f.id_high) + hex64(f.id) +
                               "|" + hex64(id.header_xxh64);
  const std::string key = hex64(xxh64(key_text.data(), key_text.size()));
  // The GGUF's path as given, absolute but with its links not resolved: a Hugging Face cache's snapshot link names the
  // model (its blob does not), so a newer download of it replaces the older one's converted file.
  std::error_code ec;
  const fs::path absolute = fs::absolute(file, ec);
  const std::string source = util::path_utf8((ec ? file : absolute).lexically_normal());
  const fs::path dir = cache_dir / "gguf";
  if (!util::make_directories(dir)) throw weights::IoError(util::path_utf8(dir) + ": cannot create the GGUF cache directory");
  const fs::path resolved_dir = util::resolved_path(dir);
  const std::string name = safe_stem(file) + "." + target_name + "." + key + std::string(kExtension);
  const fs::path out = dir / util::path_from_utf8(name);
  const fs::path lock_path = dir / util::path_from_utf8(name + ".lock");
  const auto valid = [&] { return util::file_info(out).regular() && cache_metadata(out, "mossinfer_gguf_key") == key; };

  GgufCachedFile r;
  r.path = out;
  for (;;) {
    r.lock = take_lock(lock_path, util::FileLock::Mode::kShared, hooks);
    if (valid()) {
      sweep(dir, resolved_dir, name, source, target_name);
      return r;
    }
    r.lock.unlock();
    // Converted under the exclusive lock, then used under a shared one (the loop again).
    const util::FileLock writer = take_lock(lock_path, util::FileLock::Mode::kExclusive, hooks);
    if (valid()) continue;  // another process converted it meanwhile

    // Temporaries of this file left by conversions that died (nobody writes one: this holds the lock).
    std::vector<util::DirEntry> entries;
    util::list_directory(dir, entries);
    for (const util::DirEntry& e : entries) {
      if (util::path_utf8(e.name).starts_with(name + ".tmp-")) util::remove_file_in(resolved_dir, dir / e.name);
    }
    const quant::QuantModel model = quant::open_gguf_dit(weights::GgufFile(file));
    std::uint64_t need = kSpaceMargin;
    for (const quant::QuantLinear& l : model.linears) {
      need += static_cast<std::uint64_t>(l.n * l.k) * (target.quantized() ? 1 : 2) + (target.quantized() ? 4 * l.n : 0);
    }
    for (const weights::IndexedTensor& t : model.tensors) need += static_cast<std::uint64_t>(t.numel()) * weights::dtype_size(t.dtype);
    const fs::space_info space = fs::space(dir, ec);
    if (!ec && space.available < need) {
      throw weights::IoError(util::path_utf8(dir) + ": the converted DiT needs " + std::to_string(need >> 20) + " MiB free, " +
                             std::to_string(space.available >> 20) + " MiB are available");
    }

    // The temporary is created first, so an unwritable cache directory fails before any conversion work.
    util::OutputFile tmp(out, util::OutputFile::Create::kTemporary);
    if (!tmp.is_open()) {
      throw weights::IoError(util::path_utf8(dir) + ": cannot create a file in the GGUF cache directory");
    }
    bool stop = false;
    quant::ConvertOptions o;
    o.group = 256;
    if (!target.quantized()) o.keep_bf16 = {"*"};
    o.metadata = {{"mossinfer_gguf_key", key},
                  {"mossinfer_gguf_source", source},
                  {"mossinfer_gguf_target", target_name},
                  {"mossinfer_gguf_types", info.types}};
    o.temporary = &tmp;
    o.cancelled = [&] { return stop || cancelled(hooks); };
    if (hooks.progress) {
      o.progress = [&](std::size_t done, std::size_t total) {
        if (!hooks.progress(static_cast<double>(done) / static_cast<double>(total))) stop = true;
      };
    }
    try {
      quant::convert(file, out, o);
      if (!(identity_of(file) == id)) throw weights::IoError(util::path_utf8(file) + ": the file changed while it was converted");
    } catch (const quant::ConvertCancelled&) {
      tmp.discard();
      throw_cancelled();
    } catch (const quant::OutputError& e) {
      tmp.discard();
      throw weights::IoError(util::path_utf8(dir) + ": cannot write the converted DiT into the GGUF cache directory (" + e.what() + ")");
    } catch (...) {
      tmp.discard();
      throw;
    }
    if (tmp.publish(out)) {
      r.converted = true;
    } else {
      tmp.discard();
      if (!valid()) throw weights::IoError(util::path_utf8(out) + ": cannot publish the converted DiT");
    }
  }
}

}  // namespace mi::pipeline
