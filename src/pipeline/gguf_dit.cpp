#include "pipeline/gguf_dit.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "util/file_time.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "util/rename_replace.h"
#include "util/xxh64.h"
#include "weights/file_reader.h"
#include "weights/gguf.h"
#include "weights/quant/quant_convert.h"
#include "weights/safetensors.h"
#include "weights/supported_model.h"

namespace mi::pipeline {

namespace fs = std::filesystem;

namespace {

// Bumped whenever the converted file of the same GGUF bytes would change (the reader, the dequantization, the layout);
// the converter's own version (quant::kConverterVersion) is part of the key too.
constexpr int kGgufImportVersion = 1;
// Free space kept beyond the converted file's size.
constexpr std::uint64_t kSpaceMargin = std::uint64_t{256} << 20;
// A temporary file not written for this long belongs to a conversion that died (a live one writes continuously).
constexpr auto kStaleTemp = std::chrono::minutes(10);

std::string hex64(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

// The GGUF file's identity: size, modification time and XXH64 of its header (metadata and tensor infos).
struct Identity {
  std::uint64_t size = 0;
  std::int64_t mtime_ns = 0;
  std::uint64_t header_xxh64 = 0;
};
Identity identity_of(const fs::path& file) {
  const weights::GgufFile g(file);
  std::vector<std::byte> header(static_cast<std::size_t>(g.header_bytes()));
  weights::FileReader(file).read_at(0, header.data(), header.size());
  return {g.file_size(), util::file_mtime_ns(file), xxh64(header.data(), header.size())};
}

std::uint64_t process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(_getpid());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

// The file stem for a cache name: at most 48 characters of [A-Za-z0-9._-], others replaced by '_'.
std::string safe_stem(const fs::path& file) {
  std::string s = util::path_utf8(file.stem());
  if (s.size() > 48) s.resize(48);
  for (char& c : s) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    if (!ok) c = '_';
  }
  return s.empty() ? std::string("dit") : s;
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

}  // namespace

GgufDitInfo inspect_gguf_dit(const fs::path& file) {
  const weights::GgufFile g(file);
  weights::check_supported_gguf_dit(g);
  const quant::QuantModel m = quant::open_gguf_dit(g);
  std::map<std::uint32_t, int> counts;
  for (const quant::QuantLinear& l : m.linears) ++counts[m.gguf_types.at(l.weight.name)];
  std::vector<std::pair<int, std::uint32_t>> order;
  for (const auto& [type, n] : counts) order.emplace_back(n, type);
  std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
  GgufDitInfo info;
  info.reference = true;
  for (const auto& [n, type] : order) {
    info.types += (info.types.empty() ? "" : " + ") + std::string(weights::ggml_type_name(type));
    using namespace weights::ggml;
    if (type != kQ8_0 && type != kBF16 && type != kF16 && type != kF32) info.reference = false;
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
                                  "; tier reference reads GGUF files of Q8_0, BF16 or F16 block weights");
    case planner::Tier::kTurbo: break;
  }
  throw std::invalid_argument("tier turbo runs a 4-bit DiT, but " + what + (info.reference ? " or bf16 (tier reference)" : "") +
                              "; use tier high");
}

std::string gguf_runs_as(const GgufDitInfo& info, const quant::QuantSpec& target) {
  return "GGUF " + info.types + ", runs as " + (target.quantized() ? "INT8 ConvRot" : "bf16");
}

fs::path gguf_converted_file(const fs::path& file, const GgufDitInfo& info, const quant::QuantSpec& target,
                             const fs::path& cache_dir, bool* converted) {
  if (converted != nullptr) *converted = false;
  const std::string target_name = quant::describe(target);
  if (cache_dir.empty()) {
    throw std::invalid_argument("a GGUF DiT is converted once into " + std::string(target.quantized() ? "INT8 ConvRot" : "bf16") +
                                " and kept in the cache directory: set cache_dir (moss-infer --cache-dir)");
  }
  const Identity id = identity_of(file);
  const std::string key_text = "mossinfer-gguf-dit|import " + std::to_string(kGgufImportVersion) + "|converter " +
                               std::to_string(quant::kConverterVersion) + "|" + target_name + "|" + std::to_string(id.size) +
                               "|" + std::to_string(id.mtime_ns) + "|" + hex64(id.header_xxh64);
  const std::string key = hex64(xxh64(key_text.data(), key_text.size()));
  const std::string source = util::path_utf8(util::path_key(file));
  const fs::path dir = cache_dir / "gguf";
  const fs::path out = dir / (safe_stem(file) + "." + target_name + "." + key + ".safetensors");

  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) throw weights::IoError(util::path_utf8(dir) + ": cannot create the GGUF cache directory (" + ec.message() + ")");
  if (fs::is_regular_file(out, ec)) {
    if (cache_metadata(out, "mossinfer_gguf_key") == key) return out;
    fs::remove(out, ec);  // unreadable or foreign: converted again
  }

  // Leftovers of conversions that died, and room for this one.
  for (const fs::directory_entry& e : fs::directory_iterator(dir, ec)) {
    std::error_code e2;
    if (util::path_utf8(e.path().filename()).find(".tmp-") == std::string::npos || !e.is_regular_file(e2)) continue;
    const auto age = fs::file_time_type::clock::now() - e.last_write_time(e2);
    if (!e2 && age > kStaleTemp) fs::remove(e.path(), e2);
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

  std::mt19937_64 rng(std::random_device{}());
  const fs::path tmp =
      dir / util::path_from_utf8(util::path_utf8(out.filename()) + ".tmp-" + std::to_string(process_id()) + "-" + hex64(rng()));
  quant::ConvertOptions o;
  o.group = 256;
  if (!target.quantized()) o.keep_bf16 = {"*"};
  o.metadata = {{"mossinfer_gguf_key", key},
                {"mossinfer_gguf_source", source},
                {"mossinfer_gguf_target", target_name},
                {"mossinfer_gguf_types", info.types}};
  try {
    quant::convert(file, tmp, o);
    const Identity after = identity_of(file);
    if (after.size != id.size || after.mtime_ns != id.mtime_ns || after.header_xxh64 != id.header_xxh64) {
      throw weights::IoError(util::path_utf8(file) + ": the file changed while it was converted");
    }
    util::rename_replace(tmp, out, ec);
    if (ec) throw weights::IoError(util::path_utf8(out) + ": cannot publish the converted DiT (" + ec.message() + ")");
  } catch (...) {
    std::error_code e2;
    fs::remove(tmp, e2);
    throw;
  }
  if (converted != nullptr) *converted = true;
  // The converted copy of an earlier version of this GGUF path, for the same target, is stale.
  for (const fs::directory_entry& e : fs::directory_iterator(dir, ec)) {
    std::error_code e2;
    if (e.path() == out || e.path().extension() != ".safetensors" || !e.is_regular_file(e2)) continue;
    if (cache_metadata(e.path(), "mossinfer_gguf_source") == source && cache_metadata(e.path(), "mossinfer_gguf_target") == target_name) {
      fs::remove(e.path(), e2);
    }
  }
  return out;
}

}  // namespace mi::pipeline
