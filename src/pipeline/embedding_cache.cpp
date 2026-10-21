#include "pipeline/embedding_cache.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "util/xxh64.h"
#include "weights/safetensors.h"

namespace mi::pipeline {
namespace {

// The file format is little-endian and written with memcpy of native integers.
static_assert(std::endian::native == std::endian::little, "embedding cache files assume a little-endian host");

namespace fs = std::filesystem;

constexpr char kMagic[8] = {'M', 'I', 'E', 'M', 'B', 'E', 'D', '\0'};
constexpr std::size_t kHeaderBytes = 64;
constexpr const char* kExtension = ".miemb";
// Per-entry bookkeeping counted against the memory cap besides the payload and key.
constexpr std::size_t kEntryOverhead = 128;
// Larger files are rejected before reading (a 4096-token embedding of width 4096 is 32 MiB).
constexpr std::uint64_t kMaxFileBytes = std::uint64_t{1} << 30;

std::uint64_t key_hash(const EmbeddingKey& k) {
  Xxh64 h(0x6d6f7373656d6231ull);  // "mossemb1"
  h.update(&k.model_hash, sizeof(k.model_hash));
  h.update(&k.numerics, sizeof(k.numerics));
  h.update(&k.max_sequence_length, sizeof(k.max_sequence_length));
  const std::uint64_t n = k.tokens.size();
  h.update(&n, sizeof(n));
  h.update(k.tokens.data(), k.tokens.size() * sizeof(std::int32_t));
  return h.digest();
}

std::uint64_t payload_hash(const std::int32_t* tokens, std::size_t n, const std::uint16_t* data, std::size_t m) {
  Xxh64 h;
  h.update(tokens, n * sizeof(std::int32_t));
  h.update(data, m * sizeof(std::uint16_t));
  return h.digest();
}

template <typename T>
void put_le(unsigned char* dst, T v) {
  std::memcpy(dst, &v, sizeof(T));
}
template <typename T>
T get_le(const unsigned char* src) {
  T v;
  std::memcpy(&v, src, sizeof(T));
  return v;
}

std::string hex16(std::uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

void hash_string(Xxh64& h, const std::string& s) {
  const std::uint64_t n = s.size();
  h.update(&n, sizeof(n));
  h.update(s.data(), s.size());
}

// The safetensors header (u64 length + JSON: every tensor's name, dtype, shape and offsets) of `path`, hashed with its
// length; an unreadable or malformed file hashes as a marker.
void hash_safetensors_header(Xxh64& h, const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  unsigned char len_bytes[8];
  std::string header;
  if (in.read(reinterpret_cast<char*>(len_bytes), sizeof(len_bytes))) {
    const auto len = get_le<std::uint64_t>(len_bytes);
    if (len <= weights::kMaxHeaderBytes) {
      header.resize(static_cast<std::size_t>(len));
      if (!in.read(header.data(), static_cast<std::streamsize>(len))) header.clear();
    }
  }
  const std::uint8_t ok = header.empty() ? 0 : 1;
  h.update(&ok, sizeof(ok));
  hash_string(h, header);
}

// Header checksum: bytes [0, 48) (everything before it) and [56, 64) (the numerics identity after it).
std::uint64_t header_hash(const unsigned char* header) {
  Xxh64 h;
  h.update(header, 48);
  h.update(header + 56, 8);
  return h.digest();
}

}  // namespace

std::uint64_t text_encoder_hash(const weights::ModelIndex& index) {
  const auto groups = index.groups_of(weights::Component::kTextEncoder);
  if (groups.empty()) throw std::invalid_argument("text_encoder_hash: the index has no text_encoder groups");
  const bool checksummed = std::all_of(groups.begin(), groups.end(), [](const auto* g) { return g->checksum.has_value(); });
  Xxh64 h(0x74652d6964763031ull);  // "te-idv01"
  h.update(&checksummed, sizeof(checksummed));
  std::vector<std::uint32_t> files;
  for (const weights::WeightGroup* g : groups) {
    hash_string(h, g->name);
    hash_string(h, g->layer_type);
    h.update(&g->total_bytes, sizeof(g->total_bytes));
    if (checksummed) h.update(&*g->checksum, sizeof(std::uint64_t));
    for (const weights::IndexedTensor& t : g->tensors) {
      hash_string(h, t.name);
      const auto dtype = static_cast<std::int32_t>(t.dtype);
      h.update(&dtype, sizeof(dtype));
      const std::uint64_t rank = t.shape.size();
      h.update(&rank, sizeof(rank));
      h.update(t.shape.data(), t.shape.size() * sizeof(std::int64_t));
      h.update(&t.nbytes, sizeof(t.nbytes));
      if (!checksummed) {
        // Without content checksums the bytes are identified by where they live and the file's size and mtime.
        h.update(&t.offset, sizeof(t.offset));
        files.push_back(t.file);
      }
    }
  }
  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  for (const std::uint32_t id : files) {
    const weights::IndexedFile& f = index.files.at(id);
    hash_string(h, f.path);
    h.update(&f.size, sizeof(f.size));
    h.update(&f.mtime_ns, sizeof(f.mtime_ns));
    // The header pins the tensor layout of the file actually on disk (a stale index cannot vouch for it).
    hash_safetensors_header(h, index.file_path(id));
  }
  return h.digest();
}

EmbeddingCache::EmbeddingCache(EmbeddingCacheOptions options)
    : max_bytes_(options.max_bytes), disk_dir_(std::move(options.disk_dir)), max_disk_bytes_(options.max_disk_bytes) {
  if (!disk_dir_.empty()) {
    std::error_code ec;
    fs::create_directories(disk_dir_, ec);
    if (ec || !fs::is_directory(disk_dir_, ec)) {
      disk_dir_.clear();
      ++stats_.disk_errors;
    }
  }
}

fs::path EmbeddingCache::disk_path(const EmbeddingKey& key) const {
  if (disk_dir_.empty()) return {};
  return disk_dir_ / (hex16(key_hash(key)) + kExtension);
}

EmbeddingCache::Lru::iterator EmbeddingCache::lookup(const EmbeddingKey& key, std::uint64_t hash) {
  const auto [b, e] = map_.equal_range(hash);
  for (auto it = b; it != e; ++it) {
    if (it->second->key == key) return it->second;
  }
  return lru_.end();
}

void EmbeddingCache::evict() {
  for (auto it = lru_.end(); bytes_ > max_bytes_ && it != lru_.begin();) {
    --it;
    if (it->keep) continue;
    const auto [b, e] = map_.equal_range(key_hash(it->key));
    for (auto m = b; m != e; ++m) {
      if (m->second == it) {
        map_.erase(m);
        break;
      }
    }
    bytes_ -= it->bytes;
    it = lru_.erase(it);
    ++stats_.evictions;
  }
}

void EmbeddingCache::put(const EmbeddingKey& key, std::shared_ptr<const Embedding> value, bool keep) {
  const std::uint64_t hash = key_hash(key);
  const std::size_t bytes = key.tokens.size() * sizeof(std::int32_t) + value->data.size() * sizeof(std::uint16_t) + kEntryOverhead;
  auto it = lookup(key, hash);
  if (it != lru_.end()) {
    keep = keep || it->keep;
    const auto [b, e] = map_.equal_range(hash);
    for (auto m = b; m != e; ++m) {
      if (m->second == it) {
        map_.erase(m);
        break;
      }
    }
    bytes_ -= it->bytes;
    lru_.erase(it);
  }
  if (!keep && bytes > max_bytes_) return;
  if (keep) {
    // One kept entry per model: the previous negative prompt becomes an ordinary LRU entry.
    for (Entry& other : lru_) {
      if (other.keep && other.key.model_hash == key.model_hash) other.keep = false;
    }
  }
  lru_.push_front(Entry{key, std::move(value), bytes, keep});
  map_.emplace(hash, lru_.begin());
  bytes_ += bytes;
  evict();
}

std::shared_ptr<const Embedding> EmbeddingCache::find(const EmbeddingKey& key) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = lookup(key, key_hash(key));
    if (it != lru_.end()) {
      lru_.splice(lru_.begin(), lru_, it);
      ++stats_.memory_hits;
      return it->value;
    }
  }
  std::shared_ptr<const Embedding> value;
  if (!disk_dir_.empty()) {
    std::lock_guard<std::mutex> lock(disk_mu_);
    value = read_disk(key);
  }
  std::lock_guard<std::mutex> lock(mu_);
  if (!value) {
    ++stats_.misses;
    return nullptr;
  }
  ++stats_.disk_hits;
  put(key, value, false);
  return value;
}

void EmbeddingCache::insert(const EmbeddingKey& key, Embedding value, bool keep) {
  if (key.tokens.empty() || value.hidden <= 0 ||
      value.data.size() != key.tokens.size() * static_cast<std::size_t>(value.hidden)) {
    throw std::invalid_argument("EmbeddingCache::insert: embedding size does not match tokens x hidden");
  }
  auto shared = std::make_shared<const Embedding>(std::move(value));
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.inserts;
    put(key, shared, keep);
  }
  if (!disk_dir_.empty()) {
    std::lock_guard<std::mutex> lock(disk_mu_);
    write_disk(key, *shared);
  }
}

void EmbeddingCache::set_max_bytes(std::size_t bytes) {
  std::lock_guard<std::mutex> lock(mu_);
  max_bytes_ = bytes;
  evict();
}

std::size_t EmbeddingCache::max_bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return max_bytes_;
}

EmbeddingCacheStats EmbeddingCache::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  EmbeddingCacheStats s = stats_;
  s.bytes = bytes_;
  s.entries = lru_.size();
  return s;
}

std::shared_ptr<const Embedding> EmbeddingCache::read_disk(const EmbeddingKey& key) {
  const fs::path path = disk_path(key);
  std::error_code ec;
  if (!fs::exists(path, ec)) return nullptr;
  const auto count = [&](std::uint64_t EmbeddingCacheStats::*field) {
    std::lock_guard<std::mutex> lock(mu_);
    ++(stats_.*field);
  };
  const auto reject = [&] {
    std::error_code rm;
    fs::remove(path, rm);
    count(&EmbeddingCacheStats::disk_rejects);
    return nullptr;
  };
  const std::uint64_t size = fs::file_size(path, ec);
  if (ec) {
    count(&EmbeddingCacheStats::disk_errors);
    return nullptr;
  }
  if (size < kHeaderBytes || size > kMaxFileBytes) return reject();
  std::ifstream in(path, std::ios::binary);
  unsigned char header[kHeaderBytes];
  if (!in.read(reinterpret_cast<char*>(header), kHeaderBytes)) {
    count(&EmbeddingCacheStats::disk_errors);
    return nullptr;
  }
  if (std::memcmp(header, kMagic, sizeof(kMagic)) != 0) return reject();
  // Another format version (an older or newer engine sharing the directory): not ours to judge or delete.
  if (get_le<std::uint32_t>(header + 8) != kEmbeddingFileVersion) return nullptr;
  if (get_le<std::uint32_t>(header + 12) != kHeaderBytes || get_le<std::uint64_t>(header + 48) != header_hash(header)) {
    return reject();
  }
  const auto model_hash = get_le<std::uint64_t>(header + 16);
  const auto max_len = get_le<std::int32_t>(header + 24);
  const auto hidden = get_le<std::int32_t>(header + 28);
  const auto n = get_le<std::uint64_t>(header + 32);
  // Sizes are checked against the actual file size before anything is allocated.
  if (hidden <= 0 || n == 0 || n > size || size != kHeaderBytes + n * 4 + n * static_cast<std::uint64_t>(hidden) * 2) {
    return reject();
  }
  // The same file name with another key is a hash collision (or a stale file): a miss, left for insert to replace.
  if (model_hash != key.model_hash || get_le<std::uint64_t>(header + 56) != key.numerics ||
      max_len != key.max_sequence_length || n != key.tokens.size()) {
    return nullptr;
  }
  std::vector<std::int32_t> tokens(n);
  auto value = std::make_shared<Embedding>();
  value->hidden = hidden;
  value->data.resize(n * static_cast<std::size_t>(hidden));
  if (!in.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(n * 4)) ||
      !in.read(reinterpret_cast<char*>(value->data.data()), static_cast<std::streamsize>(value->data.size() * 2))) {
    count(&EmbeddingCacheStats::disk_errors);
    return nullptr;
  }
  if (payload_hash(tokens.data(), tokens.size(), value->data.data(), value->data.size()) != get_le<std::uint64_t>(header + 40)) {
    return reject();
  }
  if (tokens != key.tokens) return nullptr;
  in.close();
  fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // disk LRU recency
  return value;
}

void EmbeddingCache::write_disk(const EmbeddingKey& key, const Embedding& value) {
  const fs::path path = disk_path(key);
  const std::size_t n = key.tokens.size();
  const std::uint64_t file_bytes = kHeaderBytes + n * 4 + value.data.size() * 2;
  if (file_bytes > kMaxFileBytes || file_bytes > max_disk_bytes_) return;
  unsigned char header[kHeaderBytes] = {};
  std::memcpy(header, kMagic, sizeof(kMagic));
  put_le<std::uint32_t>(header + 8, kEmbeddingFileVersion);
  put_le<std::uint32_t>(header + 12, static_cast<std::uint32_t>(kHeaderBytes));
  put_le<std::uint64_t>(header + 16, key.model_hash);
  put_le<std::int32_t>(header + 24, key.max_sequence_length);
  put_le<std::int32_t>(header + 28, value.hidden);
  put_le<std::uint64_t>(header + 32, n);
  put_le<std::uint64_t>(header + 40, payload_hash(key.tokens.data(), n, value.data.data(), value.data.size()));
  put_le<std::uint64_t>(header + 56, key.numerics);
  put_le<std::uint64_t>(header + 48, header_hash(header));

  // Unique per process and thread, so concurrent writers (other processes too) never share a temporary file.
  const std::size_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  const auto stamp = static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count());
  fs::path tmp = path;
  tmp += ".tmp" + hex16(tid ^ stamp) + std::to_string(++tmp_counter_);
  bool ok = false;
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header), kHeaderBytes);
    out.write(reinterpret_cast<const char*>(key.tokens.data()), static_cast<std::streamsize>(n * 4));
    out.write(reinterpret_cast<const char*>(value.data.data()), static_cast<std::streamsize>(value.data.size() * 2));
    out.close();
    ok = static_cast<bool>(out);
  }
  std::error_code ec;
  if (ok) fs::rename(tmp, path, ec);
  if (!ok || ec) {
    fs::remove(tmp, ec);
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.disk_errors;
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.disk_writes;
  }
  trim_disk();
}

void EmbeddingCache::trim_disk() {
  struct File {
    fs::path path;
    fs::file_time_type time;
    std::uint64_t size;
  };
  std::vector<File> files;
  std::uint64_t total = 0;
  std::error_code ec;
  for (fs::directory_iterator it(disk_dir_, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code fe;
    if (it->path().extension() != kExtension || !it->is_regular_file(fe)) continue;
    const std::uint64_t size = it->file_size(fe);
    const fs::file_time_type time = it->last_write_time(fe);
    if (fe) continue;
    files.push_back({it->path(), time, size});
    total += size;
  }
  if (ec) {
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.disk_errors;
    return;
  }
  if (total <= max_disk_bytes_) return;
  std::sort(files.begin(), files.end(), [](const File& a, const File& b) { return a.time < b.time; });
  for (const File& f : files) {
    if (total <= max_disk_bytes_) break;
    std::error_code rm;
    if (fs::remove(f.path, rm)) total -= f.size;
  }
}

}  // namespace mi::pipeline
