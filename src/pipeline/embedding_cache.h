// Cache of text-encoder outputs (P3 step 22): last_hidden_state of one sequence, keyed by the text encoder's weights,
// the encoder's numerics (engine numerics version, precision, GPU and GEMM library: what the output bits depend on
// besides the weights), the exact token ids and max_sequence_length. A full hit for every sequence of a request skips
// the text encoder.
//
// In memory: an LRU with a byte cap (the planner's CachePolicy). The negative prompt's embedding is "kept": exempt
// from eviction, one kept entry per model (keeping a new one demotes the previous one to a normal entry), so the
// usual empty negative prompt never costs a text-encoder pass after the first.
//
// On disk (optional spill directory): write-through on insert, so later processes hit too, with an LRU byte cap
// over the files (last-write time, refreshed on every hit). One file per entry, named by the key's XXH64, in this
// versioned little-endian format:
//   offset  0  char[8] magic "MIEMBED\0"
//           8  u32 version (kEmbeddingFileVersion)    12  u32 header bytes (64)
//          16  u64 model hash                         24  i32 max_sequence_length
//          28  i32 hidden size                        32  u64 token count n
//          40  u64 XXH64 of the payload               48  u64 XXH64 of header bytes [0, 48) and [56, 64)
//          56  u64 numerics identity
//          64  i32 token ids [n], then u16 bf16 bit patterns [n * hidden]
// Files are written to a unique temporary name and renamed into place, so readers never see a partial file. A file
// of another format version is a plain miss and stays (another engine build may share the directory; the LRU cap ages
// it out); so is a different key under the same name. A file that fails any other check (size, magic, header or
// payload checksum) is a miss and is deleted; I/O errors are misses too. Never trusted beyond its checksums: the key
// is compared exactly.
//
// Thread-safe: every member may be called concurrently.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "weights/model_index.h"

namespace mi::pipeline {

inline constexpr std::uint32_t kEmbeddingFileVersion = 2;

// Identity of the text encoder's weights in `index`: XXH64 over every text_encoder group's name, layer type, tensor
// names, dtypes, shapes and sizes, plus the group checksums when all of them have one (`moss-infer index --checksum`)
// or else each referenced file's path, size, modification time and safetensors header bytes (an unreadable file
// hashes as such). Re-downloading the weights without checksums therefore changes the hash (a cache miss, never a
// stale hit).
std::uint64_t text_encoder_hash(const weights::ModelIndex& index);

struct EmbeddingKey {
  std::uint64_t model_hash = 0;
  std::uint64_t numerics = 0;  // qwen35_text::numerics_id(): what the output bits depend on besides weights and tokens
  std::int32_t max_sequence_length = 0;
  std::vector<std::int32_t> tokens;

  bool operator==(const EmbeddingKey&) const = default;
};

// last_hidden_state of one sequence: bf16 bit patterns, [tokens, hidden] row-major.
struct Embedding {
  std::int32_t hidden = 0;
  std::vector<std::uint16_t> data;
};

struct EmbeddingCacheOptions {
  std::size_t max_bytes = std::size_t{1} << 30;  // in-memory cap (payload + key bytes)
  std::filesystem::path disk_dir;                // empty: memory only
  std::size_t max_disk_bytes = std::size_t{4} << 30;
};

struct EmbeddingCacheStats {
  std::uint64_t memory_hits = 0;
  std::uint64_t disk_hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t inserts = 0;
  std::uint64_t evictions = 0;     // in-memory LRU evictions
  std::uint64_t disk_writes = 0;
  std::uint64_t disk_rejects = 0;  // files deleted because they failed a check
  std::uint64_t disk_errors = 0;   // I/O failures (reads, writes, directory scans)
  std::size_t bytes = 0;           // current in-memory bytes
  std::size_t entries = 0;
};

class EmbeddingCache {
 public:
  // Creates disk_dir if needed; if that fails the cache runs memory-only (counted in disk_errors).
  explicit EmbeddingCache(EmbeddingCacheOptions options = {});
  EmbeddingCache(const EmbeddingCache&) = delete;
  EmbeddingCache& operator=(const EmbeddingCache&) = delete;

  // nullptr on a miss. A disk hit is verified and promoted to memory.
  std::shared_ptr<const Embedding> find(const EmbeddingKey& key);

  // Stores (or replaces) an entry; `keep` marks the negative prompt's embedding. An entry larger than the memory cap
  // is not kept in memory unless `keep` (it still goes to disk). Throws std::invalid_argument unless the key has
  // tokens, hidden > 0 and data.size() == tokens * hidden.
  void insert(const EmbeddingKey& key, Embedding value, bool keep = false);

  // New memory cap (e.g. from the planner's CachePolicy); evicts down to it.
  void set_max_bytes(std::size_t bytes);
  std::size_t max_bytes() const;

  bool disk_enabled() const { return !disk_dir_.empty(); }
  // Path of the entry's file in the spill directory (empty if disabled).
  std::filesystem::path disk_path(const EmbeddingKey& key) const;

  EmbeddingCacheStats stats() const;

 private:
  struct Entry {
    EmbeddingKey key;
    std::shared_ptr<const Embedding> value;
    std::size_t bytes = 0;
    bool keep = false;
  };
  using Lru = std::list<Entry>;

  Lru::iterator lookup(const EmbeddingKey& key, std::uint64_t hash);  // memory mutex held
  void put(const EmbeddingKey& key, std::shared_ptr<const Embedding> value, bool keep);  // memory mutex held
  void evict();                                                                          // memory mutex held
  std::shared_ptr<const Embedding> read_disk(const EmbeddingKey& key);                    // disk mutex held
  void write_disk(const EmbeddingKey& key, const Embedding& value);                       // disk mutex held
  void trim_disk();                                                                      // disk mutex held

  mutable std::mutex mu_;  // memory structures and counters
  std::mutex disk_mu_;     // spill directory
  std::size_t max_bytes_;
  std::size_t bytes_ = 0;
  Lru lru_;  // most recently used first
  std::unordered_multimap<std::uint64_t, Lru::iterator> map_;
  std::filesystem::path disk_dir_;
  std::size_t max_disk_bytes_;
  std::uint64_t tmp_counter_ = 0;
  EmbeddingCacheStats stats_;
};

}  // namespace mi::pipeline
