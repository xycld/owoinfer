// Embedding cache (src/pipeline/embedding_cache.*): LRU with a kept negative entry, the disk spill format and its
// corruption handling, the text-encoder identity hash, and concurrent use. Host only.
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pipeline/embedding_cache.h"
#include "test_util.h"

namespace {

using namespace mi::pipeline;
namespace fs = std::filesystem;
using mi::testing::TempDir;

constexpr int kHidden = 16;

EmbeddingKey key(std::vector<std::int32_t> tokens, std::uint64_t model = 7, std::int32_t max_len = 4096,
                 std::uint64_t numerics = 3) {
  return EmbeddingKey{model, numerics, max_len, std::move(tokens)};
}

Embedding value(std::size_t tokens, std::uint16_t seed) {
  Embedding e;
  e.hidden = kHidden;
  e.data.resize(tokens * kHidden);
  for (std::size_t i = 0; i < e.data.size(); ++i) e.data[i] = static_cast<std::uint16_t>(seed * 31 + i);
  return e;
}

EmbeddingCacheOptions options(std::size_t max_bytes, fs::path disk_dir = {}, std::size_t max_disk_bytes = std::size_t{4} << 30) {
  EmbeddingCacheOptions o;
  o.max_bytes = max_bytes;
  o.disk_dir = std::move(disk_dir);
  o.max_disk_bytes = max_disk_bytes;
  return o;
}
constexpr std::size_t kDefaultCap = std::size_t{1} << 30;

// Bytes one entry of n tokens counts against the cap (payload + key + fixed overhead).
std::size_t entry_bytes(std::size_t n) { return n * 4 + n * kHidden * 2 + 128; }

TEST(EmbeddingCache, HitsOnlyOnTheExactKey) {
  EmbeddingCache cache;
  cache.insert(key({1, 2, 3}), value(3, 1));
  const auto hit = cache.find(key({1, 2, 3}));
  ASSERT_NE(hit, nullptr);
  EXPECT_EQ(hit->data, value(3, 1).data);
  EXPECT_EQ(hit->hidden, kHidden);
  EXPECT_EQ(cache.find(key({1, 2})), nullptr);
  EXPECT_EQ(cache.find(key({1, 2, 4})), nullptr);
  EXPECT_EQ(cache.find(key({1, 2, 3}, 8)), nullptr);        // other weights
  EXPECT_EQ(cache.find(key({1, 2, 3}, 7, 512)), nullptr);   // other max_sequence_length
  EXPECT_EQ(cache.find(key({1, 2, 3}, 7, 4096, 4)), nullptr);  // other encoder numerics (version, precision, GPU)
  const EmbeddingCacheStats s = cache.stats();
  EXPECT_EQ(s.memory_hits, 1u);
  EXPECT_EQ(s.misses, 5u);
  EXPECT_EQ(s.entries, 1u);
  EXPECT_EQ(s.bytes, entry_bytes(3));
  // Replacing keeps one entry.
  cache.insert(key({1, 2, 3}), value(3, 2));
  EXPECT_EQ(cache.find(key({1, 2, 3}))->data, value(3, 2).data);
  EXPECT_EQ(cache.stats().entries, 1u);
}

TEST(EmbeddingCache, RejectsMismatchedSizes) {
  EmbeddingCache cache;
  EXPECT_THROW(cache.insert(key({1, 2}), value(3, 1)), std::invalid_argument);
  EXPECT_THROW(cache.insert(key({}), value(0, 1)), std::invalid_argument);
  Embedding e = value(2, 1);
  e.hidden = 0;
  EXPECT_THROW(cache.insert(key({1, 2}), e), std::invalid_argument);
}

TEST(EmbeddingCache, EvictsLeastRecentlyUsed) {
  EmbeddingCache cache(options(3 * entry_bytes(10)));
  cache.insert(key({1}), value(1, 0));  // small, oldest
  for (std::int32_t i = 0; i < 3; ++i) cache.insert(key({100 + i, 0, 0, 0, 0, 0, 0, 0, 0, 0}), value(10, 1));
  // {1} was evicted to make room for the third 10-token entry.
  EXPECT_EQ(cache.find(key({1})), nullptr);
  // Touch entry 100, insert another: 101 (now the least recent) goes.
  ASSERT_NE(cache.find(key({100, 0, 0, 0, 0, 0, 0, 0, 0, 0})), nullptr);
  cache.insert(key({103, 0, 0, 0, 0, 0, 0, 0, 0, 0}), value(10, 2));
  EXPECT_NE(cache.find(key({100, 0, 0, 0, 0, 0, 0, 0, 0, 0})), nullptr);
  EXPECT_EQ(cache.find(key({101, 0, 0, 0, 0, 0, 0, 0, 0, 0})), nullptr);
  EXPECT_NE(cache.find(key({102, 0, 0, 0, 0, 0, 0, 0, 0, 0})), nullptr);
  EXPECT_LE(cache.stats().bytes, cache.max_bytes());
  EXPECT_GE(cache.stats().evictions, 2u);
  // An entry larger than the whole cap is not kept in memory.
  cache.insert(key(std::vector<std::int32_t>(40, 5)), value(40, 3));
  EXPECT_EQ(cache.find(key(std::vector<std::int32_t>(40, 5))), nullptr);
  // Shrinking the cap evicts down to it.
  cache.set_max_bytes(entry_bytes(10));
  EXPECT_LE(cache.stats().bytes, entry_bytes(10));
  EXPECT_EQ(cache.stats().entries, 1u);
}

TEST(EmbeddingCache, NegativeEmbeddingIsAlwaysKept) {
  EmbeddingCache cache(options(2 * entry_bytes(10)));
  cache.insert(key({9, 9}), value(2, 9), /*keep=*/true);
  for (std::int32_t i = 0; i < 20; ++i) cache.insert(key({i, 0, 0, 0, 0, 0, 0, 0, 0, 0}), value(10, 1));
  ASSERT_NE(cache.find(key({9, 9})), nullptr);
  // A new negative for the same model demotes the old one, which then ages out like any entry.
  cache.insert(key({8, 8}), value(2, 8), true);
  for (std::int32_t i = 0; i < 20; ++i) cache.insert(key({i, 1, 0, 0, 0, 0, 0, 0, 0, 0}), value(10, 1));
  EXPECT_NE(cache.find(key({8, 8})), nullptr);
  EXPECT_EQ(cache.find(key({9, 9})), nullptr);
  // Kept entries of other models are independent, and kept survives even a zero cap.
  cache.insert(key({8, 8}, 99), value(2, 1), true);
  cache.set_max_bytes(0);
  EXPECT_NE(cache.find(key({8, 8})), nullptr);
  EXPECT_NE(cache.find(key({8, 8}, 99)), nullptr);
  EXPECT_EQ(cache.stats().entries, 2u);
}

TEST(EmbeddingCache, DiskSpillServesLaterInstances) {
  TempDir dir("emb_spill");
  const fs::path spill = dir.path() / "cache" / "nested";
  {
    EmbeddingCache cache(options(1 << 20, spill));
    ASSERT_TRUE(cache.disk_enabled());
    cache.insert(key({1, 2, 3}), value(3, 4));
    EXPECT_EQ(cache.stats().disk_writes, 1u);
    EXPECT_TRUE(fs::is_regular_file(cache.disk_path(key({1, 2, 3}))));
  }
  EmbeddingCache cache(options(1 << 20, spill));
  const auto hit = cache.find(key({1, 2, 3}));
  ASSERT_NE(hit, nullptr);
  EXPECT_EQ(hit->data, value(3, 4).data);
  EXPECT_EQ(cache.stats().disk_hits, 1u);
  // Promoted: the second lookup is a memory hit.
  ASSERT_NE(cache.find(key({1, 2, 3})), nullptr);
  EXPECT_EQ(cache.stats().memory_hits, 1u);
  EXPECT_EQ(cache.find(key({1, 2, 3}, 8)), nullptr);
  // Leftover temporary files are never read.
  for (const auto& e : fs::directory_iterator(spill)) EXPECT_EQ(e.path().extension(), ".miemb") << e.path();
}

class EmbeddingCacheCorruption : public ::testing::Test {
 protected:
  void SetUp() override {
    EmbeddingCache writer(options(kDefaultCap, dir_.path()));
    writer.insert(k_, value(5, 6));
    path_ = writer.disk_path(k_);
    ASSERT_TRUE(fs::is_regular_file(path_));
    std::ifstream in(path_, std::ios::binary);
    bytes_.assign(std::istreambuf_iterator<char>(in), {});
    ASSERT_EQ(bytes_.size(), 64u + 5 * 4 + 5 * kHidden * 2);
  }
  void write(const std::string& bytes) {
    std::ofstream(path_, std::ios::binary | std::ios::trunc) << bytes;
  }
  // A fresh instance (empty memory) must miss and, for a damaged file, delete it.
  void expect_rejected(const char* what) {
    EmbeddingCache cache(options(kDefaultCap, dir_.path()));
    EXPECT_EQ(cache.find(k_), nullptr) << what;
    EXPECT_EQ(cache.stats().disk_rejects, 1u) << what;
    EXPECT_FALSE(fs::exists(path_)) << what;
    // And the cache keeps working: a re-insert restores the file.
    cache.insert(k_, value(5, 6));
    EmbeddingCache again(options(kDefaultCap, dir_.path()));
    EXPECT_NE(again.find(k_), nullptr) << what;
  }

  TempDir dir_{"emb_corrupt"};
  EmbeddingKey k_ = key({10, 20, 30, 40, 50});
  fs::path path_;
  std::string bytes_;
};

TEST_F(EmbeddingCacheCorruption, PayloadBitFlip) {
  std::string b = bytes_;
  b[b.size() - 3] = static_cast<char>(b[b.size() - 3] ^ 0x10);
  write(b);
  expect_rejected("payload flip");
}

TEST_F(EmbeddingCacheCorruption, TokenBitFlip) {
  std::string b = bytes_;
  b[64] = static_cast<char>(b[64] ^ 1);  // same length, the checksum catches it
  write(b);
  expect_rejected("token flip");
}

TEST_F(EmbeddingCacheCorruption, HeaderFields) {
  std::string b = bytes_;
  b[28] = static_cast<char>(b[28] + 1);  // hidden size, header checksum catches it
  write(b);
  expect_rejected("hidden");
}

TEST_F(EmbeddingCacheCorruption, Truncated) {
  write(bytes_.substr(0, bytes_.size() - 1));
  expect_rejected("truncated");
}

TEST_F(EmbeddingCacheCorruption, TruncatedHeader) {
  write(bytes_.substr(0, 20));
  expect_rejected("short header");
}

TEST_F(EmbeddingCacheCorruption, Empty) {
  write("");
  expect_rejected("empty");
}

TEST_F(EmbeddingCacheCorruption, WrongMagic) {
  std::string b = bytes_;
  b[0] = 'X';
  write(b);
  expect_rejected("magic");
}

TEST_F(EmbeddingCacheCorruption, NumericsFieldIsChecksummed) {
  std::string b = bytes_;
  b[56] = static_cast<char>(b[56] ^ 1);
  write(b);
  expect_rejected("numerics field");
}

// A file of another format version (an older or newer engine sharing the directory) is a plain miss and stays.
TEST_F(EmbeddingCacheCorruption, OtherVersionIsAMissNotADeletion) {
  for (const std::uint32_t v : {kEmbeddingFileVersion - 1, kEmbeddingFileVersion + 1}) {
    std::string b = bytes_;
    std::memcpy(b.data() + 8, &v, sizeof(v));
    write(b);
    EmbeddingCache cache(options(kDefaultCap, dir_.path()));
    EXPECT_EQ(cache.find(k_), nullptr) << v;
    EXPECT_EQ(cache.stats().disk_rejects, 0u) << v;
    EXPECT_TRUE(fs::exists(path_)) << v;
    std::ifstream in(path_, std::ios::binary);
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(in), {}), b) << v;
  }
}

// A valid file whose encoder numerics differ (an engine whose TE kernels changed) under this key's name: a plain
// miss, not a deletion, never a stale hit.
TEST_F(EmbeddingCacheCorruption, OtherNumericsIsAMissNotADeletion) {
  TempDir other("emb_numerics");
  EmbeddingCache writer(options(kDefaultCap, other.path()));
  const EmbeddingKey stale = key({10, 20, 30, 40, 50}, 7, 4096, 99);
  writer.insert(stale, value(5, 6));
  fs::copy_file(writer.disk_path(stale), path_, fs::copy_options::overwrite_existing);
  EmbeddingCache cache(options(kDefaultCap, dir_.path()));
  EXPECT_EQ(cache.find(k_), nullptr);
  EXPECT_EQ(cache.stats().disk_rejects, 0u);
  EXPECT_TRUE(fs::exists(path_));
  EXPECT_NE(writer.disk_path(stale), cache.disk_path(k_));  // normally they never even share a name
}

TEST_F(EmbeddingCacheCorruption, CollidingNameIsAMissNotADeletion) {
  // A valid file of another key under this key's name (as a hash collision would produce).
  TempDir other("emb_other");
  EmbeddingCache writer(options(kDefaultCap, other.path()));
  const EmbeddingKey k2 = key({10, 20, 30, 40, 51});
  writer.insert(k2, value(5, 1));
  fs::copy_file(writer.disk_path(k2), path_, fs::copy_options::overwrite_existing);
  EmbeddingCache cache(options(kDefaultCap, dir_.path()));
  EXPECT_EQ(cache.find(k_), nullptr);
  EXPECT_EQ(cache.stats().disk_rejects, 0u);
  EXPECT_TRUE(fs::exists(path_));
}

TEST(EmbeddingCache, DiskCapEvictsOldestFiles) {
  TempDir dir("emb_diskcap");
  const std::size_t file_bytes = 64 + 10 * 4 + 10 * kHidden * 2;
  EmbeddingCache cache(options(kDefaultCap, dir.path(), 3 * file_bytes));
  std::vector<fs::path> paths;
  for (std::int32_t i = 0; i < 5; ++i) {
    const EmbeddingKey k = key({i, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    cache.insert(k, value(10, 1));
    paths.push_back(cache.disk_path(k));
    // Distinct modification times even on coarse-grained file systems.
    fs::last_write_time(paths.back(), fs::file_time_type::clock::now() - std::chrono::hours(10 - i));
  }
  std::uintmax_t total = 0;
  for (const auto& e : fs::directory_iterator(dir.path())) total += e.file_size();
  EXPECT_LE(total, 3 * file_bytes);
  EXPECT_FALSE(fs::exists(paths[0]));
  EXPECT_TRUE(fs::exists(paths[4]));
}

TEST(EmbeddingCache, ConcurrentUse) {
  TempDir dir("emb_threads");
  EmbeddingCache cache(options(40 * entry_bytes(8), dir.path()));
  std::atomic<int> bad{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 200; ++i) {
        const std::int32_t id = (i * 7 + t) % 64;
        const EmbeddingKey k = key({id, 1, 2, 3, 4, 5, 6, 7});
        if (const auto hit = cache.find(k)) {
          if (hit->data != value(8, static_cast<std::uint16_t>(id)).data) ++bad;
        } else {
          cache.insert(k, value(8, static_cast<std::uint16_t>(id)), id == 0);
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(bad.load(), 0);
  const EmbeddingCacheStats s = cache.stats();
  EXPECT_LE(s.bytes, 40 * entry_bytes(8));
  EXPECT_EQ(s.memory_hits + s.disk_hits + s.misses, 8u * 200u);
  EXPECT_EQ(s.disk_rejects, 0u);
  EXPECT_EQ(s.disk_errors, 0u);
}

TEST(EmbeddingCache, TextEncoderHash) {
  using namespace mi::weights;
  ModelIndex idx;
  idx.files = {{"text_encoder/a.safetensors", 1000, 111}, {"transformer/b.safetensors", 2000, 222}};
  WeightGroup te;
  te.component = Component::kTextEncoder;
  te.name = "layers.0";
  te.layer_type = "linear_attention";
  te.tensors = {{"w", DType::kBF16, {4, 4}, 0, 8, 32}};
  te.total_bytes = 32;
  WeightGroup dit;
  dit.component = Component::kTransformer;
  dit.name = "block.0";
  dit.tensors = {{"x", DType::kBF16, {2}, 1, 8, 4}};
  dit.total_bytes = 4;
  idx.groups = {te, dit};
  const std::uint64_t h = text_encoder_hash(idx);
  EXPECT_EQ(h, text_encoder_hash(idx));

  ModelIndex m = idx;
  m.groups[1].tensors[0].nbytes = 8;  // the DiT does not matter
  m.files[1].mtime_ns = 999;
  EXPECT_EQ(text_encoder_hash(m), h);
  m = idx;
  m.files[0].mtime_ns = 112;  // re-downloaded TE file without checksums: new identity
  EXPECT_NE(text_encoder_hash(m), h);
  m = idx;
  m.groups[0].tensors[0].shape = {2, 8};
  EXPECT_NE(text_encoder_hash(m), h);

  // With checksums the identity is the content: mtimes no longer matter, checksums do.
  m = idx;
  m.groups[0].checksum = 0x1234;
  const std::uint64_t hc = text_encoder_hash(m);
  EXPECT_NE(hc, h);
  m.files[0].mtime_ns = 5;
  EXPECT_EQ(text_encoder_hash(m), hc);
  m.groups[0].checksum = 0x1235;
  EXPECT_NE(text_encoder_hash(m), hc);

  m = idx;
  m.groups.erase(m.groups.begin());
  EXPECT_THROW(text_encoder_hash(m), std::invalid_argument);
}

// Without checksums the identity also covers each text-encoder file's safetensors header as it is on disk: a file
// replaced with the same size and modification time but another tensor layout gets a new identity.
TEST(EmbeddingCache, TextEncoderHashReadsSafetensorsHeaders) {
  using namespace mi::weights;
  TempDir dir("emb_te_header");
  const auto write_file = [&](const std::string& json) {
    std::ofstream out(dir.path() / "te.safetensors", std::ios::binary | std::ios::trunc);
    const std::uint64_t n = json.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out << json << std::string(32, '\0');
  };
  ModelIndex idx;
  idx.bundle_dir = dir.path();
  idx.files = {{"te.safetensors", 1000, 111}};
  WeightGroup te;
  te.component = Component::kTextEncoder;
  te.name = "layers.0";
  te.layer_type = "linear_attention";
  te.tensors = {{"w", DType::kBF16, {4, 4}, 0, 8, 32}};
  te.total_bytes = 32;
  idx.groups = {te};

  const std::uint64_t missing = text_encoder_hash(idx);  // unreadable file: hashed as such
  write_file(R"({"w":{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]}})");
  const std::uint64_t a = text_encoder_hash(idx);
  EXPECT_NE(a, missing);
  EXPECT_EQ(text_encoder_hash(idx), a);
  write_file(R"({"w":{"dtype":"BF16","shape":[2,8],"data_offsets":[0,32]}})");  // same length, other layout
  EXPECT_NE(text_encoder_hash(idx), a);
  write_file(R"({"w":{"dtype":"BF16","shape":[4,4],"data_offsets":[0,32]}})");
  EXPECT_EQ(text_encoder_hash(idx), a);
}

}  // namespace
