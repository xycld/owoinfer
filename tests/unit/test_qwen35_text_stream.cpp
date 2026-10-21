// Qwen3.5 text encoder weight streaming (P3 step 21): bit-identity with the synchronous P2 path for every source
// and buffer count, one read of each layer group per encode() (counted by the rings), exact device and host
// sizing, and the measured encode time per configuration (prompt + negative, 62+10 and 4096+16 tokens).
//
// PerfMatrix options: MOSSINFER_TE_COLD=1 also measures cold-cache encodes (the text encoder shards are dropped
// from the page cache with posix_fadvise(DONTNEED) first: no root needed, but it evicts them for every process);
// MOSSINFER_TE_HOST=1 adds the host-resident sources (about 13 GiB of host RAM each, one at a time).
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "models/qwen35_text/text_encoder.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/upload.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
using mi::qwen35_text::EncoderOptions;
using mi::qwen35_text::kHidden;
using mi::qwen35_text::kNumLayers;
using mi::qwen35_text::TextEncoder;
using mi::qwen35_text::WeightSource;
using Ids = std::vector<std::int32_t>;
using Bf16 = std::vector<std::uint16_t>;

const fs::path kGolden = fs::path(MOSSINFER_SOURCE_DIR) / "tests/golden";
constexpr int kHookLayers[] = {0, 3, 31};

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}

Ids golden_ids(const std::string& key) {
  const auto v = mi::npy::load(kGolden / (key + ".token_ids.npy")).to_vector<std::int64_t>();
  return Ids(v.begin(), v.end());
}

std::vector<std::string> golden_case_ids() {
  std::ifstream in(kGolden / "cases.json", std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  std::vector<std::string> ids;
  const mi::json::Value cases = mi::json::parse(ss.str());
  for (const auto& c : cases.as_array()) ids.push_back(c.at("id").as_string());
  return ids;
}

std::vector<Ids> all_golden_sequences() {
  std::vector<Ids> seqs;
  for (const auto& id : golden_case_ids()) {
    for (const char* branch : {"cond", "uncond"}) seqs.push_back(golden_ids(id + "." + branch));
  }
  return seqs;
}

std::vector<Ids> prompt_and_negative(const std::string& id) { return {golden_ids(id + ".cond"), golden_ids(id + ".uncond")}; }

std::vector<std::int64_t> lengths_of(const std::vector<Ids>& seqs) {
  std::vector<std::int64_t> l;
  for (const auto& s : seqs) l.push_back(static_cast<std::int64_t>(s.size()));
  return l;
}

EncoderOptions sync_options() {
  EncoderOptions o;
  o.streaming = false;
  return o;
}

EncoderOptions streaming_options(int buffers, WeightSource source = WeightSource::kFile, int threads = 4,
                                 std::size_t chunk = std::size_t{4} << 20, bool direct = false) {
  EncoderOptions o;
  o.weight_buffers = buffers;
  o.source = source;
  o.reader_threads = threads;
  o.chunk_bytes = chunk;
  o.direct_io = direct;
  return o;
}

const char* source_name(const EncoderOptions& o) {
  if (!o.streaming) return "sync (P2)";
  switch (o.source) {
    case WeightSource::kFile: return o.direct_io ? "file O_DIRECT" : "file buffered";
    case WeightSource::kHostPageable: return "host pageable";
    case WeightSource::kHostPinned: return "host pinned";
  }
  return "?";
}

std::string describe(const EncoderOptions& o) {
  char buf[128];
  if (!o.streaming) return source_name(o);
  std::snprintf(buf, sizeof(buf), "%s, %d buf, %d thr, %zu KiB chunk", source_name(o), o.weight_buffers, o.reader_threads,
                o.chunk_bytes >> 10);
  return buf;
}

// Host memory of this process in KiB, from /proc (Linux). Private = anonymous + locked/pinned; file-backed
// pages (the page cache, the mmapped embedding table) are excluded on purpose (plan step 24).
struct HostMem {
  long rss_anon = 0, rss_shmem = 0, vm_lck = 0, vm_pin = 0, private_dirty = 0;
};

HostMem host_mem() {
  HostMem m;
#ifndef _WIN32
  const auto scan = [](const char* path, std::map<std::string, long*> fields) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream ls(line);
      std::string key;
      long v = 0;
      if (ls >> key >> v) {
        auto it = fields.find(key);
        if (it != fields.end()) *it->second = v;
      }
    }
  };
  scan("/proc/self/status", {{"RssAnon:", &m.rss_anon}, {"RssShmem:", &m.rss_shmem}, {"VmLck:", &m.vm_lck}, {"VmPin:", &m.vm_pin}});
  scan("/proc/self/smaps_rollup", {{"Private_Dirty:", &m.private_dirty}});
#endif
  return m;
}

long mem_available_kib() {
  std::ifstream in("/proc/meminfo");
  std::string key, unit;
  long v = 0;
  while (in >> key >> v >> unit) {
    if (key == "MemAvailable:") return v;
  }
  return 0;
}

class Qwen35TextStream : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::is_regular_file(bundle_dir() / "moss.mossidx")) return;
    index_ = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir()));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    // Large enough for the biggest configuration below (3 buffers, all 14 golden sequences).
    const auto lengths = lengths_of(all_golden_sequences());
    arena_ = std::make_unique<mi::runtime::DeviceArena>(
        TextEncoder::workspace_bytes(*index_, lengths, streaming_options(3)) + (std::size_t{64} << 20));
  }
  static void TearDownTestSuite() {
    arena_.reset();
    streams_.reset();
    index_.reset();
  }
  void SetUp() override {
    if (!index_) GTEST_SKIP() << "weights not present at " << bundle_dir();
  }

  std::unique_ptr<TextEncoder> make(const EncoderOptions& o) {
    return std::make_unique<TextEncoder>(*index_, *arena_, streams_->compute(), o);
  }

  // Output plus the raw outputs of the hook layers, all packed.
  struct Run {
    std::vector<Bf16> out;
    std::map<int, Bf16> layers;
  };
  Run run(TextEncoder& enc, const std::vector<Ids>& seqs) {
    Run r;
    mi::qwen35_text::EncodeOptions eo;
    eo.layer_hook = [&](int layer, const mi::Tensor& hidden, cudaStream_t stream) {
      if (std::find(std::begin(kHookLayers), std::end(kHookLayers), layer) == std::end(kHookLayers)) return;
      Bf16& dst = r.layers[layer];
      dst.resize(static_cast<std::size_t>(hidden.numel()));
      MI_CUDA_CHECK(cudaMemcpyAsync(dst.data(), hidden.data, hidden.nbytes(), cudaMemcpyDeviceToHost, stream));
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    r.out = enc.encode_to_host(seqs, eo);
    return r;
  }

  std::uint64_t layer_group_bytes(int layer) const {
    return index_->find_group(mi::weights::Component::kTextEncoder, "layers." + std::to_string(layer))->total_bytes;
  }

  static std::unique_ptr<mi::weights::ModelIndex> index_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
};
std::unique_ptr<mi::weights::ModelIndex> Qwen35TextStream::index_;
std::unique_ptr<mi::runtime::StreamSet> Qwen35TextStream::streams_;
std::unique_ptr<mi::runtime::DeviceArena> Qwen35TextStream::arena_;

void expect_identical(const char* what, const Bf16& a, const Bf16& b) {
  ASSERT_EQ(a.size(), b.size()) << what;
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
  EXPECT_EQ(diff, 0u) << what << ": " << diff << " of " << a.size() << " bf16 values differ";
}

// All 14 golden sequences (the AC4 batch): every streaming configuration, including host-resident sources, gives
// the synchronous path's output and hook-layer activations bit for bit. The AC4 metrics of that output are checked
// by Qwen35Text.GoldenAllCasesLayersAndLastHidden, which runs on the streaming default.
TEST_F(Qwen35TextStream, BitIdenticalToSynchronous) {
  const std::vector<Ids> seqs = all_golden_sequences();
  ASSERT_EQ(seqs.size(), 14u);
  const Run ref = run(*make(sync_options()), seqs);
  ASSERT_EQ(ref.layers.size(), std::size(kHookLayers));
  const EncoderOptions configs[] = {
      streaming_options(2),
      streaming_options(1),
      streaming_options(3),
      // Odd chunk size (tensor slices end mid-page) and thread count, direct I/O.
      streaming_options(3, WeightSource::kFile, 3, (std::size_t{3} << 20) + 4100, true),
      streaming_options(2, WeightSource::kHostPageable),
      streaming_options(2, WeightSource::kHostPinned, 2),
  };
  for (const EncoderOptions& o : configs) {
    SCOPED_TRACE(describe(o));
    auto enc = make(o);
    const Run got = run(*enc, seqs);
    ASSERT_EQ(got.out.size(), ref.out.size());
    for (std::size_t i = 0; i < ref.out.size(); ++i) expect_identical("last_hidden", got.out[i], ref.out[i]);
    for (const auto& [layer, v] : ref.layers) expect_identical(("layer " + std::to_string(layer)).c_str(), got.layers.at(layer), v);
    std::printf("%-55s bit-identical to sync (14 sequences, layers 0/3/31 + output)\n", describe(o).c_str());
    if (o.source != WeightSource::kFile) {
      // The first encode loaded the host copy (and timed it); host_bytes() is what the encoder now holds.
      EXPECT_GT(enc->last_stats().host_load_s, 0.0);
      const std::size_t rings = o.source == WeightSource::kHostPinned
                                    ? 0
                                    : static_cast<std::size_t>(o.reader_threads) * static_cast<std::size_t>(o.ring_slots) *
                                          mi::runtime::align_up(o.chunk_bytes + 4096, 4096);
      EXPECT_EQ(enc->host_resident_bytes() + rings, TextEncoder::host_bytes(*index_, o));
      std::uint64_t layers = 0;
      for (int k = 0; k < kNumLayers; ++k) layers += layer_group_bytes(k);
      EXPECT_GE(enc->host_resident_bytes(), layers);
      // Each tensor's slab adds at most two partial pages.
      EXPECT_LE(enc->host_resident_bytes(), layers + std::size_t{kNumLayers} * 16 * 2 * 4096);
      enc->preload_host();  // already loaded: no-op
      enc->encode_to_host({golden_ids("g_short_1024.uncond")});
      EXPECT_EQ(enc->last_stats().host_load_s, 0.0);
      enc->release_host();
      EXPECT_EQ(enc->host_resident_bytes(), 0u);
    }
  }
}

// Each layer group crosses the rings exactly once per encode(), however many sequences are encoded.
TEST_F(Qwen35TextStream, ReadsEachLayerGroupOncePerCall) {
  for (const EncoderOptions& o : {streaming_options(2), streaming_options(3, WeightSource::kFile, 3, std::size_t{1} << 20)}) {
    SCOPED_TRACE(describe(o));
    auto enc = make(o);
    std::uint64_t expected_chunks = 0, total = 0;
    for (int k = 0; k < kNumLayers; ++k) {
      const auto* g = index_->find_group(mi::weights::Component::kTextEncoder, "layers." + std::to_string(k));
      for (const auto& t : g->tensors) expected_chunks += (t.nbytes + o.chunk_bytes - 1) / o.chunk_bytes;
      total += g->total_bytes;
    }
    for (const auto& seqs : {std::vector<Ids>{golden_ids("g_short_1024.cond")}, prompt_and_negative("g_fridge_1024"),
                             all_golden_sequences()}) {
      enc->encode_to_host(seqs);
      const auto& s = enc->last_stats();
      ASSERT_EQ(s.layer_bytes.size(), static_cast<std::size_t>(kNumLayers));
      for (int k = 0; k < kNumLayers; ++k) EXPECT_EQ(s.layer_bytes[static_cast<std::size_t>(k)], layer_group_bytes(k)) << "layer " << k;
      EXPECT_EQ(s.weight_bytes, total);
      EXPECT_EQ(s.read_chunks, expected_chunks);
      std::printf("%zu sequences: %llu bytes in %llu ring fills (32 layer groups, each once)\n", seqs.size(),
                  static_cast<unsigned long long>(s.weight_bytes), static_cast<unsigned long long>(s.read_chunks));
    }
  }
}

// Device high-water equals workspace_bytes(index, lengths, options) for every buffer count and the sync path, and
// the original overload sizes the default options.
TEST_F(Qwen35TextStream, DeviceHighWaterEqualsWorkspaceBytes) {
  const std::vector<std::vector<Ids>> batches = {prompt_and_negative("g_fridge_1024"), prompt_and_negative("g_long4096_1024")};
  for (const EncoderOptions& o : {sync_options(), streaming_options(1), streaming_options(2), streaming_options(3)}) {
    SCOPED_TRACE(describe(o));
    auto enc = make(o);
    for (const auto& seqs : batches) {
      const std::size_t used_before = arena_->used();
      arena_->reset_high_water();
      enc->encode_to_host(seqs);
      EXPECT_EQ(arena_->used(), used_before);
      const auto lengths = lengths_of(seqs);
      EXPECT_EQ(arena_->high_water() - used_before, TextEncoder::workspace_bytes(*index_, lengths, o));
    }
  }
  const auto lengths = lengths_of(batches[0]);
  EXPECT_EQ(TextEncoder::workspace_bytes(*index_, lengths), TextEncoder::workspace_bytes(*index_, lengths, EncoderOptions{}));
  EXPECT_EQ(TextEncoder::workspace_bytes(*index_, lengths, streaming_options(3)) -
                TextEncoder::workspace_bytes(*index_, lengths, streaming_options(2)),
            mi::runtime::align_up(TextEncoder(*index_, *arena_, streams_->compute(), sync_options()).weight_buffer_bytes()));
}

// host_bytes() for file sources is the rings alone (host-resident sources are checked against held memory in
// BitIdenticalToSynchronous); no host copy without a host-resident source.
TEST_F(Qwen35TextStream, HostBytesOfFileSources) {
  EXPECT_EQ(TextEncoder::host_bytes(*index_, streaming_options(2)), 4 * 4 * mi::runtime::align_up((std::size_t{4} << 20) + 4096, 4096));
  EXPECT_EQ(TextEncoder::host_bytes(*index_, streaming_options(3, WeightSource::kFile, 2, std::size_t{1} << 20)),
            2 * 4 * mi::runtime::align_up((std::size_t{1} << 20) + 4096, 4096));
  EXPECT_EQ(TextEncoder::host_bytes(*index_, sync_options()), 0u);
  auto enc = make(streaming_options(2));
  enc->preload_host();  // kFile: no-op
  EXPECT_EQ(enc->host_resident_bytes(), 0u);
}

// Cancel between layers (the flag is set by the layer hook of layer 5, so the check before layer 6 must see it):
// encode() throws Cancelled without issuing another layer, leaves the arena as it was, and the encoder's next encode
// is bit-identical to an uncancelled one.
TEST_F(Qwen35TextStream, CancelBetweenLayersLeavesEncoderUsable) {
  const std::vector<Ids> seqs = prompt_and_negative("g_fridge_1024");
  for (const EncoderOptions& o : {sync_options(), streaming_options(1), streaming_options(2)}) {
    SCOPED_TRACE(describe(o));
    auto enc = make(o);
    const auto ref = enc->encode_to_host(seqs);
    std::atomic<bool> cancel{false};
    int last_layer = -1;
    mi::qwen35_text::EncodeOptions eo;
    eo.cancel = &cancel;
    eo.layer_hook = [&](int layer, const mi::Tensor&, cudaStream_t) {
      last_layer = layer;
      if (layer == 5) cancel.store(true);
    };
    const std::size_t used = arena_->used();
    EXPECT_THROW(enc->encode_to_host(seqs, eo), mi::qwen35_text::Cancelled);
    EXPECT_EQ(last_layer, 5);
    EXPECT_EQ(arena_->used(), used);
    // An unset flag only throttles: same bits.
    cancel.store(false);
    const auto again = enc->encode_to_host(seqs, mi::qwen35_text::EncodeOptions{{}, &cancel, {}});
    const auto after = enc->encode_to_host(seqs);
    ASSERT_EQ(after.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
      expect_identical("throttled", again[i], ref[i]);
      expect_identical("after cancel", after[i], ref[i]);
    }
  }
}

// A reader that fails mid-stream (fault injected at layer 7 on reader 1) or after its last copy (the late failure
// that the compute thread's waits no longer see): encode() throws the reader's error instead of hanging or losing
// it, rolls the arena back, and the encoder works afterwards.
TEST_F(Qwen35TextStream, ReaderFailureIsReportedAndRecoverable) {
  const std::vector<Ids> seqs = prompt_and_negative("g_fridge_1024");
  auto enc = make(streaming_options(2));
  const auto ref = enc->encode_to_host(seqs);
  for (const int fail_layer : {7, kNumLayers}) {
    SCOPED_TRACE("fault at layer " + std::to_string(fail_layer));
    mi::qwen35_text::EncodeOptions eo;
    eo.reader_hook = [&](int reader, int layer) {
      if (reader == 1 && layer == fail_layer) throw std::runtime_error("injected reader fault");
    };
    const std::size_t used = arena_->used();
    try {
      enc->encode_to_host(seqs, eo);
      ADD_FAILURE() << "expected the injected reader fault";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("injected reader fault"), std::string::npos) << e.what();
    }
    EXPECT_EQ(arena_->used(), used);
    const auto after = enc->encode_to_host(seqs);
    ASSERT_EQ(after.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) expect_identical("after fault", after[i], ref[i]);
  }
}

TEST_F(Qwen35TextStream, RejectsInvalidOptions) {
  for (auto bad : {&EncoderOptions::weight_buffers, &EncoderOptions::reader_threads, &EncoderOptions::ring_slots}) {
    EncoderOptions o;
    o.*bad = 0;
    EXPECT_THROW(make(o), std::invalid_argument);
    EXPECT_THROW(TextEncoder::workspace_bytes(*index_, std::vector<std::int64_t>{4}, o), std::invalid_argument);
  }
  EncoderOptions o;
  o.chunk_bytes = 0;
  EXPECT_THROW(make(o), std::invalid_argument);
  o.streaming = false;  // the sync path ignores the streaming settings
  EXPECT_NO_THROW(make(o));
}

// plan_group_upload() lays a group out exactly like upload_group(), and PinnedRing::copy_host() moves bytes.
TEST_F(Qwen35TextStream, PlanGroupUploadAndRingCopyHost) {
  const auto* g = index_->find_group(mi::weights::Component::kTextEncoder, "layers.0");
  const std::size_t bytes = mi::weights::group_device_bytes(*g);
  mi::DeviceBuffer a(bytes), b(bytes);
  mi::weights::ReaderCache readers(*index_, false);
  mi::runtime::PinnedRing ring(streams_->copy(), 3, std::size_t{1} << 20);
  const mi::weights::TensorMap uploaded = mi::weights::upload_group(readers, *g, a.get(), bytes, ring);
  std::vector<mi::weights::UploadPiece> pieces;
  const mi::weights::TensorMap planned = mi::weights::plan_group_upload(*g, b.get(), bytes, pieces);
  ASSERT_EQ(pieces.size(), g->tensors.size());
  ASSERT_EQ(planned.size(), uploaded.size());
  for (const auto& [name, t] : uploaded) {
    const mi::Tensor& p = planned.at(name);
    EXPECT_EQ(static_cast<const std::byte*>(p.data) - static_cast<const std::byte*>(b.get()),
              static_cast<const std::byte*>(t.data) - static_cast<const std::byte*>(a.get()))
        << name;
  }
  for (std::size_t i = 1; i < pieces.size(); ++i) {
    EXPECT_TRUE(pieces[i - 1].tensor->file < pieces[i].tensor->file ||
                (pieces[i - 1].tensor->file == pieces[i].tensor->file && pieces[i - 1].tensor->offset < pieces[i].tensor->offset));
  }
  // Fill b through copy_host from a host copy of the group, then compare with the ring upload.
  const mi::weights::HostGroup host = mi::weights::load_group_to_host(readers, *g);
  EXPECT_EQ(host.storage_bytes, mi::weights::host_group_bytes(*g));
  const auto before = ring.stats();
  for (const auto& piece : pieces) ring.copy_host(host.tensors.at(piece.tensor->name).data, piece.tensor->nbytes, piece.dst);
  ring.synchronize();
  EXPECT_EQ(ring.stats().bytes - before.bytes, g->total_bytes);
  std::vector<std::byte> ha(bytes), hb(bytes);
  a.download(ha.data(), bytes);
  b.download(hb.data(), bytes);
  for (const auto& [name, t] : uploaded) {
    const auto off = static_cast<std::size_t>(static_cast<const std::byte*>(t.data) - static_cast<const std::byte*>(a.get()));
    EXPECT_EQ(std::memcmp(ha.data() + off, hb.data() + off, t.nbytes()), 0) << name;
  }
}

#ifndef _WIN32
// Drops the text encoder shards from the page cache (clean, unmapped pages only).
void drop_te_page_cache(const mi::weights::ModelIndex& index) {
  for (std::uint32_t f = 0; f < index.files.size(); ++f) {
    if (index.files[f].path.find("text_encoder/") != 0) continue;
    const int fd = ::open(index.file_path(f).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) continue;
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
  }
}
#endif

// Measured encode of prompt + negative per configuration (second run of each, page cache warm), with achieved
// load bandwidth, the hidden fraction of the load time, and the host private memory each mode adds.
TEST_F(Qwen35TextStream, PerfMatrix) {
  const bool cold = std::getenv("MOSSINFER_TE_COLD") != nullptr;
  const bool host = std::getenv("MOSSINFER_TE_HOST") != nullptr;
  struct Case {
    const char* id;
    std::vector<Ids> seqs;
  };
  const Case cases[] = {{"g_fridge_1024", prompt_and_negative("g_fridge_1024")},
                        {"g_long4096_1024", prompt_and_negative("g_long4096_1024")}};
  const EncoderOptions configs[] = {
      sync_options(),
      streaming_options(2),
      streaming_options(3),
      streaming_options(1),
      streaming_options(2, WeightSource::kFile, 2),
      streaming_options(2, WeightSource::kFile, 8),
      streaming_options(2, WeightSource::kFile, 4, std::size_t{4} << 20, true),
      streaming_options(2, WeightSource::kHostPageable),
      streaming_options(2, WeightSource::kHostPinned),
  };
  std::printf("MemAvailable before: %.1f GiB\n", static_cast<double>(mem_available_kib()) / (1 << 20));
  std::printf("%-46s %-16s %7s %7s %7s %7s %7s %6s %7s | %s\n", "config", "case", "total", "load", "wait", "compute",
              "GB/s", "ovl%", "1st", "host private delta (MiB): anon/locked/pinned/shmem, host_bytes");
  for (const EncoderOptions& o : configs) {
    if (o.streaming && o.source != WeightSource::kFile && !host) continue;
    const HostMem m0 = host_mem();
    auto enc = make(o);
    const auto t0 = std::chrono::steady_clock::now();
    enc->preload_host();
    const double preload_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const HostMem m1 = host_mem();
    char mem[192];
    std::snprintf(mem, sizeof(mem), "%ld/%ld/%ld/%ld, %.0f; preload %.2f s", (m1.rss_anon - m0.rss_anon) >> 10,
                  (m1.vm_lck - m0.vm_lck) >> 10, (m1.vm_pin - m0.vm_pin) >> 10, (m1.rss_shmem - m0.rss_shmem) >> 10,
                  static_cast<double>(TextEncoder::host_bytes(*index_, o)) / (1 << 20), preload_s);
    for (const Case& c : cases) {
      double first = 0;
      for (int r = 0; r < 2; ++r) {
        enc->encode_to_host(c.seqs);
        if (r == 0) first = enc->last_stats().total_s;
      }
      const auto& s = enc->last_stats();
      std::printf("%-46s %-16s %7.3f %7.3f %7.3f %7.3f %7.2f %6.1f %7.3f | %s\n", describe(o).c_str(), c.id, s.total_s,
                  s.weight_load_s, s.weight_wait_s, s.compute_s, static_cast<double>(s.weight_bytes) / 1e9 / s.weight_load_s,
                  o.streaming ? 100 * s.overlap() : 0.0, first, mem);
      RecordProperty(describe(o) + " " + c.id + " total_s", std::to_string(s.total_s));
#ifndef _WIN32
      if (cold && o.streaming && o.source == WeightSource::kFile && o.weight_buffers == 2) {
        drop_te_page_cache(*index_);
        enc->encode_to_host(c.seqs);
        const auto& sc = enc->last_stats();
        std::printf("%-46s %-16s %7.3f %7.3f %7.3f %7.3f %7.2f %6.1f   (cold page cache)\n", describe(o).c_str(), c.id,
                    sc.total_s, sc.weight_load_s, sc.weight_wait_s, sc.compute_s,
                    static_cast<double>(sc.weight_bytes) / 1e9 / sc.weight_load_s, 100 * sc.overlap());
      }
#endif
    }
  }
  if (!cold) std::printf("(cold-cache rows: set MOSSINFER_TE_COLD=1)\n");
  if (!host) std::printf("(host-resident rows, ~13 GiB of host RAM each: set MOSSINFER_TE_HOST=1)\n");
}

}  // namespace
