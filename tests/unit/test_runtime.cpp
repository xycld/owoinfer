#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "runtime/device_arena.h"
#include "runtime/graph_cache.h"
#include "runtime/pinned_ring.h"
#include "runtime/stream_set.h"
#include "test_runtime_kernels.h"
#include "test_util.h"
#include "weights/file_reader.h"

namespace {

using namespace mi::runtime;
using mi::testing::TempDir;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

constexpr std::size_t kMiB = std::size_t{1} << 20;

// ---- DeviceArena ------------------------------------------------------------------------------

TEST(DeviceArena, AlignmentAndRegions) {
  DeviceArena a(8 * kMiB);
  EXPECT_EQ(a.capacity(), 8 * kMiB);
  auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % kArenaAlign == 0; };
  void* p1 = a.alloc_persistent(1);
  void* p2 = a.alloc_persistent(257);
  void* s1 = a.alloc_stack(3);
  void* s2 = a.alloc_stack(1000);
  EXPECT_TRUE(aligned(p1) && aligned(p2) && aligned(s1) && aligned(s2));
  EXPECT_EQ(static_cast<char*>(p2) - static_cast<char*>(p1), 256);
  EXPECT_EQ(a.persistent_used(), 256u + 512u);
  EXPECT_EQ(a.stack_used(), 256u + 1024u);
  // persistent grows up from the base, stack grows down from the top
  EXPECT_EQ(p1, a.base());
  EXPECT_EQ(static_cast<char*>(s1) + 256, static_cast<char*>(a.base()) + a.capacity());
  EXPECT_LT(s2, s1);
  // zero-size request is legal and free
  const std::size_t before = a.used();
  EXPECT_NE(a.alloc_stack(0), nullptr);
  EXPECT_EQ(a.used(), before);
}

TEST(DeviceArena, TensorsAreUsableDeviceViews) {
  DeviceArena a(4 * kMiB);
  mi::Tensor w = a.alloc_persistent_tensor(mi::DType::kBF16, {5, 7});
  mi::Tensor x = a.alloc_tensor(mi::DType::kF32, {3, 4});
  EXPECT_EQ(w.device, mi::Device::kCUDA);
  EXPECT_EQ(w.dtype, mi::DType::kBF16);
  EXPECT_EQ(w.dim(0), 5);
  EXPECT_EQ(w.nbytes(), 70u);
  EXPECT_EQ(a.persistent_used(), 256u);
  EXPECT_EQ(a.stack_used(), 256u);
  MI_CUDA_CHECK(cudaMemset(x.data, 0x5A, x.nbytes()));
  std::vector<unsigned char> h(x.nbytes());
  MI_CUDA_CHECK(cudaMemcpy(h.data(), x.data, h.size(), cudaMemcpyDeviceToHost));
  for (unsigned char c : h) EXPECT_EQ(c, 0x5A);
}

TEST(DeviceArena, OverflowLeavesStateUntouched) {
  DeviceArena a(1 * kMiB);
  a.alloc_persistent(600 * 1024);
  const std::size_t used = a.used();
  EXPECT_THROW(a.alloc_stack(500 * 1024), ArenaOverflow);
  EXPECT_THROW(a.alloc_persistent(~std::size_t{0}), ArenaOverflow);
  EXPECT_EQ(a.used(), used);
  // exactly filling is fine, one more byte is not
  EXPECT_NO_THROW(a.alloc_stack(a.available()));
  EXPECT_EQ(a.available(), 0u);
  EXPECT_THROW(a.alloc_persistent(1), ArenaOverflow);
  try {
    a.alloc_stack(1);
    FAIL();
  } catch (const ArenaOverflow& e) {
    EXPECT_EQ(e.requested, 1u);
    EXPECT_EQ(e.available, 0u);
  }
}

TEST(DeviceArena, BudgetErrorReportsFreeMemory) {
  const std::size_t free_b = DeviceArena::free_device_bytes();
  const std::size_t ask = free_b + 2048 * kMiB;
  try {
    DeviceArena a(ask);
    FAIL() << "expected BudgetError";
  } catch (const BudgetError& e) {
    EXPECT_EQ(e.requested, ask / kArenaAlign * kArenaAlign);
    EXPECT_LE(e.free, free_b + 64 * kMiB);  // free memory may fluctuate slightly between calls
    EXPECT_NE(std::string(e.what()).find("MiB"), std::string::npos);
  }
  // No cudaMalloc happened and the context is healthy.
  EXPECT_NO_THROW(DeviceArena(1 * kMiB));
}

// Asking for (nearly) all free memory passes the cudaMemGetInfo check but may still fail in cudaMalloc
// (reserved/fragmented memory, another process racing us). That must surface as BudgetError and leave
// no sticky CUDA error behind.
TEST(DeviceArena, MallocFailureMapsToBudgetError) {
  const std::size_t free_b = DeviceArena::free_device_bytes();
  try {
    DeviceArena a(free_b);  // may legitimately succeed on some drivers
    SUCCEED();
  } catch (const BudgetError& e) {
    EXPECT_GT(e.requested, 0u);
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_NO_THROW(DeviceArena(1 * kMiB));
}

TEST(DeviceArena, MarkRollbackAndHighWater) {
  DeviceArena a(4 * kMiB);
  a.alloc_persistent(1024);
  EXPECT_EQ(a.high_water(), 1024u);
  const std::size_t m0 = a.mark();
  a.alloc_stack(100 * 1024);
  const std::size_t m1 = a.mark();
  a.alloc_stack(200 * 1024);
  EXPECT_EQ(a.high_water(), 1024u + 300 * 1024);
  a.rollback(m1);
  EXPECT_EQ(a.stack_used(), 100 * 1024u);
  a.rollback(m0);
  EXPECT_EQ(a.stack_used(), 0u);
  EXPECT_EQ(a.high_water(), 1024u + 300 * 1024);  // peak survives rollback
  EXPECT_THROW(a.rollback(m1), std::invalid_argument);  // mark above the top
  // reuse after rollback hands out the same addresses
  void* first = a.alloc_stack(4096);
  a.rollback(0);
  EXPECT_EQ(a.alloc_stack(4096), first);
  a.rollback(0);
  a.reset_high_water();
  EXPECT_EQ(a.high_water(), a.used());
  {
    StackScope scope(a);
    a.alloc_stack(64 * 1024);
    EXPECT_EQ(a.stack_used(), 64 * 1024u);
  }
  EXPECT_EQ(a.stack_used(), 0u);
}

TEST(DeviceArena, MoveTransfersOwnership) {
  DeviceArena a(1 * kMiB);
  a.alloc_persistent(512);
  void* base = a.base();
  DeviceArena b(std::move(a));
  EXPECT_EQ(b.base(), base);
  EXPECT_EQ(b.persistent_used(), 512u);
  EXPECT_EQ(a.base(), nullptr);
  EXPECT_EQ(a.capacity(), 0u);
}

// ---- StreamSet --------------------------------------------------------------------------------

TEST(StreamSet, CrossStreamDependencyOrdersWork) {
  StreamSet ss;
  mi::DeviceBuffer buf(2 * sizeof(int));
  int* d = static_cast<int*>(buf.get());
  int zero[2] = {0, 0};
  buf.upload(zero, sizeof zero);
  // Producer on the copy stream is slow; consumer on compute must still see its result.
  mi::testing::launch_spin(30'000'000, ss.copy());
  mi::testing::launch_add(d, 1, 41, ss.copy());
  ss.depend(ss.compute(), ss.copy());
  mi::testing::launch_copy_plus_one(d, d + 1, ss.compute());
  MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
  int out[2];
  buf.download(out, sizeof out);
  EXPECT_EQ(out[0], 41);
  EXPECT_EQ(out[1], 42);
}

TEST(StreamSet, EventPoolReusesEvents) {
  StreamSet ss;
  for (int i = 0; i < 100; ++i) ss.depend(ss.compute(), ss.copy());
  MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
  EXPECT_EQ(ss.events().created(), 1u);
  cudaEvent_t e1 = ss.record(ss.copy());
  cudaEvent_t e2 = ss.record(ss.copy());
  EXPECT_NE(e1, e2);
  ss.events().release(e1);
  ss.events().release(e2);
}

// Waiting from a capturing stream on an event recorded outside the capture is an error; depend() must
// hand its pooled event back instead of leaking it.
TEST(StreamSet, EventReturnedToPoolWhenWaitFails) {
  StreamSet ss;
  MI_CUDA_CHECK(cudaStreamBeginCapture(ss.compute(), cudaStreamCaptureModeRelaxed));
  for (int i = 0; i < 5; ++i) EXPECT_THROW(ss.depend(ss.compute(), ss.copy()), std::runtime_error);
  cudaGraph_t g = nullptr;
  cudaStreamEndCapture(ss.compute(), &g);  // invalidated capture: reports an error, g stays null
  if (g != nullptr) cudaGraphDestroy(g);
  cudaGetLastError();
  EXPECT_EQ(ss.events().created(), 1u);
  ss.depend(ss.compute(), ss.copy());
  MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
  EXPECT_EQ(ss.events().created(), 1u);
}

// ---- PinnedRing -------------------------------------------------------------------------------

// Pseudo-random file content; position-dependent so misplaced chunks are detected.
unsigned char file_byte(std::uint64_t i) { return static_cast<unsigned char>((i * 2654435761u) >> 13); }

fs::path write_pattern_file(const fs::path& path, std::size_t size) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  std::vector<char> chunk(kMiB);
  for (std::size_t base = 0; base < size; base += chunk.size()) {
    const std::size_t n = std::min(chunk.size(), size - base);
    for (std::size_t i = 0; i < n; ++i) chunk[i] = static_cast<char>(file_byte(base + i));
    out.write(chunk.data(), static_cast<std::streamsize>(n));
  }
  return path;
}

std::vector<unsigned char> download(const void* dev, std::size_t n) {
  std::vector<unsigned char> h(n);
  MI_CUDA_CHECK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}

bool matches_file(const void* dev, std::uint64_t file_off, std::size_t n) {
  auto h = download(dev, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (h[i] != file_byte(file_off + i)) return false;
  }
  return true;
}

class RingFiles : public ::testing::TestWithParam<bool> {};

TEST_P(RingFiles, OddOffsetsAndSizesThroughThreeSlots) {
  const bool direct = GetParam();
  TempDir dir("ring");
  const std::size_t file_size = 7 * kMiB + 1234;
  const fs::path path = write_pattern_file(dir.path() / "f.bin", file_size);
  mi::weights::FileReader reader(path, direct);
  StreamSet ss;
  PinnedRing ring(ss.copy(), 3, 1 * kMiB);  // slot = 1 MiB so ranges span many chunks
  EXPECT_EQ(ring.slots(), 3u);

  struct Range { std::uint64_t off; std::size_t n; };
  const std::vector<Range> ranges = {
      {0, 1},        {1, 4095},        {4095, 2},         {4096, 4096},     {4097, 3 * kMiB + 17},
      {123, 1 * kMiB}, {5, 1 * kMiB - 5}, {kMiB - 1, 1 * kMiB + 2}, {file_size - 7, 7}, {777, 5 * kMiB + 4242},
  };
  mi::DeviceBuffer dev(64 * kMiB);
  std::size_t at = 0;
  std::vector<std::size_t> dst_off;
  for (const auto& r : ranges) {
    at = align_up(at);
    dst_off.push_back(at);
    ring.copy_range(reader, r.off, r.n, static_cast<char*>(dev.get()) + at);
    at += r.n;
  }
  ring.synchronize();
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    EXPECT_TRUE(matches_file(static_cast<char*>(dev.get()) + dst_off[i], ranges[i].off, ranges[i].n))
        << "range " << i << " offset " << ranges[i].off << " size " << ranges[i].n << " direct " << direct;
  }
  EXPECT_GT(ring.stats().chunks, ranges.size());
}

TEST_P(RingFiles, CoalescedPiecesWithGaps) {
  const bool direct = GetParam();
  TempDir dir("pieces");
  const std::size_t file_size = 3 * kMiB;
  const fs::path path = write_pattern_file(dir.path() / "f.bin", file_size);
  mi::weights::FileReader reader(path, direct);
  StreamSet ss;
  PinnedRing ring(ss.copy(), 3, 256 * 1024);

  // 200 small tensors with odd sizes, some adjacent, some separated by gaps (small and > 64 KiB), plus one larger than a slot.
  std::vector<PinnedRing::Piece> pieces;
  mi::DeviceBuffer dev(8 * kMiB);
  std::uint64_t off = 11;
  std::size_t dst = 0;
  for (int i = 0; i < 200; ++i) {
    const std::size_t n = 1 + static_cast<std::size_t>((i * 7919) % 9000);
    pieces.push_back({off, n, static_cast<char*>(dev.get()) + dst});
    dst = align_up(dst + n);
    off += n + ((i % 17 == 0) ? 100'000 : (i % 3 == 0 ? 0 : 13));
  }
  const std::size_t big = 700 * 1024 + 3;
  ASSERT_LT(off + big, file_size);
  pieces.push_back({off, big, static_cast<char*>(dev.get()) + dst});
  ring.copy_pieces(reader, pieces);
  ring.synchronize();
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    ASSERT_TRUE(matches_file(pieces[i].dst, pieces[i].file_offset, pieces[i].size)) << "piece " << i;
  }
  // coalescing: far fewer disk reads than pieces
  EXPECT_LT(ring.stats().chunks, 120u);

  std::vector<PinnedRing::Piece> bad = {{100, 50, dev.get()}, {120, 10, dev.get()}};
  EXPECT_THROW(ring.copy_pieces(reader, bad), std::invalid_argument);
  std::vector<PinnedRing::Piece> past = {{file_size - 4, 8, dev.get()}};
  EXPECT_THROW(ring.copy_pieces(reader, past), mi::weights::IoError);
}

INSTANTIATE_TEST_SUITE_P(Io, RingFiles, ::testing::Values(false, true), [](const auto& i) { return i.param ? "Direct" : "Buffered"; });

TEST(PinnedRing, FirstSlotsNeverStallAndSlotsAreReusedOnlyAfterCompletion) {
  TempDir dir("stall");
  const fs::path path = write_pattern_file(dir.path() / "f.bin", 16 * kMiB);
  mi::weights::FileReader reader(path, true);
  StreamSet ss;
  PinnedRing ring(ss.copy(), 3, 1 * kMiB);
  mi::DeviceBuffer dev(16 * kMiB);
  // Block the copy stream on a host-controlled latch (not a timed spin, which can finish before slow
  // disk reads do): no copy can complete, yet the first three fills must not wait.
  std::atomic<bool> released{false};
  MI_CUDA_CHECK(cudaLaunchHostFunc(
      ss.copy(),
      [](void* flag) {
        while (!static_cast<std::atomic<bool>*>(flag)->load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      },
      &released));
  ring.copy_range(reader, 0, 3 * kMiB, dev.get());
  EXPECT_EQ(ring.stats().chunks, 3u);
  EXPECT_EQ(ring.stats().stalls, 0u);
  // The fourth fill reuses slot 0 and must wait for its copy, which sits behind the latch.
  std::thread producer([&] { ring.copy_range(reader, 3 * kMiB, 1 * kMiB, static_cast<char*>(dev.get()) + 3 * kMiB); });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (ring.stats().stalls == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(ring.stats().stalls, 1u);
  released.store(true);
  producer.join();
  ring.synchronize();
  EXPECT_TRUE(matches_file(dev.get(), 0, 4 * kMiB));
}

// If a later copy of a coalesced group throws, the earlier ones are already queued and read the slot:
// the slot must stay guarded by its event, so synchronize() still waits for them.
TEST(PinnedRing, FailedCoalescedGroupStillGuardsSlot) {
  TempDir dir("fail");
  const fs::path path = write_pattern_file(dir.path() / "f.bin", 1 * kMiB);
  mi::weights::FileReader reader(path, true);
  StreamSet ss;
  PinnedRing ring(ss.copy(), 2, 512 * 1024);
  mi::DeviceBuffer dev(1 * kMiB);
  MI_CUDA_CHECK(cudaMemset(dev.get(), 0, dev.size()));
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  mi::testing::launch_spin(50'000'000, ss.copy());  // keeps the copies below queued
  std::vector<PinnedRing::Piece> pieces = {{0, 4096, dev.get()}, {4096, 4096, nullptr}};
  EXPECT_THROW(ring.copy_pieces(reader, pieces), std::runtime_error);
  cudaGetLastError();
  ring.synchronize();  // must wait for the first piece's copy
  EXPECT_TRUE(matches_file(dev.get(), 0, 4096));
}

TEST(PinnedRing, StatsSnapshotIsThreadSafeToRead) {
  TempDir dir("stats");
  const fs::path path = write_pattern_file(dir.path() / "f.bin", 2 * kMiB);
  mi::weights::FileReader reader(path, true);
  StreamSet ss;
  PinnedRing ring(ss.copy(), 2, 256 * 1024);
  mi::DeviceBuffer dev(2 * kMiB);
  std::atomic<bool> stop{false};
  std::uint64_t last = 0;
  std::thread reader_thread([&] {
    while (!stop) {
      const std::uint64_t b = ring.stats().bytes;
      EXPECT_GE(b, last);
      last = b;
    }
  });
  ring.copy_range(reader, 0, 2 * kMiB, dev.get());
  stop = true;
  reader_thread.join();
  ring.synchronize();
  EXPECT_EQ(ring.stats().bytes, 2 * kMiB);
}

TEST(PinnedRing, CopyOverlapsComputeStream) {
  TempDir dir("overlap");
  const std::size_t n = 256 * kMiB;
  const fs::path path = write_pattern_file(dir.path() / "f.bin", n);
  mi::weights::FileReader reader(path, false);
  StreamSet ss;
  PinnedRing ring(ss.copy(), 4, 16 * kMiB);
  mi::DeviceBuffer dev(n);

  auto time_copy = [&] {
    auto t0 = Clock::now();
    ring.copy_range(reader, 0, n, dev.get());
    ring.synchronize();
    return seconds_since(t0);
  };
  time_copy();  // warm the page cache
  const double t_copy = time_copy();
  const double t_spin = std::max(0.15, t_copy);
  auto t0 = Clock::now();
  mi::testing::launch_spin(static_cast<unsigned long long>(t_spin * 1e9), ss.compute());
  ring.copy_range(reader, 0, n, dev.get());
  ring.synchronize();
  MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
  const double t_both = seconds_since(t0);
  RecordProperty("copy_s", std::to_string(t_copy));
  RecordProperty("both_s", std::to_string(t_both));
  std::printf("[overlap] copy %.3f s, compute %.3f s, concurrent %.3f s (serial would be %.3f s)\n", t_copy, t_spin, t_both,
              t_copy + t_spin);
  if (mi::testing::kAssertTimings) {
    EXPECT_LT(t_both, t_copy + t_spin - 0.5 * std::min(t_copy, t_spin));
  }
  EXPECT_TRUE(matches_file(dev.get(), 0, 1 * kMiB));
}

TEST(PinnedRing, MeasuredPinnedH2DBandwidth) {
  const std::size_t chunk = 256 * kMiB;
  void* host = nullptr;
  MI_CUDA_CHECK(cudaHostAlloc(&host, chunk, cudaHostAllocDefault));
  std::memset(host, 1, chunk);
  mi::DeviceBuffer dev(chunk);
  cudaStream_t s;
  MI_CUDA_CHECK(cudaStreamCreate(&s));
  cudaEvent_t a, b;
  MI_CUDA_CHECK(cudaEventCreate(&a));
  MI_CUDA_CHECK(cudaEventCreate(&b));
  MI_CUDA_CHECK(cudaMemcpyAsync(dev.get(), host, chunk, cudaMemcpyHostToDevice, s));  // warm-up
  MI_CUDA_CHECK(cudaEventRecord(a, s));
  constexpr int kReps = 8;
  for (int i = 0; i < kReps; ++i) MI_CUDA_CHECK(cudaMemcpyAsync(dev.get(), host, chunk, cudaMemcpyHostToDevice, s));
  MI_CUDA_CHECK(cudaEventRecord(b, s));
  MI_CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  const double gbps = static_cast<double>(chunk) * kReps / (ms * 1e-3) / 1e9;
  std::printf("[bandwidth] pinned H2D: %.2f GB/s\n", gbps);
  RecordProperty("pinned_h2d_GBps", std::to_string(gbps));
  EXPECT_GT(gbps, 1.0);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaStreamDestroy(s);
  cudaFreeHost(host);
}

// ---- GraphCache -------------------------------------------------------------------------------

TEST(GraphCache, CaptureReplayAndLruEviction) {
  StreamSet ss;
  mi::DeviceBuffer buf(4 * sizeof(int));
  int* d = static_cast<int*>(buf.get());
  int init[4] = {0, 0, 0, 0};
  buf.upload(init, sizeof init);
  GraphCache cache(2);
  int captures = 0;
  auto body_for = [&](int slot, int delta) {
    return [&, slot, delta](cudaStream_t s) {
      ++captures;
      mi::testing::launch_add(d + slot, 1, delta, s);
    };
  };
  auto read = [&](int slot) {
    MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
    int out[4];
    buf.download(out, sizeof out);
    return out[slot];
  };

  EXPECT_FALSE(cache.launch(1, ss.compute(), body_for(0, 1)));   // miss: capture + first run
  EXPECT_EQ(read(0), 1);
  EXPECT_TRUE(cache.launch(1, ss.compute(), body_for(0, 100)));  // hit: replays the *captured* delta, body not run
  EXPECT_TRUE(cache.launch(1, ss.compute(), body_for(0, 100)));
  EXPECT_EQ(read(0), 3);
  EXPECT_EQ(captures, 1);

  EXPECT_FALSE(cache.launch(2, ss.compute(), body_for(1, 10)));
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_TRUE(cache.launch(1, ss.compute(), body_for(0, 0)));  // touch key 1: key 2 becomes LRU
  EXPECT_FALSE(cache.launch(3, ss.compute(), body_for(2, 5))); // evicts key 2
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_TRUE(cache.contains(1));
  EXPECT_FALSE(cache.contains(2));
  EXPECT_TRUE(cache.contains(3));
  EXPECT_EQ(cache.stats().evictions, 1u);
  EXPECT_EQ(cache.stats().hits, 3u);
  EXPECT_EQ(cache.stats().misses, 3u);
  EXPECT_FALSE(cache.launch(2, ss.compute(), body_for(1, 10)));  // re-captured after eviction; LRU is now key 1
  EXPECT_FALSE(cache.contains(1));
  EXPECT_TRUE(cache.contains(3));
  EXPECT_EQ(read(1), 20);
  EXPECT_EQ(read(2), 5);
}

TEST(GraphCache, FailedCaptureCachesNothingAndStreamRecovers) {
  StreamSet ss;
  mi::DeviceBuffer buf(sizeof(int));
  int* d = static_cast<int*>(buf.get());
  int zero = 0;
  buf.upload(&zero, sizeof zero);
  GraphCache cache(2);
  EXPECT_THROW(cache.launch(7, ss.compute(), [&](cudaStream_t s) {
    mi::testing::launch_add(d, 1, 1, s);
    throw std::runtime_error("boom");
  }), std::runtime_error);
  EXPECT_FALSE(cache.contains(7));
  EXPECT_EQ(cache.size(), 0u);
  // Stream is usable (no capture left open) and a retry works.
  EXPECT_FALSE(cache.launch(7, ss.compute(), [&](cudaStream_t s) { mi::testing::launch_add(d, 1, 5, s); }));
  MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
  int out = 0;
  buf.download(&out, sizeof out);
  EXPECT_EQ(out, 5);
  EXPECT_THROW(GraphCache(0), std::invalid_argument);
}

}  // namespace
