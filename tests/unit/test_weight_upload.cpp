#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "core/cuda_check.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "runtime/stream_set.h"
#include "test_util.h"
#include "util/xxh64.h"
#include "weights/upload.h"

namespace {

using namespace mi::weights;
using mi::runtime::DeviceArena;
using mi::runtime::PinnedRing;
using mi::runtime::StreamSet;
using mi::testing::TempDir;
using mi::testing::TensorSpec;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kMiB = std::size_t{1} << 20;

std::vector<unsigned char> download(const void* dev, std::size_t n) {
  std::vector<unsigned char> h(n);
  MI_CUDA_CHECK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}

// Builds a one-group index over a synthetic safetensors file; the group lists tensors in a
// scrambled order to prove placement does not depend on file order.
struct Synthetic {
  Synthetic() : dir("upload") {
    std::vector<TensorSpec> specs = {{"a.bias", DType::kBF16, {3}},        {"b.weight", DType::kBF16, {37, 53}},
                                     {"c.scalar", DType::kI64, {}},        {"d.bytes", DType::kU8, {4097}},
                                     {"e.f32", DType::kF32, {5, 7, 3}},    {"f.big", DType::kBF16, {300, 1001}},
                                     {"g.empty", DType::kF32, {0, 4}},     {"h.i32", DType::kI32, {11}}};
    mi::testing::write_safetensors(dir.path() / "m.safetensors", specs);
    SafetensorsFile f(dir.path() / "m.safetensors");
    index.bundle_dir = dir.path();
    index.files.push_back({"m.safetensors", f.file_size(), 0});
    group.component = Component::kVae;
    group.name = "synthetic";
    for (const auto& t : f.tensors()) {
      group.tensors.insert(group.tensors.begin() + static_cast<std::ptrdiff_t>(group.tensors.size() / 2),
                           IndexedTensor{t.name, t.dtype, t.shape, 0, t.offset, t.nbytes});
      group.total_bytes += t.nbytes;
    }
    // XXH64 over the bytes in listed order, from an independent whole-file read.
    MappedFile map(dir.path() / "m.safetensors");
    mi::Xxh64 h;
    for (const auto& t : group.tensors) h.update(map.data() + t.offset, t.nbytes);
    group.checksum = h.digest();
    index.groups.push_back(group);
  }
  void expect_bytes(const TensorMap& m, bool on_device) const {
    MappedFile map(dir.path() / "m.safetensors");
    ASSERT_EQ(m.size(), group.tensors.size());
    for (const auto& t : group.tensors) {
      const mi::Tensor& v = m.at(t.name);
      EXPECT_EQ(v.device, on_device ? mi::Device::kCUDA : mi::Device::kCPU) << t.name;
      EXPECT_EQ(v.nbytes(), t.nbytes) << t.name;
      ASSERT_EQ(static_cast<std::size_t>(v.rank), t.shape.size()) << t.name;
      for (int i = 0; i < v.rank; ++i) EXPECT_EQ(v.dim(i), t.shape[static_cast<std::size_t>(i)]) << t.name;
      if (t.nbytes == 0) continue;
      std::vector<unsigned char> got = on_device ? download(v.data, t.nbytes)
                                                 : std::vector<unsigned char>(v.ptr<unsigned char>(), v.ptr<unsigned char>() + t.nbytes);
      EXPECT_EQ(std::memcmp(got.data(), map.data() + t.offset, t.nbytes), 0) << t.name;
      if (on_device) {
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(v.data) % mi::runtime::kArenaAlign, 0u) << t.name;
      }
    }
  }
  TempDir dir;
  ModelIndex index;
  WeightGroup group;
};

TEST(WeightUploadSynthetic, DtypeMapping) {
  EXPECT_EQ(to_engine_dtype(DType::kBF16), mi::DType::kBF16);
  EXPECT_EQ(to_engine_dtype(DType::kI64), mi::DType::kI64);
  EXPECT_EQ(to_engine_dtype(DType::kF8E4M3), mi::DType::kF8E4M3);
  EXPECT_THROW(to_engine_dtype(DType::kF64), UploadError);
  EXPECT_THROW(to_engine_dtype(DType::kBool), UploadError);
}

TEST(WeightUploadSynthetic, RingUploadIntoArena) {
  for (bool direct : {false, true}) {
    Synthetic s;
    ReaderCache readers(s.index, direct);
    StreamSet ss;
    PinnedRing ring(ss.copy(), 3, 64 * 1024);  // small slots: the 600 KB tensor streams in chunks
    DeviceArena arena(8 * kMiB);
    TensorMap m = upload_group(readers, s.group, arena, ring);
    cudaEvent_t ready = ss.record(ss.copy());
    StreamSet::wait_event(ss.compute(), ready);
    ss.events().release(ready);
    MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
    s.expect_bytes(m, true);
    EXPECT_EQ(arena.persistent_used(), group_device_bytes(s.group));
    EXPECT_EQ(m.at("b.weight").dtype, mi::DType::kBF16);
    EXPECT_EQ(m.at("c.scalar").rank, 0);
  }
}

TEST(WeightUploadSynthetic, SyncUploadAndArenaTooSmall) {
  Synthetic s;
  ReaderCache readers(s.index, true);
  mi::DeviceBuffer buf(group_device_bytes(s.group));
  TensorMap m = upload_group_sync(readers, s.group, buf.get(), buf.size());
  s.expect_bytes(m, true);
  EXPECT_THROW(upload_group_sync(readers, s.group, buf.get(), buf.size() - 256), UploadError);
  StreamSet ss;
  PinnedRing ring(ss.copy(), 2, 64 * 1024);
  DeviceArena small(64 * 1024);
  EXPECT_THROW(upload_group(readers, s.group, small, ring), mi::runtime::ArenaOverflow);
}

// upload_group_sync must leave the bytes visible to non-blocking streams (which the legacy default
// stream does not order against) the moment it returns.
TEST(WeightUploadSynthetic, SyncUploadVisibleToNonBlockingStream) {
  Synthetic s;
  ReaderCache readers(s.index, true);
  mi::DeviceBuffer buf(group_device_bytes(s.group));
  TensorMap m = upload_group_sync(readers, s.group, buf.get(), buf.size());
  const mi::Tensor& big = m.at("f.big");
  StreamSet ss;
  std::vector<unsigned char> via_stream(big.nbytes());
  MI_CUDA_CHECK(cudaHostRegister(via_stream.data(), via_stream.size(), cudaHostRegisterDefault));
  MI_CUDA_CHECK(cudaMemcpyAsync(via_stream.data(), big.data, via_stream.size(), cudaMemcpyDeviceToHost, ss.compute()));
  MI_CUDA_CHECK(cudaStreamSynchronize(ss.compute()));
  MI_CUDA_CHECK(cudaHostUnregister(via_stream.data()));
  EXPECT_EQ(via_stream, download(big.data, via_stream.size()));
  s.expect_bytes(m, true);
}

TEST(WeightUploadSynthetic, HostLoadPlainAndPinned) {
  for (HostMemory mem : {HostMemory::kPlain, HostMemory::kPinned}) {
    Synthetic s;
    ReaderCache readers(s.index, true);
    HostGroup g = load_group_to_host(readers, s.group, mem);
    s.expect_bytes(g.tensors, false);
    if (mem == HostMemory::kPinned) {
      cudaPointerAttributes attr{};
      MI_CUDA_CHECK(cudaPointerGetAttributes(&attr, g.storage.get()));
      EXPECT_EQ(attr.type, cudaMemoryTypeHost);
    }
  }
}

// A soft unload's host copy (pipeline.h release_device_memory): the files read whole into pinned or plain memory serve
// the same uploads, the pinned copy by DMA straight from it (PinnedRing::copy_pieces), the plain one through the slots.
TEST(WeightUploadSynthetic, UploadFromHostFiles) {
  for (const bool pinned : {false, true}) {
    SCOPED_TRACE(pinned);
    Synthetic s;
    const HostFiles hf = read_files_to_host({s.index.file_path(0)}, pinned, false);
    ReaderCache readers(s.index, false, &hf);
    ASSERT_NE(readers.get(0).memory(), nullptr);
    EXPECT_EQ(readers.get(0).pinned(), pinned);
    StreamSet ss;
    PinnedRing ring(ss.copy(), 3, 64 * 1024);
    DeviceArena arena(8 * kMiB);
    const mi::runtime::RingStats before = ring.stats();
    TensorMap m = upload_group(readers, s.group, arena, ring);
    ring.synchronize();  // also covers the slot-less copies of a pinned copy
    s.expect_bytes(m, true);
    EXPECT_EQ(ring.stats().bytes - before.bytes, s.group.total_bytes);
    HostGroup g = load_group_to_host(readers, s.group, HostMemory::kPlain);
    s.expect_bytes(g.tensors, false);
  }
}

TEST(WeightUploadSynthetic, VerifyGroup) {
  Synthetic s;
  EXPECT_TRUE(verify_group(s.index, s.group));
  WeightGroup none = s.group;
  none.checksum.reset();
  EXPECT_FALSE(verify_group(s.index, none));
  WeightGroup bad = s.group;
  bad.checksum = *bad.checksum ^ 1;
  EXPECT_THROW(verify_group(s.index, bad), ChecksumError);
}

// ---- Real bundle ------------------------------------------------------------------------------

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}

class WeightUploadReal : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!fs::is_regular_file(bundle_dir() / "moss.mossidx")) GTEST_SKIP() << "weights not present at " << bundle_dir();
    index = load_index(bundle_dir() / "moss.mossidx", bundle_dir());
  }
  // Byte-compares every tensor of the device map with the file contents (independent mmap path).
  void expect_device_group_equals_files(const WeightGroup& g, const TensorMap& m) {
    for (const auto& t : g.tensors) {
      MappedFile map(index.file_path(t.file));
      auto got = download(m.at(t.name).data, t.nbytes);
      ASSERT_EQ(std::memcmp(got.data(), map.data() + t.offset, t.nbytes), 0) << g.name << "/" << t.name;
    }
  }
  ModelIndex index;
};

TEST_F(WeightUploadReal, VaeDecoderGroupRingAndSync) {
  const WeightGroup* g = index.find_group(Component::kVae, "decoder");
  ASSERT_NE(g, nullptr);
  ReaderCache readers(index, true);
  StreamSet ss;
  PinnedRing ring(ss.copy());
  DeviceArena arena(256 * kMiB);
  auto t0 = Clock::now();
  TensorMap m = upload_group(readers, *g, arena, ring);
  ring.synchronize();
  const double s = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("[upload] VAE decoder %zu tensors, %.1f MiB in %.3f s (%.2f GB/s), ring chunks %llu stalls %llu\n",
              g->tensors.size(), static_cast<double>(g->total_bytes) / kMiB, s, static_cast<double>(g->total_bytes) / s / 1e9,
              static_cast<unsigned long long>(ring.stats().chunks), static_cast<unsigned long long>(ring.stats().stalls));
  EXPECT_EQ(m.size(), 143u);
  expect_device_group_equals_files(*g, m);
  // known shapes: dtype/shape come from the index
  EXPECT_EQ(m.at("bn.num_batches_tracked").dtype, mi::DType::kI64);
  EXPECT_EQ(m.at("bn.running_mean").dim(0), 128);

  mi::DeviceBuffer sync_buf(group_device_bytes(*g));
  TensorMap m2 = upload_group_sync(readers, *g, sync_buf.get(), sync_buf.size());
  expect_device_group_equals_files(*g, m2);
}

TEST_F(WeightUploadReal, DitBlocksEndToEndDiskToGpuBandwidth) {
  constexpr int kBlocks = 5;
  ReaderCache readers(index, true);
  StreamSet ss;
  PinnedRing ring(ss.copy());  // default: 4 x 64 MiB
  std::size_t total = 0;
  std::vector<const WeightGroup*> groups;
  for (int i = 0; i < kBlocks; ++i) {
    groups.push_back(index.find_group(Component::kTransformer, "block." + std::to_string(i)));
    ASSERT_NE(groups.back(), nullptr);
    total += group_device_bytes(*groups.back());
  }
  DeviceArena arena(total + kMiB);
  std::vector<TensorMap> maps;
  auto t0 = Clock::now();
  std::uint64_t payload = 0;
  for (const WeightGroup* g : groups) {
    maps.push_back(upload_group(readers, *g, arena, ring));
    payload += g->total_bytes;
  }
  ring.synchronize();
  const double s = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("[bandwidth] disk->GPU (direct I/O, ring %zux%zu MiB): %.1f MiB in %.3f s = %.2f GB/s; chunks %llu stalls %llu\n",
              ring.slots(), ring.slot_bytes() / kMiB, static_cast<double>(payload) / kMiB, s,
              static_cast<double>(payload) / s / 1e9, static_cast<unsigned long long>(ring.stats().chunks),
              static_cast<unsigned long long>(ring.stats().stalls));
  RecordProperty("disk_to_gpu_GBps", std::to_string(static_cast<double>(payload) / s / 1e9));

  // The disk is slower than PCIe, so the producer should almost never wait for a slot copy.
  EXPECT_LE(ring.stats().stalls, ring.stats().chunks / 8);

  // Block 0 byte-for-byte; the others by their first/last 64 KiB of every tensor (avoids re-reading 1 GiB).
  expect_device_group_equals_files(*groups[0], maps[0]);
  for (int b = 1; b < kBlocks; ++b) {
    for (const auto& t : groups[static_cast<std::size_t>(b)]->tensors) {
      MappedFile map(index.file_path(t.file));
      const std::size_t n = std::min<std::size_t>(t.nbytes, 64 * 1024);
      auto head = download(maps[static_cast<std::size_t>(b)].at(t.name).data, n);
      EXPECT_EQ(std::memcmp(head.data(), map.data() + t.offset, n), 0) << t.name;
      auto tail = download(static_cast<const char*>(maps[static_cast<std::size_t>(b)].at(t.name).data) + (t.nbytes - n), n);
      EXPECT_EQ(std::memcmp(tail.data(), map.data() + t.offset + t.nbytes - n, n), 0) << t.name;
    }
  }
  // Shapes of a DiT block as in the index tests: 10 tensors, 4 q/k/v/o matrices of 3072^2.
  const TensorMap& m0 = maps[0];
  EXPECT_EQ(m0.size(), 10u);
  bool saw_square = false;
  for (const auto& [name, t] : m0) saw_square |= (t.rank == 2 && t.dim(0) == 3072 && t.dim(1) == 3072 && t.dtype == mi::DType::kBF16);
  EXPECT_TRUE(saw_square);
}

TEST_F(WeightUploadReal, EmbeddingTableHostResident) {
  const WeightGroup* g = index.find_group(Component::kTextEncoder, "embed_tokens");
  ASSERT_NE(g, nullptr);
  ReaderCache readers(index, true);
  auto t0 = Clock::now();
  HostGroup h = load_group_to_host(readers, *g, HostMemory::kPlain);
  const double s = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("[host] embed_tokens %.2f GB -> host in %.3f s (%.2f GB/s)\n", static_cast<double>(g->total_bytes) / 1e9, s,
              static_cast<double>(g->total_bytes) / s / 1e9);
  const mi::Tensor& e = h.tensors.at("language_model.embed_tokens.weight");
  EXPECT_EQ(e.device, mi::Device::kCPU);
  EXPECT_EQ(e.dtype, mi::DType::kBF16);
  ASSERT_EQ(e.rank, 2);
  EXPECT_EQ(e.dim(0), 248320);
  EXPECT_EQ(e.dim(1), 4096);
  const IndexedTensor& it = g->tensors[0];
  MappedFile map(index.file_path(it.file));
  for (std::int64_t row : {0L, 1L, 4095L, 123456L, 248319L}) {
    const std::size_t off = static_cast<std::size_t>(row) * 4096 * 2;
    EXPECT_EQ(std::memcmp(e.ptr<std::byte>() + off, map.data() + it.offset + off, 4096 * 2), 0) << "row " << row;
  }
  // Rows can be gathered straight from the host view.
  EXPECT_EQ(std::memcmp(e.ptr<std::byte>() + 7 * 8192, map.data() + it.offset + 7 * 8192, 8192), 0);
}

}  // namespace
