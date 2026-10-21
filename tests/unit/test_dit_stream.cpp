// P4 DiT block streaming (models/moss_dit/dit_weights.h): streamed forwards are bit-identical to the resident model
// on the golden step-0 cases for several resident-block counts, buffer counts and sources; streaming memory matches
// its queries; cancel lands within one block and leaves the model usable; a failing copy path (reader error) fails
// the forward cleanly and the next forward is exact again.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "models/moss_dit/dit_weights.h"
#include "models/moss_dit/moss_dit.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;
using mi::moss_dit::MossDit;
using mi::moss_dit::Residency;
using mi::moss_dit::Row;
using mi::moss_dit::SkipSet;
using mi::moss_dit::StreamSource;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path golden_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "golden"; }

std::uint16_t to_bf16(float v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}

struct DevTensor {
  mi::DeviceBuffer buf;
  Tensor t;
  DevTensor(mi::DType dt, std::initializer_list<std::int64_t> shape) {
    t = Tensor::view(nullptr, dt, mi::Device::kCUDA, shape);
    buf = mi::DeviceBuffer(std::max<std::size_t>(t.nbytes(), 1));
    t.data = buf.get();
  }
};

std::vector<float> download_f32(const Tensor& t) {
  std::vector<float> h(static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, h.size() * 4, cudaMemcpyDeviceToHost));
  return h;
}

std::size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return n;
}

SkipSet read_skip(const fs::path& p) {
  std::ifstream in(p);
  std::stringstream ss;
  ss << in.rdbuf();
  SkipSet s;
  const mi::json::Value doc = mi::json::parse(ss.str());
  for (const auto& v : doc.as_array()) s.set(static_cast<std::size_t>(v.as_int()));
  return s;
}

// One golden case's inputs on the host (refined text is computed by each model: the text refiner is resident in
// every residency, so it is part of what must match).
struct HostCase {
  std::string id;
  std::vector<float> noise, te_cond, te_uncond;
  std::int64_t h = 0, w = 0, s_cond = 0, s_uncond = 0;
  float timestep = 0;
  SkipSet uncond_skip;
};

HostCase load_case(const std::string& id) {
  HostCase c;
  c.id = id;
  const fs::path g = golden_dir();
  const auto noise = mi::npy::load(g / id / "initial_noise.npy");
  c.h = noise.shape[2];
  c.w = noise.shape[3];
  c.noise = noise.to_vector<float>();
  c.timestep = static_cast<float>(mi::npy::load(g / id / "timestep0.npy").to_vector<double>().at(0));
  const auto te_c = mi::npy::load(g / (id + ".cond.te_last_hidden.npy"));
  const auto te_u = mi::npy::load(g / (id + ".uncond.te_last_hidden.npy"));
  c.s_cond = te_c.shape[0];
  c.s_uncond = te_u.shape[0];
  c.te_cond = te_c.to_vector<float>();
  c.te_uncond = te_u.to_vector<float>();
  c.uncond_skip = read_skip(g / id / "dit_step0_uncond.skip.json");
  return c;
}

// A model with one residency in its own arena, sized exactly from the queries for the cases it runs.
struct Model {
  std::unique_ptr<mi::runtime::DeviceArena> arena;
  std::unique_ptr<MossDit> dit;
  std::size_t persistent = 0, workspace = 0;
};

struct StepOut {
  std::vector<float> cond, uncond;
};

class DitStreamTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::is_regular_file(bundle_dir() / "moss.mossidx") || !fs::is_directory(golden_dir() / "g_layout_1024")) return;
    index_ = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir()));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
  }
  static void TearDownTestSuite() {
    streams_.reset();
    index_.reset();
  }
  void SetUp() override {
    if (!index_) GTEST_SKIP() << "weights or goldens not present (" << bundle_dir() << ")";
  }
  cudaStream_t stream() const { return streams_->compute(); }

  static std::size_t case_workspace(const HostCase& c) {
    const std::int64_t lengths[] = {c.s_cond, c.s_uncond};
    return std::max(MossDit::workspace_bytes(c.h, c.w, lengths), MossDit::encode_text_workspace_bytes(std::max(c.s_cond, c.s_uncond)));
  }

  Model make_model(const Residency& res, std::size_t workspace) {
    Model m;
    m.persistent = MossDit::persistent_bytes(*index_, res);
    m.workspace = workspace + MossDit::stream_workspace_bytes(*index_, res);
    m.arena = std::make_unique<mi::runtime::DeviceArena>(m.persistent + m.workspace);
    m.dit = std::make_unique<MossDit>(*index_, *m.arena, *streams_, false, nullptr, res);
    EXPECT_EQ(m.arena->persistent_used(), m.persistent);
    return m;
  }

  // Refined rows + step-0 forward of `c` on `m`; outputs downloaded. Leaves the refined text in `keep` when given.
  StepOut run(Model& m, const HostCase& c, const mi::moss_dit::BlockHook& hook = {}) {
    const auto refine = [&](const std::vector<float>& te, std::int64_t s) {
      DevTensor in(mi::DType::kBF16, {s, mi::moss_dit::kTextDim});
      std::vector<std::uint16_t> h(te.size());
      for (std::size_t i = 0; i < te.size(); ++i) h[i] = to_bf16(te[i]);
      in.buf.upload(h.data(), h.size() * 2);
      auto out = std::make_unique<DevTensor>(mi::DType::kBF16, std::initializer_list<std::int64_t>{s, mi::moss_dit::kDim});
      m.dit->encode_text(in.t, out->t, stream());
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      return out;
    };
    auto cond = refine(c.te_cond, c.s_cond);
    auto uncond = refine(c.te_uncond, c.s_uncond);
    DevTensor latents(mi::DType::kF32, {128, c.h, c.w});
    latents.buf.upload(c.noise.data(), c.noise.size() * 4);
    DevTensor oc(mi::DType::kF32, {128, c.h, c.w}), ou(mi::DType::kF32, {128, c.h, c.w});
    const Row rows[] = {{cond->t, SkipSet{}}, {uncond->t, c.uncond_skip}};
    const Tensor outs[] = {oc.t, ou.t};
    m.dit->forward(latents.t, c.timestep, rows, outs, stream(), hook);
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    return {download_f32(oc.t), download_f32(ou.t)};
  }

  static std::unique_ptr<mi::weights::ModelIndex> index_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
};

std::unique_ptr<mi::weights::ModelIndex> DitStreamTest::index_;
std::unique_ptr<mi::runtime::StreamSet> DitStreamTest::streams_;

TEST(DitResidency, ResidentMaskSpreadsBlocks) {
  for (int r = 0; r <= mi::moss_dit::kNumBlocks; ++r) {
    const auto m = mi::moss_dit::resident_mask(r);
    EXPECT_EQ(static_cast<int>(m.count()), r);
    if (r > 0) {
      EXPECT_TRUE(m.test(0)) << r;  // copies start during the first block's compute
    }
    // Evenly spread: each resident block starts a run of at most streamed / resident + 1 streamed blocks.
    int run = 0, longest = 0;
    for (int i = 0; i < mi::moss_dit::kNumBlocks; ++i) {
      run = m.test(static_cast<std::size_t>(i)) ? 0 : run + 1;
      longest = std::max(longest, run);
    }
    const int streamed = mi::moss_dit::kNumBlocks - r;
    if (r > 0) {
      EXPECT_LE(longest, streamed / r + 1) << r;
    }
  }
}

TEST_F(DitStreamTest, InvalidResidencyIsRejected) {
  Residency bad;
  bad.resident_blocks = 43;
  EXPECT_THROW(MossDit::persistent_bytes(*index_, bad), std::invalid_argument);
  bad.resident_blocks = 10;
  bad.buffers = 1;
  EXPECT_THROW(MossDit::stream_workspace_bytes(*index_, bad), std::invalid_argument);
  bad.buffers = 2;
  bad.reader_threads = 0;
  EXPECT_THROW(mi::moss_dit::BlockWeights::host_bytes(*index_, bad), std::invalid_argument);
  // Fully resident: no streaming buffers, whatever the (unused) streaming options say.
  Residency all;
  all.buffers = 1;
  EXPECT_EQ(MossDit::stream_workspace_bytes(*index_, all), 0u);
  EXPECT_EQ(MossDit::persistent_bytes(*index_, all), MossDit::persistent_bytes(*index_));
}

// The tentpole claim: every residency gives the resident model's bits, on 1024^2 and 1456^2 golden cases (short,
// long 4096-token and skip-22 uncond rows), twice per model (the second pass reuses the buffers and events).
TEST_F(DitStreamTest, StreamedBitIdenticalToResident) {
  const std::vector<HostCase> cases = {load_case("g_layout_1024"), load_case("g_long4096_1024"), load_case("g_short_1456")};
  std::size_t ws = 0;
  for (const HostCase& c : cases) ws = std::max(ws, case_workspace(c));

  std::vector<StepOut> ref;
  {
    Model m = make_model(Residency{}, ws);
    for (const HostCase& c : cases) ref.push_back(run(m, c));
    EXPECT_EQ(m.dit->last_stream_stats().streamed_blocks, 0);
  }

  struct Variant {
    const char* name;
    Residency r;
  };
  std::vector<Variant> variants;
  const auto add = [&](const char* name, int resident, int buffers, StreamSource src, int readers, bool direct) {
    Residency r;
    r.resident_blocks = resident;
    r.buffers = buffers;
    r.source = src;
    r.reader_threads = readers;
    r.direct_io = direct;
    variants.push_back({name, r});
  };
  add("0 resident, 3 buffers, file x4", 0, 3, StreamSource::kFile, 4, false);
  add("0 resident, 2 buffers, file x1", 0, 2, StreamSource::kFile, 1, false);
  add("1 resident, 2 buffers, file x3", 1, 2, StreamSource::kFile, 3, false);
  add("21 resident, 3 buffers, file x4, O_DIRECT", 21, 3, StreamSource::kFile, 4, true);
  add("36 resident, 4 buffers, file x2", 36, 4, StreamSource::kFile, 2, false);
  add("0 resident, 3 buffers, host pinned", 0, 3, StreamSource::kHostPinned, 1, false);

  const std::size_t block = mi::moss_dit::BlockWeights::block_device_bytes(*index_);
  for (const Variant& v : variants) {
    Model m = make_model(v.r, ws);
    const int streamed = mi::moss_dit::kNumBlocks - v.r.resident_blocks;
    EXPECT_EQ(m.persistent, MossDit::persistent_bytes(*index_) - static_cast<std::size_t>(streamed) * block) << v.name;
    EXPECT_EQ(m.dit->host_bytes(), mi::moss_dit::BlockWeights::host_bytes(*index_, v.r)) << v.name;
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t i = 0; i < cases.size(); ++i) {
        const HostCase& c = cases[i];
        m.arena->reset_high_water();
        const StepOut got = run(m, c);
        EXPECT_EQ(mismatches(got.cond, ref[i].cond), 0u) << v.name << " " << c.id << " cond";
        EXPECT_EQ(mismatches(got.uncond, ref[i].uncond), 0u) << v.name << " " << c.id << " uncond";
        // The forward's stack: activations + the block buffers, exactly as queried.
        const std::int64_t lengths[] = {c.s_cond, c.s_uncond};
        EXPECT_LE(m.arena->high_water(), m.persistent + MossDit::workspace_bytes(c.h, c.w, lengths) +
                                             MossDit::stream_workspace_bytes(*index_, v.r))
            << v.name;
        const mi::moss_dit::StreamStats& s = m.dit->last_stream_stats();
        EXPECT_EQ(s.streamed_blocks, streamed);
        EXPECT_GT(s.bytes, static_cast<std::uint64_t>(streamed) * (block - 4096));
        EXPECT_LE(s.bytes, static_cast<std::uint64_t>(streamed) * block);
        EXPECT_GE(s.hidden(), 0.0);
        EXPECT_LE(s.hidden(), 1.0);
        if (pass == 1) {
          std::printf("[dit stream] %-42s %-16s: %.2f GB in %.3f s copy (%.1f GB/s), forward %.3f s, stall %.3f s "
                      "(%d blocks), %.1f%% hidden\n",
                      v.name, c.id.c_str(), static_cast<double>(s.bytes) / 1e9, s.copy_s, s.gbps(), s.forward_s,
                      s.stall_s, s.stalls, 100.0 * s.hidden());
        }
      }
    }
  }
}

// AC12 with streaming: a cancel seen between blocks stops the forward before the next block (no block issued after
// the flag), waits for the copy streams, releases the stack, and the next forward is bit-identical.
TEST_F(DitStreamTest, CancelWhileStreaming) {
  const HostCase c = load_case("g_layout_1024");
  Residency r;
  r.resident_blocks = 0;
  Model m = make_model(r, case_workspace(c));
  const StepOut ref = run(m, c);

  std::atomic<bool> cancel{false};
  m.dit->set_cancel(&cancel);
  for (const int at : {0, 10, 40}) {
    cancel.store(false);
    int last = -1;
    const std::size_t used = m.arena->used();
    EXPECT_THROW(run(m, c, [&](int block, const Tensor&) {
                   last = block;
                   if (block == at) cancel.store(true);
                 }),
                 mi::moss_dit::Cancelled);
    EXPECT_EQ(last, at);
    EXPECT_EQ(m.arena->used(), used);
    EXPECT_EQ(cudaStreamQuery(stream()), cudaSuccess);
    cancel.store(false);
    const StepOut again = run(m, c);
    EXPECT_EQ(mismatches(again.cond, ref.cond), 0u) << "after a cancel at block " << at;
    EXPECT_EQ(mismatches(again.uncond, ref.uncond), 0u) << "after a cancel at block " << at;
  }
  m.dit->set_cancel(nullptr);
}

// A reader failing on the copy path (an I/O error, a full pinned allocation) fails the forward with its error on
// the calling thread; the readers are joined, the copy streams drained and the stack released, and the model stays
// usable and exact. Both sources, an early and the last streamed block, and several readers.
TEST_F(DitStreamTest, CopyFailureFailsTheForwardCleanly) {
  const HostCase c = load_case("g_layout_1024");
  for (const StreamSource src : {StreamSource::kFile, StreamSource::kHostPinned}) {
    Residency r;
    r.resident_blocks = 6;
    r.source = src;
    Model m = make_model(r, case_workspace(c));
    const StepOut ref = run(m, c);
    const int last_streamed = [&] {
      const auto mask = mi::moss_dit::resident_mask(r.resident_blocks);
      for (int b = mi::moss_dit::kNumBlocks - 1; b >= 0; --b) {
        if (!mask.test(static_cast<std::size_t>(b))) return b;
      }
      return -1;
    }();
    for (const int fail_block : {1, last_streamed}) {
      const int fail_reader = src == StreamSource::kFile ? 2 : 0;
      m.dit->set_copy_hook([&](int reader, int block) {
        if (reader == fail_reader && block == fail_block) throw std::runtime_error("injected copy failure");
      });
      const std::size_t used = m.arena->used();
      try {
        run(m, c);
        ADD_FAILURE() << "the forward did not fail (" << mi::moss_dit::stream_source_name(src) << ", block " << fail_block << ")";
      } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "injected copy failure");
      }
      EXPECT_EQ(m.arena->used(), used);
      EXPECT_EQ(cudaStreamQuery(stream()), cudaSuccess);
      m.dit->set_copy_hook({});
      const StepOut again = run(m, c);
      EXPECT_EQ(mismatches(again.cond, ref.cond), 0u) << mi::moss_dit::stream_source_name(src) << " block " << fail_block;
      EXPECT_EQ(mismatches(again.uncond, ref.uncond), 0u) << mi::moss_dit::stream_source_name(src) << " block " << fail_block;
    }
  }
}

// Delays injected on both sides of the buffer handshake: slow readers (each chunk delayed, so the compute stream waits
// for copies) and a slow GPU (a host function sleeping on the compute stream after random blocks, so readers wait for
// buffers to be released and their copies for the compute-done events). With 2 and 3 buffers, both sources, the output
// stays bit-identical to the undelayed forward.
TEST_F(DitStreamTest, DelayedCopiesAndComputeStayExact) {
  const HostCase c = load_case("g_layout_1024");
  for (const StreamSource src : {StreamSource::kFile, StreamSource::kHostPinned}) {
    for (const int buffers : {2, 3}) {
      Residency r;
      // Every block streamed from the files (4 readers); 12 from the pinned copy (one copy stream, 9.5 GB of pinned
      // host memory otherwise).
      r.resident_blocks = src == StreamSource::kFile ? 0 : 30;
      r.buffers = buffers;
      r.source = src;
      Model m = make_model(r, case_workspace(c));
      const StepOut ref = run(m, c);
      std::atomic<std::uint64_t> seed{1};
      const auto jitter = [&](std::uint64_t salt, int max_us) {
        std::uint64_t z = (seed.fetch_add(1) + salt) * 0x9E3779B97F4A7C15ull;
        z ^= z >> 31;
        return std::chrono::microseconds(static_cast<int>(z % static_cast<std::uint64_t>(max_us)));
      };
      for (const bool slow_copies : {true, false}) {
        if (slow_copies) {
          // ~55 ms per block and reader on average: slower than a 1024^2 block's compute, so the compute stream waits.
          m.dit->set_copy_hook([&](int reader, int) { std::this_thread::sleep_for(jitter(static_cast<std::uint64_t>(reader), 8000)); });
        } else {
          m.dit->set_copy_hook({});
        }
        const StepOut got = run(m, c, [&](int block, const Tensor&) {
          if (block % 3 == 0) return;
          auto* us = new std::chrono::microseconds(jitter(static_cast<std::uint64_t>(block) << 8, 8000));
          MI_CUDA_CHECK(cudaLaunchHostFunc(stream(), [](void* p) {
            auto* d = static_cast<std::chrono::microseconds*>(p);
            std::this_thread::sleep_for(*d);
            delete d;
          }, us));
        });
        const std::string what = std::string(mi::moss_dit::stream_source_name(src)) + ", " + std::to_string(buffers) +
                                 " buffers, " + (slow_copies ? "slow copies + slow compute" : "slow compute");
        EXPECT_EQ(mismatches(got.cond, ref.cond), 0u) << what;
        EXPECT_EQ(mismatches(got.uncond, ref.uncond), 0u) << what;
        const auto& st = m.dit->last_stream_stats();
        std::printf("[dit stream] delayed (%s): %d stalls, %.3f s stalled, forward %.3f s\n", what.c_str(), st.stalls,
                    st.stall_s, st.forward_s);
        if (slow_copies) {
          EXPECT_GT(st.stalls, 0) << what << ": the compute stream never waited for a copy";
        }
      }
      m.dit->set_copy_hook({});
    }
  }
}

// M5: a cancel lands within about one chunk even when the compute thread is blocked waiting for a block from slow
// storage (every chunk delayed by kChunkDelay): the readers see the flag between chunks and stop the pass, and the
// waiting acquire throws Cancelled. The model stays usable and exact.
TEST_F(DitStreamTest, CancelOnSlowStorageLandsWithinAChunk) {
  using Clock = std::chrono::steady_clock;
  const HostCase c = load_case("g_layout_1024");
  Residency r;
  r.resident_blocks = 0;
  Model m = make_model(r, case_workspace(c));
  const StepOut ref = run(m, c);
  constexpr auto kChunkDelay = std::chrono::milliseconds(25);
  // Chunks one reader copies per block (4 MiB chunks of every tensor, spread over the readers).
  const double block_chunks = static_cast<double>(mi::moss_dit::BlockWeights::block_device_bytes(*index_)) /
                              static_cast<double>(r.chunk_bytes) / r.reader_threads;
  const double block_s = block_chunks * std::chrono::duration<double>(kChunkDelay).count();
  std::atomic<bool> cancel{false};
  std::atomic<std::int64_t> flagged_ns{0};
  m.dit->set_cancel(&cancel);
  m.dit->set_copy_hook([&](int reader, int block) {
    std::this_thread::sleep_for(kChunkDelay);
    // While block 4 is being read, the compute thread waits for it (the copies are far slower than its compute).
    if (reader == 0 && block == 4 && !cancel.load()) {
      flagged_ns = Clock::now().time_since_epoch().count();
      cancel.store(true);
    }
  });
  const std::size_t used = m.arena->used();
  EXPECT_THROW(run(m, c), mi::moss_dit::Cancelled);
  const double latency = std::chrono::duration<double>(Clock::now().time_since_epoch() - Clock::duration(flagged_ns.load())).count();
  std::printf("[dit stream] cancel on slow storage (%.0f ms per chunk, ~%.2f s per block per reader): returned %.0f ms "
              "after the flag\n",
              std::chrono::duration<double, std::milli>(kChunkDelay).count(), block_s, latency * 1e3);
  EXPECT_GT(flagged_ns.load(), 0);
  EXPECT_LT(latency, 0.5 * block_s);  // a block per reader takes block_s: the cancel did not wait for it
  EXPECT_EQ(m.arena->used(), used);
  EXPECT_EQ(cudaStreamQuery(stream()), cudaSuccess);
  m.dit->set_copy_hook({});
  cancel.store(false);
  const StepOut again = run(m, c);
  EXPECT_EQ(mismatches(again.cond, ref.cond), 0u);
  EXPECT_EQ(mismatches(again.uncond, ref.uncond), 0u);
  m.dit->set_cancel(nullptr);
}

// L1: buffers beyond the streamed block count are not allocated.
TEST_F(DitStreamTest, BuffersCappedAtStreamedBlocks) {
  Residency r;
  r.resident_blocks = 41;
  r.buffers = 3;
  const std::size_t one = mi::runtime::align_up(mi::moss_dit::BlockWeights::block_device_bytes(*index_));
  EXPECT_EQ(MossDit::stream_workspace_bytes(*index_, r), one);
  r.resident_blocks = 40;
  EXPECT_EQ(MossDit::stream_workspace_bytes(*index_, r), 2 * one);
  const HostCase c = load_case("g_layout_1024");
  r.resident_blocks = 41;
  StepOut ref;
  {
    Model resident = make_model(Residency{}, case_workspace(c));
    ref = run(resident, c);
  }
  Model m = make_model(r, case_workspace(c));
  m.arena->reset_high_water();
  const StepOut got = run(m, c);
  EXPECT_EQ(mismatches(got.cond, ref.cond), 0u);
  EXPECT_EQ(mismatches(got.uncond, ref.uncond), 0u);
  EXPECT_LE(m.arena->high_water(), m.persistent + m.workspace);
}

// Memory: a fully streamed DiT fits a few GiB at 1024^2 (AC8's denoise phase), and its persistent part is the
// non-block weights only.
TEST_F(DitStreamTest, StreamedFootprint) {
  Residency r;
  r.resident_blocks = 0;
  const std::int64_t lengths[] = {282, 10};
  const std::size_t persistent = MossDit::persistent_bytes(*index_, r);
  const std::size_t buffers = MossDit::stream_workspace_bytes(*index_, r);
  const std::size_t act = MossDit::workspace_bytes(64, 64, lengths);
  std::printf("[dit stream] footprint, 0 resident: persistent %.3f GiB, 3 block buffers %.3f GiB, 1024^2 activations "
              "%.3f GiB (all resident: %.3f GiB persistent)\n",
              static_cast<double>(persistent) / kGiB, static_cast<double>(buffers) / kGiB, static_cast<double>(act) / kGiB,
              static_cast<double>(MossDit::persistent_bytes(*index_)) / kGiB);
  EXPECT_LT(static_cast<double>(persistent + buffers + act), 2.5 * kGiB);
  EXPECT_EQ(buffers, 3 * mi::runtime::align_up(mi::moss_dit::BlockWeights::block_device_bytes(*index_)));
}

}  // namespace
