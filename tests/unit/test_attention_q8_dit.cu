// 8-bit attention (src/kernels/attention_q8) on real MOSS DiT activations: per-block accuracy against the exact
// FlashAttention-2 output, the end-to-end effect on the step-0 velocity, and speed against FlashAttention-2, at three
// CFG shapes (cond + uncond rows, uncond skips block 22 as the golden cases):
//   golden_1024:  the golden case g_fridge_1024 (64x64 latents, prompts of 62 and 10 tokens)
//   bench_1024:   AC9b's shape: 64x64 latents, the first 282 tokens of the 4096-token golden prompt + 10 (8484 tokens)
//   long_1456:    91x91 latents, the full 4096-token golden prompt + its 16-token negative prompt (20674 tokens)
//
// Harness (test-only; the production forward is untouched): the engine (MossDit::forward, op by op, every block
// streamed) runs the step, and its block hook replaces the residual stream after every block with that of a test-local
// copy of MossDit::run_block whose attention is pluggable, run on the previous replaced stream. The engine's head
// then turns the last replaced stream into velocities. The input of block 0 comes from a first forward in which every
// row skips block 0 (the hook then sees the joint input unchanged). With FlashAttention-2 plugged in, the local blocks
// must reproduce the engine bit for bit (checked), so the bf16 reference is the engine's own step. At every block of
// that reference pass, each 8-bit variant also runs on the same Q/K/V (after QK RMSNorm and RoPE) for the per-block
// attention metrics; then one pass per variant replaces attention in all 42 blocks for the step-0 velocity.
// Speed: FlashAttention-2 vs each variant on block 21's real Q/K/V, interleaved, in this test's single GPU-lock hold,
// with the SM clock and board power sampled through NVML (dlopen'ed; skipped when absent).
//
// Slow (~80 s for the three cases, up to ~5 GB of device memory): the cases run only with MOSSINFER_SLOW_TESTS=1
// (GTEST_SKIP otherwise), and the speed part only with MI_RUN_PERF=1 as well.
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/attention_q8/attention_q8.h"
#include "kernels/attention_q8/attention_q8_testing.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "models/moss_dit/dit_weights.h"
#include "models/moss_dit/moss_dit.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
namespace md = mi::moss_dit;
namespace q8 = mi::attention_q8;
using mi::DType;
using mi::Tensor;
using q8::Variant;

constexpr std::int64_t kDim = md::kDim, kQkv = 3 * md::kDim, kUpGate = 2 * md::kFfnDim, kBigCols = 3 * md::kFfnDim;
constexpr int kVariantCount = 4;
constexpr Variant kVariants[kVariantCount] = {Variant::kInt8Fp16, Variant::kInt8Fp16F32Acc, Variant::kInt8Fp8,
                                              Variant::kInt8Fp8F32Acc};
// Per-block metrics also cover SageAttention2++ with upstream's own ranges (P 448, V 2.25) next to the default 112 / 4.5.
constexpr q8::Options kUpstreamRanges{Variant::kInt8Fp8, 448.f, 2.25f};
constexpr int kMetricConfigs = kVariantCount + 1;
q8::Options metric_config(int i) { return i < kVariantCount ? q8::Options{kVariants[i]} : kUpstreamRanges; }
const char* metric_name(int i) { return i < kVariantCount ? q8::variant_name(kVariants[i]) : "int8_fp8 P448/V2.25"; }
// Lower bounds on the per-block attention cosine (worst block of a case) and on the step-0 velocity cosine of each
// CFG row. Measured on sm_89 (worst of the three cases, long_1456 for the blocks): per block fp16 PV 0.99929, FP8 PV
// 0.99917; velocity 0.99982. The worst single token (block 41, a text token of golden_1024: near-hard attention whose
// top key INT8 QK can flip) is 0.9388; its floor only catches a broken row.
double block_cos_floor(int config) {
  const bool fp16 = config < kVariantCount && (kVariants[config] == Variant::kInt8Fp16 || kVariants[config] == Variant::kInt8Fp16F32Acc);
  return fp16 ? 0.999 : 0.9985;
}
constexpr double kTokenCosFloor = 0.9;
constexpr double kVelocityCosFloor = 0.9995;
constexpr int kBenchBlock = 21;  // its Q/K/V feed the speed comparison
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
const float kAttnScale = static_cast<float>(1.0 / std::sqrt(128.0));  // SDPA's default, rounded once (as moss_dit)

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path golden_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "golden"; }
bool env_is_1(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "1";
}

std::uint16_t to_bf16(float v) {
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}

Tensor bf16_view(void* p, std::initializer_list<std::int64_t> shape) {
  return Tensor::view(p, DType::kBF16, mi::Device::kCUDA, shape);
}
Tensor vec(const Tensor& mod, std::int64_t i) { return bf16_view(static_cast<std::uint16_t*>(mod.data) + i * kDim, {kDim}); }
// [rows, heads, 128] view of columns [col, col + kDim) of a row-major [rows, ld] bf16 buffer.
Tensor heads(const Tensor& buf, std::int64_t col) {
  Tensor t = Tensor::view(static_cast<std::uint16_t*>(buf.data) + col, DType::kBF16, mi::Device::kCUDA,
                          {buf.dim(0), md::kHeads, md::kHeadDim});
  t.stride[0] = buf.stride[0];
  return t;
}

// ---------------------------------------------------------------- device metrics

struct DevStats {
  double dot, aa, bb, dd;
  unsigned int max_abs_bits, max_ref_bits;
  unsigned long long differ;
};

// a = reference, b = test, both bf16 [n]: sums for cosine / relative L2, max |a - b|, max |a|, differing bit patterns.
__global__ void stats_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, std::int64_t n, DevStats* s) {
  double dot = 0, aa = 0, bb = 0, dd = 0;
  float mx = 0, mr = 0;
  unsigned long long differ = 0;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
    dot += static_cast<double>(x) * y;
    aa += static_cast<double>(x) * x;
    bb += static_cast<double>(y) * y;
    dd += static_cast<double>(x - y) * (x - y);
    mx = fmaxf(mx, fabsf(x - y));
    mr = fmaxf(mr, fabsf(x));
    differ += __bfloat16_as_ushort(a[i]) != __bfloat16_as_ushort(b[i]);
  }
  for (int o = 16; o > 0; o >>= 1) {
    dot += __shfl_xor_sync(0xffffffffu, dot, o);
    aa += __shfl_xor_sync(0xffffffffu, aa, o);
    bb += __shfl_xor_sync(0xffffffffu, bb, o);
    dd += __shfl_xor_sync(0xffffffffu, dd, o);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    mr = fmaxf(mr, __shfl_xor_sync(0xffffffffu, mr, o));
    differ += __shfl_xor_sync(0xffffffffu, differ, o);
  }
  if (threadIdx.x % 32 == 0) {
    atomicAdd(&s->dot, dot);
    atomicAdd(&s->aa, aa);
    atomicAdd(&s->bb, bb);
    atomicAdd(&s->dd, dd);
    atomicMax(&s->max_abs_bits, __float_as_uint(mx));  // non-negative floats order as their bits
    atomicMax(&s->max_ref_bits, __float_as_uint(mr));
    atomicAdd(&s->differ, differ);
  }
}

// Cosine of every row (token) of a [rows, cols] pair; min over rows via the float bits of (1 - cos) clamped >= 0.
__global__ void row_cos_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b, int cols, unsigned int* worst) {
  const std::int64_t row = blockIdx.x;
  float dot = 0, aa = 0, bb = 0;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float x = __bfloat162float(a[row * cols + i]), y = __bfloat162float(b[row * cols + i]);
    dot += x * y;
    aa += x * x;
    bb += y * y;
  }
  __shared__ float red[3][32];
  for (int o = 16; o > 0; o >>= 1) {
    dot += __shfl_xor_sync(0xffffffffu, dot, o);
    aa += __shfl_xor_sync(0xffffffffu, aa, o);
    bb += __shfl_xor_sync(0xffffffffu, bb, o);
  }
  if (threadIdx.x % 32 == 0) {
    red[0][threadIdx.x / 32] = dot;
    red[1][threadIdx.x / 32] = aa;
    red[2][threadIdx.x / 32] = bb;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float d = 0, x = 0, y = 0;
    for (int w = 0; w < static_cast<int>(blockDim.x) / 32; ++w) {
      d += red[0][w];
      x += red[1][w];
      y += red[2][w];
    }
    const float c = (x > 0 && y > 0) ? d / sqrtf(x * y) : 1.f;
    atomicMax(worst, __float_as_uint(fmaxf(1.f - c, 0.f)));
  }
}

struct Stats {
  double cos = 1, rel_l2 = 0, max_abs = 0, max_ref = 0, min_row_cos = 1;
  unsigned long long differ = 0;
};

// Contiguous bf16 [rows, cols] reference vs test.
Stats measure(const void* ref, const void* got, std::int64_t rows, std::int64_t cols, DevStats* d_stats, cudaStream_t s) {
  MI_CUDA_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(DevStats) + sizeof(unsigned int), s));
  const std::int64_t n = rows * cols;
  stats_kernel<<<512, 256, 0, s>>>(static_cast<const __nv_bfloat16*>(ref), static_cast<const __nv_bfloat16*>(got), n, d_stats);
  MI_CUDA_CHECK_LAUNCH();
  auto* worst = reinterpret_cast<unsigned int*>(d_stats + 1);
  row_cos_kernel<<<static_cast<unsigned>(rows), 256, 0, s>>>(static_cast<const __nv_bfloat16*>(ref),
                                                            static_cast<const __nv_bfloat16*>(got), static_cast<int>(cols), worst);
  MI_CUDA_CHECK_LAUNCH();
  DevStats h{};
  unsigned int w = 0;
  MI_CUDA_CHECK(cudaMemcpyAsync(&h, d_stats, sizeof h, cudaMemcpyDeviceToHost, s));
  MI_CUDA_CHECK(cudaMemcpyAsync(&w, worst, sizeof w, cudaMemcpyDeviceToHost, s));
  MI_CUDA_CHECK(cudaStreamSynchronize(s));
  Stats r;
  r.cos = h.aa > 0 && h.bb > 0 ? h.dot / std::sqrt(h.aa * h.bb) : 1.0;
  r.rel_l2 = h.aa > 0 ? std::sqrt(h.dd / h.aa) : 0.0;
  float f;
  std::memcpy(&f, &h.max_abs_bits, 4);
  r.max_abs = f;
  std::memcpy(&f, &h.max_ref_bits, 4);
  r.max_ref = f;
  std::memcpy(&f, &w, 4);
  r.min_row_cos = 1.0 - f;
  r.differ = h.differ;
  return r;
}

// fp32 velocities: cosine and relative L2 of got vs ref (host).
std::pair<double, double> cos_rel(const std::vector<float>& ref, const std::vector<float>& got) {
  double dot = 0, rr = 0, gg = 0, dd = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    dot += static_cast<double>(ref[i]) * got[i];
    rr += static_cast<double>(ref[i]) * ref[i];
    gg += static_cast<double>(got[i]) * got[i];
    dd += (static_cast<double>(ref[i]) - got[i]) * (static_cast<double>(ref[i]) - got[i]);
  }
  return {dot / std::sqrt(rr * gg), std::sqrt(dd / rr)};
}

// ---------------------------------------------------------------- NVML sampler (SM clock, board power)

class ClockSampler {
 public:
  ClockSampler() {
#ifdef _WIN32
    lib_ = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto sym = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib_), n)); };
#else
    lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    const auto sym = [&](const char* n) { return dlsym(lib_, n); };
#endif
    if (lib_ == nullptr) return;
    init_ = reinterpret_cast<int (*)()>(sym("nvmlInit_v2"));
    shutdown_ = reinterpret_cast<int (*)()>(sym("nvmlShutdown"));
    by_bus_ = reinterpret_cast<int (*)(const char*, void**)>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
    clock_ = reinterpret_cast<int (*)(void*, int, unsigned*)>(sym("nvmlDeviceGetClockInfo"));
    power_ = reinterpret_cast<int (*)(void*, unsigned*)>(sym("nvmlDeviceGetPowerUsage"));
    if (!init_ || !shutdown_ || !by_bus_ || !clock_ || !power_ || init_() != 0) return;
    char bus[32] = {};
    int dev = 0;
    MI_CUDA_CHECK(cudaGetDevice(&dev));
    MI_CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof bus, dev));
    if (by_bus_(bus, &device_) != 0) return;
    ok_ = true;
    thread_ = std::thread([this] {
      while (!stop_.load()) {
        unsigned mhz = 0, mw = 0;
        clock_(device_, 1 /* NVML_CLOCK_SM */, &mhz);
        power_(device_, &mw);
        {
          std::lock_guard<std::mutex> lock(mu_);
          samples_.push_back({std::chrono::steady_clock::now(), static_cast<double>(mhz), mw / 1000.0});
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }
  ~ClockSampler() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    if (ok_) shutdown_();
#ifdef _WIN32
    if (lib_ != nullptr) FreeLibrary(static_cast<HMODULE>(lib_));
#else
    if (lib_ != nullptr) dlclose(lib_);
#endif
  }
  ClockSampler(const ClockSampler&) = delete;
  ClockSampler& operator=(const ClockSampler&) = delete;
  bool ok() const { return ok_; }
  using Time = std::chrono::steady_clock::time_point;
  // Mean SM clock (MHz) and power (W) of the samples inside the windows.
  std::pair<double, double> mean(const std::vector<std::pair<Time, Time>>& windows) {
    std::lock_guard<std::mutex> lock(mu_);
    double mhz = 0, w = 0;
    int n = 0;
    for (const Sample& s : samples_) {
      for (const auto& [a, b] : windows) {
        if (s.t >= a && s.t <= b) {
          mhz += s.mhz;
          w += s.watts;
          ++n;
          break;
        }
      }
    }
    return n > 0 ? std::pair{mhz / n, w / n} : std::pair{0.0, 0.0};
  }

 private:
  struct Sample {
    Time t;
    double mhz, watts;
  };
  void* lib_ = nullptr;
  int (*init_)() = nullptr;
  int (*shutdown_)() = nullptr;
  int (*by_bus_)(const char*, void**) = nullptr;
  int (*clock_)(void*, int, unsigned*) = nullptr;
  int (*power_)(void*, unsigned*) = nullptr;
  void* device_ = nullptr;
  bool ok_ = false;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  std::mutex mu_;
  std::vector<Sample> samples_;
};

// ---------------------------------------------------------------- the step

struct CaseSpec {
  const char* name;
  const char* latents;      // golden case directory: initial_noise.npy, timestep0.npy
  const char* te_cond;      // golden text-encoder output stem (".te_last_hidden.npy")
  std::int64_t cond_len;    // rows of te_cond used (0: all)
  const char* te_uncond;
  const char* golden_vel;   // golden case with dit_step0_{cond,uncond}.npy for the reference, or nullptr
};

struct AttnRun {
  std::int64_t t0, t1;
  Tensor cu;
  int max_seqlen;
};

// Attention of one block: q, k, v, out views; the runs' cu_seqlens.
using AttnFn = std::function<void(int block, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                                  const AttnRun& run)>;

struct VariantMetrics {
  std::vector<Stats> per_block;  // index = block (blocks a run executes)
};

class GpuAttentionQ8Dit : public ::testing::Test {
 protected:
  static bool have_inputs() {
    return fs::is_regular_file(bundle_dir() / "moss.mossidx") && fs::is_directory(golden_dir() / "g_fridge_1024") &&
           fs::is_directory(golden_dir() / "g_fridge_1456") &&
           fs::is_regular_file(golden_dir() / "g_long4096_1024.cond.te_last_hidden.npy");
  }
  void SetUp() override {
    if (!env_is_1("MOSSINFER_SLOW_TESTS")) {
      GTEST_SKIP() << "slow (~80 s for the 3 cases, up to ~5 GB device memory): set MOSSINFER_SLOW_TESTS=1 to run";
    }
    if (!have_inputs()) GTEST_SKIP() << "weights or goldens not present (" << bundle_dir() << ")";
    for (Variant v : kVariants) {
      if (!q8::supported(v)) GTEST_SKIP() << q8::variant_name(v) << " is not supported on this device/build";
    }
  }

  static void run_case(const CaseSpec& c);
};

void GpuAttentionQ8Dit::run_case(const CaseSpec& c) {
  const fs::path g = golden_dir();
  const auto index = mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir());
  mi::runtime::StreamSet streams;
  const cudaStream_t stream = streams.compute();

  const auto noise = mi::npy::load(g / c.latents / "initial_noise.npy");
  const std::int64_t h = noise.shape[2], w = noise.shape[3], hw = h * w;
  const float timestep = static_cast<float>(mi::npy::load(g / c.latents / "timestep0.npy").to_vector<double>().at(0));
  const auto te_c = mi::npy::load(g / (std::string(c.te_cond) + ".te_last_hidden.npy"));
  const auto te_u = mi::npy::load(g / (std::string(c.te_uncond) + ".te_last_hidden.npy"));
  const std::vector<std::int64_t> lengths = {c.cond_len > 0 ? c.cond_len : te_c.shape[0], te_u.shape[0]};
  md::SkipSet uncond_skip;
  uncond_skip.set(22);  // the golden cases' uncond_skip_layers
  const std::vector<md::SkipSet> skips = {md::SkipSet{}, uncond_skip};
  const std::vector<std::int64_t> off = {0, hw + lengths[0], 2 * hw + lengths[0] + lengths[1]};
  const std::int64_t total = off[2];
  const int max_len = static_cast<int>(std::max(off[1] - off[0], off[2] - off[1]));
  std::printf("[attention_q8_dit] %s: %lldx%lld latents, rows %lld + %lld tokens (total %lld), timestep %.1f\n", c.name,
              static_cast<long long>(h), static_cast<long long>(w), static_cast<long long>(off[1]),
              static_cast<long long>(off[2] - off[1]), static_cast<long long>(total), timestep);

  // Engine: every block streamed from the files (the local blocks stream their own copy), op by op.
  md::Residency streamed;
  streamed.resident_blocks = 0;
  streamed.buffers = 2;
  const std::size_t io = 3 * mi::runtime::align_up(static_cast<std::size_t>(128 * hw * 4)) +
                         2 * mi::runtime::align_up(static_cast<std::size_t>(std::max(lengths[0], lengths[1]) * (md::kTextDim + kDim) * 2)) +
                         mi::runtime::align_up(6 * kDim * 2) + mi::runtime::align_up(2 * kDim * 2);
  const std::size_t engine_bytes =
      md::MossDit::persistent_bytes(index, streamed) + io + md::MossDit::stream_workspace_bytes(index, streamed) +
      std::max(md::MossDit::workspace_bytes(h, w, lengths, md::BlockPath::kOpByOp),
               md::MossDit::encode_text_workspace_bytes(std::max(lengths[0], lengths[1])));
  // Local blocks: activations, 2 streamed block buffers, attention workspaces, saved Q/K/V for the speed test.
  using mi::runtime::align_up;
  const std::size_t tok = static_cast<std::size_t>(total);
  std::size_t q8_ws = 0;
  for (Variant v : kVariants) {
    q8_ws = std::max(q8_ws, q8::attention_q8_workspace_bytes({v}, total, total, 2, max_len, md::kHeads, md::kHeads));
  }
  const std::size_t local_bytes = 4 * align_up(tok * kDim * 2) /* x, y, x0, attn_alt */ + align_up(tok * kBigCols * 2) +
                                  align_up(tok * 128 * 2) + align_up(6 * kDim * 2) + align_up(1024) +
                                  align_up(mi::attention::flash_fwd_workspace_bytes(total, md::kHeads)) + align_up(q8_ws) +
                                  align_up(tok * kQkv * 2) /* saved qkv */ + align_up(1024) /* stats */ +
                                  align_up(mi::kernels::Gemm::kWorkspaceBytes) +
                                  md::BlockWeights::buffer_bytes(index, streamed) + (std::size_t{64} << 20);
  std::printf("[attention_q8_dit] arenas: engine %.2f GiB + local %.2f GiB (free %.2f GiB)\n", engine_bytes / kGiB,
              local_bytes / kGiB, mi::runtime::DeviceArena::free_device_bytes() / kGiB);
  mi::runtime::DeviceArena arena(engine_bytes);
  mi::runtime::DeviceArena local(local_bytes);
  md::MossDit model(index, arena, streams, false, nullptr, streamed);
  model.set_block_path(md::BlockPath::kOpByOp);

  const auto refine = [&](const mi::npy::Array& te, std::int64_t rows) {
    const auto v = te.to_vector<float>();
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(rows * md::kTextDim));
    for (std::size_t i = 0; i < hv.size(); ++i) hv[i] = to_bf16(v[i]);
    const Tensor in = arena.alloc_persistent_tensor(DType::kBF16, {rows, md::kTextDim});
    MI_CUDA_CHECK(cudaMemcpy(in.data, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
    const Tensor out = arena.alloc_persistent_tensor(DType::kBF16, {rows, kDim});
    model.encode_text(in, out, stream);
    return out;
  };
  const Tensor text_c = refine(te_c, lengths[0]), text_u = refine(te_u, lengths[1]);
  const Tensor latents = arena.alloc_persistent_tensor(DType::kF32, {128, h, w});
  {
    const auto nv = noise.to_vector<float>();
    MI_CUDA_CHECK(cudaMemcpy(latents.data, nv.data(), nv.size() * 4, cudaMemcpyHostToDevice));
  }
  const Tensor outs[] = {arena.alloc_persistent_tensor(DType::kF32, {128, h, w}),
                         arena.alloc_persistent_tensor(DType::kF32, {128, h, w})};
  const Tensor head = arena.alloc_persistent_tensor(DType::kBF16, {2 * kDim});

  // Local activations (persistent in the local arena: they live for the whole case).
  const Tensor x = local.alloc_persistent_tensor(DType::kBF16, {total, kDim});
  const Tensor y = local.alloc_persistent_tensor(DType::kBF16, {total, kDim});
  const Tensor x0 = local.alloc_persistent_tensor(DType::kBF16, {total, kDim});
  const Tensor attn_alt = local.alloc_persistent_tensor(DType::kBF16, {total, kDim});
  const Tensor big = local.alloc_persistent_tensor(DType::kBF16, {total * kBigCols});
  const Tensor rope = local.alloc_persistent_tensor(DType::kBF16, {total, 128});
  const Tensor mod = local.alloc_persistent_tensor(DType::kBF16, {6 * kDim});
  const Tensor cu_pool = local.alloc_persistent_tensor(DType::kI32, {256});
  const std::size_t lse_bytes = mi::attention::flash_fwd_workspace_bytes(total, md::kHeads);
  void* lse = local.alloc_persistent(lse_bytes);
  void* ws = local.alloc_persistent(q8_ws);
  const Tensor saved_qkv = local.alloc_persistent_tensor(DType::kBF16, {total, kQkv});
  auto* d_stats = static_cast<DevStats*>(local.alloc_persistent(1024));
  void* gemm_ws = local.alloc_persistent(mi::kernels::Gemm::kWorkspaceBytes);
  mi::kernels::Gemm gemm;
  md::BlockWeights blocks(
      index, local, streamed, [](const mi::weights::WeightGroup&) -> mi::weights::TensorMap {
        throw std::logic_error("no resident blocks");
      },
      [](const mi::weights::WeightGroup&) {});

  model.time_modulation(timestep, mod, head, stream);
  for (int r = 0; r < 2; ++r) {
    mi::kernels::moss_rope_table(rope.slice0(off[r], off[r + 1]), static_cast<int>(h), static_cast<int>(w),
                                 static_cast<int>(lengths[r]), stream);
  }
  // The runs of every block (MossDit::forward's construction for two rows): both rows, or the cond row at block 22.
  std::vector<std::int32_t> cu_host = {0, static_cast<std::int32_t>(off[1]), static_cast<std::int32_t>(off[2]), 0,
                                       static_cast<std::int32_t>(off[1])};
  MI_CUDA_CHECK(cudaMemcpy(cu_pool.data, cu_host.data(), cu_host.size() * 4, cudaMemcpyHostToDevice));
  const AttnRun both{0, total, cu_pool.slice0(0, 3), max_len};
  const AttnRun cond_only{0, off[1], cu_pool.slice0(3, 5), static_cast<int>(off[1])};
  const auto runs_of = [&](int bi) { return std::vector<AttnRun>{bi == 22 ? cond_only : both}; };

  // One block (MossDit::run_block's ops and buffer layout) on run `run` of the local stream x, attention by `attn`.
  const auto linear = [&](const Tensor& out, const Tensor& in, const Tensor& wt) {
    gemm.linear(out, in, wt, nullptr, gemm_ws, mi::kernels::Gemm::kWorkspaceBytes, stream);
  };
  const auto block = [&](const md::BlockTensors& b, const AttnRun& run, const AttnFn& attn, int bi) {
    auto* bp = static_cast<std::uint16_t*>(big.data);
    const Tensor qkv = bf16_view(bp, {total, kQkv}).slice0(run.t0, run.t1);
    const Tensor at = bf16_view(bp + total * kQkv, {total, kDim}).slice0(run.t0, run.t1);
    const Tensor up_gate = bf16_view(bp, {total, kUpGate}).slice0(run.t0, run.t1);
    const Tensor ffn = bf16_view(bp + total * kUpGate, {total, md::kFfnDim}).slice0(run.t0, run.t1);
    const Tensor xs = x.slice0(run.t0, run.t1), ys = y.slice0(run.t0, run.t1), rs = rope.slice0(run.t0, run.t1);
    const Tensor q = heads(qkv, 0), k = heads(qkv, kDim), v = heads(qkv, 2 * kDim);
    mi::kernels::rms_norm_modulate_moss(xs, b.norm1, vec(mod, 1), vec(mod, 0), ys, stream);
    linear(qkv, ys, b.qkv);
    mi::kernels::rms_norm_moss(q, b.q_norm, q, stream);
    mi::kernels::rms_norm_moss(k, b.k_norm, k, stream);
    mi::kernels::moss_rope_apply(q, rs, q, stream);
    mi::kernels::moss_rope_apply(k, rs, k, stream);
    attn(bi, q, k, v, heads(at, 0), run);
    linear(ys, at, b.o);
    mi::kernels::gated_residual(xs, ys, vec(mod, 2), xs, stream);
    mi::kernels::rms_norm_modulate_moss(xs, b.norm2, vec(mod, 4), vec(mod, 3), ys, stream);
    linear(up_gate, ys, b.up_gate);
    mi::kernels::situ_glu(up_gate, ffn, stream);
    linear(ys, ffn, b.down);
    mi::kernels::gated_residual(xs, ys, vec(mod, 5), xs, stream);
  };
  const auto fa2 = [&](const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, const AttnRun& run) {
    mi::attention::flash_fwd(q, k, v, o, run.cu, run.cu, run.max_seqlen, false, kAttnScale, lse, lse_bytes, stream);
  };
  const auto q8run = [&](const q8::Options& opt, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o,
                         const AttnRun& run) {
    q8::attention_q8(opt, q, k, v, o, run.cu, run.cu, run.max_seqlen, kAttnScale, ws, q8_ws, stream);
  };
  const auto velocities = [&] {
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::vector<float>> v(2, std::vector<float>(static_cast<std::size_t>(128 * hw)));
    for (int i = 0; i < 2; ++i) MI_CUDA_CHECK(cudaMemcpy(v[i].data(), outs[i].data, outs[i].nbytes(), cudaMemcpyDeviceToHost));
    return v;
  };
  const md::Row rows[] = {{text_c, skips[0]}, {text_u, skips[1]}};

  // Pass 0: the joint input of block 0 (every row skips block 0, so the hook sees it unchanged); stop there.
  {
    struct Captured {};
    md::SkipSet s0 = skips[0], s1 = skips[1];
    s0.set(0);
    s1.set(0);
    const md::Row capture_rows[] = {{text_c, s0}, {text_u, s1}};
    try {
      model.forward(latents, timestep, capture_rows, outs, stream, [&](int bi, const Tensor& xe) {
        if (bi != 0) return;
        MI_CUDA_CHECK(cudaMemcpyAsync(x0.data, xe.data, x0.nbytes(), cudaMemcpyDeviceToDevice, stream));
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        throw Captured{};
      });
      FAIL() << "the capture pass did not stop at block 0";
    } catch (const Captured&) {
    }
  }

  // One step with every block replaced by the local block with attention `attn`; `after(bi, xe)` runs before the
  // replacement is written (xe = the engine's output of block bi on the same input).
  const auto local_step = [&](const AttnFn& attn, const std::function<void(int, const Tensor&)>& after) {
    MI_CUDA_CHECK(cudaMemcpyAsync(x.data, x0.data, x.nbytes(), cudaMemcpyDeviceToDevice, stream));
    mi::runtime::StackScope scope(local);  // the pass carves its block buffers from the local arena's stack
    md::BlockWeights::Pass pass(blocks, stream);
    model.forward(latents, timestep, rows, outs, stream, [&](int bi, const Tensor& xe) {
      const md::BlockTensors& b = pass.acquire(bi);
      for (const AttnRun& run : runs_of(bi)) block(b, run, attn, bi);
      pass.release(bi);
      if (after) after(bi, xe);
      MI_CUDA_CHECK(cudaMemcpyAsync(xe.data, x.data, x.nbytes(), cudaMemcpyDeviceToDevice, stream));
    });
    pass.finish();
    return velocities();
  };

  // Pass 1: reference (FlashAttention-2), with every variant measured on each block's Q/K/V.
  std::vector<VariantMetrics> metrics(kMetricConfigs);
  for (auto& m : metrics) m.per_block.resize(md::kLayers);
  std::size_t engine_mismatch = 0;
  const auto ref = local_step(
      [&](int bi, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, const AttnRun& run) {
        fa2(q, k, v, o, run);
        if (bi == kBenchBlock) {
          MI_CUDA_CHECK(cudaMemcpyAsync(saved_qkv.data, big.data, static_cast<std::size_t>(total * kQkv * 2),
                                        cudaMemcpyDeviceToDevice, stream));
        }
        const Tensor alt = heads(attn_alt.slice0(run.t0, run.t1), 0);
        for (int i = 0; i < kMetricConfigs; ++i) {
          q8run(metric_config(i), q, k, v, alt, run);
          // o is a strided view (row stride 3072 inside `big`): contiguous rows [run.t0, run.t1) of [total, 3072].
          metrics[i].per_block[bi] = measure(o.data, alt.data, run.t1 - run.t0, kDim, d_stats, stream);
        }
      },
      [&](int, const Tensor& xe) { engine_mismatch += measure(xe.data, x.data, total, kDim, d_stats, stream).differ; });
  EXPECT_EQ(engine_mismatch, 0u) << "the local FlashAttention-2 blocks must reproduce the engine bit for bit";
  if (c.golden_vel != nullptr) {
    for (int r = 0; r < 2; ++r) {
      const auto gold = mi::npy::load(g / c.golden_vel / (r == 0 ? "dit_step0_cond.npy" : "dit_step0_uncond.npy")).to_vector<float>();
      const auto [cs, rl] = cos_rel(gold, ref[r]);
      std::printf("[attention_q8_dit] %s reference (engine, bf16 FA2) vs golden %s velocity: cos %.7f rel L2 %.3e\n",
                  c.name, r == 0 ? "cond" : "uncond", cs, rl);
      EXPECT_GE(cs, 0.99988);  // AC5
    }
  }

  // Per-block attention accuracy table.
  std::printf("[attention_q8_dit] %s attention vs FlashAttention-2 (bf16 out), per block over the run's tokens:\n", c.name);
  std::printf("  %-20s %12s %12s %10s %10s %10s %12s %10s\n", "variant", "cos min", "cos mean", "relL2 max", "relL2 mean",
              "max|err|", "max|ref|", "tok cos min");
  for (int i = 0; i < kMetricConfigs; ++i) {
    double cmin = 1, csum = 0, rmax = 0, rsum = 0, amax = 0, refmax = 0, tmin = 1;
    for (const Stats& s : metrics[i].per_block) {
      cmin = std::min(cmin, s.cos);
      csum += s.cos;
      rmax = std::max(rmax, s.rel_l2);
      rsum += s.rel_l2;
      amax = std::max(amax, s.max_abs);
      refmax = std::max(refmax, s.max_ref);
      tmin = std::min(tmin, s.min_row_cos);
    }
    std::printf("  %-20s %12.7f %12.7f %10.3e %10.3e %10.3e %12.3e %10.6f\n", metric_name(i), cmin,
                csum / md::kLayers, rmax, rsum / md::kLayers, amax, refmax, tmin);
    for (int bi : {0, 21, 41}) {
      const Stats& s = metrics[i].per_block[bi];
      std::printf("      block %2d: cos %.7f relL2 %.3e max|err| %.3e (max|ref| %.3e) tok cos min %.6f\n", bi, s.cos,
                  s.rel_l2, s.max_abs, s.max_ref, s.min_row_cos);
    }
    EXPECT_GE(cmin, block_cos_floor(i)) << metric_name(i) << ": worst block cosine";
    EXPECT_GE(tmin, kTokenCosFloor) << metric_name(i) << ": worst token cosine";
  }

  // Passes 2..5: attention replaced in all 42 blocks by each variant.
  std::printf("[attention_q8_dit] %s step-0 velocity with 8-bit attention in all 42 blocks vs bf16 (FA2) step:\n", c.name);
  std::vector<float> cfg_ref(ref[0].size());
  for (std::size_t i = 0; i < cfg_ref.size(); ++i) cfg_ref[i] = ref[1][i] + 3.0f * (ref[0][i] - ref[1][i]);
  for (int i = 0; i < kVariantCount; ++i) {
    const Variant var = kVariants[i];
    const auto got = local_step(
        [&](int, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, const AttnRun& run) {
          q8run({var}, q, k, v, o, run);
        },
        {});
    const auto [cc, rc] = cos_rel(ref[0], got[0]);
    const auto [cu, ru] = cos_rel(ref[1], got[1]);
    std::vector<float> cfg(got[0].size());
    for (std::size_t j = 0; j < cfg.size(); ++j) cfg[j] = got[1][j] + 3.0f * (got[0][j] - got[1][j]);
    const auto [cg, rg] = cos_rel(cfg_ref, cfg);
    std::printf("  %-18s cond cos %.7f (relL2 %.3e)  uncond cos %.7f (relL2 %.3e)  CFG-3 cos %.7f (relL2 %.3e)\n",
                q8::variant_name(var), cc, rc, cu, ru, cg, rg);
    EXPECT_GE(cc, kVelocityCosFloor) << q8::variant_name(var) << ": cond velocity";
    EXPECT_GE(cu, kVelocityCosFloor) << q8::variant_name(var) << ": uncond velocity";
  }

  // Speed: FlashAttention-2 vs each variant on block 21's Q/K/V (both rows), interleaved rounds.
  if (!env_is_1("MI_RUN_PERF")) {
    std::printf("[attention_q8_dit] %s speed comparison skipped (set MI_RUN_PERF=1)\n", c.name);
  } else {
    const Tensor sq = heads(saved_qkv, 0), sk = heads(saved_qkv, kDim), sv = heads(saved_qkv, 2 * kDim);
    const Tensor so = heads(attn_alt, 0);
    ClockSampler sampler;
    cudaEvent_t e0, e1, emid;
    MI_CUDA_CHECK(cudaEventCreate(&e0));
    MI_CUDA_CHECK(cudaEventCreate(&e1));
    MI_CUDA_CHECK(cudaEventCreate(&emid));
    constexpr int kIters = 10, kRounds = 6;
    struct Timing {
      std::vector<double> total_ms, pre_ms;
      std::vector<std::pair<ClockSampler::Time, ClockSampler::Time>> windows;
    };
    std::vector<Timing> t(kVariantCount + 1);  // [0] = FA2
    const auto launch = [&](int m) {
      if (m == 0) {
        fa2(sq, sk, sv, so, both);
      } else {
        q8run({kVariants[m - 1]}, sq, sk, sv, so, both);
      }
    };
    for (int m = 0; m <= kVariantCount; ++m) launch(m);  // warm-up
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int round = 0; round < kRounds; ++round) {
      for (int m = 0; m <= kVariantCount; ++m) {
        const auto w0 = std::chrono::steady_clock::now();
        MI_CUDA_CHECK(cudaEventRecord(e0, stream));
        for (int i = 0; i < kIters; ++i) launch(m);
        MI_CUDA_CHECK(cudaEventRecord(e1, stream));
        MI_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        MI_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        t[m].windows.emplace_back(w0, std::chrono::steady_clock::now());
        t[m].total_ms.push_back(ms / kIters);
        if (m > 0) {  // pre-pass share: one extra call with the split event
          q8::set_prepass_event_for_testing(emid);
          MI_CUDA_CHECK(cudaEventRecord(e0, stream));
          launch(m);
          MI_CUDA_CHECK(cudaEventRecord(e1, stream));
          MI_CUDA_CHECK(cudaEventSynchronize(e1));
          q8::set_prepass_event_for_testing(nullptr);
          float pre = 0;
          MI_CUDA_CHECK(cudaEventElapsedTime(&pre, e0, emid));
          t[m].pre_ms.push_back(pre);
        }
      }
    }
    const auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v.empty() ? 0.0 : v[v.size() / 2];
    };
    double flops = 0;
    for (int r = 0; r < 2; ++r) flops += 4.0 * static_cast<double>(off[r + 1] - off[r]) * (off[r + 1] - off[r]) * md::kHeads * 128;
    const double fa2_ms = median(t[0].total_ms);
    std::printf("[attention_q8_dit] %s speed, block %d Q/K/V (rows %lld + %lld, 24 heads), median of %d rounds x %d calls%s:\n",
                c.name, kBenchBlock, static_cast<long long>(off[1]), static_cast<long long>(off[2] - off[1]), kRounds,
                kIters, sampler.ok() ? "" : " (NVML unavailable: no clocks)");
    std::printf("  %-18s %9s %9s %9s %8s %9s %9s %8s\n", "method", "total ms", "pre ms", "kernel ms", "TFLOP/s", "speedup",
                "SM MHz", "power W");
    for (int m = 0; m <= kVariantCount; ++m) {
      const double total_ms = median(t[m].total_ms), pre = m > 0 ? median(t[m].pre_ms) : 0.0;
      const auto [mhz, watts] = sampler.ok() ? sampler.mean(t[m].windows) : std::pair{0.0, 0.0};
      std::printf("  %-18s %9.3f %9.3f %9.3f %8.1f %8.3fx %9.0f %8.1f\n", m == 0 ? "flash_fwd (bf16)" : q8::variant_name(kVariants[m - 1]),
                  total_ms, pre, total_ms - pre, flops / (total_ms * 1e-3) / 1e12, fa2_ms / total_ms, mhz, watts);
    }
    MI_CUDA_CHECK(cudaEventDestroy(e0));
    MI_CUDA_CHECK(cudaEventDestroy(e1));
    MI_CUDA_CHECK(cudaEventDestroy(emid));
  }
}

TEST_F(GpuAttentionQ8Dit, Golden1024) {
  run_case({"golden_1024", "g_fridge_1024", "g_fridge_1024.cond", 0, "g_fridge_1024.uncond", "g_fridge_1024"});
}

TEST_F(GpuAttentionQ8Dit, Bench1024Prompt282) {
  run_case({"bench_1024", "g_fridge_1024", "g_long4096_1024.cond", 282, "g_fridge_1024.uncond", nullptr});
}

TEST_F(GpuAttentionQ8Dit, Long1456Prompt4096) {
  run_case({"long_1456", "g_fridge_1456", "g_long4096_1024.cond", 0, "g_long4096_1024.uncond", nullptr});
}

}  // namespace
