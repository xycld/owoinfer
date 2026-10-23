// 8-bit attention (src/kernels/attention_q8) on synthetic inputs: against exact attention (float64 torch vectors in
// tests/data/attention_q8 and an fp32 reference kernel) with explicit per-variant tolerances, against the upstream
// SageAttention kernels' outputs (port fidelity), varlen isolation, short sequences, lengths that are no multiple of
// the 128-row / 64-key tiles, empty sequences, strided views, determinism and argument checks. The real-activation
// accuracy, the end-to-end step and the speed comparison are in test_attention_q8_dit.cu.
// FP8 variants are skipped on devices below sm_89 (UnsupportedVariantsThrow checks they throw there).
//
// Build filter: MOSSINFER_TESTS_ONLY matches file-name prefixes, so "test_attention" also builds these files
// (test_attention_q8*.cu); use MOSSINFER_TESTS_ONLY="test_attention." to build only test_attention.cu.
#include <gtest/gtest.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "core/varlen.h"
#include "kernels/attention_q8/attention_q8.h"
#include "reference_gpu.h"
#include "util/npy.h"

namespace {

namespace fs = std::filesystem;
namespace q8 = mi::attention_q8;
using mi::DType;
using mi::Device;
using mi::Tensor;
using q8::Variant;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/attention_q8";
constexpr int kDim = 128;
constexpr Variant kVariants[] = {Variant::kInt8Fp16, Variant::kInt8Fp16F32Acc, Variant::kInt8Fp8, Variant::kInt8Fp8F32Acc};

bool is_fp8(Variant v) { return v == Variant::kInt8Fp8 || v == Variant::kInt8Fp8F32Acc; }

int device_cc() {
  int device = 0, major = 0, minor = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return major * 10 + minor;
}

// The e4m3 MMAs of the FP8 variants need sm_89.
bool fp8_supported() { return device_cc() >= 89; }

// Variants the tests run: the FP8 ones only where fp8_supported() (and the build has them, see attention_q8.h).
bool runnable(Variant v) { return (!is_fp8(v) || fp8_supported()) && q8::supported(v); }

float bf16_to_float(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// ---------------------------------------------------------------- device helpers

__device__ std::uint32_t hash_u32(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return static_cast<std::uint32_t>(x);
}

__device__ float normal(std::uint64_t seed, std::uint64_t i) {
  const std::uint64_t key = seed * 0x9E3779B97F4A7C15ull + i * 2;
  const float u1 = (static_cast<float>(hash_u32(key) >> 8) + 1.f) * (1.f / 16777217.f);
  const float u2 = static_cast<float>(hash_u32(key + 1) >> 8) * (1.f / 16777216.f);
  return sqrtf(-2.f * logf(u1)) * cospif(2.f * u2);
}

// bf16 values N(0, scale^2) plus a per-channel offset N(0, offset^2) shared by all tokens of a head (as the DiT's K
// has after RMSNorm with a learned weight; it is what K smoothing removes). Contiguous [tokens, heads, 128].
__global__ void fill_bf16(__nv_bfloat16* out, std::int64_t n, int head_elems, std::uint64_t seed, float scale, float offset) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const float bias = offset * normal(seed ^ 0x5bd1e995ull, static_cast<std::uint64_t>(i % head_elems));
    out[i] = __float2bfloat16(scale * normal(seed, static_cast<std::uint64_t>(i)) + bias);
  }
}

// Exact-math reference (as tests/unit/test_attention.cu): one block per (query row, head), blockDim == 128. Scores,
// softmax and P·V in fp32 (softmax sum in fp64). Contiguous [tokens, heads, 128], non-causal.
__global__ void reference_attention(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out,
                                    const int* cu_seqlens, int batch, int heads, int kv_heads, float scale) {
  extern __shared__ float smem[];
  const int d = blockDim.x, t = threadIdx.x;
  const int row = blockIdx.x, h = blockIdx.y, kvh = h / (heads / kv_heads);
  float* sq = smem;
  float* scores = smem + d;
  __shared__ float red[32];
  __shared__ double red_d[32];
  int b = 0;
  while (b + 1 < batch && cu_seqlens[b + 1] <= row) ++b;
  const int k0 = cu_seqlens[b], n = cu_seqlens[b + 1] - k0;
  sq[t] = __bfloat162float(q[(static_cast<std::int64_t>(row) * heads + h) * d + t]);
  __syncthreads();
  float local_max = -INFINITY;
  for (int j = t; j < n; j += d) {
    const __nv_bfloat16* kr = k + (static_cast<std::int64_t>(k0 + j) * kv_heads + kvh) * d;
    float s = 0.f;
    for (int i = 0; i < d; ++i) s = fmaf(sq[i], __bfloat162float(kr[i]), s);
    scores[j] = s * scale;
    local_max = fmaxf(local_max, scores[j]);
  }
  for (int o = 16; o > 0; o >>= 1) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
  if (t % 32 == 0) red[t / 32] = local_max;
  __syncthreads();
  float mx = -INFINITY;
  for (int w = 0; w < d / 32; ++w) mx = fmaxf(mx, red[w]);
  double local_sum = 0.0;
  for (int j = t; j < n; j += d) {
    scores[j] = expf(scores[j] - mx);
    local_sum += scores[j];
  }
  for (int o = 16; o > 0; o >>= 1) local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
  if (t % 32 == 0) red_d[t / 32] = local_sum;
  __syncthreads();
  double sum = 0.0;
  for (int w = 0; w < d / 32; ++w) sum += red_d[w];
  float acc = 0.f;
  for (int j = 0; j < n; ++j) acc = fmaf(scores[j], __bfloat162float(v[(static_cast<std::int64_t>(k0 + j) * kv_heads + kvh) * d + t]), acc);
  out[(static_cast<std::int64_t>(row) * heads + h) * d + t] = n > 0 ? static_cast<float>(acc / sum) : 0.f;
}

// Fills the shared memory of every SM with 0xFF bytes (fp16/bf16 NaN): kernels that read shared memory they never
// wrote see NaN. Shared memory is not cleared between kernels, so this exposes a masked-but-unwritten tile row
// (P = 0 times NaN is NaN).
__global__ void poison_shared_memory_kernel(int bytes) {
  extern __shared__ std::uint32_t poison[];
  for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x) poison[i] = 0xffffffffu;
}

void poison_shared_memory() {
  int device = 0, sms = 0, smem = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  MI_CUDA_CHECK(cudaFuncSetAttribute(poison_shared_memory_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  poison_shared_memory_kernel<<<4 * sms, 1024, smem>>>(smem);
  MI_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------- host helpers

struct Problem {
  int heads = 0, kv_heads = 0;
  std::vector<int> cu;  // [batch + 1]
  std::int64_t total() const { return cu.back(); }
  int batch() const { return static_cast<int>(cu.size()) - 1; }
  int max_seqlen() const {
    int m = 0;
    for (int b = 0; b < batch(); ++b) m = std::max(m, cu[b + 1] - cu[b]);
    return std::max(m, 1);
  }
  float scale() const { return static_cast<float>(1.0 / std::sqrt(static_cast<double>(kDim))); }
};

Problem make_problem(int heads, int kv_heads, const std::vector<int>& seqlens) {
  Problem p{heads, kv_heads, {0}};
  for (int s : seqlens) p.cu.push_back(p.cu.back() + s);
  return p;
}

std::size_t workspace_bytes(const Problem& p, int max_seqlen) {
  std::size_t m = 0;
  for (Variant v : kVariants) {
    m = std::max(m, q8::attention_q8_workspace_bytes({v}, p.total(), p.total(), p.batch(), max_seqlen, p.heads, p.kv_heads));
  }
  return m;
}

// Device buffers for one problem: q/k/v/out contiguous bf16, cu_seqlens, workspace (for every variant).
struct Buffers {
  explicit Buffers(const Problem& p, int max_seqlen = -1)
      : q(std::max<std::int64_t>(p.total(), 1) * p.heads * kDim * 2),
        k(std::max<std::int64_t>(p.total(), 1) * p.kv_heads * kDim * 2),
        v(std::max<std::int64_t>(p.total(), 1) * p.kv_heads * kDim * 2),
        out(std::max<std::int64_t>(p.total(), 1) * p.heads * kDim * 2),
        cu(p.cu.size() * sizeof(int)),
        ws(workspace_bytes(p, max_seqlen < 0 ? p.max_seqlen() : max_seqlen)) {
    cu.upload(p.cu.data(), p.cu.size() * sizeof(int));
  }
  Tensor tq(const Problem& p) const { return Tensor::view(q.get(), DType::kBF16, Device::kCUDA, {p.total(), p.heads, kDim}); }
  Tensor tk(const Problem& p) const { return Tensor::view(k.get(), DType::kBF16, Device::kCUDA, {p.total(), p.kv_heads, kDim}); }
  Tensor tv(const Problem& p) const { return Tensor::view(v.get(), DType::kBF16, Device::kCUDA, {p.total(), p.kv_heads, kDim}); }
  Tensor tout(const Problem& p) const { return Tensor::view(out.get(), DType::kBF16, Device::kCUDA, {p.total(), p.heads, kDim}); }
  Tensor tcu(const Problem& p) const { return Tensor::view(cu.get(), DType::kI32, Device::kCUDA, {static_cast<std::int64_t>(p.cu.size())}); }

  void run(const Problem& p, const q8::Options& o, int max_seqlen = -1) const {
    q8::attention_q8(o, tq(p), tk(p), tv(p), tout(p), tcu(p), tcu(p), max_seqlen < 0 ? p.max_seqlen() : max_seqlen,
                     p.scale(), ws.get(), ws.size(), nullptr);
  }
  // K offset std 3 (smoothing matters), q/k scale 1.5 as the FA2 tests, v unit normal.
  void fill_random(const Problem& p, std::uint64_t seed) const {
    fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(q.get()), p.total() * p.heads * kDim, p.heads * kDim, seed, 1.5f, 0.f);
    fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(k.get()), p.total() * p.kv_heads * kDim, p.kv_heads * kDim,
                             seed + 1, 1.5f, 3.f);
    fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(v.get()), p.total() * p.kv_heads * kDim, p.kv_heads * kDim,
                             seed + 2, 1.f, 0.f);
    MI_CUDA_CHECK_LAUNCH();
  }
  void upload(const std::vector<std::uint16_t>& hq, const std::vector<std::uint16_t>& hk,
              const std::vector<std::uint16_t>& hv) {
    q.upload(hq.data(), hq.size() * 2);
    k.upload(hk.data(), hk.size() * 2);
    v.upload(hv.data(), hv.size() * 2);
  }
  std::vector<std::uint16_t> download_out(const Problem& p) const {
    MI_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> h(static_cast<std::size_t>(p.total() * p.heads * kDim));
    out.download(h.data(), h.size() * 2);
    return h;
  }

  mi::DeviceBuffer q, k, v, out, cu, ws;
};

// fp32 reference output of every row: [total, heads, 128].
std::vector<float> reference(const Problem& p, const Buffers& buf) {
  mi::DeviceBuffer d_out(static_cast<std::size_t>(p.total() * p.heads * kDim) * sizeof(float));
  const int smem = (kDim + p.max_seqlen()) * static_cast<int>(sizeof(float));
  MI_CUDA_CHECK(cudaFuncSetAttribute(reference_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  reference_attention<<<dim3(static_cast<unsigned>(p.total()), static_cast<unsigned>(p.heads)), kDim, smem>>>(
      static_cast<const __nv_bfloat16*>(buf.q.get()), static_cast<const __nv_bfloat16*>(buf.k.get()),
      static_cast<const __nv_bfloat16*>(buf.v.get()), static_cast<float*>(d_out.get()),
      static_cast<const int*>(buf.cu.get()), p.batch(), p.heads, p.kv_heads, p.scale());
  MI_CUDA_CHECK_LAUNCH();
  std::vector<float> h(static_cast<std::size_t>(p.total() * p.heads * kDim));
  d_out.download(h.data(), h.size() * sizeof(float));
  return h;
}

struct Metrics {
  double cos = 1, rel_l2 = 0, max_abs = 0;
};

// Over elements [begin, end) of got (bf16) vs want (fp32).
Metrics measure(const std::vector<std::uint16_t>& got, const std::vector<float>& want, std::size_t begin, std::size_t end) {
  double dot = 0, ng = 0, nw = 0, nd = 0, mx = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const double g = bf16_to_float(got[i]), w = want[i];
    dot += g * w;
    ng += g * g;
    nw += w * w;
    nd += (g - w) * (g - w);
    mx = std::max(mx, std::abs(g - w));
  }
  Metrics m;  // a NaN anywhere in got propagates into cos and rel L2 (and fails every check)
  if (nw > 0) {
    m.cos = ng > 0 || std::isnan(ng) ? dot / std::sqrt(ng * nw) : 0.0;
    m.rel_l2 = std::sqrt(nd / nw);
  }
  m.max_abs = mx;
  return m;
}

// Tolerances against exact attention on these synthetic inputs (q, k ~ N(0, 1.5^2), K offset std 3, v ~ N(0, 1);
// |out| <~ 1). Worst measured on sm_89 over the tests below (1 to 2100 keys): fp16 PV cos 0.99973, rel L2 2.3e-2,
// max|err| 0.093; FP8 PV cos 0.99929, rel L2 3.8e-2, max|err| 0.16 (e4m3 P and V have 3 mantissa bits). The limits
// leave ~1.5x headroom on the error.
struct Tol {
  double cos, rel_l2, max_abs;
};
Tol tolerance(Variant v) {
  switch (v) {
    case Variant::kInt8Fp16:
    case Variant::kInt8Fp16F32Acc: return {0.9995, 0.035, 0.15};
    case Variant::kInt8Fp8:
    case Variant::kInt8Fp8F32Acc: return {0.998, 0.06, 0.25};
  }
  return {1, 0, 0};
}

void expect_within(const Metrics& m, Variant v, const std::string& what) {
  const Tol t = tolerance(v);
  std::printf("  %-18s %-40s cos %.7f  rel L2 %.3e  max|err| %.3e\n", q8::variant_name(v), what.c_str(), m.cos, m.rel_l2,
              m.max_abs);
  EXPECT_GE(m.cos, t.cos) << q8::variant_name(v) << " " << what;
  EXPECT_LE(m.rel_l2, t.rel_l2) << q8::variant_name(v) << " " << what;
  EXPECT_LE(m.max_abs, t.max_abs) << q8::variant_name(v) << " " << what;
}

// Every variant against the fp32 reference, over all rows and per sequence (a varlen error shows in one sequence).
// `prepare` may edit the random inputs before the reference runs.
void check_against_reference(const Problem& p, std::uint64_t seed, int max_seqlen = -1,
                             const std::function<void(const Buffers&)>& prepare = {}) {
  Buffers buf(p, max_seqlen);
  buf.fill_random(p, seed);
  if (prepare) prepare(buf);
  const auto want = reference(p, buf);
  const std::size_t row = static_cast<std::size_t>(p.heads) * kDim;
  for (Variant v : kVariants) {
    if (!runnable(v)) continue;
    MI_CUDA_CHECK(cudaMemset(buf.out.get(), 0xff, buf.out.size()));  // NaN pattern: every row must be written
    poison_shared_memory();
    buf.run(p, {v}, max_seqlen);
    const auto got = buf.download_out(p);
    expect_within(measure(got, want, 0, got.size()), v, "all rows");
    for (int b = 0; b < p.batch(); ++b) {
      if (p.cu[b + 1] == p.cu[b]) continue;
      const Metrics m = measure(got, want, p.cu[b] * row, p.cu[b + 1] * row);
      const Tol t = tolerance(v);
      EXPECT_GE(m.cos, t.cos) << q8::variant_name(v) << " sequence " << b << " (length " << p.cu[b + 1] - p.cu[b] << ")";
      EXPECT_LE(m.max_abs, t.max_abs) << q8::variant_name(v) << " sequence " << b;
    }
  }
}

std::vector<std::uint16_t> load_u16(const std::string& name) {
  return mi::npy::load(kData / (name + ".npy")).to_vector<std::uint16_t>();
}

// Expects std::invalid_argument whose message contains `what`.
void expect_invalid(const std::function<void()>& f, const std::string& what) {
  try {
    f();
    ADD_FAILURE() << "no exception (expected std::invalid_argument containing \"" << what << "\")";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
  }
}

// ---------------------------------------------------------------- tests

// Varlen batch (lengths 1, 37, 64, 150; K with a per-channel offset) against float64 torch attention.
TEST(GpuAttentionQ8, MatchesExactAttentionOracle) {
  const auto cu = mi::npy::load(kData / "oracle_cu_seqlens.npy").to_vector<std::int32_t>();
  Problem p{2, 2, std::vector<int>(cu.begin(), cu.end())};
  const auto want = mi::npy::load(kData / "oracle_out.npy").to_vector<float>();
  ASSERT_EQ(static_cast<std::int64_t>(want.size()), p.total() * p.heads * kDim);
  Buffers buf(p);
  buf.upload(load_u16("oracle_q"), load_u16("oracle_k"), load_u16("oracle_v"));
  const std::size_t row = static_cast<std::size_t>(p.heads) * kDim;
  for (Variant v : kVariants) {
    if (!runnable(v)) continue;
    buf.run(p, {v});
    const auto got = buf.download_out(p);
    expect_within(measure(got, want, 0, got.size()), v, "vs float64 torch, all rows");
    for (int b = 0; b < p.batch(); ++b) {
      const Metrics m = measure(got, want, p.cu[b] * row, p.cu[b + 1] * row);
      EXPECT_GE(m.cos, tolerance(v).cos) << q8::variant_name(v) << " sequence " << b;
    }
  }
}

// The varlen port reproduces the upstream SageAttention kernels (fixed-length batch 2 x 160, outputs recorded with the
// upstream package). Measured: 94-96% of the outputs bit-identical, cos >= 0.999993; the rest differ by the K-mean
// summation order (torch's reduction vs the port's chunked one) and upstream's --use_fast_math division, which can
// move single INT8/e4m3 values by one step.
TEST(GpuAttentionQ8, MatchesUpstreamSageAttention) {
  const Problem p = make_problem(2, 2, {160, 160});
  Buffers buf(p);
  buf.upload(load_u16("upstream_q"), load_u16("upstream_k"), load_u16("upstream_v"));
  struct Case {
    const char* file;
    q8::Options options;
  };
  const Case cases[] = {{"int8_fp16", {Variant::kInt8Fp16}},
                        {"int8_fp16_f32acc", {Variant::kInt8Fp16F32Acc}},
                        {"int8_fp8_upstream", {Variant::kInt8Fp8, 448.f, 2.25f}},  // upstream's fp16-accumulation ranges
                        {"int8_fp8_f32acc", {Variant::kInt8Fp8F32Acc}}};
  for (const Case& c : cases) {
    if (!runnable(c.options.variant)) continue;
    buf.run(p, c.options);
    const auto got = buf.download_out(p);
    const auto up = load_u16(std::string("upstream_out_") + c.file);
    ASSERT_EQ(up.size(), got.size());
    std::vector<float> want(up.size());
    std::transform(up.begin(), up.end(), want.begin(), bf16_to_float);
    std::size_t equal = 0;
    for (std::size_t i = 0; i < got.size(); ++i) equal += got[i] == up[i];
    const Metrics m = measure(got, want, 0, got.size());
    std::printf("  %-18s vs upstream: cos %.9f  max|diff| %.3e  bit-equal %.2f%%\n", c.file, m.cos, m.max_abs,
                100.0 * static_cast<double>(equal) / static_cast<double>(got.size()));
    EXPECT_GE(m.cos, 0.99999) << c.file;
    EXPECT_LE(m.max_abs, 0.04) << c.file;
    // The upstream outputs were recorded on sm_89; upstream's fp16-accumulating variants give other bits on sm_120
    // (int8_fp16 84.8 %, int8_fp8_upstream 89.0 % identical to their sm_89 outputs), where the port then matches the
    // sm_89 file at 83-90 % (96.5 / 96.6 / 95.6 / 94.0 % against upstream run on sm_120). Cos and max|diff| hold everywhere.
    if (mi::testing::on_reference_gpu()) {
      EXPECT_GE(static_cast<double>(equal) / static_cast<double>(got.size()), 0.9) << c.file;
    }
  }
}

// Sequences do not interact: each sequence of a packed batch gives the same bits run alone, and changing a key/value
// row of one sequence changes only that sequence's rows.
TEST(GpuAttentionQ8, VarlenIsolation) {
  const Problem p = make_problem(4, 4, {100, 257, 64, 513});
  Buffers buf(p);
  buf.fill_random(p, 21);
  const std::int64_t row_elems = p.heads * kDim;
  for (Variant v : kVariants) {
    if (!runnable(v)) continue;
    buf.run(p, {v});
    const auto packed = buf.download_out(p);
    for (int b = 0; b < p.batch(); ++b) {
      const int n = p.cu[b + 1] - p.cu[b];
      const Problem one = make_problem(p.heads, p.kv_heads, {n});
      Buffers alone(one);
      for (auto [dst, src, h] : {std::tuple{&alone.q, &buf.q, p.heads}, std::tuple{&alone.k, &buf.k, p.kv_heads},
                                 std::tuple{&alone.v, &buf.v, p.kv_heads}}) {
        const std::size_t bytes = static_cast<std::size_t>(h) * kDim * 2;
        MI_CUDA_CHECK(cudaMemcpy(dst->get(), static_cast<char*>(src->get()) + p.cu[b] * bytes, n * bytes,
                                 cudaMemcpyDeviceToDevice));
      }
      alone.run(one, {v});
      const auto got = alone.download_out(one);
      EXPECT_EQ(0, std::memcmp(got.data(), packed.data() + p.cu[b] * row_elems, got.size() * 2))
          << q8::variant_name(v) << ": sequence " << b << " differs from the same sequence run alone";
    }
    // Overwrite key/value row `pos` of sequence 1 (every head) with the next row.
    Buffers mod(p);
    for (auto [dst, src] : {std::pair{&mod.q, &buf.q}, std::pair{&mod.k, &buf.k}, std::pair{&mod.v, &buf.v}}) {
      MI_CUDA_CHECK(cudaMemcpy(dst->get(), src->get(), src->size(), cudaMemcpyDeviceToDevice));
    }
    const int seq = 1, pos = p.cu[seq] + (p.cu[seq + 1] - p.cu[seq]) / 2;
    const std::size_t kv_row = static_cast<std::size_t>(p.kv_heads) * kDim * 2;
    for (auto* b : {&mod.k, &mod.v}) {
      auto* base = static_cast<char*>(b->get());
      MI_CUDA_CHECK(cudaMemcpy(base + pos * kv_row, base + (pos + 1) * kv_row, kv_row, cudaMemcpyDeviceToDevice));
    }
    mod.run(p, {v});
    const auto after = mod.download_out(p);
    for (int r = 0; r < p.total(); ++r) {
      const bool inside = r >= p.cu[seq] && r < p.cu[seq + 1];
      const bool changed = std::memcmp(&packed[static_cast<std::size_t>(r * row_elems)],
                                       &after[static_cast<std::size_t>(r * row_elems)], static_cast<std::size_t>(row_elems) * 2) != 0;
      if (inside) {
        EXPECT_TRUE(changed) << q8::variant_name(v) << ": row " << r << " ignores key " << pos;
      } else {
        ASSERT_FALSE(changed) << q8::variant_name(v) << ": row " << r << " sees key " << pos << " of another sequence";
      }
    }
  }
}

// Sequences shorter than one warp's 16/32 query rows and one 64-key tile, at the DiT's 24 heads.
TEST(GpuAttentionQ8, ShortSequencesMatchFp32Reference) {
  check_against_reference(make_problem(24, 24, {1, 2, 3, 5, 8, 15, 16, 17, 31, 32, 33, 47}), 7);
}

// Lengths around and between the 128-row query tile and the 64-key K/V tile (partial last tiles, sequence starts in
// the middle of a tile of the packed rows).
TEST(GpuAttentionQ8, LengthsNotTileMultiplesMatchFp32Reference) {
  check_against_reference(make_problem(4, 4, {63, 65, 127, 129, 191, 193, 255, 257, 1000, 77}), 11);
}

// A CFG-shaped pair at the DiT's 24 heads (long sequences: many K tiles, stats chunks and K-scale blocks).
TEST(GpuAttentionQ8, CfgPairMatchesFp32Reference) { check_against_reference(make_problem(24, 24, {2100, 1500}), 13); }

// Empty sequences in the batch and a max_seqlen above every length (extra CTAs and scale blocks must stay inert).
TEST(GpuAttentionQ8, EmptySequencesAndOversizedMaxSeqlen) {
  check_against_reference(make_problem(4, 4, {0, 70, 0, 130, 0}), 17, 1000);
}

// GQA: query head h reads KV head h / (heads / kv_heads), as flash_fwd.
TEST(GpuAttentionQ8, GqaMatchesFp32Reference) { check_against_reference(make_problem(8, 2, {100, 300}), 19); }

// q/k/v as strided views into a fused [tokens, 3, heads, 128] projection output (the DiT's QKV buffer) and an output
// with a padded row stride give the same bits as contiguous tensors.
TEST(GpuAttentionQ8, StridedQkvViewsMatchContiguous) {
  const Problem p = make_problem(4, 4, {77, 300});
  Buffers buf(p);
  buf.fill_random(p, 5);
  const std::int64_t n = p.total(), hd = p.heads * kDim;
  mi::DeviceBuffer fused(static_cast<std::size_t>(n * 3 * hd * 2)), wide_out(static_cast<std::size_t>(n * 2 * hd * 2));
  for (int i = 0; i < 3; ++i) {
    const auto& src = i == 0 ? buf.q : i == 1 ? buf.k : buf.v;
    MI_CUDA_CHECK(cudaMemcpy2D(static_cast<char*>(fused.get()) + i * hd * 2, 3 * hd * 2, src.get(), hd * 2, hd * 2, n,
                               cudaMemcpyDeviceToDevice));
  }
  const auto view = [&](void* base, std::int64_t row_stride) {
    Tensor t = Tensor::view(base, DType::kBF16, Device::kCUDA, {n, p.heads, kDim});
    t.stride[0] = row_stride;
    return t;
  };
  for (Variant v : kVariants) {
    if (!runnable(v)) continue;
    buf.run(p, {v});
    const auto want = buf.download_out(p);
    MI_CUDA_CHECK(cudaMemset(wide_out.get(), 0, wide_out.size()));
    auto* f = static_cast<char*>(fused.get());
    q8::attention_q8({v}, view(f, 3 * hd), view(f + hd * 2, 3 * hd), view(f + 2 * hd * 2, 3 * hd),
                     view(wide_out.get(), 2 * hd), buf.tcu(p), buf.tcu(p), p.max_seqlen(), p.scale(), buf.ws.get(),
                     buf.ws.size(), nullptr);
    std::vector<std::uint16_t> got(static_cast<std::size_t>(n * hd));
    MI_CUDA_CHECK(cudaMemcpy2D(got.data(), hd * 2, wide_out.get(), 2 * hd * 2, hd * 2, n, cudaMemcpyDeviceToHost));
    EXPECT_EQ(got, want) << q8::variant_name(v);
  }
}

// Run to run (and with a reused, dirty workspace) the output bits are identical: no atomics in any reduction.
TEST(GpuAttentionQ8, Deterministic) {
  const Problem p = make_problem(8, 8, {1500, 700});
  Buffers buf(p);
  buf.fill_random(p, 23);
  for (Variant v : kVariants) {
    if (!runnable(v)) continue;
    buf.run(p, {v});
    const auto first = buf.download_out(p);
    MI_CUDA_CHECK(cudaMemset(buf.ws.get(), 0x7f, buf.ws.size()));
    buf.run(p, {v});
    EXPECT_EQ(buf.download_out(p), first) << q8::variant_name(v);
  }
}

// FP8 ranges: the variant defaults, and a P/V range pair whose fp16 partial sums could overflow is refused. The
// overflow guard uses the e4m3-rounded maxima: 117 / 8.74 passes 64 * 117 * 8.74 = 65446 <= 65504, but P's maximum
// 2^6.870 = 116.97 rounds to 120 and V's 8.74 to 9 (64 * 120 * 9 = 69120). The range checks precede the device
// checks, so they hold on any GPU.
TEST(GpuAttentionQ8, Fp8RangesAndDefaults) {
  EXPECT_EQ(q8::effective_p_range({Variant::kInt8Fp8}), 112.f);
  EXPECT_EQ(q8::effective_v_range({Variant::kInt8Fp8}), 4.5f);
  EXPECT_EQ(q8::effective_p_range({Variant::kInt8Fp8F32Acc}), 448.f);
  EXPECT_EQ(q8::effective_v_range({Variant::kInt8Fp8F32Acc}), 448.f);
  EXPECT_EQ(q8::effective_p_range({Variant::kInt8Fp16}), 0.f);
  const Problem p = make_problem(2, 2, {64});
  Buffers buf(p);
  buf.fill_random(p, 3);
  EXPECT_THROW(buf.run(p, {Variant::kInt8Fp8, 448.f, 4.5f}), std::invalid_argument);  // 64 * 448 * 4.5 > 65504
  EXPECT_THROW(buf.run(p, {Variant::kInt8Fp8, 117.f, 8.74f}), std::invalid_argument);  // 64 * 120 * 9 after e4m3
  EXPECT_THROW(buf.run(p, {Variant::kInt8Fp8, 500.f, 1.f}), std::invalid_argument);   // above the e4m3 max
  if (runnable(Variant::kInt8Fp8F32Acc)) {
    EXPECT_NO_THROW(buf.run(p, {Variant::kInt8Fp8F32Acc, 448.f, 4.5f}));  // fp32 accumulation: fine
    EXPECT_NO_THROW(buf.run(p, {Variant::kInt8Fp8F32Acc, 117.f, 8.74f}));
  }
  if (runnable(Variant::kInt8Fp8)) {
    EXPECT_NO_THROW(buf.run(p, {Variant::kInt8Fp8, 448.f, 2.25f}));  // upstream's: 64 * 448 * 2.25 = 64512
    EXPECT_NO_THROW(buf.run(p, {Variant::kInt8Fp8, 112.f, 4.5f}));   // the default
  }
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

TEST(GpuAttentionQ8, RejectsUnsupportedArguments) {
  const Problem p = make_problem(2, 2, {64, 30});
  Buffers buf(p);
  const q8::Options o{runnable(Variant::kInt8Fp8) ? Variant::kInt8Fp8 : Variant::kInt8Fp16};
  const auto run = [&](const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, int max_seqlen, void* ws,
                       std::size_t ws_bytes) {
    q8::attention_q8(o, q, k, v, out, buf.tcu(p), buf.tcu(p), max_seqlen, p.scale(), ws, ws_bytes, nullptr);
  };
  const Tensor q = buf.tq(p), k = buf.tk(p), v = buf.tv(p), out = buf.tout(p);
  // head_dim 64.
  const Tensor q64 = Tensor::view(buf.q.get(), DType::kBF16, Device::kCUDA, {p.total(), p.heads * 2, 64});
  const Tensor k64 = Tensor::view(buf.k.get(), DType::kBF16, Device::kCUDA, {p.total(), p.kv_heads * 2, 64});
  const Tensor v64 = Tensor::view(buf.v.get(), DType::kBF16, Device::kCUDA, {p.total(), p.kv_heads * 2, 64});
  const Tensor o64 = Tensor::view(buf.out.get(), DType::kBF16, Device::kCUDA, {p.total(), p.heads * 2, 64});
  EXPECT_THROW(run(q64, k64, v64, o64, 64, buf.ws.get(), buf.ws.size()), std::invalid_argument);
  // fp16 input.
  Tensor qh = q;
  qh.dtype = DType::kF16;
  EXPECT_THROW(run(qh, k, v, out, 64, buf.ws.get(), buf.ws.size()), std::invalid_argument);
  // Misaligned head stride.
  Tensor qs = q;
  qs.stride[1] = 132;
  EXPECT_THROW(run(qs, k, v, out, 64, buf.ws.get(), buf.ws.size()), std::invalid_argument);
  // Workspace too small / misaligned.
  EXPECT_THROW(run(q, k, v, out, 64, buf.ws.get(), 1024), std::invalid_argument);
  EXPECT_THROW(run(q, k, v, out, 64, static_cast<char*>(buf.ws.get()) + 16, buf.ws.size() - 16), std::invalid_argument);
  // max_seqlen 0; heads not a multiple of kv heads.
  EXPECT_THROW(run(q, k, v, out, 0, buf.ws.get(), buf.ws.size()), std::invalid_argument);
  const Tensor k3 = Tensor::view(buf.k.get(), DType::kBF16, Device::kCUDA, {p.total(), 3, kDim});
  EXPECT_THROW(run(q, k3, k3, out, 64, buf.ws.get(), buf.ws.size()), std::invalid_argument);
  // scale <= 0 or not finite (the kernels mask with a raw-score sentinel and take the max of unscaled scores).
  for (float bad : {0.f, -0.088f, INFINITY, NAN}) {
    EXPECT_THROW(q8::attention_q8(o, q, k, v, out, buf.tcu(p), buf.tcu(p), 64, bad, buf.ws.get(), buf.ws.size(), nullptr),
                 std::invalid_argument)
        << "scale " << bad;
  }
  // The 32-bit offset limit counts the head stride: rows fit easily, but head 1 starts at element 2^31.
  Tensor q_far = q;
  q_far.stride[1] = std::int64_t{1} << 31;
  expect_invalid([&] { run(q_far, k, v, out, 64, buf.ws.get(), buf.ws.size()); }, "too many tokens");
  // Grid limits: num_heads (grid y) and batch (grid z) <= 65535. Views only: nothing is read.
  const Tensor q_wide = Tensor::view(buf.q.get(), DType::kBF16, Device::kCUDA, {1, 65536, kDim});
  const Tensor kv_wide = Tensor::view(buf.k.get(), DType::kBF16, Device::kCUDA, {1, 1, kDim});
  expect_invalid([&] { run(q_wide, kv_wide, kv_wide, q_wide, 64, buf.ws.get(), buf.ws.size()); }, "65535");
  mi::DeviceBuffer big_cu(65537 * sizeof(int));
  const Tensor cu_big = Tensor::view(big_cu.get(), DType::kI32, Device::kCUDA, {65537});
  expect_invalid([&] {
    q8::attention_q8(o, q, k, v, out, cu_big, cu_big, 64, p.scale(), buf.ws.get(), buf.ws.size(), nullptr);
  }, "65535");
  // Debug checks catch a too-small max_seqlen.
  const bool debug_checks = mi::varlen_debug_checks();
  mi::set_varlen_debug_checks(true);
  EXPECT_THROW(run(q, k, v, out, 32, buf.ws.get(), buf.ws.size()), std::invalid_argument);
  mi::set_varlen_debug_checks(debug_checks);
  EXPECT_NO_THROW(run(q, k, v, out, 64, buf.ws.get(), buf.ws.size()));
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// An all-zero V channel (and a whole all-zero V head) gives finite output that matches the reference: the FP8
// variants' per-channel V scale of an all-zero channel is 1 (upstream divides by its amax 0 and writes NaN).
TEST(GpuAttentionQ8, ZeroVChannelMatchesFp32Reference) {
  const Problem p = make_problem(4, 4, {100, 333});
  check_against_reference(p, 29, -1, [&](const Buffers& buf) {
    const std::size_t pitch = static_cast<std::size_t>(p.kv_heads) * kDim * 2;
    auto* v = static_cast<char*>(buf.v.get());
    MI_CUDA_CHECK(cudaMemset2D(v + 5 * 2, pitch, 0, 2, static_cast<std::size_t>(p.total())));            // head 0, channel 5
    MI_CUDA_CHECK(cudaMemset2D(v + 2 * kDim * 2, pitch, 0, kDim * 2, static_cast<std::size_t>(p.total())));  // all of head 2
  });
}

// V beyond the fp16 PV limits (attention_q8.h): with debug checks, int8_fp16 refuses max|V| > 1023.5 and
// int8_fp16_f32acc max|V| > 65504; inside its limit a variant matches the reference, and the FP8 variants (V scaled
// per channel) take any finite V.
TEST(GpuAttentionQ8, LargeVLimits) {
  const Problem p = make_problem(4, 4, {200, 90});
  const bool debug_checks = mi::varlen_debug_checks();
  mi::set_varlen_debug_checks(true);
  for (float v_std : {300.f, 30000.f}) {  // max|V| ~ 1500 and ~ 150000
    Buffers buf(p);
    buf.fill_random(p, 31);
    fill_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(buf.v.get()), p.total() * p.kv_heads * kDim, p.kv_heads * kDim,
                             33, v_std, 0.f);
    MI_CUDA_CHECK_LAUNCH();
    const auto want = reference(p, buf);
    for (Variant v : kVariants) {
      if (!runnable(v)) continue;
      const bool over = v == Variant::kInt8Fp16 || (v == Variant::kInt8Fp16F32Acc && v_std > 1000.f);
      if (over) {
        expect_invalid([&] { buf.run(p, {v}); }, "max|V|");
        continue;
      }
      buf.run(p, {v});
      const auto got = buf.download_out(p);
      const Metrics m = measure(got, want, 0, got.size());
      std::printf("  %-18s V std %-8g cos %.7f  rel L2 %.3e  max|err|/std %.3e\n", q8::variant_name(v), v_std, m.cos,
                  m.rel_l2, m.max_abs / v_std);
      EXPECT_GE(m.cos, tolerance(v).cos) << q8::variant_name(v) << " V std " << v_std;
      EXPECT_LE(m.rel_l2, tolerance(v).rel_l2) << q8::variant_name(v) << " V std " << v_std;
    }
  }
  mi::set_varlen_debug_checks(debug_checks);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// total_q == 0: nothing runs, so a null workspace of 0 bytes is accepted.
TEST(GpuAttentionQ8, NoQueryRowsAcceptsNullWorkspace) {
  const Problem p = make_problem(2, 2, {0, 0});
  Buffers buf(p);
  for (Variant v : kVariants) {
    EXPECT_NO_THROW(q8::attention_q8({v}, buf.tq(p), buf.tk(p), buf.tv(p), buf.tout(p), buf.tcu(p), buf.tcu(p), 1,
                                     p.scale(), nullptr, 0, nullptr))
        << q8::variant_name(v);
  }
}

// supported() follows the documented requirements, and a variant this device or build cannot run throws
// std::runtime_error instead of launching (skipped where every variant runs, e.g. sm_89 with nvcc >= 12.8).
__global__ void arch_probe_kernel() {}

TEST(GpuAttentionQ8, UnsupportedVariantsThrow) {
  constexpr bool kNvcc128 = __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ >= 1208;
  const int cc = device_cc();
  // This binary is built for the library's CUDA architectures: the image a kernel of it loads on this device is
  // the one the library's kernels load.
  cudaFuncAttributes attr{};
  MI_CUDA_CHECK(cudaFuncGetAttributes(&attr, arch_probe_kernel));
  const int image = std::min(attr.ptxVersion, attr.binaryVersion);
  const Problem p = make_problem(2, 2, {64});
  Buffers buf(p);
  buf.fill_random(p, 37);
  bool all = true;
  for (Variant v : kVariants) {
    const int need = is_fp8(v) ? 89 : 80;
    const bool expect = cc >= need && image >= need && (v != Variant::kInt8Fp8 || kNvcc128);
    EXPECT_EQ(q8::supported(v), expect) << q8::variant_name(v);
    if (!expect) {
      all = false;
      EXPECT_THROW(buf.run(p, {v}), std::runtime_error) << q8::variant_name(v);
    }
  }
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  if (all) GTEST_SKIP() << "every variant runs on this device (sm_" << cc << ", image sm_" << image << ") and build";
}

}  // namespace
