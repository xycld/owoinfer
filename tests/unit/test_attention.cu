// FlashAttention-2 forward (src/kernels/attention) against torch SDPA vectors (tests/data/attention, generated with
// the PyTorch reference) and against a plain fp32 reference kernel at the full
// MOSS-Image shapes, plus varlen/causal isolation, strided views, argument checks and throughput.
#include <gtest/gtest.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "core/varlen.h"
#include "kernels/attention/flash_attention.h"
#include "util/npy.h"

namespace {

namespace fs = std::filesystem;
using mi::DType;
using mi::Device;
using mi::Tensor;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/attention";

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

// Deterministic N(0, scale^2) bf16 values (Box-Muller on a counter hash).
__global__ void fill_normal_bf16(__nv_bfloat16* out, std::int64_t n, std::uint64_t seed, float scale) {
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::uint64_t key = seed * 0x9E3779B97F4A7C15ull + static_cast<std::uint64_t>(i) * 2;
    const float u1 = (static_cast<float>(hash_u32(key) >> 8) + 1.f) * (1.f / 16777217.f);
    const float u2 = static_cast<float>(hash_u32(key + 1) >> 8) * (1.f / 16777216.f);
    out[i] = __float2bfloat16(scale * sqrtf(-2.f * logf(u1)) * cospif(2.f * u2));
  }
}

// Exact-math reference: one block per (query row, head), blockDim == head_dim. Scores, softmax and the
// P·V product all in fp32 (softmax sum in fp64), P never rounded to bf16. Contiguous [tokens, heads, d].
__global__ void reference_attention(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                                    float* out, const int* rows, const int* cu_seqlens, int batch, int heads,
                                    int kv_heads, bool causal, float scale) {
  extern __shared__ float smem[];
  const int d = blockDim.x, t = threadIdx.x;
  const int row = rows[blockIdx.x], h = blockIdx.y, kvh = h / (heads / kv_heads);
  float* sq = smem;          // [d]
  float* scores = smem + d;  // [keys]
  __shared__ float red[32];
  __shared__ double red_d[32];

  int b = 0;
  while (b + 1 < batch && cu_seqlens[b + 1] <= row) ++b;
  const int k0 = cu_seqlens[b];
  const int n = causal ? row - k0 + 1 : cu_seqlens[b + 1] - k0;

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
  for (int j = 0; j < n; ++j) {
    acc = fmaf(scores[j], __bfloat162float(v[(static_cast<std::int64_t>(k0 + j) * kv_heads + kvh) * d + t]), acc);
  }
  out[(static_cast<std::int64_t>(blockIdx.x) * heads + h) * d + t] = static_cast<float>(acc / sum);
}

// ---------------------------------------------------------------- host helpers

struct Problem {
  int heads = 0, kv_heads = 0, dim = 0;
  bool causal = false;
  std::vector<int> cu;  // [batch + 1]
  std::int64_t total() const { return cu.back(); }
  int batch() const { return static_cast<int>(cu.size()) - 1; }
  int max_seqlen() const {
    int m = 0;
    for (int b = 0; b < batch(); ++b) m = std::max(m, cu[b + 1] - cu[b]);
    return m;
  }
  float scale() const { return 1.f / std::sqrt(static_cast<float>(dim)); }
};

Problem make_problem(int heads, int kv_heads, int dim, bool causal, const std::vector<int>& seqlens) {
  Problem p{heads, kv_heads, dim, causal, {0}};
  for (int s : seqlens) p.cu.push_back(p.cu.back() + s);
  return p;
}

// Device buffers for one problem: q/k/v/out contiguous bf16, cu_seqlens, workspace.
struct Buffers {
  explicit Buffers(const Problem& p)
      : q(p.total() * p.heads * p.dim * 2),
        k(p.total() * p.kv_heads * p.dim * 2),
        v(p.total() * p.kv_heads * p.dim * 2),
        out(p.total() * p.heads * p.dim * 2),
        cu(p.cu.size() * sizeof(int)),
        ws(mi::attention::flash_fwd_workspace_bytes(p.total(), p.heads)) {
    cu.upload(p.cu.data(), p.cu.size() * sizeof(int));
  }
  Tensor tq(const Problem& p) const { return Tensor::view(q.get(), DType::kBF16, Device::kCUDA, {p.total(), p.heads, p.dim}); }
  Tensor tk(const Problem& p) const { return Tensor::view(k.get(), DType::kBF16, Device::kCUDA, {p.total(), p.kv_heads, p.dim}); }
  Tensor tv(const Problem& p) const { return Tensor::view(v.get(), DType::kBF16, Device::kCUDA, {p.total(), p.kv_heads, p.dim}); }
  Tensor tout(const Problem& p) const { return Tensor::view(out.get(), DType::kBF16, Device::kCUDA, {p.total(), p.heads, p.dim}); }
  Tensor tcu(const Problem& p) const { return Tensor::view(cu.get(), DType::kI32, Device::kCUDA, {static_cast<std::int64_t>(p.cu.size())}); }

  // max_seqlen < 0: the exact longest sequence.
  void run(const Problem& p, cudaStream_t stream = nullptr, int max_seqlen = -1) const {
    mi::attention::flash_fwd(tq(p), tk(p), tv(p), tout(p), tcu(p), tcu(p), max_seqlen < 0 ? p.max_seqlen() : max_seqlen,
                             p.causal, p.scale(), ws.get(), ws.size(), stream);
  }
  void fill_random(const Problem& p, std::uint64_t seed) const {
    fill_normal_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(q.get()), p.total() * p.heads * p.dim, seed, 1.5f);
    fill_normal_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(k.get()), p.total() * p.kv_heads * p.dim, seed + 1, 1.5f);
    fill_normal_bf16<<<1024, 256>>>(static_cast<__nv_bfloat16*>(v.get()), p.total() * p.kv_heads * p.dim, seed + 2, 1.f);
    MI_CUDA_CHECK_LAUNCH();
  }
  std::vector<std::uint16_t> download_out(const Problem& p) const {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(p.total() * p.heads * p.dim));
    out.download(h.data(), h.size() * 2);
    return h;
  }

  mi::DeviceBuffer q, k, v, out, cu, ws;
};

// fp32 reference output for the given query rows: [rows.size(), heads, dim].
std::vector<float> reference_rows(const Problem& p, const Buffers& buf, const std::vector<int>& rows) {
  mi::DeviceBuffer d_rows(rows.size() * sizeof(int)), d_out(rows.size() * p.heads * p.dim * sizeof(float));
  d_rows.upload(rows.data(), rows.size() * sizeof(int));
  const int smem = (p.dim + p.max_seqlen()) * static_cast<int>(sizeof(float));
  MI_CUDA_CHECK(cudaFuncSetAttribute(reference_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  reference_attention<<<dim3(static_cast<unsigned>(rows.size()), static_cast<unsigned>(p.heads)), p.dim, smem>>>(
      static_cast<const __nv_bfloat16*>(buf.q.get()), static_cast<const __nv_bfloat16*>(buf.k.get()),
      static_cast<const __nv_bfloat16*>(buf.v.get()), static_cast<float*>(d_out.get()),
      static_cast<const int*>(d_rows.get()), static_cast<const int*>(buf.cu.get()), p.batch(), p.heads, p.kv_heads,
      p.causal, p.scale());
  MI_CUDA_CHECK_LAUNCH();
  std::vector<float> h(rows.size() * p.heads * p.dim);
  d_out.download(h.data(), h.size() * sizeof(float));
  return h;
}

struct Diff {
  double cosine = 0, max_abs = 0;
  std::size_t bit_equal = 0, n = 0;
};

// got: bf16 output rows (all rows), compared on `rows` against want ([rows.size(), heads*dim] fp32).
Diff compare(const std::vector<std::uint16_t>& got, const std::vector<float>& want, const std::vector<int>& rows,
             std::int64_t row_elems) {
  Diff d;
  double dot = 0, ng = 0, nw = 0;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::int64_t i = 0; i < row_elems; ++i) {
      const double g = bf16_to_float(got[static_cast<std::size_t>(rows[r] * row_elems + i)]);
      const double w = want[static_cast<std::size_t>(static_cast<std::int64_t>(r) * row_elems + i)];
      dot += g * w;
      ng += g * g;
      nw += w * w;
      d.max_abs = std::max(d.max_abs, std::abs(g - w));
      d.bit_equal += (g == w);
      ++d.n;
    }
  }
  d.cosine = dot / std::sqrt(ng * nw);
  return d;
}

std::vector<int> all_rows(const Problem& p) {
  std::vector<int> r(static_cast<std::size_t>(p.total()));
  std::iota(r.begin(), r.end(), 0);
  return r;
}

// The first and last 130 rows of every sequence (tile and boundary effects) plus every 13th row.
std::vector<int> sampled_rows(const Problem& p) {
  std::vector<int> r;
  for (int b = 0; b < p.batch(); ++b) {
    for (int i = p.cu[b]; i < p.cu[b + 1]; ++i) {
      if (i - p.cu[b] < 130 || p.cu[b + 1] - i <= 130 || i % 13 == 0) r.push_back(i);
    }
  }
  return r;
}

void report(const char* what, const Diff& d) {
  std::printf("  %-34s cosine %.9f  max|diff| %.3g  bit-equal %.2f%%\n", what, d.cosine, d.max_abs,
              100.0 * static_cast<double>(d.bit_equal) / static_cast<double>(d.n));
}

// ---------------------------------------------------------------- torch vectors

void check_torch_case(const std::string& name, int heads, int kv_heads, int dim, bool causal) {
  auto load_u16 = [&](const std::string& key) { return mi::npy::load(kData / (name + "_" + key + ".npy")).to_vector<std::uint16_t>(); };
  const auto cu = mi::npy::load(kData / (name + "_cu_seqlens.npy")).to_vector<std::int32_t>();
  Problem p{heads, kv_heads, dim, causal, std::vector<int>(cu.begin(), cu.end())};
  const auto q = load_u16("q"), k = load_u16("k"), v = load_u16("v"), want = load_u16("out");
  ASSERT_EQ(static_cast<std::int64_t>(q.size()), p.total() * heads * dim);
  ASSERT_EQ(static_cast<std::int64_t>(k.size()), p.total() * kv_heads * dim);
  Buffers buf(p);
  buf.q.upload(q.data(), q.size() * 2);
  buf.k.upload(k.data(), k.size() * 2);
  buf.v.upload(v.data(), v.size() * 2);
  buf.run(p);
  const auto got = buf.download_out(p);

  std::vector<float> want_f(want.size());
  std::transform(want.begin(), want.end(), want_f.begin(), bf16_to_float);
  const auto rows = all_rows(p);
  const Diff vs_torch = compare(got, want_f, rows, heads * dim);
  const Diff vs_ref = compare(got, reference_rows(p, buf, rows), rows, heads * dim);
  const Diff torch_vs_ref = compare(want, reference_rows(p, buf, rows), rows, heads * dim);
  std::printf("%s (max_seqlen %d, batch %d):\n", name.c_str(), p.max_seqlen(), p.batch());
  report("flash_fwd vs torch SDPA", vs_torch);
  report("flash_fwd vs fp32 reference", vs_ref);
  report("torch SDPA vs fp32 reference", torch_vs_ref);
  EXPECT_GE(vs_torch.cosine, 0.99999);
  EXPECT_GE(vs_ref.cosine, 0.99999);
  // Per-sequence check: a varlen boundary or causal-mask error would show up in one sequence only.
  for (int b = 0; b < p.batch(); ++b) {
    std::vector<int> seq_rows(rows.begin() + p.cu[b], rows.begin() + p.cu[b + 1]);
    std::vector<float> seq_want(want_f.begin() + static_cast<std::ptrdiff_t>(p.cu[b]) * heads * dim,
                                want_f.begin() + static_cast<std::ptrdiff_t>(p.cu[b + 1]) * heads * dim);
    EXPECT_GE(compare(got, seq_want, seq_rows, heads * dim).cosine, 0.99999) << "sequence " << b;
  }
}

TEST(GpuAttention, DitVarlenMatchesTorchSdpa) { check_torch_case("dit", 4, 4, 128, false); }

TEST(GpuAttention, QwenCausalGqaVarlenMatchesTorchSdpa) { check_torch_case("qwen", 8, 2, 256, true); }

// ---------------------------------------------------------------- full shapes vs fp32 reference

void check_full_shape(const Problem& p, std::uint64_t seed, const char* name) {
  Buffers buf(p);
  buf.fill_random(p, seed);
  buf.run(p);
  const auto got = buf.download_out(p);
  const auto rows = sampled_rows(p);
  const Diff d = compare(got, reference_rows(p, buf, rows), rows, p.heads * p.dim);
  std::printf("%s (%zu sampled rows):\n", name, rows.size());
  report("flash_fwd vs fp32 reference", d);
  ::testing::Test::RecordProperty("max_abs_diff", std::to_string(d.max_abs));
  EXPECT_GE(d.cosine, 0.99999);
}

// The DiT CFG pair at 1456x1456: cond 8281 + 4096 text tokens, uncond 8281 + 16, one varlen launch.
TEST(GpuAttention, DitCfgPairMatchesFp32Reference) {
  check_full_shape(make_problem(24, 24, 128, false, {8281 + 4096, 8281 + 16}), 7, "dit 24h d128 [12377, 8297]");
}

// Text refiner (text tokens only) and odd lengths, incl. L = 1.
TEST(GpuAttention, DitShortSequencesMatchFp32Reference) {
  check_full_shape(make_problem(24, 24, 128, false, {1, 31, 32, 33, 127, 128, 129, 4096, 16}), 8, "dit 24h d128 short");
}

// Qwen3.5 full attention: prompt and negative prompt as one varlen launch.
TEST(GpuAttention, QwenFullShapeMatchesFp32Reference) {
  check_full_shape(make_problem(16, 4, 256, true, {4096, 1, 16, 65, 1000}), 9, "qwen 16/4h d256 causal");
}

// ---------------------------------------------------------------- isolation, views, arguments

// Perturbing key/value rows must change exactly the outputs that may attend to them: later rows of the
// same sequence when causal, the whole sequence otherwise, and nothing in other sequences (bit-exact).
void check_isolation(const Problem& p) {
  Buffers buf(p);
  buf.fill_random(p, 21);
  buf.run(p);
  const auto before = buf.download_out(p);
  // Perturb key/value row `pos` (all KV heads) by overwriting it with the next row.
  const int seq = 1, pos = p.cu[seq] + (p.cu[seq + 1] - p.cu[seq]) / 2;
  const std::size_t row_bytes = static_cast<std::size_t>(p.kv_heads * p.dim * 2);
  for (auto* b : {&buf.k, &buf.v}) {
    auto* base = static_cast<char*>(b->get());
    MI_CUDA_CHECK(cudaMemcpy(base + pos * row_bytes, base + (pos + 1) * row_bytes, row_bytes, cudaMemcpyDeviceToDevice));
  }
  buf.run(p);
  const auto after = buf.download_out(p);
  const std::int64_t row_elems = p.heads * p.dim;
  for (int r = 0; r < p.total(); ++r) {
    const bool may_change = r >= p.cu[seq] && r < p.cu[seq + 1] && (!p.causal || r >= pos);
    const bool changed = std::memcmp(&before[static_cast<std::size_t>(r * row_elems)], &after[static_cast<std::size_t>(r * row_elems)],
                                     static_cast<std::size_t>(row_elems) * 2) != 0;
    if (may_change) {
      EXPECT_TRUE(changed) << "row " << r << " ignores key " << pos;
    } else {
      ASSERT_FALSE(changed) << "row " << r << " sees key " << pos << (p.causal ? " (causal)" : "");
    }
  }
}

TEST(GpuAttention, NonCausalVarlenIsolation) { check_isolation(make_problem(4, 4, 128, false, {100, 257, 64})); }

TEST(GpuAttention, CausalVarlenIsolation) { check_isolation(make_problem(8, 2, 256, true, {100, 257, 64})); }

// q/k/v as strided views into a fused [tokens, 3, heads, d] projection output give the same bits.
TEST(GpuAttention, StridedQkvViewsMatchContiguous) {
  const Problem p = make_problem(4, 4, 128, false, {77, 300});
  Buffers buf(p);
  buf.fill_random(p, 5);
  buf.run(p);
  const auto want = buf.download_out(p);

  const std::int64_t n = p.total(), hd = p.heads * p.dim;
  mi::DeviceBuffer fused(static_cast<std::size_t>(n * 3 * hd * 2));
  for (int i = 0; i < 3; ++i) {
    const auto& src = i == 0 ? buf.q : i == 1 ? buf.k : buf.v;
    MI_CUDA_CHECK(cudaMemcpy2D(static_cast<char*>(fused.get()) + i * hd * 2, 3 * hd * 2, src.get(), hd * 2, hd * 2, n,
                               cudaMemcpyDeviceToDevice));
  }
  auto view = [&](int i) {
    Tensor t = Tensor::view(static_cast<char*>(fused.get()) + i * hd * 2, DType::kBF16, Device::kCUDA, {n, p.heads, p.dim});
    t.stride[0] = 3 * hd;
    return t;
  };
  MI_CUDA_CHECK(cudaMemset(buf.out.get(), 0, buf.out.size()));
  mi::attention::flash_fwd(view(0), view(1), view(2), buf.tout(p), buf.tcu(p), buf.tcu(p), p.max_seqlen(), false,
                           p.scale(), buf.ws.get(), buf.ws.size(), nullptr);
  EXPECT_EQ(buf.download_out(p), want);
}

// ---------------------------------------------------------------- tile coverage, edge cases

// Restores automatic tile dispatch (and the debug-check switch) when a test ends.
struct ResetOverrides {
  ~ResetOverrides() {
    mi::attention::flash_fwd_set_tile_override_for_testing(mi::attention::FlashTile::kAuto);
    mi::set_varlen_debug_checks(false);
  }
};

int max_smem_per_block() {
  int device = 0, v = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  return v;
}

// All four FA2 instantiations against the fp32 reference. The auto dispatcher picks only one of the
// two tiles per head_dim on a given GPU; the override runs the other too. hdim256 128x64 needs 128 KiB
// of shared memory per block: sm_89 allows 99 KiB, so there it must be rejected (and is only runnable on
// sm_80/90-class parts), which this test asserts instead of skipping.
void check_forced_tile(mi::attention::FlashTile tile, int need_smem, const Problem& p, std::uint64_t seed) {
  ResetOverrides reset;
  Buffers buf(p);
  buf.fill_random(p, seed);
  mi::attention::flash_fwd_set_tile_override_for_testing(tile);
  if (max_smem_per_block() < need_smem) {
    EXPECT_THROW(buf.run(p), std::invalid_argument);
    return;
  }
  buf.run(p);
  const auto got = buf.download_out(p);
  const auto rows = sampled_rows(p);
  const Diff d = compare(got, reference_rows(p, buf, rows), rows, p.heads * p.dim);
  report("forced tile vs fp32 reference", d);
  EXPECT_GE(d.cosine, 0.99999);
}

TEST(GpuAttention, Hdim128Tile128x64) {
  check_forced_tile(mi::attention::FlashTile::kHdim128_128x64, 64 * 1024, make_problem(4, 4, 128, false, {100, 257, 64, 1}), 11);
}
TEST(GpuAttention, Hdim128Tile128x32) {
  check_forced_tile(mi::attention::FlashTile::kHdim128_128x32, 48 * 1024, make_problem(4, 4, 128, false, {100, 257, 64, 1}), 12);
}
TEST(GpuAttention, Hdim256Tile64x64) {
  check_forced_tile(mi::attention::FlashTile::kHdim256_64x64, 96 * 1024, make_problem(8, 2, 256, true, {100, 257, 64, 1}), 13);
}
TEST(GpuAttention, Hdim256Tile128x64) {
  check_forced_tile(mi::attention::FlashTile::kHdim256_128x64, 128 * 1024, make_problem(8, 2, 256, true, {100, 257, 64, 1}), 14);
}

TEST(GpuAttention, ForcedTileMustMatchHeadDim) {
  ResetOverrides reset;
  const Problem p = make_problem(4, 4, 128, false, {16});
  Buffers buf(p);
  mi::attention::flash_fwd_set_tile_override_for_testing(mi::attention::FlashTile::kHdim256_64x64);
  EXPECT_THROW(buf.run(p), std::invalid_argument);
}

// Empty sequences (first, middle, last) must neither crash nor disturb their neighbours; a max_seqlen
// larger than the longest sequence only launches extra, immediately-exiting blocks.
void check_empty_and_oversized(const Problem& p, std::uint64_t seed) {
  Buffers buf(p);
  buf.fill_random(p, seed);
  const auto rows = all_rows(p);
  const auto want = reference_rows(p, buf, rows);
  for (int max_seqlen : {p.max_seqlen(), p.max_seqlen() + 1, p.max_seqlen() + 300}) {
    MI_CUDA_CHECK(cudaMemset(buf.out.get(), 0, buf.out.size()));
    buf.run(p, nullptr, max_seqlen);
    const Diff d = compare(buf.download_out(p), want, rows, p.heads * p.dim);
    EXPECT_GE(d.cosine, 0.99999) << "max_seqlen " << max_seqlen;
    EXPECT_EQ(d.n, rows.size() * static_cast<std::size_t>(p.heads * p.dim));
  }
}

TEST(GpuAttention, EmptySequencesAndOversizedMaxSeqlenNonCausal) {
  check_empty_and_oversized(make_problem(4, 4, 128, false, {0, 100, 0, 37, 0}), 21);
}
TEST(GpuAttention, EmptySequencesAndOversizedMaxSeqlenCausal) {
  check_empty_and_oversized(make_problem(8, 2, 256, true, {0, 100, 0, 37, 0}), 22);
}

// The opt-in contract check catches a max_seqlen below the longest sequence and a cu_seqlens that does
// not end at the row count; the default path does not look (silent truncation, as documented).
TEST(GpuAttention, DebugChecksCatchContractViolations) {
  ResetOverrides reset;
  const Problem p = make_problem(4, 4, 128, false, {40, 90});
  Buffers buf(p);
  buf.fill_random(p, 23);
  EXPECT_NO_THROW(buf.run(p, nullptr, 500));
  mi::set_varlen_debug_checks(true);
  EXPECT_NO_THROW(buf.run(p));
  EXPECT_NO_THROW(buf.run(p, nullptr, 500));
  EXPECT_THROW(buf.run(p, nullptr, 89), std::invalid_argument);
  Problem shorter = p;
  shorter.cu.back() -= 1;
  mi::DeviceBuffer cu(shorter.cu.size() * sizeof(int));
  cu.upload(shorter.cu.data(), shorter.cu.size() * sizeof(int));
  const Tensor tcu = Tensor::view(cu.get(), DType::kI32, Device::kCUDA, {static_cast<std::int64_t>(shorter.cu.size())});
  EXPECT_THROW(mi::attention::flash_fwd(buf.tq(p), buf.tk(p), buf.tv(p), buf.tout(p), tcu, tcu, 90, false, p.scale(),
                                        buf.ws.get(), buf.ws.size(), nullptr),
               std::invalid_argument);
}

TEST(GpuAttention, RejectsUnsupportedArguments) {
  Problem p = make_problem(4, 4, 128, false, {16});
  Buffers buf(p);
  auto run = [&](const Tensor& q, const Tensor& k, bool causal, std::size_t ws_bytes) {
    mi::attention::flash_fwd(q, k, k, q, buf.tcu(p), buf.tcu(p), 16, causal, 0.1f, buf.ws.get(), ws_bytes, nullptr);
  };
  const Tensor q = buf.tq(p), k = buf.tk(p);
  EXPECT_THROW(run(q, k, true, buf.ws.size()), std::invalid_argument);       // hdim 128 causal not built
  EXPECT_THROW(run(q, k, false, buf.ws.size() - 4), std::invalid_argument);  // workspace too small
  Tensor odd = Tensor::view(buf.k.get(), DType::kBF16, Device::kCUDA, {16, 3, 128});
  EXPECT_THROW(run(q, odd, false, buf.ws.size()), std::invalid_argument);    // 4 heads vs 3 KV heads
  Tensor f32 = q;
  f32.dtype = DType::kF32;
  EXPECT_THROW(run(f32, k, false, buf.ws.size()), std::invalid_argument);
  EXPECT_EQ(mi::attention::flash_fwd_workspace_bytes(20674, 24), 20674u * 24u * 4u);
}

// ---------------------------------------------------------------- throughput

double time_ms(const Buffers& buf, const Problem& p, int iters = 20) {
  for (int i = 0; i < 10; ++i) buf.run(p);  // also lets the clocks settle
  cudaEvent_t start, stop;
  MI_CUDA_CHECK(cudaEventCreate(&start));
  MI_CUDA_CHECK(cudaEventCreate(&stop));
  MI_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) buf.run(p);
  MI_CUDA_CHECK(cudaEventRecord(stop));
  MI_CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  MI_CUDA_CHECK(cudaEventDestroy(start));
  MI_CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// Records TFLOPS (4·L²·H·d per sequence) for the DiT shapes; compare with torch SDPA (flash backend) on the same
// GPU.
TEST(GpuAttentionPerf, DitShapes) {
  const std::vector<std::vector<int>> cases = {{8281 + 300}, {8281 + 4096}, {8281 + 4096, 8281 + 16}};
  for (const auto& seqlens : cases) {
    const Problem p = make_problem(24, 24, 128, false, seqlens);
    Buffers buf(p);
    buf.fill_random(p, 3);
    const double ms = time_ms(buf, p);
    double flops = 0;
    for (int s : seqlens) flops += 4.0 * s * s * p.heads * p.dim;
    std::string name = "L";
    for (int s : seqlens) name += "_" + std::to_string(s);
    std::printf("  dit %-18s %.3f ms  %.1f TFLOPS\n", name.c_str(), ms, flops / ms / 1e9);
    RecordProperty(name + "_tflops", std::to_string(flops / ms / 1e9));
  }
}

}  // namespace
