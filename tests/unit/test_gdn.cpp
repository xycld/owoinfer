// GatedDeltaNet kernels vs the transformers pure-torch fallback on real Qwen3.5 layer-0 activations
// (generated with the PyTorch reference into tests/data/gdn/), plus the token-by-token oracle kernel.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "core/tensor.h"
#include "kernels/gdn/gdn.h"
#include "test_util.h"
#include "util/npy.h"

namespace {

using mi::DType;
using mi::Device;
using mi::DeviceBuffer;
using mi::Tensor;
namespace gdn = mi::gdn;

const std::filesystem::path kData = std::filesystem::path(MOSSINFER_SOURCE_DIR) / "tests/data/gdn";

template <typename T>
std::vector<T> load(const std::string& name) {
  return mi::npy::load(kData / name).to_vector<T>();
}

float bf16_to_float(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

std::uint32_t float_bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

struct Stats {
  double cos = 0, max_abs = 0, max_ref = 0, rel_l2 = 0;
  double rel_to_scale() const { return max_ref > 0 ? max_abs / max_ref : 0; }
  std::string str() const {
    char buf[160];
    std::snprintf(buf, sizeof buf, "cos=%.9f max_abs=%.3e max_abs/max|ref|=%.3e rel_l2=%.3e", cos, max_abs,
                  rel_to_scale(), rel_l2);
    return buf;
  }
};

Stats compare(const std::vector<float>& got, const std::vector<float>& ref) {
  EXPECT_EQ(got.size(), ref.size());
  double dot = 0, gg = 0, rr = 0, dd = 0;
  Stats s;
  for (std::size_t i = 0; i < std::min(got.size(), ref.size()); ++i) {
    const double a = got[i], b = ref[i];
    dot += a * b;
    gg += a * a;
    rr += b * b;
    dd += (a - b) * (a - b);
    s.max_abs = std::max(s.max_abs, std::abs(a - b));
    s.max_ref = std::max(s.max_ref, std::abs(b));
  }
  s.cos = dot / std::sqrt(gg * rr);
  s.rel_l2 = std::sqrt(dd / rr);
  return s;
}

std::vector<float> to_float(const std::vector<std::uint16_t>& bits) {
  std::vector<float> f(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) f[i] = bf16_to_float(bits[i]);
  return f;
}

// fp32 -> bf16 (round to nearest even) -> fp32, as the fallback's final `.to(bfloat16)` (finite inputs).
std::vector<float> round_bf16(std::vector<float> v) {
  for (float& f : v) {
    std::uint32_t u = float_bits(f);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&f, &u, sizeof f);
  }
  return v;
}

template <typename T>
std::vector<T> download(const DeviceBuffer& buf, std::size_t n) {
  std::vector<T> v(n);
  buf.download(v.data(), n * sizeof(T));
  return v;
}

template <typename T>
DeviceBuffer upload(const std::vector<T>& v) {
  DeviceBuffer b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}

// Layer-0 weights and the per-token input tables.
struct Tables {
  std::vector<std::uint16_t> qkv, a, b, conv_w, a_log, dt_bias;
  static const Tables& get() {
    static const Tables t{load<std::uint16_t>("table_qkv.npy"), load<std::uint16_t>("table_a.npy"),
                          load<std::uint16_t>("table_b.npy"),   load<std::uint16_t>("conv_weight.npy"),
                          load<std::uint16_t>("a_log.npy"),     load<std::uint16_t>("dt_bias.npy")};
    return t;
  }
};

// One case on the device: inputs expanded from the tables (qkv rows optionally padded to a wider row stride,
// as a fused projection output would be), and buffers for every stage.
struct Case {
  std::string name;
  std::vector<std::int32_t> cu;
  std::int64_t tokens = 0;
  int max_seqlen = 0;
  std::int64_t qkv_stride = gdn::kConvDim;
  DeviceBuffer qkv_in, a, b, w, a_log, dt_bias, cu_dev, conv, g, beta, out, state, oracle_out, oracle_state, ws;

  Case(const std::string& n, std::int64_t stride = gdn::kConvDim) : name(n), qkv_stride(stride) {
    const Tables& tb = Tables::get();
    const auto ids = load<std::int32_t>(n + ".ids.npy");
    cu = load<std::int32_t>(n + ".cu_seqlens.npy");
    tokens = static_cast<std::int64_t>(ids.size());
    for (std::size_t s = 0; s + 1 < cu.size(); ++s) max_seqlen = std::max(max_seqlen, cu[s + 1] - cu[s]);
    std::vector<std::uint16_t> qkv(static_cast<std::size_t>(tokens * qkv_stride), 0x7fc0);  // NaN in the gap
    std::vector<std::uint16_t> av(static_cast<std::size_t>(tokens) * gdn::kNumVHeads), bv(av.size());
    for (std::int64_t t = 0; t < tokens; ++t) {
      const auto id = static_cast<std::size_t>(ids[static_cast<std::size_t>(t)]);
      std::copy_n(tb.qkv.begin() + static_cast<std::ptrdiff_t>(id * gdn::kConvDim), gdn::kConvDim,
                  qkv.begin() + t * qkv_stride);
      std::copy_n(tb.a.begin() + static_cast<std::ptrdiff_t>(id * gdn::kNumVHeads), gdn::kNumVHeads,
                  av.begin() + t * gdn::kNumVHeads);
      std::copy_n(tb.b.begin() + static_cast<std::ptrdiff_t>(id * gdn::kNumVHeads), gdn::kNumVHeads,
                  bv.begin() + t * gdn::kNumVHeads);
    }
    qkv_in = upload(qkv);
    a = upload(av);
    b = upload(bv);
    w = upload(tb.conv_w);
    a_log = upload(tb.a_log);
    dt_bias = upload(tb.dt_bias);
    cu_dev = upload(cu);
    const auto t = static_cast<std::size_t>(tokens);
    const std::size_t vals = t * gdn::kNumVHeads * gdn::kHeadDim;
    const std::size_t states = (cu.size() - 1) * gdn::kNumVHeads * gdn::kHeadDim * gdn::kHeadDim;
    conv = DeviceBuffer(t * gdn::kConvDim * 2);
    g = DeviceBuffer(t * gdn::kNumVHeads * 4);
    beta = DeviceBuffer(t * gdn::kNumVHeads * 2);
    out = DeviceBuffer(vals * 2);
    state = DeviceBuffer(states * 4);
    oracle_out = DeviceBuffer(vals * 4);
    oracle_state = DeviceBuffer(states * 4);
    ws = DeviceBuffer(gdn::chunk_gated_delta_rule_workspace_bytes(tokens));
  }

  std::size_t num_seqs() const { return cu.size() - 1; }
  Tensor cu_t() const { return Tensor::view(cu_dev.get(), DType::kI32, Device::kCUDA, {static_cast<std::int64_t>(cu.size())}); }
  Tensor conv_t() const { return Tensor::view(conv.get(), DType::kBF16, Device::kCUDA, {tokens, gdn::kConvDim}); }
  Tensor g_t() const { return Tensor::view(g.get(), DType::kF32, Device::kCUDA, {tokens, gdn::kNumVHeads}); }
  Tensor beta_t() const { return Tensor::view(beta.get(), DType::kBF16, Device::kCUDA, {tokens, gdn::kNumVHeads}); }
  Tensor state_t(const DeviceBuffer& buf) const {
    return Tensor::view(buf.get(), DType::kF32, Device::kCUDA,
                        {static_cast<std::int64_t>(num_seqs()), gdn::kNumVHeads, gdn::kHeadDim, gdn::kHeadDim});
  }

  void conv_and_gates(cudaStream_t stream = nullptr) {
    Tensor x = Tensor::view(qkv_in.get(), DType::kBF16, Device::kCUDA, {tokens, gdn::kConvDim});
    x.stride[0] = qkv_stride;
    const Tensor wt = Tensor::view(w.get(), DType::kBF16, Device::kCUDA, {gdn::kConvDim, gdn::kConvKernel});
    gdn::causal_conv1d_silu(x, wt, cu_t(), conv_t(), stream);
    const Tensor at = Tensor::view(a.get(), DType::kBF16, Device::kCUDA, {tokens, gdn::kNumVHeads});
    const Tensor bt = Tensor::view(b.get(), DType::kBF16, Device::kCUDA, {tokens, gdn::kNumVHeads});
    const Tensor alog = Tensor::view(a_log.get(), DType::kBF16, Device::kCUDA, {gdn::kNumVHeads});
    const Tensor dtb = Tensor::view(dt_bias.get(), DType::kBF16, Device::kCUDA, {gdn::kNumVHeads});
    gdn::gates(at, bt, alog, dtb, g_t(), beta_t(), stream);
  }

  void chunk(cudaStream_t stream = nullptr) {
    const Tensor o = Tensor::view(out.get(), DType::kBF16, Device::kCUDA, {tokens, gdn::kNumVHeads, gdn::kHeadDim});
    const Tensor st = state_t(state);
    gdn::chunk_gated_delta_rule(conv_t(), g_t(), beta_t(), cu_t(), max_seqlen, o, &st, ws.get(), ws.size(), stream);
  }

  void oracle() {
    const Tensor o = Tensor::view(oracle_out.get(), DType::kF32, Device::kCUDA, {tokens, gdn::kNumVHeads, gdn::kHeadDim});
    const Tensor st = state_t(oracle_state);
    gdn::recurrent_gated_delta_rule_reference(conv_t(), g_t(), beta_t(), cu_t(), o, &st, nullptr);
  }

  // Core output rows the reference stored, as floats.
  std::vector<float> rows_of(const std::vector<float>& full) const {
    const auto rows = load<std::int32_t>(name + ".rows.npy");
    constexpr std::size_t kRow = gdn::kNumVHeads * gdn::kHeadDim;
    std::vector<float> v;
    v.reserve(rows.size() * kRow);
    for (std::int32_t r : rows) {
      v.insert(v.end(), full.begin() + static_cast<std::ptrdiff_t>(r * kRow),
               full.begin() + static_cast<std::ptrdiff_t>((r + 1) * kRow));
    }
    return v;
  }
};

void run_case(const std::string& name, std::int64_t qkv_stride = gdn::kConvDim) {
  Case c(name, qkv_stride);
  c.conv_and_gates();
  c.chunk();
  c.oracle();
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const auto t = static_cast<std::size_t>(c.tokens);
  const std::size_t vals = t * gdn::kNumVHeads * gdn::kHeadDim;
  const std::size_t states = c.num_seqs() * gdn::kNumVHeads * gdn::kHeadDim * gdn::kHeadDim;

  // Conv + SiLU: bit-exact against bf16 F.conv1d + F.silu (where the reference stored it).
  if (std::filesystem::exists(kData / (name + ".conv.npy"))) {
    const auto ref = load<std::uint16_t>(name + ".conv.npy");
    const auto got = download<std::uint16_t>(c.conv, t * gdn::kConvDim);
    std::size_t diff = 0;
    int max_ulp = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      if (got[i] != ref[i]) {
        ++diff;
        max_ulp = std::max(max_ulp, std::abs(static_cast<int>(got[i]) - static_cast<int>(ref[i])));
      }
    }
    std::printf("[gdn %s] conv1d+silu: %zu / %zu elements differ, max %d bf16 ulp\n", name.c_str(), diff, ref.size(),
                max_ulp);
    EXPECT_EQ(diff, 0u) << "conv1d+silu not bit-exact";
  }

  // Gates: bit-exact.
  {
    const auto g_ref = load<float>(name + ".g.npy");
    const auto beta_ref = load<std::uint16_t>(name + ".beta.npy");
    const auto g_got = download<float>(c.g, t * gdn::kNumVHeads);
    const auto beta_got = download<std::uint16_t>(c.beta, t * gdn::kNumVHeads);
    std::size_t g_diff = 0, beta_diff = 0;
    for (std::size_t i = 0; i < g_ref.size(); ++i) {
      g_diff += float_bits(g_got[i]) != float_bits(g_ref[i]);
      beta_diff += beta_got[i] != beta_ref[i];
    }
    std::printf("[gdn %s] gates: g %zu / %zu differ, beta %zu / %zu differ\n", name.c_str(), g_diff, g_ref.size(),
                beta_diff, beta_ref.size());
    EXPECT_EQ(g_diff, 0u);
    EXPECT_EQ(beta_diff, 0u);
  }

  // Chunk rule (fed with our conv/gates) vs the torch fallback, and vs the oracle kernel.
  const auto out = to_float(download<std::uint16_t>(c.out, vals));
  const auto state = download<float>(c.state, states);
  const auto oracle_out = round_bf16(download<float>(c.oracle_out, vals));
  const auto oracle_state = download<float>(c.oracle_state, states);
  const auto ref_out = to_float(load<std::uint16_t>(name + ".out.npy"));
  const auto ref_state = load<float>(name + ".state.npy");

  const Stats so = compare(c.rows_of(out), ref_out), ss = compare(state, ref_state);
  const Stats oo = compare(out, oracle_out), os = compare(state, oracle_state);
  const Stats ro = compare(c.rows_of(oracle_out), ref_out), rs = compare(oracle_state, ref_state);
  std::printf("[gdn %s] chunk  vs fallback: out   %s\n", name.c_str(), so.str().c_str());
  std::printf("[gdn %s] chunk  vs fallback: state %s\n", name.c_str(), ss.str().c_str());
  std::printf("[gdn %s] chunk  vs oracle:   out   %s\n", name.c_str(), oo.str().c_str());
  std::printf("[gdn %s] chunk  vs oracle:   state %s\n", name.c_str(), os.str().c_str());
  std::printf("[gdn %s] oracle vs fallback: out   %s\n", name.c_str(), ro.str().c_str());
  std::printf("[gdn %s] oracle vs fallback: state %s\n", name.c_str(), rs.str().c_str());
  EXPECT_GE(so.cos, 0.99999);
  EXPECT_GE(ss.cos, 0.99999);
  EXPECT_GE(oo.cos, 0.99999);
  EXPECT_GE(os.cos, 0.99999);
  EXPECT_GE(ro.cos, 0.99999);
  EXPECT_GE(rs.cos, 0.99999);
}

TEST(GpuGdn, Len62) { run_case("l62"); }
TEST(GpuGdn, Len300) { run_case("l300"); }
TEST(GpuGdn, Len4096) { run_case("l4096"); }
// cond (172 tokens) + negative prompt (16) in one launch; qkv rows strided as a fused [qkv | z | b | a] projection.
TEST(GpuGdn, VarlenCondNeg) { run_case("varlen", gdn::kConvDim + gdn::kValueDim + 2 * gdn::kNumVHeads); }

// An empty sequence in the batch yields a zero state, and leaves its neighbours untouched.
TEST(GpuGdn, EmptySequenceInBatch) {
  Case c("varlen");
  c.cu = {c.cu[0], c.cu[1], c.cu[1], c.cu[2]};
  c.cu_dev = upload(c.cu);
  c.state = DeviceBuffer(3 * gdn::kNumVHeads * gdn::kHeadDim * gdn::kHeadDim * 4);
  c.oracle_state = DeviceBuffer(c.state.size());
  MI_CUDA_CHECK(cudaMemset(c.state.get(), 0x7f, c.state.size()));
  c.conv_and_gates();
  c.chunk();
  c.oracle();
  const std::size_t per = gdn::kNumVHeads * gdn::kHeadDim * gdn::kHeadDim;
  const auto state = download<float>(c.state, 3 * per);
  const auto ref = load<float>("varlen.state.npy");
  EXPECT_TRUE(std::all_of(state.begin() + static_cast<std::ptrdiff_t>(per),
                          state.begin() + static_cast<std::ptrdiff_t>(2 * per), [](float v) { return v == 0.f; }));
  const std::vector<float> s0(state.begin(), state.begin() + static_cast<std::ptrdiff_t>(per));
  const std::vector<float> s2(state.begin() + static_cast<std::ptrdiff_t>(2 * per), state.end());
  const std::vector<float> r0(ref.begin(), ref.begin() + static_cast<std::ptrdiff_t>(per));
  const std::vector<float> r1(ref.begin() + static_cast<std::ptrdiff_t>(per), ref.end());
  EXPECT_GE(compare(s0, r0).cos, 0.99999);
  EXPECT_GE(compare(s2, r1).cos, 0.99999);
}

// A single-token sequence: the output row equals row 0 of the l62 golden (causal, so it depends on
// token 0 alone) and the state matches the token-by-token oracle.
TEST(GpuGdn, SingleTokenSequence) {
  Case c("l62");
  c.tokens = 1;
  c.cu = {0, 1};
  c.max_seqlen = 1;
  c.cu_dev = upload(c.cu);
  c.conv_and_gates();
  c.chunk();
  c.oracle();
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  constexpr std::size_t kRow = gdn::kNumVHeads * gdn::kHeadDim;
  constexpr std::size_t kState = gdn::kNumVHeads * gdn::kHeadDim * gdn::kHeadDim;
  const auto out = to_float(download<std::uint16_t>(c.out, kRow));
  const auto ref = to_float(load<std::uint16_t>("l62.out.npy"));
  const std::vector<float> ref0(ref.begin(), ref.begin() + static_cast<std::ptrdiff_t>(kRow));
  EXPECT_GE(compare(out, ref0).cos, 0.99999);
  const auto oracle_out = round_bf16(download<float>(c.oracle_out, kRow));
  EXPECT_GE(compare(out, oracle_out).cos, 0.99999);
  EXPECT_GE(compare(download<float>(c.state, kState), download<float>(c.oracle_state, kState)).cos, 0.99999);
}

// The contract checks (opt-in) catch a max_seqlen that is too small and a cu_seqlens that disagrees with T.
TEST(GpuGdn, DebugChecksCatchContractViolations) {
  Case c("varlen");
  c.conv_and_gates();
  mi::set_varlen_debug_checks(true);
  EXPECT_NO_THROW(c.chunk());
  const int real = c.max_seqlen;
  c.max_seqlen = real - 1;
  EXPECT_THROW(c.chunk(), std::invalid_argument);
  c.max_seqlen = real;
  c.cu.back() -= 1;
  c.cu_dev = upload(c.cu);
  EXPECT_THROW(c.chunk(), std::invalid_argument);
  mi::set_varlen_debug_checks(false);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

// In-place or overlapping conv input/output is rejected (row t reads rows t-3..t).
TEST(GpuGdn, ConvRejectsInPlaceAndOverlap) {
  Case c("l62");
  const Tensor wt = Tensor::view(c.w.get(), DType::kBF16, Device::kCUDA, {gdn::kConvDim, gdn::kConvKernel});
  Tensor x = Tensor::view(c.qkv_in.get(), DType::kBF16, Device::kCUDA, {c.tokens, gdn::kConvDim});
  EXPECT_THROW(gdn::causal_conv1d_silu(x, wt, c.cu_t(), x, nullptr), std::invalid_argument);
  Tensor y = x;
  y.data = static_cast<char*>(c.qkv_in.get()) + 16;  // shifted by a vector, still overlapping
  EXPECT_THROW(gdn::causal_conv1d_silu(x, wt, c.cu_t(), y, nullptr), std::invalid_argument);
}

TEST(GpuGdn, WorkspaceTooSmallThrows) {
  Case c("l62");
  const Tensor o = Tensor::view(c.out.get(), DType::kBF16, Device::kCUDA, {c.tokens, gdn::kNumVHeads, gdn::kHeadDim});
  EXPECT_THROW(gdn::chunk_gated_delta_rule(c.conv_t(), c.g_t(), c.beta_t(), c.cu_t(), c.max_seqlen, o, nullptr,
                                           c.ws.get(), c.ws.size() - 1, nullptr),
               std::invalid_argument);
}

// One layer's GDN core at L = 4096 (conv + gates + chunk rule); the plan budget is 15 ms.
TEST(GpuGdn, PerfLen4096) {
  Case c("l4096");
  cudaStream_t stream;
  MI_CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t e[4];
  for (auto& ev : e) MI_CUDA_CHECK(cudaEventCreate(&ev));
  for (int i = 0; i < 3; ++i) {
    c.conv_and_gates(stream);
    c.chunk(stream);
  }
  constexpr int kIters = 20;
  float conv_ms = 0, rule_ms = 0;
  for (int i = 0; i < kIters; ++i) {
    MI_CUDA_CHECK(cudaEventRecord(e[0], stream));
    c.conv_and_gates(stream);
    MI_CUDA_CHECK(cudaEventRecord(e[1], stream));
    c.chunk(stream);
    MI_CUDA_CHECK(cudaEventRecord(e[2], stream));
    MI_CUDA_CHECK(cudaEventSynchronize(e[2]));
    float a = 0, b = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&a, e[0], e[1]));
    MI_CUDA_CHECK(cudaEventElapsedTime(&b, e[1], e[2]));
    conv_ms += a;
    rule_ms += b;
  }
  conv_ms /= kIters;
  rule_ms /= kIters;
  std::printf("[gdn perf L=4096] conv+gates %.3f ms, chunk rule %.3f ms, total %.3f ms\n", conv_ms, rule_ms,
              conv_ms + rule_ms);
  RecordProperty("gdn_conv_gates_ms", std::to_string(conv_ms));
  RecordProperty("gdn_chunk_rule_ms", std::to_string(rule_ms));
  for (auto& ev : e) MI_CUDA_CHECK(cudaEventDestroy(ev));
  MI_CUDA_CHECK(cudaStreamDestroy(stream));
  if constexpr (mi::testing::kAssertTimings) {
    EXPECT_LT(conv_ms + rule_ms, 15.0f);
  }
}

}  // namespace
