// kernels/fused qk_norm_rope_moss against the op-by-op engine path (rms_norm_moss on the q and k head views of the
// fused QKV output, then moss_rope_apply on each), bit for bit, plus its speed.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "kernels/fused/qk_norm_rope.h"

namespace {

using mi::DType;
using mi::Tensor;

constexpr int kHeads = 24, kHeadDim = 128, kDim = kHeads * kHeadDim;

// Hashed bf16 values with a spread of magnitudes (QKV outputs reach |x| ~ 1e2 for some heads).
std::vector<std::uint16_t> synth_bf16(std::uint64_t seed, std::size_t n, int scale_exp, int spread) {
  std::vector<std::uint16_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = i + seed * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float m = static_cast<float>(128 + (z & 0x7F));
    const int e = static_cast<int>((z >> 7) % static_cast<std::uint64_t>(spread));
    const float v = std::ldexp((z >> 12) & 1 ? -m : m, scale_exp - 7 - e);
    out[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16);
  }
  return out;
}

struct Dev {
  mi::DeviceBuffer buf;
  Tensor t;
  Dev(std::initializer_list<std::int64_t> shape, const std::vector<std::uint16_t>* init = nullptr) {
    t = Tensor::view(nullptr, DType::kBF16, mi::Device::kCUDA, shape);
    buf = mi::DeviceBuffer(t.nbytes());
    t.data = buf.get();
    if (init != nullptr) buf.upload(init->data(), t.nbytes());
  }
  std::vector<std::uint16_t> download() const {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(t.numel()));
    buf.download(h.data(), h.size() * 2);
    return h;
  }
};

Tensor heads_view(const Tensor& buf, std::int64_t col) {
  Tensor t = Tensor::view(static_cast<std::uint16_t*>(buf.data) + col, DType::kBF16, mi::Device::kCUDA,
                          {buf.dim(0), kHeads, kHeadDim});
  t.stride[0] = buf.stride[0];
  return t;
}

void op_by_op(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm, const Tensor& table) {
  const Tensor q = heads_view(qkv, 0), k = heads_view(qkv, kDim);
  mi::kernels::rms_norm_moss(q, q_norm, q, nullptr);
  mi::kernels::rms_norm_moss(k, k_norm, k, nullptr);
  mi::kernels::moss_rope_apply(q, table, q, nullptr);
  mi::kernels::moss_rope_apply(k, table, k, nullptr);
}

struct Case {
  const char* name;
  int grid_h, grid_w, text;  // RoPE table of one row [grid_h*grid_w | text]
  int rows;                  // CFG rows sharing the grid (tables concatenated, as the varlen batch)
};

TEST(GpuFusedQkNormRope, MatchesOpByOpBitForBit) {
  const Case cases[] = {{"1024^2 cond+uncond", 64, 64, 282, 2}, {"1456^2 4096 tokens", 91, 91, 4096, 1},
                        {"tiny 1 token", 1, 1, 0, 1}, {"odd 3x5 + 7", 3, 5, 7, 3}};
  for (const Case& c : cases) {
    const std::int64_t per_row = static_cast<std::int64_t>(c.grid_h) * c.grid_w + c.text;
    const std::int64_t l = per_row * c.rows;
    const auto qkv_v = synth_bf16(1, static_cast<std::size_t>(l * 3 * kDim), 4, 10);
    const auto qn_v = synth_bf16(2, kHeadDim, 0, 3), kn_v = synth_bf16(3, kHeadDim, 0, 3);
    Dev ref({l, 3 * kDim}, &qkv_v), got({l, 3 * kDim}, &qkv_v), qn({kHeadDim}, &qn_v), kn({kHeadDim}, &kn_v),
        table({l, kHeadDim});
    for (int r = 0; r < c.rows; ++r) {
      mi::kernels::moss_rope_table(table.t.slice0(r * per_row, (r + 1) * per_row), c.grid_h, c.grid_w, c.text, nullptr);
    }
    op_by_op(ref.t, qn.t, kn.t, table.t);
    mi::kernels::fused::qk_norm_rope_moss(got.t, kHeads, qn.t, kn.t, table.t, nullptr);
    const auto r = ref.download(), g = got.download();
    std::size_t diff = 0;
    for (std::size_t i = 0; i < r.size(); ++i) diff += r[i] != g[i];
    std::printf("[fused qk_norm_rope] %-20s L=%-6lld: %zu of %zu differ (v untouched: %s)\n", c.name,
                static_cast<long long>(l), diff, r.size(), diff == 0 ? "yes" : "see diff");
    EXPECT_EQ(diff, 0u) << c.name;
  }
}

// The fused output inside a wider row stride (the forward's big activation buffer) and a table slice.
TEST(GpuFusedQkNormRope, StridedRowsAndRejects) {
  const std::int64_t l = 300, ld = 4 * kDim;
  const auto v = synth_bf16(4, static_cast<std::size_t>(l * ld), 3, 8);
  const auto qn_v = synth_bf16(5, kHeadDim, 0, 3);
  Dev ref({l, ld}, &v), got({l, ld}, &v), qn({kHeadDim}, &qn_v), table({l + 5, kHeadDim});
  mi::kernels::moss_rope_table(table.t, 5, 1, static_cast<int>(l), nullptr);
  Tensor rq = Tensor::view(ref.t.data, DType::kBF16, mi::Device::kCUDA, {l, 3 * kDim});
  rq.stride[0] = ld;
  Tensor gq = Tensor::view(got.t.data, DType::kBF16, mi::Device::kCUDA, {l, 3 * kDim});
  gq.stride[0] = ld;
  const Tensor tab = table.t.slice0(5, l + 5);
  op_by_op(rq, qn.t, qn.t, tab);
  mi::kernels::fused::qk_norm_rope_moss(gq, kHeads, qn.t, qn.t, tab, nullptr);
  EXPECT_EQ(ref.download(), got.download());  // includes the untouched v and tail columns
  EXPECT_THROW(mi::kernels::fused::qk_norm_rope_moss(gq, kHeads, qn.t, qn.t, table.t, nullptr), std::invalid_argument);
  EXPECT_THROW(mi::kernels::fused::qk_norm_rope_moss(gq, 2 * kHeads, qn.t, qn.t, tab, nullptr), std::invalid_argument);
}

TEST(GpuFusedQkNormRope, Throughput) {
  const std::int64_t l = 2 * 4096 + 282 + 10;
  const auto v = synth_bf16(6, static_cast<std::size_t>(l * 3 * kDim), 4, 10);
  const auto qn_v = synth_bf16(7, kHeadDim, 0, 3);
  Dev qkv({l, 3 * kDim}, &v), qn({kHeadDim}, &qn_v), table({l, kHeadDim});
  mi::kernels::moss_rope_table(table.t.slice0(0, 4096 + 282), 64, 64, 282, nullptr);
  mi::kernels::moss_rope_table(table.t.slice0(4096 + 282, l), 64, 64, 10, nullptr);
  cudaEvent_t e0, e1;
  MI_CUDA_CHECK(cudaEventCreate(&e0));
  MI_CUDA_CHECK(cudaEventCreate(&e1));
  const auto time = [&](bool fused) {
    constexpr int kIters = 10;
    MI_CUDA_CHECK(cudaEventRecord(e0, nullptr));
    for (int i = 0; i < kIters; ++i) {
      if (fused) {
        mi::kernels::fused::qk_norm_rope_moss(qkv.t, kHeads, qn.t, qn.t, table.t, nullptr);
      } else {
        op_by_op(qkv.t, qn.t, qn.t, table.t);
      }
    }
    MI_CUDA_CHECK(cudaEventRecord(e1, nullptr));
    MI_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) / kIters;
  };
  time(false);
  time(true);
  double base = 0, fused = 0;
  for (int r = 0; r < 5; ++r) {
    base += time(false) / 5;
    fused += time(true) / 5;
  }
  const double bytes = 2.0 * static_cast<double>(l) * 2 * kDim * 2;  // q and k read + written once
  std::printf("[fused qk_norm_rope perf] L=%lld: op-by-op %.3f ms, fused %.3f ms (%.0f GB/s), %.2fx\n",
              static_cast<long long>(l), base, fused, bytes / fused / 1e6, base / fused);
  MI_CUDA_CHECK(cudaEventDestroy(e0));
  MI_CUDA_CHECK(cudaEventDestroy(e1));
}

}  // namespace
