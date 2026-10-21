// kernels/fused GEMMs against the op-by-op engine path they replace (cuBLASLt Gemm::linear into a bf16 tensor, then
// the kernels/basic op), bit for bit, at the DiT block shapes and with every threadblock tile; plus throughput.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/fused/fused_gemm.h"

namespace {

using mi::DType;
using mi::Tensor;
using mi::kernels::fused::GemmTile;

// splitmix64-hashed bf16 values +-m * 2^(scale_exp - 7 - e), m in [128, 255], e in 0..7 (as test_kernels_basic_gemm).
std::vector<std::uint16_t> synth_bf16(std::uint64_t seed, std::size_t n, int scale_exp) {
  std::vector<std::uint16_t> out(n);
  const std::uint64_t offset = seed * 0x9E3779B97F4A7C15ull;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = i + offset;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float m = static_cast<float>(128 + (z & 0x7F));
    const int e = static_cast<int>((z >> 7) & 7);
    const float v = std::ldexp((z >> 10) & 1 ? -m : m, scale_exp - 7 - e);
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
    if (init != nullptr) buf.upload(init->data(), std::min(init->size() * 2, t.nbytes()));  // a prefix of init
  }
  std::vector<std::uint16_t> download() const {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(t.numel()));
    buf.download(h.data(), h.size() * 2);
    return h;
  }
};

std::size_t differing(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

struct Shape {
  const char* name;
  std::int64_t m, k, n;
};

// The block linears (P:136-148) over the CFG batch of a 1024^2 step (4096 image tokens per row, 282 + 10 text
// tokens), its cond-only run of a skipped block, a 1456^2 + 4096-token cond row, and small/odd row counts.
constexpr std::int64_t kM1024 = 2 * 4096 + 282 + 10;
const Shape kShapes[] = {
    {"qkv", kM1024, 3072, 9216},  {"o", kM1024, 3072, 3072},     {"up_gate", kM1024, 3072, 16384},
    {"down", kM1024, 8192, 3072}, {"o_cond_only", 4378, 3072, 3072}, {"down_1456", 8281 + 4096, 8192, 3072},
    {"odd_rows", 37, 3072, 3072},
};

const GemmTile kTiles[] = {GemmTile::kAuto, GemmTile::k128x256, GemmTile::k256x128, GemmTile::k128x128};
const char* tile_name(GemmTile t) {
  switch (t) {
    case GemmTile::k128x256: return "128x256";
    case GemmTile::k256x128: return "256x128";
    case GemmTile::k128x128: return "128x128";
    default: return "auto";
  }
}

class GpuFusedGemm : public ::testing::Test {
 protected:
  void TearDown() override { mi::kernels::fused::set_gemm_tile_override(GemmTile::kAuto); }
  void linear(const Tensor& out, const Tensor& x, const Tensor& w) {
    gemm_.linear(out, x, w, nullptr, ws_.get(), ws_.size(), nullptr);
  }
  mi::kernels::Gemm gemm_;
  mi::DeviceBuffer ws_{mi::kernels::Gemm::kWorkspaceBytes};
};

TEST_F(GpuFusedGemm, PlainMatchesCublasLtBitForBit) {
  for (const Shape& s : kShapes) {
    const auto xv = synth_bf16(1, static_cast<std::size_t>(s.m * s.k), 1);
    const auto wv = synth_bf16(2, static_cast<std::size_t>(s.n * s.k), -5);
    Dev x({s.m, s.k}, &xv), w({s.n, s.k}, &wv), ref({s.m, s.n}), got({s.m, s.n});
    linear(ref.t, x.t, w.t);
    const auto r = ref.download();
    for (GemmTile tile : kTiles) {
      mi::kernels::fused::set_gemm_tile_override(tile);
      MI_CUDA_CHECK(cudaMemset(got.t.data, 0xff, got.t.nbytes()));
      mi::kernels::fused::gemm(got.t, x.t, w.t, nullptr);
      const std::size_t diff = differing(r, got.download());
      std::printf("[fused gemm] %-12s M=%-6lld tile %-7s: %zu of %zu differ\n", s.name, static_cast<long long>(s.m),
                  tile_name(tile), diff, r.size());
      EXPECT_EQ(diff, 0u) << s.name << " tile " << tile_name(tile);
    }
  }
}

// Row counts of every realistic DiT run: a block runs on one contiguous run of a step's rows (cond, uncond, SLG), each
// row hw image tokens (16..4096-pixel sides, at most 4.2 MP) plus its prompt (1..4096 tokens). Every count up to 1200
// (small images: where cuBLASLt's choice varies) and a sample of the larger ones, the listed real shapes included.
std::vector<std::int64_t> realistic_rows() {
  std::vector<std::int64_t> large;
  const std::int64_t prompts[] = {1, 2, 5, 10, 15, 20, 62, 282, 1000, 4096};
  for (int hs = 16; hs <= 4096; hs += 16) {
    for (int ws = 16; ws <= 4096; ws += 16) {
      if (static_cast<std::int64_t>(hs) * ws > 4'200'000) continue;
      const std::int64_t hw = static_cast<std::int64_t>(hs / 16) * (ws / 16);
      for (const std::int64_t sc : prompts) {
        for (const std::int64_t su : {std::int64_t{1}, std::int64_t{10}, sc}) {
          // cond or uncond alone, CFG, CFG + SLG, cond + SLG.
          for (const std::int64_t m : {hw + sc, hw + su, 2 * hw + sc + su, 3 * hw + 2 * sc + su, 2 * hw + 2 * sc}) {
            if (m > 1200) large.push_back(m);
          }
        }
      }
    }
  }
  std::sort(large.begin(), large.end());
  large.erase(std::unique(large.begin(), large.end()), large.end());
  std::vector<std::int64_t> rows;
  for (std::int64_t m = 1; m <= 1200; ++m) rows.push_back(m);
  constexpr std::size_t kLargeSamples = 60;
  for (std::size_t i = 0; i < kLargeSamples; ++i) rows.push_back(large[i * (large.size() - 1) / (kLargeSamples - 1)]);
  for (const std::int64_t m : {4106, 4378, 8292, 8484, 12377, 20674}) rows.push_back(m);
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

// The fused pairs as the DiT block routes them (kernels::fused::matches_cublaslt on cuBLASLt's algorithm for the
// op-by-op linear; models/moss_dit fused_route) equal the op-by-op pairs bit for bit at every realistic row count.
// A row count whose algorithm is not accepted runs op by op in the block (identical by construction); the sweep counts
// those and reports how many of them really differ when forced through CUTLASS (on sm_89: ffn_down's split-K bands).
TEST_F(GpuFusedGemm, RowCountSweep) {
  const std::vector<std::int64_t> rows = realistic_rows();
  struct Op {
    const char* name;
    std::int64_t k, n, max_rows;  // n: the op-by-op linear's N
    bool situ;
  };
  const std::int64_t max_m = rows.back();
  const Op ops[] = {{"o_proj", 3072, 3072, max_m, false}, {"ffn_down", 8192, 3072, max_m, false},
                    {"ffn_up_gate", 3072, 16384, 32768, true}};
  for (const Op& op : ops) {
    const std::int64_t mx = std::min(op.max_rows, max_m), f = op.n / 2;
    const auto wv = synth_bf16(14, static_cast<std::size_t>(op.n * op.k), op.situ ? -4 : -5);
    const auto xv = synth_bf16(13, std::size_t{1} << 24, 1);  // repeated over x's rows
    const auto gv = synth_bf16(8, static_cast<std::size_t>(op.n), -1);
    const auto rv = synth_bf16(7, std::size_t{1} << 24, 6);
    const std::int64_t out_cols = op.situ ? f : op.n;
    Dev w({op.n, op.k}, &wv), x({mx, op.k}), gate({op.n}, &gv), y({mx, op.n}), ref({mx, out_cols}), got({mx, out_cols});
    for (std::size_t off = 0; off < static_cast<std::size_t>(x.t.numel()); off += xv.size()) {
      x.buf.upload(xv.data(), std::min(xv.size(), static_cast<std::size_t>(x.t.numel()) - off) * 2, off * 2);
    }
    if (!op.situ) {
      for (std::size_t off = 0; off < static_cast<std::size_t>(ref.t.numel()); off += rv.size()) {
        ref.buf.upload(rv.data(), std::min(rv.size(), static_cast<std::size_t>(ref.t.numel()) - off) * 2, off * 2);
      }
    }
    std::size_t fused = 0, rejected = 0, rejected_differ = 0;
    std::string rejected_rows;
    for (const std::int64_t m : rows) {
      if (m > mx) continue;
      const Tensor xm = x.t.slice0(0, m), ym = y.t.slice0(0, m), rm = ref.t.slice0(0, m), gm = got.t.slice0(0, m);
      const bool accepted = mi::kernels::fused::matches_cublaslt(gemm_.algo_config(ym, xm, w.t, nullptr, ws_.size()));
      const std::size_t bytes = static_cast<std::size_t>(m * out_cols) * 2;
      // Op by op (as MossDit::run_block) into ref; the fused kernel into got (from the same residual).
      if (op.situ) {
        linear(ym, xm, w.t);
        mi::kernels::situ_glu(ym, rm, nullptr);
        mi::kernels::fused::gemm_situ_glu(gm, xm, w.t, nullptr);
      } else {
        MI_CUDA_CHECK(cudaMemcpy(gm.data, rm.data, bytes, cudaMemcpyDeviceToDevice));
        linear(ym, xm, w.t);
        mi::kernels::gated_residual(rm, ym, gate.t, rm, nullptr);
        mi::kernels::fused::gemm_gated_residual(gm, xm, w.t, gate.t, nullptr);
      }
      std::vector<std::uint16_t> r(static_cast<std::size_t>(m * out_cols)), g(r.size());
      MI_CUDA_CHECK(cudaMemcpy(r.data(), rm.data, bytes, cudaMemcpyDeviceToHost));
      MI_CUDA_CHECK(cudaMemcpy(g.data(), gm.data, bytes, cudaMemcpyDeviceToHost));
      const std::size_t diff = differing(r, g);
      if (accepted) {
        ++fused;
        EXPECT_EQ(diff, 0u) << op.name << " M=" << m << ": accepted algorithm "
                            << gemm_.algo_config(ym, xm, w.t, nullptr, ws_.size()).describe();
      } else {
        ++rejected;
        rejected_differ += diff != 0;
        if (rejected <= 12) rejected_rows += " " + std::to_string(m);
      }
      if (!op.situ) MI_CUDA_CHECK(cudaMemcpy(rm.data, gm.data, bytes, cudaMemcpyDeviceToDevice));  // same residual next M
    }
    std::printf("[fused gemm sweep] %-11s %zu row counts (1..%lld): %zu fused bit-identical, %zu op by op (%zu of them "
                "would differ fused), first op-by-op M:%s\n",
                op.name, fused + rejected, static_cast<long long>(mx), fused, rejected, rejected_differ,
                rejected_rows.c_str());
    EXPECT_GT(fused, rows.size() / 2) << op.name;
  }
}

// cuBLASLt's algorithm query: the plan linear() then runs (cached once), nothing launched.
TEST_F(GpuFusedGemm, AlgoConfigQuery) {
  Dev x({84, 8192}), w({3072, 8192}), out({84, 3072});
  const std::size_t plans = gemm_.cached_plans();
  const mi::kernels::Gemm::AlgoConfig a = gemm_.algo_config(out.t, x.t, w.t, nullptr, ws_.size());
  EXPECT_EQ(gemm_.cached_plans(), plans + 1);
  EXPECT_EQ(gemm_.algo_config(out.t, x.t, w.t, nullptr, ws_.size()), a);
  EXPECT_EQ(gemm_.cached_plans(), plans + 1);
  linear(out.t, x.t, w.t);
  EXPECT_EQ(gemm_.cached_plans(), plans + 1);  // the query's plan
  EXPECT_GE(a.split_k, 1);
  std::printf("[fused gemm] cuBLASLt for ffn_down at 84 rows: %s -> %s\n", a.describe().c_str(),
              mi::kernels::fused::matches_cublaslt(a) ? "fused" : "op by op");
  mi::kernels::Gemm::AlgoConfig split = a;
  split.split_k = 3;
  split.reduction_scheme = 2;
  EXPECT_FALSE(mi::kernels::fused::matches_cublaslt(split));
  mi::kernels::Gemm::AlgoConfig k_ordered;
  k_ordered.id = 21;
  EXPECT_TRUE(mi::kernels::fused::matches_cublaslt(k_ordered));
  k_ordered.id = 31;
  EXPECT_FALSE(mi::kernels::fused::matches_cublaslt(k_ordered));
}

// Strided operands: x a column window of a wider buffer (attention output inside [qkv | attn]), out likewise.
TEST_F(GpuFusedGemm, StridedOperands) {
  const std::int64_t m = 300, k = 3072, n = 3072, ld = 4 * 3072;
  const auto xv = synth_bf16(3, static_cast<std::size_t>(m * ld), 0);
  const auto wv = synth_bf16(4, static_cast<std::size_t>(n * k), -5);
  Dev xb({m, ld}, &xv), w({n, k}, &wv), ref({m, n}), outb({m, ld});
  Tensor x = Tensor::view(static_cast<std::uint16_t*>(xb.t.data) + 3 * 3072, DType::kBF16, mi::Device::kCUDA, {m, k});
  x.stride[0] = ld;
  Tensor out = Tensor::view(static_cast<std::uint16_t*>(outb.t.data) + 3072, DType::kBF16, mi::Device::kCUDA, {m, n});
  out.stride[0] = ld;
  linear(ref.t, x, w.t);
  mi::kernels::fused::gemm(out, x, w.t, nullptr);
  const auto r = ref.download(), all = outb.download();
  std::size_t diff = 0;
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) diff += r[static_cast<std::size_t>(i * n + j)] != all[static_cast<std::size_t>(i * ld + 3072 + j)];
  }
  EXPECT_EQ(diff, 0u);
}

TEST_F(GpuFusedGemm, GatedResidualMatchesOpByOp) {
  for (const Shape& s : kShapes) {
    if (s.n != 3072) continue;  // o-proj and ffn_down
    const auto xv = synth_bf16(5, static_cast<std::size_t>(s.m * s.k), 1);
    const auto wv = synth_bf16(6, static_cast<std::size_t>(s.n * s.k), -5);
    const auto rv = synth_bf16(7, static_cast<std::size_t>(s.m * s.n), 6);  // residual stream, larger than the update
    const auto gv = synth_bf16(8, static_cast<std::size_t>(s.n), -1);
    Dev x({s.m, s.k}, &xv), w({s.n, s.k}, &wv), gate({s.n}, &gv), y({s.m, s.n}), ref({s.m, s.n}, &rv), got({s.m, s.n});
    linear(y.t, x.t, w.t);
    mi::kernels::gated_residual(ref.t, y.t, gate.t, ref.t, nullptr);
    const auto r = ref.download();
    for (GemmTile tile : kTiles) {
      mi::kernels::fused::set_gemm_tile_override(tile);
      got.buf.upload(rv.data(), rv.size() * 2);
      mi::kernels::fused::gemm_gated_residual(got.t, x.t, w.t, gate.t, nullptr);
      const std::size_t diff = differing(r, got.download());
      std::printf("[fused gated_residual] %-12s M=%-6lld tile %-7s: %zu of %zu differ\n", s.name,
                  static_cast<long long>(s.m), tile_name(tile), diff, r.size());
      EXPECT_EQ(diff, 0u) << s.name << " tile " << tile_name(tile);
    }
  }
}

TEST_F(GpuFusedGemm, SituGluMatchesOpByOp) {
  const Shape shapes[] = {{"up_gate", kM1024, 3072, 16384}, {"up_gate_cond", 4378, 3072, 16384}, {"odd_rows", 37, 3072, 16384}};
  for (const Shape& s : shapes) {
    const std::int64_t f = s.n / 2;
    const auto xv = synth_bf16(9, static_cast<std::size_t>(s.m * s.k), 1);
    const auto wv = synth_bf16(10, static_cast<std::size_t>(s.n * s.k), -4);
    Dev x({s.m, s.k}, &xv), w({s.n, s.k}, &wv), up_gate({s.m, s.n}), ref({s.m, f}), got({s.m, f});
    linear(up_gate.t, x.t, w.t);
    mi::kernels::situ_glu(up_gate.t, ref.t, nullptr);
    const auto r = ref.download();
    for (GemmTile tile : kTiles) {
      mi::kernels::fused::set_gemm_tile_override(tile);
      MI_CUDA_CHECK(cudaMemset(got.t.data, 0xff, got.t.nbytes()));
      mi::kernels::fused::gemm_situ_glu(got.t, x.t, w.t, nullptr);
      const std::size_t diff = differing(r, got.download());
      std::printf("[fused situ_glu] %-12s M=%-6lld tile %-7s: %zu of %zu differ\n", s.name, static_cast<long long>(s.m),
                  tile_name(tile), diff, r.size());
      EXPECT_EQ(diff, 0u) << s.name << " tile " << tile_name(tile);
    }
  }
}

TEST_F(GpuFusedGemm, RejectsBadOperands) {
  Dev x({16, 3072}), w({3072, 3072}), out({16, 3072}), gate({3072}), small({16, 100});
  EXPECT_THROW(mi::kernels::fused::gemm(small.t, x.t, w.t, nullptr), std::invalid_argument);
  Tensor misaligned = Tensor::view(static_cast<std::uint16_t*>(x.t.data) + 1, DType::kBF16, mi::Device::kCUDA, {16, 3064});
  misaligned.stride[0] = 3072;
  EXPECT_THROW(mi::kernels::fused::gemm(out.t, misaligned, w.t, nullptr), std::invalid_argument);
  EXPECT_THROW(mi::kernels::fused::gemm_gated_residual(out.t, x.t, w.t, small.t, nullptr), std::invalid_argument);
  Dev w_odd({3072 + 4, 3072}), half({16, 1538});
  EXPECT_THROW(mi::kernels::fused::gemm_situ_glu(half.t, x.t, w_odd.t, nullptr), std::invalid_argument);  // F % 4 != 0
}

// Throughput of each fused op against the op-by-op pair it replaces (cuBLASLt + kernels/basic op), for every tile,
// interleaved A/B at the 1024^2 CFG batch (synthetic data; tests/unit/test_dit_block_fused has the per-op numbers on
// real activations). "up_gate plain" separates the SiTU epilogue's cost from the main loop's.
TEST_F(GpuFusedGemm, Throughput) {
  const std::int64_t m = kM1024, k = 3072, f = 8192;
  const auto xv = synth_bf16(11, static_cast<std::size_t>(m * f), 0);
  const auto wv = synth_bf16(12, static_cast<std::size_t>(2 * f * k), -5);
  Dev x({m, f}, &xv), w_up({2 * f, k}, &wv), w_o({k, k}, &wv), w_down({k, f}, &wv), gate({k}, &wv);
  Dev big({m, 2 * f}), ffn({m, f}), y({m, k}), res({m, k}, &xv);
  const Tensor xk = Tensor::view(x.t.data, DType::kBF16, mi::Device::kCUDA, {m, k});
  const Tensor qkv = Tensor::view(big.t.data, DType::kBF16, mi::Device::kCUDA, {m, 3 * k});
  const Tensor w_qkv = Tensor::view(w_up.t.data, DType::kBF16, mi::Device::kCUDA, {3 * k, k});
  struct Op {
    const char* name;
    double flops;
    std::function<void()> base, fused;
  };
  const Op ops[] = {
      {"o + gated_residual", 2.0 * m * k * k,
       [&] { linear(y.t, xk, w_o.t); mi::kernels::gated_residual(res.t, y.t, gate.t, res.t, nullptr); },
       [&] { mi::kernels::fused::gemm_gated_residual(res.t, xk, w_o.t, gate.t, nullptr); }},
      {"up_gate + situ_glu", 2.0 * m * k * 2 * f,
       [&] { linear(big.t, xk, w_up.t); mi::kernels::situ_glu(big.t, ffn.t, nullptr); },
       [&] { mi::kernels::fused::gemm_situ_glu(ffn.t, xk, w_up.t, nullptr); }},
      {"up_gate plain", 2.0 * m * k * 2 * f, [&] { linear(big.t, xk, w_up.t); },
       [&] { mi::kernels::fused::gemm(big.t, xk, w_up.t, nullptr); }},
      {"down + gated_residual", 2.0 * m * f * k,
       [&] { linear(y.t, x.t, w_down.t); mi::kernels::gated_residual(res.t, y.t, gate.t, res.t, nullptr); },
       [&] { mi::kernels::fused::gemm_gated_residual(res.t, x.t, w_down.t, gate.t, nullptr); }},
      {"qkv plain", 2.0 * m * k * 3 * k, [&] { linear(qkv, xk, w_qkv); },
       [&] { mi::kernels::fused::gemm(qkv, xk, w_qkv, nullptr); }},
  };
  cudaEvent_t e0, e1;
  MI_CUDA_CHECK(cudaEventCreate(&e0));
  MI_CUDA_CHECK(cudaEventCreate(&e1));
  const auto time = [&](const std::function<void()>& fn, int iters) {
    MI_CUDA_CHECK(cudaEventRecord(e0, nullptr));
    for (int i = 0; i < iters; ++i) fn();
    MI_CUDA_CHECK(cudaEventRecord(e1, nullptr));
    MI_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) / iters;
  };
  for (const Op& op : ops) {
    for (GemmTile tile : kTiles) {
      mi::kernels::fused::set_gemm_tile_override(tile);
      op.base();
      op.fused();
      double base = 0, fused = 0;
      constexpr int kRounds = 5, kIters = 3;
      for (int r = 0; r < kRounds; ++r) {  // interleaved A/B: clock drift hits both sides alike
        base += time(op.base, kIters);
        fused += time(op.fused, kIters);
      }
      base /= kRounds;
      fused /= kRounds;
      std::printf("[fused perf] %-22s tile %-7s op-by-op %6.3f ms (%.1f TFLOPS)  fused %6.3f ms (%.1f TFLOPS)  %.3fx\n",
                  op.name, tile_name(tile), base, op.flops / base / 1e9, fused, op.flops / fused / 1e9, base / fused);
      if (tile == GemmTile::kAuto) RecordProperty(std::string("speedup ") + op.name, std::to_string(base / fused));
    }
  }
  MI_CUDA_CHECK(cudaEventDestroy(e0));
  MI_CUDA_CHECK(cudaEventDestroy(e1));
}

}  // namespace
