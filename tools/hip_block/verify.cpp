// hip_block_verify (NVIDIA build): the same inputs through the real engine's kernels, linked from this tree's
// mossinfer_core, and through the prototype with its scalar leaf, reporting the per-op error distribution in bf16 ULPs.
// This checks the prototype's kernel logic above the leaf: the op and rounding order, the tile walks and every tail. It
// says nothing about WMMA, which does not run here, and nothing about rocWMMA's lane layout.
//
//   hip_block_verify                                 synthetic inputs: shapes with tails, edge inputs, whole blocks
//   hip_block_verify --real [--weights DIR] [--golden DIR] [--dump DIR]   real block weights and activations captured
//                                                    from the engine's forward, at the 1024^2 golden shape (DIR defaults:
//                                                    $MOSS_WEIGHTS, whose base/ holds moss.mossidx, and $MOSS_GOLDEN,
//                                                    the directory of the golden .npy files). --dump writes E13's blocks
//                                                    for hip_block_replay, with the scalar leaf's per-op figures.
//
// Tensor-core and scalar accumulation are not bit-compatible, so outputs differ in the last place where an fp32 rounding
// difference crosses a bf16 rounding boundary. Raw ULP distances also grow without limit on outputs near zero: a sum that
// nearly cancels (a GEMM output, residual + delta) keeps the absolute error of its large terms, and bf16 ULPs shrink with
// the output. So a row has `scaled` as well: the largest error in ULPs of max(|engine|, rms of the op's output).
//
// Gates, all of them also failing on a NaN on one side only:
//   exact (rms_norm_modulate from 16 rows, qk_norm_rope, no accumulation to differ in): no bit differences, NaN payload
//     aside.
//   accumulated (GEMMs, gated residual, SiTU, attention): scaled <= 4 and, with at least 1000 values, same >= 99 % and
//     p99.9 <= 1. A dropped rounding point moves many outputs by 1 ULP, which same and p99.9 see and scaled does not.
//     A 1 ULP flip of a rounded GEMM result can move a later output by more than 4 ULPs at the output's scale: SiTU is
//     steep, and gated residuals cancel (the last block's gate is near 30). So for those two an output above scaled 4
//     passes only if the op's own host evaluation reproduces both sides from a GEMM result within 1 ULP of the exact sum,
//     and an Inf on one side fails.
//     On real activations SiTU's p99.9 limit is 2: a 1 ULP input flip often moves its output by 2 there.
//   block (a whole block on its own outputs, where the errors of eight ops compound), synthetic: scaled <= 16,
//     same >= 80 %, p99 <= 16.
//   block, real activations: against a control measured in the same run, the engine's block with its QKV GEMM result
//     replaced by the exact sum rounded to bf16, which involves no prototype code: same >= control same - 5 points,
//     p99 <= 2 x control p99 (at least 16), scaled <= 2 x control scaled (at least 16). The engine's own later ops carry a
//     low-bit QKV difference that far, and how far depends on the block. A run without a control fails, and so does one
//     whose QKV fails the accumulated gate against the engine's or whose same is under 50 %. The rows
//     "engine + prototype's <op>" show each op's share, and are not gated.
// Raw outliers are compared with a host evaluation (GEMM: the exact sum; gated residual, SiTU: the exact sum, then the
// op's own rounding points; attention: replay_attention_row<double>) to see which side they come from.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "host_util.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/fused/qk_norm_rope.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/moss_dit.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "verify_kernels.h"
#include "weights/model_index.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
using namespace hip_block;
namespace md = mi::moss_dit;
using mi::DType;
using mi::Tensor;
using Bits = std::vector<bf16_bits>;
using Buf = DevBuffer<bf16_bits>;

// kReportOnly: printed, with why it is not gated, not failing.
enum class Gate { kExact, kAccumulated, kFlipChecked, kBlock, kReportOnly };

// One row's numbers.
struct Stats {
  double same_pct = 0.0, scaled = 0.0;
  int p99 = 0, p999 = 0;
};

// The block gate; the defaults are the synthetic one's.
struct BlockLimits {
  double same_min = 80.0;
  int p99_max = 16;
  double scaled_max = 16.0;
};

constexpr std::int32_t kOutlierUlps = 4;
constexpr std::size_t kMaxOutliersChecked = 200;  // the worst ones, per case
const float kScale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));

Tensor mat(const bf16_bits* p, std::int64_t rows, std::int64_t cols) {
  return Tensor::view(const_cast<bf16_bits*>(p), DType::kBF16, mi::Device::kCUDA, {rows, cols});
}
Tensor vec(const bf16_bits* p, std::int64_t n) {
  return Tensor::view(const_cast<bf16_bits*>(p), DType::kBF16, mi::Device::kCUDA, {n});
}
Tensor heads_of(bf16_bits* p, std::int64_t tokens, std::int64_t heads, std::int64_t ld) {
  Tensor t = Tensor::view(p, DType::kBF16, mi::Device::kCUDA, {tokens, heads, std::int64_t{kHeadDim}});
  t.stride[0] = ld;
  return t;
}

// bf16 bits on the number line of representable values: neighbours differ by 1, -0 and +0 coincide.
std::int32_t ordered(bf16_bits b) { return b & 0x8000u ? -static_cast<std::int32_t>(b & 0x7fffu) : b; }
bool is_nan(bf16_bits b) { return (b & 0x7fffu) > 0x7f80u; }
std::int32_t ulp_distance(bf16_bits a, bf16_bits b) {
  if (is_nan(a) || is_nan(b)) return is_nan(a) && is_nan(b) ? 0 : 0x7fffffff;
  return std::abs(ordered(a) - ordered(b));
}

int g_failures = 0;
std::string g_dump_dir;  // --dump: the E13 real blocks' weights, inputs and engine outputs, for hip_block_replay
std::vector<std::string> g_dumped;  // the block directories written under it

void print_header() {
  std::printf("| %-40s | %10s | %8s | %3s | %3s | %5s | %6s | %5s | %6s |\n", "case", "values", "same %", "p50", "p99", "p99.9",
              "p99.99", "max", "scaled");
  std::printf("|%s|%s|%s|%s|%s|%s|%s|%s|%s|\n", std::string(42, '-').c_str(), std::string(12, '-').c_str(),
              std::string(10, '-').c_str(), std::string(5, '-').c_str(), std::string(5, '-').c_str(), std::string(7, '-').c_str(),
              std::string(8, '-').c_str(), std::string(7, '-').c_str(), std::string(8, '-').c_str());
}

// One row of the table, with its gate. limits: of kBlock; p999_limit: of kAccumulated and kFlipChecked.
Stats report(const std::string& name, const Bits& ref, const Bits& got, Gate gate, const BlockLimits& limits = {}, int p999_limit = 1) {
  std::vector<std::int32_t> ulps(ref.size());
  double sq = 0.0;
  std::size_t nan_mismatch = 0, bit_differ = 0, same = 0;
  std::int64_t finite = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const bool rn = is_nan(ref[i]), gn = is_nan(got[i]);
    if (rn || gn) {
      nan_mismatch += rn != gn;
      same += rn == gn;
      continue;
    }
    bit_differ += ref[i] != got[i];
    ulps[i] = ulp_distance(ref[i], got[i]);
    same += ulps[i] == 0;
    const double v = from_bf16(ref[i]);
    if (std::isfinite(v)) {
      sq += v * v;
      ++finite;
    }
  }
  const double rms = std::sqrt(sq / static_cast<double>(std::max<std::int64_t>(finite, 1)));
  double scaled = 0.0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ulps[i] > ulps[worst]) worst = i;
    if (is_nan(ref[i]) || is_nan(got[i])) continue;
    const double r = from_bf16(ref[i]), g = from_bf16(got[i]);
    if (!std::isfinite(r) || !std::isfinite(g)) {
      if (r != g) scaled = INFINITY;
      continue;
    }
    int e = 0;
    (void)std::frexp(std::max(std::abs(r), rms), &e);
    scaled = std::max(scaled, std::abs(r - g) / std::ldexp(1.0, e - 8));
  }
  std::vector<std::int32_t> sorted = ulps;
  std::sort(sorted.begin(), sorted.end());
  const auto pct = [&](double p) {
    return sorted[std::min(sorted.size() - 1, static_cast<std::size_t>(p / 100.0 * static_cast<double>(sorted.size())))];
  };
  const double same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(ref.size());

  std::string why;
  if (nan_mismatch != 0) why += " NaN mismatch;";
  if (gate == Gate::kReportOnly && (scaled > 4.0 || same_pct < 99.0)) why += " scaled or same outside the accumulated gate;";
  if (gate == Gate::kExact && bit_differ != 0) why += " bits differ;";
  if (gate == Gate::kAccumulated || gate == Gate::kFlipChecked) {
    if (gate == Gate::kAccumulated && scaled > 4.0) why += " scaled > 4;";
    if (gate == Gate::kFlipChecked && !std::isfinite(scaled)) why += " Inf on one side;";  // check_flips sees finite outputs only
    if (ref.size() >= 1000 && same_pct < 99.0) why += " same < 99 %;";
    if (ref.size() >= 1000 && pct(99.9) > p999_limit) why += " p99.9 > " + std::to_string(p999_limit) + ";";
  }
  if (gate == Gate::kBlock) {
    if (scaled > limits.scaled_max) why += " scaled over limit;";
    if (same_pct < limits.same_min) why += " same under limit;";
    if (pct(99) > limits.p99_max) why += " p99 over limit;";
  }
  std::printf("| %-40s | %10zu | %8.4f | %3d | %3d | %5d | %6d | %5d | %6.2f |%s%s\n", name.c_str(), ref.size(), same_pct, pct(50), pct(99),
              pct(99.9), pct(99.99), sorted.back(), scaled, why.empty() || gate == Gate::kReportOnly ? "" : " FAILED:",
              gate == Gate::kReportOnly ? "" : why.c_str());
  if (sorted.back() > kOutlierUlps) {
    std::printf("|   worst raw: index %zu, engine %.6g, prototype %.6g, output rms %.4g\n", worst,
                static_cast<double>(from_bf16(ref[worst])), static_cast<double>(from_bf16(got[worst])), rms);
  }
  if (nan_mismatch != 0) {
    std::size_t first = 0;
    while (is_nan(ref[first]) == is_nan(got[first])) ++first;
    std::printf("|   %zu values are NaN on one side only; first: index %zu, engine %.6g, prototype %.6g\n", nan_mismatch, first,
                static_cast<double>(from_bf16(ref[first])), static_cast<double>(from_bf16(got[first])));
  }
  if (gate == Gate::kReportOnly) {
    if (!why.empty()) std::printf("|   not gated:%s\n", why.c_str());
  } else if (!why.empty()) {
    ++g_failures;
  }
  return {same_pct, scaled, pct(99), pct(99.9)};
}

// The worst raw outliers against `reference(index)`, a host evaluation: which side is closer.
void report_outliers(const char* what, const Bits& ref, const Bits& got, const std::function<double(std::size_t)>& reference) {
  std::vector<std::size_t> outliers;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!is_nan(ref[i]) && !is_nan(got[i]) && ulp_distance(ref[i], got[i]) > kOutlierUlps) outliers.push_back(i);
  }
  if (outliers.empty()) return;
  const std::size_t total = outliers.size();
  const std::size_t checked = std::min(total, kMaxOutliersChecked);
  std::partial_sort(outliers.begin(), outliers.begin() + static_cast<std::ptrdiff_t>(checked), outliers.end(), [&](std::size_t a, std::size_t b) {
    return ulp_distance(ref[a], got[a]) > ulp_distance(ref[b], got[b]);
  });
  std::size_t engine_closer = 0, prototype_closer = 0, ties = 0;
  double engine_err = 0.0, prototype_err = 0.0;
  for (std::size_t n = 0; n < checked; ++n) {
    const std::size_t i = outliers[n];
    const double want = reference(i);
    const double e = std::abs(from_bf16(ref[i]) - want), p = std::abs(from_bf16(got[i]) - want);
    engine_err = std::max(engine_err, e);
    prototype_err = std::max(prototype_err, p);
    engine_closer += e < p;
    prototype_closer += p < e;
    ties += p == e;
  }
  std::printf("|   %s: %zu outputs > %d ULPs apart, worst %zu checked: engine closer %zu, prototype closer %zu, tied %zu; "
              "largest error engine %.3g, prototype %.3g\n",
              what, total, kOutlierUlps, checked, engine_closer, prototype_closer, ties, engine_err, prototype_err);
}

// out[m, n] = bf16(x · wᵀ): the products are exact in fp64 and summed there, K ascending, then rounded to fp32 and to bf16,
// the engine's two conversions. Two rounding steps: rarely not the direct fp64 to bf16 result (605 of 76161024 at golden
// block 1). Host code only.
Bits exact_gemm(const Bits& x, std::size_t ldx, const Bits& w, std::size_t ldw, std::size_t m, std::size_t n, std::size_t k) {
  constexpr std::size_t kBlockK = 256, kBlockN = 256;  // keeps a thread's weight block in its L2
  std::vector<double> w_by_k(k * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < k; ++j) w_by_k[j * n + i] = from_bf16(w[i * ldw + j]);
  }
  std::vector<double> sum(m * n, 0.0);
  const unsigned threads = std::clamp(std::thread::hardware_concurrency(), 1u, 16u);
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      const std::size_t r0 = m * t / threads, r1 = m * (t + 1) / threads;
      for (std::size_t k0 = 0; k0 < k; k0 += kBlockK) {
        for (std::size_t n0 = 0; n0 < n; n0 += kBlockN) {
          const std::size_t k1 = std::min(k, k0 + kBlockK), len = std::min(n - n0, kBlockN);
          for (std::size_t r = r0; r < r1; ++r) {
            double* out = sum.data() + r * n + n0;
            for (std::size_t j = k0; j < k1; ++j) {
              const double xj = from_bf16(x[r * ldx + j]);
              const double* wj = w_by_k.data() + j * n + n0;
              for (std::size_t i = 0; i < len; ++i) out[i] += xj * wj[i];
            }
          }
        }
      }
    });
  }
  for (std::thread& th : pool) th.join();
  Bits bits(sum.size());
  for (std::size_t i = 0; i < sum.size(); ++i) bits[i] = to_bf16(static_cast<float>(sum[i]));
  return bits;
}

double exact_dot(const bf16_bits* x, const bf16_bits* w, std::size_t k) {
  double sum = 0.0;
  for (std::size_t j = 0; j < k; ++j) sum += static_cast<double>(from_bf16(x[j])) * from_bf16(w[j]);
  return sum;
}

void report_gemm_outliers(const Bits& ref, const Bits& got, const Bits& x, std::size_t ldx, const Bits& w, std::size_t ldw, std::size_t n,
                          std::size_t k) {
  report_outliers("exact sum", ref, got, [&](std::size_t i) {
    return exact_dot(x.data() + i / n * ldx, w.data() + i % n * ldw, k);
  });
}

// residual + bf16(bf16(x · wᵀ) * gate): the exact sum, the op's rounding points, no rounding of the final add.
void report_gated_outliers(const Bits& ref, const Bits& got, const Bits& residual, std::size_t ldr, const Bits& x, std::size_t ldx, const Bits& w,
                           std::size_t ldw, const bf16_bits* gate, std::size_t n, std::size_t k) {
  report_outliers("exact sum + rounding points", ref, got, [&](std::size_t i) {
    const std::size_t r = i / n, c = i % n;
    const float ab = host_rbf(static_cast<float>(exact_dot(x.data() + r * ldx, w.data() + c * ldw, k)));
    const float delta = host_rbf(ab * from_bf16(gate[c]));
    return static_cast<double>(from_bf16(residual[r * ldr + c])) + delta;
  });
}

// The output furthest from the engine in absolute terms, term by term: a large error beside a small output means terms
// far larger than the output cancelled in the final add.
void report_gated_worst(const Bits& ref, const Bits& got, const Bits& residual, std::size_t ldr, const Bits& x, std::size_t ldx, const Bits& w, std::size_t ldw,
                        const bf16_bits* gate, std::size_t n, std::size_t k) {
  std::size_t worst = 0;
  double worst_err = -1.0;
  double largest_term = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (is_nan(ref[i]) || is_nan(got[i])) continue;
    const double err = std::abs(static_cast<double>(from_bf16(ref[i])) - from_bf16(got[i]));
    if (err > worst_err) {
      worst_err = err;
      worst = i;
    }
    largest_term = std::max(largest_term, std::abs(static_cast<double>(from_bf16(residual[i / n * ldr + i % n]))));
  }
  const std::size_t r = worst / n, c = worst % n;
  const double ab = exact_dot(x.data() + r * ldx, w.data() + c * ldw, k);
  const float rounded = host_rbf(static_cast<float>(ab));
  const float delta = host_rbf(rounded * from_bf16(gate[c]));
  std::printf("|   largest absolute difference: index %zu, engine %.6g, prototype %.6g; residual %.6g, exact x·w %.6g, gate %.4g, delta %.6g; largest |residual| %.4g\n", worst,
              static_cast<double>(from_bf16(ref[worst])), static_cast<double>(from_bf16(got[worst])), static_cast<double>(from_bf16(residual[r * ldr + c])), ab,
              static_cast<double>(from_bf16(gate[c])), static_cast<double>(delta), largest_term);
}

void report_situ_outliers(const Bits& ref, const Bits& got, const Bits& x, std::size_t ldx, const Bits& w, std::size_t ldw, std::size_t f, std::size_t k) {
  report_outliers("exact sums + rounding points", ref, got, [&](std::size_t i) {
    const std::size_t r = i / f, c = i % f;
    const float h = host_rbf(static_cast<float>(exact_dot(x.data() + r * ldx, w.data() + c * ldw, k)));
    const float g = host_rbf(static_cast<float>(exact_dot(x.data() + r * ldx, w.data() + (f + c) * ldw, k)));
    return static_cast<double>(host_situ(h, g));
  });
}

bf16_bits bf16_step(bf16_bits b, int by) {
  const std::int32_t o = ordered(b) + by;
  return o < 0 ? static_cast<bf16_bits>(0x8000u | static_cast<unsigned>(-o)) : static_cast<bf16_bits>(o);
}

// The flip check of the accumulated gate. `reachable(i, out)`: the op's host evaluation gives bits within 1 ULP of `out`
// (host and device math differ in fp32's last bits) from GEMM results within 1 ULP of the exact sums. Outputs above scaled 4
// on either side that are not reachable fail. Outputs 2..4 ULPs apart are reported the same way, without failing.
void check_flips(const char* op, const Bits& ref, const Bits& got, const std::function<bool(std::size_t, bf16_bits)>& reachable) {
  double sq = 0.0;
  std::size_t finite = 0;
  for (const bf16_bits b : ref) {
    if (!is_nan(b) && std::isfinite(from_bf16(b))) {
      sq += static_cast<double>(from_bf16(b)) * from_bf16(b);
      ++finite;
    }
  }
  const double rms = std::sqrt(sq / static_cast<double>(std::max<std::size_t>(finite, 1)));
  std::size_t above = 0, above_unexplained = 0, mid = 0, mid_unexplained = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (is_nan(ref[i]) || is_nan(got[i])) continue;
    const double r = from_bf16(ref[i]), g = from_bf16(got[i]);
    if (!std::isfinite(r) || !std::isfinite(g)) continue;
    int e = 0;
    (void)std::frexp(std::max(std::abs(r), rms), &e);
    const double err = std::abs(r - g) / std::ldexp(1.0, e - 8);
    const std::int32_t ulps = ulp_distance(ref[i], got[i]);
    if (err > 4.0) {
      ++above;
      worst = std::max(worst, err);
      if (!reachable(i, ref[i]) || !reachable(i, got[i])) ++above_unexplained;
    } else if (ulps > 1 && ulps <= kOutlierUlps) {
      ++mid;
      if (!reachable(i, ref[i]) || !reachable(i, got[i])) ++mid_unexplained;
    }
  }
  if (above != 0 || mid != 0) {
    std::printf("|   %s, from GEMM results within 1 ULP of the exact sums: %zu outputs above scaled 4 (worst %.2f), %zu not reproduced%s; %zu outputs 2..%d ULPs apart, %zu not reproduced\n",
                op, above, worst, above_unexplained, above_unexplained == 0 ? "" : " FAILED", mid, kOutlierUlps, mid_unexplained);
  }
  if (above_unexplained != 0) ++g_failures;
}

void check_situ_flips(const Bits& ref, const Bits& got, const Bits& x, std::size_t ldx, const Bits& w, std::size_t ldw, std::size_t f, std::size_t k) {
  check_flips("SiTU", ref, got, [&](std::size_t i, bf16_bits out) {
    const std::size_t row = i / f, c = i % f;
    const bf16_bits h_bits = to_bf16(static_cast<float>(exact_dot(x.data() + row * ldx, w.data() + c * ldw, k)));
    const bf16_bits g_bits = to_bf16(static_cast<float>(exact_dot(x.data() + row * ldx, w.data() + (f + c) * ldw, k)));
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {
        if (ulp_distance(to_bf16(host_situ(from_bf16(bf16_step(h_bits, a)), from_bf16(bf16_step(g_bits, b)))), out) <= 1) return true;
      }
    }
    return false;
  });
}

void check_gated_flips(const Bits& ref, const Bits& got, const Bits& residual, std::size_t ldr, const Bits& x, std::size_t ldx, const Bits& w, std::size_t ldw, const bf16_bits* gate,
                       std::size_t n, std::size_t k) {
  check_flips("gated residual", ref, got, [&](std::size_t i, bf16_bits out) {
    const std::size_t r = i / n, c = i % n;
    const bf16_bits ab = to_bf16(static_cast<float>(exact_dot(x.data() + r * ldx, w.data() + c * ldw, k)));
    for (int a = -1; a <= 1; ++a) {
      const float delta = host_rbf(from_bf16(bf16_step(ab, a)) * from_bf16(gate[c]));
      if (ulp_distance(to_bf16(from_bf16(residual[r * ldr + c]) + delta), out) <= 1) return true;
    }
    return false;
  });
}

// q, k, v: [rows, heads, 128] with one row stride.
void report_attention_outliers(const Bits& ref, const Bits& got, const bf16_bits* q, const bf16_bits* k, const bf16_bits* v, std::size_t ld, std::size_t ldo,
                               const std::vector<std::int32_t>& cu_q, const std::vector<std::int32_t>& cu_k) {
  report_outliers("fp64 replay with bf16 weights", ref, got, [&](std::size_t i) {
    const std::size_t r = i / ldo, head = i % ldo / kHeadDim, d = i % kHeadDim;
    const auto seq = static_cast<std::size_t>(std::upper_bound(cu_q.begin(), cu_q.end(), static_cast<std::int32_t>(r)) - cu_q.begin() - 1);
    const std::size_t k0 = static_cast<std::size_t>(cu_k[seq]) * ld + head * kHeadDim;
    return replay_attention_row<double>(q + r * ld + head * kHeadDim, k + k0, static_cast<std::int64_t>(ld), v + k0, static_cast<std::int64_t>(ld),
                                        cu_k[seq + 1] - cu_k[seq], kScale)[d];
  });
}

void sync(hipStream_t stream) { check(hipStreamSynchronize(stream), "hipStreamSynchronize"); }

Buf device_bits(const Bits& h) {
  Buf b(h.size());
  b.upload(h);
  return b;
}

// Edge inputs, applied to a host matrix [rows, cols] of hashed values.
enum class Edge { kNone, kZeroRow, kNan, kInf, kDenormal, kWideRows };
constexpr Edge kEdges[] = {Edge::kZeroRow, Edge::kNan, Edge::kInf, Edge::kDenormal, Edge::kWideRows};
const char* edge_name(Edge e) {
  switch (e) {
    case Edge::kZeroRow: return "zero row";
    case Edge::kNan: return "one NaN";
    case Edge::kInf: return "one Inf";
    case Edge::kDenormal: return "denormal row";
    case Edge::kWideRows: return "rows 2^-60..2^60";
    default: return "";
  }
}
constexpr bf16_bits kNanBits = 0x7fc0, kInfBits = 0x7f80;

void apply_edge(Edge edge, Bits& m, std::size_t rows, std::size_t cols) {
  const std::size_t row = rows / 2;
  switch (edge) {
    case Edge::kZeroRow: std::fill_n(m.begin() + static_cast<std::ptrdiff_t>(row * cols), cols, bf16_bits{0}); break;
    case Edge::kNan: m[row * cols + cols / 3] = kNanBits; break;
    case Edge::kInf: m[row * cols + cols / 3] = kInfBits; break;
    case Edge::kDenormal:
      for (std::size_t c = 0; c < cols; ++c) m[row * cols + c] = to_bf16(std::ldexp(from_bf16(m[row * cols + c]), -130));
      break;
    case Edge::kWideRows:
      for (std::size_t r = 0; r < rows; ++r) {
        const int exp2_of_row = rows < 2 ? 0 : static_cast<int>(-60 + 120 * r / (rows - 1));
        for (std::size_t c = 0; c < cols; ++c) m[r * cols + c] = to_bf16(std::ldexp(from_bf16(m[r * cols + c]), exp2_of_row));
      }
      break;
    default: break;
  }
}

Bits edged(Bits m, Edge edge, std::size_t rows, std::size_t cols) {
  apply_edge(edge, m, rows, cols);
  return m;
}

// Rows of d values through both, all the seeds' outputs in one comparison.
void rms_norm_case(const std::string& name, std::int64_t rows, int d, const std::vector<std::uint64_t>& seeds, Edge edge, unsigned exp_spread, Gate gate, hipStream_t s) {
  const Buf w = device_bits(synth_bf16(41, d, 0)), scale = device_bits(synth_bf16(42, d, -2)), shift = device_bits(synth_bf16(43, d, -2));
  Bits all_ref, all_got;
  for (const std::uint64_t seed : seeds) {
    const auto n = static_cast<std::size_t>(rows * d);
    const Buf x = device_bits(edged(synth_bf16(seed, n, 0, exp_spread), edge, static_cast<std::size_t>(rows), static_cast<std::size_t>(d)));
    Buf ref(n), got(n);
    mi::kernels::rms_norm_modulate_moss(mat(x.get(), rows, d), vec(w.get(), d), vec(scale.get(), d), vec(shift.get(), d), mat(ref.get(), rows, d), s);
    rms_norm_modulate(x.get(), d, w.get(), scale.get(), shift.get(), got.get(), d, rows, d, s);
    sync(s);
    const Bits r = ref.download(), g = got.download();
    all_ref.insert(all_ref.end(), r.begin(), r.end());
    all_got.insert(all_got.end(), g.begin(), g.end());
  }
  report(name, all_ref, all_got, gate);
}

void check_rms_norm(hipStream_t s) {
  // From 16 rows on the prototype and the engine sum squares in one order: exact, on every input. Below 16 rows the engine's
  // block width grows (kernels/basic/norm.cu) and the prototype's does not, so the fp32 sum can differ in its last bits. That
  // is not bit-exact: it takes many inputs to see a flip (about 5 outputs in 10^6 at 7 to 15 rows of 3072), so these run 400
  // seeds, wide exponents too, at the accumulated gate. The DiT never runs fewer than 16 tokens at an image of 64x64 or more.
  const std::vector<std::uint64_t> seeds = {44, 45, 46, 47, 48, 49, 50, 51};
  std::vector<std::uint64_t> many;
  for (std::uint64_t seed = 1000; seed < 1400; ++seed) many.push_back(seed);
  for (const std::int64_t rows : {1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 37, 1000}) {
    for (const int d : {136, 3072}) {
      const std::string name = "rms_norm_modulate " + std::to_string(rows) + "x" + std::to_string(d);
      const bool below = rows < 16;
      const std::vector<std::uint64_t> use = below ? many : rows < 1000 ? seeds : std::vector<std::uint64_t>{44};
      const Gate gate = below ? Gate::kAccumulated : Gate::kExact;
      rms_norm_case(name, rows, d, use, Edge::kNone, 4, gate, s);
      rms_norm_case(name + " wide exp", rows, d, use, Edge::kNone, 32, gate, s);
    }
  }
}

struct Shape {
  int m, n, k;
  std::string text() const { return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k); }
};

void gemm_case(const std::string& name, const Shape& c, bool cublaslt, Edge edge, mi::kernels::Gemm& engine, void* ws, hipStream_t s) {
  const auto m = static_cast<std::size_t>(c.m), n = static_cast<std::size_t>(c.n), k = static_cast<std::size_t>(c.k);
  const Bits x_h = edged(synth_bf16(51, m * k, 0), edge, m, k), w_h = synth_bf16(52, n * k, -5);
  const Buf x = device_bits(x_h), w = device_bits(w_h);
  Buf ref(m * n), got(m * n);
  if (cublaslt) {
    engine.linear(mat(ref.get(), c.m, c.n), mat(x.get(), c.m, c.k), mat(w.get(), c.n, c.k), nullptr, ws, mi::kernels::Gemm::kWorkspaceBytes, s);
  } else {
    mi::kernels::fused::gemm(mat(ref.get(), c.m, c.n), mat(x.get(), c.m, c.k), mat(w.get(), c.n, c.k), s);
  }
  gemm(got.get(), c.n, x.get(), c.k, w.get(), c.k, c.m, c.n, c.k, s);
  sync(s);
  const Bits ref_h = ref.download(), got_h = got.download();
  report(name, ref_h, got_h, Gate::kAccumulated);
  report_gemm_outliers(ref_h, got_h, x_h, k, w_h, k, n, k);
}

// The fused CUTLASS GEMM needs K and N multiples of 8; other shapes compare against cuBLASLt.
void check_gemm(mi::kernels::Gemm& engine, void* ws, hipStream_t s) {
  for (const Shape c : {Shape{1, 64, 16}, Shape{15, 24, 8}, Shape{17, 40, 1000}, Shape{37, 3080, 3000}, Shape{33, 56, 3072}, Shape{300, 9216, 3072}}) {
    gemm_case("gemm " + c.text() + " (CUTLASS)", c, false, Edge::kNone, engine, ws, s);
  }
  for (const Shape c : {Shape{33, 130, 1003}, Shape{20, 31, 77}}) gemm_case("gemm " + c.text() + " (cuBLASLt)", c, true, Edge::kNone, engine, ws, s);
}

void qk_case(const std::string& name, int tokens, Edge edge, unsigned exp_spread, hipStream_t s) {
  const Buf q_norm = device_bits(synth_bf16(61, kHeadDim, 0)), k_norm = device_bits(synth_bf16(62, kHeadDim, 0));
  const auto n = static_cast<std::size_t>(tokens);
  constexpr std::size_t ld = 3 * kDim;
  const Bits in = edged(synth_bf16(63, n * ld, 0, exp_spread), edge, n, ld);
  Buf ref = device_bits(in), got = device_bits(in);
  const Buf table = device_bits(synth_rope(64, n));
  mi::kernels::fused::qk_norm_rope_moss(mat(ref.get(), tokens, ld), kHeads, vec(q_norm.get(), kHeadDim), vec(k_norm.get(), kHeadDim), mat(table.get(), tokens, kHeadDim), s);
  qk_norm_rope(got.get(), ld, tokens, kHeads, q_norm.get(), k_norm.get(), table.get(), kHeadDim, s);
  sync(s);
  report(name, ref.download(), got.download(), Gate::kExact);
}

void check_qk_norm_rope(hipStream_t s) {
  for (const int tokens : {1, 37, 300}) {
    qk_case("qk_norm_rope " + std::to_string(tokens) + " tokens", tokens, Edge::kNone, 4, s);
    qk_case("qk_norm_rope " + std::to_string(tokens) + " tokens wide exp", tokens, Edge::kNone, 32, s);
  }
}

// Hashed q, k, v [rows, heads, 128]. `sharpen` multiplies q and k: S * scale then has a standard deviation near
// sharpen^2, near 1 spreading the softmax, larger making it nearly one-hot and working the rescale harder.
struct AttentionInput {
  Bits q, k, v;
};
constexpr int kAttnLd = kHeads * kHeadDim;
AttentionInput attention_input(int total_q, int total_k, float sharpen) {
  const auto values = [&](std::uint64_t seed, int rows) {
    Bits b = synth_bf16(seed, static_cast<std::size_t>(rows) * kAttnLd, 0);
    for (bf16_bits& e : b) e = to_bf16(from_bf16(e) * sharpen);
    return b;
  };
  return {values(71, total_q), values(72, total_k), synth_bf16(73, static_cast<std::size_t>(total_k) * kAttnLd, 0)};
}

std::vector<std::int32_t> cumulative(const std::vector<int>& lengths) {
  std::vector<std::int32_t> cu = {0};
  for (const int l : lengths) cu.push_back(cu.back() + l);
  return cu;
}

void attention_case(const std::string& name, const std::vector<int>& q_lengths, const std::vector<int>& k_lengths, const AttentionInput& in, hipStream_t s,
                    Gate gate = Gate::kAccumulated, bool show_specials = false) {
  const std::vector<std::int32_t> cu_q = cumulative(q_lengths), cu_k = cumulative(k_lengths);
  const int max_seqlen = std::max(*std::max_element(q_lengths.begin(), q_lengths.end()), *std::max_element(k_lengths.begin(), k_lengths.end()));
  const int total_q = cu_q.back(), total_k = cu_k.back();
  constexpr int ld = kAttnLd;
  Buf q = device_bits(in.q), k = device_bits(in.k), v = device_bits(in.v);
  DevBuffer<std::int32_t> cu_q_dev(cu_q.size()), cu_k_dev(cu_k.size());
  cu_q_dev.upload(cu_q);
  cu_k_dev.upload(cu_k);
  Buf ref(static_cast<std::size_t>(total_q) * ld), got(ref.size());
  const auto cu_tensor = [](DevBuffer<std::int32_t>& b) {
    return Tensor::view(b.get(), DType::kI32, mi::Device::kCUDA, {static_cast<std::int64_t>(b.size())});
  };
  const std::size_t lse_bytes = mi::attention::flash_fwd_workspace_bytes(total_q, kHeads);
  DevBuffer<float> lse(lse_bytes / sizeof(float));
  mi::attention::flash_fwd(heads_of(q.get(), total_q, kHeads, ld), heads_of(k.get(), total_k, kHeads, ld), heads_of(v.get(), total_k, kHeads, ld),
                           heads_of(ref.get(), total_q, kHeads, ld), cu_tensor(cu_q_dev), cu_tensor(cu_k_dev), max_seqlen, false, kScale, lse.get(),
                           lse_bytes, s);
  attention({q.get(), k.get(), v.get(), got.get(), ld, ld, ld, ld, kHeadDim, kHeadDim, kHeadDim, kHeadDim, cu_q_dev.get(), cu_k_dev.get(),
             static_cast<int>(q_lengths.size()), max_seqlen, kHeads, kScale},
            s);
  sync(s);
  const Bits ref_h = ref.download(), got_h = got.download();
  report(name, ref_h, got_h, gate);
  report_attention_outliers(ref_h, got_h, in.q.data(), in.k.data(), in.v.data(), ld, ld, cu_q, cu_k);
  if (show_specials) {
    const auto count = [](const Bits& b, const std::function<bool(bf16_bits)>& f) { return std::count_if(b.begin(), b.end(), f); };
    const auto zero = [](bf16_bits b) { return (b & 0x7fffu) == 0; };
    std::printf("|   NaN outputs: engine %td, prototype %td; zero outputs: engine %td, prototype %td\n", count(ref_h, is_nan), count(got_h, is_nan), count(ref_h, zero),
                count(got_h, zero));
  }
  // Outputs NaN on one side only: what the replays make of the first few.
  int shown = 0;
  for (std::size_t i = 0; i < ref_h.size() && shown < 3; ++i) {
    if (is_nan(ref_h[i]) == is_nan(got_h[i])) continue;
    i += static_cast<std::size_t>(shown) * 4000;  // spread them out over the affected rows
    if (i >= ref_h.size() || is_nan(ref_h[i]) == is_nan(got_h[i])) break;
    ++shown;
    const std::size_t r = i / ld, head = i % ld / kHeadDim, d = i % kHeadDim;
    const auto seq = static_cast<std::size_t>(std::upper_bound(cu_q.begin(), cu_q.end(), static_cast<std::int32_t>(r)) - cu_q.begin() - 1);
    const std::size_t k0 = static_cast<std::size_t>(cu_k[seq]) * ld + head * kHeadDim;
    const bf16_bits* qr = in.q.data() + r * ld + head * kHeadDim;
    const float fp32 = replay_attention_row<float>(qr, in.k.data() + k0, ld, in.v.data() + k0, ld, cu_k[seq + 1] - cu_k[seq], kScale)[d];
    const double fp64 = replay_attention_row<double>(qr, in.k.data() + k0, ld, in.v.data() + k0, ld, cu_k[seq + 1] - cu_k[seq], kScale)[d];
    std::printf("|   row %zu head %zu dim %zu: engine %.6g, prototype %.6g, fp32 replay %.6g, fp64 replay %.6g\n", r, head, d, static_cast<double>(from_bf16(ref_h[i])),
                static_cast<double>(from_bf16(got_h[i])), static_cast<double>(fp32), fp64);
  }
}

void check_attention(hipStream_t s) {
  const std::vector<int> six = {37, 64, 16, 129, 1, 257};
  attention_case("attention 1 row", {1}, {1}, attention_input(1, 1, 1.0f), s);
  attention_case("attention 1 row, 15 tokens", {15}, {15}, attention_input(15, 15, 1.0f), s);
  attention_case("attention 1 row, 33 tokens", {33}, {33}, attention_input(33, 33, 1.0f), s);
  attention_case("attention varlen, 6 rows", six, six, attention_input(504, 504, 1.0f), s);
  attention_case("attention q != k lengths", {37, 64, 5}, {40, 33, 129}, attention_input(106, 202, 1.0f), s);
  attention_case("attention varlen, sharp", six, six, attention_input(504, 504, 3.0f), s);
  attention_case("attention 1000+292+17", {1000, 292, 17}, {1000, 292, 17}, attention_input(1309, 1309, 1.0f), s);
}

void gated_case(const std::string& name, const Shape& c, Edge edge, hipStream_t s) {
  const auto m = static_cast<std::size_t>(c.m), n = static_cast<std::size_t>(c.n), k = static_cast<std::size_t>(c.k);
  const Bits x_h = edged(synth_bf16(81, m * k, 0), edge, m, k), w_h = synth_bf16(82, n * k, -5), gate_h = synth_bf16(83, n, -2);
  const Buf x = device_bits(x_h), w = device_bits(w_h), gate = device_bits(gate_h);
  const Bits residual = synth_bf16(84, m * n, 0);
  Buf ref = device_bits(residual), got = device_bits(residual);
  mi::kernels::fused::gemm_gated_residual(mat(ref.get(), c.m, c.n), mat(x.get(), c.m, c.k), mat(w.get(), c.n, c.k), vec(gate.get(), c.n), s);
  gemm_gated_residual(got.get(), c.n, x.get(), c.k, w.get(), c.k, gate.get(), c.m, c.n, c.k, s);
  sync(s);
  const Bits ref_h = ref.download(), got_h = got.download();
  report(name, ref_h, got_h, Gate::kFlipChecked);
  report_gated_outliers(ref_h, got_h, residual, n, x_h, k, w_h, k, gate_h.data(), n, k);
  check_gated_flips(ref_h, got_h, residual, n, x_h, k, w_h, k, gate_h.data(), n, k);
}

void situ_case(const std::string& name, const Shape& c, Edge edge, hipStream_t s) {
  const auto m = static_cast<std::size_t>(c.m), f = static_cast<std::size_t>(c.n), k = static_cast<std::size_t>(c.k);
  const Bits x_h = edged(synth_bf16(85, m * k, 0), edge, m, k), w_h = synth_bf16(86, 2 * f * k, -5);
  const Buf x = device_bits(x_h), w = device_bits(w_h);
  Buf ref(m * f), got(m * f);
  mi::kernels::fused::gemm_situ_glu(mat(ref.get(), c.m, c.n), mat(x.get(), c.m, c.k), mat(w.get(), 2 * c.n, c.k), s);
  gemm_situ_glu(got.get(), c.n, x.get(), c.k, w.get(), c.k, c.m, c.n, c.k, s);
  sync(s);
  const Bits ref_h = ref.download(), got_h = got.download();
  report(name, ref_h, got_h, Gate::kFlipChecked);
  report_situ_outliers(ref_h, got_h, x_h, k, w_h, k, f, k);
  check_situ_flips(ref_h, got_h, x_h, k, w_h, k, f, k);
}

void check_fused_epilogues(hipStream_t s) {
  for (const Shape c : {Shape{5, 24, 8}, Shape{37, 3080, 3000}, Shape{129, 3072, 8192}, Shape{20, 56, 3072}}) {
    gated_case("gemm_gated_residual " + c.text(), c, Edge::kNone, s);
  }
  for (const Shape c : {Shape{5, 8, 8}, Shape{37, 24, 1000}, Shape{70, 16, 3072}, Shape{5, 8192, 3072}}) situ_case("gemm_situ_glu " + c.text(), c, Edge::kNone, s);
}

// Edge inputs. Attention's branches: a NaN in K or V, q = +Inf where every key is negative (every score -Inf: the kernel's
// max == -Inf and sum == 0 branches, over one softmax step and over several).
void check_edges(mi::kernels::Gemm& engine, void* ws, hipStream_t s) {
  for (const Edge e : kEdges) {
    const std::string tag = std::string(" [") + edge_name(e) + "]";
    rms_norm_case("rms_norm_modulate 37x3072" + tag, 37, 3072, {44}, e, 4, Gate::kExact, s);
    qk_case("qk_norm_rope 37 tokens" + tag, 37, e, 4, s);
    gemm_case("gemm 37x3080x3000" + tag, {37, 3080, 3000}, false, e, engine, ws, s);
    gated_case("gated 37x3080x3000" + tag, {37, 3080, 3000}, e, s);
    situ_case("situ 37x24x1000" + tag, {37, 24, 1000}, e, s);
  }

  const std::vector<int> lengths = {37, 64, 16, 129, 1};
  constexpr int total = 247;
  const auto n = static_cast<std::size_t>(total);
  const AttentionInput base = attention_input(total, total, 1.0f);
  const auto with = [&](const std::string& what, const std::function<void(AttentionInput&)>& edit) {
    AttentionInput in = base;
    edit(in);
    attention_case("attention [" + what + "]", lengths, lengths, in, s, Gate::kAccumulated, true);
  };
  with("zero q row", [&](AttentionInput& in) { apply_edge(Edge::kZeroRow, in.q, n, kAttnLd); });
  with("zero k row", [&](AttentionInput& in) { apply_edge(Edge::kZeroRow, in.k, n, kAttnLd); });
  with("zero v row", [&](AttentionInput& in) { apply_edge(Edge::kZeroRow, in.v, n, kAttnLd); });
  with("all k of a sequence zero", [&](AttentionInput& in) { std::fill(in.k.begin() + 37 * kAttnLd, in.k.begin() + 101 * kAttnLd, bf16_bits{0}); });
  with("one NaN in k", [&](AttentionInput& in) { apply_edge(Edge::kNan, in.k, n, kAttnLd); });
  with("one NaN in v", [&](AttentionInput& in) { apply_edge(Edge::kNan, in.v, n, kAttnLd); });
  with("one NaN in q", [&](AttentionInput& in) { apply_edge(Edge::kNan, in.q, n, kAttnLd); });
  with("one Inf in q", [&](AttentionInput& in) { apply_edge(Edge::kInf, in.q, n, kAttnLd); });
  with("one Inf in k", [&](AttentionInput& in) { apply_edge(Edge::kInf, in.k, n, kAttnLd); });
  with("one Inf in v", [&](AttentionInput& in) { apply_edge(Edge::kInf, in.v, n, kAttnLd); });
  with("denormal q row", [&](AttentionInput& in) { apply_edge(Edge::kDenormal, in.q, n, kAttnLd); });
  with("denormal k and v rows", [&](AttentionInput& in) {
    apply_edge(Edge::kDenormal, in.k, n, kAttnLd);
    apply_edge(Edge::kDenormal, in.v, n, kAttnLd);
  });
  // Query rows spanning 2^-20..2^20: scores far apart, the softmax one-hot on the large rows.
  with("q rows 2^-20..2^20", [&](AttentionInput& in) {
    for (std::size_t r = 0; r < n; ++r) {
      const int e = -20 + static_cast<int>(40 * r / (n - 1));
      for (std::size_t c = 0; c < kAttnLd; ++c) in.q[r * kAttnLd + c] = to_bf16(std::ldexp(from_bf16(in.q[r * kAttnLd + c]), e));
    }
  });
  {
    AttentionInput in = base;
    apply_edge(Edge::kWideRows, in.q, n, kAttnLd);
    attention_case("attention [q rows 2^-60..2^60]", lengths, lengths, in, s, Gate::kReportOnly, true);
  }
  with("v rows 2^-60..2^60", [&](AttentionInput& in) { apply_edge(Edge::kWideRows, in.v, n, kAttnLd); });
  // Every score -Inf at one query row of head 0: dim 7 of that row is +Inf and of every key of its sequence negative.
  const auto all_negative = [&](AttentionInput& in, int r0, int r1, int row) {
    in.q[static_cast<std::size_t>(row) * kAttnLd + 7] = kInfBits;
    for (int j = r0; j < r1; ++j) in.k[static_cast<std::size_t>(j) * kAttnLd + 7] = to_bf16(-1.0f);
  };
  with("all scores -Inf, 37-token row", [&](AttentionInput& in) { all_negative(in, 0, 37, 5); });  // two softmax steps
  with("all scores -Inf, 16-token row", [&](AttentionInput& in) { all_negative(in, 101, 117, 103); });  // one softmax step
  with("all scores -Inf, 129-token row", [&](AttentionInput& in) { all_negative(in, 117, 246, 120); });  // five softmax steps
}

// Edge inputs through run_block_op's attention, on the packed path that run_block takes and on the row-major one, each
// against the engine: the packer's zero tails, its per-sequence groups and the rescale's NaN test, which the edge rows above
// (attention() without packing) do not reach. q, k, v are one [tokens, 3 kDim] QKV as the block holds it.
void check_attention_op_edges(hipStream_t s) {
  const std::vector<int> lengths = {33, 37, 64, 16, 129, 1};
  const std::vector<std::int32_t> cu = cumulative(lengths);
  const int tokens = cu.back(), max_seqlen = *std::max_element(lengths.begin(), lengths.end());
  const auto n = static_cast<std::size_t>(tokens);
  constexpr std::size_t ld = 3 * kDim;
  const Bits base = synth_bf16(91, n * ld, 0);
  DevBuffer<std::int32_t> cu_dev(cu.size());
  cu_dev.upload(cu);
  const Tensor cu_t = Tensor::view(cu_dev.get(), DType::kI32, mi::Device::kCUDA, {static_cast<std::int64_t>(cu.size())});
  const std::size_t lse_bytes = mi::attention::flash_fwd_workspace_bytes(tokens, kHeads);
  DevBuffer<float> lse(lse_bytes / sizeof(float));
  BlockActivations pro(tokens);
  const BlockBuffers packed = pro.buffers(nullptr, nullptr, cu_dev.get(), static_cast<int>(lengths.size()), max_seqlen, tokens);
  BlockBuffers row_major = packed;
  row_major.pack = nullptr;
  row_major.pack_attn = nullptr;
  // Attention reads no weight; a packed QKV weight is what selects the packed path.
  const DevBuffer<bf16_bits> unread(1);
  BlockWeights w{};
  w.qkv_packed = unread.get();
  const auto idx = [&](std::size_t row, int part, int head, int dim) {
    return row * ld + static_cast<std::size_t>(part) * kDim + static_cast<std::size_t>(head) * kHeadDim + static_cast<std::size_t>(dim);
  };
  const auto run = [&](const std::string& name, const std::function<void(Bits&)>& edit) {
    Bits in = base;
    edit(in);
    Buf qkv = device_bits(in);
    Buf ref(n * kDim);
    const auto l = static_cast<std::int64_t>(tokens);
    mi::attention::flash_fwd(heads_of(qkv.get(), l, kHeads, ld), heads_of(qkv.get() + kDim, l, kHeads, ld), heads_of(qkv.get() + 2 * kDim, l, kHeads, ld),
                             heads_of(ref.get(), l, kHeads, kDim), cu_t, cu_t, max_seqlen, false, kScale, lse.get(), lse_bytes, s);
    sync(s);
    const Bits ref_h = ref.download();
    for (const bool use_packed : {true, false}) {
      pro.poison(s);
      pro.qkv.upload(in);
      run_block_op(3, w, use_packed ? packed : row_major, s);
      sync(s);
      report(std::string("op 3 ") + (use_packed ? "packed" : "row-major") + " [" + name + "]", ref_h, pro.attn.download(), Gate::kAccumulated);
    }
  };
  run("plain", [](Bits&) {});
  run("NaN k", [&](Bits& in) { in[idx(40, 1, 3, 7)] = kNanBits; });
  run("NaN v", [&](Bits& in) { in[idx(40, 2, 3, 7)] = kNanBits; });
  run("NaN q", [&](Bits& in) { in[idx(40, 0, 3, 7)] = kNanBits; });
  run("Inf k", [&](Bits& in) { in[idx(40, 1, 3, 7)] = kInfBits; });
  run("Inf v", [&](Bits& in) { in[idx(40, 2, 3, 7)] = kInfBits; });
  run("Inf q", [&](Bits& in) { in[idx(40, 0, 3, 7)] = kInfBits; });
  // The next sequence's first token NaN or Inf in every head: it must not reach this sequence through its tail.
  run("NaN v, row 1's first", [&](Bits& in) {
    for (int h = 0; h < kHeads; ++h) {
      for (int d = 0; d < kHeadDim; ++d) in[idx(33, 2, h, d)] = kNanBits;
    }
  });
  run("Inf k, row 1's first", [&](Bits& in) {
    for (int h = 0; h < kHeads; ++h) {
      for (int d = 0; d < kHeadDim; ++d) in[idx(33, 1, h, d)] = kInfBits;
    }
  });
  run("all k of row 2 zero", [&](Bits& in) {
    for (std::size_t r = 70; r < 134; ++r) std::fill_n(in.begin() + static_cast<std::ptrdiff_t>(r * ld + kDim), kDim, bf16_bits{0});
  });
  // The 33-token row's last query row is a tile of one row, over two key steps. Its scores all -Inf at head 0 make the
  // second step's rescale NaN.
  run("-Inf scores, 1-row tile", [&](Bits& in) {
    in[idx(32, 0, 0, 7)] = kInfBits;
    for (int j = 0; j < 33; ++j) in[idx(static_cast<std::size_t>(j), 1, 0, 7)] = to_bf16(-1.0f);
  });
  // The first step walked is the row's last key: all NaN at head 0, so the second step starts from a NaN maximum.
  run("last key all NaN", [&](Bits& in) {
    for (int d = 0; d < kHeadDim; ++d) in[idx(32, 1, 0, d)] = kNanBits;
  });
  // Every score -Inf for a full tile of 16 query rows of the 64-token row, over its four key steps.
  run("-Inf scores, full tile", [&](Bits& in) {
    for (int r = 70; r < 86; ++r) in[idx(static_cast<std::size_t>(r), 0, 0, 7)] = kInfBits;
    for (int j = 70; j < 134; ++j) in[idx(static_cast<std::size_t>(j), 1, 0, 7)] = to_bf16(-1.0f);
  });
}

// One block run: real or synthetic weights and inputs, device side, at the run's first token.
struct BlockCase {
  std::string name;
  BlockWeights w;
  md::BlockTensors bt;
  const bf16_bits* x_init;
  const bf16_bits* rope;
  const bf16_bits* mod;
  std::vector<int> lengths;
  const Bits* expected_engine_x = nullptr;  // the engine's own output for this run, when captured from its forward
  bool real = false;                        // real activations: the control, its block gate, the substitution rows
};

// Per op, both implementations get the engine's previous output as input, so each row is that op's own error; the engine's
// chain is checked to be run_block_fused bit for bit. Then the prototype's whole block runs on its own outputs and its
// final residual stream is compared, which also shows how errors accumulate through a block.
void check_block(const BlockCase& c, mi::kernels::Gemm& engine, void* ws, const md::FusedRouteSnapshot& snapshot, hipStream_t s) {
  const std::vector<std::int32_t> cu = cumulative(c.lengths);
  const int max_seqlen = *std::max_element(c.lengths.begin(), c.lengths.end());
  const int tokens = cu.back();
  const auto l = static_cast<std::int64_t>(tokens);
  const auto n = static_cast<std::size_t>(tokens);
  BlockActivations pro(tokens);
  DevBuffer<std::int32_t> cu_dev(cu.size());
  cu_dev.upload(cu);
  const PackedWeights packed_weights(c.w, s);
  const BlockWeights w = packed_weights.apply(c.w);
  const md::BlockTensors& bt = c.bt;
  const BlockBuffers buffers = pro.buffers(c.rope, c.mod, cu_dev.get(), static_cast<int>(c.lengths.size()), max_seqlen, tokens);

  // run_block_fused on its buffers. `big` as MossDit sizes it for a run with some pair op by op, whatever the route.
  Buf big(static_cast<std::size_t>(l * md::kOpByOpBigCols));
  Buf fused_x(n * kDim), fused_y(n * kDim);
  const std::size_t lse_bytes = mi::attention::flash_fwd_workspace_bytes(l, kHeads);
  DevBuffer<float> lse(lse_bytes / sizeof(float));
  const Tensor cu_tensor = Tensor::view(cu_dev.get(), DType::kI32, mi::Device::kCUDA, {static_cast<std::int64_t>(cu.size())});
  const auto copy = [&](bf16_bits* to, const bf16_bits* from, std::size_t count) {
    check(hipMemcpyAsync(to, from, count * sizeof(bf16_bits), hipMemcpyDeviceToDevice, s), "hipMemcpyAsync");
  };
  // Prototype ops read buffers set to the engine's state: before each op every buffer the ops write is poisoned, then its
  // inputs are restored from the engine's buffers.
  const auto restore = [&](const Buf& to, const Buf& from) { copy(to.get(), from.get(), to.size()); };
  copy(fused_x.get(), c.x_init, fused_x.size());
  const md::FusedBlockBuffers fb{mat(fused_x.get(), l, kDim), mat(fused_y.get(), l, kDim), mat(big.get(), l, md::kOpByOpBigCols),
                                 mat(c.rope, l, kHeadDim), vec(c.mod, 6 * kDim), lse.get(), lse_bytes};
  const md::BlockRun run{0, l, cu_tensor, max_seqlen};
  const md::QkvGemm qkv_gemm{&engine, ws, mi::kernels::Gemm::kWorkspaceBytes};
  const md::FusedRoute route = md::fused_route(l, snapshot.validated);
  std::printf("\n%s: %d tokens in %zu rows; engine route: o %s, up_gate %s, down %s\n", c.name.c_str(), tokens, c.lengths.size(),
              route.o ? "fused" : "op by op", route.up_gate ? "fused" : "op by op", route.down ? "fused" : "op by op");
  // The chain below always runs the fused GEMMs, so it can only match a fused route.
  if (!route.o || !route.up_gate || !route.down) {
    std::printf("| FAILED: this shape must route every pair fused for the chain below to be the engine's block\n");
    ++g_failures;
  }
  md::run_block_fused(bt, fb, run, qkv_gemm, snapshot, s);
  sync(s);
  const Bits final_engine = fused_x.download();
  if (c.expected_engine_x != nullptr) {
    std::size_t diff = 0;
    for (std::size_t i = 0; i < final_engine.size(); ++i) diff += final_engine[i] != (*c.expected_engine_x)[i];
    std::printf("run_block_fused here vs the engine's forward: %zu of %zu values differ\n", diff, final_engine.size());
    if (diff != 0) ++g_failures;
  }

  // Per-op chain on the engine's activations.
  print_header();
  const bool dumping = !g_dump_dir.empty() && c.real && c.name.find("E13") != std::string::npos;
  fs::path dump_path;
  if (dumping) {
    std::string dir_name = c.name.substr(c.name.rfind("block"));
    std::replace(dir_name.begin(), dir_name.end(), ' ', '_');
    dump_path = fs::path(g_dump_dir) / dir_name;
  }
  if (dumping) fs::create_directories(dump_path);
  const auto dump_bits = [&](const char* name, const Bits& v) {
    if (!dumping) return;
    std::ofstream f(dump_path / (std::string(name) + ".bin"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(bf16_bits)));
    if (!f) throw std::runtime_error(std::string("cannot write ") + name);
  };
  const auto mod = [&](int i) { return c.mod + static_cast<std::int64_t>(i) * kDim; };
  Buf ex(n * kDim), ey(n * kDim), eqkv(n * 3 * kDim), eattn(n * kDim), effn(n * kFfnDim);
  copy(ex.get(), c.x_init, ex.size());
  const Tensor tx = mat(ex.get(), l, kDim), ty = mat(ey.get(), l, kDim), tq = mat(eqkv.get(), l, 3 * kDim), ta = mat(eattn.get(), l, kDim),
               tf = mat(effn.get(), l, kFfnDim);
  const auto host = [](const bf16_bits* p, std::size_t count) {
    Bits h(count);
    check(hipMemcpy(h.data(), p, count * sizeof(bf16_bits), hipMemcpyDeviceToHost), "hipMemcpy");
    return h;
  };
  std::array<Agreement, kBlockOps> scalar_figures{};  // for the dump: what hip_block_replay compares, on this leaf
  const auto op_out = [&](int op, Gate gate, const Buf& ref, const Buf& got, int p999_limit = 1) {
    sync(s);
    const Bits ref_h = ref.download(), got_h = got.download();
    report(block_op_name(op), ref_h, got_h, gate, {}, p999_limit);
    if (dumping) {
      const std::size_t row_len = ref_h.size() / n;
      scalar_figures[static_cast<std::size_t>(op)] = agreement(ref_h, got_h, row_len, op == 2 ? 2 * kDim : row_len);  // qk_norm_rope: q and k
    }
    return std::make_pair(ref_h, got_h);
  };

  pro.poison(s);
  restore(pro.x, ex);
  mi::kernels::rms_norm_modulate_moss(tx, bt.norm1, vec(mod(1), kDim), vec(mod(0), kDim), ty, s);
  run_block_op(0, w, buffers, s);
  op_out(0, Gate::kExact, ey, pro.y);
  if (dumping) {
    const auto hd = [&](const char* name, const bf16_bits* p, std::size_t count) { dump_bits(name, host(p, count)); };
    hd("norm1", w.norm1, kDim);
    hd("norm2", w.norm2, kDim);
    hd("q_norm", w.q_norm, kHeadDim);
    hd("k_norm", w.k_norm, kHeadDim);
    hd("w_qkv", w.qkv, std::size_t{3} * kDim * kDim);
    hd("w_o", w.o, std::size_t{kDim} * kDim);
    hd("w_up_gate", w.up_gate, std::size_t{2} * kFfnDim * kDim);
    hd("w_down", w.down, std::size_t{kDim} * kFfnDim);
    hd("rope", c.rope, n * kHeadDim);
    hd("mod", c.mod, std::size_t{6} * kDim);
    dump_bits("x0", ex.download());
    dump_bits("y0", ey.download());
  }

  pro.poison(s);
  restore(pro.y, ey);
  engine.linear(tq, ty, bt.qkv, nullptr, ws, mi::kernels::Gemm::kWorkspaceBytes, s);
  run_block_op(1, w, buffers, s);
  {
    const auto [ref_h, got_h] = op_out(1, Gate::kAccumulated, eqkv, pro.qkv);
    report_gemm_outliers(ref_h, got_h, ey.download(), kDim, host(w.qkv, std::size_t{3} * kDim * kDim), kDim, 3 * kDim, kDim);
    dump_bits("qkv1", ref_h);
  }

  pro.poison(s);
  restore(pro.qkv, eqkv);
  mi::kernels::fused::qk_norm_rope_moss(tq, kHeads, bt.q_norm, bt.k_norm, mat(c.rope, l, kHeadDim), s);
  run_block_op(2, w, buffers, s);
  dump_bits("qkv2", op_out(2, Gate::kExact, eqkv, pro.qkv).first);

  pro.poison(s);
  restore(pro.qkv, eqkv);
  mi::attention::flash_fwd(heads_of(eqkv.get(), l, kHeads, 3 * kDim), heads_of(eqkv.get() + kDim, l, kHeads, 3 * kDim),
                           heads_of(eqkv.get() + 2 * kDim, l, kHeads, 3 * kDim), heads_of(eattn.get(), l, kHeads, kDim), cu_tensor, cu_tensor,
                           max_seqlen, false, kScale, lse.get(), lse_bytes, s);
  run_block_op(3, w, buffers, s);
  {
    const auto [ref_h, got_h] = op_out(3, Gate::kAccumulated, eattn, pro.attn);
    const Bits qkv_h = eqkv.download();
    dump_bits("attn", ref_h);
    report_attention_outliers(ref_h, got_h, qkv_h.data(), qkv_h.data() + kDim, qkv_h.data() + 2 * kDim, 3 * kDim, kDim, cu, cu);
  }

  const Bits x_before_o = ex.download();
  pro.poison(s);
  restore(pro.attn, eattn);
  restore(pro.x, ex);
  mi::kernels::fused::gemm_gated_residual(tx, ta, bt.o, vec(mod(2), kDim), s);
  run_block_op(4, w, buffers, s);
  {
    const auto [ref_h, got_h] = op_out(4, Gate::kFlipChecked, ex, pro.x);
    dump_bits("x4", ref_h);
    const Bits attn_h = eattn.download(), o_h = host(w.o, std::size_t{kDim} * kDim), gate_h = host(mod(2), kDim);
    report_gated_outliers(ref_h, got_h, x_before_o, kDim, attn_h, kDim, o_h, kDim, gate_h.data(), kDim, kDim);
    check_gated_flips(ref_h, got_h, x_before_o, kDim, attn_h, kDim, o_h, kDim, gate_h.data(), kDim, kDim);
  }

  pro.poison(s);
  restore(pro.x, ex);
  mi::kernels::rms_norm_modulate_moss(tx, bt.norm2, vec(mod(4), kDim), vec(mod(3), kDim), ty, s);
  run_block_op(5, w, buffers, s);
  dump_bits("y5", op_out(5, Gate::kExact, ey, pro.y).first);

  pro.poison(s);
  restore(pro.y, ey);
  mi::kernels::fused::gemm_situ_glu(tf, ty, bt.up_gate, s);
  run_block_op(6, w, buffers, s);
  {
    const auto [ref_h, got_h] = op_out(6, Gate::kFlipChecked, effn, pro.ffn, c.real ? 2 : 1);
    dump_bits("ffn", ref_h);
    const Bits y_h = ey.download(), up_gate_h = host(w.up_gate, std::size_t{2} * kFfnDim * kDim);
    report_situ_outliers(ref_h, got_h, y_h, kDim, up_gate_h, kDim, kFfnDim, kDim);
    check_situ_flips(ref_h, got_h, y_h, kDim, up_gate_h, kDim, kFfnDim, kDim);
  }

  const Bits x_before_down = ex.download();
  pro.poison(s);
  restore(pro.ffn, effn);
  restore(pro.x, ex);
  mi::kernels::fused::gemm_gated_residual(tx, tf, bt.down, vec(mod(5), kDim), s);
  run_block_op(7, w, buffers, s);
  {
    const auto [ref_h, got_h] = op_out(7, Gate::kFlipChecked, ex, pro.x);
    dump_bits("x7", ref_h);
    const Bits ffn_h = effn.download(), down_h = host(w.down, std::size_t{kDim} * kFfnDim), gate_h = host(mod(5), kDim);
    report_gated_outliers(ref_h, got_h, x_before_down, kDim, ffn_h, kFfnDim, down_h, kFfnDim, gate_h.data(), kDim, kFfnDim);
    report_gated_worst(ref_h, got_h, x_before_down, kDim, ffn_h, kFfnDim, down_h, kFfnDim, gate_h.data(), kDim, kFfnDim);
    check_gated_flips(ref_h, got_h, x_before_down, kDim, ffn_h, kFfnDim, down_h, kFfnDim, gate_h.data(), kDim, kFfnDim);
  }

  const Bits final_chain = ex.download();
  std::size_t chain_diff = 0;
  for (std::size_t i = 0; i < final_chain.size(); ++i) chain_diff += final_chain[i] != final_engine[i];
  std::printf("engine per-op chain vs run_block_fused: %zu of %zu values differ\n", chain_diff, final_chain.size());
  if (chain_diff != 0) ++g_failures;

  const auto engine_op = [&](int op) {
    switch (op) {
      case 0: mi::kernels::rms_norm_modulate_moss(tx, bt.norm1, vec(mod(1), kDim), vec(mod(0), kDim), ty, s); break;
      case 1: engine.linear(tq, ty, bt.qkv, nullptr, ws, mi::kernels::Gemm::kWorkspaceBytes, s); break;
      case 2: mi::kernels::fused::qk_norm_rope_moss(tq, kHeads, bt.q_norm, bt.k_norm, mat(c.rope, l, kHeadDim), s); break;
      case 3:
        mi::attention::flash_fwd(heads_of(eqkv.get(), l, kHeads, 3 * kDim), heads_of(eqkv.get() + kDim, l, kHeads, 3 * kDim),
                                 heads_of(eqkv.get() + 2 * kDim, l, kHeads, 3 * kDim), heads_of(eattn.get(), l, kHeads, kDim), cu_tensor, cu_tensor, max_seqlen, false,
                                 kScale, lse.get(), lse_bytes, s);
        break;
      case 4: mi::kernels::fused::gemm_gated_residual(tx, ta, bt.o, vec(mod(2), kDim), s); break;
      case 5: mi::kernels::rms_norm_modulate_moss(tx, bt.norm2, vec(mod(4), kDim), vec(mod(3), kDim), ty, s); break;
      case 6: mi::kernels::fused::gemm_situ_glu(tf, ty, bt.up_gate, s); break;
      default: mi::kernels::fused::gemm_gated_residual(tx, tf, bt.down, vec(mod(5), kDim), s); break;
    }
  };

  // How far the engine's own later ops carry a low-bit difference. The control: its block with the QKV result replaced by
  // the exact sum, no prototype code involved. Then each op's share: its block with one op replaced by the prototype's.
  // A broken control would open the block gate's limits, so it has gates of its own: its QKV must pass as a QKV against the
  // engine's, and its final x must stay near the engine's. These catch gross failures only: a milder one still opens the limits. The floor is far below every healthy control (69.9 % and up) and
  // far above a control whose QKV is zero (under 5 %).
  std::optional<Stats> control;
  if (c.real) {
    constexpr double kControlSameFloor = 50.0;
    try {
      copy(ex.get(), c.x_init, ex.size());
      engine_op(0);
      engine_op(1);
      sync(s);
      const Bits engine_qkv = eqkv.download();
      const Bits exact_qkv = exact_gemm(ey.download(), kDim, host(w.qkv, std::size_t{3} * kDim * kDim), kDim, n, 3 * kDim, kDim);
      report("control's QKV against the engine's", engine_qkv, exact_qkv, Gate::kAccumulated);
      eqkv.upload(exact_qkv);
      check(hipStreamSynchronize(nullptr), "hipStreamSynchronize");
      for (int op = 2; op < kBlockOps; ++op) engine_op(op);
      sync(s);
      control = report("control: engine + exact QKV", final_engine, ex.download(), Gate::kReportOnly);
      if (control->same_pct < kControlSameFloor) {
        std::printf("| FAILED: the control's same %.4f %% is under its floor of %.0f %%\n", control->same_pct, kControlSameFloor);
        ++g_failures;
      }
    } catch (const std::exception& e) {
      std::printf("| FAILED: no control for this block: %s\n", e.what());
    }

    BlockBuffers on_engine = buffers;
    on_engine.x = ex.get();
    on_engine.y = ey.get();
    on_engine.qkv = eqkv.get();
    on_engine.attn = eattn.get();
    on_engine.ffn = effn.get();
    for (const int sub : {1, 3, 4, 6, 7}) {
      copy(ex.get(), c.x_init, ex.size());
      for (int op = 0; op < kBlockOps; ++op) {
        if (op == sub) {
          run_block_op(op, w, on_engine, s);
        } else {
          engine_op(op);
        }
      }
      sync(s);
      report(std::string("engine + prototype's ") + block_op_name(sub), final_engine, ex.download(), Gate::kReportOnly);
    }
  }

  // The whole prototype block on its own outputs.
  pro.poison(s);
  copy(pro.x.get(), c.x_init, pro.x.size());
  run_block(w, buffers, s);
  sync(s);
  BlockLimits limits;
  if (control.has_value()) {
    limits.same_min = control->same_pct - 5.0;
    limits.p99_max = std::max(16, 2 * control->p99);
    limits.scaled_max = std::max(16.0, 2.0 * control->scaled);
    if (dumping) {
      // hip_block_replay's read_meta has the format.
      std::ofstream meta(dump_path / "meta.txt");
      meta << std::setprecision(10) << kDumpFormat << "\nrows " << c.lengths.size();
      for (const int len : c.lengths) meta << ' ' << len;
      meta << "\nblock " << limits.same_min << ' ' << limits.p99_max << ' ' << limits.scaled_max << '\n';
      for (int op = 0; op < kBlockOps; ++op) {
        const Agreement& a = scalar_figures[static_cast<std::size_t>(op)];
        meta << "op " << op << ' ' << a.same << ' ' << a.p99 << ' ' << a.p999 << ' ' << a.scaled << ' ' << a.min_row_same << ' ' << a.above4 << '\n';
      }
      if (!meta) throw std::runtime_error("cannot write " + (dump_path / "meta.txt").string());
      g_dumped.push_back(dump_path.filename().string());
    }
  }
  report("whole block, final x", final_engine, pro.x.download(), Gate::kBlock, limits);
  if (control.has_value()) {
    std::printf("|   control: same %.4f, p99 %d, p99.9 %d, scaled %.2f; limits: same >= %.4f, p99 <= %d, scaled <= %.2f\n", control->same_pct, control->p99, control->p999,
                control->scaled, limits.same_min, limits.p99_max, limits.scaled_max);
  } else if (c.real) {
    ++g_failures;  // no fallback to the synthetic gate
  }
}

md::BlockTensors tensors_of(const BlockInputs& in) {
  md::BlockTensors bt;
  bt.norm1 = vec(in.norm1.get(), kDim);
  bt.norm2 = vec(in.norm2.get(), kDim);
  bt.q_norm = vec(in.q_norm.get(), kHeadDim);
  bt.k_norm = vec(in.k_norm.get(), kHeadDim);
  bt.qkv = mat(in.qkv.get(), 3 * kDim, kDim);
  bt.o = mat(in.o.get(), kDim, kDim);
  bt.up_gate = mat(in.up_gate.get(), 2 * kFfnDim, kDim);
  bt.down = mat(in.down.get(), kDim, kFfnDim);
  return bt;
}

void synthetic_block(const std::string& name, const std::vector<int>& lengths, mi::kernels::Gemm& engine, void* ws, const md::FusedRouteSnapshot& snapshot, hipStream_t s) {
  const BlockInputs in(cumulative(lengths).back());
  check_block({name, in.weights(), tensors_of(in), in.x_init.get(), in.rope.get(), in.mod.get(), lengths}, engine, ws, snapshot, s);
}

// Real weights and activations. The engine's forward (blocks streamed, so the DiT need not fit in device memory) runs one
// step on the golden latents and timestep, as tests/unit/test_dit_block_fused.cpp, and its residual stream is kept around
// the blocks below. Each of those blocks then runs here with its real weights on that input.
struct RealCase {
  std::string name;
  std::int64_t h = 0, w = 0;
  std::vector<std::int64_t> lengths;
  std::map<int, Bits> x_after;
};

std::vector<bf16_bits> read_mod_and_capture(mi::weights::ModelIndex& index, const fs::path& golden, std::vector<RealCase>& cases, md::SkipSet& uncond_skip) {
  constexpr const char* kCase = "g_fridge_1024";
  constexpr std::int64_t kBenchCond = 282;
  const auto noise = mi::npy::load(golden / kCase / "initial_noise.npy");
  const std::int64_t h = noise.shape[2], w = noise.shape[3];
  const float timestep = static_cast<float>(mi::npy::load(golden / kCase / "timestep0.npy").to_vector<double>().at(0));
  const auto te_c = mi::npy::load(golden / (std::string(kCase) + ".cond.te_last_hidden.npy"));
  const auto te_u = mi::npy::load(golden / (std::string(kCase) + ".uncond.te_last_hidden.npy"));
  const auto te_long = mi::npy::load(golden / "g_long4096_1024.cond.te_last_hidden.npy");
  {
    std::ifstream in(golden / kCase / "dit_step0_uncond.skip.json");
    std::stringstream ss;
    ss << in.rdbuf();
    const mi::json::Value doc = mi::json::parse(ss.str());
    for (const auto& v : doc.as_array()) uncond_skip.set(static_cast<std::size_t>(v.as_int()));
  }
  cases.push_back({"golden S=" + std::to_string(te_c.shape[0]) + "+" + std::to_string(te_u.shape[0]), h, w, {te_c.shape[0], te_u.shape[0]}, {}});
  cases.push_back({"E13 S=282+" + std::to_string(te_u.shape[0]), h, w, {kBenchCond, te_u.shape[0]}, {}});

  md::Residency streamed;
  streamed.resident_blocks = 0;
  const std::int64_t s_max = std::max<std::int64_t>({te_c.shape[0], te_u.shape[0], kBenchCond});
  const std::size_t io = mi::runtime::align_up(static_cast<std::size_t>(128 * h * w * 4)) * 3 + 4 * mi::runtime::align_up(static_cast<std::size_t>(s_max * (md::kTextDim + kDim) * 2)) +
                         mi::runtime::align_up(6 * kDim * 2) + mi::runtime::align_up(2 * kDim * 2);
  const std::size_t bytes = md::MossDit::persistent_bytes(index, streamed) + io + md::MossDit::stream_workspace_bytes(index, streamed) +
                            std::max({md::MossDit::workspace_bytes(h, w, cases[0].lengths), md::MossDit::workspace_bytes(h, w, cases[1].lengths), md::MossDit::encode_text_workspace_bytes(s_max)});
  mi::runtime::StreamSet streams;
  mi::runtime::DeviceArena arena(bytes);
  md::MossDit model(index, arena, streams, false, nullptr, streamed);
  const cudaStream_t stream = streams.compute();

  const auto refine = [&](const mi::npy::Array& te, std::int64_t rows) {
    const auto v = te.to_vector<float>();
    Bits hv(static_cast<std::size_t>(rows * md::kTextDim));
    for (std::size_t i = 0; i < hv.size(); ++i) hv[i] = to_bf16(v[i]);
    const Tensor in = arena.alloc_persistent_tensor(DType::kBF16, {rows, md::kTextDim});
    check(hipMemcpy(in.data, hv.data(), hv.size() * sizeof(bf16_bits), hipMemcpyHostToDevice), "hipMemcpy");
    MI_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const Tensor out = arena.alloc_persistent_tensor(DType::kBF16, {rows, kDim});
    model.encode_text(in, out, stream);
    return out;
  };
  const Tensor text_c = refine(te_c, te_c.shape[0]), text_u = refine(te_u, te_u.shape[0]), text_bench = refine(te_long, kBenchCond);
  const Tensor latents = arena.alloc_persistent_tensor(DType::kF32, {128, h, w});
  {
    const auto nv = noise.to_vector<float>();
    MI_CUDA_CHECK(cudaMemcpy(latents.data, nv.data(), nv.size() * 4, cudaMemcpyHostToDevice));
    MI_CUDA_CHECK(cudaStreamSynchronize(nullptr));
  }
  const Tensor outs[] = {arena.alloc_persistent_tensor(DType::kF32, {128, h, w}), arena.alloc_persistent_tensor(DType::kF32, {128, h, w})};
  const md::Row golden_rows[] = {{text_c, md::SkipSet{}}, {text_u, uncond_skip}};
  const md::Row bench_rows[] = {{text_bench, md::SkipSet{}}, {text_u, uncond_skip}};
  for (std::size_t c = 0; c < cases.size(); ++c) {
    model.forward(latents, timestep, std::span<const md::Row>(c == 0 ? golden_rows : bench_rows, 2), outs, stream, [&](int block, const Tensor& x) {
      static constexpr int kCaptured[] = {0, 1, 20, 21, 22, 23, 40, 41};
      if (std::find(std::begin(kCaptured), std::end(kCaptured), block) == std::end(kCaptured)) return;
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      Bits hx(static_cast<std::size_t>(x.numel()));
      MI_CUDA_CHECK(cudaMemcpy(hx.data(), x.data, hx.size() * sizeof(bf16_bits), cudaMemcpyDeviceToHost));
      cases[c].x_after[block] = std::move(hx);
    });
  }
  const Tensor mod = arena.alloc_persistent_tensor(DType::kBF16, {6 * kDim});
  const Tensor head = arena.alloc_persistent_tensor(DType::kBF16, {2 * kDim});
  model.time_modulation(timestep, mod, head, stream);
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  Bits mod_h(6 * kDim);
  MI_CUDA_CHECK(cudaMemcpy(mod_h.data(), mod.data, mod_h.size() * sizeof(bf16_bits), cudaMemcpyDeviceToHost));
  return mod_h;
}

int run_real(const fs::path& bundle, const fs::path& golden, hipStream_t s) {
  // A dump is complete once its manifest is written, at the end; until then hip_block_replay refuses it.
  if (!g_dump_dir.empty()) fs::remove(fs::path(g_dump_dir) / "manifest.txt");
  mi::weights::ModelIndex index = mi::weights::load_index(bundle / "moss.mossidx", bundle);
  md::install_fused_check(md::check_fused_gemms(s));
  const md::FusedRouteSnapshot snapshot = md::fused_route_snapshot();

  std::vector<RealCase> cases;
  md::SkipSet uncond_skip;
  const Bits mod_h = read_mod_and_capture(index, golden, cases, uncond_skip);
  const Buf mod = device_bits(mod_h);

  // The blocks' weights, as MossDit's streamed copies are laid out.
  const std::vector<int> blocks = {1, 21, 22, 23, 41};
  mi::runtime::StreamSet streams;
  mi::runtime::DeviceArena arena(blocks.size() * md::BlockWeights::block_device_bytes(index) + mi::runtime::align_up(mi::kernels::Gemm::kWorkspaceBytes));
  mi::weights::ReaderCache readers(index, false);
  mi::runtime::PinnedRing ring(streams.copy());
  std::map<int, md::BlockTensors> views;
  for (const int b : blocks) {
    const auto* group = index.find_group(mi::weights::Component::kTransformer, "block." + std::to_string(b));
    if (group == nullptr) throw std::runtime_error("block group missing from the index");
    views[b] = md::detail::block_views(b, mi::weights::upload_group(readers, md::detail::with_front(*group, md::detail::block_front(b, nullptr)), arena, ring), nullptr);
  }
  ring.synchronize();

  mi::kernels::Gemm engine;
  void* ws = arena.alloc_persistent(mi::kernels::Gemm::kWorkspaceBytes);

  for (const RealCase& c : cases) {
    std::vector<std::int64_t> offsets = {0};
    for (const std::int64_t l : c.lengths) offsets.push_back(offsets.back() + c.h * c.w + l);
    const std::int64_t total = offsets.back();
    Buf rope(static_cast<std::size_t>(total) * kHeadDim);
    for (std::size_t r = 0; r < c.lengths.size(); ++r) {
      mi::kernels::moss_rope_table(mat(rope.get() + offsets[r] * kHeadDim, offsets[r + 1] - offsets[r], kHeadDim), static_cast<int>(c.h), static_cast<int>(c.w),
                                   static_cast<int>(c.lengths[r]), s);
    }
    sync(s);
    const std::vector<int> run_blocks = c.name.rfind("golden", 0) == 0 ? blocks : std::vector<int>{1, 41};
    for (const int b : run_blocks) {
      // The rows that execute this block: all of them, except the uncond row where it skips. Rows are [cond, uncond].
      const bool skip_uncond = uncond_skip.test(static_cast<std::size_t>(b));
      const std::int64_t tokens = skip_uncond ? offsets[1] : total;
      std::vector<int> lengths = {static_cast<int>(offsets[1])};
      if (!skip_uncond) lengths.push_back(static_cast<int>(offsets[2] - offsets[1]));

      const Bits& in_all = c.x_after.at(b - 1);
      const Bits in(in_all.begin(), in_all.begin() + tokens * kDim);
      const Bits& out_all = c.x_after.at(b);
      const Bits expected(out_all.begin(), out_all.begin() + tokens * kDim);
      const Buf x = device_bits(in);
      const md::BlockTensors& bt = views.at(b);
      const auto ptr = [](const Tensor& t) { return static_cast<const bf16_bits*>(t.data); };
      const BlockWeights w{ptr(bt.norm1), ptr(bt.norm2), ptr(bt.q_norm), ptr(bt.k_norm), ptr(bt.qkv), ptr(bt.o), ptr(bt.up_gate), ptr(bt.down)};
      check_block({"real, " + c.name + ", block " + std::to_string(b), w, bt, x.get(), rope.get(), mod.get(), lengths, &expected, true}, engine, ws, snapshot, s);
    }
  }
  if (!g_dump_dir.empty()) {
    if (g_failures != 0 || g_dumped.empty()) {
      std::printf("\nno manifest.txt in %s: a failed run is no reference for hip_block_replay\n", g_dump_dir.c_str());
    } else {
      std::ofstream manifest(fs::path(g_dump_dir) / "manifest.txt");
      manifest << kDumpFormat << '\n';
      for (const std::string& d : g_dumped) manifest << d << '\n';
      if (!manifest) throw std::runtime_error("cannot write manifest.txt in " + g_dump_dir);
      std::printf("\ndumped %zu blocks to %s\n", g_dumped.size(), g_dump_dir.c_str());
    }
  }
  std::printf("\n%s (%d failed checks)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}

int run_synthetic(hipStream_t s) {
  const ConversionCheck c = check_conversions();
  std::printf("device_math.h against the CUDA intrinsics, over all inputs: bf %llu differ, %llu NaNs with other bits; "
              "f %llu differ, %llu NaNs with other bits\n",
              c.round_differ, c.round_nan_bits, c.widen_differ, c.widen_nan_bits);
  if (c.round_differ != 0 || c.widen_differ != 0) ++g_failures;

  mi::kernels::Gemm engine;
  DevBuffer<char> ws(mi::kernels::Gemm::kWorkspaceBytes);

  const md::FusedCheck fused = md::check_fused_gemms(s);
  std::printf("engine self-check: %s\n", fused.ok ? "ok" : fused.detail.c_str());
  md::install_fused_check(fused);
  const md::FusedRouteSnapshot snapshot = md::fused_route_snapshot();

  std::printf("\nPer-op, tails on every axis. ULP columns are bf16 ULPs; scaled floors them at the output's rms.\n");
  print_header();
  check_rms_norm(s);
  check_gemm(engine, ws.get(), s);
  check_qk_norm_rope(s);
  check_attention(s);
  check_fused_epilogues(s);

  std::printf("\nEdge inputs.\n");
  print_header();
  check_edges(engine, ws.get(), s);
  check_attention_op_edges(s);

  synthetic_block("Block, 1024^2 shape", {4378, 4106}, engine, ws.get(), snapshot, s);
  synthetic_block("Block, varlen", {4000, 2601, 17, 1000}, engine, ws.get(), snapshot, s);

  std::printf("\n%s (%d failed checks)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  bool real = false;
  const auto env = [](const char* name) {
    const char* v = std::getenv(name);
    return std::string(v == nullptr ? "" : v);
  };
  std::string weights = env("MOSS_WEIGHTS"), golden = env("MOSS_GOLDEN");
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--real") {
      real = true;
    } else if (arg == "--dump" && i + 1 < argc) {
      g_dump_dir = argv[++i];
    } else if (arg == "--weights" && i + 1 < argc) {
      weights = argv[++i];
    } else if (arg == "--golden" && i + 1 < argc) {
      golden = argv[++i];
    } else {
      std::fprintf(stderr, "usage: %s [--real [--weights DIR] [--golden DIR] [--dump DIR]]\n", argv[0]);
      return 1;
    }
  }
  if (!g_dump_dir.empty() && !real) {
    std::fprintf(stderr, "--dump needs --real: it writes the real blocks\n");
    return 1;
  }
  try {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
      std::printf("hip_block_verify needs a CUDA device: none found.\n");
      return 2;
    }
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
    std::printf("hip_block_verify, device: %s, leaf: %s\n", prop.name, leaf_name());
    hipStream_t s = nullptr;
    check(hipStreamCreate(&s), "hipStreamCreate");
    int status = 0;
    if (real) {
      if (weights.empty() || golden.empty()) {
        std::fprintf(stderr, "--real needs --weights (or MOSS_WEIGHTS) and --golden (or MOSS_GOLDEN)\n");
        return 1;
      }
      status = run_real(fs::path(weights) / "base", golden, s);
    } else {
      status = run_synthetic(s);
    }
    (void)hipStreamDestroy(s);
    return status;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hip_block_verify: %s\n", e.what());
    return 1;
  }
}
