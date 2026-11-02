// hip_block_bench: per-op and per-block times of the prototype on synthetic 1024^2-shaped activations (by default the
// 8484 tokens of E13 in docs/experiment-log.md: two rows, 4378 and 4106 tokens). Before timing, the known-answer check
// below compares every kernel with a host replay of its op order, so a first run on a card says whether fragment loads
// and stores do what the kernels assume: the NVIDIA verify build cannot check that for WMMA. It then runs one block and
// fails on a NaN in the output.
//
// Limits, from the baseline of the scalar leaf on an RTX 4090 (in the README). `same` is the share of outputs whose bits
// equal the replay's, >= 99 %; a dropped rounding point moves far more of them. The ULP limit is on the worst distance in
// bf16 ULPs of max(|replay|, floor). The floor is the output's rms wherever the op's last add can cancel or a rounded weight
// meets V, since one flipped low bit from a card's math library (rsqrt, tanhf, exp2f) then moves a near-zero output by many
// of its own ULPs; plain GEMM outputs keep a floor of rms / 64. rms_norm_modulate and qk_norm_rope have a limit of 3 ULPs:
// a device rsqrt perturbed by 1 fp32 ULP moved qk_norm_rope by 2.00 in a model on the scalar leaf, as one flipped rbf(x * r)
// goes through the later rounding points. The other kernels have 1 ULP (SiTU, steeper: 1.5), at least twice their baseline
// of 0.5. None of this comes from an AMD measurement.
//
// Exit status: 0 ran, 1 failed (error, known-answer mismatch, NaN in a block), 2 no usable device.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "host_util.h"

#ifndef HIP_BLOCK_TARGET_NAME
#define HIP_BLOCK_TARGET_NAME "unknown"
#endif

namespace {

using namespace hip_block;

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Distance between got and ref in bf16 ULPs of max(|ref|, floor_mag): near-zero outputs of cancelling sums are not
// relatively accurate in any accumulation order.
double ulps_off(float got, double ref, double floor_mag) {
  int e = 0;
  (void)std::frexp(std::max(std::abs(ref), floor_mag), &e);
  return std::abs(static_cast<double>(got) - ref) / std::ldexp(1.0, e - 8);
}

double rms_of(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s / static_cast<double>(std::max<std::size_t>(v.size(), 1)));
}

constexpr double kMinSamePercent = 99.0;

struct Limit {
  double ulps;
  double floor_of_rms;  // the error floor as a fraction of the output's rms
};

bool report_check(const char* name, const std::vector<bf16_bits>& got, const std::vector<double>& ref, Limit limit) {
  const double floor_mag = rms_of(ref) * limit.floor_of_rms;
  double worst = 0.0;
  std::size_t nans = 0, same = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float g = from_bf16(got[i]);
    if (std::isnan(g)) ++nans;
    same += got[i] == to_bf16(static_cast<float>(ref[i]));
    worst = std::max(worst, ulps_off(g, ref[i], floor_mag));
  }
  const double same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(got.size());
  const bool ok = nans == 0 && worst <= limit.ulps && same_pct >= kMinSamePercent;
  std::printf("  %-24s %s (%zu values, %.3f %% same, limit %.0f %%; worst %.2f ULPs, limit %.1f; %zu NaN)\n", name,
              ok ? "ok" : "FAILED", got.size(), same_pct, kMinSamePercent, worst, limit.ulps, nans);
  return ok;
}

// fp64 x · wᵀ at (r, n).
double dot(const std::vector<bf16_bits>& x, int ldx, const std::vector<bf16_bits>& w, int ldw, int r, int n, int k) {
  double s = 0.0;
  for (int i = 0; i < k; ++i) s += static_cast<double>(from_bf16(x[r * ldx + i])) * from_bf16(w[n * ldw + i]);
  return s;
}

bool known_answer_norms(hipStream_t stream) {
  constexpr int rows = 37, d = kDim;
  const std::vector<bf16_bits> x_h = synth_bf16(11, static_cast<std::size_t>(rows) * d, 0), w_h = synth_bf16(12, d, 0),
                               scale_h = synth_bf16(13, d, -2), shift_h = synth_bf16(14, d, -2);
  DevBuffer<bf16_bits> x(x_h.size()), w(w_h.size()), scale(scale_h.size()), shift(shift_h.size()), out(x_h.size());
  x.upload(x_h);
  w.upload(w_h);
  scale.upload(scale_h);
  shift.upload(shift_h);
  const float factor = static_cast<float>(rows) / static_cast<float>(rows * d);
  std::vector<double> norm_ref(x_h.size());
  for (int r = 0; r < rows; ++r) {
    const bf16_bits* row = x_h.data() + static_cast<std::ptrdiff_t>(r) * d;
    const float rs = host_rsqrt_of_mean(row, d, factor);
    for (int c = 0; c < d; ++c) {
      norm_ref[static_cast<std::size_t>(r) * d + c] = from_bf16(host_norm_modulate(row[c], rs, w_h[c], scale_h[c], shift_h[c]));
    }
  }
  rms_norm_modulate(x.get(), d, w.get(), scale.get(), shift.get(), out.get(), d, rows, d, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  bool ok = report_check("rms_norm_modulate", out.download(), norm_ref, {3.0, 1.0});

  constexpr int tokens = 37;
  constexpr int ld = 3 * kDim;
  const std::vector<bf16_bits> qkv_h = synth_bf16(15, static_cast<std::size_t>(tokens) * ld, 0), q_w_h = synth_bf16(16, kHeadDim, 0),
                               k_w_h = synth_bf16(17, kHeadDim, 0), table_h = synth_rope(18, tokens);
  DevBuffer<bf16_bits> qkv(qkv_h.size()), q_w(q_w_h.size()), k_w(k_w_h.size()), table(table_h.size());
  qkv.upload(qkv_h);
  q_w.upload(q_w_h);
  k_w.upload(k_w_h);
  table.upload(table_h);
  std::vector<bf16_bits> want = qkv_h;
  const std::int64_t head_rows = std::int64_t{tokens} * kHeads;
  const float qk_factor = static_cast<float>(head_rows) / static_cast<float>(head_rows * kHeadDim);
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < 2 * kHeads; ++h) {
      host_qk_norm_rope_head(want.data() + static_cast<std::ptrdiff_t>(t) * ld + static_cast<std::ptrdiff_t>(h) * kHeadDim,
                             (h < kHeads ? q_w_h : k_w_h).data(), table_h.data() + static_cast<std::ptrdiff_t>(t) * kHeadDim, qk_factor);
    }
  }
  qk_norm_rope(qkv.get(), ld, tokens, kHeads, q_w.get(), k_w.get(), table.get(), kHeadDim, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  std::vector<double> rope_ref(want.size());
  for (std::size_t i = 0; i < want.size(); ++i) rope_ref[i] = from_bf16(want[i]);
  ok = report_check("qk_norm_rope", qkv.download(), rope_ref, {3.0, 1.0}) && ok;
  return ok;
}

// M, N, K with tails on every axis: the padded fragment paths and the tile walk.
bool known_answer_gemms(hipStream_t stream) {
  constexpr int m = 37, n = 104, k = 40, f = 24, ld = 3 * k;
  const std::vector<bf16_bits> x_h = synth_bf16(21, static_cast<std::size_t>(m) * ld, 0);
  const std::vector<bf16_bits> w_h = synth_bf16(22, static_cast<std::size_t>(n) * k, -2);
  const std::vector<bf16_bits> gate_h = synth_bf16(23, n, -2);
  const std::vector<bf16_bits> res_h = synth_bf16(24, static_cast<std::size_t>(m) * n, 0);
  DevBuffer<bf16_bits> x(x_h.size()), w(w_h.size()), gate(gate_h.size()), out(static_cast<std::size_t>(m) * n);
  x.upload(x_h);
  w.upload(w_h);
  gate.upload(gate_h);

  std::vector<double> plain(static_cast<std::size_t>(m) * n), gated(plain.size()), situ(static_cast<std::size_t>(m) * f);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      const double ab = dot(x_h, ld, w_h, k, r, c, k);
      plain[static_cast<std::size_t>(r) * n + c] = ab;
      const float rounded = host_rbf(static_cast<float>(ab));
      const float delta = host_rbf(rounded * from_bf16(gate_h[c]));
      gated[static_cast<std::size_t>(r) * n + c] = from_bf16(res_h[static_cast<std::size_t>(r) * n + c]) + delta;
    }
    for (int c = 0; c < f; ++c) {
      const float h = host_rbf(static_cast<float>(dot(x_h, ld, w_h, k, r, c, k)));
      const float g = host_rbf(static_cast<float>(dot(x_h, ld, w_h, k, r, f + c, k)));
      situ[static_cast<std::size_t>(r) * f + c] = host_situ(h, g);
    }
  }

  bool ok = true;
  gemm(out.get(), n, x.get(), ld, w.get(), k, m, n, k, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  ok = report_check("gemm", out.download(), plain, {1.0, 1.0 / 64.0}) && ok;

  out.upload(res_h);
  gemm_gated_residual(out.get(), n, x.get(), ld, w.get(), k, gate.get(), m, n, k, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  ok = report_check("gemm_gated_residual", out.download(), gated, {1.0, 1.0}) && ok;

  DevBuffer<bf16_bits> ffn(static_cast<std::size_t>(m) * f);
  gemm_situ_glu(ffn.get(), f, x.get(), ld, w.get(), k, m, f, k, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  ok = report_check("gemm_situ_glu", ffn.download(), situ, {1.5, 1.0}) && ok;
  return ok;
}

// Varlen, sequence lengths with tails on the query and the key side, against the fp32 replay of the reverse walk with
// the weights rounded to bf16.
bool known_answer_attention(hipStream_t stream) {
  constexpr int heads = 2, ld = heads * kHeadDim;
  const std::vector<int> lengths = {37, 64, 5, 33, 200};
  std::vector<std::int32_t> cu = {0};
  for (const int l : lengths) cu.push_back(cu.back() + l);
  const auto total = static_cast<std::size_t>(cu.back());
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));

  const std::vector<bf16_bits> q_h = synth_bf16(31, total * ld, 0), k_h = synth_bf16(32, total * ld, 0),
                               v_h = synth_bf16(33, total * ld, 0);
  DevBuffer<bf16_bits> q(total * ld), k(total * ld), v(total * ld), o(total * ld);
  DevBuffer<std::int32_t> cu_dev(cu.size());
  q.upload(q_h);
  k.upload(k_h);
  v.upload(v_h);
  cu_dev.upload(cu);

  std::vector<double> ref(total * ld);
  for (std::size_t b = 0; b < lengths.size(); ++b) {
    const std::int64_t r0 = cu[b];
    for (int h = 0; h < heads; ++h) {
      for (std::int64_t r = r0; r < cu[b + 1]; ++r) {
        const auto row = replay_attention_row<float>(q_h.data() + r * ld + h * kHeadDim, k_h.data() + r0 * ld + h * kHeadDim, ld,
                                                     v_h.data() + r0 * ld + h * kHeadDim, ld, lengths[b], scale);
        std::copy(row.begin(), row.end(), ref.begin() + r * ld + h * kHeadDim);
      }
    }
  }
  attention({q.get(), k.get(), v.get(), o.get(), ld, ld, ld, ld, kHeadDim, kHeadDim, kHeadDim, kHeadDim, cu_dev.get(), cu_dev.get(),
             static_cast<int>(lengths.size()), 200, heads, scale},
            stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  // A weight whose bf16 rounding flips moves an output by about one ULP at the output's rms, whatever that output's own
  // size: the floor here is the rms itself.
  return report_check("attention", o.download(), ref, {1.0, 1.0});
}

std::vector<int> parse_rows(const std::string& text) {
  std::vector<int> rows;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) rows.push_back(std::atoi(item.c_str()));
  return rows;
}

int run(const std::vector<int>& rows, int iters) {
  hipStream_t stream = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");

  std::printf("known-answer check against host replays:\n");
  const bool norms_ok = known_answer_norms(stream);
  const bool gemms_ok = known_answer_gemms(stream);
  const bool attn_ok = known_answer_attention(stream);
  if (!norms_ok || !gemms_ok || !attn_ok) {
    std::printf("known-answer check FAILED: the times below would not describe a correct block; not timing.\n");
    (void)hipStreamDestroy(stream);
    return 1;
  }

  std::vector<std::int32_t> cu_h = {0};
  int max_seqlen = 0;
  for (const int l : rows) {
    cu_h.push_back(cu_h.back() + l);
    max_seqlen = std::max(max_seqlen, l);
  }
  const int tokens = cu_h.back();
  const BlockInputs in(tokens);
  const BlockActivations act(tokens);
  DevBuffer<std::int32_t> cu(cu_h.size());
  cu.upload(cu_h);
  const BlockWeights w = in.weights();
  const BlockBuffers b = act.buffers(in.rope.get(), in.mod.get(), cu.get(), static_cast<int>(rows.size()), max_seqlen, tokens);
  const auto x_bytes = static_cast<std::size_t>(tokens) * kDim * sizeof(bf16_bits);
  const auto reset_x = [&] {
    check(hipMemcpyAsync(act.x.get(), in.x_init.get(), x_bytes, hipMemcpyDeviceToDevice, stream), "hipMemcpyAsync");
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  };

  reset_x();
  run_block(w, b, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const std::vector<bf16_bits> out = act.x.download();
  const auto nans = static_cast<std::size_t>(
      std::count_if(out.begin(), out.end(), [](bf16_bits v) { return (v & 0x7fffu) > 0x7f80u; }));
  std::printf("  %-24s %s (%zu of %zu output values are NaN)\n", "whole block", nans == 0 ? "ok" : "FAILED", nans, out.size());
  if (nans != 0) {
    std::printf("NaN in the block's output: not timing.\n");
    (void)hipStreamDestroy(stream);
    return 1;
  }

  std::vector<std::vector<double>> op_ms(kBlockOps);
  std::vector<double> block_ms;
  for (int it = 0; it < iters; ++it) {
    reset_x();
    for (int op = 0; op < kBlockOps; ++op) {
      const double t0 = now_ms();
      run_block_op(op, w, b, stream);
      check(hipStreamSynchronize(stream), "hipStreamSynchronize");
      op_ms[static_cast<std::size_t>(op)].push_back(now_ms() - t0);
    }
    reset_x();
    const double t0 = now_ms();
    run_block(w, b, stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    block_ms.push_back(now_ms() - t0);
  }

  const auto row = [](const char* name, std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    std::printf("| %-20s | %10.3f | %10.3f |\n", name, ms.front(), ms[ms.size() / 2]);
  };
  std::printf("\n%d tokens in %zu rows, %d iterations\n", tokens, rows.size(), iters);
  std::printf("| %-20s | %10s | %10s |\n", "op", "min ms", "median ms");
  std::printf("|----------------------|------------|------------|\n");
  for (int op = 0; op < kBlockOps; ++op) row(block_op_name(op), op_ms[static_cast<std::size_t>(op)]);
  row("whole block", block_ms);
  if (std::strstr(leaf_name(), "scalar") != nullptr) {
    std::printf("Scalar leaf: these times say nothing about WMMA or any AMD device.\n");
  }
  (void)hipStreamDestroy(stream);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> rows = {4378, 4106};
  int iters = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rows" && i + 1 < argc) {
      rows = parse_rows(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "usage: %s [--rows L1,L2,...] [--iters N]\n", argv[0]);
      return 1;
    }
  }
  if (rows.empty() || std::any_of(rows.begin(), rows.end(), [](int l) { return l < 1; }) || iters < 1) {
    std::fprintf(stderr, "--rows and --iters must be positive\n");
    return 1;
  }

  std::printf("hip_block_bench, build target %s, leaf: %s\n", HIP_BLOCK_TARGET_NAME, leaf_name());
  int devices = 0;
  const hipError_t count = hipGetDeviceCount(&devices);
  if (count != hipSuccess || devices < 1) {
    std::printf("No usable HIP device (%s): nothing to measure.\n",
                count == hipSuccess ? "device count is 0" : hipGetErrorString(count));
    return 2;
  }
  try {
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
    std::printf("device: %s\n", prop.name);
    return run(rows, iters);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hip_block_bench: %s\n", e.what());
    return 1;
  }
}
