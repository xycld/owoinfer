// hip_block_replay DIR [--iters N]: what hip_block_verify --real --dump DIR captured from the engine on an NVIDIA machine, run
// on this device. Each op gets the engine's input and is compared with the engine's output; then the whole block runs on its
// own outputs and is compared with the engine's final x under the limits the dump's control gave, and is timed. Before each
// op and the whole block, every buffer the ops write holds NaN, so a value the prototype did not write fails as a NaN on one
// side rather than passing as the engine's.
//
// This is the one comparison of WMMA's results with the engine's at full shapes on real activations.
//
// DIR/manifest.txt lists the blocks, each a directory with its meta.txt. A dump without a manifest (an interrupted one, or
// one of an earlier format), one that lists no block or misses a listed one, and a meta.txt of an earlier format are refused.
//
// Exit status: 0 passed, 1 failed or error, 2 no usable device.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "host_util.h"

namespace {

using namespace hip_block;
namespace fs = std::filesystem;
using Bits = std::vector<bf16_bits>;

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

Bits read_bits(const fs::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  const auto bytes = static_cast<std::size_t>(f.tellg());
  Bits v(bytes / sizeof(bf16_bits));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
  if (!f) throw std::runtime_error("short read " + p.string());
  return v;
}

DevBuffer<bf16_bits> upload(const Bits& v) {
  DevBuffer<bf16_bits> b(v.size());
  b.upload(v);
  return b;
}

int g_failures = 0;

void print_row(const char* name, const Agreement& r, double scaled_limit, const std::string& row_limit, const std::string& count_limit,
               const std::string& why) {
  std::printf("| %-24s | %9.4f | %5d | %6d | %8.2f | %8.2f | %9.4f | %9s | %8zu | %8s | %s\n", name, r.same, r.p99, r.p999, r.scaled, scaled_limit,
              r.min_row_same, row_limit.c_str(), r.above4, count_limit.c_str(), why.empty() ? "ok" : ("FAILED:" + why).c_str());
  if (!why.empty()) ++g_failures;
}

// Limits on one op. same >= 99 % and p99.9 <= 1, as verify's accumulated gate, except p99.9 <= 2 for the three GEMMs with an
// epilogue (verify allows 2 for SiTU on real activations only, and holds the gated residuals to 1 through its flip check,
// which needs the exact sums and is not ported here). The norms and qk_norm_rope get these limits, not verify's exact gate:
// a card's rsqrt may flip what the scalar leaf holds exact. Then two limits from the scalar leaf's own figures on the same
// inputs, which hip_block_verify --real --dump wrote into meta.txt:
//   scaled <= max(16, 4 x the scalar leaf's),
//   the lowest same % of a token row >= the scalar leaf's lowest minus kRowMargin points, and
//   the count of outputs above scaled 4 <= max(kCountFloor, kCountFactor x the scalar leaf's count).
// The scaled limit is a backstop: every wrong-row mutant of the third review that exceeds it also fails the row limit. It is
// wide because a gated residual near cancellation moves by many ULPs for one flipped bit, which only verify's flip check tells
// from an error: the card's block-41 down gemm + gated reached 64, twice the scalar leaf's 32.
// A whole token row gone wrong is 0.012 % of an op at E13's shape, too few values for same or p99.9 to see; the row limit
// sees it. kRowMargin is wide because the card cannot be recalibrated here. On NVIDIA, two other accumulation orders in the
// scalar leaf (each fragment's 16 products as a tree, or K descending) moved the lowest row by at most 0.36 points (K
// descending, block 41's attention, 95.28 to 95.64), and the card's per-op same was within 0.07 points of the scalar
// leaf's. The third review's wrong-row mutants reach at most 41.5 % in their row (40 norm rows 0.39 % large), 33.5 % for
// one token without RoPE, 0 % for a row x 1.02 or a zeroed QKV row.
constexpr double kRowMargin = 15.0;
// A wrong fragment or column (the shape a fragment layout bug takes) is too few values for same, p99.9 or a row, and moves
// each by a few ULPs, under the scaled limit: one attention fragment x 1.03 reaches scaled 8, one down column x 1.06 13. It
// does put outputs above scaled 4 where a correct op puts none or few. On the dump, the scalar leaf puts 0 there in every op
// but block 41's SiTU (1) and down gemm + gated (31); the two other accumulation orders put 0, 1 and 29 or 30 there; the
// card's earlier run has scaled <= 4, so 0, in every op but those two. The review's fragment and column mutants put 186 and
// 80 (attention fragment), 38 and 13 (QKV fragment), 11 and 8464 (down column) there, at blocks 1 and 41. The floor must sit
// under 11 and the factor at 4 or more; 8 and 8 leave 8x over the other orders' counts.
constexpr std::size_t kCountFloor = 8;
constexpr std::size_t kCountFactor = 8;

void check_op(const char* name, const Bits& ref, const Bits& got, std::size_t row_len, std::size_t cols, int p999_limit, const Agreement& scalar) {
  const Agreement r = agreement(ref, got, row_len, cols);
  const double scaled_limit = std::max(16.0, 4.0 * scalar.scaled), row_limit = scalar.min_row_same - kRowMargin;
  const std::size_t count_limit = std::max(kCountFloor, kCountFactor * scalar.above4);
  std::string why;
  if (r.nan_mismatch != 0) why += " NaN mismatch;";
  if (!std::isfinite(r.scaled)) why += " Inf on one side;";
  if (r.same < 99.0) why += " same < 99 %;";
  if (r.p999 > p999_limit) why += " p99.9 over limit;";
  if (r.scaled > scaled_limit) why += " scaled over limit;";
  if (r.min_row_same < row_limit) why += " a row's same under limit;";
  if (r.above4 > count_limit) why += " too many above scaled 4;";
  char limit[32];
  std::snprintf(limit, sizeof limit, "%.4f", row_limit);
  print_row(name, r, scaled_limit, limit, std::to_string(count_limit), why);
}

std::vector<std::string> read_manifest(const fs::path& dir) {
  std::ifstream f(dir / "manifest.txt");
  if (!f) {
    throw std::runtime_error("no manifest.txt in " + dir.string() + ": not a complete dump of " + kDumpFormat +
                             " (an interrupted one, or one of an earlier format: regenerate it with hip_block_verify --real --dump)");
  }
  std::string line;
  std::getline(f, line);
  if (line != kDumpFormat) throw std::runtime_error((dir / "manifest.txt").string() + " is not of " + kDumpFormat);
  std::vector<std::string> blocks;
  while (std::getline(f, line)) {
    if (!line.empty()) blocks.push_back(line);
  }
  if (blocks.empty()) throw std::runtime_error((dir / "manifest.txt").string() + " lists no block");
  return blocks;
}

// meta.txt: the header line, then
//   rows N L1 .. LN
//   block same_min p99_max scaled_max        the whole block's limits, from verify's control
//   op I same p99 p99.9 scaled min_row_same above4   for I = 0 .. 7: the scalar leaf against the engine, as agreement() gives them
struct BlockMeta {
  std::vector<int> lengths;
  double same_min = 0.0, scaled_max = 0.0;
  int p99_max = 0;
  std::array<Agreement, kBlockOps> scalar{};
};

BlockMeta read_meta(const fs::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("no " + p.string());
  std::string line;
  std::getline(f, line);
  if (line != kDumpFormat) {
    throw std::runtime_error(p.string() + " is not of " + kDumpFormat + " (format 1 has no per-op figures, format 2 no count above scaled 4): regenerate the dump with hip_block_verify --real --dump");
  }
  const auto bad = [&] { return std::runtime_error("bad " + p.string()); };
  const auto expect = [&](const char* word) {
    std::string w;
    if (!(f >> w) || w != word) throw bad();
  };
  BlockMeta m;
  int batch = 0;
  expect("rows");
  f >> batch;
  if (!f || batch < 1 || batch > BlockActivations::kMaxRows) throw bad();
  m.lengths.resize(static_cast<std::size_t>(batch));
  for (int& l : m.lengths) {
    if (!(f >> l) || l < 1) throw bad();
  }
  expect("block");
  f >> m.same_min >> m.p99_max >> m.scaled_max;
  for (int i = 0; i < kBlockOps; ++i) {
    int op = -1;
    expect("op");
    Agreement& a = m.scalar[static_cast<std::size_t>(i)];
    f >> op >> a.same >> a.p99 >> a.p999 >> a.scaled >> a.min_row_same >> a.above4;
    if (!f || op != i) throw bad();
  }
  return m;
}

int run(const fs::path& dir, int iters) {
  const std::vector<std::string> blocks = read_manifest(dir);
  hipStream_t s = nullptr;
  check(hipStreamCreate(&s), "hipStreamCreate");
  for (const std::string& name : blocks) {
    const fs::path d = dir / name;
    if (!fs::is_directory(d)) throw std::runtime_error("the manifest lists " + name + ", which is not in " + dir.string());
    const BlockMeta meta = read_meta(d / "meta.txt");
    std::vector<std::int32_t> cu_h = {0};
    int max_seqlen = 0;
    for (const int l : meta.lengths) {
      cu_h.push_back(cu_h.back() + l);
      max_seqlen = std::max(max_seqlen, l);
    }
    const int tokens = cu_h.back();
    const int batch = static_cast<int>(meta.lengths.size());

    const auto rd = [&](const char* file) { return read_bits(d / (std::string(file) + ".bin")); };
    const DevBuffer<bf16_bits> norm1 = upload(rd("norm1")), norm2 = upload(rd("norm2")), q_norm = upload(rd("q_norm")), k_norm = upload(rd("k_norm")),
                               qkv_w = upload(rd("w_qkv")), o_w = upload(rd("w_o")), up_w = upload(rd("w_up_gate")), down_w = upload(rd("w_down")), rope = upload(rd("rope")),
                               mod = upload(rd("mod"));
    const Bits x0 = rd("x0"), y0 = rd("y0"), qkv1 = rd("qkv1"), qkv2 = rd("qkv2"), attn = rd("attn"), x4 = rd("x4"), y5 = rd("y5"), ffn = rd("ffn"), x7 = rd("x7");
    DevBuffer<std::int32_t> cu(cu_h.size());
    cu.upload(cu_h);
    const BlockWeights plain{norm1.get(), norm2.get(), q_norm.get(), k_norm.get(), qkv_w.get(), o_w.get(), up_w.get(), down_w.get()};
    const PackedWeights packed(plain, s);
    const BlockWeights w = packed.apply(plain);
    BlockActivations act(tokens);
    const BlockBuffers b = act.buffers(rope.get(), mod.get(), cu.get(), batch, max_seqlen, tokens);

    std::printf("\n%s: %d tokens in %d rows, %s\n", name.c_str(), tokens, batch, leaf_name());
    std::printf("| %-24s | %9s | %5s | %6s | %8s | %8s | %9s | %9s | %8s | %8s |\n", "op, against the engine", "same %", "p99", "p99.9", "scaled", "<= this",
                "row same", ">= this", "above 4", "<= this");
    const auto op = [&](int i) {
      run_block_op(i, w, b, s);
      check(hipStreamSynchronize(s), "hipStreamSynchronize");
    };
    const auto& sc = meta.scalar;
    constexpr std::size_t kQkv = 3 * kDim;
    act.poison(s);
    act.x.upload(x0);
    op(0);
    check_op("norm1+modulate", y0, act.y.download(), kDim, kDim, 1, sc[0]);
    act.poison(s);
    act.y.upload(y0);
    op(1);
    check_op("qkv gemm", qkv1, act.qkv.download(), kQkv, kQkv, 1, sc[1]);
    act.poison(s);
    act.qkv.upload(qkv1);
    op(2);
    check_op("qk_norm_rope, q and k", qkv2, act.qkv.download(), kQkv, 2 * kDim, 1, sc[2]);
    act.poison(s);
    act.qkv.upload(qkv2);
    op(3);
    check_op("attention", attn, act.attn.download(), kDim, kDim, 1, sc[3]);
    act.poison(s);
    act.attn.upload(attn);
    act.x.upload(x0);
    op(4);
    check_op("o gemm+gated", x4, act.x.download(), kDim, kDim, 2, sc[4]);
    act.poison(s);
    act.x.upload(x4);
    op(5);
    check_op("norm2+modulate", y5, act.y.download(), kDim, kDim, 1, sc[5]);
    act.poison(s);
    act.y.upload(y5);
    op(6);
    check_op("up_gate gemm+situ", ffn, act.ffn.download(), kFfnDim, kFfnDim, 2, sc[6]);
    act.poison(s);
    act.ffn.upload(ffn);
    act.x.upload(x4);
    op(7);
    check_op("down gemm+gated", x7, act.x.download(), kDim, kDim, 2, sc[7]);

    act.poison(s);
    act.x.upload(x0);
    run_block(w, b, s);
    check(hipStreamSynchronize(s), "hipStreamSynchronize");
    const Agreement whole = agreement(x7, act.x.download(), kDim, kDim);
    std::string why;
    if (whole.nan_mismatch != 0) why += " NaN mismatch;";
    if (whole.same < meta.same_min) why += " same under limit;";
    if (whole.p99 > meta.p99_max) why += " p99 over limit;";
    if (whole.scaled > meta.scaled_max) why += " scaled over limit;";
    print_row("whole block, final x", whole, meta.scaled_max, "-", "-", why);
    std::printf("|   limits from the engine's own sensitivity: same >= %.2f, p99 <= %d, scaled <= %.2f\n", meta.same_min, meta.p99_max, meta.scaled_max);

    std::vector<double> ms;
    for (int it = 0; it < iters; ++it) {
      act.x.upload(x0);
      check(hipStreamSynchronize(s), "hipStreamSynchronize");
      const double t0 = now_ms();
      run_block(w, b, s);
      check(hipStreamSynchronize(s), "hipStreamSynchronize");
      ms.push_back(now_ms() - t0);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("|   whole block on these weights: min %.1f ms, median %.1f ms over %d runs\n", ms.front(), ms[ms.size() / 2], iters);
  }
  (void)hipStreamDestroy(s);
  std::printf("\n%s (%d failed checks)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: %s DIR [--iters N]\n", argv[0]);
      return 1;
    }
    int iters = 5;
    for (int i = 2; i < argc; ++i) {
      if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
        iters = std::atoi(argv[++i]);
      } else {
        std::fprintf(stderr, "usage: %s DIR [--iters N]\n", argv[0]);
        return 1;
      }
    }
    if (iters < 1) {
      std::fprintf(stderr, "--iters must be positive\n");
      return 1;
    }
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
      std::fprintf(stderr, "No usable HIP device: nothing to replay on.\n");
      return 2;
    }
    return run(argv[1], iters);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
