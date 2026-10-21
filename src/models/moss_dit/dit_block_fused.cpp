#include "models/moss_dit/dit_block_fused.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/norm.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/fused/qk_norm_rope.h"

namespace mi::moss_dit {
namespace {

constexpr std::int64_t kDimB = 3072, kHeadsB = 24, kHeadDimB = 128, kFfnDimB = 8192;
static_assert(kFusedBigCols >= 4 * kDimB && kFusedBigCols >= kFfnDimB && kOpByOpBigCols >= 3 * kFfnDimB);
// SDPA's default scale 1/sqrt(head_dim), computed in double and rounded once as torch does (as moss_dit.cpp).
const float kScale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDimB)));

// Rows [t0, t1) of a contiguous [rows, cols] bf16 matrix starting `offset` elements into `buf`.
Tensor rows_of(const Tensor& buf, std::int64_t offset, std::int64_t rows, std::int64_t cols, std::int64_t t0,
               std::int64_t t1) {
  return Tensor::view(static_cast<std::uint16_t*>(buf.data) + offset, DType::kBF16, Device::kCUDA, {rows, cols})
      .slice0(t0, t1);
}

Tensor heads(const Tensor& buf, std::int64_t col) {
  Tensor t = Tensor::view(static_cast<std::uint16_t*>(buf.data) + col, DType::kBF16, Device::kCUDA,
                          {buf.dim(0), kHeadsB, kHeadDimB});
  t.stride[0] = buf.stride[0];
  return t;
}

Tensor vec(const Tensor& mod, std::int64_t i) {
  return Tensor::view(static_cast<std::uint16_t*>(mod.data) + i * kDimB, DType::kBF16, Device::kCUDA, {kDimB});
}

bool aligned256(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 256 == 0; }

// Contiguous [rows, cols] bf16 view at a 256-byte aligned address that is never dereferenced: Gemm::algo_config reads
// shapes, strides and alignment only.
Tensor probe_view(std::int64_t rows, std::int64_t cols) {
  return Tensor::view(reinterpret_cast<void*>(std::uintptr_t{1} << 20), DType::kBF16, Device::kCUDA, {rows, cols});
}

// The block's fused pairs, by their op-by-op linear out[rows, n] = x[rows, k] · w[n, k]ᵀ (MossDit::run_block).
constexpr int kOps = 3;
constexpr const char* kOpNames[kOps] = {"o_proj", "ffn_up_gate", "ffn_down"};
constexpr std::int64_t kOpK[kOps] = {kDimB, kDimB, kFfnDimB};
constexpr std::int64_t kOpN[kOps] = {kDimB, 2 * kFfnDimB, kDimB};
using OpConfigs = std::array<kernels::Gemm::AlgoConfig, kOps>;

OpConfigs query_configs(kernels::Gemm& gemm, std::int64_t rows) {
  OpConfigs c;
  for (int op = 0; op < kOps; ++op) {
    c[op] = gemm.algo_config(probe_view(rows, kOpN[op]), probe_view(rows, kOpK[op]), probe_view(kOpN[op], kOpK[op]),
                             nullptr, kernels::Gemm::kWorkspaceBytes);
    c[op].workspace_bytes = 0;  // how much of the workspace it uses: not part of the accumulation order
  }
  return c;
}

// Per device: the cuBLASLt handle fused_route() asks, its answers per row count, and the installed configurations.
// Leaked on purpose: cuBLASLt handles must not be destroyed during static destruction, after the CUDA runtime.
struct RouteState {
  std::mutex mu;
  std::map<int, std::unique_ptr<kernels::Gemm>> gemms;
  std::map<std::pair<int, std::int64_t>, OpConfigs> configs;
  std::map<int, std::vector<FusedConfig>> validated;
};
RouteState& route_state() {
  static auto* state = new RouteState();
  return *state;
}

bool validated_for(std::span<const FusedConfig> validated, int op, const kernels::Gemm::AlgoConfig& c) {
  return std::any_of(validated.begin(), validated.end(),
                     [&](const FusedConfig& v) { return v.op == kOpNames[op] && v.config == c; });
}

// fused_route() on `device` with `validated`; st.mu held.
FusedRoute route_locked(RouteState& st, int device, std::int64_t rows, std::span<const FusedConfig> validated) {
  if (validated.empty()) return {false, false, false};
  auto it = st.configs.find({device, rows});
  if (it == st.configs.end()) {
    std::unique_ptr<kernels::Gemm>& gemm = st.gemms[device];
    if (!gemm) gemm = std::make_unique<kernels::Gemm>();
    it = st.configs.emplace(std::make_pair(device, rows), query_configs(*gemm, rows)).first;
  }
  bool fused[kOps];
  for (int op = 0; op < kOps; ++op) {
    fused[op] = kernels::fused::matches_cublaslt(it->second[op]) && validated_for(validated, op, it->second[op]);
  }
  return {fused[0], fused[1], fused[2]};
}

}  // namespace

FusedRoute fused_route(std::int64_t rows) {
  if (rows <= 0) return {};
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  RouteState& st = route_state();
  std::lock_guard<std::mutex> lock(st.mu);
  const auto v = st.validated.find(device);
  return route_locked(st, device, rows, v == st.validated.end() ? std::span<const FusedConfig>() : v->second);
}

FusedRoute fused_route(std::int64_t rows, std::span<const FusedConfig> validated) {
  if (rows <= 0) return {};
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  RouteState& st = route_state();
  std::lock_guard<std::mutex> lock(st.mu);
  return route_locked(st, device, rows, validated);
}

void install_fused_check(const FusedCheck& check) {
  if (!check.ok) return;
  int device = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  RouteState& st = route_state();
  std::lock_guard<std::mutex> lock(st.mu);
  std::vector<FusedConfig>& installed = st.validated[device];
  for (const FusedConfig& c : check.validated) {
    const bool known = std::any_of(installed.begin(), installed.end(),
                                   [&](const FusedConfig& v) { return v.op == c.op && v.config == c.config; });
    if (!known) installed.push_back(c);
  }
}

std::int64_t fused_big_cols(std::int64_t hw, std::span<const std::int64_t> text_lengths) {
  // Every subset of the rows: runs are contiguous rows of a forward's row set, and callers size the workspace for the
  // largest row set (a step without SLG runs a subset of it).
  const std::size_t n = text_lengths.size();
  if (n > 16) throw std::invalid_argument("mi::moss_dit::fused_big_cols: too many rows");
  std::set<std::int64_t> sums;
  for (std::uint32_t mask = 1; mask < (std::uint32_t{1} << n); ++mask) {
    std::int64_t m = 0;
    for (std::size_t r = 0; r < n; ++r) {
      if (mask & (std::uint32_t{1} << r)) m += hw + text_lengths[r];
    }
    sums.insert(m);
  }
  for (const std::int64_t m : sums) {
    if (!fused_route(m).up_gate) return kOpByOpBigCols;
  }
  return kFusedBigCols;
}

void run_block_fused(const BlockTensors& w, const FusedBlockBuffers& a, const BlockRun& run, const QkvGemm& qkv_gemm,
                     cudaStream_t stream) {
  const std::int64_t l = a.x.dim(0);
  const FusedRoute route = fused_route(run.t1 - run.t0);
  const std::int64_t big_cols = route.up_gate ? kFusedBigCols : kOpByOpBigCols;
  if (qkv_gemm.gemm == nullptr || !a.big.is_contiguous() || a.big.numel() < l * big_cols || a.mod.numel() != 6 * kDimB) {
    throw std::invalid_argument("mi::moss_dit::run_block_fused: buffers do not match x " + a.x.describe() + ": big " +
                                a.big.describe() + " (" + std::to_string(big_cols) + " per token), mod " + a.mod.describe());
  }
  for (const void* p : {a.x.data, a.y.data, a.big.data, w.o.data, w.up_gate.data, w.down.data}) {
    if (!aligned256(p)) throw std::invalid_argument("mi::moss_dit::run_block_fused: operands must be 256-byte aligned");
  }
  const Tensor x = a.x.slice0(run.t0, run.t1), y = a.y.slice0(run.t0, run.t1);
  const Tensor qkv = rows_of(a.big, 0, l, 3 * kDimB, run.t0, run.t1);
  const Tensor attn = rows_of(a.big, l * 3 * kDimB, l, kDimB, run.t0, run.t1);
  const auto linear = [&](const Tensor& out, const Tensor& in, const Tensor& weight) {
    qkv_gemm.gemm->linear(out, in, weight, nullptr, qkv_gemm.workspace, qkv_gemm.workspace_bytes, stream);
  };

  kernels::rms_norm_modulate_moss(x, w.norm1, vec(a.mod, 1), vec(a.mod, 0), y, stream);
  linear(qkv, y, w.qkv);
  kernels::fused::qk_norm_rope_moss(qkv, static_cast<int>(kHeadsB), w.q_norm, w.k_norm, a.rope.slice0(run.t0, run.t1),
                                    stream);
  attention::flash_fwd(heads(qkv, 0), heads(qkv, kDimB), heads(qkv, 2 * kDimB), heads(attn, 0), run.cu_seqlens,
                       run.cu_seqlens, run.max_seqlen, false, kScale, a.lse, a.lse_bytes, stream);
  if (route.o) {
    kernels::fused::gemm_gated_residual(x, attn, w.o, vec(a.mod, 2), stream);
  } else {
    linear(y, attn, w.o);
    kernels::gated_residual(x, y, vec(a.mod, 2), x, stream);
  }
  kernels::rms_norm_modulate_moss(x, w.norm2, vec(a.mod, 4), vec(a.mod, 3), y, stream);
  Tensor ffn;
  if (route.up_gate) {
    ffn = rows_of(a.big, 0, l, kFfnDimB, run.t0, run.t1);
    kernels::fused::gemm_situ_glu(ffn, y, w.up_gate, stream);
  } else {
    // MossDit::run_block's [up_gate | ffn] layout.
    const Tensor up_gate = rows_of(a.big, 0, l, 2 * kFfnDimB, run.t0, run.t1);
    ffn = rows_of(a.big, l * 2 * kFfnDimB, l, kFfnDimB, run.t0, run.t1);
    linear(up_gate, y, w.up_gate);
    kernels::situ_glu(up_gate, ffn, stream);
  }
  if (route.down) {
    kernels::fused::gemm_gated_residual(x, ffn, w.down, vec(a.mod, 5), stream);
  } else {
    linear(y, ffn, w.down);
    kernels::gated_residual(x, y, vec(a.mod, 5), x, stream);
  }
}

namespace {

// splitmix64-hashed bf16 values +-m * 2^(scale_exp - 7 - e), m in [128, 255], e in 0..7 (as the GEMM tests).
std::vector<std::uint16_t> synth_bf16(std::size_t n, int scale_exp) {
  std::vector<std::uint16_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t z = i + 0x9E3779B97F4A7C15ull;
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

struct DevMatrix {
  DeviceBuffer buf;
  Tensor t;
  DevMatrix(std::int64_t rows, std::int64_t cols) {
    t = Tensor::view(nullptr, DType::kBF16, Device::kCUDA, {rows, cols});
    buf = DeviceBuffer(t.nbytes());
    t.data = buf.get();
  }
  // Filled from `pool` starting at element `offset` (wrapping), so operands differ without generating each one.
  void fill(const std::vector<std::uint16_t>& pool, std::size_t offset) {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(t.numel()));
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = pool[(offset + i * 7) % pool.size()];
    buf.upload(h.data(), h.size() * sizeof(std::uint16_t));
  }
  std::vector<std::uint16_t> download() const {
    std::vector<std::uint16_t> h(static_cast<std::size_t>(t.numel()));
    buf.download(h.data(), h.size() * sizeof(std::uint16_t));
    return h;
  }
};

// Run lengths the self-check asks cuBLASLt about: every count up to 1536 (small images, where its choice changes
// most), then every 16th up to 8192 and every 64th up to the longest run (cond + uncond + SLG rows of a 4.2 MP image,
// pipeline::kMaxPixels, each with a 4096-token prompt).
constexpr std::int64_t kMaxRunRows = 3 * (4'200'000 / 256 + 4096);
std::vector<std::int64_t> realistic_rows() {
  std::vector<std::int64_t> rows;
  for (std::int64_t m = 1; m <= 1536; ++m) rows.push_back(m);
  for (std::int64_t m = 1552; m <= 8192; m += 16) rows.push_back(m);
  for (std::int64_t m = 8256; m <= kMaxRunRows; m += 64) rows.push_back(m);
  return rows;
}

// Device bytes of one op's check operands at `rows` rows: x, w, gate, the op-by-op output, ref and got (ffn_up_gate:
// x, w, the [rows, 2F] op-by-op output, ref and got).
constexpr std::size_t kCheckBytes = std::size_t{224} << 20;  // + the 32 MiB GEMM workspace: <= 0.25 GiB
std::size_t check_bytes(int op, std::int64_t rows) {
  const auto m = static_cast<std::size_t>(rows), k = static_cast<std::size_t>(kOpK[op]), n = static_cast<std::size_t>(kOpN[op]);
  return 2 * (op == 1 ? m * k + n * k + m * n + m * n : m * k + n * k + n + 3 * m * n);
}

}  // namespace

FusedCheck check_fused_gemms(cudaStream_t stream, const FusedCheckHook& hook) {
  // The distinct accepted configurations of each op, at the smallest row count that gets each (cheapest to check). A
  // separate handle: its thousands of cached plans go with it.
  std::vector<FusedConfig> found;
  {
    kernels::Gemm probe;
    for (const std::int64_t m : realistic_rows()) {
      const OpConfigs c = query_configs(probe, m);
      for (int op = 0; op < kOps; ++op) {
        if (kernels::fused::matches_cublaslt(c[op]) && !validated_for(found, op, c[op])) found.push_back({kOpNames[op], c[op], m});
      }
    }
  }
  FusedCheck result;
  if (found.empty()) {
    result.detail = "cuBLASLt runs none of the block's linears with an algorithm the fused GEMMs reproduce at any realistic "
                    "row count: every pair runs op by op";
    return result;
  }
  kernels::Gemm gemm;
  DeviceBuffer ws(kernels::Gemm::kWorkspaceBytes);
  const std::vector<std::uint16_t> pool = synth_bf16(std::size_t{1} << 22, 0);
  std::vector<FusedConfig> checked;
  std::string mismatch;
  std::size_t unchecked = 0;
  const auto compare = [&](const FusedConfig& f, const std::vector<std::uint16_t>& want, const DevMatrix& got) {
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (hook) hook(f.op.c_str(), f.rows, got.t);
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::vector<std::uint16_t> g = got.download();
    std::size_t diff = 0;
    for (std::size_t i = 0; i < g.size(); ++i) diff += g[i] != want[i];
    if (diff == 0) {
      checked.push_back(f);
    } else if (mismatch.empty()) {
      mismatch = f.op + " at " + std::to_string(f.rows) + " rows: " + std::to_string(diff) + " of " +
                 std::to_string(g.size()) + " outputs differ from cuBLASLt + the unfused op (" + f.config.describe() + ")";
    }
  };
  // The op-by-op linear, after making sure cuBLASLt runs it with the configuration under test (the probe's layout is
  // the check's: contiguous rows from 256-byte aligned allocations).
  const auto linear = [&](const FusedConfig& f, const Tensor& out, const Tensor& in, const Tensor& weight) {
    kernels::Gemm::AlgoConfig c = gemm.algo_config(out, in, weight, nullptr, ws.size());
    c.workspace_bytes = 0;
    if (!(c == f.config)) {
      throw std::logic_error("mi::moss_dit::check_fused_gemms: " + f.op + " at " + std::to_string(f.rows) +
                             " rows runs " + c.describe() + ", the probe saw " + f.config.describe());
    }
    gemm.linear(out, in, weight, nullptr, ws.get(), ws.size(), stream);
  };
  for (int op = 0; op < kOps; ++op) {
    std::vector<const FusedConfig*> todo;
    std::int64_t max_rows = 0;
    for (const FusedConfig& f : found) {
      if (f.op != kOpNames[op]) continue;
      if (check_bytes(op, f.rows) > kCheckBytes) {
        ++unchecked;
        continue;
      }
      todo.push_back(&f);
      max_rows = std::max(max_rows, f.rows);
    }
    if (todo.empty()) continue;
    const std::int64_t k = kOpK[op], n = kOpN[op];
    if (op == 1) {  // ffn_up_gate + SiTU-GLU
      DevMatrix x(max_rows, k), w(n, k), up_gate(max_rows, n), ref(max_rows, n / 2), got(max_rows, n / 2);
      x.fill(pool, 5);
      w.fill(pool, 6);
      for (const FusedConfig* f : todo) {
        const std::int64_t m = f->rows;
        MI_CUDA_CHECK(cudaMemsetAsync(ref.t.data, 0, ref.t.nbytes(), stream));
        MI_CUDA_CHECK(cudaMemsetAsync(got.t.data, 0, got.t.nbytes(), stream));
        linear(*f, up_gate.t.slice0(0, m), x.t.slice0(0, m), w.t);
        kernels::situ_glu(up_gate.t.slice0(0, m), ref.t.slice0(0, m), stream);
        kernels::fused::gemm_situ_glu(got.t.slice0(0, m), x.t.slice0(0, m), w.t, stream);
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        compare(*f, ref.download(), got);
      }
      continue;
    }
    // Gated residual after the o-proj (K = kDim) and ffn_down (K = kFfnDim).
    DevMatrix x(max_rows, k), w(n, k), gate(1, n), y(max_rows, n), ref(max_rows, n), got(max_rows, n);
    x.fill(pool, 1);
    w.fill(pool, 2);
    gate.fill(pool, 3);
    const Tensor g = Tensor::view(gate.t.data, DType::kBF16, Device::kCUDA, {n});
    for (const FusedConfig* f : todo) {
      const std::int64_t m = f->rows;
      ref.fill(pool, 4);  // the residual, updated in place
      const Tensor xm = x.t.slice0(0, m), ym = y.t.slice0(0, m), rm = ref.t.slice0(0, m);
      MI_CUDA_CHECK(cudaMemcpyAsync(got.t.data, ref.t.data, ref.t.nbytes(), cudaMemcpyDeviceToDevice, stream));
      linear(*f, ym, xm, w.t);
      kernels::gated_residual(rm, ym, g, rm, stream);
      kernels::fused::gemm_gated_residual(got.t.slice0(0, m), xm, w.t, g, stream);
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      compare(*f, ref.download(), got);
    }
  }
  const std::string skipped =
      unchecked == 0 ? std::string()
                     : "; " + std::to_string(unchecked) + " larger accepted configurations not checked (run op by op)";
  if (!mismatch.empty()) {
    result.detail = mismatch;
    return result;
  }
  if (checked.empty()) {
    result.detail = "no accepted cuBLASLt configuration could be checked within the self-check's memory" + skipped +
                    ": every pair runs op by op";
    return result;
  }
  int per_op[kOps] = {};
  std::string where;
  for (const FusedConfig& f : checked) {
    for (int op = 0; op < kOps; ++op) per_op[op] += f.op == kOpNames[op];
    where += (where.empty() ? "" : ", ") + f.op + "@" + std::to_string(f.rows);
  }
  result.ok = true;
  result.validated = std::move(checked);
  result.detail = std::to_string(result.validated.size()) + " cuBLASLt configurations (o_proj " +
                  std::to_string(per_op[0]) + ", ffn_up_gate " + std::to_string(per_op[1]) + ", ffn_down " +
                  std::to_string(per_op[2]) + ") bit-identical fused, checked at " + where + skipped;
  return result;
}

}  // namespace mi::moss_dit
