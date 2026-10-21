#include "kernels/fused/qk_norm_rope.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"
#include "kernels/basic/norm.h"

namespace mi::kernels::fused {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kHeadDim = 128;
constexpr int kWarpsPerBlock = 8;

struct Args {
  bf16* qkv;
  std::int64_t ld;  // row stride of qkv in elements
  const bf16* q_norm;
  const bf16* k_norm;
  const bf16* table;
  std::int64_t table_ld;
  std::int64_t tokens;
  int heads;
  float factor;  // torch MeanOps factor of the per-head norm: float(rows) / float(rows * 128)
};

// Four consecutive bf16 as fp32 / back, through one 8-byte access.
__device__ __forceinline__ void load4(const bf16* p, float (&v)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const auto* h = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(h[i]);
}

// One warp per (token, q-or-k head). Lane l holds dims 4l..4l+3: exactly the thread_reduce of torch's
// reduce kernel for a 128-wide row (block width 32, one 4-element vector per thread), then its warp
// shuffle-down tree (norm.cu). RoPE pairs dim d < 64 with d + 64, i.e. lane l with lane l ^ 16.
__global__ void qk_norm_rope_kernel(Args a) {
  using namespace kernels::detail;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const std::int64_t items = a.tokens * 2 * a.heads;
  const std::int64_t warp0 = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const std::int64_t warps = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
  for (std::int64_t item = warp0; item < items; item += warps) {
    const std::int64_t token = item / (2 * a.heads);
    const int head = static_cast<int>(item % (2 * a.heads));  // q heads first, then k heads (columns head*128)
    bf16* row = a.qkv + token * a.ld + static_cast<std::int64_t>(head) * kHeadDim + 4 * lane;
    const bf16* weight = (head < a.heads ? a.q_norm : a.k_norm) + 4 * lane;

    float x[4];
    load4(row, x);
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = mul(x[i], x[i]);
    float v = add(add(add(acc[0], acc[1]), acc[2]), acc[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = add(v, __shfl_down_sync(0xffffffffu, v, off));
    const float total = __shfl_sync(0xffffffffu, v, 0);
    const float r = rsqrtf(add(mul(total, a.factor), kRmsNormEps));

    float w[4], y[4];
    load4(weight, w);
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = rbf(mul(rbf(mul(x[i], r)), w[i]));  // rms_norm_moss output (bf16)

    // Table row [sin(64) | cos(64)]; pair index t = dim mod 64.
    const int t = 4 * (lane & 15);
    const bf16* tab = a.table + token * a.table_ld;
    float si[4], co[4];
    load4(tab + t, si);
    load4(tab + 64 + t, co);
    const bool lo_half = lane < 16;
    alignas(8) bf16 out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float other = __shfl_xor_sync(0xffffffffu, y[i], 16);
      const float lo = lo_half ? y[i] : other, hi = lo_half ? other : y[i];
      // moss_rope_apply's rotate_pair: lo' = lo*cos + (-hi)*sin, hi' = hi*cos + lo*sin, each op rounded.
      out[i] = lo_half ? bf(add(rbf(mul(lo, co[i])), rbf(mul(-hi, si[i]))))
                       : bf(add(rbf(mul(hi, co[i])), rbf(mul(lo, si[i]))));
    }
    *reinterpret_cast<uint2*>(row) = *reinterpret_cast<const uint2*>(out);
  }
}

[[noreturn]] void fail(const std::string& msg) { throw std::invalid_argument("mi::kernels::fused::qk_norm_rope_moss: " + msg); }

}  // namespace

void qk_norm_rope_moss(const Tensor& qkv, int heads, const Tensor& q_norm, const Tensor& k_norm, const Tensor& table,
                       cudaStream_t stream) {
  constexpr const char* op = "qk_norm_rope_moss";
  kernels::detail::check_rows(op, "qkv", qkv, DType::kBF16);
  kernels::detail::check_vector(op, "q_norm", q_norm, kHeadDim);
  kernels::detail::check_vector(op, "k_norm", k_norm, kHeadDim);
  kernels::detail::check_rows(op, "table", table, DType::kBF16, kHeadDim);
  if (qkv.rank != 2 || table.rank != 2 || heads < 1 || qkv.dim(1) < 2LL * heads * kHeadDim || table.dim(0) != qkv.dim(0)) {
    fail("expected qkv [L, >= 2*heads*128] and table [L, 128], got " + qkv.describe() + " and " + table.describe());
  }
  const std::int64_t ld = qkv.dim(0) > 1 ? qkv.stride[0] : qkv.dim(1);
  const std::int64_t table_ld = table.dim(0) > 1 ? table.stride[0] : kHeadDim;
  const auto misaligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 != 0; };
  if (misaligned(qkv.data) || misaligned(table.data) || ld % 8 != 0 || table_ld % 8 != 0 || misaligned(q_norm.data) ||
      misaligned(k_norm.data)) {
    fail("needs 16-byte aligned rows and weights");
  }
  const std::int64_t tokens = qkv.dim(0);
  if (tokens == 0) return;
  // rms_norm_moss on a [L, heads, 128] head view sees L * heads rows (torch's reduce shape is then
  // bh = 16, bw = 32 for any L >= 1, which the kernel layout reproduces).
  const std::int64_t rows = tokens * heads;
  const Args a{qkv.ptr<bf16>(), ld, q_norm.ptr<const bf16>(), k_norm.ptr<const bf16>(), table.ptr<const bf16>(), table_ld,
               tokens, heads, static_cast<float>(rows) / static_cast<float>(rows * kHeadDim)};
  const std::int64_t items = tokens * 2 * heads;
  const auto blocks = static_cast<unsigned>(std::min<std::int64_t>((items + kWarpsPerBlock - 1) / kWarpsPerBlock, 1 << 20));
  qk_norm_rope_kernel<<<blocks, 32 * kWarpsPerBlock, 0, stream>>>(a);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::fused
