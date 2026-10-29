// rotate_quantize and rms_norm_modulate_rotate_quantize (see quant.h).
//
// One thread block per row, W warps. A half-warp holds one 256-group, 16 elements per lane, so a warp holds a pair of
// groups (2p, 2p + 1) at a time; warp w takes pairs w, w + W, ... (up to kPairsPerWarp of them). With e = d0 + 4 d1 +
// 16 d2 + 64 d3 the base-4 digits of an element's index in its group:
// - load layout: lane h (of the half) holds e = 16 h + j, so j = d0 + 4 d1 and stages 0 and 1 (strides 1, 4) only
//   combine registers with compile-time indices (32-byte loads);
// - one 16 x 16 transpose through shared memory (row stride 17: conflict-free both ways) gives e = h + 16 j, so
//   j = d2 + 4 d3 and stages 2 and 3 (strides 16, 64) are register-local too.
// Every stage output is comfy-kitchen's 0.5 * (((m0' + m1') + m2') + m3') with operand 3 - d negated, in that order;
// the layout only decides where the operands come from, so the bits do not depend on it. The row absmax is a max
// (exact in any order). The quantized bytes go back through the same transpose, so each lane stores its 16 contiguous
// bytes of a group with one 16-byte store.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"
#include "kernels/quant/norm_prologue.cuh"
#include "kernels/quant/quant.h"

namespace mi::kernels::quant {
namespace {

using kernels::detail::add;
using kernels::detail::fail;
using kernels::detail::mul;
using kernels::detail::rbf;
using kernels::detail::torch_inv_rms;
using kernels::detail::torch_reduce_threads;

constexpr int kMaxWarps = 16;
constexpr int kStride = 17;                 // shared-memory row stride of the transpose
constexpr int kHalfFloats = 16 * kStride;   // one half-warp's 16 x 16 tile (272 floats: halves land on disjoint banks)
constexpr float kInv127 = 1.0f / 127.0f;  // torch: absmax / 127.0 multiplies by the fp32 reciprocal on CUDA
constexpr float kMinScale = 1e-30f;      // comfy-kitchen: .clamp(min=1e-30)
constexpr float kBf16Max = 3.38953139e38f;  // comfy-kitchen CUDA finite_absmax_for_int8_scale<bf16>

// One radix-4 butterfly on operands ordered by digit: y_d = 0.5 * (((m0' + m1') + m2') + m3') with operand 3 - d
// negated (row d of H4), as comfy-kitchen's h4_row_dot. The shared partial sums are the same roundings
// (a + b, and -a + b == -(a - b) exactly); FADD of a negated value is FSUB. _rn intrinsics: no FMA contraction.
__device__ __forceinline__ void h4(float& a, float& b, float& c, float& d) {
  const float s = add(a, b);
  const float t = add(a, -b);
  const float y0 = mul(0.5f, add(add(s, c), -d));
  const float y1 = mul(0.5f, add(add(s, -c), d));
  const float y2 = mul(0.5f, add(add(t, c), d));
  const float y3 = mul(0.5f, add(add(-t, c), d));
  a = y0;
  b = y1;
  c = y2;
  d = y3;
}

// Two radix-4 stages over register digits: first j & 3 (stride 1 in j), then j >> 2 (stride 4 in j).
__device__ __forceinline__ void two_stages(float (&v)[16]) {
#pragma unroll
  for (int q = 0; q < 4; ++q) h4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
#pragma unroll
  for (int r = 0; r < 4; ++r) h4(v[r], v[r + 4], v[r + 8], v[r + 12]);
}

// x_rot = x · H256 for the half-warp's group, fp32: in = load layout (e = 16 h + j), out = e = h + 16 j. tile: this
// half's kHalfFloats shared floats. The whole warp calls it (warp-synchronous transpose).
__device__ __forceinline__ void hadamard256(float (&v)[16], float* tile, int h) {
  two_stages(v);  // stages 0, 1: digits d0, d1
#pragma unroll
  for (int j = 0; j < 16; ++j) tile[h * kStride + j] = v[j];
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = tile[j * kStride + h];
  __syncwarp();  // the tile is reused by the next group
  two_stages(v);  // stages 2, 3: digits d2, d3
}

// 16 contiguous bf16 (32 bytes) as floats.
__device__ __forceinline__ void load16(const __nv_bfloat16* p, float (&v)[16]) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const uint4 raw = reinterpret_cast<const uint4*>(p)[half];
    const auto* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(b[i]);
      v[8 * half + 2 * i] = f.x;
      v[8 * half + 2 * i + 1] = f.y;
    }
  }
}

struct RotateArgs {
  const __nv_bfloat16* x;
  std::int64_t ldx;
  std::int8_t* q;
  std::int64_t ldq;
  float* scale;
  int k;
  // rms_norm_modulate_rotate_quantize only
  const __nv_bfloat16* weight;
  const __nv_bfloat16* mod_scale;
  const __nv_bfloat16* mod_shift;
  float factor;  // torch MeanOps factor: float(rows) / float(rows * k)
  __nv_bfloat16* y;  // optional copy of the bf16 norm output (row stride ldy)
  std::int64_t ldy;
};

// 16 floats that are bf16 values, stored as 16 contiguous bf16 (32 bytes, exact).
__device__ __forceinline__ void store16(__nv_bfloat16* p, const float (&v)[16]) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint4 raw;
    auto* b = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(v[8 * half + 2 * i], v[8 * half + 2 * i + 1]);
    reinterpret_cast<uint4*>(p)[half] = raw;
  }
}

// kVT == 0: standalone rotate_quantize. kVT > 0: the MOSS RMSNorm + modulate prologue first (torch reduce with
// 32 * kVT threads per row).
template <int kPairsPerWarp, int kVT>
__global__ void rotate_quantize_kernel(RotateArgs a) {
  extern __shared__ float tiles[];  // warps x 2 x kHalfFloats (sized at launch: occupancy)
  __shared__ float warp_max[kMaxWarps];
  __shared__ float inv_rms;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;
  const int half = lane >> 4, h = lane & 15;
  const int groups = a.k / kConvRotGroup;
  const std::int64_t row = blockIdx.x;
  const __nv_bfloat16* xr = a.x + row * a.ldx;
  float* tile = tiles + (2 * warp + half) * kHalfFloats;

  // This lane's group of pair slot i (may be past the row end for the last odd group: then unused).
  const auto group_of = [&](int i) { return 2 * (warp + i * warps) + half; };
  float v[kPairsPerWarp][16];
#pragma unroll
  for (int i = 0; i < kPairsPerWarp; ++i) {
    if (group_of(i) < groups) {
      load16(xr + group_of(i) * kConvRotGroup + 16 * h, v[i]);
    } else {
#pragma unroll
      for (int j = 0; j < 16; ++j) v[i][j] = 0.0f;  // the unused half of a last pair still runs the butterflies
    }
  }

  if constexpr (kVT > 0) {
    if (warp == 0) {
      const float r = torch_inv_rms<kVT>(xr, a.k, a.factor, lane);
      if (lane == 0) inv_rms = r;
    }
    __syncthreads();
    const float r = inv_rms;
#pragma unroll
    for (int i = 0; i < kPairsPerWarp; ++i) {
      if (group_of(i) >= groups) continue;
      const int c0 = group_of(i) * kConvRotGroup + 16 * h;
      float w[16], sc[16], sh[16];
      load16(a.weight + c0, w);
      load16(a.mod_scale + c0, sc);
      load16(a.mod_shift + c0, sh);
#pragma unroll
      for (int j = 0; j < 16; ++j) v[i][j] = kernels::detail::moss_modulate(v[i][j], r, w[j], sc[j], sh[j]);
      if (a.y != nullptr) store16(a.y + row * a.ldy + c0, v[i]);
    }
  }

  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kPairsPerWarp; ++i) {
    if (2 * (warp + i * warps) >= groups) continue;  // warp-uniform: the whole warp transposes together
    hadamard256(v[i], tile, h);
    if (group_of(i) >= groups) continue;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      v[i][j] = rbf(v[i][j]);  // the rotated activation is a bf16 tensor
      amax = fmaxf(amax, fabsf(v[i][j]));
    }
  }
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  if (lane == 0) warp_max[warp] = amax;
  __syncthreads();
  amax = warp_max[0];
  for (int w = 1; w < warps; ++w) amax = fmaxf(amax, warp_max[w]);

  // fmaxf drops NaN operands, so a NaN element never reaches amax (comfy-kitchen CUDA's block max does the same) and
  // an inf one is clamped to the bf16 maximum.
  const float s = fmaxf(mul(fminf(amax, kBf16Max), kInv127), kMinScale);
  if (threadIdx.x == 0) a.scale[row] = s;
  const float s_bf = rbf(s);  // x / scale runs on bf16 tensors
  std::int8_t* qr = a.q + row * a.ldq;
#pragma unroll
  for (int i = 0; i < kPairsPerWarp; ++i) {
    if (2 * (warp + i * warps) >= groups) continue;  // warp-uniform, as above
    // Element h + 16 j -> tile row j, column h; read back as row h: elements 16 h .. 16 h + 15.
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      // torch.round: half to even. fmaxf maps NaN to -128 (comfy-kitchen CUDA's clamp), +-inf saturate.
      const float qv = nearbyintf(rbf(__fdiv_rn(v[i][j], s_bf)));
      tile[j * kStride + h] = fminf(127.0f, fmaxf(-128.0f, qv));
    }
    __syncwarp();
    uint4 packed;
    auto* bytes = reinterpret_cast<std::int8_t*>(&packed);
#pragma unroll
    for (int j = 0; j < 16; ++j) bytes[j] = static_cast<std::int8_t>(tile[h * kStride + j]);
    __syncwarp();  // the tile is reused by the next pair
    if (group_of(i) < groups) *reinterpret_cast<uint4*>(qr + group_of(i) * kConvRotGroup + 16 * h) = packed;
  }
}

// Pairs of groups per warp and warps per block for a row of `groups` groups: at most 8 warps while 4 pairs per warp
// suffice, else up to kMaxWarps.
struct BlockShape {
  int pairs_per_warp;
  int warps;
};

BlockShape block_shape(int groups) {
  const int pairs = (groups + 1) / 2;
  for (int ppw : {1, 2, 4}) {
    if ((pairs + ppw - 1) / ppw <= 8) return {ppw, (pairs + ppw - 1) / ppw};
  }
  return {4, (pairs + 3) / 4};
}

template <int kVT>
void launch_rows(const char* op, const RotateArgs& a, std::int64_t m, cudaStream_t stream) {
  const BlockShape bs = block_shape(a.k / kConvRotGroup);
  if (bs.warps > kMaxWarps) fail(op, "row too long");
  if (m > 0x7fffffff) fail(op, "too many rows");
  const dim3 grid(static_cast<unsigned>(m)), block(static_cast<unsigned>(32 * bs.warps));
  const std::size_t smem = sizeof(float) * 2 * kHalfFloats * static_cast<std::size_t>(bs.warps);  // <= 35 KB
  switch (bs.pairs_per_warp) {
    case 1: rotate_quantize_kernel<1, kVT><<<grid, block, smem, stream>>>(a); break;
    case 2: rotate_quantize_kernel<2, kVT><<<grid, block, smem, stream>>>(a); break;
    default: rotate_quantize_kernel<4, kVT><<<grid, block, smem, stream>>>(a); break;
  }
  MI_CUDA_CHECK_LAUNCH();
}

// No shared alignment helper exists (kernels/basic/detail.h has none); int8_gemm.cu has its own aligned16.
bool aligned(const void* p, std::uintptr_t bytes) { return reinterpret_cast<std::uintptr_t>(p) % bytes == 0; }

RotateArgs check_rotate(const char* op, const __nv_bfloat16* x, std::int64_t ldx, std::int8_t* q, std::int64_t ldq,
                        float* scale, std::int64_t m, int k) {
  if (k <= 0 || k % kConvRotGroup != 0 || k > kMaxRotateK) {
    fail(op, "k must be a positive multiple of 256 up to " + std::to_string(kMaxRotateK) + ", got " + std::to_string(k));
  }
  if (m < 0) fail(op, "negative row count");
  if (m > 0 && (x == nullptr || q == nullptr || scale == nullptr)) fail(op, "null operand");
  if (ldx < k || ldq < k || ldx % 8 != 0 || ldq % 16 != 0 || !aligned(x, 16) || !aligned(q, 16)) {
    fail(op, "needs 16-byte aligned x and q with ldx >= k, ldq >= k, ldx % 8 == 0 and ldq % 16 == 0 (k " +
                 std::to_string(k) + ", ldx " + std::to_string(ldx) + ", ldq " + std::to_string(ldq) + ")");
  }
  RotateArgs a{};
  a.x = x;
  a.ldx = ldx;
  a.q = q;
  a.ldq = ldq;
  a.scale = scale;
  a.k = k;
  return a;
}

}  // namespace

void rotate_quantize(const __nv_bfloat16* x, std::int64_t ldx, std::int8_t* q, std::int64_t ldq, float* scale,
                     std::int64_t m, int k, cudaStream_t stream) {
  constexpr const char* op = "quant::rotate_quantize";
  const RotateArgs a = check_rotate(op, x, ldx, q, ldq, scale, m, k);
  if (m == 0) return;
  launch_rows<0>(op, a, m, stream);
}

void rms_norm_modulate_rotate_quantize(const __nv_bfloat16* x, std::int64_t ldx, const __nv_bfloat16* norm_weight,
                                       const __nv_bfloat16* mod_scale, const __nv_bfloat16* mod_shift, std::int8_t* q,
                                       std::int64_t ldq, float* scale, std::int64_t m, int k, cudaStream_t stream,
                                       __nv_bfloat16* y, std::int64_t ldy) {
  constexpr const char* op = "quant::rms_norm_modulate_rotate_quantize";
  RotateArgs a = check_rotate(op, x, ldx, q, ldq, scale, m, k);
  if (!aligned(norm_weight, 16) || !aligned(mod_scale, 16) || !aligned(mod_shift, 16) || norm_weight == nullptr ||
      mod_scale == nullptr || mod_shift == nullptr) {
    fail(op, "norm_weight, mod_scale and mod_shift must be 16-byte aligned device vectors");
  }
  if (y != nullptr && (!aligned(y, 16) || ldy < k || ldy % 8 != 0)) {
    fail(op, "y must be 16-byte aligned with ldy >= k and ldy % 8 == 0 (ldy " + std::to_string(ldy) + ")");
  }
  a.y = y;
  a.ldy = ldy;
  if (m == 0) return;
  const int bw = torch_reduce_threads(op, m, k);
  a.weight = norm_weight;
  a.mod_scale = mod_scale;
  a.mod_shift = mod_shift;
  a.factor = static_cast<float>(m) / static_cast<float>(m * k);
  switch (bw) {
    case 32: launch_rows<1>(op, a, m, stream); break;
    case 64: launch_rows<2>(op, a, m, stream); break;
    case 128: launch_rows<4>(op, a, m, stream); break;
    case 256: launch_rows<8>(op, a, m, stream); break;
    default: launch_rows<16>(op, a, m, stream); break;
  }
}

}  // namespace mi::kernels::quant
