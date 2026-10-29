// The NVFP4 activation quantizers (see nvfp4.h): quantize_nvfp4, rms_norm_modulate_quantize_nvfp4 and
// situ_glu_quantize_nvfp4. Built for every architecture of the library (plain fp32 arithmetic, no FP4 instructions).
//
// One thread block per row of the swizzled scale buffer, round_up(m, 128) rows: a block past m only writes its row's
// zero scale padding. Thread t holds the 16-element K blocks kb = t + i * blockDim (i < kBlocksPerThread) in registers:
// one 32-byte load per block, then the block amax, the row amax (warp shuffles + shared memory), g, the block scale
// and the 16 codes, stored as 8 bytes. Four consecutive blocks are held by four consecutive lanes (blockDim % 32 == 0),
// which gather their scale bytes with shuffles so the first lane stores them as one 32-bit word (kb % 4 == 0 starts a
// contiguous 4-byte run of the swizzled layout); blocks sfk .. round_up(sfk, 4) - 1 contribute the zero column
// padding. Every step is one IEEE fp32 operation with explicit _rn intrinsics (no FMA contraction, no FTZ), in the
// order of quantize_nvfp4_act_bf16 (weights/quant/nvfp4_codec.cpp), so the bits equal the CPU codec's.
#include <cstdint>
#include <string>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"
#include "kernels/nvfp4/nvfp4.h"
#include "kernels/quant/norm_prologue.cuh"

namespace mi::kernels::nvfp4 {
namespace {

using kernels::detail::add;
using kernels::detail::fail;
using kernels::detail::mul;
using kernels::detail::rbf;
using kernels::detail::sigmoid_f32;

constexpr int kMaxThreads = 512;
constexpr int kMaxWarps = kMaxThreads / 32;
constexpr float kBf16Max = 3.38953139e38f;  // largest finite bf16 (nvfp4_codec.cpp kBf16Max)
constexpr float kInv25 = 1.0f / 25.0f;

// MOSS _situ (P:66-67) on bf16 values. Source of truth: kernels/basic/elementwise.cu situ_value (kernels::situ_glu);
// keep the op order identical (kernels/fused has the same copy for its epilogue).
__device__ __forceinline__ float situ_value(float h, float g) {
  const float left = rbf(mul(rbf(mul(4.0f, rbf(tanhf(rbf(mul(g, 0.25f)))))), rbf(sigmoid_f32(g))));
  const float right = rbf(mul(25.0f, rbf(tanhf(rbf(mul(h, kInv25))))));
  return mul(left, right);
}

// mi::quant::e2m1_rne: midpoints .25 .75 1.25 1.75 2.5 3.5 5 with ties to the even code; the sign bit follows the fp32
// sign bit (-0.0 and small negatives give 0x8). Only called on finite values.
__device__ __forceinline__ unsigned e2m1_code(float x) {
  const float a = fabsf(x);
  const unsigned mag = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
  return mag | (__float_as_uint(x) >> 31) << 3;
}

// mi::quant::e4m3_rne on x in [0, 448] (never NaN here): subnormals (x < 2^-6) in steps of 2^-9, x * 512 exact; else
// x = m * 2^e with m in [1, 2), 3 mantissa bits rounded to nearest even ((m - 1) * 8 is exact).
__device__ __forceinline__ unsigned e4m3_code(float x) {
  if (!(x > 0.0f)) return 0;
  if (x < 0.015625f) return static_cast<unsigned>(nearbyintf(mul(x, 512.0f)));
  const int bits = __float_as_int(x);
  int e = (bits >> 23) - 127;
  const float m = __int_as_float((bits & 0x007FFFFF) | 0x3F800000);
  int q = static_cast<int>(nearbyintf(mul(add(m, -1.0f), 8.0f)));
  if (q == 8) {
    q = 0;
    ++e;
  }
  return static_cast<unsigned>((e + 7) << 3 | q);
}

// float8 e4m3fn code (sign 0, not NaN) -> its value, exactly.
__device__ __forceinline__ float e4m3_value(unsigned code) {
  const unsigned e = code >> 3, m = code & 7;
  return e == 0 ? mul(static_cast<float>(m), 0.001953125f) : __int_as_float(static_cast<int>((e + 120) << 23 | m << 20));
}

// 16 contiguous bf16 (32 bytes, 16-byte aligned) as floats.
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

enum class Mode { kPlain, kNorm, kSitu };

struct QuantArgs {
  const __nv_bfloat16* x;  // kSitu: the [m, 2F] up_gate rows
  std::int64_t ldx;
  Nvfp4Act out;
  std::int64_t m;
  int k;          // elements per output row (kSitu: F)
  int sfk;        // k / 16
  int col_tiles;  // ceil(sfk / 4)
  // kNorm only
  const __nv_bfloat16* weight;
  const __nv_bfloat16* mod_scale;
  const __nv_bfloat16* mod_shift;
  float factor;  // torch MeanOps factor: float(rows) / float(rows * k)
};

template <int kBlocksPerThread, Mode kMode, int kVT>
__global__ void quantize_kernel(QuantArgs a) {
  __shared__ float warp_max[kMaxWarps];
  __shared__ float inv_rms;
  const int t = static_cast<int>(threadIdx.x);
  const int lane = t & 31, warp = t >> 5;
  const int threads = static_cast<int>(blockDim.x);
  const std::int64_t row = blockIdx.x;
  // Scale byte of (row, kb) for kb % 4 == 0: the first of 4 contiguous bytes.
  const auto sf_word = [&](int kb) {
    return reinterpret_cast<std::uint32_t*>(a.out.sf + ((row / kSfTileRows) * a.col_tiles + kb / kSfTileCols) * 512 +
                                            (row % 32) * 16 + ((row % kSfTileRows) / 32) * 4);
  };
  if (row >= a.m) {  // padding row of the last 128-row tile: zero scales (block-uniform exit)
#pragma unroll
    for (int i = 0; i < kBlocksPerThread; ++i) {
      const int kb = t + i * threads;
      if (kb % kSfTileCols == 0 && kb < a.col_tiles * kSfTileCols) *sf_word(kb) = 0;
    }
    return;
  }

  const __nv_bfloat16* xr = a.x + row * a.ldx;
  float v[kBlocksPerThread][16];
#pragma unroll
  for (int i = 0; i < kBlocksPerThread; ++i) {
    const int kb = t + i * threads;
    if (kb >= a.sfk) continue;
    if constexpr (kMode == Mode::kSitu) {
      float h[16], g[16];
      load16(xr + kb * kBlock, h);
      load16(xr + a.k + kb * kBlock, g);
#pragma unroll
      for (int j = 0; j < 16; ++j) v[i][j] = rbf(situ_value(h[j], g[j]));  // kernels::situ_glu's bf16 output
    } else {
      load16(xr + kb * kBlock, v[i]);
    }
  }

  if constexpr (kMode == Mode::kNorm) {
    if (warp == 0) {
      const float r = kernels::detail::torch_inv_rms<kVT>(xr, a.k, a.factor, lane);
      if (lane == 0) inv_rms = r;
    }
    __syncthreads();
    const float r = inv_rms;
#pragma unroll
    for (int i = 0; i < kBlocksPerThread; ++i) {
      const int kb = t + i * threads;
      if (kb >= a.sfk) continue;
      float w[16], sc[16], sh[16];
      load16(a.weight + kb * kBlock, w);
      load16(a.mod_scale + kb * kBlock, sc);
      load16(a.mod_shift + kb * kBlock, sh);
#pragma unroll
      for (int j = 0; j < 16; ++j) v[i][j] = kernels::detail::moss_modulate(v[i][j], r, w[j], sc[j], sh[j]);
    }
  }

  // Non-finite rule, block amax, row amax. NaN -> 0 (fails v == v), +-inf -> +-bf16 max.
  float bmax[kBlocksPerThread];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlocksPerThread; ++i) {
    bmax[i] = 0.0f;
    if (t + i * threads >= a.sfk) continue;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float x = v[i][j];
      v[i][j] = x == x ? fminf(fmaxf(x, -kBf16Max), kBf16Max) : 0.0f;
      bmax[i] = fmaxf(bmax[i], fabsf(v[i][j]));
    }
    amax = fmaxf(amax, bmax[i]);
  }
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  if (lane == 0) warp_max[warp] = amax;
  __syncthreads();
  amax = warp_max[0];
  for (int w = 1; w < threads / 32; ++w) amax = fmaxf(amax, warp_max[w]);
  const float g = amax == 0.0f ? 1.0f : __fdiv_rn(amax, 2688.0f);
  if (t == 0) a.out.scale[row] = g;

  std::uint8_t* qr = a.out.q + row * (a.k / 2);
#pragma unroll
  for (int i = 0; i < kBlocksPerThread; ++i) {
    const int kb = t + i * threads;
    unsigned sf = 0;
    if (kb < a.sfk) {
      sf = e4m3_code(fminf(__fdiv_rn(__fdiv_rn(bmax[i], 6.0f), g), 448.0f));
      const float total = mul(g, e4m3_value(sf));
      uint2 packed = make_uint2(0, 0);
      if (total != 0.0f) {
#pragma unroll
        for (int j = 0; j < 16; j += 2) {
          const unsigned hi = e2m1_code(fminf(fmaxf(__fdiv_rn(v[i][j], total), -6.0f), 6.0f));
          const unsigned lo = e2m1_code(fminf(fmaxf(__fdiv_rn(v[i][j + 1], total), -6.0f), 6.0f));
          const unsigned byte = hi << 4 | lo;  // hi_first: element 2j in the high nibble
          if (j < 8) {
            packed.x |= byte << (4 * j);
          } else {
            packed.y |= byte << (4 * (j - 8));
          }
        }
      }
      *reinterpret_cast<uint2*>(qr + kb * (kBlock / 2)) = packed;
    }
    // Lanes 4s .. 4s + 3 hold blocks 4u .. 4u + 3 (u = kb / 4): little-endian word, block 4u in the low byte.
    const int base = lane & ~3;
    std::uint32_t word = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) word |= __shfl_sync(0xffffffffu, sf, base + s) << (8 * s);
    if ((lane & 3) == 0 && kb < a.col_tiles * kSfTileCols) *sf_word(kb) = word;
  }
}

// Blocks per thread and threads per block for a row of sfk blocks (cols = round_up(sfk, 4) of them are written):
// the fewest blocks per thread with at most kMaxThreads threads.
struct BlockShape {
  int blocks_per_thread;
  int threads;
};

BlockShape block_shape(int cols) {
  for (int bpt : {1, 2, 4}) {
    const int threads = ((cols + bpt - 1) / bpt + 31) / 32 * 32;
    if (threads <= kMaxThreads) return {bpt, threads};
  }
  return {4, kMaxThreads};  // cols <= 2048 (k <= kMaxQuantizeK)
}

template <Mode kMode, int kVT>
void launch(const QuantArgs& a, cudaStream_t stream) {
  const BlockShape bs = block_shape(a.col_tiles * kSfTileCols);
  const auto rows = static_cast<unsigned>((a.m + kSfTileRows - 1) / kSfTileRows * kSfTileRows);
  const dim3 grid(rows), block(static_cast<unsigned>(bs.threads));
  switch (bs.blocks_per_thread) {
    case 1: quantize_kernel<1, kMode, kVT><<<grid, block, 0, stream>>>(a); break;
    case 2: quantize_kernel<2, kMode, kVT><<<grid, block, 0, stream>>>(a); break;
    default: quantize_kernel<4, kMode, kVT><<<grid, block, 0, stream>>>(a); break;
  }
  MI_CUDA_CHECK_LAUNCH();
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// Checks the input rows (k elements each, ld >= in_cols) and the outputs; returns the arguments.
QuantArgs check(const char* op, const __nv_bfloat16* x, std::int64_t ldx, std::int64_t in_cols, const Nvfp4Act& out,
                std::int64_t m, int k) {
  if (k <= 0 || k % 32 != 0 || k > kMaxQuantizeK) {
    fail(op, "k must be a positive multiple of 32 up to " + std::to_string(kMaxQuantizeK) + ", got " + std::to_string(k));
  }
  // Grid of round_up(m, 128) blocks, and q's row offsets m * k / 2 in int64.
  if (m < 0 || m > 0x7fffff00) fail(op, "row count out of range: " + std::to_string(m));
  if (m > 0 && (x == nullptr || out.q == nullptr || out.sf == nullptr || out.scale == nullptr)) fail(op, "null operand");
  if (ldx < in_cols || ldx % 8 != 0 || !aligned16(x)) {
    fail(op, "x must be 16-byte aligned with a row stride >= " + std::to_string(in_cols) +
                 " and a multiple of 8, got " + std::to_string(ldx));
  }
  if (!aligned16(out.q) || !aligned16(out.sf) || !aligned16(out.scale)) fail(op, "q, sf and scale must be 16-byte aligned");
  QuantArgs a{};
  a.x = x;
  a.ldx = ldx;
  a.out = out;
  a.m = m;
  a.k = k;
  a.sfk = k / kBlock;
  a.col_tiles = (a.sfk + kSfTileCols - 1) / kSfTileCols;
  return a;
}

}  // namespace

void quantize_nvfp4(const __nv_bfloat16* x, std::int64_t ldx, const Nvfp4Act& out, std::int64_t m, int k,
                    cudaStream_t stream) {
  const QuantArgs a = check("nvfp4::quantize_nvfp4", x, ldx, k, out, m, k);
  if (m == 0) return;
  launch<Mode::kPlain, 0>(a, stream);
}

void rms_norm_modulate_quantize_nvfp4(const __nv_bfloat16* x, std::int64_t ldx, const __nv_bfloat16* norm_weight,
                                      const __nv_bfloat16* mod_scale, const __nv_bfloat16* mod_shift,
                                      const Nvfp4Act& out, std::int64_t m, int k, cudaStream_t stream) {
  constexpr const char* op = "nvfp4::rms_norm_modulate_quantize_nvfp4";
  QuantArgs a = check(op, x, ldx, k, out, m, k);
  if (norm_weight == nullptr || mod_scale == nullptr || mod_shift == nullptr || !aligned16(norm_weight) ||
      !aligned16(mod_scale) || !aligned16(mod_shift)) {
    fail(op, "norm_weight, mod_scale and mod_shift must be 16-byte aligned device vectors");
  }
  // kernels::rms_norm_moss's limit (norm.cu torch_reduce_shape), which torch_reduce_threads does not check.
  if (k < 128) fail(op, "row length must be >= 128, got " + std::to_string(k));
  if (m == 0) return;
  const int bw = kernels::detail::torch_reduce_threads(op, m, k);
  a.weight = norm_weight;
  a.mod_scale = mod_scale;
  a.mod_shift = mod_shift;
  a.factor = static_cast<float>(m) / static_cast<float>(m * k);
  switch (bw) {
    case 32: launch<Mode::kNorm, 1>(a, stream); break;
    case 64: launch<Mode::kNorm, 2>(a, stream); break;
    case 128: launch<Mode::kNorm, 4>(a, stream); break;
    case 256: launch<Mode::kNorm, 8>(a, stream); break;
    case 512: launch<Mode::kNorm, 16>(a, stream); break;
    default: fail(op, "unexpected torch reduce width " + std::to_string(bw));
  }
}

void situ_glu_quantize_nvfp4(const __nv_bfloat16* up_gate, std::int64_t ld, const Nvfp4Act& out, std::int64_t m,
                             int f, cudaStream_t stream) {
  const QuantArgs a = check("nvfp4::situ_glu_quantize_nvfp4", up_gate, ld, 2 * static_cast<std::int64_t>(f), out, m, f);
  if (m == 0) return;
  launch<Mode::kSitu, 0>(a, stream);
}

}  // namespace mi::kernels::nvfp4
