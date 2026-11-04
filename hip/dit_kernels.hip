// The DiT's kernels outside its blocks (dit_kernels.h). The op order and the bf16 rounding points are those of the
// engine's: cuBLASLt's bias epilogue and kernels/basic/elementwise.cu's residual_add for the linears, kernels/basic/norm.cu
// for the norm, elementwise.cu's silu and situ_glu. Every workgroup is one wave of 32 threads, as in block_kernels.hip.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "device_math.h"
#include "dit_kernels.h"
#include "gemm_tile.h"
#include "mma_leaf.h"

namespace hip_block {
namespace {

// rms_norm: one row per wave, torch's bw threads per row as lanes l, l + 32, ... of the wave.
constexpr int kMaxReduceThreads = 512;
struct PlainNormArgs {
  const bf16_bits* x;
  std::int64_t ldx;
  const bf16_bits* weight;
  bf16_bits* out;
  std::int64_t ldo;
  int d;
  int bw;        // torch's threads per row, a power of two from 32 to 512
  float factor;  // torch MeanOps factor: float(rows) / float(rows * d)
  float* row_stats;
};

__global__ __launch_bounds__(kWave) void rms_norm_kernel(const PlainNormArgs a) {
  __shared__ float red[kMaxReduceThreads];
  const int t = thread_id();
  const std::int64_t row = blockIdx.x;
  const bf16_bits* xr = a.x + row * a.ldx;

  // thread_reduce: torch's thread tx sums its 4-value groups tx, tx + bw, ... in 4 running values, then adds them.
  for (int tx = t; tx < a.bw; tx += kWave) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int idx = tx; idx * 4 + 3 < a.d; idx += a.bw) {
      HIP_BLOCK_UNROLL
      for (int i = 0; i < 4; ++i) {
        const float v = f(xr[idx * 4 + i]);
        acc[i] = add(acc[i], mul(v, v));
      }
    }
    red[tx] = add(add(add(acc[0], acc[1]), acc[2]), acc[3]);
  }
  leaf::barrier();
  // block_x_reduce's halving through shared memory down to one warp of 32.
  for (int off = a.bw / 2; off >= kWave; off >>= 1) {
    for (int tx = t; tx < off; tx += kWave) red[tx] = add(red[tx], red[tx + off]);
    leaf::barrier();
  }
  const float v = red[t];
  leaf::barrier();
  const float r = rsqrt_of_mean(v, a.factor, red);  // the shuffle-down tree over the 32; red[0] holds the sum
  if (a.row_stats != nullptr && t == 0) {
    a.row_stats[2 * row] = red[0];
    a.row_stats[2 * row + 1] = r;
  }

  bf16_bits* outr = a.out + row * a.ldo;
  for (int c = t; c < a.d; c += kWave) outr[c] = bf(mul(rbf(mul(f(xr[c]), r)), f(a.weight[c])));
}

struct UnaryArgs {
  const bf16_bits* x;
  bf16_bits* out;
  std::int64_t n;
};
__global__ __launch_bounds__(kWave) void silu_kernel(const UnaryArgs a) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * kWave + thread_id();
  if (i < a.n) a.out[i] = bf(silu_f32(f(a.x[i])));
}

// situ_glu: one row per workgroup along x, column blocks of the wave along y.
struct SituGluArgs {
  const bf16_bits* up_gate;
  std::int64_t ld;
  bf16_bits* out;
  std::int64_t ldo;
  int f;
};
constexpr int kSituColsPerWave = 256;
__global__ __launch_bounds__(kWave) void situ_glu_kernel(const SituGluArgs a) {
  const std::int64_t row = blockIdx.x;
  const bf16_bits* in = a.up_gate + row * a.ld;
  bf16_bits* out = a.out + row * a.ldo;
  const int c0 = static_cast<int>(blockIdx.y) * kSituColsPerWave;
  const int c1 = c0 + kSituColsPerWave < a.f ? c0 + kSituColsPerWave : a.f;
  for (int c = c0 + thread_id(); c < c1; c += kWave) out[c] = bf(situ_value(f(in[c]), f(in[a.f + c])));
}

// rope_table: one row per wave, lane l writing columns l, l + 32, ...
struct RopeArgs {
  bf16_bits* out;
  std::int64_t hw;
  int width, text_length;
  float h_mid, w_mid;  // the fp32 (n - 1) / 2 of each axis
  float bands[16];     // 1 / 10000^(i / 16), the reference's fp32 values (kernels/basic/rope.h)
};
__global__ __launch_bounds__(kWave) void rope_table_kernel(const RopeArgs a) {
  const std::int64_t r = blockIdx.x;
  bf16_bits* out = a.out + r * kHeadDim;
  for (int c = thread_id(); c < kHeadDim; c += kWave) {
    const int p = c & 63, axis = p >> 4, band = p & 15;
    float coord = 0.0f;
    if (r < a.hw) {
      if (axis == 1) coord = static_cast<float>(a.text_length);
      if (axis == 2) coord = static_cast<float>(r / a.width) - a.h_mid;
      if (axis == 3) coord = static_cast<float>(r % a.width) - a.w_mid;
    } else if (axis == 1) {
      coord = static_cast<float>(r - a.hw);
    }
    const float phase = mul(coord, a.bands[band]);
    out[c] = bf(c < 64 ? sinf(phase) : cosf(phase));
  }
}

// torch's setReduceConfig for a last-dim reduction (kernels/basic/norm.cu's torch_reduce_shape): the threads per row.
int last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}
int torch_threads_per_row(std::int64_t rows, int d) {
  const int dim0 = d / 4;
  const int dim0_pow2 = dim0 < kMaxReduceThreads ? last_pow2(dim0) : kMaxReduceThreads;
  const int dim1_pow2 = rows < kMaxReduceThreads ? last_pow2(static_cast<int>(rows)) : kMaxReduceThreads;
  int bw = std::min(dim0_pow2, 32);
  const int bh = std::min(dim1_pow2, kMaxReduceThreads / bw);
  bw = std::min(dim0_pow2, kMaxReduceThreads / bh);
  // torch splits a row across a block's warps once each thread would sum >= min(16 bh, 256) values; the engine refuses it.
  if ((d + bw - 1) / bw >= std::min(bh * 16, 256)) {
    throw std::runtime_error("hip_block::rms_norm: row length " + std::to_string(d) + " needs torch's cross-warp split");
  }
  return bw;
}

template <Epilogue kEpilogue>
void launch_gemm(const char* op, bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w,
                 std::int64_t ldw, const bf16_bits* bias, int m, int n, int k, hipStream_t stream) {
  if (m == 0 || n == 0) return;
  const Blocking bl = blocking_of(m, n, k, kGemmTileM, kGemmTileN, kGemmTileN);
  const GemmArgs a{out, ldo, x, ldx, w, ldw, bias, m, n, k, bl.bm, bl.bn, nullptr, nullptr};
  gemm_kernel<kEpilogue><<<dim3(bl.workgroups), dim3(kWave), 0, stream>>>(a);
  check_launch(op);
}

}  // namespace

void gemm_bias(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w, std::int64_t ldw,
               const bf16_bits* bias, int m, int n, int k, hipStream_t stream) {
  launch_gemm<Epilogue::kBias>("gemm_bias", out, ldo, x, ldx, w, ldw, bias, m, n, k, stream);
}

void gemm_bias_residual(bf16_bits* residual, std::int64_t ldr, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w,
                        std::int64_t ldw, const bf16_bits* bias, int m, int n, int k, hipStream_t stream) {
  launch_gemm<Epilogue::kBiasResidual>("gemm_bias_residual", residual, ldr, x, ldx, w, ldw, bias, m, n, k, stream);
}

void rms_norm(const bf16_bits* x, std::int64_t ldx, const bf16_bits* weight, bf16_bits* out, std::int64_t ldo,
              std::int64_t rows, int d, hipStream_t stream, float* row_stats) {
  if (d % 4 != 0 || d < 128) throw std::runtime_error("hip_block::rms_norm: d must be a multiple of 4 and >= 128");
  if (rows == 0) return;
  const PlainNormArgs a{x, ldx, weight, out, ldo, d, torch_threads_per_row(rows, d),
                        static_cast<float>(rows) / static_cast<float>(rows * d), row_stats};
  rms_norm_kernel<<<dim3(static_cast<unsigned>(rows)), dim3(kWave), 0, stream>>>(a);
  check_launch("rms_norm");
}

void silu(const bf16_bits* x, bf16_bits* out, std::int64_t n, hipStream_t stream) {
  if (n == 0) return;
  silu_kernel<<<dim3(grid_of(n, kWave)), dim3(kWave), 0, stream>>>(UnaryArgs{x, out, n});
  check_launch("silu");
}

void situ_glu(const bf16_bits* up_gate, std::int64_t ld, bf16_bits* out, std::int64_t ldo, std::int64_t rows, int f,
              hipStream_t stream) {
  if (rows == 0 || f == 0) return;
  const SituGluArgs a{up_gate, ld, out, ldo, f};
  situ_glu_kernel<<<dim3(static_cast<unsigned>(rows), grid_of(f, kSituColsPerWave)), dim3(kWave), 0, stream>>>(a);
  check_launch("situ_glu");
}

void rope_table(bf16_bits* out, int height, int width, int text_length, hipStream_t stream) {
  constexpr std::uint32_t kBandBits[16] = {0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2,
                                           0x3d0186e3, 0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528,
                                           0x3a83126f, 0x3a136a16, 0x39a5cb60, 0x393a7753};
  if (height < 0 || width < 0 || text_length < 0) throw std::runtime_error("hip_block::rope_table: negative size");
  const std::int64_t hw = static_cast<std::int64_t>(height) * width, n = hw + text_length;
  if (n == 0) return;
  RopeArgs a{out, hw, width, text_length, static_cast<float>((height - 1) / 2.0), static_cast<float>((width - 1) / 2.0), {}};
  for (int i = 0; i < 16; ++i) std::memcpy(&a.bands[i], &kBandBits[i], sizeof(float));
  rope_table_kernel<<<dim3(static_cast<unsigned>(n)), dim3(kWave), 0, stream>>>(a);
  check_launch("rope_table");
}

}  // namespace hip_block
