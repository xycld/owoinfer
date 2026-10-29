// INT8 ConvRot kernels (the High quality tier): online rotation + per-token INT8 quantization of the activations,
// and the INT8 x INT8 -> INT32 GEMM with the dequantization and the following elementwise op fused into its epilogue.
//
// Numerics, pinned from comfy-kitchen @dba9fac (Apache-2.0), the library ComfyUI runs these files with. The
// activation reference is comfy-kitchen's CUDA kernel (quantize_int8_rowwise_convrot_kernel) for the rotation order
// and the inf / NaN handling, plus its eager path's bf16 rounding of the rotated row before the absmax:
// - Rotation: x_rot = x · H per contiguous 256-column group, H = kron^4(H4) / 16 with the regular (not Sylvester)
//   H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]. Computed in fp32 in the CUDA kernel's order: 4 radix-4
//   stages over the group index digits (strides 1, 4, 16, 64), each output y_i = 0.5 * (((x0' + x1') + x2') + x3')
//   with operand 3 - i negated. Then rounded to bf16, as eager's rotated tensor is.
// - Per token: amax = max |x_rot| over the bf16 row, NaN elements ignored (fmaxf); s = max(min(amax, bf16 max) *
//   fp32(1/127), 1e-30) (torch divides by a CPU scalar as a multiply by its fp32 reciprocal); q = clamp(nearbyint(
//   bf16(x_rot / bf16(s))), -128, 127) with NaN -> -128, +inf -> 127, -inf -> -128 (fmaxf / fminf clamps). The fp32
//   s is what the GEMM epilogue multiplies with. -128 is reachable: bf16(s) may round below amax / 127.
// - Differences from comfy-kitchen's CUDA kernel: (1) its amax runs over the fp32 rotated values; here, as in eager,
//   over the bf16-rounded ones, so s (and then q) differs on rows whose largest element changes in the rounding;
//   (2) no stochastic-rounding mode (round-to-nearest-even only). Differences from eager: eager's amax propagates
//   NaN (a NaN row gets s = NaN) and has no bf16-max clamp (an inf row gets s = inf).
// - GEMM epilogue: y = bf16(((float)acc * s_x[row]) * s_w[col] (+ bias[col])), the order of comfy-kitchen's CUDA
//   CUTLASS epilogue (its eager fallback multiplies acc * (s_x * s_w) and adds the bias after rounding; that differs
//   by at most 1 bf16 ULP on rare elements). acc is exact as long as |acc| < 2^31: |acc| <= K * |x|max * |w|max,
//   which with comfy-kitchen's weights (|w| <= 127, |x| <= 128) means K <= 132104; kMaxInt8GemmK = 2^17 - 1 keeps it
//   exact for any int8 operands (128 * 128 * K < 2^31).
//
// All pointers are device pointers, all row strides are in elements. No allocation, no host synchronization:
// graph-capturable. Invalid arguments throw std::invalid_argument. sm_80 or newer.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mi::kernels::quant {

// Numerics version of the INT8 ConvRot path (rotation order, quantization, GEMM epilogues): bump on any change to the
// output bits, reported in the quality object's numerics (pipeline/quality_report.h).
inline constexpr std::uint32_t kInt8NumericsVersion = 1;

// Rotation group (every ConvRot file uses 256); rotated inputs need K % kConvRotGroup == 0.
inline constexpr int kConvRotGroup = 256;
// Largest K the rotate kernels take (a row is held in registers by one thread block).
inline constexpr int kMaxRotateK = 32768;

// Standalone rotate + per-token quantize, for the o-proj input (attention output) and the ffn_down input (SiTU-GLU
// output). x: bf16 [m, k] with row stride ldx; q: int8 [m, k] with row stride ldq; scale: f32 [m].
// Needs k % 256 == 0, k <= kMaxRotateK, 16-byte aligned x and q with ldx % 8 == 0, ldq % 16 == 0.
void rotate_quantize(const __nv_bfloat16* x, std::int64_t ldx, std::int8_t* q, std::int64_t ldq, float* scale,
                     std::int64_t m, int k, cudaStream_t stream);

// The MOSS pre-attention / pre-FFN input of the qkv and ffn_up_gate GEMMs, fused: the bf16 tensor
// kernels::rms_norm_modulate_moss(x, norm_weight, mod_scale, mod_shift) would produce for these m rows, then
// rotate_quantize of it, bit for bit, without storing the bf16 tensor. norm_weight, mod_scale, mod_shift: contiguous
// bf16 [k]. Same layout rules as rotate_quantize, plus kernels::rms_norm_moss's row-length limit (k <= 7680 always
// works).
// Row-count requirement: the fp32 mean of squares is summed in the order of torch's reduce kernel for an [m, k]
// input, and that order (threads per row) depends on m as well as k. A row's result is bit-exact only against a
// reference norm over the same m rows: call it once per run the reference normalizes as one tensor (one DiT block
// call), not in row chunks of a different size.
void rms_norm_modulate_rotate_quantize(const __nv_bfloat16* x, std::int64_t ldx, const __nv_bfloat16* norm_weight,
                                       const __nv_bfloat16* mod_scale, const __nv_bfloat16* mod_shift, std::int8_t* q,
                                       std::int64_t ldq, float* scale, std::int64_t m, int k, cudaStream_t stream);

// Largest K the INT8 GEMMs take: the int32 accumulator cannot overflow (see above).
inline constexpr std::int64_t kMaxInt8GemmK = 131071;

// Threadblock tile of the INT8 GEMMs (M x N x K). kAuto picks per shape (int8_tile_for); the others force a tile,
// for the autotune benchmark and the tests (every tile gives the same bits: the int32 sum is exact).
enum class Int8Tile { kAuto, k128x128x64, k128x256x64, k256x128x64, k64x128x64, k128x128x128, kCount };
const char* int8_tile_name(Int8Tile tile);
// The tile kAuto runs for an m x n x k problem on the current device.
Int8Tile int8_tile_for(std::int64_t m, std::int64_t n, std::int64_t k);

// Operands of one INT8 GEMM: x_q int8 [m, k] (row stride ldx) with per-row scale x_scale f32 [m], from the rotate
// kernels; w int8 [n, k] contiguous with per-output-channel w_scale f32 [n], the stored weight / weight_scale.
// Layout: 16-byte aligned x_q, w and w_scale; k % 16 == 0 and ldx % 16 == 0. 0 <= m < 2^31, 0 < n < 2^31,
// 0 < k <= kMaxInt8GemmK.
struct Int8Operands {
  const std::int8_t* x_q = nullptr;
  std::int64_t ldx = 0;
  const float* x_scale = nullptr;
  const std::int8_t* w = nullptr;
  const float* w_scale = nullptr;
  std::int64_t m = 0, n = 0, k = 0;
};

// out[m, n] = bf16((acc * x_scale[r]) * w_scale[c] + bias[c]), bias: bf16 [n] or nullptr. out: bf16, row stride ldo,
// 16-byte aligned, ldo % 8 == 0, n % 8 == 0; bias 16-byte aligned.
void int8_gemm(const Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream, Int8Tile tile = Int8Tile::kAuto);

// residual[m, n] = residual + bf16(y * gate[c]) with y the bf16 int8_gemm output (bias: bf16 [n] or nullptr), as
// kernels::fused::gemm_gated_residual on the bf16 GEMM (the o-proj / ffn_down linear then kernels::gated_residual,
// P:142, P:148). residual is read and written in place (row stride ldr); gate: bf16 [n]. Alignment as int8_gemm.
void int8_gemm_gated_residual(const Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream,
                              Int8Tile tile = Int8Tile::kAuto);

// out[m, F] = situ_glu(y) for the stored [hidden | gate] ffn_up_gate weight w [2F, k], w_scale [2F] (op.n = 2F), with
// y the bf16 int8_gemm output (no bias): both halves rounded to bf16, then MOSS _situ in the op order of
// kernels::situ_glu. The [m, 2F] intermediate is never stored. The GEMM reads w's rows (and w_scale) permuted so that
// each epilogue access holds 4 hidden columns and their 4 gate columns: F % 4 == 0. out: bf16 [m, F], row stride
// ldo, 8-byte aligned, ldo % 4 == 0. The epilogue reads SiTU's two bf16 factors from tables over every bf16 input
// (256 KiB of static device memory, filled once per device by a small kernel: the first call outside a stream
// capture fills them on `stream` and synchronizes it, the one host synchronization of these kernels): the same bits
// as computing them.
void int8_gemm_situ_glu(const Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream,
                        Int8Tile tile = Int8Tile::kAuto);

}  // namespace mi::kernels::quant
