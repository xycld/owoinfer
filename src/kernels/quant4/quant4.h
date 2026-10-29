// W4A4 kernels (the Turbo quality tier), "ConvRot W4A4" (the community convrot_w4a4 file format): 4-bit weights and
// 4-bit activations on the sm_80+ INT4 tensor-core MMA (mma.m16n8k64 s4 x s4 -> s32). ConvRot rotation as the INT8
// tier (kernels/quant), then per-row INT4. Numerics pinned from comfy-kitchen @dba9fac (Apache-2.0): its eager
// convrot_w4a4_linear (backends/eager/convrot_w4a4.py, svdquant.py) for the activation quantization, its CUDA CUTLASS
// epilogue order for the dequantization (as the INT8 tier):
// - Rotation: x_rot = bf16(x · H256), H256 the regular Hadamard of kernels/quant, computed in the same fp32 FHT order
//   as kernels::quant::rotate_quantize (bit-identical rotated values).
// - Per token (eager quantize_signed_int4_rowwise on the bf16 x_rot): amax = bf16(max(max |x_rot|, 1e-10)) (the
//   bf16 clamp), s = bf16(amax * fp32(1/7)) (torch divides a bf16 tensor by a CPU scalar as a multiply by its fp32
//   reciprocal), q = clamp(nearbyint(bf16(x_rot / s)), -7, 7). The returned f32 scale is the bf16 s widened (eager
//   returns scales.to(float32)), and the GEMM multiplies with that.
// - Non-finite rows (eager would give a NaN / inf scale): NaN elements are ignored by the absmax and quantize to -7,
//   the absmax is clamped to the bf16 maximum, so an inf row gets a finite scale and +-inf elements saturate to +-7.
// - Storage: signed nibbles, two per byte, even column in the low nibble (comfy-kitchen _pack_int4_row_major).
//   Weights w [N, K/2] bytes with a per-row f32 scale [N] (the file's weight / weight_scale); the offline quantizer
//   is weights/quant/convrot_w4a4.h.
// - GEMM: y = bf16(((float)acc * s_x[row]) * s_w[col] (+ bias[col])), acc the exact int32 dot product (it cannot
//   overflow: |acc| <= K * 64 < 2^31 for K <= kMaxInt4GemmK). (float)acc is exact while |acc| < 2^24 and rounds to
//   nearest above, as in comfy-kitchen's epilogue. comfy-kitchen's eager int4_linear instead runs the integer matmul
//   in bf16 (acc rounded to bf16, then two bf16 scale multiplies with s_w rounded to bf16); that differs by a few
//   bf16 ULP.
//
// All pointers are device pointers. No allocation, no host synchronization: graph-capturable. Invalid arguments throw
// std::invalid_argument. sm_80 or newer.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mi::kernels::quant4 {

// Numerics version of the INT4 W4A4 path (rotation, quantization, GEMM epilogue): bump on any change to the output
// bits of an INT4 kernel, reported in the quality object's numerics (pipeline/quality_report.h).
inline constexpr int kInt4NumericsVersion = 1;

inline constexpr int kConvRotGroup = 256;
inline constexpr int kMaxRotateK = 32768;
inline constexpr int kInt4Max = 7;

// Rotate + per-token INT4 quantize. x: bf16 [m, k] (row stride ldx elements); q: packed int4 [m, k/2] bytes (row
// stride ldq bytes); scale: f32 [m]. Needs k % 256 == 0, k <= kMaxRotateK, 16-byte aligned x and q, ldx % 8 == 0,
// ldq % 16 == 0, ldq >= k/2.
void convrot_quantize_int4(const __nv_bfloat16* x, std::int64_t ldx, std::uint8_t* q, std::int64_t ldq, float* scale,
                           std::int64_t m, int k, cudaStream_t stream);

// kernels::rms_norm_modulate_moss of x fused with convrot_quantize_int4, bit for bit the two-step result. Same rules
// and the same row-count requirement as kernels::quant::rms_norm_modulate_rotate_quantize (call it once per tensor
// the reference normalizes: the fp32 mean-of-squares order depends on m).
void rms_norm_modulate_convrot_quantize_int4(const __nv_bfloat16* x, std::int64_t ldx, const __nv_bfloat16* norm_weight,
                                             const __nv_bfloat16* mod_scale, const __nv_bfloat16* mod_shift,
                                             std::uint8_t* q, std::int64_t ldq, float* scale, std::int64_t m, int k,
                                             cudaStream_t stream);

// Threadblock tile of the GEMMs (M x N x K in int4 elements). kAuto picks per shape (int4_tile_for); every tile gives
// the same bits (the int32 sum is exact).
enum class Int4Tile { kAuto, k128x256x128, k256x128x128, k64x128x128, kCount };
const char* int4_tile_name(Int4Tile tile);
Int4Tile int4_tile_for(std::int64_t m, std::int64_t n, std::int64_t k);

// Largest K: the int32 accumulator cannot overflow (|acc| <= 64 K).
inline constexpr std::int64_t kMaxInt4GemmK = 1 << 24;

// Operands of one GEMM: x_q packed int4 [m, k/2] bytes (row stride ldx bytes) with x_scale f32 [m]; w packed
// int4 [n, k/2] contiguous with w_scale f32 [n]. 16-byte aligned x_q, w, w_scale; k % 32 == 0, ldx % 16 == 0,
// ldx >= k/2; 0 <= m < 2^31, 0 < n < 2^31, 0 < k <= kMaxInt4GemmK; n % 8 == 0.
struct Int4Operands {
  const std::uint8_t* x_q = nullptr;
  std::int64_t ldx = 0;
  const float* x_scale = nullptr;
  const std::uint8_t* w = nullptr;
  const float* w_scale = nullptr;
  std::int64_t m = 0, n = 0, k = 0;
};

// The three epilogues of kernels::quant's INT8 GEMMs, with the same output rules (see kernels/quant/quant.h):
// out = bf16((acc * x_scale[r]) * w_scale[c] + bias[c]); residual += bf16(y * gate[c]); out[m, F] = situ_glu(y) of
// the stored [hidden | gate] ffn_up_gate weight (n = 2F, F % 4 == 0).
void int4_gemm(const Int4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream, Int4Tile tile = Int4Tile::kAuto);
void int4_gemm_gated_residual(const Int4Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream,
                              Int4Tile tile = Int4Tile::kAuto);
void int4_gemm_situ_glu(const Int4Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream,
                        Int4Tile tile = Int4Tile::kAuto);

}  // namespace mi::kernels::quant4
