// 8-bit (approximate) scaled-dot-product attention forward, SageAttention style, on kernels vendored from
// thu-ml/SageAttention (third_party/sageattention, Apache-2.0) behind plain C++ launchers. Lossy by design: it is an
// opt-in accelerator, never the exact path (kernels/attention/flash_attention.h is).
//
// Every variant quantizes QK^T to INT8: Q per warp of rows (WARP_Q rows of a 128-row CTA tile), K per 64-key block
// after smoothing (each sequence's per-head mean of K is subtracted first: softmax is invariant to it, and it removes
// the shared offset that would otherwise eat INT8 range), int32 accumulation. (Upstream's finer per-thread Q/K scales
// were measured on the DiT's activations: no consistent accuracy gain, 3-9% slower attention kernel.) P·V then runs
// in one of:
//   kInt8Fp16        fp16 P and V, fp16 accumulation inside each 64-key tile, flushed into an fp32 accumulator
//                    (SageAttention2 "fp16+fp32"; WARP_Q = 16)
//   kInt8Fp16F32Acc  fp16 P and V, fp32 accumulation (SageAttention1-style PV; WARP_Q = 32)
//   kInt8Fp8         SageAttention2++: e4m3 P and V (V per channel), mma m16n8k32 e4m3 with fp16 accumulation
//                    over 2 MMAs (one 64-key tile) flushed into fp32, P scaled to [0, p_range] and V to
//                    [-v_range, v_range] so the fp16 partial sums cannot overflow (see Options::p_range)
//   kInt8Fp8F32Acc   e4m3 P and V with fp32 accumulation (SageAttention2 "fp32+fp32")
//
// Requirements (attention_q8 throws std::runtime_error when one is not met; supported() tells in advance):
//   - fp16 variants: a compute capability 8.0+ device and a kernel image compiled for 8.0+;
//   - fp8 variants: a compute capability 8.9+ device and a kernel image compiled for 8.9+ (CMAKE_CUDA_ARCHITECTURES
//     must include 89 or newer; a build for 80/86 only would run its sm_80 image, whose e4m3 MMAs are __brkpt);
//   - kInt8Fp8 also needs nvcc >= 12.8 at build time: older compilers have no fp16-accumulating e4m3 MMA, and the
//     vendored wrapper then compiles to __brkpt (cmake/modules/attention_q8.cmake warns at configure time).
//
// V magnitude limits (inputs are not scaled per channel on the fp16 PV path):
//   kInt8Fp16        max|V| <= 65504 / 64 = 1023.5: a 64-key tile's fp16 partial sum of P·V (P <= 1) must not
//                    overflow fp16
//   kInt8Fp16F32Acc  max|V| <= 65504: V is converted to fp16
//   fp8 variants     none beyond finite V (V is scaled per channel)
// Past the limit the output holds Inf/NaN. Not checked unless varlen_debug_checks(), which then also measures max|V|
// in the statistics pass (one stream sync) and throws std::invalid_argument when it is over the variant's limit.
//
// Interface semantics are those of attention::flash_fwd for the DiT's joint attention (non-causal, varlen):
//   q, out: bf16 [total_q, num_heads,    128]       k, v: bf16 [total_k, num_kv_heads, 128]
//   cu_seqlens_q / cu_seqlens_k: int32 [batch + 1] on the device (cu[0] = 0, non-decreasing, cu[batch] = total)
//   max_seqlen (host) >= every q and k sequence length (it sizes the grid and the scale buffers)
// Only head_dim must be contiguous; row and head strides are free (multiples of 8 elements, 16-byte aligned base),
// so q/k/v can be strided views into a fused QKV projection output. GQA as flash_fwd (contiguous head groups).
// A sequence never sees another's keys: every per-sequence statistic (K mean, V channel scale) and every tile is
// bounded by its own cu_seqlens, and padded tile rows are masked (keys) or zero (V).
//
// Workspace (attention_q8_workspace_bytes; 256-byte aligned base), carved in this order:
//   q_i8    int8 [total_q, num_heads, 128]                 quantized Q
//   k_i8    int8 [total_k, num_kv_heads, 128]              quantized smoothed K
//   v       fp16 [total_k, num_kv_heads, 128]              (fp16 variants) V converted from bf16
//        or e4m3 [num_kv_heads, 128, padded_k]             (fp8 variants) V transposed; sequence b occupies columns
//                                                          [sum_{i<b} roundup(len_i, 64), ...) zero-padded to a
//                                                          multiple of 64 = CTA_K, and the 16 keys of each 16-key
//                                                          group are stored in the order
//                                                          0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15 (the e4m3 MMA's
//                                                          A-fragment order of P); padded_k = round_up(total_k + 64 * batch, 64)
//   q_scale fp32 [batch, num_heads, ceil(max_seqlen/128) * (128/WARP_Q)]
//   k_scale fp32 [batch, num_kv_heads, ceil(max_seqlen/64)]
//   k_mean  fp32 [batch, num_kv_heads, 128]                each sequence's K mean, rounded to bf16 (as upstream)
//   v_scale fp32 [batch, num_kv_heads, 128] x 2            (fp8 variants) per-channel amax / v_range and its
//                                                          quantization multiplier v_range / amax; (fp16 variants,
//                                                          debug checks only) per-channel amax
//   stats   fp32 partial K sums and V amax per 512-key chunk (reduced deterministically, no atomics)
// No allocation and no host synchronization (except the debug checks); graph-capturable; bit-reproducible run to run.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mi::attention_q8 {

// Numerics version of every variant (quantization, tiling, accumulation order): bump on any change to the output bits,
// reported in the quality object's numerics (pipeline/quality_report.h).
inline constexpr std::uint32_t kNumericsVersion = 1;

enum class Variant { kInt8Fp16, kInt8Fp16F32Acc, kInt8Fp8, kInt8Fp8F32Acc };
const char* variant_name(Variant v);  // "int8_fp16", "int8_fp16_f32acc", "int8_fp8", "int8_fp8_f32acc"

struct Options {
  Variant variant = Variant::kInt8Fp8;
  // FP8 variants only: P is quantized to e4m3 in [0, p_range], V per channel to [-v_range, v_range]. <= 0 picks the
  // variant default: kInt8Fp8 (fp16 accumulation) 112 / 4.5 (SageAttention2++), kInt8Fp8F32Acc 448 / 448 (e4m3 max).
  // With fp16 accumulation the largest e4m3 values of P and V must keep a 64-key partial sum finite:
  // 64 * e4m3(2^p_offset) * e4m3(v_range) <= 65504, where p_offset = log2(p_range) truncated to 3 decimals is the
  // kernel's exponent offset (P's maximum) and e4m3() rounds to nearest even as the quantization does
  // (std::invalid_argument otherwise). E.g. 117 / 8.74 is refused: 64 * 120 * 9 = 69120.
  float p_range = 0;
  float v_range = 0;
};

// The ranges `o` resolves to (defaults filled in); fp16 variants return 0 / 0.
float effective_p_range(const Options& o);
float effective_v_range(const Options& o);

// Whether attention_q8 can run variant v on the current device with this build (the requirements above). The answer
// is cached per device and variant.
bool supported(Variant v);

std::size_t attention_q8_workspace_bytes(const Options& o, std::int64_t total_q, std::int64_t total_k, int batch,
                                         int max_seqlen, int num_heads, int num_kv_heads);

// out = softmax(q·kᵀ·scale)·v per sequence and head, with 8-bit operands as described above. Heads come from the
// tensor shapes (num_heads = q.dim(1), num_kv_heads = k.dim(1)), batch from cu_seqlens_q.dim(0) - 1.
// !! As flash_fwd: max_seqlen MUST be >= every sequence length, and cu_seqlens must start at 0, be non-decreasing and
// !! end at q.dim(0) / k.dim(0); NOT checked unless varlen_debug_checks() (core/varlen.h).
// workspace: device memory of at least attention_q8_workspace_bytes(...), 256-byte aligned.
// scale must be finite and > 0: the kernels take the row maximum of the unscaled scores and mask a partial key tile
// with a raw score of -5e6, so a scale <= 0 is wrong (unlike flash_fwd, which takes any finite scale), and a scale
// far below 1e-5 would let the masked keys back in.
// batch and num_heads must be <= 65535 (grid dimensions); with total_q == 0 nothing runs and workspace may be null.
// Throws std::invalid_argument for unsupported arguments (head_dim != 128, bad dtypes/strides, too many tokens),
// std::runtime_error when !supported(o.variant).
// MOSS's per-head RMS norm and RoPE of q and k (kernels::fused::qk_norm_rope_moss), run inside the pre-pass: attention_q8
// with norm_rope gives the bits of qk_norm_rope_moss on q's and k's columns followed by attention_q8 without it, in one
// pass over q, k and v instead of four. k is rewritten in place normed and roped, as qk_norm_rope_moss does; q keeps its
// input values (only its INT8 copy is normed and roped). q_norm, k_norm: contiguous bf16 [128]; table: bf16
// [total_q, 128] = [sin(64) | cos(64)] per token (kernels::moss_rope_table), row stride a multiple of 8, 16-byte
// aligned. Needs num_heads == num_kv_heads, cu_seqlens_k the same tensor as cu_seqlens_q, and q and k rows of one
// stride (the columns of a fused QKV projection output). The norm's mean factor is that of a run of total_q tokens.
struct QkNormRope {
  Tensor q_norm, k_norm, table;
};

void attention_q8(const Options& o, const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out,
                  const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, int max_seqlen, float scale, void* workspace,
                  std::size_t workspace_bytes, cudaStream_t stream, const QkNormRope* norm_rope = nullptr);

}  // namespace mi::attention_q8
