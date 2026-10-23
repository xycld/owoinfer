// One MOSS DiT block (_Block.forward, P:136-148) on the quantized compute path (P6 steps 36 and 40): each linear as
// the loaded file stores it, INT8 ConvRot (G = 256, High tier), INT4 ConvRot (G = 256, Turbo tier) or bf16, plus a
// choice of attention kernel.
//   INT8 linear: its input is rotated and quantized per token where it is produced, then an INT8 x INT8 -> INT32
//   GEMM with the dequantization and the following elementwise op in its epilogue (kernels/quant):
//     qkv:      rms_norm_modulate_rotate_quantize(x, norm1, scale_a, shift_a) -> int8_gemm
//     o:        rotate_quantize(attn)                                       -> int8_gemm_gated_residual (gate_a)
//     up_gate:  rms_norm_modulate_rotate_quantize(x, norm2, scale_f, shift_f) -> int8_gemm_situ_glu
//     down:     rotate_quantize(ffn)                                        -> int8_gemm_gated_residual (gate_f)
//   INT4 linear: the same sites and epilogues with kernels/quant4 (rotate + per-token INT4 quantize, then the
//   INT4 x INT4 -> INT32 GEMM): rms_norm_modulate_convrot_quantize_int4 -> int4_gemm / int4_gemm_situ_glu,
//   convrot_quantize_int4 -> int4_gemm_gated_residual. Numerics: comfy-kitchen's convrot_w4a4_linear (eager
//   activation quantization, CUDA epilogue order; see kernels/quant4/quant4.h).
//   bf16 linear: exactly what run_block_fused does for it (cuBLASLt QKV; o / up_gate / down fused or op by op as the
//   route says), so a file that keeps some linears bf16 runs them as the Reference tier does.
//   QK-norm + RoPE: the fused kernel (kernels/fused/qk_norm_rope.h), as the bf16 block.
//   Runtime LoRA (P6 step 42): an active side-branch site adds Δ = Σ s_i B_i (A_i x) inside its GEMM (kernels/lora).
//   An INT4 site runs the INT8 LoRA GEMM on its operands unpacked to INT8 (unpack_int4): the int32 dot products and
//   the epilogue are the INT4 GEMM's, so the base result is bit for bit the plain INT4 one (at INT8 GEMM speed).
//   On GPUs without INT4 MMA (int4_mma_native(); QuantBlockBuffers::int4_via_int8) every INT4 site runs that way,
//   with the plain INT8 GEMMs: the same bits as the INT4 GEMMs, at the INT8 rate.
//   Attention: exact (attention::flash_fwd) or 8-bit (attention_q8, any of its variants; SageAttention2++ by default).
// The INT8 path's numerics are those of comfy-kitchen's int8_linear with ConvRot (see kernels/quant/quant.h): the
// rotation in the CUDA kernel's fp32 order, the bf16-rounded rotated row, per-token absmax/127 scales, round half to
// even, exact int32 accumulation and the ((acc * s_x) * s_w) epilogue order. The fused norm kernel's mean of squares is
// summed in torch's order for the run's row count: it is called once per run, with the rows the reference norm sees.
// With every linear bf16 and exact attention the block equals run_block_fused bit for bit.
// No host synchronization and no allocation (graph-capturable); attention_q8's debug checks (varlen_debug_checks())
// are the exception.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "core/tensor.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/lora/side_branch.h"
#include "models/moss_dit/dit_weights.h"

namespace mi::moss_dit {

enum class AttentionMode {
  kExact,         // attention::flash_fwd (FA2), the Reference numerics
  // attention_q8 (lossy, opt-in), one per attention_q8::Variant: INT8 QK^T with K smoothing, then P·V in
  kQ8Fp16,        //   fp16, fp16 accumulation per 64-key tile flushed into fp32 (SageAttention2 fp16+fp32)
  kQ8Fp16F32Acc,  //   fp16, fp32 accumulation
  kQ8Fp8,         //   e4m3, fp16 accumulation per tile flushed into fp32 (SageAttention2++; the High tier's default)
  kQ8Fp8F32Acc,   //   e4m3, fp32 accumulation
};
// "exact", "q8_fp16", "q8_fp16_f32acc", "q8_fp8", "q8_fp8_f32acc" (the quality certificates' tokens).
const char* attention_mode_name(AttentionMode m);
// One of those names, or "q8" (= q8_fp8); throws std::invalid_argument otherwise.
AttentionMode parse_attention_mode(std::string_view name);
// attention_q8 can run mode `m` on the current device with this build (true for kExact).
bool attention_supported(AttentionMode m);

// Activations of one forward for the quantized block: the fused block's buffers plus scales and attention workspace.
struct QuantBlockBuffers {
  FusedBlockBuffers f;  // x, y, big (sized as quant_big_cols), rope, mod, lse
  Tensor scale;         // f32 [L]: per-token activation scales of the GEMM input being quantized
  void* attn_workspace = nullptr;  // 8-bit modes: attention_q8 workspace of q8_attention_workspace_bytes()
  std::size_t attn_workspace_bytes = 0;
  // Runtime LoRA side branches (models/lora/side_branch.h; nullptr: none): t bf16 [L, >= lora->max_rank()] and the
  // t32 workspace (lora::t_workspace_bytes(L, max_rank)).
  const lora::SideBranches* lora = nullptr;
  Tensor lora_t;
  void* lora_workspace = nullptr;
  std::size_t lora_workspace_bytes = 0;
  // INT4 linears through the INT8 GEMMs on their operands unpacked to INT8 (bit for bit the INT4 GEMMs' results): every
  // INT4 linear when int4_via_int8 (GPUs without INT4 MMA, int4_mma_native()), else those with an active LoRA site.
  // Their weight is unpacked here: kInt4UnpackBytes, 16-byte aligned.
  bool int4_via_int8 = false;
  void* int4_unpack = nullptr;
  std::size_t int4_unpack_bytes = 0;
};

// Whether INT4 tensor-core MMA (mma.sync .s4, the kernels/quant4 GEMMs) runs natively: compute capability 8.x (Ampere,
// Ada). Hopper (sm_90) and Blackwell (sm_100, sm_120) have no INT4 tensor-core path and ptxas emulates it (sm_120,
// measured: 91.7 TOPS INT4 against 620-640 TOPS INT8), so the INT4 linears run the INT8 GEMMs there.
bool int4_mma_native(int major, int minor);
bool int4_mma_native();  // the current device (true when it cannot be queried)

// Bytes of the largest block weight unpacked to INT8 (ffn_up_gate, [2 * 8192, 3072]).
inline constexpr std::size_t kInt4UnpackBytes = std::size_t{2} * 8192 * 3072;

// Big-buffer elements per token for the quantized block over rows of `hw + text_lengths[r]` tokens: kFusedBigCols,
// unless some block runs ffn_up_gate bf16 (`bf16_up_gate`) and the route gives a run of these rows the op-by-op
// [up_gate | ffn] layout (then fused_big_cols' kOpByOpBigCols). The INT8 ffn_down input ([L, kFfnDim] int8; INT4:
// packed [L, kFfnDim / 2]) lives in the part of `big` the FFN does not use.
std::int64_t quant_big_cols(std::int64_t hw, std::span<const std::int64_t> text_lengths, bool bf16_up_gate,
                            const FusedRouteSnapshot& route);

// attention_q8 workspace bytes enough for any run of any subset of these rows (0 for kExact).
std::size_t q8_attention_workspace_bytes(std::int64_t hw, std::span<const std::int64_t> text_lengths, AttentionMode mode);

// The block on run `run` of `a` with weights `w` (any mix of bf16, INT8 and INT4 linears, dit_weights.h), bf16 pairs routed
// by `route` as run_block_fused does. Operands 256-byte aligned (arena allocations, kArenaAlign weights). Buffers that
// do not match (big smaller than quant_big_cols per token, no scale buffer, a q8 workspace too small for the run) are a
// programming error: std::invalid_argument before any launch.
// `block`: the block's index, for its LoRA sites (a.lora). A site with an active side branch runs the fused LoRA GEMM of
// kernels/lora (bf16 or INT8 base, whatever the route says), its t computed from the site's bf16 input; for an INT8
// qkv / ffn_up_gate site that input (the norm output) is materialized and rotated separately, bit-identical to the
// fused norm + rotate kernel. A block whose q/k/v differ in precision takes no qkv side branch (std::invalid_argument).
// An active site on an INT4 linear needs the int4_unpack buffer (std::invalid_argument otherwise).
void run_block_quant(const BlockTensors& w, const QuantBlockBuffers& a, const BlockRun& run, const QkvGemm& qkv_gemm,
                     const FusedRouteSnapshot& route, AttentionMode attention, cudaStream_t stream, int block = 0);

}  // namespace mi::moss_dit
