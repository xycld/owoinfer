// A standalone HIP port of the bf16 exact path of one MOSS DiT block (models/moss_dit/dit_block_fused.h):
//   y    = rms_norm_modulate(x, norm1, scale_a, shift_a)
//   qkv  = y · Wqkvᵀ
//   q,k  = rope(rms_norm(q|k))  per head, in place
//   attn = attention(q, k, v)
//   x   += bf16(bf16(attn · Woᵀ) * gate_a)
//   y    = rms_norm_modulate(x, norm2, scale_f, shift_f)
//   ffn  = situ(bf16(y · Wupᵀ), bf16(y · Wgateᵀ))
//   x   += bf16(bf16(ffn · Wdownᵀ) * gate_f)
// One kernel source (block_kernels.hip) with a swappable MMA leaf (mma_leaf.h). All kernels launch workgroups of 32
// threads: one wave on RDNA3/RDNA4, one warp on NVIDIA.
//
// All tensors are bf16, passed as their bits. Operands are 16-byte aligned with row strides that are multiples of 8
// elements, as the engine's: fragment loads are 128-bit on RDNA4. Launches are asynchronous on `stream`, with no host
// synchronization and no allocation.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace hip_block {

using bf16_bits = std::uint16_t;

// The DiT's shapes.
inline constexpr int kDim = 3072;
inline constexpr int kHeads = 24;
inline constexpr int kHeadDim = 128;
inline constexpr int kFfnDim = 8192;

// Key tokens per step of the online softmax. Part of the numerics: each step's weights are rounded to bf16 relative to
// the running maximum over the steps so far. 32 is FlashAttention's hdim128 128x32 tile, which the engine runs on sm_89.
inline constexpr int kAttentionStepN = 32;

// Which leaf this build runs, e.g. for a benchmark's output.
const char* leaf_name();

// out[rows, d] = bf16(bf16(bf16(rms_norm(x)) * w) * (1 + scale)) + shift); w, scale, shift: [d], d a multiple of 4.
// The fp32 mean of squares is summed in the order of torch's reduce kernel for >= 16 rows (kernels/basic/norm.cu: 32
// threads x 4 values, then a halving tree). That order is kept below 16 rows, where the engine's block width grows, so
// outputs there are not bit-exact against the engine: about 5 in 10^6 differ at 7 to 15 rows of 3072 (hip_block_verify).
void rms_norm_modulate(const bf16_bits* x, std::int64_t ldx, const bf16_bits* weight, const bf16_bits* scale,
                       const bf16_bits* shift, bf16_bits* out, std::int64_t ldo, std::int64_t rows, int d,
                       hipStream_t stream);

// out[m, n] = bf16(x[m, k] · w[n, k]ᵀ), with fp32 accumulators and K ascending.
void gemm(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w, std::int64_t ldw,
          int m, int n, int k, hipStream_t stream);

// In place, per head: q and k = rope(rms_norm(.)). qkv: [tokens, >= 2 * heads * 128] with q heads then k heads;
// q_norm, k_norm: [128]; table: [tokens, 128] RoPE rows, [sin(64) | cos(64)].
void qk_norm_rope(bf16_bits* qkv, std::int64_t ld, std::int64_t tokens, int heads, const bf16_bits* q_norm,
                  const bf16_bits* k_norm, const bf16_bits* table, std::int64_t table_ld, hipStream_t stream);

// Varlen scaled-dot-product attention, head_dim 128, non-causal, fp32 online softmax. Sequences are packed along the
// rows: sequence b is rows [cu_q[b], cu_q[b + 1]) of q and o and [cu_k[b], cu_k[b + 1]) of k and v. cu_q, cu_k: int32
// [batch + 1] device arrays. max_seqlen must be >= every query length (it sizes the grid; a shorter one drops rows).
// Rows hold [heads, 128] at the given head stride; num_kv_heads == heads. The log-sum-exp is not written.
struct AttentionArgs {
  const bf16_bits* q;
  const bf16_bits* k;
  const bf16_bits* v;
  bf16_bits* o;
  std::int64_t ldq, ldk, ldv, ldo;
  std::int64_t hsq, hsk, hsv, hso;
  const std::int32_t* cu_q;
  const std::int32_t* cu_k;
  int batch;
  int max_seqlen;
  int heads;
  float scale;  // SDPA's 1 / sqrt(head_dim) for the DiT
};
void attention(const AttentionArgs& a, hipStream_t stream);

// residual[m, n] += bf16(bf16(x[m, k] · w[n, k]ᵀ) * gate[n]), read and written in place.
void gemm_gated_residual(bf16_bits* residual, std::int64_t ldr, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w,
                         std::int64_t ldw, const bf16_bits* gate, int m, int n, int k, hipStream_t stream);

// out[m, f] = situ(bf16(x · hiddenᵀ), bf16(x · gateᵀ)) for the stored [hidden | gate] weight w_up_gate[2f, k].
void gemm_situ_glu(bf16_bits* out, std::int64_t ldo, const bf16_bits* x, std::int64_t ldx, const bf16_bits* w_up_gate,
                   std::int64_t ldw, int m, int f, int k, hipStream_t stream);

// One block's weights: norm1, norm2 [kDim]; q_norm, k_norm [128]; qkv [3 kDim, kDim]; o [kDim, kDim];
// up_gate [2 kFfnDim, kDim]; down [kDim, kFfnDim]. All contiguous.
struct BlockWeights {
  const bf16_bits* norm1;
  const bf16_bits* norm2;
  const bf16_bits* q_norm;
  const bf16_bits* k_norm;
  const bf16_bits* qkv;
  const bf16_bits* o;
  const bf16_bits* up_gate;
  const bf16_bits* down;
};

// One run of `tokens` tokens: x, y [tokens, kDim]; qkv [tokens, 3 kDim]; attn [tokens, kDim]; ffn [tokens, kFfnDim];
// rope [tokens, 128]; mod [6 kDim]: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f. All contiguous.
struct BlockBuffers {
  bf16_bits* x;
  bf16_bits* y;
  bf16_bits* qkv;
  bf16_bits* attn;
  bf16_bits* ffn;
  const bf16_bits* rope;
  const bf16_bits* mod;
  const std::int32_t* cu_seqlens;
  int batch;
  int max_seqlen;
  int tokens;
};

// The block's ops, in order, so a caller can time each one.
inline constexpr int kBlockOps = 8;
const char* block_op_name(int op);
void run_block_op(int op, const BlockWeights& w, const BlockBuffers& b, hipStream_t stream);
void run_block(const BlockWeights& w, const BlockBuffers& b, hipStream_t stream);

}  // namespace hip_block
