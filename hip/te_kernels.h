// The Qwen3.5 text encoder's kernels (te_kernels.hip), in the engine's op order and rounding points
// (src/models/qwen35_text/text_encoder.cpp, src/kernels/gdn, src/kernels/basic, the hdim-256 causal FlashAttention):
// the Qwen RMSNorms (plain with an optional text RoPE, and gated), the GatedDeltaNet's causal conv1d + SiLU, its gates
// and its chunked gated delta rule, causal GQA attention at head dim 256 with the output gate, the MLP's SiLU product,
// and the GEMMs on packed operands. Same conventions as hip_block.h: bf16 bits, every workgroup one wave of 32 threads,
// asynchronous on `stream`, no allocation. Sequences are packed along the rows; `cu` is an int32 device array of their
// row offsets (cu[0] = 0, cu[seqs] = tokens).
//
// On the nvidia build, where exp, log1p, rsqrt and division are CUDA's as in the engine, every kernel here except the
// GEMMs and attention gives the engine's bits on the engine's inputs (checked op by op over the 4096-token golden
// prompt). On AMD those functions are OCML's, which may differ from CUDA's in the last bit of fp32, so the same op
// order does not give the engine's bits by construction there. The GEMMs and attention sum in their MMA leaf's order
// (mma_leaf.h), never cuBLAS's or FlashAttention's.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "hip_block.h"

namespace hip_block::te {

inline constexpr int kGdnKeyHeads = 16;
inline constexpr int kGdnValueHeads = 32;  // value head h reads key head h / 2
inline constexpr int kGdnHeadDim = 128;
inline constexpr int kGdnKeyDim = kGdnKeyHeads * kGdnHeadDim;                   // 2048
inline constexpr int kGdnValueDim = kGdnValueHeads * kGdnHeadDim;               // 4096
inline constexpr int kGdnConvDim = 2 * kGdnKeyDim + kGdnValueDim;              // [q | k | v], 8192
inline constexpr int kGdnConvKernel = 4;
inline constexpr int kGdnChunk = 64;
inline constexpr int kAttnHeadDim = 256;
inline constexpr int kRopeDim = 64;  // the text RoPE rotates the first 64 dims of a head: pairs (c, c + 32)

// Key tokens per step of the attention's online softmax: FlashAttention-2's kBlockN for head dim 256 (the 64x64 and
// 128x64 tiles the engine runs). Part of the numerics, as hip_block::kAttentionStepN.
inline constexpr int kAttnStepN = 64;

// torch's threads per row of a last-dim fp32 reduction over `rows` rows of length d (setReduceConfig with 4-wide vector
// loads and at most 512 threads): 32 from 16 rows up, more below. Throws std::invalid_argument where torch would split a
// row across warps, which the norm does not reproduce, or for d not a multiple of 4 or below 128.
int reduce_threads(std::int64_t rows, int d);

// The text RoPE table of positions 0 .. rows - 1, bf16 rows [cos(32) | sin(32)]: phase = fp32(position) * inv_freq
// (Qwen3_5TextRotaryEmbedding's, theta 1e7, as torch's CPU computes it), cos and sin rounded to bf16, as the engine's
// qwen_rope_table computes it on the device (cosf, sinf). out: [rows, 64] contiguous.
void rope_table(bf16_bits* out, std::int64_t rows, hipStream_t stream);

// Rows are (token, head) pairs: row (t, h) of x starts at x + t * ldx + h * hsx, likewise out and gate. out may alias x.
//   gate == nullptr: Qwen3_5RMSNorm       out = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w))
//   gate != nullptr: Qwen3_5RMSNormGated  out = bf16(bf16(w * bf16(x * rsqrt(mean(x^2) + eps))) * silu(gate))
// rope (plain norm only, d >= 64): bf16 table rows [cos(32) | sin(32)] of the tokens, row t at rope + t * rope_ld; the
// norm's bf16 output is rotated in its first 64 dims as qwen_rope_apply, each product and the sum rounded to bf16.
// The mean of squares is summed in torch's reduce order for tokens * heads rows (reduce_threads), with its MeanOps factor.
struct NormArgs {
  const bf16_bits* x = nullptr;
  std::int64_t ldx = 0, hsx = 0;
  bf16_bits* out = nullptr;
  std::int64_t ldo = 0, hso = 0;
  const bf16_bits* weight = nullptr;
  const bf16_bits* gate = nullptr;
  std::int64_t ldg = 0, hsg = 0;
  const bf16_bits* rope = nullptr;
  std::int64_t rope_ld = 0;
  std::int64_t tokens = 0;
  int heads = 1;
  int d = 0;
  float eps = 1e-6f;
};
void rms_norm(const NormArgs& a, hipStream_t stream);

// y = bf16(silu(bf16(conv(x)))) over each sequence: the causal depthwise conv1d with kernel 4 (gdn::causal_conv1d_silu),
// fused multiply-adds over the taps ascending, rows before the sequence's first as zero. x: bf16 [tokens, 8192] of row
// stride ldx; w: bf16 [8192, 4]; y: bf16 [tokens, 8192] of stride ldy, not overlapping x.
void conv_silu(const bf16_bits* x, std::int64_t ldx, const bf16_bits* w, bf16_bits* y, std::int64_t ldy,
               const std::int32_t* cu, int seqs, std::int64_t tokens, hipStream_t stream);

// beta = bf16(sigmoid(b)); g = -exp(A_log) * softplus(a + dt_bias) in fp32, softplus threshold 20 (gdn::gates).
// ba: bf16 rows [b(32) | a(32)] of stride ldba; a_log, dt_bias: bf16 [32]; g: f32 [tokens, 32]; beta: bf16 [tokens, 32].
void gates(const bf16_bits* ba, std::int64_t ldba, const bf16_bits* a_log, const bf16_bits* dt_bias, float* g,
           bf16_bits* beta, std::int64_t tokens, hipStream_t stream);

// The chunked gated delta rule's fp32 scratch for `tokens` packed tokens (about 64 KiB per token).
struct GdnWorkspace {
  float* qn;    // [16][tokens][128] q * rsqrt(sum(q^2) + 1e-6) * 128^-0.5
  float* kn;    // [16][tokens][128] k * rsqrt(sum(k^2) + 1e-6)
  float* gcum;  // [32][tokens] cumulative log decay within the chunk
  float* eg;    // [32][tokens] exp(gcum)
  float* el;    // [32][tokens] exp(gcum[last of chunk] - gcum)
  float* a;     // [32][tokens][64] the UT system's strict lower part, beta_i k_i.k_j exp(gcum_i - gcum_j)
  float* attn;  // [32][tokens][64] (q_i.k_j) exp(gcum_i - gcum_j) for j <= i, 0 above
  float* u;     // [32][tokens][128] T^-1 (beta v)
  float* w;     // [32][tokens][128] T^-1 (beta k exp(gcum))
  static std::size_t bytes(std::int64_t tokens);
  static GdnWorkspace at(void* base, std::int64_t tokens);  // base 256-byte aligned
};

// The chunked gated delta rule (chunk 64), fp32 throughout, each sequence from a zero state, as gdn::chunk_gated_delta_rule
// (fused multiply-adds in its orders): q and k L2-normalised with q scaled by 128^-0.5, the UT system solved by forward
// substitution, then the scan over the chunks. qkv: the conv output, bf16 [tokens, 8192] of row stride ld ([q | k | v]);
// g: f32 [tokens, 32]; beta: bf16 [tokens, 32]; out: bf16 [tokens, 32, 128] contiguous. max_len >= every sequence's
// length (it sizes the grid).
void gated_delta_rule(const bf16_bits* qkv, std::int64_t ld, const float* g, const bf16_bits* beta,
                      const std::int32_t* cu, int seqs, int max_len, std::int64_t tokens, bf16_bits* out,
                      const GdnWorkspace& ws, hipStream_t stream);

// Causal scaled-dot-product attention, head dim 256, query head h reading key/value head h / kv_group, fp32 online
// softmax over key steps of kAttnStepN walked from the last down (FlashAttention-2's order and rounding points, as
// hip_block::attention), sequences packed along the rows (cu). Rows hold [heads, 256] at the given head strides. With
// `gate`: out = bf16(bf16(attention) * bf16(sigmoid(gate))) (Qwen3_5Attention's output gate, sigmoid_mul), gate row t at
// gate + t * ldg + h * hsg.
struct AttentionArgs {
  const bf16_bits* q = nullptr;
  const bf16_bits* k = nullptr;
  const bf16_bits* v = nullptr;
  bf16_bits* o = nullptr;
  const bf16_bits* gate = nullptr;
  std::int64_t ldq = 0, ldk = 0, ldv = 0, ldo = 0, ldg = 0;
  std::int64_t hsq = 0, hsk = 0, hsv = 0, hso = 0, hsg = 0;
  const std::int32_t* cu = nullptr;
  int batch = 0;
  int max_seqlen = 0;
  int heads = 0;
  int kv_group = 1;
  float scale = 0.0625f;  // head_dim ** -0.5
};
void causal_attention(const AttentionArgs& a, hipStream_t stream);

// out[r, c] = bf16(bf16(silu(gu[r, c])) * gu[r, cols + c]) for c < cols (Qwen3_5MLP's silu(gate) * up over the
// [gate | up] GEMM output). cols a multiple of 8.
void silu_mul(const bf16_bits* gu, std::int64_t ldg, bf16_bits* out, std::int64_t ldo, std::int64_t rows, int cols,
              hipStream_t stream);

// The GEMMs on pack_fragments' layout (hip_block.h), gemm_tile.h's kernel with fp32 accumulators and K ascending: the same
// bits as hip_block::gemm on the row-major operands. xp: m rows packed with pack_rows(m) rows; wp: n rows packed with
// pack_rows(n) rows; k a multiple of 16.
std::int64_t pack_rows(std::int64_t rows);
//   out[m, n] = bf16(x · wᵀ)
void gemm_packed(bf16_bits* out, std::int64_t ldo, const bf16_bits* xp, const bf16_bits* wp, int m, int n, int k,
                 hipStream_t stream);
//   residual[m, n] = bf16(residual + bf16(x · wᵀ)), through the gated-residual epilogue with `ones` (bf16 1.0 [n]):
//   bf16(r + bf16(bf16(ab) * 1)) is bf16(r + bf16(ab)).
void gemm_packed_residual(bf16_bits* residual, std::int64_t ldr, const bf16_bits* xp, const bf16_bits* wp,
                          const bf16_bits* ones, int m, int n, int k, hipStream_t stream);

}  // namespace hip_block::te
