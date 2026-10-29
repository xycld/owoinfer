// Rotary position embeddings of the MOSS DiT (4-axis, half-split over 128-d heads) and the Qwen3.5 text
// encoder (partial 1-D rotary over the first 64 of 256 dims), bit-exact against torch eager.
//
// Tables are built on the GPU from fp32 phases (coordinate * frequency, sinf/cosf) cast to bf16, which
// is exactly what the references compute. The frequencies themselves are torch CPU constants (computed
// with CPU pow at model init; the GPU pow differs in the last bit for Qwen), so they are fixed here.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

#include "core/tensor.h"

namespace mi::kernels {

// _RoPE._default_bands for head_dim 128, base 1e4 (P:158-161): 1 / 1e4^(arange(16)/16), fp32 bits.
inline constexpr std::array<std::uint32_t, 16> kMossRopeBandBits = {
    0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2, 0x3d0186e3, 0x3c91ad39,
    0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528, 0x3a83126f, 0x3a136a16, 0x39a5cb60, 0x393a7753};

// Qwen3_5TextRotaryEmbedding inv_freq for rotary dim 64, theta 1e7 (Q:154-176): 1 / 1e7^(arange(0,64,2)/64).
inline constexpr std::array<std::uint32_t, 32> kQwenRopeInvFreqBits = {
    0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
    0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x3a092e02,
    0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52, 0x37d5c441, 0x37812dab, 0x371c1fc4,
    0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c, 0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44};

inline constexpr int kMossRopeDim = 128;
inline constexpr int kQwenRopeDim = 64;

// MOSS _RoPE.forward(height, width, text_length).to(bf16) (P:174-186, P:249): out is bf16
// [height*width + text_length, 128], image tokens first (coords r=0, f=text_length, centred h, w), then
// text (f = 0..T-1). Each row is [sin(64) | cos(64)] over (axis, band). The text refiner uses the text
// rows only (P:251): pass table.slice0(height*width, height*width + text_length) to moss_rope_apply.
void moss_rope_table(const Tensor& out, int height, int width, int text_length, cudaStream_t stream);

// The edit layout's reference tokens (Moss-Image-Edit _reference_inputs; ComfyUI MOSSImage.rope's ref_shapes): out is
// bf16 [height*width, 128] for reference `index` (0-based) on a height x width latent grid, at coordinates
// (r = index + 1, f = 0, centred h, w) with moss_rope_table's bands and arithmetic (the official fp32 bands, no NTK
// scaling). Its own kernel: moss_rope_table is unchanged.
void moss_rope_table_reference(const Tensor& out, int height, int width, int index, cudaStream_t stream);

// MOSS _rotate (P:77-82): out = x*cos + cat(-x2, x1)*sin in bf16, pairing dims (i, i+64).
// x and out: bf16 [..., L, heads, 128] (last dim contiguous); table: bf16 [L, 128]. out may alias x.
void moss_rope_apply(const Tensor& x, const Tensor& table, const Tensor& out, cudaStream_t stream);

// Qwen3_5TextRotaryEmbedding for text-only positions 0..S-1 (all three mRoPE components equal, so the
// interleaved recomposition is the identity): out is bf16 [S, 64] = [cos(32) | sin(32)], the unique half
// of transformers' cat(freqs, freqs) cos/sin.
void qwen_rope_table(const Tensor& out, cudaStream_t stream);

// apply_rotary_pos_emb (Q:660-695) with rotate_half over the first 64 dims, the rest passed through.
// x and out: bf16 [..., S, heads, head_dim >= 64]; table: bf16 [S, 64] from qwen_rope_table. out may alias x.
void qwen_rope_apply(const Tensor& x, const Tensor& table, const Tensor& out, cudaStream_t stream);

}  // namespace mi::kernels
