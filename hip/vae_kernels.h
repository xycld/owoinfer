// The FLUX.2 VAE decoder's kernels (vae_kernels.hip), at the engine's reference precision (DecodeMode::reference(),
// kernels/vae/vae_kernels.h): fp32 NHWC activations, convolutions on TF32-rounded operands with fp32 accumulation,
// fp32-exact attention linears, fp32-accurate attention, GroupNorm statistics from exact integer sums.
//
// The matrix products run on the backend's 16x16x16 bf16 leaf (mma_leaf.h), which multiplies bf16 values exactly into
// fp32 accumulators. An fp32 operand becomes bf16 pieces whose sum is the operand: a TF32 value (11 significant bits) is
// exactly 2 pieces, an fp32 value (24 bits) exactly 3. A product of two operands is the sum of the products of their
// pieces, each exact in fp32; which of them a kernel adds is its Products value below. Within one 16-wide K step the
// products are added B piece by B piece from the smallest piece, and for each the A pieces from the smallest.
// An infinite or NaN operand is its piece 0 alone: a product with it is non-finite, NaN where a piece of the other
// operand is 0 or has the other sign, so an infinity in an input gives ±Inf or NaN outputs, never finite ones.
//
// Packed pieces: a [rows, cols] operand as `pieces` bf16 arrays, each [rows_pad, cols_pad] (both multiples of 16) of
// contiguous 16 x 16 fragments: element (r, c) of piece t at
//   t * rows_pad * cols_pad + (r / 16 * (cols_pad / 16) + c / 16) * 256 + r % 16 * 16 + c % 16.
// Elements past rows or cols are 0. A fragment load then reads 512 contiguous bytes.
//
// Every kernel launches workgroups of one wave (32 threads). Launches are asynchronous on `stream`; nothing allocates.
// Pointers are 16-byte aligned.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace hip_vae {

// Rows and columns of a wave's output tile: kTileM x kConvTileN for the convolutions, kTileM x kTileN for piece_gemm.
// Packed weights pad their output channels to kConvTileN.
inline constexpr int kTileM = 32;
inline constexpr int kConvTileN = 48;
inline constexpr int kTileN = 96;
inline constexpr int kGroups = 32;  // GroupNorm groups

// Which piece products a convolution adds (piece indices i of A, j of B):
//   kTf32       2 x 2 pieces, all 4 products: the exact product of the TF32-rounded operands (cuDNN's TF32 convolution)
//   kTf32Drop   2 x 2 pieces without lo x lo (i + j < 2): each product short by at most 2^-16 of |a b|
//   kFp32       3 x 3 pieces, all 9 products: the exact product of fp32 operands (a strict fp32 GEMM)
enum class Products { kTf32, kTf32Drop, kFp32 };
inline int pieces_of(Products p) { return p == Products::kFp32 ? 3 : 2; }
const char* products_name(Products p);

inline std::int64_t round_up(std::int64_t v, std::int64_t m) { return (v + m - 1) / m * m; }
inline std::int64_t packed_index(std::int64_t r, std::int64_t c, std::int64_t cols_pad) {
  return (r / 16 * (cols_pad / 16) + c / 16) * 256 + r % 16 * 16 + c % 16;
}

// Host versions of the device's roundings, for packing weights and for the checks' replays.
inline float host_tf32(float v) {  // round half away from zero to 10 mantissa bits (cuDNN, cvt.rna); inf and NaN pass
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return v;
  u = (u + 0x1000u) & 0xffffe000u;
  std::memcpy(&v, &u, 4);
  return v;
}
inline std::uint16_t host_bf16_bits(float v) {  // round to nearest even; a NaN stays a (quiet) NaN, as the device's bf()
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float host_bf16_value(std::uint16_t b) {
  const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
  float v = 0.0f;
  std::memcpy(&v, &u, 4);
  return v;
}
// v (TF32-rounded first for TF32 products) as pieces_of(p) bf16 pieces, each the nearest bf16 of what the previous left.
// An infinity or NaN is piece 0, the other pieces 0.
inline void host_split(float v, Products p, std::uint16_t* piece) {
  float r = p == Products::kFp32 ? v : host_tf32(v);
  const bool finite = std::isfinite(v);
  for (int t = 0; t < pieces_of(p); ++t) {
    piece[t] = t == 0 || finite ? host_bf16_bits(r) : std::uint16_t{0};
    r = r - host_bf16_value(piece[t]);
  }
}

// ---- Convolution and linear ------------------------------------------------------------------------------------------

// out[m, n] = (sum_k A[m, k] W[n, k] + bias[n]) + residual[m, n] in fp32, each + rounded once, for the h * w output
// pixels m and the n output channels. A is the im2col of x: taps 1, a 1x1 convolution (or a linear over h * w rows);
// taps 9, a 3x3 convolution with zero padding 1 over an h x w output, k = (r * 3 + s) * c + channel, and with `upsample`
// x is h/2 x w/2 read through a nearest 2x upsample. x: NHWC fp32 [pixels, c], c a multiple of 16. w: the weight
// [n, taps c] (KRSC) as pack_weight's pieces for `products` (kTf32Drop reads kTf32's), rows_pad = round_up(n, kConvTileN).
// bias, residual: null for none; residual may equal out, out must not overlap x. ldo a multiple of 4.
struct ConvArgs {
  const float* x = nullptr;
  const std::uint16_t* w = nullptr;
  const float* bias = nullptr;
  const float* residual = nullptr;
  float* out = nullptr;
  std::int64_t ldo = 0;
  int h = 0, w_out = 0, c = 0, n = 0;
  int taps = 0;
  bool upsample = false;
};
void conv(const ConvArgs& a, Products products, hipStream_t stream);

// The packed pieces of a weight [n, k] fp32 row-major (k a multiple of 16): pieces * round_up(n, kConvTileN) * k bf16.
std::size_t packed_weight_elems(int n, int k, Products products);
void pack_weight(const float* w, int n, int k, Products products, std::uint16_t* out);

// conv_out: a 3x3 convolution (zero padding 1) to n <= 4 channels in fp32 with the TF32-rounded operands' exact products,
// one output pixel per thread: out[m, n] = sum_k + bias[n]. w_tf32: [n, 9 c] KRSC, already TF32-rounded (host_tf32);
// c a multiple of 4; out [h w, n].
void conv_narrow(const float* x, const float* w_tf32, const float* bias, float* out, int h, int w, int c, int n,
                 hipStream_t stream);

// ---- Attention ------------------------------------------------------------------------------------------------------

// src [rows, cols] fp32 (element (r, c) = src[r * ld + c], or src[c * ld + r] when `transpose`) -> 3 packed pieces of
// [rows_pad, cols_pad] (multiples of 16) in dst, zeros past rows and cols.
void split_pack(const float* src, std::int64_t ld, bool transpose, int rows, int cols, std::uint16_t* dst, int rows_pad,
                int cols_pad, hipStream_t stream);

// out[m, n] = sum_k A[m, k] B[n, k] over 3-piece packed operands with the products i + j <= 2 (fp32-accurate: every
// dropped product is at most 2^-24 |a b|), fp32, for m < m_rows and n < n_cols. a: [a_rows_pad, k_pad] packed, piece
// stride a_stride elements, a_rows_pad >= round_up(m_rows, kTileM); b likewise with b_rows_pad >= round_up(n_cols,
// kTileN). ldo a multiple of 4.
void piece_gemm(const std::uint16_t* a, std::int64_t a_stride, const std::uint16_t* b, std::int64_t b_stride, int k_pad,
                float* out, std::int64_t ldo, int m_rows, int n_cols, hipStream_t stream);

// Row softmax of attention scores into packed P pieces: p(r, j) = exp(s[r, j] * scale - max_j(s[r, :]) * scale) /
// sum_j(...) for j < l, as 3 packed pieces of [rows_pad, l_pad] at piece stride p_stride (rows r < rows written; columns
// past l written as 0). s: [rows, lds] fp32.
void softmax_pack(const float* s, std::int64_t lds, int rows, int l, float scale, std::uint16_t* p, std::int64_t p_stride,
                  int l_pad, hipStream_t stream);

// ---- GroupNorm ------------------------------------------------------------------------------------------------------

// 32 groups over x [pixels, c] fp32 NHWC (c a multiple of 32, <= 1024): the engine's exact statistics (vae::group_norm):
// each x and x^2 rounded once onto fixed-point grids (2^-64, 2^-96) and summed in integers, then mean and variance in
// fp64, rounded to stats[g] = {mean, rstd} fp32 with rstd = 1 / sqrt(var + eps). |x| < 2^33, else that group's stats are
// NaN. accum: gn_accum_bytes(), zeroed by this call.
std::size_t gn_accum_bytes();
void group_norm_stats(const float* x, std::int64_t pixels, int c, float eps, void* accum, float* stats, hipStream_t stream);
// y = fma(a, x, fma(-a, mean, beta)) with a = rstd * gamma (torch's fused GroupNorm parameters); with silu,
// y / (1 + exp(-y)). y may be x.
void group_norm_apply(const float* x, float* y, std::int64_t pixels, int c, const float* stats, const float* gamma,
                      const float* beta, bool silu, hipStream_t stream);

// ---- Pipeline ends ----------------------------------------------------------------------------------------------------

// z [128, h, w] fp32 NCHW -> y [2h, 2w, 32] fp32 NHWC: y[2i + p, 2j + q, c] = z[4c + 2p + q, i, j] * scale + mean of that
// latent channel, each op rounded (scale = sqrt(var + eps), formed on the host).
void latent_unpack(const float* z, const float* scale, const float* mean, float* y, int h, int w, hipStream_t stream);

// x [n] fp32 -> rgb [n] bytes as the pipeline: clamp(-1, 1), (x * 0.5 + 0.5).clamp(0, 1), (x * 255) rounded half to
// even; NaN gives 0.
void rgb8(const float* x, std::uint8_t* rgb, std::int64_t n, hipStream_t stream);

}  // namespace hip_vae
