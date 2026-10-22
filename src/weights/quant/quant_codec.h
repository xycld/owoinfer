// CPU math of the INT8 ConvRot weight format (ComfyUI / comfy-kitchen "int8_tensorwise" + convrot) and the exact
// decoders of the community W4A8 / W6A8 / W4A4 storage into that format.
//
// ConvRot: W_rot = W . H_G per contiguous G-column group along K, with H_G = kron^n(H4) / sqrt(G) the *regular*
// Hadamard (H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]), symmetric and orthonormal. Then per row
// s = absmax(W_rot) / 127 and q = round_half_even(W_rot / s) clamped to [-128, 127] (a rounded |W_rot / s| never
// passes 127, so -128 never appears in practice).
//
// Bit-exactness target: Comfy-Org's published files, which equal comfy-kitchen's quantize_int8_convrot_weight run
// on the fp32 weight with PyTorch on CUDA. Two details of that path are reproduced on purpose:
//   - the rotation is one fp32 GEMM whose cuBLAS kernel accumulates each output in k order from 0 (the products
//     w * (+-1/sqrt(G)) are exact), so every output is a sequential fp32 sum in column order;
//   - `absmax / 127.0` divides by a CPU scalar, which PyTorch's CUDA kernel computes as absmax * (1.0f / 127.0f).
// The quotient W_rot / s is a true IEEE fp32 division.
#pragma once

#include <cstddef>
#include <cstdint>

// mi::quant rather than mi::weights: the format is shared by the reader, the converter and the GEMM kernels.
namespace mi::quant {

// True for the ConvRot group sizes supported: powers of 4 from 4 to 1024 (real files use 16, 64 and 256).
bool valid_convrot_group(std::int64_t g);

// Entry (i, j) of the unnormalized regular Hadamard kron^n(H4): +1 or -1. g must be valid_convrot_group.
int hadamard_sign(int g, int i, int j);

// IEEE bf16 <-> fp32 (round to nearest even; NaN stays NaN).
float bf16_to_float(std::uint16_t v);
std::uint16_t float_to_bf16(float v);
// OCP float8 e4m3fn (bias 7, no infinities, 0x7F / 0xFF are NaN).
float e4m3_to_float(std::uint8_t v);

// W_rot of `rows` rows of a row-major [rows, k] fp32 matrix (see the file comment for the summation order).
// k % g == 0. out may not alias w.
void convrot_rows(const float* w, std::int64_t rows, std::int64_t k, int g, float* out);

// Per-row INT8 quantization of already-rotated rows: scale[r] = max(absmax * (1/127), 1e-30), q = rne(x / scale).
void quantize_rows_int8(const float* w_rot, std::int64_t rows, std::int64_t k, std::int8_t* q, float* scale);

// convrot_rows + quantize_rows_int8 on one row at a time (no [rows, k] fp32 temporary). Inputs must be finite;
// returns false (leaving the row's outputs unspecified) for a row holding a NaN or infinity.
bool quantize_convrot_rows(const float* w, std::int64_t rows, std::int64_t k, int g, std::int8_t* q, float* scale);
bool quantize_convrot_rows_bf16(const std::uint16_t* w, std::int64_t rows, std::int64_t k, int g, std::int8_t* q,
                                float* scale);

// Original-basis weights of quantized ConvRot rows: (q * scale) . H_G per group (H_G is its own inverse), with
// convrot_rows' summation order, which reproduces comfy-kitchen's dequantize_int8_convrot_weight on CUDA bit for bit.
// g == 0: no rotation. No heap allocation.
void dequantize_rows(const std::int8_t* q, const float* scale, std::int64_t rows, std::int64_t k, int g, float* out);

// ---- Community storage -> INT8 grid (exact; comfy-kitchen eager _dequant_int4_grouped_to_int8) ----------------

// asym_w4a8_int8 / w6a8_int8 rows. `packed` holds rows of k * bits / 8 bytes: a k/2-byte nibble plane (even column
// = low nibble), then for bits == 6 a k/4-byte plane of each code's top 2 bits (column c: byte c / 4, bit
// 2 * (c % 4)). Level = codebook[code] when `codebook` (16 entries, bits == 4) is given, else code - 2^(bits-1).
// out = clamp(rne(level * s_rel[r, c / group]), -127, 127), with s_rel e4m3 [rows, k / group].
// Each group scale is decoded once. Returns false if a group scale is NaN or negative (0 is valid: comfy-kitchen's
// e4m3 cast flushes tiny relative scales to 0, and such a group decodes to zeros).
bool decode_grouped_rows(const std::uint8_t* packed, const std::uint8_t* s_rel_e4m3, const float* codebook,
                         std::int64_t rows, std::int64_t k, int bits, int group, std::int8_t* out);

// convrot_w4a4 rows of k / 2 bytes: signed nibbles (-8..7), even column = low nibble. Lossless.
void decode_int4_rows(const std::uint8_t* packed, std::int64_t rows, std::int64_t k, std::int8_t* out);

// True when every per-row scale is finite and > 0, as comfy-kitchen's quantizers write them (int8: clamp(min=1e-30);
// W4A8 s_channel: clamp(min=1e-8); W4A4: absmax clamp(min=1e-10) / 7).
bool valid_scales(const float* s, std::int64_t n);

// Runs fn(begin, end) over [0, n) split into contiguous chunks on up to `threads` threads (0 = hardware
// concurrency); rethrows the first exception.
void parallel_rows(std::int64_t n, int threads, const void* ctx, void (*fn)(const void* ctx, std::int64_t begin, std::int64_t end));

template <typename F>
void parallel_for(std::int64_t n, int threads, const F& f) {
  parallel_rows(n, threads, &f, [](const void* ctx, std::int64_t b, std::int64_t e) { (*static_cast<const F*>(ctx))(b, e); });
}

}  // namespace mi::quant
