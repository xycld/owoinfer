// CPU math of the NVFP4 weight format (ComfyUI / comfy-kitchen comfy_quant "nvfp4"; design: NVFP4 plan §2.1, §4.1).
//
// W ~= s2 * sf[n, kb] * q[n, k]:
//   q   E2M1 codes (magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 = sign), two per byte, hi_first: element 2j in the
//       HIGH nibble of byte j (the opposite of CUTLASS/PTX; both GEMM operands share the order, so the block dot
//       products are unchanged and file bytes are used as stored);
//   sf  UE4M3 (float8 e4m3fn bits, sign 0) per 16 consecutive elements along K, in the cuBLAS "1D block scaling" /
//       CUTLASS Sm1xxBlkScaledConfig swizzle: [round_up(rows, 128), round_up(K/16, 4)] bytes, padding zero;
//   s2  fp32 per tensor (weight_scale_2). The engine's activations use one fp32 scale per row instead (g).
//
// Quantization (fp32, IEEE division), per row r with second-level scale g_r (= s2 for a weight):
//   sf = e4m3_rne(min((amax_b / 6) / g_r, 448));  q = e2m1_rne(clamp(x / (g_r * sf), -6, 6)), +0 where g_r * sf == 0.
// With every g_r = s2 this is comfy-kitchen's eager quantize_nvfp4 bit for bit (oracle: tools/reference/nvfp4_ref.py).
#pragma once

#include <cstddef>
#include <cstdint>

// mi::quant rather than mi::weights: the format is shared by the reader, the converter and the GEMM kernels.
namespace mi::quant {

inline constexpr int kNvfp4Block = 16;           // elements per block scale
inline constexpr float kNvfp4E2M1Max = 6.0f;
inline constexpr float kNvfp4E4M3Max = 448.0f;
inline constexpr float kNvfp4TensorScaleDiv = 2688.0f;  // 6 * 448: s2 = amax / 2688

// fp32 -> E2M1 code, round to nearest even, saturating at +-6. The sign bit follows the fp32 sign bit (so -0.0 and
// small negatives give 0x8, as an IEEE conversion does); NaN gives magnitude 0.
std::uint8_t e2m1_rne(float x);
float e2m1_to_float(std::uint8_t code);  // low 4 bits
// Non-negative fp32 -> float8 e4m3fn bits, round to nearest even (subnormals included), saturating at 448 (0x7E);
// values <= 0 give 0, NaN gives 0x7F.
std::uint8_t e4m3_rne(float x);

// Swizzled scale layout. sfk = K / 16 scales per row.
std::int64_t nvfp4_scale_rows(std::int64_t rows);  // round_up(rows, 128)
std::int64_t nvfp4_scale_cols(std::int64_t sfk);   // round_up(sfk, 4)
// Byte offset of the scale of row m, K-block kb:
//   ((m / 128) * ceil(sfk / 4) + kb / 4) * 512 + (m % 32) * 16 + ((m % 128) / 32) * 4 + kb % 4.
std::int64_t nvfp4_scale_offset(std::int64_t m, std::int64_t kb, std::int64_t sfk);
// Row-major [rows, sfk] <-> swizzled [nvfp4_scale_rows(rows), nvfp4_scale_cols(sfk)] (swizzle zero-fills padding).
void swizzle_scales(const std::uint8_t* row_major, std::int64_t rows, std::int64_t sfk, std::uint8_t* swizzled);
void unswizzle_scales(const std::uint8_t* swizzled, std::int64_t rows, std::int64_t sfk, std::uint8_t* row_major);

// Quantizes `rows` rows of a row-major fp32 [rows, k] matrix (k % 16 == 0) with second-level scales g[r] (g_stride 0:
// one scale g[0] for every row). Writes packed [rows, k/2] hi_first bytes and ROW-MAJOR scales [rows, k/16].
// Inputs must be finite and g > 0 finite.
void quantize_nvfp4_rows(const float* x, std::int64_t rows, std::int64_t k, const float* g, std::int64_t g_stride,
                         std::uint8_t* packed, std::uint8_t* scales);

// A whole weight into the file layout: packed [n, k/2] and swizzled scales
// [nvfp4_scale_rows(n), nvfp4_scale_cols(k/16)]. s2 is the caller's tensor scale (nvfp4_tensor_scale: the fp32 recipe).
void quantize_nvfp4_weight(const float* w, std::int64_t n, std::int64_t k, float s2, std::uint8_t* packed,
                           std::uint8_t* swizzled);
float nvfp4_tensor_scale(float amax);  // amax / 2688 in fp32

// The engine's activation recipe on bf16 rows (reference for the GPU quantizers): g_r = amax_r / 2688 (amax_r == 0:
// g_r = 1), NaN elements count as 0 and quantize to +0, +-inf is clamped to +-bf16 max first. scales_swizzled holds
// nvfp4_scale_rows(rows) x nvfp4_scale_cols(k/16) bytes and is fully written (padding zero).
void quantize_nvfp4_act_bf16(const std::uint16_t* x, std::int64_t rows, std::int64_t k, std::uint8_t* packed,
                             std::uint8_t* scales_swizzled, float* g);

// Rows of a weight in fp32: out = e2m1(q) * (s2 * sf) with s2 * sf rounded to fp32 once (comfy-kitchen's
// dequantize_nvfp4 with output dtype fp32). `packed` points at the first row ([rows, k/2]); `swizzled` is the whole
// scale buffer of a matrix whose row `row0` is packed's first row.
void dequantize_nvfp4_rows(const std::uint8_t* packed, const std::uint8_t* swizzled, std::int64_t row0, std::int64_t rows,
                           std::int64_t k, float s2, float* out);
// The same in bf16 (bits), in comfy-kitchen's bf16 order: total = bf16(bf16(s2) * sf), out = bf16(e2m1(q) * total).
// This is what ComfyUI computes when it dequantizes an nvfp4 layer into a bf16 model.
void dequantize_nvfp4_rows_bf16(const std::uint8_t* packed, const std::uint8_t* swizzled, std::int64_t row0, std::int64_t rows,
                                std::int64_t k, float s2, std::uint16_t* out);

// Why the stored scales of an n x (16 * sfk) weight are invalid, or nullptr when they are valid: every used byte
// must be a non-NaN (not 0x7F / 0xFF), non-negative (bit 7 clear) e4m3 value, and every padding byte (rows n.. and
// columns sfk.. of the partial tiles, which CUTLASS reads) must be zero.
const char* nvfp4_scales_problem(const std::uint8_t* swizzled, std::int64_t n, std::int64_t sfk);
inline bool valid_nvfp4_scales(const std::uint8_t* swizzled, std::int64_t n, std::int64_t sfk) {
  return nvfp4_scales_problem(swizzled, n, sfk) == nullptr;
}

}  // namespace mi::quant
