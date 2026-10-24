// Weight recipes of the NVFP4 converter (design: NVFP4 plan §4.4 and "N3 converter: reproducible output"). Each picks
// the UE4M3 block scales and E2M1 codes of one weight [n, k] under a given fp32 tensor scale s2 (the file's
// weight_scale_2); the element step is always the codec's (nvfp4_codec.h): total = s2 * sf in fp32, q =
// e2m1_rne(clamp(x / total, -6, 6)), +0 codes where total == 0.
//
//   rtn       sf = e4m3_rne(min((amax_b / 6) / s2, 448)): comfy-kitchen's quantize_nvfp4 bit for bit.
//   mse       per 16-block, the scale among RTN's and the 126 positive finite E4M3 values (ascending) with the smallest
//             error sum_i (x_i - deq_i)^2, deq_i = e2m1(q_i) * total in fp32, x_i - deq_i in fp32, its square and the
//             sum (i ascending) in fp64; RTN first and a strict '<', so ties keep the earlier scale.
//   gptq      GPTQ (Frantar et al. 2022) with this quantizer, natural column order, static s2, and each 16-column
//             group's scale chosen (RTN rule) from the error-updated weights when the group is reached.
//   gptq-mse  the same with the mse rule for the group scales.
// They are N2's recipes (tools/reference/nvfp4_study.py: mse_quant, gptq_quant), redefined with a fixed arithmetic so
// the bytes are a pure function of the inputs on every machine, compiler and thread count: no FMA contraction or
// fast-math in this translation unit (CMake), no BLAS, subnormals never flushed (the entry points clear flush-to-zero
// and denormals-are-zero on every thread they run on, and restore them), and every value computed by one fixed sequence
// of IEEE operations; threads only partition independent work.
//
// GPTQ, given H = X^T X (fp32 [k, k], summed over the calibration rows; only its upper triangle is read):
//   1. Hd = H in fp64, mirrored. A column j with Hd[j][j] == 0 is dead: Hd[j][j] = 1 and w[:, j] = 0.
//      damp = 0.01 * (sum_j Hd[j][j], j ascending) / k; Hd[j][j] += damp.
//   2. U = the upper Cholesky factor of Hd^-1 (Hd^-1 = U^T U), obtained without forming Hd^-1: with J the reversal,
//      J Hd J = R^T R (upper R) and U = J R^-T J. R by the right-looking Cholesky (step k: r_kk = sqrt(a_kk),
//      r_kj = a_kj / r_kk, then a_ij = a_ij - r_ki * r_kj for k < i <= j), R^-T by the right-looking forward
//      substitution (step k: m_kc = b_kc / r_kk for c <= k, then b_ic = b_ic - r_ki * m_kc for i > k), B = I. Every
//      element receives its updates in ascending k, one multiply and one subtract each; blocking and threads only
//      reorder independent elements, so the bits do not depend on them.
//   3. Per row, in fp64 (w = the fp32 weights widened), for col = 0 .. k-1: at col % 16 == 0 the group scale from the
//      group's current values rounded to fp32; x = fp32(w[col]), q from the element step; err = (w[col] -
//      e2m1(q) * total) / U[col][col]; then w[j] = w[j] - err * U[col][j] for every j > col, in that order (col
//      ascending per element). Rows are independent.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

// mi::quant rather than mi::weights: the format is shared by the reader, the converter and the GEMM kernels.
namespace mi::quant {

enum class Nvfp4Recipe : std::uint8_t { kRtn, kMse, kGptq, kGptqMse };
// "rtn" | "mse" | "gptq" | "gptq-mse"; the parser returns nullopt for anything else.
const char* recipe_name(Nvfp4Recipe r);
std::optional<Nvfp4Recipe> parse_nvfp4_recipe(std::string_view s);
// gptq and gptq-mse need calibration Hessians.
inline bool needs_hessian(Nvfp4Recipe r) { return r == Nvfp4Recipe::kGptq || r == Nvfp4Recipe::kGptqMse; }

// The converter's tensor scale of a whole weight (n values, finite): amax / 2688 in fp32 (nvfp4_tensor_scale), or 1
// for an all-zero weight (whose codes and scales are then all zero). The maximum is taken over chunks on `threads`
// threads; a maximum does not depend on the split.
float nvfp4_weight_tensor_scale(const float* w, std::int64_t n, int threads = 1);

// The mse rule for one block of 16 finite values: the chosen e4m3 scale code. Unlike the entry points below it does
// not set the floating-point environment itself (it runs once per block): call it with flush-to-zero off.
std::uint8_t nvfp4_mse_block_scale(const float* x, float s2);

// rtn (mse = false) or mse rows of a row-major fp32 [rows, k] weight (finite, k % 16 == 0): packed [rows, k/2]
// hi_first bytes and ROW-MAJOR scales [rows, k/16], as quantize_nvfp4_rows.
void quantize_nvfp4_rows_recipe(const float* w, std::int64_t rows, std::int64_t k, float s2, bool mse,
                                std::uint8_t* packed, std::uint8_t* scales);

// GPTQ steps 1-2: from h (fp32 [k, k], upper triangle read) to U (fp64 [k, k] row-major, upper triangle incl. the
// diagonal valid, the rest unspecified) and the dead columns (1 = dead). Throws QuantError when h is not finite or
// Hd is not positive definite. threads: 0 = all cores; the result does not depend on it.
struct GptqFactor {
  std::int64_t k = 0;
  std::vector<double> u;
  std::vector<std::uint8_t> dead;
};
GptqFactor gptq_factor(const float* h, std::int64_t k, int threads);

// GPTQ step 3 for `rows` rows of a row-major fp32 [rows, k] weight (finite): packed [rows, k/2] hi_first bytes and
// row-major scales [rows, k/16]. mse_scales: gptq-mse. Rows are processed in tiles on `threads` threads.
void gptq_quantize_rows(const float* w, std::int64_t rows, std::int64_t k, float s2, const GptqFactor& f, bool mse_scales,
                        int threads, std::uint8_t* packed, std::uint8_t* scales);

}  // namespace mi::quant
