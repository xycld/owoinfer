// Offline (CPU) quantization of bf16 weights into the ConvRot W4A4 format (comfy-kitchen "convrot_w4a4", the operands
// of kernels::quant4's INT4 GEMMs): q packed signed nibbles [n, k/2] (even column = low nibble, values -7..7) and a
// per-row f32 scale [n].
//
// Rotation: W_rot = bf16(W · H256) per 256-column group, H256 the regular Hadamard of quant_codec.h (comfy-kitchen's
// eager _rotate_weight runs on the bf16 weight and rounds the rotated matrix to bf16). The fp32 rotation here sums in
// column order; eager's bf16 GEMM may round a rotated value differently on rare elements, so the result equals
// comfy-kitchen's quantize_convrot_w4a4_weight except for that rotation GEMM's rounding.
// Scale: the eager rule s = bf16(bf16(max(absmax, 1e-10)) * fp32(1/7)) (kAbsmax), or the bf16 scale of the clipping
// ratio in [0.40, 1.00] (step 0.01) with the least squared error of the row (kMse; ratio 1.00 is the absmax scale, so
// the error is never larger). Values: q = clamp(rne(bf16(W_rot / s)), -7, 7).
// GPTQ (Frantar et al. 2022), optional: with a calibration Hessian H = X^T X of the layer's (unrotated) inputs, the
// rows are quantized column by column against H_rot = H256 · H · H256 (the GEMM sees x · H256), propagating each
// column's error through the upper Cholesky factor of H_rot^-1; the per-row scale is chosen first (as above) and kept.
//
// Deterministic: the result does not depend on the thread count. Inputs must be finite (throws std::invalid_argument).
#pragma once

#include <cstdint>
#include <vector>

namespace mi::quant {

enum class W4ScaleSearch { kAbsmax, kMse };

struct ConvRotW4A4Options {
  W4ScaleSearch search = W4ScaleSearch::kAbsmax;
  const double* hessian = nullptr;  // [k, k] X^T X (unrotated inputs); null: round to nearest
  double gptq_damp = 0.01;          // fraction of the mean diagonal added to H_rot's diagonal
  int threads = 0;                  // 0: hardware concurrency
};

struct ConvRotW4A4Weight {
  std::vector<std::uint8_t> q;  // [n, k/2]
  std::vector<float> scale;     // [n], bf16 values
};

// k % 256 == 0. GPTQ memory: two k x k double matrices besides the Hessian (1 GiB at k = 8192).
ConvRotW4A4Weight quantize_convrot_w4a4(const std::uint16_t* w_bf16, std::int64_t n, std::int64_t k,
                                        const ConvRotW4A4Options& opt = {});

// X^T X (double [k, k]) of `rows` rows of a bf16 matrix (row stride ld), accumulated in double.
std::vector<double> gram_bf16(const std::uint16_t* x, std::int64_t rows, std::int64_t k, std::int64_t ld,
                              int threads = 0);

}  // namespace mi::quant
