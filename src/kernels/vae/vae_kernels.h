// GPU ops for the FLUX.2 VAE decoder (diffusers AutoencoderKLFlux2 with the MOSS-Image bundle's VAE config).
//
// Activations are NHWC inside the decoder; the only NCHW tensors are the DiT latents going in (consumed by
// bn_denorm_unpack, which writes NHWC directly) and an optional NCHW export of the decoded image.
//
// Reference op sequence (diffusers 0.41: autoencoder_kl_flux2.py, vae.py Decoder, unet_2d_blocks.py
// UNetMidBlock2D/UpDecoderBlock2D, resnet.py ResnetBlock2D/Upsample2D, attention_processor.py
// AttnProcessor2_0), for an H x W image and batch B. Everything is fp32; convs run under cuDNN's default
// allow_tf32=True (torch 2.14: backends.cudnn.conv.fp32_precision == "tf32"), the attention linears under
// allow_tf32=False (matmul fp32_precision == "none", strict fp32), SDPA via the mem-efficient kernel (3xTF32,
// OpMultiplyAddFastF32, i.e. fp32-accurate).
//
//   latents  [B,128,H/16,W/16] fp32 NCHW
//   pipeline (P:430-436): z*sqrt(var+1e-4) (rounded) + mean (rounded), then
//            (b,c/4,2,2,h,w) -> (b,c/4,h,2,w,2)                  -> [B,32,H/8,W/8]   bn_denorm_unpack
//   post_quant_conv 1x1 32->32                                                       conv2d
//   decoder.conv_in 3x3 32->384                                   @ H/8              conv2d
//   mid_block (384 ch @ H/8):
//     resnets[0]: GN(32,1e-6)+SiLU, conv3x3, GN+SiLU, conv3x3, + x               group_norm, conv2d(+residual)
//     attentions[0]: 1 head, head_dim 384 (heads = 384 // attention_head_dim 384), scale 384^-0.5,
//       GN(32,1e-6, affine) on [B,C,HW] -> tokens [B,HW,C] (= NHWC rows) -> to_q/to_k/to_v (Linear+bias)
//       -> SDPA over L = H/8*W/8 tokens (16384 @1024^2, 33124 @1456^2, 65536 @2048^2) -> to_out[0]
//       (Linear+bias) -> + residual -> / rescale_output_factor (1.0)                 mid_attention
//     resnets[1]: as resnets[0]
//   up_blocks[0]: 3 x resnet 384->384 @ H/8, Upsample2D: nearest 2x, conv3x3 384->384 -> H/4
//   up_blocks[1]: 3 x resnet 384->384 @ H/4, nearest 2x + conv3x3 384             -> H/2
//   up_blocks[2]: resnet 384->192 (shortcut = conv1x1 384->192 with bias), 2 x resnet 192->192 @ H/2,
//                 nearest 2x + conv3x3 192                                           -> H
//   up_blocks[3]: resnet 192->96 (1x1 shortcut), 2 x resnet 96->96 @ H, no upsample   upsample_nearest2x
//   conv_norm_out GN(32 groups, 96 ch, 1e-6) + SiLU, conv_out 3x3 96->3                group_norm, conv2d
//   pipeline: .clamp(-1,1); VaeImageProcessor.postprocess: (x*0.5+0.5).clamp(0,1) on GPU, then numpy
//             (x*255).round() (half-to-even) .astype(uint8)                         postprocess_to_rgb8
//   ResnetBlock2D: out = (shortcut(x) + conv2(...)) / output_scale_factor(1.0); dropout is identity.
//
// Precision modes: fp32 tensors run convs as TF32 tensor-core implicit GEMM with fp32 accumulation
// (operands rounded to nearest TF32, like cuDNN); bf16 tensors run bf16 x bf16 -> fp32 accumulate.
// Every op is asynchronous on `stream`, allocates nothing, and takes scratch from the caller.
#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mi::vae {

inline constexpr int kGroupNormGroups = 32;
inline constexpr float kResnetEps = 1e-6f;    // ResnetBlock2D / attention / conv_norm_out GroupNorm eps
inline constexpr float kBatchNormEps = 1e-4f; // vae.config.batch_norm_eps

// ---- Convolution (CUTLASS implicit-GEMM fprop, NHWC) ----------------------------------------------------
// x [N,H,W,C], w [K,R,S,C] (see conv_weight_to_krsc), bias [K] fp32, y [N,H,W,K]; R == S in {1, 3}, stride 1,
// "same" zero padding (R-1)/2. x, w, y (and residual) share one dtype: kF32 (TF32 math) or kBF16.
// C must be a multiple of 4 (fp32) or 8 (bf16). If `residual` is non-null, y = (conv + bias) + residual, fused
// in the epilogue (fp32: the reference's rounding order; bf16: one rounding at the end). y must not alias x; it
// may alias residual. Alignment: x, w 16 bytes; bias 32 bytes; y, residual 16 bytes (4 bytes when K < 64).
// The fp32 path rounds operands to nearest TF32 and accumulates in the same order as cuDNN's TF32 kernels
// (bit-identical to torch 2.14 / cuDNN 9.24 on the decoder's K >= 64 convs; conv_out differs by ~2e-6 rel).
std::size_t conv2d_workspace_bytes(const Tensor& x, const Tensor& w);
void conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
            void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

// torch Conv2d weight [K,C,R,S] fp32 -> [K,R,S,C] in dst's dtype (kF32 or kBF16, round to nearest even).
void conv_weight_to_krsc(const Tensor& w_kcrs, const Tensor& w_krsc, cudaStream_t stream);

// ---- GroupNorm (NHWC; exact integer sums, fp32 {mean, rstd}) ---------------------------------------------
// Statistics are per (n, group) {mean, rstd} pairs: `stats` is fp32 [N, groups, 2]. Each element's x and x^2 are
// rounded once onto fixed-point grids (2^-64 and 2^-96) and summed in integers, so the statistics do not depend on
// how the tensor is split or in which order the parts are summed: an exact tiled decoder accumulates any spatial
// tiles of a layer (each pixel exactly once) and gets the bit-identical statistics of the whole tensor.
//   1. group_norm_accum_zero, then group_norm_accumulate for each tile (any shape, any order, same stream).
//   2. group_norm_finalize: fp64 mean and variance from the exact sums, rounded to fp32 {mean, rstd}.
// Inputs must satisfy |x| < 2^33 (a group's statistics become NaN otherwise), and a group may hold up to 2^30
// elements. group_norm_stats does all three for one whole tensor; group_norm_apply normalizes any tile
// (+ optional SiLU, computed as torch does: y / (1 + exp(-y))). One-pass group_norm = stats + apply.
std::size_t group_norm_accum_bytes(int batch, int groups);
void group_norm_accum_zero(void* accum, int batch, int groups, cudaStream_t stream);
// x_tile: contiguous f32/bf16 CUDA [N, h, w, C], 16-byte aligned (as accum).
void group_norm_accumulate(const Tensor& x_tile, int groups, void* accum, cudaStream_t stream);
void group_norm_finalize(const void* accum, int batch, int groups, std::int64_t count_per_group, float eps,
                         const Tensor& stats, cudaStream_t stream);
std::size_t group_norm_workspace_bytes(const Tensor& x, int groups);
void group_norm_stats(const Tensor& x, int groups, float eps, const Tensor& stats, void* workspace,
                      std::size_t workspace_bytes, cudaStream_t stream);
// x, y: [N,H,W,C] (kF32 or kBF16, same dtype, may alias); gamma, beta: [C] fp32.
void group_norm_apply(const Tensor& x, const Tensor& stats, const Tensor& gamma, const Tensor& beta, bool silu,
                      const Tensor& y, cudaStream_t stream);
void group_norm(const Tensor& x, int groups, float eps, const Tensor& gamma, const Tensor& beta, bool silu,
                const Tensor& y, void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

// ---- Mid-block attention (fp32) -------------------------------------------------------------------------
enum class AttnPrecision {
  kTf32x3,  // split-TF32 (big + small terms) for QK^T and PV: fp32-accurate, like the reference SDPA
  kTf32,    // single TF32 pass: ~3x less tensor-core work, max |err| ~2e-4 of the output range
};

// Single-head attention O = softmax(Q K^T * scale) V for fp32 rows of width 384. q, k, v: [B, L, 384] views
// whose row stride (stride[1]) may exceed 384 (e.g. column slices of one packed [B, L, 1152] QKV buffer);
// row stride and base pointers must be 16-byte aligned. o: contiguous [B, L, 384], must not overlap q/k/v.
// Never materializes L x L.
void attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& o, float scale,
                    AttnPrecision precision, cudaStream_t stream);

struct MidAttentionWeights {  // all fp32 on device, torch layouts (Linear weight [out, in])
  Tensor gn_gamma, gn_beta;   // [C]
  Tensor wq, bq, wk, bk, wv, bv, wo, bo;  // [C, C], [C]
};

// y = residual + to_out(SDPA(to_q/k/v(GN(x)))) for x, y [B,H,W,384] fp32 (residual null: no add, as for a
// test of the attention branch alone; the decoder passes residual = &x). y may alias x/residual.
// The four linears are cuBLASLt fp32 GEMMs; `fast_linears` switches them to TF32 (reference: strict fp32). The
// residual is added after to_out's bias, in the reference order. Host side, each call builds cuBLASLt
// descriptors and runs the heuristic (no device allocation).
std::size_t mid_attention_workspace_bytes(const Tensor& x);
void mid_attention(cublasLtHandle_t lt, const Tensor& x, const MidAttentionWeights& w, const Tensor* residual,
                   const Tensor& y, void* workspace, std::size_t workspace_bytes, AttnPrecision precision,
                   bool fast_linears, cudaStream_t stream);

// ---- Elementwise / layout -------------------------------------------------------------------------------
// [N,H,W,C] -> [N,2H,2W,C] nearest neighbour (F.interpolate(scale_factor=2, mode="nearest")). kF32 or kBF16.
void upsample_nearest2x(const Tensor& x, const Tensor& y, cudaStream_t stream);
// y = a + b (same shape and dtype, kF32 or kBF16, any of them may alias).
void residual_add(const Tensor& a, const Tensor& b, const Tensor& y, cudaStream_t stream);
// Latent preprocessing: z [B,128,h,w] fp32 NCHW, bn running mean/var [128] fp32 ->
// y [B,2h,2w,32] fp32 NHWC with y[b,2i+p,2j+q,c] = z[b,4c+2p+q,i,j] * sqrt(var+eps) + mean, rounded like the
// reference (sqrt and each of +, *, + separately round-to-nearest). Bit-exact.
void bn_denorm_unpack(const Tensor& z, const Tensor& bn_mean, const Tensor& bn_var, float eps, const Tensor& y,
                      cudaStream_t stream);
// Decoder output x [B,H,W,3] fp32 -> RGB8 [B,H,W,3] u8 matching .clamp(-1,1) + VaeImageProcessor.postprocess:
// v = clamp(clamp(x,-1,1)*0.5 + 0.5, 0, 1); round_half_even(v*255). Bit-exact (NaN maps to 0).
void postprocess_to_rgb8(const Tensor& x, const Tensor& rgb, cudaStream_t stream);
// fp32 layout conversion at the decoder boundary.
void nchw_to_nhwc(const Tensor& x_nchw, const Tensor& y_nhwc, cudaStream_t stream);
void nhwc_to_nchw(const Tensor& x_nhwc, const Tensor& y_nchw, cudaStream_t stream);

}  // namespace mi::vae
