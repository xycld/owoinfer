// GPU ops the FLUX.2 VAE encoder needs on top of vae_kernels.h (diffusers AutoencoderKLFlux2.encode with the
// MOSS-Image bundle's VAE config). Activations are NHWC fp32, as in the decoder.
//
// Reference ops (diffusers 0.41: vae.py Encoder, unet_2d_blocks.py DownEncoderBlock2D/UNetMidBlock2D,
// downsampling.py Downsample2D, autoencoder_kl_flux2.py, pipeline_flux2.py _encode_vae_image, image_processor.py):
//   VaeImageProcessor.preprocess: np.float32(u8) / 255, then 2x - 1                         encoder_preprocess
//   Downsample2D(padding=0): F.pad(x, (0,1,0,1)) then Conv2d(3x3, stride 2, padding 0)      conv2d_down
//   mid-block attention: 1 head, head_dim 512 (GN(32,1e-6) -> to_q/k/v -> SDPA -> to_out -> + residual):
//   vae::mid_attention with D = 512 (kernels/vae/attention.h)
//   quant_conv output [B,64,h,w] -> DiagonalGaussianDistribution (mean | logvar, logvar clamped to [-30, 20],
//   std = exp(0.5 logvar)), mode() = mean or sample = mean + std * randn, then Flux2Pipeline._patchify_latents
//   and (z - bn.running_mean) / sqrt(bn.running_var + eps)                                encoder_latent_out
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "core/tensor.h"
#include "kernels/vae/vae_kernels.h"

namespace mi::vae {

// ---- Convolutions (CUTLASS implicit-GEMM fprop, fp32 NHWC activations, KRSC filters, bias epilogue) --------
enum class ConvMath {
  kTf32,    // operands rounded to nearest TF32, fp32 accumulation (vae::conv2d's math)
  kTf32x3,  // 3xTF32 split (OpMultiplyAddFastF32): fp32-accurate, ~3x the tensor-core work
};

// Stride-1 conv with vae::conv2d's contract (R in {1, 3}, "same" padding, y = (conv + bias) + residual in the
// reference's rounding order, y may alias the residual but not x), restricted to fp32 and K >= 64, K % 4 == 0.
// kTf32 calls vae::conv2d.
std::size_t encoder_conv2d_workspace_bytes(const Tensor& x, const Tensor& w, ConvMath math);
void encoder_conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
                    void* workspace, std::size_t workspace_bytes, ConvMath math, cudaStream_t stream);

// Stride-2 downsample conv: x [N,H,W,C], w [K,3,3,C] (KRSC), bias [K] -> y [N,P,Q,K] with
//   y[n,i,j,k] = bias[k] + sum_{r,s,c} w[k,r,s,c] * x[n, 2i+r, 2j+s, c],   x = 0 outside [0,H) x [0,W)
// i.e. the reference's zero pad of one row/column at the bottom/right (and nothing at the top/left) followed by
// a stride-2 valid conv. P and Q come from y: 1 <= P <= (H+1)/2, 1 <= Q <= (W+1)/2 (the whole tensor: H/2, W/2).
// A P smaller than H/2 computes a top band of the output, so a caller can convolve row bands of x (starting on
// an even row) and get the very values of the whole-tensor call: each output pixel accumulates over the same
// (r, s, c) sequence wherever it lies. C % 4 == 0, K % 4 == 0 and K >= 64; x, w, y 16-byte aligned, bias 32-byte;
// x and y below 2 GiB.
std::size_t conv2d_down_workspace_bytes(const Tensor& x, const Tensor& w, const Tensor& y, ConvMath math);
void conv2d_down(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor& y, void* workspace,
                 std::size_t workspace_bytes, ConvMath math, cudaStream_t stream);

// ---- Elementwise / layout -------------------------------------------------------------------------------
// Image -> encoder input: rgb [H,W,3] (or [1,H,W,3]) u8, or f32 in [0, 1] (an image already divided by 255),
// -> y [1,H,W,4] fp32 NHWC with y[..., c] = 2 * v - 1 for c < 3 (v = u8 / 255.0f rounded to nearest, or the f32
// value) and y[..., 3] = 0 (conv_in runs with C padded to 4 and a zero weight column). Bit-exact.
void encoder_preprocess(const Tensor& rgb, const Tensor& y, cudaStream_t stream);

// quant_conv output moments [1,h,w,64] fp32 NHWC (h, w even) -> normalized DiT latents y [1,128,h/2,w/2] fp32 NCHW:
//   z = mean (noise == nullptr), or mean + exp(0.5 * clamp(logvar, -30, 20)) * noise, noise [1,32,h,w] fp32 NCHW
//   (the reference's randn layout); y[4c+2p+q, i, j] = (z[c, 2i+p, 2j+q] - bn_mean) / sqrt(bn_var + eps).
// Every op rounds separately, in the reference's order (bit-exact for the mean; exp within libm's 1-2 ulp).
void encoder_latent_out(const Tensor& moments, const Tensor& bn_mean, const Tensor& bn_var, float eps,
                        const Tensor* noise, const Tensor& y, cudaStream_t stream);

// torch Conv2d weight [K,C,R,S] fp32 -> [K,R,S,Cp] fp32 with Cp >= C, zero in channels [C, Cp).
void conv_weight_to_krsc_padded(const Tensor& w_kcrs, const Tensor& w_krsc, cudaStream_t stream);

}  // namespace mi::vae
