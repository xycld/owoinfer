// The FLUX.2 VAE decoder as the MOSS-Image pipeline runs it (the engine's src/models/flux2_vae, pipeline_moss_image.py
// 430-438): the DiT's final latents, fp32 NCHW [128, h, w], to RGB8 [16h, 16w, 3]. Activations are fp32 NHWC, as in the
// engine's reference precision (DecodeMode::reference()):
//   latent de-normalization + unpack -> post_quant_conv 1x1 -> conv_in 3x3 -> mid resnet 0 -> mid attention
//   -> mid resnet 1 -> 4 up blocks (3 resnets each, nearest 2x upsample + 3x3 conv after the first three)
//   -> GroupNorm + SiLU -> conv_out 3x3 -> clamp and the image processor's rounding to uint8.
// Convolutions multiply the TF32-rounded operands exactly and accumulate in fp32 (torch's allow_tf32 convolutions), the
// nearest upsample read inside the following convolution; the attention's linears are exact-product fp32 GEMMs, its
// scores and P V fp32-accurate (vae_kernels.h); GroupNorm statistics are the engine's exact ones. Accumulation orders
// differ from cuDNN's and torch's, so outputs are close to the engine's, not bit-identical.
//
// Memory (untiled): two regions of 2c x 16h x 16w fp32 each (c = 96 for the small decoder) as the engine's ping-pong
// schedule, the latents, the RGB output and the GroupNorm accumulators; the mid attention's scratch lives in the second
// region. workspace_bytes() is the total; decode() refuses to start when the device's free memory cannot hold it.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "vae_kernels.h"

namespace mi::hip {

struct VaeDecodeStats {
  double seconds = 0;              // decode() wall time, the latents' upload and the RGB download included
  std::uint64_t device_bytes = 0;  // device memory decode() allocated (weights not included)
  // With set_profile(true): device milliseconds per op kind, in order of first use.
  std::vector<std::pair<std::string, double>> profile;
};

class VaeDecoder {
 public:
  // Loads the decoder ("decoder.*", "post_quant_conv.*") and the latent statistics ("bn.running_mean", "bn.running_var")
  // from a FLUX.2 VAE safetensors file in diffusers' naming, fp32 or bf16. The widths come from the weights' shapes
  // (decoder_block_out_channels c, 2c, 4c, 4c; c a multiple of 32).
  VaeDecoder(Device& device, const std::filesystem::path& safetensors_path);
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  int base_channels() const { return base_; }
  std::uint64_t weight_bytes() const;
  // Device bytes decode() allocates for an h x w latent grid, for a decoder of base width c (96: the MOSS bundle's).
  static std::uint64_t workspace_bytes(int base_channels, std::uint32_t h, std::uint32_t w);
  std::uint64_t workspace_bytes(std::uint32_t h, std::uint32_t w) const { return workspace_bytes(base_, h, w); }
  // Throws std::runtime_error naming both sizes when the device's free memory is below workspace_bytes(h, w).
  void require_fits(std::uint32_t h, std::uint32_t w) const;

  // The convolutions' products: hip_vae::Products::kTf32 (the default, exact TF32 products) or kTf32Drop.
  void set_conv_products(hip_vae::Products products);
  // Times every op with device events (VaeDecodeStats::profile).
  void set_profile(bool on) { profile_ = on; }

  // latents: fp32 NCHW [128, h, w], the pipeline's final latents. rgb: resized to 16h x 16w x 3. image: when not null, the
  // decoder's fp32 output [16h, 16w, 3] before the clamp. Runs on its own stream and returns when the RGB is on the host.
  VaeDecodeStats decode(const float* latents, std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb,
                        std::vector<float>* image = nullptr);

  struct Weights;

 private:
  Device& device_;
  std::unique_ptr<Weights> weights_;
  int base_ = 0;
  hip_vae::Products conv_products_ = hip_vae::Products::kTf32;
  bool profile_ = false;
};

}  // namespace mi::hip
