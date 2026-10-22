// Fast approximate latent -> RGB preview for progress callbacks (no VAE).
//
// The DiT latent [128, h, w] (fp32, the sampler's state or its x0 estimate) is unpacked to the VAE's 32-channel grid
// at H/8 (channel 4c + 2p + q is VAE channel c at sub-pixel (p, q)); every H/8 pixel becomes RGB by a fixed affine map
// of the 32 channels of its 3x3 neighbourhood (zero outside the image; coefficients fitted against real decodes, see
// preview_coefficients.h); the H/8 image is then resampled bilinearly (align_corners = false, as
// F.interpolate(mode="bilinear")) to any output size and quantized to RGB8 (clamp to [0, 1], round half to even).
// The full image size is 16h x 16w; 2h x 2w is the preview's native resolution.
//
// Quality (held-out golden latents, preview at full size vs the true decode): 20-25 dB PSNR on final latents. For a
// sharp preview mid-sampling, pass the x0 estimate (flow matching: x0 = x_t - sigma_t * v_t) rather than x_t; x_t
// previews show the noise blend (it is the same linear map, so any latent works).
//
// Cost: two small kernels on `stream`, no allocations, no host synchronization (~0.05 ms for a 1024^2 preview on the
// dev box).
// Both decoders (small and full FLUX.2) share the latent space, so one table serves both.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mi::latent_preview {

// Scratch bytes preview() needs for a latent grid of height x width: the H/8 RGB image, fp32 [2h, 2w, 3].
std::size_t workspace_bytes(std::int64_t height, std::int64_t width);

// latents: contiguous CUDA fp32 [128, h, w] or [1, 128, h, w]. rgb: contiguous CUDA u8 [H, W, 3] or [1, H, W, 3] of any
// size >= 1 x 1 (16h x 16w: the image size; 2h x 2w: native). workspace: >= workspace_bytes(h, w), 16-byte aligned,
// device memory. Asynchronous on `stream`. Throws std::invalid_argument on bad arguments.
void preview(const Tensor& latents, const Tensor& rgb, void* workspace, std::size_t workspace_bytes, cudaStream_t stream);

}  // namespace mi::latent_preview
