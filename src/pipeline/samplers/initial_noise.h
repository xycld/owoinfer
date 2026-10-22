// Seed-defined initial noise of a generation.
//
// Latent layout. MOSS-Image samples a [128, H/16, W/16] fp32 latent (batch 1): the FLUX.2 VAE's 32 channels at H/8
// x W/8, patchified 2 x 2 into channels c * 4 + 2 * dy + dx and normalized by the VAE's batch-norm statistics. The
// official pipeline draws it as randn([1, 128, H/16, W/16]) from a CPU torch generator seeded with the seed; ComfyUI's
// FLUX.2 latents (EmptyFlux2LatentImage, used with its MOSS-Image support) have the same shape, channel order and
// normalization, and its prepare_noise draws randn([B, 128, H/16, W/16]) from torch's CPU generator seeded the same
// way. So for one image both layouts are the same values in the same order (element [c, y, x] at c * h * w + y * w + x
// of the stream), and the same seed gives the same composition. They differ for batches: ComfyUI draws the whole batch
// from one stream (image k takes the k-th block of 128 * h * w values; kBatchStream reproduces it), while in the
// official app every image is its own generation with its own seed (kOfficial: the caller passes each image's seed,
// e.g. seed + k, and the result is independent of batch_index).
//
// The CPU generator is torch's mt19937 seeded with the low 32 bits of the seed (pipeline/rng_torch_compat.h).
// Per-step sampler noise is separate (samplers/sampler_driver.h).
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace mi::pipeline {

enum class NoiseLayout {
  kOfficial,     // randn([1, 128, h, w]) per image, seeded per image (the default)
  kBatchStream,  // randn([batch_size, 128, h, w]) once from the seed, image batch_index of it
};
const char* noise_layout_name(NoiseLayout layout);
// official, batch_stream; throws std::invalid_argument otherwise.
NoiseLayout parse_noise_layout(std::string_view name);

// The initial noise [channels, latent_h, latent_w] of image `batch_index` of `batch_size` (kOfficial ignores the
// batch: every image is its own draw). Throws std::invalid_argument for non-positive sizes or batch_index out of range.
std::vector<float> initial_noise(NoiseLayout layout, std::uint64_t seed, std::int64_t channels, std::int64_t latent_h,
                                 std::int64_t latent_w, int batch_index = 0, int batch_size = 1);

}  // namespace mi::pipeline
