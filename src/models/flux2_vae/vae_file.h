// A FLUX.2 VAE given as one safetensors file (e.g. Comfy-Org's flux2-vae.safetensors, the full decoder, or
// flux2-vae_small.safetensors), indexed as the "decoder" weight group Flux2VaeDecoder loads. Tensor bytes are read in
// place from the file; nothing is converted on disk.
//
// Accepted key layouts (detected per file; an optional "vae." or "first_stage_model." prefix on every key is
// stripped):
//   - diffusers AutoencoderKLFlux2: decoder.*, post_quant_conv.*, bn.* (the bundle's names, used as is);
//   - BFL/LDM (the original FLUX.2 release, as Comfy-Org ships the full decoder): decoder.mid.block_{1,2},
//     decoder.mid.attn_1.{norm,q,k,v,proj_out} (1x1 conv weights, read as [C, C] linears), decoder.up.{i}.block.{j}
//     (up.{i} runs as up_blocks[3 - i]), nin_shortcut, upsample.conv, norm_out, decoder.post_quant_conv.
// Encoder, quant_conv and bn.num_batches_tracked tensors are listed as excluded. The decoder widths follow from the
// shapes (decoder_config()). The group keeps the file's bn running mean/var, its fp32 originals; MOSS latents are
// decoded with them rounded to bf16, as the bundle stores them: Flux2VaeDecoder's round_stats (read_latent_stats).
//
// Use the returned index whole: its tensors' file ids refer to its own one-entry file list, so a group copied into
// another index would read the wrong file. The decoder rejects tensors that lie outside their index's files.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "weights/model_index.h"

namespace mi::flux2_vae {

// Throws weights::FormatError / IoError for unreadable files and weights::IndexError for a file without a complete
// FLUX.2 decoder (missing or unexpected decoder tensors, non-F32 data).
weights::ModelIndex index_vae_file(const std::filesystem::path& file);

// The latent statistics of an index's VAE "decoder" group: bn.running_mean and bn.running_var, F32 [128], read from its
// files. bf16_rounded: rounded to bf16 (round to nearest even), still F32, as the MOSS-Image bundle stores them: the
// values a single VAE file (its fp32 originals) is decoded and encoded with (Flux2VaeDecoder's and Flux2VaeEncoder's
// round_stats). Throws std::runtime_error for a missing or misshapen tensor, weights::IoError for an unreadable file.
struct LatentStats {
  std::vector<float> mean, var;
};
LatentStats read_latent_stats(const weights::ModelIndex& index, bool bf16_rounded);

// The diffusers name of a single-file VAE key, or "" for a tensor the decoder does not use (encoder, quant_conv,
// bn.num_batches_tracked). `key` has its "vae." / "first_stage_model." prefix already removed. Exposed for tests.
std::string vae_file_tensor_name(const std::string& key);

}  // namespace mi::flux2_vae
