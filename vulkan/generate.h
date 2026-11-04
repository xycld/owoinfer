// Text to image on the Vulkan backend: the text encoder, the DiT's sampling loop with CFG and the official Euler step, and
// the VAE decode, one model at a time on the device (the text encoder streams its layers and is freed before the DiT
// loads; the DiT is freed before the VAE loads). The caller supplies what the engine computes on the host for any
// backend: token ids (tokenizer and chat template), the initial noise (the seed's torch-compatible draw) and the
// schedule, so a seed gives the same inputs as on CUDA.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "block.h"
#include "dit.h"
#include "runtime.h"

namespace mi::vk {

struct GenerateRequest {
  std::filesystem::path model_dir;  // MOSS-Image directory: text_encoder/, transformer/, vae/
  std::vector<std::int32_t> cond_ids;
  std::vector<std::int32_t> uncond_ids;  // unused when cfg == 1
  std::uint32_t width = 0, height = 0;   // pixels, multiples of 16
  std::vector<float> noise;              // f32 [128, height / 16, width / 16]
  std::vector<float> sigmas;             // steps + 1
  std::vector<float> timesteps;          // steps
  float cfg = 3.0f;
  SkipSet uncond_skip;
  bool decode = true;  // false: stop at the final latents
  // Device memory the VAE and the DiT may use, in bytes: their weights, the DiT's activations and the VAE's decode; the
  // DiT's blocks past it stream from host memory. 0: what the device has free. A decode it cannot hold is refused first.
  std::uint64_t vram_budget = 0;
  // Called with the final latents before the decode.
  std::function<void(const std::vector<float>& latents)> on_latents;
  // stage: "text encoder", "denoise", "vae"; done of total within it.
  std::function<void(const char* stage, int done, int total)> progress;
};

struct GenerateResult {
  std::uint32_t width = 0, height = 0;
  std::vector<std::uint8_t> rgb;  // [height, width, 3]; empty without the decode
  std::vector<float> latents;     // the final latents, f32 [128, height / 16, width / 16]
  double text_encoder_s = 0, dit_load_s = 0, denoise_s = 0, vae_s = 0;
  int dit_resident_blocks = 0;  // of 42; the rest streamed from host memory every step
};

// path: the DiT's and text encoder's kernels (kCoop needs Capabilities::coop_bf16, kCoopF16 coop_f16); the VAE runs
// kCoop on kCoop and kPlain otherwise (its cooperative-matrix kernels need bf16 fragments).
GenerateResult generate(Device& device, BlockPath path, const GenerateRequest& request);

}  // namespace mi::vk
