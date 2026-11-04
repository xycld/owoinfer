// Text to image on the HIP backend: the text encoder, the DiT's sampling loop with CFG and the official Euler step, and
// the VAE decode. The VAE's weights (about 0.1 GiB) load first and stay; the text encoder streams its layers and is freed
// before the DiT loads; the DiT is freed before the VAE decodes. Memory the run cannot have (the decode, the DiT's
// activations and weights, the streamed blocks' pinned host memory) is refused with the sizes before the text encoder
// runs, and the DiT's part again when it loads. The caller supplies what the engine computes on the host for any
// backend: token ids (tokenizer and chat template), the initial noise (the seed's torch-compatible draw) and the
// schedule, so a seed gives the same inputs as on CUDA.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "dit.h"
#include "runtime.h"

namespace mi::hip {

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
  // The run's device memory in bytes, as the engine's --vram-budget: the VAE's and the DiT's weights, the DiT's
  // activations and a reserve for the code objects and the runtime. The DiT's blocks past what it leaves stream from
  // pinned host memory every step. 0: the automatic budget, what the device has free.
  std::uint64_t vram_budget = 0;
  bool decode = true;  // false: stop at the final latents
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
  std::uint64_t dit_free_bytes = 0;  // device memory free once the DiT's weights loaded, for its activations
};

GenerateResult generate(Device& device, const GenerateRequest& request);

}  // namespace mi::hip
