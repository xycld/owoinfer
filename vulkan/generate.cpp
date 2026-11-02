#include "generate.h"

#include <chrono>
#include <memory>
#include <stdexcept>

#include "text_encoder.h"
#include "vae.h"

namespace mi::vk {
namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("vulkan generate: " + what); }

}  // namespace

GenerateResult generate(Device& device, BlockPath path, const GenerateRequest& r) {
  if (r.width == 0 || r.height == 0 || r.width % 16 != 0 || r.height % 16 != 0) fail("width and height must be multiples of 16");
  const std::uint32_t lh = r.height / 16, lw = r.width / 16;
  if (r.noise.size() != std::size_t{kDiTLatentChannels} * lh * lw) fail("noise must be [128, height / 16, width / 16]");
  if (r.timesteps.empty() || r.sigmas.size() != r.timesteps.size() + 1) fail("sigmas must have one more entry than timesteps");
  if (r.cond_ids.empty()) fail("the prompt has no tokens");
  const bool cfg = r.cfg != 1.0f;
  if (cfg && r.uncond_ids.empty()) fail("the negative prompt has no tokens");
  const auto report = [&](const char* stage, int done, int total) {
    if (r.progress) r.progress(stage, done, total);
  };

  GenerateResult out;
  out.width = r.width;
  out.height = r.height;

  // Text encoder: streamed layer by layer, freed before the DiT loads.
  std::vector<std::vector<std::uint16_t>> embeds;
  {
    double t0 = now_s();
    report("text encoder", 0, 1);
    TextEncoder encoder(device, (r.model_dir / "text_encoder").string(), path);
    std::vector<std::vector<std::int32_t>> sequences{r.cond_ids};
    if (cfg) sequences.push_back(r.uncond_ids);
    embeds = encoder.encode(sequences);
    out.text_encoder_s = now_s() - t0;
    report("text encoder", 1, 1);
  }

  // DiT: resident, the text refiner once per prompt, then the steps.
  {
    double t0 = now_s();
    const BlockKernels kernels(device, path);
    // Blocks past what the device can hold with this run's activations stream from host memory.
    std::uint64_t tokens = 0;
    for (const auto& e : embeds) tokens += std::uint64_t{lh} * lw + e.size() / kDiTTextDim;
    constexpr std::uint64_t kReserve = std::uint64_t{512} << 20;  // pipelines, command buffers, the driver
    const std::uint64_t free = device.free_device_memory(), need = Dit::activation_bytes(tokens, std::uint64_t{lh} * lw) + kReserve;
    DitOptions options;
    options.device_budget = free > need ? free - need : 1;
    Dit dit(device, kernels, r.model_dir / "transformer", options);
    out.dit_resident_blocks = dit.resident_blocks();
    std::vector<std::unique_ptr<Buffer>> texts;
    std::vector<Dit::Row> rows;
    for (std::size_t i = 0; i < embeds.size(); ++i) {
      const auto length = static_cast<std::uint32_t>(embeds[i].size() / kDiTTextDim);
      texts.push_back(dit.encode_text(embeds[i], length));
      rows.push_back({texts.back().get(), length, i == 0 ? SkipSet{} : r.uncond_skip});
    }
    out.dit_load_s = now_s() - t0;

    t0 = now_s();
    std::vector<float> latents = r.noise;
    const int steps = static_cast<int>(r.timesteps.size());
    report("denoise", 0, steps);
    for (int s = 0; s < steps; ++s) {
      const auto si = static_cast<std::size_t>(s);
      const std::vector<std::vector<float>> pred = dit.forward(latents, lh, lw, r.timesteps[si], rows);
      // CFG then the official Euler step in the engine's op order (sampler_kernels.cu): u + g (c - u); x + h v, with
      // h = sigma_next - sigma rounded once from double. The fp32 expressions below compile without contraction.
      const float h = static_cast<float>(static_cast<double>(r.sigmas[si + 1]) - static_cast<double>(r.sigmas[si]));
      for (std::size_t i = 0; i < latents.size(); ++i) {
        float v = pred[0][i];
        if (cfg) {
          const float diff = pred[0][i] - pred[1][i];
          const float scaled = r.cfg * diff;
          v = pred[1][i] + scaled;
        }
        const float step = h * v;
        latents[i] = latents[i] + step;
      }
      report("denoise", s + 1, steps);
    }
    out.denoise_s = now_s() - t0;
    out.latents = std::move(latents);
  }

  // VAE decode.
  {
    const double t0 = now_s();
    report("vae", 0, 1);
    VaeDecoder vae(device, (r.model_dir / "vae" / "diffusion_pytorch_model.safetensors").string(),
                   path == BlockPath::kCoop ? VaePath::kCoop : VaePath::kPlain);
    vae.decode(out.latents.data(), lh, lw, out.rgb);
    out.vae_s = now_s() - t0;
    report("vae", 1, 1);
  }
  return out;
}

}  // namespace mi::vk
