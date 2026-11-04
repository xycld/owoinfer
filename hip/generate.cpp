// <chrono> before the HIP headers: their __noinline__ macro breaks libstdc++'s <format>, which <chrono> includes in C++20.
#include <chrono>

#include "generate.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>

#include "text_encoder.h"
#include "vae.h"

namespace mi::hip {
namespace {

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(1ull << 30); }

[[noreturn]] void fail(const std::string& what) { throw std::invalid_argument("hip generate: " + what); }

}  // namespace

GenerateResult generate(Device& device, const GenerateRequest& r) {
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

  // The VAE first (about 0.1 GiB of weights, kept through the run), so a decode the device cannot hold fails before the
  // text encoder and the steps run rather than after them.
  std::optional<VaeDecoder> vae;
  if (r.decode) {
    vae.emplace(device, r.model_dir / "vae" / "diffusion_pytorch_model.safetensors");
    vae->require_fits(lh, lw);  // also refuses a size past the decoder's limits
    if (const std::uint64_t need = vae->weight_bytes() + vae->workspace_bytes(lh, lw); r.vram_budget != 0 && need > r.vram_budget) {
      char msg[256];
      std::snprintf(msg, sizeof msg,
                    "the VAE decode of a %ux%u image needs %.2f GiB of device memory with its weights, more than a VRAM "
                    "budget of %.2f GiB",
                    r.width, r.height, gib(need), gib(r.vram_budget));
      throw std::runtime_error(msg);
    }
  }

  // Text encoder: streamed layer by layer, freed before the DiT loads.
  std::vector<std::vector<std::uint16_t>> embeds;
  {
    double t0 = now_s();
    report("text encoder", 0, 1);
    TextEncoder encoder(device, r.model_dir / "text_encoder");
    std::vector<std::vector<std::int32_t>> sequences{r.cond_ids};
    if (cfg) sequences.push_back(r.uncond_ids);
    embeds = encoder.encode(sequences);
    out.text_encoder_s = now_s() - t0;
    report("text encoder", 1, 1);
  }

  // DiT: resident, the text refiner once per prompt, then the steps.
  {
    double t0 = now_s();
    // What the DiT holds beside its weights: the steps' activations, the refiner's scratch kept by the Dit (its input
    // and SiTU-GLU buffers, at the longest prompt) and the refined text rows; and a reserve for the code objects and the
    // runtime. The weights take what is left (the automatic budget) or what the VRAM budget leaves beside the VAE's
    // weights and that need, the smaller of the two; the blocks past it stream from pinned host memory.
    const std::uint64_t hw = std::uint64_t{lh} * lw;
    std::uint64_t tokens = 0, text_rows = 0, longest = 0;
    for (const auto& e : embeds) {
      const std::uint64_t length = e.size() / kDiTTextDim;
      tokens += hw + length;
      text_rows += length;
      longest = std::max(longest, length);
    }
    constexpr std::uint64_t kBf16 = 2, kReserve = std::uint64_t{512} << 20;
    const std::uint64_t activations = Dit::activation_bytes(tokens, embeds.size(), hw) +
                                      longest * (kDiTTextDim + 2 * std::uint64_t{kDiTFfnDim}) * kBf16 +
                                      text_rows * kDiTDim * kBf16;
    const std::uint64_t need = activations + kReserve;
    char budget[64];
    if (r.vram_budget != 0) {
      std::snprintf(budget, sizeof budget, "a VRAM budget of %.2f GiB", gib(r.vram_budget));
    } else {
      std::snprintf(budget, sizeof budget, "the automatic budget");
    }
    char msg[512];
    const std::uint64_t free = device.free_memory();
    if (need > free) {
      std::snprintf(msg, sizeof msg,
                    "the DiT at %ux%u needs %.2f GiB of device memory beside its weights (activations %.2f GiB, reserve "
                    "%.2f GiB); %s has %.2f GiB free",
                    r.width, r.height, gib(need), gib(activations), gib(kReserve), device.info().name.c_str(), gib(free));
      throw std::runtime_error(msg);
    }
    std::uint64_t dit_budget = free - need;
    const std::uint64_t vae_bytes = vae ? vae->weight_bytes() : 0;
    if (r.vram_budget != 0) {
      if (r.vram_budget <= vae_bytes + need) {
        std::snprintf(msg, sizeof msg,
                      "%s leaves nothing for the DiT's weights: the VAE's weights take %.2f GiB and the DiT at %ux%u needs "
                      "%.2f GiB beside its weights (activations %.2f GiB, reserve %.2f GiB)",
                      budget, gib(vae_bytes), r.width, r.height, gib(need), gib(activations), gib(kReserve));
        throw std::runtime_error(msg);
      }
      dit_budget = std::min(dit_budget, r.vram_budget - vae_bytes - need);
    }
    DitOptions options;
    options.device_budget = std::max<std::uint64_t>(dit_budget, 1);  // 0 would mean every block resident
    std::optional<Dit> loaded;
    try {
      loaded.emplace(device, r.model_dir / "transformer", options);
    } catch (const std::runtime_error& e) {
      if (std::strstr(e.what(), "hipHostMalloc") == nullptr) throw;
      // A block is its GEMM weights in bf16, about 0.21 GiB (its norms are small); streamed ones live in pinned host memory.
      constexpr std::uint64_t kBlockBytes =
          (4 * std::uint64_t{kDiTDim} * kDiTDim + 3 * std::uint64_t{kDiTFfnDim} * kDiTDim) * kBf16;
      std::snprintf(msg, sizeof msg,
                    "the DiT's blocks past %.2f GiB of device memory (%s) stream from pinned host memory, up to %.2f GiB "
                    "for all %d blocks, and this machine could not pin it (%s)",
                    gib(dit_budget), budget, gib(kDiTLayers * kBlockBytes), kDiTLayers, e.what());
      throw std::runtime_error(msg);
    }
    Dit& dit = *loaded;
    out.dit_resident_blocks = dit.resident_blocks();
    // Refused before the first step rather than in it: a budget the weights cannot keep to (their globals and two
    // streaming slots stay on the device whatever the budget), or too little left beside them.
    if (r.vram_budget != 0 && vae_bytes + dit.weight_bytes() + need > r.vram_budget) {
      std::snprintf(msg, sizeof msg,
                    "the DiT's weights take at least %.2f GiB of device memory with every block streamed; with the VAE's "
                    "%.2f GiB and %.2f GiB beside them at %ux%u, that is more than %s",
                    gib(dit.weight_bytes()), gib(vae_bytes), gib(need), r.width, r.height, budget);
      throw std::runtime_error(msg);
    }
    if (const std::uint64_t left = device.free_memory(); need > left) {
      std::snprintf(msg, sizeof msg,
                    "the DiT's weights take %.2f GiB of device memory (%s); beside them the DiT at %ux%u needs %.2f GiB "
                    "(activations %.2f GiB, reserve %.2f GiB), %s has %.2f GiB free",
                    gib(dit.weight_bytes()), budget, r.width, r.height, gib(need), gib(activations), gib(kReserve),
                    device.info().name.c_str(), gib(left));
      throw std::runtime_error(msg);
    }
    std::vector<DeviceBuffer> texts;
    texts.reserve(embeds.size());  // the rows point into it
    std::vector<Dit::Row> rows;
    for (std::size_t i = 0; i < embeds.size(); ++i) {
      const auto length = static_cast<std::uint32_t>(embeds[i].size() / kDiTTextDim);
      texts.push_back(dit.encode_text(embeds[i], length));
      rows.push_back({&texts.back(), length, i == 0 ? SkipSet{} : r.uncond_skip});
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

  // VAE decode, after handing the final latents to the caller (a failed decode keeps them).
  if (r.on_latents) r.on_latents(out.latents);
  if (r.decode) {
    const double t0 = now_s();
    report("vae", 0, 1);
    vae->decode(out.latents.data(), lh, lw, out.rgb);
    out.vae_s = now_s() - t0;
    report("vae", 1, 1);
  }
  return out;
}

}  // namespace mi::hip
