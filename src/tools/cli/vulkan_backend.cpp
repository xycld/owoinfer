#include "tools/cli/vulkan_backend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <random>
#include <stdexcept>
#include <vector>

#include "mossinfer/mossinfer.h"
#include "pipeline/png_writer.h"
#include "pipeline/samplers/initial_noise.h"
#include "pipeline/samplers/schedules.h"
#include "tokenizer/prompt_tokens.h"
#include "tokenizer/qwen_bpe.h"

#if defined(MOSSINFER_VULKAN)
#include "generate.h"
#include "runtime.h"
#endif

namespace mi::cli {
namespace {

// What the Vulkan backend runs: txt2img with the official sampler and schedule. Anything else is refused by name.
void check_supported(const pipeline::Request& req) {
  const auto refuse = [](const char* what) {
    throw std::invalid_argument(std::string("the Vulkan backend does not support ") + what);
  };
  if (req.sampler != pipeline::SamplerKind::kEuler) refuse("--sampler (it runs the official euler sampler)");
  if (req.scheduler != pipeline::SchedulerKind::kMoss) refuse("--scheduler (it runs the official moss schedule)");
  if (req.flow_multiplier != 1.0) refuse("--flow-multiplier");
  if (req.noise_layout != pipeline::NoiseLayout::kOfficial) refuse("--noise-layout");
  if (!req.slg_layers.empty() && req.slg_scale != 0.0) refuse("--slg-layers");
  if (!req.loras.empty()) refuse("LoRAs");
  if (req.init_image || req.mask || req.denoise_strength != 1.0) refuse("img2img (--init-image, --mask, --strength)");
  if (req.hires.enabled()) refuse("--hires");
  if (req.control.enabled()) refuse("--control");
  if (req.references.enabled()) refuse("reference images");
  if (req.accel.active(req.cfg != 1.0)) refuse("accelerators (--cfg-truncation, --step-cache)");
  if (req.width % 16 != 0 || req.height % 16 != 0) refuse("sizes that are not multiples of 16");
}

}  // namespace

#if defined(MOSSINFER_VULKAN)

bool vulkan_backend_built() { return true; }

void run_vulkan_generate(const pipeline::Request& req, const VulkanRun& run) {
  check_supported(req);
  if (!run.kernels.empty() && run.kernels != "coop" && run.kernels != "plain") {
    throw std::invalid_argument("--vulkan-kernels takes coop or plain");
  }
  vk::Device device(run.device);
  const vk::Capabilities& caps = device.info().caps;
  const vk::BlockPath path = run.kernels.empty() ? (caps.coop_bf16 ? vk::BlockPath::kCoop : vk::BlockPath::kPlain)
                             : run.kernels == "coop" ? vk::BlockPath::kCoop
                                                     : vk::BlockPath::kPlain;
  // The choice is always reported: the plain path is much slower, and the user should know which one ran.
  std::fprintf(stderr, "vulkan: device %u, %s; kernels: %s%s\n", device.info().index, device.info().name.c_str(),
               vk::block_path_name(path),
               path == vk::BlockPath::kPlain && !caps.coop_bf16 ? (" (no bf16 cooperative matrices: " + caps.coop_bf16_reason + ")").c_str()
                                                                : "");

  const tok::Tokenizer tokenizer = tok::Tokenizer::load(req.model / "tokenizer" / "tokenizer.json");
  vk::GenerateRequest g;
  g.model_dir = req.model;
  tok::PromptTokenReport cond_report, uncond_report;
  g.cond_ids = tok::encode_moss_prompt_report(tokenizer, req.prompt, static_cast<std::size_t>(req.max_sequence_length),
                                              cond_report);
  const bool cfg = req.cfg != 1.0;
  if (cfg) {
    g.uncond_ids = tok::encode_moss_prompt_report(tokenizer, req.negative_prompt,
                                                  static_cast<std::size_t>(req.max_sequence_length), uncond_report);
  }
  for (const auto* t : {&cond_report, &uncond_report}) {
    if (t->truncated) {
      std::fprintf(stderr, "warning: %s truncated to %zu of %zu tokens (%zu dropped; --max-seq %zu)\n",
                   t == &cond_report ? "prompt" : "negative prompt", t->kept_tokens, t->total_tokens, t->dropped_tokens(),
                   t->max_length);
    }
  }

  std::uint64_t seed = 0;
  if (req.seed >= 0) {
    seed = static_cast<std::uint64_t>(req.seed);
  } else {
    std::random_device rd;
    seed = std::uniform_int_distribution<std::uint64_t>(0, (std::uint64_t{1} << 32) - 1)(rd);
  }
  g.width = static_cast<std::uint32_t>(req.width);
  g.height = static_cast<std::uint32_t>(req.height);
  g.noise = pipeline::initial_noise(pipeline::NoiseLayout::kOfficial, seed, vk::kDiTLatentChannels, req.height / 16,
                                    req.width / 16);
  const pipeline::Schedule schedule = pipeline::moss_schedule(req.steps, req.shift);
  g.sigmas = schedule.sigmas;
  g.timesteps = schedule.timesteps;
  g.cfg = static_cast<float>(req.cfg);
  for (const int l : req.uncond_skip_layers) g.uncond_skip.set(static_cast<std::size_t>(l));
  if (!run.quiet) {
    g.progress = [](const char* stage, int done, int total) {
      std::fprintf(stderr, "\r%s %d/%d   ", stage, done, total);
      if (done == total) std::fprintf(stderr, "\n");
    };
  }

  g.decode = !run.out_png.empty();
  g.on_latents = run.on_latents;
  vk::GenerateResult r;
  try {
    r = vk::generate(device, path, g);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(e.what());  // an internal check, not a usage error: the request was checked above
  }
  if (!run.out_png.empty()) {
    std::vector<pipeline::PngText> text;
    if (req.save_metadata) {
      pipeline::GenerationMetadata m;
      m.engine_version = mi_version();
      m.tier = "reference";
      m.models.push_back({"bundle", req.model.filename().string(), ""});
      m.prompt = req.prompt;
      m.negative_prompt = req.negative_prompt;
      m.sampler = "euler";
      m.scheduler = pipeline::scheduler_name(req.scheduler);
      m.steps = req.steps;
      m.cfg = req.cfg;
      m.shift = req.shift;
      m.seed = seed;
      m.width = req.width;
      m.height = req.height;
      m.extra.emplace_back("backend", std::string("vulkan ") + vk::block_path_name(path));
      text = pipeline::generation_text_chunks(m);
    }
    pipeline::write_png_rgb8(run.out_png, r.rgb, static_cast<int>(r.width), static_cast<int>(r.height), text);
  }
  if (!run.quiet) {
    std::fprintf(stderr, "seed %llu; text encoder %.1f s, DiT load %.1f s, denoise %.1f s (%.2f s/step), VAE %.1f s\n",
                 static_cast<unsigned long long>(seed), r.text_encoder_s, r.dit_load_s, r.denoise_s,
                 r.denoise_s / static_cast<double>(g.timesteps.size()), r.vae_s);
  }
}

#else

bool vulkan_backend_built() { return false; }

void run_vulkan_generate(const pipeline::Request& req, const VulkanRun&) {
  check_supported(req);
  throw std::runtime_error("this build has no Vulkan backend (configure with -DMOSSINFER_VULKAN=ON)");
}

#endif

}  // namespace mi::cli
