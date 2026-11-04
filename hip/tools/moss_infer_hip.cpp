// moss-infer-hip: text to image on the HIP backend, a standalone program for AMD GPUs through ROCm (Linux). It needs
// neither CUDA nor an NVIDIA driver, only ROCm 7's runtime; the binary holds the kernels of each architecture it was built
// for. Its flags are those of `moss-infer generate` that the backend runs, checked as the engine checks them; the
// tokenizer, the seed's initial noise, the schedule and the PNG writer are the engine's own sources, so a seed gives the
// same inputs as on CUDA. Exit status: 0 done, 1 failed, 2 a usage error or an unsupported request.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// The engine's headers before the HIP ones (generate.h): those define __noinline__ as a macro, which libstdc++ uses.
#include "mossinfer/mossinfer.h"
#include "pipeline/png_writer.h"
#include "pipeline/samplers/initial_noise.h"
#include "pipeline/samplers/schedules.h"
#include "tokenizer/prompt_tokens.h"
#include "tokenizer/qwen_bpe.h"
#include "util/path.h"
#include "util/regular_file.h"

#include "generate.h"
#include "hip_block.h"
#include "npy.h"
#include "runtime.h"

namespace {

namespace fs = std::filesystem;
using namespace mi;

// pipeline.h's request bounds (kMaxSide, kMaxPixels, kMaxSteps, kMaxSeed, kMaxVramBytes), which validate() applies;
// pipeline.h itself needs the CUDA headers.
constexpr int kMaxSide = 4096;
constexpr std::int64_t kMaxPixels = 4'200'000;
constexpr int kMaxSteps = 10000;
constexpr std::int64_t kMaxSeed = std::int64_t{1} << 53;
constexpr double kMaxVramBytes = static_cast<double>(std::uint64_t{1024} << 30);

constexpr const char* kUsage =
    "usage: moss-infer-hip --model <bundle_dir> --prompt <text> [-o <out.png>] [--latent-out <latents.npy>]\n"
    "                      [--negative <text>] [--size <W>x<H>] [--width N] [--height N] [--steps N] [--cfg X]\n"
    "                      [--shift X] [--seed N|-1] [--skip-layers \"22\"] [--max-seq N] [--tier reference]\n"
    "                      [--device N] [--vram-budget GiB] [--no-metadata] [--quiet]\n"
    "       moss-infer-hip --list-devices | --version | --help\n"
    "Text to image with MOSS-Image Base or SFT (the bundle's bf16 files) on an AMD GPU through ROCm, with the official\n"
    "euler sampler and schedule. The flags mean what they mean for `moss-infer generate`; the defaults are the same\n"
    "(1024x1024, 50 steps, cfg 3, shift 5, seed 42, skip layers 22). At least one of -o / --latent-out is required;\n"
    "without -o the VAE decode is skipped. --device picks a HIP device (default: the first one this build runs on).\n"
    "--vram-budget caps the device memory of the VAE and the DiT: their weights, the DiT's activations and the VAE's\n"
    "decode; the DiT's blocks past it stream from host memory every step (default: what the device has free). A run the\n"
    "budget or the device cannot hold is refused before its first step. The device and the build's kernels are printed\n"
    "at the start. Progress goes to stderr one line per event: `text encoder`, `denoise <step>/<steps> <seconds>`,\n"
    "`vae`, then a summary.\n";

struct Args {
  fs::path model, out_png, out_latent;
  std::string prompt, negative;
  bool have_prompt = false, metadata = true, quiet = false, list = false, help = false, version = false;
  int width = 1024, height = 1024, steps = 50, max_seq = 4096, device = -1;
  double cfg = 3.0, shift = 5.0;
  std::uint64_t vram_budget = 0;
  std::int64_t seed = 42;
  std::vector<int> skip = {22};
};

[[noreturn]] void usage(const std::string& what) { throw std::invalid_argument(what); }

double number(const std::string& flag, const std::string& v) {
  std::size_t used = 0;
  double d = 0;
  try {
    d = std::stod(v, &used);
  } catch (const std::exception&) {
    usage(flag + " takes a number, got " + v);
  }
  if (used != v.size() || !std::isfinite(d)) usage(flag + " takes a finite number, got " + v);
  return d;
}

std::int64_t integer(const std::string& flag, const std::string& v) {
  std::size_t used = 0;
  long long i = 0;
  try {
    i = std::stoll(v, &used);
  } catch (const std::exception&) {
    usage(flag + " takes an integer, got " + v);
  }
  if (used != v.size()) usage(flag + " takes an integer, got " + v);
  return i;
}

int int32(const std::string& flag, const std::string& v) {
  const std::int64_t i = integer(flag, v);
  if (i < std::numeric_limits<int>::min() || i > std::numeric_limits<int>::max()) usage(flag + " is out of range");
  return static_cast<int>(i);
}

std::vector<int> layers(const std::string& v) {
  std::vector<int> out;
  std::size_t at = 0;
  while (at < v.size()) {
    const std::size_t comma = v.find(',', at);
    std::string item = v.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
    item.erase(0, item.find_first_not_of(' '));
    item.erase(item.find_last_not_of(' ') + 1);
    if (!item.empty()) {
      const int l = int32("--skip-layers", item);
      if (l < 0 || l >= hip::kDiTLayers) usage("--skip-layers takes block indices in [0, 41]");
      out.push_back(l);
    }
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return out;
}

// pipeline::vram_option_bytes: GiB to bytes, with its bounds and messages.
std::uint64_t vram_bytes(const std::string& flag, double gib) {
  if (!std::isfinite(gib) || gib <= 0) usage(flag + " must be a finite positive number.");
  const double bytes = gib * static_cast<double>(std::uint64_t{1} << 30);
  if (!(bytes <= kMaxVramBytes)) usage(flag + " must be at most 1024 GiB.");
  if (bytes < static_cast<double>(std::uint64_t{1} << 20)) usage(flag + " must be at least 1 MiB.");
  return static_cast<std::uint64_t>(bytes);
}

Args parse(const std::vector<std::string>& argv) {
  Args a;
  for (std::size_t i = 1; i < argv.size(); ++i) {
    const std::string& f = argv[i];
    if (f == "--no-metadata") {
      a.metadata = false;
      continue;
    }
    if (f == "--quiet") {
      a.quiet = true;
      continue;
    }
    if (f == "--list-devices") {
      a.list = true;
      continue;
    }
    if (f == "-h" || f == "--help") {
      a.help = true;
      return a;
    }
    if (f == "--version") {
      a.version = true;
      return a;
    }
    if (i + 1 >= argv.size()) usage(f + " needs a value");
    const std::string& v = argv[++i];
    if (f == "--model") a.model = util::path_from_utf8(v);
    else if (f == "--prompt") a.prompt = v, a.have_prompt = true;
    else if (f == "--negative") a.negative = v;
    else if (f == "-o" || f == "--output") a.out_png = util::path_from_utf8(v);
    else if (f == "--latent-out") a.out_latent = util::path_from_utf8(v);
    else if (f == "--size") {
      const std::size_t x = v.find('x');
      if (x == std::string::npos) usage("--size takes <W>x<H>");
      a.width = int32("--size", v.substr(0, x));
      a.height = int32("--size", v.substr(x + 1));
    }
    else if (f == "--width") a.width = int32(f, v);
    else if (f == "--height") a.height = int32(f, v);
    else if (f == "--steps") a.steps = int32(f, v);
    else if (f == "--cfg") a.cfg = number(f, v);
    else if (f == "--shift") a.shift = number(f, v);
    else if (f == "--seed") a.seed = integer(f, v);
    else if (f == "--skip-layers") a.skip = layers(v);
    else if (f == "--max-seq") a.max_seq = int32(f, v);
    else if (f == "--device") a.device = int32(f, v);
    else if (f == "--vram-budget") a.vram_budget = vram_bytes(f, number(f, v));
    else if (f == "--tier") {
      // The backend runs the bf16 files exactly as the reference tier does; the lossy tiers need the CUDA kernels.
      if (v != "reference") usage("moss-infer-hip runs --tier reference only");
    }
    else usage("unknown or unsupported flag " + f);
  }
  if (a.list) return a;
  if (a.model.empty() || !a.have_prompt) usage("--model and --prompt are required");
  if (a.out_png.empty() && a.out_latent.empty()) usage("at least one of -o / --latent-out is required");
  if (a.device < -1) usage("--device takes a device index, or -1 for the first usable device");
  // pipeline::validate()'s checks and messages, for the settings this program takes.
  if (tok::python_strip(a.prompt).empty()) usage("Enter a prompt.");
  if (a.height <= 0 || a.width <= 0 || a.height % 16 != 0 || a.width % 16 != 0) {
    usage("Height and width must be positive multiples of 16.");
  }
  if (a.height > kMaxSide || a.width > kMaxSide) usage("Height and width must be at most " + std::to_string(kMaxSide) + ".");
  if (std::int64_t{a.height} * a.width > kMaxPixels) {
    usage("Image size must be at most 4.2 megapixels (e.g. 2048x2048); the model is meant for 0.25 to 4 megapixels.");
  }
  if (a.steps < 1) usage("Steps and image count must be positive.");
  if (a.steps > kMaxSteps) usage("Steps must be at most " + std::to_string(kMaxSteps) + ".");
  if (a.cfg < 0 || a.shift <= 0) usage("CFG must be finite and nonnegative; shift must be finite and positive.");
  if (a.seed < -1 || a.seed >= kMaxSeed) usage("Seed must be -1 (random) or an integer from 0 to 2^53-1 (browser-safe range).");
  if (a.max_seq < 1 || a.max_seq > static_cast<int>(tok::kMossMaxSequenceLength)) {
    usage("Maximum text length must be between 1 and 4096.");
  }
  return a;
}

int list_devices() {
  const std::vector<hip::DeviceInfo> devices = hip::list_devices();
  if (devices.empty()) std::printf("no HIP devices\n");
  for (const hip::DeviceInfo& d : devices) {
    std::printf("device %d: %s (%s), %.1f GiB\n", d.index, d.name.c_str(), d.arch.c_str(),
                static_cast<double>(d.total_memory) / static_cast<double>(1ull << 30));
    if (!d.unusable_reason.empty()) {
      std::printf("  unusable: %s\n", d.unusable_reason.c_str());
    } else {
      std::printf("  kernels: %s, %s\n", d.arch.c_str(), hip_block::leaf_name());
    }
  }
  return 0;
}

int run(const Args& a) {
  hip::Device device(a.device);
  // Always reported: which device and kernels ran should be visible (the NVIDIA build's scalar leaf is a logic check).
  std::fprintf(stderr, "hip: device %d, %s (%s); kernels: %s, %s\n", device.info().index, device.info().name.c_str(),
               device.info().arch.c_str(), hip::build_target(), hip_block::leaf_name());

  const tok::Tokenizer tokenizer = tok::Tokenizer::load(a.model / "tokenizer" / "tokenizer.json");
  hip::GenerateRequest g;
  g.model_dir = a.model;
  tok::PromptTokenReport cond_report, uncond_report;
  g.cond_ids = tok::encode_moss_prompt_report(tokenizer, a.prompt, static_cast<std::size_t>(a.max_seq), cond_report);
  const bool cfg = a.cfg != 1.0;
  if (cfg) {
    g.uncond_ids = tok::encode_moss_prompt_report(tokenizer, a.negative, static_cast<std::size_t>(a.max_seq), uncond_report);
  }
  for (const auto* t : {&cond_report, &uncond_report}) {
    if (t->truncated) {
      std::fprintf(stderr, "warning: %s truncated to %zu of %zu tokens (%zu dropped; --max-seq %zu)\n",
                   t == &cond_report ? "prompt" : "negative prompt", t->kept_tokens, t->total_tokens, t->dropped_tokens(),
                   t->max_length);
    }
  }

  std::uint64_t seed = 0;
  if (a.seed >= 0) {
    seed = static_cast<std::uint64_t>(a.seed);
  } else {  // as the engine: a random 32-bit seed, reported
    std::random_device rd;
    seed = std::uniform_int_distribution<std::uint64_t>(0, (std::uint64_t{1} << 32) - 1)(rd);
  }
  g.width = static_cast<std::uint32_t>(a.width);
  g.height = static_cast<std::uint32_t>(a.height);
  g.noise = pipeline::initial_noise(pipeline::NoiseLayout::kOfficial, seed, hip::kDiTLatentChannels, a.height / 16, a.width / 16);
  const pipeline::Schedule schedule = pipeline::moss_schedule(a.steps, a.shift);
  g.sigmas = schedule.sigmas;
  g.timesteps = schedule.timesteps;
  g.cfg = static_cast<float>(a.cfg);
  for (const int l : a.skip) g.uncond_skip.set(static_cast<std::size_t>(l));
  g.vram_budget = a.vram_budget;
  g.decode = !a.out_png.empty();
  if (!a.out_latent.empty()) {
    g.on_latents = [&](const std::vector<float>& latents) {
      hip::npy::save(a.out_latent, "<f4", {hip::kDiTLatentChannels, a.height / 16, a.width / 16}, latents.data(),
                     latents.size() * sizeof(float));
    };
  }
  double last = 0;
  if (!a.quiet) {
    g.progress = [&](const char* stage, int done, int total) {
      const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      if (std::string_view(stage) == "denoise" && done > 0) {
        std::fprintf(stderr, "denoise %d/%d %.2f s\n", done, total, now - last);
      } else if (done == 0) {
        std::fprintf(stderr, "%s\n", stage);
      }
      last = now;
    };
  }

  hip::GenerateResult r;
  try {
    r = hip::generate(device, g);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(e.what());  // an internal check, not a usage error: the request was checked above
  }
  if (!a.out_png.empty()) {
    std::vector<pipeline::PngText> text;
    if (a.metadata) {
      pipeline::GenerationMetadata m;
      m.engine_version = std::to_string(MI_VERSION_MAJOR) + "." + std::to_string(MI_VERSION_MINOR) + "." +
                         std::to_string(MI_VERSION_PATCH);
      m.tier = "reference";
      m.models.push_back({"bundle", util::path_utf8(util::path_key(a.model).filename()), ""});
      m.prompt = a.prompt;
      m.negative_prompt = a.negative;
      m.sampler = "euler";
      m.scheduler = pipeline::scheduler_name(pipeline::SchedulerKind::kMoss);
      m.steps = a.steps;
      m.cfg = a.cfg;
      m.shift = a.shift;
      m.seed = seed;
      m.width = a.width;
      m.height = a.height;
      m.extra.emplace_back("noise_layout", pipeline::noise_layout_name(pipeline::NoiseLayout::kOfficial));
      m.extra.emplace_back("backend", "hip " + device.info().arch);
      text = pipeline::generation_text_chunks(m);
    }
    pipeline::write_png_rgb8(a.out_png, r.rgb, static_cast<int>(r.width), static_cast<int>(r.height), text);
  }
  if (!a.quiet) {
    std::fprintf(stderr, "done: seed %llu; text encoder %.1f s, DiT load %.1f s (%d of 42 blocks on the device, %.2f GiB free "
                         "beside them), denoise %.1f s (%.2f s/step), VAE %.1f s\n",
                 static_cast<unsigned long long>(seed), r.text_encoder_s, r.dit_load_s, r.dit_resident_blocks,
                 static_cast<double>(r.dit_free_bytes) / static_cast<double>(1ull << 30), r.denoise_s,
                 r.denoise_s / static_cast<double>(g.timesteps.size()), r.vae_s);
  }
  return 0;
}

int run_main(const std::vector<std::string>& argv) {
  Args a;
  try {
    a = parse(argv);
  } catch (const std::invalid_argument& e) {
    if (e.what()[0] != '\0') std::fprintf(stderr, "moss-infer-hip: %s\n", e.what());
    std::fputs(kUsage, stderr);
    return 2;
  }
  if (a.help) {
    std::fputs(kUsage, stdout);
    return 0;
  }
  if (a.version) {
    std::printf("moss-infer-hip %d.%d.%d (kernels: %s)\n", MI_VERSION_MAJOR, MI_VERSION_MINOR, MI_VERSION_PATCH,
                hip::build_target());
    return 0;
  }
  try {
    return a.list ? list_devices() : run(a);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "moss-infer-hip: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "moss-infer-hip: %s\n", e.what());
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) { return run_main(std::vector<std::string>(argv, argv + argc)); }
