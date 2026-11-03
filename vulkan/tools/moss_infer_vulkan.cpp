// moss-infer-vulkan: text to image on the Vulkan backend, a standalone program for GPUs the CUDA engine does not support.
// It needs neither CUDA nor an NVIDIA driver, only a Vulkan driver. Its flags are those of `moss-infer generate` that the
// backend runs; the tokenizer, the seed's initial noise, the schedule and the PNG writer are the engine's own sources,
// so a seed gives the same inputs as on CUDA. Exit status: 0 done, 1 failed, 2 a usage error or an unsupported request.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "generate.h"
#include "mossinfer/mossinfer.h"
#include "npy.h"
#include "pipeline/png_writer.h"
#include "pipeline/samplers/initial_noise.h"
#include "pipeline/samplers/schedules.h"
#include "runtime.h"
#include "tokenizer/prompt_tokens.h"
#include "tokenizer/qwen_bpe.h"
#include "util/path.h"

namespace {

namespace fs = std::filesystem;
using namespace mi;

constexpr const char* kUsage =
    "usage: moss-infer-vulkan --model <bundle_dir> --prompt <text> [-o <out.png>] [--latent-out <latents.npy>]\n"
    "                         [--negative <text>] [--size <W>x<H>] [--width N] [--height N] [--steps N] [--cfg X]\n"
    "                         [--shift X] [--seed N|-1] [--skip-layers \"22\"] [--max-seq N] [--tier reference]\n"
    "                         [--device N] [--kernels coop|plain] [--no-metadata] [--quiet]\n"
    "       moss-infer-vulkan --list-devices\n"
    "Text to image with MOSS-Image Base or SFT (the bundle's bf16 files) on a Vulkan GPU, with the official euler sampler\n"
    "and schedule: the fallback for GPUs the CUDA engine (moss-infer) does not support. The flags mean what they mean for\n"
    "`moss-infer generate`; the defaults are the same (1024x1024, 50 steps, cfg 3, shift 5, seed 42, skip layers 22).\n"
    "At least one of -o / --latent-out is required; without -o the VAE decode is skipped. --device picks a Vulkan device\n"
    "(default: the first usable one, a discrete GPU first); --kernels picks the bf16 cooperative-matrix kernels or the\n"
    "plain ones (default: coop where the device has them). The device and kernels are printed at the start. Progress goes\n"
    "to stderr one line per event: `text encoder`, `denoise <step>/<steps> <seconds>`, `vae`, then a summary.\n";

struct Args {
  fs::path model, out_png, out_latent;
  std::string prompt, negative;
  bool have_prompt = false, metadata = true, quiet = false, list = false;
  int width = 1024, height = 1024, steps = 50, max_seq = 4096, device = -1;
  double cfg = 3.0, shift = 5.0;
  std::int64_t seed = 42;
  std::vector<int> skip = {22};
  std::string kernels;
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
      if (l < 0 || l >= vk::kDiTLayers) usage("--skip-layers takes block indices in [0, 41]");
      out.push_back(l);
    }
    if (comma == std::string::npos) break;
    at = comma + 1;
  }
  return out;
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
    if (f == "-h" || f == "--help") usage("");
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
    else if (f == "--kernels") a.kernels = v;
    else if (f == "--tier") {
      // The backend runs the bf16 files exactly as the reference tier does; the lossy tiers need the CUDA kernels.
      if (v != "reference") usage("moss-infer-vulkan runs --tier reference only");
    }
    else usage("unknown or unsupported flag " + f);
  }
  if (a.list) return a;
  if (a.model.empty() || !a.have_prompt) usage("--model and --prompt are required");
  if (a.out_png.empty() && a.out_latent.empty()) usage("at least one of -o / --latent-out is required");
  if (a.width < 16 || a.height < 16 || a.width % 16 != 0 || a.height % 16 != 0 || a.width > 4096 || a.height > 4096) {
    usage("width and height must be multiples of 16 in [16, 4096]");
  }
  if (a.steps < 1 || a.steps > 1000) usage("--steps must be in [1, 1000]");
  if (a.cfg < 0) usage("--cfg must be >= 0");
  if (a.shift <= 0) usage("--shift must be > 0");
  if (a.seed < -1 || a.seed >= (std::int64_t{1} << 53)) usage("--seed must be -1 (random) or in [0, 2^53)");
  if (a.max_seq < 1 || a.max_seq > static_cast<int>(tok::kMossMaxSequenceLength)) usage("--max-seq must be in [1, 4096]");
  if (!a.kernels.empty() && a.kernels != "coop" && a.kernels != "plain") usage("--kernels takes coop or plain");
  return a;
}

int list_devices() {
  const std::vector<vk::DeviceInfo> devices = vk::list_devices();
  if (devices.empty()) std::printf("no Vulkan devices\n");
  for (const vk::DeviceInfo& d : devices) {
    std::printf("device %u: %s [%04x:%04x]%s\n", d.index, d.name.c_str(), d.vendor_id, d.device_id,
                d.discrete ? " discrete" : "");
    if (!d.unusable_reason.empty()) {
      std::printf("  unusable: %s\n", d.unusable_reason.c_str());
    } else if (d.caps.coop_bf16) {
      std::printf("  kernels: coop\n");
    } else {
      std::printf("  kernels: plain (no coop: %s)\n", d.caps.coop_bf16_reason.c_str());
    }
  }
  return 0;
}

int run(const Args& a) {
  vk::Device device(a.device);
  const vk::Capabilities& caps = device.info().caps;
  const vk::BlockPath path = a.kernels.empty() ? (caps.coop_bf16 ? vk::BlockPath::kCoop : vk::BlockPath::kPlain)
                             : a.kernels == "coop" ? vk::BlockPath::kCoop
                                                   : vk::BlockPath::kPlain;
  // Always reported: the plain kernels are much slower, and which ones ran should be visible.
  std::fprintf(stderr, "vulkan: device %u, %s; kernels: %s%s\n", device.info().index, device.info().name.c_str(),
               vk::block_path_name(path),
               path == vk::BlockPath::kPlain && !caps.coop_bf16 ? (" (no coop: " + caps.coop_bf16_reason + ")").c_str() : "");

  const tok::Tokenizer tokenizer = tok::Tokenizer::load(a.model / "tokenizer" / "tokenizer.json");
  vk::GenerateRequest g;
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
  g.noise = pipeline::initial_noise(pipeline::NoiseLayout::kOfficial, seed, vk::kDiTLatentChannels, a.height / 16, a.width / 16);
  const pipeline::Schedule schedule = pipeline::moss_schedule(a.steps, a.shift);
  g.sigmas = schedule.sigmas;
  g.timesteps = schedule.timesteps;
  g.cfg = static_cast<float>(a.cfg);
  for (const int l : a.skip) g.uncond_skip.set(static_cast<std::size_t>(l));
  g.decode = !a.out_png.empty();
  if (!a.out_latent.empty()) {
    g.on_latents = [&](const std::vector<float>& latents) {
      vk::npy::save(a.out_latent, "<f4", {vk::kDiTLatentChannels, a.height / 16, a.width / 16}, latents.data(),
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

  vk::GenerateResult r;
  try {
    r = vk::generate(device, path, g);
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
      m.models.push_back({"bundle", util::path_utf8(a.model.filename()), ""});
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
      m.extra.emplace_back("backend", std::string("vulkan ") + vk::block_path_name(path));
      text = pipeline::generation_text_chunks(m);
    }
    pipeline::write_png_rgb8(a.out_png, r.rgb, static_cast<int>(r.width), static_cast<int>(r.height), text);
  }
  if (!a.quiet) {
    std::fprintf(stderr, "done: seed %llu; text encoder %.1f s, DiT load %.1f s (%d of 42 blocks on the device), denoise %.1f s "
                         "(%.2f s/step), VAE %.1f s\n",
                 static_cast<unsigned long long>(seed), r.text_encoder_s, r.dit_load_s, r.dit_resident_blocks, r.denoise_s,
                 r.denoise_s / static_cast<double>(g.timesteps.size()), r.vae_s);
  }
  return 0;
}

int run_main(const std::vector<std::string>& argv) {
  Args a;
  try {
    a = parse(argv);
  } catch (const std::invalid_argument& e) {
    if (e.what()[0] != '\0') std::fprintf(stderr, "moss-infer-vulkan: %s\n", e.what());
    std::fputs(kUsage, stderr);
    return 2;
  }
  try {
    return a.list ? list_devices() : run(a);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "moss-infer-vulkan: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "moss-infer-vulkan: %s\n", e.what());
    return 1;
  }
}

}  // namespace

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

// char** argv is in the ANSI code page on Windows and loses characters outside it: take the UTF-16 command line.
int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> utf8;
  for (int i = 0; i < argc; ++i) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? static_cast<std::size_t>(n) : 1, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, s.data(), n, nullptr, nullptr);
    s.pop_back();
    utf8.push_back(std::move(s));
  }
  return run_main(utf8);
}
#else
int main(int argc, char** argv) { return run_main(std::vector<std::string>(argv, argv + argc)); }
#endif
