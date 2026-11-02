// Runs the DiT's sampling loop on Vulkan from a golden case's text-encoder outputs and initial noise (the official PyTorch
// pipeline's, tests/golden), with CFG and the official Euler step on the host in fp32, and compares step 0's predictions
// and the final latent with the golden ones. Writes the final latent as .npy for the VAE.
// Usage: vulkan_dit_run TRANSFORMER_DIR GOLDEN_DIR CASE [--steps N] [--out FILE.npy] [--path coop|plain] [--device N]
//                       [--dit-budget GiB]   (device memory for the weights; the blocks past it stream)
//   GOLDEN_DIR is tests/golden, CASE e.g. g_fridge_1024; the case's cfg and uncond skip layers are 3.0 and {22}.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "block.h"
#include "dit.h"
#include "npy.h"
#include "runtime.h"

namespace {

using namespace mi::vk;
namespace fs = std::filesystem;

std::uint16_t to_bf16(float v) {
  const auto u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

std::vector<float> load_f32(const fs::path& p) { return npy::load(p).as<float>("<f4"); }

struct Compare {
  double cosine, max_abs, rms_ref, rms_diff;
  std::size_t nans;
};

Compare compare(const std::vector<float>& got, const std::vector<float>& ref) {
  double dot = 0, gg = 0, rr = 0, dd = 0, max_abs = 0;
  std::size_t nans = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(got[i])) ++nans;
    const double g = got[i], r = ref[i], d = g - r;
    dot += g * r;
    gg += g * g;
    rr += r * r;
    dd += d * d;
    max_abs = std::max(max_abs, std::abs(d));
  }
  const double n = static_cast<double>(got.size());
  return {dot / std::sqrt(gg * rr), max_abs, std::sqrt(rr / n), std::sqrt(dd / n), nans};
}

void print(const char* name, const Compare& c) {
  std::printf("  %-22s cosine %.9f, rms diff %.3e (rms of golden %.3e), max abs diff %.3e, NaN %zu\n", name, c.cosine,
              c.rms_diff, c.rms_ref, c.max_abs, c.nans);
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s TRANSFORMER_DIR GOLDEN_DIR CASE [--steps N] [--out FILE.npy] [--path coop|plain] [--device N]\n",
                argv[0]);
    return 2;
  }
  const fs::path transformer = argv[1], golden = argv[2];
  const std::string name = argv[3];
  int steps = -1, device_index = -1;
  double budget_gib = 0;
  fs::path out_path;
  std::string path_name;
  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
    else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (arg == "--path" && i + 1 < argc) path_name = argv[++i];
    else if (arg == "--device" && i + 1 < argc) device_index = std::atoi(argv[++i]);
    else if (arg == "--dit-budget" && i + 1 < argc) budget_gib = std::atof(argv[++i]);
    else {
      std::printf("unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  try {
    const fs::path dir = golden / name;
    const npy::Array noise = npy::load(dir / "initial_noise.npy");
    if (noise.shape.size() != 4 || noise.shape[1] != kDiTLatentChannels) throw std::runtime_error("unexpected noise shape");
    const auto height = static_cast<std::uint32_t>(noise.shape[2]), width = static_cast<std::uint32_t>(noise.shape[3]);
    std::vector<float> latents = noise.as<float>("<f4");
    const std::vector<float> sigmas = load_f32(dir / "sigmas.npy"), timesteps = load_f32(dir / "timesteps.npy");
    const int total_steps = static_cast<int>(timesteps.size());
    if (steps < 0 || steps > total_steps) steps = total_steps;
    constexpr float kCfg = 3.0f;
    SkipSet uncond_skip;
    uncond_skip.set(22);

    Device device(device_index);
    const BlockKernels kernels = path_name.empty() ? BlockKernels(device)
                                                   : BlockKernels(device, path_name == "plain" ? BlockPath::kPlain : BlockPath::kCoop);
    std::printf("device %u: %s, path %s\n", device.info().index, device.info().name.c_str(), block_path_name(kernels.path()));
    double t0 = now_s();
    DitOptions options;
    options.device_budget = static_cast<std::uint64_t>(budget_gib * (1u << 30));
    Dit dit(device, kernels, transformer, options);
    std::printf("weights: %.2f GiB on the device, %.2f GiB in host memory (%d of %d blocks resident), loaded in %.1f s\n",
                static_cast<double>(dit.weight_bytes()) / (1u << 30), static_cast<double>(dit.host_weight_bytes()) / (1u << 30),
                dit.resident_blocks(), kDiTLayers, now_s() - t0);

    std::vector<std::unique_ptr<Buffer>> texts;
    std::vector<Dit::Row> rows;
    for (const char* which : {"cond", "uncond"}) {
      const npy::Array te = npy::load(golden / (name + "." + which + ".te_last_hidden.npy"));
      const std::vector<float> values = te.as<float>("<f4");
      std::vector<std::uint16_t> bits(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) bits[i] = to_bf16(values[i]);  // exact: the encoder's bf16 output
      const auto length = static_cast<std::uint32_t>(te.shape[0]);
      t0 = now_s();
      texts.push_back(dit.encode_text(bits, length));
      std::printf("text refiner, %s: %u tokens in %.3f s\n", which, length, now_s() - t0);
      rows.push_back({texts.back().get(), length, std::string(which) == "uncond" ? uncond_skip : SkipSet{}});
    }

    const std::size_t n = latents.size();
    for (int s = 0; s < steps; ++s) {
      t0 = now_s();
      const std::vector<std::vector<float>> pred = dit.forward(latents, height, width, timesteps[static_cast<std::size_t>(s)], rows);
      const double dt_s = now_s() - t0;
      if (s == 0) {
        std::printf("step 0 against the golden predictions:\n");
        print("cond", compare(pred[0], load_f32(dir / "dit_step0_cond.npy")));
        print("uncond (skips 22)", compare(pred[1], load_f32(dir / "dit_step0_uncond.npy")));
      }
      // CFG then the official Euler step, the engine's op order (sampler_kernels.cu): u + g (c - u); x + h v.
      const float h = static_cast<float>(static_cast<double>(sigmas[static_cast<std::size_t>(s) + 1]) -
                                         static_cast<double>(sigmas[static_cast<std::size_t>(s)]));
      for (std::size_t i = 0; i < n; ++i) {
        const float diff = pred[0][i] - pred[1][i];
        const float scaled = kCfg * diff;
        const float v = pred[1][i] + scaled;
        const float step = h * v;
        latents[i] = latents[i] + step;
      }
      const std::vector<double>& ms = dit.last_op_ms();
      std::printf("step %2d: %.2f s (", s, dt_s);
      for (int op = 0; op < kBlockOps; ++op) std::printf("%s %.0f, ", block_op_name(op), ms[static_cast<std::size_t>(op)]);
      std::printf("rest %.0f ms, of which block copies %.0f ms)\n", ms[kBlockOps], dit.last_copy_ms());
      std::fflush(stdout);
    }
    if (steps == total_steps) {
      std::printf("final latent against the golden one:\n");
      print("final latent", compare(latents, load_f32(dir / "final_latent.npy")));
    }
    if (!out_path.empty()) {
      npy::save(out_path, "<f4", {1, kDiTLatentChannels, height, width}, latents.data(), latents.size() * sizeof(float));
      std::printf("wrote %s\n", out_path.string().c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
