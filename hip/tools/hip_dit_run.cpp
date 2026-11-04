// Runs the DiT's sampling loop on HIP from a golden case's text-encoder outputs and initial noise (the official PyTorch
// pipeline's, tests/golden), with CFG and the official Euler step on the host in fp32, and compares step 0's predictions
// and the final latent with the golden ones. Writes the final latent as .npy for the VAE.
// Usage: hip_dit_run TRANSFORMER_DIR GOLDEN_DIR CASE [--steps N] [--out FILE.npy] [--device N]
//                    [--dit-budget GiB]   (device memory for the weights; the blocks past it stream)
//                    [--time-ops]         (per-op device times of each step)
//                    [--expect FILE.npy]  (the latent after the steps run must have exactly these bits)
//   GOLDEN_DIR is tests/golden, CASE e.g. g_fridge_1024. The case's cfg comes from GOLDEN_DIR/cases.json and the uncond
//   row's skipped blocks from CASE/dit_step0_uncond.skip.json, as the engine's test reads them.
// Gates, each failing the run:
//   - step 0, cond and uncond against the golden predictions: cosine >= 0.99988 (the CUDA engine's own test's) and rms of
//     the difference <= 0.0155 of the golden rms (what that cosine allows between vectors of equal norm, sqrt(2 (1 - c)));
//   - a NaN in any step's predictions or latent;
//   - with every step run, the final latent against the golden one: cosine >= 0.95 and rms difference <= 0.316 of the
//     golden rms (sqrt(2 x 0.05)). A gross-error gate: 50 steps of sampling amplify small differences, so a correct DiT
//     lands well below step 0's agreement. The Vulkan DiT on these golden text-encoder outputs reaches 0.957 (coop) and
//     0.971 (plain), and two runs differing only in their text encoder (cosine 0.9998 between those) ended 0.9706 apart;
//     the CUDA engine reaches 0.992. The bound was set from those, before any HIP figure;
//   - with --expect, any bit of the latent after the steps differing from the file's.
// Exit status: 0 when every gate passes, 1 otherwise or on an error, 2 without a usable device.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "dit.h"
#include "half.h"
#include "hip_block.h"
#include "npy.h"
#include "runtime.h"
#include "util/json.h"

namespace {

using namespace mi::hip;
namespace fs = std::filesystem;

constexpr double kStep0Cosine = 0.99988, kStep0RelRms = 0.0155;
constexpr double kFinalCosine = 0.95, kFinalRelRms = 0.316;

mi::json::Value read_json(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + p.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return mi::json::parse(ss.str());
}

bool has_nan(const std::vector<float>& v) {
  for (const float x : v) {
    if (std::isnan(x)) return true;
  }
  return false;
}

std::vector<float> load_f32(const fs::path& p) { return npy::load(p).to_vector<float>(); }

struct Compare {
  double cosine, max_abs, rms_ref, rms_diff;
  std::size_t nans;
};

Compare compare(const std::vector<float>& got, const std::vector<float>& ref) {
  if (got.size() != ref.size()) throw std::runtime_error("compare: sizes differ");
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

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s TRANSFORMER_DIR GOLDEN_DIR CASE [--steps N] [--out FILE.npy] [--device N] [--dit-budget GiB] "
                "[--time-ops]\n",
                argv[0]);
    return 1;
  }
  const fs::path transformer = argv[1], golden = argv[2];
  const std::string name = argv[3];
  int steps = -1, device_index = -1;
  double budget_gib = 0;
  bool time_ops = false;
  fs::path out_path, expect_path;
  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
    else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (arg == "--device" && i + 1 < argc) device_index = std::atoi(argv[++i]);
    else if (arg == "--dit-budget" && i + 1 < argc) budget_gib = std::atof(argv[++i]);
    else if (arg == "--time-ops") time_ops = true;
    else if (arg == "--expect" && i + 1 < argc) expect_path = argv[++i];
    else {
      std::printf("unknown argument %s\n", arg.c_str());
      return 1;
    }
  }
  try {
    bool usable = false;
    for (const DeviceInfo& d : list_devices()) usable = usable || d.unusable_reason.empty();
    if (!usable && device_index < 0) {
      std::printf("no device runs this build (%s)\n", build_target());
      return 2;
    }
    const fs::path dir = golden / name;
    const npy::Array noise = npy::load(dir / "initial_noise.npy");
    if (noise.shape.size() != 4 || noise.shape[1] != kDiTLatentChannels) throw std::runtime_error("unexpected noise shape");
    const auto height = static_cast<std::uint32_t>(noise.shape[2]), width = static_cast<std::uint32_t>(noise.shape[3]);
    std::vector<float> latents = noise.to_vector<float>();
    const std::vector<float> sigmas = load_f32(dir / "sigmas.npy"), timesteps = load_f32(dir / "timesteps.npy");
    if (timesteps.empty() || sigmas.size() != timesteps.size() + 1) {
      throw std::runtime_error("sigmas.npy must hold one value more than timesteps.npy");
    }
    const int total_steps = static_cast<int>(timesteps.size());
    if (steps < 0 || steps > total_steps) steps = total_steps;
    // The case's cfg (cases.json) and the uncond row's skipped blocks (its skip.json, which must agree with cases.json).
    const mi::json::Value* entry = nullptr;
    const mi::json::Value cases = read_json(golden / "cases.json");
    for (const mi::json::Value& c : cases.as_array()) {
      if (c.at("id").as_string() == name) entry = &c;
    }
    if (entry == nullptr) throw std::runtime_error(name + " is not in " + (golden / "cases.json").string());
    const auto cfg = static_cast<float>(entry->at("cfg").as_double());
    SkipSet uncond_skip;
    std::string skip_text;
    const mi::json::Value skip_json = read_json(dir / "dit_step0_uncond.skip.json");
    for (const mi::json::Value& l : skip_json.as_array()) {
      const std::int64_t layer = l.as_int();
      if (layer < 0 || layer >= kDiTLayers) throw std::runtime_error("skip layer out of range");
      uncond_skip.set(static_cast<std::size_t>(layer));
      skip_text += (skip_text.empty() ? "" : ",") + std::to_string(layer);
    }
    SkipSet listed;
    for (const mi::json::Value& l : entry->at("uncond_skip_layers").as_array()) listed.set(static_cast<std::size_t>(l.as_int()));
    if (listed != uncond_skip) throw std::runtime_error("skip.json and cases.json name different uncond skip layers");
    std::printf("case %s: %ux%u latent, %d of %d steps, cfg %.2f, uncond skips {%s}\n", name.c_str(), height, width, steps,
                total_steps, static_cast<double>(cfg), skip_text.c_str());

    Device device(device_index);
    std::printf("device %d: %s (%s), build %s, leaf %s, %.2f GiB free\n", device.info().index, device.info().name.c_str(),
                device.info().arch.c_str(), build_target(), hip_block::leaf_name(),
                static_cast<double>(device.free_memory()) / (1u << 30));
    double t0 = now_s();
    DitOptions options;
    options.device_budget = static_cast<std::uint64_t>(budget_gib * (1u << 30));
    options.time_ops = time_ops;
    Dit dit(device, transformer, options);
    std::printf("weights: %.2f GiB on the device, %.2f GiB in pinned host memory (%d of %d blocks resident), loaded in %.1f s\n",
                static_cast<double>(dit.weight_bytes()) / (1u << 30), static_cast<double>(dit.host_weight_bytes()) / (1u << 30),
                dit.resident_blocks(), kDiTLayers, now_s() - t0);

    std::vector<DeviceBuffer> texts;
    texts.reserve(2);
    std::vector<Dit::Row> rows;
    for (const char* which : {"cond", "uncond"}) {
      const npy::Array te = npy::load(golden / (name + "." + which + ".te_last_hidden.npy"));
      const std::vector<float> values = te.to_vector<float>();
      std::vector<std::uint16_t> bits(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) bits[i] = float_to_bf16(values[i]);  // exact: the encoder's bf16 output
      const auto length = static_cast<std::uint32_t>(te.shape[0]);
      t0 = now_s();
      texts.push_back(dit.encode_text(bits, length));
      std::printf("text refiner, %s: %u tokens in %.3f s\n", which, length, now_s() - t0);
      rows.push_back({&texts.back(), length, std::string(which) == "uncond" ? uncond_skip : SkipSet{}});
    }

    bool pass = steps > 0;
    const std::size_t n = latents.size();
    double total_s = 0;
    for (int s = 0; s < steps; ++s) {
      t0 = now_s();
      const std::vector<std::vector<float>> pred = dit.forward(latents, height, width, timesteps[static_cast<std::size_t>(s)], rows);
      const double dt_s = now_s() - t0;
      total_s += dt_s;
      if (s == 0) {
        std::printf("step 0 against the golden predictions (pass: cosine >= %.5f, rms diff <= %.4f of golden):\n", kStep0Cosine,
                    kStep0RelRms);
        const Compare c = compare(pred[0], load_f32(dir / "dit_step0_cond.npy"));
        const Compare u = compare(pred[1], load_f32(dir / "dit_step0_uncond.npy"));
        print("cond", c);
        print(("uncond (skips " + skip_text + ")").c_str(), u);
        for (const Compare& x : {c, u}) {
          pass = pass && x.cosine >= kStep0Cosine && x.rms_diff <= kStep0RelRms * x.rms_ref && x.nans == 0;
        }
      }
      if (has_nan(pred[0]) || has_nan(pred[1])) {
        std::printf("step %d: NaN in a prediction\n", s);
        pass = false;
      }
      // CFG then the official Euler step, the engine's op order (sampler_kernels.cu): u + g (c - u); x + h v.
      const float h = static_cast<float>(static_cast<double>(sigmas[static_cast<std::size_t>(s) + 1]) -
                                         static_cast<double>(sigmas[static_cast<std::size_t>(s)]));
      for (std::size_t i = 0; i < n; ++i) {
        const float diff = pred[0][i] - pred[1][i];
        const float scaled = cfg * diff;
        const float v = pred[1][i] + scaled;
        const float step = h * v;
        latents[i] = latents[i] + step;
      }
      std::printf("step %2d: %.2f s", s, dt_s);
      const std::vector<double>& ms = dit.last_op_ms();
      if (!ms.empty()) {
        std::printf(" (device ms:");
        for (int op = 0; op < hip_block::kBlockOps; ++op) std::printf(" %s %.0f,", hip_block::block_op_name(op), ms[static_cast<std::size_t>(op)]);
        std::printf(" copy waits %.0f, rest %.0f)", ms[hip_block::kBlockOps], ms[hip_block::kBlockOps + 1]);
      }
      std::printf("\n");
      std::fflush(stdout);
      if (has_nan(latents)) {
        std::printf("step %d: NaN in the latent\n", s);
        pass = false;
      }
    }
    std::printf("%d steps in %.1f s, %.2f s per step\n", steps, total_s, steps > 0 ? total_s / steps : 0.0);
    if (steps == total_steps) {
      std::printf("final latent against the golden one (pass: cosine >= %.2f, rms diff <= %.3f of golden):\n", kFinalCosine,
                  kFinalRelRms);
      const Compare f = compare(latents, load_f32(dir / "final_latent.npy"));
      print("final latent", f);
      pass = pass && f.cosine >= kFinalCosine && f.rms_diff <= kFinalRelRms * f.rms_ref && f.nans == 0;
    }
    if (!expect_path.empty()) {
      const npy::Array e = npy::load(expect_path);
      const std::vector<float> want = e.to_vector<float>();
      std::size_t differ = want.size() == latents.size() ? 0 : latents.size();
      if (differ == 0) {
        for (std::size_t i = 0; i < want.size(); ++i) differ += std::memcmp(&want[i], &latents[i], sizeof(float)) != 0;
      }
      std::printf("latent after %d steps against %s: %zu of %zu values differ in their bits\n", steps,
                  expect_path.string().c_str(), differ, latents.size());
      pass = pass && differ == 0;
    }
    if (!out_path.empty()) {
      npy::save(out_path, "<f4", {1, kDiTLatentChannels, height, width}, latents.data(), latents.size() * sizeof(float));
      std::printf("wrote %s\n", out_path.string().c_str());
    }
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
