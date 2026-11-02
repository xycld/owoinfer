// Runs the Vulkan text encoder on the golden token ids of the official PyTorch pipeline (tests/golden) and compares the
// raw outputs of decoder layers 0, 3 and 31 (te_layerNN, before the final norm) and last_hidden_state (te_last_hidden)
// with them: cosine over the sequence, worst row cosine, row relative L2, share of identical bf16 bits, worst bf16 ULPs
// scaled to max(|golden|, rms(golden) / 64), NaN count. The cond and uncond prompts of a case are encoded in one call;
// --iters repeats the call for timing; --profile submits each op alone and prints seconds per kind of op. Exit status: 0
// ran without NaN, 1 otherwise.
// Usage: vulkan_te_check [--device N] [--path coop|plain] [--weights DIR] [--golden DIR] [--case ID]... [--iters N] [--profile]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <string>
#include <vector>

#include "half.h"
#include "npy.h"
#include "runtime.h"
#include "text_encoder.h"

namespace {

using namespace mi::vk;

struct Metrics {
  double cosine = 0, min_row_cosine = 1, max_row_rel = 0, mean_row_rel = 0, same_pct = 0, max_ulps = 0;
  std::size_t nans = 0;
};

Metrics compare(const std::uint16_t* got, const std::vector<float>& ref) {
  Metrics m;
  const std::size_t rows = ref.size() / kTeHidden;
  double rr_all = 0;
  for (const float v : ref) rr_all += static_cast<double>(v) * v;
  const double floor_mag = std::sqrt(rr_all / static_cast<double>(ref.size())) / 64.0;
  double dot = 0, gg = 0, rr = 0;
  std::size_t same = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    double rdot = 0, rg = 0, rref = 0, rdiff = 0;
    for (std::size_t c = 0; c < kTeHidden; ++c) {
      const std::size_t i = r * kTeHidden + c;
      const double g = bf16_to_float(got[i]), w = ref[i];
      if (std::isnan(g)) ++m.nans;
      same += got[i] == float_to_bf16(ref[i]);
      rdot += g * w;
      rg += g * g;
      rref += w * w;
      rdiff += (g - w) * (g - w);
      int e = 0;
      (void)std::frexp(std::max(std::abs(w), floor_mag), &e);
      m.max_ulps = std::max(m.max_ulps, std::abs(g - w) / std::ldexp(1.0, e - 8));
    }
    dot += rdot;
    gg += rg;
    rr += rref;
    m.min_row_cosine = std::min(m.min_row_cosine, rdot / std::sqrt(rg * rref));
    m.max_row_rel = std::max(m.max_row_rel, std::sqrt(rdiff / rref));
    m.mean_row_rel += std::sqrt(rdiff / rref) / static_cast<double>(rows);
  }
  m.cosine = dot / std::sqrt(gg * rr);
  m.same_pct = 100.0 * static_cast<double>(same) / static_cast<double>(ref.size());
  return m;
}

void print(const std::string& what, const Metrics& m) {
  std::printf("  %-16s cos %.7f  min row cos %.6f  row relL2 max %.3e mean %.3e  same %6.2f %%  max %.1f ULPs  %zu NaN\n",
              what.c_str(), m.cosine, m.min_row_cosine, m.max_row_rel, m.mean_row_rel, m.same_pct, m.max_ulps, m.nans);
}

std::vector<float> golden(const std::string& dir, const std::string& key, const std::string& what, std::size_t rows) {
  const npy::Array a = npy::load(dir + "/" + key + "." + what + ".npy");
  if (a.shape != std::vector<std::int64_t>{static_cast<std::int64_t>(rows), kTeHidden}) {
    throw std::runtime_error("unexpected golden shape: " + key + "." + what);
  }
  return a.to_vector<float>();
}

}  // namespace

int main(int argc, char** argv) {
  int device_index = -1, iters = 1;
  TextEncoderOptions options;
  std::string path_name, weights = "/workspace/Models/MOSS-Image/base/text_encoder", golden_dir = "tests/golden";
  std::vector<std::string> cases;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) {
      device_index = std::atoi(argv[++i]);
    } else if (arg == "--path" && i + 1 < argc) {
      path_name = argv[++i];
    } else if (arg == "--weights" && i + 1 < argc) {
      weights = argv[++i];
    } else if (arg == "--golden" && i + 1 < argc) {
      golden_dir = argv[++i];
    } else if (arg == "--case" && i + 1 < argc) {
      cases.emplace_back(argv[++i]);
    } else if (arg == "--profile") {
      options.profile = true;
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::max(1, std::atoi(argv[++i]));
    } else {
      std::printf("usage: %s [--device N] [--path coop|plain] [--weights DIR] [--golden DIR] [--case ID]... [--iters N] [--profile]\n", argv[0]);
      return 2;
    }
  }
  if (cases.empty()) cases.emplace_back("g_fridge_1024");
  if (!path_name.empty() && path_name != "coop" && path_name != "plain") {
    std::printf("--path takes coop or plain\n");
    return 2;
  }
  try {
    Device device(device_index);
    const BlockPath path = path_name.empty() ? (device.info().caps.coop_bf16 ? BlockPath::kCoop : BlockPath::kPlain)
                                             : (path_name == "coop" ? BlockPath::kCoop : BlockPath::kPlain);
    TextEncoder encoder(device, weights, path, options);
    std::printf("device %u: %s, path %s\n", device.info().index, device.info().name.c_str(), block_path_name(encoder.path()));
    bool ok = true;
    for (const std::string& id : cases) {
      const std::vector<std::string> keys = {id + ".cond", id + ".uncond"};
      std::vector<std::vector<std::int32_t>> seqs;
      for (const std::string& key : keys) {
        const auto ids = npy::load(golden_dir + "/" + key + ".token_ids.npy").to_vector<std::int64_t>();
        seqs.emplace_back(ids.begin(), ids.end());
      }
      std::map<int, std::vector<std::uint16_t>> layers;
      const TeLayerHook hook = [&](int layer, Buffer& hidden, std::uint32_t tokens) {
        if (layer != 0 && layer != 3 && layer != 31) return;
        auto& dst = layers[layer];
        dst.resize(static_cast<std::size_t>(tokens) * kTeHidden);
        hidden.download(dst.data(), dst.size() * 2);
      };
      std::vector<std::vector<std::uint16_t>> out;
      for (int it = 0; it < iters; ++it) {
        out = encoder.encode(seqs, it == 0 ? hook : TeLayerHook{});
        const TextEncoderStats& s = encoder.last_stats();
        std::printf("%s: %u tokens, encode %.3f s (embed %.3f, file reads %.3f in the reader thread, waiting for them %.3f, "
                    "uploads %.3f, compute %.3f, %d submits), device %.1f MiB, host staging %.1f MiB\n",
                    id.c_str(), s.tokens, s.total_s, s.embed_s, s.read_s, s.read_wait_s, s.upload_s, s.compute_s, s.submits,
                    static_cast<double>(s.device_bytes) / (1 << 20), static_cast<double>(s.staging_bytes) / (1 << 20));
        for (const auto& [op, seconds] : s.op_seconds) std::printf("  %-24s %8.3f s\n", op.c_str(), seconds);
      }
      std::size_t row0 = 0;
      for (std::size_t i = 0; i < keys.size(); ++i) {
        const std::size_t rows = seqs[i].size();
        std::printf("%s (%zu tokens)\n", keys[i].c_str(), rows);
        for (const int layer : {0, 3, 31}) {
          char name[16];
          std::snprintf(name, sizeof(name), "te_layer%02d", layer);
          const Metrics m = compare(layers[layer].data() + row0 * kTeHidden, golden(golden_dir, keys[i], name, rows));
          print(name, m);
          ok = ok && m.nans == 0;
        }
        const Metrics m = compare(out[i].data(), golden(golden_dir, keys[i], "te_last_hidden", rows));
        print("te_last_hidden", m);
        ok = ok && m.nans == 0;
        row0 += rows;
      }
    }
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
