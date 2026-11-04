// Runs the HIP text encoder on the golden token ids of the official PyTorch pipeline and compares the raw outputs of
// decoder layers 0, 3 and 31 (te_layerNN, before the final norm) and last_hidden_state (te_last_hidden) with them:
// cosine over the sequence, worst row cosine, row relative L2, share of identical bf16 bits, worst bf16 ULPs scaled to
// max(|golden|, rms(golden) / 64), max abs, NaN count. By default every case of tests/golden/cases.json (--case selects
// some), cond and uncond, all encoded together in one call (every sequence packed), as the engine test does.
//
// The gate is the engine's (tests/unit/test_qwen35_text.cpp), on every case and both prompts: no NaN; layers 0 and 3,
// before 32 layers amplify ULP-level differences, at cosine >= 0.99998 and row relative L2 <= 2e-2; layer 31 and
// last_hidden_state within 2.5 times torch's own noise floor (tests/data/qwen35_text/noise_floor.json: the golden
// against torch with its attention or its GEMM kernels swapped), in 1 - cosine and in mean row relative L2. A row
// threshold would not do there: on the 4096-token prompt torch's swapped attention alone decorrelates whole rows (row
// relative L2 0.96).
//
// The factor is the engine test's off-reference 2.5, not its 2.0. The engine is held to 2.0 on sm_89 because its
// cuBLASLt GEMMs and FlashAttention-2 round like torch's own kernels there: its layer 0 is almost torch's (mean row
// relative L2 5e-5). The HIP GEMMs and attention are a different MMA family (v_wmma on AMD, the scalar leaf in the
// nvidia build), whose summation order cannot reproduce NVIDIA's tensor cores, which is the situation of a different
// GPU. On the nvidia build every other op is bit-exact with the engine on the engine's inputs (norms, conv, gates, the
// gated delta rule, the gated norm, q/k norm + RoPE, the output gate; on AMD their exp, rsqrt and log1p are OCML's, so
// not by construction there), and the GEMMs and attention are as accurate as the engine's against fp64 (relative L2
// 1.7222e-3 both for a GEMM, 1.7006e-3 vs 1.7009e-3 for attention); they start about 10 times farther from torch's
// trajectory at layer 0 (5.3e-4), which the 32 layers amplify. On the RTX 4090 Laptop,
// g_long4096_1024.cond, as factors of the floor in 1 - cosine:
//   CUDA engine        layer 31 cos 0.9993189 (1.77x), last_hidden cos 0.9986887 (1.67x), mean row rel-L2 1.16x
//   HIP, nvidia build  layer 31 cos 0.9990446 (2.48x), last_hidden cos 0.9984760 (1.94x), mean row rel-L2 1.19x
// Both have rows decorrelated from the golden there (worst row cosine 0.37 engine, 0.45 HIP). The AMD gate is the same
// 2.5, fixed before any card run: a card past it is investigated, the factor is not retuned.
//
// --iters repeats the call for timing; --time-ops prints device milliseconds per kind of op; --row-major runs the GEMMs
// on row-major operands; --prefix N encodes at most the first N tokens of each sequence (compared with the golden's
// first rows: the model is causal); --solo encodes the prompts without the negative ones; --dump DIR writes every
// decoder layer's raw output (layerNN.npy, all sequences packed in the order of the call) and each sequence's
// last_hidden_state (<key>.te_last_hidden.npy), bf16 bits as uint16. Exit status: 0 passed, 1 failed, 2 no usable
// device.
// Usage: hip_te_check [--device N] [--weights DIR] [--golden DIR] [--case ID]... [--iters N] [--time-ops] [--row-major]
//                     [--dump DIR] [--noise-floor FILE] [--prefix N] [--solo]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "half.h"
#include "npy.h"
#include "runtime.h"
#include "text_encoder.h"
#include "util/json.h"

namespace {

using namespace mi::hip;

constexpr double kEarlyMinCosine = 0.99998, kEarlyMaxRowRelL2 = 2e-2, kFloorFactor = 2.5;

struct Metrics {
  double cosine = 0, min_row_cosine = 1, max_row_rel = 0, mean_row_rel = 0, same_pct = 0, max_ulps = 0, max_abs = 0;
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
      m.max_abs = std::max(m.max_abs, std::abs(g - w));
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
  if (m.nans != 0) m.cosine = m.min_row_cosine = std::nan("");
  return m;
}

void print(const std::string& what, const Metrics& m) {
  std::printf("  %-16s cos %.7f  min row cos %.6f  row relL2 max %.3e mean %.3e  same %6.2f %%  max %.1f ULPs  max abs %.4g  "
              "%zu NaN\n",
              what.c_str(), m.cosine, m.min_row_cosine, m.max_row_rel, m.mean_row_rel, m.same_pct, m.max_ulps, m.max_abs, m.nans);
}

// The torch noise floor of one sequence and checkpoint: the worst cosine and mean row relative L2 of its perturbed
// variants (math_sdpa, cublaslt), as the engine test reads it.
struct Floor {
  double cosine = 1, mean_row_rel = 0;
};
Floor noise_floor(const mi::json::Value& nf, const std::string& key, const std::string& checkpoint) {
  Floor f;
  const mi::json::Value* seq = nf.at("sequences").find(key);
  if (seq == nullptr) throw std::runtime_error("noise floor has no sequence " + key);
  for (const char* variant : {"math_sdpa", "cublaslt"}) {
    const mi::json::Value& m = seq->at(variant).at(checkpoint);
    f.cosine = std::min(f.cosine, m.at("cosine").as_double());
    f.mean_row_rel = std::max(f.mean_row_rel, m.at("mean_row_rel_l2").as_double());
  }
  return f;
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

bool within_floor(const Metrics& m, const Floor& f) {
  const bool ok = m.nans == 0 && m.cosine >= 1 - kFloorFactor * (1 - f.cosine) && m.mean_row_rel <= kFloorFactor * f.mean_row_rel;
  std::printf("  %-16s torch floor cos %.7f, row relL2 mean %.3e: deviation %.2fx and %.2fx the floor's (limit %.1fx) %s\n", "",
              f.cosine, f.mean_row_rel, (1 - m.cosine) / (1 - f.cosine), m.mean_row_rel / f.mean_row_rel, kFloorFactor,
              ok ? "ok" : "FAIL");
  return ok;
}

std::vector<float> golden(const std::string& dir, const std::string& key, const std::string& what, std::size_t rows) {
  const npy::Array a = npy::load(dir + "/" + key + "." + what + ".npy");
  if (a.shape.size() != 2 || a.shape[0] < static_cast<std::int64_t>(rows) || a.shape[1] != kTeHidden) {
    throw std::runtime_error("unexpected golden shape: " + key + "." + what);
  }
  std::vector<float> v = a.to_vector<float>();
  v.resize(rows * kTeHidden);  // --prefix: the golden's first rows (the model is causal)
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  int device_index = -1, iters = 1;
  TextEncoderOptions options;
  std::string weights = "/workspace/Models/MOSS-Image/base/text_encoder", golden_dir = "tests/golden", dump;
  std::string floor_file = "tests/data/qwen35_text/noise_floor.json";
  std::size_t prefix = 0;
  bool solo = false;
  std::vector<std::string> cases;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) {
      device_index = std::atoi(argv[++i]);
    } else if (arg == "--weights" && i + 1 < argc) {
      weights = argv[++i];
    } else if (arg == "--golden" && i + 1 < argc) {
      golden_dir = argv[++i];
    } else if (arg == "--case" && i + 1 < argc) {
      cases.emplace_back(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--time-ops") {
      options.time_ops = true;
    } else if (arg == "--row-major") {
      options.packed = false;
    } else if (arg == "--noise-floor" && i + 1 < argc) {
      floor_file = argv[++i];
    } else if (arg == "--prefix" && i + 1 < argc) {
      prefix = static_cast<std::size_t>(std::atoll(argv[++i]));
    } else if (arg == "--solo") {
      solo = true;
    } else if (arg == "--dump" && i + 1 < argc) {
      dump = argv[++i];
    } else {
      std::printf(
          "usage: %s [--device N] [--weights DIR] [--golden DIR] [--case ID]... [--iters N] [--time-ops] [--row-major] [--dump DIR] "
          "[--noise-floor FILE] [--prefix N] [--solo]\n",
          argv[0]);
      return 2;
    }
  }
  try {
    bool usable = false;
    for (const DeviceInfo& d : list_devices()) usable = usable || d.unusable_reason.empty();
    if (!usable) {
      std::printf("no device runs this build (%s)\n", build_target());
      return 2;
    }
    const mi::json::Value nf = mi::json::parse(read_file(floor_file));
    if (cases.empty()) {
      const mi::json::Value all = mi::json::parse(read_file(golden_dir + "/cases.json"));
      for (const mi::json::Value& c : all.as_array()) {
        cases.push_back(c.at("id").as_string());
      }
    }
    std::vector<std::string> keys;
    std::vector<std::vector<std::int32_t>> seqs;
    for (const std::string& id : cases) {
      for (const char* branch : {"cond", "uncond"}) {
        if (solo && std::string(branch) == "uncond") continue;
        keys.push_back(id + "." + branch);
        const auto ids = npy::load(golden_dir + "/" + keys.back() + ".token_ids.npy").to_vector<std::int64_t>();
        seqs.emplace_back(ids.begin(), ids.end());
        if (prefix > 0 && prefix < seqs.back().size()) seqs.back().resize(prefix);
      }
    }
    Device device(device_index);
    TextEncoder encoder(device, weights, options);
    std::printf("device %d: %s (%s), build %s, GEMMs on %s operands; %zu sequences in one call\n", device.info().index,
                device.info().name.c_str(), device.info().arch.c_str(), build_target(), options.packed ? "packed" : "row-major",
                seqs.size());
    std::map<int, std::vector<std::uint16_t>> layers;
    const TeLayerHook hook = [&](int layer, const std::uint16_t* hidden, std::uint32_t tokens, hipStream_t stream) {
      if (layer != 0 && layer != 3 && layer != 31 && dump.empty()) return;
      auto& dst = layers[layer];
      dst.resize(static_cast<std::size_t>(tokens) * kTeHidden);
      check(hipStreamSynchronize(stream), "hipStreamSynchronize");
      check(hipMemcpy(dst.data(), hidden, dst.size() * 2, hipMemcpyDeviceToHost), "hipMemcpy");
      if (!dump.empty()) {
        char name[32];
        std::snprintf(name, sizeof(name), "/layer%02d.npy", layer);
        npy::save(dump + name, "<u2", {static_cast<std::int64_t>(tokens), kTeHidden}, dst.data(), dst.size() * 2);
      }
    };
    std::vector<std::vector<std::uint16_t>> out;
    for (int it = 0; it < iters; ++it) {
      out = encoder.encode(seqs, it == 0 ? hook : TeLayerHook{});
      const TextEncoderStats& s = encoder.last_stats();
      std::printf(
          "%u tokens, encode %.3f s (embed %.3f, file reads %.3f in the reader, waiting for them %.3f; device: layers "
          "%.1f ms, of which waiting for weights %.1f ms)%s, device %.1f MiB, pinned %.1f MiB\n",
          s.tokens, s.total_s, s.embed_s, s.read_s, s.read_wait_s, s.layers_ms, s.weight_wait_ms,
          it == 0 ? " [layer hook syncs]" : "", static_cast<double>(s.device_bytes) / (1 << 20),
          static_cast<double>(s.pinned_bytes) / (1 << 20));
      for (const auto& [op, ms] : s.op_ms) std::printf("  %-24s %10.2f ms\n", op.c_str(), ms);
    }
    bool ok = true;
    std::size_t row0 = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      const std::size_t rows = seqs[i].size();
      std::printf("%s (%zu tokens)\n", keys[i].c_str(), rows);
      for (const int layer : {0, 3, 31}) {
        char name[16];
        std::snprintf(name, sizeof(name), "te_layer%02d", layer);
        const Metrics m = compare(layers[layer].data() + row0 * kTeHidden, golden(golden_dir, keys[i], name, rows));
        print(name, m);
        if (layer < 31) {
          const bool pass = m.nans == 0 && m.cosine >= kEarlyMinCosine && m.max_row_rel <= kEarlyMaxRowRelL2;
          std::printf("  %-16s cos >= %.5f, row relL2 max <= %.0e: %s\n", "", kEarlyMinCosine, kEarlyMaxRowRelL2,
                      pass ? "ok" : "FAIL");
          ok = ok && pass;
        } else {
          ok = within_floor(m, noise_floor(nf, keys[i], name)) && ok;
        }
      }
      if (!dump.empty()) {
        npy::save(dump + "/" + keys[i] + ".te_last_hidden.npy", "<u2", {static_cast<std::int64_t>(rows), kTeHidden},
                  out[i].data(), out[i].size() * 2);
      }
      const Metrics m = compare(out[i].data(), golden(golden_dir, keys[i], "te_last_hidden", rows));
      print("te_last_hidden", m);
      ok = within_floor(m, noise_floor(nf, keys[i], "te_last_hidden")) && ok;
      row0 += rows;
    }
    std::printf("%s\n", ok ? "all passed" : "FAILED");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
