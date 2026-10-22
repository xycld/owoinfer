// Output digests of the small (bundle) FLUX.2 VAE decoder and the head-dim-384 attention kernel, checked against the
// digests stored in tests/data/flux2_vae/small_digests.json. Every decode mode runs untiled, and tiled schedules run,
// on square, non-square and 2048^2 latents; attention_fp32 runs on hash data in both precisions. Each run happens twice
// (deterministic) and its FNV-1a 64 digest is compared with the stored one.
//
// Digests depend on the GPU and the libraries, so the file keys them by GPU name, compute capability, CUDA runtime and
// cuBLASLt version (the attention linears). With a matching key every digest must match (byte-identical outputs);
// with none the digests are printed. The stored values come from the decoder as it was before the full-decoder
// generalization. MI_FLUX2_VAE_DIGESTS_WRITE=<path> writes this machine's entry as a one-entry file.
#include <gtest/gtest.h>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/vae/vae_kernels.h"
#include "models/flux2_vae/flux2_vae.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "test_util.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;
using mi::flux2_vae::DecodeMode;
using mi::flux2_vae::DecoderConfig;
using mi::flux2_vae::Flux2VaeDecoder;

fs::path bundle_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base";
}
fs::path source_dir() { return fs::path(MOSSINFER_SOURCE_DIR); }
fs::path digests_file() { return source_dir() / "tests" / "data" / "flux2_vae" / "small_digests.json"; }

std::uint64_t fnv1a(const void* data, std::size_t n) {
  std::uint64_t h = 1469598103934665603ull;
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
std::string hex(std::uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
  return b;
}

struct Key {
  std::string gpu, cc;
  std::int64_t cudart = 0, cublaslt = 0;
  bool operator==(const Key&) const = default;
  std::string json() const {
    std::ostringstream o;
    o << "{\"gpu\": \"" << gpu << "\", \"cc\": \"" << cc << "\", \"cudart\": " << cudart << ", \"cublaslt\": " << cublaslt << "}";
    return o.str();
  }
};

Key this_machine() {
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  MI_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  Key k;
  k.gpu = prop.name;
  k.cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  int rt = 0;
  MI_CUDA_CHECK(cudaRuntimeGetVersion(&rt));
  k.cudart = rt;
  k.cublaslt = static_cast<std::int64_t>(cublasLtGetVersion());
  return k;
}

// The stored digests for `key` (empty when the file has no entry for it).
std::map<std::string, std::string> stored_digests(const Key& key) {
  std::map<std::string, std::string> out;
  std::ifstream in(digests_file());
  if (!in) return out;
  std::stringstream ss;
  ss << in.rdbuf();
  const mi::json::Value doc = mi::json::parse(ss.str());
  for (const auto& e : doc.at("entries").as_array()) {
    const auto& k = e.at("key");
    const Key ek{k.at("gpu").as_string(), k.at("cc").as_string(), k.at("cudart").as_int(), k.at("cublaslt").as_int()};
    if (!(ek == key)) continue;
    for (const auto& [label, v] : e.at("digests").as_object()) out[label] = v.as_string();
  }
  return out;
}

class Digests {
 public:
  explicit Digests(const Key& key) : key_(key), stored_(stored_digests(key)) {
    std::printf("[flux2_vae_regress] key %s: %s\n", key_.json().c_str(),
                stored_.empty() ? "no stored entry, digests are printed only" : "checking stored digests");
  }
  void add(const std::string& label, std::uint64_t digest) {
    const std::string got = hex(digest);
    got_.emplace_back(label, got);
    const auto it = stored_.find(label);
    const char* verdict = stored_.empty() ? "" : it == stored_.end() ? "(not stored)" : it->second == got ? "matches" : "DIFFERS";
    std::printf("[flux2_vae_regress] %-52s %s %s\n", label.c_str(), got.c_str(), verdict);
    if (it != stored_.end()) {
      EXPECT_EQ(got, it->second) << label;
      ++checked_;
    }
  }
  ~Digests() {
    if (!stored_.empty()) {
      // Cases whose inputs are local-only (the 2048^2 latent) may be absent; the count shows what ran.
      std::printf("[flux2_vae_regress] %d of %zu stored digests checked\n", checked_, stored_.size());
    }
    const char* path = std::getenv("MI_FLUX2_VAE_DIGESTS_WRITE");
    if (path == nullptr || *path == '\0') return;
    std::ofstream f(path);
    f << "{\n \"entries\": [\n  {\n   \"key\": " << key_.json() << ",\n   \"digests\": {\n";
    for (std::size_t i = 0; i < got_.size(); ++i)
      f << "    \"" << got_[i].first << "\": \"" << got_[i].second << "\"" << (i + 1 < got_.size() ? ",\n" : "\n");
    f << "   }\n  }\n ]\n}\n";
  }

 private:
  Key key_;
  std::map<std::string, std::string> stored_;
  std::vector<std::pair<std::string, std::string>> got_;
  int checked_ = 0;
};

std::vector<float> hash_data(std::size_t n, std::uint64_t seed) {
  std::vector<float> v(n);
  std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  for (float& x : v) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    x = (static_cast<float>(s >> 40) * 0x1p-24f - 0.5f) * 0.5f;
  }
  return v;
}

TEST(Flux2VaeRegress, SmallDecoderDigests) {
  const fs::path idx = bundle_dir() / "moss.mossidx";
  if (!fs::is_regular_file(idx)) GTEST_SKIP() << "weights not present (" << bundle_dir() << ")";
  Digests digests(this_machine());

  // Head-dim-384 attention on packed [L, 3 * 384] q/k/v (the mid attention's layout).
  for (const auto& [batch, len] : {std::pair<std::int64_t, std::int64_t>{1, 1000}, {2, 200}, {1, 64}}) {
    constexpr std::int64_t d = 384;
    const auto qkv = hash_data(static_cast<std::size_t>(batch * len * 3 * d), static_cast<std::uint64_t>(len));
    mi::DeviceBuffer in(qkv.size() * 4), out(static_cast<std::size_t>(batch * len * d) * 4);
    in.upload(qkv.data(), qkv.size() * 4);
    const auto view = [&](int p) {
      Tensor t = Tensor::view(static_cast<float*>(in.get()) + p * d, mi::DType::kF32, mi::Device::kCUDA, {batch, len, d});
      t.stride[1] = 3 * d;
      t.stride[0] = len * 3 * d;
      return t;
    };
    const Tensor o = Tensor::view(out.get(), mi::DType::kF32, mi::Device::kCUDA, {batch, len, d});
    for (const auto prec : {mi::vae::AttnPrecision::kTf32x3, mi::vae::AttnPrecision::kTf32}) {
      std::uint64_t h[2];
      for (std::uint64_t& hh : h) {
        MI_CUDA_CHECK(cudaMemset(out.get(), 0, static_cast<std::size_t>(o.nbytes())));
        mi::vae::attention_fp32(view(0), view(1), view(2), o, 1.0f / std::sqrt(384.0f), prec, nullptr);
        std::vector<float> got(static_cast<std::size_t>(o.numel()));
        MI_CUDA_CHECK(cudaMemcpy(got.data(), out.get(), got.size() * 4, cudaMemcpyDeviceToHost));
        hh = fnv1a(got.data(), got.size() * 4);
      }
      EXPECT_EQ(h[0], h[1]) << "attention not deterministic";
      digests.add("attention384 B" + std::to_string(batch) + " L" + std::to_string(len) +
                      (prec == mi::vae::AttnPrecision::kTf32x3 ? " tf32x3" : " tf32"),
                  h[0]);
    }
  }

  const auto index = mi::weights::load_index(idx, bundle_dir());
  mi::runtime::StreamSet streams;
  const DecoderConfig small = DecoderConfig::small();
  const std::size_t bytes = Flux2VaeDecoder::persistent_bytes(index, true) +
                            Flux2VaeDecoder::workspace_bytes(small, 128, 128, DecodeMode::torch_reference()) + (std::size_t{64} << 20);
  mi::runtime::DeviceArena arena(bytes);
  Flux2VaeDecoder model(index, arena, streams, /*bf16_weights=*/true);
  const cudaStream_t s = streams.compute();

  struct Case {
    std::string name;
    fs::path latent;
    bool all_modes;
  };
  const std::vector<Case> cases = {{"g_short_1024", source_dir() / "tests/golden/g_short_1024/final_latent.npy", true},
                                   {"g_fridge_1456", source_dir() / "tests/golden/g_fridge_1456/final_latent.npy", true},
                                   {"asset001_base", source_dir() / "tests/reference_images/asset001_base.latent.npy", true},
                                   {"ac16_2048", source_dir() / "tools/reference/work/bench/images/ac16_2048.latent.npy", false}};
  for (const Case& c : cases) {
    if (!fs::is_regular_file(c.latent)) {
      std::printf("[flux2_vae_regress] %s skipped (latent absent)\n", c.name.c_str());
      continue;
    }
    const auto a = mi::npy::load(c.latent);
    const std::int64_t h = a.shape[2], w = a.shape[3];
    const auto v = a.to_vector<float>();
    mi::DeviceBuffer lat(v.size() * sizeof(float));
    lat.upload(v.data(), v.size() * sizeof(float));
    const Tensor lt = Tensor::view(lat.get(), mi::DType::kF32, mi::Device::kCUDA, {128, h, w});
    mi::DeviceBuffer out(static_cast<std::size_t>(h * w * 256 * 3));
    const Tensor rgb = Tensor::view(out.get(), mi::DType::kU8, mi::Device::kCUDA, {h * 16, w * 16, 3});
    const auto run = [&](const std::string& label, auto&& fn) {
      std::vector<std::uint8_t> r1(static_cast<std::size_t>(rgb.nbytes())), r2(r1.size());
      fn();
      MI_CUDA_CHECK(cudaStreamSynchronize(s));
      out.download(r1.data(), r1.size());
      MI_CUDA_CHECK(cudaMemset(out.get(), 0, r1.size()));
      fn();
      MI_CUDA_CHECK(cudaStreamSynchronize(s));
      out.download(r2.data(), r2.size());
      EXPECT_EQ(r1, r2) << c.name << " " << label << ": not deterministic";
      digests.add(c.name + " " + label, fnv1a(r1.data(), r1.size()));
    };
    struct M {
      const char* name;
      DecodeMode mode;
    };
    std::vector<M> modes = {{"reference", DecodeMode::reference()}};
    if (c.all_modes) {
      modes.push_back({"torch_reference", DecodeMode::torch_reference()});
      modes.push_back({"fast_attention", DecodeMode::fast_attention()});
      modes.push_back({"fastest", DecodeMode::fastest()});
    }
    for (const M& m : modes) run(std::string("untiled ") + m.name, [&] { model.decode(lt, rgb, m.mode, s); });
    if (!c.all_modes) continue;
    for (const M& m : {M{"reference", DecodeMode::reference()}, M{"fastest", DecodeMode::fastest()}}) {
      for (const int tile : {16, 40}) {
        const auto plan = Flux2VaeDecoder::plan_tiled_min_memory(small, h, w, m.mode, tile);
        run(std::string("tiled ") + m.name + " min-mem t" + std::to_string(tile), [&] { model.decode_tiled(lt, rgb, m.mode, plan, s); });
      }
      const auto plan = Flux2VaeDecoder::plan_tiled(small, h, w, m.mode, std::size_t{3} << 29);
      ASSERT_TRUE(plan.has_value());
      run(std::string("tiled ") + m.name + " 1.5GiB", [&] { model.decode_tiled(lt, rgb, m.mode, *plan, s); });
    }
  }
}

}  // namespace
