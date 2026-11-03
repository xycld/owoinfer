// Runs one DiT block's kernels on real weights and activations, and compares each output with the engine's
// own, captured from its forward at the 1024x1024 golden case. Reads files only, so it needs no CUDA.
// Usage: vulkan_block_real DIR [--device N]
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "block.h"
#include "half.h"
#include "runtime.h"

namespace {

using namespace mi::vk;
using bits = std::uint16_t;
namespace fs = std::filesystem;

std::vector<bits> load(const fs::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  const auto bytes = static_cast<std::size_t>(in.tellg());
  std::vector<bits> out(bytes / 2);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
  return out;
}

std::unique_ptr<Buffer> device_buffer(Device& device, const std::vector<bits>& values) {
  auto b = std::make_unique<Buffer>(device, values.size() * 2);
  b->upload(values.data(), values.size() * 2);
  return b;
}

// ULPs of max(|engine|, rms): near-zero outputs of cancelling sums are not relatively accurate in any accumulation order.
double scaled_ulps(float got, float engine, double rms) {
  int e = 0;
  (void)std::frexp(std::max<double>(std::abs(engine), rms), &e);
  return std::abs(static_cast<double>(got) - static_cast<double>(engine)) / std::ldexp(1.0, e - 8);
}

bool compare(const std::vector<bits>& got, const std::vector<bits>& engine) {
  double squares = 0.0;
  for (const bits b : engine) squares += static_cast<double>(bf16_to_float(b)) * bf16_to_float(b);
  const double rms = std::sqrt(squares / static_cast<double>(engine.size()));
  std::vector<double> errors(got.size());
  std::size_t same = 0, nans = 0, engine_nans = 0;
  double dot = 0.0, norm_got = 0.0, norm_engine = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float g = bf16_to_float(got[i]), e = bf16_to_float(engine[i]);
    if (std::isnan(g)) ++nans;
    if (std::isnan(e)) ++engine_nans;
    same += got[i] == engine[i];
    errors[i] = scaled_ulps(g, e, rms);
    dot += static_cast<double>(g) * e;
    norm_got += static_cast<double>(g) * g;
    norm_engine += static_cast<double>(e) * e;
  }
  std::sort(errors.begin(), errors.end());
  const auto at = [&](double q) { return errors[static_cast<std::size_t>(q * static_cast<double>(errors.size() - 1))]; };
  const double cosine = dot / std::sqrt(norm_got * norm_engine);
  std::printf("  %zu values: %.2f %% same bits, scaled ULPs p50 %.2f p99 %.2f p99.9 %.2f max %.2f, cosine %.9f, NaN %zu (engine %zu)\n",
              got.size(), 100.0 * static_cast<double>(same) / static_cast<double>(got.size()), at(0.5), at(0.99), at(0.999),
              errors.back(), cosine, nans, engine_nans);
  return nans == engine_nans && cosine >= 0.999;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: %s DIR [--device N] [--path coop|coop-f16|plain]\n", argv[0]);
    return 2;
  }
  const fs::path dir = argv[1];
  int device_index = -1;
  std::string path_name;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg == "--device") device_index = std::atoi(argv[i + 1]);
    else if (arg == "--path") path_name = argv[i + 1];
  }
  try {
    Device device(device_index);
    const BlockKernels kernels = path_name.empty() ? BlockKernels(device)
                                                   : BlockKernels(device, block_path_from_name(path_name));
    std::printf("device %u: %s, path %s\n", device.info().index, device.info().name.c_str(), block_path_name(kernels.path()));

    std::vector<std::int32_t> lengths;
    struct Case {
      int block, tokens, skip_uncond;
    };
    std::vector<Case> cases;
    std::ifstream manifest(dir / "manifest.txt");
    std::string line;
    while (std::getline(manifest, line)) {
      std::istringstream in(line);
      std::string kind, unused;
      in >> kind;
      if (kind == "lengths") {
        std::int32_t a = 0, b = 0;
        in >> a >> b;
        lengths = {a, b};
      } else if (kind == "block") {
        Case c{};
        in >> c.block >> unused >> c.tokens >> unused >> c.skip_uncond;
        cases.push_back(c);
      }
    }
    if (lengths.empty() || cases.empty()) throw std::runtime_error("no manifest in " + dir.string());

    auto rope = device_buffer(device, load(dir / "rope.bin"));
    auto mod = device_buffer(device, load(dir / "mod.bin"));
    bool ok = true;
    for (const Case& c : cases) {
      const std::string p = "block" + std::to_string(c.block) + ".";
      std::printf("block %d: %d tokens%s\n", c.block, c.tokens, c.skip_uncond ? " (the uncond row skips it)" : "");
      auto norm1 = device_buffer(device, load(dir / (p + "norm1.bin"))), norm2 = device_buffer(device, load(dir / (p + "norm2.bin"))),
           q_norm = device_buffer(device, load(dir / (p + "q_norm.bin"))), k_norm = device_buffer(device, load(dir / (p + "k_norm.bin"))),
           qkv_w = device_buffer(device, load(dir / (p + "qkv.bin"))), o_w = device_buffer(device, load(dir / (p + "o.bin"))),
           up = device_buffer(device, load(dir / (p + "up_gate.bin"))), down = device_buffer(device, load(dir / (p + "down.bin")));
      auto x = device_buffer(device, load(dir / (p + "x_in.bin")));
      const auto engine = load(dir / (p + "x_out.bin"));
      const auto n = static_cast<std::size_t>(c.tokens);
      Buffer y(device, n * kDiTDim * 2), qkv(device, n * 3 * kDiTDim * 2), attn(device, n * kDiTDim * 2), ffn(device, n * kDiTFfnDim * 2);
      std::vector<std::int32_t> cu = {0, lengths[0]};
      if (!c.skip_uncond) cu.push_back(lengths[0] + lengths[1]);
      const BlockWeights w{norm1->address(), norm2->address(), q_norm->address(), k_norm->address(), qkv_w->address(), o_w->address(), up->address(), down->address()};
      const BlockBuffers b{x->address(), y.address(), qkv.address(), attn.address(), ffn.address(), rope->address(), mod->address(), cu, static_cast<std::uint32_t>(n)};
      const auto start = std::chrono::steady_clock::now();
      for (int op = 0; op < kBlockOps; ++op) device.run([&](VkCommandBuffer cb) { record_block_op(kernels, cb, op, w, b); });
      std::printf("  %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
      std::vector<bits> got(engine.size());
      x->download(got.data(), got.size() * 2);
      ok = compare(got, engine) && ok;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
