// Bring-up check for the Vulkan backend: lists the devices, checks each GEMM path against an fp64 CPU reference, then
// times the engine's shapes. Exits non-zero when a check fails.
//
// Usage: vulkan_check [--list] [--device N] [--path auto|coop|plain] [--no-bench] [--bench M N K iterations]...
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "gemm.h"
#include "half.h"
#include "runtime.h"

namespace {

using mi::vk::Buffer;
using mi::vk::Device;
using mi::vk::Gemm;
using mi::vk::GemmArgs;
using mi::vk::GemmPath;
using mi::vk::InputType;

struct Shape {
  std::uint32_t m, n, k;
  int iterations;
};

std::vector<std::uint16_t> random_values(std::size_t count, std::uint32_t seed, InputType input) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> values(-1.0f, 1.0f);
  std::vector<std::uint16_t> out(count);
  for (auto& v : out) {
    const float f = values(rng);
    v = input == InputType::kBF16 ? mi::vk::float_to_bf16(f) : mi::vk::float_to_half(f);
  }
  return out;
}

float decode(std::uint16_t v, InputType input) {
  return input == InputType::kBF16 ? mi::vk::bf16_to_float(v) : mi::vk::half_to_float(v);
}

// Strides larger than the rows and a poisoned output, so a write past a row or past M fails the check.
bool check_gemm(Device& device, const Gemm& gemm, const Shape& s, InputType input) {
  const std::uint32_t lda = s.k + 16, ldb = s.k + 32, ldc = s.n + 8;
  const std::size_t a_count = static_cast<std::size_t>(s.m) * lda, b_count = static_cast<std::size_t>(s.n) * ldb;
  const std::size_t c_count = (static_cast<std::size_t>(s.m) + 1) * ldc;
  const auto a = random_values(a_count, 1234, input), b = random_values(b_count, 5678, input);
  const std::vector<float> poison(c_count, -123.0f);
  Buffer da(device, a_count * 2), db(device, b_count * 2), dc(device, c_count * 4);
  da.upload(a.data(), a_count * 2);
  db.upload(b.data(), b_count * 2);
  dc.upload(poison.data(), c_count * 4);
  const GemmArgs args{&da, &db, &dc, s.m, s.n, s.k, lda, ldb, ldc};
  device.run([&](VkCommandBuffer cb) { gemm.record(cb, args, input); });
  std::vector<float> c(c_count);
  dc.download(c.data(), c_count * 4);

  double max_error = 0.0, dot = 0.0, norm_got = 0.0, norm_want = 0.0, squared = 0.0, toward = 0.0;
  std::size_t overwritten = 0, wrong_nan = 0;
  for (std::size_t i = 0; i < c_count; ++i) {
    const std::size_t row = i / ldc, column = i % ldc;
    if (row >= s.m || column >= s.n) {
      if (c[i] != -123.0f) ++overwritten;
      continue;
    }
    double want = 0.0;
    for (std::uint32_t k = 0; k < s.k; ++k) {
      want += static_cast<double>(decode(a[row * lda + k], input)) * static_cast<double>(decode(b[column * ldb + k], input));
    }
    const double got = c[i];
    if (std::isnan(got)) ++wrong_nan;
    max_error = std::max(max_error, std::fabs(got - want));
    squared += (got - want) * (got - want);
    toward += (got - want) * (want >= 0.0 ? 1.0 : -1.0);
    dot += got * want;
    norm_got += got * got;
    norm_want += want * want;
  }
  const double cosine = dot / std::sqrt(norm_got * norm_want);
  // Tensor cores round their fp32 accumulator with a bias toward zero, so the error grows with K. Measured on an RTX 4090
  // Laptop: 4.8e-5 at K = 512 and 1.3e-4 at K = 1040, matching cuBLASLt on the same data, and 8.1e-4 at K = 4096, where
  // cuBLASLt reaches 1.3e-4. The plain path stays well inside this. Far below one bf16 step at these magnitudes.
  const double tolerance = 2.5e-7 * s.k + 1e-5;
  const bool ok = overwritten == 0 && wrong_nan == 0 && max_error <= tolerance && cosine >= 0.9999;
  const double cells = static_cast<double>(s.m) * s.n;
  std::printf("%-5s %-4s %5ux%5ux%5u: max error %.3g (limit %.3g), rms %.3g, mean toward |value| %.3g, cosine %.9f, "
              "outside writes %zu  %s\n",
              mi::vk::gemm_path_name(gemm.path()), input == InputType::kBF16 ? "bf16" : "f16", s.m, s.n, s.k, max_error,
              tolerance, std::sqrt(squared / cells), toward / cells, cosine, overwritten, ok ? "ok" : "FAILED");
  return ok;
}

void bench_gemm(Device& device, const Gemm& gemm, const Shape& s) {
  const std::size_t a_count = static_cast<std::size_t>(s.m) * s.k, b_count = static_cast<std::size_t>(s.n) * s.k;
  const auto a = random_values(a_count, 1234, InputType::kF16), b = random_values(b_count, 5678, InputType::kF16);
  Buffer da(device, a_count * 2), db(device, b_count * 2);
  Buffer dc(device, static_cast<VkDeviceSize>(s.m) * s.n * 4);
  da.upload(a.data(), a_count * 2);
  db.upload(b.data(), b_count * 2);
  const GemmArgs args{&da, &db, &dc, s.m, s.n, s.k, s.k, s.k, s.n};
  const auto record = [&](VkCommandBuffer cb) {
    for (int i = 0; i < s.iterations; ++i) gemm.record(cb, args);
  };
  device.run(record);  // warm-up
  const auto start = std::chrono::steady_clock::now();
  device.run(record);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / s.iterations;
  std::printf("%-5s %5ux%5ux%5u: %9.3f ms  %7.2f TFLOPS\n", mi::vk::gemm_path_name(gemm.path()), s.m, s.n, s.k,
              seconds * 1e3, 2.0 * s.m * s.n * s.k / seconds / 1e12);
}

void print_devices() {
  for (const auto& d : mi::vk::list_devices()) {
    std::printf("device %u: %s [%04x:%04x]%s\n", d.index, d.name.c_str(), d.vendor_id, d.device_id,
                d.discrete ? " discrete" : "");
    if (!d.unusable_reason.empty()) {
      std::printf("  unusable: %s\n", d.unusable_reason.c_str());
      continue;
    }
    std::printf("  shared memory %u bytes, subgroup size %u-%u\n", d.caps.shared_memory_bytes, d.caps.min_subgroup,
                d.caps.max_subgroup);
    std::printf("  GEMM: %s\n", d.caps.coop ? "cooperative matrix" : ("plain compute (" + d.caps.coop_reason + ")").c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  int device_index = -1;
  std::string path = "auto";
  bool bench = true, list_only = false;
  std::vector<Shape> shapes;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--list") {
      list_only = true;
    } else if (arg == "--device" && i + 1 < argc) {
      device_index = std::atoi(argv[++i]);
    } else if (arg == "--path" && i + 1 < argc) {
      path = argv[++i];
    } else if (arg == "--no-bench") {
      bench = false;
    } else if (arg == "--bench" && i + 4 < argc) {
      Shape s{};
      s.m = static_cast<std::uint32_t>(std::atoi(argv[++i]));
      s.n = static_cast<std::uint32_t>(std::atoi(argv[++i]));
      s.k = static_cast<std::uint32_t>(std::atoi(argv[++i]));
      s.iterations = std::atoi(argv[++i]);
      shapes.push_back(s);
    } else {
      std::printf("usage: %s [--list] [--device N] [--path auto|coop|plain] [--no-bench] [--bench M N K iterations]...\n",
                  argv[0]);
      return 2;
    }
  }
  if (path != "auto" && path != "coop" && path != "plain") {
    std::printf("--path must be auto, coop or plain\n");
    return 2;
  }
  try {
    print_devices();
    if (list_only) return 0;
    Device device(device_index);
    std::printf("using device %u: %s\n", device.info().index, device.info().name.c_str());
    const Gemm gemm = path == "auto"    ? Gemm(device)
                      : path == "coop" ? Gemm(device, GemmPath::kCoop)
                                       : Gemm(device, GemmPath::kPlain);

    // One tile; tails in M, N and K; several bands with a partial last one; the largest row-count tile boundaries.
    const std::vector<Shape> checks = {{1, 1, 16, 1},       {3, 8, 16, 1},        {64, 64, 64, 1},   {65, 65, 48, 1},
                                       {67, 136, 1040, 1},  {129, 256, 4096, 1},  {128, 640, 64, 1}, {94, 643, 64, 1},
                                       {1030, 9216, 64, 1}, {1025, 136, 1040, 1}, {512, 512, 512, 1}};
    bool ok = true;
    for (const Shape& s : checks) ok = check_gemm(device, gemm, s, InputType::kF16) && ok;
    if (gemm.path() == GemmPath::kPlain) {
      for (const Shape& s : checks) ok = check_gemm(device, gemm, s, InputType::kBF16) && ok;
    }
    if (bench) {
      // The engine's shapes at 1024x1024 with CFG: the DiT's QKV, projection, up/gate and down, then the text encoder's.
      if (shapes.empty()) {
        shapes = {{8222, 9216, 3072, 5}, {8222, 3072, 3072, 5},  {8222, 16384, 3072, 3},
                  {8222, 3072, 8192, 3}, {282, 12288, 4096, 10}, {282, 4096, 12288, 10}};
      }
      for (const Shape& s : shapes) bench_gemm(device, gemm, s);
    }
    std::printf("%s\n", ok ? "all checks passed" : "CHECKS FAILED");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
