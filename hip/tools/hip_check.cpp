// The runtime's smoke test: lists the devices HIP sees, then on the chosen one copies a pinned buffer through device memory
// on a stream and back, and runs a GEMM with tails on M, N and K on small integers, whose sums are exact in fp32 on any
// leaf, against the host.
// Usage: hip_check [--list] [--device N]. Exit status: 0 passed, 1 failed, 2 no usable device.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "half.h"
#include "hip_block.h"
#include "runtime.h"

namespace {

using namespace mi::hip;

int run(int index) {
  Device device(index);
  std::printf("device %d: %s (%s), %.2f GiB free of %.2f, build %s, leaf %s\n", device.info().index,
              device.info().name.c_str(), device.info().arch.c_str(),
              static_cast<double>(device.free_memory()) / (1u << 30),
              static_cast<double>(device.info().total_memory) / (1u << 30), build_target(), hip_block::leaf_name());
  Stream stream;
  int failures = 0;

  // Pinned -> device -> device -> pinned on one stream, timed.
  constexpr std::size_t kBytes = std::size_t{64} << 20;
  HostBuffer src(kBytes), dst(kBytes);
  for (std::size_t i = 0; i < kBytes / 4; ++i) src.as<std::uint32_t>()[i] = static_cast<std::uint32_t>(i * 2654435761u);
  std::memset(dst.data(), 0, kBytes);
  DeviceBuffer a(kBytes), b(kBytes);
  Event start(true), stop(true);
  start.record(stream.get());
  copy_async(a.data(), src.data(), kBytes, stream.get());
  copy_async(b.data(), a.data(), kBytes, stream.get());
  copy_async(dst.data(), b.data(), kBytes, stream.get());
  stop.record(stream.get());
  stop.synchronize();
  const bool copy_ok = std::memcmp(src.data(), dst.data(), kBytes) == 0;
  std::printf("copies: 64 MiB host -> device -> device -> host in %.2f ms, %s\n", static_cast<double>(stop.elapsed_ms(start)),
              copy_ok ? "identical" : "DIFFERENT");
  failures += copy_ok ? 0 : 1;

  // out[m, n] = x[m, k] . w[n, k]: integers in [-3, 3], so every partial sum is exact and any order gives the same bits.
  constexpr int m = 37, n = 53, k = 72;
  std::vector<std::uint16_t> x(std::size_t{m} * k), w(std::size_t{n} * k), out(std::size_t{m} * n, 0xffff);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = float_to_bf16(static_cast<float>(static_cast<int>((i * 3 + i / 5) % 7) - 3));
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = float_to_bf16(static_cast<float>(static_cast<int>((i * 5 + i / 3) % 7) - 3));
  DeviceBuffer dx(x.size() * 2), dw(w.size() * 2), dout(out.size() * 2);
  dx.upload(x.data(), x.size() * 2);
  dw.upload(w.data(), w.size() * 2);
  dout.upload(out.data(), out.size() * 2);
  hip_block::gemm(dout.as<std::uint16_t>(), n, dx.as<std::uint16_t>(), k, dw.as<std::uint16_t>(), k, m, n, k, stream.get());
  stream.synchronize();
  dout.download(out.data(), out.size() * 2);
  int wrong = 0;
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      float s = 0.0f;
      for (int i = 0; i < k; ++i) s += bf16_to_float(x[std::size_t{static_cast<std::size_t>(r)} * k + static_cast<std::size_t>(i)]) *
                                       bf16_to_float(w[std::size_t{static_cast<std::size_t>(c)} * k + static_cast<std::size_t>(i)]);
      if (out[static_cast<std::size_t>(r) * n + static_cast<std::size_t>(c)] != float_to_bf16(s)) ++wrong;
    }
  }
  std::printf("gemm %dx%dx%d: %d of %d outputs differ from the host\n", m, n, k, wrong, m * n);
  failures += wrong == 0 ? 0 : 1;
  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  bool list = false;
  int index = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--list") list = true;
    else if (arg == "--device" && i + 1 < argc) index = std::atoi(argv[++i]);
    else {
      std::printf("usage: %s [--list] [--device N]\n", argv[0]);
      return 1;
    }
  }
  try {
    const std::vector<DeviceInfo> devices = list_devices();
    if (list || devices.empty()) {
      std::printf("%zu device(s), build %s\n", devices.size(), build_target());
      for (const DeviceInfo& d : devices) {
        std::printf("  %d: %s, %s, %d compute units, %.2f GiB, %s\n", d.index, d.name.c_str(), d.arch.c_str(), d.compute_units,
                    static_cast<double>(d.total_memory) / (1u << 30),
                    d.unusable_reason.empty() ? "usable" : d.unusable_reason.c_str());
      }
    }
    bool usable = false;
    for (const DeviceInfo& d : devices) usable = usable || d.unusable_reason.empty();
    if (!usable) {
      std::printf("no usable device\n");
      return 2;
    }
    return list ? 0 : run(index);
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
