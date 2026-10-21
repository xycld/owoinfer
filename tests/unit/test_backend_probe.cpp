// src/backend probe: verdict and recommendation logic on synthetic inputs (BackendProbe.*, no GPU needed, safe on
// machines without a driver) and one enumeration of the real device (GpuBackendProbe.*).
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "backend/probe.h"
#include "cuda_context_check.h"

namespace {

using mi::backend::CudaArch;
using mi::backend::DeviceInfo;
using mi::backend::Flavour;
using mi::backend::ProbeResult;

// The release presets' list (CMakePresets.json).
const std::vector<CudaArch> kRelease = mi::backend::parse_cuda_architectures("80-real;86-real;89-real;120");

DeviceInfo device(int major, int minor) {
  DeviceInfo d;
  d.index = 0;
  d.name = "synthetic";
  d.cc_major = major;
  d.cc_minor = minor;
  return d;
}

bool supported(int major, int minor, int driver, int build, const std::vector<CudaArch>& archs = kRelease) {
  DeviceInfo d = device(major, minor);
  mi::backend::evaluate_device(d, driver, build, archs);
  EXPECT_EQ(d.supported, d.reason.empty()) << d.reason;
  return d.supported;
}

TEST(BackendProbe, FlavourNames) {
  EXPECT_STREQ(mi::backend::flavour_name(Flavour::kCuda12), "cu12");
  EXPECT_STREQ(mi::backend::flavour_name(Flavour::kCuda13), "cu13");
  EXPECT_STREQ(mi::backend::flavour_name(Flavour::kNone), "none");
  EXPECT_EQ(mi::backend::flavour_of_cuda_version(12080), Flavour::kCuda12);
  EXPECT_EQ(mi::backend::flavour_of_cuda_version(13040), Flavour::kCuda13);
  EXPECT_EQ(mi::backend::flavour_of_cuda_version(11080), Flavour::kNone);
  EXPECT_EQ(mi::backend::min_driver_version(Flavour::kCuda12), 12000);
  EXPECT_EQ(mi::backend::min_driver_version(Flavour::kCuda13), 13000);
}

TEST(BackendProbe, FlavourForDriver) {
  EXPECT_EQ(mi::backend::flavour_for_driver(0), Flavour::kNone);
  EXPECT_EQ(mi::backend::flavour_for_driver(11080), Flavour::kNone);
  EXPECT_EQ(mi::backend::flavour_for_driver(12000), Flavour::kCuda12);
  EXPECT_EQ(mi::backend::flavour_for_driver(12090), Flavour::kCuda12);
  EXPECT_EQ(mi::backend::flavour_for_driver(13000), Flavour::kCuda13);
  EXPECT_EQ(mi::backend::flavour_for_driver(13040), Flavour::kCuda13);
  EXPECT_EQ(mi::backend::flavour_for_driver(14000), Flavour::kCuda13);  // newest flavour we ship
}

TEST(BackendProbe, ParsesCmakeArchitectureLists) {
  ASSERT_EQ(kRelease.size(), 4u);
  EXPECT_EQ(kRelease[0].major, 8);
  EXPECT_EQ(kRelease[0].minor, 0);
  EXPECT_TRUE(kRelease[0].sass);
  EXPECT_FALSE(kRelease[0].ptx);
  EXPECT_EQ(kRelease[3].major, 12);
  EXPECT_EQ(kRelease[3].minor, 0);
  EXPECT_TRUE(kRelease[3].sass);
  EXPECT_TRUE(kRelease[3].ptx);

  const auto commas = mi::backend::parse_cuda_architectures("89,120-virtual");
  ASSERT_EQ(commas.size(), 2u);
  EXPECT_TRUE(commas[0].sass && commas[0].ptx);
  EXPECT_FALSE(commas[1].sass);
  EXPECT_TRUE(commas[1].ptx);

  const auto specific = mi::backend::parse_cuda_architectures("120a-real");
  ASSERT_EQ(specific.size(), 1u);
  EXPECT_TRUE(specific[0].exact_minor);
  EXPECT_FALSE(specific[0].ptx);

  // Lists the probe cannot reason about are "unknown": everything is assumed compatible.
  EXPECT_TRUE(mi::backend::parse_cuda_architectures("native").empty());
  EXPECT_TRUE(mi::backend::parse_cuda_architectures("all-major").empty());
  EXPECT_TRUE(mi::backend::parse_cuda_architectures("").empty());
  EXPECT_TRUE(mi::backend::compatible_image({}, 9, 0, false));
}

TEST(BackendProbe, KernelImageCompatibility) {
  // SASS runs on the same major with an equal or higher minor.
  EXPECT_TRUE(mi::backend::compatible_image(kRelease, 8, 0, false));
  EXPECT_TRUE(mi::backend::compatible_image(kRelease, 8, 7, false));  // Jetson Orin: sm_86 SASS
  EXPECT_TRUE(mi::backend::compatible_image(kRelease, 12, 1, false));
  // sm_90 / sm_100 have neither SASS nor older PTX; compute_120 PTX cannot be JIT-compiled down.
  EXPECT_FALSE(mi::backend::compatible_image(kRelease, 9, 0, true));
  EXPECT_FALSE(mi::backend::compatible_image(kRelease, 10, 0, true));
  // A future architecture JITs compute_120 PTX, but only on a driver at least as new as the toolkit.
  EXPECT_TRUE(mi::backend::compatible_image(kRelease, 13, 0, true));
  EXPECT_FALSE(mi::backend::compatible_image(kRelease, 13, 0, false));
  // Arch-specific SASS runs on exactly its compute capability.
  const auto specific = mi::backend::parse_cuda_architectures("120a-real");
  EXPECT_TRUE(mi::backend::compatible_image(specific, 12, 0, true));
  EXPECT_FALSE(mi::backend::compatible_image(specific, 12, 1, true));
  // A dev build for one architecture.
  const auto dev = mi::backend::parse_cuda_architectures("89");
  EXPECT_TRUE(mi::backend::compatible_image(dev, 8, 9, false));
  EXPECT_FALSE(mi::backend::compatible_image(dev, 8, 6, true));
  EXPECT_TRUE(mi::backend::compatible_image(dev, 12, 0, true));  // via compute_89 PTX
}

TEST(BackendProbe, DeviceVerdicts) {
  EXPECT_TRUE(supported(8, 9, 13040, 13040));
  EXPECT_TRUE(supported(8, 6, 13000, 13040));   // minor-version compatibility: any 13.x driver
  EXPECT_TRUE(supported(12, 0, 12080, 12080));
  EXPECT_FALSE(supported(7, 5, 13040, 13040));  // Turing
  EXPECT_FALSE(supported(8, 9, 12090, 13040));  // cu13 build on a 12.x driver
  EXPECT_FALSE(supported(8, 9, 0, 13040));      // no driver
  EXPECT_FALSE(supported(9, 0, 13040, 13040));  // no image for Hopper in the release list

  DeviceInfo turing = device(7, 5);
  mi::backend::evaluate_device(turing, 13040, 13040, kRelease);
  EXPECT_NE(turing.reason.find("8.0"), std::string::npos) << turing.reason;
  DeviceInfo old_driver = device(8, 9);
  mi::backend::evaluate_device(old_driver, 12090, 13040, kRelease);
  EXPECT_NE(old_driver.reason.find("580"), std::string::npos) << old_driver.reason;
  DeviceInfo future = device(13, 0);
  mi::backend::evaluate_device(future, 13000, 13040, kRelease);
  EXPECT_FALSE(future.supported);
  EXPECT_NE(future.reason.find("JIT"), std::string::npos) << future.reason;
}

TEST(BackendProbe, Recommendation) {
  auto recommend = [](int driver, std::vector<DeviceInfo> devices) {
    ProbeResult r;
    r.driver_version = driver;
    r.devices = std::move(devices);
    mi::backend::recommend(r, kRelease);
    EXPECT_FALSE(r.reason.empty());
    return r.recommended;
  };
  EXPECT_EQ(recommend(13040, {device(8, 9)}), Flavour::kCuda13);
  EXPECT_EQ(recommend(13000, {device(12, 0)}), Flavour::kCuda13);
  EXPECT_EQ(recommend(12080, {device(8, 6)}), Flavour::kCuda12);
  EXPECT_EQ(recommend(12000, {device(8, 0)}), Flavour::kCuda12);
  EXPECT_EQ(recommend(11080, {device(8, 6)}), Flavour::kNone);
  EXPECT_EQ(recommend(0, {device(8, 6)}), Flavour::kNone);
  EXPECT_EQ(recommend(13040, {}), Flavour::kNone);
  EXPECT_EQ(recommend(13040, {device(7, 5), device(6, 1)}), Flavour::kNone);
  EXPECT_EQ(recommend(13040, {device(7, 5), device(8, 6)}), Flavour::kCuda13);  // one capable device is enough
  // Compute capability >= 8.0 is not enough: the device needs a kernel image in the build's architecture list.
  EXPECT_EQ(recommend(13040, {device(9, 0)}), Flavour::kNone);
  EXPECT_EQ(recommend(13040, {device(9, 0), device(10, 0)}), Flavour::kNone);
  EXPECT_EQ(recommend(13040, {device(9, 0), device(8, 9)}), Flavour::kCuda13);
  EXPECT_EQ(recommend(13040, {device(13, 0)}), Flavour::kCuda13);  // compute_120 PTX, JIT-compiled
  ProbeResult hopper;
  hopper.driver_version = 13040;
  hopper.devices = {device(9, 0), device(10, 0)};
  mi::backend::recommend(hopper, kRelease);
  EXPECT_NE(hopper.reason.find("no kernel image for sm_90, sm_100 in this build"), std::string::npos) << hopper.reason;
  // A dev build for sm_89 only (SASS + PTX) has no image for sm_86.
  ProbeResult ampere;
  ampere.driver_version = 13040;
  ampere.devices = {device(8, 6)};
  mi::backend::recommend(ampere, mi::backend::parse_cuda_architectures("89"));
  EXPECT_EQ(ampere.recommended, Flavour::kNone);
  EXPECT_NE(ampere.reason.find("no kernel image for sm_86"), std::string::npos) << ampere.reason;
}

TEST(BackendProbe, BuildIdentity) {
  EXPECT_NE(mi::backend::flavour_of_cuda_version(CUDART_VERSION), Flavour::kNone);
  // The architecture list comes from CMake and parses (or is one of the "unknown" spellings).
  const std::string archs = mi::backend::build_architectures_string();
  EXPECT_FALSE(archs.empty());
  if (archs != "native" && archs.rfind("all", 0) != 0) {
    EXPECT_FALSE(mi::backend::build_architectures().empty()) << archs;
  }
}

// Runs the real probe without creating a CUDA context (no VRAM use), so it is safe on machines without a GPU or
// driver (CUDA_VISIBLE_DEVICES="" or none installed) and next to other GPU work.
TEST(BackendProbe, ProbeWithoutContextNeverThrows) {
  mi::backend::ProbeOptions options;
  options.query_free_memory = false;
  ProbeResult r;
  ASSERT_NO_THROW(r = mi::backend::probe(options));
  EXPECT_EQ(r.build_cuda_version, CUDART_VERSION);
  EXPECT_EQ(r.build_flavour, mi::backend::flavour_of_cuda_version(CUDART_VERSION));
  EXPECT_GT(r.runtime_version, 0);
  EXPECT_FALSE(r.reason.empty());
  if (r.devices.empty()) {
    EXPECT_EQ(r.recommended, Flavour::kNone);
  }
  int supported_count = 0;
  for (const DeviceInfo& d : r.devices) {
    EXPECT_EQ(d.free_bytes, 0u);
    supported_count += d.supported ? 1 : 0;
  }
  EXPECT_EQ(r.supported_devices, supported_count);
  // The probe leaves no error behind. Without a device the runtime re-reports its failed initialisation on every
  // call (cudaGetLastError included), so this is only checkable where a device exists.
  if (!r.devices.empty()) {
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
  EXPECT_FALSE(mi::backend::format_report(r).empty());
}

TEST(GpuBackendProbe, EnumeratesDevices) {
  const ProbeResult r = mi::backend::probe();
  std::printf("%s", mi::backend::format_report(r).c_str());
  ASSERT_FALSE(r.devices.empty()) << r.reason;
  EXPECT_TRUE(r.driver_ok);
  EXPECT_GE(r.driver_version, mi::backend::min_driver_version(r.build_flavour));
  EXPECT_NE(r.recommended, Flavour::kNone);
  EXPECT_GE(r.supported_devices, 1);
  const DeviceInfo& d = r.devices[0];
  EXPECT_FALSE(d.name.empty());
  EXPECT_GE(d.cc_major, mi::backend::kMinComputeMajor);
  EXPECT_GT(d.total_bytes, 0u);
  EXPECT_FALSE(d.pci_bus_id.empty());
  if (d.supported) {
    EXPECT_GT(d.free_bytes, 0u);
    EXPECT_LE(d.free_bytes, d.total_bytes);
    EXPECT_TRUE(d.free_memory_source == "nvml" || d.free_memory_source == "driver") << d.free_memory_source;
  }
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 0);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// Measuring free memory leaves no CUDA context behind, with NVML and with the driver-API fallback; the calling
// thread's current context is untouched.
TEST(GpuBackendProbe, FreeMemoryLeavesNoContext) {
  if (!mi::testing::cuda_context_report().empty()) GTEST_SKIP() << "the process already holds a CUDA context";
  const auto ctx_before = mi::testing::current_context();
  using mi::backend::FreeMemoryMethod;
  for (const FreeMemoryMethod method : {FreeMemoryMethod::kAuto, FreeMemoryMethod::kNvml, FreeMemoryMethod::kDriver}) {
    mi::backend::ProbeOptions options;
    options.free_memory_method = method;
    const ProbeResult r = mi::backend::probe(options);
    ASSERT_FALSE(r.devices.empty()) << r.reason;
    const DeviceInfo& d = r.devices[0];
    std::printf("[probe] method %d: device 0 free %.2f GiB via '%s'\n", static_cast<int>(method),
                static_cast<double>(d.free_bytes) / (1 << 30), d.free_memory_source.c_str());
    if (!d.supported) continue;
    if (method == FreeMemoryMethod::kDriver) {
      EXPECT_EQ(d.free_memory_source, "driver");
    }
    if (method == FreeMemoryMethod::kNvml && d.free_memory_source.empty()) continue;  // no NVML on this machine
    EXPECT_GT(d.free_bytes, 0u);
    EXPECT_LE(d.free_bytes, d.total_bytes);
    EXPECT_EQ(mi::testing::cuda_context_report(), "") << "after method " << static_cast<int>(method);
  }
  EXPECT_EQ(mi::testing::current_context(), ctx_before);
}

// The context check itself sees a context the runtime creates (so the test above cannot pass vacuously).
TEST(GpuBackendProbe, ContextCheckSeesARuntimeContext) {
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  EXPECT_NE(mi::testing::cuda_context_report().find("primary context active on device 0"), std::string::npos)
      << mi::testing::cuda_context_report();
  ASSERT_EQ(cudaDeviceReset(), cudaSuccess);
}

}  // namespace
