// Backend probe: which CUDA devices exist, whether this build can run on them, and which release flavour
// (cu12 / cu13) a launcher should install for the machine's driver. Phase I of backend pluggability
// (.omc/plans/p7a-infra.md); exposed to C callers as mi_probe().
//
// The probe never throws on a missing GPU or driver: it returns an empty device list and a reason.
// Free memory comes from NVML or a short-lived driver-API context (backend/cuda_driver.h), both loaded at run time;
// the probe never leaves a CUDA context behind.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mi::backend {

// Minimum compute capability of every kernel in the engine (bf16 tensor cores, cp.async).
inline constexpr int kMinComputeMajor = 8;
inline constexpr int kMinComputeMinor = 0;

// Release flavours, named by the CUDA major version of the toolkit they are built with.
enum class Flavour { kNone, kCuda12, kCuda13 };

// "cu12", "cu13" or "none".
const char* flavour_name(Flavour f);

// The flavour a CUDA toolkit/runtime version (e.g. 13040 for 13.4) belongs to; kNone outside 12.x / 13.x.
Flavour flavour_of_cuda_version(int cuda_version);

// The newest flavour a driver can run, from cudaDriverGetVersion() (the newest CUDA version it supports).
// CUDA minor-version compatibility lets a 13.x build run on any 13.0+ driver, and a 12.x build on any 12.0+ one.
Flavour flavour_for_driver(int driver_version);

// Lowest driver version (as cudaDriverGetVersion reports it) for a flavour: 12000 or 13000; 0 for kNone.
int min_driver_version(Flavour f);

// Human-readable minimum driver releases for a flavour, e.g. "Linux >= 580.65.06, Windows >= 580.88".
const char* min_driver_release(Flavour f);

// One entry of the CUDA architecture list the engine was compiled for (CMAKE_CUDA_ARCHITECTURES syntax:
// "86-real" = SASS only, "120-virtual" = PTX only, "120" = both).
struct CudaArch {
  int major = 0;
  int minor = 0;
  bool sass = false;  // device code that runs on the same major and an equal or higher minor
  bool ptx = false;   // intermediate code that the driver JIT-compiles for any equal or newer architecture
  bool exact_minor = false;  // "a" (arch-specific) SASS: runs on exactly major.minor
};

// Parses a comma- or semicolon-separated architecture list. Entries the probe cannot reason about
// ("native", "all", "all-major") yield an empty list, which compatible_image() treats as "assume compatible".
std::vector<CudaArch> parse_cuda_architectures(std::string_view list);

// The architecture list of this build (from CMake), as parsed by parse_cuda_architectures().
const std::vector<CudaArch>& build_architectures();
// The same list as CMake spelled it, e.g. "80-real,86-real,89-real,120".
const char* build_architectures_string();

// Whether a device of compute capability major.minor can load a kernel image of `archs`.
// `ptx_jit_ok` is false when the driver is older than the toolkit that emitted the PTX (it cannot JIT it).
bool compatible_image(const std::vector<CudaArch>& archs, int cc_major, int cc_minor, bool ptx_jit_ok);

struct DeviceInfo {
  int index = -1;
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int multiprocessors = 0;
  bool integrated = false;
  std::uint64_t total_bytes = 0;
  std::uint64_t free_bytes = 0;  // 0 when not queried (ProbeOptions::query_free_memory) or the query failed
  std::string free_memory_source;  // how free_bytes was measured: "nvml", "driver" or "" (not measured)
  std::string pci_bus_id;        // "0000:01:00.0"; empty if unavailable
  bool supported = false;
  std::string reason;            // why the device is unsupported; empty when supported
};

// Verdict for one device of this build on a driver. Sets `supported` and `reason` only.
void evaluate_device(DeviceInfo& dev, int driver_version, int build_cuda_version, const std::vector<CudaArch>& archs);

struct ProbeResult {
  int driver_version = 0;      // cudaDriverGetVersion(); 0 = no driver
  int runtime_version = 0;     // cudaRuntimeGetVersion() of the CUDA runtime in this process
  int build_cuda_version = 0;  // CUDART_VERSION the engine was compiled against
  Flavour build_flavour = Flavour::kNone;
  bool driver_ok = false;      // the driver can run this build
  std::vector<DeviceInfo> devices;
  int supported_devices = 0;
  Flavour recommended = Flavour::kNone;  // flavour to install on this machine
  std::string reason;                    // summary; explains an empty device list or kNone
};

// The flavour to install: the newest the driver supports, provided some device has compute capability >= 8.0 and a
// kernel image in `archs` (the release flavours share the build's list). Sets result.recommended and appends to
// result.reason. Pure: used by probe() and by tests.
void recommend(ProbeResult& result, const std::vector<CudaArch>& archs = build_architectures());

// How ProbeOptions::query_free_memory measures: NVML first, else the driver API (kAuto); tests pin one.
enum class FreeMemoryMethod { kAuto, kNvml, kDriver };

struct ProbeOptions {
  // Free memory of each supported device. NVML needs no CUDA context; the driver-API fallback retains the device's
  // primary context for the query only (a few hundred MiB of VRAM for milliseconds when no engine holds it).
  // Off: free_bytes stays 0.
  bool query_free_memory = true;
  FreeMemoryMethod free_memory_method = FreeMemoryMethod::kAuto;
  // With query_free_memory: only this device ordinal is measured (mi_probe_device); -1 = every supported device.
  int free_memory_device = -1;
};

// Enumerates devices and evaluates them. Never throws for a missing driver or device.
ProbeResult probe(const ProbeOptions& options = {});

// Multi-line human-readable report of a probe result (for `moss-infer` diagnostics and logs).
std::string format_report(const ProbeResult& result);

}  // namespace mi::backend
