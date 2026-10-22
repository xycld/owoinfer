#include "backend/probe.h"

#include <cuda_runtime.h>

#include <charconv>
#include <cstddef>
#include <tuple>

#include "backend/cuda_driver.h"

// The CMake architecture list of this build, set on this file only by cmake/modules/backend.cmake.
#ifndef MOSSINFER_CUDA_ARCHITECTURES
#define MOSSINFER_CUDA_ARCHITECTURES ""
#endif

namespace mi::backend {
namespace {

std::string cuda_version_string(int v) {
  return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

std::string cc_string(int major, int minor) { return std::to_string(major) + "." + std::to_string(minor); }

// One entry such as "89", "86-real", "120-virtual" or "120a-real"; false if it is not of that form.
bool parse_arch(std::string_view s, CudaArch& out) {
  bool sass = true;
  bool ptx = true;
  if (const std::size_t dash = s.find('-'); dash != std::string_view::npos) {
    const std::string_view kind = s.substr(dash + 1);
    if (kind == "real") {
      ptx = false;
    } else if (kind == "virtual") {
      sass = false;
    } else {
      return false;
    }
    s = s.substr(0, dash);
  }
  bool exact = false;
  if (!s.empty() && (s.back() == 'a' || s.back() == 'f')) {
    // 'a' (arch-specific) code runs only on exactly this compute capability; 'f' (family) code on the same major,
    // which is the plain SASS rule. Neither variant's PTX can be JIT-compiled for another architecture.
    exact = s.back() == 'a';
    ptx = false;
    s.remove_suffix(1);
  }
  int num = 0;
  const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), num);
  if (ec != std::errc() || end != s.data() + s.size() || num < 10) return false;
  out = CudaArch{num / 10, num % 10, sass, ptx, exact};
  return true;
}

}  // namespace

const char* flavour_name(Flavour f) {
  switch (f) {
    case Flavour::kCuda12: return "cu12";
    case Flavour::kCuda13: return "cu13";
    case Flavour::kNone: break;
  }
  return "none";
}

Flavour flavour_of_cuda_version(int cuda_version) {
  switch (cuda_version / 1000) {
    case 12: return Flavour::kCuda12;
    case 13: return Flavour::kCuda13;
    default: return Flavour::kNone;
  }
}

Flavour flavour_for_driver(int driver_version) {
  if (driver_version >= 13000) return Flavour::kCuda13;
  if (driver_version >= 12000) return Flavour::kCuda12;
  return Flavour::kNone;
}

int min_driver_version(Flavour f) {
  switch (f) {
    case Flavour::kCuda12: return 12000;
    case Flavour::kCuda13: return 13000;
    case Flavour::kNone: break;
  }
  return 0;
}

const char* min_driver_release(Flavour f) {
  // CUDA minor-version compatibility table (CUDA 12.0 and 13.0 release notes).
  switch (f) {
    case Flavour::kCuda12: return "Linux >= 525.60.13, Windows >= 527.41";
    case Flavour::kCuda13: return "Linux >= 580.65.06, Windows >= 580.88";
    case Flavour::kNone: break;
  }
  return "";
}

std::vector<CudaArch> parse_cuda_architectures(std::string_view list) {
  std::vector<CudaArch> archs;
  while (!list.empty()) {
    const std::size_t sep = list.find_first_of(",; ");
    const std::string_view item = list.substr(0, sep);
    list = sep == std::string_view::npos ? std::string_view() : list.substr(sep + 1);
    if (item.empty()) continue;
    CudaArch a;
    if (!parse_arch(item, a)) return {};  // "native", "all", ...: unknown, assume compatible
    archs.push_back(a);
  }
  return archs;
}

const char* build_architectures_string() { return MOSSINFER_CUDA_ARCHITECTURES; }

const std::vector<CudaArch>& build_architectures() {
  static const std::vector<CudaArch> archs = parse_cuda_architectures(MOSSINFER_CUDA_ARCHITECTURES);
  return archs;
}

bool compatible_image(const std::vector<CudaArch>& archs, int cc_major, int cc_minor, bool ptx_jit_ok) {
  if (archs.empty()) return true;
  for (const CudaArch& a : archs) {
    if (a.sass && a.major == cc_major && (a.exact_minor ? a.minor == cc_minor : a.minor <= cc_minor)) return true;
    if (a.ptx && ptx_jit_ok && std::tie(a.major, a.minor) <= std::tie(cc_major, cc_minor)) return true;
  }
  return false;
}

void evaluate_device(DeviceInfo& dev, int driver_version, int build_cuda_version, const std::vector<CudaArch>& archs) {
  dev.supported = false;
  dev.reason.clear();
  if (std::tie(dev.cc_major, dev.cc_minor) < std::tie(kMinComputeMajor, kMinComputeMinor)) {
    dev.reason = "compute capability " + cc_string(dev.cc_major, dev.cc_minor) + " is below the required " +
                 cc_string(kMinComputeMajor, kMinComputeMinor) + " (Ampere or newer)";
    return;
  }
  const Flavour build = flavour_of_cuda_version(build_cuda_version);
  const int min_driver = build == Flavour::kNone ? build_cuda_version / 1000 * 1000 : min_driver_version(build);
  if (driver_version < min_driver) {
    dev.reason = driver_version == 0 ? std::string("no NVIDIA driver is loaded")
                                     : "the driver supports CUDA " + cuda_version_string(driver_version) +
                                           ", this " + flavour_name(build) + " build needs CUDA " +
                                           cuda_version_string(min_driver) + " or newer (" +
                                           min_driver_release(build) + ")";
    return;
  }
  const bool ptx_jit_ok = driver_version >= build_cuda_version;
  if (!compatible_image(archs, dev.cc_major, dev.cc_minor, ptx_jit_ok)) {
    dev.reason = "this build has no kernel image for compute capability " + cc_string(dev.cc_major, dev.cc_minor) +
                 " (built for " + build_architectures_string() + ")";
    if (compatible_image(archs, dev.cc_major, dev.cc_minor, true)) {
      dev.reason += "; its PTX needs a driver for CUDA " + cuda_version_string(build_cuda_version) +
                    " or newer to JIT-compile, the driver supports " + cuda_version_string(driver_version);
    }
    return;
  }
  dev.supported = true;
}

void recommend(ProbeResult& r, const std::vector<CudaArch>& archs) {
  r.recommended = Flavour::kNone;
  auto append = [&r](const std::string& s) { r.reason += (r.reason.empty() ? "" : "; ") + s; };
  bool any_capable = false;
  bool any_image = false;
  std::string no_image;  // "sm_90, sm_100" of capable devices without a kernel image
  for (const DeviceInfo& d : r.devices) {
    if (std::tie(d.cc_major, d.cc_minor) < std::tie(kMinComputeMajor, kMinComputeMinor)) continue;
    any_capable = true;
    // The release flavours share this build's architecture list. PTX is assumed JIT-able: the recommended flavour's
    // driver is new enough for its own toolkit.
    if (compatible_image(archs, d.cc_major, d.cc_minor, true)) {
      any_image = true;
    } else {
      no_image += (no_image.empty() ? "sm_" : ", sm_") + std::to_string(d.cc_major) + std::to_string(d.cc_minor);
    }
  }
  if (r.devices.empty()) {
    append("no flavour recommended: no CUDA device");
    return;
  }
  if (!any_capable) {
    append("no flavour recommended: no device has compute capability >= " +
           cc_string(kMinComputeMajor, kMinComputeMinor));
    return;
  }
  if (!any_image) {
    append("no flavour recommended: no kernel image for " + no_image + " in this build (built for " +
           build_architectures_string() + ")");
    return;
  }
  const Flavour f = flavour_for_driver(r.driver_version);
  if (f == Flavour::kNone) {
    append(r.driver_version == 0 ? std::string("no flavour recommended: no NVIDIA driver is loaded")
                                 : "no flavour recommended: the driver supports CUDA " +
                                       cuda_version_string(r.driver_version) + ", MossInfer needs CUDA 12.0 or newer (" +
                                       min_driver_release(Flavour::kCuda12) + ")");
    return;
  }
  r.recommended = f;
  append(std::string("recommended flavour ") + flavour_name(f) + " (the driver supports CUDA " +
         cuda_version_string(r.driver_version) + ")");
}

ProbeResult probe(const ProbeOptions& options) {
  ProbeResult r;
  r.build_cuda_version = CUDART_VERSION;
  r.build_flavour = flavour_of_cuda_version(CUDART_VERSION);
  // Each call below reports a missing driver as an error code; none of them aborts. cudaGetLastError() clears the
  // runtime's last-error state so callers that check it later are not confused by the probe.
  if (cudaDriverGetVersion(&r.driver_version) != cudaSuccess) r.driver_version = 0;
  if (cudaRuntimeGetVersion(&r.runtime_version) != cudaSuccess) r.runtime_version = 0;
  const int min_driver =
      r.build_flavour == Flavour::kNone ? r.build_cuda_version / 1000 * 1000 : min_driver_version(r.build_flavour);
  r.driver_ok = r.driver_version >= min_driver && r.driver_version > 0;

  int count = 0;
  if (const cudaError_t err = cudaGetDeviceCount(&count); err != cudaSuccess) {
    cudaGetLastError();
    r.reason = std::string("no usable CUDA device: ") + cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")";
    recommend(r);
    return r;
  }
  if (count == 0) r.reason = "no CUDA device found";

  for (int i = 0; i < count; ++i) {
    DeviceInfo d;
    d.index = i;
    cudaDeviceProp p{};
    if (const cudaError_t err = cudaGetDeviceProperties(&p, i); err != cudaSuccess) {
      cudaGetLastError();
      d.reason = std::string("device properties unavailable: ") + cudaGetErrorName(err);
      r.devices.push_back(std::move(d));
      continue;
    }
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.multiprocessors = p.multiProcessorCount;
    d.integrated = p.integrated != 0;
    d.total_bytes = p.totalGlobalMem;
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), i) == cudaSuccess) {
      d.pci_bus_id = bus_id;
    } else {
      cudaGetLastError();
    }
    evaluate_device(d, r.driver_version, r.build_cuda_version, build_architectures());
    r.supported_devices += d.supported ? 1 : 0;
    r.devices.push_back(std::move(d));
  }

  if (options.query_free_memory) {
    // Never through the runtime: cudaMemGetInfo would create a primary context that stays for the life of the process.
    for (DeviceInfo& d : r.devices) {
      if (!d.supported) continue;  // devices the engine will not use are not touched
      if (options.free_memory_device >= 0 && d.index != options.free_memory_device) continue;
      std::optional<MemoryInfo> m;
      if (options.free_memory_method != FreeMemoryMethod::kDriver) {
        m = nvml_memory_info(d.pci_bus_id.c_str());
        if (m) d.free_memory_source = "nvml";
      }
      if (!m && options.free_memory_method != FreeMemoryMethod::kNvml) {
        m = driver_memory_info(d.index);
        if (m) d.free_memory_source = "driver";
      }
      if (m) d.free_bytes = m->free_bytes;
    }
  }
  recommend(r);
  return r;
}

std::string format_report(const ProbeResult& r) {
  constexpr double kMiB = 1024.0 * 1024.0;
  std::string s;
  s += "driver: " + (r.driver_version > 0 ? "CUDA " + cuda_version_string(r.driver_version) : std::string("none")) +
       "\n";
  s += "runtime: CUDA " + cuda_version_string(r.runtime_version) + " (built with CUDA " +
       cuda_version_string(r.build_cuda_version) + ", flavour " + flavour_name(r.build_flavour) + ", archs " +
       build_architectures_string() + ")\n";
  s += std::string("driver can run this build: ") + (r.driver_ok ? "yes" : "no") + "\n";
  for (const DeviceInfo& d : r.devices) {
    s += "device " + std::to_string(d.index) + ": " + d.name + ", sm_" + std::to_string(d.cc_major) +
         std::to_string(d.cc_minor) + ", " + std::to_string(d.multiprocessors) + " SMs, " +
         std::to_string(static_cast<long long>(static_cast<double>(d.total_bytes) / kMiB)) + " MiB total";
    if (d.free_bytes > 0) {
      s += ", " + std::to_string(static_cast<long long>(static_cast<double>(d.free_bytes) / kMiB)) + " MiB free (" +
           d.free_memory_source + ")";
    }
    if (!d.pci_bus_id.empty()) s += ", PCI " + d.pci_bus_id;
    if (d.integrated) s += ", integrated";
    s += d.supported ? std::string(": supported\n") : ": unsupported (" + d.reason + ")\n";
  }
  s += std::string("recommended flavour: ") + flavour_name(r.recommended) + "\n";
  if (!r.reason.empty()) s += "note: " + r.reason + "\n";
  return s;
}

}  // namespace mi::backend
