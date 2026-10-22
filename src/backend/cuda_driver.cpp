#include "backend/cuda_driver.h"

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mi::backend {
namespace {

// A shared library opened once and kept for the life of the process (unloading a driver library is never safe while
// other code may still use it).
class Library {
 public:
#ifdef _WIN32
  // System32 only: these DLLs come with the driver, and a copy next to the executable must not shadow them.
  explicit Library(const wchar_t* name, const wchar_t* fallback_path = nullptr) {
    handle_ = LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (handle_ == nullptr && fallback_path != nullptr) handle_ = LoadLibraryW(fallback_path);
  }
  template <class F>
  F symbol(const char* name) const {
    return handle_ != nullptr ? reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(handle_, name))) : nullptr;
  }

 private:
  HMODULE handle_ = nullptr;
#else
  explicit Library(const char* name) : handle_(dlopen(name, RTLD_NOW | RTLD_LOCAL)) {}
  template <class F>
  F symbol(const char* name) const {
    return handle_ != nullptr ? reinterpret_cast<F>(dlsym(handle_, name)) : nullptr;
  }

 private:
  void* handle_ = nullptr;
#endif
};

// The driver API calls the probe and the context guard use. Every pointer is set, or `ok` is false.
struct Driver {
  decltype(&cuDeviceGet) device_get = nullptr;
  decltype(&cuDevicePrimaryCtxRetain) primary_retain = nullptr;
  decltype(&cuDevicePrimaryCtxRelease_v2) primary_release = nullptr;
  decltype(&cuDevicePrimaryCtxGetState) primary_state = nullptr;
  decltype(&cuCtxPushCurrent_v2) push = nullptr;
  decltype(&cuCtxPopCurrent_v2) pop = nullptr;
  decltype(&cuCtxGetCurrent) get_current = nullptr;
  decltype(&cuCtxSetCurrent) set_current = nullptr;
  decltype(&cuMemGetInfo_v2) mem_get_info = nullptr;
  // Device enumeration (driver_devices()); optional, not part of `ok`.
  decltype(&cuDeviceGetCount) device_count = nullptr;
  decltype(&cuDeviceGetName) device_name = nullptr;
  decltype(&cuDeviceGetAttribute) device_attribute = nullptr;
  decltype(&cuDeviceTotalMem_v2) device_total_mem = nullptr;
  decltype(&cuDeviceGetPCIBusId) device_pci_bus_id = nullptr;
  bool ok = false;
};

const Driver& driver() {
  static const Driver d = [] {
#ifdef _WIN32
    static const Library lib(L"nvcuda.dll");
#else
    static const Library lib("libcuda.so.1");
#endif
    Driver r;
    const auto init = lib.symbol<decltype(&cuInit)>("cuInit");
    r.device_get = lib.symbol<decltype(r.device_get)>("cuDeviceGet");
    r.primary_retain = lib.symbol<decltype(r.primary_retain)>("cuDevicePrimaryCtxRetain");
    r.primary_release = lib.symbol<decltype(r.primary_release)>("cuDevicePrimaryCtxRelease_v2");
    r.primary_state = lib.symbol<decltype(r.primary_state)>("cuDevicePrimaryCtxGetState");
    r.push = lib.symbol<decltype(r.push)>("cuCtxPushCurrent_v2");
    r.pop = lib.symbol<decltype(r.pop)>("cuCtxPopCurrent_v2");
    r.get_current = lib.symbol<decltype(r.get_current)>("cuCtxGetCurrent");
    r.set_current = lib.symbol<decltype(r.set_current)>("cuCtxSetCurrent");
    r.mem_get_info = lib.symbol<decltype(r.mem_get_info)>("cuMemGetInfo_v2");
    r.device_count = lib.symbol<decltype(r.device_count)>("cuDeviceGetCount");
    r.device_name = lib.symbol<decltype(r.device_name)>("cuDeviceGetName");
    r.device_attribute = lib.symbol<decltype(r.device_attribute)>("cuDeviceGetAttribute");
    r.device_total_mem = lib.symbol<decltype(r.device_total_mem)>("cuDeviceTotalMem_v2");
    r.device_pci_bus_id = lib.symbol<decltype(r.device_pci_bus_id)>("cuDeviceGetPCIBusId");
    // cuInit creates no context; it fails without a driver or with no visible device (CUDA_VISIBLE_DEVICES="").
    r.ok = init != nullptr && r.device_get != nullptr && r.primary_retain != nullptr && r.primary_release != nullptr &&
           r.primary_state != nullptr && r.push != nullptr && r.pop != nullptr && r.get_current != nullptr &&
           r.set_current != nullptr && r.mem_get_info != nullptr && init(0) == CUDA_SUCCESS;
    return r;
  }();
  return d;
}

// NVML's C interface, declared here so the build needs no nvml.h (not every toolkit install has it).
using nvmlReturn_t = int;
using nvmlDevice_t = struct nvmlDevice_st*;
struct nvmlMemory_t {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};
constexpr nvmlReturn_t kNvmlSuccess = 0;

struct Nvml {
  nvmlReturn_t (*init)() = nullptr;
  nvmlReturn_t (*shutdown)() = nullptr;
  nvmlReturn_t (*by_pci_bus_id)(const char*, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*memory_info)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
  bool ok = false;
};

const Nvml& nvml() {
  static const Nvml n = [] {
#ifdef _WIN32
    // Drivers before R418 installed it under NVSMI instead of System32.
    static const Library lib(L"nvml.dll", L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
#else
    static const Library lib("libnvidia-ml.so.1");
#endif
    Nvml r;
    r.init = lib.symbol<decltype(r.init)>("nvmlInit_v2");
    r.shutdown = lib.symbol<decltype(r.shutdown)>("nvmlShutdown");
    r.by_pci_bus_id = lib.symbol<decltype(r.by_pci_bus_id)>("nvmlDeviceGetHandleByPciBusId_v2");
    r.memory_info = lib.symbol<decltype(r.memory_info)>("nvmlDeviceGetMemoryInfo");
    r.ok = r.init != nullptr && r.shutdown != nullptr && r.by_pci_bus_id != nullptr && r.memory_info != nullptr;
    return r;
  }();
  return n;
}

}  // namespace

std::optional<MemoryInfo> nvml_memory_info(const char* pci_bus_id) {
  const Nvml& n = nvml();
  if (!n.ok || pci_bus_id == nullptr || *pci_bus_id == '\0' || n.init() != kNvmlSuccess) return std::nullopt;
  std::optional<MemoryInfo> out;
  nvmlDevice_t dev = nullptr;
  nvmlMemory_t mem{};
  if (n.by_pci_bus_id(pci_bus_id, &dev) == kNvmlSuccess && n.memory_info(dev, &mem) == kNvmlSuccess) {
    out = MemoryInfo{mem.free, mem.total};
  }
  n.shutdown();
  return out;
}

std::optional<MemoryInfo> driver_memory_info(int device_ordinal) {
  const Driver& d = driver();
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (!d.ok || d.device_get(&dev, device_ordinal) != CUDA_SUCCESS || d.primary_retain(&ctx, dev) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  std::optional<MemoryInfo> out;
  if (d.push(ctx) == CUDA_SUCCESS) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (d.mem_get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS) out = MemoryInfo{free_bytes, total_bytes};
    CUcontext popped = nullptr;
    d.pop(&popped);
  }
  // The last release destroys the primary context (and frees its memory); an engine's own reference keeps it.
  d.primary_release(dev);
  return out;
}

std::optional<std::vector<DriverDevice>> driver_devices() {
  const Driver& d = driver();
  if (!d.ok || d.device_count == nullptr || d.device_name == nullptr || d.device_attribute == nullptr ||
      d.device_total_mem == nullptr || d.device_pci_bus_id == nullptr) {
    return std::nullopt;
  }
  int count = 0;
  if (d.device_count(&count) != CUDA_SUCCESS) return std::nullopt;
  std::vector<DriverDevice> out;
  for (int i = 0; i < count; ++i) {
    CUdevice dev = 0;
    char name[256] = {};
    DriverDevice r;
    std::size_t total = 0;
    int integrated = 0;
    if (d.device_get(&dev, i) != CUDA_SUCCESS || d.device_name(name, static_cast<int>(sizeof(name)), dev) != CUDA_SUCCESS ||
        d.device_attribute(&r.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS ||
        d.device_attribute(&r.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS ||
        d.device_attribute(&r.multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) != CUDA_SUCCESS ||
        d.device_attribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) != CUDA_SUCCESS ||
        d.device_total_mem(&total, dev) != CUDA_SUCCESS) {
      return std::nullopt;
    }
    r.name = name;
    r.integrated = integrated != 0;
    r.total_bytes = total;
    char bus_id[32] = {};
    if (d.device_pci_bus_id(bus_id, static_cast<int>(sizeof(bus_id)), dev) == CUDA_SUCCESS) r.pci_bus_id = bus_id;
    out.push_back(std::move(r));
  }
  return out;
}

std::optional<bool> primary_context_active(int device_ordinal) {
  const Driver& d = driver();
  CUdevice dev = 0;
  unsigned flags = 0;
  int active = 0;
  if (!d.ok || d.device_get(&dev, device_ordinal) != CUDA_SUCCESS || d.primary_state(dev, &flags, &active) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  return active != 0;
}

ScopedCurrentContext::ScopedCurrentContext() noexcept {
  const Driver& d = driver();
  CUcontext ctx = nullptr;
  if (d.ok && d.get_current(&ctx) == CUDA_SUCCESS) {
    driver_ = true;
    context_ = ctx;
  } else if (cudaGetDevice(&device_) != cudaSuccess) {
    cudaGetLastError();
    device_ = -1;
  }
}

ScopedCurrentContext::~ScopedCurrentContext() {
  if (driver_) {
    driver().set_current(static_cast<CUcontext>(context_));
    return;
  }
  int now = -1;
  if (device_ >= 0 && cudaGetDevice(&now) == cudaSuccess && now != device_ && cudaSetDevice(device_) != cudaSuccess) {
    cudaGetLastError();
  }
}

}  // namespace mi::backend
