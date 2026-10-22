// Test helper, independent of src/: does this process hold a CUDA context? Asks the driver (primary context state)
// and NVML (the device's compute-process list, which also shows non-primary contexts). Loads libcuda / NVML at run
// time, so it builds and answers "unknown" on machines without a driver.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

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
#include <unistd.h>
#endif

namespace mi::testing {

namespace detail {

inline void* load_symbol(const char* library, const char* name) {
#ifdef _WIN32
  HMODULE h = LoadLibraryExA(library, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  return h != nullptr ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr;
#else
  void* h = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  return h != nullptr ? dlsym(h, name) : nullptr;
#endif
}

#ifdef _WIN32
inline const char* kCudaLibrary = "nvcuda.dll";
inline const char* kNvmlLibrary = "nvml.dll";
inline unsigned current_pid() { return static_cast<unsigned>(GetCurrentProcessId()); }
#else
inline const char* kCudaLibrary = "libcuda.so.1";
inline const char* kNvmlLibrary = "libnvidia-ml.so.1";
inline unsigned current_pid() { return static_cast<unsigned>(getpid()); }
#endif

}  // namespace detail

// Ordinals of the visible devices whose primary context is active. Empty optional: no driver or no visible device.
inline std::optional<std::vector<int>> active_primary_contexts() {
  using Init = int (*)(unsigned);
  using Count = int (*)(int*);
  using Get = int (*)(int*, int);
  using State = int (*)(int, unsigned*, int*);
  const auto init = reinterpret_cast<Init>(detail::load_symbol(detail::kCudaLibrary, "cuInit"));
  const auto count = reinterpret_cast<Count>(detail::load_symbol(detail::kCudaLibrary, "cuDeviceGetCount"));
  const auto get = reinterpret_cast<Get>(detail::load_symbol(detail::kCudaLibrary, "cuDeviceGet"));
  const auto state = reinterpret_cast<State>(detail::load_symbol(detail::kCudaLibrary, "cuDevicePrimaryCtxGetState"));
  int n = 0;
  if (init == nullptr || count == nullptr || get == nullptr || state == nullptr || init(0) != 0 || count(&n) != 0) {
    return std::nullopt;
  }
  std::vector<int> active;
  for (int i = 0; i < n; ++i) {
    int dev = 0;
    unsigned flags = 0;
    int on = 0;
    if (get(&dev, i) == 0 && state(dev, &flags, &on) == 0 && on != 0) active.push_back(i);
  }
  return active;
}

// The calling thread's current CUDA context (nullptr: none). Empty optional: no driver or no visible device.
inline std::optional<void*> current_context() {
  using Init = int (*)(unsigned);
  using GetCurrent = int (*)(void**);
  const auto init = reinterpret_cast<Init>(detail::load_symbol(detail::kCudaLibrary, "cuInit"));
  const auto get_current = reinterpret_cast<GetCurrent>(detail::load_symbol(detail::kCudaLibrary, "cuCtxGetCurrent"));
  void* ctx = nullptr;
  if (init == nullptr || get_current == nullptr || init(0) != 0 || get_current(&ctx) != 0) return std::nullopt;
  return ctx;
}

// Whether NVML lists this process as a compute process on any GPU. Empty optional: NVML unavailable.
inline std::optional<bool> nvml_lists_this_process() {
  struct ProcessInfo {  // nvmlProcessInfo_v2_t
    unsigned pid;
    unsigned long long used_gpu_memory;
    unsigned gpu_instance_id;
    unsigned compute_instance_id;
  };
  using Init = int (*)();
  using Shutdown = int (*)();
  using Count = int (*)(unsigned*);
  using Handle = int (*)(unsigned, void**);
  using Processes = int (*)(void*, unsigned*, ProcessInfo*);
  const auto init = reinterpret_cast<Init>(detail::load_symbol(detail::kNvmlLibrary, "nvmlInit_v2"));
  const auto shutdown = reinterpret_cast<Shutdown>(detail::load_symbol(detail::kNvmlLibrary, "nvmlShutdown"));
  const auto count = reinterpret_cast<Count>(detail::load_symbol(detail::kNvmlLibrary, "nvmlDeviceGetCount_v2"));
  const auto handle = reinterpret_cast<Handle>(detail::load_symbol(detail::kNvmlLibrary, "nvmlDeviceGetHandleByIndex_v2"));
  const auto processes =
      reinterpret_cast<Processes>(detail::load_symbol(detail::kNvmlLibrary, "nvmlDeviceGetComputeRunningProcesses_v3"));
  if (init == nullptr || shutdown == nullptr || count == nullptr || handle == nullptr || processes == nullptr ||
      init() != 0) {
    return std::nullopt;
  }
  std::optional<bool> listed = false;
  unsigned n = 0;
  if (count(&n) != 0) listed.reset();
  for (unsigned i = 0; listed && i < n; ++i) {
    void* dev = nullptr;
    std::vector<ProcessInfo> infos(256);
    unsigned size = static_cast<unsigned>(infos.size());
    if (handle(i, &dev) != 0 || processes(dev, &size, infos.data()) != 0) {
      listed.reset();
      break;
    }
    for (unsigned k = 0; k < size; ++k) {
      if (infos[k].pid == detail::current_pid()) listed = true;
    }
  }
  shutdown();
  return listed;
}

// "" when the process holds no CUDA context (or nothing can tell), else a description for a test failure.
inline std::string cuda_context_report() {
  std::string out;
  if (const auto active = active_primary_contexts(); active && !active->empty()) {
    out += "primary context active on device";
    for (const int d : *active) out += " " + std::to_string(d);
  }
  if (const auto listed = nvml_lists_this_process(); listed && *listed) {
    out += std::string(out.empty() ? "" : "; ") + "NVML lists this process as a compute process";
  }
  return out;
}

}  // namespace mi::testing
