// The CUDA driver API and NVML, loaded at run time (dlopen / LoadLibrary), never linked: the engine must load and the
// probe must answer on machines without an NVIDIA driver. Used by the probe (free memory without leaving a CUDA
// context behind) and by the C API (restoring the caller's current context).
#pragma once

#include <cstdint>
#include <optional>

namespace mi::backend {

// Free and total memory of one device.
struct MemoryInfo {
  std::uint64_t free_bytes = 0;
  std::uint64_t total_bytes = 0;
};

// NVML's view of the device with this PCI bus id ("0000:01:00.0"). Creates no CUDA context. Empty when NVML
// (libnvidia-ml.so.1 / nvml.dll) is absent or fails.
std::optional<MemoryInfo> nvml_memory_info(const char* pci_bus_id);

// cuMemGetInfo inside the device's primary context, retained for the call and released after it: the context exists
// only for the duration of the call unless something else (an engine) already holds it. The calling thread's current
// context is unchanged. Empty when libcuda is absent or a call fails.
std::optional<MemoryInfo> driver_memory_info(int device_ordinal);

// Whether the primary context of `device_ordinal` is active (driver API). Empty when libcuda is absent or cuInit fails.
std::optional<bool> primary_context_active(int device_ordinal);

// Saves the calling thread's current CUDA context and makes it current again on destruction, whatever the code in
// between selected (cudaSetDevice). Without libcuda it falls back to the runtime's current device.
class ScopedCurrentContext {
 public:
  ScopedCurrentContext() noexcept;
  ~ScopedCurrentContext();
  ScopedCurrentContext(const ScopedCurrentContext&) = delete;
  ScopedCurrentContext& operator=(const ScopedCurrentContext&) = delete;

 private:
  bool driver_ = false;
  void* context_ = nullptr;  // CUcontext
  int device_ = -1;          // runtime fallback: cudaGetDevice() at construction, -1 if unknown
};

}  // namespace mi::backend
