// The CUDA driver API and NVML, loaded at run time (dlopen / LoadLibrary), never linked: the engine must load and the
// probe must answer on machines without an NVIDIA driver. Used by the probe (free memory without leaving a CUDA
// context behind), by the C API (restoring the caller's current context) and by runtime/exchange.h (a host's device
// pointers).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mi::backend {

// Free and total memory of one device.
struct MemoryInfo {
  std::uint64_t free_bytes = 0;
  std::uint64_t total_bytes = 0;
};

// One device as the driver API reports it, in CUDA ordinal order.
struct DriverDevice {
  std::string name;
  int cc_major = 0;
  int cc_minor = 0;
  int multiprocessors = 0;
  bool integrated = false;
  std::uint64_t total_bytes = 0;
  std::string pci_bus_id;  // "0000:01:00.0"; empty if unavailable
};

// Every device the driver sees (driver API: cuDeviceGetCount and attribute queries, no context). The probe uses it
// when the CUDA runtime cannot enumerate devices because the driver is older than this build's CUDA version. Empty when
// libcuda is absent or a call fails, and on drivers before R450 (CUDA 11), which lack the context entry points this
// file requires; the probe then reports no device, which is still the right verdict there.
std::optional<std::vector<DriverDevice>> driver_devices();

// NVML's view of the device with this PCI bus id ("0000:01:00.0"). Creates no CUDA context. Empty when NVML
// (libnvidia-ml.so.1 / nvml.dll) is absent or fails.
std::optional<MemoryInfo> nvml_memory_info(const char* pci_bus_id);

// cuMemGetInfo inside the device's primary context, retained for the call and released after it: the context exists
// only for the duration of the call unless something else (an engine) already holds it. The calling thread's current
// context is unchanged. Empty when libcuda is absent or a call fails.
std::optional<MemoryInfo> driver_memory_info(int device_ordinal);

// Whether the primary context of `device_ordinal` is active (driver API). Empty when libcuda is absent or cuInit fails.
std::optional<bool> primary_context_active(int device_ordinal);

// Retains the primary context of `device_ordinal` (cuDevicePrimaryCtxRetain) and returns it (a CUcontext); the caller
// releases it with release_primary_context(device_ordinal), so the handle stays valid in between. Empty (nothing
// retained) when libcuda is absent or a call fails.
std::optional<void*> retain_primary_context(int device_ordinal);
void release_primary_context(int device_ordinal);

// What the driver knows about an address (cuPointerGetAttributes, one call): the component API checks a host's device
// tensors against it. An address the driver does not know (pageable host memory) is kUnknown.
struct PointerInfo {
  enum Type { kUnknown, kDevice, kHost, kOther };
  Type memory_type = kUnknown;  // kHost: pinned or registered host memory
  int device = -1;              // device ordinal
  void* context = nullptr;      // CUcontext; nullptr also for context-independent memory (VMM, memory pools)
  bool managed = false;
  std::string driver_error;     // the call failed: the driver's error name (memory_type kUnknown then)
};
// Empty when libcuda or its cuPointerGetAttributes is absent.
std::optional<PointerInfo> pointer_info(const void* ptr);

// The bytes from `ptr` on, up to `want`, that are mapped device memory of `device` without a gap: allocation after
// allocation (cuMemGetAddressRange), each next one starting where the last ended. The pointer attributes' range is not
// enough: for VMM memory (cuMemAddressReserve + cuMemMap, torch's expandable segments) it spans the whole reserved
// address range, mapped or not, while cuMemGetAddressRange gives the mapped allocation holding the address. Empty when
// libcuda is absent.
std::optional<std::size_t> mapped_device_bytes(const void* ptr, std::size_t want, int device);

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
