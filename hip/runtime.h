// The HIP device runtime: device listing and choice, device and pinned host buffers, streams and events, all RAII.
// Failures throw std::runtime_error with the HIP call and its error string. The same source builds for AMD (ROCm) and,
// through HIP's NVIDIA headers, for a CUDA device, where the kernels run the scalar leaf (mma_leaf.h).
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mi::hip {

// Throws std::runtime_error("hip: <what>: <error string>") unless e is hipSuccess.
void check(hipError_t e, const char* what);

// The device architectures the kernels of this build were compiled for, comma-separated: AMD's among gfx1100, gfx1200
// and gfx1201 (e.g. "gfx1100,gfx1200,gfx1201"), or "nvidia".
const char* build_target();

struct DeviceInfo {
  int index = 0;
  std::string name;
  std::string arch;  // AMD: gcnArchName without its feature suffix ("gfx1200"); NVIDIA: "sm_89"
  std::uint64_t total_memory = 0;
  int compute_units = 0;
  std::string unusable_reason;  // empty when this build's kernels run on the device
};

// Every device the HIP runtime sees, usable or not. Empty when there is none or no driver.
std::vector<DeviceInfo> list_devices();

// The device a process runs on: made current at construction (hipSetDevice).
class Device {
 public:
  // index < 0: the first usable device.
  explicit Device(int index = -1);
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  const DeviceInfo& info() const { return info_; }
  // Device memory still free for this process (hipMemGetInfo).
  std::uint64_t free_memory() const;
  // Makes this the calling thread's device.
  void make_current() const;
  // Waits for all work on the device.
  void synchronize() const;

 private:
  DeviceInfo info_;
};

// Device memory, uninitialized.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&& o) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() const { return p_; }
  template <class T>
  T* as() const {
    return static_cast<T*>(p_);
  }
  std::size_t bytes() const { return bytes_; }

  // Copies synchronous with respect to the host and every stream; offset in bytes from the start of this buffer. Each
  // waits for all work queued on the device before it copies, and returns once the copy has landed: no kernel queued
  // earlier on any stream still reads (upload) or writes (download) the buffer, and a kernel queued afterwards on any
  // stream, non-blocking ones included, sees the new contents. They cost a device-wide synchronization each: for copies
  // that overlap kernels, use a HostBuffer, copy_async and events.
  void upload(const void* src, std::size_t bytes, std::size_t offset = 0);
  void download(void* dst, std::size_t bytes, std::size_t offset = 0) const;
  // Stream-ordered fill with a byte value.
  void fill(int byte, hipStream_t stream);

 private:
  void* p_ = nullptr;
  std::size_t bytes_ = 0;
};

// Page-locked host memory: the source of asynchronous host-to-device copies.
class HostBuffer {
 public:
  HostBuffer() = default;
  explicit HostBuffer(std::size_t bytes);
  ~HostBuffer();
  HostBuffer(HostBuffer&& o) noexcept;
  HostBuffer& operator=(HostBuffer&& o) noexcept;
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  void* data() const { return p_; }
  template <class T>
  T* as() const {
    return static_cast<T*>(p_);
  }
  std::size_t bytes() const { return bytes_; }

 private:
  void* p_ = nullptr;
  std::size_t bytes_ = 0;
};

class Stream {
 public:
  Stream();  // non-blocking: does not wait on the null stream
  ~Stream();
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  hipStream_t get() const { return s_; }
  void synchronize() const;

 private:
  hipStream_t s_ = nullptr;
};

class Event {
 public:
  explicit Event(bool timing = false);
  ~Event();
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  hipEvent_t get() const { return e_; }
  void record(hipStream_t stream);
  // Makes `stream` wait on the device for this event's last record.
  void wait(hipStream_t stream) const;
  void synchronize() const;
  // Milliseconds from `start`'s record to this one's; both timing events, both completed.
  float elapsed_ms(const Event& start) const;

 private:
  hipEvent_t e_ = nullptr;
};

// Asynchronous copy on `stream`, ordered with that stream only: other streams order against it with Events. Host
// memory must be a HostBuffer's for the copy to overlap kernels, and must stay unchanged until the copy completes.
void copy_async(void* dst, const void* src, std::size_t bytes, hipStream_t stream);

}  // namespace mi::hip
