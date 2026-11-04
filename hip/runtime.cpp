#include "runtime.h"

#include <stdexcept>
#include <utility>

#ifndef MI_HIP_TARGET
#error "MI_HIP_TARGET must name the build's device target (CMakeLists.txt)"
#endif

namespace mi::hip {

void check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string("hip: ") + what + ": " + hipGetErrorString(e));
}

const char* build_target() { return MI_HIP_TARGET; }

namespace {

DeviceInfo describe(int index) {
  hipDeviceProp_t p{};
  check(hipGetDeviceProperties(&p, index), "hipGetDeviceProperties");
  DeviceInfo d;
  d.index = index;
  d.name = p.name;
  d.total_memory = p.totalGlobalMem;
  d.compute_units = p.multiProcessorCount;
#if defined(__HIP_PLATFORM_AMD__)
  d.arch = p.gcnArchName;
  if (const std::size_t colon = d.arch.find(':'); colon != std::string::npos) d.arch.resize(colon);
  // A code object runs only on the ISA it was compiled for.
  if (d.arch != MI_HIP_TARGET) d.unusable_reason = "this build's kernels are for " MI_HIP_TARGET ", the device is " + d.arch;
#else
  d.arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
  // The build holds the kernels for MI_HIP_CUDA_ARCH and its PTX, which runs on that compute capability and newer.
  if (p.major * 10 + p.minor < MI_HIP_CUDA_ARCH) {
    d.unusable_reason = "this build's kernels need compute capability " + std::to_string(MI_HIP_CUDA_ARCH / 10) + "." +
                        std::to_string(MI_HIP_CUDA_ARCH % 10) + " or newer, the device has " + std::to_string(p.major) +
                        "." + std::to_string(p.minor);
  }
#endif
  return d;
}

}  // namespace

std::vector<DeviceInfo> list_devices() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();  // no driver or no device: not an error to keep
    return {};
  }
  std::vector<DeviceInfo> out;
  for (int i = 0; i < count; ++i) out.push_back(describe(i));
  return out;
}

Device::Device(int index) {
  const std::vector<DeviceInfo> devices = list_devices();
  if (devices.empty()) throw std::runtime_error("hip: no device (" MI_HIP_TARGET " build)");
  if (index < 0) {
    for (const DeviceInfo& d : devices) {
      if (d.unusable_reason.empty()) {
        info_ = d;
        break;
      }
    }
    if (info_.name.empty()) throw std::runtime_error("hip: no usable device; device 0: " + devices[0].unusable_reason);
  } else {
    if (index >= static_cast<int>(devices.size())) {
      throw std::runtime_error("hip: no device " + std::to_string(index) + " (" + std::to_string(devices.size()) + " found)");
    }
    info_ = devices[static_cast<std::size_t>(index)];
    if (!info_.unusable_reason.empty()) throw std::runtime_error("hip: device " + std::to_string(index) + ": " + info_.unusable_reason);
  }
  make_current();
}

std::uint64_t Device::free_memory() const {
  make_current();
  std::size_t free = 0, total = 0;
  check(hipMemGetInfo(&free, &total), "hipMemGetInfo");
  return free;
}

void Device::make_current() const { check(hipSetDevice(info_.index), "hipSetDevice"); }

void Device::synchronize() const { check(hipDeviceSynchronize(), "hipDeviceSynchronize"); }

namespace {

// Makes `device` current for its lifetime, then restores the calling thread's device.
class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    check(hipGetDevice(&previous_), "hipGetDevice");
    if (device != previous_) check(hipSetDevice(device), "hipSetDevice");
  }
  ~DeviceScope() { (void)hipSetDevice(previous_); }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

 private:
  int previous_ = 0;
};

}  // namespace

DeviceBuffer::DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes == 0) return;
  check(hipGetDevice(&device_), "hipGetDevice");
  check(hipMalloc(&p_, bytes), ("hipMalloc of " + std::to_string(bytes) + " bytes").c_str());
}

DeviceBuffer::~DeviceBuffer() {
  if (p_ != nullptr) (void)hipFree(p_);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept
    : p_(std::exchange(o.p_, nullptr)), bytes_(std::exchange(o.bytes_, 0)), device_(o.device_) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
  std::swap(p_, o.p_);
  std::swap(bytes_, o.bytes_);
  std::swap(device_, o.device_);
  return *this;
}

void DeviceBuffer::upload(const void* src, std::size_t bytes, std::size_t offset) {
  if (offset > bytes_ || bytes > bytes_ - offset) throw std::runtime_error("hip: upload past the end of a device buffer");
  if (bytes == 0) return;
  // hipMemcpy is ordered with neither non-blocking streams nor, from pageable memory, the DMA it starts: wait for every
  // stream of the buffer's device before (a kernel may still read the old contents) and after (the next kernel must see
  // the new ones).
  const DeviceScope scope(device_);
  check(hipDeviceSynchronize(), "hipDeviceSynchronize before an upload");
  check(hipMemcpy(static_cast<char*>(p_) + offset, src, bytes, hipMemcpyHostToDevice), "hipMemcpy to the device");
  check(hipDeviceSynchronize(), "hipDeviceSynchronize after an upload");
}

void DeviceBuffer::download(void* dst, std::size_t bytes, std::size_t offset) const {
  if (offset > bytes_ || bytes > bytes_ - offset) throw std::runtime_error("hip: download past the end of a device buffer");
  if (bytes == 0) return;
  // Every stream's writes on the buffer's device first; the host has the bytes when this returns.
  const DeviceScope scope(device_);
  check(hipDeviceSynchronize(), "hipDeviceSynchronize before a download");
  check(hipMemcpy(dst, static_cast<const char*>(p_) + offset, bytes, hipMemcpyDeviceToHost), "hipMemcpy to the host");
  check(hipDeviceSynchronize(), "hipDeviceSynchronize after a download");
}

void DeviceBuffer::fill(int byte, hipStream_t stream) {
  if (bytes_ == 0) return;
  check(hipMemsetAsync(p_, byte, bytes_, stream), "hipMemsetAsync");
}

HostBuffer::HostBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes == 0) return;
  check(hipHostMalloc(&p_, bytes, hipHostMallocDefault), ("hipHostMalloc of " + std::to_string(bytes) + " bytes").c_str());
}

HostBuffer::~HostBuffer() {
  if (p_ != nullptr) (void)hipHostFree(p_);
}

HostBuffer::HostBuffer(HostBuffer&& o) noexcept : p_(std::exchange(o.p_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}

HostBuffer& HostBuffer::operator=(HostBuffer&& o) noexcept {
  std::swap(p_, o.p_);
  std::swap(bytes_, o.bytes_);
  return *this;
}

Stream::Stream() { check(hipStreamCreateWithFlags(&s_, hipStreamNonBlocking), "hipStreamCreateWithFlags"); }

Stream::~Stream() {
  if (s_ != nullptr) (void)hipStreamDestroy(s_);
}

void Stream::synchronize() const { check(hipStreamSynchronize(s_), "hipStreamSynchronize"); }

Event::Event(bool timing) {
  check(hipEventCreateWithFlags(&e_, timing ? hipEventDefault : hipEventDisableTiming), "hipEventCreateWithFlags");
}

Event::~Event() {
  if (e_ != nullptr) (void)hipEventDestroy(e_);
}

void Event::record(hipStream_t stream) { check(hipEventRecord(e_, stream), "hipEventRecord"); }

void Event::wait(hipStream_t stream) const { check(hipStreamWaitEvent(stream, e_, 0), "hipStreamWaitEvent"); }

void Event::synchronize() const { check(hipEventSynchronize(e_), "hipEventSynchronize"); }

float Event::elapsed_ms(const Event& start) const {
  float ms = 0.0f;
  check(hipEventElapsedTime(&ms, start.e_, e_), "hipEventElapsedTime");
  return ms;
}

void copy_async(void* dst, const void* src, std::size_t bytes, hipStream_t stream) {
  if (bytes == 0) return;
  check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream), "hipMemcpyAsync");
}

}  // namespace mi::hip
