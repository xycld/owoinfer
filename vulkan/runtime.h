// The Vulkan device runtime: one device, one compute queue, buffers with device addresses, and pipelines from SPIR-V
// embedded in the binary. Failures throw std::runtime_error.
#pragma once

#include "vk_api.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mi::vk {

// What a device offers beyond the base requirements (Vulkan 1.2, buffer device addresses, 64-bit integers).
struct Capabilities {
  // The cooperative-matrix GEMM can run: 16x16x16 f16 x f16 -> f32 in subgroup scope, a compute subgroup size fixable at
  // 32 with 8 per workgroup, and its shared memory. Otherwise `coop_reason` says why not.
  bool coop = false;
  std::string coop_reason;
  // The DiT block's cooperative-matrix kernels can run: the same subgroup and memory-model requirements, with
  // VK_KHR_shader_bfloat16 fragments and a 16x16x16 bf16 x bf16 -> f32 shape. Otherwise `coop_bf16_reason` says why not.
  bool coop_bf16 = false;
  std::string coop_bf16_reason;
  // The same kernels on f16 fragments (16x16x16 f16 x f16 -> f32), for devices without bf16 ones: the bf16 operands
  // convert to f16 as they are staged. Otherwise `coop_f16_reason` says why not.
  bool coop_f16 = false;
  std::string coop_f16_reason;
  std::uint32_t shared_memory_bytes = 0;
  std::uint32_t min_subgroup = 0, max_subgroup = 0;
  // The largest single buffer and allocation the device takes (maxBufferSize on Vulkan 1.3, maxMemoryAllocationSize).
  std::uint64_t max_buffer_bytes = 0, max_allocation_bytes = 0;
};

struct DeviceInfo {
  std::uint32_t index = 0;
  std::string name;
  std::uint32_t vendor_id = 0, device_id = 0;
  bool discrete = false;
  std::string unusable_reason;  // empty when the runtime can use the device
  Capabilities caps;
};

// Enumerates every physical device, usable or not. Does not create a logical device.
std::vector<DeviceInfo> list_devices();

class Buffer;

class Device {
 public:
  // index < 0: the first usable device, preferring a discrete one.
  explicit Device(int index = -1);
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  const DeviceInfo& info() const { return info_; }

  // Device-local memory this process can still allocate: the driver's budget minus its usage where the device has
  // VK_EXT_memory_budget, else the size of the largest device-local heap.
  std::uint64_t free_device_memory() const;

  // Records into one command buffer, submits it and waits.
  void run(const std::function<void(VkCommandBuffer)>& record);

  // Asynchronous buffer copies on a transfer-only queue (a DMA engine) where the device has one, else on the compute
  // queue. Buffers they read or write must be created with `transfer` set. copy_async returns at once with a ticket;
  // gpu_wait(ticket) makes the next run() wait on the device for that copy, which makes its writes visible to the run
  // (once per ticket); host_wait(ticket) blocks until the copy has completed, after which its source may be reused. The
  // two may come in either order; a ticket is released once both happened (or with the device).
  bool has_transfer_queue() const { return transfer_queue_ != VK_NULL_HANDLE; }
  std::uint64_t copy_async(const Buffer& src, VkDeviceSize src_offset, const Buffer& dst, VkDeviceSize dst_offset,
                           VkDeviceSize bytes);
  void gpu_wait(std::uint64_t ticket);
  void host_wait(std::uint64_t ticket);
  std::uint32_t queue_family() const { return family_; }
  std::uint32_t transfer_family() const { return transfer_family_; }

  VkDevice handle() const { return device_; }
  bool pipeline_stats() const { return pipeline_stats_; }
  std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags flags) const;

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory_{};
  DeviceInfo info_;
  bool pipeline_stats_ = false;
  std::uint32_t family_ = 0, transfer_family_ = 0;
  VkQueue transfer_queue_ = VK_NULL_HANDLE;
  VkCommandPool transfer_pool_ = VK_NULL_HANDLE;
  struct Copy {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore done = VK_NULL_HANDLE;
    bool host_done = false;  // host_wait returned
    bool run_done = false;   // a run() waited for it on the device
  };
  void release_if_done(std::uint64_t ticket);
  std::map<std::uint64_t, Copy> copies_;
  std::uint64_t next_ticket_ = 1;
  std::vector<std::uint64_t> run_waits_;  // tickets the next run() waits for
};

// A device-local storage buffer with a device address, or a host-visible staging buffer.
class Buffer {
 public:
  enum class Kind { kDevice, kStaging };
  // `transfer`: also used by Device::copy_async (shared with the transfer queue's family).
  Buffer(Device& device, VkDeviceSize bytes, Kind kind = Kind::kDevice, bool transfer = false);
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  VkBuffer handle() const { return buffer_; }
  VkDeviceAddress address() const { return address_; }
  VkDeviceSize bytes() const { return bytes_; }
  void* mapped() const { return mapped_; }

  // Copies through a staging buffer and waits; `offset` is in bytes from the start of this buffer.
  void upload(const void* data, VkDeviceSize bytes, VkDeviceSize offset = 0);
  void download(void* data, VkDeviceSize bytes, VkDeviceSize offset = 0);

 private:
  Device& device_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkDeviceAddress address_ = 0;
  VkDeviceSize bytes_ = 0;
  void* mapped_ = nullptr;
};

// A compute pipeline with push constants only. required_subgroup_size != 0 also requires full subgroups.
class Pipeline {
 public:
  Pipeline(Device& device, const std::uint32_t* spirv, std::size_t spirv_bytes, std::uint32_t push_bytes,
           std::uint32_t required_subgroup_size = 0, const std::vector<std::uint32_t>& specialization = {});
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  void dispatch(VkCommandBuffer cb, const void* push, std::uint32_t x, std::uint32_t y) const;

 private:
  void print_statistics(std::size_t spirv_bytes) const;
  Device& device_;
  VkShaderModule module_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  std::uint32_t push_bytes_ = 0;
};

// Makes earlier compute writes visible to later compute and transfer work. A queue wait only reaches the host.
void compute_barrier(VkCommandBuffer cb);

// Records a device copy of `bytes` from src at src_offset to dst at dst_offset, after earlier compute and transfer writes
// and before later compute work.
void record_copy(VkCommandBuffer cb, const Buffer& src, VkDeviceSize src_offset, const Buffer& dst, VkDeviceSize dst_offset,
                 VkDeviceSize bytes);

}  // namespace mi::vk
