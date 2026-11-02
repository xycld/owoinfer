// The Vulkan device runtime: one device, one compute queue, buffers with device addresses, and pipelines from SPIR-V
// embedded in the binary. Failures throw std::runtime_error.
#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <functional>
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
  std::uint32_t shared_memory_bytes = 0;
  std::uint32_t min_subgroup = 0, max_subgroup = 0;
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

  // Records into one command buffer, submits it and waits.
  void run(const std::function<void(VkCommandBuffer)>& record);

  VkDevice handle() const { return device_; }
  std::uint32_t find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags flags) const;

 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory_{};
  DeviceInfo info_;
};

// A device-local storage buffer with a device address, or a host-visible staging buffer.
class Buffer {
 public:
  enum class Kind { kDevice, kStaging };
  Buffer(Device& device, VkDeviceSize bytes, Kind kind = Kind::kDevice);
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
