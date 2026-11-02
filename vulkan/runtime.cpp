#include "runtime.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mi::vk {
namespace {

// gemm_coop.comp's shared memory: an f16 A and B tile of 64x32, and an f32 output tile of 64x64.
constexpr std::uint32_t kCoopSharedBytes = 2 * 64 * 32 * 2 + 64 * 64 * 4;
constexpr std::uint32_t kCoopSubgroup = 32;
constexpr std::uint32_t kCoopSubgroupsPerWorkgroup = 4;

void check(VkResult result, const char* what) {
  if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed: VkResult " + std::to_string(result));
}

VkInstance create_instance() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "mossinfer";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  info.pApplicationInfo = &app;
  VkInstance instance = VK_NULL_HANDLE;
  check(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance");
  return instance;
}

std::vector<VkPhysicalDevice> physical_devices(VkInstance instance) {
  std::uint32_t count = 0;
  check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
  std::vector<VkPhysicalDevice> devices(count);
  check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
  return devices;
}

bool has_extension(VkPhysicalDevice physical, const char* name) {
  std::uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
  return std::any_of(extensions.begin(), extensions.end(),
                     [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
}

std::string why_no_coop(VkInstance instance, VkPhysicalDevice physical, const VkPhysicalDeviceProperties& properties,
                        Capabilities& caps) {
  if (properties.apiVersion < VK_API_VERSION_1_3) return "needs Vulkan 1.3";
  VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p13};
  vkGetPhysicalDeviceProperties2(physical, &p2);
  caps.min_subgroup = p13.minSubgroupSize;
  caps.max_subgroup = p13.maxSubgroupSize;
  if (!has_extension(physical, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) return "no VK_KHR_cooperative_matrix";

  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &coop};
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
  VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
  vkGetPhysicalDeviceFeatures2(physical, &features);
  if (coop.cooperativeMatrix == 0) return "the cooperativeMatrix feature is off";
  if (v11.storageBuffer16BitAccess == 0) return "no storageBuffer16BitAccess";
  if (v12.shaderFloat16 == 0) return "no shaderFloat16";
  if (v12.vulkanMemoryModel == 0) return "no vulkanMemoryModel";
  if (v13.subgroupSizeControl == 0 || v13.computeFullSubgroups == 0) return "no subgroup size control";
  if (p13.minSubgroupSize > kCoopSubgroup || p13.maxSubgroupSize < kCoopSubgroup ||
      (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
    return "cannot fix the compute subgroup size at 32 (device range " + std::to_string(p13.minSubgroupSize) + "-" +
           std::to_string(p13.maxSubgroupSize) + ")";
  }
  if (p13.maxComputeWorkgroupSubgroups < kCoopSubgroupsPerWorkgroup) return "needs 4 subgroups per workgroup";
  if (properties.limits.maxComputeSharedMemorySize < kCoopSharedBytes) {
    return "needs " + std::to_string(kCoopSharedBytes) + " bytes of shared memory";
  }

  auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (query == nullptr) return "no cooperative matrix query";
  std::uint32_t count = 0;
  if (query(physical, &count, nullptr) != VK_SUCCESS) return "cooperative matrix query failed";
  std::vector<VkCooperativeMatrixPropertiesKHR> shapes(count);
  for (auto& s : shapes) s.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  if (query(physical, &count, shapes.data()) != VK_SUCCESS) return "cooperative matrix query failed";
  const bool has_shape = std::any_of(shapes.begin(), shapes.end(), [](const VkCooperativeMatrixPropertiesKHR& s) {
    return s.MSize == 16 && s.NSize == 16 && s.KSize == 16 && s.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
           s.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && s.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
           s.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && s.saturatingAccumulation == VK_FALSE &&
           s.scope == VK_SCOPE_SUBGROUP_KHR;
  });
  if (!has_shape) return "no 16x16x16 f16 x f16 -> f32 cooperative matrix in subgroup scope";
  return {};
}

DeviceInfo describe(VkInstance instance, VkPhysicalDevice physical, std::uint32_t index) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical, &properties);
  DeviceInfo info;
  info.index = index;
  info.name = properties.deviceName;
  info.vendor_id = properties.vendorID;
  info.device_id = properties.deviceID;
  info.discrete = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  info.caps.shared_memory_bytes = properties.limits.maxComputeSharedMemorySize;
  if (properties.apiVersion < VK_API_VERSION_1_2) {
    info.unusable_reason = "needs Vulkan 1.2";
    return info;
  }
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
  vkGetPhysicalDeviceFeatures2(physical, &features);
  if (v12.bufferDeviceAddress == 0) info.unusable_reason = "no bufferDeviceAddress";
  if (features.features.shaderInt64 == 0) info.unusable_reason = "no shaderInt64";
  info.caps.coop_reason = why_no_coop(instance, physical, properties, info.caps);
  info.caps.coop = info.caps.coop_reason.empty();
  return info;
}

}  // namespace

std::vector<DeviceInfo> list_devices() {
  VkInstance instance = create_instance();
  std::vector<DeviceInfo> infos;
  std::uint32_t index = 0;
  for (VkPhysicalDevice physical : physical_devices(instance)) infos.push_back(describe(instance, physical, index++));
  vkDestroyInstance(instance, nullptr);
  return infos;
}

Device::Device(int index) {
  instance_ = create_instance();
  const std::vector<VkPhysicalDevice> devices = physical_devices(instance_);
  bool chosen = false;
  for (std::uint32_t i = 0; i < devices.size(); ++i) {
    DeviceInfo candidate = describe(instance_, devices[i], i);
    if (index >= 0) {
      if (static_cast<std::uint32_t>(index) != i) continue;
      if (!candidate.unusable_reason.empty()) {
        throw std::runtime_error("Vulkan device " + std::to_string(i) + " (" + candidate.name + ") cannot be used: " +
                                 candidate.unusable_reason);
      }
    } else if (!candidate.unusable_reason.empty() || (chosen && (info_.discrete || !candidate.discrete))) {
      continue;
    }
    physical_ = devices[i];
    info_ = std::move(candidate);
    chosen = true;
  }
  if (!chosen) throw std::runtime_error("no usable Vulkan device");
  vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);

  std::uint32_t families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(families);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_, &families, queue_families.data());
  std::uint32_t family = families;
  for (std::uint32_t i = 0; i < families && family == families; ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) family = i;
  }
  if (family == families) throw std::runtime_error("no compute queue on " + info_.name);

  // Only what the shaders declare, so a shader that needs more fails at pipeline creation instead of working by luck.
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  coop.cooperativeMatrix = VK_TRUE;
  VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &coop};
  v13.subgroupSizeControl = VK_TRUE;
  v13.computeFullSubgroups = VK_TRUE;
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  v12.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.features.shaderInt64 = VK_TRUE;
  std::vector<const char*> extensions;
  if (info_.caps.coop) {
    v12.pNext = &v13;
    v12.shaderFloat16 = VK_TRUE;
    v12.vulkanMemoryModel = VK_TRUE;
    v11.storageBuffer16BitAccess = VK_TRUE;
    features.pNext = &v11;
    extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  } else {
    features.pNext = &v12;  // a Vulkan 1.2 device need not know the 1.3 structure
  }

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.pNext = &features;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
  device_info.ppEnabledExtensionNames = extensions.data();
  check(vkCreateDevice(physical_, &device_info, nullptr, &device_), "vkCreateDevice");
  vkGetDeviceQueue(device_, family, 0, &queue_);

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.queueFamilyIndex = family;
  check(vkCreateCommandPool(device_, &pool_info, nullptr, &pool_), "vkCreateCommandPool");
}

Device::~Device() {
  if (pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, pool_, nullptr);
  if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
  if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
}

std::uint32_t Device::find_memory_type(std::uint32_t type_bits, VkMemoryPropertyFlags flags) const {
  for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) != 0 && (memory_.memoryTypes[i].propertyFlags & flags) == flags) return i;
  }
  throw std::runtime_error("no Vulkan memory type with flags " + std::to_string(flags));
}

void Device::run(const std::function<void(VkCommandBuffer)>& record) {
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = pool_;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  VkCommandBuffer cb = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(device_, &allocate, &cb), "vkAllocateCommandBuffers");
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check(vkBeginCommandBuffer(cb, &begin), "vkBeginCommandBuffer");
  record(cb);
  check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cb;
  const VkResult submitted = vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE);
  const VkResult waited = submitted == VK_SUCCESS ? vkQueueWaitIdle(queue_) : submitted;
  vkFreeCommandBuffers(device_, pool_, 1, &cb);
  check(submitted, "vkQueueSubmit");
  check(waited, "vkQueueWaitIdle");
}

Buffer::Buffer(Device& device, VkDeviceSize bytes, Kind kind) : device_(device), bytes_(bytes) {
  const bool on_device = kind == Kind::kDevice;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = bytes;
  info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (on_device) info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(device_.handle(), &info, nullptr, &buffer_), "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device_.handle(), buffer_, &requirements);
  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  if (on_device) allocate.pNext = &flags;
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = device_.find_memory_type(
      requirements.memoryTypeBits, on_device ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                             : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  const VkResult allocated = vkAllocateMemory(device_.handle(), &allocate, nullptr, &memory_);
  if (allocated != VK_SUCCESS) {
    vkDestroyBuffer(device_.handle(), buffer_, nullptr);
    check(allocated, "vkAllocateMemory");
  }
  check(vkBindBufferMemory(device_.handle(), buffer_, memory_, 0), "vkBindBufferMemory");
  if (on_device) {
    VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address.buffer = buffer_;
    address_ = vkGetBufferDeviceAddress(device_.handle(), &address);
  } else {
    check(vkMapMemory(device_.handle(), memory_, 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory");
  }
}

Buffer::~Buffer() {
  vkDestroyBuffer(device_.handle(), buffer_, nullptr);
  vkFreeMemory(device_.handle(), memory_, nullptr);
}

void Buffer::upload(const void* data, VkDeviceSize bytes) {
  Buffer staging(device_, bytes, Kind::kStaging);
  std::memcpy(staging.mapped(), data, static_cast<std::size_t>(bytes));
  device_.run([&](VkCommandBuffer cb) {
    const VkBufferCopy copy{0, 0, bytes};
    vkCmdCopyBuffer(cb, staging.handle(), buffer_, 1, &copy);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
  });
}

void Buffer::download(void* data, VkDeviceSize bytes) {
  Buffer staging(device_, bytes, Kind::kStaging);
  device_.run([&](VkCommandBuffer cb) {
    const VkBufferCopy copy{0, 0, bytes};
    vkCmdCopyBuffer(cb, buffer_, staging.handle(), 1, &copy);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);
  });
  std::memcpy(data, staging.mapped(), static_cast<std::size_t>(bytes));
}

Pipeline::Pipeline(Device& device, const std::uint32_t* spirv, std::size_t spirv_bytes, std::uint32_t push_bytes,
                   std::uint32_t required_subgroup_size, const std::vector<std::uint32_t>& specialization)
    : device_(device), push_bytes_(push_bytes) {
  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = spirv_bytes;
  module_info.pCode = spirv;
  check(vkCreateShaderModule(device_.handle(), &module_info, nullptr, &module_), "vkCreateShaderModule");

  VkPushConstantRange range{};
  range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  range.size = push_bytes;
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &range;
  check(vkCreatePipelineLayout(device_.handle(), &layout_info, nullptr, &layout_), "vkCreatePipelineLayout");

  std::vector<VkSpecializationMapEntry> entries;
  for (std::uint32_t i = 0; i < specialization.size(); ++i) {
    entries.push_back({i, i * static_cast<std::uint32_t>(sizeof(std::uint32_t)), sizeof(std::uint32_t)});
  }
  VkSpecializationInfo specialization_info{};
  specialization_info.mapEntryCount = static_cast<std::uint32_t>(entries.size());
  specialization_info.pMapEntries = entries.data();
  specialization_info.dataSize = specialization.size() * sizeof(std::uint32_t);
  specialization_info.pData = specialization.data();

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
  subgroup.requiredSubgroupSize = required_subgroup_size;
  VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.module = module_;
  info.stage.pName = "main";
  if (!specialization.empty()) info.stage.pSpecializationInfo = &specialization_info;
  if (required_subgroup_size != 0) {
    info.stage.pNext = &subgroup;
    info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
  }
  info.layout = layout_;
  check(vkCreateComputePipelines(device_.handle(), VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_),
        "vkCreateComputePipelines");
}

Pipeline::~Pipeline() {
  vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
  vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
  vkDestroyShaderModule(device_.handle(), module_, nullptr);
}

void Pipeline::dispatch(VkCommandBuffer cb, const void* push, std::uint32_t x, std::uint32_t y) const {
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes_, push);
  vkCmdDispatch(cb, x, y, 1);
}

void compute_barrier(VkCommandBuffer cb) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr,
                       0, nullptr);
}

}  // namespace mi::vk
