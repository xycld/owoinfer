#include "runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace mi::vk {
namespace {

// MI_VK_PIPELINE_STATS=1 prints the driver's statistics of every pipeline (registers, spills, ...), where the device has
// VK_KHR_pipeline_executable_properties. A debugging aid; off by default.
bool pipeline_stats_requested() {
  const char* v = std::getenv("MI_VK_PIPELINE_STATS");
  return v != nullptr && v[0] == '1';
}

// gemm_coop.comp's shared memory: an f16 A and B tile of 64x32, and an f32 output tile of 64x64.
constexpr std::uint32_t kCoopSharedBytes = 2 * 64 * 32 * 2 + 64 * 64 * 4;
constexpr std::uint32_t kCoopSubgroup = 32;
constexpr std::uint32_t kCoopSubgroupsPerWorkgroup = 8;
// The bf16 DiT kernels' largest shared memory: gemm_bf16_coop.comp's two stages of 128x32 A and B tiles, rows padded to 80 B.
constexpr std::uint32_t kCoopBf16SharedBytes = 4 * 128 * 80;

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

// What both cooperative-matrix paths need: Vulkan 1.3, the extension and feature, 16-bit storage, the memory model, and a
// compute subgroup size fixable at 32 with 8 per workgroup. Empty when the device has all of it.
std::string why_no_coop_base(VkPhysicalDevice physical, const VkPhysicalDeviceProperties& properties, Capabilities& caps) {
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
  if (p13.maxComputeWorkgroupSubgroups < kCoopSubgroupsPerWorkgroup) return "needs 8 subgroups per workgroup";
  return {};
}

std::vector<VkCooperativeMatrixPropertiesKHR> coop_shapes(VkInstance instance, VkPhysicalDevice physical) {
  auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (query == nullptr) return {};
  std::uint32_t count = 0;
  if (query(physical, &count, nullptr) != VK_SUCCESS) return {};
  std::vector<VkCooperativeMatrixPropertiesKHR> shapes(count);
  for (auto& s : shapes) s.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  if (query(physical, &count, shapes.data()) != VK_SUCCESS) return {};
  return shapes;
}

bool has_shape(const std::vector<VkCooperativeMatrixPropertiesKHR>& shapes, VkComponentTypeKHR input) {
  return std::any_of(shapes.begin(), shapes.end(), [&](const VkCooperativeMatrixPropertiesKHR& s) {
    return s.MSize == 16 && s.NSize == 16 && s.KSize == 16 && s.AType == input && s.BType == input &&
           s.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && s.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
           s.saturatingAccumulation == VK_FALSE && s.scope == VK_SCOPE_SUBGROUP_KHR;
  });
}

void describe_coop(VkInstance instance, VkPhysicalDevice physical, const VkPhysicalDeviceProperties& properties,
                   Capabilities& caps) {
  const std::string base = why_no_coop_base(physical, properties, caps);
  const std::vector<VkCooperativeMatrixPropertiesKHR> shapes =
      base.empty() ? coop_shapes(instance, physical) : std::vector<VkCooperativeMatrixPropertiesKHR>{};
  const std::uint32_t shared = properties.limits.maxComputeSharedMemorySize;

  caps.coop_reason = base;
  if (caps.coop_reason.empty() && shared < kCoopSharedBytes) {
    caps.coop_reason = "needs " + std::to_string(kCoopSharedBytes) + " bytes of shared memory";
  }
  if (caps.coop_reason.empty() && !has_shape(shapes, VK_COMPONENT_TYPE_FLOAT16_KHR)) {
    caps.coop_reason = "no 16x16x16 f16 x f16 -> f32 cooperative matrix in subgroup scope";
  }
  caps.coop = caps.coop_reason.empty();

  caps.coop_bf16_reason = base;
  if (caps.coop_bf16_reason.empty() && !has_extension(physical, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)) {
    caps.coop_bf16_reason = "no VK_KHR_shader_bfloat16";
  }
  if (caps.coop_bf16_reason.empty()) {
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &bf16};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    if (bf16.shaderBFloat16Type == 0 || bf16.shaderBFloat16CooperativeMatrix == 0) {
      caps.coop_bf16_reason = "no bf16 cooperative matrices";
    }
  }
  if (caps.coop_bf16_reason.empty() && shared < kCoopBf16SharedBytes) {
    caps.coop_bf16_reason = "needs " + std::to_string(kCoopBf16SharedBytes) + " bytes of shared memory";
  }
  if (caps.coop_bf16_reason.empty() && !has_shape(shapes, VK_COMPONENT_TYPE_BFLOAT16_KHR)) {
    caps.coop_bf16_reason = "no 16x16x16 bf16 x bf16 -> f32 cooperative matrix in subgroup scope";
  }
  caps.coop_bf16 = caps.coop_bf16_reason.empty();
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
  describe_coop(instance, physical, properties, info.caps);
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
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR};
  bf16.shaderBFloat16Type = VK_TRUE;
  bf16.shaderBFloat16CooperativeMatrix = VK_TRUE;
  std::vector<const char*> extensions;
  if (info_.caps.coop_bf16) {
    coop.pNext = &bf16;
    extensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
  }
  if (info_.caps.coop || info_.caps.coop_bf16) {
    v12.pNext = &v13;
    v12.shaderFloat16 = VK_TRUE;
    v12.vulkanMemoryModel = VK_TRUE;
    v11.storageBuffer16BitAccess = VK_TRUE;
    features.pNext = &v11;
    extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  } else {
    features.pNext = &v12;  // a Vulkan 1.2 device need not know the 1.3 structure
  }

  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executable{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  if (pipeline_stats_requested() && has_extension(physical_, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
    executable.pipelineExecutableInfo = VK_TRUE;
    executable.pNext = features.pNext;
    features.pNext = &executable;
    extensions.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    pipeline_stats_ = true;
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

std::uint64_t Device::free_device_memory() const {
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  const bool has_budget = has_extension(physical_, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  if (has_budget) properties.pNext = &budget;
  vkGetPhysicalDeviceMemoryProperties2(physical_, &properties);
  std::uint64_t best = 0;
  for (std::uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
    const VkMemoryHeap& heap = properties.memoryProperties.memoryHeaps[i];
    if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
    std::uint64_t free = heap.size;
    if (has_budget) free = budget.heapBudget[i] > budget.heapUsage[i] ? budget.heapBudget[i] - budget.heapUsage[i] : 0;
    best = std::max(best, free);
  }
  return best;
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

void Buffer::upload(const void* data, VkDeviceSize bytes, VkDeviceSize offset) {
  if (offset > bytes_ || bytes > bytes_ - offset) throw std::runtime_error("Buffer::upload past the end of the buffer");
  Buffer staging(device_, bytes, Kind::kStaging);
  std::memcpy(staging.mapped(), data, static_cast<std::size_t>(bytes));
  device_.run([&](VkCommandBuffer cb) {
    const VkBufferCopy copy{0, offset, bytes};
    vkCmdCopyBuffer(cb, staging.handle(), buffer_, 1, &copy);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
  });
}

void Buffer::download(void* data, VkDeviceSize bytes, VkDeviceSize offset) {
  if (offset > bytes_ || bytes > bytes_ - offset) throw std::runtime_error("Buffer::download past the end of the buffer");
  Buffer staging(device_, bytes, Kind::kStaging);
  device_.run([&](VkCommandBuffer cb) {
    const VkBufferCopy copy{offset, 0, bytes};
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
  if (device_.pipeline_stats()) info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  check(vkCreateComputePipelines(device_.handle(), VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_),
        "vkCreateComputePipelines");
  if (device_.pipeline_stats()) print_statistics(spirv_bytes);
}

void Pipeline::print_statistics(std::size_t spirv_bytes) const {
  auto properties = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
      vkGetDeviceProcAddr(device_.handle(), "vkGetPipelineExecutablePropertiesKHR"));
  auto statistics = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
      vkGetDeviceProcAddr(device_.handle(), "vkGetPipelineExecutableStatisticsKHR"));
  if (properties == nullptr || statistics == nullptr) return;
  VkPipelineInfoKHR pipeline_info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
  pipeline_info.pipeline = pipeline_;
  std::uint32_t count = 0;
  properties(device_.handle(), &pipeline_info, &count, nullptr);
  std::vector<VkPipelineExecutablePropertiesKHR> executables(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
  properties(device_.handle(), &pipeline_info, &count, executables.data());
  for (std::uint32_t e = 0; e < count; ++e) {
    VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    executable.pipeline = pipeline_;
    executable.executableIndex = e;
    std::uint32_t n = 0;
    statistics(device_.handle(), &executable, &n, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> stats(n, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    statistics(device_.handle(), &executable, &n, stats.data());
    std::fprintf(stderr, "pipeline (SPIR-V %zu bytes) %s:", spirv_bytes, executables[e].name);
    for (const auto& s : stats) {
      if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) {
        std::fprintf(stderr, " %s=%llu", s.name, static_cast<unsigned long long>(s.value.u64));
      } else if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) {
        std::fprintf(stderr, " %s=%lld", s.name, static_cast<long long>(s.value.i64));
      }
    }
    std::fprintf(stderr, "\n");
  }
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

void record_copy(VkCommandBuffer cb, const Buffer& src, VkDeviceSize src_offset, const Buffer& dst, VkDeviceSize dst_offset,
                 VkDeviceSize bytes) {
  if (src_offset > src.bytes() || bytes > src.bytes() - src_offset || dst_offset > dst.bytes() ||
      bytes > dst.bytes() - dst_offset) {
    throw std::runtime_error("record_copy past the end of a buffer");
  }
  VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
  const VkBufferCopy copy{src_offset, dst_offset, bytes};
  vkCmdCopyBuffer(cb, src.handle(), dst.handle(), 1, &copy);
  VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after, 0, nullptr,
                       0, nullptr);
}

}  // namespace mi::vk
