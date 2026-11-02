// Vulkan cooperative-matrix GEMM bring-up: one f16 GEMM kernel, checked against a CPU reference and timed at the
// engine's matrix shapes. This is the first artefact of the Vulkan backend work in docs/vulkan-backend-*.md; it is a
// standalone tool (no engine code is linked), so it can run before any backend integration exists.
//
// Kernel contract: C[M,N] = A[M,K] * B[K,N] with A row-major (stride K) and B column-major (stride K), i.e. B[k,n] lives
// at n*K + k, which is exactly the engine's [N,K] weight layout. That lets the GEMM consume weights as stored, with no
// transposes. N must be a multiple of 64 and K of 16. M is free: the kernel always runs whole 64-row tiles, so A and C
// must be allocated with M rounded up to 64 rows. Row m of C depends only on row m of A, so the padding rows never reach
// the first M rows (the check fills them with NaN to show it).
//
// Usage: vulkan_gemm [--spv <gemm_f16.spv>] [--device N] [--bench M N K iters]... [--no-check]
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#ifndef DEFAULT_SPV_PATH
#define DEFAULT_SPV_PATH "gemm_f16.spv"
#endif

namespace {

constexpr uint32_t kTile = 64;  // BM == BN, one workgroup per tile
constexpr uint32_t kSubgroupSize = 32;  // gemm_f16.comp divides its 128 invocations into four subgroups of this size

[[noreturn]] void Fail(const char* what, VkResult r) {
  std::printf("%s failed: VkResult %d\n", what, static_cast<int>(r));
  std::exit(1);
}

void Check(VkResult r, const char* what) {
  if (r != VK_SUCCESS) Fail(what, r);
}

float HalfToFloat(uint16_t h) {
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t mant = h & 0x3ff;
  float f = 0.0f;
  if (exp == 0) {
    f = std::ldexp(static_cast<float>(mant), -24);  // zero or subnormal: mant * 2^-24, exact in fp32
  } else {
    const uint32_t bits = exp == 31 ? 0x7f800000u | (mant << 13) : ((exp + 127 - 15) << 23) | (mant << 13);
    std::memcpy(&f, &bits, 4);
  }
  return (h & 0x8000u) != 0 ? -f : f;
}

// Rounds to nearest, ties to even, like the hardware conversion.
uint16_t FloatToHalf(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const auto sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const uint32_t magnitude = x & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u);  // NaN
  const int32_t exp = static_cast<int32_t>(magnitude >> 23) - 127 + 15;
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);  // overflow and infinity
  if (exp < -10) return sign;                                   // below half of the smallest subnormal
  const uint32_t mant = magnitude & 0x7fffffu;
  const uint32_t significand = exp > 0 ? mant : mant | 0x800000u;
  const int shift = exp > 0 ? 13 : 14 - exp;
  uint32_t rounded = significand >> shift;
  const uint32_t rest = significand & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1);
  if (rest > half || (rest == half && (rounded & 1u) != 0)) ++rounded;
  // An addition, not an OR: rounding can carry out of the mantissa into the exponent.
  return static_cast<uint16_t>(sign | ((exp > 0 ? static_cast<uint32_t>(exp) << 10 : 0u) + rounded));
}

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VkDeviceAddress address = 0;
  void* mapped = nullptr;
};

// Everything the tool needs from one device. Not a general runtime: the backend's runtime (buffers, arenas, queues)
// replaces this when the engine integration starts (docs/vulkan-backend-design.md).
struct Device {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory{};
  VkPhysicalDeviceProperties properties{};
};

uint32_t FindMemoryType(const Device& d, uint32_t type_bits, VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < d.memory.memoryTypeCount; ++i) {
    if ((type_bits & (1u << i)) != 0 && (d.memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
  }
  std::printf("no memory type with flags 0x%x\n", static_cast<unsigned>(flags));
  std::exit(1);
}

Buffer CreateBuffer(Device& d, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags,
                    bool device_address) {
  Buffer b;
  b.size = size;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = size;
  info.usage = usage;
  if (device_address) info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  Check(vkCreateBuffer(d.device, &info, nullptr, &b.buffer), "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(d.device, b.buffer, &requirements);
  VkMemoryAllocateFlagsInfo allocate_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  allocate_flags.flags = device_address ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
  VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocate.pNext = &allocate_flags;
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = FindMemoryType(d, requirements.memoryTypeBits, flags);
  Check(vkAllocateMemory(d.device, &allocate, nullptr, &b.memory), "vkAllocateMemory");
  Check(vkBindBufferMemory(d.device, b.buffer, b.memory, 0), "vkBindBufferMemory");
  if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
    Check(vkMapMemory(d.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped), "vkMapMemory");
  }
  if (device_address) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = b.buffer;
    b.address = vkGetBufferDeviceAddress(d.device, &info);
  }
  return b;
}

void DestroyBuffer(Device& d, Buffer& b) {
  if (b.buffer != VK_NULL_HANDLE) vkDestroyBuffer(d.device, b.buffer, nullptr);
  if (b.memory != VK_NULL_HANDLE) vkFreeMemory(d.device, b.memory, nullptr);
  b = Buffer{};
}

void Submit(Device& d, const std::function<void(VkCommandBuffer)>& record) {
  VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocate.commandPool = d.pool;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  Check(vkAllocateCommandBuffers(d.device, &allocate, &command_buffer), "vkAllocateCommandBuffers");
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  Check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer");
  record(command_buffer);
  Check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer;
  Check(vkQueueSubmit(d.queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
  Check(vkQueueWaitIdle(d.queue), "vkQueueWaitIdle");
  vkFreeCommandBuffers(d.device, d.pool, 1, &command_buffer);
}

// A global memory barrier. vkQueueWaitIdle only makes writes available to the host: work in a later submit still needs
// one of these to see them.
void Barrier(VkCommandBuffer cb, VkPipelineStageFlags src_stage, VkAccessFlags src_access,
             VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = src_access;
  barrier.dstAccessMask = dst_access;
  vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Upload(Device& d, Buffer& dst, const void* data, VkDeviceSize size) {
  Buffer staging = CreateBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
  std::memcpy(staging.mapped, data, static_cast<size_t>(size));
  Submit(d, [&](VkCommandBuffer cb) {
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cb, staging.buffer, dst.buffer, 1, &copy);
    Barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);
  });
  DestroyBuffer(d, staging);
}

void Download(Device& d, Buffer& src, void* data, VkDeviceSize size) {
  Buffer staging = CreateBuffer(d, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
  Submit(d, [&](VkCommandBuffer cb) {
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cb, src.buffer, staging.buffer, 1, &copy);
    Barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
  });
  std::memcpy(data, staging.mapped, static_cast<size_t>(size));
  DestroyBuffer(d, staging);
}

std::vector<uint32_t> ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::printf("cannot open %s\n", path.c_str());
    std::exit(1);
  }
  const auto size = static_cast<size_t>(file.tellg());
  if (size == 0 || size % 4 != 0) {
    std::printf("%s is not a SPIR-V blob\n", path.c_str());
    std::exit(1);
  }
  std::vector<uint32_t> words(size / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(size));
  return words;
}

const char* ComponentName(int type) {
  switch (type) {
    case 0: return "f16";
    case 1: return "f32";
    case 2: return "f64";
    case 3: return "s8";
    case 4: return "s16";
    case 5: return "s32";
    case 6: return "s64";
    case 7: return "u8";
    case 8: return "u16";
    case 9: return "u32";
    case 10: return "u64";
    case 1000141000: return "bf16";
    default: return "?";
  }
}
std::vector<VkCooperativeMatrixPropertiesKHR> CooperativeMatrixShapes(VkInstance instance, VkPhysicalDevice physical) {
  auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
      vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (query == nullptr) return {};
  uint32_t count = 0;
  if (query(physical, &count, nullptr) != VK_SUCCESS || count == 0) return {};
  std::vector<VkCooperativeMatrixPropertiesKHR> shapes(count);
  for (auto& s : shapes) s.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  if (query(physical, &count, shapes.data()) != VK_SUCCESS) return {};
  shapes.resize(count);
  return shapes;
}

void PrintCooperativeMatrixShapes(Device& d) {
  const auto shapes = CooperativeMatrixShapes(d.instance, d.physical);
  std::printf("cooperative matrix shapes (%zu):\n", shapes.size());
  for (const auto& s : shapes) {
    std::printf("  %ux%ux%u A=%s B=%s C=%s R=%s sat=%u\n", s.MSize, s.NSize, s.KSize,
                ComponentName(static_cast<int>(s.AType)), ComponentName(static_cast<int>(s.BType)),
                ComponentName(static_cast<int>(s.CType)), ComponentName(static_cast<int>(s.ResultType)),
                s.saturatingAccumulation);
  }
}

// Why the kernel cannot run on this device; empty when it can.
std::string WhyUnsupported(VkInstance instance, VkPhysicalDevice physical) {
  VkPhysicalDeviceVulkan13Properties properties13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  vkGetPhysicalDeviceProperties(physical, &properties.properties);
  if (properties.properties.apiVersion < VK_API_VERSION_1_3) return "needs Vulkan 1.3";
  properties.pNext = &properties13;
  vkGetPhysicalDeviceProperties2(physical, &properties);

  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
  const bool has_extension = std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& e) {
    return std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;
  });
  if (!has_extension) return "no VK_KHR_cooperative_matrix";

  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &v11;
  v11.pNext = &v12;
  v12.pNext = &v13;
  v13.pNext = &coop;
  vkGetPhysicalDeviceFeatures2(physical, &features);
  if (coop.cooperativeMatrix == 0) return "the cooperativeMatrix feature is off";
  if (features.features.shaderInt64 == 0) return "no shaderInt64";
  if (v11.storageBuffer16BitAccess == 0) return "no storageBuffer16BitAccess";
  if (v12.shaderFloat16 == 0) return "no shaderFloat16";
  if (v12.bufferDeviceAddress == 0) return "no bufferDeviceAddress";
  if (v12.vulkanMemoryModel == 0) return "no vulkanMemoryModel";
  if (v13.subgroupSizeControl == 0 || v13.computeFullSubgroups == 0) return "no subgroup size control";

  if (properties13.minSubgroupSize > kSubgroupSize || properties13.maxSubgroupSize < kSubgroupSize ||
      (properties13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
    return "cannot fix the compute subgroup size at " + std::to_string(kSubgroupSize) + " (device range " +
           std::to_string(properties13.minSubgroupSize) + "-" + std::to_string(properties13.maxSubgroupSize) + ")";
  }
  if (properties13.maxComputeWorkgroupSubgroups < 4) return "needs 4 subgroups per workgroup";

  const auto shapes = CooperativeMatrixShapes(instance, physical);
  const bool has_shape = std::any_of(shapes.begin(), shapes.end(), [](const VkCooperativeMatrixPropertiesKHR& s) {
    return s.MSize == 16 && s.NSize == 16 && s.KSize == 16 && s.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
           s.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && s.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
           s.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && s.saturatingAccumulation == VK_FALSE &&
           s.scope == VK_SCOPE_SUBGROUP_KHR;
  });
  if (!has_shape) return "no 16x16x16 f16 x f16 -> f32 cooperative matrix in subgroup scope";
  return {};
}

// device_index < 0: the first device that can run the kernel.
Device OpenDevice(int device_index) {
  Device d;
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "mossinfer-vulkan-gemm";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instance_info.pApplicationInfo = &app;
  Check(vkCreateInstance(&instance_info, nullptr, &d.instance), "vkCreateInstance");

  uint32_t count = 0;
  Check(vkEnumeratePhysicalDevices(d.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
  std::vector<VkPhysicalDevice> devices(count);
  Check(vkEnumeratePhysicalDevices(d.instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
  bool chosen = false;
  for (uint32_t i = 0; i < count; ++i) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(devices[i], &properties);
    const std::string why = WhyUnsupported(d.instance, devices[i]);
    std::printf("device %u: %s: %s\n", i, properties.deviceName, why.empty() ? "usable" : why.c_str());
    const bool wanted = device_index < 0 ? why.empty() && !chosen : static_cast<uint32_t>(device_index) == i;
    if (wanted) {
      if (!why.empty()) {
        std::printf("device %u cannot run this kernel\n", i);
        std::exit(1);
      }
      d.physical = devices[i];
      chosen = true;
    }
  }
  if (!chosen) {
    std::printf("no usable Vulkan device\n");
    std::exit(1);
  }
  vkGetPhysicalDeviceProperties(d.physical, &d.properties);
  vkGetPhysicalDeviceMemoryProperties(d.physical, &d.memory);

  uint32_t families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &families, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(families);
  vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &families, queue_families.data());
  bool found = false;
  for (uint32_t i = 0; i < families; ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
      d.queue_family = i;
      found = true;
      break;
    }
  }
  if (!found) {
    std::printf("no compute queue\n");
    std::exit(1);
  }

  // Exactly what gemm_f16.spv declares (spirv-dis lists its capabilities), so a missing one fails at vkCreateDevice.
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  coop.cooperativeMatrix = VK_TRUE;
  VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  v13.pNext = &coop;
  v13.subgroupSizeControl = VK_TRUE;
  v13.computeFullSubgroups = VK_TRUE;
  VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  v12.pNext = &v13;
  v12.shaderFloat16 = VK_TRUE;
  v12.bufferDeviceAddress = VK_TRUE;
  v12.vulkanMemoryModel = VK_TRUE;
  VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  v11.pNext = &v12;
  v11.storageBuffer16BitAccess = VK_TRUE;
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &v11;
  features.features.shaderInt64 = VK_TRUE;

  std::printf("using: %s (Vulkan %u.%u.%u)\n", d.properties.deviceName, VK_API_VERSION_MAJOR(d.properties.apiVersion),
              VK_API_VERSION_MINOR(d.properties.apiVersion), VK_API_VERSION_PATCH(d.properties.apiVersion));

  const char* extensions[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queue_info.queueFamilyIndex = d.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  device_info.pNext = &features;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = 1;
  device_info.ppEnabledExtensionNames = extensions;
  Check(vkCreateDevice(d.physical, &device_info, nullptr, &d.device), "vkCreateDevice");
  vkGetDeviceQueue(d.device, d.queue_family, 0, &d.queue);

  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.queueFamilyIndex = d.queue_family;
  Check(vkCreateCommandPool(d.device, &pool_info, nullptr, &d.pool), "vkCreateCommandPool");
  return d;
}

// The CPU reference is O(M*N*K), so only shapes up to this many multiply-adds are checked.
constexpr uint64_t kMaxCheckedMultiplyAdds = 1ull << 31;

// One GEMM run: fills A and B with deterministic data, times `iterations` dispatches, then checks shapes small enough
// for the CPU reference.
double RunGemm(Device& d, VkPipeline pipeline, VkPipelineLayout layout, uint32_t M, uint32_t N, uint32_t K,
             int iterations, bool check) {
  const uint32_t rows = (M + kTile - 1) / kTile * kTile;  // the kernel runs whole tiles
  const bool checked = check && static_cast<uint64_t>(M) * N * K <= kMaxCheckedMultiplyAdds;
  const VkDeviceSize a_bytes = static_cast<VkDeviceSize>(rows) * K * 2;
  const VkDeviceSize b_bytes = static_cast<VkDeviceSize>(N) * K * 2;
  const VkDeviceSize c_bytes = static_cast<VkDeviceSize>(rows) * N * 4;

  std::vector<uint16_t> a(static_cast<size_t>(rows) * K);
  std::vector<uint16_t> b(static_cast<size_t>(N) * K);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : a) v = FloatToHalf(dist(rng));
  for (auto& v : b) v = FloatToHalf(dist(rng));
  // NaN in the padding rows: any leak into the first M rows of C fails the check.
  if (checked) std::fill(a.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(M) * K), a.end(), uint16_t{0x7e00});

  Buffer buffer_a = CreateBuffer(d, a_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
  Buffer buffer_b = CreateBuffer(d, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
  Buffer buffer_c = CreateBuffer(d, c_bytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true);
  Upload(d, buffer_a, a.data(), a_bytes);
  Upload(d, buffer_b, b.data(), b_bytes);

  struct Push {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
  } push{buffer_a.address, buffer_b.address, buffer_c.address, M, N, K, K, K, N};

  const auto record = [&](VkCommandBuffer cb) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    for (int i = 0; i < iterations; ++i) {
      vkCmdDispatch(cb, N / kTile, rows / kTile, 1);
      Barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    }
  };

  Submit(d, record);  // warm-up (compiles or loads the pipeline cache)
  const auto start = std::chrono::steady_clock::now();
  Submit(d, record);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / static_cast<double>(iterations);

  double max_error = -1.0;
  if (checked) {
    std::vector<float> c(static_cast<size_t>(rows) * N);
    Download(d, buffer_c, c.data(), c_bytes);
    double max_abs = 0.0;
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        double reference = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
          reference += static_cast<double>(HalfToFloat(a[static_cast<size_t>(m) * K + k])) *
                       static_cast<double>(HalfToFloat(b[static_cast<size_t>(n) * K + k]));
        }
        const double error = std::fabs(static_cast<double>(c[static_cast<size_t>(m) * N + n]) - reference);
        max_abs = std::isnan(error) ? INFINITY : std::max(max_abs, error);
      }
    }
    max_error = max_abs;
    std::printf("check %ux%ux%u: max|C - CPU| = %.6g\n", M, N, K, max_abs);
  }

  const double flops = 2.0 * M * N * K;
  std::printf("gemm %ux%ux%u: %8.3f ms  %7.1f TFLOPS\n", M, N, K, seconds * 1e3, flops / seconds / 1e12);

  DestroyBuffer(d, buffer_a);
  DestroyBuffer(d, buffer_b);
  DestroyBuffer(d, buffer_c);
  return max_error;
}

}  // namespace

int main(int argc, char** argv) {
  std::string spv = DEFAULT_SPV_PATH;
  bool check = true;
  int device_index = -1;
  struct Shape {
    uint32_t M, N, K;
    int iterations;
  };
  std::vector<Shape> shapes;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--spv" && i + 1 < argc) {
      spv = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device_index = std::atoi(argv[++i]);
    } else if (arg == "--no-check") {
      check = false;
    } else if (arg == "--bench" && i + 4 < argc) {
      Shape s{};
      s.M = static_cast<uint32_t>(std::atoi(argv[++i]));
      s.N = static_cast<uint32_t>(std::atoi(argv[++i]));
      s.K = static_cast<uint32_t>(std::atoi(argv[++i]));
      s.iterations = std::atoi(argv[++i]);
      shapes.push_back(s);
    } else {
      std::printf("usage: %s [--spv <gemm_f16.spv>] [--device N] [--no-check] [--bench M N K iters]...\n", argv[0]);
      return 2;
    }
  }
  if (shapes.empty()) {
    // Checked: one tile, a partial last band, several bands, and row tails. Then the engine's real shapes at 1024x1024
    // with CFG: the DiT's QKV, projection, up/gate and down GEMMs, and the text encoder's GatedDeltaNet projections.
    shapes = {{64, 64, 64, 1},       {128, 640, 64, 1},     {256, 1024, 64, 1},    {94, 640, 64, 1},
              {1030, 9216, 64, 1},   {512, 512, 512, 20},   {8222, 9216, 3072, 5}, {8222, 3072, 3072, 5},
              {8222, 16384, 3072, 3}, {8222, 3072, 8192, 3}, {282, 12288, 4096, 10}, {282, 4096, 12288, 10}};
  }
  for (const Shape& s : shapes) {
    if (s.M == 0 || s.N == 0 || s.K == 0 || s.iterations <= 0 || s.N % kTile != 0 || s.K % 16 != 0) {
      std::printf("N must be a multiple of %u and K of 16, and all must be positive: %ux%ux%u\n", kTile, s.M, s.N, s.K);
      return 2;
    }
  }

  Device d = OpenDevice(device_index);
  PrintCooperativeMatrixShapes(d);

  const std::vector<uint32_t> spirv = ReadFile(spv);
  VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  module_info.codeSize = spirv.size() * 4;
  module_info.pCode = spirv.data();
  VkShaderModule module = VK_NULL_HANDLE;
  Check(vkCreateShaderModule(d.device, &module_info, nullptr, &module), "vkCreateShaderModule");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = 48;
  VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges = &push_range;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  Check(vkCreatePipelineLayout(d.device, &layout_info, nullptr, &layout), "vkCreatePipelineLayout");

  VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  // The kernel's tile split assumes exactly four full subgroups of 32. Without this a driver may pick another size.
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
  subgroup_size.requiredSubgroupSize = kSubgroupSize;
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.pNext = &subgroup_size;
  pipeline_info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = module;
  pipeline_info.stage.pName = "main";
  pipeline_info.layout = layout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  Check(vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
        "vkCreateComputePipelines");

  // Inputs are fp16 and the accumulator is fp32 (no fp64 reference is exact), so this bounds rounding, not bits.
  constexpr double kMaxError = 1e-2;
  bool failed = false;
  for (const Shape& s : shapes) {
    const double error = RunGemm(d, pipeline, layout, s.M, s.N, s.K, s.iterations, check);
    if (error > kMaxError) {
      std::printf("FAILED: %ux%ux%u: error %.6g is above %g\n", s.M, s.N, s.K, error, kMaxError);
      failed = true;
    }
  }

  vkDestroyPipeline(d.device, pipeline, nullptr);
  vkDestroyPipelineLayout(d.device, layout, nullptr);
  vkDestroyShaderModule(d.device, module, nullptr);
  vkDestroyCommandPool(d.device, d.pool, nullptr);
  vkDestroyDevice(d.device, nullptr);
  vkDestroyInstance(d.instance, nullptr);
  return failed ? 1 : 0;
}
