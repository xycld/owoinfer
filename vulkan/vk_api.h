// The Vulkan API as the backend uses it: the headers without prototypes, and the core functions as pointers that
// load_vulkan() resolves from the system's Vulkan loader (libvulkan.so.1, vulkan-1.dll, libvulkan.1.dylib) at run time.
// So a program with the backend built in starts on a machine without Vulkan and fails only when it asks for a device.
// Extension functions come from vkGetInstanceProcAddr / vkGetDeviceProcAddr where they are used.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <string>

#define MI_VK_FUNCTIONS(X)                     \
  X(vkGetInstanceProcAddr)                     \
  X(vkGetDeviceProcAddr)                       \
  X(vkCreateInstance)                          \
  X(vkDestroyInstance)                         \
  X(vkEnumeratePhysicalDevices)                \
  X(vkEnumerateDeviceExtensionProperties)      \
  X(vkGetPhysicalDeviceProperties)             \
  X(vkGetPhysicalDeviceProperties2)            \
  X(vkGetPhysicalDeviceFeatures2)              \
  X(vkGetPhysicalDeviceMemoryProperties)       \
  X(vkGetPhysicalDeviceMemoryProperties2)      \
  X(vkGetPhysicalDeviceQueueFamilyProperties)  \
  X(vkCreateDevice)                            \
  X(vkDestroyDevice)                           \
  X(vkGetDeviceQueue)                          \
  X(vkQueueSubmit)                             \
  X(vkQueueWaitIdle)                           \
  X(vkCreateCommandPool)                       \
  X(vkDestroyCommandPool)                      \
  X(vkAllocateCommandBuffers)                  \
  X(vkFreeCommandBuffers)                      \
  X(vkBeginCommandBuffer)                      \
  X(vkEndCommandBuffer)                        \
  X(vkCreateBuffer)                            \
  X(vkDestroyBuffer)                           \
  X(vkGetBufferMemoryRequirements)             \
  X(vkGetBufferDeviceAddress)                  \
  X(vkAllocateMemory)                          \
  X(vkFreeMemory)                              \
  X(vkBindBufferMemory)                        \
  X(vkMapMemory)                               \
  X(vkCreateShaderModule)                      \
  X(vkDestroyShaderModule)                     \
  X(vkCreatePipelineLayout)                    \
  X(vkDestroyPipelineLayout)                   \
  X(vkCreateComputePipelines)                  \
  X(vkDestroyPipeline)                         \
  X(vkCmdBindPipeline)                         \
  X(vkCmdPushConstants)                        \
  X(vkCmdDispatch)                             \
  X(vkCmdPipelineBarrier)                      \
  X(vkCmdCopyBuffer)                           \
  X(vkCmdFillBuffer)

#define MI_VK_DECLARE(name) extern PFN_##name name;
MI_VK_FUNCTIONS(MI_VK_DECLARE)
#undef MI_VK_DECLARE

namespace mi::vk {

// Opens the Vulkan loader once and resolves every function above. Returns false with a reason when the loader is
// missing or lacks one of them (a loader older than Vulkan 1.2). Thread-safe; later calls return the first result.
bool load_vulkan(std::string* error);

}  // namespace mi::vk
