# The Vulkan backend (vulkan/), the fallback for GPUs the CUDA engine does not support: `moss-infer generate --backend
# vulkan`. Off by default; it needs the Vulkan headers (1.4.313 or newer, for VK_KHR_shader_bfloat16), the loader and
# glslc. Without it the CLI refuses --backend vulkan with a message saying so.
option(MOSSINFER_VULKAN "Build the Vulkan backend into the engine (moss-infer generate --backend vulkan)" OFF)
if(MOSSINFER_VULKAN)
  set(MOSSINFER_VULKAN_TOOLS OFF CACHE BOOL "Build the Vulkan backend's check and benchmark tools")
  add_subdirectory(${CMAKE_SOURCE_DIR}/vulkan ${CMAKE_BINARY_DIR}/vulkan)
  set_target_properties(mossinfer_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(mossinfer_core PUBLIC mossinfer_vulkan)
  target_compile_definitions(mossinfer_core PUBLIC MOSSINFER_VULKAN=1)
endif()
