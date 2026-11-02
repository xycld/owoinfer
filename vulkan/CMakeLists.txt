# The Vulkan backend, the fallback for hardware the engine does not otherwise support. A standalone project for now: it
# links no engine code, and nothing here affects the engine's build. See README.md.
cmake_minimum_required(VERSION 3.20)
project(mossinfer_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)  # the CPU reference is unusably slow unoptimized
endif()

find_package(Vulkan REQUIRED)
find_program(GLSL_COMPILER NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)

# Shaders become SPIR-V word lists compiled into the binary, so nothing beyond it ships.
set(_spirv_dir ${CMAKE_CURRENT_BINARY_DIR}/spirv)
file(MAKE_DIRECTORY ${_spirv_dir})
set(_spirv_files "")
foreach(shader IN ITEMS gemm_coop gemm_plain)
  set(_source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp)
  set(_output ${_spirv_dir}/${shader}.spv.inc)
  add_custom_command(
    OUTPUT ${_output}
    COMMAND ${GLSL_COMPILER} --target-env=vulkan1.3 -fshader-stage=comp -O -mfmt=num ${_source} -o ${_output}
    DEPENDS ${_source}
    COMMENT "glslc ${shader}.comp")
  list(APPEND _spirv_files ${_output})
endforeach()

add_library(mossinfer_vulkan STATIC runtime.cpp gemm.cpp ${_spirv_files})
target_include_directories(mossinfer_vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${_spirv_dir})
target_link_libraries(mossinfer_vulkan PUBLIC Vulkan::Vulkan)

add_executable(vulkan_check tools/vulkan_check.cpp)
target_link_libraries(vulkan_check PRIVATE mossinfer_vulkan)
