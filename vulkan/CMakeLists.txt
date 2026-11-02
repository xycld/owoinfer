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
# The cooperative-matrix shader needs Vulkan 1.3; the rest must also load on Vulkan 1.2 devices, which reject newer SPIR-V.
function(_mi_add_shader shader target_env)
  set(_source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp)
  set(_output ${_spirv_dir}/${shader}.spv.inc)
  add_custom_command(
    OUTPUT ${_output}
    COMMAND ${GLSL_COMPILER} --target-env=${target_env} -fshader-stage=comp -O -mfmt=num
            -I${CMAKE_CURRENT_SOURCE_DIR}/shaders ${_source} -o ${_output}
    DEPENDS ${_source} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl
    COMMENT "glslc ${shader}.comp")
  set(_spirv_files ${_spirv_files} ${_output} PARENT_SCOPE)
endfunction()
_mi_add_shader(gemm_coop vulkan1.3)
foreach(shader IN ITEMS gemm_plain norm_modulate qk_norm_rope attention_plain)
  _mi_add_shader(${shader} vulkan1.2)
endforeach()

add_library(mossinfer_vulkan STATIC runtime.cpp gemm.cpp block.cpp ${_spirv_files})
target_include_directories(mossinfer_vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${_spirv_dir})
target_link_libraries(mossinfer_vulkan PUBLIC Vulkan::Vulkan)

add_executable(vulkan_check tools/vulkan_check.cpp)
target_link_libraries(vulkan_check PRIVATE mossinfer_vulkan)

# The host replays the block is checked against must round where the kernels do: no contraction.
add_executable(vulkan_block_check tools/vulkan_block_check.cpp)
target_compile_options(vulkan_block_check PRIVATE -ffp-contract=off)
target_link_libraries(vulkan_block_check PRIVATE mossinfer_vulkan)
