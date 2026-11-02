# Standalone build for the Vulkan cooperative-matrix GEMM bring-up tool. It is deliberately separate from the engine's
# CMake project: no engine code is linked, and nothing here affects a normal build. See README.md.
cmake_minimum_required(VERSION 3.20)
project(mossinfer_vulkan_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer the Vulkan SDK's CMake package; fall back to explicit paths for machines without the SDK (the loader's import
# library can be generated from vulkan-1.dll, see README.md).
find_package(Vulkan QUIET)
set(VULKAN_INCLUDE_DIR "" CACHE PATH "Vulkan include directory when no SDK is installed")
set(VULKAN_LIBRARY "" CACHE FILEPATH "Vulkan loader import library when no SDK is installed")

if(VULKAN_INCLUDE_DIR AND VULKAN_LIBRARY)
  set(VULKAN_INCLUDE ${VULKAN_INCLUDE_DIR})
  set(VULKAN_TARGET ${VULKAN_LIBRARY})
elseif(Vulkan_FOUND)
  set(VULKAN_INCLUDE ${Vulkan_INCLUDE_DIRS})
  set(VULKAN_TARGET Vulkan::Vulkan)
else()
  message(FATAL_ERROR
          "Vulkan not found: install the Vulkan SDK, or pass -DVULKAN_INCLUDE_DIR=<dir> -DVULKAN_LIBRARY=<vulkan-1.lib>")
endif()

find_program(GLSL_COMPILER NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(NOT GLSL_COMPILER)
  message(FATAL_ERROR "glslc not found: it ships with the Vulkan SDK")
endif()

set(GERM_SPV ${CMAKE_CURRENT_BINARY_DIR}/gemm_f16.spv)
add_custom_command(
  OUTPUT ${GERM_SPV}
  COMMAND ${GLSL_COMPILER} --target-env=vulkan1.3 -fshader-stage=comp -O ${CMAKE_CURRENT_SOURCE_DIR}/gemm_f16.comp
          -o ${GERM_SPV}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_f16.comp
  COMMENT "glslc gemm_f16.comp")
add_custom_target(gemm_spv ALL DEPENDS ${GERM_SPV})

add_executable(vulkan_gemm main.cpp)
target_include_directories(vulkan_gemm PRIVATE ${VULKAN_INCLUDE})
target_link_libraries(vulkan_gemm PRIVATE ${VULKAN_TARGET})
target_compile_definitions(vulkan_gemm PRIVATE "DEFAULT_SPV_PATH=\"${GERM_SPV}\"")
add_dependencies(vulkan_gemm gemm_spv)
