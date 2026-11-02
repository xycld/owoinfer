# Standalone: the engine's CMakeLists.txt does not know this directory. Configure once per target (README.md).
#   -DHIP_BLOCK_TARGET=gfx1100|gfx1201  rocWMMA leaf, ROCm's clang, links libamdhip64
#   -DHIP_BLOCK_TARGET=nvidia           scalar leaf, nvcc with HIP's NVIDIA headers; adds the engine comparison test
cmake_minimum_required(VERSION 3.28)

set(HIP_BLOCK_TARGET "" CACHE STRING "Device target: gfx1100, gfx1201 or nvidia")
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm root, the directory holding include/hip and lib/llvm")
option(HIP_BLOCK_WERROR "Treat warnings as errors" ON)

if(NOT HIP_BLOCK_TARGET MATCHES "^(gfx1100|gfx1201|nvidia)$")
  message(FATAL_ERROR "Set HIP_BLOCK_TARGET to gfx1100, gfx1201 or nvidia (got '${HIP_BLOCK_TARGET}')")
endif()
if(NOT EXISTS "${ROCM_PATH}/include/hip/hip_runtime.h")
  message(FATAL_ERROR "ROCM_PATH (cache variable or environment) must hold include/hip/hip_runtime.h (got '${ROCM_PATH}')")
endif()
if(HIP_BLOCK_TARGET MATCHES "^gfx")
  set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++")
endif()

project(hip_block LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(HIP_BLOCK_WARN -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(HIP_BLOCK_WERROR)
  list(APPEND HIP_BLOCK_WARN -Werror)
endif()
# The host replays of the kernels (host_util.h) round every fp32 operation.
list(APPEND HIP_BLOCK_WARN -ffp-contract=off)

if(HIP_BLOCK_TARGET STREQUAL "nvidia")
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures of the verify build")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set_source_files_properties(block_kernels.hip verify_kernels.cu PROPERTIES LANGUAGE CUDA)

  add_library(hip_block_kernels OBJECT block_kernels.hip verify_kernels.cu)
  target_compile_definitions(hip_block_kernels PUBLIC __HIP_PLATFORM_NVIDIA__=1)
  target_include_directories(hip_block_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(hip_block_kernels SYSTEM PUBLIC ${ROCM_PATH}/include ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(hip_block_kernels PRIVATE -Xcompiler=-Wall)

  add_executable(hip_block_bench bench.cpp)
  target_link_libraries(hip_block_bench PRIVATE hip_block_kernels CUDA::cudart)
  target_compile_definitions(hip_block_bench PRIVATE HIP_BLOCK_TARGET_NAME="nvidia")
  target_compile_options(hip_block_bench PRIVATE ${HIP_BLOCK_WARN})

  # The engine: a build of this tree's mossinfer_core (README.md), at a build directory of the same CUDA toolchain.
  set(MOSSINFER_ENGINE_BUILD_DIR "" CACHE PATH "Build directory holding libmossinfer_core.a")
  get_filename_component(HIP_BLOCK_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  if(NOT EXISTS "${MOSSINFER_ENGINE_BUILD_DIR}/libmossinfer_core.a")
    message(FATAL_ERROR "MOSSINFER_ENGINE_BUILD_DIR must hold libmossinfer_core.a (got '${MOSSINFER_ENGINE_BUILD_DIR}')")
  endif()
  add_executable(hip_block_verify verify.cpp)
  target_include_directories(hip_block_verify SYSTEM PRIVATE ${HIP_BLOCK_REPO_ROOT}/src ${HIP_BLOCK_REPO_ROOT}/include)
  target_link_libraries(hip_block_verify PRIVATE hip_block_kernels "${MOSSINFER_ENGINE_BUILD_DIR}/libmossinfer_core.a"
                                                 CUDA::cudart ${CMAKE_DL_LIBS})
  target_compile_options(hip_block_verify PRIVATE ${HIP_BLOCK_WARN})
else()
  set(HIP_BLOCK_HIP_FLAGS --rocm-path=${ROCM_PATH} --hip-path=${ROCM_PATH} --offload-arch=${HIP_BLOCK_TARGET})

  add_executable(hip_block_bench bench.cpp block_kernels.hip)
  set_source_files_properties(block_kernels.hip PROPERTIES LANGUAGE CXX)
  target_compile_definitions(hip_block_bench PRIVATE __HIP_PLATFORM_AMD__=1 HIP_BLOCK_LEAF_ROCWMMA
                                                     HIP_BLOCK_TARGET_NAME="${HIP_BLOCK_TARGET}")
  target_include_directories(hip_block_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(hip_block_bench SYSTEM PRIVATE ${ROCM_PATH}/include)
  set_property(SOURCE bench.cpp PROPERTY COMPILE_OPTIONS ${HIP_BLOCK_WARN})
  set_property(SOURCE block_kernels.hip PROPERTY COMPILE_OPTIONS -x hip -std=c++17 ${HIP_BLOCK_HIP_FLAGS} ${HIP_BLOCK_WARN})
  target_link_options(hip_block_bench PRIVATE -L${ROCM_PATH}/lib -Wl,-rpath,${ROCM_PATH}/lib)
  target_link_libraries(hip_block_bench PRIVATE amdhip64)

  # The device build on its own: the assembly and linked code object that check_device.sh reads.
  set(HIP_BLOCK_DEVICE_ARGS -x hip -std=c++17 -O3 ${HIP_BLOCK_HIP_FLAGS} --cuda-device-only -D__HIP_PLATFORM_AMD__=1
      -DHIP_BLOCK_LEAF_ROCWMMA -I${CMAKE_CURRENT_SOURCE_DIR} -isystem ${ROCM_PATH}/include)
  set(HIP_BLOCK_SRC ${CMAKE_CURRENT_SOURCE_DIR}/block_kernels.hip)
  set(HIP_BLOCK_DEPS ${HIP_BLOCK_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/mma_leaf.h ${CMAKE_CURRENT_SOURCE_DIR}/device_math.h
      ${CMAKE_CURRENT_SOURCE_DIR}/hip_block.h)
  add_custom_command(OUTPUT block_kernels.s DEPENDS ${HIP_BLOCK_DEPS}
    COMMAND ${CMAKE_CXX_COMPILER} ${HIP_BLOCK_DEVICE_ARGS} -S -o block_kernels.s ${HIP_BLOCK_SRC}
    COMMENT "Device assembly for ${HIP_BLOCK_TARGET}")
  add_custom_command(OUTPUT block_kernels.co DEPENDS ${HIP_BLOCK_DEPS}
    COMMAND ${CMAKE_CXX_COMPILER} ${HIP_BLOCK_DEVICE_ARGS} --no-gpu-bundle-output -o block_kernels.co ${HIP_BLOCK_SRC}
    COMMENT "Device code object for ${HIP_BLOCK_TARGET}")
  add_custom_target(hip_block_device_check ALL
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/check_device.sh ${ROCM_PATH}/lib/llvm/bin block_kernels.s block_kernels.co
    DEPENDS block_kernels.s block_kernels.co VERBATIM)
endif()
