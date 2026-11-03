# The HIP backend: MOSS-Image's kernels for AMD GPUs through ROCm, and the same sources through HIP's NVIDIA headers for
# checking on a CUDA device. Its own project; the engine's build does not include it. Configure once per target (README.md):
#   -DHIP_TARGET=gfx1100|gfx1200|gfx1201  rocWMMA leaf, ROCm's clang, links libamdhip64
#   -DHIP_TARGET=nvidia                   scalar leaf, nvcc with HIP's NVIDIA headers
cmake_minimum_required(VERSION 3.28)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/version.cmake)  # MOSSINFER_VERSION, from the C header

set(HIP_TARGET "" CACHE STRING "Device target: gfx1100, gfx1200, gfx1201 or nvidia")
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm root, the directory holding include/hip and lib/llvm")
option(MI_HIP_WERROR "Treat warnings as errors" ON)

if(NOT HIP_TARGET MATCHES "^(gfx1100|gfx1200|gfx1201|nvidia)$")
  message(FATAL_ERROR "Set HIP_TARGET to gfx1100, gfx1200, gfx1201 or nvidia (got '${HIP_TARGET}')")
endif()
if(NOT EXISTS "${ROCM_PATH}/include/hip/hip_runtime.h")
  message(FATAL_ERROR "ROCM_PATH (cache variable or environment) must hold include/hip/hip_runtime.h (got '${ROCM_PATH}')")
endif()
if(HIP_TARGET MATCHES "^gfx")
  set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++")
endif()

project(mossinfer_hip VERSION ${MOSSINFER_VERSION} LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(MI_HIP_WARN -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(MI_HIP_WERROR)
  list(APPEND MI_HIP_WARN -Werror)
endif()
# Host code that replays or extends the kernels' rounding (the CFG and Euler step, the time embedding) rounds every fp32
# operation, as the kernels do.
list(APPEND MI_HIP_WARN -ffp-contract=off)

# Kernel sources. block_kernels.hip is the DiT block (tools/hip_block checks it against the engine); dit_kernels.hip the
# DiT's other ops.
set(MI_HIP_KERNELS block_kernels.hip dit_kernels.hip)
set(MI_HIP_KERNEL_HEADERS device_math.h dit_kernels.h gemm_tile.h hip_block.h mma_leaf.h)
set(MI_HIP_HOST runtime.cpp safetensors.cpp dit.cpp)

if(HIP_TARGET STREQUAL "nvidia")
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures of the NVIDIA build")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set_source_files_properties(${MI_HIP_KERNELS} PROPERTIES LANGUAGE CUDA)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _mi_cuda_arch)
  string(REGEX REPLACE "[^0-9].*" "" _mi_cuda_arch "${_mi_cuda_arch}")

  add_library(mossinfer_hip STATIC ${MI_HIP_KERNELS} ${MI_HIP_HOST})
  target_compile_definitions(mossinfer_hip PUBLIC __HIP_PLATFORM_NVIDIA__=1 PRIVATE MI_HIP_TARGET="nvidia"
                                                  MI_HIP_CUDA_ARCH=${_mi_cuda_arch})
  target_include_directories(mossinfer_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(mossinfer_hip SYSTEM PUBLIC ${ROCM_PATH}/include ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(mossinfer_hip PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
                                               $<$<COMPILE_LANGUAGE:CXX>:${MI_HIP_WARN}>)
  target_link_libraries(mossinfer_hip PUBLIC CUDA::cudart)
else()
  set(MI_HIP_FLAGS --rocm-path=${ROCM_PATH} --hip-path=${ROCM_PATH} --offload-arch=${HIP_TARGET})
  set_source_files_properties(${MI_HIP_KERNELS} PROPERTIES LANGUAGE CXX)
  foreach(_src IN LISTS MI_HIP_KERNELS)
    set_property(SOURCE ${_src} PROPERTY COMPILE_OPTIONS -x hip -std=c++17 ${MI_HIP_FLAGS} ${MI_HIP_WARN})
  endforeach()
  set_property(SOURCE ${MI_HIP_HOST} PROPERTY COMPILE_OPTIONS ${MI_HIP_WARN})

  add_library(mossinfer_hip STATIC ${MI_HIP_KERNELS} ${MI_HIP_HOST})
  target_compile_definitions(mossinfer_hip PUBLIC __HIP_PLATFORM_AMD__=1 PRIVATE HIP_BLOCK_LEAF_ROCWMMA
                                                  MI_HIP_TARGET="${HIP_TARGET}")
  target_include_directories(mossinfer_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(mossinfer_hip SYSTEM PUBLIC ${ROCM_PATH}/include)
  target_link_options(mossinfer_hip PUBLIC -L${ROCM_PATH}/lib -Wl,-rpath,${ROCM_PATH}/lib)
  target_link_libraries(mossinfer_hip PUBLIC amdhip64)

  # Each kernel source's device build on its own: the assembly and linked code object check_device.sh reads. Same flags
  # as the library's compile.
  set(MI_HIP_DEVICE_ARGS -x hip -std=c++17 -O3 ${MI_HIP_FLAGS} --cuda-device-only -D__HIP_PLATFORM_AMD__=1
      -DHIP_BLOCK_LEAF_ROCWMMA -I${CMAKE_CURRENT_SOURCE_DIR} -isystem ${ROCM_PATH}/include ${MI_HIP_WARN})
  set(_deps "")
  foreach(_h IN LISTS MI_HIP_KERNEL_HEADERS)
    list(APPEND _deps ${CMAKE_CURRENT_SOURCE_DIR}/${_h})
  endforeach()
  set(_checks "")
  foreach(_src IN LISTS MI_HIP_KERNELS)
    get_filename_component(_name ${_src} NAME_WE)
    add_custom_command(OUTPUT ${_name}.s DEPENDS ${_src} ${_deps}
      COMMAND ${CMAKE_CXX_COMPILER} ${MI_HIP_DEVICE_ARGS} -S -o ${_name}.s ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
      COMMENT "Device assembly of ${_src} for ${HIP_TARGET}")
    add_custom_command(OUTPUT ${_name}.co DEPENDS ${_src} ${_deps}
      COMMAND ${CMAKE_CXX_COMPILER} ${MI_HIP_DEVICE_ARGS} --no-gpu-bundle-output -o ${_name}.co
              ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
      COMMENT "Device code object of ${_src} for ${HIP_TARGET}")
    add_custom_command(OUTPUT ${_name}.check DEPENDS ${_name}.s ${_name}.co ${CMAKE_CURRENT_SOURCE_DIR}/check_device.sh
      COMMAND ${CMAKE_COMMAND} -E env --unset=HIP_BLOCK_WMMA_COUNTS sh ${CMAKE_CURRENT_SOURCE_DIR}/check_device.sh
              ${ROCM_PATH}/lib/llvm/bin ${HIP_TARGET} ${_name}.s ${_name}.co
      COMMAND ${CMAKE_COMMAND} -E touch ${_name}.check
      COMMENT "check_device.sh ${_name}" VERBATIM)
    list(APPEND _checks ${_name}.check)
  endforeach()
  add_custom_target(mossinfer_hip_device_check ALL DEPENDS ${_checks})
endif()

# The checks and tools.
add_executable(hip_check tools/hip_check.cpp)
target_compile_options(hip_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_check PRIVATE mossinfer_hip)

# dit_kernels.h's ops against host replays; the replays round every fp32 operation, as the kernels do.
add_executable(hip_dit_kernel_check tools/hip_dit_kernel_check.cpp)
target_compile_options(hip_dit_kernel_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_dit_kernel_check PRIVATE mossinfer_hip)

# The DiT's sampling loop from a golden case; CFG and Euler on the host round like the engine's kernels.
add_executable(hip_dit_run tools/hip_dit_run.cpp)
target_compile_options(hip_dit_run PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_dit_run PRIVATE mossinfer_hip)
