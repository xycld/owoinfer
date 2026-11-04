# The HIP backend: MOSS-Image's kernels for AMD GPUs through ROCm, and the same sources through HIP's NVIDIA headers for
# checking on a CUDA device. Its own project; the engine's build does not include it. Configure once per target (README.md):
#   -DHIP_TARGET="gfx1100;gfx1200;gfx1201"  one or more AMD architectures in one binary (the release package: all three);
#                                           rocWMMA leaf, ROCm's clang, links libamdhip64
#   -DHIP_TARGET=nvidia                     scalar leaf, nvcc with HIP's NVIDIA headers
cmake_minimum_required(VERSION 3.28)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/version.cmake)  # MOSSINFER_VERSION, from the C header

set(HIP_TARGET "" CACHE STRING "Device targets: a list of gfx1100, gfx1200, gfx1201, or nvidia")
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm root, the directory holding include/hip and lib/llvm")
option(MI_HIP_WERROR "Treat warnings as errors" ON)

# MI_HIP_AMD: the build is for AMD architectures (HIP_TARGET, a list of them), else for an NVIDIA device.
set(MI_HIP_AMD OFF)
if(HIP_TARGET STREQUAL "nvidia")
elseif(HIP_TARGET)
  set(MI_HIP_AMD ON)
  list(REMOVE_DUPLICATES HIP_TARGET)
  foreach(_arch IN LISTS HIP_TARGET)
    if(NOT _arch MATCHES "^(gfx1100|gfx1200|gfx1201)$")
      set(MI_HIP_AMD OFF)
    endif()
  endforeach()
endif()
if(NOT MI_HIP_AMD AND NOT HIP_TARGET STREQUAL "nvidia")
  message(FATAL_ERROR "Set HIP_TARGET to a list of gfx1100, gfx1200, gfx1201, or to nvidia (got '${HIP_TARGET}')")
endif()
if(NOT EXISTS "${ROCM_PATH}/include/hip/hip_runtime.h")
  message(FATAL_ERROR "ROCM_PATH (cache variable or environment) must hold include/hip/hip_runtime.h (got '${ROCM_PATH}')")
endif()
if(MI_HIP_AMD)
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
# DiT's other ops; te_kernels.hip the text encoder's; vae_kernels.hip the VAE decoder's.
set(MI_HIP_KERNELS block_kernels.hip dit_kernels.hip te_kernels.hip vae_kernels.hip)
set(MI_HIP_KERNEL_HEADERS device_math.h dit_kernels.h gemm_tile.h hip_block.h mma_leaf.h te_kernels.h vae_kernels.h)
set(MI_HIP_HOST runtime.cpp safetensors.cpp dit.cpp text_encoder.cpp vae.cpp generate.cpp)

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
  # One fat binary: a code object per architecture. MI_HIP_TARGET names them, comma-separated (runtime.cpp).
  set(MI_HIP_PATHS --rocm-path=${ROCM_PATH} --hip-path=${ROCM_PATH})
  set(MI_HIP_FLAGS ${MI_HIP_PATHS})
  foreach(_arch IN LISTS HIP_TARGET)
    list(APPEND MI_HIP_FLAGS --offload-arch=${_arch})
  endforeach()
  list(JOIN HIP_TARGET "," MI_HIP_TARGET_NAMES)
  set_source_files_properties(${MI_HIP_KERNELS} PROPERTIES LANGUAGE CXX)
  foreach(_src IN LISTS MI_HIP_KERNELS)
    set_property(SOURCE ${_src} PROPERTY COMPILE_OPTIONS -x hip -std=c++17 ${MI_HIP_FLAGS} ${MI_HIP_WARN})
  endforeach()
  set_property(SOURCE ${MI_HIP_HOST} PROPERTY COMPILE_OPTIONS ${MI_HIP_WARN})

  add_library(mossinfer_hip STATIC ${MI_HIP_KERNELS} ${MI_HIP_HOST})
  target_compile_definitions(mossinfer_hip PUBLIC __HIP_PLATFORM_AMD__=1 PRIVATE HIP_BLOCK_LEAF_ROCWMMA
                                                  MI_HIP_TARGET="${MI_HIP_TARGET_NAMES}")
  target_include_directories(mossinfer_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(mossinfer_hip SYSTEM PUBLIC ${ROCM_PATH}/include)
  # libamdhip64 from ROCM_PATH; the executables find it there in the build tree (BUILD_RPATH), and the installed
  # moss-infer-hip in /opt/rocm/lib (INSTALL_RPATH, below).
  target_link_directories(mossinfer_hip PUBLIC ${ROCM_PATH}/lib)
  target_link_libraries(mossinfer_hip PUBLIC amdhip64)
  set(CMAKE_BUILD_RPATH ${ROCM_PATH}/lib)

  # Each kernel source's device build on its own, per architecture: the assembly and linked code object
  # check_device.sh reads (<source>.<arch>.s, .co). Same flags as the library's compile.
  set(MI_HIP_DEVICE_ARGS -x hip -std=c++17 -O3 ${MI_HIP_PATHS} --cuda-device-only -D__HIP_PLATFORM_AMD__=1
      -DHIP_BLOCK_LEAF_ROCWMMA -I${CMAKE_CURRENT_SOURCE_DIR} -isystem ${ROCM_PATH}/include ${MI_HIP_WARN})
  set(_deps "")
  foreach(_h IN LISTS MI_HIP_KERNEL_HEADERS)
    list(APPEND _deps ${CMAKE_CURRENT_SOURCE_DIR}/${_h})
  endforeach()
  set(_checks "")
  foreach(_arch IN LISTS HIP_TARGET)
    foreach(_src IN LISTS MI_HIP_KERNELS)
      get_filename_component(_name ${_src} NAME_WE)
      set(_out ${_name}.${_arch})
      add_custom_command(OUTPUT ${_out}.s DEPENDS ${_src} ${_deps}
        COMMAND ${CMAKE_CXX_COMPILER} ${MI_HIP_DEVICE_ARGS} --offload-arch=${_arch} -S -o ${_out}.s
                ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
        COMMENT "Device assembly of ${_src} for ${_arch}")
      add_custom_command(OUTPUT ${_out}.co DEPENDS ${_src} ${_deps}
        COMMAND ${CMAKE_CXX_COMPILER} ${MI_HIP_DEVICE_ARGS} --offload-arch=${_arch} --no-gpu-bundle-output -o ${_out}.co
                ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
        COMMENT "Device code object of ${_src} for ${_arch}")
      add_custom_command(OUTPUT ${_out}.check DEPENDS ${_out}.s ${_out}.co ${CMAKE_CURRENT_SOURCE_DIR}/check_device.sh
        COMMAND ${CMAKE_COMMAND} -E env --unset=HIP_BLOCK_WMMA_COUNTS sh ${CMAKE_CURRENT_SOURCE_DIR}/check_device.sh
                ${ROCM_PATH}/lib/llvm/bin ${_arch} ${_out}.s ${_out}.co
        COMMAND ${CMAKE_COMMAND} -E touch ${_out}.check
        COMMENT "check_device.sh ${_name} ${_arch}" VERBATIM)
      list(APPEND _checks ${_out}.check)
    endforeach()
  endforeach()
  add_custom_target(mossinfer_hip_device_check ALL DEPENDS ${_checks})
endif()

# moss-infer-hip, the standalone program: the backend plus the engine's host sources it shares with moss-infer (the
# tokenizer, the seed's initial noise, the schedule, the PNG writer), compiled here so that nothing needs CUDA. They are
# the engine's sources, built without this project's warnings; the torch-compatible noise is bit-exact only without
# contraction or fast-math, which clang would otherwise apply.
set(_engine_src ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_library(mossinfer_hip_host STATIC
  ${_engine_src}/tokenizer/qwen_bpe.cpp ${_engine_src}/tokenizer/unicode.cpp ${_engine_src}/tokenizer/prompt_tokens.cpp
  ${_engine_src}/pipeline/samplers/initial_noise.cpp ${_engine_src}/pipeline/rng_torch_compat.cpp
  ${_engine_src}/pipeline/samplers/schedules.cpp ${_engine_src}/pipeline/png_writer.cpp ${_engine_src}/util/json.cpp
  ${_engine_src}/util/regular_file.cpp ${_engine_src}/util/replaceable_file.cpp)
target_include_directories(mossinfer_hip_host PUBLIC ${_engine_src} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_options(mossinfer_hip_host PRIVATE -ffp-contract=off -fno-fast-math)
add_executable(moss-infer-hip tools/moss_infer_hip.cpp)
target_compile_options(moss-infer-hip PRIVATE ${MI_HIP_WARN})
target_link_libraries(moss-infer-hip PRIVATE mossinfer_hip mossinfer_hip_host)

# Package (AMD builds only; the NVIDIA build is for checking): the program and the license files, Linux x86-64. No HIP
# runtime ships: ROCm 7 provides libamdhip64.so.7, found in /opt/rocm/lib (the installed program's RUNPATH) or on the
# loader path.
if(MI_HIP_AMD)
  set_target_properties(moss-infer-hip PROPERTIES INSTALL_RPATH /opt/rocm/lib)
  # The program itself then needs no libstdc++ of the build's age; ROCm's libamdhip64 still loads the system's.
  option(MOSSINFER_STATIC_LIBSTDCXX "Link libstdc++ and libgcc statically into moss-infer-hip (release packages)" OFF)
  if(MOSSINFER_STATIC_LIBSTDCXX)
    target_link_options(moss-infer-hip PRIVATE -static-libstdc++ -static-libgcc)
  endif()
  install(TARGETS moss-infer-hip RUNTIME DESTINATION bin)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/../NOTICE
                ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md ${CMAKE_CURRENT_SOURCE_DIR}/../docs/hip.md
          DESTINATION share/doc/mossinfer)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mi_arch)
  if(_mi_arch STREQUAL "amd64")
    set(_mi_arch x86_64)
  endif()
  set(CPACK_PACKAGE_NAME mossinfer-hip)
  set(CPACK_PACKAGE_VERSION ${MOSSINFER_VERSION})
  # The release name for the release's three architectures; a build of fewer names them, so it cannot pass for it.
  set(_mi_hip_archs ${HIP_TARGET})
  list(SORT _mi_hip_archs)
  set(CPACK_PACKAGE_FILE_NAME "mossinfer-${MOSSINFER_VERSION}-linux-hip-${_mi_arch}")
  if(NOT _mi_hip_archs STREQUAL "gfx1100;gfx1200;gfx1201")
    list(JOIN _mi_hip_archs "-" _mi_hip_suffix)
    string(APPEND CPACK_PACKAGE_FILE_NAME "-${_mi_hip_suffix}")
  endif()
  set(CPACK_GENERATOR TGZ)
  include(CPack)
endif()

# The checks and tools; a release build turns them off.
option(MOSSINFER_HIP_TOOLS "Build the HIP backend's checks and benchmarks" ON)
if(NOT MOSSINFER_HIP_TOOLS)
  return()
endif()

add_executable(hip_check tools/hip_check.cpp)
target_compile_options(hip_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_check PRIVATE mossinfer_hip)

# dit_kernels.h's ops against host replays; the replays round every fp32 operation, as the kernels do.
add_executable(hip_dit_kernel_check tools/hip_dit_kernel_check.cpp)
target_compile_options(hip_dit_kernel_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_dit_kernel_check PRIVATE mossinfer_hip)

# The engine's JSON reader, for hip_dit_run's case files: built as the engine builds it, without this project's warnings.
add_library(mi_hip_engine_json OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../src/util/json.cpp)
target_include_directories(mi_hip_engine_json SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# The DiT's sampling loop from a golden case; CFG and Euler on the host round like the engine's kernels.
add_executable(hip_dit_run tools/hip_dit_run.cpp)
target_compile_options(hip_dit_run PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_dit_run PRIVATE mossinfer_hip mi_hip_engine_json)

# te_kernels.h's ops against host replays of their op order, and the gated delta rule against an fp64 recurrence.
add_executable(hip_te_kernel_check tools/hip_te_kernel_check.cpp)
target_compile_options(hip_te_kernel_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_te_kernel_check PRIVATE mossinfer_hip)

# The text encoder against the official pipeline's golden hidden states; it reads cases.json and the torch noise floor
# with the engine's JSON reader (mi_hip_engine_json above).
add_executable(hip_te_check tools/hip_te_check.cpp)
target_compile_options(hip_te_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_te_check PRIVATE mossinfer_hip mi_hip_engine_json)

# The VAE decoder's kernels against host replays, then golden latents against the reference pipeline's PNGs (zlib).
find_package(ZLIB REQUIRED)
add_executable(hip_vae_check tools/hip_vae_check.cpp)
target_compile_options(hip_vae_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_vae_check PRIVATE mossinfer_hip ZLIB::ZLIB)
