# The Vulkan backend and moss-infer-vulkan, the standalone program for GPUs the CUDA engine does not support. Its own
# project, built without CUDA; the engine's build does not include it. See README.md.
cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/version.cmake)  # MOSSINFER_VERSION, from the C header
project(mossinfer_vulkan VERSION ${MOSSINFER_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)  # the CPU reference is unusably slow unoptimized
endif()

# The Vulkan headers and glslc, from $VULKAN_SDK first. No loader library: vk_api.cpp opens it at run time.
find_path(MI_VULKAN_INCLUDE_DIR vulkan/vulkan_core.h
          HINTS "$ENV{VULKAN_SDK}/include" "$ENV{VULKAN_SDK}/Include" REQUIRED)
file(STRINGS ${MI_VULKAN_INCLUDE_DIR}/vulkan/vulkan_core.h _mi_bf16 REGEX "VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME")
if(NOT _mi_bf16)
  message(FATAL_ERROR "${MI_VULKAN_INCLUDE_DIR}/vulkan/vulkan_core.h has no VK_KHR_shader_bfloat16: use a newer Vulkan SDK")
endif()
find_program(GLSL_COMPILER NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" REQUIRED)
add_library(mi_vulkan_headers INTERFACE)
target_include_directories(mi_vulkan_headers SYSTEM INTERFACE ${MI_VULKAN_INCLUDE_DIR})

# Shaders become SPIR-V word lists compiled into the binary, so nothing beyond it ships.
set(_spirv_dir ${CMAKE_CURRENT_BINARY_DIR}/spirv)
file(MAKE_DIRECTORY ${_spirv_dir})
set(_spirv_files "")
# The cooperative-matrix shader needs Vulkan 1.3; the rest must also load on Vulkan 1.2 devices, which reject newer SPIR-V.
# _mi_add_shader_as(source output target_env [-DNAME ...]) builds shaders/<source>.comp into <output>.spv.inc.
function(_mi_add_shader_as shader output target_env)
  set(_source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp)
  set(_output ${_spirv_dir}/${output}.spv.inc)
  add_custom_command(
    OUTPUT ${_output}
    COMMAND ${GLSL_COMPILER} --target-env=${target_env} -fshader-stage=comp -O -mfmt=num ${ARGN}
            -I${CMAKE_CURRENT_SOURCE_DIR}/shaders ${_source} -o ${_output}
    DEPENDS ${_source} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl
    COMMENT "glslc ${shader}.comp -> ${output}")
  set(_spirv_files ${_spirv_files} ${_output} PARENT_SCOPE)
endfunction()
function(_mi_add_shader shader target_env)
  _mi_add_shader_as(${shader} ${shader} ${target_env})
  set(_spirv_files ${_spirv_files} PARENT_SCOPE)
endfunction()
_mi_add_shader(gemm_coop vulkan1.3)
_mi_add_shader(gemm_bf16_coop vulkan1.3)
_mi_add_shader(attention_coop vulkan1.3)
_mi_add_shader_as(gemm_bf16_coop gemm_f16_coop vulkan1.3 -DF16)
_mi_add_shader_as(attention_coop attention_f16_coop vulkan1.3 -DF16)
foreach(shader IN ITEMS gemm_plain norm_modulate qk_norm_rope attention_plain elementwise row_scale)
  _mi_add_shader(${shader} vulkan1.2)
endforeach()

add_library(mossinfer_vulkan STATIC vk_api.cpp runtime.cpp gemm.cpp block.cpp dit.cpp safetensors.cpp ${_spirv_files})
target_include_directories(mossinfer_vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${_spirv_dir})
# Hidden symbols: vk_api.cpp defines data objects named like the Vulkan entry points, which a shared library must not
# export (they would interpose with a loader in the same process).
set_target_properties(mossinfer_vulkan PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
# Headers only: vk_api.cpp opens the loader at run time, so nothing links against it.
find_package(Threads REQUIRED)
target_link_libraries(mossinfer_vulkan PUBLIC mi_vulkan_headers Threads::Threads ${CMAKE_DL_LIBS})

# The Qwen3.5 text encoder (text_encoder.h): its own kernels on top of the DiT block's GEMMs.
set(_te_spirv_files "")
foreach(shader IN ITEMS te_rms_norm te_conv_silu te_gates te_gdn_qk te_gdn_chunk te_gdn_solve te_gdn_state te_attention
                        te_gate_mul)
  set(_source ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp)
  set(_output ${_spirv_dir}/${shader}.spv.inc)
  add_custom_command(
    OUTPUT ${_output}
    COMMAND ${GLSL_COMPILER} --target-env=vulkan1.2 -fshader-stage=comp -O -mfmt=num
            -I${CMAKE_CURRENT_SOURCE_DIR}/shaders ${_source} -o ${_output}
    DEPENDS ${_source} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/te_common.glsl
    COMMENT "glslc ${shader}.comp")
  list(APPEND _te_spirv_files ${_output})
endforeach()
target_sources(mossinfer_vulkan PRIVATE te_kernels.cpp text_encoder.cpp ${_te_spirv_files})

# The FLUX.2 VAE decoder (vae.h): its cooperative-matrix kernel needs Vulkan 1.3, the rest loads on Vulkan 1.2.
set(_spirv_block ${_spirv_files})
_mi_add_shader(vae_conv_coop vulkan1.3)
foreach(shader IN ITEMS vae_conv_plain vae_conv_narrow vae_gn_stats vae_gn_finalize vae_gn_apply vae_softmax vae_transpose
                        vae_latent_unpack vae_rgb8)
  _mi_add_shader(${shader} vulkan1.2)
endforeach()
set(_spirv_vae ${_spirv_files})
list(REMOVE_ITEM _spirv_vae ${_spirv_block})
target_sources(mossinfer_vulkan PRIVATE vae.cpp generate.cpp ${_spirv_vae})
# generate.cpp's CFG and Euler step round like the engine's kernels: no contraction.
set_source_files_properties(generate.cpp PROPERTIES COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/fp:precise,-ffp-contract=off>")

# moss-infer-vulkan, the standalone program: the backend plus the engine's host sources it shares with moss-infer (the
# tokenizer, the seed's initial noise, the schedule, the PNG writer), compiled here so that nothing needs CUDA.
set(_engine_src ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_library(mossinfer_vulkan_host STATIC
  ${_engine_src}/tokenizer/qwen_bpe.cpp ${_engine_src}/tokenizer/unicode.cpp ${_engine_src}/tokenizer/prompt_tokens.cpp
  ${_engine_src}/pipeline/samplers/initial_noise.cpp ${_engine_src}/pipeline/rng_torch_compat.cpp
  ${_engine_src}/pipeline/samplers/schedules.cpp ${_engine_src}/pipeline/png_writer.cpp ${_engine_src}/util/json.cpp
  ${_engine_src}/util/regular_file.cpp ${_engine_src}/util/replaceable_file.cpp)
target_include_directories(mossinfer_vulkan_host PUBLIC ${_engine_src} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_executable(moss-infer-vulkan tools/moss_infer_vulkan.cpp)
target_link_libraries(moss-infer-vulkan PRIVATE mossinfer_vulkan mossinfer_vulkan_host)
if(MSVC)
  target_compile_options(mossinfer_vulkan_host PRIVATE /utf-8)
  target_compile_options(moss-infer-vulkan PRIVATE /utf-8)
endif()
# Release packages run on any glibc distribution of the build's age or newer, without its libstdc++.
option(MOSSINFER_STATIC_LIBSTDCXX "Link libstdc++ and libgcc statically into moss-infer-vulkan (Linux packages)" OFF)
if(MOSSINFER_STATIC_LIBSTDCXX AND NOT MSVC)
  target_link_options(moss-infer-vulkan PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Package: the program and the license files. No Vulkan loader ships: the GPU driver provides it.
install(TARGETS moss-infer-vulkan RUNTIME DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/../NOTICE
              ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md ${CMAKE_CURRENT_SOURCE_DIR}/../docs/vulkan.md
        DESTINATION share/doc/mossinfer)
if(WIN32)
  set(_mi_platform windows)
elseif(APPLE)
  set(_mi_platform macos)
else()
  set(_mi_platform linux)
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mi_arch)
if(_mi_arch STREQUAL "amd64")
  set(_mi_arch x86_64)
endif()
set(CPACK_PACKAGE_NAME mossinfer-vulkan)
set(CPACK_PACKAGE_VERSION ${MOSSINFER_VERSION})
set(CPACK_PACKAGE_FILE_NAME "mossinfer-${MOSSINFER_VERSION}-${_mi_platform}-vulkan-${_mi_arch}")
if(WIN32)
  set(CPACK_GENERATOR ZIP)
else()
  set(CPACK_GENERATOR TGZ)
endif()
include(CPack)

# The checks and benchmarks; a release build turns them off.
option(MOSSINFER_VULKAN_TOOLS "Build the Vulkan backend's checks and benchmarks" ON)
if(NOT MOSSINFER_VULKAN_TOOLS)
  return()
endif()

add_executable(vulkan_check tools/vulkan_check.cpp)
target_link_libraries(vulkan_check PRIVATE mossinfer_vulkan)

# The host replays the block is checked against must round where the kernels do: no contraction.
add_executable(vulkan_block_check tools/vulkan_block_check.cpp)
target_compile_options(vulkan_block_check PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/fp:precise,-ffp-contract=off>)
target_link_libraries(vulkan_block_check PRIVATE mossinfer_vulkan)

# One real block against the engine's captured output.
add_executable(vulkan_block_real tools/vulkan_block_real.cpp)
target_link_libraries(vulkan_block_real PRIVATE mossinfer_vulkan)

# The DiT's sampling loop from a golden case; CFG and Euler on the host round like the engine's kernels: no contraction.
add_executable(vulkan_dit_run tools/vulkan_dit_run.cpp)
target_compile_options(vulkan_dit_run PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/fp:precise,-ffp-contract=off>)
target_link_libraries(vulkan_dit_run PRIVATE mossinfer_vulkan)

# The text encoder's kernels against host replays of their op order; the replays must round where the kernels do.
add_executable(vulkan_te_kernel_check tools/vulkan_te_kernel_check.cpp)
target_compile_options(vulkan_te_kernel_check PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/fp:precise,-ffp-contract=off>)
target_link_libraries(vulkan_te_kernel_check PRIVATE mossinfer_vulkan)

# The whole text encoder against the official pipeline's golden hidden states.
add_executable(vulkan_te_check tools/vulkan_te_check.cpp)
target_link_libraries(vulkan_te_check PRIVATE mossinfer_vulkan)

# Kernels against host replays, then the real decoder against the reference pipeline's PNG (zlib reads and writes PNGs).
find_package(ZLIB REQUIRED)
add_executable(vulkan_vae_check tools/vulkan_vae_check.cpp)
target_compile_options(vulkan_vae_check PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/fp:precise,-ffp-contract=off>)
target_link_libraries(vulkan_vae_check PRIVATE mossinfer_vulkan ZLIB::ZLIB)
