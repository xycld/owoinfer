# Turing (sm_75) kernels: the engine's tensor-core work needs bf16 MMA, cp.async, TF32 and up to 99 KiB of shared
# memory, none of which exist on compute capability 7.x. src/kernels/sm75/ holds the fp16 fallbacks, built as SASS for
# sm_75 alone in their own OBJECT library (like cmake/modules/nvfp4.cmake does for sm_120a), so the rest of
# mossinfer_core keeps its sm_80/89/120 architecture list and none of the sm_80-only translation units see --arch=sm_75.
#
# The kernel family is built in every build: the host code of the sm_80 translation units references it unconditionally
# (and only executes it on a 7.x device). MOSSINFER_SM75 (declared in CMakeLists.txt) controls what makes 7.x usable:
#   - the probe accepts compute capability 7.5 (MOSSINFER_HAVE_SM75 on probe.cpp, and 75-real in its architecture list,
#     cmake/modules/backend.cmake),
#   - the portable SIMT translation units are also compiled for sm_75,
#   - flash_attention.cu / gemm.cpp / chunk_rule.cu dispatch into the sm_75 kernels,
#   - the sm75_check bring-up tool is built.
# Tolerates the sources being filtered out by MOSSINFER_ONLY.
set(_mi_sm75_sources ${MOSSINFER_SOURCES})
list(FILTER _mi_sm75_sources INCLUDE REGEX ".*/src/kernels/sm75/[^/]*\\.cu$")
if(_mi_sm75_sources)
  get_target_property(_mi_core_sources mossinfer_core SOURCES)
  list(REMOVE_ITEM _mi_core_sources ${_mi_sm75_sources})
  set_property(TARGET mossinfer_core PROPERTY SOURCES ${_mi_core_sources})
  unset(_mi_core_sources)

  add_library(mossinfer_sm75 OBJECT ${_mi_sm75_sources})
  set_target_properties(mossinfer_sm75 PROPERTIES
    CUDA_ARCHITECTURES 75-real
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF)
  target_include_directories(mossinfer_sm75 PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
  target_include_directories(mossinfer_sm75 SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include)
  target_compile_options(mossinfer_sm75 PRIVATE --expt-relaxed-constexpr -lineinfo ${MOSSINFER_CUDA_HOST_WARN})
  target_sources(mossinfer_core PRIVATE $<TARGET_OBJECTS:mossinfer_sm75>)
  message(STATUS "SM75: Turing kernel family built (${_mi_sm75_sources})")
endif()

if(MOSSINFER_SM75 AND _mi_sm75_sources)
  # Portable kernels: pure SIMT translation units (no cp.async, no tensor-core asm, no CUTLASS), which run unchanged on
  # Turing but were only compiled for the core's sm_80+ list. They are built for sm_75 too, in their own OBJECT library
  # (the core keeps its list, so the mixed-precision kernels around them never see --arch=sm_75).
  set(_mi_portable_sources
      src/kernels/basic/elementwise.cu
      src/kernels/basic/gram.cu
      src/kernels/basic/norm.cu
      src/kernels/basic/rope.cu
      src/kernels/fused/qk_norm_rope.cu
      src/kernels/gdn/chunk_rule.cu
      src/kernels/gdn/conv_gates.cu
      src/kernels/lora/lora_kernels.cu
      src/kernels/quant/rotate_quantize.cu
      src/kernels/quant/w4a8_decode.cu
      src/kernels/quant4/convrot_quantize.cu
      src/kernels/vae/elementwise.cu
      src/kernels/vae/encoder_elementwise.cu
      src/kernels/vae/group_norm.cu
      src/models/flux2_vae/torch_group_norm.cu
      src/models/latent_preview/latent_preview.cu
      src/models/moss_dit/dit_kernels.cu
      src/pipeline/accel/accel_kernels.cu
      src/pipeline/samplers/sampler_kernels.cu
      src/pipeline/samplers/torch_cuda_randn.cu)
  set(_mi_portable_present "")
  foreach(rel IN LISTS _mi_portable_sources)
    if(${CMAKE_SOURCE_DIR}/${rel} IN_LIST MOSSINFER_SOURCES)
      list(APPEND _mi_portable_present ${CMAKE_SOURCE_DIR}/${rel})
    endif()
  endforeach()
  if(_mi_portable_present)
    get_target_property(_mi_core_sources mossinfer_core SOURCES)
    list(REMOVE_ITEM _mi_core_sources ${_mi_portable_present})
    set_property(TARGET mossinfer_core PROPERTY SOURCES ${_mi_core_sources})
    unset(_mi_core_sources)
    set(_mi_portable_archs 75 ${CMAKE_CUDA_ARCHITECTURES})
    list(REMOVE_DUPLICATES _mi_portable_archs)
    add_library(mossinfer_sm75_portable OBJECT ${_mi_portable_present})
    set_target_properties(mossinfer_sm75_portable PROPERTIES
      CUDA_ARCHITECTURES "${_mi_portable_archs}"
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION OFF)
    target_include_directories(mossinfer_sm75_portable PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_compile_options(mossinfer_sm75_portable PRIVATE --expt-relaxed-constexpr -lineinfo
                           ${MOSSINFER_CUDA_HOST_WARN})
    target_sources(mossinfer_core PRIVATE $<TARGET_OBJECTS:mossinfer_sm75_portable>)
    message(STATUS "SM75: portable kernels for ${_mi_portable_archs}")
    unset(_mi_portable_archs)
  endif()
  unset(_mi_portable_present)
  unset(_mi_portable_sources)

  # src/backend/probe.h: the minimum compute capability and the architecture list this build accepts. Defined on the
  # sources that read it rather than on the whole target, so a build without MOSSINFER_SM75 keeps its old objects and
  # adding the option only recompiles the dispatchers.
  set_property(SOURCE ${CMAKE_SOURCE_DIR}/src/backend/probe.cpp
                      ${CMAKE_SOURCE_DIR}/src/kernels/attention/flash_attention.cu
                      ${CMAKE_SOURCE_DIR}/src/kernels/basic/gemm.cpp
                      ${CMAKE_SOURCE_DIR}/src/kernels/gdn/chunk_rule.cu
               APPEND PROPERTY COMPILE_DEFINITIONS MOSSINFER_HAVE_SM75=1)
  message(STATUS "SM75: compute capability 7.5 accepted")

  # Bring-up harness (not installed): the engine's GPU tests compile their reference kernels for the build's CUDA
  # architectures, so a Turing device cannot run them; this tool builds its own host reference.
  if(EXISTS ${CMAKE_SOURCE_DIR}/tools/sm75_check.cu)
    add_executable(sm75_check ${CMAKE_SOURCE_DIR}/tools/sm75_check.cu)
    set_target_properties(sm75_check PROPERTIES CUDA_ARCHITECTURES 75)
    target_include_directories(sm75_check PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(sm75_check PRIVATE mossinfer_core)
    target_compile_options(sm75_check PRIVATE --expt-relaxed-constexpr -lineinfo ${MOSSINFER_CUDA_HOST_WARN})
  endif()
endif()
unset(_mi_sm75_sources)
