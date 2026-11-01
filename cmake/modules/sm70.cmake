# Volta tensor-core kernels live outside the sm_80+ core architecture list, like sm75 and nvfp4. CUDA 12 only.
set(_mi_sm70_sources ${MOSSINFER_SOURCES})
list(FILTER _mi_sm70_sources INCLUDE REGEX ".*/src/kernels/sm70/[^/]*\\.cu$")
if(_mi_sm70_sources)
  get_target_property(_mi_core_sources mossinfer_core SOURCES)
  list(REMOVE_ITEM _mi_core_sources ${_mi_sm70_sources})
  set_property(TARGET mossinfer_core PROPERTY SOURCES ${_mi_core_sources})
  unset(_mi_core_sources)
  if(MOSSINFER_SM70)
    add_library(mossinfer_sm70 OBJECT ${_mi_sm70_sources})
    set_target_properties(mossinfer_sm70 PROPERTIES CUDA_ARCHITECTURES 70-real POSITION_INDEPENDENT_CODE ON
                          CUDA_SEPARABLE_COMPILATION OFF CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_include_directories(mossinfer_sm70 PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(mossinfer_sm70 SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
                               ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include)
    target_compile_options(mossinfer_sm70 PRIVATE --expt-relaxed-constexpr -lineinfo ${MOSSINFER_CUDA_HOST_WARN})
    target_sources(mossinfer_core PRIVATE $<TARGET_OBJECTS:mossinfer_sm70>)
    set_property(SOURCE ${CMAKE_SOURCE_DIR}/src/backend/probe.cpp
                        ${CMAKE_SOURCE_DIR}/src/kernels/sm70/sm70.cpp
                        ${CMAKE_SOURCE_DIR}/src/kernels/attention/flash_attention.cu
                        ${CMAKE_SOURCE_DIR}/src/kernels/basic/gemm.cpp
                        ${CMAKE_SOURCE_DIR}/src/kernels/gdn/chunk_rule.cu
                        ${CMAKE_SOURCE_DIR}/src/kernels/quant/int8_gemm.cu
                        ${CMAKE_SOURCE_DIR}/src/kernels/quant/int8_gemm_residual.cu
                        ${CMAKE_SOURCE_DIR}/src/kernels/quant/int8_gemm_situ.cu
                 APPEND PROPERTY COMPILE_DEFINITIONS MOSSINFER_HAVE_SM70=1)
    if(EXISTS ${CMAKE_SOURCE_DIR}/tools/sm75_check.cu)
      add_executable(sm70_check ${CMAKE_SOURCE_DIR}/tools/sm75_check.cu)
      set_target_properties(sm70_check PROPERTIES CUDA_ARCHITECTURES 70-real)
      target_include_directories(sm70_check PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
      target_compile_options(sm70_check PRIVATE --expt-relaxed-constexpr -lineinfo ${MOSSINFER_CUDA_HOST_WARN})
      target_link_libraries(sm70_check PRIVATE mossinfer_core)
    endif()
    if(EXISTS ${CMAKE_SOURCE_DIR}/tools/sm70_bench.cu)
      add_executable(sm70_bench ${CMAKE_SOURCE_DIR}/tools/sm70_bench.cu)
      set_target_properties(sm70_bench PROPERTIES CUDA_ARCHITECTURES 70-real)
      target_link_libraries(sm70_bench PRIVATE mossinfer_core)
      target_compile_options(sm70_bench PRIVATE --expt-relaxed-constexpr -lineinfo ${MOSSINFER_CUDA_HOST_WARN})
    endif()
    message(STATUS "SM70: Volta kernel family built; compute capability 7.0 accepted")
  endif()
endif()
unset(_mi_sm70_sources)
