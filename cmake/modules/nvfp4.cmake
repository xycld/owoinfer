# NVFP4 GEMMs (src/kernels/nvfp4): the block-scaled FP4 MMA exists only as sm_120a/sm_120f SASS (plain sm_120 fails
# in ptxas, and a compute_120a PTX cannot JIT to any other GPU), so src/kernels/nvfp4/gemm_sm120a_*.cu leave
# mossinfer_core's architecture list and build in an OBJECT library for 120a-real alone (SASS, no PTX), whose objects
# go into mossinfer_core. Enabled when MOSSINFER_NVFP4 is ON (the default with CUDA >= 12.8), the compiler is
# >= 12.8 and mossinfer_core builds for sm_120; otherwise those sources are left out and nvfp4::available() is false
# (src/kernels/nvfp4/nvfp4.cpp, which also refuses any GPU but cc 12.0 at run time). Every other target and source
# keeps its flags. Tolerates the sources being filtered out by MOSSINFER_ONLY.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(_mi_nvfp4_default ON)
else()
  set(_mi_nvfp4_default OFF)
endif()
option(MOSSINFER_NVFP4 "Build the sm_120a NVFP4 GEMMs (needs CUDA >= 12.8 and sm_120 among the CUDA architectures)"
       ${_mi_nvfp4_default})
unset(_mi_nvfp4_default)

set(_mi_nvfp4_sm120a ${MOSSINFER_SOURCES})
list(FILTER _mi_nvfp4_sm120a INCLUDE REGEX ".*/src/kernels/nvfp4/gemm_sm120a_[^/]*\\.cu$")
if(_mi_nvfp4_sm120a)
  get_target_property(_mi_core_sources mossinfer_core SOURCES)
  list(REMOVE_ITEM _mi_core_sources ${_mi_nvfp4_sm120a})
  set_property(TARGET mossinfer_core PROPERTY SOURCES ${_mi_core_sources})
  unset(_mi_core_sources)

  get_target_property(_mi_core_archs mossinfer_core CUDA_ARCHITECTURES)
  list(FILTER _mi_core_archs INCLUDE REGEX "^(120|all|all-major)")
  if(NOT MOSSINFER_NVFP4)
    message(STATUS "NVFP4: off (MOSSINFER_NVFP4=OFF)")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(STATUS "NVFP4: off (CUDA ${CMAKE_CUDA_COMPILER_VERSION} < 12.8)")
  elseif(NOT _mi_core_archs)
    message(STATUS "NVFP4: off (sm_120 is not among the CUDA architectures)")
  else()
    message(STATUS "NVFP4: sm_120a GEMMs on")
    add_library(mossinfer_nvfp4_sm120a OBJECT ${_mi_nvfp4_sm120a})
    # Hidden like the rest of libmossinfer (packaging.cmake); internal symbols in a static core either way.
    set_target_properties(mossinfer_nvfp4_sm120a PROPERTIES CUDA_ARCHITECTURES 120a-real POSITION_INDEPENDENT_CODE ON
                          CUDA_SEPARABLE_COMPILATION OFF CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_include_directories(mossinfer_nvfp4_sm120a PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_include_directories(mossinfer_nvfp4_sm120a SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
                               ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include)
    # 20012: CUTLASS's sm100_static_tile_scheduler.hpp annotates a defaulted constructor __host__ __device__.
    target_compile_options(mossinfer_nvfp4_sm120a PRIVATE --expt-relaxed-constexpr -lineinfo -diag-suppress=20012
                           ${MOSSINFER_CUDA_HOST_WARN})
    target_sources(mossinfer_core PRIVATE $<TARGET_OBJECTS:mossinfer_nvfp4_sm120a>)
    set_property(SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/nvfp4/nvfp4.cpp APPEND PROPERTY COMPILE_DEFINITIONS
                 MOSSINFER_HAVE_NVFP4_SM120A)
    # Each of these CUTLASS 3.x / CuTe instantiations needs ~5.5 GB of RAM to compile: at most two at a time under
    # Ninja (16 GB CI runners); other generators ignore job pools.
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS mossinfer_nvfp4=2)
    set_property(TARGET mossinfer_nvfp4_sm120a PROPERTY JOB_POOL_COMPILE mossinfer_nvfp4)
  endif()
  unset(_mi_core_archs)
endif()
unset(_mi_nvfp4_sm120a)

# bench/nvfp4_gemm_probe.cu: the N0/S1 rate probe (NVFP4 vs INT8 vs bf16 GEMMs at the DiT shapes, with NVML
# telemetry). Local research material, never built by default, installed or shipped. Links NVML directly; cuBLASLt
# comes through the engine's run-time loader (src/backend/cublaslt_shim.cpp), like every cuBLASLt call.
option(MOSSINFER_BENCH_NVFP4 "Build bench/nvfp4_gemm_probe (local GEMM rate probe, not shipped)" OFF)
if(MOSSINFER_BENCH_NVFP4)
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/bench/nvfp4_gemm_probe.cu)
    message(FATAL_ERROR "MOSSINFER_BENCH_NVFP4: bench/nvfp4_gemm_probe.cu is local research material and not in "
                        "this checkout")
  endif()
  if(MOSSINFER_ONLY)
    message(FATAL_ERROR "MOSSINFER_BENCH_NVFP4 needs a full build (MOSSINFER_ONLY must be empty)")
  endif()
  add_executable(nvfp4_gemm_probe ${CMAKE_SOURCE_DIR}/bench/nvfp4_gemm_probe.cu)
  target_link_libraries(nvfp4_gemm_probe PRIVATE mossinfer_core CUDA::nvml)
  target_compile_options(nvfp4_gemm_probe PRIVATE --expt-relaxed-constexpr -lineinfo -Werror=all-warnings)
  if(MSVC)
    target_compile_options(nvfp4_gemm_probe PRIVATE ${MOSSINFER_CUDA_HOST_WARN})
  else()
    target_compile_options(nvfp4_gemm_probe PRIVATE -Xcompiler=-Wall,-Wextra,-Wshadow,-Werror,-ffp-contract=off)
  endif()
endif()
