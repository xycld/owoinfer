# src/backend: the probe reports which compute capabilities this build has kernel images for, so it is told the
# CUDA architecture list of mossinfer_core (commas: a semicolon would split the definition). This file only.
# No-op when MOSSINFER_ONLY filtered the probe out.
set(_mi_backend_probe ${CMAKE_SOURCE_DIR}/src/backend/probe.cpp)
if(_mi_backend_probe IN_LIST MOSSINFER_SOURCES)
  set_property(SOURCE ${_mi_backend_probe} APPEND PROPERTY COMPILE_DEFINITIONS
    "MOSSINFER_CUDA_ARCHITECTURES=\"$<JOIN:$<TARGET_PROPERTY:mossinfer_core,CUDA_ARCHITECTURES>,$<COMMA>>\"")
endif()
unset(_mi_backend_probe)
# src/backend/cuda_driver.cpp loads libcuda and NVML at run time (dlopen; libdl before glibc 2.34).
if(${CMAKE_SOURCE_DIR}/src/backend/cuda_driver.cpp IN_LIST MOSSINFER_SOURCES AND CMAKE_DL_LIBS)
  target_link_libraries(mossinfer_core PRIVATE ${CMAKE_DL_LIBS})
endif()

# src/backend/cublaslt_shim.cpp defines the cublasLt* functions the engine calls and forwards them to the copy loaded
# at run time (backend/cuda_libs.h). Exported, those definitions could capture libcublasLt's own internal calls, so at
# the end of configuration: no target links cuBLASLt, and no target that links the core exports its symbols.
function(_mi_check_cublaslt_shim_stays_internal)
  get_property(_targets DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(t IN LISTS _targets)
    get_target_property(_type ${t} TYPE)
    if(_type STREQUAL "UTILITY" OR _type STREQUAL "INTERFACE_LIBRARY")
      continue()
    endif()
    set(_links "")
    foreach(_prop IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_v ${t} ${_prop})
      if(_v)
        list(APPEND _links ${_v})
      endif()
    endforeach()
    if(_links MATCHES "CUDA::cublas|cublasLt")
      message(FATAL_ERROR "${t} links cuBLASLt (${_links}): the engine loads it at run time through "
                          "src/backend/cublaslt_shim.cpp; link nothing of cuBLASLt")
    endif()
    if(_links MATCHES "mossinfer_core" OR t STREQUAL "mossinfer_core")
      get_target_property(_exports ${t} ENABLE_EXPORTS)
      get_target_property(_opts ${t} LINK_OPTIONS)
      if(_exports OR _opts MATCHES "rdynamic|export-dynamic")
        message(FATAL_ERROR "${t} links the core and exports its symbols (ENABLE_EXPORTS / -rdynamic): the cuBLASLt "
                            "shim's definitions would become dynamic symbols (src/backend/cuda_libs.h)")
      endif()
    endif()
  endforeach()
endfunction()
cmake_language(DEFER CALL _mi_check_cublaslt_shim_stays_internal)

# Test-only stand-ins for cuBLASLt (tests/unit/test_backend_fake_cublaslt.c), one per loader branch: a usable copy,
# one older than the build's cuBLASLt, one of another major version, and one without an entry point. CudaLibs.* load
# them by path (their files reach the test as compile definitions).
if(MOSSINFER_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/unit/test_backend_fake_cublaslt.c)
  foreach(_kind IN ITEMS good old major nosym)
    add_library(fake_cublaslt_${_kind} SHARED ${CMAKE_SOURCE_DIR}/tests/unit/test_backend_fake_cublaslt.c)
    string(TOUPPER ${_kind} _kind_upper)
    target_compile_definitions(fake_cublaslt_${_kind} PRIVATE MI_FAKE_${_kind_upper})
    target_include_directories(fake_cublaslt_${_kind} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endforeach()
  unset(_kind)
  unset(_kind_upper)
  # On the one test source only, so the rest of mossinfer_tests is not rebuilt for them.
  set_property(SOURCE ${CMAKE_SOURCE_DIR}/tests/unit/test_backend_cuda_libs.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    "MI_FAKE_CUBLASLT_GOOD=\"$<TARGET_FILE:fake_cublaslt_good>\"" "MI_FAKE_CUBLASLT_OLD=\"$<TARGET_FILE:fake_cublaslt_old>\""
    "MI_FAKE_CUBLASLT_MAJOR=\"$<TARGET_FILE:fake_cublaslt_major>\""
    "MI_FAKE_CUBLASLT_NOSYM=\"$<TARGET_FILE:fake_cublaslt_nosym>\"")
  function(_mi_attach_fake_cublaslt)
    if(TARGET mossinfer_tests)
      add_dependencies(mossinfer_tests fake_cublaslt_good fake_cublaslt_old fake_cublaslt_major fake_cublaslt_nosym)
    endif()
  endfunction()
  cmake_language(DEFER CALL _mi_attach_fake_cublaslt)
endif()
