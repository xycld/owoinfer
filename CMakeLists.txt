cmake_minimum_required(VERSION 3.28)
project(MossInfer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 120)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MOSSINFER_WERROR "Treat warnings as errors" ON)
option(MOSSINFER_ASAN "Build host code with AddressSanitizer" OFF)
option(MOSSINFER_TESTS "Build unit tests" ON)

find_package(CUDAToolkit 12.4 REQUIRED)

if(MSVC)
  set(MOSSINFER_WARN /W4 /permissive- /utf-8)
  if(MOSSINFER_WERROR)
    list(APPEND MOSSINFER_WARN /WX)
  endif()
else()
  set(MOSSINFER_WARN -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
  if(MOSSINFER_WERROR)
    list(APPEND MOSSINFER_WARN -Werror)
  endif()
endif()

# Every module under src/<module>/ is compiled into one static core library. Modules add files
# only; no per-module CMake edits are needed.
file(GLOB_RECURSE MOSSINFER_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/src/*.cu)
list(FILTER MOSSINFER_SOURCES EXCLUDE REGEX ".*/src/tools/.*")

# Parallel development: MOSSINFER_ONLY="core;util;kernels/gdn" builds only sources under those src/
# prefixes, and MOSSINFER_TESTS_ONLY="test_gdn;test_json" only tests/unit files with those name
# prefixes, so one module's work in progress cannot break another module's build. Empty = everything.
set(MOSSINFER_ONLY "" CACHE STRING "src/ path prefixes to build (empty = all)")
set(MOSSINFER_TESTS_ONLY "" CACHE STRING "tests/unit file-name prefixes to build (empty = all)")
function(mossinfer_filter out_var files base prefixes)
  if(NOT prefixes)
    set(${out_var} ${files} PARENT_SCOPE)
    return()
  endif()
  set(kept "")
  foreach(f IN LISTS files)
    file(RELATIVE_PATH rel ${base} ${f})
    foreach(p IN LISTS prefixes)
      string(FIND "${rel}" "${p}" pos)
      if(pos EQUAL 0)
        list(APPEND kept ${f})
        break()
      endif()
    endforeach()
  endforeach()
  set(${out_var} ${kept} PARENT_SCOPE)
endfunction()
mossinfer_filter(MOSSINFER_SOURCES "${MOSSINFER_SOURCES}" ${CMAKE_SOURCE_DIR}/src "${MOSSINFER_ONLY}")
add_library(mossinfer_core STATIC ${MOSSINFER_SOURCES})
target_include_directories(mossinfer_core PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_compile_options(mossinfer_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MOSSINFER_WARN}>)
# CUDA: host warnings via -Xcompiler (kept below -Werror level: CUTLASS/CuTe headers are not -Wshadow clean),
# device code with line info for profilers.
target_compile_options(mossinfer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xcompiler=-Wall>)
# Vendored CUTLASS/CuTe headers (BSD-3, see third_party/cutlass/VENDORED.md), CUDA translation units only.
target_include_directories(mossinfer_core SYSTEM PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_SOURCE_DIR}/third_party/cutlass/include ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include>)
target_link_libraries(mossinfer_core PUBLIC CUDA::cudart CUDA::cublasLt)
set_target_properties(mossinfer_core PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF)

# torch_cpu_randn is bit-exact only if the compiler neither contracts mul+add into FMAs nor applies
# fast-math. Source options come after CMAKE_CXX_FLAGS on the command line, so these win.
if(MSVC)
  set_source_files_properties(${CMAKE_SOURCE_DIR}/src/pipeline/rng_torch_compat.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
else()
  set_source_files_properties(${CMAKE_SOURCE_DIR}/src/pipeline/rng_torch_compat.cpp
                              PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
endif()

if(MOSSINFER_ASAN AND NOT MSVC)
  target_compile_options(mossinfer_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined -fno-omit-frame-pointer>)
  target_link_options(mossinfer_core PUBLIC -fsanitize=address,undefined)
endif()

# Module-owned build fragments (e.g. vendored kernels needing their own include paths). Each
# fragment must only touch sources of its own module and tolerate those sources being filtered out.
file(GLOB MOSSINFER_CMAKE_FRAGMENTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cmake/modules/*.cmake)
foreach(fragment IN LISTS MOSSINFER_CMAKE_FRAGMENTS)
  include(${fragment})
endforeach()

# Command-line tool: src/tools/cli/*.cpp
file(GLOB MOSSINFER_CLI_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/tools/cli/*.cpp)
# The CLI links against every module, so it is only built in full (unfiltered) builds.
if(MOSSINFER_CLI_SOURCES AND NOT MOSSINFER_ONLY)
  add_executable(moss-infer ${MOSSINFER_CLI_SOURCES})
  target_link_libraries(moss-infer PRIVATE mossinfer_core)
  target_compile_options(moss-infer PRIVATE ${MOSSINFER_WARN})
endif()

if(MOSSINFER_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  file(GLOB MOSSINFER_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/unit/*.cpp ${CMAKE_SOURCE_DIR}/tests/unit/*.cu)
  mossinfer_filter(MOSSINFER_TEST_SOURCES "${MOSSINFER_TEST_SOURCES}" ${CMAKE_SOURCE_DIR}/tests/unit "${MOSSINFER_TESTS_ONLY}")
  add_executable(mossinfer_tests ${MOSSINFER_TEST_SOURCES})
  target_link_libraries(mossinfer_tests PRIVATE mossinfer_core GTest::gtest_main)
  target_compile_options(mossinfer_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MOSSINFER_WARN}>
                                                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
  target_include_directories(mossinfer_tests SYSTEM PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_SOURCE_DIR}/third_party/cutlass/include ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include>)
  target_compile_definitions(mossinfer_tests PRIVATE
    MOSSINFER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  include(GoogleTest)
  gtest_discover_tests(mossinfer_tests DISCOVERY_MODE PRE_TEST)
endif()
