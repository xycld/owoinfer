# Fuzz targets (opt-in, host code only): configure with clang, e.g. through the main build's -DMOSSINFER_FUZZ=ON
# (cmake/modules/pipeline_image.cmake), or directly:
#   cmake -S tests/fuzz -B build-fuzz -G Ninja -DCMAKE_CXX_COMPILER=clang++ && cmake --build build-fuzz
cmake_minimum_required(VERSION 3.28)
project(mossinfer_fuzz LANGUAGES CXX)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "the fuzz targets need clang (libFuzzer): -DCMAKE_CXX_COMPILER=clang++")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(_mi_root ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
add_executable(fuzz_pipeline_image fuzz_pipeline_image.cpp ${_mi_root}/src/pipeline/pipeline_image.cpp)
target_include_directories(fuzz_pipeline_image PRIVATE ${_mi_root}/src)
target_include_directories(fuzz_pipeline_image SYSTEM PRIVATE ${_mi_root}/third_party/stb)
target_compile_options(fuzz_pipeline_image PRIVATE -g -O1 -fsanitize=fuzzer,address,undefined -ffp-contract=off)
target_link_options(fuzz_pipeline_image PRIVATE -fsanitize=fuzzer,address,undefined)
