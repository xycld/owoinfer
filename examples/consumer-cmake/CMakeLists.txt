# Out-of-tree consumer of an installed MossInfer package (API: docs/c-api.md). Needs no CUDA toolkit:
#   cmake -S examples/consumer-cmake -B build-consumer -DCMAKE_PREFIX_PATH=<mossinfer install or unzipped package>
#   cmake --build build-consumer && build-consumer/mossinfer_consumer
cmake_minimum_required(VERSION 3.21)
project(MossInferConsumer LANGUAGES C)

find_package(MossInfer 0.1 REQUIRED CONFIG)

add_executable(mossinfer_consumer main.c)
target_link_libraries(mossinfer_consumer PRIVATE MossInfer::mossinfer)
if(WIN32)
  # Windows finds DLLs next to the executable or on PATH, and mossinfer.dll needs the package's other DLLs too
  # (cuBLASLt, the MSVC runtime), which $<TARGET_RUNTIME_DLLS> does not list: copy all of bin/.
  file(GLOB _mi_runtime_dlls "${MossInfer_RUNTIME_DIR}/*.dll")
  add_custom_command(TARGET mossinfer_consumer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_mi_runtime_dlls} $<TARGET_FILE_DIR:mossinfer_consumer>
    COMMAND_EXPAND_LISTS)
endif()

enable_testing()
add_test(NAME consumer COMMAND mossinfer_consumer)
