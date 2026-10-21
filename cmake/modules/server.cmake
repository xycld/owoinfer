# moss-server (tools/server, docs/server-protocol.md) and the C API example (examples/c). Both are clients of the
# public C API only: they link the shared libmossinfer when cmake/modules/packaging.cmake defines it (target
# `mossinfer`), else the static core. Like the CLI they need every module, so they are built in full (unfiltered)
# builds only. Nothing here exists with MOSSINFER_BUILD_API=OFF.
if(NOT MOSSINFER_BUILD_API)
  return()
endif()

# The frozen C struct layouts seen from a C compiler (the library checks the same table in C++). Header only.
if(MOSSINFER_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/c/test_abi_layout.c)
  enable_testing()
  add_executable(mossinfer_abi_layout_c ${CMAKE_SOURCE_DIR}/tests/c/test_abi_layout.c)
  set_target_properties(mossinfer_abi_layout_c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  target_include_directories(mossinfer_abi_layout_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(mossinfer_abi_layout_c PRIVATE ${MOSSINFER_WARN})
  add_test(NAME c_abi_layout COMMAND mossinfer_abi_layout_c)
  set_tests_properties(c_abi_layout PROPERTIES LABELS cpu)
endif()

if(NOT MOSSINFER_ONLY AND EXISTS ${CMAKE_SOURCE_DIR}/tools/server/main.cpp)
  find_package(Threads REQUIRED)
  add_executable(moss-server ${CMAKE_SOURCE_DIR}/tools/server/main.cpp)
  target_compile_options(moss-server PRIVATE ${MOSSINFER_WARN})
  if(TARGET mossinfer)
    # The shared library exports mi_* only, so the server brings its own copy of the JSON parser.
    target_sources(moss-server PRIVATE ${CMAKE_SOURCE_DIR}/src/util/json.cpp)
    target_include_directories(moss-server PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(moss-server PRIVATE mossinfer Threads::Threads)
  else()
    target_link_libraries(moss-server PRIVATE mossinfer_core Threads::Threads)
  endif()

  # C99 without extensions: also checks that mossinfer.h is plain C.
  add_executable(mossinfer_example_c ${CMAKE_SOURCE_DIR}/examples/c/minimal.c)
  set_target_properties(mossinfer_example_c PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  target_compile_options(mossinfer_example_c PRIVATE ${MOSSINFER_WARN})
  if(TARGET mossinfer)
    target_link_libraries(mossinfer_example_c PRIVATE mossinfer)
  else()
    target_link_libraries(mossinfer_example_c PRIVATE mossinfer_core)
    set_target_properties(mossinfer_example_c PROPERTIES LINKER_LANGUAGE CXX)  # the static core is C++ and CUDA
  endif()

  if(MOSSINFER_TESTS)
    enable_testing()
    set(MOSSINFER_TEST_WEIGHTS "/workspace/Models/MOSS-Image" CACHE PATH "MOSS-Image weights root ({base,sft}) for GPU tests")
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(NAME server_protocol COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tests/server/test_protocol.py)
      set_tests_properties(server_protocol PROPERTIES LABELS cpu
        ENVIRONMENT "MOSS_SERVER=$<TARGET_FILE:moss-server>;MOSSINFER_GPU_TESTS=0")
      add_test(NAME server_protocol_gpu COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tests/server/test_protocol.py)
      set_tests_properties(server_protocol_gpu PROPERTIES LABELS gpu
        ENVIRONMENT "MOSS_SERVER=$<TARGET_FILE:moss-server>;MOSSINFER_GPU_TESTS=1;MOSS_WEIGHTS=${MOSSINFER_TEST_WEIGHTS}")
    endif()
    add_test(NAME c_example_minimal COMMAND mossinfer_example_c ${MOSSINFER_TEST_WEIGHTS}/base
             ${CMAKE_BINARY_DIR}/c_example_minimal.png "a red apple on a wooden table" 256 2)
    set_tests_properties(c_example_minimal PROPERTIES LABELS gpu)
  endif()
endif()
