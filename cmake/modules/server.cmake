# moss-server (tools/server, docs/server-protocol.md), the C API example (examples/c) and the C ABI tests. All are
# clients of the public C API only: they link the shared libmossinfer when cmake/modules/packaging.cmake defines it
# (target `mossinfer`), else the static core. Like the CLI they need every module, so they are built in full
# (unfiltered) builds only, except the header-only layout test. Nothing here exists with MOSSINFER_BUILD_API=OFF.
if(NOT MOSSINFER_BUILD_API)
  return()
endif()

# The C targets' warning flags: MSVC's /permissive- is a C++ conformance switch, so it stays off them.
set(_mi_c_warn ${MOSSINFER_WARN})
list(REMOVE_ITEM _mi_c_warn /permissive-)

# The frozen C struct layouts seen from a C compiler (the library checks the same table in C++). Header only.
if(MOSSINFER_TESTS AND EXISTS ${CMAKE_SOURCE_DIR}/tests/c/test_abi_layout.c)
  enable_testing()
  add_executable(mossinfer_abi_layout_c ${CMAKE_SOURCE_DIR}/tests/c/test_abi_layout.c)
  set_target_properties(mossinfer_abi_layout_c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
  target_include_directories(mossinfer_abi_layout_c PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(mossinfer_abi_layout_c PRIVATE ${_mi_c_warn})
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
  target_compile_options(mossinfer_example_c PRIVATE ${_mi_c_warn})
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
    # Skipped, not failed, when the weights are absent (the driver script prints the marker).
    add_test(NAME c_example_minimal COMMAND ${CMAKE_COMMAND} -DEXAMPLE=$<TARGET_FILE:mossinfer_example_c>
             -DBUNDLE=${MOSSINFER_TEST_WEIGHTS}/base -DOUT=${CMAKE_BINARY_DIR}/c_example_minimal.png
             -P ${CMAKE_SOURCE_DIR}/tests/c/run_c_example.cmake)
    set_tests_properties(c_example_minimal PROPERTIES LABELS gpu SKIP_REGULAR_EXPRESSION "MOSSINFER_TEST_SKIPPED")

    # A program built against the 0.1 header (hand-written 0.1 struct layouts) against this library. No GPU needed.
    add_executable(mossinfer_abi_compat_c ${CMAKE_SOURCE_DIR}/tests/c/test_abi_compat.c)
    set_target_properties(mossinfer_abi_compat_c PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_compile_options(mossinfer_abi_compat_c PRIVATE ${_mi_c_warn})
    if(TARGET mossinfer)
      target_link_libraries(mossinfer_abi_compat_c PRIVATE mossinfer)
    else()
      target_link_libraries(mossinfer_abi_compat_c PRIVATE mossinfer_core)
      set_target_properties(mossinfer_abi_compat_c PROPERTIES LINKER_LANGUAGE CXX)
    endif()
    add_test(NAME c_abi_compat COMMAND mossinfer_abi_compat_c)
    set_tests_properties(c_abi_compat PROPERTIES LABELS cpu)
    # The same, plus a generation from the 0.1.0 structs (a PNG without metadata). Skipped without weights or a GPU.
    add_test(NAME c_abi_compat_gpu COMMAND mossinfer_abi_compat_c ${MOSSINFER_TEST_WEIGHTS}/base
             ${CMAKE_BINARY_DIR}/c_abi_compat_gpu.png)
    set_tests_properties(c_abi_compat_gpu PROPERTIES LABELS gpu SKIP_REGULAR_EXPRESSION "MOSSINFER_TEST_SKIPPED")
  endif()
endif()
unset(_mi_c_warn)
