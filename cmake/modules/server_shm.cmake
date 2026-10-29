# moss-server's shared-memory regions (tools/server/shm.{h,cpp}): standalone, so they are tested without the server:
# shm.cpp is compiled into mossinfer_tests (tests/unit/test_server_shm.cpp), and tests/server/test_shm.py drives
# regions across processes through a stand-in server (tests/server/shm_helper.cpp).
if(NOT MOSSINFER_TESTS OR NOT EXISTS ${CMAKE_SOURCE_DIR}/tools/server/shm.cpp)
  return()
endif()

# glibc before 2.34 keeps shm_open in librt (later versions keep an empty librt for compatibility).
set(_mi_shm_libs "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_mi_shm_libs rt)
endif()

# mossinfer_tests is defined after the fragments (CMakeLists.txt): add the source once it exists.
cmake_language(EVAL CODE
  "cmake_language(DEFER CALL target_sources mossinfer_tests PRIVATE [[${CMAKE_SOURCE_DIR}/tools/server/shm.cpp]])")
if(_mi_shm_libs)
  cmake_language(EVAL CODE "cmake_language(DEFER CALL target_link_libraries mossinfer_tests PRIVATE ${_mi_shm_libs})")
endif()

add_executable(mossinfer_shm_helper ${CMAKE_SOURCE_DIR}/tests/server/shm_helper.cpp ${CMAKE_SOURCE_DIR}/tools/server/shm.cpp)
target_include_directories(mossinfer_shm_helper PRIVATE ${CMAKE_SOURCE_DIR}/tools/server)
target_compile_options(mossinfer_shm_helper PRIVATE ${MOSSINFER_WARN})
target_link_libraries(mossinfer_shm_helper PRIVATE ${_mi_shm_libs})

enable_testing()
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME server_shm COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tests/server/test_shm.py)
  set_tests_properties(server_shm PROPERTIES LABELS cpu ENVIRONMENT "MOSS_SHM_HELPER=$<TARGET_FILE:mossinfer_shm_helper>")
endif()
unset(_mi_shm_libs)
