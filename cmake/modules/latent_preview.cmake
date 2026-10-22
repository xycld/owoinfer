# Tests of src/models/latent_preview read the golden PNGs with the vendored stb_image (third_party/stb, public domain /
# MIT, see its VENDORED.md). System include: stb is not warning-clean under -Wall -Wconversion. Tests only; harmless
# when MOSSINFER_TESTS_ONLY filters the file out.
set(_mi_latent_preview_test ${CMAKE_SOURCE_DIR}/tests/unit/test_latent_preview.cpp)
if(MSVC)
  set_source_files_properties(${_mi_latent_preview_test} PROPERTIES COMPILE_OPTIONS
    "/external:I${CMAKE_SOURCE_DIR}/third_party/stb;/external:W0")
else()
  set_source_files_properties(${_mi_latent_preview_test} PROPERTIES COMPILE_OPTIONS
    "-isystem;${CMAKE_SOURCE_DIR}/third_party/stb")
endif()
unset(_mi_latent_preview_test)
