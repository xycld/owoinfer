# Tests of src/models/flux2_vae read the reference PNGs with the vendored stb_image (third_party/stb,
# public domain / MIT, see its VENDORED.md). System include: stb is not warning-clean under -Wall -Wconversion.
# Tests only; harmless when MOSSINFER_TESTS_ONLY filters the file out.
set(_mi_flux2_vae_test ${CMAKE_SOURCE_DIR}/tests/unit/test_flux2_vae.cpp)
if(MSVC)
  set_source_files_properties(${_mi_flux2_vae_test} PROPERTIES COMPILE_OPTIONS
    "/external:I${CMAKE_SOURCE_DIR}/third_party/stb;/external:W0")
else()
  set_source_files_properties(${_mi_flux2_vae_test} PROPERTIES COMPILE_OPTIONS
    "-isystem;${CMAKE_SOURCE_DIR}/third_party/stb")
endif()
unset(_mi_flux2_vae_test)
