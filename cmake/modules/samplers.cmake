# Lane D. src/pipeline/samplers: the schedulers and the Brownian tree reproduce fp32 reference arithmetic op by op
# (std::fma only where written), so the host compiler must neither contract mul+add pairs nor apply fast-math.
# No-op when MOSSINFER_ONLY filtered the files out. MSVC: /fp:strict (never contracts or reassociates).
set(_mi_samplers_sources ${MOSSINFER_SOURCES})
list(FILTER _mi_samplers_sources INCLUDE REGEX ".*/src/pipeline/samplers/(schedules|brownian_tree)\\.cpp$")
if(_mi_samplers_sources)
  if(MSVC)
    set_source_files_properties(${_mi_samplers_sources} PROPERTIES COMPILE_OPTIONS "/fp:strict")
  else()
    set_source_files_properties(${_mi_samplers_sources} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
  endif()
endif()
unset(_mi_samplers_sources)

# tests/unit/test_png_metadata.cpp decodes the PNG writer's output with the vendored stb_image (third_party/stb) as an
# independent check. System include: stb is not warning-clean under -Wall -Wconversion.
set(_mi_png_test ${CMAKE_SOURCE_DIR}/tests/unit/test_png_metadata.cpp)
if(MSVC)
  set_source_files_properties(${_mi_png_test} PROPERTIES COMPILE_OPTIONS "/external:I${CMAKE_SOURCE_DIR}/third_party/stb;/external:W0")
else()
  set_source_files_properties(${_mi_png_test} PROPERTIES COMPILE_OPTIONS "-isystem;${CMAKE_SOURCE_DIR}/third_party/stb")
endif()
unset(_mi_png_test)
