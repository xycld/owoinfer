# src/pipeline: the sigma schedule reproduces torch's fp32 CPU arithmetic op by op (explicit std::fma where torch's
# GCC build fused), so the host compiler must neither contract other mul+add pairs nor apply fast-math. No-op when
# MOSSINFER_ONLY filtered the file out.
# MSVC: Visual Studio 2022 (MSVC 19.30) or newer is required, and sampler.cpp gets /fp:strict. /fp:precise contracted
# a*b+c into FMAs before VS2022 (and does again when /fp:contract is added), which would change the schedule's last
# bits; /fp:strict never contracts or reassociates, whatever other flags are set.
set(_mi_pipeline_sources ${MOSSINFER_SOURCES})
list(FILTER _mi_pipeline_sources INCLUDE REGEX ".*/src/pipeline/sampler\\.cpp$")
if(_mi_pipeline_sources)
  if(MSVC)
    if(MSVC_VERSION LESS 1930)
      message(FATAL_ERROR "src/pipeline needs Visual Studio 2022 (MSVC 19.30) or newer, found ${MSVC_VERSION}")
    endif()
    set_source_files_properties(${_mi_pipeline_sources} PROPERTIES COMPILE_OPTIONS "/fp:strict")
  else()
    set_source_files_properties(${_mi_pipeline_sources} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
  endif()
endif()
unset(_mi_pipeline_sources)

# tests/unit/test_pipeline.cpp decodes the PNG writer's output with the vendored stb_image (third_party/stb, see its
# VENDORED.md) as an independent check. System include: stb is not warning-clean under -Wall -Wconversion.
set(_mi_pipeline_test ${CMAKE_SOURCE_DIR}/tests/unit/test_pipeline.cpp)
if(MSVC)
  set_source_files_properties(${_mi_pipeline_test} PROPERTIES COMPILE_OPTIONS
    "/external:I${CMAKE_SOURCE_DIR}/third_party/stb;/external:W0")
else()
  set_source_files_properties(${_mi_pipeline_test} PROPERTIES COMPILE_OPTIONS "-isystem;${CMAKE_SOURCE_DIR}/third_party/stb")
endif()
unset(_mi_pipeline_test)
