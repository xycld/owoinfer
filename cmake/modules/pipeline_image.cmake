# src/pipeline/pipeline_image.cpp (img2img / inpainting host code): decodes PNG and JPEG with the vendored stb_image
# (third_party/stb, public domain / MIT, see its VENDORED.md; system include: stb is not warning-clean), and computes
# the inpainting mask and blend in fp32 one torch op at a time, so the host compiler must neither contract mul+add
# pairs nor apply fast-math (MSVC: /fp:strict, as cmake/modules/pipeline.cmake). No-op when MOSSINFER_ONLY filtered
# the file out.
set(_mi_pipeline_image ${MOSSINFER_SOURCES})
list(FILTER _mi_pipeline_image INCLUDE REGEX ".*/src/pipeline/pipeline_image\\.cpp$")
if(_mi_pipeline_image)
  if(MSVC)
    set_source_files_properties(${_mi_pipeline_image} PROPERTIES COMPILE_OPTIONS
      "/fp:strict;/external:I${CMAKE_SOURCE_DIR}/third_party/stb;/external:W0")
  else()
    set_source_files_properties(${_mi_pipeline_image} PROPERTIES COMPILE_OPTIONS
      "-ffp-contract=off;-fno-fast-math;-isystem;${CMAKE_SOURCE_DIR}/third_party/stb")
  endif()
endif()
unset(_mi_pipeline_image)

# Opt-in libFuzzer target for the image decoder (tests/fuzz): a separate clang build, so the engine's own toolchain
# (GCC / MSVC with nvcc) is untouched. Needs clang++ on PATH.
option(MOSSINFER_FUZZ "Build the fuzz targets in tests/fuzz with clang (libFuzzer)" OFF)
if(MOSSINFER_FUZZ)
  include(ExternalProject)
  ExternalProject_Add(mossinfer_fuzz
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/tests/fuzz
    BINARY_DIR ${CMAKE_BINARY_DIR}/fuzz
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON)
endif()
