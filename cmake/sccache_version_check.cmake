# Included by the dev-sccache preset (CMakePresets.json, CMAKE_PROJECT_INCLUDE_BEFORE): stop the configuration when
# the sccache launcher is older than 0.18, which fails on nvcc 13.4 ("fatbinary: Could not open input file").
include_guard(GLOBAL)

if(CMAKE_CUDA_COMPILER_LAUNCHER MATCHES "(^|[/\\])sccache(\\.exe)?$")
  execute_process(COMMAND "${CMAKE_CUDA_COMPILER_LAUNCHER}" --version
                  OUTPUT_VARIABLE _mi_sccache_out RESULT_VARIABLE _mi_sccache_rc ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" _mi_sccache_version "${_mi_sccache_out}")
  if(NOT _mi_sccache_rc EQUAL 0 OR NOT _mi_sccache_version)
    message(FATAL_ERROR "dev-sccache: '${CMAKE_CUDA_COMPILER_LAUNCHER} --version' failed (${_mi_sccache_rc}): "
                        "sccache 0.18 or newer must be on PATH")
  endif()
  if(_mi_sccache_version VERSION_LESS 0.18)
    message(FATAL_ERROR "dev-sccache: sccache ${_mi_sccache_version} is too old, 0.18 or newer is needed "
                        "(older versions fail on nvcc 13.4 compiles)")
  endif()
  message(STATUS "dev-sccache: sccache ${_mi_sccache_version}")
  unset(_mi_sccache_out)
  unset(_mi_sccache_rc)
  unset(_mi_sccache_version)
endif()
