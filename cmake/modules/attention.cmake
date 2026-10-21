# FlashAttention-2 forward kernels (third_party/flash-attention, BSD-3, see its VENDORED.md). Its headers
# apply only to the CUDA sources of src/kernels/attention, which MOSSINFER_ONLY may have filtered out
# (then this fragment does nothing). -isystem: the vendored headers are not warning-clean under -Wall.
set(_mi_attention_sources ${MOSSINFER_SOURCES})
list(FILTER _mi_attention_sources INCLUDE REGEX ".*/src/kernels/attention/[^/]*\\.cu$")
if(_mi_attention_sources)
  set_source_files_properties(${_mi_attention_sources} PROPERTIES COMPILE_OPTIONS
    "-isystem;${CMAKE_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src")
endif()
unset(_mi_attention_sources)
