# 8-bit attention kernels (third_party/sageattention, Apache-2.0, see its VENDORED.md). Its headers apply only to the
# CUDA sources of src/kernels/attention_q8, which MOSSINFER_ONLY may have filtered out (then this fragment does nothing).
# -isystem: the vendored headers are not warning-clean under -Wall.
set(_mi_attention_q8_sources ${MOSSINFER_SOURCES})
list(FILTER _mi_attention_q8_sources INCLUDE REGEX ".*/src/kernels/attention_q8/[^/]*\\.cu$")
if(_mi_attention_q8_sources)
  set_source_files_properties(${_mi_attention_q8_sources} PROPERTIES COMPILE_OPTIONS
    "-isystem;${CMAKE_SOURCE_DIR}/third_party/sageattention")
  # The fp16-accumulating e4m3 MMA of the kInt8Fp8 variant needs nvcc 12.8; older compilers leave that variant out
  # (attention_q8 throws std::runtime_error for it, the other three variants work).
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(WARNING "attention_q8: CUDA ${CMAKE_CUDA_COMPILER_VERSION} < 12.8 has no fp16-accumulating e4m3 MMA; "
                    "the int8_fp8 variant is unavailable (int8_fp8_f32acc, int8_fp16 and int8_fp16_f32acc work)")
  endif()
endif()
unset(_mi_attention_q8_sources)
