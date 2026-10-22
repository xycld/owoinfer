# Runtime LoRA stack (src/models/lora): the host merge is bit-reproducible only without FMA contraction and
# fast-math, as the converter's merge (MOSSINFER_STRICT_FP_SOURCES). MOSSINFER_ONLY may have filtered it out.
set(_mi_lora_strict ${MOSSINFER_SOURCES})
list(FILTER _mi_lora_strict INCLUDE REGEX ".*/src/models/lora/lora_merge\\.cpp$")
if(_mi_lora_strict)
  if(MSVC)
    set_source_files_properties(${_mi_lora_strict} PROPERTIES COMPILE_OPTIONS "/fp:precise")
  else()
    set_source_files_properties(${_mi_lora_strict} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
  endif()
endif()
unset(_mi_lora_strict)
