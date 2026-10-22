# Lane H (W4A4): the offline ConvRot W4A4 weight quantizer (src/weights/quant/convrot_w4a4.cpp) reproduces
# comfy-kitchen's scale arithmetic op by op, so it builds without FMA contraction and fast-math. Tolerates the source
# being filtered out.
set(_mi_w4a4_prepare ${CMAKE_SOURCE_DIR}/src/weights/quant/convrot_w4a4.cpp)
if(_mi_w4a4_prepare IN_LIST MOSSINFER_SOURCES)
  if(MSVC)
    set_source_files_properties(${_mi_w4a4_prepare} PROPERTIES COMPILE_OPTIONS "/fp:precise")
  else()
    set_source_files_properties(${_mi_w4a4_prepare} PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
  endif()
endif()
unset(_mi_w4a4_prepare)
