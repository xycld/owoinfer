# The engine version has one source: the MI_VERSION_* / MI_ABI_VERSION macros of the public C header, which
# mi_version() and mi_abi_version() report. Included by CMakeLists.txt before project(); sets
# MOSSINFER_VERSION ("MAJOR.MINOR.PATCH") and MOSSINFER_ABI_VERSION (the shared library's SOVERSION).
set(_mi_header ${CMAKE_CURRENT_LIST_DIR}/../include/mossinfer/mossinfer.h)
file(STRINGS ${_mi_header} _mi_defines REGEX "^#define MI_(VERSION_MAJOR|VERSION_MINOR|VERSION_PATCH|ABI_VERSION) +[0-9]+")
foreach(_mi_part IN ITEMS VERSION_MAJOR VERSION_MINOR VERSION_PATCH ABI_VERSION)
  set(_mi_${_mi_part} "")
  foreach(_mi_line IN LISTS _mi_defines)
    if(_mi_line MATCHES "^#define MI_${_mi_part} +([0-9]+)")
      set(_mi_${_mi_part} ${CMAKE_MATCH_1})
    endif()
  endforeach()
  if(_mi_${_mi_part} STREQUAL "")
    message(FATAL_ERROR "cmake/version.cmake: #define MI_${_mi_part} not found in ${_mi_header}")
  endif()
endforeach()
set(MOSSINFER_VERSION ${_mi_VERSION_MAJOR}.${_mi_VERSION_MINOR}.${_mi_VERSION_PATCH})
set(MOSSINFER_ABI_VERSION ${_mi_ABI_VERSION})
foreach(_mi_var IN ITEMS _mi_header _mi_defines _mi_line _mi_part _mi_VERSION_MAJOR _mi_VERSION_MINOR _mi_VERSION_PATCH
                         _mi_ABI_VERSION)
  unset(${_mi_var})
endforeach()
unset(_mi_var)
