# ctest driver for cli_generate_requires_tier (CMakeLists.txt):
#   cmake -DMOSS_INFER=<moss-infer> -P generate_requires_tier.cmake
# `moss-infer generate` without --tier, and with --tier auto, exits with status 2 and names the choices. Both are
# refused while the arguments are read, so the missing model folder is never reached and no GPU is needed.
foreach(_case IN ITEMS "missing" "auto")
  set(_args generate --model /nonexistent/mossinfer-bundle --prompt "a red apple" -o mossinfer-tier-test.png)
  if(_case STREQUAL "auto")
    list(APPEND _args --tier auto)
    set(_want "tier must be reference, high or turbo, got \"auto\"")
  else()
    set(_want "--tier is required: reference, high or turbo")
  endif()
  execute_process(COMMAND "${MOSS_INFER}" ${_args} RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 2)
    message(FATAL_ERROR "tier ${_case}: exit status ${_rc}, expected 2\n${_err}")
  endif()
  string(FIND "${_err}" "${_want}" _at)
  if(_at EQUAL -1)
    message(FATAL_ERROR "tier ${_case}: stderr lacks '${_want}':\n${_err}")
  endif()
endforeach()
