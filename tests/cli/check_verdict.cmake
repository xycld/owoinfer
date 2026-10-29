# ctest driver for cli_check_verdict (CMakeLists.txt):
#   cmake -DMOSS_INFER=<moss-infer> -P check_verdict.cmake
# `moss-infer check` answers a refused request with its JSON verdict on stdout and exit status 1, also for what
# generate refuses while it reads its arguments (--strength without --init-image, a --params file that does not read,
# an image that does not decode) and for a model folder that does not exist (no weights or GPU needed: each is
# refused before or at the engine). A usage error (no --tier) stays exit status 2.
set(_base check --model /nonexistent/mossinfer-bundle --tier high --prompt "a red apple")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/mossinfer-check-not-an-image.png" "not a PNG")
set(_cases
    "strength|--strength,0.5|--strength needs --init-image"
    "params|--params,/nonexistent/mossinfer-params.json|mossinfer-params.json"
    "image|--init-image,${CMAKE_CURRENT_BINARY_DIR}/mossinfer-check-not-an-image.png|mossinfer-check-not-an-image.png"
    "model||\"ok\": false")
foreach(_case IN LISTS _cases)
  string(REPLACE "|" ";" _parts "${_case}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _extra)
  list(GET _parts 2 _want)
  string(REPLACE "," ";" _extra "${_extra}")
  execute_process(COMMAND "${MOSS_INFER}" ${_base} ${_extra} --quiet RESULT_VARIABLE _rc OUTPUT_VARIABLE _out
                  ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 1)
    message(FATAL_ERROR "check ${_name}: exit status ${_rc}, expected 1\n${_out}\n${_err}")
  endif()
  foreach(_w IN ITEMS "{\"ok\": false, \"refused_by\": \"" "${_want}")
    string(FIND "${_out}" "${_w}" _at)
    if(_at EQUAL -1)
      message(FATAL_ERROR "check ${_name}: stdout lacks '${_w}':\n${_out}\n${_err}")
    endif()
  endforeach()
endforeach()
execute_process(COMMAND "${MOSS_INFER}" check --model /nonexistent/mossinfer-bundle --prompt x RESULT_VARIABLE _rc
                OUTPUT_QUIET ERROR_QUIET)
if(NOT _rc EQUAL 2)
  message(FATAL_ERROR "check without --tier: exit status ${_rc}, expected 2")
endif()
