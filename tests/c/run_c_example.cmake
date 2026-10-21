# ctest driver for c_example_minimal (cmake/modules/server.cmake):
#   cmake -DEXAMPLE=<mossinfer_example_c> -DBUNDLE=<bundle dir> -DOUT=<png> -P run_c_example.cmake
# Without a MOSS-Image bundle at BUNDLE it prints the skip marker (the test's SKIP_REGULAR_EXPRESSION) instead of
# failing, like the gtest cases that GTEST_SKIP when the weights are absent.
if(NOT IS_DIRECTORY "${BUNDLE}/transformer")
  message("MOSSINFER_TEST_SKIPPED: no MOSS-Image bundle at ${BUNDLE} (set MOSSINFER_TEST_WEIGHTS)")
  return()
endif()
execute_process(COMMAND "${EXAMPLE}" "${BUNDLE}" "${OUT}" "a red apple on a wooden table" 256 2 RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "mossinfer_example_c failed: ${rc}")
endif()
