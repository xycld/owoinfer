# src/backend: the probe reports which compute capabilities this build has kernel images for, so it is told the
# CUDA architecture list of mossinfer_core (commas: a semicolon would split the definition). This file only.
# No-op when MOSSINFER_ONLY filtered the probe out.
set(_mi_backend_probe ${CMAKE_SOURCE_DIR}/src/backend/probe.cpp)
if(_mi_backend_probe IN_LIST MOSSINFER_SOURCES)
  set_property(SOURCE ${_mi_backend_probe} APPEND PROPERTY COMPILE_DEFINITIONS
    "MOSSINFER_CUDA_ARCHITECTURES=\"$<JOIN:$<TARGET_PROPERTY:mossinfer_core,CUDA_ARCHITECTURES>,$<COMMA>>\"")
endif()
unset(_mi_backend_probe)
# src/backend/cuda_driver.cpp loads libcuda and NVML at run time (dlopen; libdl before glibc 2.34).
if(${CMAKE_SOURCE_DIR}/src/backend/cuda_driver.cpp IN_LIST MOSSINFER_SOURCES AND CMAKE_DL_LIBS)
  target_link_libraries(mossinfer_core PRIVATE ${CMAKE_DL_LIBS})
endif()
