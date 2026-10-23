# Packaging (P7a; the consumer side is docs/c-api.md): the shared C-API library, install rules, the CMake package
# (find_package(MossInfer) -> MossInfer::mossinfer) and CPack archives. With every option at its default this file
# changes nothing about how mossinfer_core, moss-infer and the tests are compiled (install rules only).
#
# Preprocessor contract with include/mossinfer/mossinfer.h: the library build defines MOSSINFER_SHARED and
# MOSSINFER_BUILD (dllexport on Windows); consumers of MossInfer::mossinfer get MOSSINFER_SHARED (dllimport).
option(MOSSINFER_SHARED "Build the shared library libmossinfer.so / mossinfer.dll (exports only mi_* symbols)" OFF)
# Release packages: moss-infer runs the engine in libmossinfer instead of carrying its own copy. The CLI is compiled
# into the library, which then exports its entry point mi_internal_cli_main besides the C API (CMakeLists.txt). OFF
# (dev, tests, gates) keeps the statically linked moss-infer.
option(MOSSINFER_CLI_SHARED "Compile the moss-infer CLI into libmossinfer and link moss-infer against it (needs MOSSINFER_SHARED)" OFF)
if(MOSSINFER_CLI_SHARED AND NOT MOSSINFER_SHARED)
  message(FATAL_ERROR "MOSSINFER_CLI_SHARED=ON links moss-infer against libmossinfer: it needs MOSSINFER_SHARED=ON")
endif()
option(MOSSINFER_STATIC_CUDART "Link the CUDA runtime statically (no cudart library to redistribute)" OFF)
option(MOSSINFER_BUNDLE_CUDA_LIBS "Install the CUDA libraries the binaries load (cuBLASLt, shared cudart)" ON)
option(MOSSINFER_FETCH_GTEST "Download GoogleTest instead of using an installed one (CI)" OFF)
# Linux releases: the build machine's libstdc++ may be newer than the user's; link it (and libgcc) into the shipped
# binaries. Safe in the shared library because the version script keeps those symbols private.
option(MOSSINFER_STATIC_LIBSTDCXX "Linux/GCC: link libstdc++ and libgcc statically into the shipped binaries" OFF)
set(MOSSINFER_EXPECT_CUDA_MAJOR "" CACHE STRING
    "Release builds: fail unless the CUDA toolkit has this major version (12 or 13), so a package is never mislabelled")

if(MOSSINFER_EXPECT_CUDA_MAJOR)
  if(NOT CUDAToolkit_VERSION_MAJOR STREQUAL MOSSINFER_EXPECT_CUDA_MAJOR OR
     NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^${MOSSINFER_EXPECT_CUDA_MAJOR}\\.")
    message(FATAL_ERROR "MOSSINFER_EXPECT_CUDA_MAJOR=${MOSSINFER_EXPECT_CUDA_MAJOR}, but the CUDA toolkit is "
                        "${CUDAToolkit_VERSION} and nvcc is ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER}). "
                        "Point CUDAToolkit_ROOT / CMAKE_CUDA_COMPILER at the matching toolkit.")
  endif()
endif()

# GoogleTest for CI machines without a system package (Windows). Must run before CMakeLists.txt's find_package(GTest),
# which OVERRIDE_FIND_PACKAGE then satisfies with these targets.
if(MOSSINFER_TESTS AND MOSSINFER_FETCH_GTEST)
  include(FetchContent)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # match CMake's default /MD runtime on MSVC
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(GTest
    URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    OVERRIDE_FIND_PACKAGE)
  FetchContent_MakeAvailable(GTest)
endif()

# Static CUDA runtime: swap the cudart that CMakeLists.txt links into mossinfer_core (PUBLIC, so both the link and
# the interface lists name it), and make CMake's own runtime choice for CUDA-linked targets agree. Without this,
# CMake's default (Static) also appended cudart_static after the shared cudart, where it resolved nothing.
if(MOSSINFER_STATIC_CUDART)
  foreach(_mi_prop IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
    get_target_property(_mi_libs mossinfer_core ${_mi_prop})
    list(TRANSFORM _mi_libs REPLACE "^CUDA::cudart$" "CUDA::cudart_static")
    set_property(TARGET mossinfer_core PROPERTY ${_mi_prop} ${_mi_libs})
  endforeach()
  unset(_mi_prop)
  unset(_mi_libs)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  set_property(TARGET mossinfer_core PROPERTY CUDA_RUNTIME_LIBRARY Static)
elseif(MOSSINFER_SHARED)
  # The shared library with the shared cudart: one runtime, which install() bundles. Default builds (neither option)
  # keep CMake's own choice, exactly as before P7a.
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  set_property(TARGET mossinfer_core PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()

include(GNUInstallDirs)
# Archives are relocatable: always bin/ lib/ include/ share/, whatever the distribution's convention.
set(CMAKE_INSTALL_LIBDIR lib)
set(MOSSINFER_INSTALL_CMAKEDIR ${CMAKE_INSTALL_DATADIR}/cmake/MossInfer)
set(MOSSINFER_INSTALL_DOCDIR ${CMAKE_INSTALL_DATADIR}/doc/mossinfer)

if(MOSSINFER_SHARED)
  if(MOSSINFER_ONLY)
    message(FATAL_ERROR "MOSSINFER_SHARED needs a full build (MOSSINFER_ONLY must be empty)")
  endif()
  if(NOT MOSSINFER_BUILD_API)
    message(FATAL_ERROR "MOSSINFER_SHARED is the C API library: it needs MOSSINFER_BUILD_API=ON")
  endif()
  # Only symbols marked MOSSINFER_API (the C API) are visible; everything else in the core is internal.
  set_target_properties(mossinfer_core PROPERTIES CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
                                                  VISIBILITY_INLINES_HIDDEN ON)
  target_compile_definitions(mossinfer_core PRIVATE MOSSINFER_SHARED MOSSINFER_BUILD)

  # The library is the whole core archive (no sources of its own; the stub keeps every generator happy).
  set(_mi_stub ${CMAKE_CURRENT_BINARY_DIR}/mossinfer_shared_stub.cpp)
  file(CONFIGURE OUTPUT ${_mi_stub} CONTENT "// libmossinfer = mossinfer_core linked whole (cmake/modules/packaging.cmake).\n")
  add_library(mossinfer SHARED ${_mi_stub})
  unset(_mi_stub)
  add_library(MossInfer::mossinfer ALIAS mossinfer)
  target_link_libraries(mossinfer PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,mossinfer_core>")
  target_include_directories(mossinfer INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                                                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_definitions(mossinfer INTERFACE MOSSINFER_SHARED)
  set_target_properties(mossinfer PROPERTIES
    OUTPUT_NAME mossinfer
    VERSION ${PROJECT_VERSION}
    SOVERSION ${MOSSINFER_ABI_VERSION}
    LINKER_LANGUAGE CXX
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
  # No $ORIGIN: cuBLASLt is loaded at run time (src/backend/cuda_libs.h), the bundled copy by absolute path, and a
  # RUNPATH would let the plain-name search for a system copy find the bundled one. Only a shared cudart (a
  # MOSSINFER_STATIC_CUDART=OFF build; the release presets link it statically) is a load-time dependency that the
  # bundled copy next to the library must satisfy; there the loader recognises the bundled cuBLASLt by its path.
  if(NOT MOSSINFER_STATIC_CUDART)
    set_property(TARGET mossinfer PROPERTY INSTALL_RPATH "$ORIGIN")
  endif()
  if(NOT WIN32)
    # The version script is the guarantee: statically linked code (cudart_static, CUTLASS templates, libstdc++
    # inlines) cannot leak symbols even where it ignores -fvisibility. -z defs: no unresolved symbols at load time.
    # mi_* is the C API plus, with MOSSINFER_CLI_SHARED, moss-infer's entry mi_internal_cli_main (check_package.py
    # checks the exact list).
    set(_mi_map ${CMAKE_CURRENT_BINARY_DIR}/mossinfer.map)
    file(CONFIGURE OUTPUT ${_mi_map} CONTENT "{\n  global: mi_*;\n  local: *;\n};\n")
    target_link_options(mossinfer PRIVATE "LINKER:--version-script=${_mi_map}")
    if(NOT MOSSINFER_ASAN)  # the sanitizer runtime is resolved from the executable, not the library
      target_link_options(mossinfer PRIVATE "LINKER:-z,defs")
    endif()
    set_property(TARGET mossinfer APPEND PROPERTY LINK_DEPENDS ${_mi_map})
    unset(_mi_map)
  endif()

  install(TARGETS mossinfer EXPORT MossInferTargets RUNTIME_DEPENDENCY_SET mossinfer_cuda_deps
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/mossinfer DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          FILES_MATCHING PATTERN "*.h")
  install(EXPORT MossInferTargets NAMESPACE MossInfer:: DESTINATION ${MOSSINFER_INSTALL_CMAKEDIR})
  include(CMakePackageConfigHelpers)
  configure_package_config_file(${CMAKE_SOURCE_DIR}/cmake/MossInferConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MossInferConfig.cmake INSTALL_DESTINATION ${MOSSINFER_INSTALL_CMAKEDIR}
    PATH_VARS CMAKE_INSTALL_BINDIR)
  # From 1.0, releases with the same MAJOR are source and binary compatible (additions are trailing struct fields,
  # versioned by struct_size), so any 1.x at or above the requested version satisfies a request.
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/MossInferConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MossInferConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/MossInferConfigVersion.cmake
          DESTINATION ${MOSSINFER_INSTALL_CMAKEDIR})
endif()

# Licences of everything compiled into the shipped binaries. stb is test-only and not shipped. The project's
# Apache-2.0 LICENSE and NOTICE go to the archive root (where users look) and to the doc directory.
foreach(_mi_file IN ITEMS LICENSE NOTICE THIRD_PARTY_NOTICES.md)
  if(EXISTS ${CMAKE_SOURCE_DIR}/${_mi_file})
    install(FILES ${CMAKE_SOURCE_DIR}/${_mi_file} DESTINATION .)
    install(FILES ${CMAKE_SOURCE_DIR}/${_mi_file} DESTINATION ${MOSSINFER_INSTALL_DOCDIR})
  else()
    install(CODE "message(FATAL_ERROR \"${_mi_file} is missing at the repository root: a package must carry it\")")
  endif()
endforeach()
install(FILES ${CMAKE_SOURCE_DIR}/third_party/cutlass/LICENSE.txt ${CMAKE_SOURCE_DIR}/third_party/cutlass/VENDORED.md
        DESTINATION ${MOSSINFER_INSTALL_DOCDIR}/third_party/cutlass)
install(FILES ${CMAKE_SOURCE_DIR}/third_party/flash-attention/LICENSE ${CMAKE_SOURCE_DIR}/third_party/flash-attention/AUTHORS
              ${CMAKE_SOURCE_DIR}/third_party/flash-attention/VENDORED.md
        DESTINATION ${MOSSINFER_INSTALL_DOCDIR}/third_party/flash-attention)
install(FILES ${CMAKE_SOURCE_DIR}/third_party/sageattention/LICENSE ${CMAKE_SOURCE_DIR}/third_party/sageattention/VENDORED.md
        DESTINATION ${MOSSINFER_INSTALL_DOCDIR}/third_party/sageattention)
unset(_mi_file)

# Quality certificates: every *.mossinfer-cert.json of the source tree's data/quality_certs/ (none yet: the directory may
# be missing) goes to share/mossinfer/quality_certs/, the directory the engine searches last
# (src/pipeline/quality_cert.cpp install_certificate_dir: <the libmossinfer / executable directory>/../share/mossinfer/
# quality_certs, for a lib/, lib64/ or bin/ directory). The default component, so both the full package and the
# system-CUDA variant (`--component Unspecified`) carry them. The files are listed when installing, so new certificates
# need no reconfigure. MOSSINFER_QUALITY_CERTS_SOURCE in the install's environment names another source directory
# (a packager's own certificates; tests/server/test_install_tree.py).
set(MOSSINFER_QUALITY_CERTS_DIR ${CMAKE_SOURCE_DIR}/data/quality_certs CACHE PATH
    "Directory whose *.mossinfer-cert.json files are installed to share/mossinfer/quality_certs")
install(CODE "
  set(_mi_certs \"${MOSSINFER_QUALITY_CERTS_DIR}\")
  if(DEFINED ENV{MOSSINFER_QUALITY_CERTS_SOURCE})
    file(TO_CMAKE_PATH \"\$ENV{MOSSINFER_QUALITY_CERTS_SOURCE}\" _mi_certs)
  endif()
  if(IS_DIRECTORY \"\${_mi_certs}\")
    file(GLOB _mi_cert_files LIST_DIRECTORIES false \"\${_mi_certs}/*.mossinfer-cert.json\")
    if(_mi_cert_files)
      list(SORT _mi_cert_files)
      file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/share/mossinfer/quality_certs\" TYPE FILE FILES \${_mi_cert_files})
    endif()
  endif()
")

# Executables are defined after the fragments (moss-infer in CMakeLists.txt, moss-server in server.cmake), so their
# install rules run at the end of the directory.
function(_mi_install_executables)
  set(_targets "")
  foreach(t IN ITEMS moss-infer moss-server)
    if(TARGET ${t})
      list(APPEND _targets ${t})
      # RUNPATH $ORIGIN/../lib only where lib/ holds a load-time dependency: libmossinfer (moss-server in the shared
      # build, moss-infer with MOSSINFER_CLI_SHARED) or a shared cudart. An executable that links the core statically
      # and cudart statically gets none: its RUNPATH would steer the cuBLASLt loader's plain-name search for a system
      # copy to the bundled one, past the ld.so cache (src/backend/cuda_libs.h). One that links libmossinfer is safe:
      # the search runs in the library, whose own RUNPATH counts.
      get_target_property(_links ${t} LINK_LIBRARIES)
      if(NOT WIN32 AND (NOT MOSSINFER_STATIC_CUDART OR (TARGET mossinfer AND "mossinfer" IN_LIST _links)))
        set_property(TARGET ${t} PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
      endif()
    endif()
  endforeach()
  unset(_links)
  if(MOSSINFER_STATIC_LIBSTDCXX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT WIN32)
    foreach(t IN LISTS _targets ITEMS mossinfer)
      if(TARGET ${t})
        target_link_options(${t} PRIVATE -static-libstdc++ -static-libgcc)
      endif()
    endforeach()
  endif()
  if(_targets)
    install(TARGETS ${_targets} RUNTIME_DEPENDENCY_SET mossinfer_cuda_deps RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  if(MOSSINFER_BUNDLE_CUDA_LIBS AND (_targets OR TARGET mossinfer))
    # Install component `cuda`: the bundled CUDA libraries and their EULA. The full install (and the CPack archive)
    # has every component; `cmake --install <build> --component Unspecified` is the system-CUDA variant without them.
    # That variant is a release-preset product (MOSSINFER_STATIC_CUDART): with a shared cudart it needs one installed.
    # Only the CUDA toolkit's own redistributables (CUDA EULA Attachment A); never the driver (libcuda / nvcuda).
    # cuBLASLt is not a link dependency (the engine loads it at run time, src/backend/cuda_libs.h): its file is
    # installed by name, under the soname the loader opens (libcublasLt.so.<major> / cublasLt64_<major>.dll).
    if(WIN32)
      find_file(_mi_cublaslt cublasLt64_${CUDAToolkit_VERSION_MAJOR}.dll NO_CACHE NO_DEFAULT_PATH
                PATHS ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_BIN_DIR})
      set(_mi_cublaslt_dest ${CMAKE_INSTALL_BINDIR})
    else()
      find_file(_mi_cublaslt libcublasLt.so.${CUDAToolkit_VERSION_MAJOR} NO_CACHE NO_DEFAULT_PATH
                PATHS ${CUDAToolkit_LIBRARY_DIR})
      set(_mi_cublaslt_dest ${CMAKE_INSTALL_LIBDIR})
    endif()
    if(_mi_cublaslt)
      # Linux: the file the toolkit's soname link points to, installed under the soname (one regular file, which
      # every archive format keeps as it is).
      get_filename_component(_mi_cublaslt_name ${_mi_cublaslt} NAME)
      file(REAL_PATH ${_mi_cublaslt} _mi_cublaslt)
      install(PROGRAMS ${_mi_cublaslt} DESTINATION ${_mi_cublaslt_dest} RENAME ${_mi_cublaslt_name} COMPONENT cuda)
      unset(_mi_cublaslt_name)
    else()
      install(CODE "message(FATAL_ERROR \"cuBLASLt (libcublasLt.so.${CUDAToolkit_VERSION_MAJOR} / \
cublasLt64_${CUDAToolkit_VERSION_MAJOR}.dll) not found under ${CUDAToolkit_LIBRARY_DIR} or ${CUDAToolkit_BIN_DIR}: \
MOSSINFER_BUNDLE_CUDA_LIBS bundles it\")" COMPONENT cuda)
    endif()
    unset(_mi_cublaslt)
    unset(_mi_cublaslt_dest)
    # A shared cudart (MOSSINFER_STATIC_CUDART=OFF) is a link dependency: found among the binaries' dependencies.
    install(RUNTIME_DEPENDENCY_SET mossinfer_cuda_deps
      # Spelled case-insensitively: Windows reports DLL names in either case.
      PRE_INCLUDE_REGEXES "[cC][uU][dD][aA][rR][tT]"
      PRE_EXCLUDE_REGEXES ".*"
      POST_INCLUDE_REGEXES "[cC][uU][dD][aA][rR][tT][^/\\\\]*$"
      POST_EXCLUDE_REGEXES ".*"
      DIRECTORIES ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT cuda
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT cuda)
    # Redistributing cuBLASLt is allowed only together with the CUDA EULA. A missing EULA stops the install (not the
    # configuration: plain builds that are never installed stay unaffected).
    set(_eula_found OFF)
    foreach(eula IN ITEMS ${CUDAToolkit_LIBRARY_ROOT}/EULA.txt ${CUDAToolkit_BIN_DIR}/../EULA.txt)
      if(EXISTS ${eula})
        install(FILES ${eula} DESTINATION ${MOSSINFER_INSTALL_DOCDIR}/third_party/cuda COMPONENT cuda)
        set(_eula_found ON)
        break()
      endif()
    endforeach()
    if(NOT _eula_found)
      install(CODE "message(FATAL_ERROR \"CUDA EULA.txt not found under ${CUDAToolkit_LIBRARY_ROOT} or ${CUDAToolkit_BIN_DIR}/..: \
the bundled CUDA libraries (MOSSINFER_BUNDLE_CUDA_LIBS) may only be redistributed with it\")" COMPONENT cuda)
    endif()
  endif()
  # MSVC: the C++ runtime the binaries are linked against (/MD: vcruntime140, msvcp140, ...), next to them in bin/,
  # so a machine without the Visual C++ Redistributable runs the package. The UCRT is part of Windows 10 and newer.
  if(MSVC AND (_targets OR TARGET mossinfer))
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${CMAKE_INSTALL_BINDIR})
    set(CMAKE_INSTALL_UCRT_LIBRARIES OFF)
    include(InstallRequiredSystemLibraries)
  endif()
  # Both install variants of this build (full and `--component Unspecified`), each tree's moss-server asked which
  # cuBLASLt copy it uses, and check_package.py on both. Needs no GPU; writes the trees under the build directory.
  if(MOSSINFER_TESTS AND MOSSINFER_BUNDLE_CUDA_LIBS AND TARGET mossinfer AND TARGET moss-server AND
     TARGET Python3::Interpreter)
    # The system-CUDA variant is a release-preset product (static cudart); a shared-cudart build's slim tree also
    # needs a system cudart, which the test tolerates (--shared-cudart).
    set(_mi_cudart_flag "")
    if(NOT MOSSINFER_STATIC_CUDART)
      set(_mi_cudart_flag --shared-cudart)
    endif()
    add_test(NAME package_install_tree COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tests/server/test_install_tree.py
             --build ${CMAKE_BINARY_DIR} --work ${CMAKE_BINARY_DIR}/install_tree_test --cmake ${CMAKE_COMMAND}
             --check-package ${CMAKE_SOURCE_DIR}/.github/scripts/check_package.py
             --cuda-major ${CUDAToolkit_VERSION_MAJOR} ${_mi_cudart_flag})
    set_tests_properties(package_install_tree PROPERTIES LABELS cpu TIMEOUT 600)
  endif()
endfunction()
cmake_language(DEFER CALL _mi_install_executables)

# CPack: mossinfer-<version>-<os>-cu<12|13>-<arch>.{zip,tar.gz}.
if(WIN32)
  set(_mi_os windows)
  set(CPACK_GENERATOR ZIP)
else()
  string(TOLOWER ${CMAKE_SYSTEM_NAME} _mi_os)
  set(CPACK_GENERATOR TGZ ZIP)
endif()
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mi_arch)
if(_mi_arch MATCHES "^(amd64|x64|x86_64)$")
  set(_mi_arch x86_64)
elseif(_mi_arch MATCHES "^(arm64|aarch64)$")
  set(_mi_arch aarch64)
endif()
set(CPACK_PACKAGE_NAME mossinfer)
set(CPACK_PACKAGE_VENDOR MossInfer)
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native MOSS-Image inference engine (C API, moss-infer, moss-server)")
set(CPACK_PACKAGE_FILE_NAME mossinfer-${PROJECT_VERSION}-${_mi_os}-cu${CUDAToolkit_VERSION_MAJOR}-${_mi_arch})
set(CPACK_ARCHIVE_THREADS 0)
set(CPACK_PACKAGE_CHECKSUM SHA256)
include(CPack)
unset(_mi_os)
unset(_mi_arch)
