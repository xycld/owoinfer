# ctest labels for the gtest cases (P7a):
#   gpu            needs a CUDA device (run under the GPU lock on the dev box)
#   golden         needs the large tests/golden/**/*.npy files, which are not in the repository (on gpu tests too)
#   cpu            everything else: no device, no out-of-repository data. CI runs `ctest -L cpu`
#   bitexact_cu13  (in addition) compares bit for bit against torch built with CUDA 13.4 libdevice
#                  (the gate toolchain); the cu12 flavour excludes these and is validated by the
#                  noise-floor gates (AC5, AC6) instead. PipelineSchedule.* is not among them: the sigma schedule
#                  is host code (fp32 mul/div/fma, -ffp-contract=off or /fp:strict, no libm) compared against
#                  torch's CPU arithmetic, so it is exact under any CUDA toolkit. RngTorchCompat.* stays: it is golden
#                  (and its Box-Muller uses libm log/sin/cos, which differ between C runtimes).
# Tests needing the model weights are not labelled separately: they GTEST_SKIP when the weights are absent.
# cpu tests also get MOSSINFER_EXPECT_NO_GPU=1: tests/unit/test_cpu_guard.cpp then fails the test if it left a CUDA
# context on any device, so a GPU test labelled cpu fails on a GPU machine too (CI hides the devices instead).
#
# The map is by gtest suite (and test, where a suite mixes): new suites default to cpu, so a new GPU suite must
# either use the Gpu* prefix or be listed here, or CI fails on it.
#
# Mechanism: tests are discovered at ctest time (DISCOVERY_MODE PRE_TEST), so labels are applied by a ctest-side
# script that runs after gtest's discovery include file and walks its list (mossinfer_tests_TESTS).
if(NOT MOSSINFER_TESTS)
  return()
endif()

set(_mi_labels_script ${CMAKE_CURRENT_BINARY_DIR}/mossinfer_test_labels.cmake)
file(CONFIGURE OUTPUT ${_mi_labels_script} @ONLY CONTENT [==[
# Generated from cmake/modules/test_labels.cmake.
# Names are [Instantiation/]Suite.Test[/Param]; matched on "Suite." (whole suite) or "Suite.Test" (one test).
set(_mi_prefix "^([A-Za-z0-9_]+/)?")
set(_mi_gpu_re "${_mi_prefix}(Gpu[A-Za-z0-9_]*\\.|DeviceArena\\.|GraphCache\\.|PinnedRing\\.|RingFiles\\.|StreamSet\\.|WeightUploadReal\\.|WeightUploadSynthetic\\.|Vae[A-Za-z0-9_]*\\.|Flux2VaeTest\\.|Flux2VaeTiledTest\\.|Flux2VaeTorchGroupNorm\\.|MossDitGolden\\.|MossDitTest\\.|DitStreamTest\\.|DitBlockFused\\.|DitGraph\\.|DitSmallImages\\.|FusedSelfCheck\\.|Qwen35Text\\.|Qwen35TextStream\\.|PipelineLoop\\.|PipelineKernels\\.|PipelineEndToEnd\\.|Varlen\\.UploadAndValidateOnDevice$|DitInt8Block\\.|DitInt8Model\\.|DitInt8Speed\\.|DitInt4[A-Za-z0-9_]*\\.|DitDistill[A-Za-z0-9_]*\\.|Flux2VaeAttention\\.|Flux2VaeFile\\.|Flux2VaeFullTest\\.|Flux2VaeFullTiledTest\\.|Flux2VaeRegress\\.|LatentPreview\\.|LoraBlock\\.|LoraFusedGemm\\.|LoraKernels\\.|LoraSideBranch\\.|PipelineQuality\\.|Qwen35TextQuantDecode\\.|Qwen35TextQuantPerf\\.|QuantTE\\.|SamplerCudaMath\\.|SamplerCudaRandn\\.|SamplerDriver\\.|ControlPassInputs\\.|ControlEmbedding\\.|QualityReport\\.CurrentGpuArch$)")
set(_mi_golden_re "${_mi_prefix}(Flux2VaeTest\\.|Flux2VaeTiledTest\\.|MossDitGolden\\.|MossDitTest\\.|DitStreamTest\\.|DitBlockFused\\.|Qwen35Text\\.|Qwen35TextStream\\.|PipelineLoop\\.|PipelineEndToEnd\\.|RngTorchCompat\\.|Npy\\.LoadsGolden|Npy\\.ToVectorChecksDtype$|PipelineSchedule\\.GoldenSigmasAndTimestepsBitExact$|TokenizerTest\\.GoldenTokenIds$|QuantTE\\.|LatentPreview\\.QualityOnHeldOutGoldenLatents$)")
set(_mi_bitexact_re "${_mi_prefix}(GpuKernelsBasic[A-Za-z0-9_]*\\.|RngTorchCompat\\.)")
foreach(_mi_test IN LISTS mossinfer_tests_TESTS)
  set(_mi_labels "")
  if(_mi_test MATCHES "${_mi_gpu_re}")
    list(APPEND _mi_labels gpu)
  endif()
  if(_mi_test MATCHES "${_mi_golden_re}")
    list(APPEND _mi_labels golden)
  endif()
  # ctest scripts only have set_tests_properties (no APPEND); gtest_discover_tests sets no ENVIRONMENT of its own.
  set(_mi_env "")
  if(NOT _mi_labels)
    list(APPEND _mi_labels cpu)
    set(_mi_env "MOSSINFER_EXPECT_NO_GPU=1")
  endif()
  if(_mi_test MATCHES "${_mi_bitexact_re}")
    list(APPEND _mi_labels bitexact_cu13)
  endif()
  set_tests_properties("${_mi_test}" PROPERTIES LABELS "${_mi_labels}" ENVIRONMENT "${_mi_env}")
endforeach()
]==])

# After gtest_discover_tests (CMakeLists.txt, below the fragments) has registered its include file.
# EVAL: the path is expanded now, not when the deferred call runs.
cmake_language(EVAL CODE
  "cmake_language(DEFER CALL set_property DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES [[${_mi_labels_script}]])")
unset(_mi_labels_script)
