// The GPU architecture the committed reference vectors under tests/data and tests/golden were recorded on: sm_89, the
// dev box. Tests that compare bit patterns with those vectors keep their exact gates there. Elsewhere the same code
// gives other last bits: the tensor cores' fp16-accumulating and tf32 MMAs round differently on sm_120 (upstream
// SageAttention's own int8_fp16 output is 84.8 % identical between sm_89 and sm_120, its f32-accumulating variant
// 100 %), and cuBLASLt / cuDNN pick other algorithms. Off the reference architecture those tests hold noise-class
// gates instead, measured on an RTX 5090 (sm_120, CUDA 13.2).
#pragma once

#include <cuda_runtime.h>

#include "core/cuda_check.h"

namespace mi::testing {

inline bool on_reference_gpu() {
  int device = 0, major = 0, minor = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return major == 8 && minor == 9;
}

}  // namespace mi::testing
