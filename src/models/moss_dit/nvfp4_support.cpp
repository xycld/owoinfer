#include "models/moss_dit/nvfp4_support.h"

#include <cuda_runtime.h>

#include "kernels/nvfp4/nvfp4.h"

namespace mi::moss_dit {

std::string nvfp4_unsupported_reason(int major, int minor, bool kernels_available) {
  constexpr const char* kNeed = "NVFP4 needs an RTX 50 GPU (compute capability 12.0); use the INT8 file";
  if (major < 0) return std::string(kNeed) + " (no CUDA device)";
  if (major != 12 || minor != 0) {
    return std::string(kNeed) + " (this GPU is compute capability " + std::to_string(major) + "." + std::to_string(minor) + ")";
  }
  if (!kernels_available) {
    return "NVFP4 needs the sm_120a NVFP4 kernels, which this build of MossInfer does not include (built without "
           "MOSSINFER_NVFP4, CUDA < 12.8 or sm_120); use the INT8 file";
  }
  return {};
}

std::string nvfp4_unsupported_reason() {
  int count = 0, device = 0, major = 0, minor = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 || cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    return nvfp4_unsupported_reason(-1, 0, false);
  }
  // available() is queried only on cc 12.0: elsewhere the answer is already no.
  const bool cc120 = major == 12 && minor == 0;
  return nvfp4_unsupported_reason(major, minor, cc120 && kernels::nvfp4::available());
}

}  // namespace mi::moss_dit
