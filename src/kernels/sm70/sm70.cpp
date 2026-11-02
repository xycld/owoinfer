#include "kernels/sm70/sm70.h"

#include <atomic>

namespace mi::kernels::sm70 {

bool available() {
#if MOSSINFER_HAVE_SM70
  return true;
#else
  return false;
#endif
}

bool volta() {
  if (!available()) return false;
  static std::atomic<int> cached[64];
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (device < 0 || device >= 64) return false;
  int cc = cached[device].load(std::memory_order_relaxed);
  if (cc == 0) {
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    cc = major * 10 + minor;
    cached[device].store(cc, std::memory_order_relaxed);
  }
  return cc >= 70 && cc < 75;
}

}  // namespace mi::kernels::sm70
