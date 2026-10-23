// Run-time dispatch of the NVFP4 GEMMs (see nvfp4.h). Built for every architecture: it calls the sm_120a translation
// units only when cmake/modules/nvfp4.cmake built them (MOSSINFER_HAVE_NVFP4_SM120A) and the device is cc 12.0.
#include "kernels/nvfp4/nvfp4.h"

#include <atomic>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"
#include "kernels/nvfp4/gemm_sm120a.h"

namespace mi::kernels::nvfp4 {
namespace {

using kernels::detail::fail;

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

constexpr int kMaxDevices = 64;

int current_device() {
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  return dev;
}

#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
// SMs of the current device, cached per device.
int sm_count(int dev) {
  static std::atomic<int> cached[kMaxDevices];
  int n = dev >= 0 && dev < kMaxDevices ? cached[dev].load(std::memory_order_relaxed) : 0;
  if (n == 0) {
    MI_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    if (dev >= 0 && dev < kMaxDevices) cached[dev].store(n, std::memory_order_relaxed);
  }
  return n;
}
#endif

bool device_supported([[maybe_unused]] int dev) {
#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
  int major = 0, minor = 0;
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
  // sm_120a SASS runs on cc 12.0 only (not 12.1, not a later 12.x); the probe also catches a missing image.
  return major == 12 && minor == 0 && detail::probe_image() == cudaSuccess;
#else
  return false;
#endif
}

}  // namespace

std::int64_t sf_bytes(std::int64_t rows, std::int64_t k) {
  const std::int64_t sfk = (k + kBlock - 1) / kBlock;
  const std::int64_t r = (rows + kSfTileRows - 1) / kSfTileRows * kSfTileRows;
  return r * ((sfk + kSfTileCols - 1) / kSfTileCols * kSfTileCols);
}

std::int64_t sf_offset(std::int64_t row, std::int64_t kb, std::int64_t k) {
  const std::int64_t col_tiles = (k / kBlock + kSfTileCols - 1) / kSfTileCols;
  return ((row / kSfTileRows) * col_tiles + kb / kSfTileCols) * (kSfTileRows * kSfTileCols) + (row % 32) * 16 +
         ((row % kSfTileRows) / 32) * 4 + kb % kSfTileCols;
}

bool available() {
  // 0 = not yet known, 1 = no, 2 = yes.
  static std::atomic<int> cached[kMaxDevices];
  const int dev = current_device();
  if (dev < 0 || dev >= kMaxDevices) return device_supported(dev);
  int v = cached[dev].load(std::memory_order_relaxed);
  if (v == 0) {
    v = device_supported(dev) ? 2 : 1;
    cached[dev].store(v, std::memory_order_relaxed);
  }
  return v == 2;
}

const char* nvfp4_tile_name(Nvfp4Tile tile) {
  switch (tile) {
    case Nvfp4Tile::k128x128x128: return "128x128x128";
    case Nvfp4Tile::k256x128x128: return "256x128x128";
    default: return "?";
  }
}

void nvfp4_gemm(const Nvfp4Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                cudaStream_t stream, Nvfp4Tile tile) {
  constexpr const char* op = "nvfp4_gemm";
  if (a.m <= 0 || a.n <= 0 || a.k <= 0 || a.m > INT32_MAX || a.n > INT32_MAX || a.k > INT32_MAX) {
    fail(op, "need 0 < m, n, k < 2^31 (m " + std::to_string(a.m) + ", n " + std::to_string(a.n) + ", k " +
                 std::to_string(a.k) + ")");
  }
  if (a.k % 32 != 0) fail(op, "k must be a multiple of 32, got " + std::to_string(a.k));
  if (a.n % 8 != 0) fail(op, "n must be a multiple of 8, got " + std::to_string(a.n));
  if (a.x == nullptr || a.x_sf == nullptr || a.x_scale == nullptr || a.w == nullptr || a.w_sf == nullptr) {
    fail(op, "null operand");
  }
  if (!aligned16(a.x) || !aligned16(a.x_sf) || !aligned16(a.x_scale) || !aligned16(a.w) || !aligned16(a.w_sf)) {
    fail(op, "x, x_sf, x_scale, w and w_sf must be 16-byte aligned");
  }
  if (bias != nullptr && !aligned16(bias)) fail(op, "bias must be 16-byte aligned");
  if (out == nullptr || !aligned16(out) || ldo < a.n || ldo % 8 != 0) {
    fail(op, "out must be 16-byte aligned with a row stride >= n and a multiple of 8, got " + std::to_string(ldo));
  }
  if (tile != Nvfp4Tile::k128x128x128 && tile != Nvfp4Tile::k256x128x128) fail(op, "unknown tile");
  if (!available()) {
    throw std::logic_error("mi::kernels::nvfp4_gemm: NVFP4 GEMMs need a compute capability 12.0 GPU and a build with "
                           "the sm_120a kernels (MOSSINFER_NVFP4, CUDA >= 12.8)");
  }
#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
  const int sms = sm_count(current_device());
  if (tile == Nvfp4Tile::k256x128x128) {
    detail::gemm_256x128(a, bias, out, ldo, sms, stream);
  } else {
    detail::gemm_128x128(a, bias, out, ldo, sms, stream);
  }
#else
  (void)stream;
#endif
}

}  // namespace mi::kernels::nvfp4
