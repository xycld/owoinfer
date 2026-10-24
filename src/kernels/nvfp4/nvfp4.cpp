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

// SMs of the current device, cached per device (nvfp4_tile_for runs on every launch).
int sm_count() {
  static std::atomic<int> cached[kMaxDevices];
  const int dev = current_device();
  int n = dev >= 0 && dev < kMaxDevices ? cached[dev].load(std::memory_order_relaxed) : 0;
  if (n == 0) {
    MI_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    if (dev >= 0 && dev < kMaxDevices) cached[dev].store(n, std::memory_order_relaxed);
  }
  return n;
}

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

void require_available(const char* op) {
  if (!available()) {
    throw std::logic_error(std::string("mi::kernels::") + op +
                           ": NVFP4 GEMMs need a compute capability 12.0 GPU and a build with the sm_120a kernels "
                           "(MOSSINFER_NVFP4, CUDA >= 12.8)");
  }
}

// Checks the operands, the output (or residual) and the optional bias and delta (see nvfp4.h).
void check_gemm(const char* op, const Nvfp4Operands& a, const __nv_bfloat16* bias, const void* out, std::int64_t ldo,
                const Nvfp4Delta& delta, Nvfp4Tile tile) {
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
    fail(op, "the output must be 16-byte aligned with a row stride >= n and a multiple of 8, got " +
                 std::to_string(ldo));
  }
  if (delta.p != nullptr && (!aligned16(delta.p) || delta.ld < a.n || delta.ld % 8 != 0)) {
    fail(op, "delta must be 16-byte aligned with a row stride >= n and a multiple of 8, got " +
                 std::to_string(delta.ld));
  }
  if (static_cast<int>(tile) < 0 || tile >= Nvfp4Tile::kCount) fail(op, "invalid Nvfp4Tile");
}

#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
template <detail::Epi kEpi>
void run(Nvfp4Tile tile, const detail::GemmArgs& g, cudaStream_t stream) {
  const int sms = sm_count();
  switch (tile) {
    case Nvfp4Tile::k128x128x128Pingpong: detail::gemm<Nvfp4Tile::k128x128x128Pingpong, kEpi>(g, sms, stream); break;
    case Nvfp4Tile::k256x128x128: detail::gemm<Nvfp4Tile::k256x128x128, kEpi>(g, sms, stream); break;
    default: detail::gemm<Nvfp4Tile::k128x128x128, kEpi>(g, sms, stream); break;
  }
}
#endif

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
    case Nvfp4Tile::kAuto: return "auto";
    case Nvfp4Tile::k128x128x128: return "128x128x128";
    case Nvfp4Tile::k128x128x128Pingpong: return "128x128x128-pingpong";
    case Nvfp4Tile::k256x128x128: return "256x128x128";
    default: return "?";
  }
}

Nvfp4Tile nvfp4_tile_for(std::int64_t m, std::int64_t n, std::int64_t k) {
  static_cast<void>(k);
  // Measured on the RTX 5090 (170 SMs, 575 W) at the four DiT shapes with M = 1172 .. 16384 (GpuNvfp4Kernels.TileSweep,
  // weights rotating through > 2x the L2): 256x128 is best (by 2-8 %) at one wave of its tiles or less and from ~3
  // waves up (every M >= 8484 shape); in between (M = 2340 / 4378 at N = 3072, M = 1172 at N = 9216) the 128x128
  // Pingpong schedule is best by 1-11 %. The 128x128 Cooperative tile is never best.
  const std::int64_t tiles_256 = (m + 255) / 256 * ((n + 127) / 128);
  const auto sms = static_cast<std::int64_t>(sm_count());
  if (tiles_256 > sms && tiles_256 <= 3 * sms) return Nvfp4Tile::k128x128x128Pingpong;
  return Nvfp4Tile::k256x128x128;
}

void nvfp4_gemm(const Nvfp4Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                cudaStream_t stream, Nvfp4Tile tile, Nvfp4Delta delta) {
  constexpr const char* op = "nvfp4_gemm";
  check_gemm(op, a, bias, out, ldo, delta, tile);
  require_available(op);
#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
  if (tile == Nvfp4Tile::kAuto) tile = nvfp4_tile_for(a.m, a.n, a.k);
  detail::GemmArgs g;
  g.op = a;
  g.bias = bias;
  g.delta = delta;
  g.out = out;
  g.ldo = ldo;
  if (delta.p != nullptr) {
    run<detail::Epi::kPlainDelta>(tile, g, stream);
  } else {
    run<detail::Epi::kPlain>(tile, g, stream);
  }
#else
  (void)stream;
#endif
}

void nvfp4_gemm_gated_residual(const Nvfp4Operands& a, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                               __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream, Nvfp4Tile tile,
                               Nvfp4Delta delta) {
  constexpr const char* op = "nvfp4_gemm_gated_residual";
  check_gemm(op, a, bias, residual, ldr, delta, tile);
  if (gate == nullptr || !aligned16(gate)) fail(op, "gate must be a 16-byte aligned vector");
  require_available(op);
#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
  if (tile == Nvfp4Tile::kAuto) tile = nvfp4_tile_for(a.m, a.n, a.k);
  detail::GemmArgs g;
  g.op = a;
  g.bias = bias;
  g.gate = gate;
  g.delta = delta;
  g.out = residual;
  g.ldo = ldr;
  if (delta.p != nullptr) {
    run<detail::Epi::kGatedResidualDelta>(tile, g, stream);
  } else {
    run<detail::Epi::kGatedResidual>(tile, g, stream);
  }
#else
  (void)stream;
#endif
}

void e2m1_cvt_codes(const float* x, std::uint8_t* codes, std::int64_t n, cudaStream_t stream) {
  constexpr const char* op = "nvfp4::e2m1_cvt_codes";
  if (n < 0 || (n > 0 && (x == nullptr || codes == nullptr))) fail(op, "invalid arguments");
  require_available(op);
#if defined(MOSSINFER_HAVE_NVFP4_SM120A)
  detail::e2m1_cvt(x, codes, n, stream);
#else
  (void)stream;
#endif
}

}  // namespace mi::kernels::nvfp4
