// Shared checks, tile choice and the plain INT8 GEMM (see quant.h).
#include <algorithm>
#include <atomic>
#include <climits>
#include <string>

#include "kernels/quant/int8_gemm_impl.cuh"

namespace mi::kernels::quant {
namespace detail {
namespace {

using kernels::detail::fail;

// No shared alignment helper exists (kernels/basic/detail.h has none); rotate_quantize.cu has its own aligned().
bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// A device attribute of the current device, cached per device (int8_tile_for and int8_band run on every launch).
template <cudaDeviceAttr kAttr>
int device_attribute() {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cached[kMaxDevices];
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  int n = dev >= 0 && dev < kMaxDevices ? cached[dev].load(std::memory_order_relaxed) : 0;
  if (n == 0) {
    MI_CUDA_CHECK(cudaDeviceGetAttribute(&n, kAttr, dev));
    if (dev >= 0 && dev < kMaxDevices) cached[dev].store(n, std::memory_order_relaxed);
  }
  return n;
}

int sm_count() { return device_attribute<cudaDevAttrMultiProcessorCount>(); }

}  // namespace

int int8_band(const Problem& p, int tile_n) {
  // Measured on sm_89 (64 MiB L2) at the DiT shapes, steady state at the 175 W cap: at M = 32798 (2048^2 with CFG)
  // the wide bands run qkv 4.6 %, up_gate 7 %, down 3.7 %, o 1.3 % faster than the identity order (less DRAM traffic
  // under the power cap). At M = 8222 (1024^2) qkv, o and up_gate (24 MiB activations) keep the identity order (the
  // orders are within ~1-4 % either way there); ffn_down (64 MiB, over the floor) bands: 1.66 -> 1.61 ms. sm_80, sm_86
  // and sm_120 (96 MiB L2: its 1024^2 ffn_down stays on the identity order) are unmeasured.
  constexpr int kIdentity = 8;
  const std::int64_t l2 = device_attribute<cudaDevAttrL2CacheSize>();
  const std::int64_t activation = static_cast<std::int64_t>(p.m) * p.k;
  if (activation <= std::max<std::int64_t>(l2 * 3 / 4, std::int64_t{48} << 20)) return kIdentity;
  const std::int64_t band = l2 / 2 / (static_cast<std::int64_t>(tile_n) * p.k);
  const std::int64_t n_tiles = (p.n + tile_n - 1) / tile_n;
  return static_cast<int>(std::min(std::max<std::int64_t>(band, kIdentity), n_tiles));
}

Problem check_operands(const char* op, const Int8Operands& a) {
  if (a.m < 0 || a.n <= 0 || a.k <= 0 || a.m > INT32_MAX || a.n > INT32_MAX || a.k > kMaxInt8GemmK) {
    fail(op, "need 0 <= m < 2^31, 0 < n < 2^31 and 0 < k <= " + std::to_string(kMaxInt8GemmK) + " (m " +
                 std::to_string(a.m) + ", n " + std::to_string(a.n) + ", k " + std::to_string(a.k) + ")");
  }
  if (a.k % kAlignAB != 0 || a.ldx < a.k || a.ldx % kAlignAB != 0) {
    fail(op, "k and ldx must be multiples of 16 with ldx >= k (k " + std::to_string(a.k) + ", ldx " +
                 std::to_string(a.ldx) + ")");
  }
  if (a.n % kAlignC != 0) fail(op, "n must be a multiple of 8, got " + std::to_string(a.n));
  if (a.x_q == nullptr || a.x_scale == nullptr || a.w == nullptr || a.w_scale == nullptr) fail(op, "null operand");
  if (!aligned16(a.x_q) || !aligned16(a.w) || !aligned16(a.w_scale)) {
    fail(op, "x_q, w and w_scale must be 16-byte aligned");
  }
  Problem p;
  p.m = static_cast<int>(a.m);
  p.n = static_cast<int>(a.n);
  p.k = static_cast<int>(a.k);
  p.ldx = a.ldx;
  return p;
}

void check_output(const char* op, const char* name, const void* p, std::int64_t ld, std::int64_t n_out) {
  if (p == nullptr || !aligned16(p) || ld < n_out || ld % kAlignC != 0) {
    fail(op, std::string(name) + " must be 16-byte aligned with a row stride >= " + std::to_string(n_out) +
                 " and a multiple of 8, got " + std::to_string(ld));
  }
}

void check_vector(const char* op, const char* name, const void* p, bool optional) {
  if ((p == nullptr && !optional) || !aligned16(p)) fail(op, std::string(name) + " must be a 16-byte aligned vector");
}

Int8Tile resolve_tile(Int8Tile tile, const Problem& p) {
  if (tile == Int8Tile::kAuto) return int8_tile_for(p.m, p.n, p.k);
  if (static_cast<int>(tile) <= 0 || tile >= Int8Tile::kCount) {
    fail("quant", "invalid Int8Tile");
  }
  return tile;
}

void check_status(cutlass::Status status, const char* op, const char* stage) {
  if (status != cutlass::Status::kSuccess) {
    throw CudaError(std::string("mi::kernels::") + op + ": CUTLASS " + stage + " failed: " +
                    cutlassGetStatusString(status));
  }
}

namespace {

template <class Tile>
void plain(const Problem& p, const Int8Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
           cudaStream_t stream) {
  using N = Nodes<Tile>;
  using Gemm = typename Int8Kernel<Tile, typename N::Plain>::Gemm;
  using namespace cute;
  const typename N::Plain::Arguments cb{
      {{},
       {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}},
       {a.w_scale, 0.f, {_0{}, _1{}, p.n}},
       {reinterpret_cast<const bf16*>(bias), bf16(0), {_0{}, _1{}, p.n}},
       {}},
      {reinterpret_cast<bf16*>(out), {ldo, _1{}, ldo * p.m}}};
  launch<Gemm>("quant::int8_gemm", p, a, cb, stream);
}

}  // namespace
}  // namespace detail

const char* int8_tile_name(Int8Tile tile) {
  switch (tile) {
    case Int8Tile::k128x128x64: return "128x128x64";
    case Int8Tile::k128x256x64: return "128x256x64";
    case Int8Tile::k256x128x64: return "256x128x64";
    case Int8Tile::k64x128x64: return "64x128x64";
    case Int8Tile::k128x128x128: return "128x128x128";
    default: return "auto";
  }
}

Int8Tile int8_tile_for(std::int64_t m, std::int64_t n, std::int64_t k) {
  static_cast<void>(k);
  // Measured on sm_89 (76 SMs) at the 175 W cap with ~1.5 s steady-state windows (GpuQuantKernels.Throughput,
  // MI_QUANT_PERF_WINDOW_MS=1500): at the DiT shapes (M = 8484) 256x128 is best for the wide ffn_up_gate
  // (N = 16384) and 128x256 for qkv / o / down, both within ~1-2 % of each other; the 128x128 and 64x128 tiles run
  // hotter per op (lower clock under the cap) and lose 7-20 %. Below two waves of 128x256 tiles (the text encoder's
  // ~282-token prompts) 64x128 is best or within ~4 %.
  const std::int64_t tiles_256 = (m + 127) / 128 * ((n + 255) / 256);
  if (tiles_256 < 2 * static_cast<std::int64_t>(detail::sm_count())) return Int8Tile::k64x128x64;
  return n >= 16384 ? Int8Tile::k256x128x64 : Int8Tile::k128x256x64;
}

void int8_gemm(const Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream, Int8Tile tile) {
  constexpr const char* name = "quant::int8_gemm";
  const detail::Problem p = detail::check_operands(name, op);
  detail::check_output(name, "out", out, ldo, p.n);
  detail::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  detail::with_tile(detail::resolve_tile(tile, p),
                    [&](auto cfg) { detail::plain<decltype(cfg)>(p, op, bias, out, ldo, stream); });
}

}  // namespace mi::kernels::quant
