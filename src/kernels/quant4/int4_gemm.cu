// Shared checks, tile choice and the plain INT4 GEMM (see quant4.h).
#include <atomic>
#include <climits>
#include <string>

#include "kernels/quant4/int4_gemm_impl.cuh"

namespace mi::kernels::quant4 {
namespace detail {
namespace {

using kernels::detail::fail;

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

int sm_count() {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cached[kMaxDevices];
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  int n = dev >= 0 && dev < kMaxDevices ? cached[dev].load(std::memory_order_relaxed) : 0;
  if (n == 0) {
    MI_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    if (dev >= 0 && dev < kMaxDevices) cached[dev].store(n, std::memory_order_relaxed);
  }
  return n;
}

}  // namespace

Problem check_operands(const char* op, const Int4Operands& a) {
  if (a.m < 0 || a.n <= 0 || a.k <= 0 || a.m > INT32_MAX || a.n > INT32_MAX || a.k > kMaxInt4GemmK) {
    fail(op, "need 0 <= m < 2^31, 0 < n < 2^31 and 0 < k <= " + std::to_string(kMaxInt4GemmK) + " (m " +
                 std::to_string(a.m) + ", n " + std::to_string(a.n) + ", k " + std::to_string(a.k) + ")");
  }
  if (a.k % kAlignAB != 0 || a.ldx < a.k / 2 || a.ldx % 16 != 0) {
    fail(op, "k must be a multiple of 32 and ldx (bytes) a multiple of 16 with ldx >= k / 2 (k " +
                 std::to_string(a.k) + ", ldx " + std::to_string(a.ldx) + ")");
  }
  if (a.n % quant::detail::kAlignC != 0) fail(op, "n must be a multiple of 8, got " + std::to_string(a.n));
  if (a.x_q == nullptr || a.x_scale == nullptr || a.w == nullptr || a.w_scale == nullptr) fail(op, "null operand");
  if (!aligned16(a.x_q) || !aligned16(a.w) || !aligned16(a.w_scale)) {
    fail(op, "x_q, w and w_scale must be 16-byte aligned");
  }
  Problem p;
  p.m = static_cast<int>(a.m);
  p.n = static_cast<int>(a.n);
  p.k = static_cast<int>(a.k);
  p.ldx = 2 * a.ldx;
  return p;
}

void check_output(const char* op, const char* name, const void* p, std::int64_t ld, std::int64_t n_out) {
  if (p == nullptr || !aligned16(p) || ld < n_out || ld % quant::detail::kAlignC != 0) {
    fail(op, std::string(name) + " must be 16-byte aligned with a row stride >= " + std::to_string(n_out) +
                 " and a multiple of 8, got " + std::to_string(ld));
  }
}

void check_vector(const char* op, const char* name, const void* p, bool optional) {
  if ((p == nullptr && !optional) || !aligned16(p)) fail(op, std::string(name) + " must be a 16-byte aligned vector");
}

Int4Tile resolve_tile(Int4Tile tile, const Problem& p) {
  if (tile == Int4Tile::kAuto) return int4_tile_for(p.m, p.n, p.k);
  if (static_cast<int>(tile) <= 0 || tile >= Int4Tile::kCount) fail("quant4", "invalid Int4Tile");
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
void plain(const Problem& p, const Int4Operands& a, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
           cudaStream_t stream) {
  using N = Nodes<Tile>;
  using Gemm = typename Int4Kernel<Tile, typename N::Plain>::Gemm;
  using namespace cute;
  const typename N::Plain::Arguments cb{
      {{},
       {const_cast<float*>(a.x_scale), 0.f, {_1{}, _0{}, p.m}},
       {a.w_scale, 0.f, {_0{}, _1{}, p.n}},
       {reinterpret_cast<const bf16*>(bias), bf16(0), {_0{}, _1{}, p.n}},
       {}},
      {reinterpret_cast<bf16*>(out), {ldo, _1{}, ldo * p.m}}};
  launch<Gemm>("quant4::int4_gemm", p, a, cb, stream);
}

}  // namespace
}  // namespace detail

const char* int4_tile_name(Int4Tile tile) {
  switch (tile) {
    case Int4Tile::k128x256x128: return "128x256x128";
    case Int4Tile::k256x128x128: return "256x128x128";
    case Int4Tile::k64x128x128: return "64x128x128";
    default: return "auto";
  }
}

Int4Tile int4_tile_for(std::int64_t m, std::int64_t n, std::int64_t k) {
  static_cast<void>(k);
  // Measured on sm_89 at the 175 W cap (GpuQuant4Kernels.Throughput): at M = 8484, 128x256 and 256x128 are within
  // ~2 % of each other (256x128 best for N = 16384), 128x128 tiles lose 5-15 %; 64x128 for small M as the INT8 tier.
  const std::int64_t tiles_256 = (m + 127) / 128 * ((n + 255) / 256);
  if (tiles_256 < 2 * static_cast<std::int64_t>(detail::sm_count())) return Int4Tile::k64x128x128;
  return n >= 16384 ? Int4Tile::k256x128x128 : Int4Tile::k128x256x128;
}

void int4_gemm(const Int4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream, Int4Tile tile) {
  constexpr const char* name = "quant4::int4_gemm";
  const detail::Problem p = detail::check_operands(name, op);
  detail::check_output(name, "out", out, ldo, p.n);
  detail::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  detail::with_tile(detail::resolve_tile(tile, p),
                    [&](auto cfg) { detail::plain<decltype(cfg)>(p, op, bias, out, ldo, stream); });
}

}  // namespace mi::kernels::quant4
