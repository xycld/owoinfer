// Shared checks, tile choice and the plain GEMM (see fused_gemm.h).
#include <algorithm>
#include <atomic>
#include <climits>
#include <stdexcept>
#include <string>

#include "kernels/fused/fused_gemm_impl.cuh"

namespace mi::kernels::fused {
namespace detail {
namespace {

std::atomic<GemmTile> g_tile_override{GemmTile::kAuto};

[[noreturn]] void fail(const char* op, const std::string& msg) {
  throw std::invalid_argument(std::string("mi::kernels::fused::") + op + ": " + msg);
}

// Single row stride of a [..., C] view whose leading dims flatten to M rows; 0 if they do not.
std::int64_t row_stride(const Tensor& t) {
  const kernels::detail::RowMap m = kernels::detail::row_map(t);
  if (m.nd == 0) return t.dim(-1);
  return m.nd == 1 ? m.stride[0] : 0;
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// SMs of the current device, cached per device (tile_for runs on every launch).
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

Problem check_problem(const char* op, const Tensor& out, const Tensor& x, const Tensor& w, std::int64_t n_out) {
  kernels::detail::check_rows(op, "x", x, DType::kBF16);
  const std::int64_t k = x.dim(-1);
  kernels::detail::check_rows(op, "w", w, DType::kBF16, k);
  if (w.rank != 2 || !w.is_contiguous()) fail(op, "w must be a contiguous [N, K], got " + w.describe());
  kernels::detail::check_rows(op, "out", out, DType::kBF16, n_out);
  const std::int64_t m = kernels::detail::rows_of(x);
  if (kernels::detail::rows_of(out) != m) fail(op, "row count mismatch: " + x.describe() + " vs " + out.describe());
  Problem p;
  p.ldx = row_stride(x);
  p.ldw = k;
  p.ldo = row_stride(out);
  if (p.ldx < k || p.ldo < n_out) fail(op, "x and out must be rows with one stride: " + x.describe() + ", " + out.describe());
  if (k % kAlign != 0 || p.ldx % kAlign != 0 || p.ldo % kAlign != 0 || n_out % kAlign != 0 || !aligned16(x.data) ||
      !aligned16(w.data) || !aligned16(out.data)) {
    fail(op, "needs 16-byte aligned operands with K, N and row strides multiples of 8: " + x.describe() + ", " +
                 w.describe() + ", " + out.describe());
  }
  if (m > INT32_MAX || w.dim(0) > INT32_MAX || k > INT32_MAX) fail(op, "dimensions exceed int32");
  p.m = static_cast<int>(m);
  p.n = static_cast<int>(w.dim(0));
  p.k = static_cast<int>(k);
  return p;
}

GemmTile tile_for(const Problem& p) {
  const GemmTile forced = g_tile_override.load();
  if (forced != GemmTile::kAuto) return forced;
  // Measured on sm_89 (76 SMs) at the DiT shapes (M = 4k..21k): 256x128 for the wide ffn_up_gate (N = 16384),
  // 128x256 otherwise; below two waves of 128x256 tiles the 128x128 tile keeps more SMs busy. The large tiles are
  // within ~3% of each other there (power-capped clocks, GpuFusedGemm.Throughput sweeps them).
  const std::int64_t tiles = (static_cast<std::int64_t>(p.m) + 127) / 128 * ((p.n + 255) / 256);
  if (tiles < 2 * static_cast<std::int64_t>(sm_count())) return GemmTile::k128x128;
  return p.n >= 16384 ? GemmTile::k256x128 : GemmTile::k128x256;
}

void check_status(cutlass::Status status, const char* op, const char* stage) {
  if (status != cutlass::Status::kSuccess) {
    throw CudaError(std::string("mi::kernels::fused::") + op + ": CUTLASS " + stage + " failed: " +
                    cutlassGetStatusString(status));
  }
}

namespace {

template <class Tile>
void plain(const Problem& p, const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream) {
  using Gemm = DeviceGemm<Tile, PlainOp, bf16>;
  const typename Gemm::Arguments args({p.m, p.n, p.k}, {x.ptr<const bf16>(), p.ldx}, {w.ptr<const bf16>(), p.ldw},
                                      {out.ptr<bf16>(), p.ldo}, {out.ptr<bf16>(), p.ldo}, typename PlainOp::Params());
  launch<Gemm>("gemm", args, stream);
}

}  // namespace
}  // namespace detail

void set_gemm_tile_override(GemmTile tile) { detail::g_tile_override.store(tile); }

bool matches_cublaslt(const kernels::Gemm::AlgoConfig& algo) {
  const bool k_ordered = algo.id == 5 || algo.id == 6 || algo.id == 21;
  return k_ordered && algo.split_k <= 1 && algo.reduction_scheme == 0 && algo.custom_option == 0;
}

void gemm(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream) {
  const detail::Problem p = detail::check_problem("gemm", out, x, w, w.dim(0));
  if (p.m == 0) return;
  detail::with_tile(detail::tile_for(p), [&](auto tile) { detail::plain<decltype(tile)>(p, out, x, w, stream); });
}

}  // namespace mi::kernels::fused
