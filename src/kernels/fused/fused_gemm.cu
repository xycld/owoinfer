// Shared checks, tile choice, interleave_up_gate and the plain GEMM (see fused_gemm.h).
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
  // Measured on sm_89 at the DiT shapes (M = 4k..21k): 256x128 for the wide ffn_up_gate (N = 16384),
  // 128x256 otherwise; below one wave of 128x256 tiles the 128x128 tile keeps more SMs busy.
  const std::int64_t tiles = (static_cast<std::int64_t>(p.m) + 127) / 128 * ((p.n + 255) / 256);
  if (tiles < 2 * 76) return GemmTile::k128x128;
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

// out row 2*g*G + s*G + i <- w row s*F + g*G + i; one thread per 16-byte chunk of a row.
__global__ void interleave_kernel(const uint4* w, uint4* out, std::int64_t f, std::int64_t chunks_per_row) {
  const std::int64_t total = 2 * f * chunks_per_row;
  for (std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t row = idx / chunks_per_row, c = idx % chunks_per_row;
    const std::int64_t g = row / (2 * kUpGateGroup), s = (row / kUpGateGroup) % 2, i = row % kUpGateGroup;
    out[idx] = w[(s * f + g * kUpGateGroup + i) * chunks_per_row + c];
  }
}

}  // namespace
}  // namespace detail

void set_gemm_tile_override(GemmTile tile) { detail::g_tile_override.store(tile); }

void gemm(const Tensor& out, const Tensor& x, const Tensor& w, cudaStream_t stream) {
  const detail::Problem p = detail::check_problem("gemm", out, x, w, w.dim(0));
  if (p.m == 0) return;
  switch (detail::tile_for(p)) {
    case GemmTile::k256x128: detail::plain<detail::Tile256x128>(p, out, x, w, stream); break;
    case GemmTile::k128x128: detail::plain<detail::Tile128x128>(p, out, x, w, stream); break;
    default: detail::plain<detail::Tile128x256>(p, out, x, w, stream); break;
  }
}

void interleave_up_gate(const Tensor& w, const Tensor& out, cudaStream_t stream) {
  constexpr const char* op = "interleave_up_gate";
  const bool ok = w.device == Device::kCUDA && out.device == Device::kCUDA && w.dtype == DType::kBF16 &&
                  out.dtype == DType::kBF16 && w.rank == 2 && out.rank == 2 && w.is_contiguous() && out.is_contiguous() &&
                  w.dim(0) == out.dim(0) && w.dim(1) == out.dim(1) && w.dim(0) % (2 * kUpGateGroup) == 0 &&
                  w.dim(1) % 8 == 0 && detail::aligned16(w.data) && detail::aligned16(out.data) && w.data != out.data;
  if (!ok) {
    detail::fail(op, "expected distinct contiguous 16-byte aligned bf16 [2F, K] tensors (F % 4 == 0, K % 8 == 0), got " +
                         w.describe() + " -> " + out.describe());
  }
  const std::int64_t f = w.dim(0) / 2, chunks = w.dim(1) / 8;
  const std::int64_t total = 2 * f * chunks;
  const auto blocks = static_cast<unsigned>(std::min<std::int64_t>((total + 255) / 256, 65535));
  detail::interleave_kernel<<<blocks, 256, 0, stream>>>(w.ptr<const uint4>(), out.ptr<uint4>(), f, chunks);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::fused
