#include "kernels/basic/rope.h"

#include <bit>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"

namespace mi::kernels {
namespace {

using detail::RowMap;
using bf16 = __nv_bfloat16;

template <std::size_t N>
struct Freqs {
  float v[N];
};

template <std::size_t N>
Freqs<N> freqs_from_bits(const std::array<std::uint32_t, N>& bits) {
  Freqs<N> f{};
  for (std::size_t i = 0; i < N; ++i) f.v[i] = std::bit_cast<float>(bits[i]);
  return f;
}

constexpr int kRowsPerBlock = 4;

// One thread per table element. Coordinates follow P:178-185 exactly: integer grid positions converted
// to fp32, centred by subtracting the fp32 scalar (n-1)/2.
__global__ void moss_table_kernel(bf16* out, RowMap rows, std::int64_t n, int height, int width, int text_length,
                                  Freqs<16> bands) {
  using namespace detail;
  const std::int64_t hw = static_cast<std::int64_t>(height) * width;
  const float h_mid = static_cast<float>((height - 1) / 2.0);
  const float w_mid = static_cast<float>((width - 1) / 2.0);
  const int c = static_cast<int>(threadIdx.x);  // 0..127
  for (std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y; r < n;
       r += static_cast<std::int64_t>(gridDim.x) * blockDim.y) {
    const int p = c & 63, axis = p >> 4, band = p & 15;
    float coord = 0.0f;
    if (r < hw) {
      if (axis == 1) coord = static_cast<float>(text_length);
      if (axis == 2) coord = __fsub_rn(static_cast<float>(r / width), h_mid);
      if (axis == 3) coord = __fsub_rn(static_cast<float>(r % width), w_mid);
    } else if (axis == 1) {
      coord = static_cast<float>(r - hw);
    }
    const float phase = mul(coord, bands.v[band]);
    out[rows.offset(r) + c] = bf(c < 64 ? sinf(phase) : cosf(phase));
  }
}

// Rotation of one (lo, hi) pair: lo' = lo*cos + (-hi)*sin, hi' = hi*cos + lo*sin, each product and the
// sum rounded to bf16 (x * cos, cat(-x2, x1) * sin and their sum are three torch ops).
__device__ __forceinline__ void rotate_pair(bf16* lo_out, bf16* hi_out, float lo, float hi, float co, float si) {
  using namespace detail;
  *lo_out = bf(add(rbf(mul(lo, co)), rbf(mul(-hi, si))));
  *hi_out = bf(add(rbf(mul(hi, co)), rbf(mul(lo, si))));
}

// x rows run over [..., L, heads]; the token index of row r is (r / heads) % L.
struct ApplyArgs {
  const bf16* x;
  RowMap x_rows;
  bf16* out;
  RowMap out_rows;
  const bf16* table;
  RowMap table_rows;
  std::int64_t rows;
  std::int64_t heads;
  std::int64_t tokens;
  int head_dim;
};

// blockDim.x = head_dim / 2: threads below `half` rotate a pair, the rest copy pass-through dims.
template <int kRot>
__global__ void rope_apply_kernel(ApplyArgs a) {
  using namespace detail;
  constexpr int kHalf = kRot / 2;
  const int t = static_cast<int>(threadIdx.x);
  for (std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y; r < a.rows;
       r += static_cast<std::int64_t>(gridDim.x) * blockDim.y) {
    const bf16* xr = a.x + a.x_rows.offset(r);
    bf16* outr = a.out + a.out_rows.offset(r);
    const bf16* tab = a.table + a.table_rows.offset((r / a.heads) % a.tokens);
    if constexpr (kRot == kMossRopeDim) {  // MOSS row: [sin(64) | cos(64)]
      rotate_pair(outr + t, outr + t + kHalf, f(xr[t]), f(xr[t + kHalf]), f(tab[kHalf + t]), f(tab[t]));
    } else {  // Qwen row: [cos(32) | sin(32)]
      for (int c = t; c < a.head_dim - kHalf; c += static_cast<int>(blockDim.x)) {
        if (c < kHalf) {
          rotate_pair(outr + c, outr + c + kHalf, f(xr[c]), f(xr[c + kHalf]), f(tab[c]), f(tab[kHalf + c]));
        } else {
          outr[c + kHalf] = xr[c + kHalf];
        }
      }
    }
  }
}

__global__ void qwen_table_kernel(bf16* out, RowMap rows, std::int64_t n, Freqs<32> inv_freq) {
  using namespace detail;
  const int c = static_cast<int>(threadIdx.x);  // 0..63
  for (std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y; r < n;
       r += static_cast<std::int64_t>(gridDim.x) * blockDim.y) {
    const float phase = mul(static_cast<float>(r), inv_freq.v[c & 31]);
    out[rows.offset(r) + c] = bf(c < 32 ? cosf(phase) : sinf(phase));  // * attention_scaling (1.0) is exact
  }
}

unsigned blocks_for(std::int64_t rows) { return detail::grid_for_rows((rows + kRowsPerBlock - 1) / kRowsPerBlock); }

void apply(const char* op, int rot, const Tensor& x, const Tensor& table, const Tensor& out, cudaStream_t stream) {
  detail::check_rows(op, "x", x, DType::kBF16);
  detail::check_rows(op, "out", out, DType::kBF16);
  detail::check_same_shape(op, x, out);
  detail::check_rows(op, "table", table, DType::kBF16, rot);
  if (x.rank < 3 || table.rank != 2 || table.dim(0) != x.dim(-3) || x.dim(-1) < rot ||
      (rot == kMossRopeDim && x.dim(-1) != rot)) {
    detail::fail(op, "expected x [..., L, heads, D] and table [L, " + std::to_string(rot) + "], got " + x.describe() +
                         " and " + table.describe());
  }
  ApplyArgs a{x.ptr<const bf16>(), detail::row_map(x), out.ptr<bf16>(), detail::row_map(out),
              table.ptr<const bf16>(), detail::row_map(table), detail::rows_of(x), x.dim(-2), x.dim(-3),
              static_cast<int>(x.dim(-1))};
  if (a.rows == 0) return;
  const dim3 block(static_cast<unsigned>(rot == kMossRopeDim ? 64 : 128), kRowsPerBlock);
  if (rot == kMossRopeDim) {
    rope_apply_kernel<kMossRopeDim><<<blocks_for(a.rows), block, 0, stream>>>(a);
  } else {
    rope_apply_kernel<kQwenRopeDim><<<blocks_for(a.rows), block, 0, stream>>>(a);
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void moss_rope_table(const Tensor& out, int height, int width, int text_length, cudaStream_t stream) {
  constexpr const char* op = "moss_rope_table";
  detail::check_rows(op, "out", out, DType::kBF16, kMossRopeDim);
  const std::int64_t n = static_cast<std::int64_t>(height) * width + text_length;
  if (height < 1 || width < 1 || text_length < 0 || out.rank != 2 || out.dim(0) != n) {
    detail::fail(op, "expected out [" + std::to_string(n) + ", 128], got " + out.describe());
  }
  moss_table_kernel<<<blocks_for(n), dim3(kMossRopeDim, kRowsPerBlock), 0, stream>>>(
      out.ptr<bf16>(), detail::row_map(out), n, height, width, text_length, freqs_from_bits(kMossRopeBandBits));
  MI_CUDA_CHECK_LAUNCH();
}

void moss_rope_apply(const Tensor& x, const Tensor& table, const Tensor& out, cudaStream_t stream) {
  apply("moss_rope_apply", kMossRopeDim, x, table, out, stream);
}

void qwen_rope_table(const Tensor& out, cudaStream_t stream) {
  constexpr const char* op = "qwen_rope_table";
  detail::check_rows(op, "out", out, DType::kBF16, kQwenRopeDim);
  if (out.rank != 2) detail::fail(op, "expected out [S, 64], got " + out.describe());
  const std::int64_t n = out.dim(0);
  if (n == 0) return;
  qwen_table_kernel<<<blocks_for(n), dim3(kQwenRopeDim, kRowsPerBlock), 0, stream>>>(
      out.ptr<bf16>(), detail::row_map(out), n, freqs_from_bits(kQwenRopeInvFreqBits));
  MI_CUDA_CHECK_LAUNCH();
}

void qwen_rope_apply(const Tensor& x, const Tensor& table, const Tensor& out, cudaStream_t stream) {
  apply("qwen_rope_apply", kQwenRopeDim, x, table, out, stream);
}

}  // namespace mi::kernels
