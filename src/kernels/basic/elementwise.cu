#include "kernels/basic/elementwise.h"

#include <initializer_list>
#include <type_traits>
#include <utility>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"

namespace mi::kernels {
namespace {

using detail::RowMap;
using bf16 = __nv_bfloat16;

template <typename T>
struct In {
  const T* p;
  RowMap rows;
  __device__ const T* row(std::int64_t r) const { return p + rows.offset(r); }
};

template <typename T>
struct Out {
  T* p;
  RowMap rows;
  __device__ T* row(std::int64_t r) const { return p + rows.offset(r); }
};

template <typename T>
In<T> in(const Tensor& t) { return {t.ptr<const T>(), detail::row_map(t)}; }
template <typename T>
Out<T> out_of(const Tensor& t) { return {t.ptr<T>(), detail::row_map(t)}; }

// torch divides by a CPU scalar as a multiply by its fp32 reciprocal (div_true_kernel_cuda).
constexpr float kInv25 = 1.0f / 25.0f;

__device__ __forceinline__ float situ_value(float h, float g) {
  using namespace detail;
  const float left = rbf(mul(rbf(mul(4.0f, rbf(tanhf(rbf(mul(g, 0.25f)))))), rbf(sigmoid_f32(g))));
  const float right = rbf(mul(25.0f, rbf(tanhf(rbf(mul(h, kInv25))))));
  return mul(left, right);
}

// Each op resolves its row pointers once, then the block's threads stride over the columns.
#define MI_FOR_COLS for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)

struct SituFused {
  In<bf16> x;
  Out<bf16> y;
  std::int64_t half;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    const bf16* xr = x.row(r);
    bf16* yr = y.row(r);
    MI_FOR_COLS yr[c] = detail::bf(situ_value(detail::f(xr[c]), detail::f(xr[half + c])));
  }
};

struct Situ {
  In<bf16> h, g;
  Out<bf16> y;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    const bf16 *hr = h.row(r), *gr = g.row(r);
    bf16* yr = y.row(r);
    MI_FOR_COLS yr[c] = detail::bf(situ_value(detail::f(hr[c]), detail::f(gr[c])));
  }
};

struct GatedResidual {
  In<bf16> x, y;
  const bf16* gate;
  Out<bf16> o;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    using namespace detail;
    const bf16 *xr = x.row(r), *yr = y.row(r);
    bf16* orow = o.row(r);
    MI_FOR_COLS orow[c] = bf(add(f(xr[c]), rbf(mul(f(yr[c]), f(gate[c])))));
  }
};

struct ResidualAdd {
  In<bf16> x, y;
  Out<bf16> o;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    using namespace detail;
    const bf16 *xr = x.row(r), *yr = y.row(r);
    bf16* orow = o.row(r);
    MI_FOR_COLS orow[c] = bf(add(f(xr[c]), f(yr[c])));
  }
};

struct Silu {
  In<bf16> x;
  Out<bf16> o;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    const bf16* xr = x.row(r);
    bf16* orow = o.row(r);
    MI_FOR_COLS orow[c] = detail::bf(detail::silu_f32(detail::f(xr[c])));
  }
};

struct SiluMul {
  In<bf16> g, u;
  Out<bf16> o;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    using namespace detail;
    const bf16 *gr = g.row(r), *ur = u.row(r);
    bf16* orow = o.row(r);
    MI_FOR_COLS orow[c] = bf(mul(rbf(silu_f32(f(gr[c]))), f(ur[c])));
  }
};

struct SigmoidMul {
  In<bf16> x, g;
  Out<bf16> o;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    using namespace detail;
    const bf16 *xr = x.row(r), *gr = g.row(r);
    bf16* orow = o.row(r);
    MI_FOR_COLS orow[c] = bf(mul(f(xr[c]), rbf(sigmoid_f32(f(gr[c])))));
  }
};

template <typename A, typename B>
struct Cast {
  In<A> x;
  Out<B> o;
  __device__ void operator()(std::int64_t r, std::int64_t cols) const {
    const A* xr = x.row(r);
    B* orow = o.row(r);
    MI_FOR_COLS {
      if constexpr (std::is_same_v<A, B>) {
        orow[c] = xr[c];
      } else if constexpr (std::is_same_v<B, float>) {
        orow[c] = __bfloat162float(xr[c]);
      } else {
        orow[c] = __float2bfloat16_rn(xr[c]);
      }
    }
  }
};

#undef MI_FOR_COLS

template <class Op>
__global__ void rows_kernel(Op op, std::int64_t rows, std::int64_t cols) {
  for (std::int64_t r = blockIdx.x; r < rows; r += gridDim.x) op(r, cols);
}

template <class Op>
void run(const Op& op, std::int64_t rows, std::int64_t cols, cudaStream_t stream) {
  if (rows == 0 || cols == 0) return;
  rows_kernel<<<detail::grid_for_rows(rows), 256, 0, stream>>>(op, rows, cols);
  MI_CUDA_CHECK_LAUNCH();
}

// Checks a set of same-shape bf16 [..., C] operands and returns C.
std::int64_t check_same(const char* op, std::initializer_list<std::pair<const char*, const Tensor*>> ts) {
  const Tensor& first = *ts.begin()->second;
  for (const auto& [name, t] : ts) {
    detail::check_rows(op, name, *t, DType::kBF16);
    detail::check_same_shape(op, first, *t);
  }
  return first.dim(-1);
}

}  // namespace

void situ_glu(const Tensor& up_gate, const Tensor& out, cudaStream_t stream) {
  constexpr const char* op = "situ_glu";
  detail::check_rows(op, "up_gate", up_gate, DType::kBF16);
  const std::int64_t f = up_gate.dim(-1) / 2;
  detail::check_rows(op, "out", out, DType::kBF16, f);
  if (up_gate.dim(-1) != 2 * f || up_gate.rank != out.rank || detail::rows_of(up_gate) != detail::rows_of(out)) {
    detail::fail(op, "expected up_gate [..., 2F] and out [..., F], got " + up_gate.describe() + " and " + out.describe());
  }
  run(SituFused{in<bf16>(up_gate), out_of<bf16>(out), f}, detail::rows_of(out), f, stream);
}

void situ(const Tensor& hidden, const Tensor& gate, const Tensor& out, cudaStream_t stream) {
  const std::int64_t c = check_same("situ", {{"hidden", &hidden}, {"gate", &gate}, {"out", &out}});
  run(Situ{in<bf16>(hidden), in<bf16>(gate), out_of<bf16>(out)}, detail::rows_of(out), c, stream);
}

void gated_residual(const Tensor& x, const Tensor& y, const Tensor& gate, const Tensor& out, cudaStream_t stream) {
  const std::int64_t c = check_same("gated_residual", {{"x", &x}, {"y", &y}, {"out", &out}});
  detail::check_vector("gated_residual", "gate", gate, c);
  run(GatedResidual{in<bf16>(x), in<bf16>(y), gate.ptr<const bf16>(), out_of<bf16>(out)}, detail::rows_of(out), c,
      stream);
}

void residual_add(const Tensor& x, const Tensor& y, const Tensor& out, cudaStream_t stream) {
  const std::int64_t c = check_same("residual_add", {{"x", &x}, {"y", &y}, {"out", &out}});
  run(ResidualAdd{in<bf16>(x), in<bf16>(y), out_of<bf16>(out)}, detail::rows_of(out), c, stream);
}

void silu(const Tensor& x, const Tensor& out, cudaStream_t stream) {
  const std::int64_t c = check_same("silu", {{"x", &x}, {"out", &out}});
  run(Silu{in<bf16>(x), out_of<bf16>(out)}, detail::rows_of(out), c, stream);
}

void silu_mul(const Tensor& gate, const Tensor& up, const Tensor& out, cudaStream_t stream) {
  const std::int64_t c = check_same("silu_mul", {{"gate", &gate}, {"up", &up}, {"out", &out}});
  run(SiluMul{in<bf16>(gate), in<bf16>(up), out_of<bf16>(out)}, detail::rows_of(out), c, stream);
}

void sigmoid_mul(const Tensor& x, const Tensor& gate, const Tensor& out, cudaStream_t stream) {
  const std::int64_t c = check_same("sigmoid_mul", {{"x", &x}, {"gate", &gate}, {"out", &out}});
  run(SigmoidMul{in<bf16>(x), in<bf16>(gate), out_of<bf16>(out)}, detail::rows_of(out), c, stream);
}

void cast(const Tensor& x, const Tensor& out, cudaStream_t stream) {
  constexpr const char* op = "cast";
  const auto ok = [](DType t) { return t == DType::kBF16 || t == DType::kF32; };
  if (!ok(x.dtype) || !ok(out.dtype)) detail::fail(op, "only bf16 and f32 are supported, got " + x.describe() + " -> " + out.describe());
  detail::check_rows(op, "in", x, x.dtype);
  detail::check_rows(op, "out", out, out.dtype);
  detail::check_same_shape(op, x, out);
  const std::int64_t rows = detail::rows_of(out), c = out.dim(-1);
  if (x.dtype == DType::kBF16 && out.dtype == DType::kF32) {
    run(Cast<bf16, float>{in<bf16>(x), out_of<float>(out)}, rows, c, stream);
  } else if (x.dtype == DType::kF32 && out.dtype == DType::kBF16) {
    run(Cast<float, bf16>{in<float>(x), out_of<bf16>(out)}, rows, c, stream);
  } else if (x.dtype == DType::kF32) {
    run(Cast<float, float>{in<float>(x), out_of<float>(out)}, rows, c, stream);
  } else {
    run(Cast<bf16, bf16>{in<bf16>(x), out_of<bf16>(out)}, rows, c, stream);
  }
}

}  // namespace mi::kernels
