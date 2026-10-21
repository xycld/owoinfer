#include "kernels/basic/norm.h"

#include <algorithm>

#include "core/cuda_check.h"
#include "kernels/basic/detail.h"

namespace mi::kernels {
namespace {

using detail::RowMap;

// torch's launch shape for a last-dim fp32 reduction of `rows` rows of length d (setReduceConfig in
// ATen/native/cuda/Reduce.cuh, vectorize_input path, vt0 = input_vec_size = 4, 512 threads max):
// bw threads cooperate on one row, bh rows per block.
struct ReduceShape {
  int bw;
  int bh;
};

int torch_last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

ReduceShape torch_reduce_shape(const char* op, std::int64_t rows, std::int64_t d) {
  if (d % 4 != 0 || d < 128) detail::fail(op, "row length must be a multiple of 4 and >= 128, got " + std::to_string(d));
  constexpr int kMaxThreads = 512;
  const std::int64_t dim0 = d / 4;  // vectorized along the reduced dim
  const int dim0_pow2 = dim0 < kMaxThreads ? torch_last_pow2(static_cast<int>(dim0)) : kMaxThreads;
  const int dim1_pow2 = rows < kMaxThreads ? torch_last_pow2(static_cast<int>(rows)) : kMaxThreads;
  int bw = std::min(dim0_pow2, 32);
  const int bh = std::min(dim1_pow2, kMaxThreads / bw);
  bw = std::min(dim0_pow2, kMaxThreads / bh);
  // torch splits a row across the block's warps once each thread would sum >= min(16*bh, 256) values.
  const std::int64_t values_per_thread = (d + bw - 1) / bw;
  if (values_per_thread >= std::min(bh * 16, 256)) {
    detail::fail(op, "row length " + std::to_string(d) + " needs torch's cross-warp split, not supported");
  }
  return {bw, bh};
}

enum class Variant { kMoss, kMossModulate, kQwen, kQwenGated };

struct NormArgs {
  const __nv_bfloat16* x;
  RowMap x_rows;
  __nv_bfloat16* out;
  RowMap out_rows;
  const __nv_bfloat16* gate;
  RowMap gate_rows;
  const __nv_bfloat16* weight;
  const __nv_bfloat16* scale;
  const __nv_bfloat16* shift;
  std::int64_t rows;
  int d;
  float factor;  // torch MeanOps factor: float(num_outputs) / numel
  float eps;
};

// Block (bw, bh): row group ty handles row blockIdx.x * bh + ty with bw threads.
template <Variant V>
__global__ void rms_norm_kernel(NormArgs a) {
  using namespace detail;
  __shared__ float red[512];
  __shared__ float inv_rms[16];
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int bw = static_cast<int>(blockDim.x);
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.y + ty;
  const bool valid = row < a.rows;
  const __nv_bfloat16* xr = a.x + (valid ? a.x_rows.offset(row) : 0);

  // thread_reduce (input_vectorized_thread_reduce_impl): 4 accumulators over 4-element vectors.
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (valid) {
    for (int idx = tx; idx * 4 + 3 < a.d; idx += bw) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float v = f(xr[idx * 4 + i]);
        acc[i] = add(acc[i], mul(v, v));  // x.float().square() is its own rounded op
      }
    }
  }
  float v = add(add(add(acc[0], acc[1]), acc[2]), acc[3]);

  // block_x_reduce: shared-memory halving down to one warp, then a shuffle-down tree.
  if (bw > 32) {
    const int base = tx + ty * bw;
    red[base] = v;
    for (int off = bw / 2; off >= 32; off >>= 1) {
      __syncthreads();
      if (tx < off) {
        v = add(v, red[base + off]);
        red[base] = v;
      }
    }
  }
  __syncthreads();
  for (int off = 16; off > 0; off >>= 1) v = add(v, __shfl_down_sync(0xffffffffu, v, off));
  if (tx == 0) inv_rms[ty] = rsqrtf(add(mul(v, a.factor), a.eps));
  __syncthreads();
  if (!valid) return;

  const float r = inv_rms[ty];
  __nv_bfloat16* outr = a.out + a.out_rows.offset(row);
  for (int c = tx; c < a.d; c += bw) {
    const float xn = mul(f(xr[c]), r);
    float y;
    if constexpr (V == Variant::kMoss || V == Variant::kMossModulate) {
      y = mul(rbf(xn), f(a.weight[c]));  // y.to(x.dtype) * weight
      if constexpr (V == Variant::kMossModulate) {
        const float one_plus = rbf(add(1.0f, f(a.scale[c])));
        y = add(rbf(mul(rbf(y), one_plus)), f(a.shift[c]));
      }
    } else if constexpr (V == Variant::kQwen) {
      y = mul(xn, add(1.0f, f(a.weight[c])));
    } else {
      const float g = f(a.gate[a.gate_rows.offset(row) + c]);
      y = mul(rbf(mul(f(a.weight[c]), rbf(xn))), silu_f32(g));
    }
    outr[c] = bf(y);
  }
}

template <Variant V>
void launch(const char* op, const Tensor& x, const Tensor* gate, const Tensor& weight, const Tensor* scale,
            const Tensor* shift, const Tensor& out, float eps, cudaStream_t stream) {
  detail::check_rows(op, "x", x, DType::kBF16);
  const std::int64_t d = x.dim(-1);
  detail::check_rows(op, "out", out, DType::kBF16, d);
  detail::check_same_shape(op, x, out);
  detail::check_vector(op, "weight", weight, d);
  NormArgs a{};
  if (gate != nullptr) {
    detail::check_rows(op, "gate", *gate, DType::kBF16, d);
    detail::check_same_shape(op, x, *gate);
    a.gate = gate->ptr<const __nv_bfloat16>();
    a.gate_rows = detail::row_map(*gate);
  }
  if (scale != nullptr) {
    detail::check_vector(op, "scale", *scale, d);
    detail::check_vector(op, "shift", *shift, d);
    a.scale = scale->ptr<const __nv_bfloat16>();
    a.shift = shift->ptr<const __nv_bfloat16>();
  }
  a.rows = detail::rows_of(x);
  if (a.rows == 0) return;
  const ReduceShape shape = torch_reduce_shape(op, a.rows, d);
  a.x = x.ptr<const __nv_bfloat16>();
  a.x_rows = detail::row_map(x);
  a.out = out.ptr<__nv_bfloat16>();
  a.out_rows = detail::row_map(out);
  a.weight = weight.ptr<const __nv_bfloat16>();
  a.d = static_cast<int>(d);
  a.factor = static_cast<float>(a.rows) / static_cast<float>(a.rows * d);
  a.eps = eps;
  const auto blocks = static_cast<unsigned>((a.rows + shape.bh - 1) / shape.bh);
  rms_norm_kernel<V><<<blocks, dim3(static_cast<unsigned>(shape.bw), static_cast<unsigned>(shape.bh)), 0, stream>>>(a);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void rms_norm_moss(const Tensor& x, const Tensor& weight, const Tensor& out, cudaStream_t stream) {
  launch<Variant::kMoss>("rms_norm_moss", x, nullptr, weight, nullptr, nullptr, out, kRmsNormEps, stream);
}

void rms_norm_modulate_moss(const Tensor& x, const Tensor& weight, const Tensor& scale, const Tensor& shift,
                            const Tensor& out, cudaStream_t stream) {
  launch<Variant::kMossModulate>("rms_norm_modulate_moss", x, nullptr, weight, &scale, &shift, out, kRmsNormEps,
                                 stream);
}

void rms_norm_qwen(const Tensor& x, const Tensor& weight, const Tensor& out, cudaStream_t stream, float eps) {
  launch<Variant::kQwen>("rms_norm_qwen", x, nullptr, weight, nullptr, nullptr, out, eps, stream);
}

void rms_norm_gated_qwen(const Tensor& x, const Tensor& gate, const Tensor& weight, const Tensor& out,
                         cudaStream_t stream, float eps) {
  launch<Variant::kQwenGated>("rms_norm_gated_qwen", x, &gate, weight, nullptr, nullptr, out, eps, stream);
}

}  // namespace mi::kernels
