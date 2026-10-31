// CUTLASS 2.x implicit-GEMM conv2d fprop (NHWC activations, KRSC filters) with a fused bias + optional
// residual epilogue. GEMM view: M = N*H*W output pixels, N = K output channels, K = R*S*C.
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/sm75/sm75.h"
#include "kernels/vae/vae_kernels.h"

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace mi::vae {
namespace {

using cutlass::gemm::GemmShape;

// Z = (acc + bias) + residual, the reference's rounding order (cuDNN adds the bias, then ResnetBlock2D adds
// input_tensor). CUTLASS's LinearCombinationBiasElementwise evaluates (acc + residual) + bias; this subclass
// only replaces the with-source operator. beta = 1 with a residual, else 0 (the epilogue then skips reading the
// source tensor and the base's no-source path computes acc + bias). Bias stays fp32 in every mode. The alpha/beta
// copies here ignore the base's set_k_partition, so this epilogue requires split-K = 1 (problem() fixes that).
template <typename T, int kAlignOut>
class BiasThenResidual
    : public cutlass::epilogue::thread::LinearCombinationBiasElementwise<
          T, float, float, T, T, kAlignOut, cutlass::epilogue::thread::Identity<float>, cutlass::plus<float>,
          /*StoreT=*/false, /*ElementVector=*/float> {
 public:
  using Base = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      T, float, float, T, T, kAlignOut, cutlass::epilogue::thread::Identity<float>, cutlass::plus<float>, false, float>;
  using typename Base::FragmentAccumulator;
  using typename Base::FragmentC;
  using typename Base::FragmentCompute;
  using typename Base::FragmentT;
  using typename Base::FragmentZ;
  using typename Base::Params;
  using Base::operator();

  CUTLASS_HOST_DEVICE explicit BiasThenResidual(Params const& p) : Base(p), alpha_(p.alpha), beta_(p.beta) {}

  CUTLASS_HOST_DEVICE void operator()(FragmentZ& frag_z, FragmentT&, FragmentAccumulator const& ab, FragmentC const& frag_c,
                                      FragmentCompute const& v) const {
    const FragmentCompute acc = cutlass::NumericArrayConverter<float, float, kAlignOut>()(ab);
    const FragmentCompute res = cutlass::NumericArrayConverter<float, T, kAlignOut>()(frag_c);
    FragmentCompute z;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kAlignOut; ++i) z[i] = (alpha_ * acc[i] + v[i]) + beta_ * res[i];
    frag_z = cutlass::NumericArrayConverter<T, float, kAlignOut>()(z);
  }

 private:
  float alpha_, beta_;
};

template <typename T, typename TileShape, typename WarpShape, typename InstShape, int kStages, int kAlignOut>
struct ConvConfig {
  static constexpr int kAlignIn = 128 / cutlass::sizeof_bits<T>::value;
  // OpMultiplyAdd on float operands = TF32 mma with operands rounded to nearest (round_half_ulp_truncate,
  // PreferredRoundingMode on sm < 90), accumulating in fp32.
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
      T, cutlass::layout::TensorNHWC, T, cutlass::layout::TensorNHWC, T, cutlass::layout::TensorNHWC, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TileShape, WarpShape, InstShape,
      BiasThenResidual<T, kAlignOut>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, kStages,
      cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kUnity, kAlignIn, kAlignIn>::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

// Tiles for sm_89 (RTX 4090 Laptop). Outputs with K >= 64 channels: fp32 128x128 with K-tile 16 (this tile
// reproduces cuDNN's TF32 results bit for bit; K-tile 32 is ~10% faster but differs by ~1e-6 rel), bf16 256x128.
// Narrow outputs (post_quant_conv K = 32, conv_out K = 3): 128x32 with scalar output stores. A 96-wide N tile
// would suit the 96/192-channel stages, but CUTLASS's conv iterators give wrong results for N tiles that are not
// powers of two.
using F32Wide = ConvConfig<float, GemmShape<128, 128, 16>, GemmShape<64, 64, 16>, GemmShape<16, 8, 8>, 4, 4>;
using F32Narrow = ConvConfig<float, GemmShape<128, 32, 16>, GemmShape<64, 32, 16>, GemmShape<16, 8, 8>, 4, 1>;
using Bf16Wide = ConvConfig<cutlass::bfloat16_t, GemmShape<256, 128, 32>, GemmShape<64, 64, 32>, GemmShape<16, 8, 16>,
                            3, 8>;
using Bf16Narrow = ConvConfig<cutlass::bfloat16_t, GemmShape<128, 32, 32>, GemmShape<64, 32, 32>,
                              GemmShape<16, 8, 16>, 4, 1>;

struct ConvShape {
  int n, h, w, c, k, r;
};

ConvShape check_shapes(const Tensor& x, const Tensor& w) {
  if (x.rank != 4 || w.rank != 4 || x.device != Device::kCUDA || w.device != Device::kCUDA)
    throw std::invalid_argument("conv2d: x [N,H,W,C] and w [K,R,S,C] must be rank-4 CUDA tensors");
  if (x.dtype != w.dtype || (x.dtype != DType::kF32 && x.dtype != DType::kBF16))
    throw std::invalid_argument("conv2d: x and w must both be f32 or both bf16");
  if (!x.is_contiguous() || !w.is_contiguous()) throw std::invalid_argument("conv2d: tensors must be contiguous");
  ConvShape s{static_cast<int>(x.dim(0)), static_cast<int>(x.dim(1)), static_cast<int>(x.dim(2)),
              static_cast<int>(x.dim(3)), static_cast<int>(w.dim(0)), static_cast<int>(w.dim(1))};
  if (w.dim(2) != s.r || (s.r != 1 && s.r != 3) || w.dim(3) != s.c)
    throw std::invalid_argument("conv2d: weight must be [K,R,R,C] with R in {1,3}, got " + w.describe() + " for x " +
                                x.describe());
  const int align = x.dtype == DType::kF32 ? 4 : 8;
  if (s.c % align != 0) throw std::invalid_argument("conv2d: C must be a multiple of " + std::to_string(align));
  return s;
}

cutlass::conv::Conv2dProblemSize problem(const ConvShape& s) {
  const int pad = (s.r - 1) / 2;
  return cutlass::conv::Conv2dProblemSize(s.n, s.h, s.w, s.c, s.k, s.r, s.r, s.h, s.w, pad, pad, 1, 1, 1, 1,
                                          cutlass::conv::Mode::kCrossCorrelation, 1, 1);
}

template <typename Cfg>
typename Cfg::Op::Arguments make_args(const ConvShape& s, const Tensor& x, const Tensor& w, const Tensor& bias,
                                      const Tensor* residual, const Tensor& y) {
  using Element = typename Cfg::Op::ElementA;
  using Layout = cutlass::layout::TensorNHWC;
  const auto ps = problem(s);
  auto* xp = static_cast<Element*>(x.data);
  auto* wp = static_cast<Element*>(w.data);
  auto* yp = static_cast<Element*>(y.data);
  auto* rp = residual ? static_cast<Element*>(residual->data) : yp;  // never read when beta == 0
  cutlass::TensorRef<Element, Layout> ref_x(xp, Layout::packed({s.n, s.h, s.w, s.c}));
  cutlass::TensorRef<Element, Layout> ref_w(wp, Layout::packed({s.k, s.r, s.r, s.c}));
  cutlass::TensorRef<Element, Layout> ref_r(rp, Layout::packed({s.n, s.h, s.w, s.k}));
  cutlass::TensorRef<Element, Layout> ref_y(yp, Layout::packed({s.n, s.h, s.w, s.k}));
  typename Cfg::Op::EpilogueOutputOp::Params epi(1.0f, residual ? 1.0f : 0.0f);
  return typename Cfg::Op::Arguments(ps, ref_x, ref_w, ref_r, ref_y, epi, cutlass::conv::SplitKMode::kSerial,
                                     bias.data, nullptr, 0, 0);
}

void check_cutlass(cutlass::Status st, const char* what) {
  if (st != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("conv2d: CUTLASS ") + what + " failed: " + cutlassGetStatusString(st));
}

template <typename Cfg>
std::size_t workspace_for(const ConvShape& s, const Tensor& x, const Tensor& w) {
  Tensor dummy_bias = Tensor::view(nullptr, DType::kF32, Device::kCUDA, {s.k});
  Tensor dummy_y = Tensor::view(nullptr, x.dtype, Device::kCUDA, {s.n, s.h, s.w, s.k});
  return Cfg::Op::get_workspace_size(make_args<Cfg>(s, x, w, dummy_bias, nullptr, dummy_y));
}

template <typename Cfg>
void run(const ConvShape& s, const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual,
         const Tensor& y, void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  // cp.async reads x/w in 16-byte vectors; the epilogue reads bias and y/residual kAlignOut elements at a time.
  const auto aligned = [](const void* p, std::size_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; };
  constexpr std::size_t kOutBytes = Cfg::Op::EpilogueOutputOp::kCount * sizeof(typename Cfg::Op::ElementC);
  if (!aligned(x.data, 16) || !aligned(w.data, 16) || !aligned(y.data, kOutBytes) || !aligned(bias.data, 32) ||
      (residual && !aligned(residual->data, kOutBytes)))
    throw std::invalid_argument("conv2d: x/w need 16-byte, bias 32-byte and y/residual vector alignment");
  auto args = make_args<Cfg>(s, x, w, bias, residual, y);
  check_cutlass(Cfg::Op::can_implement(args), "can_implement");
  if (Cfg::Op::get_workspace_size(args) > workspace_bytes) throw std::invalid_argument("conv2d: workspace too small");
  typename Cfg::Op op;
  check_cutlass(op.initialize(args, workspace, stream), "initialize");
  check_cutlass(op.run(stream), "run");
  MI_CUDA_CHECK_LAUNCH();
}

bool wide(const ConvShape& s, DType t) { return s.k >= 64 && s.k % (t == DType::kF32 ? 4 : 8) == 0; }

// Calls fn.template operator()<Config>() for the config that runs shape s.
template <typename Fn>
auto dispatch(const ConvShape& s, DType t, Fn&& fn) {
  if (t == DType::kF32) return wide(s, t) ? fn.template operator()<F32Wide>() : fn.template operator()<F32Narrow>();
  return wide(s, t) ? fn.template operator()<Bf16Wide>() : fn.template operator()<Bf16Narrow>();
}

}  // namespace

std::size_t conv2d_workspace_bytes(const Tensor& x, const Tensor& w) {
  const ConvShape s = check_shapes(x, w);
  if (x.dtype == DType::kF32 && kernels::sm75::fp16_weights()) return 0;  // the sm_75 fp32 kernel needs none
  return dispatch(s, x.dtype, [&]<typename Cfg>() { return workspace_for<Cfg>(s, x, w); });
}

void conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
            void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  const ConvShape s = check_shapes(x, w);
  if (bias.dtype != DType::kF32 || bias.numel() != s.k || bias.device != Device::kCUDA)
    throw std::invalid_argument("conv2d: bias must be f32 [K] on CUDA");
  auto check_out = [&](const Tensor& t, const char* name) {
    if (t.dtype != x.dtype || t.rank != 4 || t.dim(0) != s.n || t.dim(1) != s.h || t.dim(2) != s.w || t.dim(3) != s.k ||
        !t.is_contiguous() || t.device != Device::kCUDA)
      throw std::invalid_argument(std::string("conv2d: ") + name + " must be contiguous [N,H,W,K] of x's dtype, got " +
                                  t.describe());
  };
  check_out(y, "y");
  if (residual) check_out(*residual, "residual");
  if (y.data == x.data) throw std::invalid_argument("conv2d: y must not alias x");
  if (kernels::sm75::fp16_weights()) {
    if (x.dtype != DType::kF32) throw std::runtime_error("conv2d: the sm_75 path runs the VAE in fp32 (reference mode)");
    kernels::sm75::vae_conv2d(x, w, bias, residual, y, stream);
    return;
  }
  dispatch(s, x.dtype, [&]<typename Cfg>() { run<Cfg>(s, x, w, bias, residual, y, workspace, workspace_bytes, stream); });
}

namespace {

template <typename Out>
__device__ Out cvt_out(float v);
template <>
__device__ float cvt_out<float>(float v) { return v; }
template <>
__device__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// dst[k][r][s][c] = src[k][c][r][s]
template <typename Out>
__global__ void kcrs_to_krsc_kernel(const float* __restrict__ src, Out* __restrict__ dst, int k, int c, int rs) {
  const std::int64_t total = static_cast<std::int64_t>(k) * c * rs;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int ci = static_cast<int>(i % c);
    const std::int64_t t = i / c;
    const int p = static_cast<int>(t % rs);
    const std::int64_t ki = t / rs;
    dst[i] = cvt_out<Out>(src[(ki * c + ci) * rs + p]);
  }
}

}  // namespace

void conv_weight_to_krsc(const Tensor& w_kcrs, const Tensor& w_krsc, cudaStream_t stream) {
  if (w_kcrs.rank != 4 || w_kcrs.dtype != DType::kF32 || w_krsc.rank != 4 || w_kcrs.device != Device::kCUDA ||
      w_krsc.device != Device::kCUDA || !w_kcrs.is_contiguous() || !w_krsc.is_contiguous())
    throw std::invalid_argument("conv_weight_to_krsc: expects contiguous f32 [K,C,R,S] and [K,R,S,C] CUDA tensors");
  const int k = static_cast<int>(w_kcrs.dim(0)), c = static_cast<int>(w_kcrs.dim(1));
  const int r = static_cast<int>(w_kcrs.dim(2)), s = static_cast<int>(w_kcrs.dim(3));
  if (w_krsc.dim(0) != k || w_krsc.dim(1) != r || w_krsc.dim(2) != s || w_krsc.dim(3) != c)
    throw std::invalid_argument("conv_weight_to_krsc: shape mismatch " + w_kcrs.describe() + " -> " + w_krsc.describe());
  const std::int64_t total = w_kcrs.numel();
  if (kernels::sm75::fp16_weights()) {
    kernels::sm75::vae_weight_to_krsc(w_kcrs, w_krsc, stream);
    return;
  }
  const int blocks = static_cast<int>(std::min<std::int64_t>((total + 255) / 256, 4096));
  if (total == 0) return;
  if (w_krsc.dtype == DType::kF32) {
    kcrs_to_krsc_kernel<<<blocks, 256, 0, stream>>>(w_kcrs.ptr<float>(), w_krsc.ptr<float>(), k, c, r * s);
  } else if (w_krsc.dtype == DType::kBF16) {
    kcrs_to_krsc_kernel<<<blocks, 256, 0, stream>>>(w_kcrs.ptr<float>(), w_krsc.ptr<__nv_bfloat16>(), k, c, r * s);
  } else {
    throw std::invalid_argument("conv_weight_to_krsc: destination must be f32 or bf16");
  }
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::vae
