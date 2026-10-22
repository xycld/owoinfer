// Convolutions for the VAE encoder: CUTLASS 2.x implicit-GEMM fprop (NHWC activations, KRSC filters) with a fused
// bias (+ residual) epilogue, in two math modes:
//   kTf32    operands rounded to nearest TF32, fp32 accumulation: the decoder's conv (conv2d.cu) for stride 1, and
//            the same tile and K order for the stride-2 downsample.
//   kTf32x3  OpMultiplyAddFastF32: each operand split into big + small TF32 parts, accumulating small*big +
//            big*small + big*big (fp32-accurate, ~3x the tensor-core work).
// Stride 2 implements Downsample2D(padding = 0): one zero row/column at the bottom and right, 3x3 stride-2 conv.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/cuda_check.h"
#include "kernels/vae/encoder_kernels.h"

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace mi::vae {
namespace {

using cutlass::gemm::GemmShape;
constexpr int kAlignOut = 4;

// Z = (acc + bias) + residual, the reference's rounding order (as conv2d.cu's epilogue: CUTLASS's
// LinearCombinationBiasElementwise evaluates (acc + residual) + bias, so only the with-source operator is replaced).
// beta = 1 with a residual, else 0 (no source read; the base computes acc + bias). Requires split-K = 1.
class BiasThenResidual
    : public cutlass::epilogue::thread::LinearCombinationBiasElementwise<
          float, float, float, float, float, kAlignOut, cutlass::epilogue::thread::Identity<float>, cutlass::plus<float>,
          /*StoreT=*/false, /*ElementVector=*/float> {
 public:
  using Base = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      float, float, float, float, float, kAlignOut, cutlass::epilogue::thread::Identity<float>, cutlass::plus<float>, false,
      float>;
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
    FragmentCompute z;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kAlignOut; ++i) z[i] = (alpha_ * ab[i] + v[i]) + beta_ * frag_c[i];
    frag_z = z;
  }

 private:
  float alpha_, beta_;
};

// The decoder's fp32 tile (128x128x16, 4 stages). TF32: 64x64 warps, as the decoder. 3xTF32 keeps big and small
// operand fragments, which spills with 64x64 warp tiles; 64x32 warps (8 per CTA) fit the registers.
template <typename MathOp, cutlass::conv::StrideSupport kStride>
struct Conv {
  static constexpr bool kX3 = std::is_same_v<MathOp, cutlass::arch::OpMultiplyAddFastF32>;
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
      float, cutlass::layout::TensorNHWC, float, cutlass::layout::TensorNHWC, float, cutlass::layout::TensorNHWC, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, GemmShape<128, 128, 16>,
      std::conditional_t<kX3, GemmShape<64, 32, 16>, GemmShape<64, 64, 16>>,
      GemmShape<16, 8, 8>, BiasThenResidual, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 4, MathOp,
      cutlass::conv::IteratorAlgorithm::kOptimized, kStride, 4, 4>::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};
using SameX3 = Conv<cutlass::arch::OpMultiplyAddFastF32, cutlass::conv::StrideSupport::kUnity>;
using DownTf32 = Conv<cutlass::arch::OpMultiplyAdd, cutlass::conv::StrideSupport::kStrided>;
using DownX3 = Conv<cutlass::arch::OpMultiplyAddFastF32, cutlass::conv::StrideSupport::kStrided>;

struct Shape {
  int n, h, w, c, k, r, p, q, stride;
};

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument(msg);
}

Shape check(const char* op, const Tensor& x, const Tensor& w, const Tensor& y, int stride) {
  for (const Tensor* t : {&x, &w, &y})
    require(t->rank == 4 && t->dtype == DType::kF32 && t->device == Device::kCUDA && t->is_contiguous(),
            std::string(op) + ": x, w, y must be contiguous rank-4 f32 CUDA tensors, got " + t->describe());
  const Shape s{static_cast<int>(x.dim(0)), static_cast<int>(x.dim(1)), static_cast<int>(x.dim(2)),
                static_cast<int>(x.dim(3)), static_cast<int>(w.dim(0)), static_cast<int>(w.dim(1)),
                static_cast<int>(y.dim(1)), static_cast<int>(y.dim(2)), stride};
  require(w.dim(2) == s.r && w.dim(3) == s.c && (stride == 1 ? (s.r == 1 || s.r == 3) : s.r == 3),
          std::string(op) + ": bad weight " + w.describe() + " for x " + x.describe());
  require(s.c % 4 == 0 && s.k % 4 == 0 && s.k >= 64, std::string(op) + ": need C % 4 == 0, K % 4 == 0, K >= 64");
  require(y.dim(0) == s.n && y.dim(3) == s.k, std::string(op) + ": y must be [N,P,Q,K], got " + y.describe());
  if (stride == 1)
    require(s.p == s.h && s.q == s.w, std::string(op) + ": y must be [N,H,W,K], got " + y.describe());
  else
    require(s.p >= 1 && s.q >= 1 && 2 * s.p <= s.h + 1 && 2 * s.q <= s.w + 1,
            std::string(op) + ": y must be [N,P,Q,K] with 1 <= P <= (H+1)/2, 1 <= Q <= (W+1)/2, got " + y.describe() +
                " for x " + x.describe());
  return s;
}

// Stride 1: "same" padding. Stride 2: padding 0 with an explicit output size, CUTLASS masks the reads past row H-1 /
// column W-1, which supplies the reference's bottom/right zero pad.
template <typename Cfg>
typename Cfg::Op::Arguments make_args(const Shape& s, const Tensor& x, const Tensor& w, const Tensor& bias,
                                      const Tensor* residual, const Tensor& y) {
  using Layout = cutlass::layout::TensorNHWC;
  const int pad = s.stride == 1 ? (s.r - 1) / 2 : 0;
  const cutlass::conv::Conv2dProblemSize ps(s.n, s.h, s.w, s.c, s.k, s.r, s.r, s.p, s.q, pad, pad, s.stride, s.stride, 1, 1,
                                            cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  auto* yp = static_cast<float*>(y.data);
  auto* rp = residual ? static_cast<float*>(residual->data) : yp;  // never read when beta == 0
  cutlass::TensorRef<float, Layout> ref_x(static_cast<float*>(x.data), Layout::packed({s.n, s.h, s.w, s.c}));
  cutlass::TensorRef<float, Layout> ref_w(static_cast<float*>(w.data), Layout::packed({s.k, s.r, s.r, s.c}));
  cutlass::TensorRef<float, Layout> ref_r(rp, Layout::packed({s.n, s.p, s.q, s.k}));
  cutlass::TensorRef<float, Layout> ref_y(yp, Layout::packed({s.n, s.p, s.q, s.k}));
  typename Cfg::Op::EpilogueOutputOp::Params epi(1.0f, residual ? 1.0f : 0.0f);
  return typename Cfg::Op::Arguments(ps, ref_x, ref_w, ref_r, ref_y, epi, cutlass::conv::SplitKMode::kSerial, bias.data,
                                     nullptr, 0, 0);
}

template <typename Cfg>
std::size_t workspace_for(const Shape& s, const Tensor& x, const Tensor& w, const Tensor& y) {
  const Tensor bias = Tensor::view(nullptr, DType::kF32, Device::kCUDA, {s.k});
  return Cfg::Op::get_workspace_size(make_args<Cfg>(s, x, w, bias, nullptr, y));
}

void check_cutlass(cutlass::Status st, const char* op, const char* what) {
  if (st != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string(op) + ": CUTLASS " + what + " failed: " + cutlassGetStatusString(st));
}

template <typename Cfg>
void run(const char* name, const Shape& s, const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual,
         const Tensor& y, void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
  require(bias.dtype == DType::kF32 && bias.numel() == s.k && bias.device == Device::kCUDA,
          std::string(name) + ": bias must be f32 [K] on CUDA");
  if (residual)
    require(residual->dtype == DType::kF32 && residual->is_contiguous() && residual->rank == 4 && residual->dim(0) == s.n &&
                residual->dim(1) == s.p && residual->dim(2) == s.q && residual->dim(3) == s.k,
            std::string(name) + ": residual must match y");
  require(y.data != x.data, std::string(name) + ": y must not alias x");
  const auto aligned = [](const void* p, std::uintptr_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; };
  require(aligned(x.data, 16) && aligned(w.data, 16) && aligned(y.data, 16) && aligned(bias.data, 32) &&
              (residual == nullptr || aligned(residual->data, 16)),
          std::string(name) + ": x/w/y/residual need 16-byte and bias 32-byte alignment");
  require(x.nbytes() < (std::size_t{1} << 31) && y.nbytes() < (std::size_t{1} << 31),
          std::string(name) + ": tensors must be smaller than 2 GiB (CUTLASS 32-bit offsets)");
  auto args = make_args<Cfg>(s, x, w, bias, residual, y);
  check_cutlass(Cfg::Op::can_implement(args), name, "can_implement");
  require(Cfg::Op::get_workspace_size(args) <= workspace_bytes, std::string(name) + ": workspace too small");
  typename Cfg::Op op;
  check_cutlass(op.initialize(args, workspace, stream), name, "initialize");
  check_cutlass(op.run(stream), name, "run");
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

std::size_t encoder_conv2d_workspace_bytes(const Tensor& x, const Tensor& w, ConvMath math) {
  if (math == ConvMath::kTf32) return conv2d_workspace_bytes(x, w);
  const Tensor y = Tensor::view(nullptr, DType::kF32, Device::kCUDA, {x.dim(0), x.dim(1), x.dim(2), w.dim(0)});
  return workspace_for<SameX3>(check("encoder_conv2d", x, w, y, 1), x, w, y);
}

void encoder_conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
                    void* workspace, std::size_t workspace_bytes, ConvMath math, cudaStream_t stream) {
  if (math == ConvMath::kTf32) {
    conv2d(x, w, bias, residual, y, workspace, workspace_bytes, stream);
    return;
  }
  run<SameX3>("encoder_conv2d", check("encoder_conv2d", x, w, y, 1), x, w, bias, residual, y, workspace, workspace_bytes,
              stream);
}

std::size_t conv2d_down_workspace_bytes(const Tensor& x, const Tensor& w, const Tensor& y, ConvMath math) {
  const Shape s = check("conv2d_down", x, w, y, 2);
  return math == ConvMath::kTf32 ? workspace_for<DownTf32>(s, x, w, y) : workspace_for<DownX3>(s, x, w, y);
}

void conv2d_down(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor& y, void* workspace,
                 std::size_t workspace_bytes, ConvMath math, cudaStream_t stream) {
  const Shape s = check("conv2d_down", x, w, y, 2);
  if (math == ConvMath::kTf32)
    run<DownTf32>("conv2d_down", s, x, w, bias, nullptr, y, workspace, workspace_bytes, stream);
  else
    run<DownX3>("conv2d_down", s, x, w, bias, nullptr, y, workspace, workspace_bytes, stream);
}

}  // namespace mi::vae
