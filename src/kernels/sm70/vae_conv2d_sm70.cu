// Volta's reference VAE stays fp32. CUTLASS SIMT implicit GEMM reuses larger spatial/channel tiles than the
// compatibility stencil, retaining fp32 operands, accumulation and the shared bias-then-residual epilogue.
#include "kernels/sm70/vae_sm70.h"

#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/vae/conv2d_epilogue.cuh"

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace mi::kernels::sm70 {
namespace {

using cutlass::gemm::GemmShape;
using Layout = cutlass::layout::TensorNHWC;

template <int N, int WarpN>
struct ConvConfig {
  using Kernel = typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
      float, Layout, float, Layout, float, Layout, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      GemmShape<128, N, 8>, GemmShape<32, WarpN, 8>, GemmShape<1, 1, 1>, vae::detail::BiasThenResidual<float, 1>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2, cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized, cutlass::conv::StrideSupport::kUnity, 4, 4>::Kernel;
  using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
};

void check_cutlass(cutlass::Status status, const char* what) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(std::string("sm70::vae_conv2d: CUTLASS ") + what + " failed: " + cutlassGetStatusString(status));
}

template <typename Cfg>
void run(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
         cudaStream_t stream) {
  const int n = static_cast<int>(x.dim(0)), h = static_cast<int>(x.dim(1)), width = static_cast<int>(x.dim(2));
  const int c = static_cast<int>(x.dim(3)), k = static_cast<int>(w.dim(0)), r = static_cast<int>(w.dim(1));
  const int pad = (r - 1) / 2;
  const cutlass::conv::Conv2dProblemSize ps(n, h, width, c, k, r, r, h, width, pad, pad, 1, 1, 1, 1,
                                           cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  const cutlass::TensorRef<float, Layout> rx(x.ptr<float>(), Layout::packed({n, h, width, c}));
  const cutlass::TensorRef<float, Layout> rw(w.ptr<float>(), Layout::packed({k, r, r, c}));
  const cutlass::TensorRef<float, Layout> ry(y.ptr<float>(), Layout::packed({n, h, width, k}));
  const cutlass::TensorRef<float, Layout> rr(residual ? residual->ptr<float>() : y.ptr<float>(), ry.layout());
  const typename Cfg::Op::EpilogueOutputOp::Params epi(1.f, residual ? 1.f : 0.f);
  const typename Cfg::Op::Arguments args(ps, rx, rw, rr, ry, epi, cutlass::conv::SplitKMode::kSerial,
                                         bias.data, nullptr, 0, 0);
  check_cutlass(Cfg::Op::can_implement(args), "can_implement");
  // Serial split-K=1 uses no semaphores. Keep the compatibility API's zero-workspace contract explicit.
  if (Cfg::Op::get_workspace_size(args) != 0) throw std::runtime_error("sm70::vae_conv2d: unexpected workspace");
  typename Cfg::Op op;
  check_cutlass(op.initialize(args, nullptr, stream), "initialize");
  check_cutlass(op.run(stream), "run");
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void vae_conv2d(const Tensor& x, const Tensor& w, const Tensor& bias, const Tensor* residual, const Tensor& y,
                cudaStream_t stream) {
  if (x.numel() == 0 || y.numel() == 0) return;
  if (w.dim(0) >= 64) run<ConvConfig<128, 64>>(x, w, bias, residual, y, stream);
  else run<ConvConfig<32, 32>>(x, w, bias, residual, y, stream);
}

}  // namespace mi::kernels::sm70
