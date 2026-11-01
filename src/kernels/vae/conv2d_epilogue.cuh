// Shared VAE convolution epilogue; used by Tensor Core and fp32 SIMT implementations.
#pragma once

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"

namespace mi::vae::detail {

// Z = (acc + bias) + residual, the reference's rounding order (cuDNN adds the bias, then ResnetBlock2D adds
// input_tensor). CUTLASS's LinearCombinationBiasElementwise evaluates (acc + residual) + bias; this subclass
// only replaces the with-source operator. beta = 1 with a residual, else 0 (the epilogue then skips reading the
// source tensor and the base's no-source path computes acc + bias). Bias stays fp32 in every mode. The alpha/beta
// copies here ignore the base's set_k_partition, so this epilogue requires split-K = 1 (both convolution callers fix that).
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

}  // namespace mi::vae::detail
