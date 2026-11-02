// NVIDIA verify build only (verify_kernels.cu).
#pragma once

#include <stdexcept>

namespace hip_block {

// Over all 2^32 fp32 inputs of bf and all 2^16 bf16 inputs of f: inputs whose result differs from the intrinsic's, not
// counting NaNs, and NaNs whose bits differ.
struct ConversionCheck {
  unsigned long long round_differ, round_nan_bits, widen_differ, widen_nan_bits;
};
ConversionCheck check_conversions();

}  // namespace hip_block
