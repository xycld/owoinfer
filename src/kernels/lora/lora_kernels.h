// Elementwise kernel of the runtime LoRA side branch (models/lora/side_branch.h).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mi::kernels::lora {

// t[i, j] = bf16(t32[i, j] * coef[j]) for i < m, j < r: the down projection x · Aᵀ (fp32, from cuBLASLt) scaled by
// each rank column's coefficient (strength * alpha / rank of its adapter) and rounded once to bf16. Row strides in
// elements. No allocation; graph-capturable. Throws std::invalid_argument on bad arguments.
void scale_columns_bf16(const float* t32, std::int64_t ld32, const float* coef, __nv_bfloat16* t, std::int64_t ldt,
                        std::int64_t m, std::int64_t r, cudaStream_t stream);

}  // namespace mi::kernels::lora
