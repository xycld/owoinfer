// Per-head QK RMSNorm + RoPE of the MOSS DiT in one pass over a fused QKV projection output (P:140-141 then
// _attention's _rotate, P:77-86), in place. Bit-exact against kernels::rms_norm_moss on the q and k head views
// followed by kernels::moss_rope_apply on each: same fp32 reduction order (torch's reduce shape for the
// per-head norm, 32 threads x 4 values per 128-d head) and the same bf16 roundings.
#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace mi::kernels::fused {

// qkv: CUDA bf16 [L, C] rows with a contiguous last dim and one row stride, C >= 2 * heads * 128; q heads at
// columns [0, heads*128), k heads at [heads*128, 2*heads*128) (v after them is not touched).
// q_norm, k_norm: bf16 [128] weights. table: bf16 [L, 128] RoPE rows of these L tokens (moss_rope_table).
// 16-byte aligned rows. No workspace; graph-capturable.
void qk_norm_rope_moss(const Tensor& qkv, int heads, const Tensor& q_norm, const Tensor& k_norm, const Tensor& table,
                       cudaStream_t stream);

}  // namespace mi::kernels::fused
