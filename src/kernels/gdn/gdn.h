// Qwen3.5 GatedDeltaNet prefill kernels (no conv/recurrent state cache, no decode): causal depthwise
// conv1d + SiLU, the beta/g gates, and the chunked gated delta rule with a slow token-by-token oracle.
// Numerics follow the transformers pure-torch fallback the goldens used (docs/model-notes.md §7):
// conv in bf16 (fp32 accumulate, rounded, then SiLU rounded), gates in fp32, the rule entirely in fp32.
//
// Varlen: `cu_seqlens` is an i32 [N+1] device tensor of token offsets (cu[0] = 0, cu[N] = total tokens T),
// as in flash-attention varlen. Every sequence is independent and starts from a zero state.
//
// !! Contract (see core/varlen.h; build the values with Varlen::from_lengths): cu[0] = 0, non-decreasing,
// !! cu[N] == T, and the host `max_seqlen` passed to chunk_gated_delta_rule must be >= the longest sequence
// !! (it sizes the grid; a smaller value silently drops the tail of longer sequences). Not checked by
// !! default; -DMOSSINFER_DEBUG_CHECKS or set_varlen_debug_checks(true) validates it (host copy + stream sync).
// !! Callers must size every tensor T == cu[N]. Rows >= cu[N], if T is larger, are processed by
// !! causal_conv1d_silu / gates as if they continued the last sequence (conv reads history, gates are
// !! per token) and are left unwritten by chunk_gated_delta_rule, so `out` there is undefined.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mi::gdn {

inline constexpr int kNumKHeads = 16;
inline constexpr int kNumVHeads = 32;  // q/k heads are repeat_interleave'd: value head h reads key head h / 2
inline constexpr int kHeadDim = 128;   // key and value head dim
inline constexpr int kKeyDim = kNumKHeads * kHeadDim;
inline constexpr int kValueDim = kNumVHeads * kHeadDim;
inline constexpr int kConvDim = 2 * kKeyDim + kValueDim;  // 8192 channels laid out [q | k | v]
inline constexpr int kConvKernel = 4;
inline constexpr int kChunk = 64;

// y = SiLU(conv(x)), causal per sequence. x: bf16 [T, 8192] (row stride may exceed 8192, a multiple of 8),
// weight: bf16 [8192, 4] (conv1d.weight squeezed), y: bf16 [T, 8192] (row stride a multiple of 8).
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& cu_seqlens, const Tensor& y,
                        cudaStream_t stream);

// beta = sigmoid(b) (bf16, as the reference keeps it), g = -exp(A_log) * softplus(a + dt_bias) (fp32).
// a, b: bf16 [T, 32] (any row stride), a_log, dt_bias: bf16 [32], g: f32 [T, 32], beta: bf16 [T, 32].
void gates(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias, const Tensor& g,
           const Tensor& beta, cudaStream_t stream);

// Scratch for chunk_gated_delta_rule over `total_tokens` tokens (fp32 intermediates, about 74 KB per token).
std::size_t chunk_gated_delta_rule_workspace_bytes(std::int64_t total_tokens);

// Chunked gated delta rule (chunk 64, q/k L2-normalised in kernel, q scaled by 1/sqrt(128)), fp32 throughout.
// qkv: bf16 [T, 8192] conv output (row stride a multiple of 8), g: f32 [T, 32], beta: bf16 [T, 32],
// out: bf16 [T, 32, 128] contiguous. final_state, if non-null: f32 [N, 32, 128, 128] contiguous ([k][v] per head).
// max_seqlen: the longest sequence (host value, sizes the grid).
void chunk_gated_delta_rule(const Tensor& qkv, const Tensor& g, const Tensor& beta, const Tensor& cu_seqlens,
                            int max_seqlen, const Tensor& out, const Tensor* final_state, void* workspace,
                            std::size_t workspace_bytes, cudaStream_t stream);

// Test oracle: the same rule computed token by token in fp32 (torch_recurrent_gated_delta_rule's order).
// Slow (one block per head and sequence). out: f32 [T, 32, 128]; final_state as above.
void recurrent_gated_delta_rule_reference(const Tensor& qkv, const Tensor& g, const Tensor& beta,
                                          const Tensor& cu_seqlens, const Tensor& out, const Tensor* final_state,
                                          cudaStream_t stream);

}  // namespace mi::gdn
