// Exact (non-approximate) scaled-dot-product attention forward on FlashAttention-2 kernels vendored from
// Dao-AILab/flash-attention v2.8.3 (third_party/flash-attention). fp32 online softmax, bf16 in/out, no
// dropout, no mask beyond optional causal. Requires sm_80 or newer (flash_fwd throws std::runtime_error on an
// older device).
//
// Layout (varlen, "unpadded"): sequences are packed back to back along dim 0.
//   q, out: bf16 [total_q, num_heads,    head_dim]
//   k, v:   bf16 [total_k, num_kv_heads, head_dim]
//   cu_seqlens_q, cu_seqlens_k: int32 [batch + 1] on the device, prefix sums of the sequence lengths
//   (cu[0] = 0, cu[batch] = total). Sequence b occupies rows [cu[b], cu[b+1]).
// Only head_dim must be contiguous (stride 1); row and head strides are free (multiples of 8 elements,
// 16-byte aligned base), so q/k/v can be strided views into a fused QKV projection output.
// GQA: num_heads % num_kv_heads == 0; query head h reads KV head h / (num_heads / num_kv_heads), i.e.
// contiguous groups, matching transformers' repeat_kv and SDPA enable_gqa.
//
// Supported configurations (the MOSS-Image ones; anything else throws std::invalid_argument):
//   head_dim 128, non-causal  (DiT joint attention, text refiner)
//   head_dim 256, causal      (Qwen3.5 full-attention layers)
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "core/tensor.h"

namespace mi::attention {

// Scratch bytes flash_fwd needs: the per-row softmax log-sum-exp the kernels always write, fp32
// [num_heads, total_q].
std::size_t flash_fwd_workspace_bytes(std::int64_t total_q, int num_heads);

// out = softmax(q·kᵀ·scale [+ causal mask])·v per sequence and head. Heads come from the tensor shapes
// (num_heads = q.dim(1), num_kv_heads = k.dim(1)).
// !! max_seqlen MUST be >= every sequence length in cu_seqlens_q and cu_seqlens_k. It is a host value that
// !! sizes the grid; a smaller value silently drops the tail of the longer sequences (no error, wrong output).
// !! cu_seqlens must start at 0, be non-decreasing and end at q.dim(0) / k.dim(0). Nothing here reads the
// !! device arrays on the host, so the contract is NOT checked by default: build it with core/varlen.h
// !! (Varlen::from_lengths) or enable the check with -DMOSSINFER_DEBUG_CHECKS / set_varlen_debug_checks(true)
// !! (copies cu_seqlens to the host and synchronizes `stream`; skipped during graph capture).
// causal: query i attends to keys j <= i within its sequence (torch SDPA is_causal, top-left aligned).
// This is self-attention: each sequence's q and k lengths must then be equal (FlashAttention aligns the
// mask bottom-right when they differ).
// workspace: device memory of at least flash_fwd_workspace_bytes(total_q, num_heads), 4-byte aligned.
// No host synchronization; graph-capturable.
void flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
               const Tensor& cu_seqlens_k, int max_seqlen, bool causal, float scale, void* workspace,
               std::size_t workspace_bytes, cudaStream_t stream);

// Test-only: force one FA2 instantiation instead of the smem-based choice, so tests can exercise all four.
// A forced tile must match head_dim and fit the device's opt-in shared memory per block, otherwise flash_fwd
// throws std::invalid_argument. Per-block needs: hdim128 128x64 64 KiB, hdim128 128x32 48 KiB,
// hdim256 64x64 96 KiB, hdim256 128x64 128 KiB. sm_89 (RTX 40xx) allows 99 KiB, so hdim256 128x64 can only
// run on sm_80/90-class smem (A100 163 KiB, H100 227 KiB). Process-wide; kAuto restores normal dispatch.
enum class FlashTile { kAuto, kHdim128_128x64, kHdim128_128x32, kHdim256_128x64, kHdim256_64x64 };
void flash_fwd_set_tile_override_for_testing(FlashTile tile);

}  // namespace mi::attention
