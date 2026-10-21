// One MOSS DiT block (_Block.forward, P:136-148) with the elementwise ops fused into their neighbours (P5 step 29),
// bit-identical to the op-by-op MossDit::run_block (tested on real weights and activations, and at every run length of
// small images: tests/unit/test_dit_block_fused, test_fused_gemm):
//   y   = rms_norm_modulate(x, norm1, scale_a, shift_a)                 kernels/basic, unchanged
//   qkv = y · Wqkvᵀ                                                     kernels::Gemm (cuBLASLt)
//   q,k = rope(rms_norm(q|k))  per head, in place                       fused::qk_norm_rope_moss (one pass)
//   attn = flash_fwd(q, k, v)                                           unchanged
//   x  += bf16(bf16(attn · Woᵀ) * gate_a)                               fused::gemm_gated_residual (epilogue)
//   y   = rms_norm_modulate(x, norm2, scale_f, shift_f)                 kernels/basic, unchanged
//   ffn = situ(bf16(y · Wupᵀ), bf16(y · Wgateᵀ))                        fused::gemm_situ_glu (epilogue)
//   x  += bf16(bf16(ffn · Wdownᵀ) * gate_f)                             fused::gemm_gated_residual (epilogue)
// Every GEMM output is rounded to bf16 before the op that follows, exactly where the reference materializes a bf16
// tensor. The CUTLASS GEMMs reproduce cuBLASLt's accumulation order only where cuBLASLt runs a K-ordered algorithm
// without split-K (kernels/fused/fused_gemm.h); fused_route() asks cuBLASLt which configuration the op-by-op linear
// gets for the run's row count, runs fused only the configurations the startup self-check validated on this device,
// and the other linears op by op (cuBLASLt + the kernels/basic op, on the same buffers as MossDit::run_block). On
// sm_89 the op-by-op ones are mostly ffn_down below 384 rows: small images, e.g. 128^2 with a short prompt. The
// plain QKV projection has nothing to fuse and stays on cuBLASLt (the CUTLASS tiles are no faster there). The
// [L, 2*kFfnDim] ffn_up_gate output is never stored while ffn_up_gate runs fused, so the activations need only the
// smaller big buffer of fused_big_cols().
// No host synchronization and no allocation: graph-capturable.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "kernels/basic/gemm.h"
#include "models/moss_dit/dit_weights.h"

namespace mi::moss_dit {

// Elements per token of the activation buffer the fused block shares between phases, laid out as MossDit's: the
// contiguous [L, 3*kDim] qkv then [L, kDim] attn during attention, the contiguous [L, kFfnDim] ffn during the FFN.
// A run whose ffn_up_gate goes op by op needs the op-by-op block's [L, 2*kFfnDim] up_gate then [L, kFfnDim] ffn.
inline constexpr std::int64_t kFusedBigCols = 4 * 3072;
inline constexpr std::int64_t kOpByOpBigCols = 3 * 8192;

// One cuBLASLt configuration of one op's op-by-op linear ("o_proj", "ffn_up_gate" or "ffn_down") under which the
// self-check found the fused pair bit-identical, at `rows` rows (Gemm::AlgoConfig::workspace_bytes is not part of it).
struct FusedConfig {
  std::string op;
  kernels::Gemm::AlgoConfig config;
  std::int64_t rows = 0;
};

// Which of the block's GEMM + epilogue pairs run fused for a run of `rows` tokens on the current device: those whose
// op-by-op linear cuBLASLt runs with a configuration (Gemm::AlgoConfig: algorithm, tile, stages, inner shape, swizzle,
// split-K, reduction, custom option, cluster) that the self-check installed for this device validated for that op, and
// that kernels::fused::matches_cublaslt accepts. Every other pair runs op by op, and all of them do on a device without
// an installed check. Asks cuBLASLt with MossDit's operand layout (contiguous rows, 256-byte aligned,
// Gemm::kWorkspaceBytes); cached per device and row count, thread-safe.
struct FusedRoute {
  bool o = true;        // o-proj + gated residual
  bool up_gate = true;  // ffn_up_gate + SiTU-GLU
  bool down = true;     // ffn_down + gated residual
};
FusedRoute fused_route(std::int64_t rows);
// The same with `validated` in place of the installed configurations (tests).
FusedRoute fused_route(std::int64_t rows, std::span<const FusedConfig> validated);

// Big-buffer elements per token of a fused forward over rows of `hw + text_lengths[r]` tokens: kOpByOpBigCols when
// some set of these rows (a run of the rows that execute a block) gets ffn_up_gate op by op, else kFusedBigCols.
std::int64_t fused_big_cols(std::int64_t hw, std::span<const std::int64_t> text_lengths);

// Startup self-check (once per GPU, library version and kernels::fused::kFusedKernelsVersion, see
// pipeline/hardware_profile.h). Asks cuBLASLt (no launches) for each op's configuration at every realistic run length
// (1..61506 rows: every count up to 1536, then sampled), collects the distinct ones matches_cublaslt accepts, and runs
// each once on synthetic inputs, fused and op by op, at the smallest row count that gets it (a configuration whose
// operands would exceed ~0.25 GiB stays unvalidated: op by op). ok iff at least one configuration was checked and every
// checked one is bit-identical; then `validated` lists them. detail says what was checked, what differed, or that
// nothing would run fused. Throws CudaError when it cannot allocate its operands. `hook` (tests) may alter each fused
// result before it is compared.
struct FusedCheck {
  bool ok = false;
  std::string detail;
  std::vector<FusedConfig> validated;
};
using FusedCheckHook = std::function<void(const char* op, std::int64_t rows, const Tensor& fused_out)>;
FusedCheck check_fused_gemms(cudaStream_t stream, const FusedCheckHook& hook = {});

// Lets fused_route() run the configurations of a passed check (check.ok) fused on the current device, added to any
// installed before (a set only grows: workspaces sized for an earlier route stay large enough). A failed check
// installs nothing.
void install_fused_check(const FusedCheck& check);

// Activations of one forward (views; rows = the joint tokens of every row of the step).
struct FusedBlockBuffers {
  Tensor x;     // bf16 [L, kDim] residual stream, updated in place
  Tensor y;     // bf16 [L, kDim] scratch
  Tensor big;   // bf16 [L, fused_big_cols(...)] contiguous
  Tensor rope;  // bf16 [L, 128] RoPE rows of every token
  Tensor mod;   // bf16 [6 * kDim]: shift_a, scale_a, gate_a, shift_f, scale_f, gate_f (P:139)
  void* lse = nullptr;  // flash_fwd workspace for L tokens
  std::size_t lse_bytes = 0;
};

// Tokens [t0, t1) of a run of contiguous rows that execute the block, as one varlen batch: cu_seqlens int32
// [rows + 1] from 0 on the device, max_seqlen >= every row length.
struct BlockRun {
  std::int64_t t0 = 0, t1 = 0;
  Tensor cu_seqlens;
  int max_seqlen = 0;
};

// The model's cuBLASLt handle and workspace: the QKV projection and the linears fused_route() runs op by op.
struct QkvGemm {
  kernels::Gemm* gemm = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
};

// The block on run `run` of `a` with weights `w` (BlockTensors as loaded: up_gate is the stored [hidden | gate]).
// Operands must be 256-byte aligned (arena allocations, kArenaAlign weights): fused_route() assumes so.
void run_block_fused(const BlockTensors& w, const FusedBlockBuffers& a, const BlockRun& run, const QkvGemm& qkv_gemm,
                     cudaStream_t stream);

}  // namespace mi::moss_dit
