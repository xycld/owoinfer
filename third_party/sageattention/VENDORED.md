# Vendored SageAttention (8-bit attention kernels, CUDA headers only)

- Upstream: https://github.com/thu-ml/SageAttention, commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 (2026-01-17).
- License: Apache-2.0 (`LICENSE`, copyright 2024 Jintao Zhang, Haofeng Huang / the SageAttention team). Every
  modified file carries a "MossInfer modifications" notice at its top, as section 4(b) of the license requires; so
  does `src/kernels/attention_q8/q8_preprocess.cu`, which is derived from upstream's `csrc/fused/fused.cu` and keeps
  its copyright and license header.
- NOTICE: upstream ships no NOTICE file at this commit (`LICENSE` is the only license-type file in its whole tree),
  so section 4(d) adds nothing to propagate. Some kept headers carry third-party
  copyrights of their own (FlashInfer team 2023: `cp_async.cuh`, `math.cuh`, `mma.cuh`, `permuted_smem.cuh`; vLLM
  team 2023 and NVIDIA 2020-2023, from FasterTransformer: `reduction_utils.cuh`), all Apache-2.0, kept as is.
- Used by `src/kernels/attention_q8/` (the torch-free launchers, workspace layout and quantization pre-pass). Include
  path set per source in `cmake/modules/attention_q8.cmake`.

## Kept

From `csrc/`:

| File | Status |
|---|---|
| `cp_async.cuh`, `mma.cuh`, `permuted_smem.cuh`, `math.cuh`, `numeric_conversion.cuh`, `reduction_utils.cuh` | unchanged |
| `qattn/attn_utils.cuh` | modified (below) |
| `qattn/qk_int_sv_f8_cuda_sm89.cuh` | modified (below): the sm89 INT8-QK / FP8-PV kernel `qk_int_sv_f8_attn_kernel` |
| `qattn/qk_int_sv_f16_cuda_sm80.cuh` | modified (below): the kernel `qk_int_sv_f16_attn_kernel` of `qattn/qk_int_sv_f16_cuda_sm80.cu` (first 672 lines), renamed `.cuh` |

Removed: everything else. That covers the PyTorch extension code (`utils.cuh`, `dispatch_utils.h`, every `pybind_*.cpp`,
the torch host wrappers of the `.cu` files, `attn_cuda_sm*.h`), the sm90 / wgmma kernels, the per-variant sm89
instantiation `.cu` files, `fused/` (its quantization kernels are re-implemented, see below), the Triton kernels, the
Python package, setup and examples.

## Local changes

`qattn/attn_utils.cuh`:
1. No PyTorch: the `torch/extension.h` and `../utils.cuh` includes are removed.
2. The predicated `load_global_to_share` takes the shared-memory fill mode as a defaulted template argument
   (default `kNoFill`, upstream's behaviour).
3. `update_mdo` takes the FP8 exponent offset as an argument (default `S_FP8_OFFSET`, upstream's 8.807 = log2 448).
   It lets the fp16-accumulating FP8 kernel run with the narrowed P range of SageAttention2++.

`qattn/qk_int_sv_f8_cuda_sm89.cuh` and `qattn/qk_int_sv_f16_cuda_sm80.cuh`:
1. No PyTorch: the `torch/extension.h`, `../utils.cuh` and `../dispatch_utils.h` includes are removed. The sm80 file
   also loses the torch host wrappers (everything after the kernel).
2. Varlen. The batch strides and the fixed `qo_len` / `kv_len` arguments become `cu_seqlens_q` / `cu_seqlens_k` (int32
   `[batch + 1]`).
   - Each CTA reads its sequence's start and lengths and offsets its Q/K/V/O pointers by the sequence start.
   - CTAs whose query tile starts past the sequence end return before any barrier. The grid covers the longest sequence.
   - A sequence with no keys writes zero output rows.
   - Upstream's predicates and its out-of-bound mask on the last K tile then stop at each sequence's own end, so no key
     of another sequence reaches the softmax.
3. The K-scale count per batch is the new `k_scale_blocks` argument instead of `div_ceil(kv_len, CTA_K)`. Scales are
   laid out for the longest sequence.
4. `return_lse` is rejected by `static_assert`, since MossInfer never uses the log-sum-exp. The upstream LSE code is
   left in place, unreachable.
5. sm89 (FP8 PV):
   - V is read from the padded per-sequence layout `[num_kv_heads, head_dim, padded]`. Sequence b starts at column
     `sum_{i<b} roundup(len_i, 64)`, which the CTA computes from `cu_seqlens_k`.
   - The exponent offset (log2 of the P range) is the `p_offset` argument, passed to `update_mdo`.
6. sm80 (fp16 PV): the two predicated V tile loads (the tile holding the sequence end) zero-fill rows past `kv_len`
   (`kFillZero`). Upstream does not fill them. Stale shared memory there can hold NaN/Inf bit patterns, and the masked
   P = 0 times NaN reaches the output. `GpuAttentionQ8.ShortSequencesMatchFp32Reference` poisons shared memory with
   NaN and fails without this change.

`mma.cuh` is unchanged, so its compiler gate stands: the fp16-accumulating e4m3 MMA needs nvcc 12.8, and below that
(and on targets below sm_89) the wrappers compile to `__brkpt`. `attention_q8` therefore refuses the affected
variants at run time (see `attention_q8.h`), and `cmake/modules/attention_q8.cmake` warns for nvcc < 12.8.

Nothing else changed: tiling, swizzles, MMA sequences, online softmax, the fp16 "instruction buffer" accumulation
(SageAttention2 / 2++) and the output path are upstream's.

## Re-implemented from `csrc/fused/fused.cu` (in `src/kernels/attention_q8/q8_preprocess.cu`)

These are varlen ports of `QuantInt8Kernel` (per-warp Q and per-block K INT8, with the K mean subtracted) and of
`TransposePadPermuteKernel` + `MeanScaleKernel` (per-channel e4m3 V, transposed, padded to 64 and permuted within each
group of 16 keys as 0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15). Differences:
- Blocks are per sequence, from `cu_seqlens`.
- The K mean, which upstream computes with torch `k.mean()`, is computed on the device. Per-chunk fp32 sums are
  reduced in a fixed order, so the pass is deterministic, and the result is rounded to bf16 as upstream's mean tensor
  is.
- The V transpose and the e4m3 quantization are fused into one kernel, so there is no bf16 transposed intermediate.
- An all-zero V channel gets scale 1. Upstream divides by its amax of 0 and writes NaN.

## Matching upstream's math (checked by `GpuAttentionQ8.MatchesUpstreamSageAttention`)

- FP8 exponent offset: `log2(p_range)` truncated to 3 decimals, which gives upstream's 8.807 for 448.
- V quantization multiplier: `v_range / amax`, the same expression as upstream.
- The K mean is rounded to bf16 as described above.

With these, the outputs of all four instantiations are 94–96% bit-identical to upstream's (built from this commit),
with cos ≥ 0.999993. The remaining differences come from the K-mean summation order and upstream's `--use_fast_math`
division.

## Instantiations (`src/kernels/attention_q8/q8_attn_*.cu`)

All use CTA_Q 128, CTA_K 64, WARP_K 64, head_dim 128, per-warp Q and K granularity, bf16 output, and no causal mask.
Each row gives the upstream entry point it reproduces.

| MossInfer variant | Upstream entry point |
|---|---|
| `kInt8Fp16` | `qk_int8_sv_f16_accum_f16_attn_inst_buf` (WARP_Q 16; fp16 accumulation per 64-key tile, flushed into fp32) |
| `kInt8Fp16F32Acc` | `qk_int8_sv_f16_accum_f32_attn` (WARP_Q 32) |
| `kInt8Fp8` | `qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf` (WARP_Q 32; `mma.sync.m16n8k32.f16.e4m3.e4m3.f16`, fp16 over 2 MMAs, flushed into fp32). Default P/V ranges 112 / 4.5; upstream's are 448 / 2.25 |
| `kInt8Fp8F32Acc` | `qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf` (WARP_Q 32; P/V ranges 448 / 448) |
