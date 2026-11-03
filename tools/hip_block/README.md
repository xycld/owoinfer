# hip_block

A standalone HIP port of the bf16 exact path of one MOSS DiT block (`src/models/moss_dit/dit_block_fused.h`): RMSNorm with modulation, the QKV GEMM, per-head QK norm + RoPE, varlen joint attention (head_dim 128, non-causal, fp32 online softmax, bf16 in and out), and the gated-residual and SiTU-GLU GEMMs. It does not use the engine's HAL and the main build does not know this directory.

One kernel source, `block_kernels.hip`, builds three ways. They differ in the leaf in `mma_leaf.h`, a 16x16x16 bf16 multiply with fp32 accumulators:

| Target | Compiler | Leaf |
|---|---|---|
| `gfx1100` (RDNA3) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `gfx1200`, `gfx1201` (RDNA4) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `nvidia` | nvcc with HIP's NVIDIA headers | scalar: the exact bf16*bf16 product in fp32, then add, K ascending |

All kernels launch workgroups of 32 threads (`dim3(32,1,1)`): one wave on AMD, one warp on NVIDIA. Fragments load straight from global memory. `__shared__` memory holds what must cross lanes: accumulators on their way out and through the softmax's per-row rescale, per-row reductions, the RoPE pairing, and zero-padded copies of tails that end inside a fragment. Tile sizes are the named constants at the top of `block_kernels.hip`. `kAttnStepN` is part of the numerics, as it sets where softmax weights are rounded; 32 is the tile the engine runs on sm_89.

## Building

`ROCM_PATH` is the ROCm root holding `include/hip` and `lib/llvm`, as a CMake cache variable or in the environment. Paths are relative to the repository root.

```sh
export ROCM_PATH=/path/to/opt/rocm

# AMD: once per target. Also prints each kernel's WMMA count and resources (check_device.sh).
cmake -S tools/hip_block -B build-hip-gfx1201 -G Ninja -DHIP_BLOCK_TARGET=gfx1201
cmake --build build-hip-gfx1201 -j4
cmake -S tools/hip_block -B build-hip-gfx1100 -G Ninja -DHIP_BLOCK_TARGET=gfx1100
cmake --build build-hip-gfx1100 -j4

# NVIDIA verify build: first the engine's core library from this tree, then the test against it.
cmake -S . -B build-hip-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DMOSSINFER_TESTS=OFF -DMOSSINFER_SM75=OFF
cmake --build build-hip-engine --target mossinfer_core -j4
cmake -S tools/hip_block -B build-hip-nvidia -G Ninja -DHIP_BLOCK_TARGET=nvidia -DMOSSINFER_ENGINE_BUILD_DIR=$PWD/build-hip-engine
cmake --build build-hip-nvidia -j4
```

The AMD builds' device check (`check_device.sh`, run by CMake with the target) fails unless each kernel holds the number of `v_wmma_f32_16x16x16_bf16` it lists for that target. Those counts are what ROCm 7.2.4's clang emits for this source; another ROCm may unroll differently and emit other counts, which then need checking in its assembly and updating in the script.

`build-*/` is ignored by git. ROCm's libraries must be on the loader path at run time if they are not installed system-wide: `LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib`, plus the directory holding `libnuma` and `libdrm` where those are unpacked with ROCm.

## Running on a card

```sh
build-hip-gfx1201/hip_block_bench            # RDNA4; build-hip-gfx1100 for RDNA3
build-hip-gfx1201/hip_block_bench --rows 4378,4106 --iters 5    # the defaults
```

`hip_block_bench` prints the device name, then a known-answer check of every kernel against a host replay of its op order, with tails on every axis, then runs one block, looks for NaNs, and requires the packed path to give the row-major path's bits. Both block runs start with NaN in every buffer the ops write, so a value one path does not write is a NaN in its output, not the other run's. It refuses to time a block that fails any of these. The replay check is the row-major path; `hip_block_replay` below is the one that compares with the engine. Then it prints min and median milliseconds per op and for the whole block, on synthetic 1024²-shaped activations: E13's 8484 tokens, two rows of 4378 and 4106 (about 1.2 GiB of buffers, packed weights and pack buffers included). Exit status: 0 ran, 1 error, known-answer failure or NaN, 2 no usable device.

On a machine where an iGPU is device 0, select the card with `HIP_VISIBLE_DEVICES`: the bench runs on device 0 of what it sees.

A binary runs only on its target's ISA. Another RDNA3 or RDNA4 chip needs its own `--offload-arch`, and rocWMMA must support it.

The known-answer limits, against the replays (`same` is the share of outputs with the replay's bits; ULPs are of max(|replay|, floor), the floor being the output's rms, or rms/64 for the plain GEMM). The baseline is this bench's scalar leaf on the RTX 4090. Nothing here says how much AMD's math library differs from the host's.

`rms_norm_modulate` and `qk_norm_rope` have a limit of 3 ULPs because a low-bit flip of `rsqrt` travels through their rounding points. In a model on the scalar-leaf build with the device `rsqrt` perturbed by 1 fp32 ULP and the host replay left alone, `qk_norm_rope` reached 2.00 ULPs, and failed the first limit of 1 in 21 of 200 runs with 25 % of inputs perturbed and 47 of 200 with 50 %. With all four math functions off by 1 to 2 ULPs on every input, attention reached 0.93 against its limit of 1, so its margin is thin. `same` is what catches a dropped rounding point on these kernels: five of the second review's nine norm, modulate and RoPE rounding mutants stay within 3 ULPs. The cost is that the limit accepts a much worse `rsqrt`: with every input off, no run of 300 failed at up to 8 ULPs, 63 at up to 32, and all at up to 2048. A card is one fixed function, so it either always fails such a limit or never does: a failure with `same` above 99 % points at the math library first.

| Kernel | Limit: same, ULPs | Baseline: same, ULPs |
|---|---|---|
| rms_norm_modulate | 99 %, 3 | 100 %, 0.00 |
| qk_norm_rope | 99 %, 3 | 100 %, 0.12 |
| gemm | 99 %, 1 | 100 %, 0.50 |
| gemm_gated_residual | 99 %, 1 | 100 %, 0.50 |
| gemm_situ_glu | 99 %, 1.5 | 100 %, 0.50 |
| attention | 99 %, 1 | 99.997 %, 0.50 |

## Verify test

`build-hip-nvidia/hip_block_verify` needs a CUDA device with about 2 GiB free. It runs the same inputs through the engine's kernels (`run_block_fused` and the individual kernels, from `libmossinfer_core.a`) and through the prototype with the scalar leaf, and prints per-op error distributions in bf16 ULPs. It checks the kernels' logic above the leaf, and nothing about WMMA or rocWMMA's lane layout. Block rows run the packed path that `run_block` uses, and the op-level shapes the row-major one. Before each of a block row's ops and before its whole block, every buffer the ops write holds NaN, so an output the prototype does not write fails as a NaN on one side rather than passing as the engine's value left there. The cases are shapes with tails on M, N and K; varlen attention with rows of different lengths and with query lengths different from key lengths; edge inputs (a zero row, a NaN, an Inf, denormals, per-row magnitudes across a wide range, attention scores all -Inf); attention's edge inputs again through `run_block_op`, packed and row-major (a NaN or Inf in q, k or v, also at the next sequence's first token, a sequence's keys all zero, every score -Inf in a one-row tile and in a full tile, the last key all NaN), which reach the packer and the rescale's NaN test; and whole blocks over two and four rows.

`hip_block_verify --real --weights DIR --golden DIR` (or `MOSS_WEIGHTS`, `MOSS_GOLDEN`) runs blocks 1, 21, 22, 23 and 41 on their real weights and on activations captured from the engine's forward at the 1024² golden shape, needing about 4 GiB (3.9 measured on the 4090). `DIR` for `--weights` holds `base/moss.mossidx`; `--golden` holds the golden `.npy` files. Blocks stream during the capture, so the DiT need not fit. It also runs the engine's block with one op at a time replaced by the prototype's.

`scaled` is the largest error in ULPs of max(|engine|, rms of the op's output). Raw ULP distances grow without limit on outputs that nearly cancel, since such a sum keeps the absolute error of its large terms. The gates, each also failing on a NaN on one side only:

| Gate | Applies to | Passes when |
|---|---|---|
| exact | `rms_norm_modulate` from 16 rows, `qk_norm_rope` | no bit differences, NaN payload aside |
| accumulated | GEMMs, attention, gated residual, SiTU, `rms_norm_modulate` below 16 rows | scaled <= 4; with at least 1000 values, same >= 99 % and p99.9 <= 1 |
| block, synthetic | whole block on its own outputs | scaled <= 16, same >= 80 %, p99 <= 16 |
| block, real activations | whole block on its own outputs | against the control below, measured in the same run: same >= control same - 5 points, p99 <= 2 x control p99 (at least 16), scaled <= 2 x control scaled (at least 16). A run fails without a control, if the control's QKV fails the accumulated gate against the engine's, or if the control's same is under 50 % |

For gated residual and SiTU, an output above scaled 4 passes only if the op's host evaluation reproduces both sides from a GEMM result within 1 ULP of the exact sum, and an Inf on one side fails. On real activations SiTU's p99.9 limit is 2; on synthetic inputs it stays 1.

These rules changed after results were in, in this order:
1. The first rule (at most 4 raw ULPs on outputs with |engine| >= rms/256) failed on outputs near cancellation.
2. `scaled` replaced it, with limits of 4 per op and 16 for the block, the latter set after seeing 6. Those let dropped rounding points through, which an independent review pointed out.
3. The review added `same` and p99.9, and these exact gates.
4. The flip check above was added after SiTU reached scaled 5 on one value in 7e7 with the block's two-row shape, and the last block's gated residual reached 32 on real activations. Plain scaled <= 4 fails on both.
5. The fixed block gate failed on real activations at blocks 21 to 41, and SiTU's p99.9 reached 2 at block 41. The real-activation block gate above and SiTU's limit of 2 replaced them. These limits were set after seeing those results.
6. A second review showed a wrong control passing: with its QKV zeroed, `--real` exited 0 because the limits opened with it. The control's own gates above were added, with the floor of 50 % set after seeing healthy controls at 69.9 % and up and the zeroed one under 5 %. The same review showed an Inf on one side passing the flip-checked rows, which now fail, and a correct `rsqrt` able to fail the bench's known answer, which raised two ULP limits from 1 to 3.
7. A third review showed outputs the prototype never wrote passing as the engine's values left in the buffers, and whole wrong token rows passing replay. Every buffer the ops write is now poisoned with NaN before each op and block, and replay gained its row and scaled limits (below), their margins chosen after seeing the scalar leaf's figures, two other accumulation orders and the review's row mutants. The scaled limit was first max(8, 2 x the scalar leaf's); the card's earlier block-41 down gemm + gated (64) met that exactly, and no mutant needed it, so it is max(16, 4 x). A later round showed one wrong fragment or column passing replay; the count of outputs above scaled 4 was added, its floor and factor chosen after seeing the scalar leaf's counts, two other accumulation orders' counts and those mutants'.

## On an AMD card

An RX 9060 XT (gfx1200, 32 CUs, 16 GB of GDDR6 on a 128-bit bus, 4 MB L2, 32 MB L3), ROCm 7.16, rocWMMA 2.2.1. One block at E13's 8484 tokens takes 109 ms on synthetic activations (min of 10, median 109) and 104 ms on real weights, against the engine's 31.3 ms on an RTX 4090 Laptop (E13). A 42-block step would take about 4.4 s against 1.33 s. The first build took 415 ms on one machine and 422 ms on another with the same card (Ryzen 5 3500, i7-8700), so the host is not what sets these times.

| Op | First build, ms | Now, ms |
|---|---|---|
| norm1 + norm2 + qk_norm_rope | 2.0 | 2.0 |
| qkv gemm | 77.0 | 9.7 |
| attention | 97.6 | 64.3 |
| o gemm + gated | 18.1 | 3.9 |
| up_gate gemm + SiTU | 149.5 | 18.5 |
| down gemm + gated | 67.3 | 10.4 |
| whole block | 415 | 109 |

On NVIDIA's scalar leaf, none of what took it there changes a bit: the old and new kernels give identical outputs for every op on the dump's real inputs, for both whole blocks, and for tail shapes and denormal attention (the third review's old-against-new harness). On the card, the new kernels pass the gates against the engine below; old against new has not run there. The card's check that the packed path gives the row-major path's bits cannot see a change both paths share: the fence, the all-lane softmax, the rescale skip, the four-wide reads, the block order and the tiles. The changes:

- **Packed operands.** A strided fragment load makes its 32 lanes touch 16 cache lines and a packed one 8, with the same instructions (one 16-byte load per lane). Packing is cheap, 0.3 ms at K 3072 and 0.9 ms at K 8192, when it walks along the row groups. Walking across them took 23 ms at K 8192.
- **All of a step's loads before its WMMAs.** They overlap in flight. A load before each group of WMMAs stalls them: 39.8 TFLOPS against 63.3 at the same tile and block.
- **Tiles in 2D blocks of about 4 MB per operand**, in one dispatch. Tile order cannot change an output, as K still ascends within a tile.
- **`__syncthreads` replaced by a wavefront fence.** It emitted a cache invalidate at every call, and a workgroup here is one wave.
- **Softmax on all 32 lanes** rather than one per row, each row's sum still in column order; four values per read from `__shared__` memory; and no accumulator trip through `__shared__` memory when no row's maximum changed, as a rescale of exactly 1 is then the only thing it could apply.
- **Tiles retuned with all of that in place:** GEMM 32x48, SiTU 48x16.

The QKV walk alone, in throwaway tests with every accumulator stored, 48x64 tiles. "Loads first" issues all of a K step's fragment loads before its WMMAs; "load per group" loads before each group of WMMAs:

| Test | Loads | TFLOPS |
|---|---|---|
| Paper peak: 32 CUs x 1024 FLOP per clock x 3.46 GHz | | about 113 |
| WMMA on registers, tiles 1x1 to 4x4 fragments | | 54 to 96 |
| Strided, raster | first | 9.2 |
| Strided, raster | per group | 8.3 |
| Strided, 2D blocks of 4x4 tiles | first | 15.4 |
| Packed, raster | first | 22.5 |
| Packed, raster | per group | 15.4 |
| Packed, 2D blocks of 13x10 tiles | per group | 39.8 |
| Packed, 2D blocks of 13x10 tiles | first | 63.3 |
| The same with FP8, INT8 operands, speed only (values not meaningful) | first | 93, 100 |

Attention is now most of the block. Removing parts of it, timing only: scores about 10 ms, V about 20, the accumulator trip about 10, and about 25 in the softmax's per-row loops and `__shared__` memory traffic. Replacing `exp2f` with `v_exp_f32` changes little. Attention at 32 rows per wave is slower and spills. Reserving 16 KB more `__shared__` memory changed nothing, which suggests it does not limit waves; I did not confirm the reservation survived compilation.

Not tried: 8-bit in the real kernels, where the bf16 path's rounding points would not carry over; fragments shared through `__shared__` memory by several waves; `swmmac`'s K of 32; packed output from the op before each GEMM, which would remove the pack; summing each softmax row in partial sums across lanes, then across them, as FlashAttention does (per-thread partials, then `quad_allreduce`). That is closer to the engine than the one lane per row in column order now, and likely cheaper. A tree of four in place of the column order, which no test rejects, already moves attention's same against the engine from 99.4171 to 99.4394 % at block 1 and from 99.1279 to 99.1336 % at block 41. It needs the card for timing.

### Against the engine, on the card

`hip_block_verify --real --dump DIR` writes the E13-shaped blocks' weights, inputs and engine outputs (about 1 GB per block), each block's `meta.txt` with the whole block's limits from that run's control and the scalar leaf's own per-op figures against the engine, and last a `manifest.txt` listing the blocks. A run that fails writes no manifest. `hip_block_replay DIR [--iters N]` runs them on any device, needing about 2 GiB. It refuses a directory without a manifest (an interrupted dump, or one of an earlier format), a manifest that lists no block or a block that is missing, and a `meta.txt` of an earlier format (format 3 added the count below). Before each op and before the whole block, every buffer the ops write holds NaN. Each op runs on the engine's input and is compared with the engine's output, `qk_norm_rope` over q and k only, under:

- same >= 99 % and p99.9 <= 1, 2 for the three GEMMs with an epilogue. These are verify's accumulated gate, apart from that 2: verify allows it for SiTU on real activations only, and holds the gated residuals to 1 with its flip check, which needs the exact sums and is not ported.
- scaled <= max(16, 4 x the scalar leaf's). A backstop: every wrong-row mutant below that exceeds it also fails the row limit. It is wide because a gated residual near cancellation moves by many ULPs for one flipped bit: the card's block-41 down gemm + gated reached 64, twice the scalar leaf's 32, and a limit of 2x would have left it no margin.
- the lowest same % of any token row >= the scalar leaf's lowest minus 15 points. One wrong token row is 0.012 % of an op at this shape, too few values for same or p99.9 to see.
- the count of outputs above scaled 4 <= max(8, 8 x the scalar leaf's count). A wrong 16x16 fragment or a wrong column, the shape a fragment layout bug takes, is too few values for the row limit and moves each by a few ULPs, under the scaled limit; it does put outputs above scaled 4 where a correct op puts none or few.

Then the whole block on its own outputs, under the limits of that run's control.

The row margin is wide because the card cannot be recalibrated from here. On NVIDIA, two other accumulation orders in the scalar leaf (each fragment's 16 products summed as a tree, or K descending) move the lowest row by at most 0.36 points (K descending, block 41 attention 95.28 to 95.64), and the card's per-op same is within 0.07 points of the scalar leaf's. The third review's wrong-row mutants (below) reach at most 41.5 % in their row, against limits of 80.3 to 85. The scalar leaf's lowest rows on the dump: attention 96.58 and 95.28 % (blocks 1 and 41), the GEMM ops 99.34 to 99.80 %, the norms and `qk_norm_rope` 100 %.

Outputs above scaled 4, per op, blocks 1 and 41 (others 0 everywhere):

| | up_gate gemm + SiTU | down gemm + gated | count limit |
|---|---|---|---|
| scalar leaf | 0, 1 | 0, 31 | 8, 8; 8, 248 |
| tree of 16 / K descending | 0, 1 / 0, 1 | 0, 30 / 0, 29 | |
| card, earlier run | 0, unknown (scaled 5) | 0, unknown (scaled 64) | |

The card's count is known to be 0 wherever its scaled was 4 or less. The third review's fragment and column mutants put, at blocks 1 and 41: one attention output fragment x 1.03, 186 and 80 (scaled 8); 16 columns of one 32-row QKV tile x 1.03, 38 and 13 (scaled 7 and 6); one down output column x 1.06, 11 and 8464 (scaled 13 and 32). All six exceed the count limit, and none fails another per-op limit. The floor must stay under 11 for the down column at block 1, and the factor at 4 or more over block 41's down; 8 and 8 leave 8x over the other accumulation orders' counts.

The card ran the earlier replay, without the poisoning, the row limit, the scaled limit and the q-and-k-only comparison. On the RX 9060 XT, exit 0:

| Op, against the engine | Block 1: same %, p99.9, scaled | Block 41: same %, p99.9, scaled |
|---|---|---|
| norm1 | 99.9998, 0, 2 | 100, 0, 2 |
| qkv gemm | 99.853, 1, 1 | 99.739, 1, 1 |
| qk_norm_rope | 99.9997, 0, 2 | 99.9998, 0, 1 |
| attention | 99.417, 1, 1.25 | 99.196, 1, 2 |
| o gemm + gated | 99.970, 0, 1 | 99.954, 0, 2 |
| norm2 | 99.9997, 0, 4 | 99.9996, 0, 2 |
| up_gate gemm + SiTU | 99.738, 1, 4 | 99.607, 2, 5 |
| down gemm + gated | 99.942, 0, 2 | 99.869, 1, 64 |
| whole block, final x | 83.28 (limit 79.28), 64, 2 | 67.91 (limit 64.93), 320, 64 (limit 128) |

The scalar leaf under the same comparison gives 83.30 and 68.17 for the whole block, so WMMA's rounding sits as close to the engine's as an exact fp32 walk does. The norms and `qk_norm_rope` are not bit-exact on the card as they are on NVIDIA: AMD's `rsqrt` flips a few low bits, in about 3 values in 10^6. Against the new limits, every figure of that run passes: scaled at most half its limit (block 41's down gemm + gated, 64 against 4 x 32 = 128; the norms' 2 to 4 against 16), same at least 0.196 points over 99 %, p99.9 within its limit. qkv gemm and attention at both blocks (1 of 1) and SiTU at block 41 (2 of 2) sit exactly at their p99.9 limits, with the same p99.9 as the scalar leaf. For this card and toolchain that is a regression gate: on another ROCm or chip, a flip there is a reason to look closer, not by itself a defect. That run did not compute row figures, so the row limit has no card data yet, and its `qk_norm_rope` figure is over q, k and v.

## Verified

On an RTX 4090 Laptop (sm_89), CUDA 13, ROCm 7.2.4 headers. Inputs are synthetic unless the line says real.

- gfx1100, gfx1200 and gfx1201 compile with warnings as errors and link against `libamdhip64`, with 12 `v_wmma_f32_16x16x16_bf16` per GEMM and SiTU kernel, and 34 in attention (48 on gfx1100). Every kernel reports wave size 32, maximum flat workgroup size 32, no scratch and no spills. The code objects have no undefined symbols. `check_device.sh`, given the target by CMake, fails if a kernel in the metadata goes unchecked, if a kernel's name is not one of the eight it lists, or if its WMMA count differs from the one listed for that target. The third review's edited assemblies fail it: a kernel renamed to contain `pack_`, an argument type renamed `pack_Args`, WMMAs removed from SiTU or attention, and a build whose packed walk holds no WMMA (6, 6, 6 and 18 left by the row-major path). The script before this check passed three of those five.
- The zero-padded tail copies are read by a fragment load that can be a FLAT load: the copy and the operand merge into one generic pointer. After the packed path removed `__syncthreads`, nothing waited for the copy's DS writes before that load. `leaf::pad_barrier()` waits for them (a release fence at workgroup scope for `__shared__` memory). In the assembly that adds 34 `s_wait_dscnt 0x0` on gfx1200 and gfx1201, and 55 `s_waitcnt lgkmcnt(0)` on gfx1100, each at the end of a padded copy, before the block holding the FLAT load (or the DS load, where the compiler kept the pointer in `__shared__` memory). Nothing else changes, apart from two `s_delay_alu` scheduling hints in the GEMM kernels' row-major loads on gfx12. The pack, norm and RoPE kernels' assembly is identical, and so is every instruction of the packed path.
- The AMD binaries exit with status 2 and a message on a machine with no AMD device.
- A workgroup count wrong for SiTU, found while building the packed path, ran half the tiles and passed every check on the card, packed against row-major included, with the known answer's F inside one tile. Only the engine comparison caught it. The known answer's F now spans tiles.
- `hip_block_verify` and `hip_block_verify --real` pass all their gates, exit status 0.
- Mutants on scratch copies of the packed path, from the third review and re-run after its fixes: see [Mutants](#mutants).
- The old and new kernels on the scalar leaf, through the third review's old-against-new harness after its fixes: every op on the dump's inputs, both whole blocks, tail shapes and denormal attention give identical bits, on the packed path and on the row-major one.
- `rms_norm_modulate` from 16 rows and `qk_norm_rope` match the engine bit for bit in every case, real activations and edge inputs included.
- Below 16 rows `rms_norm_modulate` is not bit-exact: the engine's block width grows there and the prototype keeps its sum order. Over 400 seeds, 3 to 5 values in 10^6 differ at 7 to 15 rows of 3072, up to 128 raw ULPs and 2 scaled. None differed in the tests at 1 to 5 rows. The DiT does not run fewer than 16 tokens at an image of 64x64 or more.
- On real activations, the GEMM ops differ from the engine in 0.03 to 0.4 % of outputs and attention in 0.6 to 0.9 %. Per-op p99.9 is at most 1 except SiTU at block 41 (2), and scaled at most 5 except the last block's gated residual (32).
- Outlier check, summed over the real runs; each row's outputs more than 4 ULPs apart, the worst 200 per case checked against a host evaluation:

| Op | Outliers | Checked | Engine closer | Prototype closer | Largest error: engine, prototype |
|---|---|---|---|---|---|
| QKV GEMM (exact sum) | 20412 | 1400 | 14 | 1386 | 7.79e-05, 1.43e-05 |
| SiTU (exact sums, then its rounding points) | 35327 | 1400 | 13 | 1387 | 1.7e-05, 1.75e-06 |
| Gated residual (exact sum, then its rounding points) | 18966 | 2800 | 182 | 2618 | 0.5, 0.25 |
| Attention (fp64 replay with bf16 weights) | 17484 | 1400 | 590 | 810 | 0.0168, 0.0113 |

- Edge inputs agree between engine and prototype, NaN placement included, with one exception: query rows up to 2^60, where |S * log2_scale| passes about 2^31. There both are wrong. The fp32 FMA of softmax's exponent leaves a rounding residual beyond `exp2f`'s range, so a row's largest score weighs 0 or Inf instead of 1: at row 187, head 15, dim 0, the engine gives 0, the prototype NaN, the fp32 replay NaN, the fp64 replay -0.4648. Rows up to 2^20 pass every gate. The 2^60 case is printed and not gated.
- `device_math.h`'s bf16 conversions equal `__float2bfloat16_rn` over all 2^32 fp32 inputs and `__bfloat162float` over all 2^16 bf16 inputs. NaN inputs give NaN with different payload bits.
- Other tile constants gave the same bits on the card as far as tried (the QKV walk at 16x16 to 64x64, 0 of 77856768 values differ between packed and strided at every tile and block size).
- Valid changes of accumulation order in the scalar leaf: a tree of each 16 products, then added, passes verify, `--real` and the bench; K descending passes `--real` and the bench and fails one synthetic row, attention over query rows 2^-20 to 2^20, at p99.9 of 2 against 1. Verify runs only this fixed leaf, so that row can't reject a card. These change only the order of additions, and say nothing about denormal handling or AMD's math.
- On the review's CPU emulation of rocWMMA, 30 ops on verify-style shapes are byte-identical between the scalar leaf, gfx1100 and gfx1201, and 13 leaf and kernel mutants are detected on both targets. Its model accumulates K ascending with host math.

## Whole-block gate on real activations

The control is the engine's block with its QKV GEMM result replaced by the exact sum, rounded through fp32 to bf16 from a host fp64 evaluation, with no prototype code involved. It is compared with the unmodified engine by final x, in the same run as the prototype's block. Each pair of rows below is control, then prototype:

| Case | | same % | p99 | p99.9 | scaled | Limits: same, p99, scaled |
|---|---|---|---|---|---|---|
| golden, block 1 | control | 84.72 | 6 | 64 | 3 | 79.72, 16, 16 |
| | prototype | 83.98 | 6 | 64 | 2 | |
| golden, block 21 | control | 74.76 | 16 | 128 | 2 | 69.76, 32, 16 |
| | prototype | 74.31 | 16 | 128 | 2 | |
| golden, block 22 | control | 72.82 | 16 | 128 | 2 | 67.82, 32, 16 |
| | prototype | 72.36 | 16 | 128 | 4 | |
| golden, block 23 | control | 71.80 | 16 | 128 | 2 | 66.80, 32, 16 |
| | prototype | 71.51 | 16 | 128 | 4 | |
| golden, block 41 | control | 70.92 | 16 | 224 | 96 | 65.92, 32, 192 |
| | prototype | 69.24 | 20 | 272 | 96 | |
| E13, block 1 | control | 84.28 | 6 | 64 | 3 | 79.28, 16, 16 |
| | prototype | 83.30 | 8 | 64 | 3 | |
| E13, block 41 | control | 69.93 | 17 | 248 | 64 | 64.93, 34, 128 |
| | prototype | 68.17 | 22 | 288 | 96 | |

A QKV result that is more accurate than the engine's own still moves the engine's final x this far, so these drops come from the block's sensitivity to a low-bit QKV difference. The control's same % is within 0.32 points of the engine with only the prototype's QKV GEMM swapped in, the row `engine + prototype's qkv gemm`.

The largest differences come from cancellation. At block 41's worst down output, residual 2768 + delta -2864 (gate 30.38), and the exact x·w of -94.2508 rounds to -94.5 on the prototype's side and to -94 on the engine's, leaving -96 and -80. SiTU's 2 ULPs follow from its steep gate factor: about 96.5 % of outputs 2 to 4 ULPs apart come from inputs within 1 ULP of the exact sums, and in a synthetic block all but 16 of 43065 from inputs within 4 ULPs.

These limits were set after seeing results, and the prototype passes by 3.2 to 4.7 points of same %. What makes them defensible is the reference: the control contains no prototype code and is measured on the same inputs in the same run, so it cannot move with the prototype's errors. Its own gates catch a gross failure of the control code, such as a zeroed QKV: its QKV differs from the engine's in 0.15 to 0.28 % of outputs, at scaled 1. A milder regression still opens the limits: in the second review, a control QKV off by 1 ULP on 0.6 % of outputs, 25 % of the control's final x off by 1 ULP, one Inf in its final x and one row at +20000 ULPs each passed `--real`, with the scaled limit going to Inf or past 10^10 in the last two. No bug was found to pass through these, and synthetic mode's fixed block gate does not depend on the control. The margins of 5 points and 2x are judgment, not derived. The gate is also loose where the control is: scaled <= 192 at block 41 lets through what the fixed gate would catch. Per-op gates are what catch small errors inside an op: the row_sum mutant passes 5 of 7 real whole-block rows and fails only on its attention row there.

## Mutants

The third review ran 51 mutants on scratch copies of the packed path through `hip_block_verify`, the bench and `hip_block_replay` on the scalar leaf, and 26 of them through `--real`. The rounding mutants each fail their own op's row in all 7 `--real` cases, and verify, the bench and replay:

| Mutant | Whole-block rows failing in `--real` |
|---|---|
| Gated, no rbf after the gate multiply | 7 of 7 |
| SiTU h and g unrounded | 5 of 7 |
| Norm, no rbf after x*r | 7 of 7 |
| RoPE products unrounded | 7 of 7 |
| Attention row_sum over rounded weights | 2 of 7 |
| Rescale 0.4 % too large | 7 of 7 |

The packing and walking mutants are caught by verify and replay: a wrong first group of a sequence, a key step off by one in K or V, the rescale test always false or skipping rows, V's columns at the wrong offset, wrong weight groups, `pack_rows` too short, the max and sum loops dropping values, a stale P tail, a packer lane mismatch. Block orders that drop tiles pass the bench and fail verify and replay. Removing the GEMM pack's zero fill fails by an illegal memory access on NVIDIA. Chain mutant c, `run_block` in the order 0, 1, 2, 3, 5, 4, 6, 7, fails verify, `--real` and replay (whole block 7.08 and 0.55 % same) and passes the bench.

What passed every test in that review, and what catches it after its fixes (the output buffers poisoned with NaN, replay's row and scaled limits, verify's attention edge rows through `run_block_op`), re-run on the fixed code:

| Mutant | Before | Now |
|---|---|---|
| Attention tile's last row not written | passed replay | replay: NaN mismatch in attention and the whole block, both blocks |
| SiTU tile's last row not written | passed replay | replay: the same |
| A SiTU row not written on the row-major path only | passed the bench, 0 values differ | bench: 3072 values differ |
| One QKV row zeroed | passed replay at block 1 | replay, both blocks: that row 0 % same, scaled 255 |
| One attention row x 1.02 | passed replay | replay, both blocks: row 0 % |
| One SiTU row x 2 | passed replay at block 1 | replay, both blocks: row 0 %, scaled 247 and 255 |
| One token without RoPE | passed replay | replay, both blocks: row 32.8 and 33.5 %, scaled 805 and 1486 |
| 40 norm rows 0.39 % large | passed replay at block 41 | replay, both blocks: rows 28.3 to 41.5 %; at block 41 only the row limit fails it |
| Rescale test ignoring NaN (`< 1`) | passed all | verify: op 3 with every score -Inf in a one-row tile and in a full tile, packed and row-major |
| Attention pack's zero fill removed | passed all | verify: op 3 packed with a NaN or Inf in v, and the NaN v at the next sequence's first token; a synthetic block's attention and whole-block rows |
| Packer and kernel grouping sequences by 16 tokens | passed all | verify: op 3 packed, plain and NaN k, scaled 255 |
| One attention output fragment (16 rows, 16 dims of one head) x 1.03 | passed replay | replay, both blocks: 186 and 80 outputs above scaled 4, limit 8 |
| 16 columns of one 32-row QKV tile x 1.03 | passed replay | replay, both blocks: 38 and 13 above scaled 4, limit 8 |
| One down output column x 1.06 | passed replay | replay, both blocks: 11 above scaled 4 against 8, and 8464 against 248 |

Still passing every test: the row sum reversed or summed as a tree of four (no test fixes the order of that sum; see "Not tried"), and three changes that cannot change an output: the accumulator trip taken at every step, the row-major tail walking all 32 rows of its tile, and dropping the last workgroup, which has no tile at the dump's shape (the bench's known answer and verify fail it).

The bench's known-answer check tests kernels, not the block driver, and the bench passes chaining mutants. Chaining is covered by `hip_block_verify` on NVIDIA and by `hip_block_replay` on any device. The driver is host code shared by all targets.

## Not verified

Only gfx1200 has run this, and for two sessions; gfx1100 and gfx1201 have only been compiled.

- A whole 42-block step, a VAE or a text encoder on AMD. The 4.4 s is 42 times one block.
- The old kernels against the new on the card. The bit-for-bit result is the scalar leaf's on NVIDIA.
- Anything after the third review's fixes on a card: the poisoned buffers, replay's row and scaled limits, `pad_barrier()`'s wait and the edge rows all ran on NVIDIA only.
- That a wavefront fence is enough on every chip: it holds on gfx1200, where the checks above pass. It relies on a workgroup being one wave, which `check_device.sh` checks statically.
- rocWMMA's fragment load and store order on RDNA3 and on gfx1201. An independent review checked its lane and K mapping on a CPU emulation derived from rocWMMA's own layouts, and the known-answer check confirms it on gfx1200.
- `exp2f`, `expf`, `tanhf` and `rsqrt` accuracy in AMD's device libraries beyond the known-answer inputs, where they stayed within the limits on gfx1200.
- Any time or rate on another AMD chip. The tile constants and the 4 MB block are tuned on this chip only.
- Whether the known-answer limits suit other cards: they come from the scalar leaf and a model of math-library flips, and gfx1200 passes well inside them.
- NaN, Inf and denormals through WMMA and AMD's math. The known-answer inputs contain none of these, so a card that flushes denormals passes it and shows nothing about them.
- The kernels write `__shared__` buffers right after a collective read of them with no barrier between (`mma_leaf.h`). That holds in hardware for one wave and is a data race in LLVM's terms. The third review read the scheduled MIR of every kernel on gfx1100 and gfx1200: no `__shared__` read follows a write to the same buffer without a fence between, the fences stay in source order, and where a buffer is read then rewritten the read stays first. That is the compiler's order only: the fence emits no instruction, so the argument rests on a wave's DS operations completing in order, which no card run has tested apart from the checks passing on gfx1200. The emulation adds barriers there, so it does not test this either.
- Other `gfx11xx` and `gfx12xx` chips, and Windows.

The attention tile of 16 rows per wave uses 241 VGPRs on gfx1100 and 249 on gfx1200 and gfx1201 with no spills, close to the limit. 32 rows per wave spills.
