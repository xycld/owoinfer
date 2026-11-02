# hip_block

A standalone HIP port of the bf16 exact path of one MOSS DiT block (`src/models/moss_dit/dit_block_fused.h`): RMSNorm with modulation, the QKV GEMM, per-head QK norm + RoPE, varlen joint attention (head_dim 128, non-causal, fp32 online softmax, bf16 in and out), and the gated-residual and SiTU-GLU GEMMs. It does not use the engine's HAL and the main build does not know this directory.

One kernel source, `block_kernels.hip`, builds three ways. They differ in the leaf in `mma_leaf.h`, a 16x16x16 bf16 multiply with fp32 accumulators:

| Target | Compiler | Leaf |
|---|---|---|
| `gfx1100` (RDNA3) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
| `gfx1201` (RDNA4) | ROCm clang | rocWMMA, `v_wmma_f32_16x16x16_bf16` |
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

`build-*/` is ignored by git. ROCm's libraries must be on the loader path at run time if they are not installed system-wide: `LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib`, plus the directory holding `libnuma` and `libdrm` where those are unpacked with ROCm.

## Running on a card

```sh
build-hip-gfx1201/hip_block_bench            # RDNA4; build-hip-gfx1100 for RDNA3
build-hip-gfx1201/hip_block_bench --rows 4378,4106 --iters 5    # the defaults
```

`hip_block_bench` prints the device name, then a known-answer check of every kernel against a host replay of its op order, with tails on every axis, then runs one block and looks for NaNs. It refuses to time a block that fails either. Then it prints min and median milliseconds per op and for the whole block, on synthetic 1024²-shaped activations: E13's 8484 tokens, two rows of 4378 and 4106 (about 0.8 GiB). Exit status: 0 ran, 1 error, known-answer failure or NaN, 2 no usable device.

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

`build-hip-nvidia/hip_block_verify` needs a CUDA device with about 2 GiB free. It runs the same inputs through the engine's kernels (`run_block_fused` and the individual kernels, from `libmossinfer_core.a`) and through the prototype with the scalar leaf, and prints per-op error distributions in bf16 ULPs. It checks the kernels' logic above the leaf, and nothing about WMMA or rocWMMA's lane layout. The cases are shapes with tails on M, N and K; varlen attention with rows of different lengths and with query lengths different from key lengths; edge inputs (a zero row, a NaN, an Inf, denormals, per-row magnitudes across a wide range, attention scores all -Inf); and whole blocks over two and four rows.

`hip_block_verify --real --weights DIR --golden DIR` (or `MOSS_WEIGHTS`, `MOSS_GOLDEN`) runs blocks 1, 21, 22, 23 and 41 on their real weights and on activations captured from the engine's forward at the 1024² golden shape, needing about 3 GiB. `DIR` for `--weights` holds `base/moss.mossidx`; `--golden` holds the golden `.npy` files. Blocks stream during the capture, so the DiT need not fit. It also runs the engine's block with one op at a time replaced by the prototype's.

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

## Verified

On an RTX 4090 Laptop (sm_89), CUDA 13, ROCm 7.2.4 headers. Inputs are synthetic unless the line says real.

- Both AMD targets compile with warnings as errors and link against `libamdhip64`, and each has 34 `v_wmma_f32_16x16x16_bf16` instructions (6 per GEMM kernel, 4 in SiTU, 18 in attention). Every kernel reports wave size 32, maximum flat workgroup size 32, no scratch and no spills, and `check_device.sh` fails if a kernel in the metadata goes unchecked or a GEMM or attention kernel holds no WMMA. The code objects have no undefined symbols.
- The AMD binaries exit with status 2 and a message on a machine with no AMD device.
- `hip_block_verify` and `hip_block_verify --real` pass all their gates, exit status 0.
- Fifteen mutants on scratch copies, nine in the kernels' rounding and six in the block driver, each make `hip_block_verify` and `--real` exit 1 against the engine. No mutant fails in synthetic mode only. See [Mutants](#mutants).
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
- Other tile constants (GEMM 16x32, SiTU 48x32, attention 32 rows) gave the same bits as the defaults with the scalar leaf.
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

Rounding mutants in `--real`, over its 7 real cases. Each fails its own op's row in all 7. Same % is that row's range:

| Mutant | Own row failing | Same % | Whole-block rows failing | Bench |
|---|---|---|---|---|
| Gated, no rbf after gate multiply | o and down gemm+gated | 76.0 to 89.7 | 7 of 7 | fails |
| Gated, no rbf(ab) | o and down gemm+gated | 85.9 to 93.9 | 7 of 7 | fails |
| SiTU h, g unrounded | up_gate gemm+situ | 78.0 to 81.5 | 5 of 7 | fails |
| SiTU inner rbfs removed | up_gate gemm+situ | 54.7 to 59.1 | 7 of 7 | fails |
| Attention row_sum over rounded weights | attention | 95.7 to 97.3 | 2 of 7 | fails |
| Norm, no rbf after x*r | norm1 and norm2 | 77.7 to 84.0 | 7 of 7 | fails |
| Modulate, no rbf before shift | norm1 and norm2 | 64.5 to 74.9 | 7 of 7 | fails |
| RoPE products unrounded | qk_norm_rope | 89.1 to 89.6 | 7 of 7 | fails |
| Rescale 0.4 % too large | attention | 5.8 to 64.8 | 7 of 7 | fails |

Both SiTU mutants fail on `same < 99 %`, and p99.9 reaches 3 against the real limit of 2 in some rows. In synthetic mode the row_sum mutant fails 23 attention rows and 0 of 2 whole-block rows.

Chaining mutants, applied to the host block driver in `block_kernels.hip`:

| Mutant | Whole-block rows failing, same %: synthetic | `--real` | Per-op rows that also fail | Bench |
|---|---|---|---|---|
| a. o gemm's residual written to y, so x keeps its pre-attention values | 2 of 2, 2.3 | 7 of 7, 0.4 to 4.9 | o gemm+gated | exit 0 |
| b. norm2 with norm1's shift and scale | 2 of 2, 2.0 | 7 of 7, 0.2 to 1.7 | norm2 | exit 0 |
| c. norm2 run before the gated residual, so up_gate gets the earlier x | 2 of 2, 7.1 | 7 of 7, 0.5 to 7.3 | none | exit 0 |
| d1. rope read one token row late | 2 of 2, 21.8 to 24.5 | 7 of 7, 12.2 to 28.1 | qk_norm_rope | exit 0 |
| d2. boundary between rows one token early | 2 of 2, 75.4 to 75.9 | 6 of 7, 14.6 to 70.4 | attention (6 of 7) | exit 0 |
| e. down's gate taken from o's | 2 of 2, 2.1 | 7 of 7, 0.7 to 6.5 | down gemm+gated | exit 0 |

- Mutant a is the nearest equivalent of the wrong residual: the driver's residual is x itself, updated in place, so x before attention's norm is the right buffer.
- Mutant c is the only one that only the block gate catches. It changes `run_block`'s order, while the per-op rows call `run_block_op` on buffers set from the engine, and those see the others.
- Block 22 passes under d2, with the unmodified build's numbers (72.3556 % same, attention 99.4094 %). It runs the cond row alone, so there is no boundary to move.
- At E13's block 41, d2 fails the block gate on scaled alone (864 against 128). Its same of 66.64 % and p99 of 24 are inside the limits of 64.93 % and 34.

The bench's known-answer check does not catch chaining mutants: all six exit 0, as it tests kernels and only looks for NaNs in a whole block. Chaining is covered only by `hip_block_verify` on NVIDIA. The driver is host code shared by all targets, so that coverage applies to the AMD builds' driver, and not to anything that happens on a card.

## Not verified

No AMD device has run any of this:

- WMMA's execution, its accumulation rounding, and whether its results are within a few ULPs of the engine's.
- rocWMMA's fragment load and store order on silicon. An independent review checked its lane and K mapping on a CPU emulation derived from rocWMMA's own layouts, which shows the kernels agree with rocWMMA and not that rocWMMA agrees with silicon. The known-answer check tests this on a card.
- `__shared__` memory and `__syncthreads` behaviour in a 32-thread workgroup on AMD, `exp2f`, `expf`, `tanhf` and `rsqrt` accuracy in AMD's device libraries, and how they compare with the NVIDIA ones.
- Any time or rate on AMD. The tile constants are untuned, fragment loads compile to flat loads, `__syncthreads` emits a cache invalidate in single-wave kernels (`global_inv` on gfx1201, `buffer_gl0_inv` on gfx1100), and attention moves its output accumulators through `__shared__` memory on every key step.
- Whether the known-answer limits suit a card: they come from the scalar leaf and a model of math-library flips.
- NaN, Inf and denormals through WMMA and AMD's math. The known-answer inputs contain none of these, so a card that flushes denormals passes it and shows nothing about them.
- The kernels write `__shared__` buffers right after a collective read of them with no barrier between (`mma_leaf.h`). That holds in hardware for one wave and is a data race in LLVM's terms. The scheduled assembly has not been inspected for reordering at these points, and the emulation adds barriers there, so it does not test this.
- Other `gfx11xx` and `gfx12xx` chips, and Windows.

The attention tile of 16 rows per wave uses 238 VGPRs on gfx1100 and 213 on gfx1201 with no spills. 32 rows per wave spills to scratch on gfx1100.
