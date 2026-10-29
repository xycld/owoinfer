// NVFP4 W4A4 kernels for compute capability 12.0 (RTX 50, RTX PRO Blackwell): the Turbo tier on sm_120.
// Design: .omc/plans/nvfp4-design.md §2.1, §4.1, §4.2, §4.5.
//
// Formats (design §2.1):
// - Values: E2M1, two per byte. The GEMM is indifferent to the nibble order as long as x and w use the same one (a
//   swap inside every byte of both operands permutes K identically within each 16-element block). Our convention is
//   hi_first (even element in the high nibble) for both: ComfyUI files are hi_first and are used as stored, so the
//   activation quantizers pack hi_first too.
// - Block scales: UE4M3 bytes, one per 16 consecutive K elements, in the cuBLAS / CUTLASS Sm1xxBlkScaledConfig
//   swizzle: 512-byte tiles of 128 rows x 4 scales (sf_offset). Padding rows and columns must be zero (0x7F is NaN).
// - Second level: f32 per token for x (x_scale[m]: amax / 2688, or 1 for an all-zero row, as weights/quant's NVFP4
//   codec and tests/data/nvfp4), and the host f32 per-tensor weight_scale_2 for w (w_scale2).
//
// Activation quantizers (reference: quantize_nvfp4_act_bf16 in src/weights/quant/nvfp4_codec.h and
// tools/reference/nvfp4_ref.py, bit-exact in test T3): NaN counts as 0, +-inf is clamped to +-bf16 max, then per row
// g = amax == 0 ? 1 : amax / 2688 with no lower clamp (a subnormal amax gives a subnormal g); per block
// sf = e4m3_rne(min((amax_b / 6) / g, 448)), q = e2m1_rne(clamp(x / (g * sf), -6, 6)), q = +0 where g * sf == 0.
// Every division is IEEE div.rn and no step may flush subnormals: compiled without fast-math / -use_fast_math and
// without FTZ (no --ftz=true, no __fdividef). E2M1 rounding is plain fp32 compares (no cvt.e2m1x2, which only
// sm_100a / sm_120a have), so the quantizers build and run on every architecture of the library.
//
// GEMM epilogues (per output element (r, c); acc = sum over K blocks of x_sf * w_sf * the E2M1 dot product, fp32,
// inside the MMA), in the INT8 epilogue's order (kernels/quant/quant.h) with x_scale[r] for s_x and w_scale2 for s_w,
// one fp32 rounding per operation (no FMA contraction):
//   E1   y = bf16(((acc * x_scale[r]) * w_scale2) + bias[c])
//   E1L  y = bf16((((acc * x_scale[r]) * w_scale2) + delta[r, c]) + bias[c])  (LoRA: delta, bf16, added in fp32)
//   E2   residual[r, c] = bf16(residual + bf16(y * gate[c])), y the bf16 E1 (or E1L) value, residual in place.
// The MMA's internal accumulation order and precision are hardware-defined (test G2 characterizes them); every tile
// and schedule gives the same bits (split-K 1, fixed K order; test G1c).
//
// All pointers are device pointers. No allocation, no host synchronization: graph-capturable. Invalid arguments throw
// std::invalid_argument; calling a GEMM where available() is false throws std::logic_error.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mi::kernels::nvfp4 {

// Numerics version of the NVFP4 path (activation quantizers, GEMM configurations, epilogue order): bump on any change
// to the output bits, including a CUTLASS update, reported in the quality object's numerics
// (pipeline/quality_report.h). Version 1 is defined by the N4 kernels: the quantizers above and the epilogues with
// explicit _rn operations (no FMA contraction), in the order documented here.
inline constexpr std::uint32_t kNvfp4NumericsVersion = 1;

// Elements per UE4M3 block scale.
inline constexpr int kBlock = 16;
// Scale tile of the swizzled layout: 128 rows x 4 scales (64 K elements) = 512 bytes.
inline constexpr int kSfTileRows = 128;
inline constexpr int kSfTileCols = 4;
// Largest K the activation quantizers take (a row is held in registers by one thread block).
inline constexpr int kMaxQuantizeK = 32768;

// Bytes of a swizzled scale buffer for `rows` rows of k elements: round_up(rows, 128) * round_up(k / 16, 4).
std::int64_t sf_bytes(std::int64_t rows, std::int64_t k);
// Byte offset of the scale of row `row`, K block `kb` (= k_index / 16), in that buffer:
// ((row / 128) * ceil(SFK / 4) + kb / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + kb % 4, SFK = k / 16.
std::int64_t sf_offset(std::int64_t row, std::int64_t kb, std::int64_t k);

// True when this build has the sm_120a GEMM images (MOSSINFER_NVFP4, CUDA >= 12.8, 120 among the CUDA
// architectures), the current device is exactly compute capability 12.0, and the image loads there. Cached per device.
bool available();

// ---- Activation quantizers (every architecture) ----

// Output of an activation quantizer for m rows of k elements: q [m, k / 2] packed hi_first (row stride k / 2 bytes,
// the GEMM's x), sf the swizzled scales (sf_bytes(m, k) bytes, every byte written: padding rows and columns get 0),
// scale f32 [m] (the GEMM's x_scale). All three 16-byte aligned.
struct Nvfp4Act {
  std::uint8_t* q = nullptr;
  std::uint8_t* sf = nullptr;
  float* scale = nullptr;
};

// x: bf16 [m, k] with row stride ldx (the o-proj input, attention output). Needs k % 32 == 0, k <= kMaxQuantizeK,
// 16-byte aligned x with ldx % 8 == 0 and ldx >= k.
void quantize_nvfp4(const __nv_bfloat16* x, std::int64_t ldx, const Nvfp4Act& out, std::int64_t m, int k,
                    cudaStream_t stream);

// The qkv / ffn_up_gate input, fused: the bf16 tensor kernels::rms_norm_modulate_moss(x, norm_weight, mod_scale,
// mod_shift) would produce for these m rows, then quantize_nvfp4 of it, bit for bit, without storing the bf16 tensor.
// norm_weight, mod_scale, mod_shift: contiguous 16-byte aligned bf16 [k]. Layout rules as quantize_nvfp4, plus
// kernels::rms_norm_moss's row-length limit, and the same row-count requirement as
// quant::rms_norm_modulate_rotate_quantize: bit-exact only against a norm over the same m rows (one call per tensor
// the reference normalizes as one). y (optional): the bf16 norm output is also stored there (row stride ldy, 16-byte
// aligned, ldy % 8 == 0, not overlapping x), bit for bit kernels::rms_norm_modulate_moss's (a LoRA side branch's input).
void rms_norm_modulate_quantize_nvfp4(const __nv_bfloat16* x, std::int64_t ldx, const __nv_bfloat16* norm_weight,
                                      const __nv_bfloat16* mod_scale, const __nv_bfloat16* mod_shift,
                                      const Nvfp4Act& out, std::int64_t m, int k, cudaStream_t stream,
                                      __nv_bfloat16* y = nullptr, std::int64_t ldy = 0);

// The ffn_down input, fused: kernels::situ_glu of the [hidden | gate] ffn_up_gate output (bf16 [m, 2F], row stride
// ld: the E1 GEMM output), then quantize_nvfp4 of the bf16 [m, F] result, bit for bit, without storing it. k = F.
// Needs F % 32 == 0, F <= kMaxQuantizeK, 16-byte aligned up_gate with ld % 8 == 0 and ld >= 2F. ffn (optional): the
// bf16 [m, F] SiTU-GLU output is also stored there (row stride ldf, 16-byte aligned, ldf % 8 == 0, not overlapping
// up_gate), bit for bit kernels::situ_glu's.
void situ_glu_quantize_nvfp4(const __nv_bfloat16* up_gate, std::int64_t ld, const Nvfp4Act& out, std::int64_t m,
                             int f, cudaStream_t stream, __nv_bfloat16* ffn = nullptr, std::int64_t ldf = 0);

// ---- GEMMs (sm_120a) ----

// Threadblock tile (M x N x K) and schedule of the GEMM (upstream blackwell_functionality.md, Table 17). kAuto picks
// per shape (nvfp4_tile_for); the others force one, for the tuning benchmark and the tests.
enum class Nvfp4Tile { kAuto, k128x128x128, k128x128x128Pingpong, k256x128x128, kCount };
const char* nvfp4_tile_name(Nvfp4Tile tile);
// The tile kAuto runs for an m x n x k problem on the current device.
Nvfp4Tile nvfp4_tile_for(std::int64_t m, std::int64_t n, std::int64_t k);

// Operands of one NVFP4 GEMM, both K-contiguous: x [m, k / 2] bytes with x_sf (sf_bytes(m, k)) and per-row x_scale f32
// [m] (an Nvfp4Act); w [n, k / 2] bytes with w_sf (sf_bytes(n, k)) and w_scale2. Packed rows (row stride k / 2 bytes).
// Layout: 16-byte aligned x, w, x_sf, w_sf and x_scale; k % 32 == 0 (128-bit TMA accesses). 0 < m, n, k < 2^31.
struct Nvfp4Operands {
  const std::uint8_t* x = nullptr;
  const std::uint8_t* x_sf = nullptr;
  const float* x_scale = nullptr;
  const std::uint8_t* w = nullptr;
  const std::uint8_t* w_sf = nullptr;
  float w_scale2 = 1.0f;
  std::int64_t m = 0, n = 0, k = 0;
};

// The LoRA term of E1L / E2L: bf16 [m, n] with row stride ld (16-byte aligned, ld % 8 == 0, ld >= n). p == nullptr:
// no term (E1 / E2).
struct Nvfp4Delta {
  const __nv_bfloat16* p = nullptr;
  std::int64_t ld = 0;
};

// E1 / E1L: out[m, n] = y. bias: bf16 [n] or nullptr. out: bf16, row stride ldo, 16-byte aligned, ldo % 8 == 0,
// n % 8 == 0; bias 16-byte aligned.
void nvfp4_gemm(const Nvfp4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                cudaStream_t stream, Nvfp4Tile tile = Nvfp4Tile::kAuto, Nvfp4Delta delta = {});

// E2 / E2L: residual[m, n] = residual + bf16(y * gate[c]), as kernels::fused::gemm_gated_residual on the bf16 GEMM
// (the o-proj / ffn_down linear then kernels::gated_residual). residual is read and written in place (row stride ldr);
// gate: bf16 [n]. Alignment as nvfp4_gemm.
void nvfp4_gemm_gated_residual(const Nvfp4Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                               __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream,
                               Nvfp4Tile tile = Nvfp4Tile::kAuto, Nvfp4Delta delta = {});

// Test G1e: the E2M1 codes of n floats by the sm_120a conversion cvt.rn.satfinite.e2m1x2.f32 (one code per byte,
// low 4 bits), to compare with the quantizers' fp32-compare rounding. x f32 [n], codes [n]. Needs available().
void e2m1_cvt_codes(const float* x, std::uint8_t* codes, std::int64_t n, cudaStream_t stream);

}  // namespace mi::kernels::nvfp4
