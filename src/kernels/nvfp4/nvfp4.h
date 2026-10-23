// NVFP4 W4A4 GEMMs for compute capability 12.0 (RTX 50, RTX PRO Blackwell): the Turbo tier on sm_120.
//
// Status: N0 spike (.omc/plans/nvfp4-design.md §4.2, §4.5). Only the plain bf16-output GEMM exists, for the S1 rate
// probe and the CI compile check; the activation quantizers, the gated-residual and LoRA epilogues, tile choice and
// the numerics version come with N4. The output bits are not pinned yet: no certificate may depend on them.
//
// Formats (design §2.1):
// - Values: E2M1, two per byte. The GEMM is indifferent to the nibble order as long as x and w use the same one (a
//   swap inside every byte of both operands permutes K identically within each 16-element block). Our convention is
//   hi_first (even element in the high nibble) for both: ComfyUI files are hi_first and are used as stored, so the
//   activation quantizer (N4) packs hi_first too.
// - Block scales: UE4M3 bytes, one per 16 consecutive K elements, in the cuBLAS / CUTLASS Sm1xxBlkScaledConfig
//   swizzle: 512-byte tiles of 128 rows x 4 scales (sf_offset). Padding rows and columns must be zero (0x7F is NaN).
// - Second level: f32 per token for x (x_scale[m]: amax / 2688, or 1 for an all-zero row, as weights/quant's NVFP4
//   codec and tests/data/nvfp4), and the host f32 per-tensor weight_scale_2 for w (w_scale2, the GEMM alpha).
//
// Activation quantizer contract (N4; reference: quantize_nvfp4_act_bf16 in src/weights/quant/nvfp4_codec.h and
// tools/reference/nvfp4_ref.py, bit-exact in test T3): NaN counts as 0, +-inf is clamped to +-bf16 max, then per row
// g = amax == 0 ? 1 : amax / 2688 with no lower clamp (a subnormal amax gives a subnormal g); per block
// sf = e4m3_rne(min((amax_b / 6) / g, 448)), q = e2m1_rne(clamp(x / (g * sf), -6, 6)), q = +0 where g * sf == 0.
// Every division is IEEE div.rn and no step may flush subnormals: compile without fast-math / -use_fast_math and
// without FTZ (no --ftz=true, no __fdividef).
//
// All pointers are device pointers. No allocation, no host synchronization. Invalid arguments throw
// std::invalid_argument; calling a GEMM where available() is false throws std::logic_error.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace mi::kernels::nvfp4 {

// Elements per UE4M3 block scale.
inline constexpr int kBlock = 16;
// Scale tile of the swizzled layout: 128 rows x 4 scales (64 K elements) = 512 bytes.
inline constexpr int kSfTileRows = 128;
inline constexpr int kSfTileCols = 4;

// Bytes of a swizzled scale buffer for `rows` rows of k elements: round_up(rows, 128) * round_up(k / 16, 4).
std::int64_t sf_bytes(std::int64_t rows, std::int64_t k);
// Byte offset of the scale of row `row`, K block `kb` (= k_index / 16), in that buffer:
// ((row / 128) * ceil(SFK / 4) + kb / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + kb % 4, SFK = k / 16.
std::int64_t sf_offset(std::int64_t row, std::int64_t kb, std::int64_t k);

// True when this build has the sm_120a GEMM images (MOSSINFER_NVFP4, CUDA >= 12.8, 120 among the CUDA
// architectures), the current device is exactly compute capability 12.0, and the image loads there. Cached per device.
bool available();

// Threadblock tile (M x N x K) of the GEMM: the two the S1 probe measures (upstream blackwell_functionality.md,
// Table 17; both Cooperative).
enum class Nvfp4Tile { k128x128x128, k256x128x128, kCount };
const char* nvfp4_tile_name(Nvfp4Tile tile);

// Operands of one NVFP4 GEMM, both K-contiguous: x [m, k / 2] bytes with x_sf (sf_bytes(m, k)) and per-row x_scale f32
// [m]; w [n, k / 2] bytes with w_sf (sf_bytes(n, k)) and w_scale2. Packed rows (row stride k / 2 bytes).
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

// out[m, n] = bf16((acc * x_scale[r]) * w_scale2 + bias[c]), acc = sum over K blocks of x_sf * w_sf * the E2M1 dot
// product (fp32, inside the MMA). bias: bf16 [n] or nullptr. out: bf16, row stride ldo, 16-byte aligned,
// ldo % 8 == 0, n % 8 == 0; bias 16-byte aligned.
void nvfp4_gemm(const Nvfp4Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
                cudaStream_t stream, Nvfp4Tile tile = Nvfp4Tile::k128x128x128);

}  // namespace mi::kernels::nvfp4
