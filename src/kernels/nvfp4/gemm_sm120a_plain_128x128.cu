// NVFP4 GEMM, epilogue E1 (plain), tile 128x128x128 Cooperative (sm_120a only, see gemm_sm120a_impl.cuh), plus the image probe
// and the E2M1 conversion of test G1e.
#include <algorithm>

#include "kernels/nvfp4/gemm_sm120a_impl.cuh"

namespace mi::kernels::nvfp4::detail {

template void gemm<Nvfp4Tile::k128x128x128, Epi::kPlain>(const GemmArgs&, int, cudaStream_t);

cudaError_t probe_image() { return probe<Nvfp4Tile::k128x128x128, Epi::kPlain>(); }

namespace {

// cvt.rn.satfinite.e2m1x2.f32 d, a, b packs a in the high nibble and b in the low one; codes[i] takes x[i]'s nibble.
__global__ void e2m1_cvt_kernel(const float* x, std::uint8_t* codes, std::int64_t n) {
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    unsigned short packed = 0;
    asm("{\n\t.reg .b8 r;\n\tcvt.rn.satfinite.e2m1x2.f32 r, %1, %2;\n\tcvt.u16.u8 %0, r;\n\t}"
        : "=h"(packed)
        : "f"(x[i]), "f"(0.0f));
    codes[i] = static_cast<std::uint8_t>(packed >> 4);
  }
}

}  // namespace

void e2m1_cvt(const float* x, std::uint8_t* codes, std::int64_t n, cudaStream_t stream) {
  if (n == 0) return;
  const auto blocks = static_cast<unsigned>(std::min<std::int64_t>((n + 255) / 256, 4096));
  e2m1_cvt_kernel<<<blocks, 256, 0, stream>>>(x, codes, n);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::kernels::nvfp4::detail
