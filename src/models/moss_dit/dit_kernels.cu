#include "models/moss_dit/dit_kernels.h"

#include <cuda_bf16.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/cuda_check.h"

namespace mi::moss_dit {
namespace {

constexpr int kTile = 32;
constexpr int kRowsPerThread = 4;  // block = kTile x (kTile / kRowsPerThread) threads

// out[c][r] = convert(in[r][c]) for an in matrix of rows x cols (row-major), through a padded shared tile.
template <typename In, typename Out>
__global__ void transpose_kernel(const In* in, Out* out, std::int64_t rows, std::int64_t cols) {
  __shared__ float tile[kTile][kTile + 1];
  const std::int64_t r0 = static_cast<std::int64_t>(blockIdx.y) * kTile;
  const std::int64_t c0 = static_cast<std::int64_t>(blockIdx.x) * kTile;
  const int tx = static_cast<int>(threadIdx.x);
  for (int i = static_cast<int>(threadIdx.y); i < kTile; i += kTile / kRowsPerThread) {
    const std::int64_t r = r0 + i, c = c0 + tx;
    if (r < rows && c < cols) {
      if constexpr (std::is_same_v<In, float>) {
        tile[i][tx] = in[r * cols + c];
      } else {
        tile[i][tx] = __bfloat162float(in[r * cols + c]);
      }
    }
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.y); i < kTile; i += kTile / kRowsPerThread) {
    const std::int64_t c = c0 + i, r = r0 + tx;
    if (r < rows && c < cols) {
      if constexpr (std::is_same_v<Out, float>) {
        out[c * rows + r] = tile[tx][i];
      } else {
        out[c * rows + r] = __float2bfloat16_rn(tile[tx][i]);
      }
    }
  }
}

struct Params {
  std::uint32_t words[kMaxParamBytes / 4];
};

__global__ void write_params_kernel(std::uint32_t* dst, Params p, int n) {
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) dst[i] = p.words[i];
}

void check(bool ok, const char* op, const std::string& msg) {
  if (!ok) throw std::invalid_argument(std::string("mi::moss_dit::") + op + ": " + msg);
}

template <typename In, typename Out>
void launch(const In* in, Out* out, std::int64_t rows, std::int64_t cols, cudaStream_t stream) {
  if (rows == 0 || cols == 0) return;
  const dim3 grid(static_cast<unsigned>((cols + kTile - 1) / kTile), static_cast<unsigned>((rows + kTile - 1) / kTile));
  transpose_kernel<In, Out><<<grid, dim3(kTile, kTile / kRowsPerThread), 0, stream>>>(in, out, rows, cols);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace

void latent_to_tokens(const Tensor& latents, const Tensor& tokens, cudaStream_t stream) {
  constexpr const char* op = "latent_to_tokens";
  check(latents.device == Device::kCUDA && latents.dtype == DType::kF32 && latents.rank == 3 && latents.is_contiguous(),
        op, "latents must be a contiguous CUDA f32 [C, h, w], got " + latents.describe());
  const std::int64_t c = latents.dim(0), hw = latents.dim(1) * latents.dim(2);
  check(tokens.device == Device::kCUDA && tokens.dtype == DType::kBF16 && tokens.rank == 2 && tokens.is_contiguous() &&
            tokens.dim(0) == hw && tokens.dim(1) == c,
        op, "tokens must be a contiguous CUDA bf16 [h*w, C], got " + tokens.describe());
  launch(latents.ptr<const float>(), tokens.ptr<__nv_bfloat16>(), c, hw, stream);
}

void tokens_to_latent(const Tensor& tokens, const Tensor& latents, cudaStream_t stream) {
  constexpr const char* op = "tokens_to_latent";
  check(latents.device == Device::kCUDA && latents.dtype == DType::kF32 && latents.rank == 3 && latents.is_contiguous(),
        op, "latents must be a contiguous CUDA f32 [C, h, w], got " + latents.describe());
  const std::int64_t c = latents.dim(0), hw = latents.dim(1) * latents.dim(2);
  check(tokens.device == Device::kCUDA && tokens.dtype == DType::kBF16 && tokens.rank == 2 && tokens.is_contiguous() &&
            tokens.dim(0) == hw && tokens.dim(1) == c,
        op, "tokens must be a contiguous CUDA bf16 [h*w, C], got " + tokens.describe());
  launch(tokens.ptr<const __nv_bfloat16>(), latents.ptr<float>(), hw, c, stream);
}

void write_params(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
  check(bytes <= kMaxParamBytes && bytes % 4 == 0 && reinterpret_cast<std::uintptr_t>(dst) % 4 == 0, "write_params",
        "needs <= " + std::to_string(kMaxParamBytes) + " bytes, a multiple of 4, to a 4-byte aligned address");
  if (bytes == 0) return;
  Params p{};
  std::memcpy(p.words, src, bytes);
  write_params_kernel<<<1, 256, 0, stream>>>(static_cast<std::uint32_t*>(dst), p, static_cast<int>(bytes / 4));
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::moss_dit
