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

// out[c * ld + r] = convert(in[r][c]) for an in matrix of rows x cols (row-major; ld = rows: the plain transpose), through
// a padded shared tile.
template <typename In, typename Out>
__global__ void transpose_kernel(const In* in, Out* out, std::int64_t rows, std::int64_t cols, std::int64_t ld) {
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
        out[c * ld + r] = tile[tx][i];
      } else {
        out[c * ld + r] = __float2bfloat16_rn(tile[tx][i]);
      }
    }
  }
}

struct Params {
  std::uint32_t words[kMaxParamBytes / 4];
};

__global__ void extrapolate_kernel(const float* x, const float* x_prev, float c, float* out, std::int64_t n) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + c * (x[i] - x_prev[i]);
}

// One thread per input byte (two output values).
__global__ void unpack_int4_kernel(const std::uint8_t* in, std::int64_t ld_in, std::int8_t* out, std::int64_t ld_out,
                                   std::int64_t rows, std::int64_t half) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * half) return;
  const std::int64_t r = i / half, c = i % half;
  const auto b = static_cast<std::int32_t>(in[r * ld_in + c]);
  // Sign-extend each nibble: (v << 28) >> 28 on a 32-bit int.
  const auto lo = static_cast<std::int8_t>(static_cast<std::int32_t>(static_cast<std::uint32_t>(b) << 28) >> 28);
  const auto hi = static_cast<std::int8_t>(static_cast<std::int32_t>(static_cast<std::uint32_t>(b) << 24) >> 28);
  out[r * ld_out + 2 * c] = lo;
  out[r * ld_out + 2 * c + 1] = hi;
}

// One warp per output row; 8 bf16 weights per 16-byte load.
__global__ void linear_fp32_row_kernel(const __nv_bfloat16* w, const float* x, const float* bias, __nv_bfloat16* out,
                                       int n, int k) {
  const int row = static_cast<int>(blockIdx.x) * (blockDim.x / 32) + static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  if (row >= n) return;
  const uint4* wr = reinterpret_cast<const uint4*>(w + static_cast<std::int64_t>(row) * k);
  float acc = 0.0f;
  for (int i = lane; i < k / 8; i += 32) {
    const uint4 v = wr[i];
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
    for (int j = 0; j < 8; ++j) acc = fmaf(__bfloat162float(h[j]), x[8 * i + j], acc);
  }
  for (int o = 16; o > 0; o /= 2) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) out[row] = __float2bfloat16_rn(acc + bias[row]);
}

// 16 input bytes (32 values) per thread: one 16-byte load, two 16-byte stores. Each 32-bit word of 8 nibbles becomes 8
// sign-extended bytes.
__device__ __forceinline__ uint2 unpack8(std::uint32_t w) {
  std::uint32_t lo = 0, hi = 0;
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t b = (w >> (8 * i)) & 0xFFu;
    const std::uint32_t l = static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int32_t>(b << 28) >> 28));
    const std::uint32_t h = static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int32_t>(b << 24) >> 28));
    const std::uint32_t pair = l | (h << 8);
    if (i < 2) lo |= pair << (16 * i);
    else hi |= pair << (16 * (i - 2));
  }
  return make_uint2(lo, hi);
}

__global__ void unpack_int4_vec_kernel(const std::uint8_t* in, std::int64_t ld_in, std::int8_t* out, std::int64_t ld_out,
                                       std::int64_t rows, std::int64_t chunks) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * chunks) return;
  const std::int64_t r = i / chunks, c = i % chunks;
  const uint4 v = *reinterpret_cast<const uint4*>(in + r * ld_in + 16 * c);
  const uint2 a = unpack8(v.x), b = unpack8(v.y), d = unpack8(v.z), e = unpack8(v.w);
  uint4* o = reinterpret_cast<uint4*>(out + r * ld_out + 32 * c);
  o[0] = make_uint4(a.x, a.y, b.x, b.y);
  o[1] = make_uint4(d.x, d.y, e.x, e.y);
}

__global__ void write_params_kernel(std::uint32_t* dst, Params p, int n) {
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) dst[i] = p.words[i];
}

void check(bool ok, const char* op, const std::string& msg) {
  if (!ok) throw std::invalid_argument(std::string("mi::moss_dit::") + op + ": " + msg);
}

template <typename In, typename Out>
void launch(const In* in, Out* out, std::int64_t rows, std::int64_t cols, cudaStream_t stream, std::int64_t ld = -1) {
  if (rows == 0 || cols == 0) return;
  const dim3 grid(static_cast<unsigned>((cols + kTile - 1) / kTile), static_cast<unsigned>((rows + kTile - 1) / kTile));
  transpose_kernel<In, Out><<<grid, dim3(kTile, kTile / kRowsPerThread), 0, stream>>>(in, out, rows, cols, ld < 0 ? rows : ld);
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

void latent_to_tokens(const Tensor& latents, const Tensor& tokens, std::int64_t col0, cudaStream_t stream) {
  constexpr const char* op = "latent_to_tokens";
  check(latents.device == Device::kCUDA && latents.dtype == DType::kF32 && latents.rank == 3 && latents.is_contiguous(),
        op, "latents must be a contiguous CUDA f32 [C, h, w], got " + latents.describe());
  const std::int64_t c = latents.dim(0), hw = latents.dim(1) * latents.dim(2);
  check(tokens.device == Device::kCUDA && tokens.dtype == DType::kBF16 && tokens.rank == 2 && tokens.is_contiguous() &&
            tokens.dim(0) == hw && col0 >= 0 && col0 + c <= tokens.dim(1),
        op, "tokens must be a contiguous CUDA bf16 [h*w, >= col0 + C], got " + tokens.describe());
  launch(latents.ptr<const float>(), tokens.ptr<__nv_bfloat16>() + col0, c, hw, stream, tokens.dim(1));
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

void unpack_int4(const void* in, std::int64_t ld_in, void* out, std::int64_t ld_out, std::int64_t rows, std::int64_t cols,
                 cudaStream_t stream) {
  check(in != nullptr && out != nullptr && rows >= 0 && cols >= 0 && cols % 2 == 0 && ld_in >= cols / 2 && ld_out >= cols,
        "unpack_int4", "needs even cols, ld_in >= cols / 2 and ld_out >= cols");
  const std::int64_t n = rows * (cols / 2);
  if (n == 0) return;
  const auto a16 = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
  if (cols % 32 == 0 && ld_in % 16 == 0 && ld_out % 16 == 0 && a16(in) && a16(out)) {
    const std::int64_t chunks = cols / 32, threads = rows * chunks;
    unpack_int4_vec_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(in), ld_in, static_cast<std::int8_t*>(out), ld_out, rows, chunks);
    MI_CUDA_CHECK_LAUNCH();
    return;
  }
  unpack_int4_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(in), ld_in, static_cast<std::int8_t*>(out), ld_out, rows, cols / 2);
  MI_CUDA_CHECK_LAUNCH();
}

void linear_fp32_row(const Tensor& w, const float* x, const float* bias, const Tensor& out, cudaStream_t stream) {
  check(w.device == Device::kCUDA && w.dtype == DType::kBF16 && w.rank == 2 && w.is_contiguous() && w.dim(1) % 8 == 0 &&
            reinterpret_cast<std::uintptr_t>(w.data) % 16 == 0 && x != nullptr && bias != nullptr &&
            out.device == Device::kCUDA && out.dtype == DType::kBF16 && out.numel() == w.dim(0) && out.is_contiguous(),
        "linear_fp32_row", "needs a contiguous 16-byte aligned CUDA bf16 w [n, k % 8 == 0] and a bf16 out [n]");
  const auto n = static_cast<int>(w.dim(0));
  if (n == 0) return;
  constexpr int kWarps = 8;
  linear_fp32_row_kernel<<<static_cast<unsigned>((n + kWarps - 1) / kWarps), 32 * kWarps, 0, stream>>>(
      w.ptr<const __nv_bfloat16>(), x, bias, out.ptr<__nv_bfloat16>(), n, static_cast<int>(w.dim(1)));
  MI_CUDA_CHECK_LAUNCH();
}

void extrapolate_latents(const Tensor& x, const Tensor& x_prev, float c, const Tensor& out, cudaStream_t stream) {
  const auto f32 = [](const Tensor& t) { return t.device == Device::kCUDA && t.dtype == DType::kF32 && t.is_contiguous(); };
  check(f32(x) && f32(x_prev) && f32(out) && x_prev.numel() == x.numel() && out.numel() == x.numel(), "extrapolate_latents",
        "needs contiguous CUDA f32 tensors of equal size");
  const std::int64_t n = x.numel();
  if (n == 0) return;
  extrapolate_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(x.ptr<const float>(), x_prev.ptr<const float>(), c,
                                                                                out.ptr<float>(), n);
  MI_CUDA_CHECK_LAUNCH();
}

}  // namespace mi::moss_dit
