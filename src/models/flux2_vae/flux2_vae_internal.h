// Internals shared by the untiled (flux2_vae.cpp) and exact tiled (flux2_vae_tiled.cpp) decode paths.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/tensor.h"
#include "models/flux2_vae/flux2_vae.h"

namespace mi::flux2_vae::detail {

inline constexpr int kMidCh = 384;
inline constexpr int kTailMaxCh = 192;  // widest full-resolution tensor: up_blocks[2]'s upsampler (192 ch at H x W)
// Up blocks in execution order (decoder_block_out_channels reversed): output channels, upsampler.
inline constexpr int kUpOut[4] = {384, 384, 192, 96};
inline constexpr bool kUpSample[4] = {true, true, true, false};
inline constexpr int kResnetsPerBlock = 3;  // layers_per_block + 1
// CUTLASS's conv iterators use 32-bit byte offsets: tensors of 2 GiB or more (the 192-channel full-resolution
// fp32 tensors from ~1670^2 up) are convolved in row strips with a halo (see Flux2VaeDecoder::conv()).
inline constexpr std::size_t kCutlassMaxBytes = std::size_t{1} << 31;
inline constexpr std::size_t kStripScratchBytes = std::size_t{128} << 20;

[[noreturn]] void fail(const std::string& msg);

inline Tensor act(void* p, DType dt, std::int64_t h, std::int64_t w, std::int64_t c) {
  return Tensor::view(p, dt, Device::kCUDA, {1, h, w, c});
}

// Stack layout of one untiled decode: two ping-pong regions (each split into halves) and the GroupNorm/conv scratch.
struct Layout {
  std::size_t region = 0;  // bytes per region, a multiple of 2 * kArenaAlign
  std::size_t aux = 0;     // conv / GroupNorm scratch, enough for every decoder shape at this size
  std::size_t strip = 0;   // conv output strip scratch (only for tensors CUTLASS cannot address at once)
};
Layout layout(std::int64_t h, std::int64_t w, const DecodeMode& mode);

// Validates decode()'s arguments; returns the latent grid {h, w}.
void check_decode_args(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode, bool has_bf16);

}  // namespace mi::flux2_vae::detail

namespace mi::flux2_vae {

struct Flux2VaeDecoder::Ctx {
  cudaStream_t stream;
  DType dt;  // activation dtype of the current phase (head: fp32)
  const DecodeMode* mode;
  void* aux;
  std::size_t aux_bytes;
  char* r0;  // untiled ping-pong regions (unused by the tiled tail)
  char* r1;
  std::size_t region;
  void* strip;
  std::size_t strip_bytes;
  std::int64_t h, w;  // DiT latent grid
  const StageHook* hook;

  void stage(const char* name, const Tensor& x) const {
    if (*hook) (*hook)(name, x);
  }
};

}  // namespace mi::flux2_vae
