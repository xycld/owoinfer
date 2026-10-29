// FLUX.2 VAE encoder (diffusers AutoencoderKLFlux2.encode + Flux2Pipeline._encode_vae_image): RGB image
// [H, W, 3] -> normalized DiT latents [128, H/16, W/16] fp32, the layout the DiT denoises and Flux2VaeDecoder
// decodes (the decoder's bn_denorm_unpack is the exact inverse of the last step here). The op sequence and its
// reference precision are documented in kernels/vae/encoder_kernels.h and vae_kernels.h.
//
// Structure, for an H x W image (H, W multiples of 16):
//   preprocess  2 * (u8 / 255) - 1, padded to 4 channels                                  [H, W, 4]
//   level 0     conv_in 3->128, 2 resnets 128, downsample (stride 2)          @ H         -> C1 [H/2, 128]
//   level 1     resnet 128->256 (1x1 shortcut), resnet 256, downsample        @ H/2       -> C2 [H/4, 256]
//   level 2     resnet 256->512 (1x1 shortcut), resnet 512, downsample        @ H/4       -> C3 [H/8, 512]
//   tail        2 resnets 512, mid resnet, mid attention (global, 1 head x 512), mid resnet, GN + SiLU,
//               conv_out 512->64, quant_conv 64->64 = moments (mean | logvar)                 @ H/8
//   latent_out  mean (the distribution's mode) or mean + std * noise; 2x2 patchify; BN normalisation
// Precision: fp32 activations, fp32-accurate 3xTF32 convs by default (EncodeOptions::conv; single-pass TF32 like the
// decoder's convs on request), GroupNorms with vae::group_norm's exact fixed-point statistics, fp32-accurate kTf32x3
// attention and strict fp32 attention linears (the reference's mem-efficient SDPA and matmuls).
//
// Memory: weights fp32 (~137 MB, KRSC conv layout) in the arena's persistent region, activations on its stack for
// one encode(). Levels 0-2 dominate (three 128-channel full-resolution tensors are 1.5 GiB at 1024^2 and 6 GiB at
// 2048^2), so each level can run untiled or in exact row bands:
//
// Exact banded levels: a level runs from its input checkpoint (image, C1, C2; kept whole) to its output
// checkpoint in full-width row bands, byte-identical to the untiled level, for the same reasons as the decoder's
// tiled decode:
//   - CUTLASS's implicit-GEMM convs (stride 1 and the stride-2 downsample) accumulate every output pixel over the
//     same (r, s, c) sequence wherever the pixel lies, so a band computed with a halo gives the whole-tensor values
//     (zero padding is real only at the image border; rows next to a band edge are garbage and discarded).
//   - vae::group_norm's statistics are exact integer sums: accumulated band by band they equal the whole-tensor
//     statistics bit for bit.
// A GroupNorm needs its whole input before any band can be normalized, so a banded level runs one statistics pass
// per GroupNorm (4 per level: recomputing the level's prefix band by band and keeping only the statistics), then an
// output pass that writes the output checkpoint. The band halo follows from the op graph (each 3x3 conv needs one
// row per side, the stride-2 conv rows 2i..2i+2). The tail runs untiled: its tensors are 1/64 of the image area
// and its attention is global.
#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "kernels/vae/encoder_kernels.h"
#include "kernels/vae/vae_kernels.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "weights/model_index.h"

namespace mi::flux2_vae_encoder {

inline constexpr int kLatentChannels = 128;  // DiT latent channels (32 VAE channels x 2 x 2 patch)
inline constexpr int kPixelsPerLatent = 16;  // image pixels per DiT latent pixel along each axis
inline constexpr int kLevels = 3;            // levels 0-2 (full, 1/2, 1/4 resolution) can run banded

// What encode() returns from the posterior N(mean, exp(logvar)) before the patchify + BN normalisation.
enum class LatentMode {
  kMean,    // the mode (= mean): what FLUX.2 image conditioning and img2img use (retrieve_latents "argmax")
  kSample,  // mean + std * noise, noise = torch.randn([1, 32, H/8, W/8], generator=torch.Generator("cpu").manual_seed(seed))
};

struct EncodeOptions {
  LatentMode mode = LatentMode::kMean;
  std::uint64_t seed = 0;  // kSample only
  // Conv math. kTf32x3 (fp32-accurate) is the default: the reference (cuDNN with allow_tf32) lands within ~3e-7
  // cosine of an all-fp32 encode on the test images, while single-pass TF32 convs drift to cosine 0.99998-0.99999
  // through the encoder's 30-conv chain (below the 0.99999 bar at 1456^2 and up). kTf32 is ~2.3x faster.
  vae::ConvMath conv = vae::ConvMath::kTf32x3;
  vae::AttnPrecision attention = vae::AttnPrecision::kTf32x3;
  bool fast_linears = false;  // attention linears in TF32 instead of strict fp32
};

// Debug/parity hook, called with each checkpoint: "down_blocks.0" ... "down_blocks.2" (after the downsampler),
// "down_blocks.3", "mid.resnets.0", "mid.attention", "mid.resnets.1", "conv_out", "moments" (after quant_conv),
// all NHWC [1, h, w, C] fp32, in encode()'s stack workspace. The tensor is complete only for work ordered after the
// hook call on `stream` (the producing kernels may still be queued: synchronize `stream` before a host read, or enqueue
// the copy on `stream` inside the hook), and only until encode() continues: later stages overwrite its memory and the
// workspace is released when encode() returns. Copy what you need inside the hook.
using StageHook = std::function<void(const char* stage, const Tensor& x)>;

struct LoadStats {
  double seconds = 0;       // wall time, disk to resident and converted
  std::uint64_t bytes = 0;  // payload bytes read
};

// How encode() runs levels 0-2 for one image size: band_rows[l] rows per band at level l's resolution
// (H >> l), or 0 for untiled.
struct TilePlan {
  std::int64_t height = 0, width = 0;  // image size the plan is for
  std::array<std::int64_t, kLevels> band_rows{};
  std::size_t workspace_bytes = 0;  // stack bytes encode() allocates
  double relative_cost = 1;         // estimated conv MACs vs the untiled encode (recompute + halo overhead)

  bool tiled() const { return band_rows[0] != 0 || band_rows[1] != 0 || band_rows[2] != 0; }
  std::string describe() const;
};

class Flux2VaeEncoder {
 public:
  // vae_file: the diffusers layout (vae/diffusion_pytorch_model.safetensors: encoder.down_blocks.*, quant_conv.*, bn.*)
  // or the original single-file layout (encoder.down.*.block.*, encoder.quant_conv.*, bn.*), optionally under a common
  // key prefix ending in '.' (e.g. "vae."). Every tensor under <prefix>encoder. / <prefix>quant_conv. must be one the
  // encoder uses. Throws std::runtime_error for a missing, extra, non-F32 or misshapen tensor, or a file with several
  // encoders. latent_stats (see the constructor) given: vae_file's own bn.* tensors are not needed.
  static std::size_t persistent_bytes(const std::filesystem::path& vae_file, const weights::ModelIndex* latent_stats = nullptr);

  // Uploads the encoder, quant_conv and bn statistics into the arena's persistent region (conv weights rewritten to
  // KRSC) and waits until they are resident. Uses a small stack scratch while loading. The arena and the streams
  // must outlive the encoder; the index need not.
  //
  // latent_stats: as Flux2VaeDecoder's, the index whose VAE "decoder" group supplies bn.running_mean / running_var, the
  // latent normalization the DiT works in; nullptr: vae_file's own. round_stats: those statistics are a single VAE
  // file's fp32 originals, rounded to bf16 on the host before their upload (flux2_vae::read_latent_stats), as the
  // MOSS-Image bundle stores them: the official pipeline normalizes with those values. The pipeline passes a VAE file's
  // index (flux2_vae::index_vae_file) with round_stats; the bundle's VAE file is read as stored.
  //
  // Failure: the file is validated (persistent_bytes()) before anything is allocated, so a bad file throws with the
  // arena untouched. An I/O or CUDA error during the upload (or bad latent_stats tensors) throws after part of the
  // persistent region was allocated; like Flux2VaeDecoder, the arena cannot free persistent allocations one by one, so
  // the caller recreates or resets the arena.
  Flux2VaeEncoder(const std::filesystem::path& vae_file, runtime::DeviceArena& arena, runtime::StreamSet& streams,
                  bool direct_io = false, const weights::ModelIndex* latent_stats = nullptr, bool round_stats = false);
  ~Flux2VaeEncoder();
  Flux2VaeEncoder(const Flux2VaeEncoder&) = delete;
  Flux2VaeEncoder& operator=(const Flux2VaeEncoder&) = delete;

  const LoadStats& load_stats() const { return load_stats_; }

  // ---- Plans and memory queries (image size in pixels, multiples of 16) ----
  static TilePlan plan_untiled(std::int64_t height, std::int64_t width);
  // Stack bytes of the untiled encode, plan_untiled(...).workspace_bytes.
  static std::size_t workspace_bytes(std::int64_t height, std::int64_t width);
  // The fastest plan whose workspace fits budget_bytes (untiled when it fits); nullopt when nothing fits.
  static std::optional<TilePlan> plan_tiled(std::int64_t height, std::int64_t width, std::size_t budget_bytes);
  // The smallest workspace plan: 16-row bands at every level taller than 16 rows, the others untiled.
  static TilePlan plan_min_memory(std::int64_t height, std::int64_t width);
  // An explicit plan (band_rows[l]: 0 = untiled, else even and >= 2); throws std::invalid_argument when invalid.
  static TilePlan make_plan(std::int64_t height, std::int64_t width, const std::array<std::int64_t, kLevels>& band_rows);

  // rgb: contiguous [H, W, 3] or [1, H, W, 3], u8 or f32 in [0, 1], CUDA or CPU (staged through the workspace).
  // latents: contiguous CUDA fp32 [128, H/16, W/16] or [1, 128, H/16, W/16]. Asynchronous on `stream` (for
  // kSample the noise is drawn on the host first). Allocates plan.workspace_bytes of stack; tiled plans give
  // the same bytes as the untiled one. A plan from make_plan / plan_tiled / plan_min_memory for this image size.
  // Host input: copied with cudaMemcpyAsync on `stream`. From pageable memory that call returns once the bytes are
  // staged, so the buffer may be reused when encode() returns; from pinned (page-locked) memory the copy is truly
  // asynchronous and the buffer must stay unchanged until `stream` has passed it (synchronize, or record an event).
  // A CUDA rgb is read by kernels on `stream` and follows the same rule.
  // The stack workspace is released when encode() returns, with kernels possibly still queued: reuse that arena
  // memory only on the same stream (as Flux2VaeDecoder::decode).
  void encode(const Tensor& rgb, const Tensor& latents, const EncodeOptions& options, const TilePlan& plan,
              cudaStream_t stream, const StageHook& hook = {});
  void encode(const Tensor& rgb, const Tensor& latents, const EncodeOptions& options, cudaStream_t stream,
              const StageHook& hook = {});
  // Untiled when it fits budget_bytes, else the fastest tiled plan within it. Returns the plan used; throws
  // std::runtime_error when even plan_min_memory does not fit.
  TilePlan encode_within(const Tensor& rgb, const Tensor& latents, const EncodeOptions& options, std::size_t budget_bytes,
                         cudaStream_t stream);

  struct Conv {
    Tensor w;     // fp32 [K, R, S, C]
    Tensor bias;  // fp32 [K]
  };
  struct Norm {
    Tensor gamma, beta;  // fp32 [C]
  };
  struct Resnet {
    Norm norm1, norm2;
    Conv conv1, conv2;
    int in_ch = 0, out_ch = 0;
  };
  struct Weights {
    // Levels 0-2 in the order of their op programs (see the .cpp): conv_in or the 1x1 shortcut, resnet 0 conv1,
    // conv2, resnet 1 conv1, conv2, downsampler; norms: resnet 0 norm1, norm2, resnet 1 norm1, norm2.
    std::array<std::vector<Conv>, kLevels> level_convs;
    std::array<std::vector<Norm>, kLevels> level_norms;
    Resnet tail_resnets[4];  // down_blocks.3 resnets 0, 1, then mid resnets 0, 1 (all 512 -> 512)
    vae::MidAttentionWeights attn;
    Norm norm_out;
    Conv conv_out, quant_conv;
    Tensor bn_mean, bn_var;
  };

 private:
  void run_tail(char* ws, const Tensor& c3, const Tensor& latents, const Tensor* noise, const EncodeOptions& options,
                cudaStream_t stream, const StageHook& hook) const;

  runtime::DeviceArena& arena_;
  cublasLtHandle_t lt_ = nullptr;
  LoadStats load_stats_;
  Weights w_;
};

}  // namespace mi::flux2_vae_encoder
