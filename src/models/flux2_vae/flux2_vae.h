// FLUX.2 VAE decoder (AutoencoderKLFlux2 decode path as the MOSS pipeline calls it, P:430-438 +
// VaeImageProcessor.postprocess): DiT latents [128, h, w] fp32 -> RGB8 [16h, 16w, 3]. The op sequence and
// its reference precision are documented in kernels/vae/vae_kernels.h (for the small decoder's widths).
//
// Widths (DecoderConfig): the MOSS bundle ships the small decoder (decoder_block_out_channels 96/192/384/384); the
// full FLUX.2 decoder (128/256/512/512, e.g. a single-file flux2-vae.safetensors, see vae_file.h) shares the latent
// space. Both have the same structure with every width scaled by 4/3, so the buffer schedules below hold for both;
// only the mid attention's head dim differs (384 / 512, both kernels/vae/attention.h). The config comes from the
// weights' shapes (decoder_config()); the static memory queries take it as their first argument.
//
// Structure (so the exact tiled path of plan step 23 can reuse it):
//   head: BN de-norm + unpack, post_quant_conv, conv_in, mid resnet 0, mid attention. All at latent
//         resolution (H/8 x W/8) and the attention is global, so the head always runs untiled.
//   tail: mid resnet 1, the four up blocks, conv_norm_out + SiLU, conv_out, RGB8. Every op is local
//         except the GroupNorms, whose statistics are per image. The whole tail reaches kTailReceptiveRadius
//         output pixels per side; the tiled path (below) only needs the halo of the segment it runs.
//
// Precision: DecodeMode::reference() (the pipeline's) follows the reference: fp32 activations, TF32 convs with
// cuDNN's rounding (bit-identical), fp32-accurate kTf32x3 attention and strict fp32 attention linears, with the
// GroupNorms' exact fixed-point statistics (vae::group_norm), which tiled decodes need. DecodeMode::torch_reference()
// swaps in torch-identical GroupNorms (torch_group_norm.h): then everything before the attention is bit-identical to
// torch. The attention is not, and later TF32 convs amplify its last-bit differences, so in either mode ~99.1-99.3% of
// the output bytes match the reference PNGs and the rest differ by 1 (torch itself, swapping only its SDPA backend, gets
// 99.5%). The two GroupNorms move 0.5-0.7% of the bytes by 1 against each other, but neither is closer to torch:
// against a diffusers decode of the same 2048^2 latents, 0.74% of the bytes differ with either. Faster modes switch the
// attention to single-pass TF32 / TF32 linears, and the tail's convs and GroupNorms to bf16 activations and
// weights; the head stays fp32 in every mode (~3% of the work, and the attention kernel is fp32-only).
//
// Memory: weights (fp32, KRSC conv layout; small ~112 MB + ~56 MB with bf16 copies, full ~199 MB + ~97 MB) in the
// arena's persistent region; activations on its stack region for the duration of one decode() (sized by
// workspace_bytes). Two regions of 2c * 16h * 16w activation elements each (c = 96 small, 128 full) are ping-ponged, so the peak is the two largest
// live tensors (up_blocks[2]'s upsampler and up_blocks[3]'s first resnet at full resolution), plus a 128 MiB
// strip scratch when a tensor reaches 2 GiB (CUTLASS's addressing limit; those convs run in row strips with
// a one-row halo, bit-identical). Stack memory is released when decode() returns, with kernels possibly still
// queued: reuse it only on the same stream.
//
// Exact tiled decode (decode_tiled, plan step 23): the head runs untiled; the tail runs in full-width row bands
// whose output is byte-identical to the untiled decode in the same mode. Two properties make that possible:
//   - CUTLASS's implicit-GEMM conv accumulates every output pixel over the same K = (r, s, c) sequence wherever the
//     pixel lies in the tensor, so a band computed with a halo of at least one row per 3x3 conv gives the very same
//     values as the whole tensor (zero padding is real only at the image border, garbage rows are discarded).
//   - vae::group_norm's statistics are exact integer sums (vae_kernels.h): a layer's statistics accumulated band by
//     band equal the whole-tensor statistics bit for bit. torch_group_norm's per-thread Welford chains run over the
//     whole tensor channel by channel and cannot be split, so tiled decodes need torch_group_norm == false. Every
//     mode but the parity mode torch_reference() uses exact statistics, so the pipeline's decode gives the same bytes
//     whether the planner runs it untiled or tiled.
// A GroupNorm needs its whole input before any band can be normalized, so the tail is cut into segments between
// checkpoint tensors kept whole in VRAM, and each segment runs one statistics pass per GroupNorm it contains
// (recomputing the segment's prefix band by band, keeping only the statistics), then an output pass that writes
// the next checkpoint (or RGB). Checkpoints, band height and the input/output overlay are chosen by a planner that
// minimizes the recomputation within a byte budget (plan_tiled). Recompute cost: a segment of one resnet does its
// first conv twice (~1.7x with the GroupNorm passes); a segment that cannot afford a checkpoint at the resolution
// it ends in recomputes its whole prefix once per GroupNorm. Measured (reference precision, whole decode vs
// untiled): 1456^2 in 0.99 GiB 1.6x, 2048^2 in 1.97 GiB 1.43x; 2048^2 in 1.47 GiB 2.4x (a full-resolution fp32
// checkpoint alone is 1.5 GiB there, so the full-resolution part is recomputed per GroupNorm).
#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "kernels/vae/vae_kernels.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "weights/model_index.h"

namespace mi::flux2_vae {

inline constexpr int kLatentChannels = 128;  // DiT latent channels (32 VAE channels x 2 x 2 patch)
inline constexpr int kPixelsPerLatent = 16;  // image pixels per DiT latent pixel along each axis

// Receptive-field radius of the tail in output pixels: 3x3 convs add one pixel at their resolution,
// i.e. 8 convs at H (8 px), 7 at H/2 (14 px), 7 at H/4 (28 px) and 8 at H/8 (64 px, mid resnet 1 included).
// Tile edges on multiples of 8 output pixels add no rounding from the nearest upsamplers; otherwise add up
// to one pixel per lower level (2 + 4 + 8 px). The head's receptive field is the whole image (attention).
inline constexpr int kTailReceptiveRadius = 114;

// decoder_block_out_channels = {c, 2c, 4c, 4c}. Supported: c = 96 (small) and c = 128 (full).
struct DecoderConfig {
  int base_channels = 96;

  static constexpr DecoderConfig small() { return {96}; }
  static constexpr DecoderConfig full() { return {128}; }
  constexpr int mid() const { return 4 * base_channels; }  // conv_in / mid block / attention head dim
  // Output channels of up_blocks[b] in execution order (decoder_block_out_channels reversed): 4c, 4c, 2c, c.
  constexpr int up_out(int b) const { return base_channels * (b < 2 ? 4 : b == 2 ? 2 : 1); }
  constexpr bool operator==(const DecoderConfig&) const = default;
  const char* name() const { return base_channels == 96 ? "small" : base_channels == 128 ? "full" : "unsupported"; }
};

// The VAE "decoder" group's widths, from decoder.conv_in.weight's shape. Throws std::runtime_error when the group is
// missing or its width is not a supported config.
DecoderConfig decoder_config(const weights::ModelIndex& index);

struct DecodeMode {
  bool bf16_convs = false;  // tail convs/GroupNorms on bf16 activations (decoder loaded with bf16 weights)
  vae::AttnPrecision attention = vae::AttnPrecision::kTf32x3;
  bool fast_linears = false;  // attention linears in TF32 instead of strict fp32
  // fp32 GroupNorms bit-identical to torch (torch_group_norm.h) instead of vae::group_norm's exact fixed-point
  // statistics. Parity tests only: torch's statistics cannot be split into bands, so decode_tiled refuses it and a
  // torch-GroupNorm decode differs from the tiled one (0.5-0.7% of the bytes by 1).
  bool torch_group_norm = false;

  // The pipeline's precision: untiled and tiled decodes give the same bytes.
  static constexpr DecodeMode reference() { return {}; }
  // reference() with torch's GroupNorm: bit-identical to torch up to the mid attention (VAE parity tests).
  static constexpr DecodeMode torch_reference() { return {false, vae::AttnPrecision::kTf32x3, false, true}; }
  static constexpr DecodeMode fast_attention() { return {false, vae::AttnPrecision::kTf32, true, false}; }
  static constexpr DecodeMode fastest() { return {true, vae::AttnPrecision::kTf32, true, false}; }
};

// Debug/parity hook, called after each stage with its output activation, NHWC [1, h, w, C] in the stage's
// dtype: "unpack", "post_quant_conv", "conv_in", "mid.resnets.0", "mid.attention", "mid.resnets.1",
// "up_blocks.0" ... "up_blocks.3" (after the upsampler, if any), "norm_out" (after SiLU), "conv_out".
// Enqueued work has not necessarily finished; synchronize before reading.
using StageHook = std::function<void(const char* stage, const Tensor& x)>;

struct LoadStats {
  double seconds = 0;       // wall time, disk to resident and converted
  std::uint64_t bytes = 0;  // payload bytes read
};

// A tiled decode schedule from Flux2VaeDecoder::plan_tiled / plan_tiled_min_memory. The tail is 17 units in
// execution order: mid resnet 1 (0), up_blocks[b] resnets and upsampler (1-4, 5-8, 9-12, 13-15), conv_norm_out +
// conv_out + RGB (16); a segment runs units [first_unit, end_unit) from the checkpoint before first_unit.
struct TilePlan {
  struct Segment {
    int first_unit = 0, end_unit = 0;
    std::int64_t band_rows = 0;  // band height at the segment's output resolution (all of its rows: one band)
  };
  std::int64_t height = 0, width = 0;  // latent grid the plan is for
  bool bf16 = false;                   // DecodeMode::bf16_convs the plan is for
  DecoderConfig config;                // decoder widths the plan is for
  int tile = 0;                        // band height in image pixels (a multiple of 8)
  std::vector<Segment> segments;
  std::size_t workspace_bytes = 0;  // stack bytes decode_tiled allocates
  double relative_cost = 0;         // estimated tail work (conv MACs + memory traffic) vs the untiled tail

  std::string describe() const;
};

class TiledRunner;

// A tiled decode saw the cancel flag (Flux2VaeDecoder::set_cancel). The decoder stays usable.
struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("flux2_vae: decode cancelled") {}
};

class Flux2VaeDecoder {
 public:
  // Persistent arena bytes the constructor allocates (same arguments).
  static std::size_t persistent_bytes(const weights::ModelIndex& index, bool bf16_weights = false,
                                      const weights::ModelIndex* latent_stats = nullptr);

  // Uploads the VAE "decoder" group of `index` (any supported widths, see decoder_config()) into the arena's persistent
  // region, rewrites the conv weights to the KRSC layout in place (plus bf16 KRSC copies when bf16_weights, needed by
  // DecodeMode::bf16_convs) and waits until everything is resident. Uses a small stack scratch while loading. The arena
  // and the streams must outlive the decoder; the indexes need not.
  //
  // latent_stats: the index whose VAE "decoder" group supplies bn.running_mean / bn.running_var, the latent
  // de-normalization the DiT latents were made for; nullptr: `index`'s own. The two indexes stay separate (the weights
  // are read through `index`, the statistics through `latent_stats`), so their tensors can never mix.
  // round_stats: the statistics are a single VAE file's (vae_file.h), whose fp32 originals are rounded to bf16 on the
  // host before their upload (read_latent_stats), as the MOSS-Image bundle stores them: the official pipeline
  // de-normalizes with those values. The pipeline sets it for a VAE file; the bundle's own statistics are uploaded as
  // stored. Throws std::runtime_error for a missing, misshapen or unexpected tensor, or a tensor outside its index's
  // files.
  Flux2VaeDecoder(const weights::ModelIndex& index, runtime::DeviceArena& arena, runtime::StreamSet& streams,
                  bool bf16_weights = false, bool direct_io = false, const weights::ModelIndex* latent_stats = nullptr,
                  bool round_stats = false);
  ~Flux2VaeDecoder();
  Flux2VaeDecoder(const Flux2VaeDecoder&) = delete;
  Flux2VaeDecoder& operator=(const Flux2VaeDecoder&) = delete;

  const LoadStats& load_stats() const { return load_stats_; }
  bool has_bf16_weights() const { return has_bf16_; }
  const DecoderConfig& config() const { return config_; }

  // Stack bytes decode() allocates for a latent grid of height x width (image = 16x that).
  static std::size_t workspace_bytes(const DecoderConfig& config, std::int64_t height, std::int64_t width,
                                     const DecodeMode& mode = {});

  // latents: contiguous CUDA fp32 [128, h, w] or [1, 128, h, w] (the pipeline's final latents).
  // rgb: contiguous u8 [16h, 16w, 3] or [1, 16h, 16w, 3], CUDA (asynchronous on `stream`) or CPU (returns
  // after the copy has landed).
  void decode(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode, cudaStream_t stream,
              const StageHook& hook = {});

  // ---- Exact tiled decode ----
  // Fastest schedule (least estimated recomputation) whose workspace fits budget_bytes; tile = 0 searches the band
  // height, otherwise it is fixed (image pixels, a multiple of 8). nullopt when nothing fits. Mode: as decode_tiled.
  static std::optional<TilePlan> plan_tiled(const DecoderConfig& config, std::int64_t height, std::int64_t width,
                                            const DecodeMode& mode, std::size_t budget_bytes, int tile = 0);
  // The schedule with the smallest workspace for this band height, and among those the fastest.
  static TilePlan plan_tiled_min_memory(const DecoderConfig& config, std::int64_t height, std::int64_t width,
                                        const DecodeMode& mode, int tile);
  // plan_tiled_min_memory(...).workspace_bytes.
  static std::size_t tiled_workspace_bytes(const DecoderConfig& config, std::int64_t height, std::int64_t width, int tile,
                                           const DecodeMode& mode);

  // Byte-identical to decode() in the same mode. Requires mode.torch_group_norm == false (every mode but
  // torch_reference()) and a plan made for this latent grid, mode and config(). The hook sees the head stages only. Allocates
  // plan.workspace_bytes of stack.
  void decode_tiled(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode, const TilePlan& plan,
                    cudaStream_t stream, const StageHook& hook = {});

  // Untiled when workspace_bytes(config(), ...) fits budget_bytes; otherwise the fastest tiled plan within it, byte-identical to the
  // untiled decode. A torch_group_norm mode tiles with exact GroupNorm statistics instead (max 1/255 from its untiled
  // decode). Returns the plan used (nullopt: untiled).
  // Throws std::runtime_error when even the smallest tiled plan does not fit.
  std::optional<TilePlan> decode_within(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode,
                                        std::size_t budget_bytes, cudaStream_t stream);

  // Cancel flag (nullptr: none) checked by decode_tiled (and decode_within when it tiles) before every band. Once
  // set, the decode waits for its queued work and throws Cancelled. While a flag is set the host issues at most one
  // band ahead of the GPU, so a cancel lands within about one band. The flag must outlive the decodes.
  void set_cancel(const std::atomic<bool>* cancel) { cancel_ = cancel; }

 private:
  friend class TiledRunner;

  struct Conv {
    Tensor w;       // fp32 [K, R, S, C]
    Tensor w_bf16;  // bf16 [K, R, S, C] (bf16_weights only)
    Tensor bias;    // fp32 [K]
  };
  struct Norm {
    Tensor gamma, beta;  // fp32 [C]
  };
  struct Resnet {
    Norm norm1, norm2;
    Conv conv1, conv2;
    Conv shortcut;  // 1x1, only when in != out channels (w.data == nullptr otherwise)
    int in_ch = 0, out_ch = 0;
  };
  struct UpBlock {
    Resnet resnets[3];
    Conv upsampler;  // conv after the nearest 2x upsample (w.data == nullptr for the last block)
  };
  struct Ctx;

  void conv(const Ctx& c, const Tensor& x, const Conv& cv, const Tensor* residual, const Tensor& y) const;
  void norm(const Ctx& c, const Tensor& x, const Norm& n, bool silu, const Tensor& y) const;
  char* resnet(const Ctx& c, const Resnet& r, char* x, char* a, char* b, std::int64_t h, std::int64_t w) const;
  void mid_attention(const Ctx& c, const Tensor& x, char* attn_ws, std::size_t attn_bytes) const;
  void run_head(const Ctx& c, const Tensor& latents, char* x, char* a, char* b, char* attn_ws, std::size_t attn_bytes) const;
  // Writes RGB8 into rgb when it is on the device, else into a stack staging buffer; returns the written tensor.
  Tensor run_tail(const Ctx& c, const Tensor& rgb) const;

  runtime::DeviceArena& arena_;
  DecoderConfig config_;
  cublasLtHandle_t lt_ = nullptr;
  LoadStats load_stats_;
  bool has_bf16_ = false;
  const std::atomic<bool>* cancel_ = nullptr;
  cudaEvent_t band_done_[2] = {nullptr, nullptr};  // tiled bands k - 2 and k - 1 (cancel throttle)

  Tensor bn_mean_, bn_var_;
  Conv post_quant_conv_, conv_in_, conv_out_;
  Resnet mid_resnets_[2];
  vae::MidAttentionWeights mid_attn_;
  UpBlock up_[4];
  Norm norm_out_;
};

}  // namespace mi::flux2_vae
