// The FLUX.2 VAE decoder as the MOSS-Image pipeline runs it (src/models/flux2_vae, pipeline_moss_image.py:430-438): the
// DiT's final latents, fp32 NCHW [128, h, w], to RGB8 [16h, 16w, 3]. Activations are fp32 NHWC throughout, as in the
// engine's reference precision (DecodeMode::reference()):
//   latent de-normalization + unpack -> post_quant_conv 1x1 -> conv_in 3x3 -> mid resnet 0 -> mid attention
//   -> mid resnet 1 -> 4 up blocks (3 resnets each, nearest 2x upsample + 3x3 conv after the first three)
//   -> GroupNorm + SiLU -> conv_out 3x3 -> clamp and the image processor's rounding to uint8.
// Convolutions round both operands to TF32 like cuDNN's (torch's allow_tf32 convolutions) and accumulate in fp32; the
// nearest upsample is read inside the following convolution. The attention's linears are fp32 GEMMs (allow_tf32 off in the
// reference), its scores and P V are fp32-accurate to 16 significant bits per operand on the cooperative-matrix path and
// plain fp32 on the other. GroupNorm statistics are fp32 per chunk of pixels (mean and sum of squared deviations), merged
// pairwise. Accumulation orders differ from cuDNN's and torch's, so outputs are close to the engine's, not bit-identical.
//
// Two paths, chosen as BlockKernels does: kCoop runs the convolutions and GEMMs on bf16 cooperative matrices
// (Capabilities::coop_bf16), splitting each fp32 operand into bf16 pieces whose products are exact (TF32 = 2 pieces);
// kPlain needs no cooperative matrices, 16-bit types or particular subgroup size.
//
// Memory (untiled): two regions of 2c x 16h x 16w fp32 each (c = 96 for the small decoder), as the engine's ping-pong
// schedule, plus GroupNorm partials, the input latents and the RGB output; the mid attention's scratch lives in the second
// region. At 1024x1024 that is 1.5 GiB plus about 112 MB of fp32 weights.
// Submits: every op is recorded into submits of at most `max_macs_per_submit` multiply-adds (a convolution larger than
// that runs in bands of output rows, each its own submit), so no single submit runs long on a slow GPU.
#pragma once

#include <filesystem>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"

namespace mi::vk {

enum class VaePath { kCoop, kPlain };
const char* vae_path_name(VaePath path);

// How a convolution or GEMM rounds its operands (see vae_conv_coop.comp).
enum class VaePrecision {
  kTf32,     // both operands rounded to TF32, products exact: torch's TF32 convolution up to accumulation order
  kFp32,     // fp32 GEMM: exact products on the coop path when B is bf16-valued (A in 3 bf16 pieces), else 16-bit B pieces
  kSplit16,  // coop: 2 bf16 pieces per operand (16 significant bits each); plain: fp32
};

// out[m, n] = (sum_k A[m, k] B[n, k] + bias[n]) + residual[m, n], fp32, for rows m in [m0, m), with B row-major [n, k].
// taps 0: A is row-major with stride lda. taps 1 / 9: a 1x1 / 3x3 convolution (zero padding 1) over an NHWC input of c
// channels for an h x w output, whose pixel index is the row; upsample: the input is h/2 x w/2, read through a nearest 2x
// upsample. k = taps * c for a convolution. Requirements: c a multiple of 8, k, lda and ldb multiples of 4, ldo a multiple
// of 4 on the coop path for n >= 64, 16-byte aligned addresses. residual may equal out; out must not overlap A or B.
struct ConvArgs {
  VkDeviceAddress a = 0, b = 0, out = 0, bias = 0, residual = 0;
  std::uint32_t m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldo = 0;
  std::uint32_t taps = 0;
  bool upsample = false;
  std::uint32_t h = 0, w = 0, c = 0;
  std::uint32_t m0 = 0;
  bool b_bf16 = false;  // every B value is a bf16 value (fewer pieces on the coop path)
};

class VaeKernels {
 public:
  // kCoop where the device has bf16 cooperative matrices, unless `path` names one. Throws if the device cannot run it.
  explicit VaeKernels(Device& device);
  VaeKernels(Device& device, VaePath path);
  ~VaeKernels();
  VaeKernels(const VaeKernels&) = delete;
  VaeKernels& operator=(const VaeKernels&) = delete;

  VaePath path() const { return path_; }

  // Every record_* records its dispatches and a barrier after them.
  void record_conv(VkCommandBuffer cb, const ConvArgs& args, VaePrecision precision) const;
  // Rows of one band of a convolution or GEMM: a multiple of the row tile every kernel uses.
  static constexpr std::uint32_t kRowAlign = 128;

  // GroupNorm with 32 groups over x [pixels, c] (fp32 NHWC, c a multiple of 32): statistics into stats ({mean, rstd} per
  // group, 256 bytes) through part (gn_part_bytes), then y = GN(x) * gamma + beta, SiLU optionally; y may be x.
  static std::uint64_t gn_part_bytes(std::uint32_t pixels, std::uint32_t c);
  void record_gn_stats(VkCommandBuffer cb, VkDeviceAddress x, std::uint32_t pixels, std::uint32_t c, VkDeviceAddress part,
                       VkDeviceAddress stats, float eps) const;
  void record_gn_apply(VkCommandBuffer cb, VkDeviceAddress x, VkDeviceAddress y, std::uint32_t pixels, std::uint32_t c,
                       VkDeviceAddress stats, VkDeviceAddress gamma, VkDeviceAddress beta, bool silu) const;
  // s[r, :l] (row stride ld) = softmax(s[r, :l] * scale), in place, rows r < rows.
  void record_softmax(VkCommandBuffer cb, VkDeviceAddress s, std::uint32_t rows, std::uint32_t l, std::uint32_t ld,
                      float scale) const;
  // out[c, r] = in[r, c]: in [r_count, c_count] with row stride ld_in, out with row stride ld_out.
  void record_transpose(VkCommandBuffer cb, VkDeviceAddress in, VkDeviceAddress out, std::uint32_t r_count,
                        std::uint32_t c_count, std::uint32_t ld_in, std::uint32_t ld_out) const;
  // z [128, h, w] NCHW -> y [2h, 2w, 32] NHWC: z * scale + mean per latent channel (scale = sqrt(var + eps), host-made).
  void record_latent_unpack(VkCommandBuffer cb, VkDeviceAddress z, VkDeviceAddress scale, VkDeviceAddress mean,
                            VkDeviceAddress y, std::uint32_t h, std::uint32_t w) const;
  // x [n] fp32 -> rgb, n bytes packed into 32-bit words (the last word's bytes past n are 0).
  void record_rgb8(VkCommandBuffer cb, VkDeviceAddress x, VkDeviceAddress rgb, std::uint32_t n) const;

 private:
  const Pipeline& conv_pipeline(const ConvArgs& args, VaePrecision precision, std::uint32_t& tile_rows,
                                std::uint32_t& tile_cols) const;

  Device& device_;
  VaePath path_;
  mutable std::map<std::vector<std::uint32_t>, std::unique_ptr<Pipeline>> conv_;
  std::unique_ptr<Pipeline> gn_stats_, gn_finalize_, gn_apply_, softmax_, transpose_, unpack_, rgb8_;
};

struct VaeDecodeStats {
  double seconds = 0;              // decode() wall time, uploads and downloads included
  std::uint64_t device_bytes = 0;  // device memory decode() allocated (weights not included)
  int submits = 0;
  // With set_profile(true): wall seconds per op kind (one submit per op), in order of first use.
  std::vector<std::pair<std::string, double>> profile;
};

class VaeDecoder {
 public:
  // Loads the decoder ("decoder.*", "post_quant_conv.*") and the latent statistics ("bn.running_mean", "bn.running_var")
  // from a FLUX.2 VAE safetensors file in diffusers' naming, fp32 or bf16. The widths come from the weights' shapes
  // (decoder_block_out_channels c, 2c, 4c, 4c).
  VaeDecoder(Device& device, const std::filesystem::path& safetensors_path);
  VaeDecoder(Device& device, const std::filesystem::path& safetensors_path, VaePath path);
  ~VaeDecoder();
  VaeDecoder(const VaeDecoder&) = delete;
  VaeDecoder& operator=(const VaeDecoder&) = delete;

  const VaeKernels& kernels() const { return *kernels_; }
  int base_channels() const { return base_; }
  std::uint64_t weight_bytes() const;
  // Device bytes decode() allocates for an h x w latent grid.
  std::uint64_t workspace_bytes(std::uint32_t h, std::uint32_t w) const;
  // The largest single buffer of that workspace (each ping-pong region).
  std::uint64_t largest_buffer_bytes(std::uint32_t h, std::uint32_t w) const;
  void set_max_macs_per_submit(std::uint64_t macs) { max_macs_ = macs; }
  // Every op in its own submit, timed (VaeDecodeStats::profile): slower, for finding where time goes.
  void set_profile(bool on) { profile_ = on; }

  // latents: fp32 NCHW [128, h, w], the pipeline's final latents. rgb: resized to 16h x 16w x 3. image: when not null, the
  // decoder's fp32 output [16h, 16w, 3] before the clamp.
  VaeDecodeStats decode(const float* latents, std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb,
                        std::vector<float>* image = nullptr);

  struct Weights;

 private:
  Device& device_;
  std::unique_ptr<VaeKernels> kernels_;
  std::unique_ptr<Weights> weights_;
  int base_ = 0;
  std::uint64_t max_macs_ = 8ull << 30;
  bool profile_ = false;
};

}  // namespace mi::vk
