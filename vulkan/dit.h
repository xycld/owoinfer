// The MOSS-Image DiT (MOSSImageTransformer, transformer/pipeline_moss_image.py) on the Vulkan block kernels, every weight
// resident in device memory (bf16, about 9.5 GiB):
//   - encode_text(): the text refiner (text_embedder: a linear, then 2 un-modulated blocks), once per prompt.
//   - forward(): one sampling step for several rows (cond, uncond, ...). The time embedding and the shared modulation run
//     once; the rows' joint sequences [video_r | text_r] are concatenated, so each linear runs once over all rows and
//     attention is one varlen call per run of rows. A block that some rows skip runs on the other rows' token ranges only.
// Each op follows the reference's order and bf16/fp32 rounding, as the engine's CUDA forward (src/models/moss_dit).
// Work is submitted op by op, so no single submit runs long on a slow device.
#pragma once

#include <bitset>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "block.h"
#include "runtime.h"

namespace mi::vk {

inline constexpr int kDiTLayers = 42;
inline constexpr int kDiTTextDim = 4096;
inline constexpr int kDiTFreqDim = 256;
inline constexpr int kDiTLatentChannels = 128;

using SkipSet = std::bitset<kDiTLayers>;

class Dit {
 public:
  // transformer_dir: the model's transformer/ directory (config.json and the sharded safetensors).
  Dit(Device& device, const BlockKernels& kernels, const std::filesystem::path& transformer_dir);
  ~Dit();
  Dit(const Dit&) = delete;
  Dit& operator=(const Dit&) = delete;

  std::uint64_t weight_bytes() const { return weight_bytes_; }

  // The text refiner on one prompt's text-encoder output, bf16 bits [length, 4096]: device bf16 [length, 3072].
  std::unique_ptr<Buffer> encode_text(const std::vector<std::uint16_t>& embeds, std::uint32_t length);

  struct Row {
    const Buffer* text = nullptr;  // from encode_text()
    std::uint32_t length = 0;
    SkipSet skip;
  };
  // One step: latents f32 [128, height, width] (the sampler's), timestep as the scheduler passes it. Returns each row's
  // prediction, f32 [128, height, width].
  std::vector<std::vector<float>> forward(const std::vector<float>& latents, std::uint32_t height, std::uint32_t width,
                                          float timestep, const std::vector<Row>& rows);

  // Milliseconds of the last forward(), summed per op kind over its blocks (block_op_name order), then the rest.
  const std::vector<double>& last_op_ms() const { return op_ms_; }

 private:
  struct Weight;
  struct TextBlock;
  struct Scratch;

  Buffer& scratch(std::unique_ptr<Buffer>& slot, VkDeviceSize bytes);
  void time_modulation(float timestep);

  Device& device_;
  const BlockKernels& kernels_;
  std::uint64_t weight_bytes_ = 0;
  std::vector<std::unique_ptr<Buffer>> owned_;
  std::vector<BlockWeights> blocks_;
  struct Globals {
    VkDeviceAddress video_w, video_b, time0_w, time0_b, time2_w, time2_b, mod_w, mod_b, head_w, head_b, final_norm,
        final_w, text_w, text_b;
  } g_{};
  struct TextWeights {
    VkDeviceAddress norm1, norm2, q_norm, k_norm, qkv_w, qkv_b, proj_w, proj_b, up_w, up_b, gate_w, gate_b, down_w,
        down_b;
  };
  std::vector<TextWeights> text_blocks_;

  std::unique_ptr<Buffer> x_, y_, qkv_, attn_, ffn_, rope_, tokens_, proj_, up_, gate_, embeds_;
  std::unique_ptr<Buffer> mod_, head_, t_in_, hidden_, time_;
  std::vector<double> op_ms_;
};

// The RoPE table rows of one joint sequence (P:178-185), bf16 bits [height * width + text_length, 128]: sin of the 64
// phases, then their cos. height = width = 0 gives the text rows alone.
std::vector<std::uint16_t> rope_table(std::uint32_t height, std::uint32_t width, std::uint32_t text_length);

}  // namespace mi::vk
