// The MOSS-Image DiT (MOSSImageTransformer, transformer/pipeline_moss_image.py) on the HIP kernels, weights in bf16:
//   - encode_text(): the text refiner (text_embedder: a linear, then 2 un-modulated blocks), once per prompt.
//   - forward(): one sampling step for several rows (cond, uncond, ...). The time embedding and the shared modulation run
//     once; the rows' joint sequences [video_r | text_r] are concatenated, so each linear runs once over all rows and
//     attention is one varlen call per run of rows. A block that some rows skip runs on the other rows' token ranges only.
// Each op follows the reference's order and bf16/fp32 rounding, as the engine's CUDA forward (src/models/moss_dit). The
// 42 blocks run hip_block's packed path: their GEMM weights are packed on the host at load (hip_block::pack_fragments'
// layout). Blocks past the device budget stay in pinned host memory and are copied into one of two device slots on a
// copy stream while the block before them runs.
#pragma once

#include <bitset>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "hip_block.h"
#include "runtime.h"

namespace mi::hip {

inline constexpr int kDiTLayers = 42;
inline constexpr int kDiTDim = hip_block::kDim;
inline constexpr int kDiTHeads = hip_block::kHeads;
inline constexpr int kDiTHeadDim = hip_block::kHeadDim;
inline constexpr int kDiTFfnDim = hip_block::kFfnDim;
inline constexpr int kDiTTextDim = 4096;
inline constexpr int kDiTFreqDim = 256;
inline constexpr int kDiTLatentChannels = 128;

using SkipSet = std::bitset<kDiTLayers>;

struct DitOptions {
  // Device memory the weights may take, in bytes; 0: every block resident. Blocks past it stay in pinned host memory and
  // are copied into a device slot before they run (two slots, counted in the budget), every step.
  std::uint64_t device_budget = 0;
  // Time each op of every forward with events on the stream (last_op_ms()).
  bool time_ops = false;
};

class Dit {
 public:
  // transformer_dir: the model's transformer/ directory (the sharded safetensors and their index).
  Dit(Device& device, const std::filesystem::path& transformer_dir, const DitOptions& options = {});
  ~Dit();
  Dit(const Dit&) = delete;
  Dit& operator=(const Dit&) = delete;

  std::uint64_t weight_bytes() const { return weight_bytes_; }  // device memory, the streaming slots included
  std::uint64_t host_weight_bytes() const { return host_weight_bytes_; }
  int resident_blocks() const { return resident_blocks_; }
  // Device memory of one block's packed weights, resident or in a streaming slot (the pinned host memory of a streamed
  // one is the same).
  static std::uint64_t block_bytes();
  // The least device memory the weights take under a budget: the globals and the text refiner, and the two slots.
  static std::uint64_t minimum_weight_bytes();
  // The blocks a DitOptions::device_budget keeps on the device, as the constructor chooses them (0: all 42).
  static int resident_blocks_for(std::uint64_t device_budget);
  // Device memory of forward()'s activations over `tokens` joint tokens of `rows` rows on an h x w = hw latent grid.
  static std::uint64_t activation_bytes(std::uint64_t tokens, std::uint64_t rows, std::uint64_t hw);

  // The text refiner on one prompt's text-encoder output, bf16 bits [length, 4096]: device bf16 [length, 3072].
  DeviceBuffer encode_text(const std::vector<std::uint16_t>& embeds, std::uint32_t length);

  struct Row {
    const DeviceBuffer* text = nullptr;  // from encode_text()
    std::uint32_t length = 0;
    SkipSet skip;
  };
  // One step: latents f32 [128, height, width] (the sampler's), timestep as the scheduler passes it. Returns each row's
  // prediction, f32 [128, height, width].
  std::vector<std::vector<float>> forward(const std::vector<float>& latents, std::uint32_t height, std::uint32_t width,
                                          float timestep, const std::vector<Row>& rows);

  // With DitOptions::time_ops, the last forward()'s device milliseconds per op kind summed over its blocks
  // (hip_block::block_op_name order), then the time the stream waited for streamed blocks' copies, then the rest (time
  // embedding, embedders, head). Empty otherwise.
  const std::vector<double>& last_op_ms() const { return op_ms_; }

 private:
  struct Globals {
    const std::uint16_t *video_w, *video_b, *time0_w, *time0_b, *time2_w, *time2_b, *mod_w, *mod_b, *head_w, *head_b,
        *final_norm, *final_w, *text_w, *text_b;
  };
  struct TextBlock {
    const std::uint16_t *norm1, *norm2, *q_norm, *k_norm, *qkv_w, *qkv_b, *proj_w, *proj_b, *up_gate_w, *up_gate_b,
        *down_w, *down_b;
  };
  DeviceBuffer& scratch(DeviceBuffer& b, std::size_t bytes);
  void stage_time(float timestep);  // the time sinusoid to the device, the time scratch sized
  void time_modulation();            // the time MLP and both modulations, on the stream
  void mark(int what);  // time_ops: an event on the stream, its interval since the last one counted as `what`

  Device& device_;
  DitOptions options_;
  Stream stream_, copy_stream_;
  std::uint64_t weight_bytes_ = 0, host_weight_bytes_ = 0;
  int resident_blocks_ = 0;
  std::vector<DeviceBuffer> owned_;     // globals, text blocks, resident blocks
  std::vector<HostBuffer> host_blocks_;  // per block: its packed weights when streamed, else empty
  std::vector<int> slot_of_;            // per block: its device slot when streamed, else -1
  DeviceBuffer slots_[2];
  std::unique_ptr<Event> slot_free_[2], copy_done_[2];
  std::vector<hip_block::BlockWeights> blocks_;
  Globals g_{};
  std::vector<TextBlock> text_blocks_;

  DeviceBuffer x_, y_, qkv_, attn_, ffn_, rope_, tokens_, proj_, up_gate_, embeds_, pack_, pack_attn_, cu_;
  DeviceBuffer mod_, head_, t_in_, hidden_, time_;

  std::vector<std::unique_ptr<Event>> events_;  // time_ops
  std::vector<int> event_what_;
  std::size_t events_used_ = 0;
  std::vector<double> op_ms_;
};

}  // namespace mi::hip
