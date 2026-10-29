// Control adapters (roadmap P8): conditioning-image guided generation from one file that pairs a low-rank adapter with
// a widened DiT input embedder (e.g. the MOSS-Image colour/depth control adapter).
//
// File format (safetensors, read as a file format only):
//   - video_embedder.weight [dim, 128 * (1 + G)] and video_embedder.bias [dim] (F32, F16 or BF16) replace the DiT's
//     input embedder. Its input is the channel concatenation [noisy latents (128) | control latent 1 (128) | ... |
//     control latent G (128)], flattened to tokens like the latents;
//   - every other tensor is a LoRA in a format models/lora reads (rank-64 PEFT lora_A / lora_B on the block linears and
//     the text refiner in the published file), applied by the runtime LoRA stack;
//   - metadata `control_channels` (optional): the G control kinds in channel order, comma-separated ("color,depth"),
//     else "control0" ... "control<G-1>".
//
// Semantics (the published adapter's training and sampling code, and ComfyUI's loader and conditioning node agree):
//   - a control latent is its conditioning image (8-bit RGB; a depth map is grayscale replicated to RGB, near = white)
//     at the output size, VAE-encoded to the posterior mean, 2x2-patchified and BN-normalised exactly like the image
//     latents (flux2_vae_encoder's output);
//   - it is clean (never noised), the same for every CFG row and every step it is active in;
//   - a missing input, or a step outside its window, is the zero latent ("no control": the adapter was trained with
//     control dropout to zeros);
//   - its strength multiplies the latent: the embedder is linear and zero means no control, so this scales the control's
//     contribution to the image tokens;
//   - the replacement embedder is never scaled; the LoRA strength scales only the adapters.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "weights/safetensors.h"

namespace mi::control {

inline constexpr std::int64_t kGroupChannels = 128;  // channels of one control latent (the DiT's latent channels)
inline constexpr std::int64_t kDim = 3072;           // the MOSS-Image DiT's width (the embedder's output features)
inline constexpr const char* kEmbedderWeight = "video_embedder.weight";
inline constexpr const char* kEmbedderBias = "video_embedder.bias";

struct ControlError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// A control adapter file's header facts.
struct ControlFile {
  std::filesystem::path path;
  std::vector<std::string> kinds;  // one per control latent, in channel order
  std::int64_t in_features = 0;    // kGroupChannels * (1 + kinds.size())
  weights::DType dtype = weights::DType::kBF16;  // the embedder's
  std::string base_model;          // metadata `base_model` (informational; empty when absent)
  // The tensors read here rather than by the LoRA stack (lora::LoraSpec::consumed).
  std::vector<std::string> consumed() const { return {kEmbedderWeight, kEmbedderBias}; }
  int kind_index(std::string_view kind) const;  // -1 when the file has no such kind
  int groups() const { return static_cast<int>(kinds.size()); }
};

// Reads and validates the header. Throws weights::FormatError / IoError for a broken container and ControlError for a
// file without a widened embedder (missing, a non-float dtype, a shape other than [kDim, 128 * (1 + G)] with G >= 1, a
// bias other than [kDim]) or with a control_channels entry that does not name G distinct non-empty kinds.
ControlFile open_control_file(const std::filesystem::path& path);
// Whether `path`'s header holds a widened video_embedder.weight (more than 128 input features); false when the file
// cannot be read as safetensors. Header only.
bool is_control_file(const std::filesystem::path& path);

// The replacement embedder in bf16: BF16 bit for bit, F32 / F16 rounded to nearest even (as a torch .to(bfloat16)).
struct EmbedderWeights {
  std::int64_t out_features = 0, in_features = 0;
  std::vector<std::uint16_t> w;     // [out_features, in_features], row-major
  std::vector<std::uint16_t> bias;  // [out_features]
};
// Throws weights::IoError on a read failure and ControlError for a non-finite value.
EmbedderWeights read_embedder(const ControlFile& file);

// The embedder on the device, in one allocation outside any arena (2.4 MB for G = 2), as the DiT's input extension.
class DeviceEmbedder {
 public:
  explicit DeviceEmbedder(const EmbedderWeights& w);
  static std::size_t device_bytes(std::int64_t out_features, std::int64_t in_features);
  std::size_t device_bytes() const { return buf_.size(); }
  Tensor weight() const;  // CUDA bf16 [out_features, in_features]
  Tensor bias() const;    // CUDA bf16 [out_features]
  std::int64_t in_features() const { return in_; }

 private:
  DeviceBuffer buf_;
  std::int64_t out_ = 0, in_ = 0;
};

// Whether a control with step window [start, end) is active at 0-based step `step` of a pass of `num_steps` steps:
// start <= step / num_steps < end (the guidance windows' convention, pipeline/sampler.h).
bool window_active(double start, double end, int step, int num_steps);

// The control latents of one sampling pass: per group its encoded latent already multiplied by its strength (empty:
// no input, always zero) and its window. apply() writes the step's input into the DiT's extra tensor.
class PassInputs {
 public:
  // groups x [128, h, w] fp32.
  PassInputs(int groups, std::int64_t h, std::int64_t w);
  // `latents` [128 * h * w], the encoded latent times the strength. Throws std::invalid_argument for a bad size or group.
  void set(int group, std::vector<float> latents, double start, double end);
  // extra: CUDA fp32 [groups * 128, h, w], contiguous. Writes the groups whose state differs from the last apply()
  // (all of them on the first call): the latent where active, zeros elsewhere. Asynchronous on `stream`; the host
  // latents are members, so the copies stay valid. Returns the number of groups written, which after the first call is
  // exactly the number of groups whose contents changed for this step (0: the DiT's input is the previous step's).
  int apply(const Tensor& extra, int step, int num_steps, cudaStream_t stream);
  // Writes every group again as the last apply() left it, without changing that state: `extra` was overwritten in
  // between (a component run's crop or per-call control). Nothing before the first apply().
  void rewrite(const Tensor& extra, cudaStream_t stream);
  int groups() const { return static_cast<int>(groups_.size()); }

 private:
  struct Group {
    std::vector<float> latents;
    double start = 0, end = 1;
    int written = -1;  // -1: never, 0: zeros, 1: the latent
  };
  void check_extra(const Tensor& extra, const char* what) const;
  // Group `group` of `extra` as `state` (1: its latent, 0: zeros).
  void write(const Tensor& extra, int group, int state, cudaStream_t stream) const;
  std::vector<Group> groups_;
  std::int64_t h_ = 0, w_ = 0;
};

}  // namespace mi::control
