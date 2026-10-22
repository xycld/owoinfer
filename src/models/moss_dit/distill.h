// Few-step distilled checkpoints (P6 step 41) that ship as adapter files for the stock DiT rather than as a full model:
// the lodestones data-space DMD repos (MOSS-Image-{Base,SFT}-8-steps-dmd) and their derivatives.
//
// A distill bundle is a directory holding any of
//   time_embedding.safetensors  replacements for time_embedding.{0,2}.{weight,bias} (the step embedder) and the
//                               biases of shared_block_modulation and video_head_modulation, into which the
//                               distillation folded its time vector, so the embedder only works with them. Published in
//                               fp32 with the instruction to run these three modules in fp32 (TimeEmbeddingFp32);
//   lora.safetensors            the block GEMMs' LoRA (a runtime LoRA, models/lora);
//   dataspace.safetensors       the data-space sampler's parameters (pipeline/samplers), run at CFG 1.
// The file names are the published ones. What a front end makes of them (the sampler, CFG 1, the LoRA's strength) is
// its choice; open_distill_bundle only finds them.
#pragma once

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace mi::moss_dit {

struct DistillError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct DistillBundle {
  std::filesystem::path dir;
  std::filesystem::path time_embedding;  // empty when absent
  std::filesystem::path lora;
  std::filesystem::path dataspace;
};

// The bundle files found in `dir`. Throws DistillError when `dir` is not a directory or holds none of them.
DistillBundle open_distill_bundle(const std::filesystem::path& dir);

// A replacement step embedder computed in fp32 (MossDit::set_time_embedding), the published values unrounded:
//   h = silu(W0 · emb + b0), time = W2 · h + b2 (emb the fp32 sinusoid of the timestep),
//   mod = W_mod · time + mod_bias, head = W_head · time + head_bias (W_mod / W_head the model's bf16 weights, widened),
// each rounded to bf16 once at the end, where the model's own bf16 path rounds every intermediate.
struct TimeEmbeddingFp32 {
  std::vector<float> w0;         // [3072, 256]
  std::vector<float> b0;         // [3072]
  std::vector<float> w2;         // [3072, 3072]
  std::vector<float> b2;         // [3072]
  std::vector<float> mod_bias;   // [6 * 3072]; empty: the model's shared_block_modulation.bias
  std::vector<float> head_bias;  // [2 * 3072]; empty: the model's video_head_modulation.bias
};

// Reads a time-embedding replacement file: time_embedding.0.weight [3072, 256], .0.bias [3072], .2.weight
// [3072, 3072], .2.bias [3072] and, optionally, shared_block_modulation.bias [18432] and video_head_modulation.bias
// [6144]; F32, BF16 or F16; names optionally prefixed "diffusion_model." or "model.diffusion_model.". Any other tensor,
// a missing or misshaped one, or a non-finite value: DistillError (also wrapping the reader's format / I/O errors).
TimeEmbeddingFp32 load_time_embedding(const std::filesystem::path& file);

// The fp32 step embedder's output for `timestep`: time [3072] (silu MLP over the fp32 sinusoid, sums in double).
std::vector<float> time_vector_fp32(const TimeEmbeddingFp32& te, float timestep);

}  // namespace mi::moss_dit
