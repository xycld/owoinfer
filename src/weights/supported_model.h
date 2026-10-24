// Identification of the models the engine supports: MOSS-Image Base and SFT (OpenMOSS-Team/MOSS-Image-Base, -SFT),
// and anything with their architecture (fine-tunes, quantized or converted files of them). The engine's kernels are
// built for that one architecture, so anything else is refused up front with a plain message instead of failing later
// on a tensor shape. Identification is by architecture (configs, tensor names and shapes), never by weight hashes, and
// reads only configs and safetensors headers.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "weights/model_index.h"
#include "weights/safetensors.h"

namespace mi::weights {

// Not a supported model. The message names the path, what was found and the supported models.
struct UnsupportedModelError : IndexError {
  using IndexError::IndexError;
};

// A bundle folder: the architecture fields of transformer/config.json, text_encoder/config.json and vae/config.json
// (when present) must be MOSS-Image's. Class names (model_index.json, `_class_name`, `model_type`) only describe what
// was found: a folder whose fields all match is accepted whatever its classes are called. Throws UnsupportedModelError,
// or IndexError for a missing folder, a missing config (unless model_index.json names another pipeline) or an
// unreadable or malformed config.
void check_supported_bundle(const std::filesystem::path& dir);

// A single-file DiT (dit_file), from its safetensors header: the MOSS-Image tensors that every supported format keeps
// in bf16 (embedders, time embedding, final layer, head norms) with MOSS-Image shapes, and blocks 0..41.
// ComfyUI's `model.diffusion_model.` / `diffusion_model.` prefixes are accepted. GGUF and pickle files, LoRAs, control
// adapters and distill bundle files are named as such, and so are block weights the importer cannot read (FP8,
// torchao). Throws UnsupportedModelError, or FormatError / IoError for an unreadable file (IoError "not a regular
// file" for a FIFO, a device or a directory, refused before any read).
void check_supported_dit_file(const std::filesystem::path& file);
// The same on a header already parsed (messages name file.path()), for callers that also read the header themselves:
// the file is a safetensors file by then, so only the GGUF / pickle container check of the path form is skipped.
void check_supported_dit_file(const SafetensorsFile& file);

// A single-file text encoder (text_encoder_file), from its safetensors header: the Qwen3.5 text model of MOSS-Image
// (vocabulary 248320, hidden size 4096, 32 layers, head size 256). Throws UnsupportedModelError, or FormatError /
// IoError for an unreadable file (as check_supported_dit_file).
void check_supported_text_encoder_file(const std::filesystem::path& file);
// The same on a header already parsed, as check_supported_dit_file(const SafetensorsFile&).
void check_supported_text_encoder_file(const SafetensorsFile& file);

// Whether a safetensors file holds LoRA factors (`.lora_A.` / `.lora_B.`, `.lora_down.` / `.lora_up.`, diffusers'
// `.lora.down.` / `.lora.up.` / `.lora_linear_layer.`, LyCORIS `.hada_w` / `.lokr_w`): a LoRA, or a control adapter (a
// LoRA with a widened video_embedder). From names only; what the check functions above refuse as "a LoRA" / "a control
// adapter". An `.alpha` tensor is no marker of its own (a LoRA's sits next to its module's factors; Snake activations'
// `.act.alpha` and `.alphas` belong to other models).
bool looks_like_lora(const SafetensorsFile& file);

// The linears of the MOSS-Image DiT (transformer/config.json: dim 3072, ffn_dim 8192, num_layers 42, text_dim 4096,
// freq_dim 256, latent_channels 128; 2 text-refiner blocks) under their canonical (diffusers) module names, sorted by
// name: the 6 GEMMs of every block, the text embedder, the time embedding, the modulations, the input embedder and the
// final layer. The one table of their shapes: lora::moss_dit_targets(), the quantized importer
// (quant::open_quant_model, which refuses a DiT linear of another shape before reading its bytes) and
// check_supported_dit_file build on it.
struct DitLinear {
  std::string name;           // e.g. "layers.3.proj.q", "text_embedder.proj"
  std::int64_t n = 0, k = 0;  // weight [n, k]
  bool has_bias = false;
  int block = -1;             // the block of a block GEMM ("layers.N.proj.*"), else -1
  int gemm = 0;               // block GEMMs: 0..5 for proj.q, k, v, o, ffn_up_gate, ffn_down (lora::BlockGemm's order)
};
const std::vector<DitLinear>& moss_dit_linears();
// The entry of a canonical module name, or nullptr. A fused `layers.N.proj.qkv` is not one: it is proj.q / k / v
// concatenated by rows ([3 * 3072, 3072]).
const DitLinear* find_moss_dit_linear(std::string_view module);

// The tensors of the MOSS-Image text encoder's Qwen3.5 text model (text_encoder/config.json text_config: vocab 248320,
// hidden 4096, intermediate 12288, 32 layers: three GatedDeltaNet `linear_attention` layers, then one `full_attention`
// layer, eight times. Full attention: 16 heads of 256 with an output gate (q_proj [2 * 16 * 256, 4096]), 4 KV heads.
// GatedDeltaNet: 16 key heads and 32 value heads of 128, conv kernel 4 (in_proj_qkv / conv1d: 2 * 2048 + 4096
// channels)), under the canonical names (`language_model.*`, as quant::open_quant_model keys them), sorted by name:
// every tensor the encoder reads (qwen35_text::TextEncoder's weight specs), its final norm and its embedding table.
// The one table of their shapes: the quantized importer refuses a text-encoder linear or tensor of another shape before
// reading any of its bytes. `linear`: a GEMM weight `<module>.weight` that may be stored quantized (the embedding
// table too, which Comfy-Org stores as an INT8 linear).
struct TeTensor {
  std::string name;  // e.g. "language_model.layers.3.self_attn.q_proj.weight", "language_model.norm.weight"
  std::vector<std::int64_t> shape;
  bool linear = false;
};
const std::vector<TeTensor>& qwen35_text_tensors();
// The entry of a canonical tensor name, or nullptr.
const TeTensor* find_qwen35_text_tensor(std::string_view name);

}  // namespace mi::weights
