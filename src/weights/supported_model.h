// Identification of the models the engine supports: MOSS-Image Base and SFT (OpenMOSS-Team/MOSS-Image-Base, -SFT),
// and anything with their architecture (fine-tunes, quantized or converted files of them). The engine's kernels are
// built for that one architecture, so anything else is refused up front with a plain message instead of failing later
// on a tensor shape. Identification is by architecture (configs, tensor names and shapes), never by weight hashes, and
// reads only configs and safetensors headers.
#pragma once

#include <filesystem>

#include "weights/model_index.h"

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
// torchao). Throws UnsupportedModelError, or FormatError / IoError for an unreadable file.
void check_supported_dit_file(const std::filesystem::path& file);

// A single-file text encoder (text_encoder_file), from its safetensors header: the Qwen3.5 text model of MOSS-Image
// (vocabulary 248320, hidden size 4096, 32 layers, head size 256). Throws UnsupportedModelError, or FormatError /
// IoError for an unreadable file.
void check_supported_text_encoder_file(const std::filesystem::path& file);

}  // namespace mi::weights
