// `moss-infer quantize` (P6 step 34): writes a DiT as INT8 ConvRot safetensors in ComfyUI's comfy_quant layout.
//
// Input: the official bf16 diffusers transformer (directory or bundle root), or any single DiT file open_quant_model
// reads. bf16 GEMMs are quantized round-to-nearest (quant_codec.h, bit-exact to Comfy-Org's files); already
// quantized GEMMs are copied (INT8) or decoded exactly (W4A8 / W6A8 / W4A4 -> INT8 grid with their per-row scale),
// so importing a community file this way is lossless.
//
// Output, as Comfy-Org's moss_image_*_int8_convrot files: unprefixed official names; per quantized module
// `<m>.weight` I8 [N, K], `<m>.weight_scale` F32 [N, 1] and `<m>.comfy_quant` (U8 JSON
// {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": G}); `layers.N.comfy_attention.config`
// ({"attention": "comfy_kitchen_int8"}); every other tensor copied. Separate proj.q/k/v by default, fused
// `proj.qkv` (cat along rows) on request. Tensors are ordered like the safetensors library writes them (dtype
// descending, then name). `__metadata__` keeps the source file's keys except `_quantization_metadata` (and earlier
// mossinfer_* keys) and adds mossinfer_converter (version and the count of GEMMs per written spec),
// mossinfer_kept_bf16 (JSON list of the GEMMs left bf16), mossinfer_source_xxh64 and, with LoRAs, mossinfer_lora.
// No certificate reference is written: quality certificates come from a separate harness and live next to the file.
//
// Only the DiT is converted. The VAE is out of scope (small, kept bf16); the text encoder is imported as published.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "weights/quant/quant_model.h"

// mi::quant rather than mi::weights: the format is shared by the reader, the converter and the GEMM kernels.
namespace mi::quant {

inline constexpr int kConverterVersion = 1;

struct LoraInput {
  std::filesystem::path file;
  float strength = 1.0f;
};

struct ConvertOptions {
  int group = 256;               // ConvRot group of newly quantized GEMMs
  bool fuse_qkv = false;         // write layers.N.proj.qkv instead of proj.q/k/v
  // Also write layers.N.comfy_attention.config (a ComfyUI runtime preference, not part of the weight format) so the
  // output is byte-identical to Comfy-Org's files. Off by default; files load in ComfyUI either way.
  bool attention_config = false;
  std::vector<std::string> keep_bf16;  // glob patterns ('*', '?') over canonical GEMM names, e.g. "layers.*.proj.ffn_down"
  std::vector<LoraInput> loras;        // merged in fp32 into bf16 weights before quantizing, in order
  int threads = 0;                     // 0 = all cores
  // Extension point (sensitivity sweep): the spec of each bf16-stored GEMM. Default: bf16 when a keep_bf16 pattern
  // matches, else int8_convrot(group). Must return bf16 or int8_convrot.
  std::function<QuantSpec(const QuantLinear&)> policy;
  std::function<void(std::string_view)> log;  // progress lines (optional)
};

struct ConvertStats {
  int quantized = 0;  // bf16 GEMMs quantized here
  int copied = 0;     // INT8 GEMMs copied
  int decoded = 0;    // W4A8 / W6A8 / W4A4 / scalar-scale GEMMs decoded to INT8
  int kept_bf16 = 0;
  int lora_modules = 0;
  std::uint64_t source_xxh64 = 0;
  std::uint64_t bytes_written = 0;
  double seconds = 0;
};

// Simple glob: '*' any run, '?' any one character.
bool glob_match(std::string_view pattern, std::string_view text);

// Converts `input` into `output`: written to `<output>.partial`, flushed to disk, read back and renamed; on any
// failure the partial file is removed. Throws QuantError (bad options, an output that is an input or lies in the
// input directory, LoRA keys that match nothing, a LoRA or keep_bf16 aimed at an already quantized GEMM, a
// non-finite LoRA strength / alpha / coefficient, weights non-finite after merging, invalid stored scales) or the
// reader's errors.
ConvertStats convert(const std::filesystem::path& input, const std::filesystem::path& output, const ConvertOptions& options);

// LoRA delta merge of one row (exposed for tests): w[k] += f32(coef * f32(sum_r up[n_row, r] * down[r, k])) with the
// sum in double, coef = f32(strength * alpha / rank). up_row: [rank], down: [rank, k] row-major; acc: k doubles of
// scratch.
void lora_merge_row(float* w, std::int64_t k, const float* up_row, const float* down, std::int64_t rank, float coef, double* acc);
float lora_coef(float strength, std::optional<float> alpha, std::int64_t rank);

}  // namespace mi::quant
