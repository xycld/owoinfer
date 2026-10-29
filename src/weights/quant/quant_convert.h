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
//
// NVFP4 output (ConvertOptions::nvfp4, `--format nvfp4`; NVFP4 plan §4.4 and "N3 converter: reproducible output"): the
// block GEMM units (fused proj.qkv, proj.o, proj.ffn_up_gate, proj.ffn_down) become comfy_quant "nvfp4" with a recipe
// (nvfp4_recipes.h), except the keep list's units, which become INT8 ConvRot as above, and --keep-bf16's. Per nvfp4
// module, as BennyDaBall's MOSS-Image-SFT-NVFP4: `<m>.weight` U8 [N, K/2] (hi_first), `<m>.weight_scale` F8_E4M3
// [round_up(N,128), round_up(K/16,4)] (swizzled), `<m>.weight_scale_2` F32 [] (amax / 2688 of the whole module) and
// `<m>.comfy_quant` {"format": "nvfp4"}; q/k/v are always written fused (one GEMM, one tensor scale). The inputs must
// be bf16 (or LoRA-merged bf16; LoRAs only with rtn / mse). The metadata adds mossinfer_nvfp4 (converter version,
// recipe, keep list) and, for gptq recipes, mossinfer_nvfp4_calibration (the calibration set),
// mossinfer_nvfp4_capture_class (the GPU class, text encoder and engine numerics the Hessians were captured with) and
// mossinfer_nvfp4_hessian_xxh64 (XXH64 over the units in (block, unit) order of each unit's XXH64 of its module name
// and upper triangle, row by row: a digest of exactly the Hessians used). gptq modules are written as their Hessians
// arrive; the header keeps room for the capture strings as JSON whitespace. Nothing machine- or time-specific is
// written, so equal inputs (and, for gptq, an equal capture) give equal bytes.
//
// Only the DiT is converted. The VAE is out of scope (small, kept bf16); the text encoder is imported as published.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "weights/quant/nvfp4_recipes.h"
#include "weights/quant/quant_model.h"

// mi::quant rather than mi::weights: the format is shared by the reader, the converter and the GEMM kernels.
namespace mi::quant {

inline constexpr int kConverterVersion = 1;
// Bumped whenever an nvfp4 output byte would change for the same inputs (recipes, layout, metadata).
inline constexpr int kNvfp4ConverterVersion = 1;

// A block GEMM unit whose calibration Hessian a gptq recipe needs.
enum class GemmUnit : std::uint8_t { kQkv, kO, kUpGate, kDown };
struct HessianUnit {
  std::string module;  // as written: layers.N.proj.qkv | .o | .ffn_up_gate | .ffn_down
  int block = 0;
  GemmUnit unit = GemmUnit::kQkv;
  std::int64_t k = 0;  // input features: the Hessian is [k, k]
};
// What a capture reports for the metadata: two JSON objects (compact, deterministic), each at most
// kMaxCaptureStringBytes.
struct HessianCapture {
  std::string calibration;    // the calibration set
  std::string capture_class;  // the GPU class, text encoder and engine numerics of the capture
};
inline constexpr std::size_t kMaxCaptureStringBytes = 1024;
// sink(i, h): h is unit i's X^T X, fp32 [k, k] row-major (the upper triangle is read), valid during the call.
using HessianSink = std::function<void(std::size_t unit, const float* h)>;
// Must call sink exactly once per unit, in any order (units in ascending order give the least memory), then return.
using HessianSource = std::function<HessianCapture(const std::vector<HessianUnit>& units, const HessianSink& sink)>;

struct Nvfp4Options {
  Nvfp4Recipe recipe = Nvfp4Recipe::kGptqMse;
  std::string keep_name = "g0.9";  // written to the metadata: a preset's name, "none", or "custom"
  std::vector<std::string> keep;   // units kept INT8 ConvRot, as module names (parse_nvfp4_keep)
  // gptq recipes: the capture, or a Hessian file (write_hessian_file) captured from this same DiT (checked against the
  // file's DiT digest before any GPTQ work), exactly one of the two.
  HessianSource hessians;
  std::filesystem::path hessian_file;
};

// A --keep value: "none", a preset of the NVFP4 study (E30: "g0.9", 42 units, and "g0.95", 21 units, greedy by
// teacher-forced velocity error), or comma-separated units, each "b<N>.<qkv|o|up|down>" (the study's names) or a
// module name ("layers.<N>.proj.<qkv|o|ffn_up_gate|ffn_down>"). Returns sorted unique module names and sets `name` to
// the preset or "custom". Throws QuantError for an unknown preset or unit.
std::vector<std::string> parse_nvfp4_keep(std::string_view spec, std::string* name = nullptr);
// The block GEMM unit of a module name (fused qkv, o, ffn_up_gate, ffn_down; q/k/v map to qkv), or nullopt.
std::optional<HessianUnit> gemm_unit_of(std::string_view module);

// Hessian files: a capture saved to disk and solved later (the capture needs the GPU, the GPTQ solve only the CPU).
// The units convert() asks its HessianSource for, given `input` and `options` (options.nvfp4 set), in (block, unit)
// order. Plans the modules as convert() does, so it throws for the same misuse (unknown --keep units, a --keep-bf16
// that splits q/k/v, stored-quantized GEMMs).
struct ConvertOptions;
std::vector<HessianUnit> nvfp4_hessian_units(const std::filesystem::path& input, const ConvertOptions& options);
// Runs options.nvfp4->hessians over nvfp4_hessian_units(input, options) and writes each Hessian to `path`, a
// safetensors file: per unit `<module>` F32 [k * (k + 1) / 2], the upper triangle row by row, and in the metadata the
// capture's strings (mossinfer_nvfp4_calibration, mossinfer_nvfp4_capture_class) and the captured DiT's identity
// (mossinfer_nvfp4_dit_xxh64: convert()'s source digest of the input). `path` is checked as convert() checks its
// output; written to <path>.partial, read back and renamed when complete. A conversion with
// Nvfp4Options::hessian_file = path then writes the same bytes as one that captured directly.
struct HessianFileInfo {
  HessianCapture capture;
  std::size_t units = 0;
  std::uint64_t dit_xxh64 = 0;
};
HessianFileInfo write_hessian_file(const std::filesystem::path& input, const std::filesystem::path& path, const ConvertOptions& options);

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
  // matches, else int8_convrot(group). Must return bf16 or int8_convrot. Not with nvfp4.
  std::function<QuantSpec(const QuantLinear&)> policy;
  std::optional<Nvfp4Options> nvfp4;          // NVFP4 output (see the file comment); implies fuse_qkv
  std::function<void(std::string_view)> log;  // progress lines (optional)
};

struct ConvertStats {
  int quantized = 0;  // bf16 GEMMs quantized here (q/k/v count as three), of which:
  int nvfp4 = 0;      //   to nvfp4
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

// The LoRA alpha rule, shared with the runtime stack (models/lora): each file format's own definition.
// - A module's `.alpha` tensor wins.
// - Without one, diffusers / PEFT files declare it in the `lora_adapter_metadata` header (a PEFT LoraConfig as JSON):
//   its lora_alpha applies (PEFT's scaling = lora_alpha / r). lora_metadata_alpha returns it, or nullopt when the
//   header has none. Throws QuantError for malformed metadata or a non-empty alpha_pattern (per-module alphas are
//   refused rather than half-applied).
// - Without either, alpha / rank = 1.
// (ComfyUI reads only the alpha tensors and ignores the metadata; for every file surveyed the two agree.)
std::optional<double> lora_metadata_alpha(const std::map<std::string, std::string>& metadata);
// alpha / rank in double (1 without alpha), as the Python float of the reference implementations.
double lora_alpha_scale(std::optional<double> alpha, std::int64_t rank);
// f32(strength * lora_alpha_scale(alpha, rank)), the product in double; infinity outside the fp32 range.
float lora_coef(float strength, std::optional<double> alpha, std::int64_t rank);

}  // namespace mi::quant
