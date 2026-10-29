// Quantized-weight model (P6 step 33): which GEMM weights of a DiT or text-encoder checkpoint are bf16, INT8
// ConvRot or plain INT8, where their bytes are, and how they map onto the engine's canonical (diffusers) names.
//
// Inputs:
//   - a diffusers component directory (transformer/ or text_encoder/, sharded or not) or a bundle root;
//   - a single ComfyUI-layout safetensors file: unprefixed official DiT names (optionally "diffusion_model." or
//     "model.diffusion_model."), or the Qwen3.5 TE layout ("model.language_model.*", plus visual / mtp / lm_head
//     tensors, which are excluded).
// Per-module quantization configs come from `<module>.comfy_quant` U8 tensors (JSON) or the legacy
// `_quantization_metadata` header key ({"layers": {module: config}}), as ComfyUI reads them. Supported formats:
//   int8_tensorwise (+ convrot / convrot_groupsize)  -> kInt8ConvRot(G) or kInt8Plain, raw bytes
//   asym_w4a8_int8 (Lloyd-Max codebook or uniform), w6a8_int8, convrot_w4a4  -> decoded exactly to kInt8ConvRot
//   convrot_w4a4 (INT4 activations), with IndexOptions::native_int4            -> kInt4ConvRot, raw packed bytes
//   nvfp4 (E2M1 [N, K/2] hi_first, swizzled UE4M3 block scales, F32 weight_scale_2) -> kNvfp4, raw bytes as stored
//     (nvfp4_codec.h). The NVIDIA ModelOpt convention (the same tensor names without a comfy_quant config; low-first
//     nibbles and linear scales by default, indistinguishable by shape) is refused, as are pre_quant_scale, config
//     fields of other formats and full_precision_matrix_mult (not honoured); a static input_scale is kept on the
//     QuantLinear but ignored (the engine scales activations per token).
// A fused DiT `layers.N.proj.qkv` ([3 * dim, dim] = cat([q, k, v], 0)) is split into proj.q / proj.k / proj.v by
// row slices of every per-row tensor; the slices are plain byte ranges of the file (nvfp4: whole 128-row scale tiles,
// so each third must be a multiple of 128 rows; the three parts share the tensor scale).
//
// For the engine a linear is what a GEMM consumes (QuantWeights): an INT8 [N, K] row-major weight with an f32
// per-row scale [N] (y = (x_q . w_q^T) * s_x * s_w), or a bf16 [N, K] weight. Raw linears are uploaded or
// streamed per block exactly like the bf16 groups: to_model_index() lists them as `<module>.weight` and
// `<module>.weight_scale` byte ranges, so weights::upload_group / load_group_to_host / BlockWeights move them
// unchanged. Encoded linears (W4A8, W6A8, W4A4, scalar-scale INT8) are decoded on the CPU (decode_rows) or
// converted once to an INT8 ConvRot file with `moss-infer quantize` (lossless).
//
// Comfy-Org's text encoder, as this reader returns it (the engine's encoder does not consume it yet):
//   - `language_model.embed_tokens` is a quantized linear (I8 [vocab, 4096] + F32 scale, ConvRot256-rotated), not a
//     bf16 table. A lookup must gather rows and un-rotate them: gather_embedding_rows_bf16() does that, bit-exact to
//     comfy-kitchen's whole-table dequantization.
//   - `linear_attn.A_log` and `linear_attn.norm.weight` are F32 where diffusers stores bf16; read_tensor_bf16()
//     rounds them to exactly the diffusers bf16 values.
//   - in_proj_a / in_proj_b stay bf16 linears; conv1d, dt_bias and the other norms are bf16 tensors.
// The VAE is out of scope: it is small and stays bf16, so no VAE file is read here.
//
// Everything here is header-driven and validated as untrusted input (shapes, dtypes, sizes, config values); stored
// scales are checked (finite, > 0) whenever their bytes are read. A DiT's linears must be the MOSS-Image DiT's, and a
// text encoder's linears and tensors the MOSS-Image text encoder's, with their shapes (weights::moss_dit_linears,
// weights::qwen35_text_tensors), before any of their bytes are read, so the stored shapes never size an allocation
// beyond the model's.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/tensor.h"
#include "weights/gguf.h"
#include "weights/model_index.h"
#include "weights/safetensors.h"
#include "weights/upload.h"

// mi::quant rather than mi::weights: the format is shared by the reader, the converter and the GEMM kernels.
namespace mi::quant {

struct QuantError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// kInt4ConvRot: the convrot_w4a4 weights kept packed (signed nibbles [N, K/2], even column in the low nibble, f32
// per-row scale) for the INT4 GEMMs (kernels/quant4); only to_model_index(model, {.native_int4 = true}) gives it.
// kNvfp4: E2M1 [N, K/2] bytes (hi_first), UE4M3 block scales per 16 along K in the cuBLAS/CUTLASS swizzle and an F32
// tensor scale (nvfp4_codec.h), for the sm_120a NVFP4 GEMMs (kernels/nvfp4).
enum class QuantFormat : std::uint8_t { kBF16, kInt8ConvRot, kInt8Plain, kInt4ConvRot, kNvfp4 };
const char* format_name(QuantFormat f);

// How one GEMM's weight is quantized.
struct QuantSpec {
  QuantFormat format = QuantFormat::kBF16;
  int group = 0;  // kInt8ConvRot / kInt4ConvRot: Hadamard size along K (power of 4); 0 otherwise

  static QuantSpec bf16() { return {}; }
  static QuantSpec int8_convrot(int g) { return {QuantFormat::kInt8ConvRot, g}; }
  static QuantSpec int8_plain() { return {QuantFormat::kInt8Plain, 0}; }
  static QuantSpec int4_convrot(int g) { return {QuantFormat::kInt4ConvRot, g}; }
  static QuantSpec nvfp4() { return {QuantFormat::kNvfp4, 0}; }
  bool quantized() const { return format != QuantFormat::kBF16; }
  bool operator==(const QuantSpec&) const = default;
};
std::string describe(const QuantSpec& s);  // "bf16", "int8_convrot256", "int8_plain", "int4_convrot256", "nvfp4"

// How the stored bytes become the spec's weight (+ scale).
enum class Encoding : std::uint8_t {
  kRaw,          // bf16 [N, K]; or I8 [N, K] + F32 per-row scale ([N] or [N, 1]): use the bytes as they are
  kScalarScale,  // I8 [N, K] + one F32 scale for the whole tensor
  kW4A8,         // asym_w4a8_int8: 4-bit codes [N, K/2], e4m3 group scales, F32 s_channel, optional codebook
  kW6A8,         // w6a8_int8: 6-bit codes [N, 3K/4], e4m3 group scales, F32 s_channel
  kW4A4,         // convrot_w4a4: signed nibbles [N, K/2], F32 per-row scale
  kNvfp4Raw,     // nvfp4: U8 [N, K/2] + swizzled F8_E4M3 block scales + F32 scalar tensor scale, used as stored
  kGguf,         // a GGUF tensor of the ggml type QuantModel::gguf_types names: dequantized to fp32 (weights/gguf.h)
                 // and quantized (or rounded to bf16) by the converter; the spec is bf16, the stored precision unknown
};
const char* encoding_name(Encoding e);

enum class ModelKind : std::uint8_t { kDiT, kTextEncoder };
const char* kind_name(ModelKind k);

// One GEMM's weight as stored. Tensor names are the source file's; `shape`/`offset`/`nbytes` already describe the
// row slice when the linear was carved from a fused qkv.
struct QuantLinear {
  std::string name;  // canonical module name, e.g. "layers.3.proj.q", "language_model.layers.0.mlp.up_proj"
  QuantSpec spec;
  Encoding encoding = Encoding::kRaw;
  std::int64_t n = 0, k = 0;
  weights::IndexedTensor weight;
  std::optional<weights::IndexedTensor> scale;      // kRaw int8, kScalarScale, kW4A4; kNvfp4Raw: the block scales,
                                                    // F8_E4M3 or U8 [round_up(N,128), round_up(K/16,4)] (swizzled)
  std::optional<weights::IndexedTensor> scale_2;    // kNvfp4Raw: F32 scalar tensor scale (weight_scale_2)
  std::optional<weights::IndexedTensor> input_scale;  // kNvfp4Raw: a static activation scale; ignored (per-token)
  std::optional<weights::IndexedTensor> s_rel;      // kW4A8 / kW6A8: F8_E4M3 [N, K / quant_group]
  std::optional<weights::IndexedTensor> s_channel;  // kW4A8 / kW6A8: F32 [N]
  std::optional<weights::IndexedTensor> codebook;   // kW4A8: F32 [16]; absent = uniform codes (code - 8)
  std::optional<weights::IndexedTensor> bias;       // [N], any float dtype as stored
  int quant_group = 0;          // kW4A8 / kW6A8 group size along K; kW4A4: the config's quant_group_size (0: absent)
  bool fused_slice = false;     // rows of a fused proj.qkv
  bool full_precision_mm = false;  // config asked ComfyUI to run this layer in full precision (informational)
  bool int8_activations = false;   // kW4A4 with linear_dtype "int8": INT8 activations (W4A8 compute), never native INT4
  std::string source_module;    // module name in the file (e.g. "layers.3.proj.qkv")

  // Encodings that must be decoded to the INT8 grid before an engine GEMM can use them (nvfp4 is consumed natively).
  bool needs_decode() const { return encoding != Encoding::kRaw && encoding != Encoding::kNvfp4Raw; }
};

struct QuantModel {
  ModelKind kind = ModelKind::kDiT;
  std::filesystem::path base_dir;            // IndexedTensor::file indexes `files`, relative to base_dir
  std::vector<weights::IndexedFile> files;
  std::vector<QuantLinear> linears;          // sorted by name
  std::vector<weights::IndexedTensor> tensors;  // every other kept tensor under its canonical name, sorted by name
  std::vector<weights::ExcludedSummary> excluded;
  std::map<std::string, std::string> metadata;  // __metadata__ of the first file
  // A GGUF file (open_gguf_dit): the ggml type of every stored tensor by IndexedTensor::name (a linear's weight, by its
  // source name, and every kept tensor), whose dtype is then the dtype the converter writes (BF16; F32 for
  // rope.bands) and whose nbytes the GGUF bytes. Empty for safetensors inputs.
  std::map<std::string, std::uint32_t, std::less<>> gguf_types;

  const QuantLinear* find_linear(std::string_view name) const;
  const weights::IndexedTensor* find_tensor(std::string_view name) const;
  std::filesystem::path file_path(std::uint32_t id) const { return base_dir / util::path_from_utf8(files.at(id).path); }
};

// Opens a safetensors file, a GGUF DiT (open_gguf_dit), a diffusers component directory or a bundle root (then the
// component of `kind`).
// Without `kind` a single file's kind is detected from its tensor names (bundle roots default to the DiT).
// Text-encoder names are routed with weights::classify_tensor; a text-only Qwen3.5 checkpoint (`model.layers.*`) is
// refused. A quantization config on an excluded module (visual, mtp, lm_head) is ignored, even when malformed.
// A DiT linear (a quantized module or a bf16 block GEMM) that is not a MOSS-Image DiT linear with its [n, k] (a fused
// proj.qkv: [3 * 3072, 3072]) is refused, naming the tensor, its shape and the expected one; so is a text-encoder
// linear or kept tensor that is not one of the MOSS-Image text encoder's with its shape. Scale tensors are checked
// against [n, k] per layout; no tensor data but the comfy_quant configs is read here.
// Throws QuantError / weights::FormatError / weights::IoError.
QuantModel open_quant_model(const std::filesystem::path& path, std::optional<ModelKind> kind = std::nullopt);
// The same for a single safetensors file whose header the caller already parsed (not read again; base_dir is its
// directory). Configs and tensors are read from file.path().
QuantModel open_quant_model(const weights::SafetensorsFile& file, std::optional<ModelKind> kind = std::nullopt);

// A MOSS-Image DiT in a GGUF file, for the converter (convert(): INT8 ConvRot or bf16 output; nothing reads it
// directly): every tensor must be a MOSS-Image DiT tensor with its shape (weights::moss_dit_linears, the norms, an
// optional rope.bands), under the official names, optionally prefixed `model.diffusion_model.` / `diffusion_model.`,
// with q/k/v fused (`layers.N.proj.qkv`, split by row slices) or separate, and of a type weights::gguf_dequantizable
// reads. The block GEMMs become kGguf linears, everything else kept tensors (written BF16; rope.bands F32). Names,
// shapes and types are checked from the header before any tensor byte is read. Throws QuantError naming the file and
// the first offending tensors.
QuantModel open_gguf_dit(const weights::GgufFile& file);

// The canonical DiT GEMM modules (6 per block: proj.q/k/v/o/ffn_up_gate/ffn_down) and TE GEMM modules
// (language_model.layers.N.*), in the order a converter visits them.
bool is_gemm_module(ModelKind kind, std::string_view module);

// ---- Engine view -------------------------------------------------------------------------------------------

// What a GEMM consumes, as raw device or host pointers.
struct QuantWeights {
  QuantSpec spec;
  std::int64_t n = 0, k = 0;
  const void* weight = nullptr;  // bf16 or int8 [n, k] row-major; kInt4ConvRot / kNvfp4: packed [n, k/2] bytes
  const float* scale = nullptr;  // [n] per-row scale; null for bf16 and kNvfp4
  const std::uint8_t* block_scale = nullptr;  // kNvfp4: swizzled UE4M3 [round_up(n,128) * round_up(k/16,4)] bytes
  float tensor_scale = 0.0f;                  // kNvfp4: the F32 weight_scale_2 (a host value, the GEMM's alpha)
  const void* bias = nullptr;    // [n] in bias_dtype, or null
  DType bias_dtype = DType::kBF16;
  Device device = Device::kCPU;
};

// The model as a weights::ModelIndex with the engine's group names (weights::classify_tensor, so they equal
// build_index's: DiT "globals", "text_embedder", "block.N"; TE "embed_tokens", "layers.N" with layer_type,
// "final_norm"), plus every linear's spec. A raw linear contributes `<module>.weight` (bf16 or I8 [N, K]) and, when
// quantized, `<module>.weight_scale` (F32 [N]) plus `<module>.bias`. An nvfp4 linear contributes `<module>.weight`
// (U8 [N, K/2]) and `<module>.weight_scale` (U8, the swizzled block scales, [round_up(N,128), round_up(K/16,4)]); its
// tensor scale is read here (4 bytes, shared by the parts of a fused qkv) and kept in tensor_scales, not uploaded.
struct QuantIndex {
  weights::ModelIndex index;
  std::map<std::string, QuantSpec, std::less<>> specs;  // canonical module name -> spec, bf16 linears included
  std::map<std::string, float, std::less<>> tensor_scales;  // nvfp4 linears: module -> weight_scale_2

  const QuantSpec* find_spec(std::string_view module) const;
};

// native_int4: convrot_w4a4 linears (INT4 activations, quant_group_size absent or 64: the only value comfy-kitchen
// runs) are listed as stored, kInt4ConvRot with their packed `<module>.weight` (I8 [N, K/2]) and `.weight_scale`,
// instead of needing decoding.
struct IndexOptions {
  bool native_int4 = false;
};

// Reads each quantized linear's scales to validate them (a few MB in all for INT8; nvfp4 block scales are 1/32 of the
// weight bytes, ~130 MB for a half-NVFP4 DiT, checked non-negative, non-NaN and zero-padded). Throws QuantError if a
// linear needs decoding (convert it first), a stored scale is invalid or a tensor has no group.
QuantIndex to_model_index(const QuantModel& model, const IndexOptions& options = {});

// The linear `module` out of a group uploaded from `index`: `<module>.weight` (+ `.weight_scale`, `.bias`); for
// kInt4ConvRot the weight is the packed I8 [N, K/2] tensor and the view's k is 2 * K/2; for kNvfp4 the weight is
// U8 [N, K/2] (k = 2 * K/2), `.weight_scale` the swizzled block scales and tensor_scale comes from index.tensor_scales;
// the NVFP4 GEMMs also need K % 32 == 0 and 16-byte aligned weight and block scales (checked).
// `expected` is the spec the caller's GEMM implements; it must equal the file's (index.specs), so a bf16 GEMM can
// never consume INT8 bytes or a ConvRot256 kernel a ConvRot64 weight. Checks dtypes, shapes, contiguity, devices
// and element alignment (host groups keep each tensor's file offset modulo 4096; files written by the safetensors
// library, Comfy-Org's and ours align every tensor); throws QuantError.
QuantWeights view(const QuantIndex& index, const weights::TensorMap& m, std::string_view module, const QuantSpec& expected);

// ---- CPU decode --------------------------------------------------------------------------------------------

// Rows [row0, row0 + rows) of a quantized linear as the INT8 grid [rows, k] and per-row scales [rows], reading only
// those rows from disk (any encoding but nvfp4, which has no INT8 grid). Throws QuantError for bf16 and nvfp4 linears,
// bad ranges or invalid stored values (NaN or negative group scales, non-finite codebook entries, per-row scales that
// are not finite and > 0).
void decode_rows(const QuantModel& model, const QuantLinear& l, std::int64_t row0, std::int64_t rows, std::int8_t* q,
                 float* scale);

// The whole linear decoded (decode_rows over row chunks on `threads` threads, 0 = all cores); q is [n, k].
void decode_linear(const QuantModel& model, const QuantLinear& l, std::int8_t* q, float* scale, int threads = 0);

// Rows of any linear in the original basis as fp32 (bf16 widened; quantized: decode + un-rotate; nvfp4:
// dequantize_nvfp4_rows, comfy-kitchen's fp32 dequantization; GGUF: weights::dequantize_gguf_rows).
void dequantize_linear_rows(const QuantModel& model, const QuantLinear& l, std::int64_t row0, std::int64_t rows,
                            float* out);

// Rows of an nvfp4 linear dequantized on the CPU, into fp32 (`out`, comfy-kitchen's dequantize_nvfp4 to fp32) and/or
// bf16 bits (`out_bf16`, its bf16 order: what ComfyUI computes for a bf16 model); either may be null. Reads only the
// rows and their 128-row scale tiles; validates the scales read. Throws QuantError.
void dequantize_nvfp4_linear_rows(const QuantModel& model, const QuantLinear& l, std::int64_t row0, std::int64_t rows,
                                  float* out, std::uint16_t* out_bf16);

// An embedding lookup on a linear-shaped table (Comfy-Org's TE embed_tokens): rows `ids` as bf16 bits, out is
// [ids.size(), k]. Quantized rows are decoded and un-rotated in fp32 (dequantize_rows' order), then rounded to bf16:
// bit-exact to comfy-kitchen's whole-table dequantization in the bf16 compute dtype, a fixed function of each row.
// ComfyUI's lookup (dequantize_int8_embedding) un-rotates only the gathered rows, in a summation order that depends
// on the number of tokens, so it differs where the fp32 sum cancels to near zero. bf16 rows are copied. Throws
// QuantError for an id outside [0, n).
void gather_embedding_rows_bf16(const QuantModel& model, const QuantLinear& table, std::span<const std::int64_t> ids,
                                std::uint16_t* out);

// A BF16 or F32 tensor as bf16 bits (F32 rounded to nearest even). Comfy-Org's TE stores A_log and
// linear_attn.norm.weight in F32; rounded, they equal the diffusers bf16 tensors bit for bit. Throws QuantError
// for another dtype or a value that is (or rounds to) a NaN or infinity.
std::vector<std::uint16_t> read_tensor_bf16(const QuantModel& model, const weights::IndexedTensor& t);

// Raw bytes of a tensor (whole or a slice as described by it).
std::vector<std::byte> read_tensor(const QuantModel& model, const weights::IndexedTensor& t);

}  // namespace mi::quant
