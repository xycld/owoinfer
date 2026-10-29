// Runtime LoRA stack (P6 step 42): any number of adapter files with per-file strengths on the MOSS DiT, bf16 or INT8
// base, resident or streamed.
//
// Files are the training ecosystem's safetensors formats, read as file formats only:
//   - low rank: kohya `lora_unet_<module_with_underscores>.lora_{down,up}.weight` (+ `.alpha`) and its dotted form
//     with `diffusion_model.` / `model.diffusion_model.` prefixes; PEFT / diffusers / ai-toolkit `transformer.` /
//     `base_model.model.` / unprefixed `<module>.lora_{A,B}.weight`, plus diffusers' `.lora.{down,up}.weight` and
//     `.lora_linear_layer.{down,up}.weight` spellings;
//   - LyCORIS LoHa (`.hada_w{1,2}_{a,b}`, + `.alpha`) and LoKr (`.lokr_w{1,2}` or their `_a`/`_b` factors, + `.alpha`);
//   - full deltas `.diff` (weights) and `.diff_b` (biases).
// A fused DiT `layers.N.proj.qkv` target (the ComfyUI layout) is split by rows onto proj.q / proj.k / proj.v.
//
// Scaling follows each file format's own definition: Δ = strength · (alpha / rank) · B·A for low rank, alpha from the
// module's `.alpha` tensor, else (diffusers / PEFT files) from the `lora_adapter_metadata` header's lora_alpha, which
// is then noted in FileReport::notes, else alpha / rank = 1 (quant::lora_metadata_alpha, the rule the converter
// shares). ComfyUI reads only the alpha tensors and ignores the metadata alpha; in every file surveyed the two agree.
// LoHa: strength · (alpha / w1_b rows) · (w1a·w1b) ⊙ (w2a·w2b); LoKr: strength · (alpha / dim) · kron(w1, w2), dim the
// rank of the last factored w (1 when neither is factored); full: strength · diff. coef = f32(strength · alpha / rank).
//
// Placement (where each adapter is applied):
//   - low-rank adapters on the 6 GEMMs of a DiT block -> the bf16 side branch (side_branch.h): Δ = Σ coef_i B_i (A_i x)
//     added before the GEMM's epilogue op, so strengths change instantly and the base weights stay untouched
//     (INT8 included, no requantization);
//   - everything else (LoHa / LoKr / full deltas anywhere; any adapter on text_embedder, the modulation and time
//     linears, the embedders and the final layer) -> merged into the base weight at load (merge_bf16, or
//     merge_requantize_int8 for an INT8 block GEMM, which adds the requantization error: with
//     LoadOptions::int8_block_gemms each such adapter gets a FileReport warning). A strength change re-merges from the
//     pristine base weights kept by PristineWeights.
//
// Untrusted input: every name, dtype, shape and value is validated (weights::FormatError / LoraError); keys that are
// not understood or do not match a module of the model are reported per file (FileReport::unknown), never silently
// dropped, and a file in which nothing matches is an error. LoadOptions::strict turns every unknown key and warning
// into an error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "weights/safetensors.h"

namespace mi::lora {

struct LoraError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---- Target model ---------------------------------------------------------------------------------------------

// The 6 GEMMs of a DiT block (P:126-148), "layers.N.proj.<name>".
enum class BlockGemm : std::uint8_t { kQ, kK, kV, kO, kUpGate, kDown };
inline constexpr int kBlockGemms = 6;
const char* block_gemm_name(BlockGemm g);  // "q", "k", "v", "o", "ffn_up_gate", "ffn_down"

struct TargetModule {
  std::string name;           // canonical (diffusers) module name, e.g. "layers.3.proj.q", "text_embedder.proj"
  std::int64_t n = 0, k = 0;  // weight [n, k]
  bool has_bias = false;
  int block = -1;             // the DiT block of a block GEMM, else -1
  BlockGemm gemm = BlockGemm::kQ;  // when block >= 0
};

struct TargetModel {
  std::vector<TargetModule> modules;  // sorted by name
  int blocks = 0;
  std::int64_t dim = 0;               // model width (the fused qkv split: rows [0, dim), [dim, 2 dim), [2 dim, 3 dim))
  const TargetModule* find(std::string_view name) const;
};

// The MOSS-Image DiT (config.json: dim 3072, ffn_dim 8192, num_layers 42, num_heads 24, text_dim 4096, freq_dim 256,
// latent_channels 128; 2 text-refiner blocks): every linear of the transformer, as weights::moss_dit_linears lists
// them.
TargetModel moss_dit_targets();

// ---- Parsed adapters ------------------------------------------------------------------------------------------

enum class AdapterKind : std::uint8_t { kLowRank, kLoHa, kLoKr, kFull };
const char* kind_name(AdapterKind k);

struct TensorRef {  // one tensor of a LoRA file (header facts; bytes read on demand)
  int file = -1;
  std::string name;
  weights::DType dtype = weights::DType::kF32;
  std::vector<std::int64_t> shape;
  std::uint64_t offset = 0, nbytes = 0;
};

// One adapter on one target module. Factors are stored as in the file; a fused-qkv adapter becomes three adapters
// (one per target) that share the file's tensors and select rows [row0, row0 + n) of the stored delta.
struct Adapter {
  AdapterKind kind = AdapterKind::kLowRank;
  int lora = 0;               // index of its file in the stack (its strength)
  std::string module;         // target module
  std::string source;         // module name as written in the file
  std::int64_t n = 0, k = 0;  // target weight shape
  std::int64_t row0 = 0;      // first row of the target within the stored delta
  std::int64_t stored_rows = 0;  // rows of the stored delta (n, or 3n for a fused qkv)
  std::int64_t rank = 0;      // kLowRank: r; kLoHa: r1; kLoKr: the scale divisor (0 = none); kFull: 0
  double alpha_scale = 1.0;   // alpha / rank (1 without alpha), in double as ComfyUI's Python float
  bool side_branch = false;   // applied by the side branch (else merged at load)
  // Merged without a pristine copy of its module (a block module of a merged moss-edit-lora file: its strength or
  // placement changes reload the model instead of re-merging).
  bool keep_pristine = true;
  // kLowRank: {down [r, k], up [stored_rows, r]}
  // kLoHa:    {w1a [stored_rows, r1], w1b [r1, k], w2a [stored_rows, r2], w2b [r2, k]}
  // kLoKr:    w1 = {w1 [a, b]} or {w1a [a, r], w1b [r, b]}, then w2 = {w2 [c, d]} or {w2a [c, r'], w2b [r', d]},
  //           a * c = stored_rows, b * d = k (lokr_split says which are factored)
  // kFull:    {diff [stored_rows, k]}, or nothing for a bias-only delta
  std::vector<TensorRef> parts;
  bool lokr_split[2] = {false, false};
  std::optional<TensorRef> bias_diff;  // diff_b [stored_rows]
};

struct LoraSpec {
  std::filesystem::path path;
  float strength = 1.0f;
  // Tensors of the file that another component reads (a control adapter's replacement input embedder,
  // models/control): neither applied nor reported unknown, but noted in FileReport::notes.
  std::vector<std::string> consumed = {};
};

struct FileReport {
  std::string file;
  std::string format;    // its adapter_config format ("moss-edit-lora": a Moss-Image-Edit adapter), "" for none
  int adapters = 0;      // after the fused-qkv split
  int side_branch = 0;   // of them on the side branch
  int merged = 0;        // of them merged at load
  std::vector<std::string> unknown;   // "<tensor>: <reason>" for every key not applied
  std::vector<std::string> warnings;  // "<module>: <reason>" for adapters applied with a loss (INT8 requantization)
  std::vector<std::string> notes;     // how the file was read (e.g. the metadata alpha applied)
};

// Where low-rank adapters on the block GEMMs go (the request's lora_placement), a fixed rule per module, decided when
// the stack is parsed:
//   kDefault     every low-rank block adapter on the side branch, moss-edit-lora files (Moss-Image-Edit) included:
//                strength changes are instant and loads fast (a DiT file with the adapter baked in, moss-infer
//                quantize --lora, has no per-step cost)
//   kMerge       every low-rank block adapter merged (no per-step cost, a slower load; an edit adapter's strength or
//                placement change reloads the DiT); one on a module that cannot take a merge (INT8 without ConvRot,
//                INT4, NVFP4) is a LoraError naming the module and its format
//   kSideBranch  every low-rank block adapter on the side branch
// Other adapters (LoHa / LoKr / full deltas, non-block modules) are merged whatever the placement.
enum class Placement : std::uint8_t { kDefault, kMerge, kSideBranch };
const char* placement_name(Placement p);  // "default", "merge", "side_branch"
Placement parse_placement(std::string_view name);  // one of those; std::invalid_argument otherwise

// How the base stores a block linear, one value per quant::QuantFormat (merges need kBf16 or kInt8, INT8 ConvRot).
enum class BaseFormat : std::uint8_t { kBf16, kInt8, kInt8Plain, kInt4, kNvfp4 };

struct LoadOptions {
  bool strict = false;            // throw LoraError instead of reporting unknown keys or warnings
  bool int8_block_gemms = false;  // the base stores the block GEMMs as INT8: merged adapters there are requantized
  Placement placement = Placement::kDefault;
  // Each block linear's stored format (empty: every block linear INT8 with int8_block_gemms, else bf16).
  std::function<BaseFormat(std::string_view module)> base_format{};
};

class LoraStack {
 public:
  // Parses and validates every file's header (and alpha tensors). Throws weights::FormatError / IoError for a broken
  // container or an invalid adapter, LoraError for a file with no applicable key, a non-finite strength, or (strict)
  // any unknown key.
  LoraStack(std::span<const LoraSpec> specs, const TargetModel& model, const LoadOptions& options = {});
  ~LoraStack();
  LoraStack(LoraStack&&) noexcept;
  LoraStack& operator=(LoraStack&&) noexcept;

  int files() const { return static_cast<int>(specs_.size()); }
  const LoraSpec& spec(int lora) const { return specs_.at(static_cast<std::size_t>(lora)); }
  const std::vector<Adapter>& adapters() const { return adapters_; }
  const std::vector<FileReport>& reports() const { return reports_; }
  const TargetModel& model() const { return model_; }

  float strength(int lora) const { return specs_.at(static_cast<std::size_t>(lora)).strength; }
  // A merged adapter only follows a strength change when its module is merged again from the base weights.
  void set_strength(int lora, float strength);
  float coef(const Adapter& a) const;

  // Modules with adapters to merge at load (sorted, unique), and whether `module` has any.
  std::vector<std::string> merge_modules() const;
  bool has_merge(std::string_view module) const;
  // Whether a merged `module` keeps a pristine copy (false when an adapter on it has Adapter::keep_pristine false).
  bool keeps_pristine(std::string_view module) const;
  // Adapters on `module` in stack order (file order, then the file's own order).
  std::vector<const Adapter*> adapters_of(std::string_view module) const;

  // A factor's values (F32 / F16 / BF16 widened to fp32), rows [row0, row0 + rows) of its leading dim (rows < 0:
  // all). Throws weights::FormatError for a non-finite value. Thread-safe.
  std::vector<float> read(const TensorRef& t, std::int64_t row0 = 0, std::int64_t rows = -1) const;

 private:
  struct File;
  TargetModel model_;
  std::vector<LoraSpec> specs_;
  std::vector<std::unique_ptr<File>> files_;
  std::vector<Adapter> adapters_;
  std::vector<FileReport> reports_;
};

// f32(strength * alpha_scale) with the product in double, as ComfyUI's Python-float scalar rounded by the fp32
// tensor multiply (quant::lora_coef's value); infinity when it is outside the fp32 range.
float lora_coef(float strength, double alpha_scale);

// ---- Merges (host, fp32 math; lora_merge.cpp) ------------------------------------------------------------------
// Per adapter in stack order: w = w + coef * delta, each delta element a sum of products in double rounded to fp32
// (LoHa: f32(m1) * f32(m2); LoKr: f32(w1) * f32(w2)), one fp32 multiply and one fp32 add: the converter's (and
// ComfyUI's fp32) order.
// The helpers work in place on row blocks across threads: when one throws (a non-finite factor or result), the buffer
// is left partially merged, and the caller must discard it (reload it, or restore it from PristineWeights).

// Adds the merged adapters of `module` (all of them with include_side_branch, for references) to rows
// [row0, row0 + rows) of its fp32 weight w [rows, k]; bias (length rows, or null) gets the diff_b terms. Returns the
// number of adapters added.
int add_merged_rows(const LoraStack& stack, std::string_view module, std::int64_t row0, std::int64_t rows, float* w,
                    float* bias = nullptr, bool include_side_branch = false);

// A bf16 weight [n, k] (and bias [n] or null) in host memory, merged in place: w = bf16(f32(w) + ...). Returns the
// number of adapters merged (0: untouched). threads 0 = all cores.
int merge_bf16(const LoraStack& stack, std::string_view module, std::uint16_t* w, std::int64_t n, std::int64_t k,
               std::uint16_t* bias = nullptr, int threads = 0);

// An INT8 weight q [n, k] with per-row scale [n] (ConvRot group `group`, 0 = unrotated) in host memory: dequantized
// (quant::dequantize_rows), merged, and requantized round-to-nearest (quant::quantize_convrot_rows), in place. Returns
// the number of adapters merged (0: untouched). Throws LoraError when a merged row is not finite.
int merge_requantize_int8(const LoraStack& stack, std::string_view module, std::int8_t* q, float* scale,
                          std::int64_t n, std::int64_t k, int group, int threads = 0);

// Pristine base weights of the merged modules (LoraStack::merge_modules()), kept on the host so that a strength
// change can merge again from them: a merge is not invertible (bf16 rounding, INT8 requantization). Usage: keep_*()
// each module once from the base weights before its first merge, then remerge_*() into the weights the model uses,
// at load and after every strength change of a LoRA with merged adapters. The destination is any host buffer of the
// module's size: for a resident module the staging copy that is then uploaded, for a streamed block the pinned host
// copy the block streams from (weights streamed from the file would bypass the merge). With every strength 0, the
// destination receives the pristine bits.
class PristineWeights {
 public:
  void keep_bf16(const std::string& module, const std::uint16_t* w, std::int64_t n, std::int64_t k,
                 const std::uint16_t* bias = nullptr);
  void keep_int8(const std::string& module, const std::int8_t* q, const float* scale, std::int64_t n, std::int64_t k,
                 int group);
  bool has(std::string_view module) const;
  void drop(std::string_view module);  // forgets `module`'s copy (no-op when not kept)
  std::size_t host_bytes() const;
  std::size_t host_bytes(std::string_view module) const;  // `module`'s copy (0 when not kept)
  // dst = merge(pristine) for `stack`'s current strengths; returns the adapters merged. Throws LoraError for a module
  // not kept (or kept in the other format) or a size mismatch; on a merge error dst must be discarded (see above).
  int remerge_bf16(const LoraStack& stack, std::string_view module, std::uint16_t* dst, std::uint16_t* dst_bias = nullptr,
                   int threads = 0) const;
  int remerge_int8(const LoraStack& stack, std::string_view module, std::int8_t* q, float* scale, int threads = 0) const;

 private:
  struct Entry {
    std::int64_t n = 0, k = 0;
    int group = 0;
    bool int8 = false;
    std::vector<std::uint16_t> w, bias;  // bf16
    std::vector<std::int8_t> q;          // int8
    std::vector<float> scale;
  };
  const Entry& entry(std::string_view module, bool int8) const;
  std::vector<std::pair<std::string, Entry>> entries_;
};

}  // namespace mi::lora
