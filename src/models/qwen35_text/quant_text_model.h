// A quantized Qwen3.5 text-encoder file opened for TextEncoder: a single safetensors file in the Qwen3_5Model layout
// (Comfy-Org's qwen3.5_9b_int8_convrot / qwen3.5_9b_w4a8_gptq, or our converter's output), or any directory
// mi::quant::open_quant_model accepts for the text encoder.
//
// What the encoder consumes:
//   - Decoder GEMMs (GDN in_proj_qkv / in_proj_z / out_proj, attention q/k/v/o, MLP gate/up/down), each either
//       INT8 ConvRot256 per-row (raw I8 [N, K] + F32 scale [N]): used as stored;
//       grouped W4A8 / W6A8 codes (codebook, e4m3 g16 scales, F32 s_channel [N]): streamed packed and decoded to the
//       same INT8 grid on the GPU right before the GEMM (kernels/quant/w4a8_decode.h), bit-exact to the CPU decoder;
//       bf16 [N, K] (in_proj_a / in_proj_b always; any other GEMM a file keeps in bf16).
//   - conv1d, dt_bias and the norms as bf16 tensors. A_log and linear_attn.norm.weight are F32 in Comfy-Org's files;
//     they are read once here and rounded to bf16 (exactly the diffusers values), and the encoder uploads them with
//     its scratch, so the streamed layer groups hold no F32 tensor.
//   - embed_tokens as a quantized linear: rows are gathered on the host and un-rotated in fp32
//     (mi::quant::gather_embedding_rows_bf16), bit-exact to comfy-kitchen's whole-table dequantization in bf16.
// index() lists the decoder layers ("layers.N", with layer_type) and "final_norm" as weights::WeightGroup byte ranges
// of the file, so the encoder streams, host-pins and uploads them exactly like the bf16 bundle's groups. A linear
// `m` contributes `m.weight` and, by encoding, `m.weight_scale` (raw INT8) or `m.weight_s_rel`, `m.weight_s_channel`
// and `m.weight_codebook` (grouped).
//
// Validated at open (untrusted input): shapes and dtypes of every tensor the encoder reads, per-row scales and
// s_channel finite and > 0, codebooks finite. Group scales (1/16 byte per weight) are checked by the GPU decoder as
// they are used (encode() throws on a NaN or negative one).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "weights/model_index.h"
#include "weights/quant/quant_model.h"

namespace mi::qwen35_text {

// The arithmetic of the decoder GEMMs: what the encoder's output bits depend on besides the weights.
enum class Precision : std::uint8_t {
  kBF16,  // bf16 weights and activations (cuBLASLt)
  kInt8,  // INT8 ConvRot256 weights, online rotated per-token INT8 activations, int32 accumulation
  kW4A8,  // grouped 4/6-bit weights decoded to the INT8 grid, then as kInt8
};
const char* precision_name(Precision p);

// How one decoder GEMM's weight is stored in the streamed group.
struct LinearInfo {
  enum class Kind : std::uint8_t { kBF16, kInt8, kGrouped } kind = Kind::kBF16;
  std::int64_t n = 0, k = 0;
  int bits = 0;           // kGrouped: 4 or 6
  int group = 0;          // kGrouped: scale group along K
  bool codebook = false;  // kGrouped: 4-bit codebook levels (else uniform)
};

class QuantTextModel {
 public:
  // Throws mi::quant::QuantError (or a weights:: error) for a file the encoder cannot run.
  explicit QuantTextModel(const std::filesystem::path& path);

  Precision precision() const { return precision_; }
  const weights::ModelIndex& index() const { return index_; }
  const quant::QuantModel& model() const { return model_; }
  const quant::QuantLinear& embed_tokens() const { return *embed_; }
  // Decoder GEMM `module` (e.g. "language_model.layers.3.self_attn.q_proj"), or nullptr.
  const LinearInfo* find_linear(std::string_view module) const;
  // F32 tensors of the file rounded to bf16, by canonical tensor name; uploaded by the encoder, not streamed.
  const std::map<std::string, std::vector<std::uint16_t>, std::less<>>& bf16_tensors() const { return bf16_; }
  // Shape of a bf16_tensors() entry as stored.
  const std::vector<std::int64_t>& bf16_shape(std::string_view name) const;
  // Largest n * k over the grouped linears (the encoder's decode buffer), 0 when there are none.
  std::int64_t max_grouped_elems() const { return max_grouped_elems_; }

 private:
  quant::QuantModel model_;
  weights::ModelIndex index_;
  const quant::QuantLinear* embed_ = nullptr;
  Precision precision_ = Precision::kBF16;
  std::map<std::string, LinearInfo, std::less<>> linears_;
  std::map<std::string, std::vector<std::uint16_t>, std::less<>> bf16_;
  std::map<std::string, std::vector<std::int64_t>, std::less<>> bf16_shapes_;
  std::int64_t max_grouped_elems_ = 0;
};

}  // namespace mi::qwen35_text
