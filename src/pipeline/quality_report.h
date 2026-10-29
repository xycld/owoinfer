// The "quality" object of a generation report (`--json-report`, mi_result_report_json, MI_TEXT_QUALITY_JSON): a
// factual description of what ran, so a caller (or the internal regression harness, tools/reference/certify.py) can
// confirm it:
//   {"tier": "high", "config": {"dit_precision": ..., "attention": ..., "te_precision": ..., "lora": "none"|"active"},
//    "numerics": {...engine_numerics(config)},
//    "cublaslt": {"source": "system"|"bundled"|"env", "version": 130800, "path": "<file>"}|null (none usable),
//    "gpu_arch": "sm_89", "dit": {"path": "...", "bytes": N}, "te": {"path": "...", "bytes": N}|null}
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "models/qwen35_text/quant_text_model.h"
#include "planner/planner.h"

namespace mi::pipeline {

// Report tokens: "bf16" | "int8_convrot" | "w4a4" | "nvfp4" for the DiT, "bf16" | "int8_convrot" | "w4a8" for the text
// encoder (the attention's are planner::attention_name()'s).
std::string_view precision_token(planner::Precision p);
std::string_view precision_token(qwen35_text::Precision p);

// What changes the output bits of a generation beyond the weights.
struct RunConfig {
  planner::Precision dit = planner::Precision::kBf16;
  planner::AttentionMode attention = planner::AttentionMode::kExact;
  qwen35_text::Precision te = qwen35_text::Precision::kBF16;
  bool lora = false;  // runtime LoRAs (a control adapter's included) active
  // An NVFP4 DiT that keeps some linears INT8 ConvRot (a keep list): its output depends on the INT8 numerics too.
  bool dit_int8_units = false;
};

// Engine numerics versions a configuration's output bits depend on, sorted by name; a kernel or op-order change bumps
// one. Keys:
//   "cublaslt" cublasLtGetVersion() of the cuBLASLt copy in use (backend/cuda_libs.h; 0 when none is usable): always
//     (every configuration runs cuBLASLt GEMMs, whose result bits may change between its versions, and a system copy
//     may differ from the bundled one);
//   "elementwise" kernels::kElementwiseNumericsVersion, "fused_gemm" kernels::fused::kFusedGemmVersion and "te"
//     qwen35_text::kNumericsVersion: always (every configuration runs the bf16 elementwise ops, the bf16 GEMM routing
//     of the layers kept in bf16, and the text encoder's own op order);
//   "int8" kernels::quant::kInt8NumericsVersion: when the DiT or the TE is INT8 ConvRot, the TE is W4A8 (its
//     activations are INT8) or the DiT is NVFP4 with INT8 ConvRot keep-list linears (dit_int8_units);
//   "w4a4" kernels::quant4::kInt4NumericsVersion: when the DiT is INT4 (W4A4);
//   "nvfp4" kernels::nvfp4::kNvfp4NumericsVersion: when the DiT is NVFP4;
//   "te_w4a8" kernels::quant::kGroupedDecodeVersion: when the TE is W4A8;
//   "attention_q8" attention_q8::kNumericsVersion: when the attention is not exact.
using EngineNumerics = std::vector<std::pair<std::string, std::uint64_t>>;
EngineNumerics engine_numerics(const RunConfig& c);  // with the version of the cuBLASLt copy this process uses
EngineNumerics engine_numerics(const RunConfig& c, std::uint64_t cublaslt_version);  // pure

// "sm_<major><minor>": the GPU class (one kernel image and numerics per class).
std::string gpu_arch_class(int major, int minor);
// gpu_arch_class() of the current CUDA device.
std::string current_gpu_arch();

struct QualityReport {
  planner::Tier tier = planner::Tier::kReference;
  RunConfig config;
  std::string gpu_arch;         // current_gpu_arch()
  std::filesystem::path dit;    // the DiT file or directory loaded
  std::uint64_t dit_bytes = 0;  // its total size
  std::filesystem::path te;     // the separately loaded TE; empty when bf16 from the bundle
  std::uint64_t te_bytes = 0;
};
std::string quality_report_json(const QualityReport& r);

}  // namespace mi::pipeline
