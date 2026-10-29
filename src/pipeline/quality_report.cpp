#include "pipeline/quality_report.h"

#include <cuda_runtime.h>

#include <algorithm>

#include "backend/cuda_libs.h"
#include "core/cuda_check.h"
#include "kernels/attention_q8/attention_q8.h"
#include "kernels/basic/elementwise.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/nvfp4/nvfp4.h"
#include "kernels/quant/quant.h"
#include "kernels/quant/w4a8_decode.h"
#include "kernels/quant4/quant4.h"
#include "models/qwen35_text/text_encoder.h"
#include "pipeline/report_json.h"
#include "util/path.h"

namespace mi::pipeline {
namespace {

std::string numerics_json(const EngineNumerics& n) {
  std::string s = "{";
  for (std::size_t i = 0; i < n.size(); ++i) s += (i == 0 ? "" : ", ") + json_escape(n[i].first) + ": " + std::to_string(n[i].second);
  return s + "}";
}

}  // namespace

std::string_view precision_token(planner::Precision p) {
  switch (p) {
    case planner::Precision::kBf16: return "bf16";
    case planner::Precision::kInt8: return "int8_convrot";
    case planner::Precision::kInt4: return "w4a4";
    case planner::Precision::kNvfp4: return "nvfp4";
  }
  return "?";
}

std::string_view precision_token(qwen35_text::Precision p) {
  switch (p) {
    case qwen35_text::Precision::kBF16: return "bf16";
    case qwen35_text::Precision::kInt8: return "int8_convrot";
    case qwen35_text::Precision::kW4A8: return "w4a8";
  }
  return "?";
}

EngineNumerics engine_numerics(const RunConfig& c) {
  const backend::CublasLtStatus lt = backend::cublaslt_status();
  return engine_numerics(c, lt.ok ? lt.info.version : 0);
}

EngineNumerics engine_numerics(const RunConfig& c, std::uint64_t cublaslt_version) {
  using planner::Precision;
  using TePrecision = qwen35_text::Precision;
  EngineNumerics n{{"cublaslt", cublaslt_version},
                   {"elementwise", kernels::kElementwiseNumericsVersion},
                   {"fused_gemm", kernels::fused::kFusedGemmVersion},
                   {"te", qwen35_text::kNumericsVersion}};
  if (c.dit == Precision::kInt8 || (c.dit == Precision::kNvfp4 && c.dit_int8_units) || c.te == TePrecision::kInt8 ||
      c.te == TePrecision::kW4A8) {
    n.emplace_back("int8", kernels::quant::kInt8NumericsVersion);
  }
  if (c.dit == Precision::kInt4) n.emplace_back("w4a4", kernels::quant4::kInt4NumericsVersion);
  if (c.dit == Precision::kNvfp4) n.emplace_back("nvfp4", kernels::nvfp4::kNvfp4NumericsVersion);
  if (c.te == TePrecision::kW4A8) n.emplace_back("te_w4a8", kernels::quant::kGroupedDecodeVersion);
  if (planner::is_q8(c.attention)) n.emplace_back("attention_q8", attention_q8::kNumericsVersion);
  std::sort(n.begin(), n.end());
  return n;
}

std::string gpu_arch_class(int major, int minor) { return "sm_" + std::to_string(major * 10 + minor); }

std::string current_gpu_arch() {
  int device = 0, major = 0, minor = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return gpu_arch_class(major, minor);
}

std::string quality_report_json(const QualityReport& r) {
  const auto component = [](const std::filesystem::path& p, std::uint64_t bytes) {
    if (p.empty()) return std::string("null");
    return "{\"path\": " + json_escape(util::path_utf8(p)) + ", \"bytes\": " + std::to_string(bytes) + "}";
  };
  const RunConfig& c = r.config;
  const backend::CublasLtStatus lt = backend::cublaslt_status();
  const std::string cublaslt = lt.ok ? "{\"source\": " + json_escape(lt.info.source) + ", \"version\": " +
                                           std::to_string(lt.info.version) + ", \"path\": " + json_escape(lt.info.path) + "}"
                                     : std::string("null");
  return "{\"tier\": \"" + std::string(planner::tier_name(r.tier)) + "\", \"config\": {\"dit_precision\": \"" +
         std::string(precision_token(c.dit)) + "\", \"attention\": \"" + planner::attention_name(c.attention) +
         "\", \"te_precision\": \"" + std::string(precision_token(c.te)) + "\", \"lora\": \"" + (c.lora ? "active" : "none") +
         "\"}, \"numerics\": " + numerics_json(engine_numerics(c, lt.ok ? lt.info.version : 0)) +
         ", \"cublaslt\": " + cublaslt + ", \"gpu_arch\": " + json_escape(r.gpu_arch) +
         ", \"dit\": " + component(r.dit, r.dit_bytes) + ", \"te\": " + component(r.te, r.te_bytes) + "}";
}

}  // namespace mi::pipeline
