// pipeline/quality_report: engine numerics per configuration, the GPU class, and the report's quality object.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <string>

#include "backend/cuda_libs.h"
#include "kernels/nvfp4/nvfp4.h"
#include "pipeline/quality_report.h"
#include "util/json.h"

namespace {

using namespace mi::pipeline;
using mi::planner::AttentionMode;
using mi::planner::Precision;
using TePrecision = mi::qwen35_text::Precision;

bool has_key(const EngineNumerics& n, const std::string& k) {
  for (const auto& [name, v] : n) {
    if (name == k) return true;
  }
  return false;
}

void expect_sorted(const EngineNumerics& n) {
  for (std::size_t i = 1; i < n.size(); ++i) EXPECT_LT(n[i - 1].first, n[i].first);
}

TEST(QualityReport, NumericsDependOnTheConfiguration) {
  const RunConfig ref{};
  const RunConfig int8{Precision::kInt8, AttentionMode::kExact, TePrecision::kBF16};
  const RunConfig te8{Precision::kBf16, AttentionMode::kExact, TePrecision::kInt8};
  const RunConfig q8{Precision::kBf16, AttentionMode::kQ8Fp16, TePrecision::kBF16};
  const RunConfig all{Precision::kInt8, AttentionMode::kQ8Fp8, TePrecision::kInt8};
  EXPECT_NE(engine_numerics(int8), engine_numerics(ref));
  EXPECT_NE(engine_numerics(q8), engine_numerics(ref));
  EXPECT_EQ(engine_numerics(int8), engine_numerics(te8));  // one INT8 ConvRot kernel family
  for (const char* k : {"cublaslt", "elementwise", "fused_gemm", "te"}) EXPECT_TRUE(has_key(engine_numerics(ref), k)) << k;
  EXPECT_NE(engine_numerics(int8, 130800), engine_numerics(int8, 130900));
  const mi::backend::CublasLtStatus lt = mi::backend::cublaslt_status();
  EXPECT_EQ(engine_numerics(int8), engine_numerics(int8, lt.ok ? lt.info.version : 0));
  EXPECT_FALSE(has_key(engine_numerics(ref), "int8"));
  EXPECT_FALSE(has_key(engine_numerics(ref), "attention_q8"));
  EXPECT_TRUE(has_key(engine_numerics(int8), "int8"));
  EXPECT_FALSE(has_key(engine_numerics(int8), "attention_q8"));
  EXPECT_TRUE(has_key(engine_numerics(q8), "attention_q8"));
  EXPECT_TRUE(has_key(engine_numerics(all), "int8") && has_key(engine_numerics(all), "attention_q8"));
  const RunConfig int4{Precision::kInt4, AttentionMode::kExact, TePrecision::kBF16};
  const RunConfig w4a8{Precision::kInt8, AttentionMode::kExact, TePrecision::kW4A8};
  EXPECT_TRUE(has_key(engine_numerics(int4), "w4a4"));
  EXPECT_FALSE(has_key(engine_numerics(int4), "te_w4a8"));
  EXPECT_TRUE(has_key(engine_numerics(w4a8), "te_w4a8"));
  EXPECT_TRUE(has_key(engine_numerics(w4a8), "int8"));
  EXPECT_FALSE(has_key(engine_numerics(w4a8), "w4a4"));
  for (const RunConfig& c : {ref, int8, q8, all, int4, w4a8}) expect_sorted(engine_numerics(c));
}

// NVFP4: its own numerics key, plus "int8" when the file keeps INT8 ConvRot linears.
TEST(QualityReport, Nvfp4Numerics) {
  RunConfig all{Precision::kNvfp4, AttentionMode::kQ8Fp8, TePrecision::kBF16};
  RunConfig kept = all;
  kept.dit_int8_units = true;
  const EngineNumerics a = engine_numerics(all, 130800), k = engine_numerics(kept, 130800);
  EXPECT_TRUE(has_key(a, "nvfp4"));
  EXPECT_FALSE(has_key(a, "int8"));
  EXPECT_FALSE(has_key(a, "w4a4"));
  EXPECT_TRUE(has_key(a, "attention_q8"));
  EXPECT_TRUE(has_key(k, "nvfp4") && has_key(k, "int8"));
  EXPECT_FALSE(has_key(engine_numerics(RunConfig{}, 130800), "nvfp4"));
  for (const auto& [name, v] : a) {
    if (name == "nvfp4") {
      EXPECT_EQ(v, mi::kernels::nvfp4::kNvfp4NumericsVersion);
    }
  }
  expect_sorted(a);
  expect_sorted(k);
  RunConfig te8 = all;
  te8.te = TePrecision::kInt8;
  EXPECT_TRUE(has_key(engine_numerics(te8, 130800), "int8"));
}

TEST(QualityReport, CurrentGpuArch) {
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  EXPECT_EQ(current_gpu_arch(), gpu_arch_class(prop.major, prop.minor));
}

TEST(QualityReport, ReportObject) {
  QualityReport r;
  r.tier = mi::planner::Tier::kHigh;
  r.config = {Precision::kInt8, AttentionMode::kQ8Fp16, TePrecision::kBF16};
  r.gpu_arch = "sm_89";
  r.dit = "/m/x \"q\".safetensors";
  r.dit_bytes = 9'876'543'210ull;
  const mi::json::Value v = mi::json::parse(quality_report_json(r));
  EXPECT_EQ(v.at("tier").as_string(), "high");
  EXPECT_EQ(v.at("config").at("dit_precision").as_string(), "int8_convrot");
  EXPECT_EQ(v.at("config").at("attention").as_string(), "q8_fp16");
  EXPECT_EQ(v.at("config").at("te_precision").as_string(), "bf16");
  EXPECT_EQ(v.at("config").at("lora").as_string(), "none");
  EXPECT_EQ(v.at("gpu_arch").as_string(), "sm_89");
  EXPECT_EQ(v.at("dit").at("path").as_string(), "/m/x \"q\".safetensors");
  EXPECT_EQ(v.at("dit").at("bytes").as_int(), 9'876'543'210ll);
  EXPECT_TRUE(v.at("te").is_null());
  for (const char* gone : {"reason", "certificate"}) EXPECT_EQ(v.find(gone), nullptr) << gone;
  const EngineNumerics n = engine_numerics(r.config);
  ASSERT_EQ(v.at("numerics").as_object().size(), n.size());
  for (const auto& [k, x] : n) EXPECT_EQ(static_cast<std::uint64_t>(v.at("numerics").at(k).as_int()), x) << k;
  const mi::backend::CublasLtStatus lt = mi::backend::cublaslt_status();
  if (lt.ok) {
    EXPECT_EQ(v.at("cublaslt").at("source").as_string(), lt.info.source);
    EXPECT_EQ(static_cast<std::uint64_t>(v.at("cublaslt").at("version").as_int()), lt.info.version);
    EXPECT_EQ(v.at("numerics").at("cublaslt").as_int(), v.at("cublaslt").at("version").as_int());
    EXPECT_EQ(v.at("cublaslt").at("path").as_string(), lt.info.path);
  } else {
    EXPECT_TRUE(v.at("cublaslt").is_null());
    EXPECT_EQ(v.at("numerics").at("cublaslt").as_int(), 0);
  }

  QualityReport ref;
  ref.gpu_arch = "sm_86";
  ref.dit = "/bundle/transformer";
  ref.dit_bytes = 5;
  const mi::json::Value rv = mi::json::parse(quality_report_json(ref));
  EXPECT_EQ(rv.at("tier").as_string(), "reference");
  EXPECT_EQ(rv.at("config").at("attention").as_string(), "exact");
  QualityReport turbo = r;
  turbo.tier = mi::planner::Tier::kTurbo;
  turbo.config = {Precision::kNvfp4, AttentionMode::kQ8Fp8, TePrecision::kW4A8, true};
  turbo.te = "/m/te.safetensors";
  turbo.te_bytes = 7;
  const mi::json::Value tv = mi::json::parse(quality_report_json(turbo));
  EXPECT_EQ(tv.at("tier").as_string(), "turbo");
  EXPECT_EQ(tv.at("config").at("dit_precision").as_string(), "nvfp4");
  EXPECT_EQ(tv.at("config").at("te_precision").as_string(), "w4a8");
  EXPECT_EQ(tv.at("config").at("lora").as_string(), "active");
  EXPECT_EQ(tv.at("te").at("bytes").as_int(), 7);
}

}  // namespace
