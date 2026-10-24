// Calibration Hessians for the NVFP4 converter's gptq recipes (`moss-infer quantize --format nvfp4 --recipe gptq*`;
// NVFP4 plan "N3 converter: reproducible output").
//
// Built-in calibration set (kNvfp4CalibrationSetId): the NVFP4 study's calibration prompts (E30), the official app's
// gallery assets c_asset001, c_asset005 and c_asset018, each at 1024 x 1024 with 50 steps and its own CFG scale, shift,
// seed, negative prompt and uncond_skip_layers (euler, the moss schedule, the official noise). The engine's own bf16
// pipeline samples each; at steps 6, 18, 30 and 43 (0-based, the study's timestep buckets) every block GEMM input of
// both CFG rows, the cond and the uncond row (a skipped block's uncond row included as the block runs it: not at all),
// is accumulated into H = X^T X with kernels::gram_accumulate_upper. Sampling stops after step 43.
//
// Determinism: each H element is one fixed sequence of fp32 operations over the calibration rows in a fixed order
// (prompts in order, steps ascending, each forward's rows as the DiT orders them), and the capture runs the op-by-op
// bf16 blocks, whose bits are those of the engine's bf16 numerics on this GPU class with this cuBLASLt. So two captures
// on the same capture class give the same H bits, and the converter the same file. The capture class (GPU class, the
// text encoder, which must be the bundle's bf16 one, engine numerics, Gram kernel version) goes to the file's metadata;
// the converter adds the digest of the Hessians it used.
//
// Memory: H stays on the device, fp32 [K, K] per unit (37.7 MB for K = 3072, 268 MB for K = 8192; 16 GB for all 168
// units). Blocks are captured in passes that fit next to the pipeline: each pass samples the three prompts again
// (the text embeddings are cached) with the H of its blocks. The pass split does not change any bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "pipeline/pipeline.h"
#include "weights/quant/quant_convert.h"

namespace mi::pipeline {

inline constexpr std::string_view kNvfp4CalibrationSetId = "moss-nvfp4-calib-1";

struct CalibrationJob {
  std::string id;  // the study's name, e.g. "c_asset001"
  Request request;
};
// The set's requests for bundle `model` (decode off).
std::vector<CalibrationJob> nvfp4_calibration_jobs(const std::filesystem::path& model);
// The 0-based steps whose DiT inputs are accumulated.
const std::vector<int>& nvfp4_calibration_steps();
// XXH64 (seed 0) of the set's canonical description (ids, prompts, sizes, steps, guidance, seeds, capture steps).
std::uint64_t nvfp4_calibration_set_xxh64();

struct CalibrationOptions {
  std::filesystem::path model;     // the bundle: tokenizer, text encoder, (transformer)
  std::filesystem::path dit_file;  // a single-file bf16 DiT to capture with instead of the bundle's transformer
  // Device bytes the Hessians of one pass may take; 0: the free memory less kCalibrationHeadroom.
  std::size_t hessian_bytes = 0;
  std::function<void(std::string_view)> log;
};
// Device memory left to the pipeline next to a pass's Hessians by default: the bf16 DiT (~10 GB), the denoise
// workspace at 1024 x 1024 with two rows, and the text encoder's streaming buffers.
inline constexpr std::size_t kCalibrationHeadroom = std::size_t{14} << 30;

// The converter's Hessian source (quant::ConvertOptions::nvfp4->hessians) that captures with the engine on the current
// CUDA device. Throws std::runtime_error without a CUDA device, planner::BudgetError when the pipeline does not fit.
quant::HessianSource nvfp4_hessian_source(CalibrationOptions options);

}  // namespace mi::pipeline
