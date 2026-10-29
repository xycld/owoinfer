// GGUF DiT files (PipelineOptions::dit_file, mi_inspect_file): the engine reads a community MOSS-Image GGUF DiT by
// converting it once into one of its own formats and loading that file like any other; there is no GGUF runtime path.
//   - tier high: every supported type (Q8_0, Q6_K, Q5_K, Q5_1, Q5_0, Q4_K, Q4_1, Q4_0, BF16, F16, F32) -> INT8 ConvRot
//     (G = 256);
//   - tier reference: a file whose every tensor is BF16, F16 or F32 -> bf16 (a Q8_0 file is not: rounded to bf16 its
//     weights would be rounded twice, and INT8 runs its own grid more closely);
//   - tier turbo: refused (no 4-bit target), as is any other type (Q2_K, Q3_K, IQ*, custom types).
// So VRAM and speed are the target format's. The converted file is kept in <cache_dir>/gguf/, found again by the GGUF
// file's identity (size, modification time, XXH64 of its header, the target and the converter version): only the first
// load converts (quant::convert, streamed one tensor at a time, under 0.6 GB of host RAM for a MOSS-Image DiT); later
// loads read the converted file directly. A GGUF DiT therefore needs a cache directory.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "planner/planner.h"
#include "weights/quant/quant_model.h"

namespace mi::pipeline {

struct GgufDitInfo {
  // The GGUF types of every stored tensor, the most bytes first, joined by " + " (e.g. "Q4_K + Q6_K + F32",
  // "Q8_0 + F32", "BF16 + F32").
  std::string types;
  // Every stored tensor is BF16, F16 or F32: tier reference runs the file as bf16.
  bool reference = false;
  int tensors = 0;  // in the GGUF header
};

// The file's header checked as the converter will read it (weights::check_supported_gguf_dit, quant::open_gguf_dit):
// names, shapes and types, no tensor byte. Throws weights::UnsupportedModelError, quant::QuantError,
// weights::FormatError or weights::IoError.
GgufDitInfo inspect_gguf_dit(const std::filesystem::path& file);

// The format the file runs as at `tier`: INT8 ConvRot 256 (high), bf16 (reference, when info.reference); otherwise
// throws std::invalid_argument naming the tiers that run it.
quant::QuantSpec gguf_target(const GgufDitInfo& info, planner::Tier tier);

// "GGUF Q4_K, runs as INT8 ConvRot" / "GGUF Q8_0, runs as bf16".
std::string gguf_runs_as(const GgufDitInfo& info, const quant::QuantSpec& target);

// The converted file of `file` for `target` in <cache_dir>/gguf/, converted now when the cache has none for this
// identity (written under a temporary name, read back, then renamed into place; a converted copy of an earlier version
// of the same GGUF path is removed). `converted` (optional) is set when this call converted. Throws std::invalid_argument
// for an empty cache_dir, weights::IoError when the cache directory cannot hold the file (free space is checked
// first), or the converter's errors.
std::filesystem::path gguf_converted_file(const std::filesystem::path& file, const GgufDitInfo& info,
                                          const quant::QuantSpec& target, const std::filesystem::path& cache_dir,
                                          bool* converted = nullptr);

}  // namespace mi::pipeline
