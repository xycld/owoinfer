// mi_inspect_file(): what a file is to the engine, decided by the engine's own loaders (weights, quant, models) run up
// to the point where they would upload, without an engine or a CUDA call.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "mossinfer/mossinfer.h"

namespace mi::api {

struct FileInspection {
  mi_file_info info{};
  std::string note, quantization, base_model, adapter_types;
  std::vector<std::string> control_kinds;
  std::string tier_reasons[4];  // by mi_tier
};

// Throws std::invalid_argument for a path that is not a regular file and weights::IoError for one that does not exist
// or cannot be read. Anything else the file is (or is not) is the answer, not an error.
FileInspection inspect_file(const std::filesystem::path& path);

}  // namespace mi::api
