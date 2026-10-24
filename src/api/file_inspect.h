// mi_inspect_file(): what a file is to the engine, decided by the engine's own loaders (weights, quant, models) run up
// to the point where they would upload, without an engine or a CUDA call.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "mossinfer/mossinfer.h"

namespace mi::pipeline {
struct DitFileInfo;
}

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

// A loadable DiT's MI_FILE_TEXT_QUANTIZATION and note, from pipeline::inspect_dit_file's answer `dit` (no refusal):
// DitFileInfo::linears is moss_dit::describe_linears of its index, then "; " and the static input_scale remark when
// there is one. The quantization text is describe_linears'; the note is the NVFP4 remark (for NVFP4 linears), then the
// input_scale remark. When `linears` is neither describe_linears' text nor starts with it and "; ", all of it is the
// quantization text and the note gets nothing from it.
void describe_dit_linears(const pipeline::DitFileInfo& dit, FileInspection& r);

}  // namespace mi::api
