// mi_inspect_file(): what a file is to the engine, decided by the engine's own loaders (weights, quant, models) run up
// to the point where they would upload, without an engine or a CUDA call.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "mossinfer/mossinfer.h"
#include "planner/planner.h"

namespace mi::pipeline {
struct DitFileInfo;
struct GgufDitInfo;
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
// the quantization text is DitFileInfo::linears (moss_dit::describe_linears of its index); the note is the NVFP4
// remark (for NVFP4 linears), then DitFileInfo::remarks (the static input_scale remark) when there is one.
void describe_dit_linears(const pipeline::DitFileInfo& dit, FileInspection& r);

// A loadable DiT's tier bits and reasons: planner::resolve_tier's verdict for each tier, with the bundle's bf16 text
// encoder, no explicit attention and every 8-bit kernel available (each GPU the engine supports has one, so the bits
// are generate's on every GPU), NVFP4 judged for an RTX 50 with `nvfp4_kernels` (this build's). A GGUF file (`gguf`):
// each tier judged with the precision its conversion gives (pipeline::gguf_target; a tier it refuses is not set), the
// reason ending in what the file runs as.
void describe_dit_tiers(planner::Precision precision, bool nvfp4_kernels, FileInspection& r,
                        const pipeline::GgufDitInfo* gguf = nullptr);

}  // namespace mi::api
