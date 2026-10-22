// The C API's only contact with pipeline::Pipeline (P4 is still changing its fields): converts the C structs of
// mossinfer.h into a pipeline::Request / PipelineOptions, runs one generation and flattens the pipeline::Result
// into a Generation that holds only C-API types. When Pipeline changes, this file and pipeline_adapter.cpp are the
// whole rebase.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mossinfer/mossinfer.h"

namespace mi::pipeline {
struct Request;
struct Result;
struct PipelineOptions;
}  // namespace mi::pipeline

namespace mi::api {

// A model format or feature this build does not implement (MI_ERR_NOT_SUPPORTED).
struct NotSupportedError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// The model path is not a usable bundle, or a model or LoRA file cannot be applied (MI_ERR_MODEL).
struct ModelError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// At most this many runtime LoRAs per generation.
inline constexpr std::size_t kMaxLoras = 64;

// One finished generate() call, in C-API terms.
struct Generation {
  int width = 0, height = 0;
  std::vector<std::uint8_t> rgb;  // empty when cancelled
  bool cancelled = false;
  std::uint64_t seed = 0;
  int steps_scheduled = 0, steps_run = 0;
  mi_timings timings{};
  mi_memory_stats memory{};
  std::string plan_explanation;
  mi_block_path block_path = MI_BLOCK_PATH_DEFAULT;  // what ran
  std::string block_path_note;
  mi_dit_stream_stats dit_stream{};
  // The `moss-infer generate --json-report` document is report_head + <image path as a JSON string> + report_tail.
  std::string report_head, report_tail;
  // PNG text chunks (keyword, text) for mi_save_png(): the generation metadata, empty when save_metadata was off.
  std::vector<std::pair<std::string, std::string>> png_text;
  mi_tier_info tier{};
  // mi_result_text() kinds.
  std::string tier_reason, certificate, gpu_arch, dit_source, te_source, vae_source, quality_json;
  mi_prompt_tokens tokens[2]{};  // the prompt, the negative prompt
  std::vector<mi_lora_report> loras;
  std::vector<std::string> lora_files;
  std::vector<std::string> lora_warnings, lora_notes;
};

// The pipeline request of `params` (a full-size struct). Throws std::invalid_argument (the pipeline's wording) when it
// is not valid, including LoRA and dataspace files that do not exist. CPU only; `model` is left empty.
pipeline::Request to_request(const mi_gen_params& params);
// Throws std::invalid_argument when `params` (a full-size struct) is not a valid request. CPU only.
void validate(const mi_gen_params& params);
// The pipeline options of `params` (a full-size struct). Throws std::invalid_argument on an unknown enum value, a size
// out of range or a component file that does not exist. CPU only; the model path is checked by PipelineAdapter.
pipeline::PipelineOptions to_options(const mi_engine_params& params);

// Throws std::invalid_argument on invalid engine parameters, std::runtime_error subclasses on model or CUDA errors.
class PipelineAdapter {
 public:
  explicit PipelineAdapter(const mi_engine_params& params);
  ~PipelineAdapter();
  PipelineAdapter(const PipelineAdapter&) = delete;
  PipelineAdapter& operator=(const PipelineAdapter&) = delete;

  // Not thread-safe (one generation at a time); `cancel` is polled throughout.
  Generation generate(const mi_gen_params& params, const std::atomic<bool>& cancel);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Flattens a pipeline result (its image is moved out) into a Generation: the report, the PNG metadata (when
// q.save_metadata; without the prompts unless include_prompts) and every accessor's data. Called by generate(); public
// for the CPU tests.
Generation to_generation(const pipeline::Request& q, pipeline::Result& r, bool include_prompts);

// The `moss-infer generate --json-report` document of `generation`, with `image_path_utf8` as its "image".
std::string report_json(const Generation& generation, const std::string& image_path_utf8);

// Fills generation.report_head / report_tail from a pipeline request and result, with the shared renderer
// (pipeline/report_json.h, locale-independent). Called by to_generation(); public for the CPU tests.
void render_report(const pipeline::Request& q, const pipeline::Result& r, Generation& g);

// Atomic PNG write of a generation's image, with its metadata chunks. Throws std::invalid_argument without an image,
// std::runtime_error on I/O.
void write_png(const Generation& generation, const std::string& path_utf8);

// Atomic PNG write of RGB8 [height][width][3] without metadata. Throws std::invalid_argument on a bad size,
// std::runtime_error on I/O.
void write_png_rgb(const std::string& path_utf8, const std::uint8_t* rgb, int width, int height);

}  // namespace mi::api
