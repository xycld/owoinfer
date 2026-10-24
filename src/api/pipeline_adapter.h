// The C API's only contact with pipeline::Pipeline (P4 is still changing its fields): converts the C structs of
// mossinfer.h into a pipeline::Request / PipelineOptions, runs one generation and flattens the pipeline::Result
// into a Generation that holds only C-API types. When Pipeline changes, this file and pipeline_adapter.cpp are the
// whole rebase.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
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
struct ReleaseReport;
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
// At most this many control images per generation (one per kind; the published adapters have two).
inline constexpr std::size_t kMaxControlImages = 16;
// At most this many extra certificate directories per engine (mi_engine_params.certificate_dirs).
inline constexpr std::size_t kMaxCertificateDirs = 16;

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
  mi_image_info image{};
  std::string image_note, time_embedding;  // mi_result_text() kinds
  mi_control_info control{};
  // mi_result_text() kinds (control_kinds comma-separated), and the control LoRA part's messages.
  std::string control_adapter, control_kinds, control_base_model, control_encode_schedule;
  std::vector<std::string> control_warnings, control_notes;
  mi_accel_info accel{};
  std::string accelerators;               // MI_TEXT_ACCELERATORS
  std::vector<std::int32_t> cached_steps;  // mi_result_accel_cached_steps()
  std::vector<double> step_cache_diffs;    // mi_result_accel_step_cache_diffs()
  std::vector<std::string> accel_warnings;  // MI_MESSAGE_ACCEL_WARNING
  mi_dit_load_info dit_load{};
  // mi_result_text() kinds.
  std::string dit_load_note, te_note, preload_note, step_cache_threshold_source;
};

// The pipeline request of `params` (a full-size struct): the unset markers resolved (mossinfer.h), a distill bundle
// expanded, image inputs (control images included) copied or decoded, a dataspace sampler's file read and the steps
// checked against it, the control adapter's file and the accelerators checked (its kinds are generate()'s). Throws
// std::invalid_argument (the pipeline's wording) when it is not valid, including LoRA, dataspace and image files that do not exist; ModelError for a malformed distill bundle or its
// dataspace file. `distill_time_embedding` (optional) receives the bundle's time embedding (empty: none) for the
// engine's check. CPU only; `model` is left empty.
// load_images false (mi_gen_params_validate): image inputs are checked (sizes, channels, file existence and size) but
// neither copied nor decoded; 1x1 stand-ins take their place.
// preload true (mi_engine_preload): a NULL or blank prompt is allowed (the request keeps it blank; Pipeline::preload
// plans for a short prompt); everything else is checked as for a generation.
pipeline::Request to_request(const mi_gen_params& params, std::filesystem::path* distill_time_embedding = nullptr,
                             bool load_images = true, bool preload = false);
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

  // Not thread-safe (one job at a time); `cancel` is polled throughout.
  Generation generate(const mi_gen_params& params, const std::atomic<bool>& cancel);
  // Pipeline::preload of `params` (prompt optional): the plan callback once after planning, before any load; a progress
  // callback (step 0) before each load step, in the pipeline's order (PreloadOptions::on_phase).
  Generation preload(const mi_gen_params& params, const std::atomic<bool>& cancel);
  // Pipeline::release_device_memory. `cancel` already set on entry: nothing is released and *cancelled is set (the
  // report as to_release_info()); set during the host copy: the copy stops, the release completes.
  mi_release_info release(bool keep_host_copy, const std::atomic<bool>& cancel, bool* cancelled = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Flattens a pipeline result (its image is moved out) into a Generation: the report, the PNG metadata (when
// q.save_metadata; without the prompts unless include_prompts) and every accessor's data. Called by generate(); public
// for the CPU tests.
Generation to_generation(const pipeline::Request& q, pipeline::Result& r, bool include_prompts);

// A release report in C-API terms, the note cut at 511 bytes on a UTF-8 boundary. Called by release(); public for the
// CPU tests.
// `cancelled` (optional) receives ReleaseReport::cancelled: the release was cancelled before it started and released
// nothing.
mi_release_info to_release_info(const pipeline::ReleaseReport& report, bool* cancelled = nullptr);

// A result holding `generation`, as mi_generate() returns it (release it with mi_result_free()). Defined in c_api.cpp;
// public for the CPU tests of the result accessors.
mi_result* new_result(Generation generation);

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
