// mossinfer.h over mi::api::PipelineAdapter. Every entry point catches every exception and turns it into an mi_status
// plus the thread-local message, so nothing C++ crosses the C boundary.
#include "mossinfer/mossinfer.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "api/component_adapter.h"
#include "api/file_inspect.h"
#include "api/pipeline_adapter.h"
#include "backend/cuda_driver.h"
#include "backend/cuda_libs.h"
#include "backend/probe.h"
#include "core/cuda_check.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/run_schedule.h"
#include "pipeline/sampler.h"
#include "pipeline/samplers/initial_noise.h"
#include "planner/planner.h"
#include "runtime/device_arena.h"
#include "runtime/exchange.h"
#include "runtime/job_gate.h"
#include "runtime/session_slot.h"
#include "util/json.h"
#include "util/path.h"
#include "util/regular_file.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/safetensors.h"
#include "weights/upload.h"

struct mi_engine {
  // The one gate of the engine's jobs (runtime/job_gate.h): mi_generate, mi_engine_preload, mi_engine_release, every
  // component call, and the cancel flag the running job watches. Declared first: it outlives everything that runs
  // through it.
  mi::runtime::JobGate gate;
  std::unique_ptr<mi::api::PipelineAdapter> adapter;
  // The open session, or the last one (closed): runtime/session_slot.h. The gate refuses calls on a closed one.
  mi::runtime::SessionSlot<mi::api::ComponentSession> session;
};

struct mi_result {
  mi::api::Generation generation;
  std::string image_path;  // last successful mi_save_png()
  std::string report;
};

struct mi_file_report {
  mi::api::FileInspection inspection;
};

namespace {

// ---------------------------------------------------------------- frozen layouts (ABI 1, docs/c-api.md)
// Every public struct has explicit padding (`_reserved*`), so the C compilers and this C++ compiler agree on it.
// tests/c/test_abi_layout.c checks the same table from C.
static_assert(sizeof(void*) == 8 && sizeof(std::size_t) == 8, "the C API layouts are defined for 64-bit targets");
static_assert(sizeof(mi_status) == 4 && sizeof(mi_mode) == 4 && sizeof(mi_model_format) == 4 &&
              sizeof(mi_sampler) == 4 && sizeof(mi_phase) == 4 && sizeof(mi_block_path) == 4 && sizeof(mi_tier) == 4 &&
              sizeof(mi_attention) == 4 && sizeof(mi_precision) == 4 && sizeof(mi_scheduler) == 4 &&
              sizeof(mi_noise_layout) == 4 && sizeof(mi_noise_source) == 4 && sizeof(mi_result_text_kind) == 4 &&
              sizeof(mi_message_kind) == 4 && sizeof(mi_mask_encoding) == 4 && sizeof(mi_resize_mode) == 4 &&
              sizeof(mi_image_mode) == 4 && sizeof(mi_step_cache) == 4 && sizeof(mi_direct_io) == 4 &&
              sizeof(mi_te_host_pin) == 4 && sizeof(mi_dit_load_source) == 4 && sizeof(mi_load_io) == 4 &&
              sizeof(mi_file_kind) == 4 && sizeof(mi_file_text_kind) == 4,
              "C API enums are 32-bit");
#define MI_LAYOUT(type, field, offset) static_assert(offsetof(type, field) == (offset), #type "." #field)
static_assert(sizeof(mi_device_info) == 592);
MI_LAYOUT(mi_device_info, struct_size, 0);
MI_LAYOUT(mi_device_info, index, 8);
MI_LAYOUT(mi_device_info, name, 12);
MI_LAYOUT(mi_device_info, pci_bus_id, 268);
MI_LAYOUT(mi_device_info, compute_major, 300);
MI_LAYOUT(mi_device_info, compute_minor, 304);
MI_LAYOUT(mi_device_info, _reserved0, 308);
MI_LAYOUT(mi_device_info, total_memory_bytes, 312);
MI_LAYOUT(mi_device_info, free_memory_bytes, 320);
MI_LAYOUT(mi_device_info, supported, 328);
MI_LAYOUT(mi_device_info, unsupported_reason, 332);
MI_LAYOUT(mi_device_info, _reserved1, 588);
static_assert(sizeof(mi_probe_info) == 1624);
MI_LAYOUT(mi_probe_info, struct_size, 0);
MI_LAYOUT(mi_probe_info, driver_cuda_version, 8);
MI_LAYOUT(mi_probe_info, runtime_cuda_version, 12);
MI_LAYOUT(mi_probe_info, driver_compatible, 16);
MI_LAYOUT(mi_probe_info, device_count, 20);
MI_LAYOUT(mi_probe_info, supported_device_count, 24);
MI_LAYOUT(mi_probe_info, build_flavour, 28);
MI_LAYOUT(mi_probe_info, recommended_flavour, 44);
MI_LAYOUT(mi_probe_info, message, 60);
MI_LAYOUT(mi_probe_info, _reserved0, 572);  // the 0.2.0 layout ends here (576 bytes)
MI_LAYOUT(mi_probe_info, cublaslt_source, 576);
MI_LAYOUT(mi_probe_info, cublaslt_version, 592);
MI_LAYOUT(mi_probe_info, _reserved1, 596);
MI_LAYOUT(mi_probe_info, cublaslt_path, 600);
MI_LAYOUT(mi_probe_info, cublaslt_error, 1112);
static_assert(sizeof(mi_engine_params) == 160);
MI_LAYOUT(mi_engine_params, struct_size, 0);
MI_LAYOUT(mi_engine_params, model_path, 8);
MI_LAYOUT(mi_engine_params, model_format, 16);
MI_LAYOUT(mi_engine_params, mode, 20);
MI_LAYOUT(mi_engine_params, device_index, 24);
MI_LAYOUT(mi_engine_params, _reserved0, 28);
MI_LAYOUT(mi_engine_params, vram_budget_bytes, 32);
MI_LAYOUT(mi_engine_params, vram_reserve_bytes, 40);
MI_LAYOUT(mi_engine_params, host_ram_budget_bytes, 48);
MI_LAYOUT(mi_engine_params, cache_dir, 56);
MI_LAYOUT(mi_engine_params, embedding_cache, 64);
MI_LAYOUT(mi_engine_params, direct_io, 68);
MI_LAYOUT(mi_engine_params, block_path, 72);
MI_LAYOUT(mi_engine_params, _reserved1, 76);  // the 0.1 layout ends here (80 bytes)
MI_LAYOUT(mi_engine_params, dit_file, 80);
MI_LAYOUT(mi_engine_params, text_encoder_file, 88);
MI_LAYOUT(mi_engine_params, vae_file, 96);
MI_LAYOUT(mi_engine_params, tier, 104);
MI_LAYOUT(mi_engine_params, attention, 108);
MI_LAYOUT(mi_engine_params, lora_strict, 112);
MI_LAYOUT(mi_engine_params, _reserved2, 116);  // the first 0.2.0 layout ends here (120 bytes)
MI_LAYOUT(mi_engine_params, time_embedding_file, 120);
MI_LAYOUT(mi_engine_params, distill_dir, 128);  // the 1.0.0 layout ends here (136 bytes)
MI_LAYOUT(mi_engine_params, certificate_dirs, 136);
MI_LAYOUT(mi_engine_params, certificate_dirs_count, 144);
MI_LAYOUT(mi_engine_params, dit_direct_io, 152);
MI_LAYOUT(mi_engine_params, te_host_pin, 156);
static_assert(sizeof(mi_progress) == 56);
MI_LAYOUT(mi_progress, struct_size, 0);
MI_LAYOUT(mi_progress, phase, 8);
MI_LAYOUT(mi_progress, step, 12);
MI_LAYOUT(mi_progress, total_steps, 16);
MI_LAYOUT(mi_progress, _reserved0, 20);
MI_LAYOUT(mi_progress, step_seconds, 24);
MI_LAYOUT(mi_progress, elapsed_seconds, 32);  // the 1.0.0 layout ends here (40 bytes)
MI_LAYOUT(mi_progress, fraction, 40);
MI_LAYOUT(mi_progress, eta_seconds, 48);
static_assert(sizeof(mi_gen_params) == 376);
MI_LAYOUT(mi_gen_params, struct_size, 0);
MI_LAYOUT(mi_gen_params, prompt, 8);
MI_LAYOUT(mi_gen_params, negative_prompt, 16);
MI_LAYOUT(mi_gen_params, width, 24);
MI_LAYOUT(mi_gen_params, height, 28);
MI_LAYOUT(mi_gen_params, steps, 32);
MI_LAYOUT(mi_gen_params, _reserved0, 36);
MI_LAYOUT(mi_gen_params, cfg, 40);
MI_LAYOUT(mi_gen_params, shift, 48);
MI_LAYOUT(mi_gen_params, seed, 56);
MI_LAYOUT(mi_gen_params, uncond_skip_layers, 64);
MI_LAYOUT(mi_gen_params, uncond_skip_layers_count, 72);
MI_LAYOUT(mi_gen_params, slg_layers, 80);
MI_LAYOUT(mi_gen_params, slg_layers_count, 88);
MI_LAYOUT(mi_gen_params, slg_scale, 96);
MI_LAYOUT(mi_gen_params, slg_start, 104);
MI_LAYOUT(mi_gen_params, slg_stop, 112);
MI_LAYOUT(mi_gen_params, max_sequence_length, 120);
MI_LAYOUT(mi_gen_params, sampler, 124);
MI_LAYOUT(mi_gen_params, progress_callback, 128);
MI_LAYOUT(mi_gen_params, plan_callback, 136);
MI_LAYOUT(mi_gen_params, preview_callback, 144);
MI_LAYOUT(mi_gen_params, preview_interval, 152);
MI_LAYOUT(mi_gen_params, _reserved1, 156);
MI_LAYOUT(mi_gen_params, user_data, 160);  // the 0.1 layout ends here (168 bytes)
MI_LAYOUT(mi_gen_params, scheduler, 168);
MI_LAYOUT(mi_gen_params, noise_layout, 172);
MI_LAYOUT(mi_gen_params, noise_source, 176);
MI_LAYOUT(mi_gen_params, save_metadata, 180);
MI_LAYOUT(mi_gen_params, flow_multiplier, 184);
MI_LAYOUT(mi_gen_params, eta, 192);
MI_LAYOUT(mi_gen_params, s_noise, 200);
MI_LAYOUT(mi_gen_params, dataspace_path, 208);
MI_LAYOUT(mi_gen_params, loras, 216);
MI_LAYOUT(mi_gen_params, loras_count, 224);
MI_LAYOUT(mi_gen_params, include_prompts, 232);
MI_LAYOUT(mi_gen_params, _reserved2, 236);  // the first 0.2.0 layout ends here (240 bytes)
MI_LAYOUT(mi_gen_params, init_image, 240);
MI_LAYOUT(mi_gen_params, init_image_path, 248);
MI_LAYOUT(mi_gen_params, mask, 256);
MI_LAYOUT(mi_gen_params, mask_path, 264);
MI_LAYOUT(mi_gen_params, mask_encoding, 272);
MI_LAYOUT(mi_gen_params, resize, 276);
MI_LAYOUT(mi_gen_params, denoise_strength, 280);
MI_LAYOUT(mi_gen_params, hires_width, 288);
MI_LAYOUT(mi_gen_params, hires_height, 292);
MI_LAYOUT(mi_gen_params, hires_strength, 296);
MI_LAYOUT(mi_gen_params, hires_steps, 304);
MI_LAYOUT(mi_gen_params, _reserved3, 308);
MI_LAYOUT(mi_gen_params, distill_dir, 312);  // the 1.0.0 layout ends here (320 bytes)
MI_LAYOUT(mi_gen_params, control_adapter, 320);
MI_LAYOUT(mi_gen_params, control_lora_strength, 328);
MI_LAYOUT(mi_gen_params, control_images, 336);
MI_LAYOUT(mi_gen_params, control_images_count, 344);
MI_LAYOUT(mi_gen_params, cfg_truncation, 352);
MI_LAYOUT(mi_gen_params, step_cache, 360);
MI_LAYOUT(mi_gen_params, _reserved4, 364);
MI_LAYOUT(mi_gen_params, step_cache_threshold, 368);
static_assert(sizeof(mi_control_image) == 48);  // array element: no struct_size, frozen
MI_LAYOUT(mi_control_image, kind, 0);
MI_LAYOUT(mi_control_image, image, 8);
MI_LAYOUT(mi_control_image, image_path, 16);
MI_LAYOUT(mi_control_image, strength, 24);
MI_LAYOUT(mi_control_image, start, 32);
MI_LAYOUT(mi_control_image, end, 40);
static_assert(sizeof(mi_lora) == 16);  // array element: no struct_size, frozen
MI_LAYOUT(mi_lora, path, 0);
MI_LAYOUT(mi_lora, strength, 8);
static_assert(sizeof(mi_image) == 24);  // array-style element: no struct_size, frozen
MI_LAYOUT(mi_image, pixels, 0);
MI_LAYOUT(mi_image, width, 8);
MI_LAYOUT(mi_image, height, 12);
MI_LAYOUT(mi_image, channels, 16);
MI_LAYOUT(mi_image, _reserved0, 20);
static_assert(sizeof(mi_timings) == 144);
MI_LAYOUT(mi_timings, struct_size, 0);
MI_LAYOUT(mi_timings, total_s, 8);
MI_LAYOUT(mi_timings, load_s, 16);
MI_LAYOUT(mi_timings, encode_s, 24);
MI_LAYOUT(mi_timings, denoise_s, 32);
MI_LAYOUT(mi_timings, decode_s, 40);
MI_LAYOUT(mi_timings, index_load_s, 48);
MI_LAYOUT(mi_timings, tokenize_s, 56);
MI_LAYOUT(mi_timings, plan_s, 64);
MI_LAYOUT(mi_timings, text_encode_s, 72);
MI_LAYOUT(mi_timings, dit_load_s, 80);
MI_LAYOUT(mi_timings, refine_s, 88);
MI_LAYOUT(mi_timings, vae_load_s, 96);  // the 0.1.0 layout ends here (104 bytes)
MI_LAYOUT(mi_timings, vae_encoder_load_s, 104);
MI_LAYOUT(mi_timings, image_encode_s, 112);
MI_LAYOUT(mi_timings, image_ops_s, 120);  // the 1.0.0 layout ends here (128 bytes)
MI_LAYOUT(mi_timings, lora_load_s, 128);
MI_LAYOUT(mi_timings, lora_merge_s, 136);
static_assert(sizeof(mi_memory_stats) == 80);
MI_LAYOUT(mi_memory_stats, struct_size, 0);
MI_LAYOUT(mi_memory_stats, arena_capacity_bytes, 8);
MI_LAYOUT(mi_memory_stats, arena_high_water_bytes, 16);
MI_LAYOUT(mi_memory_stats, device_peak_bytes, 24);
MI_LAYOUT(mi_memory_stats, device_free_at_start_bytes, 32);
MI_LAYOUT(mi_memory_stats, host_private_start_bytes, 40);
MI_LAYOUT(mi_memory_stats, host_private_peak_bytes, 48);  // the 1.0.0 layout ends here (56 bytes)
MI_LAYOUT(mi_memory_stats, host_copy_bytes, 56);
MI_LAYOUT(mi_memory_stats, merge_memo_bytes, 64);
MI_LAYOUT(mi_memory_stats, dit_loads, 72);
MI_LAYOUT(mi_memory_stats, _reserved0, 76);
static_assert(sizeof(mi_dit_stream_stats) == 80);
MI_LAYOUT(mi_dit_stream_stats, struct_size, 0);
MI_LAYOUT(mi_dit_stream_stats, streamed, 8);
MI_LAYOUT(mi_dit_stream_stats, resident_blocks, 12);
MI_LAYOUT(mi_dit_stream_stats, buffers, 16);
MI_LAYOUT(mi_dit_stream_stats, stalls, 20);
MI_LAYOUT(mi_dit_stream_stats, bytes, 24);
MI_LAYOUT(mi_dit_stream_stats, copy_seconds, 32);
MI_LAYOUT(mi_dit_stream_stats, stall_seconds, 40);
MI_LAYOUT(mi_dit_stream_stats, h2d_gbps, 48);
MI_LAYOUT(mi_dit_stream_stats, hidden_fraction, 56);
MI_LAYOUT(mi_dit_stream_stats, source, 64);
static_assert(sizeof(mi_tier_info) == 40);
MI_LAYOUT(mi_tier_info, struct_size, 0);
MI_LAYOUT(mi_tier_info, requested, 8);
MI_LAYOUT(mi_tier_info, effective, 12);
MI_LAYOUT(mi_tier_info, attention, 16);
MI_LAYOUT(mi_tier_info, certified, 20);
MI_LAYOUT(mi_tier_info, dit_precision, 24);
MI_LAYOUT(mi_tier_info, text_encoder_precision, 28);  // the 1.0.0 layout ends here (32 bytes)
MI_LAYOUT(mi_tier_info, run_certified, 32);
MI_LAYOUT(mi_tier_info, _reserved0, 36);
static_assert(sizeof(mi_prompt_tokens) == 32);
MI_LAYOUT(mi_prompt_tokens, struct_size, 0);
MI_LAYOUT(mi_prompt_tokens, total_tokens, 8);
MI_LAYOUT(mi_prompt_tokens, kept_tokens, 12);
MI_LAYOUT(mi_prompt_tokens, prompt_tokens, 16);
MI_LAYOUT(mi_prompt_tokens, dropped_tokens, 20);
MI_LAYOUT(mi_prompt_tokens, max_length, 24);
MI_LAYOUT(mi_prompt_tokens, truncated, 28);
static_assert(sizeof(mi_lora_report) == 32);
MI_LAYOUT(mi_lora_report, struct_size, 0);
MI_LAYOUT(mi_lora_report, adapters, 8);
MI_LAYOUT(mi_lora_report, side_branch, 12);
MI_LAYOUT(mi_lora_report, merged, 16);
MI_LAYOUT(mi_lora_report, unknown_keys, 20);
MI_LAYOUT(mi_lora_report, warnings, 24);
MI_LAYOUT(mi_lora_report, notes, 28);
static_assert(sizeof(mi_image_info) == 80);
MI_LAYOUT(mi_image_info, struct_size, 0);
MI_LAYOUT(mi_image_info, mode, 8);
MI_LAYOUT(mi_image_info, hires_ran, 12);
MI_LAYOUT(mi_image_info, first_width, 16);
MI_LAYOUT(mi_image_info, first_height, 20);
MI_LAYOUT(mi_image_info, first_steps_scheduled, 24);
MI_LAYOUT(mi_image_info, first_start_step, 28);
MI_LAYOUT(mi_image_info, first_steps_run, 32);
MI_LAYOUT(mi_image_info, first_encoded, 36);
MI_LAYOUT(mi_image_info, hires_width, 40);
MI_LAYOUT(mi_image_info, hires_height, 44);
MI_LAYOUT(mi_image_info, hires_steps_scheduled, 48);
MI_LAYOUT(mi_image_info, hires_start_step, 52);
MI_LAYOUT(mi_image_info, hires_steps_run, 56);
MI_LAYOUT(mi_image_info, _reserved0, 60);
MI_LAYOUT(mi_image_info, first_strength, 64);
MI_LAYOUT(mi_image_info, hires_strength, 72);
static_assert(sizeof(mi_control_info) == 48);
MI_LAYOUT(mi_control_info, struct_size, 0);
MI_LAYOUT(mi_control_info, active, 8);
MI_LAYOUT(mi_control_info, kinds_count, 12);
MI_LAYOUT(mi_control_info, encodes, 16);
MI_LAYOUT(mi_control_info, lora_adapters, 20);
MI_LAYOUT(mi_control_info, lora_side_branch, 24);
MI_LAYOUT(mi_control_info, lora_merged, 28);
MI_LAYOUT(mi_control_info, lora_unknown_keys, 32);
MI_LAYOUT(mi_control_info, lora_warnings, 36);
MI_LAYOUT(mi_control_info, lora_notes, 40);
MI_LAYOUT(mi_control_info, _reserved0, 44);
static_assert(sizeof(mi_accel_info) == 32);
MI_LAYOUT(mi_accel_info, struct_size, 0);
MI_LAYOUT(mi_accel_info, cfg_truncated_steps, 8);
MI_LAYOUT(mi_accel_info, dit_calls, 12);
MI_LAYOUT(mi_accel_info, cached_calls, 16);
MI_LAYOUT(mi_accel_info, lossy, 20);  // an earlier 1.1.0 header's layout ends here (24 bytes)
MI_LAYOUT(mi_accel_info, step_cache_threshold, 24);
static_assert(sizeof(mi_release_info) == 560);
MI_LAYOUT(mi_release_info, struct_size, 0);
MI_LAYOUT(mi_release_info, seconds, 8);
MI_LAYOUT(mi_release_info, vram_released_bytes, 16);
MI_LAYOUT(mi_release_info, host_copy_bytes, 24);
MI_LAYOUT(mi_release_info, host_copy_seconds, 32);
MI_LAYOUT(mi_release_info, dit_host_copy, 40);
MI_LAYOUT(mi_release_info, te_host_copy, 44);
MI_LAYOUT(mi_release_info, note, 48);
static_assert(sizeof(mi_dit_load_info) == 32);
MI_LAYOUT(mi_dit_load_info, struct_size, 0);
MI_LAYOUT(mi_dit_load_info, source, 8);
MI_LAYOUT(mi_dit_load_info, io, 12);
MI_LAYOUT(mi_dit_load_info, bytes, 16);
MI_LAYOUT(mi_dit_load_info, seconds, 24);
static_assert(sizeof(mi_file_info) == 56);
MI_LAYOUT(mi_file_info, struct_size, 0);
MI_LAYOUT(mi_file_info, kind, 8);
MI_LAYOUT(mi_file_info, loadable, 12);
MI_LAYOUT(mi_file_info, file_bytes, 16);
MI_LAYOUT(mi_file_info, tensor_count, 24);
MI_LAYOUT(mi_file_info, precision, 28);
MI_LAYOUT(mi_file_info, tiers, 32);
MI_LAYOUT(mi_file_info, lora_rank, 36);
MI_LAYOUT(mi_file_info, lora_adapters, 40);
MI_LAYOUT(mi_file_info, lora_unknown_keys, 44);
MI_LAYOUT(mi_file_info, control_kind_count, 48);
MI_LAYOUT(mi_file_info, _reserved0, 52);
// The component API (2.1).
static_assert(sizeof(mi_memory) == 4 && sizeof(mi_dtype) == 4 && sizeof(mi_guidance) == 4 &&
                  sizeof(mi_step_output) == 4,
              "C API enums are 32-bit");
static_assert(sizeof(mi_tensor) == 56);  // no struct_size, frozen
MI_LAYOUT(mi_tensor, data, 0);
MI_LAYOUT(mi_tensor, memory, 8);
MI_LAYOUT(mi_tensor, dtype, 12);
MI_LAYOUT(mi_tensor, rank, 16);
MI_LAYOUT(mi_tensor, _reserved0, 20);
MI_LAYOUT(mi_tensor, shape, 24);
static_assert(sizeof(mi_session_params) == 48);
MI_LAYOUT(mi_session_params, struct_size, 0);
MI_LAYOUT(mi_session_params, plan_request, 8);
MI_LAYOUT(mi_session_params, progress_callback, 16);
MI_LAYOUT(mi_session_params, user_data, 24);
MI_LAYOUT(mi_session_params, max_rows, 32);
MI_LAYOUT(mi_session_params, max_latents, 36);
MI_LAYOUT(mi_session_params, control_adapter, 40);
static_assert(sizeof(mi_text_params) == 32);
MI_LAYOUT(mi_text_params, struct_size, 0);
MI_LAYOUT(mi_text_params, prompts, 8);
MI_LAYOUT(mi_text_params, prompts_count, 16);
MI_LAYOUT(mi_text_params, max_sequence_length, 24);
MI_LAYOUT(mi_text_params, _reserved0, 28);
static_assert(sizeof(mi_conditioning_info) == 24);
MI_LAYOUT(mi_conditioning_info, struct_size, 0);
MI_LAYOUT(mi_conditioning_info, tokens, 8);
MI_LAYOUT(mi_conditioning_info, text_dim, 12);
MI_LAYOUT(mi_conditioning_info, dtype, 16);
MI_LAYOUT(mi_conditioning_info, engine_encoded, 20);
static_assert(sizeof(mi_run_row) == 32);  // array element: no struct_size, frozen
MI_LAYOUT(mi_run_row, conditioning, 0);
MI_LAYOUT(mi_run_row, skip_layers, 8);
MI_LAYOUT(mi_run_row, skip_layers_count, 16);
MI_LAYOUT(mi_run_row, latent, 24);
MI_LAYOUT(mi_run_row, _reserved0, 28);
static_assert(sizeof(mi_run_params) == 208);
MI_LAYOUT(mi_run_params, struct_size, 0);
MI_LAYOUT(mi_run_params, width, 8);
MI_LAYOUT(mi_run_params, height, 12);
MI_LAYOUT(mi_run_params, rows, 16);
MI_LAYOUT(mi_run_params, rows_count, 24);
MI_LAYOUT(mi_run_params, sigmas, 32);
MI_LAYOUT(mi_run_params, sigmas_count, 40);
MI_LAYOUT(mi_run_params, scheduler, 48);
MI_LAYOUT(mi_run_params, steps, 52);
MI_LAYOUT(mi_run_params, shift, 56);
MI_LAYOUT(mi_run_params, flow_multiplier, 64);
MI_LAYOUT(mi_run_params, denoise_strength, 72);
MI_LAYOUT(mi_run_params, guidance, 80);
MI_LAYOUT(mi_run_params, output, 84);
MI_LAYOUT(mi_run_params, flags, 88);
MI_LAYOUT(mi_run_params, _reserved0, 92);
MI_LAYOUT(mi_run_params, cfg, 96);
MI_LAYOUT(mi_run_params, slg_layers, 104);
MI_LAYOUT(mi_run_params, slg_layers_count, 112);
MI_LAYOUT(mi_run_params, slg_scale, 120);
MI_LAYOUT(mi_run_params, slg_start, 128);
MI_LAYOUT(mi_run_params, slg_stop, 136);
MI_LAYOUT(mi_run_params, loras, 144);
MI_LAYOUT(mi_run_params, loras_count, 152);
MI_LAYOUT(mi_run_params, control_lora_strength, 160);
MI_LAYOUT(mi_run_params, control_images, 168);
MI_LAYOUT(mi_run_params, control_images_count, 176);
MI_LAYOUT(mi_run_params, cfg_truncation, 184);
MI_LAYOUT(mi_run_params, step_cache, 192);
MI_LAYOUT(mi_run_params, control_resize, 196);
MI_LAYOUT(mi_run_params, step_cache_threshold, 200);
static_assert(sizeof(mi_run_info) == 64);
MI_LAYOUT(mi_run_info, struct_size, 0);
MI_LAYOUT(mi_run_info, latent_channels, 8);
MI_LAYOUT(mi_run_info, latent_height, 12);
MI_LAYOUT(mi_run_info, latent_width, 16);
MI_LAYOUT(mi_run_info, steps, 20);
MI_LAYOUT(mi_run_info, outputs, 24);
MI_LAYOUT(mi_run_info, rows, 28);
MI_LAYOUT(mi_run_info, schedule_matches, 32);
MI_LAYOUT(mi_run_info, off_schedule_calls, 36);
MI_LAYOUT(mi_run_info, calls, 40);
MI_LAYOUT(mi_run_info, control_channels, 44);
MI_LAYOUT(mi_run_info, latents, 48);
MI_LAYOUT(mi_run_info, _reserved0, 52);
MI_LAYOUT(mi_run_info, staging_bytes, 56);
static_assert(sizeof(mi_step_params) == 592);
MI_LAYOUT(mi_step_params, struct_size, 0);
MI_LAYOUT(mi_step_params, sigma, 8);
MI_LAYOUT(mi_step_params, step, 12);
MI_LAYOUT(mi_step_params, rows, 16);
MI_LAYOUT(mi_step_params, flags, 20);
MI_LAYOUT(mi_step_params, cuda_stream, 24);
MI_LAYOUT(mi_step_params, control, 32);
MI_LAYOUT(mi_step_params, x, 88);
MI_LAYOUT(mi_step_params, out, 144);
static_assert(sizeof(mi_step_info) == 32);
MI_LAYOUT(mi_step_info, struct_size, 0);
MI_LAYOUT(mi_step_info, step, 8);
MI_LAYOUT(mi_step_info, rows, 12);
MI_LAYOUT(mi_step_info, cached, 16);
MI_LAYOUT(mi_step_info, on_schedule, 20);
MI_LAYOUT(mi_step_info, seconds, 24);
static_assert(MI_LATENT_CHANNELS == mi::moss_dit::kLatentChannels && MI_TEXT_DIM == mi::moss_dit::kTextDim &&
                  MI_MAX_ROWS == mi::pipeline::kMaxSessionRows && MI_MAX_CALL_TOKENS == mi::pipeline::kMaxCallTokens,
              "the component macros are the engine's constants");
#undef MI_LAYOUT

// Every size a versioned input struct has had: the 0.1 layout, then today's. A struct of an older size is read as
// its prefix over the defaults.
constexpr std::size_t kEngineParamsSizes[] = {offsetof(mi_engine_params, dit_file),
                                               offsetof(mi_engine_params, time_embedding_file),
                                               offsetof(mi_engine_params, certificate_dirs), sizeof(mi_engine_params)};
constexpr std::size_t kGenParamsSizes[] = {offsetof(mi_gen_params, scheduler), offsetof(mi_gen_params, init_image),
                                            offsetof(mi_gen_params, control_adapter), sizeof(mi_gen_params)};
// Output structs that grew: mi_timings (0.1.0: 104 bytes, 1.0.0: 128), mi_probe_info (0.2.0: 576 bytes).
constexpr std::size_t kTimingsSizes[] = {offsetof(mi_timings, vae_encoder_load_s), offsetof(mi_timings, lora_load_s),
                                         sizeof(mi_timings)};
constexpr std::size_t kProbeInfoSizes[] = {offsetof(mi_probe_info, cublaslt_source), sizeof(mi_probe_info)};
// mi_tier_info (1.0.0: 32 bytes, before run_certified).
constexpr std::size_t kTierInfoSizes[] = {offsetof(mi_tier_info, run_certified), sizeof(mi_tier_info)};
static_assert(kTierInfoSizes[0] == 32, "the 1.0.0 mi_tier_info");
// mi_memory_stats (1.0.0: 56 bytes), mi_accel_info (an earlier 1.1.0 header: 24 bytes).
constexpr std::size_t kMemoryStatsSizes[] = {offsetof(mi_memory_stats, host_copy_bytes), sizeof(mi_memory_stats)};
constexpr std::size_t kAccelInfoSizes[] = {offsetof(mi_accel_info, step_cache_threshold), sizeof(mi_accel_info)};
static_assert(kEngineParamsSizes[2] == 136 && kTimingsSizes[1] == 128 && kMemoryStatsSizes[0] == 56 &&
                  kAccelInfoSizes[0] == 24,
              "the earlier layouts");
static_assert(kEngineParamsSizes[0] == 80 && kGenParamsSizes[0] == 168 && kTimingsSizes[0] == 104, "the 0.1.0 layouts");
static_assert(kProbeInfoSizes[0] == 576, "the 0.2.0 mi_probe_info");
static_assert(kEngineParamsSizes[1] == 120 && kGenParamsSizes[1] == 240, "the first 0.2.0 layouts");
static_assert(kGenParamsSizes[2] == 320, "the 1.0.0 mi_gen_params");
// The component structs (2.1): one size each so far.
constexpr std::size_t kSessionParamsSize = sizeof(mi_session_params);
constexpr std::size_t kTextParamsSize = sizeof(mi_text_params);
constexpr std::size_t kRunParamsSize = sizeof(mi_run_params);
constexpr std::size_t kStepParamsSize = sizeof(mi_step_params);

thread_local std::string t_last_error;
// Set instead of t_last_error when copying a message failed (no memory): mi_last_error_message() must stay valid.
thread_local const char* t_last_error_static = nullptr;

// Records the message for mi_last_error_message(). Never throws: it runs inside catch handlers of noexcept code.
void set_error(const char* message) noexcept {
  try {
    t_last_error.assign(message);
    t_last_error_static = nullptr;
  } catch (...) {
    t_last_error_static = "out of memory while recording the error message";
  }
}

// Raised inside guarded() for a status with no matching exception type.
struct StatusError : std::runtime_error {
  StatusError(mi_status s, const std::string& message) : std::runtime_error(message), status(s) {}
  mi_status status;
};

mi_status fail(mi_status status, const char* message) noexcept {
  set_error(message);
  return status;
}

// Runs `body` (returning mi_status) with every exception mapped to a status and message.
template <class F>
mi_status guarded(F&& body) noexcept {
  try {
    return body();
  } catch (const StatusError& e) {
    return fail(e.status, e.what());
  } catch (const std::invalid_argument& e) {
    return fail(MI_ERR_INVALID_ARGUMENT, e.what());
  } catch (const mi::api::NotSupportedError& e) {
    return fail(MI_ERR_NOT_SUPPORTED, e.what());
  } catch (const mi::planner::BudgetError& e) {
    return fail(MI_ERR_OUT_OF_MEMORY, e.what());
  } catch (const mi::runtime::BudgetError& e) {
    return fail(MI_ERR_OUT_OF_MEMORY, e.what());
  } catch (const std::bad_alloc&) {
    return fail(MI_ERR_OUT_OF_MEMORY, "host memory allocation failed");
  } catch (const mi::api::ModelError& e) {
    return fail(MI_ERR_MODEL, e.what());
  } catch (const mi::weights::IndexError& e) {
    return fail(MI_ERR_MODEL, e.what());
  } catch (const mi::weights::FormatError& e) {
    return fail(MI_ERR_MODEL, e.what());
  } catch (const mi::weights::UploadError& e) {
    return fail(MI_ERR_MODEL, e.what());
  } catch (const mi::json::ParseError& e) {
    return fail(MI_ERR_MODEL, e.what());
  } catch (const mi::weights::IoError& e) {
    return fail(MI_ERR_IO, e.what());
  } catch (const mi::pipeline::ImageReadError& e) {
    return fail(MI_ERR_IO, e.what());
  } catch (const std::filesystem::filesystem_error& e) {
    return fail(MI_ERR_IO, e.what());
  } catch (const mi::CudaError& e) {
    return fail(MI_ERR_CUDA, e.what());
  } catch (const mi::backend::CudaLibraryError& e) {
    return fail(MI_ERR_CUDA, e.what());
  } catch (const mi::runtime::JobGate::Busy& e) {
    return fail(MI_ERR_BUSY, e.what());
  } catch (const mi::pipeline::SessionCancelled& e) {
    return fail(MI_CANCELLED, e.what());
  } catch (const mi::pipeline::StagingNotPinned& e) {
    return fail(MI_ERR_NOT_SUPPORTED, e.what());
  } catch (const mi::runtime::StagingError& e) {
    return fail(MI_ERR_OUT_OF_MEMORY, e.what());
  } catch (const std::exception& e) {
    return fail(MI_ERR_INTERNAL, e.what());
  } catch (...) {
    return fail(MI_ERR_INTERNAL, "unknown exception");
  }
}

void require(const void* p, const char* name) {
  if (p == nullptr) throw std::invalid_argument(std::string(name) + " is NULL.");
}

// Input structs: the explicit padding must be zero (it may become a field in a later ABI).
void check_reserved(std::int32_t value, const char* name) {
  if (value != 0) throw std::invalid_argument(std::string(name) + " must be 0.");
}

void check_reserved(const mi_engine_params& p) {
  check_reserved(p._reserved0, "mi_engine_params._reserved0");
  check_reserved(p._reserved1, "mi_engine_params._reserved1");
  check_reserved(p._reserved2, "mi_engine_params._reserved2");
}

void check_reserved(const mi_gen_params& p) {
  check_reserved(p._reserved0, "mi_gen_params._reserved0");
  check_reserved(p._reserved1, "mi_gen_params._reserved1");
  check_reserved(p._reserved2, "mi_gen_params._reserved2");
  check_reserved(p._reserved3, "mi_gen_params._reserved3");
  check_reserved(p._reserved4, "mi_gen_params._reserved4");
}

// Versioned structs: `known` lists every size the struct has had (ABI 1: it only grows at the end).
void check_size(std::size_t got, std::span<const std::size_t> known, const char* name) {
  std::string sizes;
  for (std::size_t i = 0; i < known.size(); ++i) {
    if (got == known[i]) return;
    sizes += std::string(i == 0 ? "" : i + 1 == known.size() ? " or " : ", ") + std::to_string(known[i]);
  }
  throw StatusError(MI_ERR_ABI_MISMATCH, std::string(name) + ".struct_size is " + std::to_string(got) +
                                             "; this library accepts " + sizes + ".");
}

void check_size(std::size_t got, std::size_t known, const char* name) { check_size(got, std::span(&known, 1), name); }

// The defaults of a struct of `size` bytes (a known size). The fields after that layout are not the caller's: they take
// the values that keep that layout's behaviour.
mi_engine_params engine_params_defaults(std::size_t /*size*/ = sizeof(mi_engine_params)) {
  mi_engine_params p{};
  p.struct_size = sizeof p;
  p.model_format = MI_MODEL_FORMAT_AUTO;
  p.mode = MI_MODE_BALANCED;
  p.vram_reserve_bytes = std::uint64_t{384} << 20;  // PipelineOptions::vram_reserve
  p.embedding_cache = 1;
  p.tier = MI_TIER_AUTO;
  p.attention = MI_ATTENTION_DEFAULT;
  return p;
}

constexpr std::int32_t kDefaultUncondSkip[] = {22};

mi_gen_params gen_params_defaults(std::size_t size = sizeof(mi_gen_params)) {
  mi_gen_params p{};
  p.struct_size = sizeof p;
  p.prompt = nullptr;
  p.negative_prompt = nullptr;
  p.width = 1024;
  p.height = 1024;
  p.steps = 50;
  p.cfg = 3.0;
  p.shift = 5.0;
  p.seed = 42;
  p.uncond_skip_layers = kDefaultUncondSkip;
  p.uncond_skip_layers_count = 1;
  p.slg_stop = 1.0;
  p.max_sequence_length = 4096;
  p.sampler = MI_SAMPLER_EULER;
  p.preview_interval = 1;
  p.scheduler = MI_SCHEDULER_MOSS;
  p.noise_layout = MI_NOISE_LAYOUT_OFFICIAL;
  p.noise_source = MI_NOISE_SOURCE_PHILOX;
  p.save_metadata = 1;
  p.flow_multiplier = 1.0;
  p.eta = 1.0;
  p.s_noise = 1.0;
  p.include_prompts = 1;
  p.denoise_strength = -1.0;  // unset
  p.hires_strength = 0.45;
  p.mask_encoding = MI_MASK_U8;
  p.resize = MI_RESIZE_CROP;
  // No control adapter, the accelerators off (step_cache_threshold 0: unset, the model's default).
  p.control_lora_strength = 1.0;
  p.cfg_truncation = 1.0;
  p.step_cache = MI_STEP_CACHE_OFF;
  // A 0.1.0 program (168 bytes) wrote PNGs without metadata: it keeps doing so.
  if (size <= offsetof(mi_gen_params, scheduler)) p.save_metadata = p.include_prompts = 0;
  // The 1.0.0 layout and later leave the context-dependent fields unset (mossinfer.h, "Unset markers"); the older
  // ones keep their explicit 50 steps and CFG 3.
  if (size >= offsetof(mi_gen_params, control_adapter)) {
    p.steps = 0;
    p.cfg = -1.0;
  }
  return p;
}

// *_init(): the first `size` bytes of the defaults (a caller built against an older header has a shorter struct).
template <class T>
void init_prefix(void* out, const T& defaults, std::size_t size) {
  T p = defaults;
  p.struct_size = size;
  std::memcpy(out, &p, size);
}

// Input structs: the caller's struct_size bytes (a known size) over the defaults of that size. Reads nothing past them
// and never views the caller's (possibly shorter) object as a full T.
template <class T>
T read_input(const void* in, std::span<const std::size_t> known, T (*defaults)(std::size_t), const char* name) {
  std::size_t size = 0;
  std::memcpy(&size, in, sizeof size);  // struct_size is the first member of every versioned struct
  check_size(size, known, name);
  T p = defaults(size);
  std::memcpy(&p, in, size);
  p.struct_size = sizeof p;
  return p;
}

// Output structs: only the caller's struct_size bytes (checked by the caller with check_size) are written.
template <class T>
void write_output(void* out, T value) {
  std::size_t size = 0;
  std::memcpy(&size, out, sizeof size);  // never viewing a shorter caller struct as a full T
  value.struct_size = size;
  std::memcpy(out, &value, std::min(size, sizeof value));
}

// Getters with no status: NULL -> 0 and the message.
template <class T, class F>
T get(const mi_result* r, T fallback, F&& f) noexcept {
  if (r == nullptr) {
    set_error("result is NULL.");
    return fallback;
  }
  return f(r->generation);
}

// A string of the result, or NULL with `out_of_range` as the message.
const char* item(const std::vector<std::string>& items, int32_t index, const char* out_of_range) noexcept {
  if (index < 0 || static_cast<std::size_t>(index) >= items.size()) {
    set_error(out_of_range);
    return nullptr;
  }
  return items[static_cast<std::size_t>(index)].c_str();
}

const std::vector<std::string>* messages(const mi::api::Generation& g, mi_message_kind kind) {
  switch (kind) {
    case MI_MESSAGE_LORA_WARNING: return &g.lora_warnings;
    case MI_MESSAGE_LORA_NOTE: return &g.lora_notes;
    case MI_MESSAGE_CONTROL_WARNING: return &g.control_warnings;
    case MI_MESSAGE_CONTROL_NOTE: return &g.control_notes;
    case MI_MESSAGE_ACCEL_WARNING: return &g.accel_warnings;
    case MI_MESSAGE_MAX_ENUM: break;
  }
  return nullptr;
}

// Truncating copy into a fixed char array (always NUL-terminated). A cut never splits a UTF-8 sequence: it backs off
// to the start of the character that does not fit.
template <std::size_t N>
void copy_string(char (&dst)[N], const std::string& s) {
  std::size_t n = s.size() < N - 1 ? s.size() : N - 1;
  if (n < s.size()) {
    while (n > 0 && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;  // s[n] continues a sequence
  }
  std::memcpy(dst, s.data(), n);
  dst[n] = '\0';
}

// mi_probe_info's spelling: "cu12", "cu13", or "" for none.
std::string flavour_string(mi::backend::Flavour f) {
  return f == mi::backend::Flavour::kNone ? std::string() : mi::backend::flavour_name(f);
}

// ---------------------------------------------------------------- components

// The component structs have one size each (2.1).
mi_session_params session_params_defaults(std::size_t /*size*/ = sizeof(mi_session_params)) {
  mi_session_params p{};
  p.struct_size = sizeof p;
  p.max_rows = 3;
  p.max_latents = 1;
  return p;
}

mi_text_params text_params_defaults(std::size_t /*size*/ = sizeof(mi_text_params)) {
  mi_text_params p{};
  p.struct_size = sizeof p;
  p.max_sequence_length = 4096;
  return p;
}

mi_run_params run_params_defaults(std::size_t /*size*/ = sizeof(mi_run_params)) {
  mi_run_params p{};
  p.struct_size = sizeof p;
  p.width = 1024;
  p.height = 1024;
  p.scheduler = MI_SCHEDULER_MOSS;
  p.steps = MI_DEFAULT_STEPS;
  p.shift = 5.0;
  p.flow_multiplier = 1.0;
  p.denoise_strength = 1.0;
  p.guidance = MI_GUIDANCE_HOST;
  p.output = MI_STEP_OUTPUT_ROWS;
  p.cfg = mi::pipeline::kDefaultRunCfg;
  p.slg_stop = 1.0;
  p.control_lora_strength = 1.0;
  p.cfg_truncation = 1.0;
  p.step_cache = MI_STEP_CACHE_OFF;
  p.control_resize = MI_RESIZE_CROP;
  return p;
}

mi_step_params step_params_defaults(std::size_t /*size*/ = sizeof(mi_step_params)) {
  mi_step_params p{};
  p.struct_size = sizeof p;
  p.step = -1;
  return p;
}

// A call's hold on the engine's session object: it stays that object until the call returns.
using SessionRef = mi::runtime::SessionSlot<mi::api::ComponentSession>::Ref;

}  // namespace

mi_result* mi::api::new_result(Generation generation) {
  auto result = std::make_unique<mi_result>();
  result->generation = std::move(generation);
  return result.release();
}

extern "C" {

const char* mi_version(void) {
#define MI_STR2(x) #x
#define MI_STR(x) MI_STR2(x)
  return MI_STR(MI_VERSION_MAJOR) "." MI_STR(MI_VERSION_MINOR) "." MI_STR(MI_VERSION_PATCH);
#undef MI_STR
#undef MI_STR2
}

int32_t mi_abi_version(void) { return MI_ABI_VERSION; }

const char* mi_status_string(mi_status status) {
  switch (status) {
    case MI_OK: return "MI_OK";
    case MI_CANCELLED: return "MI_CANCELLED";
    case MI_ERR_INVALID_ARGUMENT: return "MI_ERR_INVALID_ARGUMENT";
    case MI_ERR_ABI_MISMATCH: return "MI_ERR_ABI_MISMATCH";
    case MI_ERR_NOT_SUPPORTED: return "MI_ERR_NOT_SUPPORTED";
    case MI_ERR_BUSY: return "MI_ERR_BUSY";
    case MI_ERR_OUT_OF_MEMORY: return "MI_ERR_OUT_OF_MEMORY";
    case MI_ERR_MODEL: return "MI_ERR_MODEL";
    case MI_ERR_IO: return "MI_ERR_IO";
    case MI_ERR_CUDA: return "MI_ERR_CUDA";
    case MI_ERR_INTERNAL: return "MI_ERR_INTERNAL";
    case MI_STATUS_MAX_ENUM: break;
  }
  return "MI_UNKNOWN_STATUS";
}

const char* mi_last_error_message(void) {
  return t_last_error_static != nullptr ? t_last_error_static : t_last_error.c_str();
}

uint32_t mi_capabilities(void) {
  return MI_CAP_PREVIEW | MI_CAP_MODEL_FILES | MI_CAP_LORA | MI_CAP_TIERS | MI_CAP_SAMPLERS | MI_CAP_IMAGE_INPUT | MI_CAP_HIRES |
         MI_CAP_DISTILL | MI_CAP_SYSTEM_CUDA | MI_CAP_CONTROL | MI_CAP_ACCEL | MI_CAP_PRELOAD | MI_CAP_RELEASE |
         MI_CAP_TE_HOST_PIN | MI_CAP_PROGRESS_FRACTION | MI_CAP_INSPECT |
         MI_CAP_COMPONENTS;  // MI_CAP_CERTIFICATE_DIRS: reserved since 2.0
}

mi_status mi_probe(mi_probe_info* info) {
  return guarded([&] {
    require(info, "info");
    check_size(info->struct_size, kProbeInfoSizes, "mi_probe_info");
    mi::backend::ProbeOptions options;
    options.query_free_memory = false;
    const mi::backend::ProbeResult r = mi::backend::probe(options);
    mi_probe_info out{};
    out.driver_cuda_version = r.driver_version;
    out.runtime_cuda_version = r.build_cuda_version;
    out.driver_compatible = r.driver_ok ? 1 : 0;
    out.device_count = static_cast<int32_t>(r.devices.size());
    out.supported_device_count = r.supported_devices;
    copy_string(out.build_flavour, flavour_string(r.build_flavour));
    copy_string(out.recommended_flavour, flavour_string(r.recommended));
    copy_string(out.message, r.reason);
    if (info->struct_size > kProbeInfoSizes[0]) {  // a 0.2.0 caller does not pay for the search
      // Loads a shared library only: no CUDA call, no context.
      const mi::backend::CublasLtStatus lt = mi::backend::cublaslt_status();
      copy_string(out.cublaslt_source, lt.ok ? lt.info.source : std::string());
      out.cublaslt_version = lt.ok ? static_cast<int32_t>(lt.info.version) : 0;
      if (lt.ok && lt.info.path.size() >= sizeof(out.cublaslt_path)) {
        // A cut path would name another file: no path, and why.
        copy_string(out.cublaslt_error, "the path of the cuBLASLt copy in use (" + std::to_string(lt.info.path.size()) +
                                            " bytes) does not fit cublaslt_path: " + lt.info.path);
      } else {
        copy_string(out.cublaslt_path, lt.ok ? lt.info.path : std::string());
        copy_string(out.cublaslt_error, lt.ok ? std::string() : lt.error);
      }
    }
    write_output(info, out);
    return MI_OK;
  });
}

mi_status mi_probe_device(int32_t index, mi_device_info* info) {
  return guarded([&] {
    require(info, "info");
    check_size(info->struct_size, sizeof(mi_device_info), "mi_device_info");
    // The index first: a bad one must not cost a free-memory query (a driver context) on every device.
    const int count = mi::backend::device_count();  // the same devices mi_probe() lists, old drivers included
    if (index < 0 || index >= count) {
      throw std::invalid_argument("device index " + std::to_string(index) + " out of range (" + std::to_string(count) +
                                  " CUDA devices).");
    }
    mi::backend::ProbeOptions options;
    options.free_memory_device = index;
    const mi::backend::ProbeResult r = mi::backend::probe(options);
    if (static_cast<std::size_t>(index) >= r.devices.size()) {
      throw StatusError(MI_ERR_CUDA, "device " + std::to_string(index) + " disappeared during the probe.");
    }
    const mi::backend::DeviceInfo& d = r.devices[static_cast<std::size_t>(index)];
    mi_device_info out{};
    out.index = d.index;
    copy_string(out.name, d.name);
    copy_string(out.pci_bus_id, d.pci_bus_id);
    out.compute_major = d.cc_major;
    out.compute_minor = d.cc_minor;
    out.total_memory_bytes = d.total_bytes;
    out.free_memory_bytes = d.free_bytes;
    out.supported = d.supported ? 1 : 0;
    copy_string(out.unsupported_reason, d.reason);
    write_output(info, out);
    return MI_OK;
  });
}

mi_status mi_engine_params_init(mi_engine_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, kEngineParamsSizes, "mi_engine_params");
    init_prefix(params, engine_params_defaults(struct_size), struct_size);
    return MI_OK;
  });
}

mi_status mi_engine_create(const mi_engine_params* params, mi_engine** out_engine) {
  if (out_engine != nullptr) *out_engine = nullptr;
  return guarded([&] {
    require(params, "params");
    require(out_engine, "out_engine");
    const mi_engine_params p =
        read_input<mi_engine_params>(params, kEngineParamsSizes, engine_params_defaults, "mi_engine_params");
    check_reserved(p);
    auto engine = std::make_unique<mi_engine>();
    engine->adapter = std::make_unique<mi::api::PipelineAdapter>(p);
    *out_engine = engine.release();
    return MI_OK;
  });
}

// The teardown starts (JobGate::begin_teardown: the job in flight on another thread is cancelled, a generate too), the
// session is closed and the gate closed (end_session, then close_engine: later jobs are refused, the job in flight is
// waited for), so the free never races a job; then every call still holding the session object has returned (the
// closed gate refused it). ~PipelineAdapter and the pipeline below it tear down with CUDA's C API and ignore its errors
// (never MI_CUDA_CHECK). The thread's last error stays untouched: from a callback of the engine's running job
// (forbidden) this is a silent no-op that touches no state.
static_assert(std::is_nothrow_destructible_v<mi_engine>);
void mi_engine_free(mi_engine* engine) {
  if (engine == nullptr) return;
  try {
    if (engine->gate.on_job_thread()) return;
    engine->gate.begin_teardown();
  } catch (...) {
    return;
  }
  const auto close_session = [&] {
    try {
      const SessionRef session(engine->session, false);
      if (session) session->close();
    } catch (...) {
      // A failure freeing the session's state: the engine is freed all the same.
    }
  };
  close_session();
  try {
    engine->session.begin_freeing();
    close_session();  // one a racing mi_session_open opened before the flag
    engine->gate.close_engine();
    engine->session.wait_idle();
  } catch (...) {
    return;
  }
  delete engine;
}

mi_status mi_cancel(mi_engine* engine) {
  return guarded([&] {
    require(engine, "engine");
    engine->gate.cancel();  // between jobs a no-op: the next job's start clears it
    return MI_OK;
  });
}

mi_status mi_gen_params_init(mi_gen_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, kGenParamsSizes, "mi_gen_params");
    init_prefix(params, gen_params_defaults(struct_size), struct_size);
    return MI_OK;
  });
}

mi_status mi_gen_params_validate(const mi_gen_params* params) {
  return guarded([&] {
    require(params, "params");
    const mi_gen_params p = read_input<mi_gen_params>(params, kGenParamsSizes, gen_params_defaults, "mi_gen_params");
    check_reserved(p);
    mi::api::validate(p);
    return MI_OK;
  });
}

mi_status mi_generate(mi_engine* engine, const mi_gen_params* params, mi_result** out_result) {
  if (out_result != nullptr) *out_result = nullptr;
  return guarded([&] {
    require(engine, "engine");
    require(params, "params");
    require(out_result, "out_result");
    const mi_gen_params p = read_input<mi_gen_params>(params, kGenParamsSizes, gen_params_defaults, "mi_gen_params");
    check_reserved(p);
    const mi::runtime::JobGate::Job job = engine->gate.begin(mi::runtime::JobGate::Kind::kEngine);
    auto result = std::make_unique<mi_result>();
    result->generation = engine->adapter->generate(p, *engine->gate.cancel_flag());
    const bool cancelled = result->generation.cancelled;
    *out_result = result.release();
    return cancelled ? MI_CANCELLED : MI_OK;
  });
}

mi_status mi_engine_preload(mi_engine* engine, const mi_gen_params* params, mi_result** out_result) {
  if (out_result != nullptr) *out_result = nullptr;
  return guarded([&] {
    require(engine, "engine");
    const mi_gen_params p = params != nullptr
                                ? read_input<mi_gen_params>(params, kGenParamsSizes, gen_params_defaults, "mi_gen_params")
                                : gen_params_defaults();
    check_reserved(p);
    const mi::runtime::JobGate::Job job = engine->gate.begin(mi::runtime::JobGate::Kind::kEngine);
    auto result = std::make_unique<mi_result>();
    result->generation = engine->adapter->preload(p, *engine->gate.cancel_flag());
    const bool cancelled = result->generation.cancelled;
    if (out_result != nullptr) *out_result = result.release();
    return cancelled ? MI_CANCELLED : MI_OK;
  });
}

mi_status mi_engine_release(mi_engine* engine, int32_t keep_host_copy, mi_release_info* out) {
  return guarded([&] {
    require(engine, "engine");
    if (out != nullptr) check_size(out->struct_size, sizeof(mi_release_info), "mi_release_info");
    const mi::runtime::JobGate::Job job = engine->gate.begin(mi::runtime::JobGate::Kind::kEngine);
    bool cancelled = false;
    const mi_release_info info = engine->adapter->release(keep_host_copy != 0, *engine->gate.cancel_flag(), &cancelled);
    if (out != nullptr) write_output(out, info);
    return cancelled ? MI_CANCELLED : MI_OK;  // cancelled before it started: nothing was released
  });
}

int32_t mi_result_width(const mi_result* result) {
  return get(result, 0, [](const mi::api::Generation& g) { return static_cast<int32_t>(g.width); });
}

int32_t mi_result_height(const mi_result* result) {
  return get(result, 0, [](const mi::api::Generation& g) { return static_cast<int32_t>(g.height); });
}

const uint8_t* mi_result_rgb(const mi_result* result, size_t* out_bytes) {
  if (out_bytes != nullptr) *out_bytes = 0;
  return get(result, static_cast<const uint8_t*>(nullptr), [&](const mi::api::Generation& g) -> const uint8_t* {
    if (g.rgb.empty()) return nullptr;
    if (out_bytes != nullptr) *out_bytes = g.rgb.size();
    return g.rgb.data();
  });
}

int32_t mi_result_cancelled(const mi_result* result) {
  return get(result, 0, [](const mi::api::Generation& g) { return g.cancelled ? 1 : 0; });
}

uint64_t mi_result_seed(const mi_result* result) {
  return get(result, std::uint64_t{0}, [](const mi::api::Generation& g) { return g.seed; });
}

int32_t mi_result_steps_scheduled(const mi_result* result) {
  return get(result, 0, [](const mi::api::Generation& g) { return static_cast<int32_t>(g.steps_scheduled); });
}

int32_t mi_result_steps_run(const mi_result* result) {
  return get(result, 0, [](const mi::api::Generation& g) { return static_cast<int32_t>(g.steps_run); });
}

mi_status mi_result_timings(const mi_result* result, mi_timings* out) {
  return guarded([&] {
    require(out, "out");
    std::size_t size = 0;
    std::memcpy(&size, out, sizeof size);  // an older, shorter mi_timings is accepted
    check_size(size, kTimingsSizes, "mi_timings");
    require(result, "result");
    write_output(out, result->generation.timings);
    return MI_OK;
  });
}

mi_status mi_result_memory(const mi_result* result, mi_memory_stats* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, kMemoryStatsSizes, "mi_memory_stats");  // a 1.0.0 (56-byte) one is accepted
    require(result, "result");
    write_output(out, result->generation.memory);
    return MI_OK;
  });
}

mi_status mi_result_dit_stream(const mi_result* result, mi_dit_stream_stats* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_dit_stream_stats), "mi_dit_stream_stats");
    require(result, "result");
    write_output(out, result->generation.dit_stream);
    return MI_OK;
  });
}

mi_status mi_result_dit_load(const mi_result* result, mi_dit_load_info* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_dit_load_info), "mi_dit_load_info");
    require(result, "result");
    write_output(out, result->generation.dit_load);
    return MI_OK;
  });
}

mi_block_path mi_result_block_path(const mi_result* result) {
  return get(result, MI_BLOCK_PATH_DEFAULT, [](const mi::api::Generation& g) { return g.block_path; });
}

const char* mi_result_block_path_note(const mi_result* result) {
  return get(result, static_cast<const char*>(nullptr), [](const mi::api::Generation& g) { return g.block_path_note.c_str(); });
}

const char* mi_result_plan_explanation(const mi_result* result) {
  return get(result, static_cast<const char*>(nullptr), [](const mi::api::Generation& g) { return g.plan_explanation.c_str(); });
}

const char* mi_result_report_json(mi_result* result) {
  const char* out = nullptr;
  guarded([&] {
    require(result, "result");
    result->report = mi::api::report_json(result->generation, result->image_path);
    out = result->report.c_str();
    return MI_OK;
  });
  return out;
}

mi_status mi_save_png(mi_result* result, const char* path) {
  return guarded([&] {
    require(result, "result");
    require(path, "path");
    if (*path == '\0') throw std::invalid_argument("path is empty.");
    try {
      mi::api::write_png(result->generation, path);
    } catch (const std::invalid_argument&) {
      throw;
    } catch (const std::runtime_error& e) {
      throw StatusError(MI_ERR_IO, e.what());  // the PNG writer reports I/O failures as runtime_error
    }
    result->image_path = path;
    return MI_OK;
  });
}

mi_status mi_result_image(const mi_result* result, mi_image_info* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_image_info), "mi_image_info");
    require(result, "result");
    write_output(out, result->generation.image);
    return MI_OK;
  });
}

mi_status mi_result_tier(const mi_result* result, mi_tier_info* out) {
  return guarded([&] {
    require(out, "out");
    std::size_t size = 0;
    std::memcpy(&size, out, sizeof size);  // a 1.0.0 (32-byte) mi_tier_info is accepted
    check_size(size, kTierInfoSizes, "mi_tier_info");
    require(result, "result");
    write_output(out, result->generation.tier);
    return MI_OK;
  });
}

const char* mi_result_text(const mi_result* result, mi_result_text_kind kind) {
  return get(result, static_cast<const char*>(nullptr), [&](const mi::api::Generation& g) -> const char* {
    switch (kind) {
      case MI_TEXT_TIER_REASON: return g.tier_reason.c_str();
      case MI_TEXT_CERTIFICATE: return "";  // reserved since 2.0
      case MI_TEXT_GPU_ARCH: return g.gpu_arch.c_str();
      case MI_TEXT_DIT_SOURCE: return g.dit_source.c_str();
      case MI_TEXT_TEXT_ENCODER_SOURCE: return g.te_source.c_str();
      case MI_TEXT_VAE_SOURCE: return g.vae_source.c_str();
      case MI_TEXT_QUALITY_JSON: return g.quality_json.c_str();
      case MI_TEXT_IMAGE_NOTE: return g.image_note.c_str();
      case MI_TEXT_TIME_EMBEDDING: return g.time_embedding.c_str();
      case MI_TEXT_CONTROL_ADAPTER: return g.control_adapter.c_str();
      case MI_TEXT_CONTROL_KINDS: return g.control_kinds.c_str();
      case MI_TEXT_CONTROL_BASE_MODEL: return g.control_base_model.c_str();
      case MI_TEXT_CONTROL_ENCODE_SCHEDULE: return g.control_encode_schedule.c_str();
      case MI_TEXT_ACCELERATORS: return g.accelerators.c_str();
      case MI_TEXT_DIT_LOAD_NOTE: return g.dit_load_note.c_str();
      case MI_TEXT_TE_NOTE: return g.te_note.c_str();
      case MI_TEXT_PRELOAD_NOTE: return g.preload_note.c_str();
      case MI_TEXT_STEP_CACHE_THRESHOLD_SOURCE: return g.step_cache_threshold_source.c_str();
      case MI_TEXT_COMPONENT_NOTE: return g.component_note.c_str();
      case MI_TEXT_MAX_ENUM: break;
    }
    set_error("mi_result_text: unknown kind.");
    return nullptr;
  });
}

mi_status mi_result_control(const mi_result* result, mi_control_info* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_control_info), "mi_control_info");
    require(result, "result");
    write_output(out, result->generation.control);
    return MI_OK;
  });
}

mi_status mi_result_accel(const mi_result* result, mi_accel_info* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, kAccelInfoSizes, "mi_accel_info");  // a 24-byte one is accepted
    require(result, "result");
    write_output(out, result->generation.accel);
    return MI_OK;
  });
}

const int32_t* mi_result_accel_cached_steps(const mi_result* result, size_t* count) {
  if (count == nullptr) {
    set_error("count is NULL.");
    return nullptr;
  }
  *count = 0;
  return get(result, static_cast<const int32_t*>(nullptr), [&](const mi::api::Generation& g) -> const int32_t* {
    if (g.cached_steps.empty()) return nullptr;
    *count = g.cached_steps.size();
    return g.cached_steps.data();
  });
}

const double* mi_result_accel_step_cache_diffs(const mi_result* result, size_t* count) {
  if (count == nullptr) {
    set_error("count is NULL.");
    return nullptr;
  }
  *count = 0;
  return get(result, static_cast<const double*>(nullptr), [&](const mi::api::Generation& g) -> const double* {
    if (g.step_cache_diffs.empty()) return nullptr;
    *count = g.step_cache_diffs.size();
    return g.step_cache_diffs.data();
  });
}

mi_status mi_result_prompt_tokens(const mi_result* result, int32_t negative, mi_prompt_tokens* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_prompt_tokens), "mi_prompt_tokens");
    require(result, "result");
    if (negative != 0 && negative != 1) throw std::invalid_argument("negative must be 0 (the prompt) or 1 (the negative prompt).");
    write_output(out, result->generation.tokens[negative]);
    return MI_OK;
  });
}

int32_t mi_result_lora_count(const mi_result* result) {
  return get(result, 0, [](const mi::api::Generation& g) { return static_cast<int32_t>(g.loras.size()); });
}

mi_status mi_result_lora(const mi_result* result, int32_t index, mi_lora_report* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_lora_report), "mi_lora_report");
    require(result, "result");
    const std::vector<mi_lora_report>& loras = result->generation.loras;
    if (index < 0 || static_cast<std::size_t>(index) >= loras.size()) {
      throw std::invalid_argument("LoRA index " + std::to_string(index) + " out of range (" + std::to_string(loras.size()) +
                                  " LoRAs).");
    }
    write_output(out, loras[static_cast<std::size_t>(index)]);
    return MI_OK;
  });
}

const char* mi_result_lora_file(const mi_result* result, int32_t index) {
  return get(result, static_cast<const char*>(nullptr), [&](const mi::api::Generation& g) {
    return item(g.lora_files, index, "mi_result_lora_file: index out of range.");
  });
}

int32_t mi_result_message_count(const mi_result* result, mi_message_kind kind) {
  return get(result, 0, [&](const mi::api::Generation& g) {
    const std::vector<std::string>* m = messages(g, kind);
    if (m == nullptr) set_error("mi_result_message_count: unknown kind.");
    return m != nullptr ? static_cast<int32_t>(m->size()) : 0;
  });
}

const char* mi_result_message(const mi_result* result, mi_message_kind kind, int32_t index) {
  return get(result, static_cast<const char*>(nullptr), [&](const mi::api::Generation& g) -> const char* {
    const std::vector<std::string>* m = messages(g, kind);
    if (m == nullptr) {
      set_error("mi_result_message: unknown kind.");
      return nullptr;
    }
    return item(*m, index, "mi_result_message: index out of range.");
  });
}

mi_status mi_write_png(const char* path, const uint8_t* rgb, int32_t width, int32_t height) {
  return guarded([&] {
    require(path, "path");
    require(rgb, "rgb");
    if (*path == '\0') throw std::invalid_argument("path is empty.");
    constexpr int32_t kMaxSide = 16384;  // the image-input side limit
    if (width < 1 || height < 1 || width > kMaxSide || height > kMaxSide) {
      throw std::invalid_argument("image size " + std::to_string(width) + "x" + std::to_string(height) +
                                  ": each side must be in [1, " + std::to_string(kMaxSide) + "].");
    }
    try {
      mi::api::write_png_rgb(path, rgb, width, height);
    } catch (const std::invalid_argument&) {
      throw;
    } catch (const std::runtime_error& e) {
      throw StatusError(MI_ERR_IO, e.what());  // the PNG writer reports I/O failures as runtime_error
    }
    return MI_OK;
  });
}

void mi_result_free(mi_result* result) { delete result; }

mi_status mi_inspect_file(const char* path, mi_file_report** out_report) {
  return guarded([&] {
    require(out_report, "out_report");
    *out_report = nullptr;
    require(path, "path");
    if (*path == '\0') throw std::invalid_argument("path is empty.");
    auto report = std::make_unique<mi_file_report>();
    const std::filesystem::path file = mi::util::path_from_utf8(path);
    mi::util::check_user_path(file, "path");  // a Windows device or pipe path: refused before anything touches it
    report->inspection = mi::api::inspect_file(file);
    *out_report = report.release();
    return MI_OK;
  });
}

mi_status mi_file_report_info(const mi_file_report* report, mi_file_info* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_file_info), "mi_file_info");
    require(report, "report");
    write_output(out, report->inspection.info);
    return MI_OK;
  });
}

const char* mi_file_report_text(const mi_file_report* report, mi_file_text_kind kind) {
  if (report == nullptr) {
    set_error("report is NULL.");
    return nullptr;
  }
  const mi::api::FileInspection& r = report->inspection;
  switch (kind) {
    case MI_FILE_TEXT_NOTE: return r.note.c_str();
    case MI_FILE_TEXT_QUANTIZATION: return r.quantization.c_str();
    case MI_FILE_TEXT_BASE_MODEL: return r.base_model.c_str();
    case MI_FILE_TEXT_ADAPTER_TYPES: return r.adapter_types.c_str();
    case MI_FILE_TEXT_MAX_ENUM: break;
  }
  set_error("mi_file_report_text: unknown kind.");
  return nullptr;
}

const char* mi_file_report_control_kind(const mi_file_report* report, int32_t index) {
  if (report == nullptr) {
    set_error("report is NULL.");
    return nullptr;
  }
  return item(report->inspection.control_kinds, index, "mi_file_report_control_kind: index out of range.");
}

const char* mi_file_report_tier_reason(const mi_file_report* report, mi_tier tier) {
  if (report == nullptr) {
    set_error("report is NULL.");
    return nullptr;
  }
  if (static_cast<int32_t>(tier) < MI_TIER_AUTO || static_cast<int32_t>(tier) > MI_TIER_TURBO) {
    set_error("mi_file_report_tier_reason: unknown tier.");
    return nullptr;
  }
  return report->inspection.tier_reasons[static_cast<std::size_t>(tier)].c_str();
}

void mi_file_report_free(mi_file_report* report) { delete report; }

// ---------------------------------------------------------------- components

mi_status mi_session_params_init(mi_session_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, sizeof(mi_session_params), "mi_session_params");
    init_prefix(params, session_params_defaults(), struct_size);
    return MI_OK;
  });
}

mi_status mi_session_open(mi_engine* engine, const mi_session_params* params) {
  return guarded([&] {
    require(engine, "engine");
    mi_session_params p = params != nullptr
                              ? read_input<mi_session_params>(params, std::span(&kSessionParamsSize, 1),
                                                              session_params_defaults, "mi_session_params")
                              : session_params_defaults();
    mi_gen_params plan{};
    if (p.plan_request != nullptr) {
      plan = read_input<mi_gen_params>(p.plan_request, kGenParamsSizes, gen_params_defaults,
                                       "mi_session_params.plan_request");
      check_reserved(plan);
      p.plan_request = &plan;
    }
    engine->session.replace(
        [&] {
          if (engine->gate.session_open()) throw mi::runtime::JobGate::Busy("a session is open on this engine");
        },
        [&] { return std::make_unique<mi::api::ComponentSession>(*engine->adapter, engine->gate, p); });
    return MI_OK;
  });
}

mi_status mi_session_close(mi_engine* engine) {
  return guarded([&] {
    require(engine, "engine");
    // From a callback of the engine's running job (any job) nothing is done, even with no session open.
    if (engine->gate.on_job_thread()) {
      throw mi::runtime::JobGate::Busy("the session cannot be closed from a callback of its running call");
    }
    const SessionRef session(engine->session, false);
    if (session) session->close();
    return MI_OK;
  });
}

mi_status mi_text_params_init(mi_text_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, sizeof(mi_text_params), "mi_text_params");
    init_prefix(params, text_params_defaults(), struct_size);
    return MI_OK;
  });
}

mi_status mi_encode_text(mi_engine* engine, const mi_text_params* params, mi_conditioning** out_conditionings) {
  return guarded([&] {
    require(engine, "engine");
    require(params, "params");
    require(out_conditionings, "out_conditionings");
    const mi_text_params p =
        read_input<mi_text_params>(params, std::span(&kTextParamsSize, 1), text_params_defaults, "mi_text_params");
    const SessionRef session(engine->session);
    const std::vector<std::shared_ptr<const mi::pipeline::Conditioning>> conds = session->encode_text(p);
    std::vector<std::unique_ptr<mi_conditioning>> out;
    for (const auto& c : conds) out.push_back(std::make_unique<mi_conditioning>(mi_conditioning{c}));
    for (std::size_t i = 0; i < out.size(); ++i) out_conditionings[i] = out[i].release();
    return MI_OK;
  });
}

mi_status mi_conditioning_create(const mi_tensor* embeddings, mi_conditioning** out_conditioning) {
  if (out_conditioning != nullptr) *out_conditioning = nullptr;
  return guarded([&] {
    require(embeddings, "embeddings");
    require(out_conditioning, "out_conditioning");
    const mi::backend::ScopedCurrentContext restore;  // a device tensor's copy may make a context current
    auto c = std::make_unique<mi_conditioning>(mi_conditioning{mi::api::to_conditioning(*embeddings)});
    *out_conditioning = c.release();
    return MI_OK;
  });
}

mi_status mi_conditioning_query(const mi_conditioning* c, mi_conditioning_info* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_conditioning_info), "mi_conditioning_info");
    require(c, "conditioning");
    mi_conditioning_info info{};
    info.tokens = static_cast<int32_t>(c->c->length());
    info.text_dim = MI_TEXT_DIM;
    info.dtype = MI_DTYPE_BF16;
    info.engine_encoded = c->c->engine_encoded ? 1 : 0;
    write_output(out, info);
    return MI_OK;
  });
}

mi_status mi_conditioning_prompt_tokens(const mi_conditioning* c, mi_prompt_tokens* out) {
  return guarded([&] {
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_prompt_tokens), "mi_prompt_tokens");
    require(c, "conditioning");
    mi_prompt_tokens t{};
    if (c->c->engine_encoded) t = mi::api::to_tokens(c->c->tokens);
    write_output(out, t);
    return MI_OK;
  });
}

mi_status mi_conditioning_tensor(const mi_conditioning* c, mi_tensor* out) {
  return guarded([&] {
    require(out, "out");
    require(c, "conditioning");
    mi_tensor t{};
    t.data = const_cast<std::uint16_t*>(c->c->embedding->data.data());  // read only: the conditioning is immutable
    t.memory = MI_MEMORY_HOST;
    t.dtype = MI_DTYPE_BF16;
    t.rank = 2;
    t.shape[0] = c->c->length();
    t.shape[1] = MI_TEXT_DIM;
    *out = t;
    return MI_OK;
  });
}

void mi_conditioning_free(mi_conditioning* c) { delete c; }

mi_status mi_run_params_init(mi_run_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, sizeof(mi_run_params), "mi_run_params");
    init_prefix(params, run_params_defaults(), struct_size);
    return MI_OK;
  });
}

mi_status mi_run_begin(mi_engine* engine, const mi_run_params* params) {
  return guarded([&] {
    require(engine, "engine");
    require(params, "params");
    const mi_run_params p =
        read_input<mi_run_params>(params, std::span(&kRunParamsSize, 1), run_params_defaults, "mi_run_params");
    const SessionRef session(engine->session);
    session->begin_run(p);
    return MI_OK;
  });
}

mi_status mi_run_query(const mi_engine* engine, mi_run_info* out) {
  return guarded([&] {
    require(engine, "engine");
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_run_info), "mi_run_info");
    const SessionRef session(engine->session);
    write_output(out, session->run_info());
    return MI_OK;
  });
}

const float* mi_run_sigmas(const mi_engine* engine, size_t* count) {
  if (count == nullptr) {
    set_error("count is NULL.");
    return nullptr;
  }
  *count = 0;
  const float* out = nullptr;
  guarded([&] {
    require(engine, "engine");
    const SessionRef session(engine->session);
    const std::vector<float>& s = session->run_sigmas();
    out = s.data();
    *count = s.size();
    return MI_OK;
  });
  return out;
}

mi_status mi_run_adopt_staging(mi_engine* engine, void* base, size_t bytes) {
  return guarded([&] {
    require(engine, "engine");
    require(base, "base");
    const SessionRef session(engine->session);
    session->adopt_staging(base, bytes);
    return MI_OK;
  });
}

mi_status mi_run_staging(mi_engine* engine, mi_tensor* x, mi_tensor* outs, size_t outs_count, mi_tensor* control) {
  return guarded([&] {
    require(engine, "engine");
    if (outs == nullptr && outs_count > 0) throw std::invalid_argument("outs: NULL array with a nonzero count.");
    const SessionRef session(engine->session);
    const mi::pipeline::StagingViews v = session->staging();
    if (x != nullptr) *x = mi::api::from_tensor(v.x);
    for (std::size_t k = 0; k < outs_count && k < static_cast<std::size_t>(v.outputs); ++k) {
      outs[k] = mi::api::from_tensor(v.out[k]);
    }
    if (control != nullptr) *control = mi::api::from_tensor(v.control);
    return MI_OK;
  });
}

mi_status mi_step_params_init(mi_step_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, sizeof(mi_step_params), "mi_step_params");
    init_prefix(params, step_params_defaults(), struct_size);
    return MI_OK;
  });
}

mi_status mi_denoise_step(mi_engine* engine, const mi_step_params* params, mi_step_info* out_info) {
  return guarded([&] {
    require(engine, "engine");
    require(params, "params");
    if (out_info != nullptr) check_size(out_info->struct_size, sizeof(mi_step_info), "mi_step_info");
    const mi_step_params p =
        read_input<mi_step_params>(params, std::span(&kStepParamsSize, 1), step_params_defaults, "mi_step_params");
    const SessionRef session(engine->session);
    const mi_step_info info = session->step(p);
    if (out_info != nullptr) write_output(out_info, info);
    return MI_OK;
  });
}

mi_status mi_run_end(mi_engine* engine, mi_result** out_result) {
  if (out_result != nullptr) *out_result = nullptr;
  return guarded([&] {
    require(engine, "engine");
    if (engine->gate.on_job_thread()) {
      throw mi::runtime::JobGate::Busy("the run cannot be ended from a callback of its running call");
    }
    const SessionRef session(engine->session, false);
    if (!session) return MI_OK;
    std::optional<mi::api::Generation> g = session->end_run();
    if (g && out_result != nullptr) *out_result = mi::api::new_result(std::move(*g));
    return MI_OK;
  });
}

mi_status mi_vae_encode(mi_engine* engine, const mi_image* image, const mi_tensor* out_latents) {
  return guarded([&] {
    require(engine, "engine");
    require(image, "image");
    require(out_latents, "out_latents");
    const SessionRef session(engine->session);
    session->vae_encode(*image, *out_latents);
    return MI_OK;
  });
}

mi_status mi_vae_decode(mi_engine* engine, const mi_tensor* latents, const mi_tensor* out_rgb) {
  return guarded([&] {
    require(engine, "engine");
    require(latents, "latents");
    require(out_rgb, "out_rgb");
    const SessionRef session(engine->session);
    session->vae_decode(*latents, *out_rgb);
    return MI_OK;
  });
}

mi_status mi_schedule(mi_scheduler scheduler, int32_t steps, double shift, double flow_multiplier,
                      double denoise_strength, float* sigmas, size_t capacity, size_t* count) {
  return guarded([&] {
    require(count, "count");
    *count = 0;
    if (steps < 1 || steps > mi::pipeline::kMaxDenoiseSteps) {
      throw std::invalid_argument("Steps must be between 1 and " + std::to_string(mi::pipeline::kMaxDenoiseSteps) +
                                  ".");
    }
    const std::vector<float> s = mi::pipeline::engine_sigmas(
        mi::api::to_scheduler(scheduler), steps, mi::pipeline::FlowSampling{shift, flow_multiplier}, denoise_strength);
    *count = s.size();
    if (sigmas == nullptr) return MI_OK;
    if (capacity < s.size()) {
      throw std::invalid_argument("sigmas: capacity " + std::to_string(capacity) + " is less than the schedule's " +
                                  std::to_string(s.size()) + " values.");
    }
    std::copy(s.begin(), s.end(), sigmas);
    return MI_OK;
  });
}

mi_step_output mi_sampler_prediction(mi_sampler sampler, mi_scheduler scheduler) {
  mi_step_output out = MI_STEP_OUTPUT_MAX_ENUM;
  guarded([&] {
    const mi::pipeline::SamplerKind k = mi::api::to_sampler(sampler);
    const mi::pipeline::SchedulerKind s = mi::api::to_scheduler(scheduler);
    if (k == mi::pipeline::SamplerKind::kDataSpace) {
      throw std::invalid_argument("The dataspace sampler has no step-wise form.");
    }
    // SamplerDriver::prediction(): velocity for ab2 and for the official euler (euler on the moss schedule).
    const bool velocity = k == mi::pipeline::SamplerKind::kAb2 ||
                          (k == mi::pipeline::SamplerKind::kEuler && s == mi::pipeline::SchedulerKind::kMoss);
    out = velocity ? MI_STEP_OUTPUT_GUIDED_VELOCITY : MI_STEP_OUTPUT_GUIDED_DENOISED;
    return MI_OK;
  });
  return out;
}

mi_status mi_initial_noise(int64_t seed, mi_noise_layout layout, int32_t width, int32_t height, float* out,
                           size_t count) {
  return guarded([&] {
    require(out, "out");
    if (seed < 0 || seed >= mi::pipeline::kMaxSeed) {
      throw std::invalid_argument("Seed must be an integer from 0 to 2^53-1.");
    }
    if (width < 16 || height < 16 || width % 16 != 0 || height % 16 != 0 || width > mi::pipeline::kMaxSide ||
        height > mi::pipeline::kMaxSide) {
      throw std::invalid_argument("Width and height must be multiples of 16 from 16 to " +
                                  std::to_string(mi::pipeline::kMaxSide) + ".");
    }
    const std::size_t n = static_cast<std::size_t>(MI_LATENT_CHANNELS) * (height / 16) * (width / 16);
    if (count < n) {
      throw std::invalid_argument("out: " + std::to_string(count) + " floats, the noise has " + std::to_string(n) +
                                  ".");
    }
    const std::vector<float> noise =
        mi::pipeline::initial_noise(mi::api::to_noise_layout(layout), static_cast<std::uint64_t>(seed),
                                    MI_LATENT_CHANNELS, height / 16, width / 16);
    std::copy(noise.begin(), noise.end(), out);
    return MI_OK;
  });
}

mi_status mi_image_resize(const mi_image* image, int32_t width, int32_t height, mi_resize_mode mode, uint8_t* out,
                          size_t out_bytes) {
  return guarded([&] {
    require(image, "image");
    require(out, "out");
    require(image->pixels, "image.pixels");
    check_reserved(image->_reserved0, "image._reserved0");
    constexpr int32_t kMaxSide = 16384;  // the image-input side limit
    if (image->channels != 1 && image->channels != 3) {
      throw std::invalid_argument("image.channels must be 3 (RGB) or 1 (grayscale), not " +
                                  std::to_string(image->channels) + ".");
    }
    if (image->width < 1 || image->height < 1 || image->width > kMaxSide || image->height > kMaxSide || width < 1 ||
        height < 1 || width > kMaxSide || height > kMaxSide) {
      throw std::invalid_argument("image sizes must be 1 to " + std::to_string(kMaxSide) + " on a side.");
    }
    const mi::pipeline::ResizeMode m = mi::api::to_resize(mode);
    const std::size_t need = static_cast<std::size_t>(width) * height * image->channels;
    if (out_bytes < need) {
      throw std::invalid_argument("out: " + std::to_string(out_bytes) + " bytes, the image needs " +
                                  std::to_string(need) + ".");
    }
    mi::pipeline::Image src;
    src.width = image->width;
    src.height = image->height;
    src.channels = image->channels;
    src.pixels.assign(image->pixels,
                      image->pixels + static_cast<std::size_t>(image->width) * image->height * image->channels);
    const mi::pipeline::Image dst = mi::pipeline::resize_to(src, width, height, m);
    std::copy(dst.pixels.begin(), dst.pixels.end(), out);
    return MI_OK;
  });
}

}  // extern "C"
