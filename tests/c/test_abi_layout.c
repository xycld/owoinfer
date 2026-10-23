/* The frozen ABI 1 struct layouts of mossinfer.h (docs/c-api.md), seen from a C compiler. src/api/c_api.cpp asserts
 * the same table in C++; together they show that C and C++ consumers and the library agree. Header only: needs no
 * library and no GPU. Prints every mismatch and exits 1 if there is one. */
#include <mossinfer/mossinfer.h>

#include <stddef.h>
#include <stdio.h>

static int failures = 0;

static void check(const char* what, size_t got, size_t want) {
  if (got != want) {
    printf("MISMATCH %s: %zu, frozen layout says %zu\n", what, got, want);
    ++failures;
  }
}

#define SIZE(type, want) check("sizeof(" #type ")", sizeof(type), (want))
#define FIELD(type, field, want) check(#type "." #field, offsetof(type, field), (want))

int main(void) {
  _Static_assert(sizeof(void*) == 8, "the frozen layouts are defined for 64-bit targets");
  check("sizeof(mi_status)", sizeof(mi_status), 4);
  check("sizeof(mi_mode)", sizeof(mi_mode), 4);
  check("sizeof(mi_model_format)", sizeof(mi_model_format), 4);
  check("sizeof(mi_sampler)", sizeof(mi_sampler), 4);
  check("sizeof(mi_phase)", sizeof(mi_phase), 4);
  check("sizeof(mi_block_path)", sizeof(mi_block_path), 4);
  check("sizeof(mi_tier)", sizeof(mi_tier), 4);
  check("sizeof(mi_attention)", sizeof(mi_attention), 4);
  check("sizeof(mi_precision)", sizeof(mi_precision), 4);
  check("sizeof(mi_scheduler)", sizeof(mi_scheduler), 4);
  check("sizeof(mi_noise_layout)", sizeof(mi_noise_layout), 4);
  check("sizeof(mi_noise_source)", sizeof(mi_noise_source), 4);
  check("sizeof(mi_result_text_kind)", sizeof(mi_result_text_kind), 4);
  check("sizeof(mi_message_kind)", sizeof(mi_message_kind), 4);
  check("sizeof(mi_mask_encoding)", sizeof(mi_mask_encoding), 4);
  check("sizeof(mi_resize_mode)", sizeof(mi_resize_mode), 4);
  check("sizeof(mi_image_mode)", sizeof(mi_image_mode), 4);
  check("sizeof(mi_step_cache)", sizeof(mi_step_cache), 4);
  check("sizeof(mi_direct_io)", sizeof(mi_direct_io), 4);
  check("sizeof(mi_te_host_pin)", sizeof(mi_te_host_pin), 4);
  check("sizeof(mi_dit_load_source)", sizeof(mi_dit_load_source), 4);
  check("sizeof(mi_load_io)", sizeof(mi_load_io), 4);

  SIZE(mi_device_info, 592);
  FIELD(mi_device_info, struct_size, 0);
  FIELD(mi_device_info, index, 8);
  FIELD(mi_device_info, name, 12);
  FIELD(mi_device_info, pci_bus_id, 268);
  FIELD(mi_device_info, compute_major, 300);
  FIELD(mi_device_info, compute_minor, 304);
  FIELD(mi_device_info, _reserved0, 308);
  FIELD(mi_device_info, total_memory_bytes, 312);
  FIELD(mi_device_info, free_memory_bytes, 320);
  FIELD(mi_device_info, supported, 328);
  FIELD(mi_device_info, unsupported_reason, 332);
  FIELD(mi_device_info, _reserved1, 588);

  SIZE(mi_probe_info, 1624);
  FIELD(mi_probe_info, struct_size, 0);
  FIELD(mi_probe_info, driver_cuda_version, 8);
  FIELD(mi_probe_info, runtime_cuda_version, 12);
  FIELD(mi_probe_info, driver_compatible, 16);
  FIELD(mi_probe_info, device_count, 20);
  FIELD(mi_probe_info, supported_device_count, 24);
  FIELD(mi_probe_info, build_flavour, 28);
  FIELD(mi_probe_info, recommended_flavour, 44);
  FIELD(mi_probe_info, message, 60);
  FIELD(mi_probe_info, _reserved0, 572); /* the 0.2.0 layout ends here (576 bytes) */
  FIELD(mi_probe_info, cublaslt_source, 576);
  FIELD(mi_probe_info, cublaslt_version, 592);
  FIELD(mi_probe_info, _reserved1, 596);
  FIELD(mi_probe_info, cublaslt_path, 600);
  FIELD(mi_probe_info, cublaslt_error, 1112);

  SIZE(mi_engine_params, 160);
  FIELD(mi_engine_params, struct_size, 0);
  FIELD(mi_engine_params, model_path, 8);
  FIELD(mi_engine_params, model_format, 16);
  FIELD(mi_engine_params, mode, 20);
  FIELD(mi_engine_params, device_index, 24);
  FIELD(mi_engine_params, _reserved0, 28);
  FIELD(mi_engine_params, vram_budget_bytes, 32);
  FIELD(mi_engine_params, vram_reserve_bytes, 40);
  FIELD(mi_engine_params, host_ram_budget_bytes, 48);
  FIELD(mi_engine_params, cache_dir, 56);
  FIELD(mi_engine_params, embedding_cache, 64);
  FIELD(mi_engine_params, direct_io, 68);
  FIELD(mi_engine_params, block_path, 72);
  FIELD(mi_engine_params, _reserved1, 76); /* the 0.1 layout ends here (80 bytes) */
  FIELD(mi_engine_params, dit_file, 80);
  FIELD(mi_engine_params, text_encoder_file, 88);
  FIELD(mi_engine_params, vae_file, 96);
  FIELD(mi_engine_params, tier, 104);
  FIELD(mi_engine_params, attention, 108);
  FIELD(mi_engine_params, lora_strict, 112);
  FIELD(mi_engine_params, _reserved2, 116); /* the first 0.2.0 layout ends here (120 bytes) */
  FIELD(mi_engine_params, time_embedding_file, 120);
  FIELD(mi_engine_params, distill_dir, 128); /* the 1.0.0 layout ends here (136 bytes) */
  FIELD(mi_engine_params, certificate_dirs, 136);
  FIELD(mi_engine_params, certificate_dirs_count, 144);
  FIELD(mi_engine_params, dit_direct_io, 152);
  FIELD(mi_engine_params, te_host_pin, 156);

  SIZE(mi_progress, 40);
  FIELD(mi_progress, struct_size, 0);
  FIELD(mi_progress, phase, 8);
  FIELD(mi_progress, step, 12);
  FIELD(mi_progress, total_steps, 16);
  FIELD(mi_progress, _reserved0, 20);
  FIELD(mi_progress, step_seconds, 24);
  FIELD(mi_progress, elapsed_seconds, 32);

  SIZE(mi_gen_params, 376);
  FIELD(mi_gen_params, struct_size, 0);
  FIELD(mi_gen_params, prompt, 8);
  FIELD(mi_gen_params, negative_prompt, 16);
  FIELD(mi_gen_params, width, 24);
  FIELD(mi_gen_params, height, 28);
  FIELD(mi_gen_params, steps, 32);
  FIELD(mi_gen_params, _reserved0, 36);
  FIELD(mi_gen_params, cfg, 40);
  FIELD(mi_gen_params, shift, 48);
  FIELD(mi_gen_params, seed, 56);
  FIELD(mi_gen_params, uncond_skip_layers, 64);
  FIELD(mi_gen_params, uncond_skip_layers_count, 72);
  FIELD(mi_gen_params, slg_layers, 80);
  FIELD(mi_gen_params, slg_layers_count, 88);
  FIELD(mi_gen_params, slg_scale, 96);
  FIELD(mi_gen_params, slg_start, 104);
  FIELD(mi_gen_params, slg_stop, 112);
  FIELD(mi_gen_params, max_sequence_length, 120);
  FIELD(mi_gen_params, sampler, 124);
  FIELD(mi_gen_params, progress_callback, 128);
  FIELD(mi_gen_params, plan_callback, 136);
  FIELD(mi_gen_params, preview_callback, 144);
  FIELD(mi_gen_params, preview_interval, 152);
  FIELD(mi_gen_params, _reserved1, 156);
  FIELD(mi_gen_params, user_data, 160); /* the 0.1 layout ends here (168 bytes) */
  FIELD(mi_gen_params, scheduler, 168);
  FIELD(mi_gen_params, noise_layout, 172);
  FIELD(mi_gen_params, noise_source, 176);
  FIELD(mi_gen_params, save_metadata, 180);
  FIELD(mi_gen_params, flow_multiplier, 184);
  FIELD(mi_gen_params, eta, 192);
  FIELD(mi_gen_params, s_noise, 200);
  FIELD(mi_gen_params, dataspace_path, 208);
  FIELD(mi_gen_params, loras, 216);
  FIELD(mi_gen_params, loras_count, 224);
  FIELD(mi_gen_params, include_prompts, 232);
  FIELD(mi_gen_params, _reserved2, 236); /* the first 0.2.0 layout ends here (240 bytes) */
  FIELD(mi_gen_params, init_image, 240);
  FIELD(mi_gen_params, init_image_path, 248);
  FIELD(mi_gen_params, mask, 256);
  FIELD(mi_gen_params, mask_path, 264);
  FIELD(mi_gen_params, mask_encoding, 272);
  FIELD(mi_gen_params, resize, 276);
  FIELD(mi_gen_params, denoise_strength, 280);
  FIELD(mi_gen_params, hires_width, 288);
  FIELD(mi_gen_params, hires_height, 292);
  FIELD(mi_gen_params, hires_strength, 296);
  FIELD(mi_gen_params, hires_steps, 304);
  FIELD(mi_gen_params, _reserved3, 308);
  FIELD(mi_gen_params, distill_dir, 312); /* the 1.0.0 layout ends here (320 bytes) */
  FIELD(mi_gen_params, control_adapter, 320);
  FIELD(mi_gen_params, control_lora_strength, 328);
  FIELD(mi_gen_params, control_images, 336);
  FIELD(mi_gen_params, control_images_count, 344);
  FIELD(mi_gen_params, cfg_truncation, 352);
  FIELD(mi_gen_params, step_cache, 360);
  FIELD(mi_gen_params, _reserved4, 364);
  FIELD(mi_gen_params, step_cache_threshold, 368);

  SIZE(mi_control_image, 48); /* array element without struct_size: frozen */
  FIELD(mi_control_image, kind, 0);
  FIELD(mi_control_image, image, 8);
  FIELD(mi_control_image, image_path, 16);
  FIELD(mi_control_image, strength, 24);
  FIELD(mi_control_image, start, 32);
  FIELD(mi_control_image, end, 40);

  SIZE(mi_lora, 16); /* array element without struct_size: frozen */
  FIELD(mi_lora, path, 0);
  FIELD(mi_lora, strength, 8);

  SIZE(mi_image, 24); /* array-style element without struct_size: frozen */
  FIELD(mi_image, pixels, 0);
  FIELD(mi_image, width, 8);
  FIELD(mi_image, height, 12);
  FIELD(mi_image, channels, 16);
  FIELD(mi_image, _reserved0, 20);

  SIZE(mi_timings, 144);
  FIELD(mi_timings, struct_size, 0);
  FIELD(mi_timings, total_s, 8);
  FIELD(mi_timings, load_s, 16);
  FIELD(mi_timings, encode_s, 24);
  FIELD(mi_timings, denoise_s, 32);
  FIELD(mi_timings, decode_s, 40);
  FIELD(mi_timings, index_load_s, 48);
  FIELD(mi_timings, tokenize_s, 56);
  FIELD(mi_timings, plan_s, 64);
  FIELD(mi_timings, text_encode_s, 72);
  FIELD(mi_timings, dit_load_s, 80);
  FIELD(mi_timings, refine_s, 88);
  FIELD(mi_timings, vae_load_s, 96); /* the 0.1.0 layout ends here (104 bytes) */
  FIELD(mi_timings, vae_encoder_load_s, 104);
  FIELD(mi_timings, image_encode_s, 112);
  FIELD(mi_timings, image_ops_s, 120); /* the 1.0.0 layout ends here (128 bytes) */
  FIELD(mi_timings, lora_load_s, 128);
  FIELD(mi_timings, lora_merge_s, 136);

  SIZE(mi_memory_stats, 80);
  FIELD(mi_memory_stats, struct_size, 0);
  FIELD(mi_memory_stats, arena_capacity_bytes, 8);
  FIELD(mi_memory_stats, arena_high_water_bytes, 16);
  FIELD(mi_memory_stats, device_peak_bytes, 24);
  FIELD(mi_memory_stats, device_free_at_start_bytes, 32);
  FIELD(mi_memory_stats, host_private_start_bytes, 40);
  FIELD(mi_memory_stats, host_private_peak_bytes, 48); /* the 1.0.0 layout ends here (56 bytes) */
  FIELD(mi_memory_stats, host_copy_bytes, 56);
  FIELD(mi_memory_stats, merge_memo_bytes, 64);
  FIELD(mi_memory_stats, dit_loads, 72);
  FIELD(mi_memory_stats, _reserved0, 76);

  SIZE(mi_dit_stream_stats, 80);
  FIELD(mi_dit_stream_stats, struct_size, 0);
  FIELD(mi_dit_stream_stats, streamed, 8);
  FIELD(mi_dit_stream_stats, resident_blocks, 12);
  FIELD(mi_dit_stream_stats, buffers, 16);
  FIELD(mi_dit_stream_stats, stalls, 20);
  FIELD(mi_dit_stream_stats, bytes, 24);
  FIELD(mi_dit_stream_stats, copy_seconds, 32);
  FIELD(mi_dit_stream_stats, stall_seconds, 40);
  FIELD(mi_dit_stream_stats, h2d_gbps, 48);
  FIELD(mi_dit_stream_stats, hidden_fraction, 56);
  FIELD(mi_dit_stream_stats, source, 64);

  SIZE(mi_tier_info, 40);
  FIELD(mi_tier_info, struct_size, 0);
  FIELD(mi_tier_info, requested, 8);
  FIELD(mi_tier_info, effective, 12);
  FIELD(mi_tier_info, attention, 16);
  FIELD(mi_tier_info, certified, 20);
  FIELD(mi_tier_info, dit_precision, 24);
  FIELD(mi_tier_info, text_encoder_precision, 28); /* the 1.0.0 layout ends here (32 bytes) */
  FIELD(mi_tier_info, run_certified, 32);
  FIELD(mi_tier_info, _reserved0, 36);

  SIZE(mi_prompt_tokens, 32);
  FIELD(mi_prompt_tokens, struct_size, 0);
  FIELD(mi_prompt_tokens, total_tokens, 8);
  FIELD(mi_prompt_tokens, kept_tokens, 12);
  FIELD(mi_prompt_tokens, prompt_tokens, 16);
  FIELD(mi_prompt_tokens, dropped_tokens, 20);
  FIELD(mi_prompt_tokens, max_length, 24);
  FIELD(mi_prompt_tokens, truncated, 28);

  SIZE(mi_lora_report, 32);
  FIELD(mi_lora_report, struct_size, 0);
  FIELD(mi_lora_report, adapters, 8);
  FIELD(mi_lora_report, side_branch, 12);
  FIELD(mi_lora_report, merged, 16);
  FIELD(mi_lora_report, unknown_keys, 20);
  FIELD(mi_lora_report, warnings, 24);
  FIELD(mi_lora_report, notes, 28);

  SIZE(mi_image_info, 80);
  FIELD(mi_image_info, struct_size, 0);
  FIELD(mi_image_info, mode, 8);
  FIELD(mi_image_info, hires_ran, 12);
  FIELD(mi_image_info, first_width, 16);
  FIELD(mi_image_info, first_height, 20);
  FIELD(mi_image_info, first_steps_scheduled, 24);
  FIELD(mi_image_info, first_start_step, 28);
  FIELD(mi_image_info, first_steps_run, 32);
  FIELD(mi_image_info, first_encoded, 36);
  FIELD(mi_image_info, hires_width, 40);
  FIELD(mi_image_info, hires_height, 44);
  FIELD(mi_image_info, hires_steps_scheduled, 48);
  FIELD(mi_image_info, hires_start_step, 52);
  FIELD(mi_image_info, hires_steps_run, 56);
  FIELD(mi_image_info, _reserved0, 60);
  FIELD(mi_image_info, first_strength, 64);
  FIELD(mi_image_info, hires_strength, 72);

  SIZE(mi_control_info, 48);
  FIELD(mi_control_info, struct_size, 0);
  FIELD(mi_control_info, active, 8);
  FIELD(mi_control_info, kinds_count, 12);
  FIELD(mi_control_info, encodes, 16);
  FIELD(mi_control_info, lora_adapters, 20);
  FIELD(mi_control_info, lora_side_branch, 24);
  FIELD(mi_control_info, lora_merged, 28);
  FIELD(mi_control_info, lora_unknown_keys, 32);
  FIELD(mi_control_info, lora_warnings, 36);
  FIELD(mi_control_info, lora_notes, 40);
  FIELD(mi_control_info, _reserved0, 44);

  SIZE(mi_accel_info, 32);
  FIELD(mi_accel_info, struct_size, 0);
  FIELD(mi_accel_info, cfg_truncated_steps, 8);
  FIELD(mi_accel_info, dit_calls, 12);
  FIELD(mi_accel_info, cached_calls, 16);
  FIELD(mi_accel_info, lossy, 20); /* the 1.0.0 layout ends here (24 bytes) */
  FIELD(mi_accel_info, step_cache_threshold, 24);

  SIZE(mi_release_info, 560);
  FIELD(mi_release_info, struct_size, 0);
  FIELD(mi_release_info, seconds, 8);
  FIELD(mi_release_info, vram_released_bytes, 16);
  FIELD(mi_release_info, host_copy_bytes, 24);
  FIELD(mi_release_info, host_copy_seconds, 32);
  FIELD(mi_release_info, dit_host_copy, 40);
  FIELD(mi_release_info, te_host_copy, 44);
  FIELD(mi_release_info, note, 48);

  SIZE(mi_dit_load_info, 32);
  FIELD(mi_dit_load_info, struct_size, 0);
  FIELD(mi_dit_load_info, source, 8);
  FIELD(mi_dit_load_info, io, 12);
  FIELD(mi_dit_load_info, bytes, 16);
  FIELD(mi_dit_load_info, seconds, 24);

  printf("%s: C struct layouts %s the frozen ABI %d table\n", failures ? "FAIL" : "ok",
         failures ? "differ from" : "match", MI_ABI_VERSION);
  return failures ? 1 : 0;
}
