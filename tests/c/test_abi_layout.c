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

  SIZE(mi_probe_info, 576);
  FIELD(mi_probe_info, struct_size, 0);
  FIELD(mi_probe_info, driver_cuda_version, 8);
  FIELD(mi_probe_info, runtime_cuda_version, 12);
  FIELD(mi_probe_info, driver_compatible, 16);
  FIELD(mi_probe_info, device_count, 20);
  FIELD(mi_probe_info, supported_device_count, 24);
  FIELD(mi_probe_info, build_flavour, 28);
  FIELD(mi_probe_info, recommended_flavour, 44);
  FIELD(mi_probe_info, message, 60);
  FIELD(mi_probe_info, _reserved0, 572);

  SIZE(mi_engine_params, 72);
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

  SIZE(mi_progress, 40);
  FIELD(mi_progress, struct_size, 0);
  FIELD(mi_progress, phase, 8);
  FIELD(mi_progress, step, 12);
  FIELD(mi_progress, total_steps, 16);
  FIELD(mi_progress, _reserved0, 20);
  FIELD(mi_progress, step_seconds, 24);
  FIELD(mi_progress, elapsed_seconds, 32);

  SIZE(mi_gen_params, 168);
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
  FIELD(mi_gen_params, user_data, 160);

  SIZE(mi_timings, 104);
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
  FIELD(mi_timings, vae_load_s, 96);

  SIZE(mi_memory_stats, 56);
  FIELD(mi_memory_stats, struct_size, 0);
  FIELD(mi_memory_stats, arena_capacity_bytes, 8);
  FIELD(mi_memory_stats, arena_high_water_bytes, 16);
  FIELD(mi_memory_stats, device_peak_bytes, 24);
  FIELD(mi_memory_stats, device_free_at_start_bytes, 32);
  FIELD(mi_memory_stats, host_private_start_bytes, 40);
  FIELD(mi_memory_stats, host_private_peak_bytes, 48);

  printf("%s: C struct layouts %s the frozen ABI %d table\n", failures ? "FAIL" : "ok",
         failures ? "differ from" : "match", MI_ABI_VERSION);
  return failures ? 1 : 0;
}
