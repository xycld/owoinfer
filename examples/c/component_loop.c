/* The component API from C: the host owns the sampling loop (docs/c-api.md, component API), then checks its image
 * against mi_generate's.
 *
 *   mossinfer_example_component <bundle_dir> <out.png> [prompt] [size] [steps] [dit_file tier]
 *
 * tier: reference (default, the bundle's bf16 DiT), high or turbo, e.g.
 *   mossinfer_example_component /models/MOSS-Image/sft apple.png "a red apple" 1024 20 \
 *       /models/moss_image_sft_int8_convrot.safetensors high
 *
 * The run is mi_generate's layout (MI_GUIDANCE_MOSS: rows {cond}, {uncond, skip {22}}) with the guided velocity, so
 * the host's update is euler on the moss schedule: x += fp32(s[i+1] - s[i]) * v, one rounding per operation. Build
 * this file without floating-point contraction (CMake: -ffp-contract=off), or the update fuses into an FMA and the
 * bits differ from the engine's.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mossinfer/mossinfer.h"

static int fail(const char* what, mi_status status) {
  fprintf(stderr, "%s: %s: %s\n", what, mi_status_string(status), mi_last_error_message());
  return 1;
}

#define CHECK(call)                     \
  do {                                  \
    mi_status s_ = (call);              \
    if (s_ != MI_OK) return fail(#call, s_); \
  } while (0)

int main(int argc, char** argv) {
  const char* prompt = argc > 3 ? argv[3] : "a red apple on a wooden table, studio lighting";
  const char* prompts[2];
  const int32_t skip22[1] = {22};
  int size = 1024, steps = 20, i;
  size_t k, n, count = 0, differ = 0, rgb_bytes = 0;
  mi_engine_params ep;
  mi_session_params sp;
  mi_text_params tp;
  mi_run_params rp;
  mi_run_info info;
  mi_run_row rows[2];
  mi_step_params st;
  mi_step_info si;
  mi_gen_params gp;
  mi_timings timings;
  mi_tensor x, out, latents, image;
  mi_conditioning* conds[2] = {NULL, NULL};
  mi_engine* engine = NULL;
  mi_result* result = NULL;
  const float* sigmas;
  const uint8_t* ref;
  float* lat;
  uint8_t* rgb;
  double step_sum = 0;

  if (argc != 3 && argc != 4 && argc != 5 && argc != 6 && argc != 8) {
    fprintf(stderr, "usage: %s <bundle_dir> <out.png> [prompt] [size] [steps] [dit_file reference|high|turbo]\n", argv[0]);
    return 2;
  }
  if (argc > 4) size = atoi(argv[4]);
  if (argc > 5) steps = atoi(argv[5]);

  /* 1. The engine (an explicit tier), the session, the two texts in one encoder pass. */
  mi_engine_params_init(&ep, sizeof ep);
  ep.model_path = argv[1];
  ep.tier = MI_TIER_REFERENCE;
  if (argc == 8) {
    ep.dit_file = argv[6];
    ep.tier = strcmp(argv[7], "turbo") == 0 ? MI_TIER_TURBO : strcmp(argv[7], "high") == 0 ? MI_TIER_HIGH : MI_TIER_REFERENCE;
  }
  CHECK(mi_engine_create(&ep, &engine));
  mi_session_params_init(&sp, sizeof sp);
  CHECK(mi_session_open(engine, &sp));
  prompts[0] = prompt;
  prompts[1] = "";
  mi_text_params_init(&tp, sizeof tp);
  tp.prompts = prompts;
  tp.prompts_count = 2;
  CHECK(mi_encode_text(engine, &tp, conds));

  /* 2. A MOSS run with the guided velocity; exact sigmas only (the host passes the run's own schedule). */
  memset(rows, 0, sizeof rows);
  rows[0].conditioning = conds[0];
  rows[1].conditioning = conds[1];
  rows[1].skip_layers = skip22;
  rows[1].skip_layers_count = 1;
  mi_run_params_init(&rp, sizeof rp);
  rp.width = rp.height = size;
  rp.steps = steps;
  rp.rows = rows;
  rp.rows_count = 2;
  rp.guidance = MI_GUIDANCE_MOSS;
  rp.output = MI_STEP_OUTPUT_GUIDED_VELOCITY;
  rp.flags = MI_RUN_EXACT_SIGMA;
  CHECK(mi_run_begin(engine, &rp));
  mi_conditioning_free(conds[0]); /* the run keeps what it needs */
  mi_conditioning_free(conds[1]);
  info.struct_size = sizeof info;
  CHECK(mi_run_query(engine, &info));
  sigmas = mi_run_sigmas(engine, &count);
  CHECK(mi_run_staging(engine, &x, &out, 1, NULL));
  n = (size_t)MI_LATENT_CHANNELS * (size_t)info.latent_height * (size_t)info.latent_width;

  /* 3. The host loop in the run's pinned staging: mi_generate's noise for seed 7, then euler on the moss schedule. */
  lat = (float*)x.data;
  CHECK(mi_initial_noise(7, MI_NOISE_LAYOUT_OFFICIAL, size, size, lat, n));
  for (k = 0; k < n; ++k) {
    const float a = sigmas[0] * lat[k];
    const float b = (1.0f - sigmas[0]) * 0.0f;
    lat[k] = a + b;
  }
  mi_step_params_init(&st, sizeof st);
  st.x = x;
  st.out[0] = out;
  si.struct_size = sizeof si;
  for (i = 0; i < info.steps; ++i) {
    const float h = (float)((double)sigmas[i + 1] - (double)sigmas[i]);
    const float* v = (const float*)out.data;
    st.sigma = sigmas[i];
    CHECK(mi_denoise_step(engine, &st, &si));
    step_sum += si.seconds;
    for (k = 0; k < n; ++k) {
      const float t = h * v[k];
      lat[k] = lat[k] + t;
    }
  }

  /* 4. End the run (its report is mi_generate's without an image), decode, write the PNG. */
  latents = x; /* the staging stays valid until mi_run_end: copy the latents out first */
  latents.data = malloc(n * sizeof(float));
  rgb = (uint8_t*)malloc((size_t)size * (size_t)size * 3);
  if (latents.data == NULL || rgb == NULL) return fail("malloc", MI_ERR_OUT_OF_MEMORY);
  memcpy(latents.data, lat, n * sizeof(float));
  CHECK(mi_run_end(engine, NULL));
  memset(&image, 0, sizeof image);
  image.data = rgb;
  image.memory = MI_MEMORY_HOST;
  image.dtype = MI_DTYPE_U8;
  image.rank = 3;
  image.shape[0] = size;
  image.shape[1] = size;
  image.shape[2] = 3;
  CHECK(mi_vae_decode(engine, &latents, &image));
  CHECK(mi_write_png(argv[2], rgb, size, size));
  CHECK(mi_session_close(engine));

  /* 5. The same request through mi_generate: the same bytes, and both per-step times. */
  mi_gen_params_init(&gp, sizeof gp);
  gp.prompt = prompt;
  gp.width = gp.height = size;
  gp.steps = steps;
  gp.seed = 7;
  CHECK(mi_generate(engine, &gp, &result));
  ref = mi_result_rgb(result, &rgb_bytes);
  if (ref == NULL || rgb_bytes != (size_t)size * (size_t)size * 3) return fail("mi_result_rgb", MI_ERR_INTERNAL);
  for (k = 0; k < rgb_bytes; ++k) differ += ref[k] != rgb[k];
  timings.struct_size = sizeof timings;
  mi_result_timings(result, &timings);
  printf("%s: %dx%d, %d steps\nbit-identical: %s\nper step: component %.1f ms, mi_generate %.1f ms\n", argv[2], size,
         size, info.steps, differ == 0 ? "yes" : "no", 1e3 * step_sum / info.steps, 1e3 * timings.denoise_s / info.steps);
  mi_result_free(result);
  free(latents.data);
  free(rgb);
  mi_engine_free(engine);
  return differ == 0 ? 0 : 3;
}
