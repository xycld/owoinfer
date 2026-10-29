/* The component API of mossinfer.h (2.1) seen from a C99 program: every new symbol links, the *_init() sizes and
 * defaults, MI_ERR_ABI_MISMATCH, the pure helpers, and calls without an engine. Needs the library, no GPU. Prints every
 * failure and exits 1 if there is one. */
#include <mossinfer/mossinfer.h>

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect(int ok, const char* what) {
  if (!ok) {
    printf("FAIL %s (last error: %s)\n", what, mi_last_error_message());
    ++failures;
  }
}

typedef void (*any_fn)(void);

int main(void) {
  /* Every frozen symbol (design §7.4): taking each address makes the link fail if one is missing. */
  const any_fn symbols[] = {
      (any_fn)mi_session_params_init, (any_fn)mi_session_open,     (any_fn)mi_session_close,
      (any_fn)mi_text_params_init,    (any_fn)mi_encode_text,      (any_fn)mi_conditioning_create,
      (any_fn)mi_conditioning_query,  (any_fn)mi_conditioning_prompt_tokens, (any_fn)mi_conditioning_tensor,
      (any_fn)mi_conditioning_free,   (any_fn)mi_run_params_init,  (any_fn)mi_run_begin,
      (any_fn)mi_run_query,           (any_fn)mi_run_sigmas,       (any_fn)mi_run_adopt_staging,
      (any_fn)mi_run_staging,         (any_fn)mi_step_params_init, (any_fn)mi_denoise_step,
      (any_fn)mi_run_end,             (any_fn)mi_vae_encode,       (any_fn)mi_vae_decode,
      (any_fn)mi_schedule,            (any_fn)mi_sampler_prediction, (any_fn)mi_initial_noise,
      (any_fn)mi_image_resize};
  size_t i;
  mi_session_params sp;
  mi_text_params tp;
  mi_run_params rp;
  mi_step_params st;
  float sigmas[64];
  float noise[128], noise2[128];
  size_t count = 0;
  mi_run_info info;
  mi_result* result = NULL;

  for (i = 0; i < sizeof symbols / sizeof symbols[0]; ++i) expect(symbols[i] != NULL, "symbol linked");
  expect(sizeof symbols / sizeof symbols[0] == 25, "25 component functions");
  expect((mi_capabilities() & MI_CAP_COMPONENTS) != 0, "MI_CAP_COMPONENTS is set");

  /* The init functions: their sizes, defaults, and MI_ERR_ABI_MISMATCH for any other size. */
  expect(mi_session_params_init(&sp, sizeof sp) == MI_OK && sp.struct_size == 48 && sp.max_rows == 3 &&
             sp.max_latents == 1 && sp.plan_request == NULL,
         "mi_session_params_init");
  expect(mi_text_params_init(&tp, sizeof tp) == MI_OK && tp.struct_size == 32 && tp.max_sequence_length == 4096,
         "mi_text_params_init");
  expect(mi_run_params_init(&rp, sizeof rp) == MI_OK && rp.struct_size == 208 && rp.width == 1024 &&
             rp.steps == MI_DEFAULT_STEPS && rp.guidance == MI_GUIDANCE_HOST && rp.output == MI_STEP_OUTPUT_ROWS &&
             rp.cfg == 3.0 && rp.shift == 5.0 && rp.step_cache == MI_STEP_CACHE_OFF,
         "mi_run_params_init");
  expect(mi_step_params_init(&st, sizeof st) == MI_OK && st.struct_size == 592 && st.step == -1, "mi_step_params_init");
  expect(mi_session_params_init(&sp, 40) == MI_ERR_ABI_MISMATCH, "mi_session_params: unknown size");
  expect(mi_run_params_init(&rp, 200) == MI_ERR_ABI_MISMATCH, "mi_run_params: unknown size");
  expect(mi_step_params_init(&st, 584) == MI_ERR_ABI_MISMATCH, "mi_step_params: unknown size");
  expect(mi_text_params_init(&tp, 24) == MI_ERR_ABI_MISMATCH, "mi_text_params: unknown size");

  /* The pure helpers: the moss schedule runs from 1 to 0 (steps + 1 values), a strength crops it, and the initial
   * noise is a function of the seed. */
  expect(mi_schedule(MI_SCHEDULER_MOSS, 8, 5.0, 1.0, 1.0, NULL, 0, &count) == MI_OK && count == 9, "schedule length");
  expect(mi_schedule(MI_SCHEDULER_MOSS, 8, 5.0, 1.0, 1.0, sigmas, 64, &count) == MI_OK && sigmas[0] == 1.0f &&
             sigmas[8] == 0.0f,
         "schedule values");
  for (i = 0; i + 1 < count; ++i) expect(sigmas[i] >= sigmas[i + 1], "schedule non-increasing");
  expect(mi_schedule(MI_SCHEDULER_MOSS, 8, 5.0, 1.0, 0.5, NULL, 0, &count) == MI_OK && count == 5, "strength 0.5 crops");
  expect(mi_schedule(MI_SCHEDULER_MOSS, 8, 5.0, 1.0, 0.0, NULL, 0, &count) == MI_OK && count == 0, "strength 0: empty");
  expect(mi_sampler_prediction(MI_SAMPLER_EULER, MI_SCHEDULER_MOSS) == MI_STEP_OUTPUT_GUIDED_VELOCITY, "euler on moss");
  expect(mi_sampler_prediction(MI_SAMPLER_HEUN, MI_SCHEDULER_KARRAS) == MI_STEP_OUTPUT_GUIDED_DENOISED, "heun on karras");
  expect(mi_initial_noise(42, MI_NOISE_LAYOUT_OFFICIAL, 16, 16, noise, 128) == MI_OK, "noise");
  expect(mi_initial_noise(42, MI_NOISE_LAYOUT_OFFICIAL, 16, 16, noise2, 128) == MI_OK && memcmp(noise, noise2, sizeof noise) == 0,
         "noise is deterministic");
  expect(mi_initial_noise(43, MI_NOISE_LAYOUT_OFFICIAL, 16, 16, noise2, 128) == MI_OK && memcmp(noise, noise2, sizeof noise) != 0,
         "noise depends on the seed");

  /* Without an engine. */
  expect(mi_session_open(NULL, &sp) == MI_ERR_INVALID_ARGUMENT, "mi_session_open(NULL)");
  expect(mi_run_begin(NULL, &rp) == MI_ERR_INVALID_ARGUMENT, "mi_run_begin(NULL)");
  memset(&info, 0, sizeof info);
  info.struct_size = sizeof info;
  expect(mi_run_query(NULL, &info) == MI_ERR_INVALID_ARGUMENT, "mi_run_query(NULL)");
  expect(mi_run_sigmas(NULL, &count) == NULL && count == 0, "mi_run_sigmas(NULL)");
  expect(mi_denoise_step(NULL, &st, NULL) == MI_ERR_INVALID_ARGUMENT, "mi_denoise_step(NULL)");
  expect(mi_run_end(NULL, &result) == MI_ERR_INVALID_ARGUMENT && result == NULL, "mi_run_end(NULL)");
  mi_conditioning_free(NULL);

  printf("%s: the component API %s\n", failures ? "FAIL" : "ok", failures ? "has failures" : "links and behaves");
  return failures ? 1 : 0;
}
