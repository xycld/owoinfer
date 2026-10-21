/* Minimal MossInfer C API client: one prompt -> one PNG.
 *
 *   mossinfer_example_c <bundle_dir> <out.png> [prompt] [size] [steps]
 *
 * e.g. mossinfer_example_c /models/MOSS-Image/base apple.png "a red apple on a wooden table" 512 20
 */
#include <stdio.h>
#include <stdlib.h>

#include "mossinfer/mossinfer.h"

static const char* const kPhaseNames[] = {"text encoder", "DiT load", "denoise", "decode"};

static void on_progress(const mi_progress* p, void* user_data) {
  (void)user_data;
  if (p->phase == MI_PHASE_DENOISE && p->step > 0) {
    fprintf(stderr, "\rstep %d/%d (%.2f s)", (int)p->step, (int)p->total_steps, p->step_seconds);
    if (p->step == p->total_steps) fprintf(stderr, "\n");
  } else if (p->phase >= MI_PHASE_TEXT_ENCODER && p->phase <= MI_PHASE_DECODE) {
    fprintf(stderr, "[%.2f s] %s\n", p->elapsed_seconds, kPhaseNames[p->phase]);
  }
}

static int fail(const char* what, mi_status status) {
  fprintf(stderr, "%s: %s: %s\n", what, mi_status_string(status), mi_last_error_message());
  return 1;
}

int main(int argc, char** argv) {
  mi_engine_params ep;
  mi_gen_params gp;
  mi_engine* engine = NULL;
  mi_result* result = NULL;
  mi_timings timings;
  mi_status status;
  int size = 1024;

  if (argc < 3 || argc > 6) {
    fprintf(stderr, "usage: %s <bundle_dir> <out.png> [prompt] [size] [steps]\n", argv[0]);
    return 2;
  }
  if (mi_abi_version() != MI_ABI_VERSION) {
    fprintf(stderr, "library ABI %d, header ABI %d\n", (int)mi_abi_version(), MI_ABI_VERSION);
    return 1;
  }
  fprintf(stderr, "MossInfer %s\n", mi_version());

  mi_engine_params_init(&ep, sizeof ep);
  ep.model_path = argv[1];
  status = mi_engine_create(&ep, &engine);
  if (status != MI_OK) return fail("mi_engine_create", status);

  mi_gen_params_init(&gp, sizeof gp);
  gp.prompt = argc > 3 ? argv[3] : "a red apple on a wooden table, studio lighting";
  if (argc > 4) size = atoi(argv[4]);
  gp.width = gp.height = size;
  if (argc > 5) gp.steps = atoi(argv[5]);
  gp.progress_callback = on_progress;

  status = mi_generate(engine, &gp, &result);
  if (status != MI_OK) {
    mi_result_free(result); /* set on MI_CANCELLED */
    mi_engine_free(engine);
    return fail("mi_generate", status);
  }
  status = mi_save_png(result, argv[2]);
  if (status != MI_OK) {
    mi_result_free(result);
    mi_engine_free(engine);
    return fail("mi_save_png", status);
  }
  timings.struct_size = sizeof timings;
  mi_result_timings(result, &timings);
  fprintf(stderr, "%s: %dx%d, seed %llu, %d steps, %.2f s (load %.2f, encode %.2f, denoise %.2f, decode %.2f)\n", argv[2],
          (int)mi_result_width(result), (int)mi_result_height(result), (unsigned long long)mi_result_seed(result),
          (int)mi_result_steps_run(result), timings.total_s, timings.load_s, timings.encode_s, timings.denoise_s,
          timings.decode_s);
  mi_result_free(result);
  mi_engine_free(engine);
  return 0;
}
