/* A program built against the 0.1.0 header, seen from C: its mi_engine_params (80 bytes) and mi_gen_params (168
 * bytes) are written out by hand here, exactly as the 0.1.0 header declared them. The library must accept them, fill
 * and read only their bytes, and default the later fields to the 0.1.0 behaviour (docs/c-api.md, "Struct sizes").
 * Without arguments it needs the library but no GPU: engine creation is only taken as far as the model path check,
 * before CUDA. With `<bundle> <out.png>` it also generates from these structs and checks that the PNG has no metadata.
 * Prints every failure and exits 1 if there is one. */
#include <mossinfer/mossinfer.h>

#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct engine_params_0_1 {
  size_t struct_size;
  const char* model_path;
  mi_model_format model_format;
  mi_mode mode;
  int32_t device_index;
  int32_t _reserved0;
  uint64_t vram_budget_bytes;
  uint64_t vram_reserve_bytes;
  uint64_t host_ram_budget_bytes;
  const char* cache_dir;
  int32_t embedding_cache;
  int32_t direct_io;
  mi_block_path block_path;
  int32_t _reserved1;
} engine_params_0_1;

typedef struct gen_params_0_1 {
  size_t struct_size;
  const char* prompt;
  const char* negative_prompt;
  int32_t width;
  int32_t height;
  int32_t steps;
  int32_t _reserved0;
  double cfg;
  double shift;
  int64_t seed;
  const int32_t* uncond_skip_layers;
  size_t uncond_skip_layers_count;
  const int32_t* slg_layers;
  size_t slg_layers_count;
  double slg_scale;
  double slg_start;
  double slg_stop;
  int32_t max_sequence_length;
  mi_sampler sampler;
  mi_progress_fn progress_callback;
  mi_plan_fn plan_callback;
  mi_preview_fn preview_callback;
  int32_t preview_interval;
  int32_t _reserved1;
  void* user_data;
} gen_params_0_1;

/* The old struct followed by bytes the library must never touch. */
typedef struct {
  gen_params_0_1 p;
  unsigned char guard[128];
} guarded_gen;
typedef struct {
  engine_params_0_1 p;
  unsigned char guard[128];
} guarded_engine;

static int failures = 0;

static void expect(int ok, const char* what) {
  if (!ok) {
    printf("FAIL %s (last error: %s)\n", what, mi_last_error_message());
    ++failures;
  }
}

static int untouched(const unsigned char* guard, size_t n) {
  size_t i;
  for (i = 0; i < n; ++i) {
    if (guard[i] != 0x5a) return 0;
  }
  return 1;
}

/* 1 if the PNG at `path` has a text chunk (tEXt, iTXt or zTXt), 0 if not, -1 if it cannot be read. */
static int png_has_text(const char* path) {
  unsigned char head[8];
  int found = 0;
  FILE* f = fopen(path, "rb");
  if (f == NULL) return -1;
  if (fread(head, 1, 8, f) != 8 || memcmp(head, "\x89PNG\r\n\x1a\n", 8) != 0) {
    fclose(f);
    return -1;
  }
  while (fread(head, 1, 8, f) == 8) {
    const unsigned long n = ((unsigned long)head[0] << 24) | ((unsigned long)head[1] << 16) |
                            ((unsigned long)head[2] << 8) | head[3];
    if (memcmp(head + 4, "tEXt", 4) == 0 || memcmp(head + 4, "iTXt", 4) == 0 || memcmp(head + 4, "zTXt", 4) == 0) found = 1;
    if (memcmp(head + 4, "IEND", 4) == 0 || fseek(f, (long)n + 4, SEEK_CUR) != 0) break;
  }
  fclose(f);
  return found;
}

/* With a bundle (argv[1]) and a PNG path (argv[2]): a generation from the 0.1-layout mi_gen_params keeps the 0.1.0
 * behaviour, a PNG without metadata. The 0.1 mi_engine_params (80 bytes) has no tier field, so since 2.0 (the tier is
 * required) it creates no engine; the engine comes from today's struct. Prints the skip marker without weights or a
 * supported GPU. */
static void generate_old(const char* bundle, const char* png) {
  guarded_engine e;
  guarded_gen g;
  mi_engine_params current;
  mi_engine* engine = NULL;
  mi_result* result = NULL;
  mi_probe_info probe;
  memset(&probe, 0, sizeof probe);
  probe.struct_size = sizeof probe;
  if (mi_probe(&probe) != MI_OK || probe.supported_device_count == 0) {
    printf("MOSSINFER_TEST_SKIPPED: no supported CUDA device\n");
    return;
  }
  expect(mi_engine_params_init(&current, sizeof current) == MI_OK, "engine init (current)");
  current.model_path = bundle;
  current.tier = MI_TIER_REFERENCE;
  if (mi_engine_create(&current, &engine) != MI_OK) {
    printf("MOSSINFER_TEST_SKIPPED: no MOSS-Image bundle at %s (%s)\n", bundle, mi_last_error_message());
    return;
  }
  {
    mi_engine* old = NULL;
    memset(&e, 0x5a, sizeof e);
    expect(mi_engine_params_init((mi_engine_params*)&e.p, sizeof e.p) == MI_OK, "engine init (80)");
    e.p.model_path = bundle;
    expect(mi_engine_create((const mi_engine_params*)&e.p, &old) == MI_ERR_INVALID_ARGUMENT &&
               strstr(mi_last_error_message(), "tier is required") != NULL && old == NULL,
           "a 0.1 mi_engine_params (no tier field) is refused: the tier is required");
  }
  memset(&g, 0x5a, sizeof g);
  expect(mi_gen_params_init((mi_gen_params*)&g.p, sizeof g.p) == MI_OK, "gen init (168)");
  g.p.prompt = "a red apple on a wooden table";
  g.p.width = g.p.height = 256;
  g.p.steps = 2;
  expect(mi_generate(engine, (const mi_gen_params*)&g.p, &result) == MI_OK, "mi_generate from 0.1-layout structs");
  expect(result != NULL && mi_save_png(result, png) == MI_OK, "mi_save_png");
  expect(png_has_text(png) == 0, "a 0.1-layout generation writes a PNG without text chunks");
  mi_result_free(result);
  mi_engine_free(engine);
}

int main(int argc, char** argv) {
  guarded_gen g;
  guarded_engine e;
  mi_engine* engine = NULL;
  mi_status s;

  expect(sizeof(gen_params_0_1) == 168, "the 0.1 mi_gen_params is 168 bytes");
  expect(sizeof(engine_params_0_1) == 80, "the 0.1 mi_engine_params is 80 bytes");

  /* mi_gen_params_init fills the 168 bytes only, with the 0.1 defaults. */
  memset(&g, 0x5a, sizeof g);
  expect(mi_gen_params_init((mi_gen_params*)&g.p, sizeof g.p) == MI_OK, "mi_gen_params_init accepts 168");
  expect(untouched(g.guard, sizeof g.guard), "mi_gen_params_init writes no byte past 168");
  expect(g.p.struct_size == 168, "struct_size stays 168");
  expect(g.p.width == 1024 && g.p.steps == 50 && g.p.cfg == 3.0 && g.p.seed == 42, "0.1 defaults");
  expect(g.p.sampler == MI_SAMPLER_EULER && g.p.preview_interval == 1 && g.p._reserved1 == 0, "0.1 defaults (tail)");

  /* Validated from its 168 bytes; the guard bytes (0x5a: an unknown scheduler, nonzero _reserved2) are not read. */
  g.p.prompt = "a red apple on a wooden table";
  expect(mi_gen_params_validate((const mi_gen_params*)&g.p) == MI_OK, "mi_gen_params_validate accepts a 0.1 struct");
  g.p.width = 1000;
  s = mi_gen_params_validate((const mi_gen_params*)&g.p);
  expect(s == MI_ERR_INVALID_ARGUMENT && strstr(mi_last_error_message(), "multiples of 16") != NULL,
         "a 0.1 struct is still validated");
  g.p.width = 1024;
  g.p.struct_size = 200; /* neither 168 nor 240 */
  expect(mi_gen_params_validate((const mi_gen_params*)&g.p) == MI_ERR_ABI_MISMATCH, "unknown size rejected");
  expect(mi_gen_params_init((mi_gen_params*)&g.p, 200) == MI_ERR_ABI_MISMATCH, "init rejects an unknown size");

  /* mi_engine_params: init fills 80 bytes; create reads 80 and gets as far as the model path check (before CUDA). */
  memset(&e, 0x5a, sizeof e);
  expect(mi_engine_params_init((mi_engine_params*)&e.p, sizeof e.p) == MI_OK, "mi_engine_params_init accepts 80");
  expect(untouched(e.guard, sizeof e.guard), "mi_engine_params_init writes no byte past 80");
  expect(e.p.struct_size == 80 && e.p.mode == MI_MODE_BALANCED && e.p.embedding_cache == 1 && e.p._reserved1 == 0,
         "0.1 engine defaults");
  e.p.model_path = "/nonexistent/mossinfer-abi-compat-bundle";
  s = mi_engine_create((const mi_engine_params*)&e.p, &engine);
  expect(s == MI_ERR_INVALID_ARGUMENT && strstr(mi_last_error_message(), "not found") != NULL,
         "mi_engine_create reads a 0.1 struct (model path checked, guard bytes ignored)");
  expect(engine == NULL, "no engine on failure");
  e.p.struct_size = 96;
  expect(mi_engine_create((const mi_engine_params*)&e.p, &engine) == MI_ERR_ABI_MISMATCH, "unknown engine size rejected");

  /* An earlier 0.2.0 header's mi_gen_params (240 bytes, before image input): the same rules. */
  {
    union {
      mi_gen_params p;
      unsigned char bytes[sizeof(mi_gen_params)];
    } u;
    memset(&u, 0x5a, sizeof u);
    expect(mi_gen_params_init(&u.p, 240) == MI_OK, "mi_gen_params_init accepts 240");
    expect(untouched(u.bytes + 240, sizeof u - 240), "mi_gen_params_init writes no byte past 240");
    expect(u.p.save_metadata == 1 && u.p.include_prompts == 1, "a 240-byte struct keeps the metadata on");
    u.p.prompt = "a red apple on a wooden table";
    expect(mi_gen_params_validate(&u.p) == MI_OK, "mi_gen_params_validate accepts a 240-byte struct");
  }

  if (argc > 2) generate_old(argv[1], argv[2]);

  printf("%s: 0.1-layout structs %s\n", failures ? "FAIL" : "ok", failures ? "are mishandled" : "are accepted");
  return failures ? 1 : 0;
}
