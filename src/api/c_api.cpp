// mossinfer.h over mi::api::PipelineAdapter. Every entry point catches every exception and turns it into an mi_status
// plus the thread-local message, so nothing C++ crosses the C boundary.
#include "mossinfer/mossinfer.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "api/pipeline_adapter.h"
#include "backend/probe.h"
#include "core/cuda_check.h"
#include "planner/planner.h"
#include "runtime/device_arena.h"
#include "util/json.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/safetensors.h"
#include "weights/upload.h"

struct mi_engine {
  std::unique_ptr<mi::api::PipelineAdapter> adapter;
  std::atomic<bool> cancel{false};
  // Guards busy and the cancel flag's reset, so a cancel lands on exactly the generation running when it is issued.
  std::mutex state;
  bool busy = false;
};

struct mi_result {
  mi::api::Generation generation;
  std::string image_path;  // last successful mi_save_png()
  std::string report;
};

namespace {

// ---------------------------------------------------------------- frozen layouts (ABI 1, docs/c-api.md)
// Every public struct has explicit padding (`_reserved*`), so the C compilers and this C++ compiler agree on it.
// tests/c/test_abi_layout.c checks the same table from C.
static_assert(sizeof(void*) == 8 && sizeof(std::size_t) == 8, "the C API layouts are defined for 64-bit targets");
static_assert(sizeof(mi_status) == 4 && sizeof(mi_mode) == 4 && sizeof(mi_model_format) == 4 &&
              sizeof(mi_sampler) == 4 && sizeof(mi_phase) == 4 && sizeof(mi_block_path) == 4, "C API enums are 32-bit");
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
static_assert(sizeof(mi_probe_info) == 576);
MI_LAYOUT(mi_probe_info, struct_size, 0);
MI_LAYOUT(mi_probe_info, driver_cuda_version, 8);
MI_LAYOUT(mi_probe_info, runtime_cuda_version, 12);
MI_LAYOUT(mi_probe_info, driver_compatible, 16);
MI_LAYOUT(mi_probe_info, device_count, 20);
MI_LAYOUT(mi_probe_info, supported_device_count, 24);
MI_LAYOUT(mi_probe_info, build_flavour, 28);
MI_LAYOUT(mi_probe_info, recommended_flavour, 44);
MI_LAYOUT(mi_probe_info, message, 60);
MI_LAYOUT(mi_probe_info, _reserved0, 572);
static_assert(sizeof(mi_engine_params) == 80);
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
MI_LAYOUT(mi_engine_params, _reserved1, 76);
static_assert(sizeof(mi_progress) == 40);
MI_LAYOUT(mi_progress, struct_size, 0);
MI_LAYOUT(mi_progress, phase, 8);
MI_LAYOUT(mi_progress, step, 12);
MI_LAYOUT(mi_progress, total_steps, 16);
MI_LAYOUT(mi_progress, _reserved0, 20);
MI_LAYOUT(mi_progress, step_seconds, 24);
MI_LAYOUT(mi_progress, elapsed_seconds, 32);
static_assert(sizeof(mi_gen_params) == 168);
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
MI_LAYOUT(mi_gen_params, user_data, 160);
static_assert(sizeof(mi_timings) == 104);
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
MI_LAYOUT(mi_timings, vae_load_s, 96);
static_assert(sizeof(mi_memory_stats) == 56);
MI_LAYOUT(mi_memory_stats, struct_size, 0);
MI_LAYOUT(mi_memory_stats, arena_capacity_bytes, 8);
MI_LAYOUT(mi_memory_stats, arena_high_water_bytes, 16);
MI_LAYOUT(mi_memory_stats, device_peak_bytes, 24);
MI_LAYOUT(mi_memory_stats, device_free_at_start_bytes, 32);
MI_LAYOUT(mi_memory_stats, host_private_start_bytes, 40);
MI_LAYOUT(mi_memory_stats, host_private_peak_bytes, 48);
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
#undef MI_LAYOUT

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
  } catch (const std::filesystem::filesystem_error& e) {
    return fail(MI_ERR_IO, e.what());
  } catch (const mi::CudaError& e) {
    return fail(MI_ERR_CUDA, e.what());
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
}

void check_reserved(const mi_gen_params& p) {
  check_reserved(p._reserved0, "mi_gen_params._reserved0");
  check_reserved(p._reserved1, "mi_gen_params._reserved1");
}

// Versioned structs: this library knows exactly one size of each (ABI 1). A later minor version accepts every size
// from the first to the current layout.
void check_size(std::size_t got, std::size_t known, const char* name) {
  if (got != known) {
    throw StatusError(MI_ERR_ABI_MISMATCH, std::string(name) + ".struct_size is " + std::to_string(got) +
                                               "; this library expects " + std::to_string(known) + ".");
  }
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

constexpr std::int32_t kDefaultUncondSkip[] = {22};

// Truncating copy into a fixed char array (always NUL-terminated).
template <std::size_t N>
void copy_string(char (&dst)[N], const std::string& s) {
  const std::size_t n = s.size() < N - 1 ? s.size() : N - 1;
  std::memcpy(dst, s.data(), n);
  dst[n] = '\0';
}

// mi_probe_info's spelling: "cu12", "cu13", or "" for none.
std::string flavour_string(mi::backend::Flavour f) {
  return f == mi::backend::Flavour::kNone ? std::string() : mi::backend::flavour_name(f);
}

}  // namespace

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

uint32_t mi_capabilities(void) { return 0; }

mi_status mi_probe(mi_probe_info* info) {
  return guarded([&] {
    require(info, "info");
    check_size(info->struct_size, sizeof(mi_probe_info), "mi_probe_info");
    mi::backend::ProbeOptions options;
    options.query_free_memory = false;
    const mi::backend::ProbeResult r = mi::backend::probe(options);
    info->_reserved0 = 0;
    info->driver_cuda_version = r.driver_version;
    info->runtime_cuda_version = r.build_cuda_version;
    info->driver_compatible = r.driver_ok ? 1 : 0;
    info->device_count = static_cast<int32_t>(r.devices.size());
    info->supported_device_count = r.supported_devices;
    copy_string(info->build_flavour, flavour_string(r.build_flavour));
    copy_string(info->recommended_flavour, flavour_string(r.recommended));
    copy_string(info->message, r.reason);
    return MI_OK;
  });
}

mi_status mi_probe_device(int32_t index, mi_device_info* info) {
  return guarded([&] {
    require(info, "info");
    check_size(info->struct_size, sizeof(mi_device_info), "mi_device_info");
    const mi::backend::ProbeResult r = mi::backend::probe();
    if (index < 0 || static_cast<std::size_t>(index) >= r.devices.size()) {
      throw std::invalid_argument("device index " + std::to_string(index) + " out of range (" +
                                  std::to_string(r.devices.size()) + " CUDA devices).");
    }
    const mi::backend::DeviceInfo& d = r.devices[static_cast<std::size_t>(index)];
    info->_reserved0 = 0;
    info->_reserved1 = 0;
    info->index = d.index;
    copy_string(info->name, d.name);
    copy_string(info->pci_bus_id, d.pci_bus_id);
    info->compute_major = d.cc_major;
    info->compute_minor = d.cc_minor;
    info->total_memory_bytes = d.total_bytes;
    info->free_memory_bytes = d.free_bytes;
    info->supported = d.supported ? 1 : 0;
    copy_string(info->unsupported_reason, d.reason);
    return MI_OK;
  });
}

mi_status mi_engine_params_init(mi_engine_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, sizeof(mi_engine_params), "mi_engine_params");
    mi_engine_params p{};
    p.struct_size = sizeof p;
    p.model_format = MI_MODEL_FORMAT_AUTO;
    p.mode = MI_MODE_BALANCED;
    p.vram_reserve_bytes = std::uint64_t{384} << 20;  // PipelineOptions::vram_reserve
    p.embedding_cache = 1;
    *params = p;
    return MI_OK;
  });
}

mi_status mi_engine_create(const mi_engine_params* params, mi_engine** out_engine) {
  if (out_engine != nullptr) *out_engine = nullptr;
  return guarded([&] {
    require(params, "params");
    require(out_engine, "out_engine");
    check_size(params->struct_size, sizeof(mi_engine_params), "mi_engine_params");
    check_reserved(*params);
    auto engine = std::make_unique<mi_engine>();
    engine->adapter = std::make_unique<mi::api::PipelineAdapter>(*params);
    *out_engine = engine.release();
    return MI_OK;
  });
}

void mi_engine_free(mi_engine* engine) {
  // Destructors free device memory; a CUDA error there has nowhere to go, and the documented contract is "never fails",
  // so the thread's last-error message is not touched either.
  try {
    delete engine;
  } catch (...) {
  }
}

mi_status mi_cancel(mi_engine* engine) {
  return guarded([&] {
    require(engine, "engine");
    const std::lock_guard<std::mutex> lock(engine->state);
    if (engine->busy) engine->cancel.store(true);
    return MI_OK;
  });
}

mi_status mi_gen_params_init(mi_gen_params* params, size_t struct_size) {
  return guarded([&] {
    require(params, "params");
    check_size(struct_size, sizeof(mi_gen_params), "mi_gen_params");
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
    *params = p;
    return MI_OK;
  });
}

mi_status mi_gen_params_validate(const mi_gen_params* params) {
  return guarded([&] {
    require(params, "params");
    check_size(params->struct_size, sizeof(mi_gen_params), "mi_gen_params");
    check_reserved(*params);
    mi::api::validate(*params);
    return MI_OK;
  });
}

mi_status mi_generate(mi_engine* engine, const mi_gen_params* params, mi_result** out_result) {
  if (out_result != nullptr) *out_result = nullptr;
  return guarded([&] {
    require(engine, "engine");
    require(params, "params");
    require(out_result, "out_result");
    check_size(params->struct_size, sizeof(mi_gen_params), "mi_gen_params");
    check_reserved(*params);
    {
      const std::lock_guard<std::mutex> lock(engine->state);
      if (engine->busy) throw StatusError(MI_ERR_BUSY, "the engine is already generating (one generation at a time).");
      engine->busy = true;
      engine->cancel.store(false);
    }
    struct Idle {
      mi_engine& e;
      ~Idle() {
        const std::lock_guard<std::mutex> lock(e.state);
        e.busy = false;
        e.cancel.store(false);
      }
    } idle{*engine};
    auto result = std::make_unique<mi_result>();
    result->generation = engine->adapter->generate(*params, engine->cancel);
    const bool cancelled = result->generation.cancelled;
    *out_result = result.release();
    return cancelled ? MI_CANCELLED : MI_OK;
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
    require(result, "result");
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_timings), "mi_timings");
    *out = result->generation.timings;
    return MI_OK;
  });
}

mi_status mi_result_memory(const mi_result* result, mi_memory_stats* out) {
  return guarded([&] {
    require(result, "result");
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_memory_stats), "mi_memory_stats");
    *out = result->generation.memory;
    return MI_OK;
  });
}

mi_status mi_result_dit_stream(const mi_result* result, mi_dit_stream_stats* out) {
  return guarded([&] {
    require(result, "result");
    require(out, "out");
    check_size(out->struct_size, sizeof(mi_dit_stream_stats), "mi_dit_stream_stats");
    *out = result->generation.dit_stream;
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

void mi_result_free(mi_result* result) { delete result; }

}  // extern "C"
