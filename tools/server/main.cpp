// moss-server: the engine as a sidecar process for launchers. JSON lines over stdin/stdout (docs/server-protocol.md),
// built only on the public C API (mossinfer.h).
//
// Threads: the main thread reads and answers requests; a generation runs on one worker thread, so cancel and status
// are answered while it runs. Every protocol line goes through emit() under one mutex. stdout carries protocol lines
// only: at startup the protocol stream gets a private, non-inheritable copy of fd 1 and fd 1 is pointed at stderr (or
// at the null device when stderr is unusable), so anything else printing to stdout (the engine, a library) never
// reaches the protocol stream.
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <stdlib.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#endif

#include "mossinfer/mossinfer.h"
#include "util/json.h"

namespace {

constexpr int kProtocolVersion = 1;
constexpr std::size_t kMaxLineBytes = std::size_t{1} << 20;  // a request is a prompt and a few numbers
constexpr std::size_t kMaxJsonValues = 1 << 16;

using mi::json::Value;

// ---------------------------------------------------------------- JSON output

// Length of the well-formed UTF-8 sequence at s[i] (Unicode Table 3-7), or 0 if it is ill-formed.
std::size_t utf8_sequence_length(std::string_view s, std::size_t i) {
  const auto at = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = at(i);
  if (c < 0x80) return 1;
  std::size_t n = 0;
  unsigned char lo = 0x80, hi = 0xbf;
  if (c >= 0xc2 && c <= 0xdf) n = 2;
  else if (c >= 0xe0 && c <= 0xef) n = 3, lo = c == 0xe0 ? 0xa0 : 0x80, hi = c == 0xed ? 0x9f : 0xbf;
  else if (c >= 0xf0 && c <= 0xf4) n = 4, lo = c == 0xf0 ? 0x90 : 0x80, hi = c == 0xf4 ? 0x8f : 0xbf;
  else return 0;
  if (i + n > s.size() || at(i + 1) < lo || at(i + 1) > hi) return 0;
  for (std::size_t k = 2; k < n; ++k) {
    if (at(i + k) < 0x80 || at(i + k) > 0xbf) return 0;
  }
  return n;
}

// JSON string literal; ill-formed UTF-8 becomes U+FFFD per bad byte.
std::string str(std::string_view s) {
  std::string out = "\"";
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0x80) {
      const std::size_t n = utf8_sequence_length(s, i);
      if (n == 0) {
        out += "\\ufffd";
        ++i;
      } else {
        out.append(s, i, n);
        i += n;
      }
      continue;
    }
    ++i;
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

std::string num(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", d);
  return buf;
}

const char* tf(bool b) { return b ? "true" : "false"; }

// A JSON document from the engine (report, plan) as one line: raw newlines can only be whitespace between tokens
// (inside strings they are escaped), so dropping them keeps the document intact.
std::string one_line(const char* json) {
  std::string out;
  for (const char* p = json; *p != '\0'; ++p) {
    if (*p != '\n' && *p != '\r') out += *p;
  }
  return out;
}

// Re-serializes a request id (string, integer or null) for the response.
std::string id_json(const Value* id) {
  if (id == nullptr || id->is_null()) return "null";
  if (id->is_string()) return str(id->as_string());
  return std::to_string(id->as_int());
}

// ---------------------------------------------------------------- protocol stream

class Output {
 public:
  explicit Output(std::FILE* f) : f_(f) {}
  // One protocol line. Returns false once the reader has gone (broken pipe).
  bool emit(const std::string& line) {
    const std::lock_guard<std::mutex> lock(mu_);
    if (broken_) return false;
    if (std::fwrite(line.data(), 1, line.size(), f_) != line.size() || std::fputc('\n', f_) == EOF || std::fflush(f_) != 0) {
      broken_ = true;
    }
    return !broken_;
  }
  void log(const std::string& level, const std::string& message, const std::string& id = "") {
    emit("{\"event\":\"log\"" + (id.empty() ? std::string() : ",\"id\":" + id) + ",\"level\":" + str(level) +
         ",\"message\":" + str(message) + "}");
  }

 private:
  std::FILE* f_;
  std::mutex mu_;
  bool broken_ = false;
};

// Protocol-level failure of one request: {"error": {"code", "message"}}.
struct RequestError : std::runtime_error {
  RequestError(std::string c, const std::string& message) : std::runtime_error(message), code(std::move(c)) {}
  std::string code;
};

// "MI_ERR_OUT_OF_MEMORY" -> "out_of_memory".
std::string status_code(mi_status s) {
  std::string name = mi_status_string(s);
  const std::string_view prefix = name.rfind("MI_ERR_", 0) == 0 ? "MI_ERR_" : "MI_";
  name.erase(0, prefix.size());
  for (char& c : name) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
  return name;
}

[[noreturn]] void throw_status(mi_status s) { throw RequestError(status_code(s), mi_last_error_message()); }

std::string error_json(const std::string& code, const std::string& message) {
  return "{\"code\":" + str(code) + ",\"message\":" + str(message) + "}";
}

// ---------------------------------------------------------------- params

class Params {
 public:
  explicit Params(const Value* v) {
    if (v != nullptr && !v->is_null()) {
      if (!v->is_object()) throw RequestError("invalid_params", "params must be an object");
      obj_ = &v->as_object();
    }
  }
  // Rejects members not in `known` (typos would otherwise be silently ignored).
  void only(std::initializer_list<std::string_view> known) const {
    if (obj_ == nullptr) return;
    for (const auto& [k, v] : *obj_) {
      bool ok = false;
      for (const std::string_view n : known) ok = ok || k == n;
      if (!ok) throw RequestError("invalid_params", "unknown parameter \"" + k + "\"");
    }
  }
  const Value* find(std::string_view key) const {
    if (obj_ == nullptr) return nullptr;
    for (const auto& [k, v] : *obj_) {
      if (k == key) return v.is_null() ? nullptr : &v;
    }
    return nullptr;
  }
  // Present with a JSON null (find() treats that as absent).
  bool is_null(std::string_view key) const {
    if (obj_ == nullptr) return false;
    for (const auto& [k, v] : *obj_) {
      if (k == key) return v.is_null();
    }
    return false;
  }
  std::optional<std::string> string(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_string()) throw bad(key, "a string");
    // The C API takes NUL-terminated strings: an escaped \u0000 would silently cut the prompt or path short.
    if (v->as_string().find('\0') != std::string::npos) throw bad(key, "a string without NUL characters");
    return v->as_string();
  }
  std::optional<std::int64_t> integer(std::string_view key, std::int64_t lo, std::int64_t hi) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_int() || v->as_int() < lo || v->as_int() > hi) {
      throw bad(key, "an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    }
    return v->as_int();
  }
  std::optional<double> number(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_number()) throw bad(key, "a number");
    return v->as_double();
  }
  std::optional<bool> boolean(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_bool()) throw bad(key, "true or false");
    return v->as_bool();
  }
  // An integer array or a "22, 24" string (parameters.json accepts both).
  std::optional<std::vector<std::int32_t>> layers(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    std::vector<std::int32_t> out;
    if (v->is_array()) {
      for (const Value& e : v->as_array()) {
        if (!e.is_int() || e.as_int() < 0 || e.as_int() > 1024) throw bad(key, "an array of block indices");
        out.push_back(static_cast<std::int32_t>(e.as_int()));
      }
      return out;
    }
    if (!v->is_string()) throw bad(key, "an integer array or a comma-separated string");
    const std::string& s = v->as_string();
    if (s.find_first_not_of(" \t") == std::string::npos) return out;  // blank: none
    for (std::size_t pos = 0; pos <= s.size();) {
      const std::size_t comma = std::min(s.find(',', pos), s.size());
      const std::string item = s.substr(pos, comma - pos);
      const std::size_t first = item.find_first_not_of(" \t");
      const std::string digits = first == std::string::npos ? "" : item.substr(first, item.find_last_not_of(" \t") - first + 1);
      if (digits.empty() || digits.size() > 4 || digits.find_first_not_of("0123456789") != std::string::npos) {
        throw bad(key, "comma-separated integer block indices, e.g. \"22, 24\"");
      }
      out.push_back(static_cast<std::int32_t>(std::stoi(digits)));
      pos = comma + 1;
    }
    return out;
  }

 private:
  static RequestError bad(std::string_view key, const std::string& what) {
    return RequestError("invalid_params", "\"" + std::string(key) + "\" must be " + what);
  }
  const mi::json::Object* obj_ = nullptr;
};

const char* phase_name(mi_phase p) {
  switch (p) {
    case MI_PHASE_TEXT_ENCODER: return "text_encoder";
    case MI_PHASE_DIT_LOAD: return "dit_load";
    case MI_PHASE_DENOISE: return "denoise";
    case MI_PHASE_DECODE: return "decode";
    case MI_PHASE_MAX_ENUM: break;
  }
  return "unknown";
}

const char* mode_name(mi_mode m) {
  switch (m) {
    case MI_MODE_MIN_MEMORY: return "min_memory";
    case MI_MODE_BALANCED: return "balanced";
    case MI_MODE_MAX_SPEED: return "max_speed";
    case MI_MODE_MAX_ENUM: break;
  }
  return "unknown";
}

// ---------------------------------------------------------------- server

class Server {
 public:
  explicit Server(Output& out) : out_(out) {}
  ~Server() { stop(); }

  // Returns false when the server should exit (shutdown).
  bool handle_line(const std::string& line) {
    std::string id = "null";
    try {
      for (std::size_t i = 0; i < line.size();) {
        const std::size_t n = utf8_sequence_length(line, i);
        if (n == 0) throw RequestError("parse_error", "request is not valid UTF-8 (byte " + std::to_string(i) + ")");
        i += n;
      }
      Value doc;
      try {
        mi::json::ParseLimits limits;
        limits.max_values = kMaxJsonValues;
        doc = mi::json::parse(line, limits);
      } catch (const mi::json::ParseError& e) {
        throw RequestError("parse_error", std::string("malformed JSON: ") + e.what());
      }
      if (!doc.is_object()) throw RequestError("invalid_request", "a request must be a JSON object");
      const Value* idv = doc.find("id");
      if (idv != nullptr && !idv->is_null() && !idv->is_string() && !idv->is_int()) {
        throw RequestError("invalid_request", "\"id\" must be a string, an integer or null");
      }
      id = id_json(idv);
      const Value* method = doc.find("method");
      if (method == nullptr || !method->is_string()) throw RequestError("invalid_request", "\"method\" must be a string");
      for (const auto& [k, v] : doc.as_object()) {
        if (k != "id" && k != "method" && k != "params") throw RequestError("invalid_request", "unknown member \"" + k + "\"");
      }
      const Params params(doc.find("params"));
      const std::string& m = method->as_string();
      std::string result;
      bool keep_running = true;
      if (m == "hello") result = hello(params);
      else if (m == "load") result = load(params, id);
      else if (m == "generate") result = generate(params, id);
      else if (m == "cancel") result = cancel(params);
      else if (m == "status") result = status(params);
      else if (m == "unload") result = unload(params);
      else if (m == "shutdown") {
        params.only({});
        stop();
        result = "{}";
        keep_running = false;
      } else {
        throw RequestError("method_not_found", "unknown method \"" + m + "\"");
      }
      out_.emit("{\"id\":" + id + ",\"result\":" + result + "}");
      return keep_running;
    } catch (const RequestError& e) {
      out_.emit("{\"id\":" + id + ",\"error\":" + error_json(e.code, e.what()) + "}");
    } catch (const std::exception& e) {
      out_.emit("{\"id\":" + id + ",\"error\":" + error_json("internal", e.what()) + "}");
    }
    return true;
  }

  // Cancels a running generation, waits for it and releases the engine.
  void stop() {
    request_cancel();
    join_worker();
    mi_engine_free(engine_);
    engine_ = nullptr;
  }

  void reject_long_line() {
    out_.emit("{\"id\":null,\"error\":" +
              error_json("invalid_request", "request line longer than " + std::to_string(kMaxLineBytes) + " bytes") + "}");
  }

 private:
  // State shared with the worker thread.
  struct Job {
    std::string id;  // JSON of the request id
    std::atomic<bool> cancel{false};
    std::atomic<int> phase{-1};
    std::atomic<int> step{0}, total_steps{0};
    std::atomic<double> elapsed{0};
  };

  bool generating() const { return running_.load(); }

  std::string hello(const Params& p) {
    p.only({"client", "protocol_version", "devices"});
    if (const auto v = p.integer("protocol_version", 0, std::numeric_limits<std::int32_t>::max()); v && *v != kProtocolVersion) {
      throw RequestError("unsupported_protocol", "this server speaks protocol version " + std::to_string(kProtocolVersion));
    }
    mi_probe_info probe{};
    probe.struct_size = sizeof probe;
    if (const mi_status s = mi_probe(&probe); s != MI_OK) throw_status(s);
    const bool with_devices = p.boolean("devices").value_or(false);
    std::string devices;
    for (int i = 0; with_devices && i < probe.device_count; ++i) {
      mi_device_info d{};
      d.struct_size = sizeof d;
      if (const mi_status s = mi_probe_device(i, &d); s != MI_OK) throw_status(s);
      devices += std::string(i ? "," : "") + "{\"index\":" + std::to_string(d.index) + ",\"name\":" + str(d.name) +
                 ",\"pci_bus_id\":" + str(d.pci_bus_id) + ",\"compute_capability\":\"" + std::to_string(d.compute_major) +
                 "." + std::to_string(d.compute_minor) + "\",\"total_memory_bytes\":" + std::to_string(d.total_memory_bytes) +
                 ",\"free_memory_bytes\":" + std::to_string(d.free_memory_bytes) + ",\"supported\":" + tf(d.supported) +
                 ",\"unsupported_reason\":" + str(d.unsupported_reason) + "}";
    }
    const std::uint32_t caps = mi_capabilities();
    return "{\"protocol_version\":" + std::to_string(kProtocolVersion) + ",\"server\":\"moss-server\",\"version\":" +
           str(mi_version()) + ",\"abi_version\":" + std::to_string(mi_abi_version()) +
           ",\"capabilities\":{\"preview\":" + tf(caps & MI_CAP_PREVIEW) + ",\"comfyui_models\":" +
           tf(caps & MI_CAP_COMFYUI_MODELS) + "},\"probe\":{\"driver_cuda_version\":" +
           std::to_string(probe.driver_cuda_version) + ",\"runtime_cuda_version\":" + std::to_string(probe.runtime_cuda_version) +
           ",\"driver_compatible\":" + tf(probe.driver_compatible) + ",\"device_count\":" + std::to_string(probe.device_count) +
           ",\"supported_device_count\":" + std::to_string(probe.supported_device_count) +
           ",\"build_flavour\":" + str(probe.build_flavour) + ",\"recommended_flavour\":" + str(probe.recommended_flavour) +
           ",\"message\":" + str(probe.message) + (with_devices ? ",\"devices\":[" + devices + "]" : std::string()) + "}}";
  }

  std::string load(const Params& p, const std::string& id) {
    p.only({"model", "mode", "device", "vram_budget_bytes", "vram_reserve_bytes", "host_ram_budget_bytes", "cache_dir",
            "embedding_cache", "direct_io", "block_path"});
    if (generating()) throw RequestError("busy", "a generation is running; cancel it or wait for its done event");
    mi_engine_params e;
    if (const mi_status s = mi_engine_params_init(&e, sizeof e); s != MI_OK) throw_status(s);
    const std::optional<std::string> model = p.string("model");
    if (!model) throw RequestError("invalid_params", "\"model\" (bundle directory) is required");
    e.model_path = model->c_str();
    if (const auto mode = p.string("mode")) {
      if (*mode == "min_memory") e.mode = MI_MODE_MIN_MEMORY;
      else if (*mode == "balanced") e.mode = MI_MODE_BALANCED;
      else if (*mode == "max_speed") e.mode = MI_MODE_MAX_SPEED;
      else throw RequestError("invalid_params", "\"mode\" must be \"min_memory\", \"balanced\" or \"max_speed\"");
    }
    constexpr std::int64_t kMaxBytes = std::numeric_limits<std::int64_t>::max();
    if (const auto v = p.integer("device", 0, 1024)) e.device_index = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("vram_budget_bytes", 0, kMaxBytes)) e.vram_budget_bytes = static_cast<std::uint64_t>(*v);
    if (const auto v = p.integer("vram_reserve_bytes", 0, kMaxBytes)) e.vram_reserve_bytes = static_cast<std::uint64_t>(*v);
    if (const auto v = p.integer("host_ram_budget_bytes", 0, kMaxBytes)) e.host_ram_budget_bytes = static_cast<std::uint64_t>(*v);
    const std::optional<std::string> cache_dir = p.string("cache_dir");
    if (cache_dir) e.cache_dir = cache_dir->c_str();
    if (const auto v = p.boolean("embedding_cache")) e.embedding_cache = *v ? 1 : 0;
    if (const auto v = p.boolean("direct_io")) e.direct_io = *v ? 1 : 0;
    if (const auto path = p.string("block_path")) {
      if (*path == "default") e.block_path = MI_BLOCK_PATH_DEFAULT;
      else if (*path == "op_by_op") e.block_path = MI_BLOCK_PATH_OP_BY_OP;
      else if (*path == "fused") e.block_path = MI_BLOCK_PATH_FUSED;
      else if (*path == "fused_graphs") e.block_path = MI_BLOCK_PATH_FUSED_GRAPHS;
      else throw RequestError("invalid_params",
                              "\"block_path\" must be \"default\", \"op_by_op\", \"fused\" or \"fused_graphs\"");
    }
    // The old engine goes first: two engines would hold two arenas.
    join_worker();
    mi_engine_free(engine_);
    engine_ = nullptr;
    model_.clear();
    mi_engine* engine = nullptr;
    if (const mi_status s = mi_engine_create(&e, &engine); s != MI_OK) throw_status(s);
    engine_ = engine;
    model_ = *model;
    mode_ = e.mode;
    out_.log("info", "engine created for " + model_ + " (mode " + mode_name(mode_) + ", device " +
                         std::to_string(e.device_index) + "); weights load on the first generate",
             id);
    return "{\"model\":" + str(model_) + ",\"mode\":\"" + mode_name(mode_) + "\",\"device\":" + std::to_string(e.device_index) + "}";
  }

  std::string generate(const Params& p, const std::string& id) {
    p.only({"output", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "shift", "seed", "uncond_skip_layers",
            "slg_layers", "slg_scale", "slg_start", "slg_stop", "max_sequence_length", "sampler"});
    if (engine_ == nullptr) throw RequestError("no_engine", "no model loaded; send \"load\" first");
    if (generating()) throw RequestError("busy", "a generation is running; cancel it or wait for its done event");
    auto w = std::make_unique<Work>();
    mi_gen_params& g = w->params;
    if (const mi_status s = mi_gen_params_init(&g, sizeof g); s != MI_OK) throw_status(s);
    const auto output = p.string("output");
    if (!output || output->empty()) throw RequestError("invalid_params", "\"output\" (PNG path) is required");
    w->output = *output;
    w->prompt = p.string("prompt").value_or("");
    w->negative = p.string("negative_prompt").value_or("");
    g.prompt = w->prompt.c_str();
    g.negative_prompt = w->negative.c_str();
    constexpr std::int64_t kI32 = std::numeric_limits<std::int32_t>::max();
    if (const auto v = p.integer("width", 0, kI32)) g.width = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("height", 0, kI32)) g.height = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("steps", 0, kI32)) g.steps = static_cast<std::int32_t>(*v);
    if (const auto v = p.number("cfg")) g.cfg = *v;
    if (const auto v = p.number("shift")) g.shift = *v;
    // A JSON null seed means random, like parameters.json.
    if (p.is_null("seed")) g.seed = -1;
    else if (const auto v = p.integer("seed", -1, std::numeric_limits<std::int64_t>::max())) g.seed = *v;
    if (auto v = p.layers("uncond_skip_layers")) {
      w->uncond = std::move(*v);
      g.uncond_skip_layers = w->uncond.data();
      g.uncond_skip_layers_count = w->uncond.size();
    } else if (p.is_null("uncond_skip_layers")) {
      g.uncond_skip_layers_count = 0;
    }
    if (auto v = p.layers("slg_layers")) {
      w->slg = std::move(*v);
      g.slg_layers = w->slg.data();
      g.slg_layers_count = w->slg.size();
    }
    if (const auto v = p.number("slg_scale")) g.slg_scale = *v;
    if (const auto v = p.number("slg_start")) g.slg_start = *v;
    if (const auto v = p.number("slg_stop")) g.slg_stop = *v;
    if (const auto v = p.integer("max_sequence_length", 0, kI32)) g.max_sequence_length = static_cast<std::int32_t>(*v);
    if (const auto v = p.string("sampler")) {
      if (*v == "euler") g.sampler = MI_SAMPLER_EULER;
      else if (*v == "ab2") g.sampler = MI_SAMPLER_AB2;
      else throw RequestError("invalid_params", "\"sampler\" must be \"euler\" or \"ab2\"");
    }
    if (const mi_status s = mi_gen_params_validate(&g); s != MI_OK) {
      throw RequestError(s == MI_ERR_INVALID_ARGUMENT ? "invalid_params" : status_code(s), mi_last_error_message());
    }
    join_worker();
    job_ = std::make_unique<Job>();
    job_->id = id;
    w->job = job_.get();
    w->out = &out_;
    g.progress_callback = on_progress;
    g.plan_callback = on_plan;
    g.user_data = w.get();
    // Busy before the thread exists (it clears the flag when it finishes, possibly before std::thread returns), and
    // idle again if the thread cannot be started.
    running_.store(true);
    try {
      worker_ = std::thread([this, work = std::move(w)]() mutable { run(*work); });
    } catch (...) {
      running_.store(false);
      throw;
    }
    return "{\"accepted\":true}";
  }

  std::string cancel(const Params& p) {
    p.only({"id"});
    const Value* want = p.find("id");
    if (!generating()) return "{\"cancelling\":false}";
    const std::string running_id = job_->id;
    if (want != nullptr && id_json(want) != running_id) return "{\"cancelling\":false,\"generating_id\":" + running_id + "}";
    request_cancel();
    return "{\"cancelling\":true,\"generating_id\":" + running_id + "}";
  }

  std::string status(const Params& p) {
    p.only({});
    std::string s = "{\"state\":\"" + std::string(engine_ == nullptr ? "unloaded" : generating() ? "generating" : "idle") + "\"";
    if (engine_ != nullptr) s += ",\"model\":" + str(model_) + ",\"mode\":\"" + mode_name(mode_) + "\"";
    if (generating()) {
      const int phase = job_->phase.load();
      s += ",\"generating_id\":" + job_->id + ",\"phase\":" +
           (phase < 0 ? std::string("null") : "\"" + std::string(phase_name(static_cast<mi_phase>(phase))) + "\"") +
           ",\"step\":" + std::to_string(job_->step.load()) + ",\"total_steps\":" + std::to_string(job_->total_steps.load()) +
           ",\"elapsed_seconds\":" + num(job_->elapsed.load()) + ",\"cancelling\":" + tf(job_->cancel.load());
    }
    return s + "}";
  }

  std::string unload(const Params& p) {
    p.only({});
    if (generating()) throw RequestError("busy", "a generation is running; cancel it or wait for its done event");
    join_worker();
    const bool had = engine_ != nullptr;
    mi_engine_free(engine_);
    engine_ = nullptr;
    model_.clear();
    return std::string("{\"unloaded\":") + tf(had) + "}";
  }

  // One generation's inputs; owned by the worker thread.
  struct Work {
    mi_gen_params params{};
    std::string output, prompt, negative;
    std::vector<std::int32_t> uncond, slg;
    Job* job = nullptr;
    Output* out = nullptr;
    mi_engine* engine = nullptr;
  };

  static void on_progress(const mi_progress* p, void* user) {
    Work& w = *static_cast<Work*>(user);
    Job& job = *w.job;
    // A cancel that arrived before mi_generate() was entered: it lands now, at the first phase start.
    if (job.cancel.load()) mi_cancel(w.engine);
    job.phase.store(p->phase);
    job.step.store(p->step);
    job.total_steps.store(p->total_steps);
    job.elapsed.store(p->elapsed_seconds);
    w.out->emit("{\"event\":\"progress\",\"id\":" + job.id + ",\"phase\":\"" + phase_name(p->phase) + "\",\"step\":" +
                std::to_string(p->step) + ",\"total_steps\":" + std::to_string(p->total_steps) +
                ",\"step_seconds\":" + num(p->step_seconds) + ",\"elapsed_seconds\":" + num(p->elapsed_seconds) + "}");
  }

  static void on_plan(const char* explanation, const char* plan_json, void* user) {
    Work& w = *static_cast<Work*>(user);
    w.out->emit("{\"event\":\"plan\",\"id\":" + w.job->id + ",\"explanation\":" + str(explanation) +
                ",\"plan\":" + one_line(plan_json) + "}");
  }

  void run(Work& w) {
    w.engine = engine_;
    Job& job = *w.job;
    mi_result* r = nullptr;
    const mi_status s = mi_generate(engine_, &w.params, &r);
    std::string done = "{\"event\":\"done\",\"id\":" + job.id;
    if (s == MI_OK) {
      if (const mi_status ws = mi_save_png(r, w.output.c_str()); ws == MI_OK) {
        done += ",\"status\":\"ok\",\"image\":" + str(w.output) + ",\"report\":" + one_line(mi_result_report_json(r));
      } else {
        done += ",\"status\":\"error\",\"error\":" + error_json(status_code(ws), mi_last_error_message()) +
                ",\"report\":" + one_line(mi_result_report_json(r));
      }
    } else if (s == MI_CANCELLED) {
      done += ",\"status\":\"cancelled\",\"steps_run\":" + std::to_string(mi_result_steps_run(r)) +
              ",\"report\":" + one_line(mi_result_report_json(r));
    } else {
      done += ",\"status\":\"error\",\"error\":" + error_json(status_code(s), mi_last_error_message());
    }
    mi_result_free(r);
    // Idle before the done line goes out: a client may send the next generate as soon as it reads it.
    running_.store(false);
    out_.emit(done + "}");
  }

  void request_cancel() {
    if (!generating()) return;
    job_->cancel.store(true);
    mi_cancel(engine_);
  }

  void join_worker() {
    if (worker_.joinable()) worker_.join();
  }

  Output& out_;
  mi_engine* engine_ = nullptr;
  std::string model_;
  mi_mode mode_ = MI_MODE_BALANCED;
  std::unique_ptr<Job> job_;
  std::thread worker_;
  std::atomic<bool> running_{false};
};

// fds 0-2 must exist before anything else is opened. A launcher that started the server with one of them closed would
// otherwise get the protocol copy (or a library's file) as fd 2, and stderr output inside the protocol stream.
#ifdef _WIN32
void ignore_invalid_parameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {}

bool ensure_std_fds() {
  // The CRT reports a bad fd through the invalid-parameter handler, which aborts by default.
  const _invalid_parameter_handler old = _set_thread_local_invalid_parameter_handler(ignore_invalid_parameter);
  bool ok = true;
  for (int fd = 0; fd <= 2 && ok; ++fd) {
    const intptr_t h = _get_osfhandle(fd);
    if (h != -1 && h != -2) continue;  // -2: no stream (a GUI parent without console handles)
    const int nul = _open("NUL", (fd == 0 ? _O_RDONLY : _O_WRONLY) | _O_BINARY);
    ok = nul >= 0 && (nul == fd || (_dup2(nul, fd) == 0 && _close(nul) == 0));
  }
  _set_thread_local_invalid_parameter_handler(old);
  return ok;
}
#else
bool ensure_std_fds() {
  for (int fd = 0; fd <= 2; ++fd) {
    if (fcntl(fd, F_GETFD) != -1 || errno != EBADF) continue;
    const int nul = open("/dev/null", fd == 0 ? O_RDONLY : O_WRONLY);  // the lowest free fd: this one
    if (nul < 0) return false;
    if (nul != fd && (dup2(nul, fd) < 0 || close(nul) != 0)) return false;
  }
  return true;
}

bool same_file(int a, int b) {
  struct stat sa {}, sb {};
  return fstat(a, &sa) == 0 && fstat(b, &sb) == 0 && sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}
#endif

// The protocol stream: a private duplicate of stdout (fd >= 3, not inherited by child processes). fd 1 then points at
// stderr, or at the null device when stderr is unusable: missing, or the protocol stream itself (2>&1), in which case
// stderr is silenced too.
std::FILE* claim_stdout() {
#ifdef _WIN32
  if (!ensure_std_fds()) return nullptr;
  _setmode(_fileno(stdin), _O_BINARY);
  const int fd = _dup(1);
  if (fd < 0) return nullptr;
  if (!SetHandleInformation(reinterpret_cast<HANDLE>(_get_osfhandle(fd)), HANDLE_FLAG_INHERIT, 0)) return nullptr;
  const HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
  const bool stderr_usable = err != nullptr && err != INVALID_HANDLE_VALUE && GetFileType(err) != FILE_TYPE_UNKNOWN &&
                             _get_osfhandle(2) != _get_osfhandle(fd);
  if (stderr_usable) {
    if (_dup2(2, 1) != 0) return nullptr;
  } else {
    const int nul = _open("NUL", _O_WRONLY | _O_BINARY);
    if (nul < 0 || _dup2(nul, 1) != 0 || _dup2(nul, 2) != 0 || _close(nul) != 0) return nullptr;
  }
  _setmode(fd, _O_BINARY);
  return _fdopen(fd, "wb");
#else
  std::signal(SIGPIPE, SIG_IGN);  // a launcher that went away: writes fail and the server exits
  if (!ensure_std_fds()) return nullptr;
  const int fd = fcntl(1, F_DUPFD_CLOEXEC, 3);
  if (fd < 0) return nullptr;
  if (isatty(2) || !same_file(2, fd)) {
    if (dup2(2, 1) < 0) return nullptr;
  } else {
    const int nul = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (nul < 0 || dup2(nul, 1) < 0 || dup2(nul, 2) < 0 || close(nul) != 0) return nullptr;
  }
  return fdopen(fd, "w");
#endif
}

// "--parent-pid <pid>": a positive decimal process id, else 0.
long long parse_pid(const char* s) {
  long long v = 0;
  const char* end = s + std::char_traits<char>::length(s);
  const auto [ptr, ec] = std::from_chars(s, end, v);
  return ec == std::errc() && ptr == end && ptr != s && v > 0 && v <= std::numeric_limits<std::int32_t>::max() ? v : 0;
}

[[noreturn]] void parent_gone(long long pid) {
  std::fprintf(stderr, "moss-server: parent process %lld exited; exiting\n", pid);
  std::_Exit(3);
}

// --parent-pid: exit when the launcher dies, even when its end of stdin stays open (a pipe inherited elsewhere, a
// crash that leaves the handle to a grandchild). False when the process is not running at startup.
bool start_parent_watchdog(long long pid) {
#ifdef _WIN32
  const HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
  if (h == nullptr) return false;
  if (WaitForSingleObject(h, 0) == WAIT_OBJECT_0) return false;
  std::thread([h, pid] {
    WaitForSingleObject(h, INFINITE);
    parent_gone(pid);
  }).detach();
#else
  const bool child = getppid() == static_cast<pid_t>(pid);
#ifdef __linux__
  // The kernel's notice, immediate. It fires when the thread that spawned us exits (see docs/server-protocol.md); the
  // poll below covers the rest and non-Linux systems.
  if (child && (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() != static_cast<pid_t>(pid))) return false;
#endif
  const auto alive = [pid, child] {
    // A direct parent that died has us re-parented at once, even before it is reaped (kill() still sees a zombie).
    if (child) return getppid() == static_cast<pid_t>(pid);
    return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
  };
  if (!alive()) return false;
  std::thread([alive, pid] {
    while (alive()) std::this_thread::sleep_for(std::chrono::milliseconds(250));
    parent_gone(pid);
  }).detach();
#endif
  return true;
}

// Reads one line into `line`; false at end of input. Lines over kMaxLineBytes are consumed and reported via `too_long`.
bool read_line(std::string& line, bool& too_long) {
  line.clear();
  too_long = false;
  int c = 0;
  bool any = false;
  while ((c = std::fgetc(stdin)) != EOF) {
    any = true;
    if (c == '\n') break;
    if (line.size() < kMaxLineBytes) line += static_cast<char>(c);
    else too_long = true;
  }
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return any;
}

}  // namespace

int main(int argc, char** argv) {
  long long parent_pid = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--version") {
      std::printf("moss-server %s (protocol %d, ABI %d)\n", mi_version(), kProtocolVersion, mi_abi_version());
      return 0;
    }
    if (a == "--parent-pid" && i + 1 < argc && (parent_pid = parse_pid(argv[i + 1])) > 0) {
      ++i;
      continue;
    }
    std::fprintf(stderr,
                 "usage: moss-server [--version] [--parent-pid <pid>]\n"
                 "JSON-lines protocol on stdin/stdout, see docs/server-protocol.md.\n"
                 "  --parent-pid <pid>  exit when process <pid> (the launcher) exits\n");
    return a == "-h" || a == "--help" ? 0 : 2;
  }
  std::FILE* proto = claim_stdout();
  if (proto == nullptr) {
    std::fprintf(stderr, "moss-server: cannot set up the protocol stream\n");
    return 1;
  }
  if (parent_pid > 0 && !start_parent_watchdog(parent_pid)) {
    std::fprintf(stderr, "moss-server: parent process %lld is not running\n", parent_pid);
    return 1;
  }
  Output out(proto);
  Server server(out);
  std::string line;
  bool too_long = false;
  while (read_line(line, too_long)) {
    if (too_long) {
      server.reject_long_line();
      continue;
    }
    if (line.find_first_not_of(" \t") == std::string::npos) continue;  // blank lines are ignored
    if (!server.handle_line(line)) break;
  }
  server.stop();
  std::fclose(proto);
  return 0;
}
