// moss-server: the engine as a sidecar process for launchers. JSON lines over stdin/stdout (docs/server-protocol.md),
// built only on the public C API (mossinfer.h).
//
// Threads: the main thread reads and answers requests; a job (generate, preload, release) runs on one worker thread, so
// cancel and status are answered while it runs. Every protocol line goes through emit() under one mutex. stdout carries protocol lines
// only: at startup the protocol stream gets a private, non-inheritable copy of fd 1 and fd 1 is pointed at stderr (or
// at the null device when stderr is unusable), so anything else printing to stdout (the engine, a library) never
// reaches the protocol stream.
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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

#include "json_out.h"
#include "mossinfer/mossinfer.h"
#include "names.h"
#include "util/json.h"

namespace {

constexpr int kProtocolVersion = 1;
constexpr std::size_t kMaxLineBytes = std::size_t{1} << 20;  // a request is a prompt and a few numbers
constexpr std::size_t kMaxJsonValues = 1 << 16;

using mi::json::Value;

// ---------------------------------------------------------------- JSON output

using mi::server::num;
using mi::server::one_line;
using mi::server::str;
using mi::server::utf8_sequence_length;

const char* tf(bool b) { return b ? "true" : "false"; }

// mi_progress.eta_seconds: null when unknown (negative).
std::string eta_json(double eta) { return eta < 0 ? "null" : num(eta); }

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

// An engine message with the C fields' names replaced by the protocol parameters' they come from. A name is replaced
// only as a whole token, so a file path in the message that contains one (".../my_control_adapter.safetensors",
// "/data/mask_path/x.png") comes back unchanged: it must start the message or follow a space, quote, "(" or "[" (a
// ".image_path" must follow "]"), and end the message or precede a space, quote, ",", ":", ";", ")", "[", "]", or a
// "." that ends a sentence.
std::string protocol_names(std::string message) {
  const auto token_at = [&](std::size_t at, std::string_view from) {
    const bool starts = from.front() == '.' ? at > 0 && message[at - 1] == ']'
                                            : at == 0 || std::string_view(" \"'([").find(message[at - 1]) != std::string_view::npos;
    const std::size_t end = at + from.size();
    if (!starts) return false;
    if (end == message.size()) return true;
    if (message[end] == '.') return end + 1 == message.size() || message[end + 1] == ' ';
    return std::string_view(" \"',:;)[]").find(message[end]) != std::string_view::npos;
  };
  static constexpr std::pair<std::string_view, std::string_view> kRenames[] = {
      {"dataspace_path", "dataspace"}, {"init_image_path", "init_image"}, {"mask_path", "mask"},
      {"mi_engine_params.distill_dir", "load's \"distill\""}, {"distill_dir", "distill"},
      {"hires_width / hires_height", "hires"}, {"hires_width and hires_height", "hires width and height"},
      {"with MI_MASK_BINARY01", "with mask_encoding \"binary01\""},
      // "control_images[0].image_path" -> "control.images[0].path".
      {"control_images", "control.images"}, {".image_path", ".path"}, {"control_adapter", "control.adapter"},
      {"control_lora_strength", "control.lora_strength"}};
  for (const auto& [from, to] : kRenames) {
    for (std::size_t at = 0; (at = message.find(from, at)) != std::string::npos;) {
      if (!token_at(at, from)) {
        ++at;
        continue;
      }
      message.replace(at, from.size(), to);
      at += to.size();
    }
  }
  return message;
}

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
    for (std::size_t i = 0; i < obj_->size(); ++i) {
      const std::string& k = (*obj_)[i].first;
      bool ok = false;
      for (const std::string_view n : known) ok = ok || k == n;
      if (!ok) throw RequestError("invalid_params", "unknown parameter \"" + k + "\"");
      // A repeated parameter would silently lose one of its values.
      for (std::size_t j = 0; j < i; ++j) {
        if ((*obj_)[j].first == k) throw RequestError("invalid_params", "parameter \"" + k + "\" given more than once");
      }
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
  // A path: a non-empty string.
  std::optional<std::string> path(std::string_view key) const {
    std::optional<std::string> v = string(key);
    if (v && v->empty()) throw bad(key, "a non-empty path");
    return v;
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

  // One of the names in `table` (names.h), as its C enum value.
  template <class E, std::size_t N>
  std::optional<E> name(std::string_view key, const mi::server::Name<E> (&table)[N]) const {
    const std::optional<std::string> v = string(key);
    if (!v) return std::nullopt;
    std::string names;
    for (std::size_t i = 0; i < N; ++i) {
      if (*v == table[i].name) return table[i].value;
      names += std::string(i == 0 ? "" : i + 1 == N ? " or " : ", ") + "\"" + std::string(table[i].name) + "\"";
    }
    throw bad(key, names);
  }
  // `hires`: {"width": int, "height": int, "strength": number (optional), "steps": int (optional)}.
  struct Hires {
    std::int32_t width = 0, height = 0, steps = 0;
    std::optional<double> strength;
  };
  std::optional<Hires> hires(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    const std::string k0(key);
    if (!v->is_object()) throw bad(key, "an object {\"width\", \"height\", \"strength\"?, \"steps\"?}");
    Hires h;
    bool have_width = false, have_height = false, have_steps = false;
    constexpr std::int64_t kI32 = std::numeric_limits<std::int32_t>::max();
    for (const auto& [k, m] : v->as_object()) {
      if (k == "strength") {
        if (h.strength) throw RequestError("invalid_params", "\"" + k0 + "\" has \"strength\" more than once");
        if (!m.is_number()) throw bad(k0 + ".strength", "a number");
        h.strength = m.as_double();
        continue;
      }
      bool* seen = k == "width" ? &have_width : k == "height" ? &have_height : k == "steps" ? &have_steps : nullptr;
      if (seen == nullptr) throw RequestError("invalid_params", "\"" + k0 + "\" has an unknown member \"" + k + "\"");
      if (*seen) throw RequestError("invalid_params", "\"" + k0 + "\" has \"" + k + "\" more than once");
      *seen = true;
      const std::int64_t lo = k == "steps" ? 0 : 1;  // a given size is positive; steps 0 = the request's
      if (!m.is_int() || m.as_int() < lo || m.as_int() > kI32) {
        throw bad(k0 + "." + k, "an integer in [" + std::to_string(lo) + ", " + std::to_string(kI32) + "]");
      }
      (k == "width" ? h.width : k == "height" ? h.height : h.steps) = static_cast<std::int32_t>(m.as_int());
    }
    if (!have_width || !have_height) throw RequestError("invalid_params", "\"" + k0 + "\" needs \"width\" and \"height\"");
    return h;
  }
  // `loras`: an array of {"path": string, "strength": number (default 1)}.
  std::optional<std::vector<std::pair<std::string, double>>> loras(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    if (!v->is_array()) throw bad(key, "an array of {\"path\", \"strength\"} objects");
    std::vector<std::pair<std::string, double>> out;
    for (const Value& e : v->as_array()) {
      const std::string item = std::string(key) + "[" + std::to_string(out.size()) + "]";
      if (!e.is_object()) throw bad(item, "an object {\"path\": string, \"strength\": number}");
      std::optional<std::string> path;
      double strength = 1.0;
      bool have_strength = false;
      for (const auto& [k, m] : e.as_object()) {
        if ((k == "path" && path) || (k == "strength" && have_strength)) {
          throw RequestError("invalid_params", "\"" + item + "\" has \"" + k + "\" more than once");
        }
        if (k == "path") {
          if (!m.is_string() || m.as_string().empty()) throw bad(item + ".path", "a non-empty string (the LoRA file)");
          if (m.as_string().find('\0') != std::string::npos) throw bad(item + ".path", "a string without NUL characters");
          path = m.as_string();
        } else if (k == "strength") {
          if (!m.is_number()) throw bad(item + ".strength", "a number");
          strength = m.as_double();
          have_strength = true;
        } else {
          throw RequestError("invalid_params", "\"" + item + "\" has an unknown member \"" + k + "\"");
        }
      }
      if (!path) throw RequestError("invalid_params", "\"" + item + ".path\" (the LoRA file) is required");
      out.emplace_back(*path, strength);
    }
    return out;
  }
  // `control`: {"adapter": path, "lora_strength": number (optional), "images": [{"kind": string, "path": path,
  // "strength", "start", "end": numbers (optional)}] (optional)}. Absent numbers keep the C API's defaults.
  struct ControlImage {
    std::string kind, path;
    std::optional<double> strength, start, end;
  };
  struct Control {
    std::string adapter;
    std::optional<double> lora_strength;
    std::vector<ControlImage> images;
  };
  std::optional<Control> control(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    const std::string k0(key);
    if (!v->is_object()) throw bad(key, "an object {\"adapter\", \"lora_strength\"?, \"images\"?}");
    // Each member once; strings non-empty and without NUL characters.
    const auto members = [](const mi::json::Value& obj, const std::string& field,
                            std::initializer_list<std::string_view> known) {
      const mi::json::Object& o = obj.as_object();
      for (std::size_t i = 0; i < o.size(); ++i) {
        bool ok = false;
        for (const std::string_view n : known) ok = ok || o[i].first == n;
        if (!ok) throw RequestError("invalid_params", "\"" + field + "\" has an unknown member \"" + o[i].first + "\"");
        for (std::size_t j = 0; j < i; ++j) {
          if (o[j].first == o[i].first) {
            throw RequestError("invalid_params", "\"" + field + "\" has \"" + o[i].first + "\" more than once");
          }
        }
      }
    };
    const auto nonempty = [](const mi::json::Value& m, const std::string& field, const std::string& what) {
      if (!m.is_string() || m.as_string().empty()) throw bad(field, "a non-empty string (" + what + ")");
      if (m.as_string().find('\0') != std::string::npos) throw bad(field, "a string without NUL characters");
      return m.as_string();
    };
    const auto numeric = [](const mi::json::Value& m, const std::string& field) {
      if (!m.is_number()) throw bad(field, "a number");
      return m.as_double();
    };
    members(*v, k0, {"adapter", "lora_strength", "images"});
    Control c;
    bool have_adapter = false;
    for (const auto& [k, m] : v->as_object()) {
      if (k == "adapter") {
        c.adapter = nonempty(m, k0 + ".adapter", "the control adapter file");
        have_adapter = true;
      } else if (k == "lora_strength") {
        c.lora_strength = numeric(m, k0 + ".lora_strength");
      } else if (!m.is_array()) {
        throw bad(k0 + ".images", "an array of {\"kind\", \"path\", \"strength\"?, \"start\"?, \"end\"?} objects");
      } else {
        for (const Value& e : m.as_array()) {
          const std::string item = k0 + ".images[" + std::to_string(c.images.size()) + "]";
          if (!e.is_object()) throw bad(item, "an object {\"kind\", \"path\", \"strength\"?, \"start\"?, \"end\"?}");
          members(e, item, {"kind", "path", "strength", "start", "end"});
          ControlImage im;
          for (const auto& [ik, iv] : e.as_object()) {
            if (ik == "kind") im.kind = nonempty(iv, item + ".kind", "one of the adapter's kinds, e.g. depth");
            else if (ik == "path") im.path = nonempty(iv, item + ".path", "the image file");
            else (ik == "strength" ? im.strength : ik == "start" ? im.start : im.end) = numeric(iv, item + "." + ik);
          }
          if (im.kind.empty()) throw RequestError("invalid_params", "\"" + item + ".kind\" is required");
          if (im.path.empty()) throw RequestError("invalid_params", "\"" + item + ".path\" (the image file) is required");
          c.images.push_back(std::move(im));
        }
      }
    }
    if (!have_adapter) throw RequestError("invalid_params", "\"" + k0 + ".adapter\" (the control adapter file) is required");
    return c;
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
      std::optional<std::string> result;  // none: the worker answers when the job ends (release)
      bool keep_running = true;
      if (m == "hello") result = hello(params);
      else if (m == "load") result = load(params, id);
      else if (m == "generate") result = generate(params, id);
      else if (m == "preload") result = preload(params, id);
      else if (m == "release") result = release(params, id);
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
      if (result) out_.emit("{\"id\":" + id + ",\"result\":" + *result + "}");
      return keep_running;
    } catch (const RequestError& e) {
      out_.emit("{\"id\":" + id + ",\"error\":" + error_json(e.code, e.what()) + "}");
    } catch (const std::exception& e) {
      out_.emit("{\"id\":" + id + ",\"error\":" + error_json("internal", e.what()) + "}");
    }
    return true;
  }

  // Cancels a running job, waits for it and releases the engine.
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
  enum class Kind { kGenerate, kPreload, kRelease };

  // State shared with the worker thread.
  struct Job {
    std::string id;  // JSON of the request id
    Kind kind = Kind::kGenerate;
    std::atomic<bool> cancel{false};
    std::atomic<int> phase{-1};
    std::atomic<int> step{0}, total_steps{0};
    std::atomic<double> elapsed{0};
    std::atomic<double> fraction{0}, eta{-1};  // eta < 0: unknown
  };
  struct Work;

  // A job runs on the worker (its id is job_->id).
  bool generating() const { return running_.load(); }

  static RequestError busy_error() {
    return RequestError("busy", "a generate, preload or release is running; cancel it or wait until it ends (its done "
                                "event, or the release's response)");
  }

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
           tf(caps & MI_CAP_COMFYUI_MODELS) + ",\"model_files\":" + tf(caps & MI_CAP_MODEL_FILES) + ",\"lora\":" +
           tf(caps & MI_CAP_LORA) + ",\"tiers\":" + tf(caps & MI_CAP_TIERS) + ",\"samplers\":" + tf(caps & MI_CAP_SAMPLERS) +
           ",\"image_input\":" + tf(caps & MI_CAP_IMAGE_INPUT) + ",\"hires\":" + tf(caps & MI_CAP_HIRES) + ",\"distill\":" +
           tf(caps & MI_CAP_DISTILL) + ",\"system_cuda\":" + tf(caps & MI_CAP_SYSTEM_CUDA) + ",\"control\":" +
           tf(caps & MI_CAP_CONTROL) + ",\"accel\":" + tf(caps & MI_CAP_ACCEL) + ",\"preload\":" + tf(caps & MI_CAP_PRELOAD) +
           ",\"release\":" + tf(caps & MI_CAP_RELEASE) + ",\"certificate_dirs\":" + tf(caps & MI_CAP_CERTIFICATE_DIRS) +
           ",\"te_host_pin\":" + tf(caps & MI_CAP_TE_HOST_PIN) + ",\"progress_fraction\":" +
           tf(caps & MI_CAP_PROGRESS_FRACTION) + "},\"probe\":{\"driver_cuda_version\":" +
           std::to_string(probe.driver_cuda_version) + ",\"runtime_cuda_version\":" + std::to_string(probe.runtime_cuda_version) +
           ",\"driver_compatible\":" + tf(probe.driver_compatible) + ",\"device_count\":" + std::to_string(probe.device_count) +
           ",\"supported_device_count\":" + std::to_string(probe.supported_device_count) +
           ",\"build_flavour\":" + str(probe.build_flavour) + ",\"recommended_flavour\":" + str(probe.recommended_flavour) +
           ",\"message\":" + str(probe.message) + ",\"cublaslt\":{\"source\":" + str(probe.cublaslt_source) +
           ",\"version\":" + std::to_string(probe.cublaslt_version) + ",\"path\":" + str(probe.cublaslt_path) +
           ",\"error\":" + str(probe.cublaslt_error) + "}" + (with_devices ? ",\"devices\":[" + devices + "]" : std::string()) + "}}";
  }

  std::string load(const Params& p, const std::string& id) {
    p.only({"model", "mode", "device", "vram_budget_bytes", "vram_reserve_bytes", "host_ram_budget_bytes", "cache_dir",
            "embedding_cache", "direct_io", "block_path", "dit_file", "text_encoder_file", "vae_file", "tier", "attention",
            "lora_strict", "time_embedding_file", "distill", "certificate_dirs", "dit_direct_io", "te_host_pin"});
    if (generating()) throw busy_error();
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
    const std::optional<std::string> dit_file = p.string("dit_file");
    const std::optional<std::string> te_file = p.string("text_encoder_file");
    const std::optional<std::string> vae_file = p.string("vae_file");
    if (dit_file) e.dit_file = dit_file->c_str();
    if (te_file) e.text_encoder_file = te_file->c_str();
    if (vae_file) e.vae_file = vae_file->c_str();
    if (const auto v = p.name("tier", mi::server::kTierNames)) e.tier = *v;
    if (const auto v = p.name("attention", mi::server::kAttentionNames)) e.attention = *v;
    if (const auto v = p.boolean("lora_strict")) e.lora_strict = *v ? 1 : 0;
    const std::optional<std::string> te = p.path("time_embedding_file");
    const std::optional<std::string> distill = p.path("distill");
    if (te) e.time_embedding_file = te->c_str();
    if (distill) e.distill_dir = distill->c_str();
    // The engine copies the directories; these live until mi_engine_create() returns. Its own checks count them.
    std::vector<std::string> cert_dirs;
    std::vector<const char*> cert_ptrs;
    if (const Value* v = p.find("certificate_dirs")) {
      constexpr const char* kWhat = "an array of non-empty directory paths";
      if (!v->is_array()) throw RequestError("invalid_params", std::string("\"certificate_dirs\" must be ") + kWhat);
      for (const Value& d : v->as_array()) {
        if (!d.is_string() || d.as_string().empty() || d.as_string().find('\0') != std::string::npos) {
          throw RequestError("invalid_params", "\"certificate_dirs[" + std::to_string(cert_dirs.size()) + "]\" must be a " +
                                                   "non-empty path without NUL characters (" + kWhat + ")");
        }
        cert_dirs.push_back(d.as_string());
      }
      for (const std::string& d : cert_dirs) cert_ptrs.push_back(d.c_str());
      e.certificate_dirs = cert_ptrs.data();
      e.certificate_dirs_count = cert_ptrs.size();
    }
    if (const auto v = p.name("dit_direct_io", mi::server::kDitDirectIoNames)) e.dit_direct_io = *v;
    if (const auto v = p.name("te_host_pin", mi::server::kTeHostPinNames)) e.te_host_pin = *v;
    // The old engine goes first: two engines would hold two arenas.
    join_worker();
    mi_engine_free(engine_);
    engine_ = nullptr;
    model_.clear();
    mi_engine* engine = nullptr;
    if (const mi_status s = mi_engine_create(&e, &engine); s != MI_OK) {
      throw RequestError(status_code(s), protocol_names(mi_last_error_message()));
    }
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
            "slg_layers", "slg_scale", "slg_start", "slg_stop", "max_sequence_length", "sampler", "scheduler",
            "flow_multiplier", "noise_layout", "noise_source", "eta", "s_noise", "dataspace", "loras", "save_metadata",
            "include_prompts", "preview_output", "preview_every", "init_image", "mask", "mask_encoding",
            "denoise_strength", "resize", "hires", "distill", "control", "cfg_truncation", "step_cache",
            "step_cache_threshold"});
    auto w = std::make_unique<Work>();
    const auto output = p.string("output");
    if (!output || output->empty()) throw RequestError("invalid_params", "\"output\" (PNG path) is required");
    w->output = *output;
    request_params(p, *w, false);
    return start(std::move(w), id, Kind::kGenerate);
  }

  // generate's parameters without the output, the preview and the PNG metadata; the prompt is optional.
  std::string preload(const Params& p, const std::string& id) {
    p.only({"prompt", "negative_prompt", "width", "height", "steps", "cfg", "shift", "seed", "uncond_skip_layers",
            "slg_layers", "slg_scale", "slg_start", "slg_stop", "max_sequence_length", "sampler", "scheduler",
            "flow_multiplier", "noise_layout", "noise_source", "eta", "s_noise", "dataspace", "loras", "init_image", "mask",
            "mask_encoding", "denoise_strength", "resize", "hires", "distill", "control", "cfg_truncation", "step_cache",
            "step_cache_threshold"});
    auto w = std::make_unique<Work>();
    request_params(p, *w, true);
    return start(std::move(w), id, Kind::kPreload);
  }

  // Answered by the worker when the release ends (reading the DiT into RAM takes seconds), so a cancel sent meanwhile
  // is read and interrupts the host copy.
  std::optional<std::string> release(const Params& p, const std::string& id) {
    p.only({"keep_host_copy"});
    auto w = std::make_unique<Work>();
    w->keep_host_copy = p.boolean("keep_host_copy").value_or(true);
    start(std::move(w), id, Kind::kRelease);
    return std::nullopt;
  }

  // The parameters of a generate or preload request into w.params, checked by the C API (without an engine).
  void request_params(const Params& p, Work& work, bool preload) {
    Work* w = &work;
    mi_gen_params& g = w->params;
    if (const mi_status s = mi_gen_params_init(&g, sizeof g); s != MI_OK) throw_status(s);
    w->prompt = p.string("prompt").value_or("");
    w->negative = p.string("negative_prompt").value_or("");
    g.prompt = w->prompt.c_str();
    g.negative_prompt = w->negative.c_str();
    constexpr std::int64_t kI32 = std::numeric_limits<std::int32_t>::max();
    if (const auto v = p.integer("width", 0, kI32)) g.width = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("height", 0, kI32)) g.height = static_cast<std::int32_t>(*v);
    // Absent steps and cfg are the C API's unset markers: the defaults, or a distill bundle's.
    // Absent = unset: mi_gen_params_init leaves steps, cfg and denoise_strength at their markers (mossinfer.h).
    if (const auto v = p.integer("steps", 1, kI32)) g.steps = static_cast<std::int32_t>(*v);
    if (const auto v = p.number("cfg")) {
      if (*v < 0) throw RequestError("invalid_params", "\"cfg\" must be a number >= 0");
      g.cfg = *v;
    }
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
    if (const auto v = p.name("sampler", mi::server::kSamplerNames)) g.sampler = *v;
    if (const auto v = p.name("scheduler", mi::server::kSchedulerNames)) g.scheduler = *v;
    if (const auto v = p.name("noise_layout", mi::server::kNoiseLayoutNames)) g.noise_layout = *v;
    if (const auto v = p.name("noise_source", mi::server::kNoiseSourceNames)) g.noise_source = *v;
    if (const auto v = p.number("flow_multiplier")) g.flow_multiplier = *v;
    if (const auto v = p.number("eta")) g.eta = *v;
    if (const auto v = p.number("s_noise")) g.s_noise = *v;
    if (const auto v = p.string("dataspace")) {
      w->dataspace = *v;
      g.dataspace_path = w->dataspace.c_str();
    }
    if (auto v = p.loras("loras")) {
      // Every path is stored before the array points into them.
      for (auto& [path, strength] : *v) w->lora_paths.push_back(std::move(path));
      for (std::size_t i = 0; i < v->size(); ++i) w->loras.push_back({w->lora_paths[i].c_str(), (*v)[i].second});
      g.loras = w->loras.data();
      g.loras_count = w->loras.size();
    }
    if (const auto v = p.boolean("save_metadata")) g.save_metadata = *v ? 1 : 0;
    if (const auto v = p.boolean("include_prompts")) g.include_prompts = *v ? 1 : 0;
    // Image input: paths only (the server never takes pixels over the protocol).
    if (const auto v = p.path("init_image")) {
      w->init_image = *v;
      g.init_image_path = w->init_image.c_str();
    }
    if (const auto v = p.path("mask")) {
      w->mask = *v;
      g.mask_path = w->mask.c_str();
    }
    if (const auto v = p.name("mask_encoding", mi::server::kMaskEncodingNames)) g.mask_encoding = *v;
    if (const auto v = p.name("resize", mi::server::kResizeNames)) g.resize = *v;
    if (const auto v = p.number("denoise_strength")) {
      if (*v < 0) throw RequestError("invalid_params", "\"denoise_strength\" must be a number in [0, 1]");
      g.denoise_strength = *v;
    }
    if (const auto h = p.hires("hires")) {
      g.hires_width = h->width;
      g.hires_height = h->height;
      g.hires_steps = h->steps;
      if (h->strength) g.hires_strength = *h->strength;
    }
    if (const auto v = p.path("distill")) {
      w->distill = *v;
      g.distill_dir = w->distill.c_str();
    }
    // Control adapter: paths only, like init_image.
    if (auto c = p.control("control")) {
      w->control_adapter = std::move(c->adapter);
      g.control_adapter = w->control_adapter.c_str();
      if (c->lora_strength) g.control_lora_strength = *c->lora_strength;
      // Every string is stored before the array points into them.
      for (Params::ControlImage& im : c->images) {
        w->control_strings.push_back(std::move(im.kind));
        w->control_strings.push_back(std::move(im.path));
      }
      for (std::size_t i = 0; i < c->images.size(); ++i) {
        mi_control_image ci{};
        ci.kind = w->control_strings[2 * i].c_str();
        ci.image_path = w->control_strings[2 * i + 1].c_str();
        ci.strength = c->images[i].strength.value_or(1.0);
        ci.start = c->images[i].start.value_or(0.0);
        ci.end = c->images[i].end.value_or(1.0);
        w->control_images.push_back(ci);
      }
      g.control_images = w->control_images.data();
      g.control_images_count = w->control_images.size();
    }
    if (const auto v = p.number("cfg_truncation")) g.cfg_truncation = *v;
    if (const auto v = p.name("step_cache", mi::server::kStepCacheNames)) g.step_cache = *v;
    if (const auto v = p.number("step_cache_threshold")) {
      // 0 is the C API's "default": the protocol spells that by leaving the parameter out.
      if (!(*v > 0 && *v <= 1)) throw RequestError("invalid_params", "\"step_cache_threshold\" must be a number in (0, 1]");
      g.step_cache_threshold = *v;
    }
    const auto preview_every = p.integer("preview_every", 1, kI32);
    if (const auto v = p.string("preview_output")) {
      if (v->empty()) throw RequestError("invalid_params", "\"preview_output\" must be a PNG path");
      w->preview = *v;
      g.preview_interval = static_cast<std::int32_t>(preview_every.value_or(1));
    } else if (preview_every) {
      throw RequestError("invalid_params", "\"preview_every\" needs \"preview_output\"");
    }
    // A preload may leave the prompt out (mi_engine_preload() plans for a short one); mi_gen_params_validate() checks
    // a generation's request, so it sees a stand-in prompt.
    mi_gen_params check = g;
    if (preload) check.prompt = "preload";
    if (const mi_status s = mi_gen_params_validate(&check); s != MI_OK) {
      throw RequestError(s == MI_ERR_INVALID_ARGUMENT ? "invalid_params" : status_code(s),
                         protocol_names(mi_last_error_message()));
    }
  }

  // Starts a job on the worker. The request is checked first (no engine or GPU needed), then the server's state.
  std::string start(std::unique_ptr<Work> w, const std::string& id, Kind kind) {
    if (engine_ == nullptr) throw RequestError("no_engine", "no model loaded; send \"load\" first");
    if (generating()) throw busy_error();
    join_worker();
    job_ = std::make_unique<Job>();
    job_->id = id;
    job_->kind = kind;
    w->job = job_.get();
    w->out = &out_;
    mi_gen_params& g = w->params;
    g.progress_callback = on_progress;
    g.plan_callback = on_plan;
    if (!w->preview.empty()) g.preview_callback = on_preview;
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
    const bool running = generating();
    const Kind kind = running ? job_->kind : Kind::kGenerate;
    const char* state = engine_ == nullptr        ? "unloaded"
                        : !running                ? "idle"
                        : kind == Kind::kPreload  ? "preloading"
                        : kind == Kind::kRelease  ? "releasing"
                                                  : "generating";
    std::string s = "{\"state\":\"" + std::string(state) + "\"";
    if (engine_ != nullptr) s += ",\"model\":" + str(model_) + ",\"mode\":\"" + mode_name(mode_) + "\"";
    if (running) {
      // generating_id names the running job of any kind (the field predates preload and release).
      s += ",\"generating_id\":" + job_->id;
      if (kind != Kind::kRelease) {
        const int phase = job_->phase.load();
        s += ",\"phase\":" +
             (phase < 0 ? std::string("null") : "\"" + std::string(phase_name(static_cast<mi_phase>(phase))) + "\"") +
             ",\"step\":" + std::to_string(job_->step.load()) + ",\"total_steps\":" + std::to_string(job_->total_steps.load()) +
             ",\"elapsed_seconds\":" + num(job_->elapsed.load()) + ",\"fraction\":" + num(job_->fraction.load()) +
             ",\"eta_seconds\":" + eta_json(job_->eta.load());
      }
      s += std::string(",\"cancelling\":") + tf(job_->cancel.load());
    }
    return s + "}";
  }

  std::string unload(const Params& p) {
    p.only({});
    if (generating()) throw busy_error();
    join_worker();
    const bool had = engine_ != nullptr;
    mi_engine_free(engine_);
    engine_ = nullptr;
    model_.clear();
    return std::string("{\"unloaded\":") + tf(had) + "}";
  }

  // One job's inputs; owned by the worker thread.
  struct Work {
    mi_gen_params params{};
    bool keep_host_copy = true;  // release
    std::string output, prompt, negative, dataspace, preview, init_image, mask, distill;
    std::vector<std::int32_t> uncond, slg;
    std::vector<std::string> lora_paths;
    std::vector<mi_lora> loras;  // points into lora_paths
    std::string control_adapter;
    std::vector<std::string> control_strings;    // kind, path of each control image
    std::vector<mi_control_image> control_images;  // points into control_strings
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
    // The whole-job fields come with the 56-byte struct (MI_CAP_PROGRESS_FRACTION).
    const bool whole = p->struct_size >= offsetof(mi_progress, eta_seconds) + sizeof p->eta_seconds;
    job.fraction.store(whole ? p->fraction : 0.0);
    job.eta.store(whole ? p->eta_seconds : -1.0);
    w.out->emit("{\"event\":\"progress\",\"id\":" + job.id + ",\"phase\":\"" + phase_name(p->phase) + "\",\"step\":" +
                std::to_string(p->step) + ",\"total_steps\":" + std::to_string(p->total_steps) +
                ",\"step_seconds\":" + num(p->step_seconds) + ",\"elapsed_seconds\":" + num(p->elapsed_seconds) +
                ",\"fraction\":" + num(job.fraction.load()) + ",\"eta_seconds\":" + eta_json(job.eta.load()) + "}");
  }

  static void on_plan(const char* explanation, const char* plan_json, void* user) {
    Work& w = *static_cast<Work*>(user);
    w.out->emit("{\"event\":\"plan\",\"id\":" + w.job->id + ",\"explanation\":" + str(explanation) +
                ",\"plan\":" + one_line(plan_json) + "}");
  }

  // Rewrites the preview PNG, then announces it. A failed write is logged; the generation goes on.
  static void on_preview(const std::uint8_t* rgb, std::int32_t width, std::int32_t height, std::int32_t step, void* user) {
    Work& w = *static_cast<Work*>(user);
    Job& job = *w.job;
    if (mi_write_png(w.preview.c_str(), rgb, width, height) != MI_OK) {
      w.out->log("warning", "preview not written: " + std::string(mi_last_error_message()), job.id);
      return;
    }
    w.out->emit("{\"event\":\"preview\",\"id\":" + job.id + ",\"step\":" + std::to_string(step) + ",\"total_steps\":" +
                std::to_string(job.total_steps.load()) + ",\"image\":" + str(w.preview) + ",\"width\":" +
                std::to_string(width) + ",\"height\":" + std::to_string(height) + "}");
  }

  // done.warnings: the LoRA and control adapter warnings, lossy accelerators and settings without effect, prompt
  // truncation, for launchers that do not read the report. `g` is the request.
  static std::string warnings_json(const mi_result* r, const mi_gen_params& g) {
    std::string out;
    const auto add = [&](const std::string& w) { out += (out.empty() ? "" : ",") + str(w); };
    if (r != nullptr) {
      // The active accelerators, as MI_TEXT_ACCELERATORS names them: CFG truncation counts only with CFG.
      const std::string active = mi_result_text(r, MI_TEXT_ACCELERATORS);
      const bool truncating = active.find("cfg_truncation") != std::string::npos;
      const bool caching = active.find("step_cache") != std::string::npos;
      for (const auto& [kind, prefix] : {std::pair{MI_MESSAGE_LORA_WARNING, "LoRA "},
                                         std::pair{MI_MESSAGE_CONTROL_WARNING, "control adapter LoRA "}}) {
        const std::int32_t n = mi_result_message_count(r, kind);
        for (std::int32_t i = 0; i < n; ++i) {
          if (const char* m = mi_result_message(r, kind, i)) add(prefix + std::string(m));
        }
      }
      // The tier reason says it too, but only in the report: the accelerators are the one opt-in that trades quality.
      mi_accel_info a{};
      a.struct_size = sizeof a;
      if (mi_result_accel(r, &a) == MI_OK && a.lossy) {
        std::string what;
        if (truncating) what = "CFG truncated on " + std::to_string(a.cfg_truncated_steps) + " steps";
        if (caching) {
          what += (what.empty() ? "" : ", ") + std::string("step cache reused ") + std::to_string(a.cached_calls) + " of " +
                  std::to_string(a.dit_calls) + " DiT calls";
        }
        add("lossy accelerators (" + active + "), uncertified: " + what);
      }
      // What the engine says about the accelerators for this model (a threshold that moves the base model's
      // composition, a cache that never engaged).
      const std::int32_t n = mi_result_message_count(r, MI_MESSAGE_ACCEL_WARNING);
      for (std::int32_t i = 0; i < n; ++i) {
        if (const char* m = mi_result_message(r, MI_MESSAGE_ACCEL_WARNING, i)) add(m);
      }
      // Settings without effect, as `moss-infer generate` warns about them.
      if (g.step_cache_threshold != 0 && g.step_cache == MI_STEP_CACHE_OFF) {
        add("step_cache_threshold has no effect without step_cache \"fbcache\"");
      }
      if (g.cfg_truncation < 1.0 && !truncating) add("cfg_truncation has no effect without CFG (cfg 1)");
      for (const std::int32_t negative : {0, 1}) {
        mi_prompt_tokens t{};
        t.struct_size = sizeof t;
        if (mi_result_prompt_tokens(r, negative, &t) != MI_OK || !t.truncated) continue;
        add(std::string(negative ? "negative prompt" : "prompt") + " truncated to " + std::to_string(t.kept_tokens) + " of " +
            std::to_string(t.total_tokens) + " tokens (" + std::to_string(t.dropped_tokens) +
            " dropped; max_sequence_length " + std::to_string(t.max_length) + ")");
      }
    }
    if (r != nullptr) {
      if (const char* note = mi_result_text(r, MI_TEXT_IMAGE_NOTE); note != nullptr && *note != '\0') {
        add(std::string("image input: ") + note);
      }
      if (const char* note = mi_result_text(r, MI_TEXT_TE_NOTE); note != nullptr && *note != '\0') {
        add(std::string("text encoder: ") + note);
      }
    }
    return "[" + out + "]";
  }

  // done's dit_load summary: {"source": "files" | "host_copy" | null, "io": "direct" | "buffered" | null, "bytes",
  // "seconds", "note"}.
  static std::string dit_load_json(const mi_result* r) {
    mi_dit_load_info d{};
    d.struct_size = sizeof d;
    if (mi_result_dit_load(r, &d) != MI_OK) return "null";
    const char* source = d.source == MI_DIT_LOAD_SOURCE_FILES       ? "\"files\""
                         : d.source == MI_DIT_LOAD_SOURCE_HOST_COPY ? "\"host_copy\""
                                                                    : "null";
    const char* io = d.io == MI_LOAD_IO_DIRECT ? "\"direct\"" : d.io == MI_LOAD_IO_BUFFERED ? "\"buffered\"" : "null";
    return std::string("{\"source\":") + source + ",\"io\":" + io + ",\"bytes\":" + std::to_string(d.bytes) +
           ",\"seconds\":" + num(d.seconds) + ",\"note\":" + str(mi_result_text(r, MI_TEXT_DIT_LOAD_NOTE)) + "}";
  }

  // release: the response, sent when the release ends. A cancel that lands before the release frees anything stops it
  // whole ("cancelled": true, nothing released); one during the DiT's host copy stops the copy (between 64 MiB reads),
  // and the release still frees the device memory, keeps a host-resident text encoder's copy and frees nothing else.
  // mi_cancel() only stops a running job, and a release has no callback in which a cancel issued before
  // mi_engine_release() marked the engine busy could land (as on_progress does for generate and preload). A relay thread
  // sleeps until request_cancel() wakes it, then repeats mi_cancel() every 10 ms until the release ends, so a cancel
  // lands at the latest before the copy's first read.
  void run_release(Work& w) {
    Job& job = *w.job;
    mi_release_info info{};
    info.struct_size = sizeof info;
    {
      const std::lock_guard<std::mutex> lock(relay_mu_);
      relay_finished_ = false;
    }
    std::thread relay;
    if (w.keep_host_copy) {
      try {
        relay = std::thread([this, &job] {
          std::unique_lock<std::mutex> lock(relay_mu_);
          relay_cv_.wait(lock, [&] { return relay_finished_ || job.cancel.load(); });
          while (!relay_finished_) {
            lock.unlock();
            mi_cancel(engine_);  // a no-op until the release marks the engine busy
            lock.lock();
            relay_cv_.wait_for(lock, std::chrono::milliseconds(10), [&] { return relay_finished_; });
          }
        });
      } catch (const std::system_error&) {
        // No thread: a cancel still reaches a release already running (request_cancel()'s own mi_cancel()).
      }
    }
    const mi_status s = mi_engine_release(engine_, w.keep_host_copy ? 1 : 0, &info);
    {
      const std::lock_guard<std::mutex> lock(relay_mu_);
      relay_finished_ = true;
    }
    relay_cv_.notify_all();
    if (relay.joinable()) relay.join();
    std::string line = "{\"id\":" + job.id;
    if (s == MI_OK || s == MI_CANCELLED) {
      // cancelled: the cancel landed before the release freed anything (the engine still holds its device memory).
      line += ",\"result\":{\"cancelled\":" + std::string(tf(s == MI_CANCELLED)) + ",\"seconds\":" + num(info.seconds) +
              ",\"vram_released_bytes\":" +
              std::to_string(info.vram_released_bytes) + ",\"host_copy_bytes\":" + std::to_string(info.host_copy_bytes) +
              ",\"host_copy_seconds\":" + num(info.host_copy_seconds) + ",\"dit_host_copy\":" + tf(info.dit_host_copy) +
              ",\"te_host_copy\":" + tf(info.te_host_copy) + ",\"note\":" + str(info.note) + "}}";
    } else {
      line += ",\"error\":" + error_json(status_code(s), protocol_names(mi_last_error_message())) + "}";
    }
    running_.store(false);  // idle before the answer goes out, as for done
    out_.emit(line);
  }

  void run(Work& w) {
    w.engine = engine_;
    Job& job = *w.job;
    if (job.kind == Kind::kRelease) return run_release(w);
    const bool preload = job.kind == Kind::kPreload;
    mi_result* r = nullptr;
    const mi_status s = preload ? mi_engine_preload(engine_, &w.params, &r) : mi_generate(engine_, &w.params, &r);
    std::string done = "{\"event\":\"done\",\"id\":" + job.id;
    if (s == MI_OK && preload) {
      done += ",\"status\":\"ok\",\"report\":" + one_line(mi_result_report_json(r));
    } else if (s == MI_OK) {
      if (const mi_status ws = mi_save_png(r, w.output.c_str()); ws == MI_OK) {
        done += ",\"status\":\"ok\",\"image\":" + str(w.output) + ",\"report\":" + one_line(mi_result_report_json(r));
      } else {
        done += ",\"status\":\"error\",\"error\":" + error_json(status_code(ws), protocol_names(mi_last_error_message()));
        done += ",\"report\":" + one_line(mi_result_report_json(r));  // after the message: a failing report would replace it
      }
    } else if (s == MI_CANCELLED) {
      done += ",\"status\":\"cancelled\",\"steps_run\":" + std::to_string(mi_result_steps_run(r)) +
              ",\"report\":" + one_line(mi_result_report_json(r));
    } else {
      done += ",\"status\":\"error\",\"error\":" + error_json(status_code(s), protocol_names(mi_last_error_message()));
    }
    if (r != nullptr && preload) {
      // What the preload loaded, without parsing the report (which has both too).
      done += ",\"preload_note\":" + str(mi_result_text(r, MI_TEXT_PRELOAD_NOTE)) + ",\"dit_load\":" + dit_load_json(r);
    }
    if (r != nullptr) {
      // Whether the run as a whole is certified (the report's models.dit.certified), without parsing the report.
      mi_tier_info t{};
      t.struct_size = sizeof t;
      if (mi_result_tier(r, &t) == MI_OK) done += std::string(",\"certified\":") + tf(t.run_certified != 0);
    }
    done += ",\"warnings\":" + warnings_json(r, w.params);
    mi_result_free(r);
    // Idle before the done line goes out: a client may send the next generate as soon as it reads it.
    running_.store(false);
    out_.emit(done + "}");
  }

  void request_cancel() {
    if (!generating()) return;
    job_->cancel.store(true);
    mi_cancel(engine_);
    {
      const std::lock_guard<std::mutex> lock(relay_mu_);  // a release's relay sees the flag, or is woken
    }
    relay_cv_.notify_all();
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
  // A release's cancel relay (run_release): woken by request_cancel() and by the release's end.
  std::mutex relay_mu_;
  std::condition_variable relay_cv_;
  bool relay_finished_ = false;  // guarded by relay_mu_ while a relay runs
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
