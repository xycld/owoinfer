// moss-server: the engine as a sidecar process for launchers. JSON lines over stdin/stdout (docs/server-protocol.md),
// built only on the public C API (mossinfer.h).
//
// Threads: the main thread reads and answers requests; a job (generate, preload, release) runs on one worker thread, so
// cancel and status are answered while it runs; an inspect runs on a short-lived thread of its own. An open component
// session (session_open .. session_close) has one persistent worker thread of its own that runs its calls in order. Every protocol
// line goes through emit() under one mutex. stdout carries protocol lines only: at startup the protocol stream gets
// a private, non-inheritable copy of fd 1 and fd 1 is pointed at stderr (or at the null device when stderr is
// unusable), so anything else printing to stdout (the engine, a library) never reaches the protocol stream.
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <span>
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
#include "shm.h"
#include "util/json.h"
#include "util/path.h"
#include "util/regular_file.h"

namespace {

// 2 since moss-server 2.0: load's tier is required and done has no certified (docs/server-protocol.md).
constexpr int kProtocolVersion = 2;
constexpr std::size_t kMaxLineBytes = std::size_t{1} << 20;  // a request is a prompt and a few numbers
constexpr std::size_t kMaxJsonValues = 1 << 16;

using mi::json::Value;

// ---------------------------------------------------------------- JSON output

using mi::server::num;
using mi::server::one_line;
using mi::server::str;
using mi::server::utf8_sequence_length;

const char* tf(bool b) { return b ? "true" : "false"; }

// mi_progress.fraction and eta_seconds: null when unknown (negative), or not in the library's struct.
std::string or_null(double v) { return v < 0 ? "null" : num(v); }

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
using Renames = std::span<const std::pair<std::string_view, std::string_view>>;

// generate's, preload's and load's parameters.
constexpr std::pair<std::string_view, std::string_view> kGenerateRenames[] = {
    {"dataspace_path", "dataspace"}, {"init_image_path", "init_image"}, {"mask_path", "mask"},
    {"mi_engine_params.distill_dir", "load's \"distill\""}, {"distill_dir", "distill"},
    {"hires_width / hires_height", "hires"}, {"hires_width and hires_height", "hires width and height"},
    {"with MI_MASK_BINARY01", "with mask_encoding \"binary01\""},
    // "control_images[0].image_path" -> "control.images[0].path".
    {"control_images", "control.images"}, {"reference_images", "references.images"},
    {"reference_max_edge", "references.max_edge"}, {".image_path", ".path"}, {"control_adapter", "control.adapter"},
    {"control_lora_strength", "control.lora_strength"}};

// The component methods' parameters and methods (their params keep the C fields' names, control_images and
// control_adapter included).
constexpr std::pair<std::string_view, std::string_view> kComponentRenames[] = {
    {"MI_RUN_CROPS", "\"crops\""}, {"MI_RUN_EXACT_SIGMA", "\"exact_sigma\""},
    {"MI_STEP_CONTROL_CHANGED", "\"control_changed\""}, {"mi_session_params.plan_request", "\"plan_request\""},
    {"mi_run_info.staging_bytes", "info.staging_bytes"}, {"StepArgs::rows", "\"rows\""},
    {"mi_encode_text", "encode_text"}, {"mi_run_end", "run_end"}, {"mi_run_begin", "run_begin"},
    {"mi_session_close", "session_close"}, {"mi_generate()", "generate"}, {"mi_generate", "generate"},
    {"mi_run_staging", "the run region"}, {".image_path", ".path"}};


std::string protocol_names(std::string message, Renames renames = kGenerateRenames) {
  const auto token_at = [&](std::size_t at, std::string_view from) {
    const bool starts = from.front() == '.' ? at > 0 && message[at - 1] == ']'
                                            : at == 0 || std::string_view(" \"'([").find(message[at - 1]) != std::string_view::npos;
    const std::size_t end = at + from.size();
    if (!starts) return false;
    if (end == message.size()) return true;
    if (message[end] == '.') return end + 1 == message.size() || message[end + 1] == ' ';
    return std::string_view(" \"',:;)[]").find(message[end]) != std::string_view::npos;
  };
  for (const auto& [from, to] : renames) {
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
    members(*v, k0, {"adapter", "lora_strength", "images"});
    Control c;
    bool have_adapter = false;
    for (const auto& [k, m] : v->as_object()) {
      if (k == "adapter") {
        c.adapter = nonempty(m, k0 + ".adapter", "the control adapter file");
        have_adapter = true;
      } else if (k == "lora_strength") {
        c.lora_strength = numeric(m, k0 + ".lora_strength");
      } else {
        c.images = control_image_list(m, k0 + ".images");
      }
    }
    if (!have_adapter) throw RequestError("invalid_params", "\"" + k0 + ".adapter\" (the control adapter file) is required");
    return c;
  }
  // run_begin's `control_images`: control's "images" array on its own.
  std::optional<std::vector<ControlImage>> control_images(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    return control_image_list(*v, std::string(key));
  }
  // `references`: {"images": [path, ...] (1 or more, in order), "max_edge": integer (optional)}.
  struct References {
    std::vector<std::string> images;
    std::optional<std::int64_t> max_edge;
  };
  std::optional<References> references(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) return std::nullopt;
    const std::string k0(key);
    if (!v->is_object()) throw bad(key, "an object {\"images\", \"max_edge\"?}");
    References r;
    bool have_images = false;
    for (const auto& [k, m] : v->as_object()) {
      if (k == "images") {
        if (have_images) throw RequestError("invalid_params", "\"" + k0 + "\" has \"images\" more than once");
        have_images = true;
        if (!m.is_array() || m.as_array().empty()) throw bad(k0 + ".images", "a non-empty array of image paths");
        for (const Value& e : m.as_array()) {
          const std::string item = k0 + ".images[" + std::to_string(r.images.size()) + "]";
          if (!e.is_string() || e.as_string().empty()) throw bad(item, "a non-empty string (an image file)");
          if (e.as_string().find('\0') != std::string::npos) throw bad(item, "a string without NUL characters");
          r.images.push_back(e.as_string());
        }
      } else if (k == "max_edge") {
        if (r.max_edge) throw RequestError("invalid_params", "\"" + k0 + "\" has \"max_edge\" more than once");
        if (!m.is_number() || m.as_double() != std::floor(m.as_double()) || m.as_double() < 16 || m.as_double() > 2048) {
          throw bad(k0 + ".max_edge", "an integer from 16 to 2048");
        }
        r.max_edge = static_cast<std::int64_t>(m.as_double());
      } else {
        throw RequestError("invalid_params", "\"" + k0 + "\" has an unknown member \"" + k + "\"");
      }
    }
    if (!have_images) throw RequestError("invalid_params", "\"" + k0 + ".images\" is required");
    return r;
  }

 private:
  static RequestError bad(std::string_view key, const std::string& what) {
    return RequestError("invalid_params", "\"" + std::string(key) + "\" must be " + what);
  }
  // Each member of an object once, and only the known ones.
  static void members(const mi::json::Value& obj, const std::string& field, std::initializer_list<std::string_view> known) {
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
  }
  // A non-empty string without NUL characters.
  static std::string nonempty(const mi::json::Value& m, const std::string& field, const std::string& what) {
    if (!m.is_string() || m.as_string().empty()) throw bad(field, "a non-empty string (" + what + ")");
    if (m.as_string().find('\0') != std::string::npos) throw bad(field, "a string without NUL characters");
    return m.as_string();
  }
  static double numeric(const mi::json::Value& m, const std::string& field) {
    if (!m.is_number()) throw bad(field, "a number");
    return m.as_double();
  }
  // [{"kind": string, "path": path, "strength", "start", "end": numbers (optional)}], named `field` in messages.
  static std::vector<ControlImage> control_image_list(const mi::json::Value& m, const std::string& field) {
    if (!m.is_array()) throw bad(field, "an array of {\"kind\", \"path\", \"strength\"?, \"start\"?, \"end\"?} objects");
    std::vector<ControlImage> images;
    for (const Value& e : m.as_array()) {
      const std::string item = field + "[" + std::to_string(images.size()) + "]";
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
      images.push_back(std::move(im));
    }
    return images;
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
  for (const auto& n : mi::server::kModeNames) {
    if (n.value == m) return n.name.data();  // a string literal
  }
  return "unknown";
}

const char* precision_name(mi_precision p) {
  switch (p) {
    case MI_PRECISION_BF16: return "bf16";
    case MI_PRECISION_INT8: return "int8";
    case MI_PRECISION_W4A8: return "w4a8";
    case MI_PRECISION_W4A4: return "w4a4";
    case MI_PRECISION_NVFP4: return "nvfp4";
    case MI_PRECISION_MAX_ENUM: break;
  }
  return "unknown";
}

const char* file_kind_name(mi_file_kind k) {
  switch (k) {
    case MI_FILE_KIND_UNKNOWN: return "unknown";
    case MI_FILE_KIND_DIT: return "dit";
    case MI_FILE_KIND_TEXT_ENCODER: return "text_encoder";
    case MI_FILE_KIND_VAE: return "vae";
    case MI_FILE_KIND_LORA: return "lora";
    case MI_FILE_KIND_CONTROL_ADAPTER: return "control_adapter";
    case MI_FILE_KIND_MAX_ENUM: break;
  }
  return "unknown";
}

// ---------------------------------------------------------------- server

// Test seam, for tests/server/test_protocol.py only (not a user option, not in the protocol's documentation): with
// MOSSINFER_SERVER_TEST_REFUSE_PIN=1 in the server's environment, run_begin answers as if the system refused to pin the
// run's region (mi_run_adopt_staging's MI_ERR_NOT_SUPPORTED), so the steps' copy path runs where pinning works.
bool test_refuse_pin() {
#ifdef _WIN32
  char v[2] = {};
  return GetEnvironmentVariableA("MOSSINFER_SERVER_TEST_REFUSE_PIN", v, sizeof v) == 1 && v[0] == '1';
#else
  const char* v = std::getenv("MOSSINFER_SERVER_TEST_REFUSE_PIN");
  return v != nullptr && std::string_view(v) == "1";
#endif
}

class Server {
 public:
  explicit Server(Output& out) : out_(out), refuse_pin_(test_refuse_pin()) {}
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
      std::optional<std::string> result;  // none: answered later, by the worker (release) or an inspect's thread
      bool keep_running = true;
      if (m == "hello") result = hello(params);
      else if (m == "load") result = load(params, id);
      else if (m == "generate") result = generate(params, id);
      else if (m == "preload") result = preload(params, id);
      else if (m == "release") result = release(params, id);
      else if (m == "cancel") result = cancel(params);
      else if (m == "status") result = status(params);
      else if (m == "inspect") result = inspect(params, id);
      else if (m == "check") result = check(params, id);
      else if (m == "unload") result = unload(params);
      else if (m == "session_open") result = session_open(params);
      else if (m == "session_close") result = session_close(params, id);
      else if (m == "shm_alloc") result = shm_alloc(params, id);
      else if (m == "shm_free") result = shm_free(params);
      else if (m == "encode_text") result = encode_text(params, id);
      else if (m == "conditioning_import") result = conditioning_import(params, id);
      else if (m == "conditioning_free") result = conditioning_free(params, id);
      else if (m == "initial_noise") result = initial_noise(params, id);
      else if (m == "run_begin") result = run_begin(params, id);
      else if (m == "denoise_step") result = denoise_step(params, id);
      else if (m == "run_end") result = run_end(params, id);
      else if (m == "vae_encode") result = vae_encode(params, id);
      else if (m == "vae_decode") result = vae_decode(params, id);
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

  // Cancels a running job, closes an open session (its calls cancelled), waits for the job and for the inspects (they
  // have no cancel: each reads one file's header and small tensors), and releases the engine.
  void stop() {
    request_cancel();  // first: the session's close waits for a running generation without cancelling it
    close_session(nullptr);
    join_worker();
    join_inspections();
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
    std::atomic<double> fraction{0}, eta{-1};  // < 0: unknown
  };
  struct Work;

  // One inspect's thread; `done` once its answer is out.
  struct Inspection {
    std::thread thread;
    std::atomic<bool> done{false};
  };
  static constexpr std::size_t kMaxInspects = 2;
  // The images a check may decode while a job runs (the decode's host memory and time next to the job's).
  static constexpr std::uint64_t kCheckPixelsBusy = 16'000'000;

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
    // The accepted names of every enum parameter, from the tables the parameters are validated against.
    std::string values;
    for (const mi::server::ValueTable& t : mi::server::kValueTables) {
      std::string names;
      for (const std::string_view n : t.names) names += (names.empty() ? "" : ",") + str(n);
      values += (values.empty() ? "" : ",") + str(t.param) + ":[" + names + "]";
    }
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
           tf(caps & MI_CAP_PROGRESS_FRACTION) + ",\"inspect\":" + tf(caps & MI_CAP_INSPECT) +
           ",\"references\":" + tf(caps & MI_CAP_REFERENCE_IMAGES) + ",\"check\":" + tf(caps & MI_CAP_CHECK) +
           ",\"components\":" + tf(caps & MI_CAP_COMPONENTS) +
           "},\"values\":{" + values + "},\"probe\":{\"driver_cuda_version\":" +
           std::to_string(probe.driver_cuda_version) + ",\"runtime_cuda_version\":" + std::to_string(probe.runtime_cuda_version) +
           ",\"driver_compatible\":" + tf(probe.driver_compatible) + ",\"device_count\":" + std::to_string(probe.device_count) +
           ",\"supported_device_count\":" + std::to_string(probe.supported_device_count) +
           ",\"build_flavour\":" + str(probe.build_flavour) + ",\"recommended_flavour\":" + str(probe.recommended_flavour) +
           ",\"message\":" + str(probe.message) + ",\"cublaslt\":{\"source\":" + str(probe.cublaslt_source) +
           ",\"version\":" + std::to_string(probe.cublaslt_version) + ",\"path\":" + str(probe.cublaslt_path) +
           ",\"error\":" + str(probe.cublaslt_error) + "}" + (with_devices ? ",\"devices\":[" + devices + "]" : std::string()) + "}}";
  }

  // load's parameters (also check's "load"): the strings, then the struct pointing into them (never copied).
  struct EngineRequest {
    mi_engine_params e{};
    std::optional<std::string> model, cache_dir, dit_file, te_file, vae_file, te, distill;
  };
  static void engine_request(const Params& p, EngineRequest& r) {
    p.only({"model", "mode", "device", "vram_budget_bytes", "vram_reserve_bytes", "host_ram_budget_bytes", "cache_dir",
            "embedding_cache", "direct_io", "block_path", "dit_file", "text_encoder_file", "vae_file", "tier", "attention",
            "lora_strict", "time_embedding_file", "distill", "certificate_dirs", "dit_direct_io", "te_host_pin"});
    mi_engine_params& e = r.e;
    if (const mi_status s = mi_engine_params_init(&e, sizeof e); s != MI_OK) throw_status(s);
    r.model = p.string("model");
    if (!r.model) throw RequestError("invalid_params", "\"model\" (bundle directory) is required");
    e.model_path = r.model->c_str();
    if (const auto v = p.name("mode", mi::server::kModeNames)) e.mode = *v;
    constexpr std::int64_t kMaxBytes = std::numeric_limits<std::int64_t>::max();
    if (const auto v = p.integer("device", 0, 1024)) e.device_index = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("vram_budget_bytes", 0, kMaxBytes)) e.vram_budget_bytes = static_cast<std::uint64_t>(*v);
    if (const auto v = p.integer("vram_reserve_bytes", 0, kMaxBytes)) e.vram_reserve_bytes = static_cast<std::uint64_t>(*v);
    if (const auto v = p.integer("host_ram_budget_bytes", 0, kMaxBytes)) e.host_ram_budget_bytes = static_cast<std::uint64_t>(*v);
    r.cache_dir = p.string("cache_dir");
    if (r.cache_dir) e.cache_dir = r.cache_dir->c_str();
    if (const auto v = p.boolean("embedding_cache")) e.embedding_cache = *v ? 1 : 0;
    if (const auto v = p.boolean("direct_io")) e.direct_io = *v ? 1 : 0;
    if (const auto v = p.name("block_path", mi::server::kBlockPathNames)) e.block_path = *v;
    r.dit_file = p.string("dit_file");
    r.te_file = p.string("text_encoder_file");
    r.vae_file = p.string("vae_file");
    if (r.dit_file) e.dit_file = r.dit_file->c_str();
    if (r.te_file) e.text_encoder_file = r.te_file->c_str();
    if (r.vae_file) e.vae_file = r.vae_file->c_str();
    const std::optional<mi_tier> tier = p.name("tier", mi::server::kTierNames);
    if (const auto v = p.name("attention", mi::server::kAttentionNames)) e.attention = *v;
    if (const auto v = p.boolean("lora_strict")) e.lora_strict = *v ? 1 : 0;
    r.te = p.path("time_embedding_file");
    r.distill = p.path("distill");
    if (r.te) e.time_embedding_file = r.te->c_str();
    if (r.distill) e.distill_dir = r.distill->c_str();
    // "certificate_dirs": reserved since 2.0, accepted and ignored.
    if (const auto v = p.name("dit_direct_io", mi::server::kDitDirectIoNames)) e.dit_direct_io = *v;
    if (const auto v = p.name("te_host_pin", mi::server::kTeHostPinNames)) e.te_host_pin = *v;
    // Required since 2.0 (the C API refuses MI_TIER_AUTO); checked once every parameter is valid.
    if (!tier) throw RequestError("invalid_params", "\"tier\" is required: reference, high or turbo");
    e.tier = *tier;
  }

  std::string load(const Params& p, const std::string& id) {
    EngineRequest r;
    engine_request(p, r);
    if (generating()) throw busy_error();
    const mi_engine_params& e = r.e;
    const std::string& model = *r.model;
    // The old engine goes first: two engines would hold two arenas. Its session is closed first.
    close_session(nullptr);
    join_worker();
    mi_engine_free(engine_);
    engine_ = nullptr;
    model_.clear();
    mi_engine* engine = nullptr;
    if (const mi_status s = mi_engine_create(&e, &engine); s != MI_OK) {
      throw RequestError(status_code(s), protocol_names(mi_last_error_message()));
    }
    engine_ = engine;
    model_ = model;
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
            "references", "lora_placement",
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
            "references", "lora_placement",
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
    // Reference images of an edit adapter: paths only, like init_image (read by the library when the job runs).
    if (auto r = p.references("references")) {
      w->reference_paths = std::move(r->images);
      for (const std::string& path : w->reference_paths) {
        mi_reference_image ri{};
        ri.image_path = path.c_str();
        w->references.push_back(ri);
      }
      g.reference_images = w->references.data();
      g.reference_images_count = w->references.size();
      if (r->max_edge) g.reference_max_edge = static_cast<std::int32_t>(*r->max_edge);
    }
    if (const auto v = p.number("cfg_truncation")) g.cfg_truncation = *v;
    if (const auto v = p.name("step_cache", mi::server::kStepCacheNames)) g.step_cache = *v;
    if (const auto v = p.name("lora_placement", mi::server::kLoraPlacementNames)) g.lora_placement = *v;
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
    // A job and a session call never run at once (the engine has one cancel flag: a session's cancel must never reach
    // a job, nor a job's a session call).
    if (session_ != nullptr && session_busy(*session_)) {
      throw RequestError("busy", "a component session call is queued or running; send this after its answer");
    }
    join_worker();
    job_ = std::make_unique<Job>();
    job_->id = id;
    job_->kind = kind;
    // 0 until the first progress event; null throughout with a library without the whole-job fields.
    if ((mi_capabilities() & MI_CAP_PROGRESS_FRACTION) == 0) job_->fraction.store(-1.0);
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
    // A session call: the named one (queued or running), or without an id the running one when no job runs.
    if (session_ != nullptr && (want != nullptr || !generating())) {
      const std::string wanted = want != nullptr ? id_json(want) : std::string();
      if (const std::optional<std::string> hit = cancel_session_call(want != nullptr ? &wanted : nullptr)) {
        return "{\"cancelling\":true,\"generating_id\":" + *hit + "}";
      }
    }
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
             ",\"elapsed_seconds\":" + num(job_->elapsed.load()) + ",\"fraction\":" + or_null(job_->fraction.load()) +
             ",\"eta_seconds\":" + or_null(job_->eta.load());
      }
      s += std::string(",\"cancelling\":") + tf(job_->cancel.load());
    }
    if (session_ != nullptr) s += ",\"session\":" + session_status(*session_);
    return s + "}";
  }

  // Needs no engine and touches none, so it is answered in any state. mi_inspect_file() reads files (a slow disk, a
  // network share), so it runs on a thread of its own and answers when done: cancel and status stay immediate. At most
  // kMaxInspects run at a time; another one is refused with busy.
  std::optional<std::string> inspect(const Params& p, const std::string& id) {
    p.only({"path"});
    const std::optional<std::string> path = p.path("path");
    if (!path) throw RequestError("invalid_params", "\"path\" (the file to inspect) is required");
    return on_thread(inspections_, kMaxInspects, id, "inspect", [file = *path] { return inspect_json(file); });
  }

  // check: load's and generate's parameters checked by mi_check() (no engine of the server's is touched), on its own
  // thread as an inspect (they share the slots).
  std::optional<std::string> check(const Params& p, const std::string& id) {
    p.only({"load", "generate"});
    if (p.find("load") == nullptr) throw RequestError("invalid_params", "\"load\" (load's parameters) is required");
    if (p.find("generate") == nullptr) throw RequestError("invalid_params", "\"generate\" (generate's parameters) is required");
    // The parameters are read on the thread: Params point into the request line's document. Every file access (the
    // image headers too) happens there, never on the request loop.
    auto doc = std::make_shared<Value>(*p.find("load"));
    auto gen = std::make_shared<Value>(*p.find("generate"));
    return on_thread(checks_, 1, id, "check", [this, doc, gen] {
      // A check decodes its image inputs: next to a running job, only up to kCheckPixelsBusy pixels in all.
      if (generating()) {
        const std::uint64_t pixels = image_pixels(*gen);
        if (pixels > kCheckPixelsBusy) {
          throw RequestError("busy", "a job is running and this check would decode " + std::to_string(pixels / 1000000) +
                                         " MP of images (at most " + std::to_string(kCheckPixelsBusy / 1000000) +
                                         " MP while a job runs); send it again after the job's done event");
        }
      }
      return check_json(Params(doc.get()), Params(gen.get()));
    });
  }

  // Pixels of the images a check would decode (generate's init_image, mask, control.images[].path,
  // references.images[]), from the PNG / JPEG headers; a file that is neither (or unreadable) counts 0: its decode
  // fails at once.
  static std::uint64_t image_pixels(const Value& gen) {
    std::vector<std::string> paths;
    const auto add = [&](const Value* v) {
      if (v != nullptr && v->is_string()) paths.push_back(v->as_string());
    };
    if (!gen.is_object()) return 0;
    add(gen.find("init_image"));
    add(gen.find("mask"));
    if (const Value* c = gen.find("control"); c != nullptr && c->is_object()) {
      if (const Value* im = c->find("images"); im != nullptr && im->is_array()) {
        for (const Value& e : im->as_array()) {
          if (e.is_object()) add(e.find("path"));
        }
      }
    }
    if (const Value* r = gen.find("references"); r != nullptr && r->is_object()) {
      if (const Value* im = r->find("images"); im != nullptr && im->is_array()) {
        for (const Value& e : im->as_array()) add(&e);
      }
    }
    std::uint64_t total = 0;
    for (const std::string& p : paths) total += image_header_pixels(p);
    return total;
  }

  // width * height from a PNG's IHDR or a JPEG's first SOF marker; 0 when neither is found in the first 64 KiB. The
  // path passes the boundary check and the engine's open of user files (util::check_user_path, open_regular_file: a
  // device, pipe or other non-regular file is never opened, nor blocked on); what they refuse counts 0 pixels (the
  // check itself refuses it).
  static std::uint64_t image_header_pixels(const std::string& path_utf8) {
    const std::filesystem::path path = mi::util::path_from_utf8(path_utf8);
    std::string b(std::size_t{64} << 10, '\0');
    std::size_t got = 0;
    try {
      mi::util::check_user_path(path, "image");
#ifdef _WIN32
      void* h = nullptr;
      if (mi::util::open_regular_file(path, 0, h) != mi::util::OpenStatus::kOk) return 0;
      DWORD n = 0;
      const BOOL ok = ReadFile(static_cast<HANDLE>(h), b.data(), static_cast<DWORD>(b.size()), &n, nullptr);
      CloseHandle(static_cast<HANDLE>(h));
      if (!ok) return 0;
      got = n;
#else
      int fd = -1;
      if (mi::util::open_regular_file(path, 0, fd) != mi::util::OpenStatus::kOk) return 0;
      while (got < b.size()) {
        const ssize_t n = ::read(fd, b.data() + got, b.size() - got);
        if (n <= 0) break;
        got += static_cast<std::size_t>(n);
      }
      ::close(fd);
#endif
    } catch (const std::exception&) {
      return 0;
    }
    b.resize(got);
    const auto byte = [&](std::size_t i) { return static_cast<std::uint64_t>(static_cast<unsigned char>(b[i])); };
    const auto be = [&](std::size_t i, int n) {
      std::uint64_t v = 0;
      for (int k = 0; k < n; ++k) v = (v << 8) | byte(i + static_cast<std::size_t>(k));
      return v;
    };
    if (b.size() >= 24 && b.compare(0, 8, "\x89PNG\r\n\x1a\n") == 0) return be(16, 4) * be(20, 4);
    if (b.size() < 4 || byte(0) != 0xFF || byte(1) != 0xD8) return 0;
    for (std::size_t i = 2; i + 9 < b.size();) {
      if (byte(i) != 0xFF) return 0;
      const std::uint64_t marker = byte(i + 1);
      if (marker == 0xFF) {
        ++i;
        continue;
      }
      const bool sof = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
      if (sof) return be(i + 5, 2) * be(i + 7, 2);
      i += 2 + static_cast<std::size_t>(be(i + 2, 2));
    }
    return 0;
  }

  // check's result: the first refusal of load then generate, as each would answer it, else the plan.
  std::string check_json(const Params& load_params, const Params& gen_params) {
    const auto refusal = [](const char* by, const std::string& code, const std::string& message) {
      return "{\"ok\":false,\"refused_by\":\"" + std::string(by) + "\",\"error\":" + error_json(code, message) +
             ",\"planned\":false,\"plan\":null,\"explanation\":\"\",\"tier\":\"\",\"warnings\":[],\"notes\":[],"
             "\"gguf_convert_seconds\":0}";
    };
    EngineRequest engine;
    try {
      engine_request(load_params, engine);
    } catch (const RequestError& e) {
      return refusal("load", e.code, e.what());
    }
    // generate's own checks of its parameters (as generate answers them); held until the engine's are known.
    Work w;
    std::optional<RequestError> request_error;
    try {
      gen_params.only({"output", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "shift", "seed",
                       "uncond_skip_layers", "slg_layers", "slg_scale", "slg_start", "slg_stop", "max_sequence_length",
                       "sampler", "scheduler", "flow_multiplier", "noise_layout", "noise_source", "eta", "s_noise",
                       "dataspace", "loras", "save_metadata", "include_prompts", "preview_output", "preview_every",
                       "init_image", "mask", "mask_encoding", "denoise_strength", "resize", "hires", "distill", "control",
                       "cfg_truncation", "step_cache", "references", "lora_placement", "step_cache_threshold"});
      request_params(gen_params, w, false);
    } catch (const RequestError& e) {
      request_error = e;
    }
    mi_gen_params g = w.params;
    if (request_error) {
      // Only the engine's verdict is wanted: a stand-in request.
      if (const mi_status s = mi_gen_params_init(&g, sizeof g); s != MI_OK) throw_status(s);
      g.prompt = "check";
    }
    if (!w.preview.empty()) g.preview_callback = [](const std::uint8_t*, std::int32_t, std::int32_t, std::int32_t, void*) {};
    mi_check_report* report = nullptr;
    if (const mi_status s = mi_check(&engine.e, &g, &report); s != MI_OK) throw_status(s);
    const std::unique_ptr<mi_check_report, void (*)(mi_check_report*)> owner(report, mi_check_report_free);
    mi_check_info info{};
    info.struct_size = sizeof info;
    if (const mi_status s = mi_check_report_info(report, &info); s != MI_OK) throw_status(s);
    const auto text = [&](mi_check_text_kind k) { return std::string(mi_check_report_text(report, k)); };
    if (info.stage == MI_CHECK_STAGE_ENGINE_CREATE) {
      return refusal("load", status_code(info.status), protocol_names(text(MI_CHECK_TEXT_MESSAGE)));
    }
    if (request_error) return refusal("generate", request_error->code, request_error->what());
    if (info.stage == MI_CHECK_STAGE_GENERATE) {
      return refusal("generate", status_code(info.status), protocol_names(text(MI_CHECK_TEXT_MESSAGE)));
    }
    const auto lines = [](const std::string& t) {
      std::string out;
      for (std::size_t i = 0; i < t.size();) {
        const std::size_t n = std::min(t.find('\n', i), t.size());
        out += (out.empty() ? "" : ",") + str(protocol_names(t.substr(i, n - i)));
        i = n + 1;
      }
      return "[" + out + "]";
    };
    return std::string("{\"ok\":true,\"refused_by\":null,\"error\":null,\"planned\":") + tf(info.planned != 0) +
           ",\"plan\":" + (info.planned ? one_line(text(MI_CHECK_TEXT_PLAN_JSON).c_str()) : std::string("null")) +
           ",\"explanation\":" + str(text(MI_CHECK_TEXT_EXPLANATION)) + ",\"tier\":" + str(text(MI_CHECK_TEXT_TIER_REASON)) +
           ",\"warnings\":" + lines(text(MI_CHECK_TEXT_WARNINGS)) + ",\"notes\":" + lines(text(MI_CHECK_TEXT_NOTES)) +
           ",\"gguf_convert_seconds\":" + num(info.gguf_convert_seconds) + "}";
  }

  // Runs `answer` (the result object, or a RequestError) on a thread of its own and sends its response from there:
  // inspects (at most kMaxInspects at a time) and checks (one at a time), each kind in its own `slots`.
  template <class F>
  std::optional<std::string> on_thread(std::list<std::unique_ptr<Inspection>>& slots, std::size_t max, const std::string& id,
                                       const char* what, F answer) {
    slots.remove_if([](const std::unique_ptr<Inspection>& i) {
      if (!i->done.load()) return false;
      i->thread.join();  // it has sent its answer: only returning is left
      return true;
    });
    if (slots.size() >= max) {
      throw RequestError("busy", max == 1 ? std::string("a ") + what + " is running; send it again after it answers"
                                          : std::to_string(max) + " " + what + "s are running; send it again after one "
                                                                                "answers");
    }
    auto slot = std::make_unique<Inspection>();
    Inspection* s = slot.get();
    // Prepared here: the answer when building the real one fails (out of memory), so nothing escapes the thread.
    std::string fallback =
        "{\"id\":" + id + ",\"error\":" + error_json("internal", std::string(what) + " could not build its answer") + "}";
    s->thread = std::thread([this, s, id, answer = std::move(answer), fallback = std::move(fallback)] {
      std::string line;
      try {
        try {
          line = "{\"id\":" + id + ",\"result\":" + answer() + "}";
        } catch (const RequestError& e) {
          line = "{\"id\":" + id + ",\"error\":" + error_json(e.code, e.what()) + "}";
        } catch (const std::exception& e) {
          line = "{\"id\":" + id + ",\"error\":" + error_json("internal", e.what()) + "}";
        }
      } catch (...) {
        line.clear();
      }
      // Done before the answer goes out: a client that sends the next inspect on reading it finds a free slot.
      s->done.store(true);
      try {
        out_.emit(line.empty() ? fallback : line);
      } catch (...) {
        // The protocol stream is unusable (emit's lock failed): nothing to answer on.
      }
    });
    slots.push_back(std::move(slot));
    return std::nullopt;
  }

  void join_inspections() {
    for (auto* slots : {&inspections_, &checks_}) {
      for (const std::unique_ptr<Inspection>& i : *slots) i->thread.join();
      slots->clear();
    }
  }

  // inspect's result object for `path`.
  static std::string inspect_json(const std::string& file) {
    mi_file_report* report = nullptr;
    if (const mi_status s = mi_inspect_file(file.c_str(), &report); s != MI_OK) {
      throw RequestError(s == MI_ERR_INVALID_ARGUMENT ? "invalid_params" : status_code(s), mi_last_error_message());
    }
    const std::unique_ptr<mi_file_report, void (*)(mi_file_report*)> owner(report, mi_file_report_free);
    mi_file_info info{};
    info.struct_size = sizeof info;
    if (const mi_status s = mi_file_report_info(report, &info); s != MI_OK) throw_status(s);
    const auto text = [&](mi_file_text_kind k) { return str(mi_file_report_text(report, k)); };
    std::string out = "{\"path\":" + str(file) + ",\"kind\":\"" + file_kind_name(info.kind) + "\",\"loadable\":" +
                      tf(info.loadable != 0) + ",\"file_bytes\":" + std::to_string(info.file_bytes) +
                      ",\"tensors\":" + std::to_string(info.tensor_count) + ",\"note\":" + text(MI_FILE_TEXT_NOTE);
    if (info.kind == MI_FILE_KIND_DIT || info.kind == MI_FILE_KIND_TEXT_ENCODER) {
      const std::string precision = info.loadable ? "\"" + std::string(precision_name(info.precision)) + "\"" : "null";
      out += ",\"precision\":" + precision + ",\"quantization\":" + text(MI_FILE_TEXT_QUANTIZATION);
    }
    if (info.kind == MI_FILE_KIND_DIT) {
      std::string tiers, reasons;
      for (const auto& t : mi::server::kTierNames) {
        if (info.tiers & (1u << t.value)) tiers += (tiers.empty() ? "" : ",") + str(t.name);
        reasons += (reasons.empty() ? "" : ",") + str(t.name) + ":" + str(mi_file_report_tier_reason(report, t.value));
      }
      out += ",\"tiers\":[" + tiers + "],\"tier_reasons\":{" + reasons + "}";
    }
    if (info.kind == MI_FILE_KIND_CONTROL_ADAPTER) {
      std::string kinds;
      for (std::int32_t i = 0; i < info.control_kind_count; ++i) {
        kinds += (i ? "," : "") + str(mi_file_report_control_kind(report, i));
      }
      out += ",\"control_kinds\":[" + kinds + "]";
    }
    if (info.kind == MI_FILE_KIND_LORA || info.kind == MI_FILE_KIND_CONTROL_ADAPTER) {
      std::string types;
      const std::string_view all = mi_file_report_text(report, MI_FILE_TEXT_ADAPTER_TYPES);
      for (std::size_t pos = 0; pos < all.size();) {
        const std::size_t comma = std::min(all.find(',', pos), all.size());
        types += (types.empty() ? "" : ",") + str(all.substr(pos, comma - pos));
        pos = comma + 1;
      }
      out += ",\"base_model\":" + text(MI_FILE_TEXT_BASE_MODEL) + ",\"adapter_types\":[" + types + "],\"lora_rank\":" +
             std::to_string(info.lora_rank) + ",\"lora_adapters\":" + std::to_string(info.lora_adapters) +
             ",\"lora_unknown_keys\":" + std::to_string(info.lora_unknown_keys);
    }
    return out + "}";
  }

  std::string unload(const Params& p) {
    p.only({});
    if (generating()) throw busy_error();
    close_session(nullptr);
    join_worker();
    const bool had = engine_ != nullptr;
    mi_engine_free(engine_);
    engine_ = nullptr;
    model_.clear();
    return std::string("{\"unloaded\":") + tf(had) + "}";
  }

  // ---------------------------------------------------------------- component session (MI_CAP_COMPONENTS)
  //
  // The engine's component API (mi_session_open .. mi_session_close) behind methods of the same names, tensors through
  // shared-memory regions the server creates (shm.h). One persistent worker thread per session runs every session call
  // in order (encode_text, conditioning_*, initial_noise, run_begin, denoise_step, run_end, vae_*, session_close); the
  // request loop validates a call's params, queues it and goes on reading, so cancel, status and run_end are answered
  // while a call runs. A queued call that is cancelled is answered "cancelled" without reaching the engine.

  // One queued session call.
  struct SessionTask {
    enum class Kind { kCall, kStep, kRunEnd, kClose };
    std::string id;  // JSON of the request id
    Kind kind = Kind::kCall;
    std::function<std::string()> run;  // the result object; throws RequestError
    bool silent = false;               // no response (a close the server made itself)
    // Guarded by Session::mu.
    bool cancelled = false;
    std::string cancel_reason;
  };

  // A scratch region of shm_alloc, removed 60 s after its creation unless a request named it before.
  struct Scratch {
    std::shared_ptr<mi::server::shm::Region> region;
    std::chrono::steady_clock::time_point deadline;
    bool named = false;
  };
  static constexpr std::chrono::seconds kScratchUnnamedLifetime{60};
  static constexpr std::uint64_t kMaxScratchBytes = std::uint64_t{1} << 34;  // per region, and all of them together

  // The open run as the worker sees it: the region, and the engine's staging views and their offsets in the region.
  struct RunState {
    bool open = false;
    std::shared_ptr<mi::server::shm::Region> region;
    bool adopted = false;  // the region is the engine's staging (mi_run_adopt_staging); else copies through the engine's
    bool guided = false;   // a guided output (out[0] each call), not the rows'
    mi_run_info info{};
    mi_tensor x{}, control{};
    std::array<mi_tensor, MI_MAX_ROWS> out{};
    std::size_t x_at = 0, control_at = 0;  // offsets in the region (from its start)
    std::array<std::size_t, MI_MAX_ROWS> out_at{};
  };

  struct Session {
    Output* out = nullptr;
    mi_engine* engine = nullptr;
    std::int32_t max_rows = 0, max_latents = 0;
    std::thread worker;
    std::mutex mu;
    std::condition_variable cv;
    // Guarded by mu.
    std::deque<std::shared_ptr<SessionTask>> queue;
    std::shared_ptr<SessionTask> running;
    bool stop = false;
    std::map<std::string, Scratch> scratch;  // by name
    std::shared_ptr<mi::server::shm::Region> run_region;
    // The worker's.
    RunState run;
    std::map<std::int64_t, mi_conditioning*> conditionings;
    std::int64_t next_conditioning = 1;
    std::atomic<bool> run_open{false};
    ~Session() {
      for (auto& [id, c] : conditionings) mi_conditioning_free(c);
    }
  };

  static RequestError no_session() {
    return RequestError("no_session", "no component session is open; send \"session_open\" first");
  }
  // The engine's verdict on a session call, with its message (read on the calling thread: it is thread-local).
  [[noreturn]] static void throw_call(mi_status s) {
    throw RequestError(status_code(s), protocol_names(mi_last_error_message(), kComponentRenames));
  }

  Session& session() {
    if (session_ == nullptr) throw no_session();
    return *session_;
  }

  // A session call while a generate, preload or release runs.
  static RequestError session_busy_error() {
    return RequestError("busy", "a generate, preload or release is running; send session calls after its done event "
                                "(or the release's response), or cancel it");
  }

  static bool session_busy(Session& s) {
    const std::lock_guard<std::mutex> lock(s.mu);
    return !s.queue.empty() || s.running != nullptr;
  }

  // Queues a call on the session's worker; it answers it.
  std::optional<std::string> enqueue(const std::string& id, SessionTask::Kind kind, std::function<std::string()> run) {
    Session& s = session();
    if (generating()) throw session_busy_error();
    auto t = std::make_shared<SessionTask>();
    t->id = id;
    t->kind = kind;
    t->run = std::move(run);
    {
      const std::lock_guard<std::mutex> lock(s.mu);
      s.queue.push_back(std::move(t));
    }
    s.cv.notify_all();
    return std::nullopt;
  }

  // Before each phase of a session call (text_encoder, dit_load, decode): a progress event with the call's id, and a
  // cancel that reached the call before the engine marked it running lands now.
  static void on_session_progress(const mi_progress* p, void* user) {
    Session& s = *static_cast<Session*>(user);
    std::string id;
    {
      const std::lock_guard<std::mutex> lock(s.mu);
      if (s.running == nullptr) return;
      if (s.running->cancelled) mi_cancel(s.engine);
      id = s.running->id;
    }
    s.out->emit("{\"event\":\"progress\",\"id\":" + id + ",\"phase\":\"" + phase_name(p->phase) + "\",\"step\":" +
                std::to_string(p->step) + ",\"total_steps\":" + std::to_string(p->total_steps) +
                ",\"step_seconds\":" + num(p->step_seconds) + ",\"elapsed_seconds\":" + num(p->elapsed_seconds) + "}");
  }

  // Removes the scratch regions never named within their lifetime; returns the next deadline of one still unnamed.
  // Under s.mu.
  static std::optional<std::chrono::steady_clock::time_point> expire_scratch(Session& s) {
    const auto now = std::chrono::steady_clock::now();
    std::optional<std::chrono::steady_clock::time_point> next;
    for (auto it = s.scratch.begin(); it != s.scratch.end();) {
      if (!it->second.named && it->second.deadline <= now) {
        it = s.scratch.erase(it);
        continue;
      }
      if (!it->second.named && (!next || it->second.deadline < *next)) next = it->second.deadline;
      ++it;
    }
    return next;
  }

  // The session's worker: its calls in order until session_close (or a close the server makes) has run.
  void session_loop(Session& s) {
    for (;;) {
      std::shared_ptr<SessionTask> t;
      bool unrun = false;
      std::string reason;
      {
        std::unique_lock<std::mutex> lock(s.mu);
        for (;;) {
          const auto next = expire_scratch(s);
          if (!s.queue.empty() || s.stop) break;
          if (next) s.cv.wait_until(lock, *next);
          else s.cv.wait(lock);
        }
        if (s.queue.empty()) return;
        t = std::move(s.queue.front());
        s.queue.pop_front();
        unrun = t->cancelled;  // cancelled while queued: answered without reaching the engine
        reason = t->cancel_reason;
        if (!unrun) s.running = t;
      }
      std::string line;
      try {
        if (unrun) {
          line = "{\"id\":" + t->id + ",\"error\":" + error_json("cancelled", reason + " (not started)") + "}";
        } else {
          try {
            line = "{\"id\":" + t->id + ",\"result\":" + t->run() + "}";
          } catch (const RequestError& e) {
            line = "{\"id\":" + t->id + ",\"error\":" + error_json(e.code, e.what()) + "}";
          } catch (const std::exception& e) {
            line = "{\"id\":" + t->id + ",\"error\":" + error_json("internal", e.what()) + "}";
          }
        }
      } catch (...) {
        line = "{\"id\":" + t->id + ",\"error\":" + error_json("internal", "the call could not build its answer") + "}";
      }
      {
        const std::lock_guard<std::mutex> lock(s.mu);
        s.running.reset();
      }
      if (!t->silent) out_.emit(line);
    }
  }

  // Cancels the named call (queued or running), or without a name the running one. Returns its id, or nothing when
  // no such call is queued or running. run_end and session_close are not cancelled.
  std::optional<std::string> cancel_session_call(const std::string* id) {
    Session& s = *session_;
    const std::lock_guard<std::mutex> lock(s.mu);
    const auto cancellable = [](const SessionTask& t) {
      return t.kind != SessionTask::Kind::kRunEnd && t.kind != SessionTask::Kind::kClose;
    };
    if (s.running != nullptr && cancellable(*s.running) && (id == nullptr || s.running->id == *id)) {
      s.running->cancelled = true;
      s.running->cancel_reason = "cancelled";
      mi_cancel(s.engine);
      return s.running->id;
    }
    if (id == nullptr) return std::nullopt;
    for (const std::shared_ptr<SessionTask>& t : s.queue) {
      if (t->id == *id && !t->cancelled && cancellable(*t)) {
        t->cancelled = true;
        t->cancel_reason = "cancelled";
        return t->id;
      }
    }
    return std::nullopt;
  }

  // status's "session": {"run": bool, "queued": n, "running_id": id | null}.
  static std::string session_status(Session& s) {
    const std::lock_guard<std::mutex> lock(s.mu);
    return std::string("{\"run\":") + tf(s.run_open.load()) + ",\"queued\":" + std::to_string(s.queue.size()) +
           ",\"running_id\":" + (s.running != nullptr ? s.running->id : std::string("null")) + "}";
  }

  // session_close, and the close the server makes itself (load, unload, shutdown, end of input; `id` null: no
  // response): queued calls are answered cancelled, a running one is cancelled, then the worker closes the engine's
  // session (mi_session_close ends an open run first) and destroys the regions. Waits for the worker.
  void close_session(const std::string* id) {
    if (session_ == nullptr) return;
    Session& s = *session_;
    auto t = std::make_shared<SessionTask>();
    t->id = id != nullptr ? *id : "null";
    t->silent = id == nullptr;
    t->kind = SessionTask::Kind::kClose;
    t->run = [&s] {
      const mi_status st = mi_session_close(s.engine);
      s.run_open.store(false);
      s.run = RunState{};  // after the close: the run's end unregistered the adopted region
      {
        const std::lock_guard<std::mutex> lock(s.mu);
        s.run_region.reset();
        s.scratch.clear();
      }
      if (st != MI_OK) throw_call(st);
      return std::string("{\"closed\":true}");
    };
    bool call_running = false;
    {
      const std::lock_guard<std::mutex> lock(s.mu);
      for (const std::shared_ptr<SessionTask>& q : s.queue) {
        if (!q->cancelled) {
          q->cancelled = true;
          q->cancel_reason = "the session is closing";
        }
      }
      if (s.running != nullptr) {
        s.running->cancelled = true;
        s.running->cancel_reason = "the session is closing";
        call_running = s.running->kind != SessionTask::Kind::kRunEnd;
      }
      s.queue.push_back(t);
      s.stop = true;
    }
    if (call_running) mi_cancel(s.engine);  // (mi_session_close would cancel it too, once the worker reaches it)
    s.cv.notify_all();
    if (s.worker.joinable()) s.worker.join();
    session_.reset();
  }

  // session_open {"max_rows", "max_latents", "control_adapter", "plan_request"}: mi_session_open, answered at once.
  std::string session_open(const Params& p) {
    p.only({"max_rows", "max_latents", "control_adapter", "plan_request"});
    mi_session_params sp{};
    if (const mi_status s = mi_session_params_init(&sp, sizeof sp); s != MI_OK) throw_status(s);
    if (const auto v = p.integer("max_rows", 1, MI_MAX_ROWS)) sp.max_rows = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("max_latents", 1, MI_MAX_ROWS)) sp.max_latents = static_cast<std::int32_t>(*v);
    const std::optional<std::string> adapter = p.path("control_adapter");
    if (adapter) sp.control_adapter = adapter->c_str();
    Work plan;
    if (const Value* v = p.find("plan_request")) {
      if (!v->is_object()) throw RequestError("invalid_params", "\"plan_request\" must be an object (preload's params)");
      const Params pp(v);
      // preload's params: the request the session plans for before its first run.
      pp.only({"prompt", "negative_prompt", "width", "height", "steps", "cfg", "shift", "seed", "uncond_skip_layers",
               "slg_layers", "slg_scale", "slg_start", "slg_stop", "max_sequence_length", "sampler", "scheduler",
               "flow_multiplier", "noise_layout", "noise_source", "eta", "s_noise", "dataspace", "loras", "init_image",
               "mask", "mask_encoding", "denoise_strength", "resize", "hires", "distill", "control", "cfg_truncation",
               "step_cache", "references", "lora_placement", "step_cache_threshold"});
      request_params(pp, plan, true);
      sp.plan_request = &plan.params;
    }
    if (engine_ == nullptr) throw RequestError("no_engine", "no model loaded; send \"load\" first");
    if (session_ != nullptr) throw RequestError("busy", "a component session is open; send \"session_close\" first");
    auto s = std::make_unique<Session>();
    s->out = &out_;
    s->engine = engine_;
    s->max_rows = sp.max_rows;
    s->max_latents = sp.max_latents;
    sp.progress_callback = on_session_progress;
    sp.user_data = s.get();
    if (const mi_status st = mi_session_open(engine_, &sp); st != MI_OK) throw_call(st);
    try {
      s->worker = std::thread([this, raw = s.get()] { session_loop(*raw); });
    } catch (...) {
      mi_session_close(engine_);
      throw;
    }
    session_ = std::move(s);
    return "{\"max_rows\":" + std::to_string(session_->max_rows) + ",\"max_latents\":" +
           std::to_string(session_->max_latents) + "}";
  }

  std::optional<std::string> session_close(const Params& p, const std::string& id) {
    p.only({});
    if (session_ == nullptr) return std::string("{\"closed\":false}");
    if (generating()) throw session_busy_error();
    close_session(&id);
    return std::nullopt;
  }

  // shm_alloc {"bytes"}: a scratch region whose payload holds `bytes`. On the worker: reserving a large region's blocks
  // takes a while, and the request loop keeps answering cancel and status.
  std::optional<std::string> shm_alloc(const Params& p, const std::string& id) {
    p.only({"bytes"});
    const auto bytes = p.integer("bytes", 1, static_cast<std::int64_t>(kMaxScratchBytes));
    if (!bytes) throw RequestError("invalid_params", "\"bytes\" (the payload size) is required");
    Session& s = session();
    return enqueue(id, SessionTask::Kind::kCall, [&s, n = static_cast<std::size_t>(*bytes)] {
      {
        const std::lock_guard<std::mutex> lock(s.mu);
        std::uint64_t held = 0;
        for (const auto& [name, r] : s.scratch) held += r.region->payload_bytes();
        if (held + n > kMaxScratchBytes) {
          throw RequestError("out_of_memory", "the session's scratch regions would hold more than " +
                                                  std::to_string(kMaxScratchBytes >> 30) + " GiB (" +
                                                  std::to_string(held >> 20) + " MiB held); shm_free some first");
        }
      }
      std::shared_ptr<mi::server::shm::Region> region;
      try {
        region = std::make_shared<mi::server::shm::Region>(mi::server::shm::Region::create(n));
      } catch (const std::system_error& e) {
        throw RequestError(e.code() == std::errc::no_space_on_device ? "out_of_memory" : "io", e.what());
      }
      const std::string answer = region_json(*region);
      {
        const std::lock_guard<std::mutex> lock(s.mu);
        s.scratch[region->name()] = Scratch{region, std::chrono::steady_clock::now() + kScratchUnnamedLifetime, false};
      }
      return "{\"shm\":" + answer + "}";
    });
  }

  std::string shm_free(const Params& p) {
    p.only({"name"});
    const std::optional<std::string> name = p.string("name");
    if (!name) throw RequestError("invalid_params", "\"name\" (the region's name) is required");
    Session& s = session();
    const std::lock_guard<std::mutex> lock(s.mu);
    const auto it = s.scratch.find(*name);
    if (it == s.scratch.end()) throw RequestError("invalid_params", "no scratch region named \"" + *name + "\"");
    s.scratch.erase(it);  // a call using it holds it until it returns
    return "{\"freed\":true}";
  }

  static std::string region_json(const mi::server::shm::Region& r) {
    return "{\"name\":" + str(r.name()) + ",\"token\":\"" + r.token().hex() + "\",\"bytes\":" + std::to_string(r.bytes()) + "}";
  }

  // The scratch region `key` names, for a call that needs `bytes` of its payload. The first request naming a region
  // unlinks its name (the client has attached): from then on a crash leaks nothing.
  std::shared_ptr<mi::server::shm::Region> named_scratch(const Params& p, std::string_view key, std::uint64_t bytes) {
    const std::optional<std::string> name = p.string(key);
    if (!name) throw RequestError("invalid_params", "\"" + std::string(key) + "\" (a region of shm_alloc) is required");
    Session& s = session();
    const std::lock_guard<std::mutex> lock(s.mu);
    const auto it = s.scratch.find(*name);
    if (it == s.scratch.end()) throw RequestError("invalid_params", "no scratch region named \"" + *name + "\"");
    if (it->second.region->payload_bytes() < bytes) {
      throw RequestError("invalid_params", "region \"" + *name + "\" holds " +
                                               std::to_string(it->second.region->payload_bytes()) + " payload bytes; " +
                                               std::to_string(bytes) + " needed");
    }
    it->second.region->unlink();
    it->second.named = true;
    return it->second.region;
  }

  // An image size of the VAE calls: multiples of 16.
  static std::int32_t side(const Params& p, std::string_view key) {
    const auto v = p.integer(key, 16, 16384);
    if (!v || *v % 16 != 0) throw RequestError("invalid_params", "\"" + std::string(key) + "\" must be a multiple of 16 from 16 to 16384");
    return static_cast<std::int32_t>(*v);
  }

  static mi_tensor host_tensor(void* data, mi_dtype dtype, std::initializer_list<std::int64_t> shape) {
    mi_tensor t{};
    t.data = data;
    t.memory = MI_MEMORY_HOST;
    t.dtype = dtype;
    t.rank = static_cast<std::int32_t>(shape.size());
    int i = 0;
    for (const std::int64_t d : shape) t.shape[i++] = d;
    return t;
  }

  // {"id", "tokens", "engine_encoded", "prompt_tokens": {...}} of a conditioning the session holds.
  static std::string conditioning_json(std::int64_t id, const mi_conditioning* c) {
    mi_conditioning_info info{};
    info.struct_size = sizeof info;
    if (const mi_status s = mi_conditioning_query(c, &info); s != MI_OK) throw_call(s);
    mi_prompt_tokens t{};
    t.struct_size = sizeof t;
    if (const mi_status s = mi_conditioning_prompt_tokens(c, &t); s != MI_OK) throw_call(s);
    return "{\"id\":" + std::to_string(id) + ",\"tokens\":" + std::to_string(info.tokens) + ",\"engine_encoded\":" +
           tf(info.engine_encoded != 0) + ",\"prompt_tokens\":{\"total_tokens\":" + std::to_string(t.total_tokens) +
           ",\"kept_tokens\":" + std::to_string(t.kept_tokens) + ",\"prompt_tokens\":" + std::to_string(t.prompt_tokens) +
           ",\"dropped_tokens\":" + std::to_string(t.dropped_tokens) + ",\"max_length\":" + std::to_string(t.max_length) +
           ",\"truncated\":" + tf(t.truncated != 0) + "}}";
  }

  // encode_text {"prompts": [string, ...] (1..8), "max_sequence_length"?} -> {"conditionings": [...]}.
  std::optional<std::string> encode_text(const Params& p, const std::string& id) {
    p.only({"prompts", "max_sequence_length"});
    const Value* v = p.find("prompts");
    if (v == nullptr || !v->is_array() || v->as_array().empty() || v->as_array().size() > 8) {
      throw RequestError("invalid_params", "\"prompts\" must be an array of 1 to 8 strings");
    }
    auto prompts = std::make_shared<std::vector<std::string>>();
    for (const Value& e : v->as_array()) {
      if (!e.is_string()) throw RequestError("invalid_params", "\"prompts\" must be an array of 1 to 8 strings");
      if (e.as_string().find('\0') != std::string::npos) {
        throw RequestError("invalid_params", "\"prompts\" must be strings without NUL characters");
      }
      prompts->push_back(e.as_string());
    }
    const auto length = p.integer("max_sequence_length", 1, 4096);
    Session& s = session();
    return enqueue(id, SessionTask::Kind::kCall, [&s, prompts, length] {
      mi_text_params tp{};
      if (const mi_status st = mi_text_params_init(&tp, sizeof tp); st != MI_OK) throw_call(st);
      std::vector<const char*> ptrs;
      for (const std::string& q : *prompts) ptrs.push_back(q.c_str());
      tp.prompts = ptrs.data();
      tp.prompts_count = ptrs.size();
      if (length) tp.max_sequence_length = static_cast<std::int32_t>(*length);
      std::vector<mi_conditioning*> out(ptrs.size(), nullptr);
      if (const mi_status st = mi_encode_text(s.engine, &tp, out.data()); st != MI_OK) throw_call(st);
      std::string list;
      for (mi_conditioning* c : out) {
        const std::int64_t cid = s.next_conditioning++;
        s.conditionings[cid] = c;
        list += (list.empty() ? "" : ",") + conditioning_json(cid, c);
      }
      return "{\"conditionings\":[" + list + "]}";
    });
  }

  // conditioning_import {"shm", "dtype", "shape": [S, 4096]}: a host's own embeddings from a scratch region's payload
  // (mi_conditioning_create, copied).
  std::optional<std::string> conditioning_import(const Params& p, const std::string& id) {
    p.only({"shm", "dtype", "shape"});
    const mi_dtype dtype = p.name("dtype", mi::server::kDtypeNames).value_or(MI_DTYPE_F32);
    const Value* shape = p.find("shape");
    if (shape == nullptr || !shape->is_array() || shape->as_array().size() != 2 || !shape->as_array()[0].is_int() ||
        !shape->as_array()[1].is_int() || shape->as_array()[0].as_int() < 1 || shape->as_array()[0].as_int() > 4096 ||
        shape->as_array()[1].as_int() != MI_TEXT_DIM) {
      throw RequestError("invalid_params", "\"shape\" must be [tokens, 4096] with 1 <= tokens <= 4096");
    }
    const std::int64_t tokens = shape->as_array()[0].as_int();
    const std::uint64_t bytes = static_cast<std::uint64_t>(tokens) * MI_TEXT_DIM * (dtype == MI_DTYPE_F32 ? 4 : 2);
    std::shared_ptr<mi::server::shm::Region> region = named_scratch(p, "shm", bytes);
    Session& s = session();
    return enqueue(id, SessionTask::Kind::kCall, [&s, region, dtype, tokens] {
      const mi_tensor t = host_tensor(region->payload(), dtype, {tokens, MI_TEXT_DIM});
      mi_conditioning* c = nullptr;
      if (const mi_status st = mi_conditioning_create(&t, &c); st != MI_OK) throw_call(st);
      const std::int64_t cid = s.next_conditioning++;
      s.conditionings[cid] = c;
      return conditioning_json(cid, c);
    });
  }

  // conditioning_free {"conditionings": [id, ...]} -> {"freed": n}. A run keeps what it needs of them.
  std::optional<std::string> conditioning_free(const Params& p, const std::string& id) {
    p.only({"conditionings"});
    const Value* v = p.find("conditionings");
    if (v == nullptr || !v->is_array()) throw RequestError("invalid_params", "\"conditionings\" must be an array of conditioning ids");
    auto ids = std::make_shared<std::vector<std::int64_t>>();
    for (const Value& e : v->as_array()) {
      if (!e.is_int()) throw RequestError("invalid_params", "\"conditionings\" must be an array of conditioning ids");
      ids->push_back(e.as_int());
    }
    Session& s = session();
    return enqueue(id, SessionTask::Kind::kCall, [&s, ids] {
      for (const std::int64_t c : *ids) {
        if (s.conditionings.count(c) == 0) {
          throw RequestError("invalid_params", "no conditioning " + std::to_string(c) + " (nothing freed)");
        }
      }
      std::size_t freed = 0;
      for (const std::int64_t c : *ids) {
        const auto it = s.conditionings.find(c);
        if (it == s.conditionings.end()) continue;  // listed twice
        mi_conditioning_free(it->second);
        s.conditionings.erase(it);
        ++freed;
      }
      return "{\"freed\":" + std::to_string(freed) + "}";
    });
  }

  // initial_noise {"seed", "width", "height", "noise_layout"?, "shm"}: mi_generate's initial noise, F32
  // [128, height / 16, width / 16], into a scratch region's payload.
  std::optional<std::string> initial_noise(const Params& p, const std::string& id) {
    p.only({"seed", "width", "height", "noise_layout", "shm"});
    const auto seed = p.integer("seed", 0, (std::int64_t{1} << 53) - 1);
    if (!seed) throw RequestError("invalid_params", "\"seed\" (0 to 2^53-1) is required");
    const mi_noise_layout layout = p.name("noise_layout", mi::server::kNoiseLayoutNames).value_or(MI_NOISE_LAYOUT_OFFICIAL);
    const std::int32_t w = side(p, "width"), h = side(p, "height");
    const std::uint64_t floats = std::uint64_t{MI_LATENT_CHANNELS} * (h / 16) * (w / 16);
    std::shared_ptr<mi::server::shm::Region> region = named_scratch(p, "shm", floats * 4);
    return enqueue(id, SessionTask::Kind::kCall, [region, seed = *seed, layout, w, h, floats] {
      if (const mi_status st = mi_initial_noise(seed, layout, w, h, reinterpret_cast<float*>(region->payload()),
                                                static_cast<std::size_t>(floats));
          st != MI_OK) {
        throw_call(st);
      }
      return "{\"shape\":[" + std::to_string(MI_LATENT_CHANNELS) + "," + std::to_string(h / 16) + "," +
             std::to_string(w / 16) + "]}";
    });
  }

  // vae_encode {"shm", "width", "height", "out_shm"}: RGB8 [height][width][3] in `shm` -> model-space latents F32
  // [128, height / 16, width / 16] in `out_shm`.
  std::optional<std::string> vae_encode(const Params& p, const std::string& id) {
    p.only({"shm", "width", "height", "out_shm"});
    const std::int32_t w = side(p, "width"), h = side(p, "height");
    const std::uint64_t rgb = std::uint64_t{3} * w * h, lat = std::uint64_t{4} * MI_LATENT_CHANNELS * (h / 16) * (w / 16);
    std::shared_ptr<mi::server::shm::Region> in = named_scratch(p, "shm", rgb);
    std::shared_ptr<mi::server::shm::Region> out = named_scratch(p, "out_shm", lat);
    if (in == out) throw RequestError("invalid_params", "\"shm\" and \"out_shm\" must be different regions");
    Session& s = session();
    return enqueue(id, SessionTask::Kind::kCall, [&s, in, out, w, h] {
      mi_image image{};
      image.pixels = reinterpret_cast<const std::uint8_t*>(in->payload());
      image.width = w;
      image.height = h;
      image.channels = 3;
      const mi_tensor t = host_tensor(out->payload(), MI_DTYPE_F32, {MI_LATENT_CHANNELS, h / 16, w / 16});
      if (const mi_status st = mi_vae_encode(s.engine, &image, &t); st != MI_OK) throw_call(st);
      return "{\"shape\":[" + std::to_string(MI_LATENT_CHANNELS) + "," + std::to_string(h / 16) + "," +
             std::to_string(w / 16) + "]}";
    });
  }

  // vae_decode {"shm", "width", "height", "out_shm"}: model-space latents F32 [128, height / 16, width / 16] in `shm`
  // -> RGB8 [height][width][3] in `out_shm`.
  std::optional<std::string> vae_decode(const Params& p, const std::string& id) {
    p.only({"shm", "width", "height", "out_shm"});
    const std::int32_t w = side(p, "width"), h = side(p, "height");
    const std::uint64_t rgb = std::uint64_t{3} * w * h, lat = std::uint64_t{4} * MI_LATENT_CHANNELS * (h / 16) * (w / 16);
    std::shared_ptr<mi::server::shm::Region> in = named_scratch(p, "shm", lat);
    std::shared_ptr<mi::server::shm::Region> out = named_scratch(p, "out_shm", rgb);
    if (in == out) throw RequestError("invalid_params", "\"shm\" and \"out_shm\" must be different regions");
    Session& s = session();
    return enqueue(id, SessionTask::Kind::kCall, [&s, in, out, w, h] {
      const mi_tensor lt = host_tensor(in->payload(), MI_DTYPE_F32, {MI_LATENT_CHANNELS, h / 16, w / 16});
      const mi_tensor rt = host_tensor(out->payload(), MI_DTYPE_U8, {h, w, 3});
      if (const mi_status st = mi_vae_decode(s.engine, &lt, &rt); st != MI_OK) throw_call(st);
      return "{\"shape\":[" + std::to_string(h) + "," + std::to_string(w) + ",3]}";
    });
  }

  // run_begin's parameters, kept by the call until mi_run_begin has copied them.
  struct RunRequest {
    mi_run_params params{};
    struct Row {
      std::int64_t conditioning = 0;
      std::vector<std::int32_t> skip;
      std::int32_t latent = 0;
    };
    std::vector<Row> rows;
    std::vector<float> sigmas;
    std::vector<std::int32_t> slg;
    std::vector<std::string> lora_paths;
    std::vector<mi_lora> loras;
    std::vector<Params::ControlImage> control;
    std::vector<mi_control_image> control_images;  // points into control
  };

  // run_begin {"width", "height", "rows": [{"conditioning", "skip_layers"?, "latent"?}], "guidance"?, "output"?,
  // "exact_sigma"?, "crops"?, "sigmas"? | "scheduler", "steps", "shift", "flow_multiplier", "denoise_strength", "cfg",
  // "slg_*", "loras", "control_lora_strength", "control_images", "control_resize", "cfg_truncation", "step_cache",
  // "step_cache_threshold"}: mi_run_begin, then the run's region (mi_run_adopt_staging) ->
  // {"sigmas", "info", "shm", "tensors", "adopted"}.
  std::optional<std::string> run_begin(const Params& p, const std::string& id) {
    p.only({"width", "height", "rows", "guidance", "output", "exact_sigma", "crops", "sigmas", "scheduler", "steps",
            "shift", "flow_multiplier", "denoise_strength", "cfg", "slg_layers", "slg_scale", "slg_start", "slg_stop",
            "loras", "control_lora_strength", "control_images", "control_resize", "cfg_truncation", "step_cache",
            "step_cache_threshold"});
    auto rq = std::make_shared<RunRequest>();
    mi_run_params& r = rq->params;
    if (const mi_status s = mi_run_params_init(&r, sizeof r); s != MI_OK) throw_status(s);
    constexpr std::int64_t kI32 = std::numeric_limits<std::int32_t>::max();
    if (const auto v = p.integer("width", 0, kI32)) r.width = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("height", 0, kI32)) r.height = static_cast<std::int32_t>(*v);
    const Value* rows = p.find("rows");
    if (rows == nullptr || !rows->is_array() || rows->as_array().empty()) {
      throw RequestError("invalid_params", "\"rows\" must be a non-empty array of {\"conditioning\", \"skip_layers\"?, \"latent\"?}");
    }
    for (const Value& e : rows->as_array()) {
      const std::string item = "rows[" + std::to_string(rq->rows.size()) + "]";
      if (!e.is_object()) throw RequestError("invalid_params", "\"" + item + "\" must be an object");
      const Params row(&e);
      row.only({"conditioning", "skip_layers", "latent"});
      RunRequest::Row out;
      const auto c = row.integer("conditioning", 1, std::numeric_limits<std::int64_t>::max());
      if (!c) throw RequestError("invalid_params", "\"" + item + ".conditioning\" (a conditioning id) is required");
      out.conditioning = *c;
      if (auto v = row.layers("skip_layers")) out.skip = std::move(*v);
      if (const auto v = row.integer("latent", 0, MI_MAX_ROWS - 1)) out.latent = static_cast<std::int32_t>(*v);
      rq->rows.push_back(std::move(out));
    }
    if (const auto v = p.name("guidance", mi::server::kGuidanceNames)) r.guidance = *v;
    if (const auto v = p.name("output", mi::server::kStepOutputNames)) r.output = *v;
    if (p.boolean("exact_sigma").value_or(false)) r.flags |= MI_RUN_EXACT_SIGMA;
    if (p.boolean("crops").value_or(false)) r.flags |= MI_RUN_CROPS;
    if (const Value* v = p.find("sigmas")) {
      if (!v->is_array() || v->as_array().size() < 2 || v->as_array().size() > 10001) {
        throw RequestError("invalid_params", "\"sigmas\" must be an array of 2 to 10001 numbers");
      }
      for (const Value& e : v->as_array()) {
        if (!e.is_number()) throw RequestError("invalid_params", "\"sigmas\" must be an array of 2 to 10001 numbers");
        rq->sigmas.push_back(static_cast<float>(e.as_double()));
      }
    }
    if (const auto v = p.name("scheduler", mi::server::kSchedulerNames)) r.scheduler = *v;
    if (const auto v = p.integer("steps", 1, 10000)) r.steps = static_cast<std::int32_t>(*v);
    if (const auto v = p.number("shift")) r.shift = *v;
    if (const auto v = p.number("flow_multiplier")) r.flow_multiplier = *v;
    if (const auto v = p.number("denoise_strength")) r.denoise_strength = *v;
    if (const auto v = p.number("cfg")) r.cfg = *v;
    if (auto v = p.layers("slg_layers")) rq->slg = std::move(*v);
    if (const auto v = p.number("slg_scale")) r.slg_scale = *v;
    if (const auto v = p.number("slg_start")) r.slg_start = *v;
    if (const auto v = p.number("slg_stop")) r.slg_stop = *v;
    if (auto v = p.loras("loras")) {
      for (auto& [path, strength] : *v) rq->lora_paths.push_back(std::move(path));
      for (std::size_t i = 0; i < v->size(); ++i) rq->loras.push_back({rq->lora_paths[i].c_str(), (*v)[i].second});
    }
    if (const auto v = p.number("control_lora_strength")) r.control_lora_strength = *v;
    if (auto v = p.control_images("control_images")) rq->control = std::move(*v);
    for (const Params::ControlImage& im : rq->control) {
      mi_control_image ci{};
      ci.kind = im.kind.c_str();
      ci.image_path = im.path.c_str();
      ci.strength = im.strength.value_or(1.0);
      ci.start = im.start.value_or(0.0);
      ci.end = im.end.value_or(1.0);
      rq->control_images.push_back(ci);
    }
    if (const auto v = p.name("control_resize", mi::server::kResizeNames)) r.control_resize = *v;
    if (const auto v = p.number("cfg_truncation")) r.cfg_truncation = *v;
    if (const auto v = p.name("step_cache", mi::server::kStepCacheNames)) r.step_cache = *v;
    if (const auto v = p.number("step_cache_threshold")) {
      if (!(*v > 0 && *v <= 1)) throw RequestError("invalid_params", "\"step_cache_threshold\" must be a number in (0, 1]");
      r.step_cache_threshold = *v;
    }
    Session& s = session();
    return enqueue(id, SessionTask::Kind::kCall, [this, &s, rq] { return begin_run(s, *rq); });
  }

  // The worker's part of run_begin.
  std::string begin_run(Session& s, RunRequest& rq) {
    namespace shm = mi::server::shm;
    mi_run_params& r = rq.params;
    std::vector<mi_run_row> rows;
    for (std::size_t i = 0; i < rq.rows.size(); ++i) {
      const auto it = s.conditionings.find(rq.rows[i].conditioning);
      if (it == s.conditionings.end()) {
        throw RequestError("invalid_params", "rows[" + std::to_string(i) + "].conditioning: no conditioning " +
                                                 std::to_string(rq.rows[i].conditioning));
      }
      mi_run_row row{};
      row.conditioning = it->second;
      row.skip_layers = rq.rows[i].skip.empty() ? nullptr : rq.rows[i].skip.data();
      row.skip_layers_count = rq.rows[i].skip.size();
      row.latent = rq.rows[i].latent;
      rows.push_back(row);
    }
    r.rows = rows.data();
    r.rows_count = rows.size();
    if (!rq.sigmas.empty()) {
      r.sigmas = rq.sigmas.data();
      r.sigmas_count = rq.sigmas.size();
    }
    if (!rq.slg.empty()) {
      r.slg_layers = rq.slg.data();
      r.slg_layers_count = rq.slg.size();
    }
    if (!rq.loras.empty()) {
      r.loras = rq.loras.data();
      r.loras_count = rq.loras.size();
    }
    if (!rq.control_images.empty()) {
      r.control_images = rq.control_images.data();
      r.control_images_count = rq.control_images.size();
    }
    if (const mi_status st = mi_run_begin(s.engine, &r); st != MI_OK) throw_call(st);
    // From here an error ends the run again: a failed run_begin leaves none open.
    RunState R;
    R.guided = r.output != MI_STEP_OUTPUT_ROWS;
    try {
      R.info.struct_size = sizeof R.info;
      if (const mi_status st = mi_run_query(s.engine, &R.info); st != MI_OK) throw_call(st);
      std::size_t count = 0;
      const float* sigmas = mi_run_sigmas(s.engine, &count);
      if (sigmas == nullptr) throw_call(MI_ERR_INTERNAL);
      std::string sig;
      for (std::size_t i = 0; i < count; ++i) sig += (i ? "," : "") + mi::server::exact_num(sigmas[i]);
      try {
        R.region = std::make_shared<shm::Region>(shm::Region::create(static_cast<std::size_t>(R.info.staging_bytes)));
      } catch (const std::system_error& e) {
        throw RequestError(e.code() == std::errc::no_space_on_device ? "out_of_memory" : "io", e.what());
      }
      const mi_status adopt = refuse_pin_ ? MI_ERR_NOT_SUPPORTED
                                          : mi_run_adopt_staging(s.engine, R.region->payload(), R.region->payload_bytes());
      if (adopt != MI_OK && adopt != MI_ERR_NOT_SUPPORTED) throw_call(adopt);
      R.adopted = adopt == MI_OK;
      if (!R.adopted) {
        out_.log("warning", std::string("run region not pinned, steps copy through the engine's staging: ") +
                                (refuse_pin_ ? "MOSSINFER_SERVER_TEST_REFUSE_PIN is set" : mi_last_error_message()));
      }
      if (const mi_status st = mi_run_staging(s.engine, &R.x, R.out.data(), R.out.size(), &R.control); st != MI_OK) {
        throw_call(st);
      }
      // Offsets in the region: the views' offsets in the engine's block (x first, at 0), after the header page.
      const auto* base = R.adopted ? reinterpret_cast<const std::byte*>(R.region->payload())
                                   : static_cast<const std::byte*>(R.x.data);
      const auto at = [&](const mi_tensor& t) {
        return shm::kPayloadOffset + static_cast<std::size_t>(static_cast<const std::byte*>(t.data) - base);
      };
      R.x_at = at(R.x);
      if (R.control.data != nullptr) R.control_at = at(R.control);
      std::string outs;
      for (std::int32_t k = 0; k < R.info.outputs; ++k) {
        const mi_tensor& o = R.out[static_cast<std::size_t>(k)];
        if (o.data != nullptr) R.out_at[static_cast<std::size_t>(k)] = at(o);
        outs += std::string(k ? "," : "") + (o.data != nullptr ? std::to_string(at(o)) : std::string("null"));
      }
      const mi_run_info& i = R.info;
      std::string answer =
          "{\"sigmas\":[" + sig + "],\"info\":{\"latent_channels\":" + std::to_string(i.latent_channels) +
          ",\"latent_height\":" + std::to_string(i.latent_height) + ",\"latent_width\":" + std::to_string(i.latent_width) +
          ",\"steps\":" + std::to_string(i.steps) + ",\"outputs\":" + std::to_string(i.outputs) + ",\"rows\":" +
          std::to_string(i.rows) + ",\"latents\":" + std::to_string(i.latents) + ",\"control_channels\":" +
          std::to_string(i.control_channels) + ",\"schedule_matches\":" + tf(i.schedule_matches != 0) +
          ",\"staging_bytes\":" + std::to_string(i.staging_bytes) + "},\"shm\":" + region_json(*R.region) +
          ",\"tensors\":{\"x\":" + std::to_string(R.x_at) + ",\"control\":" +
          (R.control.data != nullptr ? std::to_string(R.control_at) : std::string("null")) + ",\"out\":[" + outs +
          "]},\"adopted\":" + tf(R.adopted) + "}";
      R.open = true;
      {
        const std::lock_guard<std::mutex> lock(s.mu);
        s.run_region = R.region;
      }
      s.run = std::move(R);
      s.run_open.store(true);
      return answer;
    } catch (...) {
      mi_run_end(s.engine, nullptr);  // unregisters an adopted region before it is destroyed (R, on the way out)
      throw;
    }
  }

  // denoise_step {"sigma", "step"?, "rows"?, "shape"?: [L, h, w], "control"?, "control_changed"?}: one mi_denoise_step
  // over the run region's slots -> {"step", "rows", "cached", "on_schedule", "seconds"}.
  struct StepRequest {
    float sigma = 0;
    std::int32_t step = -1;
    std::uint32_t rows = 0;
    std::optional<std::array<std::int64_t, 3>> shape;
    bool control = false, control_changed = false;
  };

  std::optional<std::string> denoise_step(const Params& p, const std::string& id) {
    p.only({"sigma", "step", "rows", "shape", "control", "control_changed"});
    StepRequest q;
    const auto sigma = p.number("sigma");
    if (!sigma) throw RequestError("invalid_params", "\"sigma\" is required");
    q.sigma = static_cast<float>(*sigma);
    if (const auto v = p.integer("step", -1, 10000)) q.step = static_cast<std::int32_t>(*v);
    if (const auto v = p.integer("rows", 0, (1 << MI_MAX_ROWS) - 1)) q.rows = static_cast<std::uint32_t>(*v);
    if (const Value* v = p.find("shape")) {
      std::array<std::int64_t, 3> sh{};
      if (!v->is_array() || v->as_array().size() != 3) throw RequestError("invalid_params", "\"shape\" must be [latents, h, w]");
      for (std::size_t i = 0; i < 3; ++i) {
        const Value& e = v->as_array()[i];
        if (!e.is_int() || e.as_int() < 1 || e.as_int() > 1 << 16) {
          throw RequestError("invalid_params", "\"shape\" must be [latents, h, w] of positive integers");
        }
        sh[i] = e.as_int();
      }
      q.shape = sh;
    }
    q.control = p.boolean("control").value_or(false);
    q.control_changed = p.boolean("control_changed").value_or(false);
    Session& s = session();
    {
      // The first step names the run region: the client has attached, so its name goes.
      const std::lock_guard<std::mutex> lock(s.mu);
      if (s.run_region != nullptr) s.run_region->unlink();
    }
    return enqueue(id, SessionTask::Kind::kStep, [&s, q] { return step(s, q); });
  }

  static std::string step(Session& s, const StepRequest& q) {
    RunState& R = s.run;
    if (!R.open) throw RequestError("invalid_argument", "no run is open; send \"run_begin\" first");
    const std::int64_t L = R.info.latents, H = R.info.latent_height, W = R.info.latent_width;
    std::int64_t h = H, w = W;
    if (q.shape) {
      if ((*q.shape)[0] != L || (*q.shape)[1] > H || (*q.shape)[2] > W) {
        throw RequestError("invalid_params", "\"shape\" must be [" + std::to_string(L) + ", h <= " + std::to_string(H) +
                                                 ", w <= " + std::to_string(W) + "]");
      }
      h = (*q.shape)[1];
      w = (*q.shape)[2];
    }
    mi_step_params sp{};
    if (const mi_status st = mi_step_params_init(&sp, sizeof sp); st != MI_OK) throw_call(st);
    sp.sigma = q.sigma;
    sp.step = q.step;
    sp.rows = q.rows;
    if (q.control_changed) sp.flags |= MI_STEP_CONTROL_CHANGED;
    const auto shaped = [&](const mi_tensor& view, std::int64_t channels) {
      return L == 1 ? host_tensor(view.data, MI_DTYPE_F32, {channels, h, w})
                    : host_tensor(view.data, MI_DTYPE_F32, {L, channels, h, w});
    };
    sp.x = shaped(R.x, MI_LATENT_CHANNELS);
    const std::size_t latent_bytes = static_cast<std::size_t>(MI_LATENT_CHANNELS * h * w) * sizeof(float);
    const std::size_t control_bytes = static_cast<std::size_t>(L * R.info.control_channels * h * w) * sizeof(float);
    if (q.control) {
      if (R.control.data == nullptr) throw RequestError("invalid_params", "\"control\": the session has no control adapter");
      sp.control = shaped(R.control, R.info.control_channels);
    }
    for (std::int32_t k = 0; k < R.info.outputs; ++k) {
      const mi_tensor& o = R.out[static_cast<std::size_t>(k)];
      if (o.data != nullptr) sp.out[k] = host_tensor(o.data, MI_DTYPE_F32, {MI_LATENT_CHANNELS, h, w});
    }
    std::byte* region = R.region->payload() - mi::server::shm::kPayloadOffset;
    if (!R.adopted) {
      std::memcpy(R.x.data, region + R.x_at, static_cast<std::size_t>(L) * latent_bytes);
      if (q.control) std::memcpy(R.control.data, region + R.control_at, control_bytes);
    }
    mi_step_info info{};
    info.struct_size = sizeof info;
    {
      // A cancel or run_end that arrived since the worker took the call: the engine clears its cancel flag when the
      // step starts, so it is honoured here (one that lands between this check and the engine's start is lost: the
      // step runs and answers normally).
      const std::lock_guard<std::mutex> lock(s.mu);
      if (s.running != nullptr && s.running->cancelled) throw RequestError("cancelled", s.running->cancel_reason);
    }
    if (const mi_status st = mi_denoise_step(s.engine, &sp, &info); st != MI_OK) {
      if (st == MI_CANCELLED) {
        std::string reason;
        {
          const std::lock_guard<std::mutex> lock(s.mu);
          reason = s.running != nullptr ? s.running->cancel_reason : std::string();
        }
        throw RequestError("cancelled", reason.empty() ? std::string(mi_last_error_message()) : reason);
      }
      throw_call(st);
    }
    if (!R.adopted) {
      for (std::int32_t k = 0; k < R.info.outputs; ++k) {
        const bool written = R.guided ? k == 0 : (info.rows & (1u << k)) != 0;
        if (written && R.out[static_cast<std::size_t>(k)].data != nullptr) {
          std::memcpy(region + R.out_at[static_cast<std::size_t>(k)], R.out[static_cast<std::size_t>(k)].data, latent_bytes);
        }
      }
    }
    return "{\"step\":" + std::to_string(info.step) + ",\"rows\":" + std::to_string(info.rows) + ",\"cached\":" +
           tf(info.cached != 0) + ",\"on_schedule\":" + tf(info.on_schedule != 0) + ",\"seconds\":" + num(info.seconds) + "}";
  }

  // run_end: queued steps are answered cancelled unrun and a step in flight is cancelled (the client's interrupt), then
  // the worker ends the run (mi_run_end) and destroys its region -> {"ended", "report", "warnings", "note"}, generate's
  // done without an image.
  std::optional<std::string> run_end(const Params& p, const std::string& id) {
    p.only({});
    Session& s = session();
    if (generating()) throw session_busy_error();
    auto t = std::make_shared<SessionTask>();
    t->id = id;
    t->kind = SessionTask::Kind::kRunEnd;
    t->run = [&s] {
      mi_result* r = nullptr;
      const mi_status st = mi_run_end(s.engine, &r);
      s.run_open.store(false);
      s.run = RunState{};  // the region after the run's end: mi_run_end unregistered it
      {
        const std::lock_guard<std::mutex> lock(s.mu);
        s.run_region.reset();
      }
      if (st != MI_OK) throw_call(st);
      if (r == nullptr) return std::string("{\"ended\":false}");
      const std::unique_ptr<mi_result, void (*)(mi_result*)> owner(r, mi_result_free);
      mi_gen_params defaults{};
      mi_gen_params_init(&defaults, sizeof defaults);  // no request settings to warn about
      return "{\"ended\":true,\"report\":" + one_line(mi_result_report_json(r)) + ",\"warnings\":" +
             warnings_json(r, defaults) + ",\"note\":" + str(mi_result_text(r, MI_TEXT_COMPONENT_NOTE)) + "}";
    };
    bool step_running = false;
    {
      const std::lock_guard<std::mutex> lock(s.mu);
      for (const std::shared_ptr<SessionTask>& q : s.queue) {
        if (q->kind == SessionTask::Kind::kStep && !q->cancelled) {
          q->cancelled = true;
          q->cancel_reason = "the run is ending";
        }
      }
      if (s.running != nullptr && s.running->kind == SessionTask::Kind::kStep) {
        s.running->cancelled = true;
        s.running->cancel_reason = "the run is ending";
        step_running = true;
      }
      s.queue.push_back(std::move(t));
    }
    if (step_running) mi_cancel(s.engine);
    s.cv.notify_all();
    return std::nullopt;
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
    std::vector<std::string> reference_paths;
    std::vector<mi_reference_image> references;  // points into reference_paths
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
    // The whole-job fields come with the 56-byte struct (MI_CAP_PROGRESS_FRACTION): both null with an older library.
    const bool whole = p->struct_size >= offsetof(mi_progress, eta_seconds) + sizeof p->eta_seconds;
    job.fraction.store(whole ? p->fraction : -1.0);
    job.eta.store(whole ? p->eta_seconds : -1.0);
    w.out->emit("{\"event\":\"progress\",\"id\":" + job.id + ",\"phase\":\"" + phase_name(p->phase) + "\",\"step\":" +
                std::to_string(p->step) + ",\"total_steps\":" + std::to_string(p->total_steps) +
                ",\"step_seconds\":" + num(p->step_seconds) + ",\"elapsed_seconds\":" + num(p->elapsed_seconds) +
                ",\"fraction\":" + or_null(job.fraction.load()) + ",\"eta_seconds\":" + or_null(job.eta.load()) + "}");
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

  // done.warnings: the LoRA, control adapter and reference image warnings, lossy accelerators and settings without effect, prompt
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
      // Reference images without an edit adapter, an edit adapter without reference images.
      for (std::int32_t i = 0, n = mi_result_message_count(r, MI_MESSAGE_REFERENCE_WARNING); i < n; ++i) {
        if (const char* m = mi_result_message(r, MI_MESSAGE_REFERENCE_WARNING, i)) add(m);
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
        add("lossy accelerators (" + active + "): " + what);
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
  std::unique_ptr<Session> session_;  // the open component session (the request loop's)
  const bool refuse_pin_;              // test_refuse_pin()
  mi_engine* engine_ = nullptr;
  std::string model_;
  mi_mode mode_ = MI_MODE_BALANCED;
  std::unique_ptr<Job> job_;
  std::thread worker_;
  std::list<std::unique_ptr<Inspection>> inspections_;  // the main thread's: started, reaped and joined there
  std::list<std::unique_ptr<Inspection>> checks_;       // the same for check (one at a time)
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
  // Regions a crashed server left before its client attached (shm.h; Linux only, nothing on Windows).
  try {
    mi::server::shm::sweep_stale();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "moss-server: shared-memory sweep failed: %s\n", e.what());
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
