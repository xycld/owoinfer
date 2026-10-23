#include "pipeline/quality_cert.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <random>
#include <set>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "backend/cuda_libs.h"
#include "core/cuda_check.h"
#include "kernels/attention_q8/attention_q8.h"
#include "kernels/basic/elementwise.h"
#include "kernels/fused/fused_gemm.h"
#include "kernels/quant/quant.h"
#include "kernels/quant/w4a8_decode.h"
#include "kernels/quant4/quant4.h"
#include "models/qwen35_text/text_encoder.h"
#include "util/json.h"
#include "util/path.h"
#include "util/rename_replace.h"
#include "util/replaceable_file.h"
#include "util/xxh64.h"

// GCC's -Wdangling-reference heuristic flags every `const T& x = f(dom, where + "...")`: the DOM accessors below
// return references into their json::Value argument, never into the temporary message string.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif

namespace mi::pipeline {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMaxCertificates = 64;
constexpr std::size_t kMaxFiles = 64;
constexpr std::size_t kMaxEntries = 64;   // gates, numerics, samplers
constexpr std::size_t kMaxInfo = 128;
constexpr std::size_t kMaxCacheFiles = 256;
constexpr std::size_t kMaxHashCacheEntries = 512;
constexpr std::size_t kMaxHashCacheBytes = std::size_t{1} << 20;
constexpr std::size_t kMaxNameBytes = 255;
constexpr std::size_t kMaxPathBytes = 4096;
constexpr std::size_t kMaxToolBytes = 256;
constexpr std::int64_t kMaxPrompts = 100000;
constexpr std::int64_t kMaxSteps = 10000;
constexpr std::int64_t kMaxPixels = std::int64_t{1} << 40;
constexpr const char* kSidecarSuffix = ".mossinfer-cert.json";
constexpr const char* kSidecarInDir = "mossinfer-cert.json";
constexpr const char* kCacheSubdir = "quality_certs";
constexpr const char* kInstallSubdir = "share/mossinfer/quality_certs";
constexpr const char* kHashCacheFile = "file_hashes.json";
constexpr std::string_view kHashCacheFormat = "mossinfer-file-hashes";
constexpr std::int64_t kHashCacheVersion = 1;

template <typename E, std::size_t N>
using Tokens = std::array<std::pair<E, std::string_view>, N>;

constexpr Tokens<QualityTier, 3> kTiers{{{QualityTier::kReference, "reference"}, {QualityTier::kHigh, "high"},
                                         {QualityTier::kTurbo, "turbo"}}};
constexpr Tokens<DitPrecision, 3> kDit{{{DitPrecision::kBf16, "bf16"}, {DitPrecision::kInt8ConvRot, "int8_convrot"},
                                        {DitPrecision::kW4A4, "w4a4"}}};
constexpr Tokens<AttentionMode, 5> kAttention{{{AttentionMode::kExact, "exact"}, {AttentionMode::kQ8Fp16, "q8_fp16"},
                                               {AttentionMode::kQ8Fp16F32Acc, "q8_fp16_f32acc"},
                                               {AttentionMode::kQ8Fp8, "q8_fp8"},
                                               {AttentionMode::kQ8Fp8F32Acc, "q8_fp8_f32acc"}}};
constexpr Tokens<TePrecision, 3> kTe{{{TePrecision::kBf16, "bf16"}, {TePrecision::kInt8ConvRot, "int8_convrot"},
                                      {TePrecision::kW4A8, "w4a8"}}};

template <typename E, std::size_t N>
std::string_view token_of(const Tokens<E, N>& t, E e) {
  for (const auto& [v, s] : t) {
    if (v == e) return s;
  }
  return "?";
}

template <typename E, std::size_t N>
std::optional<E> parse_token(const Tokens<E, N>& t, std::string_view s) {
  for (const auto& [v, name] : t) {
    if (name == s) return v;
  }
  return std::nullopt;
}

[[noreturn]] void fail(const std::string& what) { throw CertificateError(what); }

// Shortest round-trip form; locale-independent. Callers only pass finite values (validated).
std::string num(double d) {
  char buf[40];
  const auto r = std::to_chars(buf, buf + sizeof(buf), d);
  return std::string(buf, r.ptr);
}

// JSON string literal: quotes, backslashes and control characters escaped; other bytes as they are (UTF-8).
std::string json_escape(std::string_view s) {
  std::string out = "\"";
  for (const char c : s) {
    const auto u = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (u < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", u);
      out += buf;
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string hex64(std::uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

bool is_lower_hex(std::string_view s, std::size_t len) {
  return s.size() == len && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

std::uint64_t parse_hex64(std::string_view s) {
  std::uint64_t v = 0;
  static_cast<void>(std::from_chars(s.data(), s.data() + s.size(), v, 16));  // callers validated the digits
  return v;
}

// Gate, numerics, info and sampler names, prompt-set ids.
bool is_key_name(std::string_view s) {
  return !s.empty() && s.size() <= 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
         });
}

bool is_gpu_arch(std::string_view s) {
  if (s.size() < 5 || s.size() > 6 || s.substr(0, 3) != "sm_") return false;
  return std::all_of(s.begin() + 3, s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

bool is_date(std::string_view s) {
  constexpr std::string_view kPattern = "dddd-dd-ddTdd:dd:ddZ";
  if (s.size() != kPattern.size()) return false;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (kPattern[i] == 'd' ? !(s[i] >= '0' && s[i] <= '9') : s[i] != kPattern[i]) return false;
  }
  const auto field = [&](std::size_t at) { return (s[at] - '0') * 10 + (s[at + 1] - '0'); };
  const int month = field(5), day = field(8), hour = field(11), minute = field(14), second = field(17);
  return month >= 1 && month <= 12 && day >= 1 && day <= 31 && hour <= 23 && minute <= 59 && second <= 60;
}

bool is_file_name(std::string_view s) {
  if (s.empty() || s.size() > kMaxNameBytes || s == "." || s == "..") return false;
  return std::none_of(s.begin(), s.end(), [](char c) { return c == '/' || c == '\\' || static_cast<unsigned char>(c) < 0x20; });
}

bool is_sidecar_name(std::string_view n) {
  const std::string_view suffix = kSidecarSuffix;
  return n == kSidecarInDir || (n.size() > suffix.size() && n.substr(n.size() - suffix.size()) == suffix);
}

// --- strict DOM access: every object is checked for duplicate and unknown keys ---

const json::Object& object(const json::Value& v, const std::string& where) {
  if (!v.is_object()) fail(where + ": expected an object");
  const json::Object& o = v.as_object();
  std::set<std::string_view> seen;
  for (const auto& [k, _] : o) {
    if (!seen.insert(k).second) fail(where + ": duplicate key '" + k + "'");
  }
  return o;
}

void only_keys(const json::Object& o, std::initializer_list<std::string_view> allowed, const std::string& where) {
  for (const auto& [k, _] : o) {
    if (std::find(allowed.begin(), allowed.end(), k) == allowed.end()) fail(where + ": unknown key '" + k + "'");
  }
}

const json::Value* member(const json::Object& o, std::string_view key) {
  for (const auto& [k, v] : o) {
    if (k == key) return &v;
  }
  return nullptr;
}

const json::Value& required(const json::Object& o, std::string_view key, const std::string& where) {
  const json::Value* v = member(o, key);
  if (v == nullptr) fail(where + ": missing '" + std::string(key) + "'");
  return *v;
}

const std::string& string_of(const json::Value& v, const std::string& where) {
  if (!v.is_string()) fail(where + ": expected a string");
  return v.as_string();
}

std::int64_t int_of(const json::Value& v, std::int64_t lo, std::int64_t hi, const std::string& where) {
  if (!v.is_int()) fail(where + ": expected an integer");
  const std::int64_t x = v.as_int();
  if (x < lo || x > hi) fail(where + ": out of range");
  return x;
}

double finite_of(const json::Value& v, const std::string& where) {
  if (!v.is_number()) fail(where + ": expected a number");
  const double d = v.as_double();
  if (!std::isfinite(d)) fail(where + ": not finite");
  return d;
}

bool bool_of(const json::Value& v, const std::string& where) {
  if (!v.is_bool()) fail(where + ": expected true or false");
  return v.as_bool();
}

const json::Array& array_of(const json::Value& v, std::size_t max_size, const std::string& where) {
  if (!v.is_array()) fail(where + ": expected an array");
  if (v.as_array().size() > max_size) fail(where + ": too many entries");
  return v.as_array();
}

std::uint64_t hex_of(const json::Value& v, const std::string& where) {
  const std::string& h = string_of(v, where);
  if (!is_lower_hex(h, 16)) fail(where + ": expected 16 lowercase hex digits");
  return parse_hex64(h);
}

template <typename E, std::size_t N>
E enum_of(const Tokens<E, N>& t, const json::Value& v, const std::string& where) {
  const std::string& s = string_of(v, where);
  const auto e = parse_token(t, s);
  if (!e) fail(where + ": unknown value '" + s + "'");
  return *e;
}

json::Value parse_document(std::string_view text, std::size_t max_bytes) {
  if (text.size() > max_bytes) fail("file too large");
  try {
    return json::parse(text, json::ParseLimits{65536});
  } catch (const json::ParseError& e) {
    fail(std::string("malformed JSON: ") + e.what());
  }
}

// At most `limit` bytes of `path`; throws CertificateError when it cannot be read or holds more. Read through a
// concurrent writer's rename (util/replaceable_file.h), so a replace in flight is not taken for a broken file.
std::string read_limited(const fs::path& path, std::size_t limit) {
  std::string text;
  switch (util::read_replaceable_file(path, limit, text)) {
    case util::ReadStatus::kOk: return text;
    case util::ReadStatus::kTooLarge: fail(util::path_utf8(path) + ": file too large");
    case util::ReadStatus::kMissing: fail("cannot open " + util::path_utf8(path));
    case util::ReadStatus::kError: break;
  }
  fail("read error on " + util::path_utf8(path));
}

ComponentIdentity component_from(const json::Value& v, const std::string& where) {
  const json::Object& o = object(v, where);
  only_keys(o, {"files"}, where);
  const json::Array& files = array_of(required(o, "files", where), kMaxFiles, where + ".files");
  if (files.empty()) fail(where + ".files: empty");
  ComponentIdentity c;
  for (std::size_t i = 0; i < files.size(); ++i) {
    const std::string w = where + ".files[" + std::to_string(i) + "]";
    const json::Object& f = object(files[i], w);
    only_keys(f, {"name", "size", "xxh64"}, w);
    FileIdentity id;
    id.name = string_of(required(f, "name", w), w + ".name");
    if (!is_file_name(id.name)) fail(w + ".name: not a plain file name");
    id.size = static_cast<std::uint64_t>(int_of(required(f, "size", w), 0, INT64_MAX, w + ".size"));
    id.xxh64 = hex_of(required(f, "xxh64", w), w + ".xxh64");
    c.files.push_back(std::move(id));
  }
  return c;
}

CertScope scope_from(const json::Value& v, const std::string& where) {
  const json::Object& o = object(v, where);
  only_keys(o, {"max_pixels", "samplers", "steps", "accelerators"}, where);
  CertScope s;
  s.max_pixels = int_of(required(o, "max_pixels", where), 1, kMaxPixels, where + ".max_pixels");
  const json::Array& samplers = array_of(required(o, "samplers", where), kMaxEntries, where + ".samplers");
  for (const json::Value& x : samplers) {
    const std::string& name = string_of(x, where + ".samplers");
    if (!is_key_name(name)) fail(where + ".samplers: bad name '" + name + "'");
    s.samplers.push_back(name);
  }
  const json::Array& steps = array_of(required(o, "steps", where), 2, where + ".steps");
  if (steps.size() != 2) fail(where + ".steps: expected [min, max]");
  s.min_steps = static_cast<int>(int_of(steps[0], 1, kMaxSteps, where + ".steps"));
  s.max_steps = static_cast<int>(int_of(steps[1], 1, kMaxSteps, where + ".steps"));
  const std::string& acc = string_of(required(o, "accelerators", where), where + ".accelerators");
  if (acc != "off") fail(where + ".accelerators: certificates are measured with accelerators 'off'");
  return s;
}

Certificate certificate_from(const json::Value& v, const std::string& where) {
  const json::Object& o = object(v, where);
  only_keys(o, {"tier", "model", "config", "numerics", "gpu_arch", "prompt_set", "scope", "date", "gates", "passed",
                "info", "tool"},
            where);
  Certificate c;
  c.tier = enum_of(kTiers, required(o, "tier", where), where + ".tier");

  const std::string wc = where + ".config";
  const json::Object& cfg = object(required(o, "config", where), wc);
  only_keys(cfg, {"dit_precision", "attention", "te_precision", "lora"}, wc);
  c.config.dit = enum_of(kDit, required(cfg, "dit_precision", wc), wc + ".dit_precision");
  c.config.attention = enum_of(kAttention, required(cfg, "attention", wc), wc + ".attention");
  c.config.te = enum_of(kTe, required(cfg, "te_precision", wc), wc + ".te_precision");
  if (string_of(required(cfg, "lora", wc), wc + ".lora") != "none") fail(wc + ".lora: certificates cover no LoRA ('none')");

  const std::string wm = where + ".model";
  const json::Object& model = object(required(o, "model", where), wm);
  only_keys(model, {"dit", "te"}, wm);
  c.model.dit = component_from(required(model, "dit", wm), wm + ".dit");
  if (const json::Value* te = member(model, "te"); te != nullptr && !te->is_null()) {
    c.model.te = component_from(*te, wm + ".te");
  }

  const std::string wn = where + ".numerics";
  const json::Object& numerics = object(required(o, "numerics", where), wn);
  if (numerics.empty() || numerics.size() > kMaxEntries) fail(wn + ": expected 1 to 64 entries");
  for (const auto& [k, x] : numerics) {
    if (!is_key_name(k)) fail(wn + ": bad name '" + k + "'");
    c.numerics.emplace_back(k, static_cast<std::uint64_t>(int_of(x, 0, INT64_MAX, wn + "." + k)));
  }
  std::sort(c.numerics.begin(), c.numerics.end());

  c.gpu_arch = string_of(required(o, "gpu_arch", where), where + ".gpu_arch");
  if (!is_gpu_arch(c.gpu_arch)) fail(where + ".gpu_arch: expected sm_<digits>");

  const std::string wp = where + ".prompt_set";
  const json::Object& ps = object(required(o, "prompt_set", where), wp);
  only_keys(ps, {"id", "prompts", "sha256"}, wp);
  c.prompt_set = string_of(required(ps, "id", wp), wp + ".id");
  if (!is_key_name(c.prompt_set)) fail(wp + ".id: bad id");
  c.prompts = static_cast<int>(int_of(required(ps, "prompts", wp), 1, kMaxPrompts, wp + ".prompts"));
  c.prompt_set_sha256 = string_of(required(ps, "sha256", wp), wp + ".sha256");
  if (!is_lower_hex(c.prompt_set_sha256, 64)) fail(wp + ".sha256: expected 64 lowercase hex digits");

  c.scope = scope_from(required(o, "scope", where), where + ".scope");

  c.date = string_of(required(o, "date", where), where + ".date");
  if (!is_date(c.date)) fail(where + ".date: expected YYYY-MM-DDTHH:MM:SSZ");

  const json::Array& gates = array_of(required(o, "gates", where), kMaxEntries, where + ".gates");
  for (std::size_t i = 0; i < gates.size(); ++i) {
    const std::string w = where + ".gates[" + std::to_string(i) + "]";
    const json::Object& g = object(gates[i], w);
    only_keys(g, {"name", "value", "op", "threshold", "pass"}, w);
    Gate gate;
    gate.name = string_of(required(g, "name", w), w + ".name");
    gate.value = finite_of(required(g, "value", w), w + ".value");
    const std::string& op = string_of(required(g, "op", w), w + ".op");
    if (op == "<=") gate.op = GateOp::kLe;
    else if (op == ">=") gate.op = GateOp::kGe;
    else fail(w + ".op: expected '<=' or '>='");
    gate.threshold = finite_of(required(g, "threshold", w), w + ".threshold");
    gate.pass = bool_of(required(g, "pass", w), w + ".pass");
    c.gates.push_back(std::move(gate));
  }
  c.passed = bool_of(required(o, "passed", where), where + ".passed");

  const json::Object& io = object(required(o, "info", where), where + ".info");
  if (io.size() > kMaxInfo) fail(where + ".info: too many entries");
  for (const auto& [k, x] : io) {
    if (!is_key_name(k)) fail(where + ".info: bad name '" + k + "'");
    c.info.emplace_back(k, finite_of(x, where + ".info." + k));
  }
  if (const json::Value* tool = member(o, "tool"); tool != nullptr) {
    c.tool = string_of(*tool, where + ".tool");
    if (c.tool.size() > kMaxToolBytes) fail(where + ".tool: too long");
  }
  if (const std::string err = validate_certificate(c); !err.empty()) fail(where + ": " + err);
  return c;
}

std::string component_json(const ComponentIdentity& c) {
  std::string s = "{\"files\": [";
  for (std::size_t i = 0; i < c.files.size(); ++i) {
    const FileIdentity& f = c.files[i];
    s += (i == 0 ? "" : ", ") + std::string("{\"name\": ") + json_escape(f.name) + ", \"size\": " + std::to_string(f.size) +
         ", \"xxh64\": \"" + hex64(f.xxh64) + "\"}";
  }
  return s + "]}";
}

std::string numerics_json(const EngineNumerics& n) {
  std::string s = "{";
  for (std::size_t i = 0; i < n.size(); ++i) s += (i == 0 ? "" : ", ") + json_escape(n[i].first) + ": " + std::to_string(n[i].second);
  return s + "}";
}

std::string certificate_json(const Certificate& c) {
  std::string s = "  {\"tier\": \"" + std::string(cert_token(c.tier)) + "\",\n   \"model\": {\"dit\": " +
                  component_json(c.model.dit) + ", \"te\": " + (c.model.te ? component_json(*c.model.te) : "null") +
                  "},\n   \"config\": {\"dit_precision\": \"" + std::string(cert_token(c.config.dit)) +
                  "\", \"attention\": \"" + std::string(cert_token(c.config.attention)) + "\", \"te_precision\": \"" +
                  std::string(cert_token(c.config.te)) + "\", \"lora\": \"none\"},\n   \"numerics\": " +
                  numerics_json(c.numerics) + ",\n   \"gpu_arch\": " + json_escape(c.gpu_arch) +
                  ",\n   \"prompt_set\": {\"id\": " + json_escape(c.prompt_set) + ", \"prompts\": " +
                  std::to_string(c.prompts) + ", \"sha256\": " + json_escape(c.prompt_set_sha256) +
                  "},\n   \"scope\": {\"max_pixels\": " + std::to_string(c.scope.max_pixels) + ", \"samplers\": [";
  for (std::size_t i = 0; i < c.scope.samplers.size(); ++i) s += (i == 0 ? "" : ", ") + json_escape(c.scope.samplers[i]);
  s += "], \"steps\": [" + std::to_string(c.scope.min_steps) + ", " + std::to_string(c.scope.max_steps) +
       "], \"accelerators\": \"off\"},\n   \"date\": " + json_escape(c.date) + ",\n   \"gates\": [";
  for (std::size_t i = 0; i < c.gates.size(); ++i) {
    const Gate& g = c.gates[i];
    s += (i == 0 ? "\n    " : ",\n    ") + std::string("{\"name\": ") + json_escape(g.name) + ", \"value\": " + num(g.value) +
         ", \"op\": \"" + (g.op == GateOp::kLe ? "<=" : ">=") + "\", \"threshold\": " + num(g.threshold) +
         ", \"pass\": " + (g.pass ? "true" : "false") + "}";
  }
  s += "],\n   \"passed\": " + std::string(c.passed ? "true" : "false") + ",\n   \"info\": {";
  for (std::size_t i = 0; i < c.info.size(); ++i) s += (i == 0 ? "" : ", ") + json_escape(c.info[i].first) + ": " + num(c.info[i].second);
  s += "}";
  if (!c.tool.empty()) s += ",\n   \"tool\": " + json_escape(c.tool);
  return s + "}";
}

int process_id() {
#ifdef _WIN32
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

// Writes `text` to `path` atomically and durably: a unique temporary file (mode 0644 minus the umask), flushed to
// disk, renamed over `path`, then (POSIX) the directory entry flushed. False on any failure (nothing replaced).
bool write_atomic(const fs::path& path, const std::string& text) {
  std::error_code ec;
  if (path.has_parent_path()) fs::create_directories(path.parent_path(), ec);
  char suffix[64];
  std::snprintf(suffix, sizeof(suffix), ".tmp.%d.%08x", process_id(), static_cast<unsigned>(std::random_device{}()));
  fs::path tmp = path;
  tmp += suffix;
#ifdef _WIN32
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out << text;
    out.close();
    if (!out) {
      fs::remove(tmp, ec);
      return false;
    }
  }
  HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  const bool flushed = h != INVALID_HANDLE_VALUE && FlushFileBuffers(h);
  if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
  if (!flushed) {
    fs::remove(tmp, ec);
    return false;
  }
#else
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) return false;
  bool ok = true;
  for (std::size_t done = 0; ok && done < text.size();) {
    const ssize_t n = ::write(fd, text.data() + done, text.size() - done);
    if (n < 0 && errno == EINTR) continue;
    ok = n > 0;
    if (ok) done += static_cast<std::size_t>(n);
  }
  ok = ok && ::fsync(fd) == 0;
  ok = ::close(fd) == 0 && ok;
  if (!ok) {
    fs::remove(tmp, ec);
    return false;
  }
#endif
  util::rename_replace(tmp, path, ec);
  if (ec) {
    fs::remove(tmp, ec);
    return false;
  }
#ifndef _WIN32
  const fs::path dir = path.has_parent_path() ? path.parent_path() : fs::path(".");
  if (const int dfd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC); dfd >= 0) {
    static_cast<void>(::fsync(dfd));  // best effort: the rename itself already happened
    ::close(dfd);
  }
#endif
  return true;
}

// --- file hash cache ---

struct FileKey {
  std::string path;  // canonical, UTF-8
  std::uint64_t dev = 0, ino = 0, size = 0;
  std::int64_t mtime_ns = 0;
  bool operator==(const FileKey&) const = default;
};

FileKey file_key(const fs::path& path) {
  std::error_code ec;
  const fs::path canon = fs::canonical(path, ec);
  if (ec) fail("cannot resolve " + util::path_utf8(path) + ": " + ec.message());
  FileKey k;
  k.path = util::path_utf8(canon);
#ifdef _WIN32
  HANDLE h = CreateFileW(canon.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  BY_HANDLE_FILE_INFORMATION info{};
  const bool ok = h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &info);
  if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
  if (!ok) fail("cannot stat " + k.path);
  k.dev = info.dwVolumeSerialNumber;
  k.ino = (std::uint64_t{info.nFileIndexHigh} << 32) | info.nFileIndexLow;
  k.size = (std::uint64_t{info.nFileSizeHigh} << 32) | info.nFileSizeLow;
  // FILETIME: 100 ns ticks since 1601, which overflow int64 as nanoseconds; rebased to the Unix epoch like POSIX.
  const std::uint64_t ticks = (std::uint64_t{info.ftLastWriteTime.dwHighDateTime} << 32) | info.ftLastWriteTime.dwLowDateTime;
  k.mtime_ns = (static_cast<std::int64_t>(ticks) - 116444736000000000) * 100;
#else
  struct stat st {};
  if (::stat(canon.c_str(), &st) != 0) fail("cannot stat " + k.path);
  k.dev = static_cast<std::uint64_t>(st.st_dev);
  k.ino = static_cast<std::uint64_t>(st.st_ino);
  k.size = static_cast<std::uint64_t>(st.st_size);
#ifdef __APPLE__
  k.mtime_ns = static_cast<std::int64_t>(st.st_mtimespec.tv_sec) * 1000000000 + st.st_mtimespec.tv_nsec;
#else
  k.mtime_ns = static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec;
#endif
#endif
  return k;
}

struct HashEntry {
  FileKey key;
  std::uint64_t xxh64 = 0;
};

// Anything unreadable or invalid reads as an empty cache.
std::vector<HashEntry> read_hash_cache(const fs::path& file) {
  std::vector<HashEntry> out;
  try {
    const json::Value doc = parse_document(read_limited(file, kMaxHashCacheBytes), kMaxHashCacheBytes);
    const json::Object& o = object(doc, "hash cache");
    if (string_of(required(o, "format", "hash cache"), "format") != kHashCacheFormat ||
        int_of(required(o, "version", "hash cache"), INT64_MIN, INT64_MAX, "version") != kHashCacheVersion) {
      return {};
    }
    for (const json::Value& e : array_of(required(o, "entries", "hash cache"), kMaxHashCacheEntries, "entries")) {
      const json::Object& eo = object(e, "entry");
      HashEntry h;
      h.key.path = string_of(required(eo, "path", "entry"), "path");
      if (h.key.path.empty() || h.key.path.size() > kMaxPathBytes) return {};
      h.key.dev = hex_of(required(eo, "dev", "entry"), "dev");
      h.key.ino = hex_of(required(eo, "ino", "entry"), "ino");
      h.key.size = static_cast<std::uint64_t>(int_of(required(eo, "size", "entry"), 0, INT64_MAX, "size"));
      h.key.mtime_ns = int_of(required(eo, "mtime_ns", "entry"), INT64_MIN, INT64_MAX, "mtime_ns");
      h.xxh64 = hex_of(required(eo, "xxh64", "entry"), "xxh64");
      out.push_back(std::move(h));
    }
  } catch (const std::exception&) {
    return {};
  }
  return out;
}

void write_hash_cache(const fs::path& file, const std::vector<HashEntry>& entries) {
  std::string s = "{\"format\": \"" + std::string(kHashCacheFormat) + "\", \"version\": " + std::to_string(kHashCacheVersion) +
                  ", \"entries\": [";
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const HashEntry& e = entries[i];
    s += (i == 0 ? "\n " : ",\n ") + std::string("{\"path\": ") + json_escape(e.key.path) + ", \"dev\": \"" + hex64(e.key.dev) +
         "\", \"ino\": \"" + hex64(e.key.ino) + "\", \"size\": " + std::to_string(e.key.size) +
         ", \"mtime_ns\": " + std::to_string(e.key.mtime_ns) + ", \"xxh64\": \"" + hex64(e.xxh64) + "\"}";
  }
  s += "]}\n";
  static_cast<void>(write_atomic(file, s));  // the cache is an optimization: a failed write only costs a re-hash
}

std::uint64_t hash_file(const fs::path& path, std::uint64_t expected_size) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open " + util::path_utf8(path));
  Xxh64 h;
  std::vector<char> buf(std::size_t{8} << 20);
  std::uint64_t total = 0;
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize n = in.gcount();
    if (n <= 0) break;
    h.update(buf.data(), static_cast<std::size_t>(n));
    total += static_cast<std::uint64_t>(n);
  }
  if (in.bad() || total != expected_size) fail("read error (or the file changed) while hashing " + util::path_utf8(path));
  return h.digest();
}

// The "cublaslt" entry of engine numerics; 0 when absent.
std::uint64_t cublaslt_of(const EngineNumerics& n) {
  for (const auto& [k, v] : n) {
    if (k == "cublaslt") return v;
  }
  return 0;
}

// Equal but for the cuBLASLt version.
bool same_except_cublaslt(const EngineNumerics& a, const EngineNumerics& b) {
  EngineNumerics x = a, y = b;
  const auto drop = [](EngineNumerics& n) {
    n.erase(std::remove_if(n.begin(), n.end(), [](const auto& e) { return e.first == "cublaslt"; }), n.end());
  };
  drop(x);
  drop(y);
  return x == y;
}

// A certificate that would apply but for the cuBLASLt version: which copy this process loaded, and why that matters.
std::string cublaslt_mismatch(const Certificate& c, const CertificateQuery& q) {
  const std::uint64_t measured = cublaslt_of(c.numerics), loaded = cublaslt_of(q.numerics);
  std::string copy = "cuBLASLt " + backend::cublaslt_version_string(loaded);
  if (loaded == 0) {
    copy = "no usable cuBLASLt";
  } else if (const backend::CublasLtStatus lt = backend::cublaslt_status(); lt.ok && lt.info.version == loaded) {
    copy += " (" + lt.info.source + " copy " + lt.info.path + ")";
  }
  return "certificate measured with cuBLASLt " + backend::cublaslt_version_string(measured) + "; this process loaded " +
         copy + ", so it does not apply" +
         (loaded != 0 ? std::string(" (MOSSINFER_CUDA_LIBS=bundled runs the release's own cuBLASLt)") : std::string());
}

// certificate_mismatch's checks; with sizes_only the content hashes are not compared (pre-check identities). A
// certificate that differs from the query only in the cuBLASLt version gets its own reason, once every other check
// has passed, so the report can say that the certificate exists but was measured with another cuBLASLt.
std::string mismatch(const Certificate& c, const ModelIdentity& model, const CertificateQuery& q, bool sizes_only) {
  if (std::string why = uncertifiable_reason(q.config); !why.empty()) return why;
  if (c.config != q.config) return "certificate for another configuration";
  if (c.gpu_arch != q.gpu_arch) return "certificate for " + c.gpu_arch + ", this GPU is " + q.gpu_arch;
  bool other_cublaslt = false;
  if (c.numerics != q.numerics) {
    if (!same_except_cublaslt(c.numerics, q.numerics)) {
      return "certificate from other engine numerics (re-certify after a numerics change)";
    }
    other_cublaslt = true;
  }
  if (!c.passed) return "the configuration failed certification";
  if (c.prompt_set != kCertPromptSetId || c.prompts != kCertPrompts ||
      (c.prompt_set_sha256 != kCertPromptSetSha256Base && c.prompt_set_sha256 != kCertPromptSetSha256Sft)) {
    return "measured on prompt set '" + c.prompt_set + "' (" + std::to_string(c.prompts) + " prompts), not the certification set";
  }
  for (const Gate& g : c.gates) {
    const std::size_t dot = g.name.find('.');
    if (dot == std::string::npos || g.name.substr(dot) != ".ocr_delta") continue;
    const std::string key = g.name.substr(0, dot) + ".ocr_prompts";
    const auto it = std::find_if(c.info.begin(), c.info.end(), [&](const auto& e) { return e.first == key; });
    if (it == c.info.end() || it->second < kMinOcrPrompts) return "too few text prompts behind gate '" + g.name + "'";
  }
  if (q.scope) {
    if (std::string why = scope_mismatch(c.scope, *q.scope); !why.empty()) return why;
  }
  const auto same = [&](const ComponentIdentity& a, const ComponentIdentity& b) {
    if (!sizes_only) return a.same_content(b);
    if (a.files.size() != b.files.size()) return false;
    for (std::size_t i = 0; i < a.files.size(); ++i) {
      if (a.files[i].size != b.files[i].size) return false;
    }
    return true;
  };
  if (!same(c.model.dit, model.dit)) return "certificate for other DiT weights";
  if (c.model.te.has_value() != model.te.has_value() || (c.model.te && !same(*c.model.te, *model.te))) {
    return "certificate for other text-encoder weights";
  }
  if (other_cublaslt) return cublaslt_mismatch(c, q);
  return {};
}

void add_reason(std::string& reason, const fs::path& file, const std::string& why) {
  if (reason.size() > 4000) return;
  reason += (reason.empty() ? "" : "; ") + util::path_utf8(file.filename()) + ": " + why;
}

struct LoadedFile {
  fs::path path;
  std::vector<Certificate> certs;
};

// Every file parsed once; unreadable or invalid ones are named in `reason`.
std::vector<LoadedFile> load_files(const std::vector<fs::path>& files, std::string& reason) {
  std::vector<LoadedFile> out;
  for (const fs::path& f : files) {
    try {
      out.push_back({f, read_certificate_file(f)});
    } catch (const CertificateError& e) {
      add_reason(reason, f, std::string("rejected (") + e.what() + ")");
    }
  }
  return out;
}

CertificateSearch first_usable(const std::vector<LoadedFile>& loaded, const ModelIdentity& model, const CertificateQuery& q,
                               std::string reason) {
  CertificateSearch r;
  for (const LoadedFile& f : loaded) {
    for (const Certificate& c : f.certs) {
      const std::string why = mismatch(c, model, q, false);
      if (why.empty()) {
        r.certificate = c;
        r.file = f.path;
        return r;
      }
      add_reason(reason, f.path, why);
    }
  }
  r.reason = std::move(reason);
  return r;
}

const double* info_value(const Certificate& c, std::string_view key) {
  for (const auto& [k, v] : c.info) {
    if (k == key) return &v;
  }
  return nullptr;
}

}  // namespace

std::string_view cert_token(QualityTier t) { return token_of(kTiers, t); }
std::string_view cert_token(DitPrecision p) { return token_of(kDit, p); }
std::string_view cert_token(AttentionMode a) { return token_of(kAttention, a); }
std::string_view cert_token(TePrecision p) { return token_of(kTe, p); }
std::optional<QualityTier> parse_quality_tier(std::string_view s) { return parse_token(kTiers, s); }
std::optional<DitPrecision> parse_dit_precision(std::string_view s) { return parse_token(kDit, s); }
std::optional<AttentionMode> parse_attention_mode(std::string_view s) { return parse_token(kAttention, s); }
std::optional<TePrecision> parse_te_precision(std::string_view s) { return parse_token(kTe, s); }

QualityTier tier_of(const CertConfig& c) {
  if (c.dit == DitPrecision::kW4A4) return QualityTier::kTurbo;
  if (c.dit == DitPrecision::kBf16 && c.attention == AttentionMode::kExact && c.te == TePrecision::kBf16) {
    return QualityTier::kReference;
  }
  return QualityTier::kHigh;
}

EngineNumerics engine_numerics(const CertConfig& c) {
  const backend::CublasLtStatus lt = backend::cublaslt_status();
  return engine_numerics(c, lt.ok ? lt.info.version : 0);
}

EngineNumerics engine_numerics(const CertConfig& c, std::uint64_t cublaslt_version) {
  EngineNumerics n{{"cublaslt", cublaslt_version},
                   {"elementwise", kernels::kElementwiseNumericsVersion},
                   {"fused_gemm", kernels::fused::kFusedGemmVersion},
                   {"te", qwen35_text::kNumericsVersion}};
  if (c.dit == DitPrecision::kInt8ConvRot || c.te == TePrecision::kInt8ConvRot || c.te == TePrecision::kW4A8) {
    n.emplace_back("int8", kernels::quant::kInt8NumericsVersion);
  }
  if (c.dit == DitPrecision::kW4A4) n.emplace_back("w4a4", kernels::quant4::kInt4NumericsVersion);
  if (c.te == TePrecision::kW4A8) n.emplace_back("te_w4a8", kernels::quant::kGroupedDecodeVersion);
  if (c.attention != AttentionMode::kExact) n.emplace_back("attention_q8", attention_q8::kNumericsVersion);
  std::sort(n.begin(), n.end());
  return n;
}

std::string uncertifiable_reason(const CertConfig& c) {
  if (c.lora) return "runtime LoRAs are never certified";
  return {};
}

std::string gpu_arch_class(int major, int minor) { return "sm_" + std::to_string(major * 10 + minor); }

std::string current_gpu_arch() {
  int device = 0, major = 0, minor = 0;
  MI_CUDA_CHECK(cudaGetDevice(&device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  MI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  return gpu_arch_class(major, minor);
}

bool ComponentIdentity::same_content(const ComponentIdentity& o) const {
  if (files.size() != o.files.size()) return false;
  for (std::size_t i = 0; i < files.size(); ++i) {
    if (files[i].size != o.files[i].size || files[i].xxh64 != o.files[i].xxh64) return false;
  }
  return true;
}

std::uint64_t ComponentIdentity::bytes() const {
  std::uint64_t b = 0;
  for (const FileIdentity& f : files) b += f.size;
  return b;
}

ComponentIdentity identify_component(const fs::path& path, bool hash, const fs::path& hash_cache_dir) {
  std::error_code ec;
  std::vector<fs::path> files;
  if (fs::is_directory(path, ec)) {
    fs::directory_iterator it(path, ec);
    if (ec) fail("cannot list " + util::path_utf8(path) + ": " + ec.message());
    for (const fs::directory_iterator end; it != end; it.increment(ec)) {
      std::error_code fec;
      const fs::path& p = it->path();
      const std::string name = util::path_utf8(p.filename());
      if (it->is_regular_file(fec) && (p.extension() == ".safetensors" || p.extension() == ".json") && !is_sidecar_name(name)) {
        files.push_back(p);
      }
    }
    if (ec) fail("cannot list " + util::path_utf8(path) + ": " + ec.message());
    if (files.empty()) fail(util::path_utf8(path) + " holds no .safetensors or .json files");
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
      return util::path_utf8(a.filename()) < util::path_utf8(b.filename());
    });
  } else if (fs::is_regular_file(path, ec)) {
    files.push_back(path);
  } else {
    fail(util::path_utf8(path) + " is not a weight file or directory");
  }
  if (files.size() > kMaxFiles) fail(util::path_utf8(path) + ": more than 64 identity files");

  const fs::path cache_file = hash && !hash_cache_dir.empty() ? hash_cache_dir / kHashCacheFile : fs::path();
  std::vector<HashEntry> cache;
  if (!cache_file.empty()) cache = read_hash_cache(cache_file);
  bool cache_changed = false;
  ComponentIdentity c;
  for (const fs::path& f : files) {
    FileIdentity id;
    id.name = util::path_utf8(f.filename());
    if (!hash) {
      id.size = fs::file_size(f, ec);
      if (ec) fail("cannot stat " + util::path_utf8(f) + ": " + ec.message());
    } else {
      const FileKey key = file_key(f);
      id.size = key.size;
      const auto hit = std::find_if(cache.begin(), cache.end(), [&](const HashEntry& e) { return e.key == key; });
      if (hit != cache.end()) {
        id.xxh64 = hit->xxh64;
      } else {
        id.xxh64 = hash_file(f, key.size);
        if (!(file_key(f) == key)) fail(util::path_utf8(f) + " changed while it was hashed");
        if (!cache_file.empty()) {
          std::erase_if(cache, [&](const HashEntry& e) { return e.key.path == key.path; });
          cache.push_back({key, id.xxh64});
          if (cache.size() > kMaxHashCacheEntries) cache.erase(cache.begin(), cache.end() - kMaxHashCacheEntries);
          cache_changed = true;
        }
      }
    }
    c.files.push_back(std::move(id));
  }
  if (cache_changed) write_hash_cache(cache_file, cache);
  return c;
}

std::vector<GateSpec> required_gates(QualityTier t, bool reference_ocr_gateable) {
  const double k = t == QualityTier::kReference ? 1.0 : t == QualityTier::kHigh ? 1.5 : 3.0;
  const double ocr = t == QualityTier::kReference ? -0.05 : t == QualityTier::kHigh ? -0.075 : -0.10;
  // Every tier: the Reference tier rendered every prompt twice bit-identically on the certifying machine.
  std::vector<GateSpec> out{{"determinism.mismatches", GateOp::kLe, 0.0}};
  if (t != QualityTier::kTurbo) {
    out.insert(out.end(), {{"same_seed.lpips_mean", GateOp::kLe, 0.12},
                           {"same_seed.lpips_max", GateOp::kLe, 0.25},
                           {"same_seed.dino_mean", GateOp::kGe, 0.90},
                           {"same_seed.ocr_delta", GateOp::kGe, -0.05}});
  }
  out.push_back({"reference.lpips_mean", GateOp::kLe, 0.088 * k});
  out.push_back({"reference.lpips_paired_max", GateOp::kLe, 0.088 * k / 1.5});
  out.push_back({"reference.dino_mean", GateOp::kGe, 1.0 - 0.075 * k});
  if (reference_ocr_gateable) {
    out.push_back({"reference.ocr_delta", GateOp::kGe, ocr});
  } else if (t == QualityTier::kTurbo) {
    out.push_back({"same_seed.ocr_delta", GateOp::kGe, ocr});
  }
  return out;
}

std::string scope_mismatch(const CertScope& scope, const GenerationScope& g) {
  if (g.accelerators) return "lossy accelerators are outside every certificate's scope";
  if (g.pixels > scope.max_pixels) {
    return std::to_string(g.pixels) + " pixels exceed the certified maximum " + std::to_string(scope.max_pixels);
  }
  if (std::find(scope.samplers.begin(), scope.samplers.end(), g.sampler) == scope.samplers.end()) {
    return "sampler '" + g.sampler + "' was not certified";
  }
  if (g.steps < scope.min_steps || g.steps > scope.max_steps) {
    return std::to_string(g.steps) + " steps lie outside the certified " + std::to_string(scope.min_steps) + ".." +
           std::to_string(scope.max_steps);
  }
  return {};
}

std::string validate_certificate(const Certificate& c) {
  if (c.tier != tier_of(c.config)) {
    return "tier '" + std::string(cert_token(c.tier)) + "' does not match the configuration (a '" +
           std::string(cert_token(tier_of(c.config))) + "' configuration)";
  }
  if (c.config.lora) return "certificates cover no LoRA";
  if (c.model.dit.files.empty() || c.model.dit.files.size() > kMaxFiles) return "bad DiT file count";
  if (c.model.te.has_value() != (c.config.te != TePrecision::kBf16)) {
    return "the text-encoder identity must be present exactly when te_precision is not bf16";
  }
  if (c.model.te && (c.model.te->files.empty() || c.model.te->files.size() > kMaxFiles)) return "bad text-encoder file count";
  for (const ComponentIdentity* comp : {&c.model.dit, c.model.te ? &*c.model.te : nullptr}) {
    if (comp == nullptr) continue;
    for (const FileIdentity& f : comp->files) {
      if (!is_file_name(f.name) || f.size > static_cast<std::uint64_t>(INT64_MAX)) return "bad file identity '" + f.name + "'";
    }
  }
  if (c.numerics.empty() || c.numerics.size() > kMaxEntries) return "bad engine numerics";
  for (std::size_t i = 0; i < c.numerics.size(); ++i) {
    if (!is_key_name(c.numerics[i].first)) return "bad numerics name";
    if (i > 0 && c.numerics[i - 1].first >= c.numerics[i].first) return "numerics names not sorted and unique";
  }
  if (!is_gpu_arch(c.gpu_arch)) return "bad gpu_arch";
  if (!is_key_name(c.prompt_set) || c.prompts < 1) return "bad prompt set";
  if (!is_lower_hex(c.prompt_set_sha256, 64)) return "bad prompt set digest";
  if (c.scope.max_pixels < 1 || c.scope.max_pixels > kMaxPixels) return "bad scope max_pixels";
  if (c.scope.samplers.empty() || c.scope.samplers.size() > kMaxEntries) return "scope lists no samplers";
  for (std::size_t i = 0; i < c.scope.samplers.size(); ++i) {
    if (!is_key_name(c.scope.samplers[i])) return "bad scope sampler";
    if (i > 0 && c.scope.samplers[i - 1] >= c.scope.samplers[i]) return "scope samplers not sorted and unique";
  }
  if (c.scope.min_steps < 1 || c.scope.min_steps > c.scope.max_steps || c.scope.max_steps > kMaxSteps) return "bad scope steps";
  if (c.scope.accelerators) return "certificates are measured with accelerators off";
  if (!is_date(c.date)) return "bad date";
  if (c.info.size() > kMaxInfo) return "too many info entries";
  std::set<std::string_view> info;
  for (const auto& [k, v] : c.info) {
    if (!is_key_name(k) || !std::isfinite(v)) return "bad info entry '" + k + "'";
    if (!info.insert(k).second) return "duplicate info entry '" + k + "'";
  }
  for (const char* key : {"same_seed.ocr_prompts", "reference.ocr_prompts"}) {
    const double* v = info_value(c, key);
    if (v == nullptr || *v < 0 || *v > kMaxPrompts || *v != std::floor(*v)) return std::string("info needs a count '") + key + "'";
  }
  if (c.gates.empty() || c.gates.size() > kMaxEntries) return "bad gate count";
  bool all = true;
  std::set<std::string_view> names;
  for (const Gate& g : c.gates) {
    if (!is_key_name(g.name)) return "bad gate name '" + g.name + "'";
    if (!names.insert(g.name).second) return "duplicate gate '" + g.name + "'";
    if (!std::isfinite(g.value) || !std::isfinite(g.threshold)) return "gate '" + g.name + "' is not finite";
    const bool pass = g.op == GateOp::kLe ? g.value <= g.threshold : g.value >= g.threshold;
    if (pass != g.pass) return "gate '" + g.name + "': the pass flag disagrees with its value and threshold";
    all = all && pass;
  }
  const bool ocr_gateable = *info_value(c, "reference.ocr_prompts") >= kMinOcrPrompts;
  for (const GateSpec& spec : required_gates(c.tier, ocr_gateable)) {
    const auto it = std::find_if(c.gates.begin(), c.gates.end(), [&](const Gate& g) { return g.name == spec.name; });
    if (it == c.gates.end()) return "missing the required gate '" + std::string(spec.name) + "'";
    // A small tolerance for limits like 0.088 * 1.5 written by another program in decimal.
    constexpr double kTol = 1e-12;
    const bool strict_enough = it->op == spec.op &&
                               (spec.op == GateOp::kLe ? it->threshold <= spec.limit + kTol : it->threshold >= spec.limit - kTol);
    if (!strict_enough) return "gate '" + std::string(spec.name) + "' is looser than the tier requires";
  }
  if (c.passed != all) return "'passed' disagrees with the gates";
  if (c.tool.size() > kMaxToolBytes) return "tool too long";
  return {};
}

std::vector<Certificate> parse_certificates(std::string_view text) {
  const json::Value doc = parse_document(text, kMaxCertificateFileBytes);
  const json::Object& o = object(doc, "document");
  only_keys(o, {"format", "version", "certificates"}, "document");
  if (string_of(required(o, "format", "document"), "format") != kCertificateFormat) fail("not a certificate file");
  if (int_of(required(o, "version", "document"), INT64_MIN, INT64_MAX, "version") != kCertificateFormatVersion) {
    fail("unsupported certificate format version");
  }
  const json::Array& certs = array_of(required(o, "certificates", "document"), kMaxCertificates, "certificates");
  std::vector<Certificate> out;
  for (std::size_t i = 0; i < certs.size(); ++i) out.push_back(certificate_from(certs[i], "certificates[" + std::to_string(i) + "]"));
  return out;
}

std::string certificates_json(const std::vector<Certificate>& certs) {
  if (certs.size() > kMaxCertificates) fail("more than 64 certificates");
  std::string s = "{\"format\": \"" + std::string(kCertificateFormat) + "\", \"version\": " +
                  std::to_string(kCertificateFormatVersion) + ", \"certificates\": [";
  for (std::size_t i = 0; i < certs.size(); ++i) {
    if (const std::string err = validate_certificate(certs[i]); !err.empty()) fail("certificate " + std::to_string(i) + ": " + err);
    s += (i == 0 ? "\n" : ",\n") + certificate_json(certs[i]);
  }
  return s + "]}\n";
}

std::vector<Certificate> read_certificate_file(const fs::path& path) {
  return parse_certificates(read_limited(path, kMaxCertificateFileBytes));
}

bool write_certificate_file(const fs::path& path, const std::vector<Certificate>& certs) {
  std::string doc;
  try {
    doc = certificates_json(certs);
  } catch (const CertificateError&) {
    return false;
  }
  if (doc.size() > kMaxCertificateFileBytes) return false;
  return write_atomic(path, doc);
}

fs::path certificate_sidecar(const fs::path& dit) {
  std::error_code ec;
  if (fs::is_directory(dit, ec)) return dit / kSidecarInDir;
  fs::path p = dit;
  p += kSidecarSuffix;
  return p;
}

fs::path install_certificate_dir(const fs::path& module_dir) {
  const fs::path dir = module_dir.lexically_normal();
  const fs::path name = dir.has_filename() ? dir.filename() : dir.parent_path().filename();
  if (name != "lib" && name != "lib64" && name != "bin") return {};
  return (dir / ".." / kInstallSubdir).lexically_normal();
}

fs::path install_certificate_dir() {
  static const fs::path dir = install_certificate_dir(util::path_from_utf8(backend::own_module_dir()));
  return dir;
}

std::string install_certificate_note() {
  if (!install_certificate_dir().empty()) return {};
  const std::string module = backend::own_module_dir();
  if (module.empty()) return "install-tree certificates not searched: the engine module's directory is unknown";
  return "install-tree certificates not searched: the engine module (" + module + ") is not in an install prefix's lib/ or bin/";
}

std::vector<fs::path> certificate_files(const fs::path& dit, const fs::path& cache_dir, const std::vector<fs::path>& extra_dirs,
                                        const fs::path& install_dir) {
  std::vector<fs::path> out;
  std::set<fs::path> seen;  // weakly canonical: a directory named twice (or a cache dir also named explicitly) once
  const auto add = [&](const fs::path& file) {
    std::error_code ec;
    const fs::path key = fs::weakly_canonical(file, ec);
    if (seen.insert(ec ? file : key).second) out.push_back(file);
  };
  const auto add_dir = [&](const fs::path& dir) {
    if (dir.empty()) return;
    std::vector<fs::path> found;
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    for (const fs::directory_iterator end; !ec && it != end; it.increment(ec)) {
      std::error_code fec;
      if (it->is_regular_file(fec) && it->path().extension() == ".json") found.push_back(it->path());
    }
    std::sort(found.begin(), found.end());
    if (found.size() > kMaxCacheFiles) found.resize(kMaxCacheFiles);
    for (const fs::path& f : found) add(f);
  };
  for (const fs::path& dir : extra_dirs) add_dir(dir);
  if (!cache_dir.empty()) add_dir(cache_dir / kCacheSubdir);
  std::error_code ec;
  if (const fs::path side = certificate_sidecar(dit); fs::is_regular_file(side, ec)) add(side);
  add_dir(install_dir);
  return out;
}

std::string certificate_mismatch(const Certificate& c, const ModelIdentity& model, const CertificateQuery& q) {
  return mismatch(c, model, q, false);
}

CertificateSearch find_certificate(const ModelIdentity& model, const CertificateQuery& q, const std::vector<fs::path>& files) {
  if (files.empty()) return CertificateSearch{std::nullopt, {}, "no certificate"};
  std::string reason;
  const std::vector<LoadedFile> loaded = load_files(files, reason);
  return first_usable(loaded, model, q, std::move(reason));
}

CertificateSearch find_certificate(const fs::path& dit, const fs::path& te, const CertificateQuery& q, const fs::path& cache_dir,
                                   std::optional<ModelIdentity>* identity, const std::vector<fs::path>& extra_dirs,
                                   const fs::path& install_dir) {
  CertificateSearch r;
  if (std::string why = uncertifiable_reason(q.config); !why.empty()) {
    r.reason = why;
    return r;
  }
  const std::vector<fs::path> files = certificate_files(dit, cache_dir, extra_dirs, install_dir);
  if (files.empty()) {
    r.reason = "no certificate";
    return r;
  }
  ModelIdentity cheap;
  try {
    cheap.dit = identify_component(dit, false);
    if (!te.empty()) cheap.te = identify_component(te, false);
  } catch (const CertificateError& e) {
    r.reason = e.what();
    return r;
  }
  // Pre-check on names and sizes: only certificates surviving it are worth hashing the weights for.
  std::string reason;
  std::vector<LoadedFile> loaded = load_files(files, reason);
  std::vector<LoadedFile> candidates;
  for (LoadedFile& f : loaded) {
    LoadedFile kept{f.path, {}};
    for (Certificate& c : f.certs) {
      const std::string why = mismatch(c, cheap, q, true);
      if (why.empty()) kept.certs.push_back(std::move(c));
      else add_reason(reason, f.path, why);
    }
    if (!kept.certs.empty()) candidates.push_back(std::move(kept));
  }
  if (candidates.empty()) {
    r.reason = std::move(reason);
    return r;
  }
  ModelIdentity full;
  try {
    full.dit = identify_component(dit, true, cache_dir);
    if (!te.empty()) full.te = identify_component(te, true, cache_dir);
  } catch (const CertificateError& e) {
    r.reason = e.what();
    return r;
  }
  if (identity != nullptr) *identity = full;
  return first_usable(candidates, full, q, std::move(reason));
}

std::string quality_report_json(const QualityReport& r) {
  const auto component = [](const fs::path& p, std::uint64_t bytes, const std::optional<ComponentIdentity>& id) {
    if (p.empty()) return std::string("null");
    return "{\"path\": " + json_escape(util::path_utf8(p)) + ", \"bytes\": " + std::to_string(bytes) +
           ", \"identity\": " + (id ? component_json(*id) : std::string("null")) + "}";
  };
  const CertConfig& c = r.config;
  const backend::CublasLtStatus lt = backend::cublaslt_status();
  const std::string cublaslt = lt.ok ? "{\"source\": " + json_escape(lt.info.source) + ", \"version\": " +
                                           std::to_string(lt.info.version) + ", \"path\": " + json_escape(lt.info.path) + "}"
                                     : std::string("null");
  return "{\"tier\": \"" + std::string(cert_token(tier_of(c))) + "\", \"config\": {\"dit_precision\": \"" +
         std::string(cert_token(c.dit)) + "\", \"attention\": \"" + std::string(cert_token(c.attention)) +
         "\", \"te_precision\": \"" + std::string(cert_token(c.te)) + "\", \"lora\": \"" + (c.lora ? "active" : "none") +
         "\"}, \"numerics\": " + numerics_json(engine_numerics(c, lt.ok ? lt.info.version : 0)) +
         ", \"cublaslt\": " + cublaslt + ", \"gpu_arch\": " + json_escape(r.gpu_arch) +
         ", \"reason\": " + json_escape(r.reason) + ", \"certificate\": " +
         (r.certificate.empty() ? std::string("null") : json_escape(util::path_utf8(r.certificate))) +
         ", \"dit\": " + component(r.dit, r.dit_bytes, r.dit_identity) + ", \"te\": " + component(r.te, r.te_bytes, r.te_identity) + "}";
}

std::string quality_report_json(const CertConfig& c, std::string_view gpu_arch, std::string_view reason,
                                const fs::path& certificate) {
  QualityReport r;
  r.config = c;
  r.gpu_arch = std::string(gpu_arch);
  r.reason = std::string(reason);
  r.certificate = certificate;
  return quality_report_json(r);
}

}  // namespace mi::pipeline
