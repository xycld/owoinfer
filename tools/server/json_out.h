// moss-server's JSON output helpers. Header only and free of the engine: the server links the shared libmossinfer,
// which exports mi_* only, so it cannot call pipeline/report_json.h (that file needs the pipeline). Instead
// tests/unit/test_server_json.cpp checks these against pipeline::json_escape / json_number.
#pragma once

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <system_error>

namespace mi::server {

// Length of the well-formed UTF-8 sequence at s[i] (Unicode Table 3-7), or 0 if it is ill-formed.
inline std::size_t utf8_sequence_length(std::string_view s, std::size_t i) {
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
// Keep in sync with src/pipeline/report_json.cpp json_escape (tests/unit/test_server_json.cpp compares them).
inline std::string str(std::string_view s) {
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

// A JSON number, 6 significant digits; null for NaN and infinities. std::to_chars ignores the locale, so a library
// that calls setlocale() cannot put a decimal comma into the protocol. Same as pipeline::json_number.
inline std::string num(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[32];
  const std::to_chars_result res = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::general, 6);
  if (res.ec != std::errc()) return "null";  // cannot happen: 6 significant digits fit
  return std::string(buf, res.ptr);
}

// A JSON number that reads back as exactly `d` (the shortest such text); null for NaN and infinities. For values a
// client feeds back bit for bit (a run's sigmas: a float widened to double reads back as that float).
inline std::string exact_num(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[32];
  const std::to_chars_result res = std::to_chars(buf, buf + sizeof(buf), d);
  if (res.ec != std::errc()) return "null";  // cannot happen: a double's shortest form fits
  return std::string(buf, res.ptr);
}

// A JSON document from the engine (report, plan) as one line: raw newlines can only be whitespace between tokens
// (inside strings they are escaped), so dropping them keeps the document intact. NULL (the library could not render
// the document) becomes null.
inline std::string one_line(const char* json) {
  if (json == nullptr) return "null";
  std::string out;
  for (const char* p = json; *p != '\0'; ++p) {
    if (*p != '\n' && *p != '\r') out += *p;
  }
  return out;
}

}  // namespace mi::server
