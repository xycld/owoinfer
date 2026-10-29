#include "pipeline/png_writer.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "util/path.h"
#include "util/regular_file.h"
#include "util/rename_replace.h"

namespace mi::pipeline {
namespace {

constexpr std::size_t kMaxStoredBlock = 65535;  // deflate stored-block length limit
// The zlib stream is split over IDAT chunks of at most this size (decoders handle any split; small chunks keep
// every length far below PNG's 2^31-1 chunk limit and bound a reader's per-chunk buffer).
constexpr std::size_t kMaxIdatChunk = std::size_t{1} << 20;
constexpr std::size_t kMaxChunkLength = 0x7fffffff;  // PNG: chunk lengths are at most 2^31-1
static_assert(kMaxIdatChunk <= kMaxChunkLength);

const std::array<std::uint32_t, 256>& crc_table() {
  static const std::array<std::uint32_t, 256> table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t n = 0; n < 256; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xedb88320u ^ (c >> 1) : c >> 1;
      t[n] = c;
    }
    return t;
  }();
  return table;
}

std::uint32_t crc32(const std::uint8_t* p, std::size_t n, std::uint32_t crc = 0) {
  crc = ~crc;
  for (std::size_t i = 0; i < n; ++i) crc = crc_table()[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
  return ~crc;
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) out.push_back(static_cast<std::uint8_t>(v >> s));
}

void chunk(std::vector<std::uint8_t>& out, const char type[4], const std::uint8_t* data, std::size_t n) {
  if (n > kMaxChunkLength) throw std::length_error("encode_png_rgb8: PNG chunk longer than 2^31-1 bytes");
  put_u32(out, static_cast<std::uint32_t>(n));
  const std::size_t start = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data, data + n);
  put_u32(out, crc32(out.data() + start, out.size() - start));
}

std::uint32_t get_u32(const std::uint8_t* p) {
  return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) | (std::uint32_t{p[2]} << 8) | std::uint32_t{p[3]};
}

// Length of the UTF-8 sequence starting at s[i] (Unicode Table 3-7: no overlongs, surrogates or code points above
// U+10FFFF), 0 if it is ill-formed.
std::size_t utf8_sequence(std::string_view s, std::size_t i) {
  const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = b(i);
  if (c < 0x80) return 1;
  std::size_t n = 0;
  unsigned char lo = 0x80, hi = 0xbf;
  if (c >= 0xc2 && c <= 0xdf) n = 2;
  else if (c >= 0xe0 && c <= 0xef) {
    n = 3;
    if (c == 0xe0) lo = 0xa0;
    if (c == 0xed) hi = 0x9f;
  } else if (c >= 0xf0 && c <= 0xf4) {
    n = 4;
    if (c == 0xf0) lo = 0x90;
    if (c == 0xf4) hi = 0x8f;
  } else {
    return 0;
  }
  if (i + n > s.size()) return 0;
  if (b(i + 1) < lo || b(i + 1) > hi) return 0;
  for (std::size_t k = 2; k < n; ++k) {
    if (b(i + k) < 0x80 || b(i + k) > 0xbf) return 0;
  }
  return n;
}

bool valid_utf8(std::string_view s) {
  for (std::size_t i = 0; i < s.size();) {
    const std::size_t n = utf8_sequence(s, i);
    if (n == 0) return false;
    i += n;
  }
  return true;
}

bool is_ascii(std::string_view s) {
  return std::all_of(s.begin(), s.end(), [](char c) { return static_cast<unsigned char>(c) < 0x80; });
}

// PNG keywords: 1-79 Latin-1 printable characters (32-126, 161-255), no leading, trailing or consecutive spaces.
// Ours are ASCII.
void check_keyword(const std::string& k) {
  bool ok = !k.empty() && k.size() <= 79 && k.front() != ' ' && k.back() != ' ' && k.find("  ") == std::string::npos;
  for (const char c : k) ok = ok && c >= 32 && c <= 126;
  if (!ok) throw std::invalid_argument("encode_png_rgb8: invalid text keyword '" + k + "'");
}

void text_chunk(std::vector<std::uint8_t>& out, const PngText& t) {
  check_keyword(t.keyword);
  if (!valid_utf8(t.text)) throw std::invalid_argument("encode_png_rgb8: text of '" + t.keyword + "' is not valid UTF-8");
  if (t.text.find('\0') != std::string::npos) {
    throw std::invalid_argument("encode_png_rgb8: text of '" + t.keyword + "' contains NUL");
  }
  std::vector<std::uint8_t> data(t.keyword.begin(), t.keyword.end());
  data.push_back(0);
  if (is_ascii(t.text)) {
    data.insert(data.end(), t.text.begin(), t.text.end());  // tEXt: Latin-1, ASCII is a subset
    chunk(out, "tEXt", data.data(), data.size());
  } else {
    data.insert(data.end(), {0, 0, 0, 0});  // iTXt: uncompressed, empty language tag and translated keyword
    data.insert(data.end(), t.text.begin(), t.text.end());
    chunk(out, "iTXt", data.data(), data.size());
  }
}

void json_string(std::string& out, std::string_view s) {
  out.push_back('"');
  for (std::size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    const std::size_t n = utf8_sequence(s, i);
    if (n == 0 || c == 0) {
      out += "\xef\xbf\xbd";  // ill-formed byte or NUL -> U+FFFD
      ++i;
      continue;
    }
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 0x20 || c == 0x7f) {
      char buf[8];
      std::snprintf(buf, sizeof buf, "\\u%04x", c);
      out += buf;
    } else {
      out.append(s.substr(i, n));
    }
    i += n;
  }
  out.push_back('"');
}

std::string number(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, d);
  return std::string(buf, r.ptr);
}

// Valid UTF-8 without NUL: each ill-formed byte and each NUL becomes U+FFFD.
std::string clean_text(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    const std::size_t n = utf8_sequence(s, i);
    if (n == 0 || s[i] == '\0') {
      out += "\xef\xbf\xbd";
      ++i;
      continue;
    }
    out.append(s.substr(i, n));
    i += n;
  }
  return out;
}

// The last path component (directories are not written into images).
std::string base_name(std::string_view path) {
  const std::size_t slash = path.find_last_of("/\\");
  return std::string(slash == std::string_view::npos ? path : path.substr(slash + 1));
}

// Values with ',', ':' or a newline are quoted as JSON strings, as the settings-line convention expects.
std::string param_value(std::string_view raw) {
  const std::string v = clean_text(raw);
  if (v.find_first_of(",:\n") == std::string::npos) return v;
  std::string out;
  json_string(out, v);
  return out;
}

std::string display_sampler(const std::string& s) {
  if (s == "euler") return "Euler";
  if (s == "euler_ancestral") return "Euler a";
  if (s == "heun") return "Heun";
  if (s == "dpmpp_2m") return "DPM++ 2M";
  if (s == "dpmpp_2m_sde") return "DPM++ 2M SDE";
  return s;
}

std::string display_scheduler(const std::string& s) {
  if (s == "simple") return "Simple";
  if (s == "normal") return "Normal";
  if (s == "karras") return "Karras";
  if (s == "exponential") return "Exponential";
  if (s == "beta") return "Beta";
  if (s == "sgm_uniform") return "SGM Uniform";
  if (s == "moss") return "MOSS";
  return s;
}

std::string short_hash(const std::string& sha256) { return sha256.substr(0, std::min<std::size_t>(10, sha256.size())); }

}  // namespace

std::vector<std::uint8_t> encode_png_rgb8(std::span<const std::uint8_t> rgb, int width, int height,
                                          std::span<const PngText> text) {
  if (width <= 0 || height <= 0) throw std::invalid_argument("encode_png_rgb8: dimensions must be positive");
  const std::size_t row = static_cast<std::size_t>(width) * 3;
  if (rgb.size() != row * static_cast<std::size_t>(height)) {
    throw std::invalid_argument("encode_png_rgb8: expected " + std::to_string(row * static_cast<std::size_t>(height)) +
                                " bytes, got " + std::to_string(rgb.size()));
  }
  // Raw scanlines, each prefixed with filter type 0.
  std::vector<std::uint8_t> raw;
  raw.reserve((row + 1) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    const auto line = rgb.subspan(static_cast<std::size_t>(y) * row, row);
    raw.insert(raw.end(), line.begin(), line.end());
  }

  // zlib (RFC 1950): CMF 0x78 (deflate, 32K window), FLG 0x01 (check bits, fastest), stored blocks, Adler-32.
  std::vector<std::uint8_t> z = {0x78, 0x01};
  z.reserve(raw.size() + raw.size() / kMaxStoredBlock * 5 + 16);
  std::uint32_t a = 1, b = 0;
  for (std::size_t pos = 0;; pos += kMaxStoredBlock) {
    const std::size_t len = std::min(kMaxStoredBlock, raw.size() - pos);
    const bool last = pos + len >= raw.size();
    z.push_back(last ? 1 : 0);  // BFINAL, BTYPE = 00
    z.push_back(static_cast<std::uint8_t>(len));
    z.push_back(static_cast<std::uint8_t>(len >> 8));
    z.push_back(static_cast<std::uint8_t>(~len));
    z.push_back(static_cast<std::uint8_t>(~len >> 8));
    z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + len));
    for (std::size_t i = pos; i < pos + len; ++i) {
      a = (a + raw[i]) % 65521u;
      b = (b + a) % 65521u;
    }
    if (last) break;
  }
  put_u32(z, (b << 16) | a);

  std::vector<std::uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<std::uint8_t> ihdr;
  put_u32(ihdr, static_cast<std::uint32_t>(width));
  put_u32(ihdr, static_cast<std::uint32_t>(height));
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // bit depth 8, truecolor, deflate, filter method 0, no interlace
  png.reserve(z.size() + z.size() / kMaxIdatChunk * 12 + 64);
  chunk(png, "IHDR", ihdr.data(), ihdr.size());
  for (const PngText& t : text) text_chunk(png, t);
  for (std::size_t pos = 0; pos < z.size(); pos += kMaxIdatChunk) {
    chunk(png, "IDAT", z.data() + pos, std::min(kMaxIdatChunk, z.size() - pos));
  }
  chunk(png, "IEND", nullptr, 0);
  return png;
}

void write_png_rgb8(const std::filesystem::path& path, std::span<const std::uint8_t> rgb, int width, int height,
                    std::span<const PngText> text) {
  util::check_user_path(path, "PNG output");  // a Windows device or pipe path: refused before anything opens it
  const std::vector<std::uint8_t> png = encode_png_rgb8(rgb, width, height, text);
  // Write a sibling temporary file and rename it over `path`, so a failed or interrupted write never leaves a
  // truncated image under the final name (rename within one directory replaces atomically on POSIX).
  util::OutputFile out(path, util::OutputFile::Create::kTemporary);
  if (!out.is_open()) throw std::runtime_error("cannot write " + util::path_utf8(path));
  const std::filesystem::path tmp = out.path();
  {
    if (!out.write(png.data(), png.size()) || !out.close()) {
      std::error_code ignored;
      std::filesystem::remove(tmp, ignored);
      throw std::runtime_error("cannot write " + util::path_utf8(path));
    }
  }
  std::error_code ec;
  util::rename_replace(tmp, path, ec);
  if (ec) {
    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);
    throw std::runtime_error("cannot write " + util::path_utf8(path) + ": " + ec.message());
  }
}

std::vector<PngText> read_png_text(std::span<const std::uint8_t> png) {
  static constexpr std::uint8_t kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  if (png.size() < 8 || !std::equal(kSig, kSig + 8, png.begin())) throw std::invalid_argument("read_png_text: not a PNG");
  std::vector<PngText> out;
  std::size_t pos = 8;
  for (;;) {
    if (png.size() - pos < 12) throw std::invalid_argument("read_png_text: truncated chunk");
    const std::uint32_t len = get_u32(png.data() + pos);
    if (len > kMaxChunkLength || png.size() - pos - 12 < len) throw std::invalid_argument("read_png_text: bad chunk length");
    const std::uint8_t* type = png.data() + pos + 4;
    const std::uint8_t* data = type + 4;
    if (crc32(type, len + 4) != get_u32(data + len)) throw std::invalid_argument("read_png_text: CRC mismatch");
    const std::string_view kind(reinterpret_cast<const char*>(type), 4);
    const std::string_view body(reinterpret_cast<const char*>(data), len);
    if (kind == "tEXt" || kind == "iTXt") {
      const std::size_t nul = body.find('\0');
      if (nul == std::string_view::npos || nul == 0 || nul > 79) throw std::invalid_argument("read_png_text: bad keyword");
      PngText t;
      t.keyword = std::string(body.substr(0, nul));
      std::string_view rest = body.substr(nul + 1);
      bool keep = true;
      if (kind == "tEXt") {
        for (const char c : rest) {  // Latin-1 -> UTF-8
          const auto u = static_cast<unsigned char>(c);
          if (u < 0x80) {
            t.text.push_back(c);
          } else {
            t.text.push_back(static_cast<char>(0xc0 | (u >> 6)));
            t.text.push_back(static_cast<char>(0x80 | (u & 0x3f)));
          }
        }
      } else {
        if (rest.size() < 2) throw std::invalid_argument("read_png_text: bad iTXt chunk");
        keep = rest[0] == 0;  // compression flag
        rest = rest.substr(2);
        for (int field = 0; field < 2; ++field) {  // language tag, translated keyword
          const std::size_t z = rest.find('\0');
          if (z == std::string_view::npos) throw std::invalid_argument("read_png_text: bad iTXt chunk");
          rest = rest.substr(z + 1);
        }
        t.text = std::string(rest);
      }
      if (keep) out.push_back(std::move(t));
    }
    pos += 12 + static_cast<std::size_t>(len);
    if (kind == "IEND") break;
  }
  return out;
}

std::string generation_json(const GenerationMetadata& m) {
  std::string j = "{\"schema\": ";
  json_string(j, kGenerationSchema);
  j += ", \"engine\": {\"name\": \"moss-infer\"";
  if (!m.engine_version.empty()) {
    j += ", \"version\": ";
    json_string(j, m.engine_version);
  }
  if (!m.tier.empty()) {
    j += ", \"tier\": ";
    json_string(j, m.tier);
  }
  j += "}, \"models\": [";
  for (std::size_t i = 0; i < m.models.size(); ++i) {
    const auto& f = m.models[i];
    j += i > 0 ? ", {\"role\": " : "{\"role\": ";
    json_string(j, f.role);
    j += ", \"name\": ";
    json_string(j, base_name(f.name));
    if (!f.sha256.empty()) {
      j += ", \"sha256\": ";
      json_string(j, f.sha256);
    }
    j += "}";
  }
  j += "], \"loras\": [";
  for (std::size_t i = 0; i < m.loras.size(); ++i) {
    const auto& l = m.loras[i];
    j += i > 0 ? ", {\"name\": " : "{\"name\": ";
    json_string(j, base_name(l.name));
    if (!l.sha256.empty()) {
      j += ", \"sha256\": ";
      json_string(j, l.sha256);
    }
    j += ", \"strength\": " + number(l.strength) + "}";
  }
  j += "]";
  if (m.include_prompts) {
    j += ", \"prompt\": ";
    json_string(j, m.prompt);
    j += ", \"negative_prompt\": ";
    json_string(j, m.negative_prompt);
  }
  if (!m.sampler.empty()) {
    j += ", \"sampler\": ";
    json_string(j, m.sampler);
  }
  if (!m.scheduler.empty()) {
    j += ", \"scheduler\": ";
    json_string(j, m.scheduler);
  }
  j += ", \"steps\": " + std::to_string(m.steps) + ", \"cfg\": " + number(m.cfg) + ", \"shift\": " + number(m.shift) +
       ", \"seed\": " + std::to_string(m.seed) + ", \"width\": " + std::to_string(m.width) +
       ", \"height\": " + std::to_string(m.height);
  if (!m.extra.empty()) {
    j += ", \"extra\": {";
    for (std::size_t i = 0; i < m.extra.size(); ++i) {
      if (i > 0) j += ", ";
      json_string(j, m.extra[i].first);
      j += ": ";
      json_string(j, m.extra[i].second);
    }
    j += "}";
  }
  j += "}";
  return j;
}

std::string generation_parameters_text(const GenerationMetadata& m) {
  std::string t;
  if (m.include_prompts) {
    t += clean_text(m.prompt) + "\n";
    if (!m.negative_prompt.empty()) t += "Negative prompt: " + clean_text(m.negative_prompt) + "\n";
  }
  t += "Steps: " + std::to_string(m.steps);
  if (!m.sampler.empty()) t += ", Sampler: " + param_value(display_sampler(m.sampler));
  if (!m.scheduler.empty()) t += ", Schedule type: " + param_value(display_scheduler(m.scheduler));
  t += ", CFG scale: " + number(m.cfg) + ", Shift: " + number(m.shift) + ", Seed: " + std::to_string(m.seed) +
       ", Size: " + std::to_string(m.width) + "x" + std::to_string(m.height);
  for (const auto& f : m.models) {
    if (f.role != "dit") continue;
    if (!f.sha256.empty()) t += ", Model hash: " + param_value(short_hash(f.sha256));
    t += ", Model: " + param_value(base_name(f.name));
    break;
  }
  if (!m.loras.empty()) {
    std::string list;
    for (std::size_t i = 0; i < m.loras.size(); ++i) {
      if (i > 0) list += ", ";
      list += base_name(m.loras[i].name) + ": " + (m.loras[i].sha256.empty() ? std::string("unknown") : short_hash(m.loras[i].sha256));
    }
    t += ", Lora hashes: " + param_value(list);
  }
  for (const auto& [k, v] : m.extra) t += ", " + param_value(k) + ": " + param_value(v);
  if (!m.engine_version.empty()) t += ", Version: " + param_value("moss-infer " + m.engine_version);
  return t;
}

std::vector<PngText> generation_text_chunks(const GenerationMetadata& meta, bool json, bool parameters) {
  std::vector<PngText> out;
  if (json) out.push_back({kGenerationKeyword, generation_json(meta)});
  if (parameters) out.push_back({kParametersKeyword, generation_parameters_text(meta)});
  return out;
}

}  // namespace mi::pipeline
