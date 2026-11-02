#include "safetensors.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mi::vk::st {
namespace {

// The header is one JSON object: {"name": {"dtype": "F32", "shape": [..], "data_offsets": [begin, end]}, ...}, with an
// optional "__metadata__" object of strings. The parser accepts general JSON and keeps only what the entries need.
class Parser {
 public:
  Parser(const std::string& text, const std::string& path) : s_(text), path_(path) {}

  std::vector<TensorInfo> parse_header() {
    std::vector<TensorInfo> out;
    expect('{');
    if (peek() == '}') {
      ++i_;
      return out;
    }
    for (;;) {
      const std::string name = string();
      expect(':');
      if (name == "__metadata__") {
        skip_value();
      } else {
        out.push_back(entry(name));
      }
      const char c = next();
      if (c == '}') break;
      if (c != ',') fail("expected , or } after an entry");
    }
    ws();
    if (i_ != s_.size()) fail("trailing bytes after the header object");
    return out;
  }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("safetensors " + path_ + ": header: " + what + " at byte " + std::to_string(i_));
  }
  void ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) fail("unexpected end");
    return s_[i_];
  }
  char next() {
    const char c = peek();
    ++i_;
    return c;
  }
  void expect(char c) {
    if (next() != c) fail(std::string("expected '") + c + "'");
  }

  std::string string() {
    expect('"');
    std::string out;
    for (;;) {
      if (i_ >= s_.size()) fail("unterminated string");
      const char c = s_[i_++];
      if (c == '"') return out;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (i_ >= s_.size()) fail("unterminated escape");
      const char e = s_[i_++];
      switch (e) {
        case '"': case '\\': case '/': out.push_back(e); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          // Tensor names are ASCII in practice; other code points are kept as UTF-8 of the BMP value.
          if (i_ + 4 > s_.size()) fail("short \\u escape");
          const unsigned v = static_cast<unsigned>(std::stoul(s_.substr(i_, 4), nullptr, 16));
          i_ += 4;
          if (v < 0x80) {
            out.push_back(static_cast<char>(v));
          } else if (v < 0x800) {
            out.push_back(static_cast<char>(0xc0 | (v >> 6)));
            out.push_back(static_cast<char>(0x80 | (v & 0x3f)));
          } else {
            out.push_back(static_cast<char>(0xe0 | (v >> 12)));
            out.push_back(static_cast<char>(0x80 | ((v >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (v & 0x3f)));
          }
          break;
        }
        default: fail("bad escape");
      }
    }
  }

  std::uint64_t integer() {
    ws();
    if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') fail("expected a non-negative integer");
    std::uint64_t v = 0;
    while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') {
      const auto d = static_cast<std::uint64_t>(s_[i_++] - '0');
      if (v > (std::numeric_limits<std::uint64_t>::max() - d) / 10) fail("integer overflow");
      v = v * 10 + d;
    }
    return v;
  }

  std::vector<std::uint64_t> integer_array() {
    std::vector<std::uint64_t> out;
    expect('[');
    if (peek() == ']') {
      ++i_;
      return out;
    }
    for (;;) {
      out.push_back(integer());
      const char c = next();
      if (c == ']') return out;
      if (c != ',') fail("expected , or ] in an array");
    }
  }

  void skip_value() {
    const char c = peek();
    if (c == '"') {
      string();
    } else if (c == '{' || c == '[') {
      const char close = c == '{' ? '}' : ']';
      ++i_;
      if (peek() == close) {
        ++i_;
        return;
      }
      for (;;) {
        if (c == '{') {
          string();
          expect(':');
        }
        skip_value();
        const char d = next();
        if (d == close) return;
        if (d != ',') fail("expected , in a container");
      }
    } else {
      // number, true, false, null
      ws();
      const std::size_t start = i_;
      while (i_ < s_.size() && std::strchr(",}] \n\r\t", s_[i_]) == nullptr) ++i_;
      if (i_ == start) fail("expected a value");
    }
  }

  TensorInfo entry(const std::string& name) {
    TensorInfo t;
    t.name = name;
    bool has_dtype = false, has_shape = false, has_offsets = false;
    std::vector<std::uint64_t> offsets;
    expect('{');
    for (;;) {
      const std::string key = string();
      expect(':');
      if (key == "dtype") {
        t.dtype = string();
        has_dtype = true;
      } else if (key == "shape") {
        for (const std::uint64_t d : integer_array()) {
          if (d > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) fail("dimension too large");
          t.shape.push_back(static_cast<std::int64_t>(d));
        }
        has_shape = true;
      } else if (key == "data_offsets") {
        offsets = integer_array();
        has_offsets = true;
      } else {
        skip_value();
      }
      const char c = next();
      if (c == '}') break;
      if (c != ',') fail("expected , or } in a tensor entry");
    }
    if (!has_dtype || !has_shape || !has_offsets || offsets.size() != 2 || offsets[1] < offsets[0]) {
      fail("tensor '" + name + "' lacks dtype, shape or a valid data_offsets pair");
    }
    t.offset = offsets[0];
    t.bytes = offsets[1] - offsets[0];
    return t;
  }

  const std::string& s_;
  const std::string& path_;
  std::size_t i_ = 0;
};

std::uint64_t dtype_bytes(const std::string& dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 1;
  return 0;  // unknown: the size check is skipped
}

}  // namespace

File::File(const std::string& path) : path_(path), in_(path, std::ios::binary) {
  if (!in_) throw std::runtime_error("safetensors: cannot open " + path);
  in_.seekg(0, std::ios::end);
  file_bytes_ = static_cast<std::uint64_t>(in_.tellg());
  in_.seekg(0);
  unsigned char len_bytes[8] = {};
  if (file_bytes_ < 8 || !in_.read(reinterpret_cast<char*>(len_bytes), 8)) {
    throw std::runtime_error("safetensors " + path + ": shorter than its 8-byte header length");
  }
  std::uint64_t header_bytes = 0;
  for (int i = 7; i >= 0; --i) header_bytes = header_bytes << 8 | len_bytes[i];
  // The format caps the header at 100 MB.
  if (header_bytes > 100'000'000 || header_bytes > file_bytes_ - 8) {
    throw std::runtime_error("safetensors " + path + ": header length " + std::to_string(header_bytes) + " is out of range");
  }
  std::string header(static_cast<std::size_t>(header_bytes), '\0');
  if (!in_.read(header.data(), static_cast<std::streamsize>(header_bytes))) {
    throw std::runtime_error("safetensors " + path + ": cannot read the header");
  }
  tensors_ = Parser(header, path_).parse_header();
  const std::uint64_t data_start = 8 + header_bytes, data_bytes = file_bytes_ - data_start;
  for (TensorInfo& t : tensors_) {
    if (t.offset > data_bytes || t.bytes > data_bytes - t.offset) {
      throw std::runtime_error("safetensors " + path + ": tensor '" + t.name + "' lies outside the file");
    }
    std::uint64_t count = 1;
    for (const std::int64_t d : t.shape) {
      const auto u = static_cast<std::uint64_t>(d);
      if (u != 0 && count > std::numeric_limits<std::uint64_t>::max() / u) {
        throw std::runtime_error("safetensors " + path + ": tensor '" + t.name + "' has too many elements");
      }
      count *= u;
    }
    const std::uint64_t element = dtype_bytes(t.dtype);
    if (element != 0 && (count > std::numeric_limits<std::uint64_t>::max() / element || count * element != t.bytes)) {
      throw std::runtime_error("safetensors " + path + ": tensor '" + t.name + "' byte range does not match its shape");
    }
    t.offset += data_start;
  }
}

const TensorInfo* File::find(const std::string& name) const {
  const auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const TensorInfo& t) { return t.name == name; });
  return it == tensors_.end() ? nullptr : &*it;
}

void File::read(const TensorInfo& info, void* dst) {
  in_.clear();
  in_.seekg(static_cast<std::streamoff>(info.offset));
  if (!in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(info.bytes))) {
    throw std::runtime_error("safetensors " + path_ + ": cannot read tensor '" + info.name + "'");
  }
}

std::vector<float> File::read_f32(const std::string& name) {
  const TensorInfo* t = find(name);
  if (t == nullptr) throw std::runtime_error("safetensors " + path_ + ": no tensor '" + name + "'");
  if (t->dtype == "F32") {
    std::vector<float> out(static_cast<std::size_t>(t->bytes / 4));
    read(*t, out.data());
    return out;
  }
  if (t->dtype == "BF16") {
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(t->bytes / 2));
    read(*t, raw.data());
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const std::uint32_t bits = static_cast<std::uint32_t>(raw[i]) << 16;
      std::memcpy(&out[i], &bits, 4);
    }
    return out;
  }
  throw std::runtime_error("safetensors " + path_ + ": tensor '" + name + "' is " + t->dtype + ", not F32 or BF16");
}

}  // namespace mi::vk::st
