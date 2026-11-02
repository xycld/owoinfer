#include "safetensors.h"

#include <fcntl.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>

#include <mutex>
#else
#include <unistd.h>
#endif

namespace mi::vk::st {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("safetensors: " + what); }

// The JSON subset safetensors headers and index files use: objects, arrays, strings, numbers, true/false/null.
struct Json {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject } kind = Kind::kNull;
  double number = 0;
  std::int64_t integer = 0;  // exact for integral numbers
  std::string text;
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json>> members;

  const Json* find(const std::string& key) const {
    for (const auto& [k, v] : members) {
      if (k == key) return &v;
    }
    return nullptr;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}

  Json parse_document() {
    Json v = value();
    skip();
    if (i_ != s_.size()) fail("trailing bytes after the JSON value");
    return v;
  }

 private:
  void skip() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }
  char peek() {
    skip();
    if (i_ >= s_.size()) fail("unexpected end of JSON");
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "' in JSON");
    ++i_;
  }

  Json value() {
    const char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      Json v;
      v.kind = Json::Kind::kString;
      v.text = string();
      return v;
    }
    if (s_.compare(i_, 4, "true") == 0 || s_.compare(i_, 5, "false") == 0) {
      Json v;
      v.kind = Json::Kind::kBool;
      v.integer = s_[i_] == 't';
      i_ += s_[i_] == 't' ? 4 : 5;
      return v;
    }
    if (s_.compare(i_, 4, "null") == 0) {
      i_ += 4;
      return {};
    }
    return number();
  }

  Json object() {
    Json v;
    v.kind = Json::Kind::kObject;
    expect('{');
    if (peek() == '}') {
      ++i_;
      return v;
    }
    for (;;) {
      std::string key = (peek(), string());
      expect(':');
      v.members.emplace_back(std::move(key), value());
      if (peek() == ',') {
        ++i_;
        continue;
      }
      expect('}');
      return v;
    }
  }

  Json array() {
    Json v;
    v.kind = Json::Kind::kArray;
    expect('[');
    if (peek() == ']') {
      ++i_;
      return v;
    }
    for (;;) {
      v.items.push_back(value());
      if (peek() == ',') {
        ++i_;
        continue;
      }
      expect(']');
      return v;
    }
  }

  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) break;
        const char e = s_[i_++];
        switch (e) {
          case 'n':
            c = '\n';
            break;
          case 't':
            c = '\t';
            break;
          case 'r':
            c = '\r';
            break;
          case 'b':
            c = '\b';
            break;
          case 'f':
            c = '\f';
            break;
          case 'u': {
            // Names here are ASCII; other code points are kept as UTF-8 of the BMP value.
            if (i_ + 4 > s_.size()) fail("bad \\u escape");
            const unsigned cp = static_cast<unsigned>(std::stoul(s_.substr(i_, 4), nullptr, 16));
            i_ += 4;
            if (cp < 0x80) {
              out += static_cast<char>(cp);
            } else if (cp < 0x800) {
              out += static_cast<char>(0xC0 | (cp >> 6));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (cp >> 12));
              out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (cp & 0x3F));
            }
            continue;
          }
          default:
            c = e;
            break;
        }
      }
      out += c;
    }
    if (i_ >= s_.size()) fail("unterminated JSON string");
    ++i_;
    return out;
  }

  Json number() {
    const std::size_t start = i_;
    bool integral = true;
    while (i_ < s_.size() && (std::strchr("+-0123456789.eE", s_[i_]) != nullptr)) {
      if (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E') integral = false;
      ++i_;
    }
    if (start == i_) fail("unexpected character in JSON");
    Json v;
    v.kind = Json::Kind::kNumber;
    const std::string text = s_.substr(start, i_ - start);
    v.number = std::stod(text);
    v.integer = integral ? std::stoll(text) : static_cast<std::int64_t>(v.number);
    return v;
  }

  const std::string& s_;
  std::size_t i_ = 0;
};

std::uint64_t dtype_bytes(const std::string& dtype) {
  if (dtype == "BF16" || dtype == "F16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 1;
  return 0;
}

#ifdef _WIN32
std::mutex g_read_mutex;  // _lseeki64 + _read share the descriptor's position
#endif

void read_at(int fd, void* dst, std::uint64_t offset, std::uint64_t bytes, const std::string& path) {
  auto* out = static_cast<char*>(dst);
  while (bytes > 0) {
    const std::uint64_t want = bytes < (std::uint64_t{1} << 30) ? bytes : (std::uint64_t{1} << 30);
#ifdef _WIN32
    std::lock_guard<std::mutex> lock(g_read_mutex);
    if (_lseeki64(fd, static_cast<__int64>(offset), SEEK_SET) < 0) fail("seek failed in " + path);
    const int got = _read(fd, out, static_cast<unsigned>(want));
#else
    const ssize_t got = ::pread(fd, out, static_cast<std::size_t>(want), static_cast<off_t>(offset));
#endif
    if (got <= 0) fail("read failed or hit the end of " + path);
    out += got;
    offset += static_cast<std::uint64_t>(got);
    bytes -= static_cast<std::uint64_t>(got);
  }
}

void close_fd(int fd) {
  if (fd < 0) return;
#ifdef _WIN32
  _close(fd);
#else
  ::close(fd);
#endif
}

}  // namespace

File::File(const std::string& path) : path_(path) {
#ifdef _WIN32
  fd_ = _open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
  if (fd_ < 0) fail("cannot open " + path);
  try {
    std::uint64_t header_bytes = 0;
    unsigned char le[8];
    read_at(fd_, le, 0, 8, path);
    for (int i = 7; i >= 0; --i) header_bytes = (header_bytes << 8) | le[i];
    if (header_bytes == 0 || header_bytes > (std::uint64_t{100} << 20)) fail("implausible header size in " + path);
    std::string header(static_cast<std::size_t>(header_bytes), '\0');
    read_at(fd_, header.data(), 8, header_bytes, path);
    const Json root = Parser(header).parse_document();
    if (root.kind != Json::Kind::kObject) fail("header of " + path + " is not an object");
    const std::uint64_t data_start = 8 + header_bytes;
    for (const auto& [name, v] : root.members) {
      if (name == "__metadata__") continue;
      const Json* dtype = v.find("dtype");
      const Json* shape = v.find("shape");
      const Json* offsets = v.find("data_offsets");
      if (dtype == nullptr || shape == nullptr || offsets == nullptr || dtype->kind != Json::Kind::kString ||
          shape->kind != Json::Kind::kArray || offsets->kind != Json::Kind::kArray || offsets->items.size() != 2) {
        fail("malformed entry " + name + " in " + path);
      }
      TensorInfo t;
      t.name = name;
      t.dtype = dtype->text;
      std::uint64_t numel = 1;
      for (const Json& d : shape->items) {
        if (d.kind != Json::Kind::kNumber || d.integer < 0) fail("bad shape of " + name);
        t.shape.push_back(d.integer);
        numel *= static_cast<std::uint64_t>(d.integer);
      }
      const std::int64_t begin = offsets->items[0].integer, end = offsets->items[1].integer;
      if (begin < 0 || end < begin) fail("bad data_offsets of " + name);
      t.offset = data_start + static_cast<std::uint64_t>(begin);
      t.bytes = static_cast<std::uint64_t>(end - begin);
      const std::uint64_t item = dtype_bytes(t.dtype);
      if (item != 0 && numel * item != t.bytes) fail("size of " + name + " does not match its dtype and shape");
      by_name_[t.name] = tensors_.size();
      tensors_.push_back(std::move(t));
    }
  } catch (...) {
    close_fd(fd_);  // the destructor does not run for a constructor that throws
    throw;
  }
}

File::~File() { close_fd(fd_); }

const TensorInfo* File::find(const std::string& name) const {
  const auto it = by_name_.find(name);
  return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo& File::at(const std::string& name) const {
  const TensorInfo* t = find(name);
  if (t == nullptr) fail(path_ + " has no tensor " + name);
  return *t;
}

void File::read(const TensorInfo& t, void* dst, std::uint64_t rel, std::uint64_t bytes) const {
  if (rel > t.bytes || bytes > t.bytes - rel) fail("read past the end of tensor " + t.name);
  read_at(fd_, dst, t.offset + rel, bytes, path_);
}

std::map<std::string, std::string> read_index(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  const Json root = Parser(text).parse_document();
  const Json* map = root.find("weight_map");
  if (map == nullptr || map->kind != Json::Kind::kObject) fail(path + " has no weight_map object");
  std::map<std::string, std::string> out;
  for (const auto& [name, file] : map->members) {
    if (file.kind != Json::Kind::kString) fail("weight_map entry " + name + " is not a string");
    out[name] = file.text;
  }
  return out;
}

}  // namespace mi::vk::st
