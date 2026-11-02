#include "safetensors.h"

#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace mi::vk::st {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("safetensors: " + what); }

// Just enough JSON for safetensors headers and index files: objects, arrays, strings without \u escapes beyond ASCII
// passthrough, integers, true/false/null.
struct Json {
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject } kind = Kind::kNull;
  std::string text;  // string value, or the number's spelling
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

  Json parse() {
    Json j = value();
    space();
    if (i_ != s_.size()) fail("trailing characters in JSON");
    return j;
  }

 private:
  void space() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }
  char peek() {
    space();
    if (i_ >= s_.size()) fail("unexpected end of JSON");
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "' in JSON");
    ++i_;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (true) {
      if (i_ >= s_.size()) fail("unterminated JSON string");
      const char c = s_[i_++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= s_.size()) fail("unterminated JSON escape");
      const char e = s_[i_++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
          if (i_ + 4 > s_.size()) fail("short \\u escape");
          const unsigned code = static_cast<unsigned>(std::stoul(s_.substr(i_, 4), nullptr, 16));
          i_ += 4;
          // Names and dtypes are ASCII; anything else only has to survive as some bytes.
          if (code < 0x80) out += static_cast<char>(code);
          else out += '?';
          break;
        }
        default: out += e;
      }
    }
  }
  Json value() {
    Json j;
    const char c = peek();
    if (c == '{') {
      ++i_;
      j.kind = Json::Kind::kObject;
      if (peek() == '}') {
        ++i_;
        return j;
      }
      while (true) {
        std::string key = string();
        expect(':');
        j.members.emplace_back(std::move(key), value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect('}');
        return j;
      }
    }
    if (c == '[') {
      ++i_;
      j.kind = Json::Kind::kArray;
      if (peek() == ']') {
        ++i_;
        return j;
      }
      while (true) {
        j.items.push_back(value());
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect(']');
        return j;
      }
    }
    if (c == '"') {
      j.kind = Json::Kind::kString;
      j.text = string();
      return j;
    }
    if (s_.compare(i_, 4, "true") == 0 || s_.compare(i_, 5, "false") == 0) {
      j.kind = Json::Kind::kBool;
      j.text = s_[i_] == 't' ? "true" : "false";
      i_ += j.text.size();
      return j;
    }
    if (s_.compare(i_, 4, "null") == 0) {
      i_ += 4;
      return j;
    }
    const std::size_t start = i_;
    while (i_ < s_.size() && (std::strchr("+-0123456789.eE", s_[i_]) != nullptr)) ++i_;
    if (start == i_) fail("bad JSON value");
    j.kind = Json::Kind::kNumber;
    j.text = s_.substr(start, i_ - start);
    return j;
  }

  const std::string& s_;
  std::size_t i_ = 0;
};

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::uint64_t to_u64(const Json& j) {
  if (j.kind != Json::Kind::kNumber) fail("expected a number");
  return std::stoull(j.text);
}

std::size_t dtype_bytes(const std::string& dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 1;
  fail("unknown dtype " + dtype);
}

}  // namespace

Index::Index(const std::filesystem::path& path) {
  if (!std::filesystem::is_directory(path)) {
    add_file(path);
    return;
  }
  std::vector<std::filesystem::path> indexes, files;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    const std::string name = entry.path().filename().string();
    if (name.size() > 22 && name.ends_with(".safetensors.index.json")) indexes.push_back(entry.path());
    else if (name.ends_with(".safetensors")) files.push_back(entry.path());
  }
  if (indexes.size() > 1) fail(path.string() + " holds more than one index file");
  if (indexes.empty()) {
    if (files.size() != 1) fail(path.string() + " holds no index file and not exactly one .safetensors file");
    add_file(files.front());
    return;
  }
  const Json index = Parser(read_text(indexes.front())).parse();
  const Json* map = index.find("weight_map");
  if (map == nullptr || map->kind != Json::Kind::kObject) fail(indexes.front().string() + " has no weight_map");
  std::set<std::string> shards;
  for (const auto& [name, file] : map->members) shards.insert(file.text);
  for (const std::string& shard : shards) add_file(path / shard);
  for (const auto& [name, file] : map->members) {
    if (!has(name)) fail(name + " is in the index but in no shard");
  }
}

void Index::add_file(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) fail("cannot open " + file.string());
  std::uint64_t header_bytes = 0;
  unsigned char size[8];
  if (!in.read(reinterpret_cast<char*>(size), 8)) fail(file.string() + " is too short");
  for (int i = 7; i >= 0; --i) header_bytes = (header_bytes << 8) | size[i];
  if (header_bytes > (std::uint64_t{100} << 20)) fail(file.string() + " has an implausible header size");
  std::string header(header_bytes, '\0');
  if (!in.read(header.data(), static_cast<std::streamsize>(header_bytes))) fail(file.string() + " has a short header");
  const std::uint64_t data_start = 8 + header_bytes;
  const std::uint64_t file_bytes = std::filesystem::file_size(file);

  const Json root = Parser(header).parse();
  if (root.kind != Json::Kind::kObject) fail(file.string() + ": the header is not an object");
  for (const auto& [name, t] : root.members) {
    if (name == "__metadata__") continue;
    const Json* dtype = t.find("dtype");
    const Json* shape = t.find("shape");
    const Json* offsets = t.find("data_offsets");
    if (dtype == nullptr || shape == nullptr || offsets == nullptr || offsets->items.size() != 2) {
      fail(file.string() + ": malformed entry " + name);
    }
    TensorInfo info;
    info.dtype = dtype->text;
    std::uint64_t count = 1;
    for (const Json& d : shape->items) {
      info.shape.push_back(static_cast<std::int64_t>(to_u64(d)));
      count *= to_u64(d);
    }
    const std::uint64_t begin = to_u64(offsets->items[0]), end = to_u64(offsets->items[1]);
    if (end < begin || end - begin != count * dtype_bytes(info.dtype) || data_start + end > file_bytes) {
      fail(file.string() + ": bad extent for " + name);
    }
    info.offset = data_start + begin;
    info.bytes = end - begin;
    info.file = file;
    if (!tensors_.emplace(name, std::move(info)).second) fail(name + " appears twice");
  }
}

const TensorInfo& Index::at(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) fail("no tensor " + name);
  return it->second;
}

std::vector<std::byte> Index::read(const std::string& name) const {
  std::vector<std::byte> out(at(name).bytes);
  read_into(name, out.data(), out.size());
  return out;
}

void Index::read_into(const std::string& name, void* out, std::uint64_t bytes) const {
  const TensorInfo& info = at(name);
  if (bytes != info.bytes) fail(name + ": read of " + std::to_string(bytes) + " bytes, tensor has " + std::to_string(info.bytes));
  std::ifstream in(info.file, std::ios::binary);
  if (!in) fail("cannot open " + info.file.string());
  in.seekg(static_cast<std::streamoff>(info.offset));
  if (!in.read(static_cast<char*>(out), static_cast<std::streamsize>(bytes))) fail("short read of " + name);
}

}  // namespace mi::vk::st
