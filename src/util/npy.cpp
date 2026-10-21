#include "util/npy.h"

#include <charconv>
#include <fstream>
#include <optional>
#include <string_view>

#include "util/checked_math.h"
#include "util/path.h"

namespace mi::npy {

namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
  throw std::runtime_error("npy: " + util::path_utf8(path) + ": " + what);
}

// Extracts the value text following 'key': in the header dict literal.
std::string_view dict_value(std::string_view header, std::string_view key) {
  std::string quoted = "'" + std::string(key) + "'";
  std::size_t at = header.find(quoted);
  if (at == std::string_view::npos) return {};
  std::size_t colon = header.find(':', at + quoted.size());
  if (colon == std::string_view::npos) return {};
  std::size_t start = header.find_first_not_of(' ', colon + 1);
  if (start == std::string_view::npos) return {};
  char open = header[start];
  std::size_t end;
  if (open == '\'') {
    end = header.find('\'', start + 1);
    return end == std::string_view::npos ? std::string_view{} : header.substr(start + 1, end - start - 1);
  }
  if (open == '(') {
    end = header.find(')', start);
    return end == std::string_view::npos ? std::string_view{} : header.substr(start + 1, end - start - 1);
  }
  end = header.find_first_of(",}", start);
  return header.substr(start, end - start);
}

// The whole of `text` as a non-negative decimal integer.
template <typename T>
std::optional<T> parse_uint(std::string_view text) {
  T v{};
  auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), v);
  if (ec != std::errc() || ptr != text.data() + text.size()) return std::nullopt;
  if constexpr (std::is_signed_v<T>) {
    if (v < 0) return std::nullopt;
  }
  return v;
}

}  // namespace

std::size_t Array::itemsize() const {
  std::optional<std::size_t> n = parse_uint<std::size_t>(std::string_view(dtype).substr(dtype.empty() ? 0 : 1));
  if (!n) throw std::runtime_error("npy: malformed dtype '" + dtype + "'");
  return *n;
}

std::int64_t Array::numel() const {
  std::int64_t n = 1;
  for (std::int64_t d : shape) n *= d;
  return n;
}

Array load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path, "cannot open");
  in.seekg(0, std::ios::end);
  const auto file_size = static_cast<std::uint64_t>(in.tellg());
  in.seekg(0);
  char magic[8];
  if (!in.read(magic, 8) || std::string_view(magic, 6) != "\x93NUMPY") fail(path, "bad magic");
  int major = static_cast<unsigned char>(magic[6]);
  std::uint32_t header_len = 0;
  if (major == 1) {
    unsigned char b[2];
    if (!in.read(reinterpret_cast<char*>(b), 2)) fail(path, "truncated header length");
    header_len = static_cast<std::uint32_t>(b[0] | (b[1] << 8));
  } else if (major == 2 || major == 3) {
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4)) fail(path, "truncated header length");
    header_len = static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
                 (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
  } else {
    fail(path, "unsupported format version " + std::to_string(major));
  }
  if (header_len > file_size - static_cast<std::uint64_t>(in.tellg())) fail(path, "truncated header");
  std::string header(header_len, '\0');
  if (!in.read(header.data(), header_len)) fail(path, "truncated header");

  std::string_view descr = dict_value(header, "descr");
  if (descr.size() < 3) fail(path, "missing descr");
  if (descr[0] != '<' && descr[0] != '|') fail(path, "only little-endian arrays are supported");
  if (dict_value(header, "fortran_order").starts_with("True")) fail(path, "fortran order is not supported");

  Array arr;
  arr.dtype = std::string(descr.substr(1));
  char kind = arr.dtype[0];
  if (kind != 'f' && kind != 'i' && kind != 'u' && kind != 'b') fail(path, "unsupported dtype " + arr.dtype);
  const std::optional<std::size_t> itemsize = parse_uint<std::size_t>(std::string_view(arr.dtype).substr(1));
  if (!itemsize || (*itemsize != 1 && *itemsize != 2 && *itemsize != 4 && *itemsize != 8)) fail(path, "unsupported dtype " + arr.dtype);

  std::string_view shape = dict_value(header, "shape");
  std::size_t pos = 0;
  while (pos < shape.size()) {
    std::size_t next = shape.find(',', pos);
    std::string_view item = shape.substr(pos, next == std::string_view::npos ? std::string_view::npos : next - pos);
    std::size_t first = item.find_first_not_of(' ');
    if (first != std::string_view::npos) {
      std::size_t last = item.find_last_not_of(' ');
      std::optional<std::int64_t> dim = parse_uint<std::int64_t>(item.substr(first, last - first + 1));
      if (!dim) fail(path, "malformed shape '" + std::string(shape) + "'");
      arr.shape.push_back(*dim);
    }
    if (next == std::string_view::npos) break;
    pos = next + 1;
  }

  const std::optional<std::int64_t> numel = util::checked_numel(arr.shape);
  const std::optional<std::uint64_t> bytes = numel ? util::checked_mul(static_cast<std::uint64_t>(*numel), *itemsize) : std::nullopt;
  if (!bytes) fail(path, "element count overflows");
  if (*bytes > file_size - static_cast<std::uint64_t>(in.tellg())) fail(path, "truncated data");
  arr.data.resize(static_cast<std::size_t>(*bytes));
  if (!in.read(reinterpret_cast<char*>(arr.data.data()), static_cast<std::streamsize>(*bytes))) fail(path, "truncated data");
  return arr;
}

}  // namespace mi::npy
