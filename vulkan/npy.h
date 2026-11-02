// Reader for NumPy .npy files (format versions 1-3): little-endian, C-order arrays of fixed-size numeric dtypes. A copy
// of the engine's util/npy, header-only and without its helpers, for the checks that compare against golden data.
#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace mi::vk::npy {

struct Array {
  std::string dtype;  // numpy descr without byte order, e.g. "f4", "i8", "u2"
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;  // raw element bytes, C order

  std::int64_t numel() const {
    std::int64_t n = 1;
    for (const std::int64_t d : shape) n *= d;
    return n;
  }

  // Copies the elements out as T. Throws unless dtype is exactly T's kind and size, e.g. "f4" for float.
  template <typename T>
  std::vector<T> to_vector() const {
    static_assert(std::is_arithmetic_v<T>);
    constexpr char kind = std::is_same_v<T, bool>       ? 'b'
                          : std::is_floating_point_v<T> ? 'f'
                          : std::is_signed_v<T>         ? 'i'
                                                        : 'u';
    if (dtype != kind + std::to_string(sizeof(T))) {
      throw std::runtime_error("npy: array of dtype " + dtype + " read as " + kind + std::to_string(sizeof(T)));
    }
    std::vector<T> out(data.size() / sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), data.data(), out.size() * sizeof(T));
    return out;
  }
};

namespace detail {

[[noreturn]] inline void fail(const std::filesystem::path& path, const std::string& what) {
  throw std::runtime_error("npy: " + path.string() + ": " + what);
}

// The value text following 'key': in the header dict literal.
inline std::string_view dict_value(std::string_view header, std::string_view key) {
  const std::string quoted = "'" + std::string(key) + "'";
  const std::size_t at = header.find(quoted);
  if (at == std::string_view::npos) return {};
  const std::size_t colon = header.find(':', at + quoted.size());
  if (colon == std::string_view::npos) return {};
  const std::size_t start = header.find_first_not_of(' ', colon + 1);
  if (start == std::string_view::npos) return {};
  const char open = header[start];
  if (open == '\'') {
    const std::size_t end = header.find('\'', start + 1);
    return end == std::string_view::npos ? std::string_view{} : header.substr(start + 1, end - start - 1);
  }
  if (open == '(') {
    const std::size_t end = header.find(')', start);
    return end == std::string_view::npos ? std::string_view{} : header.substr(start + 1, end - start - 1);
  }
  const std::size_t end = header.find_first_of(",}", start);
  return header.substr(start, end - start);
}

template <typename T>
std::optional<T> parse_uint(std::string_view text) {
  T v{};
  const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), v);
  if (ec != std::errc() || ptr != text.data() + text.size()) return std::nullopt;
  if constexpr (std::is_signed_v<T>) {
    if (v < 0) return std::nullopt;
  }
  return v;
}

}  // namespace detail

// Throws std::runtime_error on unsupported or malformed files.
inline Array load(const std::filesystem::path& path) {
  using detail::fail;
  std::ifstream in(path, std::ios::binary);
  if (!in) fail(path, "cannot open");
  in.seekg(0, std::ios::end);
  const auto file_size = static_cast<std::uint64_t>(in.tellg());
  in.seekg(0);
  char magic[8];
  if (!in.read(magic, 8) || std::string_view(magic, 6) != "\x93NUMPY") fail(path, "bad magic");
  const int major = static_cast<unsigned char>(magic[6]);
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

  const std::string_view descr = detail::dict_value(header, "descr");
  if (descr.size() < 3) fail(path, "missing descr");
  if (descr[0] != '<' && descr[0] != '|') fail(path, "only little-endian arrays are supported");
  if (detail::dict_value(header, "fortran_order").starts_with("True")) fail(path, "fortran order is not supported");

  Array arr;
  arr.dtype = std::string(descr.substr(1));
  const char kind = arr.dtype[0];
  if (kind != 'f' && kind != 'i' && kind != 'u' && kind != 'b') fail(path, "unsupported dtype " + arr.dtype);
  const std::optional<std::size_t> itemsize = detail::parse_uint<std::size_t>(std::string_view(arr.dtype).substr(1));
  if (!itemsize || (*itemsize != 1 && *itemsize != 2 && *itemsize != 4 && *itemsize != 8)) {
    fail(path, "unsupported dtype " + arr.dtype);
  }

  const std::string_view shape = detail::dict_value(header, "shape");
  std::size_t pos = 0;
  while (pos < shape.size()) {
    const std::size_t next = shape.find(',', pos);
    const std::string_view item =
        shape.substr(pos, next == std::string_view::npos ? std::string_view::npos : next - pos);
    const std::size_t first = item.find_first_not_of(' ');
    if (first != std::string_view::npos) {
      const std::size_t last = item.find_last_not_of(' ');
      const std::optional<std::int64_t> dim = detail::parse_uint<std::int64_t>(item.substr(first, last - first + 1));
      if (!dim) fail(path, "malformed shape '" + std::string(shape) + "'");
      arr.shape.push_back(*dim);
    }
    if (next == std::string_view::npos) break;
    pos = next + 1;
  }

  std::uint64_t bytes = *itemsize;
  for (const std::int64_t d : arr.shape) {
    if (d != 0 && bytes > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(d)) {
      fail(path, "element count overflows");
    }
    bytes *= static_cast<std::uint64_t>(d);
  }
  if (bytes > file_size - static_cast<std::uint64_t>(in.tellg())) fail(path, "truncated data");
  arr.data.resize(static_cast<std::size_t>(bytes));
  if (!in.read(reinterpret_cast<char*>(arr.data.data()), static_cast<std::streamsize>(bytes)))
    fail(path, "truncated data");
  return arr;
}

// Writes a little-endian C-order array (format 1.0); descr e.g. "<f4".
inline void save(const std::filesystem::path& path, const char* descr, const std::vector<std::int64_t>& shape,
                 const void* data, std::size_t bytes) {
  std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (";
  for (const std::int64_t d : shape) header += std::to_string(d) + ", ";
  header += "), }";
  while ((10 + header.size() + 1) % 64 != 0) header += ' ';
  header += '\n';
  std::ofstream out(path, std::ios::binary);
  out.write("\x93NUMPY\x01\x00", 8);
  const unsigned char len[2] = {static_cast<unsigned char>(header.size() & 0xff), static_cast<unsigned char>(header.size() >> 8)};
  out.write(reinterpret_cast<const char*>(len), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!out) throw std::runtime_error("npy: cannot write " + path.string());
}

}  // namespace mi::vk::npy
