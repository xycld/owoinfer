// Minimal .npy reader and writer (format 1-3 headers, little-endian, C order) for the Vulkan backend's tools.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mi::vk::npy {

struct Array {
  std::string descr;  // e.g. "<f4", "<i8"
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;

  std::int64_t numel() const {
    std::int64_t n = 1;
    for (const std::int64_t d : shape) n *= d;
    return n;
  }
  template <class T>
  std::vector<T> as(const char* want) const {
    if (descr != want) throw std::runtime_error("npy: dtype " + descr + ", expected " + want);
    std::vector<T> out(data.size() / sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), data.data(), out.size() * sizeof(T));
    return out;
  }
};

inline Array load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("npy: cannot open " + path.string());
  char magic[8];
  if (!in.read(magic, 8) || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic in " + path.string());
  std::uint32_t header_len = 0;
  if (magic[6] == 1) {
    unsigned char b[2];
    in.read(reinterpret_cast<char*>(b), 2);
    header_len = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    header_len = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  Array a;
  const auto field = [&](const char* key) {
    const std::size_t at = header.find(key);
    if (at == std::string::npos) throw std::runtime_error(std::string("npy: no ") + key + " in " + path.string());
    return at + std::strlen(key);
  };
  std::size_t at = header.find('\'', field("'descr':")) + 1;
  a.descr = header.substr(at, header.find('\'', at) - at);
  if (header.find("True", field("'fortran_order':")) == field("'fortran_order':") + 1) {
    throw std::runtime_error("npy: Fortran order in " + path.string());
  }
  at = header.find('(', field("'shape':")) + 1;
  const std::size_t end = header.find(')', at);
  std::string dims = header.substr(at, end - at);
  std::size_t pos = 0;
  while (pos < dims.size()) {
    const std::size_t comma = dims.find(',', pos);
    const std::string item = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (item.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stoll(item));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  const std::size_t item_bytes = static_cast<std::size_t>(std::stoi(a.descr.substr(2)));
  a.data.resize(static_cast<std::size_t>(a.numel()) * item_bytes);
  if (!in.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(a.data.size()))) {
    throw std::runtime_error("npy: short data in " + path.string());
  }
  return a;
}

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
