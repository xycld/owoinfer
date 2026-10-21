// Shared test helpers: scratch directories, synthetic safetensors files, timing-assertion policy.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "util/path.h"
#include "weights/safetensors.h"

#ifdef _WIN32
#include <process.h>
#define MI_GETPID _getpid
#else
#include <unistd.h>
#define MI_GETPID getpid
#endif

#if defined(__SANITIZE_ADDRESS__)
#define MI_TEST_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MI_TEST_ASAN 1
#endif
#endif

namespace mi::testing {

// Wall-clock limits only mean something in optimized, uninstrumented builds. Elsewhere perf tests
// still run and record their timings (RecordProperty) but do not assert them.
#if defined(NDEBUG) && !defined(MI_TEST_ASAN)
inline constexpr bool kAssertTimings = true;
#else
inline constexpr bool kAssertTimings = false;
#endif

// Created beside the test binary (the real file system, so O_DIRECT works; /tmp is often tmpfs).
class TempDir {
 public:
  explicit TempDir(const std::string& tag) {
    path_ = std::filesystem::current_path() / util::path_from_utf8("mi_test_" + tag + "_" + std::to_string(MI_GETPID()));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

// Byte at absolute position `i` of a synthetic data section.
inline std::uint8_t pattern_byte(std::uint64_t i) { return static_cast<std::uint8_t>((i * 131 + 7) >> 1); }

inline void write_raw(const std::filesystem::path& path, const std::string& header, std::uint64_t data_bytes,
                      std::uint64_t declared_header_len = ~0ull) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  std::uint64_t len = declared_header_len == ~0ull ? header.size() : declared_header_len;
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (std::uint64_t i = 0; i < data_bytes; ++i) out.put(static_cast<char>(pattern_byte(i)));
}

struct TensorSpec {
  std::string name;
  weights::DType dtype;
  std::vector<std::int64_t> shape;
};

// Writes a valid file with the given tensors packed in order; data bytes follow pattern_byte().
inline void write_safetensors(const std::filesystem::path& path, const std::vector<TensorSpec>& specs,
                              const std::string& metadata_json = "") {
  std::string header = "{";
  if (!metadata_json.empty()) header += "\"__metadata__\":" + metadata_json + ",";
  std::uint64_t cursor = 0;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    std::uint64_t n = weights::dtype_size(specs[i].dtype);
    std::string shape = "[";
    for (std::size_t d = 0; d < specs[i].shape.size(); ++d) {
      shape += (d ? "," : "") + std::to_string(specs[i].shape[d]);
      n *= static_cast<std::uint64_t>(specs[i].shape[d]);
    }
    shape += "]";
    header += (i ? "," : "") + std::string("\"") + specs[i].name + "\":{\"dtype\":\"" + weights::dtype_name(specs[i].dtype) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(cursor) + "," + std::to_string(cursor + n) + "]}";
    cursor += n;
  }
  header += "}";
  write_raw(path, header, cursor);
}

}  // namespace mi::testing
