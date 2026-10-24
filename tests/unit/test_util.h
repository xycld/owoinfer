// Shared test helpers: scratch directories, synthetic safetensors files, timing-assertion policy.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "util/path.h"
#include "weights/safetensors.h"
#include "weights/test_seams.h"

#ifdef _WIN32
#include <process.h>
#define MI_GETPID _getpid
#else
#include <fcntl.h>
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

// A tensor of write_sparse_safetensors: `content` holds its bytes (exactly its size), or is empty for a hole.
struct SparseTensor {
  std::string name;
  weights::DType dtype;
  std::vector<std::int64_t> shape;
  std::string content;
};

// A valid safetensors file whose data section is holes except for the tensors with content (larger dtypes first, so
// every tensor is aligned to its element size): header-only tests of full-size shapes cost no disk, and a read of a
// hole returns zeros (an invalid scale), which tells a read apart from a refusal made before reading.
inline void write_sparse_safetensors(const std::filesystem::path& path, std::vector<SparseTensor> ts,
                                     const std::string& metadata_json = "") {
  std::stable_sort(ts.begin(), ts.end(), [](const SparseTensor& a, const SparseTensor& b) {
    return weights::dtype_size(a.dtype) > weights::dtype_size(b.dtype);
  });
  std::string h = "{";
  if (!metadata_json.empty()) h += "\"__metadata__\":" + metadata_json;
  std::uint64_t off = 0;
  std::vector<std::uint64_t> offsets;
  for (const SparseTensor& t : ts) {
    std::uint64_t bytes = weights::dtype_size(t.dtype);
    for (const std::int64_t d : t.shape) bytes *= static_cast<std::uint64_t>(d);
    if (h.size() > 1) h += ",";
    h += "\"" + t.name + "\":{\"dtype\":\"" + weights::dtype_name(t.dtype) + "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
    h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
    offsets.push_back(off);
    off += bytes;
  }
  h += "}";
  h.append((8 - h.size() % 8) % 8, ' ');
  std::filesystem::create_directories(path.parent_path());
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t len = h.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (std::size_t i = 0; i < ts.size(); ++i) {
      if (ts[i].content.empty()) continue;
      out.seekp(static_cast<std::streamoff>(8 + h.size() + offsets[i]));
      out.write(ts[i].content.data(), static_cast<std::streamsize>(ts[i].content.size()));
    }
  }
  std::filesystem::resize_file(path, 8 + h.size() + off);
}

#ifndef _WIN32
// What `f` threw (its what(); "" when it returned), run on its own thread for at most `timeout`: a FIFO given where a
// file is expected must be refused at once, never waited on. On a timeout a writer end of `fifo` is opened, which
// releases an open(2) blocked on the FIFO, and nullopt is returned: the caller's expectation fails instead of the test
// hanging.
inline std::optional<std::string> error_within(const std::filesystem::path& fifo, std::function<void()> f,
                                               std::chrono::seconds timeout = std::chrono::seconds(10)) {
  auto done = std::make_shared<std::promise<std::string>>();
  std::future<std::string> result = done->get_future();
  std::thread([done, f = std::move(f)] {
    try {
      f();
      done->set_value("");
    } catch (const std::exception& e) {
      done->set_value(e.what());
    } catch (...) {
      done->set_value("a non-standard exception");
    }
  }).detach();
  if (result.wait_for(timeout) == std::future_status::ready) return result.get();
  const int w = ::open(fifo.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
  if (w >= 0) ::close(w);
  // The released call must finish before the caller's locals (which `f` references) go away.
  if (result.wait_for(timeout) != std::future_status::ready) {
    std::fprintf(stderr, "error_within: a call on %s is still blocked after releasing the FIFO; aborting\n",
                 fifo.c_str());
    std::abort();
  }
  return std::nullopt;
}
#endif

// While it lives, the quantized importer reads linears and text-encoder tensors of any shape (weights/test_seams.h):
// for tests on small synthetic models, whose shapes are not MOSS-Image's. Nests: it restores the previous setting.
class AnyModelShapes {
 public:
  AnyModelShapes() : previous_(weights::testing::any_model_shapes()) { weights::testing::allow_any_model_shapes(true); }
  ~AnyModelShapes() { weights::testing::allow_any_model_shapes(previous_); }
  AnyModelShapes(const AnyModelShapes&) = delete;
  AnyModelShapes& operator=(const AnyModelShapes&) = delete;

 private:
  bool previous_;
};

}  // namespace mi::testing
