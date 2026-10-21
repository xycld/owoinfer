#include "pipeline/png_writer.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

#include "util/path.h"

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

}  // namespace

std::vector<std::uint8_t> encode_png_rgb8(std::span<const std::uint8_t> rgb, int width, int height) {
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
  for (std::size_t pos = 0; pos < z.size(); pos += kMaxIdatChunk) {
    chunk(png, "IDAT", z.data() + pos, std::min(kMaxIdatChunk, z.size() - pos));
  }
  chunk(png, "IEND", nullptr, 0);
  return png;
}

void write_png_rgb8(const std::filesystem::path& path, std::span<const std::uint8_t> rgb, int width, int height) {
  const std::vector<std::uint8_t> png = encode_png_rgb8(rgb, width, height);
  // Write a sibling temporary file and rename it over `path`, so a failed or interrupted write never leaves a
  // truncated image under the final name (rename within one directory replaces atomically on POSIX).
  std::filesystem::path tmp = path;
  tmp += ".tmp-" + std::to_string(std::random_device{}());
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    out.close();
    if (!out) {
      std::error_code ignored;
      std::filesystem::remove(tmp, ignored);
      throw std::runtime_error("cannot write " + util::path_utf8(path));
    }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);
    throw std::runtime_error("cannot write " + util::path_utf8(path) + ": " + ec.message());
  }
}

}  // namespace mi::pipeline
