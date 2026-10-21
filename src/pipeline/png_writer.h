// Minimal PNG encoder for the pipeline's RGB8 output: 8-bit truecolor, no interlace, filter 0, and a zlib stream of
// stored (uncompressed) deflate blocks split over IDAT chunks of at most 1 MiB, so no compression library is
// needed. Files are ~3 bytes per pixel.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace mi::pipeline {

// rgb: height * width * 3 bytes, row-major. Throws std::invalid_argument on a size mismatch or zero dimension.
std::vector<std::uint8_t> encode_png_rgb8(std::span<const std::uint8_t> rgb, int width, int height);

// Writes encode_png_rgb8() to a temporary file next to `path`, then renames it to `path` (replacing it), so `path`
// never holds a partial image. Throws std::runtime_error on I/O failure.
void write_png_rgb8(const std::filesystem::path& path, std::span<const std::uint8_t> rgb, int width, int height);

}  // namespace mi::pipeline
