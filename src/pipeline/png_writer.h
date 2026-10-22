// Minimal PNG encoder for the pipeline's RGB8 output: 8-bit truecolor, no interlace, filter 0, and a zlib stream of
// stored (uncompressed) deflate blocks split over IDAT chunks of at most 1 MiB, so no compression library is
// needed. Files are ~3 bytes per pixel. Optional text chunks carry the generation settings.
//
// Generation metadata (optional, two text chunks placed right after IHDR):
//  - keyword "generation" (iTXt, UTF-8, uncompressed): one JSON object, schema "mossinfer.generation/1":
//      {"schema": "mossinfer.generation/1",
//       "engine": {"name": "moss-infer", "version": "...", "tier": "..."},
//       "models": [{"role": "dit"|"text_encoder"|"vae"|..., "name": "...", "sha256": "<64 hex>"}, ...],
//       "loras": [{"name": "...", "sha256": "...", "strength": 0.6}, ...],
//       "prompt": "...", "negative_prompt": "...",
//       "sampler": "euler", "scheduler": "moss", "steps": 50, "cfg": 3, "shift": 5, "seed": 42,
//       "width": 1024, "height": 1024,
//       "extra": {"key": "value", ...}}
//    "prompt" and "negative_prompt" are always present (possibly empty) unless include_prompts is false, then both
//    are left out. Empty engine version / tier / sampler / scheduler, empty hashes and an empty `extra` are omitted.
//    Numbers are shortest round-trip decimals; seed is an integer (readers that parse JSON numbers as doubles keep
//    it exact up to 2^53).
//  - keyword "parameters" (tEXt when ASCII, else iTXt): the plain-text settings block that many image tools and
//    galleries read:
//      <prompt>
//      Negative prompt: <negative prompt>
//      Steps: 50, Sampler: Euler, Schedule type: MOSS, CFG scale: 3, Shift: 5, Seed: 42, Size: 1024x1024,
//      Model hash: <first 10 hex of sha256>, Model: <name>, Lora hashes: "<name>: <hash10>, ...", <extra key>: <value>,
//      Version: <...>
//    The negative prompt line is left out when it is empty, both prompt lines when include_prompts is false. Keys and
//    values containing ',', ':' or a newline are written as JSON strings.
// Privacy and robustness: model and LoRA names are reduced to their base names (no directories), and every string is
// made valid UTF-8 without NUL (each ill-formed byte and NUL becomes U+FFFD), so metadata never makes the write fail.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace mi::pipeline {

// One PNG text chunk. keyword: 1-79 printable Latin-1 characters (no leading/trailing/double spaces); text: UTF-8.
struct PngText {
  std::string keyword;
  std::string text;
};

// rgb: height * width * 3 bytes, row-major. Text chunks are written after IHDR: tEXt when the text is ASCII, iTXt
// (uncompressed UTF-8) otherwise. Throws std::invalid_argument on a size mismatch, zero dimension, an invalid keyword,
// or text that is not valid UTF-8 or contains NUL (generation_text_chunks() never produces such text).
std::vector<std::uint8_t> encode_png_rgb8(std::span<const std::uint8_t> rgb, int width, int height,
                                          std::span<const PngText> text = {});

// Writes encode_png_rgb8() to a temporary file next to `path`, then renames it to `path` (replacing it), so `path`
// never holds a partial image. Throws std::runtime_error on I/O failure.
void write_png_rgb8(const std::filesystem::path& path, std::span<const std::uint8_t> rgb, int width, int height,
                    std::span<const PngText> text = {});

// The tEXt (Latin-1, returned as UTF-8) and uncompressed iTXt chunks of a PNG, in file order; compressed chunks are
// skipped. Throws std::invalid_argument on a malformed file or a CRC mismatch.
std::vector<PngText> read_png_text(std::span<const std::uint8_t> png);

struct GenerationMetadata {
  struct ModelFile {
    std::string role;    // dit, text_encoder, vae, bundle, ...
    std::string name;    // file or directory name (written as its base name)
    std::string sha256;  // lowercase hex, empty if unknown
  };
  struct Lora {
    std::string name;    // written as its base name
    std::string sha256;
    double strength = 1.0;
  };
  std::string engine_version;
  std::string tier;  // quality tier (e.g. reference, high, turbo)
  std::vector<ModelFile> models;
  std::vector<Lora> loras;
  std::string prompt;
  std::string negative_prompt;
  bool include_prompts = true;  // false: the prompt and negative prompt are not written
  std::string sampler;
  std::string scheduler;
  int steps = 0;
  double cfg = 0;
  double shift = 0;
  std::uint64_t seed = 0;
  int width = 0;
  int height = 0;
  std::vector<std::pair<std::string, std::string>> extra;  // further string settings (e.g. noise source)
};

inline constexpr const char* kGenerationKeyword = "generation";
inline constexpr const char* kGenerationSchema = "mossinfer.generation/1";
inline constexpr const char* kParametersKeyword = "parameters";

// The JSON document of the "generation" chunk.
std::string generation_json(const GenerationMetadata& meta);
// The text of the "parameters" chunk.
std::string generation_parameters_text(const GenerationMetadata& meta);
// Both chunks, in that order (either can be left out).
std::vector<PngText> generation_text_chunks(const GenerationMetadata& meta, bool json = true, bool parameters = true);

}  // namespace mi::pipeline
