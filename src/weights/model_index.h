// Model index: maps a HF-layout bundle (<bundle>/{text_encoder,transformer,vae}/) onto the weight
// groups the engine schedules (decoder layers, DiT blocks, ...), recording every tensor's file
// and byte range so weights are read in place from the original safetensors files.
//
// On-disk `.mossidx` format (JSON, version 1; a cache keyed to the files it was built from):
//   {
//     "format": "mossidx", "version": 1,
//     "files":    [ {"path": "<bundle-relative, '/', UTF-8>", "size": <bytes>, "mtime_ns": <int>} ... ],
//     "excluded": [ {"component": "text_encoder", "prefix": "visual.", "tensors": N, "bytes": B} ... ],
//     "groups":   [ { "component": "transformer", "name": "block.0",
//                     "layer_type": "full_attention",          // text_encoder layers only
//                     "bytes": B, "params": P,
//                     "xxh64": "<16 hex digits>",              // only when built with --checksum
//                     "tensors": [ ["<name>", "<DTYPE>", [shape...], <file id>, <abs offset>, <bytes>] ... ] } ... ]
//   }
// `file id` indexes "files". Files + sizes + mtimes (ns since the Unix epoch) detect staleness. xxh64 covers the group's
// tensor bytes concatenated in the listed order (sorted by file, then offset).
#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "util/path.h"
#include "weights/safetensors.h"

namespace mi::weights {

inline constexpr int kIndexVersion = 1;

// Bounds on the JSON files a downloaded bundle supplies, checked on the file size before reading and
// on the DOM's value count while parsing. Real files are far below them (MOSS-Image: configs <= 3 KB
// and <= 170 values, shard indexes <= 65 KB and <= 1.5k values, .mossidx 101 KB and ~10k values).
inline constexpr std::uintmax_t kMaxConfigBytes = std::uintmax_t{4} << 20;
inline constexpr std::size_t kMaxConfigValues = 100'000;
inline constexpr std::uintmax_t kMaxShardIndexBytes = std::uintmax_t{4} << 20;
inline constexpr std::size_t kMaxShardIndexValues = 1'000'000;
inline constexpr std::uintmax_t kMaxMossIdxBytes = std::uintmax_t{32} << 20;
inline constexpr std::size_t kMaxMossIdxValues = 2'000'000;

struct IndexError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
// The index no longer matches the files on disk; rebuild with `moss-infer index`.
struct StaleIndexError : IndexError {
  using IndexError::IndexError;
};

enum class Component { kTextEncoder, kTransformer, kVae };
const char* component_name(Component c);

struct IndexedTensor {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<std::int64_t> shape;
  std::uint32_t file = 0;  // index into ModelIndex::files
  std::uint64_t offset = 0;
  std::uint64_t nbytes = 0;

  std::int64_t numel() const;
};

struct WeightGroup {
  Component component = Component::kTextEncoder;
  // text_encoder: "embed_tokens", "layers.N", "final_norm"
  // transformer:  "globals", "text_embedder", "block.N"
  // vae:          "decoder"
  std::string name;
  std::string layer_type;  // "full_attention" / "linear_attention" for text_encoder layers, else empty
  std::vector<IndexedTensor> tensors;
  std::uint64_t total_bytes = 0;
  std::int64_t total_params = 0;
  std::optional<std::uint64_t> checksum;  // XXH64
};

// Tensors deliberately left out of every group (vision tower, lm_head, VAE encoder, ...).
struct ExcludedSummary {
  Component component = Component::kTextEncoder;
  std::string prefix;
  std::uint64_t tensors = 0;
  std::uint64_t bytes = 0;
};

struct IndexedFile {
  std::string path;  // relative to the bundle dir, '/' separated, UTF-8
  std::uint64_t size = 0;
  std::int64_t mtime_ns = 0;
};

struct ModelIndex {
  std::filesystem::path bundle_dir;  // runtime only, not serialized
  std::vector<IndexedFile> files;
  std::vector<WeightGroup> groups;
  std::vector<ExcludedSummary> excluded;

  const WeightGroup* find_group(Component c, std::string_view name) const;
  std::vector<const WeightGroup*> groups_of(Component c) const;
  std::filesystem::path file_path(std::uint32_t id) const { return bundle_dir / util::path_from_utf8(files.at(id).path); }
};

// Where a tensor of a component goes: into a group, or into a named exclusion. Exactly one of the two is set.
struct TensorRoute {
  std::string group;       // e.g. "block.3", "layers.0", "embed_tokens"
  std::string exclude;     // exclusion prefix, e.g. "visual."
  std::string layer_kind;  // text_encoder layers: attention kind implied by this tensor's name
};

// The routing build_index applies to the official names (text encoder: optionally `model.`-prefixed). Throws
// IndexError for a name no group or exclusion covers.
TensorRoute classify_tensor(Component c, const std::string& name);

// Parses "<digits>." at the front of `rest` into n and advances past it. Digits are canonical (no sign, no leading
// zero) so two spellings cannot name the same layer.
bool take_layer_index(std::string_view& rest, int& n);

struct BuildOptions {
  bool checksum = false;  // read every group's bytes and record XXH64 (full I/O on the group files)
};

// Reads safetensors headers and configs only (plus group data when options.checksum).
// Throws FormatError / IoError / IndexError on malformed or unreadable bundles or unclassified tensors.
ModelIndex build_index(const std::filesystem::path& bundle_dir, const BuildOptions& options = {});

void write_index(const ModelIndex& index, const std::filesystem::path& path);

// Loads and validates a .mossidx: version, internal consistency, staleness against the files in
// `bundle_dir` (default: the index file's directory), and agreement with build_index(bundle_dir)
// (configs and safetensors headers are re-read; the index is untrusted input). The result keeps the
// stored checksums. Throws IndexError, or StaleIndexError when a file changed size or mtime.
ModelIndex load_index(const std::filesystem::path& path, const std::filesystem::path& bundle_dir = {});

// XXH64 over the group's tensor bytes in listed order.
std::uint64_t checksum_group(const ModelIndex& index, const WeightGroup& group);

void print_summary(const ModelIndex& index, std::ostream& out);

}  // namespace mi::weights
