// The Qwen3.5 text encoder as MOSS-Image uses it (src/models/qwen35_text/text_encoder.h): Qwen3_5Model(input_ids)
// .last_hidden_state for text-only input. 32 decoder layers, a GatedDeltaNet in three of every four and full gated
// attention in the fourth, each RMSNorm -> token mixer -> residual -> RMSNorm -> SiLU MLP -> residual, then the final
// norm. bf16 [S, 4096] per sequence.
//
// All sequences go through each layer together, packed along the rows: GEMMs and elementwise ops see every token at
// once (their outputs do not depend on the row count here), norms, the conv, the gated delta rule and attention run per
// sequence. Weights stream from the safetensors files: the embedding rows of the given ids are gathered on the host, a
// reader thread reads layer k + 2 into one of two host staging buffers while layer k runs, and layer k + 1 is copied
// from staging into one of two device buffers meanwhile, on the device's transfer queue where it has one
// (Device::copy_async). Work goes to the device in submits of a bounded size (TextEncoderOptions::submit_gflop), so no
// submit runs long enough to trip a display driver's watchdog.
#pragma once

#include <filesystem>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "block.h"
#include "runtime.h"

namespace mi::vk {

inline constexpr int kTeHidden = 4096;
inline constexpr int kTeIntermediate = 12288;
inline constexpr int kTeLayers = 32;
inline constexpr int kTeHeads = 16;
inline constexpr int kTeKvHeads = 4;
inline constexpr int kTeHeadDim = 256;
inline constexpr int kTeVocab = 248320;

struct TextEncoderOptions {
  // Estimated work per submit, in GFLOP; 0 picks the path's default (20 plain, 200 cooperative matrix). A GEMM larger
  // than this is split along its rows.
  double submit_gflop = 0;
  // Threads reading layer weights from the files.
  int reader_threads = 4;
  // Submit every op on its own and time it into TextEncoderStats::op_seconds (slower; for finding costs).
  bool profile = false;
};

// Times of the last encode(), in seconds of host wall time.
struct TextEncoderStats {
  std::uint32_t tokens = 0;
  double total_s = 0;
  double embed_s = 0;      // embedding rows read and uploaded
  double read_s = 0;       // the reader thread reading layer weights from the files
  double read_wait_s = 0;  // waiting for the reader
  double upload_s = 0;     // waiting for staging -> device copies (the part not hidden behind compute)
  double compute_s = 0;    // the layers' and the final norm's submits
  int submits = 0;
  std::uint64_t device_bytes = 0;            // device memory the encoder held at the peak of the call (its buffers)
  std::uint64_t staging_bytes = 0;           // host-visible staging it held
  std::map<std::string, double> op_seconds;  // TextEncoderOptions::profile only: seconds per kind of op
};

// Raw output of decoder layer `layer` (before the final norm), all sequences packed: bf16 [tokens, 4096] in `hidden`.
// Called after the layer's work has completed; the hook may download from the buffer.
using TeLayerHook = std::function<void(int layer, Buffer& hidden, std::uint32_t tokens)>;

class TextEncoder {
 public:
  // `dir`: the text_encoder directory, with model.safetensors.index.json and its shards. The default path is the
  // cooperative-matrix one where the device has bf16 fragments.
  TextEncoder(Device& device, const std::filesystem::path& dir);
  TextEncoder(Device& device, const std::filesystem::path& dir, BlockPath path, TextEncoderOptions options = {});
  ~TextEncoder();
  TextEncoder(const TextEncoder&) = delete;
  TextEncoder& operator=(const TextEncoder&) = delete;

  BlockPath path() const;

  // last_hidden_state of every sequence as bf16 bits [S_i * 4096]. Sequences are non-empty, ids below kTeVocab.
  std::vector<std::vector<std::uint16_t>> encode(const std::vector<std::vector<std::int32_t>>& sequences,
                                                 const TeLayerHook& hook = {});

  const TextEncoderStats& last_stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mi::vk
