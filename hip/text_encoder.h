// The Qwen3.5 text encoder as MOSS-Image uses it (src/models/qwen35_text/text_encoder.h): Qwen3_5Model(input_ids)
// .last_hidden_state for text-only input, on the HIP kernels (te_kernels.h, hip_block.h). 32 decoder layers, a
// GatedDeltaNet in three of every four and full gated attention in the fourth, each RMSNorm -> token mixer -> residual ->
// RMSNorm -> SiLU MLP -> residual, then the final norm. bf16 [S, 4096] per sequence, in the engine's op order and
// rounding points: the same bits as the engine's ops apart from the GEMMs and attention on the nvidia build, not by
// construction on AMD, whose exp, rsqrt and log1p are OCML's (te_kernels.h).
//
// All sequences of a call go through each layer together, packed along the rows: GEMMs and elementwise ops see every
// token at once (a GEMM's output element does not depend on the rows beside it), norms run per sequence (torch's reduce
// order depends on the row count), the conv, the gated delta rule and attention per sequence through the row offsets.
// The GEMM weights of one projection group are adjacent in the layer's buffer and run as one GEMM: GatedDeltaNet
// in_proj_qkv | in_proj_z | in_proj_b | in_proj_a, attention q_proj | k_proj | v_proj, MLP gate_proj | up_proj.
//
// Weights stream from the safetensors files one layer at a time; the device never holds more than two layers. A reader
// thread (with helper threads) reads layer k + 2 into one of two pinned host buffers while layer k + 1 is copied on a copy
// stream into a device buffer and layer k computes. The packed path (the default) then packs the layer's GEMM weights
// into pack_fragments' layout on the device; the copy of the next layer waits only for that pack. The embedding rows of
// the ids are read from the file on the host.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "runtime.h"

namespace mi::hip {

inline constexpr int kTeHidden = 4096;
inline constexpr int kTeIntermediate = 12288;
inline constexpr int kTeLayers = 32;
inline constexpr int kTeHeads = 16;
inline constexpr int kTeKvHeads = 4;
inline constexpr int kTeHeadDim = 256;
inline constexpr std::uint32_t kTeMaxTokens = 65535;  // tokens of one call, all sequences together

struct TextEncoderOptions {
  // Threads reading a layer's weights from the files (several reads in flight are what a cold NVMe drive needs).
  int reader_threads = 4;
  // GEMMs on packed operands (pack_fragments' layout); false: on the row-major operands. Both give the same bits.
  bool packed = true;
  // Device time of every op into TextEncoderStats::op_ms (events around each op).
  bool time_ops = false;
};

// Times of the last encode().
struct TextEncoderStats {
  std::uint32_t tokens = 0;
  double total_s = 0;        // host wall time of the call
  double embed_s = 0;        // embedding rows read and uploaded
  double read_s = 0;         // the reader threads reading layer weights from the files (wall time of the reads)
  double read_wait_s = 0;    // the host waiting for the reader
  double layers_ms = 0;       // device time from the first layer's start to the final norm's end
  double weight_wait_ms = 0;  // of which the gaps between layers: the compute stream waiting for weights
  std::uint64_t device_bytes = 0;  // device memory the call held: weight buffers and activations
  std::uint64_t pinned_bytes = 0;  // pinned host staging
  std::map<std::string, double> op_ms;  // TextEncoderOptions::time_ops: device milliseconds per kind of op
};

// Raw output of decoder layer `layer` (before the final norm), all sequences packed: bf16 [tokens, 4096] at `hidden` on
// the device, valid during the call. Issued after the layer's work on `stream`: the hook synchronizes it before reading.
using TeLayerHook = std::function<void(int layer, const std::uint16_t* hidden, std::uint32_t tokens, hipStream_t stream)>;

class TextEncoder {
 public:
  // `dir`: the model's text_encoder directory (model.safetensors.index.json and its shards).
  TextEncoder(Device& device, const std::filesystem::path& dir, TextEncoderOptions options = {});
  ~TextEncoder();
  TextEncoder(const TextEncoder&) = delete;
  TextEncoder& operator=(const TextEncoder&) = delete;

  // last_hidden_state of every sequence as bf16 bits [S_i * 4096]. Sequences are non-empty with ids below vocab(), at
  // most kTeMaxTokens tokens together.
  std::vector<std::vector<std::uint16_t>> encode(const std::vector<std::vector<std::int32_t>>& sequences,
                                                 const TeLayerHook& hook = {});

  std::int64_t vocab() const;
  const TextEncoderStats& last_stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mi::hip
