// GGUF DiT files (PipelineOptions::dit_file, mi_inspect_file): the engine reads a community MOSS-Image GGUF DiT by
// converting it once into one of its own formats and loading that file like any other; there is no GGUF runtime path.
//   - tier high: every supported type (Q8_0, Q6_K, Q5_K, Q5_1, Q5_0, Q4_K, Q4_1, Q4_0, BF16, F16, F32) -> INT8 ConvRot
//     (G = 256);
//   - tier reference: a file whose every tensor is BF16, F16 or F32 -> bf16 (a Q8_0 file is not: rounded to bf16 its
//     weights would be rounded twice, and INT8 runs its own grid more closely);
//   - tier turbo: refused (no 4-bit target), as is any other type (Q2_K, Q3_K, IQ*, custom types).
// So VRAM and speed are the target format's. The converted file is kept in <cache_dir>/gguf/, found again by the GGUF
// file's identity (size, modification time, volume and 128-bit file id, XXH64 of its header, the target and the
// converter version): only the first load converts (quant::convert, streamed one tensor at a time, under 0.6 GB of host
// RAM for a MOSS-Image DiT); later loads read the converted file directly. A GGUF DiT therefore needs a cache directory.
//
// The cache folder is shared by processes. Each converted file has a lock file "<file>.lock" (util::FileLock): a load
// holds it shared for as long as it uses the file, the conversion that writes it holds it exclusive (a second process
// loading the same GGUF waits, then finds the file), and an entry is removed only under its exclusive lock (never one in
// use). A call that converted sweeps the folder (a cache hit never does: it probes no recorded source path): converted
// files whose recorded source is gone from a folder that still exists, or was replaced by a newer version (same path,
// same target), temporaries of conversions that ended without publishing, and lock files of entries that are gone. The folder's contents are trusted as the model: only the engine's account may write it.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "util/regular_file.h"

#include "planner/planner.h"
#include "weights/quant/quant_model.h"

namespace mi::pipeline {

struct GgufDitInfo {
  // The GGUF types of every stored tensor, the most bytes first, joined by " + " (e.g. "Q4_K + Q6_K + BF16 + F32",
  // "Q8_0 + BF16 + F32", "BF16 + F32").
  std::string types;
  // Every stored tensor is BF16, F16 or F32: tier reference runs the file as bf16.
  bool reference = false;
  int tensors = 0;  // in the GGUF header
};

// The file's header checked as the converter will read it (weights::check_supported_gguf_dit, quant::open_gguf_dit):
// names, shapes and types, no tensor byte. Throws weights::UnsupportedModelError, quant::QuantError,
// weights::FormatError or weights::IoError.
GgufDitInfo inspect_gguf_dit(const std::filesystem::path& file);

// The format the file runs as at `tier`: INT8 ConvRot 256 (high), bf16 (reference, when info.reference); otherwise
// throws std::invalid_argument naming the tiers that run it.
quant::QuantSpec gguf_target(const GgufDitInfo& info, planner::Tier tier);

// "GGUF Q4_K, runs as INT8 ConvRot" / "GGUF Q8_0, runs as bf16".
std::string gguf_runs_as(const GgufDitInfo& info, const quant::QuantSpec& target);

// The running call's hold on a conversion (gguf_converted_file): mi_cancel / mi_engine_free and the progress callback.
struct GgufConvertHooks {
  // Set: the conversion, or the wait for another process's, stops with SessionCancelled (the temporary removed).
  const std::atomic<bool>* cancel = nullptr;
  // Per converted module or tensor, the share done (0..1); every second with 0 while another process converts the
  // file. Returning false cancels.
  std::function<bool(double)> progress;
};

struct GgufCachedFile {
  std::filesystem::path path;
  util::FileLock lock;     // "<path>.lock" held shared: keep it while the file is in use (no sweep removes the file)
  bool converted = false;  // this call converted it
};

// The converted file of `file` for `target` in <cache_dir>/gguf/, converted now when the cache has none for this
// identity: into a temporary file created exclusively there first, converted, read back, checked against a GGUF that
// changed meanwhile, then published under its name (a file that another process published meanwhile with the same key
// is used). Throws std::invalid_argument for an empty cache_dir; weights::IoError when the cache directory cannot hold
// or take the file (it cannot be created, written or locked, too little free space, the GGUF changed while it was
// converted); SessionCancelled; the converter's errors about the GGUF.
GgufCachedFile gguf_converted_file(const std::filesystem::path& file, const GgufDitInfo& info,
                                   const quant::QuantSpec& target, const std::filesystem::path& cache_dir,
                                   const GgufConvertHooks& hooks = {});

// A plan-only pipeline (PipelineOptions::plan_only, mi_check) never converts: gguf_cached_file finds the converted
// file read-only (no folder, lock or sweep), and without one the open throws GgufNotConverted, with what the first
// load's conversion would write and an estimate of its time.
struct GgufNotConverted : std::runtime_error {
  GgufNotConverted(const std::string& message, std::string format, std::uint64_t bytes, double estimate_s)
      : std::runtime_error(message), runs_as(std::move(format)), converted_bytes(bytes), seconds(estimate_s) {}
  std::string runs_as;                // gguf_runs_as
  std::uint64_t converted_bytes = 0;  // the converted file's size (the cache directory needs this much free)
  double seconds = 0;                 // estimated conversion time on this machine (gguf_conversion_seconds)
};

// The converted file of `file` for `target` that gguf_converted_file would use, when the cache already holds it (a
// valid entry for the GGUF's current identity); nullopt otherwise. Reads only: no folder is created, no lock taken.
// Throws std::invalid_argument for an empty cache_dir (as gguf_converted_file).
std::optional<std::filesystem::path> gguf_cached_file(const std::filesystem::path& file, const quant::QuantSpec& target,
                                                      const std::filesystem::path& cache_dir);

// Bytes of the converted file of `file` for `target` (the GEMMs' weights and scales, the other tensors), as
// gguf_converted_file's free-space check counts them (without its margin).
std::uint64_t gguf_converted_bytes(const std::filesystem::path& file, const quant::QuantSpec& target);

// A rough conversion time for `converted_bytes` of output on this machine: the converter's throughput measured on a
// 32-thread laptop CPU (about 32 s for the 5.4 GB INT8 file), scaled by the hardware threads.
double gguf_conversion_seconds(std::uint64_t converted_bytes);

}  // namespace mi::pipeline
