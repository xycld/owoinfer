// The machine profile behind the planner's cost model (P4b) and the fused DiT block self-check (P5): measured once,
// cached on disk with the embedding cache as <cache_dir>/hardware_profile.json, and refined from measured generations.
//
// Persistence: the file is keyed by hardware_key() (GPU, compute capability, PCI location, driver and cuBLASLt
// versions) and probed again when the key changes or the probe is older than kProfileMaxAgeSeconds. It is replaced
// atomically (a unique temporary file, then a rename), its numbers are written locale-independently (a non-finite one
// as null, which reads as missing: a learned rate falls back to the probed one), and anything unreadable, malformed or
// implausible reads as "no profile" (measure again). The fused self-check is stored with the fused kernels' version
// (kernels::fused::kFusedKernelsVersion) and the configurations it validated; one from other kernels, or malformed,
// reads as "not checked" (check again) and leaves the rates.
//
// Learning (ProfileLearner): one disturbed generation (a busy host, a page cache another program just evicted) must not
// steer the planner. A boolean (are the weight files in the page cache?) flips only after kEvidence consecutive
// generations agree, learns from every source that read the files (streamed denoise, DiT loads, text-encoder streams),
// and a "cold" verdict with no fresh evidence is re-checked after kRecheckGenerations. Rates are the median of the
// recent estimates, bounded around the probed value, and are written to disk only once kEvidence estimates agree within
// kSpread.
#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <optional>
#include <string>

#include "models/moss_dit/dit_block_fused.h"
#include "planner/planner.h"

namespace mi::pipeline {

// GPU name, compute capability, PCI location, driver and cuBLASLt versions of the current device.
std::string hardware_key();

struct StoredProfile {
  planner::HardwareProfile probe;    // as the probe measured it (bf16_tflops: the planner's default)
  planner::HardwareProfile learned;  // refined from consistent generations: where the planner starts
  std::int64_t probed_at = 0;        // unix seconds of the probe
  // The startup self-check, once it ran on this hardware key with these fused kernels.
  std::optional<moss_dit::FusedCheck> fused_check;
};

inline constexpr std::int64_t kProfileMaxAgeSeconds = std::int64_t{30} * 24 * 3600;

// nullopt when the file is missing, unreadable, malformed, of another key or version, holds implausible rates, or was
// probed more than kProfileMaxAgeSeconds before `now` (or after it).
std::optional<StoredProfile> read_profile(const std::filesystem::path& path, const std::string& key, std::int64_t now);
// false when it could not be written (the cache is an optimization: a read-only directory means measuring again).
bool write_profile(const std::filesystem::path& path, const std::string& key, const StoredProfile& profile);

class ProfileLearner {
 public:
  static constexpr int kEvidence = 3;             // agreeing generations to flip a verdict or persist a rate
  static constexpr double kSpread = 0.15;         // estimates agree when within +-15% of their median
  static constexpr int kRecheckGenerations = 8;   // a "cold" verdict without new evidence is re-checked after these
  static constexpr int kHistory = 5;              // rate estimates kept for the median
  // Learned rates stay within these factors of the probed ones.
  static constexpr double kRateMin = 0.5, kRateMax = 2.0;
  static constexpr double kTflopsMin = 0.25, kTflopsMax = 4.0;

  explicit ProfileLearner(StoredProfile profile);

  const planner::HardwareProfile& current() const { return current_; }  // what the next plan uses
  StoredProfile& stored() { return stored_; }                            // what is persisted
  bool dit_weights_cached() const { return dit_cached_.value; }
  bool te_weights_cached() const { return te_cached_.value; }

  // One denoise's compute-rate estimate (TFLOPS). True when stored() changed (persist it).
  bool observe_tflops(double tflops);
  // One file-streamed denoise's median page-cache -> device rate (GB/s): a cached/cold vote, and when cached a rate
  // estimate. True when stored() changed.
  bool observe_dit_stream(double gbps);
  // A DiT load from the weight files (one reader, buffered): a cached/cold vote.
  void observe_dit_load(double gbps);
  // One file-streamed text-encoder pass (planner::weights_cached_from_stream): a vote.
  void observe_te_stream(bool cached);
  // After every generation: counts generations without evidence for the re-check.
  void end_generation();

 private:
  struct Verdict {
    bool value = true;
    std::deque<bool> votes;  // the last kEvidence votes
    int idle = 0;            // generations since the last vote
    void vote(bool v);
    void end_generation();
  };
  struct Series {
    std::deque<double> values;  // the last kHistory estimates
    void push(double v);
    double median() const;
    bool consistent() const;  // the last kEvidence estimates agree within kSpread
  };
  // Whether a weight-file read at `gbps` came from the page cache: above the geometric mean of the probed page-cache
  // and cold-disk rates.
  bool cached_rate(double gbps) const;

  StoredProfile stored_;
  planner::HardwareProfile current_;
  Verdict dit_cached_, te_cached_;
  Series tflops_, host_gbps_;
};

}  // namespace mi::pipeline
