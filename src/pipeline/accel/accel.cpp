#include "pipeline/accel/accel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "pipeline/accel/accel_kernels.h"

namespace mi::pipeline::accel {
namespace ak = accel_kernels;
namespace {

std::string number(double d) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", d);
  return buf;
}

std::size_t video_bytes(std::int64_t video) {
  return static_cast<std::size_t>(video) * static_cast<std::size_t>(moss_dit::kDim) * sizeof(std::uint16_t);
}

}  // namespace

const char* step_cache_name(StepCacheKind kind) {
  switch (kind) {
    case StepCacheKind::kOff: return "off";
    case StepCacheKind::kFbCache: return "fbcache";
  }
  return "?";
}

StepCacheKind parse_step_cache(std::string_view name) {
  if (name == "off") return StepCacheKind::kOff;
  if (name == "fbcache") return StepCacheKind::kFbCache;
  if (name == "teacache") {
    throw std::invalid_argument("step cache 'teacache' is not offered: its skip test needs a polynomial fitted to the "
                                "model, and none exists for MOSS-Image; use 'fbcache' (calibration-free)");
  }
  throw std::invalid_argument("unknown step cache '" + std::string(name) + "' (expected off or fbcache)");
}

std::string Options::describe(bool do_cfg) const {
  std::string out;
  if (do_cfg && cfg_truncation < 1.0) out = "cfg_truncation " + number(cfg_truncation);
  if (step_cache_on()) {
    out += (out.empty() ? "" : ", ") + std::string("step_cache ") + step_cache_name(step_cache) + " " +
           (step_cache_threshold ? number(*step_cache_threshold) : std::string("default"));
  }
  return out.empty() ? "off" : out;
}

std::string threshold_source(const Options& requested, std::string_view lineage) {
  if (requested.step_cache_threshold) return "request";
  return "default for the " + (lineage.empty() ? std::string("unrecognized") : std::string(lineage)) + " model";
}

std::string base_model_threshold_warning(const Options& requested, std::string_view lineage) {
  if (!requested.step_cache_on() || lineage != "base" || !requested.step_cache_threshold ||
      *requested.step_cache_threshold <= kBaseModelFbCacheThreshold) {
    return {};
  }
  return "step cache: fbcache " + number(*requested.step_cache_threshold) +
         " on the base model can change the composition (RTX 5090, 4 prompts: min DINOv2 0.77 at 0.03 and 0.53 at 0.05 "
         "against the uncached image; the SFT model stays at 0.93 or above): use 0.02 or below, or the SFT model, to keep "
         "the uncached image's layout";
}

std::string never_engaged_warning(const Stats& stats) {
  if (stats.step_cache_threshold <= 0 || stats.cached_calls != 0) return {};
  std::vector<double> d;
  for (const double x : stats.step_cache_diffs) {
    if (std::isfinite(x)) d.push_back(x);  // calls with nothing to compare with (NaN, null in the report) do not count
  }
  if (d.size() < 4) return {};
  std::sort(d.begin(), d.end());
  return "step cache: fbcache " + number(stats.step_cache_threshold) + " never engaged: block 0's relative change was at least " +
         number(d.front()) + " (median " + number(d[d.size() / 2]) + ") over " + std::to_string(d.size()) +
         " comparable DiT calls, so no call reused the cache";
}

double default_fbcache_threshold(std::string_view lineage) {
  return lineage == "base" ? kBaseModelFbCacheThreshold : kDefaultFbCacheThreshold;
}

void validate(const Options& o) {
  if (!std::isfinite(o.cfg_truncation) || o.cfg_truncation < 0.0 || o.cfg_truncation > 1.0) {
    throw std::invalid_argument("cfg_truncation must be between 0 and 1 (1 = off).");
  }
  if (o.step_cache_on() && o.step_cache_threshold &&
      (!std::isfinite(*o.step_cache_threshold) || *o.step_cache_threshold <= 0.0 || *o.step_cache_threshold > 1.0)) {
    throw std::invalid_argument("step_cache_threshold must be greater than 0 and at most 1.");
  }
}

std::size_t FirstBlockCache::workspace_bytes(std::int64_t height, std::int64_t width, int rows) {
  const std::size_t buffer = runtime::align_up(video_bytes(height * width));
  const auto r = static_cast<std::size_t>(std::max(rows, 0));
  return 3 * r * buffer + runtime::align_up(r * ak::kDiffBlocks * 2 * sizeof(double));
}

FirstBlockCache::FirstBlockCache(runtime::DeviceArena& arena, std::int64_t height, std::int64_t width, int rows,
                                 double threshold)
    : video_(height * width), threshold_(threshold) {
  if (rows < 1 || rows > moss_dit::kMaxRows) throw std::invalid_argument("FirstBlockCache: rows out of range");
  if (height < 1 || width < 1) throw std::invalid_argument("FirstBlockCache: bad latent grid");
  if (!std::isfinite(threshold) || threshold <= 0.0 || threshold > 1.0) {
    throw std::invalid_argument("FirstBlockCache: threshold must lie in (0, 1]");
  }
  slots_.resize(static_cast<std::size_t>(rows));
  for (Slot& s : slots_) {
    s.scratch = arena.alloc_stack(video_bytes(video_));
    s.first = arena.alloc_stack(video_bytes(video_));
    s.rest = arena.alloc_stack(video_bytes(video_));
  }
  partials_ = static_cast<double*>(arena.alloc_stack(slots_.size() * ak::kDiffBlocks * 2 * sizeof(double)));
  host_partials_.resize(slots_.size() * ak::kDiffBlocks * 2);
}

FirstBlockCache::Slot& FirstBlockCache::slot_of(const moss_dit::Row& row) {
  const auto taken = [&](const Slot& s) { return std::find(current_.begin(), current_.end(), &s) != current_.end(); };
  for (Slot& s : slots_) {
    if (s.used && s.text == row.text.data && s.text_length == row.text.dim(0) && s.skip == row.skip &&
        s.latent == row.latent && !taken(s)) {
      return s;
    }
  }
  // A new row identity: a free slot, else one no row of this call uses (its residuals are dropped).
  Slot* pick = nullptr;
  for (Slot& s : slots_) {
    if (!s.used) {
      pick = &s;
      break;
    }
  }
  for (Slot& s : slots_) {
    if (pick == nullptr && !taken(s)) pick = &s;
  }
  if (pick == nullptr) throw std::logic_error("FirstBlockCache: more rows than slots");
  pick->text = row.text.data;
  pick->text_length = row.text.dim(0);
  pick->skip = row.skip;
  pick->latent = row.latent;
  pick->used = true;
  pick->valid = false;
  return *pick;
}

void FirstBlockCache::invalidate() {
  for (Slot& s : slots_) s.valid = false;
}

void FirstBlockCache::invalidate(int latent) {
  for (Slot& s : slots_) {
    if (s.latent == latent) s.valid = false;
  }
}

void FirstBlockCache::check(const Tensor& x, std::span<const moss_dit::Row> rows, std::span<const std::int64_t> offsets,
                            std::int64_t video) const {
  if (video != video_) throw std::invalid_argument("FirstBlockCache: the latent grid changed (one cache per pass)");
  if (rows.size() > slots_.size() || offsets.size() != rows.size() + 1 || x.dtype != DType::kBF16 ||
      x.device != Device::kCUDA || x.rank != 2 || x.dim(1) != moss_dit::kDim) {
    throw std::invalid_argument("FirstBlockCache: unexpected rows or residual stream " + x.describe());
  }
}

void* FirstBlockCache::row_video(const Tensor& x, std::span<const std::int64_t> offsets, std::size_t r) {
  return x.slice0(offsets[r], offsets[r] + 1).data;
}

void FirstBlockCache::before_blocks(const Tensor& x, std::span<const moss_dit::Row> rows,
                                    std::span<const std::int64_t> offsets, std::int64_t video, cudaStream_t stream) {
  check(x, rows, offsets, video);
  current_.clear();
  for (const moss_dit::Row& row : rows) current_.push_back(&slot_of(row));
  // A row identity absent from this call (e.g. the SLG row outside its window) loses its residuals: when it comes back,
  // its first call runs in full instead of reusing an R older than the other rows'. With per-latent forwards (a
  // component run's latent batch on a resident DiT, Session::step) a forward over one latent judges only that latent's
  // identities; a forward over several latents, or any forward with set_per_latent(false) (a streamed DiT's batched
  // calls, which decide jointly), judges them all.
  const bool several =
      std::any_of(rows.begin(), rows.end(), [&](const moss_dit::Row& row) { return row.latent != rows[0].latent; });
  for (Slot& s : slots_) {
    const bool judged = !per_latent_ || several || (!rows.empty() && s.latent == rows[0].latent);
    if (judged && std::find(current_.begin(), current_.end(), &s) == current_.end()) s.valid = false;
  }
  for (std::size_t r = 0; r < rows.size(); ++r) {
    MI_CUDA_CHECK(cudaMemcpyAsync(current_[r]->scratch, row_video(x, offsets, r), video_bytes(video_),
                                  cudaMemcpyDeviceToDevice, stream));
  }
}

bool FirstBlockCache::after_first_block(const Tensor& x, std::span<const moss_dit::Row> rows,
                                        std::span<const std::int64_t> offsets, std::int64_t video, cudaStream_t stream) {
  check(x, rows, offsets, video);
  if (current_.size() != rows.size()) throw std::logic_error("FirstBlockCache: after_first_block without before_blocks");
  ++calls_;
  const std::int64_t n = video_ * moss_dit::kDim;
  const bool comparable = std::all_of(current_.begin(), current_.end(), [](const Slot* s) { return s->valid; });
  double diff = std::numeric_limits<double>::quiet_NaN();
  if (comparable) {
    for (std::size_t r = 0; r < rows.size(); ++r) {
      ak::residual_diff_partials(row_video(x, offsets, r), current_[r]->scratch, current_[r]->first, n,
                                 partials_ + r * ak::kDiffBlocks * 2, stream);
    }
    const std::size_t count = rows.size() * ak::kDiffBlocks * 2;
    MI_CUDA_CHECK(cudaMemcpyAsync(host_partials_.data(), partials_, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    double num = 0, den = 0;
    for (std::size_t i = 0; i < count; i += 2) {
      num += host_partials_[i];
      den += host_partials_[i + 1];
    }
    if (den > 0) diff = num / den;
  }
  diffs_.push_back(diff);
  const bool reuse = comparable && diff < threshold_;  // false for NaN
  if (reuse) {
    ++reused_;
    return true;
  }
  // A full call: r0 becomes the reference, x1 is kept for R. Invalid until after_blocks stores R (a cancelled forward
  // leaves the slot unusable rather than inconsistent).
  for (std::size_t r = 0; r < rows.size(); ++r) {
    Slot& s = *current_[r];
    s.valid = false;
    const void* x1 = row_video(x, offsets, r);
    ak::sub_bf16(x1, s.scratch, s.first, n, stream);
    MI_CUDA_CHECK(cudaMemcpyAsync(s.scratch, x1, video_bytes(video_), cudaMemcpyDeviceToDevice, stream));
  }
  return false;
}

void FirstBlockCache::after_blocks(const Tensor& x, std::span<const moss_dit::Row> rows,
                                   std::span<const std::int64_t> offsets, std::int64_t video, bool reused,
                                   cudaStream_t stream) {
  check(x, rows, offsets, video);
  if (current_.size() != rows.size()) throw std::logic_error("FirstBlockCache: after_blocks without before_blocks");
  const std::int64_t n = video_ * moss_dit::kDim;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    Slot& s = *current_[r];
    void* xv = row_video(x, offsets, r);
    if (reused) {
      ak::add_bf16(xv, s.rest, n, stream);
    } else {
      ak::sub_bf16(xv, s.scratch, s.rest, n, stream);
      s.valid = true;
    }
  }
  current_.clear();
}

}  // namespace mi::pipeline::accel
