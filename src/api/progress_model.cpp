#include "api/progress_model.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mi::api {

namespace {

using planner::DitPlan;
using planner::DitResidency;

double bytes_per_s(double gbps) { return std::max(1e-3, gbps) * 1e9; }

bool same_residency(const DitPlan& a, const DitPlan& b) {
  if (a.residency != b.residency) return false;
  if (a.residency != DitResidency::kStreamed) return true;
  return a.resident_blocks == b.resident_blocks && a.buffers == b.buffers && a.source == b.source;
}

// Weight files into host memory, GB/s. Cold: the measured direct-I/O load rate (a cold DiT load reads direct), else
// the cold-disk rate. Warm (files this engine read before): the learned warm load mode's rate, else the page cache's.
double file_gbps(const planner::HardwareProfile& hw, bool warm) {
  if (warm) {
    if (hw.warm_direct && hw.direct_gbps > 0) return hw.direct_gbps;
    if (hw.buffered_warm_gbps > 0) return hw.buffered_warm_gbps;
    return hw.host_gbps;
  }
  return hw.direct_gbps > 0 ? hw.direct_gbps : hw.disk_gbps;
}

}  // namespace

LoadEstimates load_estimates(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw, const HeldModels& held,
                             const ModelBytes& bytes, const JobShape& shape) {
  LoadEstimates e;
  // A larger arena than the held one is created anew, without the models it held.
  const bool arena_kept = held.loaded && held.arena_bytes >= plan.arena_bytes;
  const double h2d = bytes_per_s(hw.h2d_gbps);
  const double cold = bytes_per_s(file_gbps(hw, false));
  const double cold_device = std::min(cold, h2d);
  if (plan.te.run) {
    const double te = static_cast<double>(bytes.text_encoder);
    if (plan.te.residency == planner::TeResidency::kHostResident && !held.te_host_copy) e.text_encoder = te / cold;
    if (plan.te.residency == planner::TeResidency::kVramResident && !(held.te_vram && arena_kept)) {
      e.text_encoder = te / cold_device;
    }
  }
  const bool dit_held = arena_kept && held.dit.residency != DitResidency::kParked && same_residency(held.dit, plan.dit);
  if (shape.total_steps > 0 && !dit_held) {
    const double device = static_cast<double>(plan.dit.device_bytes);
    if (held.dit_host_copy) {
      e.dit = device / h2d;
    } else {
      const double files = bytes_per_s(file_gbps(hw, held.dit_files_read));
      e.dit = device / std::min(files, h2d);
      // Streamed blocks from a pinned host copy: the copy is read from the files first.
      if (plan.dit.residency == DitResidency::kStreamed && plan.dit.source == planner::DitSource::kHostPinned) {
        e.dit += static_cast<double>(plan.dit.stream_bytes) / files;
      }
    }
  }
  const double vae = static_cast<double>(bytes.vae);
  if (plan.vae.run && !(arena_kept && held.vae)) e.vae_decoder = vae / cold_device;
  if ((shape.encodes_image || shape.hires) && !(arena_kept && held.encoder)) e.vae_encoder = vae / cold_device;
  return e;
}

std::vector<ProgressSegment> generate_segments(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw,
                                               const HeldModels& held, const ModelBytes& bytes, const JobShape& shape) {
  const LoadEstimates load = load_estimates(plan, hw, held, bytes, shape);
  const planner::TimeEstimate& t = plan.time;
  const double per_step = shape.total_steps > 0 ? t.denoise_s / shape.total_steps : 0;
  // A parked DiT reloads every generation (dit_load_s); a first load costs at least as much.
  const double dit = shape.total_steps > 0 ? std::max(t.dit_load_s, load.dit) : 0;
  const int first = shape.hires ? shape.first_steps : shape.total_steps;
  std::vector<ProgressSegment> s;
  s.push_back({MI_PHASE_TEXT_ENCODER, t.text_encoder_s + load.text_encoder});
  s.push_back({MI_PHASE_DIT_LOAD, dit + (shape.encodes_image ? load.vae_encoder : 0)});
  s.push_back({MI_PHASE_DENOISE, 0, first, per_step});
  if (shape.hires) {
    s.push_back({MI_PHASE_DECODE, t.decode_s * shape.first_pixel_share + load.vae_decoder});
    s.push_back({MI_PHASE_DENOISE, shape.encodes_image ? 0 : load.vae_encoder, shape.total_steps - first, per_step});
    s.push_back({MI_PHASE_DECODE, t.decode_s});
  } else {
    s.push_back({MI_PHASE_DECODE, t.decode_s + load.vae_decoder});
  }
  return s;
}

std::vector<ProgressSegment> preload_segments(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw,
                                              const HeldModels& held, const ModelBytes& bytes, const JobShape& shape) {
  const LoadEstimates load = load_estimates(plan, hw, held, bytes, shape);
  std::vector<ProgressSegment> s;
  if (plan.te.run && plan.te.residency == planner::TeResidency::kHostResident) {
    s.push_back({MI_PHASE_TEXT_ENCODER, load.text_encoder});
  }
  if (shape.encodes_image || shape.hires) s.push_back({MI_PHASE_DECODE, load.vae_encoder});
  if (shape.total_steps > 0 && !plan.dit.park_for_text_encoder) s.push_back({MI_PHASE_DIT_LOAD, load.dit});
  if (plan.vae.run && !plan.dit.park_for_decode) s.push_back({MI_PHASE_DECODE, load.vae_decoder});
  return s;
}

void ProgressModel::expect(std::vector<ProgressSegment> segments) {
  segments_ = std::move(segments);
  expected_ = true;
  next_ = 0;
  current_ = -1;
}

void ProgressModel::phase(mi_phase phase, double t) {
  std::size_t i = next_;
  while (i < segments_.size() && segments_[i].phase != phase) ++i;
  if (i == segments_.size()) {
    i = next_;
    segments_.insert(segments_.begin() + static_cast<std::ptrdiff_t>(i), ProgressSegment{phase});
  }
  current_ = static_cast<long>(i);
  next_ = i + 1;
  if (phase == MI_PHASE_DENOISE) {
    pass_start_step_ = steps_;
    last_step_t_ = t;
    first_step_s_ = -1;
    later_steps_ = 0;
    later_s_ = 0;
  }
  update(t);
}

void ProgressModel::step(int step, double step_seconds, double t) {
  const int n = step - steps_;
  if (n > 0) {
    if (steps_ == pass_start_step_) {
      first_step_s_ = step_seconds > 0 ? step_seconds : (t - last_step_t_) / n;
    } else {
      later_steps_ += n;
      later_s_ += t - last_step_t_;
    }
    steps_ = step;
  }
  last_step_t_ = t;
  update(t);
}

double ProgressModel::step_price() const {
  if (later_steps_ > 0) return later_s_ / later_steps_;
  if (first_step_s_ >= 0) return first_step_s_;
  return segments_[static_cast<std::size_t>(current_)].step_seconds;
}

void ProgressModel::update(double t) {
  if (!expected_) {
    eta_ = -1;
    return;
  }
  double eta = 0;
  if (current_ >= 0) {
    const ProgressSegment& c = segments_[static_cast<std::size_t>(current_)];
    if (c.phase == MI_PHASE_DENOISE) {
      const int done = steps_ - pass_start_step_;
      if (done == 0) eta += c.seconds;
      eta += std::max(0, c.steps - done) * step_price();
    } else {
      eta += c.seconds;  // it just started: updates come at phase starts and steps only
    }
  }
  for (std::size_t i = next_; i < segments_.size(); ++i) eta += segments_[i].seconds + segments_[i].steps * segments_[i].step_seconds;
  if (!std::isfinite(eta) || eta < 0) {
    eta_ = -1;
    return;
  }
  eta_ = eta;
  // Everything before this event is done: the phases that ended and the finished steps, at their measured times.
  const double f = t + eta > 0 ? t / (t + eta) : 0;
  fraction_ = std::max(fraction_, std::clamp(f, 0.0, kFractionBeforeEnd));
}

}  // namespace mi::api
