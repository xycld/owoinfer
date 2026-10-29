#include "api/progress_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <exception>
#include <optional>
#include <utility>

#include "models/lora/lora.h"
#include "pipeline/hardware_profile.h"
#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "util/path.h"
#include "util/regular_file.h"

namespace mi::api {

namespace {

using planner::DitPlan;
using planner::DitResidency;

double bytes_per_s(double gbps) { return std::max(1e-3, gbps) * 1e9; }

// The planner's keep_dit test: the residency, and what parks or streams.
bool same_residency(const DitPlan& a, const DitPlan& b) {
  if (a.residency != b.residency) return false;
  if (a.residency == DitResidency::kParked) {
    return a.park_for_text_encoder == b.park_for_text_encoder && a.park_for_decode == b.park_for_decode;
  }
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

// The DIT_LOAD estimate of a generation: a load when the DiT is not held, else the plan's parking reload. A parked DiT
// stays loaded after a job that did not park it for its decode, until a text-encoder pass evicts it.
double dit_load_seconds(const planner::ExecutionPlan& plan, const HeldModels& held, const LoadEstimates& load,
                        const JobShape& shape) {
  if (shape.total_steps <= 0) return 0;
  if (!load.dit_held) return std::max(plan.time.dit_load_s, load.dit);
  if (plan.dit.residency == DitResidency::kParked && !plan.te.run && !held.last_dit.park_for_decode) return 0;
  return std::max(plan.time.dit_load_s, 0.0);
}

// Bytes of a component's weights: its file, else the *.safetensors files in the bundle's folder (0 when unreadable).
std::uint64_t weight_bytes(const std::filesystem::path& file, const std::filesystem::path& folder) {
  if (!file.empty()) return util::file_info(file).size;
  std::uint64_t total = 0;
  std::vector<util::DirEntry> entries;
  util::list_directory(folder, entries);
  for (const util::DirEntry& e : entries) {
    if (e.name.extension() == ".safetensors") total += e.link ? util::file_info(folder / e.name).size : e.info.size;
  }
  return total;
}

std::string seconds_text(double s) {
  char b[32];
  std::snprintf(b, sizeof b, "%.1f s", s);
  return b;
}

}  // namespace

JobShape job_shape(const pipeline::Request& q) {
  JobShape s;
  s.total_steps = static_cast<int>(pipeline::schedule_steps(q));  // both passes, after the strengths
  s.first_steps = s.total_steps;
  s.hires = q.hires.enabled();
  if (s.hires) {
    pipeline::Request first = q;
    first.hires = {};
    s.first_steps = static_cast<int>(pipeline::schedule_steps(first));
    const double target = static_cast<double>(q.hires.width) * static_cast<double>(q.hires.height);
    if (target > 0) s.first_pixel_share = static_cast<double>(q.width) * static_cast<double>(q.height) / target;
  }
  // The pipeline's img2img condition (an init image at strength 1 without a mask is not used), and control images.
  s.encodes_image = (q.init_image != nullptr && (q.denoise_strength < 1.0 || q.mask != nullptr)) ||
                    (q.control.enabled() && !q.control.images.empty() && s.total_steps > 0) ||
                    (q.references.enabled() && s.total_steps > 0);  // reference images: encoded before the DiT load
  // A LoRA change may re-merge or reload the DiT: the progress prices a load whenever the stack differs.
  for (const lora::LoraSpec& l : q.loras) s.merge_key += util::path_utf8(l.path) + '\n' + std::to_string(l.strength) + '\n';
  if (q.control.enabled()) {
    s.merge_key += "control\n" + util::path_utf8(q.control.adapter) + '\n' + std::to_string(q.control.lora_strength);
  }
  return s;
}

ModelBytes model_bytes(const std::filesystem::path& model, const std::filesystem::path& text_encoder_file,
                       const std::filesystem::path& vae_file) {
  ModelBytes b;
  b.text_encoder = weight_bytes(text_encoder_file, model / "text_encoder");
  b.vae = weight_bytes(vae_file, model / "vae");
  return b;
}

planner::HardwareProfile stored_rates(const std::filesystem::path& cache_dir) {
  if (!cache_dir.empty()) {
    try {
      const std::optional<pipeline::StoredProfile> stored =
          pipeline::read_profile(cache_dir / "hardware_profile.json", pipeline::hardware_key(), std::time(nullptr));
      if (stored) return stored->learned;
    } catch (const std::exception&) {
      // An estimate never fails a job: the defaults stand.
    }
  }
  return planner::HardwareProfile{};
}

Estimate estimate_of(const std::vector<ProgressSegment>& segments, const planner::ExecutionPlan& plan,
                     const planner::HardwareProfile& hw) {
  Estimate e;
  double step_total = 0;
  for (const ProgressSegment& s : segments) {
    const double steps_s = s.steps * s.step_seconds;
    const double t = s.seconds + steps_s;
    switch (s.phase) {
      case MI_PHASE_TEXT_ENCODER: e.text_encoder_s += t; break;
      case MI_PHASE_DIT_LOAD: e.dit_load_s += t; break;
      case MI_PHASE_DENOISE:
        e.denoise_s += t;
        e.steps += s.steps;
        step_total += steps_s;
        break;
      case MI_PHASE_DECODE: e.decode_s += t; break;
      case MI_PHASE_MAX_ENUM: break;
    }
  }
  e.total_s = e.text_encoder_s + e.dit_load_s + e.denoise_s + e.decode_s;
  e.step_s = e.steps > 0 ? step_total / e.steps : 0;
  e.reference_prefill_s = e.steps > 0 ? plan.time.reference_prefill_s : 0;
  e.rates_measured = hw.measured;
  e.load_rates_learned = hw.direct_gbps > 0 || hw.buffered_warm_gbps > 0;
  return e;
}

std::string estimate_json(const Estimate& e) {
  using pipeline::json_number;
  return "{\"total_s\": " + json_number(e.total_s) + ", \"text_encoder_s\": " + json_number(e.text_encoder_s) +
         ", \"dit_load_s\": " + json_number(e.dit_load_s) + ", \"denoise_s\": " + json_number(e.denoise_s) +
         ", \"decode_s\": " + json_number(e.decode_s) + ", \"steps\": " + std::to_string(e.steps) +
         ", \"step_s\": " + json_number(e.step_s) + ", \"reference_prefill_s\": " + json_number(e.reference_prefill_s) +
         ", \"rates\": \"" + (e.rates_measured ? "measured" : "default") + "\", \"load_rates\": \"" +
         (e.load_rates_learned ? "learned" : e.rates_measured ? "probed" : "default") + "\"}";
}

std::string estimate_text(const Estimate& e) {
  std::string t = "Estimate: " + seconds_text(e.total_s) + " (text encoder " + seconds_text(e.text_encoder_s) +
                  ", DiT load " + seconds_text(e.dit_load_s) + ", denoise " + seconds_text(e.denoise_s);
  if (e.steps > 0) {
    char b[64];
    std::snprintf(b, sizeof b, ": %d steps of %.2f s", e.steps, e.step_s);
    t += b;
  }
  t += ", decode " + seconds_text(e.decode_s) + "); ";
  t += e.rates_measured ? "rates measured on this machine" : "the planner's default rates (nothing measured yet)";
  if (e.rates_measured) t += e.load_rates_learned ? ", load rates learned" : "";
  return t + ".";
}

std::string with_estimate(const std::string& plan_json, const Estimate& e) {
  const std::size_t close = plan_json.rfind('}');
  if (close == std::string::npos) return plan_json;
  return plan_json.substr(0, close) + ", \"estimate\": " + estimate_json(e) + plan_json.substr(close);
}

void HeldModels::job_done(const planner::ExecutionPlan& plan, const JobShape& shape, bool preload, bool cancelled,
                          int arena_creations, std::string_view dit_load_source) {
  const bool uses_dit = shape.total_steps > 0 && !(preload && plan.dit.park_for_text_encoder);
  if (arena_creations > 0) {
    // A new arena dropped every device model; the job loads again what it uses.
    if (cancelled) forget_device();
    te_vram = false;
    if (!uses_dit) dit = false;
  }
  if (!cancelled) {
    warm = true;
    if (uses_dit) {
      dit = true;
      last_dit = plan.dit;
      last_merge_key = shape.merge_key;
    }
    if (plan.te.run && plan.te.residency == planner::TeResidency::kHostResident) te_host_copy = true;
    if (!preload && plan.te.run && plan.te.residency == planner::TeResidency::kVramResident) te_vram = true;
  } else if (uses_dit && (!same_residency(last_dit, plan.dit) || last_merge_key != shape.merge_key)) {
    dit = false;  // the job may have dropped the held DiT for its own before the cancel
  }
  if (dit_load_source == "files") dit_files_read = true;
  if (dit_load_source == "host_copy") dit_host_copy = false;  // the copy is freed once it served the load
}

void HeldModels::released(bool kept_dit_host_copy, bool kept_te_host_copy) {
  forget_device();
  dit_host_copy = kept_dit_host_copy;
  te_host_copy = kept_te_host_copy;
}

LoadEstimates load_estimates(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw, const HeldModels& held,
                             const ModelBytes& bytes, const JobShape& shape) {
  LoadEstimates e;
  const double h2d = bytes_per_s(hw.h2d_gbps);
  const double cold = bytes_per_s(file_gbps(hw, false));
  const double cold_device = std::min(cold, h2d);
  if (plan.te.run) {
    const double te = static_cast<double>(bytes.text_encoder);
    if (plan.te.residency == planner::TeResidency::kHostResident && !held.te_host_copy) e.text_encoder = te / cold;
    if (plan.te.residency == planner::TeResidency::kVramResident && !held.te_vram) e.text_encoder = te / cold_device;
  }
  e.dit_held = held.dit && same_residency(held.last_dit, plan.dit) && held.last_merge_key == shape.merge_key;
  if (shape.total_steps > 0 && !e.dit_held) {
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
  if (!held.warm) {
    const double vae = static_cast<double>(bytes.vae);
    if (plan.vae.run) e.vae_decoder = vae / cold_device;
    if (shape.encodes_image || shape.hires) e.vae_encoder = vae / cold_device;
  }
  return e;
}

std::vector<ProgressSegment> generate_segments(const planner::ExecutionPlan& plan, const planner::HardwareProfile& hw,
                                               const HeldModels& held, const ModelBytes& bytes, const JobShape& shape) {
  const LoadEstimates load = load_estimates(plan, hw, held, bytes, shape);
  const planner::TimeEstimate& t = plan.time;
  const int first = shape.hires ? shape.first_steps : shape.total_steps;
  // Reference images: each pass's prefill (in denoise_s) is that pass's work before its first step.
  const int passes = (first > 0 ? 1 : 0) + (shape.hires && shape.total_steps > first ? 1 : 0);
  const double prefill = passes > 0 ? t.reference_prefill_s / passes : 0;
  const double steps_s = t.denoise_s - t.reference_prefill_s;
  // The planner counts a hires fix's first-pass steps at the target's cost times their pixel share.
  double first_price = shape.total_steps > 0 ? steps_s / shape.total_steps : 0;
  double second_price = first_price;
  if (shape.hires) {
    const double share = shape.first_pixel_share;
    const double target_steps = (shape.total_steps - first) + first * share;
    second_price = target_steps > 0 ? steps_s / target_steps : 0;
    first_price = second_price * share;
  }
  std::vector<ProgressSegment> s;
  s.push_back({MI_PHASE_TEXT_ENCODER, t.text_encoder_s + load.text_encoder});
  s.push_back({MI_PHASE_DIT_LOAD, dit_load_seconds(plan, held, load, shape) + (shape.encodes_image ? load.vae_encoder : 0)});
  s.push_back({MI_PHASE_DENOISE, first > 0 ? prefill : 0, first, first_price});
  if (shape.hires) {
    s.push_back({MI_PHASE_DECODE, t.decode_s * shape.first_pixel_share + load.vae_decoder});
    s.push_back({MI_PHASE_DENOISE, (shape.encodes_image ? 0 : load.vae_encoder) + (shape.total_steps > first ? prefill : 0),
                 shape.total_steps - first, second_price});
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
  // The text encoder's copy and the VAE encoder follow conditions the progress sees only roughly: optional.
  if (plan.te.run && plan.te.residency == planner::TeResidency::kHostResident) {
    s.push_back({MI_PHASE_TEXT_ENCODER, load.text_encoder, 0, 0, true});
  }
  if (shape.encodes_image || shape.hires) s.push_back({MI_PHASE_DECODE, load.vae_encoder, 0, 0, true});
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
  while (i < segments_.size() && segments_[i].phase != phase && segments_[i].optional) ++i;
  if (i == segments_.size() || segments_[i].phase != phase) {
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
    const double plan = current_ >= 0 ? segments_[static_cast<std::size_t>(current_)].step_seconds : 0;
    if (plan > 0) scale_ = step_price() / plan;
  }
  last_step_t_ = t;
  update(t);
}

double ProgressModel::step_price() const {
  if (later_steps_ > 0) return later_s_ / later_steps_;
  if (first_step_s_ >= 0) return first_step_s_;
  return segments_[static_cast<std::size_t>(current_)].step_seconds * scale_;
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
  for (std::size_t i = next_; i < segments_.size(); ++i) {
    eta += segments_[i].seconds + segments_[i].steps * segments_[i].step_seconds * scale_;
  }
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
