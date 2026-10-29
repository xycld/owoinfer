#include "pipeline/report_json.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <locale>
#include <memory>
#include <sstream>
#include <system_error>

#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
#include "pipeline/quality_report.h"
#include "pipeline/sampler.h"
#include "planner/planner.h"
#include "util/path.h"

namespace mi::pipeline {
namespace {

// Length of the well-formed UTF-8 sequence at s[i] (Unicode Table 3-7), or 0 if it is ill-formed.
std::size_t utf8_sequence_length(std::string_view s, std::size_t i) {
  const auto at = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = at(i);
  if (c < 0x80) return 1;
  std::size_t n = 0;
  unsigned char lo = 0x80, hi = 0xbf;  // valid range of the second byte
  if (c >= 0xc2 && c <= 0xdf) n = 2;
  else if (c >= 0xe0 && c <= 0xef) n = 3, lo = c == 0xe0 ? 0xa0 : 0x80, hi = c == 0xed ? 0x9f : 0xbf;
  else if (c >= 0xf0 && c <= 0xf4) n = 4, lo = c == 0xf0 ? 0x90 : 0x80, hi = c == 0xf4 ? 0x8f : 0xbf;
  else return 0;
  if (i + n > s.size() || at(i + 1) < lo || at(i + 1) > hi) return 0;
  for (std::size_t k = 2; k < n; ++k) {
    if (at(i + k) < 0x80 || at(i + k) > 0xbf) return 0;
  }
  return n;
}

const char* tf(bool b) { return b ? "true" : "false"; }

}  // namespace

std::string json_escape(std::string_view s) {
  std::string out = "\"";
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0x80) {
      const std::size_t n = utf8_sequence_length(s, i);
      if (n == 0) {
        out += "\\ufffd";
        ++i;
      } else {
        out.append(s, i, n);
        i += n;
      }
      continue;
    }
    ++i;
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out + "\"";
}

std::string json_number(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[32];
  const std::to_chars_result res = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::general, 6);
  if (res.ec != std::errc()) return "null";  // cannot happen: 6 significant digits fit
  return std::string(buf, res.ptr);
}

std::string json_layers(const std::vector<int>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + std::to_string(v[i]);
  return out + "]";
}

ReportParts report_parts(const Request& q, const Result& r, std::string_view latent_path) {
  const auto num = json_number;
  const auto& t = r.timings;
  const auto& m = r.memory;
  const double gib = 1024.0 * 1024.0 * 1024.0;
  const auto g = [&](std::size_t bytes) { return num(static_cast<double>(bytes) / gib); };
  ReportParts parts;
  std::ostringstream o;
  o.imbue(std::locale::classic());  // integers too: a global C++ locale could add digit grouping ("1.024")
  o << " \"request\": {\"model\": " << json_escape(util::path_utf8(q.model)) << ", \"prompt\": " << json_escape(q.prompt)
    << ", \"negative_prompt\": " << json_escape(q.negative_prompt) << ", \"height\": " << q.height << ", \"width\": " << q.width
    << ", \"steps\": " << q.steps << ", \"cfg\": " << num(q.cfg) << ", \"shift\": " << num(q.shift) << ", \"seed\": " << q.seed
    << ", \"uncond_skip_layers\": " << json_layers(q.uncond_skip_layers) << ", \"slg_layers\": " << json_layers(q.slg_layers)
    << ", \"slg_scale\": " << num(q.slg_scale) << ", \"slg_start\": " << num(q.slg_start) << ", \"slg_stop\": " << num(q.slg_stop)
    << ", \"max_sequence_length\": " << q.max_sequence_length << ", \"sampler\": \"" << sampler_name(q.sampler)
    << "\", \"scheduler\": \"" << scheduler_name(q.scheduler) << "\", \"flow_multiplier\": " << num(q.flow_multiplier)
    << ", \"noise_layout\": \"" << noise_layout_name(q.noise_layout) << "\", \"noise_source\": \""
    << noise_source_name(q.noise_source) << "\", \"eta\": " << num(q.eta) << ", \"s_noise\": " << num(q.s_noise)
    << ", \"dataspace\": " << json_escape(util::path_utf8(q.dataspace_path));
  // P6b accelerators as requested.
  o << ", \"cfg_truncation\": " << num(q.accel.cfg_truncation) << ", \"step_cache\": \""
    << accel::step_cache_name(q.accel.step_cache) << "\", \"step_cache_threshold\": "
    << (q.accel.step_cache_threshold ? num(*q.accel.step_cache_threshold) : std::string("null"));
  // img2img / inpainting / hires fix (P7b): the images by name and size (their pixels are not reported).
  const auto image = [&](const std::shared_ptr<const Image>& p) {
    if (p == nullptr) return std::string("null");
    const Image& im = *p;
    std::ostringstream is;
    is.imbue(std::locale::classic());
    is << "{\"name\": " << json_escape(im.name) << ", \"width\": " << im.width << ", \"height\": " << im.height << "}";
    return is.str();
  };
  o << ", \"init_image\": " << image(q.init_image) << ", \"denoise_strength\": " << num(q.denoise_strength)
    << ", \"mask\": " << image(q.mask) << ", \"resize\": \"" << resize_mode_name(q.resize) << "\", \"hires\": ";
  if (q.hires.enabled()) {
    o << "{\"width\": " << q.hires.width << ", \"height\": " << q.hires.height << ", \"strength\": " << num(q.hires.strength)
      << ", \"steps\": " << q.hires.steps << "}";
  } else {
    o << "null";
  }
  o << "},\n";
  o << " \"seed_used\": " << r.seed << ", \"steps_scheduled\": " << r.steps_scheduled << ", \"steps_run\": " << r.steps_run
    << ", \"cancelled\": " << tf(r.cancelled) << ", \"cond_tokens\": " << r.cond_tokens
    << ", \"uncond_tokens\": " << r.uncond_tokens << ",\n";
  // Prompt token counts (tokenizer/prompt_tokens.h): what the text encoder kept and whether a prompt was truncated.
  const auto tokens = [&](const tok::PromptTokenReport& tr) {
    std::ostringstream ts;
    ts.imbue(std::locale::classic());
    ts << "{\"total\": " << tr.total_tokens << ", \"kept\": " << tr.kept_tokens << ", \"template\": " << tr.template_tokens
      << ", \"prompt\": " << tr.prompt_tokens() << ", \"dropped\": " << tr.dropped_tokens() << ", \"max_length\": "
      << tr.max_length << ", \"truncated\": " << tf(tr.truncated) << "}";
    return ts.str();
  };
  o << " \"tokens\": {\"prompt\": " << tokens(r.cond_token_report) << ", \"negative_prompt\": " << tokens(r.uncond_token_report)
    << "},\n";
  o << " \"image\": ";
  parts.head = o.str();
  o.str("");
  o << ", \"latents\": " << json_escape(latent_path) << ",\n";
  o << " \"mode\": \"" << planner::mode_name(r.plan.mode) << "\",\n";
  // What the image passes ran: the strengths' start steps, the steps run and the encodes.
  const auto pass = [&](const PassReport& p) {
    std::ostringstream ps;
    ps.imbue(std::locale::classic());
    ps << "{\"ran\": " << tf(p.ran) << ", \"width\": " << p.width << ", \"height\": " << p.height
       << ", \"schedule_steps\": " << p.schedule_steps << ", \"start_step\": " << p.start_step << ", \"steps_run\": "
       << p.steps_run << ", \"strength\": " << num(p.strength) << ", \"encoded\": " << tf(p.encoded)
       << ", \"encode_schedule\": " << json_escape(p.encode_schedule) << ", \"decode_schedule\": " << json_escape(p.decode_schedule)
       << "}";
    return ps.str();
  };
  o << " \"img2img\": {\"mode\": " << json_escape(r.image_mode) << ", \"note\": " << json_escape(r.image_note) << ", \"vae_encoder\": " << json_escape(r.vae_encoder_source)
    << ", \"upscaler\": " << (q.hires.enabled() ? "\"lanczos\"" : "null") << ", \"first_pass\": " << pass(r.first_pass)
    << ", \"hires\": " << (q.hires.enabled() ? pass(r.hires) : std::string("null")) << "},\n";
  o << " \"dit_blocks\": {\"path\": \"" << moss_dit::block_path_name(r.block_path) << "\", \"note\": "
    << json_escape(r.block_path_note) << "},\n";
  const auto strings = [&](const std::vector<std::string>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + json_escape(v[i]);
    return out + "]";
  };
  // P6b: the lossy accelerators that ran and what they skipped.
  const accel::Stats& ac = r.accel;
  accel::Options ran = q.accel;  // the threshold that ran (an unset one: the model's default the pipeline filled in)
  if (ran.step_cache_on() && ac.step_cache_threshold > 0) ran.step_cache_threshold = ac.step_cache_threshold;
  o << " \"accelerators\": {\"lossy\": " << tf(q.accel.active(q.cfg != 1.0)) << ", \"active\": "
    << json_escape(ran.describe(q.cfg != 1.0)) << ", \"step_cache_threshold\": "
    << (ac.step_cache_threshold > 0 ? num(ac.step_cache_threshold) : std::string("null")) << ", \"step_cache_threshold_source\": "
    << json_escape(ac.step_cache_threshold_source) << ", \"cfg_truncated_steps\": " << ac.cfg_truncated_steps
    << ", \"step_cache_calls\": " << ac.dit_calls << ", \"step_cache_reused\": " << ac.cached_calls
    << ", \"cached_steps\": " << json_layers(ac.cached_steps) << ", \"step_cache_diffs\": [";
  // Every call's relative change of block 0 (null where there was nothing to compare with): what the threshold saw.
  for (std::size_t i = 0; i < ac.step_cache_diffs.size(); ++i) o << (i ? ", " : "") << num(ac.step_cache_diffs[i]);
  o << "], \"warnings\": " << strings(r.accel_warnings) << "},\n";
  // P6: the quality object (pipeline/quality_report.h: tier, config, engine numerics, cuBLASLt, GPU class, models),
  // filled by generate(), and where each model came from.
  o << " \"quality\": " << quality_report_json(r.quality) << ",\n";
  o << " \"models\": {\"dit\": {\"source\": " << json_escape(r.dit_source) << ", \"linears\": " << json_escape(r.dit_linears)
    << ", \"tier\": \"" << planner::tier_name(r.tier.tier) << "\", \"time_embedding\": "
    << (r.time_embedding.empty() ? std::string("null") : json_escape(r.time_embedding))
    << "}, \"text_encoder\": {\"source\": " << json_escape(r.te_source) << ", \"precision\": "
    << json_escape(r.te_precision.empty() ? "bf16" : r.te_precision) << ", \"note\": " << json_escape(r.te_note)
    << "}, \"vae\": {\"source\": " << json_escape(r.vae_source)
    << "}},\n";
  // Runtime LoRAs: how each file was read and applied, and the lossy merges.
  o << " \"loras\": [";
  for (std::size_t i = 0; i < r.loras.size(); ++i) {
    const lora::FileReport& f = r.loras[i];
    const float strength = i < q.loras.size() ? q.loras[i].strength : 0.0f;
    o << (i ? ", " : "") << "{\"file\": " << json_escape(f.file) << ", \"strength\": " << num(strength)
      << ", \"adapters\": " << f.adapters << ", \"side_branch\": " << f.side_branch << ", \"merged\": " << f.merged
      << ", \"unknown\": " << strings(f.unknown) << ", \"warnings\": " << strings(f.warnings) << ", \"notes\": "
      << strings(f.notes) << "}";
  }
  o << "], \"lora_warnings\": " << strings(r.lora_warnings) << ", \"lora_notes\": " << strings(r.lora_notes) << ",\n";
  // Control adapter (roadmap P8): the file, its kinds, the conditioning images as requested, its LoRA part as read and
  // the encodes; null without one.
  o << " \"control\": ";
  if (q.control.enabled()) {
    const lora::FileReport& f = r.control.lora;
    o << "{\"adapter\": " << json_escape(util::path_utf8(q.control.adapter)) << ", \"lora_strength\": " << num(q.control.lora_strength)
      << ", \"kinds\": " << strings(r.control.kinds) << ", \"base_model\": " << json_escape(r.control.base_model) << ", \"inputs\": [";
    for (std::size_t i = 0; i < q.control.images.size(); ++i) {
      const Request::ControlImage& c = q.control.images[i];
      o << (i ? ", " : "") << "{\"kind\": " << json_escape(c.kind) << ", \"image\": " << image(c.image) << ", \"strength\": "
        << num(c.strength) << ", \"start\": " << num(c.start) << ", \"end\": " << num(c.end) << "}";
    }
    o << "], \"lora\": {\"file\": " << json_escape(f.file) << ", \"adapters\": " << f.adapters << ", \"side_branch\": "
      << f.side_branch << ", \"merged\": " << f.merged << ", \"unknown\": " << strings(f.unknown) << ", \"warnings\": "
      << strings(f.warnings) << ", \"notes\": " << strings(f.notes) << "}, \"encodes\": " << r.control.encodes
      << ", \"encode_schedule\": " << json_escape(r.control.encode_schedule) << "},\n";
  } else {
    o << "null,\n";
  }
  // Reference images (the edit layout): only when they ran (or an edit adapter ran without them).
  if (const std::string refs = references_json(q, r); !refs.empty()) o << " \"references\": " << refs << ",\n";
  o << " \"plan\": " << r.plan.to_json() << ",\n";
  o << " \"embedding_cache\": {\"hits\": " << r.te_cache_hits << ", \"te_skipped\": " << tf(r.te_skipped) << "},\n";
  // What ran (it may differ from plan.vae: the decode takes the fastest schedule the arena it finds leaves room for).
  o << " \"decode\": {\"ran\": " << tf(!r.rgb.empty()) << ", \"tiled\": " << tf(r.decode_tiled) << ", \"schedule\": "
    << json_escape(r.decode_schedule) << ", \"planned_tiled\": " << tf(r.plan.vae.run && r.plan.vae.kind == planner::VaeDecodeKind::kTiled)
    << ", \"planned_tile\": " << r.plan.vae.tile << "},\n";
  const double load = t.index_load_s + t.dit_load_s + t.vae_load_s + t.vae_encoder_load_s;
  const double encode = t.tokenize_s + t.text_encode_s + t.refine_s;
  o << " \"timings_s\": {\"total\": " << num(t.total_s) << ", \"load\": " << num(load) << ", \"encode\": " << num(encode)
    << ", \"denoise\": " << num(t.denoise_s) << ", \"decode\": " << num(t.decode_s) << ", \"index_load\": " << num(t.index_load_s)
    << ", \"tokenize\": " << num(t.tokenize_s) << ", \"plan\": " << num(t.plan_s) << ", \"text_encode\": " << num(t.text_encode_s)
    << ", \"te_weight_load\": " << num(t.te_weight_load_s) << ", \"te_weight_wait\": " << num(t.te_weight_wait_s)
    << ", \"te_compute\": " << num(t.te_compute_s) << ", \"dit_load\": " << num(t.dit_load_s)
    << ", \"lora_load\": " << num(t.lora_load_s) << ", \"lora_merge\": " << num(t.lora_merge_s) << ", \"refine\": " << num(t.refine_s)
    << ", \"vae_load\": " << num(t.vae_load_s) << ", \"vae_encoder_load\": " << num(t.vae_encoder_load_s)
    << ", \"vae_encode\": " << num(t.encode_s) << ", \"image_ops\": " << num(t.image_ops_s);
  if (r.references.count > 0) {
    o << ", \"reference_encode\": " << num(t.reference_encode_s) << ", \"reference_prefill\": " << num(t.reference_prefill_s);
  }
  o << ", \"steps\": [";
  for (std::size_t i = 0; i < t.step_s.size(); ++i) o << (i ? ", " : "") << num(t.step_s[i]);
  o << "]},\n";
  o << " \"memory_gib\": {\"arena_capacity\": " << g(m.arena_capacity) << ", \"arena_high_water\": " << g(m.arena_high_water)
    << ", \"device_peak\": " << g(m.device_peak) << ", \"free_at_start\": " << g(m.free_at_start)
    << ", \"arena_creations\": " << m.arena_creations << ", \"dit_parked_for_text_encoder\": " << tf(m.dit_parked_for_text_encoder)
    << ", \"dit_parked_for_decode\": " << tf(m.dit_parked_for_decode) << ", \"dit_reloaded\": " << tf(m.dit_reloaded) << ", \"dit_loads\": " << m.dit_loads << "},\n";
  o << " \"host_memory_gib\": {\"start\": " << g(m.host_private_start) << ", \"private_peak\": " << g(m.host_private_peak)
    << ", \"host_copies\": " << g(m.host_copy_bytes) << ", \"merge_memo\": " << g(m.merge_memo_bytes) << "},\n";
  // How the DiT's weights were read when the call loaded it (source "" when it was resident): from the files with
  // direct or buffered I/O (and why buffered), or from a soft unload's host copy.
  if (!r.preload_note.empty()) o << " \"preload_note\": " << json_escape(r.preload_note) << ",\n";
  const DitLoadReport& dl = r.dit_load;
  o << " \"dit_load\": {\"source\": " << (dl.source.empty() ? std::string("null") : json_escape(dl.source)) << ", \"io\": "
    << (dl.io.empty() ? std::string("null") : json_escape(dl.io)) << ", \"note\": " << json_escape(dl.note)
    << ", \"bytes\": " << dl.bytes << ", \"seconds\": " << num(dl.seconds) << ", \"gbps\": " << num(dl.gbps()) << "},\n";
  const auto& hw = r.hardware;
  o << " \"hardware\": {\"measured\": " << tf(hw.measured) << ", \"h2d_gbps\": " << num(hw.h2d_gbps) << ", \"host_gbps\": "
    << num(hw.host_gbps) << ", \"disk_gbps\": " << num(hw.disk_gbps) << ", \"bf16_tflops\": " << num(hw.bf16_tflops) << "},\n";
  // P4 step 28: DiT block streaming, per step and over the run (hidden = fraction of the copy time behind compute).
  const auto& ds = r.dit_stream;
  o << " \"dit_stream\": {\"streamed\": " << tf(ds.streamed) << ", \"resident_blocks\": " << ds.resident_blocks
    << ", \"buffers\": " << ds.buffers << ", \"source\": " << json_escape(ds.source) << ", \"bytes\": " << ds.bytes
    << ", \"copy_s\": " << num(ds.copy_s) << ", \"stall_s\": " << num(ds.stall_s) << ", \"stalls\": " << ds.stalls
    << ", \"h2d_gbps\": " << num(ds.gbps()) << ", \"hidden\": " << num(ds.hidden()) << ", \"steps\": [";
  for (std::size_t i = 0; i < ds.steps.size(); ++i) {
    const auto& st = ds.steps[i];
    o << (i ? ", " : "") << "{\"h2d_gbps\": " << num(st.gbps) << ", \"hidden\": " << num(st.hidden) << ", \"stall_s\": "
      << num(st.stall_s) << ", \"stalls\": " << st.stalls << ", \"cached\": " << tf(st.cached) << "}";
  }
  o << "]}";
  parts.tail = o.str();
  return parts;
}

std::string references_json(const Request& q, const Result& r) {
  if (r.references.count == 0 && r.reference_warnings.empty()) return {};
  std::ostringstream o;
  o.imbue(std::locale::classic());
  std::string warnings = "[";
  for (std::size_t i = 0; i < r.reference_warnings.size(); ++i) warnings += (i ? ", " : "") + json_escape(r.reference_warnings[i]);
  warnings += "]";
  if (r.references.count == 0) return "{\"count\": 0, \"warnings\": " + warnings + "}";
  o << "{\"count\": " << r.references.count << ", \"max_edge\": " << q.references.max_edge << ", \"sizes\": [";
  for (std::size_t i = 0; i < r.references.sizes.size(); ++i) {
    o << (i ? ", " : "") << "[" << r.references.sizes[i].width << ", " << r.references.sizes[i].height << "]";
  }
  o << "], \"tokens\": " << r.references.tokens << ", \"cache_bytes\": " << r.references.cache_bytes
    << ", \"prefills\": " << r.references.prefills << ", \"encode_schedule\": " << json_escape(r.references.encode_schedule)
    << ", \"warnings\": " << warnings << "}";
  return o.str();
}

std::string report_fields(const Request& q, const Result& r, std::string_view image, std::string_view latent) {
  const ReportParts parts = report_parts(q, r, latent);
  return parts.head + json_escape(image) + parts.tail;
}

}  // namespace mi::pipeline
