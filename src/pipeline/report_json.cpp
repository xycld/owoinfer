#include "pipeline/report_json.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <locale>
#include <sstream>
#include <system_error>

#include "models/moss_dit/moss_dit.h"
#include "pipeline/pipeline.h"
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
    << ", \"max_sequence_length\": " << q.max_sequence_length << ", \"sampler\": \"" << sampler_name(q.sampler) << "\"},\n";
  o << " \"seed_used\": " << r.seed << ", \"steps_scheduled\": " << r.steps_scheduled << ", \"steps_run\": " << r.steps_run
    << ", \"cancelled\": " << tf(r.cancelled) << ", \"cond_tokens\": " << r.cond_tokens
    << ", \"uncond_tokens\": " << r.uncond_tokens << ",\n";
  o << " \"image\": ";
  parts.head = o.str();
  o.str("");
  o << ", \"latents\": " << json_escape(latent_path) << ",\n";
  o << " \"mode\": \"" << planner::mode_name(r.plan.mode) << "\",\n";
  o << " \"dit_blocks\": {\"path\": \"" << moss_dit::block_path_name(r.block_path) << "\", \"note\": "
    << json_escape(r.block_path_note) << "},\n";
  o << " \"plan\": " << r.plan.to_json() << ",\n";
  o << " \"embedding_cache\": {\"hits\": " << r.te_cache_hits << ", \"te_skipped\": " << tf(r.te_skipped) << "},\n";
  // What ran (it may differ from plan.vae: the decode takes the fastest schedule the arena it finds leaves room for).
  o << " \"decode\": {\"ran\": " << tf(!r.rgb.empty()) << ", \"tiled\": " << tf(r.decode_tiled) << ", \"schedule\": "
    << json_escape(r.decode_schedule) << ", \"planned_tiled\": " << tf(r.plan.vae.run && r.plan.vae.kind == planner::VaeDecodeKind::kTiled)
    << ", \"planned_tile\": " << r.plan.vae.tile << "},\n";
  const double load = t.index_load_s + t.dit_load_s + t.vae_load_s;
  const double encode = t.tokenize_s + t.text_encode_s + t.refine_s;
  o << " \"timings_s\": {\"total\": " << num(t.total_s) << ", \"load\": " << num(load) << ", \"encode\": " << num(encode)
    << ", \"denoise\": " << num(t.denoise_s) << ", \"decode\": " << num(t.decode_s) << ", \"index_load\": " << num(t.index_load_s)
    << ", \"tokenize\": " << num(t.tokenize_s) << ", \"plan\": " << num(t.plan_s) << ", \"text_encode\": " << num(t.text_encode_s)
    << ", \"te_weight_load\": " << num(t.te_weight_load_s) << ", \"te_weight_wait\": " << num(t.te_weight_wait_s)
    << ", \"te_compute\": " << num(t.te_compute_s) << ", \"dit_load\": " << num(t.dit_load_s) << ", \"refine\": " << num(t.refine_s)
    << ", \"vae_load\": " << num(t.vae_load_s) << ", \"steps\": [";
  for (std::size_t i = 0; i < t.step_s.size(); ++i) o << (i ? ", " : "") << num(t.step_s[i]);
  o << "]},\n";
  o << " \"memory_gib\": {\"arena_capacity\": " << g(m.arena_capacity) << ", \"arena_high_water\": " << g(m.arena_high_water)
    << ", \"device_peak\": " << g(m.device_peak) << ", \"free_at_start\": " << g(m.free_at_start)
    << ", \"arena_creations\": " << m.arena_creations << ", \"dit_parked_for_text_encoder\": " << tf(m.dit_parked_for_text_encoder)
    << ", \"dit_parked_for_decode\": " << tf(m.dit_parked_for_decode) << ", \"dit_reloaded\": " << tf(m.dit_reloaded) << "},\n";
  o << " \"host_memory_gib\": {\"start\": " << g(m.host_private_start) << ", \"private_peak\": " << g(m.host_private_peak) << "},\n";
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
      << num(st.stall_s) << ", \"stalls\": " << st.stalls << "}";
  }
  o << "]}";
  parts.tail = o.str();
  return parts;
}

std::string report_fields(const Request& q, const Result& r, std::string_view image, std::string_view latent) {
  const ReportParts parts = report_parts(q, r, latent);
  return parts.head + json_escape(image) + parts.tail;
}

}  // namespace mi::pipeline
