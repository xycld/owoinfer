// `moss-infer generate`: prompt or parameters.json -> PNG (and optionally the final latents and a JSON report).
#include <atomic>
#include <charconv>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pipeline/pipeline.h"
#include "pipeline/png_writer.h"
#include "planner/planner.h"
#include "tools/cli/commands.h"
#include "util/json.h"
#include "util/path.h"

namespace mi::cli {
namespace {

namespace fs = std::filesystem;
using util::path_from_utf8;

constexpr const char* kUsage =
    "usage: moss-infer generate (--params <parameters.json> | --prompt <text>) --model <bundle_dir>\n"
    "                           [-o <out.png>] [--latent-out <latents.npy>] [--json-report <report.json>]\n"
    "                           [--negative <text>] [--size <W>x<H>] [--width N] [--height N] [--steps N]\n"
    "                           [--cfg X] [--shift X] [--seed N|-1] [--skip-layers \"22\"] [--slg-layers \"\"]\n"
    "                           [--slg-scale X] [--slg-start X] [--slg-stop X] [--max-seq N] [--sampler euler|ab2]\n"
    "                           [--mode min_memory|balanced|max_speed] [--vram-budget GiB] [--vram-reserve MiB]\n"
    "                           [--host-ram-budget GiB] [--cache-dir <dir>] [--then-prompt <text>]... [--direct-io]\n"
    "                           [--dit-blocks op_by_op|fused|fused_graphs] [--quiet]\n"
    "Flags override values read from --params. At least one of -o / --latent-out is required;\n"
    "without -o the VAE decode is skipped. --vram-budget caps the process's GPU memory (CUDA context included).\n"
    "Each --then-prompt generates one more image in the same engine session (same settings, warm models and\n"
    "embedding cache), written to <out>.<n>.png and reported under \"followups\".\n"
    "--dit-blocks (default fused, or the MOSSINFER_DIT_BLOCKS environment variable) picks the DiT block implementation;\n"
    "all give the same image. fused falls back to op_by_op when its startup self-check fails; fused_graphs adds CUDA graphs.\n";

std::atomic<bool> g_cancel{false};
// First Ctrl-C: cancel (within a text-encoder layer, DiT block or decode band). A second one terminates (default action restored).
extern "C" void on_sigint(int) {
  g_cancel.store(true);
  std::signal(SIGINT, SIG_DFL);
}

std::string read_file(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + util::path_utf8(p));
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

double to_double(const std::string& flag, const std::string& v) {
  std::size_t used = 0;
  const double d = std::stod(v, &used);
  if (used != v.size()) throw std::invalid_argument(flag + ": not a number: " + v);
  return d;
}

std::int64_t to_int(const std::string& flag, const std::string& v) {
  std::size_t used = 0;
  const long long x = std::stoll(v, &used);
  if (used != v.size()) throw std::invalid_argument(flag + ": not an integer: " + v);
  return x;
}

int to_int32(const std::string& flag, const std::string& v) {
  const std::int64_t x = to_int(flag, v);
  if (x < INT32_MIN || x > INT32_MAX) throw std::invalid_argument(flag + ": out of range: " + v);
  return static_cast<int>(x);
}

// Length of the well-formed UTF-8 sequence at s[i] (Unicode Table 3-7: no overlongs, surrogates or code points
// above U+10FFFF), or 0 if it is ill-formed.
std::size_t utf8_sequence_length(const std::string& s, std::size_t i) {
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

// JSON string literal. Ill-formed UTF-8 (e.g. a non-UTF-8 file name) becomes U+FFFD per bad byte, so the report
// stays valid JSON.
std::string json_escape(const std::string& s) {
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

std::string json_layers(const std::vector<int>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + std::to_string(v[i]);
  return out + "]";
}

// Locale-independent (std::to_chars ignores LC_NUMERIC); JSON has no inf/nan.
std::string num(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[32];
  const auto res = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::general, 6);
  return std::string(buf, res.ptr);
}

// NumPy .npy v1.0, little-endian fp32, C order.
void write_npy_f32(const fs::path& path, const std::vector<float>& data, const std::vector<std::int64_t>& shape) {
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (std::size_t i = 0; i < shape.size(); ++i) header += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
  header += "), }";
  const std::size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write("\x93NUMPY\x01\x00", 8);
  const std::uint16_t len = static_cast<std::uint16_t>(header.size());
  const char len_le[2] = {static_cast<char>(len & 0xff), static_cast<char>(len >> 8)};
  out.write(len_le, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
  out.close();
  if (!out) throw std::runtime_error("cannot write " + util::path_utf8(path));
}

const char* tf(bool b) { return b ? "true" : "false"; }

// One generation's fields of the JSON report (no braces).
std::string run_json(const pipeline::Request& q, const pipeline::Result& r, const std::string& image, const std::string& latent) {
  const auto& t = r.timings;
  const auto& m = r.memory;
  const double gib = 1024.0 * 1024.0 * 1024.0;
  const auto g = [&](std::size_t bytes) { return num(static_cast<double>(bytes) / gib); };
  std::ostringstream o;
  o << " \"request\": {\"model\": " << json_escape(util::path_utf8(q.model)) << ", \"prompt\": " << json_escape(q.prompt)
    << ", \"negative_prompt\": " << json_escape(q.negative_prompt) << ", \"height\": " << q.height << ", \"width\": " << q.width
    << ", \"steps\": " << q.steps << ", \"cfg\": " << num(q.cfg) << ", \"shift\": " << num(q.shift) << ", \"seed\": " << q.seed
    << ", \"uncond_skip_layers\": " << json_layers(q.uncond_skip_layers) << ", \"slg_layers\": " << json_layers(q.slg_layers)
    << ", \"slg_scale\": " << num(q.slg_scale) << ", \"slg_start\": " << num(q.slg_start) << ", \"slg_stop\": " << num(q.slg_stop)
    << ", \"max_sequence_length\": " << q.max_sequence_length << ", \"sampler\": \"" << pipeline::sampler_name(q.sampler) << "\"},\n";
  o << " \"seed_used\": " << r.seed << ", \"steps_scheduled\": " << r.steps_scheduled << ", \"steps_run\": " << r.steps_run
    << ", \"cancelled\": " << tf(r.cancelled) << ", \"cond_tokens\": " << r.cond_tokens
    << ", \"uncond_tokens\": " << r.uncond_tokens << ",\n";
  o << " \"image\": " << json_escape(image) << ", \"latents\": " << json_escape(latent) << ",\n";
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
  return o.str();
}

void write_text_file(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
  out.close();
  if (!out) throw std::runtime_error("cannot write " + util::path_utf8(path));
}

// <out>.<n><ext> for follow-up image n.
fs::path followup_path(const fs::path& out, std::size_t n) {
  fs::path p = out;
  p.replace_filename(out.stem().u8string() + u8"." + fs::path(std::to_string(n)).u8string() + out.extension().u8string());
  return p;
}

void print_summary(const pipeline::Result& r) {
  const auto& t = r.timings;
  std::fprintf(stderr,
               "seed %llu, %d steps, %d+%d tokens%s: total %.2f s (load %.2f, encode %.2f, denoise %.2f, decode %.2f%s); "
               "device peak %.2f GiB (arena high-water %.2f GiB), host private peak %.2f GiB%s; DiT blocks %s\n",
               static_cast<unsigned long long>(r.seed), r.steps_run, r.cond_tokens, r.uncond_tokens,
               r.te_skipped ? " (embedding cache hit)" : "", t.total_s, t.index_load_s + t.dit_load_s + t.vae_load_s,
               t.tokenize_s + t.text_encode_s + t.refine_s, t.denoise_s, t.decode_s, r.decode_tiled ? ", tiled" : "",
               static_cast<double>(r.memory.device_peak) / (1 << 30), static_cast<double>(r.memory.arena_high_water) / (1 << 30),
               static_cast<double>(r.memory.host_private_peak) / (1 << 30),
               r.memory.dit_parked_for_decode || r.memory.dit_parked_for_text_encoder ? ", DiT parked"
               : r.memory.dit_reloaded                                                ? ", DiT reloaded"
                                                                                      : "",
               moss_dit::block_path_name(r.block_path));
  if (!r.block_path_note.empty()) std::fprintf(stderr, "%s\n", r.block_path_note.c_str());
  const auto& ds = r.dit_stream;
  if (ds.streamed) {
    std::fprintf(stderr, "DiT streamed: %d of 42 blocks resident, %d buffers from %s; %.1f GB/s, %.1f%% of the copy hidden, "
                 "%d stalls (%.2f s)\n",
                 ds.resident_blocks, ds.buffers, ds.source.c_str(), ds.gbps(), 100.0 * ds.hidden(), ds.stalls, ds.stall_s);
  }
}

}  // namespace

int cmd_generate(const std::vector<std::string>& args) {
  pipeline::Request req;
  pipeline::PipelineOptions options;
  fs::path params, out_png, out_latent, report;
  std::vector<std::string> then_prompts;
  bool have_prompt = false, quiet = false;
  // Flags apply after --params is read, in command-line order.
  std::vector<std::pair<std::string, std::string>> flags;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--direct-io" || a == "--quiet") {
      flags.emplace_back(a, "");
      continue;
    }
    if (a.rfind("-", 0) != 0 || i + 1 >= args.size()) {
      std::cerr << kUsage;
      return 2;
    }
    flags.emplace_back(a, args[++i]);
  }
  for (const auto& [f, v] : flags) {
    if (f == "--params") params = path_from_utf8(v);
  }
  if (!params.empty()) req = pipeline::request_from_parameters(json::parse(read_file(params)));
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  try {
    // The environment variable overrides the default block path (A/B runs); --dit-blocks overrides both.
    if (const char* env = std::getenv("MOSSINFER_DIT_BLOCKS"); env != nullptr && *env != '\0') {
      options.block_path = moss_dit::parse_block_path(env);
    }
    for (const auto& [f, v] : flags) {
      if (f == "--params") continue;
      else if (f == "--prompt") req.prompt = v, have_prompt = true;
      else if (f == "--model") req.model = path_from_utf8(v);
      else if (f == "-o" || f == "--output") out_png = path_from_utf8(v);
      else if (f == "--latent-out") out_latent = path_from_utf8(v);
      else if (f == "--json-report") report = path_from_utf8(v);
      else if (f == "--negative") req.negative_prompt = v;
      else if (f == "--size") {
        const std::size_t x = v.find('x');
        if (x == std::string::npos) throw std::invalid_argument("--size: expected <W>x<H>");
        req.width = to_int32(f, v.substr(0, x));
        req.height = to_int32(f, v.substr(x + 1));
      }
      else if (f == "--width") req.width = to_int32(f, v);
      else if (f == "--height") req.height = to_int32(f, v);
      else if (f == "--steps") req.steps = to_int32(f, v);
      else if (f == "--cfg") req.cfg = to_double(f, v);
      else if (f == "--shift") req.shift = to_double(f, v);
      else if (f == "--seed") req.seed = to_int(f, v);
      else if (f == "--skip-layers") req.uncond_skip_layers = pipeline::parse_layer_list(v);
      else if (f == "--slg-layers") req.slg_layers = pipeline::parse_layer_list(v);
      else if (f == "--slg-scale") req.slg_scale = to_double(f, v);
      else if (f == "--slg-start") req.slg_start = to_double(f, v);
      else if (f == "--slg-stop") req.slg_stop = to_double(f, v);
      else if (f == "--max-seq") req.max_sequence_length = to_int32(f, v);
      else if (f == "--sampler") req.sampler = pipeline::parse_sampler(v);
      else if (f == "--mode") options.mode = planner::parse_mode(v);
      else if (f == "--vram-budget") options.vram_budget = pipeline::vram_option_bytes(to_double(f, v), kGiB, f);
      else if (f == "--vram-reserve") options.vram_reserve = pipeline::vram_option_bytes(to_double(f, v), 1024.0 * 1024.0, f);
      else if (f == "--host-ram-budget") options.host_ram_budget = pipeline::vram_option_bytes(to_double(f, v), kGiB, f);
      else if (f == "--cache-dir") options.cache_dir = path_from_utf8(v);
      else if (f == "--then-prompt") then_prompts.push_back(v);
      else if (f == "--direct-io") options.direct_io = true;
      else if (f == "--dit-blocks") options.block_path = moss_dit::parse_block_path(v);
      else if (f == "--quiet") quiet = true;
      else throw std::invalid_argument("unknown flag " + f);
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "moss-infer generate: " << e.what() << "\n" << kUsage;
    return 2;
  } catch (const std::out_of_range&) {
    std::cerr << "moss-infer generate: numeric flag out of range\n" << kUsage;
    return 2;
  }
  if ((params.empty() && !have_prompt) || req.model.empty() || (out_png.empty() && out_latent.empty())) {
    std::cerr << kUsage;
    return 2;
  }
  req.decode = !out_png.empty();
  pipeline::validate(req);
  for (const std::string& p : then_prompts) {
    pipeline::Request f = req;
    f.prompt = p;
    pipeline::validate(f);
  }

  std::signal(SIGINT, on_sigint);
  pipeline::Pipeline pipe(options);
  pipeline::GenerateControl control;
  control.cancel = &g_cancel;
  control.on_plan = [&](const planner::ExecutionPlan& plan) {
    if (!quiet) std::fprintf(stderr, "%s\n", plan.explanation.c_str());
  };
  control.on_step = [&](const pipeline::StepInfo& s) {
    if (!quiet) std::fprintf(stderr, "\rstep %d/%d  %.2f s/step ", s.step, s.num_steps, s.step_seconds);
    return true;
  };
  const auto run = [&](const pipeline::Request& q, const fs::path& png) {
    pipeline::Result r = pipe.generate(q, control);
    if (!quiet) std::fprintf(stderr, "\n");
    if (r.cancelled) return r;
    if (!png.empty()) pipeline::write_png_rgb8(png, r.rgb, r.width, r.height);
    if (!quiet) print_summary(r);
    return r;
  };
  const pipeline::Result r = run(req, out_png);
  if (r.cancelled) {
    std::signal(SIGINT, SIG_DFL);
    std::cerr << "moss-infer generate: cancelled after " << r.steps_run << " of " << r.steps_scheduled << " steps\n";
    return 130;
  }
  if (!out_latent.empty()) write_npy_f32(out_latent, r.latents, {moss_dit::kLatentChannels, r.height / 16, r.width / 16});
  std::string doc = "{\n" + run_json(req, r, util::path_utf8(out_png), util::path_utf8(out_latent));
  std::string followups;
  bool cancelled = false;
  for (std::size_t i = 0; i < then_prompts.size() && !cancelled; ++i) {
    pipeline::Request q = req;
    q.prompt = then_prompts[i];
    const fs::path png = out_png.empty() ? fs::path() : followup_path(out_png, i + 1);
    const pipeline::Result fr = run(q, png);
    cancelled = fr.cancelled;
    followups += std::string(i ? ",\n" : "") + "{\n" + run_json(q, fr, util::path_utf8(png), "") + "\n}";
  }
  std::signal(SIGINT, SIG_DFL);
  if (!then_prompts.empty()) doc += ",\n \"followups\": [" + followups + "]";
  doc += "\n}\n";
  if (!report.empty()) write_text_file(report, doc);
  if (cancelled) {
    std::cerr << "moss-infer generate: cancelled during a follow-up image\n";
    return 130;
  }
  return 0;
}

}  // namespace mi::cli
