// `moss-infer generate`: prompt or parameters.json -> PNG (and optionally the final latents and a JSON report).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/moss_dit/distill.h"
#include "pipeline/pipeline.h"
#include "pipeline/samplers/sampler_driver.h"
#include "pipeline/png_writer.h"
#include "pipeline/report_json.h"
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
    "                           [--slg-scale X] [--slg-start X] [--slg-stop X] [--max-seq N] [--sampler <name>]\n"
    "                           [--scheduler moss|simple|normal|karras|exponential|beta|sgm_uniform] [--flow-multiplier X]\n"
    "                           [--noise-layout <layout>] [--noise-source philox|philox_device|brownian_tree]\n"
    "                           [--eta X] [--s-noise X] [--dataspace <file>] [--no-metadata]\n"
    "                           [--mode min_memory|balanced|max_speed] [--vram-budget GiB] [--vram-reserve MiB]\n"
    "                           [--host-ram-budget GiB] [--cache-dir <dir>] [--then-prompt <text>]...\n"
    "                           [--direct-io | --buffered-io] [--te-host-pin auto|on|off] [--cert-dir <dir>]...\n"
    "                           [--dit-blocks op_by_op|fused|fused_graphs] [--dit <file>] [--te <file>] [--vae <file>]\n"
    "                           [--tier reference|high|turbo|auto] [--attention auto|exact|q8|q8_fp8|q8_fp16|...]\n"
    "                           [--preview <preview.png> [--preview-every N]] [--lora <file> [--lora-scale S]]...\n"
    "                           [--lora-strict] [--distill <dir>] [--time-embedding <file>] [--quiet]\n"
    "                           [--init-image <png|jpg> [--strength X] [--mask <png|jpg>] [--resize crop|fit|stretch]]\n"
    "                           [--hires <W>x<H>|<scale> [--hires-strength X] [--hires-steps N]]\n"
    "                           [--cfg-truncation X] [--step-cache off|fbcache] [--step-cache-threshold X]\n"
    "                           [--control <file> [--control-scale S] [--control-image <kind>=<png|jpg>\n"
    "                            [--control-strength X] [--control-start X] [--control-end X]]...]\n"
    "Flags override values read from --params. At least one of -o / --latent-out is required;\n"
    "without -o the VAE decode is skipped. --vram-budget caps the process's GPU memory (CUDA context included).\n"
    "Each --then-prompt generates one more image in the same engine session (same settings, warm models and\n"
    "embedding cache), written to <out>.<n>.png and reported under \"followups\".\n"
    "--dit-blocks (default fused, or the MOSSINFER_DIT_BLOCKS environment variable) picks the DiT block implementation;\n"
    "all give the same image. fused falls back to op_by_op when its startup self-check fails; fused_graphs adds CUDA graphs.\n"
    "--dit loads the DiT from one safetensors file (INT8 ConvRot, INT4 ConvRot or bf16, ComfyUI layout) instead of the\n"
    "bundle's;\n"
    "--te loads an INT8 ConvRot / W4A8 text encoder file, --vae a FLUX.2 VAE file (e.g. the full decoder); the tokenizer\n"
    "and configs always come from --model. --preview rewrites a latent preview PNG (no VAE) every N steps.\n"
    "--tier (default auto): reference = bf16 exact; high = the loaded DiT's GEMMs + 8-bit attention; turbo = an INT4\n"
    "ConvRot DiT (--dit) + 8-bit attention; auto = the fastest certified configuration of the loaded DiT (else\n"
    "reference; an INT8 DiT then runs with exact attention, uncertified; an uncertified INT4 DiT needs --tier turbo).\n"
    "Certificates are searched in each --cert-dir, then <cache-dir>/quality_certs/, <dit>.mossinfer-cert.json and the\n"
    "install tree's share/mossinfer/quality_certs/ (the first usable one wins).\n"
    "The DiT is loaded with direct I/O (O_DIRECT / FILE_FLAG_NO_BUFFERING) where the file system supports it and its\n"
    "files are not already in the page cache, else buffered; from the page cache, with whichever mode this machine\n"
    "measured faster (the report's \"dit_load\" says which and why);\n"
    "--buffered-io forces buffered DiT loads, --direct-io direct I/O for every weight read (cold-cache behaviour).\n"
    "--te-host-pin (default auto) decides whether a text encoder kept in RAM may be pinned: auto where the machine has\n"
    "clearly the room, on whenever it fits, off never.\n"
    "--attention overrides the tier's attention kernel: exact (FlashAttention-2) or an 8-bit variant q8_fp8 (= q8),\n"
    "q8_fp16, q8_fp8_f32acc, q8_fp16_f32acc (lossy).\n"
    "--init-image (PNG or JPEG, EXIF orientation applied) runs img2img: the image is brought to the output size with\n"
    "--resize (default crop: cover and center-crop; fit: fit inside, edge-filled bands; stretch) and the last\n"
    "--strength (default 0.75) of the schedule is sampled from it; strength 1 is txt2img, 0 returns the VAE round trip.\n"
    "Without a size (--size/--width/--height, or height/width in --params) the output size follows the init image\n"
    "(rounded down to multiples of 16, within 4.2 megapixels).\n"
    "--mask (white = regenerate, black = keep; the init image's size) inpaints by latent blending with the base model.\n"
    "--hires generates at the base size, then upscales (Lanczos, pixel space) to <W>x<H> or <scale> x the base size\n"
    "(rounded to multiples of 16) and refines with --hires-strength (default 0.45) over --hires-steps (default --steps).\n"
    "--params may also carry \"init_image\" and \"mask\" (paths, relative to the parameters file), \"denoise_strength\",\n"
    "\"resize\", \"hires\" (\"<W>x<H>\" or a scale), \"hires_strength\" and \"hires_steps\"; flags override them.\n"
    "--distill <dir> applies a distill bundle (lodestones' data-space DMD repos): its time_embedding.safetensors (as\n"
    "--time-embedding), lora.safetensors (as --lora, strength 1; a following --lora-scale sets it) and\n"
    "dataspace.safetensors (--sampler dataspace --dataspace <file>), with --cfg 1 and --steps = the file's step codes\n"
    "unless given. --time-embedding replaces the DiT's step embedder and modulation biases, computed in fp32.\n"
    "Lossy accelerators (off by default; a run with any of them is uncertified): --cfg-truncation X (0..1, default 1 =\n"
    "off) drops the negative branch for the steps from X * steps on, counting the steps each pass runs (img2img and the\n"
    "hires pass: their strength-cropped steps); multistep samplers carry one guided step into the first unguided one.\n"
    "--step-cache fbcache reuses the DiT's blocks 1..41 from the last full DiT call while block 0's output residual changed\n"
    "by less than --step-cache-threshold (relative L1; default 0.02 on the base model, 0.03 on the SFT model and its\n"
    "derived files; 0.05-0.08 are faster but may change the composition, the base model's from 0.03).\n"
    "--params may carry \"cfg_truncation\", \"step_cache\" and \"step_cache_threshold\".\n"
    "--control loads a control adapter (a LoRA with a widened DiT input embedder, e.g. the MOSS-Image color/depth\n"
    "control); --control-scale sets its LoRA strength (default 1). Each --control-image gives one input of the\n"
    "adapter's kinds (e.g. color=map.png, depth=depth.png: near = white), brought to the output size with --resize and\n"
    "VAE-encoded; the following --control-strength (default 1, 0..10) scales it and --control-start / --control-end\n"
    "(default 0 / 1) limit it to the steps i with start <= i / steps < end (e.g. 0.5 for 3D-render inputs). A kind\n"
    "without an image gets no control.\n";

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

constexpr double kDefaultStrength = 0.75;

// A JSON number as --hires accepts it ("2", "1.5").
std::string json_number_text(const json::Value& v) {
  if (v.is_int()) return std::to_string(v.as_int());
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << v.as_double();
  return os.str();
}

// The output size for an init image without --size: its own, scaled down to fit kMaxPixels and kMaxSide, rounded down
// to multiples of 16 (at least 16).
void fit_size_to(const pipeline::Image& im, int& width, int& height) {
  double scale = 1.0;
  const double pixels = static_cast<double>(im.width) * im.height;
  if (pixels > static_cast<double>(pipeline::kMaxPixels)) scale = std::sqrt(static_cast<double>(pipeline::kMaxPixels) / pixels);
  scale = std::min(scale, static_cast<double>(pipeline::kMaxSide) / std::max(im.width, im.height));
  width = std::max(16, static_cast<int>(std::floor(im.width * scale / 16)) * 16);
  height = std::max(16, static_cast<int>(std::floor(im.height * scale / 16)) * 16);
}

// --hires <W>x<H> (the target size) or <scale> (times the base size, rounded to multiples of 16).
void parse_hires(const std::string& v, pipeline::Request& req) {
  const std::size_t x = v.find('x');
  if (x != std::string::npos) {
    req.hires.width = to_int32("--hires", v.substr(0, x));
    req.hires.height = to_int32("--hires", v.substr(x + 1));
    return;
  }
  const double scale = to_double("--hires", v);
  if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("--hires: the scale must be positive");
  req.hires.width = std::max(16, static_cast<int>(std::lround(req.width * scale / 16)) * 16);
  req.hires.height = std::max(16, static_cast<int>(std::lround(req.height * scale / 16)) * 16);
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
  std::fprintf(stderr, "tier %s (requested %s): DiT %s, %s attention, text encoder %s%s\n",
               planner::tier_name(r.tier.effective), planner::tier_name(r.tier.requested), r.dit_linears.c_str(),
               planner::attention_name(r.tier.attention), r.te_precision.c_str(), r.tier.certified ? "" : ", uncertified");
  if (!r.block_path_note.empty()) std::fprintf(stderr, "%s\n", r.block_path_note.c_str());
  if (r.first_pass.encoded || r.hires.ran) {
    std::fprintf(stderr, "%s: %d of %d steps from step %d", r.image_mode.c_str(), r.first_pass.steps_run,
                 r.first_pass.schedule_steps - r.first_pass.start_step, r.first_pass.start_step);
    if (r.hires.ran) {
      std::fprintf(stderr, "; hires %dx%d: %d of %d steps from step %d", r.hires.width, r.hires.height, r.hires.steps_run,
                   r.hires.schedule_steps - r.hires.start_step, r.hires.start_step);
    }
    std::fprintf(stderr, "; VAE encode %.2f s (load %.2f s), image ops %.2f s\n", r.timings.encode_s, r.timings.vae_encoder_load_s,
                 r.timings.image_ops_s);
  }
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
  fs::path params, out_png, out_latent, report, preview_png;
  int preview_every = 1;
  std::vector<std::string> then_prompts;
  bool have_prompt = false, quiet = false, size_given = false;
  fs::path init_image, mask;
  std::vector<fs::path> control_images;  // per --control-image, in order
  std::optional<double> strength;
  std::string hires;
  bool distill = false, distill_dataspace = false;
  json::Value pdoc;
  const json::Value* pobj = nullptr;  // the parameters object of --params
  int distill_steps = 0;
  // Flags apply after --params is read, in command-line order.
  std::vector<std::pair<std::string, std::string>> flags;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--direct-io" || a == "--buffered-io" || a == "--quiet" || a == "--no-metadata" || a == "--lora-strict") {
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
  if (!params.empty()) {
    pdoc = json::parse(read_file(params));
    req = pipeline::request_from_parameters(pdoc);
    pobj = pdoc.find("parameters") != nullptr ? pdoc.find("parameters") : &pdoc;
  }
  // A key the parameters file sets: front-end defaults never override it.
  const auto in_params = [&](const char* key) { return pobj != nullptr && pobj->find(key) != nullptr; };
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
        size_given = true;
        const std::size_t x = v.find('x');
        if (x == std::string::npos) throw std::invalid_argument("--size: expected <W>x<H>");
        req.width = to_int32(f, v.substr(0, x));
        req.height = to_int32(f, v.substr(x + 1));
      }
      else if (f == "--width") req.width = to_int32(f, v), size_given = true;
      else if (f == "--height") req.height = to_int32(f, v), size_given = true;
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
      else if (f == "--scheduler") req.scheduler = pipeline::parse_scheduler(v);
      else if (f == "--flow-multiplier") req.flow_multiplier = to_double(f, v);
      else if (f == "--noise-layout") req.noise_layout = pipeline::parse_noise_layout(v);
      else if (f == "--noise-source") req.noise_source = pipeline::parse_noise_source(v);
      else if (f == "--eta") req.eta = to_double(f, v);
      else if (f == "--s-noise") req.s_noise = to_double(f, v);
      else if (f == "--dataspace") req.dataspace_path = path_from_utf8(v);
      else if (f == "--no-metadata") req.save_metadata = false;
      else if (f == "--lora") req.loras.push_back({path_from_utf8(v), 1.0f});
      else if (f == "--lora-scale") {
        if (req.loras.empty()) throw std::invalid_argument("--lora-scale must follow a --lora");
        req.loras.back().strength = static_cast<float>(to_double(f, v));
      }
      else if (f == "--lora-strict") options.lora_strict = true;
      else if (f == "--control") req.control.adapter = path_from_utf8(v);
      else if (f == "--control-scale") req.control.lora_strength = static_cast<float>(to_double(f, v));
      else if (f == "--control-image") {
        const std::size_t eq = v.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == v.size()) throw std::invalid_argument("--control-image: expected <kind>=<png|jpg>");
        pipeline::Request::ControlImage c;
        c.kind = v.substr(0, eq);
        req.control.images.push_back(c);
        control_images.push_back(path_from_utf8(v.substr(eq + 1)));
      }
      else if (f == "--control-strength" || f == "--control-start" || f == "--control-end") {
        if (req.control.images.empty()) throw std::invalid_argument(f + " must follow a --control-image");
        pipeline::Request::ControlImage& c = req.control.images.back();
        (f == "--control-strength" ? c.strength : f == "--control-start" ? c.start : c.end) = to_double(f, v);
      }
      else if (f == "--time-embedding") options.time_embedding_file = path_from_utf8(v);
      else if (f == "--distill") {
        moss_dit::DistillBundle b;
        try {
          b = moss_dit::open_distill_bundle(path_from_utf8(v));
          if (!b.dataspace.empty()) {
            distill_steps = static_cast<int>(pipeline::load_dataspace(util::path_utf8(b.dataspace)).step_codes.size());
          }
        } catch (const std::runtime_error& e) {
          throw std::invalid_argument(std::string("--distill: ") + e.what());
        }
        distill = true;
        if (!b.time_embedding.empty()) options.time_embedding_file = b.time_embedding;
        if (!b.lora.empty()) req.loras.push_back({b.lora, 1.0f});
        if (!b.dataspace.empty()) {
          req.sampler = pipeline::SamplerKind::kDataSpace;
          distill_dataspace = true;
          req.dataspace_path = b.dataspace;
        }
      }
      else if (f == "--mode") options.mode = planner::parse_mode(v);
      else if (f == "--vram-budget") options.vram_budget = pipeline::vram_option_bytes(to_double(f, v), kGiB, f);
      else if (f == "--vram-reserve") options.vram_reserve = pipeline::vram_option_bytes(to_double(f, v), 1024.0 * 1024.0, f);
      else if (f == "--host-ram-budget") options.host_ram_budget = pipeline::vram_option_bytes(to_double(f, v), kGiB, f);
      else if (f == "--cache-dir") options.cache_dir = path_from_utf8(v);
      else if (f == "--then-prompt") then_prompts.push_back(v);
      else if (f == "--direct-io") options.direct_io = true;
      else if (f == "--buffered-io") options.dit_load_direct_io = false;
      else if (f == "--te-host-pin") options.te_host_pin = planner::parse_te_host_pin(v);
      else if (f == "--cert-dir") options.certificate_dirs.push_back(path_from_utf8(v));
      else if (f == "--dit-blocks") options.block_path = moss_dit::parse_block_path(v);
      else if (f == "--dit") options.dit_file = path_from_utf8(v);
      else if (f == "--te") options.text_encoder_file = path_from_utf8(v);
      else if (f == "--vae") options.vae_file = path_from_utf8(v);
      else if (f == "--preview") preview_png = path_from_utf8(v);
      else if (f == "--preview-every") preview_every = to_int32(f, v);
      else if (f == "--tier") options.tier = planner::parse_tier(v);
      else if (f == "--attention") options.attention = v == "auto" ? std::optional<planner::AttentionMode>() : planner::parse_attention(v);
      else if (f == "--init-image") init_image = path_from_utf8(v);
      else if (f == "--strength") strength = to_double(f, v);
      else if (f == "--mask") mask = path_from_utf8(v);
      else if (f == "--resize") req.resize = pipeline::parse_resize_mode(v);
      else if (f == "--hires") hires = v;
      else if (f == "--hires-strength") req.hires.strength = to_double(f, v);
      else if (f == "--hires-steps") req.hires.steps = to_int32(f, v);
      else if (f == "--cfg-truncation") req.accel.cfg_truncation = to_double(f, v);
      else if (f == "--step-cache") req.accel.step_cache = pipeline::accel::parse_step_cache(v);
      else if (f == "--step-cache-threshold") req.accel.step_cache_threshold = to_double(f, v);
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
  const auto given = [&](const char* flag) {
    return std::any_of(flags.begin(), flags.end(), [&](const auto& fv) { return fv.first == flag; });
  };
  if (given("--direct-io") && given("--buffered-io")) {
    std::cerr << "moss-infer generate: --direct-io and --buffered-io contradict each other\n" << kUsage;
    return 2;
  }
  if (distill) {
    // The bundle's defaults where the command line does not set them: no CFG, the trained number of steps.
    if (!given("--cfg") && !in_params("cfg")) req.cfg = 1.0;
    if (!given("--steps") && !in_params("steps") && distill_steps > 0) req.steps = distill_steps;
  }
  // Accelerator settings that do nothing (from flags or --params, after the distill defaults).
  if ((given("--step-cache-threshold") || in_params("step_cache_threshold")) && !req.accel.step_cache_on()) {
    std::fprintf(stderr, "warning: the step-cache threshold has no effect without --step-cache fbcache\n");
  }
  if (req.accel.cfg_truncation < 1.0 && req.cfg == 1.0) {
    std::fprintf(stderr, "warning: CFG truncation has no effect without CFG (cfg 1)\n");
  }
  if ((params.empty() && !have_prompt) || req.model.empty() || (out_png.empty() && out_latent.empty())) {
    std::cerr << kUsage;
    return 2;
  }
  try {
    // Image inputs from --params (paths relative to the parameters file), unless a flag gave them.
    const auto param_string = [&](const char* key) -> std::string {
      const json::Value* v = pobj != nullptr ? pobj->find(key) : nullptr;
      if (v == nullptr || v->is_null()) return {};
      if (v->is_number()) return json_number_text(*v);
      if (!v->is_string()) throw std::invalid_argument(std::string("--params: ") + key + " must be a string");
      return v->as_string();
    };
    const auto param_path = [&](const char* key) {
      const std::string v = param_string(key);
      if (v.empty()) return fs::path();
      const fs::path p = path_from_utf8(v);
      return p.is_absolute() ? p : params.parent_path() / p;
    };
    if (init_image.empty()) init_image = param_path("init_image");
    if (mask.empty()) mask = param_path("mask");
    if (hires.empty()) hires = param_string("hires");
    if (distill_dataspace && req.sampler == pipeline::SamplerKind::kDataSpace && (!init_image.empty() || !hires.empty())) {
      throw std::invalid_argument("--distill selected the bundle's dataspace sampler, which cannot start from an image "
                                  "(--init-image, --hires); pass --sampler after --distill to use another sampler");
    }
    if (!init_image.empty()) {
      req.init_image = std::make_shared<const pipeline::Image>(pipeline::load_image(init_image, 3));
      if (strength) req.denoise_strength = *strength;
      else if (!in_params("denoise_strength")) req.denoise_strength = kDefaultStrength;
      if (!size_given && !in_params("height") && !in_params("width")) fit_size_to(*req.init_image, req.width, req.height);
    } else if (strength) {
      throw std::invalid_argument("--strength needs --init-image");
    }
    if (!mask.empty()) {
      if (init_image.empty()) throw std::invalid_argument("--mask needs --init-image");
      req.mask = std::make_shared<const pipeline::Image>(pipeline::load_image(mask, 1));
    }
    if (!hires.empty()) parse_hires(hires, req);
    for (std::size_t i = 0; i < control_images.size(); ++i) {
      req.control.images[i].image = std::make_shared<const pipeline::Image>(pipeline::load_image(control_images[i], 3));
    }
  } catch (const std::invalid_argument& e) {
    std::cerr << "moss-infer generate: " << e.what() << "\n" << kUsage;
    return 2;
  } catch (const std::out_of_range&) {
    std::cerr << "moss-infer generate: --hires out of range\n" << kUsage;
    return 2;
  } catch (const std::runtime_error& e) {
    std::cerr << "moss-infer generate: " << e.what() << "\n";
    return 1;
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
  if (!preview_png.empty()) {
    control.preview_every = preview_every;
    control.on_preview = [&](const pipeline::PreviewImage& p) {
      pipeline::write_png_rgb8(preview_png, std::span<const std::uint8_t>(p.rgb, static_cast<std::size_t>(p.width) * p.height * 3),
                               p.width, p.height);
    };
  }
  control.on_step = [&](const pipeline::StepInfo& s) {
    if (!quiet) std::fprintf(stderr, "\rstep %d/%d  %.2f s/step ", s.step, s.num_steps, s.step_seconds);
    return true;
  };
  const auto run = [&](const pipeline::Request& q, const fs::path& png) {
    pipeline::Result r = pipe.generate(q, control);
    if (!quiet) std::fprintf(stderr, "\n");
    if (r.cancelled) return r;
    // Lossy and uncertified: always said, --quiet or not.
    if (r.tier.effective != planner::Tier::kReference && !r.tier.certified) {
      std::fprintf(stderr, "warning: uncertified %s-tier run (%s): %s\n", planner::tier_name(r.tier.effective),
                   r.dit_linears.c_str(), r.tier.reason.c_str());
    }
    if (!png.empty()) {
      const std::vector<pipeline::PngText> text =
          q.save_metadata ? pipeline::generation_text_chunks(pipeline::generation_metadata(q, r)) : std::vector<pipeline::PngText>{};
      pipeline::write_png_rgb8(png, r.rgb, r.width, r.height, text);
    }
    for (const std::string& w : r.lora_warnings) std::fprintf(stderr, "warning: LoRA %s\n", w.c_str());
    if (!r.control.adapter.empty() && !quiet) {
      std::fprintf(stderr, "control: %s (%d LoRA adapters: %d side branch, %d merged), %d image(s) encoded\n",
                   r.control.lora.file.c_str(), r.control.lora.adapters, r.control.lora.side_branch, r.control.lora.merged,
                   r.control.encodes);
    }
    if (q.accel.active(q.cfg != 1.0)) {
      std::fprintf(stderr, "warning: lossy accelerators (%s), uncertified: CFG truncated on %d steps, step cache reused %d of %d DiT calls\n",
                   q.accel.describe(q.cfg != 1.0).c_str(), r.accel.cfg_truncated_steps, r.accel.cached_calls, r.accel.dit_calls);
    }
    for (const std::string& w : r.accel_warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    if (!r.image_note.empty()) std::fprintf(stderr, "note: %s\n", r.image_note.c_str());
    for (const auto* t : {&r.cond_token_report, &r.uncond_token_report}) {
      if (t->truncated) {
        std::fprintf(stderr, "warning: %s truncated to %zu of %zu tokens (%zu dropped; --max-seq %zu)\n",
                     t == &r.cond_token_report ? "prompt" : "negative prompt", t->kept_tokens, t->total_tokens,
                     t->dropped_tokens(), t->max_length);
      }
    }
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
  std::string doc = "{\n" + pipeline::report_fields(req, r, util::path_utf8(out_png), util::path_utf8(out_latent));
  std::string followups;
  bool cancelled = false;
  for (std::size_t i = 0; i < then_prompts.size() && !cancelled; ++i) {
    pipeline::Request q = req;
    q.prompt = then_prompts[i];
    const fs::path png = out_png.empty() ? fs::path() : followup_path(out_png, i + 1);
    const pipeline::Result fr = run(q, png);
    cancelled = fr.cancelled;
    followups += std::string(i ? ",\n" : "") + "{\n" + pipeline::report_fields(q, fr, util::path_utf8(png), "") + "\n}";
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
