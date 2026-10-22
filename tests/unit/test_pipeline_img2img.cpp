// img2img, inpainting and the hires fix end to end (P7b) with the real weights:
//  - strength 1 is txt2img bit for bit; strength 0.5 runs and is deterministic (square and non-square);
//  - strength 0 is the VAE round trip without the DiT (not even planned); an all-keep mask ends exactly at the init
//    image's encode, a half mask keeps its half;
//  - the hires fix fits an 8 GiB budget, loads the DiT once under tight budgets, counts steps and previews across its
//    passes, keeps an inpainting mask's region at the target size, and cancels between its passes and during its encode;
//  - img2img with a runtime LoRA.
// GPU (suite prefix Gpu*); skipped without the weights or the reference image.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"
#include "planner/planner.h"

namespace {

namespace fs = std::filesystem;
namespace pl = mi::pipeline;

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
fs::path init_image_file() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "reference_images" / "asset005_base.png"; }
bool have_inputs() {
  return fs::is_regular_file(weights_root() / "base" / "moss.mossidx") && fs::is_regular_file(init_image_file());
}
const fs::path kPeftLora = "/workspace/Models/community/axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors";

std::shared_ptr<const pl::Image> init_image() {
  static const auto im = std::make_shared<const pl::Image>(pl::load_image(init_image_file()));
  return im;
}

std::size_t bit_mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return n;
}

double mean_abs_diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) s += std::abs(int{a[i]} - int{b[i]});
  return s / static_cast<double>(a.size());
}

double psnr(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double se = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    se += d * d;
  }
  const double mse = se / static_cast<double>(a.size());
  return mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99.0;
}

pl::Request base_request(int width, int height, int steps, const char* bundle = "base") {
  pl::Request r;
  r.model = weights_root() / bundle;
  r.prompt = "a watercolor painting of a harbor town at sunset";
  r.negative_prompt = "blurry";
  r.width = width;
  r.height = height;
  r.steps = steps;
  r.seed = 1234;
  r.decode = false;
  return r;
}

std::shared_ptr<pl::Image> solid_mask(const pl::Image& like, std::uint8_t v) {
  auto m = std::make_shared<pl::Image>();
  m->width = like.width;
  m->height = like.height;
  m->channels = 1;
  m->pixels.assign(static_cast<std::size_t>(like.width) * like.height, v);
  return m;
}

void print_timings(const char* what, const pl::Result& r) {
  const auto& t = r.timings;
  const std::string streamed = r.dit_stream.streamed ? ", DiT streamed (" + std::to_string(r.dit_stream.resident_blocks) + " resident)" : "";
  std::printf("[img2img] %s: %dx%d, %d/%d steps, total %.2f s (TE %.2f, DiT load %.2f x%d, encoder load %.2f, encode %.3f, "
              "image ops %.3f, denoise %.2f, decode %.2f); arena %.2f GiB, high-water %.2f GiB, plan device peak %.2f GiB%s%s\n",
              what, r.width, r.height, r.steps_run, r.steps_scheduled, t.total_s, t.text_encode_s, t.dit_load_s, r.memory.dit_loads,
              t.vae_encoder_load_s, t.encode_s, t.image_ops_s, t.denoise_s, t.decode_s,
              static_cast<double>(r.memory.arena_capacity) / (1 << 30), static_cast<double>(r.memory.arena_high_water) / (1 << 30),
              static_cast<double>(r.plan.device_peak_bytes) / (1 << 30), streamed.c_str(),
              r.memory.dit_parked_for_decode ? ", DiT parked for the final decode" : "");
}

// Strength 1 without a mask is txt2img (nothing encoded, reported as such); with an all-regenerate mask the per-step
// blend keeps x as is ((1 - 1) * proper + 1 * x), so it is txt2img bit for bit too.
TEST(GpuPipelineImg2Img, StrengthOneIsTxt2ImgBitExact) {
  if (!have_inputs()) GTEST_SKIP() << "weights or reference image not present";
  pl::Pipeline pipe;
  const pl::Request txt = base_request(1024, 1024, 4);
  const pl::Result a = pipe.generate(txt);
  print_timings("txt2img", a);
  pl::Request img = txt;
  img.init_image = init_image();
  img.denoise_strength = 1.0;
  const pl::Result b = pipe.generate(img);
  print_timings("img2img strength 1", b);
  EXPECT_EQ(b.image_mode, "txt2img");
  EXPECT_FALSE(b.image_note.empty());
  EXPECT_FALSE(b.first_pass.encoded);
  EXPECT_EQ(b.first_pass.start_step, 0);
  EXPECT_EQ(b.steps_run, a.steps_run);
  EXPECT_EQ(bit_mismatches(b.latents, a.latents), 0u);
  img.mask = solid_mask(*img.init_image, 255);
  const pl::Result c = pipe.generate(img);
  print_timings("inpaint strength 1, all regenerate", c);
  EXPECT_EQ(c.image_mode, "inpaint");
  EXPECT_TRUE(c.image_note.empty());
  EXPECT_TRUE(c.first_pass.encoded);
  EXPECT_EQ(bit_mismatches(c.latents, a.latents), 0u);
}

// Strength 0.5: half the schedule from the encoded image, deterministic run to run, at 1024^2 and a non-square size,
// and closer to the init image than txt2img is.
TEST(GpuPipelineImg2Img, HalfStrengthRunsAndIsDeterministic) {
  if (!have_inputs()) GTEST_SKIP() << "weights or reference image not present";
  pl::Pipeline pipe;
  for (const auto& [w, h] : {std::pair{1024, 1024}, std::pair{1216, 832}}) {
    pl::Request req = base_request(w, h, 8);
    req.decode = true;
    req.init_image = init_image();
    req.denoise_strength = 0.5;
    const pl::Result r1 = pipe.generate(req);
    print_timings("img2img strength 0.5", r1);
    const pl::Result r2 = pipe.generate(req);
    EXPECT_FALSE(r1.cancelled);
    EXPECT_EQ(r1.image_mode, "img2img");
    EXPECT_EQ(r1.width, w);
    EXPECT_EQ(r1.height, h);
    EXPECT_EQ(r1.first_pass.schedule_steps, 8);
    EXPECT_EQ(r1.first_pass.start_step, 4);
    EXPECT_EQ(r1.steps_run, 4);
    EXPECT_EQ(r1.steps_scheduled, 4);
    EXPECT_TRUE(r1.first_pass.encoded);
    EXPECT_FALSE(r1.vae_encoder_source.empty());
    ASSERT_EQ(r1.rgb.size(), static_cast<std::size_t>(w) * h * 3);
    EXPECT_EQ(bit_mismatches(r1.latents, r2.latents), 0u) << w << "x" << h;
    EXPECT_EQ(r1.rgb, r2.rgb) << w << "x" << h;
    EXPECT_EQ(r2.timings.vae_encoder_load_s, 0.0);  // the encoder stays resident
    pl::Request txt = req;
    txt.init_image = nullptr;
    txt.denoise_strength = 1.0;
    const pl::Result t = pipe.generate(txt);
    const pl::Image sized = pl::resize_to(*init_image(), w, h, pl::ResizeMode::kCrop);
    const double d_img = mean_abs_diff(r1.rgb, sized.pixels), d_txt = mean_abs_diff(t.rgb, sized.pixels);
    std::printf("[img2img] %dx%d strength 0.5: mean |output - init| %.2f vs txt2img %.2f\n", w, h, d_img, d_txt);
    EXPECT_LT(d_img, d_txt);
  }
}

// Strength 0 runs no step: no text encoder, no DiT (not even planned); the latents are the init image's encode, the
// image its VAE round trip. An all-keep mask ends there exactly at any strength; a half mask keeps its half exactly.
TEST(GpuPipelineImg2Img, InpaintKeepsTheRoundTripOutsideTheMask) {
  if (!have_inputs()) GTEST_SKIP() << "weights or reference image not present";
  pl::Pipeline pipe;
  pl::Request req = base_request(1024, 1024, 6);
  req.init_image = init_image();
  req.denoise_strength = 0.0;
  req.decode = true;
  const pl::Result z = pipe.generate(req);
  print_timings("strength 0 (VAE round trip)", z);
  EXPECT_EQ(z.steps_run, 0);
  EXPECT_EQ(z.first_pass.start_step, 6);
  EXPECT_EQ(z.timings.text_encode_s, 0.0);
  EXPECT_EQ(z.timings.dit_load_s, 0.0);
  EXPECT_EQ(z.memory.dit_loads, 0);
  EXPECT_EQ(z.timings.denoise_s, 0.0);
  EXPECT_LT(z.plan.arena_bytes, std::size_t{4} << 30) << "the DiT must not be planned without a sampling step";
  const pl::Image sized = pl::resize_to(*req.init_image, 1024, 1024, pl::ResizeMode::kCrop);
  const double round_trip_psnr = psnr(z.rgb, sized.pixels);
  std::printf("[img2img] VAE round trip PSNR %.2f dB, strength-0 arena %.2f GiB\n", round_trip_psnr,
              static_cast<double>(z.plan.arena_bytes) / (1 << 30));
  EXPECT_GT(round_trip_psnr, 25.0);

  req.decode = false;
  req.denoise_strength = 0.6;
  req.mask = solid_mask(*req.init_image, 0);
  const pl::Result keep = pipe.generate(req);
  print_timings("inpaint, all keep", keep);
  EXPECT_EQ(keep.steps_run, 4);  // int(6 - 3.6) = 2 skipped
  EXPECT_EQ(bit_mismatches(keep.latents, z.latents), 0u);

  // Left half regenerate (mask 255 for x < 768 of the 1536-wide init image: the crop keeps x in [256, 1280), so
  // latent columns 0..31 regenerate and 32..63 keep; column 31/32's 2x2 sample stays on one side of the edge).
  auto half = solid_mask(*req.init_image, 0);
  for (int y = 0; y < half->height; ++y) {
    for (int x = 0; x < half->width / 2; ++x) half->pixels[static_cast<std::size_t>(y) * half->width + x] = 255;
  }
  req.mask = half;
  const pl::Result part = pipe.generate(req);
  const std::vector<float> lm = pl::latent_mask(pl::resize_to(*half, 1024, 1024, pl::ResizeMode::kCrop), 64, 64);
  std::size_t kept_bad = 0, kept = 0, changed = 0;
  for (std::size_t c = 0; c < 128; ++c) {
    for (std::size_t i = 0; i < 64 * 64; ++i) {
      const std::size_t k = c * 64 * 64 + i;
      if (lm[i] == 0.0f) {
        ++kept;
        kept_bad += std::memcmp(&part.latents[k], &z.latents[k], sizeof(float)) != 0;
      } else if (lm[i] == 1.0f) {
        changed += part.latents[k] != z.latents[k];
      }
    }
  }
  std::printf("[img2img] half mask: %zu kept latents (%zu differ), %zu regenerated latents changed\n", kept, kept_bad, changed);
  EXPECT_EQ(kept, std::size_t{128} * 64 * 32);
  EXPECT_EQ(kept_bad, 0u);
  EXPECT_GT(changed, std::size_t{128} * 64 * 32 * 9 / 10);
}

// The hires fix 512^2 -> 1024^2 under an 8 GiB process budget: both passes run with the DiT loaded once, the plan and
// the arena stay within the budget, and steps and previews count on across the passes.
TEST(GpuPipelineImg2Img, HiresFixFitsAnEightGiBBudget) {
  if (!have_inputs()) GTEST_SKIP() << "weights or reference image not present";
  pl::PipelineOptions options;
  options.vram_budget = std::size_t{8} << 30;
  pl::Pipeline pipe(options);
  pl::Request req = base_request(512, 512, 4);
  req.decode = true;
  req.hires.width = 1024;
  req.hires.height = 1024;
  req.hires.strength = 0.5;
  req.hires.steps = 4;
  std::vector<std::pair<int, int>> progress;
  std::vector<std::tuple<int, int, int>> previews;  // step, num_steps, width
  pl::GenerateControl control;
  control.on_step = [&](const pl::StepInfo& s) {
    progress.emplace_back(s.step, s.num_steps);
    return true;
  };
  control.on_preview = [&](const pl::PreviewImage& p) { previews.emplace_back(p.step, p.num_steps, p.width); };
  const pl::Result r = pipe.generate(req, control);
  print_timings("hires 512 -> 1024 at 8 GiB", r);
  ASSERT_FALSE(r.cancelled);
  EXPECT_EQ(r.width, 1024);
  EXPECT_EQ(r.height, 1024);
  EXPECT_EQ(r.rgb.size(), std::size_t{1024} * 1024 * 3);
  EXPECT_EQ(r.latents.size(), std::size_t{128} * 64 * 64);
  EXPECT_TRUE(r.first_pass.ran);
  EXPECT_FALSE(r.first_pass.encoded);
  EXPECT_FALSE(r.first_pass.decode_schedule.empty());
  EXPECT_FALSE(r.decode_schedule.empty());
  EXPECT_TRUE(r.hires.ran);
  EXPECT_TRUE(r.hires.encoded);
  EXPECT_EQ(r.hires.start_step, 2);
  EXPECT_EQ(r.steps_run, 6);
  EXPECT_EQ(r.steps_scheduled, 6);
  ASSERT_EQ(progress.size(), 6u);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(progress[static_cast<std::size_t>(i)], (std::pair{i + 1, 6}));
  ASSERT_EQ(previews.size(), 6u);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(previews[static_cast<std::size_t>(i)], (std::tuple{i + 1, 6, i < 4 ? 512 : 1024})) << i;
  }
  EXPECT_LE(r.plan.device_peak_bytes, options.vram_budget);
  EXPECT_LE(r.memory.arena_capacity, options.vram_budget);
  EXPECT_LE(r.memory.arena_high_water, r.memory.arena_capacity);
  EXPECT_EQ(r.memory.dit_loads, 1);
  EXPECT_FALSE(r.memory.dit_reloaded);
  std::printf("[img2img] first-pass decode: %s; final decode: %s\n", r.first_pass.decode_schedule.c_str(), r.decode_schedule.c_str());
  std::printf("[img2img] hires plan: %s\n", r.plan.explanation.c_str());
}

// Tight budgets (min_memory at 6 GiB, balanced at 10.5 GiB: a plan that may park the DiT for the decode): the first
// pass is decoded next to the DiT without parking it, so the DiT is loaded exactly once per request.
TEST(GpuPipelineImg2Img, HiresLoadsTheDitOnceUnderTightBudgets) {
  if (!have_inputs()) GTEST_SKIP() << "weights or reference image not present";
  for (const auto& [mode, gib] : {std::pair{mi::planner::Mode::kMinMemory, 6.0}, std::pair{mi::planner::Mode::kBalanced, 10.5}}) {
    pl::PipelineOptions options;
    options.mode = mode;
    options.vram_budget = static_cast<std::size_t>(gib * static_cast<double>(std::size_t{1} << 30));
    pl::Pipeline pipe(options);
    pl::Request req = base_request(512, 512, 2);
    req.decode = true;
    req.hires.width = 1024;
    req.hires.height = 1024;
    req.hires.strength = 0.5;
    req.hires.steps = 2;
    const pl::Result r = pipe.generate(req);
    const std::string what = std::string(mi::planner::mode_name(mode)) + " at " + std::to_string(gib) + " GiB";
    print_timings(what.c_str(), r);
    std::printf("[img2img] %s: plan DiT %s, park for decode %s; first-pass decode %s\n", what.c_str(),
                r.plan.dit.residency == mi::planner::DitResidency::kStreamed  ? "streamed"
                : r.plan.dit.residency == mi::planner::DitResidency::kParked ? "parked"
                                                                               : "resident",
                r.plan.dit.park_for_decode ? "yes" : "no", r.first_pass.decode_schedule.c_str());
    ASSERT_FALSE(r.cancelled) << what;
    EXPECT_EQ(r.steps_run, 3) << what;
    EXPECT_EQ(r.memory.dit_loads, 1) << what;
    EXPECT_FALSE(r.memory.dit_reloaded) << what;
    EXPECT_LE(r.memory.arena_capacity, options.vram_budget) << what;
    EXPECT_EQ(r.rgb.size(), std::size_t{1024} * 1024 * 3) << what;
  }
}

// Hires fix + inpainting: the kept region ends at the init image's encode at the target size, exactly.
TEST(GpuPipelineImg2Img, HiresInpaintKeepsTheInitImageAtTheTarget) {
  if (!have_inputs()) GTEST_SKIP() << "weights or reference image not present";
  pl::Pipeline pipe;
  pl::Request target = base_request(1024, 1024, 4);
  target.init_image = init_image();
  target.denoise_strength = 0.0;
  const pl::Result z = pipe.generate(target);  // z0 at 1024^2
  pl::Request req = base_request(512, 512, 4);
  req.init_image = init_image();
  req.denoise_strength = 0.8;
  req.mask = solid_mask(*req.init_image, 0);
  req.hires.width = 1024;
  req.hires.height = 1024;
  req.hires.strength = 0.5;
  const pl::Result r = pipe.generate(req);
  print_timings("hires + inpaint (all keep)", r);
  EXPECT_EQ(r.image_mode, "inpaint");
  EXPECT_TRUE(r.first_pass.encoded);
  EXPECT_TRUE(r.hires.encoded);
  EXPECT_EQ(r.hires.steps_run, 2);
  EXPECT_EQ(r.memory.dit_loads, 1);
  EXPECT_EQ(bit_mismatches(r.latents, z.latents), 0u);
}

// Cancelling between the passes (on_phase refusing the second denoise phase) and during the hires encode (the flag set
// while it runs: honoured right after it, before the second pass samples). The result is the first pass's.
TEST(GpuPipelineImg2Img, HiresCancelsBetweenPassesAndDuringTheEncode) {
  if (!have_inputs()) GTEST_SKIP() << "weights or reference image not present";
  pl::Pipeline pipe;
  pl::Request req = base_request(512, 512, 2);
  req.decode = true;
  req.hires.width = 1024;
  req.hires.height = 1024;
  req.hires.strength = 0.5;
  for (const bool during_encode : {false, true}) {
    std::atomic<bool> cancel{false};
    std::thread setter;
    int denoise_phases = 0;
    pl::GenerateControl control;
    control.cancel = &cancel;
    control.on_phase = [&](pl::Phase p) {
      if (p != pl::Phase::kDenoise || ++denoise_phases < 2) return true;
      if (!during_encode) return false;
      setter = std::thread([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        cancel.store(true);
      });
      return true;
    };
    const pl::Result r = pipe.generate(req, control);
    if (setter.joinable()) setter.join();
    std::printf("[img2img] cancel %s: steps %d, hires encoded %d, latents %zu, encode %.3f s\n",
                during_encode ? "during the hires encode" : "between the passes", r.steps_run, r.hires.encoded ? 1 : 0,
                r.latents.size(), r.timings.encode_s);
    EXPECT_TRUE(r.cancelled);
    EXPECT_EQ(r.steps_run, 2);
    EXPECT_FALSE(r.hires.ran);
    EXPECT_EQ(r.hires.steps_run, 0);
    EXPECT_EQ(r.hires.encoded, during_encode);
    EXPECT_EQ(r.width, 512);
    EXPECT_EQ(r.latents.size(), std::size_t{128} * 32 * 32);
    EXPECT_TRUE(r.rgb.empty());
  }
  // The pipeline stays usable.
  const pl::Result ok = pipe.generate(req);
  EXPECT_FALSE(ok.cancelled);
  EXPECT_EQ(ok.width, 1024);
}

// img2img with a runtime LoRA (SFT bundle): runs, is deterministic, and the LoRA changes the result.
TEST(GpuPipelineImg2Img, Img2ImgWithRuntimeLora) {
  if (!have_inputs() || !fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(kPeftLora)) {
    GTEST_SKIP() << "SFT bundle or LoRA file not present";
  }
  pl::Pipeline pipe;
  pl::Request req = base_request(1024, 1024, 6, "sft");
  req.init_image = init_image();
  req.denoise_strength = 0.5;
  const pl::Result plain = pipe.generate(req);
  req.loras = {{kPeftLora, 0.8f}};
  const pl::Result a = pipe.generate(req);
  print_timings("img2img + LoRA", a);
  const pl::Result b = pipe.generate(req);
  EXPECT_EQ(a.steps_run, 3);
  EXPECT_EQ(a.loras.size(), 1u);
  EXPECT_EQ(bit_mismatches(a.latents, b.latents), 0u);
  EXPECT_GT(bit_mismatches(a.latents, plain.latents), a.latents.size() / 2);
}

}  // namespace
