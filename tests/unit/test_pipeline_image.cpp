// src/pipeline/pipeline_image (P7b host side) against PIL / diffusers / torch vectors (tools/reference/
// gen_pipeline_image_vectors.py): Lanczos resizes and the crop / fit / stretch modes bit-exact, the inpainting mask's
// binarize + bilinear interpolation and the blend's fp32 op order bit-exact, JPEG EXIF orientation, PNG decoding,
// the denoise strength's start step, and the img2img / inpainting / hires fix request validation. CPU only.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include "pipeline/pipeline.h"
#include "pipeline/pipeline_image.h"
#include "pipeline/png_writer.h"
#include "pipeline/report_json.h"
#include "util/json.h"
#include "util/npy.h"

namespace {

namespace fs = std::filesystem;
using mi::pipeline::Image;
using mi::pipeline::ResizeMode;

fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "pipeline_image"; }

Image load_npy_image(const std::string& name) {
  const mi::npy::Array a = mi::npy::load(data_dir() / (name + ".npy"));
  Image im;
  im.height = static_cast<int>(a.shape.at(0));
  im.width = static_cast<int>(a.shape.at(1));
  im.channels = a.shape.size() == 3 ? static_cast<int>(a.shape[2]) : 1;
  im.pixels = a.to_vector<std::uint8_t>();
  return im;
}

std::vector<float> load_f32(const std::string& name) { return mi::npy::load(data_dir() / (name + ".npy")).to_vector<float>(); }

std::uint32_t bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

int mismatches(const Image& a, const Image& b) {
  if (a.width != b.width || a.height != b.height || a.channels != b.channels || a.pixels.size() != b.pixels.size()) return -1;
  int n = 0;
  for (std::size_t i = 0; i < a.pixels.size(); ++i) n += a.pixels[i] != b.pixels[i];
  return n;
}

bool have_data() { return fs::is_regular_file(data_dir() / "cases.json"); }

mi::json::Value read_cases() {
  std::ifstream in(data_dir() / "cases.json", std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return mi::json::parse(text);
}

TEST(PipelineImage, LanczosResizeMatchesPilBitExact) {
  if (!have_data()) GTEST_SKIP() << "test vectors absent";
  int checked = 0;
  const mi::json::Value cases = read_cases();
  for (const auto& c : cases.as_array()) {
    if (c.at("kind").as_string() != "resize") continue;
    const Image src = load_npy_image(c.at("src").as_string());
    const auto& size = c.at("size").as_array();
    const int w = static_cast<int>(size[0].as_int()), h = static_cast<int>(size[1].as_int());
    Image got;
    if (c.at("box").is_null()) {
      got = mi::pipeline::resize_lanczos(src, w, h);
    } else {
      const auto& b = c.at("box").as_array();
      got = mi::pipeline::resize_lanczos(src, w, h, static_cast<float>(b[0].as_double()), static_cast<float>(b[1].as_double()),
                                         static_cast<float>(b[2].as_double()), static_cast<float>(b[3].as_double()));
    }
    const Image want = load_npy_image("resize_" + c.at("name").as_string());
    EXPECT_EQ(mismatches(got, want), 0) << c.at("name").as_string();
    ++checked;
  }
  EXPECT_EQ(checked, 8);
}

TEST(PipelineImage, ResizeModesMatchDiffusersBitExact) {
  if (!have_data()) GTEST_SKIP() << "test vectors absent";
  int checked = 0;
  const mi::json::Value cases = read_cases();
  for (const auto& c : cases.as_array()) {
    if (c.at("kind").as_string() != "mode") continue;
    const std::string src_name = c.at("src").as_string();
    const Image src = load_npy_image(src_name);
    const auto& size = c.at("size").as_array();
    const int w = static_cast<int>(size[0].as_int()), h = static_cast<int>(size[1].as_int());
    const std::string mode = c.at("mode").as_string();
    const Image got = mi::pipeline::resize_to(src, w, h, mi::pipeline::parse_resize_mode(mode));
    const std::string id = "mode_" + mode + "_" + src_name + "_" + std::to_string(w) + "x" + std::to_string(h);
    EXPECT_EQ(mismatches(got, load_npy_image(id)), 0) << id;
    ++checked;
  }
  EXPECT_EQ(checked, 18);
  // The same size is the image itself in every mode.
  const Image src = load_npy_image("src_rgb_37x23");
  for (const ResizeMode m : {ResizeMode::kCrop, ResizeMode::kFit, ResizeMode::kStretch}) {
    EXPECT_EQ(mismatches(mi::pipeline::resize_to(src, 37, 23, m), src), 0);
  }
  EXPECT_THROW(mi::pipeline::parse_resize_mode("cover"), std::invalid_argument);
  EXPECT_STREQ(mi::pipeline::resize_mode_name(ResizeMode::kFit), "fit");
}

TEST(PipelineImage, LatentMaskMatchesTorchBilinearBitExact) {
  if (!have_data()) GTEST_SKIP() << "test vectors absent";
  int checked = 0;
  const mi::json::Value cases = read_cases();
  for (const auto& c : cases.as_array()) {
    if (c.at("kind").as_string() != "mask") continue;
    const auto& size = c.at("size").as_array();
    const auto& lat = c.at("latent").as_array();
    const std::string tag = std::to_string(size[0].as_int()) + "x" + std::to_string(size[1].as_int());
    const Image mask = load_npy_image("mask_" + tag);
    const int lw = static_cast<int>(lat[0].as_int()), lh = static_cast<int>(lat[1].as_int());
    const std::vector<float> got = mi::pipeline::latent_mask(mask, lh, lw);
    const std::vector<float> want = load_f32("latent_mask_" + tag + "_" + std::to_string(lw) + "x" + std::to_string(lh));
    ASSERT_EQ(got.size(), want.size()) << tag;
    int bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) bad += bits(got[i]) != bits(want[i]);
    EXPECT_EQ(bad, 0) << tag;
    ++checked;
  }
  EXPECT_EQ(checked, 4);
  Image rgb = load_npy_image("src_rgb_37x23");
  EXPECT_THROW(mi::pipeline::latent_mask(rgb, 2, 2), std::invalid_argument);
}

TEST(PipelineImage, InpaintBlendFollowsTorchOpOrder) {
  if (!have_data()) GTEST_SKIP() << "test vectors absent";
  std::vector<float> x = load_f32("blend_x");
  const std::vector<float> z0 = load_f32("blend_z0"), noise = load_f32("blend_noise"), mask = load_f32("blend_mask");
  const std::vector<float> want = load_f32("blend_out");
  mi::pipeline::inpaint_blend(x, z0, noise, mask, load_f32("blend_sigma")[0]);
  int bad = 0;
  for (std::size_t i = 0; i < x.size(); ++i) bad += bits(x[i]) != bits(want[i]);
  EXPECT_EQ(bad, 0);
  // sigma 0 (after the last step): the kept region is z0 exactly, the regenerated one x.
  std::vector<float> y = load_f32("blend_x");
  const std::vector<float> keep_all(mask.size(), 0.0f), regen_all(mask.size(), 1.0f);
  mi::pipeline::inpaint_blend(y, z0, noise, keep_all, 0.0f);
  EXPECT_EQ(std::memcmp(y.data(), z0.data(), y.size() * sizeof(float)), 0);
  std::vector<float> v = load_f32("blend_x");
  mi::pipeline::inpaint_blend(v, z0, noise, regen_all, 0.3f);
  const std::vector<float> x0 = load_f32("blend_x");
  for (std::size_t i = 0; i < v.size(); ++i) EXPECT_EQ(v[i], x0[i]);
  std::vector<float> wrong(3);
  EXPECT_THROW(mi::pipeline::inpaint_blend(wrong, z0, noise, mask, 0.5f), std::invalid_argument);
}

TEST(PipelineImage, StrengthStartStepIsDiffusersGetTimesteps) {
  using mi::pipeline::strength_start_step;
  // t_start = int(max(N - min(N * strength, N), 0)) in Python floats.
  EXPECT_EQ(strength_start_step(50, 1.0), 0);
  EXPECT_EQ(strength_start_step(50, 0.0), 50);
  EXPECT_EQ(strength_start_step(50, 0.5), 25);
  EXPECT_EQ(strength_start_step(50, 0.33), 33);  // 50 - 16.5 = 33.5
  EXPECT_EQ(strength_start_step(50, 0.01), 49);  // a sliver of strength still runs one step
  EXPECT_EQ(strength_start_step(7, 0.3), 4);     // 7 - 2.1 = 4.9
  EXPECT_EQ(strength_start_step(28, 0.75), 7);
  EXPECT_EQ(strength_start_step(1, 0.5), 0);     // 1 - 0.5 = 0.5 -> 0
}

TEST(PipelineImage, JpegExifOrientationMatchesPil) {
  if (!have_data()) GTEST_SKIP() << "test vectors absent";
  // stb_image's JPEG decoder is not libjpeg (IDCT rounding): a few levels apart at most on this quality-100 4:4:4 file;
  // a wrong orientation is off by tens of levels.
  for (int o = 1; o <= 8; ++o) {
    const Image got = mi::pipeline::load_image(data_dir() / ("orient_" + std::to_string(o) + ".jpg"));
    const Image want = load_npy_image("orient_" + std::to_string(o) + "_pil");
    ASSERT_EQ(got.width, want.width) << o;
    ASSERT_EQ(got.height, want.height) << o;
    int max_diff = 0;
    double sum = 0;
    for (std::size_t i = 0; i < got.pixels.size(); ++i) {
      const int d = std::abs(int{got.pixels[i]} - int{want.pixels[i]});
      max_diff = std::max(max_diff, d);
      sum += d;
    }
    const double mean = sum / static_cast<double>(got.pixels.size());
    std::printf("[pipeline_image] orientation %d: %dx%d, max |diff| %d, mean %.3f vs PIL\n", o, got.width, got.height, max_diff, mean);
    EXPECT_LE(max_diff, 6) << o;
    EXPECT_LT(mean, 1.0) << o;
    EXPECT_EQ(got.name, "orient_" + std::to_string(o) + ".jpg");
  }
  // Motorola byte order.
  const Image mm = mi::pipeline::load_image(data_dir() / "orient_6_motorola.jpg");
  const Image ref = mi::pipeline::load_image(data_dir() / "orient_6.jpg");
  EXPECT_EQ(mismatches(mm, ref), 0);
}

TEST(PipelineImage, DecodesPngsAndRejectsGarbage) {
  if (!have_data()) GTEST_SKIP() << "test vectors absent";
  // Round trip through the engine's own PNG writer.
  const Image src = load_npy_image("src_rgb_37x23");
  const std::vector<std::uint8_t> png = mi::pipeline::encode_png_rgb8(src.pixels, src.width, src.height);
  const Image back = mi::pipeline::decode_image(png);
  EXPECT_EQ(mismatches(back, src), 0);
  // Grayscale PNG -> RGB (replicated) and -> L (itself).
  const Image gray = load_npy_image("src_l_37x23");
  const Image g1 = mi::pipeline::load_image(data_dir() / "gray.png", 1);
  EXPECT_EQ(mismatches(g1, gray), 0);
  const Image g3 = mi::pipeline::load_image(data_dir() / "gray.png", 3);
  ASSERT_EQ(g3.channels, 3);
  for (std::size_t i = 0; i < gray.pixels.size(); ++i) {
    ASSERT_EQ(g3.pixels[i * 3], gray.pixels[i]);
    ASSERT_EQ(g3.pixels[i * 3 + 2], gray.pixels[i]);
  }
  // RGBA: alpha dropped, as PIL's convert("RGB").
  EXPECT_EQ(mismatches(mi::pipeline::load_image(data_dir() / "rgba.png", 3), load_npy_image("rgba_rgb")), 0);
  // RGB -> L with PIL's integer formula.
  EXPECT_EQ(mismatches(mi::pipeline::decode_image(png, 1), load_npy_image("gray_as_l_from_rgb")), 0);
  const std::vector<std::uint8_t> junk = {0x89, 'P', 'N', 'G', 1, 2, 3};
  EXPECT_THROW(mi::pipeline::decode_image(junk), mi::pipeline::ImageFormatError);
  EXPECT_THROW(mi::pipeline::load_image(data_dir() / "missing.png"), mi::pipeline::ImageReadError);
  EXPECT_THROW(mi::pipeline::decode_image(png, 2), std::invalid_argument);
}

mi::pipeline::Request base_request() {
  mi::pipeline::Request r;
  r.prompt = "a cat";
  r.width = 512;
  r.height = 384;
  return r;
}

std::shared_ptr<Image> solid(int w, int h, int channels, std::uint8_t v) {
  auto im = std::make_shared<Image>();
  im->width = w;
  im->height = h;
  im->channels = channels;
  im->pixels.assign(static_cast<std::size_t>(w) * h * channels, v);
  return im;
}

TEST(PipelineImage, ValidatesImg2ImgInpaintAndHires) {
  using mi::pipeline::validate;
  mi::pipeline::Request r = base_request();
  EXPECT_NO_THROW(validate(r));
  // Strength without an init image, out of range, not finite.
  r.denoise_strength = 0.5;
  EXPECT_THROW(validate(r), std::invalid_argument);
  r.init_image = solid(300, 200, 3, 128);
  EXPECT_NO_THROW(validate(r));
  for (const double s : {-0.1, 1.5, std::nan("")}) {
    r.denoise_strength = s;
    EXPECT_THROW(validate(r), std::invalid_argument) << s;
  }
  r.denoise_strength = 0.0;
  EXPECT_NO_THROW(validate(r));
  r.denoise_strength = 1.0;
  EXPECT_NO_THROW(validate(r));
  // Pixel buffer size mismatch, wrong channel count, empty size.
  {
    auto bad = solid(300, 200, 3, 128);
    bad->pixels.pop_back();
    r.init_image = bad;
    EXPECT_THROW(validate(r), std::invalid_argument);
  }
  r.init_image = solid(300, 200, 1, 128);
  EXPECT_THROW(validate(r), std::invalid_argument);
  {
    auto bad = solid(300, 200, 3, 128);
    bad->width = 0;
    r.init_image = bad;
    EXPECT_THROW(validate(r), std::invalid_argument);
    r.init_image = std::make_shared<Image>();  // non-null but empty
    EXPECT_THROW(validate(r), std::invalid_argument);
    // Over the input bounds: 64 MP (pixels need not be allocated: the size check comes first).
    auto huge = std::make_shared<Image>();
    huge->width = 10000;
    huge->height = 8000;
    r.init_image = huge;
    EXPECT_THROW(validate(r), std::invalid_argument);
  }
  r.init_image = solid(300, 200, 3, 128);
  // Mask: needs the init image's size, one channel, a matching buffer, and an init image at all.
  r.mask = solid(300, 200, 1, 255);
  EXPECT_NO_THROW(validate(r));
  r.mask = solid(299, 200, 1, 255);
  EXPECT_THROW(validate(r), std::invalid_argument);
  r.mask = solid(300, 200, 3, 255);
  EXPECT_THROW(validate(r), std::invalid_argument);
  {
    auto bad = solid(300, 200, 1, 255);
    bad->pixels.resize(10);
    r.mask = bad;
    EXPECT_THROW(validate(r), std::invalid_argument);
  }
  mi::pipeline::Request no_init = base_request();
  no_init.mask = solid(300, 200, 1, 255);
  EXPECT_THROW(validate(no_init), std::invalid_argument);
  // Hires fix: multiples of 16, size bounds, strength and steps ranges.
  mi::pipeline::Request h = base_request();
  h.hires.width = 1024;
  h.hires.height = 768;
  EXPECT_NO_THROW(validate(h));
  h.hires.width = 1000;
  EXPECT_THROW(validate(h), std::invalid_argument);
  h.hires.width = 1024;
  h.hires.height = 0;
  EXPECT_THROW(validate(h), std::invalid_argument);
  h.hires.height = 4096;
  h.hires.width = 4096;
  EXPECT_THROW(validate(h), std::invalid_argument);  // 16.8 MP
  h.hires.width = 1024;
  h.hires.height = 768;
  h.hires.strength = 1.2;
  EXPECT_THROW(validate(h), std::invalid_argument);
  h.hires.strength = 0.5;
  h.hires.steps = -1;
  EXPECT_THROW(validate(h), std::invalid_argument);
  h.hires.steps = 0;
  EXPECT_NO_THROW(validate(h));
  // The dataspace sampler cannot start from an image.
  h.sampler = mi::pipeline::SamplerKind::kDataSpace;
  h.dataspace_path = "x.safetensors";
  EXPECT_THROW(validate(h), std::invalid_argument);
}

TEST(PipelineImage, MetadataAndReportCarryTheImageSettings) {
  mi::pipeline::Request q = base_request();
  q.model = "/models/sft";
  auto init = solid(300, 200, 3, 128);
  init->name = "photo.png";
  q.init_image = init;
  q.denoise_strength = 0.6;
  auto mask = solid(300, 200, 1, 255);
  mask->name = "/private/dir/mask.png";
  q.mask = mask;
  q.resize = ResizeMode::kFit;
  q.hires.width = 1024;
  q.hires.height = 768;
  q.hires.strength = 0.4;
  mi::pipeline::Result r;
  r.image_mode = "inpaint";
  r.width = 1024;
  r.height = 768;
  const mi::pipeline::GenerationMetadata m = mi::pipeline::generation_metadata(q, r);
  const auto find = [&](const std::string& key) {
    for (const auto& [k, v] : m.extra) {
      if (k == key) return v;
    }
    return std::string("<absent>");
  };
  EXPECT_EQ(find("mode"), "inpaint");
  EXPECT_EQ(find("init_image"), "photo.png 300x200");
  EXPECT_EQ(find("denoise_strength"), "0.6");
  EXPECT_EQ(find("resize"), "fit");
  EXPECT_EQ(find("mask"), "mask.png 300x200");  // base name only
  EXPECT_EQ(find("hires"), "1024x768");
  EXPECT_EQ(find("hires_base"), "512x384");
  EXPECT_EQ(find("hires_strength"), "0.4");
  EXPECT_EQ(find("hires_steps"), "50");
  // The PNG text chunks are valid (the writer throws otherwise).
  EXPECT_NO_THROW(mi::pipeline::generation_text_chunks(m));
  // The report: the request's image settings and what each pass ran, as valid JSON.
  r.first_pass.ran = true;
  r.first_pass.width = 512;
  r.first_pass.start_step = 20;
  r.first_pass.encoded = true;
  r.first_pass.encode_schedule = "untiled";
  r.hires.ran = true;
  r.hires.width = 1024;
  r.vae_encoder_source = "/models/sft/vae/diffusion_pytorch_model.safetensors";
  const mi::json::Value doc = mi::json::parse("{\n" + mi::pipeline::report_fields(q, r, "out.png", "") + "\n}");
  const mi::json::Value& req = doc.at("request");
  EXPECT_EQ(req.at("init_image").at("name").as_string(), "photo.png");
  EXPECT_EQ(req.at("init_image").at("width").as_int(), 300);
  EXPECT_DOUBLE_EQ(req.at("denoise_strength").as_double(), 0.6);
  EXPECT_EQ(req.at("mask").at("height").as_int(), 200);
  EXPECT_EQ(req.at("resize").as_string(), "fit");
  EXPECT_EQ(req.at("hires").at("width").as_int(), 1024);
  const mi::json::Value& im = doc.at("img2img");
  EXPECT_EQ(im.at("mode").as_string(), "inpaint");
  EXPECT_EQ(im.at("upscaler").as_string(), "lanczos");
  EXPECT_EQ(im.at("first_pass").at("start_step").as_int(), 20);
  EXPECT_TRUE(im.at("first_pass").at("encoded").as_bool());
  EXPECT_TRUE(im.at("hires").at("ran").as_bool());
  // txt2img: nulls.
  const mi::json::Value plain = mi::json::parse("{\n" + mi::pipeline::report_fields(base_request(), mi::pipeline::Result{}, "", "") + "\n}");
  EXPECT_TRUE(plain.at("request").at("init_image").is_null());
  EXPECT_TRUE(plain.at("request").at("hires").is_null());
  EXPECT_TRUE(plain.at("img2img").at("hires").is_null());
  EXPECT_EQ(plain.at("img2img").at("mode").as_string(), "txt2img");
  // An init image the run did not use (strength 1, no mask): no image settings in the metadata, the note reported.
  mi::pipeline::Request unused = base_request();
  unused.init_image = init;
  mi::pipeline::Result ur;
  ur.image_mode = "txt2img";
  ur.image_note = "the init image was not used";
  const mi::pipeline::GenerationMetadata um = mi::pipeline::generation_metadata(unused, ur);
  for (const auto& [k, v] : um.extra) EXPECT_NE(k, "init_image");
  EXPECT_EQ(mi::json::parse("{\n" + mi::pipeline::report_fields(unused, ur, "", "") + "\n}").at("img2img").at("note").as_string(),
            ur.image_note);
}

// The engine's own keys in parameters.json.
TEST(PipelineImage, ParametersCarryImageScalars) {
  const mi::pipeline::Request r = mi::pipeline::request_from_parameters(mi::json::parse(
      R"({"prompt": "x", "denoise_strength": 0.4, "resize": "fit", "hires_strength": 0.3, "hires_steps": 12})"));
  EXPECT_DOUBLE_EQ(r.denoise_strength, 0.4);
  EXPECT_EQ(r.resize, ResizeMode::kFit);
  EXPECT_DOUBLE_EQ(r.hires.strength, 0.3);
  EXPECT_EQ(r.hires.steps, 12);
  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"resize": "cover"})")), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"resize": 3})")), std::invalid_argument);
}

// Untrusted input: deterministic mutations (bit flips, byte overwrites, truncations, inserted junk, size fields) of
// PNG and JPEG seeds either decode to a consistent image within the bounds or throw std::runtime_error; nothing else
// (no crash, no other exception: run under ASan/UBSan in the sanitizer build). Oversized headers are refused before
// any pixel allocation.
TEST(PipelineImage, DecodeSurvivesMutatedFiles) {
  if (!have_data()) GTEST_SKIP() << "test vectors absent";
  const auto read = [](const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  };
  const Image src = load_npy_image("src_rgb_37x23");
  std::vector<std::vector<std::uint8_t>> seeds = {mi::pipeline::encode_png_rgb8(src.pixels, src.width, src.height),
                                                  read(data_dir() / "gray.png"), read(data_dir() / "rgba.png"),
                                                  read(data_dir() / "orient_6.jpg"), read(data_dir() / "orient_6_motorola.jpg")};
  std::mt19937 rng(20261022);
  int decoded = 0, refused = 0;
  for (const auto& seed : seeds) {
    ASSERT_GT(seed.size(), 64u);
    for (int it = 0; it < 1500; ++it) {
      std::vector<std::uint8_t> f = seed;
      const int kind = static_cast<int>(rng() % 5);
      const int edits = 1 + static_cast<int>(rng() % 8);
      for (int e = 0; e < edits; ++e) {
        const std::size_t at = rng() % f.size();
        switch (kind) {
          case 0: f[at] ^= static_cast<std::uint8_t>(1u << (rng() % 8)); break;
          case 1: f[at] = static_cast<std::uint8_t>((rng() & 1) ? 0xFF : 0x00); break;
          case 2: f.resize(std::max<std::size_t>(1, at)); break;
          case 3: f.insert(f.begin() + static_cast<std::ptrdiff_t>(at), static_cast<std::uint8_t>(rng())); break;
          default:
            // Header fields: the first 64 bytes hold the PNG IHDR / the JPEG markers and frame size.
            f[at % 64] = static_cast<std::uint8_t>(rng());
            break;
        }
        if (f.empty()) break;
      }
      try {
        const Image im = mi::pipeline::decode_image(f, (it & 1) ? 1 : 3);
        ASSERT_GT(im.width, 0);
        ASSERT_LE(im.width, mi::pipeline::kMaxInputSide);
        ASSERT_LE(std::int64_t{im.width} * im.height, mi::pipeline::kMaxInputPixels);
        ASSERT_EQ(im.pixels.size(), static_cast<std::size_t>(im.width) * im.height * static_cast<std::size_t>(im.channels));
        ++decoded;
      } catch (const std::runtime_error&) {
        ++refused;
      }
    }
  }
  std::printf("[pipeline_image] mutated files: %d decoded, %d refused\n", decoded, refused);
  EXPECT_GT(refused, 0);
  // A PNG header claiming 20000 x 20000 (over the side bound) and 10000 x 9000 (over 64 MP): refused from the header.
  for (const auto& [w, h] : {std::pair{20000u, 20000u}, std::pair{10000u, 9000u}}) {
    std::vector<std::uint8_t> f = seeds[0];
    for (int i = 0; i < 4; ++i) {
      f[16 + i] = static_cast<std::uint8_t>(w >> (24 - 8 * i));  // IHDR width, big-endian (stb does not check the CRC)
      f[20 + i] = static_cast<std::uint8_t>(h >> (24 - 8 * i));
    }
    EXPECT_THROW(mi::pipeline::decode_image(f), std::runtime_error) << w << "x" << h;
  }
  // Files: a directory and an oversized file are refused without reading them whole.
  EXPECT_THROW(mi::pipeline::load_image(data_dir()), mi::pipeline::ImageReadError);
  const std::vector<std::uint8_t> big(mi::pipeline::kMaxImageFileBytes + 1, 0);
  EXPECT_THROW(mi::pipeline::decode_image(big), std::runtime_error);
}

}  // namespace
