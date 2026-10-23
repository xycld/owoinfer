// src/pipeline: sigma schedule (bit-exact vs tests/golden and vectors from the PyTorch reference), the fp32
// guidance/integrator kernels (bit-exact vs a host emulation of the torch ops), request parsing and validation,
// the PNG writer, and with real weights: the DiT + sampler loop against short torch runs on golden text-encoder
// states, and the whole pipeline (own tokenizer + text encoder) against the same runs.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/moss_dit/moss_dit.h"
#include "pipeline/hardware_profile.h"
#include "pipeline/pipeline.h"
#include "pipeline/report_json.h"
#include "pipeline/png_writer.h"
#include "pipeline/quality_cert.h"
#include "pipeline/rng_torch_compat.h"
#include "pipeline/sampler.h"
#include "planner/planner.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "test_util.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"

// Independent PNG decoder for the writer's round trip (static: test_flux2_vae.cpp links its own copy).
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {

namespace fs = std::filesystem;
using mi::Tensor;
using mi::pipeline::Guidance;
using mi::pipeline::SamplerKind;
using mi::pipeline::Schedule;

fs::path weights_root() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image");
}
fs::path golden_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "golden"; }
fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "pipeline"; }

mi::json::Value read_json(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return mi::json::parse(ss.str());
}

std::uint32_t bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

// Number of elements whose bit patterns differ.
std::size_t bit_mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += bits(a[i]) != bits(b[i]);
  return n;
}

std::vector<float> slice(const std::vector<float>& v, std::int64_t offset, std::int64_t n) {
  return {v.begin() + offset, v.begin() + offset + n};
}

// ---------------------------------------------------------------- schedule

TEST(PipelineSchedule, GoldenSigmasAndTimestepsBitExact) {
  int cases = 0;
  const auto golden_cases = read_json(golden_dir() / "cases.json");
  for (const auto& c : golden_cases.as_array()) {
    const fs::path dir = golden_dir() / c.at("id").as_string();
    if (!fs::is_regular_file(dir / "sigmas.npy")) continue;
    const Schedule s = mi::pipeline::moss_schedule(static_cast<int>(c.at("steps").as_int()), c.at("shift").as_double());
    const auto sigmas = mi::npy::load(dir / "sigmas.npy").to_vector<float>();
    const auto timesteps = mi::npy::load(dir / "timesteps.npy").to_vector<float>();
    EXPECT_EQ(bit_mismatches(s.sigmas, sigmas), 0u) << c.at("id").as_string();
    EXPECT_EQ(bit_mismatches(s.timesteps, timesteps), 0u) << c.at("id").as_string();
    EXPECT_EQ(s.num_steps(), 50);
    ++cases;
  }
  EXPECT_GE(cases, 7);
}

TEST(PipelineSchedule, MatchesTorchForStepsAndShifts) {
  const auto manifest = read_json(data_dir() / "manifest.json");
  const auto sigmas = mi::npy::load(data_dir() / "schedule_sigmas.npy").to_vector<float>();
  const auto timesteps = mi::npy::load(data_dir() / "schedule_timesteps.npy").to_vector<float>();
  const auto grids = mi::npy::load(data_dir() / "schedule_grid.npy").to_vector<float>();
  int pairs = 0, reduced = 0;
  for (const auto& e : manifest.at("schedule").as_array()) {
    const int steps = static_cast<int>(e.at("steps").as_int());
    const double shift = e.at("shift").as_double();
    const std::string what = "steps " + std::to_string(steps) + " shift " + std::to_string(shift);
    const Schedule s = mi::pipeline::moss_schedule(steps, shift);
    EXPECT_EQ(bit_mismatches(s.sigmas, slice(sigmas, e.at("sigma_offset").as_int(), e.at("num_sigmas").as_int())), 0u) << what;
    EXPECT_EQ(bit_mismatches(s.timesteps, slice(timesteps, e.at("timestep_offset").as_int(), e.at("num_timesteps").as_int())), 0u)
        << what;
    EXPECT_EQ(bit_mismatches(mi::pipeline::moss_sigma_grid(steps, shift), slice(grids, e.at("grid_offset").as_int(), steps)), 0u)
        << what;
    reduced += s.num_steps() < steps;
    ++pairs;
  }
  EXPECT_EQ(pairs, 210);
  EXPECT_GT(reduced, 0);  // snapping + unique_consecutive does drop steps for some settings
  std::printf("[pipeline] %d (steps, shift) schedules bit-exact; %d come out shorter than requested\n", pairs, reduced);

  const auto training = mi::npy::load(data_dir() / "schedule_training.npy").to_vector<float>();
  std::vector<float> grid = mi::pipeline::moss_sigma_grid(mi::pipeline::kNumTrainTimesteps, 1.0);
  std::reverse(grid.begin(), grid.end());
  EXPECT_EQ(bit_mismatches(grid, training), 0u);
}

TEST(PipelineSchedule, LinspaceEdgeCasesAndArgumentChecks) {
  EXPECT_EQ(mi::pipeline::torch_linspace_f32(1.0f, 0.5f, 1), std::vector<float>{1.0f});
  EXPECT_EQ(mi::pipeline::torch_linspace_f32(1.0f, 0.5f, 2), (std::vector<float>{1.0f, 0.5f}));
  const Schedule one = mi::pipeline::moss_schedule(1, 5.0);
  ASSERT_EQ(one.num_steps(), 1);
  EXPECT_EQ(one.sigmas, (std::vector<float>{1.0f, 0.0f}));
  EXPECT_EQ(one.timesteps, std::vector<float>{1000.0f});
  EXPECT_THROW(mi::pipeline::moss_schedule(0, 5.0), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::moss_schedule(10, 0.0), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::moss_schedule(10, std::nan("")), std::invalid_argument);
}

TEST(PipelineGuidance, SkipWindowFollowsStepOverNumSteps) {
  Guidance g;
  g.slg_start = 0.25;
  g.slg_stop = 0.75;
  std::vector<int> active;
  for (int i = 0; i < 8; ++i) {
    if (g.skip_active(i, 8)) active.push_back(i);
  }
  EXPECT_EQ(active, (std::vector<int>{2, 3, 4, 5}));
  EXPECT_FALSE(g.slg_active(3, 8));  // no SLG layers
  g.slg_layers.set(24);
  EXPECT_FALSE(g.slg_active(3, 8));  // scale 0
  g.slg_scale = 1.0;
  EXPECT_TRUE(g.slg_active(3, 8));
  EXPECT_FALSE(g.slg_active(6, 8));
  Guidance full;  // default [0, 1): always active
  EXPECT_TRUE(full.skip_active(0, 50));
  EXPECT_TRUE(full.skip_active(49, 50));
  full.slg_stop = 0.0;
  EXPECT_FALSE(full.skip_active(0, 50));
}

// ---------------------------------------------------------------- fp32 kernels

struct Dev {
  mi::DeviceBuffer buf;
  Tensor t;
  explicit Dev(const std::vector<float>& host) {
    buf = mi::DeviceBuffer(host.size() * 4);
    t = Tensor::view(buf.get(), mi::DType::kF32, mi::Device::kCUDA, {static_cast<std::int64_t>(host.size())});
    buf.upload(host.data(), host.size() * 4);
  }
  std::vector<float> get() const {
    std::vector<float> h(static_cast<std::size_t>(t.numel()));
    MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, h.size() * 4, cudaMemcpyDeviceToHost));
    return h;
  }
};

// One rounding per torch op: volatile keeps the host compiler from contracting the emulation.
float mul(float a, float b) {
  volatile float r = a * b;
  return r;
}

TEST(PipelineKernels, GuidanceAndIntegratorsMatchTorchOpOrder) {
  constexpr std::size_t n = 100003;
  std::mt19937 rng(7);
  std::normal_distribution<float> d(0.0f, 1.5f);
  std::vector<float> a(n), b(n), c(n), x(n);
  for (std::size_t i = 0; i < n; ++i) a[i] = d(rng), b[i] = d(rng), c[i] = d(rng), x[i] = d(rng);
  Dev da(a), db(b), dc(c), out(std::vector<float>(n, 0.0f));
  const float g = 3.7f, s = 0.85f, h = -0.0412f, ka = 1.31f, kb = 0.31f;

  mi::pipeline::cfg_combine(da.t, db.t, g, out.t, nullptr);
  std::vector<float> ref(n);
  for (std::size_t i = 0; i < n; ++i) ref[i] = b[i] + mul(g, a[i] - b[i]);
  EXPECT_EQ(bit_mismatches(out.get(), ref), 0u);

  mi::pipeline::slg_correct(da.t, db.t, dc.t, s, out.t, nullptr);
  for (std::size_t i = 0; i < n; ++i) ref[i] = a[i] + mul(s, b[i] - c[i]);
  EXPECT_EQ(bit_mismatches(out.get(), ref), 0u);

  Dev dx(x);
  mi::pipeline::euler_update(dx.t, da.t, h, nullptr);
  for (std::size_t i = 0; i < n; ++i) ref[i] = x[i] + mul(h, a[i]);
  EXPECT_EQ(bit_mismatches(dx.get(), ref), 0u);

  Dev dy(x);
  mi::pipeline::ab2_update(dy.t, da.t, db.t, h, ka, kb, nullptr);
  for (std::size_t i = 0; i < n; ++i) ref[i] = x[i] + mul(h, mul(ka, a[i]) - mul(kb, b[i]));
  EXPECT_EQ(bit_mismatches(dy.get(), ref), 0u);

  Dev small(std::vector<float>(5, 0.0f));
  EXPECT_THROW(mi::pipeline::euler_update(small.t, da.t, h, nullptr), std::invalid_argument);
}

// ---------------------------------------------------------------- requests

TEST(PipelineRequest, ParsesAssetParameters) {
  // The official app's example parameters: a local checkout of the reference, not part of the repository.
  const fs::path assets = fs::path(MOSSINFER_SOURCE_DIR) / "third_party/reference/moss-image-official/assets/images";
  if (!fs::is_regular_file(assets / "007" / "parameters.json") || !fs::is_regular_file(assets / "018" / "parameters.json")) {
    GTEST_SKIP() << "official app example parameters absent (" << assets << ")";
  }
  const mi::pipeline::Request r7 = mi::pipeline::request_from_parameters(read_json(assets / "007" / "parameters.json"));
  EXPECT_EQ(r7.height, 1776);
  EXPECT_EQ(r7.width, 1184);
  EXPECT_EQ(r7.steps, 50);
  EXPECT_EQ(r7.cfg, 3.0);
  EXPECT_EQ(r7.shift, 5.0);
  EXPECT_EQ(r7.seed, 3537712559);
  EXPECT_EQ(r7.uncond_skip_layers, std::vector<int>{22});
  EXPECT_TRUE(r7.slg_layers.empty());
  EXPECT_EQ(r7.negative_prompt, "watermark, bad anatomy");
  EXPECT_EQ(r7.max_sequence_length, 4096);
  EXPECT_EQ(r7.prompt.rfind("A young anime woman, Kazusa", 0), 0u);
  EXPECT_NO_THROW(mi::pipeline::validate(r7));

  const mi::pipeline::Request r18 = mi::pipeline::request_from_parameters(read_json(assets / "018" / "parameters.json"));
  EXPECT_TRUE(r18.uncond_skip_layers.empty());
  EXPECT_EQ(r18.shift, 6.0);
}

TEST(PipelineRequest, ParsesParameterVariants) {
  // Bare object, array layers, null seed; schema version 1 renames.
  const auto bare = mi::pipeline::request_from_parameters(
      mi::json::parse(R"({"prompt": "x", "uncond_skip_layers": [24, 26], "slg_layers": " 3 , 4", "seed": null})"));
  EXPECT_EQ(bare.uncond_skip_layers, (std::vector<int>{24, 26}));
  EXPECT_EQ(bare.slg_layers, (std::vector<int>{3, 4}));
  EXPECT_EQ(bare.seed, -1);
  const auto v1 = mi::pipeline::request_from_parameters(
      mi::json::parse(R"({"schema_version": 1, "parameters": {"prompt": "x", "slg_layers": "21", "slg": true}})"));
  EXPECT_EQ(v1.uncond_skip_layers, std::vector<int>{21});
  EXPECT_TRUE(v1.slg_layers.empty());
  const auto v1_off = mi::pipeline::request_from_parameters(
      mi::json::parse(R"({"schema_version": 1, "parameters": {"prompt": "x", "slg_layers": "21", "slg": false}})"));
  EXPECT_TRUE(v1_off.uncond_skip_layers.empty());

  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"height": 1024.5})")), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"uncond_skip_layers": "22,"})")), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"num_images": 2})")), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::request_from_parameters(mi::json::parse(R"({"schema_version": 3, "parameters": {}})")),
               std::invalid_argument);
}

TEST(PipelineRequest, ValidatesLikeTheOfficialApp) {
  mi::pipeline::Request ok;
  ok.prompt = "a cat";
  EXPECT_NO_THROW(mi::pipeline::validate(ok));
  const auto bad = [&](auto mutate) {
    mi::pipeline::Request r = ok;
    mutate(r);
    EXPECT_THROW(mi::pipeline::validate(r), std::invalid_argument);
  };
  bad([](auto& r) { r.prompt = " \n\t"; });
  bad([](auto& r) { r.height = 1000; });
  bad([](auto& r) { r.width = 0; });
  bad([](auto& r) { r.steps = 0; });
  bad([](auto& r) { r.cfg = -1; });
  bad([](auto& r) { r.cfg = INFINITY; });
  bad([](auto& r) { r.shift = 0; });
  bad([](auto& r) { r.seed = -2; });
  bad([](auto& r) { r.seed = std::int64_t{1} << 53; });
  bad([](auto& r) { r.max_sequence_length = 4097; });
  bad([](auto& r) { r.slg_scale = -0.5; });
  bad([](auto& r) { r.slg_start = 0.6, r.slg_stop = 0.5; });
  bad([](auto& r) { r.uncond_skip_layers = {42}; });
  bad([](auto& r) { r.uncond_skip_layers = {22, 22}; });
  bad([](auto& r) { r.slg_layers = {-1}; });
  bad([](auto& r) { r.height = 4112, r.width = 512; });   // side above 4096
  bad([](auto& r) { r.height = 1040, r.width = 4096; });  // 4.26 MP
  bad([](auto& r) { r.height = 2064, r.width = 2048; });  // 4.23 MP
  for (const auto& [h, w] : {std::pair{2048, 2048}, std::pair{1024, 4096}, std::pair{4096, 512}, std::pair{16, 16}}) {
    mi::pipeline::Request r = ok;
    r.height = h, r.width = w;
    EXPECT_NO_THROW(mi::pipeline::validate(r)) << w << "x" << h;
  }
  EXPECT_EQ(mi::pipeline::parse_layer_list(""), std::vector<int>{});
  EXPECT_EQ(mi::pipeline::parse_layer_list("22"), std::vector<int>{22});
  EXPECT_THROW(mi::pipeline::parse_layer_list("22;24"), std::invalid_argument);
  EXPECT_EQ(mi::pipeline::parse_sampler("ab2"), SamplerKind::kAb2);
  EXPECT_THROW(mi::pipeline::parse_sampler("lms"), std::invalid_argument);
}

TEST(PipelineRequest, VramOptionBytes) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0, kMiB = 1024.0 * 1024.0;
  EXPECT_EQ(mi::pipeline::vram_option_bytes(12, kGiB, "--vram-budget"), std::size_t{12} << 30);
  EXPECT_EQ(mi::pipeline::vram_option_bytes(1.5, kGiB, "--vram-budget"), std::size_t{3} << 29);
  EXPECT_EQ(mi::pipeline::vram_option_bytes(384, kMiB, "--vram-reserve"), std::size_t{384} << 20);
  EXPECT_EQ(mi::pipeline::vram_option_bytes(1024, kGiB, "--vram-budget"), mi::pipeline::kMaxVramBytes);
  for (const double bad : {0.0, -1.0, std::nan(""), double{INFINITY}, -double{INFINITY}, 1024.001, 1e300, 1e-9}) {
    EXPECT_THROW(mi::pipeline::vram_option_bytes(bad, kGiB, "--vram-budget"), std::invalid_argument) << bad;
  }
  EXPECT_THROW(mi::pipeline::vram_option_bytes(0.5, kMiB, "--vram-reserve"), std::invalid_argument);  // below 1 MiB
  EXPECT_THROW(mi::pipeline::vram_option_bytes(2e6, kMiB, "--vram-reserve"), std::invalid_argument);  // above 1024 GiB
}

// ---------------------------------------------------------------- PNG

std::uint32_t be32(const std::vector<std::uint8_t>& v, std::size_t at) {
  return (std::uint32_t{v[at]} << 24) | (std::uint32_t{v[at + 1]} << 16) | (std::uint32_t{v[at + 2]} << 8) | v[at + 3];
}

TEST(PipelinePng, StoredDeflateRoundTrips) {
  // 300 x 200 RGB: 180,200 raw scanline bytes, so the zlib stream spans several stored blocks.
  const int w = 300, h = 200;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w * h * 3));
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>((i * 7919) >> 3);
  const auto png = mi::pipeline::encode_png_rgb8(rgb, w, h);
  ASSERT_GT(png.size(), 33u);
  EXPECT_EQ(std::memcmp(png.data(), "\x89PNG\r\n\x1a\n", 8), 0);
  EXPECT_EQ(be32(png, 16), static_cast<std::uint32_t>(w));
  EXPECT_EQ(be32(png, 20), static_cast<std::uint32_t>(h));
  // Walk the chunks and inflate the stored blocks.
  std::size_t at = 8;
  std::vector<std::uint8_t> idat;
  bool end = false;
  while (at + 12 <= png.size() && !end) {
    const std::uint32_t len = be32(png, at);
    const std::string type(png.begin() + static_cast<std::ptrdiff_t>(at + 4), png.begin() + static_cast<std::ptrdiff_t>(at + 8));
    if (type == "IDAT") idat.insert(idat.end(), png.begin() + static_cast<std::ptrdiff_t>(at + 8),
                                    png.begin() + static_cast<std::ptrdiff_t>(at + 8 + len));
    end = type == "IEND";
    at += 12 + len;
  }
  ASSERT_TRUE(end);
  EXPECT_EQ(at, png.size());
  ASSERT_GE(idat.size(), 6u);
  EXPECT_EQ((idat[0] * 256 + idat[1]) % 31, 0);  // zlib header check
  std::vector<std::uint8_t> raw;
  std::size_t p = 2;
  for (bool last = false; !last;) {
    last = idat[p] & 1;
    const std::size_t len = idat[p + 1] | (idat[p + 2] << 8);
    EXPECT_EQ(len ^ 0xffffu, static_cast<std::size_t>(idat[p + 3] | (idat[p + 4] << 8)));
    raw.insert(raw.end(), idat.begin() + static_cast<std::ptrdiff_t>(p + 5), idat.begin() + static_cast<std::ptrdiff_t>(p + 5 + len));
    p += 5 + len;
  }
  ASSERT_EQ(raw.size(), static_cast<std::size_t>(h * (w * 3 + 1)));
  std::uint32_t a = 1, b = 0;
  for (const std::uint8_t v : raw) a = (a + v) % 65521, b = (b + a) % 65521;
  EXPECT_EQ(be32(idat, p), (b << 16) | a);
  for (int y = 0; y < h; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * (w * 3 + 1);
    ASSERT_EQ(raw[row], 0);
    ASSERT_EQ(std::memcmp(raw.data() + row + 1, rgb.data() + static_cast<std::size_t>(y * w * 3), static_cast<std::size_t>(w * 3)), 0);
  }
  EXPECT_THROW(mi::pipeline::encode_png_rgb8(rgb, w, h + 1), std::invalid_argument);

  int dw = 0, dh = 0, dc = 0;
  stbi_uc* decoded = stbi_load_from_memory(png.data(), static_cast<int>(png.size()), &dw, &dh, &dc, 3);
  ASSERT_NE(decoded, nullptr) << stbi_failure_reason();
  EXPECT_EQ(dw, w);
  EXPECT_EQ(dh, h);
  EXPECT_EQ(dc, 3);
  EXPECT_EQ(std::memcmp(decoded, rgb.data(), rgb.size()), 0);
  stbi_image_free(decoded);
}

// A large image through write_png_rgb8: many IDAT chunks of at most 1 MiB, decoded by stb_image from the file;
// the temporary file is renamed away and an existing file is replaced.
TEST(PipelinePng, LargeImageFileRoundTrip) {
  const int w = 2048, h = 1536;  // 9.4 MB of scanlines
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w) * h * 3);
  std::mt19937 rng(11);
  for (auto& v : rgb) v = static_cast<std::uint8_t>(rng());
  const fs::path dir = fs::temp_directory_path() / ("mossinfer_png_" + std::to_string(std::random_device{}()));
  fs::create_directories(dir);
  const fs::path file = dir / "out.png";
  { std::ofstream(file) << "stale"; }
  mi::pipeline::write_png_rgb8(file, rgb, w, h);
  std::vector<fs::path> entries(fs::directory_iterator(dir), fs::directory_iterator{});
  EXPECT_EQ(entries, std::vector<fs::path>{file});  // no temporary left behind

  std::ifstream in(file, std::ios::binary);
  const std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();  // Windows: remove_all below fails on an open file
  int idat_chunks = 0;
  std::uint32_t max_idat = 0;
  for (std::size_t at = 8; at + 12 <= png.size();) {
    const std::uint32_t len = be32(png, at);
    if (std::memcmp(png.data() + at + 4, "IDAT", 4) == 0) ++idat_chunks, max_idat = std::max(max_idat, len);
    at += 12 + len;
  }
  EXPECT_GE(idat_chunks, 9);
  EXPECT_LE(max_idat, 1u << 20);

  int dw = 0, dh = 0, dc = 0;
  stbi_uc* decoded = stbi_load(file.string().c_str(), &dw, &dh, &dc, 3);
  ASSERT_NE(decoded, nullptr) << stbi_failure_reason();
  EXPECT_EQ(dw, w);
  EXPECT_EQ(dh, h);
  EXPECT_EQ(std::memcmp(decoded, rgb.data(), rgb.size()), 0);
  stbi_image_free(decoded);
  fs::remove_all(dir);
}

// ---------------------------------------------------------------- real weights

struct Cmp {
  double cosine = 0, rel_l2 = 0, max_abs = 0;
};
Cmp compare(const std::vector<float>& got, const std::vector<float>& ref) {
  double dot = 0, gg = 0, rr = 0, dd = 0, mx = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double g = got[i], r = ref[i];
    dot += g * r, gg += g * g, rr += r * r, dd += (g - r) * (g - r);
    mx = std::max(mx, std::abs(g - r));
  }
  return {dot / std::sqrt(gg * rr), std::sqrt(dd / rr), mx};
}

std::uint16_t to_bf16(float v) {
  std::uint32_t u = bits(v);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}

// Loop vectors: the reference run that a case reproduces.
struct LoopCase {
  std::string id, embeds, file;
  int height, width, steps;
  double shift;
  std::uint64_t seed;
  Guidance guidance;
  SamplerKind sampler;
};

std::vector<LoopCase> loop_cases() {
  std::vector<LoopCase> out;
  const auto manifest = read_json(data_dir() / "manifest.json");
  for (const auto& e : manifest.at("loops").as_array()) {
    LoopCase c;
    c.id = e.at("id").as_string();
    c.embeds = e.at("embeds").as_string();
    c.file = e.at("file").as_string();
    c.height = static_cast<int>(e.at("height").as_int());
    c.width = static_cast<int>(e.at("width").as_int());
    c.steps = static_cast<int>(e.at("steps").as_int());
    c.shift = e.at("shift").as_double();
    c.seed = static_cast<std::uint64_t>(e.at("seed").as_int());
    c.guidance.cfg = e.at("cfg").as_double();
    for (const auto& l : e.at("uncond_skip_layers").as_array()) c.guidance.uncond_skip.set(static_cast<std::size_t>(l.as_int()));
    for (const auto& l : e.at("slg_layers").as_array()) c.guidance.slg_layers.set(static_cast<std::size_t>(l.as_int()));
    c.guidance.slg_scale = e.at("slg_scale").as_double();
    c.guidance.slg_start = e.at("slg_start").as_double();
    c.guidance.slg_stop = e.at("slg_stop").as_double();
    c.sampler = mi::pipeline::parse_sampler(e.at("sampler").as_string());
    out.push_back(c);
  }
  return out;
}

bool have_weights() { return fs::is_regular_file(weights_root() / "base" / "moss.mossidx"); }

const fs::path kInt8Dit = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
const fs::path kKohyaLora = "/workspace/Models/community/congruency/MOSS-Image-Turbo-MFD/MOSS-Image-SFT-Turbo-MFD-v0.1.comfy.safetensors";
const fs::path kPeftLora = "/workspace/Models/community/axiomofmind/Moss-image-beauty-lora/pytorch_lora_weights.safetensors";
bool have_lora_files() {
  return fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") && fs::is_regular_file(kInt8Dit) &&
         fs::is_regular_file(kKohyaLora) && fs::is_regular_file(kPeftLora);
}
mi::pipeline::Request lora_request() {
  mi::pipeline::Request req;
  req.model = weights_root() / "sft";
  req.prompt = "portrait of a woman in a garden, soft light";
  req.height = req.width = 256;
  req.steps = 3;
  req.decode = false;
  return req;
}


// DiT + sampler loop on golden text-encoder states (no text-encoder drift) vs the reference loop.
class PipelineLoop : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!have_weights()) return;
    const fs::path bundle = weights_root() / "base";
    index_ = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(bundle / "moss.mossidx", bundle));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    std::size_t stack = 0;
    for (const LoopCase& c : loop_cases()) {
      const auto sc = mi::npy::load(golden_dir() / (c.embeds + ".cond.te_last_hidden.npy")).shape[0];
      const auto su = mi::npy::load(golden_dir() / (c.embeds + ".uncond.te_last_hidden.npy")).shape[0];
      const std::size_t refine = 2 * mi::runtime::align_up(static_cast<std::size_t>(std::max(sc, su)) * mi::moss_dit::kTextDim * 2) +
                                 mi::moss_dit::MossDit::encode_text_workspace_bytes(std::max(sc, su));
      const std::size_t loop = mi::runtime::align_up(static_cast<std::size_t>(128 * (c.height / 16) * (c.width / 16)) * 4) +
                               mi::pipeline::denoise_workspace_bytes(c.height / 16, c.width / 16, sc, su, c.guidance, c.sampler);
      stack = std::max(stack, 2 * mi::runtime::align_up(static_cast<std::size_t>(std::max(sc, su)) * mi::moss_dit::kDim * 2) +
                                  std::max(refine, loop));
    }
    arena_ = std::make_unique<mi::runtime::DeviceArena>(mi::moss_dit::MossDit::persistent_bytes(*index_) + stack);
    dit_ = std::make_unique<mi::moss_dit::MossDit>(*index_, *arena_, *streams_);
  }
  static void TearDownTestSuite() {
    dit_.reset();
    arena_.reset();
    streams_.reset();
    index_.reset();
  }
  void SetUp() override {
    if (!dit_) GTEST_SKIP() << "weights not present (" << weights_root() << ")";
  }

  // Refined text of a golden text-encoder state (fp32 copy of bf16, exact) on the arena stack.
  Tensor refine(const std::string& file) {
    const auto arr = mi::npy::load(golden_dir() / file);
    const std::int64_t s = arr.shape[0];
    const auto v = arr.to_vector<float>();
    std::vector<std::uint16_t> h(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) h[i] = to_bf16(v[i]);
    const Tensor out = arena_->alloc_tensor(mi::DType::kBF16, {s, mi::moss_dit::kDim});
    mi::runtime::StackScope scope(*arena_);
    const Tensor te = arena_->alloc_tensor(mi::DType::kBF16, {s, mi::moss_dit::kTextDim});
    // Stream-ordered: the compute stream is non-blocking, so a plain cudaMemcpy would not order before it.
    MI_CUDA_CHECK(cudaMemcpyAsync(te.data, h.data(), te.nbytes(), cudaMemcpyHostToDevice, streams_->compute()));
    dit_->encode_text(te, out, streams_->compute());
    MI_CUDA_CHECK(cudaStreamSynchronize(streams_->compute()));
    return out;
  }

  static std::unique_ptr<mi::weights::ModelIndex> index_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<mi::moss_dit::MossDit> dit_;
};

std::unique_ptr<mi::weights::ModelIndex> PipelineLoop::index_;
std::unique_ptr<mi::runtime::StreamSet> PipelineLoop::streams_;
std::unique_ptr<mi::runtime::DeviceArena> PipelineLoop::arena_;
std::unique_ptr<mi::moss_dit::MossDit> PipelineLoop::dit_;

// Few-step runs are chaotic: per-forward differences far below AC5's bound are amplified by guidance (g up to 4) and
// large steps. The bound is the reference against itself (same loop with SDPA's memory-efficient kernel instead of
// flash, manifest "loop_noise_floor"): 1 - cosine may be at most 2x the reference's own drift (the AC4 v5 policy).
double floor_cosine(const std::string& id) {
  return read_json(data_dir() / "manifest.json").at("loop_noise_floor").at("cases").at(id).at("cos_latent").as_double();
}
constexpr double kFloorFactor = 2.0;

TEST_F(PipelineLoop, MatchesReferenceLoopsOnGoldenEmbeddings) {
  for (const LoopCase& c : loop_cases()) {
    mi::runtime::StackScope scope(*arena_);
    const Tensor cond = refine(c.embeds + ".cond.te_last_hidden.npy");
    const Tensor uncond = c.guidance.do_cfg() ? refine(c.embeds + ".uncond.te_last_hidden.npy") : Tensor{};
    const std::int64_t h = c.height / 16, w = c.width / 16;
    const Tensor latents = arena_->alloc_tensor(mi::DType::kF32, {128, h, w});
    const auto noise = mi::rng::torch_cpu_randn(c.seed, latents.numel());
    MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, noise.data(), latents.nbytes(), cudaMemcpyHostToDevice, streams_->compute()));
    const Schedule sched = mi::pipeline::moss_schedule(c.steps, c.shift);
    int callbacks = 0;
    const int run = mi::pipeline::denoise(*dit_, *arena_, cond, uncond, sched, c.guidance, c.sampler, latents,
                                          streams_->compute(), [&](const mi::pipeline::StepInfo& s) {
                                            EXPECT_EQ(s.step, ++callbacks);
                                            EXPECT_EQ(s.num_steps, sched.num_steps());
                                            return true;
                                          });
    EXPECT_EQ(run, sched.num_steps());
    std::vector<float> got(static_cast<std::size_t>(latents.numel()));
    MI_CUDA_CHECK(cudaMemcpy(got.data(), latents.data, latents.nbytes(), cudaMemcpyDeviceToHost));
    const auto ref = mi::npy::load(data_dir() / c.file).to_vector<float>();
    ASSERT_EQ(got.size(), ref.size());
    const Cmp cmp = compare(got, ref);
    const double floor = floor_cosine(c.id);
    std::printf("[pipeline loop] %-24s %4dx%-4d %d steps %-5s: cosine %.6f (reference vs itself %.6f)  rel-L2 %.3e  "
                "max|diff| %.4g\n",
                c.id.c_str(), c.width, c.height, sched.num_steps(), mi::pipeline::sampler_name(c.sampler), cmp.cosine, floor,
                cmp.rel_l2, cmp.max_abs);
    RecordProperty(c.id + "_cosine", std::to_string(cmp.cosine));
    EXPECT_LE(1.0 - cmp.cosine, kFloorFactor * (1.0 - floor)) << c.id;
  }
}

TEST_F(PipelineLoop, CallbackCancelsBetweenSteps) {
  mi::runtime::StackScope scope(*arena_);
  const Tensor cond = refine("g_short_1024.cond.te_last_hidden.npy");
  const Tensor latents = arena_->alloc_tensor(mi::DType::kF32, {128, 16, 16});
  const auto noise = mi::rng::torch_cpu_randn(1, latents.numel());
  MI_CUDA_CHECK(cudaMemcpyAsync(latents.data, noise.data(), latents.nbytes(), cudaMemcpyHostToDevice, streams_->compute()));
  Guidance g;
  g.cfg = 1.0;
  const Schedule sched = mi::pipeline::moss_schedule(10, 5.0);
  const int run = mi::pipeline::denoise(*dit_, *arena_, cond, Tensor{}, sched, g, SamplerKind::kEuler, latents,
                                        streams_->compute(), [](const mi::pipeline::StepInfo& s) { return s.step < 3; });
  EXPECT_EQ(run, 3);
}

// The whole pipeline (own tokenizer + text encoder + refiner + loop), latents only, vs the 4-step reference run on
// the golden prompt. Adds the text encoder's AC4-bounded drift to the loop's.
TEST(PipelineEndToEnd, FourStepLatentsMatchReference) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  const auto cases = read_json(golden_dir() / "cases.json");
  mi::pipeline::Request req;
  for (const auto& c : cases.as_array()) {
    if (c.at("id").as_string() != "g_fridge_1024") continue;
    req.prompt = c.at("prompt").as_string();
    req.negative_prompt = c.at("negative_prompt").as_string();
    req.seed = c.at("seed").as_int();
  }
  req.model = weights_root() / "base";
  req.steps = 4;
  req.decode = false;
  mi::pipeline::PipelineOptions ref_options;
  ref_options.tier = mi::planner::Tier::kReference;  // AC11: the Reference tier, pinned
  mi::pipeline::Pipeline pipe(ref_options);
  std::vector<int> seen;
  mi::pipeline::GenerateControl control;
  control.on_step = [&](const mi::pipeline::StepInfo& s) {
    seen.push_back(s.step);
    return true;
  };
  const mi::pipeline::Result r = pipe.generate(req, control);
  EXPECT_EQ(seen, (std::vector<int>{1, 2, 3, 4}));
  EXPECT_FALSE(r.cancelled);
  EXPECT_EQ(r.steps_run, 4);
  EXPECT_EQ(r.cond_tokens, static_cast<int>(mi::npy::load(golden_dir() / "g_fridge_1024.cond.token_ids.npy").numel()));
  EXPECT_EQ(r.uncond_tokens, static_cast<int>(mi::npy::load(golden_dir() / "g_fridge_1024.uncond.token_ids.npy").numel()));
  EXPECT_TRUE(r.rgb.empty());
  const auto ref = mi::npy::load(data_dir() / "loop_fridge_1024_euler4.latent.npy").to_vector<float>();
  ASSERT_EQ(r.latents.size(), ref.size());
  const Cmp cmp = compare(r.latents, ref);
  const double floor = floor_cosine("fridge_1024_euler4");
  const auto& t = r.timings;
  std::printf("[pipeline e2e] fridge 1024^2 4 steps (own TE): cosine %.6f (reference vs itself %.6f)  rel-L2 %.3e  "
              "max|diff| %.4g\n",
              cmp.cosine, floor, cmp.rel_l2, cmp.max_abs);
  std::printf("[pipeline e2e] index %.2f s, tokenize %.4f s, TE %.2f s (weights %.2f + compute %.2f), DiT load %.2f s, "
              "refine %.3f s, denoise %.2f s; arena %.2f GiB, high-water %.2f GiB, device peak %.2f GiB\n",
              t.index_load_s, t.tokenize_s, t.text_encode_s, t.te_weight_load_s, t.te_compute_s, t.dit_load_s, t.refine_s,
              t.denoise_s, static_cast<double>(r.memory.arena_capacity) / (1 << 30),
              static_cast<double>(r.memory.arena_high_water) / (1 << 30), static_cast<double>(r.memory.device_peak) / (1 << 30));
  RecordProperty("cosine", std::to_string(cmp.cosine));
  EXPECT_LE(1.0 - cmp.cosine, kFloorFactor * (1.0 - floor));

  // Second call: DiT stays resident, so no reload; cancellation stops between steps.
  std::atomic<bool> cancel{false};
  mi::pipeline::GenerateControl stop;
  stop.cancel = &cancel;
  stop.on_step = [&](const mi::pipeline::StepInfo& s) {
    if (s.step == 2) cancel.store(true);
    return true;
  };
  const mi::pipeline::Result r2 = pipe.generate(req, stop);
  EXPECT_TRUE(r2.cancelled);
  EXPECT_EQ(r2.steps_run, 2);
  EXPECT_EQ(r2.timings.dit_load_s, 0.0);
  EXPECT_EQ(r2.memory.arena_creations, 0);

  mi::pipeline::Request bad = req;
  bad.width = 1000;
  EXPECT_THROW(pipe.generate(bad), std::invalid_argument);

  // Third call with the VAE decode: untiled, or exact tiled when the fp32 decode does not fit next to the resident
  // DiT (parking only if the plan finds nothing else); the sampling result must not depend on that.
  req.decode = true;
  const mi::pipeline::Result r3 = pipe.generate(req);
  ASSERT_EQ(r3.rgb.size(), static_cast<std::size_t>(1024 * 1024 * 3));
  EXPECT_EQ(bit_mismatches(r3.latents, r.latents), 0u);  // deterministic run to run
  double mean = 0, var = 0;
  for (const std::uint8_t v : r3.rgb) mean += v;
  mean /= static_cast<double>(r3.rgb.size());
  for (const std::uint8_t v : r3.rgb) var += (v - mean) * (v - mean);
  const double stddev = std::sqrt(var / static_cast<double>(r3.rgb.size()));
  std::printf("[pipeline e2e] decode %.2f s (VAE load %.2f s), RGB mean %.1f std %.1f, device peak %.2f GiB, "
              "arena high-water %.2f GiB, DiT parked for decode: %s\n",
              r3.timings.decode_s, r3.timings.vae_load_s, mean, stddev, static_cast<double>(r3.memory.device_peak) / (1 << 30),
              static_cast<double>(r3.memory.arena_high_water) / (1 << 30), r3.memory.dit_parked_for_decode ? "yes" : "no");
  EXPECT_GT(stddev, 10.0);  // an image, not a constant or saturated buffer
  const mi::pipeline::Result r4 = pipe.generate(req);
  EXPECT_EQ(bit_mismatches(r4.latents, r.latents), 0u);
  EXPECT_EQ(r4.rgb, r3.rgb);
  EXPECT_EQ(r4.timings.dit_load_s > 0, r3.memory.dit_parked_for_decode);
}

// A budget below what some phase needs fails before any phase starts (no text encoding or weight loading first).
TEST(PipelineEndToEnd, BudgetBelowMinimumFailsBeforeTextEncoder) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  mi::pipeline::PipelineOptions options;
  options.vram_budget = std::size_t{1} << 30;  // even a fully streamed DiT keeps ~0.64 GiB resident + 2 block buffers
  mi::pipeline::Pipeline pipe(options);
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a cat";
  std::vector<mi::pipeline::Phase> seen;
  mi::pipeline::GenerateControl control;
  control.on_phase = [&](mi::pipeline::Phase p) {
    seen.push_back(p);
    return true;
  };
  bool planned = false;
  control.on_plan = [&](const mi::planner::ExecutionPlan&) { planned = true; };
  try {
    pipe.generate(req, control);
    FAIL() << "expected planner::BudgetError";
  } catch (const mi::planner::BudgetError& e) {
    EXPECT_GT(e.required_arena, e.arena_cap);
    EXPECT_NE(std::string(e.what()).find("GiB"), std::string::npos) << e.what();
  }
  EXPECT_TRUE(seen.empty());
  EXPECT_FALSE(planned);
}

// The embedding cache: a repeated prompt skips the text encoder with bit-identical latents, in memory and, through
// the disk spill, in a new pipeline (a later process). The plan is reported and keeps the DiT resident.
TEST(PipelineEndToEnd, EmbeddingCacheSkipsTextEncoderExactly) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  mi::testing::TempDir dir("pipe_emb_cache");
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a lighthouse on a cliff at night";
  req.negative_prompt = "blurry";
  req.height = req.width = 256;
  req.steps = 2;
  req.decode = false;
  mi::pipeline::PipelineOptions options;
  options.cache_dir = dir.path() / "cache";
  mi::pipeline::Result first, second;
  {
    mi::pipeline::Pipeline pipe(options);
    int plans = 0;
    mi::pipeline::GenerateControl control;
    control.on_plan = [&](const mi::planner::ExecutionPlan& p) {
      ++plans;
      EXPECT_FALSE(p.explanation.empty());
    };
    first = pipe.generate(req, control);
    EXPECT_EQ(plans, 1);
    EXPECT_EQ(first.te_cache_hits, 0);
    EXPECT_FALSE(first.te_skipped);
    EXPECT_TRUE(first.plan.te.run);
    EXPECT_GT(first.timings.text_encode_s, 0.0);
    EXPECT_EQ(first.plan.dit.residency, mi::planner::DitResidency::kResident);
    second = pipe.generate(req, control);
    EXPECT_TRUE(second.te_skipped);
    EXPECT_EQ(second.te_cache_hits, 2);
    EXPECT_FALSE(second.plan.te.run);
    EXPECT_EQ(second.timings.text_encode_s, 0.0);
    EXPECT_EQ(bit_mismatches(second.latents, first.latents), 0u);
    // A new prompt with the same negative: only the prompt is encoded (the negative is kept).
    mi::pipeline::Request other = req;
    other.prompt = "a red bicycle";
    const mi::pipeline::Result third = pipe.generate(other);
    EXPECT_EQ(third.te_cache_hits, 1);
    EXPECT_FALSE(third.te_skipped);
  }
  mi::pipeline::Pipeline fresh(options);
  const mi::pipeline::Result again = fresh.generate(req);
  EXPECT_TRUE(again.te_skipped);
  EXPECT_EQ(bit_mismatches(again.latents, first.latents), 0u);
}

// Cancellation before each phase (on_phase returning false, or the cancel flag) and at the last step: the call ends
// there, reports cancelled, and never decodes after a cancel.
TEST(PipelineEndToEnd, CancelsAtEveryPhase) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  using mi::pipeline::Phase;
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a red apple on a table";
  req.negative_prompt = "blurry";
  req.height = req.width = 256;
  req.steps = 2;
  mi::pipeline::Pipeline pipe;
  std::atomic<bool> cancel{false};
  std::vector<Phase> seen;
  const auto run = [&](std::function<bool(Phase)> on_phase, mi::pipeline::StepCallback on_step = {}) {
    seen.clear();
    cancel.store(false);
    mi::pipeline::GenerateControl control;
    control.cancel = &cancel;
    control.on_phase = [&, on_phase](Phase p) {
      seen.push_back(p);
      return on_phase(p);
    };
    control.on_step = std::move(on_step);
    return pipe.generate(req, control);
  };
  const std::size_t latent_numel = 128 * (256 / 16) * (256 / 16);

  // Before the text encoder.
  mi::pipeline::Result r = run([](Phase p) { return p != Phase::kTextEncoder; });
  EXPECT_TRUE(r.cancelled);
  EXPECT_EQ(seen, std::vector<Phase>{Phase::kTextEncoder});
  EXPECT_EQ(r.timings.text_encode_s, 0.0);
  EXPECT_TRUE(r.latents.empty());
  EXPECT_TRUE(r.rgb.empty());

  // After the text encoder, before the DiT load: the flag, set while the phase callback runs.
  r = run([&](Phase p) {
    if (p == Phase::kDitLoad) cancel.store(true);
    return true;
  });
  EXPECT_TRUE(r.cancelled);
  EXPECT_EQ(seen, (std::vector<Phase>{Phase::kTextEncoder, Phase::kDitLoad}));
  EXPECT_GT(r.timings.text_encode_s, 0.0);
  EXPECT_EQ(r.timings.dit_load_s, 0.0);
  EXPECT_TRUE(r.latents.empty());

  // After the DiT load, before sampling.
  r = run([](Phase p) { return p != Phase::kDenoise; });
  EXPECT_TRUE(r.cancelled);
  EXPECT_EQ(seen, (std::vector<Phase>{Phase::kTextEncoder, Phase::kDitLoad, Phase::kDenoise}));
  EXPECT_EQ(r.steps_run, 0);
  EXPECT_TRUE(r.timings.step_s.empty());
  EXPECT_TRUE(r.latents.empty());

  // On the last step: every step ran, but no decode.
  r = run([](Phase) { return true; }, [](const mi::pipeline::StepInfo& s) { return s.step < s.num_steps; });
  EXPECT_TRUE(r.cancelled);
  EXPECT_EQ(r.steps_run, r.steps_scheduled);
  EXPECT_EQ(r.latents.size(), latent_numel);
  EXPECT_TRUE(r.rgb.empty());
  EXPECT_EQ(std::count(seen.begin(), seen.end(), Phase::kDecode), 0);
  EXPECT_EQ(r.timings.vae_load_s, 0.0);

  // Before the decode, by the flag.
  r = run([&](Phase p) {
    if (p == Phase::kDecode) cancel.store(true);
    return true;
  });
  EXPECT_TRUE(r.cancelled);
  EXPECT_EQ(seen.back(), Phase::kDecode);
  EXPECT_EQ(r.steps_run, r.steps_scheduled);
  EXPECT_EQ(r.latents.size(), latent_numel);
  EXPECT_TRUE(r.rgb.empty());
  EXPECT_EQ(r.timings.vae_load_s, 0.0);

  // Not cancelled: decodes.
  r = run([](Phase) { return true; });
  EXPECT_FALSE(r.cancelled);
  EXPECT_EQ(seen, (std::vector<Phase>{Phase::kTextEncoder, Phase::kDitLoad, Phase::kDenoise, Phase::kDecode}));
  EXPECT_EQ(r.rgb.size(), static_cast<std::size_t>(256 * 256 * 3));
  EXPECT_LE(r.memory.arena_high_water, r.plan.arena_bytes);
}

// A decode run, then a latents-only run in the arena sized exactly by the first plan: the VAE decoder stays resident,
// and the second plan counts it, so the second run fits as planned (high-water within the plan's arena), throws
// nothing and keeps the DiT (no reload, reported or not).
TEST(PipelineEndToEnd, LatentsOnlyAfterDecodeCountsResidentVae) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a paper boat on a pond";
  req.negative_prompt = "blurry";
  req.height = req.width = 256;
  req.steps = 2;
  mi::pipeline::Pipeline pipe;
  const mi::pipeline::Result a = pipe.generate(req);
  ASSERT_FALSE(a.rgb.empty());
  EXPECT_LE(a.memory.arena_high_water, a.plan.arena_bytes);
  EXPECT_LE(a.memory.arena_capacity, a.plan.arena_bytes);  // the arena is the plan's, nothing to spare
  req.decode = false;
  mi::pipeline::Result b;
  ASSERT_NO_THROW(b = pipe.generate(req));
  EXPECT_FALSE(b.plan.vae.run);
  EXPECT_TRUE(b.te_skipped);
  EXPECT_EQ(b.plan.peaks.denoise, a.plan.peaks.denoise);  // the resident decoder is counted
  EXPECT_LE(b.memory.arena_high_water, b.plan.arena_bytes);
  EXPECT_FALSE(b.memory.dit_reloaded);
  EXPECT_EQ(b.timings.dit_load_s, 0.0);
  EXPECT_EQ(b.memory.arena_creations, 0);
  EXPECT_EQ(bit_mismatches(b.latents, a.latents), 0u);
  // And back: the decoder is still there (no reload either).
  req.decode = true;
  const mi::pipeline::Result c = pipe.generate(req);
  EXPECT_EQ(c.timings.vae_load_s, 0.0);
  EXPECT_EQ(c.memory.arena_creations, 0);
  EXPECT_EQ(c.rgb, a.rgb);
}

// Sets a cancel flag from another thread after a delay; records when.
struct DelayedCancel {
  explicit DelayedCancel(std::atomic<bool>& f) : flag(f) {}
  std::atomic<bool>& flag;
  std::thread thread;
  std::chrono::steady_clock::time_point set_at{};
  void arm(double seconds) {
    thread = std::thread([this, seconds] {
      std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
      set_at = std::chrono::steady_clock::now();
      flag.store(true);
    });
  }
  // Seconds from the flag to `t` (after join()).
  double until(std::chrono::steady_clock::time_point t) const { return std::chrono::duration<double>(t - set_at).count(); }
  void join() {
    if (thread.joinable()) thread.join();
  }
  ~DelayedCancel() { join(); }
};

// AC12: a cancel requested inside a phase lands within about one unit of it (text-encoder layer, DiT weight group,
// DiT block, tiled-decode band), reports cancelled, and the next generate() is bit-identical to an uncancelled run.
// 1024^2 with a budget that keeps the DiT resident but leaves no room for the untiled decode, so the decode is tiled;
// the embedding cache is off so every run encodes.
TEST(PipelineEndToEnd, CancelsWithinPhases) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  using Clock = std::chrono::steady_clock;
  using mi::pipeline::Phase;
  const fs::path bundle = weights_root() / "base";
  const mi::weights::ModelIndex index = mi::weights::load_index(bundle / "moss.mossidx", bundle);
  mi::pipeline::Request req;
  req.model = bundle;
  req.prompt = "a lighthouse on a rocky coast at dusk, oil painting";
  req.negative_prompt = "blurry";
  req.steps = 2;
  mi::pipeline::PipelineOptions options;
  options.embedding_cache = false;
  // Arena cap = DiT + VAE + 1.25 GiB (text encoder ~0.9 GiB, denoise ~0.4 GiB; the untiled decode needs ~1.6 GiB).
  const std::size_t cap = mi::moss_dit::MossDit::persistent_bytes(index) + mi::flux2_vae::Flux2VaeDecoder::persistent_bytes(index) +
                          (std::size_t{1280} << 20);
  const mi::planner::Budgets defaults;
  options.vram_budget = cap + defaults.context_bytes + options.vram_reserve;
  mi::pipeline::Pipeline pipe(options);
  std::atomic<bool> cancel{false};

  // Mid-DiT-load (the first load of this pipeline): stops at a weight group, nothing resident afterwards.
  {
    cancel.store(false);
    DelayedCancel dc{cancel};
    mi::pipeline::GenerateControl control;
    control.cancel = &cancel;
    control.on_phase = [&](Phase p) {
      if (p == Phase::kDitLoad) dc.arm(0.15);
      return true;
    };
    const mi::pipeline::Result r = pipe.generate(req, control);
    const auto done = Clock::now();
    dc.join();
    EXPECT_TRUE(r.cancelled);
    EXPECT_EQ(r.timings.dit_load_s, 0.0);  // the load never completed
    EXPECT_TRUE(r.latents.empty());
    std::printf("[pipeline cancel] mid-DiT-load: returned %.0f ms after the flag\n", dc.until(done) * 1e3);
    EXPECT_LT(dc.until(done), 1.5);
  }

  // Reference: uncancelled (the DiT loads now), tiled decode as planned.
  const mi::pipeline::Result ref = pipe.generate(req);
  ASSERT_FALSE(ref.cancelled);
  ASSERT_EQ(ref.plan.dit.residency, mi::planner::DitResidency::kResident) << ref.plan.explanation;
  ASSERT_TRUE(ref.decode_tiled) << ref.plan.explanation;
  EXPECT_TRUE(ref.plan.vae.kind == mi::planner::VaeDecodeKind::kTiled);
  EXPECT_LE(ref.memory.arena_high_water, ref.plan.arena_bytes);
  ASSERT_EQ(ref.timings.step_s.size(), 2u);
  const auto expect_like_ref = [&](const char* after) {
    const mi::pipeline::Result again = pipe.generate(req);
    EXPECT_FALSE(again.cancelled) << after;
    EXPECT_EQ(bit_mismatches(again.latents, ref.latents), 0u) << after;
    EXPECT_TRUE(again.rgb == ref.rgb) << after;
    EXPECT_FALSE(again.memory.dit_reloaded) << after;
  };
  expect_like_ref("mid-DiT-load cancel");

  // Latents after one step (cancelled between steps by the callback).
  mi::pipeline::GenerateControl one_step;
  one_step.on_step = [](const mi::pipeline::StepInfo& s) { return s.step < 1; };
  const mi::pipeline::Result after1 = pipe.generate(req, one_step);
  ASSERT_EQ(after1.steps_run, 1);

  const auto cancel_in = [&](Phase phase, double delay_s, const char* what, auto&& check) {
    cancel.store(false);
    DelayedCancel dc{cancel};
    mi::pipeline::GenerateControl control;
    control.cancel = &cancel;
    control.on_phase = [&](Phase p) {
      if (p == phase && phase != Phase::kDenoise) dc.arm(delay_s);
      return true;
    };
    if (phase == Phase::kDenoise) {
      control.on_step = [&](const mi::pipeline::StepInfo& s) {
        if (s.step == 1) dc.arm(delay_s);  // lands inside step 2
        return true;
      };
    }
    const mi::pipeline::Result r = pipe.generate(req, control);
    const auto done = Clock::now();
    dc.join();
    EXPECT_TRUE(r.cancelled) << what;
    std::printf("[pipeline cancel] %s: returned %.0f ms after the flag\n", what, dc.until(done) * 1e3);
    check(r, dc.until(done));
    expect_like_ref(what);
  };

  cancel_in(Phase::kTextEncoder, 0.1, "mid-text-encoder", [&](const mi::pipeline::Result& r, double latency) {
    EXPECT_GT(r.timings.text_encode_s, 0.05);
    EXPECT_TRUE(r.latents.empty());
    EXPECT_LT(latency, 0.5);
  });
  const double step_s = ref.timings.step_s[1];
  cancel_in(Phase::kDenoise, 0.4 * step_s, "mid-denoise-step", [&](const mi::pipeline::Result& r, double latency) {
    EXPECT_EQ(r.steps_run, 1);
    EXPECT_EQ(bit_mismatches(r.latents, after1.latents), 0u);  // the latents after the last finished step
    EXPECT_TRUE(r.rgb.empty());
    EXPECT_LT(latency, 0.3 * step_s + 0.1);  // about one block of a 42-block step
  });
  cancel_in(Phase::kDecode, 0.3 * ref.timings.decode_s, "mid-tiled-decode",  // the decoder is resident by now
            [&](const mi::pipeline::Result& r, double latency) {
              EXPECT_EQ(r.steps_run, 2);
              EXPECT_EQ(bit_mismatches(r.latents, ref.latents), 0u);
              EXPECT_TRUE(r.rgb.empty());
              EXPECT_LT(latency, 0.25 * ref.timings.decode_s + 0.1);
            });
}

// The image does not depend on the VRAM budget: one seed decoded untiled (default budget) and tiled (an arena cap that
// leaves no room for the untiled decode next to the resident DiT, as in CancelsWithinPhases) gives the same RGB bytes.
TEST(PipelineEndToEnd, TiledAndUntiledDecodeGiveTheSameImage) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  const fs::path bundle = weights_root() / "base";
  const mi::weights::ModelIndex index = mi::weights::load_index(bundle / "moss.mossidx", bundle);
  mi::pipeline::Request req;
  req.model = bundle;
  req.prompt = "a lighthouse on a rocky coast at dusk, oil painting";
  req.negative_prompt = "blurry";
  req.steps = 2;
  mi::pipeline::PipelineOptions options;
  options.embedding_cache = false;
  const mi::planner::Budgets defaults;
  // The default budget is the free device memory, and a larger vram_budget cannot add memory, only cap it. So the
  // untiled run may tile only on a GPU without room for the resident DiT plus the untiled decode (1 GiB of slack for
  // the planner's other terms); with room, tiling would be a planner regression.
  const std::size_t untiled_need =
      mi::moss_dit::MossDit::persistent_bytes(index) + mi::flux2_vae::Flux2VaeDecoder::persistent_bytes(index) +
      mi::flux2_vae::Flux2VaeDecoder::workspace_bytes(mi::flux2_vae::decoder_config(index), 1024, 1024) +
      defaults.context_bytes + options.vram_reserve +
      (std::size_t{1} << 30);
  const bool room_for_untiled = mi::runtime::DeviceArena::free_device_bytes() >= untiled_need;
  mi::pipeline::Result untiled;
  {
    mi::pipeline::Pipeline pipe(options);
    untiled = pipe.generate(req);
  }
  if (untiled.decode_tiled) {
    ASSERT_FALSE(room_for_untiled) << "free memory fits the untiled decode, but the plan tiled it: "
                                   << untiled.plan.explanation;
    GTEST_SKIP() << "the untiled 1024^2 decode does not fit in this GPU's free memory (the plan tiled it): "
                 << untiled.plan.explanation;
  }
  const std::size_t cap = mi::moss_dit::MossDit::persistent_bytes(index) + mi::flux2_vae::Flux2VaeDecoder::persistent_bytes(index) +
                          (std::size_t{1280} << 20);
  options.vram_budget = cap + defaults.context_bytes + options.vram_reserve;
  mi::pipeline::Pipeline pipe(options);
  const mi::pipeline::Result tiled = pipe.generate(req);
  std::printf("[pipeline decode] untiled %.2f s (%s); tiled %.2f s (%s)\n", untiled.timings.decode_s,
              untiled.decode_schedule.c_str(), tiled.timings.decode_s, tiled.decode_schedule.c_str());
  ASSERT_TRUE(tiled.decode_tiled) << tiled.plan.explanation;
  EXPECT_EQ(bit_mismatches(tiled.latents, untiled.latents), 0u);
  ASSERT_EQ(untiled.rgb.size(), static_cast<std::size_t>(1024 * 1024 * 3));
  ASSERT_EQ(tiled.rgb.size(), untiled.rgb.size());
  std::size_t differ = 0;
  for (std::size_t i = 0; i < untiled.rgb.size(); ++i) differ += tiled.rgb[i] != untiled.rgb[i];
  EXPECT_EQ(differ, 0u) << "of " << untiled.rgb.size() << " bytes";
}

// Free device memory grows between two requests (another application exits): the plan for the second would like a
// larger arena (the whole DiT resident), which means dropping and reloading the loaded, partly streamed DiT. The
// pipeline plans inside the held arena instead when that costs less than the reload: no new arena, no reload, the
// same residency and the same bits. (P3's flaky dit_reloaded came from this: a free-memory change between calls.)
TEST(PipelineEndToEnd, KeepsHeldArenaInsteadOfReloadingTheDit) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a lighthouse on a rocky coast at dusk, oil painting";
  req.negative_prompt = "blurry";
  req.height = req.width = 512;
  req.steps = 3;
  mi::pipeline::PipelineOptions options;
  // Fixed rates (the dev box's, warm page cache): the keep-or-grow decision must not depend on what the page cache
  // holds while other tests run.
  mi::planner::HardwareProfile hw;
  hw.h2d_gbps = 24.0;
  hw.host_gbps = 19.5;
  hw.disk_gbps = 2.3;
  hw.bf16_tflops = 60.0;
  hw.measured = true;
  options.hardware = hw;
  mi::pipeline::Pipeline pipe(options);
  // Other applications hold all but a 6 GiB arena: the DiT streams some of its blocks.
  const std::size_t free_now = mi::runtime::DeviceArena::free_device_bytes();
  const std::size_t want_free = (std::size_t{6} << 30) + options.vram_reserve;
  ASSERT_GE(free_now, want_free + (std::size_t{4} << 30)) << "the scenario needs ~10.4 GiB of free device memory";
  auto hold = std::make_unique<mi::DeviceBuffer>(free_now - want_free);
  const mi::pipeline::Result first = pipe.generate(req);
  ASSERT_FALSE(first.cancelled);
  ASSERT_EQ(first.plan.dit.residency, mi::planner::DitResidency::kStreamed) << first.plan.explanation;
  EXPECT_GT(first.plan.dit.resident_blocks, 0);
  hold.reset();  // the other application exits
  const mi::pipeline::Result second = pipe.generate(req);
  EXPECT_EQ(second.memory.arena_creations, 0) << second.plan.explanation;
  EXPECT_EQ(second.timings.dit_load_s, 0.0);
  EXPECT_FALSE(second.memory.dit_reloaded);
  EXPECT_EQ(second.plan.dit.resident_blocks, first.plan.dit.resident_blocks);
  EXPECT_NE(second.plan.explanation.find("Planned inside the held"), std::string::npos) << second.plan.explanation;
  EXPECT_EQ(bit_mismatches(second.latents, first.latents), 0u);
  EXPECT_TRUE(second.rgb == first.rgb);
  // A request the held arena cannot serve with the loaded DiT gets the larger arena, and the DiT the plan's residency.
  mi::pipeline::Request big = req;
  big.height = big.width = 1456;
  big.steps = 1;
  big.decode = false;
  const mi::pipeline::Result third = pipe.generate(big);
  EXPECT_FALSE(third.cancelled);
  EXPECT_EQ(third.memory.arena_creations, 1) << third.plan.explanation;
  EXPECT_GT(third.timings.dit_load_s, 0.0);
  EXPECT_EQ(third.plan.explanation.find("Planned inside the held"), std::string::npos) << third.plan.explanation;
}

// P4: the DiT streamed (min_memory: every block; balanced under a 7 GiB budget: the blocks that fit stay resident)
// gives the resident pipeline's latents and image bit for bit over several steps, within the plan's arena, with
// per-step streaming diagnostics. Then AC12 while streaming: a cancel inside a step lands within about one block, the
// streamed DiT stays loaded (no reload), and the next generate() is identical again.
TEST(PipelineEndToEnd, StreamedDitMatchesResidentAndCancels) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  using Clock = std::chrono::steady_clock;
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a lighthouse on a rocky coast at dusk, oil painting";
  req.negative_prompt = "blurry";
  req.height = req.width = 512;
  req.steps = 4;
  mi::pipeline::Result ref;
  {
    mi::pipeline::Pipeline pipe;
    ref = pipe.generate(req);
  }
  ASSERT_EQ(ref.plan.dit.residency, mi::planner::DitResidency::kResident) << ref.plan.explanation;
  ASSERT_FALSE(ref.dit_stream.streamed);
  ASSERT_EQ(ref.timings.step_s.size(), 4u);

  struct Variant {
    const char* name;
    mi::planner::Mode mode;
    double budget_gib;
  };
  for (const Variant v : {Variant{"min_memory", mi::planner::Mode::kMinMemory, 0.0},
                          Variant{"balanced, 7 GiB budget", mi::planner::Mode::kBalanced, 7.0}}) {
    mi::pipeline::PipelineOptions options;
    options.mode = v.mode;
    if (v.budget_gib > 0) options.vram_budget = static_cast<std::size_t>(v.budget_gib * (1 << 30));
    mi::pipeline::Pipeline pipe(options);
    const mi::pipeline::Result r = pipe.generate(req);
    ASSERT_FALSE(r.cancelled) << v.name;
    ASSERT_EQ(r.plan.dit.residency, mi::planner::DitResidency::kStreamed) << v.name << "\n" << r.plan.explanation;
    if (v.mode == mi::planner::Mode::kMinMemory) {
      EXPECT_EQ(r.plan.dit.resident_blocks, 0);
      EXPECT_LT(r.memory.arena_capacity, std::size_t{3} << 30);
    } else {
      EXPECT_GT(r.plan.dit.resident_blocks, 0);
      EXPECT_LT(r.plan.dit.resident_blocks, 42);
    }
    EXPECT_EQ(bit_mismatches(r.latents, ref.latents), 0u) << v.name;
    EXPECT_TRUE(r.rgb == ref.rgb) << v.name;
    EXPECT_LE(r.memory.arena_high_water, r.plan.arena_bytes) << v.name;
    EXPECT_FALSE(r.memory.dit_reloaded);
    const mi::pipeline::DitStreamStats& ds = r.dit_stream;
    ASSERT_TRUE(ds.streamed) << v.name;
    EXPECT_EQ(ds.resident_blocks, r.plan.dit.resident_blocks);
    EXPECT_EQ(ds.buffers, r.plan.dit.buffers);
    EXPECT_EQ(ds.source, "file");
    ASSERT_EQ(ds.steps.size(), 4u);
    EXPECT_GE(ds.bytes, 4 * static_cast<std::uint64_t>(r.plan.dit.stream_bytes) * 99 / 100);
    for (const auto& st : ds.steps) {
      EXPECT_GT(st.gbps, 1.0);
      EXPECT_GE(st.hidden, 0.0);
      EXPECT_LE(st.hidden, 1.0);
    }
    EXPECT_TRUE(r.hardware.measured);
    std::printf("[pipeline stream] %s: %d of 42 resident, %d buffers; arena %.2f GiB (high-water %.2f), device peak "
                "%.2f GiB, host private peak %.2f GiB; %.1f GB/s, %.1f%% hidden, %d stalls; steps",
                v.name, ds.resident_blocks, ds.buffers, static_cast<double>(r.memory.arena_capacity) / (1 << 30),
                static_cast<double>(r.memory.arena_high_water) / (1 << 30), static_cast<double>(r.memory.device_peak) / (1 << 30),
                static_cast<double>(r.memory.host_private_peak) / (1 << 30), ds.gbps(), 100.0 * ds.hidden(), ds.stalls);
    for (const double t : r.timings.step_s) std::printf(" %.3f", t);
    std::printf(" s (resident:");
    for (const double t : ref.timings.step_s) std::printf(" %.3f", t);
    std::printf(" s)\n");

    if (v.mode != mi::planner::Mode::kMinMemory) continue;
    std::atomic<bool> cancel{false};
    DelayedCancel dc{cancel};
    mi::pipeline::GenerateControl control;
    control.cancel = &cancel;
    const double step_s = r.timings.step_s[2];
    control.on_step = [&](const mi::pipeline::StepInfo& s) {
      if (s.step == 2) dc.arm(0.4 * step_s);  // lands inside step 3
      return true;
    };
    const mi::pipeline::Result c = pipe.generate(req, control);
    const auto done = Clock::now();
    dc.join();
    EXPECT_TRUE(c.cancelled);
    EXPECT_EQ(c.steps_run, 2);
    EXPECT_TRUE(c.rgb.empty());
    std::printf("[pipeline stream] cancel mid-step while streaming: returned %.0f ms after the flag (step %.3f s)\n",
                dc.until(done) * 1e3, step_s);
    EXPECT_LT(dc.until(done), 0.3 * step_s + 0.1);  // about one block of a 42-block step
    const mi::pipeline::Result again = pipe.generate(req);
    EXPECT_FALSE(again.cancelled);
    EXPECT_EQ(bit_mismatches(again.latents, ref.latents), 0u);
    EXPECT_TRUE(again.rgb == ref.rgb);
    EXPECT_FALSE(again.memory.dit_reloaded);
    EXPECT_EQ(again.timings.dit_load_s, 0.0);
    EXPECT_EQ(again.memory.arena_creations, 0);
  }
}

// M1/M2: the DiT block path. The default (fused) and the opt-in fused_graphs report themselves and give op_by_op's
// latents bit for bit; a failed fused self-check (injected through the cached hardware profile of this GPU) makes the
// pipeline run op by op and say why, with the same latents.
// P6: requests the quality tiers refuse fail with std::invalid_argument before any phase runs.
TEST(PipelineQuality, RefusalsBeforeAnyPhase) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a cat";
  req.height = req.width = 256;
  req.steps = 1;
  req.decode = false;
  const auto refused = [&](const mi::pipeline::PipelineOptions& options, const char* what) {
    mi::pipeline::Pipeline pipe(options);
    bool phase = false;
    mi::pipeline::GenerateControl control;
    control.on_phase = [&](mi::pipeline::Phase) { return phase = true; };
    EXPECT_THROW(pipe.generate(req, control), std::invalid_argument) << what;
    EXPECT_FALSE(phase) << what;
  };
  mi::pipeline::PipelineOptions o;
  o.tier = mi::planner::Tier::kHigh;
  o.attention = mi::planner::AttentionMode::kExact;
  refused(o, "tier high on a bf16 DiT with exact attention");
  o = {};
  o.tier = mi::planner::Tier::kTurbo;
  refused(o, "tier turbo");
  o = {};
  o.tier = mi::planner::Tier::kReference;
  o.attention = mi::planner::AttentionMode::kQ8Fp8;
  refused(o, "tier reference with 8-bit attention");
  o = {};
  o.text_encoder_file = weights_root() / "base" / "text_encoder";
  refused(o, "a separate text encoder file");
  o = {};
  o.dit_file = weights_root() / "does-not-exist.safetensors";
  refused(o, "a missing DiT file");
  const fs::path int8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
  if (fs::is_regular_file(int8)) {
    o = {};
    o.dit_file = int8;
    o.tier = mi::planner::Tier::kReference;
    refused(o, "tier reference on an INT8 DiT");
  }
}

// P6: the Comfy-Org INT8 SFT DiT from its single file, with the SFT bundle's tokenizer, text encoder and VAE. Tier auto
// (no certificate) runs INT8 GEMMs with exact attention, uncertified; tier high adds 8-bit attention on the loaded DiT
// (no reload); the report carries the quality object; the default (bf16 bundle, auto) stays the Reference tier.
TEST(PipelineEndToEnd, Int8DitFileRunsTheHighTier) {
  const fs::path int8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(int8)) {
    GTEST_SKIP() << "SFT bundle or INT8 DiT file not present";
  }
  mi::pipeline::Request req;
  req.model = weights_root() / "sft";
  req.prompt = "a lighthouse on a cliff at night";
  req.height = req.width = 256;
  req.steps = 3;
  req.decode = false;
  {
    mi::pipeline::PipelineOptions ref_options;
    ref_options.tier = mi::planner::Tier::kReference;  // AC11: the Reference tier, pinned
    mi::pipeline::Pipeline pipe(ref_options);
    const mi::pipeline::Result ref = pipe.generate(req);
    EXPECT_EQ(ref.tier.effective, mi::planner::Tier::kReference);
    EXPECT_EQ(ref.tier.attention, mi::planner::AttentionMode::kExact);
    EXPECT_EQ(ref.dit_linears, "bf16");
    EXPECT_EQ(ref.plan.dit.precision, mi::planner::Precision::kBf16);
  }
  mi::pipeline::PipelineOptions options;
  options.dit_file = int8;
  mi::pipeline::Pipeline pipe(options);
  const mi::pipeline::Result a = pipe.generate(req);
  EXPECT_EQ(a.steps_run, a.steps_scheduled);
  EXPECT_EQ(a.tier.effective, mi::planner::Tier::kHigh);
  EXPECT_EQ(a.tier.attention, mi::planner::AttentionMode::kExact);
  EXPECT_FALSE(a.tier.certified);
  EXPECT_EQ(a.dit_linears, "int8_convrot256");
  EXPECT_EQ(a.plan.dit.precision, mi::planner::Precision::kInt8);
  EXPECT_NE(a.plan.explanation.find("Quality: "), std::string::npos);
  EXPECT_GT(a.timings.dit_load_s, 0.0);
  const mi::json::Value doc = mi::json::parse("{" + mi::pipeline::report_fields(req, a, "", "") + "}");
  EXPECT_EQ(doc.at("quality").at("config").at("dit_precision").as_string(), "int8_convrot");
  EXPECT_EQ(doc.at("quality").at("config").at("attention").as_string(), "exact");
  EXPECT_EQ(doc.at("quality").at("tier").as_string(), "high");
  EXPECT_TRUE(doc.at("quality").at("certificate").is_null());
  for (const float v : a.latents) ASSERT_TRUE(std::isfinite(v));
  // Same request again: bit-identical (the loaded DiT is kept).
  const mi::pipeline::Result again = pipe.generate(req);
  EXPECT_EQ(again.timings.dit_load_s, 0.0);
  EXPECT_EQ(bit_mismatches(again.latents, a.latents), 0u);
}

TEST(PipelineEndToEnd, Int8DitFileHighTierAddsQ8Attention) {
  const fs::path int8 = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(int8)) {
    GTEST_SKIP() << "SFT bundle or INT8 DiT file not present";
  }
  mi::pipeline::Request req;
  req.model = weights_root() / "sft";
  req.prompt = "a lighthouse on a cliff at night";
  req.height = req.width = 256;
  req.steps = 3;
  req.decode = false;
  mi::pipeline::PipelineOptions options;
  options.dit_file = int8;
  options.tier = mi::planner::Tier::kHigh;
  mi::pipeline::Pipeline pipe(options);
  const mi::pipeline::Result r = pipe.generate(req);
  EXPECT_EQ(r.tier.effective, mi::planner::Tier::kHigh);
  EXPECT_EQ(r.tier.attention, mi::planner::AttentionMode::kQ8Fp8);
  EXPECT_EQ(r.plan.dit.attention, mi::planner::AttentionMode::kQ8Fp8);
  EXPECT_EQ(r.steps_run, r.steps_scheduled);
  for (const float v : r.latents) ASSERT_TRUE(std::isfinite(v));
}

// Runtime LoRAs on the bf16 SFT DiT (two real files, kohya and PEFT): they change the latents, report as uncertified;
// a strength change applies without reloading the DiT (side branches in place, merged adapters re-merged from the
// pristine weights), and strength 0 for every file gives the no-LoRA latents bit for bit. A latent preview reports every step.
TEST(PipelineEndToEnd, RuntimeLorasAndPreview) {
  const fs::path kohya = kKohyaLora;
  const fs::path peft = kPeftLora;
  if (!fs::is_regular_file(weights_root() / "sft" / "moss.mossidx") || !fs::is_regular_file(kohya) || !fs::is_regular_file(peft)) {
    GTEST_SKIP() << "SFT bundle or LoRA files not present";
  }
  mi::pipeline::Request req;
  req.model = weights_root() / "sft";
  req.prompt = "portrait of a woman in a garden, soft light";
  req.height = req.width = 256;
  req.steps = 3;
  req.decode = false;
  mi::pipeline::Pipeline pipe;
  int previews = 0;
  mi::pipeline::GenerateControl control;
  control.on_preview = [&](const mi::pipeline::PreviewImage& p) {
    ++previews;
    EXPECT_EQ(p.width, 256);
    EXPECT_EQ(p.height, 256);
    EXPECT_NE(p.rgb, nullptr);
  };
  const mi::pipeline::Result base = pipe.generate(req, control);
  EXPECT_EQ(previews, base.steps_run);
  EXPECT_TRUE(base.loras.empty());
  req.loras = {{kohya, 0.35f}, {peft, 0.60f}};
  const mi::pipeline::Result with = pipe.generate(req);
  ASSERT_EQ(with.loras.size(), 2u);
  EXPECT_GT(with.loras[0].side_branch + with.loras[1].side_branch, 0);
  EXPECT_FALSE(with.tier.certified);
  EXPECT_NE(with.tier.reason.find("LoRA"), std::string::npos);
  EXPECT_TRUE(with.quality.config.lora);
  EXPECT_GT(bit_mismatches(with.latents, base.latents), 0u);
  for (const float v : with.latents) ASSERT_TRUE(std::isfinite(v));
  req.loras = {{kohya, 0.0f}, {peft, 0.0f}};
  const mi::pipeline::Result zero = pipe.generate(req);
  EXPECT_EQ(zero.timings.dit_load_s, 0.0);  // side branches update in place, merges re-merge from the pristine weights
  EXPECT_EQ(bit_mismatches(zero.latents, base.latents), 0u);
  req.loras = {{kohya, 0.35f}, {peft, 0.60f}};  // back: the same latents as the first LoRA run
  const mi::pipeline::Result again = pipe.generate(req);
  EXPECT_EQ(again.timings.dit_load_s, 0.0);
  EXPECT_EQ(bit_mismatches(again.latents, with.latents), 0u);
}

// Auto with a certificate (a fixture for the Comfy-Org INT8 file, config int8 + q8_fp8, in <cache_dir>/quality_certs):
// found without LoRAs (8-bit attention turned on, the file named); with runtime LoRAs never, in either order, since
// the search is not cached for LoRA requests.
TEST(PipelineQuality, AutoUsesACertificateUnlessLorasAreActive) {
  if (!have_lora_files()) GTEST_SKIP() << "SFT bundle, INT8 DiT or LoRA files not present";
  using namespace mi::pipeline;
  mi::testing::TempDir dir("pipe_cert");
  Certificate c;
  c.tier = QualityTier::kHigh;
  c.config = CertConfig{DitPrecision::kInt8ConvRot, AttentionMode::kQ8Fp8, TePrecision::kBf16, false};
  c.model.dit = identify_component(kInt8Dit, true);
  c.numerics = engine_numerics(c.config);
  c.gpu_arch = current_gpu_arch();
  c.prompt_set = std::string(kCertPromptSetId);
  c.prompts = kCertPrompts;
  c.prompt_set_sha256 = std::string(kCertPromptSetSha256Base);
  c.scope = CertScope{1024 * 1024, {"euler"}, 1, 50, false};
  c.date = "2026-10-22T00:00:00Z";
  for (const GateSpec& gs : required_gates(QualityTier::kHigh)) {
    Gate gt;
    gt.name = std::string(gs.name);
    gt.op = gs.op;
    gt.threshold = gs.limit;
    gt.value = gs.op == GateOp::kLe ? gs.limit * 0.5 : (gs.limit > 0 ? gs.limit + 0.5 * (1 - gs.limit) : gs.limit * 0.5);
    if (gt.name.ends_with(".drift_count")) gt.value = std::floor(gt.value);  // whole prompts
    gt.pass = true;
    c.gates.push_back(gt);
  }
  c.passed = true;
  c.info = {{"same_seed.ocr_prompts", 3}, {"reference.ocr_prompts", 3}, {"official_images", 23}, {"drift_lpips", mi::pipeline::kDriftLpips}};
  ASSERT_EQ(validate_certificate(c), "");
  fs::create_directories(dir.path() / "cache" / "quality_certs");
  ASSERT_TRUE(write_certificate_file(dir.path() / "cache" / "quality_certs" / "fixture.json", {c}));

  PipelineOptions options;
  options.dit_file = kInt8Dit;
  options.cache_dir = dir.path() / "cache";
  options.embedding_cache = false;
  Pipeline pipe(options);
  Request req = lora_request();
  req.loras = {{kPeftLora, 0.5f}};
  for (int round = 0; round < 3; ++round) {
    const bool lora = round != 1;
    req.loras.clear();
    if (lora) req.loras.push_back({kPeftLora, 0.5f});
    const Result r = pipe.generate(req);
    EXPECT_EQ(r.tier.effective, mi::planner::Tier::kHigh) << round;
    EXPECT_EQ(r.tier.certified, !lora) << round << ": " << r.tier.reason;
    EXPECT_EQ(r.tier.attention, lora ? mi::planner::AttentionMode::kExact : mi::planner::AttentionMode::kQ8Fp8) << round;
    EXPECT_EQ(r.quality.config.lora, lora) << round;
    EXPECT_EQ(r.quality.certificate.empty(), lora) << round;
    if (!lora) {
      EXPECT_NE(r.tier.reason.find("certified: "), std::string::npos) << r.tier.reason;
    }
  }
}

// Runtime LoRAs on the INT8 DiT: they run (finite, different from no LoRA), the side branches' device memory counts
// against an explicit VRAM budget (the arena cap shrinks by exactly their bytes), and a failing LoRA load leaves the
// previous stack in place (the next request with it runs without rebuilding and gives the same latents).
TEST(PipelineEndToEnd, Int8DitLorasBudgetAndFailedLoad) {
  if (!have_lora_files()) GTEST_SKIP() << "SFT bundle, INT8 DiT or LoRA files not present";
  const fs::path not_a_lora = "/workspace/Models/community/comfy-org/flux2-vae_small.safetensors";
  mi::pipeline::PipelineOptions options;
  options.dit_file = kInt8Dit;
  options.vram_budget = std::size_t{8} << 30;
  mi::pipeline::Pipeline pipe(options);
  mi::pipeline::Request req = lora_request();
  const mi::pipeline::Result base = pipe.generate(req);
  req.loras = {{kKohyaLora, 0.35f}, {kPeftLora, 0.60f}};
  const mi::pipeline::Result with = pipe.generate(req);
  for (const float v : with.latents) ASSERT_TRUE(std::isfinite(v));
  EXPECT_GT(bit_mismatches(with.latents, base.latents), 0u);
  const mi::lora::LoraStack stack(req.loras, mi::lora::moss_dit_targets());
  const std::size_t sb = mi::lora::SideBranches::device_bytes(stack);
  std::printf("[pipeline] INT8 + 2 LoRAs: side branches %.1f MiB; arena cap %.3f -> %.3f GiB\n",
              static_cast<double>(sb) / (1 << 20), static_cast<double>(base.plan.arena_cap_bytes) / (1 << 30),
              static_cast<double>(with.plan.arena_cap_bytes) / (1 << 30));
  EXPECT_EQ(base.plan.arena_cap_bytes - with.plan.arena_cap_bytes, sb);
  if (!fs::is_regular_file(not_a_lora)) return;
  req.loras.push_back({not_a_lora, 1.0f});
  EXPECT_ANY_THROW(pipe.generate(req));
  req.loras.pop_back();
  const mi::pipeline::Result again = pipe.generate(req);
  EXPECT_EQ(again.timings.dit_load_s, 0.0);
  EXPECT_EQ(bit_mismatches(again.latents, with.latents), 0u);
}

TEST(PipelineEndToEnd, BlockPathFollowsTheSelfCheck) {
  if (!have_weights()) GTEST_SKIP() << "weights not present";
  using mi::moss_dit::BlockPath;
  mi::pipeline::Request req;
  req.model = weights_root() / "base";
  req.prompt = "a lighthouse on a rocky coast at dusk, oil painting";
  req.negative_prompt = "blurry";
  req.height = req.width = 256;
  req.steps = 3;
  req.decode = false;
  mi::testing::TempDir shared("block_path_cache");  // embedding cache spill + profile shared by the healthy runs
  std::vector<float> want;
  for (const BlockPath path : {BlockPath::kOpByOp, BlockPath::kFused, BlockPath::kFusedGraphs}) {
    mi::pipeline::PipelineOptions o;
    o.block_path = path;
    o.cache_dir = shared.path();
    mi::pipeline::Pipeline pipe(o);
    const mi::pipeline::Result r = pipe.generate(req);
    ASSERT_FALSE(r.cancelled);
    EXPECT_EQ(r.block_path, path) << r.block_path_note;
    EXPECT_TRUE(r.block_path_note.empty()) << r.block_path_note;
    if (path == BlockPath::kFusedGraphs) {
      EXPECT_GT(r.plan.dit.graph_bytes, 0u);
    }
    if (want.empty()) {
      want = r.latents;
    } else {
      EXPECT_EQ(bit_mismatches(r.latents, want), 0u) << mi::moss_dit::block_path_name(path);
    }
  }
  const auto stored = mi::pipeline::read_profile(shared.path() / "hardware_profile.json", mi::pipeline::hardware_key(),
                                                 std::chrono::duration_cast<std::chrono::seconds>(
                                                     std::chrono::system_clock::now().time_since_epoch()).count());
  ASSERT_TRUE(stored.has_value());
  ASSERT_TRUE(stored->fused_check.has_value());
  EXPECT_TRUE(stored->fused_check->ok) << stored->fused_check->detail;
  EXPECT_FALSE(stored->fused_check->validated.empty());

  mi::testing::TempDir injected("block_path_injected");
  mi::pipeline::StoredProfile bad = *stored;
  bad.fused_check = mi::moss_dit::FusedCheck{false, "injected mismatch (o_proj at 84 rows)", {}};
  bad.learned.h2d_gbps = 23.25;  // marks that this profile was used
  ASSERT_TRUE(mi::pipeline::write_profile(injected.path() / "hardware_profile.json", mi::pipeline::hardware_key(), bad));
  mi::pipeline::PipelineOptions o;
  o.cache_dir = injected.path();
  mi::pipeline::Pipeline pipe(o);
  const mi::pipeline::Result r = pipe.generate(req);
  ASSERT_FALSE(r.cancelled);
  EXPECT_EQ(r.hardware.h2d_gbps, 23.25);
  EXPECT_EQ(r.block_path, BlockPath::kOpByOp);
  EXPECT_NE(r.block_path_note.find("injected mismatch (o_proj at 84 rows)"), std::string::npos) << r.block_path_note;
  EXPECT_EQ(bit_mismatches(r.latents, want), 0u);
  std::printf("[pipeline] block path after a failed self-check: %s (%s)\n", mi::moss_dit::block_path_name(r.block_path),
              r.block_path_note.c_str());
}

}  // namespace
