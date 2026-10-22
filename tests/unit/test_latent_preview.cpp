// Latent preview (src/models/latent_preview): the kernels against an fp64 host implementation of the same model and
// against the fitting script's own previews (tests/data/latent_preview, made in the dev venv with torch), the preview
// quality on the held-out golden latents (vs the reference pipeline's images), and its speed.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO_WARNINGS
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "models/latent_preview/latent_preview.h"
#include "models/latent_preview/preview_coefficients.h"
#include "test_util.h"
#include "util/npy.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;

fs::path source_dir() { return fs::path(MOSSINFER_SOURCE_DIR); }
fs::path data_dir() { return source_dir() / "tests" / "data" / "latent_preview"; }

struct Dev {
  mi::DeviceBuffer buf;
  Tensor t;
  Dev(mi::DType dt, std::initializer_list<std::int64_t> shape) {
    t = Tensor::view(nullptr, dt, mi::Device::kCUDA, shape);
    buf = mi::DeviceBuffer(std::max<std::size_t>(t.nbytes(), 1));
    t.data = buf.get();
  }
};

// The model in fp64 on the host: [128, h, w] -> RGB8 [out_h, out_w, 3].
std::vector<std::uint8_t> host_preview(const std::vector<float>& z, int h, int w, int out_h, int out_w) {
  using mi::latent_preview::detail::kPreviewCoef;
  const int h8 = 2 * h, w8 = 2 * w;
  std::vector<double> low(static_cast<std::size_t>(h8 * w8 * 3));
  for (int y = 0; y < h8; ++y)
    for (int x = 0; x < w8; ++x)
      for (int k = 0; k < 3; ++k) {
        double acc = kPreviewCoef[k][288];
        for (int c = 0; c < 32; ++c)
          for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx) {
              const int yy = y + dy - 1, xx = x + dx - 1;
              if (yy < 0 || yy >= h8 || xx < 0 || xx >= w8) continue;
              const int ch = 4 * c + 2 * (yy & 1) + (xx & 1);
              acc += kPreviewCoef[k][c * 9 + dy * 3 + dx] * z[static_cast<std::size_t>((ch * h + yy / 2) * w + xx / 2)];
            }
        low[static_cast<std::size_t>((y * w8 + x) * 3 + k)] = acc;
      }
  std::vector<std::uint8_t> out(static_cast<std::size_t>(out_h * out_w * 3));
  for (int y = 0; y < out_h; ++y)
    for (int x = 0; x < out_w; ++x) {
      const double sy = std::max((y + 0.5) * h8 / out_h - 0.5, 0.0), sx = std::max((x + 0.5) * w8 / out_w - 0.5, 0.0);
      const int y0 = std::min(static_cast<int>(sy), h8 - 1), x0 = std::min(static_cast<int>(sx), w8 - 1);
      const int y1 = std::min(y0 + 1, h8 - 1), x1 = std::min(x0 + 1, w8 - 1);
      const double ly = sy - y0, lx = sx - x0;
      for (int k = 0; k < 3; ++k) {
        const auto at = [&](int yy, int xx) { return low[static_cast<std::size_t>((yy * w8 + xx) * 3 + k)]; };
        const double v = (1 - ly) * ((1 - lx) * at(y0, x0) + lx * at(y0, x1)) + ly * ((1 - lx) * at(y1, x0) + lx * at(y1, x1));
        out[static_cast<std::size_t>((y * out_w + x) * 3 + k)] =
            static_cast<std::uint8_t>(std::nearbyint(std::clamp(v, 0.0, 1.0) * 255.0));
      }
    }
  return out;
}

struct Cmp {
  double identical = 0;
  int max_abs = 0;
};
Cmp compare(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  std::size_t same = 0;
  int mx = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    same += d == 0;
    mx = std::max(mx, d);
  }
  return {static_cast<double>(same) / static_cast<double>(a.size()), mx};
}

std::vector<std::uint8_t> run_preview(const std::vector<float>& z, std::int64_t h, std::int64_t w, std::int64_t out_h,
                                      std::int64_t out_w) {
  Dev lat(mi::DType::kF32, {1, 128, h, w}), rgb(mi::DType::kU8, {out_h, out_w, 3});
  lat.buf.upload(z.data(), z.size() * sizeof(float));
  const std::size_t ws_bytes = mi::latent_preview::workspace_bytes(h, w);
  mi::DeviceBuffer ws(ws_bytes);
  mi::latent_preview::preview(lat.t, rgb.t, ws.get(), ws_bytes, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint8_t> out(static_cast<std::size_t>(rgb.t.nbytes()));
  rgb.buf.download(out.data(), out.size());
  return out;
}

std::vector<float> hash_latent(std::size_t n, std::uint64_t seed) {
  std::vector<float> v(n);
  std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  for (float& x : v) {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    x = (static_cast<float>(s >> 40) * 0x1p-24f - 0.5f) * 3.0f;
  }
  return v;
}

// fp32 kernels vs the fp64 model: the H/8 sums differ in the last bits only, so bytes differ by at most 1 (rounding
// ties), on random latents with odd sizes (borders on every side, non-integer up- and downsampling, 1x1).
TEST(LatentPreview, MatchesFp64Model) {
  struct Case {
    int h, w, out_h, out_w;
  };
  for (const Case& c : {Case{5, 7, 37, 53}, Case{5, 7, 80, 112}, Case{5, 7, 10, 14}, Case{5, 7, 3, 2}, Case{1, 1, 16, 16},
                        Case{1, 1, 1, 1}, Case{16, 9, 256, 144}}) {
    const auto z = hash_latent(static_cast<std::size_t>(128 * c.h * c.w), static_cast<std::uint64_t>(c.h * 100 + c.out_h));
    const Cmp d = compare(host_preview(z, c.h, c.w, c.out_h, c.out_w), run_preview(z, c.h, c.w, c.out_h, c.out_w));
    std::printf("[latent_preview] %dx%d latent -> %dx%d: identical %.4f%%, max|diff| %d\n", c.w, c.h, c.out_w, c.out_h,
                100 * d.identical, d.max_abs);
    EXPECT_LE(d.max_abs, 1);
    EXPECT_GE(d.identical, 0.995);
  }
}

// Against the fitting script's torch previews (unfold + matmul + F.interpolate bilinear) of the VAE tests' golden latent
// crop (tests/data/latent_preview/manifest.json records the provenance).
TEST(LatentPreview, MatchesFittingScript) {
  const auto a = mi::npy::load(source_dir() / "tests" / "data" / "flux2_vae" / "full_crop.latent.npy");
  ASSERT_EQ(a.shape.size(), 4u);
  const auto z = a.to_vector<float>();
  for (const char* name : {"crop.preview_48x40.npy", "crop.preview_413x337.npy"}) {
    const auto ref = mi::npy::load(data_dir() / name);
    const auto want = ref.to_vector<std::uint8_t>();
    const Cmp d = compare(want, run_preview(z, a.shape[2], a.shape[3], ref.shape[0], ref.shape[1]));
    std::printf("[latent_preview] vs torch %s: identical %.4f%%, max|diff| %d\n", name, 100 * d.identical, d.max_abs);
    EXPECT_LE(d.max_abs, 1) << name;
    EXPECT_GE(d.identical, 0.995) << name;
  }
}

double psnr(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  double se = 0;
  for (std::size_t i = 0; i < a.size(); ++i) se += std::pow(static_cast<double>(a[i]) - b[i], 2);
  return 10 * std::log10(255.0 * 255.0 / (se / static_cast<double>(a.size())));
}

// Preview at the image size vs the reference pipeline's image of the same final latent, on the 7 golden cases (held
// out of the fit). Measured when fitted: 20.3-24.9 dB (text-heavy layouts lowest).
TEST(LatentPreview, QualityOnHeldOutGoldenLatents) {
  const fs::path golden = source_dir() / "tests" / "golden";
  int n = 0;
  double sum = 0;
  for (const auto& e : fs::directory_iterator(golden)) {
    const fs::path lat = e.path() / "final_latent.npy", png = e.path() / "image.png";
    if (!fs::is_regular_file(lat) || !fs::is_regular_file(png)) continue;
    const auto a = mi::npy::load(lat);
    int iw = 0, ih = 0, comp = 0;
    stbi_uc* img = stbi_load(png.string().c_str(), &iw, &ih, &comp, 3);
    ASSERT_NE(img, nullptr);
    const std::vector<std::uint8_t> ref(img, img + static_cast<std::size_t>(iw) * ih * 3);
    stbi_image_free(img);
    const double p = psnr(ref, run_preview(a.to_vector<float>(), a.shape[2], a.shape[3], ih, iw));
    std::printf("[latent_preview] %-16s %4dx%-4d PSNR %.2f dB\n", e.path().filename().string().c_str(), iw, ih, p);
    EXPECT_GE(p, 19.5) << e.path();
    sum += p;
    ++n;
  }
  if (n == 0) GTEST_SKIP() << "golden latents absent";
  std::printf("[latent_preview] mean PSNR %.2f dB over %d held-out cases\n", sum / n, n);
  EXPECT_GE(sum / n, 22.0);
}

// Full-size previews: well under 1 ms (dev numbers; asserted with the repo's timing switch).
TEST(LatentPreview, Timing) {
  for (const int px : {1024, 2048}) {
    const std::int64_t h = px / 16, w = px / 16;
    Dev lat(mi::DType::kF32, {128, h, w}), rgb(mi::DType::kU8, {px, px, 3});
    const auto z = hash_latent(static_cast<std::size_t>(128 * h * w), 7);
    lat.buf.upload(z.data(), z.size() * sizeof(float));
    const std::size_t ws_bytes = mi::latent_preview::workspace_bytes(h, w);
    mi::DeviceBuffer ws(ws_bytes);
    cudaStream_t s = nullptr;
    MI_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t a = nullptr, b = nullptr;
    MI_CUDA_CHECK(cudaEventCreate(&a));
    MI_CUDA_CHECK(cudaEventCreate(&b));
    for (const bool native : {false, true}) {
      const Tensor out = native ? Tensor::view(rgb.t.data, mi::DType::kU8, mi::Device::kCUDA, {2 * h, 2 * w, 3}) : rgb.t;
      mi::latent_preview::preview(lat.t, out, ws.get(), ws_bytes, s);
      constexpr int kIters = 50;
      MI_CUDA_CHECK(cudaEventRecord(a, s));
      for (int i = 0; i < kIters; ++i) mi::latent_preview::preview(lat.t, out, ws.get(), ws_bytes, s);
      MI_CUDA_CHECK(cudaEventRecord(b, s));
      MI_CUDA_CHECK(cudaEventSynchronize(b));
      float ms = 0;
      MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      const double per = ms / kIters;
      std::printf("[latent_preview] %dx%d image, preview %lldx%lld: %.4f ms (dev)\n", px, px,
                  static_cast<long long>(out.dim(1)), static_cast<long long>(out.dim(0)), per);
      if (mi::testing::kAssertTimings) {
        EXPECT_LT(per, 1.0);
      }
    }
    MI_CUDA_CHECK(cudaEventDestroy(a));
    MI_CUDA_CHECK(cudaEventDestroy(b));
    MI_CUDA_CHECK(cudaStreamDestroy(s));
  }
}

TEST(LatentPreview, RejectsBadArguments) {
  Dev lat(mi::DType::kF32, {128, 4, 4}), rgb(mi::DType::kU8, {64, 64, 3}), bad_lat(mi::DType::kF32, {64, 4, 4}),
      bad_rgb(mi::DType::kU8, {64, 64, 4});
  mi::DeviceBuffer ws(mi::latent_preview::workspace_bytes(4, 4));
  const std::size_t n = mi::latent_preview::workspace_bytes(4, 4);
  EXPECT_THROW(mi::latent_preview::preview(bad_lat.t, rgb.t, ws.get(), n, nullptr), std::invalid_argument);
  EXPECT_THROW(mi::latent_preview::preview(lat.t, bad_rgb.t, ws.get(), n, nullptr), std::invalid_argument);
  EXPECT_THROW(mi::latent_preview::preview(lat.t, rgb.t, ws.get(), n - 4, nullptr), std::invalid_argument);
  EXPECT_THROW(mi::latent_preview::workspace_bytes(0, 4), std::invalid_argument);
  // Output rows beyond the launch grid's limit (65535 blocks of 8 rows) are refused before anything runs.
  const Tensor too_tall = Tensor::view(rgb.t.data, mi::DType::kU8, mi::Device::kCUDA, {65535 * 8 + 1, 1, 3});
  EXPECT_THROW(mi::latent_preview::preview(lat.t, too_tall, ws.get(), n, nullptr), std::invalid_argument);
  EXPECT_NO_THROW(mi::latent_preview::preview(lat.t, rgb.t, ws.get(), n, nullptr));
  MI_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace
