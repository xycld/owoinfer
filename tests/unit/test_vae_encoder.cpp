// FLUX.2 VAE encoder (src/models/flux2_vae_encoder) and its kernels (src/kernels/vae/encoder_*) against the
// reference (diffusers AutoencoderKLFlux2 in fp32 with cuDNN TF32 convs; vectors in tests/data/vae_encoder):
//   - stride-2 downsample conv and 3xTF32 convs vs fp64 host convs on synthetic data, band calls bit-identical to the
//     whole-tensor call; head-dim-512 attention vs an fp64 softmax;
//   - the loader on synthetic safetensors headers (missing / extra / BF16 / misshapen tensors, key prefixes);
//   - preprocessing and the latent distribution/patchify/BN step bit-exact against host re-implementations;
//   - the whole encoder on a tracked 256x192 crop (mode and a seeded sample), on the original single-file layout,
//     and on full-size images when the local references exist (1024^2, 1248x1664, 1456^2, 2048^2);
//   - exact banded levels byte-identical to the untiled encode, plan/budget queries, workspace vs high-water,
//     encode time per size.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/vae/encoder_kernels.h"
#include "kernels/vae/attention.h"
#include "models/flux2_vae_encoder/flux2_vae_encoder.h"
#include "pipeline/rng_torch_compat.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "test_util.h"
#include "util/npy.h"
#include "weights/safetensors.h"
#include "weights/model_index.h"

namespace {

namespace fs = std::filesystem;
using mi::Tensor;
using mi::flux2_vae_encoder::EncodeOptions;
using mi::flux2_vae_encoder::Flux2VaeEncoder;
using mi::flux2_vae_encoder::LatentMode;
using mi::flux2_vae_encoder::TilePlan;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kMiB = 1024.0 * 1024.0;

fs::path source_dir() { return fs::path(MOSSINFER_SOURCE_DIR); }
fs::path data_dir() { return source_dir() / "tests" / "data" / "vae_encoder"; }
fs::path large_dir() { return source_dir() / "tools" / "reference" / "work" / "vae_encoder"; }
fs::path vae_file() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base" / "vae" /
         "diffusion_pytorch_model.safetensors";
}
fs::path original_layout_file() { return "/workspace/Models/community/comfy-org/flux2-vae.safetensors"; }

struct DevTensor {
  mi::DeviceBuffer buf;
  Tensor t;
  DevTensor(mi::DType dt, std::initializer_list<std::int64_t> shape) {
    t = Tensor::view(nullptr, dt, mi::Device::kCUDA, shape);
    buf = mi::DeviceBuffer(std::max<std::size_t>(t.nbytes(), 1));
    t.data = buf.get();
  }
  template <typename T>
  void upload(const std::vector<T>& v) { buf.upload(v.data(), v.size() * sizeof(T)); }
  template <typename T>
  std::vector<T> download() const {
    std::vector<T> v(t.nbytes() / sizeof(T));
    buf.download(v.data(), v.size() * sizeof(T));
    return v;
  }
};

std::vector<float> load_f32(const fs::path& p) { return mi::npy::load(p).to_vector<float>(); }

struct Stats {
  double cosine = 0, max_abs = 0, max_rel = 0, ref_absmax = 0;
};
// max_rel: max |got - ref| / max |ref| (relative to the tensor's range; per-element relative errors explode near 0).
template <typename A, typename B>
Stats compare(const std::vector<A>& ref, const std::vector<B>& got) {
  Stats s;
  double dot = 0, nr = 0, ng = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = ref[i], g = got[i];
    dot += r * g;
    nr += r * r;
    ng += g * g;
    s.max_abs = std::max(s.max_abs, std::abs(r - g));
    s.ref_absmax = std::max(s.ref_absmax, std::abs(r));
  }
  s.cosine = dot / std::sqrt(nr * ng);
  s.max_rel = s.max_abs / s.ref_absmax;
  return s;
}

void print_stats(const char* tag, const std::string& label, const Stats& s) {
  std::printf("[%s] %-28s cosine %.9f  max|diff| %.3e  max rel (vs range) %.3e  (|ref| <= %.3f)\n", tag, label.c_str(),
              s.cosine, s.max_abs, s.max_rel, s.ref_absmax);
}

// ---- Kernels --------------------------------------------------------------------------------------------------

// Deterministic synthetic data in [-amp, amp) (a 32-bit hash per index; no tracked inputs needed).
std::vector<float> synth(std::uint32_t seed, std::size_t n, float amp) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761u + seed * 97u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    v[i] = (static_cast<float>(h >> 8) * 0x1p-24f - 0.5f) * 2.0f * amp;
  }
  return v;
}

// Stride-2 downsample conv on synthetic NHWC data vs an fp64 host conv (zero pad of one row/column at the bottom/right);
// row bands are bit-identical to the whole-tensor call.
TEST(VaeEncoderKernels, DownsampleConvMatchesFp64AndBandsAreBitIdentical) {
  struct Case {
    std::int64_t h, w, c;
  };
  for (const Case& cs : {Case{18, 22, 128}, Case{10, 14, 256}}) {
    const std::int64_t h = cs.h, w = cs.w, c = cs.c, p = h / 2, q = w / 2;
    const std::vector<float> xh = synth(1, static_cast<std::size_t>(h * w * c), 1.7f);
    const std::vector<float> wh = synth(2, static_cast<std::size_t>(c * 9 * c), 1.7f / std::sqrt(9.0f * static_cast<float>(c)));
    const std::vector<float> bh = synth(3, static_cast<std::size_t>(c), 0.2f);
    std::vector<double> y64(static_cast<std::size_t>(p * q * c));  // wh is KRSC
    for (std::int64_t i = 0; i < p; ++i)
      for (std::int64_t j = 0; j < q; ++j)
        for (std::int64_t k = 0; k < c; ++k) {
          double acc = bh[static_cast<std::size_t>(k)];
          for (int r = 0; r < 3; ++r)
            for (int t = 0; t < 3; ++t) {
              const std::int64_t ii = 2 * i + r, jj = 2 * j + t;
              if (ii >= h || jj >= w) continue;
              const float* xp = &xh[static_cast<std::size_t>((ii * w + jj) * c)];
              const float* wp = &wh[static_cast<std::size_t>(((k * 3 + r) * 3 + t) * c)];
              for (std::int64_t cc = 0; cc < c; ++cc) acc += static_cast<double>(wp[cc]) * xp[cc];
            }
          y64[static_cast<std::size_t>((i * q + j) * c + k)] = acc;
        }
    DevTensor x(mi::DType::kF32, {1, h, w, c}), wt(mi::DType::kF32, {c, 3, 3, c}), b(mi::DType::kF32, {c}),
        y(mi::DType::kF32, {1, p, q, c});
    x.upload(xh);
    wt.upload(wh);
    b.upload(bh);
    const std::size_t ws_bytes = std::max<std::size_t>({16, mi::vae::conv2d_down_workspace_bytes(x.t, wt.t, y.t, mi::vae::ConvMath::kTf32),
                                                        mi::vae::conv2d_down_workspace_bytes(x.t, wt.t, y.t, mi::vae::ConvMath::kTf32x3)});
    DevTensor ws(mi::DType::kU8, {static_cast<std::int64_t>(ws_bytes)});
    const auto run_whole = [&](mi::vae::ConvMath m) {
      mi::vae::conv2d_down(x.t, wt.t, b.t, y.t, ws.t.data, ws_bytes, m, nullptr);
      return y.download<float>();
    };
    const std::string name = "c" + std::to_string(c);
    const Stats t64 = compare(y64, run_whole(mi::vae::ConvMath::kTf32));
    print_stats("conv2d_down", name + " kTf32 vs fp64", t64);
    EXPECT_GE(t64.cosine, 0.99999);
    EXPECT_LE(t64.max_rel, 2e-3);  // TF32 operands (10-bit mantissa), fp32 accumulation
    const auto got = run_whole(mi::vae::ConvMath::kTf32x3);
    const Stats x64 = compare(y64, got);
    print_stats("conv2d_down", name + " kTf32x3 vs fp64", x64);
    EXPECT_GE(x64.cosine, 0.99999999);
    EXPECT_LE(x64.max_rel, 2e-6);

    // Bands: output rows [i0, i1) from input rows [2 i0, min(h, 2 i1 + 1)) equal the whole call bit for bit.
    std::vector<float> banded(got.size());
    const std::size_t orow = static_cast<std::size_t>(q * c), irow = static_cast<std::size_t>(w * c);
    for (std::int64_t i0 = 0; i0 < p; i0 += 3) {
      const std::int64_t i1 = std::min(p, i0 + 3), a = 2 * i0, e = std::min(h, 2 * i1 + 1);
      DevTensor yb(mi::DType::kF32, {1, i1 - i0, q, c});
      const Tensor xb = Tensor::view(static_cast<float*>(x.t.data) + static_cast<std::size_t>(a) * irow, mi::DType::kF32,
                                     mi::Device::kCUDA, {1, e - a, w, c});
      mi::vae::conv2d_down(xb, wt.t, b.t, yb.t, ws.t.data, ws_bytes, mi::vae::ConvMath::kTf32x3, nullptr);
      const auto part = yb.download<float>();
      std::copy(part.begin(), part.end(), banded.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i0) * orow));
    }
    EXPECT_EQ(0, std::memcmp(banded.data(), got.data(), got.size() * 4)) << name;
  }
}

// Stride-1 conv, both kernel sizes, with a residual: 3xTF32 vs an fp64 host conv ("same" zero padding).
TEST(VaeEncoderKernels, Tf32x3ConvMatchesFp64) {
  const std::int64_t h = 9, w = 13, c = 68, k = 64;
  for (const int r : {3, 1}) {
    std::vector<float> x(static_cast<std::size_t>(h * w * c)), wt(static_cast<std::size_t>(k * r * r * c)), b(k),
        res(static_cast<std::size_t>(h * w * k));
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.71f * static_cast<float>(i)) * 2.0f;
    for (std::size_t i = 0; i < wt.size(); ++i) wt[i] = std::cos(0.37f * static_cast<float>(i)) * 0.05f;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = 0.1f * static_cast<float>(i % 7);
    for (std::size_t i = 0; i < res.size(); ++i) res[i] = std::sin(1.3f * static_cast<float>(i));
    std::vector<double> ref(res.size());
    for (std::int64_t i = 0; i < h; ++i)
      for (std::int64_t j = 0; j < w; ++j)
        for (std::int64_t o = 0; o < k; ++o) {
          double acc = b[static_cast<std::size_t>(o)];
          for (int dr = 0; dr < r; ++dr)
            for (int ds = 0; ds < r; ++ds) {
              const std::int64_t ii = i + dr - (r - 1) / 2, jj = j + ds - (r - 1) / 2;
              if (ii < 0 || ii >= h || jj < 0 || jj >= w) continue;
              for (std::int64_t q = 0; q < c; ++q)
                acc += static_cast<double>(wt[static_cast<std::size_t>(((o * r + dr) * r + ds) * c + q)]) *
                       x[static_cast<std::size_t>((ii * w + jj) * c + q)];
            }
          const std::size_t idx = static_cast<std::size_t>((i * w + j) * k + o);
          ref[idx] = acc + res[idx];
        }
    DevTensor xd(mi::DType::kF32, {1, h, w, c}), wd(mi::DType::kF32, {k, r, r, c}), bd(mi::DType::kF32, {k}),
        rd(mi::DType::kF32, {1, h, w, k}), yd(mi::DType::kF32, {1, h, w, k});
    xd.upload(x);
    wd.upload(wt);
    bd.upload(b);
    std::size_t ws_bytes = 16;
    for (const auto m : {mi::vae::ConvMath::kTf32, mi::vae::ConvMath::kTf32x3})
      ws_bytes = std::max(ws_bytes, mi::vae::encoder_conv2d_workspace_bytes(xd.t, wd.t, m));
    DevTensor ws(mi::DType::kU8, {static_cast<std::int64_t>(ws_bytes)});
    for (const auto m : {mi::vae::ConvMath::kTf32, mi::vae::ConvMath::kTf32x3}) {
      rd.upload(res);
      mi::vae::encoder_conv2d(xd.t, wd.t, bd.t, &rd.t, rd.t, ws.t.data, ws_bytes, m, nullptr);  // y aliases the residual
      const Stats st = compare(ref, rd.download<float>());
      const bool x3 = m == mi::vae::ConvMath::kTf32x3;
      print_stats("encoder_conv2d", std::string(r == 3 ? "3x3 " : "1x1 ") + (x3 ? "kTf32x3" : "kTf32") + " vs fp64", st);
      EXPECT_LE(st.max_rel, x3 ? 2e-6 : 2e-3);
    }
  }
}

// The encoder runs the shared VAE attention at head dim 512 (kernels/vae/attention.h).
TEST(VaeEncoderKernels, Attention512MatchesFp64) {
  const std::int64_t len = 200, d = 512;  // L not a multiple of the 64-row tiles
  ASSERT_TRUE(mi::vae::attention_head_dim_supported(d));
  const std::vector<float> qh = synth(11, static_cast<std::size_t>(len * d), 3.4f);  // a softmax neither flat nor one-hot
  const std::vector<float> kh = synth(12, static_cast<std::size_t>(len * d), 1.7f);
  const std::vector<float> vh = synth(13, static_cast<std::size_t>(len * d), 1.7f);
  std::vector<double> ref(static_cast<std::size_t>(len * d), 0.0), sc(static_cast<std::size_t>(len));
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  for (std::int64_t i = 0; i < len; ++i) {
    double mx = -1e300, sum = 0;
    for (std::int64_t j = 0; j < len; ++j) {
      double acc = 0;
      for (std::int64_t cc = 0; cc < d; ++cc)
        acc += static_cast<double>(qh[static_cast<std::size_t>(i * d + cc)]) * kh[static_cast<std::size_t>(j * d + cc)];
      sc[static_cast<std::size_t>(j)] = acc * scale;
      mx = std::max(mx, sc[static_cast<std::size_t>(j)]);
    }
    for (double& v : sc) sum += (v = std::exp(v - mx));
    for (std::int64_t j = 0; j < len; ++j)
      for (std::int64_t cc = 0; cc < d; ++cc)
        ref[static_cast<std::size_t>(i * d + cc)] += sc[static_cast<std::size_t>(j)] / sum * vh[static_cast<std::size_t>(j * d + cc)];
  }
  DevTensor q(mi::DType::kF32, {1, len, d}), k(mi::DType::kF32, {1, len, d}), v(mi::DType::kF32, {1, len, d}),
      o(mi::DType::kF32, {1, len, d});
  q.upload(qh);
  k.upload(kh);
  v.upload(vh);
  for (const auto prec : {mi::vae::AttnPrecision::kTf32x3, mi::vae::AttnPrecision::kTf32}) {
    mi::vae::attention_fp32(q.t, k.t, v.t, o.t, 1.0f / std::sqrt(512.0f), prec, nullptr);
    const Stats st = compare(ref, o.download<float>());
    const bool x3 = prec == mi::vae::AttnPrecision::kTf32x3;
    print_stats("attention_d512", x3 ? "kTf32x3 vs fp64" : "kTf32 vs fp64", st);
    EXPECT_GE(st.cosine, x3 ? 0.9999999 : 0.99999);
    EXPECT_LE(st.max_rel, x3 ? 1e-5 : 2e-3);
  }
}

TEST(VaeEncoderKernels, PreprocessAndLatentOutAreBitExact) {
  // Preprocess: every u8 value, and f32 inputs.
  const std::int64_t h = 16, w = 16;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(h * w * 3));
  for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>((i * 7 + 3) % 256);
  DevTensor rgb_d(mi::DType::kU8, {h, w, 3}), img(mi::DType::kF32, {1, h, w, 4});
  rgb_d.upload(rgb);
  mi::vae::encoder_preprocess(rgb_d.t, img.t, nullptr);
  auto got = img.download<float>();
  for (std::size_t px = 0; px < static_cast<std::size_t>(h * w); ++px) {
    for (int c = 0; c < 3; ++c) {
      volatile float f = static_cast<float>(rgb[px * 3 + c]) / 255.0f;  // numpy float32 division
      volatile float two_f = 2.0f * f;
      const float want = two_f - 1.0f;
      ASSERT_EQ(std::memcmp(&want, &got[px * 4 + c], 4), 0) << "u8 " << int(rgb[px * 3 + c]);
    }
    ASSERT_EQ(got[px * 4 + 3], 0.0f);
  }

  // Latent out: moments [1, 8, 6, 64] -> [1, 128, 4, 3]; mean exactly, sample within libm's exp ulps.
  const std::int64_t mh = 8, mw = 6;
  std::vector<float> mom(static_cast<std::size_t>(mh * mw * 64)), bm(128), bv(128);
  for (std::size_t i = 0; i < mom.size(); ++i) mom[i] = std::sin(static_cast<float>(i) * 0.37f) * 3.0f;
  for (std::size_t i = 0; i < mom.size(); i += 64) mom[i + 40] = -40.0f;  // exercises the logvar clamp
  for (int i = 0; i < 128; ++i) {
    bm[static_cast<std::size_t>(i)] = 0.01f * static_cast<float>(i) - 0.5f;
    bv[static_cast<std::size_t>(i)] = 0.5f + 0.013f * static_cast<float>(i);
  }
  const std::vector<float> noise = mi::rng::torch_cpu_randn(7, 32 * mh * mw);
  DevTensor md(mi::DType::kF32, {1, mh, mw, 64}), bmd(mi::DType::kF32, {128}), bvd(mi::DType::kF32, {128}),
      nd(mi::DType::kF32, {1, 32, mh, mw}), y(mi::DType::kF32, {1, 128, mh / 2, mw / 2});
  md.upload(mom);
  bmd.upload(bm);
  bvd.upload(bv);
  nd.upload(noise);
  for (const bool sample : {false, true}) {
    mi::vae::encoder_latent_out(md.t, bmd.t, bvd.t, 1e-4f, sample ? &nd.t : nullptr, y.t, nullptr);
    got = y.download<float>();
    double max_abs = 0, absmax = 0;
    for (int k = 0; k < 128; ++k)
      for (std::int64_t i = 0; i < mh / 2; ++i)
        for (std::int64_t j = 0; j < mw / 2; ++j) {
          const int c = k / 4, p = (k / 2) % 2, qq = k % 2;
          const std::int64_t row = 2 * i + p, col = 2 * j + qq;
          volatile float z = mom[static_cast<std::size_t>((row * mw + col) * 64 + c)];
          if (sample) {
            const float lv = std::min(std::max(mom[static_cast<std::size_t>((row * mw + col) * 64 + 32 + c)], -30.0f), 20.0f);
            volatile float half = 0.5f * lv;
            volatile float sd = std::exp(half);
            volatile float prod = sd * noise[static_cast<std::size_t>((c * mh + row) * mw + col)];
            z = z + prod;
          }
          volatile float var_eps = bv[static_cast<std::size_t>(k)] + 1e-4f;
          volatile float sdv = std::sqrt(var_eps);
          volatile float diff = z - bm[static_cast<std::size_t>(k)];
          const float want = diff / sdv;
          const float g = got[static_cast<std::size_t>((k * (mh / 2) + i) * (mw / 2) + j)];
          if (!sample) {
            ASSERT_EQ(std::memcmp(&want, &g, 4), 0) << "k " << k << " i " << i << " j " << j;
          }
          max_abs = std::max(max_abs, std::abs(static_cast<double>(g) - want));
          absmax = std::max(absmax, std::abs(static_cast<double>(want)));
        }
    // The sample's exp() is libm's on the host and CUDA's on the device (each within 1-2 ulp).
    std::printf("[latent_out] %s: max |diff| vs host %.3e (output range %.3f)\n", sample ? "sample" : "mean", max_abs, absmax);
    EXPECT_LE(max_abs, 1e-6 * absmax);
  }
}

// ---- Loader ---------------------------------------------------------------------------------------------------

struct Spec {
  std::string name;
  mi::weights::DType dtype;
  std::vector<std::int64_t> shape;
};

// A valid safetensors file with these tensors; the data section is a sparse zero extent (the loader's checks read
// headers only), so full-size encoder layouts cost no disk writes.
void write_sparse_safetensors(const fs::path& path, const std::vector<Spec>& specs) {
  std::string header = "{";
  std::uint64_t cursor = 0;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    std::uint64_t n = mi::weights::dtype_size(specs[i].dtype);
    std::string shape = "[";
    for (std::size_t d = 0; d < specs[i].shape.size(); ++d) {
      shape += (d ? "," : "") + std::to_string(specs[i].shape[d]);
      n *= static_cast<std::uint64_t>(specs[i].shape[d]);
    }
    header += (i ? "," : "") + std::string("\"") + specs[i].name + "\":{\"dtype\":\"" + mi::weights::dtype_name(specs[i].dtype) +
              "\",\"shape\":" + shape + "],\"data_offsets\":[" + std::to_string(cursor) + "," + std::to_string(cursor + n) + "]}";
    cursor += n;
  }
  header += "}";
  {
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    ASSERT_NE(f, nullptr);
    const std::uint64_t len = header.size();
    for (int i = 0; i < 8; ++i) std::fputc(static_cast<int>((len >> (8 * i)) & 0xFF), f);
    std::fwrite(header.data(), 1, header.size(), f);
    std::fclose(f);
  }
  fs::resize_file(path, 8 + header.size() + cursor);
}

// Names, dtypes and shapes from the bundle VAE file, minus the decoder: the encoder layout in either key style.
TEST(VaeEncoderLoader, RejectsMissingExtraMistypedTensorsAndBadPrefixes) {
  if (!fs::is_regular_file(vae_file())) GTEST_SKIP() << "weights not present (" << vae_file() << ")";
  std::vector<Spec> base;
  const mi::weights::SafetensorsFile header(vae_file());
  for (const auto& t : header.tensors())
    if (t.name.rfind("decoder.", 0) != 0 && t.name.rfind("post_quant_conv.", 0) != 0) base.push_back({t.name, t.dtype, t.shape});
  const mi::testing::TempDir tmp("vae_encoder_loader");
  const fs::path& dir = tmp.path();
  const std::size_t want = Flux2VaeEncoder::persistent_bytes(vae_file());
  int n = 0;
  const auto bytes_of = [&](std::vector<Spec> specs) {
    const fs::path f = dir / ("case" + std::to_string(n++) + ".safetensors");
    write_sparse_safetensors(f, specs);
    return Flux2VaeEncoder::persistent_bytes(f);
  };
  const auto with = [&](auto&& edit) {
    std::vector<Spec> v = base;
    edit(v);
    return v;
  };
  const auto find = [](std::vector<Spec>& v, const std::string& name) -> Spec& {
    return *std::find_if(v.begin(), v.end(), [&](const Spec& sp) { return sp.name == name; });
  };
  const auto prefixed = [&](const std::string& prefix) {
    return with([&](std::vector<Spec>& v) {
      for (Spec& sp : v) sp.name = prefix + sp.name;
    });
  };

  EXPECT_EQ(bytes_of(base), want);
  EXPECT_EQ(bytes_of(prefixed("vae.")), want);  // a common prefix ending in '.'
  EXPECT_EQ(bytes_of(prefixed("first_stage_model.")), want);
  // Missing tensor.
  EXPECT_THROW(bytes_of(with([](std::vector<Spec>& v) {
                 v.erase(std::remove_if(v.begin(), v.end(), [](const Spec& sp) { return sp.name == "encoder.conv_out.bias"; }), v.end());
               })),
               std::runtime_error);
  // BF16 tensor.
  EXPECT_THROW(bytes_of(with([&](std::vector<Spec>& v) { find(v, "encoder.conv_in.bias").dtype = mi::weights::DType::kBF16; })),
               std::runtime_error);
  // Wrong shape (same element count).
  EXPECT_THROW(bytes_of(with([&](std::vector<Spec>& v) { find(v, "encoder.conv_out.weight").shape = {64, 512, 9, 1}; })),
               std::runtime_error);
  // Extra tensors under encoder. / quant_conv.
  EXPECT_THROW(bytes_of(with([](std::vector<Spec>& v) { v.push_back({"encoder.down_blocks.0.resnets.2.norm1.weight", mi::weights::DType::kF32, {128}}); })),
               std::runtime_error);
  EXPECT_THROW(bytes_of(with([](std::vector<Spec>& v) { v.push_back({"quant_conv.extra", mi::weights::DType::kF32, {4}}); })),
               std::runtime_error);
  // Tensors outside the encoder are ignored.
  EXPECT_EQ(bytes_of(with([](std::vector<Spec>& v) { v.push_back({"decoder.conv_in.bias", mi::weights::DType::kF32, {128}}); })), want);
  // A prefix that does not end in '.' does not name an encoder; two encoders are ambiguous.
  EXPECT_THROW(bytes_of(prefixed("vae")), std::runtime_error);
  EXPECT_THROW(bytes_of(with([&](std::vector<Spec>& v) {
                 std::vector<Spec> b = prefixed("b.");
                 for (Spec& sp : v) sp.name = "a." + sp.name;
                 v.insert(v.end(), b.begin(), b.end());
               })),
               std::runtime_error);
  // Without bn.* the file needs latent_stats from another index.
  const std::vector<Spec> no_bn = with([](std::vector<Spec>& v) {
    v.erase(std::remove_if(v.begin(), v.end(), [](const Spec& sp) { return sp.name.rfind("bn.", 0) == 0; }), v.end());
  });
  EXPECT_THROW(bytes_of(no_bn), std::runtime_error);
  const fs::path bundle_dir = vae_file().parent_path().parent_path();
  if (fs::is_regular_file(bundle_dir / "moss.mossidx")) {
    const auto index = mi::weights::load_index(bundle_dir / "moss.mossidx", bundle_dir);
    const fs::path f = dir / "no_bn.safetensors";
    write_sparse_safetensors(f, no_bn);
    EXPECT_EQ(Flux2VaeEncoder::persistent_bytes(f, &index), want);
  }
}

// ---- Model ----------------------------------------------------------------------------------------------------

class VaeEncoderTest : public ::testing::Test {
 protected:
  static constexpr std::int64_t kMaxSide = 2048;
  static void SetUpTestSuite() {
    if (!fs::is_regular_file(vae_file())) return;
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    persistent_ = Flux2VaeEncoder::persistent_bytes(vae_file());
    // Room for the 1456^2 untiled encode; larger images run tiled within it.
    const std::size_t bytes = persistent_ + Flux2VaeEncoder::workspace_bytes(1456, 1456);
    std::printf("[vae_encoder] arena %.3f GiB (free %.3f GiB), weights %.1f MB\n", static_cast<double>(bytes) / kGiB,
                static_cast<double>(mi::runtime::DeviceArena::free_device_bytes()) / kGiB, static_cast<double>(persistent_) / 1e6);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(bytes);
    model_ = std::make_unique<Flux2VaeEncoder>(vae_file(), *arena_, *streams_);
    const auto& st = model_->load_stats();
    std::printf("[vae_encoder] load: %.1f MB in %.3f s, persistent %.1f MB\n", static_cast<double>(st.bytes) / 1e6, st.seconds,
                static_cast<double>(arena_->persistent_used()) / 1e6);
  }
  static void TearDownTestSuite() {
    model_.reset();
    arena_.reset();
    streams_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "weights not present (" << vae_file() << ")";
  }
  cudaStream_t stream() const { return streams_->compute(); }
  std::size_t budget() const { return arena_->available(); }

  // rgb u8 [h, w, 3] host -> latents (host copy).
  std::vector<float> encode(const std::vector<std::uint8_t>& rgb, std::int64_t h, std::int64_t w, const EncodeOptions& o,
                            const std::optional<TilePlan>& plan = std::nullopt, const mi::flux2_vae_encoder::StageHook& hook = {}) {
    DevTensor lat(mi::DType::kF32, {128, h / 16, w / 16});
    const Tensor src = Tensor::view(const_cast<std::uint8_t*>(rgb.data()), mi::DType::kU8, mi::Device::kCPU, {h, w, 3});
    const TilePlan p = plan ? *plan : *Flux2VaeEncoder::plan_tiled(h, w, budget());
    model_->encode(src, lat.t, o, p, stream(), hook);
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    return lat.download<float>();
  }

  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<Flux2VaeEncoder> model_;
  static std::size_t persistent_;
};

std::unique_ptr<mi::runtime::StreamSet> VaeEncoderTest::streams_;
std::unique_ptr<mi::runtime::DeviceArena> VaeEncoderTest::arena_;
std::unique_ptr<Flux2VaeEncoder> VaeEncoderTest::model_;
std::size_t VaeEncoderTest::persistent_ = 0;

struct Rgb {
  std::vector<std::uint8_t> px;
  std::int64_t h = 0, w = 0;
};
Rgb load_rgb(const fs::path& p) {
  const auto a = mi::npy::load(p);
  return {a.to_vector<std::uint8_t>(), a.shape[0], a.shape[1]};
}

// Moments [1, 64, h, w] NCHW (reference) vs the hook's NHWC [1, h, w, 64].
std::vector<float> nhwc_to_nchw(const std::vector<float>& x, std::int64_t h, std::int64_t w, std::int64_t c) {
  std::vector<float> y(x.size());
  for (std::int64_t i = 0; i < h * w; ++i)
    for (std::int64_t k = 0; k < c; ++k) y[static_cast<std::size_t>(k * h * w + i)] = x[static_cast<std::size_t>(i * c + k)];
  return y;
}

TEST_F(VaeEncoderTest, SmallCropMatchesDiffusers) {
  const Rgb img = load_rgb(data_dir() / "small.rgb.npy");
  std::vector<float> moments;
  const char* dump = std::getenv("MI_VAE_ENCODER_DUMP");  // raw fp32 NHWC stage outputs, for debugging
  const auto hook = [&](const char* stage, const Tensor& t) {
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    if (dump != nullptr && *dump != '\0') {
      std::vector<float> v(static_cast<std::size_t>(t.numel()));
      MI_CUDA_CHECK(cudaMemcpy(v.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost));
      const fs::path f = fs::path(dump) / (std::string(stage) + "." + std::to_string(t.dim(1)) + "x" + std::to_string(t.dim(2)) +
                                           "x" + std::to_string(t.dim(3)) + ".f32");
      std::FILE* fp = std::fopen(f.string().c_str(), "wb");
      if (fp != nullptr) {
        std::fwrite(v.data(), 4, v.size(), fp);
        std::fclose(fp);
      }
    }
    if (std::string(stage) != "moments") return;
    moments.resize(static_cast<std::size_t>(t.numel()));
    MI_CUDA_CHECK(cudaMemcpy(moments.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost));
  };
  const auto mean = encode(img.px, img.h, img.w, {}, Flux2VaeEncoder::plan_untiled(img.h, img.w), hook);
  const Stats sm = compare(load_f32(data_dir() / "small.mean.npy"), mean);
  print_stats("vae_encoder", "small 192x256 mean", sm);
  const Stats sq = compare(load_f32(data_dir() / "small.moments.npy"), nhwc_to_nchw(moments, img.h / 8, img.w / 8, 64));
  print_stats("vae_encoder", "small 192x256 moments", sq);
  EncodeOptions so;
  so.mode = LatentMode::kSample;
  so.seed = 42;
  const auto sample = encode(img.px, img.h, img.w, so, Flux2VaeEncoder::plan_untiled(img.h, img.w));
  const Stats ss = compare(load_f32(data_dir() / "small.sample42.npy"), sample);
  print_stats("vae_encoder", "small 192x256 sample seed 42", ss);
  const Stats sf = compare(load_f32(data_dir() / "small.mean_fp32.npy"), mean);
  print_stats("vae_encoder", "small 192x256 mean vs fp32 ref", sf);
  for (const Stats* s : {&sm, &sq, &ss, &sf}) {
    EXPECT_GE(s->cosine, 0.99999);
    EXPECT_LE(s->max_rel, 2e-3);
  }
  // Single-pass TF32 convs: reported, held to a looser bound.
  EncodeOptions fast;
  fast.conv = mi::vae::ConvMath::kTf32;
  fast.attention = mi::vae::AttnPrecision::kTf32;
  fast.fast_linears = true;
  const Stats st = compare(load_f32(data_dir() / "small.mean.npy"), encode(img.px, img.h, img.w, fast, Flux2VaeEncoder::plan_untiled(img.h, img.w)));
  print_stats("vae_encoder", "small mean, TF32 convs+attention", st);
  EXPECT_GE(st.cosine, 0.9999);
  EXPECT_NE(0, std::memcmp(mean.data(), sample.data(), mean.size() * 4));
  // Deterministic, also through a device-resident input.
  EXPECT_EQ(0, std::memcmp(sample.data(), encode(img.px, img.h, img.w, so, Flux2VaeEncoder::plan_untiled(img.h, img.w)).data(),
                           sample.size() * 4));
  DevTensor rgb_d(mi::DType::kU8, {img.h, img.w, 3}), lat(mi::DType::kF32, {1, 128, img.h / 16, img.w / 16});
  rgb_d.upload(img.px);
  model_->encode(rgb_d.t, lat.t, {}, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(0, std::memcmp(mean.data(), lat.download<float>().data(), mean.size() * 4));
}

TEST_F(VaeEncoderTest, FloatInputEqualsU8Input) {
  const Rgb img = load_rgb(data_dir() / "small.rgb.npy");
  std::vector<float> f(img.px.size());
  for (std::size_t i = 0; i < f.size(); ++i) f[i] = static_cast<float>(img.px[i]) / 255.0f;
  DevTensor lat(mi::DType::kF32, {128, img.h / 16, img.w / 16});
  model_->encode(Tensor::view(f.data(), mi::DType::kF32, mi::Device::kCPU, {img.h, img.w, 3}), lat.t, {}, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const auto a = lat.download<float>();
  const auto b = encode(img.px, img.h, img.w, {}, Flux2VaeEncoder::plan_untiled(img.h, img.w));
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * 4));
}

// Every banded configuration gives the untiled bytes: tiny bands (2 rows, at the halo limit), bands that do not
// divide the level, mixed per-level choices.
TEST_F(VaeEncoderTest, BandedLevelsAreByteIdenticalSmall) {
  const Rgb img = load_rgb(data_dir() / "small.rgb.npy");
  const auto ref = encode(img.px, img.h, img.w, {}, Flux2VaeEncoder::plan_untiled(img.h, img.w));
  const std::array<std::int64_t, 3> configs[] = {{2, 2, 2}, {6, 10, 14}, {16, 0, 4}, {0, 30, 0}, {100, 50, 0}, {256, 128, 64}};
  for (const auto& b : configs) {
    const TilePlan p = Flux2VaeEncoder::make_plan(img.h, img.w, b);
    const auto got = encode(img.px, img.h, img.w, {}, p);
    EXPECT_EQ(0, std::memcmp(ref.data(), got.data(), ref.size() * 4)) << p.describe();
    std::printf("[vae_encoder] banded identical: %s\n", p.describe().c_str());
  }
  // Single-pass TF32 convs + attention: banded equals untiled too.
  EncodeOptions fast;
  fast.conv = mi::vae::ConvMath::kTf32;
  fast.attention = mi::vae::AttnPrecision::kTf32;
  const auto fast_ref = encode(img.px, img.h, img.w, fast, Flux2VaeEncoder::plan_untiled(img.h, img.w));
  const auto fast_banded = encode(img.px, img.h, img.w, fast, Flux2VaeEncoder::make_plan(img.h, img.w, {6, 10, 14}));
  EXPECT_EQ(0, std::memcmp(fast_ref.data(), fast_banded.data(), fast_ref.size() * 4));
  EXPECT_NE(0, std::memcmp(fast_ref.data(), ref.data(), ref.size() * 4));
  // Invalid plans are refused by make_plan and by encode() (a hand-edited plan must not loop or misindex).
  EXPECT_THROW(Flux2VaeEncoder::make_plan(img.h, img.w, {3, 0, 0}), std::invalid_argument);
  for (const std::int64_t bad : {std::int64_t{1}, std::int64_t{-2}, std::int64_t{5}}) {
    TilePlan p = Flux2VaeEncoder::plan_untiled(img.h, img.w);
    p.band_rows[1] = bad;
    EXPECT_THROW(encode(img.px, img.h, img.w, {}, p), std::invalid_argument) << bad;
  }
  EXPECT_THROW(Flux2VaeEncoder::make_plan(250, 192, {0, 0, 0}), std::invalid_argument);
}

TEST_F(VaeEncoderTest, OriginalSingleFileLayoutMatchesDiffusers) {
  if (!fs::is_regular_file(original_layout_file())) GTEST_SKIP() << "absent: " << original_layout_file();
  const fs::path ref = data_dir() / "small_original_layout.mean.npy";
  ASSERT_TRUE(fs::is_regular_file(ref));
  // The single file's encoder with the bundle's latent statistics (what the pipeline normalizes with).
  const fs::path bundle_dir = vae_file().parent_path().parent_path();
  if (!fs::is_regular_file(bundle_dir / "moss.mossidx")) GTEST_SKIP() << "bundle index absent";
  const auto index = mi::weights::load_index(bundle_dir / "moss.mossidx", bundle_dir);
  const std::size_t bytes = Flux2VaeEncoder::persistent_bytes(original_layout_file());
  EXPECT_EQ(bytes, persistent_);
  const Rgb img = load_rgb(data_dir() / "small.rgb.npy");
  const std::size_t ws = Flux2VaeEncoder::workspace_bytes(img.h, img.w);
  mi::runtime::DeviceArena arena(bytes + ws);
  Flux2VaeEncoder enc(original_layout_file(), arena, *streams_, false, &index);
  DevTensor lat(mi::DType::kF32, {128, img.h / 16, img.w / 16});
  enc.encode(Tensor::view(const_cast<std::uint8_t*>(img.px.data()), mi::DType::kU8, mi::Device::kCPU, {img.h, img.w, 3}), lat.t, {},
             stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const auto got = lat.download<float>();
  const Stats s = compare(load_f32(ref), got);
  print_stats("vae_encoder", "original layout small mean", s);
  EXPECT_GE(s.cosine, 0.99999);
  EXPECT_LE(s.max_rel, 1e-3);
  // A different checkpoint (its encoder weights differ slightly from the bundle's), so the latents differ.
  const auto bundle = encode(img.px, img.h, img.w, {}, Flux2VaeEncoder::plan_untiled(img.h, img.w));
  print_stats("vae_encoder", "original-layout file vs bundle", compare(bundle, got));
  EXPECT_NE(0, std::memcmp(bundle.data(), got.data(), got.size() * 4));
}

struct LargeCase {
  const char* name;
  std::int64_t h, w;
};
constexpr LargeCase kLarge[] = {{"fridge_1024", 1024, 1024}, {"crop_1248x1664", 1664, 1248}, {"fridge_1456", 1456, 1456},
                                {"bench_2048", 2048, 2048}};

// The local full-size cases present (generator run with --large).
std::vector<LargeCase> present_large_cases() {
  std::vector<LargeCase> v;
  for (const LargeCase& c : kLarge)
    if (fs::is_regular_file(large_dir() / (std::string(c.name) + ".rgb.npy")) &&
        fs::is_regular_file(large_dir() / (std::string(c.name) + ".mean.npy")))
      v.push_back(c);
  return v;
}

TEST_F(VaeEncoderTest, FullSizeImagesMatchDiffusers) {
  const std::vector<LargeCase> cases = present_large_cases();
  if (cases.empty()) GTEST_SKIP() << "local references absent: " << large_dir();
  std::printf("[vae_encoder] %zu full-size case(s)\n", cases.size());
  for (const LargeCase& c : cases) {
    const fs::path rgb_path = large_dir() / (std::string(c.name) + ".rgb.npy");
    const Rgb img = load_rgb(rgb_path);
    ASSERT_EQ(img.h, c.h);
    ASSERT_EQ(img.w, c.w);
    const std::optional<TilePlan> plan = Flux2VaeEncoder::plan_tiled(img.h, img.w, budget());
    ASSERT_TRUE(plan.has_value());
    const auto got = encode(img.px, img.h, img.w, {}, plan);
    const std::string label = std::string(c.name) + (plan->tiled() ? " (banded)" : " (untiled)");
    const Stats s = compare(load_f32(large_dir() / (std::string(c.name) + ".mean.npy")), got);
    print_stats("vae_encoder", label, s);
    EXPECT_GE(s.cosine, 0.99999) << c.name;
    EXPECT_LE(s.max_rel, 2e-3) << c.name;
    const fs::path fp32 = large_dir() / (std::string(c.name) + ".mean_fp32.npy");
    if (fs::is_regular_file(fp32)) {
      const Stats sf = compare(load_f32(fp32), got);
      print_stats("vae_encoder", label + " vs fp32 ref", sf);
      EXPECT_GE(sf.cosine, 0.99999) << c.name;
      const Stats rr = compare(load_f32(fp32), load_f32(large_dir() / (std::string(c.name) + ".mean.npy")));
      print_stats("vae_encoder", std::string(c.name) + " ref (TF32) vs fp32 ref", rr);
    }
    EncodeOptions fast;
    fast.conv = mi::vae::ConvMath::kTf32;
    const Stats st = compare(load_f32(large_dir() / (std::string(c.name) + ".mean.npy")), encode(img.px, img.h, img.w, fast, plan));
    print_stats("vae_encoder", label + " TF32 convs", st);
    EXPECT_GE(st.cosine, 0.9999) << c.name;
  }
}

TEST_F(VaeEncoderTest, BandedLevelsAreByteIdenticalFullSize) {
  std::vector<std::string> names;
  for (const LargeCase& c : present_large_cases())
    if (std::string(c.name) == "fridge_1024" || std::string(c.name) == "crop_1248x1664") names.push_back(c.name);
  if (names.empty()) GTEST_SKIP() << "local references absent: " << large_dir();
  for (const std::string& name : names) {
    const fs::path rgb_path = large_dir() / (name + ".rgb.npy");
    const Rgb img = load_rgb(rgb_path);
    const auto ref = encode(img.px, img.h, img.w, {}, Flux2VaeEncoder::plan_untiled(img.h, img.w));
    for (const auto& b : {std::array<std::int64_t, 3>{128, 128, 128}, std::array<std::int64_t, 3>{512, 0, 0},
                          std::array<std::int64_t, 3>{94, 62, 30}}) {
      const TilePlan p = Flux2VaeEncoder::make_plan(img.h, img.w, b);
      const auto got = encode(img.px, img.h, img.w, {}, p);
      EXPECT_EQ(0, std::memcmp(ref.data(), got.data(), ref.size() * 4)) << name << " " << p.describe();
      std::printf("[vae_encoder] %s banded identical: %s\n", name.c_str(), p.describe().c_str());
    }
  }
}

// 2048^2 untiled: the 128-channel full-resolution tensors are 2 GiB, past CUTLASS's 32-bit offsets, so their convs
// run in row strips; the result must equal the banded encode byte for byte.
TEST_F(VaeEncoderTest, Untiled2048StripsEqualBanded) {
  const fs::path rgb_path = large_dir() / "bench_2048.rgb.npy";
  if (!fs::is_regular_file(rgb_path)) GTEST_SKIP() << "local reference absent: " << rgb_path;
  const std::size_t ws = Flux2VaeEncoder::workspace_bytes(2048, 2048);
  if (persistent_ + ws + (std::size_t{512} << 20) > mi::runtime::DeviceArena::free_device_bytes())
    GTEST_SKIP() << "not enough free device memory for the untiled 2048^2 encode";
  const Rgb img = load_rgb(rgb_path);
  const auto banded = encode(img.px, img.h, img.w, {}, Flux2VaeEncoder::make_plan(2048, 2048, {256, 256, 0}));
  std::vector<float> untiled;
  {
    mi::runtime::DeviceArena arena(persistent_ + ws);
    Flux2VaeEncoder enc(vae_file(), arena, *streams_);
    DevTensor lat(mi::DType::kF32, {128, 128, 128});
    enc.encode(Tensor::view(const_cast<std::uint8_t*>(img.px.data()), mi::DType::kU8, mi::Device::kCPU, {2048, 2048, 3}), lat.t,
               {}, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    untiled = lat.download<float>();
  }
  EXPECT_EQ(0, std::memcmp(untiled.data(), banded.data(), banded.size() * 4));
  std::printf("[vae_encoder] 2048^2 untiled (%.2f GiB, strips) == banded\n", static_cast<double>(ws) / kGiB);
}

TEST(VaeEncoderPlan, BudgetsAndQueries) {
  // plan_min_memory leaves levels of at most 16 rows untiled: never more memory than one band with its statistics
  // passes, and no recompute.
  for (const std::int64_t side : {std::int64_t{32}, std::int64_t{64}, std::int64_t{128}}) {
    std::array<std::int64_t, 3> banded{};
    for (int l = 0; l < 3; ++l) banded[static_cast<std::size_t>(l)] = std::min<std::int64_t>(16, side >> l);
    const TilePlan old_plan = Flux2VaeEncoder::make_plan(side, side, banded), m = Flux2VaeEncoder::plan_min_memory(side, side);
    std::printf("[vae_encoder plan] %lld^2 min-memory: all-banded %zu B, small levels untiled %zu B (%s)\n",
                static_cast<long long>(side), old_plan.workspace_bytes, m.workspace_bytes, m.describe().c_str());
    EXPECT_LE(m.workspace_bytes, old_plan.workspace_bytes);
    EXPECT_LE(m.relative_cost, old_plan.relative_cost);
  }
  for (const auto& [h, w] : {std::pair<std::int64_t, std::int64_t>{1024, 1024}, {1664, 1248}, {1456, 1456}, {2048, 2048},
                             {4096, 4096}}) {
    const TilePlan u = Flux2VaeEncoder::plan_untiled(h, w);
    const TilePlan m = Flux2VaeEncoder::plan_min_memory(h, w);
    EXPECT_EQ(u.workspace_bytes, Flux2VaeEncoder::workspace_bytes(h, w));
    EXPECT_LT(m.workspace_bytes, u.workspace_bytes);
    std::printf("[vae_encoder plan] untiled  %s\n[vae_encoder plan] min-mem  %s\n", u.describe().c_str(), m.describe().c_str());
    for (const double gib : {0.75, 1.0, 1.5, 2.0, 3.0}) {
      const auto p = Flux2VaeEncoder::plan_tiled(h, w, static_cast<std::size_t>(gib * kGiB));
      if (!p) {
        EXPECT_GT(m.workspace_bytes, static_cast<std::size_t>(gib * kGiB));
        std::printf("[vae_encoder plan]   %.2f GiB: no plan\n", gib);
        continue;
      }
      EXPECT_LE(p->workspace_bytes, static_cast<std::size_t>(gib * kGiB));
      std::printf("[vae_encoder plan]   %.2f GiB: %s\n", gib, p->describe().c_str());
    }
    EXPECT_EQ(Flux2VaeEncoder::plan_tiled(h, w, u.workspace_bytes)->band_rows, u.band_rows);
    EXPECT_FALSE(Flux2VaeEncoder::plan_tiled(h, w, m.workspace_bytes - 1).has_value());
  }
}

TEST_F(VaeEncoderTest, WorkspaceQueryCoversPeakAndTiming) {
  struct Size {
    std::int64_t h, w;
  };
  for (const Size& sz : {Size{1024, 1024}, Size{1664, 1248}, Size{1456, 1456}, Size{2048, 2048}}) {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(sz.h * sz.w * 3));
    for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<std::uint8_t>((i * 2654435761u) >> 24);
    std::vector<TilePlan> plans;
    if (Flux2VaeEncoder::workspace_bytes(sz.h, sz.w) <= budget()) plans.push_back(Flux2VaeEncoder::plan_untiled(sz.h, sz.w));
    for (const double gib : {1.5, 1.0}) {
      const auto p = Flux2VaeEncoder::plan_tiled(sz.h, sz.w, static_cast<std::size_t>(gib * kGiB));
      if (p && p->tiled()) plans.push_back(*p);
    }
    DevTensor rgb_d(mi::DType::kU8, {sz.h, sz.w, 3}), lat(mi::DType::kF32, {128, sz.h / 16, sz.w / 16});
    rgb_d.upload(rgb);
    for (const TilePlan& p : plans) {
      model_->encode(rgb_d.t, lat.t, {}, p, stream());  // warm-up
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      arena_->reset_high_water();
      constexpr int kReps = 3;
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kReps; ++r) model_->encode(rgb_d.t, lat.t, {}, p, stream());
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / kReps;
      const std::size_t peak = arena_->high_water() - arena_->persistent_used();
      EXPECT_LE(peak, p.workspace_bytes);
      std::printf("[vae_encoder timing, dev] %s: %.1f ms, stack peak %.1f MiB\n", p.describe().c_str(), ms,
                  static_cast<double>(peak) / kMiB);
    }
    // A/B in the same hold: single-pass TF32 convs + attention, and the per-phase split of the default encode.
    if (!plans.empty()) {
      EncodeOptions fast;
      fast.conv = mi::vae::ConvMath::kTf32;
      fast.attention = mi::vae::AttnPrecision::kTf32;
      fast.fast_linears = true;
      model_->encode(rgb_d.t, lat.t, fast, plans[0], stream());
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      const auto t0 = std::chrono::steady_clock::now();
      model_->encode(rgb_d.t, lat.t, fast, plans[0], stream());
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      std::printf("[vae_encoder timing, dev]   same plan, TF32 convs + attention: %.1f ms\n",
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      std::vector<std::pair<std::string, cudaEvent_t>> ev;
      cudaEvent_t start;
      MI_CUDA_CHECK(cudaEventCreate(&start));
      MI_CUDA_CHECK(cudaEventRecord(start, stream()));
      model_->encode(rgb_d.t, lat.t, {}, plans[0], stream(), [&](const char* stage, const Tensor&) {
        cudaEvent_t e;
        MI_CUDA_CHECK(cudaEventCreate(&e));
        MI_CUDA_CHECK(cudaEventRecord(e, stream()));
        ev.emplace_back(stage, e);
      });
      MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
      std::string line;
      cudaEvent_t prev = start;
      for (auto& [name, e] : ev) {
        float t = 0;
        MI_CUDA_CHECK(cudaEventElapsedTime(&t, prev, e));
        line += " " + name + " " + std::to_string(static_cast<int>(t + 0.5f));
        prev = e;
      }
      std::printf("[vae_encoder timing, dev]   phases (ms, cumulative order):%s\n", line.c_str());
      for (auto& [name, e] : ev) MI_CUDA_CHECK(cudaEventDestroy(e));
      MI_CUDA_CHECK(cudaEventDestroy(start));
    }
  }
}

}  // namespace
