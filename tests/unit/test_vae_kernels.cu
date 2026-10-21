// kernels/vae against torch (tools/reference/gen_vae_kernels_vectors.py): convs with the real decoder weights
// at realistic shapes (sampled outputs vs cuDNN TF32 and strict fp32, plus a full-tensor fp32 direct-conv
// oracle), GroupNorm (+ two-pass/tiled equality), fp32 attention, the mid-block attention block, and the
// exact ops (upsample, BN unpack, RGB8 postprocess, layout, residual add) bit for bit. Ends with a micro
// benchmark (RecordProperty, never asserted).
#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/vae/vae_kernels.h"
#include "util/npy.h"
#include "weights/safetensors.h"

namespace {

namespace fs = std::filesystem;
using mi::DeviceBuffer;
using mi::Device;
using mi::DType;
using mi::Tensor;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/vae_kernels";

fs::path vae_file() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env && *env ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "base/vae/diffusion_pytorch_model.safetensors";
}

#define REQUIRE_WEIGHTS() \
  if (!fs::is_regular_file(vae_file())) GTEST_SKIP() << "VAE weights not present at " << vae_file()

template <typename T>
std::vector<T> load(const std::string& name) {
  return mi::npy::load(kData / (name + ".npy")).to_vector<T>();
}

// Mirrors gen_vae_kernels_vectors.py hash_u32 / synth_normal bit for bit (the arithmetic has no a*b+c pattern
// for the compiler to contract).
std::uint32_t hash_u32(std::uint32_t seed, std::uint32_t i) {
  std::uint32_t x = i * 0x9E3779B1u + seed * 0x85EBCA77u;
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

std::vector<float> synth_normal(std::uint32_t seed, std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    float u[4];
    for (int j = 0; j < 4; ++j) u[j] = static_cast<float>(hash_u32(seed, static_cast<std::uint32_t>(4 * i + j)) >> 8) * 0x1p-24f;
    const float s = (u[0] + u[1]) + (u[2] + u[3]);
    out[i] = (s - 2.0f) * 1.7320508f;
  }
  return out;
}

std::uint16_t to_bf16(float f) {  // round to nearest even (no NaNs here)
  const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  return static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }

std::vector<float> read_weight(const std::string& name, std::vector<std::int64_t>* shape = nullptr) {
  mi::weights::SafetensorsFile file(vae_file());
  const auto* info = file.find(name);
  if (!info || info->dtype != mi::weights::DType::kF32) throw std::runtime_error("missing f32 weight " + name);
  std::vector<float> out(static_cast<std::size_t>(info->numel()));
  std::ifstream in(vae_file(), std::ios::binary);
  in.seekg(static_cast<std::streamoff>(info->offset));
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(info->nbytes));
  if (!in) throw std::runtime_error("short read for " + name);
  if (shape) *shape = info->shape;
  return out;
}

struct Dev {
  DeviceBuffer buf;
  Tensor t;
};

template <typename T>
Dev upload(const std::vector<T>& host, DType dt, std::initializer_list<std::int64_t> shape) {
  Dev d{DeviceBuffer(std::max<std::size_t>(host.size() * sizeof(T), 16)), {}};
  if (!host.empty()) d.buf.upload(host.data(), host.size() * sizeof(T));
  d.t = Tensor::view(d.buf.get(), dt, Device::kCUDA, shape);
  return d;
}

Dev alloc(DType dt, std::initializer_list<std::int64_t> shape) {
  Tensor probe = Tensor::view(nullptr, dt, Device::kCUDA, shape);
  Dev d{DeviceBuffer(std::max<std::size_t>(probe.nbytes(), 16)), {}};
  d.t = Tensor::view(d.buf.get(), dt, Device::kCUDA, shape);
  return d;
}

template <typename T>
std::vector<T> download(const Dev& d) {
  std::vector<T> out(static_cast<std::size_t>(d.t.numel()));
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  d.buf.download(out.data(), out.size() * sizeof(T));
  return out;
}

struct Stats {
  double cos = 0, max_abs = 0, ref_absmax = 0, rel = 0;  // rel = max_abs / ref_absmax
};

Stats compare(const std::vector<double>& got, const std::vector<double>& ref) {
  double dot = 0, ng = 0, nr = 0;
  Stats s;
  for (std::size_t i = 0; i < got.size(); ++i) {
    dot += got[i] * ref[i];
    ng += got[i] * got[i];
    nr += ref[i] * ref[i];
    s.max_abs = std::max(s.max_abs, std::fabs(got[i] - ref[i]));
    s.ref_absmax = std::max(s.ref_absmax, std::fabs(ref[i]));
  }
  s.cos = dot / std::sqrt(ng * nr);
  s.rel = s.max_abs / s.ref_absmax;
  return s;
}

template <typename T, typename F>
Stats compare_sampled(const std::vector<T>& full, const std::vector<std::int32_t>& idx, const std::vector<float>& ref,
                      F to_double) {
  std::vector<double> g(idx.size()), r(idx.size());
  for (std::size_t i = 0; i < idx.size(); ++i) {
    g[i] = to_double(full[static_cast<std::size_t>(idx[i])]);
    r[i] = ref[i];
  }
  return compare(g, r);
}

Stats compare_sampled(const std::vector<float>& full, const std::vector<std::int32_t>& idx, const std::vector<float>& ref) {
  return compare_sampled(full, idx, ref, [](float v) { return static_cast<double>(v); });
}

Stats compare_sampled(const std::vector<float>& full, const std::vector<std::int32_t>& idx, const std::vector<double>& ref) {
  std::vector<double> g(idx.size());
  for (std::size_t i = 0; i < idx.size(); ++i) g[i] = full[static_cast<std::size_t>(idx[i])];
  return compare(g, ref);
}

// The fp32 reference's own distance from the float64 truth, for context in the reports.
Stats ref_error(const std::vector<float>& ref, const std::vector<double>& ref64) {
  return compare(std::vector<double>(ref.begin(), ref.end()), ref64);
}

void report(const std::string& what, const Stats& s) {
  std::printf("  %-34s cos %.10f (1-cos %.2e)  max|err| %.3e  max|err|/max|ref| %.3e\n", what.c_str(), s.cos, 1 - s.cos,
              s.max_abs, s.rel);
  ::testing::Test::RecordProperty(what + ".one_minus_cos", std::to_string(1 - s.cos));
  ::testing::Test::RecordProperty(what + ".max_rel", std::to_string(s.rel));
}

// Weight [K,C,R,S] from the checkpoint -> device [K,R,S,C] (through the library's conversion kernel).
Dev conv_weight(const std::string& prefix, DType dt, int* k, int* c, int* r) {
  std::vector<std::int64_t> shape;
  const auto w = read_weight(prefix + ".weight", &shape);
  *k = static_cast<int>(shape[0]), *c = static_cast<int>(shape[1]), *r = static_cast<int>(shape[2]);
  Dev src = upload(w, DType::kF32, {shape[0], shape[1], shape[2], shape[3]});
  Dev dst = alloc(dt, {shape[0], shape[2], shape[3], shape[1]});
  mi::vae::conv_weight_to_krsc(src.t, dst.t, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  return dst;
}

// Strict fp32 direct convolution (NHWC x, KRSC w), one thread per output; FMA accumulation in (r, s, c) order.
__global__ void conv_oracle_kernel(const float* x, const float* w, const float* bias, const float* res, float* y, int n,
                                   int h, int wd, int c, int k, int r) {
  const std::int64_t total = static_cast<std::int64_t>(n) * h * wd * k;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const int ko = static_cast<int>(i % k);
    const std::int64_t p = i / k;
    const int ox = static_cast<int>(p % wd), oy = static_cast<int>((p / wd) % h);
    const std::int64_t b = p / (static_cast<std::int64_t>(wd) * h);
    const int pad = (r - 1) / 2;
    float acc = 0.0f;
    for (int dy = 0; dy < r; ++dy) {
      const int iy = oy + dy - pad;
      if (iy < 0 || iy >= h) continue;
      for (int dx = 0; dx < r; ++dx) {
        const int ix = ox + dx - pad;
        if (ix < 0 || ix >= wd) continue;
        const float* xp = x + ((b * h + iy) * wd + ix) * c;
        const float* wp = w + ((static_cast<std::int64_t>(ko) * r + dy) * r + dx) * c;
        for (int ci = 0; ci < c; ++ci) acc = fmaf(xp[ci], wp[ci], acc);
      }
    }
    float out = acc + bias[ko];
    if (res) out = res[i] + out;
    y[i] = out;
  }
}

// ---- Convolution --------------------------------------------------------------------------------------

struct ConvCase {
  const char* name;
  const char* weight;
  int h, w;
  bool residual;
};
const ConvCase kConvCases[] = {
    {"mid_conv1", "decoder.mid_block.resnets.0.conv1", 128, 128, false},
    {"up0_conv2_res", "decoder.up_blocks.0.resnets.0.conv2", 91, 91, true},
    {"up2_conv1", "decoder.up_blocks.2.resnets.0.conv1", 96, 80, false},
    {"up3_conv1", "decoder.up_blocks.3.resnets.1.conv1", 256, 256, false},
    {"up3_shortcut", "decoder.up_blocks.3.resnets.0.conv_shortcut", 256, 256, false},
    {"conv_in", "decoder.conv_in", 91, 91, false},
    {"post_quant", "post_quant_conv", 64, 64, false},
    {"conv_out", "decoder.conv_out", 256, 256, false},
};

class VaeConv : public ::testing::TestWithParam<int> {};

TEST_P(VaeConv, MatchesTorchTf32) {
  REQUIRE_WEIGHTS();
  const int ci = GetParam();
  const ConvCase& cc = kConvCases[ci];
  int k = 0, c = 0, r = 0;
  Dev w = conv_weight(cc.weight, DType::kF32, &k, &c, &r);
  Dev bias = upload(read_weight(std::string(cc.weight) + ".bias"), DType::kF32, {k});
  Dev x = upload(synth_normal(100 + ci, static_cast<std::size_t>(cc.h) * cc.w * c), DType::kF32, {1, cc.h, cc.w, c});
  Dev res;
  if (cc.residual) res = upload(synth_normal(200 + ci, static_cast<std::size_t>(cc.h) * cc.w * k), DType::kF32, {1, cc.h, cc.w, k});
  Dev y = alloc(DType::kF32, {1, cc.h, cc.w, k});
  const std::size_t ws_bytes = mi::vae::conv2d_workspace_bytes(x.t, w.t);
  DeviceBuffer ws(std::max<std::size_t>(ws_bytes, 16));
  mi::vae::conv2d(x.t, w.t, bias.t, cc.residual ? &res.t : nullptr, y.t, ws.get(), ws_bytes, nullptr);
  const auto got = download<float>(y);

  Dev oracle = alloc(DType::kF32, {1, cc.h, cc.w, k});
  conv_oracle_kernel<<<2048, 256>>>(x.t.ptr<float>(), w.t.ptr<float>(), bias.t.ptr<float>(),
                                    cc.residual ? res.t.ptr<float>() : nullptr, oracle.t.ptr<float>(), 1, cc.h, cc.w, c, k, r);
  MI_CUDA_CHECK_LAUNCH();
  const auto ora = download<float>(oracle);

  const std::string base = std::string("conv_") + cc.name;
  const auto idx = load<std::int32_t>(base + ".idx");
  std::printf("conv %s: %dx%d %d->%d r=%d%s\n", cc.name, cc.h, cc.w, c, k, r, cc.residual ? " +residual" : "");
  const Stats tf32 = compare_sampled(got, idx, load<float>(base + ".tf32"));
  const Stats f32 = compare_sampled(got, idx, load<float>(base + ".f32"));
  std::vector<double> gd(got.begin(), got.end()), od(ora.begin(), ora.end());
  const Stats full = compare(gd, od);
  report(base + " vs torch tf32 (sampled)", tf32);
  report(base + " vs torch f32 (sampled)", f32);
  report(base + " vs f32 oracle (full)", full);
  EXPECT_GE(tf32.cos, 0.99999);
  if (cc.residual) {  // y aliasing the residual, as ResnetBlock2D's conv2 runs in the decoder
    mi::vae::conv2d(x.t, w.t, bias.t, &res.t, res.t, ws.get(), ws_bytes, nullptr);
    EXPECT_EQ(download<float>(res), got);
  }
  // Same TF32 rounding and accumulation order as cuDNN (bit-identical on most cases): far tighter than the
  // ~3e-4 TF32-vs-fp32 gap, so a change of rounding mode or tile order shows up here.
  EXPECT_LT(tf32.rel, 1e-5);
  EXPECT_GE(f32.cos, 0.99999);
  EXPECT_GE(full.cos, 0.99999);
  // Our TF32 error sits at the reference's own TF32 noise level: max error vs strict fp32 below 1e-3 of the range.
  EXPECT_LT(full.rel, 1e-3);
}

INSTANTIATE_TEST_SUITE_P(Decoder, VaeConv, ::testing::Range(0, static_cast<int>(std::size(kConvCases))),
                         [](const auto& info) { return std::string(kConvCases[info.param].name); });

struct Bf16Case {
  const char* name;
  const char* weight;
  int h, w;
};
const Bf16Case kBf16Cases[] = {
    {"mid_conv1", "decoder.mid_block.resnets.0.conv1", 64, 64},
    {"up3_conv1", "decoder.up_blocks.3.resnets.1.conv1", 128, 128},
    {"conv_out", "decoder.conv_out", 128, 128},
    {"post_quant", "post_quant_conv", 64, 64},
};

class VaeConvBf16 : public ::testing::TestWithParam<int> {};

TEST_P(VaeConvBf16, MatchesFp32MathOnBf16Inputs) {
  REQUIRE_WEIGHTS();
  const int bi = GetParam();
  const Bf16Case& bc = kBf16Cases[bi];
  int k = 0, c = 0, r = 0;
  Dev w = conv_weight(bc.weight, DType::kBF16, &k, &c, &r);
  Dev bias = upload(read_weight(std::string(bc.weight) + ".bias"), DType::kF32, {k});
  const auto xs = synth_normal(300 + bi, static_cast<std::size_t>(bc.h) * bc.w * c);
  std::vector<std::uint16_t> xb(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) xb[i] = to_bf16(xs[i]);
  Dev x = upload(xb, DType::kBF16, {1, bc.h, bc.w, c});
  Dev y = alloc(DType::kBF16, {1, bc.h, bc.w, k});
  const std::size_t ws_bytes = mi::vae::conv2d_workspace_bytes(x.t, w.t);
  DeviceBuffer ws(std::max<std::size_t>(ws_bytes, 16));
  mi::vae::conv2d(x.t, w.t, bias.t, nullptr, y.t, ws.get(), ws_bytes, nullptr);
  const auto got = download<std::uint16_t>(y);
  const std::string base = std::string("conv_bf16_") + bc.name;
  const Stats s = compare_sampled(got, load<std::int32_t>(base + ".idx"), load<float>(base + ".ref"),
                                  [](std::uint16_t v) { return static_cast<double>(from_bf16(v)); });
  report(base + " vs fp32 conv of bf16 inputs", s);
  EXPECT_GE(s.cos, 0.99999);  // only the bf16 output rounding (<= 2^-9 relative) separates them
  EXPECT_LT(s.rel, 1.0 / 256);
}

INSTANTIATE_TEST_SUITE_P(Decoder, VaeConvBf16, ::testing::Range(0, static_cast<int>(std::size(kBf16Cases))),
                         [](const auto& info) { return std::string(kBf16Cases[info.param].name); });

TEST(VaeConvApi, RejectsBadShapes) {
  Dev x = alloc(DType::kF32, {1, 8, 8, 6});  // C % 4 != 0
  Dev w = alloc(DType::kF32, {8, 3, 3, 6});
  EXPECT_THROW(mi::vae::conv2d_workspace_bytes(x.t, w.t), std::invalid_argument);
  Dev x2 = alloc(DType::kF32, {1, 8, 8, 8});
  Dev w2 = alloc(DType::kF32, {8, 5, 5, 8});  // 5x5 unsupported
  EXPECT_THROW(mi::vae::conv2d_workspace_bytes(x2.t, w2.t), std::invalid_argument);
}

// The exact tiled decoder relies on this: CUTLASS's implicit GEMM accumulates each output pixel over the same
// K = (r, s, c) sequence whatever the spatial extent of the tensor and wherever the pixel lies in it, so a conv run
// on a 2D tile (with a halo of (R-1)/2 pixels, zero padding only at the image border) reproduces the whole-tensor
// output bit for bit on the tile's interior. Checked for both dtypes, 3x3 and 1x1, wide and narrow outputs, with
// and without the fused residual, on tiles that do not divide the image and tiles of one pixel.
TEST(VaeConvTiles, InteriorBitIdenticalToWholeTensor) {
  struct Case {
    DType dt;
    int c, k, r;
    bool residual;
  };
  const Case cases[] = {{DType::kF32, 96, 96, 3, true},  {DType::kF32, 192, 96, 3, false}, {DType::kF32, 192, 96, 1, false},
                        {DType::kF32, 96, 3, 3, false},  {DType::kBF16, 96, 96, 3, true}, {DType::kBF16, 192, 96, 1, false}};
  const int h = 37, w = 45;
  for (const Case& cs : cases) {
    const std::size_t es = cs.dt == DType::kF32 ? 4 : 2;
    const auto raw = [&](std::uint32_t seed, std::size_t n) {
      const auto v = synth_normal(seed, n);
      std::vector<std::uint8_t> b(n * es);
      for (std::size_t i = 0; i < n; ++i) {
        if (cs.dt == DType::kF32) {
          std::memcpy(b.data() + i * 4, &v[i], 4);
        } else {
          const std::uint16_t u = to_bf16(v[i]);
          std::memcpy(b.data() + i * 2, &u, 2);
        }
      }
      return b;
    };
    const auto x = raw(11, static_cast<std::size_t>(h) * w * cs.c);
    const auto res = raw(12, static_cast<std::size_t>(h) * w * cs.k);
    const auto wt = raw(13, static_cast<std::size_t>(cs.k) * cs.r * cs.r * cs.c);
    auto bias = synth_normal(14, static_cast<std::size_t>(cs.k));
    Dev xd = upload(x, cs.dt, {1, h, w, cs.c}), rd = upload(res, cs.dt, {1, h, w, cs.k});
    Dev wd = upload(wt, cs.dt, {cs.k, cs.r, cs.r, cs.c}), bd = upload(bias, DType::kF32, {cs.k});
    Dev yd = alloc(cs.dt, {1, h, w, cs.k});
    DeviceBuffer ws(std::max<std::size_t>(mi::vae::conv2d_workspace_bytes(xd.t, wd.t), 16));
    mi::vae::conv2d(xd.t, wd.t, bd.t, cs.residual ? &rd.t : nullptr, yd.t, ws.get(), ws.size(), nullptr);
    const auto bytes_of = [](const Dev& d) {
      std::vector<std::uint8_t> out(static_cast<std::size_t>(d.t.nbytes()));
      MI_CUDA_CHECK(cudaDeviceSynchronize());
      d.buf.download(out.data(), out.size());
      return out;
    };
    const auto full = bytes_of(yd);

    // Sub-rectangle [r0, r1) x [c0, c1) of an NHWC host tensor with ch channels.
    const auto crop = [&](const std::vector<std::uint8_t>& src, int ch, int r0, int r1, int c0, int c1) {
      std::vector<std::uint8_t> out;
      for (int r = r0; r < r1; ++r)
        out.insert(out.end(), src.begin() + (static_cast<std::ptrdiff_t>(r) * w + c0) * ch * static_cast<std::ptrdiff_t>(es),
                   src.begin() + (static_cast<std::ptrdiff_t>(r) * w + c1) * ch * static_cast<std::ptrdiff_t>(es));
      return out;
    };
    const int halo = (cs.r - 1) / 2;
    std::size_t tiles = 0, mismatched = 0;
    for (const int th : {1, 8, 13}) {
      for (int r0 = 0; r0 < h; r0 += th) {
        for (int c0 = 0; c0 < w; c0 += 2 * th + 1) {
          const int r1 = std::min(h, r0 + th), c1 = std::min(w, c0 + 2 * th + 1);
          const int a = std::max(0, r0 - halo), b = std::min(h, r1 + halo), l = std::max(0, c0 - halo), rr = std::min(w, c1 + halo);
          Dev tx = upload(crop(x, cs.c, a, b, l, rr), cs.dt, {1, b - a, rr - l, cs.c});
          Dev tr = upload(crop(res, cs.k, a, b, l, rr), cs.dt, {1, b - a, rr - l, cs.k});
          Dev ty = alloc(cs.dt, {1, b - a, rr - l, cs.k});
          mi::vae::conv2d(tx.t, wd.t, bd.t, cs.residual ? &tr.t : nullptr, ty.t, ws.get(), ws.size(), nullptr);
          const auto got = bytes_of(ty);
          const auto tile_w = static_cast<std::size_t>(rr - l), px = static_cast<std::size_t>(cs.k) * es;
          for (int r = r0; r < r1; ++r)
            for (int c = c0; c < c1; ++c)
              mismatched += std::memcmp(got.data() + ((static_cast<std::size_t>(r - a)) * tile_w + static_cast<std::size_t>(c - l)) * px,
                                        full.data() + (static_cast<std::size_t>(r) * w + static_cast<std::size_t>(c)) * px, px) != 0;
          ++tiles;
        }
      }
    }
    std::printf("conv tiles %s %dx%d %d->%d residual=%d: %zu tiles, %zu mismatched pixels\n", cs.dt == DType::kF32 ? "f32" : "bf16",
                cs.r, cs.r, cs.c, cs.k, cs.residual, tiles, mismatched);
    EXPECT_EQ(mismatched, 0u);
  }
}

// ---- GroupNorm ----------------------------------------------------------------------------------------

struct GnCase {
  const char* name;
  const char* norm;
  int c, h, w;
  bool silu;
};
const GnCase kGnCases[] = {
    {"mid_norm1", "decoder.mid_block.resnets.0.norm1", 384, 128, 128, true},
    {"up3_norm2", "decoder.up_blocks.3.resnets.1.norm2", 96, 256, 256, false},
    {"norm_out", "decoder.conv_norm_out", 96, 200, 120, true},
};

std::vector<float> gn_input(int gi, const GnCase& g) {
  auto x = synth_normal(400 + gi, static_cast<std::size_t>(g.h) * g.w * g.c);
  for (float& v : x) v = v * 4.0f + 1.0f;  // 4 v is exact: contraction cannot change the result
  return x;
}

class VaeGroupNorm : public ::testing::TestWithParam<int> {};

TEST_P(VaeGroupNorm, MatchesTorchAndTilesAgree) {
  REQUIRE_WEIGHTS();
  const int gi = GetParam();
  const GnCase& g = kGnCases[gi];
  const auto xs = gn_input(gi, g);
  Dev x = upload(xs, DType::kF32, {1, g.h, g.w, g.c});
  Dev gamma = upload(read_weight(std::string(g.norm) + ".weight"), DType::kF32, {g.c});
  Dev beta = upload(read_weight(std::string(g.norm) + ".bias"), DType::kF32, {g.c});
  Dev y = alloc(DType::kF32, {1, g.h, g.w, g.c});
  const std::size_t ws_bytes = mi::vae::group_norm_workspace_bytes(x.t, 32);
  DeviceBuffer ws(ws_bytes);
  mi::vae::group_norm(x.t, 32, mi::vae::kResnetEps, gamma.t, beta.t, g.silu, y.t, ws.get(), ws_bytes, nullptr);
  const auto got = download<float>(y);

  const std::string base = std::string("gn_") + g.name;
  std::printf("group_norm %s: %dx%d C=%d silu=%d\n", g.name, g.h, g.w, g.c, g.silu);
  const Stats s = compare_sampled(got, load<std::int32_t>(base + ".idx"), load<float>(base + ".ref"));
  report(base + " vs torch", s);
  EXPECT_GE(s.cos, 0.99999);
  EXPECT_LT(s.rel, 1e-5);

  // Statistics vs aten.native_group_norm.
  Dev stats = alloc(DType::kF32, {1, 32, 2});
  mi::vae::group_norm_stats(x.t, 32, mi::vae::kResnetEps, stats.t, ws.get(), ws_bytes, nullptr);
  const auto st = download<float>(stats);
  const auto tmean = load<float>(base + ".mean"), trstd = load<float>(base + ".rstd");
  double mean_err = 0, rstd_err = 0;
  for (int i = 0; i < 32; ++i) {
    const double dm = std::fabs(static_cast<double>(st[2 * i]) - tmean[i]);
    const double dr = std::fabs(static_cast<double>(st[2 * i + 1]) - trstd[i]);
    mean_err = std::max(mean_err, dm / std::max(1e-3, std::fabs(static_cast<double>(tmean[i]))));
    rstd_err = std::max(rstd_err, dr / trstd[i]);
  }
  std::printf("  stats vs torch: max rel mean err %.3e, max rel rstd err %.3e\n", mean_err, rstd_err);
  EXPECT_LT(mean_err, 1e-5);
  EXPECT_LT(rstd_err, 1e-5);

  // Tiled statistics: the tensor cut into uneven 2D blocks (separate tensors, accumulated in a scrambled order,
  // each pixel once) must give bit-identical statistics, and applying them per block must reproduce the one-pass
  // output bit for bit.
  const int rcuts[] = {0, g.h / 3 + 1, g.h / 2 + 3, g.h};
  const int ccuts[] = {0, 5, g.w / 2 + 1, g.w};
  struct Block {
    int r0, r1, c0, c1;
  };
  std::vector<Block> blocks;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) blocks.push_back({rcuts[r], rcuts[r + 1], ccuts[c], ccuts[c + 1]});
  const auto block_data = [&](const Block& bl, const std::vector<float>& src) {
    std::vector<float> part;
    for (int r = bl.r0; r < bl.r1; ++r)
      part.insert(part.end(), src.begin() + (static_cast<std::ptrdiff_t>(r) * g.w + bl.c0) * g.c,
                  src.begin() + (static_cast<std::ptrdiff_t>(r) * g.w + bl.c1) * g.c);
    return part;
  };
  DeviceBuffer accum(mi::vae::group_norm_accum_bytes(1, 32));
  mi::vae::group_norm_accum_zero(accum.get(), 1, 32, nullptr);
  std::vector<Dev> parts;
  for (const Block& bl : blocks) parts.push_back(upload(block_data(bl, xs), DType::kF32, {1, bl.r1 - bl.r0, bl.c1 - bl.c0, g.c}));
  for (const int i : {4, 0, 8, 2, 6, 1, 7, 3, 5}) mi::vae::group_norm_accumulate(parts[i].t, 32, accum.get(), nullptr);
  Dev tstats = alloc(DType::kF32, {1, 32, 2});
  mi::vae::group_norm_finalize(accum.get(), 1, 32, static_cast<std::int64_t>(g.h) * g.w * (g.c / 32), mi::vae::kResnetEps,
                               tstats.t, nullptr);
  const auto tst = download<float>(tstats);
  EXPECT_EQ(0, std::memcmp(tst.data(), st.data(), 64 * sizeof(float))) << "tiled statistics differ from one-pass";
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const Block& bl = blocks[i];
    Dev ys = alloc(DType::kF32, {1, bl.r1 - bl.r0, bl.c1 - bl.c0, g.c});
    mi::vae::group_norm_apply(parts[i].t, tstats.t, gamma.t, beta.t, g.silu, ys.t, nullptr);
    EXPECT_EQ(block_data(bl, got), download<float>(ys)) << "block " << i;
  }
}

// Out-of-range inputs (|x| >= 2^33, inf, NaN) make the statistics NaN instead of silently overflowing; the
// largest in-range magnitudes and tiny values are summed exactly.
TEST(VaeGroupNormExact, RangeAndExactness) {
  const int h = 4, w = 8, c = 32;  // one channel per group
  std::vector<float> xs(static_cast<std::size_t>(h) * w * c, 0.0f);
  for (int p = 0; p < h * w; ++p) {
    xs[static_cast<std::size_t>(p) * c + 0] = p % 2 ? 0x1.fffffep32f : -0x1.fffffep32f;  // cancels exactly: mean 0
    xs[static_cast<std::size_t>(p) * c + 1] = 0x1p-60f;                                   // tiny, exact on the grid
    xs[static_cast<std::size_t>(p) * c + 2] = 1.0f + static_cast<float>(p);
  }
  xs[3] = 0x1p33f;          // out of range
  xs[c + 4] = INFINITY;
  xs[2 * c + 5] = NAN;
  Dev x = upload(xs, DType::kF32, {1, h, w, c});
  Dev stats = alloc(DType::kF32, {1, 32, 2});
  const std::size_t ws_bytes = mi::vae::group_norm_workspace_bytes(x.t, 32);
  DeviceBuffer ws(ws_bytes);
  mi::vae::group_norm_stats(x.t, 32, 0.0f, stats.t, ws.get(), ws_bytes, nullptr);
  const auto st = download<float>(stats);
  EXPECT_EQ(st[0], 0.0f);
  EXPECT_EQ(st[1], static_cast<float>(1.0 / (0x1.fffffep32 * 1.0)));  // var = x^2 exactly
  EXPECT_EQ(st[2], 0x1p-60f);
  EXPECT_TRUE(std::isinf(st[3]));  // variance exactly 0, eps 0
  EXPECT_EQ(st[4], 16.5f);         // mean of 1..32
  EXPECT_EQ(st[5], static_cast<float>(1.0 / std::sqrt((32.0 * 32.0 - 1.0) / 12.0)));
  for (int g = 3; g <= 5; ++g) EXPECT_TRUE(std::isnan(st[2 * g]) && std::isnan(st[2 * g + 1])) << "group " << g;
  for (int g = 6; g < 32; ++g) EXPECT_EQ(st[2 * g], 0.0f) << "group " << g;
}

INSTANTIATE_TEST_SUITE_P(Decoder, VaeGroupNorm, ::testing::Range(0, static_cast<int>(std::size(kGnCases))),
                         [](const auto& info) { return std::string(kGnCases[info.param].name); });

TEST(VaeGroupNormBf16, EqualsRoundedFp32Path) {
  const int h = 33, w = 17, c = 192;
  const auto xs = synth_normal(900, static_cast<std::size_t>(h) * w * c);
  std::vector<std::uint16_t> xb(xs.size());
  std::vector<float> xr(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) xb[i] = to_bf16(xs[i]), xr[i] = from_bf16(xb[i]);
  std::vector<float> gam(c), bet(c);
  for (int i = 0; i < c; ++i) gam[i] = 0.5f + 0.01f * static_cast<float>(i), bet[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;
  Dev x16 = upload(xb, DType::kBF16, {1, h, w, c});
  Dev x32 = upload(xr, DType::kF32, {1, h, w, c});
  Dev gamma = upload(gam, DType::kF32, {c}), beta = upload(bet, DType::kF32, {c});
  Dev y16 = alloc(DType::kBF16, {1, h, w, c}), y32 = alloc(DType::kF32, {1, h, w, c});
  const std::size_t ws_bytes = mi::vae::group_norm_workspace_bytes(x32.t, 32);
  DeviceBuffer ws(ws_bytes);
  mi::vae::group_norm(x16.t, 32, 1e-6f, gamma.t, beta.t, false, y16.t, ws.get(), ws_bytes, nullptr);
  mi::vae::group_norm(x32.t, 32, 1e-6f, gamma.t, beta.t, false, y32.t, ws.get(), ws_bytes, nullptr);
  const auto a = download<std::uint16_t>(y16);
  const auto b = download<float>(y32);
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < a.size(); ++i) mismatches += a[i] != to_bf16(b[i]);
  EXPECT_EQ(mismatches, 0u);
}

// ---- Attention ----------------------------------------------------------------------------------------

struct AttnInputs {
  std::vector<float> q, k, v;
};

AttnInputs attn_core_inputs(int ai, int h, int w) {
  const int d = 384, L = h * w, bw = (w + 7) / 8;
  const auto m = synth_normal(500 + 10 * ai, static_cast<std::size_t>((h + 7) / 8) * bw * d);
  AttnInputs in{synth_normal(501 + 10 * ai, static_cast<std::size_t>(L) * d), synth_normal(502 + 10 * ai, static_cast<std::size_t>(L) * d),
                synth_normal(503 + 10 * ai, static_cast<std::size_t>(L) * d)};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t l = static_cast<std::size_t>(y) * w + x, blk = static_cast<std::size_t>(y / 8) * bw + x / 8;
      for (int j = 0; j < d; ++j) {
        in.q[l * d + j] += m[blk * d + j];
        in.k[l * d + j] += m[blk * d + j];
      }
    }
  return in;
}

struct AttnCoreCase {
  int h, w;
  bool packed;  // q/k/v as column slices of one [L, 1152] buffer
};
const AttnCoreCase kAttnCases[] = {{182, 182, true}, {10, 10, false}, {25, 40, true}};

class VaeAttentionCore : public ::testing::TestWithParam<int> {};

TEST_P(VaeAttentionCore, MatchesSdpa) {
  const int ai = GetParam();
  const auto& ac = kAttnCases[ai];
  const int d = 384, L = ac.h * ac.w;
  const AttnInputs in = attn_core_inputs(ai, ac.h, ac.w);
  std::vector<Dev> bufs;
  Tensor q, k, v;
  if (ac.packed) {
    std::vector<float> packed(static_cast<std::size_t>(L) * 3 * d);
    for (int l = 0; l < L; ++l)
      for (int j = 0; j < d; ++j) {
        packed[(static_cast<std::size_t>(l) * 3 + 0) * d + j] = in.q[static_cast<std::size_t>(l) * d + j];
        packed[(static_cast<std::size_t>(l) * 3 + 1) * d + j] = in.k[static_cast<std::size_t>(l) * d + j];
        packed[(static_cast<std::size_t>(l) * 3 + 2) * d + j] = in.v[static_cast<std::size_t>(l) * d + j];
      }
    bufs.push_back(upload(packed, DType::kF32, {1, L, 3 * d}));
    for (int p = 0; p < 3; ++p) {
      Tensor t = Tensor::view(bufs[0].t.ptr<float>() + p * d, DType::kF32, Device::kCUDA, {1, L, d});
      t.stride[1] = 3 * d;
      t.stride[0] = static_cast<std::int64_t>(L) * 3 * d;
      (p == 0 ? q : p == 1 ? k : v) = t;
    }
  } else {
    bufs.push_back(upload(in.q, DType::kF32, {1, L, d}));
    bufs.push_back(upload(in.k, DType::kF32, {1, L, d}));
    bufs.push_back(upload(in.v, DType::kF32, {1, L, d}));
    q = bufs[0].t, k = bufs[1].t, v = bufs[2].t;
  }
  Dev o = alloc(DType::kF32, {1, L, d});
  const std::string base = "attn_core_" + std::to_string(L);
  const auto idx = load<std::int32_t>(base + ".idx");
  const auto ref = load<float>(base + ".ref");
  const auto ref64 = load<double>(base + ".ref64");
  std::printf("attention core L=%d%s\n", L, ac.packed ? " (packed qkv)" : "");
  report(base + " torch sdpa vs f64", ref_error(ref, ref64));
  for (auto prec : {mi::vae::AttnPrecision::kTf32x3, mi::vae::AttnPrecision::kTf32}) {
    MI_CUDA_CHECK(cudaMemset(o.buf.get(), 0xFF, o.t.nbytes()));  // NaN poison: every row must be written
    mi::vae::attention_fp32(q, k, v, o.t, 1.0f / std::sqrt(384.0f), prec, nullptr);
    const auto got = download<float>(o);
    std::size_t nonfinite = 0;
    for (float f : got) nonfinite += !std::isfinite(f);
    EXPECT_EQ(nonfinite, 0u);
    const bool x3 = prec == mi::vae::AttnPrecision::kTf32x3;
    const Stats s = compare_sampled(got, idx, ref);
    const Stats t = compare_sampled(got, idx, ref64);
    report(base + (x3 ? " tf32x3 vs sdpa" : " tf32 vs sdpa"), s);
    report(base + (x3 ? " tf32x3 vs f64" : " tf32 vs f64"), t);
    EXPECT_GE(s.cos, 0.99999);
    EXPECT_GE(t.cos, 0.99999);
    if (x3) EXPECT_LT(t.rel, 1e-5);  // fp32-class accuracy (torch's fp32 SDPA: 1.5e-6 to 8e-6 here)
  }
}

INSTANTIATE_TEST_SUITE_P(Shapes, VaeAttentionCore, ::testing::Range(0, static_cast<int>(std::size(kAttnCases))),
                         [](const auto& info) { return "L" + std::to_string(kAttnCases[info.param].h * kAttnCases[info.param].w); });

TEST(VaeAttentionCoreBatch, BatchesAreIndependent) {
  const int d = 384, L = 1000;
  const AttnInputs a = attn_core_inputs(2, 25, 40), b = attn_core_inputs(7, 25, 40);
  auto cat = [](const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<float> out(x);
    out.insert(out.end(), y.begin(), y.end());
    return out;
  };
  Dev q = upload(cat(a.q, b.q), DType::kF32, {2, L, d}), k = upload(cat(a.k, b.k), DType::kF32, {2, L, d});
  Dev v = upload(cat(a.v, b.v), DType::kF32, {2, L, d});
  Dev o = alloc(DType::kF32, {2, L, d});
  mi::vae::attention_fp32(q.t, k.t, v.t, o.t, 1.0f / std::sqrt(384.0f), mi::vae::AttnPrecision::kTf32x3, nullptr);
  const auto both = download<float>(o);
  for (int bi = 0; bi < 2; ++bi) {
    const AttnInputs& in = bi == 0 ? a : b;
    Dev q1 = upload(in.q, DType::kF32, {1, L, d}), k1 = upload(in.k, DType::kF32, {1, L, d}), v1 = upload(in.v, DType::kF32, {1, L, d});
    Dev o1 = alloc(DType::kF32, {1, L, d});
    mi::vae::attention_fp32(q1.t, k1.t, v1.t, o1.t, 1.0f / std::sqrt(384.0f), mi::vae::AttnPrecision::kTf32x3, nullptr);
    const auto one = download<float>(o1);
    EXPECT_EQ(0, std::memcmp(one.data(), both.data() + static_cast<std::size_t>(bi) * L * d, one.size() * sizeof(float)))
        << "batch " << bi;
  }
}

struct MidWeights {
  std::vector<Dev> keep;
  mi::vae::MidAttentionWeights w;
};

MidWeights load_mid_weights() {
  MidWeights m;
  const std::string p = "decoder.mid_block.attentions.0.";
  auto get = [&](const std::string& name, std::initializer_list<std::int64_t> shape) {
    m.keep.push_back(upload(read_weight(p + name), DType::kF32, shape));
    return m.keep.back().t;
  };
  m.keep.reserve(10);
  m.w.gn_gamma = get("group_norm.weight", {384});
  m.w.gn_beta = get("group_norm.bias", {384});
  m.w.wq = get("to_q.weight", {384, 384});
  m.w.bq = get("to_q.bias", {384});
  m.w.wk = get("to_k.weight", {384, 384});
  m.w.bk = get("to_k.bias", {384});
  m.w.wv = get("to_v.weight", {384, 384});
  m.w.bv = get("to_v.bias", {384});
  m.w.wo = get("to_out.0.weight", {384, 384});
  m.w.bo = get("to_out.0.bias", {384});
  return m;
}

struct LtHandle {
  cublasLtHandle_t h = nullptr;
  LtHandle() {
    if (cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasLtCreate failed");
  }
  ~LtHandle() { cublasLtDestroy(h); }
};

TEST(VaeMidAttention, MatchesDiffusersBlock) {
  REQUIRE_WEIGHTS();
  const int h = 128, w = 128, c = 384;
  auto xs = synth_normal(600, static_cast<std::size_t>(h) * w * c);
  const auto coarse = synth_normal(601, static_cast<std::size_t>(h / 8) * (w / 8) * c);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int j = 0; j < c; ++j)
        xs[(static_cast<std::size_t>(y) * w + x) * c + j] += coarse[(static_cast<std::size_t>(y / 8) * (w / 8) + x / 8) * c + j] * 2.0f;
  MidWeights mw = load_mid_weights();
  Dev x = upload(xs, DType::kF32, {1, h, w, c});
  Dev y = alloc(DType::kF32, {1, h, w, c});
  const std::size_t ws_bytes = mi::vae::mid_attention_workspace_bytes(x.t);
  DeviceBuffer ws(ws_bytes);
  LtHandle lt;
  const auto idx = load<std::int32_t>("attn_block.idx");
  const auto ref = load<float>("attn_block.ref"), ref_nores = load<float>("attn_block_nores.ref");
  const auto ref64 = load<double>("attn_block.ref64"), ref64_nores = load<double>("attn_block_nores.ref64");
  std::printf("mid attention block %dx%d (L=%d)\n", h, w, h * w);
  report("attn branch torch vs f64", ref_error(ref_nores, ref64_nores));
  report("attn block  torch vs f64", ref_error(ref, ref64));
  struct Mode {
    mi::vae::AttnPrecision p;
    bool fast;
    const char* name;
  };
  for (const Mode& m : {Mode{mi::vae::AttnPrecision::kTf32x3, false, "tf32x3+fp32 linears"},
                        Mode{mi::vae::AttnPrecision::kTf32, false, "tf32+fp32 linears"},
                        Mode{mi::vae::AttnPrecision::kTf32, true, "tf32+tf32 linears"}}) {
    mi::vae::mid_attention(lt.h, x.t, mw.w, nullptr, y.t, ws.get(), ws_bytes, m.p, m.fast, nullptr);
    const auto yb = download<float>(y);
    const Stats branch = compare_sampled(yb, idx, ref_nores), branch64 = compare_sampled(yb, idx, ref64_nores);
    mi::vae::mid_attention(lt.h, x.t, mw.w, &x.t, y.t, ws.get(), ws_bytes, m.p, m.fast, nullptr);
    const auto yf = download<float>(y);
    const Stats full = compare_sampled(yf, idx, ref), full64 = compare_sampled(yf, idx, ref64);
    report(std::string("attn branch ") + m.name + " vs torch", branch);
    report(std::string("attn branch ") + m.name + " vs f64", branch64);
    report(std::string("attn block  ") + m.name + " vs torch", full);
    report(std::string("attn block  ") + m.name + " vs f64", full64);
    EXPECT_GE(branch.cos, 0.99999);
    EXPECT_GE(full.cos, 0.99999);
    EXPECT_GE(branch64.cos, 0.99999);
    if (m.p == mi::vae::AttnPrecision::kTf32x3 && !m.fast) EXPECT_LT(branch64.rel, 1e-5);
  }
  // In place (y aliases x and the residual), as the decoder runs it.
  mi::vae::mid_attention(lt.h, x.t, mw.w, &x.t, y.t, ws.get(), ws_bytes, mi::vae::AttnPrecision::kTf32x3, false, nullptr);
  const auto out_of_place = download<float>(y);
  mi::vae::mid_attention(lt.h, x.t, mw.w, &x.t, x.t, ws.get(), ws_bytes, mi::vae::AttnPrecision::kTf32x3, false, nullptr);
  EXPECT_EQ(download<float>(x), out_of_place);
}

TEST(VaeMidAttention, BatchOfTwoMatchesSingles) {
  REQUIRE_WEIGHTS();
  const int h = 16, w = 24, c = 384;
  const auto x0 = synth_normal(910, static_cast<std::size_t>(h) * w * c), x1 = synth_normal(911, x0.size());
  std::vector<float> both(x0);
  both.insert(both.end(), x1.begin(), x1.end());
  MidWeights mw = load_mid_weights();
  LtHandle lt;
  Dev xb = upload(both, DType::kF32, {2, h, w, c});
  Dev yb = alloc(DType::kF32, {2, h, w, c});
  const std::size_t ws_bytes = mi::vae::mid_attention_workspace_bytes(xb.t);
  DeviceBuffer ws(ws_bytes);
  mi::vae::mid_attention(lt.h, xb.t, mw.w, &xb.t, yb.t, ws.get(), ws_bytes, mi::vae::AttnPrecision::kTf32x3, false, nullptr);
  const auto got = download<float>(yb);
  for (int bi = 0; bi < 2; ++bi) {
    Dev x1t = upload(bi == 0 ? x0 : x1, DType::kF32, {1, h, w, c});
    Dev y1 = alloc(DType::kF32, {1, h, w, c});
    mi::vae::mid_attention(lt.h, x1t.t, mw.w, &x1t.t, y1.t, ws.get(), ws_bytes, mi::vae::AttnPrecision::kTf32x3, false, nullptr);
    const auto one = download<float>(y1);
    std::vector<double> a(one.begin(), one.end());
    std::vector<double> b(got.begin() + static_cast<std::ptrdiff_t>(bi) * h * w * c,
                          got.begin() + static_cast<std::ptrdiff_t>(bi + 1) * h * w * c);
    const Stats s = compare(b, a);  // cuBLASLt may pick another algorithm for M = 2L, so not bitwise
    EXPECT_GE(s.cos, 0.9999999) << "batch " << bi;
    EXPECT_LT(s.rel, 1e-5) << "batch " << bi;
  }
}

// ---- Exact ops ----------------------------------------------------------------------------------------

TEST(VaeElementwise, UpsampleMatchesTorchExactly) {
  const auto xs = load<float>("upsample.x");
  const auto ys = load<float>("upsample.y");
  Dev x = upload(xs, DType::kF32, {1, 5, 7, 16});
  Dev y = alloc(DType::kF32, {1, 10, 14, 16});
  mi::vae::upsample_nearest2x(x.t, y.t, nullptr);
  EXPECT_EQ(download<float>(y), ys);

  // Realistic bf16 shape against the index rule the torch vector pins down.
  const int h = 91, w = 91, c = 192;
  const auto big = synth_normal(901, static_cast<std::size_t>(h) * w * c);
  std::vector<std::uint16_t> bb(big.size());
  for (std::size_t i = 0; i < big.size(); ++i) bb[i] = to_bf16(big[i]);
  Dev xb = upload(bb, DType::kBF16, {1, h, w, c});
  Dev yb = alloc(DType::kBF16, {1, 2 * h, 2 * w, c});
  mi::vae::upsample_nearest2x(xb.t, yb.t, nullptr);
  const auto got = download<std::uint16_t>(yb);
  std::size_t bad = 0;
  for (int oy = 0; oy < 2 * h; ++oy)
    for (int ox = 0; ox < 2 * w; ++ox)
      for (int j = 0; j < c; ++j)
        bad += got[(static_cast<std::size_t>(oy) * 2 * w + ox) * c + j] != bb[(static_cast<std::size_t>(oy / 2) * w + ox / 2) * c + j];
  EXPECT_EQ(bad, 0u);
}

TEST(VaeElementwise, BnDenormUnpackMatchesPipelineExactly) {
  const auto z = load<float>("bn_unpack.z");
  Dev zt = upload(z, DType::kF32, {1, 128, 6, 10});
  Dev mean = upload(load<float>("bn_unpack.mean"), DType::kF32, {128});
  Dev var = upload(load<float>("bn_unpack.var"), DType::kF32, {128});
  Dev y = alloc(DType::kF32, {1, 12, 20, 32});
  mi::vae::bn_denorm_unpack(zt.t, mean.t, var.t, mi::vae::kBatchNormEps, y.t, nullptr);
  const auto got = download<float>(y);
  const auto ref = load<float>("bn_unpack.y");
  ASSERT_EQ(got.size(), ref.size());
  EXPECT_EQ(0, std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)));
}

TEST(VaeElementwise, Rgb8MatchesVaeImageProcessorExactly) {
  const auto xs = load<float>("rgb8.x");
  const auto ref = load<std::uint8_t>("rgb8.y");
  Dev x = upload(xs, DType::kF32, {1, 32, 48, 3});
  Dev y = alloc(DType::kU8, {1, 32, 48, 3});
  mi::vae::postprocess_to_rgb8(x.t, y.t, nullptr);
  EXPECT_EQ(download<std::uint8_t>(y), ref);
}

TEST(VaeElementwise, LayoutRoundTripAndResidualAdd) {
  const int n = 2, c = 96, h = 37, w = 45;
  const auto xs = synth_normal(902, static_cast<std::size_t>(n) * c * h * w);
  Dev x = upload(xs, DType::kF32, {n, c, h, w});
  Dev y = alloc(DType::kF32, {n, h, w, c});
  Dev back = alloc(DType::kF32, {n, c, h, w});
  mi::vae::nchw_to_nhwc(x.t, y.t, nullptr);
  mi::vae::nhwc_to_nchw(y.t, back.t, nullptr);
  const auto nhwc = download<float>(y);
  std::size_t bad = 0;
  for (int b = 0; b < n; ++b)
    for (int ch = 0; ch < c; ++ch)
      for (int p = 0; p < h * w; ++p)
        bad += nhwc[(static_cast<std::size_t>(b) * h * w + p) * c + ch] != xs[(static_cast<std::size_t>(b) * c + ch) * h * w + p];
  EXPECT_EQ(bad, 0u);
  EXPECT_EQ(download<float>(back), xs);

  const auto bs = synth_normal(903, xs.size());
  Dev bt = upload(bs, DType::kF32, {n, c, h, w});
  Dev sum = alloc(DType::kF32, {n, c, h, w});
  mi::vae::residual_add(x.t, bt.t, sum.t, nullptr);
  const auto got = download<float>(sum);
  bad = 0;
  for (std::size_t i = 0; i < xs.size(); ++i) bad += got[i] != xs[i] + bs[i];
  EXPECT_EQ(bad, 0u);

  std::vector<std::uint16_t> a16(xs.size()), b16(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) a16[i] = to_bf16(xs[i]), b16[i] = to_bf16(bs[i]);
  Dev a16t = upload(a16, DType::kBF16, {n, c, h, w}), b16t = upload(b16, DType::kBF16, {n, c, h, w});
  Dev s16 = alloc(DType::kBF16, {n, c, h, w});
  mi::vae::residual_add(a16t.t, b16t.t, s16.t, nullptr);
  const auto g16 = download<std::uint16_t>(s16);
  bad = 0;
  for (std::size_t i = 0; i < xs.size(); ++i) bad += g16[i] != to_bf16(from_bf16(a16[i]) + from_bf16(b16[i]));
  EXPECT_EQ(bad, 0u);
}

// ---- Micro benchmark ----------------------------------------------------------------------------------

double time_ms(const std::function<void()>& fn, int iters, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) fn();
  cudaEvent_t a, b;
  MI_CUDA_CHECK(cudaEventCreate(&a));
  MI_CUDA_CHECK(cudaEventCreate(&b));
  MI_CUDA_CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  MI_CUDA_CHECK(cudaEventRecord(b));
  MI_CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  MI_CUDA_CHECK(cudaEventDestroy(a));
  MI_CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

TEST(VaeKernelsBench, ConvAttentionGroupNorm) {
  REQUIRE_WEIGHTS();
  struct B {
    const char* name;
    const char* weight;
    int hw;
    DType dt;
  };
  for (const B& bc : {B{"conv3x3 384->384 @256^2 tf32", "decoder.mid_block.resnets.0.conv1", 256, DType::kF32},
                      B{"conv3x3 96->96 @1024^2 tf32", "decoder.up_blocks.3.resnets.1.conv1", 1024, DType::kF32},
                      B{"conv3x3 384->384 @256^2 bf16", "decoder.mid_block.resnets.0.conv1", 256, DType::kBF16},
                      B{"conv3x3 96->96 @1024^2 bf16", "decoder.up_blocks.3.resnets.1.conv1", 1024, DType::kBF16}}) {
    int k = 0, c = 0, r = 0;
    Dev w = conv_weight(bc.weight, bc.dt, &k, &c, &r);
    Dev bias = upload(read_weight(std::string(bc.weight) + ".bias"), DType::kF32, {k});
    Dev x = alloc(bc.dt, {1, bc.hw, bc.hw, c});
    MI_CUDA_CHECK(cudaMemset(x.buf.get(), 0, x.t.nbytes()));
    Dev y = alloc(bc.dt, {1, bc.hw, bc.hw, k});
    const std::size_t ws_bytes = mi::vae::conv2d_workspace_bytes(x.t, w.t);
    DeviceBuffer ws(std::max<std::size_t>(ws_bytes, 16));
    const double ms = time_ms([&] { mi::vae::conv2d(x.t, w.t, bias.t, nullptr, y.t, ws.get(), ws_bytes, nullptr); }, 20);
    const double tflops = 2.0 * bc.hw * bc.hw * k * c * r * r / ms / 1e9;
    std::printf("  %-32s %8.3f ms  %6.1f TFLOPS\n", bc.name, ms, tflops);
    RecordProperty(std::string(bc.name) + ".tflops", std::to_string(tflops));
  }
  for (int hw : {128, 182, 256}) {
    const int L = hw * hw;
    Dev q = alloc(DType::kF32, {1, L, 384}), k = alloc(DType::kF32, {1, L, 384}), v = alloc(DType::kF32, {1, L, 384});
    Dev o = alloc(DType::kF32, {1, L, 384});
    for (Dev* d : {&q, &k, &v}) {
      const auto vals = synth_normal(904, static_cast<std::size_t>(L) * 384);
      d->buf.upload(vals.data(), vals.size() * sizeof(float));
    }
    for (auto prec : {mi::vae::AttnPrecision::kTf32x3, mi::vae::AttnPrecision::kTf32}) {
      const double ms = time_ms([&] { mi::vae::attention_fp32(q.t, k.t, v.t, o.t, 0.051031f, prec, nullptr); }, 2, 1);
      const double tflops = 4.0 * L * static_cast<double>(L) * 384 / ms / 1e9;
      const std::string name = std::string("attention L=") + std::to_string(L) +
                               (prec == mi::vae::AttnPrecision::kTf32x3 ? " tf32x3" : " tf32");
      std::printf("  %-32s %8.3f ms  %6.1f TFLOPS (useful)\n", name.c_str(), ms, tflops);
      RecordProperty(name + ".ms", std::to_string(ms));
    }
  }
  {
    Dev x = alloc(DType::kF32, {1, 1024, 1024, 96});
    MI_CUDA_CHECK(cudaMemset(x.buf.get(), 0, x.t.nbytes()));
    std::vector<float> ones(96, 1.0f);
    Dev g = upload(ones, DType::kF32, {96}), b = upload(ones, DType::kF32, {96});
    const std::size_t ws_bytes = mi::vae::group_norm_workspace_bytes(x.t, 32);
    DeviceBuffer ws(ws_bytes);
    const double ms = time_ms([&] { mi::vae::group_norm(x.t, 32, 1e-6f, g.t, b.t, true, x.t, ws.get(), ws_bytes, nullptr); }, 20);
    std::printf("  %-32s %8.3f ms  %6.1f GB/s\n", "group_norm+silu 96 @1024^2 f32", ms, 3.0 * x.t.nbytes() / ms / 1e6);
    RecordProperty("group_norm_1024.ms", std::to_string(ms));
  }
}

}  // namespace
