// Checks the Vulkan VAE decoder: each kernel against a host replay of its op order on small shapes with tails, then the
// real decoder on a final latent against the reference pipeline's PNG (PSNR, max |diff|, identical bytes), with timing.
// Errors are scaled: |got - replay| / (|replay| + floor), where a convolution's floor is the sum of its |products| (the
// size an fp32 sum can be wrong by, whatever its order) and an elementwise op's is a fraction of the replay's rms.
// Exit status: 0 everything passed, 1 a check failed, 2 bad usage.
// Usage: vulkan_vae_check [--device N] [--path coop|plain] [--kat-only | --no-kat | --bench] [--vae FILE] [--latent NPY]
//                         [--golden PNG] [--out PNG] [--crop H,W] [--iters N] [--max-macs N] [--profile] [--calibrate]
#include <zlib.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "block.h"
#include "runtime.h"
#include "vae.h"

namespace {

using namespace mi::vk;

// ---- Files ------------------------------------------------------------------------------------------------------------

std::vector<char> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::vector<char> data(static_cast<std::size_t>(in.tellg()));
  in.seekg(0);
  in.read(data.data(), static_cast<std::streamsize>(data.size()));
  return data;
}

// A little-endian fp32 C-order .npy (versions 1-3).
std::vector<float> read_npy_f32(const std::string& path, std::vector<std::int64_t>& shape) {
  const std::vector<char> d = read_file(path);
  if (d.size() < 10 || std::memcmp(d.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not a .npy file");
  const int major = static_cast<unsigned char>(d[6]);
  std::size_t header_len = 0, at = 0;
  if (major == 1) {
    header_len = static_cast<unsigned char>(d[8]) | static_cast<std::size_t>(static_cast<unsigned char>(d[9])) << 8;
    at = 10;
  } else {
    if (d.size() < 12) throw std::runtime_error(path + ": truncated");
    for (int i = 3; i >= 0; --i) header_len = header_len << 8 | static_cast<unsigned char>(d[8 + static_cast<std::size_t>(i)]);
    at = 12;
  }
  if (at + header_len > d.size()) throw std::runtime_error(path + ": truncated header");
  const std::string h(d.data() + at, header_len);
  if (h.find("'descr': '<f4'") == std::string::npos || h.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error(path + ": expected little-endian fp32 in C order, header " + h);
  }
  const std::size_t open = h.find('(', h.find("'shape'")), close = h.find(')', open);
  std::stringstream ss(h.substr(open + 1, close - open - 1));
  std::string item;
  std::size_t count = 1;
  shape.clear();
  while (std::getline(ss, item, ',')) {
    if (item.find_first_not_of(" ") == std::string::npos) continue;
    shape.push_back(std::stoll(item));
    count *= static_cast<std::size_t>(shape.back());
  }
  at += header_len;
  if (d.size() - at != count * 4) throw std::runtime_error(path + ": data size does not match the shape");
  std::vector<float> out(count);
  std::memcpy(out.data(), d.data() + at, count * 4);
  return out;
}

std::uint32_t be32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) << 24 | static_cast<std::uint32_t>(p[1]) << 16 |
         static_cast<std::uint32_t>(p[2]) << 8 | p[3];
}

// 8-bit RGB or RGBA, non-interlaced PNG -> RGB.
std::vector<std::uint8_t> read_png_rgb(const std::string& path, int& width, int& height) {
  const std::vector<char> raw = read_file(path);
  const auto* d = reinterpret_cast<const unsigned char*>(raw.data());
  if (raw.size() < 8 || std::memcmp(d, "\x89PNG\r\n\x1a\n", 8) != 0) throw std::runtime_error(path + ": not a PNG");
  std::size_t at = 8;
  int channels = 0;
  std::vector<unsigned char> idat;
  while (at + 12 <= raw.size()) {
    const std::uint32_t len = be32(d + at);
    const std::string type(reinterpret_cast<const char*>(d + at + 4), 4);
    if (at + 12 + len > raw.size()) throw std::runtime_error(path + ": truncated chunk");
    const unsigned char* body = d + at + 8;
    if (type == "IHDR") {
      width = static_cast<int>(be32(body));
      height = static_cast<int>(be32(body + 4));
      if (body[8] != 8 || (body[9] != 2 && body[9] != 6) || body[12] != 0) {
        throw std::runtime_error(path + ": only 8-bit RGB/RGBA non-interlaced PNGs are read");
      }
      channels = body[9] == 2 ? 3 : 4;
    } else if (type == "IDAT") {
      idat.insert(idat.end(), body, body + len);
    } else if (type == "IEND") {
      break;
    }
    at += 12 + len;
  }
  if (channels == 0) throw std::runtime_error(path + ": no IHDR");
  const std::size_t stride = static_cast<std::size_t>(width) * channels;
  std::vector<unsigned char> filtered((stride + 1) * static_cast<std::size_t>(height));
  uLongf out_len = static_cast<uLongf>(filtered.size());
  if (uncompress(filtered.data(), &out_len, idat.data(), static_cast<uLong>(idat.size())) != Z_OK || out_len != filtered.size()) {
    throw std::runtime_error(path + ": cannot inflate the image data");
  }
  std::vector<unsigned char> img(stride * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    const unsigned char* src = filtered.data() + static_cast<std::size_t>(y) * (stride + 1);
    unsigned char* row = img.data() + static_cast<std::size_t>(y) * stride;
    const unsigned char* prev = y > 0 ? row - stride : nullptr;
    const int filter = src[0];
    for (std::size_t x = 0; x < stride; ++x) {
      const int a = x >= static_cast<std::size_t>(channels) ? row[x - channels] : 0;
      const int b = prev ? prev[x] : 0;
      const int c = prev && x >= static_cast<std::size_t>(channels) ? prev[x - channels] : 0;
      int pred = 0;
      switch (filter) {
        case 0: pred = 0; break;
        case 1: pred = a; break;
        case 2: pred = b; break;
        case 3: pred = (a + b) / 2; break;
        case 4: {
          const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
          pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
          break;
        }
        default: throw std::runtime_error(path + ": bad filter type");
      }
      row[x] = static_cast<unsigned char>(src[x + 1] + pred);
    }
  }
  if (channels == 3) return img;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
  for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
    for (int ch = 0; ch < 3; ++ch) rgb[i * 3 + static_cast<std::size_t>(ch)] = img[i * 4 + static_cast<std::size_t>(ch)];
  }
  return rgb;
}

void put32(std::vector<unsigned char>& v, std::uint32_t x) {
  for (int s = 24; s >= 0; s -= 8) v.push_back(static_cast<unsigned char>(x >> s));
}
void chunk(std::vector<unsigned char>& out, const char* type, const std::vector<unsigned char>& body) {
  put32(out, static_cast<std::uint32_t>(body.size()));
  const std::size_t start = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), body.begin(), body.end());
  put32(out, static_cast<std::uint32_t>(crc32(0, out.data() + start, static_cast<uInt>(out.size() - start))));
}

void write_png_rgb(const std::string& path, const std::vector<std::uint8_t>& rgb, int width, int height) {
  const std::size_t stride = static_cast<std::size_t>(width) * 3;
  std::vector<unsigned char> raw((stride + 1) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    raw[static_cast<std::size_t>(y) * (stride + 1)] = 0;
    std::memcpy(raw.data() + static_cast<std::size_t>(y) * (stride + 1) + 1, rgb.data() + static_cast<std::size_t>(y) * stride, stride);
  }
  uLongf len = compressBound(static_cast<uLong>(raw.size()));
  std::vector<unsigned char> z(len);
  if (compress2(z.data(), &len, raw.data(), static_cast<uLong>(raw.size()), 6) != Z_OK) throw std::runtime_error("deflate failed");
  z.resize(len);
  std::vector<unsigned char> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<unsigned char> ihdr;
  put32(ihdr, static_cast<std::uint32_t>(width));
  put32(ihdr, static_cast<std::uint32_t>(height));
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
  chunk(png, "IHDR", ihdr);
  chunk(png, "IDAT", z);
  chunk(png, "IEND", {});
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
  if (!out) throw std::runtime_error("cannot write " + path);
}

// ---- Host replays -----------------------------------------------------------------------------------------------------

float host_tf32(float v) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7f800000u) == 0x7f800000u) return v;
  u = (u + 0x1000u) & 0xffffe000u;
  return std::bit_cast<float>(u);
}
float host_bf16(float v) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(v);
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  return std::bit_cast<float>(u);
}

std::vector<float> random_f32(std::uint64_t seed, std::size_t n, float lo, float hi, bool bf16_values = false) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = bf16_values ? host_bf16(dist(rng)) : dist(rng);
  return v;
}

std::unique_ptr<Buffer> upload(Device& device, const std::vector<float>& v) {
  auto b = std::make_unique<Buffer>(device, std::max<std::size_t>(v.size(), 4) * 4);
  b->upload(v.data(), v.size() * 4);
  return b;
}
std::vector<float> download(Buffer& b, std::size_t n) {
  std::vector<float> v(n);
  b.download(v.data(), n * 4);
  return v;
}

struct Limit {
  double max_err;
  const char* why;
};

// err_i = |got - ref| / (|ref| + floor_i).
bool report(const char* name, const std::vector<float>& got, const std::vector<double>& ref, const std::vector<double>& floor,
            Limit limit) {
  double worst = 0.0;
  std::size_t same = 0, nans = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isnan(got[i])) {
      ++nans;
      continue;
    }
    same += got[i] == static_cast<float>(ref[i]);
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - ref[i]) / (std::abs(ref[i]) + floor[i]));
  }
  const bool ok = nans == 0 && worst <= limit.max_err;
  std::printf("  %-40s %s  %zu values, %.2f %% same as the fp32-rounded replay, worst scaled error %.3g (limit %.3g: %s), %zu NaN\n",
              name, ok ? "ok    " : "FAILED", got.size(), 100.0 * static_cast<double>(same) / static_cast<double>(got.size()),
              worst, limit.max_err, limit.why, nans);
  return ok;
}

struct ConvCase {
  const char* name;
  std::uint32_t taps;  // 0 GEMM, 1, 9
  bool upsample;
  std::uint32_t h, w, c;       // conv output grid and input channels
  std::uint32_t m, k, lda;     // GEMM
  std::uint32_t n, ldb, ldo;
  bool bias, residual, b_bf16;
  VaePrecision precision;
  std::uint32_t band;  // rows per dispatch (0: one)
};

// Operand as the kernel sees it: TF32-rounded or exact.
double operand(float v, VaePrecision p) { return p == VaePrecision::kTf32 ? host_tf32(v) : v; }

bool check_conv(Device& device, const VaeKernels& k, const ConvCase& cc) {
  const bool conv = cc.taps != 0;
  const std::uint32_t m = conv ? cc.h * cc.w : cc.m, kk = conv ? cc.taps * cc.c : cc.k, lda = conv ? 0 : cc.lda;
  const std::uint32_t in_h = cc.upsample ? cc.h / 2 : cc.h, in_w = cc.upsample ? cc.w / 2 : cc.w;
  const std::size_t a_count = conv ? static_cast<std::size_t>(in_h) * in_w * cc.c : static_cast<std::size_t>(m) * lda;
  const auto a = random_f32(1, a_count, -2.0f, 2.0f);
  const auto b = random_f32(2, static_cast<std::size_t>(cc.n) * cc.ldb, -0.5f, 0.5f, cc.b_bf16);
  const auto bias = random_f32(3, cc.n, -1.0f, 1.0f);
  const auto res = random_f32(4, static_cast<std::size_t>(m) * cc.ldo, -4.0f, 4.0f);
  auto da = upload(device, a), db = upload(device, b), dbias = upload(device, bias);
  auto dout = upload(device, res);  // poisoned with the residual; written in place when residual is on
  const double nan = std::nan("");
  std::vector<double> ref(static_cast<std::size_t>(m) * cc.ldo, nan), floor(ref.size(), 0.0);
  for (std::uint32_t r = 0; r < m; ++r) {
    for (std::uint32_t n = 0; n < cc.n; ++n) {
      double sum = 0.0, mag = 0.0;
      for (std::uint32_t q = 0; q < kk; ++q) {
        float av = 0.0f;
        if (!conv) {
          av = a[static_cast<std::size_t>(r) * lda + q];
        } else {
          const std::uint32_t tap = q / cc.c, ch = q % cc.c, oy = r / cc.w, ox = r % cc.w;
          int iy = static_cast<int>(oy), ix = static_cast<int>(ox);
          if (cc.taps == 9) {
            iy += static_cast<int>(tap / 3) - 1;
            ix += static_cast<int>(tap % 3) - 1;
          }
          if (iy < 0 || ix < 0 || iy >= static_cast<int>(cc.h) || ix >= static_cast<int>(cc.w)) continue;
          if (cc.upsample) {
            iy /= 2;
            ix /= 2;
          }
          av = a[(static_cast<std::size_t>(iy) * in_w + static_cast<std::size_t>(ix)) * cc.c + ch];
        }
        const double p = operand(av, cc.precision) * operand(b[static_cast<std::size_t>(n) * cc.ldb + q], cc.precision);
        sum += p;
        mag += std::abs(p);
      }
      float y = static_cast<float>(sum);
      if (cc.bias) {
        y = y + bias[n];
        mag += std::abs(bias[n]);
      }
      if (cc.residual) {
        y = y + res[static_cast<std::size_t>(r) * cc.ldo + n];
        mag += std::abs(res[static_cast<std::size_t>(r) * cc.ldo + n]);
      }
      ref[static_cast<std::size_t>(r) * cc.ldo + n] = y;
      floor[static_cast<std::size_t>(r) * cc.ldo + n] = mag;
    }
  }
  ConvArgs args;
  args.a = da->address();
  args.b = db->address();
  args.out = dout->address();
  args.bias = cc.bias ? dbias->address() : 0;
  args.residual = cc.residual ? dout->address() : 0;
  args.m = m;
  args.n = cc.n;
  args.k = kk;
  args.lda = lda;
  args.ldb = cc.ldb;
  args.ldo = cc.ldo;
  args.taps = cc.taps;
  args.upsample = cc.upsample;
  args.h = cc.h;
  args.w = cc.w;
  args.c = cc.c;
  args.b_bf16 = cc.b_bf16;
  device.run([&](VkCommandBuffer cb) {
    const std::uint32_t band = cc.band == 0 ? m : cc.band;
    for (std::uint32_t m0 = 0; m0 < m; m0 += band) {
      ConvArgs part = args;
      part.m0 = m0;
      part.m = std::min(m, m0 + band);
      k.record_conv(cb, part, cc.precision);
    }
  });
  const std::vector<float> got_all = download(*dout, ref.size());
  // Columns past n keep the poison (the residual) untouched: checked bit for bit.
  std::vector<float> got;
  std::vector<double> want, fl;
  std::size_t clobbered = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (std::isnan(ref[i])) {
      clobbered += got_all[i] != res[i];
      continue;
    }
    got.push_back(got_all[i]);
    want.push_back(ref[i]);
    fl.push_back(floor[i]);
  }
  // Products are exact (TF32, or a bf16-valued B with A in 3 pieces) or carry 2^-17 relative error per operand (2 pieces);
  // the rest is fp32 accumulation, at most about 2^-24 per add on |products|.
  const bool exact_products = cc.precision == VaePrecision::kTf32 || (cc.precision == VaePrecision::kFp32 && cc.b_bf16) ||
                              k.path() == VaePath::kPlain;
  const Limit limit = exact_products || cc.n < 64 ? Limit{2e-6, "fp32 accumulation"}
                                                  : Limit{4e-5, "16-bit operand pieces + fp32 accumulation"};
  bool ok = report(cc.name, got, want, fl, limit);
  if (clobbered != 0) {
    std::printf("  %-40s FAILED  %zu values outside the output columns were written\n", cc.name, clobbered);
    ok = false;
  }
  return ok;
}

bool check_convs(Device& device, const VaeKernels& k) {
  const ConvCase cases[] = {
      // name, taps, up, h, w, c, m, k, lda, n, ldb, ldo, bias, res, b_bf16, precision, band
      {"conv3x3 tf32 bias+residual, banded", 9, false, 13, 21, 32, 0, 0, 0, 96, 288, 100, true, true, true, VaePrecision::kTf32, 128},
      {"conv3x3 tf32 upsample, fp32 B", 9, true, 14, 18, 48, 0, 0, 0, 136, 432, 140, true, false, false, VaePrecision::kTf32, 0},
      {"conv1x1 tf32", 1, false, 11, 13, 40, 0, 0, 0, 72, 40, 76, true, false, true, VaePrecision::kTf32, 0},
      {"conv3x3 tf32 narrow (n = 3)", 9, false, 9, 30, 24, 0, 0, 0, 3, 216, 4, true, false, true, VaePrecision::kTf32, 0},
      {"conv3x3 tf32 narrow, c = 96 (n = 3)", 9, false, 5, 7, 96, 0, 0, 0, 3, 864, 3, true, true, true, VaePrecision::kTf32, 0},
      {"gemm fp32 narrow, k > 512 (n = 4)", 0, false, 0, 0, 0, 300, 1100, 1104, 4, 1100, 4, true, false, false, VaePrecision::kFp32, 0},
      {"conv1x1 tf32 narrow (n = 32)", 1, false, 10, 15, 32, 0, 0, 0, 32, 32, 32, true, false, true, VaePrecision::kTf32, 0},
      {"gemm fp32 bf16 B bias+residual", 0, false, 0, 0, 0, 150, 100, 104, 136, 100, 140, true, true, true, VaePrecision::kFp32, 0},
      {"gemm fp32, fp32 B", 0, false, 0, 0, 0, 77, 64, 64, 384, 68, 388, false, false, false, VaePrecision::kFp32, 0},
      {"gemm split16", 0, false, 0, 0, 0, 200, 68, 72, 260, 68, 264, false, false, false, VaePrecision::kSplit16, 0},
  };
  bool ok = true;
  for (const ConvCase& c : cases) ok = check_conv(device, k, c) && ok;
  return ok;
}

bool check_group_norm(Device& device, const VaeKernels& k, std::uint32_t pixels, std::uint32_t c, bool silu, bool in_place,
                      float offset) {
  std::vector<float> x = random_f32(10 + c, static_cast<std::size_t>(pixels) * c, -3.0f, 3.0f);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] += offset * static_cast<float>(1 + (i % c) / (c / 32));
  const auto gamma = random_f32(11, c, 0.5f, 1.5f), beta = random_f32(12, c, -0.5f, 0.5f);
  auto dx = upload(device, x), dg = upload(device, gamma), db = upload(device, beta);
  auto dy = upload(device, std::vector<float>(x.size(), 0.0f));
  Buffer part(device, VaeKernels::gn_part_bytes(pixels, c)), stats(device, 256);
  const VkDeviceAddress y = in_place ? dx->address() : dy->address();
  device.run([&](VkCommandBuffer cb) {
    k.record_gn_stats(cb, dx->address(), pixels, c, part.address(), stats.address(), 1e-6f);
    k.record_gn_apply(cb, dx->address(), y, pixels, c, stats.address(), dg->address(), db->address(), silu);
  });
  const std::uint32_t cpg = c / 32;
  std::vector<double> ref(x.size()), floor(x.size());
  for (std::uint32_t g = 0; g < 32; ++g) {
    double s = 0.0;
    for (std::uint32_t p = 0; p < pixels; ++p) {
      for (std::uint32_t j = 0; j < cpg; ++j) s += x[static_cast<std::size_t>(p) * c + g * cpg + j];
    }
    const double n = static_cast<double>(pixels) * cpg, mean = s / n;
    double q = 0.0;
    for (std::uint32_t p = 0; p < pixels; ++p) {
      for (std::uint32_t j = 0; j < cpg; ++j) {
        const double d = x[static_cast<std::size_t>(p) * c + g * cpg + j] - mean;
        q += d * d;
      }
    }
    const auto mean_f = static_cast<float>(mean), rstd = static_cast<float>(1.0 / std::sqrt(q / n + 1e-6));
    for (std::uint32_t j = 0; j < cpg; ++j) {
      const std::uint32_t ch = g * cpg + j;
      const float a = rstd * gamma[ch], b = std::fma(-a, mean_f, beta[ch]);
      for (std::uint32_t p = 0; p < pixels; ++p) {
        const std::size_t i = static_cast<std::size_t>(p) * c + ch;
        float v = std::fma(a, x[i], b);
        if (silu) v = v / (1.0f + std::exp(-v));
        ref[i] = v;
        // The statistics carry fp32 rounding: up to a few ulps of mean and rstd, i.e. of |a x| and |a mean|.
        floor[i] = std::abs(a * x[i]) + std::abs(a * mean_f);
      }
    }
  }
  const auto got = download(in_place ? *dx : *dy, x.size());
  const std::string name = "group_norm c=" + std::to_string(c) + (silu ? " silu" : "") + (in_place ? " in place" : "") +
                           " offset " + std::to_string(static_cast<int>(offset));
  return report(name.c_str(), got, ref, floor, {2e-6, "fp32 statistics, exp"});
}

bool check_softmax(Device& device, const VaeKernels& k) {
  constexpr std::uint32_t rows = 37, l = 300, ld = 304;
  const float scale = 0.05103104f;
  const auto s = random_f32(20, static_cast<std::size_t>(rows) * ld, -60.0f, 60.0f);
  auto ds = upload(device, s);
  device.run([&](VkCommandBuffer cb) { k.record_softmax(cb, ds->address(), rows, l, ld, scale); });
  const auto all = download(*ds, s.size());
  std::vector<float> got;
  std::vector<double> ref, floor;
  std::size_t clobbered = 0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    const float* row = s.data() + static_cast<std::size_t>(r) * ld;
    float mx = row[0];
    for (std::uint32_t j = 1; j < l; ++j) mx = std::max(mx, row[j]);
    const float shift = mx * scale;
    double sum = 0.0;
    for (std::uint32_t j = 0; j < l; ++j) sum += std::exp(static_cast<double>(row[j] * scale - shift));
    for (std::uint32_t j = 0; j < ld; ++j) {
      const std::size_t i = static_cast<std::size_t>(r) * ld + j;
      if (j >= l) {
        clobbered += all[i] != s[i];
        continue;
      }
      got.push_back(all[i]);
      ref.push_back(std::exp(static_cast<double>(row[j] * scale - shift)) / sum);
      floor.push_back(1e-30);
    }
  }
  // exp is accurate to a few ulps of its result in Vulkan (3 + 2 |x| ulps), here |x| up to ~6.
  bool ok = report("softmax", got, ref, floor, {3e-6, "Vulkan exp precision"});
  if (clobbered != 0) {
    std::printf("  softmax: %zu values past l were written\n", clobbered);
    ok = false;
  }
  return ok;
}

// Bit-exact ops: the transpose, the latent unpack and RGB8.
bool check_exact(Device& device, const VaeKernels& k) {
  bool ok = true;
  {
    constexpr std::uint32_t r = 70, c = 45, ld_in = 48, ld_out = 72;
    const auto in = random_f32(30, static_cast<std::size_t>(r) * ld_in, -1.0f, 1.0f);
    auto din = upload(device, in), dout = upload(device, std::vector<float>(static_cast<std::size_t>(c) * ld_out, -7.0f));
    device.run([&](VkCommandBuffer cb) { k.record_transpose(cb, din->address(), dout->address(), r, c, ld_in, ld_out); });
    const auto got = download(*dout, static_cast<std::size_t>(c) * ld_out);
    std::size_t bad = 0;
    for (std::uint32_t i = 0; i < c; ++i) {
      for (std::uint32_t j = 0; j < ld_out; ++j) {
        const float want = j < r ? in[static_cast<std::size_t>(j) * ld_in + i] : -7.0f;
        bad += std::bit_cast<std::uint32_t>(got[static_cast<std::size_t>(i) * ld_out + j]) != std::bit_cast<std::uint32_t>(want);
      }
    }
    std::printf("  %-40s %s  %zu of %zu values differ\n", "transpose", bad == 0 ? "ok    " : "FAILED", bad, got.size());
    ok = ok && bad == 0;
  }
  {
    constexpr std::uint32_t h = 3, w = 5;
    const auto z = random_f32(31, 128ull * h * w, -4.0f, 4.0f);
    const auto var = random_f32(32, 128, 0.1f, 9.0f), mean = random_f32(33, 128, -1.0f, 1.0f);
    std::vector<float> scale(128);
    for (int i = 0; i < 128; ++i) scale[static_cast<std::size_t>(i)] = std::sqrt(var[static_cast<std::size_t>(i)] + 1e-4f);
    auto dz = upload(device, z), ds = upload(device, scale), dm = upload(device, mean);
    Buffer y(device, 4ull * h * w * 32 * 4);
    device.run([&](VkCommandBuffer cb) { k.record_latent_unpack(cb, dz->address(), ds->address(), dm->address(), y.address(), h, w); });
    const auto got = download(y, 4ull * h * w * 32);
    std::size_t bad = 0;
    for (std::uint32_t oy = 0; oy < 2 * h; ++oy) {
      for (std::uint32_t ox = 0; ox < 2 * w; ++ox) {
        for (std::uint32_t c = 0; c < 32; ++c) {
          const std::uint32_t ch = 4 * c + 2 * (oy & 1) + (ox & 1);
          const float v = z[(static_cast<std::size_t>(ch) * h + oy / 2) * w + ox / 2];
          const float scaled = v * scale[ch];
          const float want = scaled + mean[ch];
          bad += std::bit_cast<std::uint32_t>(got[(static_cast<std::size_t>(oy) * 2 * w + ox) * 32 + c]) != std::bit_cast<std::uint32_t>(want);
        }
      }
    }
    std::printf("  %-40s %s  %zu of %zu values differ\n", "latent unpack", bad == 0 ? "ok    " : "FAILED", bad, got.size());
    ok = ok && bad == 0;
  }
  {
    std::vector<float> x = random_f32(34, 1001, -1.3f, 1.3f);
    const float special[] = {std::nanf(""), -1.0f, 1.0f, 0.0f, -0.0f, 1.0f / 255.0f, 0.5f / 255.0f * 2.0f - 1.0f,
                             1.5f / 255.0f * 2.0f - 1.0f, 2.5f / 255.0f * 2.0f - 1.0f, INFINITY, -INFINITY};
    std::copy(std::begin(special), std::end(special), x.begin());
    auto dx = upload(device, x);
    Buffer rgb(device, 1004);
    device.run([&](VkCommandBuffer cb) { k.record_rgb8(cb, dx->address(), rgb.address(), static_cast<std::uint32_t>(x.size())); });
    std::vector<std::uint8_t> got(1004);
    rgb.download(got.data(), got.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      std::uint8_t want = 0;
      if (i < x.size() && !std::isnan(x[i])) {
        float v = std::min(std::max(x[i], -1.0f), 1.0f);
        const float half = v * 0.5f;
        v = std::min(std::max(half + 0.5f, 0.0f), 1.0f);
        want = static_cast<std::uint8_t>(std::nearbyint(v * 255.0f));
      }
      bad += got[i] != want;
    }
    std::printf("  %-40s %s  %zu of %zu bytes differ\n", "rgb8", bad == 0 ? "ok    " : "FAILED", bad, got.size());
    ok = ok && bad == 0;
  }
  return ok;
}

bool run_kat(Device& device, const VaeKernels& k) {
  std::printf("known-answer checks against host replays (path %s):\n", vae_path_name(k.path()));
  bool ok = check_convs(device, k);
  ok = check_group_norm(device, k, 1000, 96, true, false, 0.0f) && ok;
  ok = check_group_norm(device, k, 777, 384, false, true, 20.0f) && ok;
  ok = check_group_norm(device, k, 2050, 192, true, true, 5.0f) && ok;
  ok = check_softmax(device, k) && ok;
  ok = check_exact(device, k) && ok;
  return ok;
}

// ---- Kernel timing ----------------------------------------------------------------------------------------------------

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// The decoder's heaviest shapes at 1024x1024, one submit each, best of 3.
void run_bench(Device& device, const VaeKernels& k) {
  struct Shape {
    const char* name;
    std::uint32_t h, w, c, n, taps;
    bool upsample;
  };
  const Shape shapes[] = {{"up3 conv 96->96 3x3 @1024", 1024, 1024, 96, 96, 9, false},
                          {"up2 upsampler 192 3x3 @1024", 1024, 1024, 192, 192, 9, true},
                          {"up2 conv 384->192 3x3 @512", 512, 512, 384, 192, 9, false},
                          {"up1 upsampler 384 3x3 @512", 512, 512, 384, 384, 9, true},
                          {"mid conv 384 3x3 @128", 128, 128, 384, 384, 9, false},
                          {"conv_out 96->3 3x3 @1024", 1024, 1024, 96, 3, 9, false}};
  std::printf("\nkernel timing (path %s), best of 3 single submits:\n", vae_path_name(k.path()));
  for (const Shape& s : shapes) {
    const std::uint64_t in_px = static_cast<std::uint64_t>(s.h) * s.w / (s.upsample ? 4 : 1);
    Buffer x(device, in_px * s.c * 4), wt(device, 9ull * s.c * s.n * 4), y(device, static_cast<std::uint64_t>(s.h) * s.w * s.n * 4);
    device.run([&](VkCommandBuffer cb) {
      vkCmdFillBuffer(cb, x.handle(), 0, VK_WHOLE_SIZE, 0x3c000000u);
      vkCmdFillBuffer(cb, wt.handle(), 0, VK_WHOLE_SIZE, 0x3c000000u);
    });
    ConvArgs a;
    a.a = x.address();
    a.b = wt.address();
    a.out = y.address();
    a.m = s.h * s.w;
    a.n = s.n;
    a.k = s.taps * s.c;
    a.ldb = a.k;
    a.ldo = s.n;
    a.taps = s.taps;
    a.upsample = s.upsample;
    a.h = s.h;
    a.w = s.w;
    a.c = s.c;
    a.b_bf16 = true;
    double best = 1e30;
    for (int it = 0; it < 3; ++it) {
      const double t0 = now_s();
      device.run([&](VkCommandBuffer cb) {
        for (std::uint32_t m0 = 0; m0 < a.m; m0 += 65535u * 64u) {
          ConvArgs band = a;
          band.m0 = m0;
          band.m = std::min(a.m, m0 + 65535u * 64u);
          k.record_conv(cb, band, VaePrecision::kTf32);
        }
      });
      best = std::min(best, now_s() - t0);
    }
    const double macs = static_cast<double>(a.m) * a.n * a.k;
    std::printf("  %-30s %9.2f ms  %7.2f TFLOPS (useful)\n", s.name, best * 1e3, 2.0 * macs / best / 1e12);
  }
}

// The DiT block's bf16 GEMM at 8222 x 3072 x 3072 (README: 33.1 TFLOPS on the coop path, 1.62 on the plain path, on an
// idle RTX 4090 Laptop), timed next to a decode: when another process shares the GPU, its rate shows how much.
double calibration_tflops(Device& device) {
  const BlockKernels block(device);
  constexpr std::uint32_t m = 8222, n = 3072, k = 3072;
  Buffer a(device, 2ull * m * k), b(device, 2ull * n * k), c(device, 2ull * m * n);
  device.run([&](VkCommandBuffer cb) {
    vkCmdFillBuffer(cb, a.handle(), 0, VK_WHOLE_SIZE, 0x3c003c00u);
    vkCmdFillBuffer(cb, b.handle(), 0, VK_WHOLE_SIZE, 0x3c003c00u);
  });
  double best = 1e30;
  for (int it = 0; it < 3; ++it) {
    const double t0 = now_s();
    device.run([&](VkCommandBuffer cb) { block.record_gemm(cb, c.address(), n, a.address(), k, b.address(), k, m, n, k); });
    best = std::min(best, now_s() - t0);
  }
  return 2.0 * m * n * k / best / 1e12;
}

// ---- Real decode ------------------------------------------------------------------------------------------------------

struct Options {
  int device = -1;
  std::string path;
  bool kat = true, decode = true, bench = false, profile = false, calibrate = false;
  std::string vae = "/workspace/Models/MOSS-Image/base/vae/diffusion_pytorch_model.safetensors";
  std::string latent = "/workspace/Projects/MossInfer/tests/golden/g_fridge_1024/final_latent.npy";
  std::string golden = "/workspace/Projects/MossInfer/tests/golden/g_fridge_1024/image.png";
  std::string out;
  std::uint32_t crop_h = 0, crop_w = 0;
  int iters = 2;
  std::uint64_t max_macs = 0;
};

bool run_decode(Device& device, const Options& o, VaePath path) {
  std::vector<std::int64_t> shape;
  std::vector<float> latent = read_npy_f32(o.latent, shape);
  if (shape.size() == 4 && shape[0] == 1) shape.erase(shape.begin());
  if (shape.size() != 3 || shape[0] != 128) throw std::runtime_error(o.latent + ": expected [1, 128, h, w] or [128, h, w]");
  auto h = static_cast<std::uint32_t>(shape[1]), w = static_cast<std::uint32_t>(shape[2]);
  const bool cropped = o.crop_h != 0;
  if (cropped) {
    // The top-left crop_h x crop_w latents: no reference image exists for a crop; the decode runs and is written only.
    if (o.crop_h > h || o.crop_w > w) throw std::runtime_error("--crop exceeds the latent grid");
    std::vector<float> c(128ull * o.crop_h * o.crop_w);
    for (std::uint32_t ch = 0; ch < 128; ++ch) {
      for (std::uint32_t y = 0; y < o.crop_h; ++y) {
        std::memcpy(&c[(static_cast<std::size_t>(ch) * o.crop_h + y) * o.crop_w],
                    &latent[(static_cast<std::size_t>(ch) * h + y) * w], o.crop_w * 4);
      }
    }
    latent.swap(c);
    h = o.crop_h;
    w = o.crop_w;
  }
  VaeDecoder decoder(device, o.vae, path);
  if (o.max_macs != 0) decoder.set_max_macs_per_submit(o.max_macs);
  decoder.set_profile(o.profile);
  std::printf("\ndecode %ux%u (latents %ux%u, decoder c = %d, path %s): weights %.1f MB, decode allocates %.3f GiB\n", 16 * w,
              16 * h, h, w, decoder.base_channels(), vae_path_name(path), static_cast<double>(decoder.weight_bytes()) / 1e6,
              static_cast<double>(decoder.workspace_bytes(h, w)) / (1024.0 * 1024.0 * 1024.0));
  if (o.calibrate) {
    std::printf("  calibration before: the block's GEMM 8222x3072x3072 at %.2f TFLOPS (%s path)\n", calibration_tflops(device),
                device.info().caps.coop_bf16 ? "coop" : "plain");
  }
  std::vector<std::uint8_t> rgb;
  std::vector<float> image;
  std::vector<double> times;
  VaeDecodeStats st;
  for (int i = 0; i < std::max(1, o.iters); ++i) {
    st = decoder.decode(latent.data(), h, w, rgb, i == 0 ? &image : nullptr);
    times.push_back(st.seconds);
    std::printf("  run %d: %.3f s (%d submits)%s\n", i, st.seconds, st.submits, i == 0 ? ", pipelines created" : "");
  }
  if (o.calibrate) std::printf("  calibration after: %.2f TFLOPS\n", calibration_tflops(device));
  if (o.profile) {
    double total = 0.0;
    for (const auto& [name, s] : st.profile) total += s;
    std::printf("  profile of the last run (one submit per op, %.3f s in submits):\n", total);
    for (const auto& [name, s] : st.profile) std::printf("    %-48s %9.2f ms %5.1f %%\n", name.c_str(), s * 1e3, 100.0 * s / total);
  }
  std::size_t nans = 0;
  for (const float v : image) nans += std::isnan(v);
  std::printf("  decoder output: %zu NaN of %zu\n", nans, image.size());
  if (!o.out.empty()) {
    write_png_rgb(o.out, rgb, static_cast<int>(16 * w), static_cast<int>(16 * h));
    std::printf("  wrote %s\n", o.out.c_str());
  }
  if (cropped || o.golden.empty()) return nans == 0;
  int gw = 0, gh = 0;
  const std::vector<std::uint8_t> golden = read_png_rgb(o.golden, gw, gh);
  if (static_cast<std::uint32_t>(gw) != 16 * w || static_cast<std::uint32_t>(gh) != 16 * h) {
    throw std::runtime_error(o.golden + " is " + std::to_string(gw) + "x" + std::to_string(gh) + ", not the decode's size");
  }
  double se = 0.0, sum_abs = 0.0;
  int max_abs = 0;
  std::size_t same = 0;
  std::vector<std::size_t> hist(4, 0);
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    const int d = std::abs(static_cast<int>(rgb[i]) - static_cast<int>(golden[i]));
    se += static_cast<double>(d) * d;
    sum_abs += d;
    max_abs = std::max(max_abs, d);
    same += d == 0;
    ++hist[static_cast<std::size_t>(std::min(d, 3))];
  }
  const double n = static_cast<double>(rgb.size()), mse = se / n;
  const double psnr = mse == 0.0 ? INFINITY : 10.0 * std::log10(255.0 * 255.0 / mse);
  std::printf("  against %s:\n    PSNR %.2f dB, max |diff| %d, mean |diff| %.5f, identical bytes %.4f %%\n", o.golden.c_str(),
              psnr, max_abs, sum_abs / n, 100.0 * static_cast<double>(same) / n);
  std::printf("    |diff| 0: %zu, 1: %zu, 2: %zu, >= 3: %zu bytes\n", hist[0], hist[1], hist[2], hist[3]);
  return nans == 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
      return argv[++i];
    };
    try {
      if (arg == "--device") {
        o.device = std::stoi(next());
      } else if (arg == "--path") {
        o.path = next();
      } else if (arg == "--kat-only") {
        o.decode = false;
      } else if (arg == "--bench") {
        o.bench = true;
        o.decode = false;
      } else if (arg == "--calibrate") {
        o.calibrate = true;
      } else if (arg == "--profile") {
        o.profile = true;
      } else if (arg == "--no-kat") {
        o.kat = false;
      } else if (arg == "--vae") {
        o.vae = next();
      } else if (arg == "--latent") {
        o.latent = next();
      } else if (arg == "--golden") {
        o.golden = next();
      } else if (arg == "--out") {
        o.out = next();
      } else if (arg == "--crop") {
        const std::string v = next();
        const std::size_t comma = v.find(',');
        o.crop_h = static_cast<std::uint32_t>(std::stoul(v.substr(0, comma)));
        o.crop_w = static_cast<std::uint32_t>(std::stoul(v.substr(comma + 1)));
      } else if (arg == "--iters") {
        o.iters = std::stoi(next());
      } else if (arg == "--max-macs") {
        o.max_macs = std::stoull(next());
      } else {
        throw std::invalid_argument("unknown argument " + arg);
      }
    } catch (const std::exception& e) {
      std::printf("%s\nusage: %s [--device N] [--path coop|plain] [--kat-only | --no-kat | --bench] [--vae FILE] [--latent NPY]\n"
                  "       [--golden PNG] [--out PNG] [--crop H,W] [--iters N] [--max-macs N] [--profile] [--calibrate]\n",
                  e.what(), argv[0]);
      return 2;
    }
  }
  if (!o.path.empty() && o.path != "coop" && o.path != "plain") {
    std::printf("--path takes coop or plain\n");
    return 2;
  }
  try {
    Device device(o.device);
    const VaePath path = o.path.empty() ? (device.info().caps.coop_bf16 ? VaePath::kCoop : VaePath::kPlain)
                                        : (o.path == "coop" ? VaePath::kCoop : VaePath::kPlain);
    std::printf("device %u: %s, path %s%s\n", device.info().index, device.info().name.c_str(), vae_path_name(path),
                device.info().caps.coop_bf16 ? "" : (" (no coop: " + device.info().caps.coop_bf16_reason + ")").c_str());
    bool ok = true;
    if (o.kat) {
      const VaeKernels kernels(device, path);
      ok = run_kat(device, kernels);
      if (!ok) std::printf("known-answer check FAILED\n");
    }
    if (o.bench) {
      const VaeKernels kernels(device, path);
      run_bench(device, kernels);
    }
    if (o.decode) ok = run_decode(device, o, path) && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
