// Checks the VAE decoder (vae.h, vae_kernels.h): each kernel against a host replay of its op order on small shapes with
// tails on every axis, then the real decoder on golden final latents against the reference pipeline's PNGs.
//   - Convolutions and linears: the replay sums the exact products of the operands as the kernel forms them (TF32-rounded,
//     or fp32) in fp64. Errors are scaled, |got - replay| / (|replay| + floor), with the floor the sum of the |products|
//     (what an fp32 sum can be wrong by in any order), limit 2e-6: fp32 accumulation. The TF32 convolutions without the
//     lo x lo products are also measured against the exact TF32 replay.
//   - piece_gemm: against the fp64 sum of the exact fp32 products, same limit (each dropped product is at most 2^-24 |a b|).
//   - That limit cannot tell the product modes apart, so cases whose sums fp32 adds exactly in any order are gated bit for
//     bit: a K = 16 convolution of TF32 values (4 products must give the exact sums, 3 the sums without lo x lo, and the
//     two must differ), and the fp32 linear and piece_gemm on one 24-bit value times a power of two per output (needing
//     the products with piece 2 of either operand).
//   - Infinities and NaN in an operand give non-finite outputs exactly where the replay's do, the others stay right.
//   - split_pack, the latent unpack, RGB8 and the GroupNorm statistics bit for bit; the softmax's pieces must be an exact
//     split of an fp32 probability within 3e-6 of the fp64 softmax; GroupNorm apply within 2e-6 of |a x| + |a mean|.
//   - The decode: PSNR, max and mean |diff| and identical bytes against the golden PNG. It passes with no NaN, no byte off
//     by more than 1 and at least 99 % of the bytes identical (the CUDA engine's reference decode: 99.1-99.3 %, max 1).
//   - Non-square decodes (run_nonsquare) and the size limits (check_limits).
// Exit status: 0 everything passed, 1 a check failed, 2 bad usage or no usable device.
// Usage: hip_vae_check [--device N] [--kat-only | --no-kat] [--vae FILE] [--golden DIR] [--case NAME]... [--no-golden] [--no-nonsquare]
//                      [--out-dir DIR] [--iters N] [--profile] [--conv-products 4|3]
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
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy.h"
#include "runtime.h"
#include "safetensors.h"
#include "vae.h"
#include "vae_kernels.h"

namespace {

using namespace mi::hip;
using hip_vae::Products;
namespace fs = std::filesystem;

// ---- PNG -------------------------------------------------------------------------------------------------------------

std::vector<char> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::vector<char> data(static_cast<std::size_t>(in.tellg()));
  in.seekg(0);
  in.read(data.data(), static_cast<std::streamsize>(data.size()));
  return data;
}

std::uint32_t be32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) << 24 | static_cast<std::uint32_t>(p[1]) << 16 | static_cast<std::uint32_t>(p[2]) << 8 | p[3];
}

// 8-bit RGB or RGBA, non-interlaced PNG -> RGB.
std::vector<std::uint8_t> read_png_rgb(const fs::path& path, int& width, int& height) {
  const std::vector<char> raw = read_file(path);
  const auto* d = reinterpret_cast<const unsigned char*>(raw.data());
  if (raw.size() < 8 || std::memcmp(d, "\x89PNG\r\n\x1a\n", 8) != 0) throw std::runtime_error(path.string() + ": not a PNG");
  std::size_t at = 8;
  int channels = 0;
  std::vector<unsigned char> idat;
  while (at + 12 <= raw.size()) {
    const std::uint32_t len = be32(d + at);
    const std::string type(reinterpret_cast<const char*>(d + at + 4), 4);
    if (at + 12 + len > raw.size()) throw std::runtime_error(path.string() + ": truncated chunk");
    const unsigned char* body = d + at + 8;
    if (type == "IHDR") {
      width = static_cast<int>(be32(body));
      height = static_cast<int>(be32(body + 4));
      if (body[8] != 8 || (body[9] != 2 && body[9] != 6) || body[12] != 0) {
        throw std::runtime_error(path.string() + ": only 8-bit RGB/RGBA non-interlaced PNGs are read");
      }
      channels = body[9] == 2 ? 3 : 4;
    } else if (type == "IDAT") {
      idat.insert(idat.end(), body, body + len);
    } else if (type == "IEND") {
      break;
    }
    at += 12 + len;
  }
  if (channels == 0) throw std::runtime_error(path.string() + ": no IHDR");
  const std::size_t stride = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
  std::vector<unsigned char> filtered((stride + 1) * static_cast<std::size_t>(height));
  uLongf out_len = static_cast<uLongf>(filtered.size());
  if (uncompress(filtered.data(), &out_len, idat.data(), static_cast<uLong>(idat.size())) != Z_OK || out_len != filtered.size()) {
    throw std::runtime_error(path.string() + ": cannot inflate the image data");
  }
  std::vector<unsigned char> img(stride * static_cast<std::size_t>(height));
  const auto ch = static_cast<std::size_t>(channels);
  for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y) {
    const unsigned char* src = filtered.data() + y * (stride + 1);
    unsigned char* row = img.data() + y * stride;
    const unsigned char* prev = y > 0 ? row - stride : nullptr;
    const int filter = src[0];
    for (std::size_t x = 0; x < stride; ++x) {
      const int a = x >= ch ? row[x - ch] : 0;
      const int b = prev ? prev[x] : 0;
      const int c = prev && x >= ch ? prev[x - ch] : 0;
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
        default: throw std::runtime_error(path.string() + ": bad filter type");
      }
      row[x] = static_cast<unsigned char>(src[x + 1] + pred);
    }
  }
  if (channels == 3) return img;
  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
  for (std::size_t i = 0; i < rgb.size() / 3; ++i) {
    for (std::size_t k = 0; k < 3; ++k) rgb[i * 3 + k] = img[i * 4 + k];
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

void write_png_rgb(const fs::path& path, const std::vector<std::uint8_t>& rgb, int width, int height) {
  const std::size_t stride = static_cast<std::size_t>(width) * 3;
  std::vector<unsigned char> raw((stride + 1) * static_cast<std::size_t>(height));
  for (std::size_t y = 0; y < static_cast<std::size_t>(height); ++y) {
    raw[y * (stride + 1)] = 0;
    std::memcpy(raw.data() + y * (stride + 1) + 1, rgb.data() + y * stride, stride);
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
  if (!out) throw std::runtime_error("cannot write " + path.string());
}

// ---- Device helpers and the error report ------------------------------------------------------------------------------

std::vector<float> random_f32(std::uint64_t seed, std::size_t n, float lo, float hi) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

// A hipMemcpy from pageable memory can return before its DMA lands and is not ordered with the checks' non-blocking
// streams: wait for it.
template <class T>
DeviceBuffer upload(const std::vector<T>& v) {
  DeviceBuffer b(std::max<std::size_t>(v.size() * sizeof(T), 16));
  b.upload(v.data(), v.size() * sizeof(T));
  check(hipDeviceSynchronize(), "hipDeviceSynchronize");
  return b;
}
template <class T>
std::vector<T> download(const DeviceBuffer& b, std::size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}

int failures = 0;

struct Limit {
  double max_err;
  const char* why;
};

// err_i = |got - ref| / (|ref| + floor_i). Returns the worst error; counts a failure when it is over the limit.
double report(const std::string& name, const std::vector<float>& got, const std::vector<double>& ref, const std::vector<double>& floor,
              Limit limit, bool gate = true) {
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
  if (gate && !ok) ++failures;
  std::printf("  %-52s %s %zu values, %6.2f %% same as the fp32-rounded replay, worst scaled error %.3g (limit %.3g: %s), %zu NaN\n",
              name.c_str(), !gate ? "measure" : ok ? "ok     " : "FAILED ", got.size(),
              100.0 * static_cast<double>(same) / static_cast<double>(got.size()), worst, limit.max_err, limit.why, nans);
  return worst;
}

void report_exact(const char* name, std::size_t bad, std::size_t total, const char* what) {
  if (bad != 0) ++failures;
  std::printf("  %-52s %s %zu of %zu %s differ\n", name, bad == 0 ? "ok     " : "FAILED ", bad, total, what);
}

// ---- Convolutions -----------------------------------------------------------------------------------------------------

struct ConvCase {
  const char* name;
  int taps;
  bool upsample;
  int h, w, c;  // output grid and input channels
  int n, ldo;
  bool bias, residual;
  Products products;
};

// The product of a and b as the kernel forms it.
double product(float a, float b, Products p) {
  if (p == Products::kFp32) return static_cast<double>(a) * b;
  if (p == Products::kTf32) return static_cast<double>(hip_vae::host_tf32(a)) * hip_vae::host_tf32(b);
  std::uint16_t pa[2], pb[2];
  hip_vae::host_split(a, p, pa);
  hip_vae::host_split(b, p, pb);
  const auto v = [](std::uint16_t x) { return static_cast<double>(hip_vae::host_bf16_value(x)); };
  return v(pa[0]) * v(pb[0]) + v(pa[0]) * v(pb[1]) + v(pa[1]) * v(pb[0]);  // each exact in fp64
}

void check_conv(const ConvCase& cc, hipStream_t stream) {
  const int k = cc.taps * cc.c;
  const std::int64_t m = static_cast<std::int64_t>(cc.h) * cc.w;
  const int in_h = cc.upsample ? cc.h / 2 : cc.h, in_w = cc.upsample ? cc.w / 2 : cc.w;
  const auto x = random_f32(1, static_cast<std::size_t>(in_h) * static_cast<std::size_t>(in_w) * static_cast<std::size_t>(cc.c), -2.0f, 2.0f);
  const auto wt = random_f32(2, static_cast<std::size_t>(cc.n) * static_cast<std::size_t>(k), -0.5f, 0.5f);
  const auto bias = random_f32(3, static_cast<std::size_t>(cc.n), -1.0f, 1.0f);
  const auto res = random_f32(4, static_cast<std::size_t>(m * cc.ldo), -4.0f, 4.0f);
  const Products pack_as = cc.products == Products::kFp32 ? Products::kFp32 : Products::kTf32;
  std::vector<std::uint16_t> packed(hip_vae::packed_weight_elems(cc.n, k, pack_as));
  hip_vae::pack_weight(wt.data(), cc.n, k, pack_as, packed.data());
  DeviceBuffer dx = upload(x), dw = upload(packed), db = upload(bias), dout = upload(res);  // out poisoned with the residual

  // The fp64 replay, and the exact TF32 one for the products without lo x lo.
  const bool measure = cc.products == Products::kTf32Drop;
  std::vector<double> ref(res.size(), std::nan("")), floor(res.size(), 0.0), exact(measure ? res.size() : 0);
  for (std::int64_t r = 0; r < m; ++r) {
    const int oy = static_cast<int>(r / cc.w), ox = static_cast<int>(r % cc.w);
    for (int n = 0; n < cc.n; ++n) {
      double sum = 0.0, mag = 0.0, sum_exact = 0.0;
      for (int q = 0; q < k; ++q) {
        const int tap = q / cc.c, ch = q % cc.c;
        int iy = oy, ix = ox;
        if (cc.taps == 9) {
          iy += tap / 3 - 1;
          ix += tap % 3 - 1;
        }
        if (iy < 0 || ix < 0 || iy >= cc.h || ix >= cc.w) continue;
        if (cc.upsample) {
          iy /= 2;
          ix /= 2;
        }
        const float av = cc.taps == 1 ? x[static_cast<std::size_t>(r) * static_cast<std::size_t>(cc.c) + static_cast<std::size_t>(ch)]
                                      : x[(static_cast<std::size_t>(iy) * static_cast<std::size_t>(in_w) + static_cast<std::size_t>(ix)) *
                                              static_cast<std::size_t>(cc.c) + static_cast<std::size_t>(ch)];
        const float bv = wt[static_cast<std::size_t>(n) * static_cast<std::size_t>(k) + static_cast<std::size_t>(q)];
        const double p = product(av, bv, cc.products);
        sum += p;
        mag += std::abs(p);
        if (measure) sum_exact += product(av, bv, Products::kTf32);
      }
      const std::size_t i = static_cast<std::size_t>(r * cc.ldo + n);
      const auto finish = [&](double s) {
        float y = static_cast<float>(s);
        if (cc.bias) y = y + bias[static_cast<std::size_t>(n)];
        if (cc.residual) y = y + res[i];
        return static_cast<double>(y);
      };
      ref[i] = finish(sum);
      if (measure) exact[i] = finish(sum_exact);
      floor[i] = mag + (cc.bias ? std::abs(bias[static_cast<std::size_t>(n)]) : 0.0) + (cc.residual ? std::abs(res[i]) : 0.0);
    }
  }
  hip_vae::ConvArgs a;
  a.x = dx.as<float>();
  a.w = dw.as<std::uint16_t>();
  a.bias = cc.bias ? db.as<float>() : nullptr;
  a.residual = cc.residual ? dout.as<float>() : nullptr;
  a.out = dout.as<float>();
  a.ldo = cc.ldo;
  a.h = cc.h;
  a.w_out = cc.w;
  a.c = cc.c;
  a.n = cc.n;
  a.taps = cc.taps;
  a.upsample = cc.upsample;
  hip_vae::conv(a, cc.products, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const auto all = download<float>(dout, res.size());
  std::vector<float> got;
  std::vector<double> want, fl, want_exact;
  std::size_t clobbered = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (std::isnan(ref[i])) {  // columns past n keep the poison, bit for bit
      clobbered += std::bit_cast<std::uint32_t>(all[i]) != std::bit_cast<std::uint32_t>(res[i]);
      continue;
    }
    got.push_back(all[i]);
    want.push_back(ref[i]);
    fl.push_back(floor[i]);
    if (measure) want_exact.push_back(exact[i]);
  }
  const std::string name = std::string(cc.name) + " [" + hip_vae::products_name(cc.products) + "]";
  report(name, got, want, fl, {2e-6, "fp32 accumulation"});
  if (measure) report("  the same against the exact TF32 products", got, want_exact, fl, {2e-6, "fp32 accumulation"}, false);
  if (clobbered != 0) {
    ++failures;
    std::printf("  %-52s FAILED  %zu values outside the output columns were written\n", cc.name, clobbered);
  }
}

void check_convs(hipStream_t stream) {
  const ConvCase cases[] = {
      // name, taps, upsample, h, w, c, n, ldo, bias, residual, products
      {"conv3x3 bias+residual, tails", 9, false, 13, 21, 32, 96, 100, true, true, Products::kTf32},
      {"conv3x3 upsample, n = 136", 9, true, 14, 18, 48, 136, 140, true, false, Products::kTf32},
      {"conv3x3 c = 16, n = 40", 9, false, 7, 33, 16, 40, 44, true, true, Products::kTf32},
      {"conv3x3 c = 384 (K 3456), n = 48", 9, false, 6, 7, 384, 48, 48, true, true, Products::kTf32},
      {"conv1x1 n = 32", 1, false, 10, 15, 32, 32, 32, true, false, Products::kTf32},
      {"conv1x1 shortcut c = 96, n = 50", 1, false, 11, 13, 96, 50, 52, true, false, Products::kTf32},
      {"conv3x3 bias+residual, tails", 9, false, 13, 21, 32, 96, 100, true, true, Products::kTf32Drop},
      {"conv3x3 upsample, n = 136", 9, true, 14, 18, 48, 136, 140, true, false, Products::kTf32Drop},
      {"conv3x3 c = 384 (K 3456), n = 48", 9, false, 6, 7, 384, 48, 48, true, true, Products::kTf32Drop},
      {"conv1x1 n = 32", 1, false, 10, 15, 32, 32, 32, true, false, Products::kTf32Drop},
      {"linear 77 x 64 -> 144, bias+residual", 1, false, 1, 77, 64, 144, 148, true, true, Products::kFp32},
      {"linear 40 x 384 -> 1152, bias", 1, false, 1, 40, 384, 1152, 1152, true, false, Products::kFp32},
  };
  for (const ConvCase& c : cases) check_conv(c, stream);
}

void check_conv_narrow(int h, int w, int c, int n, hipStream_t stream) {
  const auto x = random_f32(5, static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * static_cast<std::size_t>(c), -2.0f, 2.0f);
  auto wt = random_f32(6, static_cast<std::size_t>(n) * 9 * static_cast<std::size_t>(c), -0.5f, 0.5f);
  for (float& v : wt) v = hip_vae::host_tf32(v);
  const auto bias = random_f32(7, static_cast<std::size_t>(n), -1.0f, 1.0f);
  const std::size_t m = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  DeviceBuffer dx = upload(x), dw = upload(wt), db = upload(bias), dout(m * static_cast<std::size_t>(n) * 4);
  hip_vae::conv_narrow(dx.as<float>(), dw.as<float>(), db.as<float>(), dout.as<float>(), h, w, c, n, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const auto got = download<float>(dout, m * static_cast<std::size_t>(n));
  std::vector<double> ref(got.size()), floor(got.size());
  for (std::size_t r = 0; r < m; ++r) {
    const int oy = static_cast<int>(r) / w, ox = static_cast<int>(r) % w;
    for (int o = 0; o < n; ++o) {
      double sum = 0.0, mag = 0.0;
      for (int tap = 0; tap < 9; ++tap) {
        const int iy = oy + tap / 3 - 1, ix = ox + tap % 3 - 1;
        if (iy < 0 || ix < 0 || iy >= h || ix >= w) continue;
        const std::size_t xa = (static_cast<std::size_t>(iy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ix)) * static_cast<std::size_t>(c);
        const std::size_t wa = (static_cast<std::size_t>(o) * 9 + static_cast<std::size_t>(tap)) * static_cast<std::size_t>(c);
        for (std::size_t ch = 0; ch < static_cast<std::size_t>(c); ++ch) {
          const double p = static_cast<double>(hip_vae::host_tf32(x[xa + ch])) * wt[wa + ch];
          sum += p;
          mag += std::abs(p);
        }
      }
      const std::size_t i = r * static_cast<std::size_t>(n) + static_cast<std::size_t>(o);
      ref[i] = static_cast<float>(sum) + bias[static_cast<std::size_t>(o)];
      floor[i] = mag + std::abs(bias[static_cast<std::size_t>(o)]);
    }
  }
  report("conv_out 3x3 " + std::to_string(h) + "x" + std::to_string(w) + ", c = " + std::to_string(c) + ", n = " + std::to_string(n), got,
         ref, floor, {2e-6, "fp32 accumulation"});
}

// ---- Attention --------------------------------------------------------------------------------------------------------

// split_pack bit for bit against host_split, the padding zero.
void check_split_pack(int rows, int cols, bool transpose, hipStream_t stream) {
  const int rows_pad = static_cast<int>(hip_vae::round_up(rows, 96)), cols_pad = static_cast<int>(hip_vae::round_up(cols, 16));
  const std::int64_t ld = transpose ? rows + 3 : cols + 5;
  auto src = random_f32(8, static_cast<std::size_t>((transpose ? cols : rows) * ld), -3.0f, 3.0f);
  src[0] = 1e-30f;  // tiny and huge values split exactly too
  src[1] = -7.5e30f;
  src[2] = INFINITY;  // non-finite values are piece 0 alone
  src[3] = -INFINITY;
  src[4] = std::nanf("");
  DeviceBuffer ds = upload(src);
  const std::size_t stride = static_cast<std::size_t>(rows_pad) * static_cast<std::size_t>(cols_pad);
  DeviceBuffer dd = upload(std::vector<std::uint16_t>(3 * stride, 0x7fc1));
  hip_vae::split_pack(ds.as<float>(), ld, transpose, rows, cols, dd.as<std::uint16_t>(), rows_pad, cols_pad, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const auto got = download<std::uint16_t>(dd, 3 * stride);
  std::size_t bad = 0;
  for (int r = 0; r < rows_pad; ++r) {
    for (int c = 0; c < cols_pad; ++c) {
      std::uint16_t want[3] = {0, 0, 0};
      if (r < rows && c < cols) {
        const float v = transpose ? src[static_cast<std::size_t>(c * ld + r)] : src[static_cast<std::size_t>(r * ld + c)];
        hip_vae::host_split(v, Products::kFp32, want);
      }
      for (std::size_t t = 0; t < 3; ++t) bad += got[t * stride + static_cast<std::size_t>(hip_vae::packed_index(r, c, cols_pad))] != want[t];
    }
  }
  // host_split itself: an infinity or NaN is piece 0 alone.
  std::uint16_t inf[3], ninf[3], nan[3];
  hip_vae::host_split(INFINITY, Products::kFp32, inf);
  hip_vae::host_split(-INFINITY, Products::kTf32, ninf);
  hip_vae::host_split(std::nanf(""), Products::kFp32, nan);
  bad += inf[0] != 0x7f80 || inf[1] != 0 || inf[2] != 0 || ninf[0] != 0xff80 || ninf[1] != 0 || nan[0] != 0x7fc0 || nan[1] != 0 || nan[2] != 0;
  const std::string name = "split_pack " + std::to_string(rows) + "x" + std::to_string(cols) + (transpose ? " transposed" : "") +
                           ", with +-Inf and NaN";
  report_exact(name.c_str(), bad, 3 * stride, "pieces");
}

// piece_gemm on split_pack's pieces against the fp64 sum of the exact fp32 products. b_transposed: B comes from a [k, n]
// source (V's layout).
void check_piece_gemm(int m, int n, int k, bool b_transposed, hipStream_t stream) {
  const int k_pad = static_cast<int>(hip_vae::round_up(k, 16));
  const int a_rows = static_cast<int>(hip_vae::round_up(m, hip_vae::kTileM)), b_rows = static_cast<int>(hip_vae::round_up(n, hip_vae::kTileN));
  const auto a = random_f32(9, static_cast<std::size_t>(m) * static_cast<std::size_t>(k), -2.0f, 2.0f);
  const auto b = random_f32(10, static_cast<std::size_t>(n) * static_cast<std::size_t>(k), -1.0f, 1.0f);  // [n, k], or [k, n] when transposed
  const std::int64_t ldo = hip_vae::round_up(n, 4) + 4;
  DeviceBuffer da = upload(a), db = upload(b), out = upload(std::vector<float>(static_cast<std::size_t>(m * ldo), -9.0f));
  DeviceBuffer ap(3ull * static_cast<std::size_t>(a_rows) * static_cast<std::size_t>(k_pad) * 2);
  DeviceBuffer bp(3ull * static_cast<std::size_t>(b_rows) * static_cast<std::size_t>(k_pad) * 2);
  hip_vae::split_pack(da.as<float>(), k, false, m, k, ap.as<std::uint16_t>(), a_rows, k_pad, stream);
  hip_vae::split_pack(db.as<float>(), b_transposed ? n : k, b_transposed, n, k, bp.as<std::uint16_t>(), b_rows, k_pad, stream);
  hip_vae::piece_gemm(ap.as<std::uint16_t>(), static_cast<std::int64_t>(a_rows) * k_pad, bp.as<std::uint16_t>(),
                      static_cast<std::int64_t>(b_rows) * k_pad, k_pad, out.as<float>(), ldo, m, n, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const auto all = download<float>(out, static_cast<std::size_t>(m * ldo));
  std::vector<float> got;
  std::vector<double> ref, floor;
  std::size_t clobbered = 0;
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < ldo; ++c) {
      const float g = all[static_cast<std::size_t>(r * ldo + c)];
      if (c >= n) {
        clobbered += g != -9.0f;
        continue;
      }
      double sum = 0.0, mag = 0.0;
      for (int q = 0; q < k; ++q) {
        const float bv = b_transposed ? b[static_cast<std::size_t>(q) * static_cast<std::size_t>(n) + static_cast<std::size_t>(c)]
                                      : b[static_cast<std::size_t>(c) * static_cast<std::size_t>(k) + static_cast<std::size_t>(q)];
        const double p = static_cast<double>(a[static_cast<std::size_t>(r) * static_cast<std::size_t>(k) + static_cast<std::size_t>(q)]) * bv;
        sum += p;
        mag += std::abs(p);
      }
      got.push_back(g);
      ref.push_back(static_cast<float>(sum));
      floor.push_back(mag);
    }
  }
  const std::string name = "piece_gemm " + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) +
                           (b_transposed ? ", B transposed" : "");
  report(name, got, ref, floor, {2e-6, "fp32 accumulation"});
  if (clobbered != 0) {
    ++failures;
    std::printf("  %-52s FAILED  %zu values outside the output columns were written\n", name.c_str(), clobbered);
  }
}

void check_softmax(hipStream_t stream) {
  constexpr int rows = 37, l = 300, lds = 304, l_pad = 304, rows_pad = 48;
  const float scale = 0.05103104f;
  const auto s = random_f32(20, static_cast<std::size_t>(rows) * lds, -60.0f, 60.0f);
  DeviceBuffer ds = upload(s);
  const std::size_t stride = static_cast<std::size_t>(rows_pad) * l_pad;
  DeviceBuffer dp = upload(std::vector<std::uint16_t>(3 * stride, 0x7fc1));
  hip_vae::softmax_pack(ds.as<float>(), lds, rows, l, scale, dp.as<std::uint16_t>(), static_cast<std::int64_t>(stride), l_pad, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const auto pieces = download<std::uint16_t>(dp, 3 * stride);
  std::vector<float> got;
  std::vector<double> ref, floor;
  std::size_t bad_split = 0, bad_pad = 0;
  for (int r = 0; r < rows; ++r) {
    const float* row = s.data() + static_cast<std::size_t>(r) * lds;
    const float mx = *std::max_element(row, row + l);
    const float shift = mx * scale;
    double sum = 0.0;
    for (int j = 0; j < l; ++j) sum += std::exp(static_cast<double>(row[j] * scale - shift));
    for (int j = 0; j < l_pad; ++j) {
      std::uint16_t p[3];
      double v = 0.0;
      for (std::size_t t = 0; t < 3; ++t) {
        p[t] = pieces[t * stride + static_cast<std::size_t>(hip_vae::packed_index(r, j, l_pad))];
        v += hip_vae::host_bf16_value(p[t]);
      }
      if (j >= l) {
        bad_pad += p[0] != 0 || p[1] != 0 || p[2] != 0;
        continue;
      }
      // The pieces must be host_split's of one fp32 value.
      std::uint16_t want[3];
      hip_vae::host_split(static_cast<float>(v), Products::kFp32, want);
      bad_split += static_cast<double>(static_cast<float>(v)) != v || want[0] != p[0] || want[1] != p[1] || want[2] != p[2];
      got.push_back(static_cast<float>(v));
      ref.push_back(std::exp(static_cast<double>(row[j] * scale - shift)) / sum);
      floor.push_back(1e-30);
    }
  }
  // exp, the sum and the division: a few ulps of the probability.
  report("softmax_pack 37 x 300", got, ref, floor, {3e-6, "fp32 exp, sum and division"});
  report_exact("softmax_pack pieces split one fp32 value", bad_split, got.size(), "values");
  report_exact("softmax_pack columns past l are 0", bad_pad, static_cast<std::size_t>(rows) * (l_pad - l), "values");
}

// ---- Exact-structure checks: sums exact in fp32 in any order, so the product modes differ bit for bit ------------------

// A random value m * 2^-10 with an 11-bit significand m in [1024, 2047] (a TF32 value in [1, 2)), either sign.
float tf32_value(std::mt19937_64& rng) {
  const std::uint64_t z = rng();
  const float v = std::ldexp(static_cast<float>(1024 + (z & 1023)), -10);
  return (z >> 10) & 1 ? -v : v;
}
// A random fp32 value with a full 24-bit significand in [1, 2) * 2^[-2, 2], either sign.
float full_value(std::mt19937_64& rng) {
  const std::uint64_t z = rng();
  const float m = std::ldexp(static_cast<float>((1u << 23) | static_cast<std::uint32_t>(z & 0x7fffffu) | 1u), -23);
  const float v = std::ldexp(m, static_cast<int>((z >> 24) % 5) - 2);
  return (z >> 30) & 1 ? -v : v;
}
float power_of_two(std::mt19937_64& rng) {
  const std::uint64_t z = rng();
  const float v = std::ldexp(1.0f, static_cast<int>(z % 7) - 3);
  return (z >> 8) & 1 ? -v : v;
}

// One K step (c = 16) of a 1x1 convolution with M and N tails and no bias, so every output is a sum of at most 3 nonzero
// products of TF32 values in [1, 2): multiples of 2^-20 below 2^4 whose partial sums stay below 2^4 even piece by piece
// (3 * (2 + 2^-8)^2 < 16), so fp32 sums them exactly in any order. kTf32 must give the exact TF32 products' sums, kTf32Drop
// the sums without lo x lo, and the two must differ.
void check_conv_exact(hipStream_t stream) {
  constexpr int h = 5, w = 7, c = 16, n = 50, ldo = 52;
  constexpr int m = h * w;
  std::mt19937_64 rng(40);
  std::vector<float> x(static_cast<std::size_t>(m) * c), wt(static_cast<std::size_t>(n) * c, 0.0f);
  for (float& v : x) v = tf32_value(rng);
  for (int o = 0; o < n; ++o) {
    for (int t = 0; t < 3; ++t) wt[static_cast<std::size_t>(o * c) + (static_cast<std::size_t>(o) * 5 + static_cast<std::size_t>(t) * 3) % c] = tf32_value(rng);
  }
  std::vector<std::uint16_t> packed(hip_vae::packed_weight_elems(n, c, Products::kTf32));
  hip_vae::pack_weight(wt.data(), n, c, Products::kTf32, packed.data());
  DeviceBuffer dx = upload(x), dw = upload(packed), dout(static_cast<std::size_t>(m) * ldo * 4);
  std::vector<float> exact(static_cast<std::size_t>(m) * n), dropped(exact.size());
  for (int r = 0; r < m; ++r) {
    for (int o = 0; o < n; ++o) {
      double se = 0.0, sd = 0.0;
      for (int k = 0; k < c; ++k) {
        const float a = x[static_cast<std::size_t>(r * c + k)], b = wt[static_cast<std::size_t>(o * c + k)];
        se += product(a, b, Products::kTf32);
        sd += product(a, b, Products::kTf32Drop);
      }
      exact[static_cast<std::size_t>(r * n + o)] = static_cast<float>(se);
      dropped[static_cast<std::size_t>(r * n + o)] = static_cast<float>(sd);
    }
  }
  std::size_t lo_lo = 0;  // outputs whose lo x lo products do not cancel
  for (std::size_t i = 0; i < exact.size(); ++i) lo_lo += exact[i] != dropped[i];
  const auto run = [&](Products p) {
    hip_vae::ConvArgs a;
    a.x = dx.as<float>();
    a.w = dw.as<std::uint16_t>();
    a.out = dout.as<float>();
    a.ldo = ldo;
    a.h = h;
    a.w_out = w;
    a.c = c;
    a.n = n;
    a.taps = 1;
    hip_vae::conv(a, p, stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    const auto all = download<float>(dout, static_cast<std::size_t>(m) * ldo);
    std::vector<float> got(exact.size());
    for (int r = 0; r < m; ++r) {
      for (int o = 0; o < n; ++o) got[static_cast<std::size_t>(r * n + o)] = all[static_cast<std::size_t>(r * ldo + o)];
    }
    return got;
  };
  const auto differ = [](const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]);
    return d;
  };
  const std::vector<float> g4 = run(Products::kTf32), g3 = run(Products::kTf32Drop);
  report_exact("exact conv1x1 K = 16 [TF32, 4 products] = exact TF32 sums", differ(g4, exact), exact.size(), "values");
  report_exact("exact conv1x1 K = 16 [TF32, 3 products] = sums without lo x lo", differ(g3, dropped), exact.size(), "values");
  const std::size_t d34 = differ(g4, g3);
  if (d34 == 0 || lo_lo == 0) ++failures;
  std::printf("  %-52s %s %zu of %zu values differ between the two modes (%zu in the replays)\n",
              "exact conv1x1 K = 16: the modes are told apart", d34 != 0 && lo_lo != 0 ? "ok     " : "FAILED ", d34, exact.size(), lo_lo);
}

// The fp32 linear (9 products) and piece_gemm (i + j <= 2) on one K step (K = 16) where each output has at most one nonzero
// product, a full 24-bit value times a power of two: exact in fp32, so the output must be that product bit for bit. With
// the full value in A this needs the products of A's piece 2 (a2 * b0), with it in B those of B's piece 2 (a0 * b2).
void check_full_products(bool full_in_a, hipStream_t stream) {
  constexpr int m = 40, n = 100, k = 16;
  std::mt19937_64 rng(full_in_a ? 41 : 42);
  std::vector<float> a(static_cast<std::size_t>(m) * k, 0.0f), b(static_cast<std::size_t>(n) * k, 0.0f);
  for (int r = 0; r < m; ++r) a[static_cast<std::size_t>(r * k + r % k)] = full_in_a ? full_value(rng) : power_of_two(rng);
  for (int c = 0; c < n; ++c) b[static_cast<std::size_t>(c * k + c % k)] = full_in_a ? power_of_two(rng) : full_value(rng);
  std::vector<float> want(static_cast<std::size_t>(m) * n);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      want[static_cast<std::size_t>(r * n + c)] = r % k == c % k ? a[static_cast<std::size_t>(r * k + r % k)] * b[static_cast<std::size_t>(c * k + c % k)] : 0.0f;
    }
  }
  const auto bad_of = [&](const std::vector<float>& all, std::int64_t ldo) {
    std::size_t bad = 0;
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) bad += all[static_cast<std::size_t>(r * ldo + c)] != want[static_cast<std::size_t>(r * n + c)];
    }
    return bad;
  };
  const std::string which = full_in_a ? " (24-bit A, power-of-two B)" : " (power-of-two A, 24-bit B)";
  {  // the linear: a 1x1 convolution over m rows of c = 16 with fp32 products
    std::vector<std::uint16_t> packed(hip_vae::packed_weight_elems(n, k, Products::kFp32));
    hip_vae::pack_weight(b.data(), n, k, Products::kFp32, packed.data());
    DeviceBuffer dx = upload(a), dw = upload(packed), dout(static_cast<std::size_t>(m) * n * 4);
    hip_vae::ConvArgs ca;
    ca.x = dx.as<float>();
    ca.w = dw.as<std::uint16_t>();
    ca.out = dout.as<float>();
    ca.ldo = n;
    ca.h = 1;
    ca.w_out = m;
    ca.c = k;
    ca.n = n;
    ca.taps = 1;
    hip_vae::conv(ca, Products::kFp32, stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    report_exact(("exact linear K = 16 [fp32, 9 products]" + which).c_str(), bad_of(download<float>(dout, static_cast<std::size_t>(m) * n), n),
                 want.size(), "values");
  }
  {
    const int a_rows = static_cast<int>(hip_vae::round_up(m, hip_vae::kTileM)), b_rows = static_cast<int>(hip_vae::round_up(n, hip_vae::kTileN));
    DeviceBuffer da = upload(a), db = upload(b), out(static_cast<std::size_t>(m) * n * 4);
    DeviceBuffer ap(3ull * static_cast<std::size_t>(a_rows) * k * 2), bp(3ull * static_cast<std::size_t>(b_rows) * k * 2);
    hip_vae::split_pack(da.as<float>(), k, false, m, k, ap.as<std::uint16_t>(), a_rows, k, stream);
    hip_vae::split_pack(db.as<float>(), k, false, n, k, bp.as<std::uint16_t>(), b_rows, k, stream);
    hip_vae::piece_gemm(ap.as<std::uint16_t>(), static_cast<std::int64_t>(a_rows) * k, bp.as<std::uint16_t>(),
                        static_cast<std::int64_t>(b_rows) * k, k, out.as<float>(), n, m, n, stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    report_exact(("exact piece_gemm K = 16" + which).c_str(), bad_of(download<float>(out, static_cast<std::size_t>(m) * n), n), want.size(),
                 "values");
  }
}

// An infinity or NaN in an operand: the outputs that use it are non-finite (+-Inf where every piece of the other operand
// is nonzero with its sign, as the TF32 or fp32 product), the others finite and right.
void check_nonfinite(hipStream_t stream) {
  // 1 + 2^-9 (+ 2^-18): TF32 (fp32) values whose 2 (3) pieces are all positive.
  const float b_tf32 = 1.0f + 0x1p-9f, b_fp32 = 1.0f + 0x1p-9f + 0x1p-18f;
  std::size_t bad = 0, total = 0;
  const auto classify = [&](float got, double want) {  // NaN to NaN, +-Inf to the same infinity
    ++total;
    if (std::isnan(want)) bad += !std::isnan(got);
    else bad += static_cast<double>(got) != want;
  };
  {  // 1x1 TF32 convolution, c = 16: x[0][0] = +Inf, x[1][3] = NaN; weights of channel 0 = +-b_tf32, or 0 for every 7th
    constexpr int m = 35, c = 16, n = 50, ldo = 52;
    auto x = random_f32(43, static_cast<std::size_t>(m) * c, -2.0f, 2.0f);
    auto wt = random_f32(44, static_cast<std::size_t>(n) * c, -0.5f, 0.5f);
    x[0] = INFINITY;
    x[c + 3] = std::nanf("");
    for (int o = 0; o < n; ++o) wt[static_cast<std::size_t>(o * c)] = o % 7 == 0 ? 0.0f : (o % 2 ? -b_tf32 : b_tf32);
    std::vector<std::uint16_t> packed(hip_vae::packed_weight_elems(n, c, Products::kTf32));
    hip_vae::pack_weight(wt.data(), n, c, Products::kTf32, packed.data());
    DeviceBuffer dx = upload(x), dw = upload(packed), dout(static_cast<std::size_t>(m) * ldo * 4);
    hip_vae::ConvArgs a;
    a.x = dx.as<float>();
    a.w = dw.as<std::uint16_t>();
    a.out = dout.as<float>();
    a.ldo = ldo;
    a.h = 5;
    a.w_out = 7;
    a.c = c;
    a.n = n;
    a.taps = 1;
    hip_vae::conv(a, Products::kTf32, stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    const auto got = download<float>(dout, static_cast<std::size_t>(m) * ldo);
    std::vector<float> fin;
    std::vector<double> ref, floor;
    for (int r = 0; r < m; ++r) {
      for (int o = 0; o < n; ++o) {
        double sum = 0.0, mag = 0.0;
        for (int k = 0; k < c; ++k) {
          const double p = product(x[static_cast<std::size_t>(r * c + k)], wt[static_cast<std::size_t>(o * c + k)], Products::kTf32);
          sum += p;
          mag += std::abs(p);
        }
        const float g = got[static_cast<std::size_t>(r * ldo + o)];
        if (!std::isfinite(sum)) {
          classify(g, sum);
        } else {
          fin.push_back(g);
          ref.push_back(static_cast<float>(sum));
          floor.push_back(mag);
        }
      }
    }
    report("non-finite conv1x1: the finite outputs", fin, ref, floor, {2e-6, "fp32 accumulation"});
  }
  {  // piece_gemm, K = 48: A[0][5] = +Inf, A[1][7] = NaN; B[:, 5] = +-b_fp32, or 0 for every 10th
    constexpr int m = 40, n = 100, k = 48;
    auto a = random_f32(45, static_cast<std::size_t>(m) * k, -2.0f, 2.0f);
    auto b = random_f32(46, static_cast<std::size_t>(n) * k, -1.0f, 1.0f);
    a[5] = INFINITY;
    a[k + 7] = std::nanf("");
    for (int c = 0; c < n; ++c) b[static_cast<std::size_t>(c * k + 5)] = c % 10 == 0 ? 0.0f : (c % 2 ? -b_fp32 : b_fp32);
    const int a_rows = static_cast<int>(hip_vae::round_up(m, hip_vae::kTileM)), b_rows = static_cast<int>(hip_vae::round_up(n, hip_vae::kTileN));
    DeviceBuffer da = upload(a), db = upload(b), out(static_cast<std::size_t>(m) * n * 4);
    DeviceBuffer ap(3ull * static_cast<std::size_t>(a_rows) * k * 2), bp(3ull * static_cast<std::size_t>(b_rows) * k * 2);
    hip_vae::split_pack(da.as<float>(), k, false, m, k, ap.as<std::uint16_t>(), a_rows, k, stream);
    hip_vae::split_pack(db.as<float>(), k, false, n, k, bp.as<std::uint16_t>(), b_rows, k, stream);
    hip_vae::piece_gemm(ap.as<std::uint16_t>(), static_cast<std::int64_t>(a_rows) * k, bp.as<std::uint16_t>(),
                        static_cast<std::int64_t>(b_rows) * k, k, out.as<float>(), n, m, n, stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    const auto got = download<float>(out, static_cast<std::size_t>(m) * n);
    std::vector<float> fin;
    std::vector<double> ref, floor;
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        double sum = 0.0, mag = 0.0;
        for (int q = 0; q < k; ++q) {
          const double p = static_cast<double>(a[static_cast<std::size_t>(r * k + q)]) * b[static_cast<std::size_t>(c * k + q)];
          sum += p;
          mag += std::abs(p);
        }
        const float g = got[static_cast<std::size_t>(r * n + c)];
        if (!std::isfinite(sum)) {
          classify(g, sum);
        } else {
          fin.push_back(g);
          ref.push_back(static_cast<float>(sum));
          floor.push_back(mag);
        }
      }
    }
    report("non-finite piece_gemm: the finite outputs", fin, ref, floor, {2e-6, "fp32 accumulation"});
  }
  report_exact("non-finite conv1x1 and piece_gemm: +-Inf and NaN outputs", bad, total, "values");
}

// ---- GroupNorm --------------------------------------------------------------------------------------------------------

// The exact statistics on the host: the same fixed-point roundings, the sums in unsigned __int128 and a 192-bit pair.
__extension__ typedef unsigned __int128 u128;

struct HostGroup {
  u128 s = 0;               // two's complement
  u128 q_lo = 0, q_hi = 0;  // q = q_hi * 2^128 + q_lo
  bool bad = false;
};

void host_accumulate(HostGroup& g, float v) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
  const int e = static_cast<int>((bits >> 23) & 0xffu);
  if (e == 0) return;
  if (e > 159) {
    g.bad = true;
    return;
  }
  const std::uint64_t mant = (bits & 0x7fffffu) | 0x800000u;
  const int sh = e - 86;
  u128 ax = 0;
  if (sh >= 0) {
    ax = static_cast<u128>(mant) << sh;
  } else {
    const int k = -sh;
    ax = k > 24 ? 0 : (mant + (1ull << (k - 1))) >> k;
  }
  g.s = (bits >> 31) ? g.s - ax : g.s + ax;
  const u128 m2 = static_cast<u128>(mant) * mant;
  const int sh2 = 2 * e - 204;
  u128 lo = 0, hi = 0;
  if (sh2 >= 0) {
    if (sh2 < 128) {
      lo = m2 << sh2;
      hi = sh2 == 0 ? 0 : m2 >> (128 - sh2);
    } else {
      hi = m2 << (sh2 - 128);
    }
  } else if (-sh2 <= 48) {
    const int k = -sh2;
    lo = (m2 + (static_cast<u128>(1) << (k - 1))) >> k;
  }
  const u128 old = g.q_lo;
  g.q_lo += lo;
  g.q_hi += hi + (g.q_lo < old ? 1 : 0);
}

void host_stats(const HostGroup& g, double inv_count, double eps, float& mean_f, float& rstd_f) {
  if (g.bad) {
    mean_f = rstd_f = std::nanf("");
    return;
  }
  const bool neg = (g.s >> 127) != 0;
  const u128 mag = neg ? ~g.s + 1 : g.s;
  double sum = (static_cast<double>(static_cast<std::uint64_t>(mag >> 64)) * 0x1p64 + static_cast<double>(static_cast<std::uint64_t>(mag))) * 0x1p-64;
  if (neg) sum = -sum;
  const double sum_sq = (static_cast<double>(static_cast<std::uint64_t>(g.q_hi)) * 0x1p128 +
                         static_cast<double>(static_cast<std::uint64_t>(g.q_lo >> 64)) * 0x1p64 + static_cast<double>(static_cast<std::uint64_t>(g.q_lo))) *
                        0x1p-96;
  const double mean = sum * inv_count;
  const double var = std::fmax(std::fma(sum_sq, inv_count, -(mean * mean)), 0.0);  // the engine's DMUL + DFMA
  mean_f = static_cast<float>(mean);
  rstd_f = static_cast<float>(1.0 / std::sqrt(var + eps));
}

void check_group_norm(std::int64_t pixels, int c, bool silu, bool in_place, float offset, bool with_huge, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(pixels) * static_cast<std::size_t>(c);
  std::vector<float> x = random_f32(static_cast<std::uint64_t>(10 + c), total, -3.0f, 3.0f);
  const int cpg = c / 32;
  for (std::size_t i = 0; i < total; ++i) x[i] += offset * static_cast<float>(1 + (i % static_cast<std::size_t>(c)) / static_cast<std::size_t>(cpg));
  if (with_huge) x[static_cast<std::size_t>(5 * c + 2)] = 1e11f;  // past the grids (2^33): that group's statistics are NaN
  x[7] = 0.0f;
  x[8] = 1e-40f;  // subnormal: below the grids
  const auto gamma = random_f32(11, static_cast<std::size_t>(c), 0.5f, 1.5f), beta = random_f32(12, static_cast<std::size_t>(c), -0.5f, 0.5f);
  DeviceBuffer dx = upload(x), dg = upload(gamma), db = upload(beta), dy(total * 4), accum(hip_vae::gn_accum_bytes()), stats(256);
  float* y = in_place ? dx.as<float>() : dy.as<float>();
  hip_vae::group_norm_stats(dx.as<float>(), pixels, c, 1e-6f, accum.data(), stats.as<float>(), stream);
  hip_vae::group_norm_apply(dx.as<float>(), y, pixels, c, stats.as<float>(), dg.as<float>(), db.as<float>(), silu, stream);
  check(hipStreamSynchronize(stream), "hipStreamSynchronize");
  const auto got_stats = download<float>(stats, 64);
  const auto got = download<float>(in_place ? dx : dy, total);

  std::vector<HostGroup> groups(32);
  for (std::size_t i = 0; i < total; ++i) host_accumulate(groups[(i % static_cast<std::size_t>(c)) / static_cast<std::size_t>(cpg)], x[i]);
  std::vector<float> mean(32), rstd(32);
  std::size_t bad_stats = 0;
  for (std::size_t g = 0; g < 32; ++g) {
    host_stats(groups[g], 1.0 / static_cast<double>(pixels * cpg), static_cast<double>(1e-6f), mean[g], rstd[g]);
    bad_stats += std::bit_cast<std::uint32_t>(mean[g]) != std::bit_cast<std::uint32_t>(got_stats[2 * g]) ||
                 std::bit_cast<std::uint32_t>(rstd[g]) != std::bit_cast<std::uint32_t>(got_stats[2 * g + 1]);
  }
  std::vector<float> g_val;
  std::vector<double> ref, floor;
  std::size_t nan_ok = 0, nan_bad = 0;
  for (std::size_t i = 0; i < total; ++i) {
    const std::size_t ch = i % static_cast<std::size_t>(c), g = ch / static_cast<std::size_t>(cpg);
    if (groups[g].bad) {  // NaN statistics give NaN outputs
      (std::isnan(got[i]) ? nan_ok : nan_bad) += 1;
      continue;
    }
    const float a = rstd[g] * gamma[ch], b = std::fma(-a, mean[g], beta[ch]);
    float v = std::fma(a, x[i], b);
    if (silu) v = v / (1.0f + std::exp(-v));
    g_val.push_back(got[i]);
    ref.push_back(v);
    floor.push_back(std::abs(a * x[i]) + std::abs(a * mean[g]));
  }
  const std::string name = "group_norm " + std::to_string(pixels) + " px, c = " + std::to_string(c) + (silu ? " silu" : "") +
                           (in_place ? " in place" : "") + (with_huge ? ", a group out of range" : "");
  report_exact((name + ": statistics").c_str(), bad_stats, 64, "values");
  report(name, g_val, ref, floor, {2e-6, "the device's exp"});
  if (with_huge) report_exact((name + ": that group's outputs are NaN").c_str(), nan_bad + (nan_ok == 0 ? 1 : 0), nan_ok + nan_bad, "outputs");
}

// ---- Bit-exact elementwise ops ----------------------------------------------------------------------------------------

void check_exact(hipStream_t stream) {
  {
    constexpr int h = 3, w = 5;
    const auto z = random_f32(31, 128ull * h * w, -4.0f, 4.0f);
    const auto var = random_f32(32, 128, 0.1f, 9.0f), mean = random_f32(33, 128, -1.0f, 1.0f);
    std::vector<float> scale(128);
    for (std::size_t i = 0; i < 128; ++i) scale[i] = std::sqrt(var[i] + 1e-4f);
    DeviceBuffer dz = upload(z), ds = upload(scale), dm = upload(mean), y(4ull * h * w * 32 * 4);
    hip_vae::latent_unpack(dz.as<float>(), ds.as<float>(), dm.as<float>(), y.as<float>(), h, w, stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    const auto got = download<float>(y, 4ull * h * w * 32);
    std::size_t bad = 0;
    for (std::size_t oy = 0; oy < 2 * h; ++oy) {
      for (std::size_t ox = 0; ox < 2 * w; ++ox) {
        for (std::size_t c = 0; c < 32; ++c) {
          const std::size_t ch = 4 * c + 2 * (oy & 1) + (ox & 1);
          const float v = z[(ch * h + oy / 2) * w + ox / 2];
          const float scaled = v * scale[ch];
          const float want = scaled + mean[ch];
          bad += std::bit_cast<std::uint32_t>(got[(oy * 2 * w + ox) * 32 + c]) != std::bit_cast<std::uint32_t>(want);
        }
      }
    }
    report_exact("latent unpack", bad, got.size(), "values");
  }
  {
    std::vector<float> x = random_f32(34, 1001, -1.3f, 1.3f);
    const float special[] = {std::nanf(""), -1.0f, 1.0f, 0.0f, -0.0f, 1.0f / 255.0f, 0.5f / 255.0f * 2.0f - 1.0f,
                             1.5f / 255.0f * 2.0f - 1.0f, 2.5f / 255.0f * 2.0f - 1.0f, INFINITY, -INFINITY};
    std::copy(std::begin(special), std::end(special), x.begin());
    DeviceBuffer dx = upload(x), rgb(1001);
    hip_vae::rgb8(dx.as<float>(), rgb.as<std::uint8_t>(), static_cast<std::int64_t>(x.size()), stream);
    check(hipStreamSynchronize(stream), "hipStreamSynchronize");
    const auto got = download<std::uint8_t>(rgb, x.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      std::uint8_t want = 0;
      if (!std::isnan(x[i])) {
        float v = std::min(std::max(x[i], -1.0f), 1.0f);
        const float half = v * 0.5f;
        v = std::min(std::max(half + 0.5f, 0.0f), 1.0f);
        want = static_cast<std::uint8_t>(std::nearbyint(v * 255.0f));
      }
      bad += got[i] != want;
    }
    report_exact("rgb8", bad, got.size(), "bytes");
  }
}

void run_kat(hipStream_t stream) {
  std::printf("known-answer checks against host replays:\n");
  check_convs(stream);
  check_conv_narrow(9, 30, 96, 3, stream);
  check_conv_narrow(5, 7, 20, 4, stream);
  check_conv_exact(stream);
  check_full_products(true, stream);
  check_full_products(false, stream);
  check_nonfinite(stream);
  check_split_pack(70, 45, false, stream);
  check_split_pack(45, 70, true, stream);
  check_piece_gemm(70, 150, 40, false, stream);
  check_piece_gemm(33, 100, 384, false, stream);
  check_piece_gemm(64, 40, 333, true, stream);
  check_softmax(stream);
  check_group_norm(1000, 96, true, false, 0.0f, false, stream);
  check_group_norm(777, 384, false, true, 20.0f, false, stream);
  check_group_norm(2050, 192, true, true, 5.0f, false, stream);
  check_group_norm(300, 64, true, false, 0.0f, true, stream);
  check_exact(stream);
}

// ---- The decode -------------------------------------------------------------------------------------------------------

// ---- A host replay of the whole decoder ------------------------------------------------------------------------------

// The decoder op by op in fp32 tensors, each sum in fp64 over the products the device forms (TF32-rounded operands for the
// convolutions, fp32 for the linears), rounded to fp32 once; the GroupNorm statistics with the device's exact method, the
// attention in fp64. With fp32_sums, every sum is instead an fp32 sum in K order: another fp32 accumulation order, whose
// distance to the fp64 replay measures how far such orders alone move the output. Slow (about 7e8 multiply-adds per
// latent pixel): for tiny latent grids.
class HostDecoder {
 public:
  HostDecoder(const fs::path& path, bool fp32_sums) : f_(path), fp32_(fp32_sums) {
    c_ = static_cast<int>(f_.at("decoder.conv_in.weight").shape[0] / 4);
    const int c = c_, mid = 4 * c;
    mean_ = f_.read_f32("bn.running_mean");
    const std::vector<float> var = f_.read_f32("bn.running_var");
    scale_.resize(128);
    for (std::size_t i = 0; i < 128; ++i) scale_[i] = std::sqrt(var[i] + 1e-4f);
    post_quant_ = conv("post_quant_conv", 1);
    conv_in_ = conv("decoder.conv_in", 3);
    for (int i = 0; i < 2; ++i) mid_[i] = resnet("decoder.mid_block.resnets." + std::to_string(i));
    const std::string a = "decoder.mid_block.attentions.0.";
    attn_norm_ = norm(a + "group_norm");
    for (const char* q : {"to_q", "to_k", "to_v"}) {
      const auto wt = f_.read_f32(a + q + ".weight"), bt = f_.read_f32(a + q + ".bias");
      qkv_.w.insert(qkv_.w.end(), wt.begin(), wt.end());
      qkv_.b.insert(qkv_.b.end(), bt.begin(), bt.end());
    }
    qkv_.k = 3 * mid;
    qkv_.c = mid;
    qkv_.r = 1;
    qkv_.tf32 = false;
    out_ = {f_.read_f32(a + "to_out.0.weight"), f_.read_f32(a + "to_out.0.bias"), mid, mid, 1, false};
    for (int b = 0; b < 4; ++b) {
      const std::string p = "decoder.up_blocks." + std::to_string(b);
      for (int r = 0; r < 3; ++r) up_[b][r] = resnet(p + ".resnets." + std::to_string(r));
      if (b < 3) upsampler_[b] = conv(p + ".upsamplers.0.conv", 3);
    }
    norm_out_ = norm("decoder.conv_norm_out");
    conv_out_ = conv("decoder.conv_out", 3);
  }

  // z [128, h, w] -> the decoder's fp32 output [16h, 16w, 3].
  std::vector<float> decode(const std::vector<float>& z, int h, int w) const {
    int ch = 2 * h, cw = 2 * w;
    std::vector<float> x(static_cast<std::size_t>(ch) * cw * 32);
    for (int oy = 0; oy < ch; ++oy) {
      for (int ox = 0; ox < cw; ++ox) {
        for (int c = 0; c < 32; ++c) {
          const int lc = 4 * c + 2 * (oy & 1) + (ox & 1);
          const float v = z[(static_cast<std::size_t>(lc) * h + static_cast<std::size_t>(oy / 2)) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ox / 2)];
          const float scaled = v * scale_[static_cast<std::size_t>(lc)];
          x[(static_cast<std::size_t>(oy) * cw + static_cast<std::size_t>(ox)) * 32 + static_cast<std::size_t>(c)] = scaled + mean_[static_cast<std::size_t>(lc)];
        }
      }
    }
    x = run(post_quant_, x, ch, cw);
    x = run(conv_in_, x, ch, cw);
    x = resnet(mid_[0], x, ch, cw);
    x = attention(x, ch * cw);
    x = resnet(mid_[1], x, ch, cw);
    for (int b = 0; b < 4; ++b) {
      for (int r = 0; r < 3; ++r) x = resnet(up_[b][r], x, ch, cw);
      if (b < 3) {
        ch *= 2;
        cw *= 2;
        x = run(upsampler_[b], x, ch, cw, nullptr, true);
      }
    }
    x = group_norm(norm_out_, x, static_cast<std::int64_t>(ch) * cw, true);
    return run(conv_out_, x, ch, cw);
  }

 private:
  struct Conv {
    std::vector<float> w;  // [k, r r c] KRSC
    std::vector<float> b;
    int k = 0, c = 0, r = 0;
    bool tf32 = true;
  };
  struct Norm {
    std::vector<float> gamma, beta;
  };
  struct Resnet {
    Norm n1, n2;
    Conv c1, c2, sc;
  };

  Conv conv(const std::string& p, int r) const {
    const st::TensorInfo& t = f_.at(p + ".weight");
    Conv cv;
    cv.k = static_cast<int>(t.shape[0]);
    cv.c = static_cast<int>(t.shape[1]);
    cv.r = r;
    const std::vector<float> kcrs = f_.read_f32(p + ".weight");
    const std::size_t taps = static_cast<std::size_t>(r * r), cs = static_cast<std::size_t>(cv.c);
    cv.w.resize(kcrs.size());
    for (std::size_t k = 0; k < static_cast<std::size_t>(cv.k); ++k) {
      for (std::size_t c = 0; c < cs; ++c) {
        for (std::size_t q = 0; q < taps; ++q) cv.w[(k * taps + q) * cs + c] = hip_vae::host_tf32(kcrs[(k * cs + c) * taps + q]);
      }
    }
    cv.b = f_.read_f32(p + ".bias");
    return cv;
  }
  Norm norm(const std::string& p) const { return {f_.read_f32(p + ".weight"), f_.read_f32(p + ".bias")}; }
  Resnet resnet(const std::string& p) const {
    Resnet r{norm(p + ".norm1"), norm(p + ".norm2"), conv(p + ".conv1", 3), conv(p + ".conv2", 3), {}};
    if (f_.find(p + ".conv_shortcut.weight") != nullptr) r.sc = conv(p + ".conv_shortcut", 1);
    return r;
  }

  // y = (conv(x) + b) + residual over an h x w output; with `upsample` x is h/2 x w/2 read through a nearest 2x upsample.
  // A linear is a 1x1 convolution over h * w rows.
  std::vector<float> run(const Conv& cv, std::vector<float> x, int h, int w, const std::vector<float>* residual = nullptr,
                         bool upsample = false) const {
    if (cv.tf32) {
      for (float& v : x) v = hip_vae::host_tf32(v);
    }
    const int iw = upsample ? w / 2 : w, taps = cv.r * cv.r, kk = taps * cv.c;
    std::vector<float> y(static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * static_cast<std::size_t>(cv.k));
    for (int oy = 0; oy < h; ++oy) {
      for (int ox = 0; ox < w; ++ox) {
        const std::size_t o = static_cast<std::size_t>(oy) * static_cast<std::size_t>(w) + static_cast<std::size_t>(ox);
        for (int n = 0; n < cv.k; ++n) {
          double sum = 0.0;
          float sum32 = 0.0f;
          for (int tap = 0; tap < taps; ++tap) {
            int iy = oy, ix = ox;
            if (taps == 9) {
              iy += tap / 3 - 1;
              ix += tap % 3 - 1;
            }
            if (iy < 0 || ix < 0 || iy >= h || ix >= w) continue;
            if (upsample) {
              iy /= 2;
              ix /= 2;
            }
            const float* xp = x.data() + (static_cast<std::size_t>(iy) * static_cast<std::size_t>(iw) + static_cast<std::size_t>(ix)) * static_cast<std::size_t>(cv.c);
            const float* wp = cv.w.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(kk) + static_cast<std::size_t>(tap * cv.c);
            if (fp32_) {
              for (int c = 0; c < cv.c; ++c) sum32 = sum32 + xp[c] * wp[c];
            } else {
              for (int c = 0; c < cv.c; ++c) sum += static_cast<double>(xp[c]) * wp[c];
            }
          }
          float v = (fp32_ ? sum32 : static_cast<float>(sum)) + cv.b[static_cast<std::size_t>(n)];
          const std::size_t i = o * static_cast<std::size_t>(cv.k) + static_cast<std::size_t>(n);
          if (residual != nullptr) v = v + (*residual)[i];
          y[i] = v;
        }
      }
    }
    return y;
  }
  static std::vector<float> group_norm(const Norm& nm, const std::vector<float>& x, std::int64_t px, bool silu) {
    const std::size_t c = nm.gamma.size(), cpg = c / 32;
    std::vector<HostGroup> groups(32);
    for (std::size_t i = 0; i < x.size(); ++i) host_accumulate(groups[(i % c) / cpg], x[i]);
    float mean[32], rstd[32];
    for (std::size_t g = 0; g < 32; ++g) host_stats(groups[g], 1.0 / static_cast<double>(px * static_cast<std::int64_t>(cpg)), static_cast<double>(1e-6f), mean[g], rstd[g]);
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      const std::size_t ch = i % c, g = ch / cpg;
      const float a = rstd[g] * nm.gamma[ch], b = std::fma(-a, mean[g], nm.beta[ch]);
      float v = std::fma(a, x[i], b);
      if (silu) v = v / (1.0f + std::exp(-v));
      y[i] = v;
    }
    return y;
  }
  std::vector<float> resnet(const Resnet& r, const std::vector<float>& x, int h, int w) const {
    const std::int64_t px = static_cast<std::int64_t>(h) * w;
    std::vector<float> t = run(r.c1, group_norm(r.n1, x, px, true), h, w);
    t = group_norm(r.n2, t, px, true);
    if (r.sc.k != 0) {
      const std::vector<float> s = run(r.sc, x, h, w);
      return run(r.c2, t, h, w, &s);
    }
    return run(r.c2, t, h, w, &x);
  }
  std::vector<float> attention(const std::vector<float>& x, int len) const {
    const int mid = 4 * c_;
    const std::size_t m = static_cast<std::size_t>(mid);
    const std::vector<float> qkv = run(qkv_, group_norm(attn_norm_, x, len, false), len, 1);
    const float scale = 1.0f / std::sqrt(static_cast<float>(mid));
    std::vector<float> o(static_cast<std::size_t>(len) * m);
    std::vector<float> s(static_cast<std::size_t>(len));
    std::vector<double> p(static_cast<std::size_t>(len));
    for (std::size_t i = 0; i < static_cast<std::size_t>(len); ++i) {
      for (std::size_t j = 0; j < s.size(); ++j) {
        double d = 0.0;
        float d32 = 0.0f;
        for (std::size_t e = 0; e < m; ++e) {
          if (fp32_) d32 = d32 + qkv[i * 3 * m + e] * qkv[j * 3 * m + m + e];
          else d += static_cast<double>(qkv[i * 3 * m + e]) * qkv[j * 3 * m + m + e];
        }
        s[j] = fp32_ ? d32 : static_cast<float>(d);
      }
      const float shift = *std::max_element(s.begin(), s.end()) * scale;
      double sum = 0.0;
      for (std::size_t j = 0; j < s.size(); ++j) sum += p[j] = std::exp(static_cast<double>(s[j] * scale - shift));
      for (std::size_t e = 0; e < m; ++e) {
        double d = 0.0;
        float d32 = 0.0f;
        for (std::size_t j = 0; j < s.size(); ++j) {
          if (fp32_) d32 = d32 + static_cast<float>(p[j] / sum) * qkv[j * 3 * m + 2 * m + e];
          else d += p[j] / sum * qkv[j * 3 * m + 2 * m + e];
        }
        o[i * m + e] = fp32_ ? d32 : static_cast<float>(d);
      }
    }
    return run(out_, o, len, 1, &x);
  }

  st::File f_;
  bool fp32_;
  int c_ = 0;
  std::vector<float> mean_, scale_;
  Conv post_quant_, conv_in_, qkv_, out_, conv_out_;
  Conv upsampler_[3];
  Norm attn_norm_, norm_out_;
  Resnet mid_[2], up_[4][3];
};

std::uint8_t host_u8(float v) {  // the pipeline's rounding (rgb8)
  if (std::isnan(v)) return 0;
  v = std::min(std::max(v, -1.0f), 1.0f);
  const float half = v * 0.5f;
  v = std::min(std::max(half + 0.5f, 0.0f), 1.0f);
  return static_cast<std::uint8_t>(std::nearbyint(v * 255.0f));
}

struct Options {
  int device = -1;
  bool kat = true, decode = true, profile = false, nonsquare = true, golden_cases = true;
  fs::path vae = "/workspace/Models/MOSS-Image/base/vae/diffusion_pytorch_model.safetensors";
  fs::path golden = "/workspace/Projects/MossInfer/tests/golden";
  std::vector<std::string> cases;
  fs::path out_dir;
  int iters = 1;
  Products conv_products = Products::kTf32;
};

void run_decode(Device& device, VaeDecoder& decoder, const Options& o, const std::string& name) {
  const npy::Array arr = npy::load(o.golden / name / "final_latent.npy");
  std::vector<std::int64_t> shape = arr.shape;
  if (shape.size() == 4 && shape[0] == 1) shape.erase(shape.begin());
  if (shape.size() != 3 || shape[0] != 128) throw std::runtime_error(name + ": final_latent.npy is not [1, 128, h, w]");
  const auto h = static_cast<std::uint32_t>(shape[1]), w = static_cast<std::uint32_t>(shape[2]);
  const std::vector<float> latent = arr.to_vector<float>();
  std::printf("\n%s: decode %ux%u (latents %ux%u), %s, decode allocates %.3f GiB, %.2f GiB free\n", name.c_str(), 16 * w, 16 * h, h,
              w, hip_vae::products_name(o.conv_products), static_cast<double>(decoder.workspace_bytes(h, w)) / (1u << 30),
              static_cast<double>(device.free_memory()) / (1u << 30));
  std::vector<std::uint8_t> rgb;
  std::vector<float> image;
  VaeDecodeStats st;
  for (int i = 0; i < std::max(1, o.iters); ++i) {
    st = decoder.decode(latent.data(), h, w, rgb, i == 0 ? &image : nullptr);
    std::printf("  run %d: %.3f s\n", i, st.seconds);
  }
  if (o.profile) {
    double total = 0.0;
    for (const auto& [op, ms] : st.profile) total += ms;
    std::printf("  device time per op of the last run (%.1f ms in all):\n", total);
    for (const auto& [op, ms] : st.profile) std::printf("    %-52s %9.2f ms %5.1f %%\n", op.c_str(), ms, 100.0 * ms / total);
  }
  std::size_t nans = 0;
  for (const float v : image) nans += std::isnan(v);
  if (!o.out_dir.empty()) {
    const fs::path out = o.out_dir / (name + ".png");
    write_png_rgb(out, rgb, static_cast<int>(16 * w), static_cast<int>(16 * h));
    std::printf("  wrote %s\n", out.string().c_str());
  }
  int gw = 0, gh = 0;
  const fs::path png = o.golden / name / "image.png";
  const std::vector<std::uint8_t> golden = read_png_rgb(png, gw, gh);
  if (static_cast<std::uint32_t>(gw) != 16 * w || static_cast<std::uint32_t>(gh) != 16 * h) {
    throw std::runtime_error(png.string() + " is " + std::to_string(gw) + "x" + std::to_string(gh) + ", not the decode's size");
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
  const double n = static_cast<double>(rgb.size()), mse = se / n, identical = static_cast<double>(same) / n;
  const double psnr = mse == 0.0 ? INFINITY : 10.0 * std::log10(255.0 * 255.0 / mse);
  const bool ok = nans == 0 && max_abs <= 1 && identical >= 0.99;
  if (!ok) ++failures;
  std::printf("  against %s: %s\n    PSNR %.2f dB, max |diff| %d, mean |diff| %.5f, identical bytes %.4f %%, decoder output NaN %zu\n",
              png.string().c_str(), ok ? "ok" : "FAILED (gate: no NaN, max |diff| <= 1, >= 99 % identical)", psnr, max_abs,
              sum_abs / n, 100.0 * identical, nans);
  std::printf("    |diff| 0: %zu, 1: %zu, 2: %zu, >= 3: %zu bytes\n", hist[0], hist[1], hist[2], hist[3]);
}

std::vector<float> crop_latent(const std::vector<float>& z, std::uint32_t h, std::uint32_t w, std::uint32_t y0, std::uint32_t x0,
                               std::uint32_t ch, std::uint32_t cw) {
  std::vector<float> c(128ull * ch * cw);
  for (std::size_t k = 0; k < 128; ++k) {
    for (std::size_t y = 0; y < ch; ++y) {
      std::memcpy(&c[(k * ch + y) * cw], &z[(k * h + y0 + y) * w + x0], cw * 4);
    }
  }
  return c;
}

std::vector<float> load_latent(const fs::path& path, std::uint32_t& h, std::uint32_t& w) {
  const npy::Array arr = npy::load(path);
  std::vector<std::int64_t> shape = arr.shape;
  if (shape.size() == 4 && shape[0] == 1) shape.erase(shape.begin());
  if (shape.size() != 3 || shape[0] != 128) throw std::runtime_error(path.string() + " is not [1, 128, h, w]");
  h = static_cast<std::uint32_t>(shape[1]);
  w = static_cast<std::uint32_t>(shape[2]);
  return arr.to_vector<float>();
}

// Non-square decodes. A 3 x 2 crop of g_fridge_1024's latent (a 32x48 image) through the device and through
// HostDecoder: the device's fp32 output within 3 times the fp32-sum replay's distance to the fp64 replay (what another
// fp32 accumulation order alone moves the output, carried through 30 convolutions and the attention; an indexing error
// moves it by about 1), and its RGB bytes within 1. Then a 64 (rows) x 48 crop
// through the device: no NaN and the right size; its PSNR against the golden image's top-left corner is printed only (a
// crop's decode is not the decode's crop: the attention is global).
void run_nonsquare(VaeDecoder& decoder, const Options& o) {
  std::uint32_t h = 0, w = 0;
  const std::vector<float> z = load_latent(o.golden / "g_fridge_1024" / "final_latent.npy", h, w);
  {
    constexpr std::uint32_t ch = 3, cw = 2;
    const std::vector<float> c = crop_latent(z, h, w, 30, 20, ch, cw);
    std::vector<std::uint8_t> rgb;
    std::vector<float> image;
    decoder.decode(c.data(), ch, cw, rgb, &image);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> ref = HostDecoder(o.vae, false).decode(c, static_cast<int>(ch), static_cast<int>(cw));
    const std::vector<float> ref32 = HostDecoder(o.vae, true).decode(c, static_cast<int>(ch), static_cast<int>(cw));
    const double host_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double floor = 0.0, floor_sum = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double d = std::abs(static_cast<double>(ref32[i]) - ref[i]);
      floor = std::max(floor, d);
      floor_sum += d;
    }
    double worst = 0.0, sum = 0.0;
    int worst_byte = 0;
    std::size_t bytes_off = 0, nans = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      nans += std::isnan(image[i]);
      const double d = std::abs(static_cast<double>(image[i]) - ref[i]);
      worst = std::max(worst, d);
      sum += d;
      const int b = std::abs(static_cast<int>(rgb[i]) - static_cast<int>(host_u8(ref[i])));
      worst_byte = std::max(worst_byte, b);
      bytes_off += b != 0;
    }
    const bool ok = image.size() == ref.size() && rgb.size() == ref.size() && nans == 0 && worst <= 3.0 * floor && worst_byte <= 1;
    if (!ok) ++failures;
    std::printf("\nnon-square: g_fridge_1024 latents [30, 33) x [20, 22) -> 32x48, device against the host replays (%.1f s): %s\n"
                "    device - fp64 replay: max |diff| %.3g (limit 3 x %.3g), mean |diff| %.3g, %zu of %zu bytes differ (max %d), %zu NaN\n"
                "    fp32-sum replay - fp64 replay: max |diff| %.3g, mean |diff| %.3g\n",
                host_s, ok ? "ok" : "FAILED", worst, floor, sum / static_cast<double>(ref.size()), bytes_off, ref.size(), worst_byte, nans,
                floor, floor_sum / static_cast<double>(ref.size()));
  }
  {
    constexpr std::uint32_t ch = 64, cw = 48;
    const std::vector<float> c = crop_latent(z, h, w, 0, 0, ch, cw);
    std::vector<std::uint8_t> rgb;
    std::vector<float> image;
    const VaeDecodeStats st = decoder.decode(c.data(), ch, cw, rgb, &image);
    std::size_t nans = 0;
    for (const float v : image) nans += std::isnan(v);
    const std::size_t want = 16ull * ch * 16 * cw * 3;
    int gw = 0, gh = 0;
    const std::vector<std::uint8_t> golden = read_png_rgb(o.golden / "g_fridge_1024" / "image.png", gw, gh);
    double se = 0.0;
    for (std::size_t y = 0; y < 16 * ch && rgb.size() == want; ++y) {
      for (std::size_t x = 0; x < 16 * cw * 3; ++x) {
        const double d = static_cast<double>(rgb[y * 16 * cw * 3 + x]) - golden[y * static_cast<std::size_t>(gw) * 3 + x];
        se += d * d;
      }
    }
    const bool ok = nans == 0 && rgb.size() == want && image.size() == want;
    if (!ok) ++failures;
    std::printf("non-square: g_fridge_1024 latents [0, 64) x [0, 48) -> 768x1024 in %.3f s: %s (%zu NaN, %zu bytes); "
                "PSNR %.2f dB against the golden image's top-left 768x1024 (not a gate)\n",
                st.seconds, ok ? "ok" : "FAILED", nans, rgb.size(), 10.0 * std::log10(255.0 * 255.0 * static_cast<double>(want) / se));
    if (!o.out_dir.empty()) write_png_rgb(o.out_dir / "g_fridge_1024_crop64x48.png", rgb, static_cast<int>(16 * cw), static_cast<int>(16 * ch));
  }
}

// The size limits: refused before anything runs, with the reason.
void check_limits(Device& device, const VaeDecoder& decoder) {
  const auto expect_refused = [&](const char* what, std::uint32_t h, std::uint32_t w) {
    try {
      (void)VaeDecoder::workspace_bytes(decoder.base_channels(), h, w);
      ++failures;
      std::printf("FAILED: %s was accepted\n", what);
    } catch (const std::exception& e) {
      std::printf("%s is refused: %s\n", what, e.what());
    }
  };
  expect_refused("a 9600x9600 decode", 600, 600);
  expect_refused("a decode of 2^31 x 2^31 latents", 0x80000000u, 0x80000000u);
  expect_refused("a decode of 1 x (2^20 + 1) latents", 1, (1u << 20) + 1);
  // 8192 x 8192 is within the limits: refused by require_fits exactly when the device cannot hold it.
  const std::uint64_t need = decoder.workspace_bytes(512, 512), free = device.free_memory();
  bool refused = false;
  try {
    decoder.require_fits(512, 512);
  } catch (const std::runtime_error& e) {
    refused = true;
    std::printf("an 8192x8192 decode: %s\n", e.what());
  }
  const bool ok = refused == (need > free);
  if (!ok) ++failures;
  std::printf("an 8192x8192 decode needs %.2f GiB with %.2f GiB free: %s %s\n", static_cast<double>(need) / (1u << 30),
              static_cast<double>(free) / (1u << 30), refused ? "refused" : "accepted", ok ? "(ok)" : "FAILED");
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
      if (arg == "--device") o.device = std::stoi(next());
      else if (arg == "--kat-only") o.decode = false;
      else if (arg == "--no-kat") o.kat = false;
      else if (arg == "--vae") o.vae = next();
      else if (arg == "--golden") o.golden = next();
      else if (arg == "--case") o.cases.push_back(next());
      else if (arg == "--no-nonsquare") o.nonsquare = false;
      else if (arg == "--no-golden") o.golden_cases = false;
      else if (arg == "--out-dir") o.out_dir = next();
      else if (arg == "--iters") o.iters = std::stoi(next());
      else if (arg == "--profile") o.profile = true;
      else if (arg == "--conv-products") {
        const std::string v = next();
        if (v != "4" && v != "3") throw std::invalid_argument("--conv-products takes 4 or 3");
        o.conv_products = v == "4" ? Products::kTf32 : Products::kTf32Drop;
      } else {
        throw std::invalid_argument("unknown argument " + arg);
      }
    } catch (const std::exception& e) {
      std::printf("%s\nusage: %s [--device N] [--kat-only | --no-kat] [--vae FILE] [--golden DIR] [--case NAME]... [--no-golden] [--no-nonsquare]\n"
                  "       [--out-dir DIR] [--iters N] [--profile] [--conv-products 4|3]\n",
                  e.what(), argv[0]);
      return 2;
    }
  }
  if (o.cases.empty() && o.golden_cases) o.cases = {"g_fridge_1024", "g_fridge_1456"};
  try {
    bool usable = false;
    for (const DeviceInfo& d : list_devices()) usable = usable || d.unusable_reason.empty();
    if (!usable && o.device < 0) {
      std::printf("no device runs this build (%s)\n", build_target());
      return 2;
    }
    Device device(o.device);
    std::printf("device %d: %s (%s), build %s, %.2f GiB free\n", device.info().index, device.info().name.c_str(),
                device.info().arch.c_str(), build_target(), static_cast<double>(device.free_memory()) / (1u << 30));
    if (o.kat) {
      Stream stream;
      run_kat(stream.get());
    }
    if (o.decode) {
      const auto t0 = std::chrono::steady_clock::now();
      VaeDecoder decoder(device, o.vae);
      decoder.set_conv_products(o.conv_products);
      decoder.set_profile(o.profile);
      std::printf("\nweights: %.1f MB on the device (decoder c = %d), loaded in %.2f s; decode workspace at 1024^2 %.2f GiB, "
                  "1456^2 %.2f GiB, 2048^2 %.2f GiB\n",
                  static_cast<double>(decoder.weight_bytes()) / 1e6, decoder.base_channels(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                  static_cast<double>(decoder.workspace_bytes(64, 64)) / (1u << 30),
                  static_cast<double>(decoder.workspace_bytes(91, 91)) / (1u << 30),
                  static_cast<double>(decoder.workspace_bytes(128, 128)) / (1u << 30));
      check_limits(device, decoder);
      for (const std::string& name : o.cases) run_decode(device, decoder, o, name);
      if (o.nonsquare) run_nonsquare(decoder, o);
    }
    std::printf("\n%s (%d failed checks)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 1;
  }
}
