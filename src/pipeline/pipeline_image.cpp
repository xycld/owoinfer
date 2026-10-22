#include "pipeline/pipeline_image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "util/path.h"

// PNG and JPEG decoding: the vendored stb_image (third_party/stb), static so the tests' own copy does not clash.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 16384  // = kMaxInputSide (static_assert below): stb refuses larger headers itself
#include <stb_image.h>

static_assert(STBI_MAX_DIMENSIONS == mi::pipeline::kMaxInputSide);

namespace mi::pipeline {
namespace {

void check_image(const Image& im, const char* what) {
  if (im.width <= 0 || im.height <= 0 || (im.channels != 1 && im.channels != 3) ||
      im.pixels.size() != static_cast<std::size_t>(im.width) * static_cast<std::size_t>(im.height) *
                              static_cast<std::size_t>(im.channels)) {
    throw std::invalid_argument(std::string(what) + ": the pixel buffer does not match its size and channel count");
  }
}

// PIL Resample.c, 8 bits per channel.
constexpr int kPrecisionBits = 32 - 8 - 2;

enum class Filter { kLanczos, kBicubic };

double sinc_filter(double x) {
  if (x == 0.0) return 1.0;
  x = x * std::numbers::pi;
  return std::sin(x) / x;
}
double lanczos_filter(double x) {
  // truncated sinc
  if (-3.0 <= x && x < 3.0) return sinc_filter(x) * sinc_filter(x / 3);
  return 0.0;
}
double bicubic_filter(double x) {
  // Keys' cubic convolution, a = -0.5
  constexpr double a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

struct Coeffs {
  int ksize = 0;
  std::vector<int> bounds;      // per output: xmin, count
  std::vector<std::int32_t> k;  // [out, ksize], fixed point
};

// precompute_coeffs + normalize_coeffs_8bpc.
Coeffs precompute_coeffs(int in_size, double in0, double in1, int out_size, Filter filter) {
  double filterscale = (in1 - in0) / out_size;
  const double scale = filterscale;
  if (filterscale < 1.0) filterscale = 1.0;
  const double support = (filter == Filter::kLanczos ? 3.0 : 2.0) * filterscale;
  Coeffs c;
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.bounds.resize(static_cast<std::size_t>(out_size) * 2);
  c.k.assign(static_cast<std::size_t>(out_size) * static_cast<std::size_t>(c.ksize), 0);
  std::vector<double> kk(static_cast<std::size_t>(c.ksize));
  for (int xx = 0; xx < out_size; ++xx) {
    const double center = in0 + (xx + 0.5) * scale;
    double ww = 0.0;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);  // C's truncating conversion, as PIL
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    for (int x = 0; x < xmax; ++x) {
      const double arg = (x + xmin - center + 0.5) * ss;
      const double w = filter == Filter::kLanczos ? lanczos_filter(arg) : bicubic_filter(arg);
      kk[static_cast<std::size_t>(x)] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x) {
      if (ww != 0.0) kk[static_cast<std::size_t>(x)] /= ww;
    }
    std::int32_t* k = &c.k[static_cast<std::size_t>(xx) * static_cast<std::size_t>(c.ksize)];
    for (int x = 0; x < xmax; ++x) {
      const double v = kk[static_cast<std::size_t>(x)] * (1 << kPrecisionBits);
      k[x] = static_cast<std::int32_t>(v < 0 ? -0.5 + v : 0.5 + v);
    }
    c.bounds[static_cast<std::size_t>(xx) * 2] = xmin;
    c.bounds[static_cast<std::size_t>(xx) * 2 + 1] = xmax;
  }
  return c;
}

std::uint8_t clip8(std::int64_t in) {
  const std::int64_t v = in >> kPrecisionBits;  // arithmetic shift, as PIL's lookup index
  return static_cast<std::uint8_t>(std::clamp<std::int64_t>(v, 0, 255));
}

// The horizontal pass over rows [offset, offset + out_h) of `in`.
void resample_horizontal(const Image& in, Image& out, int offset, const Coeffs& c) {
  const int ch = in.channels;
  for (int yy = 0; yy < out.height; ++yy) {
    const std::uint8_t* row = &in.pixels[static_cast<std::size_t>(yy + offset) * static_cast<std::size_t>(in.width) * ch];
    std::uint8_t* dst = &out.pixels[static_cast<std::size_t>(yy) * static_cast<std::size_t>(out.width) * ch];
    for (int xx = 0; xx < out.width; ++xx) {
      const int xmin = c.bounds[static_cast<std::size_t>(xx) * 2], xmax = c.bounds[static_cast<std::size_t>(xx) * 2 + 1];
      const std::int32_t* k = &c.k[static_cast<std::size_t>(xx) * static_cast<std::size_t>(c.ksize)];
      for (int b = 0; b < ch; ++b) {
        std::int64_t ss = std::int64_t{1} << (kPrecisionBits - 1);
        for (int x = 0; x < xmax; ++x) ss += std::int64_t{row[static_cast<std::size_t>(x + xmin) * ch + b]} * k[x];
        dst[static_cast<std::size_t>(xx) * ch + b] = clip8(ss);
      }
    }
  }
}

void resample_vertical(const Image& in, Image& out, const Coeffs& c, int shift) {
  const int ch = in.channels;
  const std::size_t stride = static_cast<std::size_t>(in.width) * ch;
  for (int yy = 0; yy < out.height; ++yy) {
    const int ymin = c.bounds[static_cast<std::size_t>(yy) * 2] - shift, ymax = c.bounds[static_cast<std::size_t>(yy) * 2 + 1];
    const std::int32_t* k = &c.k[static_cast<std::size_t>(yy) * static_cast<std::size_t>(c.ksize)];
    std::uint8_t* dst = &out.pixels[static_cast<std::size_t>(yy) * stride];
    for (std::size_t xb = 0; xb < stride; ++xb) {
      std::int64_t ss = std::int64_t{1} << (kPrecisionBits - 1);
      for (int y = 0; y < ymax; ++y) ss += std::int64_t{in.pixels[static_cast<std::size_t>(y + ymin) * stride + xb]} * k[y];
      dst[xb] = clip8(ss);
    }
  }
}

Image blank(int width, int height, int channels) {
  Image im;
  im.width = width;
  im.height = height;
  im.channels = channels;
  im.pixels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(channels), 0);
  return im;
}

// PIL Image.paste(src, box=(x, y)) without a mask: the overlap with dst, clipped.
void paste(Image& dst, const Image& src, int x, int y) {
  const int ch = dst.channels;
  for (int sy = std::max(0, -y); sy < src.height && sy + y < dst.height; ++sy) {
    const int sx0 = std::max(0, -x), sx1 = std::min(src.width, dst.width - x);
    if (sx1 <= sx0) return;
    std::memcpy(&dst.pixels[(static_cast<std::size_t>(sy + y) * dst.width + static_cast<std::size_t>(sx0 + x)) * ch],
                &src.pixels[(static_cast<std::size_t>(sy) * src.width + static_cast<std::size_t>(sx0)) * ch],
                static_cast<std::size_t>(sx1 - sx0) * ch);
  }
}

// Python's // on ints (floor division); the operands here are non-negative.
int floordiv(std::int64_t a, std::int64_t b) { return static_cast<int>(a / b); }

// A JPEG's EXIF orientation (1-8), 1 when absent or unreadable.
int jpeg_orientation(std::span<const std::uint8_t> f) {
  const std::size_t n = f.size();
  if (n < 4 || f[0] != 0xFF || f[1] != 0xD8) return 1;
  std::size_t p = 2;
  while (p + 4 <= n) {
    if (f[p] != 0xFF) return 1;
    const std::uint8_t marker = f[p + 1];
    if (marker == 0xFF) {  // fill byte
      ++p;
      continue;
    }
    if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {  // no length
      p += 2;
      continue;
    }
    if (marker == 0xDA || marker == 0xD9) return 1;  // start of scan / end of image: no EXIF before it
    const std::size_t len = (std::size_t{f[p + 2]} << 8) | f[p + 3];
    if (len < 2 || p + 2 + len > n) return 1;
    const std::uint8_t* seg = &f[p + 4];
    const std::size_t seg_len = len - 2;
    if (marker == 0xE1 && seg_len >= 14 && std::memcmp(seg, "Exif\0\0", 6) == 0) {
      const std::uint8_t* t = seg + 6;
      const std::size_t tn = seg_len - 6;
      const bool le = t[0] == 'I' && t[1] == 'I';
      if (!le && !(t[0] == 'M' && t[1] == 'M')) return 1;
      const auto u16 = [&](std::size_t o) -> std::uint32_t {
        return le ? (std::uint32_t{t[o]} | std::uint32_t{t[o + 1]} << 8) : (std::uint32_t{t[o]} << 8 | std::uint32_t{t[o + 1]});
      };
      const auto u32 = [&](std::size_t o) -> std::uint32_t {
        return le ? (u16(o) | u16(o + 2) << 16) : (u16(o) << 16 | u16(o + 2));
      };
      if (u16(2) != 42) return 1;
      const std::size_t ifd = u32(4);
      if (ifd + 2 > tn) return 1;
      const std::size_t count = u16(ifd);
      for (std::size_t e = 0; e < count; ++e) {
        const std::size_t o = ifd + 2 + e * 12;
        if (o + 12 > tn) return 1;
        if (u16(o) == 0x0112 && u16(o + 2) == 3) {  // Orientation, SHORT
          const std::uint32_t v = u16(o + 8);
          return v >= 1 && v <= 8 ? static_cast<int>(v) : 1;
        }
      }
      return 1;
    }
    p += 2 + len;
  }
  return 1;
}

// ImageOps.exif_transpose: 2 mirror, 3 rotate 180, 4 flip, 5 transpose, 6 rotate 90 clockwise, 7 transverse,
// 8 rotate 90 counter-clockwise.
Image apply_orientation(const Image& src, int orientation) {
  if (orientation <= 1 || orientation > 8) return src;
  const int w = src.width, h = src.height, ch = src.channels;
  const bool swap = orientation >= 5;
  Image out = blank(swap ? h : w, swap ? w : h, ch);
  out.name = src.name;
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      int sx = x, sy = y;
      switch (orientation) {
        case 2: sx = w - 1 - x; break;
        case 3: sx = w - 1 - x, sy = h - 1 - y; break;
        case 4: sy = h - 1 - y; break;
        case 5: sx = y, sy = x; break;
        case 6: sx = y, sy = h - 1 - x; break;
        case 7: sx = w - 1 - y, sy = h - 1 - x; break;
        case 8: sx = w - 1 - y, sy = x; break;
        default: break;
      }
      std::memcpy(&out.pixels[(static_cast<std::size_t>(y) * out.width + x) * ch],
                  &src.pixels[(static_cast<std::size_t>(sy) * w + sx) * ch], static_cast<std::size_t>(ch));
    }
  }
  return out;
}

// PIL Image.resize with `filter` (ImagingResampleInner).
Image resize_filtered(const Image& src, int width, int height, float box_x0, float box_y0, float box_x1, float box_y1,
                      Filter filter) {
  check_image(src, "resize");
  if (width <= 0 || height <= 0) throw std::invalid_argument("resize: the target size must be positive");
  // Image.resize: the box defaults to the whole image; the same size and box is a copy.
  if (box_x0 < 0 && box_y0 < 0 && box_x1 < 0 && box_y1 < 0) {
    box_x0 = 0, box_y0 = 0, box_x1 = static_cast<float>(src.width), box_y1 = static_cast<float>(src.height);
  }
  const bool whole = box_x0 == 0 && box_y0 == 0 && box_x1 == static_cast<float>(src.width) && box_y1 == static_cast<float>(src.height);
  if (width == src.width && height == src.height && whole) return src;
  // ImagingResampleInner (PIL's box is float; the coefficient arithmetic is double).
  const bool need_horizontal = width != src.width || box_x0 != 0 || box_x1 != static_cast<float>(width);
  const bool need_vertical = height != src.height || box_y0 != 0 || box_y1 != static_cast<float>(height);
  const Coeffs ch = precompute_coeffs(src.width, box_x0, box_x1, width, filter);
  const Coeffs cv = precompute_coeffs(src.height, box_y0, box_y1, height, filter);
  // First and last used rows of the source.
  const int ybox_first = cv.bounds[0];
  const int ybox_last = cv.bounds[static_cast<std::size_t>(height) * 2 - 2] + cv.bounds[static_cast<std::size_t>(height) * 2 - 1];
  Image cur;
  const Image* in = &src;
  int shift = 0;
  if (need_horizontal) {
    cur = blank(width, ybox_last - ybox_first, src.channels);
    resample_horizontal(src, cur, ybox_first, ch);
    in = &cur;
    shift = ybox_first;  // the vertical bounds move with the rows the horizontal pass kept
  }
  Image out;
  if (need_vertical) {
    out = blank(in->width, height, src.channels);
    resample_vertical(*in, out, cv, shift);
  } else {
    out = *in;
  }
  out.name = src.name;
  return out;
}

}  // namespace

const char* resize_mode_name(ResizeMode mode) {
  switch (mode) {
    case ResizeMode::kCrop: return "crop";
    case ResizeMode::kFit: return "fit";
    case ResizeMode::kStretch: return "stretch";
  }
  return "crop";
}

ResizeMode parse_resize_mode(std::string_view name) {
  if (name == "crop") return ResizeMode::kCrop;
  if (name == "fit") return ResizeMode::kFit;
  if (name == "stretch") return ResizeMode::kStretch;
  throw std::invalid_argument("resize mode must be crop, fit or stretch, not \"" + std::string(name) + "\"");
}

Image resize_lanczos(const Image& src, int width, int height, float box_x0, float box_y0, float box_x1, float box_y1) {
  return resize_filtered(src, width, height, box_x0, box_y0, box_x1, box_y1, Filter::kLanczos);
}

Image resize_to(const Image& src, int width, int height, ResizeMode mode) {
  check_image(src, "resize");
  if (width <= 0 || height <= 0) throw std::invalid_argument("resize: the target size must be positive");
  if (mode == ResizeMode::kStretch) return resize_lanczos(src, width, height);
  const double ratio = static_cast<double>(width) / height;
  const double src_ratio = static_cast<double>(src.width) / src.height;
  const std::int64_t iw = src.width, ih = src.height;
  Image res = blank(width, height, src.channels);
  res.name = src.name;
  if (mode == ResizeMode::kCrop) {
    // _resize_and_crop
    const int src_w = ratio > src_ratio ? width : floordiv(iw * height, ih);
    const int src_h = ratio <= src_ratio ? height : floordiv(ih * width, iw);
    const Image resized = resize_lanczos(src, src_w, src_h);
    paste(res, resized, width / 2 - src_w / 2, height / 2 - src_h / 2);
    return res;
  }
  // _resize_and_fill
  const int src_w = ratio < src_ratio ? width : floordiv(iw * height, ih);
  const int src_h = ratio >= src_ratio ? height : floordiv(ih * width, iw);
  const Image resized = resize_lanczos(src, src_w, src_h);
  paste(res, resized, width / 2 - src_w / 2, height / 2 - src_h / 2);
  const float rw = static_cast<float>(resized.width), rh = static_cast<float>(resized.height);
  // The bands stretch the first / last row (column) with Image.resize's default filter, bicubic (diffusers passes none).
  constexpr Filter kBand = Filter::kBicubic;
  if (ratio < src_ratio) {
    const int fill_height = height / 2 - src_h / 2;
    if (fill_height > 0) {
      paste(res, resize_filtered(resized, width, fill_height, 0, 0, static_cast<float>(width), 0, kBand), 0, 0);
      paste(res, resize_filtered(resized, width, fill_height, 0, rh, static_cast<float>(width), rh, kBand), 0, fill_height + src_h);
    }
  } else if (ratio > src_ratio) {
    const int fill_width = width / 2 - src_w / 2;
    if (fill_width > 0) {
      paste(res, resize_filtered(resized, fill_width, height, 0, 0, 0, static_cast<float>(height), kBand), 0, 0);
      paste(res, resize_filtered(resized, fill_width, height, rw, 0, rw, static_cast<float>(height), kBand), fill_width + src_w, 0);
    }
  }
  return res;
}

Image to_grayscale(const Image& src) {
  check_image(src, "grayscale");
  if (src.channels == 1) return src;
  Image out = blank(src.width, src.height, 1);
  out.name = src.name;
  const std::size_t n = static_cast<std::size_t>(src.width) * static_cast<std::size_t>(src.height);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t r = src.pixels[i * 3], g = src.pixels[i * 3 + 1], b = src.pixels[i * 3 + 2];
    out.pixels[i] = static_cast<std::uint8_t>((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16);
  }
  return out;
}

std::vector<float> latent_mask(const Image& mask, int latent_h, int latent_w) {
  check_image(mask, "mask");
  if (mask.channels != 1) throw std::invalid_argument("mask: expected a 1-channel image");
  if (latent_h <= 0 || latent_w <= 0) throw std::invalid_argument("mask: the latent grid must be positive");
  const int in_h = mask.height, in_w = mask.width;
  // aten upsample_bilinear2d (CPU, fp32): scale = in / out, src = max(scale * (dst + 0.5) - 0.5, 0), the upper
  // neighbour clamped to the border.
  const float scale_h = static_cast<float>(in_h) / static_cast<float>(latent_h);
  const float scale_w = static_cast<float>(in_w) / static_cast<float>(latent_w);
  const auto source = [](float scale, int dst) { return std::max(scale * (static_cast<float>(dst) + 0.5f) - 0.5f, 0.0f); };
  const auto bin = [&](int y, int x) { return mask.pixels[static_cast<std::size_t>(y) * in_w + x] >= 128 ? 1.0f : 0.0f; };
  std::vector<float> out(static_cast<std::size_t>(latent_h) * static_cast<std::size_t>(latent_w));
  for (int oy = 0; oy < latent_h; ++oy) {
    const float hr = source(scale_h, oy);
    const int h1 = std::min(static_cast<int>(hr), in_h - 1);
    const int h1p = h1 < in_h - 1 ? 1 : 0;
    const float l1 = std::clamp(hr - static_cast<float>(h1), 0.0f, 1.0f), l0 = 1.0f - l1;
    for (int ox = 0; ox < latent_w; ++ox) {
      const float wr = source(scale_w, ox);
      const int w1 = std::min(static_cast<int>(wr), in_w - 1);
      const int w1p = w1 < in_w - 1 ? 1 : 0;
      const float m1 = std::clamp(wr - static_cast<float>(w1), 0.0f, 1.0f), m0 = 1.0f - m1;
      const float top = m0 * bin(h1, w1) + m1 * bin(h1, w1 + w1p);
      const float bottom = m0 * bin(h1 + h1p, w1) + m1 * bin(h1 + h1p, w1 + w1p);
      out[static_cast<std::size_t>(oy) * latent_w + ox] = l0 * top + l1 * bottom;
    }
  }
  return out;
}

void inpaint_blend(std::span<float> x, std::span<const float> z0, std::span<const float> noise, std::span<const float> mask,
                   float sigma) {
  const std::size_t plane = mask.size();
  if (plane == 0 || x.size() % plane != 0 || z0.size() != x.size() || noise.size() != x.size()) {
    throw std::invalid_argument("inpaint_blend: mismatched sizes");
  }
  const float keep = 1.0f - sigma;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float m = mask[i % plane];
    const float a = sigma * noise[i];
    const float b = keep * z0[i];
    const float proper = a + b;
    const float p = (1.0f - m) * proper;
    const float q = m * x[i];
    x[i] = p + q;
  }
}

int strength_start_step(int num_steps, double strength) {
  const double n = static_cast<double>(num_steps);
  const double init_timestep = std::min(n * strength, n);
  return static_cast<int>(std::max(n - init_timestep, 0.0));
}

Image decode_image(std::span<const std::uint8_t> file, int channels) {
  if (channels != 1 && channels != 3) throw std::invalid_argument("decode_image: channels must be 1 or 3");
  if (file.size() > kMaxImageFileBytes) {
    throw std::runtime_error("image file larger than " + std::to_string(kMaxImageFileBytes >> 20) + " MiB");
  }
  const int size = static_cast<int>(file.size());
  // The header first: a hostile size is refused before any pixel memory is allocated.
  int w = 0, h = 0, n = 0;
  if (stbi_info_from_memory(file.data(), size, &w, &h, &n) == 0) {
    throw std::runtime_error(std::string("cannot decode the image (PNG or JPEG expected): ") + stbi_failure_reason());
  }
  if (w <= 0 || h <= 0 || w > kMaxInputSide || h > kMaxInputSide || std::int64_t{w} * h > kMaxInputPixels) {
    throw std::runtime_error("image is " + std::to_string(w) + "x" + std::to_string(h) + ": at most " +
                             std::to_string(kMaxInputSide) + " pixels on a side and " + std::to_string(kMaxInputPixels / 1000000) +
                             " megapixels");
  }
  int dw = 0, dh = 0;
  const std::unique_ptr<stbi_uc, void (*)(void*)> px(stbi_load_from_memory(file.data(), size, &dw, &dh, &n, 3), stbi_image_free);
  if (px == nullptr) throw std::runtime_error(std::string("cannot decode the image (PNG or JPEG expected): ") + stbi_failure_reason());
  if (dw != w || dh != h) throw std::runtime_error("image decode: the size changed between the header and the pixels");
  Image rgb;
  rgb.width = w;
  rgb.height = h;
  rgb.channels = 3;
  rgb.pixels.assign(px.get(), px.get() + static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3);
  const int orientation = jpeg_orientation(file);
  if (orientation != 1) rgb = apply_orientation(rgb, orientation);
  if (channels == 1) return to_grayscale(rgb);
  return rgb;
}

Image load_image(const std::filesystem::path& file, int channels) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(file, ec)) throw std::runtime_error(util::path_utf8(file) + ": not a readable image file");
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + util::path_utf8(file));
  // Bounded read: never more than the cap plus one chunk, whatever the file claims or becomes.
  std::vector<std::uint8_t> bytes;
  constexpr std::size_t kChunk = std::size_t{1} << 20;
  while (in) {
    const std::size_t at = bytes.size();
    bytes.resize(at + kChunk);
    in.read(reinterpret_cast<char*>(bytes.data() + at), static_cast<std::streamsize>(kChunk));
    bytes.resize(at + static_cast<std::size_t>(in.gcount()));
    if (bytes.size() > kMaxImageFileBytes) {
      throw std::runtime_error(util::path_utf8(file) + ": image file larger than " + std::to_string(kMaxImageFileBytes >> 20) + " MiB");
    }
  }
  if (in.bad()) throw std::runtime_error("cannot read " + util::path_utf8(file));
  Image im;
  try {
    im = decode_image(bytes, channels);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(util::path_utf8(file) + ": " + e.what());
  }
  im.name = util::path_utf8(file.filename());
  return im;
}

}  // namespace mi::pipeline
