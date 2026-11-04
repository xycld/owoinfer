// <chrono> before the HIP headers: their __noinline__ macro breaks libstdc++'s <format>, which <chrono> includes in C++20.
#include <chrono>

#include "vae.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "safetensors.h"

namespace mi::hip {
namespace {

using hip_vae::Products;

constexpr float kResnetEps = 1e-6f;
constexpr float kBatchNormEps = 1e-4f;  // vae.config.batch_norm_eps
constexpr int kUpBlocks = 4, kResnetsPerBlock = 3;
constexpr std::uint64_t kAlign = 256;
// Bytes of one attention slab: its fp32 scores and their packed pieces, [rows, L] each.
constexpr std::uint64_t kSlabBytes = 256ull << 20;

std::uint64_t align_up(std::uint64_t v, std::uint64_t a = kAlign) { return (v + a - 1) / a * a; }

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("hip vae: " + message);
}

// Device pointers as byte offsets into the weight buffer until it exists.
struct Conv {
  std::uint64_t w = 0, b = 0;  // packed pieces of the [k, r r c] KRSC weight (conv_out: TF32-rounded fp32 KRSC), bias [k]
  int k = 0, c = 0, r = 0;
};
struct Norm {
  std::uint64_t gamma = 0, beta = 0;
  int c = 0;
};
struct Resnet {
  Norm norm1, norm2;
  Conv conv1, conv2, shortcut;  // shortcut.k == 0 when in == out
  int in = 0, out = 0;
};

}  // namespace

struct VaeDecoder::Weights {
  DeviceBuffer buffer;
  std::uint64_t bn_scale = 0, bn_mean = 0;
  Conv post_quant, conv_in, conv_out, qkv, attn_out;
  Norm attn_norm, norm_out;
  Resnet mid[2];
  Resnet up[kUpBlocks][kResnetsPerBlock];
  Conv upsampler[kUpBlocks - 1];

  const char* base() const { return static_cast<const char*>(buffer.data()); }
  const float* f32(std::uint64_t at) const { return reinterpret_cast<const float*>(base() + at); }
  const std::uint16_t* u16(std::uint64_t at) const { return reinterpret_cast<const std::uint16_t*>(base() + at); }
};

namespace {

// Host image of the weight buffer: tensors at 256-byte offsets.
class Packer {
 public:
  explicit Packer(const st::File& file) : file_(file) {}

  template <class T>
  std::uint64_t add(const std::vector<T>& v) {
    const std::uint64_t at = data_.size();
    data_.resize(align_up(at + v.size() * sizeof(T)));
    std::memcpy(data_.data() + at, v.data(), v.size() * sizeof(T));
    return at;
  }
  std::vector<float> read(const std::string& name, std::initializer_list<std::int64_t> shape) const {
    const st::TensorInfo* t = file_.find(name);
    if (t == nullptr) throw std::runtime_error("hip vae: weight '" + name + "' missing from " + file_.path());
    if (t->shape != std::vector<std::int64_t>(shape)) throw std::runtime_error("hip vae: weight '" + name + "' has an unexpected shape");
    return file_.read_f32(name);
  }
  // A [k, c, r, r] convolution as its packed KRSC pieces, or (narrow) as TF32-rounded fp32 KRSC.
  Conv conv(const std::string& p, int k, int c, int r, bool narrow = false) {
    const std::vector<float> kcrs = read(p + ".weight", {k, c, r, r});
    std::vector<float> krsc(kcrs.size());
    const std::size_t taps = static_cast<std::size_t>(r) * r, cs = static_cast<std::size_t>(c);
    for (std::size_t ki = 0; ki < static_cast<std::size_t>(k); ++ki) {
      for (std::size_t ci = 0; ci < cs; ++ci) {
        for (std::size_t t = 0; t < taps; ++t) krsc[(ki * taps + t) * cs + ci] = kcrs[(ki * cs + ci) * taps + t];
      }
    }
    Conv cv;
    cv.k = k;
    cv.c = c;
    cv.r = r;
    if (narrow) {
      for (float& v : krsc) v = hip_vae::host_tf32(v);
      cv.w = add(krsc);
    } else {
      cv.w = add(packed(krsc, k, static_cast<int>(taps) * c, Products::kTf32));
    }
    cv.b = add(read(p + ".bias", {k}));
    return cv;
  }
  static std::vector<std::uint16_t> packed(const std::vector<float>& w, int n, int k, Products products) {
    std::vector<std::uint16_t> out(hip_vae::packed_weight_elems(n, k, products));
    hip_vae::pack_weight(w.data(), n, k, products, out.data());
    return out;
  }
  Norm norm(const std::string& p, int c) {
    Norm n;
    n.c = c;
    n.gamma = add(read(p + ".weight", {c}));
    n.beta = add(read(p + ".bias", {c}));
    return n;
  }
  Resnet resnet(const std::string& p, int in, int out) {
    Resnet r;
    r.in = in;
    r.out = out;
    r.norm1 = norm(p + ".norm1", in);
    r.conv1 = conv(p + ".conv1", out, in, 3);
    r.norm2 = norm(p + ".norm2", out);
    r.conv2 = conv(p + ".conv2", out, out, 3);
    if (in != out) r.shortcut = conv(p + ".conv_shortcut", out, in, 1);
    return r;
  }
  const std::vector<char>& data() const { return data_; }

 private:
  const st::File& file_;
  std::vector<char> data_;
};

// The device memory of one decode.
struct Layout {
  std::uint64_t region = 0;  // bytes per ping-pong region; each holds two halves
  std::uint64_t latents = 0, rgb = 0, aux = 0;
  // The mid attention's scratch in the second region, as offsets from its start.
  std::uint64_t xn = 0, qkv = 0, qp = 0, kp = 0, vtp = 0, scores = 0, pp = 0;
  std::int64_t l = 0, l_rows = 0, l_cols = 0, l_k = 0, mid_cols = 0;  // L; Q rows, K rows, PV's K, V^T rows (padded)
  int slab = 0;                                                       // query rows per slab
  std::uint64_t total() const { return 2 * region + latents + rgb + aux; }
};

Layout layout(int c, std::uint32_t h, std::uint32_t w) {
  require(c > 0 && c % 32 == 0, "decoder widths must be multiples of 32");
  require(h > 0 && w > 0, "latent grid must be at least 1x1");
  // The sizes the kernels can address, checked before any product below can overflow.
  const std::string image = "a " + std::to_string(16ull * w) + "x" + std::to_string(16ull * h) + " image";
  require(h <= (1u << 20) && w <= (1u << 20), image + " is past the decoder's limits");
  const std::uint64_t full = 256ull * h * w, mid = 4ull * static_cast<std::uint64_t>(c);
  // up_blocks[3]'s first GroupNorm: 2c channels at full resolution, the largest group and the largest launch (one
  // work-item per 4 channels of a pixel; AMD launches hold fewer than 2^32 work-items).
  require(full * (2ull * static_cast<std::uint64_t>(c) / 32) <= (1ull << 30),
          image + " is past the decoder's limits: a GroupNorm group would hold more than 2^30 elements");
  require(full * (2ull * static_cast<std::uint64_t>(c) / 4) < (1ull << 32),
          image + " is past the decoder's limits: a GroupNorm launch would need 2^32 work-items or more");
  Layout l;
  l.l = 4ll * h * w;
  l.l_rows = hip_vae::round_up(l.l, hip_vae::kTileM);
  l.l_cols = hip_vae::round_up(l.l, hip_vae::kTileN);
  l.l_k = hip_vae::round_up(l.l, 16);
  l.mid_cols = hip_vae::round_up(static_cast<std::int64_t>(mid), hip_vae::kTileN);
  const auto lk = static_cast<std::uint64_t>(l.l_k), ll = static_cast<std::uint64_t>(l.l);
  const std::uint64_t per_row = lk * (4 + 3 * 2);
  l.slab = static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(l.l_rows),
                                                    std::max<std::uint64_t>(32, kSlabBytes / per_row / 32 * 32)));
  const auto slab = static_cast<std::uint64_t>(l.slab);
  std::uint64_t at = 0;
  const auto take = [&](std::uint64_t bytes) {
    const std::uint64_t r = at;
    at += align_up(bytes);
    return r;
  };
  l.xn = take(ll * mid * 4);
  l.qkv = take(ll * 3 * mid * 4);
  l.qp = take(3 * static_cast<std::uint64_t>(l.l_rows) * mid * 2);
  l.kp = take(3 * static_cast<std::uint64_t>(l.l_cols) * mid * 2);
  l.vtp = take(3 * static_cast<std::uint64_t>(l.mid_cols) * lk * 2);
  l.scores = take(slab * lk * 4);
  l.pp = take(3 * slab * lk * 2);
  const std::uint64_t tail = 2ull * static_cast<std::uint64_t>(c) * full * 4;  // up_blocks[2]'s upsampler output
  l.region = align_up(std::max(tail, at), 2 * kAlign);
  l.latents = align_up(128ull * h * w * 4);
  l.rgb = align_up(full * 3);
  l.aux = align_up(hip_vae::gn_accum_bytes()) + kAlign;  // GroupNorm accumulators, then {mean, rstd} per group
  return l;
}

double gib(std::uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0); }

// Records each op between two timing events when profiling.
class Ops {
 public:
  Ops(hipStream_t stream, bool profile) : stream_(stream), profile_(profile) {}
  template <class F>
  void run(const std::string& name, F&& op) {
    if (!profile_) {
      op();
      return;
    }
    auto start = std::make_unique<Event>(true), stop = std::make_unique<Event>(true);
    start->record(stream_);
    op();
    stop->record(stream_);
    timed_.push_back({name, std::move(start), std::move(stop)});
  }
  // After the stream finished: milliseconds per name, in order of first use.
  std::vector<std::pair<std::string, double>> profile() const {
    std::vector<std::pair<std::string, double>> out;
    for (const Timed& t : timed_) {
      const double ms = t.stop->elapsed_ms(*t.start);
      auto it = std::find_if(out.begin(), out.end(), [&](const auto& e) { return e.first == t.name; });
      if (it == out.end()) {
        out.emplace_back(t.name, ms);
      } else {
        it->second += ms;
      }
    }
    return out;
  }

 private:
  struct Timed {
    std::string name;
    std::unique_ptr<Event> start, stop;
  };
  hipStream_t stream_;
  bool profile_;
  std::vector<Timed> timed_;
};

std::string dims(int h, int w) { return " @" + std::to_string(h) + "x" + std::to_string(w); }

class Run {
 public:
  Run(const VaeDecoder::Weights& w, Ops& ops, hipStream_t stream, Products products, void* accum, float* stats)
      : w_(w), ops_(ops), stream_(stream), products_(products), accum_(accum), stats_(stats) {}

  void conv(const Conv& cv, const float* x, int h, int w, float* y, const float* residual = nullptr, bool upsample = false) {
    hip_vae::ConvArgs a;
    a.x = x;
    a.w = w_.u16(cv.w);
    a.bias = w_.f32(cv.b);
    a.residual = residual;
    a.out = y;
    a.ldo = cv.k;
    a.h = h;
    a.w_out = w;
    a.c = cv.c;
    a.n = cv.k;
    a.taps = cv.r * cv.r;
    a.upsample = upsample;
    const std::string name = "conv " + std::to_string(cv.r) + "x" + std::to_string(cv.r) + " " + std::to_string(cv.c) + "->" +
                             std::to_string(cv.k) + (upsample ? " upsampled" : "") + dims(h, w);
    ops_.run(name, [&] { hip_vae::conv(a, products_, stream_); });
  }
  void norm(const Norm& n, const float* x, std::int64_t pixels, float* y, bool silu) {
    const std::string shape = " " + std::to_string(n.c) + " ch, " + std::to_string(pixels) + " px";
    ops_.run("group_norm stats" + shape, [&] { hip_vae::group_norm_stats(x, pixels, n.c, kResnetEps, accum_, stats_, stream_); });
    ops_.run(std::string(silu ? "group_norm+silu apply" : "group_norm apply") + shape, [&] {
      hip_vae::group_norm_apply(x, y, pixels, n.c, stats_, w_.f32(n.gamma), w_.f32(n.beta), silu, stream_);
    });
  }
  // ResnetBlock2D: returns the buffer holding the output, x or a.
  float* resnet(const Resnet& r, float* x, float* a, float* b, int h, int w) {
    const std::int64_t px = static_cast<std::int64_t>(h) * w;
    if (r.shortcut.k != 0) {
      conv(r.shortcut, x, h, w, a);
      norm(r.norm1, x, px, x, true);
      conv(r.conv1, x, h, w, b);
      norm(r.norm2, b, px, b, true);
      conv(r.conv2, b, h, w, a, a);
      return a;
    }
    norm(r.norm1, x, px, a, true);
    conv(r.conv1, a, h, w, b);
    norm(r.norm2, b, px, b, true);
    conv(r.conv2, b, h, w, x, x);
    return x;
  }
  // x [L, mid] += to_out(SDPA(to_q/k/v(GN(x)))), scratch in ws (the second region).
  void attention(float* x, char* ws, const Layout& l, int mid) {
    const int len = static_cast<int>(l.l);
    float* xn = reinterpret_cast<float*>(ws + l.xn);
    float* qkv = reinterpret_cast<float*>(ws + l.qkv);
    auto* qp = reinterpret_cast<std::uint16_t*>(ws + l.qp);
    auto* kp = reinterpret_cast<std::uint16_t*>(ws + l.kp);
    auto* vtp = reinterpret_cast<std::uint16_t*>(ws + l.vtp);
    float* scores = reinterpret_cast<float*>(ws + l.scores);
    auto* pp = reinterpret_cast<std::uint16_t*>(ws + l.pp);
    norm(w_.attn_norm, x, l.l, xn, false);
    linear("attention qkv", w_.qkv, xn, len, qkv, nullptr);
    const std::int64_t q_stride = l.l_rows * mid, k_stride = l.l_cols * mid, v_stride = l.mid_cols * l.l_k;
    ops_.run("attention split q, k, v", [&] {
      hip_vae::split_pack(qkv, 3 * mid, false, len, mid, qp, static_cast<int>(l.l_rows), mid, stream_);
      hip_vae::split_pack(qkv + mid, 3 * mid, false, len, mid, kp, static_cast<int>(l.l_cols), mid, stream_);
      hip_vae::split_pack(qkv + 2 * mid, 3 * mid, true, mid, len, vtp, static_cast<int>(l.mid_cols), static_cast<int>(l.l_k),
                          stream_);
    });
    // SDPA's scale as the engine forms it: 1 / sqrt(E) in fp32.
    const float scale = 1.0f / std::sqrt(static_cast<float>(mid));
    float* o = xn;  // the GroupNorm output is dead after the projections
    const std::int64_t p_stride = static_cast<std::int64_t>(l.slab) * l.l_k;
    for (int r0 = 0; r0 < len; r0 += l.slab) {
      const int rows = std::min(l.slab, len - r0);
      ops_.run("attention scores", [&] {
        hip_vae::piece_gemm(qp + static_cast<std::int64_t>(r0) * mid, q_stride, kp, k_stride, mid, scores, l.l_k, rows, len, stream_);
      });
      ops_.run("attention softmax", [&] { hip_vae::softmax_pack(scores, l.l_k, rows, len, scale, pp, p_stride, static_cast<int>(l.l_k), stream_); });
      ops_.run("attention P V", [&] {
        hip_vae::piece_gemm(pp, p_stride, vtp, v_stride, static_cast<int>(l.l_k), o + static_cast<std::int64_t>(r0) * mid, mid, rows,
                            mid, stream_);
      });
    }
    // The reference order: round(to_out(o) + bias), then + residual.
    linear("attention out", w_.attn_out, o, len, x, x);
  }

 private:
  // out [rows, k] = (x [rows, c] · Wᵀ + b) + residual, exact fp32 products.
  void linear(const char* name, const Conv& cv, const float* x, int rows, float* out, const float* residual) {
    hip_vae::ConvArgs a;
    a.x = x;
    a.w = w_.u16(cv.w);
    a.bias = w_.f32(cv.b);
    a.residual = residual;
    a.out = out;
    a.ldo = cv.k;
    a.h = 1;
    a.w_out = rows;
    a.c = cv.c;
    a.n = cv.k;
    a.taps = 1;
    ops_.run(name, [&] { hip_vae::conv(a, Products::kFp32, stream_); });
  }

  const VaeDecoder::Weights& w_;
  Ops& ops_;
  hipStream_t stream_;
  Products products_;
  void* accum_;
  float* stats_;
};

}  // namespace

VaeDecoder::VaeDecoder(Device& device, const std::filesystem::path& path) : device_(device), weights_(std::make_unique<Weights>()) {
  const st::File file(path);
  const st::TensorInfo* conv_in = file.find("decoder.conv_in.weight");
  if (conv_in == nullptr || conv_in->shape.size() != 4 || conv_in->shape[0] % 4 != 0 || conv_in->shape[1] != 32) {
    throw std::runtime_error("hip vae: " + file.path() + " has no FLUX.2 decoder.conv_in.weight [4c, 32, 3, 3]");
  }
  base_ = static_cast<int>(conv_in->shape[0] / 4);
  require(base_ % 32 == 0, "decoder widths must be multiples of 32");
  const int c = base_, mid = 4 * c;
  Weights& w = *weights_;
  Packer pk(file);

  const std::vector<float> mean = pk.read("bn.running_mean", {128}), var = pk.read("bn.running_var", {128});
  std::vector<float> scale(128);
  for (std::size_t i = 0; i < 128; ++i) scale[i] = std::sqrt(var[i] + kBatchNormEps);  // correctly rounded, as torch
  w.bn_scale = pk.add(scale);
  w.bn_mean = pk.add(mean);
  w.post_quant = pk.conv("post_quant_conv", 32, 32, 1);
  w.conv_in = pk.conv("decoder.conv_in", mid, 32, 3);
  for (int i = 0; i < 2; ++i) w.mid[i] = pk.resnet("decoder.mid_block.resnets." + std::to_string(i), mid, mid);
  const std::string a = "decoder.mid_block.attentions.0.";
  w.attn_norm = pk.norm(a + "group_norm", mid);
  {
    // to_q, to_k, to_v as one [3 mid, mid] linear: the same products and sums per output as three.
    std::vector<float> wqkv, bqkv;
    for (const char* p : {"to_q", "to_k", "to_v"}) {
      const std::vector<float> wt = pk.read(a + p + ".weight", {mid, mid}), bt = pk.read(a + p + ".bias", {mid});
      wqkv.insert(wqkv.end(), wt.begin(), wt.end());
      bqkv.insert(bqkv.end(), bt.begin(), bt.end());
    }
    w.qkv = Conv{pk.add(Packer::packed(wqkv, 3 * mid, mid, Products::kFp32)), pk.add(bqkv), 3 * mid, mid, 1};
    const std::vector<float> wo = pk.read(a + "to_out.0.weight", {mid, mid});
    w.attn_out = Conv{pk.add(Packer::packed(wo, mid, mid, Products::kFp32)), pk.add(pk.read(a + "to_out.0.bias", {mid})), mid, mid, 1};
  }
  int prev = mid;
  for (int b = 0; b < kUpBlocks; ++b) {
    const int out = c * (b < 2 ? 4 : b == 2 ? 2 : 1);
    const std::string p = "decoder.up_blocks." + std::to_string(b);
    for (int r = 0; r < kResnetsPerBlock; ++r) w.up[b][r] = pk.resnet(p + ".resnets." + std::to_string(r), r == 0 ? prev : out, out);
    if (b + 1 < kUpBlocks) w.upsampler[b] = pk.conv(p + ".upsamplers.0.conv", out, out, 3);
    prev = out;
  }
  w.norm_out = pk.norm("decoder.conv_norm_out", c);
  w.conv_out = pk.conv("decoder.conv_out", 3, c, 3, true);

  const std::vector<char>& data = pk.data();
  w.buffer = DeviceBuffer(data.size());
  w.buffer.upload(data.data(), data.size());
  // That hipMemcpy from pageable memory can return before its DMA lands, and decode()'s stream does not wait for it.
  device_.synchronize();
}

VaeDecoder::~VaeDecoder() = default;

std::uint64_t VaeDecoder::weight_bytes() const { return weights_->buffer.bytes(); }

std::uint64_t VaeDecoder::workspace_bytes(int base_channels, std::uint32_t h, std::uint32_t w) {
  return layout(base_channels, h, w).total();
}

void VaeDecoder::require_fits(std::uint32_t h, std::uint32_t w) const {
  const std::uint64_t need = workspace_bytes(h, w), free = device_.free_memory();
  if (need > free) {
    char msg[256];
    std::snprintf(msg, sizeof msg, "the VAE decode of a %ux%u image needs %.2f GiB of device memory, %s has %.2f GiB free",
                  16 * w, 16 * h, gib(need), device_.info().name.c_str(), gib(free));
    throw std::runtime_error(msg);
  }
}

void VaeDecoder::set_conv_products(Products products) {
  require(products != Products::kFp32, "the convolutions' weights are packed for TF32 products");
  conv_products_ = products;
}

VaeDecodeStats VaeDecoder::decode(const float* latents, std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb,
                                  std::vector<float>* image) {
  const auto t0 = std::chrono::steady_clock::now();
  const int c = base_, mid = 4 * c;
  const Layout l = layout(c, h, w);  // validates h and w: the sizes below fit an int
  const int hh = static_cast<int>(2 * h), ww = static_cast<int>(2 * w), H = static_cast<int>(16 * h), W = static_cast<int>(16 * w);
  device_.make_current();
  require_fits(h, w);
  // The stream outlives the buffers, and `drain` (destroyed first) waits for the stream's work, so an exception never
  // frees memory that queued kernels still use.
  Stream stream;
  DeviceBuffer r0(l.region), r1(l.region), z(l.latents), out(l.rgb), aux(l.aux);
  struct Drain {
    hipStream_t s;
    ~Drain() { (void)hipStreamSynchronize(s); }
  } drain{stream.get()};
  copy_async(z.data(), latents, 128ull * h * w * 4, stream.get());
  void* accum = aux.data();
  float* stats = reinterpret_cast<float*>(static_cast<char*>(aux.data()) + align_up(hip_vae::gn_accum_bytes()));
  const std::uint64_t half = l.region / 2;
  const Weights& wt = *weights_;
  const auto at = [](const DeviceBuffer& b, std::uint64_t off = 0) {
    return reinterpret_cast<float*>(static_cast<char*>(b.data()) + off);
  };

  Ops ops(stream.get(), profile_);
  Run run(wt, ops, stream.get(), conv_products_, accum, stats);
  float* p = at(r0);
  float* q = at(r1);

  // Head at H/8: unpack, post_quant_conv, conv_in, mid resnet 0, mid attention.
  float* unpacked = q;
  float* pq = at(r1, half);
  ops.run("latent unpack", [&] {
    hip_vae::latent_unpack(z.as<float>(), wt.f32(wt.bn_scale), wt.f32(wt.bn_mean), unpacked, static_cast<int>(h),
                           static_cast<int>(w), stream.get());
  });
  run.conv(wt.post_quant, unpacked, hh, ww, pq);
  run.conv(wt.conv_in, pq, hh, ww, p);
  run.resnet(wt.mid[0], p, q, at(r1, half), hh, ww);
  run.attention(p, static_cast<char*>(r1.data()), l, mid);

  // Tail: mid resnet 1, the up blocks, norm_out + SiLU, conv_out, RGB8.
  int ch = hh, cw = ww;
  run.resnet(wt.mid[1], p, q, at(r1, half), ch, cw);
  for (int b = 0; b < kUpBlocks; ++b) {
    float* x = p;
    for (int r = 0; r < kResnetsPerBlock; ++r) {
      float* other = x == at(r0) ? at(r1) : at(r0);
      x = run.resnet(wt.up[b][r], x, other, reinterpret_cast<float*>(reinterpret_cast<char*>(other) + half), ch, cw);
    }
    if (b + 1 < kUpBlocks) {
      float* y = x == at(r0) ? at(r1) : at(r0);
      ch *= 2;
      cw *= 2;
      run.conv(wt.upsampler[b], x, ch, cw, y, nullptr, true);
      x = y;
    }
    p = x;
    q = x == at(r0) ? at(r1) : at(r0);
  }
  run.norm(wt.norm_out, p, static_cast<std::int64_t>(ch) * cw, p, true);
  ops.run("conv_out 3x3" + dims(ch, cw), [&] {
    hip_vae::conv_narrow(p, wt.f32(wt.conv_out.w), wt.f32(wt.conv_out.b), q, ch, cw, c, 3, stream.get());
  });
  const std::int64_t n = static_cast<std::int64_t>(H) * W * 3;
  ops.run("rgb8", [&] { hip_vae::rgb8(q, out.as<std::uint8_t>(), n, stream.get()); });
  stream.synchronize();

  rgb.resize(static_cast<std::size_t>(n));
  out.download(rgb.data(), static_cast<std::size_t>(n));
  if (image != nullptr) {
    image->resize(static_cast<std::size_t>(n));
    check(hipMemcpy(image->data(), q, static_cast<std::size_t>(n) * 4, hipMemcpyDeviceToHost), "hipMemcpy of the image");
  }
  VaeDecodeStats st;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  st.device_bytes = l.total();
  if (profile_) st.profile = ops.profile();
  return st;
}

}  // namespace mi::hip
