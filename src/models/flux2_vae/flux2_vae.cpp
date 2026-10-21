#include "models/flux2_vae/flux2_vae.h"

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/basic/elementwise.h"
#include "models/flux2_vae/flux2_vae_internal.h"
#include "models/flux2_vae/torch_group_norm.h"
#include "runtime/pinned_ring.h"
#include "weights/upload.h"

namespace mi::flux2_vae {
namespace {

using namespace detail;
using runtime::align_up;
using weights::TensorMap;

// kTailReceptiveRadius from the layer structure: each 3x3 conv adds one pixel at its resolution.
constexpr int tail_receptive_radius() {
  int px = 1;     // conv_out at full resolution
  int scale = 1;  // output pixels per pixel of the current level, walking from the output back
  for (int b = 3; b >= 0; --b) {
    px += 2 * kResnetsPerBlock * scale;
    if (b > 0 && kUpSample[b - 1]) {
      px += scale;  // block b-1's upsampler conv runs at block b's resolution
      scale *= 2;
    }
  }
  return px + 2 * scale;  // mid resnet 1 at H/8
}
static_assert(tail_receptive_radius() == kTailReceptiveRadius, "kTailReceptiveRadius out of date");

const weights::WeightGroup& decoder_group(const weights::ModelIndex& index) {
  const weights::WeightGroup* g = index.find_group(weights::Component::kVae, "decoder");
  if (g == nullptr) throw std::runtime_error("mi::flux2_vae: VAE group 'decoder' missing from the index");
  return *g;
}

bool is_conv_weight(const weights::IndexedTensor& t) { return t.shape.size() == 4; }

void check_lt(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("mi::flux2_vae: ") + what + " failed (status " + std::to_string(static_cast<int>(st)) + ")");
}

}  // namespace

namespace detail {

void fail(const std::string& msg) { throw std::invalid_argument("mi::flux2_vae: " + msg); }

Layout layout(std::int64_t h, std::int64_t w, const DecodeMode& mode) {
  const DType tail = mode.bf16_convs ? DType::kBF16 : DType::kF32;
  const std::int64_t hh = 2 * h, ww = 2 * w;  // H/8 x W/8
  Layout l;
  // A region holds the full-resolution 192-channel tensor (tail dtype). Every other tensor fits in half a region:
  // 96 ch at H, <= 384 ch at H/2 (tail dtype), and the fp32 head (384 ch at H/8, 1/16 of the tail tensor); the
  // mid attention scratch takes a whole region.
  const std::size_t tail_bytes = static_cast<std::size_t>(kTailMaxCh * 64 * hh * ww) * dtype_size(tail);
  const std::size_t mid_ws = vae::mid_attention_workspace_bytes(act(nullptr, DType::kF32, hh, ww, kMidCh));
  l.region = align_up(std::max(tail_bytes, mid_ws), 2 * runtime::kArenaAlign);
  if (tail_bytes >= kCutlassMaxBytes) l.strip = kStripScratchBytes;

  // GroupNorm inputs: {channels, scale (H/8 = 1 ... H = 8), tail?}.
  struct Gn { int c, scale; bool tail; };
  for (const Gn& g : {Gn{384, 1, false}, Gn{384, 1, true}, Gn{384, 2, true}, Gn{384, 4, true}, Gn{192, 4, true},
                      Gn{192, 8, true}, Gn{96, 8, true}}) {
    const Tensor x = act(nullptr, g.tail ? tail : DType::kF32, hh * g.scale, ww * g.scale, g.c);
    l.aux = std::max(l.aux, vae::group_norm_workspace_bytes(x, vae::kGroupNormGroups));
    if (x.dtype == DType::kF32 && mode.torch_group_norm)
      l.aux = std::max(l.aux, torch_group_norm_workspace_bytes(x, vae::kGroupNormGroups));
  }
  // Convs: {in, out, kernel size, scale, tail?}.
  struct Cv { int c, k, r, scale; bool tail; };
  for (const Cv& v : {Cv{32, 32, 1, 1, false}, Cv{32, 384, 3, 1, false}, Cv{384, 384, 3, 1, false}, Cv{384, 384, 3, 1, true},
                      Cv{384, 384, 3, 2, true}, Cv{384, 384, 3, 4, true}, Cv{384, 192, 3, 4, true}, Cv{192, 192, 3, 4, true},
                      Cv{384, 192, 1, 4, true}, Cv{192, 192, 3, 8, true}, Cv{192, 96, 3, 8, true}, Cv{96, 96, 3, 8, true},
                      Cv{192, 96, 1, 8, true}, Cv{96, 3, 3, 8, true}}) {
    const DType dt = v.tail ? tail : DType::kF32;
    const Tensor x = act(nullptr, dt, hh * v.scale, ww * v.scale, v.c);
    const Tensor wt = Tensor::view(nullptr, dt, Device::kCUDA, {v.k, v.r, v.r, v.c});
    l.aux = std::max(l.aux, vae::conv2d_workspace_bytes(x, wt));
  }
  return l;
}

void check_decode_args(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode, bool has_bf16) {
  const bool batched = latents.rank == 4;
  if (latents.device != Device::kCUDA || latents.dtype != DType::kF32 || !latents.is_contiguous() ||
      (latents.rank != 3 && !(batched && latents.dim(0) == 1)) || latents.dim(-3) != kLatentChannels || latents.dim(-2) < 1 ||
      latents.dim(-1) < 1)
    fail("latents must be contiguous CUDA fp32 [128, h, w] or [1, 128, h, w], got " + latents.describe());
  const std::int64_t img_h = latents.dim(-2) * kPixelsPerLatent, img_w = latents.dim(-1) * kPixelsPerLatent;
  if (rgb.dtype != DType::kU8 || !rgb.is_contiguous() || rgb.data == nullptr || rgb.dim(-1) != 3 || rgb.dim(-2) != img_w ||
      rgb.dim(-3) != img_h || (rgb.rank != 3 && !(rgb.rank == 4 && rgb.dim(0) == 1)))
    fail("rgb must be contiguous u8 [" + std::to_string(img_h) + ", " + std::to_string(img_w) + ", 3], got " + rgb.describe());
  if (mode.bf16_convs && !has_bf16) fail("DecodeMode::bf16_convs needs a decoder loaded with bf16_weights");
}

}  // namespace detail

using namespace detail;

std::size_t Flux2VaeDecoder::persistent_bytes(const weights::ModelIndex& index, bool bf16_weights) {
  const weights::WeightGroup& g = decoder_group(index);
  std::size_t total = weights::group_device_bytes(g);
  if (bf16_weights) {
    for (const auto& t : g.tensors)
      if (is_conv_weight(t)) total += align_up(static_cast<std::size_t>(t.numel()) * 2);
  }
  return total;
}

Flux2VaeDecoder::Flux2VaeDecoder(const weights::ModelIndex& index, runtime::DeviceArena& arena,
                                 runtime::StreamSet& streams, bool bf16_weights, bool direct_io)
    : arena_(arena), has_bf16_(bf16_weights) {
  const auto t0 = std::chrono::steady_clock::now();
  const weights::WeightGroup& group = decoder_group(index);
  check_lt(cublasLtCreate(&lt_), "cublasLtCreate");
  TensorMap m;
  {
    weights::ReaderCache readers(index, direct_io);
    // The largest tensor is 5.3 MB: a small ring suffices and keeps the pinned allocation cheap.
    runtime::PinnedRing ring(streams.copy(), 2, std::size_t{16} << 20);
    m = weights::upload_group(readers, group, arena, ring);
    ring.synchronize();
  }
  load_stats_.bytes = group.total_bytes;

  const auto take = [&](const std::string& name, std::initializer_list<std::int64_t> shape) {
    auto it = m.find(name);
    if (it == m.end()) throw std::runtime_error("mi::flux2_vae: weight '" + name + "' missing");
    const Tensor& t = it->second;
    bool ok = t.dtype == DType::kF32 && t.rank == static_cast<int>(shape.size());
    int i = 0;
    for (std::int64_t d : shape) ok = ok && t.dim(i++) == d;
    if (!ok) throw std::runtime_error("mi::flux2_vae: weight '" + name + "' has unexpected type/shape " + t.describe());
    return t;
  };
  std::vector<Conv*> convs;
  const auto make_conv = [&](const std::string& p, std::int64_t k, std::int64_t c, std::int64_t r) {
    Conv cv;
    cv.w = take(p + ".weight", {k, c, r, r});  // KCRS until converted below
    cv.bias = take(p + ".bias", {k});
    return cv;
  };
  const auto make_norm = [&](const std::string& p, std::int64_t c) { return Norm{take(p + ".weight", {c}), take(p + ".bias", {c})}; };
  const auto make_resnet = [&](const std::string& p, int in, int out) {
    Resnet r;
    r.in_ch = in;
    r.out_ch = out;
    r.norm1 = make_norm(p + ".norm1", in);
    r.conv1 = make_conv(p + ".conv1", out, in, 3);
    r.norm2 = make_norm(p + ".norm2", out);
    r.conv2 = make_conv(p + ".conv2", out, out, 3);
    if (in != out) r.shortcut = make_conv(p + ".conv_shortcut", out, in, 1);
    return r;
  };

  bn_mean_ = take("bn.running_mean", {kLatentChannels});
  bn_var_ = take("bn.running_var", {kLatentChannels});
  post_quant_conv_ = make_conv("post_quant_conv", 32, 32, 1);
  conv_in_ = make_conv("decoder.conv_in", kMidCh, 32, 3);
  for (int i = 0; i < 2; ++i) mid_resnets_[i] = make_resnet("decoder.mid_block.resnets." + std::to_string(i), kMidCh, kMidCh);
  const std::string a = "decoder.mid_block.attentions.0.";
  mid_attn_.gn_gamma = take(a + "group_norm.weight", {kMidCh});
  mid_attn_.gn_beta = take(a + "group_norm.bias", {kMidCh});
  mid_attn_.wq = take(a + "to_q.weight", {kMidCh, kMidCh});
  mid_attn_.bq = take(a + "to_q.bias", {kMidCh});
  mid_attn_.wk = take(a + "to_k.weight", {kMidCh, kMidCh});
  mid_attn_.bk = take(a + "to_k.bias", {kMidCh});
  mid_attn_.wv = take(a + "to_v.weight", {kMidCh, kMidCh});
  mid_attn_.bv = take(a + "to_v.bias", {kMidCh});
  mid_attn_.wo = take(a + "to_out.0.weight", {kMidCh, kMidCh});
  mid_attn_.bo = take(a + "to_out.0.bias", {kMidCh});
  int prev = kMidCh;
  for (int b = 0; b < 4; ++b) {
    const std::string p = "decoder.up_blocks." + std::to_string(b);
    for (int r = 0; r < kResnetsPerBlock; ++r)
      up_[b].resnets[r] = make_resnet(p + ".resnets." + std::to_string(r), r == 0 ? prev : kUpOut[b], kUpOut[b]);
    if (kUpSample[b]) up_[b].upsampler = make_conv(p + ".upsamplers.0.conv", kUpOut[b], kUpOut[b], 3);
    prev = kUpOut[b];
  }
  norm_out_ = make_norm("decoder.conv_norm_out", kUpOut[3]);
  conv_out_ = make_conv("decoder.conv_out", 3, kUpOut[3], 3);

  convs = {&post_quant_conv_, &conv_in_, &conv_out_};
  for (Resnet& r : mid_resnets_) convs.insert(convs.end(), {&r.conv1, &r.conv2});
  for (UpBlock& u : up_) {
    for (Resnet& r : u.resnets) {
      convs.insert(convs.end(), {&r.conv1, &r.conv2});
      if (r.shortcut.w.data != nullptr) convs.push_back(&r.shortcut);
    }
    if (u.upsampler.w.data != nullptr) convs.push_back(&u.upsampler);
  }

  // KCRS -> KRSC in place through a stack scratch (same size), plus the bf16 copies.
  const cudaStream_t s = streams.compute();
  {
    runtime::StackScope scope(arena);
    std::size_t max_bytes = 0;
    for (const Conv* c : convs) max_bytes = std::max(max_bytes, c->w.nbytes());
    void* tmp = arena.alloc_stack(max_bytes);
    for (Conv* c : convs) {
      const Tensor src = c->w;
      const std::int64_t k = src.dim(0), ci = src.dim(1), r = src.dim(2);
      if (bf16_weights) {
        c->w_bf16 = arena.alloc_persistent_tensor(DType::kBF16, {k, r, r, ci});
        vae::conv_weight_to_krsc(src, c->w_bf16, s);
      }
      const Tensor krsc = Tensor::view(tmp, DType::kF32, Device::kCUDA, {k, r, r, ci});
      vae::conv_weight_to_krsc(src, krsc, s);
      MI_CUDA_CHECK(cudaMemcpyAsync(src.data, tmp, src.nbytes(), cudaMemcpyDeviceToDevice, s));
      c->w = Tensor::view(src.data, DType::kF32, Device::kCUDA, {k, r, r, ci});
    }
    MI_CUDA_CHECK(cudaStreamSynchronize(s));
  }
  for (cudaEvent_t& e : band_done_) MI_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  load_stats_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Flux2VaeDecoder::~Flux2VaeDecoder() {
  for (cudaEvent_t e : band_done_) {
    if (e != nullptr) cudaEventDestroy(e);
  }
  if (lt_ != nullptr) cublasLtDestroy(lt_);
}

std::size_t Flux2VaeDecoder::workspace_bytes(std::int64_t height, std::int64_t width, const DecodeMode& mode) {
  if (height < 1 || width < 1) fail("latent grid must be at least 1x1");
  const Layout l = layout(height, width, mode);
  return 2 * l.region + align_up(l.aux) + l.strip;
}

// Tensors CUTLASS cannot address at once run in row strips. Each output pixel's accumulation order does not
// depend on where its tile lies, so strips are bit-identical to one call: a 3x3 strip reads one halo row on each
// side (zero padding only at the image border), computes its rows plus the halo rows into the strip scratch and
// keeps the inner rows. The residual is elementwise, so its halo rows only reach discarded outputs (y may alias
// it). The exact tiled decode (flux2_vae_tiled.cpp) relies on the same property for its row bands.
void Flux2VaeDecoder::conv(const Ctx& c, const Tensor& x, const Conv& cv, const Tensor* residual, const Tensor& y) const {
  const Tensor& w = c.dt == DType::kBF16 ? cv.w_bf16 : cv.w;
  if (x.nbytes() < kCutlassMaxBytes && y.nbytes() < kCutlassMaxBytes) {
    vae::conv2d(x, w, cv.bias, residual, y, c.aux, c.aux_bytes, c.stream);
    return;
  }
  const std::int64_t rows = x.dim(1), cols = x.dim(2), halo = (w.dim(1) - 1) / 2;
  const std::size_t esize = dtype_size(c.dt);
  const auto row_bytes = [&](const Tensor& t) { return static_cast<std::size_t>(cols * t.dim(3)) * esize; };
  const auto band = [&](const Tensor& t, std::int64_t a, std::int64_t b) {
    return act(static_cast<char*>(t.data) + static_cast<std::size_t>(a) * row_bytes(t), c.dt, b - a, cols, t.dim(3));
  };
  const std::size_t max_rows = std::min(c.strip_bytes / row_bytes(y), (kCutlassMaxBytes - 1) / std::max(row_bytes(x), row_bytes(y)));
  const std::int64_t step = static_cast<std::int64_t>(max_rows) - 2 * halo;
  if (step < 1) fail("image too wide for the conv strip scratch");
  for (std::int64_t r0 = 0; r0 < rows; r0 += step) {
    const std::int64_t r1 = std::min(rows, r0 + step), a = std::max<std::int64_t>(0, r0 - halo),
                       b = std::min(rows, r1 + halo);
    const Tensor res = residual != nullptr ? band(*residual, a, b) : Tensor{};
    const Tensor* res_ptr = residual != nullptr ? &res : nullptr;
    if (halo == 0) {
      vae::conv2d(band(x, a, b), w, cv.bias, res_ptr, band(y, a, b), c.aux, c.aux_bytes, c.stream);
      continue;
    }
    const Tensor tmp = act(c.strip, c.dt, b - a, cols, y.dim(3));
    vae::conv2d(band(x, a, b), w, cv.bias, res_ptr, tmp, c.aux, c.aux_bytes, c.stream);
    MI_CUDA_CHECK(cudaMemcpyAsync(band(y, r0, r1).data, static_cast<char*>(c.strip) + static_cast<std::size_t>(r0 - a) * row_bytes(y),
                                  static_cast<std::size_t>(r1 - r0) * row_bytes(y), cudaMemcpyDeviceToDevice, c.stream));
  }
}

void Flux2VaeDecoder::norm(const Ctx& c, const Tensor& x, const Norm& n, bool silu, const Tensor& y) const {
  if (c.dt == DType::kF32 && c.mode->torch_group_norm)
    torch_group_norm(x, vae::kGroupNormGroups, vae::kResnetEps, n.gamma, n.beta, silu, y, c.aux, c.aux_bytes, c.stream);
  else
    vae::group_norm(x, vae::kGroupNormGroups, vae::kResnetEps, n.gamma, n.beta, silu, y, c.aux, c.aux_bytes, c.stream);
}

// ResnetBlock2D on x (in_ch) using the scratch buffers a and b; returns the buffer holding the output.
//   same channels: a = SiLU(GN(x)), b = conv1(a), b = SiLU(GN(b)), x = conv2(b) + x            -> x
//   with shortcut: a = shortcut(x), x = SiLU(GN(x)), b = conv1(x), b = SiLU(GN(b)), a = conv2(b) + a -> a
// The shortcut variant computes the 1x1 shortcut first so the GroupNorm can overwrite x: three live tensors of
// in + 2 * out channels instead of 2 * in + out (bit-identical, only the schedule changes).
char* Flux2VaeDecoder::resnet(const Ctx& c, const Resnet& r, char* x, char* a, char* b, std::int64_t h,
                              std::int64_t w) const {
  const Tensor xt = act(x, c.dt, h, w, r.in_ch);
  const Tensor bt = act(b, c.dt, h, w, r.out_ch);
  if (r.shortcut.w.data != nullptr) {
    const Tensor st = act(a, c.dt, h, w, r.out_ch);
    conv(c, xt, r.shortcut, nullptr, st);
    norm(c, xt, r.norm1, true, xt);
    conv(c, xt, r.conv1, nullptr, bt);
    norm(c, bt, r.norm2, true, bt);
    conv(c, bt, r.conv2, &st, st);
    return a;
  }
  const Tensor at = act(a, c.dt, h, w, r.in_ch);
  norm(c, xt, r.norm1, true, at);
  conv(c, at, r.conv1, nullptr, bt);
  norm(c, bt, r.norm2, true, bt);
  conv(c, bt, r.conv2, &xt, xt);
  return x;
}

// Writes the attention output, fp32 [1, H/8, W/8, 384], to x. a and b hold one such tensor each; the attention
// scratch (attn_ws, attn_bytes) may overlap them.
void Flux2VaeDecoder::run_head(const Ctx& c, const Tensor& latents, char* x_buf, char* a, char* b, char* attn_ws,
                               std::size_t attn_bytes) const {
  const std::int64_t hh = 2 * c.h, ww = 2 * c.w;
  const Tensor z = Tensor::view(latents.data, DType::kF32, Device::kCUDA, {1, kLatentChannels, c.h, c.w});
  const Tensor unpacked = act(a, DType::kF32, hh, ww, 32);
  vae::bn_denorm_unpack(z, bn_mean_, bn_var_, vae::kBatchNormEps, unpacked, c.stream);
  c.stage("unpack", unpacked);
  const Tensor pq = act(b, DType::kF32, hh, ww, 32);
  conv(c, unpacked, post_quant_conv_, nullptr, pq);
  c.stage("post_quant_conv", pq);
  const Tensor x = act(x_buf, DType::kF32, hh, ww, kMidCh);
  conv(c, pq, conv_in_, nullptr, x);
  c.stage("conv_in", x);
  resnet(c, mid_resnets_[0], x_buf, a, b, hh, ww);
  c.stage("mid.resnets.0", x);
  vae::mid_attention(lt_, x, mid_attn_, &x, x, attn_ws, attn_bytes, c.mode->attention, c.mode->fast_linears, c.stream);
  c.stage("mid.attention", x);
}

// Buffer schedule (p, q = the two regions, each with halves; sizes in the tail dtype): every resnet keeps its
// three tensors in one half each until up_blocks[2]'s upsampler, whose 192-channel full-resolution input and
// output each take a whole region; up_blocks[3]'s first resnet then reads its input from one region and writes
// shortcut and conv1 output into the two halves of the other.
Tensor Flux2VaeDecoder::run_tail(const Ctx& c, const Tensor& rgb) const {
  std::int64_t h = 2 * c.h, w = 2 * c.w;
  const std::size_t half = c.region / 2;
  char* p = c.r0;
  char* q = c.r1;
  if (c.dt == DType::kBF16) {
    kernels::cast(act(p, DType::kF32, h, w, kMidCh), act(q, DType::kBF16, h, w, kMidCh), c.stream);
    std::swap(p, q);
  }
  resnet(c, mid_resnets_[1], p, q, q + half, h, w);
  c.stage("mid.resnets.1", act(p, c.dt, h, w, kMidCh));
  for (int b = 0; b < 2; ++b) {  // 384 channels, main tensor stays in p
    for (const Resnet& r : up_[b].resnets) resnet(c, r, p, q, q + half, h, w);
    const Tensor up = act(q, c.dt, 2 * h, 2 * w, kUpOut[b]);
    vae::upsample_nearest2x(act(p, c.dt, h, w, kUpOut[b]), up, c.stream);
    h *= 2;
    w *= 2;
    conv(c, up, up_[b].upsampler, nullptr, act(p, c.dt, h, w, kUpOut[b]));
    c.stage(b == 0 ? "up_blocks.0" : "up_blocks.1", act(p, c.dt, h, w, kUpOut[b]));
  }
  // up_blocks[2]: 384 -> 192 at H/2, main tensor moves to q.
  resnet(c, up_[2].resnets[0], p, q, q + half, h, w);
  for (int r = 1; r < kResnetsPerBlock; ++r) resnet(c, up_[2].resnets[r], q, p, p + half, h, w);
  {
    const Tensor up = act(p, c.dt, 2 * h, 2 * w, kUpOut[2]);  // whole region
    vae::upsample_nearest2x(act(q, c.dt, h, w, kUpOut[2]), up, c.stream);
    h *= 2;
    w *= 2;
    conv(c, up, up_[2].upsampler, nullptr, act(q, c.dt, h, w, kUpOut[2]));  // whole region; the input above is dead
    c.stage("up_blocks.2", act(q, c.dt, h, w, kUpOut[2]));
  }
  // up_blocks[3]: 192 -> 96 at H, main tensor back in p.
  resnet(c, up_[3].resnets[0], q, p, p + half, h, w);
  for (int r = 1; r < kResnetsPerBlock; ++r) resnet(c, up_[3].resnets[r], p, q, q + half, h, w);
  const Tensor x = act(p, c.dt, h, w, kUpOut[3]);
  c.stage("up_blocks.3", x);
  norm(c, x, norm_out_, true, x);
  c.stage("norm_out", x);
  Tensor out = act(q, c.dt, h, w, 3);
  conv(c, x, conv_out_, nullptr, out);
  c.stage("conv_out", out);
  if (c.dt == DType::kBF16) {
    const Tensor out_f32 = act(q + half, DType::kF32, h, w, 3);
    kernels::cast(out, out_f32, c.stream);
    out = out_f32;
  }
  // Host output: stage in p (dead after conv_out).
  const Tensor dst = rgb.device == Device::kCUDA ? Tensor::view(rgb.data, DType::kU8, Device::kCUDA, {1, h, w, 3})
                                                 : Tensor::view(p, DType::kU8, Device::kCUDA, {1, h, w, 3});
  vae::postprocess_to_rgb8(out, dst, c.stream);
  return dst;
}

void Flux2VaeDecoder::decode(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode, cudaStream_t stream,
                             const StageHook& hook) {
  check_decode_args(latents, rgb, mode, has_bf16_);
  const std::int64_t h = latents.dim(-2), w = latents.dim(-1);
  runtime::StackScope scope(arena_);
  const Layout l = layout(h, w, mode);
  char* regions = static_cast<char*>(arena_.alloc_stack(2 * l.region));
  void* aux = arena_.alloc_stack(l.aux);
  void* strip = arena_.alloc_stack(l.strip);
  Ctx c{stream, DType::kF32, &mode, aux, l.aux, regions, regions + l.region, l.region, strip, l.strip, h, w, &hook};
  run_head(c, latents, c.r0, c.r1, c.r1 + c.region / 2, c.r1, c.region);
  c.dt = mode.bf16_convs ? DType::kBF16 : DType::kF32;
  const Tensor staged = run_tail(c, rgb);
  if (rgb.device != Device::kCUDA) {
    MI_CUDA_CHECK(cudaMemcpyAsync(rgb.data, staged.data, staged.nbytes(), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

}  // namespace mi::flux2_vae
