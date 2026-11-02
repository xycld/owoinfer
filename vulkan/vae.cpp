#include "vae.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "safetensors.h"

namespace mi::vk {
namespace {

const std::uint32_t kConvCoopSpirv[] = {
#include "vae_conv_coop.spv.inc"
};
const std::uint32_t kConvPlainSpirv[] = {
#include "vae_conv_plain.spv.inc"
};
const std::uint32_t kConvNarrowSpirv[] = {
#include "vae_conv_narrow.spv.inc"
};
const std::uint32_t kGnStatsSpirv[] = {
#include "vae_gn_stats.spv.inc"
};
const std::uint32_t kGnFinalizeSpirv[] = {
#include "vae_gn_finalize.spv.inc"
};
const std::uint32_t kGnApplySpirv[] = {
#include "vae_gn_apply.spv.inc"
};
const std::uint32_t kSoftmaxSpirv[] = {
#include "vae_softmax.spv.inc"
};
const std::uint32_t kTransposeSpirv[] = {
#include "vae_transpose.spv.inc"
};
const std::uint32_t kUnpackSpirv[] = {
#include "vae_latent_unpack.spv.inc"
};
const std::uint32_t kRgb8Spirv[] = {
#include "vae_rgb8.spv.inc"
};

struct ConvPush {
  std::uint64_t a, b, out, bias, residual;
  std::uint32_t m, n, k, lda, ldb, ldo, h, w, c, m0;
};
struct GnStatsPush {
  std::uint64_t x, part;
  std::uint32_t pixels, c, ppc, chunks;
};
struct GnFinalizePush {
  std::uint64_t part, stats;
  std::uint32_t chunks;
  float eps;
};
struct GnApplyPush {
  std::uint64_t x, y, stats, gamma, beta;
  std::uint32_t quads, c, silu;
};
struct SoftmaxPush {
  std::uint64_t s;
  std::uint32_t rows, l, ld;
  float scale;
};
struct TransposePush {
  std::uint64_t in, out;
  std::uint32_t r, c, ld_in, ld_out;
};
struct UnpackPush {
  std::uint64_t z, scale, mean, y;
  std::uint32_t h, w;
};
struct Rgb8Push {
  std::uint64_t x, rgb;
  std::uint32_t n;
};

constexpr std::uint32_t kCoopSubgroup = 32;
constexpr std::uint32_t kGroups = 32;
constexpr float kResnetEps = 1e-6f;
constexpr float kBatchNormEps = 1e-4f;  // vae.config.batch_norm_eps
constexpr std::uint32_t kGnChunkElems = 32768;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("vulkan vae: " + message);
}

std::uint32_t div_up(std::uint64_t a, std::uint64_t b) { return static_cast<std::uint32_t>((a + b - 1) / b); }

// A 1-D count of workgroups as x + y * x_count, within the guaranteed 65535 per dimension.
void dispatch_1d(const Pipeline& p, VkCommandBuffer cb, const void* push, std::uint32_t groups) {
  if (groups == 0) return;
  const std::uint32_t x = std::min<std::uint32_t>(groups, 65535);
  p.dispatch(cb, push, x, div_up(groups, x));
}

std::uint32_t gn_ppc(std::uint32_t c) { return std::max<std::uint32_t>(1, kGnChunkElems / c); }

}  // namespace

const char* vae_path_name(VaePath path) { return path == VaePath::kCoop ? "coop" : "plain"; }

VaeKernels::VaeKernels(Device& device)
    : VaeKernels(device, device.info().caps.coop_bf16 ? VaePath::kCoop : VaePath::kPlain) {}

VaeKernels::VaeKernels(Device& device, VaePath path) : device_(device), path_(path) {
  if (path == VaePath::kCoop && !device.info().caps.coop_bf16) {
    throw std::runtime_error(device.info().name + " cannot run the cooperative-matrix VAE: " +
                             device.info().caps.coop_bf16_reason);
  }
  gn_stats_ = std::make_unique<Pipeline>(device, kGnStatsSpirv, sizeof(kGnStatsSpirv), sizeof(GnStatsPush));
  gn_finalize_ = std::make_unique<Pipeline>(device, kGnFinalizeSpirv, sizeof(kGnFinalizeSpirv), sizeof(GnFinalizePush));
  gn_apply_ = std::make_unique<Pipeline>(device, kGnApplySpirv, sizeof(kGnApplySpirv), sizeof(GnApplyPush));
  softmax_ = std::make_unique<Pipeline>(device, kSoftmaxSpirv, sizeof(kSoftmaxSpirv), sizeof(SoftmaxPush));
  transpose_ = std::make_unique<Pipeline>(device, kTransposeSpirv, sizeof(kTransposeSpirv), sizeof(TransposePush));
  unpack_ = std::make_unique<Pipeline>(device, kUnpackSpirv, sizeof(kUnpackSpirv), sizeof(UnpackPush));
  rgb8_ = std::make_unique<Pipeline>(device, kRgb8Spirv, sizeof(kRgb8Spirv), sizeof(Rgb8Push));
}

VaeKernels::~VaeKernels() = default;

// Coop: outputs at least 64 wide run on bf16 fragments; narrower ones (post_quant_conv's 32, conv_out's 3) on the plain
// kernels, whose operands then get the same rounding. Outputs at most 4 wide take the narrow kernel on either path.
const Pipeline& VaeKernels::conv_pipeline(const ConvArgs& a, VaePrecision precision, std::uint32_t& tile_rows,
                                          std::uint32_t& tile_cols) const {
  const std::uint32_t taps = a.taps, up = a.upsample ? 1 : 0;
  if (path_ == VaePath::kCoop && a.n >= 64) {
    const std::uint32_t fn = a.n % 128 == 0 ? 4 : a.n % 96 == 0 ? 3 : a.n % 64 == 0 ? 2 : 4;
    std::uint32_t a_terms = 2, b_terms = a.b_bf16 ? 1 : 2, tf32 = 0;
    if (precision == VaePrecision::kTf32) tf32 = 1;
    if (precision == VaePrecision::kFp32) a_terms = 3;
    const std::uint32_t shared = 2 * (a_terms * 128 + b_terms * 32 * fn) * 2 * 16;
    require(shared <= device_.info().caps.shared_memory_bytes,
            "the cooperative-matrix convolution needs " + std::to_string(shared) + " bytes of shared memory");
    tile_rows = 128;
    tile_cols = 32 * fn;
    const std::vector<std::uint32_t> key = {0, taps, up, a_terms, b_terms, tf32, fn};
    auto& p = conv_[key];
    if (!p) {
      p = std::make_unique<Pipeline>(device_, kConvCoopSpirv, sizeof(kConvCoopSpirv), sizeof(ConvPush), kCoopSubgroup,
                                     std::vector<std::uint32_t>{taps, up, a_terms, b_terms, tf32, fn});
    }
    return *p;
  }
  const std::uint32_t tf32 = precision == VaePrecision::kTf32 ? 1 : 0;
  if (a.n <= 4) {
    tile_rows = 256;
    tile_cols = 4;
    const std::vector<std::uint32_t> key = {2, taps, up, tf32};
    auto& p = conv_[key];
    if (!p) {
      p = std::make_unique<Pipeline>(device_, kConvNarrowSpirv, sizeof(kConvNarrowSpirv), sizeof(ConvPush), 0,
                                     std::vector<std::uint32_t>{taps, up, tf32});
    }
    return *p;
  }
  // The decoder's widths are multiples of 96: tiles 96 wide, others 64.
  const std::uint32_t tn = a.n % 96 == 0 ? 6 : 4;
  tile_rows = 128;
  tile_cols = 16 * tn;
  const std::vector<std::uint32_t> key = {1, taps, up, tf32, tn};
  auto& p = conv_[key];
  if (!p) {
    p = std::make_unique<Pipeline>(device_, kConvPlainSpirv, sizeof(kConvPlainSpirv), sizeof(ConvPush), 0,
                                   std::vector<std::uint32_t>{taps, up, tf32, tn});
  }
  return *p;
}

void VaeKernels::record_conv(VkCommandBuffer cb, const ConvArgs& a, VaePrecision precision) const {
  require(a.taps == 0 || a.taps == 1 || a.taps == 9, "conv: taps must be 0, 1 or 9");
  require(a.n > 0 && a.k > 0 && a.k % 4 == 0 && a.m0 <= a.m, "conv: bad shape (k must be a multiple of 4)");
  require(a.ldb % 4 == 0 && a.ldb >= a.k && a.ldo >= a.n, "conv: ldb must be a multiple of 4; ldb, ldo must cover k, n");
  // The cooperative-matrix kernel stores 4 outputs at a time; the plain one (and narrow outputs) one.
  require(a.ldo % 4 == 0 || path_ == VaePath::kPlain || a.n < 64, "conv: ldo must be a multiple of 4");
  require(a.a % 16 == 0 && a.b % 16 == 0 && a.out % 16 == 0 && a.residual % 16 == 0, "conv: addresses must be 16-byte aligned");
  if (a.taps == 0) {
    require(a.lda % 4 == 0 && a.lda >= a.k, "conv: lda must be a multiple of 4 covering k");
  } else {
    require(a.c % 8 == 0 && a.k == a.taps * a.c && a.h > 0 && a.w > 0 && a.m <= a.h * a.w,
            "conv: c must be a multiple of 8, k = taps * c and rows within h * w");
    require(!a.upsample || (a.taps == 9 && a.h % 2 == 0 && a.w % 2 == 0), "conv: upsample needs a 3x3 conv and an even output");
  }
  if (a.m == a.m0) return;
  std::uint32_t tile_rows = 0, tile_cols = 0;
  const Pipeline& p = conv_pipeline(a, precision, tile_rows, tile_cols);
  const ConvPush push{a.a, a.b, a.out, a.bias, a.residual, a.m, a.n, a.k, a.lda, a.ldb, a.ldo, a.h, a.w, a.c, a.m0};
  const std::uint32_t rows_groups = div_up(a.m - a.m0, tile_rows);
  require(rows_groups <= 65535, "conv: too many rows for one dispatch; split it into bands");
  p.dispatch(cb, &push, div_up(a.n, tile_cols), rows_groups);
  compute_barrier(cb);
}

std::uint64_t VaeKernels::gn_part_bytes(std::uint32_t pixels, std::uint32_t c) {
  return static_cast<std::uint64_t>(div_up(pixels, gn_ppc(c))) * kGroups * 16;
}

void VaeKernels::record_gn_stats(VkCommandBuffer cb, VkDeviceAddress x, std::uint32_t pixels, std::uint32_t c,
                                 VkDeviceAddress part, VkDeviceAddress stats, float eps) const {
  require(c % kGroups == 0 && c % 4 == 0 && c <= 1024 && pixels > 0, "group norm: c must be a multiple of 32 up to 1024");
  require(x % 16 == 0 && part % 16 == 0 && stats % 8 == 0, "group norm: x and part must be 16-byte aligned");
  const std::uint32_t ppc = gn_ppc(c), chunks = div_up(pixels, ppc);
  const GnStatsPush sp{x, part, pixels, c, ppc, chunks};
  dispatch_1d(*gn_stats_, cb, &sp, chunks);
  compute_barrier(cb);
  const GnFinalizePush fp{part, stats, chunks, eps};
  gn_finalize_->dispatch(cb, &fp, kGroups, 1);
  compute_barrier(cb);
}

void VaeKernels::record_gn_apply(VkCommandBuffer cb, VkDeviceAddress x, VkDeviceAddress y, std::uint32_t pixels,
                                 std::uint32_t c, VkDeviceAddress stats, VkDeviceAddress gamma, VkDeviceAddress beta,
                                 bool silu) const {
  require(c % kGroups == 0 && c % 4 == 0, "group norm: c must be a multiple of 32");
  require(x % 16 == 0 && y % 16 == 0, "group norm: x and y must be 16-byte aligned");
  const std::uint64_t quads = static_cast<std::uint64_t>(pixels) * c / 4;
  require(quads < (1ull << 32), "group norm: tensor too large");
  const GnApplyPush push{x, y, stats, gamma, beta, static_cast<std::uint32_t>(quads), c, silu ? 1u : 0u};
  dispatch_1d(*gn_apply_, cb, &push, div_up(quads, 256));
  compute_barrier(cb);
}

void VaeKernels::record_softmax(VkCommandBuffer cb, VkDeviceAddress s, std::uint32_t rows, std::uint32_t l,
                                std::uint32_t ld, float scale) const {
  require(ld >= l, "softmax: ld must cover l");
  const SoftmaxPush push{s, rows, l, ld, scale};
  dispatch_1d(*softmax_, cb, &push, rows);
  compute_barrier(cb);
}

void VaeKernels::record_transpose(VkCommandBuffer cb, VkDeviceAddress in, VkDeviceAddress out, std::uint32_t r_count,
                                  std::uint32_t c_count, std::uint32_t ld_in, std::uint32_t ld_out) const {
  require(ld_in >= c_count && ld_out >= r_count, "transpose: strides must cover the rows");
  require(div_up(r_count, 32) <= 65535 && div_up(c_count, 32) <= 65535, "transpose: too large");
  const TransposePush push{in, out, r_count, c_count, ld_in, ld_out};
  transpose_->dispatch(cb, &push, div_up(c_count, 32), div_up(r_count, 32));
  compute_barrier(cb);
}

void VaeKernels::record_latent_unpack(VkCommandBuffer cb, VkDeviceAddress z, VkDeviceAddress scale, VkDeviceAddress mean,
                                      VkDeviceAddress y, std::uint32_t h, std::uint32_t w) const {
  const std::uint64_t n = static_cast<std::uint64_t>(h) * w * 128;
  require(n < (1ull << 32), "latent unpack: too large");
  const UnpackPush push{z, scale, mean, y, h, w};
  dispatch_1d(*unpack_, cb, &push, div_up(n, 256));
  compute_barrier(cb);
}

void VaeKernels::record_rgb8(VkCommandBuffer cb, VkDeviceAddress x, VkDeviceAddress rgb, std::uint32_t n) const {
  require(rgb % 4 == 0, "rgb8: rgb must be 4-byte aligned");
  const Rgb8Push push{x, rgb, n};
  dispatch_1d(*rgb8_, cb, &push, div_up(div_up(n, 4), 256));
  compute_barrier(cb);
}

// ---- Decoder ----------------------------------------------------------------------------------------------------------

namespace {

struct Conv {
  VkDeviceAddress w = 0, b = 0;  // fp32 [k, r, r, c] (KRSC), [k]
  std::uint32_t k = 0, c = 0, r = 0;
  bool bf16 = false;  // every weight is a bf16 value
};
struct Norm {
  VkDeviceAddress gamma = 0, beta = 0;
  std::uint32_t c = 0;
};
struct Resnet {
  Norm norm1, norm2;
  Conv conv1, conv2, shortcut;  // shortcut.k == 0 when in == out
  std::uint32_t in = 0, out = 0;
};

constexpr int kUpBlocks = 4, kResnetsPerBlock = 3;
constexpr std::uint64_t kAlign = 256;
std::uint64_t align_up(std::uint64_t v, std::uint64_t a = kAlign) { return (v + a - 1) / a * a; }

bool all_bf16(const std::vector<float>& v) {
  return std::all_of(v.begin(), v.end(), [](float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return (u & 0xffffu) == 0;
  });
}

}  // namespace

struct VaeDecoder::Weights {
  std::unique_ptr<Buffer> buffer;
  VkDeviceAddress bn_scale = 0, bn_mean = 0;
  Conv post_quant, conv_in, conv_out, qkv, attn_out;
  Norm attn_norm, norm_out;
  Resnet mid[2];
  Resnet up[kUpBlocks][kResnetsPerBlock];
  Conv upsampler[kUpBlocks - 1];
};

namespace {

// Packs tensors into one host array at 64-byte offsets; addresses are offsets until relocated by the buffer's address.
class Packer {
 public:
  explicit Packer(st::File& file) : file_(file) { data_.resize(16); }  // offset 0 stays "none"

  VkDeviceAddress add(const std::vector<float>& v) {
    const std::size_t at = data_.size();
    data_.insert(data_.end(), v.begin(), v.end());
    data_.resize(align_up(data_.size(), 16));
    return at * 4;
  }
  std::vector<float> read(const std::string& name, std::initializer_list<std::int64_t> shape) {
    const st::TensorInfo* t = file_.find(name);
    if (t == nullptr) throw std::runtime_error("vulkan vae: weight '" + name + "' missing");
    if (t->shape != std::vector<std::int64_t>(shape)) {
      throw std::runtime_error("vulkan vae: weight '" + name + "' has an unexpected shape");
    }
    return file_.read_f32(name);
  }
  Conv conv(const std::string& p, std::uint32_t k, std::uint32_t c, std::uint32_t r) {
    const std::vector<float> kcrs = read(p + ".weight", {k, c, r, r});
    std::vector<float> krsc(kcrs.size());
    for (std::uint32_t ki = 0; ki < k; ++ki) {
      for (std::uint32_t ci = 0; ci < c; ++ci) {
        for (std::uint32_t t = 0; t < r * r; ++t) krsc[(ki * r * r + t) * c + ci] = kcrs[(ki * c + ci) * r * r + t];
      }
    }
    Conv cv;
    cv.k = k;
    cv.c = c;
    cv.r = r;
    cv.bf16 = all_bf16(krsc);
    cv.w = add(krsc);
    cv.b = add(read(p + ".bias", {k}));
    return cv;
  }
  Norm norm(const std::string& p, std::uint32_t c) {
    Norm n;
    n.c = c;
    n.gamma = add(read(p + ".weight", {c}));
    n.beta = add(read(p + ".bias", {c}));
    return n;
  }
  Resnet resnet(const std::string& p, std::uint32_t in, std::uint32_t out) {
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
  const std::vector<float>& data() const { return data_; }

 private:
  st::File& file_;
  std::vector<float> data_;
};

void relocate(VkDeviceAddress& a, VkDeviceAddress base) {
  if (a != 0) a += base;
}
void relocate(Conv& c, VkDeviceAddress base) {
  relocate(c.w, base);
  relocate(c.b, base);
}
void relocate(Norm& n, VkDeviceAddress base) {
  relocate(n.gamma, base);
  relocate(n.beta, base);
}
void relocate(Resnet& r, VkDeviceAddress base) {
  relocate(r.norm1, base);
  relocate(r.norm2, base);
  relocate(r.conv1, base);
  relocate(r.conv2, base);
  relocate(r.shortcut, base);
}

}  // namespace

VaeDecoder::VaeDecoder(Device& device, const std::filesystem::path& path)
    : VaeDecoder(device, path, device.info().caps.coop_bf16 ? VaePath::kCoop : VaePath::kPlain) {}

VaeDecoder::VaeDecoder(Device& device, const std::filesystem::path& path, VaePath vae_path)
    : device_(device), kernels_(std::make_unique<VaeKernels>(device, vae_path)), weights_(std::make_unique<Weights>()) {
  st::File file(path);
  const st::TensorInfo* conv_in = file.find("decoder.conv_in.weight");
  if (conv_in == nullptr || conv_in->shape.size() != 4 || conv_in->shape[0] % 4 != 0 || conv_in->shape[1] != 32) {
    throw std::runtime_error("vulkan vae: " + file.path() + " has no FLUX.2 decoder.conv_in.weight [4c, 32, 3, 3]");
  }
  base_ = static_cast<int>(conv_in->shape[0] / 4);
  require(base_ % 32 == 0, "decoder widths must be multiples of 32");
  const auto c = static_cast<std::uint32_t>(base_), mid = 4 * c;
  Weights& w = *weights_;
  Packer pk(file);

  std::vector<float> mean = pk.read("bn.running_mean", {128}), var = pk.read("bn.running_var", {128});
  std::vector<float> scale(128);
  for (int i = 0; i < 128; ++i) scale[static_cast<std::size_t>(i)] = std::sqrt(var[static_cast<std::size_t>(i)] + kBatchNormEps);
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
    w.qkv = Conv{pk.add(wqkv), pk.add(bqkv), 3 * mid, mid, 1, all_bf16(wqkv)};
    const std::vector<float> wo = pk.read(a + "to_out.0.weight", {mid, mid});
    w.attn_out = Conv{pk.add(wo), pk.add(pk.read(a + "to_out.0.bias", {mid})), mid, mid, 1, all_bf16(wo)};
  }
  std::uint32_t prev = mid;
  for (int b = 0; b < kUpBlocks; ++b) {
    const std::uint32_t out = c * (b < 2 ? 4 : b == 2 ? 2 : 1);
    const std::string p = "decoder.up_blocks." + std::to_string(b);
    for (int r = 0; r < kResnetsPerBlock; ++r) {
      w.up[b][r] = pk.resnet(p + ".resnets." + std::to_string(r), r == 0 ? prev : out, out);
    }
    if (b + 1 < kUpBlocks) w.upsampler[b] = pk.conv(p + ".upsamplers.0.conv", out, out, 3);
    prev = out;
  }
  w.norm_out = pk.norm("decoder.conv_norm_out", c);
  w.conv_out = pk.conv("decoder.conv_out", 3, c, 3);

  const std::vector<float>& data = pk.data();
  w.buffer = std::make_unique<Buffer>(device_, data.size() * 4);
  w.buffer->upload(data.data(), data.size() * 4);
  const VkDeviceAddress base = w.buffer->address();
  relocate(w.bn_scale, base);
  relocate(w.bn_mean, base);
  for (Conv* cv : {&w.post_quant, &w.conv_in, &w.conv_out, &w.qkv, &w.attn_out}) relocate(*cv, base);
  for (Conv& cv : w.upsampler) relocate(cv, base);
  relocate(w.attn_norm, base);
  relocate(w.norm_out, base);
  for (Resnet& r : w.mid) relocate(r, base);
  for (auto& block : w.up) {
    for (Resnet& r : block) relocate(r, base);
  }
}

VaeDecoder::~VaeDecoder() = default;

std::uint64_t VaeDecoder::weight_bytes() const { return weights_->buffer->bytes(); }

namespace {

// The device memory of one decode.
struct Layout {
  std::uint64_t region = 0;  // bytes per ping-pong region; each holds two halves
  std::uint64_t part = 0;    // GroupNorm partials
  std::uint32_t slab = 0;    // query rows per attention slab
  std::uint64_t latents = 0, rgb = 0;
  std::uint64_t total() const { return 2 * region + align_up(part) + kAlign + latents + rgb; }
};

Layout layout(std::uint32_t c, std::uint32_t h, std::uint32_t w, std::uint64_t max_macs) {
  require(h > 0 && w > 0, "latent grid must be at least 1x1");
  const std::uint64_t full = 256ull * h * w, latent_px = 4ull * h * w, mid = 4ull * c;
  Layout l;
  // Attention scratch: xn [L, mid], qkv [L, 3 mid], V^T [mid, L], then query-row slabs of scores [rows, L].
  const std::uint64_t l_tokens = latent_px;
  const std::uint64_t budget_rows = std::max<std::uint64_t>(VaeKernels::kRowAlign, max_macs / (l_tokens * mid) /
                                                                                       VaeKernels::kRowAlign *
                                                                                       VaeKernels::kRowAlign);
  l.slab = static_cast<std::uint32_t>(std::min<std::uint64_t>(budget_rows, align_up(l_tokens, VaeKernels::kRowAlign)));
  const std::uint64_t attention = 3 * align_up(l_tokens * mid * 4) + align_up(l_tokens * 3 * mid * 4) +
                                  align_up(static_cast<std::uint64_t>(l.slab) * l_tokens * 4);
  const std::uint64_t tail = 2ull * c * full * 4;  // up_blocks[2]'s upsampler output: 2c channels at full resolution
  l.region = align_up(std::max(tail, attention), 2 * kAlign);
  // Every GroupNorm input: {pixels, channels}.
  const std::uint64_t norms[][2] = {{latent_px, mid}, {4 * latent_px, mid}, {16 * latent_px, mid}, {16 * latent_px, 2ull * c},
                                    {full, 2ull * c},  {full, c}};
  for (const auto& g : norms) {
    l.part = std::max(l.part, VaeKernels::gn_part_bytes(static_cast<std::uint32_t>(g[0]), static_cast<std::uint32_t>(g[1])));
  }
  l.latents = align_up(128ull * h * w * 4);
  l.rgb = align_up(full * 3);
  return l;
}

// Groups recorded ops into submits of at most `budget` multiply-adds (elementwise ops count kElementCost per element).
// With a profile, every op is its own submit and its wall time is added to the profile under the op's name.
class Submitter {
 public:
  static constexpr std::uint64_t kElementCost = 32;
  Submitter(Device& device, std::uint64_t budget, std::vector<std::pair<std::string, double>>* profile)
      : device_(device), budget_(budget), profile_(profile) {}
  void add(const std::string& name, std::uint64_t cost, std::function<void(VkCommandBuffer)> op) {
    if (profile_ != nullptr) {
      const auto t0 = std::chrono::steady_clock::now();
      device_.run(op);
      ++submits_;
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      auto it = std::find_if(profile_->begin(), profile_->end(), [&](const auto& e) { return e.first == name; });
      if (it == profile_->end()) {
        profile_->emplace_back(name, s);
      } else {
        it->second += s;
      }
      return;
    }
    if (!ops_.empty() && cost_ + cost > budget_) flush();
    ops_.push_back(std::move(op));
    cost_ += cost;
  }
  void flush() {
    if (ops_.empty()) return;
    device_.run([&](VkCommandBuffer cb) {
      for (const auto& op : ops_) op(cb);
    });
    ops_.clear();
    cost_ = 0;
    ++submits_;
  }
  int submits() const { return submits_; }

 private:
  Device& device_;
  std::uint64_t budget_;
  std::vector<std::pair<std::string, double>>* profile_;
  std::vector<std::function<void(VkCommandBuffer)>> ops_;
  std::uint64_t cost_ = 0;
  int submits_ = 0;
};

std::string dims(std::uint32_t h, std::uint32_t w) { return " @" + std::to_string(h) + "x" + std::to_string(w); }

class Run {
 public:
  Run(const VaeKernels& k, Submitter& s, std::uint64_t max_macs, VkDeviceAddress part, VkDeviceAddress stats)
      : k_(k), s_(s), max_macs_(max_macs), part_(part), stats_(stats) {}

  // A convolution or GEMM in bands of output rows, each at most max_macs multiply-adds (at least one row tile).
  void gemm(const std::string& name, ConvArgs a, VaePrecision precision) {
    const std::uint64_t per_row = static_cast<std::uint64_t>(a.n) * a.k;
    // 65535 row tiles of 128 is what one dispatch addresses.
    const std::uint64_t rows = std::min<std::uint64_t>(
        65535ull * VaeKernels::kRowAlign,
        std::max<std::uint64_t>(VaeKernels::kRowAlign, max_macs_ / per_row / VaeKernels::kRowAlign * VaeKernels::kRowAlign));
    const std::uint32_t end = a.m;
    for (std::uint64_t m0 = a.m0; m0 < end; m0 += rows) {
      a.m0 = static_cast<std::uint32_t>(m0);
      a.m = static_cast<std::uint32_t>(std::min<std::uint64_t>(end, m0 + rows));
      s_.add(name, (a.m - a.m0) * per_row, [this, a, precision](VkCommandBuffer cb) { k_.record_conv(cb, a, precision); });
    }
  }
  void conv(const Conv& cv, VkDeviceAddress x, std::uint32_t h, std::uint32_t w, VkDeviceAddress y,
            VkDeviceAddress residual = 0, bool upsample = false) {
    ConvArgs a;
    a.a = x;
    a.b = cv.w;
    a.out = y;
    a.bias = cv.b;
    a.residual = residual;
    a.m = h * w;
    a.n = cv.k;
    a.k = cv.r * cv.r * cv.c;
    a.ldb = a.k;
    a.ldo = cv.k;
    a.taps = cv.r * cv.r;
    a.upsample = upsample;
    a.h = h;
    a.w = w;
    a.c = cv.c;
    a.b_bf16 = cv.bf16;
    const std::string name = "conv " + std::to_string(cv.r) + "x" + std::to_string(cv.r) + " " + std::to_string(cv.c) + "->" +
                             std::to_string(cv.k) + (upsample ? " upsampled" : "") + dims(h, w);
    gemm(name, a, VaePrecision::kTf32);
  }
  void norm(const Norm& n, VkDeviceAddress x, std::uint32_t pixels, VkDeviceAddress y, bool silu) {
    const std::uint64_t cost = static_cast<std::uint64_t>(pixels) * n.c * Submitter::kElementCost;
    const std::string shape = " " + std::to_string(n.c) + " ch, " + std::to_string(pixels) + " px";
    s_.add("group_norm stats" + shape, cost, [this, x, pixels, c = n.c](VkCommandBuffer cb) { k_.record_gn_stats(cb, x, pixels, c, part_, stats_, kResnetEps); });
    s_.add(std::string(silu ? "group_norm+silu apply" : "group_norm apply") + shape, cost, [this, n, x, y, pixels, silu](VkCommandBuffer cb) {
      k_.record_gn_apply(cb, x, y, pixels, n.c, stats_, n.gamma, n.beta, silu);
    });
  }
  // ResnetBlock2D (as Flux2VaeDecoder::resnet): returns the buffer holding the output, x or a.
  VkDeviceAddress resnet(const Resnet& r, VkDeviceAddress x, VkDeviceAddress a, VkDeviceAddress b, std::uint32_t h,
                         std::uint32_t w) {
    const std::uint32_t px = h * w;
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
  // x [L, mid] += to_out(SDPA(to_q/k/v(GN(x)))), scratch in ws (a region).
  void attention(const VaeDecoder::Weights& w, VkDeviceAddress x, VkDeviceAddress ws, std::uint32_t l_tokens,
                 std::uint32_t mid, std::uint32_t slab) {
    const VkDeviceAddress xn = ws, qkv = xn + align_up(static_cast<std::uint64_t>(l_tokens) * mid * 4),
                          vt = qkv + align_up(static_cast<std::uint64_t>(l_tokens) * 3 * mid * 4),
                          scores = vt + align_up(static_cast<std::uint64_t>(l_tokens) * mid * 4);
    norm(w.attn_norm, x, l_tokens, xn, false);
    ConvArgs p;
    p.a = xn;
    p.b = w.qkv.w;
    p.bias = w.qkv.b;
    p.out = qkv;
    p.m = l_tokens;
    p.n = 3 * mid;
    p.k = mid;
    p.lda = mid;
    p.ldb = mid;
    p.ldo = 3 * mid;
    p.b_bf16 = w.qkv.bf16;
    gemm("attention qkv", p, VaePrecision::kFp32);
    s_.add("attention transpose V", static_cast<std::uint64_t>(l_tokens) * mid * Submitter::kElementCost, [this, qkv, vt, l_tokens, mid](VkCommandBuffer cb) {
      k_.record_transpose(cb, qkv + 2ull * mid * 4, vt, l_tokens, mid, 3 * mid, l_tokens);
    });
    // SDPA's scale as the engine forms it: 1 / sqrt(E) in fp32.
    const float scale = 1.0f / std::sqrt(static_cast<float>(mid));
    const VkDeviceAddress o = xn;  // GN output is dead after the projections
    for (std::uint32_t r0 = 0; r0 < l_tokens; r0 += slab) {
      const std::uint32_t rows = std::min(slab, l_tokens - r0);
      ConvArgs s;
      s.a = qkv + static_cast<std::uint64_t>(r0) * 3 * mid * 4;
      s.b = qkv + static_cast<std::uint64_t>(mid) * 4;
      s.out = scores;
      s.m = rows;
      s.n = l_tokens;
      s.k = mid;
      s.lda = 3 * mid;
      s.ldb = 3 * mid;
      s.ldo = l_tokens;
      gemm("attention scores", s, VaePrecision::kSplit16);
      s_.add("attention softmax", static_cast<std::uint64_t>(rows) * l_tokens * Submitter::kElementCost,
             [this, scores, rows, l_tokens, scale](VkCommandBuffer cb) { k_.record_softmax(cb, scores, rows, l_tokens, l_tokens, scale); });
      ConvArgs pv;
      pv.a = scores;
      pv.b = vt;
      pv.out = o + static_cast<std::uint64_t>(r0) * mid * 4;
      pv.m = rows;
      pv.n = mid;
      pv.k = l_tokens;
      pv.lda = l_tokens;
      pv.ldb = l_tokens;
      pv.ldo = mid;
      gemm("attention P V", pv, VaePrecision::kSplit16);
    }
    // Reference order: round(to_out(o) + bias), then + residual.
    ConvArgs out;
    out.a = o;
    out.b = w.attn_out.w;
    out.bias = w.attn_out.b;
    out.residual = x;
    out.out = x;
    out.m = l_tokens;
    out.n = mid;
    out.k = mid;
    out.lda = mid;
    out.ldb = mid;
    out.ldo = mid;
    out.b_bf16 = w.attn_out.bf16;
    gemm("attention out", out, VaePrecision::kFp32);
  }

 private:
  const VaeKernels& k_;
  Submitter& s_;
  std::uint64_t max_macs_;
  VkDeviceAddress part_, stats_;
};

}  // namespace

std::uint64_t VaeDecoder::workspace_bytes(std::uint32_t h, std::uint32_t w) const {
  return layout(static_cast<std::uint32_t>(base_), h, w, max_macs_).total();
}

std::uint64_t VaeDecoder::largest_buffer_bytes(std::uint32_t h, std::uint32_t w) const {
  return layout(static_cast<std::uint32_t>(base_), h, w, max_macs_).region;
}

VaeDecodeStats VaeDecoder::decode(const float* latents, std::uint32_t h, std::uint32_t w, std::vector<std::uint8_t>& rgb,
                                  std::vector<float>* image) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto c = static_cast<std::uint32_t>(base_), mid = 4 * c;
  const Layout l = layout(c, h, w, max_macs_);
  const std::uint32_t hh = 2 * h, ww = 2 * w, H = 16 * h, W = 16 * w;
  require(static_cast<std::uint64_t>(H) * W * 3 < (1ull << 32), "image too large");
  Buffer r0(device_, l.region), r1(device_, l.region), aux(device_, align_up(l.part) + kAlign),
      z(device_, l.latents), out(device_, l.rgb);
  z.upload(latents, 128ull * h * w * 4);
  const VkDeviceAddress part = aux.address(), stats = aux.address() + align_up(l.part);
  const std::uint64_t half = l.region / 2;
  const Weights& wt = *weights_;

  VaeDecodeStats st;
  Submitter sub(device_, max_macs_, profile_ ? &st.profile : nullptr);
  Run run(*kernels_, sub, max_macs_, part, stats);
  VkDeviceAddress p = r0.address(), q = r1.address();

  // Head at H/8: unpack, post_quant_conv, conv_in, mid resnet 0, mid attention.
  const VkDeviceAddress unpacked = q, pq = q + half;
  sub.add("latent unpack", 128ull * h * w * Submitter::kElementCost, [&, unpacked](VkCommandBuffer cb) {
    kernels_->record_latent_unpack(cb, z.address(), wt.bn_scale, wt.bn_mean, unpacked, h, w);
  });
  run.conv(wt.post_quant, unpacked, hh, ww, pq);
  run.conv(wt.conv_in, pq, hh, ww, p);
  run.resnet(wt.mid[0], p, q, q + half, hh, ww);
  run.attention(wt, p, q, hh * ww, mid, l.slab);

  // Tail: mid resnet 1, the up blocks, norm_out + SiLU, conv_out, RGB8.
  std::uint32_t ch = hh, cw = ww;
  run.resnet(wt.mid[1], p, q, q + half, ch, cw);
  for (int b = 0; b < kUpBlocks; ++b) {
    VkDeviceAddress x = p;
    for (int r = 0; r < kResnetsPerBlock; ++r) {
      const VkDeviceAddress other = x == p ? q : p;
      x = run.resnet(wt.up[b][r], x, other, other + half, ch, cw);
    }
    if (b + 1 < kUpBlocks) {
      const VkDeviceAddress y = x == p ? q : p;
      ch *= 2;
      cw *= 2;
      run.conv(wt.upsampler[b], x, ch, cw, y, 0, true);
      x = y;
    }
    p = x;
    q = x == r0.address() ? r1.address() : r0.address();
  }
  run.norm(wt.norm_out, p, ch * cw, p, true);
  run.conv(wt.conv_out, p, ch, cw, q);
  const std::uint32_t n = H * W * 3;
  sub.add("rgb8", static_cast<std::uint64_t>(n) * Submitter::kElementCost,
          [&, q, n](VkCommandBuffer cb) { kernels_->record_rgb8(cb, q, out.address(), n); });
  sub.flush();

  rgb.resize(n);
  {
    std::vector<std::uint8_t> words(l.rgb);
    out.download(words.data(), l.rgb);
    std::memcpy(rgb.data(), words.data(), n);
  }
  if (image != nullptr) {
    // conv_out's output starts its region.
    image->resize(n);
    (q == r0.address() ? r0 : r1).download(image->data(), static_cast<std::uint64_t>(n) * 4);
  }
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  st.device_bytes = l.total();
  st.submits = sub.submits();
  return st;
}

}  // namespace mi::vk
