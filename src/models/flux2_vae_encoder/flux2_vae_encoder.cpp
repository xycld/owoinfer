#include "models/flux2_vae_encoder/flux2_vae_encoder.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/vae/attention.h"
#include "models/flux2_vae/vae_file.h"
#include "pipeline/rng_torch_compat.h"
#include "util/path.h"
#include "weights/file_reader.h"
#include "weights/model_index.h"
#include "weights/safetensors.h"

namespace mi::flux2_vae_encoder {
namespace {

using runtime::align_up;
using Conv = Flux2VaeEncoder::Conv;
using Norm = Flux2VaeEncoder::Norm;
using Resnet = Flux2VaeEncoder::Resnet;

constexpr int kGroups = vae::kGroupNormGroups;
constexpr int kLevelIn[kLevels] = {4, 128, 256};   // level input channels (the image padded to 4)
constexpr int kLevelOut[kLevels] = {128, 256, 512};
constexpr int kTailCh = 512;
constexpr int kMomentCh = 64;  // 2 x 32 latent channels (mean | logvar)
constexpr int kNormsPerLevel = 4;
constexpr int kConvsPerLevel = 6;
// CUTLASS's conv iterators use 32-bit byte offsets: tensors of 2 GiB or more (128-channel full-resolution fp32
// tensors from 2048^2 up) are convolved in row strips through a scratch.
constexpr std::size_t kCutlassMaxBytes = std::size_t{1} << 31;
constexpr std::size_t kStripScratchBytes = std::size_t{128} << 20;

[[noreturn]] void fail(const std::string& msg) { throw std::invalid_argument("mi::flux2_vae_encoder: " + msg); }

Tensor act(void* p, std::int64_t h, std::int64_t w, std::int64_t c) {
  return Tensor::view(p, DType::kF32, Device::kCUDA, {1, h, w, c});
}
std::size_t act_bytes(std::int64_t h, std::int64_t w, std::int64_t c) { return static_cast<std::size_t>(h * w * c) * 4; }

// ---------------------------------------------------------------------------------------------------------------
// Level programs. Slots hold one tensor each at the level's resolution (kOut: the output checkpoint at half
// resolution). Ops in reference order; conv/norm ids index Weights::level_convs[l] / level_norms[l].
// ---------------------------------------------------------------------------------------------------------------
enum Slot : int { kNone = -1, kIn = 0, kX, kA, kB, kOut, kNumSlots };
enum class Kind { kConv, kNorm, kDown };
struct Op {
  Kind kind;
  int in, out, res;  // slots (res: residual added in the conv epilogue)
  int id;            // conv or norm index
  int in_ch, out_ch, r;
};

std::vector<Op> program(int l) {
  const int ci = kLevelIn[l], co = kLevelOut[l];
  std::vector<Op> p;
  const auto conv = [&](int in, int out, int res, int id, int a, int b, int r) { p.push_back({Kind::kConv, in, out, res, id, a, b, r}); };
  const auto norm = [&](int in, int out, int id, int c) { p.push_back({Kind::kNorm, in, out, kNone, id, c, c, 1}); };
  if (l == 0) {
    conv(kIn, kX, kNone, 0, ci, co, 3);  // conv_in
    for (int r = 0; r < 2; ++r) {       // ResnetBlock2D, same channels: x = conv2(SiLU(GN(conv1(SiLU(GN(x)))))) + x
      norm(kX, kA, 2 * r, co);
      conv(kA, kB, kNone, 1 + 2 * r, co, co, 3);
      norm(kB, kB, 2 * r + 1, co);
      conv(kB, kX, kX, 2 + 2 * r, co, co, 3);
    }
    p.push_back({Kind::kDown, kX, kOut, kNone, 5, co, co, 3});
  } else {
    // Resnet ci -> co: the 1x1 shortcut first, so the first GroupNorm can overwrite its input (as the decoder).
    conv(kIn, kA, kNone, 0, ci, co, 1);
    norm(kIn, kIn, 0, ci);
    conv(kIn, kB, kNone, 1, ci, co, 3);
    norm(kB, kB, 1, co);
    conv(kB, kA, kA, 2, co, co, 3);
    norm(kA, kX, 2, co);
    conv(kX, kB, kNone, 3, co, co, 3);
    norm(kB, kB, 3, co);
    conv(kB, kA, kA, 4, co, co, 3);
    p.push_back({Kind::kDown, kA, kOut, kNone, 5, co, co, 3});
  }
  return p;
}

bool writes_input(const std::vector<Op>& prog) {
  for (const Op& op : prog)
    if (op.out == kIn) return true;
  return false;
}

struct Range {
  std::int64_t a = 0, b = 0;
  bool empty() const { return b <= a; }
  std::int64_t size() const { return b - a; }
};
Range hull(Range x, Range y) {
  if (x.empty()) return y;
  if (y.empty()) return x;
  return {std::min(x.a, y.a), std::max(x.b, y.b)};
}
bool contains(Range outer, Range inner) { return inner.empty() || (outer.a <= inner.a && inner.b <= outer.b); }

// One level's execution: a dry run (weights == nullptr) only measures slot sizes, scratch and MACs; a real run
// enqueues the kernels. Both walk the same code.
struct LevelExec {
  int level = 0;
  std::int64_t h = 0, w = 0;  // level resolution
  const std::vector<Conv>* convs = nullptr;
  const std::vector<Norm>* norms = nullptr;
  bool alias_in = false;  // kIn is the input checkpoint itself (no copy)
  float* in_ckpt = nullptr;
  float* out_ckpt = nullptr;
  char* slot_base[kNumSlots] = {};
  float* stats = nullptr;  // [kNormsPerLevel][groups][2]
  void* accum = nullptr;
  void* aux = nullptr;
  std::size_t aux_bytes = 0;
  void* strip = nullptr;
  vae::ConvMath math = vae::ConvMath::kTf32x3;
  cudaStream_t stream = nullptr;
  // Measured by the dry run.
  std::size_t slot_need[kNumSlots] = {};
  std::size_t aux_need = 0;
  bool needs_strip = false;
  double macs = 0;

  bool dry() const { return convs == nullptr; }
};

struct SlotState {
  char* base = nullptr;
  std::int64_t origin = 0;  // image row (at the slot's resolution) of the buffer's first row
  int ch = 0;
  Range stored, valid;
};

Tensor rows_view(const SlotState& s, std::int64_t w, Range r) {
  char* p = s.base == nullptr ? nullptr : s.base + static_cast<std::size_t>(r.a - s.origin) * act_bytes(1, w, s.ch);
  return act(p, r.size(), w, s.ch);
}

Tensor conv_weight_view(const LevelExec& e, const Op& op) {
  if (!e.dry()) return (*e.convs)[static_cast<std::size_t>(op.id)].w;
  return Tensor::view(nullptr, DType::kF32, Device::kCUDA, {op.out_ch, op.r, op.r, op.in_ch});
}

Tensor rows_of(const Tensor& t, std::int64_t a, std::int64_t b) {
  char* p = t.data == nullptr ? nullptr : static_cast<char*>(t.data) + static_cast<std::size_t>(a) * act_bytes(1, t.dim(2), t.dim(3));
  return act(p, b - a, t.dim(2), t.dim(3));
}

// Stride-1 conv over whole views; views CUTLASS cannot address at once run in row strips with a halo through the
// strip scratch (bit-identical: each output pixel's accumulation does not depend on where its tile lies; the
// residual is elementwise, so its halo rows only reach discarded outputs and y may alias it).
void conv_rows(LevelExec& e, const Tensor& x, const Op& op, const Tensor* res, const Tensor& y) {
  const Tensor w = conv_weight_view(e, op);
  const auto call = [&](const Tensor& xs, const Tensor* rs, const Tensor& ys) {
    if (e.dry()) {
      for (const auto m : {vae::ConvMath::kTf32, vae::ConvMath::kTf32x3})
        e.aux_need = std::max(e.aux_need, vae::encoder_conv2d_workspace_bytes(xs, w, m));
      return;
    }
    vae::encoder_conv2d(xs, w, (*e.convs)[static_cast<std::size_t>(op.id)].bias, rs, ys, e.aux, e.aux_bytes, e.math, e.stream);
  };
  if (x.nbytes() < kCutlassMaxBytes && y.nbytes() < kCutlassMaxBytes) {
    call(x, res, y);
    return;
  }
  e.needs_strip = true;
  const std::int64_t rows = x.dim(1), halo = (op.r - 1) / 2;
  const std::size_t xrow = act_bytes(1, x.dim(2), x.dim(3)), yrow = act_bytes(1, y.dim(2), y.dim(3));
  const std::size_t max_rows = std::min(kStripScratchBytes / yrow, (kCutlassMaxBytes - 1) / std::max(xrow, yrow));
  const std::int64_t step = static_cast<std::int64_t>(max_rows) - 2 * halo;
  if (step < 1) fail("image too wide for the conv strip scratch");
  for (std::int64_t r0 = 0; r0 < rows; r0 += step) {
    const std::int64_t r1 = std::min(rows, r0 + step);
    const std::int64_t a = std::max<std::int64_t>(0, r0 - halo), b = std::min(rows, r1 + halo);
    const Tensor rb = res != nullptr ? rows_of(*res, a, b) : Tensor{};
    if (halo == 0) {
      call(rows_of(x, a, b), res != nullptr ? &rb : nullptr, rows_of(y, a, b));
      continue;
    }
    const Tensor tmp = act(e.strip, b - a, y.dim(2), y.dim(3));
    call(rows_of(x, a, b), res != nullptr ? &rb : nullptr, tmp);
    if (!e.dry())
      MI_CUDA_CHECK(cudaMemcpyAsync(rows_of(y, r0, r1).data, static_cast<char*>(e.strip) + static_cast<std::size_t>(r0 - a) * yrow,
                                    static_cast<std::size_t>(r1 - r0) * yrow, cudaMemcpyDeviceToDevice, e.stream));
  }
}

// Stride-2 conv; output strips read input rows 2i .. 2i+2 directly (no scratch).
void down_rows(LevelExec& e, const Tensor& x, const Op& op, const Tensor& y) {
  const Tensor w = conv_weight_view(e, op);
  const auto call = [&](const Tensor& xs, const Tensor& ys) {
    if (e.dry()) {
      for (const auto m : {vae::ConvMath::kTf32, vae::ConvMath::kTf32x3})
        e.aux_need = std::max(e.aux_need, vae::conv2d_down_workspace_bytes(xs, w, ys, m));
      return;
    }
    vae::conv2d_down(xs, w, (*e.convs)[static_cast<std::size_t>(op.id)].bias, ys, e.aux, e.aux_bytes, e.math, e.stream);
  };
  const std::int64_t xrows = x.dim(1), yrows = y.dim(1);
  const std::size_t xrow = act_bytes(1, x.dim(2), x.dim(3));
  const std::int64_t step = x.nbytes() < kCutlassMaxBytes ? yrows : static_cast<std::int64_t>((kCutlassMaxBytes - 1) / xrow - 1) / 2;
  if (step < 1) fail("image too wide for the downsample conv strips");
  for (std::int64_t i0 = 0; i0 < yrows; i0 += step) {
    const std::int64_t i1 = std::min(yrows, i0 + step);
    call(rows_of(x, 2 * i0, std::min(xrows, 2 * i1 + 1)), rows_of(y, i0, i1));
  }
}

// Runs ops [0, n) of a level program for the rows `final_rows` of slot `final_slot` (its value after op n-1) and
// returns that view. Every op computes only the rows its consumers need (a backward pass over the program), and
// the forward pass checks that every row it reads is valid: stored, and not next to a band edge it cannot see.
// untiled: one whole-tensor band, the GroupNorms compute their statistics inline; otherwise they use stats[id].
Tensor run_ops(LevelExec& e, const std::vector<Op>& prog, std::size_t n, int final_slot, Range final_rows, bool untiled) {
  const std::int64_t h = e.h, w = e.w, ho = h / 2, wo = w / 2;
  // Backward: rows each op reads (in_rows) and writes (out_rows).
  std::vector<Range> in_rows(n), out_rows(n);
  std::vector<char> live(n, 0);
  Range need[kNumSlots];
  need[final_slot] = final_rows;
  for (std::size_t i = n; i-- > 0;) {
    const Op& op = prog[i];
    const Range ro = need[op.out];
    if (ro.empty()) continue;  // dead in this pass (e.g. the shortcut during the first GroupNorm's statistics)
    live[i] = 1;
    need[op.out] = {};
    Range ri;
    if (op.kind == Kind::kDown) {
      ri = {2 * ro.a, std::min(h, 2 * ro.b + 1)};
      out_rows[i] = ro;
    } else {
      ri = op.r == 3 ? Range{std::max<std::int64_t>(0, ro.a - 1), std::min(h, ro.b + 1)} : ro;
      out_rows[i] = ri;
    }
    in_rows[i] = ri;
    need[op.in] = hull(need[op.in], ri);
    if (op.res != kNone) need[op.res] = hull(need[op.res], ri);
  }
  for (int s = kX; s < kOut; ++s)
    if (!need[s].empty()) fail("internal: level program reads a slot before writing it");

  SlotState st[kNumSlots];
  const Range in_need = need[kIn];
  if (!in_need.empty()) {
    SlotState& s = st[kIn];
    s.ch = kLevelIn[e.level];
    if (e.alias_in) {
      s.base = reinterpret_cast<char*>(e.in_ckpt);
      s.origin = 0;
      s.stored = s.valid = {0, h};
    } else {
      s.base = e.slot_base[kIn];
      s.origin = in_need.a;
      s.stored = s.valid = in_need;
      const std::size_t bytes = act_bytes(in_need.size(), w, s.ch);
      e.slot_need[kIn] = std::max(e.slot_need[kIn], bytes);
      if (!e.dry())
        MI_CUDA_CHECK(cudaMemcpyAsync(s.base, reinterpret_cast<char*>(e.in_ckpt) + static_cast<std::size_t>(in_need.a) * act_bytes(1, w, s.ch),
                                      bytes, cudaMemcpyDeviceToDevice, e.stream));
    }
  }
  st[kOut] = {reinterpret_cast<char*>(e.out_ckpt), 0, kLevelOut[e.level], {0, ho}, {0, ho}};

  for (std::size_t i = 0; i < n; ++i) {
    if (!live[i]) continue;
    const Op& op = prog[i];
    const Range v = in_rows[i], r = out_rows[i];
    SlotState& in = st[op.in];
    if (!contains(in.valid, v) || (op.res != kNone && !contains(st[op.res].valid, v)))
      fail("internal: band reads rows outside the valid range");
    const Tensor x = rows_view(in, w, v);
    SlotState& out = st[op.out];
    Range valid;
    if (op.kind == Kind::kDown) {
      // Output row i reads rows 2i .. 2i+2; rows at or past H are the reference's zero pad.
      valid = {r.a, v.b == h ? r.b : std::min(r.b, (v.b - 3) / 2 + 1)};
      if (!e.dry()) down_rows(e, x, op, rows_view(out, wo, r));
      else down_rows(e, x, op, act(nullptr, r.size(), wo, op.out_ch));
      e.macs += static_cast<double>(r.size()) * static_cast<double>(wo) * op.out_ch * op.in_ch * 9;
      out.valid = valid;
      continue;
    }
    const bool in_place = op.out == op.in || op.out == op.res;
    if (in_place) {
      if (!contains(out.stored, r) || out.ch != op.out_ch) fail("internal: in-place op outside its slot");
    } else {
      out.base = e.slot_base[op.out];
      out.origin = r.a;
      out.ch = op.out_ch;
      out.stored = r;
    }
    e.slot_need[op.out] = std::max(e.slot_need[op.out], act_bytes(r.b - out.origin, w, out.ch));
    const Tensor y = rows_view(out, w, r);
    if (op.kind == Kind::kNorm) {
      valid = v;
      if (!e.dry()) {
        const Norm& nm = (*e.norms)[static_cast<std::size_t>(op.id)];
        const Tensor stats = Tensor::view(e.stats + op.id * kGroups * 2, DType::kF32, Device::kCUDA, {1, kGroups, 2});
        if (untiled) {
          if (v.a != 0 || v.b != h) fail("internal: untiled GroupNorm on a partial tensor");
          vae::group_norm_stats(x, kGroups, vae::kResnetEps, stats, e.aux, e.aux_bytes, e.stream);
        }
        vae::group_norm_apply(x, stats, nm.gamma, nm.beta, /*silu=*/true, y, e.stream);
      } else if (untiled) {
        e.aux_need = std::max(e.aux_need, vae::group_norm_workspace_bytes(x, kGroups));
      }
    } else {
      // 3x3: output row i is exact when rows i-1 .. i+1 were read or lie outside the image.
      valid = op.r == 3 ? Range{v.a == 0 ? 0 : v.a + 1, v.b == h ? h : v.b - 1} : v;
      Tensor res;
      if (op.res != kNone) res = rows_view(st[op.res], w, v);
      conv_rows(e, x, op, op.res != kNone ? &res : nullptr, y);
      e.macs += static_cast<double>(r.size()) * static_cast<double>(w) * op.out_ch * op.in_ch * op.r * op.r;
    }
    out.valid = valid;
  }
  if (!contains(st[final_slot].valid, final_rows)) fail("internal: band output rows not valid");
  if (final_slot == kOut) return Tensor{};
  return rows_view(st[final_slot], w, final_rows);
}

std::size_t norm_op_index(const std::vector<Op>& prog, int id) {
  for (std::size_t i = 0; i < prog.size(); ++i)
    if (prog[i].kind == Kind::kNorm && prog[i].id == id) return i;
  fail("internal: GroupNorm missing from the level program");
}

// band == 0: untiled. Otherwise one statistics pass per GroupNorm over band-row cores of its input, then the
// output pass over cores of band/2 output rows.
void run_level(LevelExec& e, std::int64_t band) {
  const std::vector<Op> prog = program(e.level);
  if (band == 0) {
    run_ops(e, prog, prog.size(), kOut, {0, e.h / 2}, true);
    return;
  }
  for (int j = 0; j < kNormsPerLevel; ++j) {
    const std::size_t idx = norm_op_index(prog, j);
    const Op& nop = prog[idx];
    if (!e.dry()) vae::group_norm_accum_zero(e.accum, 1, kGroups, e.stream);
    for (std::int64_t c0 = 0; c0 < e.h; c0 += band) {
      const Tensor core = run_ops(e, prog, idx, nop.in, {c0, std::min(e.h, c0 + band)}, false);
      if (!e.dry()) vae::group_norm_accumulate(core, kGroups, e.accum, e.stream);
    }
    if (!e.dry()) {
      const Tensor stats = Tensor::view(e.stats + j * kGroups * 2, DType::kF32, Device::kCUDA, {1, kGroups, 2});
      vae::group_norm_finalize(e.accum, 1, kGroups, e.h * e.w * nop.in_ch / kGroups, vae::kResnetEps, stats, e.stream);
    }
  }
  const std::int64_t ob = band / 2;
  for (std::int64_t o0 = 0; o0 < e.h / 2; o0 += ob) run_ops(e, prog, prog.size(), kOut, {o0, std::min(e.h / 2, o0 + ob)}, false);
}

// Stack layout of one level's scratch: statistics, accumulator, conv/GroupNorm scratch, strip scratch, slots.
struct LevelLayout {
  std::size_t stats = 0, accum = 0, aux = 0, strip = 0, slot[kNumSlots] = {}, total = 0;
  double macs = 0;
};

LevelLayout level_layout(int level, std::int64_t height, std::int64_t width, std::int64_t band) {
  LevelExec e;
  e.level = level;
  e.h = height >> level;
  e.w = width >> level;
  e.alias_in = band == 0 || !writes_input(program(level));
  run_level(e, band);
  LevelLayout l;
  std::size_t off = 0;
  l.stats = off;
  off += align_up(kNormsPerLevel * kGroups * 2 * sizeof(float));
  l.accum = off;
  off += align_up(vae::group_norm_accum_bytes(1, kGroups));
  l.aux = off;
  off += align_up(std::max<std::size_t>(e.aux_need, 1));
  l.strip = off;
  if (e.needs_strip) off += kStripScratchBytes;
  for (int s = kIn; s < kOut; ++s) {
    l.slot[s] = off;
    off += align_up(e.slot_need[s]);
  }
  l.total = off;
  l.macs = e.macs;
  return l;
}

// ---------------------------------------------------------------------------------------------------------------
// Tail (untiled, H/8): X = C3 in place; A, B 512-channel slots; the attention scratch overlaps A and B.
// ---------------------------------------------------------------------------------------------------------------
struct TailLayout {
  std::size_t a = 0, b = 0, attn = 0, attn_bytes = 0, aux = 0, aux_bytes = 0, noise = 0, total = 0;
  double macs = 0;
};

TailLayout tail_layout(std::int64_t height, std::int64_t width) {
  const std::int64_t h = height / 8, w = width / 8;
  TailLayout l;
  const std::size_t slot = align_up(act_bytes(h, w, kTailCh));
  const Tensor x = act(nullptr, h, w, kTailCh);
  l.attn_bytes = vae::mid_attention_workspace_bytes(x);
  l.a = 0;
  l.b = slot;
  l.attn = 0;
  std::size_t aux = vae::group_norm_workspace_bytes(x, kGroups);
  const auto wt = [](int k, int r, int c) { return Tensor::view(nullptr, DType::kF32, Device::kCUDA, {k, r, r, c}); };
  for (const auto m : {vae::ConvMath::kTf32, vae::ConvMath::kTf32x3}) {
    aux = std::max(aux, vae::encoder_conv2d_workspace_bytes(x, wt(kTailCh, 3, kTailCh), m));
    aux = std::max(aux, vae::encoder_conv2d_workspace_bytes(x, wt(kMomentCh, 3, kTailCh), m));
    aux = std::max(aux, vae::encoder_conv2d_workspace_bytes(act(nullptr, h, w, kMomentCh), wt(kMomentCh, 1, kMomentCh), m));
  }
  l.aux = align_up(std::max(2 * slot, l.attn_bytes));
  l.aux_bytes = std::max<std::size_t>(aux, 1);
  l.noise = l.aux + align_up(l.aux_bytes);
  l.total = l.noise + align_up(act_bytes(h, w, 32));
  const double hw = static_cast<double>(h * w);
  l.macs = hw * (8.0 * kTailCh * kTailCh * 9 + 4.0 * kTailCh * kTailCh + kMomentCh * kTailCh * 9.0 + kMomentCh * kMomentCh) +
           2.0 * hw * hw * kTailCh;
  return l;
}

// Whole-encode stack layout: lo region [0, lo) holds the image, then C2; hi region [lo + mid, total) holds C1, then
// C3; the middle holds each phase's scratch (host-input staging, level scratch); the tail uses [0, lo + mid).
struct EncodeLayout {
  std::size_t lo = 0, mid = 0, hi = 0, total = 0;
  LevelLayout levels[kLevels];
  TailLayout tail;
  double macs = 0;
};

// vae::group_norm caps a group at 2^30 elements: level 0's groups hold 4 channels of the full image (any banding).
constexpr std::int64_t kMaxPixels = std::int64_t{1} << 28;

void check_size(std::int64_t height, std::int64_t width) {
  if (height < kPixelsPerLatent || width < kPixelsPerLatent || height % kPixelsPerLatent != 0 || width % kPixelsPerLatent != 0)
    fail("image size must be a positive multiple of 16, got " + std::to_string(width) + "x" + std::to_string(height));
  if (height > kMaxPixels / width) fail("image larger than 2^28 pixels (GroupNorm group size limit)");
}

// Band rows per level: 0 (untiled) or even and >= 2 (an output band is band / 2 >= 1 rows, so every pass advances).
void check_bands(const std::array<std::int64_t, kLevels>& band_rows) {
  for (const std::int64_t b : band_rows)
    if (b != 0 && (b < 2 || b % 2 != 0)) fail("band rows must be 0 (untiled) or even and >= 2, got " + std::to_string(b));
}

EncodeLayout encode_layout(std::int64_t height, std::int64_t width, const std::array<std::int64_t, kLevels>& bands) {
  check_size(height, width);
  EncodeLayout l;
  const std::size_t img = align_up(act_bytes(height, width, 4));
  const std::size_t c1 = align_up(act_bytes(height / 2, width / 2, kLevelOut[0]));
  const std::size_t c2 = align_up(act_bytes(height / 4, width / 4, kLevelOut[1]));
  const std::size_t c3 = align_up(act_bytes(height / 8, width / 8, kLevelOut[2]));
  l.lo = std::max(img, c2);
  l.hi = std::max(c1, c3);
  l.mid = align_up(act_bytes(height, width, 3));  // host input staging (f32 worst case)
  for (int lv = 0; lv < kLevels; ++lv) {
    l.levels[lv] = level_layout(lv, height, width, bands[static_cast<std::size_t>(lv)]);
    l.mid = std::max(l.mid, l.levels[lv].total);
    l.macs += l.levels[lv].macs;
  }
  l.tail = tail_layout(height, width);
  if (l.tail.total > l.lo + l.mid) l.mid = l.tail.total - l.lo;
  l.macs += l.tail.macs;
  l.total = l.lo + l.mid + l.hi;
  return l;
}

// ---------------------------------------------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------------------------------------------
bool starts_with(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  for (std::size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) s.replace(pos, from.size(), to);
  return s;
}

// diffusers name -> original (single-file) name.
std::string original_name(const std::string& d) {
  if (starts_with(d, "quant_conv.")) return "encoder." + d;
  if (starts_with(d, "encoder.conv_norm_out.")) return "encoder.norm_out." + d.substr(std::strlen("encoder.conv_norm_out."));
  const std::string mid_res = "encoder.mid_block.resnets.";
  if (starts_with(d, mid_res)) {
    const std::size_t dot = d.find('.', mid_res.size());
    const int j = std::stoi(d.substr(mid_res.size(), dot - mid_res.size()));
    return "encoder.mid.block_" + std::to_string(j + 1) + d.substr(dot);
  }
  const std::string attn = "encoder.mid_block.attentions.0.";
  if (starts_with(d, attn)) {
    std::string rest = d.substr(attn.size());
    static const std::pair<std::string, std::string> kMap[] = {
        {"group_norm.", "norm."}, {"to_q.", "q."}, {"to_k.", "k."}, {"to_v.", "v."}, {"to_out.0.", "proj_out."}};
    for (const auto& [from, to] : kMap)
      if (starts_with(rest, from)) return "encoder.mid.attn_1." + to + rest.substr(from.size());
    return d;
  }
  if (starts_with(d, "encoder.down_blocks.")) {
    std::string s = "encoder.down." + d.substr(std::strlen("encoder.down_blocks."));
    s = replace_all(s, ".resnets.", ".block.");
    s = replace_all(s, ".downsamplers.0.", ".downsample.");
    return replace_all(s, "conv_shortcut", "nin_shortcut");
  }
  return d;
}

// Reads the encoder tensors from one safetensors file. Without an arena it only checks names/shapes and sums the
// persistent bytes.
class Loader {
 public:
  Loader(const std::filesystem::path& path, runtime::DeviceArena* arena, cudaStream_t stream, bool direct_io)
      : file_(path), arena_(arena), stream_(stream) {
    const std::string key = "encoder.conv_in.weight";
    for (const auto& t : file_.tensors()) {
      if (t.name.size() < key.size() || t.name.compare(t.name.size() - key.size(), key.size(), key) != 0) continue;
      const std::string prefix = t.name.substr(0, t.name.size() - key.size());
      if (!prefix.empty() && prefix.back() != '.') continue;  // e.g. "xencoder.conv_in.weight" is another tensor
      if (found_) throw std::runtime_error("mi::flux2_vae_encoder: several encoders in " + util::path_utf8(path) + " ('" + prefix_ +
                                           "', '" + prefix + "')");
      prefix_ = prefix;
      found_ = true;
    }
    if (!found_) throw std::runtime_error("mi::flux2_vae_encoder: no encoder.conv_in.weight in " + util::path_utf8(path));
    original_ = file_.find(prefix_ + "encoder.down.0.block.0.conv1.weight") != nullptr;
    if (arena_ != nullptr) {
      reader_ = std::make_unique<weights::FileReader>(path, direct_io);
      tmp_ = static_cast<float*>(arena_->alloc_stack(kTmpBytes));
    }
  }

  std::size_t bytes() const { return bytes_; }
  std::uint64_t read_bytes() const { return read_; }

  // Every tensor under <prefix>encoder. / <prefix>quant_conv. must have been consumed: an unknown one means another
  // architecture (or a misnamed tensor), which loading without it would silently run wrong.
  void check_no_extra() const {
    for (const auto& t : file_.tensors()) {
      if (!starts_with(t.name, prefix_ + "encoder.") && !starts_with(t.name, prefix_ + "quant_conv.")) continue;
      if (consumed_.count(t.name) == 0)
        throw std::runtime_error("mi::flux2_vae_encoder: unexpected tensor '" + t.name + "' in " + util::path_utf8(file_.path()));
    }
  }

  // A tensor whose numel matches `shape` (the original layout stores the attention linears as [C, C, 1, 1]).
  Tensor plain(const std::string& dname, std::initializer_list<std::int64_t> shape) {
    const weights::TensorInfo& t = info(dname, shape);
    Tensor dst = Tensor::view(nullptr, DType::kF32, Device::kCUDA, shape);
    bytes_ += align_up(dst.nbytes());
    if (arena_ == nullptr) return dst;
    dst.data = arena_->alloc_persistent(dst.nbytes());
    read(t);
    MI_CUDA_CHECK(cudaMemcpyAsync(dst.data, host_.data(), dst.nbytes(), cudaMemcpyHostToDevice, stream_));
    return dst;
  }

  // torch Conv2d [K, C, R, R] -> KRSC with C padded to c_pad (0: no padding).
  Conv conv(const std::string& dprefix, std::int64_t k, std::int64_t c, std::int64_t r, std::int64_t c_pad = 0) {
    const std::int64_t cp = c_pad > 0 ? c_pad : c;
    const weights::TensorInfo& t = info(dprefix + ".weight", {k, c, r, r});
    Conv cv;
    cv.w = Tensor::view(nullptr, DType::kF32, Device::kCUDA, {k, r, r, cp});
    bytes_ += align_up(cv.w.nbytes());
    if (arena_ != nullptr) {
      cv.w.data = arena_->alloc_persistent(cv.w.nbytes());
      if (t.nbytes > kTmpBytes) throw std::runtime_error("mi::flux2_vae_encoder: conv weight larger than the load scratch");
      read(t);
      // Stream-ordered: the previous conversion has read tmp_ before this copy overwrites it.
      MI_CUDA_CHECK(cudaMemcpyAsync(tmp_, host_.data(), t.nbytes, cudaMemcpyHostToDevice, stream_));
      vae::conv_weight_to_krsc_padded(Tensor::view(tmp_, DType::kF32, Device::kCUDA, {k, c, r, r}), cv.w, stream_);
    }
    cv.bias = plain(dprefix + ".bias", {k});
    return cv;
  }

  Norm norm(const std::string& dprefix, std::int64_t c) { return {plain(dprefix + ".weight", {c}), plain(dprefix + ".bias", {c})}; }

 private:
  static constexpr std::size_t kTmpBytes = std::size_t{512} * 512 * 9 * 4;  // largest conv weight

  const weights::TensorInfo& info(const std::string& dname, std::initializer_list<std::int64_t> shape) {
    const std::string name = prefix_ + (original_ ? original_name(dname) : dname);
    const weights::TensorInfo* t = file_.find(name);
    if (t == nullptr) throw std::runtime_error("mi::flux2_vae_encoder: tensor '" + name + "' missing from " + util::path_utf8(file_.path()));
    consumed_.insert(name);
    std::int64_t numel = 1;
    for (std::int64_t d : shape) numel *= d;
    bool ok = t->dtype == weights::DType::kF32 && t->numel() == numel;
    // Same leading dims; any extra trailing dims must be 1 (original-layout attention linears).
    std::size_t i = 0;
    for (std::int64_t d : shape) ok = ok && i < t->shape.size() && t->shape[i++] == d;
    if (!ok) throw std::runtime_error("mi::flux2_vae_encoder: tensor '" + name + "' must be F32 with shape compatible with " +
                                      std::to_string(numel) + " elements");
    return *t;
  }

  void read(const weights::TensorInfo& t) {
    host_.resize(t.nbytes);
    reader_->read_at(t.offset, host_.data(), t.nbytes);
    read_ += t.nbytes;
  }

  weights::SafetensorsFile file_;
  runtime::DeviceArena* arena_;
  cudaStream_t stream_;
  std::unique_ptr<weights::FileReader> reader_;
  std::vector<std::byte> host_;
  float* tmp_ = nullptr;
  std::string prefix_;
  std::set<std::string> consumed_;
  bool found_ = false;
  bool original_ = false;
  std::size_t bytes_ = 0;
  std::uint64_t read_ = 0;
};

void assemble(Loader& ld, Flux2VaeEncoder::Weights& w) {
  for (int l = 0; l < kLevels; ++l) {
    const std::string p = "encoder.down_blocks." + std::to_string(l) + ".";
    const std::int64_t ci = l == 0 ? kLevelOut[0] : kLevelIn[l], co = kLevelOut[l];
    auto& convs = w.level_convs[static_cast<std::size_t>(l)];
    auto& norms = w.level_norms[static_cast<std::size_t>(l)];
    convs.clear();
    norms.clear();
    convs.push_back(l == 0 ? ld.conv("encoder.conv_in", co, 3, 3, kLevelIn[0]) : ld.conv(p + "resnets.0.conv_shortcut", co, ci, 1));
    convs.push_back(ld.conv(p + "resnets.0.conv1", co, ci, 3));
    convs.push_back(ld.conv(p + "resnets.0.conv2", co, co, 3));
    convs.push_back(ld.conv(p + "resnets.1.conv1", co, co, 3));
    convs.push_back(ld.conv(p + "resnets.1.conv2", co, co, 3));
    convs.push_back(ld.conv(p + "downsamplers.0.conv", co, co, 3));
    norms.push_back(ld.norm(p + "resnets.0.norm1", ci));
    norms.push_back(ld.norm(p + "resnets.0.norm2", co));
    norms.push_back(ld.norm(p + "resnets.1.norm1", co));
    norms.push_back(ld.norm(p + "resnets.1.norm2", co));
    static_assert(kConvsPerLevel == 6 && kNormsPerLevel == 4, "level program layout");
  }
  for (int i = 0; i < 4; ++i) {
    const std::string p = i < 2 ? "encoder.down_blocks.3.resnets." + std::to_string(i) : "encoder.mid_block.resnets." + std::to_string(i - 2);
    Resnet& r = w.tail_resnets[i];
    r.in_ch = r.out_ch = kTailCh;
    r.norm1 = ld.norm(p + ".norm1", kTailCh);
    r.conv1 = ld.conv(p + ".conv1", kTailCh, kTailCh, 3);
    r.norm2 = ld.norm(p + ".norm2", kTailCh);
    r.conv2 = ld.conv(p + ".conv2", kTailCh, kTailCh, 3);
  }
  const std::string a = "encoder.mid_block.attentions.0.";
  w.attn.gn_gamma = ld.plain(a + "group_norm.weight", {kTailCh});
  w.attn.gn_beta = ld.plain(a + "group_norm.bias", {kTailCh});
  w.attn.wq = ld.plain(a + "to_q.weight", {kTailCh, kTailCh});
  w.attn.bq = ld.plain(a + "to_q.bias", {kTailCh});
  w.attn.wk = ld.plain(a + "to_k.weight", {kTailCh, kTailCh});
  w.attn.bk = ld.plain(a + "to_k.bias", {kTailCh});
  w.attn.wv = ld.plain(a + "to_v.weight", {kTailCh, kTailCh});
  w.attn.bv = ld.plain(a + "to_v.bias", {kTailCh});
  w.attn.wo = ld.plain(a + "to_out.0.weight", {kTailCh, kTailCh});
  w.attn.bo = ld.plain(a + "to_out.0.bias", {kTailCh});
  w.norm_out = ld.norm("encoder.conv_norm_out", kTailCh);
  w.conv_out = ld.conv("encoder.conv_out", kMomentCh, kTailCh, 3);
  w.quant_conv = ld.conv("quant_conv", kMomentCh, kMomentCh, 1);
}

void assemble_bn(Loader& ld, Flux2VaeEncoder::Weights& w) {
  w.bn_mean = ld.plain("bn.running_mean", {kLatentChannels});
  w.bn_var = ld.plain("bn.running_var", {kLatentChannels});
}

void check_lt(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("mi::flux2_vae_encoder: ") + what + " failed (status " + std::to_string(static_cast<int>(st)) + ")");
}

}  // namespace

std::string TilePlan::describe() const {
  std::ostringstream os;
  os << width << "x" << height << ": ";
  for (int l = 0; l < kLevels; ++l) {
    os << "L" << l << " " << (band_rows[static_cast<std::size_t>(l)] == 0 ? std::string("untiled") : std::to_string(band_rows[static_cast<std::size_t>(l)]) + "-row bands")
       << (l + 1 < kLevels ? ", " : "");
  }
  os << "; workspace " << static_cast<double>(workspace_bytes) / (1024.0 * 1024.0) << " MiB, est. cost " << relative_cost << "x";
  return os.str();
}

std::size_t Flux2VaeEncoder::persistent_bytes(const std::filesystem::path& vae_file, const weights::ModelIndex* latent_stats) {
  Loader ld(vae_file, nullptr, nullptr, false);
  Weights w;
  assemble(ld, w);
  ld.check_no_extra();
  if (latent_stats == nullptr) {
    assemble_bn(ld, w);
    return ld.bytes();
  }
  return ld.bytes() + 2 * align_up(kLatentChannels * sizeof(float));
}

Flux2VaeEncoder::Flux2VaeEncoder(const std::filesystem::path& vae_file, runtime::DeviceArena& arena,
                                 runtime::StreamSet& streams, bool direct_io, const weights::ModelIndex* latent_stats,
                                 bool round_stats)
    : arena_(arena) {
  const auto t0 = std::chrono::steady_clock::now();
  check_lt(cublasLtCreate(&lt_), "cublasLtCreate");
  const cudaStream_t s = streams.compute();
  try {
    persistent_bytes(vae_file, latent_stats);  // validates every name, dtype and shape before allocating
    runtime::StackScope scope(arena);
    Loader ld(vae_file, &arena, s, direct_io);
    assemble(ld, w_);
    if (latent_stats == nullptr && !round_stats) {
      assemble_bn(ld, w_);  // as stored
    } else {
      // The statistics of the index's VAE "decoder" group, or the file's own rounded to bf16 on the host (round_stats:
      // a single VAE file, read_latent_stats), before their upload.
      const flux2_vae::LatentStats st = latent_stats != nullptr ? flux2_vae::read_latent_stats(*latent_stats, round_stats)
                                                                : flux2_vae::read_latent_stats(flux2_vae::index_vae_file(vae_file), true);
      for (auto [v, dst] : {std::pair<const std::vector<float>*, Tensor*>{&st.mean, &w_.bn_mean}, {&st.var, &w_.bn_var}}) {
        *dst = arena.alloc_persistent_tensor(DType::kF32, {kLatentChannels});
        MI_CUDA_CHECK(cudaMemcpyAsync(dst->data, v->data(), v->size() * sizeof(float), cudaMemcpyHostToDevice, s));
        load_stats_.bytes += v->size() * sizeof(float);
      }
    }
    MI_CUDA_CHECK(cudaStreamSynchronize(s));
    load_stats_.bytes += ld.read_bytes();
  } catch (...) {
    cublasLtDestroy(lt_);
    throw;
  }
  load_stats_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Flux2VaeEncoder::~Flux2VaeEncoder() {
  if (lt_ != nullptr) cublasLtDestroy(lt_);
}

TilePlan Flux2VaeEncoder::make_plan(std::int64_t height, std::int64_t width, const std::array<std::int64_t, kLevels>& band_rows) {
  check_size(height, width);
  check_bands(band_rows);
  static const std::array<std::int64_t, kLevels> untiled{};
  const EncodeLayout l = encode_layout(height, width, band_rows);
  TilePlan p;
  p.height = height;
  p.width = width;
  p.band_rows = band_rows;
  p.workspace_bytes = l.total;
  p.relative_cost = band_rows == untiled ? 1.0 : l.macs / encode_layout(height, width, untiled).macs;
  return p;
}

TilePlan Flux2VaeEncoder::plan_untiled(std::int64_t height, std::int64_t width) { return make_plan(height, width, {}); }

std::size_t Flux2VaeEncoder::workspace_bytes(std::int64_t height, std::int64_t width) {
  return plan_untiled(height, width).workspace_bytes;
}

TilePlan Flux2VaeEncoder::plan_min_memory(std::int64_t height, std::int64_t width) {
  check_size(height, width);
  std::array<std::int64_t, kLevels> b{};
  // A level of at most 16 rows runs untiled: one band would cover it anyway, plus its statistics passes and input copy.
  for (int l = 0; l < kLevels; ++l) b[static_cast<std::size_t>(l)] = (height >> l) > 16 ? 16 : 0;
  return make_plan(height, width, b);
}

// Levels only share the middle region, so each level independently takes the largest band (least recompute and
// halo) whose scratch fits next to the fixed regions.
std::optional<TilePlan> Flux2VaeEncoder::plan_tiled(std::int64_t height, std::int64_t width, std::size_t budget_bytes) {
  const TilePlan untiled = plan_untiled(height, width);
  if (untiled.workspace_bytes <= budget_bytes) return untiled;
  const EncodeLayout base = encode_layout(height, width, plan_min_memory(height, width).band_rows);
  std::size_t fixed_mid = align_up(act_bytes(height, width, 3));
  if (base.tail.total > base.lo) fixed_mid = std::max(fixed_mid, base.tail.total - base.lo);
  if (base.lo + base.hi + fixed_mid > budget_bytes) return std::nullopt;
  const std::size_t mid_budget = budget_bytes - base.lo - base.hi;
  std::array<std::int64_t, kLevels> bands{};
  for (int l = 0; l < kLevels; ++l) {
    const std::int64_t h = height >> l;
    bool found = false;
    for (std::int64_t b : {std::int64_t{0}, std::int64_t{1024}, std::int64_t{512}, std::int64_t{256}, std::int64_t{128},
                           std::int64_t{64}, std::int64_t{32}, std::int64_t{16}}) {
      if (b >= h) continue;
      if (level_layout(l, height, width, b).total <= mid_budget) {
        bands[static_cast<std::size_t>(l)] = b;
        found = true;
        break;
      }
    }
    if (!found) return std::nullopt;
  }
  TilePlan p = make_plan(height, width, bands);
  if (p.workspace_bytes > budget_bytes) return std::nullopt;
  return p;
}

void Flux2VaeEncoder::run_tail(char* ws, const Tensor& c3, const Tensor& latents, const Tensor* noise,
                               const EncodeOptions& options, cudaStream_t stream, const StageHook& hook) const {
  const std::int64_t h = c3.dim(1), w = c3.dim(2);
  const TailLayout l = tail_layout(h * 8, w * 8);
  char* a = ws + l.a;
  char* b = ws + l.b;
  void* aux = ws + l.aux;
  const auto stage = [&](const char* name, const Tensor& t) {
    if (hook) hook(name, t);
  };
  const auto conv = [&](const Tensor& x, const Conv& cv, const Tensor* res, const Tensor& y) {
    vae::encoder_conv2d(x, cv.w, cv.bias, res, y, aux, l.aux_bytes, options.conv, stream);
  };
  const auto norm = [&](const Tensor& x, const Norm& n, const Tensor& y) {
    vae::group_norm(x, kGroups, vae::kResnetEps, n.gamma, n.beta, /*silu=*/true, y, aux, l.aux_bytes, stream);
  };
  const Tensor at = act(a, h, w, kTailCh), bt = act(b, h, w, kTailCh);
  const auto resnet = [&](const Resnet& r) {
    norm(c3, r.norm1, at);
    conv(at, r.conv1, nullptr, bt);
    norm(bt, r.norm2, bt);
    conv(bt, r.conv2, &c3, c3);
  };
  resnet(w_.tail_resnets[0]);
  resnet(w_.tail_resnets[1]);
  stage("down_blocks.3", c3);
  resnet(w_.tail_resnets[2]);
  stage("mid.resnets.0", c3);
  vae::mid_attention(lt_, c3, w_.attn, &c3, c3, ws + l.attn, l.attn_bytes, options.attention, options.fast_linears,
                             stream);
  stage("mid.attention", c3);
  resnet(w_.tail_resnets[3]);
  stage("mid.resnets.1", c3);
  norm(c3, w_.norm_out, c3);
  const Tensor co = act(a, h, w, kMomentCh), mo = act(b, h, w, kMomentCh);
  conv(c3, w_.conv_out, nullptr, co);
  stage("conv_out", co);
  conv(co, w_.quant_conv, nullptr, mo);
  stage("moments", mo);
  const Tensor y = Tensor::view(latents.data, DType::kF32, Device::kCUDA, {1, kLatentChannels, h / 2, w / 2});
  vae::encoder_latent_out(mo, w_.bn_mean, w_.bn_var, vae::kBatchNormEps, noise, y, stream);
}

void Flux2VaeEncoder::encode(const Tensor& rgb, const Tensor& latents, const EncodeOptions& options, const TilePlan& plan,
                             cudaStream_t stream, const StageHook& hook) {
  const bool batched = rgb.rank == 4;
  if ((rgb.dtype != DType::kU8 && rgb.dtype != DType::kF32) || !rgb.is_contiguous() || rgb.data == nullptr ||
      (rgb.rank != 3 && !(batched && rgb.dim(0) == 1)) || rgb.dim(-1) != 3)
    fail("rgb must be contiguous u8 or f32 [H, W, 3], got " + rgb.describe());
  const std::int64_t height = rgb.dim(-3), width = rgb.dim(-2);
  check_size(height, width);
  const std::int64_t lh = height / kPixelsPerLatent, lw = width / kPixelsPerLatent;
  if (latents.device != Device::kCUDA || latents.dtype != DType::kF32 || !latents.is_contiguous() || latents.data == nullptr ||
      (latents.rank != 3 && !(latents.rank == 4 && latents.dim(0) == 1)) || latents.dim(-3) != kLatentChannels ||
      latents.dim(-2) != lh || latents.dim(-1) != lw)
    fail("latents must be contiguous CUDA fp32 [128, " + std::to_string(lh) + ", " + std::to_string(lw) + "], got " +
         latents.describe());
  if (plan.height != height || plan.width != width) fail("plan is for another image size: " + plan.describe());
  check_bands(plan.band_rows);

  std::vector<float> noise_host;
  if (options.mode == LatentMode::kSample) noise_host = rng::torch_cpu_randn(options.seed, 32 * (height / 8) * (width / 8));

  const EncodeLayout l = encode_layout(height, width, plan.band_rows);
  runtime::StackScope scope(arena_);
  char* ws = static_cast<char*>(arena_.alloc_stack(l.total));
  char* lo = ws;
  char* mid = ws + l.lo;
  char* hi = ws + l.lo + l.mid;

  // Preprocess into the lo region (host input staged in the middle).
  const Tensor img = act(lo, height, width, 4);
  Tensor src = Tensor::view(rgb.data, rgb.dtype, Device::kCUDA, {height, width, 3});
  if (rgb.device != Device::kCUDA) {
    src.data = mid;
    MI_CUDA_CHECK(cudaMemcpyAsync(mid, rgb.data, rgb.nbytes(), cudaMemcpyHostToDevice, stream));
  }
  vae::encoder_preprocess(src, img, stream);

  float* ckpt[kLevels + 1] = {reinterpret_cast<float*>(lo), reinterpret_cast<float*>(hi), reinterpret_cast<float*>(lo),
                              reinterpret_cast<float*>(hi)};
  static const char* const kStageNames[kLevels] = {"down_blocks.0", "down_blocks.1", "down_blocks.2"};
  for (int lv = 0; lv < kLevels; ++lv) {
    const LevelLayout& ll = l.levels[lv];
    const std::int64_t band = plan.band_rows[static_cast<std::size_t>(lv)];
    LevelExec e;
    e.level = lv;
    e.h = height >> lv;
    e.w = width >> lv;
    e.convs = &w_.level_convs[static_cast<std::size_t>(lv)];
    e.norms = &w_.level_norms[static_cast<std::size_t>(lv)];
    e.alias_in = band == 0 || !writes_input(program(lv));
    e.in_ckpt = ckpt[lv];
    e.out_ckpt = ckpt[lv + 1];
    for (int s = kIn; s < kOut; ++s) e.slot_base[s] = mid + ll.slot[s];
    e.stats = reinterpret_cast<float*>(mid + ll.stats);
    e.accum = mid + ll.accum;
    e.aux = mid + ll.aux;
    e.aux_bytes = ll.strip - ll.aux;
    e.strip = mid + ll.strip;
    e.math = options.conv;
    e.stream = stream;
    run_level(e, band);
    if (hook) hook(kStageNames[lv], act(ckpt[lv + 1], e.h / 2, e.w / 2, kLevelOut[lv]));
  }

  const TailLayout& tl = l.tail;
  Tensor noise;
  if (options.mode == LatentMode::kSample) {
    noise = Tensor::view(ws + tl.noise, DType::kF32, Device::kCUDA, {1, 32, height / 8, width / 8});
    MI_CUDA_CHECK(cudaMemcpyAsync(noise.data, noise_host.data(), noise.nbytes(), cudaMemcpyHostToDevice, stream));
  }
  run_tail(ws, act(ckpt[kLevels], height / 8, width / 8, kTailCh), latents,
           options.mode == LatentMode::kSample ? &noise : nullptr, options, stream, hook);
}

void Flux2VaeEncoder::encode(const Tensor& rgb, const Tensor& latents, const EncodeOptions& options, cudaStream_t stream,
                             const StageHook& hook) {
  if (rgb.rank < 3) fail("rgb must be [H, W, 3], got " + rgb.describe());
  encode(rgb, latents, options, plan_untiled(rgb.dim(-3), rgb.dim(-2)), stream, hook);
}

TilePlan Flux2VaeEncoder::encode_within(const Tensor& rgb, const Tensor& latents, const EncodeOptions& options,
                                        std::size_t budget_bytes, cudaStream_t stream) {
  if (rgb.rank < 3) fail("rgb must be [H, W, 3], got " + rgb.describe());
  const std::optional<TilePlan> p = plan_tiled(rgb.dim(-3), rgb.dim(-2), budget_bytes);
  if (!p) throw std::runtime_error("mi::flux2_vae_encoder: no encode plan fits " + std::to_string(budget_bytes) + " bytes");
  encode(rgb, latents, options, *p, stream);
  return *p;
}

}  // namespace mi::flux2_vae_encoder
