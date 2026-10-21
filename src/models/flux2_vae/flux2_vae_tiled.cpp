// Exact tiled decode of the FLUX.2 VAE tail (plan step 23): schedule planner and band executor. See the
// "Exact tiled decode" notes in flux2_vae.h for why the output is byte-identical to the untiled decode.
//
// Terms: the tail is a chain of 17 units (kUnits). A segment runs units [first, last) on full-width row bands; its
// input is a checkpoint tensor kept whole in VRAM, its output the next checkpoint (or RGB). Rows are counted at
// each tensor's own level (0 = H/8 ... 3 = H). A band owns rows [o0, o1) of the segment output; every GroupNorm
// inside the segment owns the matching rows of its own input, so the bands partition every layer exactly.
//
// Per band, the executor copies the needed input rows (owned rows + halo) into a tile buffer and runs the units
// on whole tile tensors: convs zero-pad at the tile edge, which is exact at the image border and wrong elsewhere,
// and the halo (one input-level row per 3x3 conv on the way, mapped through the upsamplers) keeps those wrong rows
// out of the owned rows of every tensor that matters.
//
// Checkpoint overlay: one block holds the segment input at one end and receives its output at the other. Bands
// run from the output's end towards the input's (top-down when the input sits at the end of the block, bottom-up
// otherwise), and band k's output is copied into the block only after band k+1's input rows were gathered, so a
// write never reaches input rows a later band still reads (the planner sizes the block to guarantee it).
#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/basic/elementwise.h"
#include "models/flux2_vae/flux2_vae.h"
#include "models/flux2_vae/flux2_vae_internal.h"

namespace mi::flux2_vae {
namespace {

using namespace detail;
using runtime::align_up;

// ---- Tail structure ---------------------------------------------------------------------------------------

enum class UnitKind : std::uint8_t { kResnet, kUpsample, kOut };
struct UnitShape {
  UnitKind kind;
  int level, in_ch, out_ch;  // an upsampler reads `level` and writes level + 1
};
constexpr int kTailUnits = 17;

constexpr std::array<UnitShape, kTailUnits> make_units() {
  std::array<UnitShape, kTailUnits> u{};
  int n = 0, ch = kMidCh;
  u[n++] = {UnitKind::kResnet, 0, kMidCh, kMidCh};  // mid_block.resnets[1]
  for (int b = 0; b < 4; ++b) {
    for (int r = 0; r < kResnetsPerBlock; ++r) {
      u[n++] = {UnitKind::kResnet, b, ch, kUpOut[b]};
      ch = kUpOut[b];
    }
    if (kUpSample[b]) u[n++] = {UnitKind::kUpsample, b, ch, ch};
  }
  u[n++] = {UnitKind::kOut, 3, ch, 3};
  return u;
}
constexpr std::array<UnitShape, kTailUnits> kUnits = make_units();
static_assert(kUnits[kTailUnits - 1].kind == UnitKind::kOut && kUnits[kTailUnits - 1].in_ch == 96, "unit table");

constexpr int out_level(const UnitShape& u) { return u.kind == UnitKind::kUpsample ? u.level + 1 : u.level; }

enum class OpKind : std::uint8_t { kConv3, kConv1, kNorm, kUp };
struct Op {
  OpKind kind;
  int level, in_ch, out_ch;  // kUp writes level + 1
};

// The ops of a unit in the executor's order (a shortcut resnet computes its 1x1 shortcut first).
void append_ops(const UnitShape& u, std::vector<Op>& ops) {
  const int l = u.level;
  switch (u.kind) {
    case UnitKind::kResnet:
      if (u.in_ch != u.out_ch) ops.push_back({OpKind::kConv1, l, u.in_ch, u.out_ch});
      ops.push_back({OpKind::kNorm, l, u.in_ch, u.in_ch});
      ops.push_back({OpKind::kConv3, l, u.in_ch, u.out_ch});
      ops.push_back({OpKind::kNorm, l, u.out_ch, u.out_ch});
      ops.push_back({OpKind::kConv3, l, u.out_ch, u.out_ch});
      break;
    case UnitKind::kUpsample:
      ops.push_back({OpKind::kUp, l, u.in_ch, u.in_ch});
      ops.push_back({OpKind::kConv3, l + 1, u.in_ch, u.in_ch});
      break;
    case UnitKind::kOut:
      ops.push_back({OpKind::kNorm, l, u.in_ch, u.in_ch});
      ops.push_back({OpKind::kConv3, l, u.in_ch, u.out_ch});
      break;
  }
}

// Cost model in MAC-equivalents: conv MACs plus the bytes moved by memory-bound kernels (GroupNorm, upsample,
// band copies) at work_per_byte, calibrated on the dev box (VaeKernelsBench): TF32 convs ~10 TMAC/s and bf16 convs
// ~24 TMAC/s against ~350 GB/s for the elementwise kernels. Used to rank schedules and to report an estimate.
constexpr double kWorkPerByteF32 = 29.0;
constexpr double kWorkPerByteBf16 = 70.0;

struct Geometry {
  std::int64_t h8, w8;  // H/8 x W/8 (twice the latent grid)
  std::size_t esize;    // tail activation bytes
  double work_per_byte;
  std::int64_t rows(int level) const { return h8 << level; }
  std::int64_t cols(int level) const { return w8 << level; }
  std::size_t row_bytes(int level, int ch) const { return static_cast<std::size_t>(cols(level) * ch) * esize; }
};

// Work per output pixel of an op at its compute level (kUp: per upsampled pixel; kNorm: the apply only).
double op_work(const Op& o, const Geometry& g) {
  const double bytes = g.work_per_byte * static_cast<double>(g.esize);
  switch (o.kind) {
    case OpKind::kConv3: return 9.0 * o.in_ch * o.out_ch;
    case OpKind::kConv1: return 1.0 * o.in_ch * o.out_ch;
    case OpKind::kNorm: return bytes * 2.0 * o.in_ch;
    case OpKind::kUp: return bytes * 1.25 * o.in_ch;
  }
  return 0;
}
// Work of reading a tensor of `rows` rows (GroupNorm statistics) or copying it (x2).
double read_work(const Geometry& g, int level, int ch, std::int64_t rows) {
  return g.work_per_byte * static_cast<double>(g.row_bytes(level, ch)) * static_cast<double>(rows);
}
int op_compute_level(const Op& o) { return o.kind == OpKind::kUp ? o.level + 1 : o.level; }

// A pass target: the input of a GroupNorm (stats pass) or the segment output (output pass).
struct Target {
  int end = 0;            // ops [0, end) run before it
  int level = 0, ch = 0;  // its resolution and channels
  // Input-level rows a band needs beyond its owned rows, per side (exact for bands on the band grid; clipped to
  // the image afterwards, which equals clipping at every level).
  std::int64_t halo_top = 0, halo_bot = 0;
  double work_per_row = 0;  // work per input-level row of a band
  bool direct = false;      // the GroupNorm input is the segment input itself (only 1x1 shortcuts before it)
};

struct SegmentInfo {
  int first = 0, last = 0;
  int in_level = 0, out_level = 0, in_ch = 0, out_ch = 0;  // out_ch = 0: RGB output
  std::vector<Op> ops;
  std::vector<Target> targets;  // the GroupNorms in order, then the output
  double buf_per_row = 0;       // largest tile tensor, bytes per input-level row of the band
  int groupnorms() const { return static_cast<int>(targets.size()) - 1; }
};

SegmentInfo segment_info(const Geometry& g, int first, int last) {
  SegmentInfo si;
  si.first = first;
  si.last = last;
  si.in_level = kUnits[first].level;
  si.out_level = out_level(kUnits[last - 1]);
  si.in_ch = kUnits[first].in_ch;
  si.out_ch = kUnits[last - 1].kind == UnitKind::kOut ? 0 : kUnits[last - 1].out_ch;
  for (int u = first; u < last; ++u) append_ops(kUnits[u], si.ops);
  const auto per_row = [&](int level, int ch) {  // tile tensor bytes per input-level row
    return static_cast<double>(g.row_bytes(level, ch)) * static_cast<double>(std::int64_t{1} << (level - si.in_level));
  };
  si.buf_per_row = per_row(si.in_level, si.in_ch);
  for (const Op& o : si.ops) {
    si.buf_per_row = std::max(si.buf_per_row, per_row(o.level, o.in_ch));
    si.buf_per_row = std::max(si.buf_per_row, per_row(op_compute_level(o), std::max(o.out_ch, 4)));  // conv_out: fp32 cast
  }
  const auto make_target = [&](int end, int level, int ch) {
    Target t;
    t.end = end;
    t.level = level;
    t.ch = ch;
    constexpr std::int64_t kAligned = std::int64_t{1} << 24;  // a row index aligned to every band grid
    std::int64_t a = kAligned, b = kAligned;
    t.direct = true;
    for (int k = end - 1; k >= 0; --k) {
      const Op& o = si.ops[static_cast<std::size_t>(k)];
      if (o.kind == OpKind::kConv3) --a, ++b;
      if (o.kind == OpKind::kUp) a >>= 1, b = (b + 1) >> 1;
      if (o.kind != OpKind::kConv1) t.direct = false;
    }
    t.halo_top = (kAligned >> (level - si.in_level)) - a;
    t.halo_bot = b - (kAligned >> (level - si.in_level));
    for (int k = 0; k < end; ++k) {
      const Op& o = si.ops[static_cast<std::size_t>(k)];
      const int cl = op_compute_level(o);
      t.work_per_row += op_work(o, g) * static_cast<double>(g.cols(cl)) * static_cast<double>(std::int64_t{1} << (cl - si.in_level));
    }
    return t;
  };
  int level = si.in_level;
  for (int k = 0; k < static_cast<int>(si.ops.size()); ++k) {
    const Op& o = si.ops[static_cast<std::size_t>(k)];
    if (o.kind == OpKind::kNorm) si.targets.push_back(make_target(k, level, o.in_ch));
    level = op_compute_level(o);
  }
  si.targets.push_back(make_target(static_cast<int>(si.ops.size()), si.out_level, si.out_ch));
  return si;
}

struct Band {
  std::int64_t o0, o1;  // owned rows at the segment output level
};
std::int64_t band_count(const Geometry& g, const SegmentInfo& si, std::int64_t band_rows) {
  const std::int64_t rows = g.rows(si.out_level);
  return (rows + band_rows - 1) / band_rows;
}
Band band(const Geometry& g, const SegmentInfo& si, std::int64_t band_rows, std::int64_t k) {
  return {k * band_rows, std::min(g.rows(si.out_level), (k + 1) * band_rows)};
}
// Owned rows of a band at `level` (<= the output level).
std::pair<std::int64_t, std::int64_t> owned(const SegmentInfo& si, const Band& b, int level) {
  const int d = si.out_level - level;
  return {b.o0 >> d, b.o1 >> d};
}
// Input rows [a, b) a band needs to compute target t.
std::pair<std::int64_t, std::int64_t> input_rows(const Geometry& g, const SegmentInfo& si, const Target& t, const Band& bd) {
  const auto [a_t, b_t] = owned(si, bd, t.level);
  const int d = t.level - si.in_level;
  return {std::max<std::int64_t>(0, (a_t >> d) - t.halo_top), std::min(g.rows(si.in_level), (b_t >> d) + t.halo_bot)};
}

// ---- Planner ------------------------------------------------------------------------------------------------

constexpr int kMaxSegmentUnits = 6;  // longer segments only add recomputation
constexpr int kStatsSlots = 2 * kTailUnits;

struct SegmentCost {
  double work = 0;
  std::size_t block = 0;  // checkpoint block bytes this segment needs (input + output overlaid)
  std::size_t buf = 0;    // bytes of each of the three tile buffers
  std::size_t work_bytes = 0;  // 3 * buf + strip scratch when a tile tensor reaches 2 GiB
};

SegmentCost segment_cost(const Geometry& g, const SegmentInfo& si, std::int64_t band_rows) {
  SegmentCost c;
  const std::int64_t n = band_count(g, si, band_rows);
  for (int k = 0; k < si.groupnorms(); ++k) {
    const Target& t = si.targets[static_cast<std::size_t>(k)];
    c.work += read_work(g, t.level, t.ch, g.rows(t.level));  // statistics (inline when n == 1)
    if (n == 1 || t.direct) continue;
    for (std::int64_t j = 0; j < n; ++j) {
      const auto [a, b] = input_rows(g, si, t, band(g, si, band_rows, j));
      c.work += static_cast<double>(b - a) * t.work_per_row + 2.0 * read_work(g, si.in_level, si.in_ch, b - a);
    }
  }
  const Target& out = si.targets.back();
  const auto ri = static_cast<std::int64_t>(g.row_bytes(si.in_level, si.in_ch));
  const auto ro = static_cast<std::int64_t>(si.out_ch ? g.row_bytes(si.out_level, si.out_ch) : 0);
  const std::int64_t in_bytes = ri * g.rows(si.in_level), out_bytes = ro * g.rows(si.out_level);
  std::int64_t max_rows = 0, block = std::max(in_bytes, out_bytes);
  std::vector<std::pair<std::int64_t, std::int64_t>> rows(static_cast<std::size_t>(n));
  for (std::int64_t j = 0; j < n; ++j) {
    rows[static_cast<std::size_t>(j)] = input_rows(g, si, out, band(g, si, band_rows, j));
    const std::int64_t r = rows[static_cast<std::size_t>(j)].second - rows[static_cast<std::size_t>(j)].first;
    max_rows = std::max(max_rows, r);
    c.work += static_cast<double>(r) * out.work_per_row + 2.0 * read_work(g, si.in_level, si.in_ch, r);
  }
  if (si.out_ch) c.work += 2.0 * read_work(g, si.out_level, si.out_ch, g.rows(si.out_level));  // commits
  // Overlay (see the file comment): top-down with the input at the end, output written at the start; bottom-up
  // mirrored. Either may happen, so the block covers both.
  for (std::int64_t k = 0; k < n; ++k) {
    const Band bk = band(g, si, band_rows, k);
    if (k + 2 < n) block = std::max(block, bk.o1 * ro + in_bytes - rows[static_cast<std::size_t>(k + 2)].first * ri);
    if (k >= 2) block = std::max(block, rows[static_cast<std::size_t>(k - 2)].second * ri + out_bytes - bk.o0 * ro);
  }
  c.block = align_up(static_cast<std::size_t>(block));
  c.buf = align_up(static_cast<std::size_t>(static_cast<double>(max_rows) * si.buf_per_row));
  c.work_bytes = 3 * c.buf + (c.buf >= kCutlassMaxBytes ? kStripScratchBytes : 0);
  return c;
}

// Fixed parts of the tiled workspace. The head (untiled, fp32) runs before any checkpoint exists, so it may use
// the checkpoint block: it needs its output x (head_x bytes) plus a scratch area (two more tensors, overlapped by
// the attention workspace), and x must end up at the end of the block as the first checkpoint (in place in fp32,
// cast into the last first_ckpt bytes in bf16).
struct Fixed {
  std::size_t head_x = 0, head_scratch = 0;
  std::size_t first_ckpt = 0;
  std::size_t head_in = 0;   // block size that holds the whole head
  std::size_t head_out = 0;  // tile-buffer region size that holds the head when the block is smaller
  std::size_t aux = 0;       // conv / GroupNorm scratch
  std::size_t gn = 0;        // tail GroupNorm accumulator + statistics slots
  std::size_t rgb = 0;       // RGB8 staging for host output
};
Fixed fixed_parts(std::int64_t h, std::int64_t w, const DecodeMode& mode) {
  Fixed f;
  const Tensor x = act(nullptr, DType::kF32, 2 * h, 2 * w, kMidCh);
  f.head_x = align_up(x.nbytes());
  f.head_scratch = std::max(2 * f.head_x, align_up(vae::mid_attention_workspace_bytes(x)));
  f.first_ckpt = align_up(x.nbytes() / (mode.bf16_convs ? 2 : 1));
  const bool in_place = !mode.bf16_convs;
  f.head_in = f.head_x + f.head_scratch + (in_place ? 0 : f.first_ckpt);
  f.head_out = (in_place ? 0 : f.head_x) + f.head_scratch;
  f.aux = align_up(layout(h, w, mode).aux);
  f.gn = align_up(vae::group_norm_accum_bytes(1, vae::kGroupNormGroups)) +
         align_up(static_cast<std::size_t>(kStatsSlots) * vae::kGroupNormGroups * 2 * sizeof(float));
  f.rgb = align_up(static_cast<std::size_t>(h * w) * kPixelsPerLatent * kPixelsPerLatent * 3);
  return f;
}

// Final block and tile-buffer region sizes: the head inside the block or in the tile-buffer region, whichever
// needs less memory (a function of the schedule only, so the executor re-derives it).
struct Regions {
  bool head_inside;
  std::size_t block, work;
};
Regions place_head(const Fixed& f, std::size_t block, std::size_t work) {
  const std::size_t inside = std::max(block, f.head_in) + work, outside = block + std::max(work, f.head_out);
  if (inside <= outside) return {true, std::max(block, f.head_in), work};
  return {false, block, std::max(work, f.head_out)};
}

struct Option {
  int first, last;
  std::int64_t band_rows;
  SegmentCost cost;
};

class Planner {
 public:
  Planner(std::int64_t h, std::int64_t w, const DecodeMode& mode)
      : g_{2 * h, 2 * w, mode.bf16_convs ? std::size_t{2} : std::size_t{4}, mode.bf16_convs ? kWorkPerByteBf16 : kWorkPerByteF32},
        h_(h), w_(w), mode_(mode),
        fixed_(fixed_parts(h, w, mode)) {
    for (int first = 0; first < kTailUnits; ++first)
      for (int last = first + 1; last <= std::min(kTailUnits, first + kMaxSegmentUnits); ++last)
        infos_.push_back(segment_info(g_, first, last));
    for (int u = 0; u < kTailUnits; ++u) {
      std::vector<Op> ops;
      append_ops(kUnits[u], ops);
      for (const Op& o : ops) {
        const int cl = op_compute_level(o);
        untiled_work_ += op_work(o, g_) * static_cast<double>(g_.rows(cl) * g_.cols(cl));
        if (o.kind == OpKind::kNorm) untiled_work_ += read_work(g_, o.level, o.in_ch, g_.rows(o.level));
      }
    }
  }

  // Candidate band heights: the smallest multiple of 8 image rows for every band count.
  std::vector<int> tiles() const {
    std::vector<int> t;
    const std::int64_t rows = g_.rows(3);
    for (std::int64_t n = rows / 8; n >= 1; --n) {
      const auto tile = static_cast<int>(((rows + n - 1) / n + 7) / 8 * 8);
      if (t.empty() || t.back() != tile) t.push_back(tile);
    }
    return t;
  }

  std::vector<Option> options(int tile) const {
    std::vector<Option> opts;
    for (const SegmentInfo& si : infos_) {
      const std::int64_t rows = g_.rows(si.out_level);
      const std::int64_t bh = tile >> (3 - si.out_level);
      if (bh < rows) opts.push_back({si.first, si.last, bh, segment_cost(g_, si, bh)});
      opts.push_back({si.first, si.last, rows, segment_cost(g_, si, rows)});
    }
    return opts;
  }

  // Least work over segmentations whose options respect both caps; empty when infeasible.
  static std::vector<const Option*> best_path(const std::vector<Option>& opts, std::size_t block_cap, std::size_t work_cap) {
    std::array<double, kTailUnits + 1> best;
    std::array<const Option*, kTailUnits + 1> via{};
    best.fill(std::numeric_limits<double>::infinity());
    best[0] = 0;
    for (int j = 1; j <= kTailUnits; ++j)
      for (const Option& o : opts) {
        if (o.last != j || o.cost.block > block_cap || o.cost.work_bytes > work_cap) continue;
        const double v = best[static_cast<std::size_t>(o.first)] + o.cost.work;
        if (v < best[static_cast<std::size_t>(j)]) best[static_cast<std::size_t>(j)] = v, via[static_cast<std::size_t>(j)] = &o;
      }
    std::vector<const Option*> path;
    if (!via[kTailUnits]) return path;
    for (int j = kTailUnits; j > 0; j = via[static_cast<std::size_t>(j)]->first) path.push_back(via[static_cast<std::size_t>(j)]);
    std::reverse(path.begin(), path.end());
    return path;
  }

  TilePlan finish(int tile, const std::vector<const Option*>& path) const {
    TilePlan p;
    p.height = h_;
    p.width = w_;
    p.bf16 = mode_.bf16_convs;
    p.tile = tile;
    std::size_t block = 0, work = 0;
    double total = 0;
    for (const Option* o : path) {
      p.segments.push_back({o->first, o->last, o->band_rows});
      block = std::max(block, o->cost.block);
      work = std::max(work, o->cost.work_bytes);
      total += o->cost.work;
    }
    const Regions r = place_head(fixed_, block, work);
    p.workspace_bytes = r.block + r.work + fixed_.aux + fixed_.gn + fixed_.rgb;
    p.relative_cost = total / untiled_work_;
    return p;
  }

  std::size_t fixed_bytes() const { return fixed_.aux + fixed_.gn + fixed_.rgb; }
  const Geometry& geometry() const { return g_; }
  const Fixed& fixed() const { return fixed_; }
  const SegmentInfo& info(int first, int last) const {
    for (const SegmentInfo& si : infos_)
      if (si.first == first && si.last == last) return si;
    fail("tile plan segment [" + std::to_string(first) + ", " + std::to_string(last) + ") is not a valid segment");
  }

 private:
  Geometry g_;
  std::int64_t h_, w_;
  DecodeMode mode_;
  Fixed fixed_;
  std::vector<SegmentInfo> infos_;
  double untiled_work_ = 0;
};

std::vector<std::size_t> block_caps(const std::vector<Option>& opts) {
  std::vector<std::size_t> caps;
  for (const Option& o : opts) caps.push_back(o.cost.block);
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  return caps;
}

void check_grid(std::int64_t h, std::int64_t w) {
  if (h < 1 || w < 1) fail("latent grid must be at least 1x1");
}
void check_tile(int tile) {
  if (tile < 8 || tile % 8 != 0) fail("tile must be a positive multiple of 8 image pixels, got " + std::to_string(tile));
}

}  // namespace

// ---- Executor -----------------------------------------------------------------------------------------------

class TiledRunner {
 public:
  TiledRunner(const Flux2VaeDecoder& d, const Flux2VaeDecoder::Ctx& c, const Geometry& g, char* block,
              std::size_t block_bytes, std::array<char*, 3> bufs, void* accum, float* stats, char* rgb)
      : d_(d), c_(c), g_(g), block_(block), block_bytes_(block_bytes), bufs_(bufs), accum_(accum), stats_(stats), rgb_(rgb) {}

  // Runs one segment whose input sits at block offset in_off; returns the output's offset.
  std::size_t run_segment(const SegmentInfo& si, std::int64_t band_rows, std::size_t in_off) {
    const std::int64_t n = band_count(g_, si, band_rows);
    const Tensor in = act(block_ + in_off, c_.dt, g_.rows(si.in_level), g_.cols(si.in_level), si.in_ch);
    const std::size_t out_bytes = si.out_ch ? g_.row_bytes(si.out_level, si.out_ch) * static_cast<std::size_t>(g_.rows(si.out_level)) : 0;
    const bool top_down = in_off + in.nbytes() == block_bytes_;
    if (!top_down && in_off != 0) fail("internal: checkpoint not at either end of the block");
    const std::size_t out_off = top_down ? 0 : block_bytes_ - out_bytes;
    out_ = si.out_ch ? act(block_ + out_off, c_.dt, g_.rows(si.out_level), g_.cols(si.out_level), si.out_ch) : Tensor{};

    // Statistics passes: one per GroupNorm, each band computed up to that GroupNorm's input.
    if (n > 1) {
      for (int k = 0; k < si.groupnorms(); ++k) {
        const Target& t = si.targets[static_cast<std::size_t>(k)];
        vae::group_norm_accum_zero(accum_, 1, vae::kGroupNormGroups, c_.stream);
        if (t.direct) {
          vae::group_norm_accumulate(in, vae::kGroupNormGroups, accum_, c_.stream);
        } else {
          for (std::int64_t j = 0; j < n; ++j) {
            begin_band();
            const Band bd = band(g_, si, band_rows, j);
            const auto [a, b] = input_rows(g_, si, t, bd);
            Tile tile = start_tile(si, bd, a, b, in, 0);
            tile.gn_stop = k;
            if (run_units(si, tile)) fail("internal: statistics pass ran past its GroupNorm");
            end_band();
          }
        }
        finalize(k, t.level, t.ch);
      }
    }

    // Output pass with the gather(k) / commit(k-1) / compute(k) order the overlay relies on.
    int held = -1;
    Tile held_tile;
    for (std::int64_t idx = 0; idx < n; ++idx) {
      begin_band();
      const std::int64_t j = top_down ? idx : n - 1 - idx;
      const Band bd = band(g_, si, band_rows, j);
      const auto [a, b] = input_rows(g_, si, si.targets.back(), bd);
      const int gbuf = held < 0 ? 0 : (held + 1) % 3;
      Tile tile = start_tile(si, bd, a, b, in, gbuf);
      if (held >= 0) commit(si, held_tile);
      tile.inline_stats = n == 1;
      if (!run_units(si, tile)) fail("internal: output pass stopped early");
      held = tile.idx[static_cast<std::size_t>(tile.x)];
      held_tile = tile;
      end_band();
    }
    commit(si, held_tile);
    return out_off;
  }

 private:
  // Cancel point before every band. With a cancel flag the host waits for band k - 2 before issuing band k, so at
  // most one band is queued ahead of the one running when the cancel is seen.
  void begin_band() {
    if (d_.cancel_ == nullptr) return;
    if (bands_ >= 2) MI_CUDA_CHECK(cudaEventSynchronize(d_.band_done_[bands_ % 2]));
    if (d_.cancel_->load()) {
      MI_CUDA_CHECK(cudaStreamSynchronize(c_.stream));
      throw Cancelled();
    }
  }
  void end_band() {
    if (d_.cancel_ == nullptr) return;
    MI_CUDA_CHECK(cudaEventRecord(d_.band_done_[bands_ % 2], c_.stream));
    ++bands_;
  }

  using Resnet = Flux2VaeDecoder::Resnet;
  using Norm = Flux2VaeDecoder::Norm;

  struct Tile {
    std::array<int, 3> idx{};  // tile buffers (indices into bufs_); idx[x] holds the current main tensor
    int x = 0;
    int level = 0;
    std::int64_t row0 = 0, rows = 0;  // image rows of the tile tensors at `level`
    Band band{0, 0};
    int gn_next = 0;
    int gn_stop = -1;           // stats pass: accumulate this GroupNorm's input over the owned rows and stop
    bool inline_stats = false;  // single band: each GroupNorm's statistics come from the tile itself
  };

  char* buf(const Tile& t, int k) const { return bufs_[static_cast<std::size_t>(t.idx[static_cast<std::size_t>((t.x + k) % 3)])]; }

  Tile start_tile(const SegmentInfo& si, const Band& bd, std::int64_t a, std::int64_t b, const Tensor& in, int gbuf) {
    Tile t;
    t.idx = {gbuf, (gbuf + 1) % 3, (gbuf + 2) % 3};
    t.level = si.in_level;
    t.row0 = a;
    t.rows = b - a;
    t.band = bd;
    const std::size_t rb = g_.row_bytes(si.in_level, si.in_ch);
    MI_CUDA_CHECK(cudaMemcpyAsync(bufs_[static_cast<std::size_t>(gbuf)], static_cast<const char*>(in.data) + static_cast<std::size_t>(a) * rb,
                                  static_cast<std::size_t>(b - a) * rb, cudaMemcpyDeviceToDevice, c_.stream));
    owned_si_ = &si;
    return t;
  }

  Tensor tensor(const Tile& t, int k, int ch, DType dt) const { return act(buf(t, k), dt, t.rows, g_.cols(t.level), ch); }

  // Rows of x (a tile tensor at t.level) that the band owns.
  Tensor owned_rows(const Tile& t, const Tensor& x) const {
    const auto [a, b] = owned(*owned_si_, t.band, t.level);
    const std::size_t rb = static_cast<std::size_t>(x.dim(2) * x.dim(3)) * dtype_size(x.dtype);
    return act(static_cast<char*>(x.data) + static_cast<std::size_t>(a - t.row0) * rb, x.dtype, b - a, x.dim(2), x.dim(3));
  }

  Tensor stats(int k) const { return Tensor::view(stats_ + static_cast<std::ptrdiff_t>(k) * vae::kGroupNormGroups * 2, DType::kF32, Device::kCUDA, {1, vae::kGroupNormGroups, 2}); }

  void finalize(int k, int level, int ch) {
    if (k >= kStatsSlots) fail("internal: too many GroupNorms in a segment");
    vae::group_norm_finalize(accum_, 1, vae::kGroupNormGroups, g_.rows(level) * g_.cols(level) * (ch / vae::kGroupNormGroups),
                             vae::kResnetEps, stats(k), c_.stream);
  }

  // GroupNorm (+ SiLU) of x into y with global statistics; false when the stats pass stops here.
  bool norm(Tile& t, const Tensor& x, const Norm& n, bool silu, const Tensor& y) {
    const int k = t.gn_next++;
    if (k == t.gn_stop) {
      vae::group_norm_accumulate(owned_rows(t, x), vae::kGroupNormGroups, accum_, c_.stream);
      return false;
    }
    if (t.inline_stats) {
      vae::group_norm_accum_zero(accum_, 1, vae::kGroupNormGroups, c_.stream);
      vae::group_norm_accumulate(owned_rows(t, x), vae::kGroupNormGroups, accum_, c_.stream);
      finalize(k, t.level, static_cast<int>(x.dim(3)));
    }
    vae::group_norm_apply(x, stats(k), n.gamma, n.beta, silu, y, c_.stream);
    return true;
  }

  // Same op order and buffer roles as Flux2VaeDecoder::resnet().
  bool resnet(Tile& t, const Resnet& r) {
    const Tensor xt = tensor(t, 0, r.in_ch, c_.dt);
    const Tensor bt = tensor(t, 2, r.out_ch, c_.dt);
    if (r.shortcut.w.data != nullptr) {
      if (t.gn_next == t.gn_stop) return norm(t, xt, r.norm1, true, xt);  // the shortcut is not needed yet
      const Tensor st = tensor(t, 1, r.out_ch, c_.dt);
      d_.conv(c_, xt, r.shortcut, nullptr, st);
      norm(t, xt, r.norm1, true, xt);
      d_.conv(c_, xt, r.conv1, nullptr, bt);
      if (!norm(t, bt, r.norm2, true, bt)) return false;
      d_.conv(c_, bt, r.conv2, &st, st);
      t.x = (t.x + 1) % 3;
      return true;
    }
    const Tensor at = tensor(t, 1, r.in_ch, c_.dt);
    if (!norm(t, xt, r.norm1, true, at)) return false;
    d_.conv(c_, at, r.conv1, nullptr, bt);
    if (!norm(t, bt, r.norm2, true, bt)) return false;
    d_.conv(c_, bt, r.conv2, &xt, xt);
    return true;
  }

  bool unit(Tile& t, int u) {
    const UnitShape& us = kUnits[static_cast<std::size_t>(u)];
    switch (us.kind) {
      case UnitKind::kResnet:
        return resnet(t, u == 0 ? d_.mid_resnets_[1] : d_.up_[(u - 1) / 4].resnets[(u - 1) % 4]);
      case UnitKind::kUpsample: {
        const Tensor xt = tensor(t, 0, us.in_ch, c_.dt);
        ++t.level;
        t.row0 *= 2;
        t.rows *= 2;
        const Tensor up = tensor(t, 1, us.in_ch, c_.dt);
        vae::upsample_nearest2x(xt, up, c_.stream);
        d_.conv(c_, up, d_.up_[(u - 1) / 4].upsampler, nullptr, tensor(t, 0, us.out_ch, c_.dt));
        return true;
      }
      case UnitKind::kOut: {
        const Tensor xt = tensor(t, 0, us.in_ch, c_.dt);
        if (!norm(t, xt, d_.norm_out_, true, xt)) return false;
        const Tensor o = tensor(t, 1, 3, c_.dt);
        d_.conv(c_, xt, d_.conv_out_, nullptr, o);
        if (c_.dt == DType::kBF16) {
          kernels::cast(o, tensor(t, 2, 3, DType::kF32), c_.stream);
          t.x = (t.x + 2) % 3;
        } else {
          t.x = (t.x + 1) % 3;
        }
        return true;
      }
    }
    return false;
  }

  bool run_units(const SegmentInfo& si, Tile& t) {
    for (int u = si.first; u < si.last; ++u)
      if (!unit(t, u)) return false;
    return true;
  }

  // Copies the band's owned output rows into the output checkpoint (or converts them to RGB8).
  void commit(const SegmentInfo& si, const Tile& t) {
    const auto [a, b] = owned(si, t.band, t.level);
    if (si.out_ch == 0) {
      const Tensor src = owned_rows(t, tensor(t, 0, 3, DType::kF32));
      const std::size_t rb = static_cast<std::size_t>(g_.cols(t.level)) * 3;
      vae::postprocess_to_rgb8(src, Tensor::view(rgb_ + static_cast<std::size_t>(a) * rb, DType::kU8, Device::kCUDA, {1, b - a, g_.cols(t.level), 3}),
                               c_.stream);
      return;
    }
    const Tensor src = owned_rows(t, tensor(t, 0, si.out_ch, c_.dt));
    const std::size_t rb = g_.row_bytes(si.out_level, si.out_ch);
    MI_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(out_.data) + static_cast<std::size_t>(a) * rb, src.data, src.nbytes(),
                                  cudaMemcpyDeviceToDevice, c_.stream));
  }

  const Flux2VaeDecoder& d_;
  const Flux2VaeDecoder::Ctx& c_;
  const Geometry& g_;
  char* block_;
  std::size_t block_bytes_;
  std::array<char*, 3> bufs_;
  void* accum_;
  float* stats_;
  char* rgb_;
  Tensor out_;
  const SegmentInfo* owned_si_ = nullptr;
  std::int64_t bands_ = 0;  // bands issued so far in this decode (cancel throttle)
};

// ---- Public API ---------------------------------------------------------------------------------------------

std::string TilePlan::describe() const {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "tile %d px, %.3f GiB, est. %.2fx tail work:", tile,
                static_cast<double>(workspace_bytes) / static_cast<double>(std::size_t{1} << 30), relative_cost);
  std::string s = buf;
  for (const Segment& sg : segments) {
    const int lout = out_level(kUnits[static_cast<std::size_t>(sg.end_unit - 1)]);
    const std::int64_t rows = (2 * height) << lout;
    s += " [" + std::to_string(sg.first_unit) + "," + std::to_string(sg.end_unit) + ")";
    s += sg.band_rows >= rows ? "whole" : "x" + std::to_string((rows + sg.band_rows - 1) / sg.band_rows);
  }
  return s;
}

std::optional<TilePlan> Flux2VaeDecoder::plan_tiled(std::int64_t height, std::int64_t width, const DecodeMode& mode,
                                                    std::size_t budget_bytes, int tile) {
  check_grid(height, width);
  if (tile != 0) check_tile(tile);
  const Planner p(height, width, mode);
  const std::size_t fixed = p.fixed_bytes();
  const Fixed& f = p.fixed();
  std::optional<TilePlan> best;
  for (const int t : tile ? std::vector<int>{tile} : p.tiles()) {
    const std::vector<Option> opts = p.options(t);
    for (const std::size_t cap : block_caps(opts)) {
      if (cap + fixed >= budget_bytes) break;
      const std::size_t work_cap = budget_bytes - fixed - cap;
      if (f.head_in > cap && f.head_out > work_cap) continue;  // the head fits neither region
      const auto path = Planner::best_path(opts, cap, work_cap);
      if (path.empty()) continue;
      TilePlan plan = p.finish(t, path);
      if (!best || plan.relative_cost < best->relative_cost ||
          (plan.relative_cost == best->relative_cost && plan.workspace_bytes < best->workspace_bytes))
        best = std::move(plan);
    }
  }
  return best;
}

TilePlan Flux2VaeDecoder::plan_tiled_min_memory(std::int64_t height, std::int64_t width, const DecodeMode& mode, int tile) {
  check_grid(height, width);
  check_tile(tile);
  const Planner p(height, width, mode);
  const std::vector<Option> opts = p.options(tile);
  // For every block cap, the smallest tile-buffer cap that still admits a segmentation (minimax over paths).
  std::size_t best_total = std::numeric_limits<std::size_t>::max(), best_block = 0, best_work = 0;
  for (const std::size_t cap : block_caps(opts)) {
    std::array<std::size_t, kTailUnits + 1> mm;
    mm.fill(std::numeric_limits<std::size_t>::max());
    mm[0] = 0;
    for (int j = 1; j <= kTailUnits; ++j)
      for (const Option& o : opts)
        if (o.last == j && o.cost.block <= cap && mm[static_cast<std::size_t>(o.first)] != std::numeric_limits<std::size_t>::max())
          mm[static_cast<std::size_t>(j)] = std::min(mm[static_cast<std::size_t>(j)], std::max(mm[static_cast<std::size_t>(o.first)], o.cost.work_bytes));
    if (mm[kTailUnits] == std::numeric_limits<std::size_t>::max()) continue;
    const std::size_t work = std::max(mm[kTailUnits], p.fixed().head_in <= cap ? 0 : p.fixed().head_out);
    if (cap + work < best_total) best_total = cap + work, best_block = cap, best_work = work;
  }
  const auto path = Planner::best_path(opts, best_block, best_work);
  if (path.empty()) fail("internal: no tiled schedule");
  return p.finish(tile, path);
}

std::size_t Flux2VaeDecoder::tiled_workspace_bytes(std::int64_t height, std::int64_t width, int tile, const DecodeMode& mode) {
  return plan_tiled_min_memory(height, width, mode, tile).workspace_bytes;
}

void Flux2VaeDecoder::decode_tiled(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode, const TilePlan& plan,
                                   cudaStream_t stream, const StageHook& hook) {
  check_decode_args(latents, rgb, mode, has_bf16_);
  if (mode.torch_group_norm)
    fail("decode_tiled needs DecodeMode::torch_group_norm == false (torch's GroupNorm statistics cannot be split into tiles)");
  const std::int64_t h = latents.dim(-2), w = latents.dim(-1);
  if (plan.height != h || plan.width != w || plan.bf16 != mode.bf16_convs) fail("tile plan was made for another latent grid or mode");
  const Planner p(h, w, mode);
  const Geometry& g = p.geometry();
  // Re-derive and validate the layout from the plan.
  std::size_t block = 0, work = 0, buf = 0;
  bool strip = false;
  int next = 0;
  std::vector<const SegmentInfo*> infos;
  for (const TilePlan::Segment& s : plan.segments) {
    if (s.first_unit != next || s.end_unit <= s.first_unit || s.end_unit > kTailUnits) fail("tile plan segments do not chain the tail");
    const SegmentInfo& si = p.info(s.first_unit, s.end_unit);
    const std::int64_t align = std::int64_t{1} << (si.out_level - si.in_level);
    if (s.band_rows < 1 || (s.band_rows < g.rows(si.out_level) && s.band_rows % align != 0))
      fail("tile plan band height off the band grid");
    const SegmentCost cost = segment_cost(g, si, s.band_rows);
    block = std::max(block, cost.block);
    work = std::max(work, cost.work_bytes);
    buf = std::max(buf, cost.buf);
    strip = strip || cost.buf >= kCutlassMaxBytes;
    infos.push_back(&si);
    next = s.end_unit;
  }
  if (next != kTailUnits) fail("tile plan does not cover the tail");
  const Fixed& f = p.fixed();
  const Regions regions = place_head(f, block, work);
  block = regions.block;
  work = regions.work;
  if (block + work + f.aux + f.gn + f.rgb != plan.workspace_bytes) fail("tile plan workspace out of date");

  runtime::StackScope scope(arena_);
  char* ws = static_cast<char*>(arena_.alloc_stack(plan.workspace_bytes));
  char* block_p = ws;
  char* work_p = block_p + block;
  char* aux = work_p + work;
  char* gn = aux + f.aux;
  char* rgb_stage = gn + f.gn;
  char* strip_p = work_p + 3 * buf;
  Ctx c{stream, DType::kF32, &mode, aux, f.aux, nullptr, nullptr, 0, strip ? strip_p : nullptr, strip ? kStripScratchBytes : 0, h, w, &hook};

  // Head (untiled); its output becomes the first checkpoint at the end of the block (see Fixed).
  const std::size_t in_bytes = g.row_bytes(0, kMidCh) * static_cast<std::size_t>(g.rows(0));
  std::size_t in_off = block - in_bytes;
  char* head_area = regions.head_inside ? block_p : work_p;
  char* x = mode.bf16_convs ? head_area : block_p + in_off;
  char* scratch = mode.bf16_convs ? head_area + f.head_x : head_area;
  run_head(c, latents, x, scratch, scratch + f.head_x, scratch, f.head_scratch);
  c.dt = mode.bf16_convs ? DType::kBF16 : DType::kF32;
  if (c.dt == DType::kBF16)
    kernels::cast(act(x, DType::kF32, g.rows(0), g.cols(0), kMidCh), act(block_p + in_off, DType::kBF16, g.rows(0), g.cols(0), kMidCh), stream);

  char* rgb_dst = rgb.device == Device::kCUDA ? static_cast<char*>(rgb.data) : rgb_stage;
  const std::size_t accum_bytes = align_up(vae::group_norm_accum_bytes(1, vae::kGroupNormGroups));
  TiledRunner runner(*this, c, g, block_p, block, {work_p, work_p + buf, work_p + 2 * buf}, gn,
                     reinterpret_cast<float*>(gn + accum_bytes), rgb_dst);
  for (std::size_t i = 0; i < infos.size(); ++i) in_off = runner.run_segment(*infos[i], plan.segments[i].band_rows, in_off);
  if (rgb.device != Device::kCUDA) {
    MI_CUDA_CHECK(cudaMemcpyAsync(rgb.data, rgb_stage, static_cast<std::size_t>(rgb.numel()), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

std::optional<TilePlan> Flux2VaeDecoder::decode_within(const Tensor& latents, const Tensor& rgb, const DecodeMode& mode,
                                                       std::size_t budget_bytes, cudaStream_t stream) {
  check_decode_args(latents, rgb, mode, has_bf16_);
  const std::int64_t h = latents.dim(-2), w = latents.dim(-1);
  if (workspace_bytes(h, w, mode) <= budget_bytes) {
    decode(latents, rgb, mode, stream);
    return std::nullopt;
  }
  DecodeMode tiled = mode;
  tiled.torch_group_norm = false;
  std::optional<TilePlan> plan = plan_tiled(h, w, tiled, budget_bytes);
  if (!plan) {
    throw std::runtime_error("mi::flux2_vae: no decode schedule fits " + std::to_string(budget_bytes) + " bytes (smallest tiled: " +
                             std::to_string(tiled_workspace_bytes(h, w, 8, tiled)) + ")");
  }
  decode_tiled(latents, rgb, tiled, *plan, stream);
  return plan;
}

}  // namespace mi::flux2_vae
