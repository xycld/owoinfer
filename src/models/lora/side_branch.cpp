#include "models/lora/side_branch.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include "kernels/lora/lora_kernels.h"
#include "weights/quant/quant_codec.h"

namespace mi::lora {

namespace {

constexpr std::size_t kAlign = 256;
constexpr std::int64_t kRankAlign = 8;  // bf16 elements per 16 bytes

std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
std::int64_t round_rank(std::int64_t r) { return (r + kRankAlign - 1) / kRankAlign * kRankAlign; }

[[noreturn]] void fail(const std::string& msg) { throw std::invalid_argument("mi::lora: " + msg); }

Site site_of(BlockGemm g) {
  switch (g) {
    case BlockGemm::kQ: case BlockGemm::kK: case BlockGemm::kV: return Site::kQkv;
    case BlockGemm::kO: return Site::kO;
    case BlockGemm::kUpGate: return Site::kUpGate;
    case BlockGemm::kDown: return Site::kDown;
  }
  return Site::kQkv;
}

// Process-wide source of generation() values.
std::uint64_t next_generation() {
  static std::atomic<std::uint64_t> counter{0};
  return ++counter;
}

// Layout of one site, shared by device_bytes() and the constructor. A site's t columns are [shared | own of slice 0 |
// own of slice 1 | ...]; a slice reads the shared range (fused-qkv adapters) and its own range.
struct Run {  // one adapter's rank columns within a range
  std::size_t adapter;  // the stack adapter (for a fused-qkv adapter: its part on this slice)
  std::int64_t offset;  // column within the range
};
struct SlicePlan {
  std::int64_t row0 = 0, rows = 0;
  std::int64_t own_col0 = 0, own_cols = 0;
  std::vector<Run> own;
  std::vector<std::size_t> shared_parts;  // per shared group: this slice's adapter
  std::size_t b_offset = 0;
};
struct SitePlan {
  std::int64_t n = 0, k = 0, rank = 0;
  std::int64_t shared_cols = 0;              // range [0, shared_cols)
  std::vector<std::int64_t> shared_offset;   // per shared group (fused-qkv adapter): first column
  std::vector<std::size_t> shared_first;     // per shared group: one of its adapters (A, coef)
  std::vector<SlicePlan> slices;
  std::size_t a_offset = 0, coef_offset = 0;
};

struct Plan {
  int blocks = 0;
  std::vector<SitePlan> sites;  // [block * kSites + site]
  std::size_t coef_bytes = 0, bytes = 0;
};

Plan make_plan(const LoraStack& stack) {
  const TargetModel& model = stack.model();
  Plan plan;
  plan.blocks = model.blocks;
  plan.sites.resize(static_cast<std::size_t>(model.blocks) * kSites);
  std::vector<std::map<std::pair<int, std::string>, std::size_t>> groups(plan.sites.size());  // (file, source) -> group
  for (std::size_t i = 0; i < stack.adapters().size(); ++i) {
    const Adapter& a = stack.adapters()[i];
    if (!a.side_branch) continue;
    const TargetModule* t = model.find(a.module);
    const Site s = site_of(t->gemm);
    const std::size_t e = static_cast<std::size_t>(t->block) * kSites + static_cast<std::size_t>(s);
    SitePlan& sp = plan.sites[e];
    std::int64_t row0 = 0;
    if (s == Site::kQkv) row0 = t->gemm == BlockGemm::kQ ? 0 : t->gemm == BlockGemm::kK ? model.dim : 2 * model.dim;
    sp.n = s == Site::kQkv ? 3 * model.dim : t->n;
    sp.k = t->k;
    auto it = std::find_if(sp.slices.begin(), sp.slices.end(), [&](const SlicePlan& p) { return p.row0 == row0; });
    if (it == sp.slices.end()) {
      sp.slices.push_back({row0, t->n, 0, 0, {}, {}, 0});
      it = sp.slices.end() - 1;
    }
    if (a.stored_rows != a.n) {  // a part of a fused-qkv adapter: one shared group per file adapter
      const auto key = std::make_pair(a.lora, a.source);
      auto g = groups[e].find(key);
      if (g == groups[e].end()) {
        g = groups[e].emplace(key, sp.shared_first.size()).first;
        sp.shared_first.push_back(i);
        sp.shared_offset.push_back(sp.shared_cols);
        sp.shared_cols += a.rank;
      }
      if (it->shared_parts.size() <= g->second) it->shared_parts.resize(g->second + 1, SIZE_MAX);
      it->shared_parts[g->second] = i;
    } else {
      it->own.push_back({i, it->own_cols});
      it->own_cols += a.rank;
    }
  }
  std::size_t coef = 0;  // coefficients first, then the factors
  for (SitePlan& sp : plan.sites) {
    if (sp.slices.empty()) continue;
    std::sort(sp.slices.begin(), sp.slices.end(), [](const SlicePlan& x, const SlicePlan& y) { return x.row0 < y.row0; });
    sp.shared_cols = round_rank(sp.shared_cols);
    std::int64_t col = sp.shared_cols;
    for (SlicePlan& sl : sp.slices) {
      sl.own_col0 = col;
      sl.own_cols = round_rank(sl.own_cols);
      col += sl.own_cols;
    }
    sp.rank = col;
    sp.coef_offset = coef;
    coef += static_cast<std::size_t>(sp.rank) * 4;
  }
  plan.coef_bytes = coef;
  std::size_t off = align_up(coef);
  for (SitePlan& sp : plan.sites) {
    if (sp.slices.empty()) continue;
    sp.a_offset = off;
    off = align_up(off + static_cast<std::size_t>(sp.rank * sp.k) * 2);
    for (SlicePlan& sl : sp.slices) {
      sl.b_offset = off;
      off = align_up(off + static_cast<std::size_t>(sl.rows * (sp.shared_cols + sl.own_cols)) * 2);
    }
  }
  plan.bytes = off;
  return plan;
}

Tensor view_at(void* base, std::size_t offset, DType dtype, std::initializer_list<std::int64_t> shape) {
  return Tensor::view(static_cast<std::byte*>(base) + offset, dtype, Device::kCUDA, shape);
}

}  // namespace

const char* site_name(Site s) {
  switch (s) {
    case Site::kQkv: return "qkv";
    case Site::kO: return "o";
    case Site::kUpGate: return "ffn_up_gate";
    case Site::kDown: return "ffn_down";
  }
  return "?";
}

std::size_t SideBranches::device_bytes(const LoraStack& stack) { return make_plan(stack).bytes; }

SideBranches::SideBranches(const LoraStack& stack) : stack_(&stack), adapters_(stack.adapters().size()) {
  const Plan plan = make_plan(stack);
  blocks_ = plan.blocks;
  entries_.resize(plan.sites.size());
  generation_ = next_generation();
  if (plan.bytes == 0) return;
  coef_bytes_ = plan.coef_bytes;
  buffer_ = DeviceBuffer(plan.bytes);
  std::vector<std::uint16_t> host(plan.bytes / 2, 0);  // bf16 staging of every factor (coefficients written later)
  const auto at = [&](std::size_t offset) { return host.data() + offset / 2; };
  for (std::size_t e = 0; e < plan.sites.size(); ++e) {
    const SitePlan& sp = plan.sites[e];
    if (sp.slices.empty()) continue;
    Entry& en = entries_[e];
    en.present = true;
    SiteBranch& br = en.branch;
    br.n = sp.n;
    br.k = sp.k;
    br.rank = sp.rank;
    br.a = view_at(buffer_.get(), sp.a_offset, DType::kBF16, {sp.rank, sp.k});
    br.coef = view_at(buffer_.get(), sp.coef_offset, DType::kF32, {sp.rank});
    max_rank_ = std::max(max_rank_, sp.rank);
    const bool permute = static_cast<Site>(e % kSites) == Site::kUpGate;
    const auto put_a = [&](const Adapter& a, std::int64_t col0) {
      const std::vector<float> down = stack.read(a.parts[0]);  // [r, k]
      std::uint16_t* arow = at(sp.a_offset) + col0 * sp.k;
      for (std::size_t i = 0; i < down.size(); ++i) arow[i] = quant::float_to_bf16(down[i]);
    };
    for (std::size_t g = 0; g < sp.shared_first.size(); ++g) {
      const Adapter& a = stack.adapters()[sp.shared_first[g]];
      put_a(a, sp.shared_offset[g]);
      en.columns.push_back({sp.shared_first[g], sp.shared_offset[g], a.rank});
    }
    for (const SlicePlan& sl : sp.slices) {
      const std::int64_t cols = sp.shared_cols + sl.own_cols;
      SiteBranch::Slice out;
      out.row0 = sl.row0;
      out.rows = sl.rows;
      if (sp.shared_cols > 0) {
        out.col0 = 0;
        out.cols = sp.shared_cols;
        out.col0b = sl.own_col0;
        out.colsb = sl.own_cols;
      } else {
        out.col0 = sl.own_col0;
        out.cols = sl.own_cols;
      }
      out.b = view_at(buffer_.get(), sl.b_offset, DType::kBF16, {sl.rows, cols});
      br.slices.push_back(out);
      std::uint16_t* b = at(sl.b_offset);
      const auto put_b = [&](const Adapter& a, std::int64_t bcol) {
        const std::vector<float> up = stack.read(a.parts[1], a.row0, a.n);  // [n, r]
        for (std::int64_t c = 0; c < sl.rows; ++c) {
          const std::int64_t src = permute ? kernels::lora::up_gate_gemm_row(c, sl.rows / 2) : c;
          for (std::int64_t j = 0; j < a.rank; ++j) {
            b[c * cols + bcol + j] = quant::float_to_bf16(up[static_cast<std::size_t>(src * a.rank + j)]);
          }
        }
      };
      for (std::size_t g = 0; g < sl.shared_parts.size(); ++g) {
        if (sl.shared_parts[g] != SIZE_MAX) put_b(stack.adapters()[sl.shared_parts[g]], sp.shared_offset[g]);
      }
      for (const Run& r : sl.own) {
        const Adapter& a = stack.adapters()[r.adapter];
        put_a(a, sl.own_col0 + r.offset);
        put_b(a, sp.shared_cols + r.offset);
        en.columns.push_back({r.adapter, sl.own_col0 + r.offset, a.rank});
      }
    }
  }
  buffer_.upload(host.data(), plan.bytes);
  upload_coefs(stack);
}

void SideBranches::upload_coefs(const LoraStack& stack) {
  // Every coefficient first (validation before any upload), then one copy of the whole coefficient region.
  std::vector<float> coef(coef_bytes_ / sizeof(float), 0.0f);
  std::vector<bool> active(entries_.size(), false);
  for (std::size_t e = 0; e < entries_.size(); ++e) {
    const Entry& en = entries_[e];
    if (!en.present) continue;
    const std::size_t base = static_cast<std::size_t>(static_cast<std::byte*>(en.branch.coef.data) - static_cast<std::byte*>(buffer_.get())) / sizeof(float);
    for (const Column& c : en.columns) {
      const float v = stack.coef(stack.adapters()[c.adapter]);
      if (!std::isfinite(v)) fail("coefficient strength * alpha / rank of '" + stack.adapters()[c.adapter].source + "' is not finite");
      std::fill(coef.begin() + static_cast<std::ptrdiff_t>(base + static_cast<std::size_t>(c.col0)),
                coef.begin() + static_cast<std::ptrdiff_t>(base + static_cast<std::size_t>(c.col0 + c.rank)), v);
      if (v != 0.0f) active[e] = true;
    }
  }
  if (!coef.empty()) buffer_.upload(coef.data(), coef.size() * sizeof(float), 0);
  bool changed = false;
  for (std::size_t e = 0; e < entries_.size(); ++e) {
    changed |= entries_[e].active != active[e];
    entries_[e].active = active[e];
  }
  if (changed) generation_ = next_generation();
}

void SideBranches::update_strengths(const LoraStack& stack) {
  if (&stack != stack_ || stack.adapters().size() != adapters_) {
    fail("update_strengths: not the LoraStack these side branches were built from");
  }
  upload_coefs(stack);
}

const SiteBranch* SideBranches::site(int block, Site s) const {
  if (block < 0 || block >= blocks_) fail("block " + std::to_string(block) + " out of range");
  const Entry& e = entries_[static_cast<std::size_t>(block) * kSites + static_cast<std::size_t>(s)];
  return e.active ? &e.branch : nullptr;
}

bool SideBranches::any_active(int block) const {
  for (int s = 0; s < kSites; ++s) {
    if (site(block, static_cast<Site>(s)) != nullptr) return true;
  }
  return false;
}

bool SideBranches::any_active() const {
  return std::any_of(entries_.begin(), entries_.end(), [](const Entry& e) { return e.active; });
}

std::size_t t_workspace_bytes(std::int64_t m, std::int64_t rank) {
  if (m < 0 || rank < 0) fail("t_workspace_bytes: negative size");
  return align_up(static_cast<std::size_t>(m) * static_cast<std::size_t>(rank) * 4);
}

void compute_t(const SiteBranch& site, const Tensor& x, const Tensor& t, void* workspace, std::size_t workspace_bytes,
               kernels::Gemm& gemm, void* gemm_ws, std::size_t gemm_ws_bytes, cudaStream_t stream) {
  if (x.device != Device::kCUDA || x.dtype != DType::kBF16 || x.rank != 2 || x.dim(1) != site.k || x.stride[1] != 1) {
    fail("compute_t: x must be CUDA bf16 [m, " + std::to_string(site.k) + "] rows, got " + x.describe());
  }
  const std::int64_t m = x.dim(0), r = site.rank;
  if (t.device != Device::kCUDA || t.dtype != DType::kBF16 || t.rank != 2 || t.dim(0) != m || t.dim(1) != r ||
      t.stride[1] != 1 || t.stride[0] < r || t.stride[0] % 8 != 0 || reinterpret_cast<std::uintptr_t>(t.data) % 16 != 0) {
    fail("compute_t: t must be CUDA bf16 [" + std::to_string(m) + ", " + std::to_string(r) +
         "] rows (16-byte aligned, stride a multiple of 8), got " + t.describe());
  }
  if (workspace_bytes < t_workspace_bytes(m, r) || reinterpret_cast<std::uintptr_t>(workspace) % kAlign != 0) {
    fail("compute_t: workspace too small or not 256-byte aligned");
  }
  if (m == 0) return;
  const Tensor t32 = Tensor::view(workspace, DType::kF32, Device::kCUDA, {m, r});
  gemm.linear(t32, x, site.a, nullptr, gemm_ws, gemm_ws_bytes, stream);
  kernels::lora::scale_columns_bf16(t32.ptr<const float>(), r, site.coef.ptr<const float>(),
                                    static_cast<__nv_bfloat16*>(t.data), t.stride[0], m, r, stream);
}

kernels::lora::LoraSlices lora_slices(const SiteBranch& site, const Tensor& t) {
  if (t.dtype != DType::kBF16 || t.rank != 2 || t.dim(1) != site.rank) fail("lora_slices: t does not match the site");
  if (site.slices.size() > static_cast<std::size_t>(kernels::lora::LoraSlices::kMaxSlices)) fail("lora_slices: too many slices");
  kernels::lora::LoraSlices l;
  l.t = static_cast<const __nv_bfloat16*>(t.data);
  l.ldt = t.stride[0];
  l.count = static_cast<int>(site.slices.size());
  for (std::size_t i = 0; i < site.slices.size(); ++i) {
    const SiteBranch::Slice& s = site.slices[i];
    l.slice[i] = {s.col0, s.cols, s.col0b, s.colsb, s.row0, s.rows, static_cast<const __nv_bfloat16*>(s.b.data)};
  }
  return l;
}

}  // namespace mi::lora
