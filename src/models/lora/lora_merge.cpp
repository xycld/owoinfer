// Host merges of adapters into base weights (see lora.h). Compiled without FMA contraction or fast-math
// (cmake/modules/lora.cmake): every fp32 op is rounded on its own, as the converter's merge.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "models/lora/lora.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_convert.h"

namespace mi::lora {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw LoraError("mi::lora: " + msg); }

// Rows per merge step of one thread (bounds the fp32 scratch: rows * k floats).
constexpr std::int64_t kRowBlock = 128;

// The row-independent factors of one adapter, read once.
struct Prepared {
  const Adapter* a = nullptr;
  float coef = 0;
  std::vector<float> down, w1b, w2b;  // kLowRank: down; kLoHa: w1b, w2b
  std::vector<float> kw1, kw2;        // kLoKr: w1 [a, b], w2 [c, d] (factored ones multiplied out)
  std::int64_t a1 = 0, b1 = 0, c2 = 0, d2 = 0;
  std::vector<float> bias;            // diff_b rows of the target
};

// f32(Σ_r x[i, r] * y[r, j]) with the sum in double (a [rows, inner] times [inner, cols] product).
std::vector<float> matmul_f64(const std::vector<float>& x, const std::vector<float>& y, std::int64_t rows,
                              std::int64_t inner, std::int64_t cols) {
  std::vector<float> out(static_cast<std::size_t>(rows * cols));
  std::vector<double> acc(static_cast<std::size_t>(cols));
  for (std::int64_t i = 0; i < rows; ++i) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (std::int64_t r = 0; r < inner; ++r) {
      const double u = x[static_cast<std::size_t>(i * inner + r)];
      const float* yr = y.data() + r * cols;
      for (std::int64_t c = 0; c < cols; ++c) acc[static_cast<std::size_t>(c)] += u * static_cast<double>(yr[c]);
    }
    for (std::int64_t c = 0; c < cols; ++c) out[static_cast<std::size_t>(i * cols + c)] = static_cast<float>(acc[static_cast<std::size_t>(c)]);
  }
  return out;
}

std::vector<Prepared> prepare(const LoraStack& stack, std::string_view module, bool include_side_branch) {
  std::vector<Prepared> out;
  for (const Adapter* a : stack.adapters_of(module)) {
    if (a->side_branch && !include_side_branch) continue;
    Prepared p;
    p.a = a;
    p.coef = stack.coef(*a);
    if (!std::isfinite(p.coef)) fail("coefficient strength * alpha / rank of '" + a->source + "' is not finite");
    if (p.coef == 0.0f) continue;  // strength 0: the weight stays bit-identical
    switch (a->kind) {
      case AdapterKind::kLowRank: p.down = stack.read(a->parts[0]); break;
      case AdapterKind::kLoHa:
        p.w1b = stack.read(a->parts[1]);
        p.w2b = stack.read(a->parts[3]);
        break;
      case AdapterKind::kLoKr: {
        std::size_t i = 0;
        for (int w = 0; w < 2; ++w) {
          std::vector<float>& dst = w == 0 ? p.kw1 : p.kw2;
          std::int64_t& rows = w == 0 ? p.a1 : p.c2;
          std::int64_t& cols = w == 0 ? p.b1 : p.d2;
          if (a->lokr_split[w]) {
            const TensorRef& ta = a->parts[i++];
            const TensorRef& tb = a->parts[i++];
            rows = ta.shape[0];
            cols = tb.shape[1];
            dst = matmul_f64(stack.read(ta), stack.read(tb), rows, ta.shape[1], cols);
          } else {
            const TensorRef& t = a->parts[i++];
            rows = t.shape[0];
            cols = t.shape[1];
            dst = stack.read(t);
          }
        }
        break;
      }
      case AdapterKind::kFull: break;
    }
    if (a->bias_diff) p.bias = stack.read(*a->bias_diff, a->row0, a->n);
    out.push_back(std::move(p));
  }
  return out;
}

// w [rows, k] (target rows row0..row0+rows) += coef * delta for one prepared adapter.
void apply(const LoraStack& stack, const Prepared& p, std::int64_t row0, std::int64_t rows, float* w, float* bias) {
  const Adapter& a = *p.a;
  const std::int64_t k = a.k, sr0 = a.row0 + row0;  // rows of the stored delta
  switch (a.kind) {
    case AdapterKind::kLowRank: {
      const std::vector<float> up = stack.read(a.parts[1], sr0, rows);
      std::vector<double> acc(static_cast<std::size_t>(k));
      for (std::int64_t i = 0; i < rows; ++i) {
        quant::lora_merge_row(w + i * k, k, up.data() + i * a.rank, p.down.data(), a.rank, p.coef, acc.data());
      }
      break;
    }
    case AdapterKind::kLoHa: {
      const std::int64_t r1 = a.parts[1].shape[0], r2 = a.parts[3].shape[0];
      const std::vector<float> m1 = matmul_f64(stack.read(a.parts[0], sr0, rows), p.w1b, rows, r1, k);
      const std::vector<float> m2 = matmul_f64(stack.read(a.parts[2], sr0, rows), p.w2b, rows, r2, k);
      for (std::int64_t i = 0; i < rows * k; ++i) {
        const std::size_t j = static_cast<std::size_t>(i);
        w[i] = w[i] + p.coef * (m1[j] * m2[j]);
      }
      break;
    }
    case AdapterKind::kLoKr: {
      // kron(w1 [a1, b1], w2 [c2, d2]) [a1 * c2, b1 * d2]: element (i1 * c2 + i2, j1 * d2 + j2) = w1[i1, j1] * w2[i2, j2].
      for (std::int64_t i = 0; i < rows; ++i) {
        const std::int64_t s = sr0 + i, i1 = s / p.c2, i2 = s % p.c2;
        float* wr = w + i * k;
        for (std::int64_t j = 0; j < k; ++j) {
          const std::int64_t j1 = j / p.d2, j2 = j % p.d2;
          const float d = p.kw1[static_cast<std::size_t>(i1 * p.b1 + j1)] * p.kw2[static_cast<std::size_t>(i2 * p.d2 + j2)];
          wr[j] = wr[j] + p.coef * d;
        }
      }
      break;
    }
    case AdapterKind::kFull: {
      if (a.parts.empty()) break;
      const std::vector<float> diff = stack.read(a.parts[0], sr0, rows);
      for (std::int64_t i = 0; i < rows * k; ++i) w[i] = w[i] + p.coef * diff[static_cast<std::size_t>(i)];
      break;
    }
  }
  if (bias != nullptr && !p.bias.empty()) {
    for (std::int64_t i = 0; i < rows; ++i) bias[i] = bias[i] + p.coef * p.bias[static_cast<std::size_t>(row0 + i)];
  }
}

const TargetModule& target(const LoraStack& stack, std::string_view module, std::int64_t n, std::int64_t k) {
  const TargetModule* t = stack.model().find(module);
  if (t == nullptr) fail("no module '" + std::string(module) + "' in the model");
  if (t->n != n || t->k != k) {
    fail("module '" + std::string(module) + "' is [" + std::to_string(t->n) + ", " + std::to_string(t->k) + "], got [" +
         std::to_string(n) + ", " + std::to_string(k) + "]");
  }
  return *t;
}

}  // namespace

int add_merged_rows(const LoraStack& stack, std::string_view module, std::int64_t row0, std::int64_t rows, float* w,
                    float* bias, bool include_side_branch) {
  const TargetModule* t = stack.model().find(module);
  if (t == nullptr) fail("no module '" + std::string(module) + "' in the model");
  if (row0 < 0 || rows < 0 || row0 + rows > t->n) fail("row range out of bounds for '" + std::string(module) + "'");
  const std::vector<Prepared> ps = prepare(stack, module, include_side_branch);
  for (const Prepared& p : ps) apply(stack, p, row0, rows, w, bias);
  return static_cast<int>(ps.size());
}

int merge_bf16(const LoraStack& stack, std::string_view module, std::uint16_t* w, std::int64_t n, std::int64_t k,
               std::uint16_t* bias, int threads) {
  target(stack, module, n, k);
  const std::vector<Prepared> ps = prepare(stack, module, false);
  if (ps.empty()) return 0;
  quant::parallel_for(n, threads, [&](std::int64_t begin, std::int64_t end) {
    std::vector<float> buf(static_cast<std::size_t>(std::min(kRowBlock, end - begin) * k));
    for (std::int64_t r0 = begin; r0 < end; r0 += kRowBlock) {
      const std::int64_t rows = std::min(kRowBlock, end - r0);
      std::uint16_t* src = w + r0 * k;
      for (std::int64_t i = 0; i < rows * k; ++i) buf[static_cast<std::size_t>(i)] = quant::bf16_to_float(src[i]);
      for (const Prepared& p : ps) apply(stack, p, r0, rows, buf.data(), nullptr);
      for (std::int64_t i = 0; i < rows * k; ++i) {
        const float v = buf[static_cast<std::size_t>(i)];
        if (!std::isfinite(v)) fail("merged weight of '" + std::string(module) + "' is not finite");
        src[i] = quant::float_to_bf16(v);
      }
    }
  });
  if (bias != nullptr) {
    std::vector<float> b(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = quant::bf16_to_float(bias[i]);
    for (const Prepared& p : ps) {
      if (p.bias.empty()) continue;
      for (std::int64_t i = 0; i < n; ++i) b[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i)] + p.coef * p.bias[static_cast<std::size_t>(i)];
    }
    for (std::int64_t i = 0; i < n; ++i) {
      if (!std::isfinite(b[static_cast<std::size_t>(i)])) fail("merged bias of '" + std::string(module) + "' is not finite");
      bias[i] = quant::float_to_bf16(b[static_cast<std::size_t>(i)]);
    }
  } else {
    for (const Prepared& p : ps) {
      if (!p.bias.empty()) fail("'" + std::string(module) + "' has a bias delta but no bias was passed");
    }
  }
  return static_cast<int>(ps.size());
}

int merge_requantize_int8(const LoraStack& stack, std::string_view module, std::int8_t* q, float* scale,
                          std::int64_t n, std::int64_t k, int group, int threads) {
  target(stack, module, n, k);
  if (group != 0 && (!quant::valid_convrot_group(group) || k % group != 0)) fail("invalid ConvRot group " + std::to_string(group));
  const std::vector<Prepared> ps = prepare(stack, module, false);
  if (ps.empty()) return 0;
  for (const Prepared& p : ps) {
    if (!p.bias.empty()) fail("bias deltas on INT8 linears are not supported ('" + std::string(module) + "')");
  }
  quant::parallel_for(n, threads, [&](std::int64_t begin, std::int64_t end) {
    std::vector<float> buf(static_cast<std::size_t>(std::min(kRowBlock, end - begin) * k));
    for (std::int64_t r0 = begin; r0 < end; r0 += kRowBlock) {
      const std::int64_t rows = std::min(kRowBlock, end - r0);
      quant::dequantize_rows(q + r0 * k, scale + r0, rows, k, group, buf.data());
      for (const Prepared& p : ps) apply(stack, p, r0, rows, buf.data(), nullptr);
      bool ok = true;
      if (group != 0) {
        ok = quant::quantize_convrot_rows(buf.data(), rows, k, group, q + r0 * k, scale + r0);
      } else {
        for (std::int64_t i = 0; i < rows * k && ok; ++i) ok = std::isfinite(buf[static_cast<std::size_t>(i)]);
        if (ok) quant::quantize_rows_int8(buf.data(), rows, k, q + r0 * k, scale + r0);
      }
      if (!ok) fail("merged weight of '" + std::string(module) + "' is not finite");
    }
  });
  return static_cast<int>(ps.size());
}

void PristineWeights::keep_bf16(const std::string& module, const std::uint16_t* w, std::int64_t n, std::int64_t k,
                                const std::uint16_t* bias) {
  if (has(module)) fail("pristine weights of '" + module + "' already kept");
  if (w == nullptr || n <= 0 || k <= 0) fail("keep_bf16: empty weight for '" + module + "'");
  Entry e;
  e.n = n;
  e.k = k;
  e.w.assign(w, w + n * k);
  if (bias != nullptr) e.bias.assign(bias, bias + n);
  entries_.emplace_back(module, std::move(e));
}

void PristineWeights::keep_int8(const std::string& module, const std::int8_t* q, const float* scale, std::int64_t n,
                                std::int64_t k, int group) {
  if (has(module)) fail("pristine weights of '" + module + "' already kept");
  if (q == nullptr || scale == nullptr || n <= 0 || k <= 0) fail("keep_int8: empty weight for '" + module + "'");
  Entry e;
  e.n = n;
  e.k = k;
  e.group = group;
  e.int8 = true;
  e.q.assign(q, q + n * k);
  e.scale.assign(scale, scale + n);
  entries_.emplace_back(module, std::move(e));
}

bool PristineWeights::has(std::string_view module) const {
  return std::any_of(entries_.begin(), entries_.end(), [&](const auto& e) { return e.first == module; });
}

std::size_t PristineWeights::host_bytes() const {
  std::size_t b = 0;
  for (const auto& [m, e] : entries_) b += e.w.size() * 2 + e.bias.size() * 2 + e.q.size() + e.scale.size() * 4;
  return b;
}

const PristineWeights::Entry& PristineWeights::entry(std::string_view module, bool int8) const {
  for (const auto& [m, e] : entries_) {
    if (m != module) continue;
    if (e.int8 != int8) fail("pristine weights of '" + std::string(module) + "' were kept as " + (e.int8 ? "INT8" : "bf16"));
    return e;
  }
  fail("no pristine weights kept for '" + std::string(module) + "'");
}

int PristineWeights::remerge_bf16(const LoraStack& stack, std::string_view module, std::uint16_t* dst,
                                  std::uint16_t* dst_bias, int threads) const {
  const Entry& e = entry(module, false);
  if (dst == nullptr || (dst_bias != nullptr) != !e.bias.empty()) fail("remerge_bf16: destination does not match '" + std::string(module) + "'");
  std::copy(e.w.begin(), e.w.end(), dst);
  if (dst_bias != nullptr) std::copy(e.bias.begin(), e.bias.end(), dst_bias);
  return merge_bf16(stack, module, dst, e.n, e.k, dst_bias, threads);
}

int PristineWeights::remerge_int8(const LoraStack& stack, std::string_view module, std::int8_t* q, float* scale,
                                  int threads) const {
  const Entry& e = entry(module, true);
  if (q == nullptr || scale == nullptr) fail("remerge_int8: null destination for '" + std::string(module) + "'");
  std::copy(e.q.begin(), e.q.end(), q);
  std::copy(e.scale.begin(), e.scale.end(), scale);
  return merge_requantize_int8(stack, module, q, scale, e.n, e.k, e.group, threads);
}

}  // namespace mi::lora
