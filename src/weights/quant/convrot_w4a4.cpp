// ConvRot W4A4 weight quantization (see convrot_w4a4.h).
#include "weights/quant/convrot_w4a4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "weights/quant/quant_codec.h"

namespace mi::quant {
namespace {

constexpr int kRotGroup = 256;
constexpr float kInv7 = 1.0f / 7.0f;  // torch: a bf16 tensor / CPU scalar 7 multiplies by the fp32 reciprocal

[[noreturn]] void fail(const std::string& msg) { throw std::invalid_argument("mi::quant::convrot_w4a4: " + msg); }

float rbf(float v) { return bf16_to_float(float_to_bf16(v)); }

float absmax_scale(float amax) { return rbf(rbf(std::max(amax, 1e-10f)) * kInv7); }

int quant_value(float x, float s) { return static_cast<int>(std::clamp(std::nearbyint(rbf(x / s)), -7.0f, 7.0f)); }

double row_error(const float* v, std::int64_t n, float s) {
  double err = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(v[i]) - static_cast<double>(s) * quant_value(v[i], s);
    err += d * d;
  }
  return err;
}

float choose_scale(const float* v, std::int64_t n, W4ScaleSearch search) {
  float amax = 0.0f;
  for (std::int64_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(v[i]));
  const float s_abs = absmax_scale(amax);
  if (search == W4ScaleSearch::kAbsmax || amax == 0.0f) return s_abs;
  float best = s_abs;
  double best_err = -1.0;
  for (int c = 100; c >= 40; --c) {  // c = 100 is the absmax scale (amax is a bf16 value); ties keep the larger ratio
    const float s = std::max(absmax_scale(rbf(amax * (static_cast<float>(c) / 100.0f))), 1e-30f);
    const double err = row_error(v, n, s);
    if (best_err < 0.0 || err < best_err) {
      best_err = err;
      best = s;
    }
  }
  return best;
}

void pack_row(const int* qv, std::int64_t k, std::uint8_t* out) {
  for (std::int64_t c = 0; c < k; c += 2) out[c / 2] = static_cast<std::uint8_t>((qv[c] & 0xF) | ((qv[c + 1] & 0xF) << 4));
}

// In-place x · H256 per 256-entry group of a double row (radix-4 stages; H256 is symmetric and its own inverse).
void rotate_row(double* x, std::int64_t k) {
  for (std::int64_t g = 0; g < k; g += kRotGroup) {
    double* v = x + g;
    for (int st = 1; st < kRotGroup; st *= 4) {
      for (int base = 0; base < kRotGroup; ++base) {
        if ((base / st) % 4 != 0) continue;
        const double x0 = v[base], x1 = v[base + st], x2 = v[base + 2 * st], x3 = v[base + 3 * st];
        v[base] = 0.5 * (x0 + x1 + x2 - x3);
        v[base + st] = 0.5 * (x0 + x1 - x2 + x3);
        v[base + 2 * st] = 0.5 * (x0 - x1 + x2 + x3);
        v[base + 3 * st] = 0.5 * (-x0 + x1 + x2 + x3);
      }
    }
  }
}

void transpose_in_place(std::vector<double>& a, std::int64_t n) {
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = i + 1; j < n; ++j) std::swap(a[static_cast<std::size_t>(i * n + j)], a[static_cast<std::size_t>(j * n + i)]);
  }
}

// In-place lower Cholesky factor of a symmetric positive-definite [n, n] matrix (lower triangle written).
void cholesky_lower(std::vector<double>& a, std::int64_t n, int threads) {
  for (std::int64_t j = 0; j < n; ++j) {
    double d = a[static_cast<std::size_t>(j * n + j)];
    for (std::int64_t p = 0; p < j; ++p) d -= a[static_cast<std::size_t>(j * n + p)] * a[static_cast<std::size_t>(j * n + p)];
    if (!(d > 0.0)) fail("Hessian is not positive definite (increase gptq_damp)");
    const double ljj = std::sqrt(d);
    a[static_cast<std::size_t>(j * n + j)] = ljj;
    const std::int64_t rows = n - j - 1;
    const auto body = [&](std::int64_t b, std::int64_t e) {
      const double* aj = a.data() + j * n;
      for (std::int64_t i = j + 1 + b; i < j + 1 + e; ++i) {
        double* ai = a.data() + i * n;
        double v = ai[j];
        for (std::int64_t p = 0; p < j; ++p) v -= ai[p] * aj[p];
        ai[j] = v / ljj;
      }
    };
    if (rows * j > (1 << 18)) {
      parallel_for(rows, threads, body);
    } else {
      body(0, rows);
    }
  }
}

// Upper factor U of H^-1 = U^T U (GPTQ's torch.linalg.cholesky(cholesky_inverse(chol(H)), upper=True)), in place:
// h holds H on entry and U on return. One extra n x n buffer.
void inverse_cholesky_upper(std::vector<double>& h, std::int64_t n, int threads) {
  cholesky_lower(h, n, threads);  // h = L
  // lt row c = column c of L^-1 (lower), solved independently per column.
  std::vector<double> lt(static_cast<std::size_t>(n * n), 0.0);
  parallel_for(n, threads, [&](std::int64_t b, std::int64_t e) {
    for (std::int64_t c = b; c < e; ++c) {
      double* col = lt.data() + c * n;
      col[c] = 1.0 / h[static_cast<std::size_t>(c * n + c)];
      for (std::int64_t i = c + 1; i < n; ++i) {
        const double* hi = h.data() + i * n;
        double v = 0.0;
        for (std::int64_t p = c; p < i; ++p) v -= hi[p] * col[p];
        col[i] = v / hi[i];
      }
    }
  });
  // H^-1 = L^-T L^-1 into h (L is no longer needed): hinv[i][j] = sum_{p >= max(i, j)} lt[i][p] lt[j][p].
  parallel_for(n, threads, [&](std::int64_t b, std::int64_t e) {
    for (std::int64_t i = b; i < e; ++i) {
      const double* li = lt.data() + i * n;
      for (std::int64_t j = 0; j <= i; ++j) {
        const double* lj = lt.data() + j * n;
        double v = 0.0;
        for (std::int64_t p = i; p < n; ++p) v += li[p] * lj[p];
        h[static_cast<std::size_t>(i * n + j)] = v;  // lower triangle only: the Cholesky below reads only it
      }
    }
  });
  lt = {};
  cholesky_lower(h, n, threads);  // h = lower factor of H^-1
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = i + 1; j < n; ++j) h[static_cast<std::size_t>(i * n + j)] = 0.0;
  }
  transpose_in_place(h, n);  // U = L_inv^T
}

}  // namespace

ConvRotW4A4Weight quantize_convrot_w4a4(const std::uint16_t* w_bf16, std::int64_t n, std::int64_t k,
                                        const ConvRotW4A4Options& opt) {
  if (n <= 0 || k <= 0 || k % kRotGroup != 0) fail("needs n > 0 and k % 256 == 0");
  for (std::int64_t i = 0; i < n * k; ++i) {
    if ((w_bf16[i] & 0x7F80) == 0x7F80) fail("non-finite weight at index " + std::to_string(i));
  }
  std::vector<double> u;  // GPTQ: upper Cholesky factor of H_rot^-1
  if (opt.hessian != nullptr) {
    u.assign(opt.hessian, opt.hessian + k * k);
    for (int pass = 0; pass < 2; ++pass) {  // H_rot = H256 · H · H256: rotate the rows, transpose, twice
      parallel_for(k, opt.threads, [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t r = b; r < e; ++r) rotate_row(u.data() + r * k, k);
      });
      transpose_in_place(u, k);
    }
    double mean_diag = 0.0;
    for (std::int64_t i = 0; i < k; ++i) mean_diag += u[static_cast<std::size_t>(i * k + i)];
    mean_diag /= static_cast<double>(k);
    for (std::int64_t i = 0; i < k; ++i) u[static_cast<std::size_t>(i * k + i)] += opt.gptq_damp * mean_diag;
    inverse_cholesky_upper(u, k, opt.threads);
  }
  ConvRotW4A4Weight out;
  out.q.resize(static_cast<std::size_t>(n * k / 2));
  out.scale.resize(static_cast<std::size_t>(n));
  parallel_for(n, opt.threads, [&](std::int64_t b, std::int64_t e) {
    std::vector<float> row(static_cast<std::size_t>(k)), rot(static_cast<std::size_t>(k));
    std::vector<double> work(static_cast<std::size_t>(k));
    std::vector<int> qv(static_cast<std::size_t>(k));
    for (std::int64_t r = b; r < e; ++r) {
      for (std::int64_t c = 0; c < k; ++c) row[static_cast<std::size_t>(c)] = bf16_to_float(w_bf16[r * k + c]);
      convrot_rows(row.data(), 1, k, kRotGroup, rot.data());
      for (float& v : rot) v = rbf(v);  // eager rotates the bf16 weight in bf16
      const float s = choose_scale(rot.data(), k, opt.search);
      out.scale[static_cast<std::size_t>(r)] = s;
      if (u.empty()) {
        for (std::int64_t c = 0; c < k; ++c) qv[static_cast<std::size_t>(c)] = quant_value(rot[static_cast<std::size_t>(c)], s);
      } else {
        for (std::int64_t c = 0; c < k; ++c) work[static_cast<std::size_t>(c)] = rot[static_cast<std::size_t>(c)];
        for (std::int64_t c = 0; c < k; ++c) {
          const int q = quant_value(static_cast<float>(work[static_cast<std::size_t>(c)]), s);
          qv[static_cast<std::size_t>(c)] = q;
          const double* uc = u.data() + c * k;
          const double err = (work[static_cast<std::size_t>(c)] - static_cast<double>(s) * q) / uc[c];
          for (std::int64_t j = c + 1; j < k; ++j) work[static_cast<std::size_t>(j)] -= err * uc[j];
        }
      }
      pack_row(qv.data(), k, out.q.data() + r * k / 2);
    }
  });
  return out;
}

std::vector<double> gram_bf16(const std::uint16_t* x, std::int64_t rows, std::int64_t k, std::int64_t ld, int threads) {
  std::vector<double> h(static_cast<std::size_t>(k * k), 0.0);
  std::vector<float> xf(static_cast<std::size_t>(rows * k));
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < k; ++c) xf[static_cast<std::size_t>(r * k + c)] = bf16_to_float(x[r * ld + c]);
  }
  // Each thread owns a band of H rows and sums every input row into it in row order (thread-count independent).
  parallel_for(k, threads, [&](std::int64_t b, std::int64_t e) {
    for (std::int64_t r = 0; r < rows; ++r) {
      const float* xr = xf.data() + r * k;
      for (std::int64_t i = b; i < e; ++i) {
        const double xi = xr[i];
        if (xi == 0.0) continue;
        double* hi = h.data() + i * k;
        for (std::int64_t j = 0; j < k; ++j) hi[j] += xi * xr[j];
      }
    }
  });
  return h;
}

}  // namespace mi::quant
