#include "weights/quant/nvfp4_recipes.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <string>
#include <thread>

#include "weights/quant/nvfp4_codec.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_model.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#define MI_NVFP4_MXCSR 1
#endif

namespace mi::quant {
namespace {

[[noreturn]] void fail(const std::string& msg) { throw QuantError("mi::quant: " + msg); }

// Subnormals are computed, never flushed, whatever the thread's floating-point environment (a library built with
// fast-math may have set flush-to-zero / denormals-are-zero for the process): cleared for the guard's lifetime on this
// thread and restored after. x86: MXCSR FTZ (bit 15) and DAZ (bit 6); aarch64 (GCC / Clang): FPCR.FZ (bit 24).
class StrictSubnormals {
 public:
  StrictSubnormals() {
#if defined(MI_NVFP4_MXCSR)
    saved_ = _mm_getcsr();
    _mm_setcsr(saved_ & ~0x8040u);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved_));
    const std::uint64_t cleared = saved_ & ~(std::uint64_t{1} << 24);
    __asm__ volatile("msr fpcr, %0" : : "r"(cleared));
#endif
  }
  ~StrictSubnormals() {
#if defined(MI_NVFP4_MXCSR)
    _mm_setcsr(saved_);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ volatile("msr fpcr, %0" : : "r"(saved_));
#endif
  }
  StrictSubnormals(const StrictSubnormals&) = delete;
  StrictSubnormals& operator=(const StrictSubnormals&) = delete;

 private:
#if defined(MI_NVFP4_MXCSR)
  unsigned int saved_ = 0;
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
  std::uint64_t saved_ = 0;
#endif
};

int resolve_threads(int threads) {
  return threads > 0 ? threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

// f(t) for t in [0, tiles), handed out one tile at a time to up to `threads` workers (tiles of uneven cost balance
// themselves); which worker runs a tile never changes its result.
template <typename F>
void parallel_tiles(std::int64_t tiles, int threads, const F& f) {
  if (tiles <= 0) return;
  std::atomic<std::int64_t> next{0};
  parallel_for(std::min<std::int64_t>(resolve_threads(threads), tiles), threads, [&](std::int64_t, std::int64_t) {
    const StrictSubnormals strict;
    for (std::int64_t t = next++; t < tiles; t = next++) f(t);
  });
}

// The 128 non-negative e4m3 codes' values (0x7F, NaN, is never used as a scale here).
const float* e4m3_values() {
  static const auto table = [] {
    std::array<float, 128> t{};
    for (int c = 0; c < 128; ++c) t[static_cast<std::size_t>(c)] = e4m3_to_float(static_cast<std::uint8_t>(c));
    return t;
  }();
  return table.data();
}

// The codec's element step for a given total = s2 * sf: +0 where total == 0.
inline std::uint8_t element_code(float x, float total) {
  return total == 0.0f ? 0 : e2m1_rne(std::clamp(x / total, -kNvfp4E2M1Max, kNvfp4E2M1Max));
}

// One element's squared error under `total`, as the block error sums it.
inline double element_err(float x, float total) {
  const float d = x - e2m1_to_float(element_code(x, total)) * total;
  return static_cast<double>(d) * static_cast<double>(d);
}

double block_err(const float* x, float total) {
  double e = 0.0;
  for (int i = 0; i < kNvfp4Block; ++i) e = e + element_err(x[i], total);
  return e;
}

std::uint8_t rtn_block_scale(float amax, float s2) { return e4m3_rne(std::min((amax / kNvfp4E2M1Max) / s2, kNvfp4E4M3Max)); }

// Codes of one block under scale code `sf`, packed hi_first.
void pack_block(const float* x, float total, std::uint8_t* packed) {
  for (int i = 0; i < kNvfp4Block; i += 2) {
    packed[i / 2] = static_cast<std::uint8_t>(element_code(x[i], total) << 4 | element_code(x[i + 1], total));
  }
}

// Blocked forms of the two triangular steps (see the header): panels of kPanel rows, and in each panel's trailing
// update the rows are independent tiles of work.
constexpr std::int64_t kPanel = 64;
constexpr std::int64_t kColChunk = 512;  // columns per tile in the panel steps
constexpr std::int64_t kTileRows = 8;    // trailing rows updated together
constexpr std::int64_t kTileCols = 256;  // columns per pass of a trailing tile (8 rows + 1 panel row fit L1)

// In place: the upper triangle of a (k x k, row-major) becomes R with a = R^T R.
void cholesky_upper(double* a, std::int64_t k, int threads) {
  for (std::int64_t k0 = 0; k0 < k; k0 += kPanel) {
    const std::int64_t k1 = std::min(k0 + kPanel, k);
    // Diagonal block.
    for (std::int64_t p = k0; p < k1; ++p) {
      double* rp = a + p * k;
      const double d = rp[p];
      if (!(d > 0.0) || !std::isfinite(d)) fail("GPTQ: the damped Hessian is not positive definite (pivot " + std::to_string(p) + ")");
      const double r = std::sqrt(d);
      rp[p] = r;
      for (std::int64_t j = p + 1; j < k1; ++j) rp[j] = rp[j] / r;
      for (std::int64_t i = p + 1; i < k1; ++i) {
        const double s = rp[i];
        double* ri = a + i * k;
        for (std::int64_t j = i; j < k1; ++j) ri[j] = ri[j] - s * rp[j];
      }
    }
    // Panel rows right of the diagonal block, by column chunks.
    const std::int64_t width = k - k1;
    parallel_tiles((width + kColChunk - 1) / kColChunk, threads, [&](std::int64_t t) {
      const std::int64_t j0 = k1 + t * kColChunk, j1 = std::min(j0 + kColChunk, k);
      for (std::int64_t p = k0; p < k1; ++p) {
        double* rp = a + p * k;
        const double r = rp[p];
        for (std::int64_t j = j0; j < j1; ++j) rp[j] = rp[j] / r;
        for (std::int64_t i = p + 1; i < k1; ++i) {
          const double s = rp[i];
          double* ri = a + i * k;
          for (std::int64_t j = j0; j < j1; ++j) ri[j] = ri[j] - s * rp[j];
        }
      }
    });
    // Trailing rows, kTileRows at a time over column chunks (each panel row's chunk is read once per tile).
    parallel_tiles((width + kTileRows - 1) / kTileRows, threads, [&](std::int64_t t) {
      const std::int64_t i0 = k1 + t * kTileRows, i1 = std::min(i0 + kTileRows, k);
      for (std::int64_t j0 = i0; j0 < k; j0 += kTileCols) {
        const std::int64_t j1 = std::min(j0 + kTileCols, k);
        for (std::int64_t p = k0; p < k1; ++p) {
          const double* rp = a + p * k;
          for (std::int64_t i = i0; i < i1; ++i) {
            const double s = rp[i];
            double* ri = a + i * k;
            for (std::int64_t j = std::max(j0, i); j < j1; ++j) ri[j] = ri[j] - s * rp[j];
          }
        }
      }
    });
  }
}

// R in the strict upper triangle of a, its diagonal in rd: the lower triangle (incl. the diagonal) of a becomes R^-T.
void inverse_transpose_lower(double* a, const double* rd, std::int64_t k, int threads) {
  for (std::int64_t i = 0; i < k; ++i) {
    double* ri = a + i * k;
    std::fill(ri, ri + i, 0.0);
    ri[i] = 1.0;
  }
  for (std::int64_t k0 = 0; k0 < k; k0 += kPanel) {
    const std::int64_t k1 = std::min(k0 + kPanel, k);
    // Panel rows, by column chunks of [0, k1).
    parallel_tiles((k1 + kColChunk - 1) / kColChunk, threads, [&](std::int64_t t) {
      const std::int64_t c0 = t * kColChunk, c1 = std::min(c0 + kColChunk, k1);
      for (std::int64_t p = k0; p < k1; ++p) {
        double* mp = a + p * k;
        const std::int64_t pe = std::min(c1, p + 1);
        for (std::int64_t c = c0; c < pe; ++c) mp[c] = mp[c] / rd[p];
        for (std::int64_t i = p + 1; i < k1; ++i) {
          const double s = mp[i];  // R[p][i]: strict upper triangle
          double* bi = a + i * k;
          for (std::int64_t c = c0; c < pe; ++c) bi[c] = bi[c] - s * mp[c];
        }
      }
    });
    // Trailing rows, tiled as in cholesky_upper.
    parallel_tiles((k - k1 + kTileRows - 1) / kTileRows, threads, [&](std::int64_t t) {
      const std::int64_t i0 = k1 + t * kTileRows, i1 = std::min(i0 + kTileRows, k);
      for (std::int64_t c0 = 0; c0 < k1; c0 += kTileCols) {
        const std::int64_t c1 = std::min(c0 + kTileCols, k1);
        for (std::int64_t p = std::max(k0, c0); p < k1; ++p) {
          const double* mp = a + p * k;
          const std::int64_t pe = std::min(c1, p + 1);
          for (std::int64_t i = i0; i < i1; ++i) {
            const double s = mp[i];
            double* bi = a + i * k;
            for (std::int64_t c = c0; c < pe; ++c) bi[c] = bi[c] - s * mp[c];
          }
        }
      }
    });
  }
}

constexpr std::int64_t kGptqRows = 32;       // rows per tile
constexpr std::int64_t kGptqBlock = 128;     // columns whose updates beyond the block are applied together
constexpr std::int64_t kGptqChunk = 512;     // columns per pass of those deferred updates

}  // namespace

const char* recipe_name(Nvfp4Recipe r) {
  switch (r) {
    case Nvfp4Recipe::kRtn: return "rtn";
    case Nvfp4Recipe::kMse: return "mse";
    case Nvfp4Recipe::kGptq: return "gptq";
    case Nvfp4Recipe::kGptqMse: return "gptq-mse";
  }
  return "?";
}

std::optional<Nvfp4Recipe> parse_nvfp4_recipe(std::string_view s) {
  for (Nvfp4Recipe r : {Nvfp4Recipe::kRtn, Nvfp4Recipe::kMse, Nvfp4Recipe::kGptq, Nvfp4Recipe::kGptqMse}) {
    if (s == recipe_name(r)) return r;
  }
  return std::nullopt;
}

float nvfp4_weight_tensor_scale(const float* w, std::int64_t n, int threads) {
  constexpr std::int64_t kChunk = std::int64_t{1} << 20;
  const std::int64_t chunks = (n + kChunk - 1) / kChunk;
  std::vector<float> part(static_cast<std::size_t>(std::max<std::int64_t>(chunks, 1)), 0.0f);
  parallel_tiles(chunks, threads, [&](std::int64_t c) {
    float m = 0.0f;
    for (std::int64_t i = c * kChunk; i < std::min(n, (c + 1) * kChunk); ++i) m = std::max(m, std::fabs(w[i]));
    part[static_cast<std::size_t>(c)] = m;
  });
  const float amax = *std::max_element(part.begin(), part.end());  // a maximum: the same for any split
  return amax == 0.0f ? 1.0f : nvfp4_tensor_scale(amax);
}

std::uint8_t nvfp4_mse_block_scale(const float* x, float s2) {
  float amax = 0.0f;
  int ia = 0;
  for (int i = 0; i < kNvfp4Block; ++i) {
    if (std::fabs(x[i]) > amax) {
      amax = std::fabs(x[i]);
      ia = i;
    }
  }
  const float* e4m3 = e4m3_values();
  std::uint8_t best = rtn_block_scale(amax, s2);
  double best_err = block_err(x, s2 * e4m3[best]);
  for (int c = 1; c <= 0x7E; ++c) {
    const float total = s2 * e4m3[c];
    // The block error is a sum of non-negative fp64 terms, so it is at least any one of them (rounding is monotone):
    // a candidate whose largest element alone errs by best_err or more cannot win. Skipping it changes nothing.
    if (element_err(x[ia], total) >= best_err) continue;
    const double e = block_err(x, total);
    if (e < best_err) {
      best = static_cast<std::uint8_t>(c);
      best_err = e;
    }
  }
  return best;
}

void quantize_nvfp4_rows_recipe(const float* w, std::int64_t rows, std::int64_t k, float s2, bool mse, std::uint8_t* packed,
                                std::uint8_t* scales) {
  const StrictSubnormals strict;
  if (!mse) {
    quantize_nvfp4_rows(w, rows, k, &s2, 0, packed, scales);
    return;
  }
  if (k <= 0 || k % kNvfp4Block != 0) fail("NVFP4 K=" + std::to_string(k) + " is not a positive multiple of 16");
  const std::int64_t sfk = k / kNvfp4Block;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t b = 0; b < sfk; ++b) {
      const float* x = w + r * k + b * kNvfp4Block;
      const std::uint8_t sf = nvfp4_mse_block_scale(x, s2);
      scales[r * sfk + b] = sf;
      pack_block(x, s2 * e4m3_values()[sf], packed + r * (k / 2) + b * (kNvfp4Block / 2));
    }
  }
}

GptqFactor gptq_factor(const float* h, std::int64_t k, int threads) {
  if (k <= 0 || k % kNvfp4Block != 0) fail("GPTQ: K=" + std::to_string(k) + " is not a positive multiple of 16");
  const StrictSubnormals strict;
  GptqFactor f;
  f.k = k;
  f.u.resize(static_cast<std::size_t>(k * k));
  f.dead.assign(static_cast<std::size_t>(k), 0);
  double* a = f.u.data();
  // Hd, mirrored from the upper triangle; dead columns; damping.
  for (std::int64_t i = 0; i < k; ++i) {
    for (std::int64_t j = i; j < k; ++j) {
      const float v = h[i * k + j];
      if (!std::isfinite(v)) fail("GPTQ: the Hessian holds a non-finite value");
      a[i * k + j] = a[j * k + i] = v;
    }
  }
  double diag_sum = 0.0;
  for (std::int64_t j = 0; j < k; ++j) {
    if (a[j * k + j] == 0.0) {
      f.dead[static_cast<std::size_t>(j)] = 1;
      a[j * k + j] = 1.0;
    }
    diag_sum = diag_sum + a[j * k + j];
  }
  const double damp = 0.01 * (diag_sum / static_cast<double>(k));
  for (std::int64_t j = 0; j < k; ++j) a[j * k + j] = a[j * k + j] + damp;
  // J Hd J is Hd with its row-major storage reversed; likewise U = J M J for M = R^-T.
  std::reverse(f.u.begin(), f.u.end());
  cholesky_upper(a, k, threads);
  std::vector<double> rd(static_cast<std::size_t>(k));
  for (std::int64_t i = 0; i < k; ++i) rd[static_cast<std::size_t>(i)] = a[i * k + i];
  inverse_transpose_lower(a, rd.data(), k, threads);
  std::reverse(f.u.begin(), f.u.end());
  return f;
}

void gptq_quantize_rows(const float* w, std::int64_t rows, std::int64_t k, float s2, const GptqFactor& f, bool mse_scales,
                        int threads, std::uint8_t* packed, std::uint8_t* scales) {
  if (f.k != k) fail("GPTQ: factor of K=" + std::to_string(f.k) + " for a weight of K=" + std::to_string(k));
  const std::int64_t sfk = k / kNvfp4Block;
  const double* u = f.u.data();
  parallel_tiles((rows + kGptqRows - 1) / kGptqRows, threads, [&](std::int64_t t) {
    const std::int64_t r0 = t * kGptqRows, nr = std::min(kGptqRows, rows - r0);
    std::vector<double> wt(static_cast<std::size_t>(nr * k));
    std::vector<double> err(static_cast<std::size_t>(nr * kGptqBlock));
    std::vector<float> total(static_cast<std::size_t>(nr));
    for (std::int64_t r = 0; r < nr; ++r) {
      for (std::int64_t j = 0; j < k; ++j) wt[static_cast<std::size_t>(r * k + j)] = f.dead[static_cast<std::size_t>(j)] ? 0.0 : w[(r0 + r) * k + j];
    }
    for (std::int64_t c0 = 0; c0 < k; c0 += kGptqBlock) {
      const std::int64_t c1 = std::min(c0 + kGptqBlock, k);
      for (std::int64_t col = c0; col < c1; ++col) {
        const double* ucol = u + col * k;
        for (std::int64_t r = 0; r < nr; ++r) {
          double* wr = wt.data() + r * k;
          std::uint8_t* pr = packed + (r0 + r) * (k / 2);
          if (col % kNvfp4Block == 0) {
            float x[kNvfp4Block];
            float amax = 0.0f;
            for (int i = 0; i < kNvfp4Block; ++i) {
              x[i] = static_cast<float>(wr[col + i]);
              amax = std::max(amax, std::fabs(x[i]));
            }
            const std::uint8_t sf = mse_scales ? nvfp4_mse_block_scale(x, s2) : rtn_block_scale(amax, s2);
            scales[(r0 + r) * sfk + col / kNvfp4Block] = sf;
            total[static_cast<std::size_t>(r)] = s2 * e4m3_values()[sf];
          }
          const float tr = total[static_cast<std::size_t>(r)];
          const std::uint8_t q = element_code(static_cast<float>(wr[col]), tr);
          if (col % 2 == 0) pr[col / 2] = static_cast<std::uint8_t>(q << 4);
          else pr[col / 2] = static_cast<std::uint8_t>(pr[col / 2] | q);
          const double deq = static_cast<double>(e2m1_to_float(q)) * static_cast<double>(tr);
          const double e = (wr[col] - deq) / ucol[col];
          err[static_cast<std::size_t>(r * kGptqBlock + (col - c0))] = e;
          for (std::int64_t j = col + 1; j < c1; ++j) wr[j] = wr[j] - e * ucol[j];
        }
      }
      // The block's updates of the columns after it, still in ascending column order per element.
      for (std::int64_t j0 = c1; j0 < k; j0 += kGptqChunk) {
        const std::int64_t j1 = std::min(j0 + kGptqChunk, k);
        for (std::int64_t r = 0; r < nr; ++r) {
          double* wr = wt.data() + r * k;
          for (std::int64_t col = c0; col < c1; ++col) {
            const double e = err[static_cast<std::size_t>(r * kGptqBlock + (col - c0))];
            const double* ucol = u + col * k;
            for (std::int64_t j = j0; j < j1; ++j) wr[j] = wr[j] - e * ucol[j];
          }
        }
      }
    }
  });
}

}  // namespace mi::quant
