// sm_75 bring-up harness: runs the Turing kernel family on the current device and checks it against a host
// reference. Built only with MOSSINFER_SM75 (cmake/modules/sm75.cmake), as a development tool, not installed shipped.
//
// Usage: sm75_check [attn128|attn256|convert|all]
//   attn128  varlen non-causal attention, head_dim 128 (the DiT's joint attention shape), GQA 2:1
//   attn256  varlen causal attention, head_dim 256 (the text encoder's full-attention shape), GQA 4:1, and causal
//            with key lengths unlike the query lengths (bottom-right alignment)
//   gemm     the fp16 linear: bf16 and f32 out, a row above fp16's range, several row chunks
//   convert  bf16 <-> fp16 round trip on device
// The engine's own GPU tests compile their reference kernels for the build's CUDA architectures (sm_80+), so the
// unit-test binary cannot exercise sm_75 kernels on a Turing device; this tool builds its own reference in host code.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend/cuda_libs.h"
#include "backend/probe.h"
#include "core/cuda_check.h"
#include "core/tensor.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/gemm.h"
#include "kernels/quant/quant.h"
#include "kernels/sm75/sm75.h"

namespace {

using mi::DType;
using mi::Device;
using mi::Tensor;

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float f = 0;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

std::uint16_t float_to_bf16_bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof u);
  const std::uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(rounded >> 16);
}

std::uint32_t hash_u32(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return static_cast<std::uint32_t>(x);
}

std::vector<float> normal(std::size_t n, std::uint64_t seed, float scale) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t key = seed * 0x9E3779B97F4A7C15ull + i * 2;
    const float u1 = (static_cast<float>(hash_u32(key) >> 8) + 1.f) * (1.f / 16777217.f);
    const float u2 = static_cast<float>(hash_u32(key + 1) >> 8) * (1.f / 16777216.f);
    v[i] = scale * std::sqrt(-2.f * std::log(u1)) * std::cos(6.283185307179586f * u2);
  }
  return v;
}

class DevBuf {
 public:
  explicit DevBuf(std::size_t bytes) { MI_CUDA_CHECK(cudaMalloc(&p_, bytes == 0 ? 1 : bytes)); }
  ~DevBuf() { cudaFree(p_); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  void* get() const { return p_; }
  template <typename T>
  void upload(const std::vector<T>& v) {
    MI_CUDA_CHECK(cudaMemcpy(p_, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  template <typename T>
  std::vector<T> download(std::size_t n) const {
    std::vector<T> v(n);
    MI_CUDA_CHECK(cudaMemcpy(v.data(), p_, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }

 private:
  void* p_ = nullptr;
};

// Host reference: one (row, head) at a time, fp32 scores/softmax/PV, bf16 inputs as the device sees them. Causal is
// aligned bottom-right (FlashAttention-2): query i of a sequence sees keys 0 .. i + len_k - len_q; none: a zero row.
std::vector<float> reference_attention(const std::vector<std::uint16_t>& q, const std::vector<std::uint16_t>& k,
                                       const std::vector<std::uint16_t>& v, const std::vector<int>& cu_q,
                                       const std::vector<int>& cu_k, int heads, int kv_heads, int dim, bool causal,
                                       float scale) {
  const int batch = static_cast<int>(cu_q.size()) - 1;
  const int total_q = cu_q.back();
  std::vector<float> out(static_cast<std::size_t>(total_q) * heads * dim, 0.f);
  const int group = heads / kv_heads;
  std::vector<float> scores;
  for (int b = 0; b < batch; ++b) {
    const int q0 = cu_q[b], q1 = cu_q[b + 1];
    const int k0 = cu_k[b], len_k = cu_k[b + 1] - k0;
    for (int h = 0; h < heads; ++h) {
      const int kh = h / group;
      for (int i = q0; i < q1; ++i) {
        const int n = causal ? std::clamp(i - q0 + 1 + len_k - (q1 - q0), 0, len_k) : len_k;
        if (n == 0) continue;
        scores.assign(static_cast<std::size_t>(n), 0.f);
        float max_s = -INFINITY;
        for (int j = 0; j < n; ++j) {
          const int row = k0 + j;
          float s = 0.f;
          for (int d = 0; d < dim; ++d) {
            s += bf16_to_float(q[(static_cast<std::size_t>(i) * heads + h) * dim + d]) *
                 bf16_to_float(k[(static_cast<std::size_t>(row) * kv_heads + kh) * dim + d]);
          }
          s *= scale;
          scores[static_cast<std::size_t>(j)] = s;
          max_s = std::max(max_s, s);
        }
        float sum = 0.f;
        for (int j = 0; j < n; ++j) {
          scores[static_cast<std::size_t>(j)] = std::exp(scores[static_cast<std::size_t>(j)] - max_s);
          sum += scores[static_cast<std::size_t>(j)];
        }
        std::size_t base = (static_cast<std::size_t>(i) * heads + h) * dim;
        for (int j = 0; j < n; ++j) {
          const float p = scores[static_cast<std::size_t>(j)] / sum;
          const int row = k0 + j;
          for (int d = 0; d < dim; ++d) {
            out[base + d] += p * bf16_to_float(v[(static_cast<std::size_t>(row) * kv_heads + kh) * dim + d]);
          }
        }
      }
    }
  }
  return out;
}

// k_lengths: the keys' per-sequence lengths (empty: the queries' own).
int run_attention(const char* name, int heads, int kv_heads, int dim, bool causal, const std::vector<int>& lengths,
                  std::uint64_t seed, bool dump = false, const std::vector<int>& k_lengths = {}) {
  const int batch = static_cast<int>(lengths.size());
  const std::vector<int>& kl = k_lengths.empty() ? lengths : k_lengths;
  std::vector<int> cu(static_cast<std::size_t>(batch) + 1, 0), cu_k(cu);
  for (int b = 0; b < batch; ++b) {
    cu[static_cast<std::size_t>(b) + 1] = cu[static_cast<std::size_t>(b)] + lengths[static_cast<std::size_t>(b)];
    cu_k[static_cast<std::size_t>(b) + 1] = cu_k[static_cast<std::size_t>(b)] + kl[static_cast<std::size_t>(b)];
  }
  const int total_q = cu.back();
  const int total_k = cu_k.back();
  const int max_len = *std::max_element(lengths.begin(), lengths.end());
  const float scale = 1.f / std::sqrt(static_cast<float>(dim));

  const auto q = normal(static_cast<std::size_t>(total_q) * heads * dim, seed, 1.f);
  const auto k = normal(static_cast<std::size_t>(total_k) * kv_heads * dim, seed + 1, 1.f);
  const auto v = normal(static_cast<std::size_t>(total_k) * kv_heads * dim, seed + 2, 1.f);
  std::vector<std::uint16_t> qb(q.size()), kb(k.size()), vb(v.size());
  for (std::size_t i = 0; i < q.size(); ++i) qb[i] = float_to_bf16_bits(q[i]);
  for (std::size_t i = 0; i < k.size(); ++i) kb[i] = float_to_bf16_bits(k[i]);
  for (std::size_t i = 0; i < v.size(); ++i) vb[i] = float_to_bf16_bits(v[i]);

  DevBuf dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2), dout(qb.size() * 2), dcu(cu.size() * sizeof(int)),
      dcu_k(cu_k.size() * sizeof(int));
  dq.upload(qb);
  dk.upload(kb);
  dv.upload(vb);
  dcu.upload(cu);
  dcu_k.upload(cu_k);
  MI_CUDA_CHECK(cudaMemset(dout.get(), 0, qb.size() * 2));
  std::vector<float> lse(static_cast<std::size_t>(total_q) * heads, 0.f);
  DevBuf dls(lse.size() * sizeof(float));
  MI_CUDA_CHECK(cudaMemset(dls.get(), 0, lse.size() * sizeof(float)));

  auto t = [](void* p, DType dt, std::initializer_list<std::int64_t> shape) {
    return Tensor::view(p, dt, Device::kCUDA, shape);
  };
  const Tensor tq = t(dq.get(), DType::kBF16, {total_q, heads, dim});
  const Tensor tk = t(dk.get(), DType::kBF16, {total_k, kv_heads, dim});
  const Tensor tv = t(dv.get(), DType::kBF16, {total_k, kv_heads, dim});
  const Tensor to = t(dout.get(), DType::kBF16, {total_q, heads, dim});
  const Tensor tcu = t(dcu.get(), DType::kI32, {batch + 1});
  const Tensor tcu_k = t(dcu_k.get(), DType::kI32, {batch + 1});
  const std::size_t ws_bytes = mi::attention::flash_fwd_workspace_bytes(total_q, heads);
  MI_CUDA_CHECK(cudaGetLastError());
  std::printf("  launch: tokens %d heads %d kv %d dim %d causal %d maxlen %d ws %zu\n", total_q, heads, kv_heads, dim,
              causal ? 1 : 0, max_len, ws_bytes);
  std::fflush(stdout);
  mi::attention::flash_fwd(tq, tk, tv, to, tcu, tcu_k, max_len, causal, scale, dls.get(), ws_bytes, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  // The inputs are read only: q/k/v hold their bf16 bits after the call.
  const bool inputs_kept = dq.download<std::uint16_t>(qb.size()) == qb && dk.download<std::uint16_t>(kb.size()) == kb &&
                           dv.download<std::uint16_t>(vb.size()) == vb;
  std::printf("  kernel done\n");
  std::fflush(stdout);

  const auto got = dout.download<std::uint16_t>(qb.size());
  const auto want = reference_attention(qb, kb, vb, cu, cu_k, heads, kv_heads, dim, causal, scale);
  double dot = 0, ng = 0, nw = 0;
  float max_abs = 0.f;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const float g = bf16_to_float(got[i]);
    dot += static_cast<double>(g) * want[i];
    ng += static_cast<double>(g) * g;
    nw += static_cast<double>(want[i]) * want[i];
    max_abs = std::max(max_abs, std::fabs(g - want[i]));
  }
  const double cosine = dot / std::sqrt(ng * nw);
  std::printf("%-8s tokens %5d keys %5d heads %2d kv %2d dim %3d causal %d: cosine %.9f max|diff| %.3g inputs %s\n",
              name, total_q, total_k, heads, kv_heads, dim, causal ? 1 : 0, cosine, max_abs,
              inputs_kept ? "kept" : "MODIFIED");
  if (dump) {
    for (std::size_t i = 0; i < std::min<std::size_t>(want.size(), 12); ++i) {
      std::printf("   [%zu] got %.6f want %.6f (q %.6f)\n", i, bf16_to_float(got[i]), want[i], q[i]);
    }
    // Diagnostic: is the mismatch a different softmax temperature?
    const float log2e = 1.4426950408889634f;
    for (float f : {1.f, log2e, 1.f / log2e, 2.f, 0.5f, 1.f / std::sqrt(static_cast<float>(dim))}) {
      const auto alt = reference_attention(qb, kb, vb, cu, cu_k, heads, kv_heads, dim, causal, scale * f);
      double d2 = 0, g2 = 0, w2 = 0;
      for (std::size_t i = 0; i < alt.size(); ++i) {
        const float g = bf16_to_float(got[i]);
        d2 += static_cast<double>(g) * alt[i];
        g2 += static_cast<double>(g) * g;
        w2 += static_cast<double>(alt[i]) * alt[i];
      }
      std::printf("   scale x %.6f: cosine %.9f\n", f, d2 / std::sqrt(g2 * w2));
    }
  }
  MI_CUDA_CHECK(cudaGetLastError());
  return cosine >= 0.9999 && inputs_kept ? 0 : 1;
}

int run_convert() {
  const std::vector<float> values = {0.f, 1.f, -1.f, 2.5f, -3.14159f, 123.456f, -1024.f, 32768.f, 1e-3f};
  DevBuf d(values.size() * 2);
  std::vector<std::uint16_t> bits(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) bits[i] = float_to_bf16_bits(values[i]);
  d.upload(bits);
  mi::kernels::sm75::convert_bf16_to_f16(d.get(), d.get(), static_cast<std::int64_t>(values.size()), nullptr);
  mi::kernels::sm75::convert_f16_to_bf16(d.get(), d.get(), static_cast<std::int64_t>(values.size()), nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got = d.download<std::uint16_t>(values.size());
  int bad = 0;
  for (std::size_t i = 0; i < values.size(); ++i) bad += got[i] != bits[i];
  std::printf("convert: %zu values, %d mismatches\n", values.size(), bad);
  return bad == 0 ? 0 : 1;
}

// The sm_75 GEMM (cuBLASLt fp16 in / fp32 out, bf16 activation input): bf16 x, fp16 weights, bf16 or f32 out with a
// bias of the output dtype added in fp32. Cross-checked against a host accumulation of the stored operand values.
// big_row >= 0 scales that row of x far above fp16's range (the per-row power-of-two scale); a small ws_bytes makes
// the GEMM run several row chunks (the workspace holds its fp16 rows and fp32 result).
int run_gemm(int M, int N, int K, bool out_f32 = false, int big_row = -1,
             std::size_t ws_bytes = std::size_t{32} << 20) {
  mi::backend::load_cublaslt();
  auto xv = normal(static_cast<std::size_t>(M) * K, 51, 1.f);
  if (big_row >= 0) {
    for (int d = 0; d < K; ++d) xv[static_cast<std::size_t>(big_row) * K + d] *= 3e5f;
  }
  const auto wv = normal(static_cast<std::size_t>(N) * K, 52, 0.05f);
  const auto bv = normal(static_cast<std::size_t>(N), 53, 0.01f);
  std::vector<std::uint16_t> xb(xv.size()), wb(wv.size()), bb(bv.size());
  for (std::size_t i = 0; i < xv.size(); ++i) xb[i] = float_to_bf16_bits(xv[i]);
  for (std::size_t i = 0; i < wv.size(); ++i) {
    const __half h = __float2half_rn(wv[i]);
    std::memcpy(&wb[i], &h, sizeof h);
  }
  for (std::size_t i = 0; i < bv.size(); ++i) bb[i] = float_to_bf16_bits(bv[i]);
  std::vector<float> bf(bv.size());
  for (std::size_t i = 0; i < bv.size(); ++i) bf[i] = bf16_to_float(bb[i]);  // the same bias values for an f32 out

  const std::size_t out_elem = out_f32 ? 4 : 2;
  DevBuf dx(xb.size() * 2), dw(wb.size() * 2), db(bb.size() * out_elem);
  DevBuf dout(static_cast<std::size_t>(M) * N * out_elem);
  DevBuf dws(ws_bytes);
  dx.upload(xb);
  dw.upload(wb);
  if (out_f32) {
    db.upload(bf);
  } else {
    db.upload(bb);
  }
  MI_CUDA_CHECK(cudaMemset(dout.get(), 0, static_cast<std::size_t>(M) * N * out_elem));

  const DType out_type = out_f32 ? DType::kF32 : DType::kBF16;
  const Tensor tx = Tensor::view(dx.get(), DType::kBF16, Device::kCUDA, {M, K});
  const Tensor tw = Tensor::view(dw.get(), DType::kF16, Device::kCUDA, {N, K});
  const Tensor tb = Tensor::view(db.get(), out_type, Device::kCUDA, {N});
  const Tensor to = Tensor::view(dout.get(), out_type, Device::kCUDA, {M, N});
  mi::kernels::Gemm gemm;
  gemm.linear(to, tx, tw, &tb, dws.get(), ws_bytes, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<std::size_t>(M) * N);
  if (out_f32) {
    got = dout.download<float>(got.size());
  } else {
    const auto bits = dout.download<std::uint16_t>(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = bf16_to_float(bits[i]);
  }

  double dot = 0, ng = 0, nw = 0;
  float max_abs = 0.f, max_rel = 0.f;
  bool finite = true;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      double acc = 0;
      for (int d = 0; d < K; ++d) {
        const float w = [&] {
          __half h;
          std::memcpy(&h, &wb[static_cast<std::size_t>(c) * K + d], sizeof h);
          return __half2float(h);
        }();
        acc += static_cast<double>(bf16_to_float(xb[static_cast<std::size_t>(r) * K + d])) * w;
      }
      const float want = static_cast<float>(acc) + bf[static_cast<std::size_t>(c)];
      const float g = got[static_cast<std::size_t>(r) * N + c];
      finite = finite && std::isfinite(g);
      dot += static_cast<double>(g) * want;
      ng += static_cast<double>(g) * g;
      nw += static_cast<double>(want) * want;
      max_abs = std::max(max_abs, std::fabs(g - want));
      max_rel = std::max(max_rel, std::fabs(g - want) / std::max(1e-3f, std::fabs(want)));
    }
  }
  const double cosine = dot / std::sqrt(ng * nw);
  std::printf("gemm     M %d N %d K %d out %s big row %d ws %zu: cosine %.9f max|diff| %.3g max|rel| %.3g%s\n", M, N, K,
              out_f32 ? "f32" : "bf16", big_row, ws_bytes, cosine, max_abs, max_rel, finite ? "" : " NON-FINITE");
  return finite && cosine >= 0.9999 && max_rel < 0.01 ? 0 : 1;
}

// VAE sm_75 conv2d: NHWC fp32 with KRSC filters, bias + residual, against a host reference.
int run_vae_conv_case(int N, int H, int W, int C, int K, int R, std::uint64_t seed) {
  const auto xv = normal(static_cast<std::size_t>(N) * H * W * C, seed, 1.f);
  const auto wv = normal(static_cast<std::size_t>(K) * R * R * C, seed + 1, 0.2f);
  const auto bv = normal(static_cast<std::size_t>(K), seed + 2, 0.1f);
  const auto rv = normal(static_cast<std::size_t>(N) * H * W * K, seed + 3, 0.5f);
  DevBuf dx(xv.size() * 4), dw(wv.size() * 4), db(bv.size() * 4), dr(rv.size() * 4), dy(rv.size() * 4);
  dx.upload(xv);
  dw.upload(wv);
  db.upload(bv);
  dr.upload(rv);
  MI_CUDA_CHECK(cudaMemset(dy.get(), 0, rv.size() * 4));
  const Tensor tx = Tensor::view(dx.get(), DType::kF32, Device::kCUDA, {N, H, W, C});
  const Tensor tw = Tensor::view(dw.get(), DType::kF32, Device::kCUDA, {K, R, R, C});
  const Tensor tb = Tensor::view(db.get(), DType::kF32, Device::kCUDA, {K});
  const Tensor tr = Tensor::view(dr.get(), DType::kF32, Device::kCUDA, {N, H, W, K});
  const Tensor ty = Tensor::view(dy.get(), DType::kF32, Device::kCUDA, {N, H, W, K});
  mi::kernels::sm75::vae_conv2d(tx, tw, tb, &tr, ty, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got = dy.download<float>(rv.size());
  const int pad = (R - 1) / 2;
  double dot = 0, ng = 0, nw = 0;
  float max_abs = 0.f;
  for (int n = 0; n < N; ++n) {
    for (int oy = 0; oy < H; ++oy) {
      for (int ox = 0; ox < W; ++ox) {
        for (int k = 0; k < K; ++k) {
          double acc = 0;
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < R; ++s) {
              const int gy = oy - pad + r, gx = ox - pad + s;
              if (gy < 0 || gy >= H || gx < 0 || gx >= W) continue;
              for (int c = 0; c < C; ++c) {
                acc += static_cast<double>(xv[(static_cast<std::size_t>(n) * H + gy) * W * C + gx * C + c]) *
                       wv[((static_cast<std::size_t>(k) * R + r) * R + s) * C + c];
              }
            }
          }
          const float want = static_cast<float>(acc) + bv[static_cast<std::size_t>(k)] +
                             rv[(static_cast<std::size_t>(n) * H + oy) * W * K + ox * K + k];
          const float g = got[(static_cast<std::size_t>(n) * H + oy) * W * K + ox * K + k];
          dot += static_cast<double>(g) * want;
          ng += static_cast<double>(g) * g;
          nw += static_cast<double>(want) * want;
          max_abs = std::max(max_abs, std::fabs(g - want));
        }
      }
    }
  }
  const double cosine = dot / std::sqrt(ng * nw);
  std::printf("vae_conv N%d H%d W%d C%d K%d R%d: cosine %.9f max|diff| %.3g\n", N, H, W, C, K, R, cosine, max_abs);
  return cosine >= 0.99999 ? 0 : 1;
}

// VAE sm_75 mid attention (fp32, single head) against a host reference.
int run_vae_attn() {
  const int B = 1, L = 12, D = 64;
  const auto qv = normal(static_cast<std::size_t>(B) * L * D, 71, 1.f);
  const auto kv = normal(static_cast<std::size_t>(B) * L * D, 72, 1.f);
  const auto vv = normal(static_cast<std::size_t>(B) * L * D, 73, 1.f);
  DevBuf dq(qv.size() * 4), dk(kv.size() * 4), dv(vv.size() * 4), dout(qv.size() * 4);
  dq.upload(qv);
  dk.upload(kv);
  dv.upload(vv);
  MI_CUDA_CHECK(cudaMemset(dout.get(), 0, qv.size() * 4));
  const Tensor tq = Tensor::view(dq.get(), DType::kF32, Device::kCUDA, {B, L, D});
  const Tensor tk = Tensor::view(dk.get(), DType::kF32, Device::kCUDA, {B, L, D});
  const Tensor tv = Tensor::view(dv.get(), DType::kF32, Device::kCUDA, {B, L, D});
  const Tensor to = Tensor::view(dout.get(), DType::kF32, Device::kCUDA, {B, L, D});
  const float scale = 1.f / std::sqrt(static_cast<float>(D));
  mi::kernels::sm75::vae_attention_fp32(tq, tk, tv, to, scale, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got = dout.download<float>(qv.size());
  std::vector<float> want(qv.size());
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < L; ++i) {
      std::vector<double> s(L);
      double m = -1e30;
      for (int j = 0; j < L; ++j) {
        double acc = 0;
        for (int d = 0; d < D; ++d) acc += static_cast<double>(qv[(static_cast<std::size_t>(b) * L + i) * D + d]) *
                                           kv[(static_cast<std::size_t>(b) * L + j) * D + d];
        s[static_cast<std::size_t>(j)] = acc * scale;
        m = std::max(m, s[static_cast<std::size_t>(j)]);
      }
      double sum = 0;
      for (int j = 0; j < L; ++j) {
        s[static_cast<std::size_t>(j)] = std::exp(s[static_cast<std::size_t>(j)] - m);
        sum += s[static_cast<std::size_t>(j)];
      }
      for (int d = 0; d < D; ++d) {
        double acc = 0;
        for (int j = 0; j < L; ++j) {
          acc += s[static_cast<std::size_t>(j)] * vv[(static_cast<std::size_t>(b) * L + j) * D + d];
        }
        want[(static_cast<std::size_t>(b) * L + i) * D + d] = static_cast<float>(acc / sum);
      }
    }
  }
  double dot = 0, ng = 0, nw = 0;
  float max_abs = 0.f;
  for (std::size_t i = 0; i < want.size(); ++i) {
    dot += static_cast<double>(got[i]) * want[i];
    ng += static_cast<double>(got[i]) * got[i];
    nw += static_cast<double>(want[i]) * want[i];
    max_abs = std::max(max_abs, std::fabs(got[i] - want[i]));
  }
  std::printf("vae_attn B%d L%d D%d: cosine %.9f max|diff| %.3g\n", B, L, D, dot / std::sqrt(ng * nw), max_abs);
  return dot / std::sqrt(ng * nw) >= 0.99999 ? 0 : 1;
}

// sm_75 INT8 ConvRot GEMM: int8 x/w with per-row/col fp32 scales, bf16 output. The int32 dot product is exact, so the
// output must be bit-identical to the host reference of the same epilogue order ((acc * xs) * ws + bias, bf16 rounding).
int run_int8_gemm(int M, int N, int K) {
  std::vector<std::int8_t> xv(static_cast<std::size_t>(M) * K), wv(static_cast<std::size_t>(N) * K);
  std::vector<float> xs(static_cast<std::size_t>(M)), ws(static_cast<std::size_t>(N));
  std::vector<float> bv(static_cast<std::size_t>(N));
  std::uint64_t seed = 81;
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<std::uint32_t>(seed >> 33);
  };
  for (auto& v : xv) v = static_cast<std::int8_t>(static_cast<int>(next() % 255) - 127);
  for (auto& v : wv) v = static_cast<std::int8_t>(static_cast<int>(next() % 255) - 127);
  for (auto& s : xs) s = 0.001f + static_cast<float>(next() % 1000) * 1e-5f;
  for (auto& s : ws) s = 0.0005f + static_cast<float>(next() % 1000) * 1e-5f;
  const auto bias_f = normal(static_cast<std::size_t>(N), 82, 0.05f);
  std::vector<std::uint16_t> bb(bias_f.size());
  for (std::size_t i = 0; i < bias_f.size(); ++i) bb[i] = float_to_bf16_bits(bias_f[i]);

  DevBuf dx(xv.size()), dw(wv.size()), dxs(xs.size() * 4), dws(ws.size() * 4), db(bb.size() * 2),
      dout(static_cast<std::size_t>(M) * N * 2);
  dx.upload(xv);
  dw.upload(wv);
  dxs.upload(xs);
  dws.upload(ws);
  db.upload(bb);
  MI_CUDA_CHECK(cudaMemset(dout.get(), 0, static_cast<std::size_t>(M) * N * 2));

  mi::kernels::quant::Int8Operands op;
  op.x_q = static_cast<const std::int8_t*>(dx.get());
  op.ldx = K;
  op.x_scale = static_cast<const float*>(dxs.get());
  op.w = static_cast<const std::int8_t*>(dw.get());
  op.w_scale = static_cast<const float*>(dws.get());
  op.m = M;
  op.n = N;
  op.k = K;
  auto* bias = static_cast<const __nv_bfloat16*>(db.get());
  auto* out = static_cast<__nv_bfloat16*>(dout.get());
  mi::kernels::quant::int8_gemm(op, bias, out, N, nullptr);
  MI_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got = dout.download<std::uint16_t>(static_cast<std::size_t>(M) * N);

  int mismatches = 0;
  float max_abs = 0.f;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      std::int32_t acc = 0;
      for (int d = 0; d < K; ++d) {
        acc += static_cast<std::int32_t>(xv[static_cast<std::size_t>(r) * K + d]) *
               static_cast<std::int32_t>(wv[static_cast<std::size_t>(c) * K + d]);
      }
      float v = static_cast<float>(acc) * xs[static_cast<std::size_t>(r)];
      v = v * ws[static_cast<std::size_t>(c)];
      v += bf16_to_float(bb[static_cast<std::size_t>(c)]);
      const std::uint16_t want = float_to_bf16_bits(v);
      const std::uint16_t g = got[static_cast<std::size_t>(r) * N + c];
      if (g != want) {
        ++mismatches;
        max_abs = std::max(max_abs, std::fabs(bf16_to_float(g) - bf16_to_float(want)));
      }
    }
  }
  std::printf("int8     M %d N %d K %d: %d/%d mismatches, max|diff| %.3g\n", M, N, K, mismatches, M * N, max_abs);
  return mismatches == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string what = argc > 1 ? argv[1] : "all";
  mi::backend::ProbeOptions options;
  options.query_free_memory = false;
  const mi::backend::ProbeResult probe = mi::backend::probe(options);
  std::printf("%s", mi::backend::format_report(probe).c_str());
  int failures = 0;
  try {
    if (what == "convert" || what == "all") failures += run_convert();
    if (what == "gemm" || what == "all") {
      failures += run_gemm(64, 128, 256);
      failures += run_gemm(3, 8, 32);  // a tail case: m not a multiple of anything
      failures += run_gemm(64, 128, 256, /*out_f32=*/true);
      failures += run_gemm(64, 128, 256, false, /*big_row=*/5);  // a row above fp16's range (scaled, not inf)
      // Several row chunks (1 MiB workspace: 357 rows of N 512 / K 256 each), f32 and bf16 out.
      failures += run_gemm(1000, 512, 256, /*out_f32=*/true, 999, std::size_t{1} << 20);
      failures += run_gemm(1000, 512, 256, false, 7, std::size_t{1} << 20);
    }
    if (what == "vae") {
      failures += run_vae_conv_case(1, 6, 6, 8, 8, 3, 61);
      failures += run_vae_conv_case(1, 8, 8, 96, 96, 3, 65);   // decoder widths: several channel chunks and K blocks
      failures += run_vae_conv_case(1, 5, 7, 96, 3, 1, 69);    // conv_out: 1x1, K = 3
      failures += run_vae_conv_case(1, 4, 4, 32, 96, 1, 73);   // 1x1 shortcut
      failures += run_vae_attn();
    }
    if (what == "int8") {
      failures += run_int8_gemm(64, 128, 256);
      failures += run_int8_gemm(3, 8, 64);   // tiny M: one tile row
      failures += run_int8_gemm(257, 384, 1024);
    }
    if (what == "small" || what == "all") {
      if (what == "small") {
        failures += run_attention("one", 1, 1, 128, false, {1}, 31, /*dump=*/true);
        failures += run_attention("two", 1, 1, 128, false, {2}, 32, /*dump=*/true);
        failures += run_attention("small", 1, 1, 128, false, {8}, 21, /*dump=*/true);
      }
    }
    if (what == "attn128" || what == "all") {
      failures += run_attention("attn128", 4, 2, 128, false, {64, 128, 1}, 11);
      failures += run_attention("attn128", 4, 2, 128, false, {257}, 12);
    }
    if (what == "attn256" || what == "all") {
      failures += run_attention("attn256", 4, 1, 256, true, {32, 100}, 13);
      failures += run_attention("attn256", 4, 1, 256, true, {256}, 14);
      // Causal with more keys than queries (bottom-right: the last query sees every key) and fewer (the first queries
      // see none: zero rows).
      failures += run_attention("attn256", 4, 1, 256, true, {40, 64}, 15, false, {100, 64});
      failures += run_attention("attn256", 4, 1, 256, true, {100}, 16, false, {30});
    }
    if (what == "big") {
      // Engine-shaped cases: 24 heads at head_dim 128 (the DiT's joint attention) and the text encoder's 16/4 GQA at
      // head_dim 256. The host reference is O(L^2 D), so these stay at a few hundred tokens.
      failures += run_attention("dit512", 24, 24, 128, false, {512, 512}, 41);
      failures += run_attention("te282", 16, 4, 256, true, {282}, 42);
      failures += run_attention("dit76", 24, 24, 128, false, {76, 200, 1}, 43);
    }
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 2;
  } catch (...) {
    std::printf("UNKNOWN EXCEPTION\n");
    return 3;
  }
  std::printf("%s\n", failures == 0 ? "OK" : "FAILED");
  return failures == 0 ? 0 : 1;
}
