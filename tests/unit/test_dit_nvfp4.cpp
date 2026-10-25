// NVFP4 plan phase N5: the DiT on the NVFP4 compute path (models/moss_dit/dit_block_quant with kernels/nvfp4), on an
// RTX 50 (kernels::nvfp4::available(); every test here skips elsewhere).
//
// Block level, against an emulation of the same quantized math (test G3: cos >= 0.99999 and <= 2 bf16 ULP on >= 99 %
// of the block's outputs): each NVFP4 GEMM is its input quantized by the CPU codec (quantize_nvfp4_act_bf16, bit-exact
// to the GPU quantizers, test T3), the exact sum of the decoded E2M1 x UE4M3 products in double, and the epilogue in
// fp32 in the kernels' documented order; the ops around the GEMMs (norm and modulation, QK-norm + RoPE, attention,
// SiTU-GLU) and the INT8 keep-list sites run the engine's own kernels on the reference's tensors.
//   - a synthetic all-NVFP4 block (codec-quantized random weights), at an M below one 128-row tile and one across
//     several, two varlen rows;
//   - the same block with q / k / v as three NVFP4 parts (split QKV) == the fused GEMM, bit for bit;
//   - runtime LoRA on all four sites: B = 0 is bit-identical to no LoRA (E1L / E2L with Delta = 0, and the
//     materialized ffn the ffn_down side branch needs), a real rank-8 LoRA (q and v only at the QKV site, so k's columns
//     get no term) against the emulation with Delta = bf16(t . B^T) from the adapter's own factors;
//   - `big` sized as forward() sizes it (nothing written past it), and the buffer refusals;
//   - blocks of the NVFP4 files present (MOSS_NVFP4_DIT, e.g. our converter's output, and BennyDaBall's community file):
//     an all-NVFP4 block and one with INT8 keep-list linears.
// Model level, with an NVFP4 file: one varlen batch == each row alone, streamed blocks (file and pinned host sources) ==
// resident, CUDA graphs == eager, bit for bit (exact and 8-bit attention); NVFP4 blocks move ~0.3 of the bf16 bytes.
// Speed: the block-level NVFP4 / INT8 GEMM rate (the planner's kNvfp4SpeedVsInt8) and a 1024^2 CFG step.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/attention_q8/attention_q8.h"
#include "kernels/basic/elementwise.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/norm.h"
#include "kernels/basic/rope.h"
#include "kernels/fused/qk_norm_rope.h"
#include "kernels/nvfp4/nvfp4.h"
#include "kernels/quant/quant.h"
#include "models/lora/lora.h"
#include "models/lora/side_branch.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/dit_block_quant.h"
#include "models/moss_dit/dit_weights.h"
#include "models/moss_dit/moss_dit.h"
#include "models/moss_dit/nvfp4_support.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "runtime/stream_set.h"
#include "test_lora_util.h"
#include "weights/model_index.h"
#include "weights/quant/nvfp4_codec.h"
#include "weights/quant/quant_model.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
namespace md = mi::moss_dit;
namespace kn = mi::kernels::nvfp4;
namespace mq = mi::quant;
namespace tl = mi::testing::lora;
using mi::DType;
using mi::Tensor;

constexpr std::int64_t kD = md::kDim, kF = md::kFfnDim, kH = md::kHeads, kHd = md::kHeadDim;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr int kBlock = 5;
const float kScale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHd)));

fs::path env_path(const char* var, const char* fallback) {
  const char* env = std::getenv(var);
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path(fallback);
}
fs::path community() { return env_path("MOSS_COMMUNITY", "/workspace/Models/community"); }
// The NVFP4 files to test: MOSS_NVFP4_DIT (our converter's output) and BennyDaBall's, those present.
std::vector<fs::path> nvfp4_files() {
  std::vector<fs::path> out;
  if (const char* env = std::getenv("MOSS_NVFP4_DIT"); env != nullptr && *env != '\0' && fs::is_regular_file(env)) out.emplace_back(env);
  const fs::path benny = community() / "bennydaball" / "moss_image_sft_nvfp4.safetensors";
  if (fs::is_regular_file(benny)) out.push_back(benny);
  return out;
}
fs::path int8_file() { return env_path("MOSS_INT8_DIT", "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors"); }
fs::path sft_dir() { return env_path("MOSS_WEIGHTS", "/workspace/Models/MOSS-Image") / "sft"; }

bool nvfp4_here() { return kn::available(); }

float from_bf16(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }
std::uint16_t to_bf16(float v) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(v);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}
float round_bf16(float v) { return from_bf16(to_bf16(v)); }

struct DevTensor {
  mi::DeviceBuffer buf;
  Tensor t;
  DevTensor(DType dt, std::initializer_list<std::int64_t> shape) {
    t = Tensor::view(nullptr, dt, mi::Device::kCUDA, shape);
    buf = mi::DeviceBuffer(std::max<std::size_t>(t.nbytes(), 1));
    t.data = buf.get();
  }
};

template <typename T>
std::vector<T> download(const void* p, std::size_t n) {
  std::vector<T> h(n);
  MI_CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
template <typename T>
std::vector<T> download(const Tensor& t) {
  return download<T>(t.data, static_cast<std::size_t>(t.numel()));
}

// Hash activations (splitmix64, as test_dit_int4): per-channel exponents, every 397th channel 64x larger.
std::uint64_t mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
std::vector<std::uint16_t> synth_act(std::uint64_t seed, std::int64_t rows, std::int64_t cols) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows * cols));
  for (std::int64_t c = 0; c < cols; ++c) {
    const int c_exp = static_cast<int>(mix(static_cast<std::uint64_t>(c) * 7919 + seed) % 6) - 3 + (c % 397 == 5 ? 6 : 0);
    for (std::int64_t r = 0; r < rows; ++r) {
      const std::uint64_t z = mix(static_cast<std::uint64_t>(r * cols + c) + seed * 0x9E3779B97F4A7C15ull);
      const float m = static_cast<float>(128 + (z & 0x7F));
      const int e = static_cast<int>((z >> 7) & 7);
      out[static_cast<std::size_t>(r * cols + c)] =
          static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(std::ldexp((z >> 10) & 1 ? -m : m, c_exp - 7 - e)) >> 16);
    }
  }
  return out;
}

// The modulation vectors (shift_a, scale_a, gate_a, shift_f, scale_f, gate_f), DiT-like magnitudes.
std::vector<std::uint16_t> synth_mod() {
  const std::vector<float> v = tl::randn(static_cast<std::size_t>(6 * kD), 77, 0.3f);
  return tl::to_bf16(v);
}

// ---- Agreement (G3) --------------------------------------------------------------------------------------------

int ordered(std::uint16_t b) { return (b & 0x8000) ? -static_cast<int>(b & 0x7fff) : static_cast<int>(b); }
struct Agreement {
  double exact = 0, within2 = 0, cosine = 0, update_cosine = 0;
  int max_ulp = 0;
};
Agreement agree(const std::vector<std::uint16_t>& ref, const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& x) {
  Agreement a;
  double dot = 0, rr = 0, gg = 0, udot = 0, uu = 0, vv = 0;
  std::size_t exact = 0, w2 = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const int ulp = std::abs(ordered(ref[i]) - ordered(got[i]));
    a.max_ulp = std::max(a.max_ulp, ulp);
    exact += ulp == 0;
    w2 += ulp <= 2;
    const double r = from_bf16(ref[i]), g = from_bf16(got[i]), x0 = from_bf16(x[i]);
    dot += r * g;
    rr += r * r;
    gg += g * g;
    udot += (r - x0) * (g - x0);
    uu += (r - x0) * (r - x0);
    vv += (g - x0) * (g - x0);
  }
  a.exact = static_cast<double>(exact) / static_cast<double>(ref.size());
  a.within2 = static_cast<double>(w2) / static_cast<double>(ref.size());
  a.cosine = dot / std::sqrt(rr * gg);
  a.update_cosine = udot / std::sqrt(uu * vv);
  return a;
}
void print(const char* what, const Agreement& a) {
  std::printf("[dit_nvfp4] %s: %.4f%% bit-exact, %.4f%% within 2 ULP (max %d), cos %.9f, update cos %.7f\n", what,
              100 * a.exact, 100 * a.within2, a.max_ulp, a.cosine, a.update_cosine);
}
void expect_g3(const Agreement& a) {
  EXPECT_GE(a.cosine, 0.99999);
  EXPECT_GE(a.within2, 0.99);
}

std::size_t differing(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return n;
}

// ---- The emulation of one NVFP4 GEMM ---------------------------------------------------------------------------

float e4m3_value(std::uint8_t b) {
  const int e = (b >> 3) & 0xF, m = b & 7;
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
  return (b & 0x80) ? -v : v;
}

// Decoded values e2m1(q) * e4m3(sf) of `rows` rows of k elements (hi_first codes, swizzled scales of `rows` rows).
std::vector<float> decode(const std::uint8_t* packed, const std::uint8_t* sf, std::int64_t rows, std::int64_t k) {
  std::vector<float> v(static_cast<std::size_t>(rows * k));
  const std::int64_t sfk = k / 16;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t c = 0; c < k; ++c) {
      const std::uint8_t byte = packed[r * (k / 2) + c / 2];
      const std::uint8_t code = c % 2 == 0 ? byte >> 4 : byte & 0xF;
      v[static_cast<std::size_t>(r * k + c)] = mq::e2m1_to_float(code) * e4m3_value(sf[mq::nvfp4_scale_offset(r, c / 16, sfk)]);
    }
  }
  return v;
}

struct HostNvfp4 {
  std::int64_t n = 0, k = 0;
  std::vector<float> w;  // decoded e2m1 * e4m3 [n, k]
  float s2 = 1.0f;
};
HostNvfp4 host_of(const md::Nvfp4Linear& l) {
  HostNvfp4 h;
  h.n = l.w.dim(0);
  h.k = l.k();
  const auto packed = download<std::uint8_t>(l.w);
  const auto sf = download<std::uint8_t>(l.sf);
  h.w = decode(packed.data(), sf.data(), h.n, h.k);
  h.s2 = l.scale2;
  return h;
}

// The sum of x̂ · ŵ per output in double (exact up to rounding far below fp32's), on all cores.
std::vector<double> exact_dot(const std::vector<float>& xv, std::int64_t m, const HostNvfp4& w) {
  std::vector<double> acc(static_cast<std::size_t>(m * w.n));
  const unsigned threads = std::max(1u, std::min(24u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (std::int64_t n = t; n < w.n; n += threads) {
        const float* wr = w.w.data() + n * w.k;
        for (std::int64_t r = 0; r < m; ++r) {
          const float* xr = xv.data() + r * w.k;
          double s = 0;
          for (std::int64_t c = 0; c < w.k; ++c) s += static_cast<double>(xr[c]) * static_cast<double>(wr[c]);
          acc[static_cast<std::size_t>(r * w.n + n)] = s;
        }
      }
    });
  }
  for (auto& th : pool) th.join();
  return acc;
}

// E1 / E1L of nvfp4.h on bf16 input rows: y = bf16(((acc * g[r]) * s2) [+ delta]); no bias in the DiT.
std::vector<std::uint16_t> emulate_e1(const std::vector<std::uint16_t>& x, std::int64_t m, const HostNvfp4& w,
                                      const std::vector<float>* delta = nullptr) {
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(m * w.k / 2));
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(mq::nvfp4_scale_rows(m) * mq::nvfp4_scale_cols(w.k / 16)));
  std::vector<float> g(static_cast<std::size_t>(m));
  mq::quantize_nvfp4_act_bf16(x.data(), m, w.k, packed.data(), sf.data(), g.data());
  const std::vector<double> acc = exact_dot(decode(packed.data(), sf.data(), m, w.k), m, w);
  std::vector<std::uint16_t> y(acc.size());
  for (std::int64_t r = 0; r < m; ++r) {
    for (std::int64_t n = 0; n < w.n; ++n) {
      const std::size_t i = static_cast<std::size_t>(r * w.n + n);
      float v = static_cast<float>(acc[i]) * g[static_cast<std::size_t>(r)];
      v = v * w.s2;
      if (delta != nullptr) v = v + (*delta)[i];
      y[i] = to_bf16(v);
    }
  }
  return y;
}
// E2: residual = bf16(residual + bf16(y * gate[c])), y the bf16 E1 value.
void emulate_e2(std::vector<std::uint16_t>& residual, const std::vector<std::uint16_t>& y, const std::uint16_t* gate,
                std::int64_t m, std::int64_t n) {
  for (std::int64_t r = 0; r < m; ++r) {
    for (std::int64_t c = 0; c < n; ++c) {
      const std::size_t i = static_cast<std::size_t>(r * n + c);
      residual[i] = to_bf16(from_bf16(residual[i]) + round_bf16(from_bf16(y[i]) * from_bf16(gate[c])));
    }
  }
}

// ---- Synthetic NVFP4 weights -----------------------------------------------------------------------------------

struct DevNvfp4 {
  mi::DeviceBuffer w, sf;
  md::Nvfp4Linear lin;
};
// Random fp32 [n, k] weights with row-dependent spread, quantized by the codec (the converter's rtn recipe).
std::unique_ptr<DevNvfp4> synth_weight(std::int64_t n, std::int64_t k, std::uint32_t seed) {
  std::vector<float> w = tl::randn(static_cast<std::size_t>(n * k), seed, 0.02f);
  float amax = 0;
  for (std::int64_t r = 0; r < n; ++r) {
    const float spread = 1.0f + static_cast<float>(r % 7);
    for (std::int64_t c = 0; c < k; ++c) {
      float& v = w[static_cast<std::size_t>(r * k + c)];
      v *= spread;
      amax = std::max(amax, std::abs(v));
    }
  }
  const float s2 = mq::nvfp4_tensor_scale(amax);
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(n * k / 2));
  std::vector<std::uint8_t> sf(static_cast<std::size_t>(mq::nvfp4_scale_rows(n) * mq::nvfp4_scale_cols(k / 16)));
  mq::quantize_nvfp4_weight(w.data(), n, k, s2, packed.data(), sf.data());
  auto d = std::make_unique<DevNvfp4>();
  d->w = mi::DeviceBuffer(packed.size());
  d->w.upload(packed.data(), packed.size());
  d->sf = mi::DeviceBuffer(sf.size());
  d->sf.upload(sf.data(), sf.size());
  d->lin.w = Tensor::view(d->w.get(), DType::kU8, mi::Device::kCUDA, {n, k / 2});
  d->lin.sf = Tensor::view(d->sf.get(), DType::kU8, mi::Device::kCUDA, {mq::nvfp4_scale_rows(n), mq::nvfp4_scale_cols(k / 16)});
  d->lin.scale2 = s2;
  return d;
}

struct SynthBlock {
  std::unique_ptr<DevNvfp4> qkv, o, up_gate, down;
  std::unique_ptr<DevTensor> norm1, norm2, q_norm, k_norm;
  md::BlockTensors w;
  SynthBlock() {
    qkv = synth_weight(3 * kD, kD, 1);
    o = synth_weight(kD, kD, 2);
    up_gate = synth_weight(2 * kF, kD, 3);
    down = synth_weight(kD, kF, 4);
    const auto norm = [](std::int64_t n, std::uint32_t seed) {
      auto t = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{n});
      std::vector<float> v = tl::randn(static_cast<std::size_t>(n), seed, 0.1f);
      for (float& x : v) x += 1.0f;
      const auto b = tl::to_bf16(v);
      t->buf.upload(b.data(), b.size() * 2);
      return t;
    };
    norm1 = norm(kD, 5);
    norm2 = norm(kD, 6);
    q_norm = norm(kHd, 7);
    k_norm = norm(kHd, 8);
    w.norm1 = norm1->t;
    w.norm2 = norm2->t;
    w.q_norm = q_norm->t;
    w.k_norm = k_norm->t;
    w.qkv_nvfp4 = qkv->lin;
    w.o_nvfp4 = o->lin;
    w.up_gate_nvfp4 = up_gate->lin;
    w.down_nvfp4 = down->lin;
  }
};

// ---- One block call, as MossDit::forward lays out its buffers ---------------------------------------------------

struct BlockCall {
  std::int64_t l = 0, hw = 0;
  std::vector<std::int32_t> cu_h;
  std::unique_ptr<DevTensor> x, y, big, rope, mod, scale, cu, lora_t, delta;
  mi::DeviceBuffer lse, q8, gemm_ws, lora_ws, sf, scratch;
  md::BlockRun run;
  md::QuantBlockBuffers qa;
  mi::kernels::Gemm gemm;
  BlockCall(std::int64_t grid_h, std::int64_t grid_w, const std::vector<std::uint16_t>& xv, std::span<const std::int64_t> text,
      const std::vector<std::uint16_t>& mv, cudaStream_t s, const mi::lora::SideBranches* lora = nullptr) {
    hw = grid_h * grid_w;
    cu_h = {0};
    int max_len = 0;
    for (const std::int64_t t : text) {
      cu_h.push_back(cu_h.back() + static_cast<std::int32_t>(hw + t));
      max_len = std::max(max_len, static_cast<int>(hw + t));
    }
    l = cu_h.back();
    x = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, kD});
    y = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, kD});
    big = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, md::kOpByOpBigCols});
    rope = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, 128});
    mod = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{6 * kD});
    scale = std::make_unique<DevTensor>(DType::kF32, std::initializer_list<std::int64_t>{l});
    cu = std::make_unique<DevTensor>(DType::kI32, std::initializer_list<std::int64_t>{static_cast<std::int64_t>(cu_h.size())});
    EXPECT_EQ(xv.size(), static_cast<std::size_t>(l * kD));
    x->buf.upload(xv.data(), xv.size() * 2);
    mod->buf.upload(mv.data(), mv.size() * 2);
    cu->buf.upload(cu_h.data(), cu_h.size() * 4);
    for (std::size_t r = 0; r < text.size(); ++r) {
      mi::kernels::moss_rope_table(rope->t.slice0(cu_h[r], cu_h[r + 1]), static_cast<int>(grid_h), static_cast<int>(grid_w),
                                   static_cast<int>(text[r]), s);
    }
    lse = mi::DeviceBuffer(mi::attention::flash_fwd_workspace_bytes(l, kH));
    const std::size_t q8_bytes = md::q8_attention_workspace_bytes(hw, text, md::AttentionMode::kQ8Fp8);
    q8 = mi::DeviceBuffer(std::max<std::size_t>(q8_bytes, 1));
    gemm_ws = mi::DeviceBuffer(mi::kernels::Gemm::kWorkspaceBytes);
    sf = mi::DeviceBuffer(md::nvfp4_sf_bytes(l));
    run = {0, l, cu->t, max_len};
    qa.f = {x->t, y->t, big->t, rope->t, mod->t, lse.get(), lse.size()};
    qa.scale = scale->t;
    qa.attn_workspace = q8.get();
    qa.attn_workspace_bytes = q8_bytes;
    qa.nvfp4_sf = sf.get();
    qa.nvfp4_sf_bytes = sf.size();
    if (lora != nullptr) {
      lora_t = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, lora->max_rank()});
      lora_ws = mi::DeviceBuffer(mi::lora::t_workspace_bytes(l, lora->max_rank()));
      delta = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, 2 * kF});
      scratch = mi::DeviceBuffer(mi::lora::delta_scratch_bytes(l, lora->max_rank(), 2 * kF));
      qa.lora = lora;
      qa.lora_t = lora_t->t;
      qa.lora_workspace = lora_ws.get();
      qa.lora_workspace_bytes = lora_ws.size();
      qa.nvfp4_delta = delta->t;
      qa.nvfp4_scratch = scratch.get();
      qa.nvfp4_scratch_bytes = scratch.size();
    }
  }
  md::QkvGemm qkv_gemm() { return {&gemm, gemm_ws.get(), gemm_ws.size()}; }
};

std::vector<std::uint16_t> run_block(const md::BlockTensors& w, std::int64_t gh, std::int64_t gw,
                                     const std::vector<std::uint16_t>& xv, std::span<const std::int64_t> text,
                                     const std::vector<std::uint16_t>& mv, cudaStream_t s,
                                     md::AttentionMode mode = md::AttentionMode::kExact,
                                     const mi::lora::SideBranches* lora = nullptr) {
  BlockCall r(gh, gw, xv, text, mv, s, lora);
  md::run_block_quant(w, r.qa, r.run, r.qkv_gemm(), md::FusedRouteSnapshot{}, mode, s, kBlock);
  MI_CUDA_CHECK(cudaStreamSynchronize(s));
  return download<std::uint16_t>(r.x->t);
}

// ---- The reference block ---------------------------------------------------------------------------------------

// A LoRA site's adapter factors for the emulation: Delta[r, out0 + j] = bf16(sum_i t[r, i] * up[j, i]) with
// t = bf16(fp32(x . down^T) * coef), coef = 1 (alpha = rank, strength 1).
struct HostAdapter {
  std::int64_t out0 = 0, n = 0, k = 0, rank = 0;
  std::vector<float> down, up;  // [rank, k], [n, rank] (bf16 values)
};
std::vector<float> host_delta(const std::vector<HostAdapter>& ads, const std::vector<std::uint16_t>& x, std::int64_t m,
                              std::int64_t n_out) {
  std::vector<float> d(static_cast<std::size_t>(m * n_out), 0.0f);
  for (const HostAdapter& a : ads) {
    for (std::int64_t r = 0; r < m; ++r) {
      std::vector<float> t(static_cast<std::size_t>(a.rank));
      for (std::int64_t i = 0; i < a.rank; ++i) {
        double s = 0;
        for (std::int64_t c = 0; c < a.k; ++c) s += static_cast<double>(from_bf16(x[static_cast<std::size_t>(r * a.k + c)])) * a.down[static_cast<std::size_t>(i * a.k + c)];
        t[static_cast<std::size_t>(i)] = round_bf16(static_cast<float>(s));
      }
      for (std::int64_t j = 0; j < a.n; ++j) {
        double s = 0;
        for (std::int64_t i = 0; i < a.rank; ++i) s += static_cast<double>(t[static_cast<std::size_t>(i)]) * a.up[static_cast<std::size_t>(j * a.rank + i)];
        d[static_cast<std::size_t>(r * n_out + a.out0 + j)] = round_bf16(static_cast<float>(s));
      }
    }
  }
  return d;
}
// The adapters of block kBlock for the emulation. With `engine` set, each NVFP4 site's Delta is the engine's own
// (lora::compute_t + lora::compute_delta on the reference's bf16 input of the site), so the block's composition is
// checked exactly; the independent Delta from the factors (host_delta) is then compared with it per site (delta_*).
struct HostLora {
  std::vector<HostAdapter> qkv, o, up_gate, down;
  const mi::lora::SideBranches* engine = nullptr;
  mutable std::size_t delta_n = 0, delta_within2 = 0;
  mutable double delta_dot = 0, delta_ee = 0, delta_hh = 0;
};

// The engine's Delta of `site` for the bf16 input `in` (CUDA [l, k]), as float.
std::vector<float> engine_delta(const mi::lora::SiteBranch& site, const Tensor& in, bool natural, cudaStream_t s) {
  const std::int64_t l = in.dim(0);
  mi::kernels::Gemm gemm;
  mi::DeviceBuffer gws(mi::kernels::Gemm::kWorkspaceBytes), tws(mi::lora::t_workspace_bytes(l, site.rank)),
      scratch(mi::lora::delta_scratch_bytes(l, site.rank, site.n));
  DevTensor t(DType::kBF16, {l, site.rank}), d(DType::kBF16, {l, site.n});
  mi::lora::compute_t(site, in, t.t, tws.get(), tws.size(), gemm, gws.get(), gws.size(), s);
  mi::lora::compute_delta(site, t.t, d.t, natural, scratch.get(), scratch.size(), gemm, gws.get(), gws.size(), s);
  MI_CUDA_CHECK(cudaStreamSynchronize(s));
  const auto bits = download<std::uint16_t>(d.t);
  std::vector<float> out(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) out[i] = from_bf16(bits[i]);
  return out;
}

// Delta of a site for the emulation: the engine's (recording its agreement with the factors' own) or the factors'.
std::vector<float> site_delta(const HostLora& lora, mi::lora::Site site, const std::vector<HostAdapter>& ads,
                              const std::vector<std::uint16_t>& x, const Tensor& x_dev, std::int64_t m, std::int64_t n,
                              cudaStream_t s) {
  std::vector<float> host = host_delta(ads, x, m, n);
  if (lora.engine == nullptr) return host;
  std::vector<float> e = engine_delta(*lora.engine->site(kBlock, site), x_dev, site == mi::lora::Site::kUpGate, s);
  for (std::size_t i = 0; i < e.size(); ++i) {
    const int ulp = std::abs(ordered(to_bf16(e[i])) - ordered(to_bf16(host[i])));
    lora.delta_within2 += ulp <= 2;
    lora.delta_dot += static_cast<double>(e[i]) * host[i];
    lora.delta_ee += static_cast<double>(e[i]) * e[i];
    lora.delta_hh += static_cast<double>(host[i]) * host[i];
  }
  lora.delta_n += e.size();
  return e;
}

mi::kernels::quant::Int8Operands int8_ops(const std::int8_t* xq, std::int64_t ldx, const float* sc, const md::Int8Linear& w,
                                          std::int64_t m) {
  mi::kernels::quant::Int8Operands op;
  op.x_q = xq;
  op.ldx = ldx;
  op.x_scale = sc;
  op.w = static_cast<const std::int8_t*>(w.w.data);
  op.w_scale = static_cast<const float*>(w.scale.data);
  op.m = m;
  op.n = w.w.dim(0);
  op.k = w.w.dim(1);
  return op;
}

// The block's output for rows `text` (one varlen run of all rows), its NVFP4 linears emulated, its INT8 ones on the
// engine's INT8 kernels, its attention the engine's kernel for `attention` (exact or q8_fp8).
std::vector<std::uint16_t> reference_block(const md::BlockTensors& w, std::int64_t gh, std::int64_t gw,
                                           const std::vector<std::uint16_t>& xv, std::span<const std::int64_t> text,
                                           const std::vector<std::uint16_t>& mv, cudaStream_t s, const HostLora* lora = nullptr,
                                           md::AttentionMode attention = md::AttentionMode::kExact) {
  namespace kq = mi::kernels::quant;
  BlockCall r(gh, gw, xv, text, mv, s);  // its x, rope, mod, cu, lse and scale
  const std::int64_t l = r.l;
  const auto vec = [&](int i) { return Tensor::view(static_cast<std::uint16_t*>(r.mod->t.data) + i * kD, DType::kBF16, mi::Device::kCUDA, {kD}); };
  const auto bfp = [](const Tensor& t) { return static_cast<const __nv_bfloat16*>(t.data); };
  const auto bfm = [](const Tensor& t) { return static_cast<__nv_bfloat16*>(t.data); };
  DevTensor qkv(DType::kBF16, {l, 3 * kD}), attn(DType::kBF16, {l, kD}), up(DType::kBF16, {l, 2 * kF}), ffn(DType::kBF16, {l, kF});
  mi::DeviceBuffer xq(static_cast<std::size_t>(l * kF));
  auto* const xq8 = static_cast<std::int8_t*>(xq.get());
  float* const sc = static_cast<float*>(r.scale->t.data);
  const auto sync = [&] { MI_CUDA_CHECK(cudaStreamSynchronize(s)); };
  const std::span<const std::uint16_t> mod_h(mv);
  // qkv
  mi::kernels::rms_norm_modulate_moss(r.x->t, w.norm1, vec(1), vec(0), r.y->t, s);
  if (!w.qkv_nvfp4.empty()) {
    sync();
    const auto y = download<std::uint16_t>(r.y->t);
    std::vector<float> d;
    if (lora != nullptr) d = site_delta(*lora, mi::lora::Site::kQkv, lora->qkv, y, r.y->t, l, 3 * kD, s);
    const auto out = emulate_e1(y, l, host_of(w.qkv_nvfp4), lora != nullptr ? &d : nullptr);
    qkv.buf.upload(out.data(), out.size() * 2);
  } else {
    kq::rotate_quantize(bfp(r.y->t), kD, xq8, kD, sc, l, static_cast<int>(kD), s);
    kq::int8_gemm(int8_ops(xq8, kD, sc, w.qkv8, l), nullptr, bfm(qkv.t), 3 * kD, s);
  }
  mi::kernels::fused::qk_norm_rope_moss(qkv.t, static_cast<int>(kH), w.q_norm, w.k_norm, r.rope->t, s);
  const auto heads = [&](const Tensor& buf, std::int64_t col, std::int64_t ld) {
    Tensor t = Tensor::view(static_cast<std::uint16_t*>(buf.data) + col, DType::kBF16, mi::Device::kCUDA, {l, kH, kHd});
    t.stride[0] = ld;
    return t;
  };
  if (attention == md::AttentionMode::kQ8Fp8) {
    mi::attention_q8::attention_q8(mi::attention_q8::Options{}, heads(qkv.t, 0, 3 * kD), heads(qkv.t, kD, 3 * kD),
                                   heads(qkv.t, 2 * kD, 3 * kD), heads(attn.t, 0, kD), r.cu->t, r.cu->t, r.run.max_seqlen,
                                   kScale, r.q8.get(), r.q8.size(), s);
  } else {
    mi::attention::flash_fwd(heads(qkv.t, 0, 3 * kD), heads(qkv.t, kD, 3 * kD), heads(qkv.t, 2 * kD, 3 * kD),
                             heads(attn.t, 0, kD), r.cu->t, r.cu->t, r.run.max_seqlen, false, kScale, r.lse.get(), r.lse.size(), s);
  }
  // o
  if (!w.o_nvfp4.empty()) {
    sync();
    const auto a = download<std::uint16_t>(attn.t);
    std::vector<float> d;
    if (lora != nullptr) d = site_delta(*lora, mi::lora::Site::kO, lora->o, a, attn.t, l, kD, s);
    const auto y = emulate_e1(a, l, host_of(w.o_nvfp4), lora != nullptr ? &d : nullptr);
    auto x = download<std::uint16_t>(r.x->t);
    emulate_e2(x, y, mod_h.data() + 2 * kD, l, kD);
    r.x->buf.upload(x.data(), x.size() * 2);
  } else {
    kq::rotate_quantize(bfp(attn.t), kD, xq8, kD, sc, l, static_cast<int>(kD), s);
    kq::int8_gemm_gated_residual(int8_ops(xq8, kD, sc, w.o8, l), nullptr, bfp(vec(2)), bfm(r.x->t), kD, s);
  }
  // up_gate -> ffn
  mi::kernels::rms_norm_modulate_moss(r.x->t, w.norm2, vec(4), vec(3), r.y->t, s);
  if (!w.up_gate_nvfp4.empty()) {
    sync();
    const auto y = download<std::uint16_t>(r.y->t);
    std::vector<float> d;
    if (lora != nullptr) d = site_delta(*lora, mi::lora::Site::kUpGate, lora->up_gate, y, r.y->t, l, 2 * kF, s);
    const auto out = emulate_e1(y, l, host_of(w.up_gate_nvfp4), lora != nullptr ? &d : nullptr);
    up.buf.upload(out.data(), out.size() * 2);
    mi::kernels::situ_glu(up.t, ffn.t, s);
  } else {
    kq::rotate_quantize(bfp(r.y->t), kD, xq8, kD, sc, l, static_cast<int>(kD), s);
    kq::int8_gemm_situ_glu(int8_ops(xq8, kD, sc, w.up_gate8, l), bfm(ffn.t), kF, s);
  }
  // down
  if (!w.down_nvfp4.empty()) {
    sync();
    const auto f = download<std::uint16_t>(ffn.t);
    std::vector<float> d;
    if (lora != nullptr) d = site_delta(*lora, mi::lora::Site::kDown, lora->down, f, ffn.t, l, kD, s);
    const auto y = emulate_e1(f, l, host_of(w.down_nvfp4), lora != nullptr ? &d : nullptr);
    auto x = download<std::uint16_t>(r.x->t);
    emulate_e2(x, y, mod_h.data() + 5 * kD, l, kD);
    return x;
  }
  kq::rotate_quantize(bfp(ffn.t), kF, xq8, kF, sc, l, static_cast<int>(kF), s);
  kq::int8_gemm_gated_residual(int8_ops(xq8, kF, sc, w.down8, l), nullptr, bfp(vec(5)), bfm(r.x->t), kD, s);
  sync();
  return download<std::uint16_t>(r.x->t);
}

// ---- Block level: synthetic weights ----------------------------------------------------------------------------

class GpuDitNvfp4Block : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!nvfp4_here()) return;
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    block_ = std::make_unique<SynthBlock>();
    mod_ = synth_mod();
  }
  static void TearDownTestSuite() {
    block_.reset();
    streams_.reset();
  }
  void SetUp() override {
    if (!nvfp4_here()) GTEST_SKIP() << "the NVFP4 GEMMs do not run here: " << md::nvfp4_unsupported_reason();
  }
  cudaStream_t stream() const { return streams_->compute(); }
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<SynthBlock> block_;
  static std::vector<std::uint16_t> mod_;
};
std::unique_ptr<mi::runtime::StreamSet> GpuDitNvfp4Block::streams_;
std::unique_ptr<SynthBlock> GpuDitNvfp4Block::block_;
std::vector<std::uint16_t> GpuDitNvfp4Block::mod_;

struct Shape {
  std::int64_t gh, gw;
  std::vector<std::int64_t> text;
  std::int64_t l() const {
    std::int64_t n = 0;
    for (const std::int64_t t : text) n += gh * gw + t;
    return n;
  }
};
// Below one 128-row tile (122 rows) and across several with a partial last one (565 rows).
const Shape kShapes[] = {{6, 8, {17, 9}}, {16, 16, {40, 13}}};

TEST_F(GpuDitNvfp4Block, SyntheticMatchesEmulation) {
  EXPECT_TRUE(block_->w.nvfp4());
  EXPECT_TRUE(block_->w.quantized());
  for (const Shape& sh : kShapes) {
    const auto xv = synth_act(201 + static_cast<std::uint64_t>(sh.gh), sh.l(), kD);
    const auto ref = reference_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream());
    const auto got = run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream());
    const Agreement a = agree(ref, got, xv);
    const std::string what = "synthetic all-NVFP4 block, " + std::to_string(sh.l()) + " rows, vs emulation";
    print(what.c_str(), a);
    RecordProperty("g3_cos_" + std::to_string(sh.l()), std::to_string(a.cosine));
    RecordProperty("g3_within2_" + std::to_string(sh.l()), std::to_string(a.within2));
    expect_g3(a);
    // Deterministic: the same call twice gives the same bits.
    EXPECT_EQ(run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream()), got);
  }
}

// q, k and v as three NVFP4 linears (qkv_split; one quantized input, three GEMMs into their columns) == the fused GEMM.
TEST_F(GpuDitNvfp4Block, SplitQkvEqualsFused) {
  const Shape& sh = kShapes[1];
  const auto xv = synth_act(301, sh.l(), kD);
  md::BlockTensors w = block_->w;
  w.qkv_nvfp4 = {};
  w.qkv_split = true;
  const std::int64_t sf_part = mq::nvfp4_scale_rows(kD) * mq::nvfp4_scale_cols(kD / 16);
  for (int i = 0; i < 3; ++i) {
    md::Nvfp4Linear& p = w.qkv_nvfp4_part[i];
    p.w = block_->w.qkv_nvfp4.w.slice0(i * kD, (i + 1) * kD);
    p.sf = Tensor::view(static_cast<std::uint8_t*>(block_->w.qkv_nvfp4.sf.data) + i * sf_part, DType::kU8, mi::Device::kCUDA,
                        {mq::nvfp4_scale_rows(kD), mq::nvfp4_scale_cols(kD / 16)});
    p.scale2 = block_->w.qkv_nvfp4.scale2;
  }
  EXPECT_EQ(run_block(w, sh.gh, sh.gw, xv, sh.text, mod_, stream()), run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream()));
}

fs::path lora_dir() {
  static const fs::path dir = fs::temp_directory_path() / ("mossinfer_test_dit_nvfp4_" + std::to_string(std::random_device{}()));
  return dir;
}

// A kohya LoRA on block kBlock: `mods` (name, n, k), rank 8 (alpha 8: coefficient 1); zero_up: B = 0.
struct LoraMod {
  const char* name;
  std::int64_t n, k;
};
fs::path write_lora(const std::string& file, std::span<const LoraMod> mods, bool zero_up, HostLora* host) {
  constexpr std::int64_t kRank = 8;
  std::vector<tl::StTensor> ts;
  std::uint32_t seed = 31;
  for (const LoraMod& m : mods) {
    const std::string p = "diffusion_model.layers." + std::to_string(kBlock) + ".proj." + m.name + ".";
    HostAdapter a;
    a.n = m.n;
    a.k = m.k;
    a.rank = kRank;
    a.down = tl::randn(static_cast<std::size_t>(kRank * m.k), seed++, 0.02f);
    a.up = zero_up ? std::vector<float>(static_cast<std::size_t>(m.n * kRank), 0.0f)
                   : tl::randn(static_cast<std::size_t>(m.n * kRank), seed++, 0.05f);
    ts.push_back({p + "lora_down.weight", mi::weights::DType::kBF16, {kRank, m.k}, a.down});
    ts.push_back({p + "lora_up.weight", mi::weights::DType::kBF16, {m.n, kRank}, a.up});
    ts.push_back({p + "alpha", mi::weights::DType::kF32, {}, {static_cast<float>(kRank)}});
    if (host != nullptr) {
      const std::string name = m.name;
      if (name == "q" || name == "k" || name == "v") {
        a.out0 = (name == "q" ? 0 : name == "k" ? 1 : 2) * kD;
        host->qkv.push_back(a);
      } else if (name == "o") {
        host->o.push_back(a);
      } else if (name == "ffn_up_gate") {
        host->up_gate.push_back(a);
      } else {
        host->down.push_back(a);
      }
    }
  }
  const fs::path path = lora_dir() / file;
  tl::write_st(path, ts);
  return path;
}

// Runtime LoRA on NVFP4 sites, at one 128-row tile (122 rows), several (565) and a 1024^2 CFG step's 8484 rows (many
// waves of GEMM tiles: a GEMM whose input overlapped its own output would read codes other tiles' epilogues had
// overwritten):
//   - B = 0 on every site: Delta = 0 through E1L / E2L, the ffn materialized for ffn_down's t: the plain block's bits;
//   - a real rank-8 LoRA: q and v at the QKV site (k's columns get no term), o, ffn_up_gate (Delta in the stored
//     [hidden | gate] order: compute_delta's natural order) and ffn_down: the same bits twice; bit-exact against the
//     emulation with the engine's Delta of each site; that Delta against the one from the adapters' own factors
//     (bf16 t and Delta summed in another order: within 2 ULP); and the block against the emulation with the factors'
//     Delta, reported (a last-bit difference of Delta before a 4-bit re-quantization can move a value a whole step).
TEST_F(GpuDitNvfp4Block, RuntimeLora) {
  const LoraMod all[] = {{"q", kD, kD}, {"k", kD, kD}, {"v", kD, kD}, {"o", kD, kD}, {"ffn_up_gate", 2 * kF, kD},
                         {"ffn_down", kD, kF}};
  const mi::lora::LoraSpec zero_spec{write_lora("zero.safetensors", all, true, nullptr), 1.0f};
  mi::lora::LoraStack zero_stack(std::span<const mi::lora::LoraSpec>(&zero_spec, 1), mi::lora::moss_dit_targets());
  mi::lora::SideBranches zero(zero_stack);
  for (const mi::lora::Site s : {mi::lora::Site::kQkv, mi::lora::Site::kO, mi::lora::Site::kUpGate, mi::lora::Site::kDown}) {
    ASSERT_NE(zero.site(kBlock, s), nullptr) << mi::lora::site_name(s);
  }
  const LoraMod mods[] = {{"q", kD, kD}, {"v", kD, kD}, {"o", kD, kD}, {"ffn_up_gate", 2 * kF, kD}, {"ffn_down", kD, kF}};
  HostLora host;
  const mi::lora::LoraSpec spec{write_lora("real.safetensors", mods, false, &host), 1.0f};
  mi::lora::LoraStack stack(std::span<const mi::lora::LoraSpec>(&spec, 1), mi::lora::moss_dit_targets());
  mi::lora::SideBranches sb(stack);
  const Shape shapes[] = {kShapes[0], kShapes[1], {64, 64, {282, 10}}};
  for (const Shape& sh : shapes) {
    SCOPED_TRACE(std::to_string(sh.l()) + " rows");
    const auto xv = synth_act(401 + static_cast<std::uint64_t>(sh.l()), sh.l(), kD);
    const auto plain = run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream());
    EXPECT_EQ(run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream(), md::AttentionMode::kExact, &zero), plain);
    const auto got = run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream(), md::AttentionMode::kExact, &sb);
    EXPECT_EQ(run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream(), md::AttentionMode::kExact, &sb), got);
    HostLora with_engine = host;
    with_engine.engine = &sb;
    const Agreement a = agree(reference_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream(), &with_engine), got, xv);
    const std::string rows = std::to_string(sh.l()) + " rows";
    print(("synthetic block + rank-8 LoRA, " + rows + ", vs emulation with the engine's Delta").c_str(), a);
    EXPECT_EQ(a.exact, 1.0);
    const double dcos = with_engine.delta_dot / std::sqrt(with_engine.delta_ee * with_engine.delta_hh);
    const double dw2 = static_cast<double>(with_engine.delta_within2) / static_cast<double>(with_engine.delta_n);
    std::printf("[dit_nvfp4]   engine Delta vs the factors' Delta: %.4f%% within 2 bf16 ULP, cos %.9f\n", 100 * dw2, dcos);
    EXPECT_GE(dcos, 0.99999);
    EXPECT_GE(dw2, 0.99);
    const auto ref = reference_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream(), &host);
    const Agreement f = agree(ref, got, xv), base = agree(ref, plain, xv);
    print(("synthetic block + rank-8 LoRA, " + rows + ", vs emulation with the factors' Delta (reported)").c_str(), f);
    std::printf("[dit_nvfp4]   (the block without the LoRA vs that reference: update cos %.6f)\n", base.update_cosine);
    EXPECT_GE(f.cosine, 0.9999);
    EXPECT_LT(base.update_cosine, f.update_cosine);
  }
  // Without the Delta buffers, or with Delta narrower than the block's widest NVFP4 site with a branch (ffn_up_gate's
  // 2F): refused before any launch.
  const Shape& sh = kShapes[0];
  const auto xv = synth_act(402, sh.l(), kD);
  BlockCall r(sh.gh, sh.gw, xv, sh.text, mod_, stream(), &zero);
  r.qa.nvfp4_scratch = nullptr;
  EXPECT_THROW(md::run_block_quant(block_->w, r.qa, r.run, r.qkv_gemm(), md::FusedRouteSnapshot{}, md::AttentionMode::kExact,
                                   stream(), kBlock),
               std::invalid_argument);
  BlockCall narrow(sh.gh, sh.gw, xv, sh.text, mod_, stream(), &zero);
  narrow.qa.nvfp4_delta = Tensor::view(narrow.delta->t.data, DType::kBF16, mi::Device::kCUDA, {narrow.l, 3 * kD});
  EXPECT_THROW(md::run_block_quant(block_->w, narrow.qa, narrow.run, narrow.qkv_gemm(), md::FusedRouteSnapshot{},
                                   md::AttentionMode::kExact, stream(), kBlock),
               std::invalid_argument);
  // The model's forward sizes Delta and `big` by the NVFP4 sites that have a branch (compute_config).
  mq::QuantIndex qi;
  for (int b = 0; b < md::kNumBlocks; ++b) {
    for (const char* p : {"q", "k", "v", "o", "ffn_up_gate", "ffn_down"}) {
      qi.specs["layers." + std::to_string(b) + ".proj." + p] = mq::QuantSpec::nvfp4();
    }
  }
  const auto config = [&](std::span<const LoraMod> ms, const char* file) {
    const mi::lora::LoraSpec one{write_lora(file, ms, false, nullptr), 1.0f};
    mi::lora::LoraStack st(std::span<const mi::lora::LoraSpec>(&one, 1), mi::lora::moss_dit_targets());
    mi::lora::SideBranches branches(st);
    return md::MossDit::compute_config(&qi, md::AttentionMode::kExact, md::Int4Route::kAuto, &branches);
  };
  const md::ComputeConfig none = md::MossDit::compute_config(&qi, md::AttentionMode::kExact);
  EXPECT_EQ(none.nvfp4_delta_cols, 0);
  EXPECT_FALSE(none.nvfp4_lora_ffn);
  EXPECT_EQ(none.lora_rank, 0);
  const LoraMod o_only[] = {{"o", kD, kD}}, q_only[] = {{"q", kD, kD}}, down_only[] = {{"ffn_down", kD, kF}};
  const md::ComputeConfig co = config(o_only, "o.safetensors"), cq = config(q_only, "q.safetensors"),
                          cd = config(down_only, "down.safetensors"), cm = config(mods, "all.safetensors");
  EXPECT_EQ(co.nvfp4_delta_cols, kD);
  EXPECT_FALSE(co.nvfp4_lora_ffn);
  EXPECT_EQ(cq.nvfp4_delta_cols, 3 * kD);
  EXPECT_EQ(cd.nvfp4_delta_cols, kD);
  EXPECT_TRUE(cd.nvfp4_lora_ffn);  // ffn_down's t needs the bf16 ffn after the up_gate output
  EXPECT_EQ(cm.nvfp4_delta_cols, 2 * kF);
  EXPECT_TRUE(cm.nvfp4_lora_ffn);
  EXPECT_EQ(cm.lora_rank, 16);  // the QKV site: q's and v's own rank-8 ranges
  const std::int64_t two[] = {17, 9};
  const std::int64_t l = 2 * 48 + 26;
  const std::size_t base_ws = md::MossDit::workspace_bytes(6, 8, two, md::BlockPath::kFused, none, nullptr);
  const std::size_t o_ws = md::MossDit::workspace_bytes(6, 8, two, md::BlockPath::kFused, co, nullptr);
  const std::size_t m_ws = md::MossDit::workspace_bytes(6, 8, two, md::BlockPath::kFused, cm, nullptr);
  EXPECT_GT(o_ws, base_ws);
  EXPECT_LT(o_ws + static_cast<std::size_t>(l * (2 * kF - kD) * 2), m_ws + 1);  // the o-only Delta is kD wide, not 2F
  std::error_code ec;
  fs::remove_all(lora_dir(), ec);
}

// `big` of the columns forward() gives an all-NVFP4 model (nvfp4_big_cols: the [L, 2F] up_gate output), a guard after
// it; the refusals of missing NVFP4 buffers.
TEST_F(GpuDitNvfp4Block, BigBufferAndRefusals) {
  const Shape& sh = kShapes[1];
  const auto xv = synth_act(501, sh.l(), kD);
  const auto want = run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream());
  BlockCall tight(sh.gh, sh.gw, xv, sh.text, mod_, stream());
  const std::int64_t l = tight.l;
  const std::int64_t cols = md::quant_big_cols(tight.hw, sh.text, false, md::FusedRouteSnapshot{}, md::nvfp4_big_cols(true, false));
  EXPECT_EQ(cols, 2 * kF);
  constexpr std::int64_t kGuard = 4096;
  mi::DeviceBuffer big(static_cast<std::size_t>(l * cols + kGuard) * 2);
  const std::vector<std::uint16_t> sentinel(static_cast<std::size_t>(kGuard), 0x7F81);
  MI_CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(big.get()) + l * cols, sentinel.data(), kGuard * 2, cudaMemcpyHostToDevice));
  tight.qa.f.big = Tensor::view(big.get(), DType::kBF16, mi::Device::kCUDA, {l, cols});
  md::run_block_quant(block_->w, tight.qa, tight.run, tight.qkv_gemm(), md::FusedRouteSnapshot{}, md::AttentionMode::kExact, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(download<std::uint16_t>(tight.x->t), want);
  EXPECT_EQ(download<std::uint16_t>(static_cast<std::uint16_t*>(big.get()) + l * cols, kGuard), sentinel);
  // A block whose ffn_down is not NVFP4 materializes the bf16 ffn after the up_gate output: it needs 3F columns.
  EXPECT_EQ(md::nvfp4_big_cols(true, true), md::kOpByOpBigCols);
  EXPECT_EQ(md::nvfp4_big_cols(false, true), 0);
  md::BlockTensors no_down = block_->w;
  no_down.down_nvfp4 = {};
  no_down.down = Tensor::view(big.get(), DType::kBF16, mi::Device::kCUDA, {kD, kF});  // any bf16 [kD, kF]: refused first
  EXPECT_THROW(md::run_block_quant(no_down, tight.qa, tight.run, tight.qkv_gemm(), md::FusedRouteSnapshot{},
                                   md::AttentionMode::kExact, stream()),
               std::invalid_argument);
  // No scale buffer for the NVFP4 inputs.
  BlockCall bare(sh.gh, sh.gw, xv, sh.text, mod_, stream());
  bare.qa.nvfp4_sf = nullptr;
  EXPECT_THROW(md::run_block_quant(block_->w, bare.qa, bare.run, bare.qkv_gemm(), md::FusedRouteSnapshot{},
                                   md::AttentionMode::kExact, stream()),
               std::invalid_argument);
}

// 8-bit attention (q8_fp8, the Turbo default) on the NVFP4 block: the emulation with the same attention kernel (G3's
// gates); the loss against exact attention is reported (these hash activations' outlier channels are harsh on it).
TEST_F(GpuDitNvfp4Block, Q8AttentionMatchesEmulation) {
  if (!md::attention_supported(md::AttentionMode::kQ8Fp8)) GTEST_SKIP() << "no q8_fp8 attention here";
  const Shape& sh = kShapes[1];
  const auto xv = synth_act(601, sh.l(), kD);
  const auto got = run_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream(), md::AttentionMode::kQ8Fp8);
  const Agreement a = agree(reference_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream(), nullptr,
                                            md::AttentionMode::kQ8Fp8),
                            got, xv);
  print("synthetic block, q8_fp8 attention, vs emulation", a);
  expect_g3(a);
  print("synthetic block, q8_fp8 attention, vs exact-attention emulation (lossy, reported)",
        agree(reference_block(block_->w, sh.gh, sh.gw, xv, sh.text, mod_, stream()), got, xv));
}

// ---- Block level: the NVFP4 files ------------------------------------------------------------------------------

TEST(GpuDitNvfp4File, BlocksMatchEmulation) {
  if (!nvfp4_here()) GTEST_SKIP() << "the NVFP4 GEMMs do not run here: " << md::nvfp4_unsupported_reason();
  const std::vector<fs::path> files = nvfp4_files();
  if (files.empty()) GTEST_SKIP() << "no NVFP4 DiT file (MOSS_NVFP4_DIT, or BennyDaBall's under $MOSS_COMMUNITY)";
  mi::runtime::StreamSet streams;
  const cudaStream_t s = streams.compute();
  const std::vector<std::uint16_t> mod = synth_mod();
  const Shape sh = kShapes[0];
  for (const fs::path& file : files) {
    SCOPED_TRACE(file.string());
    const mq::QuantIndex qi = mq::to_model_index(mq::open_quant_model(file, mq::ModelKind::kDiT));
    EXPECT_TRUE(md::has_nvfp4_linears(&qi));
    const double fn = md::nvfp4_gemm_fraction(&qi), f8 = md::int8_gemm_fraction(&qi);
    std::printf("[dit_nvfp4] %s: %s; NVFP4 share of the block GEMMs %.3f, INT8 %.3f\n", file.filename().string().c_str(),
                md::describe_linears(&qi).c_str(), fn, f8);
    EXPECT_NEAR(fn + f8, 1.0, 1e-9);
    // An all-NVFP4 block and one with INT8 keep-list linears, the first of each.
    std::optional<int> all, mixed;
    for (int b = 0; b < md::kNumBlocks; ++b) {
      int nv = 0;
      for (const char* p : {"q", "k", "v", "o", "ffn_up_gate", "ffn_down"}) {
        nv += qi.find_spec("layers." + std::to_string(b) + ".proj." + p)->format == mq::QuantFormat::kNvfp4;
      }
      if (nv == 6 && !all) all = b;
      if (nv > 0 && nv < 6 && !mixed) mixed = b;
    }
    for (const std::optional<int>& blk : {all, mixed}) {
      if (!blk) continue;
      const mi::weights::WeightGroup* g = qi.index.find_group(mi::weights::Component::kTransformer, "block." + std::to_string(*blk));
      ASSERT_NE(g, nullptr);
      mi::runtime::DeviceArena arena(std::size_t{1} << 30);
      mi::weights::ReaderCache readers(qi.index, false);
      mi::runtime::PinnedRing ring(streams.copy());
      const mi::weights::TensorMap m =
          mi::weights::upload_group(readers, md::detail::with_front(*g, md::detail::block_front(*blk, &qi)), arena, ring);
      ring.synchronize();
      const md::BlockTensors w = md::detail::block_views(*blk, m, &qi);
      EXPECT_TRUE(w.nvfp4());
      if (!w.qkv_nvfp4.empty()) {
        EXPECT_FALSE(w.qkv_split);  // the file's fused proj.qkv: one GEMM
        EXPECT_EQ(w.qkv_nvfp4.w.dim(0), 3 * kD);
        EXPECT_EQ(w.qkv_nvfp4.sf.dim(0), 3 * kD);
      }
      const auto xv = synth_act(701 + static_cast<std::uint64_t>(*blk), sh.l(), kD);
      const auto ref = reference_block(w, sh.gh, sh.gw, xv, sh.text, mod, s);
      const auto got = run_block(w, sh.gh, sh.gw, xv, sh.text, mod, s);
      const Agreement a = agree(ref, got, xv);
      const std::string what = file.filename().string() + " block " + std::to_string(*blk) +
                               (blk == all ? " (all NVFP4)" : " (with INT8 linears)") + " vs emulation";
      print(what.c_str(), a);
      expect_g3(a);
    }
  }
}

// ---- Model level -----------------------------------------------------------------------------------------------

struct ModelIo {
  std::unique_ptr<DevTensor> latents, cond, uncond;
  std::vector<std::unique_ptr<DevTensor>> outs;
  std::int64_t h, w;
  ModelIo(std::int64_t h_, std::int64_t w_, std::int64_t s_cond, std::int64_t s_uncond) : h(h_), w(w_) {
    latents = std::make_unique<DevTensor>(DType::kF32, std::initializer_list<std::int64_t>{128, h, w});
    const std::vector<float> noise = tl::randn(static_cast<std::size_t>(128 * h * w), 91, 1.0f);
    latents->buf.upload(noise.data(), noise.size() * 4);
    const auto text = [](std::int64_t len, std::uint32_t seed) {
      auto t = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{len, kD});
      const auto v = tl::to_bf16(tl::randn(static_cast<std::size_t>(len * kD), seed, 1.0f));
      t->buf.upload(v.data(), v.size() * 2);
      return t;
    };
    cond = text(s_cond, 92);
    uncond = text(s_uncond, 93);
    for (int i = 0; i < 3; ++i) {
      outs.push_back(std::make_unique<DevTensor>(DType::kF32, std::initializer_list<std::int64_t>{128, h, w}));
    }
  }
  std::vector<std::vector<float>> forward(md::MossDit& m, std::span<const md::Row> rows, float t, cudaStream_t s) {
    std::vector<Tensor> o;
    for (std::size_t i = 0; i < rows.size(); ++i) o.push_back(outs[i]->t);
    m.forward(latents->t, t, rows, o, s);
    MI_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<std::vector<float>> v;
    for (std::size_t i = 0; i < rows.size(); ++i) v.push_back(download<float>(outs[i]->t));
    return v;
  }
};

class GpuDitNvfp4Model : public ::testing::Test {
 protected:
  static constexpr std::int64_t kGh = 48, kGw = 48, kCond = 77, kUncond = 12;
  static constexpr float kT = 0.62f;
  static void SetUpTestSuite() {
    if (!nvfp4_here() || nvfp4_files().empty()) return;
    file_ = nvfp4_files().front();
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    quant_ = std::make_unique<mq::QuantIndex>(mq::to_model_index(mq::open_quant_model(file_, mq::ModelKind::kDiT)));
    work_ = 0;
    for (const md::AttentionMode a : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
      work_ = std::max(work_, md::MossDit::workspace_bytes(kGh, kGw, lengths(), md::BlockPath::kFusedGraphs,
                                                           md::MossDit::compute_config(quant_.get(), a), nullptr));
    }
    persistent_ = md::MossDit::persistent_bytes(quant_->index);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(persistent_ + work_);
    model_ = std::make_unique<md::MossDit>(*quant_, *arena_, *streams_);
    io_ = std::make_unique<ModelIo>(kGh, kGw, kCond, kUncond);
    std::printf("[dit_nvfp4] %s model: %.3f GiB persistent, %.3f GiB forward workspace, loaded in %.2f s\n",
                file_.filename().string().c_str(), static_cast<double>(persistent_) / kGiB, static_cast<double>(work_) / kGiB,
                model_->load_stats().seconds);
  }
  static void TearDownTestSuite() {
    io_.reset();
    model_.reset();
    arena_.reset();
    quant_.reset();
    streams_.reset();
  }
  static std::span<const std::int64_t> lengths() {
    static const std::int64_t l[] = {kCond, kUncond, kCond};
    return l;
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "the NVFP4 GEMMs do not run here, or no NVFP4 DiT file";
    model_->set_block_path(md::BlockPath::kFused);
    model_->set_attention(md::AttentionMode::kExact);
  }
  cudaStream_t stream() const { return streams_->compute(); }
  std::vector<md::Row> cfg_rows() const {
    md::SkipSet s22;
    s22.set(22);
    return {{io_->cond->t, {}}, {io_->uncond->t, s22}};
  }
  std::vector<md::AttentionMode> modes() const {
    std::vector<md::AttentionMode> m{md::AttentionMode::kExact};
    if (md::attention_supported(md::AttentionMode::kQ8Fp8)) m.push_back(md::AttentionMode::kQ8Fp8);
    return m;
  }

  static fs::path file_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mq::QuantIndex> quant_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<md::MossDit> model_;
  static std::unique_ptr<ModelIo> io_;
  static std::size_t work_, persistent_;
};
fs::path GpuDitNvfp4Model::file_;
std::unique_ptr<mi::runtime::StreamSet> GpuDitNvfp4Model::streams_;
std::unique_ptr<mq::QuantIndex> GpuDitNvfp4Model::quant_;
std::unique_ptr<mi::runtime::DeviceArena> GpuDitNvfp4Model::arena_;
std::unique_ptr<md::MossDit> GpuDitNvfp4Model::model_;
std::unique_ptr<ModelIo> GpuDitNvfp4Model::io_;
std::size_t GpuDitNvfp4Model::work_ = 0, GpuDitNvfp4Model::persistent_ = 0;

TEST_F(GpuDitNvfp4Model, ComputeConfig) {
  const md::ComputeConfig cc = model_->compute_config();
  EXPECT_TRUE(cc.quantized);
  EXPECT_TRUE(cc.nvfp4);
  EXPECT_FALSE(cc.int4);
  EXPECT_FALSE(cc.bf16_up_gate);
  // The workspace holds the NVFP4 scale buffer; a side branch on an NVFP4 ffn_up_gate adds Delta (2F wide) and its
  // scratch.
  md::ComputeConfig lora = cc;
  lora.lora_rank = 16;
  lora.nvfp4_delta_cols = 2 * kF;
  const std::int64_t l = 2 * (kGh * kGw) + kCond + kUncond;
  const std::int64_t two[] = {kCond, kUncond};
  EXPECT_GE(md::MossDit::workspace_bytes(kGh, kGw, two, md::BlockPath::kFused, lora, nullptr),
            md::MossDit::workspace_bytes(kGh, kGw, two, md::BlockPath::kFused, cc, nullptr) +
                static_cast<std::size_t>(l * 2 * kF * 2) + mi::lora::delta_scratch_bytes(l, 16, 2 * kF));
}

TEST_F(GpuDitNvfp4Model, VarlenBatchBitExactPerRow) {
  md::SkipSet s22, s24;
  s22.set(22);
  s24.set(24);
  const md::Row set[] = {{io_->cond->t, {}}, {io_->uncond->t, s22}, {io_->cond->t, s24}};
  for (const md::AttentionMode mode : modes()) {
    model_->set_attention(mode);
    const auto batched = io_->forward(*model_, set, kT, stream());
    for (std::size_t r = 0; r < 3; ++r) {
      const auto alone = io_->forward(*model_, std::span<const md::Row>(&set[r], 1), kT, stream());
      EXPECT_EQ(differing(alone[0], batched[r]), 0u) << md::attention_mode_name(mode) << " row " << r;
    }
  }
}

TEST_F(GpuDitNvfp4Model, GraphsBitIdenticalToEager) {
  const auto rows = cfg_rows();
  for (const md::AttentionMode mode : modes()) {
    model_->set_block_path(md::BlockPath::kFused);
    model_->set_attention(mode);
    const auto eager = io_->forward(*model_, rows, kT, stream());
    model_->set_block_path(md::BlockPath::kFusedGraphs);
    std::vector<std::vector<float>> got;
    for (int i = 0; i < 3; ++i) got = io_->forward(*model_, rows, kT, stream());  // eager, capture, replay
    EXPECT_GT(model_->graph_stats().replays, 0u);
    for (std::size_t r = 0; r < rows.size(); ++r) {
      EXPECT_EQ(differing(got[r], eager[r]), 0u) << md::attention_mode_name(mode) << " row " << r;
    }
  }
  model_->set_block_path(md::BlockPath::kFused);
}

TEST_F(GpuDitNvfp4Model, StreamedBitIdenticalToResident) {
  const auto rows = cfg_rows();
  const auto want = io_->forward(*model_, rows, kT, stream());
  const std::int64_t two[] = {kCond, kUncond};
  const std::size_t work = md::MossDit::workspace_bytes(kGh, kGw, two, md::BlockPath::kFused, model_->compute_config(), nullptr);
  std::optional<mi::weights::ModelIndex> bundle;
  if (fs::is_regular_file(sft_dir() / "moss.mossidx")) bundle.emplace(mi::weights::load_index(sft_dir() / "moss.mossidx", sft_dir()));
  for (const md::StreamSource source : {md::StreamSource::kFile, md::StreamSource::kHostPinned}) {
    md::Residency res;
    res.resident_blocks = 10;
    res.source = source;
    mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(quant_->index, res) + work +
                                   md::MossDit::stream_workspace_bytes(quant_->index, res));
    md::MossDit streamed(*quant_, arena, *streams_, false, nullptr, res);
    const auto got = io_->forward(streamed, rows, kT, stream());
    const md::StreamStats& st = streamed.last_stream_stats();
    std::uint64_t want_bytes = 0, bf16_bytes = 0;
    const auto mask = md::resident_mask(res.resident_blocks);
    for (int b = 0; b < md::kNumBlocks; ++b) {
      if (mask.test(static_cast<std::size_t>(b))) continue;
      const std::string g = "block." + std::to_string(b);
      want_bytes += quant_->index.find_group(mi::weights::Component::kTransformer, g)->total_bytes;
      if (bundle) bf16_bytes += bundle->find_group(mi::weights::Component::kTransformer, g)->total_bytes;
    }
    std::printf("[dit_nvfp4] streamed (%s, 10 of 42 resident): %.2f GB per step (bf16 %.2f GB, ratio %.3f) at %.1f GB/s, "
                "%.0f%% hidden\n", md::stream_source_name(source), static_cast<double>(st.bytes) / 1e9,
                static_cast<double>(bf16_bytes) / 1e9,
                bf16_bytes > 0 ? static_cast<double>(st.bytes) / static_cast<double>(bf16_bytes) : 0.0, st.gbps(), 100 * st.hidden());
    EXPECT_EQ(st.bytes, want_bytes);
    if (bundle) {
      EXPECT_LT(static_cast<double>(st.bytes) / static_cast<double>(bf16_bytes), 0.5);
    }
    for (std::size_t r = 0; r < rows.size(); ++r) {
      EXPECT_EQ(differing(got[r], want[r]), 0u) << md::stream_source_name(source) << " row " << r;
    }
  }
}

// A rank-8 LoRA on every GEMM of every block (q, k, v, o, ffn_up_gate, ffn_down), at a 1024^2 CFG step (8484 rows,
// uncond skipping block 22) on an NVFP4 file: the same bits twice, CUDA graphs == eager, and it changes the output
// (cosine to the plain model reported). The keep-list INT8 sites run the fused INT8 LoRA GEMMs beside the NVFP4 ones.
TEST(GpuDitNvfp4Lora, Model1024) {
  if (!nvfp4_here()) GTEST_SKIP() << "the NVFP4 GEMMs do not run here";
  if (nvfp4_files().empty()) GTEST_SKIP() << "no NVFP4 DiT file";
  constexpr std::int64_t kGrid = 64, kCond = 282, kUncond = 10, kRank = 8;
  const fs::path file = lora_dir() / "model.safetensors";
  {
    const LoraMod mods[] = {{"q", kD, kD}, {"k", kD, kD}, {"v", kD, kD}, {"o", kD, kD}, {"ffn_up_gate", 2 * kF, kD},
                            {"ffn_down", kD, kF}};
    std::vector<tl::StTensor> ts;
    std::uint32_t seed = 1001;
    for (int b = 0; b < md::kNumBlocks; ++b) {
      for (const LoraMod& m : mods) {
        const std::string p = "diffusion_model.layers." + std::to_string(b) + ".proj." + m.name + ".";
        ts.push_back({p + "lora_down.weight", mi::weights::DType::kBF16, {kRank, m.k},
                      tl::randn(static_cast<std::size_t>(kRank * m.k), seed++, 0.02f)});
        ts.push_back({p + "lora_up.weight", mi::weights::DType::kBF16, {m.n, kRank},
                      tl::randn(static_cast<std::size_t>(m.n * kRank), seed++, 0.02f)});
        ts.push_back({p + "alpha", mi::weights::DType::kF32, {}, {static_cast<float>(kRank)}});
      }
    }
    tl::write_st(file, ts);
  }
  const mi::lora::LoraSpec spec{file, 1.0f};
  mi::lora::LoraStack stack(std::span<const mi::lora::LoraSpec>(&spec, 1), mi::lora::moss_dit_targets());
  mi::lora::SideBranches sb(stack);
  const mq::QuantIndex qi = mq::to_model_index(mq::open_quant_model(nvfp4_files().front(), mq::ModelKind::kDiT));
  mi::runtime::StreamSet streams;
  const cudaStream_t s = streams.compute();
  const std::int64_t two[] = {kCond, kUncond};
  std::size_t work = 0;
  for (const md::AttentionMode a : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    const md::ComputeConfig c = md::MossDit::compute_config(&qi, a, md::Int4Route::kAuto, &sb);
    EXPECT_EQ(c.nvfp4_delta_cols, 2 * kF);  // every site has a branch: the widest is up_gate (2F > 3kD)
    work = std::max(work, md::MossDit::workspace_bytes(kGrid, kGrid, two, md::BlockPath::kFusedGraphs, c, nullptr));
  }
  mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(qi.index) + work);
  md::MossDit model(qi, arena, streams);
  ModelIo io(kGrid, kGrid, kCond, kUncond);
  md::SkipSet s22;
  s22.set(22);
  const md::Row rows[] = {{io.cond->t, {}}, {io.uncond->t, s22}};
  const auto cos = [](const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      d += static_cast<double>(a[i]) * b[i];
      aa += static_cast<double>(a[i]) * a[i];
      bb += static_cast<double>(b[i]) * b[i];
    }
    return d / std::sqrt(aa * bb);
  };
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    if (!md::attention_supported(mode)) continue;
    model.set_attention(mode);
    model.set_block_path(md::BlockPath::kFused);
    model.set_lora(nullptr);
    const auto plain = io.forward(model, rows, 0.5f, s);
    model.set_lora(&sb);
    const auto first = io.forward(model, rows, 0.5f, s);
    const auto second = io.forward(model, rows, 0.5f, s);
    model.set_block_path(md::BlockPath::kFusedGraphs);
    std::vector<std::vector<float>> graphs;
    for (int i = 0; i < 3; ++i) graphs = io.forward(model, rows, 0.5f, s);  // eager, capture, replay
    EXPECT_GT(model.graph_stats().replays, 0u);
    for (std::size_t r = 0; r < 2; ++r) {
      EXPECT_EQ(differing(first[r], second[r]), 0u) << md::attention_mode_name(mode) << " row " << r;
      EXPECT_EQ(differing(graphs[r], first[r]), 0u) << md::attention_mode_name(mode) << " graphs row " << r;
      EXPECT_GT(differing(first[r], plain[r]), 0u);
      const double c = cos(first[r], plain[r]);
      std::printf("[dit_nvfp4] 1024^2 step, %s attention, rank-8 LoRA on every GEMM: row %zu velocity cos to the plain "
                  "model %.6f\n", md::attention_mode_name(mode), r, c);
      EXPECT_GT(c, 0.9);
    }
  }
  std::error_code ec;
  fs::remove_all(lora_dir(), ec);
}

// ---- Speed -----------------------------------------------------------------------------------------------------

// The planner's NVFP4 / INT8 GEMM rate at the block level (1024^2, CFG, 282 + 10 tokens: 8484 rows), every site with
// its quantizer: an all-NVFP4 block of the NVFP4 file against block 20 of Comfy-Org's INT8 file, each minus the block's
// GEMM-free part (QK-norm + RoPE and exact attention, timed alone); then one CFG step of each model at 1024^2 (exact
// and q8_fp8 attention). Medians of 20 calls after warm-up; timings only.
TEST(GpuDitNvfp4Speed, BlockRateAndStep1024) {
  if (!nvfp4_here()) GTEST_SKIP() << "the NVFP4 GEMMs do not run here";
  const std::vector<fs::path> files = nvfp4_files();
  if (files.empty() || !fs::is_regular_file(int8_file())) GTEST_SKIP() << "NVFP4 or INT8 DiT file absent";
  mi::runtime::StreamSet streams;
  const cudaStream_t s = streams.compute();
  constexpr std::int64_t kGrid = 64, kCond = 282, kUncond = 10;
  const std::int64_t text[] = {kCond, kUncond};
  const mq::QuantIndex qn = mq::to_model_index(mq::open_quant_model(files.front(), mq::ModelKind::kDiT));
  const mq::QuantIndex q8 = mq::to_model_index(mq::open_quant_model(int8_file(), mq::ModelKind::kDiT));
  int blk = -1;
  for (int b = 20; b < md::kNumBlocks && blk < 0; ++b) {
    bool all = true;
    for (const char* p : {"q", "o", "ffn_up_gate", "ffn_down"}) {
      all = all && qn.find_spec("layers." + std::to_string(b) + ".proj." + p)->format == mq::QuantFormat::kNvfp4;
    }
    if (all) blk = b;
  }
  if (blk < 0) GTEST_SKIP() << "no all-NVFP4 block from 20 on in " << files.front();
  double rate = 0;
  {
  mi::runtime::DeviceArena arena(std::size_t{2} << 30);
  const auto load = [&](const mq::QuantIndex& qi, int b) {
    const mi::weights::WeightGroup* g = qi.index.find_group(mi::weights::Component::kTransformer, "block." + std::to_string(b));
    mi::weights::ReaderCache readers(qi.index, false);
    mi::runtime::PinnedRing ring(streams.copy());
    const mi::weights::TensorMap m = mi::weights::upload_group(readers, md::detail::with_front(*g, md::detail::block_front(b, &qi)), arena, ring);
    ring.synchronize();
    return md::detail::block_views(b, m, &qi);
  };
  const md::BlockTensors wn = load(qn, blk), w8 = load(q8, 20);
  const std::int64_t l = 2 * kGrid * kGrid + kCond + kUncond;
  const auto xv = synth_act(801, l, kD);
  const std::vector<std::uint16_t> mod = synth_mod();
  BlockCall r(kGrid, kGrid, xv, text, mod, s);
  const md::FusedRouteSnapshot route;
  cudaEvent_t e0, e1;
  MI_CUDA_CHECK(cudaEventCreate(&e0));
  MI_CUDA_CHECK(cudaEventCreate(&e1));
  const auto median_ms = [&](const auto& fn) {
    for (int i = 0; i < 5; ++i) fn();
    std::vector<float> t;
    for (int i = 0; i < 20; ++i) {
      MI_CUDA_CHECK(cudaEventRecord(e0, s));
      fn();
      MI_CUDA_CHECK(cudaEventRecord(e1, s));
      MI_CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0;
      MI_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      t.push_back(ms);
      r.x->buf.upload(xv.data(), xv.size() * 2);  // the residual grows every call
    }
    std::sort(t.begin(), t.end());
    return static_cast<double>(t[t.size() / 2]);
  };
  const double t8 = median_ms([&] { md::run_block_quant(w8, r.qa, r.run, r.qkv_gemm(), route, md::AttentionMode::kExact, s); });
  const double tn = median_ms([&] { md::run_block_quant(wn, r.qa, r.run, r.qkv_gemm(), route, md::AttentionMode::kExact, s); });
  const Tensor qkv = Tensor::view(r.big->t.data, DType::kBF16, mi::Device::kCUDA, {l, 3 * kD});
  const auto heads = [&](std::int64_t col, std::int64_t ld, void* base) {
    Tensor t = Tensor::view(static_cast<std::uint16_t*>(base) + col, DType::kBF16, mi::Device::kCUDA, {l, kH, kHd});
    t.stride[0] = ld;
    return t;
  };
  void* attn = static_cast<std::uint16_t*>(r.big->t.data) + l * 3 * kD;
  const double tc = median_ms([&] {
    mi::kernels::fused::qk_norm_rope_moss(qkv, static_cast<int>(kH), wn.q_norm, wn.k_norm, r.rope->t, s);
    mi::attention::flash_fwd(heads(0, 3 * kD, qkv.data), heads(kD, 3 * kD, qkv.data), heads(2 * kD, 3 * kD, qkv.data),
                             heads(0, kD, attn), r.cu->t, r.cu->t, r.run.max_seqlen, false, kScale, r.lse.get(), r.lse.size(), s);
  });
  rate = (t8 - tc) / (tn - tc);
  std::printf("[dit_nvfp4 speed] block (8484 rows, exact attention): INT8 %.3f ms, NVFP4 %.3f ms (block %d), GEMM-free part "
              "%.3f ms: NVFP4 / INT8 GEMM rate %.3f (kNvfp4SpeedVsInt8)\n", t8, tn, blk, tc, rate);
  RecordProperty("nvfp4_speed_vs_int8", std::to_string(rate));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  }

  // One CFG step per model at 1024^2 (uncond skipping block 22), every block resident, each model alone.
  ModelIo io(kGrid, kGrid, kCond, kUncond);
  md::SkipSet s22;
  s22.set(22);
  const md::Row rows[] = {{io.cond->t, {}}, {io.uncond->t, s22}};
  const std::size_t free = mi::runtime::DeviceArena::free_device_bytes() - (std::size_t{512} << 20);
  const auto step = [&](const mq::QuantIndex& qi, md::AttentionMode mode) {
    const std::size_t work = md::MossDit::workspace_bytes(kGrid, kGrid, text, md::BlockPath::kFused,
                                                          md::MossDit::compute_config(&qi, mode), nullptr);
    if (md::MossDit::persistent_bytes(qi.index) + work > free) return -1.0;
    mi::runtime::DeviceArena a(md::MossDit::persistent_bytes(qi.index) + work);
    md::MossDit m(qi, a, streams);
    m.set_attention(mode);
    std::vector<double> t;
    for (int i = 0; i < 9; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      io.forward(m, rows, 0.5f, s);
      if (i >= 2) t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  };
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    if (!md::attention_supported(mode)) continue;
    const double s8 = step(q8, mode), sn = step(qn, mode);
    std::printf("[dit_nvfp4 speed] 1024^2 CFG step, %s attention: INT8 %.4f s, NVFP4 (%s) %.4f s (%.3fx)\n",
                md::attention_mode_name(mode), s8, files.front().filename().string().c_str(), sn, s8 / sn);
    RecordProperty(std::string("step_s_int8_") + md::attention_mode_name(mode), std::to_string(s8));
    RecordProperty(std::string("step_s_nvfp4_") + md::attention_mode_name(mode), std::to_string(sn));
  }
  EXPECT_GT(rate, 0.0);
}

}  // namespace
