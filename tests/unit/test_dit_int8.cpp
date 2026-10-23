// P6 (lane A): the DiT on the High-tier compute path (models/moss_dit/dit_block_quant, MossDit from a quantized index).
//
// Block level, on block 5 of Comfy-Org's moss_image_sft_int8_convrot file (and of the official SFT bf16 transformer):
//   - every linear INT8, two varlen rows: against the reference _Block with comfy-kitchen's int8_linear (ConvRot,
//     CUDA epilogue order) and SDPA, from tests/data/moss_dit_int8 (tools/reference/gen_dit_int8_vectors.py);
//   - a mixed block (q bf16, k/v INT8: split QKV; o bf16; up_gate INT8; down bf16) against the same reference;
//   - every linear bf16 through run_block_quant == run_block_fused bit for bit (the dispatcher adds nothing);
//   - 8-bit attention: accuracy against the reference (lossy, reported);
//   - a linear stored in a spec the GEMMs do not implement is refused naming it.
// Model level, on the golden 1024^2 step-0 case (g_fridge_1024 text states and noise) with the INT8 file:
//   - the bf16 globals and text refiner of the file equal the official SFT bundle's bit for bit;
//   - step-0 velocity cosine vs the bf16 engine (SFT bundle), exact and 8-bit attention (reported);
//   - one varlen batch (cond, uncond skipping 22, SLG row skipping 24) == each row alone, bit for bit;
//   - streamed INT8 blocks (file and pinned host sources) == resident, bit for bit; INT8 blocks move ~half the bytes;
//   - CUDA graphs == eager, bit for bit (exact and 8-bit attention);
//   - cancel between blocks leaves the model usable and the next forward exact.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/rope.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/dit_block_quant.h"
#include "models/moss_dit/dit_weights.h"
#include "models/moss_dit/moss_dit.h"
#include "reference_gpu.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
namespace md = mi::moss_dit;
using mi::DType;
using mi::Tensor;

constexpr std::int64_t kDim = md::kDim;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr int kBlock = 5;
constexpr std::int64_t kGridH = 6, kGridW = 8;
constexpr std::int64_t kText[] = {17, 9};
constexpr md::AttentionMode kQ8Modes[] = {md::AttentionMode::kQ8Fp8, md::AttentionMode::kQ8Fp16, md::AttentionMode::kQ8Fp8F32Acc,
                                          md::AttentionMode::kQ8Fp16F32Acc};

fs::path int8_file() {
  const char* env = std::getenv("MOSS_INT8_DIT");
  return env != nullptr && *env != '\0' ? fs::path(env)
                                        : fs::path("/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors");
}
fs::path sft_dir() {
  const char* env = std::getenv("MOSS_WEIGHTS");
  return (env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/MOSS-Image")) / "sft";
}
fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "moss_dit_int8"; }
fs::path golden_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "golden"; }
bool have_files() { return fs::is_regular_file(int8_file()) && fs::is_regular_file(sft_dir() / "moss.mossidx"); }

float from_bf16(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16); }
std::uint16_t to_bf16(float v) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(v);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<std::uint16_t>(u >> 16);
}

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
std::vector<T> download(const Tensor& t) {
  std::vector<T> h(static_cast<std::size_t>(t.numel()));
  MI_CUDA_CHECK(cudaMemcpy(h.data(), t.data, h.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

std::vector<std::uint16_t> load_u16(const std::string& name) { return mi::npy::load(data_dir() / (name + ".npy")).to_vector<std::uint16_t>(); }

// bf16 agreement of [rows, kDim] outputs: bit-exact fraction, the ULP distance on the ordered bf16 line (max), the
// fraction within 1 bf16 ULP of the element's scale (max(|ref|, the row's RMS): the residual stream mixes channels of
// very different size, so a value that cancels to near zero is judged at its row's scale), and the cosine.
struct Agreement {
  double exact = 0, within1 = 0, cosine = 0, max_scaled = 0;
  int max_ulp = 0;
};
int ordered(std::uint16_t b) { return (b & 0x8000) ? -static_cast<int>(b & 0x7fff) : static_cast<int>(b); }
double bf16_ulp(double v) {
  int e = 0;
  std::frexp(std::max(std::abs(v), 1e-30), &e);
  return std::ldexp(1.0, e - 8);  // bf16: 8 significant bits
}
Agreement agree(const std::vector<std::uint16_t>& ref, const std::vector<std::uint16_t>& got) {
  Agreement a;
  double dot = 0, rr = 0, gg = 0;
  std::size_t exact = 0, w1 = 0;
  for (std::size_t row = 0; row * kDim < ref.size(); ++row) {
    double ms = 0;
    for (std::int64_t c = 0; c < kDim; ++c) ms += std::pow(from_bf16(ref[row * kDim + c]), 2);
    const double rms = std::sqrt(ms / kDim);
    for (std::int64_t c = 0; c < kDim; ++c) {
      const std::size_t i = row * kDim + c;
      a.max_ulp = std::max(a.max_ulp, std::abs(ordered(ref[i]) - ordered(got[i])));
      exact += ref[i] == got[i];
      const double r = from_bf16(ref[i]), g = from_bf16(got[i]);
      const double scaled = std::abs(g - r) / bf16_ulp(std::max(std::abs(r), rms));
      a.max_scaled = std::max(a.max_scaled, scaled);
      w1 += scaled <= 1.0;
      dot += r * g;
      rr += r * r;
      gg += g * g;
    }
  }
  a.exact = static_cast<double>(exact) / static_cast<double>(ref.size());
  a.within1 = static_cast<double>(w1) / static_cast<double>(ref.size());
  a.cosine = dot / std::sqrt(rr * gg);
  return a;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, aa = 0, bb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  return dot / std::sqrt(aa * bb);
}

std::size_t differing(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::max(a.size(), b.size());
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
  return n;
}

// ---- Block level ---------------------------------------------------------------------------------------------

class DitInt8Block : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!have_files() || !fs::is_regular_file(data_dir() / "all_int8_y.npy")) return;
    quant_ = std::make_unique<mi::quant::QuantIndex>(
        mi::quant::to_model_index(mi::quant::open_quant_model(int8_file(), mi::quant::ModelKind::kDiT)));
    bundle_ = std::make_unique<mi::weights::ModelIndex>(mi::weights::load_index(sft_dir() / "moss.mossidx", sft_dir()));
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    arena_ = std::make_unique<mi::runtime::DeviceArena>(std::size_t{1} << 30);
    const auto upload = [&](const mi::weights::ModelIndex& index, const mi::quant::QuantIndex* q) {
      const mi::weights::WeightGroup* g = index.find_group(mi::weights::Component::kTransformer, "block." + std::to_string(kBlock));
      mi::weights::ReaderCache readers(index, false);
      mi::runtime::PinnedRing ring(streams_->copy());
      const mi::weights::TensorMap m =
          mi::weights::upload_group(readers, md::detail::with_front(*g, md::detail::block_front(kBlock, q)), *arena_, ring);
      ring.synchronize();
      return m;
    };
    int8_map_ = upload(quant_->index, quant_.get());
    bf16_map_ = upload(*bundle_, nullptr);
    int8_ = md::detail::block_views(kBlock, int8_map_, quant_.get());
    bf16_ = md::detail::block_views(kBlock, bf16_map_, nullptr);
    const md::FusedCheck check = md::check_fused_gemms(streams_->compute());
    if (check.ok) fused_route_.validated = check.validated;
  }
  static void TearDownTestSuite() {
    int8_map_.clear();
    bf16_map_.clear();
    arena_.reset();
    streams_.reset();
    bundle_.reset();
    quant_.reset();
  }
  void SetUp() override {
    if (!quant_) GTEST_SKIP() << "INT8 DiT file, SFT bundle or tests/data/moss_dit_int8 absent";
  }
  cudaStream_t stream() const { return streams_->compute(); }

  // Buffers of one block call over rows of hw + text[r] tokens (x from `x`), as MossDit::forward lays them out.
  struct Run {
    std::int64_t l = 0;
    std::unique_ptr<DevTensor> x, y, big, rope, mod, scale, cu;
    mi::DeviceBuffer lse, q8, gemm_ws;
    md::BlockRun run;
    md::QuantBlockBuffers qa;
    mi::kernels::Gemm gemm;
    Run(const std::vector<std::uint16_t>& xv, std::span<const std::int64_t> text, cudaStream_t s) {
      const std::int64_t hw = kGridH * kGridW;
      std::vector<std::int32_t> cu_h{0};
      int max_len = 0;
      for (const std::int64_t t : text) {
        cu_h.push_back(cu_h.back() + static_cast<std::int32_t>(hw + t));
        max_len = std::max(max_len, static_cast<int>(hw + t));
      }
      l = cu_h.back();
      x = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, kDim});
      y = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, kDim});
      big = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, md::kOpByOpBigCols});
      rope = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, 128});
      mod = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{6 * kDim});
      scale = std::make_unique<DevTensor>(DType::kF32, std::initializer_list<std::int64_t>{l});
      cu = std::make_unique<DevTensor>(DType::kI32, std::initializer_list<std::int64_t>{static_cast<std::int64_t>(cu_h.size())});
      EXPECT_EQ(xv.size(), static_cast<std::size_t>(l * kDim));
      x->buf.upload(xv.data(), xv.size() * 2);
      const auto mv = load_u16("mod");
      mod->buf.upload(mv.data(), mv.size() * 2);
      cu->buf.upload(cu_h.data(), cu_h.size() * 4);
      for (std::size_t r = 0; r < text.size(); ++r) {
        mi::kernels::moss_rope_table(rope->t.slice0(cu_h[r], cu_h[r + 1]), static_cast<int>(kGridH), static_cast<int>(kGridW),
                                     static_cast<int>(text[r]), s);
      }
      lse = mi::DeviceBuffer(mi::attention::flash_fwd_workspace_bytes(l, md::kHeads));
      std::size_t q8_bytes = 0;
      for (const md::AttentionMode m : kQ8Modes) q8_bytes = std::max(q8_bytes, md::q8_attention_workspace_bytes(hw, text, m));
      q8 = mi::DeviceBuffer(q8_bytes);
      gemm_ws = mi::DeviceBuffer(mi::kernels::Gemm::kWorkspaceBytes);
      run = {0, l, cu->t, max_len};
      qa.f = {x->t, y->t, big->t, rope->t, mod->t, lse.get(), lse.size()};
      qa.scale = scale->t;
      qa.attn_workspace = q8.get();
      qa.attn_workspace_bytes = q8_bytes;
    }
    md::QkvGemm qkv_gemm() { return {&gemm, gemm_ws.get(), gemm_ws.size()}; }
    void reset(const std::vector<std::uint16_t>& xv) { x->buf.upload(xv.data(), xv.size() * 2); }
  };

  static std::unique_ptr<mi::quant::QuantIndex> quant_;
  static std::unique_ptr<mi::weights::ModelIndex> bundle_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static mi::weights::TensorMap int8_map_, bf16_map_;
  static md::BlockTensors int8_, bf16_;
  static md::FusedRouteSnapshot fused_route_;
};

std::unique_ptr<mi::quant::QuantIndex> DitInt8Block::quant_;
std::unique_ptr<mi::weights::ModelIndex> DitInt8Block::bundle_;
std::unique_ptr<mi::runtime::StreamSet> DitInt8Block::streams_;
std::unique_ptr<mi::runtime::DeviceArena> DitInt8Block::arena_;
mi::weights::TensorMap DitInt8Block::int8_map_, DitInt8Block::bf16_map_;
md::BlockTensors DitInt8Block::int8_, DitInt8Block::bf16_;
md::FusedRouteSnapshot DitInt8Block::fused_route_;

TEST_F(DitInt8Block, ViewsOfTheComfyOrgFile) {
  EXPECT_TRUE(int8_.quantized());
  EXPECT_FALSE(bf16_.quantized());
  EXPECT_FALSE(int8_.qkv_split);
  EXPECT_EQ(int8_.qkv8.w.dim(0), 3 * kDim);  // fused proj.qkv carved into q/k/v slices, contiguous again on the device
  EXPECT_EQ(int8_.qkv8.scale.numel(), 3 * kDim);
  EXPECT_EQ(int8_.up_gate8.w.dim(0), 2 * md::kFfnDim);
  EXPECT_EQ(int8_.down8.w.dim(1), md::kFfnDim);
  EXPECT_EQ(int8_.qkv.data, nullptr);
  EXPECT_EQ(md::describe_linears(quant_.get()), "int8_convrot256");
  EXPECT_TRUE(md::has_quantized_linears(quant_.get()));
  // An INT8 block is about half a bf16 one (weights + per-row scales).
  const double ratio = static_cast<double>(md::BlockWeights::block_device_bytes(quant_->index)) /
                       static_cast<double>(md::BlockWeights::block_device_bytes(*bundle_));
  std::printf("[dit_int8] block bytes: INT8 %.1f MiB, bf16 %.1f MiB (ratio %.4f)\n",
              static_cast<double>(md::BlockWeights::block_device_bytes(quant_->index)) / (1 << 20),
              static_cast<double>(md::BlockWeights::block_device_bytes(*bundle_)) / (1 << 20), ratio);
  EXPECT_LT(ratio, 0.51);
}

TEST_F(DitInt8Block, UnsupportedSpecIsRefusedNamingTheLinear) {
  mi::quant::QuantIndex other = *quant_;
  other.specs["layers.5.proj.o"] = mi::quant::QuantSpec::int8_plain();
  try {
    md::detail::block_views(kBlock, int8_map_, &other);
    ADD_FAILURE() << "an int8_plain linear was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("layers.5.proj.o"), std::string::npos) << e.what();
  }
  // A spec that disagrees with the stored bytes (bf16 claimed for an I8 weight) fails quant::view's check.
  other = *quant_;
  other.specs["layers.5.proj.ffn_down"] = mi::quant::QuantSpec::bf16();
  EXPECT_THROW(md::detail::block_views(kBlock, int8_map_, &other), std::runtime_error);
}

TEST_F(DitInt8Block, AllInt8MatchesQuantizedReference) {
  const auto xv = load_u16("all_int8_x");
  const auto ref = load_u16("all_int8_y");
  Run r(xv, kText, stream());
  md::run_block_quant(int8_, r.qa, r.run, r.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const Agreement a = agree(ref, download<std::uint16_t>(r.x->t));
  std::printf("[dit_int8] all-INT8 block vs reference (exact attention): %.4f%% bit-exact, %.4f%% within 1 ULP of scale "
              "(max %.2f), max %d raw ULP, "
              "cos %.9f\n", 100 * a.exact, 100 * a.within1, a.max_scaled, a.max_ulp, a.cosine);
  // Not bit-exact everywhere: attention (FA2 here, SDPA in the reference) differs in last bits, and the o-proj and
  // ffn_down inputs are re-quantized after it, where a 1-ULP input difference can move an INT8 value by one step
  // (absmax / 127). The mixed case, whose o and down are bf16, stays within 1 ULP on 99.9 %.
  // The reference was recorded on sm_89. On sm_120 the attention's last bits differ (tensor-core rounding), and the
  // re-quantized o / ffn_down inputs amplify that: 95.9 % bit-exact, 98.6 % within 1 ULP, cos 0.999998 there.
  if (mi::testing::on_reference_gpu()) {
    EXPECT_GE(a.cosine, 0.99999);
    EXPECT_GE(a.exact, 0.98);
    EXPECT_GE(a.within1, 0.99);
  } else {
    EXPECT_GE(a.cosine, 0.99995);
  }

  // 8-bit attention on the same inputs: lossy, reported against the same (exact-attention) reference.
  for (const md::AttentionMode mode : kQ8Modes) {
    if (!md::attention_supported(mode)) continue;
    r.reset(xv);
    md::run_block_quant(int8_, r.qa, r.run, r.qkv_gemm(), fused_route_, mode, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const Agreement q = agree(ref, download<std::uint16_t>(r.x->t));
    std::printf("[dit_int8] all-INT8 block + %s attention vs reference: %.4f%% bit-exact, %.4f%% within 1 ULP of scale "
                "(max %.2f), cos %.9f\n", md::attention_mode_name(mode), 100 * q.exact, 100 * q.within1, q.max_scaled, q.cosine);
    EXPECT_GE(q.cosine, 0.999) << md::attention_mode_name(mode);
  }
}

// The all-INT8 block with `big` sized exactly as forward() sizes it for an INT8 file (kFusedBigCols per token, the
// ffn_down input in its last third): the same bits as with the wide buffer, and nothing written past it.
TEST_F(DitInt8Block, TightFusedBigBuffer) {
  const auto xv = load_u16("all_int8_x");
  Run wide(xv, kText, stream()), tight(xv, kText, stream());
  const std::int64_t l = tight.l;
  constexpr std::int64_t kGuard = 4096;  // bf16 elements after the buffer
  const std::size_t elems = static_cast<std::size_t>(l * md::kFusedBigCols + kGuard);
  mi::DeviceBuffer big(elems * 2);
  const std::vector<std::uint16_t> sentinel(static_cast<std::size_t>(kGuard), 0x7F81);
  MI_CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(big.get()) + l * md::kFusedBigCols, sentinel.data(), kGuard * 2,
                           cudaMemcpyHostToDevice));
  tight.qa.f.big = Tensor::view(big.get(), DType::kBF16, mi::Device::kCUDA, {l, md::kFusedBigCols});
  ASSERT_EQ(md::quant_big_cols(kGridH * kGridW, kText, false, md::FusedRouteSnapshot{}), md::kFusedBigCols);
  md::run_block_quant(int8_, wide.qa, wide.run, wide.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream());
  md::run_block_quant(int8_, tight.qa, tight.run, tight.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(download<std::uint16_t>(tight.x->t), download<std::uint16_t>(wide.x->t));
  std::vector<std::uint16_t> after(static_cast<std::size_t>(kGuard));
  MI_CUDA_CHECK(cudaMemcpy(after.data(), static_cast<std::uint16_t*>(big.get()) + l * md::kFusedBigCols, kGuard * 2,
                           cudaMemcpyDeviceToHost));
  EXPECT_EQ(after, sentinel);
}

TEST_F(DitInt8Block, MixedBlockMatchesReference) {
  md::BlockTensors w = int8_;
  // q bf16, k and v INT8 (split QKV), o bf16, up_gate INT8, down bf16 (the generator's mixed case).
  w.qkv8 = {};
  w.qkv_split = true;
  const auto rows = [](const Tensor& t, std::int64_t r0, std::int64_t r1) { return t.slice0(r0, r1); };
  w.qkv_part[0] = rows(bf16_.qkv, 0, kDim);
  for (int i = 1; i < 3; ++i) w.qkv8_part[i] = {rows(int8_.qkv8.w, i * kDim, (i + 1) * kDim), rows(int8_.qkv8.scale, i * kDim, (i + 1) * kDim)};
  w.o8 = {};
  w.o = bf16_.o;
  w.down8 = {};
  w.down = bf16_.down;
  ASSERT_TRUE(w.quantized());
  const auto xv = load_u16("mixed_x");
  const auto ref = load_u16("mixed_y");
  Run r(xv, std::span<const std::int64_t>(kText, 1), stream());
  md::run_block_quant(w, r.qa, r.run, r.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  const Agreement a = agree(ref, download<std::uint16_t>(r.x->t));
  std::printf("[dit_int8] mixed block vs reference: %.4f%% bit-exact, %.4f%% within 1 ULP of scale (max %.2f), max %d raw "
              "ULP, cos %.9f\n", 100 * a.exact, 100 * a.within1, a.max_scaled, a.max_ulp, a.cosine);
  // Off sm_89 the bf16 linears also run other cuBLASLt algorithms than torch did (sm_120: 91.4 % within 1 ULP, cos 0.99999).
  if (mi::testing::on_reference_gpu()) {
    EXPECT_GE(a.cosine, 0.99999);
    EXPECT_GE(a.within1, 0.999);
  } else {
    EXPECT_GE(a.cosine, 0.99995);
  }
}

TEST_F(DitInt8Block, AllBf16ThroughQuantPathEqualsFusedBlock) {
  const auto xv = load_u16("all_int8_x");
  for (const bool fused : {false, true}) {
    const md::FusedRouteSnapshot route = fused ? fused_route_ : md::FusedRouteSnapshot{};
    Run a(xv, kText, stream()), b(xv, kText, stream());
    md::run_block_fused(bf16_, a.qa.f, a.run, a.qkv_gemm(), route, stream());
    md::run_block_quant(bf16_, b.qa, b.run, b.qkv_gemm(), route, md::AttentionMode::kExact, stream());
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    const auto va = download<std::uint16_t>(a.x->t), vb = download<std::uint16_t>(b.x->t);
    EXPECT_EQ(std::memcmp(va.data(), vb.data(), va.size() * 2), 0) << (fused ? "fused route" : "op-by-op route");
  }
}

// ---- Model level ---------------------------------------------------------------------------------------------

struct GoldenCase {
  std::vector<float> noise, te_cond, te_uncond;
  std::int64_t h = 0, w = 0, s_cond = 0, s_uncond = 0;
  float timestep = 0;
  md::SkipSet uncond_skip;
};

GoldenCase load_golden(const std::string& id) {
  GoldenCase c;
  const fs::path g = golden_dir();
  const auto noise = mi::npy::load(g / id / "initial_noise.npy");
  c.h = noise.shape[2];
  c.w = noise.shape[3];
  c.noise = noise.to_vector<float>();
  c.timestep = static_cast<float>(mi::npy::load(g / id / "timestep0.npy").to_vector<double>().at(0));
  const auto tc = mi::npy::load(g / (id + ".cond.te_last_hidden.npy"));
  const auto tu = mi::npy::load(g / (id + ".uncond.te_last_hidden.npy"));
  c.s_cond = tc.shape[0];
  c.s_uncond = tu.shape[0];
  c.te_cond = tc.to_vector<float>();
  c.te_uncond = tu.to_vector<float>();
  std::ifstream in(g / id / "dit_step0_uncond.skip.json");
  std::stringstream ss;
  ss << in.rdbuf();
  const mi::json::Value doc = mi::json::parse(ss.str());
  for (const auto& v : doc.as_array()) c.uncond_skip.set(static_cast<std::size_t>(v.as_int()));
  return c;
}

// Refined text, the latents and velocity outputs of one model on a golden case.
struct ModelIo {
  std::unique_ptr<DevTensor> latents, cond, uncond;
  std::vector<std::unique_ptr<DevTensor>> outs;
  ModelIo(md::MossDit& m, const GoldenCase& c, cudaStream_t s) {
    latents = std::make_unique<DevTensor>(DType::kF32, std::initializer_list<std::int64_t>{128, c.h, c.w});
    latents->buf.upload(c.noise.data(), c.noise.size() * 4);
    const auto refine = [&](const std::vector<float>& te, std::int64_t len) {
      std::vector<std::uint16_t> b(te.size());
      for (std::size_t i = 0; i < te.size(); ++i) b[i] = to_bf16(te[i]);
      DevTensor in(DType::kBF16, {len, md::kTextDim});
      in.buf.upload(b.data(), b.size() * 2);
      auto out = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{len, kDim});
      m.encode_text(in.t, out->t, s);
      MI_CUDA_CHECK(cudaStreamSynchronize(s));
      return out;
    };
    cond = refine(c.te_cond, c.s_cond);
    uncond = refine(c.te_uncond, c.s_uncond);
    for (int i = 0; i < 3; ++i) {
      outs.push_back(std::make_unique<DevTensor>(DType::kF32, std::initializer_list<std::int64_t>{128, c.h, c.w}));
    }
  }
  std::vector<std::vector<float>> forward(md::MossDit& m, std::span<const md::Row> rows, float t, cudaStream_t s,
                                          const md::BlockHook& hook = {}) {
    std::vector<Tensor> o;
    for (std::size_t i = 0; i < rows.size(); ++i) o.push_back(outs[i]->t);
    m.forward(latents->t, t, rows, o, s, hook);
    MI_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<std::vector<float>> v;
    for (std::size_t i = 0; i < rows.size(); ++i) v.push_back(download<float>(outs[i]->t));
    return v;
  }
};

// Residency that fits `free` device bytes with `work` stack bytes.
md::Residency fitting(const mi::weights::ModelIndex& index, std::size_t work, std::size_t free) {
  md::Residency r;
  while (r.resident_blocks > 0 &&
         md::MossDit::persistent_bytes(index, r) + work + md::MossDit::stream_workspace_bytes(index, r) > free) {
    --r.resident_blocks;
  }
  return r;
}

class DitInt8Model : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!have_files() || !fs::is_directory(golden_dir() / "g_fridge_1024")) return;
    case_ = std::make_unique<GoldenCase>(load_golden("g_fridge_1024"));
    const GoldenCase& c = *case_;
    streams_ = std::make_unique<mi::runtime::StreamSet>();
    const cudaStream_t s = streams_->compute();
    const std::int64_t lengths[] = {c.s_cond, c.s_uncond, c.s_cond};
    const std::size_t free = mi::runtime::DeviceArena::free_device_bytes() - (std::size_t{512} << 20);
    {
      // bf16 engine on the official SFT weights (streamed as far as needed: bit-identical, test_dit_stream).
      const mi::weights::ModelIndex bundle = mi::weights::load_index(sft_dir() / "moss.mossidx", sft_dir());
      const std::size_t work = std::max(md::MossDit::workspace_bytes(c.h, c.w, lengths),
                                        md::MossDit::encode_text_workspace_bytes(std::max(c.s_cond, c.s_uncond)));
      const md::Residency res = fitting(bundle, work, free);
      mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(bundle, res) + work +
                                     md::MossDit::stream_workspace_bytes(bundle, res));
      md::MossDit bf16(bundle, arena, *streams_, false, nullptr, res);
      ModelIo io(bf16, c, s);
      const md::Row rows[] = {{io.cond->t, {}}, {io.uncond->t, c.uncond_skip}};
      bf16_v_ = io.forward(bf16, rows, c.timestep, s);
      bf16_refined_ = {download<std::uint16_t>(io.cond->t), download<std::uint16_t>(io.uncond->t)};
      std::printf("[dit_int8] bf16 SFT reference step: %d of 42 blocks resident\n", res.resident_blocks);
    }
    {
      // The same weights through a quantized index (every linear bf16): the path a single-file bf16 DiT takes.
      const mi::quant::QuantIndex qi = mi::quant::to_model_index(mi::quant::open_quant_model(sft_dir(), mi::quant::ModelKind::kDiT));
      bf16_file_linears_ = md::describe_linears(&qi);
      const std::size_t work = std::max(md::MossDit::workspace_bytes(c.h, c.w, lengths),
                                        md::MossDit::encode_text_workspace_bytes(std::max(c.s_cond, c.s_uncond)));
      const md::Residency res = fitting(qi.index, work, free);
      mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(qi.index, res) + work +
                                     md::MossDit::stream_workspace_bytes(qi.index, res));
      md::MossDit bf16(qi, arena, *streams_, false, nullptr, res);
      bf16_file_config_ = bf16.compute_config();
      ModelIo io(bf16, c, s);
      const md::Row rows[] = {{io.cond->t, {}}, {io.uncond->t, c.uncond_skip}};
      bf16_file_v_ = io.forward(bf16, rows, c.timestep, s);
    }
    quant_ = std::make_unique<mi::quant::QuantIndex>(
        mi::quant::to_model_index(mi::quant::open_quant_model(int8_file(), mi::quant::ModelKind::kDiT)));
    md::ComputeConfig q8 = md::MossDit::compute_config(quant_.get(), md::AttentionMode::kQ8Fp8);
    work_ = std::max({md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFusedGraphs, q8, nullptr),
                      md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFusedGraphs,
                                                   md::MossDit::compute_config(quant_.get(), md::AttentionMode::kExact), nullptr),
                      md::MossDit::encode_text_workspace_bytes(std::max(c.s_cond, c.s_uncond))});
    persistent_ = md::MossDit::persistent_bytes(quant_->index);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(persistent_ + work_);
    model_ = std::make_unique<md::MossDit>(*quant_, *arena_, *streams_);
    io_ = std::make_unique<ModelIo>(*model_, c, s);
    std::printf("[dit_int8] INT8 model: %.3f GiB persistent, %.3f GiB forward workspace, loaded in %.2f s\n",
                static_cast<double>(persistent_) / kGiB, static_cast<double>(work_) / kGiB, model_->load_stats().seconds);
  }
  static void TearDownTestSuite() {
    io_.reset();
    model_.reset();
    arena_.reset();
    quant_.reset();
    streams_.reset();
    case_.reset();
  }
  void SetUp() override {
    if (!model_) GTEST_SKIP() << "INT8 DiT file, SFT bundle or goldens absent";
    model_->set_block_path(md::BlockPath::kFused);
    model_->set_attention(md::AttentionMode::kExact);
    model_->set_cancel(nullptr);
  }
  cudaStream_t stream() const { return streams_->compute(); }
  std::vector<md::Row> cfg_rows() const { return {{io_->cond->t, {}}, {io_->uncond->t, case_->uncond_skip}}; }

  static std::unique_ptr<GoldenCase> case_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::vector<std::vector<float>> bf16_v_;
  static std::vector<std::vector<std::uint16_t>> bf16_refined_;
  static std::vector<std::vector<float>> bf16_file_v_;
  static std::string bf16_file_linears_;
  static md::ComputeConfig bf16_file_config_;
  static std::unique_ptr<mi::quant::QuantIndex> quant_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<md::MossDit> model_;
  static std::unique_ptr<ModelIo> io_;
  static std::size_t work_, persistent_;
};

std::unique_ptr<GoldenCase> DitInt8Model::case_;
std::unique_ptr<mi::runtime::StreamSet> DitInt8Model::streams_;
std::vector<std::vector<float>> DitInt8Model::bf16_v_;
std::vector<std::vector<std::uint16_t>> DitInt8Model::bf16_refined_;
std::vector<std::vector<float>> DitInt8Model::bf16_file_v_;
std::string DitInt8Model::bf16_file_linears_;
md::ComputeConfig DitInt8Model::bf16_file_config_;
std::unique_ptr<mi::quant::QuantIndex> DitInt8Model::quant_;
std::unique_ptr<mi::runtime::DeviceArena> DitInt8Model::arena_;
std::unique_ptr<md::MossDit> DitInt8Model::model_;
std::unique_ptr<ModelIo> DitInt8Model::io_;
std::size_t DitInt8Model::work_ = 0, DitInt8Model::persistent_ = 0;

TEST_F(DitInt8Model, Bf16PartsOfTheFileEqualTheBundle) {
  // The text refiner (bf16 in the file) gives the bundle's refined text bit for bit.
  EXPECT_EQ(download<std::uint16_t>(io_->cond->t), bf16_refined_[0]);
  EXPECT_EQ(download<std::uint16_t>(io_->uncond->t), bf16_refined_[1]);
  const md::ComputeConfig c = model_->compute_config();
  EXPECT_TRUE(c.quantized);
  EXPECT_FALSE(c.bf16_up_gate);
}

// A DiT loaded through a quantized index whose linears are all bf16 (a single-file bf16 DiT) runs the P5 bf16 blocks:
// bit-identical to the bundle's model.
TEST_F(DitInt8Model, Bf16QuantIndexEqualsTheBundle) {
  EXPECT_EQ(bf16_file_linears_, "bf16");
  EXPECT_FALSE(bf16_file_config_.quantized);
  for (std::size_t r = 0; r < 2; ++r) EXPECT_EQ(differing(bf16_file_v_[r], bf16_v_[r]), 0u) << r;
}

TEST_F(DitInt8Model, StepVelocityVsBf16Engine) {
  const auto rows = cfg_rows();
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    model_->set_attention(mode);
    const auto v = io_->forward(*model_, rows, case_->timestep, stream());
    const double cc = cosine(v[0], bf16_v_[0]), cu = cosine(v[1], bf16_v_[1]);
    std::printf("[dit_int8] step-0 velocity INT8 + %s attention vs bf16 engine (1024^2, %lld+%lld tokens): cos cond "
                "%.6f, uncond %.6f\n", md::attention_mode_name(mode), static_cast<long long>(case_->s_cond),
                static_cast<long long>(case_->s_uncond), cc, cu);
    EXPECT_GE(cc, 0.99);
    EXPECT_GE(cu, 0.99);
  }
}

TEST_F(DitInt8Model, VarlenBatchBitExactPerRow) {
  md::SkipSet s24;
  s24.set(24);
  const md::Row set[] = {{io_->cond->t, {}}, {io_->uncond->t, case_->uncond_skip}, {io_->cond->t, s24}};
  const bool checks = mi::varlen_debug_checks();
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    model_->set_attention(mode);
    mi::set_varlen_debug_checks(true);
    const auto batched = io_->forward(*model_, set, case_->timestep, stream());
    mi::set_varlen_debug_checks(checks);
    for (std::size_t r = 0; r < 3; ++r) {
      const auto alone = io_->forward(*model_, std::span<const md::Row>(&set[r], 1), case_->timestep, stream());
      const std::size_t d = differing(alone[0], batched[r]);
      std::printf("[dit_int8] varlen %s attention: row %zu, %zu values differ from the row alone\n",
                  md::attention_mode_name(mode), r, d);
      EXPECT_EQ(d, 0u) << md::attention_mode_name(mode) << " row " << r;
    }
  }
}

TEST_F(DitInt8Model, GraphsBitIdenticalToEager) {
  const auto rows = cfg_rows();
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    model_->set_block_path(md::BlockPath::kFused);
    model_->set_attention(mode);
    const auto eager = io_->forward(*model_, rows, case_->timestep, stream());
    model_->set_block_path(md::BlockPath::kFusedGraphs);
    std::vector<std::vector<float>> got;
    for (int i = 0; i < 3; ++i) got = io_->forward(*model_, rows, case_->timestep, stream());  // eager, capture, replay
    EXPECT_GT(model_->graph_stats().replays, 0u);
    for (std::size_t r = 0; r < rows.size(); ++r) {
      EXPECT_EQ(differing(got[r], eager[r]), 0u) << md::attention_mode_name(mode) << " row " << r;
    }
  }
  model_->set_block_path(md::BlockPath::kFused);
}

TEST_F(DitInt8Model, CancelBetweenBlocksThenExact) {
  const auto rows = cfg_rows();
  model_->set_attention(md::AttentionMode::kQ8Fp8);
  const auto want = io_->forward(*model_, rows, case_->timestep, stream());
  std::atomic<bool> cancel{false};
  model_->set_cancel(&cancel);
  int last = -1;
  EXPECT_THROW(io_->forward(*model_, rows, case_->timestep, stream(),
                            [&](int block, const Tensor&) {
                              last = block;
                              if (block == 10) cancel = true;
                            }),
               md::Cancelled);
  EXPECT_EQ(last, 10);
  cancel = false;
  const auto again = io_->forward(*model_, rows, case_->timestep, stream());
  for (std::size_t r = 0; r < rows.size(); ++r) EXPECT_EQ(differing(again[r], want[r]), 0u) << r;
  model_->set_cancel(nullptr);
}

TEST_F(DitInt8Model, StreamedBitIdenticalToResident) {
  const auto rows = cfg_rows();
  const auto want = io_->forward(*model_, rows, case_->timestep, stream());
  const GoldenCase& c = *case_;
  const std::int64_t lengths[] = {c.s_cond, c.s_uncond};
  const std::size_t work = std::max(md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFused,
                                                                 model_->compute_config(), nullptr),
                                    md::MossDit::encode_text_workspace_bytes(std::max(c.s_cond, c.s_uncond)));
  for (const md::StreamSource source : {md::StreamSource::kFile, md::StreamSource::kHostPinned}) {
    md::Residency res;
    res.resident_blocks = 10;
    res.source = source;
    mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(quant_->index, res) + work +
                                   md::MossDit::stream_workspace_bytes(quant_->index, res));
    md::MossDit streamed(*quant_, arena, *streams_, false, nullptr, res);
    ModelIo io(streamed, c, stream());
    const md::Row srows[] = {{io.cond->t, {}}, {io.uncond->t, c.uncond_skip}};
    const auto got = io.forward(streamed, srows, c.timestep, stream());
    const md::StreamStats& st = streamed.last_stream_stats();
    std::printf("[dit_int8] streamed (%s, 10 of 42 resident): %.2f GB per step at %.1f GB/s, %.0f%% hidden\n",
                md::stream_source_name(source), static_cast<double>(st.bytes) / 1e9, st.gbps(), 100 * st.hidden());
    std::uint64_t want_bytes = 0;  // the streamed blocks' stored bytes: INT8 weights and scales
    const auto mask = md::resident_mask(res.resident_blocks);
    for (int b = 0; b < md::kNumBlocks; ++b) {
      if (!mask.test(static_cast<std::size_t>(b))) {
        want_bytes += quant_->index.find_group(mi::weights::Component::kTransformer, "block." + std::to_string(b))->total_bytes;
      }
    }
    EXPECT_EQ(st.bytes, want_bytes);
    for (std::size_t r = 0; r < 2; ++r) EXPECT_EQ(differing(got[r], want[r]), 0u) << md::stream_source_name(source) << " row " << r;
  }
}

// AC20 (dev numbers): one CFG step at 1024^2 with 282 + 10 text tokens (uncond skipping block 22), every block
// resident, for the bf16 fused engine (SFT bundle), INT8 + exact attention and INT8 + 8-bit attention (Comfy-Org file),
// in one process: bf16, then INT8 (exact and q8 interleaved), then bf16 again to bracket clock drift. Median of 7
// steps after 2 warm-up steps; timings only (labelled dev: other processes share the GPU).
TEST(DitInt8Speed, Step1024) {
  if (!have_files() || !fs::is_directory(golden_dir() / "g_fridge_1024")) GTEST_SKIP() << "files absent";
  const GoldenCase c = load_golden("g_fridge_1024");
  mi::runtime::StreamSet streams;
  const cudaStream_t s = streams.compute();
  constexpr std::int64_t kCond = 282, kUncond = 10;
  const std::int64_t lengths[] = {kCond, kUncond};
  DevTensor latents(DType::kF32, {128, c.h, c.w});
  latents.buf.upload(c.noise.data(), c.noise.size() * 4);
  DevTensor cond(DType::kBF16, {kCond, kDim}), uncond(DType::kBF16, {kUncond, kDim});
  {
    std::vector<std::uint16_t> v(static_cast<std::size_t>(kCond * kDim));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = to_bf16(0.5f * std::sin(0.013f * static_cast<float>(i)));
    cond.buf.upload(v.data(), v.size() * 2);
    uncond.buf.upload(v.data(), static_cast<std::size_t>(kUncond * kDim) * 2);
  }
  DevTensor o0(DType::kF32, {128, c.h, c.w}), o1(DType::kF32, {128, c.h, c.w});
  md::SkipSet s22;
  s22.set(22);
  const md::Row rows[] = {{cond.t, {}}, {uncond.t, s22}};
  const Tensor outs[] = {o0.t, o1.t};
  const auto step = [&](md::MossDit& m) {
    cudaEvent_t e0, e1;
    MI_CUDA_CHECK(cudaEventCreate(&e0));
    MI_CUDA_CHECK(cudaEventCreate(&e1));
    MI_CUDA_CHECK(cudaEventRecord(e0, s));
    m.forward(latents.t, c.timestep, rows, outs, s);
    MI_CUDA_CHECK(cudaEventRecord(e1, s));
    MI_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    MI_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return static_cast<double>(ms) / 1e3;
  };
  const auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const std::size_t free = mi::runtime::DeviceArena::free_device_bytes() - (std::size_t{512} << 20);
  const auto bf16_step = [&] {
    const mi::weights::ModelIndex bundle = mi::weights::load_index(sft_dir() / "moss.mossidx", sft_dir());
    const std::size_t work = md::MossDit::workspace_bytes(c.h, c.w, lengths);
    const std::size_t bytes = md::MossDit::persistent_bytes(bundle) + work;
    if (bytes > free) return -1.0;  // not every block fits next to the other processes
    mi::runtime::DeviceArena arena(bytes);
    md::MossDit m(bundle, arena, streams);
    for (int i = 0; i < 2; ++i) step(m);
    std::vector<double> t;
    for (int i = 0; i < 7; ++i) t.push_back(step(m));
    return median(t);
  };
  const double bf16_a = bf16_step();
  double int8_exact = 0, int8_q8 = 0, int8_q8_fp16 = 0;
  {
    const mi::quant::QuantIndex qi = mi::quant::to_model_index(mi::quant::open_quant_model(int8_file(), mi::quant::ModelKind::kDiT));
    std::size_t work = 0;
    for (const md::AttentionMode a : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8, md::AttentionMode::kQ8Fp16}) {
      work = std::max(work, md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFused,
                                                         md::MossDit::compute_config(&qi, a), nullptr));
    }
    mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(qi.index) + work);
    md::MossDit m(qi, arena, streams);
    std::vector<double> te, tq, tf;
    for (int i = 0; i < 9; ++i) {
      for (const md::AttentionMode a : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8, md::AttentionMode::kQ8Fp16}) {
        m.set_attention(a);
        const double t = step(m);
        if (i < 2) continue;
        (a == md::AttentionMode::kExact ? te : a == md::AttentionMode::kQ8Fp8 ? tq : tf).push_back(t);
      }
    }
    int8_exact = median(te);
    int8_q8 = median(tq);
    int8_q8_fp16 = median(tf);
  }
  const double bf16_b = bf16_step();
  const double bf16 = bf16_a > 0 && bf16_b > 0 ? std::min(bf16_a, bf16_b) : std::max(bf16_a, bf16_b);
  std::printf("[dit_int8 speed, dev] 1024^2 CFG step (282+10 tokens): bf16 fused %.3f / %.3f s; INT8 + exact %.3f s (%.2fx); "
              "INT8 + q8_fp8 %.3f s (%.2fx); INT8 + q8_fp16 %.3f s (%.2fx)\n",
              bf16_a, bf16_b, int8_exact, bf16 / int8_exact, int8_q8, bf16 / int8_q8, int8_q8_fp16, bf16 / int8_q8_fp16);
  RecordProperty("bf16_step_s", std::to_string(bf16));
  RecordProperty("int8_exact_step_s", std::to_string(int8_exact));
  RecordProperty("int8_q8_step_s", std::to_string(int8_q8));
  EXPECT_GT(int8_exact, 0.0);
}

}  // namespace
