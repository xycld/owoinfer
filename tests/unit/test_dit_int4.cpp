// P6c (lane T): the DiT on the Turbo-tier compute path: INT4 ConvRot linears (kernels/quant4) in
// models/moss_dit/dit_block_quant, loaded natively from toxicdog's moss_image_sft_int4_convrot file (convrot_w4a4,
// legacy header, separate q/k/v) through quant::to_model_index(..., {.native_int4 = true}).
//
// Block level, on block 5:
//   - the file's views: q/k/v fused into one packed [3 dim, dim / 2] weight with a [3 dim] scale; the index without
//     native_int4 still refuses the file (needs decoding), as before;
//   - every linear INT4, two varlen rows: against the reference _Block with comfy-kitchen's convrot_w4a4_linear (CUDA
//     numerics) and SDPA, from tests/data/moss_dit_int4 (tools/reference/gen_dit_int4_vectors.py);
//   - a block mixing all three precisions (q INT4, k INT8, v bf16: split QKV; o bf16; up_gate INT4; down INT8) against
//     the same reference;
//   - 8-bit attention: accuracy against the reference (lossy, reported);
//   - `big` sized as forward() sizes it: nothing written past it;
//   - runtime LoRA on INT4 sites: a side branch whose B is zero is bit-identical to the plain INT4 block (the INT8 LoRA
//     GEMMs on the unpacked values give the INT4 GEMMs' results), and a real side branch changes the block as it
//     changes the bf16 block;
//   - unpack_int4 against a host unpack.
// The reference comparisons use the generator's stored inputs; the engine-vs-engine identities use hash inputs.
// Model level, on the golden 1024^2 step-0 case (g_fridge_1024 text states and noise) with the INT4 file:
//   - step-0 velocity cosine vs the bf16 engine (SFT bundle), exact and 8-bit attention (reported);
//   - one varlen batch (cond, uncond skipping 22, SLG row skipping 24) == each row alone, bit for bit;
//   - streamed INT4 blocks (file and pinned host sources) == resident, bit for bit; INT4 blocks move ~1/4 of bf16;
//   - CUDA graphs == eager, bit for bit (exact and 8-bit attention).
// Speed (dev): one 1024^2 CFG step, INT4 vs INT8 vs bf16 in one process.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/varlen.h"
#include "kernels/attention/flash_attention.h"
#include "kernels/basic/gemm.h"
#include "kernels/basic/rope.h"
#include "models/lora/lora.h"
#include "models/lora/side_branch.h"
#include "models/moss_dit/dit_block_fused.h"
#include "models/moss_dit/dit_block_quant.h"
#include "models/moss_dit/dit_kernels.h"
#include "models/moss_dit/dit_weights.h"
#include "models/moss_dit/moss_dit.h"
#include "runtime/device_arena.h"
#include "runtime/pinned_ring.h"
#include "reference_gpu.h"
#include "runtime/stream_set.h"
#include "test_lora_util.h"
#include "test_util.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"
#include "weights/quant/quant_model.h"
#include "weights/upload.h"

namespace {

namespace fs = std::filesystem;
namespace md = mi::moss_dit;
namespace tl = mi::testing::lora;
using mi::DType;
using mi::Tensor;

constexpr std::int64_t kDim = md::kDim;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr int kBlock = 5;
constexpr std::int64_t kGridH = 6, kGridW = 8;
constexpr std::int64_t kText[] = {17, 9};
constexpr md::AttentionMode kQ8Modes[] = {md::AttentionMode::kQ8Fp8, md::AttentionMode::kQ8Fp16, md::AttentionMode::kQ8Fp8F32Acc,
                                          md::AttentionMode::kQ8Fp16F32Acc};

fs::path env_path(const char* var, const char* fallback) {
  const char* env = std::getenv(var);
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path(fallback);
}
fs::path int4_file() { return env_path("MOSS_INT4_DIT", "/workspace/Models/community/toxicdog/moss_image_sft_int4_convrot.safetensors"); }
fs::path int8_file() { return env_path("MOSS_INT8_DIT", "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors"); }
fs::path sft_dir() { return env_path("MOSS_WEIGHTS", "/workspace/Models/MOSS-Image") / "sft"; }
fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "moss_dit_int4"; }
fs::path golden_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "golden"; }
bool have_files() {
  return fs::is_regular_file(int4_file()) && fs::is_regular_file(int8_file()) && fs::is_regular_file(sft_dir() / "moss.mossidx");
}

mi::quant::QuantIndex int4_index() {
  return mi::quant::to_model_index(mi::quant::open_quant_model(int4_file(), mi::quant::ModelKind::kDiT), {.native_int4 = true});
}

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

// Hash activations (splitmix64, as test_quant4_kernels.cu): per-channel exponents, every 397th channel 64x larger.
std::uint64_t mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
std::uint16_t bf16_trunc(float v) { return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16); }
std::vector<std::uint16_t> synth_act(std::uint64_t seed, std::int64_t rows, std::int64_t cols) {
  std::vector<std::uint16_t> out(static_cast<std::size_t>(rows * cols));
  for (std::int64_t c = 0; c < cols; ++c) {
    const int c_exp = static_cast<int>(mix(static_cast<std::uint64_t>(c) * 7919 + seed) % 6) - 3 + (c % 397 == 5 ? 6 : 0);
    for (std::int64_t r = 0; r < rows; ++r) {
      const std::uint64_t z = mix(static_cast<std::uint64_t>(r * cols + c) + seed * 0x9E3779B97F4A7C15ull);
      const float m = static_cast<float>(128 + (z & 0x7F));
      const int e = static_cast<int>((z >> 7) & 7);
      out[static_cast<std::size_t>(r * cols + c)] = bf16_trunc(std::ldexp((z >> 10) & 1 ? -m : m, c_exp - 7 - e));
    }
  }
  return out;
}
// A two-row input for the gates that do not depend on the input distribution (bit identities between two engine runs):
// the comparisons against the reference use its stored DiT-like residual streams (all_int4_x, mixed_x).
std::vector<std::uint16_t> hash_x() { return synth_act(201, 2 * 6 * 8 + 17 + 9, 3072); }

// bf16 agreement of [rows, kDim] outputs (as test_dit_int8): bit-exact fraction, max raw ULP, the fraction within 1 bf16
// ULP of the element's scale (max(|ref|, the row's RMS)) and the cosine; also the cosine of the block's update
// (out - x), which the residual does not dominate.
struct Agreement {
  double exact = 0, within1 = 0, cosine = 0, max_scaled = 0, update_cosine = 0;
  int max_ulp = 0;
};
int ordered(std::uint16_t b) { return (b & 0x8000) ? -static_cast<int>(b & 0x7fff) : static_cast<int>(b); }
double bf16_ulp(double v) {
  int e = 0;
  std::frexp(std::max(std::abs(v), 1e-30), &e);
  return std::ldexp(1.0, e - 8);
}
Agreement agree(const std::vector<std::uint16_t>& ref, const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& x) {
  Agreement a;
  double dot = 0, rr = 0, gg = 0, udot = 0, uu = 0, vv = 0;
  std::size_t exact = 0, w1 = 0;
  for (std::size_t row = 0; row * kDim < ref.size(); ++row) {
    double ms = 0;
    for (std::int64_t c = 0; c < kDim; ++c) ms += std::pow(from_bf16(ref[row * kDim + c]), 2);
    const double rms = std::sqrt(ms / kDim);
    for (std::int64_t c = 0; c < kDim; ++c) {
      const std::size_t i = row * kDim + c;
      a.max_ulp = std::max(a.max_ulp, std::abs(ordered(ref[i]) - ordered(got[i])));
      exact += ref[i] == got[i];
      const double r = from_bf16(ref[i]), g = from_bf16(got[i]), x0 = from_bf16(x[i]);
      const double scaled = std::abs(g - r) / bf16_ulp(std::max(std::abs(r), rms));
      a.max_scaled = std::max(a.max_scaled, scaled);
      w1 += scaled <= 1.0;
      dot += r * g;
      rr += r * r;
      gg += g * g;
      udot += (r - x0) * (g - x0);
      uu += (r - x0) * (r - x0);
      vv += (g - x0) * (g - x0);
    }
  }
  a.exact = static_cast<double>(exact) / static_cast<double>(ref.size());
  a.within1 = static_cast<double>(w1) / static_cast<double>(ref.size());
  a.cosine = dot / std::sqrt(rr * gg);
  a.update_cosine = udot / std::sqrt(uu * vv);
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

// ---- Loader --------------------------------------------------------------------------------------------------

TEST(DitInt4Index, NativeViewOfTheCommunityFile) {
  if (!fs::is_regular_file(int4_file())) GTEST_SKIP() << "absent: " << int4_file();
  const mi::quant::QuantModel m = mi::quant::open_quant_model(int4_file(), mi::quant::ModelKind::kDiT);
  // Unchanged default: W4A4 linears need decoding (the converter's path to INT8).
  EXPECT_THROW(mi::quant::to_model_index(m), mi::quant::QuantError);
  const mi::quant::QuantIndex qi = mi::quant::to_model_index(m, {.native_int4 = true});
  int int4 = 0;
  for (const auto& [module, spec] : qi.specs) {
    if (module.rfind("layers.", 0) != 0) continue;
    EXPECT_EQ(spec, mi::quant::QuantSpec::int4_convrot(256)) << module;
    int4 += spec == mi::quant::QuantSpec::int4_convrot(256);
  }
  EXPECT_EQ(int4, 42 * 6);
  EXPECT_EQ(mi::quant::describe(mi::quant::QuantSpec::int4_convrot(256)), "int4_convrot256");
  EXPECT_EQ(md::describe_linears(&qi), "int4_convrot256");
  EXPECT_TRUE(md::has_int4_linears(&qi));
  EXPECT_DOUBLE_EQ(md::int4_gemm_fraction(&qi), 1.0);
  EXPECT_DOUBLE_EQ(md::int8_gemm_fraction(&qi), 0.0);
  const mi::weights::WeightGroup* g = qi.index.find_group(mi::weights::Component::kTransformer, "block.5");
  ASSERT_NE(g, nullptr);
  for (const auto& t : g->tensors) {
    if (t.name == "layers.5.proj.ffn_down.weight") {
      EXPECT_EQ(t.shape, (std::vector<std::int64_t>{3072, 4096}));  // packed: K / 2 bytes per row
    }
  }
  // quant::view of a host copy: k is the unpacked K.
  mi::weights::ReaderCache readers(qi.index, false);
  const mi::weights::HostGroup host = mi::weights::load_group_to_host(readers, *g, mi::weights::HostMemory::kPlain);
  const mi::quant::QuantWeights v = mi::quant::view(qi, host.tensors, "layers.5.proj.ffn_down", mi::quant::QuantSpec::int4_convrot(256));
  EXPECT_EQ(v.n, 3072);
  EXPECT_EQ(v.k, 8192);
  EXPECT_NE(v.scale, nullptr);
  EXPECT_THROW(mi::quant::view(qi, host.tensors, "layers.5.proj.ffn_down", mi::quant::QuantSpec::int8_convrot(256)),
               mi::quant::QuantError);
}

// comfy-kitchen runs convrot_w4a4 only with quant_group_size 64 (its default when absent): other values keep the
// decode path (to_model_index then asks for conversion), never the native INT4 view.
// Pure CPU (the index only): a suite name outside the GPU labels' DitInt4* pattern, so CI runs it.
TEST(QuantInt4Index, QuantGroupSizeGatesTheNativeView) {
  const mi::testing::AnyModelShapes any_shapes;  // a small synthetic model
  const fs::path file = fs::temp_directory_path() / ("mossinfer_test_dit_int4_qgs_" + std::to_string(std::random_device{}()) + ".safetensors");
  for (const int g : {0, 64, 128}) {
    const std::string conf = std::string(R"({\"format\":\"convrot_w4a4\",\"convrot_groupsize\":256)") +
                             (g != 0 ? R"(,\"quant_group_size\":)" + std::to_string(g) : std::string()) + "}";
    tl::write_st(file,
                 {{"layers.0.proj.o.weight", mi::weights::DType::kI8, {8, 128}, std::vector<float>(8 * 128, 17.0f)},
                  {"layers.0.proj.o.weight_scale", mi::weights::DType::kF32, {8}, std::vector<float>(8, 0.5f)}},
                 R"({"_quantization_metadata":"{\"layers\":{\"layers.0.proj.o\":)" + conf + "}}\"}");
    const mi::quant::QuantModel m = mi::quant::open_quant_model(file, mi::quant::ModelKind::kDiT);
    if (g == 128) {
      EXPECT_THROW(mi::quant::to_model_index(m, {.native_int4 = true}), mi::quant::QuantError);
    } else {
      EXPECT_EQ(*mi::quant::to_model_index(m, {.native_int4 = true}).find_spec("layers.0.proj.o"),
                mi::quant::QuantSpec::int4_convrot(256)) << g;
    }
  }
  fs::remove(file);
}

// LoRA on an INT4 DiT: merged adapters (LoHa here) on INT4 block linears are refused up front; side branches are
// costed at INT8 speed on their sites; a step-embedder replacement shadows time-embedding / modulation-bias deltas.
TEST(DitInt4Index, LoraOnInt4File) {
  if (!fs::is_regular_file(int4_file())) GTEST_SKIP() << "absent: " << int4_file();
  const mi::quant::QuantIndex qi = int4_index();
  const fs::path dir = fs::temp_directory_path() / ("mossinfer_test_dit_int4_lora_" + std::to_string(std::random_device{}()));
  const std::string o = "diffusion_model.layers.0.proj.o";
  const auto stack_of = [&](const char* name, const std::vector<tl::StTensor>& ts) {
    tl::write_st(dir / name, ts);
    const mi::lora::LoraSpec spec{dir / name, 1.0f};
    mi::lora::LoadOptions lo;
    lo.int8_block_gemms = true;
    return mi::lora::LoraStack(std::span<const mi::lora::LoraSpec>(&spec, 1), mi::lora::moss_dit_targets(), lo);
  };
  const auto w = mi::weights::DType::kBF16;
  const mi::lora::LoraStack loha = stack_of("loha.safetensors", {{o + ".hada_w1_a", w, {kDim, 2}, tl::randn(kDim * 2, 1, 0.1f)},
                                                                  {o + ".hada_w1_b", w, {2, kDim}, tl::randn(kDim * 2, 2, 0.1f)},
                                                                  {o + ".hada_w2_a", w, {kDim, 2}, tl::randn(kDim * 2, 3, 0.1f)},
                                                                  {o + ".hada_w2_b", w, {2, kDim}, tl::randn(kDim * 2, 4, 0.1f)}});
  try {
    md::check_lora_merges(loha, &qi);
    ADD_FAILURE() << "a merged adapter on an INT4 linear was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("INT4 and NVFP4 linears take side-branch adapters only"), std::string::npos) << e.what();
  }
  EXPECT_NO_THROW(md::check_lora_merges(loha, nullptr));  // a bf16 DiT merges it
  const mi::lora::LoraStack low = stack_of("low.safetensors", {{o + ".lora_down.weight", w, {4, kDim}, tl::randn(4 * kDim, 5, 0.02f)},
                                                                {o + ".lora_up.weight", w, {kDim, 4}, tl::randn(4 * kDim, 6, 0.02f)}});
  EXPECT_NO_THROW(md::check_lora_merges(low, &qi));
  mi::lora::SideBranches sb(low);
  const double per_block = 4.0 * kDim * kDim + 3.0 * md::kFfnDim * kDim;
  EXPECT_DOUBLE_EQ(md::int4_side_branch_fraction(&qi, sb), static_cast<double>(kDim * kDim) / (42 * per_block));
  EXPECT_DOUBLE_EQ(md::int4_side_branch_fraction(nullptr, sb), 0.0);
  const mi::lora::LoraStack shadow = stack_of("shadow.safetensors",
      {{"diffusion_model.time_embedding.0.diff", w, {kDim, 256}, std::vector<float>(kDim * 256, 0.001f)},
       {"diffusion_model.shared_block_modulation.diff_b", w, {6 * kDim}, std::vector<float>(6 * kDim, 0.001f)}});
  const std::vector<std::string> msgs = md::shadowed_by_time_embedding(shadow);
  ASSERT_EQ(msgs.size(), 2u);
  EXPECT_TRUE(md::shadowed_by_time_embedding(low).empty());
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// Both kernels: 16-byte vectors (cols % 32 == 0, 16-byte strides) and the scalar fallback (odd strides).
TEST(DitInt4Kernels, UnpackMatchesHost) {
  const struct {
    std::int64_t cols, ld_in, ld_out;
  } cases[] = {{512, 272, 528}, {510, 263, 517}};
  constexpr std::int64_t kRows = 37;
  for (const auto& cs : cases) {
    std::vector<std::uint8_t> in(static_cast<std::size_t>(kRows * cs.ld_in));
    std::mt19937 rng(7);
    for (auto& b : in) b = static_cast<std::uint8_t>(rng());
    mi::DeviceBuffer din(in.size()), dout(static_cast<std::size_t>(kRows * cs.ld_out));
    din.upload(in.data(), in.size());
    MI_CUDA_CHECK(cudaMemset(dout.get(), 0x55, dout.size()));
    md::unpack_int4(din.get(), cs.ld_in, dout.get(), cs.ld_out, kRows, cs.cols, nullptr);
    std::vector<std::int8_t> out(dout.size());
    MI_CUDA_CHECK(cudaMemcpy(out.data(), dout.get(), out.size(), cudaMemcpyDeviceToHost));
    for (std::int64_t r = 0; r < kRows; ++r) {
      for (std::int64_t c = 0; c < cs.ld_out; ++c) {
        const auto got = out[static_cast<std::size_t>(r * cs.ld_out + c)];
        if (c >= cs.cols) {
          ASSERT_EQ(static_cast<std::uint8_t>(got), 0x55) << "wrote past cols " << cs.cols;
          continue;
        }
        const int byte = in[static_cast<std::size_t>(r * cs.ld_in + c / 2)];
        const int nib = c % 2 == 0 ? byte & 0xF : byte >> 4;
        ASSERT_EQ(got, nib >= 8 ? nib - 16 : nib) << r << "," << c << " (cols " << cs.cols << ")";
      }
    }
  }
}

// ---- Block level ---------------------------------------------------------------------------------------------

class DitInt4Block : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!have_files() || !fs::is_regular_file(data_dir() / "all_int4_y.npy")) return;
    int4_ = std::make_unique<mi::quant::QuantIndex>(int4_index());
    int8_ = std::make_unique<mi::quant::QuantIndex>(
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
    int4_map_ = upload(int4_->index, int4_.get());
    int8_map_ = upload(int8_->index, int8_.get());
    bf16_map_ = upload(*bundle_, nullptr);
    w4_ = md::detail::block_views(kBlock, int4_map_, int4_.get());
    w8_ = md::detail::block_views(kBlock, int8_map_, int8_.get());
    w16_ = md::detail::block_views(kBlock, bf16_map_, nullptr);
    const md::FusedCheck check = md::check_fused_gemms(streams_->compute());
    if (check.ok) fused_route_.validated = check.validated;
  }
  static void TearDownTestSuite() {
    int4_map_.clear();
    int8_map_.clear();
    bf16_map_.clear();
    arena_.reset();
    streams_.reset();
    bundle_.reset();
    int8_.reset();
    int4_.reset();
  }
  void SetUp() override {
    if (!int4_) GTEST_SKIP() << "INT4 / INT8 DiT file, SFT bundle or tests/data/moss_dit_int4 absent";
  }
  cudaStream_t stream() const { return streams_->compute(); }

  // Buffers of one block call over rows of hw + text[r] tokens (x from `x`), as MossDit::forward lays them out.
  struct Run {
    std::int64_t l = 0;
    std::unique_ptr<DevTensor> x, y, big, rope, mod, scale, cu, lora_t;
    mi::DeviceBuffer lse, q8, gemm_ws, lora_ws, unpack;
    md::BlockRun run;
    md::QuantBlockBuffers qa;
    mi::kernels::Gemm gemm;
    // via: every INT4 linear through the INT8 GEMMs on unpacked operands (QuantBlockBuffers::int4_via_int8).
    Run(const std::vector<std::uint16_t>& xv, std::span<const std::int64_t> text, cudaStream_t s,
        const mi::lora::SideBranches* lora = nullptr, bool via = false) {
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
      if (lora != nullptr) {
        lora_t = std::make_unique<DevTensor>(DType::kBF16, std::initializer_list<std::int64_t>{l, lora->max_rank()});
        lora_ws = mi::DeviceBuffer(mi::lora::t_workspace_bytes(l, lora->max_rank()));
        qa.lora = lora;
        qa.lora_t = lora_t->t;
        qa.lora_workspace = lora_ws.get();
        qa.lora_workspace_bytes = lora_ws.size();
      }
      if (lora != nullptr || via) {
        unpack = mi::DeviceBuffer(md::kInt4UnpackBytes);
        qa.int4_unpack = unpack.get();
        qa.int4_unpack_bytes = unpack.size();
      }
      qa.int4_via_int8 = via;
    }
    md::QkvGemm qkv_gemm() { return {&gemm, gemm_ws.get(), gemm_ws.size()}; }
    void reset(const std::vector<std::uint16_t>& xv) { x->buf.upload(xv.data(), xv.size() * 2); }
  };

  std::vector<std::uint16_t> run_block(const md::BlockTensors& w, const std::vector<std::uint16_t>& xv,
                                       std::span<const std::int64_t> text, md::AttentionMode mode = md::AttentionMode::kExact,
                                       const mi::lora::SideBranches* lora = nullptr, bool via = false) {
    Run r(xv, text, stream(), lora, via);
    md::run_block_quant(w, r.qa, r.run, r.qkv_gemm(), fused_route_, mode, stream(), kBlock);
    MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
    return download<std::uint16_t>(r.x->t);
  }

  static std::unique_ptr<mi::quant::QuantIndex> int4_, int8_;
  static std::unique_ptr<mi::weights::ModelIndex> bundle_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static mi::weights::TensorMap int4_map_, int8_map_, bf16_map_;
  static md::BlockTensors w4_, w8_, w16_;
  static md::FusedRouteSnapshot fused_route_;
};

std::unique_ptr<mi::quant::QuantIndex> DitInt4Block::int4_, DitInt4Block::int8_;
std::unique_ptr<mi::weights::ModelIndex> DitInt4Block::bundle_;
std::unique_ptr<mi::runtime::StreamSet> DitInt4Block::streams_;
std::unique_ptr<mi::runtime::DeviceArena> DitInt4Block::arena_;
mi::weights::TensorMap DitInt4Block::int4_map_, DitInt4Block::int8_map_, DitInt4Block::bf16_map_;
md::BlockTensors DitInt4Block::w4_, DitInt4Block::w8_, DitInt4Block::w16_;
md::FusedRouteSnapshot DitInt4Block::fused_route_;

TEST_F(DitInt4Block, ViewsOfTheInt4File) {
  EXPECT_TRUE(w4_.int4());
  EXPECT_TRUE(w4_.quantized());
  EXPECT_FALSE(w8_.int4());
  EXPECT_FALSE(w4_.qkv_split);
  EXPECT_EQ(w4_.qkv4.w.dim(0), 3 * kDim);  // separate q/k/v uploaded back to back: one fused packed weight
  EXPECT_EQ(w4_.qkv4.w.dim(1), kDim / 2);
  EXPECT_EQ(w4_.qkv4.scale.numel(), 3 * kDim);
  EXPECT_EQ(w4_.up_gate4.w.dim(0), 2 * md::kFfnDim);
  EXPECT_EQ(w4_.down4.k(), md::kFfnDim);
  EXPECT_EQ(w4_.qkv.data, nullptr);
  EXPECT_EQ(w4_.qkv8.w.data, nullptr);
  const double b4 = static_cast<double>(md::BlockWeights::block_device_bytes(int4_->index));
  const double b8 = static_cast<double>(md::BlockWeights::block_device_bytes(int8_->index));
  const double b16 = static_cast<double>(md::BlockWeights::block_device_bytes(*bundle_));
  std::printf("[dit_int4] block bytes: INT4 %.1f MiB, INT8 %.1f MiB, bf16 %.1f MiB (INT4 / bf16 %.4f)\n", b4 / (1 << 20),
              b8 / (1 << 20), b16 / (1 << 20), b4 / b16);
  EXPECT_LT(b4 / b16, 0.27);
}

TEST_F(DitInt4Block, AllInt4MatchesQuantizedReference) {
  const auto xv = load_u16("all_int4_x");
  const auto ref = load_u16("all_int4_y");
  const Agreement a = agree(ref, run_block(w4_, xv, kText), xv);
  std::printf("[dit_int4] all-INT4 block vs reference (exact attention): %.4f%% bit-exact, %.4f%% within 1 ULP of scale "
              "(max %.2f), max %d raw ULP, cos %.9f, update cos %.7f\n",
              100 * a.exact, 100 * a.within1, a.max_scaled, a.max_ulp, a.cosine, a.update_cosine);
  // Bit-exact on these inputs (measured 100 %). Elsewhere attention (FA2 here, SDPA in the reference) can differ in last
  // bits, and the o-proj / ffn_down inputs are re-quantized after it, where a 1-ULP difference moves an INT4 value by a
  // step (absmax / 7): a change of GPU, driver or attention kernel may need this gate revisited.
  // Recorded on sm_89; on sm_120 the attention's last bits differ (tensor-core rounding): 99.19 % bit-exact, cos 0.99998.
  if (mi::testing::on_reference_gpu()) {
    EXPECT_EQ(a.exact, 1.0);
    EXPECT_GE(a.cosine, 0.99999);
  } else {
    EXPECT_GE(a.cosine, 0.99995);
  }
  EXPECT_GE(a.update_cosine, 0.9999);

  // The INT4 block itself against the bf16 block (same input): the Turbo tier's own loss at this block, for scale.
  const auto bf16 = run_block(w16_, xv, kText);
  const Agreement l = agree(bf16, run_block(w4_, xv, kText), xv);
  std::printf("[dit_int4] all-INT4 block vs the bf16 block: cos %.6f, update cos %.5f\n", l.cosine, l.update_cosine);

  // 8-bit attention (lossy): against the same exact-attention INT4 reference. The o-proj and ffn_down inputs are
  // re-quantized to INT4 after attention, so the attention's small differences move many INT4 values by a step
  // (absmax / 7): the block update differs by more than on the INT8 block (reported against the bf16 block too).
  for (const md::AttentionMode mode : kQ8Modes) {
    if (!md::attention_supported(mode)) continue;
    const auto got = run_block(w4_, xv, kText, mode);
    const Agreement q = agree(ref, got, xv), qb = agree(bf16, got, xv);
    std::printf("[dit_int4] all-INT4 block + %s attention vs reference: %.4f%% bit-exact, cos %.6f, update cos %.5f; vs "
                "the bf16 block: update cos %.5f\n",
                md::attention_mode_name(mode), 100 * q.exact, q.cosine, q.update_cosine, qb.update_cosine);
    EXPECT_GE(q.cosine, 0.98) << md::attention_mode_name(mode);
  }
}

// A block mixing all three precisions, as a per-layer mixed file would: q INT4, k INT8, v bf16 (split QKV), o bf16,
// ffn_up_gate INT4, ffn_down INT8.
TEST_F(DitInt4Block, MixedPrecisionBlockMatchesReference) {
  md::BlockTensors w = w4_;
  const auto rows = [](const Tensor& t, std::int64_t r0, std::int64_t r1) { return t.slice0(r0, r1); };
  w.qkv4 = {};
  w.qkv_split = true;
  w.qkv4_part[0] = {rows(w4_.qkv4.w, 0, kDim), rows(w4_.qkv4.scale, 0, kDim)};
  w.qkv8_part[1] = {rows(w8_.qkv8.w, kDim, 2 * kDim), rows(w8_.qkv8.scale, kDim, 2 * kDim)};
  w.qkv_part[2] = rows(w16_.qkv, 2 * kDim, 3 * kDim);
  w.o4 = {};
  w.o = w16_.o;
  w.down4 = {};
  w.down8 = w8_.down8;
  ASSERT_TRUE(w.int4());
  const auto xv = load_u16("mixed_x");
  const auto ref = load_u16("mixed_y");
  const Agreement a = agree(ref, run_block(w, xv, std::span<const std::int64_t>(kText, 1)), xv);
  std::printf("[dit_int4] mixed INT4/INT8/bf16 block vs reference: %.4f%% bit-exact, %.4f%% within 1 ULP of scale (max "
              "%.2f), max %d raw ULP, cos %.9f, update cos %.7f\n",
              100 * a.exact, 100 * a.within1, a.max_scaled, a.max_ulp, a.cosine, a.update_cosine);
  // Off sm_89 the bf16 linears also run other cuBLASLt algorithms than torch did (sm_120: 76 % within 1 ULP, cos 0.99997).
  if (mi::testing::on_reference_gpu()) {
    EXPECT_GE(a.cosine, 0.99999);
    EXPECT_GE(a.within1, 0.99);
  } else {
    EXPECT_GE(a.cosine, 0.99995);
  }
  EXPECT_GE(a.update_cosine, 0.9999);
}

// `big` sized exactly as forward() sizes it for an INT4 file (kFusedBigCols per token): the same bits as with the wide
// buffer, and nothing written past it.
TEST_F(DitInt4Block, TightFusedBigBuffer) {
  const auto xv = hash_x();
  Run wide(xv, kText, stream()), tight(xv, kText, stream());
  const std::int64_t l = tight.l;
  constexpr std::int64_t kGuard = 4096;
  const std::size_t elems = static_cast<std::size_t>(l * md::kFusedBigCols + kGuard);
  mi::DeviceBuffer big(elems * 2);
  const std::vector<std::uint16_t> sentinel(static_cast<std::size_t>(kGuard), 0x7F81);
  MI_CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(big.get()) + l * md::kFusedBigCols, sentinel.data(), kGuard * 2,
                           cudaMemcpyHostToDevice));
  tight.qa.f.big = Tensor::view(big.get(), DType::kBF16, mi::Device::kCUDA, {l, md::kFusedBigCols});
  md::run_block_quant(w4_, wide.qa, wide.run, wide.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream());
  md::run_block_quant(w4_, tight.qa, tight.run, tight.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(download<std::uint16_t>(tight.x->t), download<std::uint16_t>(wide.x->t));
  std::vector<std::uint16_t> after(static_cast<std::size_t>(kGuard));
  MI_CUDA_CHECK(cudaMemcpy(after.data(), static_cast<std::uint16_t*>(big.get()) + l * md::kFusedBigCols, kGuard * 2,
                           cudaMemcpyDeviceToHost));
  EXPECT_EQ(after, sentinel);
}

// GPUs without INT4 MMA run every INT4 linear through the INT8 GEMMs on unpacked operands: forced here, it gives the
// native INT4 path's bits exactly, for every site, the split QKV of a mixed block, 8-bit attention, a side branch and
// a `big` sized as forward() sizes it.
TEST_F(DitInt4Block, ViaInt8EqualsNativeInt4) {
  EXPECT_EQ(md::int4_mma_native(8, 9), true);
  EXPECT_EQ(md::int4_mma_native(8, 0), true);
  EXPECT_EQ(md::int4_mma_native(9, 0), false);
  EXPECT_EQ(md::int4_mma_native(10, 0), false);
  EXPECT_EQ(md::int4_mma_native(12, 0), false);
  const auto xv = load_u16("all_int4_x");
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    if (!md::attention_supported(mode)) continue;
    EXPECT_EQ(run_block(w4_, xv, kText, mode, nullptr, true), run_block(w4_, xv, kText, mode)) << md::attention_mode_name(mode);
  }
  // Split QKV: q INT4, k INT8, v bf16; o bf16; up_gate INT4; down INT8.
  md::BlockTensors w = w4_;
  const auto rows = [](const Tensor& t, std::int64_t r0, std::int64_t r1) { return t.slice0(r0, r1); };
  w.qkv4 = {};
  w.qkv_split = true;
  w.qkv4_part[0] = {rows(w4_.qkv4.w, 0, kDim), rows(w4_.qkv4.scale, 0, kDim)};
  w.qkv8_part[1] = {rows(w8_.qkv8.w, kDim, 2 * kDim), rows(w8_.qkv8.scale, kDim, 2 * kDim)};
  w.qkv_part[2] = rows(w16_.qkv, 2 * kDim, 3 * kDim);
  w.o4 = {};
  w.o = w16_.o;
  w.down4 = {};
  w.down8 = w8_.down8;
  const auto mx = load_u16("mixed_x");
  const std::span<const std::int64_t> one(kText, 1);
  EXPECT_EQ(run_block(w, mx, one, md::AttentionMode::kExact, nullptr, true), run_block(w, mx, one));
  // With a side branch on every site.
  const mi::lora::LoraSpec spec{data_dir() / "lora_block5.safetensors", 1.0f};
  mi::lora::LoraStack stack(std::span<const mi::lora::LoraSpec>(&spec, 1), mi::lora::moss_dit_targets());
  mi::lora::SideBranches sb(stack);
  EXPECT_EQ(run_block(w4_, xv, kText, md::AttentionMode::kExact, &sb, true),
            run_block(w4_, xv, kText, md::AttentionMode::kExact, &sb));
  // `big` of kFusedBigCols per token, a guard after it.
  Run tight(xv, kText, stream(), nullptr, true);
  const std::int64_t l = tight.l;
  constexpr std::int64_t kGuard = 4096;
  mi::DeviceBuffer big(static_cast<std::size_t>(l * md::kFusedBigCols + kGuard) * 2);
  const std::vector<std::uint16_t> sentinel(static_cast<std::size_t>(kGuard), 0x7F81);
  MI_CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(big.get()) + l * md::kFusedBigCols, sentinel.data(), kGuard * 2,
                           cudaMemcpyHostToDevice));
  tight.qa.f.big = Tensor::view(big.get(), DType::kBF16, mi::Device::kCUDA, {l, md::kFusedBigCols});
  md::run_block_quant(w4_, tight.qa, tight.run, tight.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream());
  MI_CUDA_CHECK(cudaStreamSynchronize(stream()));
  EXPECT_EQ(download<std::uint16_t>(tight.x->t), run_block(w4_, xv, kText));
  std::vector<std::uint16_t> after(static_cast<std::size_t>(kGuard));
  MI_CUDA_CHECK(cudaMemcpy(after.data(), static_cast<std::uint16_t*>(big.get()) + l * md::kFusedBigCols, kGuard * 2,
                           cudaMemcpyDeviceToHost));
  EXPECT_EQ(after, sentinel);
  // Without the unpack buffer: refused before any launch.
  Run bare(xv, kText, stream(), nullptr, true);
  bare.qa.int4_unpack = nullptr;
  EXPECT_THROW(md::run_block_quant(w4_, bare.qa, bare.run, bare.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream()),
               std::invalid_argument);
}

// A kohya LoRA with B = 0 on every GEMM of block kBlock (rank 16): an active side branch adding exact zeros.
fs::path lora_dir() {
  static const fs::path dir = fs::temp_directory_path() / ("mossinfer_test_dit_int4_" + std::to_string(std::random_device{}()));
  return dir;
}

fs::path write_zero_lora() {
  const fs::path file = lora_dir() / "zero.safetensors";
  constexpr std::int64_t kRank = 16;
  const struct {
    const char* name;
    std::int64_t n, k;
  } mods[] = {{"q", kDim, kDim}, {"k", kDim, kDim}, {"v", kDim, kDim}, {"o", kDim, kDim},
              {"ffn_up_gate", 2 * md::kFfnDim, kDim}, {"ffn_down", kDim, md::kFfnDim}};
  std::vector<tl::StTensor> ts;
  std::uint32_t seed = 11;
  for (const auto& m : mods) {
    const std::string p = "diffusion_model.layers." + std::to_string(kBlock) + ".proj." + m.name + ".";
    ts.push_back({p + "lora_down.weight", mi::weights::DType::kF32, {kRank, m.k}, tl::randn(static_cast<std::size_t>(kRank * m.k), seed++, 0.02f)});
    ts.push_back({p + "lora_up.weight", mi::weights::DType::kF32, {m.n, kRank}, std::vector<float>(static_cast<std::size_t>(m.n * kRank), 0.0f)});
    ts.push_back({p + "alpha", mi::weights::DType::kF32, {}, {static_cast<float>(kRank)}});
  }
  tl::write_st(file, ts);
  return file;
}

// Runtime LoRA on INT4 sites (the INT8 LoRA GEMMs on the operands unpacked to INT8).
TEST_F(DitInt4Block, LoraOnInt4Sites) {
  {
    const auto hx = hash_x();
    const auto plain = run_block(w4_, hx, kText);
    // B = 0: every site active, Δ exactly 0: the INT8 LoRA GEMMs on the unpacked operands give the INT4 GEMMs' bits.
    const mi::lora::LoraSpec spec{write_zero_lora(), 1.0f};
    mi::lora::LoraStack stack(std::span<const mi::lora::LoraSpec>(&spec, 1), mi::lora::moss_dit_targets());
    mi::lora::SideBranches sb(stack);
    for (const mi::lora::Site s : {mi::lora::Site::kQkv, mi::lora::Site::kO, mi::lora::Site::kUpGate, mi::lora::Site::kDown}) {
      ASSERT_NE(sb.site(kBlock, s), nullptr) << mi::lora::site_name(s);
    }
    EXPECT_EQ(run_block(w4_, hx, kText, md::AttentionMode::kExact, &sb), plain);
    // Without the unpack buffer the block refuses before any launch.
    Run r(hx, kText, stream(), &sb);
    r.qa.int4_unpack = nullptr;
    EXPECT_THROW(md::run_block_quant(w4_, r.qa, r.run, r.qkv_gemm(), fused_route_, md::AttentionMode::kExact, stream(), kBlock),
                 std::invalid_argument);
  }
  // A rank-8 LoRA on all six linears (tests/data/moss_dit_int4/lora_block5.safetensors) against the reference block
  // with each INT4 linear adding the side branch in fp32 before its rounding (lora_y), on the stored inputs. Not
  // bit-exact everywhere: the kernel accumulates Δ into the dequantized base one MMA at a time, the reference adds a
  // separately summed Δ, and a last-bit difference before an INT4 re-quantization can move a value by a step.
  const auto xv = load_u16("all_int4_x");
  const auto plain4 = run_block(w4_, xv, kText);
  const mi::lora::LoraSpec spec{data_dir() / "lora_block5.safetensors", 1.0f};
  mi::lora::LoraStack stack(std::span<const mi::lora::LoraSpec>(&spec, 1), mi::lora::moss_dit_targets());
  mi::lora::SideBranches sb(stack);
  const auto got = run_block(w4_, xv, kText, md::AttentionMode::kExact, &sb);
  const Agreement a = agree(load_u16("lora_y"), got, xv);
  const Agreement plain_vs = agree(load_u16("lora_y"), plain4, xv);
  std::printf("[dit_int4] all-INT4 block + rank-8 LoRA vs reference: %.4f%% bit-exact, %.4f%% within 1 ULP of scale (max "
              "%.2f), cos %.9f, update cos %.7f (the plain INT4 block vs this reference: update cos %.5f)\n",
              100 * a.exact, 100 * a.within1, a.max_scaled, a.cosine, a.update_cosine, plain_vs.update_cosine);
  EXPECT_GE(a.exact, 0.97);
  EXPECT_GE(a.cosine, 0.9999);
  EXPECT_GE(a.update_cosine, 0.9999);
  EXPECT_GE(a.within1, 0.97);
  std::error_code ec;
  fs::remove_all(lora_dir(), ec);
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

md::Residency fitting(const mi::weights::ModelIndex& index, std::size_t work, std::size_t free) {
  md::Residency r;
  while (r.resident_blocks > 0 &&
         md::MossDit::persistent_bytes(index, r) + work + md::MossDit::stream_workspace_bytes(index, r) > free) {
    --r.resident_blocks;
  }
  return r;
}

class DitInt4Model : public ::testing::Test {
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
    }
    quant_ = std::make_unique<mi::quant::QuantIndex>(int4_index());
    // Every attention mode and INT4 route the tests switch to (the via-INT8 route adds the unpack buffer).
    work_ = std::max({md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFusedGraphs,
                                                   md::MossDit::compute_config(quant_.get(), md::AttentionMode::kQ8Fp8,
                                                                               md::Int4Route::kViaInt8), nullptr),
                      md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFusedGraphs,
                                                   md::MossDit::compute_config(quant_.get(), md::AttentionMode::kExact,
                                                                               md::Int4Route::kViaInt8), nullptr),
                      md::MossDit::encode_text_workspace_bytes(std::max(c.s_cond, c.s_uncond))});
    persistent_ = md::MossDit::persistent_bytes(quant_->index);
    arena_ = std::make_unique<mi::runtime::DeviceArena>(persistent_ + work_);
    model_ = std::make_unique<md::MossDit>(*quant_, *arena_, *streams_);
    io_ = std::make_unique<ModelIo>(*model_, c, s);
    std::printf("[dit_int4] INT4 model: %.3f GiB persistent, %.3f GiB forward workspace, loaded in %.2f s\n",
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
    if (!model_) GTEST_SKIP() << "INT4 DiT file, SFT bundle or goldens absent";
    model_->set_block_path(md::BlockPath::kFused);
    model_->set_attention(md::AttentionMode::kExact);
    model_->set_int4_route(md::Int4Route::kAuto);
  }
  cudaStream_t stream() const { return streams_->compute(); }
  std::vector<md::Row> cfg_rows() const { return {{io_->cond->t, {}}, {io_->uncond->t, case_->uncond_skip}}; }

  static std::unique_ptr<GoldenCase> case_;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::vector<std::vector<float>> bf16_v_;
  static std::unique_ptr<mi::quant::QuantIndex> quant_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
  static std::unique_ptr<md::MossDit> model_;
  static std::unique_ptr<ModelIo> io_;
  static std::size_t work_, persistent_;
};

std::unique_ptr<GoldenCase> DitInt4Model::case_;
std::unique_ptr<mi::runtime::StreamSet> DitInt4Model::streams_;
std::vector<std::vector<float>> DitInt4Model::bf16_v_;
std::unique_ptr<mi::quant::QuantIndex> DitInt4Model::quant_;
std::unique_ptr<mi::runtime::DeviceArena> DitInt4Model::arena_;
std::unique_ptr<md::MossDit> DitInt4Model::model_;
std::unique_ptr<ModelIo> DitInt4Model::io_;
std::size_t DitInt4Model::work_ = 0, DitInt4Model::persistent_ = 0;

TEST_F(DitInt4Model, StepVelocityVsBf16Engine) {
  const md::ComputeConfig cc = model_->compute_config();
  EXPECT_TRUE(cc.quantized);
  EXPECT_TRUE(cc.int4);
  EXPECT_FALSE(cc.bf16_up_gate);
  const auto rows = cfg_rows();
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    model_->set_attention(mode);
    const auto v = io_->forward(*model_, rows, case_->timestep, stream());
    const double c0 = cosine(v[0], bf16_v_[0]), c1 = cosine(v[1], bf16_v_[1]);
    std::printf("[dit_int4] step-0 velocity INT4 + %s attention vs bf16 engine (1024^2): cos cond %.6f, uncond %.6f\n",
                md::attention_mode_name(mode), c0, c1);
    EXPECT_GE(c0, 0.95);
    EXPECT_GE(c1, 0.95);
  }
}

// The model's INT4 route: through the INT8 GEMMs equals native INT4 bit for bit (eager and CUDA graphs), and the
// forward stack follows the route (the unpack buffer).
TEST_F(DitInt4Model, Int4RouteViaInt8EqualsNative) {
  const auto rows = cfg_rows();
  const GoldenCase& c = *case_;
  const std::int64_t lengths[] = {c.s_cond, c.s_uncond};
  for (const md::AttentionMode mode : {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8}) {
    model_->set_attention(mode);
    model_->set_int4_route(md::Int4Route::kNative);
    EXPECT_FALSE(model_->compute_config().int4_via_int8);
    const std::size_t native_ws = md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFused, model_->compute_config(), nullptr);
    const auto native = io_->forward(*model_, rows, case_->timestep, stream());
    model_->set_int4_route(md::Int4Route::kViaInt8);
    EXPECT_TRUE(model_->compute_config().int4_via_int8);
    EXPECT_EQ(md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFused, model_->compute_config(), nullptr),
              native_ws + mi::runtime::align_up(md::kInt4UnpackBytes));
    const auto via = io_->forward(*model_, rows, case_->timestep, stream());
    model_->set_block_path(md::BlockPath::kFusedGraphs);
    std::vector<std::vector<float>> graphs;
    for (int i = 0; i < 3; ++i) graphs = io_->forward(*model_, rows, case_->timestep, stream());
    model_->set_block_path(md::BlockPath::kFused);
    for (std::size_t r = 0; r < rows.size(); ++r) {
      EXPECT_EQ(differing(via[r], native[r]), 0u) << md::attention_mode_name(mode) << " row " << r;
      EXPECT_EQ(differing(graphs[r], native[r]), 0u) << md::attention_mode_name(mode) << " graphs row " << r;
    }
  }
  model_->set_int4_route(md::Int4Route::kAuto);
  EXPECT_EQ(model_->compute_config().int4_via_int8, !md::int4_mma_native());
}

TEST_F(DitInt4Model, VarlenBatchBitExactPerRow) {
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
      EXPECT_EQ(differing(alone[0], batched[r]), 0u) << md::attention_mode_name(mode) << " row " << r;
    }
  }
}

TEST_F(DitInt4Model, GraphsBitIdenticalToEager) {
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

TEST_F(DitInt4Model, StreamedBitIdenticalToResident) {
  const auto rows = cfg_rows();
  const auto want = io_->forward(*model_, rows, case_->timestep, stream());
  const GoldenCase& c = *case_;
  const std::int64_t lengths[] = {c.s_cond, c.s_uncond};
  const std::size_t work = std::max(md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFused,
                                                                 model_->compute_config(), nullptr),
                                    md::MossDit::encode_text_workspace_bytes(std::max(c.s_cond, c.s_uncond)));
  const mi::weights::ModelIndex bundle = mi::weights::load_index(sft_dir() / "moss.mossidx", sft_dir());
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
    std::uint64_t want_bytes = 0, bf16_bytes = 0;
    const auto mask = md::resident_mask(res.resident_blocks);
    for (int b = 0; b < md::kNumBlocks; ++b) {
      if (mask.test(static_cast<std::size_t>(b))) continue;
      const std::string g = "block." + std::to_string(b);
      want_bytes += quant_->index.find_group(mi::weights::Component::kTransformer, g)->total_bytes;
      bf16_bytes += bundle.find_group(mi::weights::Component::kTransformer, g)->total_bytes;
    }
    std::printf("[dit_int4] streamed (%s, 10 of 42 resident): %.2f GB per step (bf16: %.2f GB, ratio %.3f) at %.1f GB/s, "
                "%.0f%% hidden\n", md::stream_source_name(source), static_cast<double>(st.bytes) / 1e9,
                static_cast<double>(bf16_bytes) / 1e9, static_cast<double>(st.bytes) / static_cast<double>(bf16_bytes),
                st.gbps(), 100 * st.hidden());
    EXPECT_EQ(st.bytes, want_bytes);
    EXPECT_LT(static_cast<double>(st.bytes) / static_cast<double>(bf16_bytes), 0.27);
    for (std::size_t r = 0; r < 2; ++r) EXPECT_EQ(differing(got[r], want[r]), 0u) << md::stream_source_name(source) << " row " << r;
  }
}

// Dev numbers: one CFG step at 1024^2 with 282 + 10 text tokens (uncond skipping block 22), every block resident, for
// INT4 (exact and q8_fp8 attention, toxicdog's file), INT8 (exact, Comfy-Org's file) and the bf16 fused engine when it
// fits, in one process. Median of 7 steps after 2 warm-ups; timings only (other processes share the GPU).
TEST(DitInt4Speed, Step1024) {
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
  // Each model alone in its own arena; attention modes interleaved per model.
  const auto measure = [&](const mi::quant::QuantIndex* qi, const mi::weights::ModelIndex& index,
                           std::span<const md::AttentionMode> modes, md::Int4Route route = md::Int4Route::kAuto) {
    std::size_t work = 0;
    for (const md::AttentionMode a : modes) {
      work = std::max(work, md::MossDit::workspace_bytes(c.h, c.w, lengths, md::BlockPath::kFused,
                                                         md::MossDit::compute_config(qi, a, route), nullptr));
    }
    std::vector<double> med(modes.size(), -1.0);
    if (md::MossDit::persistent_bytes(index) + work > free) return med;
    mi::runtime::DeviceArena arena(md::MossDit::persistent_bytes(index) + work);
    std::unique_ptr<md::MossDit> m = qi != nullptr ? std::make_unique<md::MossDit>(*qi, arena, streams)
                                                   : std::make_unique<md::MossDit>(index, arena, streams);
    m->set_int4_route(route);
    std::vector<std::vector<double>> t(modes.size());
    for (int i = 0; i < 9; ++i) {
      for (std::size_t j = 0; j < modes.size(); ++j) {
        m->set_attention(modes[j]);
        const double dt = step(*m);
        if (i >= 2) t[j].push_back(dt);
      }
    }
    for (std::size_t j = 0; j < modes.size(); ++j) med[j] = median(t[j]);
    return med;
  };
  const md::AttentionMode both[] = {md::AttentionMode::kExact, md::AttentionMode::kQ8Fp8};
  const md::AttentionMode exact[] = {md::AttentionMode::kExact};
  const mi::quant::QuantIndex q4 = int4_index();
  const mi::quant::QuantIndex q8 = mi::quant::to_model_index(mi::quant::open_quant_model(int8_file(), mi::quant::ModelKind::kDiT));
  const mi::weights::ModelIndex bundle = mi::weights::load_index(sft_dir() / "moss.mossidx", sft_dir());
  const auto t16 = measure(nullptr, bundle, exact);
  const auto t8 = measure(&q8, q8.index, both);
  const auto t4 = measure(&q4, q4.index, both, md::Int4Route::kNative);
  const auto t4v = measure(&q4, q4.index, both, md::Int4Route::kViaInt8);
  std::printf("[dit_int4 speed, dev] 1024^2 CFG step (282+10 tokens), INT4 MMA native on this GPU: %s\n"
              "  bf16 fused %.3f s; INT8 + exact %.3f s, + q8_fp8 %.3f s; INT4 native + exact %.3f s, + q8_fp8 %.3f s; "
              "INT4 via INT8 + exact %.3f s, + q8_fp8 %.3f s (INT4 native vs INT8 exact %.2fx, vs bf16 %.2fx)\n",
              md::int4_mma_native() ? "yes" : "no", t16[0], t8[0], t8[1], t4[0], t4[1], t4v[0], t4v[1], t8[0] / t4[0],
              t16[0] > 0 ? t16[0] / t4[0] : 0.0);
  RecordProperty("int4_via_int8_exact_step_s", std::to_string(t4v[0]));
  RecordProperty("int4_exact_step_s", std::to_string(t4[0]));
  RecordProperty("int4_q8_step_s", std::to_string(t4[1]));
  RecordProperty("int8_exact_step_s", std::to_string(t8[0]));
  EXPECT_GT(t4[0], 0.0);
}

}  // namespace
