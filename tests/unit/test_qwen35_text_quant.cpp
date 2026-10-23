// Quantized Qwen3.5 text encoder (P6 step 38, AC23): Comfy-Org's INT8 ConvRot and W4A8 files.
//
// - The GPU W4A8 / W6A8 decoder is bit-exact to the CPU decoder (random codes, every group size, invalid scales).
// - Engine vs a torch reference of the same quantized math (transformers' layers with comfy-kitchen's eager INT8 /
//   W4A8 linears; tests/golden/qwen35_text_quant/<fmt>/, made by the local generator script), held to the AC4 method
//   with the quantized model's own floor (see GoldenParity). These end-to-end bounds are coarse (32 layers of amplified
//   rounding flips); the teacher-forced per-layer check (TeacherForcedPerLayer) is the sharp one, and it is shown to
//   catch a 10 % scale error in one linear.
//   Known engine-vs-reference differences, all at rounding level: the GEMM epilogue computes (acc * s_x) * s_w (the
//   order of comfy-kitchen's CUDA epilogue) where the eager reference computes acc * (s_x * s_w), which differs by at
//   most 1 bf16 ULP on rare elements; the end-to-end reference rotates with eager's bf16 matmul and runs in_proj_a/b
//   with torch's reduced-precision bf16 reduction (the per-layer reference uses the CUDA rotation order and exact
//   bf16 GEMMs, as the engine does); attention and the GatedDeltaNet rule are different kernels.
// - AC23, quantized vs the bf16 goldens: measured and reported per sequence against the bf16 floor; asserted relative
//   to the torch quantized reference's own distance from the goldens.
// - Varlen batching and every streaming / host-resident configuration give the same bits.
// - Perf (prompt + negative 282+10 and 4096+16 tokens): bf16 streamed vs INT8 vs W4A8, file-streamed and
//   RAM-resident. MOSSINFER_TE_COLD=1 adds cold page-cache (embedding-cache miss from disk) rows.
//
// Files: MOSS_TE_INT8 / MOSS_TE_W4A8 override the default paths; each test skips when its file is absent.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/quant/w4a8_decode.h"
#include "models/qwen35_text/quant_text_model.h"
#include "models/qwen35_text/text_encoder.h"
#include "reference_gpu.h"
#include "runtime/device_arena.h"
#include "runtime/stream_set.h"
#include "util/json.h"
#include "util/npy.h"
#include "weights/model_index.h"
#include "weights/quant/quant_codec.h"
#include "weights/quant/quant_model.h"

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
using mi::qwen35_text::EncoderOptions;
using mi::qwen35_text::kHidden;
using mi::qwen35_text::Precision;
using mi::qwen35_text::QuantTextModel;
using mi::qwen35_text::TextEncoder;
using mi::qwen35_text::WeightSource;
using Ids = std::vector<std::int32_t>;
using Bf16 = std::vector<std::uint16_t>;

const fs::path kGolden = fs::path(MOSSINFER_SOURCE_DIR) / "tests/golden";
const fs::path kNoiseFloor = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/qwen35_text/noise_floor.json";
const fs::path kQuantNoiseFloor = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/qwen35_text/quant_noise_floor.json";
// As AC4: two perturbation sources at once; 2.5x off sm_89, where the floors were measured (sm_120: up to 2.09x).
double floor_factor() { return mi::testing::on_reference_gpu() ? 2.0 : 2.5; }
constexpr double kEarlyMinCosine = 0.99998;   // AC4 layers 0 and 3
constexpr double kEarlyMaxRowRelL2 = 2e-2;

fs::path env_path(const char* var, const char* fallback) {
  const char* env = std::getenv(var);
  return env != nullptr && *env != '\0' ? fs::path(env) : fs::path(fallback);
}
fs::path int8_file() { return env_path("MOSS_TE_INT8", "/workspace/Models/community/comfy-org/qwen3.5_9b_int8_convrot.safetensors"); }
fs::path w4a8_file() { return env_path("MOSS_TE_W4A8", "/workspace/Models/community/comfy-org/qwen3.5_9b_w4a8_gptq.safetensors"); }
fs::path bundle_dir() { return env_path("MOSS_WEIGHTS", "/workspace/Models/MOSS-Image") / "base"; }

float bf16_to_f32(std::uint16_t b) { return mi::quant::bf16_to_float(b); }

mi::json::Value read_json(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return mi::json::parse(ss.str());
}

Ids golden_ids(const std::string& key) {
  const auto v = mi::npy::load(kGolden / (key + ".token_ids.npy")).to_vector<std::int64_t>();
  return Ids(v.begin(), v.end());
}

std::vector<std::string> golden_keys() {
  std::vector<std::string> keys;
  const mi::json::Value cases = read_json(kGolden / "cases.json");
  for (const auto& c : cases.as_array()) {
    for (const char* b : {"cond", "uncond"}) keys.push_back(c.at("id").as_string() + "." + b);
  }
  return keys;
}

std::vector<std::int64_t> lengths_of(const std::vector<Ids>& seqs) {
  std::vector<std::int64_t> l;
  for (const auto& s : seqs) l.push_back(static_cast<std::int64_t>(s.size()));
  return l;
}

struct Metrics {
  double cosine = 0, min_row_cosine = 1, max_row_rel_l2 = 0, mean_row_rel_l2 = 0;
};

Metrics compare(const std::uint16_t* got, const std::vector<float>& ref) {
  Metrics m;
  double dot = 0, gg = 0, rr = 0;
  const std::size_t rows = ref.size() / kHidden;
  for (std::size_t r = 0; r < rows; ++r) {
    double rdot = 0, rg = 0, rref = 0, rdiff = 0;
    for (std::size_t c = 0; c < kHidden; ++c) {
      const double g = bf16_to_f32(got[r * kHidden + c]), w = ref[r * kHidden + c];
      rdot += g * w;
      rg += g * g;
      rref += w * w;
      rdiff += (g - w) * (g - w);
    }
    dot += rdot;
    gg += rg;
    rr += rref;
    m.min_row_cosine = std::min(m.min_row_cosine, rdot / std::sqrt(rg * rref));
    m.max_row_rel_l2 = std::max(m.max_row_rel_l2, std::sqrt(rdiff / rref));
    m.mean_row_rel_l2 += std::sqrt(rdiff / rref) / static_cast<double>(rows);
  }
  m.cosine = dot / std::sqrt(gg * rr);
  return m;
}

struct Floor {
  double cosine = 1, mean_row_rel_l2 = 0;
};

Floor noise_floor(const mi::json::Value& nf, const std::string& key, const std::string& checkpoint) {
  Floor f;
  const mi::json::Value* seq = nf.at("sequences").find(key);
  if (seq == nullptr) throw std::runtime_error("noise floor has no sequence " + key);
  for (const char* variant : {"math_sdpa", "cublaslt"}) {
    const mi::json::Value& m = seq->at(variant).at(checkpoint);
    f.cosine = std::min(f.cosine, m.at("cosine").as_double());
    f.mean_row_rel_l2 = std::max(f.mean_row_rel_l2, m.at("mean_row_rel_l2").as_double());
  }
  return f;
}

std::vector<float> load_f32(const fs::path& p, std::int64_t rows) {
  const mi::npy::Array a = mi::npy::load(p);
  if (a.shape != std::vector<std::int64_t>{rows, kHidden}) throw std::runtime_error("unexpected shape: " + p.string());
  return a.to_vector<float>();
}

void print(const std::string& what, const Metrics& m, const Floor& f) {
  std::printf("  %-34s cos %.7f (1-cos %.2e = %5.2fx floor)  row relL2 mean %.3e (%5.2fx floor) max %.3e  min row cos %.6f\n",
              what.c_str(), m.cosine, 1 - m.cosine, (1 - m.cosine) / std::max(1e-12, 1 - f.cosine), m.mean_row_rel_l2,
              m.mean_row_rel_l2 / std::max(1e-12, f.mean_row_rel_l2), m.max_row_rel_l2, m.min_row_cosine);
}

void expect_identical(const std::string& what, const Bf16& a, const Bf16& b) {
  ASSERT_EQ(a.size(), b.size()) << what;
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
  EXPECT_EQ(diff, 0u) << what << ": " << diff << " of " << a.size() << " bf16 values differ";
}

// ---- GPU decoder vs CPU decoder ------------------------------------------------------------------------------

struct DecodeCase {
  int bits, group;
  bool codebook;
};

void run_decode_case(const DecodeCase& dc, std::int64_t n, std::int64_t k, std::mt19937& rng, bool poison) {
  const std::int64_t row_bytes = k * dc.bits / 8, groups = k / dc.group;
  std::vector<std::uint8_t> packed(static_cast<std::size_t>(n * row_bytes)), s_rel(static_cast<std::size_t>(n * groups));
  std::uniform_int_distribution<int> byte(0, 255);
  for (auto& b : packed) b = static_cast<std::uint8_t>(byte(rng));
  for (auto& b : s_rel) {
    // Positive e4m3 values: zero, subnormals and normals up to 448 (0x7E); never NaN (0x7F).
    b = static_cast<std::uint8_t>(byte(rng) % 0x7F);
  }
  s_rel[1] = 0x80;  // -0: valid, decodes to zeros
  std::vector<float> cb(16);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  for (auto& v : cb) v = nd(rng);
  cb[3] = 1e30f;  // overflows to +-inf before the clamp
  if (poison) {
    s_rel[static_cast<std::size_t>(groups + 2)] = 0x7F;  // NaN
    s_rel[static_cast<std::size_t>(2 * groups)] = 0x88;  // negative
  }
  std::vector<std::int8_t> want(static_cast<std::size_t>(n * k));
  const bool ok = mi::quant::decode_grouped_rows(packed.data(), s_rel.data(), dc.codebook ? cb.data() : nullptr, n, k, dc.bits,
                                                 dc.group, want.data());
  ASSERT_EQ(ok, !poison);

  std::uint8_t *d_packed = nullptr, *d_srel = nullptr;
  float* d_cb = nullptr;
  std::int8_t* d_out = nullptr;
  unsigned* d_flag = nullptr;
  MI_CUDA_CHECK(cudaMalloc(&d_packed, packed.size()));
  MI_CUDA_CHECK(cudaMalloc(&d_srel, s_rel.size()));
  MI_CUDA_CHECK(cudaMalloc(&d_cb, 16 * sizeof(float)));
  MI_CUDA_CHECK(cudaMalloc(&d_out, want.size()));
  MI_CUDA_CHECK(cudaMalloc(&d_flag, sizeof(unsigned)));
  MI_CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemcpy(d_srel, s_rel.data(), s_rel.size(), cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemcpy(d_cb, cb.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(unsigned)));
  mi::kernels::quant::GroupedCodes c;
  c.packed = d_packed;
  c.s_rel = d_srel;
  c.codebook = dc.codebook ? d_cb : nullptr;
  c.n = n;
  c.k = k;
  c.bits = dc.bits;
  c.group = dc.group;
  mi::kernels::quant::decode_grouped_int8(c, d_out, d_flag, nullptr);
  std::vector<std::int8_t> got(want.size());
  unsigned flag = 0;
  MI_CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size(), cudaMemcpyDeviceToHost));
  MI_CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(flag), cudaMemcpyDeviceToHost));
  for (void* p : {static_cast<void*>(d_packed), static_cast<void*>(d_srel), static_cast<void*>(d_cb), static_cast<void*>(d_out),
                  static_cast<void*>(d_flag)}) {
    MI_CUDA_CHECK(cudaFree(p));
  }
  const std::string what = "bits " + std::to_string(dc.bits) + " group " + std::to_string(dc.group) + (dc.codebook ? " codebook" : "");
  EXPECT_EQ(flag != 0, poison) << what;
  if (poison) {
    // The poisoned groups decode to zeros; every other group as the CPU decoder (which stops at the first error,
    // so it is re-run without the poison for the reference).
    for (int j = 0; j < dc.group; ++j) {
      EXPECT_EQ(got[static_cast<std::size_t>(k + 2 * dc.group + j)], 0) << what;
      EXPECT_EQ(got[static_cast<std::size_t>(2 * k + j)], 0) << what;
    }
    return;
  }
  std::size_t diff = 0;
  for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != want[i];
  EXPECT_EQ(diff, 0u) << what << ": " << diff << " of " << got.size() << " int8 values differ";
}

TEST(Qwen35TextQuantDecode, GpuDecoderBitExactToCpu) {
  std::mt19937 rng(1234);
  const DecodeCase cases[] = {{4, 16, true},  {4, 16, false}, {4, 4, true},  {4, 8, false},
                              {4, 32, true},  {4, 64, false}, {6, 16, false}, {6, 32, false}};
  for (const DecodeCase& dc : cases) {
    run_decode_case(dc, 37, 512, rng, false);
    run_decode_case(dc, 37, 512, rng, true);
  }
}

TEST(Qwen35TextQuantDecode, RejectsInvalidGeometry) {
  mi::kernels::quant::GroupedCodes c;
  std::int8_t* out = reinterpret_cast<std::int8_t*>(std::uintptr_t{256});
  unsigned* flag = reinterpret_cast<unsigned*>(std::uintptr_t{256});
  c.packed = reinterpret_cast<const std::uint8_t*>(std::uintptr_t{256});
  c.s_rel = c.packed;
  c.n = 4;
  c.k = 24;  // not a multiple of 16
  EXPECT_THROW(mi::kernels::quant::decode_grouped_int8(c, out, flag, nullptr), std::invalid_argument);
  c.k = 32;
  c.group = 12;
  EXPECT_THROW(mi::kernels::quant::decode_grouped_int8(c, out, flag, nullptr), std::invalid_argument);
  c.group = 16;
  c.bits = 5;
  EXPECT_THROW(mi::kernels::quant::decode_grouped_int8(c, out, flag, nullptr), std::invalid_argument);
  c.bits = 6;
  c.codebook = reinterpret_cast<const float*>(std::uintptr_t{256});
  EXPECT_THROW(mi::kernels::quant::decode_grouped_int8(c, out, flag, nullptr), std::invalid_argument);
}

// A real W4A8 linear of the community file, decoded on the GPU and by the CPU reader.
TEST(Qwen35TextQuantDecode, CommunityLinearBitExact) {
  if (!fs::is_regular_file(w4a8_file())) GTEST_SKIP() << "absent: " << w4a8_file();
  const mi::quant::QuantModel m = mi::quant::open_quant_model(w4a8_file(), mi::quant::ModelKind::kTextEncoder);
  const mi::quant::QuantLinear* l = m.find_linear("language_model.layers.7.mlp.down_proj");
  ASSERT_NE(l, nullptr);
  ASSERT_EQ(l->encoding, mi::quant::Encoding::kW4A8);
  std::vector<std::int8_t> want(static_cast<std::size_t>(l->n * l->k));
  std::vector<float> scale(static_cast<std::size_t>(l->n));
  mi::quant::decode_linear(m, *l, want.data(), scale.data());
  const std::vector<std::byte> packed = mi::quant::read_tensor(m, l->weight), s_rel = mi::quant::read_tensor(m, *l->s_rel),
                               cb = mi::quant::read_tensor(m, *l->codebook);
  std::uint8_t *d_packed = nullptr, *d_srel = nullptr;
  float* d_cb = nullptr;
  std::int8_t* d_out = nullptr;
  unsigned* d_flag = nullptr;
  MI_CUDA_CHECK(cudaMalloc(&d_packed, packed.size()));
  MI_CUDA_CHECK(cudaMalloc(&d_srel, s_rel.size()));
  MI_CUDA_CHECK(cudaMalloc(&d_cb, cb.size()));
  MI_CUDA_CHECK(cudaMalloc(&d_out, want.size()));
  MI_CUDA_CHECK(cudaMalloc(&d_flag, sizeof(unsigned)));
  MI_CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemcpy(d_srel, s_rel.data(), s_rel.size(), cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemcpy(d_cb, cb.data(), cb.size(), cudaMemcpyHostToDevice));
  MI_CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(unsigned)));
  mi::kernels::quant::GroupedCodes c;
  c.packed = d_packed;
  c.s_rel = d_srel;
  c.codebook = d_cb;
  c.n = l->n;
  c.k = l->k;
  c.bits = 4;
  c.group = l->quant_group;
  mi::kernels::quant::decode_grouped_int8(c, d_out, d_flag, nullptr);
  std::vector<std::int8_t> got(want.size());
  unsigned flag = 1;
  MI_CUDA_CHECK(cudaMemcpy(got.data(), d_out, got.size(), cudaMemcpyDeviceToHost));
  MI_CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(flag), cudaMemcpyDeviceToHost));
  for (void* p : {static_cast<void*>(d_packed), static_cast<void*>(d_srel), static_cast<void*>(d_cb), static_cast<void*>(d_out),
                  static_cast<void*>(d_flag)}) {
    MI_CUDA_CHECK(cudaFree(p));
  }
  EXPECT_EQ(flag, 0u);
  EXPECT_TRUE(got == want) << "GPU decode differs from decode_linear";
}

// ---- Model ---------------------------------------------------------------------------------------------------

TEST(Qwen35TextQuantModel, OpensCommunityFiles) {
  for (const auto& [path, precision] : {std::pair{int8_file(), Precision::kInt8}, std::pair{w4a8_file(), Precision::kW4A8}}) {
    if (!fs::is_regular_file(path)) {
      std::printf("absent: %s\n", path.string().c_str());
      continue;
    }
    const QuantTextModel m(path);
    EXPECT_EQ(m.precision(), precision) << path;
    EXPECT_EQ(m.embed_tokens().k, kHidden);
    EXPECT_EQ(m.embed_tokens().spec, mi::quant::QuantSpec::int8_convrot(256));
    // 24 GDN layers x (A_log, linear_attn.norm.weight) stored F32.
    EXPECT_EQ(m.bf16_tensors().size(), 48u);
    const auto* a = m.find_linear("language_model.layers.0.linear_attn.in_proj_a");
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->kind, mi::qwen35_text::LinearInfo::Kind::kBF16);
    const auto* q = m.find_linear("language_model.layers.3.self_attn.q_proj");
    ASSERT_NE(q, nullptr);
    EXPECT_EQ(q->kind, precision == Precision::kInt8 ? mi::qwen35_text::LinearInfo::Kind::kInt8
                                                     : mi::qwen35_text::LinearInfo::Kind::kGrouped);
    std::uint64_t layer_bytes = 0;
    for (const auto& g : m.index().groups) layer_bytes += g.total_bytes;
    const EncoderOptions pinned{.source = WeightSource::kHostPinned};
    std::printf("%s: %s, streamed groups %.3f GB, host-resident (pinned) %.3f GB, max layer buffer %.1f MiB\n",
                path.filename().string().c_str(), mi::qwen35_text::precision_name(m.precision()),
                static_cast<double>(layer_bytes) / 1e9, static_cast<double>(TextEncoder::host_bytes(m, pinned)) / 1e9,
                static_cast<double>([&] {
                  std::size_t mx = 0;
                  for (const auto& g : m.index().groups) mx = std::max(mx, mi::weights::group_device_bytes(g));
                  return mx;
                }()) / (1 << 20));
    for (const std::vector<std::int64_t>& lengths : {std::vector<std::int64_t>{282, 10}, std::vector<std::int64_t>{4096, 16}}) {
      std::printf("  device workspace (2 buffers) for %lld+%lld tokens: %.1f MiB\n", static_cast<long long>(lengths[0]),
                  static_cast<long long>(lengths[1]), static_cast<double>(TextEncoder::workspace_bytes(m, lengths)) / (1 << 20));
    }
  }
  if (fs::is_regular_file(bundle_dir() / "moss.mossidx")) {
    const mi::weights::ModelIndex index = mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir());
    for (const std::vector<std::int64_t>& lengths : {std::vector<std::int64_t>{282, 10}, std::vector<std::int64_t>{4096, 16}}) {
      std::printf("bf16 bundle: device workspace (2 buffers) for %lld+%lld tokens: %.1f MiB\n", static_cast<long long>(lengths[0]),
                  static_cast<long long>(lengths[1]), static_cast<double>(TextEncoder::workspace_bytes(index, lengths)) / (1 << 20));
    }
  }
}

TEST(Qwen35TextQuantModel, RejectsDitFile) {
  const fs::path dit = "/workspace/Models/community/comfy-org/moss_image_sft_int8_convrot.safetensors";
  if (!fs::is_regular_file(dit)) GTEST_SKIP() << "absent: " << dit;
  EXPECT_ANY_THROW(QuantTextModel m(dit));
}

// One fixture per file: model, streams, arena sized for all 14 golden sequences with 3 buffers.
class QuantTE : public ::testing::TestWithParam<const char*> {
 protected:
  static std::map<std::string, std::unique_ptr<QuantTextModel>>& models() {
    static std::map<std::string, std::unique_ptr<QuantTextModel>> m;
    return m;
  }
  static void TearDownTestSuite() {
    arena_.reset();
    streams_.reset();
    models().clear();
  }
  void SetUp() override {
    path_ = std::string(GetParam()) == "int8" ? int8_file() : w4a8_file();
    if (!fs::is_regular_file(path_)) GTEST_SKIP() << "absent: " << path_;
    if (!fs::is_regular_file(kGolden / "cases.json")) GTEST_SKIP() << "no golden cases at " << kGolden;
    auto& m = models()[GetParam()];
    if (!m) m = std::make_unique<QuantTextModel>(path_);
    model_ = m.get();
    if (!streams_) streams_ = std::make_unique<mi::runtime::StreamSet>();
    std::vector<Ids> all;
    for (const auto& k : golden_keys()) all.push_back(golden_ids(k));
    EncoderOptions three;
    three.weight_buffers = 3;
    const std::size_t need = TextEncoder::workspace_bytes(*model_, lengths_of(all), three) + (std::size_t{64} << 20);
    if (!arena_ || arena_->capacity() < need) {
      arena_.reset();
      arena_ = std::make_unique<mi::runtime::DeviceArena>(need);
    }
  }
  std::unique_ptr<TextEncoder> make(const EncoderOptions& o = {}) {
    return std::make_unique<TextEncoder>(*model_, *arena_, streams_->compute(), o);
  }
  fs::path ref_dir() const { return kGolden / "qwen35_text_quant" / GetParam(); }

  fs::path path_;
  QuantTextModel* model_ = nullptr;
  static std::unique_ptr<mi::runtime::StreamSet> streams_;
  static std::unique_ptr<mi::runtime::DeviceArena> arena_;
};
std::unique_ptr<mi::runtime::StreamSet> QuantTE::streams_;
std::unique_ptr<mi::runtime::DeviceArena> QuantTE::arena_;

// Worst deviation of the torch quantized reference under a kernel swap (SDPA math / cuBLASLt) from its default run:
// the quantized model's own floor (tests/data/qwen35_text/quant_noise_floor.json). Wider than the bf16 floor: an
// ULP-level change flips INT8 roundings, each worth up to absmax/127 of a row.
Floor quant_floor(const mi::json::Value& qnf, const std::string& fmt, const std::string& key, const std::string& checkpoint) {
  const mi::json::Value* f = qnf.find(fmt);
  if (f == nullptr) throw std::runtime_error("quant noise floor has no format " + fmt);
  Floor out;
  const mi::json::Value* seq = f->at("sequences").find(key);
  if (seq == nullptr) throw std::runtime_error("quant noise floor has no sequence " + key);
  for (const char* variant : {"math_sdpa", "cublaslt"}) {
    const mi::json::Value& m = seq->at(variant).at(checkpoint);
    out.cosine = std::min(out.cosine, m.at("cosine").as_double());
    out.mean_row_rel_l2 = std::max(out.mean_row_rel_l2, m.at("mean_row_rel_l2").as_double());
  }
  return out;
}

// Layer 3 vs the torch quantized reference: AC4's early bounds widened by the measured amplification of the INT8 path
// (the engine's deviation from the reference is ~3x the bf16 engine's from the bf16 goldens at layers 0, 3 and the
// output, the signature of rounding flips, not of a defect: a wrong op shows up at its layer at full scale).
constexpr double kLayer3MinCosine = 0.9999;
constexpr double kLayer3MaxRowRelL2 = 3e-2;

// Engine vs torch quantized reference (same math): layer 0 and 3 fixed bounds, layer 31 and last_hidden within 2x the
// quantized model's own floor. AC23, engine vs bf16 goldens: no further from them than the torch quantized reference
// plus 2x that floor (the quantization error is the file's; the engine must not add to it beyond noise). Both are
// reported against the bf16 floor too.
TEST_P(QuantTE, GoldenParity) {
  const std::vector<std::string> keys = golden_keys();
  if (!fs::is_regular_file(ref_dir() / (keys[0] + ".te_last_hidden.npy")) || !fs::is_regular_file(kQuantNoiseFloor)) {
    GTEST_SKIP() << "no torch quantized reference at " << ref_dir() << " or no " << kQuantNoiseFloor;
  }
  const mi::json::Value nf = read_json(kNoiseFloor);
  const mi::json::Value qnf = read_json(kQuantNoiseFloor);
  std::vector<Ids> seqs;
  for (const auto& k : keys) seqs.push_back(golden_ids(k));
  std::map<int, Bf16> captured;
  mi::qwen35_text::EncodeOptions eo;
  eo.layer_hook = [&](int layer, const mi::Tensor& hidden, cudaStream_t stream) {
    if (layer != 0 && layer != 3 && layer != 31) return;
    Bf16& dst = captured[layer];
    dst.resize(static_cast<std::size_t>(hidden.numel()));
    MI_CUDA_CHECK(cudaMemcpyAsync(dst.data(), hidden.data, hidden.nbytes(), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  auto enc = make();
  const std::size_t before = arena_->mark();
  const auto out = enc->encode_to_host(seqs, eo);
  EXPECT_EQ(arena_->mark(), before);
  ASSERT_EQ(out.size(), seqs.size());
  double worst_cos_bf16 = 1, worst_cos_ref = 1, worst_ratio_bf16 = 0, worst_ratio_qfloor = 0, worst_excess = 0;
  std::size_t row0 = 0;
  const auto at = [&](int layer) { return captured[layer].data() + row0 * kHidden; };
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto rows = static_cast<std::int64_t>(seqs[i].size());
    const std::string& k = keys[i];
    std::printf("%s %s (%lld tokens)\n", GetParam(), k.c_str(), static_cast<long long>(rows));
    const Floor f = noise_floor(nf, k, "te_last_hidden");
    std::printf("  %-34s cos %.7f  row relL2 mean %.3e\n", "(torch bf16 kernel-swap floor)", f.cosine, f.mean_row_rel_l2);
    const std::vector<float> golden = load_f32(kGolden / (k + ".te_last_hidden.npy"), rows);
    const Metrics g = compare(out[i].data(), golden);
    print("last_hidden vs bf16 golden (AC23)", g, f);
    worst_ratio_bf16 = std::max(worst_ratio_bf16, std::max((1 - g.cosine) / (1 - f.cosine), g.mean_row_rel_l2 / f.mean_row_rel_l2));
    worst_cos_bf16 = std::min(worst_cos_bf16, g.cosine);
    const Floor qf = quant_floor(qnf, GetParam(), k, "te_last_hidden");
    std::printf("  %-34s cos %.7f  row relL2 mean %.3e\n", "(torch quantized kernel-swap floor)", qf.cosine, qf.mean_row_rel_l2);
    const Metrics r0 = compare(at(0), load_f32(ref_dir() / (k + ".te_layer00.npy"), rows));
    print("layer00 vs torch quantized ref", r0, noise_floor(nf, k, "te_layer03"));
    EXPECT_GE(r0.cosine, kEarlyMinCosine) << k;
    EXPECT_LE(r0.max_row_rel_l2, kEarlyMaxRowRelL2) << k;
    const Metrics r3 = compare(at(3), load_f32(ref_dir() / (k + ".te_layer03.npy"), rows));
    print("layer03 vs torch quantized ref", r3, noise_floor(nf, k, "te_layer03"));
    EXPECT_GE(r3.cosine, kLayer3MinCosine) << k;
    EXPECT_LE(r3.max_row_rel_l2, kLayer3MaxRowRelL2) << k;
    for (const char* cp : {"te_layer31", "te_last_hidden"}) {
      const bool last = std::string(cp) == "te_last_hidden";
      const Metrics r = compare(last ? out[i].data() : at(31), load_f32(ref_dir() / (k + "." + cp + ".npy"), rows));
      const Floor qcp = quant_floor(qnf, GetParam(), k, cp);
      print(std::string(cp).substr(3) + " vs torch quantized ref", r, f);
      std::printf("  %-34s (1-cos %5.2fx, row relL2 mean %5.2fx the quantized floor)\n", "", (1 - r.cosine) / (1 - qcp.cosine),
                  r.mean_row_rel_l2 / qcp.mean_row_rel_l2);
      EXPECT_GE(r.cosine, 1 - floor_factor() * (1 - qcp.cosine)) << k << " " << cp;
      EXPECT_LE(r.mean_row_rel_l2, floor_factor() * qcp.mean_row_rel_l2) << k << " " << cp;
      worst_ratio_qfloor = std::max(worst_ratio_qfloor, std::max((1 - r.cosine) / (1 - qcp.cosine), r.mean_row_rel_l2 / qcp.mean_row_rel_l2));
      if (last) worst_cos_ref = std::min(worst_cos_ref, r.cosine);
    }
    // AC23: the reference's own distance from the bf16 golden, plus 2x the quantized floor.
    const std::vector<float> ref = load_f32(ref_dir() / (k + ".te_last_hidden.npy"), rows);
    Bf16 ref_bits(ref.size());
    for (std::size_t j = 0; j < ref.size(); ++j) ref_bits[j] = mi::quant::float_to_bf16(ref[j]);
    const Metrics rg = compare(ref_bits.data(), golden);
    print("(torch quantized ref vs bf16 golden)", rg, f);
    EXPECT_LE(1 - g.cosine, (1 - rg.cosine) + floor_factor() * (1 - qf.cosine)) << k;
    EXPECT_LE(g.mean_row_rel_l2, rg.mean_row_rel_l2 + floor_factor() * qf.mean_row_rel_l2) << k;
    worst_excess = std::max(worst_excess, (g.mean_row_rel_l2 - rg.mean_row_rel_l2) / qf.mean_row_rel_l2);
    row0 += static_cast<std::size_t>(rows);
  }
  std::printf("%s over 14 sequences: vs bf16 goldens worst cos %.6f (worst %.2fx the bf16 floor); vs torch quantized ref worst "
              "cos %.7f (worst %.2fx the quantized floor); engine minus reference distance to the goldens <= %.2fx the "
              "quantized floor\n",
              GetParam(), worst_cos_bf16, worst_ratio_bf16, worst_cos_ref, worst_ratio_qfloor, worst_excess);
  RecordProperty("worst_cos_vs_bf16", std::to_string(worst_cos_bf16));
  RecordProperty("worst_bf16_floor_ratio", std::to_string(worst_ratio_bf16));
  RecordProperty("worst_cos_vs_ref", std::to_string(worst_cos_ref));
  RecordProperty("worst_quant_floor_ratio", std::to_string(worst_ratio_qfloor));
}

// ---- Teacher-forced per-layer check ----------------------------------------------------------------------------
//
// The end-to-end bounds above cannot see a small error in one linear: rounding flips amplified over 32 layers are as
// large as the whole quantization effect. Here every layer gets the torch quantized reference's input: the layer hook
// replaces the residual stream after engine layer i with the reference's layer-i output, so engine layer i + 1 runs on
// exactly the reference's input, and its contribution (output - input) is compared with the reference's. Subtracting
// the input removes the residual stream (and its large outlier channels), so the metric is the layer's own error.
// References: the local generator dumps every layer for kTfKeys (tests/golden/qwen35_text_quant/<fmt>-layers/), and
// the per-layer floor: the same layer, same input, one torch kernel swapped (attention layers: SDPA math backend;
// GatedDeltaNet layers: recurrent instead of chunked delta rule), tests/data/qwen35_text/quant_layer_floor.json.
const fs::path kQuantLayerFloor = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/qwen35_text/quant_layer_floor.json";
const char* const kTfKeys[] = {"g_short_1024.cond", "g_layout_1024.uncond"};
// Bound on a layer's contribution rel-L2: kTfFloorFactor x the per-layer floor of its kind (GatedDeltaNet or attention)
// for the file: the largest floor measured over every layer of that kind and both sequences. One layer's floor is a
// count of rounding flips (often zero, since most swaps change no bf16 bit), so a single layer's value is too noisy to
// scale; pooled over 48 GDN (16 attention) samples it is the noise scale of one layer. The factor 2 is AC4's (the engine
// differs from the reference in several kernels at once).
constexpr double kTfFloorFactor = 2.0;

struct LayerError {
  double rel_l2 = 0;          // ||c_engine - c_ref|| / ||c_ref|| over the sequence
  double max_row_rel_l2 = 0;  // worst token row
};

class TeacherForced {
 public:
  TeacherForced(const fs::path& dir, const mi::json::Value& floor_doc, const std::string& fmt) {
    for (const char* k : kTfKeys) {
      const auto rows = static_cast<std::int64_t>(golden_ids(k).size());
      embed_.push_back(load_f32(dir / (std::string(k) + ".te_embed.npy"), rows));
      std::vector<std::vector<float>> layers;
      for (int i = 0; i < mi::qwen35_text::kNumLayers; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), ".te_layer%02d.npy", i);
        layers.push_back(load_f32(dir / (std::string(k) + name), rows));
      }
      ref_.push_back(std::move(layers));
      rows_.push_back(rows);
      const mi::json::Value& per_layer = floor_doc.at(fmt).at("sequences").at(k);
      std::vector<double> f;
      for (const auto& m : per_layer.as_array()) f.push_back(m.at("rel_l2").as_double());
      if (f.size() != static_cast<std::size_t>(mi::qwen35_text::kNumLayers)) throw std::runtime_error("bad layer floor");
      for (int i = 0; i < mi::qwen35_text::kNumLayers; ++i) {
        double& kf = kind_floor_[i % 4 == 3 ? 1 : 0];
        kf = std::max(kf, f[static_cast<std::size_t>(i)]);
      }
      floor_.push_back(std::move(f));
    }
  }

  // errors[seq][layer]; `linear_hook` is passed to the encode (fault injection).
  std::vector<std::vector<LayerError>> run(TextEncoder& enc, const mi::qwen35_text::LinearHook& linear_hook = {}) const {
    std::vector<Ids> seqs;
    for (const char* k : kTfKeys) seqs.push_back(golden_ids(k));
    std::vector<std::vector<LayerError>> errors(seqs.size(), std::vector<LayerError>(mi::qwen35_text::kNumLayers));
    mi::qwen35_text::EncodeOptions eo;
    eo.linear_hook = linear_hook;
    Bf16 host;
    eo.layer_hook = [&](int layer, const mi::Tensor& hidden, cudaStream_t stream) {
      host.resize(static_cast<std::size_t>(hidden.numel()));
      MI_CUDA_CHECK(cudaMemcpyAsync(host.data(), hidden.data, hidden.nbytes(), cudaMemcpyDeviceToHost, stream));
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
      std::size_t row0 = 0;
      for (std::size_t s = 0; s < seqs.size(); ++s) {
        const std::vector<float>& in = layer == 0 ? embed_[s] : ref_[s][static_cast<std::size_t>(layer - 1)];
        const std::vector<float>& out = ref_[s][static_cast<std::size_t>(layer)];
        double num = 0, den = 0;
        LayerError& e = errors[s][static_cast<std::size_t>(layer)];
        for (std::int64_t r = 0; r < rows_[s]; ++r) {
          double rn = 0, rd = 0;
          for (std::int64_t c = 0; c < kHidden; ++c) {
            const std::size_t j = static_cast<std::size_t>(r * kHidden + c);
            const double ce = static_cast<double>(bf16_to_f32(host[row0 * kHidden + j])) - in[j];
            const double cr = static_cast<double>(out[j]) - in[j];
            rn += (ce - cr) * (ce - cr);
            rd += cr * cr;
          }
          num += rn;
          den += rd;
          e.max_row_rel_l2 = std::max(e.max_row_rel_l2, std::sqrt(rn / rd));
        }
        e.rel_l2 = std::sqrt(num / den);
        // Teacher forcing: the next layer starts from the reference's output.
        for (std::size_t j = 0; j < out.size(); ++j) host[row0 * kHidden + j] = mi::quant::float_to_bf16(out[j]);
        row0 += static_cast<std::size_t>(rows_[s]);
      }
      MI_CUDA_CHECK(cudaMemcpyAsync(hidden.data, host.data(), hidden.nbytes(), cudaMemcpyHostToDevice, stream));
      MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    enc.encode_to_host(seqs, eo);
    return errors;
  }

  double floor(std::size_t seq, int layer) const { return floor_[seq][static_cast<std::size_t>(layer)]; }
  double bound(int layer) const { return kTfFloorFactor * kind_floor_[layer % 4 == 3 ? 1 : 0]; }

 private:
  std::vector<std::vector<float>> embed_;
  std::vector<std::vector<std::vector<float>>> ref_;
  std::vector<std::int64_t> rows_;
  std::vector<std::vector<double>> floor_;
  double kind_floor_[2] = {0, 0};  // GatedDeltaNet, full attention
};

std::unique_ptr<TeacherForced> load_teacher_forced(const fs::path& dir, const std::string& fmt) {
  if (!fs::is_regular_file(dir / (std::string(kTfKeys[0]) + ".te_layer31.npy")) || !fs::is_regular_file(kQuantLayerFloor)) {
    return nullptr;
  }
  const mi::json::Value doc = read_json(kQuantLayerFloor);
  if (doc.find(fmt) == nullptr) return nullptr;
  return std::make_unique<TeacherForced>(dir, doc, fmt);
}

TEST_P(QuantTE, TeacherForcedPerLayer) {
  const auto tf = load_teacher_forced(kGolden / "qwen35_text_quant" / (std::string(GetParam()) + "-layers"), GetParam());
  if (!tf) GTEST_SKIP() << "no per-layer torch quantized reference or floor (local generator not run)";
  auto enc = make();
  const auto errors = tf->run(*enc);
  double worst_ratio = 0, worst_rel = 0;
  for (std::size_t s = 0; s < errors.size(); ++s) {
    std::printf("%s %s: layer  contribution rel-L2 (worst row)  layer floor  bound  use of bound\n", GetParam(), kTfKeys[s]);
    for (int i = 0; i < mi::qwen35_text::kNumLayers; ++i) {
      const LayerError& e = errors[s][static_cast<std::size_t>(i)];
      std::printf("  %2d %s  %.3e (%.3e)  %.3e  %.3e  %5.2f\n", i, i % 4 == 3 ? "attn" : "gdn ", e.rel_l2, e.max_row_rel_l2,
                  tf->floor(s, i), tf->bound(i), e.rel_l2 / tf->bound(i));
      EXPECT_LE(e.rel_l2, tf->bound(i)) << kTfKeys[s] << " layer " << i;
      worst_ratio = std::max(worst_ratio, e.rel_l2 / tf->bound(i));
      worst_rel = std::max(worst_rel, e.rel_l2);
    }
  }
  std::printf("%s: worst per-layer contribution rel-L2 %.3e, worst use of the bound %.2f\n", GetParam(), worst_rel, worst_ratio);
  RecordProperty("worst_layer_rel_l2", std::to_string(worst_rel));
  RecordProperty("worst_bound_use", std::to_string(worst_ratio));
}

// The check catches a single-linear defect: layer 17's down_proj output scaled by 1.1 (a 10 % weight-scale error)
// through the linear hook must break layer 17's bound, and only that layer's.
TEST_P(QuantTE, TeacherForcedCatchesScaledLinear) {
  const auto tf = load_teacher_forced(kGolden / "qwen35_text_quant" / (std::string(GetParam()) + "-layers"), GetParam());
  if (!tf) GTEST_SKIP() << "no per-layer torch quantized reference or floor (local generator not run)";
  constexpr int kLayer = 17;
  const std::string target = "language_model.layers." + std::to_string(kLayer) + ".mlp.down_proj";
  int hits = 0;
  auto enc = make();
  const auto errors = tf->run(*enc, [&](const std::string& module, const mi::Tensor& out, cudaStream_t stream) {
    if (module != target) return;
    ++hits;
    Bf16 h(static_cast<std::size_t>(out.numel()));
    MI_CUDA_CHECK(cudaMemcpyAsync(h.data(), out.data, out.nbytes(), cudaMemcpyDeviceToHost, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (auto& v : h) v = mi::quant::float_to_bf16(1.1f * bf16_to_f32(v));
    MI_CUDA_CHECK(cudaMemcpyAsync(out.data, h.data(), out.nbytes(), cudaMemcpyHostToDevice, stream));
    MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  });
  EXPECT_EQ(hits, 1);
  for (std::size_t s = 0; s < errors.size(); ++s) {
    for (int i = 0; i < mi::qwen35_text::kNumLayers; ++i) {
      const double rel = errors[s][static_cast<std::size_t>(i)].rel_l2, bound = tf->bound(i);
      if (i == kLayer) {
        std::printf("%s %s: injected layer %d contribution rel-L2 %.3e = %.1fx its bound %.3e\n", GetParam(), kTfKeys[s], i, rel,
                    rel / bound, bound);
        EXPECT_GT(rel, bound) << kTfKeys[s] << ": the injected error went undetected";
      } else {
        EXPECT_LE(rel, bound) << kTfKeys[s] << " layer " << i;
      }
    }
  }
}

// A NaN group scale reaching the GPU decoder makes encode() throw (the stored scales are only checked as they are
// decoded); the encoder stays usable and the arena is restored.
TEST_P(QuantTE, BadGroupScaleThrows) {
  if (model_->precision() != Precision::kW4A8) GTEST_SKIP() << "no grouped linears in this file";
  const std::vector<Ids> seqs = {golden_ids("g_short_1024.uncond")};
  auto enc = make();
  const auto good = enc->encode_to_host(seqs);
  mi::qwen35_text::EncodeOptions eo;
  int hits = 0;
  eo.decode_hook = [&](const std::string& module, std::uint8_t* s_rel, std::int64_t bytes, cudaStream_t stream) {
    if (module != "language_model.layers.9.mlp.up_proj") return;
    ++hits;
    ASSERT_GT(bytes, 1000);
    MI_CUDA_CHECK(cudaMemsetAsync(s_rel + 1000, 0x7F, 1, stream));  // e4m3 NaN
  };
  const std::size_t before = arena_->mark();
  EXPECT_THROW(enc->encode_to_host(seqs, eo), mi::quant::QuantError);
  EXPECT_EQ(hits, 1);
  EXPECT_EQ(arena_->mark(), before);
  expect_identical("after the failed encode", good[0], enc->encode_to_host(seqs)[0]);
}

TEST_P(QuantTE, VarlenBatchEqualsPerSequence) {
  const std::vector<Ids> seqs = {golden_ids("g_layout_1024.cond"), golden_ids("g_fridge_1024.uncond"),
                                 golden_ids("g_short_1024.cond")};
  auto enc = make();
  const auto batch = enc->encode_to_host(seqs);
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    const auto alone = enc->encode_to_host({seqs[i]});
    expect_identical("sequence " + std::to_string(i), alone[0], batch[i]);
  }
}

// Synchronous, 1/2/3 buffers, host-pageable and host-pinned: the same bits; device peak == workspace_bytes; host
// copy == host_bytes.
TEST_P(QuantTE, StreamingConfigurationsBitIdentical) {
  const std::vector<Ids> seqs = {golden_ids("g_layout_1024.cond"), golden_ids("g_layout_1024.uncond")};
  Bf16 ref;
  EncoderOptions sync;
  sync.streaming = false;
  EncoderOptions configs[] = {sync, {}, {}, {}, {}, {}};
  configs[2].weight_buffers = 1;
  configs[3].weight_buffers = 3;
  configs[4].source = WeightSource::kHostPageable;
  configs[5].source = WeightSource::kHostPinned;
  for (const EncoderOptions& o : configs) {
    auto enc = make(o);
    const std::size_t used = arena_->used();
    arena_->reset_high_water();
    const auto out = enc->encode_to_host(seqs);
    EXPECT_EQ(arena_->high_water() - used, TextEncoder::workspace_bytes(*model_, lengths_of(seqs), o));
    EXPECT_EQ(enc->host_resident_bytes() + (o.streaming && o.source != WeightSource::kHostPinned
                                                ? std::size_t(o.reader_threads) * std::size_t(o.ring_slots) *
                                                      mi::runtime::align_up(o.chunk_bytes + mi::weights::kDirectAlign,
                                                                            mi::weights::kDirectAlign)
                                                : 0),
              TextEncoder::host_bytes(*model_, o));
    Bf16 flat;
    for (const auto& v : out) flat.insert(flat.end(), v.begin(), v.end());
    if (ref.empty()) {
      ref = flat;
    } else {
      expect_identical("streaming=" + std::to_string(o.streaming) + " buffers=" + std::to_string(o.weight_buffers) +
                           " source=" + std::to_string(static_cast<int>(o.source)),
                       ref, flat);
    }
    EXPECT_EQ(enc->precision(), model_->precision());
    EXPECT_NE(enc->numerics(), mi::qwen35_text::numerics_id());
  }
}

INSTANTIATE_TEST_SUITE_P(Files, QuantTE, ::testing::Values("int8", "w4a8"));

// ---- Perf ----------------------------------------------------------------------------------------------------

#ifndef _WIN32
void drop_page_cache(const fs::path& file) {
  const int fd = ::open(file.c_str(), O_RDONLY);
  if (fd < 0) return;
  static_cast<void>(::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED));
  ::close(fd);
}
#endif

// Prompt + negative, 282+10 (the AC9 / AC15 prompt length) and 4096+16 tokens. Per configuration: second encode
// (page cache warm or host copy loaded), and with MOSSINFER_TE_COLD=1 a cold page-cache encode of the file-streamed
// configurations. "dev" numbers: other GPU work may run.
TEST(Qwen35TextQuantPerf, EncodeTimes) {
  const bool cold = std::getenv("MOSSINFER_TE_COLD") != nullptr;
  const Ids long_cond = golden_ids("g_long4096_1024.cond");
  const std::vector<std::pair<const char*, std::vector<Ids>>> cases = {
      {"282+10", {Ids(long_cond.begin(), long_cond.begin() + 282), golden_ids("g_fridge_1024.uncond")}},
      {"4096+16", {long_cond, golden_ids("g_long4096_1024.uncond")}}};
  std::vector<std::int64_t> big = lengths_of(cases[1].second);
  mi::runtime::StreamSet streams;
  EncoderOptions pinned;
  pinned.source = WeightSource::kHostPinned;
  std::printf("%-24s %-8s %8s %8s %8s %8s %8s %8s  %s\n", "config", "case", "total", "embed", "load", "wait", "compute",
              "GB", "host copy GB / preload s");
  const auto run = [&](const char* name, auto&& make_enc, const std::vector<fs::path>& files) {
    for (const EncoderOptions& o : {EncoderOptions{}, pinned}) {
      std::unique_ptr<mi::runtime::DeviceArena> arena;
      std::unique_ptr<TextEncoder> enc = make_enc(o, arena);
      if (!enc) continue;  // configuration not measured for this model
      const auto t0 = std::chrono::steady_clock::now();
      enc->preload_host();
      const double preload = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const std::string cfg = std::string(name) + (o.source == WeightSource::kHostPinned ? " RAM pinned" : " file");
      for (const auto& [cname, seqs] : cases) {
        for (int r = 0; r < 2; ++r) enc->encode_to_host(seqs);
        const auto& s = enc->last_stats();
        std::printf("%-24s %-8s %8.3f %8.4f %8.3f %8.3f %8.3f %8.2f  %.2f / %.2f\n", cfg.c_str(), cname, s.total_s, s.embed_s,
                    s.weight_load_s, s.weight_wait_s, s.compute_s, static_cast<double>(s.weight_bytes) / 1e9,
                    static_cast<double>(enc->host_resident_bytes()) / 1e9, preload);
        RecordProperty(cfg + " " + cname + " total_s", std::to_string(s.total_s));
#ifndef _WIN32
        if (cold && o.source == WeightSource::kFile) {
          for (const auto& f : files) drop_page_cache(f);
          enc->encode_to_host(seqs);
          const auto& sc = enc->last_stats();
          std::printf("%-24s %-8s %8.3f %8.4f %8.3f %8.3f %8.3f %8.2f  (cold page cache)\n", cfg.c_str(), cname, sc.total_s,
                      sc.embed_s, sc.weight_load_s, sc.weight_wait_s, sc.compute_s, static_cast<double>(sc.weight_bytes) / 1e9);
        }
#endif
      }
    }
  };
  if (fs::is_regular_file(bundle_dir() / "moss.mossidx")) {
    const mi::weights::ModelIndex index = mi::weights::load_index(bundle_dir() / "moss.mossidx", bundle_dir());
    std::vector<fs::path> files;
    for (std::uint32_t i = 0; i < index.files.size(); ++i) files.push_back(index.file_path(i));
    run("bf16 bundle",
        [&](const EncoderOptions& o, std::unique_ptr<mi::runtime::DeviceArena>& arena) {
          if (o.source == WeightSource::kHostPinned) return std::unique_ptr<TextEncoder>();
          arena = std::make_unique<mi::runtime::DeviceArena>(TextEncoder::workspace_bytes(index, big, o));
          return std::make_unique<TextEncoder>(index, *arena, streams.compute(), o);
        },
        files);
  }
  for (const auto& [name, path] : {std::pair{"int8", int8_file()}, std::pair{"w4a8", w4a8_file()}}) {
    if (!fs::is_regular_file(path)) continue;
    const QuantTextModel model(path);
    run(name,
        [&](const EncoderOptions& o, std::unique_ptr<mi::runtime::DeviceArena>& arena) {
          arena = std::make_unique<mi::runtime::DeviceArena>(TextEncoder::workspace_bytes(model, big, o));
          return std::make_unique<TextEncoder>(model, *arena, streams.compute(), o);
        },
        {path});
  }
}

// The alternative to GPU decoding (W4A8 decoded to INT8 on the CPU at load): its cost, for the design record.
// MOSSINFER_TE_CPU_DECODE=1; needs ~7 GB of host RAM for the decoded copy.
TEST(Qwen35TextQuantPerf, CpuDecodeAtLoad) {
  if (std::getenv("MOSSINFER_TE_CPU_DECODE") == nullptr) GTEST_SKIP() << "set MOSSINFER_TE_CPU_DECODE=1";
  if (!fs::is_regular_file(w4a8_file())) GTEST_SKIP() << "absent: " << w4a8_file();
  const mi::quant::QuantModel m = mi::quant::open_quant_model(w4a8_file(), mi::quant::ModelKind::kTextEncoder);
  std::size_t bytes = 0;
  for (const auto& l : m.linears) {
    if (l.needs_decode()) bytes += static_cast<std::size_t>(l.n * l.k + l.n * 4);
  }
  std::vector<std::int8_t> q;
  std::vector<float> s;
  q.reserve(bytes);
  const auto t0 = std::chrono::steady_clock::now();
  std::size_t done = 0;
  for (const auto& l : m.linears) {
    if (!l.needs_decode()) continue;
    q.resize(static_cast<std::size_t>(l.n * l.k));
    s.resize(static_cast<std::size_t>(l.n));
    mi::quant::decode_linear(m, l, q.data(), s.data());
    done += q.size() + s.size() * 4;
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("CPU decode of every W4A8 linear: %.2f GB of INT8 + scales in %.2f s (%.2f GB/s, all cores, page cache warm)\n",
              static_cast<double>(done) / 1e9, secs, static_cast<double>(done) / 1e9 / secs);
}

}  // namespace
