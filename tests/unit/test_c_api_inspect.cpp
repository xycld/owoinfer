// mi_inspect_file() (MI_CAP_INSPECT) on synthetic files written here, and on the community files when present. CPU
// only: inspection never touches CUDA (the cpu label's guard fails a test that leaves a context).
#include "mossinfer/mossinfer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "models/lora/lora.h"
#include "test_util.h"
#include "util/path.h"
#include "weights/safetensors.h"

namespace {

namespace fs = std::filesystem;
using mi::weights::DType;

// A tensor: zeros, an F32 `*weight_scale` of ones (the readers check scales are finite and > 0), or a comfy_quant
// config's JSON (U8).
struct T {
  std::string name;
  DType dtype;
  std::vector<std::int64_t> shape;
  std::string json = "";
};

// Laid out like the safetensors library writes files: larger dtypes first, the header padded to 8 bytes. Zero tensors
// are left as a hole (sparse where the file system allows), so files with MOSS-Image's shapes stay cheap.
void write_file(const fs::path& path, std::vector<T> ts, const std::string& metadata_json = "") {
  std::stable_sort(ts.begin(), ts.end(),
                   [](const T& a, const T& b) {
                     return mi::weights::dtype_size(a.dtype) > mi::weights::dtype_size(b.dtype);
                   });
  std::vector<std::pair<std::uint64_t, std::string>> data;  // non-zero tensors: offset in the data section, bytes
  std::string h = "{";
  if (!metadata_json.empty()) h += "\"__metadata__\":" + metadata_json;
  std::uint64_t off = 0;
  for (T& t : ts) {
    std::string bytes;
    std::uint64_t nbytes = 0;
    if (!t.json.empty()) {
      bytes = t.json;
      t.shape = {static_cast<std::int64_t>(bytes.size())};
      nbytes = bytes.size();
    } else {
      std::uint64_t n = 1;
      for (const std::int64_t d : t.shape) n *= static_cast<std::uint64_t>(d);
      nbytes = n * mi::weights::dtype_size(t.dtype);
      if (t.dtype == DType::kF32 && t.name.ends_with("weight_scale")) {
        bytes.assign(nbytes, '\0');
        const float one = 1.0f;
        for (std::uint64_t i = 0; i < n; ++i) std::memcpy(bytes.data() + 4 * i, &one, 4);
      }
    }
    if (h.size() > 1) h += ",";
    h += "\"" + t.name + "\":{\"dtype\":\"" + mi::weights::dtype_name(t.dtype) + "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
    h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + nbytes) + "]}";
    if (!bytes.empty()) data.emplace_back(off, std::move(bytes));
    off += nbytes;
  }
  h += "}";
  h.append((8 - h.size() % 8) % 8, ' ');
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t len = h.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (const auto& [at, bytes] : data) {
      out.seekp(static_cast<std::streamoff>(8 + h.size() + at));
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
  }
  fs::resize_file(path, 8 + h.size() + off);
}

void write_bytes(const fs::path& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

enum class Gemm { kBf16, kInt8, kInt4, kInt8ScalarScale };

// The comfy_quant configs of the INT8 ConvRot256 and INT4 ConvRot256 linears.
constexpr const char* kInt8Config = R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})";
constexpr const char* kInt4Config = R"({"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64})";

// Linear `w` [n, k] stored as `gemm` (ComfyUI comfy_quant configs).
void add_linear(std::vector<T>& t, const std::string& w, Gemm gemm, std::int64_t n, std::int64_t k) {
  switch (gemm) {
    case Gemm::kBf16: t.push_back({w + ".weight", DType::kBF16, {n, k}}); break;
    case Gemm::kInt8:
      t.push_back({w + ".weight", DType::kI8, {n, k}});
      t.push_back({w + ".weight_scale", DType::kF32, {n}});
      t.push_back({w + ".comfy_quant", DType::kU8, {}, kInt8Config});
      break;
    case Gemm::kInt4:
      t.push_back({w + ".weight", DType::kI8, {n, k / 2}});
      t.push_back({w + ".weight_scale", DType::kF32, {n}});
      t.push_back({w + ".comfy_quant", DType::kU8, {}, kInt4Config});
      break;
    case Gemm::kInt8ScalarScale:
      t.push_back({w + ".weight", DType::kI8, {n, k}});
      t.push_back({w + ".weight_scale", DType::kF32, {}});
      t.push_back({w + ".comfy_quant", DType::kU8, {}, R"({"format": "int8_tensorwise"})"});
      break;
  }
}

// A single-file DiT with MOSS-Image's names and shapes as check_supported_dit_file reads them: 42 blocks, the bf16
// embedders, and block 0's proj.q [3072, 3072] and ffn_down [3072, 8192] stored as `gemm` unless `extra` has them.
// `extra` and `metadata_json` are added as they are.
fs::path write_dit(const fs::path& path, Gemm gemm, std::vector<T> extra = {}, const std::string& metadata_json = "") {
  std::vector<T> t = std::move(extra);
  const auto given = [&](const std::string& name) {
    return std::any_of(t.begin(), t.end(), [&](const T& x) { return x.name == name; });
  };
  for (int i = 0; i < 42; ++i) {
    const std::string b = "layers." + std::to_string(i) + ".";
    t.push_back({b + "norm1.weight", DType::kBF16, {3072}});
    t.push_back({b + "q_norm.weight", DType::kBF16, {128}});
  }
  if (!given("layers.0.proj.q.weight")) add_linear(t, "layers.0.proj.q", gemm, 3072, 3072);
  if (!given("layers.0.proj.ffn_down.weight")) add_linear(t, "layers.0.proj.ffn_down", gemm, 3072, 8192);
  t.push_back({"video_embedder.weight", DType::kBF16, {3072, 128}});
  t.push_back({"text_embedder.proj.weight", DType::kBF16, {3072, 4096}});
  t.push_back({"time_embedding.0.weight", DType::kBF16, {3072, 256}});
  t.push_back({"final_linear_video.weight", DType::kBF16, {128, 3072}});
  write_file(path, t, metadata_json);
  return path;
}

// A PEFT LoRA on block 0's q (rank 4), optionally with an embedder widened for `kinds` control latents.
std::vector<T> lora_tensors(int control_groups = 0) {
  std::vector<T> t = {{"transformer.layers.0.proj.q.lora_A.weight", DType::kBF16, {4, 3072}},
                      {"transformer.layers.0.proj.q.lora_B.weight", DType::kBF16, {3072, 4}}};
  if (control_groups > 0) {
    t.push_back({"video_embedder.weight", DType::kBF16, {3072, 128 * (1 + control_groups)}});
    t.push_back({"video_embedder.bias", DType::kBF16, {3072}});
  }
  return t;
}

// The report of `path`, which must inspect with MI_OK.
struct Report {
  explicit Report(const fs::path& path) {
    const std::string p = mi::util::path_utf8(path);
    EXPECT_EQ(mi_inspect_file(p.c_str(), &r), MI_OK) << mi_last_error_message();
    info.struct_size = sizeof info;
    if (r != nullptr) {
      EXPECT_EQ(mi_file_report_info(r, &info), MI_OK);
    }
  }
  ~Report() { mi_file_report_free(r); }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
  static std::string checked(const char* s) {
    EXPECT_NE(s, nullptr);
    return s != nullptr ? s : "";
  }
  std::string text(mi_file_text_kind k) const { return checked(r != nullptr ? mi_file_report_text(r, k) : nullptr); }
  std::string tier_reason(mi_tier t) const {
    return checked(r != nullptr ? mi_file_report_tier_reason(r, t) : nullptr);
  }
  mi_file_report* r = nullptr;
  mi_file_info info{};
};

constexpr std::uint32_t bit(mi_tier t) { return 1u << t; }

TEST(CApiInspect, Bf16Dit) {
  mi::testing::TempDir dir("inspect_bf16");
  const Report rep(write_dit(dir.path() / "dit.safetensors", Gemm::kBf16));
  EXPECT_EQ(rep.info.kind, MI_FILE_KIND_DIT);
  EXPECT_EQ(rep.info.loadable, 1) << rep.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(rep.info.precision, MI_PRECISION_BF16);
  EXPECT_EQ(rep.info.tensor_count, 42 * 2 + 2 + 4);
  EXPECT_EQ(rep.info.file_bytes, fs::file_size(dir.path() / "dit.safetensors"));
  EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), "bf16");
  EXPECT_EQ(rep.text(MI_FILE_TEXT_NOTE), "");
  EXPECT_EQ(rep.info.tiers, bit(MI_TIER_AUTO) | bit(MI_TIER_REFERENCE) | bit(MI_TIER_HIGH));
  EXPECT_NE(rep.tier_reason(MI_TIER_AUTO).find("Reference"), std::string::npos) << rep.tier_reason(MI_TIER_AUTO);
  EXPECT_NE(rep.tier_reason(MI_TIER_HIGH).find("bf16 GEMMs, q8_fp8 attention"), std::string::npos);
  EXPECT_NE(rep.tier_reason(MI_TIER_TURBO).find("tier turbo runs an INT4 ConvRot DiT"), std::string::npos);
  EXPECT_EQ(rep.info.lora_adapters, 0);
  EXPECT_EQ(rep.info.control_kind_count, 0);
  EXPECT_EQ(rep.info._reserved0, 0);
}

TEST(CApiInspect, QuantizedDits) {
  mi::testing::TempDir dir("inspect_quant");
  {
    const Report rep(write_dit(dir.path() / "int8.safetensors", Gemm::kInt8));
    EXPECT_EQ(rep.info.kind, MI_FILE_KIND_DIT);
    EXPECT_EQ(rep.info.loadable, 1) << rep.text(MI_FILE_TEXT_NOTE);
    EXPECT_EQ(rep.info.precision, MI_PRECISION_INT8);
    EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), "int8_convrot256");
    EXPECT_EQ(rep.info.tiers, bit(MI_TIER_AUTO) | bit(MI_TIER_HIGH));
    EXPECT_NE(rep.tier_reason(MI_TIER_REFERENCE).find("quantized"), std::string::npos)
        << rep.tier_reason(MI_TIER_REFERENCE);
    EXPECT_NE(rep.tier_reason(MI_TIER_AUTO).find("INT8 ConvRot GEMMs, exact attention, uncertified"), std::string::npos)
        << rep.tier_reason(MI_TIER_AUTO);
  }
  {
    const Report rep(write_dit(dir.path() / "int4.safetensors", Gemm::kInt4));
    EXPECT_EQ(rep.info.loadable, 1) << rep.text(MI_FILE_TEXT_NOTE);
    EXPECT_EQ(rep.info.precision, MI_PRECISION_W4A4);
    EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), "int4_convrot256");
    EXPECT_EQ(rep.info.tiers, bit(MI_TIER_TURBO));
    EXPECT_NE(rep.tier_reason(MI_TIER_AUTO).find("choose tier turbo"), std::string::npos)
        << rep.tier_reason(MI_TIER_AUTO);
    EXPECT_NE(rep.tier_reason(MI_TIER_TURBO).find("Turbo tier: INT4 ConvRot GEMMs"), std::string::npos);
  }
  {
    // A layout the importer reads but the engine runs only after `moss-infer quantize`: a DiT, not loadable.
    const Report rep(write_dit(dir.path() / "scalar.safetensors", Gemm::kInt8ScalarScale));
    EXPECT_EQ(rep.info.kind, MI_FILE_KIND_DIT);
    EXPECT_EQ(rep.info.loadable, 0);
    EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("needs decoding"), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
    EXPECT_EQ(rep.info.tiers, 0u);
    EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), "");
    EXPECT_EQ(rep.tier_reason(MI_TIER_HIGH), "not a loadable DiT file");
  }
}

// A header that would make the importer read and allocate scales larger than the file's own worth: every linear is
// checked against MOSS-Image's shapes before any scale is read.
TEST(CApiInspect, HostileDitShapes) {
  mi::testing::TempDir dir("inspect_hostile");
  constexpr std::int64_t kRows = std::int64_t{1} << 20;
  {
    std::vector<T> q;
    q.push_back({"layers.0.proj.q.weight", DType::kI8, {kRows, 1}});
    q.push_back({"layers.0.proj.q.weight_scale", DType::kF32, {kRows}});
    q.push_back({"layers.0.proj.q.comfy_quant", DType::kU8, {}, R"({"format": "int8_tensorwise"})"});
    const Report rep(write_dit(dir.path() / "rows.safetensors", Gemm::kBf16, q));
    EXPECT_EQ(rep.info.kind, MI_FILE_KIND_DIT);
    EXPECT_EQ(rep.info.loadable, 0);
    const std::string want = "linear 'layers.0.proj.q' is [1048576, 1]; MOSS-Image's is [3072, 3072]";
    EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find(want),
              std::string::npos)
        << rep.text(MI_FILE_TEXT_NOTE);
    EXPECT_EQ(rep.info.tiers, 0u);
  }
  {
    // The right weight with a scale of another size: the importer matches every scale to its linear's rows.
    std::vector<T> q;
    q.push_back({"layers.0.proj.q.weight", DType::kI8, {3072, 3072}});
    q.push_back({"layers.0.proj.q.weight_scale", DType::kF32, {kRows}});
    q.push_back({"layers.0.proj.q.comfy_quant", DType::kU8, {}, kInt8Config});
    const Report rep(write_dit(dir.path() / "scale.safetensors", Gemm::kInt8, q));
    EXPECT_EQ(rep.info.kind, MI_FILE_KIND_DIT);
    EXPECT_EQ(rep.info.loadable, 0);
    EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("weight_scale must be F32 [N]"), std::string::npos)
        << rep.text(MI_FILE_TEXT_NOTE);
  }
}

// Every block GEMM in NVFP4 (ComfyUI's configs, here the legacy metadata form): its swizzled block scales take about
// 283 MiB, and the file gets the NVFP4 verdict, never a size refusal.
TEST(CApiInspect, Nvfp4Dit) {
  mi::testing::TempDir dir("inspect_nvfp4");
  const auto round_up = [](std::int64_t v, std::int64_t to) { return (v + to - 1) / to * to; };
  std::vector<T> t;
  std::string layers;
  std::uint64_t scale_bytes = 0;
  for (const mi::lora::TargetModule& m : mi::lora::moss_dit_targets().modules) {
    if (m.block < 0) continue;
    const std::vector<std::int64_t> sf = {round_up(m.n, 128), round_up(m.k / 16, 4)};
    t.push_back({m.name + ".weight", DType::kU8, {m.n, m.k / 2}});
    t.push_back({m.name + ".weight_scale", DType::kF8E4M3, sf});
    t.push_back({m.name + ".weight_scale_2", DType::kF32, {}});
    layers += (layers.empty() ? "" : ", ") + std::string("\\\"") + m.name + "\\\": {\\\"format\\\": \\\"nvfp4\\\"}";
    scale_bytes += static_cast<std::uint64_t>(sf[0] * sf[1]);
  }
  ASSERT_EQ(t.size(), 42u * 6 * 3);
  EXPECT_GT(scale_bytes, std::uint64_t{256} << 20);
  const Report rep(write_dit(dir.path() / "nvfp4.safetensors", Gemm::kBf16, t,
                             R"({"_quantization_metadata":"{\"layers\": {)" + layers + R"(}}"})"));
  EXPECT_EQ(rep.info.kind, MI_FILE_KIND_DIT);
  EXPECT_EQ(rep.info.loadable, 0);
  // nvfp4_unsupported_reason()'s verdict for this build (no kernels, or no NVFP4 DiT support yet): both point to INT8.
  EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("NVFP4"), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
  EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("INT8"), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(rep.text(MI_FILE_TEXT_NOTE).find("MiB"), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
}

TEST(CApiInspect, Lora) {
  mi::testing::TempDir dir("inspect_lora");
  std::vector<T> t = lora_tensors();
  t.push_back({"training.step_count", DType::kF32, {1}});
  write_file(dir.path() / "lora.safetensors", t, R"({"base_model": "moss-sft"})");
  const Report rep(dir.path() / "lora.safetensors");
  EXPECT_EQ(rep.info.kind, MI_FILE_KIND_LORA);
  EXPECT_EQ(rep.info.loadable, 1);
  EXPECT_EQ(rep.info.lora_rank, 4);
  EXPECT_EQ(rep.info.lora_adapters, 1);
  EXPECT_EQ(rep.info.lora_unknown_keys, 1);
  const std::string want = "1 tensors not applied, first: training.step_count: not an adapter tensor";
  EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find(want),
            std::string::npos)
      << rep.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(rep.text(MI_FILE_TEXT_BASE_MODEL), "moss-sft");
  EXPECT_EQ(rep.text(MI_FILE_TEXT_ADAPTER_TYPES), "lora");
  EXPECT_EQ(rep.info.tiers, 0u);
  EXPECT_EQ(rep.tier_reason(MI_TIER_AUTO), "not a loadable DiT file");

  // A LoRA for another model: a LoRA (its names say so) that the stack refuses, with the stack's reason.
  write_file(dir.path() / "sdxl.safetensors",
             {{"lora_unet_down_blocks_0_attn.lora_down.weight", DType::kBF16, {4, 320}},
              {"lora_unet_down_blocks_0_attn.lora_up.weight", DType::kBF16, {320, 4}}},
             R"({"base_model": "sdxl"})");
  const Report other(dir.path() / "sdxl.safetensors");
  EXPECT_EQ(other.info.kind, MI_FILE_KIND_LORA);
  EXPECT_EQ(other.info.loadable, 0);
  EXPECT_NE(other.text(MI_FILE_TEXT_NOTE).find("no tensor applies to this model"), std::string::npos)
      << other.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(other.text(MI_FILE_TEXT_BASE_MODEL), "sdxl");
  EXPECT_EQ(other.info.lora_adapters, 0);

  // A malformed MOSS-Image LoRA: the factors' ranks differ.
  write_file(dir.path() / "broken.safetensors",
             {{"transformer.layers.0.proj.q.lora_A.weight", DType::kBF16, {4, 3072}},
              {"transformer.layers.0.proj.q.lora_B.weight", DType::kBF16, {3072, 8}}});
  const Report broken(dir.path() / "broken.safetensors");
  EXPECT_EQ(broken.info.kind, MI_FILE_KIND_LORA);
  EXPECT_EQ(broken.info.loadable, 0);
  EXPECT_NE(broken.text(MI_FILE_TEXT_NOTE).find("layers.0.proj.q"), std::string::npos)
      << broken.text(MI_FILE_TEXT_NOTE);
}

// The kinds are the file's control_channels, in channel order: the strings mi_control_image.kind takes.
TEST(CApiInspect, ControlAdapter) {
  mi::testing::TempDir dir("inspect_control");
  write_file(dir.path() / "control.safetensors", lora_tensors(2),
             R"({"control_channels": "color, depth", "base_model": "moss-base"})");
  const Report rep(dir.path() / "control.safetensors");
  EXPECT_EQ(rep.info.kind, MI_FILE_KIND_CONTROL_ADAPTER);
  EXPECT_EQ(rep.info.loadable, 1) << rep.text(MI_FILE_TEXT_NOTE);
  ASSERT_EQ(rep.info.control_kind_count, 2);
  EXPECT_STREQ(mi_file_report_control_kind(rep.r, 0), "color");
  EXPECT_STREQ(mi_file_report_control_kind(rep.r, 1), "depth");
  EXPECT_EQ(mi_file_report_control_kind(rep.r, 2), nullptr);
  EXPECT_EQ(mi_file_report_control_kind(rep.r, -1), nullptr);
  EXPECT_EQ(rep.text(MI_FILE_TEXT_BASE_MODEL), "moss-base");
  EXPECT_EQ(rep.info.lora_rank, 4);
  EXPECT_EQ(rep.info.lora_adapters, 1);
  EXPECT_EQ(rep.info.lora_unknown_keys, 0);  // the embedder is the control adapter's, not an unknown key
  EXPECT_EQ(rep.text(MI_FILE_TEXT_NOTE), "");

  // Without control_channels: the engine's default kinds. Without a LoRA part: the adapter's LoRA stack refuses it.
  write_file(dir.path() / "plain.safetensors", lora_tensors(1));
  const Report plain(dir.path() / "plain.safetensors");
  ASSERT_EQ(plain.info.control_kind_count, 1);
  EXPECT_STREQ(mi_file_report_control_kind(plain.r, 0), "control0");
  std::vector<T> embedder_only = lora_tensors(2);
  embedder_only.erase(embedder_only.begin(), embedder_only.begin() + 2);
  write_file(dir.path() / "embedder.safetensors", embedder_only);
  const Report bare(dir.path() / "embedder.safetensors");
  EXPECT_EQ(bare.info.kind, MI_FILE_KIND_CONTROL_ADAPTER);
  EXPECT_EQ(bare.info.loadable, 0);
  EXPECT_NE(bare.text(MI_FILE_TEXT_NOTE).find("no tensor applies"), std::string::npos) << bare.text(MI_FILE_TEXT_NOTE);
  // control_channels that do not name the embedder's latents: a control adapter the engine refuses.
  write_file(dir.path() / "bad.safetensors", lora_tensors(2), R"({"control_channels": "depth"})");
  const Report bad(dir.path() / "bad.safetensors");
  EXPECT_EQ(bad.info.kind, MI_FILE_KIND_CONTROL_ADAPTER);
  EXPECT_EQ(bad.info.loadable, 0);
  EXPECT_NE(bad.text(MI_FILE_TEXT_NOTE).find("control_channels"), std::string::npos) << bad.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(bad.info.control_kind_count, 0);
  // A non-finite embedder value: generate's read_embedder refuses it, so inspect does.
  write_file(dir.path() / "nan.safetensors", lora_tensors(2));
  {
    const std::uint16_t nan_bf16 = 0x7fc0;
    std::fstream f(dir.path() / "nan.safetensors", std::ios::binary | std::ios::in | std::ios::out);
    char len[8];
    f.read(len, 8);
    std::uint64_t header = 0;
    for (int i = 7; i >= 0; --i) header = (header << 8) | static_cast<unsigned char>(len[i]);
    // A NaN as the embedder weight's first value, found through the header.
    f.seekg(8);
    std::string h(static_cast<std::size_t>(header), '\0');
    f.read(h.data(), static_cast<std::streamsize>(h.size()));
    const std::size_t at = h.find("\"video_embedder.weight\"");
    ASSERT_NE(at, std::string::npos);
    const std::size_t offsets = h.find("\"data_offsets\":[", at) + 16;
    const std::uint64_t begin = std::stoull(h.substr(offsets));
    f.seekp(static_cast<std::streamoff>(8 + header + begin));
    f.write(reinterpret_cast<const char*>(&nan_bf16), 2);
  }
  const Report nan(dir.path() / "nan.safetensors");
  EXPECT_EQ(nan.info.kind, MI_FILE_KIND_CONTROL_ADAPTER);
  EXPECT_EQ(nan.info.loadable, 0);
  EXPECT_NE(nan.text(MI_FILE_TEXT_NOTE).find("non-finite"), std::string::npos) << nan.text(MI_FILE_TEXT_NOTE);
}

// A Qwen3.5 text encoder in ComfyUI's layout: a plain INT8 embedding table (its rows are the vocabulary) and the bf16
// tensors the encoder check reads.
TEST(CApiInspect, TextEncoder) {
  mi::testing::TempDir dir("inspect_te");
  std::vector<T> t = {{"model.language_model.embed_tokens.weight", DType::kI8, {248320, 4}},
                      {"model.language_model.embed_tokens.weight_scale", DType::kF32, {248320}},
                      {"model.language_model.embed_tokens.comfy_quant", DType::kU8, {},
                       R"({"format": "int8_tensorwise"})"},
                      {"model.language_model.norm.weight", DType::kBF16, {4096}}};
  for (int i = 0; i < 32; ++i) {
    const std::string l = "model.language_model.layers." + std::to_string(i) + ".";
    t.push_back({l + "input_layernorm.weight", DType::kBF16, {4096}});
    if (i % 4 == 3) t.push_back({l + "self_attn.q_norm.weight", DType::kBF16, {256}});
  }
  write_file(dir.path() / "te.safetensors", t);
  const Report rep(dir.path() / "te.safetensors");
  EXPECT_EQ(rep.info.kind, MI_FILE_KIND_TEXT_ENCODER);
  EXPECT_EQ(rep.info.loadable, 1) << rep.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(rep.info.precision, MI_PRECISION_BF16);  // no decoder linear is quantized
  EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), "bf16");
  EXPECT_EQ(rep.info.tiers, 0u);

  // A 180 MiB F32 layer tensor: the encoder would read it and keep a bf16 copy, past the 256 MiB inspection reads.
  t.push_back({"model.language_model.layers.0.linear_attn.A_log", DType::kF32, {std::int64_t{45} << 20}});
  write_file(dir.path() / "big.safetensors", t);
  const Report big(dir.path() / "big.safetensors");
  EXPECT_EQ(big.info.kind, MI_FILE_KIND_TEXT_ENCODER);
  EXPECT_EQ(big.info.loadable, 0);
  EXPECT_NE(big.text(MI_FILE_TEXT_NOTE).find("more than 256 MiB"), std::string::npos) << big.text(MI_FILE_TEXT_NOTE);
}

// A FLUX.2 VAE decoder in the diffusers layout, narrowed: the decoder's widths follow from its shapes.
TEST(CApiInspect, Vae) {
  mi::testing::TempDir dir("inspect_vae");
  std::vector<T> t;
  const auto conv = [&](const std::string& n, std::int64_t out, std::int64_t in, std::int64_t k) {
    t.push_back({n + ".weight", DType::kF32, {out, in, k, k}});
    t.push_back({n + ".bias", DType::kF32, {out}});
  };
  const auto norm = [&](const std::string& n, std::int64_t c) {
    t.push_back({n + ".weight", DType::kF32, {c}});
    t.push_back({n + ".bias", DType::kF32, {c}});
  };
  const auto resnet = [&](const std::string& n, std::int64_t in, std::int64_t out) {
    norm(n + ".norm1", in);
    conv(n + ".conv1", out, in, 3);
    norm(n + ".norm2", out);
    conv(n + ".conv2", out, out, 3);
    if (in != out) conv(n + ".conv_shortcut", out, in, 1);
  };
  const std::int64_t outs[] = {128, 128, 64, 32};
  conv("decoder.conv_in", outs[0], 32, 3);
  for (int r = 0; r < 2; ++r) resnet("decoder.mid_block.resnets." + std::to_string(r), outs[0], outs[0]);
  norm("decoder.mid_block.attentions.0.group_norm", outs[0]);
  for (const char* p : {"to_q", "to_k", "to_v", "to_out.0"}) {
    t.push_back({std::string("decoder.mid_block.attentions.0.") + p + ".weight", DType::kF32, {outs[0], outs[0]}});
    t.push_back({std::string("decoder.mid_block.attentions.0.") + p + ".bias", DType::kF32, {outs[0]}});
  }
  for (int b = 0; b < 4; ++b) {
    const std::string u = "decoder.up_blocks." + std::to_string(b);
    for (int r = 0; r < 3; ++r) {
      resnet(u + ".resnets." + std::to_string(r), r == 0 && b > 0 ? outs[b - 1] : outs[b], outs[b]);
    }
    if (b < 3) conv(u + ".upsamplers.0.conv", outs[b], outs[b], 3);
  }
  norm("decoder.conv_norm_out", outs[3]);
  conv("decoder.conv_out", 3, outs[3], 3);
  conv("post_quant_conv", 32, 32, 1);
  t.push_back({"bn.running_mean", DType::kF32, {128}});
  t.push_back({"bn.running_var", DType::kF32, {128}});
  write_file(dir.path() / "vae.safetensors", t);
  const Report rep(dir.path() / "vae.safetensors");
  EXPECT_EQ(rep.info.kind, MI_FILE_KIND_VAE) << rep.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(rep.info.loadable, 1);
  EXPECT_EQ(rep.text(MI_FILE_TEXT_NOTE), "");
}

// Hostile or broken files are answers (MI_FILE_KIND_UNKNOWN with the reader's reason), never crashes or errors.
TEST(CApiInspect, MalformedFiles) {
  mi::testing::TempDir dir("inspect_bad");
  const auto unknown = [&](const std::string& name, const std::string& bytes, const std::string& reason) {
    SCOPED_TRACE(name);
    write_bytes(dir.path() / name, bytes);
    const Report rep(dir.path() / name);
    EXPECT_EQ(rep.info.kind, MI_FILE_KIND_UNKNOWN);
    EXPECT_EQ(rep.info.loadable, 0);
    EXPECT_EQ(rep.info.tensor_count, 0);
    EXPECT_EQ(rep.info.file_bytes, bytes.size());
    ASSERT_FALSE(reason.empty());
    EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find(reason), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
  };
  const auto le64 = [](std::uint64_t v) {
    std::string s(8, '\0');
    for (int i = 0; i < 8; ++i) s[static_cast<std::size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
    return s;
  };
  std::string garbage(4096, '\0');
  for (std::size_t i = 0; i < garbage.size(); ++i) garbage[i] = static_cast<char>((i * 2654435761u) >> 13);
  unknown("garbage.bin", garbage, "header length");
  unknown("empty.safetensors", "", "too small");
  unknown("truncated.safetensors", le64(1000) + "{\"a\":{\"dtype\":\"BF16\"", "truncated");
  unknown("huge_header.safetensors", le64(std::uint64_t{1} << 40) + "{}", "16 MiB");
  unknown("max_header.safetensors", le64(~std::uint64_t{0}) + "{}", "16 MiB");
  unknown("deep.safetensors", le64(20000) + std::string(10000, '[') + std::string(10000, ']'), "not valid JSON");
  // A 2 MiB tensor claimed in a file of a few bytes: never read.
  const std::string past_end = R"({"x":{"dtype":"BF16","shape":[1048576],"data_offsets":[0,2097152]}})";
  unknown("past_end.safetensors", le64(past_end.size()) + past_end + "ab", "past the data section");
  unknown("model.gguf", "GGUF" + std::string(60, '\0'), "GGUF");
  unknown("model.pt", std::string("PK\x03\x04", 4) + std::string(60, '\0'), "pickle");
  // A valid header with nothing the engine reads.
  write_file(dir.path() / "other.safetensors", {{"backbone.embeddings.cls_token", DType::kF32, {1, 4}}});
  const Report other(dir.path() / "other.safetensors");
  EXPECT_EQ(other.info.kind, MI_FILE_KIND_UNKNOWN);
  EXPECT_EQ(other.info.tensor_count, 1);
  EXPECT_NE(other.text(MI_FILE_TEXT_NOTE).find("backbone.embeddings.cls_token"), std::string::npos);
  // Alpha parameters without LoRA factors (another model's): not a LoRA.
  write_file(dir.path() / "alpha.safetensors", {{"blocks.0.attn.alpha", DType::kF32, {1}},
                                                {"blocks.0.attn.weight", DType::kF32, {4, 4}}});
  const Report alpha(dir.path() / "alpha.safetensors");
  EXPECT_EQ(alpha.info.kind, MI_FILE_KIND_UNKNOWN);
  EXPECT_EQ(alpha.info.loadable, 0);
  EXPECT_FALSE(alpha.text(MI_FILE_TEXT_NOTE).empty());
}

// A FIFO is refused at once: opening one for reading would wait for a writer.
TEST(CApiInspect, Fifo) {
#ifdef _WIN32
  GTEST_SKIP() << "no mkfifo on Windows";
#else
  mi::testing::TempDir dir("inspect_fifo");
  const fs::path fifo = dir.path() / "pipe.safetensors";
  ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);
  mi_file_report* r = nullptr;
  EXPECT_EQ(mi_inspect_file(fifo.c_str(), &r), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(std::string(mi_last_error_message()).find("not a regular file"), std::string::npos)
      << mi_last_error_message();
  EXPECT_EQ(r, nullptr);
#endif
}

TEST(CApiInspect, ArgumentsAndAccessors) {
  EXPECT_NE(mi_capabilities() & MI_CAP_INSPECT, 0u);
  EXPECT_EQ(MI_CAP_INSPECT, 0x20000u);
  mi::testing::TempDir dir("inspect_args");
  mi_file_report* r = reinterpret_cast<mi_file_report*>(&dir);  // must be reset on failure
  EXPECT_EQ(mi_inspect_file(nullptr, &r), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(r, nullptr);
  EXPECT_EQ(mi_inspect_file("", &r), MI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(mi_inspect_file("x", nullptr), MI_ERR_INVALID_ARGUMENT);
  const std::string missing = mi::util::path_utf8(dir.path() / "missing.safetensors");
  EXPECT_EQ(mi_inspect_file(missing.c_str(), &r), MI_ERR_IO);
  EXPECT_NE(std::string(mi_last_error_message()).find("no such file"), std::string::npos) << mi_last_error_message();
  EXPECT_EQ(r, nullptr);
  const std::string folder = mi::util::path_utf8(dir.path());
  EXPECT_EQ(mi_inspect_file(folder.c_str(), &r), MI_ERR_INVALID_ARGUMENT);
  EXPECT_NE(std::string(mi_last_error_message()).find("not a regular file"), std::string::npos);
#ifndef _WIN32
  EXPECT_EQ(mi_inspect_file("/dev/zero", &r), MI_ERR_INVALID_ARGUMENT);  // would never end
  // A folder the process may not search: the system's reason, not "no such file".
  if (::geteuid() != 0) {
    const fs::path locked = dir.path() / "locked";
    fs::create_directory(locked);
    write_bytes(locked / "x.safetensors", "x");
    fs::permissions(locked, fs::perms::none);
    struct Restore {  // whatever the assertions do, so TempDir can remove the folder
      fs::path dir;
      ~Restore() {
        std::error_code ec;
        fs::permissions(dir, fs::perms::owner_all, ec);
      }
    } restore{locked};
    const std::string inside = mi::util::path_utf8(locked / "x.safetensors");
    EXPECT_EQ(mi_inspect_file(inside.c_str(), &r), MI_ERR_IO);
    EXPECT_NE(std::string(mi_last_error_message()).find("ermission denied"), std::string::npos)
        << mi_last_error_message();
  }
#endif
  EXPECT_EQ(r, nullptr);

  mi_file_info info{};
  info.struct_size = sizeof info;
  EXPECT_EQ(mi_file_report_info(nullptr, &info), MI_ERR_INVALID_ARGUMENT);
  write_file(dir.path() / "lora.safetensors", lora_tensors());
  const Report rep(dir.path() / "lora.safetensors");
  EXPECT_EQ(mi_file_report_info(rep.r, nullptr), MI_ERR_INVALID_ARGUMENT);
  info.struct_size = sizeof info - 8;
  EXPECT_EQ(mi_file_report_info(rep.r, &info), MI_ERR_ABI_MISMATCH);
  EXPECT_EQ(mi_file_report_text(nullptr, MI_FILE_TEXT_NOTE), nullptr);
  EXPECT_EQ(mi_file_report_text(rep.r, static_cast<mi_file_text_kind>(99)), nullptr);
  EXPECT_EQ(mi_file_report_control_kind(nullptr, 0), nullptr);
  EXPECT_EQ(mi_file_report_control_kind(rep.r, 0), nullptr);  // a LoRA has no kinds
  EXPECT_EQ(mi_file_report_tier_reason(nullptr, MI_TIER_AUTO), nullptr);
  EXPECT_EQ(mi_file_report_tier_reason(rep.r, static_cast<mi_tier>(4)), nullptr);
  EXPECT_EQ(mi_file_report_tier_reason(rep.r, MI_TIER_MAX_ENUM), nullptr);
  mi_file_report_free(nullptr);
}

// The community files the engine was verified with, when present: each in its role, with its facts.
TEST(CApiInspect, CommunityFiles) {
  const char* env = std::getenv("MOSS_COMMUNITY_MODELS");
  const fs::path c = env != nullptr && *env != '\0' ? fs::path(env) : fs::path("/workspace/Models/community");
  struct Expect {
    const char* file;
    mi_file_kind kind;
    int loadable;
    const char* quantization;
    std::uint32_t tiers;
    mi_precision precision;
  };
  const Expect files[] = {
      {"comfy-org/moss_image_sft_int8_convrot.safetensors", MI_FILE_KIND_DIT, 1, "int8_convrot256",
       bit(MI_TIER_AUTO) | bit(MI_TIER_HIGH), MI_PRECISION_INT8},
      {"toxicdog/moss_image_sft_int4_convrot.safetensors", MI_FILE_KIND_DIT, 1, "int4_convrot256", bit(MI_TIER_TURBO),
       MI_PRECISION_W4A4},
      {"tsolful/moss_image_sft_W4A8_GPTQ.safetensors", MI_FILE_KIND_DIT, 0, "", 0, MI_PRECISION_BF16},  // to quantize
      {"bennydaball/moss_image_sft_nvfp4.safetensors", MI_FILE_KIND_DIT, 0, "", 0, MI_PRECISION_BF16},  // NVFP4
      {"comfy-org/qwen3.5_9b_int8_convrot.safetensors", MI_FILE_KIND_TEXT_ENCODER, 1, "int8_convrot", 0,
       MI_PRECISION_INT8},
      {"comfy-org/qwen3.5_9b_w4a8_gptq.safetensors", MI_FILE_KIND_TEXT_ENCODER, 1, "w4a8", 0, MI_PRECISION_W4A8},
      {"comfy-org/flux2-vae.safetensors", MI_FILE_KIND_VAE, 1, "", 0, MI_PRECISION_BF16},
      {"comfy-org/flux2-vae_small.safetensors", MI_FILE_KIND_VAE, 1, "", 0, MI_PRECISION_BF16},
      {"congruency/MOSS-Image-Turbo-MFD/MOSS-Image-SFT-Turbo-MFD-v0.1.comfy.safetensors", MI_FILE_KIND_LORA, 1, "", 0,
       MI_PRECISION_BF16},
      {"comfyanonymous/MOSS_image_MAX_ULTRA_hires_lora_test/moss_hires_lora_test_v0.1.safetensors",
       MI_FILE_KIND_LORA, 1, "", 0, MI_PRECISION_BF16},
      {"comfy-org/moss_image_color_depth_control.safetensors", MI_FILE_KIND_CONTROL_ADAPTER, 1, "", 0,
       MI_PRECISION_BF16},
      {"lodestones/MOSS-Image-SFT-8-steps-dmd/time_embedding.safetensors", MI_FILE_KIND_UNKNOWN, 0, "", 0,
       MI_PRECISION_BF16},
  };
  int checked = 0;
  for (const Expect& e : files) {
    if (!fs::is_regular_file(c / e.file)) continue;
    SCOPED_TRACE(e.file);
    const Report rep(c / e.file);
    EXPECT_EQ(rep.info.kind, e.kind);
    EXPECT_EQ(rep.info.loadable, e.loadable) << rep.text(MI_FILE_TEXT_NOTE);
    EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), e.quantization);
    EXPECT_EQ(rep.info.tiers, e.tiers);
    EXPECT_EQ(rep.info.precision, e.precision);
    if (e.kind == MI_FILE_KIND_CONTROL_ADAPTER) {
      ASSERT_EQ(rep.info.control_kind_count, 2);
      EXPECT_STREQ(mi_file_report_control_kind(rep.r, 0), "color");
      EXPECT_STREQ(mi_file_report_control_kind(rep.r, 1), "depth");
      EXPECT_EQ(rep.info.lora_rank, 64);
    }
    if (e.kind == MI_FILE_KIND_UNKNOWN) {
      EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("distill bundle file"), std::string::npos)
          << rep.text(MI_FILE_TEXT_NOTE);
    }
    if (std::string(e.file).find("nvfp4") != std::string::npos) {
      EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("NVFP4"), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
      EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("INT8"), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
    }
    ++checked;
  }
  if (checked == 0) {
    GTEST_SKIP() << "no community file under " << mi::util::path_utf8(c) << " (set MOSS_COMMUNITY_MODELS)";
  }
}

}  // namespace
