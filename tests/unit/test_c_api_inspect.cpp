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
#include <vector>

#include "test_util.h"
#include "util/path.h"
#include "weights/safetensors.h"

namespace {

namespace fs = std::filesystem;
using mi::weights::DType;

// A tensor with explicit bytes: zeros, an F32 `*weight_scale` of ones (the readers check scales are finite and > 0),
// or a comfy_quant config's JSON.
struct T {
  std::string name;
  DType dtype;
  std::vector<std::int64_t> shape;
  std::string json;  // comfy_quant: the U8 bytes
};

// Laid out like the safetensors library writes files: larger dtypes first, the header padded to 8 bytes.
void write_file(const fs::path& path, std::vector<T> ts, const std::string& metadata_json = "") {
  std::stable_sort(ts.begin(), ts.end(), [](const T& a, const T& b) { return mi::weights::dtype_size(a.dtype) > mi::weights::dtype_size(b.dtype); });
  std::vector<std::string> data;
  std::string h = "{";
  if (!metadata_json.empty()) h += "\"__metadata__\":" + metadata_json;
  std::uint64_t off = 0;
  for (T& t : ts) {
    std::string bytes;
    if (!t.json.empty()) {
      bytes = t.json;
      t.shape = {static_cast<std::int64_t>(bytes.size())};
    } else {
      std::uint64_t n = 1;
      for (const std::int64_t d : t.shape) n *= static_cast<std::uint64_t>(d);
      bytes.assign(n * mi::weights::dtype_size(t.dtype), '\0');
      if (t.dtype == DType::kF32 && t.name.ends_with("weight_scale")) {
        const float one = 1.0f;
        for (std::uint64_t i = 0; i < n; ++i) std::memcpy(bytes.data() + 4 * i, &one, 4);
      }
    }
    if (h.size() > 1) h += ",";
    h += "\"" + t.name + "\":{\"dtype\":\"" + mi::weights::dtype_name(t.dtype) + "\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
    h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes.size()) + "]}";
    off += bytes.size();
    data.push_back(std::move(bytes));
  }
  h += "}";
  h.append((8 - h.size() % 8) % 8, ' ');
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const std::uint64_t len = h.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((len >> (8 * i)) & 0xFF));
  out.write(h.data(), static_cast<std::streamsize>(h.size()));
  for (const std::string& d : data) out.write(d.data(), static_cast<std::streamsize>(d.size()));
}

void write_bytes(const fs::path& path, const std::string& bytes) {
  std::ofstream(path, std::ios::binary | std::ios::trunc).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

enum class Gemm { kBf16, kInt8, kInt4, kInt8ScalarScale };

// A single-file DiT with MOSS-Image's names and the shapes check_supported_dit_file reads (42 blocks, the bf16
// embedders), and small proj.q / ffn_down GEMMs [4, 256] stored as `gemm` (ComfyUI comfy_quant configs).
fs::path write_dit(const fs::path& path, Gemm gemm) {
  std::vector<T> t;
  for (int i = 0; i < 42; ++i) {
    const std::string b = "layers." + std::to_string(i) + ".";
    t.push_back({b + "norm1.weight", DType::kBF16, {3072}, ""});
    t.push_back({b + "q_norm.weight", DType::kBF16, {128}, ""});
    for (const char* m : {"proj.q", "proj.ffn_down"}) {
      const std::string w = b + m;
      switch (gemm) {
        case Gemm::kBf16: t.push_back({w + ".weight", DType::kBF16, {4, 256}, ""}); break;
        case Gemm::kInt8:
          t.push_back({w + ".weight", DType::kI8, {4, 256}, ""});
          t.push_back({w + ".weight_scale", DType::kF32, {4}, ""});
          t.push_back({w + ".comfy_quant", DType::kU8, {}, R"({"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256})"});
          break;
        case Gemm::kInt4:
          t.push_back({w + ".weight", DType::kI8, {4, 128}, ""});
          t.push_back({w + ".weight_scale", DType::kF32, {4}, ""});
          t.push_back({w + ".comfy_quant", DType::kU8, {}, R"({"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64})"});
          break;
        case Gemm::kInt8ScalarScale:
          t.push_back({w + ".weight", DType::kI8, {4, 256}, ""});
          t.push_back({w + ".weight_scale", DType::kF32, {}, ""});
          t.push_back({w + ".comfy_quant", DType::kU8, {}, R"({"format": "int8_tensorwise"})"});
          break;
      }
    }
  }
  t.push_back({"video_embedder.weight", DType::kBF16, {3072, 128}, ""});
  t.push_back({"text_embedder.proj.weight", DType::kBF16, {3072, 4096}, ""});
  t.push_back({"time_embedding.0.weight", DType::kBF16, {3072, 256}, ""});
  t.push_back({"final_linear_video.weight", DType::kBF16, {128, 3072}, ""});
  write_file(path, t);
  return path;
}

// A PEFT LoRA on block 0's q (rank 4), optionally with an embedder widened for `kinds` control latents.
std::vector<T> lora_tensors(int control_groups = 0) {
  std::vector<T> t = {{"transformer.layers.0.proj.q.lora_A.weight", DType::kBF16, {4, 3072}, ""},
                      {"transformer.layers.0.proj.q.lora_B.weight", DType::kBF16, {3072, 4}, ""}};
  if (control_groups > 0) {
    t.push_back({"video_embedder.weight", DType::kBF16, {3072, 128 * (1 + control_groups)}, ""});
    t.push_back({"video_embedder.bias", DType::kBF16, {3072}, ""});
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
  std::string text(mi_file_text_kind k) const { return mi_file_report_text(r, k); }
  std::string tier_reason(mi_tier t) const { return mi_file_report_tier_reason(r, t); }
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
  EXPECT_EQ(rep.info.tensor_count, 42 * 4 + 4);
  EXPECT_EQ(rep.info.file_bytes, fs::file_size(dir.path() / "dit.safetensors"));
  EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), "bf16");
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
    EXPECT_NE(rep.tier_reason(MI_TIER_REFERENCE).find("quantized"), std::string::npos) << rep.tier_reason(MI_TIER_REFERENCE);
    EXPECT_NE(rep.tier_reason(MI_TIER_AUTO).find("INT8 ConvRot GEMMs, exact attention, uncertified"), std::string::npos)
        << rep.tier_reason(MI_TIER_AUTO);
  }
  {
    const Report rep(write_dit(dir.path() / "int4.safetensors", Gemm::kInt4));
    EXPECT_EQ(rep.info.loadable, 1) << rep.text(MI_FILE_TEXT_NOTE);
    EXPECT_EQ(rep.info.precision, MI_PRECISION_W4A4);
    EXPECT_EQ(rep.text(MI_FILE_TEXT_QUANTIZATION), "int4_convrot256");
    EXPECT_EQ(rep.info.tiers, bit(MI_TIER_TURBO));
    EXPECT_NE(rep.tier_reason(MI_TIER_AUTO).find("choose tier turbo"), std::string::npos) << rep.tier_reason(MI_TIER_AUTO);
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

TEST(CApiInspect, Lora) {
  mi::testing::TempDir dir("inspect_lora");
  std::vector<T> t = lora_tensors();
  t.push_back({"training.step_count", DType::kF32, {1}, ""});
  write_file(dir.path() / "lora.safetensors", t, R"({"base_model": "moss-sft"})");
  const Report rep(dir.path() / "lora.safetensors");
  EXPECT_EQ(rep.info.kind, MI_FILE_KIND_LORA);
  EXPECT_EQ(rep.info.loadable, 1);
  EXPECT_EQ(rep.info.lora_rank, 4);
  EXPECT_EQ(rep.info.lora_adapters, 1);
  EXPECT_EQ(rep.info.lora_unknown_keys, 1);
  EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("1 tensors not applied, first: training.step_count: not an adapter tensor"),
            std::string::npos)
      << rep.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(rep.text(MI_FILE_TEXT_BASE_MODEL), "moss-sft");
  EXPECT_EQ(rep.text(MI_FILE_TEXT_ADAPTER_TYPES), "lora");
  EXPECT_EQ(rep.info.tiers, 0u);
  EXPECT_EQ(rep.tier_reason(MI_TIER_AUTO), "not a loadable DiT file");

  // A LoRA for another model: no tensor applies to the DiT. The DiT check names what it is.
  write_file(dir.path() / "sdxl.safetensors", {{"lora_unet_down_blocks_0_attn.lora_down.weight", DType::kBF16, {4, 320}, ""},
                                               {"lora_unet_down_blocks_0_attn.lora_up.weight", DType::kBF16, {320, 4}, ""}});
  const Report other(dir.path() / "sdxl.safetensors");
  EXPECT_EQ(other.info.kind, MI_FILE_KIND_UNKNOWN);
  EXPECT_EQ(other.info.loadable, 0);
  EXPECT_NE(other.text(MI_FILE_TEXT_NOTE).find("found a LoRA"), std::string::npos) << other.text(MI_FILE_TEXT_NOTE);
  EXPECT_EQ(other.info.lora_adapters, 0);
}

// The kinds are the file's control_channels, in channel order: the strings mi_control_image.kind takes.
TEST(CApiInspect, ControlAdapter) {
  mi::testing::TempDir dir("inspect_control");
  write_file(dir.path() / "control.safetensors", lora_tensors(2), R"({"control_channels": "color, depth", "base_model": "moss-base"})");
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
    EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find(reason), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
  };
  const auto le64 = [](std::uint64_t v) {
    std::string s(8, '\0');
    for (int i = 0; i < 8; ++i) s[static_cast<std::size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
    return s;
  };
  std::string garbage(4096, '\0');
  for (std::size_t i = 0; i < garbage.size(); ++i) garbage[i] = static_cast<char>((i * 2654435761u) >> 13);
  unknown("garbage.bin", garbage, "");
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
  write_file(dir.path() / "other.safetensors", {{"backbone.embeddings.cls_token", DType::kF32, {1, 4}, ""}});
  const Report other(dir.path() / "other.safetensors");
  EXPECT_EQ(other.info.kind, MI_FILE_KIND_UNKNOWN);
  EXPECT_EQ(other.info.tensor_count, 1);
  EXPECT_NE(other.text(MI_FILE_TEXT_NOTE).find("backbone.embeddings.cls_token"), std::string::npos);
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
  };
  const Expect files[] = {
      {"comfy-org/moss_image_sft_int8_convrot.safetensors", MI_FILE_KIND_DIT, 1, "int8_convrot256", bit(MI_TIER_AUTO) | bit(MI_TIER_HIGH)},
      {"toxicdog/moss_image_sft_int4_convrot.safetensors", MI_FILE_KIND_DIT, 1, "int4_convrot256", bit(MI_TIER_TURBO)},
      {"tsolful/moss_image_sft_W4A8_GPTQ.safetensors", MI_FILE_KIND_DIT, 0, "", 0},  // needs `moss-infer quantize`
      {"bennydaball/moss_image_sft_nvfp4.safetensors", MI_FILE_KIND_DIT, 0, "", 0},  // NVFP4: not in this version
      {"comfy-org/qwen3.5_9b_int8_convrot.safetensors", MI_FILE_KIND_TEXT_ENCODER, 1, "int8_convrot", 0},
      {"comfy-org/qwen3.5_9b_w4a8_gptq.safetensors", MI_FILE_KIND_TEXT_ENCODER, 1, "w4a8", 0},
      {"comfy-org/flux2-vae.safetensors", MI_FILE_KIND_VAE, 1, "", 0},
      {"comfy-org/flux2-vae_small.safetensors", MI_FILE_KIND_VAE, 1, "", 0},
      {"congruency/MOSS-Image-Turbo-MFD/MOSS-Image-SFT-Turbo-MFD-v0.1.comfy.safetensors", MI_FILE_KIND_LORA, 1, "", 0},
      {"comfyanonymous/MOSS_image_MAX_ULTRA_hires_lora_test/moss_hires_lora_test_v0.1.safetensors", MI_FILE_KIND_LORA, 1, "", 0},
      {"comfy-org/moss_image_color_depth_control.safetensors", MI_FILE_KIND_CONTROL_ADAPTER, 1, "", 0},
      {"lodestones/MOSS-Image-SFT-8-steps-dmd/time_embedding.safetensors", MI_FILE_KIND_UNKNOWN, 0, "", 0},
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
    if (e.kind == MI_FILE_KIND_CONTROL_ADAPTER) {
      ASSERT_EQ(rep.info.control_kind_count, 2);
      EXPECT_STREQ(mi_file_report_control_kind(rep.r, 0), "color");
      EXPECT_STREQ(mi_file_report_control_kind(rep.r, 1), "depth");
      EXPECT_EQ(rep.info.lora_rank, 64);
    }
    if (e.kind == MI_FILE_KIND_UNKNOWN) {
      EXPECT_NE(rep.text(MI_FILE_TEXT_NOTE).find("distill bundle file"), std::string::npos) << rep.text(MI_FILE_TEXT_NOTE);
    }
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no community file under " << mi::util::path_utf8(c) << " (set MOSS_COMMUNITY_MODELS)";
}

}  // namespace
