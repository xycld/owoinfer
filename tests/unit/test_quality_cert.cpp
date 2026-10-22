// pipeline/quality_cert: certificate round trip (text and file, exact doubles), untrusted-input validation (corrupt,
// truncated, oversized, inconsistent or lax documents are rejected whole; mutation fuzz), the tier gate tables and
// their OCR rule, engine numerics per configuration, matching (model content, configuration, GPU class, numerics,
// prompt set, OCR prompt counts, request scope), the file search with its pre-check and cache-folder cap, the
// stat-keyed hash cache, the certification harness's fixture and tables (cross-language drift), the report object,
// and the writer/reader under a comma-decimal process locale.
#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <locale>
#include <string>
#include <vector>

#include "pipeline/quality_cert.h"
#include "test_util.h"
#include "util/json.h"
#include "util/xxh64.h"

namespace {

namespace fs = std::filesystem;
using namespace mi::pipeline;

const fs::path kData = fs::path(MOSSINFER_SOURCE_DIR) / "tests/data/quality_cert";

std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_bytes(const fs::path& p, const std::string& s) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out << s;
}

std::string pattern(std::size_t n, int salt = 0) {
  std::string s(n, '\0');
  for (std::size_t i = 0; i < n; ++i) s[i] = static_cast<char>(mi::testing::pattern_byte(i + static_cast<std::size_t>(salt)));
  return s;
}

ComponentIdentity one_file(std::string name, std::uint64_t size, std::uint64_t hash) {
  return ComponentIdentity{{FileIdentity{std::move(name), size, hash}}};
}

// The gates of `tier`, each passing comfortably (or the first one failing).
std::vector<Gate> gates_for(QualityTier tier, bool fail_first = false, bool gateable = true) {
  std::vector<Gate> out;
  for (const GateSpec& s : required_gates(tier, gateable)) {
    Gate g;
    g.name = std::string(s.name);
    g.op = s.op;
    g.threshold = s.limit;
    g.value = s.op == GateOp::kLe ? s.limit * 0.5 : (s.limit > 0 ? s.limit + 0.5 * (1 - s.limit) : s.limit * 0.5);
    if (fail_first && out.empty()) g.value = s.op == GateOp::kLe ? s.limit + 0.01 : s.limit - 0.01;
    g.pass = s.op == GateOp::kLe ? g.value <= g.threshold : g.value >= g.threshold;
    out.push_back(g);
  }
  return out;
}

CertScope test_scope() { return CertScope{2102784, {"euler"}, 30, 50, false}; }

Certificate sample() {
  Certificate c;
  c.config = {DitPrecision::kInt8ConvRot, AttentionMode::kQ8Fp8, TePrecision::kInt8ConvRot, false};
  c.tier = QualityTier::kHigh;
  c.model.dit = one_file("moss_image_sft_int8_convrot.safetensors", 9'876'543'210ull, 0xfedcba9876543210ull);
  c.model.te = ComponentIdentity{{FileIdentity{"te-00001.safetensors", 4'000'000'000ull, 0x0123456789abcdefull},
                                  FileIdentity{"te-00002.safetensors", 1, 0}}};
  c.numerics = engine_numerics(c.config);
  c.gpu_arch = "sm_89";
  c.prompt_set = std::string(kCertPromptSetId);
  c.prompts = kCertPrompts;
  c.prompt_set_sha256 = std::string(kCertPromptSetSha256Base);
  c.scope = test_scope();
  c.date = "2026-10-21T12:34:56Z";
  c.gates = gates_for(QualityTier::kHigh);
  c.gates[1].value = 1.0 / 30.0;  // same_seed.lpips_mean; not exactly representable in decimal: must round-trip
  c.passed = true;
  c.info = {{"same_seed.ocr_prompts", 13}, {"reference.ocr_prompts", 13}, {"noise.bf16_vs_official.lpips_mean", 0.1 + 0.2},
            {"tiny", 1e-300}, {"neg", -2.5}};
  c.tool = "certify.py 2 \"quoted\" \xE2\x9C\x93";
  return c;
}

ModelIdentity identity_of(const Certificate& c) { return ModelIdentity{c.model.dit, c.model.te}; }

CertificateQuery query_of(const Certificate& c) { return CertificateQuery{c.config, c.gpu_arch, c.numerics, std::nullopt}; }

void set_info(Certificate& c, const std::string& key, double v) {
  for (auto& [k, x] : c.info) {
    if (k == key) {
      x = v;
      return;
    }
  }
  c.info.emplace_back(key, v);
}

void expect_same(const Certificate& a, const Certificate& b) {
  EXPECT_EQ(a.tier, b.tier);
  EXPECT_EQ(a.config, b.config);
  ASSERT_EQ(a.model.dit.files.size(), b.model.dit.files.size());
  for (std::size_t i = 0; i < a.model.dit.files.size(); ++i) {
    EXPECT_EQ(a.model.dit.files[i].name, b.model.dit.files[i].name);
    EXPECT_EQ(a.model.dit.files[i].size, b.model.dit.files[i].size);
    EXPECT_EQ(a.model.dit.files[i].xxh64, b.model.dit.files[i].xxh64);
  }
  ASSERT_EQ(a.model.te.has_value(), b.model.te.has_value());
  if (a.model.te) {
    EXPECT_TRUE(a.model.te->same_content(*b.model.te));
    for (std::size_t i = 0; i < a.model.te->files.size(); ++i) EXPECT_EQ(a.model.te->files[i].name, b.model.te->files[i].name);
  }
  EXPECT_EQ(a.numerics, b.numerics);
  EXPECT_EQ(a.gpu_arch, b.gpu_arch);
  EXPECT_EQ(a.prompt_set, b.prompt_set);
  EXPECT_EQ(a.prompts, b.prompts);
  EXPECT_EQ(a.prompt_set_sha256, b.prompt_set_sha256);
  EXPECT_EQ(a.scope, b.scope);
  EXPECT_EQ(a.date, b.date);
  ASSERT_EQ(a.gates.size(), b.gates.size());
  for (std::size_t i = 0; i < a.gates.size(); ++i) {
    EXPECT_EQ(a.gates[i].name, b.gates[i].name);
    EXPECT_EQ(a.gates[i].value, b.gates[i].value) << a.gates[i].name;  // bit-exact
    EXPECT_EQ(a.gates[i].op, b.gates[i].op);
    EXPECT_EQ(a.gates[i].threshold, b.gates[i].threshold);
    EXPECT_EQ(a.gates[i].pass, b.gates[i].pass);
  }
  EXPECT_EQ(a.passed, b.passed);
  EXPECT_EQ(a.info, b.info);
  EXPECT_EQ(a.tool, b.tool);
}

void replace_once(std::string& s, const std::string& from, const std::string& to) {
  const std::size_t at = s.find(from);
  ASSERT_NE(at, std::string::npos) << from;
  s.replace(at, from.size(), to);
}

bool has_key(const EngineNumerics& n, const std::string& k) {
  return std::any_of(n.begin(), n.end(), [&](const auto& e) { return e.first == k; });
}

TEST(QualityCert, TokensAndTiers) {
  for (auto t : {QualityTier::kReference, QualityTier::kHigh, QualityTier::kTurbo}) EXPECT_EQ(parse_quality_tier(cert_token(t)), t);
  for (auto p : {DitPrecision::kBf16, DitPrecision::kInt8ConvRot, DitPrecision::kW4A4}) EXPECT_EQ(parse_dit_precision(cert_token(p)), p);
  for (auto a : {AttentionMode::kExact, AttentionMode::kQ8Fp16, AttentionMode::kQ8Fp16F32Acc, AttentionMode::kQ8Fp8,
                 AttentionMode::kQ8Fp8F32Acc}) {
    EXPECT_EQ(parse_attention_mode(cert_token(a)), a);
  }
  for (auto p : {TePrecision::kBf16, TePrecision::kInt8ConvRot, TePrecision::kW4A8}) EXPECT_EQ(parse_te_precision(cert_token(p)), p);
  EXPECT_FALSE(parse_quality_tier("auto").has_value());
  EXPECT_FALSE(parse_attention_mode("Exact").has_value());
  EXPECT_FALSE(parse_dit_precision("").has_value());

  EXPECT_EQ(tier_of({}), QualityTier::kReference);
  EXPECT_EQ(tier_of({DitPrecision::kInt8ConvRot, AttentionMode::kExact, TePrecision::kBf16, false}), QualityTier::kHigh);
  EXPECT_EQ(tier_of({DitPrecision::kBf16, AttentionMode::kQ8Fp16, TePrecision::kBf16, false}), QualityTier::kHigh);
  EXPECT_EQ(tier_of({DitPrecision::kBf16, AttentionMode::kExact, TePrecision::kInt8ConvRot, false}), QualityTier::kHigh);
  EXPECT_EQ(tier_of({DitPrecision::kW4A4, AttentionMode::kQ8Fp8, TePrecision::kW4A8, false}), QualityTier::kTurbo);

  EXPECT_EQ(gpu_arch_class(8, 9), "sm_89");
  EXPECT_EQ(gpu_arch_class(12, 0), "sm_120");
}

// Every tier's limits (AC21, the tier table, and the paired / worst-case gates), with and without official OCR.
TEST(QualityCert, GateTables) {
  const auto limits = [](QualityTier t, bool g) {
    std::vector<std::pair<std::string, double>> out;
    for (const GateSpec& s : required_gates(t, g)) out.emplace_back(std::string(s.name), s.limit);
    return out;
  };
  using L = std::vector<std::pair<std::string, double>>;
  const L det = {{"determinism.mismatches", 0.0}};
  const L same_seed = {{"determinism.mismatches", 0.0}, {"same_seed.lpips_mean", 0.12}, {"same_seed.lpips_max", 0.25}, {"same_seed.dino_mean", 0.90},
                       {"same_seed.ocr_delta", -0.05}};
  const auto with = [](L a, const L& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
  };
  const auto near = [](const L& got, const L& want) {
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i].first, want[i].first);
      EXPECT_NEAR(got[i].second, want[i].second, 1e-12) << got[i].first;
    }
  };
  near(limits(QualityTier::kReference, true), with(same_seed, {{"reference.lpips_mean", 0.088},
                                                               {"reference.lpips_paired_max", 0.088 / 1.5},
                                                               {"reference.dino_mean", 0.925},
                                                               {"reference.ocr_delta", -0.05}}));
  near(limits(QualityTier::kHigh, true), with(same_seed, {{"reference.lpips_mean", 0.132},
                                                          {"reference.lpips_paired_max", 0.088},
                                                          {"reference.dino_mean", 0.8875},
                                                          {"reference.ocr_delta", -0.075}}));
  near(limits(QualityTier::kHigh, false), with(same_seed, {{"reference.lpips_mean", 0.132},
                                                           {"reference.lpips_paired_max", 0.088},
                                                           {"reference.dino_mean", 0.8875}}));
  near(limits(QualityTier::kTurbo, true), with(det, {{"reference.lpips_mean", 0.264}, {"reference.lpips_paired_max", 0.176},
                                           {"reference.dino_mean", 0.775}, {"reference.ocr_delta", -0.10}}));
  near(limits(QualityTier::kTurbo, false), with(det, {{"reference.lpips_mean", 0.264}, {"reference.lpips_paired_max", 0.176},
                                            {"reference.dino_mean", 0.775}, {"same_seed.ocr_delta", -0.10}}));
  for (const QualityTier t : {QualityTier::kReference, QualityTier::kHigh, QualityTier::kTurbo}) {
    for (const bool g : {true, false}) EXPECT_EQ(required_gates(t, g).front().name, "determinism.mismatches");
  }
}

TEST(QualityCert, NumericsDependOnTheConfiguration) {
  const CertConfig ref{};
  const CertConfig int8{DitPrecision::kInt8ConvRot, AttentionMode::kExact, TePrecision::kBf16, false};
  const CertConfig te8{DitPrecision::kBf16, AttentionMode::kExact, TePrecision::kInt8ConvRot, false};
  const CertConfig q8{DitPrecision::kBf16, AttentionMode::kQ8Fp16, TePrecision::kBf16, false};
  const CertConfig all{DitPrecision::kInt8ConvRot, AttentionMode::kQ8Fp8, TePrecision::kInt8ConvRot, false};
  EXPECT_NE(engine_numerics(int8), engine_numerics(ref));
  EXPECT_NE(engine_numerics(q8), engine_numerics(ref));
  EXPECT_EQ(engine_numerics(int8), engine_numerics(te8));  // one INT8 ConvRot kernel family
  for (const char* k : {"elementwise", "fused_gemm", "te"}) EXPECT_TRUE(has_key(engine_numerics(ref), k)) << k;
  EXPECT_FALSE(has_key(engine_numerics(ref), "int8"));
  EXPECT_FALSE(has_key(engine_numerics(ref), "attention_q8"));
  EXPECT_TRUE(has_key(engine_numerics(int8), "int8"));
  EXPECT_FALSE(has_key(engine_numerics(int8), "attention_q8"));
  EXPECT_TRUE(has_key(engine_numerics(q8), "attention_q8"));
  EXPECT_TRUE(has_key(engine_numerics(all), "int8") && has_key(engine_numerics(all), "attention_q8"));
  for (const CertConfig& c : {ref, int8, q8, all}) {
    const EngineNumerics n = engine_numerics(c);
    for (std::size_t i = 1; i < n.size(); ++i) EXPECT_LT(n[i - 1].first, n[i].first);
    EXPECT_EQ(uncertifiable_reason(c), "");
  }
  // Paths without a numerics version, and runtime LoRAs, cannot carry a certificate.
  EXPECT_NE(uncertifiable_reason({DitPrecision::kW4A4, AttentionMode::kExact, TePrecision::kBf16, false}), "");
  EXPECT_NE(uncertifiable_reason({DitPrecision::kInt8ConvRot, AttentionMode::kExact, TePrecision::kW4A8, false}), "");
  EXPECT_NE(uncertifiable_reason({DitPrecision::kInt8ConvRot, AttentionMode::kExact, TePrecision::kBf16, true}), "");
}

TEST(QualityCert, CurrentGpuArch) {
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  EXPECT_EQ(current_gpu_arch(), gpu_arch_class(prop.major, prop.minor));
}

TEST(QualityCert, RoundTripText) {
  const Certificate c = sample();
  EXPECT_EQ(validate_certificate(c), "");
  Certificate ref;
  ref.model.dit = one_file("transformer", 1, 2);
  ref.numerics = engine_numerics(ref.config);
  ref.gpu_arch = "sm_120";
  ref.prompt_set = "moss-ac6-quick";
  ref.prompts = 3;
  ref.prompt_set_sha256 = std::string(64, '0');
  ref.scope = CertScope{1048576, {"ab2", "euler"}, 50, 50, false};
  ref.date = "2026-01-01T00:00:00Z";
  ref.gates = gates_for(QualityTier::kReference, true, false);
  ref.info = {{"same_seed.ocr_prompts", 2}, {"reference.ocr_prompts", 2}};
  ref.passed = false;
  EXPECT_EQ(validate_certificate(ref), "");

  const std::string text = certificates_json({c, ref});
  const std::vector<Certificate> back = parse_certificates(text);
  ASSERT_EQ(back.size(), 2u);
  expect_same(c, back[0]);
  expect_same(ref, back[1]);
  EXPECT_EQ(certificates_json(back), text);  // byte-stable
  EXPECT_TRUE(parse_certificates(certificates_json({})).empty());
  // The writer refuses an invalid certificate, or too many, rather than producing what the reader rejects.
  Certificate bad = c;
  bad.passed = false;
  EXPECT_THROW(certificates_json({bad}), CertificateError);
  EXPECT_THROW(certificates_json(std::vector<Certificate>(65, c)), CertificateError);
}

TEST(QualityCert, RoundTripFile) {
  mi::testing::TempDir dir("qcert_roundtrip");
  const fs::path file = dir.path() / "sub" / "x.safetensors.mossinfer-cert.json";
  ASSERT_TRUE(write_certificate_file(file, {sample()}));
  const auto back = read_certificate_file(file);
  ASSERT_EQ(back.size(), 1u);
  expect_same(sample(), back[0]);
#ifndef _WIN32
  // 0644 minus the umask (the test runner's umask clears no read bits we check here).
  const fs::perms p = fs::status(file).permissions();
  EXPECT_NE(p & fs::perms::owner_read, fs::perms::none);
  EXPECT_NE(p & fs::perms::owner_write, fs::perms::none);
  EXPECT_EQ(p & (fs::perms::group_write | fs::perms::others_write | fs::perms::owner_exec), fs::perms::none);
#endif
  ASSERT_TRUE(write_certificate_file(file, {sample(), sample()}));  // replaced in place
  EXPECT_EQ(read_certificate_file(file).size(), 2u);
  int entries = 0;
  for (const auto& e : fs::directory_iterator(file.parent_path())) entries += e.is_regular_file();
  EXPECT_EQ(entries, 1);  // no temporary files left
  Certificate bad = sample();
  bad.tier = QualityTier::kTurbo;
  EXPECT_FALSE(write_certificate_file(file, {bad}));
  EXPECT_EQ(read_certificate_file(file).size(), 2u);  // untouched
  EXPECT_THROW(read_certificate_file(dir.path() / "missing.json"), CertificateError);
}

// Every malformed, inconsistent or lax variant of a valid document is rejected whole, with a reason.
TEST(QualityCert, RejectsCorruptAndInconsistentDocuments) {
  const std::string good = certificates_json({sample()});
  ASSERT_NO_THROW(parse_certificates(good));
  struct Case {
    const char* what;
    std::string from, to;
  };
  const std::string te_json = "\"te\": {\"files\": [{\"name\": \"te-00001.safetensors\", \"size\": 4000000000, "
                              "\"xxh64\": \"0123456789abcdef\"}, {\"name\": \"te-00002.safetensors\", \"size\": 1, "
                              "\"xxh64\": \"0000000000000000\"}]}";
  const std::vector<Case> cases = {
      {"version", "\"version\": 2", "\"version\": 1"},
      {"format", "mossinfer-quality-certificates", "mossinfer-quality-certificate"},
      {"unknown top key", "\"version\": 2", "\"version\": 2, \"extra\": 0"},
      {"unknown cert key", "\"passed\": true", "\"passed\": true, \"extra\": 1"},
      {"duplicate key", "\"passed\": true", "\"passed\": true, \"passed\": true"},
      {"duplicate config key", "\"lora\": \"none\"", "\"lora\": \"none\", \"lora\": \"none\""},
      {"uppercase hash", "fedcba9876543210", "FEDCBA9876543210"},
      {"short hash", "\"fedcba9876543210\"", "\"fedcba987654321\""},
      {"hash as number", "\"fedcba9876543210\"", "1234"},
      {"negative size", "9876543210", "-1"},
      {"fractional size", "9876543210", "9876543210.5"},
      {"lora", "\"lora\": \"none\"", "\"lora\": \"style.safetensors\""},
      {"unknown attention", "\"q8_fp8\"", "\"q4\""},
      {"tier vs config", "\"tier\": \"high\"", "\"tier\": \"turbo\""},
      {"te identity missing", te_json, "\"te\": null"},
      {"empty files", "\"files\": [{\"name\": \"moss_image_sft_int8_convrot.safetensors\", \"size\": 9876543210, "
                      "\"xxh64\": \"fedcba9876543210\"}]",
       "\"files\": []"},
      {"path in name", "\"moss_image_sft_int8_convrot.safetensors\"", "\"../moss.safetensors\""},
      {"control char in name", "\"moss_image_sft_int8_convrot.safetensors\"", "\"moss\\u0001.safetensors\""},
      {"gpu arch", "\"sm_89\"", "\"sm89\""},
      {"date", "2026-10-21T12:34:56Z", "2026-13-21T12:34:56Z"},
      {"date format", "2026-10-21T12:34:56Z", "2026-10-21 12:34:56"},
      {"prompts", "\"prompts\": 23", "\"prompts\": 0"},
      {"digest", std::string(kCertPromptSetSha256Base), std::string(kCertPromptSetSha256Base.substr(1))},
      {"numerics negative", "\"te\": 1}", "\"te\": -1}"},
      {"numerics bad name", "\"te\": 1}", "\"T E\": 1}"},
      {"scope missing", "\"scope\": {", "\"scopes\": {"},
      {"scope pixels", "\"max_pixels\": 2102784", "\"max_pixels\": 0"},
      {"scope samplers empty", "\"samplers\": [\"euler\"]", "\"samplers\": []"},
      {"scope sampler name", "\"samplers\": [\"euler\"]", "\"samplers\": [\"Euler a\"]"},
      {"scope samplers unsorted", "\"samplers\": [\"euler\"]", "\"samplers\": [\"euler\", \"ab2\"]"},
      {"scope steps order", "\"steps\": [30, 50]", "\"steps\": [50, 30]"},
      {"scope steps shape", "\"steps\": [30, 50]", "\"steps\": [30]"},
      {"scope accelerators", "\"accelerators\": \"off\"", "\"accelerators\": \"on\""},
      {"gate op", "\"op\": \"<=\"", "\"op\": \"<\""},
      {"gate value type", "\"value\": 0.03333333333333333", "\"value\": \"0.03\""},
      {"gate value overflow", "\"value\": 0.03333333333333333", "\"value\": 1e999"},
      {"pass flag", "\"threshold\": 0.12, \"pass\": true", "\"threshold\": 0.12, \"pass\": false"},
      {"looser threshold", "\"threshold\": 0.12,", "\"threshold\": 0.2,"},
      {"looser max", "\"threshold\": 0.25,", "\"threshold\": 0.3,"},
      {"looser dino", "\"threshold\": 0.9,", "\"threshold\": 0.8,"},
      {"wrong op for gate", "\"op\": \"<=\", \"threshold\": 0.12", "\"op\": \">=\", \"threshold\": 0.12"},
      {"passed", "\"passed\": true", "\"passed\": false"},
      {"info missing", "\"info\": {", "\"infos\": {"},
      {"info count missing", "\"same_seed.ocr_prompts\": 13, ", ""},
      {"info count fractional", "\"reference.ocr_prompts\": 13", "\"reference.ocr_prompts\": 12.5"},
      {"info count negative", "\"reference.ocr_prompts\": 13", "\"reference.ocr_prompts\": -1"},
      {"info type", "\"neg\": -2.5", "\"neg\": \"x\""},
      {"tool too long", "\"tool\": \"certify", "\"tool\": \"" + std::string(300, 'x')},
  };
  for (const Case& k : cases) {
    std::string text = good;
    replace_once(text, k.from, k.to);
    try {
      parse_certificates(text);
      ADD_FAILURE() << k.what << ": accepted";
    } catch (const CertificateError& e) {
      EXPECT_NE(std::string(e.what()), "") << k.what;
    }
  }
  // A missing required gate: drop one at a time.
  for (const GateSpec& spec : required_gates(QualityTier::kHigh)) {
    Certificate c = sample();
    std::erase_if(c.gates, [&](const Gate& g) { return g.name == spec.name; });
    EXPECT_NE(validate_certificate(c).find("missing the required gate"), std::string::npos) << spec.name;
  }
  // Every tier requires the determinism gate, and a non-reproducible Reference tier fails the certificate.
  for (const QualityTier t : {QualityTier::kReference, QualityTier::kHigh, QualityTier::kTurbo}) {
    Certificate c = sample();
    c.tier = t;
    c.config = t == QualityTier::kReference ? CertConfig{}
             : t == QualityTier::kTurbo     ? CertConfig{DitPrecision::kW4A4, AttentionMode::kExact, TePrecision::kBf16, false}
                                            : c.config;
    if (c.config.te == TePrecision::kBf16) c.model.te.reset();
    c.numerics = engine_numerics(c.config);
    c.gates = gates_for(t);
    ASSERT_EQ(validate_certificate(c), "") << cert_token(t);
    Certificate missing = c;
    std::erase_if(missing.gates, [](const Gate& g) { return g.name == "determinism.mismatches"; });
    EXPECT_EQ(validate_certificate(missing), "missing the required gate 'determinism.mismatches'") << cert_token(t);
    std::string text = certificates_json({c});
    replace_once(text, "{\"name\": \"determinism.mismatches\", \"value\": 0, \"op\": \"<=\", \"threshold\": 0, \"pass\": true},", "");
    EXPECT_THROW(parse_certificates(text), CertificateError) << cert_token(t);
    Certificate flaky = c;
    flaky.gates.front().value = 1;
    flaky.gates.front().pass = false;
    flaky.passed = false;
    ASSERT_EQ(validate_certificate(flaky), "");
    EXPECT_NE(certificate_mismatch(flaky, identity_of(flaky), query_of(flaky)), "") << "one non-reproducible prompt";
  }
  // The official OCR gate may only be dropped when its comparison has too few text prompts.
  {
    Certificate c = sample();
    std::erase_if(c.gates, [](const Gate& g) { return g.name == "reference.ocr_delta"; });
    EXPECT_NE(validate_certificate(c), "");
    set_info(c, "reference.ocr_prompts", kMinOcrPrompts - 1);
    EXPECT_EQ(validate_certificate(c), "");
    Certificate t;  // turbo then needs same_seed.ocr_delta
    t = sample();
    t.config.dit = DitPrecision::kW4A4;
    t.tier = QualityTier::kTurbo;
    t.gates = gates_for(QualityTier::kTurbo, false, false);
    set_info(t, "reference.ocr_prompts", 2);
    EXPECT_EQ(validate_certificate(t), "");
    std::erase_if(t.gates, [](const Gate& g) { return g.name == "same_seed.ocr_delta"; });
    EXPECT_NE(validate_certificate(t), "");
  }
  // Truncated at every 37th length, garbage, wrong shapes, nesting bombs, oversize.
  for (std::size_t n = 0; n < good.size() - 2; n += 37) EXPECT_THROW(parse_certificates(good.substr(0, n)), CertificateError) << n;
  for (const std::string& bad : {std::string(""), std::string("\xff\xfe{"), std::string("[]"), std::string("null"),
                                 std::string("{\"format\": \"mossinfer-quality-certificates\", \"version\": 2}"),
                                 std::string(300, '[') + std::string(300, ']')}) {
    EXPECT_THROW(parse_certificates(bad), CertificateError);
  }
  std::string big = "{\"format\": \"mossinfer-quality-certificates\", \"version\": 2, \"certificates\": [], \"pad\": \"";
  big += std::string(kMaxCertificateFileBytes, 'x') + "\"}";
  EXPECT_THROW(parse_certificates(big), CertificateError);
  mi::testing::TempDir dir("qcert_rejects");
  write_bytes(dir.path() / "big.json", big);
  EXPECT_THROW(read_certificate_file(dir.path() / "big.json"), CertificateError);
  // Exactly at the limit is readable size-wise (then rejected as a document, not for its size).
  write_bytes(dir.path() / "edge.json", std::string(kMaxCertificateFileBytes, ' '));
  try {
    read_certificate_file(dir.path() / "edge.json");
    ADD_FAILURE() << "blank document accepted";
  } catch (const CertificateError& e) {
    EXPECT_EQ(std::string(e.what()).find("too large"), std::string::npos) << e.what();
  }
  // More certificates than the cap.
  std::string many = "{\"format\": \"mossinfer-quality-certificates\", \"version\": 2, \"certificates\": [";
  const std::string one = good.substr(good.find('[') + 1, good.rfind(']') - good.find('[') - 1);
  for (int i = 0; i < 65; ++i) many += (i == 0 ? "" : ",") + one;
  EXPECT_THROW(parse_certificates(many + "]}"), CertificateError);
}

// Random byte mutations of a valid document: each is either rejected with CertificateError or still a valid document
// (e.g. a changed digit inside a name); never another exception, a crash or an invalid certificate.
TEST(QualityCert, MutationFuzz) {
  const std::string good = certificates_json({sample()});
  std::uint64_t state = 0x9E3779B97F4A7C15ull;
  const auto next = [&] {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state >> 33;
  };
  int rejected = 0, accepted = 0;
  for (int i = 0; i < 3000; ++i) {
    std::string text = good;
    for (int m = 0, n = 1 + static_cast<int>(next() % 3); m < n; ++m) {
      const std::size_t at = next() % text.size();
      switch (next() % 3) {
        case 0: text[at] = static_cast<char>(next() & 0xFF); break;
        case 1: text.erase(at, 1 + next() % 8); break;
        default: text.insert(at, 1, "{}[]\",:0e-.\\"[next() % 12]); break;
      }
      if (text.empty()) text = "x";
    }
    try {
      for (const Certificate& c : parse_certificates(text)) EXPECT_EQ(validate_certificate(c), "");
      ++accepted;
    } catch (const CertificateError&) {
      ++rejected;
    }
  }
  RecordProperty("rejected", rejected);
  RecordProperty("accepted", accepted);
  EXPECT_GT(rejected, 2000);
}

TEST(QualityCert, MatchesOnlyTheCertifiedSetup) {
  const Certificate c = sample();
  const ModelIdentity id = identity_of(c);
  const CertificateQuery q = query_of(c);
  EXPECT_EQ(certificate_mismatch(c, id, q), "");

  const auto expect_mismatch = [&](const Certificate& cert, const ModelIdentity& m, const CertificateQuery& qq, const char* what) {
    EXPECT_NE(certificate_mismatch(cert, m, qq), "") << what;
  };
  CertificateQuery other = q;
  other.config.attention = AttentionMode::kExact;
  expect_mismatch(c, id, other, "attention");
  other = q;
  other.config.lora = true;
  expect_mismatch(c, id, other, "runtime LoRA");
  other = q;
  other.gpu_arch = "sm_86";
  expect_mismatch(c, id, other, "GPU class");
  other = q;
  other.numerics.back().second += 1;
  expect_mismatch(c, id, other, "numerics value");
  other = q;
  other.numerics.emplace_back("zz_new_kernel", 1);
  expect_mismatch(c, id, other, "numerics key");

  ModelIdentity m = id;
  m.dit.files[0].xxh64 ^= 1;
  expect_mismatch(c, m, q, "DiT content");
  m = id;
  m.dit.files[0].size += 1;
  expect_mismatch(c, m, q, "DiT size");
  m = id;
  m.dit.files[0].name = "renamed.safetensors";
  EXPECT_EQ(certificate_mismatch(c, m, q), "") << "renaming keeps the identity";
  m = id;
  m.te.reset();
  expect_mismatch(c, m, q, "TE absent");
  m = id;
  m.te->files.pop_back();
  expect_mismatch(c, m, q, "TE files");
  m = id;
  m.te->files[0].xxh64 ^= 2;
  expect_mismatch(c, m, q, "TE content");

  Certificate failed = c;
  failed.gates = gates_for(QualityTier::kHigh, true);
  failed.passed = false;
  ASSERT_EQ(validate_certificate(failed), "");
  expect_mismatch(failed, id, q, "failed certification");
  Certificate set = c;
  set.prompt_set = "moss-ac6-quick";
  expect_mismatch(set, id, q, "quick prompt set");
  set = c;
  set.prompts = 3;
  expect_mismatch(set, id, q, "too few prompts");
  set = c;
  set.prompt_set_sha256 = std::string(64, 'b');
  expect_mismatch(set, id, q, "other prompt-set digest");
  set = c;
  set.prompt_set_sha256 = std::string(kCertPromptSetSha256Sft);
  EXPECT_EQ(certificate_mismatch(set, id, q), "") << "the SFT set";
  // OCR gates need enough text prompts to be usable.
  Certificate ocr = c;
  set_info(ocr, "same_seed.ocr_prompts", kMinOcrPrompts - 1);
  ASSERT_EQ(validate_certificate(ocr), "");
  expect_mismatch(ocr, id, q, "too few same-seed OCR prompts");

  // Scope: inside passes, every dimension outside falls back.
  CertificateQuery scoped = q;
  scoped.scope = GenerationScope{1024 * 1024, "euler", 50, false};
  EXPECT_EQ(certificate_mismatch(c, id, scoped), "");
  scoped.scope = GenerationScope{2102784, "euler", 30, false};
  EXPECT_EQ(certificate_mismatch(c, id, scoped), "") << "the scope's edges are inside";
  for (const GenerationScope& g : {GenerationScope{2048 * 2048, "euler", 50, false}, GenerationScope{1024 * 1024, "ab2", 50, false},
                                   GenerationScope{1024 * 1024, "euler", 8, false}, GenerationScope{1024 * 1024, "euler", 51, false},
                                   GenerationScope{1024 * 1024, "euler", 50, true}}) {
    scoped.scope = g;
    EXPECT_NE(certificate_mismatch(c, id, scoped), "") << g.pixels << " " << g.sampler << " " << g.steps;
    EXPECT_NE(scope_mismatch(c.scope, g), "");
  }
}

TEST(QualityCert, FindAmongFiles) {
  mi::testing::TempDir dir("qcert_find");
  const Certificate c = sample();
  Certificate other_gpu = c;
  other_gpu.gpu_arch = "sm_86";
  const fs::path corrupt = dir.path() / "a_corrupt.json";
  write_bytes(corrupt, "{\"format\": \"mossinfer-quality-certificates\", \"vers");
  const fs::path good = dir.path() / "b_good.json";
  ASSERT_TRUE(write_certificate_file(good, {other_gpu, c}));

  CertificateSearch r = find_certificate(identity_of(c), query_of(c), {dir.path() / "missing.json", corrupt, good});
  ASSERT_TRUE(r.certificate.has_value()) << r.reason;
  EXPECT_EQ(r.file, good);
  EXPECT_EQ(r.certificate->gpu_arch, "sm_89");

  CertificateQuery q = query_of(c);
  q.gpu_arch = "sm_120";
  r = find_certificate(identity_of(c), q, {corrupt, good});
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_NE(r.reason.find("a_corrupt.json: rejected"), std::string::npos) << r.reason;
  EXPECT_NE(r.reason.find("sm_120"), std::string::npos) << r.reason;
  r = find_certificate(identity_of(c), q, {});
  EXPECT_EQ(r.reason, "no certificate");
}

// The sidecar first, then the first 256 cache files by name.
TEST(QualityCert, CertificateFilesOrderAndCap) {
  mi::testing::TempDir dir("qcert_files");
  const fs::path dit = dir.path() / "dit.safetensors";
  write_bytes(dit, "x");
  const fs::path certs = dir.path() / "cache" / "quality_certs";
  fs::create_directories(certs);
  for (int i = 259; i >= 0; --i) {
    char name[32];
    std::snprintf(name, sizeof(name), "c%03d.json", i);
    write_bytes(certs / name, "{}");
  }
  write_bytes(certs / "notes.txt", "");
  fs::create_directories(certs / "dir.json");  // not a regular file
  std::vector<fs::path> files = certificate_files(dit, dir.path() / "cache");
  ASSERT_EQ(files.size(), 256u);
  EXPECT_EQ(files.front().filename(), "c000.json");
  EXPECT_EQ(files.back().filename(), "c255.json");
  EXPECT_TRUE(std::is_sorted(files.begin(), files.end()));
  write_bytes(certificate_sidecar(dit), "{}");
  files = certificate_files(dit, dir.path() / "cache");
  ASSERT_EQ(files.size(), 257u);
  EXPECT_EQ(files.front(), certificate_sidecar(dit));
  EXPECT_EQ(certificate_files(dit, {}).size(), 1u);
  EXPECT_EQ(certificate_files(dit, dir.path() / "nonexistent").size(), 1u);
}

TEST(QualityCert, IdentifyAndFindFromPaths) {
  mi::testing::TempDir dir("qcert_paths");
  const std::string bytes = pattern(100'003);
  const fs::path dit = dir.path() / "dit.safetensors";
  write_bytes(dit, bytes);

  const ComponentIdentity id = identify_component(dit);
  ASSERT_EQ(id.files.size(), 1u);
  EXPECT_EQ(id.files[0].name, "dit.safetensors");
  EXPECT_EQ(id.files[0].size, bytes.size());
  EXPECT_EQ(id.files[0].xxh64, mi::xxh64(bytes.data(), bytes.size()));
  EXPECT_EQ(id.bytes(), bytes.size());
  const ComponentIdentity cheap = identify_component(dit, false);
  EXPECT_EQ(cheap.files[0].size, bytes.size());
  EXPECT_EQ(cheap.files[0].xxh64, 0u);
  EXPECT_THROW(identify_component(dir.path() / "missing.safetensors"), CertificateError);

  Certificate c = sample();
  c.config.te = TePrecision::kBf16;
  c.model.te.reset();
  c.numerics = engine_numerics(c.config);
  c.model.dit = id;
  const CertificateQuery q = query_of(c);
  const fs::path cache = dir.path() / "cache";

  CertificateSearch r = find_certificate(dit, {}, q, cache);
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_EQ(r.reason, "no certificate");

  ASSERT_TRUE(write_certificate_file(certificate_sidecar(dit), {c}));
  std::optional<ModelIdentity> full;
  r = find_certificate(dit, {}, q, cache, &full);
  ASSERT_TRUE(r.certificate.has_value()) << r.reason;
  EXPECT_EQ(r.file, certificate_sidecar(dit));
  ASSERT_TRUE(full.has_value());
  EXPECT_TRUE(full->dit.same_content(id));
  EXPECT_TRUE(fs::is_regular_file(cache / "file_hashes.json")) << "the hash went into the cache";

  // In the cache directory instead of beside the file.
  fs::remove(certificate_sidecar(dit));
  ASSERT_TRUE(write_certificate_file(cache / "quality_certs" / "x.json", {c}));
  write_bytes(cache / "quality_certs" / "notes.txt", "ignored");
  r = find_certificate(dit, {}, q, cache);
  ASSERT_TRUE(r.certificate.has_value()) << r.reason;
  EXPECT_EQ(r.file, cache / "quality_certs" / "x.json");
  EXPECT_FALSE(find_certificate(dit, {}, q, {}).certificate.has_value());  // no cache dir given

  // Same size, one byte changed (new mtime): passes the size pre-check, fails on content.
  std::string changed = bytes;
  changed[50'000] = static_cast<char>(changed[50'000] ^ 0x40);
  write_bytes(dit, changed);
  fs::last_write_time(dit, fs::last_write_time(dit) + std::chrono::seconds(5));
  r = find_certificate(dit, {}, q, cache);
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_NE(r.reason.find("other DiT weights"), std::string::npos) << r.reason;
  // A different size fails the pre-check (no hashing needed).
  write_bytes(dit, bytes + "x");
  r = find_certificate(dit, {}, q, cache);
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_NE(r.reason.find("other DiT weights"), std::string::npos) << r.reason;
  write_bytes(dit, bytes);
  EXPECT_TRUE(find_certificate(dit, {}, q, cache).certificate.has_value());
  // Uncertifiable queries are refused before any file is read.
  CertificateQuery lora = q;
  lora.config.lora = true;
  r = find_certificate(dit, {}, lora, cache);
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_NE(r.reason.find("LoRA"), std::string::npos) << r.reason;
}

// A diffusers component directory as the DiT, and a separately loaded quantized TE.
TEST(QualityCert, DirectoryDitAndLossyTe) {
  mi::testing::TempDir dir("qcert_dirte");
  const fs::path comp = dir.path() / "transformer";
  fs::create_directories(comp);
  EXPECT_THROW(identify_component(comp), CertificateError);  // nothing to identify yet
  write_bytes(comp / "b.safetensors", pattern(10, 1));
  write_bytes(comp / "a.safetensors", pattern(20, 2));
  write_bytes(comp / "config.json", "{\"num_layers\": 42}");
  write_bytes(comp / "readme.txt", "ignored");
  fs::create_directories(comp / "sub.safetensors");  // a directory: ignored
  const ComponentIdentity dir_id = identify_component(comp);
  ASSERT_EQ(dir_id.files.size(), 3u);
  EXPECT_EQ(dir_id.files[0].name, "a.safetensors");
  EXPECT_EQ(dir_id.files[1].name, "b.safetensors");
  EXPECT_EQ(dir_id.files[2].name, "config.json");  // configs belong to the identity
  EXPECT_EQ(certificate_sidecar(comp), comp / "mossinfer-cert.json");

  const fs::path te = dir.path() / "te_int8.safetensors";
  write_bytes(te, pattern(777, 3));
  Certificate c = sample();  // INT8 DiT + q8 attention + INT8 TE
  c.model.dit = dir_id;
  c.model.te = identify_component(te);
  const CertificateQuery q = query_of(c);
  ASSERT_TRUE(write_certificate_file(certificate_sidecar(comp), {c}));
  // The sidecar itself is not part of the directory's identity.
  EXPECT_TRUE(identify_component(comp).same_content(dir_id));
  CertificateSearch r = find_certificate(comp, te, q, {});
  ASSERT_TRUE(r.certificate.has_value()) << r.reason;
  EXPECT_FALSE(find_certificate(comp, {}, q, {}).certificate.has_value()) << "TE missing";
  write_bytes(dir.path() / "other_te.safetensors", pattern(777, 4));
  r = find_certificate(comp, dir.path() / "other_te.safetensors", q, {});
  EXPECT_FALSE(r.certificate.has_value());
  EXPECT_NE(r.reason.find("text-encoder"), std::string::npos) << r.reason;
  // A changed config changes the directory's identity.
  write_bytes(comp / "config.json", "{\"num_layers\": 41}");
  EXPECT_FALSE(find_certificate(comp, te, q, {}).certificate.has_value());
}

// The stat-keyed hash cache: hit on an unchanged key, re-hash when the key changes, survive corruption.
TEST(QualityCert, HashCache) {
  mi::testing::TempDir dir("qcert_hashcache");
  const fs::path f = dir.path() / "w.safetensors";
  const fs::path cache = dir.path() / "cache";
  const std::string a = pattern(5000, 1), b = pattern(5000, 2);
  write_bytes(f, a);
  const auto ha = mi::xxh64(a.data(), a.size()), hb = mi::xxh64(b.data(), b.size());
  ASSERT_NE(ha, hb);
  EXPECT_EQ(identify_component(f, true, cache).files[0].xxh64, ha);
  ASSERT_TRUE(fs::is_regular_file(cache / "file_hashes.json"));
  const std::string cache_text = read_text(cache / "file_hashes.json");
  EXPECT_NE(cache_text.find(mi::util::path_utf8(fs::canonical(f))), std::string::npos) << cache_text;

  // Same size, content changed, modification time restored: the key is unchanged, so the cached hash is served
  // (proof the cache is used; the documented limit of a metadata-keyed cache).
  const auto mtime = fs::last_write_time(f);
  write_bytes(f, b);
  fs::last_write_time(f, mtime);
  EXPECT_EQ(identify_component(f, true, cache).files[0].xxh64, ha);
  EXPECT_EQ(identify_component(f, true).files[0].xxh64, hb) << "without the cache";
  // A new modification time changes the key: hashed again, and the entry replaced.
  fs::last_write_time(f, mtime + std::chrono::seconds(3));
  EXPECT_EQ(identify_component(f, true, cache).files[0].xxh64, hb);
  EXPECT_EQ(identify_component(f, true, cache).files[0].xxh64, hb);
  // A different size changes the key too.
  write_bytes(f, a + "z");
  EXPECT_EQ(identify_component(f, true, cache).files[0].xxh64, mi::xxh64((a + "z").data(), a.size() + 1));
  // Another path with the same content is its own entry.
  const fs::path g = dir.path() / "copy.safetensors";
  write_bytes(g, a);
  EXPECT_EQ(identify_component(g, true, cache).files[0].xxh64, ha);
  EXPECT_EQ(identify_component(f, true, cache).files[0].xxh64, mi::xxh64((a + "z").data(), a.size() + 1));
  // Corrupt or foreign cache files are ignored and replaced.
  for (const std::string& junk : {std::string("{\"format\": \"mossinfer-file-hashes\", \"version\": 1, \"entries\": [{}]}"),
                                  std::string("not json"), std::string("{\"format\": \"x\", \"version\": 1, \"entries\": []}")}) {
    write_bytes(cache / "file_hashes.json", junk);
    EXPECT_EQ(identify_component(g, true, cache).files[0].xxh64, ha);
    EXPECT_NE(read_text(cache / "file_hashes.json"), junk);
  }
}

// The certification harness's writer, identity (python-xxhash) and tables against this implementation.
TEST(QualityCert, HarnessFixture) {
  const fs::path weights = kData / "tiny.safetensors";
  const std::vector<Certificate> certs = read_certificate_file(certificate_sidecar(weights));
  ASSERT_EQ(certs.size(), 1u);
  const Certificate& c = certs[0];
  EXPECT_EQ(c.tier, QualityTier::kHigh);
  EXPECT_EQ(c.config, (CertConfig{DitPrecision::kInt8ConvRot, AttentionMode::kQ8Fp8, TePrecision::kBf16, false}));
  EXPECT_EQ(c.gpu_arch, "sm_89");
  EXPECT_EQ(c.prompt_set, kCertPromptSetId);
  EXPECT_EQ(c.prompts, kCertPrompts);
  EXPECT_EQ(c.gates.size(), required_gates(QualityTier::kHigh).size());
  EXPECT_EQ(c.gates[0].name, "determinism.mismatches");
  EXPECT_EQ(c.gates[0].value, 0.0);
  EXPECT_EQ(c.gates[1].value, 0.0412);
  EXPECT_EQ(c.scope, test_scope());
  EXPECT_TRUE(c.passed);
  EXPECT_FALSE(c.model.te.has_value());
  const ComponentIdentity id = identify_component(weights);
  EXPECT_TRUE(id.same_content(c.model.dit)) << "python xxhash vs mi::xxh64";
  EXPECT_EQ(id.files[0].name, c.model.dit.files[0].name);
  EXPECT_EQ(c.numerics, engine_numerics(c.config)) << "harness header parse vs engine_numerics";

  const CertificateSearch r = find_certificate(weights, {}, CertificateQuery{c.config, "sm_89", c.numerics, std::nullopt}, {});
  ASSERT_TRUE(r.certificate.has_value()) << r.reason;
  expect_same(c, parse_certificates(certificates_json(certs))[0]);

  // tables.json: the harness's gate limits, numerics keys and prompt-set constants.
  const mi::json::Value t = mi::json::parse(read_text(kData / "tables.json"));
  for (const auto& [tier_name, list] : t.at("gates").as_object()) {
    const std::size_t dot = tier_name.find('.');
    const auto tier = parse_quality_tier(tier_name.substr(0, dot));
    ASSERT_TRUE(tier.has_value()) << tier_name;
    const std::vector<GateSpec> mine = required_gates(*tier, dot == std::string::npos);
    ASSERT_EQ(mine.size(), list.as_array().size()) << tier_name;
    for (std::size_t i = 0; i < mine.size(); ++i) {
      const mi::json::Array& g = list.as_array()[i].as_array();
      EXPECT_EQ(g[0].as_string(), mine[i].name) << tier_name;
      EXPECT_EQ(g[1].as_string(), mine[i].op == GateOp::kLe ? "<=" : ">=") << tier_name << " " << mine[i].name;
      EXPECT_EQ(g[2].as_double(), mine[i].limit) << tier_name << " " << mine[i].name;  // bit-exact
    }
  }
  for (const mi::json::Value& e : t.at("numerics").as_array()) {
    const mi::json::Value& cfg = e.at("config");
    const CertConfig conf{*parse_dit_precision(cfg.at("dit_precision").as_string()),
                          *parse_attention_mode(cfg.at("attention").as_string()),
                          *parse_te_precision(cfg.at("te_precision").as_string()), false};
    EngineNumerics theirs;
    for (const auto& [k, v] : e.at("numerics").as_object()) theirs.emplace_back(k, static_cast<std::uint64_t>(v.as_int()));
    EXPECT_EQ(theirs, engine_numerics(conf)) << cfg.at("dit_precision").as_string() << " " << cfg.at("attention").as_string();
  }
  const mi::json::Value& ps = t.at("prompt_set");
  EXPECT_EQ(ps.at("id").as_string(), kCertPromptSetId);
  EXPECT_EQ(ps.at("prompts").as_int(), kCertPrompts);
  EXPECT_EQ(ps.at("sha256").at("base").as_string(), kCertPromptSetSha256Base);
  EXPECT_EQ(ps.at("sha256").at("sft").as_string(), kCertPromptSetSha256Sft);
  EXPECT_EQ(ps.at("min_ocr_prompts").as_int(), kMinOcrPrompts);
}

// The report object the harness reads back (certify.py check_quality).
TEST(QualityCert, ReportObject) {
  QualityReport r;
  r.config = {DitPrecision::kInt8ConvRot, AttentionMode::kQ8Fp16, TePrecision::kBf16, false};
  r.gpu_arch = "sm_89";
  r.reason = "certified \"x\"";
  r.certificate = "/m/x.mossinfer-cert.json";
  r.dit = "/m/x.safetensors";
  r.dit_bytes = 9'876'543'210ull;
  r.dit_identity = one_file("x.safetensors", 9'876'543'210ull, 0xfedcba9876543210ull);
  const mi::json::Value v = mi::json::parse(quality_report_json(r));
  EXPECT_EQ(v.at("tier").as_string(), "high");
  EXPECT_EQ(v.at("config").at("dit_precision").as_string(), "int8_convrot");
  EXPECT_EQ(v.at("config").at("attention").as_string(), "q8_fp16");
  EXPECT_EQ(v.at("config").at("te_precision").as_string(), "bf16");
  EXPECT_EQ(v.at("config").at("lora").as_string(), "none");
  EXPECT_EQ(v.at("gpu_arch").as_string(), "sm_89");
  EXPECT_EQ(v.at("reason").as_string(), "certified \"x\"");
  EXPECT_EQ(v.at("certificate").as_string(), "/m/x.mossinfer-cert.json");
  EXPECT_EQ(v.at("dit").at("path").as_string(), "/m/x.safetensors");
  EXPECT_EQ(v.at("dit").at("bytes").as_int(), 9'876'543'210ll);
  EXPECT_EQ(v.at("dit").at("identity").at("files").as_array()[0].at("xxh64").as_string(), "fedcba9876543210");
  EXPECT_TRUE(v.at("te").is_null());
  const EngineNumerics n = engine_numerics(r.config);
  ASSERT_EQ(v.at("numerics").as_object().size(), n.size());
  for (const auto& [k, x] : n) EXPECT_EQ(static_cast<std::uint64_t>(v.at("numerics").at(k).as_int()), x) << k;

  QualityReport ref;
  ref.gpu_arch = "sm_86";
  ref.dit = "/bundle/transformer";
  ref.dit_bytes = 5;
  const mi::json::Value rv = mi::json::parse(quality_report_json(ref));
  EXPECT_EQ(rv.at("tier").as_string(), "reference");
  EXPECT_TRUE(rv.at("certificate").is_null());
  EXPECT_TRUE(rv.at("dit").at("identity").is_null());
  const mi::json::Value old = mi::json::parse(quality_report_json(r.config, "sm_89", "requested", {}));
  EXPECT_EQ(old.at("tier").as_string(), "high");
  EXPECT_TRUE(old.at("dit").is_null());
  EXPECT_TRUE(old.at("certificate").is_null());
  QualityReport lora = r;
  lora.config.lora = true;
  lora.te = "/m/te.safetensors";
  lora.te_bytes = 7;
  const mi::json::Value lv = mi::json::parse(quality_report_json(lora));
  EXPECT_EQ(lv.at("config").at("lora").as_string(), "active");
  EXPECT_EQ(lv.at("te").at("bytes").as_int(), 7);
}

// A comma-decimal locale for this process: an installed one, else one compiled with localedef into a temporary
// LOCPATH. Empty when neither is possible (then the C++ facet below still exercises the iostream side).
std::string set_comma_locale(const fs::path& tmp) {
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8", "German_Germany.1252"}) {
    if (std::setlocale(LC_ALL, name) != nullptr) return name;
  }
#ifndef _WIN32
  const fs::path dir = tmp / "locale";
  fs::create_directories(dir);
  const std::string cmd = "localedef -c -i de_DE -f UTF-8 '" + (dir / "de_DE.UTF-8").string() + "' >/dev/null 2>&1";
  static_cast<void>(std::system(cmd.c_str()));  // warnings make the status nonzero; setlocale decides
  setenv("LOCPATH", dir.c_str(), 1);
  if (std::setlocale(LC_ALL, "de_DE.UTF-8") != nullptr) return "de_DE.UTF-8 (localedef)";
  unsetenv("LOCPATH");
#endif
  return {};
}

struct CommaNumpunct : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

TEST(QualityCert, LocaleIndependent) {
  mi::testing::TempDir dir("qcert_locale");
  const std::string c_text = certificates_json({sample()});
  const std::string used = set_comma_locale(dir.path());
  const std::locale saved = std::locale::global(std::locale(std::locale::classic(), new CommaNumpunct));
  std::string text;
  std::vector<Certificate> back;
  bool file_ok = false;
  try {
    text = certificates_json({sample()});
    back = parse_certificates(c_text);
    file_ok = write_certificate_file(dir.path() / "c.json", {sample()});
  } catch (...) {
    std::locale::global(saved);
    std::setlocale(LC_ALL, "C");
    throw;
  }
  std::locale::global(saved);
  std::setlocale(LC_ALL, "C");
#ifndef _WIN32
  unsetenv("LOCPATH");
#endif
  EXPECT_EQ(text, c_text);
  ASSERT_EQ(back.size(), 1u);
  expect_same(sample(), back[0]);
  EXPECT_TRUE(file_ok);
  EXPECT_EQ(read_text(dir.path() / "c.json"), c_text);
  RecordProperty("comma_locale", used.empty() ? "none (C++ facet only)" : used);
  std::printf("comma-decimal C locale for this test: %s\n", used.empty() ? "none (C++ facet only)" : used.c_str());
}

}  // namespace
