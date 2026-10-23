// Quality certificates (P6 step 39, AC21): the evidence that a lossy configuration (an INT8 or W4A4 DiT, 8-bit
// attention, a quantized text encoder) keeps image quality on a given model, engine numerics and GPU class. The
// planner's Auto tier may only pick a lossy configuration that has a usable certificate (find_certificate) for the
// request at hand; Reference needs none.
//
// A certificate is written by the offline certification harness after scoring the evaluation prompt set (same seed
// against the bf16 engine, and against the official reference images) and stored as a JSON sidecar next to the DiT
// (certificate_sidecar()) or in the engine's cache directory (<cache_dir>/quality_certs/*.json).
//
// File format (version kCertificateFormatVersion), one document holding up to 64 certificates:
//   {"format": "mossinfer-quality-certificates", "version": 2, "certificates": [
//     {"tier": "high",
//      "model": {"dit": {"files": [{"name": "x.safetensors", "size": 123, "xxh64": "0123456789abcdef"}]},
//                "te": null},                     // the TE file(s), required exactly when te_precision != bf16
//      "config": {"dit_precision": "int8_convrot", "attention": "exact", "te_precision": "bf16", "lora": "none"},
//      "numerics": {"cublaslt": 130800, "elementwise": 1, "fused_gemm": 1, "int8": 1, "te": 1},  // engine_numerics()
//      "gpu_arch": "sm_89",
//      "prompt_set": {"id": "moss-ac6", "prompts": 23, "sha256": "<64 hex>"},
//      "scope": {"max_pixels": 2102784, "samplers": ["euler"], "steps": [30, 50], "accelerators": "off"},
//      "date": "2026-10-21T12:00:00Z",
//      "gates": [{"name": "same_seed.lpips_mean", "value": 0.031, "op": "<=", "threshold": 0.12, "pass": true}, ...],
//      "passed": true,
//      "info": {"same_seed.ocr_prompts": 13, "reference.ocr_prompts": 13, ...},  // required keys + informational
//      "tool": "certify.py 2"}                                                   // optional, informational
//   ]}
// Hashes are 16 lowercase hex digits (JSON numbers cannot hold 64 bits); sizes are integers; other numbers are
// written in shortest round-trip form with std::to_chars, so reading and writing never depend on the process locale.
//
// Files are untrusted input: the reader caps the file size and the DOM, rejects duplicate or unknown keys, checks
// every string against its grammar and every number for range, and validates the semantics (validate_certificate):
// the tier must be the one the configuration belongs to, every gate the tier requires (required_gates) must be present
// with a threshold at least as strict as the tier's limit, every stored pass flag must agree with its value and
// threshold, and "passed" must be the conjunction of the gates. Any violation rejects the whole document.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mi::pipeline {

inline constexpr int kCertificateFormatVersion = 2;
inline constexpr std::string_view kCertificateFormat = "mossinfer-quality-certificates";
// The evaluation set Auto accepts: the AC6 quality test set (20 official assets with their own parameters, 5 SFT
// prompts, 3 golden prompts; 28 jobs) run on the one model the certified DiT belongs to. The 5 SFT jobs reuse asset
// prompts and parameters, so on one model that is 23 distinct prompts, all compared same-seed with the bf16 engine;
// the official-image comparison covers the jobs whose official image comes from that model (base 23, SFT 5).
// Certificates over any other set (a --quick development run) are valid documents but never usable.
inline constexpr std::string_view kCertPromptSetId = "moss-ac6";
inline constexpr int kCertPrompts = 23;
// SHA-256 of the harness's canonical job list ({"model", "jobs": [{"id", "params"}]}, sorted keys, compact) for the
// two models: a certificate is usable only over exactly one of these sets.
inline constexpr std::string_view kCertPromptSetSha256Base = "c0b4b6069d008a51ddac0d91b1399de5e533203f3791ad7f464439905675b876";
inline constexpr std::string_view kCertPromptSetSha256Sft = "b7f3afd744c9fed3fcf55b2d5d3adbea78f62f961a753f7dd2177938ea03760b";
// An OCR gate needs at least this many text-rendering prompts in its comparison to be usable; with fewer official
// text prompts (the SFT set has 2) OCR is gated same-seed against the bf16 engine only.
inline constexpr int kMinOcrPrompts = 3;
inline constexpr std::size_t kMaxCertificateFileBytes = std::size_t{1} << 20;

struct CertificateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class QualityTier { kReference, kHigh, kTurbo };
enum class DitPrecision { kBf16, kInt8ConvRot, kW4A4 };
// The exact attention, or one of the 8-bit attention variants (attention_q8::Variant).
enum class AttentionMode { kExact, kQ8Fp16, kQ8Fp16F32Acc, kQ8Fp8, kQ8Fp8F32Acc };
enum class TePrecision { kBf16, kInt8ConvRot, kW4A8 };

// Tokens as written in certificates (and accepted by the CLI): "reference" | "high" | "turbo"; "bf16" |
// "int8_convrot" | "w4a4"; "exact" | "q8_fp16" | "q8_fp16_f32acc" | "q8_fp8" | "q8_fp8_f32acc"; "bf16" |
// "int8_convrot" | "w4a8". The parsers return nullopt for anything else.
std::string_view cert_token(QualityTier t);
std::string_view cert_token(DitPrecision p);
std::string_view cert_token(AttentionMode a);
std::string_view cert_token(TePrecision p);
std::optional<QualityTier> parse_quality_tier(std::string_view s);
std::optional<DitPrecision> parse_dit_precision(std::string_view s);
std::optional<AttentionMode> parse_attention_mode(std::string_view s);
std::optional<TePrecision> parse_te_precision(std::string_view s);

// What changes the output bits of a generation beyond the weights: the certified dimensions.
struct CertConfig {
  DitPrecision dit = DitPrecision::kBf16;
  AttentionMode attention = AttentionMode::kExact;
  TePrecision te = TePrecision::kBf16;
  bool lora = false;  // runtime LoRAs active: certificates cover the base model only, so never certified

  bool operator==(const CertConfig&) const = default;
};

// The tier a configuration belongs to: Reference when everything is bf16 and exact, Turbo with a W4A4 DiT, High
// otherwise.
QualityTier tier_of(const CertConfig& c);

// Engine numerics versions a configuration's output bits depend on, sorted by name. A certificate is usable only when
// its stored versions equal these exactly, so bumping any of them (a kernel or op-order change) retires the
// certificates measured before it. Keys:
//   "cublaslt" cublasLtGetVersion() of the cuBLASLt copy in use (backend/cuda_libs.h; 0 when none is usable): always
//     (every configuration runs cuBLASLt GEMMs, whose result bits may change between its versions, and a system copy
//     may differ from the bundled one);
//   "elementwise" kernels::kElementwiseNumericsVersion, "fused_gemm" kernels::fused::kFusedGemmVersion and "te"
//     qwen35_text::kNumericsVersion: always (every configuration runs the bf16 elementwise ops, the bf16 GEMM routing
//     of the layers kept in bf16, and the text encoder's own op order);
//   "int8" kernels::quant::kInt8NumericsVersion: when the DiT or the TE is INT8 ConvRot or the TE is W4A8 (its
//     activations are INT8);
//   "w4a4" kernels::quant4::kInt4NumericsVersion: when the DiT is W4A4;
//   "te_w4a8" kernels::quant::kGroupedDecodeVersion: when the TE is W4A8;
//   "attention_q8" attention_q8::kNumericsVersion: when the attention is not exact.
using EngineNumerics = std::vector<std::pair<std::string, std::uint64_t>>;
EngineNumerics engine_numerics(const CertConfig& c);  // with the version of the cuBLASLt copy this process uses
EngineNumerics engine_numerics(const CertConfig& c, std::uint64_t cublaslt_version);  // pure
// Empty when the configuration can carry a certificate, else why not (runtime LoRA).
std::string uncertifiable_reason(const CertConfig& c);

// "sm_<major><minor>": the GPU class certificates are measured on (one kernel image and numerics per class).
std::string gpu_arch_class(int major, int minor);
// gpu_arch_class() of the current CUDA device.
std::string current_gpu_arch();

// Content identity of a model component: its files, each by size and XXH64 (seed 0) of the whole file. The name is
// kept for messages only; renaming a file keeps its identity.
struct FileIdentity {
  std::string name;  // file name without directories, UTF-8
  std::uint64_t size = 0;
  std::uint64_t xxh64 = 0;
};
struct ComponentIdentity {
  std::vector<FileIdentity> files;  // one file, or a directory's identity files in name order
  // Same sizes and hashes, in order (names ignored).
  bool same_content(const ComponentIdentity& o) const;
  std::uint64_t bytes() const;
};
struct ModelIdentity {
  ComponentIdentity dit;
  std::optional<ComponentIdentity> te;  // the separately loaded TE file(s), when the TE is not bf16
};

// The identity of a single file, or of a directory (a diffusers component): every regular *.safetensors and *.json
// file directly inside it (weights and configs), except certificate sidecars, sorted by UTF-8 name. Reads every byte
// when `hash` (several seconds for a multi-GB DiT on a cold page cache) unless `hash_cache_dir` holds the file's hash
// under an unchanged key; without `hash` only names and sizes are filled (xxh64 = 0), for a cheap pre-check. Throws
// CertificateError when the path is missing or unreadable, a directory holds no such file, or a file changes while it
// is hashed.
//
// Hash cache: <hash_cache_dir>/file_hashes.json maps (canonical path, device and inode (Windows: volume serial and
// file index), size, modification time in ns) to the XXH64, for up to 512 files (oldest dropped). A file whose key
// changes is hashed again. A file rewritten in place with the same size and its modification time restored keeps its
// stale hash: the cache trusts the file system's metadata. An unreadable or invalid cache file is ignored and replaced.
ComponentIdentity identify_component(const std::filesystem::path& path, bool hash = true,
                                     const std::filesystem::path& hash_cache_dir = {});

enum class GateOp { kLe, kGe };  // value <= threshold, value >= threshold

struct Gate {
  std::string name;  // [a-z0-9_.-], e.g. "same_seed.lpips_mean"
  double value = 0;
  GateOp op = GateOp::kLe;
  double threshold = 0;
  bool pass = false;
};

struct GateSpec {
  std::string_view name;
  GateOp op;
  double limit;  // the loosest threshold the tier allows
};

// The gates a tier requires. Two comparisons, both over the whole prompt set:
//   same_seed: the configuration against the bf16 engine, same seeds;
//   reference: the configuration against the official (torch) reference images.
// Metrics: lpips_mean / dino_mean (means over the images), lpips_max (worst image), lpips_paired_max
// (max over images of LPIPS(test, official) - LPIPS(bf16 engine, official): how much farther from the official image
// than the bf16 engine's own image any one image moved), ocr_delta (mean over the text-rendering prompts of the OCR
// exact-match rate of the prompt's target strings, minus the baseline's rate).
//   every tier: determinism.mismatches <= 0 (the harness renders every Reference-tier prompt twice and counts the
//     prompts whose final latents or PNGs differ: a machine where the Reference tier is not bit-reproducible certifies
//     nothing);
//   reference tier (k = 1), high (k = 1.5):
//     same_seed.lpips_mean <= 0.12, same_seed.lpips_max <= 0.25, same_seed.dino_mean >= 0.90,
//     same_seed.ocr_delta >= -0.05;
//     reference.lpips_mean <= 0.088 k, reference.dino_mean >= 1 - 0.075 k (AC6's per-image limits scaled as
//     distances), reference.lpips_paired_max <= 0.088 k / 1.5 (k x P0's torch kernel-swap LPIPS floor 0.0587),
//     reference.ocr_delta >= -0.05 (reference tier) / -0.075 (high);
//   turbo (k = 3): reference.lpips_mean <= 0.264, reference.dino_mean >= 0.775, reference.lpips_paired_max <= 0.176,
//     reference.ocr_delta >= -0.10.
// When the official comparison has fewer than kMinOcrPrompts text prompts (`reference_ocr_gateable` false),
// reference.ocr_delta is not required; OCR is then gated same seed only (turbo: same_seed.ocr_delta >= -0.10).
// Extra gates may be stored; they must pass too.
std::vector<GateSpec> required_gates(QualityTier t, bool reference_ocr_gateable = true);

// What a certificate's measurements covered. A request outside it falls back to Reference (scope_mismatch).
struct CertScope {
  std::int64_t max_pixels = 0;        // largest width * height measured
  std::vector<std::string> samplers;  // sampler names measured, sorted, e.g. {"euler"}
  int min_steps = 0, max_steps = 0;   // step counts measured
  // Always false in a valid certificate ("accelerators": "off"): measured with every lossy accelerator outside the
  // certified configuration (step caching and the like) disabled.
  bool accelerators = false;
  bool operator==(const CertScope&) const = default;
};

// One request, as the planner sees it.
struct GenerationScope {
  std::int64_t pixels = 0;  // width * height
  std::string sampler;      // the sampler name, in the engine's own vocabulary
  int steps = 0;
  bool accelerators = false;  // any lossy accelerator outside the configuration (step caching, ...) enabled
};

// Empty when the request lies inside the scope, else why not.
std::string scope_mismatch(const CertScope& scope, const GenerationScope& g);

struct Certificate {
  QualityTier tier = QualityTier::kReference;
  ModelIdentity model;
  CertConfig config;
  EngineNumerics numerics;
  std::string gpu_arch;
  std::string prompt_set;         // id, e.g. kCertPromptSetId
  int prompts = 0;                // prompts scored
  std::string prompt_set_sha256;  // digest of the prompt set's parameters (64 lowercase hex)
  CertScope scope;
  std::string date;               // UTC, "YYYY-MM-DDTHH:MM:SSZ"
  std::vector<Gate> gates;
  bool passed = false;
  // Required: "same_seed.ocr_prompts" and "reference.ocr_prompts" (text-rendering prompts in each comparison).
  // Informational: noise-floor baselines, worst cases, standard errors and confidence intervals, ungated metrics.
  std::vector<std::pair<std::string, double>> info;
  std::string tool;  // informational
};

// Empty when the certificate is internally consistent (see the file comment), else what is wrong.
std::string validate_certificate(const Certificate& c);

// Parses and validates a whole document. Throws CertificateError (with the reason) on any problem.
std::vector<Certificate> parse_certificates(std::string_view text);
// Throws CertificateError when a certificate is invalid or there are more than 64.
std::string certificates_json(const std::vector<Certificate>& certs);
// Throws CertificateError when the file cannot be read, holds more than kMaxCertificateFileBytes, or is invalid.
std::vector<Certificate> read_certificate_file(const std::filesystem::path& path);
// Atomic, durable replace: a unique temporary file (mode 0644 minus the umask), flushed to disk, renamed over `path`.
// False when it could not be written (or a certificate is invalid).
bool write_certificate_file(const std::filesystem::path& path, const std::vector<Certificate>& certs);

// The sidecar of a DiT file "<file>.mossinfer-cert.json", or "<dir>/mossinfer-cert.json" for a directory.
std::filesystem::path certificate_sidecar(const std::filesystem::path& dit);
// Where to look: the sidecar, then the first 256 (by name) *.json files in <cache_dir>/quality_certs (cache_dir may be
// empty). Only existing files are returned.
std::vector<std::filesystem::path> certificate_files(const std::filesystem::path& dit, const std::filesystem::path& cache_dir);

struct CertificateQuery {
  CertConfig config;
  std::string gpu_arch;     // current_gpu_arch()
  EngineNumerics numerics;  // engine_numerics(config)
  // The request; Auto always sets it. Unset: the scope is not checked (e.g. listing what a file is certified for).
  std::optional<GenerationScope> scope = std::nullopt;
};

// Empty when `c` certifies this model, configuration, GPU class, engine numerics and request (passed, over the
// certification prompt set, with enough OCR prompts behind every OCR gate), else why not. A certificate that matches in
// everything but the "cublaslt" numerics entry says so: "certificate measured with cuBLASLt <X>; this process loaded
// cuBLASLt <Y> (<source> copy <path>), so it does not apply ..." (find_certificate's reason carries it, after the file
// name, into the tier report).
std::string certificate_mismatch(const Certificate& c, const ModelIdentity& model, const CertificateQuery& q);

struct CertificateSearch {
  std::optional<Certificate> certificate;  // the first usable one
  std::filesystem::path file;              // where it was found
  std::string reason;                      // why nothing was found (each rejected candidate), for the tier report
};

// The first usable certificate among `files`, in order. Unreadable or invalid files are skipped (named in reason).
CertificateSearch find_certificate(const ModelIdentity& model, const CertificateQuery& q,
                                   const std::vector<std::filesystem::path>& files);

// Same, from paths: looks in certificate_files(dit, cache_dir), pre-checks every candidate on names-and-sizes
// identities, and hashes the files (through the hash cache in cache_dir) only when a candidate survives, so the usual
// "no certificate" costs no read. `te` is the separately loaded TE file or directory, empty when the TE is bf16 from
// the bundle. `identity`, when given, receives the full identity whenever it was computed.
CertificateSearch find_certificate(const std::filesystem::path& dit, const std::filesystem::path& te,
                                   const CertificateQuery& q, const std::filesystem::path& cache_dir,
                                   std::optional<ModelIdentity>* identity = nullptr);

// The "quality" object of a generation report (`--json-report`): what the run used and why, in the certificate's
// own vocabulary, so the certification harness can confirm what it certifies:
//   {"tier": "high", "config": {"dit_precision": ..., "attention": ..., "te_precision": ..., "lora": "none"|"active"},
//    "numerics": {...engine_numerics(config)},
//    "cublaslt": {"source": "system"|"bundled"|"env", "version": 130800, "path": "<file>"}|null (none usable),
//    "gpu_arch": "sm_89", "reason": "...", "certificate": "<file>"|null,
//    "dit": {"path": "...", "bytes": N, "identity": {"files": [...]}|null}, "te": {...}|null}
struct QualityReport {
  CertConfig config;
  std::string gpu_arch;                   // current_gpu_arch()
  std::string reason;                     // why this tier (Auto's choice, or "requested")
  std::filesystem::path certificate;      // the certificate used; empty for none
  std::filesystem::path dit;              // the DiT file or directory loaded
  std::uint64_t dit_bytes = 0;            // its total size
  std::optional<ComponentIdentity> dit_identity;  // when computed (find_certificate's `identity`)
  std::filesystem::path te;               // the separately loaded TE; empty when bf16 from the bundle
  std::uint64_t te_bytes = 0;
  std::optional<ComponentIdentity> te_identity;
};
std::string quality_report_json(const QualityReport& r);
// Without the model paths ("dit" and "te" null): the earlier signature, kept so callers can migrate.
std::string quality_report_json(const CertConfig& c, std::string_view gpu_arch, std::string_view reason,
                                const std::filesystem::path& certificate);

}  // namespace mi::pipeline
