#include "pipeline/hardware_profile.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "core/cuda_check.h"
#include "kernels/fused/fused_gemm.h"
#include "util/json.h"

namespace mi::pipeline {
namespace {

constexpr std::int64_t kVersion = 2;

// Locale-independent (std::to_chars ignores LC_NUMERIC); JSON has no inf/nan (null reads back as missing).
std::string num(double d) {
  if (!std::isfinite(d)) return "null";
  char buf[32];
  const auto res = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::general, 6);
  return std::string(buf, res.ptr);
}

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    if (static_cast<unsigned char>(c) >= 0x20) out += c;
  }
  return out + "\"";
}

std::string rates_json(const planner::HardwareProfile& hw) {
  return "{\"h2d_gbps\": " + num(hw.h2d_gbps) + ", \"host_gbps\": " + num(hw.host_gbps) + ", \"disk_gbps\": " +
         num(hw.disk_gbps) + ", \"bf16_tflops\": " + num(hw.bf16_tflops) + "}";
}

// The rates of `v`; a missing (absent or null) rate is taken from `fallback`, else the rates are unusable.
std::optional<planner::HardwareProfile> rates_from(const json::Value& v,
                                                   const std::optional<planner::HardwareProfile>& fallback = {}) {
  planner::HardwareProfile hw;
  bool missing = false;
  const auto rate = [&](const char* name, double planner::HardwareProfile::*field) {
    const json::Value* r = v.find(name);
    if (r == nullptr || r->is_null()) {
      missing = missing || !fallback;
      if (fallback) hw.*field = (*fallback).*field;
      return;
    }
    hw.*field = r->as_double();
  };
  rate("h2d_gbps", &planner::HardwareProfile::h2d_gbps);
  rate("host_gbps", &planner::HardwareProfile::host_gbps);
  rate("disk_gbps", &planner::HardwareProfile::disk_gbps);
  rate("bf16_tflops", &planner::HardwareProfile::bf16_tflops);
  const auto sane = [](double g) { return std::isfinite(g) && g > 0.01 && g < 1e4; };
  if (missing || !sane(hw.h2d_gbps) || !sane(hw.host_gbps) || !sane(hw.disk_gbps) || !sane(hw.bf16_tflops)) return std::nullopt;
  hw.measured = true;
  return hw;
}

std::string config_json(const moss_dit::FusedConfig& f) {
  const kernels::Gemm::AlgoConfig& c = f.config;
  return "{\"op\": " + json_string(f.op) + ", \"rows\": " + std::to_string(f.rows) + ", \"id\": " + std::to_string(c.id) +
         ", \"tile\": " + std::to_string(c.tile) + ", \"stages\": " + std::to_string(c.stages) +
         ", \"swizzle\": " + std::to_string(c.swizzle) + ", \"custom\": " + std::to_string(c.custom_option) +
         ", \"split_k\": " + std::to_string(c.split_k) + ", \"reduction\": " + std::to_string(c.reduction_scheme) +
         ", \"inner\": " + std::to_string(c.inner_shape) + ", \"cluster\": " + std::to_string(c.cluster_shape) + "}";
}

// The stored self-check, or nullopt when it is malformed or from other fused kernels (kFusedKernelsVersion): run it
// again.
std::optional<moss_dit::FusedCheck> fused_check_from(const json::Value& v) {
  try {
    const json::Value* version = v.find("kernels_version");
    if (version == nullptr || !version->is_int() || version->as_int() != kernels::fused::kFusedKernelsVersion) {
      return std::nullopt;
    }
    moss_dit::FusedCheck c{v.at("ok").as_bool(), v.at("detail").as_string(), {}};
    const auto field = [](const json::Value& e, const char* name, std::int64_t lo, std::int64_t hi) {
      const std::int64_t x = e.at(name).as_int();
      if (x < lo || x > hi) throw std::out_of_range(name);
      return x;
    };
    constexpr std::int64_t kU32 = 0xFFFFFFFFll, kU16 = 0xFFFF;
    for (const json::Value& e : v.at("validated").as_array()) {
      moss_dit::FusedConfig f;
      f.op = e.at("op").as_string();
      f.rows = field(e, "rows", 1, INT32_MAX);
      f.config.id = static_cast<int>(field(e, "id", INT32_MIN, INT32_MAX));
      f.config.tile = static_cast<std::uint32_t>(field(e, "tile", 0, kU32));
      f.config.stages = static_cast<std::uint32_t>(field(e, "stages", 0, kU32));
      f.config.swizzle = static_cast<std::uint32_t>(field(e, "swizzle", 0, kU32));
      f.config.custom_option = static_cast<std::uint32_t>(field(e, "custom", 0, kU32));
      f.config.split_k = static_cast<std::int32_t>(field(e, "split_k", INT32_MIN, INT32_MAX));
      f.config.reduction_scheme = static_cast<std::uint32_t>(field(e, "reduction", 0, kU32));
      f.config.inner_shape = static_cast<std::uint16_t>(field(e, "inner", 0, kU16));
      f.config.cluster_shape = static_cast<std::uint16_t>(field(e, "cluster", 0, kU16));
      c.validated.push_back(std::move(f));
    }
    if (c.ok && c.validated.empty()) return std::nullopt;  // a pass checks at least one configuration
    return c;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

int process_id() {
#ifdef _WIN32
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

}  // namespace

std::string hardware_key() {
  int dev = 0, driver = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  MI_CUDA_CHECK(cudaDriverGetVersion(&driver));
  cudaDeviceProp prop{};
  MI_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  char buf[400];
  std::snprintf(buf, sizeof(buf), "%s sm_%d%d @ %04x:%02x:%02x, driver %d, cuBLASLt %zu", prop.name, prop.major, prop.minor,
                prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, driver, static_cast<std::size_t>(cublasLtGetVersion()));
  return buf;
}

std::optional<StoredProfile> read_profile(const std::filesystem::path& path, const std::string& key, std::int64_t now) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    const json::Value v = json::parse(ss.str(), json::ParseLimits{65536});
    if (v.at("version").as_int() != kVersion || v.at("key").as_string() != key) return std::nullopt;
    StoredProfile p;
    p.probed_at = v.at("probed_at").as_int();
    if (p.probed_at > now || now - p.probed_at > kProfileMaxAgeSeconds) return std::nullopt;
    // A learned rate written as null (not finite) is not learned: the probed one stands.
    const auto probe = rates_from(v.at("probe"));
    if (!probe) return std::nullopt;
    const auto learned = rates_from(v.at("learned"), probe);
    if (!learned) return std::nullopt;
    p.probe = *probe;
    p.learned = *learned;
    if (const json::Value* fc = v.find("fused_check"); fc != nullptr) p.fused_check = fused_check_from(*fc);
    return p;
  } catch (const std::exception&) {
    return std::nullopt;  // unreadable or stale: measure again
  }
}

bool write_profile(const std::filesystem::path& path, const std::string& key, const StoredProfile& p) {
  std::string doc = "{\"version\": " + std::to_string(kVersion) + ", \"key\": " + json_string(key) +
                    ", \"probed_at\": " + std::to_string(p.probed_at) + ",\n \"probe\": " + rates_json(p.probe) +
                    ",\n \"learned\": " + rates_json(p.learned);
  if (p.fused_check) {
    doc += ",\n \"fused_check\": {\"kernels_version\": " + std::to_string(kernels::fused::kFusedKernelsVersion) +
           ", \"ok\": " + std::string(p.fused_check->ok ? "true" : "false") + ", \"detail\": " +
           json_string(p.fused_check->detail) + ",\n  \"validated\": [";
    for (std::size_t i = 0; i < p.fused_check->validated.size(); ++i) {
      doc += (i == 0 ? "\n   " : ",\n   ") + config_json(p.fused_check->validated[i]);
    }
    doc += "]}";
  }
  doc += "}\n";
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  // Unique per process and call: concurrent writers never share a temporary file; the rename replaces atomically.
  char suffix[64];
  std::snprintf(suffix, sizeof(suffix), ".tmp.%d.%08x", process_id(), static_cast<unsigned>(std::random_device{}()));
  const std::filesystem::path tmp = path.string() + suffix;
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out << doc;
    out.close();
    if (!out) {
      std::filesystem::remove(tmp, ec);
      return false;
    }
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

void ProfileLearner::Verdict::vote(bool v) {
  idle = 0;
  votes.push_back(v);
  while (votes.size() > static_cast<std::size_t>(kEvidence)) votes.pop_front();
  if (votes.size() == static_cast<std::size_t>(kEvidence) && std::all_of(votes.begin(), votes.end(), [&](bool b) { return b != value; })) {
    value = !value;
    votes.clear();
  }
}

void ProfileLearner::Verdict::end_generation() {
  if (++idle < kRecheckGenerations || value) return;
  // Cold with no fresh evidence (the plan avoided the files): try the files again; kEvidence cold votes undo it.
  value = true;
  votes.clear();
  idle = 0;
}

void ProfileLearner::Series::push(double v) {
  values.push_back(v);
  while (values.size() > static_cast<std::size_t>(kHistory)) values.pop_front();
}

double ProfileLearner::Series::median() const {
  std::vector<double> v(values.begin(), values.end());
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

bool ProfileLearner::Series::consistent() const {
  if (values.size() < static_cast<std::size_t>(kEvidence)) return false;
  Series last;
  last.values.assign(values.end() - kEvidence, values.end());
  const double m = last.median();
  return std::all_of(last.values.begin(), last.values.end(), [&](double x) { return std::abs(x - m) <= kSpread * m; });
}

ProfileLearner::ProfileLearner(StoredProfile profile) : stored_(std::move(profile)), current_(stored_.learned) {}

bool ProfileLearner::cached_rate(double gbps) const {
  return gbps >= std::sqrt(stored_.probe.host_gbps * stored_.probe.disk_gbps);
}

bool ProfileLearner::observe_tflops(double tflops) {
  if (!std::isfinite(tflops) || tflops <= 0) return false;
  tflops_.push(tflops);
  const double probe = stored_.probe.bf16_tflops;
  current_.bf16_tflops = std::clamp(tflops_.median(), probe * kTflopsMin, probe * kTflopsMax);
  if (!tflops_.consistent() || std::abs(stored_.learned.bf16_tflops - current_.bf16_tflops) < 1e-3 * probe) return false;
  stored_.learned.bf16_tflops = current_.bf16_tflops;
  return true;
}

bool ProfileLearner::observe_dit_stream(double gbps) {
  if (!std::isfinite(gbps) || gbps <= 0) return false;
  const bool cached = cached_rate(gbps);
  dit_cached_.vote(cached);
  if (!cached) return false;
  host_gbps_.push(gbps);
  const double probe = stored_.probe.host_gbps;
  current_.host_gbps = std::clamp(host_gbps_.median(), probe * kRateMin, probe * kRateMax);
  if (!host_gbps_.consistent() || std::abs(stored_.learned.host_gbps - current_.host_gbps) < 1e-3 * probe) return false;
  stored_.learned.host_gbps = current_.host_gbps;
  return true;
}

void ProfileLearner::observe_dit_load(double gbps) {
  if (std::isfinite(gbps) && gbps > 0) dit_cached_.vote(cached_rate(gbps));
}

void ProfileLearner::observe_te_stream(bool cached) { te_cached_.vote(cached); }

void ProfileLearner::end_generation() {
  dit_cached_.end_generation();
  te_cached_.end_generation();
}

}  // namespace mi::pipeline
