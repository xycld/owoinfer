// pipeline/hardware_profile: the cached machine profile (round trip, corruption, staleness, atomic replacement), the
// learning rules (one disturbed generation never flips a verdict or reaches the disk; recovery; bounds), and the
// JSON writers under a comma-decimal process locale. Host-only.
#include <gtest/gtest.h>

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "kernels/fused/fused_gemm.h"
#include "pipeline/hardware_profile.h"
#include "planner/planner.h"
#include "test_util.h"
#include "util/json.h"
#include "util/rename_replace.h"
#include "util/replaceable_file.h"

namespace {

namespace fs = std::filesystem;
using mi::pipeline::ProfileLearner;
using mi::pipeline::StoredProfile;

constexpr std::int64_t kNow = 1'800'000'000;
const std::string kKey = "NVIDIA GeForce RTX 4090 Laptop GPU sm_89 @ 0000:01:00, driver 13040, cuBLASLt 130400";

StoredProfile sample() {
  StoredProfile p;
  p.probe.h2d_gbps = 24.0;
  p.probe.host_gbps = 19.5;
  p.probe.disk_gbps = 2.3;
  p.probe.bf16_tflops = 40.0;
  p.probe.measured = true;
  p.learned = p.probe;
  p.learned.bf16_tflops = 73.7;
  p.probed_at = kNow - 3600;
  mi::moss_dit::FusedConfig f;
  f.op = "ffn_down";
  f.rows = 384;
  f.config.id = 21;
  f.config.tile = 18;
  f.config.stages = 0x16;
  f.config.swizzle = 1;
  f.config.inner_shape = 7;
  p.fused_check = mi::moss_dit::FusedCheck{true, "2 cuBLASLt configurations \"bit-identical\"", {f, f}};
  p.fused_check->validated[1].op = "o_proj";
  p.fused_check->validated[1].rows = 1;
  p.fused_check->validated[1].config.id = 6;
  return p;
}

std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_text(const fs::path& p, const std::string& s) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out << s;
}

void expect_same(const StoredProfile& a, const StoredProfile& b) {
  EXPECT_EQ(a.probe.h2d_gbps, b.probe.h2d_gbps);
  EXPECT_EQ(a.probe.host_gbps, b.probe.host_gbps);
  EXPECT_EQ(a.probe.disk_gbps, b.probe.disk_gbps);
  EXPECT_EQ(a.probe.bf16_tflops, b.probe.bf16_tflops);
  EXPECT_EQ(a.learned.bf16_tflops, b.learned.bf16_tflops);
  EXPECT_EQ(a.learned.host_gbps, b.learned.host_gbps);
  EXPECT_EQ(a.probed_at, b.probed_at);
  ASSERT_EQ(a.fused_check.has_value(), b.fused_check.has_value());
  if (a.fused_check) {
    EXPECT_EQ(a.fused_check->ok, b.fused_check->ok);
    EXPECT_EQ(a.fused_check->detail, b.fused_check->detail);
    ASSERT_EQ(a.fused_check->validated.size(), b.fused_check->validated.size());
    for (std::size_t i = 0; i < a.fused_check->validated.size(); ++i) {
      EXPECT_EQ(a.fused_check->validated[i].op, b.fused_check->validated[i].op);
      EXPECT_EQ(a.fused_check->validated[i].rows, b.fused_check->validated[i].rows);
      EXPECT_EQ(a.fused_check->validated[i].config, b.fused_check->validated[i].config) << i;
    }
  }
}

// A comma-decimal locale for this process: an installed one, else one compiled with localedef into a temporary
// LOCPATH. Empty when neither is possible (then the checks run in the C locale only).
std::string set_comma_locale(const fs::path& tmp) {
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8", "German_Germany.1252"}) {
    if (std::setlocale(LC_ALL, name) != nullptr) return name;
  }
#ifndef _WIN32
  const fs::path dir = tmp / "locale";
  fs::create_directories(dir);
  const std::string cmd = "localedef -c -i de_DE -f UTF-8 '" + (dir / "de_DE.UTF-8").string() + "' >/dev/null 2>&1";
  // glibc declares system() warn_unused_result under _FORTIFY_SOURCE (Ubuntu's GCC defines it by default), and a
  // void cast does not silence that.
  [[maybe_unused]] const int status = std::system(cmd.c_str());  // warnings make it nonzero; setlocale decides
  setenv("LOCPATH", dir.c_str(), 1);
  if (std::setlocale(LC_ALL, "de_DE.UTF-8") != nullptr) return "de_DE.UTF-8 (localedef)";
  unsetenv("LOCPATH");
#endif
  return {};
}

void restore_c_locale() {
  std::setlocale(LC_ALL, "C");
#ifndef _WIN32
  unsetenv("LOCPATH");
#endif
}

TEST(HardwareProfile, RoundTrip) {
  mi::testing::TempDir dir("profile_roundtrip");
  const fs::path file = dir.path() / "sub" / "hardware_profile.json";
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, sample()));
  const auto back = mi::pipeline::read_profile(file, kKey, kNow);
  ASSERT_TRUE(back.has_value()) << read_text(file);
  expect_same(sample(), *back);
  EXPECT_TRUE(back->probe.measured);
  // No temporary files left behind.
  int entries = 0;
  for (const auto& e : fs::directory_iterator(file.parent_path())) entries += e.is_regular_file();
  EXPECT_EQ(entries, 1);
  // Without a self-check result.
  StoredProfile p = sample();
  p.fused_check.reset();
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, p));
  EXPECT_FALSE(mi::pipeline::read_profile(file, kKey, kNow)->fused_check.has_value());
}

// Missing, corrupt, foreign, stale or implausible files read as "no profile" (measure again), never as numbers.
TEST(HardwareProfile, RejectsCorruptForeignAndStaleFiles) {
  mi::testing::TempDir dir("profile_rejectscorru");
  const fs::path file = dir.path() / "hardware_profile.json";
  EXPECT_FALSE(mi::pipeline::read_profile(file, kKey, kNow).has_value());  // missing
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, sample()));
  const std::string good = read_text(file);
  EXPECT_FALSE(mi::pipeline::read_profile(file, kKey + " ", kNow).has_value());  // another GPU / driver / library
  EXPECT_FALSE(mi::pipeline::read_profile(file, kKey, kNow + mi::pipeline::kProfileMaxAgeSeconds).has_value());  // old
  EXPECT_FALSE(mi::pipeline::read_profile(file, kKey, sample().probed_at - 1).has_value());  // from the future
  const auto replaced = [&](const std::string& from, const std::string& to) {
    std::string s = good;
    const std::size_t at = s.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    if (at != std::string::npos) s.replace(at, from.size(), to);
    return s;
  };
  const std::vector<std::string> bad = {
      good.substr(0, good.size() / 2),               // truncated
      "",                                             // empty
      std::string(64, '\0'),                          // garbage
      "[1, 2, 3]",                                    // not an object
      replaced("\"version\": 2", "\"version\": 1"),  // other format
      replaced("\"h2d_gbps\": 24", "\"h2d_gbps\": -24"),
      replaced("\"host_gbps\": 19.5", "\"host_gbps\": 1e300"),
      replaced("\"bf16_tflops\": 73.7", "\"bf16_tflops\": \"fast\""),
      replaced("\"probed_at\": ", "\"probed_at\": 0.5, \"x\": "),
      replaced("\"learned\"", "\"learnt\""),
  };
  for (const std::string& text : bad) {
    write_text(file, text);
    EXPECT_FALSE(mi::pipeline::read_profile(file, kKey, kNow).has_value()) << text;
  }
  write_text(file, good);
  EXPECT_TRUE(mi::pipeline::read_profile(file, kKey, kNow).has_value());
  // A malformed self-check result reads as "not checked" (check again); the rates stay.
  for (const std::string& text : {replaced("\"ok\": true", "\"ok\": 1"), replaced("\"rows\": 384", "\"rows\": -1"),
                                  replaced("\"validated\"", "\"valid\"")}) {
    write_text(file, text);
    const auto p = mi::pipeline::read_profile(file, kKey, kNow);
    ASSERT_TRUE(p.has_value()) << text;
    EXPECT_FALSE(p->fused_check.has_value()) << text;
    EXPECT_EQ(p->probe.h2d_gbps, 24.0);
  }
}

// N2: the stored self-check belongs to the fused kernels that ran it. Another kFusedKernelsVersion (or none: an older
// build), or a pass without a validated configuration, reads as "not checked", so it runs again; the rates stay.
TEST(HardwareProfile, FusedCheckIsTiedToTheKernelsVersion) {
  mi::testing::TempDir dir("profile_kernelsversion");
  const fs::path file = dir.path() / "hardware_profile.json";
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, sample()));
  const std::string good = read_text(file);
  const std::string field = "\"kernels_version\": " + std::to_string(mi::kernels::fused::kFusedKernelsVersion);
  const std::size_t at = good.find(field);
  ASSERT_NE(at, std::string::npos) << good;
  ASSERT_TRUE(mi::pipeline::read_profile(file, kKey, kNow)->fused_check.has_value());
  std::string other = good, none = good, empty = good;
  other.replace(at, field.size(), "\"kernels_version\": " + std::to_string(mi::kernels::fused::kFusedKernelsVersion + 1));
  none.replace(at, field.size() + 2, "");  // drops `"kernels_version": N, `
  StoredProfile no_configs = sample();
  no_configs.fused_check->validated.clear();
  for (const std::string& text : {other, none}) {
    write_text(file, text);
    const auto p = mi::pipeline::read_profile(file, kKey, kNow);
    ASSERT_TRUE(p.has_value()) << text;
    EXPECT_FALSE(p->fused_check.has_value()) << text;
    EXPECT_EQ(p->learned.bf16_tflops, 73.7);
  }
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, no_configs));
  EXPECT_FALSE(mi::pipeline::read_profile(file, kKey, kNow)->fused_check.has_value());
  // A failed check has no configurations and stays failed.
  no_configs.fused_check->ok = false;
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, no_configs));
  const auto failed = mi::pipeline::read_profile(file, kKey, kNow);
  ASSERT_TRUE(failed->fused_check.has_value());
  EXPECT_FALSE(failed->fused_check->ok);
}

// N5: a non-finite rate is written as JSON null (never inf/nan, which no JSON reader accepts) and reads as missing: a
// learned rate falls back to the probed one, a probed rate makes the profile unusable (measure again).
TEST(HardwareProfile, NonFiniteRatesWriteNullAndReadAsMissing) {
  mi::testing::TempDir dir("profile_nonfinite");
  const fs::path file = dir.path() / "hardware_profile.json";
  StoredProfile p = sample();
  p.learned.bf16_tflops = std::numeric_limits<double>::quiet_NaN();
  p.learned.host_gbps = std::numeric_limits<double>::infinity();
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, p));
  const std::string text = read_text(file);
  EXPECT_EQ(text.find("nan"), std::string::npos) << text;
  EXPECT_EQ(text.find("inf"), std::string::npos) << text;
  EXPECT_NE(text.find("\"bf16_tflops\": null"), std::string::npos) << text;
  EXPECT_NO_THROW(static_cast<void>(mi::json::parse(text)));
  const auto back = mi::pipeline::read_profile(file, kKey, kNow);
  ASSERT_TRUE(back.has_value()) << text;
  EXPECT_EQ(back->learned.bf16_tflops, p.probe.bf16_tflops);
  EXPECT_EQ(back->learned.host_gbps, p.probe.host_gbps);
  EXPECT_EQ(back->learned.h2d_gbps, p.probe.h2d_gbps);
  EXPECT_TRUE(back->fused_check.has_value());
  p = sample();
  p.probe.disk_gbps = -std::numeric_limits<double>::infinity();
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, p));
  EXPECT_NO_THROW(static_cast<void>(mi::json::parse(read_text(file))));
  EXPECT_FALSE(mi::pipeline::read_profile(file, kKey, kNow).has_value());
}

// Concurrent writers (two pipelines sharing a cache directory) never leave a torn file: each write goes to its own
// temporary file and is renamed over the profile.
TEST(HardwareProfile, ConcurrentWritersNeverTearTheFile) {
  mi::testing::TempDir dir("profile_concurrentwr");
  const fs::path file = dir.path() / "hardware_profile.json";
  ASSERT_TRUE(mi::pipeline::write_profile(file, kKey, sample()));
  std::vector<std::thread> writers;
  for (int t = 0; t < 4; ++t) {
    writers.emplace_back([&, t] {
      StoredProfile p = sample();
      for (int i = 0; i < 50; ++i) {
        p.learned.bf16_tflops = 40.0 + t + i * 0.25;
        EXPECT_TRUE(mi::pipeline::write_profile(file, kKey, p));
      }
    });
  }
  int reads = 0;
  for (int i = 0; i < 200; ++i, ++reads) EXPECT_TRUE(mi::pipeline::read_profile(file, kKey, kNow).has_value());
  for (std::thread& w : writers) w.join();
  int entries = 0;
  for (const auto& e : fs::directory_iterator(dir.path())) entries += e.is_regular_file();
  EXPECT_EQ(entries, 1);
  EXPECT_EQ(reads, 200);
}

// The cache files' reader (util/replaceable_file.h): whole files with a size limit, and a file that writers keep
// replacing with rename_replace reads as one of its complete versions every time (Windows: a replace in flight is
// retried, never reported as missing or unreadable).
TEST(ReplaceableFile, StatusesAndReadsThroughRenames) {
  using mi::util::ReadStatus;
  mi::testing::TempDir dir("replaceable_file");
  const fs::path file = dir.path() / "cache.json";
  std::string text = "stale";
  EXPECT_EQ(mi::util::read_replaceable_file(file, 100, text), ReadStatus::kMissing);
  EXPECT_TRUE(text.empty());
  std::ofstream(file, std::ios::binary) << std::string(100, 'a');
  EXPECT_EQ(mi::util::read_replaceable_file(file, 100, text), ReadStatus::kOk);
  EXPECT_EQ(text, std::string(100, 'a'));
  EXPECT_EQ(mi::util::read_replaceable_file(file, 99, text), ReadStatus::kTooLarge);
  EXPECT_TRUE(text.empty());
  std::ofstream(file, std::ios::binary | std::ios::trunc).close();
  EXPECT_EQ(mi::util::read_replaceable_file(file, 0, text), ReadStatus::kOk);
  EXPECT_TRUE(text.empty());

  // Each version: 4000 copies of one digit (a torn or partial read would mix digits or be short).
  std::vector<std::thread> writers;
  for (int t = 0; t < 3; ++t) {
    writers.emplace_back([&, t] {
      for (int i = 0; i < 60; ++i) {
        const fs::path tmp = dir.path() / ("cache.json.tmp." + std::to_string(t));
        std::ofstream(tmp, std::ios::binary | std::ios::trunc) << std::string(4000, static_cast<char>('0' + (t * 3 + i) % 10));
        std::error_code ec;
        mi::util::rename_replace(tmp, file, ec);
        EXPECT_FALSE(ec) << ec.message();
      }
    });
  }
  for (int i = 0; i < 300; ++i) {
    ASSERT_EQ(mi::util::read_replaceable_file(file, 1 << 20, text), ReadStatus::kOk) << "read " << i;
    if (text.empty()) continue;  // the initial empty version
    ASSERT_EQ(text.size(), 4000u) << "read " << i;
    EXPECT_EQ(text.find_first_not_of(text[0]), std::string::npos) << "read " << i << " mixes versions";
  }
  for (std::thread& w : writers) w.join();
}

// LOW-2: the JSON writers ignore the process locale. Under a comma-decimal locale (installed, or generated with
// localedef), the profile and the planner's report still parse and keep their values.
TEST(HardwareProfile, JsonWritersIgnoreTheProcessLocale) {
  mi::testing::TempDir dir("profile_jsonwritersi");
  const std::string used = set_comma_locale(dir.path());
  std::printf("comma-decimal locale for this test: %s\n", used.empty() ? "none available (C locale only)" : used.c_str());
  RecordProperty("comma_locale", used.empty() ? "none" : used);
  if (!used.empty()) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.1f", 0.5);
    EXPECT_STREQ(buf, "0,5") << "the locale does not use a decimal comma";
  }
  const fs::path file = dir.path() / "hardware_profile.json";
  const bool written = mi::pipeline::write_profile(file, kKey, sample());
  const auto back = mi::pipeline::read_profile(file, kKey, kNow);
  const std::string text = read_text(file);

  mi::planner::ExecutionPlan plan;
  plan.dit.transfer_s = 0.125;
  plan.dit.hidden = 0.96875;
  plan.vae.cost = 2.5;
  plan.time.denoise_s = 61.25;
  plan.explanation = "Est. time 61.25 s";
  const std::string report = plan.to_json();
  restore_c_locale();

  ASSERT_TRUE(written);
  ASSERT_TRUE(back.has_value()) << text;
  expect_same(sample(), *back);
  const mi::json::Value v = mi::json::parse(report);
  EXPECT_EQ(v.at("dit").at("transfer_s_per_step").as_double(), 0.125);
  EXPECT_EQ(v.at("dit").at("expected_hidden").as_double(), 0.96875);
  EXPECT_EQ(v.at("vae").at("relative_cost").as_double(), 2.5);
  EXPECT_EQ(v.at("time_s").at("denoise").as_double(), 61.25);
  // The CLI and C-API report (pipeline/report_json.cpp, run under a locale by PipelineReport.* and CApiReport.*)
  // formats its numbers the same way; checked at the source too.
  for (const char* src : {"/src/pipeline/report_json.cpp", "/src/planner/planner.cpp", "/src/pipeline/hardware_profile.cpp"}) {
    std::ifstream in(std::string(MOSSINFER_SOURCE_DIR) + src);
    const std::string code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(code.empty()) << src;
    EXPECT_NE(code.find("std::to_chars"), std::string::npos) << src;
    for (const char* banned : {"\"%.6g\"", "\"%.4g\"", "%.4g,"}) EXPECT_EQ(code.find(banned), std::string::npos) << src << " " << banned;
  }
  // Non-finite numbers are written as null, not as invalid JSON.
  plan.dit.hidden = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(mi::json::parse(plan.to_json()).at("dit").at("expected_hidden").is_null());
}

// M4: verdicts flip only after kEvidence agreeing generations, from any source, and recover.
TEST(ProfileLearner, VerdictsNeedConsistentEvidence) {
  ProfileLearner l(sample());
  EXPECT_TRUE(l.dit_weights_cached());
  // One disturbed streamed denoise (another program evicted the page cache) changes nothing.
  l.observe_dit_stream(3.0);
  l.end_generation();
  EXPECT_TRUE(l.dit_weights_cached());
  l.observe_dit_stream(19.0);  // and the streak is broken
  l.observe_dit_stream(3.0);
  l.observe_dit_stream(3.1);
  EXPECT_TRUE(l.dit_weights_cached());
  l.observe_dit_load(2.4);  // a third cold read, from a DiT load: any source counts
  EXPECT_FALSE(l.dit_weights_cached());
  // Recovery: the plan now avoids the files, but DiT loads still read them.
  l.observe_dit_load(9.8);  // one reader from the page cache: above the cold/cached split
  l.observe_dit_load(10.1);
  EXPECT_FALSE(l.dit_weights_cached());
  l.observe_dit_stream(19.2);
  EXPECT_TRUE(l.dit_weights_cached());
  // Text encoder: its own verdict, same rule.
  for (int i = 0; i < ProfileLearner::kEvidence - 1; ++i) l.observe_te_stream(false);
  EXPECT_TRUE(l.te_weights_cached());
  l.observe_te_stream(false);
  EXPECT_FALSE(l.te_weights_cached());
  // No evidence for kRecheckGenerations generations: re-checked (assumed cached again).
  for (int i = 0; i < ProfileLearner::kRecheckGenerations - 1; ++i) l.end_generation();
  EXPECT_FALSE(l.te_weights_cached());
  l.end_generation();
  EXPECT_TRUE(l.te_weights_cached());
  EXPECT_TRUE(l.dit_weights_cached());  // never re-checked while it says cached
}

// M4: rates are medians of recent estimates, bounded around the probe; only agreeing estimates reach the disk.
TEST(ProfileLearner, RatesAreRobustAndPersistOnlyWhenConsistent) {
  StoredProfile p = sample();
  p.learned = p.probe;  // 40 TFLOPS, 19.5 GB/s
  ProfileLearner l(p);
  EXPECT_FALSE(l.observe_tflops(20.0));  // one disturbed run (a busy host): used, never persisted
  EXPECT_EQ(l.current().bf16_tflops, 20.0);
  EXPECT_EQ(l.stored().learned.bf16_tflops, 40.0);
  EXPECT_FALSE(l.observe_tflops(72.0));
  EXPECT_FALSE(l.observe_tflops(74.0));  // 20, 72, 74: not consistent
  EXPECT_EQ(l.stored().learned.bf16_tflops, 40.0);
  EXPECT_TRUE(l.observe_tflops(73.0));  // 72, 74, 73 agree
  EXPECT_EQ(l.stored().learned.bf16_tflops, 72.5);  // the median of 20, 72, 74, 73
  EXPECT_FALSE(l.observe_tflops(15.0));  // another disturbed run: the median barely moves, nothing persisted
  EXPECT_EQ(l.current().bf16_tflops, 72.0);
  EXPECT_EQ(l.stored().learned.bf16_tflops, 72.5);
  // Bounds: kTflopsMax x the probe.
  ProfileLearner wild(p);
  for (int i = 0; i < 3; ++i) wild.observe_tflops(5000.0);
  EXPECT_EQ(wild.current().bf16_tflops, 40.0 * ProfileLearner::kTflopsMax);
  // Page-cache rate: cached reads only, bounded, persisted once consistent.
  EXPECT_FALSE(l.observe_dit_stream(2.5));  // cold: no rate
  EXPECT_EQ(l.current().host_gbps, 19.5);
  EXPECT_FALSE(l.observe_dit_stream(18.0));
  EXPECT_FALSE(l.observe_dit_stream(18.4));
  EXPECT_TRUE(l.observe_dit_stream(18.2));
  EXPECT_EQ(l.stored().learned.host_gbps, 18.2);
  for (int i = 0; i < 5; ++i) l.observe_dit_stream(1000.0);
  EXPECT_EQ(l.current().host_gbps, 19.5 * ProfileLearner::kRateMax);
  // Non-finite or non-positive estimates are ignored.
  EXPECT_FALSE(l.observe_tflops(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(l.observe_tflops(-1.0));
}

}  // namespace
