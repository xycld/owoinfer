// Lane D (AC25): the standard schedulers, the flow sigma table and the Brownian tree against vectors from the
// reference implementations (k-diffusion family code run as an oracle, tests/data/samplers/manifest.json).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "pipeline/rng_torch_compat.h"
#include "pipeline/samplers/brownian_tree.h"
#include "pipeline/samplers/initial_noise.h"
#include "pipeline/samplers/schedules.h"
#include "util/json.h"

namespace {

namespace fs = std::filesystem;
using mi::pipeline::FlowSampling;
using mi::pipeline::SchedulerKind;

fs::path data_dir() { return fs::path(MOSSINFER_SOURCE_DIR) / "tests" / "data" / "samplers"; }

const mi::json::Value& manifest() {
  static const mi::json::Value doc = [] {
    std::ifstream in(data_dir() / "manifest.json", std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return mi::json::parse(ss.str());
  }();
  return doc;
}

std::uint32_t bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

float float_from_bits(const mi::json::Value& v) {
  const auto u = static_cast<std::uint32_t>(v.as_int());
  float f = 0;
  std::memcpy(&f, &u, 4);
  return f;
}

std::vector<float> floats_from_bits(const mi::json::Value& arr) {
  std::vector<float> out;
  for (const auto& v : arr.as_array()) out.push_back(float_from_bits(v));
  return out;
}

// Distance in units in the last place between two finite floats of the same sign.
std::int64_t ulp_distance(float a, float b) {
  if (a == b) return 0;
  const auto key = [](float f) {
    const std::int64_t u = bits(f);
    return (u & 0x80000000) ? -(u & 0x7fffffff) : u;
  };
  return std::llabs(key(a) - key(b));
}

TEST(SamplerSchedules, SigmaTableBitExact) {
  int tables = 0;
  for (const auto& t : manifest().at("tables").as_array()) {
    const FlowSampling fs{t.at("shift").as_double(), t.at("multiplier").as_double()};
    const std::vector<float> want = floats_from_bits(t.at("sigmas"));
    const std::vector<float> got = mi::pipeline::flow_sigma_table(fs);
    ASSERT_EQ(got.size(), want.size());
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) bad += bits(got[i]) != bits(want[i]);
    EXPECT_EQ(bad, 0u) << "shift " << fs.shift << " multiplier " << fs.multiplier;
    ++tables;
  }
  EXPECT_EQ(tables, 6);
}

TEST(SamplerSchedules, AllSchedulersMatchReference) {
  // Per scheduler: schedules compared, bit-exact ones, worst ulp distance.
  struct Stat {
    int cases = 0, exact = 0;
    std::int64_t max_ulp = 0;
  };
  std::map<std::string, Stat> stats;
  for (const auto& e : manifest().at("schedules").as_array()) {
    const std::string name = e.at("scheduler").as_string();
    const int steps = static_cast<int>(e.at("steps").as_int());
    const FlowSampling fs{e.at("shift").as_double(), e.at("multiplier").as_double()};
    const std::vector<float> want = floats_from_bits(e.at("sigmas"));
    const std::vector<float> got = mi::pipeline::compute_sigmas(mi::pipeline::parse_scheduler(name), steps, fs);
    const std::string what = name + " steps " + std::to_string(steps) + " shift " + std::to_string(fs.shift) +
                             " multiplier " + std::to_string(fs.multiplier);
    ASSERT_EQ(got.size(), want.size()) << what;
    std::int64_t worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, ulp_distance(got[i], want[i]));
    Stat& st = stats[name];
    ++st.cases;
    st.exact += worst == 0;
    st.max_ulp = std::max(st.max_ulp, worst);
    // AC25: bit-exact, or <= 1 ulp where the reference uses a vectorized transcendental (karras pow, exponential exp).
    EXPECT_LE(worst, (name == "karras" || name == "exponential") ? 1 : 0) << what;
  }
  ASSERT_EQ(stats.size(), 6u);
  for (const auto& [name, st] : stats) {
    std::printf("[samplers] %-12s %3d schedules, %3d bit-exact, max %lld ulp\n", name.c_str(), st.cases, st.exact,
                static_cast<long long>(st.max_ulp));
    if (name != "karras" && name != "exponential") {
      EXPECT_EQ(st.exact, st.cases) << name;
    }
  }
}

TEST(SamplerSchedules, PercentToSigma) {
  for (const auto& e : manifest().at("percent_to_sigma").as_array()) {
    const FlowSampling fs{e.at("shift").as_double(), 1.0};
    EXPECT_EQ(mi::pipeline::flow_percent_to_sigma(fs, e.at("percent").as_double()), e.at("sigma").as_double());
  }
}

TEST(SamplerSchedules, MossAndDenoiseAndArguments) {
  // "moss" is the official schedule; make_schedule fills the DiT timesteps.
  const FlowSampling fs{5.0, 1.0};
  const auto moss = mi::pipeline::make_schedule(SchedulerKind::kMoss, 50, fs);
  EXPECT_EQ(moss.sigmas, mi::pipeline::moss_schedule(50, 5.0).sigmas);
  EXPECT_EQ(moss.scheduler, SchedulerKind::kMoss);
  const auto simple = mi::pipeline::make_schedule(SchedulerKind::kSimple, 8, fs);
  ASSERT_EQ(simple.num_steps(), 8);
  EXPECT_EQ(simple.scheduler, SchedulerKind::kSimple);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(simple.timesteps[i], simple.sigmas[i] * 1000.0f);
  // Partial denoise: the tail of the schedule for floor(steps / denoise) steps.
  const auto full = mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 20, fs);
  const auto half = mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 10, fs, 0.5);
  ASSERT_EQ(half.size(), 11u);
  EXPECT_TRUE(std::equal(half.begin(), half.end(), full.end() - 11));
  EXPECT_TRUE(mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 10, fs, 0.0).empty());
  for (const double bad : {-0.1, 1.5, std::nan(""), std::numeric_limits<double>::infinity()}) {
    EXPECT_THROW(mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 10, fs, bad), std::invalid_argument) << bad;
  }
  EXPECT_THROW(mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 10, fs, 1e-9), std::invalid_argument);  // 10^10 steps
  // The schedule carries its flow sampling (denoise() hands it to the sampler).
  const FlowSampling fs3{3.0, 1000.0};
  EXPECT_EQ(mi::pipeline::make_schedule(SchedulerKind::kNormal, 8, fs3).flow.shift, 3.0);
  EXPECT_EQ(mi::pipeline::make_schedule(SchedulerKind::kNormal, 8, fs3).flow.multiplier, 1000.0);
  EXPECT_EQ(mi::pipeline::make_schedule(SchedulerKind::kMoss, 8, fs3).flow.shift, 3.0);
  EXPECT_EQ(mi::pipeline::moss_schedule(8, 2.5).flow.shift, 2.5);
  EXPECT_THROW(mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 0, fs), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 4, FlowSampling{0.0, 1.0}), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::compute_sigmas(SchedulerKind::kSimple, 4, FlowSampling{5.0, -1.0}), std::invalid_argument);
  EXPECT_THROW(mi::pipeline::parse_scheduler("ddim_uniform"), std::invalid_argument);
  for (const char* n : {"moss", "simple", "normal", "karras", "exponential", "beta", "sgm_uniform"}) {
    EXPECT_STREQ(mi::pipeline::scheduler_name(mi::pipeline::parse_scheduler(n)), n);
  }
}

TEST(SamplerSchedules, BetaQuantile) {
  // Symmetry and endpoints; the scheduler test above covers the reference's rint'ed indices.
  EXPECT_EQ(mi::pipeline::beta_quantile(0.0, 0.6, 0.6), 0.0);
  EXPECT_EQ(mi::pipeline::beta_quantile(1.0, 0.6, 0.6), 1.0);
  EXPECT_NEAR(mi::pipeline::beta_quantile(0.5, 0.6, 0.6), 0.5, 1e-15);
  for (const double p : {0.01, 0.1, 0.3, 0.77}) {
    EXPECT_NEAR(mi::pipeline::beta_quantile(p, 0.6, 0.6) + mi::pipeline::beta_quantile(1 - p, 0.6, 0.6), 1.0, 1e-14);
  }
  // Beta(1, 1) is uniform; Beta(2, 1) has quantile sqrt(p).
  EXPECT_NEAR(mi::pipeline::beta_quantile(0.3, 1.0, 1.0), 0.3, 1e-15);
  EXPECT_NEAR(mi::pipeline::beta_quantile(0.36, 2.0, 1.0), 0.6, 1e-15);
  EXPECT_THROW(mi::pipeline::beta_quantile(1.5, 0.6, 0.6), std::invalid_argument);
}

TEST(SamplerBrownian, SeedSequenceMatchesNumpy) {
  int n = 0;
  for (const auto& e : manifest().at("seed_sequence").as_array()) {
    const std::uint64_t entropy = std::stoull(e.at("entropy").as_string());
    std::vector<std::uint64_t> key;
    for (const auto& k : e.at("spawn_key").as_array()) key.push_back(std::stoull(k.as_string()));
    const std::vector<std::uint32_t> got = mi::pipeline::seed_sequence_state(entropy, key, 24, 4);
    std::vector<std::uint32_t> want;
    for (const auto& v : e.at("state").as_array()) want.push_back(static_cast<std::uint32_t>(v.as_int()));
    EXPECT_EQ(got, want) << "entropy " << entropy << " key size " << key.size();
    ++n;
  }
  EXPECT_EQ(n, 30);
}

TEST(SamplerBrownian, PythonRound) {
  EXPECT_EQ(mi::pipeline::python_round(0.17241378128528595, 6), 0.172414);
  EXPECT_EQ(mi::pipeline::python_round(0.9999999, 6), 1.0);
  EXPECT_EQ(mi::pipeline::python_round(0.0049800807, 6), 0.00498);
  EXPECT_EQ(mi::pipeline::python_round(0.5, 6), 0.5);
}

TEST(SamplerBrownian, NoiseBitExact) {
  int draws = 0;
  for (const auto& e : manifest().at("brownian").as_array()) {
    const std::uint64_t seed = std::stoull(e.at("seed").as_string());
    const std::int64_t numel = e.at("numel").as_int();
    const std::vector<float> sig = floats_from_bits(e.at("sigmas"));
    const float smin = float_from_bits(e.at("sigma_min")), smax = float_from_bits(e.at("sigma_max"));
    const auto stride = static_cast<std::size_t>(e.at("stride").as_int());
    mi::pipeline::BrownianTreeNoise tree(numel, smin, smax, seed);
    const auto& all = e.at("draws").as_array();
    for (std::size_t i = 0; i < all.size(); ++i) {
      const std::vector<float> want = floats_from_bits(all[i]);
      const std::vector<float> got = tree.noise(sig[i], sig[i + 1]);
      std::size_t bad = 0;
      for (std::size_t k = 0; k < want.size(); ++k) bad += bits(got[k * stride]) != bits(want[k]);
      EXPECT_EQ(bad, 0u) << "seed " << seed << " step " << i;
      ++draws;
    }
  }
  EXPECT_EQ(draws, 9 + 6);
}

}  // namespace

namespace {

TEST(SamplerInitialNoise, BatchStreamLayoutMatchesReference) {
  using mi::pipeline::NoiseLayout;
  int cases = 0;
  for (const auto& e : manifest().at("initial_noise").as_array()) {
    const std::uint64_t seed = std::stoull(e.at("seed").as_string());
    const int b = static_cast<int>(e.at("batch_size").as_int()), k = static_cast<int>(e.at("batch_index").as_int());
    const std::int64_t h = e.at("h").as_int(), w = e.at("w").as_int();
    const auto stride = static_cast<std::size_t>(e.at("stride").as_int());
    const std::vector<float> got = mi::pipeline::initial_noise(NoiseLayout::kBatchStream, seed, 128, h, w, k, b);
    ASSERT_EQ(got.size(), static_cast<std::size_t>(128 * h * w));
    const std::vector<float> want = floats_from_bits(e.at("values"));
    std::size_t bad = 0;
    for (std::size_t i = 0; i < want.size(); ++i) bad += bits(got[i * stride]) != bits(want[i]);
    EXPECT_EQ(bad, 0u) << "seed " << seed << " batch " << k << "/" << b;
    // One image: the official layout is the same draw.
    if (b == 1) {
      EXPECT_EQ(got, mi::pipeline::initial_noise(NoiseLayout::kOfficial, seed, 128, h, w));
    }
    ++cases;
  }
  EXPECT_EQ(cases, 6);
  EXPECT_EQ(mi::pipeline::initial_noise(NoiseLayout::kOfficial, 7, 128, 4, 4, 1, 2), mi::rng::torch_cpu_randn(7, 2048));
  EXPECT_THROW(mi::pipeline::initial_noise(NoiseLayout::kBatchStream, 7, 128, 4, 4, 2, 2), std::invalid_argument);
  EXPECT_EQ(mi::pipeline::parse_noise_layout("batch_stream"), NoiseLayout::kBatchStream);
  EXPECT_STREQ(mi::pipeline::noise_layout_name(NoiseLayout::kOfficial), "official");
  EXPECT_THROW(mi::pipeline::parse_noise_layout("comfy"), std::invalid_argument);
}

}  // namespace
