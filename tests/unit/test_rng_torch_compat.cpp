#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "pipeline/rng_torch_compat.h"
#include "test_util.h"
#include "util/npy.h"

namespace {

const std::filesystem::path kSource(MOSSINFER_SOURCE_DIR);

// Distance in units in the last place between two finite floats (sign-magnitude -> ordered ints).
std::int64_t ulp_distance(float a, float b) {
  auto ordered = [](float f) {
    const auto bits = static_cast<std::int64_t>(std::bit_cast<std::uint32_t>(f));
    return bits & 0x80000000 ? 0x80000000 - bits : bits;
  };
  return std::llabs(ordered(a) - ordered(b));
}

// Requires bit-identical output; on mismatch reports how many values differ and the max ULP gap.
void expect_matches(const mi::npy::Array& ref, std::uint64_t seed) {
  ASSERT_EQ(ref.dtype, "f4");
  const std::vector<float> got = mi::rng::torch_cpu_randn(seed, ref.numel());
  ASSERT_EQ(static_cast<std::int64_t>(got.size()), ref.numel());
  if (std::memcmp(got.data(), ref.data.data(), ref.data.size()) == 0) return;
  const std::vector<float> ref_values = ref.to_vector<float>();
  std::int64_t mismatches = 0;
  std::int64_t max_ulp = 0;
  std::int64_t first = -1;
  for (std::int64_t i = 0; i < ref.numel(); ++i) {
    const float want = ref_values[i];
    if (std::bit_cast<std::uint32_t>(got[i]) == std::bit_cast<std::uint32_t>(want)) continue;
    if (first < 0) first = i;
    ++mismatches;
    max_ulp = std::max(max_ulp, ulp_distance(got[i], want));
  }
  ADD_FAILURE() << mismatches << " of " << ref.numel() << " values differ, max " << max_ulp
                << " ULP, first at index " << first << ": got " << got[first] << " want "
                << ref_values[first];
}

}  // namespace

TEST(RngTorchCompat, MatchesAllNoiseGoldens) {
  const std::uint64_t seeds[] = {0, 42, 2026};
  const char* sizes[] = {"16x16",     "1024x1024", "1456x1456", "1776x1184",
                         "2048x2048", "4096x512",  "720x2880"};
  int checked = 0;
  for (std::uint64_t seed : seeds) {
    for (const char* size : sizes) {
      const std::string name = "seed" + std::to_string(seed) + "_" + size + ".npy";
      SCOPED_TRACE(name);
      const auto ref = mi::npy::load(kSource / "tests/golden/noise" / name);
      ASSERT_EQ(ref.shape.size(), 4u);
      EXPECT_EQ(ref.shape[1], 128);
      expect_matches(ref, seed);
      ++checked;
    }
  }
  // Every golden in the directory is covered by the lists above.
  int on_disk = 0;
  for (const auto& entry : std::filesystem::directory_iterator(kSource / "tests/golden/noise")) {
    on_disk += entry.path().extension() == ".npy";
  }
  EXPECT_EQ(checked, 21);
  EXPECT_EQ(on_disk, checked);
}

// Vectors from torch's generator: large seeds, the numel < 16 scalar path, and
// tails with numel % 16 != 0.
TEST(RngTorchCompat, MatchesGeneratedVectors) {
  const std::uint64_t seeds[] = {0,           111769238,         2570432421ull, 3072627948ull,
                                 4294967295ull, 4294967296ull, (1ull << 53) - 1};
  const std::int64_t numels[] = {1, 2, 15, 16, 17, 31, 33, 1000, 4097};
  for (std::uint64_t seed : seeds) {
    for (std::int64_t numel : numels) {
      const std::string name = "seed" + std::to_string(seed) + "_n" + std::to_string(numel) + ".npy";
      SCOPED_TRACE(name);
      const auto ref = mi::npy::load(kSource / "tests/data/rng_vectors" / name);
      ASSERT_EQ(ref.numel(), numel);
      expect_matches(ref, seed);
    }
  }
}

TEST(RngTorchCompat, MatchesAssetSeedFullShape) {
  // Asset seed at 1776x1184: latent [1, 128, 111, 74] drawn as one flat randn.
  const auto ref = mi::npy::load(kSource / "tests/data/rng_vectors/seed111769238_n1051392.npy");
  expect_matches(ref, 111769238);
}

TEST(RngTorchCompat, SeedUsesLow32Bits) {
  EXPECT_EQ(mi::rng::torch_cpu_randn(4294967296ull + 42, 64), mi::rng::torch_cpu_randn(42, 64));
}

TEST(RngTorchCompat, HandlesEmptyAndRejectsNegative) {
  EXPECT_TRUE(mi::rng::torch_cpu_randn(0, 0).empty());
  EXPECT_THROW(mi::rng::torch_cpu_randn(0, -1), std::invalid_argument);
}

TEST(RngTorchCompat, LargeShapeIsDeterministicAndFast) {
  constexpr std::int64_t kNumel = 128LL * 91 * 91;  // 1456x1456 latent
  const auto start = std::chrono::steady_clock::now();
  const std::vector<float> a = mi::rng::torch_cpu_randn(2026, kNumel);
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  const std::vector<float> b = mi::rng::torch_cpu_randn(2026, kNumel);
  ASSERT_EQ(static_cast<std::int64_t>(a.size()), kNumel);
  EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0);
  RecordProperty("randn_1456_ms", std::to_string(ms));
  if constexpr (mi::testing::kAssertTimings) {
    EXPECT_LT(ms, 200.0) << "torch_cpu_randn took " << ms << " ms for " << kNumel << " values";
  }
}
