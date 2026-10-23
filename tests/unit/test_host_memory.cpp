// planner/host_memory.h on fixture cgroup trees (CPU only): how a container's memory limit caps what the planner thinks
// the host can give (cgroup v2 limits at the cgroup or an ancestor, memory.high, the working-set rule, cgroup v1's
// "no limit" value, hybrid hosts, the Docker v1 layout, a namespace's bind-mounted host tree, missing and malformed
// files), and the pin-margin rule.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "planner/host_memory.h"
#include "test_util.h"

namespace {

namespace fs = std::filesystem;
using mi::planner::CgroupMemory;
using mi::planner::cgroup_memory;

constexpr std::size_t kGiB = std::size_t{1} << 30;

void put(const fs::path& file, const std::string& text) {
  fs::create_directories(file.parent_path());
  std::ofstream(file, std::ios::binary) << text;
}

const char* kV2Mounts = "30 1 0:26 / /sys/fs/cgroup rw,nosuid shared:9 - cgroup2 cgroup2 rw,nsdelegate\n";

TEST(HostMemoryCgroup, V2LimitAtAnAncestorWithTheWorkingSetRule) {
  mi::testing::TempDir dir("cg_v2");
  const fs::path cg = dir.path() / "sys/fs/cgroup";
  put(cg / "a/memory.max", "8589934592\n");  // 8 GiB
  put(cg / "a/memory.current", "6442450944\n");  // 6 GiB, of which 2 GiB reclaimable page cache
  put(cg / "a/memory.stat", "anon 4294967296\nfile 2147483648\ninactive_file 2147483648\n");
  put(cg / "a/b/memory.max", "max\n");
  put(cg / "a/b/memory.current", "1073741824\n");
  const CgroupMemory m = cgroup_memory("0::/a/b\n", kV2Mounts, dir.path());
  ASSERT_TRUE(m.limited);
  EXPECT_EQ(m.limit, 8 * kGiB);
  EXPECT_EQ(m.headroom, 4 * kGiB) << "8 GiB - (6 GiB - 2 GiB inactive file)";
}

TEST(HostMemoryCgroup, V2MemoryHighAndUnlimitedTrees) {
  mi::testing::TempDir dir("cg_v2high");
  const fs::path cg = dir.path() / "sys/fs/cgroup";
  put(cg / "app/memory.max", "8589934592\n");
  put(cg / "app/memory.high", "4294967296\n");
  put(cg / "app/memory.current", "1073741824\n");
  CgroupMemory m = cgroup_memory("0::/app\n", kV2Mounts, dir.path());
  ASSERT_TRUE(m.limited);
  EXPECT_EQ(m.limit, 4 * kGiB) << "min(memory.high, memory.max)";
  EXPECT_EQ(m.headroom, 3 * kGiB);
  put(cg / "free/memory.max", "max\n");
  put(cg / "free/memory.current", "1073741824\n");
  EXPECT_FALSE(cgroup_memory("0::/free\n", kV2Mounts, dir.path()).limited);
  // Used beyond the limit: still limited, with the smallest headroom (never 0, which would read as unknown).
  put(cg / "full/memory.max", "1073741824\n");
  put(cg / "full/memory.current", "2147483648\n");
  m = cgroup_memory("0::/full\n", kV2Mounts, dir.path());
  ASSERT_TRUE(m.limited);
  EXPECT_EQ(m.headroom, 1u);
}

TEST(HostMemoryCgroup, MissingAndMalformedFilesAreNoLimit) {
  mi::testing::TempDir dir("cg_bad");
  const fs::path cg = dir.path() / "sys/fs/cgroup";
  EXPECT_FALSE(cgroup_memory("0::/nowhere\n", kV2Mounts, dir.path()).limited);
  put(cg / "odd/memory.max", "lots\n");
  put(cg / "odd/memory.current", "12\n");
  EXPECT_FALSE(cgroup_memory("0::/odd\n", kV2Mounts, dir.path()).limited);
  put(cg / "nocur/memory.max", "1073741824\n");
  EXPECT_FALSE(cgroup_memory("0::/nocur\n", kV2Mounts, dir.path()).limited) << "a limit without usage is skipped";
  EXPECT_FALSE(cgroup_memory("", "", dir.path()).limited);
  EXPECT_FALSE(cgroup_memory("garbage line\n", "also garbage\n", dir.path()).limited);
}

TEST(HostMemoryCgroup, V1NoLimitValueHybridAndDockerLayout) {
  mi::testing::TempDir dir("cg_v1");
  const fs::path mem = dir.path() / "sys/fs/cgroup/memory";
  const std::string v1_mounts = "40 30 0:30 / /sys/fs/cgroup/memory rw shared:20 - cgroup cgroup rw,memory\n";
  // v1 reports "no limit" as 2^63 - 4096.
  put(mem / "user/memory.limit_in_bytes", "9223372036854771712\n");
  put(mem / "user/memory.usage_in_bytes", "1073741824\n");
  EXPECT_FALSE(cgroup_memory("5:memory:/user\n", v1_mounts, dir.path()).limited);
  // A hybrid host: the unified hierarchy has no limit, the v1 memory controller has one.
  put(mem / "job/memory.limit_in_bytes", "4294967296\n");
  put(mem / "job/memory.usage_in_bytes", "3221225472\n");
  put(mem / "job/memory.stat", "cache 1073741824\ntotal_inactive_file 1073741824\n");
  const std::string hybrid = "0::/job\n5:memory:/job\n";
  const std::string hybrid_mounts = "29 1 0:25 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n" + v1_mounts;
  put(dir.path() / "sys/fs/cgroup/unified/job/memory.max", "max\n");
  CgroupMemory m = cgroup_memory(hybrid, hybrid_mounts, dir.path());
  ASSERT_TRUE(m.limited);
  EXPECT_EQ(m.limit, 4 * kGiB);
  EXPECT_EQ(m.headroom, 2 * kGiB) << "4 GiB - (3 GiB - 1 GiB inactive file)";
  // Docker's v1 layout: the container sees its own cgroup (/docker/<id>) mounted at the controller's mount point.
  mi::testing::TempDir docker("cg_docker");
  put(docker.path() / "sys/fs/cgroup/memory/memory.limit_in_bytes", "2147483648\n");
  put(docker.path() / "sys/fs/cgroup/memory/memory.usage_in_bytes", "1073741824\n");
  const std::string docker_mounts = "600 590 0:40 /docker/abc123 /sys/fs/cgroup/memory ro - cgroup cgroup rw,memory\n";
  m = cgroup_memory("9:memory:/docker/abc123\n", docker_mounts, docker.path());
  ASSERT_TRUE(m.limited);
  EXPECT_EQ(m.limit, 2 * kGiB);
  EXPECT_EQ(m.headroom, 1 * kGiB);
}

// A namespace whose /proc/self/cgroup says "/" while the host's tree is bind-mounted with its pod as the mount root.
TEST(HostMemoryCgroup, NamespaceWithBindMountedHostTree) {
  mi::testing::TempDir dir("cg_ns");
  const fs::path cg = dir.path() / "sys/fs/cgroup";
  put(cg / "memory.max", "17179869184\n");  // 16 GiB at the mount point (the pod)
  put(cg / "memory.current", "4294967296\n");
  const std::string mounts = "700 690 0:50 /kubepods/pod1 /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n";
  const CgroupMemory m = cgroup_memory("0::/\n", mounts, dir.path());
  ASSERT_TRUE(m.limited);
  EXPECT_EQ(m.limit, 16 * kGiB);
  EXPECT_EQ(m.headroom, 12 * kGiB);
}

// Two cgroup2 mounts: the one whose root holds the process's cgroup is used, whichever comes first.
TEST(HostMemoryCgroup, MountWhoseRootHoldsTheCgroupWins) {
  mi::testing::TempDir dir("cg_twomounts");
  put(dir.path() / "other/memory.max", "1073741824\n");  // a foreign tree mounted first: 1 GiB
  put(dir.path() / "other/memory.current", "0\n");
  put(dir.path() / "sys/fs/cgroup/memory.max", "4294967296\n");  // the process's pod, mounted second: 4 GiB
  put(dir.path() / "sys/fs/cgroup/memory.current", "1073741824\n");
  const std::string mounts =
      "800 1 0:60 /system.slice /other rw - cgroup2 cgroup2 rw\n"
      "801 1 0:61 /kubepods/pod7 /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n";
  const CgroupMemory m = cgroup_memory("0::/kubepods/pod7\n", mounts, dir.path());
  ASSERT_TRUE(m.limited);
  EXPECT_EQ(m.limit, 4 * kGiB);
  EXPECT_EQ(m.headroom, 3 * kGiB);
  // No root holds it: the mount with root "/" (here the second) beats the first.
  const std::string slash =
      "800 1 0:60 /system.slice /other rw - cgroup2 cgroup2 rw\n"
      "801 1 0:61 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n";
  put(dir.path() / "sys/fs/cgroup/job/memory.max", "2147483648\n");
  put(dir.path() / "sys/fs/cgroup/job/memory.current", "0\n");
  const CgroupMemory j = cgroup_memory("0::/job\n", slash, dir.path());
  ASSERT_TRUE(j.limited);
  EXPECT_EQ(j.limit, 2 * kGiB) << "the job's own limit, below the 4 GiB at the mount point";
}

TEST(HostMemoryCgroup, PinMarginRule) {
  using mi::planner::fits_pin_margin;
  using mi::planner::host_pin_margin;
  EXPECT_EQ(host_pin_margin(64 * kGiB), 16 * kGiB);
  EXPECT_EQ(host_pin_margin(16 * kGiB), 8 * kGiB);
  EXPECT_TRUE(fits_pin_margin(10 * kGiB, 30 * kGiB, 64 * kGiB));    // leaves 20 >= 16
  EXPECT_FALSE(fits_pin_margin(15 * kGiB, 30 * kGiB, 64 * kGiB));   // leaves 15 < 16
  EXPECT_FALSE(fits_pin_margin(1, 0, 64 * kGiB)) << "unknown room never fits";
  EXPECT_FALSE(fits_pin_margin(40 * kGiB, 30 * kGiB, 64 * kGiB));
}

}  // namespace
