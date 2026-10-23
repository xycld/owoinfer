#include "planner/host_memory.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// windows.h first: psapi.h needs its types.
#include <psapi.h>
#else
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#endif

namespace mi::planner {

#ifdef _WIN32

HostMemory process_host_memory() {
  HostMemory m;
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
    m.valid = true;
    m.private_bytes = static_cast<std::size_t>(pmc.PrivateUsage);
    m.rss_bytes = static_cast<std::size_t>(pmc.WorkingSetSize);
  }
  return m;
}

std::size_t host_available_bytes() {
  MEMORYSTATUSEX s{};
  s.dwLength = sizeof(s);
  return GlobalMemoryStatusEx(&s) ? static_cast<std::size_t>(s.ullAvailPhys) : 0;
}

CgroupMemory cgroup_memory(std::string_view, std::string_view, const std::filesystem::path&) { return {}; }

std::size_t host_total_bytes() {
  MEMORYSTATUSEX s{};
  s.dwLength = sizeof(s);
  return GlobalMemoryStatusEx(&s) ? static_cast<std::size_t>(s.ullTotalPhys) : 0;
}

#else

namespace {

// Values of "<key>:   <n> kB" lines of a /proc file, in bytes. Returns false if the file cannot be read.
template <std::size_t N>
bool read_kb_fields(const char* path, const char* (&keys)[N], std::size_t (&out)[N], bool (&found)[N]) {
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) return false;
  char line[256];
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t len = std::strlen(keys[i]);
      if (std::strncmp(line, keys[i], len) != 0 || line[len] != ':') continue;
      unsigned long long kb = 0;
      if (std::sscanf(line + len + 1, "%llu", &kb) == 1) {
        out[i] = static_cast<std::size_t>(kb) * 1024;
        found[i] = true;
      }
    }
  }
  std::fclose(f);
  return true;
}

}  // namespace

HostMemory process_host_memory() {
  const char* keys[] = {"RssAnon", "RssShmem", "VmRSS", "VmSwap"};
  std::size_t v[4] = {};
  bool found[4] = {};
  HostMemory m;
  if (!read_kb_fields("/proc/self/status", keys, v, found) || !found[0] || !found[1]) return m;
  m.valid = true;
  m.anon_bytes = v[0];
  m.shmem_bytes = v[1];
  m.rss_bytes = v[2];
  m.swap_bytes = v[3];  // absent (0) on kernels without swap accounting
  m.private_bytes = v[0] + v[1] + v[3];
  return m;
}

namespace {

std::string read_text(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// The first line of a cgroup file as a number: true with `out` set, or with `unlimited` for "max"; false when absent or
// not a number.
bool cgroup_value(const std::filesystem::path& p, std::size_t& out, bool& unlimited) {
  std::istringstream in(read_text(p));
  std::string word;
  if (!(in >> word)) return false;
  unlimited = word == "max";
  if (unlimited) return true;
  std::uint64_t v = 0;
  const auto [end, ec] = std::from_chars(word.data(), word.data() + word.size(), v);
  if (ec != std::errc{} || end != word.data() + word.size()) return false;  // not a number, or out of range
  out = static_cast<std::size_t>(v);
  return true;
}

// `key`'s value in a memory.stat file (0 when absent).
std::size_t stat_value(const std::filesystem::path& p, const char* key) {
  std::istringstream in(read_text(p));
  std::string k;
  unsigned long long v = 0;
  while (in >> k >> v) {
    if (k == key) return static_cast<std::size_t>(v);
  }
  return 0;
}

// /proc/self/mountinfo escapes spaces, tabs, newlines and backslashes in paths as \ooo octal.
std::string unescape(const std::string& s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 3 < s.size() && s.substr(i + 1, 3).find_first_not_of("01234567") == std::string::npos) {
      out += static_cast<char>(std::stoi(s.substr(i + 1, 3), nullptr, 8));
      i += 3;
    } else {
      out += s[i];
    }
  }
  return out;
}

struct Mount {
  std::string root, point;
};

// The cgroup v2 mount and the cgroup v1 memory controller's mount, from mountinfo (defaults: the usual locations). With
// several mounts of a kind, the one whose root is the longest prefix of the process's cgroup path (v2_path, v1_path)
// wins, else one with root "/", else the first.
void cgroup_mounts(std::string_view mountinfo, const std::string& v2_path, const std::string& v1_path, Mount& v2, Mount& v1) {
  std::vector<Mount> v2s, v1s;
  std::istringstream lines{std::string(mountinfo)};
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream f(line);
    std::string id, parent, dev, root, point, opts, field;
    if (!(f >> id >> parent >> dev >> root >> point >> opts)) continue;
    while (f >> field && field != "-") {
    }
    std::string fstype, source, super;
    f >> fstype >> source >> super;
    if (fstype == "cgroup2") v2s.push_back({unescape(root), unescape(point)});
    if (fstype == "cgroup" && ("," + super + ",").find(",memory,") != std::string::npos) {
      v1s.push_back({unescape(root), unescape(point)});
    }
  }
  const auto pick = [](const std::vector<Mount>& ms, const std::string& path, Mount fallback) {
    const Mount* best = nullptr;
    for (const Mount& m : ms) {
      const bool contains = m.root != "/" && (path == m.root || path.rfind(m.root + "/", 0) == 0);
      if (contains && (best == nullptr || m.root.size() > best->root.size())) best = &m;
    }
    if (best == nullptr) {
      for (const Mount& m : ms) {
        if (m.root == "/") {
          best = &m;
          break;
        }
      }
    }
    if (best == nullptr && !ms.empty()) best = &ms.front();
    return best != nullptr ? *best : fallback;
  };
  v2 = pick(v2s, v2_path, {"/", "/sys/fs/cgroup"});
  v1 = pick(v1s, v1_path, {"/", "/sys/fs/cgroup/memory"});
}

// The cgroup directory of `path` (a /proc/self/cgroup path) under `root`, through `m`; the mount point when the path
// is outside the mount's root or does not exist.
std::filesystem::path cgroup_dir(const std::filesystem::path& root, const Mount& m, const std::string& path) {
  std::string rel;
  if (m.root == "/") {
    rel = path;
  } else if (path == m.root || path.rfind(m.root + "/", 0) == 0) {
    rel = path.substr(m.root.size());
  }
  const std::filesystem::path point = root / std::filesystem::path(m.point).relative_path();
  const std::filesystem::path dir = rel.empty() || rel == "/" ? point : point / std::filesystem::path(rel).relative_path();
  std::error_code ec;
  return std::filesystem::is_directory(dir, ec) ? dir : point;
}

}  // namespace

CgroupMemory cgroup_memory(std::string_view self_cgroup, std::string_view mountinfo, const std::filesystem::path& root) {
  std::string v2_path, v1_path;
  bool have_v2 = false, have_v1 = false;
  std::istringstream lines{std::string(self_cgroup)};
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t a = line.find(':'), b = a == std::string::npos ? a : line.find(':', a + 1);
    if (b == std::string::npos) continue;
    const std::string controllers = line.substr(a + 1, b - a - 1);
    if (line.compare(0, a, "0") == 0 && controllers.empty()) {
      v2_path = line.substr(b + 1);
      have_v2 = true;
    }
    if (("," + controllers + ",").find(",memory,") != std::string::npos) {
      v1_path = line.substr(b + 1);
      have_v1 = true;
    }
  }
  Mount v2, v1;
  cgroup_mounts(mountinfo, v2_path, v1_path, v2, v1);
  CgroupMemory out;
  const auto consider = [&](std::size_t limit, std::size_t current, std::size_t inactive) {
    const std::size_t used = current - std::min(current, inactive);
    const std::size_t headroom = std::max<std::size_t>(1, limit - std::min(limit, used));
    if (!out.limited || headroom < out.headroom) out.headroom = headroom;
    if (!out.limited || limit < out.limit) out.limit = limit;
    out.limited = true;
  };
  // Every level from the cgroup up to the mount point.
  const auto walk = [&](const std::filesystem::path& start, const std::filesystem::path& top, auto&& level) {
    for (std::filesystem::path dir = start;; dir = dir.parent_path()) {
      level(dir);
      if (dir == top || !dir.has_relative_path() || dir.parent_path() == dir) break;
    }
  };
  if (have_v2) {
    const std::filesystem::path top = root / std::filesystem::path(v2.point).relative_path();
    walk(cgroup_dir(root, v2, v2_path), top, [&](const std::filesystem::path& dir) {
      std::size_t max = 0, high = 0, current = 0;
      bool max_unlimited = true, high_unlimited = true, unused = false;
      const bool has_max = cgroup_value(dir / "memory.max", max, max_unlimited) && !max_unlimited;
      const bool has_high = cgroup_value(dir / "memory.high", high, high_unlimited) && !high_unlimited;
      if (!has_max && !has_high) return;
      if (!cgroup_value(dir / "memory.current", current, unused)) return;
      const std::size_t limit = has_max && has_high ? std::min(max, high) : has_max ? max : high;
      consider(limit, current, stat_value(dir / "memory.stat", "inactive_file"));
    });
  }
  if (!out.limited && have_v1) {
    const std::filesystem::path top = root / std::filesystem::path(v1.point).relative_path();
    walk(cgroup_dir(root, v1, v1_path), top, [&](const std::filesystem::path& dir) {
      std::size_t limit = 0, usage = 0;
      bool unlimited = false, unused = false;
      // v1 reports "no limit" as a huge page-rounded number (2^63 - 4096 on x86-64): anything at or above 2^62 is none.
      if (!cgroup_value(dir / "memory.limit_in_bytes", limit, unlimited) || unlimited || limit >= (std::size_t{1} << 62)) return;
      if (!cgroup_value(dir / "memory.usage_in_bytes", usage, unused)) return;
      std::size_t inactive = stat_value(dir / "memory.stat", "total_inactive_file");
      if (inactive == 0) inactive = stat_value(dir / "memory.stat", "inactive_file");
      consider(limit, usage, inactive);
    });
  }
  return out;
}

std::size_t host_available_bytes() {
  const char* keys[] = {"MemAvailable"};
  std::size_t v[1] = {};
  bool found[1] = {};
  if (!read_kb_fields("/proc/meminfo", keys, v, found) || !found[0]) return 0;
  const CgroupMemory cg = cgroup_memory(read_text("/proc/self/cgroup"), read_text("/proc/self/mountinfo"), "/");
  return cg.limited && cg.headroom < v[0] ? cg.headroom : v[0];
}

std::size_t host_total_bytes() {
  const char* keys[] = {"MemTotal"};
  std::size_t v[1] = {};
  bool found[1] = {};
  if (!read_kb_fields("/proc/meminfo", keys, v, found) || !found[0]) return 0;
  const CgroupMemory cg = cgroup_memory(read_text("/proc/self/cgroup"), read_text("/proc/self/mountinfo"), "/");
  return cg.limited && cg.limit < v[0] ? cg.limit : v[0];
}

#endif

}  // namespace mi::planner
