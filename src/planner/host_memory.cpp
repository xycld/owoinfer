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
#include <cstdio>
#include <cstring>
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

std::size_t host_available_bytes() {
  const char* keys[] = {"MemAvailable"};
  std::size_t v[1] = {};
  bool found[1] = {};
  if (!read_kb_fields("/proc/meminfo", keys, v, found) || !found[0]) return 0;
  return v[0];
}

#endif

}  // namespace mi::planner
