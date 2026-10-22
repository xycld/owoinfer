// A file's last write time as nanoseconds since the system_clock (Unix) epoch, the value persisted in .mossidx files.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>

namespace mi::util {

inline std::int64_t file_mtime_ns(const std::filesystem::path& p) {
  const auto t = std::filesystem::last_write_time(p);
  // libstdc++ before GCC 13 has no std::chrono::clock_cast; its file_clock converts with to_sys (what clock_cast
  // calls). MSVC's file_clock has to_utc instead, so the other compilers take clock_cast.
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE < 13
  const auto sys = std::filesystem::file_time_type::clock::to_sys(t);
#else
  const auto sys = std::chrono::clock_cast<std::chrono::system_clock>(t);
#endif
  return std::chrono::duration_cast<std::chrono::nanoseconds>(sys.time_since_epoch()).count();
}

}  // namespace mi::util
