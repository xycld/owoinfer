// std::filesystem::rename of a finished temporary file over its destination (the atomic-replace writers).
#pragma once

#include <filesystem>
#include <system_error>

#ifdef _WIN32
#include <chrono>
#include <thread>
#endif

namespace mi::util {

// Windows refuses to replace a file that another thread or process has open at that moment (a reader, a concurrent
// writer's rename), so there the rename is retried for up to about 200 ms. POSIX: one rename.
inline void rename_replace(const std::filesystem::path& from, const std::filesystem::path& to, std::error_code& ec) {
  std::filesystem::rename(from, to, ec);
#ifdef _WIN32
  for (int i = 0; ec && i < 200; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::filesystem::rename(from, to, ec);
  }
#endif
}

}  // namespace mi::util
