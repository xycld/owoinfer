#include "util/replaceable_file.h"

#include <system_error>

#include "util/regular_file.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <thread>
#else
#include <algorithm>
#include <cerrno>

#include <unistd.h>
#endif

namespace mi::util {

#ifdef _WIN32

ReadStatus read_replaceable_file(const std::filesystem::path& path, std::size_t max_bytes, std::string& out) {
  out.clear();
  HANDLE h = INVALID_HANDLE_VALUE;
  // Backoff 1, 2, 4, 8, 16, 16, ... ms: about 100 ms before giving up on a file a writer keeps replacing.
  DWORD error = 0;
  for (int attempt = 0, waited_ms = 0;; ++attempt) {
    void* handle = nullptr;
    const OpenStatus opened = open_regular_file(path, FILE_ATTRIBUTE_NORMAL, handle);
    if (opened == OpenStatus::kNotRegular) return ReadStatus::kError;  // a pipe or a device: never waited on
    if (opened == OpenStatus::kOk) {
      h = static_cast<HANDLE>(handle);
      break;
    }
    error = GetLastError();
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return ReadStatus::kMissing;
    if ((error != ERROR_SHARING_VIOLATION && error != ERROR_ACCESS_DENIED) || waited_ms >= 100) return ReadStatus::kError;
    const int ms = std::min(1 << std::min(attempt, 4), 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    waited_ms += ms;
  }
  ReadStatus status = ReadStatus::kOk;
  LARGE_INTEGER size;
  if (!GetFileSizeEx(h, &size) || size.QuadPart < 0) {
    status = ReadStatus::kError;
  } else if (static_cast<unsigned long long>(size.QuadPart) > max_bytes) {
    status = ReadStatus::kTooLarge;
  } else {
    // Read to the end (the size is a hint: a replaced file may differ), at most max_bytes + 1.
    out.resize(static_cast<std::size_t>(size.QuadPart) + 1);
    std::size_t done = 0;
    for (;;) {
      if (done == out.size()) {
        if (out.size() > max_bytes) break;
        out.resize(std::min(out.size() * 2, max_bytes + 1));
      }
      DWORD got = 0;
      const DWORD want = static_cast<DWORD>(std::min<std::size_t>(out.size() - done, std::size_t{1} << 30));
      if (!ReadFile(h, out.data() + done, want, &got, nullptr)) {
        status = ReadStatus::kError;
        break;
      }
      if (got == 0) break;
      done += got;
    }
    if (status == ReadStatus::kOk && done > max_bytes) status = ReadStatus::kTooLarge;
    out.resize(status == ReadStatus::kOk ? done : 0);
  }
  CloseHandle(h);
  return status;
}

#else

ReadStatus read_replaceable_file(const std::filesystem::path& path, std::size_t max_bytes, std::string& out) {
  out.clear();
  int fd = -1;
  // A FIFO or a device is an error at once (open_regular_file), never a read that blocks.
  const OpenStatus opened = open_regular_file(path, 0, fd);
  if (opened == OpenStatus::kNotRegular) return ReadStatus::kError;
  if (opened != OpenStatus::kOk) {
    std::error_code ec;
    return std::filesystem::exists(path, ec) || ec ? ReadStatus::kError : ReadStatus::kMissing;
  }
  // At most max_bytes + 1, growing as the file goes (no size query: the file may be replaced meanwhile).
  std::size_t done = 0;
  bool failed = false;
  out.resize(std::min<std::size_t>(max_bytes + 1, std::size_t{64} << 10));
  for (;;) {
    const ssize_t n = ::read(fd, out.data() + done, std::min<std::size_t>(out.size() - done, std::size_t{1} << 30));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      failed = n < 0;
      break;
    }
    done += static_cast<std::size_t>(n);
    if (done > max_bytes) break;
    if (done == out.size()) out.resize(std::min(out.size() * 2, max_bytes + 1));
  }
  ::close(fd);
  if (failed) {
    out.clear();
    return ReadStatus::kError;
  }
  if (done > max_bytes) {
    out.clear();
    return ReadStatus::kTooLarge;
  }
  out.resize(done);
  return ReadStatus::kOk;
}

#endif

}  // namespace mi::util
