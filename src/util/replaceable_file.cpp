#include "util/replaceable_file.h"

#include <system_error>

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
#include <fstream>
#endif

namespace mi::util {

#ifdef _WIN32

ReadStatus read_replaceable_file(const std::filesystem::path& path, std::size_t max_bytes, std::string& out) {
  out.clear();
  HANDLE h = INVALID_HANDLE_VALUE;
  // Backoff 1, 2, 4, 8, 16, 16, ... ms: about 100 ms before giving up on a file a writer keeps replacing.
  DWORD error = 0;
  for (int attempt = 0, waited_ms = 0;; ++attempt) {
    h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) break;
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
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::error_code ec;
    return std::filesystem::exists(path, ec) || ec ? ReadStatus::kError : ReadStatus::kMissing;
  }
  // At most max_bytes + 1, growing as the file goes (no size query: the file may be replaced meanwhile).
  std::size_t done = 0;
  out.resize(std::min<std::size_t>(max_bytes + 1, std::size_t{64} << 10));
  for (;;) {
    in.read(out.data() + done, static_cast<std::streamsize>(out.size() - done));
    done += static_cast<std::size_t>(in.gcount());
    if (!in) break;
    if (done > max_bytes) break;
    out.resize(std::min(out.size() * 2, max_bytes + 1));
  }
  if (in.bad()) {
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
