#include "util/regular_file.h"

#include <cstddef>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <chrono>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mi::util {

namespace {

bool is_sep(wchar_t c) { return c == L'\\' || c == L'/'; }

bool iequals(std::wstring_view a, std::wstring_view b) {
  if (a.size() != b.size()) return false;
  const auto lower = [](wchar_t c) { return c >= L'A' && c <= L'Z' ? static_cast<wchar_t>(c - L'A' + L'a') : c; };
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (lower(a[i]) != lower(b[i])) return false;
  }
  return true;
}

// The path component starting at `pos`; `pos` moves past it and the separator after it.
std::wstring_view component(std::wstring_view s, std::size_t& pos) {
  const std::size_t begin = pos;
  while (pos < s.size() && !is_sep(s[pos])) ++pos;
  const std::wstring_view c = s.substr(begin, pos - begin);
  if (pos < s.size()) ++pos;
  return c;
}

// A UNC path's share, from `pos` (just after the server): a named pipe's or a mailslot's, or missing.
bool device_share(std::wstring_view s, std::size_t pos) {
  const std::wstring_view share = component(s, pos);
  return share.empty() || iequals(share, L"pipe") || iequals(share, L"mailslot");
}

}  // namespace

bool windows_path_names_device(std::wstring_view p) {
  // \\?\ and \\.\ (the Win32 device namespace) and \??\ (the NT one), with either separator.
  const bool ns = p.size() >= 4 && is_sep(p[0]) && is_sep(p[3]) &&
                  ((is_sep(p[1]) && (p[2] == L'?' || p[2] == L'.')) || (p[1] == L'?' && p[2] == L'?'));
  if (ns) {
    std::size_t pos = 4;
    const std::wstring_view first = component(p, pos);
    const wchar_t d = first.empty() ? L'\0' : first[0];
    const bool letter = first.size() == 2 && ((d >= L'a' && d <= L'z') || (d >= L'A' && d <= L'Z'));
    if (letter && first[1] == L':') return !is_sep(p[pos - 1]);  // X:\... is a file; a bare X: is the raw volume
    if (!iequals(first, L"UNC")) return true;
    return component(p, pos).empty() || device_share(p, pos);  // UNC\server\share\...
  }
  if (p.size() >= 2 && is_sep(p[0]) && is_sep(p[1])) {  // \\server\share\...
    std::size_t pos = 2;
    const std::wstring_view server = component(p, pos);
    if (server == L"." || server == L"?") return true;  // \\. or \\? alone
    return device_share(p, pos);
  }
  return false;
}

#ifdef _WIN32

bool is_windows_device_path(const std::filesystem::path& path) {
  if (windows_path_names_device(path.native())) return true;
  // DOS device names (NUL, COM1, C:\m\CON) come out of GetFullPathNameW as \\.\<name>.
  const DWORD size = GetFullPathNameW(path.c_str(), 0, nullptr, nullptr);
  if (size == 0) return true;
  std::wstring full(size, L'\0');
  const DWORD length = GetFullPathNameW(path.c_str(), size, full.data(), nullptr);
  if (length == 0 || length >= size) return true;
  full.resize(length);
  return windows_path_names_device(full);
}

OpenStatus open_regular_file(const std::filesystem::path& path, unsigned long flags, void*& handle) {
  if (is_windows_device_path(path)) return OpenStatus::kNotRegular;
  // Before the open, as on POSIX; a missing path falls through to the open, which reports the error.
  const DWORD attributes = GetFileAttributesW(path.c_str());
  if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE)) != 0) {
    return OpenStatus::kNotRegular;
  }
  // SECURITY_IDENTIFICATION: were it a pipe after all, its server could identify the caller but not impersonate it.
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, flags | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
  if (h == INVALID_HANDLE_VALUE) return OpenStatus::kError;
  // What was opened (the path may have been replaced since the check): a disk file that is neither a directory nor a
  // device.
  FILE_ATTRIBUTE_TAG_INFO tag{};
  const bool regular = GetFileType(h) == FILE_TYPE_DISK &&
                       GetFileInformationByHandleEx(h, FileAttributeTagInfo, &tag, sizeof tag) &&
                       (tag.FileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE)) == 0;
  if (!regular) {
    CloseHandle(h);
    return OpenStatus::kNotRegular;
  }
  handle = h;
  return OpenStatus::kOk;
}

#else

OpenStatus open_regular_file(const std::filesystem::path& path, int flags, int& fd) {
  // Nothing but a regular file is opened at all: the open of a device node can act on the device.
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) return OpenStatus::kError;
  if (!S_ISREG(st.st_mode)) return OpenStatus::kNotRegular;
  // O_NONBLOCK still guards the open against a FIFO swapped in since the stat. A regular file under a lease another
  // process holds makes a non-blocking open fail with EWOULDBLOCK while the lease is being broken, which may take the
  // 45 s of /proc/sys/fs/lease-break-time: retried a few times over about a second, never blocking.
  const auto open_nonblocking = [&] {
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK | flags);
  };
  const auto leased = [] { return errno == EWOULDBLOCK || errno == EAGAIN; };
  int f = open_nonblocking();
  for (const int ms : {50, 100, 200, 300, 350}) {
    if (f >= 0 || !leased()) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    if (::stat(path.c_str(), &st) != 0) return OpenStatus::kError;
    if (!S_ISREG(st.st_mode)) return OpenStatus::kNotRegular;
    f = open_nonblocking();
  }
  if (f < 0 && leased()) {
    errno = EWOULDBLOCK;
    return OpenStatus::kLeased;
  }
  if (f < 0) return OpenStatus::kError;
  // What was opened (the path may have been replaced since the stat).
  if (::fstat(f, &st) != 0) {
    const int code = errno;
    ::close(f);
    errno = code;
    return OpenStatus::kError;
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(f);
    return OpenStatus::kNotRegular;
  }
  const int fl = ::fcntl(f, F_GETFL);
  if (fl < 0 || ::fcntl(f, F_SETFL, fl & ~O_NONBLOCK) != 0) {
    const int code = errno;
    ::close(f);
    errno = code;
    return OpenStatus::kError;
  }
  fd = f;
  return OpenStatus::kOk;
}

#endif

}  // namespace mi::util
