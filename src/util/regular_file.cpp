#include "util/regular_file.h"

#include <cstddef>
#include <stdexcept>
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

// ASCII case folding, plus the two letters whose NT upcase is ASCII (dotless i U+0131 -> I, long s U+017F -> S), so a
// share name cannot pass for another by them.
bool iequals(std::wstring_view a, std::wstring_view b) {
  if (a.size() != b.size()) return false;
  const auto lower = [](wchar_t c) {
    if (c == L'\x0131') return L'i';
    if (c == L'\x017F') return L's';
    return c >= L'A' && c <= L'Z' ? static_cast<wchar_t>(c - L'A' + L'a') : c;
  };
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

// A UNC path's share, from `pos` (just after the server): the named pipes' (`pipe`, and `IPC$` that carries them), the
// mailslots', or missing. Trailing dots and spaces are dropped first, as Win32 path normalization drops them.
bool device_share(std::wstring_view s, std::size_t pos) {
  std::wstring_view share = component(s, pos);
  while (!share.empty() && (share.back() == L'.' || share.back() == L' ')) share.remove_suffix(1);
  return share.empty() || iequals(share, L"pipe") || iequals(share, L"mailslot") || iequals(share, L"IPC$");
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
    const std::wstring_view server = component(p, pos);  // UNC\server\share\...
    return server.empty() || server[0] == L';' || device_share(p, pos);
  }
  if (p.size() >= 2 && is_sep(p[0]) && is_sep(p[1])) {  // \\server\share\...
    std::size_t pos = 2;
    const std::wstring_view server = component(p, pos);
    // \\. or \\? alone; a provider-qualified \\;Provider\... names a redirector, never a model path
    if (server == L"." || server == L"?" || (!server.empty() && server[0] == L';')) return true;
    return device_share(p, pos);
  }
  return false;
}

void check_user_path(const std::filesystem::path& path, std::string_view what) {
#ifdef _WIN32
  if (!path.empty() && is_windows_device_path(path)) {  // an unset path is not a path (GetFullPathNameW fails on it)
    const std::u8string text = path.u8string();
    throw std::invalid_argument(std::string(what) + ": " + kNotRegularFile + ": " +
                                std::string(text.begin(), text.end()));
  }
#else
  static_cast<void>(path);
  static_cast<void>(what);
#endif
}

std::filesystem::path path_key(const std::filesystem::path& path) {
#ifdef _WIN32
  return std::filesystem::absolute(path).lexically_normal();
#else
  return std::filesystem::weakly_canonical(path);
#endif
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

namespace {

// A disk file that is neither a directory nor a device.
bool regular_handle(HANDLE h) {
  FILE_ATTRIBUTE_TAG_INFO tag{};
  return GetFileType(h) == FILE_TYPE_DISK && GetFileInformationByHandleEx(h, FileAttributeTagInfo, &tag, sizeof tag) &&
         (tag.FileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE)) == 0;
}

}  // namespace

OpenStatus open_regular_file(const std::filesystem::path& path, unsigned long flags, void*& handle) {
  if (is_windows_device_path(path)) return OpenStatus::kNotRegular;
  // SECURITY_IDENTIFICATION on every open: were it a pipe after all, its server could identify the caller but not
  // impersonate it. The probe reads attributes only (BACKUP_SEMANTICS: a directory opens too, to be refused here).
  const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  const DWORD sqos = SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION;
  HANDLE probe = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, share, nullptr, OPEN_EXISTING,
                             FILE_FLAG_BACKUP_SEMANTICS | sqos, nullptr);
  if (probe == INVALID_HANDLE_VALUE) return OpenStatus::kError;
  const bool probed = regular_handle(probe);
  CloseHandle(probe);
  if (!probed) return OpenStatus::kNotRegular;
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING, flags | sqos, nullptr);
  if (h == INVALID_HANDLE_VALUE) return OpenStatus::kError;
  if (!regular_handle(h)) {  // replaced since the probe
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
