#include "util/regular_file.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
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

// SECURITY_IDENTIFICATION on every open of a user path: were it a pipe after all, its server could identify the caller
// but not impersonate it.
constexpr DWORD kShareAll = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
static_assert(kUserPathSqos == (SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION));
constexpr DWORD kSqos = kUserPathSqos;

// The attributes-only open of file_info: reads no data (BACKUP_SEMANTICS: a directory opens too).
HANDLE open_attributes(const std::filesystem::path& path) {
  return CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, kShareAll, nullptr, OPEN_EXISTING,
                     FILE_FLAG_BACKUP_SEMANTICS | kSqos, nullptr);
}

// FILETIME ticks (100 ns since 1601) as nanoseconds since the Unix epoch: plain arithmetic, no time zone database
// (clock_cast through utc_clock would read one and apply leap seconds).
std::int64_t filetime_ns(FILETIME t) {
  const auto ticks = static_cast<std::int64_t>((std::uint64_t{t.dwHighDateTime} << 32) | t.dwLowDateTime);
  return (ticks - 116444736000000000LL) * 100;
}

// What an open handle is (file_info).
FileInfo handle_info(HANDLE h) {
  FileInfo info;
  if (GetFileType(h) != FILE_TYPE_DISK) {
    info.kind = FileKind::kOther;
    return info;
  }
  BY_HANDLE_FILE_INFORMATION bh{};
  if (!GetFileInformationByHandle(h, &bh)) return info;
  if ((bh.dwFileAttributes & FILE_ATTRIBUTE_DEVICE) != 0) {
    info.kind = FileKind::kOther;
    return info;
  }
  info.kind = (bh.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ? FileKind::kDirectory : FileKind::kRegular;
  if (info.regular()) info.size = (std::uint64_t{bh.nFileSizeHigh} << 32) | bh.nFileSizeLow;
  info.mtime_ns = filetime_ns(bh.ftLastWriteTime);
  FILE_ID_INFO fid{};
  if (GetFileInformationByHandleEx(h, FileIdInfo, &fid, sizeof fid)) {
    info.volume = fid.VolumeSerialNumber;
    std::memcpy(&info.id, fid.FileId.Identifier, sizeof info.id);
    std::memcpy(&info.id_high, fid.FileId.Identifier + sizeof info.id, sizeof info.id_high);
  } else {
    info.volume = bh.dwVolumeSerialNumber;
    info.id = (std::uint64_t{bh.nFileIndexHigh} << 32) | bh.nFileIndexLow;
  }
  return info;
}

// The handle's final path (GetFinalPathNameByHandleW, \\?\ form); empty when it cannot be read.
std::filesystem::path final_path(HANDLE h) {
  std::wstring buf(MAX_PATH, L'\0');
  constexpr DWORD kFlags = FILE_NAME_NORMALIZED | VOLUME_NAME_DOS;
  DWORD n = GetFinalPathNameByHandleW(h, buf.data(), static_cast<DWORD>(buf.size()), kFlags);
  if (n >= buf.size()) {  // too small: n counts the terminator
    buf.resize(n);
    n = GetFinalPathNameByHandleW(h, buf.data(), static_cast<DWORD>(buf.size()), kFlags);
  }
  if (n == 0 || n >= buf.size()) return {};
  buf.resize(n);
  return buf;
}

}  // namespace

FileInfo file_info(const std::filesystem::path& path) {
  FileInfo info;
  if (path.empty()) return info;
  if (is_windows_device_path(path)) {
    info.kind = FileKind::kOther;
    return info;
  }
  HANDLE h = open_attributes(path);
  if (h == INVALID_HANDLE_VALUE) {
    if (GetLastError() == ERROR_PIPE_BUSY) info.kind = FileKind::kOther;  // a pipe with no free instance
    return info;
  }
  info = handle_info(h);
  CloseHandle(h);
  return info;
}

std::filesystem::path resolved_path(const std::filesystem::path& path) {
  const std::filesystem::path full = path_key(path);
  if (is_windows_device_path(full)) return full;
  std::filesystem::path prefix = full, rest;
  for (;;) {
    HANDLE h = open_attributes(prefix);
    if (h != INVALID_HANDLE_VALUE) {
      const FileInfo info = handle_info(h);
      const std::filesystem::path resolved = info.regular() || info.directory() ? final_path(h) : std::filesystem::path();
      CloseHandle(h);
      if (resolved.empty()) return full;
      return rest.empty() ? resolved : resolved / rest;
    }
    const DWORD error = GetLastError();
    const std::filesystem::path parent = prefix.parent_path();
    if ((error != ERROR_FILE_NOT_FOUND && error != ERROR_PATH_NOT_FOUND) || parent == prefix || parent.empty()) return full;
    rest = rest.empty() ? prefix.filename() : prefix.filename() / rest;
    prefix = parent;
  }
}

OpenStatus open_regular_file(const std::filesystem::path& path, unsigned long flags, void*& handle) {
  if (is_windows_device_path(path)) return OpenStatus::kNotRegular;
  // The probe reads attributes only: a directory or a device is refused before the data open.
  HANDLE probe = open_attributes(path);
  if (probe == INVALID_HANDLE_VALUE) return OpenStatus::kError;
  const bool probed = handle_info(probe).regular();
  CloseHandle(probe);
  if (!probed) return OpenStatus::kNotRegular;
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, kShareAll, nullptr, OPEN_EXISTING, flags | kSqos, nullptr);
  if (h == INVALID_HANDLE_VALUE) return OpenStatus::kError;
  if (!handle_info(h).regular()) {  // replaced since the probe
    CloseHandle(h);
    return OpenStatus::kNotRegular;
  }
  handle = h;
  return OpenStatus::kOk;
}

OutputFile::OutputFile(const std::filesystem::path& path) {
  if (is_windows_device_path(path)) return;
  // FILE_READ_ATTRIBUTES for handle_info (GENERIC_WRITE does not include it).
  HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE | FILE_READ_ATTRIBUTES, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL | kSqos, nullptr);
  if (h == INVALID_HANDLE_VALUE) return;
  if (!handle_info(h).regular()) {  // a reparse point to a pipe or a device: refused before any byte is written
    CloseHandle(h);
    return;
  }
  handle_ = h;
  ok_ = true;
}

OutputFile::~OutputFile() {
  if (handle_ != nullptr) CloseHandle(static_cast<HANDLE>(handle_));
}

bool OutputFile::write(const void* data, std::size_t size) {
  const auto* p = static_cast<const char*>(data);
  while (ok_ && size > 0) {
    const DWORD want = static_cast<DWORD>(std::min<std::size_t>(size, std::size_t{1} << 30));
    DWORD done = 0;
    ok_ = WriteFile(static_cast<HANDLE>(handle_), p, want, &done, nullptr) && done == want;
    p += done;
    size -= done;
  }
  return ok_;
}

bool OutputFile::seek(std::uint64_t offset) {
  LARGE_INTEGER to;
  to.QuadPart = static_cast<LONGLONG>(offset);
  ok_ = ok_ && SetFilePointerEx(static_cast<HANDLE>(handle_), to, nullptr, FILE_BEGIN);
  return ok_;
}

bool OutputFile::close() {
  if (handle_ != nullptr) {
    ok_ = CloseHandle(static_cast<HANDLE>(handle_)) && ok_;
    handle_ = nullptr;
  }
  return ok_;
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


FileInfo file_info(const std::filesystem::path& path) {
  FileInfo info;
  struct stat st {};
  if (path.empty() || ::stat(path.c_str(), &st) != 0) return info;
  info.kind = S_ISREG(st.st_mode) ? FileKind::kRegular : S_ISDIR(st.st_mode) ? FileKind::kDirectory : FileKind::kOther;
  if (info.regular()) info.size = static_cast<std::uint64_t>(st.st_size);
  info.mtime_ns = static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1'000'000'000 + st.st_mtim.tv_nsec;
  info.volume = static_cast<std::uint64_t>(st.st_dev);
  info.id = static_cast<std::uint64_t>(st.st_ino);
  return info;
}

std::filesystem::path resolved_path(const std::filesystem::path& path) { return std::filesystem::weakly_canonical(path); }

OutputFile::OutputFile(const std::filesystem::path& path) : out_(path, std::ios::binary | std::ios::trunc) {
  ok_ = static_cast<bool>(out_);
}

OutputFile::~OutputFile() = default;

bool OutputFile::write(const void* data, std::size_t size) {
  if (ok_) ok_ = static_cast<bool>(out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size)));
  return ok_;
}

bool OutputFile::seek(std::uint64_t offset) {
  if (ok_) ok_ = static_cast<bool>(out_.seekp(static_cast<std::streamoff>(offset)));
  return ok_;
}

bool OutputFile::close() {
  if (out_.is_open()) out_.close();
  ok_ = ok_ && static_cast<bool>(out_);
  return ok_;
}

#endif

}  // namespace mi::util
