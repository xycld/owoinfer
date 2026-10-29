#include "util/regular_file.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>

#include "util/rename_replace.h"

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

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mi::util {

namespace {

// "<path>.tmp-<16 hex digits>" (OutputFile::Create::kTemporary).
std::filesystem::path temporary_name(const std::filesystem::path& path) {
  std::random_device rd;
  const auto v = (static_cast<unsigned long long>(rd()) << 32) ^ rd();
  char hex[24];
  std::snprintf(hex, sizeof hex, ".tmp-%016llx", v);
  std::filesystem::path out = path;
  out += hex;
  return out;
}

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

OutputFile::OutputFile(const std::filesystem::path& path, Create create) {
  // FILE_READ_ATTRIBUTES for handle_info (GENERIC_WRITE does not include it); DELETE for publish and discard.
  const DWORD access = GENERIC_WRITE | FILE_READ_ATTRIBUTES | (create == Create::kTemporary ? DELETE : 0);
  HANDLE h = INVALID_HANDLE_VALUE;
  if (create == Create::kTemporary) {
    for (int attempt = 0; attempt < 16; ++attempt) {
      path_ = temporary_name(path);
      if (is_windows_device_path(path_)) return;
      h = CreateFileW(path_.c_str(), access, FILE_SHARE_READ, nullptr, CREATE_NEW,
                      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT | kSqos, nullptr);
      if (h != INVALID_HANDLE_VALUE || (GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS)) {
        break;
      }
    }
  } else {
    path_ = path;
    if (is_windows_device_path(path)) return;
    h = CreateFileW(path.c_str(), access, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | kSqos, nullptr);
  }
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

bool OutputFile::sync() {
  ok_ = ok_ && FlushFileBuffers(static_cast<HANDLE>(handle_));
  return ok_;
}

bool OutputFile::publish(const std::filesystem::path& target) {
  if (!ok_ || handle_ == nullptr) return false;
  // FILE_RENAME_INFO: the flags (a BOOLEAN ReplaceIfExists before Windows 10 1607's DWORD Flags), the root directory
  // (none), the name's length in bytes and the name: the target's file name alone renames within the same folder.
  constexpr DWORD kReplaceIfExists = 0x1, kPosixSemantics = 0x2;
  constexpr auto kFileRenameInfoEx = static_cast<FILE_INFO_BY_HANDLE_CLASS>(22);
  const std::wstring name = target.filename().native();
  std::vector<std::uint64_t> buf((offsetof(FILE_RENAME_INFO, FileName) + (name.size() + 1) * sizeof(wchar_t) + 7) / 8);
  auto* info = reinterpret_cast<FILE_RENAME_INFO*>(buf.data());
  info->RootDirectory = nullptr;
  info->FileNameLength = static_cast<DWORD>(name.size() * sizeof(wchar_t));
  std::memcpy(info->FileName, name.c_str(), (name.size() + 1) * sizeof(wchar_t));
  const auto size = static_cast<DWORD>(buf.size() * sizeof(std::uint64_t));
  const auto set_name = [&](FILE_INFO_BY_HANDLE_CLASS cls, DWORD flags) {
    std::memcpy(buf.data(), &flags, sizeof flags);
    return SetFileInformationByHandle(static_cast<HANDLE>(handle_), cls, info, size) != 0;
  };
  bool renamed = false;
  for (int attempt = 0; !renamed && attempt < 200; ++attempt) {
    if (attempt > 0) Sleep(1);
    renamed = set_name(kFileRenameInfoEx, kReplaceIfExists | kPosixSemantics);
    const DWORD error = renamed ? 0 : GetLastError();
    if (!renamed && (error == ERROR_INVALID_PARAMETER || error == ERROR_NOT_SUPPORTED)) {
      renamed = set_name(FileRenameInfo, kReplaceIfExists);  // no FileRenameInfoEx on this file system
    }
    if (!renamed && error != ERROR_ACCESS_DENIED && error != ERROR_SHARING_VIOLATION) break;
  }
  if (!renamed) return false;
  path_ = target;
  return close();
}

void OutputFile::discard() {
  if (handle_ != nullptr) {
    FILE_DISPOSITION_INFO del{TRUE};
    SetFileInformationByHandle(static_cast<HANDLE>(handle_), FileDispositionInfo, &del, sizeof del);
  }
  close();
}

bool list_directory(const std::filesystem::path& dir, std::vector<DirEntry>& out) {
  out.clear();
  if (dir.empty() || is_windows_device_path(dir)) return false;
  HANDLE h = CreateFileW(dir.c_str(), FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES, kShareAll, nullptr, OPEN_EXISTING,
                         FILE_FLAG_BACKUP_SEMANTICS | kSqos, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  bool listed = handle_info(h).directory();
  std::vector<std::uint64_t> buf(std::size_t{64} << 10 >> 3);  // 8-byte aligned for the records
  for (bool first = true; listed; first = false) {
    const FILE_INFO_BY_HANDLE_CLASS cls = first ? FileIdBothDirectoryRestartInfo : FileIdBothDirectoryInfo;
    if (!GetFileInformationByHandleEx(h, cls, buf.data(), static_cast<DWORD>(buf.size() * sizeof(std::uint64_t)))) {
      listed = GetLastError() == ERROR_NO_MORE_FILES;
      break;
    }
    const auto* p = reinterpret_cast<const std::byte*>(buf.data());
    for (;;) {
      const auto* e = reinterpret_cast<const FILE_ID_BOTH_DIR_INFO*>(p);
      const std::wstring name(e->FileName, e->FileNameLength / sizeof(wchar_t));
      if (name != L"." && name != L"..") {
        DirEntry d;
        d.name = name;
        d.link = (e->FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
        d.info.kind = d.link || (e->FileAttributes & FILE_ATTRIBUTE_DEVICE) != 0 ? FileKind::kOther
                      : (e->FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0  ? FileKind::kDirectory
                                                                              : FileKind::kRegular;
        if (d.info.regular()) d.info.size = static_cast<std::uint64_t>(e->EndOfFile.QuadPart);
        d.info.mtime_ns = (e->LastWriteTime.QuadPart - 116444736000000000LL) * 100;
        d.info.id = static_cast<std::uint64_t>(e->FileId.QuadPart);
        out.push_back(std::move(d));
      }
      if (e->NextEntryOffset == 0) break;
      p += e->NextEntryOffset;
    }
  }
  CloseHandle(h);
  return listed;
}

namespace {

// `path` opened without following a final reparse point, if it is a plain regular file in the folder `dir`.
HANDLE open_file_in(const std::filesystem::path& dir, const std::filesystem::path& path, DWORD access) {
  if (is_windows_device_path(path)) return INVALID_HANDLE_VALUE;
  HANDLE h = CreateFileW(path.c_str(), access | FILE_READ_ATTRIBUTES, kShareAll, nullptr, OPEN_EXISTING,
                         FILE_FLAG_OPEN_REPARSE_POINT | kSqos, nullptr);
  if (h == INVALID_HANDLE_VALUE) return h;
  FILE_ATTRIBUTE_TAG_INFO tag{};
  const bool plain = handle_info(h).regular() &&
                     GetFileInformationByHandleEx(h, FileAttributeTagInfo, &tag, sizeof tag) &&
                     (tag.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0 && final_path(h).parent_path() == dir;
  if (!plain) {
    CloseHandle(h);
    return INVALID_HANDLE_VALUE;
  }
  return h;
}

}  // namespace

bool touch_file_in(const std::filesystem::path& dir, const std::filesystem::path& path) {
  HANDLE h = open_file_in(dir, path, FILE_WRITE_ATTRIBUTES);
  if (h == INVALID_HANDLE_VALUE) return false;
  FILETIME now;
  GetSystemTimeAsFileTime(&now);
  const bool ok = SetFileTime(h, nullptr, nullptr, &now) != 0;
  CloseHandle(h);
  return ok;
}

bool remove_file_in(const std::filesystem::path& dir, const std::filesystem::path& path) {
  HANDLE h = open_file_in(dir, path, DELETE);
  if (h == INVALID_HANDLE_VALUE) return false;
  FILE_DISPOSITION_INFO del{TRUE};
  const bool ok = SetFileInformationByHandle(h, FileDispositionInfo, &del, sizeof del) != 0;
  CloseHandle(h);
  return ok;
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

std::filesystem::path resolved_path(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::path resolved = std::filesystem::weakly_canonical(path, ec);
  return ec ? path.lexically_normal() : resolved;
}

OutputFile::OutputFile(const std::filesystem::path& path, Create create) {
  if (create == Create::kTemporary) {
    for (int attempt = 0; attempt < 16; ++attempt) {
      path_ = temporary_name(path);
      fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NOCTTY, 0666);
      if (fd_ >= 0 || errno != EEXIST) break;
    }
  } else {
    path_ = path;
    struct stat st {};
    if (::stat(path.c_str(), &st) == 0 && !S_ISREG(st.st_mode)) return;  // a device or FIFO is never opened
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOCTTY | O_NONBLOCK, 0666);
  }
  if (fd_ < 0) return;
  struct stat st {};
  const int fl = ::fcntl(fd_, F_GETFL);
  if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || fl < 0 || ::fcntl(fd_, F_SETFL, fl & ~O_NONBLOCK) != 0) {
    ::close(fd_);
    fd_ = -1;
    return;
  }
  ok_ = true;
  created_ = true;
}

OutputFile::~OutputFile() {
  if (fd_ >= 0) ::close(fd_);
}

bool OutputFile::write(const void* data, std::size_t size) {
  const auto* p = static_cast<const char*>(data);
  while (ok_ && size > 0) {
    const ssize_t n = ::write(fd_, p, std::min<std::size_t>(size, std::size_t{1} << 30));
    if (n < 0 && errno == EINTR) continue;
    ok_ = n > 0;
    if (ok_) {
      p += n;
      size -= static_cast<std::size_t>(n);
    }
  }
  return ok_;
}

bool OutputFile::seek(std::uint64_t offset) {
  ok_ = ok_ && ::lseek(fd_, static_cast<off_t>(offset), SEEK_SET) == static_cast<off_t>(offset);
  return ok_;
}

bool OutputFile::sync() {
  ok_ = ok_ && ::fsync(fd_) == 0;
  return ok_;
}

bool OutputFile::publish(const std::filesystem::path& target) {
  if (!close()) return false;
  std::error_code ec;
  rename_replace(path_, target, ec);
  if (ec) return false;
  path_ = target;
  created_ = false;
  return true;
}

void OutputFile::discard() {
  close();
  if (created_) ::unlink(path_.c_str());
  created_ = false;
}

bool list_directory(const std::filesystem::path& dir, std::vector<DirEntry>& out) {
  out.clear();
  DIR* d = dir.empty() ? nullptr : ::opendir(dir.c_str());
  if (d == nullptr) return false;
  errno = 0;
  while (const dirent* e = ::readdir(d)) {
    const std::string_view name = e->d_name;
    if (name == "." || name == "..") continue;
    DirEntry entry;
    entry.name = std::string(name);
    struct stat st {};
    if (::fstatat(::dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW) == 0) {
      entry.link = S_ISLNK(st.st_mode);
      entry.info.kind = S_ISREG(st.st_mode) ? FileKind::kRegular : S_ISDIR(st.st_mode) ? FileKind::kDirectory : FileKind::kOther;
      if (entry.info.regular()) entry.info.size = static_cast<std::uint64_t>(st.st_size);
      entry.info.mtime_ns = static_cast<std::int64_t>(st.st_mtim.tv_sec) * 1'000'000'000 + st.st_mtim.tv_nsec;
      entry.info.id = static_cast<std::uint64_t>(st.st_ino);
    }
    out.push_back(std::move(entry));
    errno = 0;
  }
  const bool listed = errno == 0;
  ::closedir(d);
  return listed;
}

bool touch_file_in(const std::filesystem::path&, const std::filesystem::path& path) {
  return ::utimensat(AT_FDCWD, path.c_str(), nullptr, 0) == 0;
}

bool remove_file_in(const std::filesystem::path&, const std::filesystem::path& path) {
  return ::unlink(path.c_str()) == 0;
}

bool OutputFile::close() {
  if (fd_ >= 0) {
    ok_ = ::close(fd_) == 0 && ok_;
    fd_ = -1;
  }
  return ok_;
}

#endif

}  // namespace mi::util
