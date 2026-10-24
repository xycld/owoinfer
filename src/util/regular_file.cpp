#include "util/regular_file.h"

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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mi::util {

#ifdef _WIN32

OpenStatus open_regular_file(const std::filesystem::path& path, unsigned long flags, void*& handle) {
  // Device namespace paths (\\.\pipe\x, \\.\PhysicalDrive0, \\.\COM1) name no file; their open can act on a device.
  const std::wstring& native = path.native();
  if (native.starts_with(L"\\\\.\\") || native.starts_with(L"//./")) return OpenStatus::kNotRegular;
  // Before the open, as on POSIX; a missing path falls through to the open, which reports the error.
  const DWORD attributes = GetFileAttributesW(path.c_str());
  if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE)) != 0) {
    return OpenStatus::kNotRegular;
  }
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, flags, nullptr);
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
  // process holds makes a non-blocking open fail with EWOULDBLOCK (the lease is being broken): when the path is still
  // a regular file, it is opened blocking, as any reader would wait for the lease break.
  int f = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK | flags);
  if (f < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
    if (::stat(path.c_str(), &st) != 0) return OpenStatus::kError;
    if (!S_ISREG(st.st_mode)) return OpenStatus::kNotRegular;
    f = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOCTTY | flags);
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
