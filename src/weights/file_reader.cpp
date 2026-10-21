#include "weights/file_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <system_error>
#include <utility>

#include "util/path.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mi::weights {

namespace {

// Single syscalls are capped below 2 GiB on Linux and by DWORD on Windows; chunk well under both.
constexpr std::size_t kMaxIoChunk = std::size_t{1} << 30;
// Unaligned direct reads bounce through a buffer of this size.
constexpr std::size_t kBounceBytes = std::size_t{16} << 20;

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what, int code) {
  throw IoError(util::path_utf8(path) + ": " + what + ": " + std::system_category().message(code));
}

bool is_aligned(std::uint64_t v) { return v % kDirectAlign == 0; }

}  // namespace

bool direct_short_read_continues(std::uint64_t pos, std::size_t remaining, std::uint64_t file_size) {
  return pos < file_size && is_aligned(pos) && is_aligned(remaining);
}

AlignedBuffer::AlignedBuffer(std::size_t size)
    : p_(static_cast<std::byte*>(::operator new(size == 0 ? 1 : size, std::align_val_t{kDirectAlign}))),
      size_(size) {}

AlignedRange align_range(std::uint64_t offset, std::size_t size) {
  std::uint64_t begin = offset & ~std::uint64_t{kDirectAlign - 1};
  std::uint64_t end = (offset + size + kDirectAlign - 1) & ~std::uint64_t{kDirectAlign - 1};
  return {begin, static_cast<std::size_t>(end - begin), static_cast<std::size_t>(offset - begin)};
}

// ---- FileReader -----------------------------------------------------------------------------

#ifdef _WIN32

namespace {
HANDLE open_win(const std::filesystem::path& path, bool direct) {
  DWORD flags = FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0);
  // FILE_SHARE_DELETE lets the app replace or delete model files (an update) while a reader is open.
  return CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, flags,
                     nullptr);
}
}  // namespace

FileReader::FileReader(const std::filesystem::path& path, bool direct) : path_(path) {
  HANDLE h = open_win(path, direct);
  direct_ = direct;
  if (h == INVALID_HANDLE_VALUE && direct) {
    h = open_win(path, false);
    direct_ = false;
  }
  if (h == INVALID_HANDLE_VALUE) fail(path_, "cannot open", static_cast<int>(GetLastError()));
  handle_ = h;
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(h, &sz)) {
    int code = static_cast<int>(GetLastError());
    close();
    fail(path_, "cannot stat", code);
  }
  size_ = static_cast<std::uint64_t>(sz.QuadPart);
}

void FileReader::close() noexcept {
  if (handle_ != nullptr) CloseHandle(handle_);
  handle_ = nullptr;
}

std::size_t FileReader::pread_raw(std::uint64_t offset, void* dst, std::size_t size) const {
  auto* out = static_cast<char*>(dst);
  std::size_t done = 0;
  // A private event per call keeps concurrent reads on one handle from waiting on each other.
  HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  if (ev == nullptr) fail(path_, "CreateEvent failed", static_cast<int>(GetLastError()));
  while (done < size) {
    DWORD want = static_cast<DWORD>(std::min(size - done, kMaxIoChunk));
    OVERLAPPED ov{};
    std::uint64_t pos = offset + done;
    ov.Offset = static_cast<DWORD>(pos & 0xFFFFFFFFu);
    ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
    ov.hEvent = ev;
    ResetEvent(ev);
    DWORD got = 0;
    BOOL ok = ReadFile(handle_, out + done, want, nullptr, &ov);
    if (ok || GetLastError() == ERROR_IO_PENDING) {
      ok = GetOverlappedResult(handle_, &ov, &got, TRUE);
    }
    if (!ok) {
      DWORD err = GetLastError();
      if (err == ERROR_HANDLE_EOF) break;
      CloseHandle(ev);
      fail(path_, "read failed", static_cast<int>(err));
    }
    if (got == 0) break;
    done += got;
    // In direct mode a short read at EOF ends the loop (NO_BUFFERING handles reject the unaligned
    // offset a retry would use); a mid-file one retries only when the retry is aligned.
    if (direct_ && got < want && !direct_short_read_continues(offset + done, size - done, size_)) break;
  }
  CloseHandle(ev);
  return done;
}

#else

FileReader::FileReader(const std::filesystem::path& path, bool direct) : path_(path) {
  int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_DIRECT
  if (direct) {
    fd_ = ::open(path.c_str(), flags | O_DIRECT);
    direct_ = fd_ >= 0;
  }
#endif
  if (fd_ < 0) fd_ = ::open(path.c_str(), flags);
  if (fd_ < 0) fail(path_, "cannot open", errno);
  struct stat st;
  if (::fstat(fd_, &st) != 0) {
    int code = errno;
    close();
    fail(path_, "cannot stat", code);
  }
  size_ = static_cast<std::uint64_t>(st.st_size);
}

void FileReader::close() noexcept {
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

std::size_t FileReader::pread_raw(std::uint64_t offset, void* dst, std::size_t size) const {
  auto* out = static_cast<char*>(dst);
  std::size_t done = 0;
  while (done < size) {
    const std::size_t want = std::min(size - done, kMaxIoChunk);
    ssize_t n = ::pread(fd_, out + done, want, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) continue;
      fail(path_, "read failed", errno);
    }
    if (n == 0) break;
    done += static_cast<std::size_t>(n);
    // In direct mode a short read at EOF ends the loop (the next offset would be unaligned: EINVAL
    // on file systems that check before EOF); a mid-file one retries only when the retry is aligned.
    if (direct_ && static_cast<std::size_t>(n) < want && !direct_short_read_continues(offset + done, size - done, size_)) break;
  }
  return done;
}

#endif

FileReader::~FileReader() { close(); }

FileReader::FileReader(FileReader&& other) noexcept
    : path_(std::move(other.path_)), size_(other.size_), direct_(other.direct_) {
#ifdef _WIN32
  handle_ = std::exchange(other.handle_, nullptr);
#else
  fd_ = std::exchange(other.fd_, -1);
#endif
}

FileReader& FileReader::operator=(FileReader&& other) noexcept {
  if (this != &other) {
    close();
    path_ = std::move(other.path_);
    size_ = other.size_;
    direct_ = other.direct_;
#ifdef _WIN32
    handle_ = std::exchange(other.handle_, nullptr);
#else
    fd_ = std::exchange(other.fd_, -1);
#endif
  }
  return *this;
}

AlignedRange FileReader::read_aligned(std::uint64_t offset, std::size_t size, std::byte* buf,
                                      std::size_t buf_size) const {
  if (offset > size_ || size > size_ - offset) {
    throw IoError(util::path_utf8(path_) + ": read of " + std::to_string(size) + " bytes at offset " +
                  std::to_string(offset) + " is past end of file (" + std::to_string(size_) + " bytes)");
  }
  AlignedRange r = align_range(offset, size);
  if (buf_size < r.length || !is_aligned(reinterpret_cast<std::uintptr_t>(buf))) {
    throw IoError(util::path_utf8(path_) + ": read_aligned needs a " + std::to_string(kDirectAlign) +
                  "-aligned buffer of at least " + std::to_string(r.length) + " bytes");
  }
  // Buffered files gain nothing from over-reading, so fetch only the wanted bytes.
  std::size_t want = direct_ ? r.length : size;
  std::byte* dst = direct_ ? buf : buf + r.skip;
  std::uint64_t at = direct_ ? r.offset : offset;
  std::size_t got = pread_raw(at, dst, want);
  // Direct reads of the last block legitimately come back short at EOF.
  if (direct_ ? got < r.skip + size : got < size) {
    throw IoError(util::path_utf8(path_) + ": short read at offset " + std::to_string(offset));
  }
  return r;
}

void FileReader::read_at(std::uint64_t offset, void* dst, std::size_t size) const {
  if (offset > size_ || size > size_ - offset) {
    throw IoError(util::path_utf8(path_) + ": read of " + std::to_string(size) + " bytes at offset " +
                  std::to_string(offset) + " is past end of file (" + std::to_string(size_) + " bytes)");
  }
  if (!direct_) {
    if (pread_raw(offset, dst, size) != size) throw IoError(util::path_utf8(path_) + ": short read at offset " + std::to_string(offset));
    return;
  }
  if (size == 0) return;
  if (is_aligned(offset) && is_aligned(size) && is_aligned(reinterpret_cast<std::uintptr_t>(dst))) {
    if (pread_raw(offset, dst, size) != size) throw IoError(util::path_utf8(path_) + ": short read at offset " + std::to_string(offset));
    return;
  }
  // Unaligned request: over-read aligned pieces through a bounce buffer and copy out. The widened
  // range of a piece is at most piece + 2 * kDirectAlign, which is what the piece cap leaves room for.
  AlignedBuffer bounce(std::min(kBounceBytes, align_range(offset, size).length));
  auto* out = static_cast<std::byte*>(dst);
  std::size_t done = 0;
  while (done < size) {
    std::size_t piece = std::min(size - done, kBounceBytes - 2 * kDirectAlign);
    AlignedRange r = read_aligned(offset + done, piece, bounce.data(), bounce.size());
    std::memcpy(out + done, bounce.data() + r.skip, piece);
    done += piece;
  }
}

// ---- MappedFile -----------------------------------------------------------------------------

#ifdef _WIN32

MappedFile::MappedFile(const std::filesystem::path& path) {
  HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (f == INVALID_HANDLE_VALUE) fail(path, "cannot open", static_cast<int>(GetLastError()));
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(f, &sz)) {
    int code = static_cast<int>(GetLastError());
    CloseHandle(f);
    fail(path, "cannot stat", code);
  }
  size_ = static_cast<std::size_t>(sz.QuadPart);
  if (size_ == 0) {
    CloseHandle(f);
    return;
  }
  HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
  int code = static_cast<int>(GetLastError());
  CloseHandle(f);
  if (m == nullptr) fail(path, "CreateFileMapping failed", code);
  void* view = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  code = static_cast<int>(GetLastError());
  CloseHandle(m);  // the view keeps the mapping alive
  if (view == nullptr) fail(path, "MapViewOfFile failed", code);
  data_ = static_cast<const std::byte*>(view);
}

void MappedFile::unmap() noexcept {
  if (data_ != nullptr) UnmapViewOfFile(data_);
  data_ = nullptr;
  size_ = 0;
}

#else

MappedFile::MappedFile(const std::filesystem::path& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) fail(path, "cannot open", errno);
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    int code = errno;
    ::close(fd);
    fail(path, "cannot stat", code);
  }
  size_ = static_cast<std::size_t>(st.st_size);
  if (size_ != 0) {
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    int code = errno;
    ::close(fd);
    if (p == MAP_FAILED) fail(path, "mmap failed", code);
    data_ = static_cast<const std::byte*>(p);
  } else {
    ::close(fd);
  }
}

void MappedFile::unmap() noexcept {
  if (data_ != nullptr) ::munmap(const_cast<std::byte*>(data_), size_);
  data_ = nullptr;
  size_ = 0;
}

#endif

MappedFile::~MappedFile() { unmap(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    unmap();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

}  // namespace mi::weights
