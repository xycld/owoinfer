#include "weights/file_reader.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "util/path.h"
#include "util/regular_file.h"
#include "weights/test_seams.h"

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
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
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
// FILE_SHARE_DELETE lets the app replace or delete model files (an update) while a reader is open, and with
// FILE_SHARE_WRITE a reader never blocks a writer (util::open_regular_file opens with both). The trade-off: a program
// writing into a weight file in place while it is read can hand the reader torn bytes (no lock stops it); the engine
// and its tools never write weight files in place (they write a temporary file and rename it over the old one), and a
// torn read fails the format checks or the quality identity rather than passing silently in most cases.
util::OpenStatus open_win(const std::filesystem::path& path, bool direct, HANDLE& h) {
  void* handle = nullptr;
  const DWORD flags = FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0);
  const util::OpenStatus status = util::open_regular_file(path, flags, handle);
  h = status == util::OpenStatus::kOk ? static_cast<HANDLE>(handle) : INVALID_HANDLE_VALUE;
  return status;
}
}  // namespace

FileReader::FileReader(const std::filesystem::path& path, bool direct) : path_(path) {
  HANDLE h = INVALID_HANDLE_VALUE;
  util::OpenStatus status = open_win(path, direct, h);
  direct_ = direct;
  if (status == util::OpenStatus::kError && direct) {
    status = open_win(path, false, h);
    direct_ = false;
  }
  if (status == util::OpenStatus::kNotRegular) throw IoError(util::path_utf8(path_) + ": " + util::kNotRegularFile);
  if (status != util::OpenStatus::kOk) fail(path_, "cannot open", static_cast<int>(GetLastError()));
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
  if (memory_) return copy_from_memory(offset, dst, size);
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

double page_cache_fraction(const std::filesystem::path&) { return -1.0; }

DirectIoSupport direct_io_support(const std::filesystem::path& path) {
  const std::string name = util::path_utf8(path.filename());
  const DWORD attributes = GetFileAttributesW(path.c_str());
  if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_COMPRESSED) != 0) {
    return {false, name + " is NTFS-compressed: unbuffered reads gain nothing"};
  }
  HANDLE h = INVALID_HANDLE_VALUE;
  const util::OpenStatus status = open_win(path, true, h);
  if (status == util::OpenStatus::kNotRegular) return {false, name + " is " + util::kNotRegularFile};
  if (status != util::OpenStatus::kOk) {
    const int code = static_cast<int>(GetLastError());
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND) return {false, name + " does not exist"};
    return {false, name + ": the file system refuses unbuffered reads (" + std::system_category().message(code) + ")"};
  }
  CloseHandle(h);
  return {true, {}};
}

#else

namespace {

constexpr long kBtrfsMagic = static_cast<long>(0x9123683e);
// Network file systems: their O_DIRECT reads bypass the client cache but still cross the network per call.
constexpr long kNfsMagic = 0x6969;
constexpr long kSmbMagic = 0x517b;
constexpr long kCifsMagic = static_cast<long>(0xff534d42);
constexpr long kSmb2Magic = static_cast<long>(0xfe534d42);
// Pass-through file systems (FUSE, 9p, virtiofs): O_DIRECT reaches a userspace or host server per call, no faster.
constexpr long kFuseMagic = 0x65735546;
constexpr long kV9fsMagic = 0x01021997;
constexpr long kVirtiofsMagic = 0x6a656a63;

// /proc/self/mountinfo escapes spaces, tabs, newlines and backslashes in paths as \ooo octal.
std::string unescape_mount(const std::string& s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 3 < s.size()) {
      const std::string oct = s.substr(i + 1, 3);
      if (oct.find_first_not_of("01234567") == std::string::npos) {
        out += static_cast<char>(std::stoi(oct, nullptr, 8));
        i += 3;
        continue;
      }
    }
    out += s[i];
  }
  return out;
}

// The per-mount and super-block options of the mount holding `file` (the longest mount point that contains its
// canonical path), "" when unknown.
std::string mount_options(const std::filesystem::path& file) {
  std::error_code ec;
  const std::string target = std::filesystem::weakly_canonical(file, ec).string();
  if (ec) return {};
  std::ifstream in("/proc/self/mountinfo");
  std::string line, best_options;
  std::size_t best = 0;
  while (std::getline(in, line)) {
    // id parent major:minor root mount_point mount_options [optional fields] - fstype source super_options
    std::istringstream fields(line);
    std::string id, parent, dev, root, point, options, field;
    if (!(fields >> id >> parent >> dev >> root >> point >> options)) continue;
    while (fields >> field && field != "-") {
    }
    std::string fstype, source, super;
    fields >> fstype >> source >> super;
    point = unescape_mount(point);
    const bool contains = point == "/" || target == point ||
                          (target.size() > point.size() && target.compare(0, point.size(), point) == 0 &&
                           target[point.size()] == '/');
    if (contains && point.size() >= best) {
      best = point.size();
      best_options = options + "," + super;
    }
  }
  return best_options;
}

// Fraction of the file's bytes in compressed ("encoded") extents, from FIEMAP (after a sync, so delayed allocations are
// mapped); negative when FIEMAP is unavailable. The scan stops once the answer to "more than kMaxCompressedFraction?"
// is settled either way, so the returned fraction is exact only near the threshold.
double encoded_fraction(int fd, std::uint64_t size) {
  constexpr std::uint32_t kBatch = 512;
  std::vector<std::uint64_t> buf((sizeof(fiemap) + kBatch * sizeof(fiemap_extent)) / sizeof(std::uint64_t) + 1);
  auto* fm = reinterpret_cast<fiemap*>(buf.data());
  std::uint64_t start = 0, encoded = 0, mapped = 0;
  while (start < size) {
    std::fill(buf.begin(), buf.end(), 0);
    fm->fm_start = start;
    fm->fm_length = size - start;
    fm->fm_flags = start == 0 ? FIEMAP_FLAG_SYNC : 0;
    fm->fm_extent_count = kBatch;
    if (::ioctl(fd, FS_IOC_FIEMAP, fm) != 0) return -1.0;
    if (fm->fm_mapped_extents == 0) break;
    const fiemap_extent* last = nullptr;
    for (std::uint32_t i = 0; i < fm->fm_mapped_extents; ++i) {
      last = &fm->fm_extents[i];
      mapped += last->fe_length;
      if ((last->fe_flags & FIEMAP_EXTENT_ENCODED) != 0) encoded += last->fe_length;
    }
    if ((last->fe_flags & FIEMAP_EXTENT_LAST) != 0 || last->fe_logical + last->fe_length <= start) break;
    start = last->fe_logical + last->fe_length;
    // Settled: already over the threshold, or under it even if every byte not scanned yet were compressed.
    const double limit = kMaxCompressedFraction * static_cast<double>(size);
    if (static_cast<double>(encoded) > limit) return static_cast<double>(encoded) / static_cast<double>(size);
    if (static_cast<double>(encoded + (size > start ? size - start : 0)) <= limit) break;
  }
  return mapped > 0 ? static_cast<double>(encoded) / static_cast<double>(mapped) : 0.0;
}

// cachestat(2) (Linux 6.5): its number is the same on every architecture (the unified syscall table).
struct CachestatRange {
  std::uint64_t off, len;
};
struct Cachestat {
  std::uint64_t nr_cache, nr_dirty, nr_writeback, nr_evicted, nr_recently_evicted;
};
constexpr long kSysCachestat = 451;

}  // namespace

double page_cache_fraction(const std::filesystem::path& path) {
  const testing::ResidencyQuery query = testing::residency_query();
  if (query == testing::ResidencyQuery::kNone) return -1.0;
  int fd = -1;
  if (util::open_regular_file(path, 0, fd) != util::OpenStatus::kOk) return -1.0;
  struct stat st {};
  double fraction = -1.0;
  if (::fstat(fd, &st) == 0 && st.st_size > 0) {
    const long page = ::sysconf(_SC_PAGESIZE);
    const auto size = static_cast<std::uint64_t>(st.st_size);
    const std::uint64_t pages = (size + static_cast<std::uint64_t>(page) - 1) / static_cast<std::uint64_t>(page);
    CachestatRange range{0, 0};  // len 0: to the end of the file
    Cachestat cs{};
    const bool cachestat = query == testing::ResidencyQuery::kAuto && ::syscall(kSysCachestat, fd, &range, &cs, 0) == 0;
    // The kernel answers mincore truthfully only for files the caller could write (can_do_mincore: the owner, or write
    // permission); for the others it reports every page resident, so they stay unknown. cachestat refuses those with
    // EPERM; an EPERM on a file mincore may answer (a seccomp filter on the new syscall, most likely) falls back to it.
    const bool may_mincore = st.st_uid == ::geteuid() || ::faccessat(AT_FDCWD, path.c_str(), W_OK, AT_EACCESS) == 0;
    if (cachestat) {
      fraction = std::min(1.0, static_cast<double>(cs.nr_cache) / static_cast<double>(pages));
    } else if (may_mincore) {
      // mincore on a sample of pages of a read-only mapping (mapping reads nothing; mincore faults nothing in).
      void* map = ::mmap(nullptr, static_cast<std::size_t>(size), PROT_READ, MAP_SHARED, fd, 0);
      if (map != MAP_FAILED) {
        int resident = 0, sampled = 0;
        for (int i = 0; i < kResidencySamples; ++i) {
          const std::uint64_t p = pages * static_cast<std::uint64_t>(i) / kResidencySamples;
          unsigned char vec = 0;
          char* at = static_cast<char*>(map) + p * static_cast<std::uint64_t>(page);
          if (::mincore(at, static_cast<std::size_t>(page), &vec) == 0) {
            resident += vec & 1;
            ++sampled;
          }
        }
        ::munmap(map, static_cast<std::size_t>(size));
        if (sampled > 0) fraction = static_cast<double>(resident) / sampled;
      }
    }
  }
  ::close(fd);
  return fraction;
}

DirectIoSupport direct_io_support(const std::filesystem::path& path) {
  const std::string name = util::path_utf8(path.filename());
#ifndef O_DIRECT
  return {false, "this platform has no O_DIRECT"};
#else
  int fd = -1;
  const util::OpenStatus status = util::open_regular_file(path, O_DIRECT, fd);
  if (status == util::OpenStatus::kNotRegular) return {false, name + " is " + util::kNotRegularFile};
  if (status != util::OpenStatus::kOk) {
    if (errno == ENOENT) return {false, name + " does not exist"};
    return {false, name + ": the file system refuses O_DIRECT (" + std::system_category().message(errno) + ")"};
  }
  DirectIoSupport out{true, {}};
  struct statfs fsinfo {};
  struct stat st {};
  if (::fstatfs(fd, &fsinfo) == 0 && ::fstat(fd, &st) == 0) {
    const long type = static_cast<long>(fsinfo.f_type);
    if (static_cast<std::uint32_t>(type) == kTmpfsMagic || static_cast<std::uint32_t>(type) == kRamfsMagic) {
      out = {false, name + " is on a RAM file system (tmpfs / ramfs): direct I/O gains nothing"};
    } else if (type == kNfsMagic || type == kSmbMagic || type == kCifsMagic || type == kSmb2Magic) {
      out = {false, name + " is on a network file system (NFS / SMB): buffered reads keep its client cache"};
    } else if (type == kFuseMagic || type == kV9fsMagic || type == kVirtiofsMagic) {
      out = {false, name + " is on a pass-through file system (FUSE / 9p / virtiofs): direct I/O gains nothing"};
    } else if (type == kBtrfsMagic) {
      const double fraction = encoded_fraction(fd, static_cast<std::uint64_t>(st.st_size));
      if (fraction > kMaxCompressedFraction) {
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%.0f%%", 100.0 * fraction);
        out = {false,
               name + ": " + pct + " of the file is in btrfs compressed extents, which O_DIRECT reads through the page cache"};
      } else if (fraction < 0 && mount_options(path).find("compress") != std::string::npos) {
        out = {false, name + " is on a btrfs mount with compression: O_DIRECT reads compressed extents through the page cache"};
      }
    }
  }
  ::close(fd);
  return out;
#endif
}

#endif

#ifndef _WIN32

FileReader::FileReader(const std::filesystem::path& path, bool direct) : path_(path) {
  // util::open_regular_file: a FIFO or a device is refused instead of blocking the open (or the reads) forever.
  util::OpenStatus status = util::OpenStatus::kError;
#ifdef O_DIRECT
  if (direct) {
    status = util::open_regular_file(path, O_DIRECT, fd_);
    direct_ = status == util::OpenStatus::kOk;
  }
#else
  static_cast<void>(direct);
#endif
  if (status == util::OpenStatus::kError) status = util::open_regular_file(path, 0, fd_);
  if (status == util::OpenStatus::kNotRegular) throw IoError(util::path_utf8(path_) + ": " + util::kNotRegularFile);
  if (status != util::OpenStatus::kOk) fail(path_, "cannot open", errno);
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
  if (memory_) return copy_from_memory(offset, dst, size);
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

FileReader::FileReader(const std::filesystem::path& path, std::shared_ptr<const std::byte> memory, std::uint64_t size,
                       bool pinned)
    : path_(path), size_(size), memory_(std::move(memory)), pinned_(pinned) {
  if (!memory_ && size_ > 0) throw IoError(util::path_utf8(path_) + ": in-memory reader without memory");
}

std::size_t FileReader::copy_from_memory(std::uint64_t offset, void* dst, std::size_t size) const {
  if (offset >= size_) return 0;
  const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(size, size_ - offset));
  std::memcpy(dst, memory_.get() + offset, n);
  return n;
}

FileReader::~FileReader() { close(); }

FileReader::FileReader(FileReader&& other) noexcept
    : path_(std::move(other.path_)),
      size_(other.size_),
      direct_(other.direct_),
      memory_(std::move(other.memory_)),
      pinned_(other.pinned_) {
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
    memory_ = std::move(other.memory_);
    pinned_ = other.pinned_;
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
  void* handle = nullptr;
  const util::OpenStatus status = util::open_regular_file(path, FILE_ATTRIBUTE_NORMAL, handle);
  if (status == util::OpenStatus::kNotRegular) throw IoError(util::path_utf8(path) + ": " + util::kNotRegularFile);
  if (status != util::OpenStatus::kOk) fail(path, "cannot open", static_cast<int>(GetLastError()));
  HANDLE f = static_cast<HANDLE>(handle);
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
  int fd = -1;
  const util::OpenStatus status = util::open_regular_file(path, 0, fd);
  if (status == util::OpenStatus::kNotRegular) throw IoError(util::path_utf8(path) + ": " + util::kNotRegularFile);
  if (status != util::OpenStatus::kOk) fail(path, "cannot open", errno);
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
