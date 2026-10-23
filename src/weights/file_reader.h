// Positional file reads (pread / overlapped ReadFile) with optional unbuffered "direct" I/O, and a
// read-only memory map. Direct I/O bypasses the page cache so streaming multi-GB weight groups
// into pinned ring buffers does not evict everything else; mmap serves resident loads.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace mi::weights {

struct IoError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Offset, length and buffer address must all be multiples of this for direct I/O (covers 4K-sector
// devices; 512e devices accept it too).
inline constexpr std::size_t kDirectAlign = 4096;

// Heap block aligned to kDirectAlign; usable as a direct-I/O target.
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t size);
  AlignedBuffer(AlignedBuffer&&) noexcept = default;
  AlignedBuffer& operator=(AlignedBuffer&&) noexcept = default;

  std::byte* data() { return p_.get(); }
  const std::byte* data() const { return p_.get(); }
  std::size_t size() const { return size_; }

 private:
  struct Deleter {
    void operator()(std::byte* p) const { ::operator delete(p, std::align_val_t{kDirectAlign}); }
  };
  std::unique_ptr<std::byte, Deleter> p_;
  std::size_t size_ = 0;
};

// Byte range [begin, end) of a file widened to kDirectAlign multiples; `skip` is where the wanted
// bytes start inside the widened range.
struct AlignedRange {
  std::uint64_t offset;
  std::size_t length;
  std::size_t skip;
};
AlignedRange align_range(std::uint64_t offset, std::size_t size);

// Decides what a direct-I/O read loop does after a read returned fewer bytes than asked. `pos` is
// the file position the next read would start at and `remaining` the bytes still wanted. A short
// read at or past `file_size` is EOF (stop). Mid-file (a network or FUSE file system returning a
// partial read) the loop may continue only if the retry is itself aligned; otherwise it stops and
// the caller reports the short read.
bool direct_short_read_continues(std::uint64_t pos, std::size_t remaining, std::uint64_t file_size);

// Thread-safe for concurrent read_at calls (no shared file position).
class FileReader {
 public:
  // `direct` requests unbuffered I/O. If the file system refuses it (tmpfs, some network mounts)
  // the file is opened buffered and direct() reports false, so callers keep working.
  explicit FileReader(const std::filesystem::path& path, bool direct = false);
  // An in-memory reader: `size` bytes at `memory` hold the file's contents (`path` names it in messages). Reads copy
  // out of them (never direct); `pinned` says they are page-locked, so the device can DMA from them (memory()).
  FileReader(const std::filesystem::path& path, std::shared_ptr<const std::byte> memory, std::uint64_t size, bool pinned);
  ~FileReader();
  FileReader(FileReader&& other) noexcept;
  FileReader& operator=(FileReader&& other) noexcept;
  FileReader(const FileReader&) = delete;
  FileReader& operator=(const FileReader&) = delete;

  const std::filesystem::path& path() const { return path_; }
  std::uint64_t size() const { return size_; }
  bool direct() const { return direct_; }
  // The contents of an in-memory reader (null for a file), and whether they are page-locked.
  const std::byte* memory() const { return memory_.get(); }
  bool pinned() const { return pinned_; }

  // Reads exactly `size` bytes at `offset` into dst. Any alignment is accepted: in direct mode an
  // unaligned request is served through an internal aligned over-read. Throws IoError on a short read.
  void read_at(std::uint64_t offset, void* dst, std::size_t size) const;

  // Zero-copy-friendly direct path: reads align_range(offset, size) into `buf` (which must be
  // kDirectAlign-aligned and at least range.length bytes) and returns the range; the wanted bytes
  // are at buf + range.skip. Works in buffered mode too.
  AlignedRange read_aligned(std::uint64_t offset, std::size_t size, std::byte* buf, std::size_t buf_size) const;

 private:
  // Raw loop; requires alignment in direct mode. Returns bytes actually read (short only at EOF).
  std::size_t pread_raw(std::uint64_t offset, void* dst, std::size_t size) const;
  void close() noexcept;

  std::size_t copy_from_memory(std::uint64_t offset, void* dst, std::size_t size) const;

  std::filesystem::path path_;
  std::uint64_t size_ = 0;
  bool direct_ = false;
  std::shared_ptr<const std::byte> memory_;
  bool pinned_ = false;
#ifdef _WIN32
  void* handle_ = nullptr;
#else
  int fd_ = -1;
#endif
};

// Whether reading `path` with direct I/O bypasses the page cache, and why not. Linux: false when O_DIRECT is refused
// (tmpfs and ramfs, some FUSE and network mounts: the open fails), or when the kernel would serve it buffered anyway:
// on btrfs, compressed extents are read through the page cache whatever the flag says, so a file with more than
// kMaxCompressedFraction of its bytes in compressed extents (FIEMAP's "encoded" extents; when FIEMAP is unavailable:
// a compress / compress-force mount option) counts as unsupported. Windows: false when a FILE_FLAG_NO_BUFFERING open
// fails. `reason` is empty when supported, else one human-readable sentence naming the file system or the error.
struct DirectIoSupport {
  bool supported = false;
  std::string reason;
};
inline constexpr double kMaxCompressedFraction = 0.10;
// statfs(2) f_type of the RAM file systems (Linux): direct I/O gains nothing there, and the page cache is the storage.
inline constexpr std::uint32_t kTmpfsMagic = 0x01021994;
inline constexpr std::uint32_t kRamfsMagic = 0x858458f6;
DirectIoSupport direct_io_support(const std::filesystem::path& path);

// The fraction of `path`'s bytes in the page cache now, in [0, 1]; negative when unknown. Linux: cachestat(2) (kernel
// 6.5+) over the whole file; where it does not exist, mincore(2) on a read-only mapping sampled at kResidencySamples
// pages spread over the file, but only for a file the process could write (owner, or W_OK): for the others the kernel
// reports every page resident, so they are unknown (cachestat refuses them with EPERM). An EPERM on a file mincore may
// answer (a seccomp filter on cachestat) uses the sample. Windows: unknown. Cheap: no
// data is read. test_seams.h can force the mincore path or "unknown".
inline constexpr int kResidencySamples = 64;
double page_cache_fraction(const std::filesystem::path& path);

// Read-only view of a whole file. Empty files map to a null/zero-size view.
class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path);
  ~MappedFile();
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const std::byte* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  void unmap() noexcept;

  const std::byte* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace mi::weights
