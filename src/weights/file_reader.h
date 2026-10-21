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
  ~FileReader();
  FileReader(FileReader&& other) noexcept;
  FileReader& operator=(FileReader&& other) noexcept;
  FileReader(const FileReader&) = delete;
  FileReader& operator=(const FileReader&) = delete;

  const std::filesystem::path& path() const { return path_; }
  std::uint64_t size() const { return size_; }
  bool direct() const { return direct_; }

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

  std::filesystem::path path_;
  std::uint64_t size_ = 0;
  bool direct_ = false;
#ifdef _WIN32
  void* handle_ = nullptr;
#else
  int fd_ = -1;
#endif
};

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
