// Opening a user-supplied path for reading only when it names a regular file. open(2) on a FIFO blocks until a writer
// appears, and a character device may block or never end, so a model, LoRA or image path naming one would hang the
// caller; a directory would fail later with a less clear error. Every reader of such paths opens through here:
// weights::FileReader and MappedFile (safetensors files, the supported-model checks, images, hashes) and
// read_replaceable_file (caches, the model index). Their metadata is queried through file_info and their outputs are
// written through OutputFile, never through std::filesystem or std::ofstream.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>

namespace mi::util {

enum class OpenStatus {
  kOk,
  kNotRegular,  // a FIFO, a device, a socket or a directory: closed again before any read
  kError,       // the open (or fstat) failed: errno (POSIX) / GetLastError() (Windows) holds the cause
  kLeased,      // POSIX: another process holds a lease on the file and it was not released within about a second
};

// The messages for kNotRegular and kLeased, after the path: "<path>: not a regular file".
inline constexpr const char* kNotRegularFile = "not a regular file";
inline constexpr const char* kLeasedFile = "file is leased by another process (EWOULDBLOCK)";

// Whether a Windows path's text names a device or a pipe rather than a file. Pure (no system call), so it is built
// and tested on every platform; open_regular_file applies it on Windows, before any open, to the path and to its
// GetFullPathNameW form. Separators are `\` or `/`, names compare case-insensitively:
//   - the device and NT namespaces (\\?\, \\.\, \??\): only a drive path (X:\...) or
//     UNC\server\share\... names a file; everything else is refused (\\?\pipe\x, \\.\COM1, \\?\PhysicalDrive0,
//     \\?\GLOBALROOT\..., \\.\C: the raw volume, \\?\Volume{...});
//   - UNC paths (\\server\share\..., or UNC\ under a namespace prefix) whose share is `pipe`, `mailslot` or `IPC$`
//     (trailing dots and spaces ignored, as Win32 path normalization drops them), on any server; other UNC shares
//     (models on a NAS, `pipe$`) are files.
// DOS device names in any folder (NUL, CONIN$, C:\m\COM1, COM1.txt) are not visible in the text: GetFullPathNameW
// maps them to \\.\<name>, which this refuses.
bool windows_path_names_device(std::wstring_view path);

// The boundary check of a user-supplied path, called once where it enters (C API and server fields, CLI arguments, the
// converter's input and output, the pipeline's options and request, image and PNG paths) before anything touches it.
// It classifies the text only; what a reparse point inside the path leads to is checked by the handle that opens it
// (file_info, open_regular_file, OutputFile). On Windows it throws
// std::invalid_argument "<what>: not a regular file: <path>" when is_windows_device_path says so; elsewhere it does
// nothing (POSIX device files are refused where they are opened, open_regular_file).
void check_user_path(const std::filesystem::path& path, std::string_view what);

// What a path names, from one query that follows links.
enum class FileKind {
  kMissing,  // no such file, or the path cannot be queried (std::filesystem::status's not_found / none)
  kRegular,
  kDirectory,
  kOther,  // a pipe, a device, a socket: never read or written as a file
};

// A path's kind and, for a file or a directory, its size, last write time and identity. Equal infos: the same file with
// the same size and time, the stamp of state built from a file's contents (a file replaced under the same path, or
// written in place, gets another).
struct FileInfo {
  FileKind kind = FileKind::kMissing;
  std::uint64_t size = 0;
  std::int64_t mtime_ns = 0;  // since the system_clock (Unix) epoch: the value .mossidx files persist
  std::uint64_t volume = 0;   // st_dev; the volume serial number
  std::uint64_t id = 0;       // st_ino; the low half of the 128-bit file id (FileIdInfo: ReFS and Dev Drive ids
  std::uint64_t id_high = 0;  // are not unique in 64 bits); its high half (0 on POSIX)
  bool exists() const { return kind != FileKind::kMissing; }
  bool regular() const { return kind == FileKind::kRegular; }
  bool directory() const { return kind == FileKind::kDirectory; }
  bool operator==(const FileInfo&) const = default;
};

// Every metadata query of a user-supplied path (and of the files inside a user folder) goes through here, never
// through std::filesystem: MSVC's status, file_size, last_write_time, equivalent and canonical open the path without
// SECURITY_SQOS_PRESENT, and check_user_path classifies the text only, so a junction or symlink inside a folder the
// engine reads could still redirect an ordinary-looking path to a named pipe whose server impersonates the caller.
//   POSIX: stat(2), what std::filesystem::status reads.
//   Windows: kOther without opening anything for is_windows_device_path; else one CreateFileW(FILE_READ_ATTRIBUTES,
//   share-all, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION), which
//   follows the reparse points and reads no data, and its handle answers: GetFileType (not FILE_TYPE_DISK: kOther),
//   then GetFileInformationByHandle (FILE_ATTRIBUTE_DEVICE: kOther; FILE_ATTRIBUTE_DIRECTORY: kDirectory; the size,
//   the last write time) and GetFileInformationByHandleEx(FileIdInfo) (the 64-bit volume serial number and the 128-bit
//   file id; the 32-bit serial and 64-bit file index where a file system has no FileIdInfo). An open that fails with ERROR_PIPE_BUSY is a
//   pipe (kOther).
FileInfo file_info(const std::filesystem::path& path);

// Whether two infos are of one existing file (std::filesystem::equivalent).
inline bool same_file(const FileInfo& a, const FileInfo& b) {
  return a.exists() && b.exists() && a.volume == b.volume && a.id == b.id && a.id_high == b.id_high;
}

// The path with its links resolved, for comparing output and input paths: std::filesystem::weakly_canonical on POSIX.
// Windows: the final path (GetFinalPathNameByHandleW) of the longest existing prefix, opened as file_info opens it,
// with the rest appended, lexically normalized; the absolute path (path_key) when no prefix opens as a file or a
// directory.
std::filesystem::path resolved_path(const std::filesystem::path& path);

// A path's identity for cache keys, without opening it on Windows: std::filesystem::weakly_canonical on POSIX (as
// before); on Windows std::filesystem::absolute (GetFullPathNameW, which opens nothing), lexically normalized.
std::filesystem::path path_key(const std::filesystem::path& path);

#ifdef _WIN32
// SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, carried by every CreateFileW of a user path: a pipe reached
// through a reparse point can identify the engine's account but never act as it. Identification, not Anonymous: an
// SMB server receives the level with every file open on a share (model folders on a NAS are supported) and an
// anonymous one may be refused there; what Identification discloses is the account's name and groups.
inline constexpr unsigned long kUserPathSqos = 0x00110000;

// windows_path_names_device of the path or of its GetFullPathNameW form (true too when there is no full path): such a
// path is never handed to CreateFileW. (A pipe opened without SECURITY_SQOS_PRESENT would let its server impersonate
// the caller; a device's open runs its driver.) Every CreateFileW of a user path checks it first.
bool is_windows_device_path(const std::filesystem::path& path);

// kNotRegular without opening anything for is_windows_device_path. Then a probe: CreateFileW(FILE_READ_ATTRIBUTES,
// share-all, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION), which reads
// no data and runs no device read; its handle must be a regular file by file_info's checks (FILE_TYPE_DISK, neither
// FILE_ATTRIBUTE_DIRECTORY nor FILE_ATTRIBUTE_DEVICE). Then CreateFileW(GENERIC_READ, share-all, OPEN_EXISTING, `flags`
// | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION), whose handle is checked the same way (the path may have been
// replaced since the probe). Every kernel open of the path carries SECURITY_SQOS_PRESENT. `handle` receives the HANDLE
// on kOk.
OpenStatus open_regular_file(const std::filesystem::path& path, unsigned long flags, void*& handle);
#else
// stat(2) first: kNotRegular without opening anything that is not a regular file (the open of a device node can act on
// the device). Then open(2) with O_RDONLY | O_CLOEXEC | O_NOCTTY | O_NONBLOCK | `flags` (e.g. O_DIRECT), so a FIFO
// swapped in since the stat cannot block the open; an EWOULDBLOCK / EAGAIN (a lease being broken) on a path that is
// still a regular file is retried, still non-blocking, 5 times over about a second (a lease break may take the
// 45 s of /proc/sys/fs/lease-break-time), then kLeased with errno EWOULDBLOCK. fstat then refuses anything but a
// regular file (the path may have been replaced meanwhile), and O_NONBLOCK is cleared again for the reads. `fd`
// receives the descriptor on kOk.
OpenStatus open_regular_file(const std::filesystem::path& path, int flags, int& fd);
#endif

// A file the engine writes under a user-supplied path, through its own descriptor (handle) from open to close:
//   - kReplace: an output the caller names (an image, a report), created or truncated. POSIX: stat(2) first (an
//     existing device or FIFO is never opened), then open(O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOCTTY |
//     O_NONBLOCK) (a FIFO swapped in since cannot block the open). Windows: refused without opening anything for
//     is_windows_device_path; else CreateFileW(GENERIC_WRITE | FILE_READ_ATTRIBUTES, FILE_SHARE_READ, CREATE_ALWAYS,
//     FILE_ATTRIBUTE_NORMAL | kUserPathSqos).
//   - kTemporary: a new file next to `path` (a cache entry's, the hardware profile's, the model index's, a PNG's or
//     the converter's temporary file, renamed over its target once complete), named "<path>.tmp-<16 hex digits>" from
//     std::random_device, created exclusively and never through a link: POSIX O_CREAT | O_EXCL | O_NOFOLLOW, Windows
//     CREATE_NEW | FILE_FLAG_OPEN_REPARSE_POINT; a name that exists (planted or another writer's) is never opened,
//     another is drawn (16 tries). path() is the name created.
// Either way the open file must be a regular file (fstat; file_info's handle checks) before a byte is written: a
// reparse point redirecting the path to a pipe is refused, and the pipe's server could not impersonate the open. Every
// call returns false once any call failed (is_open() false too when the open failed or was refused).
class OutputFile {
 public:
  enum class Create { kReplace, kTemporary };
  explicit OutputFile(const std::filesystem::path& path, Create create = Create::kReplace);
  ~OutputFile();
  OutputFile(const OutputFile&) = delete;
  OutputFile& operator=(const OutputFile&) = delete;

  const std::filesystem::path& path() const { return path_; }
  bool is_open() const { return ok_; }
  bool write(const void* data, std::size_t size);
  bool write(std::string_view text) { return write(text.data(), text.size()); }
  bool seek(std::uint64_t offset);  // the position of the next write (past the end: the gap reads as zeros)
  bool sync();                      // the data to stable storage (fsync; FlushFileBuffers) through the same descriptor
  bool close();

 private:
  std::filesystem::path path_;
#ifdef _WIN32
  void* handle_ = nullptr;
#else
  int fd_ = -1;
#endif
  bool ok_ = false;
};

}  // namespace mi::util
